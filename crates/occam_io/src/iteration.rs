//! Iteration/startup file format parser and writer
//!
//! Supports OCCAMITER_FLEX format

use crate::error::{OccamIoError, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Roughness type for regularization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoughnessType {
    /// First derivative (standard smoothness)
    #[default]
    FirstDerivative = 1,
    /// Second derivative
    SecondDerivative = 2,
    /// Depth-weighted first derivative
    DepthWeighted = 4,
    /// Sum of parameters
    SumOfParams = 5,
    /// Minimum gradient support
    MinGradientSupport = 6,
    /// Two-model (for anisotropy)
    TwoModel = 7,
}

impl RoughnessType {
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            1 => Some(RoughnessType::FirstDerivative),
            2 => Some(RoughnessType::SecondDerivative),
            4 => Some(RoughnessType::DepthWeighted),
            5 => Some(RoughnessType::SumOfParams),
            6 => Some(RoughnessType::MinGradientSupport),
            7 => Some(RoughnessType::TwoModel),
            _ => None,
        }
    }

    /// Parse from string (integer code or Fortran name)
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();
        // Try integer first
        if let Ok(code) = s.parse::<i32>() {
            return Self::from_code(code);
        }
        // Try string names (Fortran convention)
        match s.as_str() {
            "firstdiff" | "first_derivative" => Some(RoughnessType::FirstDerivative),
            "seconddiff" | "second_derivative" => Some(RoughnessType::SecondDerivative),
            "depthweighted" | "depth_weighted" => Some(RoughnessType::DepthWeighted),
            "sumofparams" | "sum_of_params" => Some(RoughnessType::SumOfParams),
            "mingradient" | "min_gradient_support" => Some(RoughnessType::MinGradientSupport),
            "twomodel" | "two_model" => Some(RoughnessType::TwoModel),
            _ => None,
        }
    }

    pub fn code(&self) -> i32 {
        *self as i32
    }
}

impl std::fmt::Display for RoughnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Iteration file / startup configuration
#[derive(Debug, Clone)]
pub struct IterationFile {
    /// Description string
    pub description: String,
    /// Path to model file
    pub model_file: String,
    /// Path to data file
    pub data_file: String,
    /// Date/time string (optional)
    pub datetime: Option<String>,
    /// Maximum iterations to run
    pub max_iterations: usize,
    /// Target RMS misfit
    pub target_misfit: f64,
    /// Roughness type
    pub roughness_type: RoughnessType,
    /// Maximum step size reductions
    pub stepsize_cut_count: usize,
    /// Debug level
    pub debug_level: i32,
    /// Current iteration number
    pub iteration: usize,
    /// Log10 of Lagrange multiplier
    pub lagrange_value: f64,
    /// Current roughness
    pub roughness_value: f64,
    /// Current RMS misfit
    pub misfit_value: f64,
    /// Whether target misfit has been reached
    pub misfit_reached: bool,
    /// Delta misfit limit for convergence (Fortran: Delta Misfit Limit)
    pub delta_misfit_limit: f64,
    /// Bounds transform type ("exponential" or "bandpass")
    pub bounds_transform: Option<String>,
    /// Model bounds (lower, upper) in log10 resistivity
    pub model_bounds: Option<(f64, f64)>,
    /// Free parameter values (log10 resistivity typically)
    pub parameters: Vec<f64>,
}

impl Default for IterationFile {
    fn default() -> Self {
        Self {
            description: "Occam 1D Inversion".to_string(),
            model_file: "model".to_string(),
            data_file: "data".to_string(),
            datetime: None,
            max_iterations: 100,
            target_misfit: 1.0,
            roughness_type: RoughnessType::FirstDerivative,
            stepsize_cut_count: 8,
            debug_level: 0,
            iteration: 0,
            lagrange_value: 5.0,
            roughness_value: 0.0,
            misfit_value: 1000.0,
            misfit_reached: false,
            delta_misfit_limit: 0.0,
            bounds_transform: None,
            model_bounds: None,
            parameters: Vec::new(),
        }
    }
}

impl IterationFile {
    /// Create a new startup file with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a startup file with initial parameters
    pub fn with_params(n_params: usize, initial_value: f64) -> Self {
        let mut iter = Self::default();
        iter.parameters = vec![initial_value; n_params];
        iter
    }

    /// Read from file
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::read(reader)
    }

    /// Read from reader
    #[allow(unused_assignments)]
    pub fn read<R: BufRead>(reader: R) -> Result<Self> {
        let mut iter = IterationFile::default();
        let mut lines = reader.lines().enumerate();
        let mut line_num;

        // Helper to get next non-empty line
        let next_line = |lines: &mut std::iter::Enumerate<std::io::Lines<R>>| -> Result<Option<(usize, String)>> {
            loop {
                match lines.next() {
                    Some((num, Ok(line))) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        return Ok(Some((num + 1, line)));
                    }
                    Some((num, Err(e))) => {
                        return Err(OccamIoError::Parse {
                            line: num + 1,
                            message: e.to_string(),
                        })
                    }
                    None => return Ok(None),
                }
            }
        };

        // Read format line
        let (ln, format_line) = next_line(&mut lines)?
            .ok_or_else(|| OccamIoError::MissingField("Format".to_string()))?;
        line_num = ln;

        let format_str = parse_key_value(&format_line, "Format")
            .ok_or_else(|| OccamIoError::MissingField("Format".to_string()))?;
        if format_str != "OCCAMITER_FLEX" {
            return Err(OccamIoError::InvalidFormat {
                expected: "OCCAMITER_FLEX".to_string(),
                got: format_str,
            });
        }

        // Read remaining key-value pairs until parameters
        loop {
            let Some((ln, line)) = next_line(&mut lines)? else {
                break;
            };
            line_num = ln;

            // Check if this is the start of parameter values (no colon)
            if !line.contains(':') {
                // This is parameter data
                let parts: Vec<&str> = line.split_whitespace().collect();
                for part in parts {
                    iter.parameters.push(parse_float(part, line_num)?);
                }
                break;
            }

            // Parse key-value pairs
            if let Some(v) = parse_key_value(&line, "Description") {
                iter.description = v;
            } else if let Some(v) = parse_key_value(&line, "Model File") {
                iter.model_file = v;
            } else if let Some(v) = parse_key_value(&line, "Data File") {
                iter.data_file = v;
            } else if let Some(v) = parse_key_value(&line, "Date/Time") {
                iter.datetime = Some(v);
            } else if let Some(v) = parse_key_value(&line, "Iterations to run") {
                iter.max_iterations = v.parse().map_err(|_| OccamIoError::Parse {
                    line: line_num,
                    message: format!("Invalid iterations: {}", v),
                })?;
            } else if let Some(v) = parse_key_value(&line, "Target Misfit") {
                iter.target_misfit = parse_float(&v, line_num)?;
            } else if let Some(v) = parse_key_value(&line, "Roughness Type") {
                // Strip inline comments (! ...)
                let v = v.split('!').next().unwrap_or(&v).trim().to_string();
                iter.roughness_type = RoughnessType::from_str_flexible(&v).ok_or_else(|| {
                    OccamIoError::Parse {
                        line: line_num,
                        message: format!("Unknown roughness type: {}", v),
                    }
                })?;
            } else if let Some(v) = parse_key_value(&line, "Stepsize Cut Count") {
                iter.stepsize_cut_count = v.parse().map_err(|_| OccamIoError::Parse {
                    line: line_num,
                    message: format!("Invalid stepsize cut count: {}", v),
                })?;
            } else if let Some(v) = parse_key_value(&line, "Debug Level") {
                iter.debug_level = v.parse().map_err(|_| OccamIoError::Parse {
                    line: line_num,
                    message: format!("Invalid debug level: {}", v),
                })?;
            } else if let Some(v) = parse_key_value(&line, "Iteration") {
                iter.iteration = v.parse().map_err(|_| OccamIoError::Parse {
                    line: line_num,
                    message: format!("Invalid iteration: {}", v),
                })?;
            } else if let Some(v) = parse_key_value(&line, "Lagrange Value") {
                iter.lagrange_value = parse_float(&v, line_num)?;
            } else if let Some(v) = parse_key_value(&line, "Roughness Value") {
                iter.roughness_value = parse_float(&v, line_num)?;
            } else if let Some(v) = parse_key_value(&line, "Misfit Value") {
                iter.misfit_value = parse_float(&v, line_num)?;
            } else if let Some(v) = parse_key_value(&line, "Misfit Reached") {
                iter.misfit_reached = v.trim() == "1";
            } else if let Some(v) = parse_key_value(&line, "Delta Misfit Limit") {
                iter.delta_misfit_limit = parse_float(&v, line_num)?;
            } else if let Some(v) = parse_key_value(&line, "Bounds Transform") {
                // Strip inline comments (! ...)
                let v = v.split('!').next().unwrap_or(&v).trim().to_lowercase();
                iter.bounds_transform = Some(v);
            } else if let Some(v) = parse_key_value(&line, "Model Bounds") {
                // Strip inline comments
                let v = v.split('!').next().unwrap_or(&v).trim().to_string();
                let parts: Vec<&str> = v.split(',').collect();
                if parts.len() == 2 {
                    let lo = parse_float(parts[0].trim(), line_num)?;
                    let hi = parse_float(parts[1].trim(), line_num)?;
                    iter.model_bounds = Some((lo, hi));
                } else {
                    return Err(OccamIoError::Parse {
                        line: line_num,
                        message: format!("Model Bounds must be 'lower,upper', got: {}", v),
                    });
                }
            } else if let Some(_v) = parse_key_value(&line, "Model Limits") {
                // Alias for Model Bounds — strip inline comments
                let v = _v.split('!').next().unwrap_or(&_v).trim().to_string();
                let parts: Vec<&str> = v.split(',').collect();
                if parts.len() == 2 {
                    let lo = parse_float(parts[0].trim(), line_num)?;
                    let hi = parse_float(parts[1].trim(), line_num)?;
                    iter.model_bounds = Some((lo, hi));
                } else {
                    return Err(OccamIoError::Parse {
                        line: line_num,
                        message: format!("Model Limits must be 'lower,upper', got: {}", v),
                    });
                }
            } else if let Some(v) = parse_key_value(&line, "Param Count") {
                let _n_params: usize = v.parse().map_err(|_| OccamIoError::Parse {
                    line: line_num,
                    message: format!("Invalid param count: {}", v),
                })?;
                // Don't need to store this, we'll count from actual params
            }
        }

        // Continue reading parameter values
        while let Some((ln, line)) = next_line(&mut lines)? {
            line_num = ln;
            let parts: Vec<&str> = line.split_whitespace().collect();
            for part in parts {
                iter.parameters.push(parse_float(part, line_num)?);
            }
        }

        Ok(iter)
    }

    /// Write to file
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write(writer)
    }

    /// Write to writer
    pub fn write<W: Write>(&self, mut writer: W) -> Result<()> {
        writeln!(writer, "Format:             OCCAMITER_FLEX")?;
        writeln!(writer, "Description:        {}", self.description)?;
        writeln!(writer, "Model File:         {}", self.model_file)?;
        writeln!(writer, "Data File:          {}", self.data_file)?;

        if let Some(ref dt) = self.datetime {
            writeln!(writer, "Date/Time:          {}", dt)?;
        }

        writeln!(writer, "Iterations to run:  {}", self.max_iterations)?;
        writeln!(writer, "Target Misfit:      {:.6}", self.target_misfit)?;
        writeln!(writer, "Roughness Type:     {}", self.roughness_type)?;
        writeln!(writer, "Stepsize Cut Count: {}", self.stepsize_cut_count)?;

        if let Some(ref bt) = self.bounds_transform {
            writeln!(writer, "Bounds Transform:   {}", bt)?;
        }
        if let Some((lo, hi)) = self.model_bounds {
            writeln!(writer, "Model Bounds:       {},{}", lo, hi)?;
        }
        writeln!(
            writer,
            "Delta Misfit Limit: {:.16}",
            self.delta_misfit_limit
        )?;
        writeln!(writer, "Debug Level:        {}", self.debug_level)?;
        writeln!(writer, "Iteration:          {}", self.iteration)?;
        writeln!(writer, "Lagrange Value:     {:.6e}", self.lagrange_value)?;
        writeln!(writer, "Roughness Value:    {:.6e}", self.roughness_value)?;
        writeln!(writer, "Misfit Value:       {:.6e}", self.misfit_value)?;
        writeln!(
            writer,
            "Misfit Reached:     {}",
            if self.misfit_reached { 1 } else { 0 }
        )?;
        writeln!(writer, "Param Count:        {}", self.parameters.len())?;

        // Write parameters, 4 per line
        for (i, param) in self.parameters.iter().enumerate() {
            if i % 4 == 0 && i > 0 {
                writeln!(writer)?;
            }
            write!(writer, "{:14.6e}", param)?;
        }
        writeln!(writer)?;

        Ok(())
    }

    /// Get number of parameters
    pub fn n_params(&self) -> usize {
        self.parameters.len()
    }

    /// Update iteration status
    pub fn update_status(
        &mut self,
        iteration: usize,
        lagrange: f64,
        roughness: f64,
        misfit: f64,
        params: &[f64],
    ) {
        self.iteration = iteration;
        self.lagrange_value = lagrange;
        self.roughness_value = roughness;
        self.misfit_value = misfit;
        self.misfit_reached = misfit <= self.target_misfit;
        self.parameters = params.to_vec();
    }
}

/// Parse a key: value line
fn parse_key_value(line: &str, key: &str) -> Option<String> {
    let line = line.trim();
    let line_lower = line.to_lowercase();
    let key_lower = key.to_lowercase();

    if line_lower.starts_with(&key_lower) {
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() == 2 {
            return Some(parts[1].trim().to_string());
        }
    }
    None
}

/// Parse a float value with error context
fn parse_float(s: &str, line: usize) -> Result<f64> {
    // Handle Fortran-style exponents (1d12 -> 1e12)
    let s = s.replace('d', "e").replace('D', "E");
    s.trim().parse().map_err(|_| OccamIoError::Parse {
        line,
        message: format!("Invalid float: {}", s),
    })
}
