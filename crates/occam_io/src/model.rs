//! Resistivity model file format parser and writer
//!
//! Supports Resistivity1DMod_1.0 format

use crate::error::{OccamIoError, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// A single layer in the 1D model
#[derive(Debug, Clone)]
pub struct Layer {
    /// Depth to top of layer (m)
    pub top_depth: f64,
    /// Resistivity (ohm-m), or None if free parameter
    pub resistivity: Option<f64>,
    /// Roughness penalty weight
    pub penalty: f64,
    /// Preferred model value (optional)
    pub preference: f64,
    /// Weight for preference term
    pub pref_penalty: f64,
}

impl Layer {
    /// Create a fixed layer with given resistivity
    pub fn fixed(top_depth: f64, resistivity: f64) -> Self {
        Self {
            top_depth,
            resistivity: Some(resistivity),
            penalty: 0.0,
            preference: 0.0,
            pref_penalty: 0.0,
        }
    }

    /// Create a free parameter layer
    pub fn free(top_depth: f64) -> Self {
        Self {
            top_depth,
            resistivity: None,
            penalty: 1.0,
            preference: 0.0,
            pref_penalty: 0.0,
        }
    }

    /// Check if this layer is a free parameter
    pub fn is_free(&self) -> bool {
        self.resistivity.is_none()
    }
}

/// 1D layered resistivity model
#[derive(Debug, Clone)]
pub struct LayeredModel {
    /// Layers from top (air) to bottom
    pub layers: Vec<Layer>,
}

impl Default for LayeredModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LayeredModel {
    /// Create an empty model
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Create a simple model with air, seawater, and sediment layers
    ///
    /// # Arguments
    /// * `water_depth` - Depth of water column (m)
    /// * `water_resistivity` - Resistivity of seawater (ohm-m), typically 0.3
    /// * `layer_depths` - Depths of free parameter layers below seafloor
    /// * `initial_resistivity` - Initial resistivity for free layers (ohm-m)
    pub fn create_marine(
        water_depth: f64,
        water_resistivity: f64,
        layer_depths: &[f64],
        initial_resistivity: f64,
    ) -> Self {
        let mut model = Self::new();

        // Air layer (always first, depth ignored)
        model.layers.push(Layer::fixed(-100000.0, 1e12));

        // Water layer
        model.layers.push(Layer::fixed(0.0, water_resistivity));

        // Free parameter layers below seafloor
        for &depth in layer_depths {
            // Create free layers (resistivity = None means free parameter)
            let layer = Layer::free(water_depth + depth);
            model.layers.push(layer);
        }

        model
    }

    /// Read model from file
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::read(reader)
    }

    /// Read model from reader
    pub fn read<R: BufRead>(reader: R) -> Result<Self> {
        let mut model = LayeredModel::new();
        let mut lines = reader.lines().enumerate();
        let mut line_num;

        // Helper to get next non-comment line
        let mut next_line = |lines: &mut std::iter::Enumerate<std::io::Lines<R>>| -> Result<Option<(usize, String)>> {
            loop {
                match lines.next() {
                    Some((num, Ok(line))) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        // Skip comment lines (lines starting with !)
                        if trimmed.starts_with('!') {
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
        if format_str != "Resistivity1DMod_1.0" {
            return Err(OccamIoError::InvalidFormat {
                expected: "Resistivity1DMod_1.0".to_string(),
                got: format_str,
            });
        }

        // Read layer count
        let (ln, layers_line) = next_line(&mut lines)?
            .ok_or_else(|| OccamIoError::MissingField("#LAYERS".to_string()))?;
        line_num = ln;

        let n_layers = parse_key_value(&layers_line, "#LAYERS")
            .ok_or_else(|| OccamIoError::MissingField("#LAYERS".to_string()))?;
        let n_layers: usize = n_layers.parse().map_err(|_| OccamIoError::Parse {
            line: line_num,
            message: format!("Invalid layer count: {}", n_layers),
        })?;

        // Read layers
        for _ in 0..n_layers {
            let Some((ln, line)) = next_line(&mut lines)? else {
                return Err(OccamIoError::Parse {
                    line: line_num,
                    message: "Unexpected end of file while reading layers".to_string(),
                });
            };
            line_num = ln;

            // Remove inline comments
            let line = line.split('!').next().unwrap_or(&line);
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < 5 {
                return Err(OccamIoError::Parse {
                    line: line_num,
                    message: "Layer requires 5 values (depth, resistivity, penalty, preference, pref_penalty)".to_string(),
                });
            }

            let top_depth = parse_float(parts[0], line_num)?;
            // "?" or negative value means free parameter
            let resistivity = if parts[1].trim() == "?" {
                None
            } else {
                let resistivity_raw = parse_float(parts[1], line_num)?;
                if resistivity_raw < 0.0 { None } else { Some(resistivity_raw) }
            };
            let penalty = parse_float(parts[2], line_num)?;
            let preference = parse_float(parts[3], line_num)?;
            let pref_penalty = parse_float(parts[4], line_num)?;

            model.layers.push(Layer {
                top_depth,
                resistivity,
                penalty,
                preference,
                pref_penalty,
            });
        }

        Ok(model)
    }

    /// Write model to file
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write(writer)
    }

    /// Write model to writer
    pub fn write<W: Write>(&self, mut writer: W) -> Result<()> {
        writeln!(writer, "Format: Resistivity1DMod_1.0")?;
        writeln!(writer, "#LAYERS: {}", self.layers.len())?;
        writeln!(writer, "! Layer block in file is:")?;
        writeln!(
            writer,
            "! [top_depth    resistivity  penalty    preference   pref_penalty]"
        )?;
        writeln!(
            writer,
            "! note that top_depth and penalty of 1st layer ignored"
        )?;

        for layer in &self.layers {
            let res_str = match layer.resistivity {
                Some(r) => format!("{:.6e}", r),
                None => "-1".to_string(),
            };
            writeln!(
                writer,
                "{:12.2}  {:>12}  {:8.2}  {:8.2}  {:8.2}",
                layer.top_depth, res_str, layer.penalty, layer.preference, layer.pref_penalty
            )?;
        }

        Ok(())
    }

    /// Get the number of layers
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the number of free parameters
    pub fn n_free_params(&self) -> usize {
        self.layers.iter().filter(|l| l.is_free()).count()
    }

    /// Get indices of free parameter layers
    pub fn free_param_indices(&self) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_free())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get layer thicknesses
    pub fn thicknesses(&self) -> Vec<f64> {
        let mut thicknesses = Vec::with_capacity(self.layers.len());
        for i in 0..self.layers.len() {
            if i + 1 < self.layers.len() {
                thicknesses.push(self.layers[i + 1].top_depth - self.layers[i].top_depth);
            } else {
                // Last layer is infinite halfspace
                thicknesses.push(f64::INFINITY);
            }
        }
        thicknesses
    }

    /// Get resistivities as a vector (using given values for free parameters)
    pub fn resistivities(&self, free_params: &[f64]) -> Vec<f64> {
        let mut res = Vec::with_capacity(self.layers.len());
        let mut free_idx = 0;

        for layer in &self.layers {
            if layer.is_free() {
                res.push(free_params[free_idx]);
                free_idx += 1;
            } else {
                res.push(layer.resistivity.unwrap());
            }
        }
        res
    }

    /// Get conductivities as a vector (using given values for free parameters)
    pub fn conductivities(&self, free_params: &[f64]) -> Vec<f64> {
        self.resistivities(free_params)
            .into_iter()
            .map(|r| 1.0 / r)
            .collect()
    }

    /// Get layer depths (top of each layer)
    pub fn depths(&self) -> Vec<f64> {
        self.layers.iter().map(|l| l.top_depth).collect()
    }

    /// Get penalties for roughness calculation
    pub fn penalties(&self) -> Vec<f64> {
        self.layers.iter().map(|l| l.penalty).collect()
    }

    /// Get preferences for prior model
    pub fn preferences(&self) -> Vec<f64> {
        self.layers.iter().map(|l| l.preference).collect()
    }

    /// Get preference penalties
    pub fn pref_penalties(&self) -> Vec<f64> {
        self.layers.iter().map(|l| l.pref_penalty).collect()
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
