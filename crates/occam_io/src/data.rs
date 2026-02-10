//! EMData file format parser and writer
//!
//! Supports EMData_1.0, EMData_1.1, and EMData_1.2 formats

use crate::error::{OccamIoError, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

/// Phase convention for EM data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhaseConvention {
    /// exp(-i*omega*t) convention
    #[default]
    Lag,
    /// exp(+i*omega*t) convention
    Lead,
}

impl FromStr for PhaseConvention {
    type Err = OccamIoError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().trim() {
            "lag" => Ok(PhaseConvention::Lag),
            "lead" => Ok(PhaseConvention::Lead),
            _ => Err(OccamIoError::Parse {
                line: 0,
                message: format!("Invalid phase convention: {}", s),
            }),
        }
    }
}

impl std::fmt::Display for PhaseConvention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhaseConvention::Lag => write!(f, "lag"),
            PhaseConvention::Lead => write!(f, "lead"),
        }
    }
}

/// CSEM/MT data type codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    // CSEM Electric fields (Real/Imaginary)
    ExReal = 1,
    ExImag = 2,
    EyReal = 3,
    EyImag = 4,
    EzReal = 5,
    EzImag = 6,
    // CSEM Magnetic fields (Real/Imaginary)
    BxReal = 11,
    BxImag = 12,
    ByReal = 13,
    ByImag = 14,
    BzReal = 15,
    BzImag = 16,
    // CSEM Electric fields (Amplitude/Phase)
    ExAmp = 21,
    ExPhase = 22,
    EyAmp = 23,
    EyPhase = 24,
    EzAmp = 25,
    EzPhase = 26,
    // CSEM Magnetic fields (Amplitude/Phase)
    BxAmp = 31,
    BxPhase = 32,
    ByAmp = 33,
    ByPhase = 34,
    BzAmp = 35,
    BzPhase = 36,
    // CSEM Polarization ellipse parameters
    PEmax = 41,
    PEmin = 42,
    PBmax = 43,
    PBmin = 44,
    // MT Apparent resistivity and phase
    RhoXX = 101,
    PhiXX = 102,
    RhoXY = 103,
    PhiXY = 104,
    RhoYX = 105,
    PhiYX = 106,
    RhoYY = 107,
    PhiYY = 108,
    // MT Impedance (Real/Imaginary)
    ZxxReal = 111,
    ZxxImag = 112,
    ZxyReal = 113,
    ZxyImag = 114,
    ZyxReal = 115,
    ZyxImag = 116,
    ZyyReal = 117,
    ZyyImag = 118,
}

impl DataType {
    pub fn from_code(code: i32) -> Result<Self> {
        match code {
            1 => Ok(DataType::ExReal),
            2 => Ok(DataType::ExImag),
            3 => Ok(DataType::EyReal),
            4 => Ok(DataType::EyImag),
            5 => Ok(DataType::EzReal),
            6 => Ok(DataType::EzImag),
            11 => Ok(DataType::BxReal),
            12 => Ok(DataType::BxImag),
            13 => Ok(DataType::ByReal),
            14 => Ok(DataType::ByImag),
            15 => Ok(DataType::BzReal),
            16 => Ok(DataType::BzImag),
            21 => Ok(DataType::ExAmp),
            22 => Ok(DataType::ExPhase),
            23 => Ok(DataType::EyAmp),
            24 => Ok(DataType::EyPhase),
            25 => Ok(DataType::EzAmp),
            26 => Ok(DataType::EzPhase),
            31 => Ok(DataType::BxAmp),
            32 => Ok(DataType::BxPhase),
            33 => Ok(DataType::ByAmp),
            34 => Ok(DataType::ByPhase),
            35 => Ok(DataType::BzAmp),
            36 => Ok(DataType::BzPhase),
            41 => Ok(DataType::PEmax),
            42 => Ok(DataType::PEmin),
            43 => Ok(DataType::PBmax),
            44 => Ok(DataType::PBmin),
            101 => Ok(DataType::RhoXX),
            102 => Ok(DataType::PhiXX),
            103 => Ok(DataType::RhoXY),
            104 => Ok(DataType::PhiXY),
            105 => Ok(DataType::RhoYX),
            106 => Ok(DataType::PhiYX),
            107 => Ok(DataType::RhoYY),
            108 => Ok(DataType::PhiYY),
            111 => Ok(DataType::ZxxReal),
            112 => Ok(DataType::ZxxImag),
            113 => Ok(DataType::ZxyReal),
            114 => Ok(DataType::ZxyImag),
            115 => Ok(DataType::ZyxReal),
            116 => Ok(DataType::ZyxImag),
            117 => Ok(DataType::ZyyReal),
            118 => Ok(DataType::ZyyImag),
            _ => Err(OccamIoError::InvalidDataType(code)),
        }
    }

    pub fn code(&self) -> i32 {
        *self as i32
    }

    /// Check if this is a CSEM data type
    pub fn is_csem(&self) -> bool {
        (*self as i32) < 100
    }

    /// Check if this is an MT data type
    pub fn is_mt(&self) -> bool {
        (*self as i32) >= 100
    }

    /// Check if this is an electric field component
    pub fn is_electric(&self) -> bool {
        matches!(
            self,
            DataType::ExReal
                | DataType::ExImag
                | DataType::EyReal
                | DataType::EyImag
                | DataType::EzReal
                | DataType::EzImag
                | DataType::ExAmp
                | DataType::ExPhase
                | DataType::EyAmp
                | DataType::EyPhase
                | DataType::EzAmp
                | DataType::EzPhase
                | DataType::PEmax
                | DataType::PEmin
        )
    }

    /// Check if this is a magnetic field component
    pub fn is_magnetic(&self) -> bool {
        matches!(
            self,
            DataType::BxReal
                | DataType::BxImag
                | DataType::ByReal
                | DataType::ByImag
                | DataType::BzReal
                | DataType::BzImag
                | DataType::BxAmp
                | DataType::BxPhase
                | DataType::ByAmp
                | DataType::ByPhase
                | DataType::BzAmp
                | DataType::BzPhase
                | DataType::PBmax
                | DataType::PBmin
        )
    }
}

/// Transmitter definition
#[derive(Debug, Clone, Default)]
pub struct Transmitter {
    /// X coordinate (m)
    pub x: f64,
    /// Y coordinate (m)
    pub y: f64,
    /// Z coordinate (m), positive downward
    pub z: f64,
    /// Azimuth angle (degrees from x-axis)
    pub azimuth: f64,
    /// Dip angle (degrees, positive downward)
    pub dip: f64,
}

/// Receiver definition
#[derive(Debug, Clone, Default)]
pub struct Receiver {
    /// X coordinate (m)
    pub x: f64,
    /// Y coordinate (m)
    pub y: f64,
    /// Z coordinate (m), positive downward
    pub z: f64,
    /// Theta rotation angle (degrees) - or None if to be solved
    pub theta: Option<f64>,
    /// Alpha rotation angle (degrees) - or None if to be solved
    pub alpha: Option<f64>,
    /// Beta rotation angle (degrees) - or None if to be solved
    pub beta: Option<f64>,
}

/// A single data point observation
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Data type code
    pub data_type: DataType,
    /// Frequency index (0-based)
    pub freq_idx: usize,
    /// Transmitter index (0-based)
    pub tx_idx: usize,
    /// Receiver index (0-based)
    pub rx_idx: usize,
    /// Observed value
    pub value: f64,
    /// Standard error
    pub std_error: f64,
}

/// EMData file format version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EMDataFormat {
    V1_0,
    V1_1,
    V1_2,
}

impl FromStr for EMDataFormat {
    type Err = OccamIoError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim() {
            "EMData_1.0" => Ok(EMDataFormat::V1_0),
            "EMData_1.1" => Ok(EMDataFormat::V1_1),
            "EMData_1.2" => Ok(EMDataFormat::V1_2),
            _ => Err(OccamIoError::InvalidFormat {
                expected: "EMData_1.0, EMData_1.1, or EMData_1.2".to_string(),
                got: s.to_string(),
            }),
        }
    }
}

impl std::fmt::Display for EMDataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EMDataFormat::V1_0 => write!(f, "EMData_1.0"),
            EMDataFormat::V1_1 => write!(f, "EMData_1.1"),
            EMDataFormat::V1_2 => write!(f, "EMData_1.2"),
        }
    }
}

/// Complete EMData structure
#[derive(Debug, Clone)]
pub struct EMData {
    /// File format version
    pub format: EMDataFormat,
    /// Optional comment/description
    pub comment: Option<String>,
    /// Transmitter definitions
    pub transmitters: Vec<Transmitter>,
    /// Frequencies (Hz)
    pub frequencies: Vec<f64>,
    /// Receiver definitions
    pub receivers: Vec<Receiver>,
    /// Data points
    pub data_points: Vec<DataPoint>,
    /// Phase convention
    pub phase_convention: PhaseConvention,
    /// Dipole length (m) - for EMData_1.2 finite dipoles
    pub dipole_length: Option<f64>,
}

impl Default for EMData {
    fn default() -> Self {
        Self {
            format: EMDataFormat::V1_0,
            comment: None,
            transmitters: Vec::new(),
            frequencies: Vec::new(),
            receivers: Vec::new(),
            data_points: Vec::new(),
            phase_convention: PhaseConvention::Lag,
            dipole_length: None,
        }
    }
}

impl EMData {
    /// Create a new empty EMData structure
    pub fn new() -> Self {
        Self::default()
    }

    /// Read EMData from a file
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::read(reader)
    }

    /// Read EMData from a reader
    pub fn read<R: BufRead>(reader: R) -> Result<Self> {
        let mut data = EMData::default();
        let mut lines = reader.lines().enumerate();
        let mut line_num = 0;

        // Helper to get next non-comment line
        let mut next_line = |lines: &mut std::iter::Enumerate<std::io::Lines<R>>| -> Result<(usize, String)> {
            loop {
                match lines.next() {
                    Some((num, Ok(line))) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() || trimmed.starts_with('!') {
                            continue;
                        }
                        return Ok((num + 1, line));
                    }
                    Some((num, Err(e))) => {
                        return Err(OccamIoError::Parse {
                            line: num + 1,
                            message: e.to_string(),
                        })
                    }
                    None => {
                        return Err(OccamIoError::Parse {
                            line: 0,
                            message: "Unexpected end of file".to_string(),
                        })
                    }
                }
            }
        };

        // Read format line
        let (ln, format_line) = next_line(&mut lines)?;
        line_num = ln;
        let format_str = parse_key_value(&format_line, "Format")
            .ok_or_else(|| OccamIoError::MissingField("Format".to_string()))?;
        data.format = format_str.parse()?;

        // Look for optional dipole length (EMData_1.2)
        let (ln, mut current_line) = next_line(&mut lines)?;
        line_num = ln;

        if let Some(dipole_len) = parse_key_value(&current_line, "Dipole Length") {
            data.dipole_length = Some(parse_float(&dipole_len, line_num)?);
            let (ln, line) = next_line(&mut lines)?;
            line_num = ln;
            current_line = line;
        }

        // Read transmitters
        let n_tx = parse_key_value(&current_line, "# Transmitters")
            .ok_or_else(|| OccamIoError::MissingField("# Transmitters".to_string()))?;
        let n_tx: usize = n_tx.parse().map_err(|_| OccamIoError::Parse {
            line: line_num,
            message: format!("Invalid transmitter count: {}", n_tx),
        })?;

        for _ in 0..n_tx {
            let (ln, line) = next_line(&mut lines)?;
            line_num = ln;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 5 {
                return Err(OccamIoError::Parse {
                    line: line_num,
                    message: "Transmitter requires 5 values (x, y, z, azimuth, dip)".to_string(),
                });
            }
            data.transmitters.push(Transmitter {
                x: parse_float(parts[0], line_num)?,
                y: parse_float(parts[1], line_num)?,
                z: parse_float(parts[2], line_num)?,
                azimuth: parse_float(parts[3], line_num)?,
                dip: parse_float(parts[4], line_num)?,
            });
        }

        // Read frequencies
        let (ln, freq_line) = next_line(&mut lines)?;
        line_num = ln;
        let n_freq = parse_key_value(&freq_line, "# Frequencies")
            .ok_or_else(|| OccamIoError::MissingField("# Frequencies".to_string()))?;
        let n_freq: usize = n_freq.parse().map_err(|_| OccamIoError::Parse {
            line: line_num,
            message: format!("Invalid frequency count: {}", n_freq),
        })?;

        for _ in 0..n_freq {
            let (ln, line) = next_line(&mut lines)?;
            line_num = ln;
            let freq = parse_float(line.trim(), line_num)?;
            data.frequencies.push(freq);
        }

        // Read phase convention
        let (ln, phase_line) = next_line(&mut lines)?;
        line_num = ln;
        let phase_str = parse_key_value(&phase_line, "Phase Convention")
            .ok_or_else(|| OccamIoError::MissingField("Phase Convention".to_string()))?;
        data.phase_convention = phase_str.parse()?;

        // Read receivers
        let (ln, rx_line) = next_line(&mut lines)?;
        line_num = ln;
        let n_rx = parse_key_value(&rx_line, "# Receivers")
            .ok_or_else(|| OccamIoError::MissingField("# Receivers".to_string()))?;
        let n_rx: usize = n_rx.parse().map_err(|_| OccamIoError::Parse {
            line: line_num,
            message: format!("Invalid receiver count: {}", n_rx),
        })?;

        for _ in 0..n_rx {
            let (ln, line) = next_line(&mut lines)?;
            line_num = ln;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                return Err(OccamIoError::Parse {
                    line: line_num,
                    message: "Receiver requires 6 values (x, y, z, theta, alpha, beta)".to_string(),
                });
            }
            data.receivers.push(Receiver {
                x: parse_float(parts[0], line_num)?,
                y: parse_float(parts[1], line_num)?,
                z: parse_float(parts[2], line_num)?,
                theta: parse_optional_float(parts[3], line_num)?,
                alpha: parse_optional_float(parts[4], line_num)?,
                beta: parse_optional_float(parts[5], line_num)?,
            });
        }

        // Read data points
        let (ln, data_line) = next_line(&mut lines)?;
        line_num = ln;
        let n_data = parse_key_value(&data_line, "# Data")
            .ok_or_else(|| OccamIoError::MissingField("# Data".to_string()))?;
        let n_data: usize = n_data.parse().map_err(|_| OccamIoError::Parse {
            line: line_num,
            message: format!("Invalid data count: {}", n_data),
        })?;

        for _ in 0..n_data {
            let (ln, line) = next_line(&mut lines)?;
            line_num = ln;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                return Err(OccamIoError::Parse {
                    line: line_num,
                    message: "Data point requires 6 values (type, freq#, tx#, rx#, value, error)"
                        .to_string(),
                });
            }

            let type_code: i32 = parts[0].parse().map_err(|_| OccamIoError::Parse {
                line: line_num,
                message: format!("Invalid data type: {}", parts[0]),
            })?;

            let data_type = DataType::from_code(type_code)?;

            // File uses 1-based indices, convert to 0-based
            let freq_idx: usize = parts[1].parse::<usize>().map_err(|_| OccamIoError::Parse {
                line: line_num,
                message: format!("Invalid frequency index: {}", parts[1]),
            })? - 1;

            // MT data uses Tx# = 0 to indicate no transmitter (natural source)
            let raw_tx: usize = parts[2].parse::<usize>().map_err(|_| OccamIoError::Parse {
                line: line_num,
                message: format!("Invalid transmitter index: {}", parts[2]),
            })?;
            let tx_idx = if raw_tx == 0 { 0 } else { raw_tx - 1 };

            let rx_idx: usize = parts[3].parse::<usize>().map_err(|_| OccamIoError::Parse {
                line: line_num,
                message: format!("Invalid receiver index: {}", parts[3]),
            })? - 1;

            // Validate indices
            if freq_idx >= data.frequencies.len() {
                return Err(OccamIoError::IndexOutOfBounds {
                    field: "frequency".to_string(),
                    index: freq_idx + 1,
                    max: data.frequencies.len(),
                });
            }
            // Skip transmitter bounds check for MT data (Tx# = 0 means no transmitter)
            if raw_tx > 0 && tx_idx >= data.transmitters.len() {
                return Err(OccamIoError::IndexOutOfBounds {
                    field: "transmitter".to_string(),
                    index: tx_idx + 1,
                    max: data.transmitters.len(),
                });
            }
            if rx_idx >= data.receivers.len() {
                return Err(OccamIoError::IndexOutOfBounds {
                    field: "receiver".to_string(),
                    index: rx_idx + 1,
                    max: data.receivers.len(),
                });
            }

            data.data_points.push(DataPoint {
                data_type,
                freq_idx,
                tx_idx,
                rx_idx,
                value: parse_float(parts[4], line_num)?,
                std_error: parse_float(parts[5], line_num)?,
            });
        }

        Ok(data)
    }

    /// Write EMData to a file
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write(writer)
    }

    /// Write EMData to a writer
    pub fn write<W: Write>(&self, mut writer: W) -> Result<()> {
        writeln!(writer, "Format:           {}", self.format)?;

        if let Some(ref comment) = self.comment {
            writeln!(writer, "! {}", comment)?;
        }

        if let Some(dipole_len) = self.dipole_length {
            writeln!(writer, "Dipole Length:    {:.6e}", dipole_len)?;
        }

        // Transmitters
        writeln!(writer, "# Transmitters:   {}", self.transmitters.len())?;
        writeln!(
            writer,
            "!             X            Y            Z      Azimuth          Dip"
        )?;
        for tx in &self.transmitters {
            writeln!(
                writer,
                "{:14.6e}{:14.6e}{:14.6e}{:12.2}{:12.2}",
                tx.x, tx.y, tx.z, tx.azimuth, tx.dip
            )?;
        }

        // Frequencies
        writeln!(writer, "# Frequencies:    {}", self.frequencies.len())?;
        for freq in &self.frequencies {
            writeln!(writer, "{:14.6e}", freq)?;
        }

        // Phase convention
        writeln!(writer, "Phase Convention: {}", self.phase_convention)?;

        // Receivers
        writeln!(writer, "# Receivers:      {}", self.receivers.len())?;
        writeln!(
            writer,
            "!             X            Y            Z        Theta        Alpha         Beta"
        )?;
        for rx in &self.receivers {
            let theta_str = rx.theta.map_or("?".to_string(), |v| format!("{:.2}", v));
            let alpha_str = rx.alpha.map_or("?".to_string(), |v| format!("{:.2}", v));
            let beta_str = rx.beta.map_or("?".to_string(), |v| format!("{:.2}", v));
            writeln!(
                writer,
                "{:14.6e}{:14.6e}{:14.6e}{:>12}{:>12}{:>12}",
                rx.x, rx.y, rx.z, theta_str, alpha_str, beta_str
            )?;
        }

        // Data
        writeln!(writer, "# Data:           {}", self.data_points.len())?;
        writeln!(
            writer,
            "!         Type        Freq#          Tx#          Rx#         Data    Std_Error"
        )?;
        for dp in &self.data_points {
            // MT data uses Tx# = 0 (no transmitter); CSEM uses 1-based
            let tx_file = if dp.data_type.is_mt() { 0 } else { dp.tx_idx + 1 };
            writeln!(
                writer,
                "{:12}{:12}{:12}{:12}{:14.6e}{:14.6e}",
                dp.data_type.code(),
                dp.freq_idx + 1, // Convert back to 1-based
                tx_file,
                dp.rx_idx + 1,
                dp.value,
                dp.std_error
            )?;
        }

        Ok(())
    }

    /// Get the number of data points
    pub fn n_data(&self) -> usize {
        self.data_points.len()
    }

    /// Check if this dataset contains CSEM data
    pub fn has_csem(&self) -> bool {
        self.data_points.iter().any(|dp| dp.data_type.is_csem())
    }

    /// Check if this dataset contains MT data
    pub fn has_mt(&self) -> bool {
        self.data_points.iter().any(|dp| dp.data_type.is_mt())
    }
}

/// Parse a key: value line
fn parse_key_value(line: &str, key: &str) -> Option<String> {
    let line = line.trim();
    if line.to_lowercase().starts_with(&key.to_lowercase()) {
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

/// Parse an optional float (can be "?" for unknown)
fn parse_optional_float(s: &str, line: usize) -> Result<Option<f64>> {
    let s = s.trim();
    if s == "?" {
        Ok(None)
    } else {
        Ok(Some(parse_float(s, line)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_codes() {
        assert_eq!(DataType::ExReal.code(), 1);
        assert_eq!(DataType::BzImag.code(), 16);
        assert_eq!(DataType::RhoXY.code(), 103);
        assert!(DataType::ExReal.is_csem());
        assert!(DataType::ExReal.is_electric());
        assert!(DataType::BxReal.is_magnetic());
        assert!(DataType::RhoXY.is_mt());

        // Polarization ellipse data types
        assert_eq!(DataType::PEmax.code(), 41);
        assert_eq!(DataType::PEmin.code(), 42);
        assert_eq!(DataType::PBmax.code(), 43);
        assert_eq!(DataType::PBmin.code(), 44);
        assert_eq!(DataType::from_code(41).unwrap(), DataType::PEmax);
        assert_eq!(DataType::from_code(44).unwrap(), DataType::PBmin);
        assert!(DataType::PEmax.is_csem());
        assert!(DataType::PEmin.is_electric());
        assert!(DataType::PBmax.is_magnetic());
        assert!(!DataType::PEmax.is_mt());
    }

    #[test]
    fn test_parse_key_value() {
        assert_eq!(
            parse_key_value("Format: EMData_1.0", "Format"),
            Some("EMData_1.0".to_string())
        );
        assert_eq!(
            parse_key_value("# Transmitters:   41", "# Transmitters"),
            Some("41".to_string())
        );
    }
}
