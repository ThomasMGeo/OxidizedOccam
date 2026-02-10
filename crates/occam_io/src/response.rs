//! Response file writers (EMResp_1.2 and CSV)
//!
//! Writes forward-modeled responses alongside observed data.

use crate::data::{DataType, EMData};
use crate::error::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Human-readable name for a data type code
fn data_type_name(dt: DataType) -> &'static str {
    match dt {
        DataType::ExReal => "ExReal",
        DataType::ExImag => "ExImag",
        DataType::EyReal => "EyReal",
        DataType::EyImag => "EyImag",
        DataType::EzReal => "EzReal",
        DataType::EzImag => "EzImag",
        DataType::BxReal => "BxReal",
        DataType::BxImag => "BxImag",
        DataType::ByReal => "ByReal",
        DataType::ByImag => "ByImag",
        DataType::BzReal => "BzReal",
        DataType::BzImag => "BzImag",
        DataType::ExAmp => "ExAmp",
        DataType::ExPhase => "ExPhase",
        DataType::EyAmp => "EyAmp",
        DataType::EyPhase => "EyPhase",
        DataType::EzAmp => "EzAmp",
        DataType::EzPhase => "EzPhase",
        DataType::BxAmp => "BxAmp",
        DataType::BxPhase => "BxPhase",
        DataType::ByAmp => "ByAmp",
        DataType::ByPhase => "ByPhase",
        DataType::BzAmp => "BzAmp",
        DataType::BzPhase => "BzPhase",
        DataType::PEmax => "PEmax",
        DataType::PEmin => "PEmin",
        DataType::PBmax => "PBmax",
        DataType::PBmin => "PBmin",
        DataType::RhoXX => "RhoXX",
        DataType::PhiXX => "PhiXX",
        DataType::RhoXY => "RhoXY",
        DataType::PhiXY => "PhiXY",
        DataType::RhoYX => "RhoYX",
        DataType::PhiYX => "PhiYX",
        DataType::RhoYY => "RhoYY",
        DataType::PhiYY => "PhiYY",
        DataType::ZxxReal => "ZxxReal",
        DataType::ZxxImag => "ZxxImag",
        DataType::ZxyReal => "ZxyReal",
        DataType::ZxyImag => "ZxyImag",
        DataType::ZyxReal => "ZyxReal",
        DataType::ZyxImag => "ZyxImag",
        DataType::ZyyReal => "ZyyReal",
        DataType::ZyyImag => "ZyyImag",
    }
}

/// Write an EMResp_1.2 response file (Fortran-compatible format).
///
/// Format reference: CSEM1D.f90 lines 2968-2993, Matlab reader: readEMResp.m
pub fn write_resp_file<P: AsRef<Path>>(
    path: P,
    data: &EMData,
    computed: &[f64],
) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let dipole_len = data.dipole_length.unwrap_or(0.0);

    // Header
    writeln!(w, "Format:    EMResp_1.2")?;
    writeln!(w, "Dipole length:    {:.5}", dipole_len)?;

    // Transmitters
    writeln!(w, "# Transmitters:    {}", data.transmitters.len())?;
    writeln!(w, "!            X            Y            Z      Azimuth          Dip")?;
    for tx in &data.transmitters {
        // Fortran format: (1x,5(f12.1,1x))
        writeln!(w, " {:12.1} {:12.1} {:12.1} {:12.1} {:12.1}",
            tx.x, tx.y, tx.z, tx.azimuth, tx.dip)?;
    }

    // Frequencies
    writeln!(w, "# Frequencies:    {}", data.frequencies.len())?;
    for freq in &data.frequencies {
        writeln!(w, " {:13.5e}", freq)?;
    }

    // Receivers
    writeln!(w, "# Receivers:    {}", data.receivers.len())?;
    writeln!(w, "!            X            Y            Z        Theta        Alpha         Beta")?;
    for rx in &data.receivers {
        let theta = rx.theta.unwrap_or(0.0);
        let alpha = rx.alpha.unwrap_or(0.0);
        let beta = rx.beta.unwrap_or(0.0);
        // Fortran format: (1x,6(f12.1,1x))
        writeln!(w, " {:12.1} {:12.1} {:12.1} {:12.1} {:12.1} {:12.1}",
            rx.x, rx.y, rx.z, theta, alpha, beta)?;
    }

    // Data
    writeln!(w, "# Data:    {}", data.data_points.len())?;
    writeln!(w, "!        Type        Freq#          Tx#          Rx#          Data      StdError      Response      Residual")?;
    for (i, dp) in data.data_points.iter().enumerate() {
        let obs = dp.value;
        let comp = computed[i];
        let residual = (obs - comp) / dp.std_error;
        // Fortran format: (1x,4(i12,1x),3(g13.5,1x),f13.2)
        // MT data uses Tx# = 0 (no transmitter); CSEM uses 1-based
        let tx_file = if dp.data_type.is_mt() { 0 } else { dp.tx_idx + 1 };
        writeln!(w, " {:12} {:12} {:12} {:12} {:13.5e} {:13.5e} {:13.5e} {:13.2}",
            dp.data_type.code(),
            dp.freq_idx + 1,
            tx_file,
            dp.rx_idx + 1,
            obs,
            dp.std_error,
            comp,
            residual)?;
    }

    Ok(())
}

/// Write a tidy CSV response file for plotting in Python/R.
///
/// One row per data point with all metadata denormalized.
pub fn write_csv_file<P: AsRef<Path>>(
    path: P,
    data: &EMData,
    computed: &[f64],
) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header
    writeln!(w, "type,type_name,freq_hz,tx_idx,tx_x,tx_y,tx_z,tx_azimuth,tx_dip,rx_idx,rx_x,rx_y,rx_z,offset_m,observed,std_error,computed,residual")?;

    for (i, dp) in data.data_points.iter().enumerate() {
        let tx = &data.transmitters[dp.tx_idx];
        let rx = &data.receivers[dp.rx_idx];
        let freq = data.frequencies[dp.freq_idx];
        let obs = dp.value;
        let comp = computed[i];
        let residual = (obs - comp) / dp.std_error;
        let offset = ((tx.x - rx.x).powi(2) + (tx.y - rx.y).powi(2)).sqrt();

        writeln!(w, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            dp.data_type.code(),
            data_type_name(dp.data_type),
            freq,
            dp.tx_idx + 1,
            tx.x, tx.y, tx.z, tx.azimuth, tx.dip,
            dp.rx_idx + 1,
            rx.x, rx.y, rx.z,
            offset,
            obs,
            dp.std_error,
            comp,
            residual)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataPoint, EMDataFormat, PhaseConvention, Receiver, Transmitter};

    fn sample_data() -> (EMData, Vec<f64>) {
        let data = EMData {
            format: EMDataFormat::V1_2,
            comment: None,
            transmitters: vec![Transmitter {
                x: 0.0, y: 0.0, z: 950.0, azimuth: 0.0, dip: 0.0,
            }],
            frequencies: vec![0.25, 1.0],
            receivers: vec![Receiver {
                x: 500.0, y: 0.0, z: 1000.0,
                theta: Some(0.0), alpha: Some(0.0), beta: Some(0.0),
            }],
            data_points: vec![
                DataPoint {
                    data_type: DataType::ExReal, freq_idx: 0,
                    tx_idx: 0, rx_idx: 0, value: 1.0e-12, std_error: 1.0e-13,
                },
                DataPoint {
                    data_type: DataType::ExImag, freq_idx: 0,
                    tx_idx: 0, rx_idx: 0, value: -2.0e-12, std_error: 2.0e-13,
                },
            ],
            phase_convention: PhaseConvention::Lag,
            dipole_length: Some(0.0),
        };
        let computed = vec![1.1e-12, -1.9e-12];
        (data, computed)
    }

    #[test]
    fn test_write_csv_file_header() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_response.csv");
        let (data, computed) = sample_data();
        write_csv_file(&path, &data, &computed).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines[0].starts_with("type,type_name,freq_hz"));
        assert_eq!(lines.len(), 3); // header + 2 data rows
        assert!(lines[1].contains("ExReal"));
        assert!(lines[2].contains("ExImag"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_resp_file_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_response.resp");
        let (data, computed) = sample_data();
        write_resp_file(&path, &data, &computed).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Format:    EMResp_1.2"));
        assert!(content.contains("# Data:    2"));
        // Check residual column is present
        assert!(content.contains("Residual"));

        std::fs::remove_file(&path).ok();
    }
}
