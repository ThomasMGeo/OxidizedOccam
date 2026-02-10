//! 1D Magnetotelluric (MT) Forward Modeling
//!
//! Computes MT impedance, apparent resistivity, and phase for 1D layered earth models.

mod error;
mod impedance;
mod sensitivity;

pub use error::*;
pub use impedance::*;
pub use sensitivity::*;

use num_complex::Complex64;
use std::f64::consts::PI;

/// Physical constants
pub mod constants {
    use std::f64::consts::PI;

    /// Permeability of free space (H/m)
    pub const MU0: f64 = 4.0e-7 * PI;
}

/// 1D layered earth model for MT calculations
#[derive(Debug, Clone)]
pub struct MT1DModel {
    /// Layer thicknesses (m), last layer is halfspace (thickness ignored)
    pub thicknesses: Vec<f64>,
    /// Layer resistivities (ohm-m)
    pub resistivities: Vec<f64>,
}

impl MT1DModel {
    /// Create a new MT model
    pub fn new(thicknesses: Vec<f64>, resistivities: Vec<f64>) -> Result<Self> {
        if thicknesses.len() != resistivities.len() {
            return Err(MT1DError::InvalidModel(
                "Thicknesses and resistivities must have same length".to_string(),
            ));
        }
        if thicknesses.is_empty() {
            return Err(MT1DError::InvalidModel(
                "Model must have at least one layer".to_string(),
            ));
        }
        Ok(Self {
            thicknesses,
            resistivities,
        })
    }

    /// Create a simple halfspace model
    pub fn halfspace(resistivity: f64) -> Self {
        Self {
            thicknesses: vec![f64::INFINITY],
            resistivities: vec![resistivity],
        }
    }

    /// Create a two-layer model
    pub fn two_layer(h1: f64, rho1: f64, rho2: f64) -> Self {
        Self {
            thicknesses: vec![h1, f64::INFINITY],
            resistivities: vec![rho1, rho2],
        }
    }

    /// Get number of layers
    pub fn n_layers(&self) -> usize {
        self.thicknesses.len()
    }

    /// Get conductivities
    pub fn conductivities(&self) -> Vec<f64> {
        self.resistivities.iter().map(|&r| 1.0 / r).collect()
    }
}

/// MT response at a single frequency
#[derive(Debug, Clone)]
pub struct MTResponse {
    /// Frequency (Hz)
    pub frequency: f64,
    /// Complex impedance Z (ohm)
    pub impedance: Complex64,
    /// Apparent resistivity (ohm-m)
    pub apparent_resistivity: f64,
    /// Phase (degrees)
    pub phase: f64,
}

impl MTResponse {
    /// Create from impedance
    pub fn from_impedance(frequency: f64, z: Complex64) -> Self {
        let omega = 2.0 * PI * frequency;
        let rho_a = z.norm().powi(2) / (omega * constants::MU0);
        let phase = z.arg().to_degrees();

        Self {
            frequency,
            impedance: z,
            apparent_resistivity: rho_a,
            phase,
        }
    }

    /// Get real part of impedance
    pub fn z_real(&self) -> f64 {
        self.impedance.re
    }

    /// Get imaginary part of impedance
    pub fn z_imag(&self) -> f64 {
        self.impedance.im
    }

    /// Get amplitude of impedance
    pub fn z_amplitude(&self) -> f64 {
        self.impedance.norm()
    }
}

/// Main MT1D computation engine
pub struct MT1D {
    /// Earth model
    pub model: MT1DModel,
}

impl MT1D {
    /// Create a new MT1D engine
    pub fn new(model: MT1DModel) -> Self {
        Self { model }
    }

    /// Compute MT response at a single frequency
    pub fn compute(&self, frequency: f64) -> Result<MTResponse> {
        if frequency <= 0.0 {
            return Err(MT1DError::InvalidFrequency(frequency));
        }

        let z = compute_impedance(&self.model, frequency)?;
        Ok(MTResponse::from_impedance(frequency, z))
    }

    /// Compute MT response at multiple frequencies
    pub fn compute_frequencies(&self, frequencies: &[f64]) -> Result<Vec<MTResponse>> {
        frequencies.iter().map(|&f| self.compute(f)).collect()
    }

    /// Compute response with analytical sensitivities
    pub fn compute_with_sensitivity(
        &self,
        frequency: f64,
    ) -> Result<(MTResponse, MTSensitivity)> {
        let response = self.compute(frequency)?;
        let sensitivity = compute_impedance_sensitivity_analytical(&self.model, frequency)?;
        Ok((response, sensitivity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halfspace_response() {
        // For a halfspace, apparent resistivity should equal true resistivity
        // at all frequencies
        let model = MT1DModel::halfspace(100.0);
        let mt = MT1D::new(model);

        let response = mt.compute(1.0).unwrap();
        assert!((response.apparent_resistivity - 100.0).abs() < 1.0);
        assert!((response.phase - 45.0).abs() < 1.0);
    }
}
