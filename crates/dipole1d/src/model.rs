//! Layered earth model for Dipole1D

use crate::constants::MU0;
use crate::error::{Dipole1DError, Result};
use crate::layers::LayerCoefficients;
use num_complex::Complex64;
use std::f64::consts::PI;

/// 1D layered earth model
#[derive(Debug, Clone)]
pub struct LayeredEarth {
    /// Depth to top of each layer (m), first element is typically large negative (air)
    pub depths: Vec<f64>,
    /// Conductivity of each layer (S/m)
    pub conductivities: Vec<f64>,
}

impl LayeredEarth {
    /// Create a new layered earth model
    ///
    /// # Arguments
    /// * `depths` - Depth to top of each layer (m)
    /// * `conductivities` - Conductivity of each layer (S/m)
    pub fn new(depths: Vec<f64>, conductivities: Vec<f64>) -> Result<Self> {
        if depths.len() != conductivities.len() {
            return Err(Dipole1DError::InvalidModel(
                "Depths and conductivities must have same length".to_string(),
            ));
        }
        if depths.is_empty() {
            return Err(Dipole1DError::InvalidModel(
                "Model must have at least one layer".to_string(),
            ));
        }

        Ok(Self {
            depths,
            conductivities,
        })
    }

    /// Create a simple halfspace model
    pub fn halfspace(conductivity: f64) -> Self {
        Self {
            depths: vec![-100000.0, 0.0],
            conductivities: vec![1e-9, conductivity], // Air (1e9 ohm-m = 1e-9 S/m) + halfspace
        }
    }

    /// Create a marine model with air, water, and sediment layers
    ///
    /// # Arguments
    /// * `water_depth` - Depth of water column (m)
    /// * `water_conductivity` - Conductivity of seawater (S/m)
    /// * `layer_depths` - Depths below seafloor for additional layers
    /// * `layer_conductivities` - Conductivities for additional layers
    pub fn marine(
        water_depth: f64,
        water_conductivity: f64,
        layer_depths: &[f64],
        layer_conductivities: &[f64],
    ) -> Result<Self> {
        if layer_depths.len() != layer_conductivities.len() {
            return Err(Dipole1DError::InvalidModel(
                "Layer depths and conductivities must have same length".to_string(),
            ));
        }

        let mut depths = vec![-100000.0, 0.0]; // Air starts at -100km
        let mut conductivities = vec![1e-12, water_conductivity]; // Air conductivity ~0

        // Add sediment layers
        for (&depth, &cond) in layer_depths.iter().zip(layer_conductivities.iter()) {
            depths.push(water_depth + depth);
            conductivities.push(cond);
        }

        Ok(Self {
            depths,
            conductivities,
        })
    }

    /// Get number of layers
    pub fn n_layers(&self) -> usize {
        self.depths.len()
    }

    /// Find which layer contains a given depth
    /// Following Fortran convention: sites on boundaries use the layer above
    /// (i.e., strictly greater than the layer top depth)
    pub fn find_layer(&self, z: f64) -> usize {
        for i in (0..self.depths.len()).rev() {
            if z > self.depths[i] {
                return i;
            }
        }
        0
    }

    /// Get layer thickness (infinity for last layer)
    pub fn thickness(&self, layer: usize) -> f64 {
        if layer + 1 < self.depths.len() {
            self.depths[layer + 1] - self.depths[layer]
        } else {
            f64::INFINITY
        }
    }

    /// Compute wavenumber for a layer
    ///
    /// k^2 = i * omega * mu0 * sigma (ignoring displacement currents)
    ///
    /// Note: This uses the e^{+iωt} convention (positive imaginary part).
    /// The primary forward model in fields.rs uses the Fortran e^{-iωt} convention:
    /// k2 = -iωμ₀σ (Dipole1D.f90 line 1667)
    pub fn wavenumber_squared(&self, layer: usize, omega: f64) -> Complex64 {
        let sigma = self.conductivities[layer];
        Complex64::new(0.0, omega * MU0 * sigma)
    }

    /// Compute vertical wavenumber u = sqrt(lambda^2 - k^2)
    /// where lambda is the horizontal wavenumber
    pub fn vertical_wavenumber(&self, layer: usize, lambda: f64, omega: f64) -> Complex64 {
        let k2 = self.wavenumber_squared(layer, omega);
        let u2 = Complex64::new(lambda * lambda, 0.0) - k2;
        complex_sqrt(u2)
    }

    /// Compute intrinsic impedance of layer
    /// eta = i * omega * mu0 / u
    pub fn impedance(&self, layer: usize, lambda: f64, omega: f64) -> Complex64 {
        let u = self.vertical_wavenumber(layer, lambda, omega);
        Complex64::new(0.0, omega * MU0) / u
    }

    /// Compute admittance of layer
    /// gamma = (sigma + i * omega * eps) / u ≈ sigma / u
    pub fn admittance(&self, layer: usize, lambda: f64, omega: f64) -> Complex64 {
        let u = self.vertical_wavenumber(layer, lambda, omega);
        Complex64::new(self.conductivities[layer], 0.0) / u
    }

    /// Compute all layer coefficients for a given frequency
    pub fn compute_coefficients(&self, omega: f64, tx_layer: usize) -> LayerCoefficients {
        LayerCoefficients::compute(self, omega, tx_layer)
    }

    /// Get resistivities (1/conductivity) for all layers
    pub fn resistivities(&self) -> Vec<f64> {
        self.conductivities.iter().map(|&s| 1.0 / s).collect()
    }

    /// Update conductivities from resistivity values
    pub fn set_resistivities(&mut self, resistivities: &[f64]) {
        for (i, &r) in resistivities.iter().enumerate() {
            if i < self.conductivities.len() {
                self.conductivities[i] = 1.0 / r;
            }
        }
    }
}

/// Complex square root with proper branch cut
/// Returns the principal square root with positive real part
fn complex_sqrt(z: Complex64) -> Complex64 {
    let r = z.norm();
    let theta = z.arg();
    let sqrt_r = r.sqrt();
    Complex64::from_polar(sqrt_r, theta / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_layer() {
        let model = LayeredEarth::new(
            vec![-100000.0, 0.0, 1000.0, 2000.0],
            vec![0.0, 3.0, 1.0, 0.01],
        )
        .unwrap();

        assert_eq!(model.find_layer(-50000.0), 0); // In air
        assert_eq!(model.find_layer(500.0), 1); // In water
        assert_eq!(model.find_layer(1500.0), 2); // In first sediment
        assert_eq!(model.find_layer(3000.0), 3); // In last layer (halfspace)
    }

    #[test]
    fn test_wavenumber() {
        let model = LayeredEarth::halfspace(1.0);
        let omega = 2.0 * PI * 1.0; // 1 Hz

        let k2 = model.wavenumber_squared(1, omega);
        // k^2 = i * omega * mu0 * sigma
        let expected = Complex64::new(0.0, omega * MU0 * 1.0);
        assert!((k2 - expected).norm() < 1e-20);
    }
}
