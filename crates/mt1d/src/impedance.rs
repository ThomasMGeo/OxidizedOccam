//! MT impedance calculation using recursive algorithm
//!
//! Computes surface impedance for a 1D layered earth model

use crate::constants::MU0;
use crate::error::Result;
use crate::MT1DModel;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Compute MT surface impedance using recursive algorithm
///
/// Uses the Wait (1954) recursive formula starting from the bottom layer
/// (halfspace) and working upward to the surface.
///
/// Z_n = Z_n^intrinsic (for halfspace)
/// Z_i = Z_i^intrinsic * (Z_{i+1} + Z_i^intrinsic * tanh(k_i * h_i)) /
///                       (Z_i^intrinsic + Z_{i+1} * tanh(k_i * h_i))
pub fn compute_impedance(model: &MT1DModel, frequency: f64) -> Result<Complex64> {
    let omega = 2.0 * PI * frequency;
    let n = model.n_layers();

    // Start from bottom layer (halfspace)
    let sigma_n = 1.0 / model.resistivities[n - 1];
    let k_n = complex_sqrt(Complex64::new(0.0, omega * MU0 * sigma_n));
    let z_intrinsic_n = Complex64::new(0.0, omega * MU0) / k_n;

    let mut z = z_intrinsic_n;

    // Recurse upward through layers
    for i in (0..n - 1).rev() {
        let sigma_i = 1.0 / model.resistivities[i];
        let k_i = complex_sqrt(Complex64::new(0.0, omega * MU0 * sigma_i));
        let z_intrinsic_i = Complex64::new(0.0, omega * MU0) / k_i;

        let h_i = model.thicknesses[i];
        let kh = k_i * h_i;

        // Compute tanh(k*h) with numerical stability
        let tanh_kh = complex_tanh(kh);

        // Recursive relation
        z = z_intrinsic_i * (z + z_intrinsic_i * tanh_kh) / (z_intrinsic_i + z * tanh_kh);
    }

    Ok(z)
}

/// Compute intrinsic impedance for a layer
///
/// Z_intrinsic = i*omega*mu0 / k
/// where k = sqrt(i*omega*mu0*sigma)
pub fn intrinsic_impedance(resistivity: f64, frequency: f64) -> Complex64 {
    let omega = 2.0 * PI * frequency;
    let sigma = 1.0 / resistivity;
    let k = complex_sqrt(Complex64::new(0.0, omega * MU0 * sigma));
    Complex64::new(0.0, omega * MU0) / k
}

/// Compute propagation constant for a layer
///
/// k = sqrt(i*omega*mu0*sigma)
pub fn propagation_constant(resistivity: f64, frequency: f64) -> Complex64 {
    let omega = 2.0 * PI * frequency;
    let sigma = 1.0 / resistivity;
    complex_sqrt(Complex64::new(0.0, omega * MU0 * sigma))
}

/// Compute skin depth for a layer
///
/// delta = sqrt(2*rho / (omega*mu0))
pub fn skin_depth(resistivity: f64, frequency: f64) -> f64 {
    let omega = 2.0 * PI * frequency;
    (2.0 * resistivity / (omega * MU0)).sqrt()
}

/// Complex square root with proper branch cut
pub(crate) fn complex_sqrt(z: Complex64) -> Complex64 {
    let r = z.norm();
    let theta = z.arg();
    Complex64::from_polar(r.sqrt(), theta / 2.0)
}

/// Complex hyperbolic tangent with numerical stability
pub(crate) fn complex_tanh(z: Complex64) -> Complex64 {
    // For large |Re(z)|, tanh -> +/- 1
    if z.re.abs() > 20.0 {
        return Complex64::new(z.re.signum(), 0.0);
    }

    let exp_2z = (2.0 * z).exp();
    (exp_2z - Complex64::new(1.0, 0.0)) / (exp_2z + Complex64::new(1.0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_layer_high_freq() {
        // At very high frequency, should see only first layer
        let model = crate::MT1DModel::two_layer(1000.0, 10.0, 1000.0);
        let freq = 1000.0; // High frequency

        let z = compute_impedance(&model, freq).unwrap();
        let omega = 2.0 * PI * freq;
        let rho_a = z.norm().powi(2) / (omega * MU0);

        // Should be close to first layer resistivity
        assert!((rho_a - 10.0).abs() / 10.0 < 0.1);
    }

    #[test]
    fn test_two_layer_low_freq() {
        // At very low frequency, should see deeper
        let model = crate::MT1DModel::two_layer(100.0, 10.0, 1000.0);
        let freq = 0.001; // Very low frequency

        let z = compute_impedance(&model, freq).unwrap();
        let omega = 2.0 * PI * freq;
        let rho_a = z.norm().powi(2) / (omega * MU0);

        // Should be closer to second layer resistivity
        // (but not exactly, depends on layer thickness)
        assert!(rho_a > 50.0); // More resistive than first layer alone
    }
}
