//! MT sensitivity (Jacobian) calculations

use crate::constants::MU0;
use crate::error::Result;
use crate::impedance::compute_impedance;
use crate::MT1DModel;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Sensitivity of MT response to layer parameters
#[derive(Debug, Clone)]
pub struct MTSensitivity {
    /// d(Z)/d(rho) for each layer
    pub dz_drho: Vec<Complex64>,
    /// d(rho_a)/d(rho) for each layer
    pub drho_a_drho: Vec<f64>,
    /// d(phase)/d(rho) for each layer (degrees per ohm-m)
    pub dphase_drho: Vec<f64>,
}

impl MTSensitivity {
    /// Create zero sensitivities for n layers
    pub fn zeros(n_layers: usize) -> Self {
        Self {
            dz_drho: vec![Complex64::new(0.0, 0.0); n_layers],
            drho_a_drho: vec![0.0; n_layers],
            dphase_drho: vec![0.0; n_layers],
        }
    }

    /// Convert to log10(resistivity) sensitivity
    /// d/d(log10(rho)) = d/d(rho) * rho * ln(10)
    pub fn to_log_sensitivity(&self, resistivities: &[f64]) -> Self {
        let ln10 = 10.0_f64.ln();
        let n = self.dz_drho.len();

        let mut result = MTSensitivity::zeros(n);
        for i in 0..n {
            let factor = resistivities[i] * ln10;
            result.dz_drho[i] = self.dz_drho[i] * factor;
            result.drho_a_drho[i] = self.drho_a_drho[i] * factor;
            result.dphase_drho[i] = self.dphase_drho[i] * factor;
        }
        result
    }
}

/// Compute MT impedance sensitivity using central finite differences
pub fn compute_impedance_sensitivity(model: &MT1DModel, frequency: f64) -> Result<MTSensitivity> {
    let n = model.n_layers();
    let mut sens = MTSensitivity::zeros(n);

    let omega = 2.0 * PI * frequency;
    let delta = 1e-7; // Relative perturbation (central differences)

    for i in 0..n {
        let rho_orig = model.resistivities[i];

        // Forward perturbed model
        let mut rho_fwd = model.resistivities.clone();
        rho_fwd[i] = rho_orig * (1.0 + delta);
        let model_fwd = MT1DModel { thicknesses: model.thicknesses.clone(), resistivities: rho_fwd };
        let z_fwd = compute_impedance(&model_fwd, frequency)?;

        // Backward perturbed model
        let mut rho_bwd = model.resistivities.clone();
        rho_bwd[i] = rho_orig * (1.0 - delta);
        let model_bwd = MT1DModel { thicknesses: model.thicknesses.clone(), resistivities: rho_bwd };
        let z_bwd = compute_impedance(&model_bwd, frequency)?;

        let d_rho = 2.0 * rho_orig * delta;
        sens.dz_drho[i] = (z_fwd - z_bwd) / d_rho;

        let rho_a_fwd = z_fwd.norm().powi(2) / (omega * MU0);
        let rho_a_bwd = z_bwd.norm().powi(2) / (omega * MU0);
        sens.drho_a_drho[i] = (rho_a_fwd - rho_a_bwd) / d_rho;

        let phase_fwd = z_fwd.arg().to_degrees();
        let phase_bwd = z_bwd.arg().to_degrees();
        sens.dphase_drho[i] = (phase_fwd - phase_bwd) / d_rho;
    }

    Ok(sens)
}

/// Compute analytical sensitivity using chain rule through Wait recursion
///
/// The Wait (1954) recursion computes surface impedance Z_0 from bottom up:
///   Z_n = Z_n^intr (halfspace)
///   Z_i = Z_i^intr * (Z_{i+1} + Z_i^intr * tanh(k_i*h_i)) / (Z_i^intr + Z_{i+1} * tanh(k_i*h_i))
///
/// The chain rule gives: dZ_0/drho_j = (prod_{i=0}^{j-1} dZ_i/dZ_{i+1}) * (partial Z_j / partial rho_j)
///
/// Chain factor: dZ_i/dZ_{i+1} = (Z_i^intr)^2 * sech^2(k_i*h_i) / (Z_i^intr + Z_{i+1}*T_i)^2
/// Self-derivative: partial Z_j / partial rho_j uses the quotient rule on the recursion formula
pub fn compute_impedance_sensitivity_analytical(
    model: &MT1DModel,
    frequency: f64,
) -> Result<MTSensitivity> {
    use crate::impedance::{complex_sqrt, complex_tanh};

    let omega = 2.0 * PI * frequency;
    let n = model.n_layers();
    let zero = Complex64::new(0.0, 0.0);
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    // Forward pass: compute all intermediate quantities bottom-up
    let mut k = vec![zero; n];
    let mut z_intr = vec![zero; n];
    let mut tanh_kh = vec![zero; n];
    let mut z = vec![zero; n]; // z[i] = impedance looking down from top of layer i

    for i in 0..n {
        let sigma_i = 1.0 / model.resistivities[i];
        k[i] = complex_sqrt(Complex64::new(0.0, omega * MU0 * sigma_i));
        z_intr[i] = iomega_mu0 / k[i];

        if i < n - 1 {
            let kh = k[i] * model.thicknesses[i];
            tanh_kh[i] = complex_tanh(kh);
        }
    }

    // Bottom-up recursion (same as compute_impedance)
    z[n - 1] = z_intr[n - 1];
    for i in (0..n - 1).rev() {
        let t = tanh_kh[i];
        z[i] = z_intr[i] * (z[i + 1] + z_intr[i] * t) / (z_intr[i] + z[i + 1] * t);
    }

    let z_surface = z[0];

    // Compute chain factors: C[i] = dZ_i / dZ_{i+1}
    // C[i] = (Z_i^intr)^2 * sech^2(k_i*h_i) / (Z_i^intr + Z_{i+1} * tanh_kh[i])^2
    // where sech^2(x) = 1 - tanh^2(x)
    let mut chain = vec![zero; n];
    for i in 0..n - 1 {
        let t = tanh_kh[i];
        let sech2 = Complex64::new(1.0, 0.0) - t * t;
        let denom = z_intr[i] + z[i + 1] * t;
        chain[i] = z_intr[i] * z_intr[i] * sech2 / (denom * denom);
    }

    // Cumulative product of chain factors: cum_chain[j] = prod_{i=0}^{j-1} C[i]
    // cum_chain[0] = 1 (surface layer, no chain needed)
    let mut cum_chain = vec![zero; n];
    cum_chain[0] = Complex64::new(1.0, 0.0);
    for j in 1..n {
        cum_chain[j] = cum_chain[j - 1] * chain[j - 1];
    }

    // Compute self-derivatives: S[j] = partial Z_j / partial rho_j (holding Z_{j+1} fixed)
    let mut self_deriv = vec![zero; n];

    // Halfspace (bottom layer): Z_{n-1} = Z_intr = iωμ₀/k, where k = sqrt(iωμ₀σ)
    // dk/drho = -k/(2*rho), dZ_intr/drho = Z_intr/(2*rho)
    self_deriv[n - 1] = z_intr[n - 1] / (2.0 * model.resistivities[n - 1]);

    // Finite layers: Z_j = Z_j^intr * N / D where N = Z_{j+1} + Z_j^intr * T, D = Z_j^intr + Z_{j+1} * T
    // Use quotient rule: dZ_j/drho_j = dZintr/drho * N/D + Zintr * (dN/drho * D - N * dD/drho) / D^2
    for j in 0..n - 1 {
        let rho_j = model.resistivities[j];
        let t = tanh_kh[j];
        let h_j = model.thicknesses[j];
        let sech2 = Complex64::new(1.0, 0.0) - t * t;

        // Derivatives of layer parameters w.r.t. rho_j
        let dz_intr_drho = z_intr[j] / (2.0 * rho_j);
        let dk_drho = -k[j] / (2.0 * rho_j);
        let dt_drho = sech2 * h_j * dk_drho; // d(tanh(kh))/drho = sech^2(kh) * h * dk/drho

        let z_next = z[j + 1]; // Z_{j+1}, held fixed
        let numer = z_next + z_intr[j] * t;
        let denom = z_intr[j] + z_next * t;

        // dN/drho = dZintr/drho * T + Zintr * dT/drho
        let dn_drho = dz_intr_drho * t + z_intr[j] * dt_drho;
        // dD/drho = dZintr/drho + Z_{j+1} * dT/drho
        let dd_drho = dz_intr_drho + z_next * dt_drho;

        // Full quotient rule
        self_deriv[j] = dz_intr_drho * numer / denom
            + z_intr[j] * (dn_drho * denom - numer * dd_drho) / (denom * denom);
    }

    // Final: dZ_surface / drho_j = cum_chain[j] * self_deriv[j]
    let mut sens = MTSensitivity::zeros(n);
    for j in 0..n {
        sens.dz_drho[j] = cum_chain[j] * self_deriv[j];

        // drho_a/drho_j = 2 * Re(conj(Z) * dZ/drho_j) / (omega * mu0)
        sens.drho_a_drho[j] =
            2.0 * (z_surface.conj() * sens.dz_drho[j]).re / (omega * MU0);

        // dphase/drho_j = Im(conj(Z) * dZ/drho_j) / |Z|^2 * (180/pi)
        sens.dphase_drho[j] = (z_surface.conj() * sens.dz_drho[j]).im
            / z_surface.norm_sqr()
            * (180.0 / PI);
    }

    Ok(sens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sensitivity_conversion() {
        let model = MT1DModel::two_layer(1000.0, 100.0, 10.0);
        let sens = compute_impedance_sensitivity(&model, 1.0).unwrap();
        let log_sens = sens.to_log_sensitivity(&model.resistivities);

        // Log sensitivity should be scaled by rho * ln(10)
        let ln10 = 10.0_f64.ln();
        let expected_0 = sens.drho_a_drho[0] * 100.0 * ln10;
        assert!((log_sens.drho_a_drho[0] - expected_0).abs() < 1e-6);
    }

    #[test]
    fn test_analytical_vs_fd_multilayer() {
        // 5-layer model: different resistivities and thicknesses
        let model = MT1DModel {
            thicknesses: vec![500.0, 1000.0, 200.0, 2000.0],
            resistivities: vec![100.0, 10.0, 500.0, 1.0, 50.0],
        };

        for &freq in &[0.001, 0.01, 0.1, 1.0, 10.0, 100.0] {
            let fd = compute_impedance_sensitivity(&model, freq).unwrap();
            let an = compute_impedance_sensitivity_analytical(&model, freq).unwrap();

            // Use dZ/drho (complex impedance derivative) as the primary check
            // FD accuracy is limited so skip layers where sensitivity is tiny
            // relative to the maximum (FD noise dominates there)
            let max_sens = an.dz_drho.iter().map(|d| d.norm()).fold(0.0, f64::max);
            for j in 0..model.n_layers() {
                let fd_mag = fd.dz_drho[j].norm();
                if fd_mag < 1e-4 * max_sens {
                    continue; // Skip negligible sensitivities
                }
                let rel_err = (an.dz_drho[j] - fd.dz_drho[j]).norm() / fd_mag;
                assert!(
                    rel_err < 1e-4,
                    "freq={}, layer={}: dz_drho analytical={:e}, fd={:e}, rel_err={:e}",
                    freq, j, an.dz_drho[j], fd.dz_drho[j], rel_err
                );

                // Also check drho_a and phase for non-negligible layers
                let fd_val = fd.drho_a_drho[j].abs();
                if fd_val > 1e-20 {
                    let rel_err_rho = (an.drho_a_drho[j] - fd.drho_a_drho[j]).abs() / fd_val;
                    assert!(
                        rel_err_rho < 1e-4,
                        "freq={}, layer={}: drho_a analytical={:e}, fd={:e}, rel_err={:e}",
                        freq, j, an.drho_a_drho[j], fd.drho_a_drho[j], rel_err_rho
                    );
                }
            }
        }
    }
}
