//! Sensitivity (Jacobian) calculations for Dipole1D
//!
//! Computes analytical derivatives of EM fields with respect to layer conductivities
//! Following the Fortran Dipole1D implementation by Kerry Key

use crate::constants::MU0;
use crate::error::Result;
use crate::fields::{
    compute_gamma_expmgh,
    compute_hed_coefficients_with_derivs_buf,
    compute_tm_te_recursion_with_derivs_buf,
    compute_tm_recursion_with_derivs_buf,
};
use crate::filters::HankelFilter;
use crate::layers::LayerCoefficients;
use crate::model::LayeredEarth;
use crate::work_buffers::WorkBuffers;
use crate::EMFields;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Minimum horizontal distance (m) to avoid singularity
const MIN_RHO: f64 = 1.0;
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Sensitivity of EM fields to layer conductivity
#[derive(Debug, Clone, Default)]
pub struct EMFieldSensitivity {
    /// d(Ex)/d(sigma) for each layer
    pub dex_dsigma: Vec<Complex64>,
    /// d(Ey)/d(sigma) for each layer
    pub dey_dsigma: Vec<Complex64>,
    /// d(Ez)/d(sigma) for each layer
    pub dez_dsigma: Vec<Complex64>,
    /// d(Bx)/d(sigma) for each layer
    pub dbx_dsigma: Vec<Complex64>,
    /// d(By)/d(sigma) for each layer
    pub dby_dsigma: Vec<Complex64>,
    /// d(Bz)/d(sigma) for each layer
    pub dbz_dsigma: Vec<Complex64>,
}

impl EMFieldSensitivity {
    /// Create zero sensitivities for n layers
    pub fn zeros(n_layers: usize) -> Self {
        Self {
            dex_dsigma: vec![ZERO; n_layers],
            dey_dsigma: vec![ZERO; n_layers],
            dez_dsigma: vec![ZERO; n_layers],
            dbx_dsigma: vec![ZERO; n_layers],
            dby_dsigma: vec![ZERO; n_layers],
            dbz_dsigma: vec![ZERO; n_layers],
        }
    }

    /// Add another sensitivity
    pub fn add(&mut self, other: &EMFieldSensitivity) {
        for i in 0..self.dex_dsigma.len().min(other.dex_dsigma.len()) {
            self.dex_dsigma[i] += other.dex_dsigma[i];
            self.dey_dsigma[i] += other.dey_dsigma[i];
            self.dez_dsigma[i] += other.dez_dsigma[i];
            self.dbx_dsigma[i] += other.dbx_dsigma[i];
            self.dby_dsigma[i] += other.dby_dsigma[i];
            self.dbz_dsigma[i] += other.dbz_dsigma[i];
        }
    }

    /// Scale all sensitivities
    pub fn scale(&mut self, factor: Complex64) {
        for i in 0..self.dex_dsigma.len() {
            self.dex_dsigma[i] *= factor;
            self.dey_dsigma[i] *= factor;
            self.dez_dsigma[i] *= factor;
            self.dbx_dsigma[i] *= factor;
            self.dby_dsigma[i] *= factor;
            self.dbz_dsigma[i] *= factor;
        }
    }

    /// Convert conductivity sensitivity to resistivity sensitivity
    /// d/d(rho) = d/d(sigma) * d(sigma)/d(rho) = -sigma^2 * d/d(sigma)
    pub fn to_resistivity_sensitivity(&self, conductivities: &[f64]) -> Self {
        let n = self.dex_dsigma.len();
        let mut result = EMFieldSensitivity::zeros(n);

        for i in 0..n {
            let factor = Complex64::new(-conductivities[i] * conductivities[i], 0.0);
            result.dex_dsigma[i] = self.dex_dsigma[i] * factor;
            result.dey_dsigma[i] = self.dey_dsigma[i] * factor;
            result.dez_dsigma[i] = self.dez_dsigma[i] * factor;
            result.dbx_dsigma[i] = self.dbx_dsigma[i] * factor;
            result.dby_dsigma[i] = self.dby_dsigma[i] * factor;
            result.dbz_dsigma[i] = self.dbz_dsigma[i] * factor;
        }

        result
    }

    /// Convert conductivity sensitivity to log10(resistivity) sensitivity
    /// d/d(log10(rho)) = d/d(rho) * rho * ln(10)
    pub fn to_log_resistivity_sensitivity(&self, conductivities: &[f64]) -> Self {
        let rho_sens = self.to_resistivity_sensitivity(conductivities);
        let ln10 = 10.0_f64.ln();
        let n = rho_sens.dex_dsigma.len();
        let mut result = EMFieldSensitivity::zeros(n);

        for i in 0..n {
            let rho = 1.0 / conductivities[i];
            let factor = Complex64::new(rho * ln10, 0.0);
            result.dex_dsigma[i] = rho_sens.dex_dsigma[i] * factor;
            result.dey_dsigma[i] = rho_sens.dey_dsigma[i] * factor;
            result.dez_dsigma[i] = rho_sens.dez_dsigma[i] * factor;
            result.dbx_dsigma[i] = rho_sens.dbx_dsigma[i] * factor;
            result.dby_dsigma[i] = rho_sens.dby_dsigma[i] * factor;
            result.dbz_dsigma[i] = rho_sens.dbz_dsigma[i] * factor;
        }

        result
    }
}

/// Compute HED fields with analytical sensitivity using pre-allocated WorkBuffers.
///
/// Returns fields and a single EMFieldSensitivity (wrapped in Vec for API compatibility).
pub fn compute_hed_fields_with_sensitivity(
    model: &LayeredEarth,
    _coeffs: &LayerCoefficients,
    filter: &HankelFilter,
    z_tx: f64,
    z_rx: f64,
    tx_layer: usize,
    rx_layer: usize,
    omega: f64,
    rho: f64,
    theta: f64,
    compute_b: bool,
    buf: &mut WorkBuffers,
) -> Result<(EMFields, Vec<EMFieldSensitivity>)> {
    let filter_coeffs = filter.coefficients();
    let n_filter = filter_coeffs.base.len();
    let n = model.n_layers();

    let r = if rho < MIN_RHO { MIN_RHO } else { rho };

    let sintheta = theta.sin();
    let costheta = theta.cos();
    let sin2theta = (2.0 * theta).sin();
    let cos2theta = (2.0 * theta).cos();

    // Field accumulators
    let mut exh = ZERO;
    let mut eyh = ZERO;
    let mut jzh = ZERO;
    let mut bxh = ZERO;
    let mut byh = ZERO;
    let mut bzh = ZERO;

    // Sensitivity accumulators (one per layer)
    let mut dexhdsig = vec![ZERO; n];
    let mut deyhdsig = vec![ZERO; n];
    let mut djzhdsig = vec![ZERO; n];
    let mut dbxhdsig = vec![ZERO; n];
    let mut dbyhdsig = vec![ZERO; n];
    let mut dbzhdsig = vec![ZERO; n];

    let k2: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(0.0, -omega * MU0 * model.conductivities[i]))
        .collect();

    let h: Vec<f64> = (0..n).map(|i| model.thickness(i)).collect();

    let z_rx_top = model.depths[rx_layer];
    let depth_rx = z_rx - z_rx_top;
    let height_rx = if rx_layer + 1 < n {
        model.depths[rx_layer + 1] - z_rx
    } else {
        f64::INFINITY
    };

    let sigma_rx = Complex64::new(model.conductivities[rx_layer], 0.0);

    let same_layer = tx_layer == rx_layer;
    let isgnsrc: f64 = if same_layer { 1.0 } else { 0.0 };
    let isgndsrcdz: f64 = if same_layer {
        if z_rx > z_tx {
            -1.0
        } else if z_rx < z_tx {
            1.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    let sdm = 1.0;
    let iomega = Complex64::new(0.0, omega);
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    // Hankel transform integration
    for fi in 0..n_filter {
        let lam = filter_coeffs.base[fi] / r;
        let lam2 = lam * lam;
        let lam3 = lam2 * lam;

        buf.clear();
        compute_gamma_expmgh(model, &k2, &h, lam2, &mut buf.gamma, &mut buf.expmgh);

        // Compute merged TM+TE recursion coefficients WITH derivatives
        compute_tm_te_recursion_with_derivs_buf(model, &h, omega, tx_layer, buf);

        // Compute potential coefficients WITH derivatives
        compute_hed_coefficients_with_derivs_buf(
            model, &h, tx_layer, rx_layer, z_tx, lam, sdm, omega, buf,
        );

        let aa = buf.a[rx_layer];
        let bb = buf.b[rx_layer];
        let cc = buf.c[rx_layer];
        let dd = buf.d[rx_layer];

        let gg = buf.gamma[rx_layer];
        let gg2 = gg * gg;

        let expcoef1 = if height_rx.is_finite() {
            (-gg * height_rx).exp()
        } else {
            ZERO
        };
        let expcoef2 = (-gg * depth_rx).exp();

        let expsrc1 = (-buf.gamma[tx_layer] * (z_rx - z_tx).abs()).exp();

        let ae1 = aa * expcoef1;
        let be2 = bb * expcoef2;
        let ae1pbe2 = ae1 + be2;
        let ae1mbe2 = ae1 - be2;
        let ce1 = cc * expcoef1;
        let de2 = dd * expcoef2;

        let srcterm = isgnsrc * MU0 * sdm / 2.0 / buf.gamma[tx_layer] * expsrc1;
        let dzsrcterm = isgndsrcdz * MU0 * sdm / 2.0 * expsrc1;

        let ay = ae1pbe2 + srcterm;
        let daydz = gg * ae1mbe2 + dzsrcterm;
        let az = ce1 + de2 - gg * ae1mbe2 / lam2;

        let aypdazdz = gg * (ce1 - de2) - ae1pbe2 * k2[rx_layer] / lam2 + srcterm;
        let daydzpdazdz2 =
            gg2 * (ce1 + de2) - gg * k2[rx_layer] * ae1mbe2 / lam2 + dzsrcterm;

        let htj0 = filter_coeffs.j0[fi];
        let htj1 = filter_coeffs.j1[fi];

        // Forward field kernels
        let exj0 = -aypdazdz * lam3;
        let exj1 = aypdazdz * 2.0 * lam2 / r;
        exh += exj0 * htj0 + exj1 * htj1;

        let eyj0 = iomega * ay * lam / (2.0 * PI)
            - ONE / (2.0 * PI * MU0 * sigma_rx) * aypdazdz * lam3 * sintheta * sintheta;
        let eyj1 = -ONE / (2.0 * PI * MU0 * sigma_rx)
            * aypdazdz * lam2 * cos2theta / r;
        eyh += eyj0 * htj0 + eyj1 * htj1;

        let jzj1 = (sigma_rx * iomega * az + daydzpdazdz2 / MU0) * lam2;
        jzh += jzj1 * htj1;

        if compute_b {
            let bxj0 = -lam * daydz - az * lam3 * sintheta * sintheta;
            let bxj1 = -az * lam2 * cos2theta / r;
            bxh += bxj0 * htj0 + bxj1 * htj1;

            let byj0 = az * lam3;
            let byj1 = -az * 2.0 * lam2 / r;
            byh += byj0 * htj0 + byj1 * htj1;

            let bzj1 = ay * lam2;
            bzh += bzj1 * htj1;
        }

        // ================================================================
        // Potential derivatives at receiver (Fortran hed_pot_nlayer 3510-3585)
        // Using buf.dae1dsig etc. from WorkBuffers
        // ================================================================
        let dgammadsig_rx = -iomega_mu0 / (Complex64::new(2.0, 0.0) * gg);

        // dae1/dsig, dbe2/dsig
        for k in 0..n {
            buf.dae1dsig[k] = buf.dadsig[k * n + rx_layer] * expcoef1;
            buf.dbe2dsig[k] = buf.dbdsig[k * n + rx_layer] * expcoef2;
        }
        if height_rx.is_finite() {
            buf.dae1dsig[rx_layer] -= ae1 * height_rx * dgammadsig_rx;
        }
        buf.dbe2dsig[rx_layer] -= be2 * depth_rx * dgammadsig_rx;

        // dAy/dsig
        for k in 0..n {
            buf.day_dsig[k] = buf.dae1dsig[k] + buf.dbe2dsig[k];
        }
        let srcterm_dsig = -isgnsrc * MU0 * sdm / 2.0 * expsrc1
            * (ONE / gg2 + (z_rx - z_tx).abs() / gg)
            * dgammadsig_rx;
        buf.day_dsig[rx_layer] += srcterm_dsig;

        // dAy/dsig/dz
        for k in 0..n {
            buf.daydsigdz[k] = gg * (buf.dae1dsig[k] - buf.dbe2dsig[k]);
        }
        buf.daydsigdz[rx_layer] += dgammadsig_rx * ae1mbe2;
        let dzsrcterm_dsig =
            -isgndsrcdz * MU0 * sdm / 2.0 * expsrc1 * dgammadsig_rx * (z_rx - z_tx).abs();
        buf.daydsigdz[rx_layer] += dzsrcterm_dsig;

        // dce1/dsig, dde2/dsig
        for k in 0..n {
            buf.dce1dsig[k] = buf.dcdsig[k * n + rx_layer] * expcoef1;
            buf.dde2dsig[k] = buf.dddsig[k * n + rx_layer] * expcoef2;
        }
        if height_rx.is_finite() {
            buf.dce1dsig[rx_layer] -= ce1 * height_rx * dgammadsig_rx;
        }
        buf.dde2dsig[rx_layer] -= de2 * depth_rx * dgammadsig_rx;

        // dAz/dsig
        for k in 0..n {
            buf.daz_dsig[k] =
                buf.dce1dsig[k] + buf.dde2dsig[k] - gg / lam2 * (buf.dae1dsig[k] - buf.dbe2dsig[k]);
        }
        buf.daz_dsig[rx_layer] -= dgammadsig_rx / lam2 * ae1mbe2;

        // dAz/dsig/dz
        for k in 0..n {
            buf.dazdsigdz[k] = gg * (buf.dce1dsig[k] - buf.dde2dsig[k])
                - gg2 / lam2 * (buf.dae1dsig[k] + buf.dbe2dsig[k]);
        }
        buf.dazdsigdz[rx_layer] += dgammadsig_rx * (ce1 - de2)
            - ae1pbe2 * (-iomega_mu0 / lam2);

        // dAz/dsig/dz^2
        for k in 0..n {
            buf.dazdsigdz2[k] = gg2 * buf.daz_dsig[k];
        }
        buf.dazdsigdz2[rx_layer] += Complex64::new(2.0, 0.0) * gg * dgammadsig_rx * az;

        // Combined potential derivatives
        for k in 0..n {
            buf.daypdazdsigdz[k] = buf.day_dsig[k] + buf.dazdsigdz[k];
        }

        for k in 0..n {
            buf.daydsigdzpdazdsigdz2[k] = buf.daydsigdz[k] + buf.dazdsigdz2[k];
        }

        // ================================================================
        // Field kernel derivatives (Fortran lines 2478-2813)
        // ================================================================
        for k in 0..n {
            // dEx kernels
            let dfdsigj0 = -buf.daypdazdsigdz[k] * lam3;
            let dfdsigj1 = buf.daypdazdsigdz[k] * 2.0 * lam2 / r;
            dexhdsig[k] += dfdsigj0 * htj0 + dfdsigj1 * htj1;

            // dEy kernels
            let mut deyj0 = iomega * buf.day_dsig[k] * lam / (2.0 * PI)
                - ONE / (2.0 * PI * MU0 * sigma_rx)
                    * buf.daypdazdsigdz[k] * lam3 * sintheta * sintheta;
            let mut deyj1 = -ONE / (2.0 * PI * MU0 * sigma_rx)
                * buf.daypdazdsigdz[k] * lam2 * cos2theta / r;
            if k == rx_layer {
                deyj0 += ONE / (2.0 * PI * MU0 * sigma_rx * sigma_rx)
                    * aypdazdz * lam3 * sintheta * sintheta;
                deyj1 += ONE / (2.0 * PI * MU0 * sigma_rx * sigma_rx)
                    * aypdazdz * lam2 * cos2theta / r;
            }
            deyhdsig[k] += deyj0 * htj0 + deyj1 * htj1;

            // dJz kernel
            let mut djzj1 = (sigma_rx * iomega * buf.daz_dsig[k]
                + buf.daydsigdzpdazdsigdz2[k] / MU0) * lam2;
            if k == rx_layer {
                djzj1 += iomega * az * lam2;
            }
            djzhdsig[k] += djzj1 * htj1;

            if compute_b {
                // dBx kernels
                let dbxj0 =
                    -lam * buf.daydsigdz[k] - buf.daz_dsig[k] * lam3 * sintheta * sintheta;
                let dbxj1 = -buf.daz_dsig[k] * lam2 * cos2theta / r;
                dbxhdsig[k] += dbxj0 * htj0 + dbxj1 * htj1;

                // dBy kernels
                let dbyj0 = buf.daz_dsig[k] * lam3;
                let dbyj1 = -buf.daz_dsig[k] * 2.0 * lam2 / r;
                dbyhdsig[k] += dbyj0 * htj0 + dbyj1 * htj1;

                // dBz kernel
                let dbzj1 = buf.day_dsig[k] * lam2;
                dbzhdsig[k] += dbzj1 * htj1;
            }
        }
    }

    // ================================================================
    // Post-processing
    // ================================================================
    let ex = sin2theta / (4.0 * PI * MU0 * sigma_rx) * exh / r;
    let ey = eyh / r;
    let jz = -sintheta / (2.0 * PI) * jzh / r;
    let ez = jz / sigma_rx;

    let bx = bxh / r / (2.0 * PI);
    let by = sin2theta / (4.0 * PI) * byh / r;
    let bz = -costheta / (2.0 * PI) * bzh / r;

    let mut sens = EMFieldSensitivity::zeros(n);

    for k in 0..n {
        sens.dex_dsigma[k] = sin2theta / (4.0 * PI * MU0 * sigma_rx) * dexhdsig[k] / r;
        sens.dey_dsigma[k] = deyhdsig[k] / r;
        sens.dez_dsigma[k] = -sintheta / (2.0 * PI) * djzhdsig[k] / r / sigma_rx;

        if compute_b {
            sens.dbx_dsigma[k] = dbxhdsig[k] / r / (2.0 * PI);
            sens.dby_dsigma[k] = sin2theta / (4.0 * PI) * dbyhdsig[k] / r;
            sens.dbz_dsigma[k] = -costheta / (2.0 * PI) * dbzhdsig[k] / r;
        }
    }

    sens.dex_dsigma[rx_layer] -= ex / sigma_rx;
    sens.dez_dsigma[rx_layer] -= ez / sigma_rx;

    let fields = EMFields { ex, ey, ez, bx, by, bz };

    Ok((fields, vec![sens]))
}

/// Compute VED fields with analytical sensitivity using pre-allocated WorkBuffers.
pub fn compute_ved_fields_with_sensitivity(
    model: &LayeredEarth,
    _coeffs: &LayerCoefficients,
    filter: &HankelFilter,
    z_tx: f64,
    z_rx: f64,
    tx_layer: usize,
    rx_layer: usize,
    omega: f64,
    rho: f64,
    compute_b: bool,
    buf: &mut WorkBuffers,
) -> Result<(EMFields, EMFieldSensitivity)> {
    let filter_coeffs = filter.coefficients();
    let n_filter = filter_coeffs.base.len();
    let n = model.n_layers();

    let r = if rho < MIN_RHO { MIN_RHO } else { rho };

    // Field accumulators
    let mut exh = ZERO;
    let mut jzh = ZERO;
    let mut byh = ZERO;

    // Sensitivity accumulators
    let mut dexhdsig = vec![ZERO; n];
    let mut djzhdsig = vec![ZERO; n];
    let mut dbyhdsig = vec![ZERO; n];

    let k2: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(0.0, -omega * MU0 * model.conductivities[i]))
        .collect();

    let h: Vec<f64> = (0..n).map(|i| model.thickness(i)).collect();

    let sigma_rx = Complex64::new(model.conductivities[rx_layer], 0.0);
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    let same_layer = rx_layer == tx_layer;

    let z_rx_top = model.depths[rx_layer];
    let depth_rx = z_rx - z_rx_top;
    let height_rx = if rx_layer + 1 < n {
        model.depths[rx_layer + 1] - z_rx
    } else {
        f64::INFINITY
    };

    for fi in 0..n_filter {
        let lam = filter_coeffs.base[fi] / r;
        let lam2 = lam * lam;

        buf.clear();
        compute_gamma_expmgh(model, &k2, &h, lam2, &mut buf.gamma, &mut buf.expmgh);

        // TM recursion with derivatives (only TM for VED)
        compute_tm_recursion_with_derivs_buf(model, &h, omega, tx_layer, buf);

        let gg = buf.gamma[rx_layer];
        let dgammadsig_rx = -iomega_mu0 / (Complex64::new(2.0, 0.0) * gg);

        // VED coefficient computation
        let src_norm = MU0 / 2.0 / buf.gamma[tx_layer];
        let (cc, dd) = if same_layer {
            let rmrp = buf.rm[tx_layer] * buf.rp[tx_layer];
            let denom = ONE - rmrp;
            let cc = src_norm * (ONE + buf.rm[tx_layer]) / denom;
            let dd = src_norm * buf.rp[tx_layer] / denom;
            (cc, dd)
        } else {
            (ZERO, ZERO)
        };

        // Coefficient derivatives
        // Reuse buf.dccdsig and buf.ddddsig
        if same_layer {
            let tl = tx_layer;
            let dgammadsig_tx = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[tl]);
            let dsrc_norm_dsig_tx = -src_norm / buf.gamma[tl] * dgammadsig_tx;

            let rmrp = buf.rm[tl] * buf.rp[tl];
            let denom = ONE - rmrp;
            let numer_cc = ONE + buf.rm[tl];
            let numer_dd = buf.rp[tl];

            for k in 0..n {
                let drmk = buf.drmdsig[k * n + tl];
                let drpk = buf.drpdsig[k * n + tl];
                let ddenom = -(drmk * buf.rp[tl] + buf.rm[tl] * drpk);

                let dnumer_cc = drmk;
                let dnumer_dd = drpk;

                let dsrc_norm_k = if k == tl { dsrc_norm_dsig_tx } else { ZERO };

                buf.dccdsig[k] = dsrc_norm_k * numer_cc / denom
                    + src_norm * (dnumer_cc * denom - numer_cc * ddenom) / (denom * denom);

                buf.ddddsig[k] = dsrc_norm_k * numer_dd / denom
                    + src_norm * (dnumer_dd * denom - numer_dd * ddenom) / (denom * denom);
            }
        }

        // Receiver-level potential derivatives
        let expcoef1 = if height_rx.is_finite() {
            (-gg * height_rx).exp()
        } else {
            ZERO
        };
        let expcoef2 = (-gg * depth_rx).exp();

        let ce1 = cc * expcoef1;
        let de2 = dd * expcoef2;
        let az = ce1 + de2;
        let dazdz = gg * (ce1 - de2);

        let htj0 = filter_coeffs.j0[fi];
        let htj1 = filter_coeffs.j1[fi];

        // Forward field kernels
        let exh_kernel = lam * dazdz / (MU0 * sigma_rx);
        let jzh_kernel = lam2 * az / (MU0 * sigma_rx);
        exh += exh_kernel * htj1;
        jzh += jzh_kernel * htj0;
        if compute_b {
            byh += (-lam * az) * htj1;
        }

        // Potential derivatives at receiver
        for k in 0..n {
            buf.dce1dsig[k] = buf.dccdsig[k] * expcoef1;
            buf.dde2dsig[k] = buf.ddddsig[k] * expcoef2;
        }
        if height_rx.is_finite() {
            buf.dce1dsig[rx_layer] -= ce1 * height_rx * dgammadsig_rx;
        }
        buf.dde2dsig[rx_layer] -= de2 * depth_rx * dgammadsig_rx;

        // dAz/dsig, dAz_dz/dsig
        for k in 0..n {
            buf.daz_dsig[k] = buf.dce1dsig[k] + buf.dde2dsig[k];
            buf.ddazdzdsig[k] = gg * (buf.dce1dsig[k] - buf.dde2dsig[k]);
        }
        buf.ddazdzdsig[rx_layer] += dgammadsig_rx * (ce1 - de2);

        // Field kernel derivatives
        for k in 0..n {
            let mut dexk = lam * buf.ddazdzdsig[k] / (MU0 * sigma_rx);
            if k == rx_layer {
                dexk -= exh_kernel / sigma_rx;
            }
            dexhdsig[k] += dexk * htj1;

            let mut djzk = lam2 * buf.daz_dsig[k] / (MU0 * sigma_rx);
            if k == rx_layer {
                djzk -= jzh_kernel / sigma_rx;
            }
            djzhdsig[k] += djzk * htj0;

            if compute_b {
                dbyhdsig[k] += (-lam * buf.daz_dsig[k]) * htj1;
            }
        }
    }

    // Post-processing
    let ex = exh / r / (2.0 * PI);
    let ez = jzh / (2.0 * PI);
    let by = byh / r / (2.0 * PI);

    let mut sens = EMFieldSensitivity::zeros(n);
    for k in 0..n {
        sens.dex_dsigma[k] = dexhdsig[k] / r / (2.0 * PI);
        sens.dez_dsigma[k] = djzhdsig[k] / (2.0 * PI);
        if compute_b {
            sens.dby_dsigma[k] = dbyhdsig[k] / r / (2.0 * PI);
        }
    }

    let fields = EMFields {
        ex,
        ey: ZERO,
        ez,
        bx: ZERO,
        by,
        bz: ZERO,
    };

    Ok((fields, sens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resistivity_conversion() {
        let mut sens = EMFieldSensitivity::zeros(2);
        sens.dex_dsigma[0] = Complex64::new(1.0, 0.0);
        sens.dex_dsigma[1] = Complex64::new(2.0, 0.0);

        let conductivities = vec![1.0, 0.5];
        let rho_sens = sens.to_resistivity_sensitivity(&conductivities);

        assert!((rho_sens.dex_dsigma[0].re - (-1.0)).abs() < 1e-10);
        assert!((rho_sens.dex_dsigma[1].re - (-0.5)).abs() < 1e-10);
    }
}
