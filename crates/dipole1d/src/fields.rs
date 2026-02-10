//! Electric and magnetic field computation
//!
//! Implements the Hankel transform integration for computing EM fields
//! Based on the formulation in Dipole1D.f90 by Kerry Key, Scripps Institution of Oceanography

use crate::constants::MU0;
use crate::error::Result;
use crate::filters::HankelFilter;
use crate::layers::LayerCoefficients;
use crate::model::LayeredEarth;
use crate::work_buffers::{ForwardBuffers, WorkBuffers};
use crate::EMFields;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Minimum horizontal distance (m) to avoid singularity
/// From Dipole1D.f90: "Move r to 1 so that division by r doesn't explode"
const MIN_RHO: f64 = 1.0;
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Compute HED (Horizontal Electric Dipole) fields using Hankel transform
///
/// Fortran reference: Dipole1D.f90
/// - Potential computation: hed_pot_nlayer (lines 3391-3587)
/// - Coefficient recursion: hed_coef_nlayer (lines 2818-3386)
/// - Field kernels: ex_hed_j0/j1 (lines 2460-2527), ey_hed_j0/j1 (lines 2529-2595),
///   jz_hed_j1 (lines 2597-2629), bx_hed_j0/j1 (lines 2631-2697),
///   by_hed_j0/j1 (lines 2699-2765), bz_hed_j1 (lines 2767-2799)
/// - Post-processing normalization: lines 1414-1421
///
/// Arguments:
/// - theta: Angle (radians) from the crossline direction (perpendicular to dipole) to the receiver.
///          theta=0 means receiver is broadside (perpendicular to dipole)
///          theta=90° means receiver is inline (along dipole direction)
pub fn compute_hed_fields(
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
) -> Result<EMFields> {
    let filter_coeffs = filter.coefficients();
    let n_filter = filter_coeffs.base.len();
    let n = model.n_layers();

    let r = if rho < MIN_RHO { MIN_RHO } else { rho };

    let sintheta = theta.sin();
    let costheta = theta.cos();
    let sin2theta = (2.0 * theta).sin();
    let cos2theta = (2.0 * theta).cos();

    let mut exh = ZERO;
    let mut eyh = ZERO;
    let mut jzh = ZERO;
    let mut bxh = ZERO;
    let mut byh = ZERO;
    let mut bzh = ZERO;

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

    let mut buf = ForwardBuffers::new(n);

    for i in 0..n_filter {
        let lam = filter_coeffs.base[i] / r;
        let lam2 = lam * lam;
        let lam3 = lam2 * lam;

        buf.clear();
        compute_gamma_expmgh(model, &k2, &h, lam2, &mut buf.gamma, &mut buf.expmgh);
        compute_tm_te_recursion_buf(model, &buf.gamma, &buf.expmgh, omega, tx_layer,
            &mut buf.rm, &mut buf.rp, &mut buf.sm, &mut buf.sp);
        compute_hed_coefficients_buf(
            model, &buf.gamma, &buf.expmgh, &buf.rm, &buf.rp, &buf.sm, &buf.sp,
            tx_layer, rx_layer, z_tx, lam, sdm,
            &mut buf.a, &mut buf.b, &mut buf.c, &mut buf.d,
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
        let daydzpdazdz2 = gg2 * (ce1 + de2) - gg * k2[rx_layer] * ae1mbe2 / lam2 + dzsrcterm;

        let htj0 = filter_coeffs.j0[i];
        let htj1 = filter_coeffs.j1[i];

        let exj0 = -aypdazdz * lam3;
        let exj1 = aypdazdz * 2.0 * lam2 / r;
        exh += exj0 * htj0 + exj1 * htj1;

        let eyj0 = iomega * ay * lam / (2.0 * PI)
            - ONE / (2.0 * PI * MU0 * sigma_rx) * aypdazdz * lam3 * sintheta * sintheta;
        let eyj1 = -ONE / (2.0 * PI * MU0 * sigma_rx) * aypdazdz * lam2 * cos2theta / r;
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
    }

    let ex = sin2theta / (4.0 * PI * MU0 * sigma_rx) * exh / r;
    let ey = eyh / r;
    let jz = -sintheta / (2.0 * PI) * jzh / r;
    let ez = jz / sigma_rx;

    let bx = bxh / r / (2.0 * PI);
    let by = sin2theta / (4.0 * PI) * byh / r;
    let bz = -costheta / (2.0 * PI) * bzh / r;

    Ok(EMFields { ex, ey, ez, bx, by, bz })
}

/// Compute VED (Vertical Electric Dipole) fields using Hankel transform
pub fn compute_ved_fields(
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
) -> Result<EMFields> {
    let filter_coeffs = filter.coefficients();
    let n_filter = filter_coeffs.base.len();
    let n = model.n_layers();

    let r = if rho < MIN_RHO { MIN_RHO } else { rho };

    let mut exh = ZERO;
    let mut jzh = ZERO;
    let mut byh = ZERO;

    let k2: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(0.0, -omega * MU0 * model.conductivities[i]))
        .collect();

    let h: Vec<f64> = (0..n).map(|i| model.thickness(i)).collect();

    let sigma_rx = Complex64::new(model.conductivities[rx_layer], 0.0);

    let mut buf = ForwardBuffers::new(n);

    for i in 0..n_filter {
        let lam = filter_coeffs.base[i] / r;
        let lam2 = lam * lam;

        buf.clear();
        compute_gamma_expmgh(model, &k2, &h, lam2, &mut buf.gamma, &mut buf.expmgh);
        compute_tm_recursion_buf(model, &buf.gamma, &buf.expmgh, omega, tx_layer, &mut buf.rm, &mut buf.rp);

        // Simplified VED coefficient computation
        let src_norm = MU0 / 2.0 / buf.gamma[tx_layer];

        let (cc, dd) = if rx_layer == tx_layer {
            let z_top = model.depths[tx_layer];
            let z_bot = if tx_layer + 1 < n {
                model.depths[tx_layer + 1]
            } else {
                f64::INFINITY
            };

            let depth_tx = z_tx - z_top;
            let height_tx = if z_bot.is_finite() { z_bot - z_tx } else { f64::INFINITY };

            let _exp_up = (-buf.gamma[tx_layer] * depth_tx).exp();
            let _exp_down = if height_tx.is_finite() {
                (-buf.gamma[tx_layer] * height_tx).exp()
            } else {
                ZERO
            };

            let rmrp = buf.rm[tx_layer] * buf.rp[tx_layer];
            let denom = ONE - rmrp;

            let cc = src_norm * (ONE + buf.rm[tx_layer]) / denom;
            let dd = src_norm * buf.rp[tx_layer] / denom;

            (cc, dd)
        } else {
            (ZERO, ZERO)
        };

        let gg = buf.gamma[rx_layer];

        let z_rx_top = model.depths[rx_layer];
        let depth_rx = z_rx - z_rx_top;
        let height_rx = if rx_layer + 1 < n {
            model.depths[rx_layer + 1] - z_rx
        } else {
            f64::INFINITY
        };

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

        let htj0 = filter_coeffs.j0[i];
        let htj1 = filter_coeffs.j1[i];

        exh += (lam * dazdz / (MU0 * sigma_rx)) * htj1;
        jzh += (lam2 * az / (MU0 * sigma_rx)) * htj0;

        if compute_b {
            byh += (-lam * az) * htj1;
        }
    }

    let ex = exh / r / (2.0 * PI);
    let ez = jzh / (2.0 * PI);
    let by = byh / r / (2.0 * PI);

    Ok(EMFields {
        ex,
        ey: ZERO,
        ez,
        bx: ZERO,
        by,
        bz: ZERO,
    })
}

// ============================================================================
// Shared helper: compute gamma and expmgh into pre-allocated slices
// ============================================================================

/// Compute gamma (vertical wavenumber) and exp(-gamma*h) for all layers into buffers.
#[inline]
pub fn compute_gamma_expmgh(
    model: &LayeredEarth,
    k2: &[Complex64],
    h: &[f64],
    lam2: f64,
    gamma: &mut [Complex64],
    expmgh: &mut [Complex64],
) {
    let lam2c = Complex64::new(lam2, 0.0);
    for j in 0..model.n_layers() {
        gamma[j] = complex_sqrt(lam2c + k2[j]);
        expmgh[j] = if h[j].is_finite() {
            (-gamma[j] * h[j]).exp()
        } else {
            ZERO
        };
    }
}

// ============================================================================
// Buffer-based recursion functions (no allocation)
// ============================================================================

/// Compute TM recursion into pre-allocated rm, rp slices.
#[inline]
pub fn compute_tm_recursion_buf(
    model: &LayeredEarth,
    gamma: &[Complex64],
    expmgh: &[Complex64],
    omega: f64,
    tx_layer: usize,
    rm: &mut [Complex64],
    rp: &mut [Complex64],
) {
    let n = model.n_layers();
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    for i in 1..=tx_layer {
        rm[i - 1] *= expmgh[i - 1];
        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_im1 = Complex64::new(model.conductivities[i - 1], 0.0);
        let gmogp = -iomega_mu0 * (sigma_i - sigma_im1) / (gamma[i] + gamma[i - 1]).powi(2);
        let rjexp = rm[i - 1] * expmgh[i - 1];
        rm[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);
    }

    for i in (tx_layer..n - 1).rev() {
        rp[i + 1] *= expmgh[i + 1];
        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_ip1 = Complex64::new(model.conductivities[i + 1], 0.0);
        let gmogp = -iomega_mu0 * (sigma_i - sigma_ip1) / (gamma[i] + gamma[i + 1]).powi(2);
        let rjexp = rp[i + 1] * expmgh[i + 1];
        rp[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);
    }
}

/// Merged TM+TE recursion for forward-only HED path.
/// Computes rm, rp, sm, sp in a single upward + downward pass for better cache reuse.
#[inline]
pub fn compute_tm_te_recursion_buf(
    model: &LayeredEarth,
    gamma: &[Complex64],
    expmgh: &[Complex64],
    omega: f64,
    tx_layer: usize,
    rm: &mut [Complex64],
    rp: &mut [Complex64],
    sm: &mut [Complex64],
    sp: &mut [Complex64],
) {
    let n = model.n_layers();
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    // Upward pass: Rm + Sm together
    for i in 1..=tx_layer {
        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_im1 = Complex64::new(model.conductivities[i - 1], 0.0);
        let gsum = gamma[i] + gamma[i - 1];

        // TM: Rm
        rm[i - 1] *= expmgh[i - 1];
        let gmogp = -iomega_mu0 * (sigma_i - sigma_im1) / gsum.powi(2);
        let rjexp = rm[i - 1] * expmgh[i - 1];
        rm[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);

        // TE: Sm
        sm[i - 1] *= expmgh[i - 1];
        let sgmosgp = (gamma[i] * sigma_im1 - gamma[i - 1] * sigma_i)
            / (gamma[i] * sigma_im1 + gamma[i - 1] * sigma_i);
        let sjexp = sm[i - 1] * expmgh[i - 1];
        sm[i] = (sgmosgp + sjexp) / (ONE + sgmosgp * sjexp);
    }

    // Downward pass: Rp + Sp together
    for i in (tx_layer..n - 1).rev() {
        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_ip1 = Complex64::new(model.conductivities[i + 1], 0.0);
        let gsum = gamma[i] + gamma[i + 1];

        // TM: Rp
        rp[i + 1] *= expmgh[i + 1];
        let gmogp = -iomega_mu0 * (sigma_i - sigma_ip1) / gsum.powi(2);
        let rjexp = rp[i + 1] * expmgh[i + 1];
        rp[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);

        // TE: Sp
        sp[i + 1] *= expmgh[i + 1];
        let sgmosgp = (gamma[i] * sigma_ip1 - gamma[i + 1] * sigma_i)
            / (gamma[i] * sigma_ip1 + gamma[i + 1] * sigma_i);
        let sjexp = sp[i + 1] * expmgh[i + 1];
        sp[i] = (sgmosgp + sjexp) / (ONE + sgmosgp * sjexp);
    }
}

/// Compute HED coefficients into pre-allocated a, b, c, d slices.
#[inline]
pub fn compute_hed_coefficients_buf(
    model: &LayeredEarth,
    gamma: &[Complex64],
    expmgh: &[Complex64],
    rm: &[Complex64],
    rp: &[Complex64],
    sm: &[Complex64],
    sp: &[Complex64],
    tx_layer: usize,
    rx_layer: usize,
    z_tx: f64,
    lam: f64,
    sdm: f64,
    a: &mut [Complex64],
    b: &mut [Complex64],
    c: &mut [Complex64],
    d: &mut [Complex64],
) {
    let n = model.n_layers();

    let z_top = model.depths[tx_layer];
    let z_bot = if tx_layer + 1 < n {
        model.depths[tx_layer + 1]
    } else {
        f64::INFINITY
    };

    let depth_tx = z_tx - z_top;
    let height_tx = if z_bot.is_finite() { z_bot - z_tx } else { f64::INFINITY };

    let gamma_tx = gamma[tx_layer];
    let expmgh_tx = expmgh[tx_layer];

    // Ay potential coefficients
    let rmrp = rm[tx_layer] * rp[tx_layer];
    let onemrmrp_ay = ONE - rmrp * expmgh_tx * expmgh_tx;

    let rhs_ay = MU0 * sdm / 2.0 / gamma_tx;
    let srcp_ay = if height_tx.is_finite() {
        (-gamma_tx * height_tx).exp() * rhs_ay
    } else {
        ZERO
    };
    let srcm_ay = (-gamma_tx * depth_tx).exp() * rhs_ay;

    a[tx_layer] = (rmrp * srcm_ay * expmgh_tx + rp[tx_layer] * srcp_ay) / onemrmrp_ay;
    b[tx_layer] = (rmrp * srcp_ay * expmgh_tx + rm[tx_layer] * srcm_ay) / onemrmrp_ay;

    // Propagate above source — short-circuit to rx_layer
    if rx_layer < tx_layer {
        let mut src = srcm_ay;
        for i in (rx_layer..tx_layer).rev() {
            let aytop = a[i + 1] * expmgh[i + 1] + b[i + 1] + src;
            a[i] = aytop / (ONE + rm[i] * expmgh[i]);
            b[i] = a[i] * rm[i];
            src = ZERO;
        }
    }

    // Propagate below source — short-circuit to rx_layer
    if rx_layer > tx_layer {
        let mut src = srcp_ay;
        for i in (tx_layer + 1)..=rx_layer {
            let aytop = a[i - 1] + b[i - 1] * expmgh[i - 1] + src;
            b[i] = aytop / (ONE + rp[i] * expmgh[i]);
            a[i] = b[i] * rp[i];
            src = ZERO;
        }
    }

    // Lambda_z potential coefficients
    let smsp = sm[tx_layer] * sp[tx_layer];
    let onemsmsp = ONE - smsp * expmgh_tx * expmgh_tx;

    let lam2 = lam * lam;
    let rhs_lz = MU0 * sdm / 2.0 / lam2;
    let srcp_lz = if height_tx.is_finite() {
        -(-gamma_tx * height_tx).exp() * rhs_lz
    } else {
        ZERO
    };
    let srcm_lz = (-gamma_tx * depth_tx).exp() * rhs_lz;

    c[tx_layer] = (smsp * srcm_lz * expmgh_tx + sp[tx_layer] * srcp_lz) / onemsmsp;
    d[tx_layer] = (smsp * srcp_lz * expmgh_tx + sm[tx_layer] * srcm_lz) / onemsmsp;

    // Propagate above source — short-circuit to rx_layer
    if rx_layer < tx_layer {
        let mut src = srcm_lz;
        for i in (rx_layer..tx_layer).rev() {
            let top = c[i + 1] * expmgh[i + 1] + d[i + 1] + src;
            c[i] = top / (ONE + sm[i] * expmgh[i]);
            d[i] = c[i] * sm[i];
            src = ZERO;
        }
    }

    // Propagate below source — short-circuit to rx_layer
    if rx_layer > tx_layer {
        let mut src = srcp_lz;
        for i in (tx_layer + 1)..=rx_layer {
            let top = c[i - 1] + d[i - 1] * expmgh[i - 1] + src;
            d[i] = top / (ONE + sp[i] * expmgh[i]);
            c[i] = d[i] * sp[i];
            src = ZERO;
        }
    }
}

// ============================================================================
// Buffer-based derivative functions (flat n*n matrices)
// ============================================================================

/// Safe multiplication of h * expmgh that avoids INFINITY * 0 = NaN
#[inline]
fn h_times_expmgh(h: f64, expmgh: Complex64) -> Complex64 {
    if h.is_finite() {
        Complex64::new(h, 0.0) * expmgh
    } else {
        ZERO
    }
}

/// Compute TM recursion with derivatives into WorkBuffers.
/// Writes buf.rm, buf.rp, buf.drmdsig, buf.drpdsig.
pub fn compute_tm_recursion_with_derivs_buf(
    model: &LayeredEarth,
    h: &[f64],
    omega: f64,
    tx_layer: usize,
    buf: &mut WorkBuffers,
) {
    let n = buf.n;
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    // Upward recursion for Rm
    for i in 1..=tx_layer {
        buf.rm[i - 1] *= buf.expmgh[i - 1];

        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_im1 = Complex64::new(model.conductivities[i - 1], 0.0);

        let gmogp = -iomega_mu0 * (sigma_i - sigma_im1) / (buf.gamma[i] + buf.gamma[i - 1]).powi(2);
        let rjexp = buf.rm[i - 1] * buf.expmgh[i - 1];
        buf.rm[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);

        let dgammadsig_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);
        let drdsig_i = Complex64::new(2.0, 0.0) * buf.gamma[i - 1]
            / (buf.gamma[i] + buf.gamma[i - 1]).powi(2)
            * dgammadsig_i;

        for k in 0..i {
            buf.drmdsig[k * n + i - 1] *= buf.expmgh[i - 1];
        }

        let h_dgamma_i = if h[i].is_finite() { h[i] * dgammadsig_i } else { ZERO };
        buf.drmdsig[i * n + i] = drdsig_i * (ONE - buf.rm[i] * rjexp) / (ONE + gmogp * rjexp)
            - buf.rm[i] * h_dgamma_i;

        let dgammadsig_im1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i - 1]);
        let drdsig_im1 = Complex64::new(-2.0, 0.0) * buf.gamma[i]
            / (buf.gamma[i] + buf.gamma[i - 1]).powi(2)
            * dgammadsig_im1;

        let db = buf.drmdsig[(i - 1) * n + i - 1] * buf.expmgh[i - 1]
            - if h[i - 1].is_finite() { rjexp * h[i - 1] * dgammadsig_im1 } else { ZERO };

        buf.drmdsig[(i - 1) * n + i] = (drdsig_im1 * (ONE - buf.rm[i] * rjexp)
            + db * (ONE - buf.rm[i] * gmogp))
            / (ONE + gmogp * rjexp);

        let chain_factor = (buf.expmgh[i - 1] * (ONE - buf.rm[i] * gmogp)) / (ONE + gmogp * rjexp);
        for k in 0..i.saturating_sub(1) {
            buf.drmdsig[k * n + i] = buf.drmdsig[k * n + i - 1] * chain_factor;
        }
    }

    // Downward recursion for Rp
    for i in (tx_layer..n - 1).rev() {
        buf.rp[i + 1] *= buf.expmgh[i + 1];

        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_ip1 = Complex64::new(model.conductivities[i + 1], 0.0);

        let gmogp = -iomega_mu0 * (sigma_i - sigma_ip1) / (buf.gamma[i] + buf.gamma[i + 1]).powi(2);
        let rjexp = buf.rp[i + 1] * buf.expmgh[i + 1];
        buf.rp[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);

        let dgammadsig_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);
        let drdsig_i = Complex64::new(2.0, 0.0) * buf.gamma[i + 1]
            / (buf.gamma[i] + buf.gamma[i + 1]).powi(2)
            * dgammadsig_i;

        for k in (i + 1)..n {
            buf.drpdsig[k * n + i + 1] *= buf.expmgh[i + 1];
        }

        let h_dgamma_i = if h[i].is_finite() { h[i] * dgammadsig_i } else { ZERO };
        buf.drpdsig[i * n + i] = drdsig_i * (ONE - buf.rp[i] * rjexp) / (ONE + gmogp * rjexp)
            - buf.rp[i] * h_dgamma_i;

        let dgammadsig_ip1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i + 1]);
        let drdsig_ip1 = Complex64::new(-2.0, 0.0) * buf.gamma[i]
            / (buf.gamma[i] + buf.gamma[i + 1]).powi(2)
            * dgammadsig_ip1;

        let db = buf.drpdsig[(i + 1) * n + i + 1] * buf.expmgh[i + 1]
            - if h[i + 1].is_finite() { rjexp * h[i + 1] * dgammadsig_ip1 } else { ZERO };

        buf.drpdsig[(i + 1) * n + i] = (drdsig_ip1 * (ONE - buf.rp[i] * rjexp)
            + db * (ONE - buf.rp[i] * gmogp))
            / (ONE + gmogp * rjexp);

        let chain_factor = (buf.expmgh[i + 1] * (ONE - buf.rp[i] * gmogp)) / (ONE + gmogp * rjexp);
        for k in (i + 2)..n {
            buf.drpdsig[k * n + i] = buf.drpdsig[k * n + i + 1] * chain_factor;
        }
    }
}

/// Merged TM+TE recursion with derivatives for HED sensitivity path.
/// Computes rm, rp, sm, sp and their derivatives in a single upward + downward pass.
pub fn compute_tm_te_recursion_with_derivs_buf(
    model: &LayeredEarth,
    h: &[f64],
    omega: f64,
    tx_layer: usize,
    buf: &mut WorkBuffers,
) {
    let n = buf.n;
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    // Upward pass: Rm + Sm together
    for i in 1..=tx_layer {
        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_im1 = Complex64::new(model.conductivities[i - 1], 0.0);
        let dgammadsig_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);
        let dgammadsig_im1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i - 1]);
        let h_dgamma_i = if h[i].is_finite() { h[i] * dgammadsig_i } else { ZERO };

        // === TM: Rm ===
        {
            buf.rm[i - 1] *= buf.expmgh[i - 1];
            let gmogp = -iomega_mu0 * (sigma_i - sigma_im1) / (buf.gamma[i] + buf.gamma[i - 1]).powi(2);
            let rjexp = buf.rm[i - 1] * buf.expmgh[i - 1];
            buf.rm[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);

            let drdsig_i = Complex64::new(2.0, 0.0) * buf.gamma[i - 1]
                / (buf.gamma[i] + buf.gamma[i - 1]).powi(2)
                * dgammadsig_i;

            for k in 0..i {
                buf.drmdsig[k * n + i - 1] *= buf.expmgh[i - 1];
            }

            buf.drmdsig[i * n + i] = drdsig_i * (ONE - buf.rm[i] * rjexp) / (ONE + gmogp * rjexp)
                - buf.rm[i] * h_dgamma_i;

            let drdsig_im1 = Complex64::new(-2.0, 0.0) * buf.gamma[i]
                / (buf.gamma[i] + buf.gamma[i - 1]).powi(2)
                * dgammadsig_im1;

            let db = buf.drmdsig[(i - 1) * n + i - 1] * buf.expmgh[i - 1]
                - if h[i - 1].is_finite() { rjexp * h[i - 1] * dgammadsig_im1 } else { ZERO };

            buf.drmdsig[(i - 1) * n + i] = (drdsig_im1 * (ONE - buf.rm[i] * rjexp)
                + db * (ONE - buf.rm[i] * gmogp))
                / (ONE + gmogp * rjexp);

            let chain_factor = (buf.expmgh[i - 1] * (ONE - buf.rm[i] * gmogp)) / (ONE + gmogp * rjexp);
            for k in 0..i.saturating_sub(1) {
                buf.drmdsig[k * n + i] = buf.drmdsig[k * n + i - 1] * chain_factor;
            }
        }

        // === TE: Sm ===
        {
            buf.sm[i - 1] *= buf.expmgh[i - 1];
            let sgmosgp = (buf.gamma[i] * sigma_im1 - buf.gamma[i - 1] * sigma_i)
                / (buf.gamma[i] * sigma_im1 + buf.gamma[i - 1] * sigma_i);
            let sjexp = buf.sm[i - 1] * buf.expmgh[i - 1];
            buf.sm[i] = (sgmosgp + sjexp) / (ONE + sgmosgp * sjexp);

            let denom_te = buf.gamma[i] * sigma_im1 + buf.gamma[i - 1] * sigma_i;
            let dsdsig_i = (dgammadsig_i * sigma_im1 * (ONE - sgmosgp)
                - buf.gamma[i - 1] * (ONE + sgmosgp))
                / denom_te;

            for k in 0..i {
                buf.dsmdsig[k * n + i - 1] *= buf.expmgh[i - 1];
            }

            buf.dsmdsig[i * n + i] = dsdsig_i * (ONE - buf.sm[i] * sjexp) / (ONE + sgmosgp * sjexp)
                - buf.sm[i] * h_dgamma_i;

            let dsdsig_im1 = (buf.gamma[i] * (ONE - sgmosgp)
                - dgammadsig_im1 * sigma_i * (ONE + sgmosgp))
                / denom_te;

            let db = buf.dsmdsig[(i - 1) * n + i - 1] * buf.expmgh[i - 1]
                - if h[i - 1].is_finite() { sjexp * h[i - 1] * dgammadsig_im1 } else { ZERO };

            buf.dsmdsig[(i - 1) * n + i] = (dsdsig_im1 * (ONE - buf.sm[i] * sjexp)
                + db * (ONE - buf.sm[i] * sgmosgp))
                / (ONE + sgmosgp * sjexp);

            let chain_factor = (buf.expmgh[i - 1] * (ONE - buf.sm[i] * sgmosgp)) / (ONE + sgmosgp * sjexp);
            for k in 0..i.saturating_sub(1) {
                buf.dsmdsig[k * n + i] = buf.dsmdsig[k * n + i - 1] * chain_factor;
            }
        }
    }

    // Downward pass: Rp + Sp together
    for i in (tx_layer..n - 1).rev() {
        let sigma_i = Complex64::new(model.conductivities[i], 0.0);
        let sigma_ip1 = Complex64::new(model.conductivities[i + 1], 0.0);
        let dgammadsig_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);
        let dgammadsig_ip1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i + 1]);
        let h_dgamma_i = if h[i].is_finite() { h[i] * dgammadsig_i } else { ZERO };

        // === TM: Rp ===
        {
            buf.rp[i + 1] *= buf.expmgh[i + 1];
            let gmogp = -iomega_mu0 * (sigma_i - sigma_ip1) / (buf.gamma[i] + buf.gamma[i + 1]).powi(2);
            let rjexp = buf.rp[i + 1] * buf.expmgh[i + 1];
            buf.rp[i] = (gmogp + rjexp) / (ONE + gmogp * rjexp);

            let drdsig_i = Complex64::new(2.0, 0.0) * buf.gamma[i + 1]
                / (buf.gamma[i] + buf.gamma[i + 1]).powi(2)
                * dgammadsig_i;

            for k in (i + 1)..n {
                buf.drpdsig[k * n + i + 1] *= buf.expmgh[i + 1];
            }

            buf.drpdsig[i * n + i] = drdsig_i * (ONE - buf.rp[i] * rjexp) / (ONE + gmogp * rjexp)
                - buf.rp[i] * h_dgamma_i;

            let drdsig_ip1 = Complex64::new(-2.0, 0.0) * buf.gamma[i]
                / (buf.gamma[i] + buf.gamma[i + 1]).powi(2)
                * dgammadsig_ip1;

            let db = buf.drpdsig[(i + 1) * n + i + 1] * buf.expmgh[i + 1]
                - if h[i + 1].is_finite() { rjexp * h[i + 1] * dgammadsig_ip1 } else { ZERO };

            buf.drpdsig[(i + 1) * n + i] = (drdsig_ip1 * (ONE - buf.rp[i] * rjexp)
                + db * (ONE - buf.rp[i] * gmogp))
                / (ONE + gmogp * rjexp);

            let chain_factor = (buf.expmgh[i + 1] * (ONE - buf.rp[i] * gmogp)) / (ONE + gmogp * rjexp);
            for k in (i + 2)..n {
                buf.drpdsig[k * n + i] = buf.drpdsig[k * n + i + 1] * chain_factor;
            }
        }

        // === TE: Sp ===
        {
            buf.sp[i + 1] *= buf.expmgh[i + 1];
            let sgmosgp = (buf.gamma[i] * sigma_ip1 - buf.gamma[i + 1] * sigma_i)
                / (buf.gamma[i] * sigma_ip1 + buf.gamma[i + 1] * sigma_i);
            let sjexp = buf.sp[i + 1] * buf.expmgh[i + 1];
            buf.sp[i] = (sgmosgp + sjexp) / (ONE + sgmosgp * sjexp);

            let denom_te = buf.gamma[i] * sigma_ip1 + buf.gamma[i + 1] * sigma_i;
            let dsdsig_i = (dgammadsig_i * sigma_ip1 * (ONE - sgmosgp)
                - buf.gamma[i + 1] * (ONE + sgmosgp))
                / denom_te;

            for k in (i + 1)..n {
                buf.dspdsig[k * n + i + 1] *= buf.expmgh[i + 1];
            }

            buf.dspdsig[i * n + i] = dsdsig_i * (ONE - buf.sp[i] * sjexp) / (ONE + sgmosgp * sjexp)
                - buf.sp[i] * h_dgamma_i;

            let dsdsig_ip1 = (buf.gamma[i] * (ONE - sgmosgp)
                - dgammadsig_ip1 * sigma_i * (ONE + sgmosgp))
                / denom_te;

            let db = buf.dspdsig[(i + 1) * n + i + 1] * buf.expmgh[i + 1]
                - if h[i + 1].is_finite() { sjexp * h[i + 1] * dgammadsig_ip1 } else { ZERO };

            buf.dspdsig[(i + 1) * n + i] = (dsdsig_ip1 * (ONE - buf.sp[i] * sjexp)
                + db * (ONE - buf.sp[i] * sgmosgp))
                / (ONE + sgmosgp * sjexp);

            let chain_factor = (buf.expmgh[i + 1] * (ONE - buf.sp[i] * sgmosgp)) / (ONE + sgmosgp * sjexp);
            for k in (i + 2)..n {
                buf.dspdsig[k * n + i] = buf.dspdsig[k * n + i + 1] * chain_factor;
            }
        }
    }
}

/// Compute HED potential coefficients with derivatives into WorkBuffers.
/// Writes buf.a, buf.b, buf.c, buf.d, buf.dadsig, buf.dbdsig, buf.dcdsig, buf.dddsig.
pub fn compute_hed_coefficients_with_derivs_buf(
    model: &LayeredEarth,
    h: &[f64],
    tx_layer: usize,
    rx_layer: usize,
    z_tx: f64,
    lam: f64,
    sdm: f64,
    omega: f64,
    buf: &mut WorkBuffers,
) {
    let n = buf.n;
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);

    let z_top = model.depths[tx_layer];
    let z_bot = if tx_layer + 1 < n {
        model.depths[tx_layer + 1]
    } else {
        f64::INFINITY
    };
    let depth_tx = z_tx - z_top;
    let height_tx = if z_bot.is_finite() { z_bot - z_tx } else { f64::INFINITY };

    let gamma_tx = buf.gamma[tx_layer];
    let expmgh_tx = buf.expmgh[tx_layer];

    // ===================================================================
    // Ay potential coefficients (a, b) using TM reflection (Rm, Rp)
    // ===================================================================
    let rmrp = buf.rm[tx_layer] * buf.rp[tx_layer];
    let onemrmrp_ay = ONE - rmrp * expmgh_tx * expmgh_tx;

    let rhs_ay = MU0 * sdm / 2.0 / gamma_tx;
    let srcp_ay = if height_tx.is_finite() {
        (-gamma_tx * height_tx).exp() * rhs_ay
    } else {
        ZERO
    };
    let srcm_ay = (-gamma_tx * depth_tx).exp() * rhs_ay;

    buf.a[tx_layer] = (rmrp * srcm_ay * expmgh_tx + buf.rp[tx_layer] * srcp_ay) / onemrmrp_ay;
    buf.b[tx_layer] = (rmrp * srcp_ay * expmgh_tx + buf.rm[tx_layer] * srcm_ay) / onemrmrp_ay;

    // Propagate Ay above source layer — short-circuit to rx_layer
    if rx_layer < tx_layer {
        let mut src = srcm_ay;
        for i in (rx_layer..tx_layer).rev() {
            let aytop = buf.a[i + 1] * buf.expmgh[i + 1] + buf.b[i + 1] + src;
            buf.a[i] = aytop / (ONE + buf.rm[i] * buf.expmgh[i]);
            buf.b[i] = buf.a[i] * buf.rm[i];
            src = ZERO;
        }
    }

    // Propagate Ay below source layer — short-circuit to rx_layer
    if rx_layer > tx_layer {
        let mut src = srcp_ay;
        for i in (tx_layer + 1)..=rx_layer {
            let aytop = buf.a[i - 1] + buf.b[i - 1] * buf.expmgh[i - 1] + src;
            buf.b[i] = aytop / (ONE + buf.rp[i] * buf.expmgh[i]);
            buf.a[i] = buf.b[i] * buf.rp[i];
            src = ZERO;
        }
    }

    // ===================================================================
    // Lambda_z potential coefficients (c, d) using TE reflection (Sm, Sp)
    // ===================================================================
    let smsp = buf.sm[tx_layer] * buf.sp[tx_layer];
    let onemsmsp = ONE - smsp * expmgh_tx * expmgh_tx;

    let lam2 = lam * lam;
    let rhs_lz = MU0 * sdm / 2.0 / lam2;
    let srcp_lz = if height_tx.is_finite() {
        -(-gamma_tx * height_tx).exp() * rhs_lz
    } else {
        ZERO
    };
    let srcm_lz = (-gamma_tx * depth_tx).exp() * rhs_lz;

    buf.c[tx_layer] = (smsp * srcm_lz * expmgh_tx + buf.sp[tx_layer] * srcp_lz) / onemsmsp;
    buf.d[tx_layer] = (smsp * srcp_lz * expmgh_tx + buf.sm[tx_layer] * srcm_lz) / onemsmsp;

    // Propagate Lambda_z above source layer — short-circuit to rx_layer
    if rx_layer < tx_layer {
        let mut src = srcm_lz;
        for i in (rx_layer..tx_layer).rev() {
            let top = buf.c[i + 1] * buf.expmgh[i + 1] + buf.d[i + 1] + src;
            buf.c[i] = top / (ONE + buf.sm[i] * buf.expmgh[i]);
            buf.d[i] = buf.c[i] * buf.sm[i];
            src = ZERO;
        }
    }

    // Propagate Lambda_z below source layer — short-circuit to rx_layer
    if rx_layer > tx_layer {
        let mut src = srcp_lz;
        for i in (tx_layer + 1)..=rx_layer {
            let top = buf.c[i - 1] + buf.d[i - 1] * buf.expmgh[i - 1] + src;
            buf.d[i] = top / (ONE + buf.sp[i] * buf.expmgh[i]);
            buf.c[i] = buf.d[i] * buf.sp[i];
            src = ZERO;
        }
    }

    // ===================================================================
    // Coefficient derivatives (Fortran lines 3058-3375)
    // Using flat matrices: dadsig[k*n+j] = d(a[j])/d(sigma[k])
    // ===================================================================

    let (isgn_rm, isgn_rp) = if tx_layer == 0 || tx_layer == n - 1 {
        (0.0, 0.0)
    } else {
        (1.0, 1.0)
    };

    let dgammadsig_tx = -iomega_mu0 / (Complex64::new(2.0, 0.0) * gamma_tx);

    // ---- da/dsig in source layer ----
    {
        let p = if height_tx.is_finite() { (-gamma_tx * height_tx).exp() } else { ZERO };
        let rr = if depth_tx.is_finite() { (-gamma_tx * depth_tx).exp() } else { ZERO };
        let dp = if height_tx.is_finite() { -p * height_tx * dgammadsig_tx } else { ZERO };
        let dr = if depth_tx.is_finite() { -rr * depth_tx * dgammadsig_tx } else { ZERO };

        let q = buf.rm[tx_layer] * expmgh_tx;
        let s = buf.rp[tx_layer] * expmgh_tx;
        let t = buf.rp[tx_layer];
        let u = rhs_ay;
        let du = -dgammadsig_tx * rhs_ay / gamma_tx;

        let ff = buf.a[tx_layer];

        for k in 0..n {
            buf.dadsig[k * n + tx_layer] = ((t * u * rr + ff * s) * buf.drmdsig[k * n + tx_layer] * expmgh_tx
                + q * ff * buf.drpdsig[k * n + tx_layer] * expmgh_tx
                + (p + q * rr) * u * buf.drpdsig[k * n + tx_layer])
                / onemrmrp_ay;
        }
        buf.dadsig[tx_layer * n + tx_layer] += (t * u * (dp + q * dr)
            + (p + q * rr) * t * du
            + Complex64::new(isgn_rp, 0.0) * (p + q * rr) * u * t * h[tx_layer] * dgammadsig_tx)
            / onemrmrp_ay;
    }

    // ---- db/dsig in source layer ----
    {
        let p = if depth_tx.is_finite() { (-gamma_tx * depth_tx).exp() } else { ZERO };
        let rr = if height_tx.is_finite() { (-gamma_tx * height_tx).exp() } else { ZERO };
        let dp = if depth_tx.is_finite() { -p * depth_tx * dgammadsig_tx } else { ZERO };
        let dr = if height_tx.is_finite() { -rr * height_tx * dgammadsig_tx } else { ZERO };

        let q = buf.rp[tx_layer] * expmgh_tx;
        let s = buf.rm[tx_layer] * expmgh_tx;
        let t = buf.rm[tx_layer];
        let u = rhs_ay;
        let du = -dgammadsig_tx * rhs_ay / gamma_tx;

        let ff = buf.b[tx_layer];

        for k in 0..n {
            buf.dbdsig[k * n + tx_layer] = ((t * u * rr + ff * s) * buf.drpdsig[k * n + tx_layer] * expmgh_tx
                + q * ff * buf.drmdsig[k * n + tx_layer] * expmgh_tx
                + (p + q * rr) * u * buf.drmdsig[k * n + tx_layer])
                / onemrmrp_ay;
        }
        buf.dbdsig[tx_layer * n + tx_layer] += (t * u * (dp + q * dr)
            + (p + q * rr) * t * du
            + Complex64::new(isgn_rm, 0.0) * (p + q * rr) * u * t * h[tx_layer] * dgammadsig_tx)
            / onemrmrp_ay;
    }

    // Source terms for continuation
    let srcm_inv_ay = if depth_tx.is_finite() {
        -srcm_ay / gamma_tx * (gamma_tx * depth_tx + ONE) * dgammadsig_tx
    } else {
        ZERO
    };
    let srcp_inv_ay = if height_tx.is_finite() {
        -srcp_ay / gamma_tx * (gamma_tx * height_tx + ONE) * dgammadsig_tx
    } else {
        ZERO
    };

    // ---- Propagate da/db above source — short-circuit to rx_layer ----
    if rx_layer < tx_layer {
        let mut src_inv = srcm_inv_ay;
        for i in (rx_layer..tx_layer).rev() {
            let denom = ONE + buf.rm[i] * buf.expmgh[i];
            let dgamma_ip1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i + 1]);
            let dgamma_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);

            for k in 0..n {
                buf.dadsig[k * n + i] = buf.dbdsig[k * n + i + 1]
                    + buf.dadsig[k * n + i + 1] * buf.expmgh[i + 1]
                    - buf.a[i] * buf.drmdsig[k * n + i] * buf.expmgh[i];
            }
            buf.dadsig[(i + 1) * n + i] += -buf.a[i + 1] * h_times_expmgh(h[i + 1], buf.expmgh[i + 1]) * dgamma_ip1;
            buf.dadsig[(i + 1) * n + i] += src_inv;
            buf.dadsig[i * n + i] += buf.b[i] * h_times_expmgh(h[i], buf.expmgh[i]) * dgamma_i;

            for k in 0..n {
                buf.dadsig[k * n + i] /= denom;
            }

            for k in 0..n {
                buf.dbdsig[k * n + i] = buf.a[i] * buf.drmdsig[k * n + i] + buf.rm[i] * buf.dadsig[k * n + i];
            }

            src_inv = ZERO;
        }
    }

    // ---- Propagate da/db below source — short-circuit to rx_layer ----
    if rx_layer > tx_layer {
        let mut src_inv = srcp_inv_ay;
        for i in (tx_layer + 1)..=rx_layer {
            let denom = ONE + buf.rp[i] * buf.expmgh[i];
            let dgamma_im1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i - 1]);
            let dgamma_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);

            for k in 0..n {
                buf.dbdsig[k * n + i] = buf.dadsig[k * n + i - 1]
                    + buf.dbdsig[k * n + i - 1] * buf.expmgh[i - 1]
                    - buf.b[i] * buf.drpdsig[k * n + i] * buf.expmgh[i];
            }
            buf.dbdsig[(i - 1) * n + i] += -buf.b[i - 1] * h_times_expmgh(h[i - 1], buf.expmgh[i - 1]) * dgamma_im1;
            buf.dbdsig[(i - 1) * n + i] += src_inv;
            buf.dbdsig[i * n + i] += buf.a[i] * h_times_expmgh(h[i], buf.expmgh[i]) * dgamma_i;

            for k in 0..n {
                buf.dbdsig[k * n + i] /= denom;
            }

            for k in 0..n {
                buf.dadsig[k * n + i] = buf.b[i] * buf.drpdsig[k * n + i] + buf.rp[i] * buf.dbdsig[k * n + i];
            }

            src_inv = ZERO;
        }
    }

    // ===================================================================
    // Lambda_z coefficient derivatives (dc/dsig, dd/dsig)
    // ===================================================================

    // ---- dc/dsig in source layer ----
    {
        let p = if height_tx.is_finite() { -(-gamma_tx * height_tx).exp() } else { ZERO };
        let rr = if depth_tx.is_finite() { (-gamma_tx * depth_tx).exp() } else { ZERO };
        let dp = if height_tx.is_finite() { -p * height_tx * dgammadsig_tx } else { ZERO };
        let dr = if depth_tx.is_finite() { -rr * depth_tx * dgammadsig_tx } else { ZERO };

        let q = buf.sm[tx_layer] * expmgh_tx;
        let s = buf.sp[tx_layer] * expmgh_tx;
        let t = buf.sp[tx_layer];
        let u = rhs_lz;
        let du = ZERO;

        let ff = buf.c[tx_layer];

        for k in 0..n {
            buf.dcdsig[k * n + tx_layer] = ((t * u * rr + ff * s) * buf.dsmdsig[k * n + tx_layer] * expmgh_tx
                + q * ff * buf.dspdsig[k * n + tx_layer] * expmgh_tx
                + (p + q * rr) * u * buf.dspdsig[k * n + tx_layer])
                / onemsmsp;
        }
        buf.dcdsig[tx_layer * n + tx_layer] += (t * u * (dp + q * dr)
            + (p + q * rr) * t * du
            + Complex64::new(isgn_rp, 0.0) * (p + q * rr) * u * t * h[tx_layer] * dgammadsig_tx)
            / onemsmsp;
    }

    // ---- dd/dsig in source layer ----
    {
        let p = if depth_tx.is_finite() { (-gamma_tx * depth_tx).exp() } else { ZERO };
        let rr = if height_tx.is_finite() { -(-gamma_tx * height_tx).exp() } else { ZERO };
        let dp = if depth_tx.is_finite() { -p * depth_tx * dgammadsig_tx } else { ZERO };
        let dr = if height_tx.is_finite() { -rr * height_tx * dgammadsig_tx } else { ZERO };

        let q = buf.sp[tx_layer] * expmgh_tx;
        let s = buf.sm[tx_layer] * expmgh_tx;
        let t = buf.sm[tx_layer];
        let u = rhs_lz;
        let du = ZERO;

        let ff = buf.d[tx_layer];

        for k in 0..n {
            buf.dddsig[k * n + tx_layer] = ((t * u * rr + ff * s) * buf.dspdsig[k * n + tx_layer] * expmgh_tx
                + q * ff * buf.dsmdsig[k * n + tx_layer] * expmgh_tx
                + (p + q * rr) * u * buf.dsmdsig[k * n + tx_layer])
                / onemsmsp;
        }
        buf.dddsig[tx_layer * n + tx_layer] += (t * u * (dp + q * dr)
            + (p + q * rr) * t * du
            + Complex64::new(isgn_rm, 0.0) * (p + q * rr) * u * t * h[tx_layer] * dgammadsig_tx)
            / onemsmsp;
    }

    // Source terms for Lambda_z continuation
    let srcp_inv_lz = if height_tx.is_finite() {
        -srcp_lz * height_tx * dgammadsig_tx
    } else {
        ZERO
    };
    let srcm_inv_lz = if depth_tx.is_finite() {
        -srcm_lz * depth_tx * dgammadsig_tx
    } else {
        ZERO
    };

    // ---- Propagate dc/dd above source — short-circuit to rx_layer ----
    if rx_layer < tx_layer {
        let mut src_inv = srcm_inv_lz;
        for i in (rx_layer..tx_layer).rev() {
            let denom = ONE + buf.sm[i] * buf.expmgh[i];
            let dgamma_ip1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i + 1]);
            let dgamma_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);

            for k in 0..n {
                buf.dcdsig[k * n + i] = buf.dddsig[k * n + i + 1]
                    + buf.dcdsig[k * n + i + 1] * buf.expmgh[i + 1]
                    - buf.c[i] * buf.dsmdsig[k * n + i] * buf.expmgh[i];
            }
            buf.dcdsig[(i + 1) * n + i] += -buf.c[i + 1] * h_times_expmgh(h[i + 1], buf.expmgh[i + 1]) * dgamma_ip1;
            buf.dcdsig[(i + 1) * n + i] += src_inv;
            buf.dcdsig[i * n + i] += buf.d[i] * h_times_expmgh(h[i], buf.expmgh[i]) * dgamma_i;

            for k in 0..n {
                buf.dcdsig[k * n + i] /= denom;
            }

            for k in 0..n {
                buf.dddsig[k * n + i] = buf.c[i] * buf.dsmdsig[k * n + i] + buf.sm[i] * buf.dcdsig[k * n + i];
            }

            src_inv = ZERO;
        }
    }

    // ---- Propagate dc/dd below source — short-circuit to rx_layer ----
    if rx_layer > tx_layer {
        let mut src_inv = srcp_inv_lz;
        for i in (tx_layer + 1)..=rx_layer {
            let denom = ONE + buf.sp[i] * buf.expmgh[i];
            let dgamma_im1 = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i - 1]);
            let dgamma_i = -iomega_mu0 / (Complex64::new(2.0, 0.0) * buf.gamma[i]);

            for k in 0..n {
                buf.dddsig[k * n + i] = buf.dcdsig[k * n + i - 1]
                    + buf.dddsig[k * n + i - 1] * buf.expmgh[i - 1]
                    - buf.d[i] * buf.dspdsig[k * n + i] * buf.expmgh[i];
            }
            buf.dddsig[(i - 1) * n + i] += -buf.d[i - 1] * h_times_expmgh(h[i - 1], buf.expmgh[i - 1]) * dgamma_im1;
            buf.dddsig[(i - 1) * n + i] += src_inv;
            buf.dddsig[i * n + i] += buf.c[i] * h_times_expmgh(h[i], buf.expmgh[i]) * dgamma_i;

            for k in 0..n {
                buf.dddsig[k * n + i] /= denom;
            }

            for k in 0..n {
                buf.dcdsig[k * n + i] = buf.d[i] * buf.dspdsig[k * n + i] + buf.sp[i] * buf.dddsig[k * n + i];
            }

            src_inv = ZERO;
        }
    }
}

/// Complex square root with proper branch cut
/// Returns the principal square root with positive real part
pub fn complex_sqrt(z: Complex64) -> Complex64 {
    let r = z.norm();
    let theta = z.arg();
    Complex64::from_polar(r.sqrt(), theta / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::HankelFilter;
    use crate::model::LayeredEarth;

    #[test]
    fn test_min_rho_handling() {
        let model = LayeredEarth::halfspace(1.0);
        let omega = 2.0 * PI * 1.0;
        let coeffs = model.compute_coefficients(omega, 1);
        let filter = HankelFilter::Kong201;

        let fields = compute_hed_fields(
            &model,
            &coeffs,
            &filter,
            50.0,
            100.0,
            1,
            1,
            omega,
            0.0,
            PI / 2.0,
            true,
        )
        .unwrap();

        assert!(!fields.ex.re.is_nan(), "Ex should not be NaN at r=0");
        assert!(!fields.ey.re.is_nan(), "Ey should not be NaN at r=0");
        assert!(!fields.ez.re.is_nan(), "Ez should not be NaN at r=0");
    }
}
