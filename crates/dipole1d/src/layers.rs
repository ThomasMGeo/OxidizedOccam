//! Layer coefficient calculations for Dipole1D
//!
//! Implements the recursive calculation of reflection/transmission coefficients
//! using decaying exponential formulation (Wait 1982)

use crate::constants::MU0;
use crate::model::LayeredEarth;
use num_complex::Complex64;

/// Precomputed layer coefficients for a given frequency
#[derive(Debug, Clone)]
pub struct LayerCoefficients {
    /// Number of layers
    pub n_layers: usize,
    /// Transmitter layer index
    pub tx_layer: usize,
    /// Angular frequency (rad/s)
    pub omega: f64,
    /// Vertical wavenumbers for each layer (computed per lambda)
    /// These are computed on-the-fly during field calculation
    _placeholder: (),
}

impl LayerCoefficients {
    /// Compute layer coefficients for given model and frequency
    pub fn compute(model: &LayeredEarth, omega: f64, tx_layer: usize) -> Self {
        Self {
            n_layers: model.n_layers(),
            tx_layer,
            omega,
            _placeholder: (),
        }
    }
}

/// TM mode reflection coefficients (for electric sources)
/// Rm: downward reflection from layer i
/// Rp: upward reflection from layer i
#[derive(Debug, Clone)]
pub struct TMCoefficients {
    /// Downward reflection coefficients
    pub rm: Vec<Complex64>,
    /// Upward reflection coefficients
    pub rp: Vec<Complex64>,
}

/// TE mode reflection coefficients (for magnetic sources)
/// Sm: downward reflection from layer i
/// Sp: upward reflection from layer i
#[derive(Debug, Clone)]
pub struct TECoefficients {
    /// Downward reflection coefficients
    pub sm: Vec<Complex64>,
    /// Upward reflection coefficients
    pub sp: Vec<Complex64>,
}

/// Compute TM mode reflection coefficients for a given horizontal wavenumber
///
/// Uses the recursive algorithm from Wait (1982) with decaying exponentials
/// to avoid numerical overflow.
///
/// Note: This is an alternative admittance-based TM formulation (gamma_i = sigma_i / u_i).
/// The primary forward model uses the equivalent formulation in fields.rs with
/// gamma = sqrt(lam^2 + k2) (Fortran Dipole1D.f90 hed_coef_nlayer, lines 2895-2972).
pub fn compute_tm_coefficients(
    model: &LayeredEarth,
    lambda: f64,
    omega: f64,
    tx_layer: usize,
) -> TMCoefficients {
    let n = model.n_layers();
    let mut rm = vec![Complex64::new(0.0, 0.0); n];
    let mut rp = vec![Complex64::new(0.0, 0.0); n];

    // Compute vertical wavenumbers for all layers
    let mut u = Vec::with_capacity(n);
    for i in 0..n {
        u.push(model.vertical_wavenumber(i, lambda, omega));
    }

    // Compute admittances gamma_i = sigma_i / u_i
    let mut gamma = Vec::with_capacity(n);
    for i in 0..n {
        let sigma = Complex64::new(model.conductivities[i], 0.0);
        gamma.push(sigma / u[i]);
    }

    // Downward recursion from bottom (Rm)
    // Start from last layer (halfspace): Rm[n-1] = 0
    rm[n - 1] = Complex64::new(0.0, 0.0);

    for i in (tx_layer..n - 1).rev() {
        // Reflection coefficient at interface i/i+1
        let r_local = (gamma[i] - gamma[i + 1]) / (gamma[i] + gamma[i + 1]);

        // Layer thickness
        let h = model.thickness(i);

        // Exponential decay factor
        let exp_factor = if h.is_finite() {
            (-2.0 * u[i] * h).exp()
        } else {
            Complex64::new(0.0, 0.0)
        };

        // Recursive relation
        rm[i] = (r_local + rm[i + 1] * exp_factor) / (Complex64::new(1.0, 0.0) + r_local * rm[i + 1] * exp_factor);
    }

    // Upward recursion from top (Rp)
    // Start from first layer (air): Rp[0] = 0 (or -1 for perfect reflector)
    rp[0] = Complex64::new(0.0, 0.0);

    for i in 1..=tx_layer {
        // Reflection coefficient at interface i-1/i
        let r_local = (gamma[i] - gamma[i - 1]) / (gamma[i] + gamma[i - 1]);

        // Layer thickness of layer above
        let h = model.thickness(i - 1);

        // Exponential decay factor
        let exp_factor = if h.is_finite() {
            (-2.0 * u[i - 1] * h).exp()
        } else {
            Complex64::new(0.0, 0.0)
        };

        // Recursive relation
        rp[i] = (r_local + rp[i - 1] * exp_factor) / (Complex64::new(1.0, 0.0) + r_local * rp[i - 1] * exp_factor);
    }

    TMCoefficients { rm, rp }
}

/// Compute TE mode reflection coefficients for a given horizontal wavenumber
///
/// Uses the recursive algorithm from Wait (1982) with decaying exponentials.
///
/// Note: This is an alternative impedance-based TE formulation (eta_i = iωμ₀ / u_i).
/// The primary forward model uses the equivalent formulation in fields.rs with
/// gamma = sqrt(lam^2 + k2) (Fortran Dipole1D.f90 hed_coef_nlayer, lines 2895-2972).
pub fn compute_te_coefficients(
    model: &LayeredEarth,
    lambda: f64,
    omega: f64,
    tx_layer: usize,
) -> TECoefficients {
    let n = model.n_layers();
    let mut sm = vec![Complex64::new(0.0, 0.0); n];
    let mut sp = vec![Complex64::new(0.0, 0.0); n];

    // Compute vertical wavenumbers for all layers
    let mut u = Vec::with_capacity(n);
    for i in 0..n {
        u.push(model.vertical_wavenumber(i, lambda, omega));
    }

    // Compute impedances eta_i = i*omega*mu0 / u_i
    let iomega_mu0 = Complex64::new(0.0, omega * MU0);
    let mut eta = Vec::with_capacity(n);
    for i in 0..n {
        eta.push(iomega_mu0 / u[i]);
    }

    // Downward recursion from bottom (Sm)
    sm[n - 1] = Complex64::new(0.0, 0.0);

    for i in (tx_layer..n - 1).rev() {
        let s_local = (eta[i] - eta[i + 1]) / (eta[i] + eta[i + 1]);
        let h = model.thickness(i);
        let exp_factor = if h.is_finite() {
            (-2.0 * u[i] * h).exp()
        } else {
            Complex64::new(0.0, 0.0)
        };
        sm[i] = (s_local + sm[i + 1] * exp_factor) / (Complex64::new(1.0, 0.0) + s_local * sm[i + 1] * exp_factor);
    }

    // Upward recursion from top (Sp)
    sp[0] = Complex64::new(0.0, 0.0);

    for i in 1..=tx_layer {
        let s_local = (eta[i] - eta[i - 1]) / (eta[i] + eta[i - 1]);
        let h = model.thickness(i - 1);
        let exp_factor = if h.is_finite() {
            (-2.0 * u[i - 1] * h).exp()
        } else {
            Complex64::new(0.0, 0.0)
        };
        sp[i] = (s_local + sp[i - 1] * exp_factor) / (Complex64::new(1.0, 0.0) + s_local * sp[i - 1] * exp_factor);
    }

    TECoefficients { sm, sp }
}

/// Compute potential coefficients for HED source
///
/// Returns (A_te, A_tm, B_te, B_tm) for up/down going waves
///
/// Note: This is an alternative transmission-coefficient formulation. The primary
/// forward model uses the reflection-coefficient formulation in fields.rs
/// (Fortran Dipole1D.f90 hed_coef_nlayer, lines 3046-3200 for a,b and 3215-3320 for c,d).
pub fn compute_hed_potential_coefficients(
    model: &LayeredEarth,
    lambda: f64,
    omega: f64,
    tx_layer: usize,
    rx_layer: usize,
    z_tx: f64,
    z_rx: f64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let tm = compute_tm_coefficients(model, lambda, omega, tx_layer);
    let te = compute_te_coefficients(model, lambda, omega, tx_layer);

    let u_tx = model.vertical_wavenumber(tx_layer, lambda, omega);
    let u_rx = model.vertical_wavenumber(rx_layer, lambda, omega);

    // Source amplitude (normalized)
    let source = Complex64::new(1.0, 0.0);

    // Compute coefficients based on receiver position relative to source
    let (a_te, a_tm, b_te, b_tm) = if rx_layer == tx_layer {
        // Same layer as source
        let z_top = model.depths[tx_layer];
        let z_bot = if tx_layer + 1 < model.n_layers() {
            model.depths[tx_layer + 1]
        } else {
            f64::INFINITY
        };

        let exp_up = (u_tx * (z_tx - z_top)).exp();
        let exp_down = if z_bot.is_finite() {
            (-u_tx * (z_bot - z_tx)).exp()
        } else {
            Complex64::new(0.0, 0.0)
        };

        // Direct + reflected contributions
        let rm_eff = tm.rm[tx_layer];
        let rp_eff = tm.rp[tx_layer];
        let sm_eff = te.sm[tx_layer];
        let sp_eff = te.sp[tx_layer];

        let denom_tm = Complex64::new(1.0, 0.0) - rm_eff * rp_eff * exp_up * exp_down;
        let denom_te = Complex64::new(1.0, 0.0) - sm_eff * sp_eff * exp_up * exp_down;

        let a_tm = source * (Complex64::new(1.0, 0.0) + rm_eff * exp_down) / (2.0 * u_tx * denom_tm);
        let b_tm = source * (rp_eff * exp_up + rm_eff * rp_eff * exp_up * exp_down) / (2.0 * u_tx * denom_tm);

        let a_te = source * (Complex64::new(1.0, 0.0) + sm_eff * exp_down) / (2.0 * u_tx * denom_te);
        let b_te = source * (sp_eff * exp_up + sm_eff * sp_eff * exp_up * exp_down) / (2.0 * u_tx * denom_te);

        (a_te, a_tm, b_te, b_tm)
    } else if rx_layer > tx_layer {
        // Receiver below source - downward propagation
        let mut a_tm = source / (2.0 * u_tx);
        let mut a_te = source / (2.0 * u_tx);

        // Propagate through layers
        for i in tx_layer..rx_layer {
            let h = model.thickness(i);
            let u_i = model.vertical_wavenumber(i, lambda, omega);
            let u_ip1 = model.vertical_wavenumber(i + 1, lambda, omega);

            // Transmission coefficients
            let gamma_i = Complex64::new(model.conductivities[i], 0.0) / u_i;
            let gamma_ip1 = Complex64::new(model.conductivities[i + 1], 0.0) / u_ip1;
            let t_tm = 2.0 * gamma_i / (gamma_i + gamma_ip1);

            let iomega_mu0 = Complex64::new(0.0, omega * MU0);
            let eta_i = iomega_mu0 / u_i;
            let eta_ip1 = iomega_mu0 / u_ip1;
            let t_te = 2.0 * eta_i / (eta_i + eta_ip1);

            let exp_factor = (-u_i * h).exp();
            a_tm *= t_tm * exp_factor;
            a_te *= t_te * exp_factor;
        }

        (a_te, a_tm, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0))
    } else {
        // Receiver above source - upward propagation
        let mut b_tm = source / (2.0 * u_tx);
        let mut b_te = source / (2.0 * u_tx);

        // Propagate through layers
        for i in (rx_layer + 1..=tx_layer).rev() {
            let h = model.thickness(i - 1);
            let u_i = model.vertical_wavenumber(i, lambda, omega);
            let u_im1 = model.vertical_wavenumber(i - 1, lambda, omega);

            let gamma_i = Complex64::new(model.conductivities[i], 0.0) / u_i;
            let gamma_im1 = Complex64::new(model.conductivities[i - 1], 0.0) / u_im1;
            let t_tm = 2.0 * gamma_i / (gamma_i + gamma_im1);

            let iomega_mu0 = Complex64::new(0.0, omega * MU0);
            let eta_i = iomega_mu0 / u_i;
            let eta_im1 = iomega_mu0 / u_im1;
            let t_te = 2.0 * eta_i / (eta_i + eta_im1);

            let exp_factor = (-u_im1 * h).exp();
            b_tm *= t_tm * exp_factor;
            b_te *= t_te * exp_factor;
        }

        (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), b_te, b_tm)
    };

    (a_te, a_tm, b_te, b_tm)
}

/// Compute potential coefficients for VED source
///
/// Note: This is an alternative transmission-coefficient formulation. The primary
/// forward model uses the reflection-coefficient formulation in fields.rs
/// (Fortran Dipole1D.f90 ved_pot_nlayer, lines 3925-4069). VED uses TM mode only.
pub fn compute_ved_potential_coefficients(
    model: &LayeredEarth,
    lambda: f64,
    omega: f64,
    tx_layer: usize,
    rx_layer: usize,
    z_tx: f64,
    z_rx: f64,
) -> (Complex64, Complex64) {
    let tm = compute_tm_coefficients(model, lambda, omega, tx_layer);

    let u_tx = model.vertical_wavenumber(tx_layer, lambda, omega);

    // Source amplitude (normalized)
    let source = Complex64::new(1.0, 0.0);

    // VED only generates TM mode
    let (a_tm, b_tm) = if rx_layer == tx_layer {
        let z_top = model.depths[tx_layer];
        let z_bot = if tx_layer + 1 < model.n_layers() {
            model.depths[tx_layer + 1]
        } else {
            f64::INFINITY
        };

        let exp_up = (u_tx * (z_tx - z_top)).exp();
        let exp_down = if z_bot.is_finite() {
            (-u_tx * (z_bot - z_tx)).exp()
        } else {
            Complex64::new(0.0, 0.0)
        };

        let rm_eff = tm.rm[tx_layer];
        let rp_eff = tm.rp[tx_layer];

        let denom = Complex64::new(1.0, 0.0) - rm_eff * rp_eff * exp_up * exp_down;

        let a = source * (Complex64::new(1.0, 0.0) + rm_eff * exp_down) / (2.0 * u_tx * denom);
        let b = source * (rp_eff * exp_up + rm_eff * rp_eff * exp_up * exp_down) / (2.0 * u_tx * denom);

        (a, b)
    } else if rx_layer > tx_layer {
        let mut a = source / (2.0 * u_tx);

        for i in tx_layer..rx_layer {
            let h = model.thickness(i);
            let u_i = model.vertical_wavenumber(i, lambda, omega);
            let u_ip1 = model.vertical_wavenumber(i + 1, lambda, omega);

            let gamma_i = Complex64::new(model.conductivities[i], 0.0) / u_i;
            let gamma_ip1 = Complex64::new(model.conductivities[i + 1], 0.0) / u_ip1;
            let t = 2.0 * gamma_i / (gamma_i + gamma_ip1);

            let exp_factor = (-u_i * h).exp();
            a *= t * exp_factor;
        }

        (a, Complex64::new(0.0, 0.0))
    } else {
        let mut b = source / (2.0 * u_tx);

        for i in (rx_layer + 1..=tx_layer).rev() {
            let h = model.thickness(i - 1);
            let u_i = model.vertical_wavenumber(i, lambda, omega);
            let u_im1 = model.vertical_wavenumber(i - 1, lambda, omega);

            let gamma_i = Complex64::new(model.conductivities[i], 0.0) / u_i;
            let gamma_im1 = Complex64::new(model.conductivities[i - 1], 0.0) / u_im1;
            let t = 2.0 * gamma_i / (gamma_i + gamma_im1);

            let exp_factor = (-u_im1 * h).exp();
            b *= t * exp_factor;
        }

        (Complex64::new(0.0, 0.0), b)
    };

    (a_tm, b_tm)
}

