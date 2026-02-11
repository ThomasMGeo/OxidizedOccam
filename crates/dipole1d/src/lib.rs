//! # Dipole1D — 1D CSEM Forward Modeling
//!
//! Computes electromagnetic fields from electric dipole sources in a 1D layered
//! earth model. Based on the Fortran implementation by Kerry Key, Scripps
//! Institution of Oceanography.
//!
//! # Algorithm
//!
//! 1. Lorenz-gauged vector potentials Ay and Lambda_z (Wait 1982)
//! 2. TM/TE mode reflection coefficients via decaying exponential recursion
//! 3. Cross-layer coefficient propagation from source to receiver
//! 4. Hankel transform via 201-point digital filter (Kerry Key kk_ht_201)
//! 5. Optional finite dipole using Gauss-Legendre quadrature
//!
//! # Coordinate Convention
//!
//! - z is positive downward (depth)
//! - Azimuth is measured from the x-axis (0° = x-directed, 90° = y-directed)
//! - theta = 0 is the crossline direction (perpendicular to dipole)
//! - theta = 90° is the inline direction (along dipole axis)
//! - Time convention: exp(-iωt) (consistent with Fortran Dipole1D)
//!
//! # Example
//!
//! ```
//! use dipole1d::{Dipole1D, HankelFilter, LayeredEarth, Receiver, Transmitter};
//!
//! // 2-layer model: air + 1 ohm-m halfspace
//! let model = LayeredEarth::halfspace(1.0);
//! let engine = Dipole1D::new(model);
//!
//! let tx = Transmitter::hed(0.0, 0.0, 50.0, 0.0); // x-directed HED at 50m depth
//! let rx = Receiver::new(1000.0, 0.0, 100.0);       // 1km inline, 100m depth
//!
//! let fields = engine.compute(&tx, &[rx], 1.0).unwrap();
//! println!("Ex = {:e}", fields[0].ex);
//! ```
//!
//! # References
//!
//! - Key, K. (2009). 1D inversion of multicomponent, multifrequency marine CSEM
//!   data. *Geophysics*, 74(2), F9-F20.
//! - Wait, J. R. (1982). *Geo-Electromagnetism*. Academic Press.

mod error;
mod fields;
mod filters;
mod layers;
mod model;
mod sensitivity;
pub mod work_buffers;

pub use error::*;
pub use fields::*;
pub use filters::*;
pub use layers::*;
pub use model::*;
pub use sensitivity::*;

use num_complex::Complex64;
use std::f64::consts::PI;

/// Physical constants
pub mod constants {
    use std::f64::consts::PI;

    /// Permeability of free space (H/m)
    pub const MU0: f64 = 4.0e-7 * PI;
}

/// Transmitter configuration
#[derive(Debug, Clone)]
pub struct Transmitter {
    /// X position (m)
    pub x: f64,
    /// Y position (m)
    pub y: f64,
    /// Z position (m), positive downward
    pub z: f64,
    /// Azimuth angle (degrees from x-axis)
    pub azimuth: f64,
    /// Dip angle (degrees, positive downward)
    pub dip: f64,
    /// Dipole moment (A·m)
    pub moment: f64,
    /// Dipole length for finite dipole (m), 0 for point source
    pub length: f64,
}

impl Default for Transmitter {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            azimuth: 0.0,
            dip: 0.0,
            moment: 1.0,
            length: 0.0,
        }
    }
}

impl Transmitter {
    /// Create a horizontal electric dipole (HED) at given position
    pub fn hed(x: f64, y: f64, z: f64, azimuth: f64) -> Self {
        Self {
            x,
            y,
            z,
            azimuth,
            dip: 0.0,
            moment: 1.0,
            length: 0.0,
        }
    }

    /// Create a vertical electric dipole (VED) at given position
    pub fn ved(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            azimuth: 0.0,
            dip: 90.0,
            moment: 1.0,
            length: 0.0,
        }
    }

    /// Set dipole moment
    pub fn with_moment(mut self, moment: f64) -> Self {
        self.moment = moment;
        self
    }

    /// Set finite dipole length
    pub fn with_length(mut self, length: f64) -> Self {
        self.length = length;
        self
    }

    /// Check if this is a point dipole
    pub fn is_point_dipole(&self) -> bool {
        self.length == 0.0
    }

    /// Get unit direction vector (in transmitter-local coordinates)
    pub fn direction(&self) -> (f64, f64, f64) {
        let az_rad = self.azimuth.to_radians();
        let dip_rad = self.dip.to_radians();
        let cos_dip = dip_rad.cos();
        (cos_dip * az_rad.cos(), cos_dip * az_rad.sin(), dip_rad.sin())
    }
}

/// Receiver position
#[derive(Debug, Clone, Default)]
pub struct Receiver {
    /// X position (m)
    pub x: f64,
    /// Y position (m)
    pub y: f64,
    /// Z position (m), positive downward
    pub z: f64,
}

impl Receiver {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Computed electromagnetic fields at a receiver
#[derive(Debug, Clone, Default)]
pub struct EMFields {
    /// Electric field Ex (V/m)
    pub ex: Complex64,
    /// Electric field Ey (V/m)
    pub ey: Complex64,
    /// Electric field Ez (V/m)
    pub ez: Complex64,
    /// Magnetic field Bx (T)
    pub bx: Complex64,
    /// Magnetic field By (T)
    pub by: Complex64,
    /// Magnetic field Bz (T)
    pub bz: Complex64,
}

impl EMFields {
    /// Create zero fields
    pub fn zero() -> Self {
        Self::default()
    }

    /// Scale all fields by a factor
    pub fn scale(&mut self, factor: Complex64) {
        self.ex *= factor;
        self.ey *= factor;
        self.ez *= factor;
        self.bx *= factor;
        self.by *= factor;
        self.bz *= factor;
    }

    /// Add another set of fields
    pub fn add(&mut self, other: &EMFields) {
        self.ex += other.ex;
        self.ey += other.ey;
        self.ez += other.ez;
        self.bx += other.bx;
        self.by += other.by;
        self.bz += other.bz;
    }
}

/// Main Dipole1D computation engine
pub struct Dipole1D {
    /// Layered earth model
    pub model: LayeredEarth,
    /// Hankel filter to use
    pub filter: HankelFilter,
    /// Whether to compute magnetic fields
    pub compute_b_fields: bool,
    /// Number of integration points for finite dipole
    pub n_integ_points: usize,
}

impl Dipole1D {
    /// Create a new Dipole1D computation engine
    pub fn new(model: LayeredEarth) -> Self {
        Self {
            model,
            filter: HankelFilter::Kong201,
            compute_b_fields: true,
            n_integ_points: 5,
        }
    }

    /// Set the Hankel filter
    pub fn with_filter(mut self, filter: HankelFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Set whether to compute magnetic fields
    pub fn with_b_fields(mut self, compute: bool) -> Self {
        self.compute_b_fields = compute;
        self
    }

    /// Compute electromagnetic fields at receivers
    ///
    /// Fortran reference: CallDipole1D.f90 main program (lines 104-165)
    /// orchestrates Tx loop, frequency loop, and calls comp_dipole1D (line 141).
    pub fn compute(
        &self,
        tx: &Transmitter,
        receivers: &[Receiver],
        frequency: f64,
    ) -> Result<Vec<EMFields>> {
        let omega = 2.0 * PI * frequency;

        // Find transmitter layer
        let tx_layer = self.model.find_layer(tx.z);

        // Compute layer coefficients
        let coeffs = self.model.compute_coefficients(omega, tx_layer);

        let mut results = Vec::with_capacity(receivers.len());

        for rx in receivers {
            // Compute relative position
            let dx = rx.x - tx.x;
            let dy = rx.y - tx.y;
            let rho = (dx * dx + dy * dy).sqrt();
            let phi = dy.atan2(dx);

            // Find receiver layer
            let rx_layer = self.model.find_layer(rx.z);

            // Compute fields based on dipole orientation
            let fields = if tx.is_point_dipole() {
                self.compute_point_dipole(tx, rx, &coeffs, tx_layer, rx_layer, omega, rho, phi)?
            } else {
                self.compute_finite_dipole(tx, rx, &coeffs, tx_layer, rx_layer, omega)?
            };

            results.push(fields);
        }

        Ok(results)
    }

    /// Compute fields for a point dipole source
    ///
    /// Decomposes the dipole into HED + VED components based on dip angle,
    /// computes fields for each, applies rotation to cardinal coordinates.
    ///
    /// Fortran reference: Dipole1D.f90
    /// - HED/VED decomposition and moment scaling: comp_spatial (lines 905-925)
    /// - Field rotation to cardinal coordinates: rotate_fields (lines 1558-1583),
    ///   called at lines 1440-1441
    /// - VED cylindrical-to-Cartesian conversion: lines 917-922
    fn compute_point_dipole(
        &self,
        tx: &Transmitter,
        rx: &Receiver,
        coeffs: &LayerCoefficients,
        tx_layer: usize,
        rx_layer: usize,
        omega: f64,
        rho: f64,
        phi: f64,
    ) -> Result<EMFields> {
        let mut fields = EMFields::zero();

        // Decompose into HED and VED components
        let (dx, dy, dz) = tx.direction();
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        // HED contribution (horizontal component)
        let h_moment = tx.moment * (dx * dx + dy * dy).sqrt();
        if h_moment.abs() > 1e-30 {
            // Compute theta following Fortran convention:
            // thetaRx = angle from x-axis to receiver direction
            // rotang = azimuth - 90° (angle from x to crossline direction)
            // theta = thetaRx - rotang = angle from crossline to receiver
            let theta_rx = if rho < 1.0 {
                PI / 2.0  // Default to 90° for very close receivers
            } else {
                phi  // phi = atan2(dy, dx)
            };
            let rotang = (tx.azimuth - 90.0).to_radians();
            let theta = theta_rx - rotang;

            let hed_fields = compute_hed_fields(
                &self.model,
                coeffs,
                &self.filter,
                tx.z,
                rx.z,
                tx_layer,
                rx_layer,
                omega,
                rho,
                theta,
                self.compute_b_fields,
            )?;

            // Rotate HED fields from dipole-local to global coordinates
            // Following Fortran: rotate_fields(exh, eyh, -rotang)
            // where rotang = azimuth - 90° (already computed above)
            let neg_rotang = -rotang;
            let cos_rot = neg_rotang.cos();
            let sin_rot = neg_rotang.sin();

            let moment = Complex64::new(h_moment, 0.0);

            // Fortran rotate_fields: tmpx = fx*cos + fy*sin, tmpy = -fx*sin + fy*cos
            let ex_rot = hed_fields.ex * cos_rot + hed_fields.ey * sin_rot;
            let ey_rot = -hed_fields.ex * sin_rot + hed_fields.ey * cos_rot;
            fields.ex += moment * ex_rot;
            fields.ey += moment * ey_rot;
            fields.ez += moment * hed_fields.ez;

            if self.compute_b_fields {
                let bx_rot = hed_fields.bx * cos_rot + hed_fields.by * sin_rot;
                let by_rot = -hed_fields.bx * sin_rot + hed_fields.by * cos_rot;
                fields.bx += moment * bx_rot;
                fields.by += moment * by_rot;
                fields.bz += moment * hed_fields.bz;
            }
        }

        // VED contribution (vertical component)
        let v_moment = tx.moment * dz;
        if v_moment.abs() > 1e-30 {
            let ved_fields = compute_ved_fields(
                &self.model,
                coeffs,
                &self.filter,
                tx.z,
                rx.z,
                tx_layer,
                rx_layer,
                omega,
                rho,
                self.compute_b_fields,
            )?;

            // VED fields are already in cylindrical, convert to Cartesian
            fields.ex += Complex64::new(v_moment, 0.0) * ved_fields.ex * cos_phi;
            fields.ey += Complex64::new(v_moment, 0.0) * ved_fields.ex * sin_phi;
            fields.ez += Complex64::new(v_moment, 0.0) * ved_fields.ez;

            if self.compute_b_fields {
                fields.bx += Complex64::new(v_moment, 0.0) * (-ved_fields.by * sin_phi);
                fields.by += Complex64::new(v_moment, 0.0) * (ved_fields.by * cos_phi);
                // VED has no Bz component
            }
        }

        Ok(fields)
    }

    /// Compute fields for a finite-length dipole using Gauss quadrature
    ///
    /// Fortran reference: Dipole1D.f90
    /// - Gauss-Legendre integration: lines 450-530
    /// - Integration method selection (finite_integ_method = 2): line 235
    /// - Quadrature nodes via legendre_compute_dr: line 473
    /// - Integration loop over point sources: line 639
    fn compute_finite_dipole(
        &self,
        tx: &Transmitter,
        rx: &Receiver,
        coeffs: &LayerCoefficients,
        tx_layer: usize,
        rx_layer: usize,
        omega: f64,
    ) -> Result<EMFields> {
        // Gauss-Legendre nodes and weights for n=5
        let (nodes, weights) = gauss_legendre_nodes(self.n_integ_points);

        let (dx, dy, dz) = tx.direction();
        let half_len = tx.length / 2.0;

        let mut total_fields = EMFields::zero();

        for (&node, &weight) in nodes.iter().zip(weights.iter()) {
            // Position along dipole
            let t = node * half_len;
            let point_tx = Transmitter {
                x: tx.x + t * dx,
                y: tx.y + t * dy,
                z: tx.z + t * dz,
                azimuth: tx.azimuth,
                dip: tx.dip,
                moment: tx.moment * weight * half_len,
                length: 0.0, // Point source
            };

            let dx_rx = rx.x - point_tx.x;
            let dy_rx = rx.y - point_tx.y;
            let rho = (dx_rx * dx_rx + dy_rx * dy_rx).sqrt();
            let phi = dy_rx.atan2(dx_rx);

            let point_fields =
                self.compute_point_dipole(&point_tx, rx, coeffs, tx_layer, rx_layer, omega, rho, phi)?;
            total_fields.add(&point_fields);
        }

        Ok(total_fields)
    }

    /// Compute fields and sensitivities (Jacobian) for all receivers
    ///
    /// Returns one EMFieldSensitivity per receiver (in cardinal coordinates).
    /// Currently supports point dipole HED sources only (VED sensitivity not yet implemented).
    pub fn compute_with_sensitivity(
        &self,
        tx: &Transmitter,
        receivers: &[Receiver],
        frequency: f64,
    ) -> Result<(Vec<EMFields>, Vec<EMFieldSensitivity>)> {
        let omega = 2.0 * PI * frequency;
        let tx_layer = self.model.find_layer(tx.z);
        let coeffs = self.model.compute_coefficients(omega, tx_layer);
        let n_layers = self.model.n_layers();

        let mut fields_vec = Vec::with_capacity(receivers.len());
        let mut sensitivities_vec = Vec::with_capacity(receivers.len());

        // Pre-allocate work buffers once, reused across all receivers
        let mut buf = work_buffers::WorkBuffers::new(n_layers);

        // Decompose into HED and VED components
        let (dx_dir, dy_dir, dz_dir) = tx.direction();
        let h_moment = tx.moment * (dx_dir * dx_dir + dy_dir * dy_dir).sqrt();

        for rx in receivers {
            let dx = rx.x - tx.x;
            let dy = rx.y - tx.y;
            let rho = (dx * dx + dy * dy).sqrt();
            let phi = dy.atan2(dx);
            let rx_layer = self.model.find_layer(rx.z);

            let mut fields = EMFields::zero();
            let mut total_sens = EMFieldSensitivity::zeros(n_layers);

            // HED contribution
            if h_moment.abs() > 1e-30 {
                let theta_rx = if rho < 1.0 { PI / 2.0 } else { phi };
                let rotang = (tx.azimuth - 90.0).to_radians();
                let theta = theta_rx - rotang;

                let (hed_fields, hed_sens_vec) = compute_hed_fields_with_sensitivity(
                    &self.model,
                    &coeffs,
                    &self.filter,
                    tx.z,
                    rx.z,
                    tx_layer,
                    rx_layer,
                    omega,
                    rho,
                    theta,
                    self.compute_b_fields,
                    &mut buf,
                )?;

                let hed_sens = &hed_sens_vec[0];

                // Rotate fields and sensitivities from dipole-local to global coordinates
                // (Fortran lines 1439-1449)
                let neg_rotang = -rotang;
                let cos_rot = neg_rotang.cos();
                let sin_rot = neg_rotang.sin();
                let moment = Complex64::new(h_moment, 0.0);

                // Rotate forward fields
                let ex_rot = hed_fields.ex * cos_rot + hed_fields.ey * sin_rot;
                let ey_rot = -hed_fields.ex * sin_rot + hed_fields.ey * cos_rot;
                fields.ex += moment * ex_rot;
                fields.ey += moment * ey_rot;
                fields.ez += moment * hed_fields.ez;

                if self.compute_b_fields {
                    let bx_rot = hed_fields.bx * cos_rot + hed_fields.by * sin_rot;
                    let by_rot = -hed_fields.bx * sin_rot + hed_fields.by * cos_rot;
                    fields.bx += moment * bx_rot;
                    fields.by += moment * by_rot;
                    fields.bz += moment * hed_fields.bz;
                }

                // Rotate sensitivities (same rotation as fields)
                for k in 0..n_layers {
                    let dex_rot = hed_sens.dex_dsigma[k] * cos_rot
                        + hed_sens.dey_dsigma[k] * sin_rot;
                    let dey_rot = -hed_sens.dex_dsigma[k] * sin_rot
                        + hed_sens.dey_dsigma[k] * cos_rot;
                    total_sens.dex_dsigma[k] += moment * dex_rot;
                    total_sens.dey_dsigma[k] += moment * dey_rot;
                    total_sens.dez_dsigma[k] += moment * hed_sens.dez_dsigma[k];

                    if self.compute_b_fields {
                        let dbx_rot = hed_sens.dbx_dsigma[k] * cos_rot
                            + hed_sens.dby_dsigma[k] * sin_rot;
                        let dby_rot = -hed_sens.dbx_dsigma[k] * sin_rot
                            + hed_sens.dby_dsigma[k] * cos_rot;
                        total_sens.dbx_dsigma[k] += moment * dbx_rot;
                        total_sens.dby_dsigma[k] += moment * dby_rot;
                        total_sens.dbz_dsigma[k] += moment * hed_sens.dbz_dsigma[k];
                    }
                }
            }

            // VED contribution with analytical sensitivity
            let v_moment = tx.moment * dz_dir;
            if v_moment.abs() > 1e-30 {
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();

                let (ved_fields, ved_sens) = compute_ved_fields_with_sensitivity(
                    &self.model,
                    &coeffs,
                    &self.filter,
                    tx.z,
                    rx.z,
                    tx_layer,
                    rx_layer,
                    omega,
                    rho,
                    self.compute_b_fields,
                    &mut buf,
                )?;

                let vm = Complex64::new(v_moment, 0.0);
                fields.ex += vm * ved_fields.ex * cos_phi;
                fields.ey += vm * ved_fields.ex * sin_phi;
                fields.ez += vm * ved_fields.ez;

                if self.compute_b_fields {
                    fields.bx += vm * (-ved_fields.by * sin_phi);
                    fields.by += vm * (ved_fields.by * cos_phi);
                }

                // Rotate VED sensitivities (same rotation as fields)
                for k in 0..n_layers {
                    total_sens.dex_dsigma[k] += vm * ved_sens.dex_dsigma[k] * cos_phi;
                    total_sens.dey_dsigma[k] += vm * ved_sens.dex_dsigma[k] * sin_phi;
                    total_sens.dez_dsigma[k] += vm * ved_sens.dez_dsigma[k];

                    if self.compute_b_fields {
                        total_sens.dbx_dsigma[k] += vm * (-ved_sens.dby_dsigma[k] * sin_phi);
                        total_sens.dby_dsigma[k] += vm * (ved_sens.dby_dsigma[k] * cos_phi);
                    }
                }
            }

            fields_vec.push(fields);
            sensitivities_vec.push(total_sens);
        }

        Ok((fields_vec, sensitivities_vec))
    }
}

/// Gauss-Legendre quadrature nodes and weights
fn gauss_legendre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.5773502691896257, 0.5773502691896257],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.7745966692414834, 0.0, 0.7745966692414834],
            vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
        ),
        4 => (
            vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            vec![
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        ),
        5 | _ => (
            vec![
                -0.9061798459386640,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.9061798459386640,
            ],
            vec![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_legendre() {
        let (nodes, weights) = gauss_legendre_nodes(5);
        assert_eq!(nodes.len(), 5);
        assert_eq!(weights.len(), 5);
        // Weights should sum to 2 (interval [-1, 1])
        let sum: f64 = weights.iter().sum();
        assert!((sum - 2.0).abs() < 1e-10);
    }
}
