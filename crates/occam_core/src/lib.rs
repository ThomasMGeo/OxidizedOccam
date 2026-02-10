//! Core Occam Inversion Algorithm
//!
//! Implements the Occam's inversion algorithm (Constable et al., 1987) for
//! 1D CSEM and MT data inversion.
//!
//! # Algorithm Overview
//!
//! Occam's inversion minimizes:
//!   Φ(m) = Φ_d(m) + μ * Φ_r(m)
//!
//! Where:
//! - Φ_d = normalized data misfit (chi-squared)
//! - Φ_r = model roughness
//! - μ = Lagrange multiplier (trade-off parameter)

mod error;
mod forward;
mod inversion;
mod jacobian;
mod linesearch;
mod penalty;
mod transform;

pub use error::*;
pub use forward::*;
pub use inversion::*;
pub use jacobian::*;
pub use linesearch::*;
pub use penalty::*;
pub use transform::*;
