//! Error types for dipole1d

use thiserror::Error;

/// Errors that can occur during forward modeling
#[derive(Error, Debug)]
pub enum Dipole1DError {
    #[error("Invalid model: {0}")]
    InvalidModel(String),
}

pub type Result<T> = std::result::Result<T, Dipole1DError>;
