//! Error types for mt1d

use thiserror::Error;

/// Errors that can occur during MT forward modeling
#[derive(Error, Debug)]
pub enum MT1DError {
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    #[error("Invalid frequency: {0} Hz (must be positive)")]
    InvalidFrequency(f64),

    #[error("Numerical error: {0}")]
    Numerical(String),
}

pub type Result<T> = std::result::Result<T, MT1DError>;
