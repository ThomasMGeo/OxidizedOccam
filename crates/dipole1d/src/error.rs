//! Error types for dipole1d

use thiserror::Error;

/// Errors that can occur during forward modeling
#[derive(Error, Debug)]
pub enum Dipole1DError {
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    #[error("Computation error: {0}")]
    Computation(String),

    #[error("Invalid frequency: {0} Hz (must be positive)")]
    InvalidFrequency(f64),

    #[error("Receiver at depth {0} is outside model bounds")]
    ReceiverOutOfBounds(f64),

    #[error("Transmitter at depth {0} is outside model bounds")]
    TransmitterOutOfBounds(f64),

    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
}

pub type Result<T> = std::result::Result<T, Dipole1DError>;
