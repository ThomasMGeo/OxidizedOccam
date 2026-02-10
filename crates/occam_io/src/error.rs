//! Error types for occam_io

use thiserror::Error;

/// Errors that can occur during file I/O operations
#[derive(Error, Debug)]
pub enum OccamIoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Invalid format: expected {expected}, got {got}")]
    InvalidFormat { expected: String, got: String },

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid data type code: {0}")]
    InvalidDataType(i32),

    #[error("Index out of bounds: {field} index {index} (max {max})")]
    IndexOutOfBounds {
        field: String,
        index: usize,
        max: usize,
    },

    #[error("Parameter count mismatch: model has {model}, iteration has {iteration}")]
    ParameterCountMismatch { model: usize, iteration: usize },
}

pub type Result<T> = std::result::Result<T, OccamIoError>;
