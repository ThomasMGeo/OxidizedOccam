//! Error types for occam_core

use thiserror::Error;

/// Errors that can occur during inversion
#[derive(Error, Debug)]
pub enum OccamError {
    #[error("Forward modeling error: {0}")]
    Forward(String),

    #[error("Linear algebra error: {0}")]
    LinearAlgebra(String),

    #[error("Convergence failed: {0}")]
    Convergence(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("IO error: {0}")]
    Io(#[from] occam_io::OccamIoError),

    #[error("Dipole1D error: {0}")]
    Dipole1D(#[from] dipole1d::Dipole1DError),

    #[error("MT1D error: {0}")]
    MT1D(#[from] mt1d::MT1DError),
}

pub type Result<T> = std::result::Result<T, OccamError>;

/// Convergence status of inversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Still iterating, not converged
    Iterating,
    /// Target misfit reached, now smoothing
    TargetReached,
    /// Converged to smooth model at target misfit
    Converged,
    /// Model is perfectly smooth
    PerfectlySmooth,
    /// Diverging, cannot reduce misfit
    Diverging,
    /// Roughness increasing when should be decreasing
    RoughnessIncreasing,
    /// Auto-convergence triggered
    AutoConverged,
    /// Cholesky decomposition of the normal equations failed, so the step
    /// produced no model; the last successfully computed model is kept
    FactorizationFailed,
}

impl ConvergenceStatus {
    /// Check if inversion should stop
    pub fn should_stop(&self) -> bool {
        matches!(
            self,
            ConvergenceStatus::Converged
                | ConvergenceStatus::PerfectlySmooth
                | ConvergenceStatus::Diverging
                | ConvergenceStatus::AutoConverged
                | ConvergenceStatus::FactorizationFailed
        )
    }

    /// Check if this is a successful convergence
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            ConvergenceStatus::Converged
                | ConvergenceStatus::PerfectlySmooth
                | ConvergenceStatus::AutoConverged
        )
    }
}
