//! File I/O for Occam 1D inversion
//!
//! This crate handles reading and writing of:
//! - EMData files (EMData_1.0, EMData_1.1, EMData_1.2)
//! - Model files (Resistivity1DMod_1.0)
//! - Iteration/startup files (OCCAMITER_FLEX)

mod data;
mod error;
mod iteration;
mod model;
mod response;

pub use data::*;
pub use error::*;
pub use iteration::*;
pub use model::*;
pub use response::*;
