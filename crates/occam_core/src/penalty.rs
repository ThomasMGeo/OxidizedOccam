//! Roughness penalty matrices for Occam inversion

use nalgebra::DMatrix;
use occam_io::RoughnessType;

/// Build the roughness penalty matrix for the given type
pub fn build_penalty_matrix(
    n_params: usize,
    roughness_type: RoughnessType,
    penalties: &[f64],
    depths: &[f64],
) -> DMatrix<f64> {
    match roughness_type {
        RoughnessType::FirstDerivative => build_first_derivative(n_params, penalties),
        RoughnessType::SecondDerivative => build_second_derivative(n_params, penalties),
        RoughnessType::DepthWeighted => build_depth_weighted(n_params, penalties, depths),
        RoughnessType::SumOfParams => build_sum_of_params(n_params),
        RoughnessType::MinGradientSupport => build_first_derivative(n_params, penalties), // Simplified
        RoughnessType::TwoModel => build_first_derivative(n_params, penalties), // Simplified
    }
}

/// First derivative roughness: R[i] = m[i+1] - m[i]
///
/// Produces a matrix P such that ||Pm||^2 = sum of squared first differences
fn build_first_derivative(n_params: usize, penalties: &[f64]) -> DMatrix<f64> {
    if n_params < 2 {
        return DMatrix::zeros(1, n_params);
    }

    let n_constraints = n_params - 1;
    let mut p = DMatrix::zeros(n_constraints, n_params);

    for i in 0..n_constraints {
        let penalty = if i + 1 < penalties.len() {
            penalties[i + 1]
        } else {
            1.0
        };

        if penalty > 0.0 {
            let sqrt_penalty = penalty.sqrt();
            p[(i, i)] = -sqrt_penalty;
            p[(i, i + 1)] = sqrt_penalty;
        }
    }

    p
}

/// Second derivative roughness: R[i] = m[i-1] - 2*m[i] + m[i+1]
fn build_second_derivative(n_params: usize, penalties: &[f64]) -> DMatrix<f64> {
    if n_params < 3 {
        return build_first_derivative(n_params, penalties);
    }

    let n_constraints = n_params - 2;
    let mut p = DMatrix::zeros(n_constraints, n_params);

    for i in 0..n_constraints {
        let penalty = if i + 1 < penalties.len() {
            penalties[i + 1]
        } else {
            1.0
        };

        if penalty > 0.0 {
            let sqrt_penalty = penalty.sqrt();
            p[(i, i)] = sqrt_penalty;
            p[(i, i + 1)] = -2.0 * sqrt_penalty;
            p[(i, i + 2)] = sqrt_penalty;
        }
    }

    p
}

/// Depth-weighted first derivative
fn build_depth_weighted(n_params: usize, penalties: &[f64], depths: &[f64]) -> DMatrix<f64> {
    if n_params < 2 {
        return DMatrix::zeros(1, n_params);
    }

    let n_constraints = n_params - 1;
    let mut p = DMatrix::zeros(n_constraints, n_params);

    for i in 0..n_constraints {
        let penalty = if i + 1 < penalties.len() {
            penalties[i + 1]
        } else {
            1.0
        };

        // Depth weighting: scale by 1/sqrt(depth)
        let depth_weight = if i + 1 < depths.len() && depths[i + 1] > 0.0 {
            1.0 / depths[i + 1].sqrt()
        } else {
            1.0
        };

        if penalty > 0.0 {
            let weight = (penalty * depth_weight).sqrt();
            p[(i, i)] = -weight;
            p[(i, i + 1)] = weight;
        }
    }

    p
}

/// Sum of parameters (minimum norm)
fn build_sum_of_params(n_params: usize) -> DMatrix<f64> {
    DMatrix::identity(n_params, n_params)
}

/// Compute model roughness: ||Pm||^2
pub fn compute_roughness(params: &[f64], penalty_matrix: &DMatrix<f64>) -> f64 {
    let m = nalgebra::DVector::from_column_slice(params);
    let pm = penalty_matrix * &m;
    pm.dot(&pm)
}

/// Build P^T * P matrix for efficient computation
pub fn build_ptp(penalty_matrix: &DMatrix<f64>) -> DMatrix<f64> {
    penalty_matrix.transpose() * penalty_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_derivative() {
        let p = build_first_derivative(4, &[0.0, 1.0, 1.0, 1.0]);

        // Should have 3 rows (constraints) and 4 columns (params)
        assert_eq!(p.nrows(), 3);
        assert_eq!(p.ncols(), 4);

        // First row: -1, 1, 0, 0
        assert_eq!(p[(0, 0)], -1.0);
        assert_eq!(p[(0, 1)], 1.0);
        assert_eq!(p[(0, 2)], 0.0);
    }

    #[test]
    fn test_second_derivative() {
        let p = build_second_derivative(4, &[0.0, 1.0, 1.0, 1.0]);

        // Should have 2 rows and 4 columns
        assert_eq!(p.nrows(), 2);
        assert_eq!(p.ncols(), 4);

        // First row: 1, -2, 1, 0
        assert_eq!(p[(0, 0)], 1.0);
        assert_eq!(p[(0, 1)], -2.0);
        assert_eq!(p[(0, 2)], 1.0);
    }

    #[test]
    fn test_roughness_calculation() {
        let params = vec![1.0, 2.0, 3.0, 4.0];
        let p = build_first_derivative(4, &[0.0, 1.0, 1.0, 1.0]);

        let roughness = compute_roughness(&params, &p);

        // All differences are 1.0, so roughness = 3 * 1^2 = 3
        assert!((roughness - 3.0).abs() < 1e-10);
    }

}
