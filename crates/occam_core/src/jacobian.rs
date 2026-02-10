//! Jacobian matrix operations for Occam inversion

use nalgebra::{DMatrix, DVector};

/// Jacobian matrix (partial derivatives of data with respect to model parameters)
#[derive(Debug, Clone)]
pub struct Jacobian {
    /// The Jacobian matrix J[i,j] = d(data_i)/d(param_j)
    pub matrix: DMatrix<f64>,
}

impl Jacobian {
    /// Create a new zero Jacobian
    pub fn new(n_data: usize, n_params: usize) -> Self {
        Self {
            matrix: DMatrix::zeros(n_data, n_params),
        }
    }

    /// Create from existing matrix
    pub fn from_matrix(matrix: DMatrix<f64>) -> Self {
        Self { matrix }
    }

    /// Get number of data points
    pub fn n_data(&self) -> usize {
        self.matrix.nrows()
    }

    /// Get number of parameters
    pub fn n_params(&self) -> usize {
        self.matrix.ncols()
    }

    /// Set a single element
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.matrix[(i, j)] = value;
    }

    /// Get a single element
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.matrix[(i, j)]
    }

    /// Multiply an entire column j by a scalar factor
    pub fn scale_column(&mut self, j: usize, factor: f64) {
        self.matrix.column_mut(j).scale_mut(factor);
    }

    /// Compute weighted Jacobian: W * J where W is diagonal weight matrix
    pub fn weighted(&self, weights: &[f64]) -> DMatrix<f64> {
        let mut wj = self.matrix.clone();
        for i in 0..self.n_data() {
            for j in 0..self.n_params() {
                wj[(i, j)] *= weights[i];
            }
        }
        wj
    }

    /// Compute J^T * W^2 * J (normal equations matrix)
    pub fn jtw2j(&self, weights: &[f64]) -> DMatrix<f64> {
        let wj = self.weighted(weights);
        wj.transpose() * &wj
    }

    /// Compute J^T * W^2 * d (right-hand side of normal equations)
    pub fn jtw2d(&self, weights: &[f64], residuals: &[f64]) -> DVector<f64> {
        let n = self.n_params();
        let mut result = DVector::zeros(n);

        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..self.n_data() {
                let w2 = weights[i] * weights[i];
                sum += self.matrix[(i, j)] * w2 * residuals[i];
            }
            result[j] = sum;
        }

        result
    }
}

/// Compute data weights from standard errors
pub fn compute_weights(errors: &[f64]) -> Vec<f64> {
    errors.iter().map(|&e| 1.0 / e).collect()
}

/// Compute weighted residuals: W * (d_obs - d_calc)
pub fn compute_weighted_residuals(
    observed: &[f64],
    computed: &[f64],
    weights: &[f64],
) -> Vec<f64> {
    observed
        .iter()
        .zip(computed.iter())
        .zip(weights.iter())
        .map(|((&obs, &comp), &w)| w * (obs - comp))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_jacobian() {
        let mut j = Jacobian::new(2, 2);
        j.set(0, 0, 1.0);
        j.set(0, 1, 2.0);
        j.set(1, 0, 3.0);
        j.set(1, 1, 4.0);

        let weights = vec![2.0, 0.5];
        let wj = j.weighted(&weights);

        assert_eq!(wj[(0, 0)], 2.0);
        assert_eq!(wj[(0, 1)], 4.0);
        assert_eq!(wj[(1, 0)], 1.5);
        assert_eq!(wj[(1, 1)], 2.0);
    }

}
