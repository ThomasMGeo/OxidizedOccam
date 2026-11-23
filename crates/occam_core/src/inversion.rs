//! Main Occam inversion algorithm

use crate::error::{ConvergenceStatus, OccamError, Result};
use crate::forward::{ForwardEngine, ForwardResult};
use crate::jacobian::compute_weights;
use crate::linesearch::{bracket_minimum, brent_minimize, find_root};
use crate::penalty::{build_penalty_matrix, build_ptp, compute_roughness};
use crate::transform::{ParameterBounds, TransformType, transform_to_bounded, transform_to_unbounded};
use nalgebra::{DMatrix, DVector};
use occam_io::{EMData, IterationFile, LayeredModel, RoughnessType};

/// Occam inversion configuration
#[derive(Debug, Clone)]
pub struct OccamConfig {
    /// Target RMS misfit
    pub target_misfit: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Maximum step size reductions
    pub max_stepsize_cuts: usize,
    /// Roughness type
    pub roughness_type: RoughnessType,
    /// Initial Lagrange multiplier (log10)
    pub initial_lagrange: f64,
    /// Minimum Lagrange multiplier (log10)
    pub min_lagrange: f64,
    /// Maximum Lagrange multiplier (log10)
    pub max_lagrange: f64,
    /// Convergence tolerance for Lagrange search
    pub lagrange_tolerance: f64,
    /// Parameter bounds
    pub bounds: ParameterBounds,
    /// Debug level
    pub debug_level: i32,
    /// Auto-converge threshold on the absolute change in RMS misfit over one
    /// iteration (iteration file: `Delta Misfit Limit`).
    /// 0.0 (the default) disables the check. See `misfit_stalled`.
    pub delta_misfit_limit: f64,
}

impl Default for OccamConfig {
    fn default() -> Self {
        Self {
            target_misfit: 1.0,
            max_iterations: 100,
            max_stepsize_cuts: 8,
            roughness_type: RoughnessType::FirstDerivative,
            initial_lagrange: 5.0,
            min_lagrange: -6.0,
            max_lagrange: 8.0,
            lagrange_tolerance: 0.1,
            bounds: ParameterBounds::default(),
            debug_level: 0,
            delta_misfit_limit: 0.0,
        }
    }
}

/// Result of a single Occam iteration
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// Iteration number
    pub iteration: usize,
    /// Updated model parameters (log10 resistivity, bounded m-space)
    pub parameters: Vec<f64>,
    /// Computed forward response (same order as input data points)
    pub computed: Vec<f64>,
    /// RMS misfit
    pub rms_misfit: f64,
    /// Model roughness
    pub roughness: f64,
    /// Lagrange multiplier (log10)
    pub lagrange: f64,
    /// Step size used
    pub step_size: f64,
    /// Convergence status
    pub status: ConvergenceStatus,
}

/// Main Occam inversion engine
pub struct OccamInversion {
    /// Forward modeling engine
    forward: ForwardEngine,
    /// Inversion configuration
    config: OccamConfig,
    /// Penalty matrix
    penalty_matrix: DMatrix<f64>,
    /// P^T * P matrix
    ptp: DMatrix<f64>,
    /// Data weights (1/error)
    weights: Vec<f64>,
    /// Observed data
    observed: Vec<f64>,
    /// Current iteration
    iteration: usize,
    /// Previous roughness (for convergence check)
    prev_roughness: f64,
    /// Previous Lagrange multiplier (log10) for next iteration starting point
    prev_lagrange: f64,
    /// Target misfit reached flag
    target_reached: bool,
}

impl OccamInversion {
    /// Create a new Occam inversion
    pub fn new(
        data: EMData,
        model: LayeredModel,
        config: OccamConfig,
    ) -> Result<Self> {
        let forward = ForwardEngine::new(data.clone(), model.clone());

        // Build penalty matrix
        let n_params = model.n_free_params();
        let penalties = model.penalties();
        let depths = model.depths();

        // Get penalties only for free parameters
        let free_indices = model.free_param_indices();
        let free_penalties: Vec<f64> = free_indices.iter().map(|&i| penalties[i]).collect();
        let free_depths: Vec<f64> = free_indices.iter().map(|&i| depths[i]).collect();

        let penalty_matrix = build_penalty_matrix(
            n_params,
            config.roughness_type,
            &free_penalties,
            &free_depths,
        );
        let ptp = build_ptp(&penalty_matrix);

        let weights = compute_weights(&forward.data_errors());
        let observed = forward.observed_data();

        let initial_lagrange = config.initial_lagrange;

        Ok(Self {
            forward,
            config,
            penalty_matrix,
            ptp,
            weights,
            observed,
            iteration: 0,
            prev_roughness: 0.0,
            prev_lagrange: initial_lagrange,
            target_reached: false,
        })
    }

    /// Whether bounds transforms are active
    fn bounds_active(&self) -> bool {
        self.config.bounds.transform != TransformType::None
    }

    /// Transform x-space params to bounded m-space (for forward modeling)
    fn to_bounded(&self, x: &[f64]) -> Vec<f64> {
        if self.bounds_active() {
            transform_to_bounded(x, &self.config.bounds)
        } else {
            x.to_vec()
        }
    }

    /// Transform bounded m-space params to x-space (for inversion)
    fn to_unbounded(&self, m: &[f64]) -> Vec<f64> {
        if self.bounds_active() {
            transform_to_unbounded(m, &self.config.bounds)
        } else {
            m.to_vec()
        }
    }

    /// Load inversion from files.
    ///
    /// Reads model, data, and iteration files and constructs the inversion
    /// engine. Parameters in the iteration file are in bounded m-space
    /// (log10 resistivity); if bounds are active they are transformed to
    /// unbounded x-space for the internal solver.
    ///
    /// Fortran ref: Occam.f90, compOccamIteration (lines 1307-1384)
    ///   - Read iteration file: lines 1307-1380
    ///   - Model param -> conductivity: CSEM1D.f90 lines 1720-1747
    ///     sig1D(i) = 1/(10^currentMod(iparam))
    pub fn from_files(iteration_file: &IterationFile) -> Result<(Self, Vec<f64>)> {
        let data = EMData::read_from_file(&iteration_file.data_file)?;
        let model = LayeredModel::read_from_file(&iteration_file.model_file)?;

        let mut config = OccamConfig {
            target_misfit: iteration_file.target_misfit,
            max_iterations: iteration_file.max_iterations,
            max_stepsize_cuts: iteration_file.stepsize_cut_count,
            roughness_type: iteration_file.roughness_type,
            initial_lagrange: iteration_file.lagrange_value,
            debug_level: iteration_file.debug_level,
            delta_misfit_limit: iteration_file.delta_misfit_limit,
            ..Default::default()
        };

        // Wire bounds from iteration file
        if let Some((lo, hi)) = iteration_file.model_bounds {
            let transform = match iteration_file.bounds_transform.as_deref() {
                Some("exponential") => TransformType::Exponential,
                _ => TransformType::Bandpass, // default
            };
            config.bounds = ParameterBounds::new(lo, hi).with_transform(transform);
        }

        let mut inversion = Self::new(data, model, config)?;
        inversion.iteration = iteration_file.iteration;
        inversion.target_reached = iteration_file.misfit_reached;

        // Parameters in the startup file are in bounded m-space.
        // If bounds are active, nudge inside bounds and transform to x-space.
        let mut params = iteration_file.parameters.clone();
        if inversion.bounds_active() {
            let bounds = &inversion.config.bounds;
            let tol = (bounds.upper - bounds.lower) / 232.0;
            for p in params.iter_mut() {
                *p = p.clamp(bounds.lower + tol, bounds.upper - tol);
            }
            params = inversion.to_unbounded(&params);
        }

        Ok((inversion, params))
    }

    /// Get number of free parameters
    pub fn n_params(&self) -> usize {
        self.forward.n_params()
    }

    /// Get number of data points
    pub fn n_data(&self) -> usize {
        self.forward.n_data()
    }

    /// Run forward model (params are in x-space if bounds active)
    pub fn forward(&self, params: &[f64]) -> Result<ForwardResult> {
        let m = self.to_bounded(params);
        self.forward.compute(&m)
    }

    /// Compute model roughness (in x-space — roughness is of unbounded params)
    pub fn roughness(&self, params: &[f64]) -> f64 {
        compute_roughness(params, &self.penalty_matrix)
    }

    /// Perform a single Occam iteration.
    ///
    /// `params` are in x-space (unbounded) if bounds are active.
    /// The returned `IterationResult.parameters` are in m-space (bounded).
    ///
    /// Step-size damping is applied **inside** the Lagrange search, matching
    /// Fortran. When the best misfit from the search exceeds the starting
    /// misfit, `frac` is doubled and the entire search is repeated with
    /// damped forward evaluations. Similarly, if roughness grows in the
    /// smoothing phase, `frac` is doubled and the search is retried.
    ///
    /// Fortran ref: Occam.f90, compOccamIteration (lines 1307-1571)
    ///   1. Compute forward + Jacobian: line 1384 (call computeForward)
    ///   2. Build normal equations: lines 1715-1789 (makjtj)
    ///   3. Lagrange search with frac: lines 1416-1468, tofmu lines 2827-2829
    ///      frac=1 initially; if tmin >= tol0, frac *= 2, goto 120
    ///   4. Roughness check: lines 1518-1535
    ///      if roughness >= prev_roughness && smoothing, frac *= 2, goto 120
    ///   5. Convergence check: lines 1450-1571
    pub fn iterate(&mut self, params: &[f64]) -> Result<IterationResult> {
        self.iteration += 1;
        let debug = self.config.debug_level >= 1;

        // Transform to bounded m-space for forward computation
        let m_params = self.to_bounded(params);

        // Compute forward response and Jacobian (in m-space)
        let (forward_result, mut jacobian) = self.forward.compute_with_jacobian(&m_params)?;

        // If bounds active, scale Jacobian by dm/dx for each parameter
        // dF/dx_j = (dm/dx_j) * dF/dm_j  (chain rule)
        if self.bounds_active() {
            let bounds = &self.config.bounds;
            for j in 0..jacobian.n_params() {
                let dmdx = bounds.transform_jacobian(params[j]);
                jacobian.scale_column(j, dmdx);
            }
        }

        // Check if we've reached target
        if forward_result.rms_misfit <= self.config.target_misfit && !self.target_reached {
            self.target_reached = true;
        }

        // Compute weighted data residuals: d - F[m_k]
        let residuals: Vec<f64> = self.observed
            .iter()
            .zip(forward_result.computed.iter())
            .map(|(&obs, &comp)| obs - comp)
            .collect();

        // Build normal equations matrices (in x-space).
        // Constable et al. (1987) Eq. 9: solve for full model x, not update dx.
        //
        // Fortran ref: Occam.f90, makjtj subroutine (lines 1715-1789)
        //   wj(k,j) = sd(k) * wj(k,j)                   [W * J]
        //   pjtj(j1,j2) += wj(k,j1) * wj(k,j2)         [J^T W^2 J]
        //   pjtd(j) += wj(k,j) * sd(k) * wd(k)          [J^T W^2 (d - F[m_k])]
        //
        // Then in tofmu (lines 2752-2758):
        //   pwkn(j) = pjtd(j) + sum(pjtj(j,:) * pm(:))   [= J^T W^2 d_hat]
        //   where d_hat = d - F[m_k] + J * x_k
        let jtw2j = jacobian.jtw2j(&self.weights);
        let jtw2d = jacobian.jtw2d(&self.weights, &residuals);

        // RHS = J^T W^2 (d - F[m_k]) + J^T W^2 J * x_k = J^T W^2 d_hat
        let x_k = DVector::from_column_slice(params);
        let jtw2d_hat = &jtw2d + &jtw2j * &x_k;

        // Step-size damping loop with two strategies.
        //
        // Strategy 1 (post-processing): interpolate the full-step solution toward
        // the previous model. Fast (no re-solve), works well for unbounded case.
        //
        // Strategy 2 (Fortran-style frac): redo the entire Lagrange search with
        // frac applied inside every solve (tofmu lines 2827-2829). Essential for
        // bounded inversions where the nonlinear transform makes interpolation
        // ineffective.
        //
        // Fortran ref: Occam.f90, lines 1416-1535
        //   label 120: bracket + minimize with current frac
        //   line 1452: if tmin >= tol0 (fitting) or tmin > tolreq (smoothing)
        //     → frac *= 2, goto 120  (lines 1459, 1467)
        //   line 1519: if roughness >= prev_roughness (smoothing)
        //     → frac *= 2, goto 120  (lines 1521, 1535)
        //   line 1462: if frac >= 2^nfractimes → give up
        let max_frac = (2.0_f64).powi(self.config.max_stepsize_cuts as i32);
        let obs = &self.observed;
        let errors = self.forward.data_errors();

        // First: full-step Lagrange search (frac=1)
        let (calc_x, calc_misfit, mut lagrange, calc_computed) = self.lagrange_search(
            &jtw2j,
            &jtw2d_hat,
            forward_result.rms_misfit,
            params,
            1.0,
        )?;

        let mut new_x = calc_x;
        let mut new_misfit = calc_misfit;
        let mut new_computed = calc_computed;
        let mut frac = 1.0_f64;

        // Check if full step needs damping
        let needs_damping = if !self.target_reached {
            new_misfit >= forward_result.rms_misfit
        } else {
            new_misfit > self.config.target_misfit
        };

        if needs_damping {
            // Strategy 1: try post-processing interpolation first
            // x_damped = (1 - step) * x_old + step * x_new
            let mut found_improvement = false;
            let mut interp_frac = 2.0_f64;

            for _ in 0..self.config.max_stepsize_cuts {
                let step = 1.0 / interp_frac;
                let damped_x: Vec<f64> = params
                    .iter()
                    .zip(new_x.iter())
                    .map(|(&old, &new)| (1.0 - step) * old + step * new)
                    .collect();

                let damped_m = self.to_bounded(&damped_x);
                let result = self.forward.compute_with_cache(&damped_m, obs, &errors)?;

                if result.rms_misfit < forward_result.rms_misfit {
                    new_x = damped_x;
                    new_misfit = result.rms_misfit;
                    new_computed = result.computed;
                    frac = interp_frac;
                    found_improvement = true;
                    break;
                }

                interp_frac *= 2.0;
            }

            // Strategy 2: if interpolation didn't help, redo search with frac
            // (Fortran-style — essential for bounded inversions)
            if !found_improvement {
                frac = 2.0;

                loop {
                    if debug {
                        eprintln!("  Divergence: cutting step size (frac={})", frac);
                    }

                    let (x, misfit, mu, computed) = self.lagrange_search(
                        &jtw2j,
                        &jtw2d_hat,
                        forward_result.rms_misfit,
                        params,
                        frac,
                    )?;
                    new_x = x;
                    new_misfit = misfit;
                    lagrange = mu;
                    new_computed = computed;

                    let still_bad = if !self.target_reached {
                        new_misfit >= forward_result.rms_misfit
                    } else {
                        new_misfit > self.config.target_misfit
                    };

                    if !still_bad {
                        break;
                    }

                    frac *= 2.0;
                    if frac >= max_frac {
                        if debug {
                            eprintln!("  Step-size damping exhausted (frac={})", frac);
                        }
                        break;
                    }
                }
            }
        }

        // Fortran lines 1518-1535: roughness growth check (smoothing phase)
        // If roughness grows, redo search with increasing frac.
        if self.target_reached
            && self.roughness(&new_x) >= self.prev_roughness
            && self.prev_roughness > 0.0
        {
            let mut rough_frac = frac.max(1.0) * 2.0;

            loop {
                if rough_frac >= max_frac {
                    if debug {
                        eprintln!("  Roughness not resolved by small steps (frac={})", rough_frac);
                    }
                    break;
                }
                if debug {
                    eprintln!(
                        "  Roughness growing, cutting step (frac={})",
                        rough_frac
                    );
                }

                let (x, misfit, mu, computed) = self.lagrange_search(
                    &jtw2j,
                    &jtw2d_hat,
                    forward_result.rms_misfit,
                    params,
                    rough_frac,
                )?;

                let roughness = self.roughness(&x);
                new_x = x;
                new_misfit = misfit;
                lagrange = mu;
                new_computed = computed;
                frac = rough_frac;

                if roughness < self.prev_roughness {
                    break;
                }

                rough_frac *= 2.0;
            }
        }

        let step_size = 1.0 / frac;

        // Compute roughness of new model (in x-space)
        let new_roughness = self.roughness(&new_x);

        // Check convergence
        let status = self.check_convergence(
            forward_result.rms_misfit,
            new_misfit,
            new_roughness,
        );

        self.prev_roughness = new_roughness;
        self.prev_lagrange = lagrange;

        let final_m = self.to_bounded(&new_x);

        Ok(IterationResult {
            iteration: self.iteration,
            parameters: final_m, // Output in bounded m-space
            computed: new_computed,
            rms_misfit: new_misfit,
            roughness: new_roughness,
            lagrange,
            step_size,
            status,
        })
    }

    /// Run complete inversion
    pub fn run(&mut self, initial_params: &[f64]) -> Result<Vec<IterationResult>> {
        self.run_timed(initial_params, |_, _| {})
    }

    /// Run complete inversion with per-iteration timing callback
    ///
    /// `initial_params` are in x-space (unbounded) if bounds are active.
    /// Each `IterationResult` returned has parameters in bounded m-space.
    /// Internally, the iteration loop passes x-space params between iterations.
    pub fn run_timed<F>(
        &mut self,
        initial_params: &[f64],
        mut callback: F,
    ) -> Result<Vec<IterationResult>>
    where
        F: FnMut(&IterationResult, std::time::Duration),
    {
        use std::time::Instant;

        let mut x_params = initial_params.to_vec();
        let mut results = Vec::new();

        // Initial forward computation (transform to bounded for forward)
        let t0 = Instant::now();
        let m_params = self.to_bounded(&x_params);
        let initial = self.forward.compute(&m_params)?;
        let elapsed = t0.elapsed();

        let init_result = IterationResult {
            iteration: 0,
            parameters: m_params, // Output in bounded m-space
            computed: initial.computed,
            rms_misfit: initial.rms_misfit,
            roughness: self.roughness(&x_params),
            lagrange: self.config.initial_lagrange,
            step_size: 1.0,
            status: ConvergenceStatus::Iterating,
        };
        callback(&init_result, elapsed);
        results.push(init_result);

        for _ in 0..self.config.max_iterations {
            let t0 = Instant::now();
            let result = self.iterate(&x_params)?;
            let elapsed = t0.elapsed();

            // Convert result parameters back to x-space for next iteration
            x_params = self.to_unbounded(&result.parameters);

            let should_stop = result.status.should_stop();
            callback(&result, elapsed);
            results.push(result);

            if should_stop {
                break;
            }
        }

        Ok(results)
    }

    /// Search for optimal Lagrange multiplier with step-size damping.
    ///
    /// All parameters here are in x-space. The solve produces x-space solutions,
    /// which are transformed to m-space for forward evaluations.
    ///
    /// For each candidate log10(mu), solves the normal equations
    /// (Constable et al. 1987, Eq. 9):
    ///   [mu * P^T P + J^T W^2 J] x = J^T W^2 d_hat
    /// where d_hat = d - F[m_k] + J * x_k  (linearized data vector).
    ///
    /// When `frac > 1`, the solved x is damped toward the previous model:
    ///   x_damped = (1 - 1/frac) * pm + (1/frac) * x_solved
    /// This damping is applied INSIDE every forward evaluation during the
    /// search, so that the bracketing and minimization work with damped models.
    ///
    /// Fortran ref: Occam.f90, tofmu subroutine (lines 2666-2867)
    ///   - Normal equations solve: lines 2752-2782
    ///   - Step size damping: lines 2827-2829
    ///     if (frac > 1.001) part = 1./frac; pwk2 = (1-part)*pm + part*pwk2
    ///   - Forward eval: line 2841  computefwd(transformToBound(pwk2))
    ///   - Misfit: line 2848  tofmu = sqrt(anorm(nd, (d-dm)/sd) / nd)
    fn lagrange_search(
        &self,
        jtw2j: &DMatrix<f64>,
        jtw2d_hat: &DVector<f64>,
        _current_misfit: f64,
        pm: &[f64],
        frac: f64,
    ) -> Result<(Vec<f64>, f64, f64, Vec<f64>)> {
        // Cache observed/errors to avoid re-allocating per forward call
        let obs = &self.observed;
        let errors = self.forward.data_errors();

        // Step-size damping factor: part = 1/frac (Fortran line 2828)
        let use_damping = frac > 1.001;
        let part = 1.0 / frac;

        // Function to solve normal equations and optionally damp.
        // Returns None if Cholesky decomposition fails.
        //
        // Fortran ref: Occam.f90 lines 2794-2830
        let solve_for_mu = |log_mu: f64| -> Option<Vec<f64>> {
            let mu = 10.0_f64.powf(log_mu);

            // Build system: [μ P^T P + J^T W^2 J] x = J^T W^2 d̂
            let mut lhs = jtw2j.clone();
            lhs += mu * &self.ptp;

            // Solve for full model x (not an update dx).
            let x = lhs.cholesky()?.solve(jtw2d_hat);

            let mut result = x.as_slice().to_vec();

            // Fortran lines 2827-2829: damp toward previous model
            if use_damping {
                for (r, &p) in result.iter_mut().zip(pm.iter()) {
                    *r = (1.0 - part) * p + part * *r;
                }
            }

            Some(result)
        };

        // Function to compute misfit for a given Lagrange multiplier.
        // Returns a huge misfit if Cholesky fails (Fortran line 2809: tofmu = 1e10).
        let debug = self.config.debug_level >= 2;
        let mut compute_misfit = |log_mu: f64| -> Result<f64> {
            let Some(new_x) = solve_for_mu(log_mu) else {
                if debug {
                    eprintln!("  lagrange_search: log10(mu)={:.4}  Cholesky failed", log_mu);
                }
                return Ok(1e10);
            };
            let new_m = self.to_bounded(&new_x);
            let result = self.forward.compute_with_cache(&new_m, obs, &errors)?;
            if debug {
                eprintln!(
                    "  lagrange_search: log10(mu)={:.4}  RMS={:.6}  frac={}",
                    log_mu, result.rms_misfit, frac
                );
            }
            Ok(result.rms_misfit)
        };

        // Initial Lagrange value (start near previous iteration's optimal value)
        let initial_log_mu = if self.target_reached {
            self.prev_lagrange
        } else {
            self.config.max_lagrange.min(self.prev_lagrange + 1.0)
        };

        // Bracket minimum
        let bracket = bracket_minimum(
            initial_log_mu,
            -1.0,
            &mut compute_misfit,
            20,
        )?;

        // Find minimum using Brent's method
        let (opt_log_mu, opt_misfit) = brent_minimize(
            &bracket,
            self.config.lagrange_tolerance,
            &mut compute_misfit,
            50,
        )?;

        // If target reached, find intercept with target misfit
        let final_log_mu = if self.target_reached && opt_misfit < self.config.target_misfit {
            find_root(
                opt_log_mu,
                self.config.max_lagrange,
                self.config.target_misfit,
                self.config.lagrange_tolerance,
                &mut compute_misfit,
                30,
            ).unwrap_or(opt_log_mu)
        } else {
            opt_log_mu
        };

        // Compute final model with chosen mu (already damped)
        let new_x = solve_for_mu(final_log_mu)
            .ok_or_else(|| OccamError::LinearAlgebra("Cholesky decomposition failed".to_string()))?;
        let new_m = self.to_bounded(&new_x);
        let result = self.forward.compute_with_cache(&new_m, obs, &errors)?;

        Ok((new_x, result.rms_misfit, final_log_mu, result.computed))
    }

    /// Check convergence status
    ///
    /// Fortran reference: Occam.f90, compOccamIteration (lines 1450-1571)
    fn check_convergence(
        &self,
        prev_misfit: f64,
        new_misfit: f64,
        new_roughness: f64,
    ) -> ConvergenceStatus {
        const TINY_ROUGHNESS: f64 = 1e-5;
        const PERFECT_FIT_FACTOR: f64 = 1.01;
        const TARGET_TOLERANCE: f64 = 1.05; // Accept misfit within 5% of target

        // Check for perfectly smooth model
        if new_roughness < TINY_ROUGHNESS
            && new_misfit <= PERFECT_FIT_FACTOR * self.config.target_misfit
        {
            return ConvergenceStatus::PerfectlySmooth;
        }

        // Smoothing phase: we've already reached target, looking for smoothest model
        if self.target_reached {
            if new_misfit <= self.config.target_misfit * TARGET_TOLERANCE {
                // Misfit is at or near target — check roughness convergence
                if new_roughness > self.prev_roughness * 1.01 && self.prev_roughness > 0.0 {
                    return ConvergenceStatus::RoughnessIncreasing;
                }
                return ConvergenceStatus::Converged;
            }
            // Misfit went too far above target — the root-finding overshot
            // Fall through to fitting phase behavior
        }

        // Check if target reached (fitting phase)
        if new_misfit <= self.config.target_misfit {
            return ConvergenceStatus::TargetReached;
        }

        // Check for divergence (fitting phase only)
        if !self.target_reached && new_misfit > prev_misfit * 1.1 {
            return ConvergenceStatus::Diverging;
        }

        // Auto-converge: misfit stalled above target (Delta Misfit Limit)
        if misfit_stalled(
            prev_misfit,
            new_misfit,
            self.config.target_misfit,
            self.config.delta_misfit_limit,
        ) {
            return ConvergenceStatus::AutoConverged;
        }

        ConvergenceStatus::Iterating
    }
}

/// `Delta Misfit Limit` stall test ("auto converge").
///
/// Returns true when an iteration ended with the RMS misfit still above the
/// target and the absolute change in RMS misfit over that iteration is
/// strictly smaller than `delta_misfit_limit`:
///
///   new_misfit > target_misfit  &&  |new_misfit - prev_misfit| < limit
///
/// `prev_misfit` is the misfit of the model the iteration started from and
/// `new_misfit` the misfit of the model it produced (Fortran `tol0`/`tobt`).
/// The change is absolute (RMS units), not relative. A limit of 0.0 (the
/// default when the field is absent) can never trigger, nor can a negative
/// or NaN limit.
///
/// Fortran ref: Occam1DCSEM v3.13, Occam.f90, end of compOccamIteration:
///   if (tmin < tolreq) then ... else
///       tobt = tmin
///       if (abs(tobt - tol0) < rDeltaMisfit) konv = 4
/// The test only runs in the branch where the new misfit is not below the
/// target, so it never cuts short the smoothing phase of a model that fits.
/// (At exact equality `new_misfit == target_misfit` this code reports
/// `TargetReached` before the stall test is consulted.)
fn misfit_stalled(
    prev_misfit: f64,
    new_misfit: f64,
    target_misfit: f64,
    delta_misfit_limit: f64,
) -> bool {
    new_misfit > target_misfit && (new_misfit - prev_misfit).abs() < delta_misfit_limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_misfit_zero_limit_never_triggers() {
        // Default (field absent or 0.0): no stall detection, even for an
        // iteration that did not change the misfit at all.
        assert!(!misfit_stalled(1.3590, 1.3590, 1.0, 0.0));
        assert!(!misfit_stalled(1.3591, 1.3590, 1.0, 0.0));
        assert!(!misfit_stalled(1.3591, 1.3590, 1.0, -1.0));
        assert!(!misfit_stalled(1.3591, 1.3590, 1.0, f64::NAN));
    }

    #[test]
    fn delta_misfit_stall_above_target_triggers() {
        // |1.3590 - 1.3591| = 1e-4 < 1e-3, still above target 1.0
        assert!(misfit_stalled(1.3591, 1.3590, 1.0, 1e-3));
        // Absolute change, either sign
        assert!(misfit_stalled(1.3590, 1.3591, 1.0, 1e-3));
    }

    #[test]
    fn delta_misfit_large_change_keeps_iterating() {
        assert!(!misfit_stalled(5.0, 2.0, 1.0, 1e-3));
        // Strict inequality: a change equal to the limit does not trigger
        assert!(!misfit_stalled(1.5, 1.25, 1.0, 0.25));
    }

    #[test]
    fn delta_misfit_ignored_once_target_reached() {
        // Misfit at or below target: normal Occam smoothing/convergence logic
        // applies, however small the change.
        assert!(!misfit_stalled(1.0001, 0.9999, 1.0, 1e-3));
        assert!(!misfit_stalled(1.0, 1.0, 1.0, 1e-3));
        assert!(!misfit_stalled(0.9, 0.9, 1.0, 1e-3));
    }

    #[test]
    fn auto_converged_stops_but_is_not_success() {
        assert!(ConvergenceStatus::AutoConverged.should_stop());
        assert!(!ConvergenceStatus::AutoConverged.is_success());
    }
}
