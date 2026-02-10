//! Forward modeling interface for Occam inversion

use crate::error::{OccamError, Result};
use crate::jacobian::Jacobian;
use dipole1d::{Dipole1D, EMFields, EMFieldSensitivity, LayeredEarth, Receiver, Transmitter};
use mt1d::{MT1D, MT1DModel, MTResponse, MTSensitivity, compute_impedance_sensitivity_analytical};
use num_complex::Complex64;
use occam_io::{DataType, EMData, LayeredModel};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Forward modeling result
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Computed data values (same order as input data points)
    pub computed: Vec<f64>,
    /// RMS misfit
    pub rms_misfit: f64,
    /// Chi-squared misfit (normalized)
    pub chi_squared: f64,
}

impl ForwardResult {
    /// Compute misfit from observed and computed data
    pub fn compute_misfit(observed: &[f64], computed: &[f64], errors: &[f64]) -> (f64, f64) {
        let n = observed.len() as f64;
        let chi_sq: f64 = observed
            .iter()
            .zip(computed.iter())
            .zip(errors.iter())
            .map(|((&obs, &comp), &err)| {
                let residual = (obs - comp) / err;
                residual * residual
            })
            .sum();

        let rms = (chi_sq / n).sqrt();
        (rms, chi_sq / n)
    }
}

/// Forward modeling engine
pub struct ForwardEngine {
    /// EM data configuration
    data: EMData,
    /// Layered model structure
    model: LayeredModel,
}

impl ForwardEngine {
    /// Create a new forward engine
    pub fn new(data: EMData, model: LayeredModel) -> Self {
        Self { data, model }
    }

    /// Get number of data points
    pub fn n_data(&self) -> usize {
        self.data.data_points.len()
    }

    /// Get number of free parameters
    pub fn n_params(&self) -> usize {
        self.model.n_free_params()
    }

    /// Get observed data values
    pub fn observed_data(&self) -> Vec<f64> {
        self.data.data_points.iter().map(|dp| dp.value).collect()
    }

    /// Get data standard errors
    pub fn data_errors(&self) -> Vec<f64> {
        self.data.data_points.iter().map(|dp| dp.std_error).collect()
    }

    /// Run forward model with given parameters
    pub fn compute(&self, params: &[f64]) -> Result<ForwardResult> {
        self.compute_with_cache(params, &self.observed_data(), &self.data_errors())
    }

    /// Run forward model, reusing pre-computed observed/errors vecs
    pub fn compute_with_cache(
        &self,
        params: &[f64],
        observed: &[f64],
        errors: &[f64],
    ) -> Result<ForwardResult> {
        // Convert params (log10 resistivity) to resistivities
        let resistivities: Vec<f64> = params.iter().map(|&p| 10.0_f64.powf(p)).collect();

        let mut computed = vec![0.0; self.n_data()];

        if self.data.has_csem() {
            self.compute_csem(&resistivities, &mut computed)?;
        }

        if self.data.has_mt() {
            self.compute_mt(&resistivities, &mut computed)?;
        }

        let (rms, chi_sq) = ForwardResult::compute_misfit(observed, &computed, errors);

        Ok(ForwardResult {
            computed,
            rms_misfit: rms,
            chi_squared: chi_sq,
        })
    }

    /// Run forward model and compute Jacobian using analytical derivatives
    pub fn compute_with_jacobian(&self, params: &[f64]) -> Result<(ForwardResult, Jacobian)> {
        let resistivities: Vec<f64> = params.iter().map(|&p| 10.0_f64.powf(p)).collect();

        let n_data = self.n_data();
        let n_params = self.n_params();
        let mut computed = vec![0.0; n_data];
        let mut jacobian = Jacobian::new(n_data, n_params);

        // Build layer-to-free-param mapping
        let free_indices = self.model.free_param_indices();
        let n_layers = self.model.n_layers();
        let mut layer_to_param = vec![None; n_layers];
        for (col, &layer_idx) in free_indices.iter().enumerate() {
            layer_to_param[layer_idx] = Some(col);
        }

        if self.data.has_csem() {
            self.compute_csem_with_jacobian(
                &resistivities,
                &mut computed,
                &mut jacobian,
                &layer_to_param,
            )?;
        }

        if self.data.has_mt() {
            self.compute_mt_with_jacobian(
                &resistivities,
                &mut computed,
                &mut jacobian,
                &layer_to_param,
            )?;
        }

        let observed = self.observed_data();
        let errors = self.data_errors();
        let (rms, chi_sq) = ForwardResult::compute_misfit(&observed, &computed, &errors);

        Ok((
            ForwardResult {
                computed,
                rms_misfit: rms,
                chi_squared: chi_sq,
            },
            jacobian,
        ))
    }

    /// Group CSEM data points by (tx_idx, freq_idx).
    ///
    /// Returns a Vec of (tx_idx, freq_idx, data_point_indices) groups.
    /// Each group gets a single Dipole1D call with all receivers.
    ///
    /// Fortran ref: CSEM1D.f90 lines 1849-1862 (nested iTx/iFreq loops),
    /// lines 1869-1911 (site filtering: marks active receivers per group,
    /// skips group if no matching data points)
    fn build_csem_groups(&self) -> Vec<((usize, usize), Vec<usize>)> {
        let mut map: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (i, dp) in self.data.data_points.iter().enumerate() {
            if dp.data_type.is_csem() {
                map.entry((dp.tx_idx, dp.freq_idx)).or_default().push(i);
            }
        }
        map.into_iter().collect()
    }

    /// Compute CSEM forward response, grouped by (Tx, freq) and parallelized.
    ///
    /// Groups data points by (tx_idx, freq_idx) and makes one Dipole1D call per
    /// group with all unique receivers. Uses rayon for parallel execution across
    /// groups.
    ///
    /// Fortran ref: CSEM1D.f90, computeFwd_CSEM (lines 1789-2262)
    /// - Lines 1849-1862: outer iTx / inner iFreq loops
    /// - Lines 1869-1911: collect active receivers per (iTx, iFreq)
    /// - Line 1938: single call to comp_dipole1D per group
    /// - Lines 2080-2150: extract data values per data point
    fn compute_csem(&self, resistivities: &[f64], computed: &mut [f64]) -> Result<()> {
        let conductivities = self.model.conductivities(resistivities);
        let depths = self.model.depths();
        let earth = LayeredEarth::new(depths, conductivities)
            .map_err(|e| OccamError::Forward(e.to_string()))?;
        let dipole = Dipole1D::new(earth);
        let dipole_length = self.data.dipole_length.unwrap_or(0.0);

        let groups = self.build_csem_groups();

        // Process each (Tx, freq) group in parallel
        let group_results: Vec<Vec<(usize, f64)>> = groups
            .par_iter()
            .map(|((tx_idx, freq_idx), data_indices)| -> Result<Vec<(usize, f64)>> {
                let tx_def = &self.data.transmitters[*tx_idx];
                let freq = self.data.frequencies[*freq_idx];
                let tx = Transmitter {
                    x: tx_def.x,
                    y: tx_def.y,
                    z: tx_def.z,
                    azimuth: tx_def.azimuth,
                    dip: tx_def.dip,
                    moment: 1.0, // unit dipole moment (1 Am); CSEM data normalized to 1 Am by convention
                    length: dipole_length,
                };

                // Collect unique receivers for this group
                // Fortran ref: CSEM1D.f90 lines 1894-1905 (build n1d active receiver list)
                let (unique_rx, rx_map) = collect_unique_receivers(data_indices, &self.data);
                let receivers: Vec<Receiver> = unique_rx
                    .iter()
                    .map(|&ri| {
                        let r = &self.data.receivers[ri];
                        Receiver::new(r.x, r.y, r.z)
                    })
                    .collect();

                // Single Dipole1D call for all receivers in this group
                // Fortran ref: CSEM1D.f90 line 1938
                let mut fields_vec = dipole
                    .compute(&tx, &receivers, freq)
                    .map_err(|e| OccamError::Forward(e.to_string()))?;

                // Apply rotation per receiver
                // Fortran ref: CSEM1D.f90 lines 1976-2023
                for (local, &global) in unique_rx.iter().enumerate() {
                    let rx_def = &self.data.receivers[global];
                    if has_rotation(rx_def) {
                        rotate_fields(
                            &mut fields_vec[local],
                            rx_def.theta.unwrap_or(0.0),
                            rx_def.alpha.unwrap_or(0.0),
                            rx_def.beta.unwrap_or(0.0),
                        );
                    }
                }

                // Extract data values per data point
                // Fortran ref: CSEM1D.f90 lines 2080-2150
                let results: Vec<(usize, f64)> = data_indices
                    .iter()
                    .map(|&i| {
                        let dp = &self.data.data_points[i];
                        let local = rx_map[dp.rx_idx];
                        (i, extract_field_component(&fields_vec[local], dp.data_type))
                    })
                    .collect();

                Ok(results)
            })
            .collect::<Result<_>>()?;

        // Scatter results into output array
        for results in group_results {
            for (i, value) in results {
                computed[i] = value;
            }
        }

        Ok(())
    }

    /// Compute CSEM forward response with analytical Jacobian, grouped and parallelized.
    ///
    /// Same grouping as compute_csem() but also computes analytical sensitivities.
    /// Converts conductivity sensitivity to log10(resistivity) sensitivity and
    /// extracts real-valued Jacobian entries based on data type (Real/Imag/Amp/Phase).
    ///
    /// Fortran ref: CSEM1D.f90, computeFwd_CSEM (lines 1789-2262)
    /// - Lines 2171-2240: Jacobian extraction per data point
    /// - Dipole1D.f90 lines 2896-3375: hed_coef_nlayer derivative routines
    fn compute_csem_with_jacobian(
        &self,
        resistivities: &[f64],
        computed: &mut [f64],
        jacobian: &mut Jacobian,
        layer_to_param: &[Option<usize>],
    ) -> Result<()> {
        let conductivities = self.model.conductivities(resistivities);
        let depths = self.model.depths();
        let earth = LayeredEarth::new(depths, conductivities.clone())
            .map_err(|e| OccamError::Forward(e.to_string()))?;
        let dipole = Dipole1D::new(earth);
        let dipole_length = self.data.dipole_length.unwrap_or(0.0);

        let groups = self.build_csem_groups();

        // Process each (Tx, freq) group in parallel
        // Each returns: Vec<(data_idx, computed_value, jacobian_entries)>
        let group_results: Vec<Vec<(usize, f64, Vec<(usize, f64)>)>> = groups
            .par_iter()
            .map(|((tx_idx, freq_idx), data_indices)| -> Result<_> {
                let tx_def = &self.data.transmitters[*tx_idx];
                let freq = self.data.frequencies[*freq_idx];
                let tx = Transmitter {
                    x: tx_def.x,
                    y: tx_def.y,
                    z: tx_def.z,
                    azimuth: tx_def.azimuth,
                    dip: tx_def.dip,
                    moment: 1.0, // unit dipole moment (1 Am); CSEM data normalized to 1 Am by convention
                    length: dipole_length,
                };

                // Collect unique receivers
                let (unique_rx, rx_map) = collect_unique_receivers(data_indices, &self.data);
                let receivers: Vec<Receiver> = unique_rx
                    .iter()
                    .map(|&ri| {
                        let r = &self.data.receivers[ri];
                        Receiver::new(r.x, r.y, r.z)
                    })
                    .collect();

                // Single Dipole1D call with sensitivity for all receivers
                let (mut fields_vec, mut sens_vec) = dipole
                    .compute_with_sensitivity(&tx, &receivers, freq)
                    .map_err(|e| OccamError::Forward(e.to_string()))?;

                // Apply rotation and convert sensitivities per unique receiver
                // (once per receiver, not per data point)
                for (local, &global) in unique_rx.iter().enumerate() {
                    let rx_def = &self.data.receivers[global];
                    if has_rotation(rx_def) {
                        rotate_fields(
                            &mut fields_vec[local],
                            rx_def.theta.unwrap_or(0.0),
                            rx_def.alpha.unwrap_or(0.0),
                            rx_def.beta.unwrap_or(0.0),
                        );
                        rotate_sensitivity(
                            &mut sens_vec[local],
                            rx_def.theta.unwrap_or(0.0),
                            rx_def.alpha.unwrap_or(0.0),
                            rx_def.beta.unwrap_or(0.0),
                        );
                    }
                }

                // Convert all sensitivities from d/d(sigma) to d/d(log10_rho)
                let sens_log: Vec<EMFieldSensitivity> = sens_vec
                    .iter()
                    .map(|s| s.to_log_resistivity_sensitivity(&conductivities))
                    .collect();

                // Extract data + Jacobian per data point
                // Fortran ref: CSEM1D.f90 lines 2080-2240
                let results: Vec<(usize, f64, Vec<(usize, f64)>)> = data_indices
                    .iter()
                    .map(|&i| {
                        let dp = &self.data.data_points[i];
                        let local = rx_map[dp.rx_idx];
                        let fields = &fields_vec[local];
                        let sens = &sens_log[local];

                        let value = extract_field_component(fields, dp.data_type);

                        let jac_entries: Vec<(usize, f64)> = match dp.data_type {
                            // Polarization ellipse types need two field components
                            DataType::PEmax | DataType::PEmin | DataType::PBmax | DataType::PBmin => {
                                let is_max = matches!(dp.data_type, DataType::PEmax | DataType::PBmax);
                                let (f1, f2) = match dp.data_type {
                                    DataType::PEmax | DataType::PEmin => (fields.ex, fields.ey),
                                    _ => (fields.bx, fields.by),
                                };
                                layer_to_param
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(layer_idx, param_col)| {
                                        param_col.map(|col| {
                                            let (df1, df2) = match dp.data_type {
                                                DataType::PEmax | DataType::PEmin => {
                                                    (sens.dex_dsigma[layer_idx], sens.dey_dsigma[layer_idx])
                                                }
                                                _ => {
                                                    (sens.dbx_dsigma[layer_idx], sens.dby_dsigma[layer_idx])
                                                }
                                            };
                                            (col, compute_pe_deriv(f1, f2, df1, df2, is_max))
                                        })
                                    })
                                    .collect()
                            }
                            // All other types use single complex field + derivative
                            _ => {
                                let complex_field = get_complex_field(fields, dp.data_type);
                                layer_to_param
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(layer_idx, param_col)| {
                                        param_col.map(|col| {
                                            let complex_deriv =
                                                get_complex_sensitivity(sens, dp.data_type, layer_idx);
                                            let jac_val = extract_real_derivative(
                                                complex_field,
                                                complex_deriv,
                                                dp.data_type,
                                            );
                                            (col, jac_val)
                                        })
                                    })
                                    .collect()
                            }
                        };

                        (i, value, jac_entries)
                    })
                    .collect();

                Ok(results)
            })
            .collect::<Result<_>>()?;

        // Scatter results into output arrays
        for results in group_results {
            for (i, value, jac_entries) in results {
                computed[i] = value;
                for (col, val) in jac_entries {
                    jacobian.set(i, col, val);
                }
            }
        }

        Ok(())
    }

    /// Compute MT forward response and analytical Jacobian
    ///
    /// Uses analytical chain-rule sensitivity through Wait recursion
    /// (mt1d::compute_impedance_sensitivity_analytical) instead of finite differences.
    /// Find the MT receiver layer for a given receiver depth.
    ///
    /// Returns the index of the last layer whose top depth <= receiver depth.
    /// Sites on boundaries use the layer below (i.e., the deeper layer starting
    /// at that boundary).
    ///
    /// Fortran ref: CSEM1D.f90 lines 2797-2804
    ///   do j= 1,nlay1D
    ///       if (zRx_MT.ge.zlay1D(j))  then
    ///           iRxlayer = j
    ///       endif
    ///   enddo
    fn find_mt_receiver_layer(&self, rx_depth: f64) -> usize {
        let depths = self.model.depths();
        let mut rx_layer = 0;
        for (j, &d) in depths.iter().enumerate() {
            if rx_depth >= d {
                rx_layer = j;
            }
        }
        rx_layer
    }

    /// Build MT 1D model from layers at and below the receiver layer.
    ///
    /// The MT impedance is computed at the top of the receiver layer, which
    /// equals the receiver depth. Only layers from rx_layer downward are
    /// included in the Wait recursion.
    ///
    /// Works for both ocean-bottom (seafloor receiver, sea layer excluded)
    /// and land (surface receiver) MT sites.
    ///
    /// Fortran ref: CSEM1D.f90 lines 2809-2822
    ///   C = 1./SQRT(OMI*sig1D(nlay1d))
    ///   DO I = nlay1d-1, iRxlayer, -1
    ///       CK = SQRT(OMI*sig1D(i))
    ///       thick = zlay1d(i+1) - zlay1d(i)
    ///       C = EXP(-2.*CK*THICK)*(CK*C-1.)/(CK*C+1.)
    ///       C = (1.+C)/(1.-C)/CK
    ///   ENDDO
    fn build_mt_model(
        &self,
        all_resistivities: &[f64],
        rx_layer: usize,
    ) -> std::result::Result<(MT1DModel, Vec<f64>), OccamError> {
        let depths = self.model.depths();
        let n_full = depths.len();

        let mut mt_thicknesses = Vec::with_capacity(n_full - rx_layer);
        let mut mt_resistivities = Vec::with_capacity(n_full - rx_layer);

        for i in rx_layer..n_full {
            mt_resistivities.push(all_resistivities[i]);
            if i + 1 < n_full {
                mt_thicknesses.push(depths[i + 1] - depths[i]);
            } else {
                mt_thicknesses.push(f64::INFINITY);
            }
        }

        let mt_model = MT1DModel::new(mt_thicknesses, mt_resistivities.clone())
            .map_err(|e| OccamError::Forward(e.to_string()))?;

        Ok((mt_model, mt_resistivities))
    }

    /// Compute MT forward response and Jacobian.
    ///
    /// The MT impedance is evaluated at the receiver depth. Only model layers
    /// from the receiver layer downward contribute to the response. This
    /// correctly handles both ocean-bottom MT (receiver on seafloor, sea layer
    /// excluded from recursion) and land MT (receiver at surface).
    ///
    /// Fortran ref: CSEM1D.f90, computeFwd_MT subroutine (lines 2620-2755)
    ///   - Model setup: sig1D populated at lines 1720-1747 (all layers)
    ///   - Receiver layer: lines 2797-2804 (find iRxlayer)
    ///   - Wait recursion: lines 2809-2822 (DO I = nlay1d-1, iRxlayer, -1)
    ///   - Jacobian chain rule: lines 2674-2680 (dC/drho -> dC/dlog10(rho))
    ///   - Data extraction: lines 2695-2701 (getAppRes, getAppResDeriv)
    fn compute_mt_with_jacobian(
        &self,
        resistivities: &[f64],
        computed: &mut [f64],
        jacobian: &mut Jacobian,
        layer_to_param: &[Option<usize>],
    ) -> Result<()> {
        // Map free-parameter resistivities to full model layers
        let all_resistivities = self.model.resistivities(resistivities);

        // Find the MT receiver layer from receiver Z coordinate.
        // Fortran ref: CSEM1D.f90 line 2665: zRx_MT = zRxIn(dp(i,4))
        let first_mt = self.data.data_points.iter()
            .find(|dp| dp.data_type.is_mt())
            .ok_or_else(|| OccamError::Forward("No MT data points found".to_string()))?;
        let rx_depth = self.data.receivers[first_mt.rx_idx].z;
        let rx_layer = self.find_mt_receiver_layer(rx_depth);

        // Build MT model from receiver layer downward.
        // Only layers from iRxlayer to nlay1D enter the Wait recursion.
        // Fortran ref: CSEM1D.f90 lines 2809-2822
        let (mt_model, mt_resistivities) = self.build_mt_model(&all_resistivities, rx_layer)?;
        let mt = MT1D::new(mt_model.clone());

        // Cache sensitivities per frequency to avoid recomputation
        let mut sens_cache: std::collections::HashMap<u64, MTSensitivity> =
            std::collections::HashMap::new();

        for (i, dp) in self.data.data_points.iter().enumerate() {
            if !dp.data_type.is_mt() {
                continue;
            }

            let freq = self.data.frequencies[dp.freq_idx];

            let response = mt
                .compute(freq)
                .map_err(|e| OccamError::Forward(e.to_string()))?;
            computed[i] = extract_mt_component(&response, dp.data_type);

            // Get or compute analytical sensitivity (in log10 rho units)
            let freq_key = freq.to_bits();
            let log_sens = sens_cache.entry(freq_key).or_insert_with(|| {
                let sens = compute_impedance_sensitivity_analytical(&mt_model, freq)
                    .expect("MT analytical sensitivity failed");
                sens.to_log_sensitivity(&mt_resistivities)
            });

            // Fill Jacobian row: MT layer j maps to full model layer (j + rx_layer).
            // Fortran ref: CSEM1D.f90 lines 2674-2680 convert dC/drho to dC/dlog10(rho),
            //   then lines 2695-2701 extract getAppResDeriv / getPhaseDeriv per data type.
            // Only layers from rx_layer onward have nonzero sensitivity.
            for mt_layer in 0..mt_model.n_layers() {
                let full_layer = mt_layer + rx_layer;
                if let Some(&Some(col)) = layer_to_param.get(full_layer) {
                    let jac_val = extract_mt_derivative(log_sens, dp.data_type, mt_layer);
                    jacobian.set(i, col, jac_val);
                }
            }
        }

        Ok(())
    }

    /// Compute MT forward response (forward-only, no Jacobian).
    ///
    /// Same receiver-layer logic as compute_mt_with_jacobian. See that
    /// function's doc comment for full Fortran references.
    fn compute_mt(&self, resistivities: &[f64], computed: &mut [f64]) -> Result<()> {
        // Map free-parameter resistivities to full model layers
        let all_resistivities = self.model.resistivities(resistivities);

        // Find the MT receiver layer (CSEM1D.f90 lines 2797-2804)
        let first_mt = self.data.data_points.iter()
            .find(|dp| dp.data_type.is_mt())
            .ok_or_else(|| OccamError::Forward("No MT data points found".to_string()))?;
        let rx_depth = self.data.receivers[first_mt.rx_idx].z;
        let rx_layer = self.find_mt_receiver_layer(rx_depth);

        // Build MT model from receiver layer downward (CSEM1D.f90 lines 2809-2822)
        let (mt_model, _) = self.build_mt_model(&all_resistivities, rx_layer)?;
        let mt = MT1D::new(mt_model);

        for (i, dp) in self.data.data_points.iter().enumerate() {
            if !dp.data_type.is_mt() {
                continue;
            }

            let freq = self.data.frequencies[dp.freq_idx];
            let response = mt
                .compute(freq)
                .map_err(|e| OccamError::Forward(e.to_string()))?;
            computed[i] = extract_mt_component(&response, dp.data_type);
        }

        Ok(())
    }
}

/// Collect unique receiver indices from a group of data point indices.
///
/// Returns (unique_rx_indices, rx_map) where rx_map[global_rx_idx] = local index
/// into the unique_rx_indices vector.
///
/// Fortran ref: CSEM1D.f90 lines 1894-1905 (builds active receiver array)
fn collect_unique_receivers(
    data_indices: &[usize],
    data: &EMData,
) -> (Vec<usize>, Vec<usize>) {
    let mut unique_rx: Vec<usize> = data_indices
        .iter()
        .map(|&i| data.data_points[i].rx_idx)
        .collect();
    unique_rx.sort_unstable();
    unique_rx.dedup();

    // Build global_rx_idx -> local index map
    // Use a flat Vec indexed by rx_idx (fast lookup, small memory for typical models)
    let max_rx = unique_rx.iter().copied().max().unwrap_or(0);
    let mut rx_map = vec![0usize; max_rx + 1];
    for (local, &global) in unique_rx.iter().enumerate() {
        rx_map[global] = local;
    }

    (unique_rx, rx_map)
}

/// Apply 3D receiver rotation to EM fields
///
/// Rotation matrix: RotR = Rotz(theta) * Roty(alpha) * Rotx(beta)
/// Applied as row-vector * matrix: [Ex', Ey', Ez'] = [Ex, Ey, Ez] * RotR
///
/// Fortran ref: CSEM1D.f90 lines 1976-2023
fn rotate_fields(fields: &mut EMFields, theta: f64, alpha: f64, beta: f64) {
    let deg2rad = PI / 180.0;
    let cct = (theta * deg2rad).cos();
    let sst = (theta * deg2rad).sin();
    let cca = (alpha * deg2rad).cos();
    let ssa = (alpha * deg2rad).sin();
    let ccb = (beta * deg2rad).cos();
    let ssb = (beta * deg2rad).sin();

    // RotR = Rotz(theta) * Roty(alpha) * Rotx(beta)
    let r = [
        [
            cct * cca,
            cct * ssa * ssb - sst * ccb,
            cct * ssa * ccb + sst * ssb,
        ],
        [
            sst * cca,
            sst * ssa * ssb + cct * ccb,
            sst * ssa * ccb - cct * ssb,
        ],
        [-ssa, cca * ssb, cca * ccb],
    ];

    let apply_rotation =
        |x: Complex64, y: Complex64, z: Complex64| -> (Complex64, Complex64, Complex64) {
            let x_new = x * r[0][0] + y * r[1][0] + z * r[2][0];
            let y_new = x * r[0][1] + y * r[1][1] + z * r[2][1];
            let z_new = x * r[0][2] + y * r[1][2] + z * r[2][2];
            (x_new, y_new, z_new)
        };

    let (ex, ey, ez) = apply_rotation(fields.ex, fields.ey, fields.ez);
    fields.ex = ex;
    fields.ey = ey;
    fields.ez = ez;

    let (bx, by, bz) = apply_rotation(fields.bx, fields.by, fields.bz);
    fields.bx = bx;
    fields.by = by;
    fields.bz = bz;
}

/// Apply 3D receiver rotation to field sensitivities
fn rotate_sensitivity(sens: &mut EMFieldSensitivity, theta: f64, alpha: f64, beta: f64) {
    let deg2rad = PI / 180.0;
    let cct = (theta * deg2rad).cos();
    let sst = (theta * deg2rad).sin();
    let cca = (alpha * deg2rad).cos();
    let ssa = (alpha * deg2rad).sin();
    let ccb = (beta * deg2rad).cos();
    let ssb = (beta * deg2rad).sin();

    let r = [
        [
            cct * cca,
            cct * ssa * ssb - sst * ccb,
            cct * ssa * ccb + sst * ssb,
        ],
        [
            sst * cca,
            sst * ssa * ssb + cct * ccb,
            sst * ssa * ccb - cct * ssb,
        ],
        [-ssa, cca * ssb, cca * ccb],
    ];

    for k in 0..sens.dex_dsigma.len() {
        let (dx, dy, dz) = (
            sens.dex_dsigma[k],
            sens.dey_dsigma[k],
            sens.dez_dsigma[k],
        );
        sens.dex_dsigma[k] = dx * r[0][0] + dy * r[1][0] + dz * r[2][0];
        sens.dey_dsigma[k] = dx * r[0][1] + dy * r[1][1] + dz * r[2][1];
        sens.dez_dsigma[k] = dx * r[0][2] + dy * r[1][2] + dz * r[2][2];

        let (dbx, dby, dbz) = (
            sens.dbx_dsigma[k],
            sens.dby_dsigma[k],
            sens.dbz_dsigma[k],
        );
        sens.dbx_dsigma[k] = dbx * r[0][0] + dby * r[1][0] + dbz * r[2][0];
        sens.dby_dsigma[k] = dbx * r[0][1] + dby * r[1][1] + dbz * r[2][1];
        sens.dbz_dsigma[k] = dbx * r[0][2] + dby * r[1][2] + dbz * r[2][2];
    }
}

/// Check if receiver has non-zero rotation
fn has_rotation(rx_def: &occam_io::Receiver) -> bool {
    let theta = rx_def.theta.unwrap_or(0.0);
    let alpha = rx_def.alpha.unwrap_or(0.0);
    let beta = rx_def.beta.unwrap_or(0.0);
    theta != 0.0 || alpha != 0.0 || beta != 0.0
}

/// Compute polarization ellipse axis length
///
/// Given two complex horizontal field components (e.g., Ex and Ey), computes
/// the semi-major or semi-minor axis of the polarization ellipse traced by the
/// field vector over one period.
///
/// Fortran ref: CSEM1D.f90 lines 363-413 (getPE)
fn compute_pe(e1: Complex64, e2: Complex64, is_max: bool) -> f64 {
    let x1 = e1.re;
    let y1 = e1.im;
    let x2 = e2.re;
    let y2 = e2.im;

    let a = -(x1 * y1 + x2 * y2);
    let b = 0.5 * (y1 * y1 + y2 * y2 - x1 * x1 - x2 * x2);
    let phi = a.atan2(b) / 2.0;

    let compute_axis = |theta: f64| -> f64 {
        let s = theta.sin();
        let c = theta.cos();
        let val = (x1 * x1 + x2 * x2) * c * c
            + (y1 * y1 + y2 * y2) * s * s
            + 2.0 * s * c * (x1 * y1 + x2 * y2);
        val.abs().sqrt()
    };

    let p1 = compute_axis(phi);
    let p2 = compute_axis(phi + std::f64::consts::FRAC_PI_2);

    if is_max {
        p1.max(p2)
    } else {
        p1.min(p2)
    }
}

/// Compute derivative of polarization ellipse axis length
///
/// Fortran ref: CSEM1D.f90 lines 418-490 (getPEDeriv)
/// Note: Fortran ignores dphi/dsig because angle is stationary at critical point.
fn compute_pe_deriv(
    e1: Complex64,
    e2: Complex64,
    de1: Complex64,
    de2: Complex64,
    is_max: bool,
) -> f64 {
    let x1 = e1.re;
    let y1 = e1.im;
    let x2 = e2.re;
    let y2 = e2.im;
    let dx1 = de1.re;
    let dy1 = de1.im;
    let dx2 = de2.re;
    let dy2 = de2.im;

    let a = -(x1 * y1 + x2 * y2);
    let b = 0.5 * (y1 * y1 + y2 * y2 - x1 * x1 - x2 * x2);
    let phi = a.atan2(b) / 2.0;

    let compute_axis_and_deriv = |theta: f64| -> (f64, f64) {
        let s = theta.sin();
        let c = theta.cos();
        let val = (x1 * x1 + x2 * x2) * c * c
            + (y1 * y1 + y2 * y2) * s * s
            + 2.0 * s * c * (x1 * y1 + x2 * y2);
        let p = val.abs().sqrt();
        if p < 1e-300 {
            return (p, 0.0);
        }
        let dp = (c * c * (x1 * dx1 + x2 * dx2)
            + s * s * (y1 * dy1 + y2 * dy2)
            + s * c * (x1 * dy1 + dx1 * y1 + x2 * dy2 + dx2 * y2))
            / p;
        (p, dp)
    };

    let (p1, d1) = compute_axis_and_deriv(phi);
    let (p2, d2) = compute_axis_and_deriv(phi + std::f64::consts::FRAC_PI_2);

    if p1 >= p2 {
        if is_max { d1 } else { d2 }
    } else {
        if is_max { d2 } else { d1 }
    }
}

/// Extract the appropriate field component based on data type
fn extract_field_component(fields: &EMFields, data_type: DataType) -> f64 {
    match data_type {
        DataType::ExReal => fields.ex.re,
        DataType::ExImag => fields.ex.im,
        DataType::EyReal => fields.ey.re,
        DataType::EyImag => fields.ey.im,
        DataType::EzReal => fields.ez.re,
        DataType::EzImag => fields.ez.im,
        DataType::BxReal => fields.bx.re,
        DataType::BxImag => fields.bx.im,
        DataType::ByReal => fields.by.re,
        DataType::ByImag => fields.by.im,
        DataType::BzReal => fields.bz.re,
        DataType::BzImag => fields.bz.im,
        DataType::ExAmp => fields.ex.norm(),
        DataType::ExPhase => fields.ex.arg().to_degrees(),
        DataType::EyAmp => fields.ey.norm(),
        DataType::EyPhase => fields.ey.arg().to_degrees(),
        DataType::EzAmp => fields.ez.norm(),
        DataType::EzPhase => fields.ez.arg().to_degrees(),
        DataType::BxAmp => fields.bx.norm(),
        DataType::BxPhase => fields.bx.arg().to_degrees(),
        DataType::ByAmp => fields.by.norm(),
        DataType::ByPhase => fields.by.arg().to_degrees(),
        DataType::BzAmp => fields.bz.norm(),
        DataType::BzPhase => fields.bz.arg().to_degrees(),
        DataType::PEmax => compute_pe(fields.ex, fields.ey, true),
        DataType::PEmin => compute_pe(fields.ex, fields.ey, false),
        DataType::PBmax => compute_pe(fields.bx, fields.by, true),
        DataType::PBmin => compute_pe(fields.bx, fields.by, false),
        _ => 0.0, // MT types handled separately
    }
}

/// Extract MT component based on data type
fn extract_mt_component(response: &MTResponse, data_type: DataType) -> f64 {
    match data_type {
        // For 1D, Zxy = -Zyx = Z, Zxx = Zyy = 0
        DataType::RhoXX | DataType::RhoYY => 0.0,
        DataType::RhoXY | DataType::RhoYX => response.apparent_resistivity,
        DataType::PhiXX | DataType::PhiYY => 0.0,
        DataType::PhiXY | DataType::PhiYX => response.phase,
        DataType::ZxxReal | DataType::ZxxImag | DataType::ZyyReal | DataType::ZyyImag => 0.0,
        DataType::ZxyReal => response.impedance.re,
        DataType::ZxyImag => response.impedance.im,
        DataType::ZyxReal => -response.impedance.re,
        DataType::ZyxImag => -response.impedance.im,
        _ => 0.0,
    }
}

/// Extract MT Jacobian entry for a given data type and layer
fn extract_mt_derivative(sens: &MTSensitivity, data_type: DataType, layer: usize) -> f64 {
    match data_type {
        DataType::RhoXY | DataType::RhoYX => sens.drho_a_drho[layer],
        DataType::PhiXY | DataType::PhiYX => sens.dphase_drho[layer],
        DataType::ZxyReal => sens.dz_drho[layer].re,
        DataType::ZxyImag => sens.dz_drho[layer].im,
        DataType::ZyxReal => -sens.dz_drho[layer].re,
        DataType::ZyxImag => -sens.dz_drho[layer].im,
        _ => 0.0, // XX/YY components are zero in 1D
    }
}

/// Get the complex field value for a given data type
fn get_complex_field(fields: &EMFields, data_type: DataType) -> Complex64 {
    match data_type {
        DataType::ExReal | DataType::ExImag | DataType::ExAmp | DataType::ExPhase => fields.ex,
        DataType::EyReal | DataType::EyImag | DataType::EyAmp | DataType::EyPhase => fields.ey,
        DataType::EzReal | DataType::EzImag | DataType::EzAmp | DataType::EzPhase => fields.ez,
        DataType::BxReal | DataType::BxImag | DataType::BxAmp | DataType::BxPhase => fields.bx,
        DataType::ByReal | DataType::ByImag | DataType::ByAmp | DataType::ByPhase => fields.by,
        DataType::BzReal | DataType::BzImag | DataType::BzAmp | DataType::BzPhase => fields.bz,
        _ => Complex64::new(0.0, 0.0),
    }
}

/// Get the complex sensitivity for a given data type and layer
fn get_complex_sensitivity(
    sens: &EMFieldSensitivity,
    data_type: DataType,
    layer: usize,
) -> Complex64 {
    match data_type {
        DataType::ExReal | DataType::ExImag | DataType::ExAmp | DataType::ExPhase => {
            sens.dex_dsigma[layer]
        }
        DataType::EyReal | DataType::EyImag | DataType::EyAmp | DataType::EyPhase => {
            sens.dey_dsigma[layer]
        }
        DataType::EzReal | DataType::EzImag | DataType::EzAmp | DataType::EzPhase => {
            sens.dez_dsigma[layer]
        }
        DataType::BxReal | DataType::BxImag | DataType::BxAmp | DataType::BxPhase => {
            sens.dbx_dsigma[layer]
        }
        DataType::ByReal | DataType::ByImag | DataType::ByAmp | DataType::ByPhase => {
            sens.dby_dsigma[layer]
        }
        DataType::BzReal | DataType::BzImag | DataType::BzAmp | DataType::BzPhase => {
            sens.dbz_dsigma[layer]
        }
        _ => Complex64::new(0.0, 0.0),
    }
}

/// Extract real-valued derivative from complex field and its derivative
///
/// For real/imag data: take Re/Im of complex derivative
/// For amplitude: d|z|/dp = (Re(z)*Re(dz) + Im(z)*Im(dz)) / |z|
/// For phase: d(arg(z))/dp = (Re(z)*Im(dz) - Im(z)*Re(dz)) / |z|^2 * (180/pi)
fn extract_real_derivative(z: Complex64, dz: Complex64, data_type: DataType) -> f64 {
    match data_type {
        DataType::ExReal | DataType::EyReal | DataType::EzReal | DataType::BxReal
        | DataType::ByReal | DataType::BzReal => dz.re,

        DataType::ExImag | DataType::EyImag | DataType::EzImag | DataType::BxImag
        | DataType::ByImag | DataType::BzImag => dz.im,

        DataType::ExAmp | DataType::EyAmp | DataType::EzAmp | DataType::BxAmp
        | DataType::ByAmp | DataType::BzAmp => {
            let amp = z.norm();
            if amp < 1e-300 {
                0.0
            } else {
                (z.re * dz.re + z.im * dz.im) / amp
            }
        }

        DataType::ExPhase | DataType::EyPhase | DataType::EzPhase | DataType::BxPhase
        | DataType::ByPhase | DataType::BzPhase => {
            let norm_sq = z.norm_sqr();
            if norm_sq < 1e-300 {
                0.0
            } else {
                (z.re * dz.im - z.im * dz.re) / norm_sq * (180.0 / PI)
            }
        }

        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_misfit_calculation() {
        let observed = vec![1.0, 2.0, 3.0];
        let computed = vec![1.1, 2.0, 2.9];
        let errors = vec![0.1, 0.1, 0.1];

        let (rms, _chi_sq) = ForwardResult::compute_misfit(&observed, &computed, &errors);

        // residuals: (0.1/0.1)^2 + 0 + (0.1/0.1)^2 = 2
        // chi_sq/n = 2/3
        // rms = sqrt(2/3) ≈ 0.816
        assert!((rms - 0.816).abs() < 0.01);
    }

    #[test]
    fn test_compute_pe_linearly_polarized() {
        // Linearly polarized: Ex = 3+0i, Ey = 4+0i → ellipse collapses to a line
        // Major axis = sqrt(3²+4²) = 5, minor axis = 0
        let ex = Complex64::new(3.0, 0.0);
        let ey = Complex64::new(4.0, 0.0);
        let pmax = compute_pe(ex, ey, true);
        let pmin = compute_pe(ex, ey, false);
        assert!((pmax - 5.0).abs() < 1e-12, "pmax = {}, expected 5.0", pmax);
        assert!(pmin.abs() < 1e-12, "pmin = {}, expected 0.0", pmin);
    }

    #[test]
    fn test_compute_pe_circularly_polarized() {
        // Circularly polarized: Ex = 1+0i, Ey = 0+1i → circle of radius 1
        // Major = minor = 1
        let ex = Complex64::new(1.0, 0.0);
        let ey = Complex64::new(0.0, 1.0);
        let pmax = compute_pe(ex, ey, true);
        let pmin = compute_pe(ex, ey, false);
        assert!((pmax - 1.0).abs() < 1e-12, "pmax = {}, expected 1.0", pmax);
        assert!((pmin - 1.0).abs() < 1e-12, "pmin = {}, expected 1.0", pmin);
    }

    #[test]
    fn test_compute_pe_elliptical() {
        // Ex = 2+0i, Ey = 0+1i → ellipse with semi-axes 2 and 1
        let ex = Complex64::new(2.0, 0.0);
        let ey = Complex64::new(0.0, 1.0);
        let pmax = compute_pe(ex, ey, true);
        let pmin = compute_pe(ex, ey, false);
        assert!((pmax - 2.0).abs() < 1e-12, "pmax = {}, expected 2.0", pmax);
        assert!((pmin - 1.0).abs() < 1e-12, "pmin = {}, expected 1.0", pmin);
    }

    #[test]
    fn test_compute_pe_deriv_fd() {
        // Verify analytical derivative against finite differences
        let ex = Complex64::new(1.5, 0.3);
        let ey = Complex64::new(-0.7, 1.2);
        // Arbitrary derivative directions
        let dex = Complex64::new(0.1, -0.05);
        let dey = Complex64::new(0.02, 0.08);

        let eps = 1e-7;

        for is_max in [true, false] {
            let analytical = compute_pe_deriv(ex, ey, dex, dey, is_max);

            // FD: PE(e + eps*de) - PE(e - eps*de) / (2*eps)
            let pe_plus = compute_pe(ex + dex * eps, ey + dey * eps, is_max);
            let pe_minus = compute_pe(ex - dex * eps, ey - dey * eps, is_max);
            let fd = (pe_plus - pe_minus) / (2.0 * eps);

            let label = if is_max { "pmax" } else { "pmin" };
            assert!(
                (analytical - fd).abs() < 1e-5,
                "{}: analytical={}, fd={}, diff={}",
                label,
                analytical,
                fd,
                (analytical - fd).abs()
            );
        }
    }
}
