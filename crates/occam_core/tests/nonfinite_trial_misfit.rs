//! Regression test: non-finite trial misfits in the Lagrange search.
//!
//! The data below are synthetic land-MT apparent resistivities and phases
//! that no 1D model can fit (phase far from what the resistivity curve
//! allows). When the search tries a small Lagrange multiplier, the
//! under-regularised solve returns log10(resistivity) values of several
//! hundred, 10^x overflows to infinity and the forward response, hence the
//! misfit, is NaN.
//!
//! Without treating such a trial as failed, the bracketing walks through the
//! NaN values down to multipliers where the normal matrix cannot be
//! factorised, and the run aborts with "Cholesky decomposition failed" after
//! iteration 6. With only the trials guarded, iteration 11 (where every trial
//! overflows) is accepted with a NaN misfit and the next iteration aborts.

use occam_core::{OccamConfig, OccamInversion};
use occam_io::{EMData, LayeredModel};

const DATA: &str = "\
Format:  EMData_1.1
# Transmitters:   1
  0  0  0  0  0
# Frequencies:    4
  10
  1
  0.01
  0.001
Phase Convention: lag
# Receivers:      1
  0  0  0  0  0  0
# Data:       8
  103  1  0  1  100  10
  104  1  0  1  70  2.86479
  103  2  0  1  80  8
  104  2  0  1  60  2.86479
  103  3  0  1  60  6
  104  3  0  1  27  2.86479
  103  4  0  1  40  4
  104  4  0  1  45  2.86479
";

const MODEL: &str = "\
Format: Resistivity1DMod_1.0
#LAYERS:    16
  -100000  1d12  0  0  0
  0.00  ?  1  0  0
  25.00  ?  1  0  0
  43.13  ?  1  0  0
  74.41  ?  1  0  0
  128.39  ?  1  0  0
  221.50  ?  1  0  0
  382.16  ?  1  0  0
  659.33  ?  1  0  0
  1137.52  ?  1  0  0
  1962.55  ?  1  0  0
  3385.95  ?  1  0  0
  5841.72  ?  1  0  0
  10078.62  ?  1  0  0
  17388.46  ?  1  0  0
  30000.00  ?  1  0  0
";

#[test]
fn nonfinite_trial_misfit_does_not_abort_the_inversion() {
    let data = EMData::read(DATA.as_bytes()).unwrap();
    let model = LayeredModel::read(MODEL.as_bytes()).unwrap();
    let n_params = model.n_free_params();

    let config = OccamConfig {
        max_iterations: 40,
        ..Default::default()
    };
    let mut inversion = OccamInversion::new(data, model, config).unwrap();

    // 100 ohm-m starting halfspace (parameters are log10 resistivity)
    let results = inversion
        .run(&vec![2.0; n_params])
        .expect("inversion must not abort on a non-finite trial misfit");

    // The run used to abort after iteration 6 (and, with only the trials
    // guarded, after a NaN iteration 11).
    let last = results.last().unwrap();
    assert!(
        last.iteration > 11,
        "stopped at iteration {} with status {:?}",
        last.iteration,
        last.status
    );

    for r in &results {
        assert!(
            r.rms_misfit.is_finite(),
            "iteration {}: misfit {}",
            r.iteration,
            r.rms_misfit
        );
        assert!(
            r.parameters.iter().all(|p| p.is_finite()),
            "iteration {}: non-finite model parameter",
            r.iteration
        );
    }
}
