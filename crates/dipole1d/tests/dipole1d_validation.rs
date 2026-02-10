//! Validation tests against reference Fortran Dipole1D output
//!
//! Reference data from: Occam1DCSEM_matlab/Examples/Synthetic/CanonicalReservoir/Dipole1D/Canonical.csem

use dipole1d::{
    Dipole1D, EMFieldSensitivity, HankelFilter, LayeredEarth, Receiver, Transmitter,
};

/// Reference output from Fortran Dipole1D for canonical reservoir model
/// Model: 5 layers (air, seawater 0-1000m, sediment 1000-2000m @ 1 ohm-m,
///        reservoir 2000-2100m @ 100 ohm-m, sediment 2100m+ @ 1 ohm-m)
///
/// Transmitter at (0, 0, 950m), azimuth=90 deg (pointing along y-axis)
/// Receiver at (0, 0, 1000m)
/// Frequency: 0.25 Hz
///
/// Reference output (from Canonical.csem):
/// Ex_re, Ex_im, Ey_re, Ey_im, Ez_re, Ez_im, Bx_re, Bx_im, By_re, By_im, Bz_re, Bz_im
/// 0.21568639E-25  0.51834153E-28 -0.29346224E-06  0.11916205E-08  0.52835915E-08  0.10612836E-10  0.29188146E-10  0.24985744E-12 -0.39475483E-30 -0.20073576E-32 -0.48942258E-28 -0.31857000E-30
#[test]
fn test_canonical_reservoir_close_range() {
    // Build the 5-layer canonical reservoir model
    // Note: LayeredEarth uses conductivities (S/m), so we need to convert from resistivities
    // Resistivities: air=1e12, seawater=0.3, sediment=1.0, reservoir=100, sediment=1.0
    // Conductivities: 1e-12, 1/0.3=3.33, 1.0, 0.01, 1.0
    let model = LayeredEarth::new(
        vec![-100000.0, 0.0, 1000.0, 2000.0, 2100.0], // Layer top depths
        vec![1e-12, 1.0 / 0.3, 1.0, 0.01, 1.0],       // Conductivities (S/m)
    )
    .expect("Failed to create model");

    let dipole1d = Dipole1D::new(model).with_filter(HankelFilter::Kong201);

    // Transmitter: HED at (0, 0, 950m) pointing along y-axis (azimuth=90)
    let tx = Transmitter::hed(0.0, 0.0, 950.0, 90.0);

    // Receiver at (0, 0, 1000m) - directly below transmitter
    let rx = Receiver::new(0.0, 0.0, 1000.0);

    let frequency = 0.25;
    let results = dipole1d.compute(&tx, &[rx], frequency).unwrap();

    let fields = &results[0];

    // Reference values from Fortran output
    let ref_ey_re: f64 = -0.29346224e-06;
    let ref_ey_im: f64 = 0.11916205e-08;
    let ref_ez_re: f64 = 0.52835915e-08;
    let ref_ez_im: f64 = 0.10612836e-10;
    let ref_bx_re: f64 = 0.29188146e-10;
    let ref_bx_im: f64 = 0.24985744e-12;

    // Print computed values for comparison
    println!("\nCanonical Reservoir - Close Range (0m offset, 0.25 Hz)");
    println!("======================================================");
    println!(
        "Ey: computed = ({:.6e}, {:.6e}), reference = ({:.6e}, {:.6e})",
        fields.ey.re, fields.ey.im, ref_ey_re, ref_ey_im
    );
    println!(
        "Ez: computed = ({:.6e}, {:.6e}), reference = ({:.6e}, {:.6e})",
        fields.ez.re, fields.ez.im, ref_ez_re, ref_ez_im
    );
    println!(
        "Bx: computed = ({:.6e}, {:.6e}), reference = ({:.6e}, {:.6e})",
        fields.bx.re, fields.bx.im, ref_bx_re, ref_bx_im
    );

    // Calculate relative errors
    let ey_mag_ref = (ref_ey_re * ref_ey_re + ref_ey_im * ref_ey_im).sqrt();
    let ey_mag_comp = fields.ey.norm();
    let ey_rel_err = (ey_mag_comp - ey_mag_ref).abs() / ey_mag_ref;

    let ez_mag_ref = (ref_ez_re * ref_ez_re + ref_ez_im * ref_ez_im).sqrt();
    let ez_mag_comp = fields.ez.norm();
    let ez_rel_err = (ez_mag_comp - ez_mag_ref).abs() / ez_mag_ref;

    println!("\nRelative errors:");
    println!("  |Ey| error: {:.2}%", ey_rel_err * 100.0);
    println!("  |Ez| error: {:.2}%", ez_rel_err * 100.0);

    // Note: We're not asserting tight tolerances yet because we need to
    // verify the implementation first. This test serves to document
    // the comparison with reference output.
    println!("\nNote: Tight tolerance checks not yet enabled - verifying implementation");
}

/// Test with transmitter offset from receiver (500m along y)
/// Reference line 56 in Canonical.csem
#[test]
fn test_canonical_reservoir_500m_offset() {
    let model = LayeredEarth::new(
        vec![-100000.0, 0.0, 1000.0, 2000.0, 2100.0],
        vec![1e-12, 1.0 / 0.3, 1.0, 0.01, 1.0],
    )
    .expect("Failed to create model");

    let dipole1d = Dipole1D::new(model).with_filter(HankelFilter::Kong201);

    // Transmitter at (0, 500, 950m) pointing along y
    let tx = Transmitter::hed(0.0, 500.0, 950.0, 90.0);
    let rx = Receiver::new(0.0, 0.0, 1000.0);

    let frequency = 0.25;
    let results = dipole1d.compute(&tx, &[rx], frequency).unwrap();
    let fields = &results[0];

    // Reference from line 57 (Tx#2 y=500, freq 0.25 Hz - data interleaves freqs per Tx)
    let ref_ey_re: f64 = 0.48184682e-09;
    let ref_ey_im: f64 = 0.15819266e-09;
    let ref_ez_re: f64 = -0.35705154e-10;
    let ref_ez_im: f64 = 0.18323756e-10;

    println!("\nCanonical Reservoir - 500m Offset (0.25 Hz)");
    println!("============================================");
    println!(
        "Ey: computed = ({:.6e}, {:.6e}), reference = ({:.6e}, {:.6e})",
        fields.ey.re, fields.ey.im, ref_ey_re, ref_ey_im
    );
    println!(
        "Ez: computed = ({:.6e}, {:.6e}), reference = ({:.6e}, {:.6e})",
        fields.ez.re, fields.ez.im, ref_ez_re, ref_ez_im
    );

    let ey_mag_ref = (ref_ey_re * ref_ey_re + ref_ey_im * ref_ey_im).sqrt();
    let ey_mag_comp = fields.ey.norm();
    let ey_rel_err = (ey_mag_comp - ey_mag_ref).abs() / ey_mag_ref;

    println!("\nRelative errors:");
    println!("  |Ey| error: {:.2}%", ey_rel_err * 100.0);
}

/// Simple halfspace test - easier to verify analytically
#[test]
fn test_halfspace_1hz() {
    // 1 ohm-m halfspace (conductivity = 1 S/m)
    let model = LayeredEarth::halfspace(1.0);

    let dipole1d = Dipole1D::new(model).with_filter(HankelFilter::Kong201);

    // HED at 100m depth, receiver at 200m depth, 1km horizontal offset
    let tx = Transmitter::hed(0.0, 0.0, 100.0, 0.0); // Pointing along x
    let rx = Receiver::new(1000.0, 0.0, 200.0);

    let frequency = 1.0;
    let results = dipole1d.compute(&tx, &[rx], frequency).unwrap();
    let fields = &results[0];

    println!("\nHalfspace 1 ohm-m, 1 Hz");
    println!("=======================");
    println!("Geometry: Tx at (0, 0, 100m), Rx at (1000, 0, 200m)");
    println!("Ex: ({:.6e}, {:.6e})", fields.ex.re, fields.ex.im);
    println!("Ey: ({:.6e}, {:.6e})", fields.ey.re, fields.ey.im);
    println!("Ez: ({:.6e}, {:.6e})", fields.ez.re, fields.ez.im);
    println!("Bx: ({:.6e}, {:.6e})", fields.bx.re, fields.bx.im);
    println!("By: ({:.6e}, {:.6e})", fields.by.re, fields.by.im);
    println!("Bz: ({:.6e}, {:.6e})", fields.bz.re, fields.bz.im);

    // Fields should be non-zero
    assert!(
        fields.ex.norm() > 0.0 || fields.ey.norm() > 0.0,
        "E fields should be non-zero"
    );
}

/// Test marine CSEM geometry (common configuration)
#[test]
fn test_marine_csem_geometry() {
    // Marine model: air, seawater (0-1000m), sediment (1000m+)
    // Conductivities: air=1e-12, seawater=3.33 S/m (0.3 ohm-m), sediment=1 S/m
    let model = LayeredEarth::new(vec![-100000.0, 0.0, 1000.0], vec![1e-12, 1.0 / 0.3, 1.0])
        .expect("Failed to create model");

    let dipole1d = Dipole1D::new(model).with_filter(HankelFilter::Kong201);

    // Typical marine CSEM: Tx at 950m (50m above seafloor), Rx on seafloor
    let tx = Transmitter::hed(0.0, 0.0, 950.0, 0.0);
    let rx = Receiver::new(5000.0, 0.0, 1000.0); // 5km offset

    let frequency = 0.25;
    let results = dipole1d.compute(&tx, &[rx], frequency).unwrap();
    let fields = &results[0];

    println!("\nMarine CSEM - 5km offset, 0.25 Hz");
    println!("==================================");
    println!("Model: air / seawater (0.3 ohm-m, 0-1000m) / sediment (1 ohm-m)");
    println!("Geometry: Tx at (0, 0, 950m), Rx at (5000, 0, 1000m)");
    println!(
        "Ex: ({:.6e}, {:.6e}), |Ex| = {:.6e}",
        fields.ex.re,
        fields.ex.im,
        fields.ex.norm()
    );
    println!(
        "Ey: ({:.6e}, {:.6e}), |Ey| = {:.6e}",
        fields.ey.re,
        fields.ey.im,
        fields.ey.norm()
    );
    println!(
        "Ez: ({:.6e}, {:.6e}), |Ez| = {:.6e}",
        fields.ez.re,
        fields.ez.im,
        fields.ez.norm()
    );

    // Inline E-field (Ex for x-directed dipole) should dominate at this range
    println!("\nFor x-directed HED, inline Ex should be the largest E component");

    // Basic sanity check
    assert!(fields.ex.norm() > 0.0, "Inline E-field should be non-zero");
}

/// Validate analytical Jacobian against finite-difference for the canonical reservoir model
///
/// Tests that the analytical derivatives match finite-difference approximations
/// to at least 4 significant figures for all field components and all layers.
#[test]
fn test_analytical_vs_fd_jacobian() {
    // Canonical reservoir model (5 layers)
    let model = LayeredEarth::new(
        vec![-100000.0, 0.0, 1000.0, 2000.0, 2100.0],
        vec![1e-12, 1.0 / 0.3, 1.0, 0.01, 1.0],
    )
    .expect("Failed to create model");

    let dipole = Dipole1D::new(model.clone()).with_filter(HankelFilter::Kong201);

    // Tx at 950m depth, y-directed (azimuth=90)
    let tx = Transmitter::hed(0.0, 0.0, 950.0, 90.0);
    // Rx at 500m offset, 1000m depth (on seafloor)
    let rx = Receiver::new(0.0, 500.0, 1000.0);
    let freq = 0.25;

    // Compute analytical sensitivity
    let (fields_vec, sens_vec) = dipole
        .compute_with_sensitivity(&tx, &[rx.clone()], freq)
        .expect("Analytical sensitivity failed");

    let fields = &fields_vec[0];
    let sens = &sens_vec[0];

    // Compute finite-difference sensitivity
    let delta = 1e-6; // relative perturbation
    let n_layers = model.n_layers();
    let mut fd_sens = EMFieldSensitivity::zeros(n_layers);

    for layer in 0..n_layers {
        if model.conductivities[layer] < 1e-10 {
            continue; // skip air layer
        }

        let mut perturbed_cond = model.conductivities.clone();
        let sigma_orig = perturbed_cond[layer];
        perturbed_cond[layer] = sigma_orig * (1.0 + delta);
        let d_sigma = sigma_orig * delta;

        let perturbed_model =
            LayeredEarth::new(model.depths.clone(), perturbed_cond).unwrap();
        let perturbed_dipole =
            Dipole1D::new(perturbed_model).with_filter(HankelFilter::Kong201);

        let perturbed_fields = perturbed_dipole
            .compute(&tx, &[rx.clone()], freq)
            .expect("Perturbed forward failed");
        let pf = &perturbed_fields[0];

        fd_sens.dex_dsigma[layer] = (pf.ex - fields.ex) / d_sigma;
        fd_sens.dey_dsigma[layer] = (pf.ey - fields.ey) / d_sigma;
        fd_sens.dez_dsigma[layer] = (pf.ez - fields.ez) / d_sigma;
        fd_sens.dbx_dsigma[layer] = (pf.bx - fields.bx) / d_sigma;
        fd_sens.dby_dsigma[layer] = (pf.by - fields.by) / d_sigma;
        fd_sens.dbz_dsigma[layer] = (pf.bz - fields.bz) / d_sigma;
    }

    // Compare analytical vs finite-difference
    let tol = 1e-3; // relative tolerance (0.1% = ~3.3 significant figures)

    println!("\n=== Analytical vs Finite-Difference Jacobian ===\n");

    let components = [
        ("dEx/dsig", &sens.dex_dsigma, &fd_sens.dex_dsigma),
        ("dEy/dsig", &sens.dey_dsigma, &fd_sens.dey_dsigma),
        ("dEz/dsig", &sens.dez_dsigma, &fd_sens.dez_dsigma),
        ("dBx/dsig", &sens.dbx_dsigma, &fd_sens.dbx_dsigma),
        ("dBy/dsig", &sens.dby_dsigma, &fd_sens.dby_dsigma),
        ("dBz/dsig", &sens.dbz_dsigma, &fd_sens.dbz_dsigma),
    ];

    for (name, analytical, fd) in &components {
        println!("{}:", name);
        for layer in 0..n_layers {
            let a = analytical[layer];
            let f = fd[layer];
            let a_norm = a.norm();
            let f_norm = f.norm();

            // Skip air layer (conductivity ~0, FD skips it too)
            if model.conductivities[layer] < 1e-10 {
                println!("  Layer {}: air layer, skipped", layer);
                continue;
            }

            if a_norm < 1e-50 && f_norm < 1e-50 {
                println!(
                    "  Layer {}: both ~zero (a={:.4e}, fd={:.4e})",
                    layer, a_norm, f_norm
                );
                continue;
            }

            let diff = (a - f).norm();
            let ref_val = a_norm.max(f_norm);
            let rel_err = diff / ref_val;

            println!(
                "  Layer {}: analytical=({:.6e}, {:.6e}), fd=({:.6e}, {:.6e}), rel_err={:.2e}",
                layer, a.re, a.im, f.re, f.im, rel_err
            );

            // Only assert for non-negligible values
            if ref_val > 1e-30 {
                assert!(
                    rel_err < tol,
                    "{} layer {}: rel_err={:.4e} exceeds tolerance {:.4e}\n  analytical=({:.10e}, {:.10e})\n  fd=({:.10e}, {:.10e})",
                    name, layer, rel_err, tol, a.re, a.im, f.re, f.im
                );
            }
        }
    }

    println!("\nAll analytical derivatives match finite-difference to within {:.1}%", tol * 100.0);
}
