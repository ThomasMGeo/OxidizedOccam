//! Command-line interface for Occam 1D inversion
//!
//! Usage:
//!   occam1d invert --startup <file>     Run inversion
//!   occam1d forward --startup <file>    Forward modeling only
//!   occam1d create-model <options>      Create a model file
//!   occam1d validate <file>             Validate file format

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use occam_core::OccamInversion;
use occam_io::{EMData, IterationFile, LayeredModel, write_resp_file, write_csv_file};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "occam1d")]
#[command(author = "OxidizedOccam Contributors")]
#[command(version = "0.1.0")]
#[command(about = "1D Occam inversion for CSEM and MT data", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run Occam inversion
    Invert {
        /// Startup/iteration file
        #[arg(short, long)]
        startup: PathBuf,

        /// Output iteration file prefix
        #[arg(short, long, default_value = "iter")]
        output: String,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run forward modeling only
    Forward {
        /// Startup/iteration file
        #[arg(short, long)]
        startup: PathBuf,

        /// Output response file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Create a model file
    CreateModel {
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Water depth (m)
        #[arg(long, default_value = "1000")]
        water_depth: f64,

        /// Water resistivity (ohm-m)
        #[arg(long, default_value = "0.3")]
        water_resistivity: f64,

        /// Number of layers below seafloor
        #[arg(long, default_value = "50")]
        n_layers: usize,

        /// Maximum depth below seafloor (m)
        #[arg(long, default_value = "4000")]
        max_depth: f64,

        /// Initial resistivity for free layers (ohm-m)
        #[arg(long, default_value = "10")]
        initial_resistivity: f64,
    },

    /// Validate file format
    Validate {
        /// File to validate
        file: PathBuf,

        /// File type (auto-detect if not specified)
        #[arg(short = 't', long)]
        file_type: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Invert { startup, output, verbose } => {
            run_inversion(&startup, &output, verbose)
        }
        Commands::Forward { startup, output, verbose } => {
            run_forward(&startup, output.as_deref(), verbose)
        }
        Commands::CreateModel {
            output,
            water_depth,
            water_resistivity,
            n_layers,
            max_depth,
            initial_resistivity,
        } => create_model(
            &output,
            water_depth,
            water_resistivity,
            n_layers,
            max_depth,
            initial_resistivity,
        ),
        Commands::Validate { file, file_type } => validate_file(&file, file_type.as_deref()),
    }
}

fn run_inversion(startup_path: &PathBuf, output_prefix: &str, verbose: bool) -> Result<()> {
    use std::time::Instant;

    println!("OxidizedOccam 1D Inversion");
    println!("==========================");
    println!();

    // Load startup file
    let iteration_file = IterationFile::read_from_file(startup_path)
        .context("Failed to read startup file")?;

    if verbose {
        println!("Startup file: {}", startup_path.display());
        println!("Model file: {}", iteration_file.model_file);
        println!("Data file: {}", iteration_file.data_file);
        println!("Target misfit: {:.4}", iteration_file.target_misfit);
        println!("Max iterations: {}", iteration_file.max_iterations);
        if let Some((lo, hi)) = iteration_file.model_bounds {
            let transform = iteration_file.bounds_transform.as_deref().unwrap_or("bandpass");
            println!("Model bounds: [{}, {}] ({})", lo, hi, transform);
        }
        println!();
    }

    // Load data for response file writing
    let data = EMData::read_from_file(&iteration_file.data_file)
        .context("Failed to read data file")?;

    // Initialize inversion
    let (mut inversion, initial_params) = OccamInversion::from_files(&iteration_file)
        .context("Failed to initialize inversion")?;

    println!("Parameters: {}", inversion.n_params());
    println!("Data points: {}", inversion.n_data());
    println!();

    // Print header
    if verbose {
        println!("{:>5} {:>12} {:>12} {:>12} {:>8} {:>10}",
            "Iter", "RMS Misfit", "Roughness", "log10(mu)", "Time(s)", "Status");
        println!("{}", "-".repeat(65));
    } else {
        println!("{:>5} {:>12} {:>12} {:>12} {:>10}",
            "Iter", "RMS Misfit", "Roughness", "log10(mu)", "Status");
        println!("{}", "-".repeat(55));
    }

    let inversion_start = Instant::now();

    // Run inversion with per-iteration timing
    let results = inversion.run_timed(&initial_params, |result, elapsed| {
        let status_str = format_status(result.status);
        if verbose {
            println!("{:>5} {:>12.4} {:>12.4} {:>12.4} {:>8.2} {:>10}",
                result.iteration,
                result.rms_misfit,
                result.roughness,
                result.lagrange,
                elapsed.as_secs_f64(),
                status_str);
        } else {
            println!("{:>5} {:>12.4} {:>12.4} {:>12.4} {:>10}",
                result.iteration,
                result.rms_misfit,
                result.roughness,
                result.lagrange,
                status_str);
        }

        // Write iteration file
        let iter_path = format!("{}_{:04}.iter", output_prefix, result.iteration);
        let mut iter_out = iteration_file.clone();
        iter_out.update_status(
            result.iteration,
            result.lagrange,
            result.roughness,
            result.rms_misfit,
            &result.parameters,
        );
        let _ = iter_out.write_to_file(&iter_path);

        // Write response files
        let resp_path = format!("{}_{:04}.resp", output_prefix, result.iteration);
        let csv_path = format!("{}_{:04}.csv", output_prefix, result.iteration);
        let _ = write_resp_file(&resp_path, &data, &result.computed);
        let _ = write_csv_file(&csv_path, &data, &result.computed);
    }).context("Inversion failed")?;

    let total_elapsed = inversion_start.elapsed();

    println!();
    if let Some(final_result) = results.last() {
        if final_result.status.is_success() {
            println!("Inversion completed successfully.");
        } else {
            println!("Inversion stopped: {:?}", final_result.status);
        }
    }
    println!("Total wall time: {:.2}s ({} iterations)", total_elapsed.as_secs_f64(), results.len() - 1);

    Ok(())
}

fn format_status(status: occam_core::ConvergenceStatus) -> &'static str {
    match status {
        occam_core::ConvergenceStatus::Iterating => "...",
        occam_core::ConvergenceStatus::TargetReached => "target",
        occam_core::ConvergenceStatus::Converged => "converged",
        occam_core::ConvergenceStatus::PerfectlySmooth => "smooth",
        occam_core::ConvergenceStatus::Diverging => "diverge",
        occam_core::ConvergenceStatus::RoughnessIncreasing => "rough+",
        occam_core::ConvergenceStatus::AutoConverged => "auto",
    }
}

fn run_forward(startup_path: &PathBuf, output_path: Option<&std::path::Path>, verbose: bool) -> Result<()> {
    println!("OxidizedOccam Forward Modeling");
    println!("==============================");
    println!();

    // Load startup file
    let iteration_file = IterationFile::read_from_file(startup_path)
        .context("Failed to read startup file")?;

    // Load data and model
    let data = EMData::read_from_file(&iteration_file.data_file)
        .context("Failed to read data file")?;
    let model = LayeredModel::read_from_file(&iteration_file.model_file)
        .context("Failed to read model file")?;

    if verbose {
        println!("Data file: {}", iteration_file.data_file);
        println!("Model file: {}", iteration_file.model_file);
        println!("Frequencies: {}", data.frequencies.len());
        println!("Transmitters: {}", data.transmitters.len());
        println!("Receivers: {}", data.receivers.len());
        println!("Data points: {}", data.n_data());
        println!();
    }

    // Run forward model
    let forward = occam_core::ForwardEngine::new(data.clone(), model);
    let params = iteration_file.parameters.clone();

    println!("Running forward model...");
    let result = forward.compute(&params)
        .context("Forward modeling failed")?;

    println!();
    println!("RMS Misfit: {:.4}", result.rms_misfit);
    println!("Chi-squared: {:.4}", result.chi_squared);

    // Write response files
    let base = output_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("forward"));

    let resp_path = base.with_extension("resp");
    let csv_path = base.with_extension("csv");

    write_resp_file(&resp_path, &data, &result.computed)
        .context("Failed to write .resp file")?;
    write_csv_file(&csv_path, &data, &result.computed)
        .context("Failed to write .csv file")?;

    println!("Response written to: {}", resp_path.display());
    println!("CSV written to: {}", csv_path.display());

    Ok(())
}

fn create_model(
    output: &PathBuf,
    water_depth: f64,
    water_resistivity: f64,
    n_layers: usize,
    max_depth: f64,
    initial_resistivity: f64,
) -> Result<()> {
    println!("Creating 1D model...");

    // Create layer depths with logarithmic spacing
    let mut layer_depths = Vec::with_capacity(n_layers);
    let log_min = 0.0_f64.ln();
    let log_max = max_depth.ln();

    for i in 0..n_layers {
        let t = i as f64 / (n_layers - 1) as f64;
        let log_depth = log_min + t * (log_max - log_min);
        layer_depths.push(log_depth.exp());
    }

    let model = LayeredModel::create_marine(
        water_depth,
        water_resistivity,
        &layer_depths,
        initial_resistivity,
    );

    model.write_to_file(output)
        .context("Failed to write model file")?;

    println!("Model written to: {}", output.display());
    println!("  Water depth: {} m", water_depth);
    println!("  Water resistivity: {} ohm-m", water_resistivity);
    println!("  Free layers: {}", n_layers);
    println!("  Max depth below seafloor: {} m", max_depth);

    Ok(())
}

fn validate_file(path: &PathBuf, file_type: Option<&str>) -> Result<()> {
    println!("Validating: {}", path.display());

    let file_type = file_type.unwrap_or_else(|| {
        // Auto-detect from extension or content
        let ext = path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        match ext {
            "data" => "data",
            "iter" => "iteration",
            "mod" => "model",
            _ => "auto"
        }
    });

    match file_type {
        "data" | "emdata" => {
            let data = EMData::read_from_file(path)
                .context("Failed to parse as EMData file")?;
            println!("  Format: {:?}", data.format);
            println!("  Transmitters: {}", data.transmitters.len());
            println!("  Frequencies: {}", data.frequencies.len());
            println!("  Receivers: {}", data.receivers.len());
            println!("  Data points: {}", data.n_data());
            println!("  Phase convention: {}", data.phase_convention);
            println!("  Valid: YES");
        }
        "model" | "resistivity" => {
            let model = LayeredModel::read_from_file(path)
                .context("Failed to parse as model file")?;
            println!("  Layers: {}", model.n_layers());
            println!("  Free parameters: {}", model.n_free_params());
            println!("  Valid: YES");
        }
        "iteration" | "startup" | "iter" => {
            let iter = IterationFile::read_from_file(path)
                .context("Failed to parse as iteration file")?;
            println!("  Description: {}", iter.description);
            println!("  Model file: {}", iter.model_file);
            println!("  Data file: {}", iter.data_file);
            println!("  Parameters: {}", iter.n_params());
            println!("  Target misfit: {}", iter.target_misfit);
            println!("  Current iteration: {}", iter.iteration);
            println!("  Current misfit: {}", iter.misfit_value);
            println!("  Valid: YES");
        }
        "auto" => {
            // Try each format
            if EMData::read_from_file(path).is_ok() {
                println!("  Detected: EMData file");
                return validate_file(path, Some("data"));
            }
            if LayeredModel::read_from_file(path).is_ok() {
                println!("  Detected: Model file");
                return validate_file(path, Some("model"));
            }
            if IterationFile::read_from_file(path).is_ok() {
                println!("  Detected: Iteration file");
                return validate_file(path, Some("iteration"));
            }
            anyhow::bail!("Could not detect file type");
        }
        _ => {
            anyhow::bail!("Unknown file type: {}", file_type);
        }
    }

    Ok(())
}
