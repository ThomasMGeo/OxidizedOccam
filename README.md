# OxidizedOccam

A Rust implementation of the Occam 1D inversion algorithm for Controlled Source
Electromagnetic (CSEM) and Magnetotelluric (MT) data.

Based on the Fortran implementation by
[Kerry Key](https://marineemlab.ucsd.edu/Projects/Occam/),
Scripps Institution of Oceanography.

## Features

- **CSEM forward modeling** -- HED and VED sources in layered media, arbitrary
  azimuth and dip, point or finite-length dipoles
- **MT forward modeling** -- impedance, apparent resistivity and phase
- **Occam inversion** -- minimum-structure regularization (1st or 2nd derivative),
  optional model bounds (bandpass or exponential transform)
- **Analytical Jacobians** for all data types -- no finite differences
- **Polarization ellipse** parameters (PEmax, PEmin, PBmax, PBmin) with
  analytical Jacobians
- **Fortran-compatible I/O** -- reads and writes the same file formats

### Supported Data Types

| Code | Type | Components |
|------|------|------------|
| 1-16 | Real / Imag | Ex, Ey, Ez, Bx, By, Bz |
| 21-36 | Amplitude / Phase | Ex, Ey, Ez, Bx, By, Bz |
| 41-44 | Polarization Ellipse | PEmax, PEmin, PBmax, PBmin |
| 101-118 | MT | Rho_xy, Phi_xy, Zxy Real/Imag |

3D receiver rotation (theta, alpha, beta) is supported for all CSEM types.

### Performance

Benchmarked on Apple Silicon, median ms/iter:

| Example | Rust | Fortran | Speedup |
|---------|------|---------|---------|
| CSEM unbounded | 48 ms | 57 ms | 1.2x |
| CSEM bounded | 45 ms | 65 ms | 1.4x |
| CSEM + PE (656 data) | 119 ms | -- | -- |
| MT | 3 ms | 23 ms | 9x |

## Quick Start

```bash
cargo build --release

# Run the canonical reservoir example
occam1d invert --startup examples/canonical_reservoir/startup -v

# Forward model only
occam1d forward --startup examples/canonical_reservoir/startup -v
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `dipole1d` | 1D CSEM forward modeling (Hankel transform, layered potentials, analytical sensitivity) |
| `mt1d` | 1D MT forward modeling (Wait recursion, analytical sensitivity) |
| `occam_core` | Occam inversion (regularization, line search, bounds transforms) |
| `occam_io` | File I/O for data, model, iteration, and response files |
| `occam_cli` | Command-line interface (`occam1d`) |

## CLI

```bash
# Inversion
occam1d invert --startup startup -v

# Forward modeling
occam1d forward --startup startup -o response -v

# Create a model file
occam1d create-model --output model --water-depth 1000 --n-layers 75 --max-depth 4000

# Validate files
occam1d validate startup
occam1d validate model -t model
occam1d validate data.data -t data
```

Output files: `iter_NNNN.iter` (parameters), `iter_NNNN.resp` (EMResp_1.2
format), `iter_NNNN.csv` (tidy CSV for plotting).

## Examples

### canonical_reservoir -- CSEM inversion

41 transmitters, 300 data points (Ey/Ez real+imag at 0.25 and 1 Hz), 75 free
parameters. Converges RMS 16.7 to 1.0 in 20 iterations, recovering a 100 ohm-m
reservoir at 2 km depth.

### canonical_reservoir_bounded -- CSEM with model bounds

Same geometry with log10(resistivity) bounded to [0, 2]. Converges in 58
iterations using bandpass transform.

### canonical_mt -- MT inversion

Seafloor MT with apparent resistivity and phase at 6 frequencies.
Converges RMS 37.6 to 1.0 in 4 iterations.

### canonical_pe -- Tilted transmitter polarization ellipse

Transmitter tilted 20 deg off-inline and 20 deg below horizontal, generating
genuinely elliptical polarization. 656 data points (Ey/Ez real+imag + all 4 PE
types). Includes a three-way comparison showing that PE data provides structural
constraint that raw field components miss with off-axis geometry.

## Algorithms

**Dipole1D**: Lorenz-gauged vector potentials (Wait 1982), decaying exponential
TM/TE recursions, cross-layer coefficient propagation, 201-point Hankel filter
(Key 2012), angular post-processing, Gauss-Legendre quadrature for finite
dipoles, analytical d(fields)/d(sigma).

**MT1D**: Wait (1954) bottom-up impedance recursion with chain-rule analytical
sensitivity.

**Occam**: Linearized minimum-structure inversion (Constable et al. 1987).
Two-phase Lagrange multiplier search (fitting then smoothing), step-size
damping, optional bandpass/exponential bounds transforms.

## Validation

- CSEM forward: matches Fortran Dipole1D to full precision on canonical model
- MT forward: matches analytical halfspace and multilayer solutions
- CSEM Jacobian: analytical vs finite-difference to 1e-4 relative error
- MT Jacobian: analytical vs central FD on 5-layer model, 6 frequencies
- PE Jacobian: analytical vs central FD for elliptical polarization
- Iteration files: round-trip compatible with Fortran Occam1DCSEM

35 tests: `cargo test`

## References

- Constable, S. C., Parker, R. L., & Constable, C. G. (1987). Occam's
  inversion: A practical algorithm for generating smooth models from
  electromagnetic sounding data. *Geophysics*, 52(3), 289-300.
- Key, K. (2009). 1D inversion of multicomponent, multifrequency marine CSEM
  data. *Geophysics*, 74(2), F9-F20.

## License

MIT
