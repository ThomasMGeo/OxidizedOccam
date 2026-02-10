# dipole1d

1D CSEM forward modeling using the Dipole1D algorithm by Kerry Key (Scripps Institution of Oceanography).

Computes all six electromagnetic field components (Ex, Ey, Ez, Bx, By, Bz) from horizontal and vertical electric dipole sources in a 1D layered earth.

## Usage

```rust
use dipole1d::{Dipole1D, LayeredEarth, Transmitter, Receiver};

// Marine CSEM model: air / seawater (0.3 ohm-m, 0-1000m) / sediment (1 ohm-m)
let model = LayeredEarth::new(
    vec![-100000.0, 0.0, 1000.0],
    vec![1e-12, 1.0 / 0.3, 1.0],
).unwrap();

let engine = Dipole1D::new(model);

let tx = Transmitter::hed(0.0, 0.0, 950.0, 90.0); // y-directed HED at 950m
let rx = Receiver::new(5000.0, 0.0, 1000.0);        // 5km inline on seafloor

let fields = engine.compute(&tx, &[rx], 0.25).unwrap();
println!("Inline Ex = {:e} V/m", fields[0].ex);
```

## Algorithm

The implementation follows the Fortran Dipole1D code exactly:

1. **Potentials**: Lorenz-gauged Ay (vector potential) and Λz (scalar potential) in cylindrical coordinates
2. **Layer coefficients**: TM mode (Rm, Rp) and TE mode (Sm, Sp) reflection coefficients computed via upward/downward recursion with decaying exponentials for numerical stability
3. **Cross-layer propagation**: Source layer coefficients (a, b, c, d) propagated to the receiver layer through transmission matrices
4. **Hankel transform**: Integrals over horizontal wavenumber λ evaluated using Kerry Key's custom 201-point digital filter (kk_ht_201), combining J₀ and J₁ Bessel function transforms
5. **Post-processing**: Angular factors (sin θ, cos 2θ, etc.) and coordinate rotation from dipole-local to cardinal directions

## Model Definition

```rust
// 5-layer canonical reservoir model
let model = LayeredEarth::new(
    vec![-100000.0, 0.0, 1000.0, 2000.0, 2100.0],  // Layer top depths (m)
    vec![1e-12, 1.0/0.3, 1.0, 0.01, 1.0],           // Conductivities (S/m)
).unwrap();
// Layer 0: Air          (σ = 1e-12 S/m)
// Layer 1: Seawater     (σ = 3.33 S/m, 0-1000m)
// Layer 2: Sediment     (σ = 1.0 S/m, 1000-2000m)
// Layer 3: Reservoir    (σ = 0.01 S/m, 2000-2100m)
// Layer 4: Sediment     (σ = 1.0 S/m, 2100m+)
```

## Conventions

- **Coordinates**: z positive downward, azimuth from x-axis
- **Time dependence**: exp(-iωt)
- **Layer boundaries**: receivers on a boundary belong to the layer above
- **Singularity at r=0**: horizontal distance clamped to 1m minimum
