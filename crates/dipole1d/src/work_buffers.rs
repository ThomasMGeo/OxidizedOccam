//! Pre-allocated work buffers for Hankel transform integration
//!
//! Eliminates per-filter-point heap allocation by reusing fixed-size arrays.
//! All flat matrices use row-major layout: element [k][j] = buf[k * n + j].

use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Pre-allocated buffers for the sensitivity (Jacobian) code path.
///
/// Created once per `compute_with_sensitivity()` call (before the receiver loop)
/// and reused across all 201 filter points and all receivers.
pub struct WorkBuffers {
    pub n: usize,

    // --- Per-filter-point vectors (length n each) ---
    pub gamma: Vec<Complex64>,
    pub expmgh: Vec<Complex64>,

    // TM recursion coefficients
    pub rm: Vec<Complex64>,
    pub rp: Vec<Complex64>,
    // TE recursion coefficients
    pub sm: Vec<Complex64>,
    pub sp: Vec<Complex64>,

    // --- Derivative matrices (n*n flat, row-major [k*n+j]) ---
    pub drmdsig: Vec<Complex64>,
    pub drpdsig: Vec<Complex64>,
    pub dsmdsig: Vec<Complex64>,
    pub dspdsig: Vec<Complex64>,

    // --- Coefficient vectors (length n each) ---
    pub a: Vec<Complex64>,
    pub b: Vec<Complex64>,
    pub c: Vec<Complex64>,
    pub d: Vec<Complex64>,

    // --- Coefficient derivative matrices (n*n flat each) ---
    pub dadsig: Vec<Complex64>,
    pub dbdsig: Vec<Complex64>,
    pub dcdsig: Vec<Complex64>,
    pub dddsig: Vec<Complex64>,

    // --- Receiver potential derivatives (length n each) ---
    pub dae1dsig: Vec<Complex64>,
    pub dbe2dsig: Vec<Complex64>,
    pub day_dsig: Vec<Complex64>,
    pub daydsigdz: Vec<Complex64>,
    pub dce1dsig: Vec<Complex64>,
    pub dde2dsig: Vec<Complex64>,
    pub daz_dsig: Vec<Complex64>,
    pub dazdsigdz: Vec<Complex64>,
    pub dazdsigdz2: Vec<Complex64>,
    pub daypdazdsigdz: Vec<Complex64>,
    pub daydsigdzpdazdsigdz2: Vec<Complex64>,

    // --- VED-specific coefficient derivatives (length n each) ---
    pub dccdsig: Vec<Complex64>,
    pub ddddsig: Vec<Complex64>,
    pub ddazdzdsig: Vec<Complex64>,
}

impl WorkBuffers {
    /// Allocate all buffers for `n` layers. Call once, reuse across filter points.
    pub fn new(n: usize) -> Self {
        let nn = n * n;
        Self {
            n,
            gamma: vec![ZERO; n],
            expmgh: vec![ZERO; n],
            rm: vec![ZERO; n],
            rp: vec![ZERO; n],
            sm: vec![ZERO; n],
            sp: vec![ZERO; n],
            drmdsig: vec![ZERO; nn],
            drpdsig: vec![ZERO; nn],
            dsmdsig: vec![ZERO; nn],
            dspdsig: vec![ZERO; nn],
            a: vec![ZERO; n],
            b: vec![ZERO; n],
            c: vec![ZERO; n],
            d: vec![ZERO; n],
            dadsig: vec![ZERO; nn],
            dbdsig: vec![ZERO; nn],
            dcdsig: vec![ZERO; nn],
            dddsig: vec![ZERO; nn],
            dae1dsig: vec![ZERO; n],
            dbe2dsig: vec![ZERO; n],
            day_dsig: vec![ZERO; n],
            daydsigdz: vec![ZERO; n],
            dce1dsig: vec![ZERO; n],
            dde2dsig: vec![ZERO; n],
            daz_dsig: vec![ZERO; n],
            dazdsigdz: vec![ZERO; n],
            dazdsigdz2: vec![ZERO; n],
            daypdazdsigdz: vec![ZERO; n],
            daydsigdzpdazdsigdz2: vec![ZERO; n],
            dccdsig: vec![ZERO; n],
            ddddsig: vec![ZERO; n],
            ddazdzdsig: vec![ZERO; n],
        }
    }

    /// Zero all buffers for reuse at the next filter point.
    #[inline]
    pub fn clear(&mut self) {
        let n = self.n;
        let nn = n * n;

        self.gamma[..n].fill(ZERO);
        self.expmgh[..n].fill(ZERO);
        self.rm[..n].fill(ZERO);
        self.rp[..n].fill(ZERO);
        self.sm[..n].fill(ZERO);
        self.sp[..n].fill(ZERO);
        self.drmdsig[..nn].fill(ZERO);
        self.drpdsig[..nn].fill(ZERO);
        self.dsmdsig[..nn].fill(ZERO);
        self.dspdsig[..nn].fill(ZERO);
        self.a[..n].fill(ZERO);
        self.b[..n].fill(ZERO);
        self.c[..n].fill(ZERO);
        self.d[..n].fill(ZERO);
        self.dadsig[..nn].fill(ZERO);
        self.dbdsig[..nn].fill(ZERO);
        self.dcdsig[..nn].fill(ZERO);
        self.dddsig[..nn].fill(ZERO);
        self.dae1dsig[..n].fill(ZERO);
        self.dbe2dsig[..n].fill(ZERO);
        self.day_dsig[..n].fill(ZERO);
        self.daydsigdz[..n].fill(ZERO);
        self.dce1dsig[..n].fill(ZERO);
        self.dde2dsig[..n].fill(ZERO);
        self.daz_dsig[..n].fill(ZERO);
        self.dazdsigdz[..n].fill(ZERO);
        self.dazdsigdz2[..n].fill(ZERO);
        self.daypdazdsigdz[..n].fill(ZERO);
        self.daydsigdzpdazdsigdz2[..n].fill(ZERO);
        self.dccdsig[..n].fill(ZERO);
        self.ddddsig[..n].fill(ZERO);
        self.ddazdzdsig[..n].fill(ZERO);
    }
}

/// Pre-allocated buffers for the forward-only (no sensitivity) code path.
///
/// Much smaller than `WorkBuffers` — only the arrays needed for field computation
/// without derivatives.
pub struct ForwardBuffers {
    pub n: usize,
    pub gamma: Vec<Complex64>,
    pub expmgh: Vec<Complex64>,
    pub rm: Vec<Complex64>,
    pub rp: Vec<Complex64>,
    pub sm: Vec<Complex64>,
    pub sp: Vec<Complex64>,
    pub a: Vec<Complex64>,
    pub b: Vec<Complex64>,
    pub c: Vec<Complex64>,
    pub d: Vec<Complex64>,
}

impl ForwardBuffers {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            gamma: vec![ZERO; n],
            expmgh: vec![ZERO; n],
            rm: vec![ZERO; n],
            rp: vec![ZERO; n],
            sm: vec![ZERO; n],
            sp: vec![ZERO; n],
            a: vec![ZERO; n],
            b: vec![ZERO; n],
            c: vec![ZERO; n],
            d: vec![ZERO; n],
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        let n = self.n;
        self.gamma[..n].fill(ZERO);
        self.expmgh[..n].fill(ZERO);
        self.rm[..n].fill(ZERO);
        self.rp[..n].fill(ZERO);
        self.sm[..n].fill(ZERO);
        self.sp[..n].fill(ZERO);
        self.a[..n].fill(ZERO);
        self.b[..n].fill(ZERO);
        self.c[..n].fill(ZERO);
        self.d[..n].fill(ZERO);
    }
}
