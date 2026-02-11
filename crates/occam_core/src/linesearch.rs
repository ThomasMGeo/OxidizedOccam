//! Line search for optimal Lagrange multiplier

use crate::error::{OccamError, Result};

/// Bracket for minimum search
#[derive(Debug, Clone)]
pub struct Bracket {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub fa: f64,
    pub fb: f64,
    pub fc: f64,
}

/// Golden section search parameters
const GOLDEN_RATIO: f64 = 0.381966011250105; // (3 - sqrt(5)) / 2

/// Find bracket containing minimum of f(x)
///
/// Starting from initial point, expand bracket until minimum is enclosed.
pub fn bracket_minimum<F>(
    initial: f64,
    step: f64,
    f: &mut F,
    max_iter: usize,
) -> Result<Bracket>
where
    F: FnMut(f64) -> Result<f64>,
{
    let mut a = initial;
    let mut fa = f(a)?;

    let mut b = initial + step;
    let mut fb = f(b)?;

    // Ensure we're going downhill
    if fb > fa {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    // Expand bracket
    let mut c = b + 1.618 * (b - a);
    let mut fc = f(c)?;

    for _ in 0..max_iter {
        if fb <= fc {
            // Found bracket: a < b < c with f(b) <= f(a), f(c)
            return Ok(Bracket { a, b, c, fa, fb, fc });
        }

        // Keep expanding
        let new_c = c + 1.618 * (c - b);
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = new_c;
        fc = f(c)?;
    }

    // Return best bracket found
    Ok(Bracket { a, b, c, fa, fb, fc })
}

/// Brent's method for minimum finding
pub fn brent_minimize<F>(
    bracket: &Bracket,
    tol: f64,
    f: &mut F,
    max_iter: usize,
) -> Result<(f64, f64)>
where
    F: FnMut(f64) -> Result<f64>,
{
    let mut a = bracket.a.min(bracket.c);
    let mut b = bracket.a.max(bracket.c);
    let mut x = bracket.b;
    let mut w = x;
    let mut v = x;
    let mut fx = bracket.fb;
    let mut fw = fx;
    let mut fv = fx;

    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;

    for _ in 0..max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        // Check convergence
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok((x, fx));
        }

        // Try parabolic interpolation
        let mut use_golden = true;
        if e.abs() > tol1 {
            // Fit parabola
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            let r_old = e;
            e = d;

            // Check if parabolic step is acceptable
            if p.abs() < (0.5 * q * r_old).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;

                // Ensure u is not too close to boundaries
                if u - a < tol2 || b - u < tol2 {
                    d = if x < xm { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            // Golden section step
            e = if x < xm { b - x } else { a - x };
            d = GOLDEN_RATIO * e;
        }

        // Ensure step is not too small
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };

        let fu = f(u)?;

        // Update bracket
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    Ok((x, fx))
}

/// Find root of f(x) = target using bisection and Brent's method
pub fn find_root<F>(
    a: f64,
    b: f64,
    target: f64,
    tol: f64,
    f: &mut F,
    max_iter: usize,
) -> Result<f64>
where
    F: FnMut(f64) -> Result<f64>,
{
    let mut a = a;
    let mut b = b;
    let mut fa = f(a)? - target;
    let mut fb = f(b)? - target;

    // Check if root is bracketed
    if fa * fb > 0.0 {
        return Err(OccamError::Convergence(
            "Root not bracketed".to_string(),
        ));
    }

    // Ensure fa < 0 < fb
    if fa > 0.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let _c = a;
    let _fc = fa;

    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            return Ok(0.5 * (a + b));
        }

        // Bisection step
        let m = 0.5 * (a + b);
        let fm = f(m)? - target;

        if fm < 0.0 {
            a = m;
            let _ = fm;
        } else {
            b = m;
            let _ = fm;
        }
    }

    Ok(0.5 * (a + b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brent_minimize() {
        let mut f = |x: f64| -> Result<f64> { Ok((x - 3.0).powi(2) + 1.0) };

        let bracket = Bracket {
            a: 0.0,
            b: 3.0,
            c: 6.0,
            fa: 10.0,
            fb: 1.0,
            fc: 10.0,
        };

        let (x_min, f_min) = brent_minimize(&bracket, 1e-6, &mut f, 100).unwrap();

        assert!((x_min - 3.0).abs() < 1e-5);
        assert!((f_min - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_root() {
        // Find x where x^2 = 4, i.e., x = 2
        let mut f = |x: f64| -> Result<f64> { Ok(x * x) };

        let root = find_root(0.0, 5.0, 4.0, 1e-6, &mut f, 100).unwrap();

        assert!((root - 2.0).abs() < 1e-5);
    }
}
