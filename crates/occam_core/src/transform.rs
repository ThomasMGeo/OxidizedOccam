//! Parameter transforms for bounded inversion

/// Bandpass transform sharpness factor (Fortran: bandPassfactor = 15)
const BANDPASS_FACTOR: f64 = 15.0;

/// Transform type for bounded parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransformType {
    /// No transform (unbounded)
    #[default]
    None,
    /// Exponential transform (Habashy & Abubakar 2004)
    Exponential,
    /// Bandpass transform (flat sensitivity between bounds)
    Bandpass,
}

/// Parameter bounds
#[derive(Debug, Clone)]
pub struct ParameterBounds {
    /// Lower bound (log10 resistivity)
    pub lower: f64,
    /// Upper bound (log10 resistivity)
    pub upper: f64,
    /// Transform type
    pub transform: TransformType,
}

impl Default for ParameterBounds {
    fn default() -> Self {
        Self {
            lower: -3.0, // 0.001 ohm-m
            upper: 6.0,  // 1,000,000 ohm-m
            transform: TransformType::None,
        }
    }
}

impl ParameterBounds {
    /// Create with specified bounds
    pub fn new(lower: f64, upper: f64) -> Self {
        Self {
            lower,
            upper,
            transform: TransformType::None,
        }
    }

    /// Set transform type
    pub fn with_transform(mut self, transform: TransformType) -> Self {
        self.transform = transform;
        self
    }

    /// Transform from bounded parameter m to unbounded x
    ///
    /// Fortran reference: Occam.f90, transformToUnbound (lines 2934-2961)
    pub fn to_unbounded(&self, m: f64) -> f64 {
        let a = self.upper;
        let b = self.lower;
        match self.transform {
            TransformType::None => m,
            TransformType::Exponential => {
                // x = ln(m - b) - ln(a - m)
                (m - b).ln() - (a - m).ln()
            }
            TransformType::Bandpass => {
                // Fortran: c = bandPassFactor / (a - b)
                //          p = m * c * (1 - exp(c*(b-a)))
                //          x = (ln(exp(a*c)*(exp(p)-exp(b*c))) - ln(exp(a*c)-exp(p))) / c
                let c = BANDPASS_FACTOR / (a - b);
                let p = m * c * (1.0 - (c * (b - a)).exp());
                // Use log-sum-exp tricks to avoid overflow:
                // ln(exp(a*c)*(exp(p)-exp(b*c))) = a*c + ln(exp(p) - exp(b*c))
                //   = a*c + ln(exp(b*c)*(exp(p-b*c)-1))  when p > b*c
                //   = a*c + b*c + ln(exp(p-b*c) - 1)
                // ln(exp(a*c) - exp(p)) = ln(exp(a*c)*(1-exp(p-a*c)))
                //   = a*c + ln(1 - exp(p-a*c))
                // So x = (a*c + b*c + ln(exp(p-b*c)-1) - a*c - ln(1-exp(p-a*c))) / c
                //      = (b*c + ln(exp(p-b*c)-1) - ln(1-exp(p-a*c))) / c
                // But the direct formula works fine for typical values:
                let ac = a * c;
                let bc = b * c;
                let num = (ac.exp() * (p.exp() - bc.exp())).ln();
                let den = (ac.exp() - p.exp()).ln();
                (num - den) / c
            }
        }
    }

    /// Transform from unbounded x to bounded parameter m
    ///
    /// Fortran reference: Occam.f90, transformToBound (lines 2896-2932)
    pub fn to_bounded(&self, x: f64) -> f64 {
        let a = self.upper;
        let b = self.lower;
        match self.transform {
            TransformType::None => x.clamp(self.lower, self.upper),
            TransformType::Exponential => {
                // Fortran: numerically stable form
                if x <= 0.0 {
                    (a * x.exp() + b) / (x.exp() + 1.0)
                } else {
                    (b * (-x).exp() + a) / ((-x).exp() + 1.0)
                }
            }
            TransformType::Bandpass => {
                // c = bandPassFactor / (a - b)
                // p = c * (1 - exp(c*(b-a)))   [normalization]
                // q = log((exp(c*x)+exp(c*b)) / (exp(c*x)+exp(c*a)))  for x<=0
                //   = log((1+exp(c*(b-x))) / (1+exp(c*(a-x))))        for x>0
                // m = (a*c + q) / p
                let c = BANDPASS_FACTOR / (a - b);
                let p = c * (1.0 - (c * (b - a)).exp());

                let q = if x <= 0.0 {
                    let cx = c * x;
                    let cb = c * b;
                    let ca = c * a;
                    ((cx.exp() + cb.exp()) / (cx.exp() + ca.exp())).ln()
                } else {
                    let cbx = c * (b - x);
                    let cax = c * (a - x);
                    ((1.0 + cbx.exp()) / (1.0 + cax.exp())).ln()
                };

                let m = (a * c + q) / p;
                // Clamp for numerical precision at extremes
                m.clamp(b, a)
            }
        }
    }

    /// Compute Jacobian of transform: dm/dx
    ///
    /// Fortran reference: Occam.f90, transformWJ (lines 2963-3013)
    /// Note: the Fortran subroutine computes this using the unbounded parameter x (called pm).
    pub fn transform_jacobian(&self, x: f64) -> f64 {
        let a = self.upper;
        let b = self.lower;
        match self.transform {
            TransformType::None => 1.0,
            TransformType::Exponential => {
                // Fortran: dm/dx = (a-b)*p/(1+p)^2  where p=exp(x) for x<=0, exp(-x) for x>0
                if x <= 0.0 {
                    let p = x.exp();
                    (a - b) * p / ((1.0 + p) * (1.0 + p))
                } else {
                    let p = (-x).exp();
                    (a - b) * p / ((1.0 + p) * (1.0 + p))
                }
            }
            TransformType::Bandpass => {
                // Fortran: c = bandPassFactor / (a-b)
                // dm/dx = p / ((1+p)*(1+q))    for x<=0
                //       = q / ((1+p)*(1+q))    for x>0
                // where p = exp(c*(x-b)) or exp(-c*(x-b))
                //       q = exp(c*(x-a)) or exp(-c*(x-a))
                // Fortran ref: Occam.f90 transformWJ (lines 2963-3013)
                let c = BANDPASS_FACTOR / (a - b);

                if x <= 0.0 {
                    let p = (c * (x - b)).exp();
                    let q = (c * (x - a)).exp();
                    p / ((1.0 + p) * (1.0 + q))
                } else {
                    let p = (-c * (x - b)).exp();
                    let q = (-c * (x - a)).exp();
                    q / ((1.0 + p) * (1.0 + q))
                }
            }
        }
    }

    /// Check if value is within bounds
    pub fn is_within_bounds(&self, m: f64) -> bool {
        m >= self.lower && m <= self.upper
    }
}

/// Transform a vector of parameters
pub fn transform_to_unbounded(params: &[f64], bounds: &ParameterBounds) -> Vec<f64> {
    params.iter().map(|&m| bounds.to_unbounded(m)).collect()
}

/// Transform a vector of unbounded parameters back to bounded
pub fn transform_to_bounded(x: &[f64], bounds: &ParameterBounds) -> Vec<f64> {
    x.iter().map(|&xi| bounds.to_bounded(xi)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_transform_roundtrip() {
        let bounds = ParameterBounds::new(-2.0, 4.0).with_transform(TransformType::Exponential);

        for &m_orig in &[-1.5, 0.0, 1.0, 2.5, 3.5] {
            let x = bounds.to_unbounded(m_orig);
            let m_back = bounds.to_bounded(x);
            assert!(
                (m_orig - m_back).abs() < 1e-12,
                "Exponential roundtrip failed for m={}: got {}",
                m_orig,
                m_back
            );
        }
    }

    #[test]
    fn test_bandpass_transform_roundtrip() {
        let bounds = ParameterBounds::new(0.0, 2.0).with_transform(TransformType::Bandpass);

        // Test across the full range
        for &m_orig in &[0.1, 0.5, 1.0, 1.5, 1.9] {
            let x = bounds.to_unbounded(m_orig);
            let m_back = bounds.to_bounded(x);
            assert!(
                (m_orig - m_back).abs() < 1e-10,
                "Bandpass roundtrip failed for m={}: got {} (x={})",
                m_orig,
                m_back,
                x
            );
        }
    }

    #[test]
    fn test_exponential_jacobian() {
        let bounds = ParameterBounds::new(0.0, 1.0).with_transform(TransformType::Exponential);

        for &x in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let dx = 1e-6;
            let numerical = (bounds.to_bounded(x + dx) - bounds.to_bounded(x - dx)) / (2.0 * dx);
            let analytical = bounds.transform_jacobian(x);
            assert!(
                (numerical - analytical).abs() < 1e-5,
                "Exponential Jacobian failed at x={}: numerical={}, analytical={}",
                x,
                numerical,
                analytical
            );
        }
    }

    #[test]
    fn test_bandpass_jacobian() {
        let bounds = ParameterBounds::new(0.0, 2.0).with_transform(TransformType::Bandpass);

        for &x in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let dx = 1e-6;
            let numerical = (bounds.to_bounded(x + dx) - bounds.to_bounded(x - dx)) / (2.0 * dx);
            let analytical = bounds.transform_jacobian(x);
            assert!(
                (numerical - analytical).abs() / numerical.abs().max(1e-15) < 1e-4,
                "Bandpass Jacobian failed at x={}: numerical={}, analytical={}",
                x,
                numerical,
                analytical
            );
        }
    }

}
