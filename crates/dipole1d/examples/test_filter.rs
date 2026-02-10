//! Test that the Hankel filter produces correct results for known transforms
//! 
//! Known pair: ∫₀^∞ λ * exp(-a*λ) * J₀(λ*r) dλ = a/(a²+r²)^(3/2)
//! Also: ∫₀^∞ λ² * exp(-a*λ) * J₁(λ*r) dλ = 3ar/((a²+r²)^(5/2))

use dipole1d::HankelFilter;

fn main() {
    let filter = HankelFilter::Kong201;
    let coeffs = filter.coefficients();
    
    let a = 1.0_f64;
    
    println!("Testing Hankel transform accuracy:");
    println!("Known pair: ∫₀^∞ λ * exp(-a*λ) * J₀(λ*r) dλ = a/(a²+r²)^(3/2)");
    println!();
    
    for &r in &[0.1, 1.0, 10.0, 100.0, 1000.0] {
        // J0 transform: kernel K(λ) = λ*exp(-a*λ), result should be a/(a²+r²)^(3/2)
        let expected_j0 = a / (a * a + r * r).powf(1.5);
        
        let mut sum_j0 = 0.0_f64;
        for i in 0..coeffs.base.len() {
            let lam = coeffs.base[i] / r;
            let kernel = lam * (-a * lam).exp();
            sum_j0 += kernel * coeffs.j0[i];
        }
        let computed_j0 = sum_j0 / r;  // 1/r normalization
        
        let ratio_j0 = computed_j0 / expected_j0;
        println!("r={:8.1}, J0: computed={:12.6e}, expected={:12.6e}, ratio={:.6}", 
            r, computed_j0, expected_j0, ratio_j0);
    }
    
    println!();
    println!("Testing J1 transform: ∫₀^∞ λ² * exp(-a*λ) * J₁(λ*r) dλ = 3ar/(a²+r²)^(5/2)");
    println!();
    
    for &r in &[0.1, 1.0, 10.0, 100.0, 1000.0] {
        let expected_j1 = 3.0 * a * r / (a * a + r * r).powf(2.5);
        
        let mut sum_j1 = 0.0_f64;
        for i in 0..coeffs.base.len() {
            let lam = coeffs.base[i] / r;
            let kernel = lam * lam * (-a * lam).exp();
            sum_j1 += kernel * coeffs.j1[i];
        }
        let computed_j1 = sum_j1 / r;
        
        let ratio_j1 = computed_j1 / expected_j1;
        println!("r={:8.1}, J1: computed={:12.6e}, expected={:12.6e}, ratio={:.6}", 
            r, computed_j1, expected_j1, ratio_j1);
    }
}
