// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Closed-form real roots of low-degree polynomials, shared by the minimal
//! solvers ([`crate::geometry::absolute_pose`]'s Lambda Twist cubic,
//! [`crate::geometry::epipolar_estimation`]'s 7-point rank-2 pencil cubic).
//! Pure functions, no iteration beyond the explicit Newton polish, bit-stable
//! across runs.

/// Leading coefficients below this are treated as zero: the cubic collapses to
/// the quadratic rather than dividing by a vanishing `c3`.
pub(crate) const CUBIC_LEADING_EPS: f64 = 1e-13;

/// Real roots of `c3·x³ + c2·x² + c1·x + c0`. Collapses to lower degree when
/// leading coefficients vanish.
pub(crate) fn solve_cubic(c3: f64, c2: f64, c1: f64, c0: f64) -> Vec<f64> {
    if c3.abs() < CUBIC_LEADING_EPS {
        return solve_quadratic(c2, c1, c0);
    }
    // Depressed cubic t³ + p·t + q via x = t − b/3.
    let (a, b, c) = (c2 / c3, c1 / c3, c0 / c3);
    let shift = a / 3.0;
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let mut roots = Vec::with_capacity(3);
    if disc > 0.0 {
        // One real root.
        let sqrt_disc = disc.sqrt();
        let u = (-q / 2.0 + sqrt_disc).cbrt();
        let v = (-q / 2.0 - sqrt_disc).cbrt();
        roots.push(u + v - shift);
    } else if disc.abs() <= 1e-300 || (p.abs() < 1e-300 && q.abs() < 1e-300) {
        // Triple/degenerate real root.
        let t = if p.abs() < 1e-300 { 0.0 } else { 3.0 * q / p };
        roots.push(t - shift);
    } else {
        // Three real roots (trigonometric form).
        let m = 2.0 * (-p / 3.0).sqrt();
        let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos();
        for k in 0..3 {
            let t = m * ((theta + 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos();
            roots.push(t - shift);
        }
    }
    roots
}

/// Real roots of `a·x² + b·x + c`.
pub(crate) fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < 1e-300 {
        if b.abs() < 1e-300 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    let sqrt_disc = disc.sqrt();
    // Numerically stable roots (avoid cancellation).
    let q = -0.5 * (b + b.signum() * sqrt_disc);
    let mut roots = Vec::with_capacity(2);
    if q.abs() > 1e-300 {
        roots.push(q / a);
        roots.push(c / q);
    } else {
        roots.push((-b + sqrt_disc) / (2.0 * a));
        roots.push((-b - sqrt_disc) / (2.0 * a));
    }
    roots
}

/// A few Newton steps polishing a cubic root against the exact polynomial.
pub(crate) fn polish_cubic_root(c3: f64, c2: f64, c1: f64, c0: f64, mut x: f64) -> f64 {
    for _ in 0..3 {
        let f = ((c3 * x + c2) * x + c1) * x + c0;
        let df = (3.0 * c3 * x + 2.0 * c2) * x + c1;
        if df.abs() < 1e-300 {
            break;
        }
        let step = f / df;
        x -= step;
        if step.abs() < 1e-15 * (1.0 + x.abs()) {
            break;
        }
    }
    x
}
