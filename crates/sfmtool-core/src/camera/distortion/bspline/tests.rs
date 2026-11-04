// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The spline evaluation against a reference written the textbook way: the
//! Cox–de Boor recurrence (Piegl & Tiller A2.2) on the knot values themselves,
//! with a division for every knot interval, run once for the cubic values and
//! once for the quadratic functions the derivative is built from. The
//! evaluation in the parent module counts the knots in spacings and multiplies
//! by reciprocals instead, so the two agree to rounding and no more.

use super::*;

/// Reference basis values: the `p + 1` degree-`p` functions active in span
/// `k` at `u`, from the knots in the units of `d`.
fn reference_basis_funs(k: usize, u: f64, p: usize, m: usize, d_max: f64) -> [f64; 4] {
    let mut n = [0.0f64; 4];
    let mut left = [0.0f64; 4];
    let mut right = [0.0f64; 4];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knot(k + 1 - j, m, d_max);
        right[j] = knot(k + j, m, d_max) - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// Reference `basis_at`: span from `d / d_max`, values and derivatives with
/// divisions by the knot intervals.
fn reference_basis_at(n_coeffs: usize, d_max: f64, d: f64) -> (usize, [f64; 4], [f64; 4]) {
    let m = n_coeffs + 2;
    let segs = m - 3;
    let u = d.clamp(0.0, d_max);
    let k = 3 + ((u / d_max * segs as f64).floor() as usize).min(segs - 1);
    let values = reference_basis_funs(k, u, 3, m, d_max);
    let quad = reference_basis_funs(k, u, 2, m, d_max);
    let mut derivatives = [0.0f64; 4];
    for (j, dv) in derivatives.iter_mut().enumerate() {
        let i = k - 3 + j;
        let mut a = 0.0;
        if j >= 1 {
            let den = knot(i + 3, m, d_max) - knot(i, m, d_max);
            if den > 0.0 {
                a = quad[j - 1] / den;
            }
        }
        let mut b = 0.0;
        if j <= 2 {
            let den = knot(i + 4, m, d_max) - knot(i + 1, m, d_max);
            if den > 0.0 {
                b = quad[j] / den;
            }
        }
        *dv = 3.0 * (a - b);
    }
    (k - 3, values, derivatives)
}

/// Reference `(δ, δ')`, linear continuation past `d_max` included.
fn reference_delta_and_deriv(bspline: &[f64], d_max: f64, d: f64) -> (f64, f64) {
    let (first, values, derivatives) = reference_basis_at(bspline.len(), d_max, d);
    let (mut delta, mut deriv) = (0.0, 0.0);
    for j in 0..4 {
        if first + j >= 2 {
            delta += bspline[first + j - 2] * values[j];
            deriv += bspline[first + j - 2] * derivatives[j];
        }
    }
    if d > d_max {
        delta += deriv * (d - d_max);
    }
    (delta, deriv)
}

/// Sample points for a basis of `segs` spans on `[0, d_max]`: every knot, a
/// relative `1e-12` either side of it, seven points inside every span, below
/// zero, and past `d_max` on the continuation.
fn samples(segs: usize, d_max: f64) -> Vec<f64> {
    let h = d_max / segs as f64;
    let mut out = vec![-0.25 * d_max, 0.0];
    for i in 0..=segs {
        let t = h * i as f64;
        out.extend([t, t * (1.0 - 1e-12), t * (1.0 + 1e-12)]);
        if i < segs {
            out.extend((1..8).map(|s| t + h * s as f64 / 8.0));
        }
    }
    out.extend([1.0 + 1e-12, 1.01, 1.5, 3.0].map(|f| d_max * f));
    out
}

/// Fixed pseudo-random coefficients in `[-0.5, 0.5)`.
fn coefficients(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// The largest difference seen in one quantity, and where.
#[derive(Default)]
struct Worst {
    diff: f64,
    at: String,
}

impl Worst {
    fn record(&mut self, diff: f64, at: &str) {
        // A NaN difference is recorded too, so it fails the bound.
        if diff > self.diff || diff.is_nan() {
            self.diff = diff;
            self.at = at.to_string();
        }
    }
}

#[test]
fn spline_evaluation_matches_the_textbook_recurrence() {
    // Each difference is divided by the quantity's scale: basis values and
    // `δ` as they are (the coefficients are below 1), derivatives by `1/h`
    // (they scale with it), and `δ` past `d_max` by the continuation's lever
    // arm `1 + (d − d_max)/h`. Both evaluations subtract a knot from a
    // coordinate of up to `segs` spacings, so each carries a rounding error of
    // a few `ε·segs`, and that is the bound.
    let mut worst: [Worst; 4] = Default::default();
    // Two coefficients (one span, no interior knot) up to twenty, so the
    // clamped boundary spans overlap, touch and are separated by uniform ones.
    for n_coeffs in [2, 3, 4, 5, 6, 8, 13, 20] {
        let segs = n_coeffs - 1;
        let bound = segs as f64 * f64::EPSILON;
        for d_max in [0.3, 0.84, 2.0, 3.1] {
            let per_d = segs as f64 / d_max;
            let bspline = coefficients(n_coeffs, (n_coeffs * 31) as u64 + d_max.to_bits());
            for d in samples(segs, d_max) {
                let at = format!("n = {n_coeffs}, d_max = {d_max}, d = {d}");
                let (first, values, derivs) = basis_at(n_coeffs, d_max, d);
                let (ref_first, ref_values, ref_derivs) = reference_basis_at(n_coeffs, d_max, d);
                assert_eq!(first, ref_first, "{at}");
                for j in 0..BSPLINE_SUPPORT {
                    worst[0].record((values[j] - ref_values[j]).abs() / bound, &at);
                    worst[1].record((derivs[j] - ref_derivs[j]).abs() / per_d / bound, &at);
                }

                let (value, slope) = delta_and_deriv(&bspline, d_max, d);
                let (ref_value, ref_slope) = reference_delta_and_deriv(&bspline, d_max, d);
                let lever = 1.0 + (d - d_max).max(0.0) * per_d;
                worst[2].record((value - ref_value).abs() / lever / bound, &at);
                worst[3].record((slope - ref_slope).abs() / per_d / bound, &at);
                // The value-only path sums the same products in the same order.
                assert_eq!(super::delta(&bspline, d_max, d), value, "{at}");
            }
        }
    }
    for (name, worst) in ["basis value", "basis derivative", "δ", "δ'"]
        .iter()
        .zip(worst)
    {
        assert!(
            worst.diff <= 16.0,
            "{name}: {} ε·segs at {}",
            worst.diff,
            worst.at
        );
    }
}
