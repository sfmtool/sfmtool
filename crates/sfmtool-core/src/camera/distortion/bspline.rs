// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Monotone radial spline for the `SFMTOOL_FISHEYE` camera model.
//!
//! The model's radial map is `r(θ) = f·(θ + δ(θ))`: an equidistant base plus
//! a dimensionless correction `δ(θ) = Σ cᵢ·Bᵢ(θ)` evaluated here. The basis
//! is a cubic open-uniform (clamped) B-spline on `[0, θ_max]` with the
//! **first two** basis functions of the full clamped basis omitted — their
//! coefficients are pinned to zero — so `δ(0) = 0` and `δ'(0) = 0` by
//! construction. That is the center-anchored gauge: the focal length alone
//! carries the central scale, and the spline cannot express a central-scale
//! correction, which keeps `f` a pure multiplier of an `f`-independent
//! distorted coordinate (the property the bundle adjustment's focal column
//! relies on). Beyond `θ_max` the correction is held constant at
//! `δ(θ_max)` with zero slope, so the map continues linearly with `r' = f`.
//!
//! A coefficient vector of `N` values spans a full clamped basis of `N + 2`
//! functions, which needs `N ≥ 2` ([`MIN_BSPLINE_COEFFS`]); shorter
//! coefficient vectors (including the empty one) evaluate as the identity
//! `δ ≡ 0`. At any `θ` at most [`BSPLINE_SUPPORT`] basis functions are
//! non-zero (cubic local support), which is what lets a bundle-adjustment
//! kernel treat the per-observation coefficient block as fixed-size.
//!
//! Everything here is pure arithmetic on the caller's slice — no allocation.

/// Number of basis functions active at any `θ` (cubic local support). The
/// per-`θ` outputs of [`basis_at`] are fixed arrays of this length.
pub(crate) const BSPLINE_SUPPORT: usize = 4;

/// Minimum coefficient count for the spline to be defined: `N` coefficients
/// span a clamped cubic basis of `N + 2` functions, and a cubic basis needs
/// at least 4. Coefficient vectors shorter than this evaluate as `δ ≡ 0`.
pub(crate) const MIN_BSPLINE_COEFFS: usize = 2;

/// Whether `bspline` evaluates as the identity `δ ≡ 0` — empty (or below
/// [`MIN_BSPLINE_COEFFS`]) or every coefficient exactly `0.0`.
///
/// The distortion kernels short-circuit on this to the distortion-free
/// equidistant arithmetic, keeping a zero-spline `SFMTOOL_FISHEYE`
/// bit-identical to `EQUIDISTANT_FISHEYE` (the same convention as
/// `SIMPLE_RADIAL_FISHEYE` at `k1 == 0.0`: exact zero, not an epsilon).
pub(crate) fn bspline_is_identity(bspline: &[f64]) -> bool {
    bspline.len() < MIN_BSPLINE_COEFFS || bspline.iter().all(|&c| c == 0.0)
}

/// Whether the spline contributes nothing to the radial map — either it is
/// the identity by its coefficients ([`bspline_is_identity`]) or its domain
/// end is degenerate (`theta_max` not positive and finite), which leaves no
/// interval for the basis to live on.
///
/// This is the short-circuit predicate the distortion kernels use, so a
/// camera with live coefficients but a degenerate `theta_max` runs the
/// `EQUIDISTANT_FISHEYE` arithmetic bit for bit rather than merely agreeing
/// with it: [`delta_and_deriv`] already returns `(0, 0)` on such a domain,
/// but only the short-circuit reproduces the equidistant rounding exactly.
pub(crate) fn bspline_is_inactive(bspline: &[f64], theta_max: f64) -> bool {
    bspline_is_identity(bspline) || !(theta_max > 0.0 && theta_max.is_finite())
}

/// Knot `i` of the clamped cubic knot vector for `m` full basis functions on
/// `[0, theta_max]`: four zeros, `m − 4` uniform interior knots, four at
/// `theta_max` (`m + 4` knots, indices `0..=m + 3`).
fn knot(i: usize, m: usize, theta_max: f64) -> f64 {
    if i <= 3 {
        0.0
    } else if i >= m {
        theta_max
    } else {
        theta_max * (i - 3) as f64 / (m - 3) as f64
    }
}

/// The knot span index `k` (with `t_k ≤ θ < t_{k+1}`) for `θ ∈ [0, θ_max]`,
/// in `3..=m − 1`; `θ = θ_max` lands in the last non-empty span.
fn span_index(theta: f64, m: usize, theta_max: f64) -> usize {
    let segs = m - 3;
    3 + ((theta / theta_max * segs as f64).floor() as usize).min(segs - 1)
}

/// Cox–de Boor basis evaluation (Piegl & Tiller A2.2) of the `p + 1` degree-`p`
/// functions `N^p_{k−p} .. N^p_k` at `u` in span `k`, in `out[0..=p]`
/// (`p ≤ 3`; the remaining entries stay zero).
fn basis_funs(k: usize, u: f64, p: usize, m: usize, theta_max: f64) -> [f64; BSPLINE_SUPPORT] {
    let mut n = [0.0f64; BSPLINE_SUPPORT];
    let mut left = [0.0f64; BSPLINE_SUPPORT];
    let mut right = [0.0f64; BSPLINE_SUPPORT];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knot(k + 1 - j, m, theta_max);
        right[j] = knot(k + j, m, theta_max) - u;
        let mut saved = 0.0;
        for r in 0..j {
            // The denominator is the `j`-interval knot span containing the
            // non-empty span `[t_k, t_{k+1})`, hence strictly positive.
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// The active cubic basis functions and their derivatives at `theta`.
///
/// `n_coeffs` is the coefficient count (`≥ MIN_BSPLINE_COEFFS`); the full
/// clamped basis has `n_coeffs + 2` functions, of which the first two are the
/// gauge-anchored pair with no coefficient. Returns
/// `(first, values, derivatives)` where `first` is the **full-basis** index of
/// `values[0]`: full index `first + j` maps to coefficient index
/// `first + j − 2`, and indices below 2 are the anchored pair (coefficient
/// zero). `theta` is clamped to `[0, theta_max]`.
///
/// This is the per-observation primitive a bundle-adjustment kernel reuses:
/// the [`BSPLINE_SUPPORT`] entries are the only non-zero columns of
/// `∂δ/∂c` at this `theta`.
pub(crate) fn basis_at(
    n_coeffs: usize,
    theta_max: f64,
    theta: f64,
) -> (usize, [f64; BSPLINE_SUPPORT], [f64; BSPLINE_SUPPORT]) {
    debug_assert!(n_coeffs >= MIN_BSPLINE_COEFFS);
    debug_assert!(theta_max > 0.0);
    let m = n_coeffs + 2;
    let u = theta.clamp(0.0, theta_max);
    let k = span_index(u, m, theta_max);
    let values = basis_funs(k, u, 3, m, theta_max);
    // N³'_i = 3·(N²_i/(t_{i+3} − t_i) − N²_{i+1}/(t_{i+4} − t_{i+1})), terms
    // with a zero-span denominator dropped (their N² factor is zero there).
    // The active degree-2 functions in span `k` are N²_{k−2..k} = quad[0..3].
    let quad = basis_funs(k, u, 2, m, theta_max);
    let mut derivatives = [0.0f64; BSPLINE_SUPPORT];
    for (j, d) in derivatives.iter_mut().enumerate() {
        let i = k - 3 + j;
        let a = if j >= 1 {
            let den = knot(i + 3, m, theta_max) - knot(i, m, theta_max);
            if den > 0.0 {
                quad[j - 1] / den
            } else {
                0.0
            }
        } else {
            0.0
        };
        let b = if j <= 2 {
            let den = knot(i + 4, m, theta_max) - knot(i + 1, m, theta_max);
            if den > 0.0 {
                quad[j] / den
            } else {
                0.0
            }
        } else {
            0.0
        };
        *d = 3.0 * (a - b);
    }
    (k - 3, values, derivatives)
}

/// `(δ(θ), δ'(θ))` for the given spline coefficients.
///
/// Below [`MIN_BSPLINE_COEFFS`] coefficients (or a `theta_max` that is not
/// positive and finite — `+∞` included, which would put every knot at
/// infinity and turn the basis recurrence into `inf · 0`) this is the
/// identity `(0, 0)`. Beyond `theta_max` the correction is held constant:
/// `δ(θ) = δ(θ_max)` with `δ'(θ) = 0`, so the radial map continues with unit
/// slope there.
pub(crate) fn delta_and_deriv(bspline: &[f64], theta_max: f64, theta: f64) -> (f64, f64) {
    if bspline.len() < MIN_BSPLINE_COEFFS || theta_max <= 0.0 || !theta_max.is_finite() {
        return (0.0, 0.0);
    }
    let (first, values, derivatives) = basis_at(bspline.len(), theta_max, theta);
    let mut d = 0.0;
    let mut dp = 0.0;
    for j in 0..BSPLINE_SUPPORT {
        let full = first + j;
        if full < 2 {
            continue; // gauge-anchored pair: coefficient pinned to zero
        }
        let c = bspline[full - 2];
        d += c * values[j];
        dp += c * derivatives[j];
    }
    if theta > theta_max {
        dp = 0.0;
    }
    (d, dp)
}

/// `δ(θ)` alone — [`delta_and_deriv`] without the derivative.
pub(crate) fn delta(bspline: &[f64], theta_max: f64, theta: f64) -> f64 {
    delta_and_deriv(bspline, theta_max, theta).0
}

/// Whether the radial map `θ_d(θ) = θ + δ(θ)` is strictly increasing
/// (`1 + δ'(θ) > 0`) over `[0, min(theta_span, theta_max)]`.
///
/// Beyond `theta_max` the slope is exactly 1, so only the spline's own domain
/// needs checking. Two-stage: first the **sufficient** condition via the
/// derivative spline's control points `3·(a_{i+1} − a_i)/(t_{i+4} − t_{i+1})`
/// (the quadratic `δ'` is a convex combination of them, so all
/// `1 + d_i > 0` proves monotonicity outright); when that conservative test
/// fails, a dense sampling of `δ'` over the requested span decides.
///
/// This is the bundle adjustment's spline step guard's primitive
/// (`bspline_step_admissible`).
pub(crate) fn bspline_is_monotone(bspline: &[f64], theta_max: f64, theta_span: f64) -> bool {
    if bspline.len() < MIN_BSPLINE_COEFFS || theta_max <= 0.0 || !theta_max.is_finite() {
        // Identity map: θ_d = θ. The domain-end half of the guard must match
        // `delta_and_deriv`'s exactly, or a `theta_max` this reports monotone
        // on would still evaluate a basis (and `+∞` knots evaluate to NaN).
        return true;
    }
    let m = bspline.len() + 2;
    let coeff = |i: usize| if i < 2 { 0.0 } else { bspline[i - 2] };
    let mut sufficient = true;
    for i in 0..m - 1 {
        let den = knot(i + 4, m, theta_max) - knot(i + 1, m, theta_max);
        if den <= 0.0 {
            continue;
        }
        if 1.0 + 3.0 * (coeff(i + 1) - coeff(i)) / den <= 0.0 {
            sufficient = false;
            break;
        }
    }
    if sufficient {
        return true;
    }
    let end = theta_span.min(theta_max).max(0.0);
    // 64 samples per knot span: dense against a quadratic per-span δ'.
    let n = 64 * (m - 3);
    (0..=n).all(|s| {
        let t = end * s as f64 / n as f64;
        1.0 + delta_and_deriv(bspline, theta_max, t).1 > 0.0
    })
}
