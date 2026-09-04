// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Monotone radial spline shared by the `SFMTOOL_FISHEYE` and
//! `SFMTOOL_PINHOLE` camera models.
//!
//! Both models write their radial map as `r(d) = f·(d + δ(d))`: a
//! distortion-free base plus a dimensionless correction
//! `δ(d) = Σ cᵢ·Bᵢ(d)` evaluated here. `d` is whichever radial coordinate the
//! base model projects a ray to — the incidence angle `θ` in radians for the
//! equidistant base, the normalized image-plane radius `ρ = tan θ` for the
//! pinhole base — and `d_max` is the model's domain-end parameter
//! (`bspline_theta_max` / `bspline_rho_max`). Everything below is arithmetic
//! on that scalar and on the caller's coefficient slice: one basis serves both
//! models, and nothing here allocates.
//!
//! The basis is a cubic open-uniform (clamped) B-spline on `[0, d_max]` with
//! the **first two** functions of the full clamped basis omitted and their
//! coefficients pinned to zero — the center-anchored gauge, which fixes
//! `δ(0) = 0` and `δ'(0) = 0` by construction. Beyond `d_max` the correction is
//! held constant at `δ(d_max)` with zero slope, so the map continues linearly
//! with `r' = f`.
//!
//! See `specs/core/camera/sfmtool-fisheye-kernels.md` for the design — basis
//! evaluation, the monotonicity invariant and where it is enforced — and
//! `specs/formats/sfmtool-camera-models.md` for why the gauge is anchored at
//! the centre rather than left free.

/// Number of basis functions active at any `d` (cubic local support). The
/// per-`d` outputs of [`basis_at`] are fixed arrays of this length.
pub(crate) const BSPLINE_SUPPORT: usize = 4;

/// Minimum coefficient count for the spline to be defined: `N` coefficients
/// span a clamped cubic basis of `N + 2` functions, and a cubic basis needs
/// at least 4. Coefficient vectors shorter than this evaluate as `δ ≡ 0`.
pub(crate) const MIN_BSPLINE_COEFFS: usize = 2;

/// Whether `bspline` evaluates as the identity `δ ≡ 0` — empty (or below
/// [`MIN_BSPLINE_COEFFS`]) or every coefficient exactly `0.0`.
///
/// The distortion kernels short-circuit on this to the distortion-free base
/// model's arithmetic, keeping a zero-spline camera bit-identical to that base
/// (the same convention as `SIMPLE_RADIAL_FISHEYE` at `k1 == 0.0`: exact zero,
/// not an epsilon).
pub(crate) fn bspline_is_identity(bspline: &[f64]) -> bool {
    bspline.len() < MIN_BSPLINE_COEFFS || bspline.iter().all(|&c| c == 0.0)
}

/// Whether the spline contributes nothing to the radial map — either it is
/// the identity by its coefficients ([`bspline_is_identity`]) or its domain
/// end is degenerate (`d_max` not positive and finite), which leaves no
/// interval for the basis to live on.
///
/// This is the short-circuit predicate the distortion kernels use, so a
/// camera with live coefficients but a degenerate `d_max` runs the base
/// model's arithmetic bit for bit rather than merely agreeing with it:
/// [`delta_and_deriv`] already returns `(0, 0)` on such a domain, but only the
/// short-circuit reproduces the base model's rounding exactly.
pub(crate) fn bspline_is_inactive(bspline: &[f64], d_max: f64) -> bool {
    bspline_is_identity(bspline) || !(d_max > 0.0 && d_max.is_finite())
}

/// Knot `i` of the clamped cubic knot vector for `m` full basis functions on
/// `[0, d_max]`: four zeros, `m − 4` uniform interior knots, four at `d_max`
/// (`m + 4` knots, indices `0..=m + 3`).
fn knot(i: usize, m: usize, d_max: f64) -> f64 {
    if i <= 3 {
        0.0
    } else if i >= m {
        d_max
    } else {
        d_max * (i - 3) as f64 / (m - 3) as f64
    }
}

/// The knot span index `k` (with `t_k ≤ d < t_{k+1}`) for `d ∈ [0, d_max]`,
/// in `3..=m − 1`; `d = d_max` lands in the last non-empty span.
fn span_index(d: f64, m: usize, d_max: f64) -> usize {
    let segs = m - 3;
    3 + ((d / d_max * segs as f64).floor() as usize).min(segs - 1)
}

/// Cox–de Boor basis evaluation (Piegl & Tiller A2.2) of the `p + 1` degree-`p`
/// functions `N^p_{k−p} .. N^p_k` at `u` in span `k`, in `out[0..=p]`
/// (`p ≤ 3`; the remaining entries stay zero).
fn basis_funs(k: usize, u: f64, p: usize, m: usize, d_max: f64) -> [f64; BSPLINE_SUPPORT] {
    let mut n = [0.0f64; BSPLINE_SUPPORT];
    let mut left = [0.0f64; BSPLINE_SUPPORT];
    let mut right = [0.0f64; BSPLINE_SUPPORT];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knot(k + 1 - j, m, d_max);
        right[j] = knot(k + j, m, d_max) - u;
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

/// The active cubic basis functions and their derivatives at the radial
/// coordinate `d`.
///
/// `n_coeffs` is the coefficient count (`≥ MIN_BSPLINE_COEFFS`); the full
/// clamped basis has `n_coeffs + 2` functions, of which the first two are the
/// gauge-anchored pair with no coefficient. Returns
/// `(first, values, derivatives)` where `first` is the **full-basis** index of
/// `values[0]`: full index `first + j` maps to coefficient index
/// `first + j − 2`, and indices below 2 are the anchored pair (coefficient
/// zero). `d` is clamped to `[0, d_max]`.
///
/// This is the per-observation primitive a bundle-adjustment kernel reuses:
/// the [`BSPLINE_SUPPORT`] entries are the only non-zero columns of
/// `∂δ/∂c` at this `d`.
pub(crate) fn basis_at(
    n_coeffs: usize,
    d_max: f64,
    d: f64,
) -> (usize, [f64; BSPLINE_SUPPORT], [f64; BSPLINE_SUPPORT]) {
    debug_assert!(n_coeffs >= MIN_BSPLINE_COEFFS);
    debug_assert!(d_max > 0.0);
    let m = n_coeffs + 2;
    let u = d.clamp(0.0, d_max);
    let k = span_index(u, m, d_max);
    let values = basis_funs(k, u, 3, m, d_max);
    // N³'_i = 3·(N²_i/(t_{i+3} − t_i) − N²_{i+1}/(t_{i+4} − t_{i+1})), terms
    // with a zero-span denominator dropped (their N² factor is zero there).
    // The active degree-2 functions in span `k` are N²_{k−2..k} = quad[0..3].
    let quad = basis_funs(k, u, 2, m, d_max);
    let mut derivatives = [0.0f64; BSPLINE_SUPPORT];
    for (j, dv) in derivatives.iter_mut().enumerate() {
        let i = k - 3 + j;
        let a = if j >= 1 {
            let den = knot(i + 3, m, d_max) - knot(i, m, d_max);
            if den > 0.0 {
                quad[j - 1] / den
            } else {
                0.0
            }
        } else {
            0.0
        };
        let b = if j <= 2 {
            let den = knot(i + 4, m, d_max) - knot(i + 1, m, d_max);
            if den > 0.0 {
                quad[j] / den
            } else {
                0.0
            }
        } else {
            0.0
        };
        *dv = 3.0 * (a - b);
    }
    (k - 3, values, derivatives)
}

/// `(δ(d), δ'(d))` for the given spline coefficients.
///
/// Below [`MIN_BSPLINE_COEFFS`] coefficients (or a `d_max` that is not
/// positive and finite — `+∞` included, which would put every knot at
/// infinity and turn the basis recurrence into `inf · 0`) this is the
/// identity `(0, 0)`. Beyond `d_max` the correction is held constant:
/// `δ(d) = δ(d_max)` with `δ'(d) = 0`, so the radial map continues with unit
/// slope there.
pub(crate) fn delta_and_deriv(bspline: &[f64], d_max: f64, d: f64) -> (f64, f64) {
    if bspline.len() < MIN_BSPLINE_COEFFS || d_max <= 0.0 || !d_max.is_finite() {
        return (0.0, 0.0);
    }
    let (first, values, derivatives) = basis_at(bspline.len(), d_max, d);
    let mut delta = 0.0;
    let mut deriv = 0.0;
    for j in 0..BSPLINE_SUPPORT {
        let full = first + j;
        if full < 2 {
            continue; // gauge-anchored pair: coefficient pinned to zero
        }
        let c = bspline[full - 2];
        delta += c * values[j];
        deriv += c * derivatives[j];
    }
    if d > d_max {
        deriv = 0.0;
    }
    (delta, deriv)
}

/// `δ(d)` alone — [`delta_and_deriv`] without the derivative.
pub(crate) fn delta(bspline: &[f64], d_max: f64, d: f64) -> f64 {
    delta_and_deriv(bspline, d_max, d).0
}

/// Whether the radial map `d_d(d) = d + δ(d)` is strictly increasing
/// (`1 + δ'(d) > 0`) over `[0, min(d_span, d_max)]`.
///
/// Beyond `d_max` the slope is exactly 1, so only the spline's own domain
/// needs checking. Two-stage: a **sufficient** convexity test on the derivative
/// spline's control points, which proves monotonicity outright over the whole
/// domain when it passes, and — only when that conservative test fails — a
/// dense sampling of `δ'` over the requested span.
///
/// This is the bundle adjustment's spline step guard's primitive
/// (`bspline_step_admissible`).
pub(crate) fn bspline_is_monotone(bspline: &[f64], d_max: f64, d_span: f64) -> bool {
    if bspline.len() < MIN_BSPLINE_COEFFS || d_max <= 0.0 || !d_max.is_finite() {
        // Identity map: d_d = d. The domain-end half of the guard must match
        // `delta_and_deriv`'s exactly, or a `d_max` this reports monotone
        // on would still evaluate a basis (and `+∞` knots evaluate to NaN).
        return true;
    }
    let m = bspline.len() + 2;
    let coeff = |i: usize| if i < 2 { 0.0 } else { bspline[i - 2] };
    let mut sufficient = true;
    for i in 0..m - 1 {
        let den = knot(i + 4, m, d_max) - knot(i + 1, m, d_max);
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
    let end = d_span.min(d_max).max(0.0);
    // 64 samples per knot span: dense against a quadratic per-span δ'.
    let n = 64 * (m - 3);
    (0..=n).all(|s| {
        let t = end * s as f64 / n as f64;
        1.0 + delta_and_deriv(bspline, d_max, t).1 > 0.0
    })
}
