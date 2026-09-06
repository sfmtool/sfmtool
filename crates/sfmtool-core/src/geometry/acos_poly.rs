// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The focal vote's arccosine, as a polynomial rather than libm's.
//!
//! One [`ACOS_POLY`] table and one operation order, read by the scalar
//! evaluation here and by its AVX2 twin `acos_pd` in
//! [`crate::geometry::simd`]. Both properties the vote depends on follow from
//! that arrangement: the two dispatch arms produce identical bits, and they
//! produce the *same* bits on every platform, where libm's `acos` differs in
//! the last bits between operating systems. A second copy of the table would
//! be free to drift, which is why there is not one.
//!
//! Nothing here is architecture-gated: these are ordinary `f64` (and `f32`)
//! arithmetic that computes the same bits everywhere, a property of the
//! kernel rather than of `x86_64`.

/// Diagnostic switch restoring the platform libm `acos` in the focal vote's
/// rotation residuals, set by `SFMTOOL_FOCAL_VOTE_LIBM_ACOS`.
///
/// The production path is [`acos_poly_scalar`] and its vector twin, which are
/// platform-deterministic where libm is not; this exists only to reproduce an
/// older run's bits when a difference has to be attributed.
static LIBM_ACOS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_FOCAL_VOTE_LIBM_ACOS").is_some());

/// Whether the rotation residuals take the libm `acos` instead of
/// [`acos_poly_scalar`]. See [`LIBM_ACOS`].
#[inline]
pub(crate) fn libm_acos_enabled() -> bool {
    *LIBM_ACOS
}

/// Coefficients of the degree-13 polynomial `P` behind [`acos_poly_scalar`] and
/// its AVX2 twin `acos_pd` in `crate::geometry::simd`, ascending
/// (`A[0]` multiplies `z⁰`).
///
/// Degree-13 Chebyshev interpolation of `P(z) = (asin(√z) − √z)/(z·√z)` on
/// `[0, 0.25]`, node values computed with exact rational series arithmetic and
/// converted to the monomial basis. The trailing coefficients are
/// interpolation artifacts that compensate one another and are exact as
/// written — rounding them degrades the fit.
///
/// One table, read by both evaluations: a second copy would be free to drift,
/// and the scalar/vector bit-identity is the whole point of the arrangement.
pub(crate) const ACOS_POLY: [f64; 14] = [
    0.16666666666666666,
    0.07500000000000406,
    0.04464285714150171,
    0.030381944571586314,
    0.02237215339997836,
    0.017352913993464503,
    0.013962288953824856,
    0.01158174875867126,
    0.009513962068006003,
    0.009846417300000855,
    0.0012909120006875199,
    0.02336097864008306,
    -0.024103731139602444,
    0.03238761648605816,
];

/// [`ACOS_POLY`] at single precision, for the `f32` residual arms of the focal
/// vote — the same table, narrowed once, never a second fit.
///
/// The coefficients past `A[9]` are interpolation artifacts that compensate one
/// another; at `z ≤ 0.25` they contribute below `f32` epsilon, so narrowing
/// them changes nothing the evaluation can see.
pub(crate) static ACOS_POLY_F32: [f32; 14] = {
    let mut out = [0.0f32; 14];
    let mut k = 0;
    while k < 14 {
        out[k] = ACOS_POLY[k] as f32;
        k += 1;
    }
    out
};

/// `asin(s)` for `s ∈ [0, 1]` at single precision, the scalar twin of the AVX2
/// `asin_ps` in `crate::geometry::simd`.
///
/// The same asin core [`acos_poly_scalar`] evaluates, read forwards instead of
/// through `acos`: for `s ≤ 0.5`, `z = s²` and `asin(s) = s + s·z·P(z)`; for
/// `s > 0.5`, `w = (1 − s)/2`, `t = √w`, and `asin(s) = π/2 − 2·(t + t·w·P(w))`.
/// Both branches evaluate `P` on `[0, 0.25]`, which is the range
/// [`ACOS_POLY`] was fitted on.
///
/// This exists because the rotation cell's `f32` arm measures the angle
/// between two unit rays through the **norm of their cross product** rather
/// than the arccosine of their dot: near zero the dot is `1 − θ²/2`, whose
/// `f32` rounding swamps `θ` itself, while `|r₁ × r₂| = sin θ` carries the
/// small angle in its own leading digits.
pub(crate) fn asin_poly_scalar_f32(s: f32) -> f32 {
    let big = s > 0.5;
    let z = if big { (1.0 - s) * 0.5 } else { s * s };
    let base = if big { z.sqrt() } else { s };
    let mut p = ACOS_POLY_F32[13];
    let mut k = 13usize;
    while k > 0 {
        k -= 1;
        p = p * z + ACOS_POLY_F32[k];
    }
    let r = base + (base * z) * p;
    if big {
        std::f32::consts::FRAC_PI_2 - (r + r)
    } else {
        r
    }
}

/// `acos(d)` for `d ∈ [−1, 1]` by polynomial evaluation, the scalar twin of
/// the AVX2 `acos_pd` in `crate::geometry::simd`.
///
/// Through the asin core: `a = |d|`; for `a ≤ 0.5`, `z = a²` and `s = a`; for
/// `a > 0.5`, `z = (1 − a)/2` and `s = √z` (so `z ∈ [0, 0.25]` and
/// `s ∈ [0, 0.5]` either way); `asin(s) = s + s·z·P(z)` with `P` the
/// [`ACOS_POLY`] Horner evaluation from the top coefficient down; then
/// `acos = π/2 − copysign(asin, d)` for the small branch, `2·asin` or
/// `π − 2·asin` by the sign of `d` for the big one. Measured accuracy is 1 ULP
/// against libm over dense `[−1, 1]` sampling plus adversarial near-`±1`
/// populations, and a NaN argument yields a NaN as libm's does.
///
/// Two properties are why the focal vote uses this rather than [`f64::acos`],
/// and both depend on the operation order below matching the vector form term
/// for term: the two dispatch arms of the rotation residual produce identical
/// bits, and they produce the *same* bits on every platform, where libm's
/// `acos` differs in the last bits between operating systems.
pub(crate) fn acos_poly_scalar(d: f64) -> f64 {
    let a = d.abs();
    let big = a > 0.5;
    let z = if big { (1.0 - a) * 0.5 } else { a * a };
    let s = if big { z.sqrt() } else { a };
    let mut p = ACOS_POLY[13];
    let mut k = 13usize;
    while k > 0 {
        k -= 1;
        p = p * z + ACOS_POLY[k];
    }
    let r = s + (s * z) * p;
    if big {
        let two_r = r + r;
        if d < 0.0 {
            std::f64::consts::PI - two_r
        } else {
            two_r
        }
    } else {
        std::f64::consts::FRAC_PI_2 - f64::copysign(r, d)
    }
}

#[cfg(test)]
mod tests;
