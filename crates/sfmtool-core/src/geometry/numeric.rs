// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Small numeric primitives shared across the geometry kernels.
//!
//! What lands here is arithmetic that more than one module has to perform
//! *identically*: RNG and threshold primitives feeding RANSAC sampling and
//! robust statistics, and the deterministic kernels of the focal vote — the
//! [`ACOS_POLY`] table with its scalar evaluation [`acos_poly_scalar`] (whose
//! AVX2 twin in `crate::geometry::simd` reads the same table) and the minimal
//! solvers' null space [`null9_from_8rows`]. A copy that drifts changes
//! reconstruction results without failing to compile, which is why these are
//! centralized rather than left co-located.
//!
//! Nothing here is architecture-gated: the focal vote's residuals and minimal
//! samples are ordinary `f64` arithmetic that computes the same bits on every
//! platform, and that is a property of the kernel, not of `x86_64`.

use nalgebra::Matrix3;

use crate::camera::{CameraIntrinsics, CameraModel};

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

/// Diagnostic switch restoring the `AᵀA` + 9×9 `symmetric_eigen` minimal
/// solvers in place of [`null9_from_8rows`], set by
/// `SFMTOOL_FOCAL_VOTE_EIGEN_MINSOLVE`.
static EIGEN_MINSOLVE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_FOCAL_VOTE_EIGEN_MINSOLVE").is_some());

/// Whether the minimal samples take the eigen solvers instead of
/// [`null9_from_8rows`]. See [`EIGEN_MINSOLVE`].
#[inline]
pub(crate) fn eigen_minsolve_enabled() -> bool {
    *EIGEN_MINSOLVE
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

/// Unit-norm right null vector of an 8×9 system by Gaussian elimination with
/// partial pivoting.
///
/// A generic rank-8 minimal sample has a one-dimensional null space, and this
/// returns it directly instead of taking the smallest eigenvector of `AᵀA`:
/// better conditioned (no squaring of the condition number), about an order of
/// magnitude cheaper, and free of an iterative decomposition. The pivot rule is
/// part of the determinism contract — strict `>` when scanning a column, so the
/// *first* maximal pivot is kept.
///
/// Rank-deficient input takes the last free column with the other free
/// coordinates zero: a deterministic member of the null space, and such
/// degenerate samples score few inliers and lose the RANSAC regardless.
/// `None` when the design carries no pivot at all, or the result is
/// non-finite.
pub(crate) fn null9_from_8rows(mut a: [[f64; 9]; 8]) -> Option<[f64; 9]> {
    let mut pivot_cols = [usize::MAX; 8];
    let mut rank = 0usize;
    let mut col = 0usize;
    while rank < 8 && col < 9 {
        let mut best = rank;
        let mut best_v = a[rank][col].abs();
        for (r, row) in a.iter().enumerate().skip(rank + 1) {
            let v = row[col].abs();
            if v > best_v {
                best = r;
                best_v = v;
            }
        }
        if best_v <= 0.0 {
            col += 1;
            continue;
        }
        a.swap(rank, best);
        let (upper, lower) = a.split_at_mut(rank + 1);
        let pivot_row = &upper[rank];
        for row in lower.iter_mut() {
            let f = row[col] / pivot_row[col];
            if f != 0.0 {
                row[col] = 0.0;
                for (x, &p) in row.iter_mut().zip(pivot_row.iter()).skip(col + 1) {
                    *x -= f * p;
                }
            }
        }
        pivot_cols[rank] = col;
        rank += 1;
        col += 1;
    }
    if rank == 0 {
        return None;
    }
    let mut is_pivot = [false; 9];
    for &pc in &pivot_cols[..rank] {
        is_pivot[pc] = true;
    }
    let free = (0..9).rev().find(|&c| !is_pivot[c])?;
    let mut v = [0.0f64; 9];
    v[free] = 1.0;
    for r in (0..rank).rev() {
        let pc = pivot_cols[r];
        let mut s = 0.0;
        for c in pc + 1..9 {
            if v[c] != 0.0 {
                s += a[r][c] * v[c];
            }
        }
        v[pc] = -s / a[r][pc];
    }
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    v.iter().all(|x| x.is_finite()).then_some(v)
}

/// SplitMix64 step: advance `state` and return the mixed output.
///
/// The deterministic RANSAC samplers seed one of these per kernel, so the
/// exact bit mixing is part of every sampling-dependent result.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Nearest rotation to `m` by polar decomposition (`U Vᵀ` from the SVD).
///
/// The conjugate-homography and pose-verification callers recover `R` only up
/// to scale *including sign*, so `M ≈ −R` must come back as `R`, not as the
/// (distant) proper projection of `−R`. `None` for a non-finite or degenerate
/// input.
pub(crate) fn polar_rotation(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let p = u * v_t;
    if !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(if p.determinant() < 0.0 { -p } else { p })
}

/// Nearest rotation to `m` by polar decomposition, **preserving orientation**.
///
/// The sibling of [`polar_rotation`], and the choice between them is the sign
/// convention, not the algorithm — both take `U Vᵀ` from the SVD and differ
/// only in what they do when that product reflects:
///
/// - [`polar_rotation`] negates the whole product, because its callers recover
///   `R` only up to scale *including sign*, so `M ≈ −R` must come back as `R`.
/// - this one folds the sign into the last singular direction
///   (`U · diag(1, 1, det) · Vᵀ`), which is the *proper* projection of `m`
///   itself. Its caller has a matrix that is already the right rotation to
///   within accumulated rounding — a Kabsch fit — and wants the nearest
///   rotation to that, not to its negation.
///
/// Returns `m` unchanged when the SVD does not produce both factors.
pub(crate) fn orthonormalized(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => {
            let d = (u * v_t).determinant().signum();
            u * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d) * v_t
        }
        _ => *m,
    }
}

/// Rotation angle of `r` in radians.
pub(crate) fn rotation_angle(r: &Matrix3<f64>) -> f64 {
    (((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)).acos()
}

/// The camera at focal `f` — identity for every model but the five the focal
/// release admits: `SIMPLE_PINHOLE`, `EQUIDISTANT_FISHEYE`,
/// `SIMPLE_RADIAL_FISHEYE`, `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`, whose
/// projections all multiply `f` onto a distorted coordinate that does not
/// itself read `f` (for the last two, the dimensionless radial spline rides on
/// the ray's own radial coordinate).
///
/// Focal optimization is gated on exactly those five models, so no other
/// camera ever sees a moved focal; this matches the bundle adjustment's focal
/// handling.
pub(crate) fn cam_at(cam: &CameraIntrinsics, f: f64) -> CameraIntrinsics {
    let mut out = cam.clone();
    match &mut out.model {
        CameraModel::SimplePinhole { focal_length, .. }
        | CameraModel::EquidistantFisheye { focal_length, .. }
        | CameraModel::SimpleRadialFisheye { focal_length, .. }
        | CameraModel::SfmtoolFisheye { focal_length, .. }
        | CameraModel::SfmtoolPinhole { focal_length, .. } => *focal_length = f,
        _ => {}
    }
    out
}

/// The camera at focal `f` and radial coefficient `k1` — [`cam_at`] plus the
/// one distortion parameter the bundle adjustment can release, which exists
/// only on `SIMPLE_RADIAL_FISHEYE`. `k1` is ignored for every other model
/// (`opt_k1` is gated on that one).
pub(crate) fn cam_with(cam: &CameraIntrinsics, f: f64, k1: f64) -> CameraIntrinsics {
    let mut out = cam_at(cam, f);
    if let CameraModel::SimpleRadialFisheye {
        radial_distortion_k1,
        ..
    } = &mut out.model
    {
        *radial_distortion_k1 = k1;
    }
    out
}

/// The camera at focal `f` and spline coefficients `bspline` — [`cam_at`]
/// plus the coefficient vector the bundle adjustment's spline release moves,
/// which exists on the two sfmtool spline models, `SFMTOOL_FISHEYE` and
/// `SFMTOOL_PINHOLE`. `bspline` is ignored for every other model
/// (`opt_bspline` is gated on those two). The sibling of [`cam_with`]: the two
/// never apply together, because no model carries both a `k1` and a spline.
pub(crate) fn cam_with_bspline(
    cam: &CameraIntrinsics,
    f: f64,
    bspline: &[f64],
) -> CameraIntrinsics {
    let mut out = cam_at(cam, f);
    let coeffs = match &mut out.model {
        CameraModel::SfmtoolFisheye {
            bspline: coeffs, ..
        }
        | CameraModel::SfmtoolPinhole {
            bspline: coeffs, ..
        } => coeffs,
        _ => return out,
    };
    coeffs.clear();
    coeffs.extend_from_slice(bspline);
    out
}

#[cfg(test)]
mod tests;
