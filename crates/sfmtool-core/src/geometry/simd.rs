// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Runtime SIMD dispatch for the focal-vote kernel's residual loops.
//!
//! The kernels guarded by [`avx2_enabled`] are **lane-per-point**: lane `j`
//! performs exactly the IEEE-754 operations the scalar loop performed for point
//! `j`, in the same order, so every SIMD path here is bit-identical to its
//! scalar fallback and the dispatch is a pure performance switch. That is the
//! whole discipline, and it rules out three things everywhere in this crate's
//! vectorized residual code:
//!
//! - **No vectorized horizontal reductions.** A sum whose order changes is a
//!   different `f64`. The per-point dot products of fixed length 3 keep their
//!   scalar order *inside* each lane (`(a·b + c·d) + e·f`, matching nalgebra's
//!   unrolled `U3` dot), which is why they are safe.
//! - **No libm in one arm and an approximation in the other.** A transcendental
//!   is vectorized only as a polynomial that has a scalar twin performing the
//!   same operations in the same order, so both arms evaluate the same
//!   arithmetic: [`acos_pd`] and
//!   [`crate::geometry::acos_poly::acos_poly_scalar`] read one shared coefficient
//!   table and each ragged tail takes the scalar twin, never `f64::acos`.
//! - **No skipped lanes.** Where the scalar code takes a guard branch to a
//!   sentinel (`f64::INFINITY` for a degenerate homography transfer), the SIMD
//!   path reproduces it with a compare and a blend.
//!
//! One asymmetry is easy to get wrong and worth stating once, because both
//! kernels depend on it: `_mm256_min_pd(a, b)` returns `b` whenever the compare
//! `a < b` is unordered, and `_mm256_max_pd(a, b)` returns `b` whenever `a > b`
//! is unordered. So `x.min(c)` (Rust's NaN-quieting [`f64::min`], which returns
//! `c` for a NaN `x`) is `_mm256_min_pd(x, c)` with the *value* first, while
//! [`f64::clamp`]'s `if x > max { max }` (which propagates a NaN `x`) is
//! `_mm256_min_pd(c, x)` with the *constant* first. Same intrinsic, opposite
//! operand order, opposite NaN behaviour.

/// Force the scalar residual paths in the focal-vote kernel (for A/B timing and
/// parity checks).
///
/// The SIMD kernels are bit-identical to the scalar loops, so toggling this only
/// affects performance, never output — which is exactly what makes it usable as
/// a parity harness: run any capture both ways and hex-compare the result.
#[cfg(target_arch = "x86_64")]
static FOCAL_VOTE_NO_SIMD: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_FOCAL_VOTE_NO_SIMD").is_some());

/// Whether the focal-vote residual loops may take their AVX2 path: the CPU has
/// AVX2 and `SFMTOOL_FOCAL_VOTE_NO_SIMD` is unset.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx2_enabled() -> bool {
    !*FOCAL_VOTE_NO_SIMD && std::is_x86_feature_detected!("avx2")
}

/// Whether the focal vote's epipolar cell computes its residuals in `f32`
/// — the default, with `SFMTOOL_FOCAL_VOTE_F64_EPI` as the diagnostic
/// restore of the double-precision path (the forensics convention of
/// `SFMTOOL_FOCAL_VOTE_LIBM_ACOS` and `SFMTOOL_FOCAL_VOTE_EIGEN_MINSOLVE`).
///
/// Unlike [`avx2_enabled`], this is **not** a pure performance switch: the
/// residual is a different arithmetic and the rays it measures are narrowed to
/// `f32` (widened back for the elimination rows, so a cell has one ray truth
/// rather than two). Positions are stored `f32`, keypoint noise is three
/// orders above `f32` resolution, and the fleet measured every verdict,
/// confirmation and escalation invariant with consensus focals within 7e-6 —
/// while the kernel runs 2.7-3.2x and the escalated vote 12-19% faster.
static F32_EPI: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_FOCAL_VOTE_F64_EPI").is_none());

/// Whether the focal vote's rotation cell computes its residuals in `f32`
/// through the cross-product-norm form (`SFMTOOL_FOCAL_VOTE_F32_ROT`). The
/// sibling of [`F32_EPI`], and equally not a pure performance switch: the
/// angle comes from `asin|r₁ × r₂|` folded by the sign of the dot, where the
/// `f64` path takes `acos` of the dot.
static F32_ROT: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_FOCAL_VOTE_F32_ROT").is_some());

/// See [`F32_EPI`].
#[inline]
pub(crate) fn f32_epipolar_enabled() -> bool {
    *F32_EPI
}

/// See [`F32_ROT`].
#[inline]
pub(crate) fn f32_rotation_enabled() -> bool {
    *F32_ROT
}

/// `Vector3<f64>` is three `f64` with no padding, which is what lets the ray
/// kernels reinterpret a `&[Vector3<f64>]` as a flat `*const f64` and load four
/// consecutive rays as twelve contiguous doubles.
#[cfg(target_arch = "x86_64")]
const _: () = assert!(std::mem::size_of::<nalgebra::Vector3<f64>>() == 3 * 8);

/// Load four consecutive 3-vectors and transpose them into `(x, y, z)` lanes.
///
/// `p` addresses twelve readable, consecutive `f64`
/// (`[x0 y0 z0 x1 | y1 z1 x2 y2 | z2 x3 y3 z3]`), and the result is
/// `([x0 x1 x2 x3], [y0 y1 y2 y3], [z0 z1 z2 z3])`.
///
/// The `x` and `z` lanes fall out of one cross-lane `permute2f128` plus blends
/// because their elements already sit at the right positions in the three raw
/// loads; only `y` needs per-vector `permute4x64`s. Four shuffle-port µops per
/// four points is the cost of leaving the rays in their `Vec<Vector3<f64>>`
/// layout, which the epipolar and rotation cells, `kabsch` and
/// [`crate::geometry::relative_pose`] all share.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller) and twelve
/// readable `f64` at `p`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn load_vec3x4(
    p: *const f64,
) -> (
    std::arch::x86_64::__m256d,
    std::arch::x86_64::__m256d,
    std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::*;

    let a = _mm256_loadu_pd(p); // x0 y0 z0 x1
    let b = _mm256_loadu_pd(p.add(4)); // y1 z1 x2 y2
    let c = _mm256_loadu_pd(p.add(8)); // z2 x3 y3 z3

    // [a2 a3 c0 c1] = [z0 x1 z2 x3] — carries x1/x3 and z0/z2 at their final
    // lane positions, so x and z need only blends from here.
    let t = _mm256_permute2f128_pd(a, c, 0x21);

    let x = _mm256_blend_pd(_mm256_blend_pd(a, t, 0b1010), b, 0b0100);
    let z = _mm256_blend_pd(_mm256_blend_pd(t, b, 0b0010), c, 0b1000);

    let ya = _mm256_permute4x64_pd(a, 0b11_10_01_01); // lane0 ← a1 = y0
    let yb = _mm256_permute4x64_pd(b, 0b11_11_00_00); // lane1 ← b0, lane2 ← b3
    let yc = _mm256_permute4x64_pd(c, 0b10_10_10_10); // lane3 ← c2 = y3
    let y = _mm256_blend_pd(_mm256_blend_pd(ya, yb, 0b0110), yc, 0b1000);

    (x, y, z)
}

/// Load four consecutive `[f64; 2]` points and transpose them into `(x, y)`
/// lanes.
///
/// `p` addresses eight readable, consecutive `f64` (`[x0 y0 x1 y1 x2 y2 x3 y3]`)
/// and the result is `([x0 x1 x2 x3], [y0 y1 y2 y3])`.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller) and eight
/// readable `f64` at `p`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn load_vec2x4(
    p: *const f64,
) -> (std::arch::x86_64::__m256d, std::arch::x86_64::__m256d) {
    use std::arch::x86_64::*;

    let a = _mm256_loadu_pd(p); // x0 y0 x1 y1
    let b = _mm256_loadu_pd(p.add(4)); // x2 y2 x3 y3
    let lo = _mm256_permute2f128_pd(a, b, 0x20); // x0 y0 x2 y2
    let hi = _mm256_permute2f128_pd(a, b, 0x31); // x1 y1 x3 y3
    (
        _mm256_unpacklo_pd(lo, hi), // x0 x1 x2 x3
        _mm256_unpackhi_pd(lo, hi), // y0 y1 y2 y3
    )
}

/// One row of a `3×3` matvec, four points at a time: `(a₀·x + a₁·y) + a₂·z`.
///
/// That is nalgebra's `gemv` order for a `Matrix3 * Vector3` — it accumulates
/// column by column, `y ← col₀·v₀`, then `y ← colⱼ·vⱼ + y` — so the lane
/// arithmetic matches the scalar product term for term.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn row3(
    a: &[std::arch::x86_64::__m256d; 3],
    x: std::arch::x86_64::__m256d,
    y: std::arch::x86_64::__m256d,
    z: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;
    _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(a[0], x), _mm256_mul_pd(a[1], y)),
        _mm256_mul_pd(a[2], z),
    )
}

/// Length-3 dot product, four points at a time:
/// `(ax·bx + ay·by) + az·bz`.
///
/// That is nalgebra's unrolled `U3` special case of `dot` (`a + b + c`, left to
/// right, with no zero-initialized accumulator), which is what makes this a
/// lane-local reduction rather than a horizontal one.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dot3(
    ax: std::arch::x86_64::__m256d,
    ay: std::arch::x86_64::__m256d,
    az: std::arch::x86_64::__m256d,
    bx: std::arch::x86_64::__m256d,
    by: std::arch::x86_64::__m256d,
    bz: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;
    _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(ax, bx), _mm256_mul_pd(ay, by)),
        _mm256_mul_pd(az, bz),
    )
}

/// Broadcast every entry of a `3×3` matrix, row-major, for [`row3`].
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn broadcast_mat3(
    m: &nalgebra::Matrix3<f64>,
) -> [[std::arch::x86_64::__m256d; 3]; 3] {
    use std::arch::x86_64::*;
    [
        [
            _mm256_set1_pd(m[(0, 0)]),
            _mm256_set1_pd(m[(0, 1)]),
            _mm256_set1_pd(m[(0, 2)]),
        ],
        [
            _mm256_set1_pd(m[(1, 0)]),
            _mm256_set1_pd(m[(1, 1)]),
            _mm256_set1_pd(m[(1, 2)]),
        ],
        [
            _mm256_set1_pd(m[(2, 0)]),
            _mm256_set1_pd(m[(2, 1)]),
            _mm256_set1_pd(m[(2, 2)]),
        ],
    ]
}

/// `acos` over `[-1, 1]`, four lanes at a time — the vector twin of
/// [`crate::geometry::acos_poly::acos_poly_scalar`].
///
/// Same coefficient table ([`crate::geometry::acos_poly::ACOS_POLY`]), same
/// operations in the same order, with the scalar form's branches as compares
/// and blends: `copysign(r, d)` is `or(r, and(signmask, d))`, and the big/small
/// and sign selections are `blendv`. Every lane therefore yields the bits the
/// scalar twin yields for that argument, which is what lets the rotation
/// residual dispatch — vector body, ragged tail, scalar fallback — stay one
/// arithmetic.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn acos_pd(d: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    use crate::geometry::acos_poly::ACOS_POLY as A;
    use std::arch::x86_64::*;

    let signmask = _mm256_set1_pd(-0.0);
    let half = _mm256_set1_pd(0.5);
    let one = _mm256_set1_pd(1.0);
    let pi = _mm256_set1_pd(std::f64::consts::PI);
    let pi_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);

    let a = _mm256_andnot_pd(signmask, d);
    let big = _mm256_cmp_pd(a, half, _CMP_GT_OQ);
    let z = _mm256_blendv_pd(
        _mm256_mul_pd(a, a),
        _mm256_mul_pd(_mm256_sub_pd(one, a), half),
        big,
    );
    let s = _mm256_blendv_pd(a, _mm256_sqrt_pd(z), big);

    let mut p = _mm256_set1_pd(A[13]);
    let mut k = 13usize;
    while k > 0 {
        k -= 1;
        p = _mm256_add_pd(_mm256_mul_pd(p, z), _mm256_set1_pd(A[k]));
    }
    let r = _mm256_add_pd(s, _mm256_mul_pd(_mm256_mul_pd(s, z), p));

    let r_signed = _mm256_or_pd(r, _mm256_and_pd(signmask, d));
    let res_small = _mm256_sub_pd(pi_2, r_signed);
    let two_r = _mm256_add_pd(r, r);
    let neg = _mm256_cmp_pd(d, _mm256_setzero_pd(), _CMP_LT_OQ);
    let res_big = _mm256_blendv_pd(two_r, _mm256_sub_pd(pi, two_r), neg);
    _mm256_blendv_pd(res_small, res_big, big)
}

// ── Single-precision kernels (the f32 residual arms) ─────────────────────────
//
// These read rays held **structure-of-arrays** (`x`, `y`, `z` in three
// separate `Vec<f32>`), so a lane load is one `loadu_ps` and there is no
// transpose at all — the `f64` kernels pay four shuffle µops per four points
// to keep their rays in `Vec<Vector3<f64>>`, which several other callers
// share. Nothing here has a bit-identical relationship to the `f64` forms;
// each has a scalar twin performing the same `f32` operations in the same
// order, and that is the only identity claimed.

/// One row of a `3×3` matvec, eight points at a time: `(a₀·x + a₁·y) + a₂·z`.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn row3_ps(
    a: &[std::arch::x86_64::__m256; 3],
    x: std::arch::x86_64::__m256,
    y: std::arch::x86_64::__m256,
    z: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_add_ps(
        _mm256_add_ps(_mm256_mul_ps(a[0], x), _mm256_mul_ps(a[1], y)),
        _mm256_mul_ps(a[2], z),
    )
}

/// Length-3 dot product, eight points at a time: `(ax·bx + ay·by) + az·bz`.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dot3_ps(
    ax: std::arch::x86_64::__m256,
    ay: std::arch::x86_64::__m256,
    az: std::arch::x86_64::__m256,
    bx: std::arch::x86_64::__m256,
    by: std::arch::x86_64::__m256,
    bz: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_add_ps(
        _mm256_add_ps(_mm256_mul_ps(ax, bx), _mm256_mul_ps(ay, by)),
        _mm256_mul_ps(az, bz),
    )
}

/// Broadcast every entry of an `f64` `3×3` matrix, narrowed to `f32`,
/// row-major, for [`row3_ps`].
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn broadcast_mat3_ps(
    m: &nalgebra::Matrix3<f64>,
) -> [[std::arch::x86_64::__m256; 3]; 3] {
    use std::arch::x86_64::*;
    let b = |r: usize, c: usize| _mm256_set1_ps(m[(r, c)] as f32);
    [
        [b(0, 0), b(0, 1), b(0, 2)],
        [b(1, 0), b(1, 1), b(1, 2)],
        [b(2, 0), b(2, 1), b(2, 2)],
    ]
}

/// `asin` over `[0, 1]`, eight lanes at a time — the vector twin of
/// [`crate::geometry::acos_poly::asin_poly_scalar_f32`].
///
/// Same coefficient table ([`crate::geometry::acos_poly::ACOS_POLY_F32`], the
/// narrowed [`crate::geometry::acos_poly::ACOS_POLY`]), same operations in the
/// same order, with the scalar form's branch as a compare and two blends.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn asin_ps(s: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use crate::geometry::acos_poly::ACOS_POLY_F32 as A;
    use std::arch::x86_64::*;

    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let pi_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    let big = _mm256_cmp_ps(s, half, _CMP_GT_OQ);
    let z = _mm256_blendv_ps(
        _mm256_mul_ps(s, s),
        _mm256_mul_ps(_mm256_sub_ps(one, s), half),
        big,
    );
    let base = _mm256_blendv_ps(s, _mm256_sqrt_ps(z), big);

    let mut p = _mm256_set1_ps(A[13]);
    let mut k = 13usize;
    while k > 0 {
        k -= 1;
        p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(A[k]));
    }
    let r = _mm256_add_ps(base, _mm256_mul_ps(_mm256_mul_ps(base, z), p));
    let two_r = _mm256_add_ps(r, r);
    _mm256_blendv_ps(r, _mm256_sub_ps(pi_2, two_r), big)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
