// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared x86_64 AVX2 SIMD helpers for the gradient-sampling hot path that the
//! orientation and descriptor stages share (`pixel_gradient` + the Gaussian
//! weight). Both stages sample a per-keypoint window and, for every pixel,
//! compute a gradient magnitude/angle (`atan2`) and a Gaussian weight (`exp`).
//! These are the expensive per-sample transcendentals; vectorizing them 8-wide
//! (with the histogram scatter left scalar) is the point of this module.
//!
//! The approximations are well within the angular/score resolution the
//! consumers need (36-bin orientation histogram = 10°/bin; 8-bin descriptor =
//! 45°/bin), and are validated against `libm` in the unit tests.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Cached `avx2 && fma` runtime support (shared with the blur path). The
/// `SFMTOOL_SIFT_NO_AVX2` env var forces the scalar fallbacks everywhere, for A-B
/// timing and reproducibility.
#[cfg(target_arch = "x86_64")]
pub(crate) static HAS_AVX2_FMA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var_os("SFMTOOL_SIFT_NO_AVX2").is_none()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
});

/// Vectorized `atan2(y, x)` returning the angle in `[-π, π]`, lane-wise. Max
/// absolute error ≈ 1.3e-3 rad (the polynomial is a minimax fit of `atan` on
/// `[0, 1]`; quadrant/sign reconstruction is exact). `atan2(0, 0)` yields `0`.
///
/// # Safety
/// Requires the `avx2` + `fma` target features (guarded by [`HAS_AVX2_FMA`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn atan2_approx(y: __m256, x: __m256) -> __m256 {
    let sign = _mm256_set1_ps(-0.0); // 0x8000_0000 mask
    let zero = _mm256_setzero_ps();
    let ax = _mm256_andnot_ps(sign, x); // |x|
    let ay = _mm256_andnot_ps(sign, y); // |y|
    let mn = _mm256_min_ps(ax, ay);
    let mx = _mm256_max_ps(ax, ay);
    // a = mn / mx, defined as 0 where mx == 0 (both components zero).
    let a = _mm256_div_ps(mn, mx);
    let mx_zero = _mm256_cmp_ps(mx, zero, _CMP_EQ_OQ);
    let a = _mm256_andnot_ps(mx_zero, a);
    let s = _mm256_mul_ps(a, a);
    // atan(a) ≈ ((c3·s + c2)·s + c1)·s·a + a  (minimax on [0, 1]).
    let mut p = _mm256_fmadd_ps(
        _mm256_set1_ps(-0.046_496_473),
        s,
        _mm256_set1_ps(0.159_314_22),
    );
    p = _mm256_fmadd_ps(p, s, _mm256_set1_ps(-0.327_622_77));
    p = _mm256_mul_ps(p, s);
    let mut r = _mm256_fmadd_ps(p, a, a);
    // If |y| > |x| the ratio was for the complementary angle: r = π/2 − r.
    let swap = _mm256_cmp_ps(ay, ax, _CMP_GT_OQ);
    r = _mm256_blendv_ps(
        r,
        _mm256_sub_ps(_mm256_set1_ps(std::f32::consts::FRAC_PI_2), r),
        swap,
    );
    // x < 0 → second/third quadrant: r = π − r.
    let xneg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    r = _mm256_blendv_ps(
        r,
        _mm256_sub_ps(_mm256_set1_ps(std::f32::consts::PI), r),
        xneg,
    );
    // y < 0 → lower half: negate (flip the sign bit).
    let yneg = _mm256_cmp_ps(y, zero, _CMP_LT_OQ);
    _mm256_blendv_ps(r, _mm256_xor_ps(r, sign), yneg)
}

/// Vectorized `exp(x)`, lane-wise, via base-2 range reduction and a degree-5
/// minimax polynomial on the reduced range. Relative error < 1e-5 (≈1e-6 near
/// 0, growing toward the reduction-range edges) over the inputs this module uses
/// (`x ∈ [-~5, 0]`, the Gaussian window weights). Inputs are clamped below at
/// `-87` to stay in the normal `f32` range.
///
/// # Safety
/// Requires the `avx2` + `fma` target features (guarded by [`HAS_AVX2_FMA`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp_approx(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    // n = round(x · log2(e)); r = x − n·ln2 ∈ [−ln2/2, ln2/2].
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(std::f32::consts::LN_2), x);
    // exp(r) ≈ Σ r^k/k!, k=0..5 (Horner).
    let mut p = _mm256_set1_ps(1.0 / 120.0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    // Scale by 2^n by constructing the exponent field directly.
    let ni = _mm256_cvtps_epi32(n);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ni, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(p, pow2n)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
