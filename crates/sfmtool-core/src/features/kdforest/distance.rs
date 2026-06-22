// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Metric abstraction for the randomized kd-tree forest.
//!
//! The forest is generic over a scalar type `S: ForestScalar`. The trait keeps
//! squared-distance arithmetic in the scalar's natural domain — integer (`i64`)
//! for `u8` descriptors (matching [`crate::feature_match::descriptor`]) and
//! `f32` for general vectors — so the priority-queue lower bounds, the
//! bounded-result cutoff, and the leaf scan all compose without lossy
//! conversions. `sqrt` is taken only when a distance is reported to a caller.

use std::cmp::Ordering;

/// A scalar coordinate type the forest can index over.
///
/// Implementors define how to accumulate squared distances. The associated
/// [`Dist`](ForestScalar::Dist) type is the squared-distance domain; it must be
/// totally ordered (so it can key the search priority queue and the bounded
/// result set) and additive (so the best-bin-first lower bound can accumulate a
/// per-axis plane distance).
pub trait ForestScalar: Copy + Send + Sync + PartialEq + 'static {
    /// Squared-distance accumulator. Totally ordered and additive.
    type Dist: Copy + Ord + Send + Sync + std::ops::Add<Output = Self::Dist> + std::fmt::Debug;

    /// The additive identity of the distance domain (a zero lower bound).
    const ZERO_DIST: Self::Dist;
    /// The maximum representable distance (an unbounded result cutoff).
    const MAX_DIST: Self::Dist;

    /// Squared distance contributed by a single axis. Used for the
    /// split-plane lower bound `lb + (q - split_val)²`.
    fn axis_dist_sq(a: Self, b: Self) -> Self::Dist;

    /// Full squared distance between two equal-length points.
    fn dist_sq(a: &[Self], b: &[Self]) -> Self::Dist;

    /// Convert a squared distance to `f32` for reporting to callers.
    fn dist_sq_to_f32(d: Self::Dist) -> f32;

    /// Square a Euclidean cutoff into the squared-distance domain.
    ///
    /// Rounds *up* for integer domains so the cutoff never prunes a candidate
    /// that is genuinely within `max_dist`.
    fn cutoff_sq(max_dist: f32) -> Self::Dist;

    /// Total order over raw coordinates (for median selection / partitioning).
    fn coord_cmp(a: Self, b: Self) -> Ordering;

    /// Widen a coordinate to `f64` for variance estimation.
    fn to_f64(self) -> f64;
}

// ── u8 descriptors (integer squared-L2) ─────────────────────────────────────

impl ForestScalar for u8 {
    type Dist = i64;
    const ZERO_DIST: i64 = 0;
    const MAX_DIST: i64 = i64::MAX;

    #[inline]
    fn axis_dist_sq(a: u8, b: u8) -> i64 {
        let d = a as i64 - b as i64;
        d * d
    }

    #[inline]
    fn dist_sq(a: &[u8], b: &[u8]) -> i64 {
        u8_dist_sq(a, b)
    }

    #[inline]
    fn dist_sq_to_f32(d: i64) -> f32 {
        d as f32
    }

    #[inline]
    fn cutoff_sq(max_dist: f32) -> i64 {
        let sq = (max_dist as f64) * (max_dist as f64);
        if sq >= i64::MAX as f64 {
            i64::MAX
        } else {
            sq.ceil() as i64
        }
    }

    #[inline]
    fn coord_cmp(a: u8, b: u8) -> Ordering {
        a.cmp(&b)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// ── f32 vectors (floating squared-L2) ───────────────────────────────────────

/// Totally ordered wrapper around `f32` for the squared-distance domain.
///
/// Squared distances and lower bounds are always non-negative and finite (or
/// `+∞` for the unbounded cutoff), so [`f32::total_cmp`] is a genuine total
/// order here and `Add` is plain floating addition. `PartialEq` is defined
/// through `total_cmp` (not derived from `f32`'s `==`) so that `a == b` agrees
/// with `cmp(a, b) == Equal`, upholding the `Eq`/`Ord` contract.
#[derive(Clone, Copy, Debug)]
pub struct OrdF32(pub f32);

impl PartialEq for OrdF32 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl std::ops::Add for OrdF32 {
    type Output = OrdF32;
    fn add(self, rhs: OrdF32) -> OrdF32 {
        OrdF32(self.0 + rhs.0)
    }
}

impl ForestScalar for f32 {
    type Dist = OrdF32;
    const ZERO_DIST: OrdF32 = OrdF32(0.0);
    const MAX_DIST: OrdF32 = OrdF32(f32::INFINITY);

    #[inline]
    fn axis_dist_sq(a: f32, b: f32) -> OrdF32 {
        let d = a - b;
        OrdF32(d * d)
    }

    #[inline]
    fn dist_sq(a: &[f32], b: &[f32]) -> OrdF32 {
        let mut acc = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = x - y;
            acc += d * d;
        }
        OrdF32(acc)
    }

    #[inline]
    fn dist_sq_to_f32(d: OrdF32) -> f32 {
        d.0
    }

    #[inline]
    fn cutoff_sq(max_dist: f32) -> OrdF32 {
        OrdF32(max_dist * max_dist)
    }

    #[inline]
    fn coord_cmp(a: f32, b: f32) -> Ordering {
        a.total_cmp(&b)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// ── u8 squared-L2 kernel (SSE2 + scalar fallback) ───────────────────────────

/// Force the scalar `u8` distance path (for A/B timing and parity tests).
///
/// The SSE2 kernel produces bit-identical integer results to the scalar loop,
/// so toggling this only affects performance, never output.
#[cfg(target_arch = "x86_64")]
static KDFOREST_NO_SIMD: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_KDFOREST_NO_SIMD").is_some());

/// Squared L2 distance between two equal-length `u8` vectors.
///
/// This is the forest's hottest loop (the leaf scan). Dispatches at runtime to
/// an AVX2 then SSE2 sum-of-squared-differences kernel on x86_64, falling back
/// to scalar elsewhere / for the tail. All paths produce the identical integer
/// result as [`crate::feature_match::descriptor::descriptor_distance_l2_squared`]
/// (integer accumulation is exact regardless of reduction order).
#[inline]
pub fn u8_dist_sq(a: &[u8], b: &[u8]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if !*KDFOREST_NO_SIMD {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: avx2 confirmed available just above.
                return unsafe { u8_dist_sq_avx2(a, b) };
            }
            if std::is_x86_feature_detected!("sse2") {
                // SAFETY: sse2 confirmed available just above.
                return unsafe { u8_dist_sq_sse2(a, b) };
            }
        }
    }
    u8_dist_sq_scalar(a, b)
}

/// AVX2 sum of squared `u8` differences, 32 bytes per iteration.
///
/// Same scheme as [`u8_dist_sq_sse2`] one lane width up: `|a − b|` via saturating
/// `max−min`, widened to 16-bit lanes (per 128-bit half) and squared+summed with
/// `madd_epi16` into eight 32-bit accumulator lanes, reduced to `i64` at the end.
/// For 128-byte SIFT this is 4 iterations with no tail. Spreading the sum over
/// eight 32-bit lanes leaves even more overflow headroom than the SSE2 path.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn u8_dist_sq_avx2(a: &[u8], b: &[u8]) -> i64 {
    use std::arch::x86_64::*;

    let n = a.len();
    let blocks = n / 32;
    let mut acc = _mm256_setzero_si256();
    let zero = _mm256_setzero_si256();

    let pa = a.as_ptr();
    let pb = b.as_ptr();
    for i in 0..blocks {
        let va = _mm256_loadu_si256(pa.add(i * 32) as *const __m256i);
        let vb = _mm256_loadu_si256(pb.add(i * 32) as *const __m256i);
        let diff = _mm256_subs_epu8(_mm256_max_epu8(va, vb), _mm256_min_epu8(va, vb));
        // unpack works per-128-bit lane; both halves are covered and summed
        // together at the end, so the lane grouping does not matter.
        let lo = _mm256_unpacklo_epi8(diff, zero);
        let hi = _mm256_unpackhi_epi8(diff, zero);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo, lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi, hi));
    }

    // Horizontal sum of the eight 32-bit lanes.
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut total = lanes.iter().map(|&l| l as i64).sum::<i64>();

    // Scalar tail past the last full 32-byte block.
    let rem = blocks * 32;
    total += u8_dist_sq_scalar(&a[rem..], &b[rem..]);
    total
}

/// Scalar sum of squared `u8` differences.
///
/// Delegates to [`crate::feature_match::descriptor::descriptor_distance_l2_squared`]
/// so the crate keeps a single definition of the scalar `u8` squared-L2 metric.
/// The SIMD paths above must stay bit-identical to it; `u8_kernel_matches_scalar`
/// checks the path selected by feature detection on the test host.
#[inline]
fn u8_dist_sq_scalar(a: &[u8], b: &[u8]) -> i64 {
    crate::feature_match::descriptor::descriptor_distance_l2_squared(a, b)
}

/// SSE2 sum of squared `u8` differences, 16 bytes per iteration.
///
/// `|a − b|` is computed via saturating `max−min` (exact for `u8`), widened to
/// 16-bit lanes, then `madd_epi16(d, d)` accumulates pairwise squares into
/// 32-bit lanes, reduced to `i64` at the end. A lane would only overflow `i32`
/// at descriptor widths in the hundreds of thousands, far past any in use. The
/// remainder past the last full 16-byte block is handled by the scalar fallback.
///
/// # Safety
/// Requires the `sse2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn u8_dist_sq_sse2(a: &[u8], b: &[u8]) -> i64 {
    use std::arch::x86_64::*;

    let n = a.len();
    let blocks = n / 16;
    let mut acc = _mm_setzero_si128();
    let zero = _mm_setzero_si128();

    let pa = a.as_ptr();
    let pb = b.as_ptr();
    for i in 0..blocks {
        let va = _mm_loadu_si128(pa.add(i * 16) as *const __m128i);
        let vb = _mm_loadu_si128(pb.add(i * 16) as *const __m128i);
        // |a - b| as u8 (saturating subtraction both ways; max - min is exact).
        let diff = _mm_subs_epu8(_mm_max_epu8(va, vb), _mm_min_epu8(va, vb));
        // Widen to 16-bit lanes and square via pairwise multiply-add.
        let lo = _mm_unpacklo_epi8(diff, zero);
        let hi = _mm_unpackhi_epi8(diff, zero);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(lo, lo));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(hi, hi));
    }

    // Horizontal sum of the four 32-bit lanes.
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
    let mut total = lanes.iter().map(|&l| l as i64).sum::<i64>();

    // Scalar tail.
    let rem = blocks * 16;
    total += u8_dist_sq_scalar(&a[rem..], &b[rem..]);
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_kernel_matches_scalar() {
        // Exercise several lengths, including non-multiples of 16 and > 16.
        for &len in &[0usize, 1, 7, 15, 16, 17, 31, 128, 130] {
            let a: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let b: Vec<u8> = (0..len).map(|i| (i * 13 + 200) as u8).collect();
            assert_eq!(
                u8_dist_sq(&a, &b),
                u8_dist_sq_scalar(&a, &b),
                "mismatch at len {len}"
            );
        }
    }

    /// `u8_dist_sq` only exercises the *auto-selected* path (AVX2 on most CI
    /// hosts), so test each SIMD kernel directly against the scalar reference —
    /// otherwise a bug in the SSE2 path would never be compared on an AVX2 box.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_kernels_match_scalar() {
        for &len in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 128, 130] {
            let a: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let b: Vec<u8> = (0..len).map(|i| (i * 13 + 200) as u8).collect();
            let scalar = u8_dist_sq_scalar(&a, &b);
            if std::is_x86_feature_detected!("sse2") {
                // SAFETY: sse2 confirmed available.
                assert_eq!(unsafe { u8_dist_sq_sse2(&a, &b) }, scalar, "sse2 len {len}");
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: avx2 confirmed available.
                assert_eq!(unsafe { u8_dist_sq_avx2(&a, &b) }, scalar, "avx2 len {len}");
            }
        }
    }

    #[test]
    fn u8_kernel_extremes() {
        let a = [0u8; 128];
        let b = [255u8; 128];
        assert_eq!(u8_dist_sq(&a, &b), 255i64 * 255 * 128);
        assert_eq!(u8_dist_sq(&a, &a), 0);
    }

    #[test]
    fn f32_cutoff_and_order() {
        assert_eq!(f32::cutoff_sq(2.0), OrdF32(4.0));
        assert!(OrdF32(1.0) < OrdF32(2.0));
        assert!(OrdF32(0.0) < OrdF32(f32::INFINITY));
        assert_eq!(OrdF32(1.0) + OrdF32(2.0), OrdF32(3.0));
    }

    #[test]
    fn u8_cutoff_rounds_up() {
        // sqrt(5) ≈ 2.236; squared back must not drop below 5.
        let c = u8::cutoff_sq(5.0f32.sqrt());
        assert!(c >= 5, "cutoff {c} under-rounded");
    }
}
