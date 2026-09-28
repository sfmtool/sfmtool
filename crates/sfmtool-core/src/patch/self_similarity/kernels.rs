// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cross-sum kernels of the ZNCC self-similarity radius: for one template
//! rectangle of a centred, padded plane, the dot product of the template with
//! the window moved by every whole-pixel shift of the `(2r + 1)²` square.
//!
//! [`cross_sums`] dispatches at run time to the hand-written AVX2 kernel where
//! the CPU has AVX2 and FMA and `2r + 1 ≤ 8`, and to [`cross_sums_scalar`]
//! otherwise. The scalar form is the reference the AVX2 kernel must match, the
//! fallback on other CPUs and other architectures, and the equivalence test's
//! oracle.
//!
//! The plane is laid out row-major with row stride `stride ≥ width + 8`, the
//! padding columns zero, so an 8-lane load that starts at any column a shift
//! can read stays inside the row's storage. `P` below is the plane.

/// **Compute and overwrite** the raw cross sums of one template rectangle
/// `rect = [x, y, w, h]`: `out[(dy + r)·(2r + 1) + (dx + r)] = Σ_k P[k]·P[k + d]`
/// over the pixels `k` of the rectangle, for every `d = (dx, dy)` with `|dx|,
/// |dy| ≤ r`. The caller keeps `r` pixels of plane around the rectangle on
/// every side.
pub(super) fn cross_sums(
    plane: &[f32],
    stride: usize,
    rect: [usize; 4],
    r: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if 2 * r < 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the runtime feature check is `cross_sums_avx2`'s CPU
            // precondition, `2r + 1 ≤ 8` picks a `ROWS` that fits one register
            // per shift row, and the bounds checked in `check_bounds` below are
            // the in-bounds-load contract its SAFETY block states.
            check_bounds(plane, stride, rect, r, out);
            unsafe {
                match r {
                    0 => cross_sums_avx2::<1>(plane, stride, rect, out),
                    1 => cross_sums_avx2::<3>(plane, stride, rect, out),
                    2 => cross_sums_avx2::<5>(plane, stride, rect, out),
                    _ => cross_sums_avx2::<7>(plane, stride, rect, out),
                }
            }
            return;
        }
    }
    cross_sums_scalar(plane, stride, rect, r, out);
}

/// The bounds every load of either kernel stays inside: the rectangle keeps
/// `r` pixels of plane on every side, and each row has the 8 padding columns
/// the widest load reads past the last shift. Checked in release builds too,
/// since the AVX2 kernel's safety rests on it and the check is a few integer
/// comparisons per call.
fn check_bounds(plane: &[f32], stride: usize, rect: [usize; 4], r: usize, out: &[f32]) {
    let [x, y, w, h] = rect;
    let side = 2 * r + 1;
    assert!(w > 0 && h > 0, "cross_sums: empty template {w}×{h}");
    assert!(
        x >= r && y >= r,
        "cross_sums: the template at ({x}, {y}) has less than {r} px of plane above or left of it"
    );
    // The last column a load reads: the scalar kernel's `x + w - 1 + r`, or
    // the end of the AVX2 kernel's 8-lane load that starts at `x + w - 1 - r`.
    let last_col = (x + w - 1 + r).max(x + w - 1 - r + 7);
    assert!(
        last_col < stride && (y + h + r) * stride <= plane.len(),
        "cross_sums: the template {w}×{h} at ({x}, {y}) with r = {r} reads past the plane \
         (stride {stride}, {} values)",
        plane.len()
    );
    assert!(
        out.len() >= side * side,
        "cross_sums: output shorter than (2r + 1)²"
    );
}

/// Scalar reference for [`cross_sums`]: for each shift row `dy`, each template
/// pixel and each `dx`, accumulate `P[k]·P[k + d]` in `f32`, visiting the
/// template pixels in the same row-major order as the AVX2 kernel. It rounds
/// each product before adding where the AVX2 kernel fuses the two, so the two
/// agree to `f32` rounding, not bit for bit.
pub(super) fn cross_sums_scalar(
    plane: &[f32],
    stride: usize,
    rect: [usize; 4],
    r: usize,
    out: &mut [f32],
) {
    check_bounds(plane, stride, rect, r, out);
    let [x0, y0, w, h] = rect;
    let side = 2 * r + 1;
    out[..side * side].fill(0.0);
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let t = plane[y * stride + x];
            for j in 0..side {
                let src = &plane[(y + j - r) * stride + (x - r)..][..side];
                let row = &mut out[j * side..][..side];
                for (acc, &v) in row.iter_mut().zip(src) {
                    *acc += t * v;
                }
            }
        }
    }
}

/// Hand-written AVX2 form of [`cross_sums`], for `ROWS = 2r + 1 ≤ 8`.
///
/// Lanes run across the `2r + 1` horizontal shifts, so one 8-lane register
/// holds a whole shift row and `ROWS` registers hold the square of shifts. Per
/// template pixel it broadcasts `P[k]` and, for each shift row, loads the 8
/// plane values starting at `k + (−r, dy)` and fuses a multiply-add. The lanes
/// past `2r + 1` are computed and dropped when the registers are stored.
///
/// # Safety
///
/// 1. **CPU features:** `is_x86_feature_detected!("avx2") &&
///    is_x86_feature_detected!("fma")`.
/// 2. **Loads in bounds:** with `r = (ROWS − 1) / 2` and `rect = [x, y, w,
///    h]`, `x ≥ r`, `y ≥ r`, `x + w − 1 − r + 8 ≤ stride` and `(y + h + r) ·
///    stride ≤ plane.len()`, which is what [`check_bounds`] asserts: every
///    8-wide load starts at `(yk + dy)·stride + xk − r` for a template pixel
///    `(xk, yk)` and `|dy| ≤ r`, and ends inside the same row's storage.
/// 3. **Output covers the square:** `out.len() ≥ ROWS²`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cross_sums_avx2<const ROWS: usize>(
    plane: &[f32],
    stride: usize,
    rect: [usize; 4],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let r = (ROWS - 1) / 2;
    let [x0, y0, w, h] = rect;
    debug_assert!(out.len() >= ROWS * ROWS);
    let ptr = plane.as_ptr();
    let mut acc = [_mm256_setzero_ps(); ROWS];
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let t = _mm256_set1_ps(*ptr.add(y * stride + x));
            let base = (y - r) * stride + (x - r);
            for (j, a) in acc.iter_mut().enumerate() {
                let v = _mm256_loadu_ps(ptr.add(base + j * stride));
                *a = _mm256_fmadd_ps(t, v, *a);
            }
        }
    }
    let mut lanes = [0.0f32; 8];
    for (j, a) in acc.iter().enumerate() {
        _mm256_storeu_ps(lanes.as_mut_ptr(), *a);
        out[j * ROWS..][..ROWS].copy_from_slice(&lanes[..ROWS]);
    }
}
