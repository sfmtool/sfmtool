// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-shift and per-cell windowed-ZNCC correlation kernels for the translation
//! search, split out of the search orchestration ([`super::search`]).
//!
//! Two shapes of the same algebra, each with a scalar reference and a hand-rolled
//! AVX2 kernel (runtime-dispatched):
//!
//! - [`compute_channel_grids`] / [`accumulate_count`] score the **whole**
//!   `span × span` shift grid at once (a fused SAXPY across each grid row) — the
//!   [`SearchStrategy::Exhaustive`](super::SearchStrategy::Exhaustive) path.
//! - [`score_cell_one_channel`] / [`count_invalid_at_cell`] score **one** shift
//!   cell (a direct loop / `vgatherdps` over support pixels) — the fundamental
//!   primitive for the
//!   [`SearchStrategy::PlusDescent`](super::SearchStrategy::PlusDescent) walk.
//!
//! `I` is always the channel's **centered** planar plane
//! (`I = I_raw − cache_mean`, in `ContextTile::planes`'s `istride`-stride layout).

use crate::patch::normal_refine::Support;

/// **Compute and overwrite** one channel's per-shift correlation grids over the
/// `span × span` shift window: numerator `g_n[s] = Σ_k kern[k]·I[s+k]` and the
/// centered window moments `g_s1[s] = Σ_k w[k]·I[s+k]`,
/// `g_s2[s] = Σ_k w[k]·I[s+k]²`. `I` is the channel's **centered** planar plane
/// (`I = I_raw − cache_mean`, stored in the `istride`-row-stride
/// `ContextTile::planes`). `(win_oy, win_ox)` is the cache index of the search
/// grid's `(gy, gx) = (0, 0)` cell (the window's top-left, at the view's integer
/// base offset minus `margin`). On return, `g_n / g_s1 / g_s2` hold the per-shift
/// grids — they are overwritten, not accumulated; the caller does **not** need to
/// pre-zero them.
///
/// Runtime-dispatched to a hand-rolled AVX2 kernel where available (mirrors
/// [`crate::patch::normal_refine::fronto_cache::resample_support_avx2`]); the
/// scalar form is the reference, the non-x86 / non-AVX2 fallback, and the path for
/// spans larger than the AVX2 kernel's 16-lane row.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_channel_grids(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g_n: &mut [f32],
    g_s1: &mut [f32],
    g_s2: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // The AVX2 kernel handles spans up to 16 (one pair of `__m256` lanes per
        // gy row). The default `span = 13` fits; larger spans fall back to scalar.
        if span <= 16 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: see `compute_channel_grids_avx2`'s SAFETY block — the runtime
            // feature check, `span ≤ 16`, and the cache's `istride` padding plus the
            // `win_o*`/`r_k`/`c_k` invariants ensure every 16-wide load stays in
            // bounds of `plane[..istride * cache_rows]`.
            unsafe {
                return compute_channel_grids_avx2(
                    plane, support, kern, w, resolution, istride, span, win_oy, win_ox, g_n, g_s1,
                    g_s2,
                );
            }
        }
    }
    compute_channel_grids_scalar(
        plane, support, kern, w, resolution, istride, span, win_oy, win_ox, g_n, g_s1, g_s2,
    );
}

/// Scalar reference for [`compute_channel_grids`]: the algebra the AVX2 kernel
/// must match, the non-x86 / non-AVX2 fallback, and the equivalence test's oracle.
/// Internally uses a k-outer in-memory accumulator over the `(N, S1, S2)` grids;
/// the function owns its outputs (zeroes them before accumulating) so the external
/// contract is "overwrite", matching the AVX2 path.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_channel_grids_scalar(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g_n: &mut [f32],
    g_s1: &mut [f32],
    g_s2: &mut [f32],
) {
    let gsz = span * span;
    g_n[..gsz].fill(0.0);
    g_s1[..gsz].fill(0.0);
    g_s2[..gsz].fill(0.0);
    for (k, &p) in support.pixels.iter().enumerate() {
        let r_k = p / resolution;
        let c_k = p % resolution;
        let w_k = w[k];
        let kern_k = kern[k];
        for gy in 0..span {
            let src = &plane[(gy + win_oy + r_k) * istride + (win_ox + c_k)..][..span];
            let gbase = gy * span;
            let on = &mut g_n[gbase..][..span];
            let o1 = &mut g_s1[gbase..][..span];
            let o2 = &mut g_s2[gbase..][..span];
            for gx in 0..span {
                let v = src[gx];
                on[gx] += kern_k * v;
                o1[gx] += w_k * v;
                o2[gx] += w_k * v * v;
            }
        }
    }
}

/// Hand-rolled AVX2 implementation of [`compute_channel_grids`]. Per `gy` row,
/// holds the row's `span ≤ 16` cells of `(N, S1, S2)` in 6 YMM accumulators
/// (two `__m256` each) across the `k`-loop; the centered plane streams through
/// once. Inner loop: **2 loads, 2 mul, 6 FMA, 2 broadcasts** for 16 lanes — the
/// whole plane is in L1, so every load hits L1. Overwrites the output grids
/// (per [`compute_channel_grids`]); the lanes past `span` are spilled to a stack
/// scratch and discarded by the `copy_from_slice` that follows.
///
/// The `span → 16` lane padding wastes `(16 − span) / 16` of the FMA throughput
/// (~19% at `span = 13`), per the spec — acceptable for the gain of holding the
/// row's accumulators in registers. Combine (per-row, after the k-loop) is scalar
/// over `span` cells — small relative to the inner loop (AVX2 `rsqrt` is an open
/// question called out in the spec; skip for stage 1).
///
/// # Safety
///
/// The caller must uphold:
///
/// 1. **CPU features:** `is_x86_feature_detected!("avx2") &&
///    is_x86_feature_detected!("fma")` (the `#[target_feature]` enable is
///    sound only when the runtime has these).
/// 2. **Span fits in 16 lanes:** `span ≤ 16` (one pair of `__m256` per `gy`
///    row); larger spans must take the scalar fallback.
/// 3. **Plane buffer covers every 16-wide load:** for every support pixel `k`
///    with `r_k = pixel/resolution`, `c_k = pixel%resolution`, and every
///    `gy ∈ [0, span)`,
///    `(gy + win_oy + r_k) * istride + (win_ox + c_k) + 16 ≤ plane.len()`.
///    This is what the cache's `istride = align_up(cacheW − 1 + 16, 8)`
///    padding plus the `win_o*`/`r_k`/`c_k` bounds in `search_shift` guarantee.
/// 4. **Output buffers cover the span × span grid:**
///    `g_n.len()`, `g_s1.len()`, `g_s2.len() ≥ span * span`.
/// 5. **`support.pixels` is laid out as `row * resolution + col`** so `r_k <
///    resolution` and `c_k < resolution` — both contribute to invariant 3.
///
/// `debug_assert!`s spot-check (2), (4), and the tight per-call worst case of
/// (3); release builds rely on the caller's contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn compute_channel_grids_avx2(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g_n: &mut [f32],
    g_s1: &mut [f32],
    g_s2: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert!(span <= 16, "AVX2 kernel handles spans up to 16, got {span}");
    let gsz = span * span;
    debug_assert!(
        g_n.len() >= gsz && g_s1.len() >= gsz && g_s2.len() >= gsz,
        "output grids must cover span * span"
    );
    // Worst-case 16-wide load index: largest `r_k`, `c_k`, and `gy`. The support
    // is `row * resolution + col` with both `< resolution`, so `r_k, c_k ≤
    // resolution − 1`; `gy ≤ span − 1`. (Spot-check the bound at the per-call
    // worst case; the caller is responsible per the SAFETY contract.)
    if let Some(&p_max) = support.pixels.iter().max() {
        let r_max = p_max / resolution;
        let c_max = p_max % resolution;
        let worst_base = (span - 1 + win_oy + r_max) * istride + (win_ox + c_max);
        debug_assert!(
            worst_base + 16 <= plane.len(),
            "AVX2 load would read past the centered plane: \
             worst_base+16={} > plane.len()={} (span={span}, istride={istride}, \
             win_oy={win_oy}, win_ox={win_ox}, r_max={r_max}, c_max={c_max})",
            worst_base + 16,
            plane.len(),
        );
    }
    let plane_ptr = plane.as_ptr();
    let pixels = support.pixels.as_ptr();
    let n = support.pixels.len();
    let kern_ptr = kern.as_ptr();
    let w_ptr = w.as_ptr();
    // Per-row temporaries that mirror the YMM accumulators when we store them.
    let mut tmp = [0.0f32; 16];
    for gy in 0..span {
        let row_y = gy + win_oy;
        let mut n_lo = _mm256_setzero_ps();
        let mut n_hi = _mm256_setzero_ps();
        let mut s1_lo = _mm256_setzero_ps();
        let mut s1_hi = _mm256_setzero_ps();
        let mut s2_lo = _mm256_setzero_ps();
        let mut s2_hi = _mm256_setzero_ps();
        for k in 0..n {
            let p = *pixels.add(k);
            let r_k = p / resolution;
            let c_k = p % resolution;
            let base = (row_y + r_k) * istride + (win_ox + c_k);
            let src_lo = _mm256_loadu_ps(plane_ptr.add(base));
            let src_hi = _mm256_loadu_ps(plane_ptr.add(base + 8));
            let kb = _mm256_set1_ps(*kern_ptr.add(k));
            let wb = _mm256_set1_ps(*w_ptr.add(k));
            n_lo = _mm256_fmadd_ps(kb, src_lo, n_lo);
            n_hi = _mm256_fmadd_ps(kb, src_hi, n_hi);
            s1_lo = _mm256_fmadd_ps(wb, src_lo, s1_lo);
            s1_hi = _mm256_fmadd_ps(wb, src_hi, s1_hi);
            let sq_lo = _mm256_mul_ps(src_lo, src_lo);
            let sq_hi = _mm256_mul_ps(src_hi, src_hi);
            s2_lo = _mm256_fmadd_ps(wb, sq_lo, s2_lo);
            s2_hi = _mm256_fmadd_ps(wb, sq_hi, s2_hi);
        }
        // Combine: spill the row's 16-lane accumulators and copy the first `span`
        // lanes into the channel-shared grid maps. The spilled buffer is `[lo|hi]`.
        let gbase = gy * span;
        let mut store_first = |acc_lo: __m256, acc_hi: __m256, dst: &mut [f32]| {
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc_lo);
            _mm256_storeu_ps(tmp.as_mut_ptr().add(8), acc_hi);
            dst[..span].copy_from_slice(&tmp[..span]);
        };
        store_first(n_lo, n_hi, &mut g_n[gbase..][..span]);
        store_first(s1_lo, s1_hi, &mut g_s1[gbase..][..span]);
        store_first(s2_lo, s2_hi, &mut g_s2[gbase..][..span]);
    }
}

/// Accumulate the per-shift count of out-of-frame support pixels (`invalid` is
/// `1.0` where the tile pixel is invalid, else `0.0`). A shift is scorable iff its
/// count is `0` (every support pixel in frame), the grid analogue of
/// [`extract_core`](super::extract_core) returning `false` for any invalid support
/// pixel. The invalidity plane lives in the same `istride` row layout as the
/// centered planes.
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_count(
    invalid: &[f32],
    support: &Support,
    resolution: usize,
    istride: usize,
    span: usize,
    win_oy: usize,
    win_ox: usize,
    g: &mut [f32],
) {
    for &p in &support.pixels {
        let r_k = p / resolution;
        let c_k = p % resolution;
        for gy in 0..span {
            let src = &invalid[(gy + win_oy + r_k) * istride + (win_ox + c_k)..][..span];
            let gr = &mut g[gy * span..][..span];
            for gx in 0..span {
                gr[gx] += src[gx];
            }
        }
    }
}

/// Per-cell counterpart to [`compute_channel_grids`]: score one (and only one)
/// shift `(win_y, win_x)` in the cache plane, returning the three correlation
/// sums needed to assemble the windowed ZNCC at that single cell:
///
///   `n  = Σ_k kern[k] · I[(win_y + r_k, win_x + c_k)]`
///   `s1 = Σ_k w[k]    · I[(win_y + r_k, win_x + c_k)]`
///   `s2 = Σ_k w[k]    · I[(win_y + r_k, win_x + c_k)]²`
///
/// Algebraically a single shift's slice of the grid form, but evaluated by a
/// direct loop over support pixels (no SAXPY across the grid row) — the
/// fundamental primitive for the "+"-descent search strategy, where each call
/// scores one cell and the descent visits ≲ 10 cells per search instead of
/// 169 = `(2·6+1)²`. Mirrors [`compute_channel_grids_scalar`]'s algebra
/// pixel-for-pixel; the AVX2 form below uses `vgatherdps`.
///
/// Runtime-dispatched to the AVX2 kernel where available; otherwise scalar.
#[allow(clippy::too_many_arguments)]
pub(super) fn score_cell_one_channel(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the runtime feature check is the `#[target_feature]`
            // pre-condition; the per-pixel cache index is the same expression
            // `compute_channel_grids_avx2` validates, so its in-bounds-load
            // contract carries over for the gather lanes here.
            unsafe {
                return score_cell_one_channel_avx2(
                    plane, support, kern, w, resolution, istride, win_y, win_x,
                );
            }
        }
    }
    score_cell_one_channel_scalar(plane, support, kern, w, resolution, istride, win_y, win_x)
}

/// Scalar reference for [`score_cell_one_channel`]: the algebra the AVX2 kernel
/// must match, and the non-x86 / non-AVX2 fallback.
#[allow(clippy::too_many_arguments)]
pub(super) fn score_cell_one_channel_scalar(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> (f32, f32, f32) {
    let mut n = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    for (k, &p) in support.pixels.iter().enumerate() {
        let r_k = p / resolution;
        let c_k = p % resolution;
        let v = plane[(win_y + r_k) * istride + (win_x + c_k)];
        n += kern[k] * v;
        s1 += w[k] * v;
        s2 += w[k] * v * v;
    }
    (n, s1, s2)
}

/// Hand-rolled AVX2 implementation of [`score_cell_one_channel`]: 8 support
/// pixels per iteration via `_mm256_i32gather_ps` against a freshly-built
/// per-batch index vector, three FMAs per iteration into 8-lane accumulators
/// (`n`, `s1`, `s2`), with a horizontal reduce at the end and a scalar tail for
/// the support's remainder.
///
/// The gather is the path's bottleneck (Intel `vgatherdps` is roughly
/// 10-12 cycles per lane on this generation, so 8 lanes ≈ 80 cycles); even so,
/// the per-call total stays well under the SAXPY's per-cell amortized cost
/// because we only visit a handful of cells per descent. The lane-fill of the
/// index vector is done from a stack buffer, so the gather sees a fresh
/// 8-element index array each iteration.
///
/// # Safety
///
/// 1. **CPU features:** `is_x86_feature_detected!("avx2") &&
///    is_x86_feature_detected!("fma")`.
/// 2. **Plane covers every gathered index:** for each support pixel `k` with
///    `r_k = pixel/resolution`, `c_k = pixel%resolution`, the byte offset
///    `(win_y + r_k) * istride + (win_x + c_k)` must lie within
///    `plane[..]` — exactly the contract `compute_channel_grids_avx2` already
///    relies on for one shift cell, and the same `cache_res = R + 4·margin`
///    plus `|win_y - cache_c0|, |win_x - cache_c0| ≤ margin` invariants guard
///    it.
/// 3. **Kern / weight slices cover the support:** `kern.len() >=
///    support.pixels.len()`, `w.len() >= support.pixels.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_cell_one_channel_avx2(
    plane: &[f32],
    support: &Support,
    kern: &[f32],
    w: &[f32],
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    let n = support.pixels.len();
    let pixels = support.pixels.as_ptr();
    let kern_ptr = kern.as_ptr();
    let w_ptr = w.as_ptr();
    let plane_ptr = plane.as_ptr();

    let mut acc_n = _mm256_setzero_ps();
    let mut acc_s1 = _mm256_setzero_ps();
    let mut acc_s2 = _mm256_setzero_ps();

    let chunks = n / 8;
    let mut idx_buf = [0_i32; 8];
    for chunk in 0..chunks {
        let k0 = chunk * 8;
        // Build the 8 cache indices for this batch (per-pixel `(win_y + r_k) *
        // istride + (win_x + c_k)`; the `* 4` byte-scaling lives in the gather's
        // scale argument). `usize` → `i32` is sound here because the cache size
        // is bounded by `cache_res² < i32::MAX` for any realistic patch.
        for (lane, slot) in idx_buf.iter_mut().enumerate() {
            let p = *pixels.add(k0 + lane);
            let r_k = p / resolution;
            let c_k = p % resolution;
            *slot = ((win_y + r_k) * istride + (win_x + c_k)) as i32;
        }
        let indices = _mm256_loadu_si256(idx_buf.as_ptr().cast::<__m256i>());
        // `vgatherdps` with scale = 4 (f32 stride).
        let v8 = _mm256_i32gather_ps::<4>(plane_ptr, indices);
        let kern8 = _mm256_loadu_ps(kern_ptr.add(k0));
        let w8 = _mm256_loadu_ps(w_ptr.add(k0));
        acc_n = _mm256_fmadd_ps(kern8, v8, acc_n);
        acc_s1 = _mm256_fmadd_ps(w8, v8, acc_s1);
        let v_sq = _mm256_mul_ps(v8, v8);
        acc_s2 = _mm256_fmadd_ps(w8, v_sq, acc_s2);
    }

    // Horizontal-reduce the three accumulators into scalar sums.
    let mut tmp = [0.0_f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc_n);
    let mut n_acc = tmp.iter().sum::<f32>();
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc_s1);
    let mut s1_acc = tmp.iter().sum::<f32>();
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc_s2);
    let mut s2_acc = tmp.iter().sum::<f32>();

    // Scalar tail for the trailing support pixels (`n % 8`).
    for k in (chunks * 8)..n {
        let p = *pixels.add(k);
        let r_k = p / resolution;
        let c_k = p % resolution;
        let v = *plane_ptr.add((win_y + r_k) * istride + (win_x + c_k));
        n_acc += *kern_ptr.add(k) * v;
        s1_acc += *w_ptr.add(k) * v;
        s2_acc += *w_ptr.add(k) * v * v;
    }
    (n_acc, s1_acc, s2_acc)
}

/// Per-cell counterpart to [`accumulate_count`]: count out-of-frame support
/// pixels at one shift `(win_y, win_x)`. A shift is scorable iff this returns
/// `0` (every support pixel in frame).
pub(super) fn count_invalid_at_cell(
    invalid: &[f32],
    support: &Support,
    resolution: usize,
    istride: usize,
    win_y: usize,
    win_x: usize,
) -> f32 {
    let mut acc = 0.0_f32;
    for &p in &support.pixels {
        let r_k = p / resolution;
        let c_k = p % resolution;
        acc += invalid[(win_y + r_k) * istride + (win_x + c_k)];
    }
    acc
}
