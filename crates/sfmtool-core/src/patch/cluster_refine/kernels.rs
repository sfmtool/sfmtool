// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Numeric kernels for cluster-patch refinement: the per-(member, level)
//! centered planar tile, the fused affine-gather windowed-ZNCC objective
//! (scalar reference + hand-rolled AVX2), and the Nelder-Mead simplex
//! optimizer the cascade runs on.
//!
//! The objective's hot loop samples `resolution²` support points through an
//! affine grid→source map (4 bilinear taps + weight blend per point) and folds
//! the windowed moments (`Σ kern·I`, `Σ w·I`, `Σ w·I²`) into the same pass —
//! the shape of `keypoint_localize::search_shift`'s three-map accumulation,
//! transplanted from integer shifts to gathered affine samples. Sources are
//! read from a [`LevelTile`]: a **centered** planar-f32 copy of the touched
//! pyramid-level region, converted once per (member, level) rather than per
//! evaluation (the `ContextTile` render-once convention; centering keeps the
//! `f32` variance `S2 − S1²/W` cancellation-safe, and the windowed ZNCC is
//! shift-invariant so nothing needs undoing).

use super::prof;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::patch::normal_refine::{Support, FLAT_NORM_SQ_EPS};
use crate::patch::view_selection::AffineCoreMap;

/// An integer pixel rectangle in level coordinates, half-open (`[x0, x1) ×
/// [y0, y1)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rect {
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
}

impl Rect {
    pub(super) const EMPTY: Rect = Rect {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    fn clip(&self, w: i64, h: i64) -> Rect {
        Rect {
            x0: self.x0.max(0),
            y0: self.y0.max(0),
            x1: self.x1.min(w),
            y1: self.y1.min(h),
        }
    }

    fn union(&self, other: &Rect) -> Rect {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Rect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    fn contains(&self, other: &Rect) -> bool {
        other.is_empty()
            || (self.x0 <= other.x0
                && self.y0 <= other.y0
                && self.x1 >= other.x1
                && self.y1 >= other.y1)
    }

    fn expand(&self, m: i64) -> Rect {
        Rect {
            x0: self.x0 - m,
            y0: self.y0 - m,
            x1: self.x1 + m,
            y1: self.y1 + m,
        }
    }
}

/// The axis-aligned bounding box of the affine image of the full grid
/// (`col, row ∈ [0, resolution)`), expanded by 2 px so every bilinear tap of
/// every in-box sample is covered. Affine ⇒ the extremes sit at the four grid
/// corners. [`Rect::EMPTY`] when the map is non-finite (a degenerate warp
/// scores as out-of-frame).
pub(super) fn grid_bbox(map: &AffineCoreMap, resolution: u32) -> Rect {
    let a = &map.a;
    let r1 = (resolution.max(1) - 1) as f64;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (c, r) in [(0.0, 0.0), (r1, 0.0), (0.0, r1), (r1, r1)] {
        let x = a[0] * c + a[1] * r + a[2];
        let y = a[3] * c + a[4] * r + a[5];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if !(min_x.is_finite() && max_x.is_finite() && min_y.is_finite() && max_y.is_finite()) {
        return Rect::EMPTY;
    }
    Rect {
        x0: min_x.floor() as i64 - 2,
        y0: min_y.floor() as i64 - 2,
        x1: max_x.ceil() as i64 + 2,
        y1: max_y.ceil() as i64 + 2,
    }
}

/// A centered planar-f32 copy of one rectangular region of one pyramid level.
/// `planes[c · w·h + row · w + col] = I_c − mean_c` over the tile; the tile is
/// always a subset of the image, so an in-tile bilinear tap is an in-frame
/// tap, and (given the tile covers the evaluation's clipped
/// [`grid_bbox`]) an out-of-tile tap is an out-of-frame tap — the tile-bounds
/// lane test doubles as the all-in-frame test.
pub(super) struct LevelTile {
    pub(super) level: usize,
    pub(super) x0: i64,
    pub(super) y0: i64,
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) channels: usize,
    planes: Vec<f32>,
}

impl LevelTile {
    fn rect(&self) -> Rect {
        Rect {
            x0: self.x0,
            y0: self.y0,
            x1: self.x0 + self.w as i64,
            y1: self.y0 + self.h as i64,
        }
    }

    pub(super) fn plane(&self, c: usize) -> &[f32] {
        &self.planes[c * self.w * self.h..][..self.w * self.h]
    }

    /// Copy `rect` (already clipped and non-empty) of `img` into a centered
    /// planar tile.
    fn build(img: &ImageU8, level: usize, rect: Rect) -> LevelTile {
        prof::count(&prof::N_TILE_BUILDS, 1);
        prof::count(
            &prof::N_TILE_PIXELS,
            ((rect.x1 - rect.x0) * (rect.y1 - rect.y0)) as u64 * img.channels() as u64,
        );
        let w = (rect.x1 - rect.x0) as usize;
        let h = (rect.y1 - rect.y0) as usize;
        let ch = img.channels() as usize;
        let stride = img.width() as usize * ch;
        let data = img.data();
        let mut planes = vec![0f32; ch * w * h];
        for c in 0..ch {
            let plane = &mut planes[c * w * h..][..w * h];
            let mut sum = 0f64;
            for row in 0..h {
                let src = &data[(rect.y0 as usize + row) * stride + rect.x0 as usize * ch..];
                let dst = &mut plane[row * w..][..w];
                for (col, d) in dst.iter_mut().enumerate() {
                    let v = src[col * ch + c] as f32;
                    *d = v;
                    sum += v as f64;
                }
            }
            let mean = (sum / (w * h) as f64) as f32;
            for d in plane.iter_mut() {
                *d -= mean;
            }
        }
        LevelTile {
            level,
            x0: rect.x0,
            y0: rect.y0,
            w,
            h,
            channels: ch,
            planes,
        }
    }
}

/// Growth margin (level px) added around a requested region when a tile is
/// (re)built, so small optimizer excursions don't force a rebuild per step.
const TILE_MARGIN: i64 = 8;

/// Per-member cache of [`LevelTile`]s, one per touched pyramid level, grown
/// lazily to cover each evaluation's clipped [`grid_bbox`].
#[derive(Default)]
pub(super) struct TileCache {
    tiles: Vec<LevelTile>,
}

impl TileCache {
    /// The tile for `level` covering `needed ∩ image`; `None` when that
    /// intersection is empty (the whole footprint is out of frame). Rebuilds
    /// (union of the old region and `needed`, plus [`TILE_MARGIN`]) when the
    /// existing tile does not cover the request.
    pub(super) fn get_or_build(
        &mut self,
        pyramid: &ImageU8Pyramid,
        level: usize,
        needed: Rect,
    ) -> Option<&LevelTile> {
        let img = pyramid.level(level);
        let (w, h) = (img.width() as i64, img.height() as i64);
        let clipped = needed.clip(w, h);
        if clipped.is_empty() {
            return None;
        }
        let idx = self.tiles.iter().position(|t| t.level == level);
        if let Some(i) = idx {
            if !self.tiles[i].rect().contains(&clipped) {
                let grown = self.tiles[i]
                    .rect()
                    .union(&clipped)
                    .expand(TILE_MARGIN)
                    .clip(w, h);
                self.tiles[i] = prof::TILE.time(|| LevelTile::build(img, level, grown));
            }
            return Some(&self.tiles[i]);
        }
        let grown = clipped.expand(TILE_MARGIN).clip(w, h);
        let tile = prof::TILE.time(|| LevelTile::build(img, level, grown));
        self.tiles.push(tile);
        Some(self.tiles.last().unwrap())
    }
}

/// The support-grid tables shared by every cluster of one refine call: per
/// support pixel its grid column/row (f32, for the FMA coordinate generation)
/// and window weight, padded to a multiple of 8 lanes. Pad entries duplicate
/// the last support point's coordinates (always addressable whenever the real
/// point is) with zero weights, so they contribute nothing to any sum.
pub(super) struct SupportTables {
    /// Real support-pixel count.
    pub(super) n: usize,
    /// Padded lane count (`n` rounded up to a multiple of 8).
    pub(super) n_padded: usize,
    pub(super) cols: Vec<f32>,
    pub(super) rows: Vec<f32>,
    /// Window weights (f32), zero-padded.
    pub(super) w32: Vec<f32>,
    /// `Σ w` over the real support (f64, the windowed-mean denominator).
    pub(super) total_weight: f64,
}

impl SupportTables {
    pub(super) fn new(support: &Support, resolution: u32) -> SupportTables {
        let n = support.pixels.len();
        let n_padded = n.div_ceil(8).max(1) * 8;
        let r = resolution as usize;
        let mut cols = Vec::with_capacity(n_padded);
        let mut rows = Vec::with_capacity(n_padded);
        let mut w32 = Vec::with_capacity(n_padded);
        for (k, &p) in support.pixels.iter().enumerate() {
            cols.push((p % r) as f32);
            rows.push((p / r) as f32);
            w32.push(support.weights[k] as f32);
        }
        let (lc, lr) = (cols[n - 1], rows[n - 1]);
        cols.resize(n_padded, lc);
        rows.resize(n_padded, lr);
        w32.resize(n_padded, 0.0);
        SupportTables {
            n,
            n_padded,
            cols,
            rows,
            w32,
            total_weight: support.total_weight,
        }
    }
}

/// One cluster's z-normalized reference template in correlation-kernel form:
/// per surviving channel, `kern[k] = √w_k · t̂_c[k]` (the z-normalized
/// template already carries one `√w` fold, so `Σ kern·v` realizes the
/// windowed inner product against a raw sample stack) plus `Σ kern` for the
/// mean-removal term. `src_channels` maps each compacted template channel
/// back to its original source channel, so the reference's channel A is never
/// correlated against a member's channel B.
pub(super) struct TemplateKernel {
    /// Surviving channel count.
    pub(super) channels: usize,
    /// Original source-channel index per template channel.
    pub(super) src_channels: Vec<usize>,
    /// `channels × n_padded`, zero-padded per channel.
    pub(super) kern: Vec<f32>,
    /// `Σ kern` per channel (f64).
    pub(super) kern_sums: Vec<f64>,
}

/// The three per-channel accumulator sums of one fused pass, reduced to f64.
struct ChannelSums {
    n_cross: f64,
    s1: f64,
    s2: f64,
}

/// Assemble one channel's windowed ZNCC from the fused sums: mean-remove the
/// cross term and normalize by the member's windowed norm. A flat member
/// channel (windowed norm² below [`FLAT_NORM_SQ_EPS`]) contributes `0` rather
/// than a garbage dot — the `score_raw_against_reference` convention.
fn combine_channel(sums: ChannelSums, total_weight: f64, kern_sum: f64) -> f64 {
    let mean = sums.s1 / total_weight;
    let norm_sq = sums.s2 - sums.s1 * mean;
    if norm_sq < FLAT_NORM_SQ_EPS {
        return 0.0;
    }
    (sums.n_cross - mean * kern_sum) / norm_sq.sqrt()
}

/// Score the member image (via its [`LevelTile`]) at `map` (level
/// coordinates) against the template: the channel-averaged windowed ZNCC.
/// `None` when any support sample's bilinear taps leave the tile — which,
/// given the tile covers the evaluation's clipped [`grid_bbox`], is exactly
/// the all-in-frame rejection.
///
/// Runtime-dispatched to the AVX2+FMA kernel where available; the scalar form
/// is the reference, the non-x86 fallback, and the dual-path test's oracle.
pub(super) fn eval_zncc(
    map: &AffineCoreMap,
    tile: &LevelTile,
    tables: &SupportTables,
    tmpl: &TemplateKernel,
) -> Option<f64> {
    // Tile-local f32 coefficients with the −0.5 pixel-center shift folded in:
    // the per-lane result is directly `gx` of `bilinear_geometry`'s
    // convention, in tile coordinates (small magnitudes keep f32 exact).
    let a = &map.a;
    let al = [
        a[0] as f32,
        a[1] as f32,
        (a[2] - 0.5 - tile.x0 as f64) as f32,
        a[3] as f32,
        a[4] as f32,
        (a[5] - 0.5 - tile.y0 as f64) as f32,
    ];
    #[cfg(target_arch = "x86_64")]
    {
        if tmpl.channels <= MAX_AVX2_CHANNELS
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature check above.
            return unsafe { eval_zncc_avx2(al, tile, tables, tmpl) };
        }
    }
    eval_zncc_scalar(al, tile, tables, tmpl)
}

/// Scalar reference for [`eval_zncc`]: the algebra the AVX2 kernel must
/// match, and the non-x86 / non-AVX2 fallback. Four independent accumulator
/// lanes per sum keep the reduction shape SSE-friendly (the house
/// `weighted_moments_scalar` convention).
pub(super) fn eval_zncc_scalar(
    a: [f32; 6],
    tile: &LevelTile,
    tables: &SupportTables,
    tmpl: &TemplateKernel,
) -> Option<f64> {
    let n = tables.n;
    let (tw, th) = (tile.w as i64, tile.h as i64);
    let mut score = 0.0f64;
    for (tc, &src_c) in tmpl.src_channels.iter().enumerate() {
        if src_c >= tile.channels {
            // The member image has fewer channels than the reference space;
            // that channel is undefined here — no contribution.
            continue;
        }
        let plane = tile.plane(src_c);
        let kern = &tmpl.kern[tc * tables.n_padded..][..n];
        let mut acc_n = [0f32; 4];
        let mut acc_s1 = [0f32; 4];
        let mut acc_s2 = [0f32; 4];
        for (k, &kern_k) in kern.iter().enumerate() {
            let gx = a[0] * tables.cols[k] + a[1] * tables.rows[k] + a[2];
            let gy = a[3] * tables.cols[k] + a[4] * tables.rows[k] + a[5];
            if !gx.is_finite() || !gy.is_finite() {
                return None;
            }
            let x0 = gx.floor();
            let y0 = gy.floor();
            let ix = x0 as i64;
            let iy = y0 as i64;
            if ix < 0 || iy < 0 || ix + 1 >= tw || iy + 1 >= th {
                return None;
            }
            let fx = gx - x0;
            let fy = gy - y0;
            let base = iy as usize * tile.w + ix as usize;
            let v00 = plane[base];
            let v10 = plane[base + 1];
            let v01 = plane[base + tile.w];
            let v11 = plane[base + tile.w + 1];
            // Same blend grouping as `corner_weights` / `sample_bilinear_u8`.
            let v = (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11;
            let l = k & 3;
            acc_n[l] += kern_k * v;
            acc_s1[l] += tables.w32[k] * v;
            acc_s2[l] += tables.w32[k] * v * v;
        }
        let sums = ChannelSums {
            n_cross: acc_n.iter().map(|&x| x as f64).sum(),
            s1: acc_s1.iter().map(|&x| x as f64).sum(),
            s2: acc_s2.iter().map(|&x| x as f64).sum(),
        };
        score += combine_channel(sums, tables.total_weight, tmpl.kern_sums[tc]);
    }
    Some(score / tmpl.channels as f64)
}

/// Channel-count ceiling of the AVX2 kernel's fixed pointer tables; templates
/// wider than this (no real image produces them) take the scalar path.
pub(super) const MAX_AVX2_CHANNELS: usize = 4;

/// Horizontal sum of an 8-lane f32 vector.
///
/// # Safety
/// Requires `avx` (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), v);
    tmp.iter().sum()
}

/// Gather the horizontally adjacent bilinear tap pairs `(plane[idx],
/// plane[idx + 1])` for 8 lanes as two deinterleaved vectors, using eight
/// scalar 64-bit loads (each fetches both taps of a pair in one load)
/// instead of two 8-element 32-bit gathers — half the fetched elements per
/// tap row, and plain loads beat microcoded hardware gathers on hybrid
/// (E-core) parts.
///
/// # Safety
///
/// Requires `avx2`; every lane of `idx` must satisfy `idx + 1 <
/// plane.len()` (the caller's lane mask guarantees it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gather_pairs(
    plane: *const f32,
    idx: &[i32; 8],
) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;
    // 8 bytes at plane + idx·4 = floats idx and idx+1. The f32 plane is only
    // 4-byte aligned, so the pair loads must be unaligned reads.
    let pair = |i: usize| (plane.add(idx[i] as usize) as *const i64).read_unaligned();
    let lo = _mm256_set_epi64x(pair(3), pair(2), pair(1), pair(0));
    let hi = _mm256_set_epi64x(pair(7), pair(6), pair(5), pair(4));
    let a = _mm256_castsi256_ps(lo); // [e0 o0 e1 o1 | e2 o2 e3 o3]
    let b = _mm256_castsi256_ps(hi); // [e4 o4 e5 o5 | e6 o6 e7 o7]
    let ev = _mm256_shuffle_ps::<0b10_00_10_00>(a, b); // [e0 e1 e4 e5 | e2 e3 e6 e7]
    let od = _mm256_shuffle_ps::<0b11_01_11_01>(a, b); // [o0 o1 o4 o5 | o2 o3 o6 o7]
                                                       // Restore lane order (64-bit block permutation 0,2,1,3).
    let ev = _mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(_mm256_castps_pd(ev)));
    let od = _mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(_mm256_castps_pd(od)));
    (ev, od)
}

/// Hand-rolled AVX2 implementation of [`eval_zncc`]: 8 support points per
/// iteration, all template channels fused into one pass over the support so
/// the coordinate generation, in-frame mask, tap indices, and fractional
/// blend weights are computed once and shared (they are channel-invariant).
/// Sample coordinates are affine in the grid index, so the `x`/`y` vectors
/// come from FMA on the lane column/row tables; the four bilinear taps come
/// from [`gather_pairs`] 64-bit pair gathers on the centered planes;
/// fractional blends and the three accumulations (`Σ kern·I`, `Σ w·I`,
/// `Σ w·I²`) are FMA. The in-frame test is a vectorized integer lane mask on
/// the tap indices; any failing lane aborts the evaluation (the all-in-frame
/// rule), so the mask doubles as the early-out. Non-finite coordinates
/// convert to `i32::MIN` and are caught by the same mask.
///
/// Per-channel accumulation order matches the channel-major original exactly
/// (each channel folds the same `v` sequence in the same `k` order into the
/// same 8-lane accumulators), so the fusion is bit-exact.
///
/// # Safety
///
/// 1. **CPU features:** `is_x86_feature_detected!("avx2") &&
///    is_x86_feature_detected!("fma")`.
/// 2. **Gather bounds:** every gathered index is validated by the lane mask
///    *before* the gathers (`0 ≤ ix ≤ tile.w − 2`, `0 ≤ iy ≤ tile.h − 2`), so
///    `iy·w + ix + w + 1 ≤ w·h − 1` and both pair gathers of both tap rows
///    stay inside the channel plane.
/// 3. **Table lengths:** `tables.cols/rows/w32` and each `tmpl.kern` channel
///    hold `n_padded` (multiple of 8) lanes; pad lanes carry zero weights and
///    duplicate a real support point's coordinates ([`SupportTables`]).
/// 4. **Channel count:** `tmpl.channels ≤ MAX_AVX2_CHANNELS` (the dispatch
///    gate in [`eval_zncc`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn eval_zncc_avx2(
    a: [f32; 6],
    tile: &LevelTile,
    tables: &SupportTables,
    tmpl: &TemplateKernel,
) -> Option<f64> {
    let n_padded = tables.n_padded;
    // Compact the active channels (source plane present in this tile) into
    // fixed pointer tables so the monomorphized bodies can unroll over them.
    let mut planes = [std::ptr::null::<f32>(); MAX_AVX2_CHANNELS];
    let mut kerns = [std::ptr::null::<f32>(); MAX_AVX2_CHANNELS];
    let mut which = [0usize; MAX_AVX2_CHANNELS];
    let mut n_active = 0usize;
    for (tc, &src_c) in tmpl.src_channels.iter().enumerate() {
        if src_c >= tile.channels {
            continue;
        }
        planes[n_active] = tile.plane(src_c).as_ptr();
        kerns[n_active] = tmpl.kern[tc * n_padded..].as_ptr();
        which[n_active] = tc;
        n_active += 1;
    }
    if n_active == 0 {
        return Some(0.0);
    }
    // SAFETY: forwarded from this function's contract; `n_active` channels
    // of `planes`/`kerns` are valid for `n_padded` lanes.
    let sums = match n_active {
        1 => eval_zncc_avx2_ch::<1>(a, tile, tables, &planes, &kerns),
        2 => eval_zncc_avx2_ch::<2>(a, tile, tables, &planes, &kerns),
        3 => eval_zncc_avx2_ch::<3>(a, tile, tables, &planes, &kerns),
        _ => eval_zncc_avx2_ch::<4>(a, tile, tables, &planes, &kerns),
    }?;
    let mut score = 0.0f64;
    for (c, sum) in sums.into_iter().take(n_active).enumerate() {
        score += combine_channel(sum, tables.total_weight, tmpl.kern_sums[which[c]]);
    }
    Some(score / tmpl.channels as f64)
}

/// The monomorphized fused pass of [`eval_zncc_avx2`] over `CH` active
/// channels: `CH` is a compile-time constant so the per-channel accumulator
/// arrays stay in registers and the channel loop unrolls.
///
/// # Safety
///
/// Same contract as [`eval_zncc_avx2`]; additionally the first `CH` entries
/// of `planes`/`kerns` must be valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn eval_zncc_avx2_ch<const CH: usize>(
    a: [f32; 6],
    tile: &LevelTile,
    tables: &SupportTables,
    planes: &[*const f32; MAX_AVX2_CHANNELS],
    kerns: &[*const f32; MAX_AVX2_CHANNELS],
) -> Option<[ChannelSums; MAX_AVX2_CHANNELS]> {
    use std::arch::x86_64::*;
    let n_padded = tables.n_padded;
    let a0 = _mm256_set1_ps(a[0]);
    let a1 = _mm256_set1_ps(a[1]);
    let a2 = _mm256_set1_ps(a[2]);
    let a3 = _mm256_set1_ps(a[3]);
    let a4 = _mm256_set1_ps(a[4]);
    let a5 = _mm256_set1_ps(a[5]);
    let zero_i = _mm256_setzero_si256();
    let max_ix = _mm256_set1_epi32(tile.w as i32 - 2);
    let max_iy = _mm256_set1_epi32(tile.h as i32 - 2);
    let twv = _mm256_set1_epi32(tile.w as i32);
    let one_ps = _mm256_set1_ps(1.0);
    let cols_ptr = tables.cols.as_ptr();
    let rows_ptr = tables.rows.as_ptr();
    let w_ptr = tables.w32.as_ptr();

    let mut acc_n = [_mm256_setzero_ps(); CH];
    let mut acc_s1 = [_mm256_setzero_ps(); CH];
    let mut acc_s2 = [_mm256_setzero_ps(); CH];
    let mut k = 0usize;
    while k < n_padded {
        let col = _mm256_loadu_ps(cols_ptr.add(k));
        let row = _mm256_loadu_ps(rows_ptr.add(k));
        let gx = _mm256_fmadd_ps(a0, col, _mm256_fmadd_ps(a1, row, a2));
        let gy = _mm256_fmadd_ps(a3, col, _mm256_fmadd_ps(a4, row, a5));
        let x0 = _mm256_floor_ps(gx);
        let y0 = _mm256_floor_ps(gy);
        // Exact: x0/y0 are integral; NaN/overflow become i32::MIN and
        // fail the mask below.
        let ix = _mm256_cvtps_epi32(x0);
        let iy = _mm256_cvtps_epi32(y0);
        let bad = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpgt_epi32(zero_i, ix),
                _mm256_cmpgt_epi32(zero_i, iy),
            ),
            _mm256_or_si256(
                _mm256_cmpgt_epi32(ix, max_ix),
                _mm256_cmpgt_epi32(iy, max_iy),
            ),
        );
        if _mm256_testz_si256(bad, bad) == 0 {
            return None;
        }
        let fx = _mm256_sub_ps(gx, x0);
        let fy = _mm256_sub_ps(gy, y0);
        let idx = _mm256_add_epi32(_mm256_mullo_epi32(iy, twv), ix);
        let idx_b = _mm256_add_epi32(idx, twv);
        let mut idx_arr = [0i32; 8];
        let mut idx_b_arr = [0i32; 8];
        _mm256_storeu_si256(idx_arr.as_mut_ptr() as *mut __m256i, idx);
        _mm256_storeu_si256(idx_b_arr.as_mut_ptr() as *mut __m256i, idx_b);
        let ofx = _mm256_sub_ps(one_ps, fx);
        let ofy = _mm256_sub_ps(one_ps, fy);
        let w00 = _mm256_mul_ps(ofx, ofy);
        let w10 = _mm256_mul_ps(fx, ofy);
        let w01 = _mm256_mul_ps(ofx, fy);
        let w11 = _mm256_mul_ps(fx, fy);
        let w8 = _mm256_loadu_ps(w_ptr.add(k));
        for c in 0..CH {
            let (v00, v10) = gather_pairs(planes[c], &idx_arr);
            let (v01, v11) = gather_pairs(planes[c], &idx_b_arr);
            let v = _mm256_fmadd_ps(
                w11,
                v11,
                _mm256_fmadd_ps(w01, v01, _mm256_fmadd_ps(w10, v10, _mm256_mul_ps(w00, v00))),
            );
            let k8 = _mm256_loadu_ps(kerns[c].add(k));
            acc_n[c] = _mm256_fmadd_ps(k8, v, acc_n[c]);
            acc_s1[c] = _mm256_fmadd_ps(w8, v, acc_s1[c]);
            acc_s2[c] = _mm256_fmadd_ps(w8, _mm256_mul_ps(v, v), acc_s2[c]);
        }
        k += 8;
    }
    let mut sums = [const {
        ChannelSums {
            n_cross: 0.0,
            s1: 0.0,
            s2: 0.0,
        }
    }; MAX_AVX2_CHANNELS];
    for c in 0..CH {
        sums[c] = ChannelSums {
            n_cross: hsum256_ps(acc_n[c]) as f64,
            s1: hsum256_ps(acc_s1[c]) as f64,
            s2: hsum256_ps(acc_s2[c]) as f64,
        };
    }
    Some(sums)
}

/// Dimension ceiling of the allocation-free simplex buffers (the affine
/// stage's 6 parameters — the largest cascade stage).
pub(super) const NM_MAX_DIM: usize = 6;

/// Minimal deterministic Nelder-Mead: simplex seeded at `x0 + scale_i·e_i`,
/// standard coefficients (reflect 1, expand 2, contract 0.5, shrink 0.5),
/// stopping at `max_iters`, a simplex value spread below `tol`, or a stalled
/// search — no improvement of the best value by more than `stall_tol` for
/// `stall_iters` consecutive iterations (effectively clamped to ≥ 1: the
/// first iteration always resets the counter from the `f64::INFINITY` seed;
/// `u32::MAX` puts the exit beyond any reachable `max_iters` — the practical
/// "disabled").
/// Returns the best point and its value. The prototype's optimizer,
/// transcribed; ties in the sort keep insertion order and the returned
/// argmin is the first minimum, so the routine is fully deterministic.
///
/// The stall exit exists for the affine stage: a reflect-heavy 6-dim simplex
/// crawl on a flat objective shrinks its value spread far more slowly than
/// it stops making progress, so the spread test alone runs most members into
/// the `max_iters` cap long after the score stopped moving.
///
/// The simplex lives in fixed `[f64; NM_MAX_DIM]` buffers (`n ≤` 6 across the
/// cascade) and the per-iteration reorder is a stable insertion sort, so the
/// loop allocates nothing — with ~10⁸ iterations per batch the `Vec` churn of
/// the transcription dominated the optimizer's own cost. Arithmetic, sort
/// order (stable, `partial_cmp` ties keep insertion order), and evaluation
/// order are unchanged, so results are bit-identical to the transcription.
#[allow(clippy::too_many_arguments)]
pub(super) fn nelder_mead(
    mut f: impl FnMut(&[f64]) -> f64,
    x0: &[f64],
    scales: &[f64],
    max_iters: u32,
    tol: f64,
    stall_iters: u32,
    stall_tol: f64,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    debug_assert_eq!(scales.len(), n);
    debug_assert!(n <= NM_MAX_DIM);
    let mut pts = [[0.0f64; NM_MAX_DIM]; NM_MAX_DIM + 1];
    let mut vals = [0.0f64; NM_MAX_DIM + 1];
    for (i, p) in pts.iter_mut().take(n + 1).enumerate() {
        p[..n].copy_from_slice(x0);
        if i > 0 {
            p[i - 1] += scales[i - 1];
        }
    }
    for i in 0..=n {
        vals[i] = f(&pts[i][..n]);
    }
    let mut stall_best = f64::INFINITY;
    let mut stalled = 0u32;
    for _ in 0..max_iters {
        // Stable insertion sort by value (ties keep insertion order —
        // deterministic, the same permutation as the stable `sort_by`).
        for i in 1..=n {
            let mut j = i;
            while j > 0
                && vals[j]
                    .partial_cmp(&vals[j - 1])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    == std::cmp::Ordering::Less
            {
                vals.swap(j, j - 1);
                pts.swap(j, j - 1);
                j -= 1;
            }
        }
        if (vals[n] - vals[0]).abs() < tol {
            break;
        }
        if vals[0] < stall_best - stall_tol {
            stall_best = vals[0];
            stalled = 0;
        } else {
            stalled += 1;
            if stalled >= stall_iters {
                break;
            }
        }
        let mut centroid = [0.0f64; NM_MAX_DIM];
        for p in &pts[..n] {
            for (c, &x) in centroid.iter_mut().zip(p) {
                *c += x;
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }
        let worst = pts[n];
        let mut xr = [0.0f64; NM_MAX_DIM];
        for i in 0..n {
            xr[i] = centroid[i] + (centroid[i] - worst[i]);
        }
        let fr = f(&xr[..n]);
        if fr < vals[0] {
            let mut xe = [0.0f64; NM_MAX_DIM];
            for i in 0..n {
                xe[i] = centroid[i] + 2.0 * (centroid[i] - worst[i]);
            }
            let fe = f(&xe[..n]);
            if fe < fr {
                pts[n] = xe;
                vals[n] = fe;
            } else {
                pts[n] = xr;
                vals[n] = fr;
            }
        } else if fr < vals[n - 1] {
            pts[n] = xr;
            vals[n] = fr;
        } else {
            let mut xc = [0.0f64; NM_MAX_DIM];
            for i in 0..n {
                xc[i] = centroid[i] + 0.5 * (worst[i] - centroid[i]);
            }
            let fc = f(&xc[..n]);
            if fc < vals[n] {
                pts[n] = xc;
                vals[n] = fc;
            } else {
                // Shrink every non-best point toward the best.
                let best = pts[0];
                for i in 1..=n {
                    for (x, &b) in pts[i][..n].iter_mut().zip(&best[..n]) {
                        *x = b + 0.5 * (*x - b);
                    }
                    vals[i] = f(&pts[i][..n]);
                }
            }
        }
    }
    // First minimum (numpy-argmin convention).
    let mut best = 0;
    for (i, &v) in vals.iter().enumerate().take(n + 1).skip(1) {
        if v < vals[best] {
            best = i;
        }
    }
    (pts[best][..n].to_vec(), vals[best])
}
