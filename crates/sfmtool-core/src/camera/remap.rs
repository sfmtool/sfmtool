// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image resampling for warp maps.

use rayon::prelude::*;

use crate::camera::sampler::bilinear_mip_level;
use crate::camera::warp_map::{parallelize_rows, WarpMap, REMAP_PAR_MIN_PIXELS};

mod aniso_avx2;
pub mod prof;
mod texels;

pub(crate) use texels::{bilinear_cell, HeldLevel, LevelTexels, PyramidTexels};

pub use crate::camera::image::{ImageF32WithGrad, ImageU8, ImageU8Pyramid};

/// Run `fill_row` over every destination row, sequentially for small outputs
/// and for remaps nested inside an already-parallel caller (where rayon
/// scaffolding dwarfs the row work — e.g. every patch-sized remap, which runs
/// from inside the per-patch `par_iter`), and via rayon otherwise. See
/// [`parallelize_rows`]. Returns the concatenated row data.
fn remap_rows(
    out_w: u32,
    out_h: u32,
    channels: u32,
    fill_row: impl Fn(u32, &mut [u8]) + Sync,
) -> Vec<u8> {
    let out_stride = out_w as usize * channels as usize;
    if !parallelize_rows((out_w as usize) * (out_h as usize), REMAP_PAR_MIN_PIXELS) {
        let mut data = vec![0u8; out_stride * out_h as usize];
        for (row, row_data) in data.chunks_exact_mut(out_stride).enumerate() {
            fill_row(row as u32, row_data);
        }
        data
    } else {
        let rows: Vec<Vec<u8>> = (0..out_h)
            .into_par_iter()
            .map(|row| {
                let mut row_data = vec![0u8; out_stride];
                fill_row(row, &mut row_data);
                row_data
            })
            .collect();
        let mut data = Vec::with_capacity(out_stride * out_h as usize);
        for row in rows {
            data.extend_from_slice(&row);
        }
        data
    }
}

/// Bilinear sample from a multi-channel u8 image, returning an f32 value for
/// the specified channel.
///
/// Coordinates use the pixel-center-at-0.5 convention: sampling at (0.5, 0.5)
/// returns the exact value of the top-left pixel. Out-of-bounds coordinates
/// are clamped to the nearest edge pixel.
///
/// `#[inline]` because the anisotropic multi-tap footprint walk
/// (`sample_aniso_with_grad` / `remap_aniso_with_pyramid`) calls this up to
/// `channels · n · 2` times per output pixel; inlining lets the corner geometry
/// fold across taps and removes the call overhead on that hot path.
#[inline]
pub fn sample_bilinear_u8(img: &ImageU8, x: f32, y: f32, channel: u32) -> f32 {
    let (v00, v10, v01, v11, fx, fy) = bilinear_taps(&HeldLevel::whole(img), x, y, channel);
    (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
}

/// Bilinear sample plus the analytic image gradient `(∂I/∂x, ∂I/∂y)` in
/// source-pixel coords, from the same four taps as [`sample_bilinear_u8`] (no
/// extra fetch). Returns `(value, dI_dx, dI_dy)`.
///
/// Gradient closed-forms (with `fx, fy ∈ [0, 1]` the in-cell fractions):
///
/// ```text
/// dI_dx = (1-fy)·(v10 − v00) + fy·(v11 − v01)
/// dI_dy = (1-fx)·(v01 − v00) + fx·(v11 − v10)
/// ```
///
/// At the image boundary the gradient is *clamped*: when both taps fall on the
/// same clamped pixel the per-axis difference is zero by construction, so the
/// returned gradient is the natural extension of bilinear's edge-clamping.
pub fn sample_bilinear_with_grad_u8(
    img: &ImageU8,
    x: f32,
    y: f32,
    channel: u32,
) -> (f32, f32, f32) {
    bilinear_with_grad(&HeldLevel::whole(img), x, y, channel)
}

/// [`sample_bilinear_with_grad_u8`] of one channel of any level.
#[inline]
fn bilinear_with_grad<T: LevelTexels>(level: &T, x: f32, y: f32, channel: u32) -> (f32, f32, f32) {
    let (v00, v10, v01, v11, fx, fy) = bilinear_taps(level, x, y, channel);
    let val = (1.0 - fx) * (1.0 - fy) * v00
        + fx * (1.0 - fy) * v10
        + (1.0 - fx) * fy * v01
        + fx * fy * v11;
    let di_dx = (1.0 - fy) * (v10 - v00) + fy * (v11 - v01);
    let di_dy = (1.0 - fx) * (v01 - v00) + fx * (v11 - v10);
    (val, di_dx, di_dy)
}

/// The four bilinear blend weights for `(v00, v10, v01, v11)` from the in-cell
/// fractions. Grouped exactly as the `(1.0 - fx) * (1.0 - fy) * v` products in
/// [`sample_bilinear_u8`], so weight-based gathers stay bit-identical to the
/// per-channel path.
#[inline]
fn corner_weights(fx: f32, fy: f32) -> [f32; 4] {
    [
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ]
}

/// Fetch the four bilinear taps `(v00, v10, v01, v11)` plus the in-cell fractions
/// `(fx, fy)` for the sample at `(x, y)` in channel `channel`. Shared by the
/// value-only and value+gradient single-channel samplers; builds on
/// [`bilinear_cell`] and [`LevelTexels::with_cell`].
#[inline]
fn bilinear_taps<T: LevelTexels>(
    level: &T,
    x: f32,
    y: f32,
    channel: u32,
) -> (f32, f32, f32, f32, f32, f32) {
    let (x0, y0, fx, fy) = bilinear_cell(x, y);
    let ch = channel as usize;
    level.with_cell(x0, y0, |data, idx| {
        (
            data[idx[0] + ch] as f32,
            data[idx[1] + ch] as f32,
            data[idx[2] + ch] as f32,
            data[idx[3] + ch] as f32,
            fx,
            fy,
        )
    })
}

/// Bilinearly sample **every channel** of `img` at `(x, y)` in one shot, writing
/// the rounded/clamped `u8` result to `dst[0..channels]`: [`bilinear_all`] of
/// the whole image, the reference the tests pin the affine fast path and the
/// per-channel [`sample_bilinear_u8`] against.
#[cfg(test)]
#[inline]
pub(crate) fn sample_bilinear_u8_all(img: &ImageU8, x: f32, y: f32, dst: &mut [u8]) {
    bilinear_all(&HeldLevel::whole(img), x, y, dst);
}

/// Bilinearly sample **every channel** of `level` at `(x, y)` in one shot,
/// writing the rounded/clamped `u8` result to `dst[0..channels]`. The corner
/// geometry is computed once and reused across channels, where per-channel
/// [`sample_bilinear_u8`] recomputes floor/clamp/index/fractions for each
/// channel. Output is bit-identical to calling `sample_bilinear_u8` per
/// channel and rounding with `(val + 0.5).clamp(0.0, 255.0)`.
///
/// `dst.len()` must be at least the level's channels.
#[inline]
pub(crate) fn bilinear_all<T: LevelTexels>(level: &T, x: f32, y: f32, dst: &mut [u8]) {
    let c = level.channels();
    debug_assert!(
        dst.len() >= c,
        "sample_bilinear_u8_all: dst.len() {} < channels {c}",
        dst.len()
    );
    let (x0, y0, fx, fy) = bilinear_cell(x, y);
    let w = corner_weights(fx, fy);
    level.with_cell(x0, y0, |data, idx| {
        for (ch, slot) in dst.iter_mut().take(c).enumerate() {
            let val = w[0] * data[idx[0] + ch] as f32
                + w[1] * data[idx[1] + ch] as f32
                + w[2] * data[idx[2] + ch] as f32
                + w[3] * data[idx[3] + ch] as f32;
            *slot = (val + 0.5).clamp(0.0, 255.0) as u8;
        }
    });
}

/// Bilinearly sample **every channel** of `img` at `(x, y)`, writing the raw
/// (unrounded) value and the analytic gradient `(∂I/∂x, ∂I/∂y)` for channel `ch`
/// to `value[ch]` / `grad_x[ch]` / `grad_y[ch]`. The channel-batched counterpart
/// of [`sample_bilinear_with_grad_u8`]: geometry is computed once and reused
/// across channels, and the value/gradient closed-forms are identical, so the
/// per-channel results are bit-identical. Values are *not* rounded to `u8` (the
/// GN refiner needs the unquantized value and gradient).
///
/// `value`, `grad_x`, `grad_y` must each have length at least `img.channels`.
#[inline]
fn sample_bilinear_with_grad_u8_all<T: LevelTexels>(
    level: &T,
    x: f32,
    y: f32,
    value: &mut [f32],
    grad_x: &mut [f32],
    grad_y: &mut [f32],
) {
    let c = level.channels();
    debug_assert!(
        value.len() >= c && grad_x.len() >= c && grad_y.len() >= c,
        "sample_bilinear_with_grad_u8_all: dst slices too short for {c} channels"
    );
    let (x0, y0, fx, fy) = bilinear_cell(x, y);
    let w = corner_weights(fx, fy);
    level.with_cell(x0, y0, |data, idx| {
        for ch in 0..c {
            let v00 = data[idx[0] + ch] as f32;
            let v10 = data[idx[1] + ch] as f32;
            let v01 = data[idx[2] + ch] as f32;
            let v11 = data[idx[3] + ch] as f32;
            // Same associativity as `sample_bilinear_with_grad_u8`.
            value[ch] = w[0] * v00 + w[1] * v10 + w[2] * v01 + w[3] * v11;
            grad_x[ch] = (1.0 - fy) * (v10 - v00) + fy * (v11 - v01);
            grad_y[ch] = (1.0 - fx) * (v01 - v00) + fx * (v11 - v10);
        }
    });
}

/// Apply a warp map to an image using bilinear interpolation.
///
/// For each pixel `(col, row)` in the output:
///   1. Look up source coordinates `(sx, sy)` from the warp map.
///   2. If `(sx, sy)` is valid (not NaN), bilinearly interpolate from the
///      source image.
///   3. If invalid, write zero (black).
///
/// The output image has the same dimensions as the warp map and the same
/// number of channels as the input image. Rows are processed in parallel
/// with rayon.
pub fn remap_bilinear(src: &ImageU8, map: &WarpMap) -> ImageU8 {
    remap_bilinear_held(&HeldLevel::whole(src), map)
}

/// [`remap_bilinear`] of a held level: level 0 of a pyramid, whole or a
/// region of it, as [`render_tile`](crate::camera::sampler::render_tile)
/// reads it.
pub(crate) fn remap_bilinear_held(src: &HeldLevel<'_>, map: &WarpMap) -> ImageU8 {
    let c = src.image().channels();
    let data = remap_rows(map.width(), map.height(), c, |row, row_data| {
        bilinear_row(src, map, row, row_data);
    });
    count_call(map);
    ImageU8::new(map.width(), map.height(), c, data)
}

/// The `SFMTOOL_PROFILE` counters of one remap call.
pub(crate) fn count_call(map: &WarpMap) {
    prof::add(&prof::CALLS, 1);
    prof::add(&prof::PX_TOTAL, map.width() as u64 * map.height() as u64);
}

/// One output row of [`remap_bilinear`] from any level, into `row_data`
/// (zeroed beforehand).
pub(crate) fn bilinear_row<T: LevelTexels>(src: &T, map: &WarpMap, row: u32, row_data: &mut [u8]) {
    let c = src.channels();
    let mut sampled = 0u64;
    for col in 0..map.width() {
        let (sx, sy) = map.get(col, row);
        if sx.is_nan() || sy.is_nan() {
            // Leave as zero (black).
            continue;
        }
        sampled += 1;
        let base = col as usize * c;
        bilinear_all(src, sx, sy, &mut row_data[base..base + c]);
    }
    prof::add(&prof::PX_SAMPLED, sampled);
    prof::add(&prof::TAPS, sampled * c as u64);
}

/// Pick the pyramid level for a single bilinear tap from the warp's local
/// compression footprint `rho = sigma_major` (the **larger** singular value of
/// the local Jacobian — the GL texture-LOD convention: it bounds the source
/// footprint in every direction, so the chosen level never aliases; on
/// anisotropic warps it over-blurs the minor axis, which remains
/// [`remap_aniso_with_pyramid`]'s job to resolve). The level is
/// `l = round(log2(max(rho, 1)))`, clamped to `[0, num_levels - 1]`.
///
/// This deliberately differs from `cluster_refine::level_for_map`, which uses
/// `floor(log2(s_min))` — a sharpness-biased choice that tolerates up to 2×
/// aliasing along the compressed axis and is a pinned kernel contract there.
/// Here the nearest-level rounding on `sigma_major` keeps the residual
/// aliasing bounded by √2× while staying within half an octave of the ideal
/// blur.
///
/// The unclamped level is [`bilinear_mip_level`] of `sigma_major` widened to
/// `f64`, so the sampler rule's prediction of the level and the level a kernel
/// reads are one formula. In `f64` the rounding is exact for every `f32`
/// input; an `f32` `log2` instead rounds up at a few `f32` values just below
/// each boundary `2^(k + ½)`, where its result rounds to `k + ½`.
///
/// A NaN `sigma_major` (degenerate local Jacobian) resolves to level 0, and an
/// infinite one to the top level.
///
/// `pub(crate)` so view selection's affine fast path selects its level by the
/// **same** rule the per-pixel slow path applies, evaluated once on the affine
/// map's constant linear part (see
/// [`affine_core_map`](crate::patch::view_selection)).
#[inline]
pub(crate) fn mip_level_for_sigma(sigma_major: f32, num_levels: usize) -> usize {
    (bilinear_mip_level(f64::from(sigma_major)) as usize).min(num_levels - 1)
}

/// Apply a warp map with a single bilinear sample from the nearest mip level.
///
/// The middle point between [`remap_bilinear`] (plain bilinear from the
/// full-resolution image — aliases when the warp compresses the source, e.g.
/// cross-scale views where one camera is much closer) and
/// [`remap_aniso_with_pyramid`] (Jacobian-SVD mip selection + multi-tap along
/// the major axis — de-aliases; about the same cost with its AVX2 kernel,
/// 1.8–4× on the scalar path): pick the closest pyramid
/// level from the warp's local compression and take a single bilinear sample
/// there — aliasing bounded, cost ≈ bilinear.
///
/// For each output pixel, reads the precomputed SVD of the local Jacobian and
/// selects the pyramid level from `sigma_major` (see `mip_level_for_sigma`,
/// including how this differs from `cluster_refine::level_for_map`). The
/// full-resolution source coordinate `(x, y)` maps to `(x / 2^l, y / 2^l)` at
/// level `l` under the pixel-center-at-0.5 convention (matching
/// [`ImageU8Pyramid`]'s 2×2 box downsample). NaN map entries stay black,
/// exactly like [`remap_bilinear`]. On a non-compressive map every pixel
/// selects level 0, so the output is bit-identical to [`remap_bilinear`].
///
/// Requires [`WarpMap::compute_svd`] to have been called first (panics if not).
pub fn remap_bilinear_mip(pyramid: &ImageU8Pyramid, map: &WarpMap) -> ImageU8 {
    assert!(
        map.has_svd(),
        "remap_bilinear_mip requires WarpMap SVD data; call compute_svd() first"
    );

    let c = pyramid.level(0).channels();
    let data = remap_rows(map.width(), map.height(), c, |row, row_data| {
        bilinear_mip_row(pyramid, map, row, row_data);
    });
    count_call(map);
    ImageU8::new(map.width(), map.height(), c, data)
}

/// One output row of [`remap_bilinear_mip`] from any pyramid, into
/// `row_data` (zeroed beforehand).
pub(crate) fn bilinear_mip_row<P: PyramidTexels>(
    pyramid: &P,
    map: &WarpMap,
    row: u32,
    row_data: &mut [u8],
) {
    let c = pyramid.channels();
    let num_levels = pyramid.num_levels();
    let mut sampled = 0u64;
    for col in 0..map.width() {
        let (sx, sy) = map.get(col, row);
        if sx.is_nan() || sy.is_nan() {
            // Leave as zero (black).
            continue;
        }
        sampled += 1;
        let (sigma_major, _, _, _) = map.get_svd(col, row);
        let level = mip_level_for_sigma(sigma_major, num_levels);
        let scale = (1u32 << level) as f32;
        let base = col as usize * c;
        bilinear_all(
            &pyramid.texels(level),
            sx / scale,
            sy / scale,
            &mut row_data[base..base + c],
        );
    }
    prof::add(&prof::PX_SAMPLED, sampled);
    prof::add(&prof::TAPS, sampled * c as u64);
}

/// Apply a warp map with anisotropic filtering.
///
/// Requires [`WarpMap::compute_svd`] to have been called first (panics if not).
///
/// Builds a Gaussian pyramid of the source image. For each output pixel,
/// reads the precomputed SVD of the local Jacobian, selects the pyramid level
/// from the minor singular value, and takes multiple trilinearly-blended
/// samples along the major axis: the direction in the source image along
/// which the pixel's footprint is longest
/// ([`WarpMapSvd::source_major_dir`](crate::camera::warp_map::WarpMapSvd::source_major_dir)),
/// over the footprint's length `sigma_major`. Falls back to a single bilinear
/// sample when the mapping is non-compressive (`sigma_major <= 1`).
///
/// `max_anisotropy` caps the number of samples along the major axis.
pub fn remap_aniso(src: &ImageU8, map: &WarpMap, max_anisotropy: u32) -> ImageU8 {
    let pyramid = ImageU8Pyramid::build(src, ImageU8Pyramid::full_levels(src.width, src.height));
    remap_aniso_with_pyramid(&pyramid, map, max_anisotropy)
}

/// Like [`remap_aniso`], but resamples from a prebuilt [`ImageU8Pyramid`].
///
/// Use this to warp a single source image through many different maps (e.g.
/// many small per-keypoint patches) without rebuilding the Gaussian pyramid on
/// every call. Requires [`WarpMap::compute_svd`] to have been called first.
///
/// On an x86-64 CPU with AVX2 the rows run through an AVX2 kernel that
/// samples eight output pixels at once wherever they read the same pyramid
/// levels (`aniso_avx2`), and through the scalar per-pixel path elsewhere. The
/// two do the same `f32` operations in the same order, so the output is
/// identical, bit for bit, either way.
pub fn remap_aniso_with_pyramid(
    pyramid: &ImageU8Pyramid,
    map: &WarpMap,
    max_anisotropy: u32,
) -> ImageU8 {
    remap_aniso_dispatch(pyramid, map, max_anisotropy, true).0
}

/// [`remap_aniso_with_pyramid`], using the AVX2 kernel only where `simd` is
/// true and the CPU has it, so a test can run the scalar path on any CPU.
/// Returns the image and the number of eight-pixel groups the AVX2 kernel
/// rendered, so a test can tell that the kernel ran.
pub(crate) fn remap_aniso_dispatch(
    pyramid: &ImageU8Pyramid,
    map: &WarpMap,
    max_anisotropy: u32,
    simd: bool,
) -> (ImageU8, u64) {
    assert!(
        map.has_svd(),
        "remap_aniso requires WarpMap SVD data; call compute_svd() first"
    );

    let out_w = map.width();
    let out_h = map.height();
    let c = pyramid.level(0).channels();
    let simd = simd && aniso_avx2::available() && (1..=4).contains(&c);
    let simd_groups = std::sync::atomic::AtomicU64::new(0);

    let data = remap_rows(out_w, out_h, c, |row, row_data| {
        let mut tally = AnisoTally::default();
        if simd {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `aniso_avx2::available()` checked the CPU features, and
            // the channel count is in 1..=4.
            unsafe {
                aniso_avx2::fill_row(pyramid, map, row, row_data, max_anisotropy, &mut tally)
            };
        } else {
            for col in 0..out_w {
                aniso_pixel(pyramid, map, col, row, max_anisotropy, row_data, &mut tally);
            }
        }
        tally.count();
        simd_groups.fetch_add(tally.simd_groups, std::sync::atomic::Ordering::Relaxed);
    });

    prof::add(&prof::CALLS, 1);
    prof::add(&prof::PX_TOTAL, out_w as u64 * out_h as u64);

    let image = ImageU8 {
        width: out_w,
        height: out_h,
        channels: c,
        data,
    };
    (image, simd_groups.into_inner())
}

/// What one row of an anisotropic remap did, for the `SFMTOOL_PROFILE`
/// counters.
#[derive(Default)]
pub(crate) struct AnisoTally {
    pub(crate) sampled: u64,
    pub(crate) fast: u64,
    pub(crate) multi: u64,
    pub(crate) sum_n: u64,
    pub(crate) taps: u64,
    /// Eight-pixel groups the AVX2 kernel rendered.
    pub(crate) simd_groups: u64,
}

impl AnisoTally {
    /// Add the row's counts to the `SFMTOOL_PROFILE` counters.
    pub(crate) fn count(&self) {
        prof::add(&prof::PX_SAMPLED, self.sampled);
        prof::add(&prof::TAPS, self.taps);
        prof::add(&prof::ANISO_FAST, self.fast);
        prof::add(&prof::ANISO_MULTI, self.multi);
        prof::add(&prof::ANISO_SUM_N, self.sum_n);
    }
}

/// The footprint walk of one output pixel: which pyramid levels it reads,
/// how it blends them, and how many samples it takes along the major axis.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct AnisoFootprint {
    pub(crate) level_lo: usize,
    pub(crate) level_hi: usize,
    pub(crate) frac: f32,
    pub(crate) n: u32,
}

/// The footprint of a pixel whose Jacobian has the singular values
/// `sigma_major > 1` and `sigma_minor`: the level from `sigma_minor`, blended
/// with the next one by its fractional part, and `ceil` of the anisotropy
/// samples, capped at `max_anisotropy`.
#[inline]
pub(crate) fn aniso_footprint(
    sigma_major: f32,
    sigma_minor: f32,
    num_levels: usize,
    max_anisotropy: u32,
) -> AnisoFootprint {
    let level_f = sigma_minor.max(1.0_f32).log2();
    let level_lo = (level_f.floor() as usize).min(num_levels - 1);
    let level_hi = (level_lo + 1).min(num_levels - 1);
    let frac = if level_lo == level_hi {
        0.0
    } else {
        level_f - level_lo as f32
    };
    let ratio = sigma_major / sigma_minor.max(1.0);
    let n = (ratio.ceil() as u32).clamp(1, max_anisotropy);
    AnisoFootprint {
        level_lo,
        level_hi,
        frac,
        n,
    }
}

/// One output pixel of [`remap_aniso_with_pyramid`], written to its channels
/// of `row_data`: the scalar path, and the AVX2 kernel's fallback for pixels
/// it does not take eight at a time.
///
/// For each channel the samples along the major axis are summed in order, at
/// each level, and the two levels blended by `frac`. The corner geometry of a
/// sample is computed once and shared by its channels, with the same products
/// and sums as [`sample_bilinear_u8`] per channel.
#[inline]
pub(crate) fn aniso_pixel<P: PyramidTexels>(
    pyramid: &P,
    map: &WarpMap,
    col: u32,
    row: u32,
    max_anisotropy: u32,
    row_data: &mut [u8],
    tally: &mut AnisoTally,
) {
    let (sx, sy) = map.get(col, row);
    if sx.is_nan() || sy.is_nan() {
        return;
    }
    tally.sampled += 1;
    let c = pyramid.channels();
    let base = col as usize * c;
    aniso_sample(
        pyramid,
        sx,
        sy,
        map.get_svd(col, row),
        max_anisotropy,
        &mut row_data[base..base + c],
        tally,
    );
}

/// The anisotropic sample at the level-0 source position `(sx, sy)` of a
/// pixel whose Jacobian has the SVD `svd` (`(σ_major, σ_minor, major_dx,
/// major_dy)`, as [`WarpMap::get_svd`] returns it), written to `out`, one
/// byte per channel.
///
/// `(major_dx, major_dy)` is the unit direction **in the source image** along
/// which the footprint is longest, the Jacobian's left singular vector `u`:
/// the samples lie at `(sx, sy) + t·σ_major·u` for `t` spread evenly over
/// `[−½, ½]`, which is the source position moved by `t` destination pixels
/// along the right singular vector `v`, since `J·v = σ_major·u`.
///
/// The body of [`aniso_pixel`], shared with view selection's affine fast
/// path, which has no warp map and passes the SVD of its constant Jacobian:
/// given the same position and SVD the two write the same bytes. Takes any
/// channel count; the channels are summed four at a time.
#[inline]
pub(crate) fn aniso_sample<P: PyramidTexels>(
    pyramid: &P,
    sx: f32,
    sy: f32,
    svd: (f32, f32, f32, f32),
    max_anisotropy: u32,
    out: &mut [u8],
    tally: &mut AnisoTally,
) {
    let c = pyramid.channels();
    let (sigma_major, sigma_minor, _, _) = svd;

    // Non-compressive case: single bilinear sample from level 0.
    if sigma_major <= 1.0 {
        tally.fast += 1;
        tally.taps += c as u64;
        bilinear_all(&pyramid.texels(0), sx, sy, out);
        return;
    }

    let fp = aniso_footprint(
        sigma_major,
        sigma_minor,
        pyramid.num_levels(),
        max_anisotropy,
    );
    let n = fp.n;
    let scale_lo = (1u32 << fp.level_lo) as f32;
    let scale_hi = (1u32 << fp.level_hi) as f32;
    // `frac == 0` exactly whenever `sigma_minor <= 1` (the common
    // stretched-but-not-minified case): the hi-level taps would be
    // multiplied by zero, so skip computing them entirely.
    let need_hi = fp.frac > 0.0;

    tally.multi += 1;
    tally.sum_n += n as u64;
    tally.taps += c as u64 * n as u64 * if need_hi { 2 } else { 1 };

    let lo = pyramid.texels(fp.level_lo);
    let hi = pyramid.texels(fp.level_hi);
    // Four channels per walk along the major axis: one walk for the 1 to 4
    // channels of a photograph, and as many as it takes for more. Each
    // channel's sum is the same whichever walk carries it.
    for first in (0..c).step_by(4) {
        let mut sum_lo = [0.0f32; 4];
        let mut sum_hi = [0.0f32; 4];
        for i in 0..n {
            let (sample_x, sample_y) = aniso_tap(sx, sy, svd, i, n);
            accumulate_bilinear(
                &lo,
                sample_x / scale_lo,
                sample_y / scale_lo,
                first,
                &mut sum_lo,
            );
            if need_hi {
                accumulate_bilinear(
                    &hi,
                    sample_x / scale_hi,
                    sample_y / scale_hi,
                    first,
                    &mut sum_hi,
                );
            }
        }
        let slots = &mut out[first..c.min(first + 4)];
        for (ch, slot) in slots.iter_mut().enumerate() {
            let avg_lo = sum_lo[ch] / n as f32;
            let avg_hi = sum_hi[ch] / n as f32;
            let val = avg_lo * (1.0 - fp.frac) + avg_hi * fp.frac;
            *slot = (val + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Add the bilinear sample of channels `first..first + 4` of `img` (those it
/// has) at `(x, y)` to `sums`: [`sample_bilinear_u8`]'s value per channel, its
/// corner geometry computed once.
#[inline]
fn accumulate_bilinear<T: LevelTexels>(
    level: &T,
    x: f32,
    y: f32,
    first: usize,
    sums: &mut [f32; 4],
) {
    let (x0, y0, fx, fy) = bilinear_cell(x, y);
    let c = level.channels();
    let count = c.saturating_sub(first).min(4);
    level.with_cell(x0, y0, |data, idx| {
        for (k, sum) in sums.iter_mut().take(count).enumerate() {
            let ch = first + k;
            let v00 = data[idx[0] + ch] as f32;
            let v10 = data[idx[1] + ch] as f32;
            let v01 = data[idx[2] + ch] as f32;
            let v11 = data[idx[3] + ch] as f32;
            // The grouping of `sample_bilinear_u8`.
            *sum += (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11;
        }
    });
}

/// The level-0 position of sample `i` of the `n` an anisotropic footprint
/// takes along its major axis: `(sx, sy) + t·σ_major·u` with
/// `t = (i + 0.5) / n − 0.5`. The value, gradient and region walks all take
/// their positions from it, and the AVX2 kernel does the same operations.
#[inline]
pub(crate) fn aniso_tap(sx: f32, sy: f32, svd: (f32, f32, f32, f32), i: u32, n: u32) -> (f32, f32) {
    let (sigma_major, _, major_dx, major_dy) = svd;
    let t = (i as f32 + 0.5) / n as f32 - 0.5;
    (
        sx + t * sigma_major * major_dx,
        sy + t * sigma_major * major_dy,
    )
}

/// Per-pixel value+gradient anisotropic sample, mirroring the LOD selection /
/// footprint walk of [`remap_aniso_with_pyramid`]. Returns
/// `(value, dI_dx, dI_dy)` where the gradient is expressed in **level-0**
/// source-pixel coords — each per-level bilinear gradient is **divided** by the
/// level's `2^level` (i.e. multiplied by `1/2^level`) before being blended with
/// the same `frac` the value uses, so `x_level = x_0 / 2^level` gives
/// `∂I/∂x_0 = (∂I/∂x_level) / 2^level`.
///
/// `(sx, sy)` is the level-0 source coordinate; `(sigma_major, sigma_minor,
/// major_dx, major_dy)` comes from `WarpMap::get_svd`, the major direction a
/// unit vector in source pixels as `aniso_sample` walks it. Caller is
/// responsible for the NaN-guard on `(sx, sy)`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn sample_aniso_with_grad<P: PyramidTexels>(
    pyramid: &P,
    sx: f32,
    sy: f32,
    sigma_major: f32,
    sigma_minor: f32,
    major_dx: f32,
    major_dy: f32,
    max_anisotropy: u32,
    channel: u32,
) -> (f32, f32, f32) {
    let num_levels = pyramid.num_levels();

    // Non-compressive case: single bilinear sample from level 0 (no scaling).
    if sigma_major <= 1.0 {
        return bilinear_with_grad(&pyramid.texels(0), sx, sy, channel);
    }

    // The same footprint the value-only path walks.
    let AnisoFootprint {
        level_lo,
        level_hi,
        frac,
        n,
    } = aniso_footprint(sigma_major, sigma_minor, num_levels, max_anisotropy);

    let scale_lo = (1u32 << level_lo) as f32;
    let scale_hi = (1u32 << level_hi) as f32;
    let need_hi = frac > 0.0;

    let mut sum_lo = 0.0f32;
    let mut sum_hi = 0.0f32;
    let mut sum_gx_lo = 0.0f32;
    let mut sum_gy_lo = 0.0f32;
    let mut sum_gx_hi = 0.0f32;
    let mut sum_gy_hi = 0.0f32;

    let svd = (sigma_major, sigma_minor, major_dx, major_dy);
    let lo = pyramid.texels(level_lo);
    let hi = pyramid.texels(level_hi);
    for i in 0..n {
        let (sample_x, sample_y) = aniso_tap(sx, sy, svd, i, n);

        let (v_lo, gx_lo, gy_lo) =
            bilinear_with_grad(&lo, sample_x / scale_lo, sample_y / scale_lo, channel);
        sum_lo += v_lo;
        // Per-level bilinear gradient is in that level's pixel coords; the
        // change of variables `x_level = x_0 / scale` gives
        // `∂I/∂x_0 = (∂I/∂x_level) / scale`.
        sum_gx_lo += gx_lo / scale_lo;
        sum_gy_lo += gy_lo / scale_lo;

        if need_hi {
            let (v_hi, gx_hi, gy_hi) =
                bilinear_with_grad(&hi, sample_x / scale_hi, sample_y / scale_hi, channel);
            sum_hi += v_hi;
            sum_gx_hi += gx_hi / scale_hi;
            sum_gy_hi += gy_hi / scale_hi;
        }
    }

    let avg_lo = sum_lo / n as f32;
    let avg_hi = sum_hi / n as f32;
    let g_lo_x = sum_gx_lo / n as f32;
    let g_lo_y = sum_gy_lo / n as f32;
    let g_hi_x = sum_gx_hi / n as f32;
    let g_hi_y = sum_gy_hi / n as f32;

    let val = avg_lo * (1.0 - frac) + avg_hi * frac;
    let di_dx = g_lo_x * (1.0 - frac) + g_hi_x * frac;
    let di_dy = g_lo_y * (1.0 - frac) + g_hi_y * frac;
    (val, di_dx, di_dy)
}

/// Walk `out_h` rows of an `ImageF32WithGrad`-shaped output, invoking
/// `fill_row(row, value_row, grad_x_row, grad_y_row)` to fill the per-row stride
/// slices of each of the three buffers. Switches to a per-row rayon pass per
/// [`parallelize_rows`]. Factored from the otherwise-duplicated
/// scaffolding in [`remap_bilinear_with_grad_into`] /
/// [`remap_aniso_with_grad_into`].
fn remap_rows_f32x3_into(
    out_w: u32,
    out_h: u32,
    channels: u32,
    value: &mut [f32],
    grad_x: &mut [f32],
    grad_y: &mut [f32],
    fill_row: impl Fn(u32, &mut [f32], &mut [f32], &mut [f32]) + Sync,
) {
    let stride = out_w as usize * channels as usize;
    let total = stride * out_h as usize;
    debug_assert_eq!(value.len(), total);
    debug_assert_eq!(grad_x.len(), total);
    debug_assert_eq!(grad_y.len(), total);
    if !parallelize_rows((out_w as usize) * (out_h as usize), REMAP_PAR_MIN_PIXELS) {
        for row in 0..out_h {
            let off = row as usize * stride;
            fill_row(
                row,
                &mut value[off..off + stride],
                &mut grad_x[off..off + stride],
                &mut grad_y[off..off + stride],
            );
        }
    } else {
        // Parallel: rayon needs disjoint row slices. `par_chunks_exact_mut` over
        // each of the three buffers gives independent row views; `zip` them and
        // walk in parallel. The closure borrows `fill_row` by reference (Sync).
        // `IndexedParallelIterator` (for `.enumerate()` and `.zip()`) is in
        // scope via the `rayon::prelude::*` import at the top of the module.
        value
            .par_chunks_exact_mut(stride)
            .zip(grad_x.par_chunks_exact_mut(stride))
            .zip(grad_y.par_chunks_exact_mut(stride))
            .enumerate()
            .for_each(|(row, ((val_row, gx_row), gy_row))| {
                fill_row(row as u32, val_row, gx_row, gy_row);
            });
    }
}

/// Like [`remap_aniso_with_pyramid`], but additionally returns the analytic
/// image gradient `(∂I/∂x, ∂I/∂y)` in source-pixel coords per output pixel and
/// channel. Value and gradient are computed at the same LOD(s) / footprint
/// (see `sample_aniso_with_grad`).
///
/// Owning version that allocates a fresh [`ImageF32WithGrad`] each call. For
/// repeated rendering (e.g. the photometric refiner's per-GN-step gradient
/// build) prefer [`remap_aniso_with_grad_into`] with a reused scratch.
///
/// Requires [`WarpMap::compute_svd`] to have been called first.
pub fn remap_aniso_with_grad(
    pyramid: &ImageU8Pyramid,
    map: &WarpMap,
    max_anisotropy: u32,
) -> ImageF32WithGrad {
    let mut out = ImageF32WithGrad::empty();
    remap_aniso_with_grad_into(pyramid, map, max_anisotropy, &mut out);
    out
}

/// [`remap_aniso_with_grad`] writing into an `out` scratch, resized in place to
/// fit the warp's dimensions. Reuse `out` across calls to avoid the per-call
/// `~3·W·H·C·sizeof(f32)` allocation.
///
/// Requires [`WarpMap::compute_svd`] to have been called first.
pub fn remap_aniso_with_grad_into(
    pyramid: &ImageU8Pyramid,
    map: &WarpMap,
    max_anisotropy: u32,
    out: &mut ImageF32WithGrad,
) {
    assert!(
        map.has_svd(),
        "remap_aniso_with_grad requires WarpMap SVD data; call compute_svd() first"
    );
    let out_w = map.width();
    let out_h = map.height();
    let c = pyramid.level(0).channels();
    out.resize(out_w, out_h, c);

    let fill_row = |row: u32, val_row: &mut [f32], gx_row: &mut [f32], gy_row: &mut [f32]| {
        aniso_grad_row(
            pyramid,
            map,
            max_anisotropy,
            row,
            GradRow {
                value: val_row,
                grad_x: gx_row,
                grad_y: gy_row,
            },
        );
    };

    remap_rows_f32x3_into(
        out_w,
        out_h,
        c,
        &mut out.value,
        &mut out.grad_x,
        &mut out.grad_y,
        fill_row,
    );
}

/// One output row of a value and gradient render: the row's slices of the
/// value and of the two gradients.
pub(crate) struct GradRow<'a> {
    pub(crate) value: &'a mut [f32],
    pub(crate) grad_x: &'a mut [f32],
    pub(crate) grad_y: &'a mut [f32],
}

impl GradRow<'_> {
    /// Zero the `c` channels of the pixel at `base`, as a NaN map entry is.
    #[inline]
    fn zero(&mut self, base: usize, c: usize) {
        for ch in 0..c {
            self.value[base + ch] = 0.0;
            self.grad_x[base + ch] = 0.0;
            self.grad_y[base + ch] = 0.0;
        }
    }
}

/// One output row of [`remap_aniso_with_grad_into`] from any pyramid.
pub(crate) fn aniso_grad_row<P: PyramidTexels>(
    pyramid: &P,
    map: &WarpMap,
    max_anisotropy: u32,
    row: u32,
    out: GradRow<'_>,
) {
    let mut out = out;
    let c = pyramid.channels();
    for col in 0..map.width() {
        let (sx, sy) = map.get(col, row);
        let base = col as usize * c;
        if sx.is_nan() || sy.is_nan() {
            out.zero(base, c);
            continue;
        }
        let (sigma_major, sigma_minor, major_dx, major_dy) = map.get_svd(col, row);
        for ch in 0..c {
            let (v, gx, gy) = sample_aniso_with_grad(
                pyramid,
                sx,
                sy,
                sigma_major,
                sigma_minor,
                major_dx,
                major_dy,
                max_anisotropy,
                ch as u32,
            );
            out.value[base + ch] = v;
            out.grad_x[base + ch] = gx;
            out.grad_y[base + ch] = gy;
        }
    }
}

/// Float bilinear remap that additionally returns the analytic image gradient
/// `(∂I/∂x, ∂I/∂y)` per output pixel and channel. Mirrors [`remap_bilinear`] in
/// layout/looping; the per-pixel sampler is [`sample_bilinear_with_grad_u8`].
///
/// Owning version. For repeated rendering prefer [`remap_bilinear_with_grad_into`].
pub fn remap_bilinear_with_grad(src: &ImageU8, map: &WarpMap) -> ImageF32WithGrad {
    let mut out = ImageF32WithGrad::empty();
    remap_bilinear_with_grad_into(src, map, &mut out);
    out
}

/// [`remap_bilinear_with_grad`] writing into an `out` scratch, resized in place.
pub fn remap_bilinear_with_grad_into(src: &ImageU8, map: &WarpMap, out: &mut ImageF32WithGrad) {
    remap_bilinear_with_grad_held_into(&HeldLevel::whole(src), map, out);
}

/// [`remap_bilinear_with_grad_into`] of a held level: level 0 of a pyramid,
/// whole or a region of it.
pub(crate) fn remap_bilinear_with_grad_held_into(
    src: &HeldLevel<'_>,
    map: &WarpMap,
    out: &mut ImageF32WithGrad,
) {
    let out_w = map.width();
    let out_h = map.height();
    let c = src.image().channels();
    out.resize(out_w, out_h, c);

    let fill_row = |row: u32, val_row: &mut [f32], gx_row: &mut [f32], gy_row: &mut [f32]| {
        bilinear_grad_row(
            src,
            map,
            row,
            GradRow {
                value: val_row,
                grad_x: gx_row,
                grad_y: gy_row,
            },
        );
    };

    remap_rows_f32x3_into(
        out_w,
        out_h,
        c,
        &mut out.value,
        &mut out.grad_x,
        &mut out.grad_y,
        fill_row,
    );

    prof::add(&prof::CALLS, 1);
    prof::add(&prof::PX_TOTAL, out_w as u64 * out_h as u64);
}

/// One output row of [`remap_bilinear_with_grad_into`] from any level.
pub(crate) fn bilinear_grad_row<T: LevelTexels>(
    src: &T,
    map: &WarpMap,
    row: u32,
    out: GradRow<'_>,
) {
    let mut out = out;
    let c = src.channels();
    let mut sampled = 0u64;
    for col in 0..map.width() {
        let (sx, sy) = map.get(col, row);
        let base = col as usize * c;
        if sx.is_nan() || sy.is_nan() {
            out.zero(base, c);
            continue;
        }
        sampled += 1;
        let end = base + c;
        sample_bilinear_with_grad_u8_all(
            src,
            sx,
            sy,
            &mut out.value[base..end],
            &mut out.grad_x[base..end],
            &mut out.grad_y[base..end],
        );
    }
    prof::add(&prof::PX_SAMPLED, sampled);
    prof::add(&prof::TAPS, sampled * c as u64);
}

/// Like [`remap_bilinear_mip`], but additionally returns the analytic image
/// gradient `(∂I/∂x, ∂I/∂y)` in **full-resolution** source-pixel coords per
/// output pixel and channel. Value and gradient are computed at the same mip
/// level: a bilinear gradient at level `l` is per *level*-pixel, and the change
/// of variables `x_l = x_0 / 2^l` gives `∂I/∂x_0 = (∂I/∂x_l) / 2^l` — the same
/// convention as [`remap_aniso_with_grad`].
///
/// Owning version. For repeated rendering prefer
/// [`remap_bilinear_mip_with_grad_into`].
///
/// Requires [`WarpMap::compute_svd`] to have been called first.
pub fn remap_bilinear_mip_with_grad(pyramid: &ImageU8Pyramid, map: &WarpMap) -> ImageF32WithGrad {
    let mut out = ImageF32WithGrad::empty();
    remap_bilinear_mip_with_grad_into(pyramid, map, &mut out);
    out
}

/// [`remap_bilinear_mip_with_grad`] writing into an `out` scratch, resized in
/// place. Level selection is per pixel from the precomputed SVD (see
/// `mip_level_for_sigma`, including how it differs from
/// `cluster_refine::level_for_map`); NaN map entries write zero value and
/// gradient, exactly like [`remap_bilinear_with_grad_into`].
///
/// Requires [`WarpMap::compute_svd`] to have been called first.
pub fn remap_bilinear_mip_with_grad_into(
    pyramid: &ImageU8Pyramid,
    map: &WarpMap,
    out: &mut ImageF32WithGrad,
) {
    assert!(
        map.has_svd(),
        "remap_bilinear_mip_with_grad requires WarpMap SVD data; call compute_svd() first"
    );
    let out_w = map.width();
    let out_h = map.height();
    let c = pyramid.level(0).channels();
    out.resize(out_w, out_h, c);

    let fill_row = |row: u32, val_row: &mut [f32], gx_row: &mut [f32], gy_row: &mut [f32]| {
        bilinear_mip_grad_row(
            pyramid,
            map,
            row,
            GradRow {
                value: val_row,
                grad_x: gx_row,
                grad_y: gy_row,
            },
        );
    };

    remap_rows_f32x3_into(
        out_w,
        out_h,
        c,
        &mut out.value,
        &mut out.grad_x,
        &mut out.grad_y,
        fill_row,
    );

    prof::add(&prof::CALLS, 1);
    prof::add(&prof::PX_TOTAL, out_w as u64 * out_h as u64);
}

/// One output row of [`remap_bilinear_mip_with_grad_into`] from any pyramid.
pub(crate) fn bilinear_mip_grad_row<P: PyramidTexels>(
    pyramid: &P,
    map: &WarpMap,
    row: u32,
    out: GradRow<'_>,
) {
    let mut out = out;
    let c = pyramid.channels();
    let num_levels = pyramid.num_levels();
    let mut sampled = 0u64;
    for col in 0..map.width() {
        let (sx, sy) = map.get(col, row);
        let base = col as usize * c;
        if sx.is_nan() || sy.is_nan() {
            out.zero(base, c);
            continue;
        }
        sampled += 1;
        let (sigma_major, _, _, _) = map.get_svd(col, row);
        let level = mip_level_for_sigma(sigma_major, num_levels);
        let scale = (1u32 << level) as f32;
        let end = base + c;
        sample_bilinear_with_grad_u8_all(
            &pyramid.texels(level),
            sx / scale,
            sy / scale,
            &mut out.value[base..end],
            &mut out.grad_x[base..end],
            &mut out.grad_y[base..end],
        );
        // Per-level bilinear gradient is in that level's pixel coords;
        // rescale to full-res source-pixel coords (`/ scale`). Level 0 is
        // untouched (division by 1 is exact), keeping the level-0 output
        // bit-identical to `remap_bilinear_with_grad_into`.
        if level > 0 {
            for k in base..end {
                out.grad_x[k] /= scale;
                out.grad_y[k] /= scale;
            }
        }
    }
    prof::add(&prof::PX_SAMPLED, sampled);
    prof::add(&prof::TAPS, sampled * c as u64);
}

#[cfg(test)]
mod tests;
