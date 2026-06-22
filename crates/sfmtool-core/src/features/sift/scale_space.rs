// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Gaussian scale-space and Difference-of-Gaussians (DoG) pyramids.
//!
//! Stages 1 and 2 of SIFT (`specs/core/sift.md`). The Gaussian pyramid is the
//! shared, expensive artifact that detection, orientation, and description all
//! read:
//!
//! - The **Gaussian pyramid**: `octave_layers + 3` Gaussian-blurred images per
//!   octave, with absolute blur `σ · k^i` at level `i` (`k = 2^(1/s)`). The input
//!   is optionally doubled (bilinear) before octave 0; the next octave is seeded
//!   by decimating the Gaussian image `octave_layers` levels up.
//! - The **DoG** (`octave_layers + 2` adjacent-Gaussian differences per octave)
//!   is **not stored**. It is needed only for detection, which fuses it per row
//!   stripe in cache (`dog(l) = gaussian(l+1) − gaussian(l)`, computed on the
//!   fly; see `detect.rs` and `specs/core/sift.md`). Only the Gaussian pyramid
//!   is retained.
//!
//! Gradient **magnitude** and **orientation** are likewise *not* stored; the
//! orientation and descriptor stages sample gradients on the fly from the
//! Gaussian pyramid via [`pixel_gradient`], which avoids computing (and holding)
//! a full magnitude/orientation map for every level when only small per-keypoint
//! windows are ever read.
//!
//! Conventions follow the optical-flow module: [`GrayImage`], separable Gaussian
//! blur with an SSE2 inner pass plus scalar fallback, and rayon row parallelism.
//!
//! # Coordinate and sigma mapping
//!
//! Octave 0 operates on the (optionally doubled) base image. With doubling on,
//! the base image has half the pixel spacing of the full-resolution input, so a
//! step of one octave-0 pixel is 0.5 full-resolution pixels; with doubling off it
//! is 1.0. Octave `o` decimates by `2^o` relative to octave 0. The pixel step of
//! octave `o` in full-resolution units is therefore
//! `octave_pixel_step(o) = 2^o · base_step`, where `base_step = 0.5` when doubling
//! is on and `1.0` otherwise (see [`ScaleSpace::octave_pixel_step`]).
//!
//! Mapping an octave-`o` pixel `(xo, yo)` (pixel-center convention) to
//! full-resolution coordinates (also pixel-center) is
//! `x_full = xo · step + c`, where `c = base_step / 2` is a constant offset
//! (see [`ScaleSpace::octave_to_image`]), and
//! the absolute blur at `(octave, layer)` measured in **octave-`o` pixels** is
//! `σ · k^layer`; in full-resolution pixels it is `σ · k^layer · step`
//! (see [`ScaleSpace::abs_sigma`] / [`ScaleSpace::abs_sigma_full`]).

use super::{GrayImage, SiftParams};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Cached `avx2 && fma` runtime support is shared with the orientation/descriptor
// SIMD path; see `super::simd::HAS_AVX2_FMA`. When set, the blur interior loops
// use the 8-wide AVX2 + FMA path, else the 4-wide SSE2 path. (FMA fuses each tap
// into one rounded op, so the AVX2 output differs from SSE2 in the last ULP —
// both are valid Gaussian blurs.) `SFMTOOL_SIFT_NO_AVX2` forces the SSE2 path.
#[cfg(target_arch = "x86_64")]
use super::simd::HAS_AVX2_FMA;

/// A single octave's worth of images.
struct Octave {
    /// Width of every image in this octave.
    width: u32,
    /// Height of every image in this octave.
    height: u32,
    /// `octave_layers + 3` Gaussian-blurred images, level 0 first.
    gaussians: Vec<GrayImage>,
}

/// The Gaussian scale-space pyramid for an image (the DoG is computed on the
/// fly during detection, not stored — see the module docs).
///
/// Built by [`ScaleSpace::build`]. See the module docs for the coordinate and
/// sigma conventions.
pub struct ScaleSpace {
    octaves: Vec<Octave>,
    /// Number of intervals `s` per octave (`octave_layers`).
    s: u32,
    /// Base blur `σ`.
    sigma: f64,
    /// Whether the input was doubled before octave 0.
    double_image: bool,
}

impl ScaleSpace {
    /// Build the Gaussian pyramid for `image` (the DoG is fused into detection,
    /// not materialized here — see the module docs).
    pub fn build(image: &GrayImage, params: &SiftParams) -> Self {
        let s = params.octave_layers;
        let sigma = params.sigma;
        let double_image = params.double_image;
        // Multiplicative scale step between adjacent levels.
        let k = 2.0f64.powf(1.0 / s as f64);
        let gaussians_per_octave = (s + 3) as usize;

        // Env-gated per-stage timers (see `crate::features::sift::SIFT_TIMING`).
        let timing = *crate::features::sift::SIFT_TIMING;
        let mut t_base = std::time::Duration::ZERO;
        let mut t_blur = std::time::Duration::ZERO;
        let mut t_decimate = std::time::Duration::ZERO;
        let t_base_start = std::time::Instant::now();

        // Scratch buffer for every blur's horizontal-pass intermediate, reused
        // across the whole build (grows to the octave-0 size on first use).
        let mut blur_scratch: Vec<f32> = Vec::new();

        // --- Build octave-0 level 0 (bring the base image to absolute blur σ). ---
        let (base, base_sigma) = if double_image {
            // Doubling halves pixel spacing, so the assumed input blur of
            // `input_sigma` becomes `2 * input_sigma` relative to the new grid.
            let doubled = upsample_2x(image);
            (doubled, 2.0 * params.input_sigma)
        } else {
            (
                GrayImage::new(image.width(), image.height(), image.data().to_vec()),
                params.input_sigma,
            )
        };
        // Incremental blur to lift the base from `base_sigma` to `σ`.
        let level0 = if sigma > base_sigma {
            let inc = (sigma * sigma - base_sigma * base_sigma).sqrt();
            gaussian_blur(&base, inc, params.blur_radius_factor, &mut blur_scratch)
        } else {
            base
        };
        t_base += t_base_start.elapsed();

        // Number of octaves: continue halving until the smaller side would be too
        // small to detect extrema. We require the smallest side to stay >= a few
        // pixels; matching the spec's "floor(log2(min(W,H))) - offset" rule with an
        // offset that keeps min side >= ~8 px (enough for a 3x3x3 extremum test plus
        // a localization border).
        let min_side = level0.width().min(level0.height());
        let num_octaves = if min_side < 8 {
            1
        } else {
            // floor(log2(min_side)) - 3, clamped to at least 1.
            let log2 = (min_side as f64).log2().floor() as i32;
            (log2 - 3).max(1) as u32
        };

        // Precompute the incremental sigmas within an octave (level 1.. relative to
        // level 0). Level i has absolute blur σ·k^i, so the incremental blur from
        // level i-1 to i is σ·sqrt(k^(2i) - k^(2(i-1))).
        let inc_sigmas: Vec<f64> = (1..gaussians_per_octave)
            .map(|i| {
                let prev = k.powi((i - 1) as i32);
                let cur = k.powi(i as i32);
                sigma * (cur * cur - prev * prev).sqrt()
            })
            .collect();

        let mut octaves: Vec<Octave> = Vec::with_capacity(num_octaves as usize);
        let mut current_base = level0;

        for o in 0..num_octaves {
            // Build the Gaussian stack for this octave by incremental blur.
            let tm = std::time::Instant::now();
            let mut gaussians: Vec<GrayImage> = Vec::with_capacity(gaussians_per_octave);
            gaussians.push(current_base);
            for inc in &inc_sigmas {
                let prev = gaussians.last().unwrap();
                gaussians.push(gaussian_blur(
                    prev,
                    *inc,
                    params.blur_radius_factor,
                    &mut blur_scratch,
                ));
            }
            t_blur += tm.elapsed();

            let width = gaussians[0].width();
            let height = gaussians[0].height();

            // The DoG pyramid is no longer materialized: detection fuses it per
            // row-stripe in cache, computing `dog(l) = gaussian(l+1) − gaussian(l)`
            // on the fly (see `detect.rs` and `specs/core/sift.md`). Only the
            // Gaussian stack is retained — it is also what orientation and
            // description sample.

            // Seed the next octave by decimating the Gaussian image `s` levels up
            // (absolute blur 2σ), taking every second pixel.
            let tm = std::time::Instant::now();
            let next_base = if o + 1 < num_octaves {
                Some(decimate_2x(&gaussians[s as usize]))
            } else {
                None
            };
            t_decimate += tm.elapsed();

            octaves.push(Octave {
                width,
                height,
                gaussians,
            });

            if let Some(nb) = next_base {
                current_base = nb;
            } else {
                break;
            }
        }

        if timing {
            eprintln!(
                "SIFT_TIMING build base_ms={:.3} blur_ms={:.3} decimate_ms={:.3}",
                t_base.as_secs_f64() * 1e3,
                t_blur.as_secs_f64() * 1e3,
                t_decimate.as_secs_f64() * 1e3,
            );
        }

        Self {
            octaves,
            s,
            sigma,
            double_image,
        }
    }

    /// Number of octaves in the pyramid.
    pub fn num_octaves(&self) -> usize {
        self.octaves.len()
    }

    /// Number of intervals `s` per octave.
    pub fn octave_layers(&self) -> u32 {
        self.s
    }

    /// Number of Gaussian levels per octave (`s + 3`).
    pub fn gaussians_per_octave(&self) -> usize {
        (self.s + 3) as usize
    }

    /// Number of DoG levels per octave (`s + 2`). The DoG is not stored
    /// (detection fuses it per stripe); this is the count it *would* have.
    pub fn dogs_per_octave(&self) -> usize {
        if self.octaves.is_empty() {
            0
        } else {
            (self.s + 2) as usize
        }
    }

    /// The `(width, height)` of every image in `octave`.
    pub fn octave_dims(&self, octave: usize) -> (u32, u32) {
        let o = &self.octaves[octave];
        (o.width, o.height)
    }

    /// The Gaussian image at `(octave, level)`.
    pub fn gaussian(&self, octave: usize, level: usize) -> &GrayImage {
        &self.octaves[octave].gaussians[level]
    }

    /// The DoG image at `(octave, level)`, computed on demand as
    /// `gaussian(level+1) − gaussian(level)`.
    ///
    /// The DoG pyramid is not stored — detection fuses it per row-stripe in
    /// cache (see `detect.rs`). This convenience materializes one level (an
    /// allocation) and is intended for tests/diagnostics, not the hot path.
    pub fn dog(&self, octave: usize, level: usize) -> GrayImage {
        difference(
            self.gaussian(octave, level + 1),
            self.gaussian(octave, level),
        )
    }

    /// The full-resolution pixel step of one octave-`octave` pixel.
    ///
    /// `2^octave · base_step`, where `base_step = 0.5` when doubling is on (the
    /// base image is at 2x) and `1.0` otherwise.
    pub fn octave_pixel_step(&self, octave: i32) -> f64 {
        let base_step = if self.double_image { 0.5 } else { 1.0 };
        2.0f64.powi(octave) * base_step
    }

    /// Map an octave-`octave` pixel coordinate `(xo, yo)` (pixel-center
    /// convention) to full-resolution image coordinates (also pixel-center,
    /// COLMAP convention where the upper-left pixel center is `(0.5, 0.5)`).
    ///
    /// The pyramid resample is affine: octave `o` index `xo` maps to octave-0
    /// index `2^o · xo` (decimation takes `new[i] = old[2i]`), and octave-0
    /// index `x0` maps to the full-resolution center coordinate `x0 + 0.5`
    /// (no doubling) or `0.5·x0 + 0.25` (doubling). Both collapse to
    /// `x_full = xo · step + c`, with the **constant** offset `c = base_step / 2`
    /// (0.5 without doubling, 0.25 with). The offset must not scale with `step`,
    /// or keypoints drift by half a pixel per octave.
    pub fn octave_to_image(&self, octave: i32, xo: f64, yo: f64) -> (f64, f64) {
        let step = self.octave_pixel_step(octave);
        let c = if self.double_image { 0.25 } else { 0.5 };
        (xo * step + c, yo * step + c)
    }

    /// Map a full-resolution image coordinate `(x, y)` (pixel-center convention)
    /// back to octave-`octave` pixel coordinates (also pixel-center).
    ///
    /// Inverse of [`ScaleSpace::octave_to_image`]: `xo = (x_full − c) / step`.
    pub fn image_to_octave(&self, octave: i32, x: f64, y: f64) -> (f64, f64) {
        let step = self.octave_pixel_step(octave);
        let c = if self.double_image { 0.25 } else { 0.5 };
        ((x - c) / step, (y - c) / step)
    }

    /// The absolute Gaussian blur at `(octave, layer)` measured in octave-`octave`
    /// pixels: `σ · k^layer` with `k = 2^(1/s)`.
    pub fn abs_sigma(&self, layer: f64) -> f64 {
        let k = 2.0f64.powf(1.0 / self.s as f64);
        self.sigma * k.powf(layer)
    }

    /// The absolute Gaussian blur at `(octave, layer)` in full-resolution pixels:
    /// `abs_sigma(layer) · octave_pixel_step(octave)`.
    pub fn abs_sigma_full(&self, octave: i32, layer: f64) -> f64 {
        self.abs_sigma(layer) * self.octave_pixel_step(octave)
    }
}

/// Upsample an image 2x using bilinear interpolation (pixel-center convention).
fn upsample_2x(img: &GrayImage) -> GrayImage {
    let t0 = std::time::Instant::now();
    let in_w = img.width();
    let in_h = img.height();
    let out_w = in_w * 2;
    let out_h = in_h * 2;
    let in_wi = in_w as usize;
    let in_hi = in_h as usize;
    let src = img.data();

    let mut data = vec![0.0f32; (out_w as usize) * (out_h as usize)];
    data.par_chunks_mut(out_w as usize)
        .enumerate()
        .for_each(|(row, out_row)| {
            // Map output pixel center to input coordinates: x_in = (x_out+0.5)/2 - 0.5
            let sy = (row as f32 + 0.5) * 0.5 - 0.5;
            let y0 = sy.floor();
            let fy = sy - y0;
            let y0i = (y0 as i32).clamp(0, in_hi as i32 - 1) as usize;
            let y1i = ((y0 as i32) + 1).clamp(0, in_hi as i32 - 1) as usize;
            for (col, out) in out_row.iter_mut().enumerate() {
                let sx = (col as f32 + 0.5) * 0.5 - 0.5;
                let x0 = sx.floor();
                let fx = sx - x0;
                let x0i = (x0 as i32).clamp(0, in_wi as i32 - 1) as usize;
                let x1i = ((x0 as i32) + 1).clamp(0, in_wi as i32 - 1) as usize;
                let v00 = src[y0i * in_wi + x0i];
                let v10 = src[y0i * in_wi + x1i];
                let v01 = src[y1i * in_wi + x0i];
                let v11 = src[y1i * in_wi + x1i];
                let top = v00 * (1.0 - fx) + v10 * fx;
                let bot = v01 * (1.0 - fx) + v11 * fx;
                *out = top * (1.0 - fy) + bot * fy;
            }
        });

    log_op("upsample", out_w, out_h, 0, t0);
    GrayImage::new(out_w, out_h, data)
}

/// Decimate an image 2x by taking every second pixel in x and y.
fn decimate_2x(img: &GrayImage) -> GrayImage {
    let t0 = std::time::Instant::now();
    let in_w = img.width() as usize;
    let out_w = (img.width() / 2).max(1) as usize;
    let out_h = (img.height() / 2).max(1) as usize;
    let src = img.data();
    let mut data = vec![0.0f32; out_w * out_h];
    for or in 0..out_h {
        let src_row = (2 * or) * in_w;
        let dst_row = or * out_w;
        for oc in 0..out_w {
            data[dst_row + oc] = src[src_row + 2 * oc];
        }
    }
    log_op("decimate", out_w as u32, out_h as u32, 0, t0);
    GrayImage::new(out_w as u32, out_h as u32, data)
}

/// Per-pixel difference `a − b` of two equally-sized images (one DoG level).
fn difference(a: &GrayImage, b: &GrayImage) -> GrayImage {
    debug_assert_eq!(a.width(), b.width());
    debug_assert_eq!(a.height(), b.height());
    let n = a.data().len();
    let mut out = vec![0.0f32; n];
    let ad = a.data();
    let bd = b.data();

    out.par_chunks_mut(4096)
        .enumerate()
        .for_each(|(chunk, dst)| {
            let base = chunk * 4096;
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: SSE2 is baseline on x86_64.
                unsafe {
                    sub_slice_sse2(
                        &ad[base..base + dst.len()],
                        &bd[base..base + dst.len()],
                        dst,
                    )
                };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = ad[base + i] - bd[base + i];
                }
            }
        });

    GrayImage::new(a.width(), a.height(), out)
}

/// SSE2 elementwise subtraction `dst = a − b` for equal-length slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn sub_slice_sse2(a: &[f32], b: &[f32], dst: &mut [f32]) {
    let n = dst.len();
    let mut i = 0;
    while i + 4 <= n {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        _mm_storeu_ps(dst.as_mut_ptr().add(i), _mm_sub_ps(va, vb));
        i += 4;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i);
        i += 1;
    }
}

/// Build a normalized 1D Gaussian kernel for the given sigma.
///
/// The radius is `ceil(radius_factor·sigma)` (default factor 2.25, ~97.6% of the
/// mass); the kernel has `2·radius + 1` taps, sampled at integer offsets and
/// normalized to sum to 1.
fn gaussian_kernel(sigma: f64, radius_factor: f64) -> Vec<f32> {
    let radius = (radius_factor * sigma).ceil().max(1.0) as i32;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f64;
    for t in -radius..=radius {
        let v = (-(t as f64) * (t as f64) * inv_2s2).exp();
        kernel.push(v);
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel.iter().map(|&v| v as f32).collect()
}

/// Emit one `SIFT_OP` detail line for a scale-space operator when `SFMTOOL_SIFT_OPS`
/// is set (see [`crate::features::sift::SIFT_OPS`]). `taps` is the 1D kernel length for
/// blur (0 for operators without a kernel). `px = w · h` is the operated pixel
/// count (output pixels for resampling operators). Zero-cost when the flag is
/// off — only a cached bool check and one `Instant::now()` at each call site.
#[inline]
fn log_op(op: &str, w: u32, h: u32, taps: u32, t0: std::time::Instant) {
    if *crate::features::sift::SIFT_OPS {
        eprintln!(
            "SIFT_OP op={} w={} h={} px={} taps={} ms={:.4}",
            op,
            w,
            h,
            w as u64 * h as u64,
            taps,
            t0.elapsed().as_secs_f64() * 1e3,
        );
    }
}

/// Allocate a `Vec<f32>` of length `n` without zero-initializing it.
///
/// Sound because `f32` has no invalid bit patterns (every 32-bit value is a
/// valid `f32`) and no `Drop`, and the blur passes overwrite every element
/// before any read, so no `undef` is ever observed.
#[inline]
#[allow(clippy::uninit_vec)] // write-before-read is guaranteed by the callers
pub(crate) fn uninit_vec_f32(n: usize) -> Vec<f32> {
    let mut v = Vec::<f32>::with_capacity(n);
    // SAFETY: capacity >= n; see the function doc for the write-before-read and
    // valid-bit-pattern argument that makes the uninitialized contents sound.
    unsafe { v.set_len(n) };
    v
}

/// Resize `buf` to exactly `n` uninitialized elements, reusing its allocation
/// when it is already large enough (no reallocation, no zero-fill). Same
/// soundness argument as [`uninit_vec_f32`].
#[inline]
#[allow(clippy::uninit_vec)] // write-before-read is guaranteed by the callers
fn resize_uninit_f32(buf: &mut Vec<f32>, n: usize) {
    buf.clear();
    buf.reserve(n);
    // SAFETY: capacity >= n after `reserve`; every element is written before read.
    unsafe { buf.set_len(n) };
}

/// Separable Gaussian blur with edge clamping. Horizontal pass then vertical
/// pass; rows are parallelized with rayon and the horizontal inner loop uses an
/// SSE2/AVX2 path with a scalar fallback for borders / non-x86.
///
/// `scratch` holds the horizontal-pass intermediate and is reused across calls
/// (a build threads one buffer through every blur), so the only per-call
/// allocation is the returned image's own pixel buffer.
fn gaussian_blur(
    img: &GrayImage,
    sigma: f64,
    radius_factor: f64,
    scratch: &mut Vec<f32>,
) -> GrayImage {
    let t0 = std::time::Instant::now();
    let kernel = gaussian_kernel(sigma, radius_factor);
    let radius = (kernel.len() / 2) as i32;
    let w = img.width() as usize;
    let h = img.height() as usize;

    // Horizontal pass into the reused `scratch` buffer; `scratch` and `out` are
    // both fully overwritten before being read (the H pass writes every column of
    // every row; the V pass writes every pixel), so neither is zero-filled.
    let src = img.data();
    resize_uninit_f32(scratch, w * h);
    scratch
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(row, out_row)| {
            let row_data = &src[row * w..row * w + w];
            blur_row(row_data, out_row, &kernel, radius);
        });
    // Immutable view of the horizontal-pass result for the vertical pass.
    let horiz: &[f32] = scratch;

    // Vertical pass: treat columns. Process by output row, gathering the
    // vertical neighborhood (clamped) for each pixel. Interior rows (where the
    // full kernel fits without clamping) use an SSE2 path that vectorizes across
    // columns; border rows fall back to the clamped scalar path.
    let mut out = uninit_vec_f32(w * h);
    out.par_chunks_mut(w)
        .enumerate()
        .for_each(|(row, out_row)| {
            #[cfg(target_arch = "x86_64")]
            {
                let radius_us = radius as usize;
                if row >= radius_us && row + radius_us < h {
                    // For interior rows every tap row (row-radius ..= row+radius)
                    // is in bounds, so no clamping is needed.
                    if *HAS_AVX2_FMA {
                        // SAFETY: guarded by runtime avx2+fma detection.
                        unsafe {
                            blur_col_interior_avx2(horiz, out_row, &kernel, radius_us, row, w)
                        };
                    } else {
                        // SAFETY: SSE2 is baseline on x86_64.
                        unsafe {
                            blur_col_interior_sse2(horiz, out_row, &kernel, radius_us, row, w)
                        };
                    }
                    return;
                }
            }
            for (col, dst) in out_row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (kidx, &kw) in kernel.iter().enumerate() {
                    let r = (row as i32 + kidx as i32 - radius).clamp(0, h as i32 - 1) as usize;
                    acc += kw * horiz[r * w + col];
                }
                *dst = acc;
            }
        });

    log_op("blur", img.width(), img.height(), kernel.len() as u32, t0);
    GrayImage::new(img.width(), img.height(), out)
}

/// Convolve one row with the 1D kernel, clamping at the borders.
///
/// The interior (where the full kernel fits) uses an SSE2 path; the borders use
/// a clamped scalar path.
fn blur_row(row: &[f32], out: &mut [f32], kernel: &[f32], radius: i32) {
    let w = row.len() as i32;
    let r = radius;

    // Left border (clamped).
    let left_end = r.min(w);
    for (col, dst) in out.iter_mut().enumerate().take(left_end as usize) {
        let mut acc = 0.0f32;
        for (kidx, &kw) in kernel.iter().enumerate() {
            let c = (col as i32 + kidx as i32 - r).clamp(0, w - 1) as usize;
            acc += kw * row[c];
        }
        *dst = acc;
    }

    // Interior: the whole kernel fits, no clamping needed.
    let interior_start = left_end;
    let interior_end = (w - r).max(interior_start);
    if interior_end > interior_start {
        #[cfg(target_arch = "x86_64")]
        {
            let (s, e) = (interior_start as usize, interior_end as usize);
            if *HAS_AVX2_FMA {
                // SAFETY: guarded by runtime avx2+fma detection.
                unsafe { blur_row_interior_avx2(row, out, kernel, r, s, e) };
            } else {
                // SAFETY: SSE2 is baseline on x86_64.
                unsafe { blur_row_interior_sse2(row, out, kernel, r, s, e) };
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for col in interior_start as usize..interior_end as usize {
                let mut acc = 0.0f32;
                for (kidx, &kw) in kernel.iter().enumerate() {
                    acc += kw * row[col + kidx - r as usize];
                }
                out[col] = acc;
            }
        }
    }

    // Right border (clamped).
    for col in interior_end.max(left_end)..w {
        let mut acc = 0.0f32;
        for (kidx, &kw) in kernel.iter().enumerate() {
            let c = (col + kidx as i32 - r).clamp(0, w - 1) as usize;
            acc += kw * row[c];
        }
        out[col as usize] = acc;
    }
}

/// SSE2 interior convolution: for each tap, broadcast the weight and FMA against
/// 4 contiguous pixels. Operates on `[start, end)` where the full kernel fits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn blur_row_interior_sse2(
    row: &[f32],
    out: &mut [f32],
    kernel: &[f32],
    radius: i32,
    start: usize,
    end: usize,
) {
    let r = radius as usize;
    let mut col = start;
    while col + 4 <= end {
        let mut acc = _mm_setzero_ps();
        for (kidx, &kw) in kernel.iter().enumerate() {
            let kv = _mm_set1_ps(kw);
            // Source pixels at col + kidx - r .. +4.
            let base = col + kidx - r;
            let pv = _mm_loadu_ps(row.as_ptr().add(base));
            acc = _mm_add_ps(acc, _mm_mul_ps(kv, pv));
        }
        _mm_storeu_ps(out.as_mut_ptr().add(col), acc);
        col += 4;
    }
    // Scalar tail within the interior.
    while col < end {
        let mut acc = 0.0f32;
        for (kidx, &kw) in kernel.iter().enumerate() {
            acc += kw * *row.get_unchecked(col + kidx - r);
        }
        *out.get_unchecked_mut(col) = acc;
        col += 1;
    }
}

/// AVX2 + FMA horizontal interior convolution: 8 contiguous pixels per step,
/// one `fmadd` per tap. Same structure as the SSE2 path at twice the width.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn blur_row_interior_avx2(
    row: &[f32],
    out: &mut [f32],
    kernel: &[f32],
    radius: i32,
    start: usize,
    end: usize,
) {
    let r = radius as usize;
    let mut col = start;
    while col + 8 <= end {
        let mut acc = _mm256_setzero_ps();
        for (kidx, &kw) in kernel.iter().enumerate() {
            let kv = _mm256_set1_ps(kw);
            let pv = _mm256_loadu_ps(row.as_ptr().add(col + kidx - r));
            acc = _mm256_fmadd_ps(kv, pv, acc);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(col), acc);
        col += 8;
    }
    // Scalar tail (< 8 columns), FMA-consistent via mul_add.
    while col < end {
        let mut acc = 0.0f32;
        for (kidx, &kw) in kernel.iter().enumerate() {
            acc = kw.mul_add(*row.get_unchecked(col + kidx - r), acc);
        }
        *out.get_unchecked_mut(col) = acc;
        col += 1;
    }
}

/// SSE2 vertical convolution for one interior output row. Accumulates each tap
/// (a contiguous source row at stride `w`) across 4 columns at a time, keeping
/// the accumulator in a register so each output pixel is written once. The
/// caller guarantees `radius <= row` and `row + radius < h`, so every tap row
/// `row - radius ..= row + radius` is in bounds (no clamping needed).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn blur_col_interior_sse2(
    horiz: &[f32],
    out_row: &mut [f32],
    kernel: &[f32],
    radius: usize,
    row: usize,
    w: usize,
) {
    // Flat index of the top tap row (row - radius) at column 0.
    let top = (row - radius) * w;
    let mut col = 0;
    while col + 4 <= w {
        let mut acc = _mm_setzero_ps();
        let mut base = top + col;
        for &kw in kernel.iter() {
            let kv = _mm_set1_ps(kw);
            let pv = _mm_loadu_ps(horiz.as_ptr().add(base));
            acc = _mm_add_ps(acc, _mm_mul_ps(kv, pv));
            base += w;
        }
        _mm_storeu_ps(out_row.as_mut_ptr().add(col), acc);
        col += 4;
    }
    // Scalar tail for the last < 4 columns.
    while col < w {
        let mut acc = 0.0f32;
        let mut base = top + col;
        for &kw in kernel.iter() {
            acc += kw * *horiz.get_unchecked(base);
            base += w;
        }
        *out_row.get_unchecked_mut(col) = acc;
        col += 1;
    }
}

/// AVX2 + FMA vertical convolution for one interior output row. 8 columns per
/// step, one `fmadd` per tap, source rows at stride `w`. Same preconditions as
/// [`blur_col_interior_sse2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn blur_col_interior_avx2(
    horiz: &[f32],
    out_row: &mut [f32],
    kernel: &[f32],
    radius: usize,
    row: usize,
    w: usize,
) {
    let top = (row - radius) * w;
    let mut col = 0;
    while col + 8 <= w {
        let mut acc = _mm256_setzero_ps();
        let mut base = top + col;
        for &kw in kernel.iter() {
            let kv = _mm256_set1_ps(kw);
            let pv = _mm256_loadu_ps(horiz.as_ptr().add(base));
            acc = _mm256_fmadd_ps(kv, pv, acc);
            base += w;
        }
        _mm256_storeu_ps(out_row.as_mut_ptr().add(col), acc);
        col += 8;
    }
    // Scalar tail (< 8 columns), FMA-consistent via mul_add.
    while col < w {
        let mut acc = 0.0f32;
        let mut base = top + col;
        for &kw in kernel.iter() {
            acc = kw.mul_add(*horiz.get_unchecked(base), acc);
            base += w;
        }
        *out_row.get_unchecked_mut(col) = acc;
        col += 1;
    }
}

/// Central-difference gradient `(magnitude, orientation)` at interior pixel
/// `idx` of a row-major image `data` with width `w`:
/// `dx = L[idx+1] − L[idx−1]`, `dy = L[idx+w] − L[idx−w]`,
/// `m = sqrt(dx² + dy²)`, `θ = atan2(dy, dx)` (radians, in `(-π, π]`).
///
/// Gradients are sampled on the fly from the Gaussian pyramid rather than
/// precomputed into per-level maps — detection never needs them, and the
/// orientation/descriptor consumers only read a small window per keypoint.
/// The caller must ensure the pixel is in the interior (`1 ≤ x < w−1`,
/// `1 ≤ y < h−1`) so the four neighbors are in bounds; both consumers already
/// clamp their sample windows to that range, where this is identical to the
/// old edge-clamped maps.
#[inline(always)]
pub(crate) fn pixel_gradient(data: &[f32], w: usize, idx: usize) -> (f32, f32) {
    let dx = data[idx + 1] - data[idx - 1];
    let dy = data[idx + w] - data[idx - w];
    ((dx * dx + dy * dy).sqrt(), dy.atan2(dx))
}

#[cfg(test)]
mod tests;
