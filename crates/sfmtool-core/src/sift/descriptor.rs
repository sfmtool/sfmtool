// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Descriptor computation (stage 6 of SIFT, `specs/core/sift.md`).
//!
//! In a window rotated to the keypoint orientation, sample gradients over the
//! pixels covered by a `d × d` (4×4) grid of subregions. Each sample is
//! accumulated into a `d × d` array of `b`-bin (8-bin) orientation histograms
//! (`d·d·b = 128` values) via **trilinear interpolation** — splitting its
//! magnitude-and-Gaussian-weighted contribution across the two nearest spatial
//! bins on each axis and the two nearest (circular) orientation bins. The result
//! is L2-normalized, every component clamped to `≤ 0.2`, renormalized, and
//! quantized to `u8` via `round(512·v)` clamped to `[0, 255]` (the COLMAP/OpenCV
//! `.sift` convention).
//!
//! # Sampling geometry
//!
//! For a keypoint in octave `o`, the descriptor reads the precomputed gradient
//! images at the Gaussian level nearest the keypoint's `layer`. The keypoint's
//! full-resolution `(x, y)` is converted back to octave-pixel coordinates with
//! [`ScaleSpace::image_to_octave`]. The orientation `θ` comes from the affine
//! shape.
//!
//! - **Subregion spacing.** Each of the `d` subregions spans `hist_width =
//!   m_descr · σ_oct` octave-pixels, where `σ_oct = abs_sigma(layer)` is the
//!   keypoint blur in octave pixels and `m_descr = 3`.
//! - **Window.** The descriptor covers a square of side `d · hist_width`, rotated
//!   by `θ`. To cover the rotated square plus the one-cell interpolation margin we
//!   iterate the integer octave-pixels inside the axis-aligned bounding box of
//!   half-width `radius = round(hist_width · (d + 1) / 2 · √2)`.
//! - **Bin coordinates.** A sample's offset from the center is rotated by `−θ`,
//!   then scaled by `1 / hist_width` to subregion units and shifted by
//!   `d/2 − 0.5` so that subregion centers fall at integer `(rbin, cbin)` in
//!   `[0, d)`. Continuous coordinates therefore range over `[−1, d]`; samples
//!   outside `(−1, d)` are skipped. The orientation bin is `(ori − θ)` (keypoint
//!   minus gradient orientation), scaled to `b/2π`, giving a continuous `obin` in
//!   `[0, b)` (wrapped circularly). The `ori − θ` sign reverses the bin direction
//!   so the layout matches OpenCV/COLMAP (see [`accumulate_histogram`]).
//! - **Gaussian weight.** Each sample is weighted by gradient magnitude times a
//!   Gaussian of the sub-region-unit distance from the center, with σ equal to
//!   **half the descriptor window width**, i.e. `d/2` subregion units
//!   (`exp(−(rbin'² + cbin'²) / (2·(d/2)²))` with `rbin' = rbin − (d/2 − 0.5)`).
//!
//! # Bin ordering
//!
//! The flattened 128-vector is row-major in `(rbin, cbin, obin)`: index
//! `((r · d) + c) · b + o`, with `r` the row subregion (along the rotated
//! y-axis), `c` the column subregion (rotated x-axis), and `o` the orientation
//! bin. This matches the trilinear accumulation order below.

use super::{Descriptors, ScaleSpace, SiftKeypoint};
use std::f32::consts::PI;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Descriptor width `d` (the `d × d` subregion grid). Fixed at 4 (128-D output).
const D: usize = 4;
/// Orientation bins `b` per subregion histogram. Fixed at 8 (128-D output).
const B: usize = 8;
/// Total descriptor length (`d·d·b`).
const LEN: usize = D * D * B;

/// Compute the 128-D descriptor for a single keypoint.
///
/// Reads the precomputed gradient images at the Gaussian level nearest the
/// keypoint's `layer`, samples the rotated window (subregion spacing scaled by
/// `magnification`), accumulates a `d × d × b` histogram by trilinear
/// interpolation, then L2-normalizes, clamps each component to `clamp`,
/// renormalizes, and quantizes to `u8`. Pure function of its inputs.
pub fn compute_descriptor(
    scale_space: &ScaleSpace,
    keypoint: &SiftKeypoint,
    magnification: f32,
    clamp: f32,
) -> [u8; 128] {
    let raw = accumulate_histogram(scale_space, keypoint, magnification);
    let normalized = normalize_clamp_renorm(&raw, clamp);
    quantize(&normalized)
}

/// Compute descriptors for many keypoints, one row per keypoint.
///
/// Parallelized over keypoints with rayon; `collect` preserves input order so
/// `rows()[i]` describes `keypoints[i]`.
pub fn compute_descriptors(
    scale_space: &ScaleSpace,
    keypoints: &[SiftKeypoint],
    magnification: f32,
    clamp: f32,
) -> Descriptors {
    use rayon::prelude::*;
    let rows: Vec<[u8; 128]> = keypoints
        .par_iter()
        .map(|kp| compute_descriptor(scale_space, kp, magnification, clamp))
        .collect();
    Descriptors::from_rows(rows)
}

/// Build the raw (pre-normalization) `d·d·b` histogram for one keypoint by
/// trilinear scatter of every gradient sample in the rotated window.
///
/// Bin layout: `hist[((r * D) + c) * B + o]`.
fn accumulate_histogram(
    scale_space: &ScaleSpace,
    keypoint: &SiftKeypoint,
    magnification: f32,
) -> [f32; LEN] {
    let mut hist = [0.0f32; LEN];

    let octave = keypoint.octave as usize;
    if octave >= scale_space.num_octaves() {
        return hist;
    }
    let (w, h) = scale_space.octave_dims(octave);
    let (w, h) = (w as usize, h as usize);

    // Gaussian level nearest the (fractional) layer, clamped to a valid level.
    let n_levels = scale_space.gaussians_per_octave();
    let level = (keypoint.layer.round() as i32).clamp(0, n_levels as i32 - 1) as usize;
    // Gradients are sampled on the fly from this Gaussian level (see
    // `scale_space::pixel_gradient`); the window below is clamped to the interior.
    let gauss = scale_space.gaussian(octave, level).data();

    // Keypoint center in octave-pixel coordinates (inverse of octave_to_image).
    let (cx, cy) =
        scale_space.image_to_octave(keypoint.octave, keypoint.x as f64, keypoint.y as f64);
    let cx = cx as f32;
    let cy = cy as f32;

    let ori = keypoint.orientation();
    let (sin_t, cos_t) = ori.sin_cos();

    // sigma in octave pixels at this layer; each subregion spans hist_width
    // octave-pixels.
    let sigma_oct = scale_space.abs_sigma(keypoint.layer as f64) as f32;
    let hist_width = magnification * sigma_oct;
    // Guard against a non-positive or non-finite subregion spacing (degenerate
    // scale), which would make the geometry ill-defined.
    if !hist_width.is_finite() || hist_width <= 0.0 {
        return hist;
    }
    let inv_hist_width = 1.0 / hist_width;

    // Bounding box half-width in octave pixels: cover the rotated d×d square plus
    // a one-subregion interpolation margin, hence (d + 1)/2, times sqrt(2) for the
    // diagonal of the rotated square.
    let radius = (hist_width * (D as f32 + 1.0) * 0.5 * std::f32::consts::SQRT_2).round() as i32;
    let radius = radius.max(1);

    // Continuous bin coordinate of the window center, in [0, d): subregion centers
    // sit at integers, so the center maps to d/2 - 0.5.
    let bin_center = D as f32 / 2.0 - 0.5;
    // Gaussian weighting sigma = half the descriptor window width = d/2 subregion
    // units. Precompute -1 / (2 sigma^2).
    let gauss_sigma = D as f32 / 2.0;
    let exp_denom = -1.0 / (2.0 * gauss_sigma * gauss_sigma);
    let bin_to_ori = B as f32 / (2.0 * PI);

    let icx = cx.round() as i32;
    let icy = cy.round() as i32;

    let p = DescParams {
        w,
        h,
        icx,
        icy,
        cx,
        cy,
        sin_t,
        cos_t,
        inv_hist_width,
        radius,
        bin_center,
        exp_denom,
        bin_to_ori,
        ori,
    };

    #[cfg(target_arch = "x86_64")]
    if *super::simd::HAS_AVX2_FMA {
        // SAFETY: guarded by the AVX2+FMA runtime check.
        unsafe { fill_desc_hist_avx2(gauss, &p, &mut hist) };
        return hist;
    }

    fill_desc_hist_scalar(gauss, &p, &mut hist);
    hist
}

/// Geometry of one keypoint's descriptor window, shared by the scalar and AVX2
/// histogram fills. Coordinates are in octave pixels.
struct DescParams {
    w: usize,
    h: usize,
    icx: i32,
    icy: i32,
    cx: f32,
    cy: f32,
    sin_t: f32,
    cos_t: f32,
    inv_hist_width: f32,
    radius: i32,
    bin_center: f32,
    exp_denom: f32,
    bin_to_ori: f32,
    ori: f32,
}

/// Scalar reference fill of the raw descriptor histogram: rotate each in-window
/// pixel into the descriptor frame, then trilinear-scatter its
/// magnitude-and-Gaussian-weighted gradient into the 4×4×8 histogram.
fn fill_desc_hist_scalar(gauss: &[f32], p: &DescParams, hist: &mut [f32; LEN]) {
    for dy in -p.radius..=p.radius {
        let ry = p.icy + dy;
        if ry < 1 || ry >= p.h as i32 - 1 {
            continue;
        }
        let oy = ry as f32 - p.cy;
        for dx in -p.radius..=p.radius {
            let rx = p.icx + dx;
            if rx < 1 || rx >= p.w as i32 - 1 {
                continue;
            }

            // Offset of this pixel from the (sub-pixel) keypoint center, rotated
            // by -theta into the descriptor frame, then scaled to subregion units.
            let ox = rx as f32 - p.cx;
            let rot_x = (ox * p.cos_t + oy * p.sin_t) * p.inv_hist_width;
            let rot_y = (-ox * p.sin_t + oy * p.cos_t) * p.inv_hist_width;
            // Continuous spatial bins in [0, d): cbin from the rotated x-axis,
            // rbin from the rotated y-axis.
            let cbin = rot_x + p.bin_center;
            let rbin = rot_y + p.bin_center;

            // Only samples that can land in a valid spatial bin matter. After
            // trilinear distribution the lowest contributing integer bin is
            // floor(bin); require bins in (-1, d).
            if rbin <= -1.0 || rbin >= D as f32 || cbin <= -1.0 || cbin >= D as f32 {
                continue;
            }

            let idx = ry as usize * p.w + rx as usize;
            let (m, grad_theta) = super::scale_space::pixel_gradient(gauss, p.w, idx);
            // Gaussian weight on distance from center, in subregion units.
            let weight = (p.exp_denom * (rot_x * rot_x + rot_y * rot_y)).exp();
            let value = m * weight;

            // Orientation bin: keypoint orientation minus gradient orientation.
            // The sign (ori − θ, not θ − ori) reverses the bin direction so the
            // layout matches OpenCV/COLMAP — without it our bins are an exact
            // orientation reflection of OpenCV's. Still relative to the keypoint,
            // so rotation invariance holds.
            let t = (p.ori - grad_theta).rem_euclid(2.0 * PI);
            let obin = t * p.bin_to_ori; // in [0, b)

            trilinear_add(hist, rbin, cbin, obin, value);
        }
    }
}

/// AVX2 fill of the raw descriptor histogram. Each row is processed in 8-pixel
/// chunks: the rotation into the descriptor frame, gradient, magnitude, angle
/// (`atan2_approx`) and Gaussian weight (`exp_approx`) are all computed 8-wide;
/// only the per-sample trilinear scatter (up to 8 bin updates) is scalar.
/// Identical to the scalar fill modulo the transcendental approximations.
///
/// # Safety
/// Requires `avx2` + `fma` (guarded by [`super::simd::HAS_AVX2_FMA`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fill_desc_hist_avx2(gauss: &[f32], p: &DescParams, hist: &mut [f32; LEN]) {
    let lanes = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    let cos_v = _mm256_set1_ps(p.cos_t);
    let sin_v = _mm256_set1_ps(p.sin_t);
    let inv_v = _mm256_set1_ps(p.inv_hist_width);
    let binc_v = _mm256_set1_ps(p.bin_center);
    let expd_v = _mm256_set1_ps(p.exp_denom);
    let ori_v = _mm256_set1_ps(p.ori);
    let inv_twopi = _mm256_set1_ps(1.0 / (2.0 * PI));
    let twopi = _mm256_set1_ps(2.0 * PI);
    let bori_v = _mm256_set1_ps(p.bin_to_ori);
    let dlim = D as f32;
    let wi = p.w as i32;

    for dy in -p.radius..=p.radius {
        let ry = p.icy + dy;
        if ry < 1 || ry >= p.h as i32 - 1 {
            continue;
        }
        let row_base = ry as usize * p.w;
        let oy = ry as f32 - p.cy;
        // oy is constant across the row; fold it into the rotation up front.
        let oy_sin = _mm256_set1_ps(oy * p.sin_t);
        let oy_cos = _mm256_set1_ps(oy * p.cos_t);
        // In-bounds column range: rx ∈ [1, w−2] ⇒ every idx±1 / idx±w load is in
        // bounds. The (−1, d) spatial-bin cutoff is applied per lane in the scatter.
        let lo = (1 - p.icx).max(-p.radius);
        let hi = (wi - 2 - p.icx).min(p.radius);

        let mut dx = lo;
        while dx + 8 <= hi + 1 {
            let base = row_base + (p.icx + dx) as usize;
            // ox = (icx + dx + lane) − cx; rotate into the descriptor frame.
            let ox = _mm256_add_ps(_mm256_set1_ps((p.icx + dx) as f32 - p.cx), lanes);
            let rot_x = _mm256_mul_ps(_mm256_fmadd_ps(ox, cos_v, oy_sin), inv_v);
            let rot_y = _mm256_mul_ps(_mm256_fnmadd_ps(ox, sin_v, oy_cos), inv_v);
            let cbin = _mm256_add_ps(rot_x, binc_v);
            let rbin = _mm256_add_ps(rot_y, binc_v);

            let gx = _mm256_sub_ps(
                _mm256_loadu_ps(gauss.as_ptr().add(base + 1)),
                _mm256_loadu_ps(gauss.as_ptr().add(base - 1)),
            );
            let gy = _mm256_sub_ps(
                _mm256_loadu_ps(gauss.as_ptr().add(base + p.w)),
                _mm256_loadu_ps(gauss.as_ptr().add(base - p.w)),
            );
            let m = _mm256_sqrt_ps(_mm256_fmadd_ps(gx, gx, _mm256_mul_ps(gy, gy)));
            let gtheta = super::simd::atan2_approx(gy, gx);

            // weight = exp(exp_denom·(rot_x² + rot_y²)); value = m·weight.
            let r2 = _mm256_fmadd_ps(rot_x, rot_x, _mm256_mul_ps(rot_y, rot_y));
            let weight = super::simd::exp_approx(_mm256_mul_ps(expd_v, r2));
            let value = _mm256_mul_ps(m, weight);

            // obin = rem_euclid(ori − gtheta, 2π) · bin_to_ori. ori − gtheta ∈
            // (−π, 3π), so floor(t/2π) ∈ {−1, 0, 1} reduces it to [0, 2π).
            let t = _mm256_sub_ps(ori_v, gtheta);
            let fl = _mm256_floor_ps(_mm256_mul_ps(t, inv_twopi));
            let t = _mm256_fnmadd_ps(fl, twopi, t);
            let obin = _mm256_mul_ps(t, bori_v);

            let mut rb = [0.0f32; 8];
            let mut cb = [0.0f32; 8];
            let mut ob = [0.0f32; 8];
            let mut va = [0.0f32; 8];
            _mm256_storeu_ps(rb.as_mut_ptr(), rbin);
            _mm256_storeu_ps(cb.as_mut_ptr(), cbin);
            _mm256_storeu_ps(ob.as_mut_ptr(), obin);
            _mm256_storeu_ps(va.as_mut_ptr(), value);
            for k in 0..8 {
                if rb[k] <= -1.0 || rb[k] >= dlim || cb[k] <= -1.0 || cb[k] >= dlim {
                    continue;
                }
                trilinear_add(hist, rb[k], cb[k], ob[k], va[k]);
            }
            dx += 8;
        }
        // Scalar tail for the remaining in-bounds columns [dx, hi].
        while dx <= hi {
            let rx = p.icx + dx;
            let ox = rx as f32 - p.cx;
            let rot_x = (ox * p.cos_t + oy * p.sin_t) * p.inv_hist_width;
            let rot_y = (-ox * p.sin_t + oy * p.cos_t) * p.inv_hist_width;
            let cbin = rot_x + p.bin_center;
            let rbin = rot_y + p.bin_center;
            if !(rbin <= -1.0 || rbin >= dlim || cbin <= -1.0 || cbin >= dlim) {
                let idx = row_base + rx as usize;
                let (m, grad_theta) = super::scale_space::pixel_gradient(gauss, p.w, idx);
                let weight = (p.exp_denom * (rot_x * rot_x + rot_y * rot_y)).exp();
                let value = m * weight;
                let t = (p.ori - grad_theta).rem_euclid(2.0 * PI);
                trilinear_add(hist, rbin, cbin, t * p.bin_to_ori, value);
            }
            dx += 1;
        }
    }
}

/// Distribute `value` into the 8 bins surrounding the continuous coordinate
/// `(rbin, cbin, obin)` by trilinear interpolation. Spatial bins are clamped to
/// valid `[0, d)` cells; the orientation bin wraps circularly in `[0, b)`.
fn trilinear_add(hist: &mut [f32; LEN], rbin: f32, cbin: f32, obin: f32, value: f32) {
    let r0 = rbin.floor();
    let c0 = cbin.floor();
    let o0 = obin.floor();
    let dr = rbin - r0;
    let dc = cbin - c0;
    let do_ = obin - o0;
    let r0 = r0 as i32;
    let c0 = c0 as i32;
    let o0 = o0 as i32;

    for (ri, rw) in [(r0, 1.0 - dr), (r0 + 1, dr)] {
        if ri < 0 || ri >= D as i32 {
            continue;
        }
        for (ci, cw) in [(c0, 1.0 - dc), (c0 + 1, dc)] {
            if ci < 0 || ci >= D as i32 {
                continue;
            }
            let base = ((ri as usize * D) + ci as usize) * B;
            for (oi, ow) in [(o0, 1.0 - do_), (o0 + 1, do_)] {
                let ob = oi.rem_euclid(B as i32) as usize;
                hist[base + ob] += value * rw * cw * ow;
            }
        }
    }
}

/// L2-normalize the raw histogram to unit length, clamp each component to
/// `clamp`, and renormalize to unit length. Returns the normalized vector
/// (pre-quantization).
fn normalize_clamp_renorm(raw: &[f32; LEN], clamp: f32) -> [f32; LEN] {
    let mut v = *raw;
    l2_normalize(&mut v);
    let mut clamped = false;
    for x in v.iter_mut() {
        if *x > clamp {
            *x = clamp;
            clamped = true;
        }
    }
    if clamped {
        l2_normalize(&mut v);
    }
    v
}

/// Normalize a 128-vector to unit L2 length in place (no-op for a zero vector).
fn l2_normalize(v: &mut [f32; LEN]) {
    let norm = l2_norm(v);
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 is baseline on x86_64.
            unsafe { scale_sse2(v, inv) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }
}

/// L2 norm of a 128-vector. SSE2 4-wide sum of squares with a horizontal sum,
/// scalar fallback elsewhere.
fn l2_norm(v: &[f32; LEN]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64; LEN is a multiple of 4.
        unsafe { l2_norm_sse2(v) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut acc = 0.0f32;
        for &x in v.iter() {
            acc += x * x;
        }
        acc.sqrt()
    }
}

/// SSE2 sum-of-squares (then sqrt) over a 128-vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn l2_norm_sse2(v: &[f32; LEN]) -> f32 {
    let mut acc = _mm_setzero_ps();
    let mut i = 0;
    while i + 4 <= LEN {
        let p = _mm_loadu_ps(v.as_ptr().add(i));
        acc = _mm_add_ps(acc, _mm_mul_ps(p, p));
        i += 4;
    }
    // Horizontal sum of the 4 lanes.
    let shuf = _mm_shuffle_ps(acc, acc, 0b01_00_11_10);
    let sums = _mm_add_ps(acc, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0b00_00_00_01);
    let total = _mm_add_ps(sums, shuf2);
    _mm_cvtss_f32(total).sqrt()
}

/// SSE2 in-place scale of a 128-vector by `s`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn scale_sse2(v: &mut [f32; LEN], s: f32) {
    let sv = _mm_set1_ps(s);
    let mut i = 0;
    while i + 4 <= LEN {
        let p = _mm_loadu_ps(v.as_ptr().add(i));
        _mm_storeu_ps(v.as_mut_ptr().add(i), _mm_mul_ps(p, sv));
        i += 4;
    }
}

/// Quantize a normalized descriptor to `u8` via `round(512·v)` clamped to
/// `[0, 255]` (the COLMAP/OpenCV `.sift` convention).
fn quantize(v: &[f32; LEN]) -> [u8; 128] {
    let mut out = [0u8; 128];
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let q = (512.0 * x).round();
        *o = q.clamp(0.0, 255.0) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical_flow::GrayImage;
    use crate::sift::SiftParams;

    // Default magnification / clamp (mirroring `SiftParams::default`) for the
    // tests that call the descriptor functions directly.
    const DEFAULT_MAGNIFICATION: f32 = 3.0;
    const DEFAULT_CLAMP: f32 = 0.2;

    /// A smooth image with a single dominant gradient direction (linear ramp
    /// along `dir`), scaled by `amplitude`.
    fn directional_ramp(w: u32, h: u32, dir: f32, amplitude: f32) -> GrayImage {
        let (s, c) = dir.sin_cos();
        let mut data = vec![0.0f32; (w * h) as usize];
        for row in 0..h {
            for col in 0..w {
                let x = col as f32;
                let y = row as f32;
                data[(row * w + col) as usize] = 0.5 + amplitude * (x * c + y * s);
            }
        }
        GrayImage::new(w, h, data)
    }

    fn center_keypoint(orientation: f32) -> SiftKeypoint {
        // Octave 0 of a doubled-by-default 80x80 image is 160x160; its center in
        // full-resolution coords is ~ (40, 40). Use abs_sigma at layer 1 as scale.
        SiftKeypoint::from_similarity(40.0, 40.0, 4.0, orientation, 0, 1.0, 0.1)
    }

    #[test]
    fn test_single_direction_concentrates_orientation_bins() {
        // Gradient everywhere points along +x (dir = 0). With the descriptor
        // oriented to 0, every sample's relative orientation is ~0, so mass should
        // pile into orientation bin 0 of the histograms.
        let img = directional_ramp(80, 80, 0.0, 0.002);
        let ss = ScaleSpace::build(&img, &SiftParams::default());
        let kp = center_keypoint(0.0);
        let hist = accumulate_histogram(&ss, &kp, DEFAULT_MAGNIFICATION);

        // Sum the mass per orientation bin across all spatial cells.
        let mut per_ori = [0.0f32; B];
        for cell in 0..D * D {
            for o in 0..B {
                per_ori[o] += hist[cell * B + o];
            }
        }
        let total: f32 = per_ori.iter().sum();
        assert!(total > 0.0, "no descriptor mass accumulated");
        // Bin 0 (relative orientation 0) should dominate.
        let max_bin = (0..B)
            .max_by(|&a, &b| per_ori[a].partial_cmp(&per_ori[b]).unwrap())
            .unwrap();
        assert_eq!(max_bin, 0, "orientation mass not in bin 0: {per_ori:?}");
        assert!(
            per_ori[0] > 0.5 * total,
            "bin 0 should hold most mass: {per_ori:?}"
        );
    }

    #[test]
    fn test_normalization_invariants() {
        let img = directional_ramp(80, 80, 0.7, 0.003);
        let ss = ScaleSpace::build(&img, &SiftParams::default());
        let kp = center_keypoint(0.3);
        let raw = accumulate_histogram(&ss, &kp, DEFAULT_MAGNIFICATION);
        let norm = normalize_clamp_renorm(&raw, DEFAULT_CLAMP);

        // Unit length.
        let len = l2_norm(&norm);
        assert!((len - 1.0).abs() < 1e-4, "not unit length: {len}");
        // The spec's normalization is a single clamp-then-renormalize pass
        // (matching OpenCV/COLMAP). Clamping to the clamp value and renormalizing
        // scales the whole vector up, so components that were pinned at the clamp
        // can creep modestly above it. Bound that creep generously rather than
        // expecting the strict `x <= clamp` invariant a multi-pass scheme would give.
        for &x in norm.iter() {
            assert!(
                x <= DEFAULT_CLAMP * 1.1,
                "component {x} exceeds clamp by too much"
            );
        }
    }

    #[test]
    fn test_contrast_invariance() {
        // Multiplying the image gradients by a positive constant must not change
        // the quantized descriptor (normalization removes the scale).
        let img_a = directional_ramp(80, 80, 0.4, 0.002);
        let img_b = directional_ramp(80, 80, 0.4, 0.006); // 3x the gradient.
        let ss_a = ScaleSpace::build(&img_a, &SiftParams::default());
        let ss_b = ScaleSpace::build(&img_b, &SiftParams::default());
        let kp = center_keypoint(0.4);
        let da = compute_descriptor(&ss_a, &kp, DEFAULT_MAGNIFICATION, DEFAULT_CLAMP);
        let db = compute_descriptor(&ss_b, &kp, DEFAULT_MAGNIFICATION, DEFAULT_CLAMP);
        assert_eq!(da, db, "descriptor changed under contrast scaling");
    }

    #[test]
    fn test_rotation_permutes_layout() {
        // Rotating the keypoint orientation by +90 degrees cyclically permutes the
        // orientation bins. With the `ori - theta` binning, increasing the keypoint
        // orientation by +90deg (2 of 8 bins) shifts relative-orientation bins *up*
        // by 2, so ori90[o] matches ori0[(o - 2) mod 8].
        let img = directional_ramp(80, 80, 0.0, 0.002);
        let ss = ScaleSpace::build(&img, &SiftParams::default());

        let hist0 = accumulate_histogram(&ss, &center_keypoint(0.0), DEFAULT_MAGNIFICATION);
        let hist90 = accumulate_histogram(&ss, &center_keypoint(PI / 2.0), DEFAULT_MAGNIFICATION);

        let mut ori0 = [0.0f32; B];
        let mut ori90 = [0.0f32; B];
        for cell in 0..D * D {
            for o in 0..B {
                ori0[o] += hist0[cell * B + o];
                ori90[o] += hist90[cell * B + o];
            }
        }
        // With `ori - theta` binning, +90deg (2 of 8 bins) shifts relative-orientation
        // bins up by 2. Compare ori90[o] to ori0[(o - 2) mod B].
        let total0: f32 = ori0.iter().sum();
        let total90: f32 = ori90.iter().sum();
        assert!(total0 > 0.0 && total90 > 0.0);
        for o in 0..B {
            let expected = ori0[(o + B - 2) % B] / total0;
            let got = ori90[o] / total90;
            assert!(
                (expected - got).abs() < 0.15,
                "rotation mismatch at bin {o}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_zero_image_zero_descriptor() {
        let img = GrayImage::new_constant(80, 80, 0.5);
        let ss = ScaleSpace::build(&img, &SiftParams::default());
        let kp = center_keypoint(0.0);
        let d = compute_descriptor(&ss, &kp, DEFAULT_MAGNIFICATION, DEFAULT_CLAMP);
        assert!(d.iter().all(|&b| b == 0), "flat image should give zeros");
    }
}
