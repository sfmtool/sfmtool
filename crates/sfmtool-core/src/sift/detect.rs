// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Scale-space extrema detection and subpixel keypoint localization.
//!
//! Stages 3 and 4 of SIFT (`specs/core/sift.md`): the 26-neighbor DoG extrema
//! test, then the 3D quadratic fit that pins each keypoint's subpixel `(x, y)`
//! and continuous scale, followed by the contrast and edge-response rejections.
//!
//! # Coordinate convention
//!
//! [`LocalizedKeypoint::x`] / `y` are **octave-pixel** coordinates (the column /
//! row in the keypoint's octave grid, pixel-center convention), *not*
//! full-resolution coordinates. Orientation assignment samples gradients from
//! the octave's Gaussian image, so it is natural to keep the location in octave
//! pixels and defer the full-resolution conversion to the very end (in
//! `orientation`, via [`ScaleSpace::octave_to_image`]). The `scale` field, by
//! contrast, is the absolute blur in **full-resolution** pixels
//! (`abs_sigma_full(octave, layer)`), since size is reported in image units.

use super::scale_space::uninit_vec_f32;
use super::{ScaleSpace, SiftParams};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A detected DoG extremum that has been localized to subpixel/subscale
/// precision but does **not** yet carry an orientation.
///
/// `x` and `y` are **octave-pixel** coordinates in the keypoint's octave (see
/// the module docs); `scale` is the absolute blur in full-resolution pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalizedKeypoint {
    /// Refined x coordinate, in octave-pixel units.
    pub x: f32,
    /// Refined y coordinate, in octave-pixel units.
    pub y: f32,
    /// Keypoint scale (size) in full-resolution units.
    pub scale: f32,
    /// Octave the extremum was found in.
    pub octave: i32,
    /// Continuous sub-level (layer) within the octave.
    pub layer: f32,
    /// Contrast response `|D(x̂)|`.
    pub response: f32,
}

/// Border (in pixels) kept clear of the image edge during detection and
/// localization, so the 3x3 finite-difference stencils stay in bounds.
const BORDER: usize = 1;

/// Maximum number of re-centering iterations during subpixel localization.
const MAX_INTERP_STEPS: usize = 5;

/// Owned interior rows per detection stripe (the tiling granularity).
///
/// Each stripe materializes its `s+2` DoG levels into a scratch buffer of
/// `(s+2) × (STRIPE_ROWS+2) × W × f32`; the value trades cache residency
/// (smaller = better L2/L3 fit, more tasks) against a little halo recompute and
/// per-task overhead. Overridable via `SFM_SIFT_STRIPE_ROWS` for tuning.
static STRIPE_ROWS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("SFM_SIFT_STRIPE_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        // Cap the value so `y0 + stripe_rows` in the stripe loop cannot
        // overflow, and so a large override cannot collapse an octave into a
        // single stripe that no longer fits in cache.
        .map(|n| n.min(1 << 16))
        .unwrap_or(32)
});

/// Detect scale-space extrema in the DoG and localize each to subpixel
/// precision, rejecting low-contrast and edge-like responses.
///
/// The DoG is **not** materialized: detection is tiled into `(octave, row-stripe)`
/// jobs parallelized with rayon, and each stripe computes its DoG band in a
/// cache-resident scratch buffer from the resident Gaussian stack (see
/// `detect_in_stripe`). Within a stripe the flat pre-reject (`|D| <= 0.8·C`) is
/// vectorized with SSE2 (scalar fallback), then surviving candidates run the
/// full 26-neighbor test and the quadratic fit. Stripes own disjoint row bands
/// and only read a 1-row halo, so the keypoint set is independent of
/// `STRIPE_ROWS` and thread count.
pub fn detect_and_localize(
    scale_space: &ScaleSpace,
    params: &SiftParams,
) -> Vec<LocalizedKeypoint> {
    let num_octaves = scale_space.num_octaves();
    let contrast_threshold = params.contrast_threshold as f32;
    // Cheap pre-threshold: skip samples whose magnitude is well below contrast.
    let prelim = 0.8 * contrast_threshold;
    let stripe_rows = *STRIPE_ROWS;

    // Build the (octave, [y0, y1)) stripe work list over interior rows.
    let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
    for o in 0..num_octaves {
        let (w, h) = scale_space.octave_dims(o);
        let (w, h) = (w as usize, h as usize);
        if w <= 2 * BORDER || h <= 2 * BORDER {
            continue;
        }
        let mut y0 = BORDER;
        while y0 < h - BORDER {
            let y1 = y0.saturating_add(stripe_rows).min(h - BORDER);
            jobs.push((o, y0, y1));
            y0 = y1;
        }
    }

    jobs.par_iter()
        .flat_map(|&(o, y0, y1)| detect_in_stripe(scale_space, params, o, y0, y1, prelim))
        .collect()
}

/// Fill a stripe's DoG band for octave `o` over scratch rows `[rlo, rhi)`.
///
/// Returns `ndog × level_stride` f32 where `dog[d] = gaussian(d+1) − gaussian(d)`
/// (bit-identical to the previously materialized DoG). Lives in cache, never RAM.
fn fill_stripe_dog(
    scale_space: &ScaleSpace,
    o: usize,
    rlo: usize,
    rhi: usize,
    w: usize,
    ndog: usize,
) -> Vec<f32> {
    let level_stride = (rhi - rlo) * w;
    let src = rlo * w;
    // Uninitialized: every element is written below before any read.
    let mut dog = uninit_vec_f32(ndog * level_stride);
    for d in 0..ndog {
        let g_lo = &scale_space.gaussian(o, d).data()[src..src + level_stride];
        let g_hi = &scale_space.gaussian(o, d + 1).data()[src..src + level_stride];
        let dst = &mut dog[d * level_stride..(d + 1) * level_stride];
        // dst = g_hi − g_lo, SSE2 on x86_64 (bit-identical to the scalar form;
        // `_mm_sub_ps` is a plain subtract, no FMA contraction), matching the
        // `difference()` blur helper. Scalar fallback elsewhere.
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE2 is baseline on x86_64; the three slices are equal-length.
        unsafe {
            super::scale_space::sub_slice_sse2(g_hi, g_lo, dst);
        }
        #[cfg(not(target_arch = "x86_64"))]
        for i in 0..level_stride {
            dst[i] = g_hi[i] - g_lo[i];
        }
    }
    dog
}

/// Detect and localize extrema in octave `o`, owned interior rows `[y0, y1)`.
///
/// Computes the `s+2` DoG levels for this stripe — owned rows plus a 1-row halo
/// each side for the 3×3×3 test — into a cache-resident scratch buffer from the
/// resident Gaussians (`dog(d) = gaussian(d+1) − gaussian(d)`, bit-identical to
/// the previously materialized DoG), then scans interior levels `1..=s`.
/// `localize` reads DoG on the fly from the gaussians, so its iterative,
/// multi-level, re-centering access is unconstrained by the stripe.
fn detect_in_stripe(
    scale_space: &ScaleSpace,
    params: &SiftParams,
    o: usize,
    y0: usize,
    y1: usize,
    prelim: f32,
) -> Vec<LocalizedKeypoint> {
    let s = scale_space.octave_layers() as usize;
    let (w, h) = scale_space.octave_dims(o);
    let (w, h) = (w as usize, h as usize);

    // Scratch covers owned rows [y0, y1) plus a 1-row halo each side. Interior
    // rows satisfy BORDER (=1) <= y0 < y1 <= h-BORDER, so y0-1 and y1 are valid.
    let rlo = y0 - 1;
    let rhi = y1 + 1;
    let level_stride = (rhi - rlo) * w;
    let ndog = s + 2; // virtual DoG levels 0..=s+1

    let dog = fill_stripe_dog(scale_space, o, rlo, rhi, w, ndog);

    // Reused across every (level, row) of this stripe: one scratch instead of a
    // fresh Vec<bool> per row. All interior spans have the same length.
    let mut mask = vec![false; w - 2 * BORDER];

    let mut out = Vec::new();
    for l in 1..=s {
        let below = &dog[(l - 1) * level_stride..l * level_stride];
        let cur = &dog[l * level_stride..(l + 1) * level_stride];
        let above = &dog[(l + 1) * level_stride..(l + 2) * level_stride];
        for row in y0..y1 {
            // Owned global row -> local scratch row (owned rows start at local 1).
            let local_row = row - rlo;
            let row_start = local_row * w;
            let span = &cur[row_start + BORDER..row_start + w - BORDER];
            fill_candidate_mask(span, prelim, &mut mask);
            for (i, &val) in mask.iter().enumerate() {
                if !val {
                    continue;
                }
                let col = BORDER + i;
                let idx = row_start + col;
                let center = cur[idx];
                if is_extremum(below, cur, above, idx, w, center) {
                    // localize takes global (col, row) and reads gaussians directly.
                    if let Some(kp) = localize(scale_space, params, o, l, col, row, w, h) {
                        out.push(kp);
                    }
                }
            }
        }
    }
    out
}

/// Fill `mask` with the per-column pre-threshold `|D| > prelim` for `span`
/// (`mask.len() == span.len()`, every element overwritten). SSE2 inner loop on
/// x86_64, scalar elsewhere. The caller reuses one `mask` across rows.
fn fill_candidate_mask(span: &[f32], prelim: f32, mask: &mut [bool]) {
    debug_assert_eq!(span.len(), mask.len());
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { abs_gt_mask_sse2(span, prelim, mask) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for (m, &v) in mask.iter_mut().zip(span.iter()) {
            *m = v.abs() > prelim;
        }
    }
}

/// SSE2 `|span[i]| > thr` into a bool mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn abs_gt_mask_sse2(span: &[f32], thr: f32, mask: &mut [bool]) {
    let n = span.len();
    // |x| via clearing the sign bit.
    let sign_mask = _mm_set1_ps(f32::from_bits(0x7fff_ffff));
    let vthr = _mm_set1_ps(thr);
    let mut i = 0;
    while i + 4 <= n {
        let v = _mm_loadu_ps(span.as_ptr().add(i));
        let absv = _mm_and_ps(v, sign_mask);
        let cmp = _mm_cmpgt_ps(absv, vthr);
        let bits = _mm_movemask_ps(cmp);
        *mask.get_unchecked_mut(i) = (bits & 1) != 0;
        *mask.get_unchecked_mut(i + 1) = (bits & 2) != 0;
        *mask.get_unchecked_mut(i + 2) = (bits & 4) != 0;
        *mask.get_unchecked_mut(i + 3) = (bits & 8) != 0;
        i += 4;
    }
    while i < n {
        *mask.get_unchecked_mut(i) = (*span.get_unchecked(i)).abs() > thr;
        i += 1;
    }
}

/// Strict-extremum test over the 26 neighbors (3x3 in each of the levels below,
/// current, above). `idx` is the flat index of the center in the current level.
#[inline]
fn is_extremum(
    below: &[f32],
    cur: &[f32],
    above: &[f32],
    idx: usize,
    w: usize,
    center: f32,
) -> bool {
    // Offsets of the 3x3 block around idx.
    let offs = [
        idx - w - 1,
        idx - w,
        idx - w + 1,
        idx - 1,
        idx + 1,
        idx + w - 1,
        idx + w,
        idx + w + 1,
    ];
    if center > 0.0 {
        // Strict maximum: greater than all 26.
        if center <= below[idx] || center <= above[idx] {
            return false;
        }
        for &o in &offs {
            if center <= cur[o] || center <= below[o] || center <= above[o] {
                return false;
            }
        }
        // The 3 center-column neighbors in below/above already checked (below[idx], above[idx]);
        // also the diagonal/edge neighbors in below/above covered by offs loop.
        true
    } else {
        // Strict minimum: less than all 26.
        if center >= below[idx] || center >= above[idx] {
            return false;
        }
        for &o in &offs {
            if center >= cur[o] || center >= below[o] || center >= above[o] {
                return false;
            }
        }
        true
    }
}

/// The four resident gaussian rows bracketing DoG level `l`, for on-the-fly DoG
/// (`dog(L) = gaussian(L+1) − gaussian(L)`, bit-identical to the materialized
/// DoG). Returned as slices so each call site builds its below/cur/above closures.
fn dog_gaussians(scale_space: &ScaleSpace, o: usize, l: usize) -> (&[f32], &[f32], &[f32], &[f32]) {
    (
        scale_space.gaussian(o, l - 1).data(),
        scale_space.gaussian(o, l).data(),
        scale_space.gaussian(o, l + 1).data(),
        scale_space.gaussian(o, l + 2).data(),
    )
}

/// Subpixel/subscale localization (Lowe eq. 2/3) plus the contrast and
/// edge-response rejections. Returns `None` if the candidate is rejected or the
/// fit does not converge inside the valid range.
#[allow(clippy::too_many_arguments)]
fn localize(
    scale_space: &ScaleSpace,
    params: &SiftParams,
    o: usize,
    l0: usize,
    col0: usize,
    row0: usize,
    w: usize,
    h: usize,
) -> Option<LocalizedKeypoint> {
    let s = scale_space.octave_layers() as usize;
    let mut col = col0;
    let mut row = row0;
    let mut l = l0;

    let contrast_threshold = params.contrast_threshold as f32;
    let r = params.edge_threshold as f32;

    let mut offset = Vector3::zeros();
    let mut grad;
    let mut converged = false;

    for _ in 0..MAX_INTERP_STEPS {
        // DoG samplers around (o, l), read on the fly from the resident
        // gaussians (full random access, so re-centering is unconstrained).
        let (gm1, g0, gp1, gp2) = dog_gaussians(scale_space, o, l);
        let below = |i: usize| g0[i] - gm1[i];
        let cur = |i: usize| gp1[i] - g0[i];
        let above = |i: usize| gp2[i] - gp1[i];
        let idx = row * w + col;

        // 3D gradient (central differences) in (x, y, sigma).
        let dx = 0.5 * (cur(idx + 1) - cur(idx - 1));
        let dy = 0.5 * (cur(idx + w) - cur(idx - w));
        let ds = 0.5 * (above(idx) - below(idx));
        grad = Vector3::new(dx, dy, ds);

        // 3x3 Hessian (central second differences).
        let v2 = 2.0 * cur(idx);
        let dxx = cur(idx + 1) + cur(idx - 1) - v2;
        let dyy = cur(idx + w) + cur(idx - w) - v2;
        let dss = above(idx) + below(idx) - v2;
        let dxy =
            0.25 * (cur(idx + w + 1) - cur(idx + w - 1) - cur(idx - w + 1) + cur(idx - w - 1));
        let dxs = 0.25 * (above(idx + 1) - above(idx - 1) - below(idx + 1) + below(idx - 1));
        let dys = 0.25 * (above(idx + w) - above(idx - w) - below(idx + w) + below(idx - w));

        #[rustfmt::skip]
        let hessian = Matrix3::new(
            dxx, dxy, dxs,
            dxy, dyy, dys,
            dxs, dys, dss,
        );

        // x̂ = -H⁻¹ g
        let solved = hessian.lu().solve(&grad);
        let xhat = match solved {
            Some(v) => -v,
            None => return None, // singular Hessian
        };

        // If within half a pixel in every dimension, we are done.
        if xhat[0].abs() < 0.5 && xhat[1].abs() < 0.5 && xhat[2].abs() < 0.5 {
            offset = xhat;
            converged = true;
            break;
        }

        // A near-singular (but not exactly singular) Hessian can make the LU solve
        // return a huge or non-finite step. Such a candidate is diverging, so reject
        // it rather than re-centering with it: `xhat.round() as i32` saturates to
        // i32::MAX and `col as i32 + i32::MAX` would overflow (a panic in debug
        // builds, a wrap in release). A valid refinement never moves more than the
        // octave's own extent.
        if !(xhat[0].is_finite() && xhat[1].is_finite() && xhat[2].is_finite())
            || xhat[0].abs() > w as f32
            || xhat[1].abs() > h as f32
            || xhat[2].abs() > s as f32
        {
            return None;
        }

        // Otherwise re-center on the rounded offset and refit.
        col = (col as i32 + xhat[0].round() as i32) as usize;
        row = (row as i32 + xhat[1].round() as i32) as usize;
        l = (l as i32 + xhat[2].round() as i32).max(0) as usize;

        // Stay in the interior of valid DoG levels (1..=s) and away from the border.
        if l < 1 || l > s || col < BORDER || col >= w - BORDER || row < BORDER || row >= h - BORDER
        {
            return None;
        }
    }

    if !converged {
        return None;
    }

    // Re-evaluate gradient at the converged sample for the contrast value.
    let (gm1, g0, gp1, gp2) = dog_gaussians(scale_space, o, l);
    let below = |i: usize| g0[i] - gm1[i];
    let cur = |i: usize| gp1[i] - g0[i];
    let above = |i: usize| gp2[i] - gp1[i];
    let idx = row * w + col;
    let dx = 0.5 * (cur(idx + 1) - cur(idx - 1));
    let dy = 0.5 * (cur(idx + w) - cur(idx - w));
    let ds = 0.5 * (above(idx) - below(idx));
    let grad_f = Vector3::new(dx, dy, ds);

    // Contrast: D(x̂) = D + 0.5 gᵀ x̂.
    let d_hat = cur(idx) + 0.5 * grad_f.dot(&offset);
    if d_hat.abs() < contrast_threshold {
        return None;
    }

    // Edge response from the 2x2 spatial Hessian.
    let v2 = 2.0 * cur(idx);
    let dxx = cur(idx + 1) + cur(idx - 1) - v2;
    let dyy = cur(idx + w) + cur(idx - w) - v2;
    let dxy = 0.25 * (cur(idx + w + 1) - cur(idx + w - 1) - cur(idx - w + 1) + cur(idx - w - 1));
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if det <= 0.0 {
        return None;
    }
    let edge_limit = (r + 1.0) * (r + 1.0) / r;
    if tr * tr / det >= edge_limit {
        return None;
    }

    // Refined location / layer.
    let layer = l as f32 + offset[2];
    let scale = scale_space.abs_sigma_full(o as i32, layer as f64) as f32;

    Some(LocalizedKeypoint {
        x: col as f32 + offset[0],
        y: row as f32 + offset[1],
        scale,
        octave: o as i32,
        layer,
        response: d_hat.abs(),
    })
}

#[cfg(test)]
mod tests;
