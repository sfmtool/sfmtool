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

/// Detect scale-space extrema in the DoG pyramid and localize each to subpixel
/// precision, rejecting low-contrast and edge-like responses.
///
/// Parallelized across `(octave, level)` slices with rayon; within each slice
/// the flat pre-reject (`|D| <= 0.8·C`) is vectorized with SSE2 (scalar
/// fallback), then surviving candidates run the full 26-neighbor test and the
/// quadratic fit.
pub fn detect_and_localize(
    scale_space: &ScaleSpace,
    params: &SiftParams,
) -> Vec<LocalizedKeypoint> {
    let s = scale_space.octave_layers() as usize;
    let num_octaves = scale_space.num_octaves();
    let contrast_threshold = params.contrast_threshold as f32;
    // Cheap pre-threshold: skip samples whose magnitude is well below contrast.
    let prelim = 0.8 * contrast_threshold;

    // Build the (octave, level) work list: interior DoG levels 1..=s.
    let mut jobs: Vec<(usize, usize)> = Vec::new();
    for o in 0..num_octaves {
        for l in 1..=s {
            jobs.push((o, l));
        }
    }

    jobs.par_iter()
        .flat_map(|&(o, l)| detect_in_level(scale_space, params, o, l, prelim))
        .collect()
}

/// Detect and localize extrema in DoG level `l` of octave `o`.
fn detect_in_level(
    scale_space: &ScaleSpace,
    params: &SiftParams,
    o: usize,
    l: usize,
    prelim: f32,
) -> Vec<LocalizedKeypoint> {
    let (w, h) = scale_space.octave_dims(o);
    let (w, h) = (w as usize, h as usize);
    if w <= 2 * BORDER || h <= 2 * BORDER {
        return Vec::new();
    }

    let below = scale_space.dog(o, l - 1).data();
    let cur = scale_space.dog(o, l).data();
    let above = scale_space.dog(o, l + 1).data();

    let mut out = Vec::new();
    for row in BORDER..h - BORDER {
        // Vectorized pre-reject over this interior row span, producing a mask of
        // columns worth the full 26-neighbor test.
        let row_start = row * w;
        let span = &cur[row_start + BORDER..row_start + w - BORDER];
        for (i, &val) in candidate_cols(span, prelim).iter().enumerate() {
            if !val {
                continue;
            }
            let col = BORDER + i;
            let idx = row_start + col;
            let center = cur[idx];
            if is_extremum(below, cur, above, idx, w, center) {
                if let Some(kp) = localize(scale_space, params, o, l, col, row, w, h) {
                    out.push(kp);
                }
            }
        }
    }
    out
}

/// Per-column boolean mask: `|D| > prelim`. SSE2 inner loop on x86_64, scalar
/// elsewhere. The mask has the same length as `span`.
fn candidate_cols(span: &[f32], prelim: f32) -> Vec<bool> {
    let n = span.len();
    let mut mask = vec![false; n];
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { abs_gt_mask_sse2(span, prelim, &mut mask) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for (m, &v) in mask.iter_mut().zip(span.iter()) {
            *m = v.abs() > prelim;
        }
    }
    mask
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
        let below = scale_space.dog(o, l - 1).data();
        let cur = scale_space.dog(o, l).data();
        let above = scale_space.dog(o, l + 1).data();
        let idx = row * w + col;

        // 3D gradient (central differences) in (x, y, sigma).
        let dx = 0.5 * (cur[idx + 1] - cur[idx - 1]);
        let dy = 0.5 * (cur[idx + w] - cur[idx - w]);
        let ds = 0.5 * (above[idx] - below[idx]);
        grad = Vector3::new(dx, dy, ds);

        // 3x3 Hessian (central second differences).
        let v2 = 2.0 * cur[idx];
        let dxx = cur[idx + 1] + cur[idx - 1] - v2;
        let dyy = cur[idx + w] + cur[idx - w] - v2;
        let dss = above[idx] + below[idx] - v2;
        let dxy =
            0.25 * (cur[idx + w + 1] - cur[idx + w - 1] - cur[idx - w + 1] + cur[idx - w - 1]);
        let dxs = 0.25 * (above[idx + 1] - above[idx - 1] - below[idx + 1] + below[idx - 1]);
        let dys = 0.25 * (above[idx + w] - above[idx - w] - below[idx + w] + below[idx - w]);

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
    let cur = scale_space.dog(o, l).data();
    let below = scale_space.dog(o, l - 1).data();
    let above = scale_space.dog(o, l + 1).data();
    let idx = row * w + col;
    let dx = 0.5 * (cur[idx + 1] - cur[idx - 1]);
    let dy = 0.5 * (cur[idx + w] - cur[idx - w]);
    let ds = 0.5 * (above[idx] - below[idx]);
    let grad_f = Vector3::new(dx, dy, ds);

    // Contrast: D(x̂) = D + 0.5 gᵀ x̂.
    let d_hat = cur[idx] + 0.5 * grad_f.dot(&offset);
    if d_hat.abs() < contrast_threshold {
        return None;
    }

    // Edge response from the 2x2 spatial Hessian.
    let v2 = 2.0 * cur[idx];
    let dxx = cur[idx + 1] + cur[idx - 1] - v2;
    let dyy = cur[idx + w] + cur[idx - w] - v2;
    let dxy = 0.25 * (cur[idx + w + 1] - cur[idx + w - 1] - cur[idx - w + 1] + cur[idx - w - 1]);
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
mod tests {
    use super::super::detect_keypoints;
    use super::*;
    use crate::optical_flow::GrayImage;

    /// Render a Gaussian blob (bright on a flat mid-gray background) centered at
    /// `(cx, cy)` with standard deviation `blob_sigma`.
    fn gaussian_blob(w: u32, h: u32, cx: f32, cy: f32, blob_sigma: f32, amp: f32) -> GrayImage {
        let mut data = vec![0.5f32; (w * h) as usize];
        let inv = 1.0 / (2.0 * blob_sigma * blob_sigma);
        for row in 0..h {
            for col in 0..w {
                let dx = col as f32 + 0.5 - cx;
                let dy = row as f32 + 0.5 - cy;
                let v = 0.5 + amp * (-(dx * dx + dy * dy) * inv).exp();
                data[(row * w + col) as usize] = v;
            }
        }
        GrayImage::new(w, h, data)
    }

    #[test]
    fn test_constant_image_no_keypoints() {
        let img = GrayImage::new_constant(64, 64, 0.5);
        let ss = ScaleSpace::build(&img, &SiftParams::default());
        let kps = detect_and_localize(&ss, &SiftParams::default());
        assert!(
            kps.is_empty(),
            "constant image yielded {} keypoints",
            kps.len()
        );
    }

    #[test]
    fn test_gaussian_blob_detected() {
        let img = gaussian_blob(64, 64, 32.0, 32.0, 4.0, 0.4);
        let detection = detect_keypoints(&img, &SiftParams::default());
        let kps = &detection.keypoints;
        assert!(!kps.is_empty(), "blob produced no keypoints");
        // A keypoint near the blob center should exist.
        let near = kps
            .iter()
            .any(|k| (k.x - 32.0).abs() < 4.0 && (k.y - 32.0).abs() < 4.0);
        assert!(near, "no keypoint near blob center: {:?}", kps);
        // Its scale should be on the order of the blob sigma (a few px), not tiny
        // or enormous.
        let center_kp = kps
            .iter()
            .filter(|k| (k.x - 32.0).abs() < 4.0 && (k.y - 32.0).abs() < 4.0)
            .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
            .unwrap();
        let size = center_kp.scale();
        assert!(size > 1.0 && size < 30.0, "blob keypoint size {size}");
    }

    #[test]
    fn test_edge_ridge_rejected() {
        // A long vertical bright ridge: strong response along an edge, so the
        // edge test should reject the (few) candidates it produces.
        let w = 80u32;
        let h = 80u32;
        let mut data = vec![0.5f32; (w * h) as usize];
        for row in 0..h {
            for col in 38..42 {
                data[(row * w + col) as usize] = 0.9;
            }
        }
        // Soften it a little so it has interior structure.
        let img = GrayImage::new(w, h, data);
        let detection = detect_keypoints(&img, &SiftParams::default());
        // Any surviving keypoints should not cluster along the ridge interior as
        // a well-localized blob; assert none sit squarely in the long uniform
        // middle of the ridge.
        let on_ridge_middle = detection
            .keypoints
            .iter()
            .filter(|k| k.x > 37.0 && k.x < 43.0 && k.y > 20.0 && k.y < 60.0);
        assert_eq!(
            on_ridge_middle.count(),
            0,
            "edge ridge produced keypoints along its length"
        );
    }

    #[test]
    fn test_low_contrast_rejected() {
        // A very faint blob: above the structural noise but below the contrast
        // threshold, so it should be rejected.
        let img = gaussian_blob(64, 64, 32.0, 32.0, 4.0, 0.005);
        let detection = detect_keypoints(&img, &SiftParams::default());
        assert!(
            detection.keypoints.is_empty(),
            "low-contrast blob produced {} keypoints",
            detection.keypoints.len()
        );
    }
}
