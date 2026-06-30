// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch window kernel, the frozen `R×R` scoring [`Support`], and the
//! patch-placement helpers ([`repose_patch`], [`view_render_patch`]) shared by
//! the refinement and the per-point patch operations.

use nalgebra::Vector3;

use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize;

use super::params::{PatchWindow, ProjectedImage};

/// Per-pixel window weight over the `R×R` patch grid, in row-major order.
pub(in crate::patch) fn window_weights(window: PatchWindow, resolution: u32) -> Vec<f64> {
    let r = resolution as usize;
    let step = 2.0 / r as f64;
    let mut w = Vec::with_capacity(r * r);
    for row in 0..r {
        let t = (row as f64 + 0.5) * step - 1.0;
        for col in 0..r {
            let s = (col as f64 + 0.5) * step - 1.0;
            let r2 = s * s + t * t;
            let weight = match window {
                PatchWindow::Uniform => 1.0,
                PatchWindow::Gaussian { sigma } => (-r2 / (2.0 * sigma * sigma)).exp(),
                PatchWindow::GaussianDisk { sigma } => {
                    if r2 > 1.0 {
                        0.0
                    } else {
                        (-r2 / (2.0 * sigma * sigma)).exp()
                    }
                }
            };
            w.push(weight);
        }
    }
    w
}

/// The frozen window support over the `R×R` core: the linear `row * R + col`
/// indices of the positive-weight pixels, their window weights, `√weight` per
/// pixel (folded into the z-normalized space so a plain dot product realizes the
/// windowed inner product), and the total weight (the windowed mean's
/// denominator). Shared by the per-point patch operations (keypoint localize,
/// keypoint subpixel refine) that score on a fixed `R×R` core.
pub(in crate::patch) struct Support {
    pub pixels: Vec<usize>,
    pub weights: Vec<f64>,
    pub sqrt_weights: Vec<f32>,
    pub total_weight: f64,
}

/// Build the frozen [`Support`] for the `R×R` core from the window kernel — keep
/// only pixels whose window weight is positive, in row-major order.
pub(in crate::patch) fn build_support(window: PatchWindow, resolution: u32) -> Support {
    let w_full = window_weights(window, resolution);
    let mut pixels = Vec::new();
    let mut weights = Vec::new();
    for (p, &w) in w_full.iter().enumerate() {
        if w > 0.0 {
            pixels.push(p);
            weights.push(w);
        }
    }
    let total_weight: f64 = weights.iter().sum();
    let sqrt_weights: Vec<f32> = weights.iter().map(|&w| w.sqrt() as f32).collect();
    Support {
        pixels,
        weights,
        sqrt_weights,
        total_weight,
    }
}

/// Rebuild the patch on a new plane: same `center` / `half_extent`, the input
/// `u_axis` reprojected onto the plane of `n` (`v = n × u`).
pub(super) fn repose_patch(base: &OrientedPatch, n: &Vector3<f64>) -> OrientedPatch {
    let mut p = OrientedPatch::from_center_normal(base.center, *n, base.u_axis, base.half_extent);
    p.w = base.w;
    p
}

/// The patch to render into `view`: when `keypoint` is given, recenter `patch`
/// in-plane so it projects at that keypoint (falling back to `patch` if the ray
/// is degenerate); otherwise `patch` unchanged.
///
/// Borrows `patch` in the no-keypoint hot path (and on a degenerate ray), so a
/// refinement without keypoints allocates nothing extra. With a keypoint, the
/// `wpp` factors cancel in the `seed_offset → shifted_center` round trip, so they
/// are passed as `1.0`.
///
// TODO(perf): the keypoint's world ray in `seed_offset` is invariant across
// candidate normals (only the ray∩plane intersection depends on the plane), but
// it is recomputed every candidate/level here. If keypoint-anchored refine ever
// becomes hot, precompute the per-view world ray once per refine call.
pub(super) fn view_render_patch<'a>(
    patch: &'a OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: Option<[f64; 2]>,
) -> std::borrow::Cow<'a, OrientedPatch> {
    use std::borrow::Cow;
    let Some(kp) = keypoint else {
        return Cow::Borrowed(patch);
    };
    let Some([au, av]) = keypoint_localize::seed_offset(patch, view, kp, 1.0, 1.0) else {
        return Cow::Borrowed(patch);
    };
    let center = keypoint_localize::shifted_center(patch, au, av, 1.0, 1.0);
    let mut shifted =
        OrientedPatch::from_center_normal(center, patch.normal(), patch.u_axis, patch.half_extent);
    shifted.w = patch.w;
    Cow::Owned(shifted)
}
