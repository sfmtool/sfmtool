// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The frozen common support of several views of one patch: the kept views and
//! the commonly-valid masked pixels ([`LevelContext`]), computed once and held
//! fixed for every render scored over it.

use crate::camera::WarpMap;
use crate::patch::cloud::OrientedPatch;

use super::params::{ProjectedImage, MIN_MASK_PIXELS};
use super::support::view_render_patch;

/// The frozen common support of several views of one patch: the kept views and
/// the commonly-valid masked pixels. Every view's render is scored over the
/// same pixel set, so the z-normalized renders live in one space and their
/// inner products are comparable.
pub(in crate::patch) struct LevelContext {
    /// Indices into the caller's `views` slice.
    pub(in crate::patch) kept: Vec<usize>,
    /// Linear `row * R + col` indices of the masked pixels.
    pub(in crate::patch) pixels: Vec<usize>,
    /// Window weight per masked pixel (parallel to `pixels`).
    pub(in crate::patch) weights: Vec<f64>,
}

/// Per-pixel validity of `patch` in `view` (window support only). When
/// `keypoint` is given, the patch is slid onto its ray so the mask matches
/// where pixels are actually sampled (the keypoint-anchored render).
fn view_valid_mask(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    resolution: u32,
    w_full: &[f64],
    keypoint: Option<[f64; 2]>,
) -> Vec<bool> {
    let patch = view_render_patch(patch, view, keypoint);
    let map = WarpMap::from_patch(&patch, view.camera, view.cam_from_world, resolution);
    let r = resolution;
    let mut mask = vec![false; (r as usize) * (r as usize)];
    for row in 0..r {
        for col in 0..r {
            let p = (row * r + col) as usize;
            if w_full[p] > 0.0 && map.is_valid(col, row) {
                mask[p] = true;
            }
        }
    }
    mask
}

/// Build the frozen common support of `patch` over `views`: cull back-facing
/// views, gate each view on its window-weighted valid fraction
/// (`min_valid_fraction`), and intersect the survivors' validity. Every view
/// that passes the gate is kept. `None` when fewer than two views, or fewer
/// than [`MIN_MASK_PIXELS`] commonly-valid pixels, survive.
///
/// `view_keypoints`, when given, is parallel to `views`: a view's mask is built
/// through its keypoint-anchored render ([`view_render_patch`]).
pub(in crate::patch) fn build_level_context(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    w_full: &[f64],
    min_valid_fraction: f64,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> Option<LevelContext> {
    let support_mass: f64 = w_full.iter().filter(|&&w| w > 0.0).sum();
    if support_mass <= 0.0 {
        return None;
    }

    let mut kept = Vec::new();
    let mut masks: Vec<Vec<bool>> = Vec::new();
    for (i, view) in views.iter().enumerate() {
        // Back-face cull before building any warp map.
        if !patch.is_front_facing(view.cam_from_world) {
            continue;
        }
        let mask = view_valid_mask(
            patch,
            view,
            resolution,
            w_full,
            view_keypoints.and_then(|k| k[i]),
        );
        let mass: f64 = mask
            .iter()
            .zip(w_full)
            .filter(|(&m, _)| m)
            .map(|(_, &w)| w)
            .sum();
        if mass / support_mass >= min_valid_fraction {
            kept.push(i);
            masks.push(mask);
        }
    }

    if kept.len() < 2 {
        return None;
    }

    let mut pixels = Vec::new();
    let mut weights = Vec::new();
    for p in 0..w_full.len() {
        if w_full[p] > 0.0 && masks.iter().all(|m| m[p]) {
            pixels.push(p);
            weights.push(w_full[p]);
        }
    }
    if pixels.len() < MIN_MASK_PIXELS {
        return None;
    }
    Some(LevelContext {
        kept,
        pixels,
        weights,
    })
}
