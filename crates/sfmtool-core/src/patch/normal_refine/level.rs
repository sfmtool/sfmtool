// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-level frozen scoring support: the kept views and commonly-valid masked
//! pixels ([`LevelContext`]) computed at a level's center normal and held fixed
//! across that level's candidates.

use nalgebra::Vector3;

use crate::camera::WarpMap;
use crate::patch::cloud::OrientedPatch;

use super::params::{NormalRefineParams, ProjectedImage, MIN_MASK_PIXELS};
use super::prof;
use super::support::{repose_patch, view_render_patch};

/// The frozen scoring support of one grid level: the kept views and the
/// commonly-valid masked pixels (computed at the level's *center* normal and
/// held fixed across that level's candidates, so `Φ` stays continuous and
/// tilts can't shrink the support onto an easy region).
pub(in crate::patch) struct LevelContext {
    /// Indices into the caller's `views` slice.
    pub(in crate::patch) kept: Vec<usize>,
    /// Linear `row * R + col` indices of the masked pixels.
    pub(in crate::patch) pixels: Vec<usize>,
    /// Window weight per masked pixel (parallel to `pixels`).
    pub(in crate::patch) weights: Vec<f64>,
}

/// Per-pixel validity of `patch` in `view` (window support only). When
/// `keypoint` is given, the patch is recentered in-plane so the mask matches
/// where pixels are actually sampled (the keypoint-anchored render).
pub(super) fn view_valid_mask(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    resolution: u32,
    w_full: &[f64],
    keypoint: Option<[f64; 2]>,
) -> Vec<bool> {
    let patch = view_render_patch(patch, view, keypoint);
    let map = prof::MASK
        .time(|| WarpMap::from_patch(&patch, view.camera, view.cam_from_world, resolution));
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

/// Build the frozen support at `center_n`: cull back-facing views, gate each
/// view on its window-weighted valid fraction, intersect the survivors'
/// validity. `None` when fewer than `min_views` views (or too few pixels)
/// survive.
pub(in crate::patch) fn build_level_context(
    base: &OrientedPatch,
    center_n: &Vector3<f64>,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    w_full: &[f64],
    params: &NormalRefineParams,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> Option<LevelContext> {
    let patch = repose_patch(base, center_n);
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
            &patch,
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
        if mass / support_mass >= params.min_valid_fraction {
            kept.push(i);
            masks.push(mask);
        }
    }

    if kept.len() < params.min_views.max(2) as usize {
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
