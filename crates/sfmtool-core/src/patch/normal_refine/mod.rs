// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photometric patch-normal refinement.
//!
//! See `specs/core/patch-normal-refinement.md`. Given an [`OrientedPatch`] and
//! the views that observe it, [`refine_patch_normal`] searches the 2-DOF
//! normal (the only thing that affects cross-view consistency) for the plane
//! whose rendered patches agree the most, scored by the consensus all-pairs
//! mean ZNCC. [`refine_patch_cloud_normals`] batches the routine over a [`PatchCloud`]
//! in parallel.
//!
//! The implementation is split across sibling modules: [`params`] (config /
//! result types), [`parameterization`] (sphere exp-map), [`support`] (window /
//! patch placement), [`level`] (per-level frozen support), [`znorm`] (render +
//! z-normalize), [`consensus`] (the `Φ` objective), [`search`] (coarse-to-fine),
//! and [`view_stack`] (the multi-view render substrate). This module wires them
//! together into the public [`refine_patch_normal`] / [`refine_patch_cloud_normals`]
//! entry points.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::patch::cloud::{mean_viewing_normal, OrientedPatch, PatchCloud};
use crate::reconstruction::SfmrReconstruction;

mod consensus;
mod fronto_cache;
mod level;
mod parameterization;
mod params;
pub mod prof;
mod search;
mod support;
mod view_stack;
mod znorm;

// Public API.
pub use parameterization::{exp_map_normal, tangent_basis};
pub use params::{
    CacheMode, NormalRefineParams, NormalRefineResult, Objective, PatchWindow, ProjectedImage,
    Sampler,
};

// Patch-internal helpers consumed by sibling `patch` modules (keypoint_localize,
// keypoint_subpixel, view_selection) at their historical
// `crate::patch::normal_refine::<name>` paths.
pub(in crate::patch) use consensus::{
    irls_view_weights, weighted_unit_template_into, ConsensusScratch,
};
pub(in crate::patch) use level::{build_level_context, LevelContext};
pub(in crate::patch) use params::FLAT_NORM_SQ_EPS;
pub(in crate::patch) use support::{build_support, window_weights, Support};
pub(in crate::patch) use znorm::{normalized_stack, weighted_moments_pub, znormalize_into_kept};

// Internal helpers used by the orchestration below.
use search::{build_final_context, coarse_to_fine, eval_phi, grid_confidence};
use support::repose_patch;
use view_stack::{PatchViewStack, AGREEMENT_SIGMA};

/// Refine one patch's normal photometrically. Takes the patch and returns an
/// updated copy plus diagnostics; see `specs/core/patch-normal-refinement.md`.
///
/// In-plane rotation can't affect photoconsistency, so the routine searches
/// only the 2-DOF normal; it reprojects the input `u_axis` onto each candidate
/// plane (`v = n × u`) and keeps the input's `center`/`half_extent`, so the
/// frame moves as little as the new plane forces and no `up` hint is needed.
///
/// The search is seeded from the patch's current normal and the mean-viewing
/// normal of the supplied views, runs a coarse-to-fine exp-map grid per seed,
/// then scores every seed winner *and* the init normal over one final frozen
/// support — so the returned `photoconsistency` is never below
/// `init_photoconsistency` when both are finite. Patches whose validity gates
/// fail outright are returned unrefined (NaN scores, zero confidence).
///
/// Not idempotent by design: feeding a refined normal back in can improve it
/// further (each pass re-seeds and re-explores), so running to convergence is the
/// thorough setting — this is intentional, not a fixed-point operation.
/// `view_keypoints`, when given, is parallel to `views` (one entry per view, by
/// the view's index): `Some([x, y])` positions that view's patch at the given
/// source-image keypoint instead of the reprojected point center, `None` leaves
/// it centered. Passing `None` for the whole slice (or `view_keypoints = None`)
/// is byte-for-byte the no-keypoint behavior.
pub fn refine_patch_normal(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> NormalRefineResult {
    prof::TOTAL.time(|| refine_patch_normal_impl(patch, views, resolution, params, view_keypoints))
}

fn refine_patch_normal_impl(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> NormalRefineResult {
    let resolution = resolution.max(2);
    if let Some(kps) = view_keypoints {
        debug_assert_eq!(
            kps.len(),
            views.len(),
            "view_keypoints must be parallel to views"
        );
    }
    let init_n = patch.normal();
    let w_full = window_weights(params.window, resolution);
    // The fronto cache is keypoint-aware: `prerender` renders each view's base at
    // its keypoint-anchored center and `eval_phi` recenters candidates to match
    // (the offset is held at the seed normal — a second-order approximation inside
    // the cache's resampling budget), so keypoints use the same `params.cache`
    // (no special-casing in coarse_to_fine).

    let unrefined = |valid_view_count: u32| NormalRefineResult {
        patch: patch.clone(),
        photoconsistency: f64::NAN,
        init_photoconsistency: f64::NAN,
        valid_view_count,
        confidence: 0.0,
        representative: None,
    };

    // A point at infinity has a fixed outward normal (`normalize(-d)`, set by its
    // direction) — there is nothing to refine, and the finite-depth render path
    // is invalid for it. Leave its frame untouched.
    if patch.w == 0.0 {
        return unrefined(0);
    }

    // Degenerate points skip the search outright.
    if views.len() < params.min_views.max(2) as usize {
        return unrefined(0);
    }

    // Seeds: the patch's current normal, plus the mean-viewing normal of the
    // supplied views when it differs.
    let mut seeds = vec![init_n];
    let centers: Vec<Point3<f64>> = views
        .iter()
        .map(|v| v.cam_from_world.inverse_translation_origin())
        .collect();
    let mean_view = mean_viewing_normal(&patch.center, &centers);
    if mean_view.dot(&init_n) < (0.5f64).to_radians().cos() {
        seeds.push(mean_view);
    }

    // Stage 1: coarse-to-fine grid per seed; keep each seed's winner.
    let mut winners: Vec<Vector3<f64>> = Vec::new();
    for seed in &seeds {
        if let Some(n) = coarse_to_fine(
            patch,
            *seed,
            views,
            resolution,
            &w_full,
            params,
            view_keypoints,
        ) {
            winners.push(n);
        }
    }

    // Final pass: one frozen support for the init and all winners.
    let Some((ctx, survivors)) = build_final_context(
        patch,
        &init_n,
        &winners,
        views,
        resolution,
        &w_full,
        params,
        view_keypoints,
    ) else {
        return unrefined(0);
    };
    let valid_view_count = ctx.kept.len() as u32;

    // Final scoring: score the init and every survivor over the same frozen
    // support `ctx`, so Φ is never below the init's. Only the bitmap path needs
    // the per-view render kept around, so it scores through a `PatchViewStack`
    // (retaining the *winner's* stack + consensus weights, which the
    // representative then fuses with no extra render or IRLS pass); the default
    // path stays on the lean masked-only `eval_phi`, paying nothing for a feature
    // it doesn't use. `extra` carries the winner's stack only in the bitmap path.
    type Winner = Option<(Vec<f64>, PatchViewStack)>;
    let score = |n: &Vector3<f64>| -> Option<(f64, Winner)> {
        prof::count(&prof::N_EVAL, 1);
        if params.render_bitmap {
            let stack = PatchViewStack::render(
                &repose_patch(patch, n),
                views,
                &ctx.kept,
                resolution,
                params.sampler,
                view_keypoints,
            );
            let (phi, weights) = stack.score(&ctx, params)?;
            Some((phi, Some((weights, stack))))
        } else {
            let phi = eval_phi(
                patch,
                n,
                &ctx,
                views,
                resolution,
                params,
                params.objective,
                view_keypoints,
            )?;
            Some((phi, None))
        }
    };

    let init = score(&init_n);
    let phi_init = init.as_ref().map(|(phi, _)| *phi);
    let mut best_n = init_n;
    let mut best_phi = phi_init.unwrap_or(f64::NEG_INFINITY);
    // The winner's rendered stack + consensus weights (bitmap path only).
    let mut best: Winner = init.and_then(|(_, extra)| extra);
    let mut improved = false;
    for n in &survivors {
        if let Some((phi, extra)) = score(n) {
            if phi > best_phi {
                best_phi = phi;
                best_n = *n;
                best = extra;
                improved = true;
            }
        }
    }
    if !best_phi.is_finite() {
        return unrefined(valid_view_count);
    }

    // Confidence (optional): an extra source-render stencil around the optimum,
    // un-cached and purely informational, so it is computed only on request.
    let confidence = if params.compute_confidence {
        // Stencil spacing: the final grid spacing of the coarse-to-fine schedule,
        // clamped to a sane angular band.
        let steps = params.init_steps.max(3) as f64;
        let shrink = 2.0 / (steps - 1.0);
        let h = (params.angular_range_deg.to_radians()
            * shrink.powi(params.refine_levels.max(1) as i32))
        .clamp(0.2f64.to_radians(), 5.0f64.to_radians());
        prof::CONFIDENCE.time(|| {
            grid_confidence(
                patch,
                &best_n,
                &ctx,
                views,
                resolution,
                params,
                h,
                view_keypoints,
            )
        })
    } else {
        f64::NAN
    };

    // Representative RGBA texture (optional): fuse the winner's already-rendered
    // view stack — no extra render or IRLS pass (the scoring pass produced both).
    // Unlike scoring (which reads only the masked common support), fusion spans
    // the full R×R grid, filling every pixel a kept view covers. `best` is `Some`
    // exactly in the bitmap path (and then non-empty, since `best_phi` is finite).
    let representative = best
        .as_ref()
        .map(|(weights, stack)| stack.fuse(weights, AGREEMENT_SIGMA));

    NormalRefineResult {
        patch: if improved {
            repose_patch(patch, &best_n)
        } else {
            patch.clone()
        },
        photoconsistency: best_phi,
        init_photoconsistency: phi_init.unwrap_or(f64::NAN),
        valid_view_count,
        confidence,
        representative,
    }
}

/// Batch [`refine_patch_normal`] over a [`PatchCloud`], parallel across
/// patches (rayon). `patch_views[i]` lists, for patch `i`, the indices into
/// `views` of the cameras observing it (see
/// [`view_indices_from_reconstruction`]). Each patch is replaced with
/// its refined copy; the per-patch results are returned in order.
///
/// `patch_view_keypoints`, when given, is parallel to the cloud (one entry per
/// patch); entry `i` is parallel to `patch_views[i]` (one `Option<[x, y]>` per
/// view in that patch's view list, in the same order). `Some([x, y])` positions
/// that view's patch at the keypoint; `None` leaves it at the point center.
/// Passing `None` is byte-for-byte the no-keypoint behavior.
///
/// # Panics
///
/// Panics if `patch_views.len() != cloud.len()` (or `patch_view_keypoints` is
/// given and not parallel to the cloud) or an index is out of range.
pub fn refine_patch_cloud_normals(
    cloud: &mut PatchCloud,
    views: &[ProjectedImage<'_>],
    patch_views: &[Vec<u32>],
    resolution: u32,
    params: &NormalRefineParams,
    patch_view_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
) -> Vec<NormalRefineResult> {
    assert_eq!(
        patch_views.len(),
        cloud.len(),
        "patch_views must be parallel to the cloud"
    );
    if let Some(kps) = patch_view_keypoints {
        assert_eq!(
            kps.len(),
            cloud.len(),
            "patch_view_keypoints must be parallel to the cloud"
        );
    }
    if prof::enabled() {
        prof::reset();
    }
    let wall_start = std::time::Instant::now();
    let results: Vec<NormalRefineResult> = cloud
        .patches
        .par_iter()
        .enumerate()
        .zip(patch_views.par_iter())
        .map(|((i, patch), vidx)| {
            let pv: Vec<ProjectedImage<'_>> = vidx.iter().map(|&i| views[i as usize]).collect();
            let kps = patch_view_keypoints.map(|k| k[i].as_slice());
            refine_patch_normal(patch, &pv, resolution, params, kps)
        })
        .collect();
    if prof::enabled() {
        prof::report(cloud.len(), wall_start.elapsed().as_secs_f64());
    }
    for (p, r) in cloud.patches.iter_mut().zip(&results) {
        *p = r.patch.clone();
    }
    results
}

/// For each patch of `cloud` (linked to `recon` via `point_indexes`), the image
/// indices observing its source 3D point — ready to use as the `patch_views`
/// of [`refine_patch_cloud_normals`] with one [`ProjectedImage`] per reconstruction image.
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
pub fn view_indices_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>> {
    assert_eq!(
        cloud.point_indexes.len(),
        cloud.len(),
        "cloud must carry a point_index per patch"
    );
    cloud
        .point_indexes
        .iter()
        .map(|&p| {
            let p = p as usize;
            recon.tracks[recon.observation_offsets[p]..recon.observation_offsets[p + 1]]
                .iter()
                .map(|o| o.image_index)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests;
