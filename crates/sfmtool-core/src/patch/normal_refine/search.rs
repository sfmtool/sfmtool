// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The coarse-to-fine normal search: the per-candidate source scorer
//! ([`eval_phi`]), the per-seed [`coarse_to_fine`] grid walk (with the
//! fronto-parallel cache), the final frozen support ([`build_final_context`]),
//! and the finite-difference [`grid_confidence`].

use nalgebra::Vector3;

use crate::patch::cloud::OrientedPatch;

use super::consensus::{consensus_phi, ConsensusScratch};
use super::level::{build_level_context, view_valid_mask, LevelContext};
use super::obliquity::{fill_kept_obliquity_priors, fronto_prior};
use super::parameterization::{exp_map_in_basis, tangent_basis};
use super::params::{CacheMode, NormalRefineParams, Objective, ProjectedImage, MIN_MASK_PIXELS};
use super::support::repose_patch;
use super::znorm::{normalized_stack, znormalize_into};
use super::{fronto_cache, prof};

/// Evaluate `Φ` for the candidate normal `n` over the frozen support `ctx`,
/// scoring with `objective` (the search path passes a possibly-cheaper
/// objective than `params.objective`; the final / confidence passes pass
/// `params.objective`).
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_phi(
    base: &OrientedPatch,
    n: &Vector3<f64>,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    view_dirs: &[Vector3<f64>],
    resolution: u32,
    params: &NormalRefineParams,
    objective: Objective,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> Option<f64> {
    prof::count(&prof::N_EVAL, 1);
    let patch = repose_patch(base, n);
    // Obliquity view-weight (A): |v̂·n|^power per kept view, filled before `n` is
    // shadowed by the pixel count below. Inactive (`power == 0`) leaves `priors`
    // empty and heap-free; this un-cached source path already builds a fresh
    // `ConsensusScratch` per call, so a local buffer matches its allocation class.
    let mut priors = Vec::new();
    let priors_active = fill_kept_obliquity_priors(
        &mut priors,
        view_dirs,
        &ctx.kept,
        n,
        params.obliquity_weight_power,
    );
    let (raw, channels) = normalized_stack(
        &patch,
        ctx,
        views,
        resolution,
        params.sampler,
        view_keypoints,
    )?;
    let n = ctx.pixels.len();
    let total_weight: f64 = ctx.weights.iter().sum();
    if total_weight <= 0.0 {
        return None;
    }
    let sqrt_weights: Vec<f32> = ctx.weights.iter().map(|&w| w.sqrt() as f32).collect();
    let mut xs = Vec::new();
    let kept = prof::ZNORM.time(|| {
        znormalize_into(
            &raw,
            ctx.kept.len(),
            channels,
            n,
            &ctx.weights,
            total_weight,
            &sqrt_weights,
            &mut xs,
        )
    })?;
    let mut cons = ConsensusScratch::default();
    let priors = priors_active.then_some(priors.as_slice());
    prof::CONSENSUS
        .time(|| consensus_phi(&xs, ctx.kept.len(), kept, n, objective, priors, &mut cons))
}

/// Coarse-to-fine exp-map grid search from one seed normal. Each level
/// freezes the common-valid mask at its center normal, scans the δ-disk
/// (`‖δ‖ ≤ range`), recenters on the best candidate, and shrinks the cone to
/// one previous grid spacing. Returns the seed's winning normal, or `None`
/// when nothing could be evaluated from this seed.
#[allow(clippy::too_many_arguments)]
pub(super) fn coarse_to_fine(
    base: &OrientedPatch,
    seed: Vector3<f64>,
    views: &[ProjectedImage<'_>],
    view_dirs: &[Vector3<f64>],
    resolution: u32,
    w_full: &[f64],
    params: &NormalRefineParams,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> Option<Vector3<f64>> {
    // At least 3 grid samples per axis: with 2 the only nonzero candidates land
    // on the disk corners and are clamped away, leaving the center as the sole
    // sample (a no-op search).
    let steps = params.init_steps.max(3) as usize;
    let mut center = seed.normalize();
    let mut range = params.angular_range_deg.to_radians().max(1e-4);
    let mut any_eval = false;
    // The search ranks candidates with a possibly-cheaper objective; the final
    // pass re-scores the survivors with `params.objective`.
    let search_obj = params.search_objective();

    // Fronto-parallel cache: render one supersampled base per view here, reused
    // across all levels. `None` (cache off, or the base could not be built for
    // this patch) means score every candidate from the source images instead.
    let cache = match params.cache {
        CacheMode::FrontoParallel => prof::PRERENDER.time(|| {
            fronto_cache::prerender(
                base,
                &center,
                views,
                resolution,
                params.cache_supersample,
                params,
                view_keypoints,
            )
        }),
        CacheMode::Off => None,
    };
    // Reused across every candidate of every level (cache path only).
    let mut scratch = fronto_cache::Scratch::default();

    for _ in 0..params.refine_levels.max(1) {
        let Some(ctx) = build_level_context(
            base,
            &center,
            views,
            resolution,
            w_full,
            params,
            view_keypoints,
        ) else {
            break;
        };
        let (u, v) = tangent_basis(&center);
        // The cache resamples the level's frozen support; its grid coords and
        // window weights are candidate-independent, so compute them once per level.
        let (cols, rows): (Vec<i32>, Vec<i32>) = if cache.is_some() {
            ctx.pixels
                .iter()
                .map(|&p| {
                    (
                        (p % resolution as usize) as i32,
                        (p / resolution as usize) as i32,
                    )
                })
                .unzip()
        } else {
            (Vec::new(), Vec::new())
        };
        let (sqrt_weights, total_weight): (Vec<f32>, f64) = if cache.is_some() {
            (
                ctx.weights.iter().map(|&w| w.sqrt() as f32).collect(),
                ctx.weights.iter().sum(),
            )
        } else {
            (Vec::new(), 0.0)
        };
        // Rank candidates by Φ plus the fronto-parallel prior (B); the prior is a
        // geometric function of the candidate normal (independent of the render),
        // so it is added to each Φ here. It only tips near-ties where Φ is flat.
        let mut eval = |n: &Vector3<f64>| -> Option<f64> {
            let phi = match &cache {
                Some(c) => fronto_cache::eval_phi(
                    base,
                    n,
                    c,
                    &ctx,
                    views,
                    view_dirs,
                    resolution,
                    &cols,
                    &rows,
                    &sqrt_weights,
                    total_weight,
                    params.obliquity_weight_power,
                    &mut scratch,
                    search_obj,
                ),
                None => eval_phi(
                    base,
                    n,
                    &ctx,
                    views,
                    view_dirs,
                    resolution,
                    params,
                    search_obj,
                    view_keypoints,
                ),
            }?;
            Some(phi + fronto_prior(view_dirs, n, params.fronto_prior_weight))
        };
        let mut best_n = center;
        let mut best_phi = f64::NEG_INFINITY;
        if let Some(phi) = eval(&center) {
            best_phi = phi;
        }
        for i in 0..steps {
            let a = -range + 2.0 * range * i as f64 / (steps - 1) as f64;
            for j in 0..steps {
                let b = -range + 2.0 * range * j as f64 / (steps - 1) as f64;
                if a == 0.0 && b == 0.0 {
                    continue; // Center already evaluated.
                }
                // Clamp the square grid to the δ-disk (circular cone).
                if a.hypot(b) > range * (1.0 + 1e-9) {
                    continue;
                }
                let n = exp_map_in_basis(&center, &u, &v, [a, b]);
                if let Some(phi) = eval(&n) {
                    if phi > best_phi {
                        best_phi = phi;
                        best_n = n;
                    }
                }
            }
        }
        if !best_phi.is_finite() {
            break;
        }
        any_eval = true;
        center = best_n;
        // Next cone: ± one grid spacing of this level.
        range = (2.0 * range / (steps - 1) as f64).max(1e-4);
    }
    any_eval.then_some(center)
}

/// Final-pass support: kept views and mask frozen at the *init* normal, with
/// the mask further intersected with each surviving winner's validity, so the
/// init and every winner are scored over the same pixel set. Winners that are
/// back-facing in a kept view are discarded. Returns the context and the
/// surviving winners.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_final_context(
    base: &OrientedPatch,
    init_n: &Vector3<f64>,
    winners: &[Vector3<f64>],
    views: &[ProjectedImage<'_>],
    resolution: u32,
    w_full: &[f64],
    params: &NormalRefineParams,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> Option<(LevelContext, Vec<Vector3<f64>>)> {
    let mut ctx = build_level_context(
        base,
        init_n,
        views,
        resolution,
        w_full,
        params,
        view_keypoints,
    )?;
    let mut survivors = Vec::new();
    for n in winners {
        let patch = repose_patch(base, n);
        if !ctx
            .kept
            .iter()
            .all(|&i| patch.is_front_facing(views[i].cam_from_world))
        {
            continue;
        }
        let mut pixels = Vec::with_capacity(ctx.pixels.len());
        let mut weights = Vec::with_capacity(ctx.pixels.len());
        let masks: Vec<Vec<bool>> = ctx
            .kept
            .iter()
            .map(|&i| {
                view_valid_mask(
                    &patch,
                    &views[i],
                    resolution,
                    w_full,
                    view_keypoints.and_then(|k| k[i]),
                )
            })
            .collect();
        for (k, &p) in ctx.pixels.iter().enumerate() {
            if masks.iter().all(|m| m[p]) {
                pixels.push(p);
                weights.push(ctx.weights[k]);
            }
        }
        if pixels.len() < MIN_MASK_PIXELS {
            continue; // This winner would leave too little common support.
        }
        ctx.pixels = pixels;
        ctx.weights = weights;
        survivors.push(*n);
    }
    Some((ctx, survivors))
}

/// Finite-difference confidence: the curvature of `Φ(δ)` at the optimum,
/// estimated from a 3×3 stencil of grid samples (spacing `h`). The smaller
/// eigenvalue of the negated 2×2 Hessian measures how tightly the normal is
/// constrained; it is normalized against the larger eigenvalue plus an
/// absolute floor, so a flat `Φ` (the narrow-baseline degeneracy, where
/// tilting the plane shifts all patches identically) reports ≈ 0 and an
/// isotropically peaked `Φ` reports ≈ 1.
///
/// The spec's analytic *centered Gauss-Newton Hessian*
/// (`H̃ = Σ wᵢ JᵢᵀJᵢ − J̄ᵀJ̄`) is the fast-follow; this grid estimate is
/// GN-free and captures the same degeneracy because `Φ` itself is genuinely
/// flat there.
#[allow(clippy::too_many_arguments)]
pub(super) fn grid_confidence(
    base: &OrientedPatch,
    n: &Vector3<f64>,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    view_dirs: &[Vector3<f64>],
    resolution: u32,
    params: &NormalRefineParams,
    h: f64,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> f64 {
    let (u, v) = tangent_basis(n);
    // Pure-Φ curvature: the obliquity view-weight (A) is part of the objective and
    // flows through `eval_phi`, but the fronto prior (B) is deliberately excluded —
    // confidence must report how tightly the *data* pins the normal, not how hard
    // the prior pulls it fronto-parallel (which would fake confidence on flat Φ).
    let phi = |a: f64, b: f64| -> Option<f64> {
        eval_phi(
            base,
            &exp_map_in_basis(n, &u, &v, [a, b]),
            ctx,
            views,
            view_dirs,
            resolution,
            params,
            params.objective,
            view_keypoints,
        )
    };
    let stencil = [
        phi(0.0, 0.0),
        phi(h, 0.0),
        phi(-h, 0.0),
        phi(0.0, h),
        phi(0.0, -h),
        phi(h, h),
        phi(h, -h),
        phi(-h, h),
        phi(-h, -h),
    ];
    let Some(vals) = stencil.into_iter().collect::<Option<Vec<f64>>>() else {
        return 0.0;
    };
    let [f00, fp0, fm0, f0p, f0m, fpp, fpm, fmp, fmm] = vals[..] else {
        return 0.0;
    };
    let h2 = h * h;
    // Negated Hessian of Φ (positive curvature at a maximum).
    let haa = -(fp0 - 2.0 * f00 + fm0) / h2;
    let hbb = -(f0p - 2.0 * f00 + f0m) / h2;
    let hab = -(fpp - fpm - fmp + fmm) / (4.0 * h2);
    let mean = 0.5 * (haa + hbb);
    let disc = (0.25 * (haa - hbb) * (haa - hbb) + hab * hab).sqrt();
    let lmax = (mean + disc).max(0.0);
    let lmin = (mean - disc).max(0.0);
    if !lmin.is_finite() || !lmax.is_finite() {
        return 0.0;
    }
    // Scale normalization: dimensionless in texture contrast (Φ is already a
    // correlation); the +1 floor (Φ per radian²) keeps weakly-curved optima
    // from reporting full confidence.
    lmin / (lmax + 1.0)
}
