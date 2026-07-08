// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photometric patch-view selection.
//!
//! See `specs/core/patch-view-selection.md`. Given one 3D point with its
//! oriented patch and the track views that already observe it,
//! [`select_patch_views`] returns the **view set** `G` — the track views plus
//! every other image that *photometrically* sees the patch.
//!
//! The pipeline is the same render → z-normalize → robust-consensus machinery
//! as [normal refinement](super::normal_refine): the track views' renders are
//! fused into a robust (IRLS) **reference** appearance, then each
//! geometrically-visible **candidate** is admitted when its windowed ZNCC to
//! the reference clears `min_relative_zncc ×` the track's own self-agreement
//! (the mean ZNCC of the track views to the reference). Track views are always
//! admitted. When the track's self-agreement is below `min_self_agreement` (no
//! trustworthy reference) the track is admitted verbatim with no expansion.
//!
//! **Affine candidate scoring.** The gate score exists only to admit/reject —
//! nothing downstream reuses the candidate render — so scoring a view does not
//! need the full per-pixel projective warp. With the bilinear sampler, a
//! candidate is scored through an **affine** patch→image map fit on its four
//! exactly-projected patch corners ([`affine_core_map`]), sampling only the
//! reference-support pixels; the exact warp remains the fallback whenever the
//! 4th-corner residual shows the affine fit is poor (wide-angle / heavy
//! distortion), a corner fails to project, or the patch comes close to the
//! frame border (where the exact path owns the out-of-frame rejection
//! semantics). See `specs/core/patch-view-selection.md` for the accepted
//! admission-flip loss and the measured numbers.

use crate::camera::remap::ImageU8;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::normal_refine::{
    build_level_context, irls_view_weights, normalized_stack, weighted_moments_pub, window_weights,
    znormalize_into_kept, ConsensusScratch, LevelContext, PatchWindow, ProjectedImage, Sampler,
    FLAT_NORM_SQ_EPS,
};
use crate::reconstruction::SfmrReconstruction;
use rayon::prelude::*;

pub mod prof;

/// Tunables for [`select_patch_views`].
///
/// The render/window/validity knobs mirror
/// [`NormalRefineParams`](super::normal_refine::NormalRefineParams) so the
/// reference appearance is built on the same conventions as refinement.
#[derive(Debug, Clone)]
pub struct ViewSelectParams {
    /// Admit a candidate whose windowed ZNCC to the reference clears this
    /// fraction of the track's own self-agreement (the track views' mean ZNCC to
    /// the reference). Track views are always admitted regardless.
    pub min_relative_zncc: f64,
    /// The `R×R` patch grid the reference and candidate ZNCC are scored on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// How to sample the source pyramids when rendering patches.
    pub sampler: Sampler,
    /// Per-view floor on the window-weighted valid-pixel fraction; a candidate
    /// (or track view) below it does not cover enough of the patch to be scored.
    pub min_valid_fraction: f64,
    /// Minimum number of *valid* track views (those passing the per-view validity
    /// gate over the common support) needed to build a reference. A point whose
    /// valid track-view count falls below this admits its track views verbatim
    /// with no candidate vetting (not enough cross-view evidence to vet against).
    pub min_track_views: u32,
    /// IRLS reweighting passes for the robust reference consensus (down-weights
    /// occluded / wrong-surface track views).
    pub robust_iters: u32,
    /// Trust gate on the track's self-agreement: when the track views' mean ZNCC
    /// to the reference is below this, there is no trustworthy reference appearance
    /// to vet against, so the track is admitted verbatim and **no** candidates are
    /// added. When self-agreement meets the gate, the admission bar for a candidate
    /// is `min_relative_zncc × self_agreement`.
    pub min_self_agreement: f64,
}

impl Default for ViewSelectParams {
    fn default() -> Self {
        Self {
            min_relative_zncc: 0.7,
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: Sampler::Bilinear,
            min_valid_fraction: 0.6,
            min_track_views: 2,
            robust_iters: 3,
            min_self_agreement: 0.3,
        }
    }
}

/// The selected view set `G` for one point.
#[derive(Debug, Clone, Default)]
pub struct ViewSelection {
    /// Admitted image indices (into the `views` slice): the track views plus the
    /// photometrically-vetted candidates. Track views come first (in their input
    /// order), then the admitted extra views in ascending index order.
    pub admitted: Vec<u32>,
    /// Windowed ZNCC of each admitted view to the reference, parallel to
    /// [`admitted`](Self::admitted). `NaN` for a view that could not be scored
    /// against the reference (e.g. a track view admitted unconditionally whose
    /// render did not cover the reference support).
    pub scores: Vec<f64>,
    /// The track's self-agreement: the mean ZNCC of the track views to the
    /// reference. `NaN` when no reference could be built (track too small / no
    /// common support), in which case the track views are admitted verbatim. When
    /// finite but below `min_self_agreement`, the reference is not trusted: the
    /// track is still admitted verbatim (no candidate expansion) and the measured
    /// value is reported here.
    pub self_agreement: f64,
}

/// The robust reference appearance built from a point's track views: the
/// per-channel, per-pixel IRLS-weighted consensus over the frozen common
/// support, re-normalized to unit windowed-norm per channel so a plain dot
/// product against another z-normalized render is a windowed ZNCC.
struct Reference {
    /// The frozen scoring support (kept track views + common masked pixels).
    ctx: LevelContext,
    /// Unit-norm reference per kept channel, flat `[c * n + pixel]` (the `√w`
    /// window weighting is folded into the z-normalized space the candidates are
    /// scored in, so this is directly dot-able). Length `channels * n`, indexed by
    /// the *compacted* channel order; [`kept_channels`](Self::kept_channels) maps
    /// each slot back to its original source channel.
    template: Vec<f32>,
    /// The *original* source-channel index of each compacted template channel
    /// (length `channels`). A candidate is scored on exactly these original
    /// channels, so the reference's channel A is never correlated against a
    /// candidate's channel B even when the two drop different flat channels.
    kept_channels: Vec<usize>,
    /// Channel count of the consensus space (`kept_channels.len()`).
    channels: usize,
    /// Number of support pixels (`ctx.pixels.len()`).
    n: usize,
}

/// Build the robust reference from the track views and report the track's
/// self-agreement. `track_views` indexes into `views`. `None` when no reference
/// could be built (fewer than `min_track_views` track views survive the validity
/// gates, or there is no common support / textured channel).
fn build_reference(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    track_views: &[u32],
    w_full: &[f64],
    params: &ViewSelectParams,
) -> Option<(Reference, f64)> {
    let n_normal = patch.normal();
    // Restrict to the track views, preserving order; build the frozen support at
    // the point's own normal exactly as refinement does.
    let track_proj: Vec<ProjectedImage<'_>> =
        track_views.iter().map(|&i| views[i as usize]).collect();

    let ctx = prof::REF_SUPPORT.time(|| {
        build_level_context(
            patch,
            &n_normal,
            &track_proj,
            params.resolution,
            w_full,
            &normal_refine_shim(params),
            None,
        )
    })?;
    if ctx.kept.len() < params.min_track_views.max(2) as usize {
        return None;
    }

    // Render + z-normalize the kept track views over the frozen support.
    let (raw, channels) = prof::REF_RENDER.time(|| {
        normalized_stack(
            patch,
            &ctx,
            &track_proj,
            params.resolution,
            params.sampler,
            None,
        )
    })?;
    let n = ctx.pixels.len();
    let total_weight: f64 = ctx.weights.iter().sum();
    if total_weight <= 0.0 {
        return None;
    }
    let sqrt_weights: Vec<f32> = ctx.weights.iter().map(|&w| w.sqrt() as f32).collect();
    prof::REF_CONSENSUS.time(|| {
        let mut xs = Vec::new();
        let (kept, keep_mask) = znormalize_into_kept(
            &raw,
            ctx.kept.len(),
            channels,
            n,
            &ctx.weights,
            total_weight,
            &sqrt_weights,
            &mut xs,
        )?;
        // The *original* source-channel index of each compacted reference channel, so
        // a candidate is scored on the reference's channel identity (not its own
        // post-compaction channel order).
        let kept_channels: Vec<usize> = keep_mask
            .iter()
            .enumerate()
            .filter_map(|(c, &k)| k.then_some(c))
            .collect();
        let views_kept = ctx.kept.len();

        // Robust per-view weights for the reference consensus (same IRLS as Φ).
        let mut sc = ConsensusScratch::default();
        irls_view_weights(&xs, views_kept, kept, n, params.robust_iters, None, &mut sc);
        let weights = &sc.w;

        // Weighted consensus template per channel, then unit-normalize per channel so
        // a dot product against another z-normalized render is a ZNCC. The stack `xs`
        // already carries the √w window weighting, so the template lives in the same
        // windowed space and no further weighting is needed.
        let mut template = vec![0f32; kept * n];
        for (v, &w) in weights.iter().enumerate().take(views_kept) {
            let wv = w as f32;
            for c in 0..kept {
                let src = &xs[(v * kept + c) * n..][..n];
                let dst = &mut template[c * n..][..n];
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d += wv * s;
                }
            }
        }
        for c in 0..kept {
            let col = &mut template[c * n..][..n];
            let norm = (col.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt();
            if norm > 1e-12 {
                let inv = (1.0 / norm) as f32;
                for x in col.iter_mut() {
                    *x *= inv;
                }
            }
        }

        let reference = Reference {
            ctx,
            template,
            kept_channels,
            channels: kept,
            n,
        };

        // Self-agreement: the mean ZNCC of the kept track views to the reference. The
        // template is unit-norm per channel and each view's z-normalized column is
        // unit-norm too, so the per-channel dot is a correlation; average over
        // channels, then over views. `views_kept >= 2` (gated above), so the mean is
        // always well-defined.
        let mut agree_sum = 0.0;
        for v in 0..views_kept {
            let mut s = 0.0;
            for c in 0..kept {
                let row = &xs[(v * kept + c) * n..][..n];
                let tmpl = &reference.template[c * n..][..n];
                s += row
                    .iter()
                    .zip(tmpl)
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum::<f64>();
            }
            agree_sum += s / kept as f64;
        }
        let self_agreement = agree_sum / views_kept as f64;

        Some((reference, self_agreement))
    })
}

/// Maximum 4th-corner residual (source px) below which a candidate's
/// patch→image map is scored by the **affine** fast path instead of the exact
/// per-pixel projective warp. The residual measures the **asymmetric**
/// component of the projective + lens-distortion curvature over the patch
/// (three corners fit the affine exactly; the fourth is where an asymmetric
/// deviation shows). It does NOT bound curvature that is symmetric about the
/// patch centre — e.g. radial distortion around a patch near the principal
/// point displaces all four corners symmetrically, cancelling in the residual
/// while edge-midpoint error persists — so interior position error is bounded
/// only for the asymmetric class; the symmetric remainder folds into the
/// accepted admission-flip loss. 0.5 px keeps the value
/// error inside the sampler's own bilinear blur (measured on dino: mean
/// per-point admitted-set Jaccard vs the exact warp stays ≥ 0.994) while
/// letting the large-patch candidates — whose projective curvature over a
/// 100+ px quad exceeds a tighter bound — still take the fast path; wide-angle
/// / heavily distorted views exceed it and fall back to the exact warp.
const AFFINE_MAX_RESIDUAL_PX: f64 = 0.5;

/// Frame-border margin (source px): the affine fast path requires the mapped
/// patch quad to stay this far inside the frame, so (a) every interior sample
/// is safely in-bounds despite the ≤ [`AFFINE_MAX_RESIDUAL_PX`] position
/// error, and (b) the exact path keeps sole authority over the
/// out-of-frame-support rejection semantics near the border.
const AFFINE_BORDER_MARGIN_PX: f64 = 1.0 + AFFINE_MAX_RESIDUAL_PX;

/// The affine patch-grid → source-pixel map of the candidate fast path:
/// `x = a[0]·col + a[1]·row + a[2]`, `y = a[3]·col + a[4]·row + a[5]` (grid
/// pixel indices, matching `WarpMap::from_patch`'s pixel-center raster).
struct AffineCoreMap {
    a: [f64; 6],
}

/// Fit the candidate's patch→image map from three **exactly projected** grid
/// corners and vet it: `None` (→ exact warp) when a corner fails to project,
/// the 4th-corner residual exceeds [`AFFINE_MAX_RESIDUAL_PX`], or the mapped
/// quad comes within [`AFFINE_BORDER_MARGIN_PX`] of the frame border. Uses the
/// same affine camera-frame ray basis as `WarpMap::from_patch` (`w = 0` folds
/// into the origin identically), so the corner projections are exactly the
/// warp's own corner samples.
fn affine_core_map(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    resolution: u32,
) -> Option<AffineCoreMap> {
    let rot = view.cam_from_world.rotation.to_rotation_matrix();
    let q0 = rot * patch.center.coords + view.cam_from_world.translation * patch.w;
    let qu = (rot * patch.u_axis) * patch.half_extent[0];
    let qv = (rot * patch.v_axis) * (-patch.half_extent[1]);
    let step = 2.0 / resolution as f64;
    let s0 = 0.5 * step - 1.0;
    let origin = q0 + (qu + qv) * s0;
    let cs = qu * step;
    let rs = qv * step;
    let r1 = (resolution.max(2) - 1) as f64;
    let project_corner = |c: f64, r: f64| -> Option<(f64, f64)> {
        let ray = origin + cs * c + rs * r;
        view.camera.ray_to_pixel([ray.x, ray.y, ray.z])
    };
    let p00 = project_corner(0.0, 0.0)?;
    let p10 = project_corner(r1, 0.0)?;
    let p01 = project_corner(0.0, r1)?;
    let p11 = project_corner(r1, r1)?;

    // 4th-corner residual: how far the true (projective + distorted) map
    // deviates from the 3-corner affine over the patch.
    let p11_pred = (p10.0 + p01.0 - p00.0, p10.1 + p01.1 - p00.1);
    let resid = (p11.0 - p11_pred.0).hypot(p11.1 - p11_pred.1);
    if resid > AFFINE_MAX_RESIDUAL_PX {
        return None;
    }

    // Border gate: the affine image of the grid square is the parallelogram
    // {p00, p10, p01, p11_pred} (true p11 included for the residual slack);
    // its bounding box inside the margin keeps every interior sample
    // in-frame.
    let (w, h) = (view.camera.width as f64, view.camera.height as f64);
    let xs = [p00.0, p10.0, p01.0, p11.0, p11_pred.0];
    let ys = [p00.1, p10.1, p01.1, p11.1, p11_pred.1];
    let m = AFFINE_BORDER_MARGIN_PX;
    let inside = xs.iter().all(|&x| x >= m && x <= w - 1.0 - m)
        && ys.iter().all(|&y| y >= m && y <= h - 1.0 - m);
    if !inside {
        return None;
    }

    Some(AffineCoreMap {
        a: [
            (p10.0 - p00.0) / r1,
            (p01.0 - p00.0) / r1,
            p00.0,
            (p10.1 - p00.1) / r1,
            (p01.1 - p00.1) / r1,
            p00.1,
        ],
    })
}

/// Sample the reference-support pixels of `img` through the affine map into
/// the planar `out` (`[channel · n + support_index]`, `f32`), with exactly
/// [`remap_bilinear`](crate::camera::remap::remap_bilinear)'s per-sample value
/// convention (same `x − 0.5` pixel-center geometry, same bilinear blend
/// grouping, same `u8` rounding), so where the affine position matches the
/// exact warp the sampled value is bit-identical.
///
/// The inner loop is hand-rolled rather than calling the generic
/// [`sample_bilinear_u8_all`]: the [`AFFINE_BORDER_MARGIN_PX`] gate on the map
/// guarantees every sample sits ≥ 1 px inside the frame, so the per-pixel
/// floor/clamp/saturate edge handling drops out entirely (the generic sampler
/// is kept as the reference the `affine_sampler_matches_reference` test pins
/// this against).
fn sample_support_affine(
    img: &ImageU8,
    map: &AffineCoreMap,
    pixels: &[usize],
    resolution: usize,
    out: &mut [f32],
) {
    let n = pixels.len();
    let ch = img.channels() as usize;
    let stride = img.width() as usize * ch;
    let data = img.data();
    let a = &map.a;
    for (k, &p) in pixels.iter().enumerate() {
        let col = (p % resolution) as f64;
        let row = (p / resolution) as f64;
        // Same pixel-center convention as `bilinear_geometry` (x − 0.5).
        let gx = (a[0] * col + a[1] * row + a[2] - 0.5) as f32;
        let gy = (a[3] * col + a[4] * row + a[5] - 0.5) as f32;
        let x0 = gx.floor();
        let y0 = gy.floor();
        let fx = gx - x0;
        let fy = gy - y0;
        // Border-gated: x0/y0 are in-frame with the +1 neighbour, no clamping.
        let base = (y0 as usize) * stride + (x0 as usize) * ch;
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;
        let top = &data[base..base + 2 * ch];
        let bot = &data[base + stride..base + stride + 2 * ch];
        for c in 0..ch {
            let val = w00 * top[c] as f32
                + w10 * top[ch + c] as f32
                + w01 * bot[c] as f32
                + w11 * bot[ch + c] as f32;
            // remap_bilinear's u8 rounding, kept for value parity.
            out[c * n + k] = (val + 0.5).clamp(0.0, 255.0).floor();
        }
    }
    // Keep the shared resample counters honest (one call, n samples, one
    // bilinear tap per sample per channel — same accounting as
    // `remap_bilinear`, which the exact path routes through).
    crate::camera::remap::prof::add(&crate::camera::remap::prof::CALLS, 1);
    crate::camera::remap::prof::add(&crate::camera::remap::prof::PX_TOTAL, n as u64);
    crate::camera::remap::prof::add(&crate::camera::remap::prof::PX_SAMPLED, n as u64);
    crate::camera::remap::prof::add(&crate::camera::remap::prof::TAPS, (n * ch) as u64);
}

/// Whether `patch`'s centre is in front of the `cam_from_world` camera —
/// camera-frame depth `−z > 0` (the canonical camera looks down −Z).
/// [`OrientedPatch::is_front_facing`] only tests the normal vs. the camera
/// centre and does **not** guarantee positive depth; equirect / wide-fisheye
/// projection can map behind-camera points in-frame, so scoring needs this
/// explicit cheirality gate (camera-model agnostic).
///
/// For a point at infinity (`w = 0`) the test is on the ray direction `R·d`: the
/// camera sees the point iff it looks toward `+d` (camera-frame `z < 0`).
fn is_in_front(patch: &OrientedPatch, cam_from_world: &crate::geometry::RigidTransform) -> bool {
    cam_from_world
        .transform_point_homogeneous(patch.center.coords, patch.w)
        .z
        < 0.0
}

/// Windowed ZNCC of `patch` rendered in `view` against the reference template,
/// over the reference's frozen support. `None` when the render does not cover the
/// reference support (a masked pixel falls out of frame) — the candidate is then
/// not scoreable and is rejected.
///
/// `single_ctx` is the reference's frozen support re-expressed as a one-view
/// context (`kept = [0]`, the reference's `pixels` / `weights`); it is built once
/// by the caller and shared across candidates, so scoring a candidate clones no
/// per-call support. `sqrt_weights` is `√weight` per support pixel (also caller-built).
///
/// **Channel alignment (A1).** The candidate is rendered into the *full* original
/// channel stack and scored only on the reference's surviving *original* channels
/// ([`Reference::kept_channels`]). Each such channel is z-normalized in place and
/// correlated against the matching reference template column; a channel that is
/// flat in the candidate (windowed norm² below [`FLAT_NORM_SQ_EPS`]) contributes
/// `0` rather than a misaligned dot. This anchors the score to the reference's
/// channel identity, so the reference's channel A is never correlated against a
/// candidate's channel B.
fn candidate_zncc(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    reference: &Reference,
    single_ctx: &LevelContext,
    sqrt_weights: &[f32],
    params: &ViewSelectParams,
    raw_scratch: &mut Vec<f32>,
) -> Option<f64> {
    // Affine fast path (bilinear sampler only — the anisotropic footprint has
    // no affine shortcut): project the four patch corners exactly, fit the
    // affine patch→image map, and sample the reference support through it —
    // skipping the per-pixel camera projection + distortion of the full warp.
    // A view the fit declines (corner fails to project, residual over the
    // curvature bound, or too close to the frame border) is scored by the
    // exact warp below, so the fast path never decides rejection on its own.
    if matches!(params.sampler, Sampler::Bilinear) {
        let map = prof::AFFINE_MAP.time(|| affine_core_map(patch, view, params.resolution));
        if let Some(map) = map {
            prof::count(&prof::N_AFFINE, 1);
            let img = view.pyramid.level(0);
            let channels = img.channels() as usize;
            let n = reference.n;
            raw_scratch.clear();
            raw_scratch.resize(channels * n, 0.0);
            prof::AFFINE_SAMPLE.time(|| {
                sample_support_affine(
                    img,
                    &map,
                    &single_ctx.pixels,
                    params.resolution as usize,
                    raw_scratch,
                )
            });
            return score_raw_against_reference(
                raw_scratch,
                channels,
                reference,
                single_ctx,
                sqrt_weights,
            );
        }
        prof::count(&prof::N_AFFINE_FALLBACK, 1);
    }

    let single = [*view];
    // Render over the reference's frozen support. `normalized_stack` returns the
    // *raw* (un-compacted) per-original-channel pixel stack and rejects (returns
    // `None`) if any support pixel is out of frame in this view. The first slot of
    // its tuple is `[(0*channels + c)*n + pixel]` for the single view.
    let (raw, channels) = prof::ZNCC_RENDER.time(|| {
        normalized_stack(
            patch,
            single_ctx,
            &single,
            params.resolution,
            params.sampler,
            None,
        )
    })?;
    score_raw_against_reference(&raw, channels, reference, single_ctx, sqrt_weights)
}

/// Score a raw support stack (`[channel · n + pixel]`, the scored view's
/// original channels) against the reference template — the shared back half of
/// [`candidate_zncc`] (affine and exact paths).
///
/// For each of the reference's surviving original channels, z-normalize the
/// candidate's matching channel directly and dot it against the reference
/// template column (which already carries the `√w` window weighting and is
/// unit-norm). A flat candidate channel contributes 0 (no misaligned dot).
fn score_raw_against_reference(
    raw: &[f32],
    channels: usize,
    reference: &Reference,
    single_ctx: &LevelContext,
    sqrt_weights: &[f32],
) -> Option<f64> {
    let n = reference.n;
    let total_weight: f64 = single_ctx.weights.iter().sum();
    if total_weight <= 0.0 {
        return None;
    }
    prof::ZNCC_DOT.time(|| {
        let mut s = 0.0;
        let mut scratch = vec![0f32; n];
        for (tc, &orig_c) in reference.kept_channels.iter().enumerate() {
            if orig_c >= channels {
                // The candidate image has fewer channels than the reference space; that
                // channel is undefined here — no contribution.
                continue;
            }
            let col = &raw[orig_c * n..][..n];
            let (s1, s2) = weighted_moments_pub(col, &single_ctx.weights);
            let mean = (s1 / total_weight) as f32;
            let norm_sq = s2 - s1 * (mean as f64);
            if norm_sq < FLAT_NORM_SQ_EPS {
                continue; // Flat in the candidate -> 0 contribution, not a bad dot.
            }
            let inv_norm = (1.0 / norm_sq.sqrt()) as f32;
            for (out, (&x, &sw)) in scratch.iter_mut().zip(col.iter().zip(sqrt_weights)) {
                *out = sw * (x - mean) * inv_norm;
            }
            let tmpl = &reference.template[tc * n..][..n];
            s += scratch
                .iter()
                .zip(tmpl)
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum::<f64>();
        }
        // Average over the reference's channel count, mirroring the self-agreement
        // convention (channels absent / flat in the candidate score 0 in the mean).
        Some(s / reference.channels as f64)
    })
}

/// A `NormalRefineParams` shim carrying just the gating knobs
/// [`build_level_context`] and [`normalized_stack`] read, so view selection
/// drives the shared support/render machinery without re-deriving it.
fn normal_refine_shim(params: &ViewSelectParams) -> super::normal_refine::NormalRefineParams {
    super::normal_refine::NormalRefineParams {
        window: params.window,
        sampler: params.sampler,
        min_valid_fraction: params.min_valid_fraction,
        // `build_level_context` floors the kept-view count at `min_views.max(2)`;
        // we want it to keep every track view that passes the per-view fraction
        // gate (the `min_track_views` check happens in `build_reference`), so set
        // the floor to 2 (its minimum) here.
        min_views: 2,
        ..super::normal_refine::NormalRefineParams::default()
    }
}

/// Select the view set `G` for one oriented patch.
///
/// `views` is one [`ProjectedImage`] per reconstruction image (indexed by image
/// index); `track_views` lists the image indices that already observe the point
/// (its track). Returns the admitted view indices and their ZNCC scores; see
/// [`ViewSelection`] and `specs/core/patch-view-selection.md`.
///
/// The candidate set is the track views plus every other image that
/// geometrically sees the surfel (front-facing patch, point in front of the
/// camera, projects in-frame with enough coverage). A reference appearance is
/// fused from the track views; a candidate is admitted when its windowed ZNCC to
/// the reference clears `min_relative_zncc ×` the track's self-agreement. Track
/// views are always admitted. When no reference can be built (track too small or
/// degenerate) **or** the track's self-agreement is below `min_self_agreement`
/// (no trustworthy reference to vet against), the track views are admitted
/// verbatim and no extra candidates are added.
pub fn select_patch_views(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    track_views: &[u32],
    params: &ViewSelectParams,
) -> ViewSelection {
    prof::TOTAL.time(|| select_patch_views_impl(patch, views, track_views, params))
}

/// Untimed body of [`select_patch_views`] (split so the enclosing
/// [`prof::TOTAL`] phase is a single wrap).
fn select_patch_views_impl(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    track_views: &[u32],
    params: &ViewSelectParams,
) -> ViewSelection {
    let resolution = params.resolution.max(2);
    let params = ViewSelectParams {
        resolution,
        ..params.clone()
    };
    let w_full = window_weights(params.window, resolution);

    // Dedup the track order-preserving (first-seen wins): a point can carry two
    // observations in the same image (rigs / retriangulation), which would admit
    // that view twice and double-weight it in the reference.
    let mut seen = std::collections::HashSet::new();
    let track_views: Vec<u32> = track_views
        .iter()
        .copied()
        .filter(|&i| seen.insert(i))
        .collect();
    let is_track = seen; // the dedup set doubles as the membership test

    // Build the robust reference from the track views.
    let reference =
        prof::REFERENCE.time(|| build_reference(patch, views, &track_views, &w_full, &params));

    let admit_verbatim = || ViewSelection {
        admitted: track_views.clone(),
        scores: vec![f64::NAN; track_views.len()],
        self_agreement: f64::NAN,
    };

    let Some((reference, self_agreement)) = reference else {
        // No reference: admit the track views verbatim, no candidate vetting.
        prof::count(&prof::N_VERBATIM, 1);
        return admit_verbatim();
    };

    // The reference's frozen support re-expressed as a one-view context, built
    // once and shared across every scored view (the candidate fills view slot 0).
    let single_ctx = LevelContext {
        kept: vec![0],
        pixels: reference.ctx.pixels.clone(),
        weights: reference.ctx.weights.clone(),
    };
    let sqrt_weights: Vec<f32> = single_ctx
        .weights
        .iter()
        .map(|&w| w.sqrt() as f32)
        .collect();

    // Reused raw-sample buffer for the affine fast path (one allocation per
    // point, shared across every scored view).
    let mut raw_scratch: Vec<f32> = Vec::new();

    // Track views: always admitted; score them against the reference for the
    // returned diagnostics (NaN when they can't be scored, e.g. dropped by the
    // per-view validity gate).
    let mut admitted: Vec<u32> = Vec::with_capacity(track_views.len());
    let mut scores: Vec<f64> = Vec::with_capacity(track_views.len());
    for &ti in &track_views {
        admitted.push(ti);
        scores.push(
            prof::TRACK_SCORE
                .time(|| {
                    candidate_zncc(
                        patch,
                        &views[ti as usize],
                        &reference,
                        &single_ctx,
                        &sqrt_weights,
                        &params,
                        &mut raw_scratch,
                    )
                })
                .unwrap_or(f64::NAN),
        );
    }

    // Trust gate: only vet candidates when the track agrees with itself well
    // enough to trust the reference. Below the floor there is no trustworthy
    // reference, so the track is admitted verbatim with no expansion (but we still
    // report the measured self-agreement).
    if self_agreement < params.min_self_agreement {
        prof::count(&prof::N_VERBATIM, 1);
        return ViewSelection {
            admitted,
            scores,
            self_agreement,
        };
    }

    let bar = params.min_relative_zncc * self_agreement;

    // Candidates: every non-track view that geometrically sees the surfel
    // (front-facing, in front of the camera, projects in-frame with enough
    // coverage), admitted when its ZNCC clears the bar. The candidate loop is
    // sequential (plain iterator): the batch entry already parallelizes over
    // patches, so a nested rayon here would oversubscribe.
    let mut extra: Vec<(u32, f64)> = (0..views.len() as u32)
        .filter(|i| !is_track.contains(i))
        .filter_map(|i| {
            let view = &views[i as usize];
            // Geometric visibility: the patch must face this camera *and* the point
            // must be in front of it (cheirality — `is_front_facing` alone does not
            // guarantee positive depth on wide-fisheye / equirect projection).
            if !patch.is_front_facing(view.cam_from_world)
                || !is_in_front(patch, view.cam_from_world)
            {
                return None;
            }
            // Photometric vetting (also enforces in-frame coverage: a candidate
            // whose render misses the reference support is unscoreable -> rejected).
            prof::count(&prof::N_CANDIDATES, 1);
            let zncc = prof::CAND_SCORE.time(|| {
                candidate_zncc(
                    patch,
                    view,
                    &reference,
                    &single_ctx,
                    &sqrt_weights,
                    &params,
                    &mut raw_scratch,
                )
            })?;
            if zncc >= bar {
                prof::count(&prof::N_ADMITTED, 1);
            }
            (zncc >= bar).then_some((i, zncc))
        })
        .collect();
    extra.sort_by_key(|&(i, _)| i);

    for (i, zncc) in extra {
        admitted.push(i);
        scores.push(zncc);
    }

    ViewSelection {
        admitted,
        scores,
        self_agreement,
    }
}

/// Batch [`select_patch_views`] over a [`PatchCloud`], parallel across patches
/// (rayon). `track_views[i]` lists, for patch `i`, the track image indices of its
/// source point (see
/// [`view_indices_from_reconstruction`](super::normal_refine::view_indices_from_reconstruction)).
/// Results are returned in cloud order.
///
/// # Panics
///
/// Panics if `track_views.len() != cloud.len()` or an index is out of range.
pub fn select_patch_cloud_views(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    track_views: &[Vec<u32>],
    params: &ViewSelectParams,
    progress: Option<&std::sync::atomic::AtomicUsize>,
) -> Vec<ViewSelection> {
    assert_eq!(
        track_views.len(),
        cloud.len(),
        "track_views must be parallel to the cloud"
    );
    if prof::enabled() {
        prof::reset();
    }
    let wall_start = std::time::Instant::now();
    let out: Vec<ViewSelection> = cloud
        .patches
        .par_iter()
        .zip(track_views.par_iter())
        .map(|(patch, tv)| {
            let out = select_patch_views(patch, views, tv, params);
            // Bump the shared work counter per patch for a Python progress poller.
            if let Some(c) = progress {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            out
        })
        .collect();
    if prof::enabled() {
        prof::report(cloud.len(), wall_start.elapsed().as_secs_f64());
    }
    out
}

/// For each patch of `cloud` (linked to `recon` via `point_indexes`), the track image
/// indices observing its source 3D point — ready to use as the `track_views` of
/// [`select_patch_cloud_views`]. Identical to
/// [`view_indices_from_reconstruction`](super::normal_refine::view_indices_from_reconstruction);
/// re-exported here for symmetry with the selection API.
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
pub fn track_views_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>> {
    super::normal_refine::view_indices_from_reconstruction(recon, cloud)
}

#[cfg(test)]
mod tests;
