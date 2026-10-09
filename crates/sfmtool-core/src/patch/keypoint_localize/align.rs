// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Aligning every view of a point to its reference render, in one pass.
//!
//! The template is the point's stored bitmap as the views' starting keypoints
//! render it: the reference observation's `R×R` render at its own keypoint, or,
//! where the point has no reference observation and the reference-view rule
//! picks none it would store, the fused mean of the views. Each other view is
//! searched once against it. The reference observation is the anchor of the
//! track and is not moved. See `specs/core/patch/patch-keypoint-localization.md`.

use super::search::{search_shift, search_shift_plus_descent, SearchScratch};
use super::{
    below_absolute_floor, extract_core, member_self_similarity_radius, prof, project,
    render_context, seed_offset, shifted_center, ContextTile, KeypointLocalization,
    KeypointLocalizeParams, LocalizeError, SearchStrategy,
};
use crate::camera::sampler::Sampler;
use crate::numeric::median_in_place;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_subpixel::{fuse_patch_bitmap_reporting, KeypointSubpixelParams};
use crate::patch::normal_refine::{
    build_support, znormalize_into_kept, PatchWindow, ProjectedImage, SamplerChoice, Support,
};
use crate::patch::stored_bitmap::{render_reference, stored_view};
use crate::progress::Progress;

/// What a point's views are aligned to, decided from the views that survived
/// the pre-filters: the position among them of the reference observation, or
/// the fused mean that stands as the stored bitmap where there is none.
pub(in crate::patch) struct ResolvedReference {
    /// The position of the reference observation among the views.
    pub(in crate::patch) anchor: Option<usize>,
    /// The fused mean of the views, `R×R` RGBA, where there is no reference
    /// observation and the mean renders.
    pub(in crate::patch) fused: Option<Vec<u8>>,
}

/// Decide what the views are aligned to.
///
/// `view_set` and `seeds` describe the views that survived the caller's
/// pre-filters, parallel; `given` is the position among them of the caller's
/// reference. With none, the reference-view rule reads the views' renders at
/// their starting keypoints (a view with no seed rendered through the patch
/// itself) and its pick is the reference, unless it picks none or reaches its
/// pick only by its last fallback, where the stored bitmap is the fused mean of
/// the views ([`stored_view`]) and that mean is the template. Where no fused
/// mean renders, a last-fallback pick is the reference after all, as it is the
/// stored bitmap then ([`render_patch_bitmap`](crate::patch::stored_bitmap::render_patch_bitmap)).
#[allow(clippy::too_many_arguments)]
pub(in crate::patch) fn resolve_reference(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    seeds: &[Option<[f64; 2]>],
    given: Option<usize>,
    resolution: u32,
    window: PatchWindow,
    sampler: SamplerChoice,
    robust_iters: u32,
    progress: &Progress<'_>,
) -> ResolvedReference {
    if given.is_some() || view_set.len() < 2 {
        return ResolvedReference {
            anchor: given.or((view_set.len() == 1).then_some(0)),
            fused: None,
        };
    }
    let render = render_reference(patch, views, view_set, seeds, resolution, sampler, progress);
    let choice = &render.reading.choice;
    if let Some(anchor) = stored_view(choice) {
        return ResolvedReference {
            anchor: Some(anchor),
            fused: None,
        };
    }
    let projections: Vec<Option<[f64; 2]>> = view_set
        .iter()
        .map(|&i| project(&views[i as usize], &patch.center, patch.w).map(|(x, y)| [x, y]))
        .collect();
    let keypoints: Option<Vec<[f64; 2]>> = seeds
        .iter()
        .zip(&projections)
        .map(|(seed, proj)| seed.or(*proj))
        .collect();
    let fused = keypoints.and_then(|keypoints| {
        let params = KeypointSubpixelParams {
            resolution,
            window,
            sampler,
            robust_iters,
            ..Default::default()
        };
        fuse_patch_bitmap_reporting(patch, views, view_set, &keypoints, &params, progress)
    });
    // Where no fused mean renders either, the rule's pick stands as the stored
    // bitmap after all, and so as the reference.
    ResolvedReference {
        anchor: if fused.is_none() {
            choice.reference
        } else {
            None
        },
        fused,
    }
}

/// One view of the point that survived the pre-filter.
struct AlignedView {
    /// Image index into the caller's `views`.
    idx: u32,
    /// Position of the view in the caller's `view_set`.
    slot: usize,
    /// The view's starting keypoint, as given (`None` starts at the projection).
    seed: Option<[f64; 2]>,
    /// The starting offset on the patch grid, `[u, v]` in grid px.
    start: [f64; 2],
    /// The point's projection into the view, source px.
    proj: [f64; 2],
    /// The sampler every render of the view uses.
    sampler: Sampler,
    /// The final offset on the patch grid, `[u, v]` in grid px.
    off: [f64; 2],
    /// The plain ZNCC against the template at the integer peak.
    zncc: f64,
    /// Whether the view was placed: the anchor, a view the search scored, or
    /// every view where there was no template to search against.
    placed: bool,
    /// Whether the view passed the member self-similarity gate.
    localizable: bool,
}

/// The template the views are aligned to: z-normalized, `kept · n` long with
/// `√w` folded in, and which of the original channels it keeps.
pub(in crate::patch) struct Template {
    pub(in crate::patch) values: Vec<f32>,
    pub(in crate::patch) mask: Vec<bool>,
}

impl Template {
    /// The template from a raw `channels · n` core over the support. `None`
    /// when no channel carries texture.
    pub(in crate::patch) fn from_raw(
        raw: &[f32],
        channels: usize,
        support: &Support,
    ) -> Option<Self> {
        let n = support.pixels.len();
        let mut values = Vec::new();
        let (_, mask) = znormalize_into_kept(
            raw,
            1,
            channels,
            n,
            &support.weights,
            support.total_weight,
            &support.sqrt_weights,
            &mut values,
        )?;
        Some(Self { values, mask })
    }

    /// The template from an `R×R` bitmap of `channels` interleaved `u8`
    /// values, of which the leading three are read, so an RGBA bitmap's alpha
    /// (a confidence, not a colour) is not scored. `None` when the bitmap is
    /// not on the support's grid or no channel carries texture.
    pub(in crate::patch) fn from_bitmap(
        bitmap: &[u8],
        channels: usize,
        resolution: usize,
        support: &Support,
    ) -> Option<Self> {
        if channels == 0 || bitmap.len() != resolution * resolution * channels {
            return None;
        }
        let colour = channels.min(3);
        let n = support.pixels.len();
        let mut raw = vec![0f32; colour * n];
        for (k, &p) in support.pixels.iter().enumerate() {
            for c in 0..colour {
                raw[c * n + k] = f32::from(bitmap[p * channels + c]);
            }
        }
        Self::from_raw(&raw, colour, support)
    }
}

/// Align every view of `view_set` to the point's reference render, once.
///
/// See [`super::try_localize_patch_keypoints`], which this is the body of.
pub(super) fn align_to_reference(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[Option<[f64; 2]>]>,
    reference: Option<usize>,
    params: &KeypointLocalizeParams,
    progress: &Progress<'_>,
) -> Result<KeypointLocalization, LocalizeError> {
    let resolution = params.resolution.max(2);
    let r = resolution as usize;
    let margin = params.search.ceil().max(1.0) as i64;
    let wpp_u = 2.0 * patch.half_extent[0] / resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / resolution as f64;
    let support = build_support(params.window, resolution);
    let n = support.pixels.len();

    // Dedup (a point can carry two observations in one image), the grazing
    // pre-filter and the projection, in the view set's order.
    let mut seen = std::collections::HashSet::new();
    let mut states: Vec<AlignedView> = Vec::new();
    let normal = patch.normal();
    for (slot, &i) in view_set.iter().enumerate() {
        if !seen.insert(i) {
            continue;
        }
        let view = &views[i as usize];
        // The viewing direction is camera→point: `center − cam_c` for a finite
        // point, or the direction itself for a point at infinity (every ray to
        // it is parallel, so it is fully frontal).
        let d = if patch.w == 0.0 {
            patch.center.coords
        } else {
            patch.center - view.cam_from_world.inverse_translation_origin()
        };
        let dn = d.norm();
        let grazing_cos = if dn > 1e-12 {
            (d.dot(&normal) / dn).abs()
        } else {
            0.0
        };
        if dn <= 1e-12 || grazing_cos < params.min_grazing_cos {
            continue;
        }
        let Some(proj) = project(view, &patch.center, patch.w) else {
            continue;
        };
        let seed = starting_keypoints.and_then(|seeds| seeds[slot]);
        let start = seed
            .and_then(|kp| seed_offset(patch, view, kp, wpp_u, wpp_v))
            .unwrap_or([0.0, 0.0]);
        let sampler = params.sampler.for_observation(
            patch,
            view.camera,
            view.cam_from_world,
            seed,
            resolution,
        );
        states.push(AlignedView {
            idx: i,
            slot,
            seed,
            start,
            proj: [proj.0, proj.1],
            sampler,
            off: start,
            zncc: f64::NAN,
            placed: false,
            localizable: true,
        });
    }

    // The reference: the caller's, where it survived the pre-filter, else the
    // rule's pick from the renders at the starting keypoints.
    let given = reference.and_then(|k| states.iter().position(|st| st.slot == k));
    let resolved = prof::REFERENCE.time(|| {
        let set: Vec<u32> = states.iter().map(|st| st.idx).collect();
        let seeds: Vec<Option<[f64; 2]>> = states.iter().map(|st| st.seed).collect();
        resolve_reference(
            patch,
            views,
            &set,
            &seeds,
            given,
            resolution,
            params.window,
            params.sampler,
            params.robust_iters,
            progress,
        )
    });
    progress.check_cancel()?;
    let anchor = resolved.anchor;

    // The template, never blurred.
    let template = match (anchor, &resolved.fused) {
        (Some(a), _) => {
            let st = &states[a];
            prof::count(&prof::N_RENDER, 1);
            let tile = prof::RENDER.time(|| {
                render_context(
                    patch,
                    &views[st.idx as usize],
                    st.start[0],
                    st.start[1],
                    wpp_u,
                    wpp_v,
                    resolution,
                    resolution,
                    st.sampler,
                    progress,
                )
            })?;
            let mut raw = vec![0f32; tile.channels * n];
            if extract_core(&tile, &support, r, 0, 0, &mut raw) {
                Template::from_raw(&raw, tile.channels, &support)
            } else {
                None
            }
        }
        (None, Some(bitmap)) => Template::from_bitmap(bitmap, 4, r, &support),
        (None, None) => None,
    };

    let Some(template) = template else {
        // Nothing to align to (the reference's tile leaves the frame or is
        // flat, or no fused mean renders): every view stays at its start,
        // unscored, and the keypoints mean what they meant before.
        for st in &mut states {
            st.placed = true;
        }
        return Ok(finish(patch, views, states, None, wpp_u, wpp_v, params));
    };
    if let Some(a) = anchor {
        states[a].zncc = 1.0;
        states[a].placed = true;
    }

    let mut scratch = SearchScratch::default();
    // The shift grids, reserved once and fallibly: they are `(2 · margin + 1)²`
    // per grid and the search resizes into them, so reserving the capacity here
    // means every later `resize` sits inside it and cannot be the allocation
    // that aborts.
    scratch.try_reserve_grids((2 * margin + 1) as usize)?;
    let context_res = resolution + 2 * margin as u32;
    // The context tile is centred on the view's start, so the `(0, 0)` shift
    // reads its core at `margin`.
    let c0 = margin as usize;
    let mut member_scratch: Vec<f32> = Vec::new();
    for (si, st) in states.iter_mut().enumerate() {
        if Some(si) == anchor {
            continue;
        }
        // The cancel flag is polled per view: the render is the expensive half
        // of a wide search.
        progress.check_cancel()?;
        let view = &views[st.idx as usize];
        prof::count(&prof::N_RENDER, 1);
        let tile = prof::RENDER.time(|| {
            render_context(
                patch,
                view,
                st.start[0],
                st.start[1],
                wpp_u,
                wpp_v,
                resolution,
                context_res,
                st.sampler,
                progress,
            )
        })?;
        // Member self-similarity gate: a view whose own tile pins no 2D
        // position matches itself a few pixels away and correlates to noise
        // against anything, so it is refused before it is searched.
        if params.member_self_similarity_gate_is_on() {
            let radius = member_self_similarity_radius(&tile, r, c0, c0, &mut member_scratch);
            if !params.admits_member_zncc_self_similarity_radius(radius) {
                prof::count(&prof::N_DROP_UNLOCALIZABLE, 1);
                st.localizable = false;
                continue;
            }
        }
        if let Some((d, peak)) = search_tile(
            &tile,
            &mut scratch,
            &support,
            &template,
            r,
            margin,
            c0,
            params.search_strategy,
        ) {
            st.off = [st.start[0] + d[0], st.start[1] + d[1]];
            st.zncc = peak;
            st.placed = true;
        }
    }

    let reference = anchor.map(|a| states[a].idx);
    states.retain(|st| st.localizable);
    let mut out = finish(patch, views, states, reference, wpp_u, wpp_v, params);
    out.reference = reference.filter(|idx| out.views.contains(idx));
    Ok(out)
}

/// Search one context tile against the template over `±margin` around the
/// core at `(c0, c0)`: the sub-pixel shift `[dx, dy]` in grid px and the ZNCC
/// at the integer peak, or `None` when no shift could be scored in frame or no
/// channel the template scores on is in the tile.
///
/// **Mixed channel counts.** A view can be narrower than the template (a
/// grayscale frame beside a colour reference). The mask is truncated to the
/// tile's own width, which can only remove trailing kept channels, so the
/// template's leading rows still line up.
#[allow(clippy::too_many_arguments)]
fn search_tile(
    tile: &ContextTile,
    scratch: &mut SearchScratch,
    support: &Support,
    template: &Template,
    r: usize,
    margin: i64,
    c0: usize,
    strategy: SearchStrategy,
) -> Option<([f64; 2], f64)> {
    let n = support.pixels.len();
    let mask = &template.mask[..template.mask.len().min(tile.channels)];
    let kept = mask.iter().filter(|&&k| k).count();
    if kept == 0 {
        return None;
    }
    scratch.tmpl.clear();
    scratch.tmpl.extend_from_slice(&template.values[..kept * n]);
    prof::count(&prof::N_SEARCH, 1);
    let sh = prof::SEARCH.time(|| match strategy {
        SearchStrategy::Exhaustive => {
            search_shift(tile, scratch, support, mask, kept, r, margin, c0, c0)
        }
        SearchStrategy::PlusDescent => {
            search_shift_plus_descent(tile, scratch, support, mask, kept, r, margin, c0, c0)
        }
    })?;
    Some(([sh.dx, sh.dy], sh.peak))
}

/// Map the final offsets to keypoints and apply the gates that read them:
/// a view the search could not place is dropped, as is one whose keypoint
/// leaves the frame, sits more than `max_shift_px` from the projection, or
/// scores below the absolute or relative bar. The reference observation,
/// `anchor`, keeps the keypoint it was given and faces no gate.
fn finish(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: Vec<AlignedView>,
    anchor: Option<u32>,
    wpp_u: f64,
    wpp_v: f64,
    params: &KeypointLocalizeParams,
) -> KeypointLocalization {
    let mut others: Vec<f64> = states
        .iter()
        .filter(|st| Some(st.idx) != anchor)
        .map(|st| st.zncc)
        .filter(|z| z.is_finite())
        .collect();
    let median = median_in_place(&mut others);
    let relative = params.min_relative_zncc;
    let bar = if relative.is_finite() && relative > 0.0 && median.is_finite() {
        relative * median
    } else {
        f64::NEG_INFINITY
    };
    let mut out = KeypointLocalization::default();
    for st in states {
        let view = &views[st.idx as usize];
        let is_anchor = Some(st.idx) == anchor;
        let keypoint = if is_anchor {
            // The anchor's keypoint is returned exactly as it was given.
            Some(st.seed.unwrap_or(st.proj))
        } else if st.placed {
            let center = shifted_center(patch, st.off[0], st.off[1], wpp_u, wpp_v);
            project(view, &center, patch.w).map(|(x, y)| [x, y])
        } else {
            None
        };
        let Some(kp) = keypoint else {
            continue;
        };
        let shift = (kp[0] - st.proj[0]).hypot(kp[1] - st.proj[1]);
        if !is_anchor {
            if shift > params.max_shift_px {
                prof::count(&prof::N_DROP_SHIFT, 1);
                continue;
            }
            if below_absolute_floor(st.zncc, params.min_absolute_zncc) {
                prof::count(&prof::N_DROP_ABS_ZNCC, 1);
                continue;
            }
            if st.zncc.is_finite() && st.zncc < bar {
                prof::count(&prof::N_DROP_REL_ZNCC, 1);
                continue;
            }
        }
        out.views.push(st.idx);
        out.keypoints.push(kp);
        out.offsets_px.push(shift);
        out.zncc.push(st.zncc);
    }
    out
}
