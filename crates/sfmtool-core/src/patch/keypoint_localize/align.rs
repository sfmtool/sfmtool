// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Aligning every view of a point to its reference render, in one pass.
//!
//! The template is the point's stored bitmap as the views' starting keypoints
//! render it: the reference observation's `R×R` render at its own keypoint, or,
//! where the point has no reference observation and the reference-view rule
//! picks none it would store, the fused mean of the views. Each other view is
//! searched once against it. The reference observation's keypoint is not
//! moved. See `specs/core/patch/patch-keypoint-localization.md`.

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
use crate::patch::reference_view::render_view_tile;
use crate::patch::stored_bitmap::{
    bitmap_from_tile, bitmap_planes, render_reference, stored_view, BitmapScorer,
};
use crate::progress::Progress;

/// What a point's views are aligned to: the position among them of the
/// reference observation, or the fused mean that is the stored bitmap where
/// there is none.
pub(in crate::patch) struct ResolvedReference {
    /// The position of the reference observation among the views.
    pub(in crate::patch) anchor: Option<usize>,
    /// The fused mean of the views, `R×R` RGBA, where there is no reference
    /// observation and the mean renders.
    pub(in crate::patch) fused: Option<Vec<u8>>,
}

/// Whether `view` sees `patch` at least `min_grazing_cos` away from edge-on:
/// the absolute cosine between the viewing direction (camera to point, or the
/// direction itself for a point at infinity, to which every ray is parallel)
/// and the patch normal. The localizer drops a view that fails this, and the
/// reference-view rule in [`resolve_reference`] never picks one.
pub(in crate::patch) fn faces_view(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    min_grazing_cos: f64,
) -> bool {
    let d = if patch.w == 0.0 {
        patch.center.coords
    } else {
        patch.center - view.cam_from_world.inverse_translation_origin()
    };
    let dn = d.norm();
    dn > 1e-12 && (d.dot(&patch.normal()) / dn).abs() >= min_grazing_cos
}

/// Decide what the views are aligned to. The localizer and the sub-pixel
/// refiner both call this, so given the same views and keypoints they align to
/// the same reference render.
///
/// `view_set` and `seeds` are parallel; `given` is the position in them of the caller's reference,
/// which is the reference. With none, only views that pass the grazing pre-filter ([`faces_view`]
/// at `min_grazing_cos`) are candidates: the reference-view rule reads the facing views' renders at
/// their starting keypoints (a view with no seed rendered through the patch itself) and its pick is
/// the reference, unless it picks none or reaches its pick only by its last fallback, where the
/// stored bitmap is the fused mean of the facing views ([`stored_view`]) and that mean is the
/// template. Where no fused mean renders, a last-fallback pick is the reference after all, as it is
/// the stored bitmap then
/// ([`render_patch_bitmap`](crate::patch::stored_bitmap::render_patch_bitmap)).
#[allow(clippy::too_many_arguments)]
pub(in crate::patch) fn resolve_reference(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    seeds: &[Option<[f64; 2]>],
    given: Option<usize>,
    min_grazing_cos: f64,
    resolution: u32,
    window: PatchWindow,
    sampler: SamplerChoice,
    robust_iters: u32,
    progress: &Progress<'_>,
) -> ResolvedReference {
    let facing: Vec<usize> = (0..view_set.len())
        .filter(|&k| faces_view(patch, &views[view_set[k] as usize], min_grazing_cos))
        .collect();
    if given.is_some() || facing.len() < 2 {
        return ResolvedReference {
            anchor: given.or((facing.len() == 1).then(|| facing[0])),
            fused: None,
        };
    }
    let view_set: Vec<u32> = facing.iter().map(|&k| view_set[k]).collect();
    let seeds: Vec<Option<[f64; 2]>> = facing.iter().map(|&k| seeds[k]).collect();
    let render = render_reference(
        patch, views, &view_set, &seeds, resolution, sampler, progress,
    );
    let choice = &render.reading.choice;
    if let Some(anchor) = stored_view(choice) {
        return ResolvedReference {
            anchor: Some(facing[anchor]),
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
        fuse_patch_bitmap_reporting(patch, views, &view_set, &keypoints, &params, progress)
    });
    // Where no fused mean renders either, the rule's pick is the stored bitmap
    // after all, and so the reference.
    ResolvedReference {
        anchor: if fused.is_none() {
            choice.reference.map(|pick| facing[pick])
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
    /// The view's starting keypoint, as given (`None` starts at the projection).
    seed: Option<[f64; 2]>,
    /// The starting offset on the patch grid, `[u, v]` in grid px.
    start: [f64; 2],
    /// Whether `start` is where `seed` is: no seed was given, or it maps onto
    /// the patch plane. When it does not, `start` is the projection instead.
    start_is_seed: bool,
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
    for (slot, &i) in view_set.iter().enumerate() {
        if !seen.insert(i) {
            continue;
        }
        let view = &views[i as usize];
        if !faces_view(patch, view, params.min_grazing_cos) {
            continue;
        }
        let Some(proj) = project(view, &patch.center, patch.w) else {
            continue;
        };
        let seed = starting_keypoints.and_then(|seeds| seeds[slot]);
        let unprojected = seed.map(|kp| seed_offset(patch, view, kp, wpp_u, wpp_v));
        let start_is_seed = !matches!(unprojected, Some(None));
        let start = unprojected.flatten().unwrap_or([0.0, 0.0]);
        let sampler = params.sampler.for_observation(
            patch,
            view.camera,
            view.cam_from_world,
            seed,
            resolution,
        );
        states.push(AlignedView {
            idx: i,
            seed,
            start,
            start_is_seed,
            proj: [proj.0, proj.1],
            sampler,
            off: start,
            zncc: f64::NAN,
            placed: false,
            localizable: true,
        });
    }

    // The reference: the caller's, where it survived the pre-filter, else the
    // rule's pick from the renders at the starting keypoints. It is matched by
    // image, so a reference given at a repeated image's dropped slot is kept.
    let given = reference
        .and_then(|k| view_set.get(k))
        .and_then(|&img| states.iter().position(|st| st.idx == img));
    let resolved = prof::REFERENCE.time(|| {
        let set: Vec<u32> = states.iter().map(|st| st.idx).collect();
        let seeds: Vec<Option<[f64; 2]>> = states.iter().map(|st| st.seed).collect();
        resolve_reference(
            patch,
            views,
            &set,
            &seeds,
            given,
            params.min_grazing_cos,
            resolution,
            params.window,
            params.sampler,
            params.robust_iters,
            progress,
        )
    });
    progress.check_cancel()?;
    let anchor = resolved.anchor;

    // The template, never blurred. The reference's tile is rendered at its
    // keypoint; one whose keypoint does not map onto the patch plane has no
    // tile there, and so no template.
    let template = match (anchor, &resolved.fused) {
        (Some(a), _) if !states[a].start_is_seed => None,
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
        // Nothing to align to (the reference's tile leaves the frame, is flat
        // or cannot be placed at its keypoint, or no fused mean renders): there
        // is no reference, every view stays at its start, unscored, and the
        // keypoints mean what they meant before.
        for st in &mut states {
            st.placed = true;
        }
        return Ok(finish(
            patch, views, states, None, None, wpp_u, wpp_v, params,
        ));
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
        // Member self-similarity gate: a view whose own tile fixes no 2D
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
    // The template as a stored bitmap, which the agreement gates read each
    // placed view against blur-matched: the reference's render at its
    // keypoint, as a stored bitmap renders it, or the fused mean. Only built
    // where a gate is on.
    let gates_on = gate_is_on(params.min_absolute_zncc) || gate_is_on(params.min_relative_zncc);
    let bitmap = if !gates_on {
        None
    } else if let Some(a) = anchor {
        let st = &states[a];
        Some(bitmap_from_tile(&render_view_tile(
            patch,
            &views[st.idx as usize],
            Some(st.seed.unwrap_or(st.proj)),
            r,
            params.sampler,
            progress,
        )))
    } else {
        resolved.fused.clone()
    };
    states.retain(|st| st.localizable);
    let mut out = finish(
        patch,
        views,
        states,
        reference,
        bitmap.as_deref(),
        wpp_u,
        wpp_v,
        params,
    );
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

/// Whether an agreement gate's bar turns it on: finite and above `0`.
fn gate_is_on(bar: f64) -> bool {
    bar.is_finite() && bar > 0.0
}

/// Map the final offsets to keypoints and apply the gates that read them:
/// a view the search could not place is dropped, as is one whose keypoint
/// leaves the frame, sits more than `max_shift_px` from the projection, or
/// scores below the absolute or relative bar. The reference observation,
/// `anchor`, keeps the keypoint it was given and faces no gate.
///
/// The two agreement gates read each placed view's **blur-matched score**
/// against `bitmap`, the template as a stored bitmap: the view's tile rendered
/// at its final keypoint ([`render_view_tile`]) and read by [`BitmapScorer`],
/// the bitmap blurred to the tile's sharpness where blur matching selects it,
/// as the bench reads a row against the stored bitmap. The alignment itself
/// read the unblurred template. `bitmap` is `None` where neither gate is on,
/// and then no view is scored this way.
#[allow(clippy::too_many_arguments)]
fn finish(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: Vec<AlignedView>,
    anchor: Option<u32>,
    bitmap: Option<&[u8]>,
    wpp_u: f64,
    wpp_v: f64,
    params: &KeypointLocalizeParams,
) -> KeypointLocalization {
    // Each view's keypoint, before any gate.
    let placed: Vec<(AlignedView, Option<[f64; 2]>)> = states
        .into_iter()
        .map(|st| {
            let view = &views[st.idx as usize];
            let is_anchor = Some(st.idx) == anchor;
            let keypoint = if is_anchor || (st.placed && st.off == st.start && st.start_is_seed) {
                // The reference's keypoint, and any view's that did not move
                // (there was nothing to align it to), is returned exactly as
                // it was given rather than through a round trip onto the
                // plane.
                Some(st.seed.unwrap_or(st.proj))
            } else if st.placed {
                let center = shifted_center(patch, st.off[0], st.off[1], wpp_u, wpp_v);
                project(view, &center, patch.w).map(|(x, y)| [x, y])
            } else {
                None
            };
            (st, keypoint)
        })
        .collect();

    // The blur-matched score of every searched view at its keypoint.
    let resolution = params.resolution.max(2) as usize;
    let planes = bitmap
        .filter(|b| b.len() == resolution * resolution * 4)
        .map(|b| bitmap_planes(b, resolution));
    let mut scorer = planes.as_ref().map(|p| BitmapScorer::new(p, params.window));
    let blur_matched: Vec<f64> = placed
        .iter()
        .map(|(st, kp)| {
            if Some(st.idx) == anchor {
                return 1.0;
            }
            match (scorer.as_mut(), kp) {
                (Some(scorer), Some(kp)) if st.zncc.is_finite() => {
                    prof::count(&prof::N_RENDER, 1);
                    let tile = prof::RENDER.time(|| {
                        render_view_tile(
                            patch,
                            &views[st.idx as usize],
                            Some(*kp),
                            resolution,
                            params.sampler,
                            &Progress::none(),
                        )
                    });
                    scorer.score(&tile.planes(), None).blur_matched_zncc
                }
                _ => f64::NAN,
            }
        })
        .collect();

    let mut others: Vec<f64> = placed
        .iter()
        .zip(&blur_matched)
        .filter(|((st, _), _)| Some(st.idx) != anchor)
        .map(|(_, &z)| z)
        .filter(|z| z.is_finite())
        .collect();
    let median = median_in_place(&mut others);
    let relative = params.min_relative_zncc;
    let bar = if gate_is_on(relative) && median.is_finite() {
        relative * median
    } else {
        f64::NEG_INFINITY
    };
    let mut out = KeypointLocalization::default();
    for ((st, keypoint), bz) in placed.into_iter().zip(blur_matched) {
        let is_anchor = Some(st.idx) == anchor;
        let Some(kp) = keypoint else {
            continue;
        };
        let shift = (kp[0] - st.proj[0]).hypot(kp[1] - st.proj[1]);
        if !is_anchor {
            if shift > params.max_shift_px {
                prof::count(&prof::N_DROP_SHIFT, 1);
                continue;
            }
            if below_absolute_floor(bz, params.min_absolute_zncc) {
                prof::count(&prof::N_DROP_ABS_ZNCC, 1);
                continue;
            }
            if bz.is_finite() && bz < bar {
                prof::count(&prof::N_DROP_REL_ZNCC, 1);
                continue;
            }
        }
        out.views.push(st.idx);
        out.keypoints.push(kp);
        out.offsets_px.push(shift);
        out.zncc.push(st.zncc);
        out.blur_matched_zncc.push(bz);
    }
    out
}
