// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Phase-B registration of consensus-basis tail views and result finalization.

use super::search::{search_shift, search_shift_plus_descent, SearchScratch};
use super::{
    below_absolute_floor, extract_core, member_is_localizable, prof, project, render_context,
    shifted_center, ContextTile, KeypointLocalization, KeypointLocalizeParams, LocalizeError,
    SearchStrategy, ViewState,
};
use crate::numeric::median_in_place;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::{
    irls_view_weights, weighted_unit_template_into, znormalize_into_kept, ConsensusScratch,
    ProjectedImage, Support,
};
use crate::progress::Progress;

/// The grid geometry phase-B registration needs, bundled so
/// [`register_tail`] keeps a readable signature.
pub(super) struct TailGeometry {
    /// Search resolution `R_s`.
    pub(super) resolution: u32,
    /// In-round search radius in `R_s`-grid steps.
    pub(super) margin: i64,
    /// Bound on the accumulated integer drift.
    pub(super) search_steps: i64,
    /// Cache index of the `R_s×R_s` core at zero offset in a **basis** cache.
    pub(super) cache_c0: usize,
    pub(super) wpp_u: f64,
    pub(super) wpp_v: f64,
}

/// Build the final all-basis consensus template into `out` from the surviving
/// basis members' cores, read at their final integer offsets. Returns the kept
/// channel count and mask, or `None` when fewer than two basis cores are still
/// in frame or no channel carries texture (no template to register against).
fn basis_template(
    states: &[ViewState],
    caches: &[ContextTile],
    support: &Support,
    geom: &TailGeometry,
    robust_iters: u32,
    out: &mut Vec<f32>,
) -> Option<(usize, Vec<bool>)> {
    let r = geom.resolution as usize;
    let n = support.pixels.len();
    let mut raws: Vec<Vec<f32>> = Vec::with_capacity(states.len());
    let mut live_channels: Vec<usize> = Vec::with_capacity(states.len());
    for (si, st) in states.iter().enumerate() {
        let cache = &caches[si];
        let mut raw = vec![0f32; cache.channels * n];
        let ox = (geom.cache_c0 as i64 + st.iacc[0]) as usize;
        let oy = (geom.cache_c0 as i64 + st.iacc[1]) as usize;
        if extract_core(cache, support, r, oy, ox, &mut raw) {
            raws.push(raw);
            live_channels.push(cache.channels);
        }
    }
    if raws.len() < 2 {
        return None;
    }
    let channels0 = *live_channels.iter().min().unwrap();
    let mut flat = vec![0f32; raws.len() * channels0 * n];
    for (vk, raw) in raws.iter().enumerate() {
        flat[vk * channels0 * n..][..channels0 * n].copy_from_slice(&raw[..channels0 * n]);
    }
    let mut xs = Vec::new();
    let (kept_ch, keep_mask) = prof::ZNORM.time(|| {
        znormalize_into_kept(
            &flat,
            raws.len(),
            channels0,
            n,
            &support.weights,
            support.total_weight,
            &support.sqrt_weights,
            &mut xs,
        )
    })?;
    // Same robust consensus as a congealing round, without a holdout.
    prof::TEMPLATE.time(|| {
        let mut sc = ConsensusScratch::default();
        irls_view_weights(&xs, raws.len(), kept_ch, n, robust_iters, None, &mut sc);
        weighted_unit_template_into(&xs, &sc.w, raws.len(), kept_ch, n, out);
    });
    Some((kept_ch, keep_mask))
}

/// Whether a view's refined keypoint is close enough to the point's projection
/// to keep — the `max_shift_px` half of the drop gates, shared by the scored
/// path and the no-template early exit so a tail view is never emitted without
/// at least this check.
fn within_max_shift(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    st: &ViewState,
    wpp_u: f64,
    wpp_v: f64,
    max_shift_px: f64,
) -> bool {
    let off = st.offset_steps();
    let center = shifted_center(patch, off[0], off[1], wpp_u, wpp_v);
    let shift_px = match project(view, &center, patch.w) {
        Some((x, y)) => (x - st.proj[0]).hypot(y - st.proj[1]),
        None => f64::INFINITY, // keypoint left the frame
    };
    shift_px <= max_shift_px
}

/// Phase B: register every `tail` view once against the basis consensus and
/// apply the per-view drop gates, leaving `tail` holding only the kept views.
///
/// Each tail view renders a cache centred on its **own** seed offset, sized
/// `R_s + 2·margin` — it searches one `±margin` window around that seed and so
/// needs no drift headroom (basis caches keep the `R_s + 4·margin` sizing that
/// covers a whole round loop). The gates are the loop's verbatim: drop a view
/// whose own tile fails the member localizability gate (scored before the search,
/// so an unlocalizable tail view is never even registered), whose refined
/// keypoint sits more than `max_shift_px` from the projection, whose ZNCC is
/// below the absolute `min_absolute_zncc` floor, or whose ZNCC falls below
/// `min_relative_zncc ×` the **basis members'** median final ZNCC (the same
/// threshold rule as the round loop, measured against a different reference —
/// the no-holdout basis template). There is no two-view floor here: the basis
/// already carries the point, so a failing tail view is simply not registered.
///
/// **Mixed channel counts.** The template's channel space is the one
/// [`basis_template`] built, i.e. the minimum over the *basis* caches. A tail
/// view can be narrower than that (a grayscale frame among colour ones), and
/// its tile then has no plane for the template's trailing channels. The round
/// loop's rule is "score in the channel space common to the views taking part",
/// so the tail applies the same rule pairwise: the mask is truncated to the
/// tail tile's own width. Only trailing *original* channels drop out, so the
/// surviving kept channels are a prefix of the template's rows and the template
/// needs no rebuild. A tail view with no kept channel left is unscorable and is
/// dropped by the gates below, exactly like one whose window is out of frame.
#[allow(clippy::too_many_arguments)]
pub(super) fn register_tail(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
    caches: &[ContextTile],
    tail: &mut Vec<ViewState>,
    support: &Support,
    search: &mut SearchScratch,
    geom: TailGeometry,
    params: &KeypointLocalizeParams,
    progress: &Progress<'_>,
) -> Result<(), LocalizeError> {
    let Some((kept_ch, keep_mask)) = basis_template(
        states,
        caches,
        support,
        &geom,
        params.robust_iters,
        &mut search.tmpl,
    ) else {
        // No usable basis consensus (the loop collapsed below two in-frame
        // views, or no channel carries texture): there is nothing to register
        // against, so the tail keeps its seed offsets with an unknown ZNCC. The
        // agreement gate cannot be evaluated without a template, but the
        // positional one can and still must be — a seed can already sit further
        // than `max_shift_px` from the projection, and nothing downstream would
        // catch it. The member localizability gate needs a rendered tile, and
        // this path renders none (that is the cost it exists to avoid), so it is
        // not applied; the point has already collapsed below two in-frame basis
        // views, and `min_views` is what decides its fate.
        prof::count(&prof::N_TAIL_NO_BASIS, tail.len() as u64);
        tail.retain(|st| {
            within_max_shift(
                patch,
                &views[st.idx as usize],
                st,
                geom.wpp_u,
                geom.wpp_v,
                params.max_shift_px,
            )
        });
        return Ok(());
    };

    // The tail's relative-agreement bar, from the basis members' final ZNCCs.
    let mut basis_loo: Vec<f64> = states
        .iter()
        .map(|st| st.loo)
        .filter(|z| z.is_finite())
        .collect();
    let med = median_in_place(&mut basis_loo);
    let bar = if med.is_finite() {
        params.min_relative_zncc * med
    } else {
        f64::NEG_INFINITY
    };

    let r = geom.resolution as usize;
    let tail_res = geom.resolution + 2 * geom.margin as u32;
    // The tail cache is centred on the view's seed, so the `(0, 0)` shift reads
    // its core at `margin`.
    let tail_c0 = geom.margin as usize;
    prof::count(&prof::N_RENDER, tail.len() as u64);
    // Parallel to `tail`: whether the view cleared the member localizability
    // gate. A view that did not is never searched and never kept, whatever it
    // would have scored against the basis template.
    let mut member_ok: Vec<bool> = Vec::with_capacity(tail.len());
    let mut grid_scratch: Vec<f32> = Vec::new();
    for st in tail.iter_mut() {
        progress.check_cancel()?;
        let view = &views[st.idx as usize];
        let cache = prof::RENDER.time(|| {
            render_context(
                patch,
                view,
                st.iacc[0] as f64,
                st.iacc[1] as f64,
                geom.wpp_u,
                geom.wpp_v,
                geom.resolution,
                tail_res,
                params.sampler,
            )
        })?;
        // Member localizability gate, the loop's verbatim: score this view's own
        // core tile (at its seed, where the cache is centred) and refuse a tile
        // that pins no 2D position before it is scored against the template.
        let ok = member_is_localizable(
            &cache,
            support,
            r,
            tail_c0,
            tail_c0,
            params.max_member_keypoint_uncertainty,
            &mut grid_scratch,
        );
        member_ok.push(ok);
        if !ok {
            prof::count(&prof::N_DROP_UNLOCALIZABLE, 1);
            st.loo = f64::NAN;
            continue;
        }
        // Score in the channel space this tail tile actually has (see the
        // "Mixed channel counts" note above); `sub_mask` is a prefix of the
        // template's mask, so `search.tmpl`'s leading rows still line up.
        let sub_mask = &keep_mask[..keep_mask.len().min(cache.channels)];
        let sub_kept = sub_mask.iter().filter(|&&k| k).count();
        // Truncating the mask can only remove kept channels, and only trailing
        // ones — so the template's leading `sub_kept` rows are the right ones.
        debug_assert!(sub_kept <= kept_ch);
        if sub_kept == 0 {
            // No channel the template scores on survives in this view.
            st.loo = f64::NAN;
            continue;
        }
        prof::count(&prof::N_SEARCH, 1);
        let sh = prof::SEARCH.time(|| match params.search_strategy {
            SearchStrategy::Exhaustive => search_shift(
                &cache,
                search,
                support,
                sub_mask,
                sub_kept,
                r,
                geom.margin,
                tail_c0,
                tail_c0,
            ),
            SearchStrategy::PlusDescent => search_shift_plus_descent(
                &cache,
                search,
                support,
                sub_mask,
                sub_kept,
                r,
                geom.margin,
                tail_c0,
                tail_c0,
            ),
        });
        match sh {
            Some(sh) => {
                st.iacc[0] = (st.iacc[0] + sh.ix).clamp(-geom.search_steps, geom.search_steps);
                st.iacc[1] = (st.iacc[1] + sh.iy).clamp(-geom.search_steps, geom.search_steps);
                st.residual = [sh.dx - sh.ix as f64, sh.dy - sh.iy as f64];
                st.loo = sh.peak;
            }
            // No scorable window: the view's core is out of frame at its seed.
            None => st.loo = f64::NAN,
        }
    }

    let mut i = 0;
    tail.retain(|st| {
        let ok = member_ok[i];
        i += 1;
        if below_absolute_floor(st.loo, params.min_absolute_zncc) {
            prof::count(&prof::N_DROP_ABS_ZNCC, 1);
            return false;
        }
        ok && within_max_shift(
            patch,
            &views[st.idx as usize],
            st,
            geom.wpp_u,
            geom.wpp_v,
            params.max_shift_px,
        ) && st.loo.is_finite()
            && st.loo >= bar
    });
    Ok(())
}

/// Where `keypoint` sits relative to the point's own projection, in **patch-grid
/// px** on the `params.resolution` grid: the offset
/// [`super::localize_patch_keypoints`] would seed that view's search at.
///
/// Published because the seed offset is what decides how wide a search window
/// has to be. The kernel clips the integer part of a seed beyond
/// [`KeypointLocalizeParams::search`] back onto that bound, so a caller that
/// means to read a view *where its observation actually sits* -- the bench's
/// evaluation -- asks this first and widens `search` to cover the furthest
/// answer. `None` on the same refusals the seeding itself makes: a ray parallel
/// to the plane, a hit behind the camera, a ray pointing away from a direction
/// patch, or a degenerate patch with no extent.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::patch::keypoint_localize::{keypoint_grid_offset, KeypointLocalizeParams};
/// # fn run(
/// #     patch: &sfmtool_core::patch::cloud::OrientedPatch,
/// #     view: &sfmtool_core::patch::normal_refine::ProjectedImage<'_>,
/// #     keypoint: [f64; 2],
/// # ) {
/// let mut params = KeypointLocalizeParams::default();
/// if let Some(off) = keypoint_grid_offset(patch, view, keypoint, &params) {
///     params.search += off[0].hypot(off[1]);   // reach the seed, then search
/// }
/// # }
/// ```
pub fn keypoint_grid_offset(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: [f64; 2],
    params: &KeypointLocalizeParams,
) -> Option<[f64; 2]> {
    let resolution = f64::from(params.resolution.max(2));
    seed_offset(
        patch,
        view,
        keypoint,
        2.0 * patch.half_extent[0] / resolution,
        2.0 * patch.half_extent[1] / resolution,
    )
}

/// Unproject a starting keypoint onto the patch plane and express the in-plane
/// offset of its hit point (from the patch centre) in patch-grid px.
///
/// The world-space unprojection itself is
/// [`OrientedPatch::anchored_at_keypoint`]'s — the same offset the renderer
/// re-anchors a frame by — so a seed and a re-anchored frame can never disagree
/// about where a keypoint puts the patch. `None` on each of that method's
/// refusals (ray parallel to the plane, a ray-path hit behind the camera, a ray
/// pointing away from a direction patch), and on a degenerate patch whose zero
/// extent makes `wpp` zero: seeding at the projection (`acc = 0`) beats
/// propagating a NaN/inf offset.
///
/// [`keypoint_grid_offset`] is the same question in a caller's terms: a params'
/// own grid resolution rather than a `wpp` pair.
pub(in crate::patch) fn seed_offset(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: [f64; 2],
    wpp_u: f64,
    wpp_v: f64,
) -> Option<[f64; 2]> {
    if wpp_u <= 0.0 || wpp_v <= 0.0 {
        return None;
    }
    let off = patch.keypoint_plane_offset(view.camera, view.cam_from_world, keypoint)?;
    // Grid rows count downward from `+v̂` (they map to `−v_axis`), so the
    // v-grid coordinate negates the in-plane `v̂` component — the inverse of
    // `shifted_center`.
    Some([
        off.dot(&patch.u_axis) / wpp_u,
        -off.dot(&patch.v_axis) / wpp_v,
    ])
}

/// Build the result from the final view states: the refined keypoint
/// `project_i(center_v)`, its offset from the projection, and the last
/// leave-one-out ZNCC, per surviving view.
pub(super) fn finalize(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    states: &[ViewState],
    wpp_u: f64,
    wpp_v: f64,
) -> KeypointLocalization {
    let mut out = KeypointLocalization::default();
    for st in states {
        let view = &views[st.idx as usize];
        // `offset_steps` is in `R_s`-grid steps (integer read accumulator + the
        // sub-pixel residual); `wpp` is the matching `R_s`-resolution world-per-step.
        let off = st.offset_steps();
        let center = shifted_center(patch, off[0], off[1], wpp_u, wpp_v);
        // The refined keypoint is the projection of the re-anchored centre. Fall
        // back to the point's own projection if the shifted centre fails to project.
        let (kx, ky) = project(view, &center, patch.w).unwrap_or((st.proj[0], st.proj[1]));
        out.views.push(st.idx);
        out.keypoints.push([kx, ky]);
        out.offsets_px
            .push((kx - st.proj[0]).hypot(ky - st.proj[1]));
        out.loo_zncc.push(st.loo);
        out.is_basis.push(st.is_basis);
    }
    out
}
