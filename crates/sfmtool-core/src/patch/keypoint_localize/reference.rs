// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Aligning one new view of a point to the point's reference render, without
//! moving its existing observations.
//!
//! Every kernel that places a view aligns it to the point's reference render,
//! and this is that alignment for a view the point does not have yet. The
//! template is the point's stored bitmap where the caller supplies one, else
//! the bitmap the point would store: its reference observation rendered at its
//! own keypoint, or, where it has none, the reference-view rule's pick from the
//! existing observations rendered at theirs, or the fused mean where the rule
//! would store that. The new view is searched once against the template and
//! scored. Nothing about the existing observations moves. See
//! `specs/core/reconstruction/add-image-to-tracks.md`, which is the operation
//! built on it.
//!
//! The numbers it reports are all one kind of measurement: the windowed ZNCC of
//! a core rendered on the patch grid at a given keypoint, against the template
//! or another core. An existing observation's score and the new view's are
//! therefore comparable: each is one view's plain ZNCC against the template.

use super::align::Template;
use super::search::{search_shift, search_shift_plus_descent, SearchScratch};
use super::{
    extract_core, member_self_similarity_radius, project, render_context, resolve_reference,
    seed_offset, shifted_center, KeypointLocalizeParams, LocalizeError,
};
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_subpixel::ReferenceTemplate;
use crate::patch::normal_refine::{build_support, znormalize_into_kept, ProjectedImage, Support};
use crate::progress::Progress;

/// Below this windowed norm² a channel of one core is flat and contributes `0`
/// to a ZNCC rather than a normalised noise pattern. The value the localizer's
/// own z-normalisation drops a channel at.
const FLAT_NORM_SQ: f64 = 1e-6;

/// What the template of a [`TrackReferences`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    /// The point's stored patch bitmap, as the caller supplied it.
    StoredBitmap,
    /// One reference, the point's reference observation, rendered at its own
    /// keypoint: the point's stored reference observation, or the
    /// reference-view rule's pick where it has none.
    ReferenceObservation,
    /// The fused mean of the references at their keypoints, where the point
    /// has no reference observation and the rule picks none it would store.
    FusedMean,
}

/// A point's existing observations (its references), each rendered on the
/// patch grid at its own keypoint, and the template a new view is aligned to
/// and judged against, with each reference's plain ZNCC against it.
///
/// Built by [`TrackReferences::build`]. The per-reference cores are kept so
/// that a new view can also be scored against each reference on its own.
pub struct TrackReferences {
    /// The image index of each reference that rendered in frame, in the order
    /// given (a reference whose core leaves its frame is left out).
    pub references: Vec<u32>,
    /// What the template is.
    pub template_kind: TemplateKind,
    /// The position in [`Self::references`] of the point's reference
    /// observation: the one whose render is the template under
    /// [`TemplateKind::ReferenceObservation`], or under
    /// [`TemplateKind::StoredBitmap`] the caller's reference observation where
    /// it rendered. `None` otherwise. A bar judging a new view leaves its
    /// score out, since it is the template or what the template was rendered
    /// from.
    pub reference: Option<usize>,
    /// Per reference, its plain ZNCC against the template at its own keypoint,
    /// parallel to [`Self::references`]. The reference observation reads
    /// exactly `1.0` where the template is its render.
    pub zncc: Vec<f64>,
    /// The pairwise ZNCC between references, row-major `n × n` with `1.0` on the
    /// diagonal.
    pub pair_zncc: Vec<f64>,
    resolution: u32,
    support: Support,
    wpp: [f64; 2],
    /// Which of the references' original channels are textured in all of them.
    core_mask: Vec<bool>,
    /// The z-normalised reference cores over the kept channels, one per
    /// reference, each `kept · n` long with `√w` folded in.
    cores: Vec<Vec<f32>>,
    /// The template, never blurred.
    template: Template,
    /// The `R×R` RGBA bitmap the template was read from, where it is a stored
    /// bitmap or the fused mean.
    bitmap: Option<Vec<u8>>,
    /// The reference observation's image index and keypoint, where the
    /// template is its render.
    anchor: Option<(u32, [f64; 2])>,
}

/// One searched view: where its correlation peak put the keypoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewSearch {
    /// The point's projection into the view, source px.
    pub projection: [f64; 2],
    /// The keypoint at the correlation peak (the integer peak plus the
    /// sub-pixel step of the 3×3 quadratic fit), source px. `None` when no shift of the window could
    /// be scored in frame, or the shifted centre does not project.
    pub keypoint: Option<[f64; 2]>,
    /// The ZNCC at the integer peak, against the template. `NaN` without a
    /// peak.
    pub peak_zncc: f64,
    /// Whether the peak sits on the edge of the searched window, where the true
    /// maximum may lie outside it.
    pub at_edge: bool,
    /// Whether the window's highest peak was on its edge and the keypoint is
    /// instead the local maximum an ascent from the start reached (see
    /// [`TrackReferences::search`]'s `ascend_on_edge`).
    pub ascended: bool,
    /// The ZNCC self-similarity radius of the view's own core at the search's
    /// start, in patch-grid px: the number the member gate
    /// ([`KeypointLocalizeParams::max_member_zncc_self_similarity_radius`])
    /// judges. `NaN` when the point does not project into the view.
    pub zncc_self_similarity_radius: f64,
}

/// One view scored at a given keypoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewScore {
    /// Plain ZNCC against the template.
    pub zncc: f64,
    /// ZNCC against each reference on its own, parallel to
    /// [`TrackReferences::references`].
    pub pair_zncc: Vec<f64>,
}

impl TrackReferences {
    /// Render each of `references` at its keypoint and settle the template.
    ///
    /// `views` is indexed by image index; `keypoints` is parallel to
    /// `references`, in source px. `reference` is the position in `references`
    /// of the point's stored reference observation, `None` where it has none.
    /// `bitmap` is the point's stored `R×R` RGBA bitmap on the grid of
    /// `params.resolution`, row-major as a `.sfmr` stores it; where it is given
    /// and textured it is the template. Otherwise the template is the bitmap
    /// the point would store ([`TemplateKind`]): the reference observation's
    /// render at its keypoint where it rendered, else the reference-view rule's
    /// pick from the references' renders at their keypoints, or their fused
    /// mean where the rule would store that. Of `params`, `resolution`,
    /// `window`, `sampler` and `robust_iters` (the fused mean's) are read.
    ///
    /// `None` when fewer than two references render in frame, no channel is
    /// textured in all of them, or no template renders.
    ///
    /// # Panics
    ///
    /// Panics if `keypoints.len() != references.len()` or a reference is out of
    /// range for `views`.
    pub fn build(
        patch: &OrientedPatch,
        views: &[ProjectedImage<'_>],
        references: &[u32],
        keypoints: &[[f64; 2]],
        reference: Option<usize>,
        bitmap: Option<&[u8]>,
        params: &KeypointLocalizeParams,
    ) -> Option<Self> {
        assert_eq!(
            references.len(),
            keypoints.len(),
            "keypoints must be parallel to references"
        );
        let resolution = params.resolution.max(2);
        let r = resolution as usize;
        let wpp = [
            2.0 * patch.half_extent[0] / resolution as f64,
            2.0 * patch.half_extent[1] / resolution as f64,
        ];
        let support = build_support(params.window, resolution);
        let n = support.pixels.len();

        let mut kept_refs = Vec::with_capacity(references.len());
        let mut kept_keypoints = Vec::with_capacity(references.len());
        let mut given = None;
        let mut raws: Vec<(usize, Vec<f32>)> = Vec::with_capacity(references.len());
        for (k, (&image, &kp)) in references.iter().zip(keypoints).enumerate() {
            let view = &views[image as usize];
            let Some(off) = seed_offset(patch, view, kp, wpp[0], wpp[1]) else {
                continue;
            };
            let Ok(tile) = render_context(
                patch,
                view,
                off[0],
                off[1],
                wpp[0],
                wpp[1],
                resolution,
                resolution,
                params.sampler.for_observation(
                    patch,
                    view.camera,
                    view.cam_from_world,
                    Some(kp),
                    resolution,
                ),
                &Progress::none(),
            ) else {
                continue;
            };
            let mut raw = vec![0f32; tile.channels * n];
            if extract_core(&tile, &support, r, 0, 0, &mut raw) {
                if reference == Some(k) {
                    given = Some(kept_refs.len());
                }
                kept_refs.push(image);
                kept_keypoints.push(kp);
                raws.push((tile.channels, raw));
            }
        }
        if raws.len() < 2 {
            return None;
        }
        let channels = raws.iter().map(|(c, _)| *c).min().unwrap();
        let mut flat = vec![0f32; raws.len() * channels * n];
        for (v, (_, raw)) in raws.iter().enumerate() {
            flat[v * channels * n..][..channels * n].copy_from_slice(&raw[..channels * n]);
        }
        let mut xs = Vec::new();
        let (kept, core_mask) = znormalize_into_kept(
            &flat,
            raws.len(),
            channels,
            n,
            &support.weights,
            support.total_weight,
            &support.sqrt_weights,
            &mut xs,
        )?;
        let views_n = raws.len();
        let cn = kept * n;
        let cores: Vec<Vec<f32>> = (0..views_n).map(|v| xs[v * cn..][..cn].to_vec()).collect();

        // The template: the stored bitmap, else the bitmap the point would
        // store, rendered from the references at their keypoints.
        let stored =
            bitmap.and_then(|b| Template::from_bitmap(b, 4, r, &support).map(|t| (t, b.to_vec())));
        let (template, template_kind, reference, bitmap) = match stored {
            Some((template, b)) => (template, TemplateKind::StoredBitmap, given, Some(b)),
            None => {
                let seeds: Vec<Option<[f64; 2]>> =
                    kept_keypoints.iter().copied().map(Some).collect();
                let resolved = resolve_reference(
                    patch,
                    views,
                    &kept_refs,
                    &seeds,
                    given,
                    resolution,
                    params.window,
                    params.sampler,
                    params.robust_iters,
                    &Progress::none(),
                );
                match (resolved.anchor, resolved.fused) {
                    (Some(a), _) => {
                        let (ch, raw) = &raws[a];
                        let template = Template::from_raw(raw, *ch, &support)?;
                        (template, TemplateKind::ReferenceObservation, Some(a), None)
                    }
                    (None, Some(fused)) => {
                        let template = Template::from_bitmap(&fused, 4, r, &support)?;
                        (template, TemplateKind::FusedMean, None, Some(fused))
                    }
                    (None, None) => return None,
                }
            }
        };
        let rendered_from =
            |v: usize| template_kind == TemplateKind::ReferenceObservation && reference == Some(v);

        let zncc = raws
            .iter()
            .enumerate()
            .map(|(v, (ch, raw))| {
                if rendered_from(v) {
                    1.0
                } else {
                    let core = znorm_masked(raw, *ch, &template.mask, &support);
                    masked_zncc(&core, &template.values, n)
                }
            })
            .collect();
        let mut pair_zncc = vec![1.0; views_n * views_n];
        for a in 0..views_n {
            for b in (a + 1)..views_n {
                let z = dot(&cores[a], &cores[b]) / kept as f64;
                pair_zncc[a * views_n + b] = z;
                pair_zncc[b * views_n + a] = z;
            }
        }
        let anchor = reference
            .filter(|&a| rendered_from(a))
            .map(|a| (kept_refs[a], kept_keypoints[a]));
        Some(Self {
            references: kept_refs,
            template_kind,
            reference,
            zncc,
            pair_zncc,
            resolution,
            support,
            wpp,
            core_mask,
            cores,
            template,
            bitmap,
            anchor,
        })
    }

    /// The template as the sub-pixel refiner
    /// ([`refine_view_against_reference`](crate::patch::keypoint_subpixel::refine_view_against_reference))
    /// takes it: the bitmap the template was read from, or the reference
    /// observation at its keypoint.
    pub fn refine_template(&self) -> ReferenceTemplate<'_> {
        match (&self.bitmap, self.anchor) {
            (Some(bitmap), _) => ReferenceTemplate::Bitmap(bitmap),
            (None, Some((image, keypoint))) => ReferenceTemplate::Observation { image, keypoint },
            (None, None) => unreachable!("a template is a bitmap or an observation's render"),
        }
    }

    /// The number of references.
    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// Whether there are no references (never true of a built value).
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    /// Search `view` once for the shift that best matches the template, over
    /// `±params.search` patch-grid px around `seed` (source px; `None` starts
    /// at the point's projection).
    ///
    /// The view's context tile is rendered once centred on the start, its own
    /// core's ZNCC self-similarity radius is read there, and every shift of the
    /// window is correlated with the template. It is run exhaustively because
    /// it is one view per point. No gate is applied; the caller reads
    /// [`ViewSearch::zncc_self_similarity_radius`], [`ViewSearch::at_edge`] and
    /// the peak and decides. `Ok` with no keypoint when the point does not
    /// project into the view's frame.
    ///
    /// With `ascend_on_edge`, a window whose highest peak sits on its edge is
    /// searched again by the "+"-descent from the start, which climbs to the
    /// local maximum nearest the start. Where that maximum is inside the window
    /// it is the answer (and [`ViewSearch::ascended`] says so): a repeated
    /// texture can put a stronger correlation a pattern period away, and the
    /// start (the projection) is the evidence for which period is meant.
    pub fn search(
        &self,
        patch: &OrientedPatch,
        view: &ProjectedImage<'_>,
        seed: Option<[f64; 2]>,
        ascend_on_edge: bool,
        params: &KeypointLocalizeParams,
    ) -> Result<ViewSearch, LocalizeError> {
        let mut out = ViewSearch {
            projection: [f64::NAN; 2],
            keypoint: None,
            peak_zncc: f64::NAN,
            at_edge: false,
            ascended: false,
            zncc_self_similarity_radius: f64::NAN,
        };
        let Some(proj) = project(view, &patch.center, patch.w) else {
            return Ok(out);
        };
        out.projection = [proj.0, proj.1];
        let r = self.resolution as usize;
        let margin = params.search.ceil().max(1.0) as i64;
        let start = seed
            .and_then(|kp| seed_offset(patch, view, kp, self.wpp[0], self.wpp[1]))
            .unwrap_or([0.0, 0.0]);
        let tile = render_context(
            patch,
            view,
            start[0],
            start[1],
            self.wpp[0],
            self.wpp[1],
            self.resolution,
            self.resolution + 2 * margin as u32,
            params.sampler.for_observation(
                patch,
                view.camera,
                view.cam_from_world,
                seed,
                self.resolution,
            ),
            &Progress::none(),
        )?;
        let c0 = margin as usize;
        out.zncc_self_similarity_radius =
            member_self_similarity_radius(&tile, r, c0, c0, &mut Vec::new());

        let mask = &self.template.mask[..self.template.mask.len().min(tile.channels)];
        let kept = mask.iter().filter(|&&k| k).count();
        if kept == 0 {
            return Ok(out);
        }
        let mut scratch = SearchScratch::default();
        scratch.try_reserve_grids((2 * margin + 1) as usize)?;
        scratch.tmpl.clear();
        scratch
            .tmpl
            .extend_from_slice(&self.template.values[..kept * self.support.pixels.len()]);
        let Some(mut sh) = search_shift(
            &tile,
            &mut scratch,
            &self.support,
            mask,
            kept,
            r,
            margin,
            c0,
            c0,
        ) else {
            return Ok(out);
        };
        let on_edge = |ix: i64, iy: i64| ix.abs() == margin || iy.abs() == margin;
        if ascend_on_edge && on_edge(sh.ix, sh.iy) {
            if let Some(local) = search_shift_plus_descent(
                &tile,
                &mut scratch,
                &self.support,
                mask,
                kept,
                r,
                margin,
                c0,
                c0,
            ) {
                if !on_edge(local.ix, local.iy) {
                    sh = local;
                    out.ascended = true;
                }
            }
        }
        out.peak_zncc = sh.peak;
        out.at_edge = on_edge(sh.ix, sh.iy);
        let center = shifted_center(
            patch,
            start[0] + sh.dx,
            start[1] + sh.dy,
            self.wpp[0],
            self.wpp[1],
        );
        out.keypoint = project(view, &center, patch.w).map(|(x, y)| [x, y]);
        Ok(out)
    }

    /// Score `view` at `keypoint` (source px): render its core there and take
    /// its plain ZNCC against the template and against each reference.
    ///
    /// The same measurement [`Self::build`] took of each reference, so the
    /// numbers are comparable with [`Self::zncc`] and [`Self::pair_zncc`].
    /// `None` when the keypoint does not unproject onto the patch or the core
    /// leaves the frame.
    pub fn score(
        &self,
        patch: &OrientedPatch,
        view: &ProjectedImage<'_>,
        keypoint: [f64; 2],
        params: &KeypointLocalizeParams,
    ) -> Option<ViewScore> {
        let r = self.resolution as usize;
        let n = self.support.pixels.len();
        let off = seed_offset(patch, view, keypoint, self.wpp[0], self.wpp[1])?;
        let tile = render_context(
            patch,
            view,
            off[0],
            off[1],
            self.wpp[0],
            self.wpp[1],
            self.resolution,
            self.resolution,
            params.sampler.for_observation(
                patch,
                view.camera,
                view.cam_from_world,
                Some(keypoint),
                self.resolution,
            ),
            &Progress::none(),
        )
        .ok()?;
        let mut raw = vec![0f32; tile.channels * n];
        if !extract_core(&tile, &self.support, r, 0, 0, &mut raw) {
            return None;
        }
        let against_template =
            znorm_masked(&raw, tile.channels, &self.template.mask, &self.support);
        let zncc = masked_zncc(&against_template, &self.template.values, n);
        let against_cores = znorm_masked(&raw, tile.channels, &self.core_mask, &self.support);
        let pair_zncc = self
            .cores
            .iter()
            .map(|core| masked_zncc(&against_cores, core, n))
            .collect();
        Some(ViewScore { zncc, pair_zncc })
    }
}

/// A raw core z-normalised over the channels `mask` keeps (the leading
/// `mask.len()` channels of `raw`, truncated to the ones `raw` has), each
/// channel with `√w` folded in. A kept channel flat in this core comes back as
/// zeros. Each kept channel is `Some(values)`, or `None` where `raw` has no such
/// channel, so the caller can skip the template's rows for it.
fn znorm_masked(
    raw: &[f32],
    raw_channels: usize,
    mask: &[bool],
    support: &Support,
) -> Vec<Option<Vec<f32>>> {
    let n = support.pixels.len();
    let mut out = Vec::new();
    for (c, &keep) in mask.iter().enumerate() {
        if !keep {
            continue;
        }
        if c >= raw_channels {
            out.push(None);
            continue;
        }
        let col = &raw[c * n..][..n];
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for (&x, &w) in col.iter().zip(&support.weights) {
            s1 += w * f64::from(x);
            s2 += w * f64::from(x) * f64::from(x);
        }
        let mean = s1 / support.total_weight;
        let norm_sq = s2 - s1 * mean;
        let values = if norm_sq < FLAT_NORM_SQ {
            vec![0.0; n]
        } else {
            let inv = 1.0 / norm_sq.sqrt();
            col.iter()
                .zip(&support.sqrt_weights)
                .map(|(&x, &sw)| (f64::from(sw) * (f64::from(x) - mean) * inv) as f32)
                .collect()
        };
        out.push(Some(values));
    }
    out
}

/// The channel-averaged ZNCC of a masked core against a template laid out
/// `[kept_channel · n + k]`, over the channels the core has.
fn masked_zncc(core: &[Option<Vec<f32>>], template: &[f32], n: usize) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (kc, channel) in core.iter().enumerate() {
        if let Some(values) = channel {
            sum += dot(values, &template[kc * n..][..n]);
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}
