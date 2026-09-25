// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Registering one view against the consensus of a point's existing
//! observations, without moving them.
//!
//! The consensus-basis tail registration ([`super::tail`]) searches a view once
//! against a finished template the view did not help build. This is the same
//! search with the basis supplied rather than congealed: the point's existing
//! observations are rendered where their keypoints already put them, their
//! robust consensus is the template, and one other view is searched and scored
//! against it. Nothing about the references moves. See
//! `specs/core/reconstruction/add-image-to-tracks.md`, which is the operation built on it.
//!
//! The numbers it reports are all one kind of measurement: the windowed ZNCC of
//! a core rendered on the patch grid at a given keypoint, against a template or
//! another core. A reference's leave-one-out ZNCC and a searched view's ZNCC are
//! therefore comparable, because the searched view never contributed to the
//! consensus it is scored against.

use super::search::{search_shift, search_shift_plus_descent, SearchScratch};
use super::{
    extract_core, extract_core_grid, project, render_context, seed_offset, shifted_center,
    ContextTile, KeypointLocalizeParams, LocalizeError,
};
use crate::patch::cloud::OrientedPatch;
use crate::patch::localizability::{patch_localizability, SIGMA_NOISE};
use crate::patch::normal_refine::{
    build_support, irls_view_weights, weighted_unit_template_into, znormalize_into_kept,
    ConsensusScratch, ProjectedImage, Support,
};

/// Below this windowed norm² a channel of one core is flat and contributes `0`
/// to a ZNCC rather than a normalised noise pattern. The value the localizer's
/// own z-normalisation drops a channel at.
const FLAT_NORM_SQ: f64 = 1e-6;

/// The consensus of a point's existing observations, each rendered on the patch
/// grid at its own keypoint, with the scores the references give each other.
///
/// Built by [`ReferenceConsensus::build`]. The template is what a searched view
/// is registered against; the per-reference cores are kept so that the searched
/// view can also be scored against each reference on its own.
pub struct ReferenceConsensus {
    /// The image index of each reference that rendered in frame, in the order
    /// given (a reference whose core leaves its frame is left out).
    pub references: Vec<u32>,
    /// Per reference, its ZNCC against the robust consensus of the **other**
    /// references, parallel to [`Self::references`]. With two references this
    /// is their pairwise ZNCC, twice.
    pub loo_zncc: Vec<f64>,
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
    /// Which original channels the template scores on.
    template_mask: Vec<bool>,
    /// The unit-norm-per-channel template over the template's kept channels.
    template: Vec<f32>,
}

/// One searched view: where its correlation peak put the keypoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewSearch {
    /// The point's projection into the view, source px.
    pub projection: [f64; 2],
    /// The keypoint at the correlation peak (the integer peak plus the
    /// parabolic sub-step), source px. `None` when no shift of the window could
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
    /// [`ReferenceConsensus::search`]'s `ascend_on_edge`).
    pub ascended: bool,
    /// The weak-axis positional uncertainty `σ_pos` of the view's own core at
    /// the search's start, in patch-grid px (the member localizability score).
    /// `NaN` when the core is out of frame.
    pub sigma_pos: f64,
}

/// One view scored at a given keypoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewScore {
    /// ZNCC against the template.
    pub zncc: f64,
    /// ZNCC against each reference on its own, parallel to
    /// [`ReferenceConsensus::references`].
    pub pair_zncc: Vec<f64>,
}

impl ReferenceConsensus {
    /// Render each of `references` at its keypoint and build their consensus.
    ///
    /// `views` is indexed by image index; `keypoints` is parallel to
    /// `references`, in source px. Of `params`, `resolution`, `window`,
    /// `sampler` and `robust_iters` shape the renders and the consensus; the
    /// search resolution multiplier is not read (the grid is `resolution`).
    /// `None` when fewer than two references render in frame or no channel is
    /// textured in all of them.
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
        let mut raws: Vec<(usize, Vec<f32>)> = Vec::with_capacity(references.len());
        for (&image, &kp) in references.iter().zip(keypoints) {
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
                params.sampler,
            ) else {
                continue;
            };
            let mut raw = vec![0f32; tile.channels * n];
            if extract_core(&tile, &support, r, 0, 0, &mut raw) {
                kept_refs.push(image);
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

        let mut sc = ConsensusScratch::default();
        let mut template = Vec::new();
        irls_view_weights(&xs, views_n, kept, n, params.robust_iters, None, &mut sc);
        weighted_unit_template_into(&xs, &sc.w, views_n, kept, n, &mut template);

        // Leave-one-out: the consensus of the others, by the same IRLS.
        let mut loo_zncc = Vec::with_capacity(views_n);
        let mut others = Vec::with_capacity((views_n - 1) * cn);
        let mut loo_template = Vec::new();
        for v in 0..views_n {
            others.clear();
            for (u, core) in cores.iter().enumerate() {
                if u != v {
                    others.extend_from_slice(core);
                }
            }
            irls_view_weights(
                &others,
                views_n - 1,
                kept,
                n,
                params.robust_iters,
                None,
                &mut sc,
            );
            weighted_unit_template_into(&others, &sc.w, views_n - 1, kept, n, &mut loo_template);
            loo_zncc.push(dot(&cores[v], &loo_template) / kept as f64);
        }
        let mut pair_zncc = vec![1.0; views_n * views_n];
        for a in 0..views_n {
            for b in (a + 1)..views_n {
                let z = dot(&cores[a], &cores[b]) / kept as f64;
                pair_zncc[a * views_n + b] = z;
                pair_zncc[b * views_n + a] = z;
            }
        }
        Some(Self {
            references: kept_refs,
            loo_zncc,
            pair_zncc,
            resolution,
            support,
            wpp,
            template_mask: core_mask.clone(),
            core_mask,
            cores,
            template,
        })
    }

    /// Replace the template by a stored bitmap of the point: `R × R` pixels of
    /// `channels` interleaved `u8` values in the patch grid's row-major order,
    /// the layout of a `.sfmr` patch bitmap. Only the leading three channels
    /// are read, so an RGBA bitmap's alpha (a confidence, not a colour) is not
    /// scored. Returns `false`, leaving the rendered template in place, when
    /// the bitmap is not on this consensus's grid or has no textured channel.
    pub fn use_bitmap_template(&mut self, bitmap: &[u8], channels: usize) -> bool {
        let r = self.resolution as usize;
        if channels == 0 || bitmap.len() != r * r * channels {
            return false;
        }
        let colour = channels.min(3);
        let n = self.support.pixels.len();
        let mut raw = vec![0f32; colour * n];
        for (k, &p) in self.support.pixels.iter().enumerate() {
            for c in 0..colour {
                raw[c * n + k] = f32::from(bitmap[p * channels + c]);
            }
        }
        let mut xs = Vec::new();
        let Some((_, mask)) = znormalize_into_kept(
            &raw,
            1,
            colour,
            n,
            &self.support.weights,
            self.support.total_weight,
            &self.support.sqrt_weights,
            &mut xs,
        ) else {
            return false;
        };
        self.template = xs;
        self.template_mask = mask;
        true
    }

    /// The number of references.
    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// Whether there are no references (never true of a built consensus).
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    /// Search `view` once for the shift that best matches the template, over
    /// `±params.search` patch-grid px around `seed` (source px; `None` starts
    /// at the point's projection).
    ///
    /// The tail registration's search, run exhaustively: the view's context
    /// tile is rendered once centred on the start, its own core is scored for
    /// localizability there, and every shift of the window is correlated with
    /// the template. No gate is applied; the caller reads
    /// [`ViewSearch::sigma_pos`], [`ViewSearch::at_edge`] and the peak and
    /// decides. `Ok` with no keypoint when the point does not project into the
    /// view's frame.
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
            sigma_pos: f64::NAN,
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
            params.sampler,
        )?;
        let c0 = margin as usize;
        out.sigma_pos = core_sigma_pos(&tile, &self.support, r, c0);

        let mask = &self.template_mask[..self.template_mask.len().min(tile.channels)];
        let kept = mask.iter().filter(|&&k| k).count();
        if kept == 0 {
            return Ok(out);
        }
        let mut scratch = SearchScratch::default();
        scratch.try_reserve_grids((2 * margin + 1) as usize)?;
        scratch.tmpl.clear();
        scratch
            .tmpl
            .extend_from_slice(&self.template[..kept * self.support.pixels.len()]);
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
    /// its ZNCC against the template and against each reference.
    ///
    /// The same measurement [`Self::build`] took of each reference, so the
    /// numbers are comparable with [`Self::loo_zncc`] and
    /// [`Self::pair_zncc`]. `None` when the keypoint does not unproject onto
    /// the patch or the core leaves the frame.
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
            params.sampler,
        )
        .ok()?;
        let mut raw = vec![0f32; tile.channels * n];
        if !extract_core(&tile, &self.support, r, 0, 0, &mut raw) {
            return None;
        }
        let against_template =
            znorm_masked(&raw, tile.channels, &self.template_mask, &self.support);
        let zncc = masked_zncc(&against_template, &self.template, n);
        let against_cores = znorm_masked(&raw, tile.channels, &self.core_mask, &self.support);
        let pair_zncc = self
            .cores
            .iter()
            .map(|core| masked_zncc(&against_cores, core, n))
            .collect();
        Some(ViewScore { zncc, pair_zncc })
    }
}

/// The member localizability score of the `R × R` core sitting at `(c0, c0)`
/// of `tile`, in patch-grid px.
fn core_sigma_pos(tile: &ContextTile, support: &Support, r: usize, c0: usize) -> f64 {
    let mut grid = vec![0f32; r * r * tile.channels];
    extract_core_grid(tile, r, c0, c0, &mut grid);
    patch_localizability(&grid, r, tile.channels, support, SIGMA_NOISE).sigma_pos_grid
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
