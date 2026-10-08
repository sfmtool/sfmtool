// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The stored patch bitmap: the render of the point's reference view, and how
//! each of the point's observations scores against it.
//!
//! `specs/core/patch/reference-view.md` § "The stored bitmap" and
//! `specs/core/patch/blur-matched-zncc.md` § "Scores against the stored
//! bitmap" are the design.
//!
//! **The bitmap.** [`render_patch_bitmap`] renders every view's `R×R` tile at
//! its keypoint ([`render_view_tile`]), runs the reference-view rule over the
//! tiles ([`read_track`]), and stores the picked view's tile as it is: its
//! colour, with alpha `255` on the samples that carry image data and `0` on
//! the rest ([`bitmap_from_tile`]). The bitmap is then exactly the tile the
//! rule read, through the patch re-anchored on the reference observation's
//! keypoint, at the reconstruction's patch resolution, with the sampler the
//! sampler rule picks for that view. Only the tiles are read: no coarser grid,
//! and no pixels of the photograph outside them.
//!
//! Where the rule picks no view, which happens only where no candidate has a
//! self-similarity reading or every view sees the patch edge on or from
//! behind, the bitmap is the fused mean of the views
//! ([`fuse_patch_bitmap`](crate::patch::keypoint_subpixel::fuse_patch_bitmap)),
//! and no observation is named as its reference.
//!
//! **The scores.** [`BitmapScorer`] scores observations' tiles against the
//! bitmap, plain and blur-matched. Only the bitmap is ever blurred: where its
//! self-similarity semi-major axis is shorter than the observation's
//! semi-minor axis, by at least [`DEFAULT_MIN_ELLIPSE_RATIO`], it is blurred
//! to that semi-minor axis, at most
//! [`MAX_MATCHED_LENGTH`](crate::patch::pair_sharpness::MAX_MATCHED_LENGTH)
//! ([`bitmap_blur`]). An observation sharper than the bitmap, or one that
//! does not pass that test, is read plain. The bitmap's blur assessment
//! ([`assess_blur`]) is read at most once, when the first observation needs
//! it. The reference observation's own score is `1` and is not computed
//! ([`score_against_bitmap`]). Alignment never reads these scores' blur: it
//! runs against the bitmap as rendered.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::camera::sampler::SamplerChoice;
use crate::patch::blur_matched::{
    assess_blur, blur_to_length_into, read_tile_ellipse, semi_axes, windowed_zncc, BlurAssessment,
    BlurScratch, TilePlanes,
};
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::keypoint_subpixel::{fuse_patch_bitmap_reporting, KeypointSubpixelParams};
use crate::patch::normal_refine::{window_weights, PatchWindow, ProjectedImage};
use crate::patch::pair_sharpness::{bitmap_blur, DEFAULT_MIN_ELLIPSE_RATIO};
use crate::patch::reference_view::{
    read_track, render_view_tile, tile_semi_axes, TrackReading, ViewTile,
};
use crate::progress::{Cancelled, Progress};

#[cfg(test)]
mod tests;

/// A point's stored patch bitmap and the view it is the render of.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchBitmap {
    /// The `R·R·4` RGBA texture, row-major.
    pub rgba: Vec<u8>,
    /// The view whose render the bitmap is, as an index into the views it
    /// was rendered from; `None` where the bitmap is the fused mean of the
    /// views because the reference-view rule picked none.
    pub reference: Option<usize>,
}

/// The RGBA bitmap of a view's tile: its colour as rendered, a grey tile's
/// value in all three colour channels, and alpha `255` on the samples that
/// carry image data and `0` on the rest. `R·R·4`, row-major.
///
/// Alpha is the stored bitmap's per-pixel confidence: a sample of the
/// reference view's tile either holds the photograph or does not, and the
/// renderer and the readers of a stored bitmap take a sample whose alpha is 0
/// as carrying no data.
pub fn bitmap_from_tile(tile: &ViewTile) -> Vec<u8> {
    let r = tile.resolution();
    let channels = tile.channels();
    let mut rgba = vec![0u8; r * r * 4];
    for row in 0..r {
        for col in 0..r {
            let k = row * r + col;
            let out = &mut rgba[k * 4..k * 4 + 4];
            for (c, value) in out.iter_mut().take(3).enumerate() {
                *value = tile.samples[[row, col, if channels >= 3 { c } else { 0 }]];
            }
            out[3] = if tile.valid[k] { u8::MAX } else { 0 };
        }
    }
    rgba
}

/// The reference-view rule over one point's views, with the tiles it read
/// ([`render_reference`]).
#[derive(Debug, Clone)]
pub struct ReferenceRender {
    /// Per view: its `R×R` tile at its keypoint.
    pub tiles: Vec<ViewTile>,
    /// Per view: its tile's self-similarity semi-axes `[major, minor]`, in
    /// grid px.
    pub semi_axes: Vec<Option<[f64; 2]>>,
    /// What the rule read and picked.
    pub reading: TrackReading,
}

/// Render each of `view_set`'s views of `patch` at its keypoint and run the
/// reference-view rule over the tiles.
///
/// `view_set` indexes `views`, and `keypoints` is parallel to it, in
/// source-image px; a view with no keypoint is rendered through the patch
/// itself rather than re-anchored. The tiles are rendered at `resolution` with
/// the sampler `sampler` picks for each view ([`render_view_tile`]), each
/// tile's self-similarity read ([`tile_semi_axes`]), and the rule's readings
/// taken across them ([`read_track`]). The renders and member coherence's are
/// timed in their samplers' detail phases of `progress`.
///
/// # Panics
///
/// Panics if `keypoints` is not parallel to `view_set` or a view index is out
/// of range.
pub fn render_reference(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    keypoints: &[Option<[f64; 2]>],
    resolution: u32,
    sampler: SamplerChoice,
    progress: &Progress<'_>,
) -> ReferenceRender {
    assert_eq!(
        keypoints.len(),
        view_set.len(),
        "keypoints must be parallel to view_set"
    );
    let resolution = resolution.max(2) as usize;
    let tiles: Vec<ViewTile> = view_set
        .iter()
        .zip(keypoints)
        .map(|(&v, &kp)| {
            render_view_tile(patch, &views[v as usize], kp, resolution, sampler, progress)
        })
        .collect();
    let semi_axes: Vec<Option<[f64; 2]>> = tiles.iter().map(tile_semi_axes).collect();
    let refs: Vec<&ViewTile> = tiles.iter().collect();
    let reading = read_track(
        patch, views, view_set, keypoints, &refs, &semi_axes, sampler, progress,
    );
    ReferenceRender {
        tiles,
        semi_axes,
        reading,
    }
}

/// Render `patch`'s stored bitmap from `view_set`'s views at `keypoints`: the
/// tile of the view the reference-view rule picks ([`render_reference`],
/// [`bitmap_from_tile`]), or, where it picks none, the fused mean of the
/// views ([`fuse_patch_bitmap_reporting`]) with no reference. `None` with
/// fewer than two views, or where the rule picks none and fewer than two
/// views render in frame for the mean.
///
/// Of `params`, `resolution` and `sampler` shape the tiles; the fused mean
/// also reads `window` and `robust_iters`. Moves nothing.
///
/// ```no_run
/// # use sfmtool_core::patch::cloud::OrientedPatch;
/// # use sfmtool_core::patch::keypoint_subpixel::KeypointSubpixelParams;
/// # use sfmtool_core::patch::normal_refine::ProjectedImage;
/// # use sfmtool_core::patch::stored_bitmap::render_patch_bitmap;
/// # use sfmtool_core::progress::Progress;
/// # fn run(patch: &OrientedPatch, views: &[ProjectedImage<'_>]) {
/// let keypoints = [[812.4, 377.9], [640.2, 301.5], [702.0, 355.1]];
/// let params = KeypointSubpixelParams { resolution: 24, ..Default::default() };
/// if let Some(bitmap) =
///     render_patch_bitmap(patch, views, &[0, 1, 2], &keypoints, &params, &Progress::none())
/// {
///     println!("{} bytes, reference view {:?}", bitmap.rgba.len(), bitmap.reference);
/// }
/// # }
/// ```
///
/// # Panics
///
/// Panics if `keypoints` is not parallel to `view_set` or a view index is out
/// of range.
pub fn render_patch_bitmap(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    keypoints: &[[f64; 2]],
    params: &KeypointSubpixelParams,
    progress: &Progress<'_>,
) -> Option<PatchBitmap> {
    assert_eq!(
        keypoints.len(),
        view_set.len(),
        "keypoints must be parallel to view_set"
    );
    if view_set.len() < 2 {
        return None;
    }
    let anchors: Vec<Option<[f64; 2]>> = keypoints.iter().map(|&kp| Some(kp)).collect();
    let render = render_reference(
        patch,
        views,
        view_set,
        &anchors,
        params.resolution,
        params.sampler,
        progress,
    );
    match render.reading.choice.reference {
        Some(r) => Some(PatchBitmap {
            rgba: bitmap_from_tile(&render.tiles[r]),
            reference: Some(r),
        }),
        None => fuse_patch_bitmap_reporting(patch, views, view_set, keypoints, params, progress)
            .map(|rgba| PatchBitmap {
                rgba,
                reference: None,
            }),
    }
}

/// The stored bitmap column of a reconstruction's points and the reference
/// observation of each ([`render_patch_cloud_bitmaps`]).
#[derive(Debug, Clone)]
pub struct PatchBitmapColumn {
    /// `(P, R, R, 4)`, one row per point of the reconstruction; a zero row
    /// for a point with no bitmap.
    pub bitmaps: ndarray::Array4<u8>,
    /// Per point, the index of its reference observation within its own
    /// track, `-1` where it has none: the column `tracks/reference_observations`
    /// holds.
    pub reference_observations: Vec<i32>,
}

/// One point's rendered bitmap and the index of its reference observation
/// within its track, `-1` for none.
type RenderedRow = (Vec<u8>, i32);

/// [`render_patch_bitmap`] over every patch of `cloud`, parallel across
/// patches (rayon), each from its point's track in `recon` at the stored
/// per-observation keypoints.
///
/// `views` holds one entry per image of `recon`. A `None` entry is an image
/// whose photograph is not to hand, and it is left out of every patch's view
/// set rather than failing the call. Each point's reference is reported as
/// the index of that observation within the point's track, so the column is
/// what `tracks/reference_observations` holds; a point with no patch, no
/// bitmap or a fused-mean bitmap gets `-1`, and a point with no bitmap a zero
/// row. `done`, when given, is bumped once per patch, and `progress` receives
/// a `patches` count about every hundredth of the way through and is polled
/// for cancellation before each patch.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled before every patch was
/// rendered.
///
/// # Panics
///
/// Panics if `recon` carries no inline keypoints, if a patch's point index is
/// out of range for `recon`, or a track's image index is out of range for
/// `views`.
pub fn render_patch_cloud_bitmaps(
    cloud: &PatchCloud,
    recon: &crate::SfmrReconstruction,
    views: &[Option<ProjectedImage<'_>>],
    params: &KeypointSubpixelParams,
    done: Option<&AtomicUsize>,
    progress: &Progress<'_>,
) -> Result<PatchBitmapColumn, Cancelled> {
    let keypoints_xy = recon
        .keypoints_xy()
        .expect("render_patch_cloud_bitmaps needs a reconstruction with inline keypoints");
    let resolution = params.resolution.max(2) as usize;
    let point_count = recon.point_count();
    let offsets = &recon.point_set.observation_offsets;
    let tracks = &recon.point_set.tracks;
    // The views that are to hand, packed, and where each image's view went.
    let mut present: Vec<ProjectedImage<'_>> = Vec::with_capacity(views.len());
    let mut slot: Vec<Option<u32>> = Vec::with_capacity(views.len());
    for view in views {
        slot.push(view.as_ref().map(|view| {
            present.push(*view);
            (present.len() - 1) as u32
        }));
    }
    let total = cloud.patches.len();
    let step = (total / 100).max(1);
    let rendered_so_far = AtomicUsize::new(0);
    let rendered: Vec<(usize, Option<RenderedRow>)> = cloud
        .patches
        .par_iter()
        .zip(cloud.point_indexes.par_iter())
        .map(|(patch, &pid)| {
            let p = pid as usize;
            if progress.is_cancelled() {
                return (p, None);
            }
            let mut view_set: Vec<u32> = Vec::new();
            let mut keypoints: Vec<[f64; 2]> = Vec::new();
            // The position within the point's track of each view in the set.
            let mut within: Vec<usize> = Vec::new();
            for (k, j) in (offsets[p]..offsets[p + 1]).enumerate() {
                let Some(view) = slot[tracks[j].image_index as usize] else {
                    continue;
                };
                view_set.push(view);
                keypoints.push([
                    f64::from(keypoints_xy[[j, 0]]),
                    f64::from(keypoints_xy[[j, 1]]),
                ]);
                within.push(k);
            }
            let bitmap = render_patch_bitmap(
                patch, &present, &view_set, &keypoints, params, progress,
            )
            .map(|b| {
                let reference = b.reference.map_or(-1, |r| within[r] as i32);
                (b.rgba, reference)
            });
            if let Some(counter) = done {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            let n = rendered_so_far.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(step) || n == total {
                progress.count(n as u64, Some(total as u64), "patches");
            }
            (p, bitmap)
        })
        .collect();
    progress.check_cancel()?;
    let mut bitmaps = ndarray::Array4::<u8>::zeros((point_count, resolution, resolution, 4));
    let mut reference_observations = vec![-1i32; point_count];
    let row_len = resolution * resolution * 4;
    let flat = bitmaps
        .as_slice_mut()
        .expect("a freshly allocated array is contiguous");
    for (p, bitmap) in rendered {
        if let Some((rgba, reference)) = bitmap {
            flat[p * row_len..(p + 1) * row_len].copy_from_slice(&rgba);
            reference_observations[p] = reference;
        }
    }
    Ok(PatchBitmapColumn {
        bitmaps,
        reference_observations,
    })
}

/// The colour planes of an `R·R·4` RGBA bitmap, its samples whose alpha is
/// above 0 flagged as carrying data: what [`BitmapScorer`] reads a stored
/// bitmap as.
///
/// # Panics
///
/// Panics if `rgba` is not `R·R·4`.
pub fn bitmap_planes(rgba: &[u8], resolution: usize) -> TilePlanes {
    assert_eq!(
        rgba.len(),
        resolution * resolution * 4,
        "an R×R RGBA bitmap has R·R·4 values"
    );
    let data: Vec<bool> = rgba.as_chunks::<4>().0.iter().map(|p| p[3] > 0).collect();
    TilePlanes::from_interleaved(rgba, resolution, 4, &data)
}

/// One observation's scores against the point's stored bitmap
/// ([`BitmapScorer::score`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitmapScore {
    /// The windowed ZNCC of the observation's tile with the bitmap as stored,
    /// over the samples with data in both ([`windowed_zncc`]); `NaN` where it
    /// cannot be read.
    pub zncc: f64,
    /// The same after the bitmap is blurred to the observation's sharpness,
    /// where [`bitmap_blur`] says to blur it; equal to [`Self::zncc`] where
    /// the pair is read plain.
    pub blur_matched_zncc: f64,
    /// The width of the round blur the bitmap was blurred by, in grid px; `0`
    /// where the pair was read plain.
    pub blur_sigma: f64,
    /// Whether the observation's tile is sharper than the bitmap along every
    /// direction: its self-similarity semi-major axis is shorter than the
    /// bitmap's semi-minor axis. Such a pair is read plain, neither tile
    /// blurred; the observation is a candidate to replace the reference.
    pub sharper_than_bitmap: bool,
}

/// Scores observations against one stored bitmap, plain and blur-matched,
/// blurring only the bitmap.
///
/// It holds the bitmap's planes, its self-similarity ellipse read once, and
/// its blur assessment, read at most once, when the first observation needs
/// the bitmap blurred. Every score reads its width off that one assessment.
pub struct BitmapScorer<'a> {
    bitmap: &'a TilePlanes,
    ellipse: Option<[[f64; 2]; 2]>,
    /// `None` until read; then the assessment, or `None` where it could not
    /// be read.
    assessment: Option<Option<BlurAssessment>>,
    weights: Vec<f64>,
    scratch: BlurScratch,
    blurred: TilePlanes,
}

impl<'a> BitmapScorer<'a> {
    /// A scorer for `bitmap` ([`bitmap_planes`]), the pairs read over
    /// `window`. The bitmap's self-similarity ellipse is read here
    /// ([`read_tile_ellipse`]).
    pub fn new(bitmap: &'a TilePlanes, window: PatchWindow) -> Self {
        let ellipse = read_tile_ellipse(&bitmap.values, bitmap.channels, bitmap.side, &bitmap.data);
        Self {
            bitmap,
            ellipse,
            assessment: None,
            weights: window_weights(window, bitmap.side as u32),
            scratch: BlurScratch::default(),
            blurred: TilePlanes::default(),
        }
    }

    /// The bitmap's self-similarity semi-axes `[major, minor]`, in grid px,
    /// where its ellipse could be read.
    pub fn bitmap_semi_axes(&self) -> Option<[f64; 2]> {
        self.ellipse.as_ref().map(semi_axes)
    }

    /// The bitmap's blur assessment, where one has been read.
    pub fn assessment(&self) -> Option<&BlurAssessment> {
        self.assessment.as_ref().and_then(Option::as_ref)
    }

    /// Score `observation`'s tile against the bitmap. `ellipse` is the
    /// observation tile's self-similarity ellipse matrix in grid px²; `None`
    /// reads it here ([`read_tile_ellipse`]).
    ///
    /// The pair is blur-matched where [`bitmap_blur`] finds the bitmap sharper
    /// than the observation along every direction by at least
    /// [`DEFAULT_MIN_ELLIPSE_RATIO`]: the bitmap is blurred by the width its
    /// assessment gives to reach the observation's semi-minor axis, at most
    /// 2 grid px, and correlated with the observation's tile as rendered.
    /// Otherwise both readings are the plain one.
    ///
    /// # Panics
    ///
    /// Panics if the observation's tile is not the bitmap's side.
    pub fn score(
        &mut self,
        observation: &TilePlanes,
        ellipse: Option<[[f64; 2]; 2]>,
    ) -> BitmapScore {
        let zncc = windowed_zncc(self.bitmap, observation, &self.weights);
        let ellipse = ellipse.or_else(|| {
            read_tile_ellipse(
                &observation.values,
                observation.channels,
                observation.side,
                &observation.data,
            )
        });
        let sharper_than_bitmap = match (self.bitmap_semi_axes(), ellipse.as_ref()) {
            (Some([_, bitmap_minor]), Some(e)) => semi_axes(e)[0] < bitmap_minor,
            _ => false,
        };
        let plain = BitmapScore {
            zncc,
            blur_matched_zncc: zncc,
            blur_sigma: 0.0,
            sharper_than_bitmap,
        };
        let (Some(bitmap_ellipse), Some(e)) = (self.ellipse, ellipse) else {
            return plain;
        };
        let Some(blur) = bitmap_blur(&bitmap_ellipse, &e, DEFAULT_MIN_ELLIPSE_RATIO) else {
            return plain;
        };
        if self.assessment.is_none() {
            let bitmap = self.bitmap;
            let read = |values: &[f32]| {
                read_tile_ellipse(values, bitmap.channels, bitmap.side, &bitmap.data)
            };
            self.assessment = Some(assess_blur(
                bitmap,
                &bitmap_ellipse,
                read,
                &mut self.scratch,
            ));
        }
        let Some(assessment) = self.assessment.as_ref().and_then(Option::as_ref) else {
            return plain;
        };
        let Some(sigma) = blur_to_length_into(
            self.bitmap,
            assessment,
            blur.target,
            &mut self.blurred,
            &mut self.scratch,
        ) else {
            return plain;
        };
        if sigma <= 0.0 {
            return plain;
        }
        BitmapScore {
            blur_matched_zncc: windowed_zncc(&self.blurred, observation, &self.weights),
            blur_sigma: sigma,
            ..plain
        }
    }
}

/// Score each of `observations` against `bitmap` ([`BitmapScorer`]), over
/// `window`, `ellipses[v]` observation `v`'s self-similarity ellipse matrix
/// (`None` reads it). The observation `reference` names, whose render the
/// bitmap is, gets `None`: its score is `1` and is not computed.
///
/// ```
/// use sfmtool_core::patch::blur_matched::TilePlanes;
/// use sfmtool_core::patch::normal_refine::PatchWindow;
/// use sfmtool_core::patch::stored_bitmap::score_against_bitmap;
///
/// let side = 24;
/// let tile = |seed: usize| TilePlanes {
///     values: (0..side * side)
///         .map(|k| (((k + seed) * 7919) % 251) as f32)
///         .collect(),
///     data: vec![true; side * side],
///     side,
///     channels: 1,
/// };
/// let (bitmap, other) = (tile(0), tile(0));
/// let scores = score_against_bitmap(
///     &bitmap,
///     &[&bitmap, &other],
///     &[None, None],
///     Some(0),
///     PatchWindow::GaussianDisk { sigma: 0.6 },
/// );
/// assert!(scores[0].is_none());
/// assert!((scores[1].unwrap().zncc - 1.0).abs() < 1e-9);
/// ```
///
/// # Panics
///
/// Panics if `ellipses` is not parallel to `observations`, or a tile is not
/// the bitmap's side.
pub fn score_against_bitmap(
    bitmap: &TilePlanes,
    observations: &[&TilePlanes],
    ellipses: &[Option<[[f64; 2]; 2]>],
    reference: Option<usize>,
    window: PatchWindow,
) -> Vec<Option<BitmapScore>> {
    assert_eq!(
        ellipses.len(),
        observations.len(),
        "ellipses must be parallel to observations"
    );
    let mut scorer = BitmapScorer::new(bitmap, window);
    observations
        .iter()
        .zip(ellipses)
        .enumerate()
        .map(|(v, (observation, ellipse))| {
            (Some(v) != reference).then(|| scorer.score(observation, *ellipse))
        })
        .collect()
}
