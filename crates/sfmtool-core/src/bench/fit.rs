// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The fit: the step that moves a track, at whichever stage it is in.
//!
//! `specs/core/bench/editable-track.md` is the design. Where
//! [`evaluate`](super::evaluate::evaluate()) reads a track and writes only what it
//! read, [`fit`] is the modification: at the track stage it localizes every
//! sighting against the patch, refines each to sub-pixel, re-triangulates the
//! `in` results, renders the patch bitmap and writes the keypoints, the
//! position and the frame. It sets no verdict -- the thresholds propose and
//! [`apply_thresholds`](super::steps::apply_thresholds) applies the proposal --
//! so what it writes is geometry and a report, and a decision about nothing.
//!
//! **A fit ends by evaluating its own result.** Every number in a [`FitReport`]
//! and in the observations' slots is the reading's, so pressing *Fit* and then
//! *Evaluate* cannot produce two accounts of one track. The kernels run
//! gate-free and cap-free, for the reason
//! [`open_localizer`] gives: a sighting that
//! does not belong is turned out by the person or by a threshold, not deleted
//! from the evidence by a kernel.

use std::collections::HashMap;

use nalgebra::Point3;
use ndarray::Array3;

use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::{
    try_localize_patch_keypoints, KeypointLocalizeParams, LocalizeError,
};
use crate::patch::keypoint_subpixel::{refine_patch_keypoints_reporting, KeypointSubpixelParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::render_view_tile;
use crate::patch::stored_bitmap::{bitmap_from_tile, render_patch_bitmap};
use crate::progress::{Cancelled, Progress};
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;
use crate::reconstruction::triangulation::{
    triangulate_batch, Triangulation, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
};

use super::classify::{classify_track_rays, TrackClassification, TrackRays};
use super::evaluate::{
    check_observation_views, check_views, evaluate, evaluate_cluster, evaluated, finite,
    grid_distance, open_localizer, plan_rounds, seed_of, shown_bytes, EvaluateError,
    EvaluateOptions, EvaluateReport,
};
use super::track::{EditableTrack, Stage, StageKind, TrackPayload, Verdict};

/// What the kernels a fit runs are allowed to do, and how its result is read
/// back.
///
/// The localizer defaults to [`open_localizer`]: a fit that let the kernel drop
/// a sighting would be deciding, and deciding is the person's. The track's
/// [`Thresholds`](super::track::Thresholds) are not here for the same reason
/// they are not in [`EvaluateOptions`] -- a bar the person moved paints
/// verdicts and must not change what the numbers under the painting are.
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// The discrete localization kernel.
    ///
    /// Its [`resolution`](KeypointLocalizeParams::resolution), and
    /// [`Self::refine`]'s, apply only to a reconstruction that stores no patch
    /// bitmaps. Otherwise a track-stage fit runs both kernels at the
    /// reconstruction's own patch resolution
    /// ([`stored_patch_resolution`](super::evaluate::stored_patch_resolution)),
    /// the grid the evaluation it ends with reads at.
    pub localize: KeypointLocalizeParams,
    /// The sub-pixel kernel, chained after the discrete one, and the render of
    /// the patch bitmap: its resolution and sampler shape the reference view's
    /// tile, and its window and robust iterations the fused mean where that
    /// is stored.
    pub refine: KeypointSubpixelParams,
    /// The evaluation a fit ends with, which is where every number it reports
    /// comes from. A caller that reads with its own search radius fits with the
    /// same one, so the two buttons speak in one set of terms.
    pub evaluate: EvaluateOptions,
    /// The per-axis pixel noise the finite-versus-bearing classification
    /// weights each sighting's ray by, in source-image px. `None`, the default,
    /// is the reconstruction's own: the base's measured
    /// [`reprojection noise`](EditedReconstruction::base_reprojection_noise_px),
    /// taken once per base.
    ///
    /// Here rather than on [`Thresholds`](super::track::Thresholds) because it
    /// is not a bar a verdict is painted from: it is what the lens, the poses
    /// and the keypoints are worth, and moving it changes which representation
    /// the geometry earns rather than which sightings are kept.
    pub sigma_px: Option<f64>,
    /// The threshold the classification judges the depth score, the midpoint
    /// bound and the point fit's likelihood ratio against.
    /// [`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`] by default.
    pub depth_likelihood_ratio_threshold: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            localize: open_localizer(),
            refine: KeypointSubpixelParams::default(),
            evaluate: EvaluateOptions::default(),
            sigma_px: None,
            depth_likelihood_ratio_threshold: DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
        }
    }
}

impl FitOptions {
    /// These options with every track-stage kernel at `recon`'s patch
    /// resolution where it stores patch bitmaps: the localizer, the sub-pixel
    /// kernel and the evaluation the fit ends with, so the fit and its reading
    /// are stated in one grid.
    fn at_patch_resolution(&self, recon: &crate::SfmrReconstruction) -> Self {
        let mut options = self.clone();
        if let Some(resolution) = super::evaluate::stored_patch_resolution(recon) {
            let resolution = resolution.max(2);
            options.localize.resolution = resolution;
            options.refine.resolution = resolution;
            options.evaluate.localize.resolution = resolution;
        }
        options
    }
}

/// Why a fit was refused. Every variant names what did not hold, because the
/// caller is a button that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// Fewer decoded views were supplied than the reconstruction has images.
    ViewsMissing {
        /// How many were supplied.
        got: usize,
        /// How many the reconstruction holds.
        expected: usize,
    },
    /// An observation names an image no decoded, posed view was supplied for.
    NoView {
        /// The image it named.
        image: u32,
    },
    /// Fewer than two observations are `in`, so the track stage has no
    /// consensus to register against and nothing to triangulate.
    TooFewObservations(usize),
    /// The track carries no patch, so there is nothing the localizer can
    /// register a view against.
    NoFrame,
    /// The `in` observations do not triangulate: the linear solve came back with
    /// a coordinate that is not a number.
    ///
    /// Rays too nearly parallel to fix a depth are **not** this. They are a
    /// track at infinity, and the classification writes it as the bearing it is.
    Triangulation,
    /// One round's per-view tiles would take more memory than
    /// [`EvaluateOptions::max_cache_bytes`] allows, and were not attempted.
    TooLarge {
        /// What the round's tiles would have taken, in bytes.
        bytes: usize,
        /// The budget it passed.
        budget: usize,
    },
    /// A buffer the localizer needed could not be allocated.
    OutOfMemory {
        /// What the allocator refused, in bytes.
        bytes: usize,
    },
    /// Two or more `in` observations carry a pixel, but fewer than two of those
    /// pixels give a ray the camera model takes back to the pixel (one past a
    /// fisheye's wide-angle blend, say), so there is nothing to triangulate.
    TooFewRays {
        /// How many gave a ray.
        usable: usize,
        /// How many carried a pixel.
        sightings: usize,
    },
    /// No noise level to weight the rays by: no `sigma_px` was given and the
    /// reconstruction's could not be measured. The sentence says why.
    NoNoiseLevel(String),
    /// The caller asked the fit to stop.
    Cancelled,
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::ViewsMissing { got, expected } => write!(
                f,
                "{got} decoded views were supplied and the reconstruction has {expected} images"
            ),
            FitError::NoView { image } => {
                write!(
                    f,
                    "image {image} has no decoded view with a pose in this fit"
                )
            }
            FitError::TooFewObservations(n) => write!(
                f,
                "{n} observations are in, and the track stage needs two or more"
            ),
            FitError::NoFrame => write!(
                f,
                "the track carries no patch frame to register against, as a point put \
                 on the bench from a sift_files reconstruction has none; convert the \
                 reconstruction to embedded patches and put the point on the bench again"
            ),
            FitError::Triangulation => write!(
                f,
                "the in observations do not triangulate: the solve came back with a \
                 coordinate that is not a number"
            ),
            FitError::TooLarge { bytes, budget } => write!(
                f,
                "this round's tiles would take {} and the budget is {}; \
                 narrow the search or turn out the observations furthest from \
                 the projection",
                shown_bytes(*bytes),
                shown_bytes(*budget)
            ),
            FitError::OutOfMemory { bytes } => {
                write!(f, "{bytes} bytes could not be allocated for the fit")
            }
            FitError::TooFewRays { usable, sightings } => write!(
                f,
                "only {usable} of the {sightings} in observations give a ray the camera \
                 model takes back to its pixel, and the track stage needs two or more"
            ),
            FitError::NoNoiseLevel(why) => write!(
                f,
                "the track cannot be classified as a point or a bearing: {why}"
            ),
            FitError::Cancelled => write!(f, "the fit was cancelled"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<EvaluateError> for FitError {
    fn from(e: EvaluateError) -> Self {
        match e {
            EvaluateError::ViewsMissing { got, expected } => {
                FitError::ViewsMissing { got, expected }
            }
            EvaluateError::NoView { image } => FitError::NoView { image },
            EvaluateError::NoFrame => FitError::NoFrame,
            EvaluateError::TooLarge { bytes, budget } => FitError::TooLarge { bytes, budget },
            EvaluateError::OutOfMemory { bytes } => FitError::OutOfMemory { bytes },
            EvaluateError::Cancelled => FitError::Cancelled,
        }
    }
}

impl From<Cancelled> for FitError {
    fn from(_: Cancelled) -> Self {
        FitError::Cancelled
    }
}

impl From<LocalizeError> for FitError {
    fn from(e: LocalizeError) -> Self {
        match e {
            LocalizeError::OutOfMemory { bytes } => FitError::OutOfMemory { bytes },
            LocalizeError::Cancelled => FitError::Cancelled,
        }
    }
}

/// What one fit did.
#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    /// The reading of the fitted track. Every per-observation number a fit
    /// leaves behind is this evaluation's, so a fit and a reading of its result
    /// state one thing.
    pub evaluate: EvaluateReport,
    /// How many observations the localizer placed. A row it did not place keeps
    /// the pixel it had, so it is still in the triangulation and still read;
    /// this count is how many the kernels themselves moved.
    pub placed: usize,
    /// How many `in` sightings the localizer moved further than
    /// [`Thresholds::max_shift_px`](super::track::Thresholds::max_shift_px) and
    /// were therefore left at their seeds.
    pub kept_at_seed: usize,
    /// At the track stage, the coordinate the `in` observations resolved to: a
    /// world point, or a unit bearing when the classification put the track at
    /// infinity. Read [`Self::classification`] to tell which.
    pub position: Option<Point3<f64>>,
    /// At the track stage, that triangulation's condition number.
    pub condition_number: Option<f64>,
    /// At the track stage, which representation the rays earned and on which
    /// test. `None` at the cluster stage, which triangulates nothing.
    pub classification: Option<TrackClassification>,
}

impl std::fmt::Display for FitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.evaluate.stage {
            StageKind::Cluster => write!(f, "{}", self.evaluate),
            StageKind::Track => {
                write!(f, "placed {}", self.placed)?;
                if self.kept_at_seed > 0 {
                    write!(f, ", {} kept at seed", self.kept_at_seed)?;
                }
                if let Some(call) = &self.classification {
                    write!(f, ", {call}")?;
                }
                write!(f, ", {}", self.evaluate)
            }
        }
    }
}

/// Fit `track` at the stage it is in: localize, refine, re-triangulate and
/// render the patch bitmap at the track stage, and read the result back.
///
/// `images` is one [`ProjectedImage`] per image of `edited`, indexed by image
/// index, exactly as every other photometric step of the bench takes them.
///
/// **At the track stage** the patch is registered into every view of every
/// round by the two kernels the embed pass chains -- as the
/// track carries it, bearing and all -- the `in` results are re-triangulated,
/// the frame is placed at what they resolve to and the patch bitmap is rendered
/// over them. An observation the kernels did not place keeps the pixel it had,
/// so the column still says where the sighting is and the re-triangulation still
/// has its ray. The fitted track is then evaluated, which is where its
/// measurement slots and the report's counts come from.
///
/// **Which representation the track leaves with is the rays' to say.** The
/// re-triangulation goes through [`classify_track_rays`], the point-or-bearing
/// test the reconstruction's own reclassification decides with, at the
/// reconstruction's measured noise level unless [`FitOptions::sigma_px`] gives
/// one: a bearing whose sightings now carry a depth becomes a point at it, a
/// point whose rays no longer fix one becomes a bearing, and either way the
/// frame is carried across at the apparent size it had. A finite track is
/// placed by the point fit started from where it stood.
/// [`FitReport::classification`] says which it chose and on which outcome.
///
/// **A sighting the kernels would walk further than
/// [`Thresholds::max_shift_px`](super::track::Thresholds::max_shift_px) keeps
/// its seed.** The kernels themselves stay gate-free, so nothing is dropped;
/// what is bounded is where the fit may *put* a sighting, because a correlation
/// that jumped onto a similar detail elsewhere would otherwise hand the
/// re-triangulation a place the person never pointed at. Such a row says so in
/// [`TrackMeasurement::walked_px`](super::track::TrackMeasurement::walked_px)
/// and still casts its ray from the seed. The pixel the walk would have reached
/// and the ZNCC the localizer scored there are kept beside it
/// ([`TrackMeasurement::walked_to`](super::track::TrackMeasurement::walked_to),
/// [`TrackMeasurement::walked_zncc`](super::track::TrackMeasurement::walked_zncc)),
/// so a person can accept the walk with
/// [`sight_observation`](super::steps::sight_observation).
///
/// **At the cluster stage** a fit is the refinement, which is also what a
/// reading is: a cluster has no geometry behind it to move, so the one kernel
/// that reads the patches writes what it read and nothing else changes.
///
/// `progress` names the phases the batch kernels carry: `localize`, `refine` and
/// `bitmap` for the fit itself, then `localize`, `self-similarity`, `reference
/// view` and `bitmap scores` for the reading; `refine` and `self-similarity` at
/// the cluster stage.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{fit, FitOptions};
/// # use sfmtool_core::progress::Progress;
/// # use sfmtool_core::EditedReconstruction;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (fitted, report) = fit(
///     track,
///     edited,
///     images,
///     &FitOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}");           // "placed 12, measured 12 of 12 at (…)"
/// # let _ = fitted;
/// # Ok(())
/// # }
/// ```
pub fn fit(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &FitOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, FitReport), FitError> {
    check_views(edited, images)?;
    fit_preconditions(track)?;
    match &track.stage {
        Stage::Cluster(payload) => {
            let (next, report) =
                evaluate_cluster(track, payload, images, &options.evaluate, progress)?;
            Ok((
                next,
                FitReport {
                    placed: report.measured,
                    kept_at_seed: 0,
                    position: None,
                    condition_number: None,
                    classification: None,
                    evaluate: report,
                },
            ))
        }
        Stage::Track(payload) => {
            // The frame goes in as the track carries it, bearing and all: a
            // `w = 0` patch is tangent to the direction sphere and registering
            // against it is what reads the photographs the track actually has.
            // Which representation the track leaves with is the
            // re-triangulation's to say, not the frame it arrived on.
            let frame = payload.placement.clone().ok_or(FitError::NoFrame)?;
            let options = options.at_patch_resolution(&edited.base);
            fit_track(track, edited, images, &frame, &options, progress)
        }
    }
}

/// Render the stored bitmap of a track-stage track where it stands, and move
/// nothing.
///
/// "In place" is the patch's: the bitmap is rendered through the placement
/// the track already has. The track itself is not changed; the rendered track
/// is returned.
///
/// The render a [`fit`] ends its geometry with, for a track whose last step
/// moved the patch and so left no bitmap: `build_track_at_pixel` ends on a
/// slide of the patch onto the queried pixel, and renders with this before it
/// returns, and the viewer's live evaluation renders with it after a patch
/// step. The placement, the position, the verdicts and every keypoint come
/// back as they were; what is written is the tile of the track's reference
/// observation ([`TrackPayload::reference`]) where it holds one that is `in`
/// with a keypoint, and otherwise the tile of the `in` sighting the
/// reference-view rule picks (or the mean of the `in` sightings' tiles where
/// it picks none or reaches its pick only through its last fallback, see
/// [`ReferenceRender::stored_reference`](crate::patch::stored_bitmap::ReferenceRender::stored_reference)),
/// with [`TrackPayload::reference`] naming that sighting, on the
/// reconstruction's own bitmap grid where it stores one, and the colour at its
/// centre.
///
/// The rows' bitmap scores are not touched: a caller scores them against the
/// new bitmap with [`score_bitmap`](super::evaluate::score_bitmap), or runs
/// [`evaluate_rendering_bitmap`](super::evaluate::evaluate_rendering_bitmap),
/// which reads, renders and scores in one call.
///
/// A cluster, a track with no placement, and one with fewer than two `in`
/// sightings that carry a keypoint come back unchanged, since there is
/// nothing to render a bitmap from.
///
/// The track's [`RepaintMark`](super::track::RepaintMark) comes across: the
/// bitmap is nothing a verdict depends on, so an evaluation whose answer is
/// rendered here still hands on what its repaint did.
pub fn render_bitmap_in_place(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &FitOptions,
) -> EditableTrack {
    let mut next = rendered_in_place(track, edited, images, options);
    next.repaint = track.repaint.carried();
    next
}

/// [`render_bitmap_in_place`] before the mark is carried across.
fn rendered_in_place(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &FitOptions,
) -> EditableTrack {
    let Stage::Track(payload) = &track.stage else {
        return track.clone();
    };
    let Some(placement) = &payload.placement else {
        return track.clone();
    };
    let ins = track.in_observations();
    let (bitmap, reference, color) = render_bitmap(
        track,
        edited,
        images,
        placement,
        &ins,
        options,
        &Progress::none(),
    );
    let Some(bitmap) = bitmap else {
        return track.clone();
    };
    let mut next = track.clone();
    install_bitmap(&mut next, bitmap, reference, color);
    next
}

/// Write a rendered bitmap, the row it is the render of, and the colour at its
/// centre into `track`'s track-stage payload.
pub(super) fn install_bitmap(
    track: &mut EditableTrack,
    bitmap: Array3<u8>,
    reference: Option<usize>,
    color: Option<[u8; 3]>,
) {
    if let Stage::Track(payload) = &mut track.stage {
        payload.bitmap = Some(bitmap);
        payload.reference = reference;
        if let Some(color) = color {
            payload.color = color;
        }
    }
}

/// The `(resolution, channels)` a track's bitmap is rendered at: the
/// reconstruction's own bitmap grid where it stores one, and otherwise the
/// sub-pixel kernel's resolution with four channels.
pub(super) fn bitmap_layout(edited: &EditedReconstruction, options: &FitOptions) -> (usize, usize) {
    match edited.base.point_set.patch_bitmaps_y_x_rgba.as_ref() {
        Some(bitmaps) => (bitmaps.shape()[1], bitmaps.shape()[3]),
        None => (options.refine.resolution.max(2) as usize, 4),
    }
}

/// An `R·R·4` RGBA render as a `(R, R, channels)` bitmap, keeping as many
/// channels as the column carries, and the colour at its centre.
pub(super) fn column_bitmap(
    rgba: &[u8],
    resolution: usize,
    channels: usize,
) -> (Array3<u8>, [u8; 3]) {
    let mut bitmap = Array3::<u8>::zeros((resolution, resolution, channels));
    for row in 0..resolution {
        for col in 0..resolution {
            for c in 0..channels {
                bitmap[[row, col, c]] = if c < 4 {
                    rgba[(row * resolution + col) * 4 + c]
                } else {
                    u8::MAX
                };
            }
        }
    }
    let (row, col) = (resolution / 2, resolution / 2);
    let mut color = [0u8; 3];
    for (c, out) in color.iter_mut().enumerate() {
        *out = bitmap[[row, col, if channels >= 3 { c } else { 0 }]];
    }
    (bitmap, color)
}

/// Whether `track` can be fitted at the stage it stands in, judged on the track
/// alone.
///
/// The half of [`fit`]'s validation that reads no photograph: whether the track
/// stage has a patch to register against, and whether enough observations are
/// `in` for the consensus it registers against to exist. A caller that runs the
/// fit somewhere expensive -- on a worker, after decoding a dozen images -- asks
/// this first and refuses in front of the decode.
///
/// The two-in rule is a fit's and not a reading's: fitting one sighting against
/// nothing would move it to wherever a template of itself sits, while *reading*
/// one sighting is a report that it has nothing to correlate against. That is
/// why [`evaluate_preconditions`](super::evaluate::evaluate_preconditions) has
/// no such bar.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{fit_preconditions, EditableTrack};
/// # fn run(track: &EditableTrack) -> Result<(), Box<dyn std::error::Error>> {
/// fit_preconditions(track)?;   // refuse here, before a photograph is read
/// # Ok(())
/// # }
/// ```
pub fn fit_preconditions(track: &EditableTrack) -> Result<(), FitError> {
    let Stage::Track(payload) = &track.stage else {
        return Ok(());
    };
    if payload.placement.is_none() {
        return Err(FitError::NoFrame);
    }
    let ins = track.in_observations().len();
    if ins < 2 {
        return Err(FitError::TooFewObservations(ins));
    }
    Ok(())
}

/// How large `frame` looks in `view`, in source-image px: the geometric mean of
/// its two projected half-axes, each half the pixel distance between the
/// projections of the patch's two opposite edge midpoints.
///
/// Measured through the view's own camera model, so a fisheye's compression
/// toward the rim of its image circle is in the number. The edge midpoints are
/// homogeneous with the frame's own `w`, so a bearing's project as directions.
/// `None` where one of them does not project or the size is not a positive
/// finite number.
fn projected_size_px(frame: &OrientedPatch, view: &ProjectedImage<'_>) -> Option<f64> {
    let project = |s: f64, t: f64| -> Option<[f64; 2]> {
        let (xyz, w) = frame.corner_homogeneous(s, t);
        let cam = view.cam_from_world.transform_point_homogeneous(xyz, w);
        let (x, y) = view.camera.ray_to_pixel([cam.x, cam.y, cam.z])?;
        Some([x, y])
    };
    let half_axis = |a: [f64; 2], b: [f64; 2]| 0.5 * (a[0] - b[0]).hypot(a[1] - b[1]);
    let u = half_axis(project(1.0, 0.0)?, project(-1.0, 0.0)?);
    let v = half_axis(project(0.0, 1.0)?, project(0.0, -1.0)?);
    let size = (u * v).sqrt();
    (size.is_finite() && size > 0.0).then_some(size)
}

/// The factor `candidate`'s half-extents are multiplied by so that it looks the
/// size `target` looks in the views `in_images` names, or `None` when no view
/// measures both.
///
/// With `t_i` the target's projected size in view `i` and `c_i` the
/// candidate's ([`projected_size_px`]), a small patch's projected size is
/// linear in its half-extent, so the scaled candidate measures `k c_i`, and the
/// `k` that minimises `sum_i (ln(k c_i) - ln t_i)^2` is
/// `exp(mean_i ln(t_i / c_i))`: the geometric mean of the per-view ratios. The
/// fit is in logarithms because the error is a ratio -- a view where the patch
/// comes out twice too large and one where it comes out half the size are
/// equally wrong -- and so that a view which sees the patch much larger than the
/// others, from a camera much closer to it, does not outvote them by the size
/// of its numbers.
fn size_matching_factor(
    target: &OrientedPatch,
    candidate: &OrientedPatch,
    images: &[ProjectedImage<'_>],
    in_images: &[usize],
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for view in in_images.iter().filter_map(|&i| images.get(i)) {
        let (Some(t), Some(c)) = (
            projected_size_px(target, view),
            projected_size_px(candidate, view),
        ) else {
            continue;
        };
        sum += (t / c).ln();
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let k = (sum / count as f64).exp();
    (k.is_finite() && k > 0.0).then_some(k)
}

/// The geometric mean of the distances from the cameras of `in_images` to
/// `position`, or `None` when none of them is a positive finite distance.
///
/// The distance at which an angular extent and a world one are the same size
/// for a camera looking straight at the patch, so it is where the size match
/// starts: [`size_matching_factor`]'s linearity then only has to hold across
/// the small correction that is left, rather than across the whole distance
/// from one unit to the patch.
fn observing_distance(
    position: &Point3<f64>,
    images: &[ProjectedImage<'_>],
    in_images: &[usize],
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for view in in_images.iter().filter_map(|&i| images.get(i)) {
        let d = (position - view.cam_from_world.inverse_translation_origin()).norm();
        if d.is_finite() && d > 0.0 {
            sum += d.ln();
            count += 1;
        }
    }
    (count > 0).then(|| (sum / count as f64).exp())
}

/// `half_extent` multiplied by `by` on both axes.
fn scaled_extent(half_extent: [f64; 2], by: f64) -> [f64; 2] {
    [half_extent[0] * by, half_extent[1] * by]
}

/// The patch the fit's classification says the track now stands on --
/// `coordinate`, a place or a bearing as `at_infinity` says -- built from the
/// one it was fitted against.
///
/// Four cases:
///
/// - **Bearing to bearing.** The direction moves to the refined one and the
///   tangent frame is re-pinned on it, at the half-extents it already had, which
///   are angular.
/// - **Bearing to point.** The frame keeps its axes and takes the world
///   half-extents that make the patch look, in each image `in_images` names, as
///   large as the bearing did there, as nearly as one scale can
///   ([`size_matching_factor`]).
/// - **Point to bearing.** The frame is re-expressed as the tangent one the
///   format states for a `w = 0` row, and takes the angular half-extents that
///   best keep the size the point's patch had in those images, by the same
///   criterion. One criterion for both directions is what makes a round trip
///   keep the size.
/// - **Point to point.** The centre moves and the frame keeps its axes, with
///   the world half-extents scaled to keep the size the patch looked in those
///   images. A fit that moves the point along its rays, from 170 units out to
///   50, would otherwise grow the patch more than threefold in every image.
///
/// The sizes are those in the `in` observations' images because the patch in
/// those images is what the person judges and what the next round registers.
/// The distance to the camera-cloud centroid, which
/// `SfmrReconstruction::materialize_points_at_infinity` places a whole
/// reconstruction's bearings by, is not a size in any photograph: a point two
/// units from the three cameras that see it and eleven from the centroid would
/// come out five times too large in every one of them.
///
/// Where no `in` view measures both patches -- none projects, or a size is not
/// finite -- the half-extents are the ones a camera at the observing cameras'
/// geometric mean distance ([`observing_distance`]), looking straight at the
/// patch, would see at the same size; and where that distance is not defined
/// either, the numbers are carried over unchanged.
pub(super) fn placed_frame(
    frame: &OrientedPatch,
    coordinate: Point3<f64>,
    at_infinity: bool,
    images: &[ProjectedImage<'_>],
    in_images: &[usize],
) -> OrientedPatch {
    match (frame.w == 0.0, at_infinity) {
        (true, true) => {
            OrientedPatch::from_infinity_direction(coordinate, frame.v_axis, frame.half_extent)
        }
        (true, false) => {
            let start = observing_distance(&coordinate, images, in_images).unwrap_or(1.0);
            let trial = OrientedPatch::new(
                coordinate,
                frame.u_axis,
                frame.v_axis,
                scaled_extent(frame.half_extent, start),
            );
            let k = size_matching_factor(frame, &trial, images, in_images).unwrap_or(1.0);
            OrientedPatch {
                half_extent: scaled_extent(trial.half_extent, k),
                ..trial
            }
        }
        (false, true) => {
            let start = observing_distance(&frame.center, images, in_images)
                .map_or(1.0, |distance| 1.0 / distance);
            let trial = OrientedPatch::from_infinity_direction(
                coordinate,
                frame.v_axis,
                scaled_extent(frame.half_extent, start),
            );
            let k = size_matching_factor(frame, &trial, images, in_images).unwrap_or(1.0);
            OrientedPatch {
                half_extent: scaled_extent(trial.half_extent, k),
                ..trial
            }
        }
        (false, false) => {
            let trial = OrientedPatch {
                center: coordinate,
                u_axis: frame.u_axis,
                v_axis: frame.v_axis,
                half_extent: frame.half_extent,
                w: 1.0,
            };
            let k = size_matching_factor(frame, &trial, images, in_images).unwrap_or(1.0);
            OrientedPatch {
                half_extent: scaled_extent(trial.half_extent, k),
                ..trial
            }
        }
    }
}

/// Register `frame` into every view, re-triangulate the `in` results, render
/// the patch bitmap, write the geometry, and read the whole of it back.
///
/// Shared by [`fit`] at the track stage and by the upgrade
/// ([`set_stage`](super::stage::set_stage)), which differ only in where the
/// frame came from: the payload's own, or one framed from the cluster's
/// reference at the triangulated depth.
pub(super) fn fit_track(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    options: &FitOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, FitReport), FitError> {
    check_observation_views(track, images)?;
    // A patch is square. A frame that arrives otherwise -- written before the
    // upgrade framed square -- is squared before anything registers against
    // it, so a fit is also the repair.
    let squared = frame.squared();
    let frame = &squared;
    // The `in` count is [`fit_preconditions`]'s, checked before the caller
    // spent anything on the views.
    let ins = track.in_observations();
    // The rounds are the reading's, bound and all: an observation the evaluation
    // that follows will refuse to read for sitting too far from the projection
    // is one the kernels do not move either, so the fit and its own report
    // cannot come to disagree about which sightings were in play.
    let plan = plan_rounds(
        track,
        images,
        frame,
        &options.localize,
        options.evaluate.max_seed_offset_px,
    );

    let mut fits: HashMap<usize, Fit> = HashMap::new();
    fit_round(
        track,
        images,
        frame,
        &plan.first,
        options,
        progress,
        &mut fits,
    )?;
    for &i in &plan.contested {
        progress.check_cancel()?;
        let round = plan.contested_round(track, i);
        fit_round(track, images, frame, &round, options, progress, &mut fits)?;
    }

    // ── The keypoints, and the re-triangulation they feed ──
    //
    // **The walk is bounded by the person's own bar.** The kernels run gate-free
    // so that nothing is dropped, but a sighting the correlation carried further
    // than `max_shift_px` from its seed has been walked onto some other piece of
    // the photograph, and taking that pixel would feed the re-triangulation a
    // place the person never pointed at. Such a row keeps its seed, says on its
    // own row how far the peak sat, and still casts its ray -- from the seed.
    let bound = track.thresholds.max_shift_px;
    let mut next = track.clone();
    let mut kept_at_seed = 0usize;
    for i in evaluated(track) {
        let mut measurement = next.observations[i].track.clone().unwrap_or_default();
        let seed = seed_of(&track.observations[i]);
        measurement.walked_px = None;
        measurement.walked_to = None;
        measurement.walked_zncc = None;
        measurement.walked_zncc_middle = None;
        measurement.walked_zncc_grid = None;
        measurement.keypoint = match fits.get(&i) {
            Some(fit) => {
                // In grid px, the unit of the bar.
                let view = &images[track.observations[i].image as usize];
                let walked =
                    seed.map(|s| grid_distance(frame, view, s, fit.keypoint, &options.localize));
                match (walked, seed) {
                    (Some(walked), Some(seed)) if walked.is_finite() && walked > bound => {
                        // Where the walk would have gone and what it scored
                        // there, so a person can overrule the bar.
                        measurement.walked_px = Some(walked);
                        measurement.walked_to = Some(fit.keypoint);
                        measurement.walked_zncc = fit.zncc;
                        measurement.walked_zncc_middle = fit.zncc_middle;
                        measurement.walked_zncc_grid = fit.zncc_grid;
                        kept_at_seed += 1;
                        Some([seed[0] as f32, seed[1] as f32])
                    }
                    _ => Some([fit.keypoint[0] as f32, fit.keypoint[1] as f32]),
                }
            }
            // The kernels did not place it: out of frame, or in no round at
            // all. Its keypoint is then wherever it already sat -- the one it
            // arrived with, or the cluster stage's refined position for a track
            // that has just been upgraded -- so the column still says where this
            // sighting is and the re-triangulation still has its ray.
            None => measurement
                .keypoint
                .or_else(|| seed.map(|p| [p[0] as f32, p[1] as f32])),
        };
        next.observations[i].track = Some(measurement);
    }

    // ── What the rays resolve to: a point, or a bearing ──
    //
    // The one classification every bench step that triangulates goes through, so
    // a fit cannot write a depth the geometry does not carry -- nor refuse a
    // track whose sightings have just given it one.
    let sigma_px = noise_level(edited, options)?;
    let (triangulation, rays) = triangulate_keypoints(&next, images, &ins, sigma_px)?;
    let held = track
        .track()
        .and_then(|p| p.position.map(|x| (x, p.at_infinity)));
    let classification = classify(&rays, held, edited, options)?;
    let position = classification.coordinate;

    // ── The frame at the coordinate the fit found, and the consensus it shows ──
    let in_images: Vec<usize> = ins
        .iter()
        .map(|&i| next.observations[i].image as usize)
        .collect();
    let placed = placed_frame(
        frame,
        classification.coordinate,
        classification.at_infinity,
        images,
        &in_images,
    );
    let (bitmap, reference, color) = {
        let mut phase = progress.phase("bitmap");
        let rendered = render_bitmap(&next, edited, images, &placed, &ins, options, &phase);
        progress_note!(phase, "{} observations", ins.len());
        rendered
    };

    let previous = track.track();
    // A render that drew nothing leaves the reference the track carried, as
    // `render_bitmap_in_place` does, where that observation is still `in`.
    let reference = if bitmap.is_some() {
        reference
    } else {
        previous.and_then(|p| p.reference).filter(|&r| {
            next.observations
                .get(r)
                .is_some_and(|o| o.verdict == Verdict::In)
        })
    };
    next.stage = Stage::Track(TrackPayload {
        position: Some(position),
        at_infinity: classification.at_infinity,
        placement: Some(placed),
        color: color
            .or_else(|| previous.map(|p| p.color))
            .unwrap_or([0; 3]),
        bitmap,
        reference,
        normal_confidence: previous.and_then(|p| p.normal_confidence),
        condition_number: finite(triangulation.condition_number),
    });

    // ── What the fitted track says about itself ──
    let (read, report) = evaluate(&next, edited, images, &options.evaluate, progress)?;
    Ok((
        read,
        FitReport {
            evaluate: report,
            placed: fits.len(),
            kept_at_seed,
            position: Some(position),
            condition_number: finite(triangulation.condition_number),
            classification: Some(classification),
        },
    ))
}

/// What one round's two kernels said about one observation.
struct Fit {
    /// Where the sub-pixel stage left the keypoint, in source-image px.
    keypoint: [f64; 2],
    /// The localizer's leave-one-out ZNCC for this view against the round's
    /// consensus, at the peak it registered to, when it scored one.
    zncc: Option<f64>,
    /// The same reading over the middle of the tile, when there is one.
    zncc_middle: Option<f64>,
    /// The same reading over each cell of the ZNCC grid, when `zncc` is there.
    zncc_grid: Option<[[f64; 3]; 3]>,
}

/// Localize and refine one round's observations against `frame`, and record
/// where each landed.
///
/// The round holds one observation per image, so an image index names exactly
/// one of its observations and the kernels' per-view answers scatter back
/// without ambiguity.
fn fit_round(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    round: &[usize],
    options: &FitOptions,
    progress: &Progress<'_>,
    fits: &mut HashMap<usize, Fit>,
) -> Result<(), FitError> {
    if round.len() < 2 {
        return Ok(());
    }
    let view_set: Vec<u32> = round.iter().map(|&i| track.observations[i].image).collect();
    let seeds: Vec<Option<[f64; 2]>> = round
        .iter()
        .map(|&i| seed_of(&track.observations[i]))
        .collect();
    let of_image: HashMap<u32, usize> = view_set
        .iter()
        .copied()
        .zip(round.iter().copied())
        .collect();

    // The window a fit searches is the localizer's own rather than the widened
    // one a reading runs at, so its tiles are the kernel's default size; the
    // budget the reading states is checked there, where the widening happens.
    let localized = {
        let mut phase = progress.phase("localize");
        let localized = try_localize_patch_keypoints(
            frame,
            images,
            &view_set,
            Some(&seeds),
            &options.localize,
            progress,
        )?;
        progress_note!(phase, "{} views", localized.views.len());
        localized
    };
    let refined = {
        let mut phase = progress.phase("refine");
        let refined = refine_patch_keypoints_reporting(
            frame,
            images,
            &localized.views,
            Some(
                &localized
                    .keypoints
                    .iter()
                    .map(|&k| Some(k))
                    .collect::<Vec<_>>(),
            ),
            &options.refine,
            &phase,
        );
        progress_note!(phase, "{} views", refined.views.len());
        refined
    };

    for (slot, &image) in localized.views.iter().enumerate() {
        let Some(&i) = of_image.get(&image) else {
            continue;
        };
        let at = refined.views.iter().position(|&v| v == image);
        let keypoint = at.map_or(localized.keypoints[slot], |k| refined.keypoints[k]);
        let zncc = localized
            .loo_zncc
            .get(slot)
            .copied()
            .filter(|z| z.is_finite());
        let zncc_middle = localized
            .loo_zncc_middle
            .get(slot)
            .copied()
            .filter(|z| zncc.is_some() && z.is_finite());
        let zncc_grid = localized
            .loo_zncc_grid
            .get(slot)
            .copied()
            .filter(|_| zncc.is_some());
        fits.insert(
            i,
            Fit {
                keypoint,
                zncc,
                zncc_middle,
                zncc_grid,
            },
        );
    }
    Ok(())
}

/// The noise level a classification weights a track's rays by: the caller's,
/// or the base's measured reprojection noise, which the reconstruction keeps
/// once measured. The measure is never finer than the base's keypoints can be
/// stored at (`keypoint_resolution_px`), so keypoints at the exact projections
/// of their points still weight their rays, as they do for reclassification
/// and discovery.
pub(super) fn noise_level(
    edited: &EditedReconstruction,
    options: &FitOptions,
) -> Result<f64, FitError> {
    match options.sigma_px {
        Some(s) if s.is_finite() && s > 0.0 => Ok(s),
        Some(s) => Err(FitError::NoNoiseLevel(format!(
            "a noise level of {} px weights no ray",
            crate::readable::Readable(s)
        ))),
        None => edited
            .base_reprojection_noise_px()
            .map_err(FitError::NoNoiseLevel),
    }
}

/// [`classify_track_rays`] at the reconstruction's minimum point depth and the
/// options' threshold, for a track that held `held`.
pub(super) fn classify(
    rays: &TrackRays,
    held: Option<(Point3<f64>, bool)>,
    edited: &EditedReconstruction,
    options: &FitOptions,
) -> Result<TrackClassification, FitError> {
    classify_track_rays(
        rays,
        held,
        edited.base_min_point_depth(),
        options.depth_likelihood_ratio_threshold,
    )
    .ok_or(FitError::Triangulation)
}

/// Triangulate the observations at `which` from the keypoints they carry, and
/// give back the rays as well as the solve.
///
/// The rays go back to the caller because the classification is a statement
/// about *them* rather than about the point, and a near-parallel track's point
/// is the one thing in the answer that does not mean anything.
fn triangulate_keypoints(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    which: &[usize],
    sigma_px: f64,
) -> Result<(Triangulation, TrackRays), FitError> {
    let mut rays: Vec<([f64; 2], usize)> = Vec::with_capacity(which.len());
    for &i in which {
        let observation = &track.observations[i];
        if let Some(keypoint) = observation.track.as_ref().and_then(|m| m.keypoint) {
            rays.push((
                [f64::from(keypoint[0]), f64::from(keypoint[1])],
                observation.image as usize,
            ));
        }
    }
    triangulate_rays(&rays, images, sigma_px)
}

/// Triangulate the `in` observations from wherever they currently sit: the
/// cluster stage's refined positions, or their seeds.
pub(super) fn triangulate_in_seeds(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    sigma_px: f64,
) -> Result<(Triangulation, TrackRays), FitError> {
    let mut rays: Vec<([f64; 2], usize)> = Vec::new();
    for &i in &track.in_observations() {
        let observation = &track.observations[i];
        if (observation.image as usize) < images.len() {
            if let Some(seed) = seed_of(observation) {
                rays.push((seed, observation.image as usize));
            }
        }
    }
    triangulate_rays(&rays, images, sigma_px)
}

/// The batch triangulator over `(pixel, image)` pairs, with the crate's
/// camera-to-world convention, and the rays it solved over, weighted at
/// `sigma_px` for the classification.
///
/// A sighting whose pixel the camera model cannot take back to a ray
/// ([`TrackRays::of_sightings`]) is in neither, so the solve and the test read
/// one set of rays.
///
/// **The only refusal left here is a non-finite solve.** An infinite condition
/// number and a point behind a camera used to be refusals too, and both are
/// exactly what a track at infinity looks like: the classification reads them as
/// the evidence for a bearing rather than as a broken solve, so refusing here
/// would take the answer away from the step whose job it is to give one.
pub(super) fn triangulate_rays(
    rays: &[([f64; 2], usize)],
    images: &[ProjectedImage<'_>],
    sigma_px: f64,
) -> Result<(Triangulation, TrackRays), FitError> {
    if rays.len() < 2 {
        return Err(FitError::TooFewObservations(rays.len()));
    }
    let built = TrackRays::of_sightings(rays, images, sigma_px);
    if built.len() < 2 {
        return Err(FitError::TooFewRays {
            usable: built.len(),
            sightings: rays.len(),
        });
    }
    let offsets = [0usize, built.len()];
    let triangulation = triangulate_batch(&built.dirs, &built.centers, &offsets)[0];
    if !triangulation.point.coords.iter().all(|c| c.is_finite()) {
        return Err(FitError::Triangulation);
    }
    Ok((triangulation, built))
}

/// Render the track's stored bitmap from the `in` observations at their final
/// keypoints, and read the point's colour off its centre.
///
/// Where the track holds a defined reference observation
/// ([`TrackPayload::reference`]) that is one of the `in` rows with a keypoint,
/// the bitmap is that row's tile at its keypoint ([`render_view_tile`]), and
/// the reference stays. Otherwise the reference is undefined and the bitmap is
/// [`render_patch_bitmap`]: the tile of the observation the reference-view
/// rule picks among the `in` observations, or the fused mean where it picks
/// none or reaches its pick only through its last fallback. Nothing moves: the
/// keypoints are the ones the fit already settled. The grid is the
/// reconstruction's own bitmap grid where it stores one, so what is rendered
/// is a tile the column can hold. The second value is the row of the track
/// whose tile the bitmap is.
fn render_bitmap(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    patch: &OrientedPatch,
    ins: &[usize],
    options: &FitOptions,
    progress: &Progress<'_>,
) -> (Option<Array3<u8>>, Option<usize>, Option<[u8; 3]>) {
    let (resolution, channels) = bitmap_layout(edited, options);
    let mut rows: Vec<usize> = Vec::with_capacity(ins.len());
    let mut view_set: Vec<u32> = Vec::with_capacity(ins.len());
    let mut keypoints: Vec<[f64; 2]> = Vec::with_capacity(ins.len());
    for &i in ins {
        let observation = &track.observations[i];
        let Some(keypoint) = observation.track.as_ref().and_then(|m| m.keypoint) else {
            continue;
        };
        rows.push(i);
        view_set.push(observation.image);
        keypoints.push([f64::from(keypoint[0]), f64::from(keypoint[1])]);
    }
    if view_set.len() < 2 {
        return (None, None, None);
    }
    // A defined reference is rendered from; the rule sets one only where the
    // track holds none.
    let held = track
        .track()
        .and_then(|p| p.reference)
        .and_then(|r| rows.iter().position(|&i| i == r));
    if let Some(k) = held {
        let tile = render_view_tile(
            patch,
            &images[view_set[k] as usize],
            Some(keypoints[k]),
            resolution,
            options.refine.sampler,
            progress,
        );
        let (bitmap, color) = column_bitmap(&bitmap_from_tile(&tile), resolution, channels);
        return (Some(bitmap), Some(rows[k]), Some(color));
    }
    let params = KeypointSubpixelParams {
        resolution: resolution as u32,
        ..options.refine.clone()
    };
    let Some(rendered) =
        render_patch_bitmap(patch, images, &view_set, &keypoints, &params, progress)
    else {
        return (None, None, None);
    };
    // The kernel renders RGBA; the column takes as many channels as it carries.
    let (bitmap, color) = column_bitmap(&rendered.rgba, resolution, channels);
    (
        Some(bitmap),
        rendered.reference.map(|r| rows[r]),
        Some(color),
    )
}
