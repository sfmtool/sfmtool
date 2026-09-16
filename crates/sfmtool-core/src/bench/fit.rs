// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The fit: the step that moves a track, at whichever stage it is in.
//!
//! `specs/core/bench/editable-track.md` is the design. Where
//! [`evaluate`](super::evaluate::evaluate()) reads a track and writes only what it
//! read, [`fit`] is the modification: at the track stage it localizes every
//! sighting against the surfel, refines each to sub-pixel, re-triangulates the
//! `in` results, fuses the consensus bitmap and writes the keypoints, the
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

use nalgebra::{Point3, Vector3};
use ndarray::Array3;

use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::{localize_patch_keypoints, KeypointLocalizeParams};
use crate::patch::keypoint_subpixel::{refine_patch_keypoints, KeypointSubpixelParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::progress_note;
use crate::reconstruction::add_observation::placement_scale;
use crate::reconstruction::edited::EditedReconstruction;
use crate::reconstruction::triangulation::{triangulate_batch, Triangulation};

use super::evaluate::{
    check_observation_views, check_views, evaluate, evaluate_cluster, evaluated, finite,
    open_localizer, plan_rounds, seed_of, EvaluateError, EvaluateOptions, EvaluateReport,
};
use super::track::{EditableTrack, Stage, StageKind, TrackPayload};

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
    pub localize: KeypointLocalizeParams,
    /// The sub-pixel kernel, chained after the discrete one, and the fuse that
    /// renders the consensus bitmap.
    pub refine: KeypointSubpixelParams,
    /// The evaluation a fit ends with, which is where every number it reports
    /// comes from. A caller that reads with its own search radius fits with the
    /// same one, so the two buttons speak in one set of terms.
    pub evaluate: EvaluateOptions,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            localize: open_localizer(),
            refine: KeypointSubpixelParams::default(),
            evaluate: EvaluateOptions::default(),
        }
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
    /// The track carries no patch frame, so there is no surfel the localizer
    /// can register a view against.
    NoFrame,
    /// The `in` observations do not triangulate: the depth is not observable,
    /// or the solve puts the point behind a camera that sees it.
    Triangulation,
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
                "the track carries no patch frame to register against; upgrade it \
                 from the cluster stage to build one"
            ),
            FitError::Triangulation => write!(
                f,
                "the in observations do not triangulate: the depth is not observable, \
                 or the point falls behind a camera that sees it"
            ),
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
    /// At the track stage, where the `in` observations triangulated.
    pub position: Option<Point3<f64>>,
    /// At the track stage, that triangulation's condition number.
    pub condition_number: Option<f64>,
}

impl std::fmt::Display for FitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.evaluate.stage {
            StageKind::Cluster => write!(f, "{}", self.evaluate),
            StageKind::Track => write!(f, "placed {}, {}", self.placed, self.evaluate),
        }
    }
}

/// Fit `track` at the stage it is in: localize, refine, re-triangulate and fuse
/// at the track stage, and read the result back.
///
/// `images` is one [`ProjectedImage`] per image of `edited`, indexed by image
/// index, exactly as
/// [`add_observation`](crate::reconstruction::add_observation::add_observation)
/// takes them.
///
/// **At the track stage** the surfel is registered into every view of every
/// round by the two kernels the embed pass and `add_observation` chain, the `in`
/// results are re-triangulated, the frame is re-centred there and the consensus
/// bitmap is fused over them. An observation the kernels did not place keeps the
/// pixel it had, so the column still says where the sighting is and the
/// re-triangulation still has its ray. The fitted track is then evaluated, which
/// is where its measurement slots and the report's counts come from.
///
/// **At the cluster stage** a fit is the refinement, which is also what a
/// reading is: a cluster has no geometry behind it to move, so the one kernel
/// that reads the patches writes what it read and nothing else changes.
///
/// `progress` names the phases the batch kernels carry: `localize`, `refine` and
/// `fuse` for the fit itself, then `localize` and `localizability` for the
/// reading; `refine` and `localizability` at the cluster stage.
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
                    position: None,
                    condition_number: None,
                    evaluate: report,
                },
            ))
        }
        Stage::Track(payload) => {
            let frame = payload.frame.clone().ok_or(FitError::NoFrame)?;
            let frame = finite_frame(track, &frame, images)?;
            fit_track(track, edited, images, &frame, options, progress)
        }
    }
}

/// Whether `track` can be fitted at the stage it stands in, judged on the track
/// alone.
///
/// The half of [`fit`]'s validation that reads no photograph: whether the track
/// stage has a surfel to register against, and whether enough observations are
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
    if payload.frame.is_none() {
        return Err(FitError::NoFrame);
    }
    let ins = track.in_observations().len();
    if ins < 2 {
        return Err(FitError::TooFewObservations(ins));
    }
    Ok(())
}

/// The surfel a track-stage fit registers against, made finite.
///
/// A track whose point is at infinity has a frame tangent to the direction
/// sphere, and registering against one would pull every sighting back onto the
/// bearing's projection and undo the depth the other sightings carry. So the
/// seeds are triangulated first and the frame is promoted to the depth that
/// gives -- the same two-pass shape
/// [`add_observation`](mod@crate::reconstruction::add_observation) takes, and the
/// same rescale, measured from the camera-cloud centroid.
fn finite_frame(
    track: &EditableTrack,
    frame: &OrientedPatch,
    images: &[ProjectedImage<'_>],
) -> Result<OrientedPatch, FitError> {
    if frame.w != 0.0 {
        return Ok(frame.clone());
    }
    let provisional = triangulate_in_seeds(track, images)?.point;
    let scale = placement_scale(&provisional, images);
    if !(scale.is_finite() && scale > 0.0) {
        return Err(FitError::Triangulation);
    }
    Ok(OrientedPatch::new(
        provisional,
        frame.u_axis,
        frame.v_axis,
        [frame.half_extent[0] * scale, frame.half_extent[1] * scale],
    ))
}

/// Register `frame` into every view, re-triangulate the `in` results, fuse the
/// consensus bitmap, write the geometry, and read the whole of it back.
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
    // The `in` count is [`fit_preconditions`]'s, checked before the caller
    // spent anything on the views.
    let ins = track.in_observations();
    let plan = plan_rounds(track, images, frame, options.localize.min_grazing_cos);

    let mut fits: HashMap<usize, Fit> = HashMap::new();
    fit_round(
        track,
        images,
        frame,
        &plan.first,
        options,
        progress,
        &mut fits,
    );
    for &i in &plan.contested {
        let round = plan.contested_round(track, i);
        fit_round(track, images, frame, &round, options, progress, &mut fits);
    }

    // ── The keypoints, and the re-triangulation they feed ──
    let mut next = track.clone();
    for i in evaluated(track) {
        let mut measurement = next.observations[i].track.clone().unwrap_or_default();
        measurement.keypoint = match fits.get(&i) {
            Some(fit) => Some([fit.keypoint[0] as f32, fit.keypoint[1] as f32]),
            // The kernels did not place it: out of frame, or in no round at
            // all. Its keypoint is then wherever it already sat -- the one it
            // arrived with, or the cluster stage's refined position for a track
            // that has just been upgraded -- so the column still says where this
            // sighting is and the re-triangulation still has its ray.
            None => measurement
                .keypoint
                .or_else(|| seed_of(&track.observations[i]).map(|p| [p[0] as f32, p[1] as f32])),
        };
        next.observations[i].track = Some(measurement);
    }

    let triangulation = triangulate_keypoints(&next, images, &ins)?;
    let position = triangulation.point;

    // ── The frame at the position the fit found, and the consensus it shows ──
    let placed = OrientedPatch {
        center: position,
        u_axis: frame.u_axis,
        v_axis: frame.v_axis,
        half_extent: frame.half_extent,
        w: 1.0,
    };
    let (bitmap, color) = {
        let mut phase = progress.phase("fuse");
        let fused = fuse_bitmap(&next, edited, images, &placed, &ins, options);
        progress_note!(phase, "{} observations", ins.len());
        fused
    };

    let previous = track.track();
    next.stage = Stage::Track(TrackPayload {
        position: Some(position),
        frame: Some(placed),
        color: color
            .or_else(|| previous.map(|p| p.color))
            .unwrap_or([0; 3]),
        bitmap,
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
            position: Some(position),
            condition_number: finite(triangulation.condition_number),
        },
    ))
}

/// What one round's two kernels said about one observation.
struct Fit {
    /// Where the sub-pixel stage left the keypoint, in source-image px.
    keypoint: [f64; 2],
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
) {
    if round.len() < 2 {
        return;
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

    let localized = {
        let mut phase = progress.phase("localize");
        let localized =
            localize_patch_keypoints(frame, images, &view_set, Some(&seeds), &options.localize);
        progress_note!(phase, "{} views", localized.views.len());
        localized
    };
    let refined = {
        let mut phase = progress.phase("refine");
        let refined = refine_patch_keypoints(
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
        fits.insert(i, Fit { keypoint });
    }
}

/// Triangulate the observations at `which` from the keypoints they carry.
///
/// Refused on the three signals every other caller refuses on: a non-finite
/// position, an infinite condition number (the depth is not observable), or a
/// solution behind one of the cameras that see it.
fn triangulate_keypoints(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    which: &[usize],
) -> Result<Triangulation, FitError> {
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
    triangulate_rays(&rays, images)
}

/// Triangulate the `in` observations from wherever they currently sit: the
/// cluster stage's refined positions, or their seeds.
pub(super) fn triangulate_in_seeds(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
) -> Result<Triangulation, FitError> {
    let mut rays: Vec<([f64; 2], usize)> = Vec::new();
    for &i in &track.in_observations() {
        let observation = &track.observations[i];
        if (observation.image as usize) < images.len() {
            if let Some(seed) = seed_of(observation) {
                rays.push((seed, observation.image as usize));
            }
        }
    }
    triangulate_rays(&rays, images)
}

/// The batch triangulator over `(pixel, image)` pairs, with the crate's
/// camera-to-world convention.
fn triangulate_rays(
    rays: &[([f64; 2], usize)],
    images: &[ProjectedImage<'_>],
) -> Result<Triangulation, FitError> {
    if rays.len() < 2 {
        return Err(FitError::TooFewObservations(rays.len()));
    }
    let mut dirs: Vec<Vector3<f64>> = Vec::with_capacity(rays.len());
    let mut centers: Vec<Point3<f64>> = Vec::with_capacity(rays.len());
    for &(pixel, image) in rays {
        let view = &images[image];
        let ray = view.camera.pixel_to_ray(pixel[0], pixel[1]);
        // Camera-to-world carries the canonical (-Z forward) ray into the world
        // frame the triangulator solves in.
        let rotation = view.cam_from_world.to_rotation_matrix();
        dirs.push(rotation.transpose() * Vector3::new(ray[0], ray[1], ray[2]));
        centers.push(view.cam_from_world.inverse_translation_origin());
    }
    let offsets = [0usize, dirs.len()];
    let triangulation = triangulate_batch(&dirs, &centers, &offsets)[0];
    if !triangulation.point.coords.iter().all(|c| c.is_finite())
        || !triangulation.condition_number.is_finite()
        || !triangulation.in_front_of_all_cameras
    {
        return Err(FitError::Triangulation);
    }
    Ok(triangulation)
}

/// Fuse the `in` observations into one consensus tile at their final keypoints,
/// and read the point's colour off its centre.
///
/// The fuse is the sub-pixel kernel's own (`render_bitmaps`), run with no
/// Gauss-Newton step so it moves nothing: the keypoints are the ones the fit
/// already settled, and this pass only renders and blends them. The grid is the
/// reconstruction's own bitmap grid where it stores one, so what is fused is a
/// tile the column can hold.
fn fuse_bitmap(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    patch: &OrientedPatch,
    ins: &[usize],
    options: &FitOptions,
) -> (Option<Array3<u8>>, Option<[u8; 3]>) {
    let stored = edited.base.point_set.patch_bitmaps_y_x_rgba.as_ref();
    let (resolution, channels) = match stored {
        Some(bitmaps) => (bitmaps.shape()[1], bitmaps.shape()[3]),
        None => (options.refine.resolution.max(2) as usize, 4),
    };
    let mut view_set: Vec<u32> = Vec::with_capacity(ins.len());
    let mut seeds: Vec<Option<[f64; 2]>> = Vec::with_capacity(ins.len());
    for &i in ins {
        let observation = &track.observations[i];
        let Some(keypoint) = observation.track.as_ref().and_then(|m| m.keypoint) else {
            continue;
        };
        view_set.push(observation.image);
        seeds.push(Some([f64::from(keypoint[0]), f64::from(keypoint[1])]));
    }
    if view_set.len() < 2 {
        return (None, None);
    }
    let params = KeypointSubpixelParams {
        resolution: resolution as u32,
        // Nothing moves: the keypoints are settled, and this pass is the fuse.
        max_gn_steps: 0,
        max_outer_sweeps: 1,
        render_bitmaps: true,
        ..options.refine.clone()
    };
    let Some(fused) =
        refine_patch_keypoints(patch, images, &view_set, Some(&seeds), &params).representative
    else {
        return (None, None);
    };
    // The kernel fuses RGBA; the column takes as many channels as it carries.
    let mut bitmap = Array3::<u8>::zeros((resolution, resolution, channels));
    for row in 0..resolution {
        for col in 0..resolution {
            for c in 0..channels {
                bitmap[[row, col, c]] = if c < 4 {
                    fused[(row * resolution + col) * 4 + c]
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
    (Some(bitmap), Some(color))
}
