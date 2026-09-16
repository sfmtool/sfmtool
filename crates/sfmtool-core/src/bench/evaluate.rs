// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The evaluation: the one step that reads photographs, at whichever stage the
//! track is in.
//!
//! `specs/core/bench/editable-track.md` is the design. [`evaluate`] is a pure
//! function from a track, the reconstruction it belongs to and one decoded view
//! per image to the same track with its measurement slots filled. It sets no
//! verdict: the thresholds propose and
//! [`apply_thresholds`](super::steps::apply_thresholds) is what applies the
//! proposal, so what this writes is a report about every observation and a
//! decision about none.
//!
//! At the **cluster stage** it is the `.matches` refinement kernel over an
//! in-memory cluster; at the **track stage** it is the two kernels
//! [`add_observation`](mod@crate::reconstruction::add_observation) chains, run over
//! the whole track at once. Both are the batch pipeline's own kernels at their
//! own defaults, so a number the bench shows is a number the pipeline would
//! have produced.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};
use ndarray::{Array2, Array3};

use crate::camera::remap::ImageU8Pyramid;
use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{
    refine_cluster_patches_borrowed, sample_member_grid, ClusterRefineParams, FeatureGeometry,
    MemberStatus, REFERENCE_UNREFINABLE,
};
use crate::patch::keypoint_localize::{
    localize_patch_keypoints, project_unclipped, KeypointLocalizeParams,
};
use crate::patch::keypoint_subpixel::{refine_patch_keypoints, KeypointSubpixelParams};
use crate::patch::localizability::{score_localizability_stack, SIGMA_NOISE};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::progress_note;
use crate::reconstruction::add_observation::placement_scale;
use crate::reconstruction::create_point::render_bitmap;
use crate::reconstruction::edited::EditedReconstruction;
use crate::reconstruction::triangulation::{triangulate_batch, Triangulation};

use super::track::{
    ClusterPayload, ClusterTemplate, EditableTrack, Observation, Stage, StageKind, TrackPayload,
    Verdict,
};

/// What the kernels an evaluation runs are allowed to do.
///
/// Every field defaults to the kernel's own default, which is what the batch
/// pass runs with, so the bench measures what the pipeline would have measured.
/// The track's [`Thresholds`](super::track::Thresholds) are **not** here on
/// purpose: a bar the person moved paints verdicts, and it must not change what
/// the numbers under the painting are.
#[derive(Debug, Clone, Default)]
pub struct EvaluateOptions {
    /// The cluster stage's refinement kernel.
    ///
    /// Its
    /// [`radius`](crate::patch::cluster_refine::ClusterRefineParams::radius) is
    /// the one field an evaluation does **not** take from here: the template's
    /// half-width is
    /// [`ClusterPayload::radius`](super::track::ClusterPayload::radius),
    /// because it is what every one of the track's shapes was written against
    /// and a round run at another radius would be measuring a different square
    /// from the one the person put on the bench.
    pub cluster: ClusterRefineParams,
    /// The track stage's discrete localization kernel.
    pub localize: KeypointLocalizeParams,
    /// The track stage's sub-pixel kernel, chained after the discrete one.
    pub refine: KeypointSubpixelParams,
}

/// Why an evaluation was refused. Every variant names what did not hold,
/// because the caller is a button that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluateError {
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

impl std::fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluateError::ViewsMissing { got, expected } => write!(
                f,
                "{got} decoded views were supplied and the reconstruction has {expected} images"
            ),
            EvaluateError::NoView { image } => write!(
                f,
                "image {image} has no decoded view with a pose in this evaluation"
            ),
            EvaluateError::TooFewObservations(n) => write!(
                f,
                "{n} observations are in, and the track stage needs two or more"
            ),
            EvaluateError::NoFrame => write!(
                f,
                "the track carries no patch frame to register against; upgrade it \
                 from the cluster stage to build one"
            ),
            EvaluateError::Triangulation => write!(
                f,
                "the in observations do not triangulate: the depth is not observable, \
                 or the point falls behind a camera that sees it"
            ),
        }
    }
}

impl std::error::Error for EvaluateError {}

/// What one evaluation measured.
///
/// The counts are of every observation of the track, whatever its verdict: an
/// evaluation measures the `out` ones too, so a refusal is shown beside the
/// number it would have been judged on.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluateReport {
    /// The stage it ran at.
    pub stage: StageKind,
    /// How many observations came back with a measurement.
    pub measured: usize,
    /// How many the kernels did not place: out of frame, refused by a gate, or
    /// carrying nothing to seed from.
    pub unmeasured: usize,
    /// At the cluster stage, the observation the template was cut around. The
    /// kernel picks it -- the largest-scale usable member -- so this is where
    /// it landed, and it is `None` when no member could anchor one.
    pub reference: Option<usize>,
    /// At the track stage, where the `in` observations triangulate.
    pub position: Option<Point3<f64>>,
    /// At the track stage, that triangulation's condition number.
    pub condition_number: Option<f64>,
}

impl std::fmt::Display for EvaluateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = self.measured + self.unmeasured;
        match self.stage {
            StageKind::Cluster => {
                write!(f, "measured {} of {total} observations", self.measured)?;
                match self.reference {
                    Some(reference) => write!(f, " against observation {reference}"),
                    None => write!(f, "; no observation could anchor a template"),
                }
            }
            StageKind::Track => {
                write!(f, "measured {} of {total} observations", self.measured)?;
                match self.position {
                    Some(p) => write!(f, " at ({:.4}, {:.4}, {:.4})", p.x, p.y, p.z),
                    None => write!(f, "; nothing triangulated"),
                }
            }
        }
    }
}

/// Fill the measurement slots of every observation of `track`, whatever its
/// verdict, at the stage it is in, and leave every verdict where it is.
///
/// `images` is one [`ProjectedImage`] per image of `edited`, indexed by image
/// index -- the decoded pixels the kernels need, which a reconstruction value
/// does not carry, exactly as
/// [`add_observation`](crate::reconstruction::add_observation::add_observation)
/// takes them.
///
/// **At the cluster stage** every observation's seed is a member of an
/// in-memory `.matches` cluster, and
/// [`refine_cluster_patches`](crate::patch::cluster_refine::refine_cluster_patches)
/// is run over it: the kernel picks the reference (its largest-scale usable
/// member), cuts the template there, and warps every other seed onto it. What
/// lands in each observation's slot is the refined position and shape, the
/// achieved ZNCC, the drift from the seed, the observation's own tile
/// localizability and the kernel's `member_status`. No pose is read. The
/// template's half-width is the cluster's own
/// [`ClusterPayload::radius`](super::track::ClusterPayload::radius) rather than
/// the one in `options`, so the square the round registers is the square the
/// seeds were written against.
///
/// **At the track stage** the track's surfel is registered into every view by
/// the two kernels the embed pass and `add_observation` chain, an `out`
/// observation being scored the way a candidate is, against the `in` set and
/// never as part of it; the `in` results are re-triangulated, and the consensus
/// bitmap is fused over them. What lands in each slot is the keypoint, the leave-one-out
/// ZNCC, the drift from the surfel's projection, the reprojection error against
/// the triangulated position, the ray angle and the tile localizability.
///
/// `progress` is where the call names its phases, the names the batch kernels
/// carry: `refine` and `localizability` at the cluster stage, `localize`,
/// `refine`, `localizability` and `fuse` at the track stage. Pass
/// `&Progress::none()` to report nothing.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{evaluate, EvaluateOptions};
/// # use sfmtool_core::progress::Progress;
/// # use sfmtool_core::EditedReconstruction;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (measured, report) = evaluate(
///     track,
///     edited,
///     images,
///     &EvaluateOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}");
/// # let _ = measured;
/// # Ok(())
/// # }
/// ```
pub fn evaluate(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError> {
    check_views(edited, images)?;
    evaluate_preconditions(track)?;
    match &track.stage {
        Stage::Cluster(payload) => evaluate_cluster(track, payload, images, options, progress),
        Stage::Track(payload) => {
            let frame = payload.frame.clone().ok_or(EvaluateError::NoFrame)?;
            let frame = finite_frame(track, &frame, images)?;
            evaluate_track(track, edited, images, &frame, options, progress)
        }
    }
}

/// Whether `track` can be evaluated at the stage it stands in, judged on the
/// track alone.
///
/// The half of [`evaluate`]'s validation that reads no photograph: whether the
/// track stage has a surfel to register against, and whether enough
/// observations are `in` for the consensus it registers against to exist. A
/// caller that runs the evaluation somewhere expensive -- on a worker, after
/// decoding a dozen images -- asks this first and refuses in front of the
/// decode, which is a refusal the person who asked for it sees immediately
/// rather than a task that fails a second later.
///
/// [`evaluate`] calls it before anything else it does with the track, so the
/// two cannot come to disagree about what is refused; what is left to
/// [`evaluate`] is everything that needs the views, which is the rest.
///
/// The cluster stage has no such condition: it registers each seed against a
/// template cut from the photographs, so what it can do is a question about
/// the pixels and not about the track.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{evaluate_preconditions, EditableTrack};
/// # fn run(track: &EditableTrack) -> Result<(), Box<dyn std::error::Error>> {
/// evaluate_preconditions(track)?;   // refuse here, before a photograph is read
/// # Ok(())
/// # }
/// ```
pub fn evaluate_preconditions(track: &EditableTrack) -> Result<(), EvaluateError> {
    let Stage::Track(payload) = &track.stage else {
        return Ok(());
    };
    if payload.frame.is_none() {
        return Err(EvaluateError::NoFrame);
    }
    let ins = track.in_observations().len();
    if ins < 2 {
        return Err(EvaluateError::TooFewObservations(ins));
    }
    Ok(())
}

/// Every image of `edited` has a decoded view, and there are at least as many
/// views as images: the kernels index `images` by image index.
fn check_views(
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
) -> Result<(), EvaluateError> {
    let expected = edited.image_count();
    if images.len() < expected {
        return Err(EvaluateError::ViewsMissing {
            got: images.len(),
            expected,
        });
    }
    Ok(())
}

/// The observations an evaluation runs over: every one, in index order. An
/// `out` observation is measured like a candidate, so the person sees the
/// number the refusal stands beside; only the `in` set decides anything.
fn evaluated(track: &EditableTrack) -> Vec<usize> {
    (0..track.observations.len()).collect()
}

/// Where an observation currently is, in its image's pixels: the keypoint a
/// track-stage measurement carries, else the cluster stage's refined position
/// or its seed, else nothing.
fn seed_of(observation: &Observation) -> Option<[f64; 2]> {
    if let Some(keypoint) = observation.track.as_ref().and_then(|m| m.keypoint) {
        return Some([f64::from(keypoint[0]), f64::from(keypoint[1])]);
    }
    observation.cluster.as_ref().map(|m| m.best_position())
}

// ---- The cluster stage -----------------------------------------------------

/// One seed as the kernel's per-image feature tables hold it: a position and
/// the affine shape at it, in that image's pixels.
type SeedRow = ([f64; 2], [[f64; 2]; 2]);

/// Refine the in-memory cluster every observation's seed makes, and write the
/// kernel's answer into their cluster slots.
fn evaluate_cluster(
    track: &EditableTrack,
    payload: &ClusterPayload,
    images: &[ProjectedImage<'_>],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError> {
    // The cluster's own radius, not the options': the seeds were written
    // against it, so it is what says which square is being registered.
    let params = ClusterRefineParams {
        radius: payload.radius,
        ..options.cluster.clone()
    };
    let mut next = track.clone();
    let mut members: Vec<usize> = Vec::new();
    // The per-image feature tables the kernel reads its seeds out of. The bench
    // holds no `.sift`, so the cluster's own seeds are the detections as far as
    // the kernel is concerned, and a member's "feature index" is its place in
    // its image's list.
    let mut per_image: Vec<Vec<SeedRow>> = vec![Vec::new(); images.len()];
    let mut member_images: Vec<u32> = Vec::new();
    let mut member_features: Vec<u32> = Vec::new();
    for &i in &evaluated(track) {
        let observation = &track.observations[i];
        let image = observation.image as usize;
        if image >= images.len() {
            return Err(EvaluateError::NoView {
                image: observation.image,
            });
        }
        let Some(measurement) = &observation.cluster else {
            // Nothing said where this observation sits in its image, so there
            // is no seed to warp a template onto.
            continue;
        };
        member_images.push(observation.image);
        member_features.push(per_image[image].len() as u32);
        per_image[image].push((measurement.seed_position, measurement.seed_shape));
        members.push(i);
    }

    let positions: Vec<Array2<f32>> = per_image
        .iter()
        .map(|rows| Array2::from_shape_fn((rows.len(), 2), |(k, c)| rows[k].0[c] as f32))
        .collect();
    let shapes: Vec<Array3<f32>> = per_image
        .iter()
        .map(|rows| Array3::from_shape_fn((rows.len(), 2, 2), |(k, r, c)| rows[k].1[r][c] as f32))
        .collect();
    let features: Vec<FeatureGeometry<'_>> = positions
        .iter()
        .zip(&shapes)
        .map(|(p, s)| FeatureGeometry {
            positions_xy: p.view(),
            affine_shapes: s.view(),
        })
        .collect();
    let pyramids: Vec<&ImageU8Pyramid> = images.iter().map(|v| v.pyramid).collect();

    let result = {
        let mut phase = progress.phase("refine");
        let result = refine_cluster_patches_borrowed(
            &pyramids,
            &features,
            &[0, members.len() as u32],
            &member_images,
            &member_features,
            &params,
            None,
        );
        progress_note!(phase, "{} observations", members.len());
        result
    };

    let mut measured = 0;
    for (k, &i) in members.iter().enumerate() {
        let status = result.member_status[k];
        let fitted = matches!(
            status,
            MemberStatus::Reference
                | MemberStatus::Kept
                | MemberStatus::RejectedLowZncc
                | MemberStatus::RejectedShift
        );
        let measurement = next.observations[i]
            .cluster
            .as_mut()
            .expect("every member carries the seed it was built from");
        measurement.position = fitted.then(|| {
            [
                result.member_positions[[k, 0]],
                result.member_positions[[k, 1]],
            ]
        });
        measurement.shape = fitted.then(|| {
            [
                [
                    result.member_affine_shapes[[k, 0, 0]],
                    result.member_affine_shapes[[k, 0, 1]],
                ],
                [
                    result.member_affine_shapes[[k, 1, 0]],
                    result.member_affine_shapes[[k, 1, 1]],
                ],
            ]
        });
        measurement.zncc = finite(f64::from(result.member_zncc[k]));
        measurement.shift_px = finite(f64::from(result.member_shift_px[k]));
        measurement.status = Some(status);
        if fitted {
            measured += 1;
        }
    }

    // The tile localizability, at each member's **seed** geometry: that is
    // where the kernel's own gate scores it, so the column and the
    // `RejectedUnlocalizable` status are the same measurement.
    {
        let mut phase = progress.phase("localizability");
        for &i in &members {
            let observation = &track.observations[i];
            let measurement = observation
                .cluster
                .as_ref()
                .expect("every member carries the seed it was built from");
            let sigma = tile_localizability(
                images[observation.image as usize].pyramid,
                measurement.seed_position,
                measurement.seed_shape,
                &params,
            );
            next.observations[i]
                .cluster
                .as_mut()
                .expect("the same member")
                .localizability = sigma;
        }
        progress_note!(phase, "{} observations", members.len());
    }

    // Where the kernel cut the template, and the tile it cut.
    let reference = (result.reference_members[0] != REFERENCE_UNREFINABLE)
        .then(|| members[result.reference_members[0] as usize]);
    let template = reference.and_then(|i| {
        let observation = &track.observations[i];
        let measurement = observation.cluster.as_ref()?;
        let grid = sample_member_grid(
            images[observation.image as usize].pyramid,
            measurement.seed_position,
            measurement.seed_shape,
            &params,
        )?;
        let resolution = params.resolution.max(2) as usize;
        let channels = grid.len() / (resolution * resolution);
        Some(ClusterTemplate {
            samples: Array3::from_shape_vec((resolution, resolution, channels), grid)
                .expect("the sampler fills every grid cell of every channel"),
        })
    });
    next.stage = Stage::Cluster(ClusterPayload {
        reference: reference.unwrap_or(payload.reference),
        radius: payload.radius,
        template,
    });

    Ok((
        next,
        EvaluateReport {
            stage: StageKind::Cluster,
            measured,
            unmeasured: evaluated(track).len() - measured,
            reference,
            position: None,
            condition_number: None,
        },
    ))
}

/// One observation's own tile localizability, `sigma_pos` in template-grid px,
/// or `None` when the geometry is degenerate or the tile leaves the pyramid.
fn tile_localizability(
    pyramid: &ImageU8Pyramid,
    position: [f64; 2],
    shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> Option<f64> {
    let grid = sample_member_grid(pyramid, position, shape, params)?;
    let resolution = params.resolution.max(2) as usize;
    let channels = grid.len() / (resolution * resolution);
    let scored =
        score_localizability_stack(&grid, 1, resolution, channels, params.window, SIGMA_NOISE);
    finite(scored[0].sigma_pos_grid)
}

// ---- The track stage -------------------------------------------------------

/// The surfel a track-stage evaluation registers against, made finite.
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
) -> Result<OrientedPatch, EvaluateError> {
    if frame.w != 0.0 {
        return Ok(frame.clone());
    }
    let provisional = triangulate_in_seeds(track, images)?.point;
    let scale = placement_scale(&provisional, images);
    if !(scale.is_finite() && scale > 0.0) {
        return Err(EvaluateError::Triangulation);
    }
    Ok(OrientedPatch::new(
        provisional,
        frame.u_axis,
        frame.v_axis,
        [frame.half_extent[0] * scale, frame.half_extent[1] * scale],
    ))
}

/// Register `frame` into every view, `out` ones scored like candidates,
/// re-triangulate the `in` results, fuse the consensus bitmap, and write the
/// whole of it into the track's track-stage slots.
///
/// Shared by [`evaluate`] at the track stage and by the upgrade
/// ([`set_stage`](super::stage::set_stage)), which differ only in where the
/// frame came from: the payload's own, or one framed from the cluster's
/// reference at the triangulated depth.
pub(super) fn evaluate_track(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError> {
    let run = evaluated(track);
    for &i in &run {
        let image = track.observations[i].image;
        if image as usize >= images.len() {
            return Err(EvaluateError::NoView { image });
        }
    }
    // The `in` count is [`evaluate_preconditions`]'s, checked before the
    // caller spent anything on the views.
    let ins = track.in_observations();

    // ── The rounds ──
    //
    // One localization places every observation it holds, and it holds one
    // observation per image: the kernel registers a *point's* sighting in a
    // view, and two sightings in one view are two hypotheses about that view.
    // So the first round is the `in` observations plus every candidate in an
    // image none of them holds, which is the shape `add_observation` runs; a
    // candidate in an image that is already spoken for gets a round of its own,
    // against the `in` observations minus the one whose image it wants, which
    // is the same leave-one-out question asked about the other hypothesis.
    //
    // An observation whose seed is off its image's sensor is in no round at
    // all: it names a place that image does not have, so nothing can be
    // registered there and the row stays unmeasured. This is the refusal
    // `add_observation` makes up front about a clicked pixel, made here about
    // one observation instead of about the whole call.
    let placeable = |i: usize| -> bool {
        let observation = &track.observations[i];
        let view = &images[observation.image as usize];
        seed_of(observation).is_some_and(|seed| {
            seed[0] >= 0.0
                && seed[1] >= 0.0
                && seed[0] < f64::from(view.camera.width)
                && seed[1] < f64::from(view.camera.height)
        })
    };
    let mut first: Vec<usize> = ins.iter().copied().filter(|&i| placeable(i)).collect();
    let mut taken: Vec<u32> = first.iter().map(|&i| track.observations[i].image).collect();
    let mut contested: Vec<usize> = Vec::new();
    for &i in &run {
        if track.observations[i].verdict == Verdict::In || !placeable(i) {
            continue;
        }
        let image = track.observations[i].image;
        if taken.contains(&image) {
            contested.push(i);
        } else {
            taken.push(image);
            first.push(i);
        }
    }

    let mut fits: HashMap<usize, Fit> = HashMap::new();
    fit_round(track, images, frame, &first, options, progress, &mut fits);
    for &i in &contested {
        let image = track.observations[i].image;
        let mut round: Vec<usize> = ins
            .iter()
            .copied()
            .filter(|&j| track.observations[j].image != image && placeable(j))
            .collect();
        round.push(i);
        fit_round(track, images, frame, &round, options, progress, &mut fits);
    }

    // ── The measurements, and the re-triangulation they feed ──
    let mut next = track.clone();
    for &i in &run {
        let mut measurement = next.observations[i].track.clone().unwrap_or_default();
        match fits.get(&i) {
            Some(fit) => {
                measurement.keypoint = Some([fit.keypoint[0] as f32, fit.keypoint[1] as f32]);
                measurement.zncc = finite(fit.zncc);
                measurement.shift_px = finite(fit.shift_px);
            }
            None => {
                // The fit did not place it: out of frame, or refused by one of
                // the localizer's own gates. Its keypoint is then wherever it
                // already sat -- the one it arrived with, or the cluster
                // stage's refined position for a track that has just been
                // upgraded -- so the column still says where this sighting is
                // and the re-triangulation still has its ray. The scores are
                // this round's, and it has none: an unscored row is what says
                // the fit did not place it.
                measurement.keypoint = measurement.keypoint.or_else(|| {
                    seed_of(&track.observations[i]).map(|p| [p[0] as f32, p[1] as f32])
                });
                measurement.zncc = None;
                measurement.shift_px = None;
            }
        }
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

    // ── Per observation: where the position puts it, and how well its own
    //    tile pins a keypoint down at all ──
    let measured = fits.len();
    {
        let mut phase = progress.phase("localizability");
        for &i in &run {
            let image = next.observations[i].image as usize;
            let view = &images[image];
            let Some(measurement) = next.observations[i].track.as_mut() else {
                continue;
            };
            let Some(keypoint) = measurement.keypoint else {
                continue;
            };
            let keypoint = [f64::from(keypoint[0]), f64::from(keypoint[1])];
            let (error, angle) = observation_metrics(view, &position, keypoint);
            measurement.reprojection_error = finite(error);
            measurement.ray_angle_deg = finite(angle);
            measurement.localizability = surfel_tile_localizability(
                &placed,
                view,
                keypoint,
                options.refine.resolution.max(2) as usize,
                options.refine.window,
            );
        }
        progress_note!(phase, "{} observations", run.len());
    }

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

    Ok((
        next,
        EvaluateReport {
            stage: StageKind::Track,
            measured,
            unmeasured: run.len() - measured,
            reference: None,
            position: Some(position),
            condition_number: finite(triangulation.condition_number),
        },
    ))
}

/// What one round's two kernels said about one observation.
struct Fit {
    /// Where the sub-pixel stage left the keypoint, in source-image px.
    keypoint: [f64; 2],
    /// The discrete stage's leave-one-out ZNCC against the other views'
    /// consensus, which is the score `add_observation` accepts an observation
    /// on and the one a commit stores.
    zncc: f64,
    /// How far the keypoint sits from the surfel's projection, in px.
    shift_px: f64,
}

/// Localize and refine one round's observations against `frame`, and record
/// what each got.
///
/// The round holds one observation per image, so an image index names exactly
/// one of its observations and the kernels' per-view answers scatter back
/// without ambiguity.
fn fit_round(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    round: &[usize],
    options: &EvaluateOptions,
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
        let shift_px = at.map_or(localized.offsets_px[slot], |k| refined.offsets_px[k]);
        fits.insert(
            i,
            Fit {
                keypoint,
                zncc: localized.loo_zncc[slot],
                shift_px,
            },
        );
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
) -> Result<Triangulation, EvaluateError> {
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
) -> Result<Triangulation, EvaluateError> {
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
) -> Result<Triangulation, EvaluateError> {
    if rays.len() < 2 {
        return Err(EvaluateError::TooFewObservations(rays.len()));
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
        return Err(EvaluateError::Triangulation);
    }
    Ok(triangulation)
}

/// One observation's reprojection error in px and the angle its ray makes with
/// the direction to `position`, in degrees -- the two numbers the Point Track
/// Detail panel tabulates for a committed track, over the position and the
/// keypoint this evaluation found rather than the stored ones.
fn observation_metrics(
    view: &ProjectedImage<'_>,
    position: &Point3<f64>,
    keypoint: [f64; 2],
) -> (f64, f64) {
    let Some((u, v)) = project_unclipped(view, position, 1.0) else {
        return (f64::NAN, f64::NAN);
    };
    let error = (u - keypoint[0]).hypot(v - keypoint[1]);
    let ray = view.camera.pixel_to_ray(keypoint[0], keypoint[1]);
    let ray = Vector3::new(ray[0], ray[1], ray[2]);
    let towards = view.cam_from_world.transform_point(position).coords;
    let (ray_norm, towards_norm) = (ray.norm(), towards.norm());
    if !(ray_norm > 0.0 && towards_norm > 0.0) {
        return (error, f64::NAN);
    }
    let cos = (ray.dot(&towards) / (ray_norm * towards_norm)).clamp(-1.0, 1.0);
    (error, cos.acos().to_degrees())
}

/// The localizability of what one view shows of the surfel at its keypoint:
/// the tile rendered through the keypoint-anchored frame, scored by the same
/// kernel the consensus is scored by.
fn surfel_tile_localizability(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: [f64; 2],
    resolution: usize,
    window: crate::patch::normal_refine::PatchWindow,
) -> Option<f64> {
    let anchored = patch.anchored_at_keypoint(view.camera, view.cam_from_world, keypoint);
    let frame = anchored.as_ref().unwrap_or(patch);
    let channels = view.pyramid.level(0).channels() as usize;
    let tile = render_bitmap(frame, view, resolution, channels);
    let samples: Vec<f32> = tile.iter().map(|&v| f32::from(v)).collect();
    let scored = score_localizability_stack(&samples, 1, resolution, channels, window, SIGMA_NOISE);
    finite(scored[0].sigma_pos_grid)
}

/// Fuse the `in` observations into one consensus tile at their final keypoints,
/// and read the point's colour off its centre.
///
/// The fuse is the sub-pixel kernel's own (`render_bitmaps`), run with no
/// Gauss-Newton step so it moves nothing: the keypoints are the ones the
/// evaluation already settled, and this pass only renders and blends them. The
/// grid is the reconstruction's own bitmap grid where it stores one, so what is
/// fused is a tile the column can hold.
fn fuse_bitmap(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    patch: &OrientedPatch,
    ins: &[usize],
    options: &EvaluateOptions,
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

/// `Some(value)` when it is a number, `None` when the kernel reported nothing.
///
/// A `NaN` out of a kernel means "this was not scored", and the slots say that
/// with `None`; a `NaN` left in a slot would read as a measured failure to the
/// painting, which is a different thing entirely.
fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}
