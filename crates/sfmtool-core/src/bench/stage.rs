// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Moving a track between its two representations.
//!
//! `specs/core/bench/editable-track.md` is the design. [`set_stage`] is the one
//! operation, in both directions: **up** from a set of image patches to a
//! surfel at a position, which triangulates and frames and then evaluates; and
//! **down** from the surfel back to the patches, which projects the frame
//! through each observation's camera and throws the 3D away on purpose.
//!
//! Setting the stage a track is already at changes nothing and says so, so a
//! caller can wire a toggle straight to this and push no version for a step
//! that did not happen.

use nalgebra::{Point3, Vector3};

use crate::patch::cloud::{mean_viewing_normal, OrientedPatch};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::reconstruction::data::{patch_affine_shape, Point3D};
use crate::reconstruction::edited::EditedReconstruction;

use super::evaluate::{evaluate_track, EvaluateError, EvaluateOptions, EvaluateReport};
use super::track::{
    ClusterMeasurement, ClusterPayload, EditableTrack, Stage, StageKind, TrackPayload, Verdict,
};

/// Why a stage could not be set. Every variant names what did not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageError {
    /// The upgrade's own work, or the evaluation that follows it, was refused.
    Evaluate(EvaluateError),
    /// A downgrade needs the frame it projects into each observation's camera,
    /// and the track carries none.
    NoFrame,
    /// A downgrade needs the position the frame stands at, and the track
    /// carries none.
    NoPosition,
    /// No `in` observation could anchor the cluster: none carries a seed, or
    /// none has a usable affine shape to cut a template at.
    NoReference,
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageError::Evaluate(e) => write!(f, "{e}"),
            StageError::NoFrame => write!(
                f,
                "the track carries no patch frame, so there is nothing to project \
                 into each observation's image"
            ),
            StageError::NoPosition => write!(
                f,
                "the track carries no position, so its frame stands nowhere"
            ),
            StageError::NoReference => write!(
                f,
                "no in observation carries a usable seed for the cluster to be \
                 cut around"
            ),
        }
    }
}

impl std::error::Error for StageError {}

impl From<EvaluateError> for StageError {
    fn from(e: EvaluateError) -> Self {
        StageError::Evaluate(e)
    }
}

/// What one stage change did.
#[derive(Debug, Clone, PartialEq)]
pub struct StageReport {
    /// The stage the track was in.
    pub from: StageKind,
    /// The stage it is in now.
    pub to: StageKind,
    /// Whether anything happened. Setting the stage a track is already at is
    /// reported rather than refused, and the caller pushes no version for it.
    pub changed: bool,
    /// The evaluation the upgrade ran, which is the track stage's own
    /// measurement pass.
    pub evaluate: Option<EvaluateReport>,
    /// At a downgrade, the observation the cluster is now cut around.
    pub reference: Option<usize>,
}

impl StageReport {
    /// What the change did beyond moving the stage: the clause that follows the
    /// stage phrase, its own separator included, and empty where there is
    /// nothing more to say.
    ///
    /// Split out of [`Display`](std::fmt::Display) so that a caller writing the
    /// stage phrase in its own words -- the viewer's Action Log row names the
    /// item, which core cannot -- adds this to its sentence rather than
    /// appending a report that states the stage a second time.
    pub fn detail(&self) -> String {
        if !self.changed {
            return String::new();
        }
        match (&self.evaluate, self.reference) {
            (Some(report), _) => format!(": {report}"),
            (None, Some(reference)) => format!(", cut around observation {reference}"),
            (None, None) => String::new(),
        }
    }
}

impl std::fmt::Display for StageReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.changed {
            return write!(f, "already at the {} stage", self.to);
        }
        write!(f, "set to the {} stage{}", self.to, self.detail())
    }
}

/// Put `track` into `stage`, in whichever direction that is.
///
/// **Up, cluster to track.** The `in` observations' refined cluster positions
/// are triangulated; the patch is framed at that position from the reference
/// observation's affine shape, projected back through its camera at the
/// triangulated depth, with the mean viewing direction as the normal; and the
/// track-stage evaluation then localizes and refines every `in` and
/// `candidate` observation against that surfel, re-triangulates the `in`
/// results and fuses the consensus. The cluster-stage measurements are
/// dropped: they describe a registration against a reference and a template
/// the track no longer has.
///
/// **Down, track to cluster.** Always possible, and lossy on purpose: the
/// reference becomes the `in` observation with the largest projected patch
/// scale, each observation is re-seeded at its keypoint with the affine shape
/// the format derives by projecting the frame at that observation's anchor, and
/// the position, the frame, the bitmap and the track-stage measurements are
/// dropped. This is the step for a track whose
/// observations were right and whose 3D hypothesis was the problem: the cluster
/// kernel then judges the observations on appearance alone.
///
/// Both directions go through the cluster's
/// [`radius`](super::track::ClusterPayload::radius): the format's rule states a
/// patch's half-axes in pixels while a cluster shape is per keypoint-frame unit
/// over `[-radius, radius]`, so a downgrade divides the projected columns by it
/// and an upgrade multiplies the reference's by it. A track taken down and put
/// back up is therefore the size it was.
///
/// `images` is one [`ProjectedImage`] per image of `edited`, as every
/// photometric step takes them; a downgrade reads none of them, because
/// projecting a frame needs poses and lenses rather than pixels.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{set_stage, EvaluateOptions, StageKind};
/// # use sfmtool_core::progress::Progress;
/// # use sfmtool_core::EditedReconstruction;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (upgraded, report) = set_stage(
///     track,
///     edited,
///     images,
///     StageKind::Track,
///     &EvaluateOptions::default(),
///     &Progress::none(),
/// )?;
/// assert_eq!(upgraded.stage_kind(), StageKind::Track);
/// println!("{report}");
/// # Ok(())
/// # }
/// ```
pub fn set_stage(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    stage: StageKind,
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, StageReport), StageError> {
    let from = track.stage_kind();
    if from == stage {
        return Ok((
            track.clone(),
            StageReport {
                from,
                to: stage,
                changed: false,
                evaluate: None,
                reference: None,
            },
        ));
    }
    match stage {
        StageKind::Track => {
            let (next, report) = upgrade(track, edited, images, options, progress)?;
            Ok((
                next,
                StageReport {
                    from,
                    to: stage,
                    changed: true,
                    evaluate: Some(report),
                    reference: None,
                },
            ))
        }
        StageKind::Cluster => {
            let (next, reference) = downgrade(track, edited)?;
            Ok((
                next,
                StageReport {
                    from,
                    to: stage,
                    changed: true,
                    evaluate: None,
                    reference: Some(reference),
                },
            ))
        }
    }
}

/// Cluster to track: triangulate, frame, then run the track stage's own
/// evaluation over the result.
fn upgrade(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), StageError> {
    let expected = edited.image_count();
    if images.len() < expected {
        return Err(EvaluateError::ViewsMissing {
            got: images.len(),
            expected,
        }
        .into());
    }
    let payload = track
        .cluster()
        .expect("the caller checked the stage")
        .clone();
    let ins = track.in_observations();
    if ins.len() < 2 {
        return Err(EvaluateError::TooFewObservations(ins.len()).into());
    }
    for &i in &ins {
        let image = track.observations[i].image;
        if image as usize >= images.len() {
            return Err(EvaluateError::NoView { image }.into());
        }
    }

    // 1. Where the refined cluster positions put the point.
    let seeded = EditableTrack {
        stage: Stage::Track(TrackPayload::default()),
        ..track.clone()
    };
    let triangulation = super::evaluate::triangulate_in_seeds(&seeded, images)?;

    // 2. The surfel: the reference observation's own shape, unprojected onto
    //    the plane at the depth it stands, turned to face the views that see it.
    let reference = upgrade_reference(track, &payload, &ins).ok_or(StageError::NoReference)?;
    let observation = &track.observations[reference];
    let measurement = observation
        .cluster
        .as_ref()
        .expect("the reference was picked among the observations that carry a seed");
    let view = &images[observation.image as usize];
    let depth = (triangulation.point - view.cam_from_world.inverse_translation_origin()).norm();
    // A cluster shape maps keypoint-frame units to pixels and the patch is
    // `[-radius, radius]` of them, while the framing rule reads a shape whose
    // columns are the patch's own pixel half-axes. The radius is what carries
    // between the two conventions, and it is the cluster's own.
    let shape = scaled(
        measurement.shape.unwrap_or(measurement.seed_shape),
        payload.radius,
    );
    let framed = OrientedPatch::from_affine_shape_at_depth(
        view.camera,
        view.cam_from_world,
        measurement.best_position(),
        shape,
        depth,
    )
    .ok_or(StageError::NoReference)?;
    let centers: Vec<Point3<f64>> = ins
        .iter()
        .map(|&i| {
            images[track.observations[i].image as usize]
                .cam_from_world
                .inverse_translation_origin()
        })
        .collect();
    let normal = mean_viewing_normal(&triangulation.point, &centers);
    let frame = OrientedPatch::from_center_normal(
        triangulation.point,
        normal,
        framed.v_axis,
        framed.half_extent,
    );

    // 3-4. Localize, refine, re-triangulate and fuse: the track stage's own
    //      evaluation, over seeds that are the cluster's refined positions.
    let (mut next, report) = evaluate_track(&seeded, edited, images, &frame, options, progress)?;
    // A cluster measurement is a registration against a reference and a
    // template the track no longer has, so it goes with the stage.
    for observation in &mut next.observations {
        observation.cluster = None;
    }
    Ok((next, report))
}

/// The observation an upgrade frames the patch from: the cluster's own
/// reference when it is `in`, and otherwise the largest-scale `in` observation,
/// which is what the cluster kernel would have picked among them.
fn upgrade_reference(
    track: &EditableTrack,
    payload: &ClusterPayload,
    ins: &[usize],
) -> Option<usize> {
    let usable = |i: usize| -> Option<(usize, f64)> {
        let measurement = track.observations[i].cluster.as_ref()?;
        let shape = measurement.shape.unwrap_or(measurement.seed_shape);
        let det = shape[0][0] * shape[1][1] - shape[0][1] * shape[1][0];
        (det.is_finite() && det != 0.0).then(|| (i, det.abs().sqrt()))
    };
    if ins.contains(&payload.reference) {
        if let Some((i, _)) = usable(payload.reference) {
            return Some(i);
        }
    }
    ins.iter()
        .filter_map(|&i| usable(i))
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        })
        .map(|(i, _)| i)
}

/// Scale both columns of an affine shape, which is how a patch's pixel
/// half-axes and a keypoint-frame shape convert into each other: the patch is
/// `[-radius, radius]` keypoint-frame units across, so its half-axes are
/// `radius` times the shape's columns.
fn scaled(shape: [[f64; 2]; 2], by: f64) -> [[f64; 2]; 2] {
    [
        [shape[0][0] * by, shape[0][1] * by],
        [shape[1][0] * by, shape[1][1] * by],
    ]
}

/// Track to cluster: re-seed every observation from its keypoint and the shape
/// the frame projects to there, and drop the 3D with the measurements made
/// against it.
fn downgrade(
    track: &EditableTrack,
    edited: &EditedReconstruction,
) -> Result<(EditableTrack, usize), StageError> {
    let payload = track.track().expect("the caller checked the stage");
    let frame = payload.frame.as_ref().ok_or(StageError::NoFrame)?;
    let position = payload.position.ok_or(StageError::NoPosition)?;
    let point = Point3D {
        position,
        w: frame.w,
        color: payload.color,
        error: 0.0,
        normal: Vector3::zeros(),
    };
    let u = frame.u_axis * frame.half_extent[0];
    let v = frame.v_axis * frame.half_extent[1];
    let table = &edited.base.image_table;

    // Each observation's own seed: where it sits, and the footprint the frame
    // has there. The shape is the format's own rule for deriving a keypoint's
    // shape from a patch frame, which is the inverse of the framing the upgrade
    // does, so the two directions state one relationship.
    //
    // That rule's columns are the patch's projected pixel half-axes, and a
    // cluster seed is a keypoint-frame shape read over `[-radius, radius]`, so
    // each column is divided by the radius the new cluster takes. Without that
    // the patch would arrive at the cluster stage `radius` times the size the
    // surfel really has, and the next evaluation would register that square.
    let cluster = ClusterPayload::default();
    let mut next = track.clone();
    let mut seeded: Vec<Option<(usize, f64)>> = Vec::new();
    for (i, observation) in track.observations.iter().enumerate() {
        let image = observation.image as usize;
        let Some(keypoint) = observation
            .track
            .as_ref()
            .and_then(|m| m.keypoint)
            .or_else(|| {
                observation.cluster.as_ref().map(|m| {
                    let p = m.best_position();
                    [p[0] as f32, p[1] as f32]
                })
            })
        else {
            seeded.push(None);
            continue;
        };
        let Some(shape) = patch_affine_shape(&point, u, v, table, image, keypoint) else {
            // The frame does not project into this image: the seed keeps
            // whatever the observation already carried, and the next evaluation
            // reports it as unevaluated.
            seeded.push(None);
            continue;
        };
        let shape = scaled(
            [
                [f64::from(shape[0][0]), f64::from(shape[0][1])],
                [f64::from(shape[1][0]), f64::from(shape[1][1])],
            ],
            1.0 / cluster.radius,
        );
        let det = shape[0][0] * shape[1][1] - shape[0][1] * shape[1][0];
        if !det.is_finite() || det == 0.0 {
            seeded.push(None);
            continue;
        }
        next.observations[i].cluster = Some(ClusterMeasurement::from_seed(
            [f64::from(keypoint[0]), f64::from(keypoint[1])],
            shape,
        ));
        next.observations[i].track = None;
        seeded.push((observation.verdict == Verdict::In).then_some((i, det.abs().sqrt())));
    }

    // The reference is the `in` observation the patch is largest in, which is
    // the most detail any of them shows of it.
    let reference = seeded
        .iter()
        .flatten()
        .copied()
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        })
        .map(|(i, _)| i)
        .ok_or(StageError::NoReference)?;
    next.stage = Stage::Cluster(ClusterPayload {
        reference,
        // A template is a cut around a particular reference in a particular
        // image, and this one has just been picked: the next evaluation cuts it.
        ..cluster
    });
    Ok((next, reference))
}
