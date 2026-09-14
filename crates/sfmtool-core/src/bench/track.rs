// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The [`EditableTrack`] value: the observations being considered for one
//! track, what has been measured about each, what the person has decided about
//! each, and which of the two representations the track is currently in.
//!
//! `specs/core/bench/editable-track.md` is the design. Everything here is a
//! plain value: `Clone`, no interior mutability, no handle to any device, cache
//! or window. The steps that produce a new one live in
//! [`steps`](super::steps) and [`commit`](mod@super::commit).

use nalgebra::Point3;
use ndarray::Array3;

use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{ClusterRefineParams, MemberStatus};
use crate::patch::keypoint_localize::KeypointLocalizeParams;
use crate::patch::view_selection::ViewSelectParams;

/// Where an observation came from.
///
/// Shown to the person, and read by exactly one step: a commit deletes the
/// points that [`Provenance::Point`] observations were pulled from. No kernel
/// reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// The committed track the item was put on the bench from.
    Origin,
    /// A `.sift` feature a descriptor search returned.
    Descriptor {
        /// The feature's index in its image's `.sift` file.
        feature: u32,
    },
    /// An image the view sweep proposed.
    Sweep,
    /// A pixel the person pointed at.
    Pixel,
    /// An observation of another point, pulled in. A commit that keeps it
    /// absorbs that point.
    Point {
        /// The point it was pulled from, by the index it had when it was
        /// pulled.
        point: u32,
    },
}

/// What the person has decided about one observation.
///
/// A measurement is a report and never a decision: the thresholds propose a
/// verdict and a step applies the proposal, but the verdict itself is always
/// the person's, and [`Observation::pinned`] says when one was set by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The observation belongs to the track. The kernels run over these, and a
    /// commit writes exactly these.
    In,
    /// The observation was refused. It stays in the list so a search does not
    /// propose it again and so the refusal is visible.
    Out,
    /// Proposed by something and not yet ruled on.
    Candidate,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::In => write!(f, "in"),
            Verdict::Out => write!(f, "out"),
            Verdict::Candidate => write!(f, "candidate"),
        }
    }
}

/// What the cluster stage has measured about one observation.
///
/// The seed is what the observation was put on the track with and is always
/// present; everything below it is the refinement's answer and is `None` until
/// an evaluation has run.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterMeasurement {
    /// Where the seed put the observation, in that image's pixels.
    pub seed_position: [f64; 2],
    /// The seed's affine shape: the detector's canonical unit frame mapped onto
    /// this image's pixels, the same `S` the `.matches` cluster-patches section
    /// stores.
    pub seed_shape: [[f64; 2]; 2],
    /// Where the refinement put the observation, in that image's pixels.
    pub position: Option<[f64; 2]>,
    /// The refined absolute affine shape, in the same convention as
    /// [`Self::seed_shape`].
    pub shape: Option<[[f64; 2]; 2]>,
    /// The windowed ZNCC the refinement achieved against the template.
    pub zncc: Option<f64>,
    /// How far the refinement moved off the seed, in source-image px.
    pub shift_px: Option<f64>,
    /// The observation's own tile localizability, sigma_pos in template-grid
    /// px.
    pub localizability: Option<f64>,
    /// The refinement's own verdict on the observation, in the `member_status`
    /// legend.
    pub status: Option<MemberStatus>,
}

impl ClusterMeasurement {
    /// A measurement that is a seed and nothing else.
    pub fn from_seed(position: [f64; 2], shape: [[f64; 2]; 2]) -> Self {
        Self {
            seed_position: position,
            seed_shape: shape,
            position: None,
            shape: None,
            zncc: None,
            shift_px: None,
            localizability: None,
            status: None,
        }
    }

    /// Where the observation is: the refined position when there is one, and
    /// the seed otherwise.
    pub fn best_position(&self) -> [f64; 2] {
        self.position.unwrap_or(self.seed_position)
    }
}

/// What the track stage has measured about one observation.
///
/// A track put on the bench from a committed point arrives with
/// [`Self::keypoint`] and [`Self::zncc`] read off the stored columns; the rest
/// is what an evaluation computes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrackMeasurement {
    /// Where the localizer put the observation, in that image's pixels. This is
    /// the pixel a commit writes.
    pub keypoint: Option<[f32; 2]>,
    /// The leave-one-out ZNCC against the consensus of the other `in`
    /// observations.
    pub zncc: Option<f64>,
    /// How far the keypoint sits from the surfel's projection, in px.
    pub shift_px: Option<f64>,
    /// The reprojection error against the triangulated position, in px.
    pub reprojection_error: Option<f64>,
    /// The angle between this observation's own ray and the direction from its
    /// camera to the triangulated position, in degrees: the reprojection
    /// residual stated as an angle, which is what makes it comparable across
    /// lenses and depths. The same number the Point Track Detail panel's
    /// *Angle* column shows for a committed track.
    pub ray_angle_deg: Option<f64>,
    /// The observation's own tile localizability, sigma_pos in grid px.
    pub localizability: Option<f64>,
}

/// One observation of an editable track: an image, a place in it, what has been
/// measured about it at each stage, and the verdict.
///
/// Observations are appended and never renumbered, so an index into
/// [`EditableTrack::observations`] is stable for the life of the track and an
/// evaluation that finishes late still lands on the observation it measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// The image, as an index into the node's image table.
    pub image: u32,
    /// Where it came from.
    pub provenance: Provenance,
    /// The person's decision.
    pub verdict: Verdict,
    /// Whether the verdict was set by hand. A pinned verdict is left alone by
    /// [`apply_thresholds`](super::steps::apply_thresholds).
    pub pinned: bool,
    /// The cluster stage's slot, filled for an observation the track carried
    /// through that stage.
    pub cluster: Option<ClusterMeasurement>,
    /// The track stage's slot, filled for an observation the track carried
    /// through that stage.
    pub track: Option<TrackMeasurement>,
}

impl Observation {
    /// A candidate in `image`, seeded for the cluster stage at `position` with
    /// `shape`, measured at neither stage.
    pub fn seeded(
        image: u32,
        provenance: Provenance,
        position: [f64; 2],
        shape: [[f64; 2]; 2],
    ) -> Self {
        Self {
            image,
            provenance,
            verdict: Verdict::Candidate,
            pinned: false,
            cluster: Some(ClusterMeasurement::from_seed(position, shape)),
            track: None,
        }
    }
}

/// The cluster stage's own data: a `.matches` cluster with its cluster-patches
/// section, in memory.
///
/// There is no pose, no position and no normal here. What makes the
/// observations one thing is that they all register onto one template cut from
/// one of them.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterPayload {
    /// Which observation the template is cut around, as an index into
    /// [`EditableTrack::observations`].
    pub reference: usize,
    /// The cut itself. `None` until an evaluation cuts it, because the cut is a
    /// function of the reference's pixels and the bench holds no photographs.
    pub template: Option<ClusterTemplate>,
}

/// The template the cluster stage registers its observations onto.
///
/// The samples are the reference observation's own tile on the template grid,
/// as the refinement's sampler reads it, which is the tile a panel draws. The
/// correlation the cascade runs z-normalizes it inside the kernel, over the
/// window and without the pixels that window drops, so what is kept here is the
/// picture rather than the kernel's working copy of it.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterTemplate {
    /// The `(resolution, resolution, channels)` samples.
    pub samples: Array3<f32>,
    /// The half-width the cut used, in the reference's keypoint-frame units.
    pub radius: f64,
}

/// The track stage's own data: an `embedded_patches` point that is not in the
/// reconstruction yet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrackPayload {
    /// Where the track's point stands. `None` for a track whose observations
    /// have not been triangulated, which a commit refuses.
    pub position: Option<Point3<f64>>,
    /// The surfel the localizer registers against. Its centre is
    /// [`Self::position`] when both are present, and its `w` says whether the
    /// point is finite or a bearing.
    pub frame: Option<OrientedPatch>,
    /// The `(R, R, C)` consensus bitmap the observations were fused into.
    pub bitmap: Option<Array3<u8>>,
    /// The colour the point carries, used when there is no bitmap to read one
    /// from.
    pub color: [u8; 3],
    /// Confidence in the frame's normal, in the stored column's byte scale.
    pub normal_confidence: Option<u8>,
    /// The last triangulation's condition number.
    pub condition_number: Option<f64>,
}

/// Which of the two representations a track is in, and that representation's
/// own data.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// A set of image patches that register onto one template, with no geometry
    /// behind them.
    Cluster(ClusterPayload),
    /// A surfel at a position, with a keypoint per observation.
    Track(TrackPayload),
}

/// A stage without its data, for a caller that only wants to say which one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// [`Stage::Cluster`].
    Cluster,
    /// [`Stage::Track`].
    Track,
}

impl std::fmt::Display for StageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageKind::Cluster => write!(f, "cluster"),
            StageKind::Track => write!(f, "track"),
        }
    }
}

impl Stage {
    /// Which stage this is, without its data.
    pub fn kind(&self) -> StageKind {
        match self {
            Stage::Cluster(_) => StageKind::Cluster,
            Stage::Track(_) => StageKind::Track,
        }
    }
}

/// The point an editable track was put on the bench from.
///
/// The serial is opaque here: it is whatever the caller numbers its versions
/// with, and core neither mints nor interprets it. What core does with the
/// origin is decide whether a commit replaces a point or creates one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin {
    /// The version the point was read out of, as the caller numbers versions.
    pub version: u64,
    /// The index the point had in that version.
    pub point: u32,
}

/// The bars the threshold painting judges an observation against.
///
/// The defaults are read from the kernels' own parameter types rather than
/// written out again, so the bench and the batch pass start from the same bar
/// and moving one is the person choosing to differ.
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    /// The ZNCC an observation has to reach: the achieved template ZNCC at the
    /// cluster stage, the leave-one-out ZNCC at the track stage.
    pub min_zncc: f64,
    /// How far an observation may sit from its seed (cluster) or from the
    /// surfel's projection (track), in source-image px.
    pub max_shift_px: f64,
    /// The largest tile localizability sigma_pos an observation may have, in
    /// grid px.
    pub max_keypoint_uncertainty: f64,
    /// The fraction of the track's own self-agreement a candidate's ZNCC has to
    /// reach, which view selection scores a sweep candidate by.
    pub min_relative_zncc: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        let cluster = ClusterRefineParams::default();
        Self {
            min_zncc: cluster.min_zncc,
            max_shift_px: cluster.max_shift_px,
            max_keypoint_uncertainty: KeypointLocalizeParams::default()
                .max_member_keypoint_uncertainty,
            min_relative_zncc: ViewSelectParams::default().min_relative_zncc,
        }
    }
}

/// A track being worked on: everything that has been tried against it, and
/// nothing about the window it is being looked at in.
///
/// The observations are the list; the stage says which kernels apply and what a
/// commit can do; the origin says whether a commit replaces a point or creates
/// one; the thresholds are the bars the painting proposes verdicts against.
///
/// One `in` observation per image is the invariant every step that sets a
/// verdict holds: a track cannot observe an image twice, so a second candidate
/// in an image already held is shown and scored but cannot be turned `in` until
/// the other is turned `out`.
#[derive(Debug, Clone, PartialEq)]
pub struct EditableTrack {
    /// The observations, in the order they were added. Never renumbered.
    pub observations: Vec<Observation>,
    /// Which representation the track is in, and that representation's data.
    pub stage: Stage,
    /// The point this track was put on the bench from, when there was one.
    pub origin: Option<Origin>,
    /// The bars the painting judges against.
    pub thresholds: Thresholds,
}

impl EditableTrack {
    /// An empty track at the cluster stage, with the default thresholds and no
    /// origin.
    ///
    /// Its reference names observation 0, which does not exist yet; the first
    /// observation added takes that place.
    pub fn empty_cluster() -> Self {
        Self {
            observations: Vec::new(),
            stage: Stage::Cluster(ClusterPayload {
                reference: 0,
                template: None,
            }),
            origin: None,
            thresholds: Thresholds::default(),
        }
    }

    /// Which stage the track is in.
    pub fn stage_kind(&self) -> StageKind {
        self.stage.kind()
    }

    /// How many observations carry each verdict, as `(in, candidate, out)`.
    pub fn verdict_counts(&self) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for observation in &self.observations {
            match observation.verdict {
                Verdict::In => counts.0 += 1,
                Verdict::Candidate => counts.1 += 1,
                Verdict::Out => counts.2 += 1,
            }
        }
        counts
    }

    /// The indexes of the `in` observations, ascending.
    pub fn in_observations(&self) -> Vec<usize> {
        self.observations
            .iter()
            .enumerate()
            .filter(|(_, o)| o.verdict == Verdict::In)
            .map(|(i, _)| i)
            .collect()
    }

    /// The index of the `in` observation in `image`, when the track holds one.
    pub fn in_observation_of_image(&self, image: u32) -> Option<usize> {
        self.observations
            .iter()
            .position(|o| o.image == image && o.verdict == Verdict::In)
    }

    /// A copy of this track whose origin is the point at `point` in version
    /// `version`.
    ///
    /// What a caller re-seats a committed track with, so a second commit of it
    /// replaces what the first wrote.
    pub fn with_origin(&self, version: u64, point: u32) -> Self {
        let mut next = self.clone();
        next.origin = Some(Origin { version, point });
        next
    }

    /// The cluster payload, or `None` at the track stage.
    pub fn cluster(&self) -> Option<&ClusterPayload> {
        match &self.stage {
            Stage::Cluster(payload) => Some(payload),
            Stage::Track(_) => None,
        }
    }

    /// The track payload, or `None` at the cluster stage.
    pub fn track(&self) -> Option<&TrackPayload> {
        match &self.stage {
            Stage::Track(payload) => Some(payload),
            Stage::Cluster(_) => None,
        }
    }
}
