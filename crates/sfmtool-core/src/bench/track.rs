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
    /// A `.sift` feature the person or a caller named directly: the detected
    /// keypoint a cluster was started on, or one added to a track by index.
    Descriptor {
        /// The feature's index in its image's `.sift` file.
        feature: u32,
    },
    /// An image a descriptor search found the patch in.
    ///
    /// Separate from [`Provenance::Descriptor`] because the two name different
    /// things. A descriptor provenance names **one detected feature**, which
    /// the observation sits exactly on. A search's observation sits wherever
    /// the image's affine warp puts the pixel that was searched from, which is
    /// in general no feature at all; what stands behind it is the number of
    /// correspondences that agreed on that warp, and that is what is worth
    /// showing beside the row.
    Search {
        /// Correspondences that voted for the warp this observation was placed
        /// by. It ranks the search's candidates against one another, and it is
        /// an admission the photometry then judges.
        inliers: u32,
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
    /// The seed's affine shape: the detector's canonical **keypoint frame**
    /// mapped onto this image's pixels, the same `S` the `.matches`
    /// cluster-patches section stores.
    ///
    /// A shape is a scale and not a size on its own. The patch is the square
    /// `[-r, r]^2` of keypoint-frame units, where `r` is
    /// [`ClusterPayload::radius`], so the sighting's pixel half-width along a
    /// column is `r * ||column||` and a shape read without that radius says
    /// nothing about how large the patch is.
    pub seed_shape: [[f64; 2]; 2],
    /// Where the refinement put the observation, in that image's pixels.
    pub position: Option<[f64; 2]>,
    /// The refined absolute affine shape, in the same convention as
    /// [`Self::seed_shape`]: keypoint frame to pixels, over the same
    /// `[-r, r]^2` square.
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

/// Why an observation carries no measurement at the track stage.
///
/// An evaluation drops nothing: it turns off the localizer's own gates and its
/// consensus-basis cap, so every observation it can read comes back with a
/// number. What is left is the observation it cannot read at all, and this says
/// which of those it was, in one short sentence, so a row without a ZNCC never
/// reads as an unexplained refusal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unmeasured {
    /// Nothing says where the observation sits in its photograph: it carries
    /// neither a keypoint nor a cluster seed.
    NoSeed,
    /// Where it sits is off its photograph's sensor, so there is no tile to cut
    /// around it.
    OffSensor,
    /// The track's point does not project into this view, so there is no anchor
    /// to render a tile about.
    NoProjection,
    /// The view's ray runs near-parallel to the patch plane, where nothing pins
    /// an in-plane position.
    Grazing {
        /// `|d_hat . n_hat|`, the cosine the grazing cutoff judges.
        cosine: f64,
    },
    /// Fewer than two observations of its round could be read together, so
    /// there was no consensus to correlate this one against.
    NoConsensus,
    /// The correlation could not be scored: the tile around the observation
    /// runs off the photograph, or no channel of it carries texture.
    Unscorable,
}

impl std::fmt::Display for Unmeasured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unmeasured::NoSeed => write!(f, "nothing says where it sits"),
            Unmeasured::OffSensor => write!(f, "it sits off the photograph"),
            Unmeasured::NoProjection => write!(f, "the point misses this view"),
            Unmeasured::Grazing { cosine } => write!(f, "its ray grazes the patch ({cosine:.2})"),
            Unmeasured::NoConsensus => write!(f, "nothing to correlate against"),
            Unmeasured::Unscorable => write!(f, "its tile could not be scored"),
        }
    }
}

/// What the track stage has measured about one observation.
///
/// A track put on the bench from a committed point arrives with
/// [`Self::keypoint`] and [`Self::zncc`] read off the stored columns; the rest
/// is what an evaluation computes.
///
/// The two distances are different questions, and both are here because a
/// person reading a row has to tell them apart: [`Self::seed_shift_px`] is
/// about the **observation** -- how far the correlation peak sits from where
/// the sighting is -- and [`Self::projection_offset_px`] is about the
/// **point** -- how far the sighting sits from where the position puts it. A
/// mis-triangulated point gives every row a large offset while the shifts stay
/// at zero.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrackMeasurement {
    /// Where the localizer put the observation, in that image's pixels. This is
    /// the pixel a commit writes.
    ///
    /// An evaluation never writes it: it reads the track as it stands, and this
    /// pixel is the thing it reads.
    pub keypoint: Option<[f32; 2]>,
    /// The leave-one-out ZNCC against the consensus of the round's other
    /// observations, at the correlation peak within the search radius of this
    /// observation's own keypoint. With [`Self::seed_shift_px`] near zero it is
    /// the agreement at the keypoint itself.
    pub zncc: Option<f64>,
    /// How far that correlation peak sits from the observation's own keypoint,
    /// in source-image px: the observation's own evidence, and what
    /// [`Thresholds::max_shift_px`] paints on.
    pub seed_shift_px: Option<f64>,
    /// How far the observation's keypoint sits from the point's projection, in
    /// source-image px: the number that says how far the **point** is off,
    /// rather than the sighting.
    pub projection_offset_px: Option<f64>,
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
    /// Why there is no ZNCC, when there is none: an evaluation that could not
    /// read an observation says which of its refusals it was rather than
    /// leaving the row blank.
    pub reason: Option<Unmeasured>,
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
    /// The template's half-width, in keypoint-frame units: the patch every
    /// observation's shape is read over is the square `[-radius, radius]^2` of
    /// those units.
    ///
    /// **This is the cluster's one scale.** The seed and refined shapes in
    /// [`ClusterMeasurement`] are maps from keypoint-frame units to pixels and
    /// carry no size of their own, so what says how large the patches are is
    /// this number and nothing else. An evaluation runs the refinement kernel
    /// at exactly this radius rather than at
    /// [`ClusterRefineParams::radius`](crate::patch::cluster_refine::ClusterRefineParams::radius),
    /// so the meaning of a seed cannot change under it; everything drawn --
    /// the overlay's parallelogram, the Track Edit tile -- is read at it too.
    pub radius: f64,
    /// The cut itself. `None` until an evaluation cuts it, because the cut is a
    /// function of the reference's pixels and the bench holds no photographs.
    pub template: Option<ClusterTemplate>,
}

impl Default for ClusterPayload {
    /// A cluster cut around observation 0, at the refinement kernel's own
    /// template radius, with no template yet.
    ///
    /// The radius is read from the kernel's parameter type rather than written
    /// out again, so a bench cluster and a batch pass start from one scale.
    fn default() -> Self {
        Self {
            reference: 0,
            radius: ClusterRefineParams::default().radius,
            template: None,
        }
    }
}

/// The template the cluster stage registers its observations onto.
///
/// The samples are the reference observation's own tile on the template grid,
/// as the refinement's sampler reads it, which is the tile a panel draws. The
/// correlation the cascade runs z-normalizes it inside the kernel, over the
/// window and without the pixels that window drops, so what is kept here is the
/// picture rather than the kernel's working copy of it.
///
/// The cut's half-width is [`ClusterPayload::radius`] and is not repeated here:
/// one number says how large the cluster's patches are, and a template that
/// carried a second copy of it could disagree with the seeds it was cut from.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterTemplate {
    /// The `(resolution, resolution, channels)` samples.
    pub samples: Array3<f32>,
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
    /// How far the correlation peak may sit from where the observation sits, in
    /// source-image px: [`ClusterMeasurement::shift_px`] at the cluster stage
    /// and [`TrackMeasurement::seed_shift_px`] at the track stage.
    ///
    /// Both are the observation's **own** evidence. The bar is deliberately not
    /// judged on [`TrackMeasurement::projection_offset_px`], which is a verdict
    /// on the point rather than on the sighting: a mis-triangulated point would
    /// otherwise turn out every observation of the track that would fix it.
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
            stage: Stage::Cluster(ClusterPayload::default()),
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
