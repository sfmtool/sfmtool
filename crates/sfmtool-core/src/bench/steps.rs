// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The steps that put an item on the bench and change one: each a pure function
//! from a value plus inputs to a new value and a report.
//!
//! `specs/core/bench/editable-track.md` is the design. Nothing here reads a
//! photograph or runs a kernel: these are the steps that are decided by what
//! the reconstruction and the person already say. The steps that register
//! pixels are the evaluation and the stage change, and they are separate --
//! with one seam: a split of a track-stage track puts the half it takes off
//! down to the cluster stage, which is the stage step's own work over the
//! cameras and reads no photograph either.

use std::sync::Arc;

use nalgebra::{Point3, Vector3};

use crate::patch::cloud::OrientedPatch;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::fit::FitOptions;
use super::stage::{set_stage, StageError};
use super::track::{
    ClusterMeasurement, ClusterPayload, EditableTrack, Observation, Origin, Provenance, Stage,
    StageKind, Thresholds, TrackMeasurement, TrackPayload, Verdict,
};
use super::{Bench, BenchItem, ItemKind};

/// How many hex digits of a content hash a point id names it by, which is what
/// a label minted from a point row reads as.
const HASH_PREFIX_LEN: usize = 8;

/// A determinant this small makes an affine seed shape unusable: its columns
/// span no area, so there is no frame to warp a template through.
const MIN_ABS_DET: f64 = 1e-9;

/// What putting an item on the bench did.
///
/// The label is the whole of it: an item is named by its label everywhere, and
/// a caller that wants the value reads it back off the returned bench.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateReport {
    /// The label the new item took, after any collision suffix.
    pub label: String,
    /// Which kind of item it is.
    pub kind: ItemKind,
    /// How many observations it started with.
    pub observation_count: usize,
}

impl std::fmt::Display for CreateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "put {} on the bench: {} {} observations",
            self.label, self.kind, self.observation_count
        )
    }
}

// ---- Putting a committed point on the bench -------------------------------

/// What a track put on the bench from a point is labelled and dated by.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CreateTrackOptions {
    /// The version serial the origin records, as the caller numbers versions.
    /// Core neither mints nor interprets it.
    pub version: u64,
    /// The label to put the track on the bench under, before the collision
    /// suffix. The viewer passes the point's portable id, so the item is named
    /// by the point it came from.
    ///
    /// `None` mints one from what core can see: `pt3d_<hash>_<index>` over the
    /// first eight hex digits of the base's content hash for a point that is a
    /// row of that base, and `point_<index>` for a point the overlay added,
    /// which is a row of no content and which only the caller's own version
    /// graph can name.
    pub label: Option<String>,
}

/// Why a point could not be put on the bench.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateTrackError {
    /// The edited index names no live point: past the end, or deleted.
    NoSuchPoint(u32),
}

impl std::fmt::Display for CreateTrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreateTrackError::NoSuchPoint(i) => write!(f, "no live point at index {i}"),
        }
    }
}

impl std::error::Error for CreateTrackError {}

/// Put the point at `point` on the bench as a track-stage editable track.
///
/// The track arrives with the point's own frame, bitmap and keypoints, its
/// origin set to that point, and every observation `in` and unpinned. The
/// measurements are carried from what the record stores and nothing is
/// recomputed: the leave-one-out ZNCC is `observation_confidence` read back out
/// of its byte scale where the column exists, and everything an evaluation
/// would compute is left unmeasured. So putting a track on the bench and doing
/// nothing shows the numbers the reconstruction already holds, plus the verdict
/// column.
///
/// A `sift_files` reconstruction is put on the bench like any other: inspecting
/// a track is allowed everywhere, and it is [`commit`](super::commit::commit)
/// that refuses to write one back.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{create_track, Bench, CreateTrackOptions};
/// # use sfmtool_core::EditedReconstruction;
/// # fn run(edited: &EditedReconstruction) -> Result<(), Box<dyn std::error::Error>> {
/// let bench = Bench::new();
/// let (bench, report) = create_track(&bench, edited, 1207, &CreateTrackOptions::default())?;
/// let track = bench.track(&report.label).expect("just put on");
/// assert_eq!(track.verdict_counts().0, track.observations.len());
/// # Ok(())
/// # }
/// ```
pub fn create_track(
    bench: &Bench,
    edited: &EditedReconstruction,
    point: u32,
    options: &CreateTrackOptions,
) -> Result<(Bench, CreateReport), CreateTrackError> {
    let view = edited
        .point(point)
        .ok_or(CreateTrackError::NoSuchPoint(point))?;
    let stored = view.point().clone();

    let observations = view
        .observations()
        .iter()
        .enumerate()
        .map(|(k, obs)| Observation {
            image: obs.image_index,
            provenance: Provenance::Origin,
            verdict: Verdict::In,
            pinned: false,
            cluster: None,
            track: Some(TrackMeasurement {
                keypoint: view.keypoint_xy(k),
                // The stored column is the fit's own leave-one-out score in a
                // byte scale; reading it back is carrying a measurement, not
                // making one.
                zncc: view
                    .observation_confidence()
                    .map(|c| f64::from(c[k]) / f64::from(u8::MAX)),
                ..TrackMeasurement::default()
            }),
        })
        .collect::<Vec<_>>();

    let frame = match (view.patch_u_halfvec(), view.patch_v_halfvec()) {
        (Some(u), Some(v)) => {
            let u = Vector3::new(f64::from(u[0]), f64::from(u[1]), f64::from(u[2]));
            let v = Vector3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
            let (hu, hv) = (u.norm(), v.norm());
            (hu > 0.0 && hv > 0.0).then(|| {
                let mut patch = OrientedPatch::new(stored.position, u / hu, v / hv, [hu, hv]);
                // A bearing's frame is tangent to the direction sphere and its
                // corners are directions, so the renderer has to be told which
                // kind it is; the two stored half-vectors are the same either
                // way.
                patch.w = if stored.is_at_infinity() { 0.0 } else { 1.0 };
                patch
            })
        }
        _ => None,
    };

    let payload = TrackPayload {
        position: Some(stored.position),
        frame,
        bitmap: view.patch_bitmap().map(|b| b.to_owned()),
        color: stored.color,
        normal_confidence: view.normal_confidence(),
        condition_number: None,
    };

    let track = EditableTrack {
        observations,
        stage: Stage::Track(payload),
        origin: Some(Origin {
            version: options.version,
            point,
        }),
        thresholds: Thresholds::default(),
    };

    let base = match &options.label {
        Some(id) => id.clone(),
        None => default_point_label(edited, point),
    };
    let count = track.observations.len();
    let (bench, label) = bench.put(&base, BenchItem::Track(Arc::new(track)));
    Ok((
        bench,
        CreateReport {
            label,
            kind: ItemKind::Track,
            observation_count: count,
        },
    ))
}

/// The label core mints for a point when the caller names no point id.
///
/// A row of the base is named the way a portable point id names it, by the
/// content it is a row of; a point the overlay added is a row of no content at
/// all, and only the caller's version graph can say what created it, so it is
/// named by its index alone.
fn default_point_label(edited: &EditedReconstruction, point: u32) -> String {
    if (point as usize) < edited.base_point_count() {
        if let Ok(hash) = edited.base_content_hash() {
            let prefix = &hash.content_xxh128;
            if prefix.len() >= HASH_PREFIX_LEN {
                return format!("pt3d_{}_{point}", &prefix[..HASH_PREFIX_LEN]);
            }
        }
    }
    format!("point_{point}")
}

// ---- Starting a cluster from one image ------------------------------------

/// The one observation a cluster-stage track starts with.
///
/// The affine shape is the seed's own frame: the map from the detector's
/// canonical **keypoint frame** onto this image's pixels, the same `S` the
/// `.matches` cluster-patches section stores. A pixel someone pointed at has no
/// detector behind it, so [`Self::from_pixel`] makes one from a radius in
/// pixels; a `.sift` feature carries its own.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterSeed {
    /// The image, as an index into the node's image table.
    pub image: u32,
    /// The image's stem, which the label is minted from.
    pub image_stem: String,
    /// Where in it, in source-image px.
    pub pixel: [f64; 2],
    /// The affine shape at that pixel: keypoint-frame units to this image's
    /// pixels, read over the square `[-r, r]^2` where `r` is the
    /// [`ClusterPayload::radius`] the cluster takes. So the seed's pixel
    /// half-width along a column is `r * ||column||`, and a shape without that
    /// radius says nothing about how large the patch is.
    pub shape: [[f64; 2]; 2],
    /// The `.sift` feature this seed is, when it is one.
    pub feature: Option<u32>,
}

impl ClusterSeed {
    /// The isotropic keypoint-frame shape whose patch is `radius_px` pixels
    /// across its half-width, for the cluster a [`create_cluster`] makes.
    ///
    /// The template spans `[-r, r]^2` keypoint-frame units at
    /// `ClusterPayload::default().radius`, so one keypoint-frame unit has to be
    /// `radius_px / r` pixels for the patch to be the size that was asked for.
    /// The division is here, and the multiplication is in the kernel and in
    /// everything that draws, so a person naming a radius in pixels and a
    /// kernel reading a shape never mean two different squares.
    pub fn shape_from_radius_px(radius_px: f64) -> [[f64; 2]; 2] {
        let unit = radius_px / ClusterPayload::default().radius;
        [[unit, 0.0], [0.0, unit]]
    }

    /// A seed at a pixel someone pointed at, whose patch is `radius_px` pixels
    /// from the pixel to the edge.
    ///
    /// Nothing in a pixel says how large the patch around it is, so the caller
    /// names it, exactly as creating a point from a pixel does, and in the same
    /// units: source-image pixels. The shape it becomes is
    /// [`Self::shape_from_radius_px`].
    pub fn from_pixel(
        image: u32,
        image_stem: impl Into<String>,
        pixel: [f64; 2],
        radius_px: f64,
    ) -> Self {
        Self {
            image,
            image_stem: image_stem.into(),
            pixel,
            shape: Self::shape_from_radius_px(radius_px),
            feature: None,
        }
    }

    /// A seed at a `.sift` feature, which carries its own position and shape.
    pub fn from_feature(
        image: u32,
        image_stem: impl Into<String>,
        feature: u32,
        pixel: [f64; 2],
        shape: [[f64; 2]; 2],
    ) -> Self {
        Self {
            image,
            image_stem: image_stem.into(),
            pixel,
            shape,
            feature: Some(feature),
        }
    }

    /// The label this seed mints: the image stem and the pixel for a hand-placed
    /// seed, the image stem and the feature index for a detected one.
    pub fn label(&self) -> String {
        match self.feature {
            Some(feature) => format!("{}#{feature}", self.image_stem),
            None => format!(
                "{}@{},{}",
                self.image_stem,
                self.pixel[0].round() as i64,
                self.pixel[1].round() as i64
            ),
        }
    }
}

/// Why a cluster could not be started.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateClusterError {
    /// The pixel is not a finite pair of coordinates.
    BadPixel([f64; 2]),
    /// The seed's affine shape spans no area, so nothing can be warped through
    /// it.
    DegenerateShape([[f64; 2]; 2]),
}

impl std::fmt::Display for CreateClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreateClusterError::BadPixel(p) => {
                write!(f, "({}, {}) is not a pixel", p[0], p[1])
            }
            CreateClusterError::DegenerateShape(_) => write!(
                f,
                "the seed's affine shape spans no area, so it frames no patch"
            ),
        }
    }
}

impl std::error::Error for CreateClusterError {}

/// Put a new cluster-stage track on the bench with `seed` as its one
/// observation and its reference.
///
/// The observation is `in`: it is the thing the person pointed at, and the
/// cluster is the set of images that register onto it. The template is left
/// uncut, because cutting it reads the reference's pixels and this step reads
/// no photograph.
///
/// The cluster takes the default [`ClusterPayload::radius`], which is the
/// radius [`ClusterSeed::from_pixel`] sized its shape against, so a seed named
/// in pixels arrives on the bench at the size it was named.
pub fn create_cluster(
    bench: &Bench,
    seed: &ClusterSeed,
) -> Result<(Bench, CreateReport), CreateClusterError> {
    if !seed.pixel.iter().all(|c| c.is_finite()) {
        return Err(CreateClusterError::BadPixel(seed.pixel));
    }
    let det = seed.shape[0][0] * seed.shape[1][1] - seed.shape[0][1] * seed.shape[1][0];
    if !det.is_finite() || det.abs() < MIN_ABS_DET {
        return Err(CreateClusterError::DegenerateShape(seed.shape));
    }

    let provenance = match seed.feature {
        Some(feature) => Provenance::Descriptor { feature },
        None => Provenance::Pixel,
    };
    let mut observation = Observation::seeded(seed.image, provenance, seed.pixel, seed.shape);
    observation.verdict = Verdict::In;

    let track = EditableTrack {
        observations: vec![observation],
        stage: Stage::Cluster(ClusterPayload::default()),
        origin: None,
        thresholds: Thresholds::default(),
    };

    let (bench, label) = bench.put(&seed.label(), BenchItem::Track(Arc::new(track)));
    Ok((
        bench,
        CreateReport {
            label,
            kind: ItemKind::Track,
            observation_count: 1,
        },
    ))
}

// ---- Steps on one track ----------------------------------------------------

/// Why a step on one track was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackEditError {
    /// The observation index is past the end of the track.
    NoSuchObservation {
        /// The index named.
        observation: usize,
        /// How many observations the track holds.
        observation_count: usize,
    },
    /// An observation was to be turned `in` in an image another `in`
    /// observation already holds. A track cannot observe an image twice.
    ImageAlreadyIn {
        /// The image.
        image: u32,
        /// The observation that already holds it.
        observation: usize,
    },
    /// The seed is not a finite pair of coordinates.
    BadPixel([f64; 2]),
    /// The step belongs to the other stage: the surfel's own size and turn are
    /// the track stage's, and one sighting's affine shape is the cluster
    /// stage's.
    WrongStage {
        /// The stage the step acts at.
        wanted: StageKind,
        /// The stage the track is in.
        is: StageKind,
    },
    /// The track is at the track stage and carries no surfel to resize or turn.
    NoFrame,
    /// The half-length asked for is not a positive finite length.
    BadSize(f64),
    /// The turn asked for is not a finite angle.
    BadAngle(f64),
    /// The affine shape has no area, so there is no frame to warp a template
    /// through.
    BadShape([[f64; 2]; 2]),
    /// Nothing says where the observation sits, so there is no place to re-seat
    /// it at.
    NoPlace {
        /// The observation named.
        observation: usize,
    },
    /// The observation is in an image the reconstruction does not have, or one
    /// whose camera it does not have.
    NoSuchImage {
        /// The image named.
        image: u32,
    },
    /// The patch and the pixel do not meet in that observation's view: the
    /// centre falls outside the lens model's domain, or the pointer's ray runs
    /// parallel to the patch's plane.
    NoProjection {
        /// The observation whose view was used.
        observation: usize,
    },
}

impl std::fmt::Display for TrackEditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackEditError::NoSuchObservation {
                observation,
                observation_count,
            } => write!(
                f,
                "observation {observation} is past the {observation_count} \
                 observations of this track"
            ),
            TrackEditError::ImageAlreadyIn { image, observation } => write!(
                f,
                "image {image} already has observation {observation} in; turn it out first"
            ),
            TrackEditError::BadPixel(p) => write!(f, "({}, {}) is not a pixel", p[0], p[1]),
            TrackEditError::WrongStage { wanted, is } => {
                write!(f, "that is a {wanted}-stage step and this track is a {is}")
            }
            TrackEditError::NoFrame => {
                write!(f, "this track has no surfel yet; fit it first")
            }
            TrackEditError::BadSize(size) => write!(f, "{size} is not a size"),
            TrackEditError::BadAngle(angle) => write!(f, "{angle} is not an angle"),
            TrackEditError::BadShape(shape) => write!(
                f,
                "the shape [[{}, {}], [{}, {}]] spans no area",
                shape[0][0], shape[0][1], shape[1][0], shape[1][1]
            ),
            TrackEditError::NoPlace { observation } => {
                write!(f, "nothing says where observation {observation} sits")
            }
            TrackEditError::NoSuchImage { image } => {
                write!(f, "image {image} is not in this reconstruction")
            }
            TrackEditError::NoProjection { observation } => write!(
                f,
                "the patch and that pixel do not meet in observation {observation}'s view"
            ),
        }
    }
}

impl std::error::Error for TrackEditError {}

/// Where a new observation goes, and what put it there.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationSeed {
    /// The image, as an index into the node's image table.
    pub image: u32,
    /// Where in it, in source-image px.
    pub pixel: [f64; 2],
    /// The affine shape at that pixel, when the caller has one, in the cluster
    /// stage's own convention: keypoint-frame units to that image's pixels,
    /// read over `[-r, r]^2` at the track's [`ClusterPayload::radius`].
    ///
    /// A pixel gesture on a track that already has a reference has none:
    /// [`add_observation`] takes the reference's own shape, which is the scale
    /// the track already works at.
    pub shape: Option<[[f64; 2]; 2]>,
    /// What put it there.
    pub provenance: Provenance,
}

impl ObservationSeed {
    /// A seed at a pixel the person pointed at.
    pub fn at_pixel(image: u32, pixel: [f64; 2]) -> Self {
        Self {
            image,
            pixel,
            shape: None,
            provenance: Provenance::Pixel,
        }
    }
}

/// What one added observation did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddObservationReport {
    /// The index the new observation took, which is the end of the list and is
    /// stable for the life of the track.
    pub observation: usize,
    /// The image it is in.
    pub image: u32,
}

/// Add a candidate observation to `track`.
///
/// It joins as a `candidate` and unpinned: something proposed it and nobody has
/// ruled on it, which is what a candidate is. A second observation in an image
/// the track already holds is allowed and is scored like any other; what it
/// cannot do is be turned `in` while the other is
/// ([`set_verdict`] refuses that).
///
/// The seed lands in the cluster slot, which is where a seed means something:
/// a position and a shape in one image's pixels, with no geometry behind them.
/// The shape defaults to the reference observation's own when the track has a
/// reference with a seed, so a pixel gesture on a track that already has a
/// scale needs no radius prompt and lands at that track's size.
///
/// With no reference to copy the shape is the identity, which is one pixel to
/// the keypoint-frame unit: a patch of `[-r, r]` **pixels** at the track's
/// [`ClusterPayload::radius`]. That is the cluster stage's convention like any
/// other shape, and it is what a track with nothing to say about its own scale
/// is worth.
pub fn add_observation(
    track: &EditableTrack,
    seed: &ObservationSeed,
) -> Result<(EditableTrack, AddObservationReport), TrackEditError> {
    if !seed.pixel.iter().all(|c| c.is_finite()) {
        return Err(TrackEditError::BadPixel(seed.pixel));
    }
    let shape = seed
        .shape
        .or_else(|| reference_shape(track))
        .unwrap_or([[1.0, 0.0], [0.0, 1.0]]);

    let mut next = track.clone();
    next.observations.push(Observation::seeded(
        seed.image,
        seed.provenance,
        seed.pixel,
        shape,
    ));
    Ok((
        next,
        AddObservationReport {
            observation: track.observations.len(),
            image: seed.image,
        },
    ))
}

/// The reference observation's own seed shape, when the track is a cluster with
/// a reference that has one.
pub(super) fn reference_shape(track: &EditableTrack) -> Option<[[f64; 2]; 2]> {
    let reference = track.cluster()?.reference;
    let cluster = track.observations.get(reference)?.cluster.as_ref()?;
    Some(cluster.shape.unwrap_or(cluster.seed_shape))
}

/// What one verdict did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerdictReport {
    /// The observation the verdict was set on.
    pub observation: usize,
    /// What it was.
    pub was: Verdict,
    /// What it is now.
    pub is: Verdict,
    /// Whether anything changed. A verdict an observation already has, set
    /// again by hand, pins it and changes nothing else.
    pub changed: bool,
}

/// Set the verdict of one observation, by hand.
///
/// The verdict is pinned by this: it is the person's, and
/// [`apply_thresholds`] leaves a pinned verdict where it is.
///
/// Turning an observation `in` is refused when another `in` observation already
/// holds its image, because a track observes an image once.
pub fn set_verdict(
    track: &EditableTrack,
    observation: usize,
    verdict: Verdict,
) -> Result<(EditableTrack, VerdictReport), TrackEditError> {
    let current = track
        .observations
        .get(observation)
        .ok_or(TrackEditError::NoSuchObservation {
            observation,
            observation_count: track.observations.len(),
        })?;
    let was = current.verdict;
    if verdict == Verdict::In {
        if let Some(held) = track.in_observation_of_image(current.image) {
            if held != observation {
                return Err(TrackEditError::ImageAlreadyIn {
                    image: current.image,
                    observation: held,
                });
            }
        }
    }
    let mut next = track.clone();
    let target = &mut next.observations[observation];
    target.verdict = verdict;
    target.pinned = true;
    Ok((
        next,
        VerdictReport {
            observation,
            was,
            is: verdict,
            changed: was != verdict,
        },
    ))
}

// ---- Placing a sighting, and sizing and turning the patch ------------------

/// One of a patch's two in-plane axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// The `u` axis, which the patch's first half-vector lies along.
    U,
    /// The `v` axis.
    V,
}

/// One of the four edges of a patch's square, named by the axis it lies across
/// and the side it is on.
///
/// What a person grabs when they resize a patch by its outline, and what the
/// wire's resize tool names as `"+u"`, `"-u"`, `"+v"` or `"-v"`. The edge
/// matters and not just the axis, because a resize holds the **opposite** edge
/// still: which of the two moves is the whole of the difference between the
/// patch growing up and growing down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// The edge at `s = +1`.
    PlusU,
    /// The edge at `s = -1`.
    MinusU,
    /// The edge at `t = +1`.
    PlusV,
    /// The edge at `t = -1`.
    MinusV,
}

impl Edge {
    /// Which axis the edge is read on.
    pub fn axis(self) -> Axis {
        match self {
            Edge::PlusU | Edge::MinusU => Axis::U,
            Edge::PlusV | Edge::MinusV => Axis::V,
        }
    }

    /// Which side of the centre it sits on, as `-1.0` or `+1.0`.
    pub fn sign(self) -> f64 {
        match self {
            Edge::PlusU | Edge::PlusV => 1.0,
            Edge::MinusU | Edge::MinusV => -1.0,
        }
    }

    /// The four edges, in the order `"+u"`, `"-u"`, `"+v"`, `"-v"`.
    pub const ALL: [Edge; 4] = [Edge::PlusU, Edge::MinusU, Edge::PlusV, Edge::MinusV];

    /// The edge's name on the wire and in a report.
    pub fn name(self) -> &'static str {
        match self {
            Edge::PlusU => "+u",
            Edge::MinusU => "-u",
            Edge::PlusV => "+v",
            Edge::MinusV => "-v",
        }
    }
}

impl std::fmt::Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Edge {
    type Err = String;

    fn from_str(word: &str) -> Result<Self, Self::Err> {
        Edge::ALL
            .into_iter()
            .find(|edge| edge.name() == word)
            .ok_or_else(|| format!("{word:?} is not an edge; use +u, -u, +v or -v"))
    }
}

/// What one hand-placed sighting did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveObservationReport {
    /// The observation that was moved.
    pub observation: usize,
    /// The image it is in.
    pub image: u32,
    /// Where it sat, or `None` for an observation nothing said the place of.
    pub was: Option<[f64; 2]>,
    /// Where it sits now.
    pub pixel: [f64; 2],
    /// How far it moved, in that image's px, or `None` when it came from
    /// nowhere.
    pub moved_px: Option<f64>,
    /// Whether anything changed. A sighting put back exactly where it was
    /// pins it and changes nothing else.
    pub changed: bool,
}

/// Put one observation's sighting at `pixel`, by hand.
///
/// What a person dragging the observation's mark in the Image Detail panel
/// means, and the one step behind it at either stage:
///
/// - At the **track stage** it writes the observation's keypoint, which is the
///   pixel a commit writes and the place every reading is anchored at, and
///   drops the rest of that measurement. Everything else in a
///   [`TrackMeasurement`] -- the leave-one-out ZNCC, both distances, the
///   reprojection residual, the localizability, the reason -- was computed
///   *for the old keypoint* and says nothing about the new one, and an
///   evaluation recomputes all of it from the track as it stands, so clearing
///   is both honest and cheap to undo.
/// - At the **cluster stage** it re-seeds the observation at `pixel` with the
///   shape it is being read at, and drops the refinement. The shape is kept
///   because a person moving a mark is saying where the patch is and not how
///   large it is; what the refinement found around the old pixel is not an
///   answer about the new one.
///
/// Nothing else on the track moves: the surfel keeps its place, its size and
/// its turn, and every other sighting keeps its own.
///
/// **The observation is pinned either way**, at both stages: a sighting a
/// person placed is a sighting they have ruled on, and
/// [`apply_thresholds`] leaves a pinned observation's verdict where it is
/// rather than painting over a placement by hand.
pub fn set_observation_keypoint(
    track: &EditableTrack,
    observation: usize,
    pixel: [f64; 2],
) -> Result<(EditableTrack, MoveObservationReport), TrackEditError> {
    if !pixel.iter().all(|c| c.is_finite()) {
        return Err(TrackEditError::BadPixel(pixel));
    }
    let current = observation_at(track, observation)?;
    let was = current.site();
    let image = current.image;
    let shape = current.shape();

    let mut next = track.clone();
    let target = &mut next.observations[observation];
    match track.stage {
        Stage::Track(_) => {
            target.track = Some(TrackMeasurement {
                keypoint: Some([pixel[0] as f32, pixel[1] as f32]),
                ..TrackMeasurement::default()
            });
        }
        Stage::Cluster(_) => {
            let shape = shape
                .or_else(|| reference_shape(track))
                .unwrap_or([[1.0, 0.0], [0.0, 1.0]]);
            target.cluster = Some(ClusterMeasurement::from_seed(pixel, shape));
        }
    }
    target.pinned = true;
    let moved_px = was.map(|was| (pixel[0] - was[0]).hypot(pixel[1] - was[1]));
    Ok((
        next,
        MoveObservationReport {
            observation,
            image,
            was,
            pixel,
            moved_px,
            changed: was != Some(pixel),
        },
    ))
}

/// What one resize did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResizeReport {
    /// The observation whose outline was dragged, for
    /// [`resize_from_edge`]; the resize is of the one patch either way, and
    /// this says which sighting's view of it the size was named in.
    pub observation: Option<usize>,
    /// The image that observation is in.
    pub image: Option<u32>,
    /// The patch's new half-length: world units at the track stage, and the
    /// half-width along `u` in that image's own pixels at the cluster stage.
    pub half: f64,
    /// What it was, in the same unit.
    pub was: f64,
    /// Whether anything changed.
    pub changed: bool,
}

/// Resize the track's surfel to `half_length` along **both** of its axes, about
/// its own centre.
///
/// One scalar rather than two, because a patch frame is square: the half-vector
/// pair a `.sfmr` stores and every kernel that renders a tile reads has
/// `|u| == |v|`, and the grid the tile is sampled on is square with it
/// (`specs/core/patch/patch-cloud.md`). A resize that moved one axis alone
/// would make the patch a rectangle sampled as a square, which is a stretched
/// template rather than a larger one.
///
/// The centre, the axes' directions and the normal are untouched, so the patch
/// still sits on the same plane facing the same way and every edge moves: this
/// is the form a caller that knows the size it wants asks for. The gesture --
/// one edge dragged, the opposite one left where it is -- is
/// [`resize_from_edge`].
///
/// **What the resize invalidates is cleared.** The consensus bitmap is the
/// observations fused over the old square, so a frame of another size no longer
/// has a picture; and every track measurement but its keypoint was read over
/// that square, so what is kept is where each sighting sits and what is dropped
/// is every number about it. An [`evaluate`](super::evaluate::evaluate)
/// restores them, and the next [`fit`](super::fit::fit) fuses a new bitmap at
/// the size the frame now has.
pub fn resize_frame(
    track: &EditableTrack,
    half_length: f64,
) -> Result<(EditableTrack, ResizeReport), TrackEditError> {
    if !half_length.is_finite() || half_length <= 0.0 {
        return Err(TrackEditError::BadSize(half_length));
    }
    let frame = frame_of(track)?;
    let was = frame.half_extent[0];
    let mut next = track.clone();
    let changed = frame.half_extent != [half_length, half_length];
    if changed {
        let (position, frame, bitmap) = track_payload_mut(&mut next);
        frame.half_extent = [half_length, half_length];
        let _ = position;
        *bitmap = None;
        keep_keypoints_only(&mut next);
    }
    Ok((
        next,
        ResizeReport {
            observation: None,
            image: None,
            half: half_length,
            was,
            changed,
        },
    ))
}

/// Resize the patch by putting one edge of the outline drawn at `observation`
/// under `pixel`, with the **opposite edge left where it is**.
///
/// This is the gesture: a person grabs an edge of the square they can see and
/// pulls it, and what they expect is the edge under the pointer and the other
/// three where the geometry puts them -- not the patch breathing about its
/// centre with the far edge running away. So the arithmetic is the one that
/// holds the far edge still. With the dragged edge at `+h` from the centre and
/// the far one at `-h`, and the pointer naming the offset `p` along the dragged
/// direction, the new half-length is `(p + h) / 2` and the centre moves by
/// `h' - h` along that direction: the far edge stays at `-h` and the dragged
/// one lands on `p`.
///
/// **What the outline shows is what is resized.** At the track stage the
/// outline is the surfel re-anchored on `observation`'s own sighting
/// (`OrientedPatch::anchored_at_keypoint`), which is where a person sees the
/// patch in that photograph, so that is the frame the pointer is read against
/// and the frame the resize writes back. The surfel therefore takes the centre
/// the outline had, plus the edge's shift, and the track's position follows it;
/// `observation`'s keypoint is carried along the plane by the centre's own
/// displacement, keeping its in-plane offset, so the dot and the outline move
/// together and the far edge really does hold still on screen. Every other
/// sighting is carried by the same displacement and keeps its own offset too,
/// and all of them lose the measurements the move and the resize invalidate, as
/// they do for [`resize_frame`]. Nothing is pinned.
///
/// At the **cluster stage** there is no geometry, so the same arithmetic runs
/// in that image's pixels: the sighting's affine shape is scaled by one scalar,
/// which preserves whatever anisotropy the detector read at the keypoint, and
/// the sighting moves by half the change along the dragged edge's own
/// direction, which is what holds the far edge of the parallelogram still. Only
/// that observation is touched.
pub fn resize_from_edge(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    observation: usize,
    edge: Edge,
    pixel: [f64; 2],
) -> Result<(EditableTrack, ResizeReport), TrackEditError> {
    if !pixel.iter().all(|c| c.is_finite()) {
        return Err(TrackEditError::BadPixel(pixel));
    }
    let current = observation_at(track, observation)?;
    let image = current.image;
    let site = current
        .site()
        .ok_or(TrackEditError::NoPlace { observation })?;
    match &track.stage {
        Stage::Track(_) => {
            let frame = frame_of(track)?;
            let (camera, cam_from_world) = view_of(edited, image)?;
            let anchored = frame
                .anchored_at_keypoint(&camera, &cam_from_world, site)
                .ok_or(TrackEditError::NoProjection { observation })?;
            let direction = match edge.axis() {
                Axis::U => anchored.u_axis,
                Axis::V => anchored.v_axis,
            } * edge.sign();
            let offset = anchored
                .keypoint_plane_offset(&camera, &cam_from_world, pixel)
                .ok_or(TrackEditError::NoProjection { observation })?;
            let was = anchored.half_extent[0];
            let half = (offset.dot(&direction) + was) / 2.0;
            if !half.is_finite() || half <= 0.0 {
                return Err(TrackEditError::BadSize(half));
            }
            let displacement = direction * (half - was);
            let mut center = frame.center + displacement;
            // A direction patch's centre is a unit bearing and its half-extents
            // are stated against one, so the moved centre is renormalized and
            // the half-length divided by the same factor. Scaling a bearing and
            // its tangent frame together leaves every corner the same
            // direction, which is what keeps the far edge exactly where it was.
            let scale = if frame.w == 0.0 {
                let norm = center.coords.norm();
                if norm <= 1e-12 {
                    return Err(TrackEditError::BadSize(half));
                }
                center = nalgebra::Point3::from(center.coords / norm);
                norm
            } else {
                1.0
            };
            let half = half / scale;

            let mut next = track.clone();
            {
                let (position, frame, bitmap) = track_payload_mut(&mut next);
                frame.center = center;
                frame.half_extent = [half, half];
                if frame.w != 0.0 {
                    *position = Some(center);
                }
                *bitmap = None;
            }
            carry_keypoints(&mut next, edited, displacement);
            Ok((
                next,
                ResizeReport {
                    observation: Some(observation),
                    image: Some(image),
                    half,
                    was,
                    changed: half != was,
                },
            ))
        }
        Stage::Cluster(payload) => {
            let radius = payload.radius;
            let shape = current
                .shape()
                .ok_or(TrackEditError::NoPlace { observation })?;
            let det = shape[0][0] * shape[1][1] - shape[0][1] * shape[1][0];
            if det.abs() < MIN_ABS_DET {
                return Err(TrackEditError::BadShape(shape));
            }
            // The pointer's offset from the sighting, written in the shape's own
            // two columns: one coefficient per axis, in keypoint-frame units.
            let (dx, dy) = (pixel[0] - site[0], pixel[1] - site[1]);
            let along = match edge.axis() {
                Axis::U => (shape[1][1] * dx - shape[0][1] * dy) / det,
                Axis::V => (-shape[1][0] * dx + shape[0][0] * dy) / det,
            } / (radius * edge.sign());
            let scale = (along + 1.0) / 2.0;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(TrackEditError::BadSize(scale));
            }
            // From the centre to the dragged edge's midpoint, in pixels.
            let column = match edge.axis() {
                Axis::U => [shape[0][0], shape[1][0]],
                Axis::V => [shape[0][1], shape[1][1]],
            };
            let reach = [
                column[0] * radius * edge.sign(),
                column[1] * radius * edge.sign(),
            ];
            let position = [
                site[0] + reach[0] * (scale - 1.0),
                site[1] + reach[1] * (scale - 1.0),
            ];
            let scaled = [
                [shape[0][0] * scale, shape[0][1] * scale],
                [shape[1][0] * scale, shape[1][1] * scale],
            ];
            let was = half_width_px(shape, radius);
            let mut next = track.clone();
            next.observations[observation].cluster =
                Some(ClusterMeasurement::from_seed(position, scaled));
            Ok((
                next,
                ResizeReport {
                    observation: Some(observation),
                    image: Some(image),
                    half: half_width_px(scaled, radius),
                    was,
                    changed: scale != 1.0,
                },
            ))
        }
    }
}

/// What one translation of the surfel did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslateFrameReport {
    /// The observation whose image the pointer named.
    pub observation: usize,
    /// That image.
    pub image: u32,
    /// Where the centre now projects in it.
    pub pixel: [f64; 2],
    /// Where the centre now stands. A unit bearing for a direction patch.
    pub center: Point3<f64>,
    /// How far it moved, in world units.
    pub moved: f64,
    /// How many sightings the moved centre projects into, and so how many
    /// keypoints were written.
    pub placed: usize,
    /// Whether anything changed.
    pub changed: bool,
}

/// Slide the surfel across its own plane until its centre sits under `pixel` in
/// `observation`'s photograph.
///
/// **This moves the patch, not one sighting.** A track-stage track has one
/// surfel and every observation is a view of it, so dragging the mark in one
/// photograph is a statement about where that surfel is: the centre moves, the
/// half-vectors and the normal are kept, and **every** observation's keypoint
/// moves by the same displacement along the plane, so the outline moves in
/// every image at once. That is what makes the gesture worth having -- a patch
/// can be slid, turned and sized until it covers the piece of surface a person
/// means, and each photograph shows where it lands.
///
/// The pointer is read against the outline as drawn: the frame re-anchored on
/// `observation`'s own sighting, so the offset is measured from the square the
/// person can see. The move is in-plane by construction -- a ray-plane meeting
/// minus a point on the plane -- so the normal and the plane are untouched. **Every**
/// observation's keypoint is carried along the plane by that same displacement,
/// keeping its own in-plane offset from the centre: a keypoint is where that
/// photograph sees the patch's content, and the offset is what the tile is cut
/// on, so resetting keypoints to the centre's projection would scramble the
/// correlation the next reading scores. The sighting the drag came through
/// therefore lands under the pointer, and the others move with the patch.
///
/// **Nothing is pinned.** A translation says where the patch is and not whether
/// any sighting belongs to it, which is what a pin protects from the threshold
/// painting. The measurements go, as they do for a resize: every number beside
/// a keypoint was read at a place the patch has left. The bitmap goes with
/// them.
///
/// A sighting the moved centre no longer projects into is left with no keypoint
/// and [`Unmeasured::NoProjection`](super::track::Unmeasured::NoProjection) as
/// its reason, which is the truth about it: the patch is no longer in that
/// photograph.
pub fn translate_frame(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    observation: usize,
    pixel: [f64; 2],
) -> Result<(EditableTrack, TranslateFrameReport), TrackEditError> {
    if !pixel.iter().all(|c| c.is_finite()) {
        return Err(TrackEditError::BadPixel(pixel));
    }
    let current = observation_at(track, observation)?;
    let image = current.image;
    let site = current
        .site()
        .ok_or(TrackEditError::NoPlace { observation })?;
    let frame = frame_of(track)?;
    let was = frame.center;
    let (camera, cam_from_world) = view_of(edited, image)?;
    let anchored = frame
        .anchored_at_keypoint(&camera, &cam_from_world, site)
        .ok_or(TrackEditError::NoProjection { observation })?;
    let offset = anchored
        .keypoint_plane_offset(&camera, &cam_from_world, pixel)
        .ok_or(TrackEditError::NoProjection { observation })?;
    // The whole patch moves by this, which is what makes the gesture a
    // translation: the centre is carried by the drag and so is every sighting,
    // rather than the surfel being re-seated onto the one observation the
    // pointer came through.
    let displacement = offset;
    let mut center = frame.center + displacement;
    // A direction patch's centre is a unit bearing, which is what rendering and
    // the half-extents are stated against; the corner directions are unchanged
    // by the renormalization, so the patch keeps its size.
    if frame.w == 0.0 {
        let norm = center.coords.norm();
        if norm <= 1e-12 {
            return Err(TrackEditError::BadPixel(pixel));
        }
        center = Point3::from(center.coords / norm);
    }

    let mut next = track.clone();
    {
        let (position, frame, bitmap) = track_payload_mut(&mut next);
        frame.center = center;
        if frame.w != 0.0 {
            *position = Some(center);
        }
        *bitmap = None;
    }
    let placed = carry_keypoints(&mut next, edited, displacement);
    // Where the dragged sighting now sits, which is the pointer: its own plane
    // point plus the displacement is, by construction, the plane point under
    // the pixel.
    let landed = next.observations[observation].site().unwrap_or(pixel);
    Ok((
        next,
        TranslateFrameReport {
            observation,
            image,
            pixel: landed,
            center,
            moved: (center - was).norm(),
            placed,
            changed: center != was,
        },
    ))
}

/// What one turn of the surfel did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotateFrameReport {
    /// How far it turned, in degrees, positive about the outward normal.
    pub degrees: f64,
    /// Whether anything changed.
    pub changed: bool,
}

/// Turn the track's surfel by `angle_rad` about its own outward normal.
///
/// The axes are rotated as a pair, by a rotation whose axis **is** the frame's
/// normal, so both keep their lengths, the frame keeps its handedness and the
/// patch keeps the plane and the face it had: what changes is which way up the
/// square sits on that plane, which is what a person turning the outline in the
/// Image Detail panel is saying.
///
/// The centre is untouched, so a turn moves no sighting: a corner drag spins
/// the square in place while every keypoint stays where it is. The consensus
/// bitmap and the track measurements are dropped for the reason a resize drops
/// them -- both were read over the square as it stood, and the square has
/// turned under them.
pub fn rotate_frame(
    track: &EditableTrack,
    angle_rad: f64,
) -> Result<(EditableTrack, RotateFrameReport), TrackEditError> {
    if !angle_rad.is_finite() {
        return Err(TrackEditError::BadAngle(angle_rad));
    }
    frame_of(track)?;
    let mut next = track.clone();
    let changed = angle_rad != 0.0;
    if changed {
        keep_keypoints_only(&mut next);
        let (_, frame, bitmap) = track_payload_mut(&mut next);
        let axis = nalgebra::Unit::new_normalize(frame.normal());
        let rotation = nalgebra::Rotation3::from_axis_angle(&axis, angle_rad);
        frame.u_axis = rotation * frame.u_axis;
        frame.v_axis = rotation * frame.v_axis;
        *bitmap = None;
    }
    Ok((
        next,
        RotateFrameReport {
            degrees: angle_rad.to_degrees(),
            changed,
        },
    ))
}

/// What one hand-set affine shape did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeReport {
    /// The observation whose shape was set.
    pub observation: usize,
    /// The image it is in.
    pub image: u32,
    /// The shape it now carries.
    pub shape: [[f64; 2]; 2],
    /// The patch's half-width along its `u` axis in that image's own pixels at
    /// the cluster's radius, which is `radius * ||column 0||`: the size a
    /// person reads off the outline.
    pub half_px: f64,
    /// What that half-width was.
    pub was_half_px: f64,
    /// Whether anything changed.
    pub changed: bool,
}

/// Give one cluster-stage observation the affine shape `shape`, by hand.
///
/// The shape is the cluster stage's own convention -- the detector's canonical
/// keypoint frame mapped onto this image's pixels, read over the square
/// `[-r, r]^2` at the cluster's [`ClusterPayload::radius`]. What sets one by
/// hand is a corner drag, which turns the parallelogram about the sighting;
/// the edge drag, which scales it and moves it, is [`resize_from_edge`],
/// because holding the far edge still is arithmetic the caller should not have
/// to repeat.
///
/// The observation is re-seeded at the place it is already drawn and its
/// refinement is dropped: the ZNCC, the drift and the status were the
/// refinement's answer about the shape it was run at, and this is another
/// shape. The verdict is **not** pinned: a size or a turn is not a ruling on
/// whether the sighting belongs, which is what a pin protects from the
/// painting.
pub fn set_observation_shape(
    track: &EditableTrack,
    observation: usize,
    shape: [[f64; 2]; 2],
) -> Result<(EditableTrack, ShapeReport), TrackEditError> {
    let det = shape[0][0] * shape[1][1] - shape[0][1] * shape[1][0];
    if !shape.iter().flatten().all(|c| c.is_finite()) || det.abs() < MIN_ABS_DET {
        return Err(TrackEditError::BadShape(shape));
    }
    let radius = match &track.stage {
        Stage::Cluster(payload) => payload.radius,
        Stage::Track(_) => {
            return Err(TrackEditError::WrongStage {
                wanted: StageKind::Cluster,
                is: StageKind::Track,
            })
        }
    };
    let current = observation_at(track, observation)?;
    let image = current.image;
    let was = current.shape();
    let position = current
        .site()
        .ok_or(TrackEditError::NoPlace { observation })?;

    let mut next = track.clone();
    next.observations[observation].cluster = Some(ClusterMeasurement::from_seed(position, shape));
    Ok((
        next,
        ShapeReport {
            observation,
            image,
            shape,
            half_px: half_width_px(shape, radius),
            was_half_px: was.map_or(0.0, |was| half_width_px(was, radius)),
            changed: was != Some(shape),
        },
    ))
}

/// The patch's half-width along its `u` axis, in the image's own pixels: the
/// first column of `shape` is one keypoint-frame unit in pixels and the patch
/// is `radius` of them.
pub fn half_width_px(shape: [[f64; 2]; 2], radius: f64) -> f64 {
    radius * shape[0][0].hypot(shape[1][0])
}

/// The observation at `observation`, or the refusal a step owes an index past
/// the end of the list.
fn observation_at(
    track: &EditableTrack,
    observation: usize,
) -> Result<&Observation, TrackEditError> {
    track
        .observations
        .get(observation)
        .ok_or(TrackEditError::NoSuchObservation {
            observation,
            observation_count: track.observations.len(),
        })
}

/// The track-stage surfel, or the refusal a track-stage step owes a cluster or
/// a track nothing has triangulated.
fn frame_of(track: &EditableTrack) -> Result<&OrientedPatch, TrackEditError> {
    match &track.stage {
        Stage::Track(payload) => payload.frame.as_ref().ok_or(TrackEditError::NoFrame),
        Stage::Cluster(_) => Err(TrackEditError::WrongStage {
            wanted: StageKind::Track,
            is: StageKind::Cluster,
        }),
    }
}

/// The position, the surfel and the bitmap of a track whose payload has already
/// been read as a track stage carrying a frame.
fn track_payload_mut(
    track: &mut EditableTrack,
) -> (
    &mut Option<nalgebra::Point3<f64>>,
    &mut OrientedPatch,
    &mut Option<ndarray::Array3<u8>>,
) {
    let Stage::Track(payload) = &mut track.stage else {
        unreachable!("the stage was read as a track stage before the clone");
    };
    let TrackPayload {
        position,
        frame,
        bitmap,
        ..
    } = payload;
    (
        position,
        frame.as_mut().expect("the frame was read before the clone"),
        bitmap,
    )
}

/// Carry every sighting along the plane by `displacement`, and drop every
/// measurement read before the patch moved.
///
/// The rule the two steps that move the centre share, and **not** a
/// reprojection of the centre: a keypoint is where that photograph sees the
/// patch's *content*, and the gap between it and the centre's projection is
/// that observation's own in-plane offset -- the thing the tile is cut on and
/// the correlation is scored at. Resetting every keypoint to the centre's
/// projection would throw all of those away and scramble the correlation, so
/// what moves is the **place**: each sighting's own plane point is found by
/// re-anchoring the patch on it (`OrientedPatch::anchored_at_keypoint`, its own
/// camera and pose), the displacement is added to that point, and the result is
/// projected back. Every offset therefore survives the move exactly, and the
/// sighting the drag came through lands under the pointer, because its plane
/// point plus the displacement *is* the plane point under the pixel.
///
/// A sighting that has never been localized has no keypoint to carry, so it
/// takes the projection of the new centre -- the only place the patch says it
/// could be. One that no longer projects at all is left with no keypoint and
/// the reason that says so, rather than with a stale one.
///
/// Returns how many keypoints were written, which is how many photographs still
/// hold the patch.
fn carry_keypoints(
    track: &mut EditableTrack,
    edited: &EditedReconstruction,
    displacement: Vector3<f64>,
) -> usize {
    let Some(frame) = track.track().and_then(|payload| payload.frame.clone()) else {
        return 0;
    };
    let mut placed = 0;
    for observation in &mut track.observations {
        let was = observation.track.as_ref().and_then(|m| m.keypoint);
        let landed = view_of(edited, observation.image).ok().and_then(
            |(camera, cam_from_world)| match was {
                // Its own plane point, carried by the drag: the offset between
                // the feature and the centre's projection is preserved.
                Some(keypoint) => {
                    let at = [f64::from(keypoint[0]), f64::from(keypoint[1])];
                    let anchored = frame.anchored_at_keypoint(&camera, &cam_from_world, at)?;
                    project_center(
                        &camera,
                        &cam_from_world,
                        anchored.center + displacement,
                        frame.w,
                    )
                }
                // Never localized, so there is no offset to keep.
                None => project_center(&camera, &cam_from_world, frame.center, frame.w),
            },
        );
        observation.track = Some(match landed {
            Some(pixel) => {
                placed += 1;
                TrackMeasurement {
                    keypoint: Some([pixel[0] as f32, pixel[1] as f32]),
                    ..TrackMeasurement::default()
                }
            }
            None => TrackMeasurement {
                reason: Some(super::track::Unmeasured::NoProjection),
                ..TrackMeasurement::default()
            },
        });
    }
    placed
}

/// Drop every track measurement but the keypoint it was read at.
///
/// What a change to the surfel invalidates, stated once: the ZNCC, both
/// distances, the reprojection residual, the localizability and the reason were
/// all read over the square as it stood and against the position it stood at,
/// and neither holds after the patch has been resized, moved or turned. Where
/// each sighting sits is not one of those things, so it stays.
fn keep_keypoints_only(track: &mut EditableTrack) {
    for observation in &mut track.observations {
        if let Some(measurement) = &observation.track {
            observation.track = Some(TrackMeasurement {
                keypoint: measurement.keypoint,
                ..TrackMeasurement::default()
            });
        }
    }
}

/// The camera and the pose of one image of the reconstruction.
fn view_of(
    edited: &EditedReconstruction,
    image: u32,
) -> Result<
    (
        crate::camera::CameraIntrinsics,
        crate::geometry::RigidTransform,
    ),
    TrackEditError,
> {
    let table = &edited.base.image_table;
    let row = table
        .images
        .get(image as usize)
        .ok_or(TrackEditError::NoSuchImage { image })?;
    let camera = table
        .cameras
        .get(row.camera_index as usize)
        .ok_or(TrackEditError::NoSuchImage { image })?;
    let q = row.quaternion_wxyz.quaternion();
    let pose = crate::geometry::RigidTransform::from_wxyz_translation(
        [q.w, q.i, q.j, q.k],
        [
            row.translation_xyz.x,
            row.translation_xyz.y,
            row.translation_xyz.z,
        ],
    );
    Ok((camera.clone(), pose))
}

/// Where a patch's centre lands in a view, as a pixel.
fn project_center(
    camera: &crate::camera::CameraIntrinsics,
    cam_from_world: &crate::geometry::RigidTransform,
    center: nalgebra::Point3<f64>,
    w: f64,
) -> Option<[f64; 2]> {
    let pc = cam_from_world.transform_point_homogeneous(center.coords, w);
    if !camera.model.needs_ray_path() && pc.z >= 0.0 {
        return None;
    }
    camera.ray_to_pixel([pc.x, pc.y, pc.z]).map(|(u, v)| [u, v])
}

/// What one painting did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThresholdReport {
    /// How many observations the painting turned `in`.
    pub turned_in: usize,
    /// How many it turned `out`.
    pub turned_out: usize,
    /// How many it left alone because their verdict was pinned.
    pub pinned: usize,
    /// How many it left alone because nothing at this stage has measured them.
    pub unmeasured: usize,
}

/// Paint the proposed verdicts from the stored measurements onto the
/// unpinned observations.
///
/// The thresholds propose and the person decides, so this is the step that
/// turns a proposal into verdicts, and it turns only the ones nobody has ruled
/// on by hand. An observation nothing has measured at the track's current stage
/// is left where it is: there is no proposal to apply.
///
/// One `in` per image survives the painting. Where several unpinned
/// observations of one image would pass, the one with the best score takes the
/// `in` and the rest stay candidates, so the painting can never produce a track
/// that observes an image twice.
pub fn apply_thresholds(track: &EditableTrack) -> (EditableTrack, ThresholdReport) {
    let stage = track.stage_kind();
    let mut report = ThresholdReport {
        turned_in: 0,
        turned_out: 0,
        pinned: 0,
        unmeasured: 0,
    };
    let mut next = track.clone();
    // The images an `in` observation this painting cannot move already holds.
    let mut held: Vec<u32> = track
        .observations
        .iter()
        .filter(|o| o.pinned && o.verdict == Verdict::In)
        .map(|o| o.image)
        .collect();

    // Best score first, so the observation that takes an image's one `in` slot
    // is the one that registered best rather than the one that was added first.
    let mut order: Vec<usize> = (0..track.observations.len()).collect();
    order.sort_by(|&a, &b| {
        score(&track.observations[b], stage)
            .partial_cmp(&score(&track.observations[a], stage))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in order {
        let observation = &track.observations[i];
        if observation.pinned {
            report.pinned += 1;
            continue;
        }
        match proposed_verdict(observation, stage, &track.thresholds) {
            None => report.unmeasured += 1,
            Some(Verdict::Out) => {
                if observation.verdict != Verdict::Out {
                    report.turned_out += 1;
                }
                next.observations[i].verdict = Verdict::Out;
            }
            Some(Verdict::In) => {
                if held.contains(&observation.image) {
                    // The image is spoken for. The observation is not refused,
                    // only not taken: the person turns the other one out first.
                    next.observations[i].verdict = Verdict::Candidate;
                } else {
                    held.push(observation.image);
                    if observation.verdict != Verdict::In {
                        report.turned_in += 1;
                    }
                    next.observations[i].verdict = Verdict::In;
                }
            }
            Some(Verdict::Candidate) => {}
        }
    }
    (next, report)
}

/// The ZNCC the painting ranks an observation by at `stage`, or negative
/// infinity when there is none, so an unmeasured observation sorts last.
fn score(observation: &Observation, stage: StageKind) -> f64 {
    let zncc = match stage {
        StageKind::Cluster => observation.cluster.as_ref().and_then(|m| m.zncc),
        StageKind::Track => observation.track.as_ref().and_then(|m| m.zncc),
    };
    zncc.unwrap_or(f64::NEG_INFINITY)
}

/// What the thresholds propose for one observation at `stage`, or `None` when
/// nothing at that stage has measured it.
///
/// A `NaN` score is one no round ever produced and clears no bar, so it
/// proposes `out` rather than reading as unmeasured: the difference matters,
/// because an observation that was measured and failed is a refusal the person
/// should see.
///
/// The distance the `max_shift_px` bar is judged on is the observation's **own**
/// evidence at either stage -- the drift from its seed at the cluster stage, and
/// [`TrackMeasurement::seed_shift_px`](super::track::TrackMeasurement::seed_shift_px),
/// how far the correlation peak sits from where the sighting is, at the track
/// stage. The other track-stage distance,
/// [`projection_offset_px`](super::track::TrackMeasurement::projection_offset_px),
/// is a verdict on the point: judging sightings by it would turn out the very
/// observations that would move a mis-triangulated point back.
fn proposed_verdict(
    observation: &Observation,
    stage: StageKind,
    thresholds: &Thresholds,
) -> Option<Verdict> {
    let (zncc, shift, localizability) = match stage {
        StageKind::Cluster => {
            let m = observation.cluster.as_ref()?;
            (m.zncc?, m.shift_px, m.localizability)
        }
        StageKind::Track => {
            let m = observation.track.as_ref()?;
            (m.zncc?, m.seed_shift_px, m.localizability)
        }
    };
    let passes = !zncc.is_nan()
        && zncc >= thresholds.min_zncc
        && shift.is_none_or(|s| !s.is_nan() && s <= thresholds.max_shift_px)
        && localizability.is_none_or(|s| !s.is_nan() && s <= thresholds.max_keypoint_uncertainty);
    Some(if passes { Verdict::In } else { Verdict::Out })
}

// ---- Splitting one track into two ------------------------------------------

/// Why a duplicate was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DuplicateError {
    /// Nothing on the bench carries that label.
    NoSuchTrack(String),
}

impl std::fmt::Display for DuplicateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DuplicateError::NoSuchTrack(label) => {
                write!(f, "nothing on the bench is called `{label}`")
            }
        }
    }
}

impl std::error::Error for DuplicateError {}

/// What one duplicate did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateReport {
    /// The label the copy took.
    pub label: String,
    /// The label it was copied from.
    pub from: String,
    /// How many observations it carries, which is the original's count.
    pub observation_count: usize,
}

/// Put a copy of the track called `label` on the bench beside it.
///
/// **What a second patch over the same ground is started from.** A patch that
/// has been slid, turned and sized until it covers one piece of surface is most
/// of the work of covering the piece next to it, so the copy carries everything
/// that describes the geometry and the judgements made about it: the stage and
/// all of its data (the surfel, the consensus bitmap, the cluster's template and
/// radius), every observation with its keypoint, its seed, its shape, its
/// verdict and its pin, and the thresholds. The measurements come too -- they
/// were read against this geometry and still describe it, and the moment the
/// copy is moved the steps that move it drop the ones that no longer hold.
///
/// **The copy has no origin.** An origin is what makes a commit *replace* a
/// point, and a copy is a new patch over new ground: it has to create one, or
/// the second commit would delete what the first wrote. That is the one field
/// the copy does not carry, and it is the whole of the difference between the
/// two items.
///
/// The label is minted from `<label> copy` through the bench's own collision
/// rule ([`Bench::mint_label`]), so a second duplicate of the same track is
/// `<label> copy (2)`, and the copy becomes the active track, because it is the
/// thing the person is about to work on.
pub fn duplicate(bench: &Bench, label: &str) -> Result<(Bench, DuplicateReport), DuplicateError> {
    let track = bench
        .track(label)
        .ok_or_else(|| DuplicateError::NoSuchTrack(label.to_string()))?;
    let mut copy = (**track).clone();
    copy.origin = None;
    let observation_count = copy.observations.len();
    let (bench, new_label) = bench.put(&format!("{label} copy"), BenchItem::Track(Arc::new(copy)));
    Ok((
        bench,
        DuplicateReport {
            label: new_label,
            from: label.to_string(),
            observation_count,
        },
    ))
}

/// Why a split was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    /// Nothing on the bench carries the label the split named.
    NoSuchTrack(String),
    /// No observation was named. A split of nothing is not a split.
    NoObservations,
    /// Every observation was named. A split that leaves nothing behind is a
    /// rename, and the person has one of those.
    EveryObservation(usize),
    /// An observation index is past the end of the track.
    NoSuchObservation {
        /// The index named.
        observation: usize,
        /// How many observations the track holds.
        observation_count: usize,
    },
    /// The split-off half is a cluster, and putting the track-stage half down
    /// to that stage was refused.
    Downgrade(StageError),
}

impl std::fmt::Display for SplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SplitError::NoSuchTrack(label) => {
                write!(f, "no track on the bench is called `{label}`")
            }
            SplitError::NoObservations => {
                write!(f, "splitting off no observations would make an empty track")
            }
            SplitError::EveryObservation(n) => write!(
                f,
                "splitting off all {n} observations would leave an empty track"
            ),
            SplitError::NoSuchObservation {
                observation,
                observation_count,
            } => write!(
                f,
                "observation {observation} is past the {observation_count} \
                 observations of this track"
            ),
            SplitError::Downgrade(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SplitError {}

/// What one split did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitReport {
    /// The label the second track took.
    pub label: String,
    /// How many observations moved to it.
    pub moved: usize,
    /// How many stayed on the first.
    pub kept: usize,
}

/// Split the observations at `observations` off the track called `label` into a
/// second track beside it on the bench.
///
/// The observations are named explicitly rather than read off the verdicts,
/// because `out` says an observation does not belong *here*: an observation the
/// localizer refused and one that belongs to the track next door are both
/// `out`, and a verdict cannot say which of them the new track should take. The
/// moved observations keep their verdicts, their provenance and both stages'
/// measurements; the new track has no origin, so a commit of it creates a point
/// while a commit of the first still replaces the one it came from.
///
/// **The second track is a cluster.** A split is the step for a track that is
/// two surfaces, and the half being taken off is a set of sightings that agree
/// with each other and not with a 3D hypothesis fitted to both; carrying that
/// hypothesis onto it would state as fact the thing the split is questioning.
/// So a track-stage half is put down to the cluster stage through the same
/// downgrade the stage step runs ([`set_stage`](super::stage::set_stage())),
/// which needs the reconstruction for the cameras it projects the frame
/// through, and a half whose frame projects nowhere is refused there rather
/// than half-moved. The first track keeps its stage, its origin and everything
/// it was.
///
/// A cluster whose reference moved out takes the first observation it has left
/// as its reference, and the new track takes the one the downgrade picks -- the
/// observation the patch is largest in, whatever its verdict -- or its own
/// first when it was already a cluster; the template is dropped on both,
/// because a template is a cut around a particular reference.
///
/// **A half whose every row is `out` still splits.** The rows the thresholds
/// rejected are exactly the ones a person cuts off to look at on their own, so
/// the downgrade's reference falls back past the verdicts to the whole seeded
/// set; what it cannot do without is a seed, and a half carrying none is
/// refused.
pub fn split(
    bench: &Bench,
    edited: &EditedReconstruction,
    label: &str,
    observations: &[usize],
) -> Result<(Bench, SplitReport), SplitError> {
    let track = bench
        .track(label)
        .ok_or_else(|| SplitError::NoSuchTrack(label.to_string()))?;
    let count = track.observations.len();
    let mut taken: Vec<usize> = observations.to_vec();
    taken.sort_unstable();
    taken.dedup();
    if taken.is_empty() {
        return Err(SplitError::NoObservations);
    }
    if let Some(&past) = taken.iter().find(|&&i| i >= count) {
        return Err(SplitError::NoSuchObservation {
            observation: past,
            observation_count: count,
        });
    }
    if taken.len() == count {
        return Err(SplitError::EveryObservation(count));
    }

    let mut moved = Vec::with_capacity(taken.len());
    let mut kept = Vec::with_capacity(count - taken.len());
    for (i, observation) in track.observations.iter().enumerate() {
        if taken.binary_search(&i).is_ok() {
            moved.push(observation.clone());
        } else {
            kept.push(observation.clone());
        }
    }

    let mut first = (**track).clone();
    first.observations = kept;
    reseat_reference(&mut first);
    let mut second = (**track).clone();
    second.observations = moved;
    second.origin = None;
    reseat_reference(&mut second);
    if second.stage_kind() == StageKind::Track {
        let (down, _) = set_stage(
            &second,
            edited,
            &[],
            StageKind::Cluster,
            &FitOptions::default(),
            &Progress::none(),
        )
        .map_err(SplitError::Downgrade)?;
        second = down;
    }

    let report_moved = second.observations.len();
    let report_kept = first.observations.len();
    let bench = bench
        .replace(label, BenchItem::Track(Arc::new(first)))
        .expect("the label was just read off this bench");
    let (bench, new_label) = bench.put(
        &format!("{label}-split"),
        BenchItem::Track(Arc::new(second)),
    );
    Ok((
        bench,
        SplitReport {
            label: new_label,
            moved: report_moved,
            kept: report_kept,
        },
    ))
}

/// Point a cluster's reference at an observation it still has, and drop the
/// template, whose cut was around whatever the reference used to be.
fn reseat_reference(track: &mut EditableTrack) {
    if let Stage::Cluster(payload) = &mut track.stage {
        payload.template = None;
        if payload.reference >= track.observations.len() {
            payload.reference = 0;
        }
    }
}
