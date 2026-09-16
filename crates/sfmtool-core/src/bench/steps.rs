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

use nalgebra::Vector3;

use crate::patch::cloud::OrientedPatch;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::fit::FitOptions;
use super::stage::{set_stage, StageError};
use super::track::{
    ClusterPayload, EditableTrack, Observation, Origin, Provenance, Stage, StageKind, Thresholds,
    TrackMeasurement, TrackPayload, Verdict,
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
/// observation the patch is largest in -- or its own first when it was already
/// a cluster; the template is dropped on both, because a template is a cut
/// around a particular reference.
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
