// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Build a track at a pixel: given an image and a pixel in it, find the other
//! photographs' sightings of that spot, fit a track-stage patch to them that
//! stays centred on the pixel, and either return it or say which step refused.
//!
//! `specs/core/bench/track-at-pixel.md` is the design. The operation is a
//! **cascade** of four ways of finding the sightings, tried in order until one
//! returns a track that passes its own gates:
//!
//! 1. [`CascadeMember::Clusters`]: the cluster-patches `.matches` clusters with
//!    a member near the pixel, the pixel carried into each kept member's image
//!    through the two members' affine shapes.
//! 2. [`CascadeMember::Transfer`]: the reconstruction's points around the pixel,
//!    whose own matched keypoints fix a local affine map into each other image.
//! 3. [`CascadeMember::Sweep`]: a plane through the neighbours' points, met by
//!    the pixel's ray and projected into the views that face it.
//! 4. [`CascadeMember::Constellation`]: the SIFT index's constellation query from
//!    the pixel, then the cluster stage and the upgrade.
//!
//! Every member ends in the same finish: an anchored fit, a tilt toward the
//! neighbours' normal, the geometry search, cleaning and the gates.
//!
//! Everything here composes the bench steps in [`super`]; nothing writes the
//! reconstruction, and the track that comes back is a bench item like any
//! other.

mod finish;
mod members;
mod neighbourhood;

#[cfg(test)]
mod tests;

use nalgebra::Vector3;

use crate::features::kdforest::{ImageKeypoints, LazyKdForestU8};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::classify::ClassificationReason;
use super::track::EditableTrack;

pub use neighbourhood::{
    ClusterMember, MatchesClusters, MatchesClustersError, NearbyCluster, NearbyObservation,
};

use neighbourhood::{ObservationIndex, ViewCamera};

/// One way of finding the pixel's sightings in the other photographs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeMember {
    /// The cluster-patches `.matches` clusters near the pixel.
    Clusters,
    /// The neighbouring points' own matched keypoints.
    Transfer,
    /// A plane through the neighbouring points, projected into the views.
    Sweep,
    /// The SIFT index's constellation query from the pixel.
    Constellation,
}

impl CascadeMember {
    /// The member's name, as a report and the Python binding spell it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Clusters => "clusters",
            Self::Transfer => "transfer",
            Self::Sweep => "sweep",
            Self::Constellation => "constellation",
        }
    }
}

impl std::fmt::Display for CascadeMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for CascadeMember {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "clusters" => Ok(Self::Clusters),
            "transfer" => Ok(Self::Transfer),
            "sweep" => Ok(Self::Sweep),
            "constellation" => Ok(Self::Constellation),
            other => Err(format!(
                "unknown cascade member {other:?} (expected clusters|transfer|sweep|constellation)"
            )),
        }
    }
}

/// The step at which a member refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalStage {
    /// The clusters member found no cluster near the pixel, or none that
    /// carried it into another photograph.
    Clusters,
    /// The transfer member found too few points of one surface near the pixel.
    Neighbourhood,
    /// The transfer member's neighbourhoods carried the pixel into no other
    /// photograph.
    Transfer,
    /// The sweep member found no point near the pixel to say what surface it is
    /// on.
    Prior,
    /// The sweep member's surface hypotheses could not be fitted.
    Hypothesis,
    /// The constellation query matched no other image.
    Constellation,
    /// The cluster-stage reading refused, or kept no candidate.
    ClusterEvaluate,
    /// The upgrade to the track stage refused.
    Upgrade,
    /// The first anchored fit of the finish refused.
    Anchor,
    /// The finished track failed one of the final gates.
    Gate,
}

impl RefusalStage {
    /// The stage's name, as a report and the Python binding spell it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Clusters => "clusters",
            Self::Neighbourhood => "neighbourhood",
            Self::Transfer => "transfer",
            Self::Prior => "prior",
            Self::Hypothesis => "hypothesis",
            Self::Constellation => "constellation",
            Self::ClusterEvaluate => "cluster evaluate",
            Self::Upgrade => "upgrade",
            Self::Anchor => "anchor",
            Self::Gate => "gate",
        }
    }
}

impl std::fmt::Display for RefusalStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The finish every member ends in: anchoring, the neighbours' normal, the
/// geometry search, cleaning and the gates.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishOptions {
    /// Fit-then-anchor rounds after each anchoring.
    pub anchor_refits: usize,
    /// Tilt toward the neighbours' normal, keeping the tilt unless the median
    /// ZNCC falls by more than [`Self::normal_prior_tolerance`].
    pub normal_prior: bool,
    /// How far from the pixel the neighbours whose normals are averaged may be,
    /// in px.
    pub normal_prior_radius_px: f64,
    /// How many of the nearest neighbours are averaged.
    pub normal_prior_k: usize,
    /// How far the median ZNCC may fall under the tilt and still keep it.
    pub normal_prior_tolerance: f64,
    /// Grow the track by the geometry search from the queried sighting.
    pub geometry_search: bool,
    /// Turn out the views that disagree with the track's geometry, and refit.
    pub clean: bool,
    /// A view whose correlation peak sits further than this from its keypoint
    /// is turned out by the cleaning, in px.
    pub clean_max_shift_px: f64,
    /// A view whose keypoint sits further than this from the point's
    /// projection is turned out by the cleaning, in px.
    pub clean_max_projection_px: f64,
    /// How many rounds of cleaning run at most.
    pub clean_rounds: usize,
    /// Gate: the fewest `in` views.
    pub min_in_views: usize,
    /// Gate: the lowest median leave-one-out ZNCC over the `in` views.
    pub min_zncc_median: f64,
    /// Gate: how far the queried sighting's keypoint may sit from the pixel, in
    /// px.
    pub max_query_offset_px: f64,
    /// Gate: how far any `in` view's keypoint may sit from the point's
    /// projection, in px.
    pub max_projection_offset_px: f64,
}

impl Default for FinishOptions {
    fn default() -> Self {
        Self {
            anchor_refits: 1,
            normal_prior: true,
            normal_prior_radius_px: 40.0,
            normal_prior_k: 10,
            normal_prior_tolerance: 0.01,
            geometry_search: true,
            clean: true,
            clean_max_shift_px: 1.5,
            clean_max_projection_px: 1.5,
            clean_rounds: 2,
            min_in_views: 3,
            min_zncc_median: 0.8,
            max_query_offset_px: 2.0,
            max_projection_offset_px: 1.5,
        }
    }
}

/// How the clusters member reads the `.matches` clusters.
#[derive(Debug, Clone, PartialEq)]
pub struct ClustersOptions {
    /// How far from the pixel a cluster's member in the queried image may be,
    /// in px.
    pub search_radius_px: f64,
    /// How far the pixel may be from that member, in units of the member's own
    /// scale (the square root of its shape's determinant).
    pub max_offset_in_scales: f64,
    /// How many of the nearest clusters are tried.
    pub max_clusters: usize,
    /// The patch half-width, in member scales.
    pub radius_in_scales: f64,
    /// The smallest patch half-width, in px.
    pub min_radius_px: f64,
    /// The largest patch half-width, in px.
    pub max_radius_px: f64,
}

impl Default for ClustersOptions {
    fn default() -> Self {
        Self {
            search_radius_px: 16.0,
            max_offset_in_scales: 15.0,
            max_clusters: 3,
            radius_in_scales: 2.0,
            min_radius_px: 4.0,
            max_radius_px: 40.0,
        }
    }
}

/// How the transfer member fits its local affine maps.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferOptions {
    /// How far from the pixel the neighbours may be, in px.
    pub neighbour_radius_px: f64,
    /// How many of the nearest neighbours of one depth mode are used.
    pub max_neighbours: usize,
    /// The depth ratio between consecutive neighbours that separates two
    /// surfaces.
    pub depth_mode_gap: f64,
    /// The fewest keypoint pairs an affine map is fitted to.
    pub min_pairs: usize,
    /// A pair the map misses by more than this is dropped and the map refitted,
    /// in px.
    pub max_residual_px: f64,
    /// The patch half-width when no neighbour states one, in px.
    pub default_radius_px: f64,
    /// The smallest patch half-width, in px.
    pub min_radius_px: f64,
    /// The largest patch half-width, in px.
    pub max_radius_px: f64,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            neighbour_radius_px: 60.0,
            max_neighbours: 16,
            depth_mode_gap: 1.15,
            min_pairs: 4,
            max_residual_px: 3.0,
            default_radius_px: 8.0,
            min_radius_px: 4.0,
            max_radius_px: 40.0,
        }
    }
}

/// How the sweep member builds its surface hypotheses.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepOptions {
    /// How far from the pixel the neighbours may be, in px.
    pub prior_radius_px: f64,
    /// How many of the nearest neighbours of one depth mode are used.
    pub prior_k: usize,
    /// The patch half-width when no neighbour states one, in px.
    pub default_radius_px: f64,
    /// The smallest patch half-width, in px.
    pub min_radius_px: f64,
    /// The largest patch half-width, in px.
    pub max_radius_px: f64,
    /// The depth ratio between consecutive neighbours that separates two
    /// surfaces.
    pub depth_mode_gap: f64,
    /// A depth mode smaller than this is skipped when there are others.
    pub min_mode_size: usize,
    /// How many of the most nearly face-on views seed a hypothesis.
    pub seed_views: usize,
    /// The largest angle between the hypothesis's normal and the direction to
    /// a camera for that view to seed it, in degrees.
    pub max_view_angle_deg: f64,
}

impl Default for SweepOptions {
    fn default() -> Self {
        Self {
            prior_radius_px: 40.0,
            prior_k: 10,
            default_radius_px: 8.0,
            min_radius_px: 4.0,
            max_radius_px: 40.0,
            depth_mode_gap: 1.15,
            min_mode_size: 2,
            seed_views: 5,
            max_view_angle_deg: 70.0,
        }
    }
}

/// How the constellation member searches the SIFT index.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstellationOptions {
    /// How far from the pixel the neighbours of the local prior may be, in px.
    pub prior_radius_px: f64,
    /// How many of the nearest neighbours of the pixel's depth mode set the
    /// prior's size and normal.
    pub prior_k: usize,
    /// The patch half-width when no neighbour states one, in px.
    pub default_radius_px: f64,
    /// The smallest patch half-width, in px.
    pub min_radius_px: f64,
    /// The largest patch half-width, in px.
    pub max_radius_px: f64,
    /// The depth ratio between consecutive neighbours that separates two
    /// surfaces.
    pub depth_mode_gap: f64,
    /// How many keypoints the constellation should hold; the search radius is
    /// the one a uniform keypoint density puts this many inside.
    pub constellation_target: usize,
    /// The fewest agreeing correspondences an image needs.
    pub min_inliers: usize,
    /// How many further searches start from the best images the first found.
    pub lateral_searches: usize,
    /// Tilt toward the local prior's normal after the upgrade, keeping the
    /// better reading.
    pub normal_prior: bool,
}

impl Default for ConstellationOptions {
    fn default() -> Self {
        Self {
            prior_radius_px: 40.0,
            prior_k: 8,
            default_radius_px: 8.0,
            min_radius_px: 4.0,
            max_radius_px: 40.0,
            depth_mode_gap: 1.15,
            constellation_target: 50,
            min_inliers: 6,
            lateral_searches: 2,
            normal_prior: true,
        }
    }
}

/// What a track-at-pixel query runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackAtPixelOptions {
    /// The members, in the order they are tried.
    pub members: Vec<CascadeMember>,
    /// The finish every member ends in.
    pub finish: FinishOptions,
    /// The clusters member's own options.
    pub clusters: ClustersOptions,
    /// The transfer member's own options.
    pub transfer: TransferOptions,
    /// The sweep member's own options.
    pub sweep: SweepOptions,
    /// The constellation member's own options.
    pub constellation: ConstellationOptions,
}

impl Default for TrackAtPixelOptions {
    fn default() -> Self {
        Self {
            members: vec![
                CascadeMember::Clusters,
                CascadeMember::Transfer,
                CascadeMember::Sweep,
                CascadeMember::Constellation,
            ],
            finish: FinishOptions::default(),
            clusters: ClustersOptions::default(),
            transfer: TransferOptions::default(),
            sweep: SweepOptions::default(),
            constellation: ConstellationOptions::default(),
        }
    }
}

/// The evidence a query reads beside the reconstruction and the photographs.
///
/// Both are built once per capture and shared by every query, because neither
/// holds anything of the reconstruction's points. Either may be left out: the
/// member that reads it then refuses, naming what was missing, and the others
/// run as usual.
#[derive(Clone, Copy, Default)]
pub struct TrackAtPixelSources<'a> {
    /// The SIFT index and the keypoints it is queried with, which the
    /// constellation member reads.
    pub sift_index: Option<SiftIndexSource<'a>>,
    /// The cluster-patches `.matches` clusters, indexed onto the
    /// reconstruction's images, which the clusters member reads.
    pub clusters: Option<&'a MatchesClusters>,
}

/// The SIFT index and every image's keypoints, as the constellation member
/// queries them.
#[derive(Clone, Copy)]
pub struct SiftIndexSource<'a> {
    /// The SIFT index, whose corpus indexes the reconstruction's images in the
    /// reconstruction's order.
    pub forest: &'a LazyKdForestU8,
    /// Every image's `.sift` keypoints, one entry per image of the
    /// reconstruction, in its order.
    pub keypoints: &'a [ImageKeypoints],
}

/// One candidate a member tried: a cluster, a depth mode or a surface
/// hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRecord {
    /// What was tried.
    pub candidate: CandidateKind,
    /// How many other photographs it placed the pixel in.
    pub sightings: usize,
    /// The fitted track's score (`in` views times the median ZNCC, floored at
    /// zero), when it was fitted.
    pub score: Option<f64>,
    /// Why it was not fitted, when it was not.
    pub error: Option<String>,
}

/// What kind of candidate a [`CandidateRecord`] is.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateKind {
    /// A `.matches` cluster, with its member's distance from the pixel.
    Cluster {
        /// The cluster's index in the file.
        cluster: u32,
        /// Its member's distance from the pixel, in px.
        distance_px: f64,
    },
    /// A depth mode of the neighbours, and the images its affine maps reached.
    DepthMode {
        /// How many neighbours the mode holds.
        support: usize,
        /// The images an affine map was fitted for, ascending.
        images: Vec<u32>,
    },
    /// A surface hypothesis, and the views that seeded it.
    Hypothesis {
        /// The seeding views' images, most nearly face-on first.
        views: Vec<u32>,
    },
}

/// A surface hypothesis of the sweep member.
#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisRecord {
    /// How many neighbours its depth mode holds.
    pub support: usize,
    /// The patch half-width it proposes, in px of the queried image.
    pub half_px: f64,
    /// The distance of its nearest neighbour from the pixel, in px.
    pub nearest_px: f64,
}

/// The constellation member's local prior.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalPriorRecord {
    /// How many observations lie within the prior's radius.
    pub neighbours: usize,
    /// The depth modes of the finite neighbours, as `(median depth, count)`,
    /// nearest first.
    pub depth_modes: Vec<(f64, usize)>,
    /// Whether two of the modes hold two or more neighbours each.
    pub edge: bool,
    /// The median depth of the pixel's mode.
    pub depth: Option<f64>,
    /// The median apparent half-width of the pixel's mode, in px.
    pub half_px: Option<f64>,
    /// The distance-weighted mean normal of the pixel's mode.
    pub normal: Option<Vector3<f64>>,
    /// The mean angle of the mode's normals from that mean, in degrees.
    pub normal_spread_deg: Option<f64>,
}

/// One lateral search of the constellation member.
#[derive(Debug, Clone, PartialEq)]
pub struct LateralRecord {
    /// The image it searched from.
    pub from: u32,
    /// How many images it added, or why it was refused.
    pub added: Result<usize, String>,
}

/// A tilt toward a prior normal and what it did to the reading.
#[derive(Debug, Clone, PartialEq)]
pub struct TiltRecord {
    /// How far the patch turned, in degrees, when the tilt step reports it.
    pub degrees: Option<f64>,
    /// Whether an observation's viewing-angle cap stopped the turn short.
    pub stopped: bool,
    /// The median ZNCC before.
    pub zncc_before: f64,
    /// The median ZNCC after.
    pub zncc_after: f64,
    /// Whether the tilted track was kept.
    pub kept: bool,
}

/// What one step of a query did and measured, in the order the steps ran.
#[derive(Debug, Clone, PartialEq)]
pub enum StageRecord {
    /// The clusters member: how many clusters have a member near the pixel.
    NearbyClusters {
        /// That count.
        count: usize,
    },
    /// The transfer member: the sizes of the depth modes large enough to fit.
    DepthModes {
        /// One size per mode, nearest mode first.
        sizes: Vec<usize>,
    },
    /// The sweep member's surface hypotheses.
    Hypotheses {
        /// One per depth mode that gave one, nearest first.
        hypotheses: Vec<HypothesisRecord>,
    },
    /// The candidates a member tried, each with its score or the reason it was
    /// not fitted.
    Candidates {
        /// One per candidate, in the order tried.
        tried: Vec<CandidateRecord>,
    },
    /// The constellation member's local prior and the patch radius it set.
    LocalPrior {
        /// The prior.
        prior: LocalPriorRecord,
        /// The cluster's half-width, in px.
        radius_px: f64,
    },
    /// The constellation query from the pixel.
    Constellation {
        /// Its radius, in px.
        search_radius_px: f64,
        /// How many keypoints it held.
        keypoints: usize,
        /// Every image it matched, with the inliers that voted for it.
        images: Vec<(u32, usize)>,
    },
    /// The lateral searches from the best images the first search found.
    Lateral {
        /// One per search.
        searches: Vec<LateralRecord>,
    },
    /// The cluster-stage reading and its verdicts.
    ClusterEvaluate {
        /// How many observations the cluster holds.
        observations: usize,
        /// How many of them are `in`.
        in_views: usize,
    },
    /// The upgrade to the track stage.
    Upgrade {
        /// Whether the rays made a bearing.
        at_infinity: bool,
        /// Why the classification came out as it did.
        reason: Option<ClassificationReason>,
        /// The median ZNCC after it.
        zncc_median: f64,
    },
    /// The constellation member's tilt toward its local prior's normal.
    PriorTilt(Result<TiltRecord, String>),
    /// The finish's first anchored fit.
    Anchor {
        /// How many views are `in` after it.
        in_views: usize,
        /// The median ZNCC after it.
        zncc_median: f64,
    },
    /// The finish's tilt toward the neighbours' normal.
    NormalPrior(Result<TiltRecord, String>),
    /// The finish's geometry search.
    GeometrySearch(Result<(usize, f64), String>),
    /// The images the cleaning turned out.
    Clean {
        /// Their images, in the order they were turned out.
        removed_images: Vec<u32>,
    },
    /// The track the gates judged.
    Final {
        /// How many views are `in`.
        in_views: usize,
        /// The median ZNCC over them.
        zncc_median: f64,
        /// How far the queried sighting's keypoint sits from the pixel, in px.
        query_offset_px: Option<f64>,
        /// The largest projection offset over the `in` views, in px, once the
        /// earlier gates have passed.
        max_projection_offset_px: Option<f64>,
    },
}

/// A member's refusal: the stage, the sentence and what it measured.
#[derive(Debug, Clone, PartialEq)]
pub struct MemberRefusal {
    /// The member that refused.
    pub member: CascadeMember,
    /// The step it refused at.
    pub stage: RefusalStage,
    /// The sentence a person is shown.
    pub reason: String,
    /// What it did and measured before it refused.
    pub stages: Vec<StageRecord>,
}

/// How a built track was built.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackAtPixelReport {
    /// The member whose track was returned.
    pub member: CascadeMember,
    /// The observation of the track that sits in the queried image.
    pub query_observation: usize,
    /// The members tried before it, each with its refusal.
    pub refusals: Vec<MemberRefusal>,
    /// What the returning member did and measured, in order.
    pub stages: Vec<StageRecord>,
}

/// Why no track was built.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackAtPixelError {
    /// The queried image is not one of the reconstruction's.
    NoSuchImage {
        /// The image asked about.
        image: u32,
        /// How many images the reconstruction holds.
        image_count: usize,
    },
    /// The pixel is not a place on the photograph.
    PixelOffImage {
        /// The pixel asked about.
        pixel: [f64; 2],
        /// The photograph's width, in px.
        width: u32,
        /// The photograph's height, in px.
        height: u32,
    },
    /// An input does not have one entry per image of the reconstruction.
    InputMismatch {
        /// Which input.
        input: &'static str,
        /// How many entries it has.
        got: usize,
        /// How many images the reconstruction holds.
        image_count: usize,
    },
    /// Every member refused.
    Refused {
        /// Each member's refusal, in the order they were tried.
        refusals: Vec<MemberRefusal>,
    },
    /// The progress handle was cancelled.
    Cancelled,
}

impl TrackAtPixelError {
    /// The name of the stage that refused: `query` for a query that names no
    /// place, `cascade` when every member refused, `cancelled` for a
    /// cancellation.
    pub fn stage(&self) -> &'static str {
        match self {
            Self::NoSuchImage { .. } | Self::PixelOffImage { .. } | Self::InputMismatch { .. } => {
                "query"
            }
            Self::Refused { .. } => "cascade",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for TrackAtPixelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchImage { image, image_count } => write!(
                f,
                "image {image} is not one of the reconstruction's {image_count} images"
            ),
            Self::PixelOffImage {
                pixel,
                width,
                height,
            } => write!(
                f,
                "the pixel ({:.1}, {:.1}) is not on the {width}x{height} photograph",
                pixel[0], pixel[1]
            ),
            Self::InputMismatch {
                input,
                got,
                image_count,
            } => write!(
                f,
                "{input} has {got} entries, but the reconstruction has {image_count} images"
            ),
            Self::Refused { refusals } => match refusals.last() {
                Some(last) => write!(
                    f,
                    "every member refused; the last, {}, at {}: {}",
                    last.member, last.stage, last.reason
                ),
                None => write!(f, "no member was asked"),
            },
            Self::Cancelled => write!(f, "the track-at-pixel query was cancelled"),
        }
    }
}

impl std::error::Error for TrackAtPixelError {}

/// Build a track-stage track centred on `pixel` in `image`, or say why none
/// could be built.
///
/// `views` holds one decoded, posed view per image of `edited`, in its order,
/// as every photometric bench step takes them. The members are tried in the
/// order `options.members` names them, and the first track that passes its
/// member's gates is returned, with the observation in `image` at index
/// [`TrackAtPixelReport::query_observation`] and every earlier member's
/// refusal in the report. When every member refuses, the error carries each
/// refusal with its stage, its sentence and what it measured.
///
/// Nothing is committed. The reconstruction is read through `edited`, whose
/// deleted points no query here sees.
pub fn build_track_at_pixel(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    sources: &TrackAtPixelSources<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, TrackAtPixelReport), TrackAtPixelError> {
    let image_count = edited.image_count();
    for (input, got) in [
        ("views", Some(views.len())),
        ("keypoints", sources.sift_index.map(|s| s.keypoints.len())),
        (
            "clusters",
            sources.clusters.map(MatchesClusters::image_count),
        ),
    ] {
        if let Some(got) = got.filter(|&got| got != image_count) {
            return Err(TrackAtPixelError::InputMismatch {
                input,
                got,
                image_count,
            });
        }
    }
    let Some(view) = views.get(image as usize) else {
        return Err(TrackAtPixelError::NoSuchImage { image, image_count });
    };
    let (width, height) = (view.camera.width, view.camera.height);
    let on_photo = pixel.iter().all(|c| c.is_finite())
        && pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && pixel[0] < f64::from(width)
        && pixel[1] < f64::from(height);
    if !on_photo {
        return Err(TrackAtPixelError::PixelOffImage {
            pixel,
            width,
            height,
        });
    }

    let ctx = Ctx {
        edited,
        views,
        cameras: views.iter().map(ViewCamera::new).collect(),
        observations: ObservationIndex::new(edited),
        sources,
    };
    let mut refusals = Vec::new();
    for &member in &options.members {
        progress
            .check_cancel()
            .map_err(|_| TrackAtPixelError::Cancelled)?;
        let _phase = progress.phase(member.name());
        let mut stages = Vec::new();
        let result = match member {
            CascadeMember::Clusters => members::clusters(&ctx, image, pixel, options, &mut stages),
            CascadeMember::Transfer => members::transfer(&ctx, image, pixel, options, &mut stages),
            CascadeMember::Sweep => members::sweep(&ctx, image, pixel, options, &mut stages),
            CascadeMember::Constellation => {
                members::constellation(&ctx, image, pixel, options, &mut stages)
            }
        };
        match result {
            Ok(track) => {
                // Every member ends by sliding the patch onto the pixel, which
                // drops the bitmap fused where the patch stood before; fuse it
                // again where it stands now, moving nothing, so the track can
                // be committed into a reconstruction that stores one.
                let track = super::fit::fuse_where_it_stands(
                    &track,
                    edited,
                    views,
                    &super::fit::FitOptions::default(),
                );
                return Ok((
                    track,
                    TrackAtPixelReport {
                        member,
                        query_observation: 0,
                        refusals,
                        stages,
                    },
                ));
            }
            Err(Refusal { stage, reason }) => refusals.push(MemberRefusal {
                member,
                stage,
                reason,
                stages,
            }),
        }
    }
    progress
        .check_cancel()
        .map_err(|_| TrackAtPixelError::Cancelled)?;
    Err(TrackAtPixelError::Refused { refusals })
}

/// What a member reads: the version, the views and their cameras, the
/// observation index and the sources.
struct Ctx<'a> {
    edited: &'a EditedReconstruction,
    views: &'a [ProjectedImage<'a>],
    cameras: Vec<ViewCamera<'a>>,
    observations: ObservationIndex<'a>,
    sources: &'a TrackAtPixelSources<'a>,
}

impl Ctx<'_> {
    /// The observations in `image` within `radius_px` of `pixel`, nearest
    /// first.
    fn observations_near(
        &self,
        image: u32,
        pixel: [f64; 2],
        radius_px: f64,
    ) -> Vec<NearbyObservation> {
        self.observations
            .near(image, pixel, radius_px, &self.cameras[image as usize])
    }

    /// The stem of an image's name, which a cluster's label is minted from.
    fn image_stem(&self, image: u32) -> String {
        let name = &self.edited.base.image_table.images[image as usize].name;
        std::path::Path::new(name)
            .file_stem()
            .map_or_else(|| name.clone(), |s| s.to_string_lossy().into_owned())
    }
}

/// A member's refusal before it is labelled with the member.
#[derive(Debug)]
struct Refusal {
    stage: RefusalStage,
    reason: String,
}

impl Refusal {
    fn new(stage: RefusalStage, reason: impl Into<String>) -> Self {
        Self {
            stage,
            reason: reason.into(),
        }
    }
}
