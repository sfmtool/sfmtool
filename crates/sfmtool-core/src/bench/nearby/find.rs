// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Finding the tracks near a pixel: the matching sources in order with the
//! stopping rule, each candidate's distance range and class, the far-field
//! sweep when the sources leave the pixel's distance open, the depth layers,
//! the tracks the bench takes and the labels they go on it under.
//!
//! `specs/core/bench/nearby-tracks.md` is the design.

use std::time::Instant;

use rayon::prelude::*;

use crate::bench::steps::ClusterSeed;
use crate::bench::track::{EditableTrack, Thresholds};
use crate::bench::track_at_pixel::{
    seed_cluster_with, upgrade_sightings, MatchesClusters, SiftIndexSource, ViewCamera,
};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::candidate::{NearbyCandidate, NearbySource, NearbySourceError};
use super::clusters::{nearby_cluster_tracks, ClusterTracksOptions};
use super::constellation::{constellation_seeds, ConstellationSeedOptions};
use super::far_field::{far_field_sweep, FarFieldError, FarFieldOptions, FarFieldReading};
use super::grey::GreyImages;
use super::guided::{guided_matches, GuidedOptions, GuidedSource};
use super::layers::{
    depth_layers, DepthLayer, DepthLayerError, DepthLayerOptions, LayerCandidate, AT_PIXEL_PX,
};
use super::points::{nearby_points, PointsOptions};
use super::range::{
    camera_spread, classify_range, distance_range, DistanceRangeError, RangeClass, RangeOptions,
};

/// When the matching sources stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopRule {
    /// After the first source that leaves [`NearbyTrackOptions::enough_count`]
    /// usable candidates within [`NearbyTrackOptions::enough_px`] of the pixel.
    #[default]
    Enough,
    /// Never: every source in [`NearbyTrackOptions::sources`] runs.
    Never,
}

impl StopRule {
    /// The rule's name, as the harness spells it (`stop`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Enough => "enough",
            Self::Never => "never",
        }
    }
}

impl std::str::FromStr for StopRule {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "enough" => Ok(Self::Enough),
            "never" => Ok(Self::Never),
            other => Err(format!(
                "unknown stop rule {other:?} (expected enough|never)"
            )),
        }
    }
}

/// When the far-field sweep runs, after the matching sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FarFieldWhen {
    /// When the sources leave the pixel's distance open: they found no usable
    /// candidate, candidates in more than one depth layer, or none usable
    /// within a pixel of the pixel.
    #[default]
    Needed,
    /// Always.
    Always,
    /// Never.
    Never,
}

impl FarFieldWhen {
    /// The name, as the harness spells it (`infinity`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Needed => "needed",
            Self::Always => "always",
            Self::Never => "never",
        }
    }
}

impl std::str::FromStr for FarFieldWhen {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "needed" => Ok(Self::Needed),
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            other => Err(format!(
                "unknown far-field rule {other:?} (expected needed|always|never)"
            )),
        }
    }
}

/// Whether and how the tracks for the bench are built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchTrackOptions {
    /// Build a track-stage [`EditableTrack`] for every usable candidate that
    /// is not an existing point. Off, the result carries the candidates, their
    /// layers and their labels, and no tracks.
    pub build: bool,
    /// The half-width of the patch each built track starts with, in px of the
    /// queried image.
    pub radius_px: f64,
}

impl Default for BenchTrackOptions {
    fn default() -> Self {
        Self {
            build: true,
            radius_px: 8.0,
        }
    }
}

/// What finding the nearby tracks runs with. The defaults are the harness's.
#[derive(Debug, Clone, PartialEq)]
pub struct NearbyTrackOptions {
    /// The matching sources, in the order they run (harness `sources`). Never
    /// [`NearbySource::FarField`], which [`Self::far_field_when`] runs.
    pub sources: Vec<NearbySource>,
    /// When the sources stop (harness `stop`).
    pub stop: StopRule,
    /// How many usable candidates within [`Self::enough_px`] are enough to stop
    /// (harness `min_anchors`).
    pub enough_count: usize,
    /// How near the pixel, in px, a candidate counts toward
    /// [`Self::enough_count`] (harness `enough_px`).
    pub enough_px: f64,
    /// When the far-field sweep runs (harness `infinity`).
    pub far_field_when: FarFieldWhen,
    /// The points source's options.
    pub points: PointsOptions,
    /// The clusters source's options.
    pub clusters: ClusterTracksOptions,
    /// Guided matching's options.
    pub guided: GuidedOptions,
    /// The constellation source's options.
    pub constellation: ConstellationSeedOptions,
    /// The ranges' tolerance and the classes' thresholds (harness `range_px`,
    /// `max_span`, `far_spread`).
    pub range: RangeOptions,
    /// The far-field sweep's options.
    pub far_field: FarFieldOptions,
    /// The depth layers' options.
    pub layers: DepthLayerOptions,
    /// The tracks for the bench.
    pub tracks: BenchTrackOptions,
    /// The group label the tracks' labels start with, in place of
    /// [`nearby_group_label`]'s `<stem>@<x>,<y>`.
    pub label: Option<String>,
}

impl Default for NearbyTrackOptions {
    fn default() -> Self {
        Self {
            sources: vec![
                NearbySource::Points,
                NearbySource::Clusters,
                NearbySource::Guided,
                NearbySource::Constellation,
            ],
            stop: StopRule::Enough,
            enough_count: 2,
            enough_px: 20.0,
            far_field_when: FarFieldWhen::Needed,
            points: PointsOptions::default(),
            clusters: ClusterTracksOptions::default(),
            guided: GuidedOptions::default(),
            constellation: ConstellationSeedOptions::default(),
            range: RangeOptions::default(),
            far_field: FarFieldOptions::default(),
            layers: DepthLayerOptions::default(),
            tracks: BenchTrackOptions::default(),
            label: None,
        }
    }
}

/// The optional inputs of the matching sources, each enabling the source that
/// reads it. Built once per capture and shared by every query.
#[derive(Clone, Copy, Default)]
pub struct NearbyTrackSources<'a> {
    /// The cluster-patches clusters, which the clusters source reads.
    pub clusters: Option<&'a MatchesClusters>,
    /// The keypoints, descriptors and keypoint rays guided matching reads.
    pub guided: Option<GuidedSource<'a>>,
    /// The SIFT index and keypoints the constellation source queries.
    pub sift_index: Option<SiftIndexSource<'a>>,
}

impl NearbyTrackSources<'_> {
    /// The input `source` reads that is missing, or `None` when it has what it
    /// needs. The points source reads only the reconstruction.
    pub fn missing(&self, source: NearbySource) -> Option<&'static str> {
        match source {
            NearbySource::Points | NearbySource::FarField => None,
            NearbySource::Clusters => self.clusters.is_none().then_some("clusters"),
            NearbySource::Guided => self.guided.is_none().then_some("descriptors"),
            NearbySource::Constellation => self.sift_index.is_none().then_some("SIFT index"),
        }
    }
}

/// What found a nearby track: a matching source's candidate or a far-field
/// reading.
#[derive(Debug, Clone, PartialEq)]
pub enum NearbyFinding {
    /// A matching source's candidate.
    Candidate(NearbyCandidate),
    /// A far-field sweep's reading, boxed: it carries the sweep's profiles
    /// and grouping, several times a candidate's size.
    FarField(Box<FarFieldReading>),
}

/// One track found near the pixel: what found it, its distance range and
/// class, its depth layer and label when it is usable, and what the bench
/// takes for it.
#[derive(Debug, Clone, PartialEq)]
pub struct NearbyTrack {
    /// What found it, with its sightings and metrics.
    pub finding: NearbyFinding,
    /// Its distance along its pixel's ray from the queried camera's centre,
    /// infinite for a bearing.
    pub distance: f64,
    /// The distances along its pixel's ray its sightings allow, `[near, far]`.
    pub range: [f64; 2],
    /// Whether [`Self::range`] is bounded or far; usable when either.
    pub class: RangeClass,
    /// How many other usable tracks support it ([`super::DepthLayers::support`]).
    pub support: usize,
    /// Its depth layer, an index into [`NearbyTracks::layers`], when usable.
    pub layer: Option<usize>,
    /// Its place in its layer, 0 first, when usable: by distance from the
    /// pixel, nearest first.
    pub order: Option<usize>,
    /// Its label on the bench, when usable ([`nearby_track_label`]).
    pub label: Option<String>,
    /// The reconstruction's point it is, for [`NearbySource::Points`]: the
    /// bench takes that point's own track rather than a new one.
    pub point: Option<u32>,
    /// The track built from its sightings, when usable, not an existing point
    /// and [`BenchTrackOptions::build`] is on; the reason when building failed.
    pub track: Option<Result<EditableTrack, String>>,
}

impl NearbyTrack {
    /// What found it.
    pub fn source(&self) -> NearbySource {
        match &self.finding {
            NearbyFinding::Candidate(c) => c.source,
            NearbyFinding::FarField(_) => NearbySource::FarField,
        }
    }

    /// Its sightings, `(image, pixel)`, the queried image's first.
    pub fn sightings(&self) -> &[(u32, [f64; 2])] {
        match &self.finding {
            NearbyFinding::Candidate(c) => &c.sightings,
            NearbyFinding::FarField(r) => &r.views,
        }
    }

    /// Where it sits in the queried image.
    pub fn query_pixel(&self) -> [f64; 2] {
        match &self.finding {
            NearbyFinding::Candidate(c) => c.query_pixel,
            NearbyFinding::FarField(r) => r.query_pixel,
        }
    }

    /// How far [`Self::query_pixel`] is from the pixel asked about, in px.
    pub fn distance_px(&self) -> f64 {
        match &self.finding {
            NearbyFinding::Candidate(c) => c.distance_px,
            NearbyFinding::FarField(r) => r.distance_px,
        }
    }

    /// How many images see it.
    pub fn n_views(&self) -> usize {
        self.sightings().len()
    }

    /// Whether the depth layers use it: its range is bounded or far.
    pub fn usable(&self) -> bool {
        self.class.usable()
    }

    fn layer_candidate(&self) -> LayerCandidate<'_> {
        match &self.finding {
            NearbyFinding::Candidate(c) => {
                LayerCandidate::from_candidate(c, self.range, self.class)
            }
            NearbyFinding::FarField(r) => LayerCandidate::from_far_field(r, self.range, self.class),
        }
    }
}

/// What one matching source, or the far-field sweep, did.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceReport {
    /// The source.
    pub source: NearbySource,
    /// How many candidates it found.
    pub found: usize,
    /// The input it needed and did not have, when it was skipped.
    pub skipped: Option<&'static str>,
    /// How long it took, its candidates' ranges included, in seconds.
    pub seconds: f64,
    /// How much of [`Self::seconds`] went to the ranges.
    pub range_seconds: f64,
}

/// Why the far-field sweep ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FarFieldTrigger {
    /// [`FarFieldWhen::Always`].
    Always,
    /// The sources found no usable candidate.
    NoLayer,
    /// The sources' usable candidates fell in more than one depth layer.
    SeveralLayers,
    /// No usable candidate lies within a pixel of the pixel.
    NoneAtPixel,
}

/// The far-field sweep's run.
#[derive(Debug, Clone, PartialEq)]
pub struct FarFieldRun {
    /// Why it ran.
    pub trigger: FarFieldTrigger,
    /// What it found and how long it took.
    pub report: SourceReport,
    /// How many readings its refit dropped.
    pub dropped: usize,
}

/// What a query did, step by step.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NearbyTracksReport {
    /// The matching sources that ran or were skipped, in order. A source the
    /// stopping rule left out is not listed.
    pub sources: Vec<SourceReport>,
    /// The source after which the stopping rule stopped, when it did.
    pub stopped_after: Option<NearbySource>,
    /// The far-field sweep, when it ran.
    pub far_field: Option<FarFieldRun>,
    /// How long the depth layers took, support and evidence included, in
    /// seconds.
    pub layers_seconds: f64,
    /// How long building the tracks for the bench took, in seconds.
    pub tracks_seconds: f64,
}

/// The tracks near a pixel, their depth layers, and what the query did.
#[derive(Debug, Clone, PartialEq)]
pub struct NearbyTracks {
    /// The label every track's label starts with: [`nearby_group_label`], or
    /// the caller's [`NearbyTrackOptions::label`].
    pub group_label: String,
    /// Every track found, usable or not, in the order found: the sources in
    /// order, each in its own order, then the far-field readings.
    pub tracks: Vec<NearbyTrack>,
    /// The depth layers, nearest first; their members index into
    /// [`Self::tracks`].
    pub layers: Vec<DepthLayer>,
    /// What the query did.
    pub report: NearbyTracksReport,
}

impl NearbyTracks {
    /// The usable tracks in the order their labels run: by their layer's rank
    /// (by the layer's place, nearest first, when the layers are not ranked),
    /// then by their order in the layer.
    pub fn bench_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.tracks.len())
            .filter(|&k| self.tracks[k].layer.is_some())
            .collect();
        order.sort_by_key(|&k| {
            let t = &self.tracks[k];
            let layer = t.layer.expect("filtered to tracks in a layer");
            (layer_rank(&self.layers, layer), t.order)
        });
        order
    }

    /// The rank-1 layer, when the layers are ranked and there is one.
    pub fn first_layer(&self) -> Option<&DepthLayer> {
        self.layers
            .iter()
            .find(|l| l.ranking.as_ref().is_some_and(|r| r.rank == 1))
    }
}

/// A layer's rank, or its place (1 for the nearest) when it is not ranked.
fn layer_rank(layers: &[DepthLayer], layer: usize) -> usize {
    layers[layer].ranking.as_ref().map_or(layer + 1, |r| r.rank)
}

/// Why no nearby tracks could be looked for.
#[derive(Debug, Clone, PartialEq)]
pub enum NearbyTracksError {
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
    /// An image's descriptors are not row for row with its keypoints.
    RowMismatch {
        /// The image.
        image: u32,
        /// How many keypoints it has.
        keypoints: usize,
        /// How many descriptors it has.
        descriptors: usize,
    },
    /// [`NearbyTrackOptions::sources`] names the far-field sweep, which is not
    /// a matching source.
    NotAMatchingSource(NearbySource),
    /// The progress handle was cancelled.
    Cancelled,
}

impl std::fmt::Display for NearbyTracksError {
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
            Self::RowMismatch {
                image,
                keypoints,
                descriptors,
            } => write!(
                f,
                "image {image} has {keypoints} keypoints but {descriptors} descriptors"
            ),
            Self::NotAMatchingSource(source) => write!(
                f,
                "{source} is not a matching source; the far-field sweep runs after them"
            ),
            Self::Cancelled => write!(f, "finding the nearby tracks was cancelled"),
        }
    }
}

impl std::error::Error for NearbyTracksError {}

impl From<NearbySourceError> for NearbyTracksError {
    fn from(e: NearbySourceError) -> Self {
        match e {
            NearbySourceError::NoSuchImage { image, image_count } => {
                Self::NoSuchImage { image, image_count }
            }
            NearbySourceError::PixelOffImage {
                pixel,
                width,
                height,
            } => Self::PixelOffImage {
                pixel,
                width,
                height,
            },
            NearbySourceError::InputMismatch {
                input,
                got,
                image_count,
            } => Self::InputMismatch {
                input,
                got,
                image_count,
            },
            NearbySourceError::RowMismatch {
                image,
                keypoints,
                descriptors,
            } => Self::RowMismatch {
                image,
                keypoints,
                descriptors,
            },
        }
    }
}

impl From<FarFieldError> for NearbyTracksError {
    fn from(e: FarFieldError) -> Self {
        match e {
            FarFieldError::NoSuchImage { image, image_count } => {
                Self::NoSuchImage { image, image_count }
            }
            FarFieldError::PixelOffImage {
                pixel,
                width,
                height,
            } => Self::PixelOffImage {
                pixel,
                width,
                height,
            },
            FarFieldError::InputMismatch {
                input,
                got,
                image_count,
            } => Self::InputMismatch {
                input,
                got,
                image_count,
            },
            FarFieldError::Cancelled => Self::Cancelled,
        }
    }
}

impl From<DepthLayerError> for NearbyTracksError {
    fn from(e: DepthLayerError) -> Self {
        match e {
            DepthLayerError::NoSuchImage { image, image_count } => {
                Self::NoSuchImage { image, image_count }
            }
            DepthLayerError::GreyMismatch { grey, image_count } => Self::InputMismatch {
                input: "grey",
                got: grey,
                image_count,
            },
        }
    }
}

impl From<DistanceRangeError> for NearbyTracksError {
    fn from(e: DistanceRangeError) -> Self {
        match e {
            DistanceRangeError::NoSuchImage { image, image_count } => {
                Self::NoSuchImage { image, image_count }
            }
        }
    }
}

/// The label a query's tracks share: the queried image's stem and the pixel,
/// rounded, `<stem>@<x>,<y>`, the form a cluster seeded at the pixel is
/// labelled with ([`ClusterSeed::label`]).
pub fn nearby_group_label(image_stem: &str, pixel: [f64; 2]) -> String {
    ClusterSeed::from_pixel(0, image_stem, pixel, 1.0).label()
}

/// One track's label: the group label, then the layer's rank and a letter for
/// the track's order in the layer (`a` first, `z` then `aa`), then for an
/// existing point ` pt <index>`: `frame_13@412,230 1a`,
/// `frame_13@412,230 2b pt 812`.
pub fn nearby_track_label(group: &str, rank: usize, order: usize, point: Option<u32>) -> String {
    let mut letters = Vec::new();
    let mut n = order + 1;
    while n > 0 {
        n -= 1;
        letters.push(b'a' + (n % 26) as u8);
        n /= 26;
    }
    letters.reverse();
    let letters = String::from_utf8(letters).expect("ASCII letters");
    match point {
        Some(point) => format!("{group} {rank}{letters} pt {point}"),
        None => format!("{group} {rank}{letters}"),
    }
}

/// The stem of image `image`'s name.
fn image_stem(edited: &EditedReconstruction, image: u32) -> String {
    let name = &edited.base.image_table.images[image as usize].name;
    std::path::Path::new(name)
        .file_stem()
        .map_or_else(|| name.clone(), |s| s.to_string_lossy().into_owned())
}

/// Find the tracks near `pixel` in `image`: 3D points that several of the
/// photographs agree on, grouped into depth layers and ranked by how well the
/// pixel's own patch reads at each, each ready for the bench.
///
/// The matching sources run in [`NearbyTrackOptions::sources`]' order, a
/// source whose input `sources` lacks skipped and named in the report, until
/// the [`StopRule`] says there are enough. Every candidate gets its distance
/// range along its pixel's ray and its class; the far-field sweep then runs
/// when [`NearbyTrackOptions::far_field_when`] says so, each reading keeping
/// the range the sweep gave it (or its sightings' range once the refit moved
/// it). The usable tracks are grouped into depth layers and ranked; each gets
/// its layer, its order in the layer, its label and, when
/// [`BenchTrackOptions::build`] is on and it is not an existing point, a
/// track-stage track built from its sightings.
///
/// `views` and `grey` hold one entry per image of `edited`, in its order.
/// Nothing is committed.
// The arguments are `build_track_at_pixel`'s and the grey images the patch
// reads sample, which the caller keeps across queries.
#[allow(clippy::too_many_arguments)]
pub fn find_nearby_tracks(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    sources: &NearbyTrackSources<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &NearbyTrackOptions,
    progress: &Progress<'_>,
) -> Result<NearbyTracks, NearbyTracksError> {
    let image_count = edited.image_count();
    for (input, got) in [
        ("views", Some(views.len())),
        ("grey", Some(grey.len())),
        (
            "clusters",
            sources.clusters.map(MatchesClusters::image_count),
        ),
        ("keypoints", sources.guided.map(|g| g.keypoints.len())),
        ("descriptors", sources.guided.map(|g| g.descriptors.len())),
        ("keypoint rays", sources.guided.map(|g| g.rays.len())),
        ("keypoints", sources.sift_index.map(|s| s.keypoints.len())),
    ] {
        if let Some(got) = got.filter(|&got| got != image_count) {
            return Err(NearbyTracksError::InputMismatch {
                input,
                got,
                image_count,
            });
        }
    }
    let Some(view) = views.get(image as usize) else {
        return Err(NearbyTracksError::NoSuchImage { image, image_count });
    };
    let (width, height) = (view.camera.width, view.camera.height);
    let on_photo = pixel.iter().all(|c| c.is_finite())
        && pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && pixel[0] < f64::from(width)
        && pixel[1] < f64::from(height);
    if !on_photo {
        return Err(NearbyTracksError::PixelOffImage {
            pixel,
            width,
            height,
        });
    }
    if let Some(&source) = options
        .sources
        .iter()
        .find(|&&s| s == NearbySource::FarField)
    {
        return Err(NearbyTracksError::NotAMatchingSource(source));
    }
    let cancelled = |_| NearbyTracksError::Cancelled;

    let spread = camera_spread(views);
    let cq = ViewCamera::new(view);
    let classify = |range: [f64; 2]| classify_range(range, spread, &options.range);
    let mut tracks: Vec<NearbyTrack> = Vec::new();
    let mut report = NearbyTracksReport::default();

    for &source in &options.sources {
        progress.check_cancel().map_err(cancelled)?;
        let _phase = progress.phase(source.name());
        let start = Instant::now();
        if let Some(missing) = sources.missing(source) {
            report.sources.push(SourceReport {
                source,
                found: 0,
                skipped: Some(missing),
                seconds: start.elapsed().as_secs_f64(),
                range_seconds: 0.0,
            });
        } else {
            let found = run_source(edited, views, sources, source, image, pixel, options)?;
            let ranged = Instant::now();
            let n = found.len();
            for c in found {
                let range = c.range(views, options.range.tolerance_px)?;
                tracks.push(unplaced(
                    c.ray_distance(views),
                    range,
                    classify(range),
                    c.point,
                    NearbyFinding::Candidate(c),
                ));
            }
            report.sources.push(SourceReport {
                source,
                found: n,
                skipped: None,
                seconds: start.elapsed().as_secs_f64(),
                range_seconds: ranged.elapsed().as_secs_f64(),
            });
        }
        let close = tracks
            .iter()
            .filter(|t| t.usable() && t.distance_px() <= options.enough_px)
            .count();
        if options.stop == StopRule::Enough && close >= options.enough_count {
            report.stopped_after = Some(source);
            break;
        }
    }

    let trigger = match options.far_field_when {
        FarFieldWhen::Never => None,
        FarFieldWhen::Always => Some(FarFieldTrigger::Always),
        FarFieldWhen::Needed => {
            let grouped = group_count(views, grey, image, pixel, &tracks)?;
            if grouped == 0 {
                Some(FarFieldTrigger::NoLayer)
            } else if grouped > 1 {
                Some(FarFieldTrigger::SeveralLayers)
            } else if !tracks
                .iter()
                .any(|t| t.usable() && t.distance_px() <= AT_PIXEL_PX)
            {
                Some(FarFieldTrigger::NoneAtPixel)
            } else {
                None
            }
        }
    };
    if let Some(trigger) = trigger {
        progress.check_cancel().map_err(cancelled)?;
        let start = Instant::now();
        let sweep = far_field_sweep(
            edited,
            views,
            grey,
            image,
            pixel,
            &options.far_field,
            progress,
        )?;
        let ranged = Instant::now();
        let n = sweep.readings.len();
        for r in sweep.readings {
            let distance = if r.at_infinity {
                f64::INFINITY
            } else {
                let ray = cq.ray(r.query_pixel);
                (r.position - cq.center).dot(&(ray / ray.norm()))
            };
            let range = match r.range {
                Some(range) => range,
                None => distance_range(
                    views,
                    image,
                    r.query_pixel,
                    &r.views,
                    distance,
                    options.range.tolerance_px,
                )?,
            };
            tracks.push(unplaced(
                distance,
                range,
                classify(range),
                None,
                NearbyFinding::FarField(Box::new(r)),
            ));
        }
        report.far_field = Some(FarFieldRun {
            trigger,
            report: SourceReport {
                source: NearbySource::FarField,
                found: n,
                skipped: None,
                seconds: start.elapsed().as_secs_f64(),
                range_seconds: ranged.elapsed().as_secs_f64(),
            },
            dropped: sweep.dropped.len(),
        });
    }

    progress.check_cancel().map_err(cancelled)?;
    let start = Instant::now();
    let found = {
        let _phase = progress.phase("depth layers");
        let candidates: Vec<LayerCandidate<'_>> =
            tracks.iter().map(NearbyTrack::layer_candidate).collect();
        depth_layers(views, grey, image, pixel, &candidates, &options.layers)?
    };
    report.layers_seconds = start.elapsed().as_secs_f64();
    for (t, support) in tracks.iter_mut().zip(found.support) {
        t.support = support;
    }
    let layers = found.layers;

    let group_label = options
        .label
        .clone()
        .unwrap_or_else(|| nearby_group_label(&image_stem(edited, image), pixel));
    for (n, layer) in layers.iter().enumerate() {
        let mut members = layer.members.clone();
        // A stable sort, so members at one distance keep the layer's order.
        members.sort_by(|&a, &b| {
            tracks[a]
                .distance_px()
                .partial_cmp(&tracks[b].distance_px())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let rank = layer_rank(&layers, n);
        for (order, k) in members.into_iter().enumerate() {
            let t = &mut tracks[k];
            t.layer = Some(n);
            t.order = Some(order);
            t.label = Some(nearby_track_label(&group_label, rank, order, t.point));
        }
    }

    let start = Instant::now();
    if options.tracks.build {
        let _phase = progress.phase("bench tracks");
        let stem = image_stem(edited, image);
        let todo: Vec<usize> = (0..tracks.len())
            .filter(|&k| tracks[k].layer.is_some() && tracks[k].point.is_none())
            .collect();
        // Each track is built on its own, so they are built side by side; the
        // steps inside a build are small enough that this is where the
        // parallelism pays.
        let built: Vec<Result<EditableTrack, String>> = todo
            .par_iter()
            .map(|&k| {
                if progress.is_cancelled() {
                    return Err("cancelled".to_string());
                }
                bench_track(
                    edited,
                    views,
                    &stem,
                    tracks[k].sightings(),
                    options.tracks.radius_px,
                )
            })
            .collect();
        progress.check_cancel().map_err(cancelled)?;
        for (k, track) in todo.into_iter().zip(built) {
            tracks[k].track = Some(track);
        }
    }
    report.tracks_seconds = start.elapsed().as_secs_f64();

    Ok(NearbyTracks {
        group_label,
        tracks,
        layers,
        report,
    })
}

/// A track found, before it is placed in a layer.
fn unplaced(
    distance: f64,
    range: [f64; 2],
    class: RangeClass,
    point: Option<u32>,
    finding: NearbyFinding,
) -> NearbyTrack {
    NearbyTrack {
        finding,
        distance,
        range,
        class,
        support: 0,
        layer: None,
        order: None,
        label: None,
        point,
        track: None,
    }
}

/// Run the matching source `source`, whose inputs are all present.
fn run_source(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    sources: &NearbyTrackSources<'_>,
    source: NearbySource,
    image: u32,
    pixel: [f64; 2],
    options: &NearbyTrackOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError> {
    match source {
        NearbySource::Points => nearby_points(edited, views, image, pixel, &options.points),
        NearbySource::Clusters => nearby_cluster_tracks(
            views,
            sources.clusters.expect("checked by `missing`"),
            image,
            pixel,
            &options.clusters,
        ),
        NearbySource::Guided => guided_matches(
            views,
            &sources.guided.expect("checked by `missing`"),
            image,
            pixel,
            &options.guided,
        ),
        NearbySource::Constellation => constellation_seeds(
            edited,
            views,
            &sources.sift_index.expect("checked by `missing`"),
            image,
            pixel,
            &options.constellation,
        ),
        NearbySource::FarField => unreachable!("refused before the sources run"),
    }
}

/// How many depth layers the usable tracks so far fall into: the grouping
/// alone, which reads no photograph.
fn group_count(
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    image: u32,
    pixel: [f64; 2],
    tracks: &[NearbyTrack],
) -> Result<usize, DepthLayerError> {
    let candidates: Vec<LayerCandidate<'_>> =
        tracks.iter().map(NearbyTrack::layer_candidate).collect();
    let grouping = DepthLayerOptions {
        evidence: false,
        ..DepthLayerOptions::default()
    };
    Ok(
        depth_layers(views, grey, image, pixel, &candidates, &grouping)?
            .layers
            .len(),
    )
}

/// A track-stage track at `sightings`, the queried image's first: a cluster
/// seeded at its pixel there, with every other sighting added `in` by hand,
/// upgraded to the track stage, which triangulates and fits it.
fn bench_track(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    stem: &str,
    sightings: &[(u32, [f64; 2])],
    radius_px: f64,
) -> Result<EditableTrack, String> {
    let (image, pixel) = sightings[0];
    let seed = ClusterSeed::from_pixel(image, stem, pixel, radius_px);
    let track = seed_cluster_with(&seed, &Thresholds::default())?;
    upgrade_sightings(edited, views, track, &sightings[1..])
}
