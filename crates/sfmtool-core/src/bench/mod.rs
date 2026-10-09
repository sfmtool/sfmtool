// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench: a place beside a reconstruction where things are put to be worked
//! on, and the [`EditableTrack`] that is the first kind of thing put there.
//!
//! `specs/core/bench/bench.md` and `specs/core/bench/editable-track.md` are the
//! design. A [`Bench`] is an ordered list of labelled items and nothing else;
//! which item a caller works on is the caller's to hold. An item is not part of
//! the reconstruction, so nothing here writes one except
//! [`commit`](commit::commit), which is the one step that does.
//!
//! Everything in this module is a plain value or a pure function over one. No
//! step takes `&mut`: each returns the next value and a report, so a caller
//! that keeps both can undo by pointing at the one it had, and a refusal leaves
//! nothing half-applied.
//!
//! One step reads neither the reconstruction nor a photograph but a file of its
//! own: [`search_descriptors`], which asks a
//! descriptor index which other photographs hold the patch around one
//! observation and adds each as a candidate.
//!
//! [`search_geometry`] asks the same question of the reconstruction instead:
//! it projects a track-stage patch into every supplied view and adds each
//! photometrically admitted photograph as a candidate. It reads photographs,
//! but it is not one of the three below: it publishes no separate precondition
//! half, because everything it needs of a track it checks as it starts.
//!
//! The three steps that read photographs are [`evaluate`](evaluate::evaluate),
//! which fills a track's measurement slots at whichever stage it is in and
//! moves nothing; [`fit`](fit::fit), which localizes, re-triangulates and
//! writes the geometry, and then evaluates its own result; and
//! [`set_stage`](stage::set_stage()), which moves a track between the two
//! stages. All three take the decoded views as a named input, one per image, so
//! decoding and caching stay the caller's. Each also publishes the half of its
//! validation that reads no photograph -- [`evaluate_preconditions`],
//! [`fit_preconditions`] and [`set_stage_preconditions`] -- so a caller that
//! would decode a dozen images first can refuse in front of that work instead
//! of behind it. The step itself calls its own, so the two answers cannot
//! drift.

pub mod classify;
pub mod commit;
pub mod evaluate;
pub mod fit;
pub mod geometry_search;
pub mod nearby;
pub mod normal;
pub mod search;
pub mod stage;
pub mod steps;
pub mod track;
pub mod track_at_pixel;

#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub use classify::{classify_track_rays, ClassificationReason, TrackClassification, TrackRays};
pub use commit::{commit, CommitError, CommitReport};
pub use evaluate::{
    evaluate, evaluate_preconditions, evaluate_rendering_bitmap, open_localizer, score_bitmap,
    stored_patch_resolution, EvaluateError, EvaluateOptions, EvaluateReport,
    DEFAULT_MAX_CACHE_BYTES, DEFAULT_MAX_SEED_OFFSET_PX,
};
pub use fit::{fit, fit_preconditions, render_bitmap_in_place, FitError, FitOptions, FitReport};
pub use geometry_search::{
    search_geometry, GeometryMatch, GeometrySearchError, GeometrySearchOptions,
    GeometrySearchReport,
};
pub use nearby::{
    camera_spread, classify_range, constellation_seeds, depth_layers, distance_range,
    far_field_sweep, find_nearby_tracks, guided_matches, nearby_cluster_tracks, nearby_group_label,
    nearby_points, nearby_track_label, read_patch_along_ray, triangulate_sightings,
    BenchTrackOptions, ClusterMembers, ClusterTracksOptions, ConstellationAt,
    ConstellationSeedOptions, DepthLayer, DepthLayerError, DepthLayerOptions, DepthLayers,
    DistanceRangeError, FarFieldError, FarFieldGrouping, FarFieldMetrics, FarFieldOptions,
    FarFieldReading, FarFieldRun, FarFieldSweep, FarFieldTrigger, FarFieldWhen, GreyImage,
    GreyImages, GuidedOptions, GuidedSource, ImageDescriptors, KeypointRays, LayerCandidate,
    LayerEvidence, LayerRankBy, LayerRanking, NearbyCandidate, NearbyFinding, NearbySource,
    NearbySourceError, NearbyTrack, NearbyTrackOptions, NearbyTrackSources, NearbyTracks,
    NearbyTracksError, NearbyTracksReport, PatchRead, PatchSamples, PointsOptions, RangeClass,
    RangeOptions, RayMeeting, RayPatch, Refit, SourceReport, StopRule, WideAmong, PATCH_GRID,
};
pub use normal::{
    finite_difference_normal, fit_normal, normal_preconditions, FiniteDifferenceOptions,
    FitNormalOptions, NormalError, NormalEstimate, NormalReport, PieceLayout,
};
pub use search::{
    search_descriptors, Found, SearchError, SearchMatch, SearchOptions, SearchReport,
    DEFAULT_RADIUS_PX,
};
pub use stage::{set_stage, set_stage_preconditions, StageError, StageReport};
pub use steps::{
    add_observation, apply_thresholds, bar_checks, clamp_to_photograph, create_cluster,
    create_track, duplicate, half_width_px, pin_verdicts, resize_patch, resize_patch_to_pixel,
    set_reference, set_verdict, shape_observation, sight_observation, spin_patch, split,
    tilt_patch, translate_patch, translate_patch_to_pixel, unpin_verdicts, verdicts_if_unpinned,
    AddObservationReport, Axis, BarCheck, BarChecks, ClusterSeed, CreateClusterError, CreateReport,
    CreateTrackError, CreateTrackOptions, DuplicateError, DuplicateReport, Edge, ObservationSeed,
    PinReport, ReferenceReport, ResizeReport, ShapeReport, SightReport, SpinReport, SplitError,
    SplitReport, ThresholdReport, TiltReport, TiltStop, TrackEditError, TranslateReport,
    TranslateToPixelReport, UnpinReport, VerdictReport, Viewpoint, MAX_TILT_DEG,
};
pub use track::{
    ClusterMeasurement, ClusterPayload, ClusterTemplate, EditableTrack, Observation, Origin,
    Provenance, RepaintMark, Stage, StageKind, Thresholds, TrackMeasurement, TrackPayload,
    Unmeasured, Verdict, BENCH_MAX_PROJECTION_ERROR_PX, BENCH_MAX_SHIFT_PX,
    BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS, BENCH_MIN_ZNCC, BENCH_MIN_ZNCC_MIDDLE,
};
pub use track_at_pixel::{
    build_track_at_pixel, CandidateKind, CandidateRecord, CascadeMember, ClusterMember,
    ClustersOptions, ConstellationOptions, FinishOptions, HypothesisRecord, LateralRecord,
    LocalPriorRecord, MatchesClusters, MatchesClustersError, MemberRefusal, NearbyCluster,
    NearbyObservation, RefusalStage, SiftIndexSource, StageRecord, SweepOptions, TiltRecord,
    TrackAtPixelError, TrackAtPixelOptions, TrackAtPixelReport, TrackAtPixelSources,
    TransferOptions,
};

/// One thing on the bench.
///
/// An `enum` with one variant today. The point of the enum is that the list and
/// the labels belong to the [`Bench`] and not to the value being worked on, so
/// a second kind of item joins by adding a variant rather than by widening the
/// track.
#[derive(Debug, Clone, PartialEq)]
pub enum BenchItem {
    /// A track being worked on.
    Track(Arc<EditableTrack>),
}

/// Which kind of item something is, without its value.
///
/// Each kind is edited in a panel of its own, and a step's report names the
/// kind of item it put on or took off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ItemKind {
    /// [`BenchItem::Track`].
    Track,
}

impl std::fmt::Display for ItemKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ItemKind::Track => write!(f, "track"),
        }
    }
}

impl BenchItem {
    /// Which kind of item this is.
    pub fn kind(&self) -> ItemKind {
        match self {
            BenchItem::Track(_) => ItemKind::Track,
        }
    }

    /// The track, when this item is one.
    pub fn as_track(&self) -> Option<&Arc<EditableTrack>> {
        match self {
            BenchItem::Track(track) => Some(track),
        }
    }
}

/// The identity of one item on a bench, which stays the same while its label
/// changes.
///
/// A label changes on [`Bench::rename`], and an undo across a rename changes it
/// back, so a caller that holds on to an item between frames (a selection, a
/// list of recent items, a cached evaluation) holds its `ItemId` and asks the
/// bench for the current label with [`Bench::label_of`]. The label stays what a
/// log row, a tab and a wire call name the item by.
///
/// [`Bench::put`] mints each ID from one counter shared by the whole process,
/// so an ID is never given out twice, not even on another bench. A caller that
/// keeps benches as versions can undo past a put, drop the redo versions with
/// a new put, and still never see the new item take the ID of the one it
/// replaced in history. [`Bench::replace`] and [`Bench::rename`] keep the ID;
/// [`Bench::discard`] takes it off with the item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ItemId(u64);

/// The next [`ItemId`] to give out. It starts at 1, so no ID is 0.
static NEXT_ITEM_ID: AtomicU64 = AtomicU64::new(1);

impl ItemId {
    /// An ID no item in this process has had before.
    fn mint() -> Self {
        ItemId(NEXT_ITEM_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// The number the ID is.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ItemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// An item on the bench, with the label it is named by everywhere.
///
/// The label lives here rather than inside the item because uniqueness is a
/// property of the bench: two benches may hold items minted from the same
/// origin, and neither knows about the other.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchEntry {
    /// Minted by [`Bench::put`] and kept by every step that installs a new
    /// value under the same entry, so it names the item across a rename.
    pub id: ItemId,
    /// Unique on this bench, and how the item is named in a log row, on a tab
    /// and on the wire.
    pub label: String,
    /// The value being worked on.
    pub item: BenchItem,
}

/// Why a bench operation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchError {
    /// No item on the bench carries that label.
    NoSuchItem(String),
    /// A rename was asked for a label another item already holds.
    LabelTaken(String),
    /// A label was asked for that is not one: empty, or all whitespace.
    EmptyLabel,
    /// A label was asked for that holds a control character, such as a
    /// newline, a tab or a NUL ([`char::is_control`]). The character is
    /// carried so the message can name it.
    ControlCharacter(char),
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchError::NoSuchItem(label) => {
                write!(f, "nothing on the bench is called `{label}`")
            }
            BenchError::LabelTaken(label) => {
                write!(f, "something on the bench is already called `{label}`")
            }
            BenchError::EmptyLabel => write!(f, "an item needs a label with something in it"),
            BenchError::ControlCharacter(c) => write!(
                f,
                "a label cannot hold a control character, and this one holds {:?}",
                c
            ),
        }
    }
}

impl std::error::Error for BenchError {}

/// Whether `label` may name an item on a bench: refused when it is empty or
/// all whitespace ([`BenchError::EmptyLabel`]), or when it holds a control
/// character ([`BenchError::ControlCharacter`]).
///
/// A label is a handle a person reads in the Scene tree and an agent types
/// on the wire, and it is carried into the Action Log and version labels. A
/// newline would draw one item on two rows, and a NUL or tab cannot be typed
/// back. Every step that takes a label from its caller checks it here:
/// [`Bench::rename`], [`create_track`], [`create_cluster`] and
/// [`find_nearby_tracks`]. [`Bench::put`] checks nothing: a label a step
/// mints itself, from an image stem or a portable id, is not a caller's, and
/// the suffixes it appends hold no control character.
///
/// ```
/// use sfmtool_core::bench::{check_label, BenchError};
/// assert_eq!(check_label("bull-nose"), Ok(()));
/// assert_eq!(check_label("  "), Err(BenchError::EmptyLabel));
/// assert_eq!(check_label("a\nb"), Err(BenchError::ControlCharacter('\n')));
/// ```
pub fn check_label(label: &str) -> Result<(), BenchError> {
    if label.trim().is_empty() {
        return Err(BenchError::EmptyLabel);
    }
    match label.chars().find(|c| c.is_control()) {
        Some(c) => Err(BenchError::ControlCharacter(c)),
        None => Ok(()),
    }
}

/// The things being worked on beside one reconstruction, in the order they were
/// put there.
///
/// The bench is only its list: two benches are equal when their items are.
/// Which item a caller is working on is the caller's to hold, and a step that
/// puts an item on gives back the label it minted for that.
///
/// A plain value: `Clone`, no interior mutability. Every operation returns the
/// next bench rather than changing this one, and every item that the operation
/// did not touch is the same `Arc` in both, so a step on one track costs the
/// size of that track.
///
/// Nothing here is part of the reconstruction: a save does not write an item,
/// the point count does not include one, and only [`commit`](commit::commit)
/// crosses from an item to a point.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bench {
    /// The items, oldest first.
    entries: Vec<BenchEntry>,
}

impl Bench {
    /// A bench with nothing on it.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many items are on the bench.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is on the bench.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The items, oldest first.
    pub fn entries(&self) -> &[BenchEntry] {
        &self.entries
    }

    /// Every label, in the order the items were put on.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.label.as_str())
    }

    /// Where `label` sits in the list, or `None` when nothing is called that.
    pub fn position(&self, label: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.label == label)
    }

    /// The item called `label`, or `None` when nothing is.
    pub fn get(&self, label: &str) -> Option<&BenchItem> {
        self.position(label).map(|i| &self.entries[i].item)
    }

    /// The track called `label`, or `None` when nothing is or the item is of
    /// another kind.
    pub fn track(&self, label: &str) -> Option<&Arc<EditableTrack>> {
        self.get(label)?.as_track()
    }

    /// The ID of the item called `label`, or `None` when nothing is.
    pub fn id(&self, label: &str) -> Option<ItemId> {
        self.position(label).map(|i| self.entries[i].id)
    }

    /// The label the item with ID `id` carries on this bench, or `None` when
    /// no item on it has that ID.
    pub fn label_of(&self, id: ItemId) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.label.as_str())
    }

    /// `base`, or the first free `"base (n)"` when `base` is already taken.
    ///
    /// The same disambiguation a scene node's label takes, so a reader who
    /// knows one knows the other. A discarded item frees its label, because a
    /// label that names nothing is available to be minted again.
    pub fn mint_label(&self, base: &str) -> String {
        if self.position(base).is_none() {
            return base.to_string();
        }
        (2..)
            .map(|n| format!("{base} ({n})"))
            .find(|candidate| self.position(candidate).is_none())
            .expect("the candidate sequence is infinite")
    }

    /// Put `item` at the end of the bench under a label minted from `base`, and
    /// give back the bench and the label it took.
    ///
    /// The item gets a new [`ItemId`], one no other item in the process has
    /// had.
    pub fn put(&self, base: &str, item: BenchItem) -> (Bench, String) {
        let label = self.mint_label(base);
        let mut next = self.clone();
        next.entries.push(BenchEntry {
            id: ItemId::mint(),
            label: label.clone(),
            item,
        });
        (next, label)
    }

    /// Put the value called `label` back with `item` in its place, leaving the
    /// order, the label and the [`ItemId`] as they were.
    ///
    /// What a step on one item is installed with: the item is a new `Arc` and
    /// every other is the old one.
    pub fn replace(&self, label: &str, item: BenchItem) -> Result<Bench, BenchError> {
        let at = self
            .position(label)
            .ok_or_else(|| BenchError::NoSuchItem(label.to_string()))?;
        let mut next = self.clone();
        next.entries[at].item = item;
        Ok(next)
    }

    /// Take the item called `label` off the bench, every other item keeping its
    /// place.
    ///
    /// Its label is free to be minted again, because it names nothing now.
    pub fn discard(&self, label: &str) -> Result<Bench, BenchError> {
        let at = self
            .position(label)
            .ok_or_else(|| BenchError::NoSuchItem(label.to_string()))?;
        let mut next = self.clone();
        next.entries.remove(at);
        Ok(next)
    }

    /// Rename the item called `label` to `to`.
    ///
    /// The item keeps its [`ItemId`]. The old label then names nothing, and is
    /// free to be minted again. `to` is refused when [`check_label`] refuses it.
    pub fn rename(&self, label: &str, to: &str) -> Result<Bench, BenchError> {
        let at = self
            .position(label)
            .ok_or_else(|| BenchError::NoSuchItem(label.to_string()))?;
        check_label(to)?;
        if let Some(other) = self.position(to) {
            if other != at {
                return Err(BenchError::LabelTaken(to.to_string()));
            }
        }
        let mut next = self.clone();
        next.entries[at].label = to.to_string();
        Ok(next)
    }

    /// The bench as it reads after image `image` is deleted from its
    /// reconstruction, which moves every later image down by one.
    ///
    /// Each track is put through [`EditableTrack::delete_image`]: its
    /// observations in `image` are dropped and those in later images are
    /// renumbered, so every observation still names the photograph it was
    /// sighted in. **A track left with no observation, because every one it
    /// had was in `image`, is discarded**: nothing it was made of is left in
    /// the reconstruction, and a track with no observations cannot be
    /// evaluated, fitted or committed. A track that had no observations to
    /// begin with is left alone, as is every track that observes no image at
    /// or past `image`; those keep the same `Arc`.
    ///
    /// The report says what changed, so a caller can follow its own indexes
    /// into the observation lists and say in a sentence what the delete did
    /// to the bench.
    pub fn delete_image(&self, image: u32) -> (Bench, ImageDeletion) {
        let mut next = Bench::new();
        let mut report = ImageDeletion::default();
        for entry in &self.entries {
            let BenchItem::Track(track) = &entry.item;
            let Some((track, map)) = track.delete_image(image) else {
                next.entries.push(entry.clone());
                continue;
            };
            report.dropped += map.iter().filter(|m| m.is_none()).count();
            if track.observations.is_empty() {
                report.discarded.push(entry.label.clone());
                continue;
            }
            report.renumbered.push((entry.id, map));
            next.entries.push(BenchEntry {
                id: entry.id,
                label: entry.label.clone(),
                item: BenchItem::Track(Arc::new(track)),
            });
        }
        (next, report)
    }
}

/// What [`Bench::delete_image`] did to a bench.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImageDeletion {
    /// Each item still on the bench whose observations changed, with where
    /// each of its observations went: entry `i` is the new index of
    /// observation `i`, or `None` for one in the deleted image.
    pub renumbered: Vec<(ItemId, Vec<Option<usize>>)>,
    /// How many observations were in the deleted image, over the items kept
    /// and the items discarded.
    pub dropped: usize,
    /// The labels of the items discarded because every observation they had
    /// was in the deleted image, in bench order.
    pub discarded: Vec<String>,
}

impl ImageDeletion {
    /// Whether the delete changed anything on the bench.
    pub fn changed(&self) -> bool {
        !self.renumbered.is_empty() || !self.discarded.is_empty()
    }

    /// Where the delete moved the observations of item `id`: `None` when it
    /// moved none of them, which is also the answer for an item it discarded.
    pub fn observation_map(&self, id: ItemId) -> Option<&[Option<usize>]> {
        self.renumbered
            .iter()
            .find(|(item, _)| *item == id)
            .map(|(_, map)| map.as_slice())
    }
}
