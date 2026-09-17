// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench: a place beside a reconstruction where things are put to be worked
//! on, and the [`EditableTrack`] that is the first kind of thing put there.
//!
//! `specs/core/bench/bench.md` and `specs/core/bench/editable-track.md` are the
//! design. A [`Bench`] is an ordered list of labelled items with, per kind of
//! item, the label of the active one; an item is not part of the
//! reconstruction, so nothing here writes one except
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

pub mod commit;
pub mod evaluate;
pub mod fit;
pub mod search;
pub mod stage;
pub mod steps;
pub mod track;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use commit::{commit, CommitError, CommitReport};
pub use evaluate::{
    evaluate, evaluate_preconditions, open_localizer, EvaluateError, EvaluateOptions,
    EvaluateReport, DEFAULT_MAX_CACHE_BYTES, DEFAULT_MAX_SEED_OFFSET_PX,
};
pub use fit::{fit, fit_preconditions, FitError, FitOptions, FitReport};
pub use search::{
    search_descriptors, Found, SearchError, SearchMatch, SearchOptions, SearchReport,
    DEFAULT_RADIUS_PX,
};
pub use stage::{set_stage, set_stage_preconditions, StageError, StageReport};
pub use steps::{
    add_observation, apply_thresholds, create_cluster, create_track, set_verdict, split,
    AddObservationReport, ClusterSeed, CreateClusterError, CreateReport, CreateTrackError,
    CreateTrackOptions, ObservationSeed, SplitError, SplitReport, ThresholdReport, TrackEditError,
    VerdictReport,
};
pub use track::{
    ClusterMeasurement, ClusterPayload, ClusterTemplate, EditableTrack, Observation, Origin,
    Provenance, Stage, StageKind, Thresholds, TrackMeasurement, TrackPayload, Unmeasured, Verdict,
};

/// One thing on the bench.
///
/// An `enum` with one variant today. The point of the enum is that the list,
/// the labels and the activation belong to the [`Bench`] and not to the value
/// being worked on, so a second kind of item joins by adding a variant and its
/// own active label rather than by widening the track.
#[derive(Debug, Clone, PartialEq)]
pub enum BenchItem {
    /// A track being worked on.
    Track(Arc<EditableTrack>),
}

/// Which kind of item something is, without its value.
///
/// The bench holds one active label per kind, and each kind is edited in a
/// panel of its own, so the kind is what an activation is keyed by.
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

/// An item on the bench, with the label it is named by everywhere.
///
/// The label lives here rather than inside the item because uniqueness is a
/// property of the bench: two benches may hold items minted from the same
/// origin, and neither knows about the other.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchEntry {
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
    /// A rename was asked for a label that is not one: empty, or all
    /// whitespace.
    EmptyLabel,
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
        }
    }
}

impl std::error::Error for BenchError {}

/// The things being worked on beside one reconstruction, in the order they were
/// put there, and which one of each kind is active.
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
    /// The active label per kind. A kind with no entry has no active item,
    /// which is the state of an empty bench.
    active: BTreeMap<ItemKind, String>,
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

    /// The label of the active item of `kind`, or `None` when the bench holds
    /// none of that kind.
    pub fn active_label(&self, kind: ItemKind) -> Option<&str> {
        self.active.get(&kind).map(String::as_str)
    }

    /// The active track, or `None` when no track is on the bench.
    pub fn active_track(&self) -> Option<&Arc<EditableTrack>> {
        self.track(self.active_label(ItemKind::Track)?)
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

    /// Put `item` on the bench under a label minted from `base`, make it the
    /// active item of its kind, and give back the bench and the label it took.
    pub fn put(&self, base: &str, item: BenchItem) -> (Bench, String) {
        let label = self.mint_label(base);
        let kind = item.kind();
        let mut next = self.clone();
        next.entries.push(BenchEntry {
            label: label.clone(),
            item,
        });
        next.active.insert(kind, label.clone());
        (next, label)
    }

    /// Put the value called `label` back with `item` in its place, leaving the
    /// order, the label and the activation as they were.
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

    /// Make the item called `label` the active one of its kind.
    pub fn activate(&self, label: &str) -> Result<Bench, BenchError> {
        let at = self
            .position(label)
            .ok_or_else(|| BenchError::NoSuchItem(label.to_string()))?;
        let mut next = self.clone();
        next.active
            .insert(next.entries[at].item.kind(), label.to_string());
        Ok(next)
    }

    /// Take the item called `label` off the bench.
    ///
    /// When it was the active item of its kind, the item before it in the list
    /// becomes active, and the one after it when it was the first; a kind with
    /// nothing left has no active item. Its label is free to be minted again,
    /// because it names nothing now.
    pub fn discard(&self, label: &str) -> Result<Bench, BenchError> {
        let at = self
            .position(label)
            .ok_or_else(|| BenchError::NoSuchItem(label.to_string()))?;
        let kind = self.entries[at].item.kind();
        let mut next = self.clone();
        next.entries.remove(at);
        if next.active.get(&kind).is_some_and(|l| l == label) {
            match next.neighbour_of_kind(at, kind) {
                Some(neighbour) => {
                    next.active.insert(kind, neighbour);
                }
                None => {
                    next.active.remove(&kind);
                }
            }
        }
        Ok(next)
    }

    /// Rename the item called `label` to `to`.
    ///
    /// The old label then names nothing, and is free to be minted again.
    pub fn rename(&self, label: &str, to: &str) -> Result<Bench, BenchError> {
        let at = self
            .position(label)
            .ok_or_else(|| BenchError::NoSuchItem(label.to_string()))?;
        if to.trim().is_empty() {
            return Err(BenchError::EmptyLabel);
        }
        if let Some(other) = self.position(to) {
            if other != at {
                return Err(BenchError::LabelTaken(to.to_string()));
            }
        }
        let kind = self.entries[at].item.kind();
        let mut next = self.clone();
        next.entries[at].label = to.to_string();
        if next.active.get(&kind).is_some_and(|l| l == label) {
            next.active.insert(kind, to.to_string());
        }
        Ok(next)
    }

    /// The label of the item of `kind` nearest to the slot `at`, which a
    /// removal has already closed up: the one before it, or the one after it
    /// when the removal was at the front.
    fn neighbour_of_kind(&self, at: usize, kind: ItemKind) -> Option<String> {
        self.entries[..at]
            .iter()
            .rev()
            .chain(self.entries[at..].iter())
            .find(|e| e.item.kind() == kind)
            .map(|e| e.label.clone())
    }
}
