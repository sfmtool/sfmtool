// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The document model: a node's reconstruction, its bench and its display
//! transform as a sequence of versions with a cursor.
//!
//! See `specs/gui/document-model.md`, `specs/gui/edit-history.md` and
//! `specs/gui/bench.md`. A node ([`crate::scene::SceneNode`]) holds one
//! [`History`]. A [`Version`] in it has three halves, under a serial that is
//! minted once and never reused: an [`EditedReconstruction`], which is a shared
//! immutable base plus this version's point edits; the [`Bench`] as it stood
//! beside it; and the display transform the node was drawn under. An edit is a function from the value at the cursor to the next
//! value; undo and redo move the cursor; a new edit at a cursor that is not at
//! the end discards the versions after it.
//!
//! A step states only what it changed. A document edit carries the bench and
//! the transform along ([`History::push`]), a bench step carries the value and
//! the transform ([`History::push_bench`]), a reframe carries the value and the
//! bench ([`History::push_transform`]), the commit states the value and the
//! bench, and the bake states all three ([`History::push_pair`]). One Undo
//! therefore walks all three at once, which is the whole reason the bench and
//! the transform live here rather than in stacks of their own.
//!
//! The transform is view state that the timeline remembers, not data: it is in
//! no save, no content hash and no dirty comparison.
//!
//! Two things are deliberately separate here:
//!
//! - **Values are droppable, maps are not.** A version's value is what the
//!   viewer can return to, and the oldest of those are released when a node's
//!   unshared bytes exceed [`HISTORY_BUDGET_BYTES`]. The maps are one
//!   [`PointMap`] per version ever minted, discarded redo tails included, and
//!   are never pruned: a map is the size of the edit that produced it, and it
//!   is what lets an index taken at any version be followed to any other.
//! - **A map is the core edits' own [`PointMap`].** A point edit hands one back
//!   in its report and a bulk edit's is read off its two values, so a version
//!   stores what the edit said it did rather than a translation of it, and one
//!   question -- [`PointMap::forward`] -- is asked of every step whatever kind
//!   of edit made it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use jiff::Timestamp;
use sfmtool_core::bench::{Bench, BenchItem};
use sfmtool_core::{EditedReconstruction, Se3Transform, SfmrReconstruction};

/// What one step did to point indexes, which is the core edits' own vocabulary:
/// every edit that can be a step of a history answers in it, so a version keeps
/// the map its edit handed back rather than one translated here.
pub use sfmtool_core::reconstruction::edited::PointMap;

#[cfg(test)]
mod tests;

/// How many unshared bytes of history one node holds before the oldest values
/// are released.
///
/// A constant rather than a setting: it is a ceiling that keeps a session from
/// exhausting memory, not a quantity anyone has a reason to tune from the
/// window. On the largest reconstruction measured (1 354 MB in memory, of which
/// 1 189 MB are the shared thumbnail and patch-bitmap columns) a bulk edit's
/// unshared cost is the light columns, some 165 MB, so this holds twenty-odd
/// bulk edits or any number of point edits.
pub const HISTORY_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Source of [`VersionSerial`] values. Process-wide and never reset, so a
/// serial names one version for the life of the session however many nodes are
/// loaded.
static NEXT_VERSION_SERIAL: AtomicU64 = AtomicU64::new(0);

/// Identity of one version. Minted once, never reused -- including by a version
/// a discarded redo tail took with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VersionSerial(u64);

impl VersionSerial {
    /// Mint the next unused serial. The only way to make one.
    fn next() -> Self {
        Self(NEXT_VERSION_SERIAL.fetch_add(1, Ordering::Relaxed))
    }

    /// The number the serial is, for a caller that has to hand it to something
    /// outside the viewer.
    ///
    /// `sfmtool_core::bench::Origin` numbers the version a track was put on the
    /// bench from with an opaque `u64`, which core neither mints nor
    /// interprets; this is what the viewer puts in it.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for VersionSerial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// The points one version's edit brought into existence, and the hash they are
/// named by.
///
/// A point that no ancestor holds cannot be named by a base's hash and a row in
/// it, because there is no such row. It is named instead by the content hash of
/// the edit that created it (`EditedReconstruction::point_edit_hash`) and its
/// position among that edit's creations, which is what
/// [`crate::point_ids::mint`] mints and [`crate::point_ids::resolve`] resolves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatedPoints {
    /// The point edit's content hash, 32 lowercase hex digits.
    pub hash: String,
    /// The indexes the created points hold **in this version**, in the order the
    /// edit created them. Position `k` in this list is the `k` of the id.
    pub indexes: Vec<u32>,
}

/// One version of a node's reconstruction, the bench beside it and the display
/// transform it was drawn under.
pub struct Version {
    /// Minted once, never reused.
    pub serial: VersionSerial,
    /// The sentence the Action Log recorded for the edit that produced it, or
    /// how the node was loaded for the first version.
    pub label: String,
    /// When it was made. Wall clock, for the same reason the Action Log's is:
    /// a version is read next to something else that happened.
    pub at: Timestamp,
    /// The value, or `None` once the budget has released it. A released version
    /// keeps its place, its label and its map; it is simply no longer a version
    /// the cursor can reach.
    pub value: Option<EditedReconstruction>,
    /// The bench as it stood, which is the version's other half. Kept whether
    /// or not the value is: a bench is a few tracks and the budget is about
    /// the reconstruction.
    pub bench: Arc<Bench>,
    /// The display transform in force when this version was made.
    ///
    /// View state the timeline remembers: never written, never hashed, and no
    /// part of what [`History::is_dirty`] compares. Kept whether or not the
    /// value is, like the bench, for the same reason: it is eight floats.
    pub transform: Se3Transform,
    /// The version whose document half this one shares, which is itself for a
    /// version that changed it.
    ///
    /// What "dirty" is asked of ([`History::is_dirty`]): a run of bench steps
    /// over a clean value is clean, because saving any of them would write the
    /// same bytes.
    pub document_serial: VersionSerial,
    /// What this version holds that its predecessor did not, as the budget
    /// counts it: the unshared half of the value plus the items of the bench
    /// its predecessor's bench does not share.
    pub unshared_bytes: u64,
}

impl Version {
    /// The value, for a version the budget has not released.
    fn value(&self) -> Option<&EditedReconstruction> {
        self.value.as_ref()
    }
}

/// One edge of the version graph: a version, the version it was made from, what
/// the step did to point indexes, and which points it created.
struct Step {
    serial: VersionSerial,
    parent: VersionSerial,
    map: PointMap,
    created: Option<CreatedPoints>,
}

/// A node's versions, its cursor, and the maps between every version it has
/// ever minted.
pub struct History {
    versions: Vec<Version>,
    /// Which version the node currently shows. Always in range, and always a
    /// version whose value is present.
    cursor: usize,
    /// One entry per version that was made from another. Never pruned -- not by
    /// the budget, and not by a truncation -- so it is the whole graph of the
    /// session, discarded redo tails included, and every walk over it can reach
    /// a version the viewer can no longer show.
    steps: Vec<Step>,
    /// The version the node's file on disk holds: the one it was loaded at,
    /// until a save says otherwise.
    disk_serial: VersionSerial,
}

impl History {
    /// A history holding `base` as its only version, labelled `label`.
    pub fn new(base: SfmrReconstruction, label: impl Into<String>) -> Self {
        let value = EditedReconstruction::new(Arc::new(base));
        let unshared_bytes = value_bytes(&value, None);
        let serial = VersionSerial::next();
        Self {
            versions: vec![Version {
                serial,
                label: label.into(),
                at: Timestamp::now(),
                value: Some(value),
                bench: Arc::new(Bench::new()),
                transform: Se3Transform::identity(),
                document_serial: serial,
                unshared_bytes,
            }],
            cursor: 0,
            steps: Vec::new(),
            disk_serial: serial,
        }
    }

    /// The value the node shows.
    pub fn current(&self) -> &EditedReconstruction {
        self.versions[self.cursor]
            .value()
            .expect("the cursor never rests on a released version")
    }

    /// The value the node shows, mutably, for a fixture that is still being
    /// built. Test-only: an edit in the app is a `push`, never a write.
    #[cfg(test)]
    pub fn current_mut(&mut self) -> &mut EditedReconstruction {
        let cursor = self.cursor;
        self.versions[cursor]
            .value
            .as_mut()
            .expect("the cursor never rests on a released version")
    }

    /// The version the node shows.
    pub fn current_version(&self) -> &Version {
        &self.versions[self.cursor]
    }

    /// The bench the node shows, which is the other half of the value at the
    /// cursor.
    pub fn current_bench(&self) -> &Arc<Bench> {
        &self.versions[self.cursor].bench
    }

    /// The display transform the node is drawn under, which is the third half
    /// of the version at the cursor.
    ///
    /// Read off that version rather than kept beside it, so undo, redo and a
    /// jump restore it by moving the cursor and there is no second copy to
    /// fall out of step.
    pub fn transform(&self) -> &Se3Transform {
        &self.versions[self.cursor].transform
    }

    /// The display transform at the cursor, mutably, for a fixture that is
    /// still being built. Test-only: a reframe in the app is a
    /// [`History::push_transform`], never a write.
    #[cfg(test)]
    pub fn transform_mut(&mut self) -> &mut Se3Transform {
        let cursor = self.cursor;
        &mut self.versions[cursor].transform
    }

    /// Whether the version at the cursor holds a document half other than the
    /// one on disk.
    ///
    /// The **document** half: a bench step pushes a version like any other, and
    /// a run of them over a clean value stays clean, because a save of any of
    /// them would write the same bytes and the bench is not written at all. A
    /// disk version a truncation has taken with it leaves the node dirty, since
    /// what the file holds is then no version of this history.
    pub fn is_dirty(&self) -> bool {
        let current = self.versions[self.cursor].document_serial;
        match self.versions.iter().find(|v| v.serial == self.disk_serial) {
            Some(disk) => disk.document_serial != current,
            None => true,
        }
    }

    /// Every version, oldest first.
    pub fn versions(&self) -> &[Version] {
        &self.versions
    }

    /// Every version, mutably, so a test can charge them against the budget
    /// without building a reconstruction large enough to reach it.
    #[cfg(test)]
    pub fn versions_mut_for_test(&mut self) -> &mut [Version] {
        &mut self.versions
    }

    /// Where the cursor is, as an index into [`History::versions`].
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The version the node's file on disk holds.
    ///
    /// It is the version the node was loaded at until something writes the
    /// node out; a save moves it with [`History::set_disk_serial`], and the
    /// Edit History panel marks whichever version it names.
    pub fn disk_serial(&self) -> VersionSerial {
        self.disk_serial
    }

    /// Say that `serial` is now the version on disk. Called by a save.
    pub fn set_disk_serial(&mut self, serial: VersionSerial) {
        self.disk_serial = serial;
    }

    /// Where `serial` sits in [`History::versions`], for a caller holding a
    /// serial rather than a position.
    pub fn position_of(&self, serial: VersionSerial) -> Option<usize> {
        self.versions.iter().position(|v| v.serial == serial)
    }

    /// Move the cursor one step towards `target`, as an undo or a redo would.
    ///
    /// `Some((from, to))` when it moved; `None` when it is already there or the
    /// step would land on a released version.
    pub fn step_towards(&mut self, target: usize) -> Option<(VersionSerial, VersionSerial)> {
        match target.cmp(&self.cursor) {
            std::cmp::Ordering::Less => self.undo(),
            std::cmp::Ordering::Greater => self.redo(),
            std::cmp::Ordering::Equal => None,
        }
    }

    /// The map from the version with serial `parent` to the version with serial
    /// `serial`, for any version ever minted on this node.
    pub fn map_between(&self, parent: VersionSerial, serial: VersionSerial) -> Option<&PointMap> {
        self.steps
            .iter()
            .find(|s| s.serial == serial && s.parent == parent)
            .map(|s| &s.map)
    }

    /// How many maps are held. The budget never touches these.
    #[allow(dead_code, reason = "read by the version-graph walks and by the tests")]
    pub fn map_count(&self) -> usize {
        self.steps.len()
    }

    /// The version `serial` was made from, or `None` for the node's first
    /// version.
    pub fn parent_of(&self, serial: VersionSerial) -> Option<VersionSerial> {
        self.steps
            .iter()
            .find(|s| s.serial == serial)
            .map(|s| s.parent)
    }

    /// The chain of serials from `serial` back to the node's first version,
    /// `serial` first.
    ///
    /// Defined for every version the node has ever minted, including one a
    /// discarded redo tail took with it: the steps outlive the version rows.
    pub fn ancestry(&self, serial: VersionSerial) -> Vec<VersionSerial> {
        let mut chain = vec![serial];
        while let Some(parent) = self.parent_of(*chain.last().expect("non-empty")) {
            chain.push(parent);
        }
        chain
    }

    /// The points the step that produced `serial` created, when it created any.
    pub fn created_by(&self, serial: VersionSerial) -> Option<&CreatedPoints> {
        self.steps
            .iter()
            .find(|s| s.serial == serial)
            .and_then(|s| s.created.as_ref())
    }

    /// Every version the node has ever minted, oldest first, the discarded ones
    /// included.
    pub fn all_serials(&self) -> Vec<VersionSerial> {
        let mut serials: Vec<VersionSerial> = self
            .versions
            .first()
            .map(|v| v.serial)
            .into_iter()
            .chain(self.steps.iter().map(|s| s.serial))
            .collect();
        serials.sort_unstable();
        serials.dedup();
        serials
    }

    /// Where the point that is index `index` in version `from` is in version
    /// `to`.
    ///
    /// The two need not be on one line of descent. The walk goes back from
    /// `from` to the last version both share, inverting each step's map, and
    /// then forward to `to`; when `from` is already an ancestor of `to` the
    /// first leg is empty and this is one forward walk. Every map is a bijection
    /// on the points that survive it, so both legs are well defined.
    ///
    /// `Err` carries the version the walk stopped at: the step into or out of it
    /// is where the point ceased to exist, which is the only thing worth saying
    /// to someone whose id did not resolve.
    pub fn follow(
        &self,
        from: VersionSerial,
        to: VersionSerial,
        index: u32,
    ) -> Result<u32, VersionSerial> {
        let up = self.ancestry(from);
        let down = self.ancestry(to);
        let meet = up.iter().copied().find(|s| down.contains(s)).ok_or(from)?;

        let mut index = index;
        // Back to the meeting point, inverting the step that made each version
        // along the way.
        for serial in up.iter().take_while(|s| **s != meet) {
            let parent = self.parent_of(*serial).ok_or(*serial)?;
            let map = self.map_between(parent, *serial).ok_or(*serial)?;
            index = map.inverse(index).ok_or(*serial)?;
        }
        // Then forward, in order, to the destination.
        let forward_leg: Vec<VersionSerial> =
            down.iter().copied().take_while(|s| *s != meet).collect();
        for serial in forward_leg.iter().rev() {
            let parent = self.parent_of(*serial).ok_or(*serial)?;
            let map = self.map_between(parent, *serial).ok_or(*serial)?;
            index = map.forward(index).ok_or(*serial)?;
        }
        Ok(index)
    }

    /// Append `value` as the next version, discarding any redo tail.
    ///
    /// A **document edit**: the bench and the display transform at the cursor
    /// are carried along unchanged, because the step did nothing to them. `map`
    /// says what the step did to
    /// point indexes and is kept for good; `label` is the sentence the Action
    /// Log recorded. Returns the new version's serial.
    pub fn push(
        &mut self,
        value: EditedReconstruction,
        map: PointMap,
        label: impl Into<String>,
    ) -> VersionSerial {
        self.push_creating(value, map, label, None)
    }

    /// [`History::push`] for an edit that created points, which names them.
    ///
    /// `created` is kept on the version for good, so an id minted against the
    /// edit's hash resolves for the rest of the session however far the cursor
    /// travels afterwards.
    pub fn push_creating(
        &mut self,
        value: EditedReconstruction,
        map: PointMap,
        label: impl Into<String>,
        created: Option<CreatedPoints>,
    ) -> VersionSerial {
        let bench = Arc::clone(&self.versions[self.cursor].bench);
        self.push_pair(Some(value), bench, None, map, label, created)
    }

    /// Append the next version with `bench` in place of the one at the cursor
    /// and the document half exactly as it stands.
    ///
    /// A **bench step**: every step on an item, every change to the list and
    /// every activation. Point indexes are untouched, so the map is an empty
    /// `Removed` -- the identity -- and a selection, an id copied before the
    /// step and an undo across it all resolve unchanged.
    pub fn push_bench(&mut self, bench: Arc<Bench>, label: impl Into<String>) -> VersionSerial {
        self.push_pair(
            None,
            bench,
            None,
            PointMap::Removed(Vec::new()),
            label,
            None,
        )
    }

    /// Append a version that states the display transform and nothing else.
    ///
    /// A **reframe**. The document half is untouched, so the version shares its
    /// predecessor's document serial and a clean node stays clean; the bench at
    /// the cursor is carried along; the map is the empty `Removed`, the
    /// identity.
    pub fn push_transform(
        &mut self,
        transform: Se3Transform,
        label: impl Into<String>,
    ) -> VersionSerial {
        let bench = Arc::clone(&self.versions[self.cursor].bench);
        self.push_pair(
            None,
            bench,
            Some(transform),
            PointMap::Removed(Vec::new()),
            label,
            None,
        )
    }

    /// Append the next version stating as many of the three halves as the step
    /// changed. A commit of a bench track is the only step that states two,
    /// the value and the bench; a bake is the only one that states all three.
    ///
    /// `value` is `None` for a step that left the document half alone; such a
    /// version shares its predecessor's document serial, which is what keeps a
    /// run of bench steps, or of reframes, over a clean value clean.
    /// `transform` is `None` to carry the display transform in force, so no
    /// step that is not about the framing can record a different one.
    pub fn push_pair(
        &mut self,
        value: Option<EditedReconstruction>,
        bench: Arc<Bench>,
        transform: Option<Se3Transform>,
        map: PointMap,
        label: impl Into<String>,
        created: Option<CreatedPoints>,
    ) -> VersionSerial {
        // The redo tail's values go; its maps stay, which is what lets an index
        // taken on a discarded version still be followed.
        self.versions.truncate(self.cursor + 1);
        let parent = self.versions[self.cursor].serial;
        let serial = VersionSerial::next();
        let changed_document = value.is_some();
        let value = value.unwrap_or_else(|| {
            self.versions[self.cursor]
                .value()
                .expect("the cursor never rests on a released version")
                .clone()
        });
        let unshared_bytes = value_bytes(&value, self.versions[self.cursor].value())
            + bench_bytes(&bench, Some(&self.versions[self.cursor].bench));
        let document_serial = if changed_document {
            serial
        } else {
            self.versions[self.cursor].document_serial
        };
        let transform = transform.unwrap_or_else(|| self.versions[self.cursor].transform.clone());
        self.versions.push(Version {
            serial,
            label: label.into(),
            at: Timestamp::now(),
            value: Some(value),
            bench,
            transform,
            document_serial,
            unshared_bytes,
        });
        self.cursor = self.versions.len() - 1;
        self.steps.push(Step {
            serial,
            parent,
            map,
            created,
        });
        self.enforce_budget();
        serial
    }

    /// Whether there is a version to step back to that still holds its value.
    pub fn can_undo(&self) -> bool {
        self.cursor > 0 && self.versions[self.cursor - 1].value.is_some()
    }

    /// Whether there is a version to step forward to.
    pub fn can_redo(&self) -> bool {
        self.cursor + 1 < self.versions.len() && self.versions[self.cursor + 1].value.is_some()
    }

    /// Step the cursor back one version. `(undone, now)` serials, or `None`
    /// when there is nothing to undo.
    ///
    /// All three halves come back with the cursor, the display transform
    /// included, which is what lets an undo of a bake leave the picture where
    /// it was.
    pub fn undo(&mut self) -> Option<(VersionSerial, VersionSerial)> {
        if !self.can_undo() {
            return None;
        }
        let undone = self.versions[self.cursor].serial;
        self.cursor -= 1;
        Some((undone, self.versions[self.cursor].serial))
    }

    /// Step the cursor forward one version. `(from, redone)` serials, or `None`
    /// when there is nothing to redo.
    pub fn redo(&mut self) -> Option<(VersionSerial, VersionSerial)> {
        if !self.can_redo() {
            return None;
        }
        let from = self.versions[self.cursor].serial;
        self.cursor += 1;
        Some((from, self.versions[self.cursor].serial))
    }

    /// The map of the step that produced the version at the cursor, for a
    /// caller following an index across an edit or an undo.
    pub fn map_into_cursor(&self) -> Option<&PointMap> {
        let serial = self.versions[self.cursor].serial;
        let parent = self.versions.get(self.cursor.checked_sub(1)?)?.serial;
        self.map_between(parent, serial)
    }

    /// Release the oldest values until the unshared bytes held fit
    /// [`HISTORY_BUDGET_BYTES`].
    ///
    /// Values only: the maps and the version rows stay, so the history still
    /// says what happened and an index can still be followed through a version
    /// the viewer can no longer return to. The version at the cursor is never
    /// released -- it is what the node is showing.
    fn enforce_budget(&mut self) {
        let mut held: u64 = self
            .versions
            .iter()
            .filter(|v| v.value.is_some())
            .map(|v| v.unshared_bytes)
            .sum();
        let mut i = 0;
        while held > HISTORY_BUDGET_BYTES && i < self.cursor {
            if let Some(version) = self.versions.get_mut(i) {
                if version.value.take().is_some() {
                    held -= version.unshared_bytes;
                }
            }
            i += 1;
        }
    }
}

/// What `value` holds that `previous` did not, in bytes, as the budget counts
/// it.
///
/// Two values sharing a base share everything in it, so a point edit costs the
/// overlay alone. A value with a base of its own costs that base's columns,
/// less the thumbnail and patch-bitmap arrays when it points at the same
/// allocations its predecessor did -- which is every bulk edit that does not
/// refit patches.
fn value_bytes(value: &EditedReconstruction, previous: Option<&EditedReconstruction>) -> u64 {
    let overlay = (value.deleted_points.len() * std::mem::size_of::<u32>()
        + value.replaces.len() * std::mem::size_of::<Option<u32>>()
        + point_set_bytes(&value.added)) as u64;
    let shares_base = previous.is_some_and(|p| Arc::ptr_eq(&p.base, &value.base));
    if shares_base {
        return overlay;
    }
    let base = &*value.base;
    let mut bytes = point_set_bytes(&base.point_set) as u64;
    bytes +=
        (base.image_table.images.len() * std::mem::size_of::<sfmtool_core::SfmrImage>()) as u64;
    // The two heavy columns count only when this value does not point at the
    // same allocation its predecessor's base did.
    // A node's display thumbnails built by the open are the node's, not any
    // version's, so they are held once beside the history and never counted
    // here. Patch bitmaps the open rendered for display are the value's column
    // and count like any other.
    let previous_base = previous.map(|p| &*p.base);
    if let Some(thumbnails) = &base.image_table.thumbnails_y_x_rgb {
        let shared = previous_base.is_some_and(|p| {
            p.image_table
                .thumbnails_y_x_rgb
                .as_ref()
                .is_some_and(|q| Arc::ptr_eq(q, thumbnails))
        });
        if !shared {
            bytes += thumbnails.len() as u64;
        }
    }
    if let Some(bitmaps) = &base.point_set.patch_bitmaps_y_x_rgba {
        let shared = previous_base.is_some_and(|p| {
            p.point_set
                .patch_bitmaps_y_x_rgba
                .as_ref()
                .is_some_and(|q| Arc::ptr_eq(q, bitmaps))
        });
        if !shared {
            bytes += bitmaps.len() as u64;
        }
    }
    bytes + overlay
}

/// What `bench` holds that `previous` did not, in bytes, as the budget counts
/// it.
///
/// Every item the step did not touch is the same `Arc` in both benches, so a
/// step on one track costs that track and nothing else. An item is charged its
/// observations with their two measurement slots plus the bitmap the track
/// stage carries, which is the whole of what one costs against a
/// reconstruction's columns.
fn bench_bytes(bench: &Bench, previous: Option<&Bench>) -> u64 {
    bench
        .entries()
        .iter()
        .filter(|entry| {
            !previous.is_some_and(|before| {
                before
                    .get(&entry.label)
                    .is_some_and(|held| match (held, &entry.item) {
                        (BenchItem::Track(held), BenchItem::Track(now)) => Arc::ptr_eq(held, now),
                    })
            })
        })
        .map(|entry| match &entry.item {
            BenchItem::Track(track) => {
                let observations = track.observations.len()
                    * std::mem::size_of::<sfmtool_core::bench::Observation>();
                let bitmap = track
                    .track()
                    .and_then(|payload| payload.bitmap.as_ref())
                    .map_or(0, |bitmap| bitmap.len());
                (entry.label.len() + observations + bitmap) as u64
            }
        })
        .sum()
}

/// A point set's light columns in bytes: the points, the tracks and the
/// per-observation columns. The patch bitmaps are counted by the caller, which
/// is the only place that knows whether they are shared.
fn point_set_bytes(set: &sfmtool_core::PointSet) -> usize {
    set.points.len() * std::mem::size_of::<sfmtool_core::Point3D>()
        + set.tracks.len() * std::mem::size_of::<sfmtool_core::TrackObservation>()
        + set.observation_counts.len() * std::mem::size_of::<u32>()
        + set.observation_offsets.len() * std::mem::size_of::<usize>()
        + set.tracks.len() * std::mem::size_of::<u32>()
        + set
            .patch_u_halfvec_xyz
            .as_ref()
            .map_or(0, |a| a.len() * std::mem::size_of::<f32>())
        + set
            .patch_v_halfvec_xyz
            .as_ref()
            .map_or(0, |a| a.len() * std::mem::size_of::<f32>())
}
