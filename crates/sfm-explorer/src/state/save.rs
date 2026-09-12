// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Writing a node back out: Save, Save As, and what the two of them owe the
//! history.
//!
//! See `specs/gui/saving.md`. Save writes the value at the node's cursor over
//! the node's own path; Save As writes it to a chosen path and re-points the
//! node at it. Both go through [`AppState::write_node`], because everything
//! interesting is the same in the two of them:
//!
//! - **A save with an overlay materialises first**, and the materialisation is a
//!   version like any other, pushed onto the node's history with its row map. So
//!   the value that reached the disk is a version the cursor is sitting on, not
//!   a thing assembled on the way out and forgotten -- which is what lets
//!   [`crate::document::History::set_disk_serial`] name it and the dirty marker
//!   mean something.
//! - **Provenance is stamped before the hash is taken.** The metadata is inside
//!   `content_xxh128`, so stamping an operation onto a value after hashing it
//!   would hand the session a hash the file does not have and break every id
//!   minted against it. The stamp therefore goes on the materialised value
//!   before it becomes a version, and a value that needs no materialisation is
//!   written exactly as it stands, provenance and all.
//! - **Lineage is composed on the way out.** The file records where each
//!   ancestor's rows went, so an id minted in this session, or in the session
//!   that wrote the file this node was loaded from, still resolves against the
//!   file just written. See [`lineage_for`].

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use sfmtool_core::progress::Progress;
use sfmtool_core::progress_note;
use sfmtool_core::{
    EditedReconstruction, LineageEntry, LineageMap, LINEAGE_KIND_BASE, LINEAGE_KIND_POINT_EDIT,
};

use crate::action_log::Kind;
use crate::document::{PointMap, VersionSerial};
use crate::progress::Collector;
use crate::scene::{ReconId, SceneNode};

use super::AppState;

#[cfg(test)]
mod tests;

/// What the viewer records as the operation behind a file it writes.
const SAVE_OPERATION: &str = "edit";

/// What the viewer records as the tool behind a file it writes.
const SAVE_TOOL: &str = "sfm-explorer";

impl AppState {
    /// Whether `id`'s cursor is somewhere other than the version its file holds.
    ///
    /// The one question the dirty marker and the close prompt ask. A node that
    /// came from no file is dirty from the moment it is made: there is no file
    /// holding it, so leaving it loses it.
    pub fn is_dirty(&self, id: ReconId) -> bool {
        self.node(id).is_some_and(SceneNode::is_dirty)
    }

    /// Whether any loaded node is dirty, which is what closing the window asks.
    pub fn any_dirty(&self) -> bool {
        self.scene.iter().any(SceneNode::is_dirty)
    }

    /// The labels of the dirty nodes, for the close prompt's sentence: it names
    /// what is at stake rather than saying that something is.
    pub fn dirty_labels(&self) -> Vec<String> {
        self.scene
            .iter()
            .filter(|node| node.is_dirty())
            .map(|node| node.label.clone())
            .collect()
    }

    /// The ids of the dirty nodes, for the close prompt's Save.
    pub fn dirty_ids(&self) -> Vec<ReconId> {
        self.scene
            .iter()
            .filter(|node| node.is_dirty())
            .map(|node| node.id)
            .collect()
    }

    /// Write `id` over its own path.
    ///
    /// `Err` when the node came from no file -- which is Save As's case, not
    /// this one -- or when the write itself failed.
    pub fn save_node(&mut self, id: ReconId) -> Result<(), String> {
        let path = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?
            .path
            .clone()
            .ok_or_else(|| {
                format!(
                    "{} came from no file — use Save As to choose one.",
                    self.node(id).map_or("That node", |n| n.label.as_str())
                )
            })?;
        self.write_node(id, &path, false)
    }

    /// Write `id` to `path` and re-point the node at it.
    pub fn save_node_as(&mut self, id: ReconId, path: &Path) -> Result<(), String> {
        self.write_node(id, path, true)
    }

    /// The body of both: materialise if there is an overlay, write, and mark the
    /// version that reached the disk.
    ///
    /// The entry carries what the save reported, recorded with
    /// [`crate::action_log::ActionLog::record_done`] from the instant below so
    /// that the row says how long writing the file took rather than how long
    /// writing the row took. A refusal is returned rather than logged, because
    /// the menu item and the MCP tool each phrase it their own way.
    fn write_node(&mut self, id: ReconId, path: &Path, repoint: bool) -> Result<(), String> {
        // A save of a busy node would write the version before the one the
        // operation is about to install, and mark that as what reached the
        // disk. Refused with the same sentence every other change to the node
        // is refused with.
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let started = Instant::now();
        // The level the Action Log toolbar's checkbox last left, read as the
        // operation starts so that a change to it takes effect on the next one.
        let collector = Collector::new(self.action_log.detailed_timing());
        let message = self.write_node_inner(id, path, repoint, &collector)?;
        self.action_log
            .record_done(Kind::File, started, message, collector.take());
        Ok(())
    }

    /// The save itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn write_node_inner(
        &mut self,
        id: ReconId,
        path: &Path,
        repoint: bool,
        collector: &Collector,
    ) -> Result<String, String> {
        let save = collector.phase("save");
        let index = self
            .scene
            .iter()
            .position(|n| n.id == id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;

        // A released value is not a value: there is nothing at the cursor to
        // write, and the budget only ever releases what the cursor is not on, so
        // this is the "the version I want is gone" case rather than a bug.
        if self.scene[index].history.current_version().value.is_none() {
            return Err(format!(
                "{}'s current version was released to keep it inside the history budget; \
                 there is nothing to write.",
                self.scene[index].label
            ));
        }

        self.materialize_for_save(index, path, &save)?;

        let node = &self.scene[index];
        let value = node.history.current();
        {
            // One phase for the whole write: serialising the columns,
            // compressing them and hashing the sections are stages of
            // `SfmrReconstruction::save` rather than of this call.
            let _phase = save.phase("write");
            value
                .base
                .save(path)
                .map_err(|e| format!("Cannot write {}: {e}", path.display()))?;
        }

        let serial = node.history.current_version().serial;
        let label = node.label.clone();
        let node = &mut self.scene[index];
        node.history.set_disk_serial(serial);
        if repoint {
            node.path = Some(path.to_path_buf());
            node.label = crate::scene::label_for_path(path);
        }
        let label = if repoint { node.label.clone() } else { label };
        // `save` closes as this returns, which is before the caller empties the
        // collector into the entry.
        Ok(format!("Saved {label} at {serial} to {}", path.display()))
    }

    /// Fold `index`'s overlay into a base of its own and push it as a version,
    /// when there is an overlay to fold.
    ///
    /// The pushed value carries the provenance and the lineage the file will,
    /// so its hash -- which is what every id minted afterwards is built on -- is
    /// the hash of the file about to be written.
    ///
    /// `save` is the phase the save's stages sit under. The `materialise` row
    /// is recorded whichever branch is taken: a save with nothing to fold is a
    /// save that skipped a stage, and its note is what says so.
    fn materialize_for_save(
        &mut self,
        index: usize,
        path: &Path,
        save: &Progress<'_>,
    ) -> Result<(), String> {
        let mut phase = save.phase("materialise");
        let node = &self.scene[index];
        let edited = node.history.current();
        let (deleted, added) = (edited.deleted_points.len(), edited.added.points.len());
        if deleted == 0 && added == 0 {
            progress_note!(phase, "nothing to fold");
            return Ok(());
        }
        progress_note!(phase, "{deleted} deleted, {added} added");

        let (mut base, row_map) = edited.materialize();
        base.metadata.operation = SAVE_OPERATION.to_string();
        base.metadata.tool = SAVE_TOOL.to_string();
        base.metadata.tool_version = env!("CARGO_PKG_VERSION").to_string();
        base.metadata.lineage = {
            // The stamp itself is four assignments; the walk that builds the
            // lineage is the stage worth a row, because it composes a map per
            // ancestor over the rows of the value being written.
            let mut lineage = phase.phase("lineage");
            let entries = lineage_for(node, &row_map);
            progress_note!(lineage, "{} ancestors", entries.len());
            entries
        };

        let label = format!(
            "Saved {} to {}",
            node.label,
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string())
        );
        {
            let _phase = phase.phase("push version");
            let node = &mut self.scene[index];
            node.history.push(
                EditedReconstruction::new(Arc::new(base)),
                PointMap::Rows(row_map),
                label,
            );
        }
        // The materialisation renumbers, so whatever the selection named has
        // moved with it. One index through one map, which is not a stage.
        let id = self.scene[index].id;
        self.follow_selection_forward(id);
        Ok(())
    }
}

/// The lineage the file about to be written records: where every ancestor's rows
/// are in it.
///
/// `row_map` is the materialisation's own map, from the value at the cursor into
/// the base being written; everything else is a walk over the version graph from
/// an ancestor to the cursor, composed with it. Three kinds of ancestor
/// contribute, and all three are what a Point ID can name:
///
/// - every earlier **base** of this session that still holds its value,
/// - every **point edit** that created points, whose hash names points that are
///   in no base at all,
/// - and every entry of an earlier base's **own** lineage, which is how an id
///   minted two files ago keeps resolving: the earlier file already composed its
///   ancestors' maps into itself, so composing that with the walk to here is the
///   whole chain in one step.
///
/// Entries come out oldest ancestor first, one per hash. The base being written
/// is not its own ancestor, so its hash never appears.
fn lineage_for(node: &SceneNode, row_map: &sfmtool_core::RowMap) -> Vec<LineageEntry> {
    let history = &node.history;
    let cursor = history.current_version().serial;
    // Oldest first: the ancestry runs the other way.
    let chain: Vec<VersionSerial> = history.ancestry(cursor).into_iter().rev().collect();
    let mut entries: Vec<LineageEntry> = Vec::new();

    for serial in chain {
        // From this version's index space into the base being written.
        let into_target = |index: u32| -> Option<u32> {
            history
                .follow(serial, cursor, index)
                .ok()
                .and_then(|at_cursor| row_map.forward(at_cursor))
        };

        if let Some(created) = history.created_by(serial) {
            let rows: Vec<Option<u32>> = created.indexes.iter().map(|&i| into_target(i)).collect();
            push_entry(&mut entries, &created.hash, LINEAGE_KIND_POINT_EDIT, rows);
        }

        let Some(value) = history
            .versions()
            .iter()
            .find(|v| v.serial == serial)
            .and_then(|v| v.value.as_ref())
        else {
            continue;
        };
        let base_rows: Vec<Option<u32>> = (0..value.base_point_count() as u32)
            .map(into_target)
            .collect();

        // The ancestor's own recorded ancestors, composed straight through.
        for entry in &value.base.metadata.lineage {
            let rows: Vec<Option<u32>> = (0..entry.map.source_rows())
                .map(|i| {
                    entry
                        .map
                        .forward(i)
                        .and_then(|row| base_rows.get(row as usize).copied().flatten())
                })
                .collect();
            push_entry(&mut entries, &entry.hash, &entry.kind, rows);
        }

        if let Ok(hash) = value.base_content_hash() {
            push_entry(
                &mut entries,
                &hash.content_xxh128,
                LINEAGE_KIND_BASE,
                base_rows,
            );
        }
    }
    entries
}

/// Add one entry, in the smaller of the two encodings, unless its hash is
/// already recorded or nothing of it survives.
fn push_entry(entries: &mut Vec<LineageEntry>, hash: &str, kind: &str, rows: Vec<Option<u32>>) {
    if rows.iter().all(Option::is_none) || entries.iter().any(|e| e.hash == hash) {
        return;
    }
    entries.push(LineageEntry {
        hash: hash.to_string(),
        kind: kind.to_string(),
        map: compress(rows),
    });
}

/// The stored form of a dense map: the monotone encoding when the map preserves
/// order, and the dense one when it does not.
///
/// A materialisation preserves order, so the monotone form is what a save
/// ordinarily writes and it costs the size of the edit rather than the size of
/// the reconstruction. The dense form is the fallback for a step that reorders,
/// which no edit does today and which the encoding must still be able to say.
fn compress(rows: Vec<Option<u32>>) -> LineageMap {
    let mut deleted: Vec<u32> = Vec::new();
    let mut landed: Vec<u32> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        match row {
            None => deleted.push(i as u32),
            Some(row) => landed.push(*row),
        }
    }
    if landed.windows(2).any(|w| w[0] >= w[1]) {
        return LineageMap::Dense { rows };
    }
    // Every row of the target that no source row landed in, up to the last one
    // that did: past that there is nothing to disambiguate.
    let mut created: Vec<u32> = Vec::new();
    let mut next = 0;
    for &row in &landed {
        created.extend(next..row);
        next = row + 1;
    }
    LineageMap::Monotone {
        source_rows: rows.len() as u32,
        deleted,
        created,
    }
}
