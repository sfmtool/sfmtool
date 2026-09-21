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
//! - **A save records no ancestry.** The file carries its own rows and its own
//!   hash; where an earlier content's rows landed in it is not written down.
//!   What keeps an id taken before the save naming the same point afterwards is
//!   the session's version graph, which holds the map of every step including
//!   the materialisation ([`crate::point_ids::resolve`] walks it).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use sfmtool_core::progress::Progress;
use sfmtool_core::progress_note;
use sfmtool_core::EditedReconstruction;

use crate::action_log::Kind;
use crate::document::PointMap;
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
    /// The pushed value carries the provenance the file will, so its hash, which
    /// is what every id minted afterwards is built on, is the hash of the file
    /// about to be written.
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
