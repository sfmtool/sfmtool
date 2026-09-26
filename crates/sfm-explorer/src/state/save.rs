// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Writing a node back out: Save, Save As, and what the two of them owe the
//! history; and the minimal copy, which owes it nothing.
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
//!   mean something. A Save As that states a workspace path materialises for
//!   the same reason, whether or not there is an overlay: the stated path is
//!   metadata, and the hash covers the metadata.
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
//!
//! Save As Minimal is the third command and the odd one out:
//! [`AppState::save_minimal_copy`] writes a copy with the heavy columns and the
//! incidental metadata left out, and leaves the node, its history and its disk
//! mark exactly as they were. A plain save writes the columns the file had:
//! display thumbnails never reach a value, and patch bitmaps the open rendered
//! for display are left out by the writer.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use sfmtool_core::progress::Progress;
use sfmtool_core::progress_note;
use sfmtool_core::reconstruction::minimal::SaveStamp;
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

/// What the viewer records as the operation behind a minimal copy it writes.
const MINIMAL_OPERATION: &str = "minimal";

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
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?
            .path
            .clone()
            .ok_or_else(|| {
                format!(
                    "{} came from no file — use Save As to choose one.",
                    self.node(id).map_or("That node", |n| n.label.as_str())
                )
            })?;
        self.write_node(id, &path, false, None)
    }

    /// Write `id` to `path` and re-point the node at it.
    ///
    /// `workspace_path` states the `workspace.relative_path` the file records,
    /// in place of the one the value already carries. That is a change to the
    /// metadata, which sits inside the content hash, so the value carrying it is
    /// folded into a version of its own exactly as a value with point edits is
    /// (see [`Self::save_minimal_copy`] for the copy that states it without
    /// touching the node). `None` writes the path the value has, which is right
    /// for the dialog's save: nobody typed a workspace path into it.
    pub fn save_node_as(
        &mut self,
        id: ReconId,
        path: &Path,
        workspace_path: Option<&str>,
    ) -> Result<(), String> {
        self.write_node(id, path, true, workspace_path)
    }

    /// Write a minimal copy of `id`'s value at the cursor to `path`: the file
    /// `sfm xform --minimal` writes, with the viewer as its tool.
    ///
    /// No thumbnails, no patch bitmaps (the file's own or ones the open
    /// rendered for display), no `lineage`, an empty `workspace.absolute_path`,
    /// `workspace.relative_path` from `path`'s directory unless `workspace_path`
    /// states one, and `operation`
    /// `minimal` by `sfm-explorer` with empty `tool_options`
    /// ([`SfmrReconstruction::to_minimal`](sfmtool_core::SfmrReconstruction::to_minimal),
    /// the one definition the binding's `save(minimal=True)` shares). A value
    /// with point edits on it is materialised for the copy and nothing else.
    ///
    /// **The node is left exactly as it was.** A minimal copy is an export: the
    /// node keeps its path, its label, its history and its disk mark, so it is
    /// no cleaner and no dirtier than before, and the Action Log row says where
    /// the copy went. For the same reason the copy is refused over the node's
    /// own file, which would leave the node claiming a file that no longer
    /// holds what it shows. A running operation does not refuse it, since
    /// nothing about the node changes.
    ///
    /// The refusals are [`Self::minimal_copy_refusal`], which the File menu asks
    /// first so that its workspace-path prompt does not stand in front of a save
    /// that cannot happen.
    pub fn save_minimal_copy(
        &mut self,
        id: ReconId,
        path: &Path,
        workspace_path: Option<&str>,
    ) -> Result<(), String> {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        let message = {
            let save = collector.phase("save minimal");
            if let Some(why) = self.minimal_copy_refusal(id, path) {
                return Err(why);
            }
            let node = self
                .node(id)
                .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
            let version = node.history.current_version();
            let edited = node.history.current();
            let materialised;
            let base = if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                &*edited.base
            } else {
                let _phase = save.phase("materialise");
                materialised = edited.materialize().0;
                &materialised
            };
            let minimal = {
                let _phase = save.phase("minimal");
                base.to_minimal(
                    path,
                    &SaveStamp {
                        operation: MINIMAL_OPERATION,
                        tool: SAVE_TOOL,
                        tool_version: env!("CARGO_PKG_VERSION"),
                        workspace_path,
                    },
                    Default::default(),
                )
            };
            {
                let _phase = save.phase("write");
                minimal
                    .save(path)
                    .map_err(|e| format!("Cannot write {}: {e}", path.display()))?;
            }
            format!(
                "Saved a minimal copy of {} at {} to {}",
                node.label,
                version.serial,
                path.display()
            )
        };
        self.action_log
            .record_done(Kind::File, started, message, collector.take());
        Ok(())
    }

    /// Why a minimal copy of `id` cannot be written to `path`, or `None` when it
    /// can.
    ///
    /// The refusals that hold before any of the work is done: the node is not
    /// loaded, `path` is the file the node came from, or the version at the
    /// cursor was released to keep the node inside the history budget.
    /// [`Self::save_minimal_copy`] asks this itself, and the File menu asks it as
    /// soon as the save dialog names a path, so a user is not made to fill in a
    /// workspace path for a save that is going to be refused.
    pub fn minimal_copy_refusal(&self, id: ReconId, path: &Path) -> Option<String> {
        let Some(node) = self.node(id) else {
            return Some(crate::state::NOT_LOADED.to_string());
        };
        if node.path.as_deref().is_some_and(|own| same_file(own, path)) {
            return Some(format!(
                "A minimal copy of {} cannot replace the file it came from; choose \
                 another path.",
                node.label
            ));
        }
        if node.history.current_version().value.is_none() {
            return Some(format!(
                "{}'s current version was released to keep it inside the history \
                 budget; there is nothing to write.",
                node.label
            ));
        }
        None
    }

    /// The workspace path the `Save As Minimal...` prompt offers for a copy of
    /// `id` written to `path`.
    ///
    /// The measurement the save would make on its own
    /// ([`SfmrReconstruction::measured_workspace_path`](sfmtool_core::SfmrReconstruction::measured_workspace_path)),
    /// so a user who accepts it unread gets the save the viewer made before the
    /// prompt existed. Where there is nothing to measure, because the value
    /// carries no workspace directory or the two paths share no root, the path
    /// the value already records, which is what the file it came from said; and
    /// where that is empty too, empty, since an empty value is the format's "none
    /// recorded" and a path invented here would claim a workspace nobody named.
    pub fn minimal_copy_workspace_path(&self, id: ReconId, path: &Path) -> String {
        let Some(node) = self.node(id) else {
            return String::new();
        };
        if node.history.current_version().value.is_none() {
            return String::new();
        }
        let base = &node.history.current().base;
        base.measured_workspace_path(path)
            .unwrap_or_else(|| base.metadata.workspace.relative_path.clone())
    }

    /// The body of both: materialise if there is an overlay, write, and mark the
    /// version that reached the disk.
    ///
    /// The entry carries what the save reported, recorded with
    /// [`crate::action_log::ActionLog::record_done`] from the instant below so
    /// that the row says how long writing the file took rather than how long
    /// writing the row took. A refusal is returned rather than logged, because
    /// the menu item and the MCP tool each phrase it their own way.
    fn write_node(
        &mut self,
        id: ReconId,
        path: &Path,
        repoint: bool,
        workspace_path: Option<&str>,
    ) -> Result<(), String> {
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
        let message = self.write_node_inner(id, path, repoint, workspace_path, &collector)?;
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
        workspace_path: Option<&str>,
        collector: &Collector,
    ) -> Result<String, String> {
        let save = collector.phase("save");
        let index = self
            .scene
            .iter()
            .position(|n| n.id == id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;

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

        self.materialize_for_save(index, path, workspace_path, &save)?;

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
    /// when there is an overlay to fold or a workspace path to state.
    ///
    /// The pushed value carries the provenance the file will, so its hash, which
    /// is what every id minted afterwards is built on, is the hash of the file
    /// about to be written. A stated `workspace_path` goes on here for that
    /// reason: it is metadata inside the content hash, so writing it without a
    /// version to carry it would hand the session a hash the file does not have.
    /// An empty overlay materialises to the same points and the identity map, so
    /// the only thing that version says is what the metadata now reads.
    ///
    /// `save` is the phase the save's stages sit under. The `materialise` row
    /// is recorded whichever branch is taken: a save with nothing to fold is a
    /// save that skipped a stage, and its note is what says so.
    fn materialize_for_save(
        &mut self,
        index: usize,
        path: &Path,
        workspace_path: Option<&str>,
        save: &Progress<'_>,
    ) -> Result<(), String> {
        let mut phase = save.phase("materialise");
        let node = &self.scene[index];
        let edited = node.history.current();
        let (deleted, added) = (edited.deleted_points.len(), edited.added.points.len());
        if deleted == 0 && added == 0 && workspace_path.is_none() {
            progress_note!(phase, "nothing to fold");
            return Ok(());
        }
        progress_note!(phase, "{deleted} deleted, {added} added");

        let (mut base, row_map) = edited.materialize();
        base.metadata.operation = SAVE_OPERATION.to_string();
        base.metadata.tool = SAVE_TOOL.to_string();
        base.metadata.tool_version = env!("CARGO_PKG_VERSION").to_string();
        if let Some(stated) = workspace_path {
            base.set_workspace_relative_path(stated);
        }

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

/// Whether `a` and `b` name one file: compared canonically where both exist,
/// and as written otherwise.
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}
