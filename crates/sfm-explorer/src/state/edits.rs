// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`AppState`]'s edits: the two operations that give a node a new version, and
//! the three that move its cursor -- undo, redo, and the Edit History panel's jump,
//! which is the two of them repeated.
//!
//! See `specs/gui/document-model.md` and `specs/gui/edit-history.md`. Each edit
//! is a function from the value at the node's cursor to the next value, pushed
//! onto that node's history with the map that says what it did to point
//! indexes. Nothing here mutates a reconstruction in place: a point edit writes
//! the *overlay* the new version owns, and a bulk edit builds a whole new base
//! out of the old one.
//!
//! The two are here together because they are the two shapes:
//!
//! - [`AppState::delete_selected_point`] is the **point edit**. It calls
//!   `EditedReconstruction::delete_point` on a clone of the current value, so
//!   the base is the same `Arc` and nothing but the deleted set changed.
//! - [`AppState::delete_image`] is the **bulk edit**. It materialises the
//!   current value when the overlay is not empty, runs
//!   `SfmrReconstruction::subset_by_image_indices` over it, and the output is
//!   the next version's base with an empty overlay, under the row map
//!   `RowMap::by_scan` reads off that call's input and output.

use std::sync::Arc;

use sfmtool_core::{EditedReconstruction, RowMap, SfmrReconstruction};

use crate::action_log::Kind;
use crate::document::{PointMap, VersionSerial};
use crate::scene::{ImageRef, PointRef, ReconId};

use super::AppState;

#[cfg(test)]
mod tests;

impl AppState {
    /// Delete the selected point from the node it belongs to.
    ///
    /// A point edit: the version's overlay gains one deleted index, the base is
    /// untouched, and every other index still means what it meant. Returns the
    /// message the caller reports, or `Err` when there is nothing to delete.
    pub fn delete_selected_point(&mut self) -> Result<(), String> {
        let point = self
            .selected_point
            .ok_or_else(|| "No point is selected.".to_string())?;
        self.delete_point(point)
    }

    /// Delete one point, named by ref.
    pub fn delete_point(&mut self, point: PointRef) -> Result<(), String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == point.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let node = &mut self.scene[index];
        let mut next = node.history.current().clone();
        next.delete_point(point.point)
            .map_err(|e| format!("Cannot delete that point: {e}"))?;
        let label = node.label.clone();
        let text = format!("Deleted point {} in {label}", point.point);
        let serial = node
            .history
            .push(next, PointMap::Removed(vec![point.point]), text.clone());
        let parent = version_before(node, serial);
        self.follow_selection_forward(point.recon);
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        Ok(())
    }

    /// Delete one image, and with it its observations and any track that is
    /// left with none.
    ///
    /// A bulk edit: the node's next version is a whole new base, so every
    /// image index at or after the deleted one moves down by one and the
    /// surviving points are renumbered. The caller drops its panel-local caches
    /// for the node afterwards, exactly as it does when a node is closed.
    pub fn delete_image(&mut self, image: ImageRef) -> Result<(), String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let removed = image.image;
        let node = &self.scene[index];
        let edited = node.history.current();
        if removed as usize >= edited.image_count() {
            return Err("That image is no longer in the reconstruction.".to_string());
        }
        if edited.image_count() == 1 {
            return Err(format!(
                "{} has only one image; deleting it would leave nothing.",
                node.label
            ));
        }
        let name = node.recon().image_table.images[removed as usize]
            .name
            .clone();

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, and the subset can read that directly.
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &edited.base,
        };

        let keep: Vec<u32> = (0..source.image_count() as u32)
            .filter(|&i| i != removed)
            .collect();
        let subset = source
            .subset_by_image_indices(&keep, true)
            .map_err(|e| format!("Cannot delete that image: {e}"))?;

        // Where each image of `source` went, which is the keep list read the
        // other way round: the deleted one goes nowhere, and everything past it
        // moves down by one.
        let mut image_map: Vec<Option<u32>> = vec![None; source.image_count()];
        for (new, &old) in keep.iter().enumerate() {
            image_map[old as usize] = Some(new as u32);
        }
        // The subset says nothing about which points it dropped, so the map is
        // read off its input and its output.
        let subset_map = RowMap::by_scan(source, &subset, Some(&image_map))
            .map_err(|e| format!("Cannot delete that image: {e}"))?;

        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(subset_map));
        let map = PointMap::Chain(steps);

        let label = node.label.clone();
        let text = format!("Deleted image {name} from {label}");
        let node = &mut self.scene[index];
        let serial = node.history.push(
            EditedReconstruction::new(Arc::new(subset)),
            map,
            text.clone(),
        );
        let parent = version_before(node, serial);
        self.follow_selection_forward(image.recon);
        // Every image index at or past the deleted one moved, so an image,
        // camera or cached texture named by one of them is now a statement
        // about a different image. The node keeps its identity; what it held
        // about images does not.
        self.forget_images_of(image.recon);
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        Ok(())
    }

    /// Step `id`'s cursor back one version.
    pub fn undo(&mut self, id: ReconId) -> Result<(), String> {
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let node = &mut self.scene[index];
        let undone_label = node.history.current_version().label.clone();
        let Some((undone, now)) = node.history.undo() else {
            return Err(format!("Nothing to undo in {}.", node.label));
        };
        self.follow_selection_backward(id, undone);
        self.forget_images_of(id);
        self.action_log.record(
            Kind::Edit,
            format!("Undo: {undone_label} ({undone} → {now})"),
        );
        Ok(())
    }

    /// Step `id`'s cursor forward one version.
    pub fn redo(&mut self, id: ReconId) -> Result<(), String> {
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let node = &mut self.scene[index];
        let Some((from, redone)) = node.history.redo() else {
            return Err(format!("Nothing to redo in {}.", node.label));
        };
        let redone_label = node.history.current_version().label.clone();
        self.follow_selection_forward(id);
        self.forget_images_of(id);
        self.action_log.record(
            Kind::Edit,
            format!("Redo: {redone_label} ({from} → {redone})"),
        );
        Ok(())
    }

    /// Move `id`'s cursor straight to the version with serial `serial`.
    ///
    /// The move is one action to the user and one Action Log entry, and it is
    /// the sequence of undos or redos that separates the two versions to
    /// everything that follows a map: the cursor is walked one step at a time
    /// and the selection is put through each step's map in turn, so a jump over
    /// three edits lands the selection exactly where three undos would have.
    /// Refused as a whole when any version it would pass through, the
    /// destination included, has had its value released by the budget: there is
    /// nothing there to show.
    pub fn jump_to_version(&mut self, id: ReconId, serial: VersionSerial) -> Result<(), String> {
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let node = &self.scene[index];
        let Some(target) = node.history.position_of(serial) else {
            return Err(format!("{serial} is not a version of {}.", node.label));
        };
        let cursor = node.history.cursor();
        if target == cursor {
            return Err(format!("{serial} is already what {} shows.", node.label));
        }
        let from = node.history.current_version().serial;
        let span = target.min(cursor)..=target.max(cursor);
        if node.history.versions()[span]
            .iter()
            .any(|v| v.value.is_none())
        {
            return Err(format!(
                "Cannot go to {serial}: its value, or one on the way to it, was released to keep {} inside the history budget.",
                node.label
            ));
        }
        while self.scene[index].history.cursor() != target {
            let node = &mut self.scene[index];
            let stepping_back = target < node.history.cursor();
            let Some((left, _)) = node.history.step_towards(target) else {
                break;
            };
            if stepping_back {
                self.follow_selection_backward(id, left);
            } else {
                self.follow_selection_forward(id);
            }
        }
        let node = &self.scene[index];
        let label = node.history.current_version().label.clone();
        let to = node.history.current_version().serial;
        self.forget_images_of(id);
        self.action_log
            .record(Kind::Edit, format!("Go to: {label} ({from} → {to})"));
        Ok(())
    }

    /// Whether `id` has a version to step back to.
    pub fn can_undo(&self, id: ReconId) -> bool {
        self.node(id).is_some_and(|n| n.history.can_undo())
    }

    /// Whether `id` has a version to step forward to.
    pub fn can_redo(&self, id: ReconId) -> bool {
        self.node(id).is_some_and(|n| n.history.can_redo())
    }

    /// Follow the selected point through the step that produced the version now
    /// at `id`'s cursor: a surviving point keeps its place, a deleted one
    /// clears the selection.
    pub(super) fn follow_selection_forward(&mut self, id: ReconId) {
        let Some(point) = self.selected_point.filter(|p| p.recon == id) else {
            return;
        };
        let Some(node) = self.node(id) else { return };
        let moved = match node.history.map_into_cursor() {
            Some(map) => map.forward(point.point),
            None => Some(point.point),
        };
        self.selected_point = moved.map(|index| PointRef::new(id, index as usize));
    }

    /// Follow the selected point back through the step `undone` was made by.
    fn follow_selection_backward(&mut self, id: ReconId, undone: crate::document::VersionSerial) {
        let Some(point) = self.selected_point.filter(|p| p.recon == id) else {
            return;
        };
        let Some(node) = self.node(id) else { return };
        let parent = node.history.current_version().serial;
        let moved = match node.history.map_between(parent, undone) {
            Some(map) => map.inverse(point.point),
            None => Some(point.point),
        };
        self.selected_point = moved.map(|index| PointRef::new(id, index as usize));
    }

    /// Drop everything this state holds that is keyed by an image of `id`, and
    /// clear the image and camera selections in it.
    ///
    /// What a bulk edit owes: it renumbers the image table, so a cached decode
    /// or a selected index would silently become a statement about a different
    /// image. The panels' own texture caches are dropped by the caller, which
    /// is where they are reachable.
    fn forget_images_of(&mut self, id: ReconId) {
        self.sift_cache.retain(|image, _| image.recon != id);
        self.full_res_cache.retain(|image, _| image.recon != id);
        self.selected_image = self.selected_image.filter(|i| i.recon != id);
        self.selected_camera = self.selected_camera.filter(|c| c.recon != id);
        self.hovered_image = self.hovered_image.filter(|i| i.recon != id);
        self.hovered_point = self.hovered_point.filter(|p| p.recon != id);
    }
}

/// The serial of the version `serial` was made from, for the log entry.
fn version_before(
    node: &crate::scene::SceneNode,
    serial: crate::document::VersionSerial,
) -> crate::document::VersionSerial {
    let versions = node.history.versions();
    let position = versions
        .iter()
        .position(|v| v.serial == serial)
        .expect("the version was just pushed");
    versions[position.saturating_sub(1)].serial
}
