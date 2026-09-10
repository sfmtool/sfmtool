// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The tools that give a node a new version, move its cursor, read its history,
//! and write it out.
//!
//! See `specs/gui/mcp-server.md` § "Editing reconstruction data". Every one of
//! them is one `AppState` call -- the same call the menu or the panel makes --
//! so an agent's edit is a version like any other, in the history the human is
//! looking at and undoable by either of them. What this module adds is the
//! resolution of a wire handle to a node, and a reply that says which version
//! was pushed.
//!
//! Two shapes account for all of it:
//!
//! - [`edited`] wraps every edit. It reads the Action Log's revision before the
//!   call and the node's cursor after it, so the reply carries the version's
//!   serial and label and the **sentence the edit recorded** -- which is where
//!   each family's own numbers already are (add-observation's ZNCC, the
//!   adjustment's residuals), in the words the human is reading off the panel.
//!   One text, not a second rendering of the same report.
//! - [`moved`] wraps the three cursor moves, whose answer is the version now
//!   current rather than one that was made.

use std::path::Path;

use serde_json::{json, Value};

use super::{
    resolve_camera_image, resolve_point_in, resolve_reconstruction, CameraImageSel, JsonReply,
    ToolError,
};
use crate::document::VersionSerial;
use crate::goto_point::PointQuery;
use crate::scene::{ReconId, SceneNode};
use crate::state::AppState;

// ── The history ─────────────────────────────────────────────────────────

/// `get_history`: the node's versions in order, with the cursor, the version on
/// disk, and what undo and redo would do.
///
/// The Edit History panel's own reading of [`crate::document::History`], as
/// JSON: a released row is listed and marked rather than dropped, because the
/// history still knows what happened there while having nothing to show.
pub(super) fn get_history(state: &AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let node = state.node(id).expect("just resolved");
    let history = &node.history;
    let cursor = history.current_version().serial;
    let disk = history.disk_serial();
    // A node that came from no file has no version on disk, whatever the
    // history's own bookkeeping says: there is no file holding one.
    let on_disk = node.path.is_some().then_some(disk);
    let versions: Vec<Value> = history
        .versions()
        .iter()
        .map(|version| {
            json!({
                "serial": version.serial.to_string(),
                "label": version.label,
                "at": state.action_log.format_rfc3339(version.at),
                // False where the budget released the value. The row keeps its
                // place, its label and its map; it is simply no longer a
                // version the cursor can reach.
                "held": version.value.is_some(),
                "is_cursor": version.serial == cursor,
                "is_on_disk": on_disk == Some(version.serial),
            })
        })
        .collect();
    Ok(json!({
        "reconstruction_label": node.label,
        "path": node.path.as_ref().map(|path| path.display().to_string()),
        "cursor": cursor.to_string(),
        "disk_serial": on_disk.map(|serial| serial.to_string()),
        "dirty": node.is_dirty(),
        "can_undo": history.can_undo(),
        "can_redo": history.can_redo(),
        "versions": versions,
    }))
}

pub(super) fn undo(state: &mut AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    moved(state, id, |state| state.undo(id))
}

pub(super) fn redo(state: &mut AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    moved(state, id, |state| state.redo(id))
}

pub(super) fn jump_to_version(state: &mut AppState, label: &str, serial: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let node = state.node(id).expect("just resolved");
    let serial = resolve_serial(node, serial)?;
    moved(state, id, |state| state.jump_to_version(id, serial))
}

// ── The save ────────────────────────────────────────────────────────────

/// `save_reconstruction`: write the node out, over its own path or a named one.
///
/// The two are `AppState::save_node` and `AppState::save_node_as`, which is the
/// same split the File menu makes: Save writes where the node came from and
/// refuses when it came from nowhere, Save As writes where it is told and
/// re-points the node. No dialog is involved on either side -- the path-taking
/// half is the method, and the chooser is the menu's own.
pub(super) fn save_reconstruction(
    state: &mut AppState,
    label: &str,
    path: Option<&Path>,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    match path {
        Some(path) => state.save_node_as(id, path),
        None => state.save_node(id),
    }
    .map_err(ToolError::new)?;
    let node = state.node(id).expect("just resolved");
    Ok(json!({
        // Save As re-points the node, and re-labels it after the file it now
        // holds, so the reply says which label the next call should use.
        "reconstruction_label": node.label,
        "path": node.path.as_ref().map(|path| path.display().to_string()),
        "serial": node.history.disk_serial().to_string(),
    }))
}

// ── The edit families ───────────────────────────────────────────────────

pub(super) fn delete_point(state: &mut AppState, label: &str, query: &PointQuery) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let point = resolve_point_in(state, id, query)?;
    edited(state, id, |state| state.delete_point(point))
}

pub(super) fn delete_camera_image(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let image = resolve_camera_image(state, id, selector)?;
    edited(state, id, |state| state.delete_image(image))
}

/// `add_observation`: the explicit-pixel form, since an agent has no pointer to
/// right-click with.
///
/// `AppState::add_observation` reads the pixel a click left on the state;
/// `add_observation_at` takes one, which is the same edit with the click
/// supplied rather than remembered.
pub(super) fn add_observation(
    state: &mut AppState,
    label: &str,
    query: &PointQuery,
    selector: &CameraImageSel,
    pixel: [f32; 2],
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let point = resolve_point_in(state, id, query)?;
    let image = resolve_camera_image(state, id, selector)?;
    edited(state, id, |state| {
        state.add_observation_at(point, image, pixel)
    })
}

pub(super) fn create_point(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
    pixel: [f32; 2],
    radius_px: Option<f32>,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let image = resolve_camera_image(state, id, selector)?;
    // The prompt's own default, so a call that names no radius makes the point
    // the prompt would have made: the median radius this image's observations
    // already project to.
    let radius = radius_px.unwrap_or_else(|| state.create_point_default_radius(image));
    edited(state, id, |state| state.create_point(image, pixel, radius))
}

pub(super) fn remove_observation(
    state: &mut AppState,
    label: &str,
    query: &PointQuery,
    selector: &CameraImageSel,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let point = resolve_point_in(state, id, query)?;
    let image = resolve_camera_image(state, id, selector)?;
    edited(state, id, |state| state.remove_observation(point, image))
}

/// `move_camera_image`: one camera image put at a pose, by a caller with no
/// hand to place it with.
///
/// The pose is world-from-camera in the reconstruction's **own** frame -- the
/// rotation carries camera axes onto world axes, the translation is the camera
/// centre -- which is the frame every read on this surface reports poses in, so
/// a pose read from `get_camera_image` can be adjusted and sent straight back.
///
/// The lock the human places a camera with ([`crate::camera_lock`]) is
/// deliberately not on the wire: an agent has no viewport to steer, and the
/// pose is the whole input. What the wire owes a lock somebody is holding is to
/// end it before an edit lands under it, and that is [`super::apply_as_agent`]'s
/// job, the lock being the viewport's rather than the state's.
pub(super) fn move_camera_image(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
    quaternion_wxyz: [f64; 4],
    translation: [f64; 3],
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let image = resolve_camera_image(state, id, selector)?;
    let pose = sfmtool_core::Se3Transform::new(
        sfmtool_core::RotQuaternion::from_wxyz_array(quaternion_wxyz),
        nalgebra::Vector3::from_row_slice(&translation),
        1.0,
    );
    edited(state, id, |state| state.move_camera(image, &pose))
}

/// `resect_camera_image_in_place`: the resection landed as the node's next
/// version rather than as a derived node beside it.
///
/// `from_matches` chooses the correspondence source. The matches one reads the
/// `.matches` file remembered for this node, which the Scene panel's own
/// chooser puts there; with none remembered the state refuses in its own words,
/// and this surface does not open a file dialog on an agent's behalf.
pub(super) fn resect_camera_image_in_place(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
    from_matches: bool,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let image = resolve_camera_image(state, id, selector)?;
    let from = if from_matches {
        crate::resect::ResectFrom::Matches
    } else {
        crate::resect::ResectFrom::Observations
    };
    edited(state, id, |state| {
        state.resect_image_in_place(id, image.index(), from)
    })
}

pub(super) fn bundle_adjust(state: &mut AppState, label: &str, release_focal: bool) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let options = sfmtool_core::BundleAdjustOptions {
        opt_f: release_focal,
        ..sfmtool_core::BundleAdjustOptions::default()
    };
    edited(state, id, |state| state.bundle_adjust(id, &options))
}

// ── What an edit and a cursor move answer with ──────────────────────────

/// Run one edit and report the version it pushed.
fn edited(
    state: &mut AppState,
    id: ReconId,
    edit: impl FnOnce(&mut AppState) -> Result<(), String>,
) -> JsonReply {
    let before = state.action_log.revision();
    edit(state).map_err(ToolError::new)?;
    version_reply(state, id, Some(before))
}

/// Move the cursor and report the version it landed on.
fn moved(
    state: &mut AppState,
    id: ReconId,
    step: impl FnOnce(&mut AppState) -> Result<(), String>,
) -> JsonReply {
    step(state).map_err(ToolError::new)?;
    version_reply(state, id, None)
}

/// The version at `id`'s cursor, and -- for an edit -- the sentence it recorded.
fn version_reply(state: &AppState, id: ReconId, since: Option<u64>) -> JsonReply {
    let report = since.and_then(|since| recorded_text(state, since));
    let node = state
        .node(id)
        .ok_or_else(|| ToolError::new("That reconstruction is no longer loaded."))?;
    let version = node.history.current_version();
    let mut reply = json!({
        "reconstruction_label": node.label,
        "cursor": version.serial.to_string(),
        "serial": version.serial.to_string(),
        "label": version.label,
        "dirty": node.is_dirty(),
    });
    if let Some(report) = report {
        reply
            .as_object_mut()
            .expect("a version reply is an object")
            .insert("report".into(), json!(report));
    }
    Ok(reply)
}

/// The last thing the edit recorded, which is its own report of what it did.
///
/// An edit's numbers live in that sentence and nowhere else -- the `AppState`
/// methods return `Result<(), String>` and hand their reports to the Action Log
/// -- so reading it back is what puts them on the wire without a second
/// rendering that could come to disagree with the one the human sees.
fn recorded_text(state: &AppState, since: u64) -> Option<String> {
    state
        .action_log
        .since(since)
        .filter(|entry| !entry.failed)
        .last()
        .map(|entry| entry.text.clone())
}

/// The version of `node` a call named, by the spelling the Edit History panel
/// and the Action Log both show.
///
/// `"v12"` rather than a bare number, because that is the form a human reads
/// off a row and off `Go to: … (v11 → v12)`, and the surface should take back
/// what it hands out. Looked up in the node's own list, so a serial from
/// another node is refused here rather than jumped to.
fn resolve_serial(node: &SceneNode, serial: &str) -> Result<VersionSerial, ToolError> {
    node.history
        .versions()
        .iter()
        .map(|version| version.serial)
        .find(|candidate| candidate.to_string() == serial)
        .ok_or_else(|| {
            ToolError::new(format!(
                "{serial:?} is not a version of {}; get_history lists its versions, spelled as \
                 \"v12\".",
                node.label
            ))
        })
}
