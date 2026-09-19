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
//!   each family's own numbers already are (the commit's counts, the
//!   adjustment's residuals), in the words the human is reading off the panel.
//!   One text, not a second rendering of the same report. It also carries
//!   `changed`, which is whether the cursor moved: a step that had no effect
//!   pushes no version and answers with the one the node still stands at, and
//!   `changed: false` is what tells the two apart without comparing serials
//!   across calls (`specs/gui/bench.md` section "The wire").
//! - [`moved`] wraps the three cursor moves, whose answer is the version now
//!   current rather than one that was made.
//!
//! [`bundle_adjust`], [`convert_to_embedded_patches`] and
//! [`retriangulate_all_points`] are the exceptions to the first sentence, and it
//! is the operations' doing rather than the surface's: each runs on a worker, so
//! the call starts it and the frame answers it, with the version it pushed or
//! with a handle to an operation that is still going
//! ([`background_reply`]).

use std::path::Path;

use serde_json::{json, Value};

use super::{
    resolve_camera_image, resolve_point_in, resolve_reconstruction, BackgroundReply,
    CameraImageSel, Deferred, JsonReply, ToolError, REPLY_DIRECTLY_WITHIN,
};
use crate::action_log::{Actor, Kind};
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

/// `retriangulate_point`: one point re-solved from its own observations, at
/// the poses and the lens the reconstruction already holds.
///
/// A point edit, and it finishes inside the call: one track's rays are a
/// microsecond of arithmetic whatever the reconstruction's size. The reply is
/// the edit's own Action Log sentence, which carries the verdict the operation
/// reached -- finite, at infinity, behind a camera that sees it -- so an agent
/// learns what the observations supported and not merely that something moved.
pub(super) fn retriangulate_point(
    state: &mut AppState,
    label: &str,
    query: &PointQuery,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let point = resolve_point_in(state, id, query)?;
    edited(state, id, |state| state.retriangulate_point(point))
}

/// `retriangulate_all_points`: start the whole-value retriangulation, and
/// answer with its version or with a handle, whichever the clock reaches first.
///
/// The third background operation on the surface, and it replies the way the
/// other two do: a reconstruction of any size spends real time re-solving every
/// track, so a large node outlives [`REPLY_DIRECTLY_WITHIN`] and answers with a
/// [`BackgroundReply`] the frame resolves. A refusal to begin is immediate and
/// in `AppState`'s own words, which are the words the greyed menu entry
/// carries.
pub(super) fn retriangulate_all_points(state: &mut AppState, label: &str) -> super::Outcome {
    let id = match resolve_reconstruction(state, Some(label)) {
        Ok(id) => id,
        Err(error) => return super::Outcome::Done(Err(error)),
    };
    if let Err(message) = state.start_retriangulate_all_points(id) {
        return super::Outcome::Done(Err(ToolError::new(message)));
    }
    let task = state.background_task().expect("the operation just started");
    super::Outcome::Deferred(Deferred::Background(BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        node: id,
        label: task.label.clone(),
        started: task.started,
    }))
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

/// `resect_camera_image`: the resection landed as the node's next version.
///
/// `from_matches` chooses the correspondence source. The matches one reads the
/// `.matches` file remembered for this node, which the Scene panel's own
/// chooser puts there; with none remembered the state refuses in its own words,
/// and this surface does not open a file dialog on an agent's behalf.
pub(super) fn resect_camera_image(
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
        state.resect_image(id, image.index(), from)
    })
}

/// `bundle_adjust`: start the solve, and answer with its result or with a
/// handle, whichever the clock reaches first.
///
/// The one tool here that does not finish inside the call, because the
/// operation it asks for does not: the solve runs on a worker
/// ([`crate::background`]), so what this returns is a [`Deferred`] that the
/// frame answers once the operation has finished or once
/// [`REPLY_DIRECTLY_WITHIN`] has passed.
///
/// Nothing waits and no frame is held. An adjustment of a small reconstruction
/// answers with its version and its residuals exactly as it did when it ran on
/// the GUI thread; a long one answers with a handle instead of timing out on
/// the transport, which is what the call used to do for any reconstruction
/// worth adjusting.
pub(super) fn bundle_adjust(
    state: &mut AppState,
    label: &str,
    release_focal: bool,
) -> super::Outcome {
    let id = match resolve_reconstruction(state, Some(label)) {
        Ok(id) => id,
        Err(error) => return super::Outcome::Done(Err(error)),
    };
    let options = sfmtool_core::BundleAdjustOptions {
        opt_f: release_focal,
        ..sfmtool_core::BundleAdjustOptions::default()
    };
    if let Err(message) = state.start_bundle_adjust(id, &options) {
        return super::Outcome::Done(Err(ToolError::new(message)));
    }
    let task = state.background_task().expect("the operation just started");
    super::Outcome::Deferred(Deferred::Background(BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        node: id,
        label: task.label.clone(),
        started: task.started,
    }))
}

/// `convert_to_embedded_patches`: start the conversion, and answer with its
/// version or with a handle, whichever the clock reaches first.
///
/// The second background operation on the surface, and it replies the way the
/// first one does: it reads a `.sift` file per image twice over, so a large
/// node outlives [`REPLY_DIRECTLY_WITHIN`] and answers with a
/// [`BackgroundReply`] the frame resolves. A refusal to begin -- an already
/// embedded reconstruction, or a node an operation is already running on -- is
/// immediate and in `AppState`'s own words, which are the words the greyed
/// menu entry carries.
pub(super) fn convert_to_embedded_patches(state: &mut AppState, label: &str) -> super::Outcome {
    let id = match resolve_reconstruction(state, Some(label)) {
        Ok(id) => id,
        Err(error) => return super::Outcome::Done(Err(error)),
    };
    if let Err(message) = state.start_convert_to_embedded_patches(id) {
        return super::Outcome::Done(Err(ToolError::new(message)));
    }
    let task = state.background_task().expect("the operation just started");
    super::Outcome::Deferred(Deferred::Background(BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        node: id,
        label: task.label.clone(),
        started: task.started,
    }))
}

/// `cancel_background_task`: ask the running operation to stop.
///
/// Refuses when there is nothing to stop, and when the operation never asks
/// whether it should -- in the sentence the button's tooltip carries, so the
/// window and the wire give one answer.
pub(super) fn cancel_background_task(state: &mut AppState) -> JsonReply {
    if let Some(why) = state.cancel_refusal() {
        return Err(ToolError::new(why));
    }
    let task = state.background_task().expect("a cancellable operation");
    let reply = json!({
        "cancelling": task.operation.name,
        "reconstruction_label": task.label,
        "operation_id": task.id,
    });
    state.cancel_background_task();
    Ok(reply)
}

/// Whether a deferred `bundle_adjust` can be answered yet, and with what.
///
/// `None` means "ask again next frame", which is the only thing this waits by:
/// the frame moves on, the window redraws, and the operation goes on running.
///
/// Three cases, in the order they are asked. The operation has finished, and
/// the answer is its result. It is still running and has been for longer than
/// [`REPLY_DIRECTLY_WITHIN`], and the answer is the handle. It is still running
/// and has not, and there is no answer yet.
pub(super) fn background_reply(
    state: &AppState,
    pending: &BackgroundReply,
) -> Option<super::Reply> {
    let running = state
        .background_task()
        .is_some_and(|task| task.id == pending.operation_id);
    if !running {
        let Some(outcome) = state
            .last_background_task
            .as_ref()
            .filter(|last| last.id == pending.operation_id)
        else {
            // The operation finished and a later one displaced the record of
            // it, which takes a frame between the two and so should not be
            // reachable from here. Answered anyway rather than left pending:
            // a call nothing ever replies to is a client waiting out its own
            // timeout, which is the failure this whole path exists to end.
            return Some(Err(ToolError::new(format!(
                "{} of {} has finished, and a later operation has displaced its outcome. \
                 get_action_log carries it.",
                pending.operation_name, pending.label
            ))));
        };
        return Some(match &outcome.outcome {
            // An operation that finished with a report pushed a version: the
            // three photometric bench steps end in `Finished::BenchTrack` and
            // the two solves in an edit, and a run that was cancelled or refused
            // is the `Err` arm. So the field is here for the reason it is on
            // every other edit reply -- one question, one answer, whichever
            // family the caller is in.
            Ok(report) => {
                version_reply(state, pending.node, Some(report.clone())).map(|mut reply| {
                    reply
                        .as_object_mut()
                        .expect("a version reply is an object")
                        .insert("changed".into(), json!(true));
                    super::ToolOutput::Json(reply)
                })
            }
            Err(message) => Err(ToolError::new(message.clone())),
        });
    }
    if pending.started.elapsed() < REPLY_DIRECTLY_WITHIN {
        return None;
    }
    Some(Ok(super::ToolOutput::Json(json!({
        // The discriminator: present and true only where the operation
        // outlived the window, so a reader tests one field rather than
        // inspecting the shape.
        "running": true,
        "operation": pending.operation_name,
        "reconstruction_label": pending.label,
        "operation_id": pending.operation_id,
    }))))
}

// ── What an edit and a cursor move answer with ──────────────────────────

/// Run one edit and report the version it pushed.
pub(super) fn edited(
    state: &mut AppState,
    id: ReconId,
    edit: impl FnOnce(&mut AppState) -> Result<(), String>,
) -> JsonReply {
    let before = state.action_log.revision();
    let was = cursor_of(state, id);
    edit(state).map_err(ToolError::new)?;
    let report = recorded_text(state, before);
    let mut reply = version_reply(state, id, report)?;
    // Whether a version was pushed, read off the cursor rather than taken from
    // the step: every step reports what it did in its own shape, and "did the
    // history move" is one question with one answer for all of them.
    reply
        .as_object_mut()
        .expect("a version reply is an object")
        .insert("changed".into(), json!(cursor_of(state, id) != was));
    Ok(reply)
}

/// The serial at `id`'s cursor, or `None` for a node that is no longer loaded.
fn cursor_of(state: &AppState, id: ReconId) -> Option<VersionSerial> {
    Some(state.node(id)?.history.current_version().serial)
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
///
/// The sentence is handed in rather than looked up here, because the two
/// callers find it in different places: an edit that ran inside the call reads
/// the newest entry it wrote, and a background operation kept its own
/// ([`AppState::last_background_task`]) rather than trusting that nothing was
/// recorded in the minutes it was running.
pub(super) fn version_reply(state: &AppState, id: ReconId, report: Option<String>) -> JsonReply {
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

/// The version the node stands at, for a step that pushed none.
///
/// `changed: false` and the step's **own** sentence, read back from `since`: a
/// step that found nothing to do wrote a no-effect row, and a reply that carried
/// no report would leave an agent reading the previous step's `label` instead.
pub(super) fn unchanged_reply(state: &AppState, id: ReconId, since: u64) -> JsonReply {
    let mut reply = version_reply(state, id, recorded_text(state, since))?;
    reply
        .as_object_mut()
        .expect("a version reply is an object")
        .insert("changed".into(), json!(false));
    Ok(reply)
}

/// The last thing the edit recorded, which is its own report of what it did.
///
/// An edit's numbers live in that sentence and nowhere else -- the `AppState`
/// methods return `Result<(), String>` and hand their reports to the Action Log
/// -- so reading it back is what puts them on the wire without a second
/// rendering that could come to disagree with the one the human sees.
///
/// **A row the viewer wrote on its own is not the edit's.** A step can set
/// something else off -- putting the first item on a bench opens the node's
/// default descriptor index, which writes a row of its own after the step's --
/// and the reply would then carry that row's sentence under the step's own
/// label. Those rows are [`Actor::Viewer`]'s, because nobody asked for them,
/// and skipping them here is what keeps a reply saying what the call did.
///
/// **Nor is where the edit left the selection.** A commit of a bench track
/// selects the point it wrote, which is a row of its own after the step's and
/// in the caller's name; what the call *did* is the sentence before it.
fn recorded_text(state: &AppState, since: u64) -> Option<String> {
    state
        .action_log
        .since(since)
        .filter(|entry| {
            !entry.failed && entry.actor != Actor::Viewer && entry.kind != Kind::Selection
        })
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
