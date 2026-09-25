// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The tools that change the scene: what is loaded, what is selected, and how
//! each reconstruction is drawn.
//!
//! Every one of them goes through an `AppState` method rather than assigning a
//! field. That is the whole point of the boundary: `select_image` also selects
//! the intrinsics that image was shot through, `select_camera` clears an image
//! that does not use the lens, `close_node` unwinds every ref into the node it
//! removed — and MCP must not be a second door into the state that skips any of
//! it. What the tools return is the state *after* those rules ran.

use serde_json::json;

use super::{
    render, resolve_camera_image, resolve_camera_intrinsics, resolve_point, resolve_reconstruction,
    CameraImageSel, CloseTarget, DisplayChange, JsonReply, SelectionScope, ToolError,
};
use crate::action_log::{interactive_text, tint_text, visibility_text, Kind, Layer};
use crate::scene::{NodeTint, TINT_PALETTE};
use crate::state::AppState;

/// `open_reconstruction`: start the open, and answer with the reconstruction
/// or with a handle, whichever the clock reaches first.
///
/// An open is a background task like an adjustment, and replies the way one
/// does: a small file lands inside [`super::REPLY_DIRECTLY_WITHIN`] and answers
/// with its reconstruction entry, and one that is still being read, or whose
/// thumbnails and patch bitmaps are still being built, answers with a
/// [`super::BackgroundReply`] handle. A refusal to begin, a path that is not a
/// file or another operation running, is immediate, and the drain records it
/// once as `open_reconstruction failed: ...`.
pub(super) fn open_reconstruction(state: &mut AppState, path: &std::path::Path) -> super::Outcome {
    let already_open = state
        .scene
        .iter()
        .any(|node| node.path.as_deref() == Some(path));
    if let Err(message) = state.start_open(vec![path.to_path_buf()]) {
        return super::Outcome::Done(Err(ToolError::new(message)));
    }
    let task = state.background_task().expect("the open just started");
    super::Outcome::Deferred(super::Deferred::Background(super::BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        answer: super::Answer::Opened { already_open },
        label: task.label.clone(),
        started: task.started,
    }))
}

/// What a landed open answers with: the reconstruction entry of the node it
/// made, and whether its path was already open when the call was made.
pub(super) fn opened_reply(
    state: &AppState,
    opened: Option<crate::scene::ReconId>,
    already_open: bool,
) -> JsonReply {
    let node = opened.and_then(|id| state.node(id)).ok_or_else(|| {
        ToolError::new("The open finished, but its reconstruction is no longer loaded.")
    })?;
    let id = node.id;
    let mut entry = render::reconstruction(node, state.solo, super::bench::index_files(state, id));
    entry
        .as_object_mut()
        .expect("a reconstruction entry is an object")
        .insert("already_open".into(), json!(already_open));
    Ok(entry)
}

pub(super) fn close_reconstruction(state: &mut AppState, target: CloseTarget) -> JsonReply {
    let closed: Vec<String> = match target {
        CloseTarget::All => {
            let labels: Vec<String> = state.scene.iter().map(|n| n.label.clone()).collect();
            state.close_all().map_err(ToolError::new)?;
            labels
        }
        CloseTarget::One(label) => {
            let id = resolve_reconstruction(state, Some(&label))?;
            state.close_node(id).map_err(ToolError::new)?;
            vec![label]
        }
    };
    Ok(json!({ "closed": closed }))
}

pub(super) fn select_reconstruction(state: &mut AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    state.select_recon(id);
    render::selection_reply(state)
}

pub(super) fn select_camera_image(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    selector: &CameraImageSel,
) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let image = resolve_camera_image(state, id, selector)?;
    state.select_image(Some(image));
    render::selection_reply(state)
}

pub(super) fn select_camera_intrinsics(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    index: usize,
) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let camera = resolve_camera_intrinsics(state, id, index)?;
    state.select_camera(Some(camera));
    render::selection_reply(state)
}

pub(super) fn select_point(
    state: &mut AppState,
    query: &crate::goto_point::PointQuery,
) -> JsonReply {
    let point = resolve_point(state, query)?;
    state.select_point(point);
    render::selection_reply(state)
}

pub(super) fn clear_selection(state: &mut AppState, scope: SelectionScope) -> JsonReply {
    match scope {
        // One `AppState` method per scope, and `clear_selection` exists so that
        // "everything" is one Action Log entry rather than the three deselects
        // it is made of.
        SelectionScope::All => state.clear_selection(),
        SelectionScope::CameraImage => state.select_image(None),
        SelectionScope::CameraIntrinsics => state.select_camera(None),
        SelectionScope::Point => state.deselect_point(),
    }
    render::selection_reply(state)
}

pub(super) fn set_reconstruction_display(
    state: &mut AppState,
    label: &str,
    change: DisplayChange,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    // Resolve the tint before touching the node: an unknown palette name is a
    // refusal, and a refusal must not have applied half the call's other fields
    // on its way to being one.
    let tint = match &change.tint {
        None => None,
        Some(None) => Some(NodeTint::Original),
        Some(Some(name)) => Some(NodeTint::Tint(resolve_tint(name)?)),
    };

    let solo = state.solo;
    // Read before the mutable walk below: the entry carries the node's search
    // files, which live on `AppState`, and the walk holds the scene.
    let index = super::bench::index_files(state, id);
    // One entry per field the call *changed*, in the same words the Scene
    // panel's own eyes and tint use: a `set_reconstruction_display` naming four
    // fields is four things to the person watching the window, and the
    // catalogue has a text for each of them rather than one for the call.
    let (scene, log) = state.scene_and_log();
    let node = scene
        .iter_mut()
        .find(|node| node.id == id)
        .expect("just resolved");
    let mut toggle = |current: &mut bool, asked: Option<bool>, layer: Layer, label: &str| {
        if let Some(value) = asked.filter(|value| *value != *current) {
            *current = value;
            log.record(Kind::Scene, visibility_text(label, layer, value));
        }
    };
    toggle(&mut node.visible, change.visible, Layer::Node, label);
    toggle(
        &mut node.show_points,
        change.show_points,
        Layer::Points,
        label,
    );
    toggle(
        &mut node.show_camera_images,
        change.show_camera_images,
        Layer::CameraImages,
        label,
    );
    toggle(
        &mut node.show_patches,
        change.show_patches,
        Layer::Patches,
        label,
    );
    toggle(
        &mut node.show_points_at_infinity,
        change.show_points_at_infinity,
        Layer::PointsAtInfinity,
        label,
    );
    if let Some(interactive) = change.interactive.filter(|v| *v != node.interactive) {
        node.interactive = interactive;
        log.record(Kind::Scene, interactive_text(label, interactive));
    }
    if let Some(tint) = tint.filter(|tint| *tint != node.tint) {
        node.tint = tint;
        log.record(Kind::Scene, tint_text(label, tint));
    }
    Ok(render::reconstruction(node, solo, index))
}

/// The palette entry a name asks for.
///
/// A fixed palette rather than a free colour, for the reason
/// [`crate::scene::TINT_PALETTE`] is fixed in the first place: the job of a
/// tint is telling two reconstructions apart, which a mutually-distinguishable
/// set does by construction. So an unknown name lists the seven rather than
/// falling back to something.
fn resolve_tint(name: &str) -> Result<&'static crate::scene::TintColor, ToolError> {
    TINT_PALETTE
        .iter()
        .find(|color| color.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let names: Vec<String> = TINT_PALETTE
                .iter()
                .map(|color| format!("{:?}", color.name))
                .collect();
            ToolError::new(format!(
                "{name:?} is not a tint — the palette is {}, or null for the reconstruction's \
                 own colors.",
                names.join(", ")
            ))
        })
}

/// Draw only one reconstruction, or end the solo.
///
/// Goes through `AppState::set_solo` rather than `toggle_solo`. A toggle is
/// right for a click, where the user can see the current state; an agent
/// issuing one cannot know the outcome without reading the scene first, and a
/// retried call would undo itself — and a retry that changes nothing writes no
/// Action Log entry either.
pub(super) fn set_solo(state: &mut AppState, label: Option<&str>) -> JsonReply {
    match label {
        Some(label) => {
            let id = resolve_reconstruction(state, Some(label))?;
            state.set_solo(Some(id));
        }
        None => state.set_solo(None),
    }
    Ok(json!({
        "solo": state.solo.and_then(|id| render::label_of(state, id)),
        "scene": state
            .scene
            .iter()
            .map(|node| {
                render::reconstruction(node, state.solo, super::bench::index_files(state, node.id))
            })
            .collect::<Vec<_>>(),
    }))
}
