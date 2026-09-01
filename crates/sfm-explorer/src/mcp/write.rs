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
    announce, render, resolve_camera_image, resolve_camera_intrinsics, resolve_point,
    resolve_reconstruction, selection_reply, CameraImageSel, CloseTarget, DisplayChange, JsonReply,
    SelectionScope, ToolError,
};
use crate::scene::{NodeTint, TINT_PALETTE};
use crate::state::AppState;

pub(super) fn open_reconstruction(state: &mut AppState, path: &std::path::Path) -> JsonReply {
    let already_open = state
        .scene
        .iter()
        .any(|node| node.path.as_deref() == Some(path));
    let before: Vec<crate::scene::ReconId> = state.scene.iter().map(|node| node.id).collect();

    // `load_file` is used unchanged, including its already-loaded rule: opening
    // a path that is open reloads that node in place, keeping its label,
    // display state and transform. A failure lands in `status_message` rather
    // than being returned, so that is where the message is read back from.
    state.load_file(path);

    if let Some(message) = state.status_message.clone() {
        return Err(ToolError::new(message));
    }

    // Whichever node is not in `before` is the one that arrived. A reload
    // replaces the node in place with a fresh `ReconId`, so this finds it in
    // both cases and neither has to be special-cased.
    let node = state
        .scene
        .iter()
        .find(|node| !before.contains(&node.id))
        .ok_or_else(|| {
            ToolError::new(format!(
                "Loading {} produced no reconstruction.",
                path.display()
            ))
        })?;
    let label = node.label.clone();
    let mut entry = render::reconstruction(node, state.solo);
    entry
        .as_object_mut()
        .expect("a reconstruction entry is an object")
        .insert("reloaded".into(), json!(already_open));
    announce(
        state,
        format!(
            "{} {label}",
            if already_open { "reloaded" } else { "opened" }
        ),
    );
    Ok(entry)
}

pub(super) fn close_reconstruction(state: &mut AppState, target: CloseTarget) -> JsonReply {
    let closed: Vec<String> = match target {
        CloseTarget::All => {
            let labels: Vec<String> = state.scene.iter().map(|n| n.label.clone()).collect();
            state.close_all();
            labels
        }
        CloseTarget::One(label) => {
            let id = resolve_reconstruction(state, Some(&label))?;
            state.close_node(id);
            vec![label]
        }
    };
    announce(
        state,
        if closed.is_empty() {
            "closed nothing".to_string()
        } else {
            format!("closed {}", closed.join(", "))
        },
    );
    Ok(json!({ "closed": closed }))
}

pub(super) fn select_reconstruction(state: &mut AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    state.select_recon(id);
    selection_reply(state)
}

pub(super) fn select_camera_image(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    selector: &CameraImageSel,
) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let image = resolve_camera_image(state, id, selector)?;
    state.select_image(Some(image));
    selection_reply(state)
}

pub(super) fn select_camera_intrinsics(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    index: usize,
) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let camera = resolve_camera_intrinsics(state, id, index)?;
    state.select_camera(Some(camera));
    selection_reply(state)
}

pub(super) fn select_point(
    state: &mut AppState,
    query: &crate::goto_point::PointQuery,
) -> JsonReply {
    let point = resolve_point(state, query)?;
    state.select_point(point);
    selection_reply(state)
}

pub(super) fn clear_selection(state: &mut AppState, scope: SelectionScope) -> JsonReply {
    match scope {
        // `select_image(None)` keeps the intrinsics and `select_camera(None)`
        // clears the image with it, so "everything" is the camera clear
        // followed by the point.
        SelectionScope::All => {
            state.select_camera(None);
            state.selected_point = None;
        }
        SelectionScope::CameraImage => state.select_image(None),
        SelectionScope::CameraIntrinsics => state.select_camera(None),
        SelectionScope::Point => state.selected_point = None,
    }
    selection_reply(state)
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
    let node = state
        .scene
        .iter_mut()
        .find(|node| node.id == id)
        .expect("just resolved");
    if let Some(visible) = change.visible {
        node.visible = visible;
    }
    if let Some(interactive) = change.interactive {
        node.interactive = interactive;
    }
    if let Some(show) = change.show_points {
        node.show_points = show;
    }
    if let Some(show) = change.show_camera_images {
        node.show_camera_images = show;
    }
    if let Some(show) = change.show_patches {
        node.show_patches = show;
    }
    if let Some(show) = change.show_points_at_infinity {
        node.show_points_at_infinity = show;
    }
    if let Some(tint) = tint {
        node.tint = tint;
    }
    let entry = render::reconstruction(node, solo);
    announce(state, format!("display of {label}"));
    Ok(entry)
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
/// Sets rather than toggles, unlike the GUI's `AppState::toggle_solo`. A toggle
/// is right for a click, where the user can see the current state; an agent
/// issuing one cannot know the outcome without reading the scene first, and a
/// retried call would undo itself.
pub(super) fn set_solo(state: &mut AppState, label: Option<&str>) -> JsonReply {
    match label {
        Some(label) => {
            let id = resolve_reconstruction(state, Some(label))?;
            state.solo = Some(id);
            announce(state, format!("soloed {label}"));
        }
        None => {
            state.solo = None;
            announce(state, "ended the solo");
        }
    }
    Ok(json!({
        "solo": state.solo.and_then(|id| render::label_of(state, id)),
        "scene": state
            .scene
            .iter()
            .map(|node| render::reconstruction(node, state.solo))
            .collect::<Vec<_>>(),
    }))
}
