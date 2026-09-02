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
    selection_reply, CameraImageSel, CloseTarget, DisplayChange, JsonReply, SelectionScope,
    ToolError,
};
use crate::action_log::{interactive_text, tint_text, visibility_text, Kind, Layer};
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
    // display state and transform. Its failure is *returned*, and becomes this
    // tool's refusal — which the drain then records once, as
    // `open_reconstruction failed: …`.
    state.load_file(path).map_err(ToolError::new)?;

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
    let mut entry = render::reconstruction(node, state.solo);
    entry
        .as_object_mut()
        .expect("a reconstruction entry is an object")
        .insert("reloaded".into(), json!(already_open));
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
        // One `AppState` method per scope, and `clear_selection` exists so that
        // "everything" is one Action Log entry rather than the three deselects
        // it is made of.
        SelectionScope::All => state.clear_selection(),
        SelectionScope::CameraImage => state.select_image(None),
        SelectionScope::CameraIntrinsics => state.select_camera(None),
        SelectionScope::Point => state.deselect_point(),
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
    Ok(render::reconstruction(node, solo))
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
            .map(|node| render::reconstruction(node, state.solo))
            .collect::<Vec<_>>(),
    }))
}
