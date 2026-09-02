// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The JSON the read tools emit, and the blocks the write tools echo back.
//!
//! Every shape an agent sees is built here, once. Several tools return the same
//! block — all six selection tools return [`selection`], `get_scene` and
//! `open_reconstruction` both return a [`reconstruction`] entry — and a block
//! that two tools rendered separately would eventually disagree with itself
//! about a field name, which is the drift the wire vocabulary exists to
//! prevent (see `specs/gui/mcp-server.md`).
//!
//! Nothing here reads the wire: these functions take the viewer's own types and
//! produce [`serde_json::Value`]. Parsing the other direction is
//! [`super::tools`].

use nalgebra::{Point3, Vector3};
use serde_json::{json, Value};
use sfmtool_core::SfmrReconstruction;

use crate::scene::{self, ReconId, SceneNode};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

/// The whole `get_scene` reply.
pub(super) fn scene(state: &AppState, viewer: &Viewer3D) -> Value {
    json!({
        "scene": state
            .scene
            .iter()
            .map(|node| reconstruction(node, state.solo))
            .collect::<Vec<_>>(),
        "selection": selection(state),
        "solo": state.solo.and_then(|id| label_of(state, id)),
        "view": view(state, viewer),
        "status_message": state.status_message(),
        "window_title": state.window_title(),
    })
}

/// One scene entry: what a reconstruction is, how much of it there is, and how
/// it is being drawn.
///
/// `path` is `null` for the two node kinds that came from no file — demo data
/// and a derived node such as a resection — which is also exactly when the
/// viewer greys out `Reload from Disk`.
pub(super) fn reconstruction(node: &SceneNode, solo: Option<ReconId>) -> Value {
    let recon = &node.recon;
    json!({
        "label": node.label,
        "path": node.path.as_ref().map(|p| p.display().to_string()),
        "content_hash": scene::hash_prefix(recon),
        "counts": {
            "points": recon.points.len(),
            // Read the way `scene::visible_stats` reads it, so this number and
            // the one in the viewport's stats overlay are the same number — an
            // agent comparing a reply against a screenshot should not find two.
            "points_at_infinity": recon.metadata.infinity_point_count as usize,
            "camera_images": recon.images.len(),
            "camera_intrinsics": recon.cameras.len(),
            "observations": recon.tracks.len(),
        },
        "display": {
            "visible": node.visible,
            // The composition `visible && (no solo, or the solo is me)` —
            // `scene::is_visible`, the single definition the draw loop uses.
            // Reported alongside `visible` rather than instead of it: an agent
            // needs to tell a node it hid from one hidden by another node's
            // solo, and only the pair says which.
            "drawn": scene::is_visible(node, solo),
            "interactive": node.interactive,
            "show_points": node.show_points,
            "show_camera_images": node.show_camera_images,
            "show_patches": node.show_patches,
            "show_points_at_infinity": node.show_points_at_infinity,
            "tint": tint_name(node),
        },
        "transformed": node.has_transform(),
        "has_patch_data": node.has_patch_data(),
    })
}

/// The palette name of a node's tint, or `None` when it is drawn in its own
/// colors.
///
/// The name rather than the RGB triple, because a name is what
/// `set_reconstruction_display` accepts back and what the `Tint` menu shows the
/// human — the two of them have to be able to say the same word.
fn tint_name(node: &SceneNode) -> Option<String> {
    match node.tint {
        scene::NodeTint::Original => None,
        scene::NodeTint::Tint(color) => Some(color.name.to_string()),
    }
}

/// The `selection` block, which all six selection tools return and `get_scene`
/// embeds.
///
/// Each of the three finer selections is rendered whole rather than as a bare
/// index, because a selection can belong to a reconstruction other than the
/// one the agent named — `select_point` on a qualified point id moves the
/// selected reconstruction with it — so each says which one it is in.
pub(super) fn selection(state: &AppState) -> Value {
    json!({
        "reconstruction_label": state.selected_recon.and_then(|id| label_of(state, id)),
        "camera_image": state.selected_image.map(|image| json!({
            "reconstruction_label": label_of(state, image.recon),
            "index": image.index(),
            "name": state
                .node(image.recon)
                .and_then(|node| node.recon.images.get(image.index()))
                .map(|im| im.name.clone()),
        })),
        "camera_intrinsics": state.selected_camera.map(|camera| json!({
            "reconstruction_label": label_of(state, camera.recon),
            "index": camera.index(),
        })),
        "point": state.selected_point.map(|point| json!({
            "reconstruction_label": label_of(state, point.recon),
            "index": point.index(),
            "id": state
                .node(point.recon)
                .map(|node| scene::point_id(&node.recon, point.index())),
        })),
    })
}

/// The `view` block: the viewport camera's stored state, with everything
/// computable from it under `derived`.
///
/// The split is the point. The six stored fields are what `set_view`'s exact
/// form writes back, so a view read here round-trips; `derived` saves the agent
/// the arithmetic and is ignored on the way in. See "The view block" in
/// `specs/gui/mcp-server.md` for why the camera is stored this way.
pub(super) fn view(state: &AppState, viewer: &Viewer3D) -> Value {
    let camera = &viewer.camera;
    let orientation = camera.camera.orientation;
    let position = camera.camera.position;
    let forward = camera.camera.forward();
    let up = camera.camera.up();
    let target = camera.camera.target();
    let [width, height] = viewer.panel_size;
    // Before the 3D panel has been laid out once there is no aspect ratio, so
    // the two fixed-axis fields are absent rather than zero.
    let (fov_horizontal, fov_vertical) = match viewer.panel_aspect() {
        Some(aspect) => {
            let vertical = camera.vertical_fov(aspect);
            let horizontal = ((vertical / 2.0).tan() * aspect).atan() * 2.0;
            (Some(horizontal.to_degrees()), Some(vertical.to_degrees()))
        }
        None => (None, None),
    };

    json!({
        "position": point(&position),
        "orientation_wxyz": [orientation.w, orientation.i, orientation.j, orientation.k],
        "target_distance": camera.camera.target_distance,
        "world_up": vector(&camera.world_up),
        "fov_short_axis_deg": camera.fov.to_degrees(),
        "near": camera.near,
        "derived": {
            "target": point(&target),
            "forward": vector(&forward),
            "up": vector(&up),
            "viewport_px": [width, height],
            "fov_horizontal_deg": fov_horizontal,
            "fov_vertical_deg": fov_vertical,
        },
        "looking_through": viewer.camera_view.as_ref().map(|camera_view| json!({
            "reconstruction_label": label_of(state, camera_view.image.recon),
            "camera_image_index": camera_view.image.index(),
            "name": state
                .node(camera_view.image.recon)
                .and_then(|node| node.recon.images.get(camera_view.image.index()))
                .map(|im| im.name.clone()),
        })),
    })
}

/// One row of `list_camera_images`.
pub(super) fn camera_image_row(
    recon: &SfmrReconstruction,
    index: usize,
    observations: usize,
) -> Value {
    let image = &recon.images[index];
    json!({
        "index": index,
        "name": image.name,
        "camera_intrinsics_index": image.camera_index as usize,
        "center": point(&image.camera_center()),
        "observations": observations,
    })
}

/// A camera intrinsics record as a name to value map, plus the model and
/// sensor size.
///
/// The parameters are a map rather than the model's positional vector, and in
/// [`sfmtool_core::CameraIntrinsics::parameters`] declaration order: a
/// positional vector cannot be read without also shipping the model's
/// parameter order, and an agent will get that wrong. The order matches what
/// `sfm inspect` prints and what the Intrinsics panel shows, so the three can
/// be diffed against each other.
pub(super) fn camera_intrinsics(camera: &sfmtool_core::CameraIntrinsics) -> Value {
    let params: serde_json::Map<String, Value> = camera
        .parameters()
        .into_iter()
        .map(|(name, value)| (name.into_owned(), json!(value)))
        .collect();
    json!({
        "model": camera.model.model_name(),
        "width": camera.width,
        "height": camera.height,
        "params": params,
    })
}

/// How many track observations each image carries, by image index.
///
/// One pass over `tracks` rather than a filter per image: `list_camera_images`
/// reports the count for every image it returns, and a per-image scan would
/// make listing a reconstruction quadratic in its observation count.
pub(super) fn observations_per_image(recon: &SfmrReconstruction) -> Vec<usize> {
    let mut counts = vec![0usize; recon.images.len()];
    for observation in &recon.tracks {
        if let Some(slot) = counts.get_mut(observation.image_index as usize) {
            *slot += 1;
        }
    }
    counts
}

/// Mean, median and 95th percentile of a set of per-observation reprojection
/// errors, or `null` for an empty set.
///
/// Sorts `errors` in place. Percentiles are nearest-rank, which is what a table
/// of a few hundred observations wants: no interpolation between two
/// neighbouring measurements that were never averaged in the first place.
pub(super) fn error_stats(errors: &mut [f32]) -> Value {
    let finite = |e: &f32| e.is_finite();
    if !errors.iter().any(finite) {
        return Value::Null;
    }
    // A behind-the-camera observation reprojects to NaN
    // (`point_track_detail::metrics`), which would sort unpredictably and
    // poison the mean. Those rows drop out of the statistics and stay in the
    // track, where the agent can see them for what they are.
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let n = errors.iter().filter(|e| finite(e)).count();
    let sum: f64 = errors.iter().filter(|e| finite(e)).map(|e| *e as f64).sum();
    json!({
        "mean": sum / n as f64,
        "median": errors[n / 2] as f64,
        "p95": errors[((n * 95) / 100).min(n - 1)] as f64,
    })
}

/// A reconstruction's label, or `None` if the id has left the scene.
pub(super) fn label_of(state: &AppState, id: ReconId) -> Option<String> {
    state.node(id).map(|node| node.label.clone())
}

/// A 3D position as a three-element array.
pub(super) fn point(p: &Point3<f64>) -> Value {
    json!([p.x, p.y, p.z])
}

/// A 3D direction as a three-element array.
pub(super) fn vector(v: &Vector3<f64>) -> Value {
    json!([v.x, v.y, v.z])
}
