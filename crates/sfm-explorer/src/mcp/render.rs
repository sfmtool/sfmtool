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
use sfmtool_core::patch::cloud::OrientedPatch;
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
            .map(|node| {
                reconstruction(node, state.solo, super::bench::index_files(state, node.id))
            })
            .collect::<Vec<_>>(),
        "selection": selection(state),
        "solo": state.solo.and_then(|id| label_of(state, id)),
        "view": view(state, viewer),
        "status_message": state.status_message(),
        // What the viewer is busy with, or `null`. Beside `status_message` for
        // the reason the revision is: an agent already polling this should not
        // need a second call to learn that an edit it is about to send would be
        // refused. Deliberately the short form -- the phase table is
        // `get_background_task`'s (§ "get_background_task").
        "background_task": super::read::background_summary(state),
        // The Action Log's clock, so an agent that already reads `get_scene`
        // knows whether anything has happened since its last `get_action_log`
        // without a second call. `status_message` stays beside it: the status
        // line is one row of the log, which is a different thing from the log.
        "action_log_revision": state.action_log.revision(),
        "window_title": state.window_title(),
        // The window itself, from the snapshot the frame refreshes — `null`
        // before there is a window to observe. Beside `window_title`, which it
        // subsumes, and it saves an agent a `get_window` before deciding
        // whether a screenshot is worth taking at all.
        "window": state
            .window
            .as_ref()
            .map(|info| super::window::block(info, None)),
    })
}

/// One scene entry: what a reconstruction is, how much of it there is, and how
/// it is being drawn.
///
/// `path` is `null` for a node that came from no file, which is demo data.
///
/// `index_files` is handed in rather than read here, because the files live
/// on `AppState` and one caller holds the node mutably while it builds this
/// entry. It is [`super::bench::index_files`]'s object either way, so the
/// scene reply and `get_bench` say the same thing about the same files.
pub(super) fn reconstruction(node: &SceneNode, solo: Option<ReconId>, index_files: Value) -> Value {
    let recon = node.recon();
    json!({
        "label": node.label,
        "path": node.path.as_ref().map(|p| p.display().to_string()),
        // The hash the version's own point ids are minted from
        // (`scene::version_hash_prefix`), so an agent holding this field and an
        // agent holding an id read off a point are holding the same digits.
        "content_hash": scene::version_hash_prefix(node),
        "counts": {
            "points": node.point_count(),
            // Read the way `scene::visible_stats` reads it, so this number and
            // the one in the viewport's stats overlay are the same number — an
            // agent comparing a reply against a screenshot should not find two.
            "points_at_infinity": node.infinity_point_count(),
            "camera_images": node.image_count(),
            "camera_intrinsics": recon.image_table.cameras.len(),
            // Through the overlay, as `points` is: a deleted point takes its
            // whole track with it and a committed one brings its own, so the
            // base's count is what the file held before the session started
            // rather than what this version holds.
            "observations": node.edited().observation_count(),
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
        // The unit the file declares its lengths in, null for scene units.
        // The unit the *view* is in is the view block's, since the display
        // transform can draw this node at another scale.
        "world_space_unit": recon.metadata.world_space_unit,
        "transformed": node.has_transform(),
        // The display transform in force, always present and reading as the
        // identity for a node that carries none, so an agent that wants the
        // numbers never branches on `transformed` first. Named for
        // `Se3Transform`'s own three fields, the rotation in the `_wxyz` order
        // every other quaternion on this surface is in.
        "transform": transform(node.transform()),
        // How every observation is located: `"sift_files"` names a feature in
        // a `.sift` file beside the workspace, `"embedded_patches"` carries a
        // keypoint inline against a per-point patch frame. It is what
        // convert_to_embedded_patches changes, and what every tool that needs
        // one mode or the other is gated on.
        "feature_source": node.recon().point_set.observations.name(),
        // Narrower than `feature_source`: whether the node also carries the
        // reference bitmaps the surfel renderer textures its patches with. An
        // `embedded_patches` node converted in the viewer has frames and no
        // bitmaps, so this stays false.
        "has_patch_data": node.has_patch_data(),
        // The SIFT index and the cluster patches beside this node's `.sfmr`,
        // in the shape `get_bench` reports them: the files a bench search
        // reads are a fact about the reconstruction, so the scene entry is
        // where an agent finds out whether they are there and still good.
        "index_files": index_files,
    })
}

/// A similarity as `set_reconstruction_transform` takes it back.
fn transform(transform: &sfmtool_core::Se3Transform) -> Value {
    json!({
        "rotation_wxyz": transform.rotation.to_wxyz_array(),
        "translation": vector(&transform.translation),
        "scale": transform.scale,
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
                .and_then(|node| node.recon().image_table.images.get(image.index()))
                .map(|im| im.name.clone()),
        })),
        "camera_intrinsics": state.selected_camera.map(|camera| json!({
            "reconstruction_label": label_of(state, camera.recon),
            "camera_intrinsics_index": camera.index(),
        })),
        "point": state.selected_point.map(|point| json!({
            "reconstruction_label": label_of(state, point.recon),
            "index": point.index(),
            "id": state
                .node(point.recon)
                .map(|node| scene::point_id(node, point.index())),
        })),
    })
}

/// The reply every selection tool returns: the resulting `selection` block.
///
/// All six return it after the fact, so the agent sees what the coupling
/// rules in `AppState` did to its request.
pub(super) fn selection_reply(state: &AppState) -> super::JsonReply {
    Ok(json!({ "selection": selection(state) }))
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
        // What the lengths above are in, and what `set_view`'s `move`
        // converts a stated unit to: null for scene units.
        "world_space_unit": view_world_space_unit(state).ok(),
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
                .and_then(|node| node.recon().image_table.images.get(camera_view.image.index()))
                .map(|im| im.name.clone()),
        })),
    })
}

/// Why the view is in scene units: what is wrong, and the label of the selected
/// reconstruction it is wrong about, if there is one.
pub(super) struct NoViewUnit {
    /// Words that follow "the view is in scene units:".
    pub(super) reason: String,
    /// The selected reconstruction, for a refusal to name what to fix.
    pub(super) label: Option<String>,
}

/// The physical unit the viewport's world is in: the **selected**
/// reconstruction's, as that reconstruction is drawn.
///
/// The viewport's world is shared, and each reconstruction is drawn in it
/// through its display transform, a similarity whose scale `s` stretches every
/// length. So for a selected reconstruction that declares unit `U`, one length
/// of the world is `metres(U) / s` metres. That is the view's unit when it is
/// one of the format's five units, which at `s = 1` it always is, and a scale
/// that maps one unit onto another (`mm` drawn at 0.001 is `m`) names the unit
/// it lands on. A scale that lands between the units, no selection, and a
/// selection that declares no unit are scene units, `null` on the wire.
///
/// A rescaled selection is reported as `null` rather than as a factor, because
/// the field is a unit name on every other reply of this surface and an agent
/// reading it should not have to tell a name from a number; the refusal a
/// physical `move.unit` gets then says what the scale is.
pub(super) fn view_world_space_unit(state: &AppState) -> Result<&'static str, NoViewUnit> {
    let Some(node) = state.selected_recon.and_then(|id| state.node(id)) else {
        return Err(NoViewUnit {
            reason: "no reconstruction is selected".to_string(),
            label: None,
        });
    };
    let label = node.label.as_str();
    let no_unit = |reason: String| NoViewUnit {
        reason,
        label: Some(label.to_string()),
    };
    let Some(unit) = node.recon().metadata.world_space_unit.as_deref() else {
        return Err(no_unit(format!(
            "the selected reconstruction, {label}, declares no world_space_unit"
        )));
    };
    let Some(metres) = sfmtool_core::world_space_unit_in_metres(unit) else {
        return Err(no_unit(format!(
            "the selected reconstruction, {label}, declares world_space_unit {unit:?}, which \
             is not one of {}",
            super::tools::world_space_unit_names()
        )));
    };
    let scale = node.transform().scale;
    let length = metres / scale;
    // The same length, allowing for the rounding a scale such as 0.001 picks up.
    let same = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs());
    sfmtool_core::WORLD_SPACE_UNITS
        .iter()
        .find(|&&(_, metres)| length.is_finite() && same(metres, length))
        .map(|&(name, _)| name)
        .ok_or_else(|| {
            no_unit(format!(
                "the selected reconstruction, {label}, is in {unit} but drawn at a display \
                 scale of {scale}, which makes one length of the view {length} m, none of {}",
                super::tools::world_space_unit_names()
            ))
        })
}

/// One row of `list_camera_images`.
pub(super) fn camera_image_row(
    recon: &SfmrReconstruction,
    index: usize,
    observations: usize,
) -> Value {
    let image = &recon.image_table.images[index];
    json!({
        "index": index,
        "name": image.name,
        "camera_intrinsics_index": image.camera_index as usize,
        "center": point(&image.camera_center()),
        "observations": observations,
    })
}

/// A camera intrinsics record as a name to value map, plus the camera model
/// and sensor size.
///
/// The parameters are a map rather than the model's positional vector, and in
/// [`sfmtool_core::CameraIntrinsics::parameters`] declaration order: a
/// positional vector cannot be read without also shipping the model's
/// parameter order, and an agent will get that wrong. The order matches what
/// `sfm inspect` prints and what the Camera Intrinsics panel shows, so the
/// three can be diffed against each other.
pub(super) fn camera_intrinsics(camera: &sfmtool_core::CameraIntrinsics) -> Value {
    let params: serde_json::Map<String, Value> = camera
        .parameters()
        .into_iter()
        .map(|(name, value)| (name.into_owned(), json!(value)))
        .collect();
    json!({
        "camera_model": camera.model.model_name(),
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
    let mut counts = vec![0usize; recon.image_table.images.len()];
    for observation in &recon.point_set.tracks {
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
    // (`crate::metrics`), which would sort unpredictably and
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

/// A patch's placement, as `get_point` and `get_bench_track` both report it,
/// or null where there is none.
///
/// The axes are unit and `half_extent` is the world half-size along each, so a
/// committed point and the bench track it came from read the same numbers.
/// `normal` is stated rather than left to `u_axis × v_axis`, because it is the
/// outward normal `tilt_bench_patch` takes, and a caller can hand it straight
/// back. For a point at infinity `center` is a bearing and the patch is tangent
/// to the direction sphere; the `at_infinity` beside the block says which.
pub(super) fn placement(patch: Option<&OrientedPatch>) -> Value {
    let Some(patch) = patch else {
        return Value::Null;
    };
    json!({
        "center": point(&patch.center),
        "u_axis": vector(&patch.u_axis),
        "v_axis": vector(&patch.v_axis),
        "normal": vector(&patch.normal()),
        "half_extent": patch.half_extent,
    })
}
