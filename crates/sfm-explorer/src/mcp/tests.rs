// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the command vocabulary.
//!
//! No GPU and no window, which is what the `apply(&mut AppState, &mut Viewer3D,
//! …)` signature is for: every tool but `screenshot` is exercised here against a
//! two-reconstruction scene, and `screenshot` — the one that needs a real frame
//! — belongs in `ui_basic`.
//!
//! Two things these tests are really about, beyond the JSON shapes:
//!
//! - **MCP cannot route around `AppState`'s invariants.** Selecting a camera
//!   image sets its intrinsics; selecting an intrinsics record the selected
//!   image does not use clears that image; selecting a reconstruction drops
//!   another's point. Those are the guarantees that stop two panels showing two
//!   different files' selections, and a second door into the state that skipped
//!   them would be a bug the panels could not defend against.
//! - **A schema and its parser cannot drift.** Every advertised argument name
//!   is one [`tools::parse`] accepts, checked over the whole catalog rather than
//!   tool by tool, so a tool added later is covered by construction.

use serde_json::{json, Map, Value};
use sfmtool_core::SfmrReconstruction;

use super::tools::{self, ToolKind};
use super::{apply, Command, Outcome, ToolError, ToolOutput};
use crate::scene::{PointRef, SceneNode};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

// ── Fixtures ────────────────────────────────────────────────────────────

/// A demo reconstruction grown to `images` images named `<prefix>_<i>.jpg`,
/// resolving to **two** intrinsics records: the first half through camera 0,
/// the rest through camera 1.
///
/// Two cameras because a one-camera reconstruction cannot tell the camera-image
/// and camera-intrinsics selections apart — every image uses the only lens, so
/// every coupling rule looks like a no-op.
///
/// Grown and never shortened: `SfmrReconstruction::demo` fixes the camera ring
/// at [`DEMO_IMAGES`] and its tracks observe all of them, so dropping an image
/// would leave observations pointing past the end of the image list.
fn recon(images: usize, prefix: &str) -> SfmrReconstruction {
    assert!(images >= DEMO_IMAGES, "the demo tracks observe every image");
    let mut recon = SfmrReconstruction::demo(64);
    let template = recon.images[0].clone();
    while recon.images.len() < images {
        recon.images.push(template.clone());
    }
    let second_camera = recon.cameras[0].clone();
    recon.cameras.push(second_camera);
    for (i, image) in recon.images.iter_mut().enumerate() {
        image.name = format!("images/{prefix}_{i:03}.jpg");
        image.camera_index = if i < images / 2 { 0 } else { 1 };
    }
    recon.metadata.camera_count = recon.cameras.len() as u32;
    // Resize the per-image derived tables to match the grown image list.
    recon.rebuild_derived_fields();
    recon
}

/// How many images `SfmrReconstruction::demo` builds its camera ring from.
const DEMO_IMAGES: usize = 8;

/// A scene holding two file-backed reconstructions, `alpha` and `beta`, with
/// `alpha` selected — the state most of these tests start from.
fn two_reconstructions() -> (AppState, Viewer3D) {
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/alpha.sfmr"),
        recon(8, "A"),
    ));
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/beta.sfmr"),
        recon(10, "B"),
    ));
    let alpha = state.scene[0].id;
    state.select_recon(alpha);
    // A laid-out viewport, so the view tools have an aspect ratio to work
    // against and the view block can report the two fixed-axis FOVs.
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    (state, viewer)
}

/// Run one command and unwrap the JSON it produced.
#[track_caller]
fn ok(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Value {
    match apply(state, viewer, command) {
        Outcome::Done(Ok(ToolOutput::Json(value))) => value,
        Outcome::Done(Ok(ToolOutput::Png { .. })) => panic!("expected JSON, got an image"),
        Outcome::Done(Err(e)) => panic!("expected success, got refusal: {e}"),
        Outcome::Deferred(_) => panic!("expected an answer in this frame"),
    }
}

/// Run one command and unwrap the refusal it produced.
#[track_caller]
fn refused(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> ToolError {
    match apply(state, viewer, command) {
        Outcome::Done(Err(e)) => e,
        Outcome::Done(Ok(_)) => panic!("expected a refusal, got success"),
        Outcome::Deferred(_) => panic!("expected a refusal, got a deferral"),
    }
}

/// Parse a tool call the way the transport would, then apply it.
#[track_caller]
fn call(state: &mut AppState, viewer: &mut Viewer3D, name: &str, arguments: Value) -> Value {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    let command = tools::parse(name, Some(&map)).unwrap_or_else(|e| panic!("{name}: {e}"));
    ok(state, viewer, command)
}

/// Parse a tool call the way the transport would and unwrap the refusal, from
/// wherever it came.
///
/// Both halves count as one answer to the agent: whether a call is turned away
/// at the parse or by the viewer is an implementation detail of where the
/// knowledge lives, and a test that fixed which half refused would be asserting
/// that detail rather than the refusal.
#[track_caller]
fn refused_call(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    name: &str,
    arguments: Value,
) -> ToolError {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    match tools::parse(name, Some(&map)) {
        Ok(command) => refused(state, viewer, command),
        Err(error) => error,
    }
}

// ── get_scene ───────────────────────────────────────────────────────────

#[test]
fn get_scene_reports_both_reconstructions_by_label() {
    let (mut state, mut viewer) = two_reconstructions();
    let scene = ok(&mut state, &mut viewer, Command::GetScene);

    let labels: Vec<&str> = scene["scene"]
        .as_array()
        .expect("scene is an array")
        .iter()
        .map(|node| node["label"].as_str().expect("a label"))
        .collect();
    assert_eq!(labels, ["alpha", "beta"]);
    assert_eq!(scene["selection"]["reconstruction_label"], "alpha");
    assert_eq!(scene["solo"], Value::Null);
    assert_eq!(scene["scene"][0]["counts"]["camera_images"], 8);
    assert_eq!(scene["scene"][0]["counts"]["camera_intrinsics"], 2);
    assert_eq!(scene["scene"][0]["display"]["drawn"], true);
}

/// The view block's six stored fields are what `set_view`'s exact form takes
/// back, so a view read out of `get_scene` has to restore the same view.
#[test]
fn the_view_block_round_trips_through_set_view() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [2.0, -3.0, 1.0], "target": [0.0, 0.0, 0.0] }),
    );
    let before = ok(&mut state, &mut viewer, Command::GetScene)["view"].clone();

    // Move somewhere else entirely, then restore from what was read.
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [-9.0, 4.0, 7.0], "target": [1.0, 1.0, 1.0] }),
    );
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({
            "position": before["position"],
            "orientation_wxyz": before["orientation_wxyz"],
            "target_distance": before["target_distance"],
            "world_up": before["world_up"],
            "fov_short_axis_deg": before["fov_short_axis_deg"],
        }),
    );

    let after = ok(&mut state, &mut viewer, Command::GetScene)["view"].clone();
    for field in [
        "position",
        "orientation_wxyz",
        "target_distance",
        "world_up",
        "fov_short_axis_deg",
    ] {
        assert_eq!(after[field], before[field], "{field} did not round-trip");
    }
    // And so does everything derived from them, which is the actual claim:
    // `derived` is not extra state, it is arithmetic over those six.
    assert_eq!(after["derived"]["target"], before["derived"]["target"]);
    assert_eq!(after["derived"]["forward"], before["derived"]["forward"]);
}

// ── The per-entity reads ────────────────────────────────────────────────

#[test]
fn list_camera_images_pages_and_reports_the_total() {
    let (mut state, mut viewer) = two_reconstructions();
    let page = call(
        &mut state,
        &mut viewer,
        "list_camera_images",
        json!({ "reconstruction_label": "alpha", "offset": 2, "limit": 3 }),
    );
    assert_eq!(page["total"], 8);
    assert_eq!(page["offset"], 2);
    let rows = page["camera_images"].as_array().expect("rows");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["index"], 2);
    assert_eq!(rows[0]["name"], "images/A_002.jpg");
    // The images in the first half were built on camera 0.
    assert_eq!(rows[0]["camera_intrinsics_index"], 0);
}

/// An offset past the end is an empty page rather than a refusal: a caller
/// walking a reconstruction should learn it has reached the end from `total`
/// and an empty array, not from an error it has to distinguish from a real one.
#[test]
fn listing_past_the_end_is_an_empty_page() {
    let (mut state, mut viewer) = two_reconstructions();
    let page = call(
        &mut state,
        &mut viewer,
        "list_camera_images",
        json!({ "offset": 99 }),
    );
    assert_eq!(page["total"], 8);
    assert!(page["camera_images"].as_array().expect("rows").is_empty());
}

/// A camera image takes either handle, and both have to land on the same image.
#[test]
fn a_camera_image_is_addressable_by_index_or_by_name() {
    let (mut state, mut viewer) = two_reconstructions();
    let by_index = call(
        &mut state,
        &mut viewer,
        "get_camera_image",
        json!({ "camera_image": 5 }),
    );
    let by_name = call(
        &mut state,
        &mut viewer,
        "get_camera_image",
        json!({ "camera_image": "images/A_005.jpg" }),
    );
    assert_eq!(by_index, by_name);
    assert_eq!(by_index["index"], 5);
    // Second half of the reconstruction, so the other lens.
    assert_eq!(by_index["camera_intrinsics"]["index"], 1);
}

#[test]
fn get_camera_intrinsics_names_its_parameters_and_its_users() {
    let (mut state, mut viewer) = two_reconstructions();
    let lens = call(
        &mut state,
        &mut viewer,
        "get_camera_intrinsics",
        json!({ "reconstruction_label": "alpha", "camera_intrinsics_index": 1 }),
    );
    assert_eq!(lens["index"], 1);
    assert_eq!(lens["camera_image_indices"], json!([4, 5, 6, 7]));
    let params = lens["params"].as_object().expect("a parameter map");
    assert!(
        !params.is_empty(),
        "every camera model has at least a focal length"
    );
    assert!(
        params.values().all(|v| v.is_number()),
        "parameters are numbers keyed by name: {params:?}"
    );
}

/// A bare index resolves against the selected reconstruction; a qualified id
/// names its own, which is what lets a pasted point id reach a different one.
#[test]
fn get_point_takes_both_id_shapes() {
    let (mut state, mut viewer) = two_reconstructions();
    let bare = call(&mut state, &mut viewer, "get_point", json!({ "point": 3 }));
    assert_eq!(bare["reconstruction_label"], "alpha");
    assert_eq!(bare["index"], 3);

    let id = bare["id"].as_str().expect("a point id").to_string();
    let qualified = call(&mut state, &mut viewer, "get_point", json!({ "point": id }));
    assert_eq!(qualified, bare);
    assert!(qualified["track"].as_array().expect("a track").len() > 1);
}

// ── The selection invariants, across the boundary ───────────────────────

#[test]
fn selecting_a_camera_image_selects_the_intrinsics_it_was_shot_through() {
    let (mut state, mut viewer) = two_reconstructions();
    let selection = call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 6 }),
    )["selection"]
        .clone();
    assert_eq!(selection["camera_image"]["index"], 6);
    assert_eq!(selection["camera_intrinsics"]["index"], 1);
    assert_eq!(selection["reconstruction_label"], "alpha");
}

#[test]
fn selecting_a_different_lens_clears_the_camera_image() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 6 }),
    );
    let selection = call(
        &mut state,
        &mut viewer,
        "select_camera_intrinsics",
        json!({ "camera_intrinsics_index": 0 }),
    )["selection"]
        .clone();
    assert_eq!(selection["camera_intrinsics"]["index"], 0);
    assert_eq!(
        selection["camera_image"],
        Value::Null,
        "image 6 uses lens 1, so asking for lens 0 is a different subject"
    );
}

/// Asking for the lens the selected image already uses is not a statement about
/// the image, so the image stays.
#[test]
fn selecting_the_lens_an_image_already_uses_keeps_that_image() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 6 }),
    );
    let selection = call(
        &mut state,
        &mut viewer,
        "select_camera_intrinsics",
        json!({ "camera_intrinsics_index": 1 }),
    )["selection"]
        .clone();
    assert_eq!(selection["camera_image"]["index"], 6);
}

#[test]
fn selecting_a_reconstruction_drops_another_ones_finer_selection() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "reconstruction_label": "beta", "camera_image": 1 }),
    );
    let selection = call(
        &mut state,
        &mut viewer,
        "select_reconstruction",
        json!({ "reconstruction_label": "alpha" }),
    )["selection"]
        .clone();
    assert_eq!(selection["reconstruction_label"], "alpha");
    assert_eq!(selection["camera_image"], Value::Null);
    assert_eq!(selection["camera_intrinsics"], Value::Null);
}

/// The joint states are reached by composing calls, which is what makes one
/// target per call sufficient. The second call filters the image only on the
/// reconstruction, so it keeps it.
#[test]
fn a_camera_image_and_a_point_can_be_selected_together() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 2 }),
    );
    let selection = call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": 7 }),
    )["selection"]
        .clone();
    assert_eq!(selection["camera_image"]["index"], 2);
    assert_eq!(selection["point"]["index"], 7);

    // And the other order reaches the same place.
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": 7 }),
    );
    let selection = call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 2 }),
    )["selection"]
        .clone();
    assert_eq!(selection["camera_image"]["index"], 2);
    assert_eq!(selection["point"]["index"], 7);
}

/// Dismissing a photograph says nothing about the lens — the viewer's own rule,
/// and the reason `clear_selection` has a scope at all.
#[test]
fn clearing_the_camera_image_keeps_its_intrinsics() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 6 }),
    );
    let selection = call(
        &mut state,
        &mut viewer,
        "clear_selection",
        json!({ "scope": "camera_image" }),
    )["selection"]
        .clone();
    assert_eq!(selection["camera_image"], Value::Null);
    assert_eq!(selection["camera_intrinsics"]["index"], 1);
}

#[test]
fn clearing_everything_leaves_only_the_reconstruction() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 6 }),
    );
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": 7 }),
    );
    let selection =
        call(&mut state, &mut viewer, "clear_selection", json!({}))["selection"].clone();
    assert_eq!(selection["camera_image"], Value::Null);
    assert_eq!(selection["camera_intrinsics"], Value::Null);
    assert_eq!(selection["point"], Value::Null);
    assert_eq!(selection["reconstruction_label"], "alpha");
}

// ── Display, and solo ───────────────────────────────────────────────────

#[test]
fn set_reconstruction_display_leaves_the_fields_it_was_not_given() {
    let (mut state, mut viewer) = two_reconstructions();
    let entry = call(
        &mut state,
        &mut viewer,
        "set_reconstruction_display",
        json!({ "reconstruction_label": "beta", "show_points": false, "tint": "Sky Blue" }),
    );
    assert_eq!(entry["display"]["show_points"], false);
    assert_eq!(entry["display"]["tint"], "Sky Blue");
    assert_eq!(entry["display"]["visible"], true);
    assert_eq!(entry["display"]["show_camera_images"], true);
}

#[test]
fn an_unknown_tint_lists_the_palette_and_changes_nothing() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused(
        &mut state,
        &mut viewer,
        Command::SetReconstructionDisplay {
            reconstruction_label: "beta".into(),
            change: super::DisplayChange {
                visible: Some(false),
                tint: Some(Some("Puce".into())),
                ..Default::default()
            },
        },
    );
    assert!(error.0.contains("Sky Blue"), "{error}");
    assert!(
        state.scene[1].visible,
        "a refused call must not have applied its other fields on the way out"
    );
}

/// Solo is scene-level and independent of selection: soloing one reconstruction
/// while another's camera image is selected is a normal state, and the
/// selection must survive it.
#[test]
fn solo_moves_rather_than_accumulating_and_leaves_selection_alone() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 1 }),
    );

    let out = call(
        &mut state,
        &mut viewer,
        "set_solo",
        json!({ "reconstruction_label": "beta" }),
    );
    assert_eq!(out["solo"], "beta");
    assert_eq!(out["scene"][0]["display"]["drawn"], false);
    assert_eq!(out["scene"][1]["display"]["drawn"], true);
    // The eyes themselves are untouched, so ending the solo restores what the
    // user had rather than what the solo left behind.
    assert_eq!(out["scene"][0]["display"]["visible"], true);

    let out = call(
        &mut state,
        &mut viewer,
        "set_solo",
        json!({ "reconstruction_label": "alpha" }),
    );
    assert_eq!(out["solo"], "alpha", "a second solo moves it");

    let out = call(&mut state, &mut viewer, "set_solo", json!({}));
    assert_eq!(out["solo"], Value::Null);

    let selection = ok(&mut state, &mut viewer, Command::GetScene)["selection"].clone();
    assert_eq!(selection["camera_image"]["index"], 1);
    assert_eq!(selection["reconstruction_label"], "alpha");
}

/// A reconstruction hidden by hand and one hidden by another's solo look the
/// same in the viewport and must not look the same in the reply.
#[test]
fn hidden_by_hand_and_hidden_by_a_solo_are_distinguishable() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_reconstruction_display",
        json!({ "reconstruction_label": "alpha", "visible": false }),
    );
    let entry = call(
        &mut state,
        &mut viewer,
        "set_solo",
        json!({ "reconstruction_label": "beta" }),
    )["scene"][0]
        .clone();
    assert_eq!(entry["display"]["visible"], false);
    assert_eq!(entry["display"]["drawn"], false);
}

// ── set_view ────────────────────────────────────────────────────────────

#[test]
fn looking_through_a_camera_image_reports_which_one() {
    let (mut state, mut viewer) = two_reconstructions();
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    );
    assert_eq!(out["view"]["looking_through"]["camera_image_index"], 3);
    assert_eq!(out["view"]["looking_through"]["name"], "images/A_003.jpg");

    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "exit_camera_view": true }),
    );
    assert_eq!(out["view"]["looking_through"], Value::Null);
}

/// `fit` is a statement about the free camera. The Z key's own fit ends its
/// animated transition with the camera view dropped; the MCP form jumps past
/// that transition, so it must land the same state directly.
#[test]
fn a_fit_leaves_camera_view() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    );
    let out = call(&mut state, &mut viewer, "set_view", json!({ "fit": null }));
    assert_eq!(out["view"]["looking_through"], Value::Null);
}

/// The camera must not still be easing when the reply comes back, or an agent
/// that screenshots next photographs the middle of the transition.
#[test]
fn a_view_change_lands_immediately() {
    let (mut state, mut viewer) = two_reconstructions();
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [2.0, -3.0, 1.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0] }),
    );
    assert_eq!(out["view"]["position"], json!([2.0, -3.0, 1.0]));
    let target = out["view"]["derived"]["target"]
        .as_array()
        .expect("a target");
    for (axis, value) in target.iter().enumerate() {
        let value = value.as_f64().expect("a number");
        assert!(value.abs() < 1e-9, "target axis {axis} was {value}");
    }
}

#[test]
fn the_view_forms_are_exclusive() {
    let map: Map<String, Value> = json!({ "fit": null, "exit_camera_view": true })
        .as_object()
        .cloned()
        .expect("an object");
    let error = tools::parse("set_view", Some(&map)).expect_err("rejected");
    assert!(error.0.contains("exclusive"), "{error}");
}

#[test]
fn a_degenerate_look_at_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0] }),
    );
    assert!(error.0.contains("no direction"), "{error}");
}

// ── set_view: the explicit camera, a piece at a time ────────────────────

/// The view block, as a `(position, target, target_distance)` triple, for
/// tests that ask what a placement preserved.
#[track_caller]
fn placement_of(view: &Value) -> ([f64; 3], [f64; 3], f64) {
    let vector = |value: &Value| {
        let numbers: Vec<f64> = value
            .as_array()
            .expect("a vector")
            .iter()
            .map(|n| n.as_f64().expect("a number"))
            .collect();
        [numbers[0], numbers[1], numbers[2]]
    };
    (
        vector(&view["position"]),
        vector(&view["derived"]["target"]),
        view["target_distance"].as_f64().expect("a distance"),
    )
}

#[track_caller]
fn assert_close(actual: [f64; 3], expected: [f64; 3], what: &str) {
    for (axis, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{what}: axis {axis} was {actual} not {expected}"
        );
    }
}

/// A view the explicit camera can be moved a piece at a time from: three
/// distinct coordinates, a known distance, and nothing axis-aligned about it.
fn a_placed_view(state: &mut AppState, viewer: &mut Viewer3D) -> Value {
    call(
        state,
        viewer,
        "set_view",
        json!({ "position": [3.0, 0.0, 4.0], "target": [0.0, 0.0, 0.0] }),
    )["view"]
        .clone()
}

/// `target` alone re-centres the view: the camera keeps its orientation and
/// its distance, and moves to look at the point named.
#[test]
fn a_target_alone_recentres_the_view() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0] }),
    )["view"]
        .clone();

    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);
    assert_eq!(after["target_distance"], before["target_distance"]);
    let (position, target, _) = placement_of(&after);
    assert_close(target, [1.0, 2.0, 3.0], "the target it was given");
    assert_ne!(position, placement_of(&before).0);
}

/// `forward` alone is an orbit, not a turn in place: the camera swings around
/// what it is looking at, which stays where it was.
#[test]
fn a_forward_alone_orbits_rather_than_turning_in_place() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "forward": [0.0, 1.0, 0.0] }),
    )["view"]
        .clone();

    let (position, target, distance) = placement_of(&after);
    assert_close(target, placement_of(&before).1, "the standing target");
    assert_eq!(distance, placement_of(&before).2);
    // Looking along +y from `distance` back along it.
    assert_close(position, [0.0, -distance, 0.0], "swung onto the -y side");
}

/// `position` alone moves the camera and takes the target with it: the
/// orientation is not touched, so the view does not swing round to keep
/// looking at what it was.
#[test]
fn a_position_alone_keeps_the_orientation() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "position": [10.0, 10.0, 10.0] }),
    )["view"]
        .clone();

    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);
    assert_eq!(after["position"], json!([10.0, 10.0, 10.0]));
}

/// `target_distance` alone is a dolly: the target holds still and the camera
/// moves along the view axis.
#[test]
fn a_target_distance_alone_dollies() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target_distance": 12.0 }),
    )["view"]
        .clone();

    assert_eq!(after["orientation_wxyz"], before["orientation_wxyz"]);
    assert_eq!(after["target_distance"], json!(12.0));
    let (position, target, _) = placement_of(&after);
    assert_close(target, placement_of(&before).1, "the standing target");
    // Twelve units back along the same view direction: 3/5, 0, 4/5.
    assert_close(position, [7.2, 0.0, 9.6], "dollied out along the view axis");
}

/// `target` with `forward` views a point from a direction, at the distance the
/// view already had.
#[test]
fn a_target_with_a_forward_keeps_the_distance() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = a_placed_view(&mut state, &mut viewer);
    let after = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [0.0, 0.0, 1.0], "forward": [-2.0, 0.0, 0.0] }),
    )["view"]
        .clone();

    assert_eq!(after["target_distance"], before["target_distance"]);
    let (position, target, distance) = placement_of(&after);
    assert_close(target, [0.0, 0.0, 1.0], "the target it was given");
    assert_close(position, [distance, 0.0, 1.0], "on the +x side, looking -x");
    assert_close(
        [
            after["derived"]["forward"][0].as_f64().expect("a number"),
            after["derived"]["forward"][1].as_f64().expect("a number"),
            after["derived"]["forward"][2].as_f64().expect("a number"),
        ],
        [-1.0, 0.0, 0.0],
        "the direction it was given, normalized",
    );
}

/// A partial placement is a free camera placement like the two whole ones, so
/// it leaves camera view: the background image belongs to a viewpoint the
/// camera has just moved off.
#[test]
fn a_partial_placement_leaves_camera_view() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "look_through": { "camera_image": "images/A_003.jpg" } }),
    );
    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0] }),
    );
    assert_eq!(out["view"]["looking_through"], Value::Null);
}

/// A call that says the same thing twice is refused rather than resolved in
/// some order the agent cannot see.
#[test]
fn an_overdetermined_placement_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let refusals = [
        (
            json!({ "position": [1.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0],
                    "target_distance": 4.0 }),
            "the separation of position and target is the distance",
        ),
        (
            json!({ "position": [1.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0],
                    "forward": [0.0, 0.0, -1.0] }),
            "already fixes the view direction",
        ),
        (
            json!({ "position": [1.0, 0.0, 0.0], "forward": [0.0, 0.0, -1.0],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "target_distance": 4.0 }),
            "there is no direction to derive one from",
        ),
    ];
    for (arguments, expected) in refusals {
        let error = refused_call(&mut state, &mut viewer, "set_view", arguments);
        assert!(error.0.contains(expected), "{error}");
    }
}

/// Every form refuses the arguments it does not read. An argument silently
/// ignored leaves the agent believing it asked for something it did not.
#[test]
fn a_placement_refuses_the_arguments_it_does_not_read() {
    let (mut state, mut viewer) = two_reconstructions();
    let refusals = [
        // No orientation is being derived, so there is no roll to steer.
        (
            json!({ "position": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0] }),
            "nothing to roll",
        ),
        // The roll of a derived view is `up`; `world_up` is the exact form's.
        (
            json!({ "position": [1.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0],
                    "world_up": [0.0, 0.0, 1.0] }),
            "world_up outside the exact form",
        ),
        // The exact form's roll is already in `world_up`.
        (
            json!({ "position": [1.0, 0.0, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "target_distance": 4.0, "up": [0.0, 0.0, 1.0] }),
            "carries its roll in world_up",
        ),
    ];
    for (arguments, expected) in refusals {
        let error = refused_call(&mut state, &mut viewer, "set_view", arguments);
        assert!(error.0.contains(expected), "{error}");
    }
}

/// `forward` is validated like the other directions: it must point somewhere,
/// and it must leave the roll defined.
#[test]
fn a_degenerate_forward_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "forward": [0.0, 0.0, 0.0] }),
    );
    assert!(error.0.contains("forward has no direction"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "forward": [0.0, 0.0, 1.0], "up": [0.0, 0.0, 2.0] }),
    );
    assert!(error.0.contains("roll is undefined"), "{error}");
}

// ── Closing, and stale references ───────────────────────────────────────

#[test]
fn closing_one_reconstruction_leaves_the_other_selected() {
    let (mut state, mut viewer) = two_reconstructions();
    let out = call(
        &mut state,
        &mut viewer,
        "close_reconstruction",
        json!({ "reconstruction_label": "alpha" }),
    );
    assert_eq!(out["closed"], json!(["alpha"]));
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(scene["selection"]["reconstruction_label"], "beta");
    assert_eq!(scene["scene"].as_array().expect("scene").len(), 1);
}

#[test]
fn closing_everything_empties_the_scene() {
    let (mut state, mut viewer) = two_reconstructions();
    let out = call(
        &mut state,
        &mut viewer,
        "close_reconstruction",
        json!({ "all": true }),
    );
    assert_eq!(out["closed"], json!(["alpha", "beta"]));
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert!(scene["scene"].as_array().expect("scene").is_empty());
    assert_eq!(scene["selection"]["reconstruction_label"], Value::Null);
}

/// A ref to a reconstruction that has closed is a refusal naming what *is*
/// loaded — never a silent no-op, and never a bare "not found".
#[test]
fn an_unknown_label_names_what_is_loaded() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused(
        &mut state,
        &mut viewer,
        Command::SelectReconstruction {
            reconstruction_label: "gamma".into(),
        },
    );
    assert!(error.0.contains("gamma"), "{error}");
    assert!(error.0.contains("alpha"), "{error}");
    assert!(error.0.contains("beta"), "{error}");
}

#[test]
fn every_ref_taking_tool_refuses_an_out_of_range_index() {
    let (mut state, mut viewer) = two_reconstructions();
    for command in [
        Command::GetCameraImage {
            reconstruction_label: None,
            camera_image: super::CameraImageSel::Index(99),
        },
        Command::SelectCameraImage {
            reconstruction_label: None,
            camera_image: super::CameraImageSel::Index(99),
        },
        Command::GetCameraIntrinsics {
            reconstruction_label: None,
            camera_intrinsics_index: 99,
        },
        Command::SelectCameraIntrinsics {
            reconstruction_label: None,
            camera_intrinsics_index: 99,
        },
        Command::GetPoint {
            point: crate::goto_point::PointQuery::Index(9_999),
        },
        Command::SelectPoint {
            point: crate::goto_point::PointQuery::Index(9_999),
        },
    ] {
        let described = format!("{command:?}");
        let error = refused(&mut state, &mut viewer, command);
        assert!(
            error.0.contains("out of range"),
            "{described} said {error:?}"
        );
    }
}

#[test]
fn an_unknown_camera_image_name_says_what_a_name_looks_like() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused(
        &mut state,
        &mut viewer,
        Command::GetCameraImage {
            reconstruction_label: None,
            camera_image: super::CameraImageSel::Name("A_003.jpg".into()),
        },
    );
    assert!(error.0.contains("A_003.jpg"), "{error}");
    assert!(error.0.contains("relative path"), "{error}");
}

/// A label survives `Reload from Disk`, which mints a fresh `ReconId` — the
/// whole reason the label rather than the id is the wire handle.
#[test]
fn a_label_still_resolves_after_a_reload() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = state.scene[0].id;
    // Standing in for a real reload, which needs a file on disk: replace the
    // node in place, keeping its label, exactly as `reload_node` does.
    let mut reloaded =
        SceneNode::from_path(std::path::Path::new("/runs/alpha.sfmr"), recon(8, "A"));
    reloaded.label = "alpha".to_string();
    state.scene[0] = reloaded;
    assert_ne!(state.scene[0].id, before, "a reload mints a fresh id");
    state.selected_recon = Some(state.scene[0].id);

    let out = call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 1 }),
    );
    assert_eq!(out["selection"]["camera_image"]["index"], 1);
}

/// The selection can go stale against the reconstruction under it, and the
/// scene report has to survive that rather than panicking on an index that no
/// longer exists.
#[test]
fn get_scene_survives_a_selection_that_has_gone_stale() {
    let (mut state, mut viewer) = two_reconstructions();
    let alpha = state.scene[0].id;
    state.selected_point = Some(PointRef::new(alpha, 9_999));
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(scene["selection"]["point"]["index"], 9999);
}

// ── Status, and the announcement ────────────────────────────────────────

/// Every mutating command says so in the place the viewer already reports what
/// it did, prefixed so a human can tell it from something they did themselves.
#[test]
fn a_mutating_call_announces_itself_in_the_status_line() {
    let (mut state, mut viewer) = two_reconstructions();
    call(
        &mut state,
        &mut viewer,
        "set_solo",
        json!({ "reconstruction_label": "beta" }),
    );
    let message = state.status_message.clone().expect("a status message");
    assert!(message.starts_with("MCP: "), "{message}");
    assert!(message.contains("beta"), "{message}");
}

// ── The schema and its parser ───────────────────────────────────────────

/// Every property a tool advertises is one its parser accepts.
///
/// Over the whole catalog rather than tool by tool, so a tool added later is
/// covered without anyone remembering to add a test — which is the only way
/// this check keeps working.
#[test]
fn every_advertised_argument_is_one_the_parser_knows() {
    for spec in tools::catalog() {
        let properties = spec.schema["properties"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        for name in properties.keys() {
            // Send the argument alone with a deliberately wrong type. A parser
            // that knows the name complains about the *value*; one that does
            // not complains about the name, which is what this rules out.
            let mut arguments = Map::new();
            arguments.insert(name.clone(), json!("<probe>"));
            if let Err(error) = tools::parse(spec.name, Some(&arguments)) {
                assert!(
                    !error.0.contains("has no argument"),
                    "{} advertises {name:?} but rejects it: {error}",
                    spec.name
                );
            }
        }
    }
}

/// A misspelled argument is refused rather than ignored. An agent that believes
/// it asked for something it did not is the failure this surface is shaped to
/// avoid, and `additionalProperties: false` only binds clients that enforce it.
#[test]
fn an_unknown_argument_is_refused_by_name() {
    let arguments = json!({ "reconstruction_labelz": "alpha" })
        .as_object()
        .cloned()
        .expect("an object");
    let error = tools::parse("list_camera_images", Some(&arguments)).expect_err("rejected");
    assert!(error.0.contains("reconstruction_labelz"), "{error}");
    assert!(error.0.contains("reconstruction_label"), "{error}");
}

#[test]
fn every_tool_advertises_an_object_schema_and_a_description() {
    for spec in tools::catalog() {
        assert_eq!(spec.schema["type"], "object", "{}", spec.name);
        assert!(
            spec.schema["additionalProperties"] == json!(false),
            "{} must close its schema",
            spec.name
        );
        assert!(
            spec.description.len() > 40,
            "{} needs a description an agent can choose it from",
            spec.name
        );
    }
}

/// The names are the API, so the vocabulary rule is asserted rather than left
/// to review: no abbreviation, and no bare `camera` or `image` — the two words
/// that each name two things.
#[test]
fn the_wire_vocabulary_holds_across_the_catalog() {
    let mut names: Vec<&str> = Vec::new();
    for spec in tools::catalog() {
        names.push(spec.name);
        let properties = spec.schema["properties"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        for property in properties.keys() {
            assert!(
                !property.contains("recon_") && property != "recon",
                "{}: {property:?} abbreviates reconstruction",
                spec.name
            );
            assert!(
                property != "camera" && property != "image",
                "{}: {property:?} is a word that names two things",
                spec.name
            );
        }
    }
    assert!(!names.iter().any(|name| name.contains("recon_")));
    let unique: std::collections::BTreeSet<&&str> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "tool names must be unique");
}

/// The one tool that cannot answer in the frame it arrives in says so, rather
/// than returning an empty or stale picture.
#[test]
fn screenshot_defers_to_the_frame() {
    let (mut state, mut viewer) = two_reconstructions();
    match apply(
        &mut state,
        &mut viewer,
        Command::Screenshot {
            max_dimension: None,
        },
    ) {
        Outcome::Deferred(super::Deferred::Screenshot { caption, .. }) => {
            // The caption is built here, while the state is still borrowed, so
            // the picture and the description of it are of the same instant.
            assert!(caption.contains("alpha"), "{caption}");
            assert!(caption.contains("beta"), "{caption}");
        }
        _ => panic!("screenshot must defer"),
    }
}

#[test]
fn only_the_reads_are_annotated_read_only() {
    let catalog = tools::catalog();
    let reads: Vec<&str> = catalog
        .iter()
        .filter(|spec| spec.kind == ToolKind::Read)
        .map(|spec| spec.name)
        .collect();
    assert_eq!(
        reads,
        [
            "get_scene",
            "list_camera_images",
            "get_camera_image",
            "get_camera_intrinsics",
            "get_point",
            "screenshot",
        ]
    );
}

// ── The protocol, over a real socket ────────────────────────────────────

/// A running server with an ordinary thread standing in for the GUI.
///
/// This is what [`super::serve`] taking a wake *closure* rather than an
/// `EventLoopProxy` buys: the transport can be driven end to end with no event
/// loop, no window and no GPU, so the handshake, the tool list, a real tool
/// call and the `Origin` rejection are all covered by a normal `cargo test`.
struct RunningServer {
    address: std::net::SocketAddr,
    /// Kept for the life of the test. The stand-in GUI loop ends when the
    /// server's sender is dropped, which happens when the process does.
    _gui: std::thread::JoinHandle<()>,
}

fn running_server() -> RunningServer {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<super::Request>();
    let address = super::serve(0, tx, || {}).expect("an ephemeral port is bindable");

    // The stand-in for `App::drain_mcp`: one owner of the state, applying one
    // command at a time. Exactly the discipline the real frame keeps, which is
    // the point — the transport must not need anything more than this.
    let gui = std::thread::spawn(move || {
        let (mut state, mut viewer) = two_reconstructions();
        while let Some(super::Request { command, reply }) = rx.blocking_recv() {
            let answer = match apply(&mut state, &mut viewer, command) {
                Outcome::Done(answer) => answer,
                // No frame here to render, so the one deferred tool says so
                // rather than hanging the caller until the timeout.
                Outcome::Deferred(_) => Err(ToolError::new("no frame in this harness")),
            };
            let _ = reply.send(answer);
        }
    });

    RunningServer { address, _gui: gui }
}

/// POST one JSON-RPC body to the endpoint and return `(status, body)`.
///
/// Hand-written HTTP/1.1 rather than an HTTP client dev-dependency: a POST with
/// a JSON body and a connection-close response is a dozen lines, and the point
/// of the test is the bytes on the wire.
fn post(server: &RunningServer, body: &str, extra_headers: &[(&str, &str)]) -> (u16, String) {
    use std::io::{Read as _, Write as _};

    let mut stream =
        std::net::TcpStream::connect(server.address).expect("the endpoint is listening");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(20)))
        .expect("a read timeout is settable");
    let mut request = format!(
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nContent-Length: {}\r\n\
         Connection: close\r\n",
        server.address.port(),
        body.len()
    );
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    request.push_str(body);
    stream
        .write_all(request.as_bytes())
        .expect("the request is writable");

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("the response is readable");
    let response = String::from_utf8_lossy(&response).into_owned();
    let status = response
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {response:?}"));
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default();
    (status, body)
}

/// The JSON object of a response body, whether it arrived as
/// `application/json`, chunked, or inside an SSE frame.
fn rpc_body(body: &str) -> Value {
    let json = body
        .lines()
        .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
        .find(|line| line.starts_with('{'))
        .unwrap_or_else(|| panic!("no JSON in {body:?}"));
    serde_json::from_str(json).unwrap_or_else(|e| panic!("{e} in {json:?}"))
}

/// The JSON-RPC `result`, asserting there was no `error` beside it.
fn rpc_result(body: &str) -> Value {
    let parsed = rpc_body(body);
    assert_eq!(parsed["error"], Value::Null, "JSON-RPC error in {parsed}");
    parsed["result"].clone()
}

const PROTOCOL_VERSION: &str = "2025-06-18";

fn initialize_body() -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{PROTOCOL_VERSION}","capabilities":{{}},"clientInfo":{{"name":"sfm-explorer-test","version":"0"}}}}}}"#
    )
}

#[test]
fn the_endpoint_completes_a_handshake_and_advertises_its_tools() {
    let server = running_server();

    let (status, body) = post(&server, &initialize_body(), &[]);
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert!(
        result["capabilities"]["tools"].is_object(),
        "the server declares the tools capability: {result}"
    );
    assert!(
        result["instructions"]
            .as_str()
            .expect("instructions")
            .contains("get_scene"),
        "the instructions point an agent at the first call it should make"
    );

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let listed: Vec<String> = rpc_result(&body)["tools"]
        .as_array()
        .expect("a tool array")
        .iter()
        .map(|tool| tool["name"].as_str().expect("a name").to_string())
        .collect();
    let advertised: Vec<String> = tools::catalog()
        .iter()
        .map(|spec| spec.name.to_string())
        .collect();
    assert_eq!(listed, advertised);
}

#[test]
fn a_tool_call_reaches_the_gui_thread_and_comes_back() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_scene","arguments":{}}}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert_ne!(result["isError"], json!(true), "{result}");
    let labels: Vec<&str> = result["structuredContent"]["scene"]
        .as_array()
        .expect("a scene array")
        .iter()
        .map(|node| node["label"].as_str().expect("a label"))
        .collect();
    assert_eq!(labels, ["alpha", "beta"]);
}

/// A viewer refusal is a tool-level error the agent can read and act on, not a
/// transport failure it has to guess at.
#[test]
fn a_viewer_refusal_arrives_as_an_is_error_result() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"select_reconstruction","arguments":{"reconstruction_label":"gamma"}}}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert_eq!(result["isError"], json!(true), "{result}");
    let text = result["content"][0]["text"].as_str().expect("a message");
    assert!(text.contains("gamma") && text.contains("alpha"), "{text}");
}

/// A page the user has open must not be able to drive their viewer. The
/// allowlist is the endpoint's own loopback origins, so any other `Origin` is
/// rejected before the request reaches a tool.
#[test]
fn a_foreign_origin_is_rejected() {
    let server = running_server();

    let (status, _) = post(
        &server,
        &initialize_body(),
        &[("Origin", "http://evil.example")],
    );
    assert_eq!(status, 403);

    // …while the endpoint's own origin is fine, and so is no origin at all,
    // which is what a real MCP client sends.
    let own = format!("http://127.0.0.1:{}", server.address.port());
    let (status, _) = post(&server, &initialize_body(), &[("Origin", &own)]);
    assert_eq!(status, 200);
    let (status, _) = post(&server, &initialize_body(), &[]);
    assert_eq!(status, 200);
}

/// Arguments that do not fit the advertised schema are the *client's* problem,
/// so they come back as a JSON-RPC error rather than as a tool result.
#[test]
fn a_malformed_argument_is_a_protocol_error() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"list_camera_images","arguments":{"offset":"soon"}}}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let parsed = rpc_body(&body);
    let message = parsed["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("expected a JSON-RPC error in {parsed}"));
    assert!(message.contains("offset"), "{message}");
}

/// The newest revision a real client asks for. The server may negotiate *down*
/// from this — `rmcp` 3.2's own `LATEST` is 2025-11-25 — which is the point:
/// the test asks the way a current client asks and then works with whatever
/// comes back, so an SDK bump changes what is exercised without breaking it.
const NEWEST_PROTOCOL_VERSION: &str = "2026-07-28";

/// `tools/list` carries `ttlMs` and `cacheScope` at the revision a real client
/// negotiates.
///
/// SEP-2549 made both mandatory on a list result. `rmcp` models them as
/// `Option` so one type can serve the older revisions too, which means a
/// handler that simply does not set them compiles, passes a 2025-06-18
/// conformance check, and is then **rejected outright** by a current client:
/// the server shows as connected, its tool list fails schema validation, and
/// its tools are absent for the whole session with only a message about
/// `ttlMs` to explain why. That is exactly how this was found — by attaching
/// Claude Code to it — and this test is why it cannot come back.
#[test]
fn the_tool_list_carries_the_cache_hints_a_current_client_requires() {
    let server = running_server();
    let (status, body) = post(
        &server,
        &format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{NEWEST_PROTOCOL_VERSION}","capabilities":{{}},"clientInfo":{{"name":"t","version":"0"}}}}}}"#
        ),
        &[],
    );
    assert_eq!(status, 200, "{body}");
    let negotiated = rpc_result(&body)["protocolVersion"]
        .as_str()
        .expect("a negotiated version")
        .to_string();

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        &[("MCP-Protocol-Version", &negotiated)],
    );
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert!(
        result["ttlMs"].is_number(),
        "ttlMs must be present and numeric at {negotiated}: {result}"
    );
    assert!(
        matches!(result["cacheScope"].as_str(), Some("public" | "private")),
        "cacheScope must be present and public/private at {negotiated}: {result}"
    );
    assert!(!result["tools"].as_array().expect("tools").is_empty());
}

/// The catalog is not advertised as cacheable. It is fixed within one process
/// but changes across a rebuild, and the viewer exists to be rebuilt — a client
/// holding a cached list across a relaunch would call tools the new binary no
/// longer has.
#[test]
fn the_tool_list_is_not_cacheable() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);
    let (_, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(rpc_result(&body)["ttlMs"], json!(0));
}
