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
use sfmtool_core::progress::Level;
use sfmtool_core::SfmrReconstruction;

use super::tools::{self, ToolKind};
use super::{apply, apply_as_agent, Command, Outcome, ToolError, ToolOutput};
use crate::action_log::{ActionLog, Actor, Kind};
use crate::dock::Tab;
use crate::layout::WindowLayout;
use crate::progress::{Collector, Detail};
use crate::scene::{PointRef, SceneNode};
use crate::state::AppState;
use crate::test_support::{FakeWindow, NoWindow};
use crate::viewer_3d::Viewer3D;
use crate::window::{WindowHost, WindowState};

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
    let template = recon.image_table.images[0].clone();
    while recon.image_table.images.len() < images {
        recon.image_table.images.push(template.clone());
    }
    let second_camera = recon.image_table.cameras[0].clone();
    recon.image_table.cameras.push(second_camera);
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name = format!("images/{prefix}_{i:03}.jpg");
        image.camera_index = if i < images / 2 { 0 } else { 1 };
    }
    recon.metadata.camera_count = recon.image_table.cameras.len() as u32;
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
    // The snapshot the frame would have taken at the top of this frame, so the
    // window block and the minimized check have a window to read even where
    // the test hands no host to change one.
    state.window = Some(FakeWindow::default().info());
    (state, viewer)
}

/// A `screenshot` command spelled out, since three of its fields are optional
/// and most tests care about one of them.
fn screenshot(panel: Option<Tab>, hud: bool, max_dimension: Option<u32>) -> Command {
    Command::Screenshot {
        panel,
        hud,
        max_dimension,
    }
}

/// The `Deferred::Screenshot` a command produced, or a panic saying it did not
/// defer.
#[track_caller]
fn deferred_screenshot(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    command: Command,
) -> (super::ScreenshotSource, String) {
    match agent(state, viewer, command) {
        Outcome::Deferred(super::Deferred::Screenshot {
            source, caption, ..
        }) => (source, caption),
        Outcome::Deferred(_) => {
            panic!("expected a screenshot's deferral, got another tool's")
        }
        Outcome::Done(Ok(_)) => panic!("a screenshot must defer, not answer in the frame"),
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
    }
}

/// Apply one command the way the frame does — as the agent, with the Action Log
/// entries that go with it.
///
/// [`apply_as_agent`] rather than bare [`apply`] throughout: attribution is
/// part of what an MCP call *is*, and a test that skipped it would be
/// exercising a path the viewer never takes.
fn agent(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome {
    agent_with(state, viewer, &mut NoWindow, command)
}

/// The same, against a window host — the two window tools, and anything that
/// wants to see what the window was asked.
fn agent_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    command: Command,
) -> Outcome {
    apply_as_agent(state, viewer, host, vec![command])
        .outcomes
        .pop()
        .expect("one command, one outcome")
}

/// Run one command and unwrap the JSON it produced.
#[track_caller]
fn ok(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Value {
    ok_with(state, viewer, &mut NoWindow, command)
}

/// The same, against a window host.
#[track_caller]
fn ok_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    command: Command,
) -> Value {
    match agent_with(state, viewer, host, command) {
        Outcome::Done(Ok(ToolOutput::Json(value))) => value,
        Outcome::Done(Ok(ToolOutput::Png { .. })) => panic!("expected JSON, got an image"),
        Outcome::Done(Err(e)) => panic!("expected success, got refusal: {e}"),
        Outcome::Deferred(_) => panic!("expected an answer in this frame"),
    }
}

/// Run one command and unwrap the refusal it produced.
#[track_caller]
fn refused(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> ToolError {
    refused_with(state, viewer, &mut NoWindow, command)
}

/// The same, against a window host.
#[track_caller]
fn refused_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    command: Command,
) -> ToolError {
    match agent_with(state, viewer, host, command) {
        Outcome::Done(Err(e)) => e,
        Outcome::Done(Ok(_)) => panic!("expected a refusal, got success"),
        Outcome::Deferred(_) => panic!("expected a refusal, got a deferral"),
    }
}

/// Parse a tool call the way the transport would, then apply it.
#[track_caller]
fn call(state: &mut AppState, viewer: &mut Viewer3D, name: &str, arguments: Value) -> Value {
    call_with(state, viewer, &mut NoWindow, name, arguments)
}

/// The same, against a window host.
#[track_caller]
fn call_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    name: &str,
    arguments: Value,
) -> Value {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    let command = tools::parse(name, Some(&map)).unwrap_or_else(|e| panic!("{name}: {e}"));
    ok_with(state, viewer, host, command)
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
    refused_call_with(state, viewer, &mut NoWindow, name, arguments)
}

/// The same, against a window host.
#[track_caller]
fn refused_call_with(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn WindowHost,
    name: &str,
    arguments: Value,
) -> ToolError {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    match tools::parse(name, Some(&map)) {
        Ok(command) => refused_with(state, viewer, host, command),
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

// ── The Image Detail panel's controls ───────────────────────────────────

/// The document `get_image_detail_display` hands back, unwrapped.
#[track_caller]
fn image_detail_display(state: &mut AppState, viewer: &mut Viewer3D) -> Value {
    ok(state, viewer, Command::GetImageDetailDisplay)["image_detail_display"].clone()
}

/// Parse and apply a `set_image_detail_display`, returning its document.
#[track_caller]
fn set_display(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Value {
    call(state, viewer, "set_image_detail_display", arguments)["image_detail_display"].clone()
}

/// A fresh viewer reports what the panel would draw: the feature overlay on,
/// unfiltered, with the intrinsics layer underneath it.
#[test]
fn get_image_detail_display_returns_the_defaults() {
    let (mut state, mut viewer) = two_reconstructions();
    let document = image_detail_display(&mut state, &mut viewer);
    assert_eq!(
        document,
        json!({
            "overlay_mode": "features",
            "max_features": Value::Null,
            "feature_size_px": Value::Null,
            "tracked_only": true,
            "intrinsics": {
                "enabled": true,
                "axes": true,
                "rings": false,
                "distortion": true,
                "distortion_scale": Value::Null,
                "grid_cols": 16,
            },
        })
    );
}

/// A call changes exactly the fields it names, at either level, and the reply
/// is what the next read would say.
#[test]
fn set_image_detail_display_changes_only_what_it_names() {
    let (mut state, mut viewer) = two_reconstructions();
    let reply = set_display(
        &mut state,
        &mut viewer,
        json!({ "overlay_mode": "reproj_error", "max_features": 500 }),
    );
    assert_eq!(reply["overlay_mode"], "reproj_error");
    assert_eq!(reply["max_features"], 500);
    // Untouched, at both levels.
    assert_eq!(reply["tracked_only"], true);
    assert_eq!(reply["intrinsics"]["grid_cols"], 16);
    assert_eq!(reply, image_detail_display(&mut state, &mut viewer));

    let reply = set_display(
        &mut state,
        &mut viewer,
        json!({ "intrinsics": { "rings": true, "distortion_scale": 10, "grid_cols": 32 } }),
    );
    assert_eq!(reply["intrinsics"]["rings"], true);
    assert_eq!(reply["intrinsics"]["distortion_scale"], 10.0);
    assert_eq!(reply["intrinsics"]["grid_cols"], 32);
    // The top level the second call did not mention is still the first's.
    assert_eq!(reply["overlay_mode"], "reproj_error");
    assert_eq!(reply["max_features"], 500);
    assert_eq!(reply, image_detail_display(&mut state, &mut viewer));

    // …and the two doubly-optional fields take an explicit null back to their
    // "no filter" state.
    let reply = set_display(
        &mut state,
        &mut viewer,
        json!({ "max_features": Value::Null, "intrinsics": { "distortion_scale": Value::Null } }),
    );
    assert_eq!(reply["max_features"], Value::Null);
    assert_eq!(reply["intrinsics"]["distortion_scale"], Value::Null);
}

/// Every mode the panel offers is reachable by its wire name, and comes back
/// spelled the same way.
#[test]
fn every_overlay_mode_round_trips_over_the_wire() {
    let (mut state, mut viewer) = two_reconstructions();
    for mode in crate::state::OverlayMode::ALL {
        let reply = set_display(
            &mut state,
            &mut viewer,
            json!({ "overlay_mode": mode.wire_name() }),
        );
        assert_eq!(reply["overlay_mode"], mode.wire_name());
        assert_eq!(state.feature_display.overlay_mode, mode);
    }
}

/// The size filter is one thing on the wire because it is one checkbox in the
/// toolbar: setting it writes all four fields, so the toolbar's per-frame
/// re-derivation finds the drag values it also wrote and changes nothing.
#[test]
fn a_feature_size_filter_survives_the_toolbars_next_frame() {
    let (mut state, mut viewer) = two_reconstructions();
    let reply = set_display(
        &mut state,
        &mut viewer,
        json!({ "feature_size_px": { "min": 2.0, "max": 40.0 } }),
    );
    assert_eq!(reply["feature_size_px"], json!({ "min": 2.0, "max": 40.0 }));
    // All four, which is what the next frame re-derives the pair from.
    assert_eq!(state.feature_display.min_feature_size, Some(2.0));
    assert_eq!(state.feature_display.max_feature_size, Some(40.0));
    assert_eq!(state.feature_display.min_feature_size_value, 2.0);
    assert_eq!(state.feature_display.max_feature_size_value, 40.0);

    // The toolbar, every frame: ticked, both options come from the drag
    // values; unticked, both are cleared. Neither is a change here.
    let before = crate::state::ImageDetailDisplay::snapshot(
        &state.feature_display,
        &state.intrinsics_display,
    );
    let feature = &mut state.feature_display;
    let ticked = feature.min_feature_size.is_some() || feature.max_feature_size.is_some();
    assert!(ticked);
    feature.min_feature_size = Some(feature.min_feature_size_value);
    feature.max_feature_size = Some(feature.max_feature_size_value);
    let after = crate::state::ImageDetailDisplay::snapshot(
        &state.feature_display,
        &state.intrinsics_display,
    );
    assert_eq!(before, after, "the toolbar's re-derivation moved something");

    // Null turns the filter off and leaves the drag values where they were,
    // which is what unticking the checkbox does.
    let reply = set_display(
        &mut state,
        &mut viewer,
        json!({ "feature_size_px": Value::Null }),
    );
    assert_eq!(reply["feature_size_px"], Value::Null);
    assert_eq!(state.feature_display.min_feature_size_value, 2.0);
    assert_eq!(state.feature_display.max_feature_size_value, 40.0);
}

/// Every vocabulary here is static, so every refusal is at the parse — and a
/// refused call has applied none of its good fields on the way out.
#[test]
fn set_image_detail_display_refuses_a_value_it_cannot_show() {
    let (mut state, mut viewer) = two_reconstructions();

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "overlay_mode": "heatmap", "tracked_only": false }),
    );
    assert!(error.0.contains("heatmap"), "{error}");
    for mode in crate::state::OverlayMode::ALL {
        assert!(error.0.contains(mode.wire_name()), "{error}");
    }
    assert!(state.feature_display.tracked_only, "a refusal applied half");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "tracked_only": false, "intrinsics": { "distortion_scale": 4 } }),
    );
    assert!(error.0.contains("1, 2, 3, 5, 10, 20, 50"), "{error}");
    assert!(state.feature_display.tracked_only, "a refusal applied half");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "intrinsics": { "grid_cols": 20 } }),
    );
    assert!(error.0.contains("8, 12, 16, 24, 32"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "max_features": 0 }),
    );
    assert!(error.0.contains("overlay_mode"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "feature_size_px": { "min": 40.0, "max": 2.0 } }),
    );
    assert!(error.0.contains("no feature at all"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "feature_size_px": { "min": -1.0, "max": 2.0 } }),
    );
    assert!(error.0.contains("zero or more"), "{error}");

    // A call with nothing in it has asked for nothing.
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({}),
    );
    assert!(error.0.contains("nothing to change"), "{error}");

    // Nothing above touched the document.
    assert_eq!(
        image_detail_display(&mut state, &mut viewer),
        json!({
            "overlay_mode": "features",
            "max_features": Value::Null,
            "feature_size_px": Value::Null,
            "tracked_only": true,
            "intrinsics": {
                "enabled": true,
                "axes": true,
                "rings": false,
                "distortion": true,
                "distortion_scale": Value::Null,
                "grid_cols": 16,
            },
        })
    );
}

/// One `Display` entry per field the call changed, in the words the panel's
/// own controls record under — and nothing for a field set to the value it
/// already had.
#[test]
fn set_image_detail_display_records_one_display_entry_per_changed_field() {
    let (mut state, mut viewer) = quiet_scene();
    call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "intrinsics": { "enabled": false } }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].text, "Intrinsics off");
    assert_eq!(entries[0].kind, Kind::Display);
    assert_eq!(entries[0].actor, Actor::Mcp);

    // Three fields, three runs, three rows in field order. Folding by kind
    // alone — the rule the log started with — kept only the last of them.
    let (mut state, mut viewer) = quiet_scene();
    let before = state.action_log.revision();
    call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({
            "overlay_mode": "track_length",
            "intrinsics": { "rings": true, "distortion_scale": 10.0 },
        }),
    );
    assert_eq!(state.action_log.revision() - before, 3);
    assert_eq!(
        state
            .action_log
            .entries()
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>(),
        [
            "Overlay Track Length",
            "Intrinsics rings on",
            "Distortion scale ×10"
        ]
    );

    // A repeat of one of those fields folds into that field's row and leaves
    // the other two standing.
    call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "intrinsics": { "distortion_scale": 20.0 } }),
    );
    assert_eq!(
        state
            .action_log
            .entries()
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>(),
        [
            "Overlay Track Length",
            "Intrinsics rings on",
            "Distortion scale ×20"
        ]
    );

    // A field set to the value it had is not a change.
    let (mut state, mut viewer) = quiet_scene();
    call(
        &mut state,
        &mut viewer,
        "set_image_detail_display",
        json!({ "overlay_mode": "features", "tracked_only": true }),
    );
    assert_eq!(
        state.action_log.entries().count(),
        0,
        "{:?}",
        state.action_log.entries().collect::<Vec<_>>()
    );
}

/// Every one of this tool's vocabularies is static, so its refusals are
/// protocol errors: they never reach the viewer, change nothing, and — like
/// every protocol error — leave no Action Log row behind (§ "Errors").
///
/// The kind a `SetImageDetailDisplay` would be filed under is still
/// `Kind::Display`, which is where the entries a *successful* call writes go.
#[test]
fn a_refused_display_call_never_reaches_the_viewer() {
    let (mut state, mut viewer) = quiet_scene();
    tools::parse(
        "set_image_detail_display",
        json!({ "intrinsics": { "grid_cols": 20 } }).as_object(),
    )
    .expect_err("off the ladder");
    assert_eq!(
        state.action_log.entries().count(),
        0,
        "a protocol error was logged"
    );
    assert_eq!(state.intrinsics_display.grid_cols, 16);

    assert_eq!(
        Command::SetImageDetailDisplay {
            change: super::ImageDetailDisplayChange {
                tracked_only: Some(false),
                ..Default::default()
            },
        }
        .kind(),
        Kind::Display
    );
    // …and the read is a query, like every other read on the surface.
    ok(&mut state, &mut viewer, Command::GetImageDetailDisplay);
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Query("get_image_detail_display"));
    assert_eq!(entries[0].text, "get_image_detail_display");
    assert_eq!(entries[0].actor, Actor::Mcp);
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
fn every_ref_taking_tool_refuses_an_index_that_names_nothing() {
    let (mut state, mut viewer) = two_reconstructions();
    // The image and camera tables are dense and are refused by their count; a
    // point index is a place in a version's own index space, which an edit
    // leaves holes in, so what it is refused by is naming no live point.
    for (command, expected) in [
        (
            Command::GetCameraImage {
                reconstruction_label: None,
                camera_image: super::CameraImageSel::Index(99),
            },
            "out of range",
        ),
        (
            Command::SelectCameraImage {
                reconstruction_label: None,
                camera_image: super::CameraImageSel::Index(99),
            },
            "out of range",
        ),
        (
            Command::GetCameraIntrinsics {
                reconstruction_label: None,
                camera_intrinsics_index: 99,
            },
            "out of range",
        ),
        (
            Command::SelectCameraIntrinsics {
                reconstruction_label: None,
                camera_intrinsics_index: 99,
            },
            "out of range",
        ),
        (
            Command::GetPoint {
                point: crate::goto_point::PointQuery::Index(9_999),
            },
            "no point 9999",
        ),
        (
            Command::SelectPoint {
                point: crate::goto_point::PointQuery::Index(9_999),
            },
            "no point 9999",
        ),
    ] {
        let described = format!("{command:?}");
        let error = refused(&mut state, &mut viewer, command);
        assert!(error.0.contains(expected), "{described} said {error:?}");
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

/// A label survives a node being replaced under a fresh `ReconId` — the whole
/// reason the label rather than the id is the wire handle.
#[test]
fn a_label_still_resolves_after_a_node_is_replaced() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = state.scene[0].id;
    // A node swapped out under its label, which is what the handle has to
    // survive.
    let mut replacement =
        SceneNode::from_path(std::path::Path::new("/runs/alpha.sfmr"), recon(8, "A"));
    replacement.label = "alpha".to_string();
    state.scene[0] = replacement;
    assert_ne!(state.scene[0].id, before, "a replacement mints a fresh id");
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

// ── Status, and the Action Log ──────────────────────────────────────────

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
    let message = state.status_message().expect("a status message");
    assert!(message.starts_with("MCP: "), "{message}");
    assert!(message.contains("beta"), "{message}");
}

/// The scene the attribution tests start from: `alpha` selected with one of its
/// images picked, so that every command below is a *change* — only a change is
/// logged — and the log emptied of everything setting that up put in it.
fn quiet_scene() -> (AppState, Viewer3D) {
    let (mut state, viewer) = two_reconstructions();
    let alpha = state.scene[0].id;
    state.select_image(Some(crate::scene::ImageRef::new(alpha, 1)));
    state.action_log.clear();
    (state, viewer)
}

/// One entry per mutating command, attributed to the agent — and the ambient
/// actor back where it was, so the user's next click is not filed as the
/// agent's.
#[test]
fn each_mutating_command_records_one_entry_as_the_agent() {
    let commands: Vec<Command> = vec![
        Command::SelectReconstruction {
            reconstruction_label: "beta".into(),
        },
        Command::SelectCameraImage {
            reconstruction_label: Some("beta".into()),
            camera_image: super::CameraImageSel::Index(2),
        },
        Command::SelectCameraIntrinsics {
            reconstruction_label: Some("beta".into()),
            camera_intrinsics_index: 0,
        },
        Command::SelectPoint {
            point: crate::goto_point::PointQuery::Index(3),
        },
        Command::ClearSelection {
            scope: super::SelectionScope::All,
        },
        Command::SetSolo {
            reconstruction_label: Some("beta".into()),
        },
        Command::SetReconstructionDisplay {
            reconstruction_label: "beta".into(),
            change: super::DisplayChange {
                visible: Some(false),
                ..Default::default()
            },
        },
        Command::SetImageDetailDisplay {
            change: super::ImageDetailDisplayChange {
                tracked_only: Some(false),
                ..Default::default()
            },
        },
        Command::SetTimingDetail { enabled: true },
        Command::SetView {
            view: super::ViewCommand::Fit {
                reconstruction_label: None,
            },
        },
        Command::CloseReconstruction {
            target: super::CloseTarget::One("beta".into()),
        },
    ];
    for command in commands {
        let (mut state, mut viewer) = quiet_scene();
        let name = command.tool_name();
        ok(&mut state, &mut viewer, command);
        let entries: Vec<_> = state.action_log.entries().collect();
        assert_eq!(entries.len(), 1, "{name} recorded {entries:?}");
        assert_eq!(entries[0].actor, Actor::Mcp, "{name}: {entries:?}");
        assert!(!entries[0].failed, "{name}: {entries:?}");
        assert!(
            !matches!(entries[0].kind, Kind::Query(_)),
            "{name} was filed as a query: {entries:?}"
        );
        assert_eq!(
            state.action_log.actor(),
            Actor::User,
            "{name} left the actor moved"
        );
    }
}

/// A read is logged too — from the command, since it changes no state and has
/// no state method to log through — but it never reaches the status line, so an
/// agent polling `get_scene` does not read its own polling back as the viewer's
/// status.
#[test]
fn each_read_only_command_records_a_query_the_status_line_ignores() {
    let commands: Vec<Command> = vec![
        Command::GetScene,
        Command::ListCameraImages {
            reconstruction_label: None,
            offset: 0,
            limit: 5,
        },
        Command::GetCameraImage {
            reconstruction_label: None,
            camera_image: super::CameraImageSel::Index(0),
        },
        Command::GetCameraIntrinsics {
            reconstruction_label: None,
            camera_intrinsics_index: 0,
        },
        Command::GetPoint {
            point: crate::goto_point::PointQuery::Index(1),
        },
        Command::GetImageDetailDisplay,
        Command::GetTimingDetail,
    ];
    for command in commands {
        let (mut state, mut viewer) = quiet_scene();
        let name = command.tool_name();
        ok(&mut state, &mut viewer, command);
        let entries: Vec<_> = state.action_log.entries().collect();
        assert_eq!(entries.len(), 1, "{name} recorded {entries:?}");
        assert_eq!(entries[0].kind, Kind::Query(name), "{name}: {entries:?}");
        assert_eq!(entries[0].actor, Actor::Mcp, "{name}: {entries:?}");
        assert_eq!(
            state.status_message(),
            None,
            "{name} reached the status line"
        );
    }
}

/// A `screenshot` is logged in the frame it was *applied*, not when the pixels
/// come back, so its line sits in order with the commands around it.
#[test]
fn a_deferred_screenshot_is_logged_when_it_is_drained() {
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    let outcome = agent(&mut state, &mut viewer, screenshot(None, true, None));
    assert!(matches!(outcome, Outcome::Deferred(_)), "not deferred");
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    // The window, at the size the window snapshot reports.
    assert_eq!(
        entries[0].text, "screenshot window 1920×1080",
        "{entries:?}"
    );
    // …and `max_dimension` is reported at the size the picture comes back at.
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    agent(&mut state, &mut viewer, screenshot(None, true, Some(640)));
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "screenshot window 640×360"
    );
}

/// A screenshot is not a value an agent is scrubbing through: it is a picture
/// it took and presumably looked at, so every one taken is its own row however
/// fast they arrive.
#[test]
fn every_screenshot_taken_is_its_own_row() {
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    for _ in 0..3 {
        agent(&mut state, &mut viewer, screenshot(None, true, None));
    }
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 3, "{entries:?}");
    assert!(
        entries.iter().all(|entry| entry.run.is_none()),
        "a screenshot carries a run: {entries:?}"
    );
    // …while the read beside it polls into one row, as every other read does.
    let (mut state, mut viewer) = quiet_scene();
    for _ in 0..3 {
        ok(&mut state, &mut viewer, Command::GetScene);
    }
    assert_eq!(state.action_log.entries().count(), 1);
}

/// A refusal is one failed entry, in the words the agent was given — not two,
/// one from the method and one from the drain.
#[test]
fn a_refused_command_records_one_failed_entry_carrying_the_refusal() {
    let (mut state, mut viewer) = quiet_scene();
    let error = refused(
        &mut state,
        &mut viewer,
        Command::SelectReconstruction {
            reconstruction_label: "globl".into(),
        },
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].failed, "{entries:?}");
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(
        entries[0].text,
        format!("select_reconstruction failed: {error}")
    );
    // The status line shows it too, where before only a success reached it.
    assert_eq!(
        state.status_message(),
        Some(format!("MCP: select_reconstruction failed: {error}"))
    );
}

/// `load_file` returns its failure rather than writing it anywhere, so an
/// unreadable path is one entry and not two.
#[test]
fn open_reconstruction_on_an_unreadable_path_records_one_failed_entry() {
    let (mut state, mut viewer) = quiet_scene();
    refused(
        &mut state,
        &mut viewer,
        Command::OpenReconstruction {
            path: std::path::PathBuf::from("/runs/there-is-no-such-file.sfmr"),
        },
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].failed, "{entries:?}");
    assert!(
        entries[0]
            .text
            .starts_with("open_reconstruction failed: Failed to load "),
        "{entries:?}"
    );
}

/// An open goes to a worker like every other background operation, and the
/// answer it comes back with is the reconstruction entry it always gave.
#[test]
fn open_reconstruction_defers_and_answers_with_the_reconstruction() {
    let dir = temp_dir("open_defers");
    std::fs::write(dir.join(".sfm-workspace.json"), "{}").unwrap();
    let path = dir.join("scene.sfmr");
    SfmrReconstruction::demo(16).save(&path).unwrap();
    let (mut state, mut viewer) = quiet_scene();

    let open = |state: &mut AppState, viewer: &mut Viewer3D| {
        let map = json!({ "path": path.display().to_string() })
            .as_object()
            .cloned()
            .expect("an object");
        let command = tools::parse("open_reconstruction", Some(&map)).expect("a well-formed call");
        let pending = match agent(state, viewer, command) {
            Outcome::Deferred(super::Deferred::Background(pending)) => pending,
            Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
            _ => panic!("the open must defer"),
        };
        assert_eq!(pending.operation_name, "Open");
        state.finish_background_task();
        match super::edit::background_reply(state, &pending).expect("it finished") {
            Ok(ToolOutput::Json(value)) => value,
            Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
            Err(e) => panic!("expected success, got refusal: {e}"),
        }
    };
    let reply = open(&mut state, &mut viewer);
    assert_eq!(reply["label"], "scene", "{reply}");
    assert_eq!(reply["already_open"], false);
    // A second open of the same path is a second node, and says so.
    let reply = open(&mut state, &mut viewer);
    assert_eq!(reply["label"], "scene (2)", "{reply}");
    assert_eq!(reply["already_open"], true);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Reading the Action Log back ─────────────────────────────────────────

/// A `get_action_log` command with every field spelled out.
fn action_log_read(since_revision: u64, actors: &[Actor]) -> Command {
    Command::GetActionLog {
        since_revision,
        limit: super::read::ACTION_LOG_DEFAULT_LIMIT,
        actors: actors.to_vec(),
        detail: false,
    }
}

/// Every entry text in a `get_action_log` reply, oldest first.
fn log_texts(reply: &Value) -> Vec<&str> {
    reply["entries"]
        .as_array()
        .expect("entries is an array")
        .iter()
        .map(|entry| entry["text"].as_str().expect("a text"))
        .collect()
}

/// The reply is a transcript: oldest first, each row saying when, who, what
/// kind and whether it failed, with the log's clock beside it.
#[test]
fn the_action_log_read_returns_a_transcript_with_the_clock_beside_it() {
    let (mut state, mut viewer) = quiet_scene();
    state
        .action_log
        .record(Kind::File, "Opened alpha from /runs/alpha.sfmr");
    let expected_revision = state.action_log.revision();
    let reply = ok(&mut state, &mut viewer, action_log_read(0, &Actor::ALL));

    assert_eq!(reply["revision"], json!(expected_revision));
    assert_eq!(reply["oldest_revision"], json!(expected_revision));
    assert_eq!(reply["truncated"], json!(false));
    let entries = reply["entries"].as_array().expect("an array");
    assert_eq!(entries.len(), 1, "{reply}");
    let row = &entries[0];
    assert_eq!(row["revision"], json!(expected_revision));
    assert_eq!(row["actor"], "user");
    assert_eq!(row["kind"], "file");
    assert_eq!(row["failed"], json!(false));
    assert_eq!(row["text"], "Opened alpha from /runs/alpha.sfmr");
    assert!(row["tool"].is_null(), "only a query row carries a tool");
    assert!(
        row["took_ms"].is_null(),
        "a row the viewer has not drawn a frame for does not claim a cost yet",
    );
    // RFC 3339 in the panel's zone, so the agent's time and the human's row
    // are the same time.
    let at = row["at"].as_str().expect("a timestamp");
    assert!(at.len() >= 24 && at.contains('T'), "{at}");
}

/// The read an agent makes most: what the human did, with none of the agent's
/// own rows in it.
#[test]
fn the_actors_filter_separates_the_human_from_the_agent() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record(Kind::Selection, "Selected image a");
    ok(
        &mut state,
        &mut viewer,
        Command::SelectReconstruction {
            reconstruction_label: "beta".into(),
        },
    );
    ok(&mut state, &mut viewer, Command::GetScene);

    let human = ok(&mut state, &mut viewer, action_log_read(0, &[Actor::User]));
    assert_eq!(log_texts(&human), ["Selected image a"]);

    let agent_rows = ok(&mut state, &mut viewer, action_log_read(0, &[Actor::Mcp]));
    let entries = agent_rows["entries"].as_array().expect("an array");
    assert_eq!(
        log_texts(&agent_rows),
        [
            "Selected reconstruction beta",
            "get_scene",
            // The human's read above was the agent's own call, and a read of
            // the log that the log did not record would be the one action the
            // human could not see.
            "get_action_log since 0",
        ],
        "the agent audits itself, queries included"
    );
    // A query row carries its tool beside the kind, which stays the one word.
    assert_eq!(entries[1]["kind"], "query");
    assert_eq!(entries[1]["tool"], "get_scene");
    assert!(entries[0]["tool"].is_null());

    // Omitted, the filter is every actor, which is the whole log.
    let all = call(&mut state, &mut viewer, "get_action_log", json!({}));
    assert_eq!(all["entries"].as_array().expect("an array").len(), 4);
}

/// A call that can return nothing by construction has asked no question, and a
/// misspelled actor is a typo rather than a filter.
#[test]
fn an_empty_or_unknown_actors_list_is_refused_at_the_parse() {
    let (mut state, mut viewer) = quiet_scene();
    let empty = refused_call(
        &mut state,
        &mut viewer,
        "get_action_log",
        json!({ "actors": [] }),
    );
    assert!(empty.0.contains("empty actors"), "{empty}");
    assert!(
        empty.0.contains("user") && empty.0.contains("mcp"),
        "{empty}"
    );

    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "get_action_log",
        json!({ "actors": ["human"] }),
    );
    assert!(unknown.0.contains("\"human\""), "{unknown}");
    assert!(unknown.0.contains("viewer"), "{unknown}");
}

/// Past `limit` the reply says so, and the last entry's revision is where the
/// next call picks up.
#[test]
fn the_read_truncates_at_its_limit_and_continues_from_the_last_revision() {
    let (mut state, mut viewer) = quiet_scene();
    for i in 0..5 {
        state.action_log.record(Kind::File, format!("Opened {i}"));
    }
    let first = ok(
        &mut state,
        &mut viewer,
        Command::GetActionLog {
            since_revision: 0,
            limit: 2,
            actors: Actor::ALL.to_vec(),
            detail: false,
        },
    );
    assert_eq!(log_texts(&first), ["Opened 0", "Opened 1"]);
    assert_eq!(first["truncated"], json!(true));

    let from = first["entries"][1]["revision"]
        .as_u64()
        .expect("a revision");
    let rest = ok(
        &mut state,
        &mut viewer,
        action_log_read(from, &[Actor::User]),
    );
    assert_eq!(log_texts(&rest), ["Opened 2", "Opened 3", "Opened 4"]);
    assert_eq!(rest["truncated"], json!(false));

    // A reader that is up to date is told nothing, which is the poll that
    // costs nothing. Under `["user"]`, which is what keeps the agent's own
    // polling rows out of the answer.
    let caught_up = ok(
        &mut state,
        &mut viewer,
        action_log_read(
            rest["revision"].as_u64().expect("a revision"),
            &[Actor::User],
        ),
    );
    assert_eq!(log_texts(&caught_up), Vec::<&str>::new());
}

/// A limit above the cap is capped rather than honoured: the surface is not a
/// data channel.
#[test]
fn a_limit_above_the_cap_is_capped() {
    let (mut state, mut viewer) = quiet_scene();
    for i in 0..super::read::ACTION_LOG_MAX_LIMIT + 5 {
        state.action_log.record(Kind::File, format!("Opened {i}"));
    }
    let reply = ok(
        &mut state,
        &mut viewer,
        Command::GetActionLog {
            since_revision: 0,
            limit: 100_000,
            actors: Actor::ALL.to_vec(),
            detail: false,
        },
    );
    assert_eq!(
        reply["entries"].as_array().expect("an array").len(),
        super::read::ACTION_LOG_MAX_LIMIT
    );
    assert_eq!(reply["truncated"], json!(true));
}

/// A read of the log that the log did not record would be the one action the
/// human could not see.
#[test]
fn the_action_log_read_records_itself_as_a_query() {
    let (mut state, mut viewer) = quiet_scene();
    ok(&mut state, &mut viewer, action_log_read(512, &Actor::ALL));
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Query("get_action_log"));
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(entries[0].text, "get_action_log since 512");
    // A poll is one row however often it asks, like every other read.
    ok(&mut state, &mut viewer, action_log_read(512, &Actor::ALL));
    assert_eq!(state.action_log.entries().count(), 1);
}

/// One field on `get_scene`, so an agent that already polls it knows whether
/// anything happened without a second call.
#[test]
fn get_scene_carries_the_action_log_revision() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record(Kind::File, "Opened alpha");
    let expected = state.action_log.revision();
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(scene["action_log_revision"], json!(expected));
    assert!(
        scene["status_message"].is_string(),
        "the status line is a different thing from the log and stays: {scene}"
    );
}

// ── The breakdown on the wire ───────────────────────────────────────────
//
// What these are really about is one property: the breakdown an agent reads is
// the breakdown the human is looking at. The Action Log panel is where a
// reader meets an expanded entry, and a wire that ordered its events
// differently, folded them differently or dropped what a row says would be
// telling the two of them different stories about the same operation. So the
// assertion most of these end on is [`assert_the_panel_agrees`], which rebuilds
// the panel's rows out of the JSON and compares them with the rows the panel
// would draw.

/// A `get_action_log` that asks for each row's breakdown as well.
fn action_log_detail(since_revision: u64) -> Command {
    Command::GetActionLog {
        since_revision,
        limit: super::read::ACTION_LOG_DEFAULT_LIMIT,
        actors: Actor::ALL.to_vec(),
        detail: true,
    }
}

/// The reply's row whose text begins `prefix`: the operation a test drove,
/// picked out of the query rows the reads themselves leave.
#[track_caller]
fn row_starting(reply: &Value, prefix: &str) -> Value {
    reply["entries"]
        .as_array()
        .expect("entries is an array")
        .iter()
        .find(|row| {
            row["text"]
                .as_str()
                .is_some_and(|text| text.starts_with(prefix))
        })
        .unwrap_or_else(|| panic!("no row starting {prefix:?} in {reply}"))
        .clone()
}

/// A phase row that ran once and said nothing, for the entries these build by
/// hand.
fn phase(name: &'static str, depth: u8, ms: u64) -> Detail {
    folded(name, depth, ms, 1, None, None)
}

/// A phase row as the collector leaves one that ran more than once, with
/// whatever its runs said at either end.
fn folded(
    name: &'static str,
    depth: u8,
    ms: u64,
    runs: u32,
    note: Option<&str>,
    note_last: Option<&str>,
) -> Detail {
    Detail::Phase {
        name,
        depth,
        took: std::time::Duration::from_millis(ms),
        cpu: None,
        note: note.map(str::to_string),
        note_last: note_last.map(str::to_string),
        runs,
    }
}

/// A message row, for the entries these build by hand.
fn message(level: Level, depth: u8, text: &str) -> Detail {
    Detail::Message {
        level,
        depth,
        text: text.to_string(),
    }
}

/// One wire row's breakdown, each event spelled the way the panel draws its
/// row: the indent, the marker, and the name with its count and its note.
///
/// Rebuilt from the JSON rather than read off the panel, which is the whole
/// point of it: everything the panel puts in a row has to be on the wire for
/// this to come out equal to what the panel drew.
fn wire_detail_lines(row: &Value) -> Vec<String> {
    row["detail"]
        .as_array()
        .expect("the row carries a detail array")
        .iter()
        .map(|event| {
            let indent = " ".repeat(2 * event["depth"].as_u64().expect("a depth") as usize);
            match event["kind"].as_str().expect("a kind") {
                "phase" => {
                    let runs = match event["runs"].as_u64() {
                        Some(runs) => format!(" x{runs}"),
                        None => String::new(),
                    };
                    let note = match event["note"].as_str() {
                        Some(note) => format!("  {note}"),
                        None => String::new(),
                    };
                    format!(
                        "{indent}{}{runs}{note}",
                        event["name"].as_str().expect("a name")
                    )
                }
                "message" => format!(
                    "{indent}{} {}",
                    match event["level"].as_str().expect("a level") {
                        "info" => "\u{2022}",
                        "warn" => "!",
                        other => panic!("unknown message level {other:?}"),
                    },
                    event["text"].as_str().expect("a text")
                ),
                other => panic!("unknown detail kind {other:?}"),
            }
        })
        .collect()
}

/// The wire's breakdown is the panel's, row for row.
///
/// `elsewhere` is the one row of the panel's that is not an event: the wire
/// carries it as a field beside `took_ms`, so it is lifted out of the drawn
/// rows here and its presence checked on both sides.
#[track_caller]
fn assert_the_panel_agrees(state: &AppState, row: &Value) {
    let revision = row["revision"].as_u64().expect("a revision");
    let entry = state
        .action_log
        .entries()
        .find(|entry| entry.revision == revision)
        .expect("the row is an entry the log still holds");
    let mut drawn = ActionLog::drawn_detail(entry);
    match (
        drawn.iter().position(|line| line == "elsewhere"),
        row.get("elsewhere_ms"),
    ) {
        (Some(at), Some(_)) => {
            drawn.remove(at);
        }
        (None, None) => {}
        (drawn_at, wire) => {
            panic!("the panel draws elsewhere at {drawn_at:?} and the wire carries {wire:?}: {row}")
        }
    }
    assert_eq!(wire_detail_lines(row), drawn, "{row}");
}

/// A frame's own events, as the viewer's frame collector leaves them.
fn frame_events() -> Vec<Detail> {
    let frame = Collector::new(false);
    {
        let uploads = frame.phase("uploads");
        drop(uploads.phase("points"));
    }
    drop(frame.phase("scene render"));
    frame.take()
}

/// Stamp what is waiting, so the entries have a cost and an account to close.
fn settle(state: &mut AppState) {
    state
        .action_log
        .settle(std::time::Instant::now(), frame_events());
}

/// An `undo`, which is an operation that names its stages: the step, the
/// selection following it and the caches the version it left was holding.
fn undo_something(state: &mut AppState, viewer: &mut Viewer3D) {
    call(
        state,
        viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    call(
        state,
        viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
}

/// The breakdown is several times the size of the row it hangs off, so an
/// agent reading the log to find out what happened is not handed it.
#[test]
fn the_breakdown_is_omitted_until_it_is_asked_for() {
    let (mut state, mut viewer) = editable();
    undo_something(&mut state, &mut viewer);
    settle(&mut state);

    let plain = ok(&mut state, &mut viewer, action_log_read(0, &Actor::ALL));
    let row = row_starting(&plain, "Undo:");
    assert!(row["detail"].is_null(), "{row}");
    assert!(row["elsewhere_ms"].is_null(), "{row}");
    assert!(
        row["took_ms"].is_number(),
        "the cost is not the detail: {row}"
    );

    let asked = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&asked, "Undo:");
    assert!(row["detail"].is_array(), "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// The stages an operation named, at the depths it named them, in the order a
/// reader sees them.
#[test]
fn the_breakdown_carries_the_stages_at_their_depths() {
    let (mut state, mut viewer) = editable();
    undo_something(&mut state, &mut viewer);

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Undo:");
    let stages: Vec<(&str, u64)> = row["detail"]
        .as_array()
        .expect("a detail array")
        .iter()
        .map(|event| {
            assert_eq!(event["kind"], "phase", "{event}");
            (
                event["name"].as_str().expect("a name"),
                event["depth"].as_u64().expect("a depth"),
            )
        })
        .collect();
    assert_eq!(
        stages,
        [
            ("undo", 0),
            ("history step", 1),
            ("selection follow", 1),
            ("forget images", 1),
        ],
        "{row}",
    );
    // Every stage ran once, so none of them claims a count.
    assert!(
        row["detail"]
            .as_array()
            .expect("a detail array")
            .iter()
            .all(|event| event["runs"].is_null()),
        "{row}",
    );
    assert_the_panel_agrees(&state, &row);
}

/// `elsewhere` is what makes the breakdown add up, so it arrives with the cost
/// it closes the account of and not before.
#[test]
fn elsewhere_sits_beside_took_and_waits_for_the_frame() {
    let (mut state, mut viewer) = editable();
    undo_something(&mut state, &mut viewer);

    let before = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&before, "Undo:");
    assert!(row["took_ms"].is_null(), "{row}");
    assert!(
        row["elsewhere_ms"].is_null(),
        "an entry with no cost yet has no account to close: {row}",
    );
    assert_the_panel_agrees(&state, &row);

    settle(&mut state);
    let after = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&after, "Undo:");
    let took = row["took_ms"].as_f64().expect("a cost");
    let elsewhere = row["elsewhere_ms"].as_f64().expect("an elsewhere");
    let named: f64 = row["detail"]
        .as_array()
        .expect("a detail array")
        .iter()
        .filter(|event| event["depth"] == 0)
        .map(|event| event["ms"].as_f64().expect("a cost"))
        .sum();
    assert!(
        (took - named - elsewhere).abs() < 1e-6,
        "the breakdown does not reconcile with the headline: {row}",
    );
    assert_the_panel_agrees(&state, &row);
}

/// A row the collector folded says how many runs it is, and one that ran once
/// says nothing, and a note that has two ends reaches the agent with both,
/// because one end alone is a claim about the row only one run supports.
#[test]
fn a_folded_row_carries_its_runs_and_both_ends_of_its_note() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record_done(
        Kind::Edit,
        std::time::Instant::now(),
        "Bundle adjusted alpha",
        vec![
            phase("materialise", 0, 30),
            folded("round", 1, 50, 3, Some("trim 50 px"), Some("trim 4 px")),
            folded("linearise", 2, 12, 180, Some("reused"), None),
        ],
    );

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Bundle adjusted alpha");
    let events = row["detail"].as_array().expect("a detail array");
    assert!(events[0]["runs"].is_null(), "{row}");
    assert!(events[0]["note"].is_null(), "{row}");
    assert_eq!(events[1]["runs"], json!(3), "{row}");
    assert_eq!(events[1]["note"], "trim 50 px ... trim 4 px", "{row}");
    assert_eq!(events[2]["runs"], json!(180), "{row}");
    // Runs that came back to what the first one said leave one note, not a
    // span that is not there.
    assert_eq!(events[2]["note"], "reused", "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// What the operation said, marked the way the panel marks it and nested under
/// the stage it was said inside.
#[test]
fn a_message_carries_its_level_and_its_depth() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record_done(
        Kind::Edit,
        std::time::Instant::now(),
        "Bundle adjusted alpha",
        vec![
            phase("solve", 0, 30),
            message(Level::Info, 1, "3 rounds, trim 50/12/4 px"),
            message(Level::Warn, 1, "3 points left unsupported and were dropped"),
        ],
    );

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Bundle adjusted alpha");
    let events = row["detail"].as_array().expect("a detail array");
    assert_eq!(events[1]["kind"], "message", "{row}");
    assert_eq!(events[1]["level"], "info", "{row}");
    assert_eq!(events[1]["depth"], json!(1), "{row}");
    assert_eq!(events[1]["text"], "3 rounds, trim 50/12/4 px", "{row}");
    assert_eq!(events[2]["level"], "warn", "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// The order is the panel's, which is not the order the events were recorded
/// in: the operation's own stages, then the frame's overhead under the rule,
/// with `elsewhere` between them as a field rather than a row.
#[test]
fn the_overhead_comes_last_on_the_wire_as_it_does_in_the_panel() {
    let (mut state, mut viewer) = quiet_scene();
    state.action_log.record_done(
        Kind::Edit,
        std::time::Instant::now(),
        "Opened alpha",
        vec![phase("open", 0, 30), phase("read", 1, 20)],
    );
    settle(&mut state);

    let reply = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&reply, "Opened alpha");
    let names: Vec<&str> = row["detail"]
        .as_array()
        .expect("a detail array")
        .iter()
        .map(|event| event["name"].as_str().expect("a name"))
        .collect();
    assert_eq!(
        names,
        [
            "open",
            "read",
            ActionLog::OVERHEAD,
            "uploads",
            "points",
            "scene render",
        ],
        "{row}",
    );
    assert!(row["elsewhere_ms"].is_number(), "{row}");
    assert_the_panel_agrees(&state, &row);
}

/// The level decides what is *recorded*, so it changes the next operation and
/// nothing already in the log.
#[test]
fn set_timing_detail_changes_what_the_next_operation_records() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a" }),
    );
    let overview = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&overview, "Bundle adjusted run_a");
    assert!(
        !wire_detail_lines(&row)
            .iter()
            .any(|line| line.trim_start().starts_with("linearise")),
        "the kernel's detailed stages were recorded with detail off: {row}",
    );
    assert_the_panel_agrees(&state, &row);

    let revision = state.action_log.revision();
    assert_eq!(
        call(
            &mut state,
            &mut viewer,
            "set_timing_detail",
            json!({ "enabled": true })
        )["timing_detail"]["enabled"],
        json!(true),
    );
    adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a" }),
    );
    settle(&mut state);

    let detailed = ok(&mut state, &mut viewer, action_log_detail(revision));
    let row = row_starting(&detailed, "Bundle adjusted run_a");
    assert!(
        wire_detail_lines(&row)
            .iter()
            .any(|line| line.trim_start().starts_with("linearise")),
        "detail was on and the kernel's stages did not reach the entry: {row}",
    );
    // The strongest thing here: whatever a real solve reported, the agent's
    // breakdown and the human's expanded row are the same rows in the same
    // order.
    assert_the_panel_agrees(&state, &row);

    // And the entry recorded before the change keeps the detail it was
    // recorded with: nothing is re-timed.
    let unchanged = ok(&mut state, &mut viewer, action_log_detail(0));
    let row = row_starting(&unchanged, "Bundle adjusted run_a");
    assert!(
        !wire_detail_lines(&row)
            .iter()
            .any(|line| line.trim_start().starts_with("linearise")),
        "an entry was re-timed by the level changing under it: {row}",
    );
}

/// The pair reads back what it set, through the log the checkbox writes to, so
/// an agent and the window cannot hold two different levels.
#[test]
fn get_timing_detail_reads_back_what_set_timing_detail_wrote() {
    let (mut state, mut viewer) = quiet_scene();
    assert_eq!(
        call(&mut state, &mut viewer, "get_timing_detail", json!({}))["timing_detail"]["enabled"],
        json!(false),
        "detail is off until it is asked for",
    );

    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": true }),
    );
    assert!(state.action_log.detailed_timing(), "the level did not move");
    assert_eq!(
        call(&mut state, &mut viewer, "get_timing_detail", json!({}))["timing_detail"]["enabled"],
        json!(true),
    );

    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": false }),
    );
    assert!(!state.action_log.detailed_timing());
}

/// The same property the checkbox holds, because it is the same call: one
/// `Display` entry when the value changes, and none when it does not.
#[test]
fn set_timing_detail_records_the_change_and_nothing_else() {
    let (mut state, mut viewer) = quiet_scene();
    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": true }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Display);
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(entries[0].text, "Detailed timing on");

    // Handed the value it already has, it records nothing.
    call(
        &mut state,
        &mut viewer,
        "set_timing_detail",
        json!({ "enabled": true }),
    );
    assert_eq!(state.action_log.entries().count(), 1);

    // `enabled` is what the tool is for, so a call without it is a call that
    // asked for nothing.
    let error = refused_call(&mut state, &mut viewer, "set_timing_detail", json!({}));
    assert!(error.0.contains("enabled"), "{error}");
}

// ── Screenshots of the window and of a panel ────────────────────────────

/// Give a panel's node the body rectangle a frame's egui pass would have laid
/// out, since a headless dock has never been drawn.
fn lay_out(state: &mut AppState, panel: Tab, rect: egui::Rect) {
    let path = state.dock.find_tab(&panel).expect("the panel is docked");
    state
        .dock
        .leaf_mut(path.node_path())
        .expect("the path names a leaf")
        .viewport = rect;
}

/// Bring `panel` to the front of its node without a log entry, so a test whose
/// subject is the *first* entry still starts from an empty log.
fn raise(state: &mut AppState, panel: Tab) {
    state.action_log.mute();
    state.show_panel(panel);
    state.action_log.unmute();
}

/// A 320 × 180 point body at (100, 40), which at the fixture's scale factor of
/// 1.5 is 480 × 270 physical pixels at (150, 60).
fn body() -> egui::Rect {
    egui::Rect::from_min_size(egui::pos2(100.0, 40.0), egui::vec2(320.0, 180.0))
}

/// No panel is the whole window, at the size the window snapshot reports, with
/// the frame description kept because the 3D view is in the picture.
#[test]
fn a_screenshot_with_no_panel_photographs_the_window() {
    let (mut state, mut viewer) = two_reconstructions();
    let (source, caption) =
        deferred_screenshot(&mut state, &mut viewer, screenshot(None, true, None));
    assert_eq!(source, super::ScreenshotSource::Window);
    assert!(caption.starts_with("The window, 1920×1080."), "{caption}");
    assert!(caption.contains("alpha"), "{caption}");
}

/// A panel defers with its tab — the rectangle is resolved at readback, not
/// here — and its caption and log line name the panel and its last laid-out
/// size.
#[test]
fn a_screenshot_of_a_panel_defers_with_the_tab_and_names_it() {
    let (mut state, mut viewer) = quiet_scene();
    // Behind the viewport in the stock grid, and a picture wants the tab in
    // front.
    raise(&mut state, Tab::ImageDetail);
    lay_out(&mut state, Tab::ImageDetail, body());
    let (source, caption) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::ImageDetail), true, None),
    );
    assert_eq!(source, super::ScreenshotSource::Panel(Tab::ImageDetail));
    assert_eq!(caption, "The Image Detail panel, 480×270.");
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "screenshot image_detail 480×270"
    );
}

/// The 3D Viewer's crop keeps the frame description, because that picture is
/// of the scene.
#[test]
fn a_screenshot_of_the_viewport_keeps_the_frame_description() {
    let (mut state, mut viewer) = two_reconstructions();
    lay_out(&mut state, Tab::Viewer3D, body());
    let (source, caption) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::Viewer3D), true, None),
    );
    assert_eq!(source, super::ScreenshotSource::Panel(Tab::Viewer3D));
    assert!(
        caption.starts_with("The 3D Viewer panel, 480×270."),
        "{caption}"
    );
    assert!(caption.contains("points"), "{caption}");
}

/// A panel that is not drawn cannot be photographed, and the two ways that
/// happens have two different fixes.
#[test]
fn a_panel_that_is_not_drawn_is_refused_naming_show_panel() {
    let (mut state, mut viewer) = quiet_scene();
    state.hide_panel(Tab::ActionLog);
    let closed = refused(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::ActionLog), true, None),
    );
    assert!(closed.0.contains("closed"), "{closed}");
    assert!(closed.0.contains("show_panel"), "{closed}");
    assert!(closed.0.contains("action_log"), "{closed}");

    // Behind a sibling: a picture of it would be a picture of the tab in
    // front, so the refusal names that tab too.
    let behind = refused(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::IntrinsicsDetail), true, None),
    );
    assert!(behind.0.contains("Track View"), "{behind}");
    assert!(behind.0.contains("show_panel"), "{behind}");

    // An unknown name is the panel vocabulary's own refusal, listing all seven.
    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "screenshot",
        json!({ "panel_name": "viewport" }),
    );
    assert!(unknown.0.contains("viewer_3d") && unknown.0.contains("action_log"));
}

/// Track View is a panel like any other on the wire, and the two panels whose
/// place it took are unknown names, refused with the list of the ones that
/// exist rather than read as an alias.
#[test]
fn track_view_is_the_panel_name_and_the_retired_ones_are_refused() {
    let (mut state, mut viewer) = quiet_scene();
    let shown = call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "track_view" }),
    );
    assert_eq!(
        panel(&shown, "track_view")["active"],
        json!(true),
        "{shown}"
    );
    let arguments = json!({ "panel_name": "track_view" });
    let parsed = tools::parse("screenshot", arguments.as_object()).expect("a panel name");
    assert!(
        matches!(
            parsed,
            Command::Screenshot {
                panel: Some(Tab::TrackView),
                ..
            }
        ),
        "{parsed:?}"
    );
    for retired in ["point_track", "track_edit"] {
        for tool in ["show_panel", "hide_panel", "screenshot"] {
            let why = refused_call(
                &mut state,
                &mut viewer,
                tool,
                json!({ "panel_name": retired }),
            );
            assert!(why.0.contains(&format!("\"{retired}\"")), "{tool}: {why}");
            assert!(why.0.contains("track_view"), "{tool}: {why}");
        }
    }
}

/// Both checks are against the dock at *apply* time, so a `show_panel` earlier
/// in the same batch satisfies them.
#[test]
fn show_panel_then_a_screenshot_of_it_is_accepted_in_one_batch() {
    let (mut state, mut viewer) = quiet_scene();
    let outcomes = apply_as_agent(
        &mut state,
        &mut viewer,
        &mut NoWindow,
        vec![
            Command::ShowPanel {
                panel: Tab::IntrinsicsDetail,
            },
            screenshot(Some(Tab::IntrinsicsDetail), true, None),
        ],
    )
    .outcomes;
    assert!(
        matches!(outcomes[1], Outcome::Deferred(_)),
        "the raised panel was still refused"
    );
}

/// `hud: false` asks for the picture underneath what egui painted, and only the
/// 3D Viewer has one.
#[test]
fn hud_false_reads_the_render_target_and_is_refused_elsewhere() {
    let (mut state, mut viewer) = quiet_scene();
    viewer.panel_size = [1280, 720];
    let (source, caption) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::Viewer3D), false, None),
    );
    assert_eq!(source, super::ScreenshotSource::ViewportRender);
    assert!(
        caption.starts_with("The 3D Viewer panel without its HUD, 1280×720."),
        "{caption}"
    );
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "screenshot viewer_3d 1280×720 without HUD"
    );

    for arguments in [
        json!({ "panel_name": "image_detail", "hud": false }),
        json!({ "hud": false }),
    ] {
        let error = refused_call(&mut state, &mut viewer, "screenshot", arguments.clone());
        assert!(
            error.0.contains("hud applies to the 3D Viewer only"),
            "{arguments}: {error}"
        );
    }

    // `hud: true` is accepted anywhere and changes nothing.
    raise(&mut state, Tab::ImageDetail);
    let (with_hud, _) = deferred_screenshot(
        &mut state,
        &mut viewer,
        screenshot(Some(Tab::ImageDetail), true, None),
    );
    assert_eq!(with_hud, super::ScreenshotSource::Panel(Tab::ImageDetail));
}

/// The crop rectangle is the dock's own points scaled to pixels, clipped to the
/// frame — pure arithmetic over the dock, which is what puts it under headless
/// test.
#[test]
fn the_crop_rectangle_scales_the_docks_points_and_clips_to_the_frame() {
    let (mut state, _) = two_reconstructions();
    lay_out(&mut state, Tab::ImageDetail, body());
    assert_eq!(
        super::panel_crop(&state.dock, Tab::ImageDetail, 1.5, [1920, 1080]),
        Some([150, 60, 480, 270])
    );
    // A frame smaller than the layout clips rather than reading past the end.
    assert_eq!(
        super::panel_crop(&state.dock, Tab::ImageDetail, 1.5, [400, 200]),
        Some([150, 60, 250, 140])
    );
    // A panel the dock has never laid out has no rectangle to crop to, and one
    // that lies wholly outside the frame has none either.
    assert_eq!(
        super::panel_crop(&state.dock, Tab::SceneGraph, 1.5, [1920, 1080]),
        None
    );
    assert_eq!(
        super::panel_crop(&state.dock, Tab::ImageDetail, 1.5, [100, 100]),
        None
    );
}

// ── The layout tools ────────────────────────────────────────────────────

/// A panel by its wire name, for a reply's `panels` map.
#[track_caller]
fn panel(reply: &Value, name: &str) -> Value {
    reply["panels"][name].clone()
}

/// A fake in one of the four states, with the snapshot to match.
fn windowed(state: WindowState) -> (AppState, Viewer3D, FakeWindow) {
    let host = FakeWindow::in_state(state);
    let (mut app_state, viewer) = two_reconstructions();
    app_state.observe_window(&host);
    (app_state, viewer, host)
}

/// The reply's three views of one arrangement: the document the file holds,
/// the live window, and the panels.
#[test]
fn get_window_layout_returns_the_file_the_window_and_the_panels() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let reply = ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout);

    // `window_layout` is the file itself, not a rendering of it: read back
    // through the file's own parser it is what Save Layout… would write.
    let text = serde_json::to_string(&reply["window_layout"]).expect("a document");
    assert_eq!(
        WindowLayout::from_json(&text).expect("the reply parses as a layout file"),
        state.window_layout()
    );

    // The block beside it is the observation, with every monitor.
    assert_eq!(reply["window"]["state"], "normal");
    let names: Vec<&str> = reply["window"]["monitors"]
        .as_array()
        .expect("a monitor list")
        .iter()
        .map(|monitor| monitor["name"].as_str().expect("a name"))
        .collect();
    assert_eq!(names, ["DISPLAY1", "DISPLAY2"], "the current monitor first");

    let panels = reply["panels"].as_object().expect("a panels map");
    assert_eq!(panels.len(), Tab::ALL.len(), "all of them, always");
    for tab in Tab::ALL {
        assert_eq!(
            panel(&reply, tab.wire_name())["open"],
            json!(true),
            "{} is open in the default layout",
            tab.wire_name()
        );
    }
    // The default layout has three multi-tab nodes, so four of the nine sit
    // behind a sibling rather than in front of it.
    let active: Vec<&str> = Tab::ALL
        .iter()
        .filter(|tab| panel(&reply, tab.wire_name())["active"] == json!(true))
        .map(|tab| tab.wire_name())
        .collect();
    assert_eq!(
        active,
        [
            "scene",
            "background_task",
            "viewer_3d",
            "image_browser",
            "track_view"
        ]
    );
}

/// A maximized window's document says what it will restore to; the block says
/// what it is. The difference is the information.
#[test]
fn the_document_carries_the_normal_rectangle_and_the_block_the_current_one() {
    let (mut state, mut viewer) = two_reconstructions();
    let mut host = FakeWindow::default();
    state.observe_window(&host);
    host.set_state(WindowState::Maximized);
    host.inner_size = [3840, 2160];

    let reply = ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout);
    assert_eq!(reply["window"]["state"], "maximized");
    assert_eq!(reply["window"]["inner_size"], json!([3840, 2160]));
    assert_eq!(reply["window_layout"]["window"]["state"], "maximized");
    assert_eq!(
        reply["window_layout"]["window"]["inner_size"],
        json!([1920, 1080])
    );
}

/// The panels half of the answer is still an answer where there is no window.
#[test]
fn get_window_layout_answers_without_a_window() {
    let (mut state, mut viewer) = two_reconstructions();
    let reply = ok(&mut state, &mut viewer, Command::GetWindowLayout);
    assert_eq!(reply["window"], Value::Null);
    assert_eq!(reply["window_layout"]["window"], Value::Null);
    assert_eq!(panel(&reply, "scene")["open"], json!(true));
}

#[test]
fn hide_panel_closes_and_show_panel_takes_it_home() {
    let (mut state, mut viewer) = two_reconstructions();
    let hidden = call(
        &mut state,
        &mut viewer,
        "hide_panel",
        json!({ "panel_name": "action_log" }),
    );
    assert_eq!(
        panel(&hidden, "action_log"),
        json!({ "open": false, "active": false })
    );
    assert!(!state.is_panel_open(Tab::ActionLog));

    // Idempotent, as the method is: hiding a closed panel succeeds and
    // changes nothing. Both tools *set* rather than toggle, for the reason
    // `set_solo` does.
    let again = call(
        &mut state,
        &mut viewer,
        "hide_panel",
        json!({ "panel_name": "action_log" }),
    );
    assert_eq!(again, hidden);

    // Shown again it goes home to its default group-mate's node — behind the
    // Image Browser — and comes to the front of it.
    let shown = call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "action_log" }),
    );
    assert_eq!(
        panel(&shown, "action_log"),
        json!({ "open": true, "active": true })
    );
    assert_eq!(
        panel(&shown, "image_browser"),
        json!({ "open": true, "active": false })
    );
}

#[test]
fn show_panel_on_an_open_panel_raises_it_and_moves_nothing_else() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = ok(&mut state, &mut viewer, Command::GetWindowLayout);
    let raised = call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "track_view" }),
    );
    assert_eq!(panel(&raised, "track_view")["active"], json!(true));
    assert_eq!(panel(&raised, "image_detail")["active"], json!(false));
    for tab in Tab::ALL {
        assert_eq!(
            panel(&raised, tab.wire_name())["open"],
            panel(&before, tab.wire_name())["open"],
            "{} moved",
            tab.wire_name()
        );
    }
}

#[test]
fn an_unknown_panel_name_lists_the_seven() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "show_panel",
        json!({ "panel_name": "viewer3d" }),
    );
    assert!(error.0.contains("viewer3d"), "{error}");
    assert!(
        error.0.contains("viewer_3d") && error.0.contains("action_log"),
        "the refusal lists the panels: {error}"
    );
}

/// The document `get_window_layout` hands out is a document
/// `set_window_layout` takes back, tag and all — one schema, one parser, and a
/// file an agent saved is a file the viewer reads at startup.
#[test]
fn a_whole_reply_can_be_sent_back() {
    let (mut state, mut viewer) = two_reconstructions();
    // A fake that clamps nothing, so the round trip is a round trip rather
    // than a demonstration of the clamp.
    let mut host = FakeWindow {
        minimum: [1, 1],
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "hide_panel",
        json!({ "panel_name": "camera_intrinsics" }),
    );
    let saved = ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout)
        ["window_layout"]
        .clone();
    assert_eq!(saved["sfm_explorer_layout"], json!(2));

    let reset = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "layout": "default" }),
    );
    assert!(state.is_panel_open(Tab::IntrinsicsDetail));
    assert_eq!(panel(&reset, "camera_intrinsics")["open"], json!(true));

    // The whole reply, unedited, including its version tag and window section.
    let restored = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        saved.clone(),
    );
    assert_eq!(restored["window_layout"], saved);
    assert!(!state.is_panel_open(Tab::IntrinsicsDetail));
}

/// And a file's text, parsed and sent as the argument, is the same document.
#[test]
fn a_file_can_be_sent_as_the_argument() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let file = state.window_layout().to_json();
    state.hide_panel(Tab::ActionLog);
    let document: Value = serde_json::from_str(&file).expect("the file parses");
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        document,
    );
    assert!(state.is_panel_open(Tab::ActionLog));
}

/// Every form in the spec's list, and what each leaves behind.
#[test]
fn each_form_of_set_window_layout_does_what_it_says() {
    // Maximize where it is.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "maximized" } }),
    );
    assert_eq!(reply["window"]["state"], "maximized");
    assert_eq!(host.applied, ["state maximized"]);

    // Restore, and resize.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Maximized);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "normal", "inner_size": [1700, 1300] } }),
    );
    assert_eq!(reply["window"]["state"], "normal");
    assert_eq!(reply["window"]["inner_size"], json!([1700, 1300]));

    // A size against a maximized window changes what it restores to and leaves
    // it maximized — the rule that replaced `set_window`'s refusal.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Maximized);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "inner_size": [1700, 1300] } }),
    );
    assert_eq!(reply["window"]["state"], "maximized");
    assert_eq!(
        reply["window_layout"]["window"]["inner_size"],
        json!([1700, 1300]),
        "the normal rectangle changed"
    );

    // Both portions, in that order.
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    state.hide_panel(Tab::ActionLog);
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "maximized" }, "layout": "default" }),
    );
    assert!(host.maximized);
    assert!(state.is_panel_open(Tab::ActionLog));
}

/// The reply is a read-back rather than an echo, so a size the platform
/// clamped comes back clamped — while the Action Log says what was asked for.
#[test]
fn the_reply_is_a_read_back_and_the_log_is_what_was_asked() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "inner_size": [640, 480] } }),
    );
    assert_eq!(reply["window"]["inner_size"], json!([1600, 1200]));
    assert_eq!(
        state
            .action_log
            .entries()
            .next_back()
            .expect("an entry")
            .text,
        "Resized window to 640×480"
    );
}

#[test]
fn a_piece_a_window_section_does_not_carry_is_preserved() {
    let mut host = FakeWindow {
        focused: false,
        ..FakeWindow::default()
    };
    let (mut state, mut viewer) = two_reconstructions();
    state.observe_window(&host);
    let before =
        ok_with(&mut state, &mut viewer, &mut host, Command::GetWindowLayout)["window"].clone();
    let after = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "focus": true } }),
    )["window"]
        .clone();
    assert_eq!(after["state"], before["state"]);
    assert_eq!(after["inner_size"], before["inner_size"]);
    assert_eq!(after["outer_position"], before["outer_position"]);
    assert_eq!(after["focused"], json!(true));
    assert_eq!(host.applied, ["focus"]);
}

#[test]
fn the_window_moves_between_the_four_states() {
    for from in WindowState::ALL {
        for to in WindowState::ALL {
            let (mut state, mut viewer, mut host) = windowed(from);
            let reply = call_with(
                &mut state,
                &mut viewer,
                &mut host,
                "set_window_layout",
                json!({ "window": { "state": to.wire_name() } }),
            );
            assert_eq!(
                reply["window"]["state"],
                json!(to.wire_name()),
                "{} -> {}",
                from.wire_name(),
                to.wire_name()
            );
        }
    }
}

/// `normal` means all three flags off. Restoring a minimized window can bring
/// a maximized one back, and the caller asked for normal.
#[test]
fn normal_clears_a_minimized_and_maximized_window() {
    let mut host = FakeWindow {
        minimized: true,
        maximized: true,
        ..FakeWindow::default()
    };
    let (mut state, mut viewer) = two_reconstructions();
    state.observe_window(&host);
    let reply = call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "normal" } }),
    );
    assert_eq!(reply["window"]["state"], "normal");
    assert!(!host.minimized && !host.maximized, "{host:?}");
}

/// The window portion is applied before the panels, so a platform refusal in
/// it stops the call with the dock untouched.
#[test]
fn a_position_refusal_leaves_the_panels_alone() {
    let mut host = FakeWindow {
        position: None,
        ..FakeWindow::default()
    };
    let (mut state, mut viewer) = two_reconstructions();
    state.observe_window(&host);
    state.hide_panel(Tab::ActionLog);
    let error = refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "outer_position": [10, 20] }, "layout": "default" }),
    );
    assert!(error.0.contains("position its own window"), "{error}");
    assert!(!state.is_panel_open(Tab::ActionLog), "the panels changed");
}

#[test]
fn a_document_that_does_not_validate_is_refused_whole() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    let before = state.layout();
    let error = refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({
            "window": { "state": "maximized" },
            "layout": {
                "main": {
                    "split": "left_right",
                    "fracton": 0.5,
                    "first": { "tabs": ["scene"] },
                    "second": { "tabs": ["viewer_3d"] },
                },
                "windows": [],
            }
        }),
    );
    // The layout parser's own message, path and all.
    assert_eq!(error.0, "layout.main: unknown key \"fracton\"", "{error}");
    assert_eq!(
        state.layout(),
        before,
        "a refusal leaves the dock untouched"
    );
    assert!(host.applied.is_empty(), "{:?}", host.applied);

    // …and a window key's own rule reads the same way.
    let error = refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "state": "big" } }),
    );
    assert!(error.0.starts_with("window.state: "), "{error}");
}

#[test]
fn the_only_named_layout_is_the_default() {
    let (mut state, mut viewer) = two_reconstructions();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_window_layout",
        json!({ "layout": "tidy" }),
    );
    assert!(
        error.0.contains("only named layout") && error.0.contains("default"),
        "{error}"
    );
}

/// A call is a request, and these ask for nothing.
#[test]
fn a_call_that_asks_for_nothing_is_refused() {
    let (mut state, mut viewer, mut host) = windowed(WindowState::Normal);
    for arguments in [
        json!({}),
        json!({ "window": {} }),
        json!({ "sfm_explorer_layout": 2 }),
    ] {
        let error = refused_call_with(
            &mut state,
            &mut viewer,
            &mut host,
            "set_window_layout",
            arguments.clone(),
        );
        assert!(error.0.contains("nothing to do"), "{arguments}: {error}");
    }
    assert!(host.applied.is_empty(), "{:?}", host.applied);
}

/// Where there is no window, a window portion is refused — and a panel portion
/// on its own succeeds, because the panels do not need one.
#[test]
fn a_window_portion_needs_a_window() {
    let (mut state, mut viewer) = two_reconstructions();
    state.window = None;
    let error = refused(
        &mut state,
        &mut viewer,
        Command::SetWindowLayout {
            document: json!({ "window": { "state": "maximized" } }),
        },
    );
    assert!(error.0.contains("no window"), "{error}");

    state.hide_panel(Tab::ActionLog);
    ok(
        &mut state,
        &mut viewer,
        Command::SetWindowLayout {
            document: json!({ "layout": "default" }),
        },
    );
    assert!(state.is_panel_open(Tab::ActionLog));
}

/// The three panel writes go through the same `AppState` methods the Panels
/// menu does, so the rows are the menu's rows with the agent in the actor
/// column.
#[test]
fn the_layout_writes_record_the_menus_own_entries_as_the_agent() {
    let cases: Vec<(Command, &str)> = vec![
        (
            Command::HidePanel {
                panel: Tab::ActionLog,
            },
            "Closed Action Log panel",
        ),
        (
            Command::ShowPanel {
                panel: Tab::TrackView,
            },
            "Raised Track View panel",
        ),
        (
            Command::SetWindowLayout {
                document: json!({ "layout": "default" }),
            },
            "Reset layout",
        ),
    ];
    for (command, text) in cases {
        let (mut state, mut viewer) = quiet_scene();
        ok(&mut state, &mut viewer, command);
        let entries: Vec<_> = state.action_log.entries().collect();
        assert_eq!(entries.len(), 1, "{entries:?}");
        assert_eq!(entries[0].text, text, "{entries:?}");
        assert_eq!(entries[0].kind, Kind::Layout, "{entries:?}");
        assert_eq!(entries[0].actor, Actor::Mcp, "{entries:?}");
    }

    // A panel that was closed is *opened*, not raised…
    let (mut state, mut viewer) = quiet_scene();
    state.hide_panel(Tab::ActionLog);
    state.action_log.clear();
    ok(
        &mut state,
        &mut viewer,
        Command::ShowPanel {
            panel: Tab::ActionLog,
        },
    );
    assert_eq!(
        state.action_log.entries().next().expect("an entry").text,
        "Opened Action Log panel"
    );

    // …and a document says which tool set it, since `apply_layout` records
    // nothing itself.
    let (mut state, mut viewer) = quiet_scene();
    let document = ok(&mut state, &mut viewer, Command::GetWindowLayout)["window_layout"].clone();
    state.action_log.clear();
    ok(
        &mut state,
        &mut viewer,
        Command::SetWindowLayout { document },
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].text, "Set layout");
    assert_eq!(entries[0].kind, Kind::Layout);
}

/// One row per portion, because the two portions are two kinds — and the
/// window row is composed in the order the pieces were applied.
#[test]
fn a_call_carrying_both_portions_records_both() {
    let (mut state, mut viewer) = quiet_scene();
    let mut host = FakeWindow {
        minimum: [1, 1],
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    state.action_log.clear();
    call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({
            "window": {
                "state": "maximized",
                "outer_position": [120, 64],
                "inner_size": [1280, 720],
            },
            "layout": "default",
        }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 2, "{entries:?}");
    assert_eq!(
        entries[0].text,
        "Moved window to (120, 64); resized window to 1280×720; maximized window"
    );
    assert_eq!(entries[0].kind, Kind::Window);
    assert_eq!(entries[0].actor, Actor::Mcp);
    assert_eq!(entries[1].text, "Reset layout");
    assert_eq!(entries[1].kind, Kind::Layout);
}

/// A refusal is one failed row, filed under the kind of the portion the call
/// carried.
#[test]
fn a_refusal_is_filed_under_the_portion_it_carried() {
    let (mut state, mut viewer) = quiet_scene();
    let mut host = FakeWindow {
        position: None,
        ..FakeWindow::default()
    };
    state.observe_window(&host);
    state.action_log.clear();
    refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "outer_position": [10, 20] } }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].failed, "{entries:?}");
    assert!(
        entries[0].text.starts_with("set_window_layout failed: "),
        "{entries:?}"
    );
    assert_eq!(entries[0].kind, Kind::Window);

    // The same call carrying panels is a layout refusal.
    let (mut state, mut viewer) = quiet_scene();
    state.observe_window(&host);
    state.action_log.clear();
    refused_call_with(
        &mut state,
        &mut viewer,
        &mut host,
        "set_window_layout",
        json!({ "window": { "outer_position": [10, 20] }, "layout": "default" }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Layout);
}

#[test]
fn the_layout_read_is_a_query_entry() {
    let (mut state, mut viewer) = quiet_scene();
    ok(&mut state, &mut viewer, Command::GetWindowLayout);
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Query("get_window_layout"));
    assert_eq!(entries[0].text, "get_window_layout");
    // A query never reaches the status line: an agent polling must not read
    // its own polling back as the viewer's status.
    assert_eq!(state.status_message(), None);
}

// ── The window block ────────────────────────────────────────────────────

#[test]
fn get_scene_embeds_the_window_block() {
    let (mut state, mut viewer, _) = windowed(WindowState::Normal);
    let window = ok(&mut state, &mut viewer, Command::GetScene)["window"].clone();
    assert_eq!(window["state"], "normal");
    assert_eq!(window["focused"], json!(true));
    assert_eq!(window["scale_factor"], json!(1.5));
    assert_eq!(window["inner_size"], json!([1920, 1080]));
    assert_eq!(window["outer_size"], json!([1936, 1119]));
    assert_eq!(window["outer_position"], json!([120, 64]));
    // Physical pixels throughout, with the logical size under `derived` next
    // to the scale factor it comes from.
    assert_eq!(
        window["derived"]["inner_size_logical"],
        json!([1280.0, 720.0])
    );
    assert_eq!(window["monitor"]["name"], "DISPLAY1");
    assert!(
        window["monitors"].is_null(),
        "get_scene does not list every monitor: {window}"
    );
    let fraction = window["derived"]["monitor_fraction"][0]
        .as_f64()
        .expect("a fraction");
    assert!((fraction - 1936.0 / 3840.0).abs() < 1e-9, "{fraction}");
}

/// A picture of a window the human cannot see answers nothing an agent asked
/// of a shared viewer, and whether a minimized window's swapchain still
/// presents is platform-dependent.
#[test]
fn a_screenshot_of_a_minimized_window_is_refused() {
    let (mut state, mut viewer, _) = windowed(WindowState::Minimized);
    let error = refused(&mut state, &mut viewer, screenshot(None, true, None));
    assert!(
        error.0.contains("minimized") && error.0.contains("set_window_layout"),
        "{error}"
    );

    // Not minimized, it defers as it always did.
    state.window = Some(FakeWindow::default().info());
    assert!(
        matches!(
            agent(&mut state, &mut viewer, screenshot(None, true, None)),
            Outcome::Deferred(_)
        ),
        "a screenshot must defer once there is something to photograph"
    );
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

/// One valid call for every catalog entry, kept as an exact set so adding a
/// tool also requires choosing arguments that exercise its parser.
fn representative_tool_calls() -> Vec<(&'static str, Value)> {
    vec![
        ("get_scene", json!({})),
        ("list_camera_images", json!({})),
        ("get_camera_image", json!({ "camera_image": 0 })),
        (
            "get_camera_intrinsics",
            json!({ "camera_intrinsics_index": 0 }),
        ),
        ("get_point", json!({ "point": 0 })),
        ("get_action_log", json!({})),
        ("get_timing_detail", json!({})),
        ("get_window_layout", json!({})),
        ("get_image_detail_display", json!({})),
        ("get_image_detail_view", json!({})),
        (
            "set_image_detail_view",
            json!({ "reconstruction_label": "alpha", "pixel": [142.0, 197.5], "zoom": 4.0 }),
        ),
        ("get_history", json!({ "reconstruction_label": "alpha" })),
        ("open_reconstruction", json!({ "path": "scene.sfmr" })),
        (
            "close_reconstruction",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "select_reconstruction",
            json!({ "reconstruction_label": "alpha" }),
        ),
        ("select_camera_image", json!({ "camera_image": 0 })),
        (
            "select_camera_intrinsics",
            json!({ "camera_intrinsics_index": 0 }),
        ),
        ("select_point", json!({ "point": 0 })),
        ("clear_selection", json!({ "scope": "all" })),
        (
            "set_reconstruction_display",
            json!({ "reconstruction_label": "alpha", "visible": true }),
        ),
        ("set_solo", json!({ "reconstruction_label": "alpha" })),
        ("set_image_detail_display", json!({ "tracked_only": true })),
        ("set_timing_detail", json!({ "enabled": true })),
        ("set_view", json!({ "fit": "alpha" })),
        ("set_window_layout", json!({ "layout": "default" })),
        ("show_panel", json!({ "panel_name": "scene" })),
        ("hide_panel", json!({ "panel_name": "scene" })),
        ("undo", json!({ "reconstruction_label": "alpha" })),
        ("redo", json!({ "reconstruction_label": "alpha" })),
        (
            "jump_to_version",
            json!({ "reconstruction_label": "alpha", "serial": "v1" }),
        ),
        (
            "save_reconstruction",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "delete_point",
            json!({ "reconstruction_label": "alpha", "point": 0 }),
        ),
        (
            "retriangulate_point",
            json!({ "reconstruction_label": "alpha", "point": 0 }),
        ),
        (
            "retriangulate_all_points",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "prune_covered_observations",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "delete_camera_image",
            json!({ "reconstruction_label": "alpha", "camera_image": 0 }),
        ),
        (
            "move_camera_image",
            json!({
                "reconstruction_label": "alpha",
                "camera_image": 0,
                "world_from_camera": {
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                },
            }),
        ),
        (
            "resect_camera_image",
            json!({ "reconstruction_label": "alpha", "camera_image": 0 }),
        ),
        ("bundle_adjust", json!({ "reconstruction_label": "alpha" })),
        (
            "convert_to_embedded_patches",
            json!({ "reconstruction_label": "alpha" }),
        ),
        ("get_bench", json!({ "reconstruction_label": "alpha" })),
        (
            "get_bench_track",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "create_bench_cluster",
            json!({
                "reconstruction_label": "alpha",
                "camera_image": 0,
                "pixel": [142.0, 197.5],
                "radius_px": 7.5,
            }),
        ),
        (
            "create_bench_track",
            json!({ "reconstruction_label": "alpha", "point": 0 }),
        ),
        (
            "activate_bench_item",
            json!({ "reconstruction_label": "alpha", "item": "bull-nose" }),
        ),
        (
            "deactivate_bench_item",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "rename_bench_item",
            json!({
                "reconstruction_label": "alpha",
                "item": "IMG_0042@142,198",
                "label": "bull-nose",
            }),
        ),
        (
            "discard_bench_item",
            json!({ "reconstruction_label": "alpha", "item": "bull-nose" }),
        ),
        (
            "duplicate_bench_item",
            json!({ "reconstruction_label": "alpha", "item": "bull-nose" }),
        ),
        (
            "add_bench_track_observation",
            json!({
                "reconstruction_label": "alpha",
                "track": "bull-nose",
                "camera_image": 0,
                "pixel": [142.0, 197.5],
            }),
        ),
        (
            "translate_bench_patch",
            json!({
                "reconstruction_label": "alpha",
                "observation": 0,
                "pixel": [142.0, 197.5],
            }),
        ),
        (
            "sight_bench_observation",
            json!({
                "reconstruction_label": "alpha",
                "observation": 0,
                "pixel": [142.0, 197.5],
            }),
        ),
        (
            "shape_bench_observation",
            json!({
                "reconstruction_label": "alpha",
                "observation": 0,
                "shape": [[7.1, -0.4], [0.4, 7.1]],
            }),
        ),
        (
            "resize_bench_patch",
            json!({
                "reconstruction_label": "alpha",
                "observation": 0,
                "edge": "+u",
                "pixel": [150.0, 197.5],
            }),
        ),
        (
            "resize_bench_shape",
            json!({
                "reconstruction_label": "alpha",
                "observation": 0,
                "edge": "+u",
                "pixel": [150.0, 197.5],
            }),
        ),
        (
            "tilt_bench_patch",
            json!({ "reconstruction_label": "alpha", "normal": [0.1, -0.2, 0.97] }),
        ),
        (
            "spin_bench_patch",
            json!({ "reconstruction_label": "alpha", "degrees": 12.5 }),
        ),
        (
            "spin_bench_shape",
            json!({
                "reconstruction_label": "alpha",
                "observation": 0,
                "degrees": 12.5,
            }),
        ),
        (
            "set_bench_track_verdict",
            json!({
                "reconstruction_label": "alpha",
                "observation": 0,
                "verdict": "in",
            }),
        ),
        (
            "apply_bench_track_thresholds",
            json!({ "reconstruction_label": "alpha", "min_zncc": 0.8 }),
        ),
        (
            "split_bench_track",
            json!({ "reconstruction_label": "alpha", "observations": [1] }),
        ),
        (
            "commit_bench_track",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "evaluate_bench_track",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "fit_bench_track",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "set_bench_track_stage",
            json!({ "reconstruction_label": "alpha", "stage": "track" }),
        ),
        (
            "search_bench_track_descriptors",
            json!({ "reconstruction_label": "alpha", "observation": 0 }),
        ),
        (
            "search_bench_track_geometry",
            json!({ "reconstruction_label": "alpha", "observation": 0 }),
        ),
        (
            "open_sift_index",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "build_sift_index",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "close_sift_index",
            json!({ "reconstruction_label": "alpha" }),
        ),
        ("get_background_task", json!({})),
        ("cancel_background_task", json!({})),
        ("screenshot", json!({})),
    ]
}

/// The catalog, parser and command metadata are three descriptions of the same
/// tool. Exercise the parser rather than constructing commands by hand so name
/// and read-only drift on either side is caught at the boundary.
#[test]
fn every_advertised_tool_parses_to_matching_command_metadata() {
    let catalog = tools::catalog();
    let calls = representative_tool_calls();
    let catalog_names: std::collections::BTreeSet<_> =
        catalog.iter().map(|spec| spec.name).collect();
    let fixture_names: std::collections::BTreeSet<_> =
        calls.iter().map(|(name, _)| *name).collect();

    assert_eq!(
        catalog_names.len(),
        catalog.len(),
        "duplicate tool in catalog"
    );
    assert_eq!(
        fixture_names.len(),
        calls.len(),
        "duplicate tool in fixture"
    );
    assert_eq!(
        fixture_names, catalog_names,
        "representative calls must cover exactly the advertised tools"
    );

    for spec in catalog {
        let arguments = calls
            .iter()
            .find_map(|(name, arguments)| (*name == spec.name).then_some(arguments))
            .expect("catalog and fixture names were checked above")
            .as_object()
            .expect("representative arguments are objects");
        let command = tools::parse(spec.name, Some(arguments))
            .unwrap_or_else(|error| panic!("{} representative call: {error}", spec.name));

        assert_eq!(command.tool_name(), spec.name, "{} command name", spec.name);
        assert_eq!(
            matches!(command.kind(), Kind::Query(_)),
            spec.kind == ToolKind::Read,
            "{} read-only classification",
            spec.name
        );
    }
}

/// A misspelled argument is refused by every advertised tool. Starting from
/// the representative valid calls keeps this check independent of each
/// branch's other validation rules.
#[test]
fn every_advertised_tool_refuses_an_unknown_top_level_argument() {
    for (name, arguments) in representative_tool_calls() {
        let mut arguments = arguments.as_object().cloned().expect("an object");
        arguments.insert("unknown_argument".into(), json!(true));
        let error = tools::parse(name, Some(&arguments)).expect_err("rejected");
        assert!(
            error
                .0
                .starts_with(&format!("{name} has no argument \"unknown_argument\" — ")),
            "{name}: {error}"
        );
    }
}

#[test]
fn schema_driven_unknown_argument_errors_remain_compatible() {
    let arguments = json!({ "reconstruction_labelz": "alpha" })
        .as_object()
        .cloned()
        .expect("an object");
    let error = tools::parse("list_camera_images", Some(&arguments)).expect_err("rejected");
    assert_eq!(
        error.0,
        "list_camera_images has no argument \"reconstruction_labelz\" — it takes limit, \
         offset, reconstruction_label."
    );

    let arguments = json!({
        "look_through": { "camera_image": 0, "unknown_argument": true }
    })
    .as_object()
    .cloned()
    .expect("an object");
    let error = tools::parse("set_view", Some(&arguments)).expect_err("rejected");
    assert_eq!(
        error.0,
        "set_view.look_through has no argument \"unknown_argument\" — it takes \
         reconstruction_label, camera_image."
    );

    let arguments = json!({ "unknown_argument": true })
        .as_object()
        .cloned()
        .expect("an object");
    assert_eq!(
        tools::parse("get_scene", Some(&arguments))
            .expect_err("rejected")
            .0,
        "get_scene has no argument \"unknown_argument\" — it takes none."
    );

    let arguments = json!({ "offset": "bad", "unknown_argument": true })
        .as_object()
        .cloned()
        .expect("an object");
    assert!(
        tools::parse("list_camera_images", Some(&arguments))
            .expect_err("rejected")
            .0
            .contains("has no argument \"unknown_argument\""),
        "unknown-key validation must precede value validation"
    );

    assert_eq!(
        tools::parse("not_a_tool", Some(&arguments))
            .expect_err("rejected")
            .0,
        "There is no tool named \"not_a_tool\". Call tools/list for what this viewer offers."
    );
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
            // A panel argument carries a *name*, so it says so — the same rule
            // that makes the reconstruction argument `reconstruction_label`.
            assert!(
                property != "panel",
                "{}: a panel argument carries a name, so it is panel_name",
                spec.name
            );
        }
    }
    assert!(!names.iter().any(|name| name.contains("recon_")));
    let unique: std::collections::BTreeSet<&&str> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "tool names must be unique");

    // `hud` is the one initialism on the surface, and it is here because it is
    // the GUI's own word for the overlay (specs/gui/viewport-hud.md), which the
    // agent and the human have to be able to say the same way. Asserted by name
    // so a second one cannot arrive quietly.
    let hud_takers: Vec<&str> = tools::catalog()
        .iter()
        .filter(|spec| {
            spec.schema["properties"]
                .as_object()
                .is_some_and(|properties| properties.contains_key("hud"))
        })
        .map(|spec| spec.name)
        .collect();
    assert_eq!(hud_takers, ["screenshot"]);
}

/// The tool that hands back a picture advertises what it can photograph.
#[test]
fn screenshot_advertises_the_panel_the_hud_and_the_size() {
    let catalog = tools::catalog();
    let spec = catalog
        .iter()
        .find(|spec| spec.name == "screenshot")
        .expect("the tool is in the catalog");
    let properties = spec.schema["properties"]
        .as_object()
        .expect("an object schema");
    let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["hud", "max_dimension", "panel_name"]);
    // The panel names are the layout file's, so there is no second spelling of
    // them anywhere.
    assert_eq!(
        properties["panel_name"]["enum"],
        json!(Tab::ALL.map(|tab| tab.wire_name()))
    );
}

/// The one tool that cannot answer in the frame it arrives in says so, rather
/// than returning an empty or stale picture.
#[test]
fn screenshot_defers_to_the_frame() {
    let (mut state, mut viewer) = two_reconstructions();
    match apply(&mut state, &mut viewer, screenshot(None, true, None)) {
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
            "get_action_log",
            "get_timing_detail",
            "get_window_layout",
            "get_image_detail_display",
            "get_image_detail_view",
            "get_history",
            "get_bench",
            "get_bench_track",
            "get_background_task",
            "screenshot",
        ]
    );
    // Fifteen reads, fifty-seven writes, the one that writes a file, and the
    // one that hands back a picture.
    assert_eq!(catalog.len(), 73, "the catalog has grown or shrunk");
    assert_eq!(
        catalog
            .iter()
            .filter(|spec| spec.kind == ToolKind::Write)
            .count(),
        57
    );
    // One tool can overwrite something the human cannot undo, and it is the
    // only one annotated destructive.
    let saves: Vec<&str> = catalog
        .iter()
        .filter(|spec| spec.kind == ToolKind::Save)
        .map(|spec| spec.name)
        .collect();
    assert_eq!(saves, ["save_reconstruction"]);
}

/// The spec's prose carries counts the code owns, and a count written out in
/// words is the first thing to go stale: the panel list said eight while the
/// viewer had ten for two releases, and the tool count has been three
/// different numbers. So the sentences that carry one are read back here,
/// against the catalog and the tab list themselves.
#[test]
fn the_spec_s_counts_are_the_catalog_s_and_the_panels() {
    let spec = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/gui/mcp-server.md"),
    )
    .expect("specs/gui/mcp-server.md is in the repo beside the crate");
    let prose = spec.to_lowercase();
    let catalog = tools::catalog();

    let total = format!("{} tools", spelled(catalog.len()));
    assert!(
        prose.contains(&total),
        "the spec never says {total:?}: § \"The tool surface\" and § \"Testing\" carry the count"
    );
    let reads = catalog
        .iter()
        .filter(|spec| spec.kind == ToolKind::Read)
        .count();
    let reads_sentence = format!("{} of them reads", spelled(reads));
    assert!(
        prose.contains(&reads_sentence),
        "the spec never says {reads_sentence:?}"
    );
    // § "The tool surface" gives the write count twice -- in the sentence that
    // opens the section and again in the one about the annotations -- and the
    // two had drifted apart, forty-seven against forty-four. Both sentences
    // wrap, so they are looked for in a copy with the line breaks taken out;
    // the assertions above match inside one line and use `prose` as it is.
    let unwrapped = prose.split_whitespace().collect::<Vec<_>>().join(" ");
    let writes = catalog
        .iter()
        .filter(|spec| spec.kind == ToolKind::Write)
        .count();
    for sentence in [
        format!("{} write, and one writes a file", spelled(writes)),
        format!("the {} writes `destructivehint: false`", spelled(writes)),
    ] {
        assert!(
            unwrapped.contains(&sentence),
            "the spec never says {sentence:?}: § \"The tool surface\" carries the count twice"
        );
    }

    // The bench family says its own size three times -- once in the heading
    // sentence and twice in the back-references that split it -- and the three
    // have to be one number. They were "twenty-three" and "the twenty-two".
    let bench = catalog
        .iter()
        .filter(|spec| spec.name.contains("bench") || spec.name.contains("sift_index"))
        .count();
    let family = format!("{} tools that read and work the", spelled(bench));
    assert!(
        prose.contains(&family),
        "the spec never says {family:?}: § \"The bench family\" carries the count"
    );
    let back = format!("of the {}", spelled(bench));
    assert_eq!(
        prose.matches(&back).count(),
        2,
        "the two sentences that split the bench family should both say {back:?}"
    );

    let panels = format!(
        "the {} names are the layout file's",
        spelled(Tab::ALL.len())
    );
    assert!(prose.contains(&panels), "the spec never says {panels:?}");
    for tab in Tab::ALL {
        let name = format!("`{}`", tab.wire_name());
        assert!(
            spec.contains(&name),
            "{name} is a panel on the wire and is in no sentence of the spec"
        );
    }
}

/// § "The tool surface"'s table names every tool the catalog does, and no
/// others.
///
/// That table is the first thing anyone writing a client reads, so a name in it
/// that is not on the wire is worse than no table at all: `cancel_background`
/// sat there while the wire had always said `cancel_background_task`, and
/// nothing said so. The counts beside it are read back in the test above; this
/// reads back the names, which is the part a client actually calls.
///
/// Only the names. What each row *says* a tool does is prose, and prose is what
/// review is for.
#[test]
fn the_spec_s_tool_table_names_the_catalog() {
    let spec = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/gui/mcp-server.md"),
    )
    .expect("specs/gui/mcp-server.md is in the repo beside the crate");

    // The table runs from its header row to the first line that is not a row.
    // One row can name two tools (`undo` / `redo` share a line), so every
    // backticked word in the first cell counts.
    let body = spec
        .split_once("| Tool | Kind | What it does |")
        .expect("§ \"The tool surface\" opens its table with that header")
        .1;
    let mut tabled: Vec<&str> = Vec::new();
    for line in body.lines() {
        let Some(first_cell) = line
            .strip_prefix("| ")
            .and_then(|row| row.split('|').next())
        else {
            // The rest of the header line and the `|---|` rule come first and
            // name nothing; the blank line after the last row ends the table.
            if tabled.is_empty() {
                continue;
            }
            break;
        };
        tabled.extend(first_cell.split('`').skip(1).step_by(2));
    }

    tabled.sort_unstable();
    tabled.dedup();
    let mut advertised: Vec<&str> = tools::catalog().iter().map(|spec| spec.name).collect();
    advertised.sort_unstable();

    let missing: Vec<&&str> = advertised.iter().filter(|n| !tabled.contains(n)).collect();
    let extra: Vec<&&str> = tabled.iter().filter(|n| !advertised.contains(n)).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "§ \"The tool surface\"'s table is not the catalog: missing {missing:?}, \
         listing {extra:?} which no tool is called"
    );
}

/// A small number as the spec's prose spells it: `54` is `fifty-four`.
fn spelled(n: usize) -> String {
    const UNITS: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    match n {
        0..=19 => UNITS[n].to_string(),
        20..=99 if n.is_multiple_of(10) => TENS[n / 10].to_string(),
        20..=99 => format!("{}-{}", TENS[n / 10], UNITS[n % 10]),
        _ => panic!("no word for {n}: the spec would not spell it out either"),
    }
}

/// The two halves of the layout surface advertise the document they share.
#[test]
fn set_window_layout_advertises_the_document() {
    let catalog = tools::catalog();
    let spec = catalog
        .iter()
        .find(|spec| spec.name == "set_window_layout")
        .expect("the tool is in the catalog");
    let properties = spec.schema["properties"]
        .as_object()
        .expect("an object schema");
    let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["layout", "sfm_explorer_layout", "window"]);

    let mut window_keys: Vec<&str> = properties["window"]["properties"]
        .as_object()
        .expect("the window section has a schema")
        .keys()
        .map(String::as_str)
        .collect();
    window_keys.sort_unstable();
    assert_eq!(
        window_keys,
        ["focus", "inner_size", "monitor", "outer_position", "state"]
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
    /// server's sender is dropped, which happens when the task does.
    _gui: std::thread::JoinHandle<()>,
}

fn running_server() -> RunningServer {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<super::Request>();
    let address =
        super::serve(0, tx, Default::default(), || {}).expect("an ephemeral port is bindable");

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
    // The viewer introduces itself, not the SDK. `ServerConfig::new` fills
    // `serverInfo` from `rmcp`'s own build environment, so a handler that does
    // not set it announces `{"name":"rmcp","version":"3.4.0"}` — which is what
    // a client lists the server as and what a human reads in a failure report.
    assert_eq!(
        result["serverInfo"]["name"],
        json!("sfm-explorer"),
        "{result}"
    );
    assert_eq!(
        result["serverInfo"]["version"],
        json!(env!("CARGO_PKG_VERSION")),
        "{result}"
    );
    assert_eq!(
        result["serverInfo"]["title"],
        json!("SfM Explorer"),
        "{result}"
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

/// The catalog is not advertised as cacheable. It is fixed within one task
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

// ── The editing surface ─────────────────────────────────────────────────
//
// One fixture for all of it: a node a resection and an adjustment can both run
// on, which is also an `embedded_patches` node with a path, so every family
// here has something to work with. It is the Scene Graph tests' own node, for
// the reason those tests borrow it from each other -- "a node an edit can run
// on" is one thing, and a second answer to it would be a second thing to keep
// in step.
//
// What these assert about an edit is the boundary and not the edit: the version
// it pushed, the reply's shape, the refusal's words, and the one Action Log row
// it left as the agent. What each family *does* to a reconstruction is asserted
// in `state::edits::tests`, over the same `AppState` calls these make.

/// A scene holding one editable node, `run_a`, selected, with a photograph
/// cached for its first two images so the edits that read pixels find them.
fn editable() -> (AppState, Viewer3D) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    state.select_recon(id);
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (width, height) = (camera.width, camera.height);
    for index in 0..2 {
        let data: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 251) as u8).collect();
        state.full_res_cache.insert(
            crate::scene::ImageRef::new(id, index),
            Some(std::sync::Arc::new(
                sfmtool_core::camera::remap::ImageU8Pyramid::from_image(
                    sfmtool_core::camera::remap::ImageU8::new(width, height, 3, data),
                    crate::state::PYRAMID_LEVELS,
                ),
            )),
        );
    }
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    state.window = Some(FakeWindow::default().info());
    (state, viewer)
}

/// How many versions `run_a` holds.
fn version_count(state: &AppState) -> usize {
    state.scene[0].history.versions().len()
}

/// The Action Log's rows, oldest first, as `(actor, failed, text)`.
fn rows(state: &AppState) -> Vec<(Actor, bool, String)> {
    state
        .action_log
        .entries()
        .map(|entry| (entry.actor, entry.failed, entry.text.clone()))
        .collect()
}

/// The failed rows of the Action Log, which is what "one refusal, one entry"
/// is counted over.
fn failures(state: &AppState) -> Vec<(Actor, bool, String)> {
    rows(state).into_iter().filter(|row| row.1).collect()
}

/// Nudge image 1 off its stored pose, which is what gives a resection and an
/// adjustment something to pull back.
fn perturb(state: &mut AppState, distance: f64) {
    let image = &mut state.scene[0].recon_mut().image_table.images[1];
    image.translation_xyz += nalgebra::Vector3::new(distance, -distance * 0.7, distance * 0.5);
}

/// A point edit pushes a version, and the reply says which one and what it did.
#[test]
fn delete_point_pushes_a_version_the_reply_names() {
    let (mut state, mut viewer) = editable();
    let reply = call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );

    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["reconstruction_label"], "run_a");
    let serial = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["serial"], serial);
    assert_eq!(reply["cursor"], serial);
    assert_eq!(reply["label"], "Deleted point 3 in run_a");
    assert_eq!(reply["dirty"], true);
    // The report is the sentence the edit recorded, serials and all, which is
    // where each family's own numbers are.
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Deleted point 3 in run_a"), "{report}");
    assert!(report.contains(&serial), "{report}");

    // And it is the agent's row, not the human's.
    let last = rows(&state).pop().expect("one entry per edit");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
}

/// A retriangulated point pushes a version, and the report names the verdict
/// its own observations supported rather than merely that something happened.
#[test]
fn retriangulate_point_pushes_a_version_that_names_its_verdict() {
    let (mut state, mut viewer) = editable();
    let reply = call(
        &mut state,
        &mut viewer,
        "retriangulate_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );

    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["reconstruction_label"], "run_a");
    let serial = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["serial"], serial);
    assert_eq!(reply["label"], "Retriangulated point 3 in run_a");
    let report = reply["report"].as_str().expect("a report");
    assert!(
        report.starts_with("Retriangulated point 3 in run_a"),
        "{report}"
    );
    assert!(
        report.contains("finite") || report.contains("infinity") || report.contains("left where"),
        "the report names no verdict: {report}"
    );
    let last = rows(&state).pop().expect("one entry per edit");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
}

/// The whole-value retriangulation goes to a worker, and the version it comes
/// back with is the node's next one.
#[test]
fn retriangulate_all_points_defers_and_comes_back_with_a_version() {
    let (mut state, mut viewer) = editable();
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("retriangulate_all_points", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the retriangulation must defer"),
    };
    assert_eq!(pending.operation_name, "Retriangulate all points");
    state.finish_background_task();

    let reply = match super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["label"], json!("Retriangulated run_a"), "{reply}");
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("points moved"), "{report}");
}

/// The prune goes to a worker, and the version it comes back with is the
/// node's next one.
#[test]
fn prune_covered_observations_defers_and_comes_back_with_a_version() {
    let (mut state, mut viewer) = editable();
    // The fixture's points carry no patch frame, so the node the wire acts on
    // is the one that does.
    state.scene.clear();
    state.append_node(crate::scene_graph::tests::prunable_node("/runs/run_a.sfmr"));
    let id = state.scene[0].id;
    state.select_recon(id);

    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command =
        tools::parse("prune_covered_observations", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the prune must defer"),
    };
    assert_eq!(pending.operation_name, "Prune covered observations");
    state.finish_background_task();

    let reply = match super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(
        reply["label"],
        json!("Pruned covered observations in run_a"),
        "{reply}"
    );
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("observations retired"), "{report}");
}

/// The three thresholds cross the wire, and a call that names none takes the
/// operation's own defaults.
#[test]
fn prune_covered_observations_carries_its_thresholds() {
    let defaults = sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions::default();
    let bare = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    match tools::parse("prune_covered_observations", Some(&bare)).expect("a well-formed call") {
        Command::PruneCoveredObservations { options, .. } => assert_eq!(options, defaults),
        other => panic!("parsed to {other:?}"),
    }

    let named = json!({
        "reconstruction_label": "run_a",
        "footprint_fraction": 0.4545,
        "ratio": 3.0,
        "min_fine_radius_px": 0.0,
    })
    .as_object()
    .cloned()
    .expect("an object");
    match tools::parse("prune_covered_observations", Some(&named)).expect("a well-formed call") {
        Command::PruneCoveredObservations { options, .. } => {
            assert_eq!(options.footprint_fraction, 0.4545);
            assert_eq!(options.ratio, 3.0);
            assert_eq!(options.min_fine_radius_px, 0.0);
            assert_eq!(options.min_observations, defaults.min_observations);
        }
        other => panic!("parsed to {other:?}"),
    }
}

/// An edit the state refuses pushes no version, answers in the state's words,
/// and leaves exactly one failed row.
#[test]
fn a_refused_edit_pushes_no_version_and_is_logged_once() {
    let (mut state, mut viewer) = editable();
    state.action_log.clear();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 99 }),
    );
    assert!(error.0.contains("out of range"), "{error}");
    assert_eq!(version_count(&state), 1);
    assert_eq!(failures(&state).len(), 1, "{:?}", rows(&state));
}

/// A bulk edit renumbers the image table, and the reply is read off the version
/// it pushed rather than off the request.
#[test]
fn delete_camera_image_renumbers_and_reports_the_version() {
    let (mut state, mut viewer) = editable();
    let before = state.scene[0].recon().image_table.images.len();
    let reply = call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );
    assert_eq!(
        state.scene[0].recon().image_table.images.len(),
        before - 1,
        "the image table did not shrink"
    );
    assert!(
        reply["label"]
            .as_str()
            .expect("a label")
            .starts_with("Deleted image "),
        "{reply}"
    );
}

/// The resection lands as the node's next version, and its report is the
/// resection's own summary.
#[test]
fn resecting_pushes_a_version_and_reports_the_estimate() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.30);
    let reply = call(
        &mut state,
        &mut viewer,
        "resect_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );
    assert_eq!(version_count(&state), 2);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Resected "), "{report}");
    assert!(report.contains("inliers"), "{report}");
}

/// From-matches reads the file chosen for the node in the viewer; with none
/// chosen the state refuses in its own words, and that refusal is the one row
/// the log gets.
#[test]
fn resecting_from_matches_without_a_chosen_file_is_refused_in_the_states_words() {
    let (mut state, mut viewer) = editable();
    state.action_log.clear();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "resect_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1, "from_matches": true }),
    );
    assert!(error.0.contains(".matches"), "{error}");
    assert_eq!(version_count(&state), 1);

    let failed = failures(&state);
    assert_eq!(failed.len(), 1, "one refusal is one entry");
    // The state's own sentence, not the drain's `{tool} failed: …` wrapper.
    assert!(
        !failed[0].2.starts_with("resect_camera_image failed"),
        "{:?}",
        failed[0]
    );
    assert_eq!(failed[0].0, Actor::Mcp);
}

/// Start a `bundle_adjust` the way the frame does, let the operation finish,
/// and answer it the way the readback phase does.
///
/// The tool defers rather than answering inside the call, so a test that wants
/// the reply has to do what the frame does: start it, let the worker run, and
/// ask [`super::edit::background_reply`] for the answer.
#[track_caller]
fn adjusted(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Value {
    let map = arguments.as_object().cloned().expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    let pending = match agent(state, viewer, command) {
        Outcome::Deferred(super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("bundle_adjust must defer"),
    };
    state.finish_background_task();
    match super::edit::background_reply(state, &pending).expect("the operation finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
}

/// The adjustment runs on the node's value and reports its residuals.
#[test]
fn bundle_adjust_pushes_a_version_and_reports_its_residuals() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    let reply = adjusted(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(version_count(&state), 2);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.contains("median residual"), "{report}");
    assert!(!report.contains("focal released"), "{report}");
    // An operation that finished inside the window answers as it always did:
    // no handle, and nothing for a reader to discriminate on.
    assert_eq!(reply["running"], Value::Null, "{reply}");
}

/// An operation that outlives the window answers with a handle instead.
#[test]
fn a_slow_adjustment_answers_with_a_handle_naming_it() {
    let (mut state, _viewer) = editable();
    let id = state.scene[0].id;
    // A worker held open, so the operation is still running when the reply is
    // asked for; and a call that started long enough ago to be past the window,
    // which is what the frame's clock would have reached.
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |_progress| {
                let _ = held.recv();
                crate::background::Finished::Failed("nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    let task = state.background_task().expect("running");
    let pending = super::BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        answer: super::Answer::Version(id),
        label: task.label.clone(),
        started: std::time::Instant::now() - super::REPLY_DIRECTLY_WITHIN,
    };

    let reply = match super::edit::background_reply(&state, &pending).expect("past the window") {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("a handle is JSON"),
    };
    assert_eq!(reply["running"], json!(true), "{reply}");
    assert_eq!(reply["operation"], json!("Bundle adjust"), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");
    assert_eq!(
        reply["operation_id"],
        json!(pending.operation_id),
        "{reply}"
    );

    // Inside the window and still running, there is no answer yet: the frame
    // moves on and asks again.
    let fresh = super::BackgroundReply {
        started: std::time::Instant::now(),
        ..pending
    };
    assert!(
        super::edit::background_reply(&state, &fresh).is_none(),
        "an operation inside the window answered early",
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
}

/// The conversion is a background task on the wire, and `get_scene` says what
/// it changed: one call defers, the frame answers with the version it pushed,
/// and the node's `feature_source` has flipped.
#[test]
fn convert_to_embedded_patches_defers_and_flips_the_feature_source() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(dir.path());
    state.select_recon(id);
    state.window = Some(FakeWindow::default().info());
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let before = call(&mut state, &mut viewer, "get_scene", json!({}));
    let node = &before["scene"][0];
    assert_eq!(node["feature_source"], json!("sift_files"), "{before}");
    assert_eq!(node["has_patch_data"], json!(false), "{before}");

    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command =
        tools::parse("convert_to_embedded_patches", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("the conversion must defer"),
    };
    assert_eq!(pending.operation_name, "Convert to embedded patches");
    state.finish_background_task();

    let reply = match super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    };
    assert_eq!(version_count(&state), 2);
    assert_eq!(
        reply["label"],
        json!("Converted run_a to embedded patches"),
        "{reply}"
    );
    let report = reply["report"].as_str().expect("a report");
    assert_eq!(
        report.split(" (").next().expect("a sentence"),
        "Converted run_a to embedded patches: 24 points framed, 8 images read",
        "{report}"
    );

    // The same operation, read back through the tool an agent polls, under the
    // same name and id the handle carried.
    let over = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(over["running"], json!(false), "{over}");
    assert_eq!(over["finished"], json!(true), "{over}");
    assert_eq!(
        over["operation"],
        json!("Convert to embedded patches"),
        "{over}"
    );
    assert_eq!(over["reconstruction_label"], json!("run_a"), "{over}");
    assert_eq!(over["operation_id"], json!(pending.operation_id), "{over}");

    let after = call(&mut state, &mut viewer, "get_scene", json!({}));
    let node = &after["scene"][0];
    assert_eq!(node["feature_source"], json!("embedded_patches"), "{after}");
    // The minimal conversion fuses no reference bitmap, so the narrower field
    // beside it does not move.
    assert_eq!(node["has_patch_data"], json!(false), "{after}");
}

/// A node that already carries embedded patches is refused inline, in the
/// sentence the greyed menu entry carries, and nothing is pushed.
#[test]
fn convert_to_embedded_patches_is_refused_on_an_embedded_node() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "convert_to_embedded_patches",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(error.0.contains("already an embedded_patches"), "{error}");
    assert_eq!(version_count(&state), 1);
    assert!(state.background_task().is_none());
}

/// The id in a handle goes on naming its operation after that operation has
/// finished, which is what an agent comes back with.
#[test]
fn a_handle_s_id_still_names_the_operation_once_it_has_finished() {
    let (mut state, mut viewer) = editable();
    perturb(&mut state, 0.02);
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::Deferred::Background(pending)) => pending,
        _ => panic!("bundle_adjust must defer"),
    };
    state.finish_background_task();
    assert!(state.background_task().is_none());

    let reply = match super::edit::background_reply(&state, &pending).expect("it finished") {
        Ok(ToolOutput::Json(value)) => value,
        other => panic!("expected the version, got {}", other.is_err()),
    };
    let cursor = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["cursor"], json!(cursor), "{reply}");
    assert!(reply["report"]
        .as_str()
        .expect("a report")
        .contains("Bundle adjusted run_a"));

    // An id that names no operation this session is never answered with
    // somebody else's run: it is told the outcome is gone and where to look.
    let stranger = super::BackgroundReply {
        operation_id: pending.operation_id + 1,
        ..pending
    };
    match super::edit::background_reply(&state, &stranger).expect("an answer, not a wait") {
        Err(error) => assert!(error.0.contains("get_action_log"), "{error}"),
        Ok(_) => panic!("a stale handle was answered with another operation's version"),
    }
}

/// Cancelling is refused when there is nothing to cancel, and stops the
/// operation when there is.
#[test]
fn cancel_background_stops_what_is_running_and_refuses_when_nothing_is() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(&mut state, &mut viewer, "cancel_background_task", json!({}));
    assert!(error.0.contains("Nothing is running"), "{error}");

    perturb(&mut state, 0.02);
    let map = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("bundle_adjust", Some(&map)).expect("a well-formed call");
    match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::Deferred::Background(_)) => {}
        _ => panic!("bundle_adjust must defer"),
    }
    let reply = call(&mut state, &mut viewer, "cancel_background_task", json!({}));
    assert_eq!(reply["cancelling"], json!("Bundle adjust"), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");

    state.finish_background_task();
    // Whether the solve reached a poll before finishing is a race, so what is
    // asserted is the flag's effect on the state machine rather than which of
    // the two rows landed: either way the operation is over and the node free.
    assert!(state.background_task().is_none());
    assert_eq!(state.busy_refusal(state.scene[0].id), None);
}

// ── What an agent reads about a background operation ────────────────────
//
// The seam under these is `background::tests`, which drives a real worker over
// the real channel. What is asserted here is only what reaches the wire: one
// shape for a running operation and a finished one, a `get_scene` block that
// does not grow with the solve, and a breakdown that is the entry's.

/// Start a fake operation on `run_a` that reports through `work` and then waits,
/// so a test can read a running operation at an instant it chose rather than
/// one the scheduler did.
///
/// Returns the sender that lets the job finish. A fake rather than a real
/// adjustment because what is under test is the reply: a solve fast enough to
/// be deterministic reports nothing worth reading, and one slow enough to read
/// is a test that takes a minute.
fn running_operation(
    state: &mut AppState,
    work: impl FnOnce(&sfmtool_core::progress::Progress<'_>) + Send + 'static,
) -> std::sync::mpsc::Sender<()> {
    let id = state.scene[0].id;
    let (open, held) = std::sync::mpsc::channel::<()>();
    let (said, heard) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |progress| {
                work(progress);
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                crate::background::Finished::Failed("the fake worker produced nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_task();
    open
}

/// A job that reports the shape a solve reports: a stage that closed, a stage
/// with a folded child and a count, and a stage still open when it is read.
fn reporting_job(progress: &sfmtool_core::progress::Progress<'_>) {
    drop(progress.phase("gather arrays"));
    {
        let solve = progress.phase("solve");
        for round in 1..=2u64 {
            drop(solve.phase("round"));
            solve.count(round, Some(3), "round");
        }
    }
    // Left open on purpose: the open stage is what `phase` names, and a reply
    // that only ever saw closed ones would not exercise it.
    std::mem::forget(progress.phase("damping ladder"));
}

/// The keys of a JSON object, sorted, for asserting a block's whole shape
/// rather than the fields a test happened to think of.
#[track_caller]
fn keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .unwrap_or_else(|| panic!("expected an object, got {value}"))
        .keys()
        .cloned()
        .collect();
    keys.sort();
    keys
}

/// One call answers both "is it still going" and "what did it cost", and the
/// two answers are the same shape with `running` telling them apart.
#[test]
fn get_background_task_reads_one_shape_running_and_finished() {
    let (mut state, mut viewer) = editable();

    // Nothing has run: both discriminators are there and both are false, so a
    // reader never has to tell a missing key from a false one.
    let idle = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(keys(&idle), ["finished", "running"], "{idle}");
    assert_eq!(idle["running"], json!(false), "{idle}");
    assert_eq!(idle["finished"], json!(false), "{idle}");

    let open = running_operation(&mut state, reporting_job);
    let operation_id = state.background_task().expect("running").id;
    let live = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(live["running"], json!(true), "{live}");
    assert_eq!(live.get("finished"), None, "{live}");
    assert_eq!(live["operation"], json!("Bundle adjust"), "{live}");
    assert_eq!(live["reconstruction_label"], json!("run_a"), "{live}");
    assert_eq!(live["operation_id"], json!(operation_id), "{live}");
    assert_eq!(live["cancellable"], json!(true), "{live}");
    assert!(
        live["elapsed_s"].as_f64().expect("a duration") >= 0.0,
        "{live}"
    );
    // The kernel's own count, unit and all, and the stage it is inside.
    assert_eq!(
        live["progress"],
        json!({ "done": 2, "total": 3, "unit": "round" }),
        "{live}"
    );
    assert_eq!(live["phase"], json!("damping ladder"), "{live}");
    // The stages it has reported so far, in the Background panel's own order
    // and spelling, which is unfolded: the two rounds are two rows, because
    // this tool answers about an operation being watched rather than one read
    // about afterwards.
    assert_eq!(
        wire_detail_lines(&json!({ "detail": live["phases"] })),
        [
            "gather arrays",
            "solve",
            "  round",
            "  round",
            "damping ladder"
        ],
        "{live}"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();

    let over = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(over["running"], json!(false), "{over}");
    assert_eq!(over["finished"], json!(true), "{over}");
    // The same operation, under the same names: an agent parses one shape.
    for field in ["operation", "reconstruction_label", "operation_id"] {
        assert_eq!(over[field], live[field], "{field} moved: {over}");
    }
    assert!(
        over["elapsed_s"].as_f64().expect("a cost") >= live["elapsed_s"].as_f64().expect("so far"),
        "the finished cost is shorter than the elapsed it was read at: {over}"
    );
    // How it ended, in the Action Log's own two fields and its own words.
    assert_eq!(over["failed"], json!(true), "{over}");
    assert_eq!(
        over["text"],
        json!("the fake worker produced nothing"),
        "{over}"
    );
}

/// The breakdown this tool reports is the transcript, and the Action Log entry
/// is the summary of that transcript: every run the wire reported is counted in
/// the row the entry folds it into.
///
/// The two present one operation differently on purpose, for the reason the
/// Background panel and the Action Log panel differ
/// (`specs/gui/operation-progress.md`). What they must never do is disagree
/// about what happened.
#[test]
fn the_entry_is_the_summary_of_the_breakdown_the_wire_reported() {
    let (mut state, mut viewer) = editable();
    let open = running_operation(&mut state, reporting_job);
    let since = state.action_log.revision();
    open.send(()).expect("the worker is waiting");
    state.finish_background_task();

    let reply = call(&mut state, &mut viewer, "get_background_task", json!({}));
    let log = ok(&mut state, &mut viewer, action_log_detail(since));
    let row = row_starting(&log, "the fake worker produced nothing");
    let wire = reply["phases"].as_array().expect("an array");
    for folded in row["detail"].as_array().expect("an array") {
        if folded["kind"] != json!("phase") {
            continue;
        }
        // Absent means one, which is how this surface spells a stage that ran
        // once.
        let runs = folded["runs"].as_u64().unwrap_or(1);
        let reported = wire
            .iter()
            .filter(|run| {
                run["kind"] == folded["kind"]
                    && run["name"] == folded["name"]
                    && run["depth"] == folded["depth"]
            })
            .count();
        assert_eq!(
            reported as u64, runs,
            "the entry folded {runs} runs of {} and the wire reported {reported}",
            folded["name"],
        );
    }
    assert_the_panel_agrees(&state, &row);
}

/// A breakdown on the wire is capped at the same size an entry's is, and says
/// how many it dropped. The **last** rows rather than the first, because this
/// is a transcript and nothing has collapsed the repetition in it: the first
/// hundred and twenty-eight rows of a long solve are its first few seconds and
/// say nothing about where it has got to.
#[test]
fn a_long_breakdown_is_capped_and_says_how_many_it_dropped() {
    let (mut state, mut viewer) = editable();
    let reported = ActionLog::DETAIL_EVENTS + 40;
    let open = running_operation(&mut state, move |progress| {
        for i in 0..reported {
            progress.message(Level::Info, format_args!("event {i}"));
        }
    });

    let live = call(&mut state, &mut viewer, "get_background_task", json!({}));
    let rows = live["phases"].as_array().expect("an array").clone();
    assert_eq!(rows.len(), ActionLog::DETAIL_EVENTS + 1, "{live}");
    // What it left out is said first, and what it kept is the recent end.
    assert_eq!(
        rows[0]["text"],
        json!("40 earlier events dropped"),
        "{live}"
    );
    assert_eq!(rows[1]["text"], json!("event 40"), "{live}");
    assert_eq!(
        rows[ActionLog::DETAIL_EVENTS]["text"],
        json!(format!("event {}", reported - 1)),
        "{live}"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
    let over = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(over["phases"], live["phases"], "{over}");
}

/// `get_scene` says the viewer is busy and stops there.
///
/// The most-polled tool on the surface carries a block whose size does not
/// depend on the operation: no phase table, no open phase, no status line, and
/// nothing at all once the operation is over.
#[test]
fn get_scene_says_the_viewer_is_busy_without_carrying_the_solve() {
    let (mut state, mut viewer) = editable();
    let idle = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(
        idle["background_task"],
        Value::Null,
        "{}",
        idle["background_task"]
    );

    let open = running_operation(&mut state, |progress| {
        reporting_job(progress);
        progress.set_status_message(format_args!("refining images/IMG_0007.jpg"));
    });
    let scene = ok(&mut state, &mut viewer, Command::GetScene);
    let block = &scene["background_task"];
    assert_eq!(
        keys(block),
        [
            "elapsed_s",
            "fraction",
            "operation",
            "operation_id",
            "reconstruction_label",
            "running",
        ],
        "the get_scene block has grown: {block}"
    );
    assert_eq!(block["running"], json!(true), "{block}");
    assert_eq!(block["operation"], json!("Bundle adjust"), "{block}");
    assert_eq!(block["reconstruction_label"], json!("run_a"), "{block}");
    // The one field a caller polls for movement, and the whole of what
    // `get_background_task` would add is absent here.
    assert!(block["fraction"].is_number(), "{block}");

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
    // Null again, rather than the operation that just ended: `background` is
    // read as "may I edit", and a block that outlived the operation would be
    // carried by every poll for the rest of the session.
    let after = ok(&mut state, &mut viewer, Command::GetScene);
    assert_eq!(after["background_task"], Value::Null, "{after}");
}

/// The apply timeout names what is running, and keeps its old guesses when
/// nothing is.
#[test]
fn the_timeout_message_names_an_operation_only_while_one_is_running() {
    let idle = super::server::timeout_message(None);
    assert!(idle.contains("did not answer within 10 seconds"), "{idle}");
    assert!(idle.contains("modal dialog"), "{idle}");
    assert!(!idle.contains("get_background_task"), "{idle}");

    let busy = super::server::timeout_message(Some(crate::background::Busy {
        operation: "Bundle adjust",
        label: "dino_dog_toy-embedded".to_string(),
    }));
    assert!(busy.contains("did not answer within 10 seconds"), "{busy}");
    assert!(
        busy.contains("Bundle adjust is running in the background on dino_dog_toy-embedded"),
        "{busy}"
    );
    assert!(busy.contains("get_background_task"), "{busy}");
    // The two guesses are gone: they name things that did not happen.
    assert!(!busy.contains("modal dialog"), "{busy}");
    assert!(!busy.contains("mid-drag"), "{busy}");
}

/// The notice the message reads is written where the task is written, so the
/// two cannot disagree about whether anything is running.
#[test]
fn the_busy_notice_tracks_the_operation() {
    let (mut state, _viewer) = editable();
    assert_eq!(crate::background::busy(&state.busy_notice), None);

    let open = running_operation(&mut state, |_| {});
    assert_eq!(
        crate::background::busy(&state.busy_notice),
        Some(crate::background::Busy {
            operation: "Bundle adjust",
            label: "run_a".to_string(),
        })
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
    assert_eq!(crate::background::busy(&state.busy_notice), None);
}

/// Undo, redo and the jump answer with the version now showing, and refuse at
/// the ends in the state's words.
#[test]
fn the_cursor_moves_answer_with_the_version_now_showing() {
    let (mut state, mut viewer) = editable();
    let first = state.scene[0].history.current_version().serial.to_string();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    let edited = state.scene[0].history.current_version().serial.to_string();

    let undone = call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(undone["cursor"], first);
    assert_eq!(undone["dirty"], false);
    // A cursor move made no version, so it reports none of its own.
    assert_eq!(undone["report"], Value::Null);

    let error = refused_call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(error.0, "Nothing to undo in run_a.");

    let redone = call(
        &mut state,
        &mut viewer,
        "redo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(redone["cursor"], edited);

    let jumped = call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": first }),
    );
    assert_eq!(jumped["cursor"], first);
    assert_eq!(jumped["label"], state.scene[0].history.versions()[0].label);
}

/// A serial the node does not hold is refused naming the read that lists them.
#[test]
fn a_serial_that_is_not_a_version_of_the_node_is_refused() {
    let (mut state, mut viewer) = editable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": "v99999" }),
    );
    assert!(error.0.contains("v99999"), "{error}");
    assert!(error.0.contains("get_history"), "{error}");
}

/// `get_history` is the Edit History panel's reading of the same list: every
/// version in order, the cursor, the version on disk, and the released rows.
#[test]
fn get_history_lists_the_versions_with_the_cursor_and_the_released_rows() {
    let (mut state, mut viewer) = editable();
    for point in [3, 4] {
        call(
            &mut state,
            &mut viewer,
            "delete_point",
            json!({ "reconstruction_label": "run_a", "point": point }),
        );
    }
    // The budget releases a version's value while keeping its row. Reaching the
    // real budget from a test would mean a reconstruction of gigabytes, so the
    // release itself is what is arranged -- on the oldest version, which is the
    // one the budget takes first and the one an undo does not need.
    state.scene[0].history.versions_mut_for_test()[0].value = None;

    let history = call(
        &mut state,
        &mut viewer,
        "get_history",
        json!({ "reconstruction_label": "run_a" }),
    );
    let versions = history["versions"].as_array().expect("a version list");
    assert_eq!(versions.len(), 3);
    assert_eq!(history["reconstruction_label"], "run_a");
    assert_eq!(history["dirty"], true);
    assert_eq!(history["can_undo"], true);
    assert_eq!(history["can_redo"], false);
    assert_eq!(history["cursor"], versions[2]["serial"]);
    assert_eq!(history["disk_serial"], versions[0]["serial"]);

    assert_eq!(versions[0]["is_on_disk"], true);
    assert_eq!(versions[2]["is_cursor"], true);
    assert_eq!(versions[0]["is_cursor"], false);
    assert_eq!(versions[0]["held"], false, "the released row says so");
    assert_eq!(versions[2]["held"], true);
    assert!(
        versions[2]["label"]
            .as_str()
            .expect("a label")
            .starts_with("Deleted point 4"),
        "{history}"
    );
    // The same instant format the Action Log's own rows carry.
    let at = versions[0]["at"].as_str().expect("a timestamp");
    assert!(at.len() >= 19 && at.as_bytes()[10] == b'T', "{at}");
}

/// A node that came from no file has no version on disk, and says so rather
/// than naming one.
#[test]
fn a_node_from_no_file_reports_no_version_on_disk() {
    let (mut state, mut viewer) = two_reconstructions();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));
    let history = call(
        &mut state,
        &mut viewer,
        "get_history",
        json!({ "reconstruction_label": "demo" }),
    );
    assert_eq!(history["path"], Value::Null);
    assert_eq!(history["disk_serial"], Value::Null);
    assert_eq!(history["versions"][0]["is_on_disk"], false);
}

/// A directory of this test's own under the system temp dir, emptied first so a
/// rerun does not read a previous run's file.
fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("sfm_explorer_mcp_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// Save As writes where it is told and re-points the node; Save afterwards
/// writes over what Save As chose.
#[test]
fn save_reconstruction_writes_to_a_path_and_then_over_it() {
    let dir = temp_dir("save");
    let path = dir.join("edited.sfmr");
    // A plain demo node rather than [`editable`]'s: what a save is about is the
    // file it writes, and this one is a value the writer accepts whole.
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        &dir.join("recon.sfmr"),
        SfmrReconstruction::demo(64),
    ));
    let mut viewer = Viewer3D::new();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "recon", "point": 3 }),
    );

    let saved = call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "recon", "path": path.display().to_string() }),
    );
    assert!(path.exists(), "the file was not written");
    // Save As re-points and re-labels the node, so the reply says what to call
    // it next time.
    assert_eq!(saved["reconstruction_label"], "edited");
    assert_eq!(saved["path"], path.display().to_string());
    assert_eq!(
        saved["serial"],
        state.scene[0].history.disk_serial().to_string()
    );
    assert!(!state.is_dirty(state.scene[0].id), "a save leaves it clean");

    // And with no path it goes over the file the node now has.
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "edited", "point": 4 }),
    );
    let again = call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "edited" }),
    );
    assert_eq!(again["path"], path.display().to_string());
    assert!(!state.is_dirty(state.scene[0].id));
}

/// A node that came from no file is refused a pathless save, in the words the
/// File menu's own Save uses.
#[test]
fn saving_a_node_from_no_file_without_a_path_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));
    let error = refused_call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "demo" }),
    );
    assert!(error.0.contains("came from no file"), "{error}");
}

/// `minimal: true` writes a copy where it is told and leaves the node where it
/// was; it needs a path, and never goes over the node's own file.
#[test]
fn save_reconstruction_minimal_writes_a_copy_and_leaves_the_node() {
    let dir = temp_dir("save_minimal");
    let own = dir.join("recon.sfmr");
    let copy = dir.join("recon-minimal.sfmr");
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(&own, SfmrReconstruction::demo(64)));
    let mut viewer = Viewer3D::new();
    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "recon", "point": 3 }),
    );
    let points = state.scene[0].point_count();

    let reply = call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({
            "reconstruction_label": "recon",
            "path": copy.display().to_string(),
            "minimal": true,
        }),
    );
    assert_eq!(reply["minimal"], true);
    assert_eq!(reply["reconstruction_label"], "recon");
    assert_eq!(reply["path"], copy.display().to_string());
    let written = sfmtool_sfmr_format::read_sfmr(&copy).expect("a readable copy");
    assert!(written.thumbnails_y_x_rgb.is_none());
    assert!(written.metadata.lineage.is_empty());
    assert!(written.metadata.workspace.absolute_path.is_empty());
    assert_eq!(written.metadata.operation, "minimal");
    assert_eq!(written.metadata.point_count as usize, points);
    let node = &state.scene[0];
    assert_eq!(node.path.as_deref(), Some(own.as_path()));
    assert!(
        state.is_dirty(node.id),
        "a minimal copy saves nothing of the node"
    );

    let error = refused_call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "recon", "minimal": true }),
    );
    assert!(error.0.contains("pass path"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `workspace_path` is recorded as it stands, wherever the file is written, and
/// it needs a path of its own: a save over the node's own file leaves the file
/// where it already is, so the path it records is already the right one.
#[test]
fn save_reconstruction_states_the_workspace_path_and_needs_a_path() {
    let dir = temp_dir("save_workspace_path");
    let own = dir.join("recon.sfmr");
    let copy = dir.join("published").join("recon.sfmr");
    std::fs::create_dir_all(copy.parent().expect("a parent")).expect("a writable temp dir");
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(&own, SfmrReconstruction::demo(32)));
    let mut viewer = Viewer3D::new();

    call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({
            "reconstruction_label": "recon",
            "path": copy.display().to_string(),
            "minimal": true,
            "workspace_path": "../shared/ws",
        }),
    );
    let written = sfmtool_sfmr_format::read_sfmr(&copy).expect("a readable copy");
    assert_eq!(written.metadata.workspace.relative_path, "../shared/ws");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "save_reconstruction",
        json!({ "reconstruction_label": "recon", "workspace_path": "." }),
    );
    assert!(error.0.contains("pass path with workspace_path"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The editing tools name their reconstruction rather than defaulting to the
/// selection, and an unknown label is refused naming what is loaded.
#[test]
fn every_editing_tool_requires_its_reconstruction_label() {
    let (mut state, mut viewer) = editable();
    for (name, arguments) in [
        ("get_history", json!({})),
        ("undo", json!({})),
        ("redo", json!({})),
        ("jump_to_version", json!({ "serial": "v0" })),
        ("save_reconstruction", json!({})),
        ("delete_point", json!({ "point": 3 })),
        ("delete_camera_image", json!({ "camera_image": 1 })),
        (
            "move_camera_image",
            json!({ "camera_image": 1, "world_from_camera": {
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0] } }),
        ),
        ("resect_camera_image", json!({ "camera_image": 1 })),
        ("bundle_adjust", json!({})),
        ("convert_to_embedded_patches", json!({})),
    ] {
        let error = refused_call(&mut state, &mut viewer, name, arguments);
        assert!(error.0.contains("reconstruction_label"), "{name}: {error}");
    }

    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_history",
        json!({ "reconstruction_label": "run_b" }),
    );
    assert!(error.0.contains("run_a"), "{error}");
}

/// A point id belonging to another reconstruction is refused rather than
/// edited: the two handles one call carries have to name the same node.
#[test]
fn a_point_from_another_reconstruction_is_refused() {
    let (mut state, mut viewer) = editable();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/other.sfmr"),
        SfmrReconstruction::demo(16),
    ));
    let other = state.scene[1].id;
    let id = crate::point_ids::mint(&state.scene[1], 2).expect("a point id");
    state.select_recon(other);

    let error = refused_call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": id }),
    );
    assert!(error.0.contains("other"), "{error}");
    assert_eq!(version_count(&state), 1);
}

/// The editing arguments are parsed by shape before anything is applied.
#[test]
fn the_editing_arguments_are_parsed_by_shape() {
    for (name, arguments, expected) in [
        (
            "jump_to_version",
            json!({ "reconstruction_label": "run_a", "serial": 4 }),
            "serial",
        ),
        (
            "move_camera_image",
            json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
            "world_from_camera",
        ),
        (
            "move_camera_image",
            json!({ "reconstruction_label": "run_a", "camera_image": 1,
                    "world_from_camera": { "quaternion_wxyz": [1.0, 0.0, 0.0],
                                           "translation": [0.0, 0.0, 0.0] } }),
            "quaternion_wxyz",
        ),
        (
            "bundle_adjust",
            json!({ "reconstruction_label": "run_a", "release_focal": "yes" }),
            "release_focal",
        ),
        (
            "save_reconstruction",
            json!({ "reconstruction_label": "run_a", "path": 7 }),
            "path",
        ),
        (
            "delete_point",
            json!({ "reconstruction_label": "run_a", "point_index": 3 }),
            "point_index",
        ),
        (
            "undo",
            json!({ "reconstruction_label": "run_a", "steps": 2 }),
            "steps",
        ),
    ] {
        let map = arguments.as_object().cloned().expect("an object");
        let error = tools::parse(name, Some(&map)).expect_err("refused at the parse");
        assert!(error.0.contains(expected), "{name}: {error}");
    }
}

/// The optional arguments have the defaults the tools advertise.
#[test]
fn the_editing_defaults_are_what_the_schemas_say() {
    let parse = |name: &str, arguments: Value| {
        let map = arguments.as_object().cloned().expect("an object");
        tools::parse(name, Some(&map)).expect("a well-formed call")
    };
    assert_eq!(
        parse(
            "resect_camera_image",
            json!({ "reconstruction_label": "a", "camera_image": "images/x.jpg" })
        ),
        Command::ResectCameraImage {
            reconstruction_label: "a".to_string(),
            camera_image: super::CameraImageSel::Name("images/x.jpg".to_string()),
            from_matches: false,
        }
    );
    // The pose arrives in a sub-object, and comes out of the parse as the two
    // arrays the command carries.
    assert_eq!(
        parse(
            "move_camera_image",
            json!({ "reconstruction_label": "a", "camera_image": 1,
                    "world_from_camera": { "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                                           "translation": [1.5, -2.0, 3.0] } })
        ),
        Command::MoveCameraImage {
            reconstruction_label: "a".to_string(),
            camera_image: super::CameraImageSel::Index(1),
            quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
            translation: [1.5, -2.0, 3.0],
        }
    );
    assert_eq!(
        parse(
            "bundle_adjust",
            json!({ "reconstruction_label": "a", "release_focal": true })
        ),
        Command::BundleAdjust {
            reconstruction_label: "a".to_string(),
            release_focal: true,
        }
    );
    assert_eq!(
        parse(
            "save_reconstruction",
            json!({ "reconstruction_label": "a" })
        ),
        Command::SaveReconstruction {
            reconstruction_label: "a".to_string(),
            path: None,
            minimal: false,
            workspace_path: None,
        }
    );
}

// ── move_camera_image ───────────────────────────────────────────────────
//
// The pose edit the human makes with a lock and the viewport, which an agent
// makes by sending the pose: it has no hand to place a camera with, and the
// pose is the whole input.

/// A pose for `run_a`'s image `index`, turned and shifted off where it stands,
/// in the node's own frame -- which is the frame the tool takes and every read
/// on this surface reports.
fn moved_pose(state: &AppState, index: usize) -> (Vec<f64>, Vec<f64>) {
    let stored = sfmtool_core::reconstruction::move_camera::pose_of(state.scene[0].recon(), index);
    let turn = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.2_f64);
    let rotation = turn * stored.rotation.as_nalgebra();
    let quaternion = rotation.as_ref();
    (
        vec![quaternion.w, quaternion.i, quaternion.j, quaternion.k],
        vec![
            stored.translation.x + 0.3,
            stored.translation.y - 0.1,
            stored.translation.z + 0.05,
        ],
    )
}

/// One `move_camera_image` call, with the pose in the sub-object the wire
/// carries it in.
#[track_caller]
fn move_camera(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    index: usize,
    quaternion: &[f64],
    translation: &[f64],
) -> Value {
    call(
        state,
        viewer,
        "move_camera_image",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": index,
            "world_from_camera": {
                "quaternion_wxyz": quaternion,
                "translation": translation,
            },
        }),
    )
}

/// The pose edit pushes a version like every other edit here, and the reply is
/// that version rather than a second rendering of the pose: where the camera
/// now stands is what the reads answer.
#[test]
fn move_camera_image_pushes_a_version_the_reply_names() {
    let (mut state, mut viewer) = editable();
    let (quaternion, translation) = moved_pose(&state, 1);

    let reply = move_camera(&mut state, &mut viewer, 1, &quaternion, &translation);

    assert_eq!(version_count(&state), 2);
    assert_eq!(reply["reconstruction_label"], "run_a");
    let serial = state.scene[0].history.current_version().serial.to_string();
    assert_eq!(reply["serial"], serial);
    assert_eq!(reply["cursor"], serial);
    assert_eq!(reply["dirty"], true);
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Moved camera "), "{report}");
    assert!(report.contains(&serial), "{report}");
    assert!(
        reply["label"]
            .as_str()
            .expect("a label")
            .starts_with("Moved camera "),
        "{reply}"
    );

    // The camera stands where the call put it, in the node's own frame.
    let centre = state.scene[0].recon().image_table.images[1].camera_center();
    for (axis, expected) in translation.iter().enumerate() {
        assert!(
            (centre[axis] - expected).abs() < 1e-9,
            "centre {axis}: {centre:?}"
        );
    }

    // And it is the agent's row, under the kind every edit of a value is filed
    // under.
    let last = rows(&state).pop().expect("one entry per edit");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));
}

/// An image the node does not have is refused before anything moves, and the
/// refusal is the one row it leaves.
#[test]
fn move_camera_image_refuses_an_image_that_is_not_there() {
    let (mut state, mut viewer) = editable();
    state.action_log.clear();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "move_camera_image",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 99,
            "world_from_camera": {
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
            },
        }),
    );
    assert!(error.0.contains("out of range"), "{error}");
    assert_eq!(version_count(&state), 1);
    assert_eq!(failures(&state).len(), 1, "{:?}", rows(&state));
}

/// A camera in hand on the node is ended before the wire's own pose lands on
/// it, and committed because it had been moved: two hands on one image, two
/// versions, in the order they happened.
#[test]
fn move_camera_image_commits_a_lock_held_on_the_same_node_first() {
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);

    let (quaternion, translation) = moved_pose(&state, 1);
    move_camera(&mut state, &mut viewer, 1, &quaternion, &translation);

    assert!(viewer.camera_lock.is_none(), "the lock survived an edit");
    assert_eq!(
        version_count(&state),
        3,
        "the hand's move and the wire's are two versions"
    );
}

/// A view command that leaves camera view is a step away from a camera in
/// hand, as `,` and `.` are: the lock ends first, as a commit when it has been
/// moved. A field-of-view change keeps camera view, so it keeps the lock.
#[test]
fn a_view_command_that_leaves_camera_view_commits_a_held_lock_first() {
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);

    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "fov_short_axis_deg": 40.0 }),
    );
    assert!(
        viewer.camera_lock.is_some(),
        "a field-of-view change ended the lock"
    );
    assert_eq!(version_count(&state), 1);

    call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "target": [1.0, 2.0, 3.0] }),
    );
    assert!(
        viewer.camera_lock.is_none(),
        "the lock survived a placement"
    );
    assert!(
        viewer.camera_view.is_none(),
        "the placement left camera view up"
    );
    assert_eq!(version_count(&state), 2, "the hand's move was committed");
    let texts: Vec<String> = rows(&state).into_iter().map(|row| row.2).collect();
    assert!(
        texts.iter().any(|text| text.starts_with("Moved camera ")),
        "the commit recorded itself; the log holds {texts:?}"
    );
}

/// The rule is the wire's rather than the pose edit's: any edit landing on the
/// node ends the lock first, since an edit under one would leave the reviewer
/// holding a camera whose stored pose had moved beneath them.
#[test]
fn another_edit_on_the_node_commits_a_held_lock_too() {
    let (mut state, mut viewer) = editable();
    hold_the_camera(&mut state, &mut viewer, 1);

    call(
        &mut state,
        &mut viewer,
        "delete_point",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );

    assert!(viewer.camera_lock.is_none(), "the lock survived an edit");
    assert_eq!(version_count(&state), 3);
    // The commit is recorded before the edit that displaced it, as it happened.
    let texts: Vec<String> = rows(&state).into_iter().map(|row| row.2).collect();
    let moved = texts
        .iter()
        .position(|text| text.starts_with("Moved camera "))
        .expect("the commit recorded itself");
    let deleted = texts
        .iter()
        .position(|text| text.starts_with("Deleted point 3"))
        .expect("the edit recorded itself");
    assert!(moved < deleted, "{texts:?}");
}

/// Take image `index` of `run_a` in hand and move it past the dead band, which
/// is what makes an implicit end a commit rather than a silent drop.
fn hold_the_camera(state: &mut AppState, viewer: &mut Viewer3D, index: usize) {
    let image = crate::scene::ImageRef::new(state.scene[0].id, index);
    state.select_image(Some(image));
    viewer.jump_to_camera_view(image, &state.scene[0]);
    crate::camera_lock::enter(viewer, state).expect("camera view of a posed image");
    viewer.camera.nodal_pan(50.0, 0.0);
}

/// `run_a`'s stored pose for image `index`, in its own frame.
fn stored_pose(state: &AppState, index: usize) -> sfmtool_core::Se3Transform {
    sfmtool_core::reconstruction::move_camera::pose_of(state.scene[0].recon(), index)
}

/// Assert the viewport is standing at `expected`, in `run_a`'s own frame.
#[track_caller]
fn assert_viewport_at(
    state: &AppState,
    viewer: &Viewer3D,
    expected: &sfmtool_core::Se3Transform,
    what: &str,
) {
    let at = crate::camera_lock::pending_pose(viewer, &state.scene[0]);
    let degrees = at
        .rotation
        .as_nalgebra()
        .rotation_to(expected.rotation.as_nalgebra())
        .angle()
        .to_degrees();
    let distance = (at.translation - expected.translation).norm();
    assert!(
        degrees < 1e-9 && distance < 1e-9,
        "{what}: the viewport is {degrees} deg and {distance} away from the pose it should show"
    );
}

/// A cursor move can restore the pose of the very camera the viewport is
/// looking through, and camera view follows the value rather than staying where
/// a hand left it -- so all three moves re-snap it, as the Edit menu's own Undo
/// and Redo do.
#[test]
fn the_cursor_moves_resnap_a_camera_view_onto_the_version_they_land_on() {
    let (mut state, mut viewer) = editable();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 1);
    let first = state.scene[0].history.current_version().serial.to_string();
    let stored = stored_pose(&state, 1);

    // Moved on the wire, then looked through again: the viewport stands at the
    // new pose, which is what each cursor move below has to take it off.
    let (quaternion, translation) = moved_pose(&state, 1);
    move_camera(&mut state, &mut viewer, 1, &quaternion, &translation);
    let moved = stored_pose(&state, 1);
    state.select_image(Some(image));
    viewer.jump_to_camera_view(image, &state.scene[0]);
    assert_viewport_at(&state, &viewer, &moved, "the look-through");

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_viewport_at(&state, &viewer, &stored, "undo");

    call(
        &mut state,
        &mut viewer,
        "redo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_viewport_at(&state, &viewer, &moved, "redo");

    call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": first }),
    );
    assert_viewport_at(&state, &viewer, &stored, "jump_to_version");

    // A refused move landed on nothing, so it moves the viewport no more than
    // the version: a free-look offset the human is holding is theirs to keep.
    viewer.camera.nodal_pan(40.0, 10.0);
    let held = crate::camera_lock::pending_pose(&viewer, &state.scene[0]);
    refused_call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_viewport_at(&state, &viewer, &held, "a refused undo");
}

/// A drawn action carries what it cost on the wire, in milliseconds.
///
/// The agent's half of the Action Log's duration column: an agent driving the
/// viewer can read back how long its own command took to reach the screen,
/// which is the measurement that `get_action_log` is otherwise silent about.
#[test]
fn get_action_log_reports_what_a_drawn_action_cost() {
    let (mut state, mut viewer) = quiet_scene();
    state
        .action_log
        .record(Kind::File, "Opened alpha from /runs/alpha.sfmr");
    state
        .action_log
        .settle(std::time::Instant::now(), Vec::new());

    let reply = ok(&mut state, &mut viewer, action_log_read(0, &Actor::ALL));

    let entries = reply["entries"].as_array().expect("an array");
    let row = entries.last().expect("at least the open");
    let took = row["took_ms"].as_f64().expect("a drawn row carries a cost");
    assert!(
        (0.0..60_000.0).contains(&took),
        "a plausible number of milliseconds, got {took}",
    );
}

// ── The bench ───────────────────────────────────────────────────────────
//
// One fixture again: a node whose keypoints are each point's exact projection,
// with a textured photograph cached for every image, so the two steps that read
// pixels have something to register against. What these assert is the boundary
// -- that each tool is the `AppState` call the panel makes, that an observation
// index survives the steps that follow it, and that a refusal is the bench's own
// sentence -- while what each step *does* to a track is asserted in
// `bench::tests` and in core's own tests over a plane whose numbers are known.

/// A scene holding one bench-capable node, `run_a`, selected, with a photograph
/// cached for every image.
///
/// The photograph is a pattern rather than a flat field, and its periods are a
/// few pixels and differ between the axes, for the reason `bench::tests` gives:
/// a field with nothing in it hands the correlation kernels no tile to register,
/// and the refusal would be the fixture's rather than the code's.
fn benchable() -> (AppState, Viewer3D) {
    benchable_with(crate::state::edits::tests::projected_embedded_demo(12))
}

/// [`benchable`] over a reconstruction the caller built, so a test of the
/// finite/infinity boundary can hand it one holding a bearing.
fn benchable_with(recon: sfmtool_core::SfmrReconstruction) -> (AppState, Viewer3D) {
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/run_a.sfmr"),
        recon,
    ));
    let id = state.scene[0].id;
    state.select_recon(id);
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    for image in 0..state.scene[0].image_count() {
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| {
                let p = i / 3;
                ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
            })
            .collect();
        state.full_res_cache.insert(
            crate::scene::ImageRef::new(id, image),
            Some(std::sync::Arc::new(
                sfmtool_core::camera::remap::ImageU8Pyramid::from_image(
                    sfmtool_core::camera::remap::ImageU8::new(w, h, 3, data),
                    crate::state::PYRAMID_LEVELS,
                ),
            )),
        );
    }
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    state.window = Some(FakeWindow::default().info());
    (state, viewer)
}

/// The point the bench tests put on the bench: it observes images 0, 1 and 2 at
/// their exact projections, which is what makes a triangulation of it well
/// conditioned.
const BENCH_POINT: u32 = 2;

/// Put [`BENCH_POINT`] on the bench through the wire and give back its label.
#[track_caller]
fn on_the_bench(state: &mut AppState, viewer: &mut Viewer3D) -> String {
    let reply = call(
        state,
        viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": BENCH_POINT }),
    );
    reply["item"]
        .as_str()
        .expect("a create names the item it made")
        .to_string()
}

/// Start a bench tool that goes to a worker, let it finish, and answer it the
/// way the readback phase does -- which is what [`adjusted`] does for the
/// adjustment, and for the same reason.
#[track_caller]
fn worked(state: &mut AppState, viewer: &mut Viewer3D, name: &str, arguments: Value) -> Value {
    let map = arguments.as_object().cloned().expect("an object");
    let command = tools::parse(name, Some(&map)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let pending = match agent(state, viewer, command) {
        Outcome::Deferred(super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("{name} must defer to a worker"),
    };
    state.finish_background_task();
    match super::edit::background_reply(state, &pending).expect("the operation finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
}

/// The three pixel steps in order: a cluster started at a pixel, an observation
/// added at another, and a verdict on it -- each one version, and the verdict
/// visible under the index the add reported.
#[test]
fn a_cluster_an_observation_and_a_verdict_round_trip_through_get_bench_track() {
    let (mut state, mut viewer) = benchable();
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let item = made["item"].as_str().expect("the new item").to_string();
    assert_eq!(made["cursor"], made["serial"], "{made}");
    assert!(
        made["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Started {item}")),
        "{made}"
    );

    let added = call(
        &mut state,
        &mut viewer,
        "add_bench_track_observation",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 1,
            "pixel": [124.0, 93.0],
        }),
    );
    // Named no track, so it landed on the active one, which is the cluster the
    // create just made.
    assert_eq!(added["item"], json!(item), "{added}");
    let observation = added["observation"].as_u64().expect("the index it took");
    assert_eq!(observation, 1, "{added}");

    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observation": observation,
            "verdict": "out",
        }),
    );

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["item"], json!(item), "{track}");
    assert_eq!(track["stage"], json!("cluster"), "{track}");
    assert_eq!(track["active"], json!(true), "{track}");
    let rows = track["observations"].as_array().expect("the observations");
    assert_eq!(rows.len(), 2, "{track}");
    assert_eq!(rows[0]["verdict"], json!("in"), "{track}");
    assert_eq!(rows[0]["provenance"]["kind"], json!("pixel"), "{track}");
    // The verdict is under the index the add reported, which is the claim: an
    // index an agent is holding goes on naming the observation it named.
    assert_eq!(rows[1]["observation"], json!(observation), "{track}");
    assert_eq!(rows[1]["verdict"], json!("out"), "{track}");
    assert_eq!(rows[1]["pinned"], json!(true), "{track}");
    assert_eq!(rows[1]["camera_image"], json!(1), "{track}");
    assert_eq!(
        rows[1]["cluster"]["seed_pixel"],
        json!([124.0, 93.0]),
        "{track}"
    );

    // And every step was a version of the node, walked back one at a time.
    assert_eq!(version_count(&state), 4);
}

/// A copy commits as a **creation**: it carries no origin, so it writes a new
/// point rather than replacing the one the original came from.
#[test]
fn a_duplicate_carries_the_patch_and_commits_as_a_creation() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let made = call(
        &mut state,
        &mut viewer,
        "duplicate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    let copy = made["item"].as_str().expect("the copy").to_string();
    assert_eq!(copy, format!("{item} copy"), "{made}");
    assert_eq!(made["copy_of"], json!(item), "{made}");
    assert_eq!(
        made["label"].as_str().expect("a version label"),
        format!("Duplicated {item} as {copy}"),
        "{made}"
    );
    assert_eq!(version_count(&state), before + 1);

    // The copy is the active track, so a call that names none acts on it, and
    // it carries the same sightings.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["item"], json!(copy), "{track}");
    assert_eq!(track["active"], json!(true), "{track}");
    assert_eq!(
        track["origin"],
        json!(null),
        "a copy has no origin: {track}"
    );
    let original = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(
        track["observations"], original["observations"],
        "the copy's sightings differ from the original's"
    );
    assert!(original["origin"].is_object(), "{original}");

    let committed = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        committed["point"]["replaced"],
        json!(null),
        "a copy has to create: {committed}"
    );
    assert!(committed["point"]["index"].is_number(), "{committed}");
}

/// The patch tools are the panel's handles: the patch slid, one edge put under
/// a pixel with the far one held, and a turn in the patch's own plane. Each is
/// one version, and what they write is what the exactness claim says it is --
/// including that at the track stage a move of the centre carries **every**
/// sighting with it, because the patch is the thing they are all views of.
#[test]
fn the_patch_tools_slide_resize_and_turn_and_each_is_one_version() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    // The sighting of observation 0, where it stands.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let was: Vec<f64> = track["observations"][0]["pixel"]
        .as_array()
        .expect("every observation says where it sits")
        .iter()
        .map(|n| n.as_f64().expect("a number"))
        .collect();
    let to = [was[0] + 2.5, was[1] - 1.5];

    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "observation": 0,
            "pixel": to,
        }),
    );
    assert_eq!(moved["item"], json!(item), "{moved}");
    assert_eq!(moved["observation"], json!(0), "{moved}");
    assert!(
        moved["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Moved {item} by ")),
        "{moved}"
    );
    assert_eq!(version_count(&state), before + 1);

    // The tool moved the patch, not the one observation it was aimed through:
    // every sighting moved, and each kept its own offset from where the centre
    // projects, which is what the tiles are cut on. The fixture's keypoints are
    // each point's exact projection, so every offset is zero and stays zero --
    // what is asserted is that the carry preserved them rather than that they
    // were reset.
    let sightings = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = sightings["observations"]
        .as_array()
        .expect("the observations");
    assert!(rows.len() > 1, "the fixture's track is a single sighting");
    for row in rows {
        assert_eq!(row["pinned"], json!(false), "a slide is not a verdict");
    }
    {
        let track = state
            .bench_track(state.scene[0].id, &item)
            .expect("the track is on the bench");
        let frame = track
            .track()
            .and_then(|payload| payload.placement.clone())
            .expect("a frame");
        for observation in &track.observations {
            let (camera, pose) = crate::bench::geometry::view_of(
                &state.scene[0].edited().base.image_table,
                observation.image as usize,
            )
            .expect("the fixture's images have cameras");
            let expected =
                crate::bench::geometry::project(&camera, &pose, frame.center.coords, frame.w)
                    .expect("the demo's patch is in front of every camera");
            let site = observation.site().expect("a sighting");
            assert!(
                (site[0] - expected[0]).abs() < 1e-3 && (site[1] - expected[1]).abs() < 1e-3,
                "image {} should sight the centre at {expected:?}, it sights {site:?}",
                observation.image,
            );
        }
    }

    // The outline as it now stands at that sighting, so the resize can be
    // aimed at a place on it.
    let outline = |state: &AppState| {
        let track = state
            .bench_track(state.scene[0].id, &item)
            .expect("the track is on the bench");
        let sighting = &track.observations[0];
        let (camera, pose) = crate::bench::geometry::view_of(
            &state.scene[0].edited().base.image_table,
            sighting.image as usize,
        )
        .expect("the fixture's images have cameras");
        let frame = track
            .track()
            .and_then(|payload| payload.placement.clone())
            .expect("a track from a point carries the stored patch");
        let anchored = crate::bench::geometry::anchored_frame(&frame, &camera, &pose, sighting);
        (anchored, camera, pose)
    };
    let corner = |patch: &sfmtool_core::patch::cloud::OrientedPatch,
                  camera: &sfmtool_core::camera::CameraIntrinsics,
                  pose: &sfmtool_core::geometry::RigidTransform,
                  s: f64,
                  t: f64| {
        let (xyz, w) = patch.corner_homogeneous(s, t);
        crate::bench::geometry::project(camera, pose, xyz, w).expect("in front of the camera")
    };

    let (before_patch, camera, pose) = outline(&state);
    let far_before = corner(&before_patch, &camera, &pose, -1.0, 0.0);
    let target = corner(&before_patch, &camera, &pose, 1.7, 0.0);
    let resized = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "observation": 0,
            "edge": "+u",
            "pixel": target,
        }),
    );
    assert_eq!(resized["edge"], json!("+u"), "{resized}");
    assert!(
        resized["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Resized {item} to ")),
        "{resized}"
    );
    assert_eq!(version_count(&state), before + 2);

    // The claim, on the outline as it is **redrawn** -- the patch re-anchored
    // on the sighting whose edge was dragged, which is what a person sees:
    // that edge lands on the pixel the call named, the far edge has not moved,
    // and the frame is still square. A thousandth of a pixel, because the
    // redrawing goes through an `f32` keypoint slot.
    let (redrawn, _, _) = outline(&state);
    let frame = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a frame");
    assert_eq!(frame.half_extent[0], frame.half_extent[1]);
    let landed = corner(&redrawn, &camera, &pose, 1.0, 0.0);
    assert!(
        (landed[0] - target[0]).abs() < 1e-3 && (landed[1] - target[1]).abs() < 1e-3,
        "the +u edge should land on {target:?}, it landed on {landed:?}",
    );
    let far_after = corner(&redrawn, &camera, &pose, -1.0, 0.0);
    assert!(
        (far_after[0] - far_before[0]).abs() < 1e-3 && (far_after[1] - far_before[1]).abs() < 1e-3,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );

    let turned = call(
        &mut state,
        &mut viewer,
        "spin_bench_patch",
        json!({ "reconstruction_label": "run_a", "degrees": 30.0 }),
    );
    assert_eq!(turned["degrees"], json!(30.0), "{turned}");
    assert_eq!(
        turned["label"].as_str().expect("a version label"),
        format!("Spun {item} by 30.0 degrees"),
        "{turned}"
    );
    assert_eq!(version_count(&state), before + 3);
    let after = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a frame");
    assert_eq!(after.center, frame.center, "a turn moves the patch nowhere");
    assert!((after.normal() - frame.normal()).norm() < 1e-12);

    // And the three are versions of one history, walked back one at a time.
    for _ in 0..3 {
        call(
            &mut state,
            &mut viewer,
            "undo",
            json!({ "reconstruction_label": "run_a" }),
        );
    }
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let back: Vec<f64> = track["observations"][0]["pixel"]
        .as_array()
        .expect("a place")
        .iter()
        .map(|n| n.as_f64().expect("a number"))
        .collect();
    assert!(
        (back[0] - was[0]).abs() < 1e-6 && (back[1] - was[1]).abs() < 1e-6,
        "three undos did not put the sighting back: {back:?} != {was:?}",
    );

    // And the tool that moves **one** sighting is still there, for the cluster
    // stage's dot and for a script that means one keypoint: it writes that
    // observation alone and pins it.
    let one = call(
        &mut state,
        &mut viewer,
        "sight_bench_observation",
        json!({
            "reconstruction_label": "run_a",
            "observation": 0,
            "pixel": [was[0] + 2.0, was[1]],
        }),
    );
    assert!(
        one["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Moved observation 0 of {item} to (")),
        "{one}"
    );
    let rows = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(rows["observations"][0]["pinned"], json!(true), "{rows}");
    assert_eq!(rows["observations"][1]["pinned"], json!(false), "{rows}");
}

/// The two pixel forms also take a `camera_image` in place of an observation,
/// which reads the pixel against the patch **as it stands** in that image: the
/// ghost outline's square, in an image the track has no sighting in. The slide
/// lands the patch's own centre under the pixel, the resize puts the patch's
/// own edge there, each is one version, and naming both photographs at once is
/// refused.
#[test]
fn the_patch_tools_take_a_pixel_in_a_camera_image_the_track_has_no_sighting_in() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let id = state.scene[0].id;
    let track = state.bench_track(id, &item).expect("on the bench").clone();
    let unseen = (0..state.scene[0].image_count())
        .find(|i| track.observations.iter().all(|o| o.image as usize != *i))
        .expect("the fixture's track does not span every image");
    let (camera, pose) =
        crate::bench::geometry::view_of(&state.scene[0].edited().base.image_table, unseen)
            .expect("the fixture's images have cameras");
    let patch = |state: &AppState| {
        state
            .bench_track(id, &item)
            .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
            .expect("a frame")
    };
    let corner = |patch: &sfmtool_core::patch::cloud::OrientedPatch, s: f64, t: f64| {
        let (xyz, w) = patch.corner_homogeneous(s, t);
        crate::bench::geometry::project(&camera, &pose, xyz, w).expect("in front of the camera")
    };
    let before = version_count(&state);

    let was = patch(&state);
    let target = corner(&was, 0.6, -0.3);
    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": unseen,
            "pixel": target,
        }),
    );
    assert_eq!(moved["camera_image"], json!(unseen), "{moved}");
    assert!(moved.get("observation").is_none(), "{moved}");
    assert_eq!(version_count(&state), before + 1);
    let landed = corner(&patch(&state), 0.0, 0.0);
    assert!(
        (landed[0] - target[0]).abs() < 1e-6 && (landed[1] - target[1]).abs() < 1e-6,
        "the patch's own centre should land on {target:?}, it landed on {landed:?}",
    );

    // By name, as every camera_image argument may be.
    let name = state.scene[0].recon().image_table.images[unseen]
        .name
        .clone();
    let was = patch(&state);
    let far = corner(&was, -1.0, 0.0);
    let target = corner(&was, 1.5, 0.0);
    let resized = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": name,
            "edge": "+u",
            "pixel": target,
        }),
    );
    assert_eq!(resized["camera_image"], json!(unseen), "{resized}");
    assert_eq!(resized["edge"], json!("+u"), "{resized}");
    assert_eq!(version_count(&state), before + 2);
    let grown = patch(&state);
    let landed = corner(&grown, 1.0, 0.0);
    let held = corner(&grown, -1.0, 0.0);
    assert!(
        (landed[0] - target[0]).abs() < 1e-6 && (landed[1] - target[1]).abs() < 1e-6,
        "the +u edge should land on {target:?}, it landed on {landed:?}",
    );
    assert!(
        (held[0] - far[0]).abs() < 1e-6 && (held[1] - far[1]).abs() < 1e-6,
        "the far edge moved from {far:?} to {held:?}",
    );

    for (tool, arguments) in [
        (
            "translate_bench_patch",
            json!({
                "reconstruction_label": "run_a", "observation": 0,
                "camera_image": unseen, "pixel": [10.0, 10.0],
            }),
        ),
        (
            "resize_bench_patch",
            json!({
                "reconstruction_label": "run_a", "observation": 0,
                "camera_image": unseen, "edge": "+u", "pixel": [10.0, 10.0],
            }),
        ),
    ] {
        let refused = refused_call(&mut state, &mut viewer, tool, arguments).to_string();
        assert!(
            refused.contains("observation") && refused.contains("camera_image"),
            "{tool}: {refused}"
        );
    }
    let refused = refused_call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "camera_image": 999, "pixel": [10.0, 10.0] }),
    )
    .to_string();
    assert!(refused.contains("999"), "{refused}");
    assert_eq!(
        version_count(&state),
        before + 2,
        "a refusal pushes nothing"
    );
}

/// The translation's normal part, which no photograph can say: the depth of the
/// patch. The version's sentence carries the signed distance and the place the
/// centre reached.
#[test]
fn translating_a_bench_patch_along_its_normal_says_how_far_and_which_way() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);
    let frame = |state: &AppState| {
        state
            .bench_track(state.scene[0].id, &item)
            .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
            .expect("a track from a point carries the stored patch")
    };
    let was = frame(&state);

    let distance = -0.042;
    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.0, 0.0, distance] }),
    );
    assert_eq!(moved["item"], json!(item), "{moved}");
    assert_eq!(moved["by"], json!([0.0, 0.0, distance]), "{moved}");
    assert_eq!(moved["changed"], json!(true), "{moved}");
    assert_eq!(version_count(&state), before + 1);

    // The sentence a drag of the normal's segment writes, and the sign is in
    // it: an offset toward the cameras and one away from them are opposite
    // answers about how far off the patch is.
    let now = frame(&state);
    assert_eq!(
        moved["label"].as_str().expect("a version label"),
        format!(
            "Moved {item} by {distance:.3} units along its normal to ({:.3}, {:.3}, {:.3})",
            now.center.x, now.center.y, now.center.z
        ),
        "{moved}"
    );

    // And the patch really went that far along the normal, with its axes and
    // its size left alone.
    assert_eq!(now.half_extent, was.half_extent);
    assert!((now.normal() - was.normal()).norm() < 1e-12);
    assert!(
        ((now.center - was.center) - was.normal() * distance).norm() < 1e-12,
        "the patch went from {:?} to {:?}",
        was.center,
        now.center,
    );

    // A distance is relative where a pixel is absolute, so the same call again
    // moves it again rather than naming where it already stands.
    let again = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.0, 0.0, distance] }),
    );
    assert_eq!(again["changed"], json!(true), "{again}");

    // A displacement inside the patch's own tolerance changes nothing, and it
    // says so in the normal's own words rather than in a slide's: the two
    // gestures move the patch in different directions and a reader of the Action
    // Log should be able to tell them apart without a version to read it off.
    let still = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "by": [0.0, 0.0, now.half_extent[0] * 1e-9],
        }),
    );
    assert_eq!(still["changed"], json!(false), "{still}");
    let report = still["report"].as_str().expect("a sentence");
    assert!(report.starts_with("Moved"), "{report}");
    assert!(report.contains("along its normal"), "{report}");
    assert!(report.contains("no effect"), "{report}");
    assert_eq!(version_count(&state), before + 2, "{still}");
}

/// A track at infinity has no normal standing off it -- a direction patch's
/// normal is its own bearing -- so the **normal part** of a translation is
/// refused in those words while a purely tangential one is carried; and a
/// displacement JSON cannot carry is turned away at the parse. The refusals push
/// nothing.
#[test]
fn a_bearing_refuses_a_translation_along_its_normal_and_takes_a_tangential_one() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.0, 0.0, 0.05] }),
    )
    .to_string();
    assert!(refused.contains("infinity"), "{refused}");
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": "a little" }),
    )
    .to_string();
    assert!(refused.contains("by"), "{refused}");
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    // Neither of the two ways to name where it goes, and both of them: the pair
    // is exclusive, and the sentence says which two spellings it is choosing
    // between.
    for arguments in [
        json!({ "reconstruction_label": "run_a" }),
        json!({
            "reconstruction_label": "run_a", "by": [0.01, 0.0, 0.0],
            "observation": 0, "pixel": [10.0, 10.0],
        }),
    ] {
        let refused =
            refused_call(&mut state, &mut viewer, "translate_bench_patch", arguments).to_string();
        assert!(refused.contains("by"), "{refused}");
    }
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    // And the tangential part of the very same step is carried, the moved
    // bearing renormalized onto the sphere.
    let slid = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.02, -0.01, 0.0] }),
    );
    assert_eq!(slid["changed"], json!(true), "{slid}");
    let patch = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a bearing carries its tangent patch");
    assert_eq!(patch.w, 0.0, "a bearing stays a bearing");
    assert!((patch.center.coords.norm() - 1.0).abs() < 1e-12);
}

/// The other patch tool that names no pixel: which way the patch faces, which
/// no photograph can say. The version's sentence carries the turn actually
/// made.
#[test]
fn tilting_a_bench_patch_turns_its_normal_and_says_how_far() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);
    let frame = |state: &AppState| {
        state
            .bench_track(state.scene[0].id, &item)
            .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
            .expect("a track from a point carries the stored patch")
    };
    let was = frame(&state);

    // Ten degrees over toward the patch's own `+u`, which every observation can
    // still see it at, so the turn is made whole. Named at three times unit
    // length, because only the direction is read.
    let (sin, cos) = 10.0_f64.to_radians().sin_cos();
    let asked = (was.normal() * cos + was.u_axis * sin) * 3.0;
    let tilted = call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({ "reconstruction_label": "run_a", "normal": [asked.x, asked.y, asked.z] }),
    );
    assert_eq!(tilted["item"], json!(item), "{tilted}");
    assert_eq!(
        tilted["normal"],
        json!([asked.x, asked.y, asked.z]),
        "{tilted}"
    );
    assert_eq!(tilted["changed"], json!(true), "{tilted}");
    assert_eq!(version_count(&state), before + 1);
    assert_eq!(
        tilted["label"].as_str().expect("a version label"),
        format!("Tilted {item} by 10.0 degrees"),
        "{tilted}"
    );

    // The patch really faces there now, and its centre and size did not move:
    // a tilt is a turn about the centre and nothing else.
    let now = frame(&state);
    assert_eq!(now.center, was.center);
    assert_eq!(now.half_extent, was.half_extent);
    assert!(
        (now.normal() - asked.normalize()).norm() < 1e-9,
        "the patch faces {:?}, not {:?}",
        now.normal(),
        asked.normalize(),
    );

    // A normal is absolute where a distance is relative, so naming the one it
    // already faces is no turn, and it says so in its own words.
    let still = call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "normal": [now.normal().x, now.normal().y, now.normal().z],
        }),
    );
    assert_eq!(still["changed"], json!(false), "{still}");
    let report = still["report"].as_str().expect("a sentence");
    assert!(report.starts_with(&format!("Tilted {item}:")), "{report}");
    assert!(report.contains("no effect"), "{report}");
    assert_eq!(version_count(&state), before + 1, "{still}");
}

/// The cap is the whole point of the handle: the normal turns until a
/// photograph would be looking along the surface and no further, and the
/// sentence names the observation's image that stopped it.
#[test]
fn a_tilt_past_what_the_observations_can_see_stops_and_names_the_image() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let id = state.scene[0].id;
    let was = state
        .bench_track(id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a track from a point carries the stored patch");

    // Square to the normal the patch has, which is further over than any
    // photograph of it can still see.
    let across = was.u_axis;
    let tilted = call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "normal": [across.x, across.y, across.z],
        }),
    );
    assert_eq!(tilted["changed"], json!(true), "{tilted}");
    let label = tilted["label"].as_str().expect("a version label");
    assert!(
        label.contains("stopped 80.0 degrees from "),
        "a capped tilt should say so: {label}",
    );
    // The image it names is one of the track's own, read back off the state
    // rather than written into the test.
    let names: Vec<String> = state
        .bench_track(id, &item)
        .expect("the track")
        .observations
        .iter()
        .map(|observation| {
            state.image_name(crate::scene::ImageRef::new(id, observation.image as usize))
        })
        .collect();
    assert!(
        names.iter().any(|name| label.ends_with(name.as_str())),
        "{label} names no image of {names:?}",
    );
    // And it stopped short of what was asked: the normal is still well clear of
    // the direction named.
    let now = state
        .bench_track(id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a patch");
    assert!(
        now.normal().dot(&across) < 0.9,
        "the turn was not capped at all",
    );
}

/// A track at infinity has no normal standing off it to turn -- a direction
/// patch's normal is its own bearing -- and a direction that is not one is
/// turned away at the parse. Neither pushes a version.
#[test]
fn tilting_refuses_a_patch_at_infinity_and_a_normal_that_is_not_one() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({ "reconstruction_label": "run_a", "normal": [0.0, 0.0, 1.0] }),
    )
    .to_string();
    assert!(refused.contains("infinity"), "{refused}");

    for normal in [json!([0.0, 0.0]), json!("up"), json!([0.0, "up", 1.0])] {
        let refused = refused_call(
            &mut state,
            &mut viewer,
            "tilt_bench_patch",
            json!({ "reconstruction_label": "run_a", "normal": normal }),
        )
        .to_string();
        assert!(refused.contains("normal"), "{normal} gave {refused}");
    }
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");
}

/// A turn at the cluster stage is one sighting's affine shape and a turn at the
/// track stage is the patch's own, so they are two tools, each refusing the
/// other's stage and naming it.
#[test]
fn the_two_spins_each_refuse_the_other_stage_and_name_it() {
    let (mut state, mut viewer) = benchable();
    // A track-stage track first, so the cluster made after it is the active one
    // a call that names no track resolves to.
    let track_stage = on_the_bench(&mut state, &mut viewer);
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let item = made["item"].as_str().expect("the new item").to_string();
    let before = version_count(&state);
    let was = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.observations[0].shape())
        .expect("a seeded sighting has a shape");

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "spin_bench_patch",
        json!({ "reconstruction_label": "run_a", "degrees": 45.0 }),
    )
    .to_string();
    assert!(refused.contains("spin_bench_shape"), "{refused}");
    // And the other way about, on a track-stage track.
    let refused_there = refused_call(
        &mut state,
        &mut viewer,
        "spin_bench_shape",
        json!({
            "reconstruction_label": "run_a", "track": track_stage,
            "degrees": 45.0, "observation": 0,
        }),
    )
    .to_string();
    assert!(
        refused_there.contains("spin_bench_patch"),
        "{refused_there}"
    );
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    let turned = call(
        &mut state,
        &mut viewer,
        "spin_bench_shape",
        json!({
            "reconstruction_label": "run_a",
            "track": item,
            "degrees": 90.0,
            "observation": 0,
        }),
    );
    assert_eq!(
        turned["label"].as_str().expect("a version label"),
        format!("Spun observation 0 of {item} by 90.0 degrees"),
        "{turned}"
    );
    assert_eq!(version_count(&state), before + 1);

    // A quarter turn in the raster's own sense: `R(90) * shape`, which sends
    // each column onto the perpendicular of the one it was.
    let shape = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.observations[0].shape())
        .expect("a seeded sighting has a shape");
    let expected = [[-was[1][0], -was[1][1]], [was[0][0], was[0][1]]];
    for (row, want) in shape.iter().zip(&expected) {
        for (got, want) in row.iter().zip(want) {
            assert!((got - want).abs() < 1e-9, "{shape:?} is not {expected:?}");
        }
    }
}

/// The cluster stage's own two shape tools: an edge that scales the
/// parallelogram and holds its far edge, and the 2x2 stated outright. Each is
/// named for the part it acts on, so each refuses a track-stage track and names
/// the tool that does belong to it.
#[test]
fn the_shape_tools_size_a_cluster_sighting_and_refuse_a_track_stage_track() {
    let (mut state, mut viewer) = benchable();
    let track_stage = on_the_bench(&mut state, &mut viewer);
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let item = made["item"].as_str().expect("the new item").to_string();
    let before = version_count(&state);
    let shape_of = |state: &AppState| {
        state
            .bench_track(state.scene[0].id, &item)
            .and_then(|track| track.observations[0].shape())
            .expect("a seeded sighting has a shape")
    };
    let was = shape_of(&state);

    // The edge two px further out along `+u` than the parallelogram reaches,
    // with the far edge held: the new half-width is half way between.
    let radius = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.cluster().map(|payload| payload.radius))
        .expect("a cluster carries its radius");
    let half_px = sfmtool_core::bench::half_width_px(was, radius);
    let resized = call(
        &mut state,
        &mut viewer,
        "resize_bench_shape",
        json!({
            "reconstruction_label": "run_a", "observation": 0, "edge": "+u",
            "pixel": [120.0 + half_px + 2.0, 90.0],
        }),
    );
    assert_eq!(resized["edge"], json!("+u"), "{resized}");
    assert_eq!(resized["changed"], json!(true), "{resized}");
    let grown = sfmtool_core::bench::half_width_px(shape_of(&state), radius);
    assert!(
        (grown - (half_px + 1.0)).abs() < 1e-6,
        "the far edge did not hold: {half_px} became {grown}",
    );

    // And the whole 2x2 stated outright, shear and all, which is the general
    // form the spin and the resize are two special cases of.
    let asked = [[9.0, 1.5], [0.0, 9.0]];
    let shaped = call(
        &mut state,
        &mut viewer,
        "shape_bench_observation",
        json!({
            "reconstruction_label": "run_a", "observation": 0, "shape": asked,
        }),
    );
    assert_eq!(shaped["shape"], json!(asked), "{shaped}");
    assert_eq!(shape_of(&state), asked);
    assert_eq!(version_count(&state), before + 2);

    // Both belong to the cluster stage, and both say which tool a track-stage
    // track wants instead.
    let refused = refused_call(
        &mut state,
        &mut viewer,
        "resize_bench_shape",
        json!({
            "reconstruction_label": "run_a", "track": track_stage, "observation": 0,
            "edge": "+u", "pixel": [120.0, 90.0],
        }),
    )
    .to_string();
    assert!(refused.contains("resize_bench_patch"), "{refused}");
    let refused = refused_call(
        &mut state,
        &mut viewer,
        "shape_bench_observation",
        json!({
            "reconstruction_label": "run_a", "track": track_stage, "observation": 0,
            "shape": asked,
        }),
    )
    .to_string();
    assert!(refused.contains("cluster"), "{refused}");
    assert_eq!(
        version_count(&state),
        before + 2,
        "a refusal pushes nothing"
    );
}

/// Every observation says where it sits, whether or not anything has read it.
///
/// A candidate added to a track-stage track -- by the wire here, by a
/// descriptor search in the panel -- carries a seed and no keypoint until a
/// reading is run, and `pixel` is that one answer: the keypoint where there is
/// one, the seed where there is not. So an agent can look at a fresh candidate
/// without first evaluating the track, and it is looking at the place the
/// panel's own mark and tile are drawn at
/// ([`crate::bench::observation_site`]).
#[test]
fn every_observation_reports_where_it_sits_read_or_not() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let added = call(
        &mut state,
        &mut viewer,
        "add_bench_track_observation",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 3,
            "pixel": [130.5, 95.25],
        }),
    );
    let candidate = added["observation"].as_u64().expect("the index it took") as usize;

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = track["observations"].as_array().expect("the observations");
    assert_eq!(rows[candidate]["track"], Value::Null, "{track}");
    assert_eq!(
        rows[candidate]["pixel"],
        json!([130.5, 95.25]),
        "a fresh candidate does not say where it sits: {track}"
    );
    // And an observation a reading has written reports that keypoint, which is
    // the other half of the one rule.
    assert_eq!(
        rows[0]["pixel"], rows[0]["track"]["keypoint"],
        "a read observation reports something other than its keypoint: {track}"
    );
}

/// A seed carrying an affine shape puts that shape on the observation, which is
/// what a caller holding a detector's keypoint frame has to be able to state.
#[test]
fn a_cluster_seeded_with_an_affine_keeps_the_shape_it_was_given() {
    let (mut state, mut viewer) = benchable();
    let shape = json!([[7.1, -0.4], [0.4, 7.1]]);
    call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "affine": shape,
        }),
    );
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        track["observations"][0]["cluster"]["seed_shape"], shape,
        "{track}"
    );
}

/// The three seed forms are alternatives, and a call that mixes them is refused
/// rather than having one of them silently win.
#[test]
fn the_seed_forms_are_exclusive() {
    let (mut state, mut viewer) = benchable();
    let both = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "feature": 4,
        }),
    );
    assert!(both.0.contains("carries its own position"), "{both}");

    let sized_twice = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
            "affine": [[7.1, 0.0], [0.0, 7.1]],
        }),
    );
    assert!(
        sized_twice.0.contains("how large the patch is"),
        "{sized_twice}"
    );

    let nowhere = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({ "reconstruction_label": "run_a", "camera_image": 0 }),
    );
    assert!(
        nowhere.0.contains("needs somewhere to seed from"),
        "{nowhere}"
    );
}

/// A feature seed is read out of the image's `.sift` file, so a node that has
/// none is refused naming the image rather than seeding from nothing.
#[test]
fn a_feature_seed_on_a_node_with_no_sift_file_is_refused() {
    let (mut state, mut viewer) = benchable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "feature": 4 }),
    );
    assert!(error.0.contains(".sift"), "{error}");
    assert_eq!(version_count(&state), 1, "a refusal pushed a version");
}

/// A point put on the bench is an item `get_bench` lists, active, at the track
/// stage, seated on the point it came from.
#[test]
fn create_bench_track_lists_the_item_on_get_bench() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["active"]["track"], json!(item), "{bench}");
    let items = bench["items"].as_array().expect("the items");
    assert_eq!(items.len(), 1, "{bench}");
    assert_eq!(items[0]["item"], json!(item), "{bench}");
    assert_eq!(items[0]["kind"], json!("track"), "{bench}");
    assert_eq!(items[0]["active"], json!(true), "{bench}");
    assert_eq!(items[0]["stage"], json!("track"), "{bench}");
    assert_eq!(items[0]["origin"]["point"], json!(BENCH_POINT), "{bench}");
    assert_eq!(items[0]["counts"]["in"], json!(3), "{bench}");

    // And the track's own table says the same, with a row per observation.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let rows = track["observations"].as_array().expect("the observations");
    assert_eq!(rows.len(), 3, "{track}");
    assert_eq!(rows[0]["provenance"]["kind"], json!("origin"), "{track}");
    assert!(rows[0]["track"]["keypoint"].is_array(), "{track}");
    assert!(track["thresholds"]["min_zncc"].is_number(), "{track}");
}

/// A split answers with the label the half that came off took, and leaves two
/// items on the bench.
#[test]
fn split_answers_with_the_new_items_label() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let reply = call(
        &mut state,
        &mut viewer,
        "split_bench_track",
        json!({ "reconstruction_label": "run_a", "observations": [2] }),
    );
    let made = reply["item"].as_str().expect("the new item").to_string();
    assert_ne!(made, item, "{reply}");
    assert_eq!(reply["split_from"], json!(item), "{reply}");

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    let items = bench["items"].as_array().expect("the items");
    assert_eq!(items.len(), 2, "{bench}");
    let split = items
        .iter()
        .find(|entry| entry["item"] == json!(made))
        .expect("the half that came off");
    assert_eq!(split["counts"]["observations"], json!(1), "{bench}");
    // It comes off at the cluster stage: a split questions the position fitted
    // to both halves, so it is not carried onto the new one.
    assert_eq!(split["stage"], json!("cluster"), "{bench}");
}

/// The thresholds and the painting they produce are one step, and the bars a
/// call did not name stay where the track has them.
#[test]
fn apply_bench_track_thresholds_moves_the_bars_it_names() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let before = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    )["thresholds"]
        .clone();

    let reply = call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "min_zncc": 0.42 }),
    );
    assert_eq!(reply["thresholds"]["min_zncc"], json!(0.42), "{reply}");
    assert_eq!(
        reply["thresholds"]["max_shift_px"], before["max_shift_px"],
        "an unnamed bar moved: {reply}"
    );
    assert!(
        reply["report"]
            .as_str()
            .expect("a report")
            .contains("Applied the thresholds"),
        "{reply}"
    );
    let after = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(after["thresholds"]["min_zncc"], json!(0.42), "{after}");
}

/// A commit answers with the version it pushed and the sentence it recorded,
/// and an undo takes it back -- the bench steps being versions of the node like
/// any other, which is why there is no bench undo.
#[test]
fn commit_answers_with_a_version_an_undo_takes_back() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let points = state.scene[0].point_count();

    let reply = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(reply["item"], json!(item), "{reply}");
    assert_eq!(reply["dirty"], json!(true), "{reply}");
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Committed track:"), "{report}");
    assert_eq!(version_count(&state), 3);
    assert_eq!(state.scene[0].point_count(), points, "{report}");
    // It is an `Edit` row and the agent's, as every other commit's is, and the
    // selection the commit moved onto the written point is the row after it.
    let mut rows = rows(&state);
    let selection = rows.pop().expect("the commit selected what it wrote");
    assert!(
        selection.2.starts_with("Selected point "),
        "{}",
        selection.2
    );
    let last = rows.pop().expect("one row per step");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        state.scene[0]
            .edited()
            .point(BENCH_POINT)
            .is_some_and(|point| point.observations().len() == 3),
        "the undo did not restore the point the commit replaced"
    );
}

/// `deactivate_bench_item` is Track View's Edit box cleared: one version that
/// leaves the item on the bench, `get_bench` then reporting no active track
/// over a bench that has items, and a track tool that names none refused with
/// the remedies. A second call, with nothing active, is a no-effect reply.
#[test]
fn deactivate_leaves_the_item_and_nothing_active() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let reply = call(
        &mut state,
        &mut viewer,
        "deactivate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(reply["changed"], json!(true), "{reply}");
    assert_eq!(version_count(&state), before + 1, "{reply}");
    assert_eq!(
        reply["label"],
        json!(format!("Stopped editing {item}; it stays on the bench")),
        "{reply}"
    );

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["active"]["track"], Value::Null, "{bench}");
    assert_eq!(bench["items"].as_array().map(Vec::len), Some(1), "{bench}");

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        refused.0,
        "No track is active on run_a's bench. Name one with track, activate one with \
         activate_bench_item, or put one on with create_bench_track or create_bench_cluster."
    );

    let again = call(
        &mut state,
        &mut viewer,
        "deactivate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(again["changed"], json!(false), "{again}");
    assert_eq!(
        version_count(&state),
        before + 1,
        "a no-effect call pushed a version"
    );
    let report = again["report"].as_str().expect("the step's own sentence");
    assert!(report.contains("no effect"), "{again}");
}

/// The three item steps: a rename hands back the new label, an activation moves
/// which item a call that names none acts on, and a discard empties the bench.
#[test]
fn rename_activate_and_discard_answer_with_the_item_they_acted_on() {
    let (mut state, mut viewer) = benchable();
    let first = on_the_bench(&mut state, &mut viewer);
    let second = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    )["item"]
        .as_str()
        .expect("the new item")
        .to_string();

    let renamed = call(
        &mut state,
        &mut viewer,
        "rename_bench_item",
        json!({ "reconstruction_label": "run_a", "item": second, "label": "bull-nose" }),
    );
    assert_eq!(renamed["item"], json!("bull-nose"), "{renamed}");
    let gone = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": second }),
    );
    assert!(gone.0.contains("Nothing on the bench is called"), "{gone}");

    call(
        &mut state,
        &mut viewer,
        "activate_bench_item",
        json!({ "reconstruction_label": "run_a", "item": first }),
    );
    let active = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(active["item"], json!(first), "{active}");

    for item in [first.as_str(), "bull-nose"] {
        call(
            &mut state,
            &mut viewer,
            "discard_bench_item",
            json!({ "reconstruction_label": "run_a", "item": item }),
        );
    }
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["items"].as_array().expect("an array").len(), 0);
    assert_eq!(bench["active"]["track"], Value::Null, "{bench}");
}

/// Every refusal is the bench's own sentence and pushes no version.
#[test]
fn the_bench_refuses_in_its_own_words() {
    let (mut state, mut viewer) = benchable();

    // With nothing on the bench there is no active track to act on, and the
    // refusal says how to get one.
    let empty = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(empty.0.contains("create_bench_track"), "{empty}");

    let item = on_the_bench(&mut state, &mut viewer);
    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "track": "nothing-is-called-this",
            "observation": 0,
            "verdict": "out",
        }),
    );
    assert!(
        unknown.0.contains("Nothing on the bench is called"),
        "{unknown}"
    );

    let past_the_end = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": 9, "verdict": "out" }),
    );
    assert!(
        past_the_end.0.contains("no observation 9"),
        "{past_the_end}"
    );

    // A cluster cannot be committed: there is no position to write.
    call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let cluster = refused_call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(cluster.0.contains("cluster"), "{cluster}");

    let before = version_count(&state);
    let discarded = refused_call(
        &mut state,
        &mut viewer,
        "discard_bench_item",
        json!({ "reconstruction_label": "run_a", "item": "nothing-is-called-this" }),
    );
    assert!(
        discarded.0.contains("nothing on the bench is called"),
        "{discarded}"
    );
    assert_eq!(version_count(&state), before, "a refusal pushed a version");
    // The item that is there is untouched by any of it.
    assert!(state.bench_track(state.scene[0].id, &item).is_some());
}

/// The three steps that read photographs go to a worker and report through the
/// same two-level reply `bundle_adjust` uses.
#[test]
fn evaluate_fit_and_set_stage_run_as_background_tasks() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let staged = worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "cluster" }),
    );
    assert!(
        staged["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Set {item} to the cluster stage")),
        "{staged}"
    );
    assert_eq!(
        call(
            &mut state,
            &mut viewer,
            "get_bench_track",
            json!({ "reconstruction_label": "run_a" }),
        )["stage"],
        json!("cluster")
    );

    let evaluated = worked(
        &mut state,
        &mut viewer,
        "evaluate_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert!(
        evaluated["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Evaluated {item}")),
        "{evaluated}"
    );

    // The operation is the one an agent polls for, under the name the panel
    // shows it as.
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["running"], json!(false), "{task}");
    assert_eq!(task["operation"], json!("Evaluate track"), "{task}");
    assert_eq!(task["reconstruction_label"], json!("run_a"), "{task}");

    // And the measurements are on the wire, under the observation indexes they
    // were computed for.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        track["observations"][0]["cluster"]["zncc"].is_number(),
        "the evaluation measured nothing: {track}"
    );

    // The fit is its own step, under its own operation name, and it ends by
    // reading its result: the track stage's two distances are both on the wire
    // afterwards.
    let staged = worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "track" }),
    );
    assert!(staged["report"].is_string(), "{staged}");
    let fitted = worked(
        &mut state,
        &mut viewer,
        "fit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item, "search_px": 8.0 }),
    );
    assert!(
        fitted["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Fitted {item}")),
        "{fitted}"
    );
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["operation"], json!("Fit track"), "{task}");

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let measured = &track["observations"][0]["track"];
    for column in ["seed_shift_px", "projection_offset_px"] {
        assert!(
            measured[column].is_number(),
            "{column} is not on the wire: {track}"
        );
    }
    assert!(
        measured["reason"].is_null(),
        "a measured row carries no reason: {track}"
    );
}

/// An observation seeded a long way from the point's projection is **named** on
/// its row rather than searched for.
///
/// The gesture is one an agent makes by hand: a pixel typed into
/// `add_bench_track_observation` that is nowhere near where the point lands in
/// that photograph. The reading widens its window to reach the furthest seed
/// and each view's tile is the square of that width, so this is the call that
/// used to ask for hundreds of gigabytes; what it does now is say so on the row
/// and read everything else.
#[test]
fn an_observation_far_from_the_projection_is_named_rather_than_searched_for() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let added = call(
        &mut state,
        &mut viewer,
        "add_bench_track_observation",
        json!({
            "reconstruction_label": "run_a",
            "track": item,
            "camera_image": 5,
            "pixel": [24.0, 24.0],
        }),
    );
    let at = added["observation"]
        .as_u64()
        .expect("an add names the index it took") as usize;

    let evaluated = worked(
        &mut state,
        &mut viewer,
        "evaluate_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert!(evaluated["report"].is_string(), "{evaluated}");

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let row = &track["observations"][at]["track"];
    assert!(
        row["zncc"].is_null(),
        "nothing was searched for it: {track}"
    );
    let reason = row["reason"]
        .as_str()
        .expect("a row without a score says why");
    assert!(
        reason.contains("beyond the 64 px bound"),
        "the row names the bound it passed: {reason}"
    );
    // The rows that could be read were read: the bound takes one observation
    // out of the round and leaves the rest of the reading alone.
    let measured = track["observations"]
        .as_array()
        .expect("a list")
        .iter()
        .filter(|row| row["track"]["zncc"].is_number())
        .count();
    assert!(measured >= 2, "{track}");
}

/// Setting the stage a track is already at changes nothing: no task, no
/// version, the version the node stands at as the answer, and the step's own
/// no-effect sentence rather than the silence that would leave an agent reading
/// the previous step's label.
#[test]
fn setting_the_stage_a_track_is_already_at_starts_nothing() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let reply = call(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "track" }),
    );
    assert_eq!(version_count(&state), before, "{reply}");
    assert!(state.background_task().is_none());
    assert_eq!(reply["changed"], json!(false), "{reply}");
    let report = reply["report"].as_str().expect("the step's own sentence");
    assert!(report.contains("no effect"), "{reply}");
    assert!(report.contains("that stage already"), "{reply}");
    assert_ne!(
        report,
        reply["label"].as_str().expect("a version label"),
        "the reply echoed the previous step's label: {reply}"
    );
}

/// A commit that replaces a point hands the replacement an index one past the
/// end of the version it came from, and that index is a handle like any other:
/// what `get_scene` reports as the selection is what `get_point` and
/// `select_point` take back.
#[test]
fn a_committed_replacement_is_reachable_by_the_index_get_scene_reports() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": BENCH_POINT }),
    );
    call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );

    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    let selected = &scene["selection"]["point"];
    let index = selected["index"].as_u64().expect("a selected point");
    assert!(
        index
            >= scene["scene"][0]["counts"]["points"]
                .as_u64()
                .expect("a count"),
        "the fixture no longer exercises a sparse index: {scene}"
    );

    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": index }),
    );
    assert_eq!(point["index"], json!(index), "{point}");
    assert_eq!(point["id"], selected["id"], "{point}");
    // And the selection tool takes it too, which is the other half of the
    // handle being a handle.
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": index }),
    );
}

/// `get_scene`'s observation count is the version's and not the file's: a
/// commit that replaces a thirty-observation track with a two-observation one
/// moves it by twenty-eight.
#[test]
fn the_observation_count_follows_the_version() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let observations = |state: &mut AppState, viewer: &mut Viewer3D| -> u64 {
        call(state, viewer, "get_scene", json!({}))["scene"][0]["counts"]["observations"]
            .as_u64()
            .expect("a count")
    };
    let before = observations(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = track["observations"].as_array().expect("the rows").len();
    assert!(rows > 2, "the fixture needs a row to turn out: {track}");

    // One sighting refused, so the track commits with one fewer than the point
    // it replaces.
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": rows - 1, "verdict": "out" }),
    );
    call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );

    assert_eq!(
        observations(&mut state, &mut viewer),
        before - 1,
        "the count did not follow the commit"
    );
    // And an undo takes the observation back with the version.
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(observations(&mut state, &mut viewer), before);
}

/// The stage change's sentence is written once. The report used to carry the
/// stage phrase twice -- the viewer's, then core's report printed whole behind
/// it -- which is the one thing an agent reads to find out what happened.
#[test]
fn the_stage_report_states_the_stage_once() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let staged = worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "cluster" }),
    );
    let report = staged["report"].as_str().expect("a report");
    assert_eq!(
        report.matches("stage").count(),
        1,
        "the stage is stated twice: {report}"
    );
    assert!(
        report.starts_with(&format!("Set {item} to the cluster stage")),
        "{report}"
    );
    // The Action Log row is that sentence, up to the serials.
    let last = rows(&state).pop().expect("one row per step");
    assert!(last.2.starts_with(report), "{last:?} against {report}");
}

/// The two steps that read photographs decode **on the worker**: a node whose
/// photographs are neither decoded nor readable starts the task all the same,
/// and the reply is the handle the frame hands back once the window has passed.
#[test]
fn a_slow_evaluate_answers_with_a_handle_naming_it() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    // Nothing decoded: the evaluation has every photograph to read, which is
    // the work that must not happen before the deferral.
    state.full_res_cache.clear();

    let arguments = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("evaluate_bench_track", Some(&arguments)).expect("a valid call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("evaluate must defer to a worker"),
    };
    assert_eq!(pending.operation_name, "Evaluate track");

    // Past the window with the operation still the one running -- nothing has
    // drained its reports -- which is the handle case.
    let operation_id = pending.operation_id;
    let past = super::BackgroundReply {
        started: std::time::Instant::now() - super::REPLY_DIRECTLY_WITHIN,
        ..pending
    };
    let reply = match super::edit::background_reply(&state, &past).expect("past the window") {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("a handle is JSON"),
    };
    assert_eq!(reply["running"], json!(true), "{reply}");
    assert_eq!(reply["operation"], json!("Evaluate track"), "{reply}");
    assert_eq!(reply["operation_id"], json!(operation_id), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");

    // The photographs the worker could not read are its refusal and not the
    // gesture's: the step began either way.
    state.finish_background_task();
    match super::edit::background_reply(&state, &past).expect("the operation finished") {
        Err(e) => assert!(e.0.contains(&format!("Cannot evaluate {item}")), "{e}"),
        Ok(_) => panic!("the fixture's photographs are not on disk"),
    }
}

// ── The Image Detail panel's view ───────────────────────────────────────
//
// The panel's own geometry is the frame's, and these run without one, so the
// reading a drawn frame would have published is seeded directly -- which is
// exactly what the dock does with `ImageDetailResponse::view`. What is under
// test here is the boundary: that each target resolves to the right place in
// the right photograph, that the reply reports where the view landed rather
// than what the call asked for, and that a refusal names the fix. Where the
// arithmetic itself lands is `image_detail::view`'s, and `image_detail::tests`
// drives it through real frames.

/// The panel body these tests measure against, in points. Its aspect ratio
/// (4:3) differs from the demo photograph's (16:9), so the fit is settled by
/// the width and the letterboxed axis is a real one.
const VIEW_PANEL: [f32; 2] = [800.0, 600.0];

/// The demo photograph's size, which is its lens's.
const VIEW_IMAGE: [f32; 2] = [1920.0, 1080.0];

/// `alpha` with image 0 selected and the Image Detail panel standing at
/// `zoom`, centred, as a frame that drew it would have left things.
fn looking_at(zoom: f32) -> (AppState, Viewer3D) {
    let (mut state, viewer) = two_reconstructions();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));
    state.image_detail_view = Some(crate::image_detail::ViewGeometry {
        image,
        image_size: VIEW_IMAGE,
        panel_size: VIEW_PANEL,
        pan: [0.0, 0.0],
        zoom,
    });
    (state, viewer)
}

/// The image pixel at the centre of the panel, which is what every target
/// aims: the centre of the reported visible rectangle.
#[track_caller]
fn view_centre(reply: &Value) -> [f64; 2] {
    let view = &reply["image_detail_view"];
    let rect = view["visible_rect_px"]
        .as_array()
        .unwrap_or_else(|| panic!("a visible rectangle: {reply}"));
    let at = |i: usize| rect[i].as_f64().expect("a number");
    [(at(0) + at(2)) / 2.0, (at(1) + at(3)) / 2.0]
}

/// The reported visible rectangle's extent in image pixels.
#[track_caller]
fn view_extent(reply: &Value) -> [f64; 2] {
    let rect = reply["image_detail_view"]["visible_rect_px"]
        .as_array()
        .unwrap_or_else(|| panic!("a visible rectangle: {reply}"));
    let at = |i: usize| rect[i].as_f64().expect("a number");
    [at(2) - at(0), at(3) - at(1)]
}

#[track_caller]
fn assert_about(actual: [f64; 2], expected: [f64; 2], what: &str) {
    assert!(
        (actual[0] - expected[0]).abs() < 0.05 && (actual[1] - expected[1]).abs() < 0.05,
        "{what}: {actual:?} != {expected:?}"
    );
}

/// A pixel, a point's observation and a feature all mean "put this place at the
/// centre of the panel", and the reply says so in the one field that can be
/// checked against the request.
#[test]
fn a_pixel_target_centres_the_pixel_and_reports_the_zoom_it_took() {
    let (mut state, mut viewer) = looking_at(1.0);
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [300.0, 200.0], "zoom": 4.0 }),
    );
    assert_about(
        view_centre(&reply),
        [300.0, 200.0],
        "the pixel is off centre",
    );
    assert_eq!(reply["image_detail_view"]["zoom"], json!(4.0), "{reply}");
    assert_eq!(
        reply["image_detail_view"]["camera_image"],
        json!(0),
        "{reply}"
    );
    assert_eq!(
        reply["image_detail_view"]["image_size_px"],
        json!([1920.0, 1080.0]),
        "{reply}"
    );
    assert_eq!(
        reply["image_detail_view"]["panel_size_points"],
        json!([800.0, 600.0]),
        "{reply}"
    );
}

/// A rectangle settles its own zoom: it ends exactly as wide as the panel on
/// the axis that binds, and no narrower on the other.
#[test]
fn a_rect_target_fits_the_rectangle_to_the_panel() {
    let (mut state, mut viewer) = looking_at(1.0);
    // 400 x 150, wider in proportion than the 4:3 panel, so the width binds.
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [100.0, 100.0, 500.0, 250.0] }),
    );
    assert_about(
        view_centre(&reply),
        [300.0, 175.0],
        "the rect is off centre",
    );
    let [width, height] = view_extent(&reply);
    assert!(
        (width - 400.0).abs() < 0.05,
        "the binding axis does not fill the panel: {reply}"
    );
    assert!(
        height >= 150.0 - 0.05,
        "the rectangle does not fit: {reply}"
    );
    // The zoom the fit implies: the panel spans 400 px where fitted it spans
    // the whole 1920.
    let fit = (VIEW_PANEL[0] / VIEW_IMAGE[0]).min(VIEW_PANEL[1] / VIEW_IMAGE[1]);
    let expected = f64::from((VIEW_PANEL[0] / 400.0) / fit);
    let zoom = reply["image_detail_view"]["zoom"].as_f64().expect("a zoom");
    assert!((zoom - expected).abs() < 1e-3, "{zoom} != {expected}");
}

/// A corner-to-corner rectangle given the other way round is the rectangle the
/// caller drew, not an empty one.
#[test]
fn a_rect_named_from_its_far_corner_frames_the_same_region() {
    let (mut state, mut viewer) = looking_at(1.0);
    let forwards = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [100.0, 100.0, 500.0, 250.0] }),
    );
    let backwards = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [500.0, 250.0, 100.0, 100.0] }),
    );
    assert_eq!(forwards, backwards, "the corners' order changed the view");
}

/// `fit` is what `Z` does: the whole photograph, centred.
#[test]
fn a_fit_target_shows_the_whole_photograph() {
    let (mut state, mut viewer) = looking_at(8.0);
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "fit": true }),
    );
    assert_eq!(reply["image_detail_view"]["zoom"], json!(1.0), "{reply}");
    assert_about(view_centre(&reply), [960.0, 540.0], "the fit is off centre");
}

/// A point target is looked for in the photograph being looked at, and lands on
/// the pixel `get_point` reports for that row of its track.
#[test]
fn a_point_target_centres_that_point_s_observation() {
    let (mut state, mut viewer) = looking_at(1.0);
    let point = call(&mut state, &mut viewer, "get_point", json!({ "point": 0 }));
    let row = point["track"]
        .as_array()
        .expect("a track")
        .iter()
        .find(|row| row["camera_image_index"] == json!(0))
        .expect("the demo tracks observe every image")
        .clone();
    let xy = row["xy"].as_array().expect("a pixel");
    let expected = [
        xy[0].as_f64().expect("a number"),
        xy[1].as_f64().expect("a number"),
    ];

    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "point": 0, "zoom": 6.0 }),
    );
    assert_about(
        view_centre(&reply),
        expected,
        "the observation is off centre",
    );
}

/// A feature target reads the same cache the panel's own overlay draws its
/// ellipses from, so what an agent centres is the mark it is looking at.
#[test]
fn a_feature_target_centres_that_feature() {
    let (mut state, mut viewer) = looking_at(1.0);
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.sift_cache.insert(
        image,
        crate::state::CachedSiftFeatures {
            positions_xy: vec![[10.0, 20.0], [640.0, 480.0]],
            affine_shapes: vec![[[1.0, 0.0], [0.0, 1.0]]; 2],
            read_count: 2,
        },
    );
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "feature": 1, "zoom": 3.0 }),
    );
    assert_about(
        view_centre(&reply),
        [640.0, 480.0],
        "the feature is off centre",
    );

    let past_the_end = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "feature": 7 }),
    );
    // Out of range, one way or the other: this fixture has no `.sift` file on
    // disk, so the index past the end of the cache is refused by the read that
    // would have gone looking for it. Either way the refusal names the
    // photograph, which is what an agent needs to act on it.
    assert!(
        past_the_end.0.contains("images/A_000.jpg"),
        "{past_the_end}"
    );
}

/// A zoom past the panel's own range is clamped, and the reply says where it
/// landed rather than echoing what was asked for.
#[test]
fn a_zoom_past_the_panel_s_range_comes_back_clamped() {
    let (mut state, mut viewer) = looking_at(1.0);
    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [100.0, 100.0], "zoom": 500.0 }),
    );
    assert_eq!(
        reply["image_detail_view"]["zoom"],
        json!(f64::from(crate::image_detail::MAX_ZOOM)),
        "{reply}"
    );
    let floored = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [100.0, 100.0], "zoom": 0.25 }),
    );
    assert_eq!(
        floored["image_detail_view"]["zoom"],
        json!(1.0),
        "{floored}"
    );
}

/// The tool's own description carries the closest the panel goes, and the panel
/// owns that number.
#[test]
fn the_view_tool_advertises_the_panel_s_own_zoom_limit() {
    let catalog = tools::catalog();
    let spec = catalog
        .iter()
        .find(|spec| spec.name == "set_image_detail_view")
        .expect("the tool is in the catalog");
    let limit = format!("{:.1}", crate::image_detail::MAX_ZOOM);
    let zoom = spec.schema["properties"]["zoom"]["description"]
        .as_str()
        .expect("the zoom argument is described");
    assert!(
        zoom.contains(&limit),
        "the zoom argument does not name the panel's own limit {limit}: {zoom}"
    );
}

/// Every way of asking for nothing, or for two things at once, or for
/// something that is not there.
#[test]
fn the_view_tool_refuses_in_its_own_words() {
    let (mut state, mut viewer) = looking_at(1.0);

    let nothing = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "zoom": 2.0 }),
    );
    assert!(nothing.0.contains("no place to look"), "{nothing}");

    let both = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [10.0, 10.0], "fit": true }),
    );
    assert!(both.0.contains("2 places to look"), "{both}");

    // A rect settles its own zoom, so one beside it is a contradiction.
    let zoomed_rect = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [0.0, 0.0, 10.0, 10.0], "zoom": 2.0 }),
    );
    assert!(zoomed_rect.0.contains("settles its own"), "{zoomed_rect}");

    let flat = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "rect": [10.0, 10.0, 10.0, 90.0] }),
    );
    assert!(flat.0.contains("frames nothing"), "{flat}");

    // A track named with no bench observation to find in it.
    let stray = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "fit": true, "track": "bull-nose" }),
    );
    assert!(stray.0.contains("bench_observation"), "{stray}");

    // Nothing selected, and nothing named.
    call(&mut state, &mut viewer, "clear_selection", json!({}));
    let unselected = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "fit": true }),
    );
    assert!(unselected.0.contains("select_camera_image"), "{unselected}");
}

/// A point with no observation in the photograph being looked at is refused by
/// name, rather than the panel quietly moving to some other image.
#[test]
fn a_point_not_in_this_photograph_is_refused_rather_than_followed() {
    let (mut state, mut viewer) = looking_at(1.0);
    // A point of the demo whose track does not reach image 0. Found rather
    // than assumed: the demo's tracks are subsets of its camera ring, and
    // which point misses which image is the fixture's business.
    let missing = (0..40)
        .map(|index| {
            (
                index,
                call(
                    &mut state,
                    &mut viewer,
                    "get_point",
                    json!({ "point": index }),
                ),
            )
        })
        .find(|(_, point)| {
            !point["track"]
                .as_array()
                .expect("a track")
                .iter()
                .any(|row| row["camera_image_index"] == json!(0))
        })
        .map(|(index, _)| index)
        .expect("some demo point misses image 0");

    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "point": missing }),
    );
    assert!(
        refusal
            .0
            .contains(&format!("no observation of point {missing}")),
        "{refusal}"
    );
    assert!(refusal.0.contains("get_point"), "{refusal}");
    // And the photograph was not changed for it.
    assert_eq!(state.selected_image.map(|image| image.index()), Some(0));
}

/// One frame of the Image Detail panel, as the dock runs one: the standing
/// request taken and applied in a panel of [`VIEW_PANEL`] points, and the
/// reading published on `AppState`. That publication is the whole of what a
/// deferred view call waits for.
fn panel_draws(state: &mut AppState, image: crate::scene::ImageRef) {
    panel_draws_at(state, image, VIEW_PANEL);
}

/// The same, in a panel body of a named size: what a frame after a layout change
/// publishes, which is a different panel from the one before it.
fn panel_draws_at(state: &mut AppState, image: crate::scene::ImageRef, panel: [f32; 2]) {
    let mut geometry = crate::image_detail::ViewGeometry {
        image,
        image_size: VIEW_IMAGE,
        panel_size: panel,
        pan: [0.0, 0.0],
        zoom: 1.0,
    };
    if let Some(look) = state.take_look(image) {
        geometry = crate::image_detail::look_at(geometry, &look);
    }
    // Through the publication, not the field: the serial and the layout it was
    // drawn in are what a waiting call and a view tool read it by.
    state.publish_image_detail_view(geometry);
}

/// The wait a view call left behind, or a panic saying it did not wait.
#[track_caller]
fn deferred_view(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    arguments: Value,
) -> super::PendingView {
    let map = arguments
        .as_object()
        .cloned()
        .expect("test arguments are an object");
    let command = tools::parse("set_image_detail_view", Some(&map))
        .unwrap_or_else(|e| panic!("set_image_detail_view: {e}"));
    match agent(state, viewer, command) {
        Outcome::Deferred(super::Deferred::ImageDetailView(pending)) => pending,
        Outcome::Deferred(_) => panic!("expected the view's deferral, got another tool's"),
        Outcome::Done(Ok(_)) => panic!("expected the call to wait for a drawn frame"),
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
    }
}

/// What a waiting call is answered with, once a frame has drawn.
#[track_caller]
fn view_reply(state: &AppState, pending: &super::PendingView) -> Value {
    match super::display::pending_view_reply(state, pending).expect("the frame answered it") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
}

/// Before the panel has drawn anything there is no frame to do the arithmetic
/// in, so the read reports nulls and the write **waits** rather than refusing:
/// the look is standing, and the reply is the reading of the frame that applies
/// it.
#[test]
fn a_panel_that_has_never_drawn_reports_nothing_and_waits_for_a_frame() {
    let (mut state, mut viewer) = two_reconstructions();
    let view = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(
        view["image_detail_view"]["camera_image"],
        Value::Null,
        "{view}"
    );
    assert_eq!(view["image_detail_view"]["zoom"], Value::Null, "{view}");

    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));
    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "pixel": [300.0, 200.0], "zoom": 4.0 }),
    );
    assert_eq!(pending.image, image);
    assert!(
        super::display::pending_view_reply(&state, &pending).is_none(),
        "answered before any frame had drawn"
    );

    panel_draws(&mut state, image);
    let landed = view_reply(&state, &pending);
    assert_eq!(landed["image_detail_view"]["zoom"], json!(4.0), "{landed}");
    assert_about(
        view_centre(&landed),
        [300.0, 200.0],
        "the pixel is off centre",
    );
}

/// A frame that never comes is not waited on for ever: a photograph the
/// workspace no longer holds draws nothing however long it is given, so the
/// wait has a deadline and the answer past it says what did not happen.
#[test]
fn a_view_the_panel_never_draws_is_given_up_on() {
    let (mut state, mut viewer) = two_reconstructions();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));
    let mut pending = deferred_view(&mut state, &mut viewer, json!({ "fit": true }));
    pending.started = std::time::Instant::now() - std::time::Duration::from_secs(5);
    let refusal = match super::display::pending_view_reply(&state, &pending)
        .expect("the deadline answers it")
    {
        Err(refusal) => refusal,
        Ok(_) => panic!("a frame that never drew has no view to report"),
    };
    assert!(refusal.0.contains("has not drawn"), "{refusal}");
    assert!(
        refusal.0.contains(&state.image_name(image)),
        "the refusal does not name the photograph: {refusal}"
    );
}

/// A sighting *is* a place in a particular photograph, so a
/// `bench_observation` needs no selection and no open panel: the tool selects
/// the image, opens the panel if it was closed, and the look lands on the first
/// frame that draws it.
#[test]
fn a_bench_observation_target_selects_its_photograph_with_none_selected() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    state.hide_panel(Tab::ImageDetail);
    assert_eq!(state.selected_image, None, "the fixture selected an image");

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let row = &track["observations"][1];
    let pixel = row["pixel"].as_array().expect("where it sits").clone();
    let at = |i: usize| pixel[i].as_f64().expect("a number");
    let image = row["camera_image"].as_u64().expect("its photograph") as usize;

    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "reconstruction_label": "run_a", "bench_observation": 1, "zoom": 6.0 }),
    );
    assert_eq!(
        state.selected_image.map(|image| image.index()),
        Some(image),
        "the call did not select the photograph its target named"
    );
    assert!(
        state.panel_is_in_front(Tab::ImageDetail),
        "the call left the panel where nothing would draw it"
    );

    panel_draws(&mut state, pending.image);
    let landed = view_reply(&state, &pending);
    assert_eq!(landed["image_detail_view"]["zoom"], json!(6.0), "{landed}");
    assert_eq!(
        landed["image_detail_view"]["camera_image"],
        json!(image),
        "{landed}"
    );
    assert_about(
        view_centre(&landed),
        [at(0), at(1)],
        "the observation is off centre",
    );
}

/// `camera_image` names the photograph just as directly, so a `pixel` target
/// beside one also needs nothing selected.
#[test]
fn a_pixel_target_naming_a_camera_image_needs_no_selection() {
    let (mut state, mut viewer) = two_reconstructions();
    call(&mut state, &mut viewer, "clear_selection", json!({}));
    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "camera_image": 3, "pixel": [640.0, 480.0], "zoom": 8.0 }),
    );
    assert_eq!(state.selected_image.map(|image| image.index()), Some(3));

    panel_draws(&mut state, pending.image);
    let landed = view_reply(&state, &pending);
    assert_eq!(
        landed["image_detail_view"]["camera_image"],
        json!(3),
        "{landed}"
    );
    assert_about(
        view_centre(&landed),
        [640.0, 480.0],
        "the pixel is off centre",
    );
}

/// What is left to refuse: a target that names no photograph, with none
/// selected. The sentence says both ways out of it.
#[test]
fn a_pixel_target_with_no_photograph_named_or_selected_is_refused() {
    let (mut state, mut viewer) = two_reconstructions();
    call(&mut state, &mut viewer, "clear_selection", json!({}));
    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "pixel": [10.0, 20.0], "zoom": 4.0 }),
    );
    assert!(refusal.0.contains("camera_image"), "{refusal}");
    assert!(refusal.0.contains("select_camera_image"), "{refusal}");
    assert_eq!(state.look, None, "a refused call left a request standing");
}

/// `get_image_detail_view` answers from the panel's own last frame, which is
/// what a screenshot of it would show -- not from the selection, which a tool
/// may have moved since.
#[test]
fn get_image_detail_view_reports_the_panel_s_last_frame() {
    let (mut state, mut viewer) = looking_at(4.0);
    let before = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(before["image_detail_view"]["zoom"], json!(4.0), "{before}");
    assert_eq!(
        before["image_detail_view"]["camera_image_name"],
        json!("images/A_000.jpg"),
        "{before}"
    );

    // A selection the panel has not drawn yet does not move the reading.
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 5 }),
    );
    let after = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(after, before, "the reading followed the selection");
}

/// A bench observation names its own photograph, so walking a track's sightings
/// takes one argument.
#[test]
fn a_bench_observation_target_selects_its_own_camera_image() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let image = track["observations"][1]["camera_image"]
        .as_u64()
        .expect("an image index");
    // The panel is standing on some *other* photograph, so the target has to
    // move the selection for the view to mean anything.
    let first = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(first));
    let size = {
        let camera = &state.scene[0].recon().image_table.cameras[0];
        [camera.width as f32, camera.height as f32]
    };
    state.image_detail_view = Some(crate::image_detail::ViewGeometry {
        image: first,
        image_size: size,
        panel_size: VIEW_PANEL,
        pan: [0.0, 0.0],
        zoom: 1.0,
    });

    let reply = call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "reconstruction_label": "run_a", "bench_observation": 1, "track": item }),
    );
    assert_eq!(
        reply["image_detail_view"]["camera_image"],
        json!(image),
        "{reply}"
    );
    assert_eq!(
        state.selected_image.map(|image| image.index() as u64),
        Some(image),
        "the target did not select its own photograph"
    );

    let past_the_end = refused_call(
        &mut state,
        &mut viewer,
        "set_image_detail_view",
        json!({ "reconstruction_label": "run_a", "bench_observation": 99 }),
    );
    assert!(
        past_the_end.0.contains("no observation 99"),
        "{past_the_end}"
    );
}

// ── What the two photometric steps refuse before they start ─────────────

/// Turn every observation of the active track out but the first.
#[track_caller]
fn leave_one_in(state: &mut AppState, viewer: &mut Viewer3D) {
    let track = call(
        state,
        viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = track["observations"].as_array().expect("the rows").len();
    for observation in 1..rows {
        call(
            state,
            viewer,
            "set_bench_track_verdict",
            json!({
                "reconstruction_label": "run_a",
                "observation": observation,
                "verdict": "out",
            }),
        );
    }
}

/// A stage change the track alone rules out is refused **inline**, before a
/// photograph is read: no deferral, no operation, no version.
///
/// The failure this pins down is a refusal that cost a decode: the step used to
/// answer `running: true`, hand the task a dozen images to read, and fail it a
/// second later with a sentence that was knowable before anything started.
#[test]
fn a_stage_change_the_track_rules_out_is_refused_before_the_worker() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "cluster" }),
    );
    leave_one_in(&mut state, &mut viewer);
    let before = version_count(&state);

    // `refused_call` panics on a deferral, so this is also the assertion that
    // nothing went to a worker.
    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "track" }),
    );
    assert_eq!(
        refusal.0,
        format!(
            "Cannot set the stage of {item}: 1 observations are in, and the track stage needs \
             two or more"
        )
    );
    assert!(
        state.background_task().is_none(),
        "a refusal started a task"
    );
    assert_eq!(version_count(&state), before, "a refusal pushed a version");
}

/// The same for a fit of a track-stage track with nothing to register against
/// -- and, beside it, the reading of that very track, which is **not** refused:
/// one sighting is something to report, and only moving it needs a consensus.
#[test]
fn a_fit_the_track_rules_out_is_refused_before_the_worker() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    leave_one_in(&mut state, &mut viewer);
    let before = version_count(&state);

    let refusal = refused_call(
        &mut state,
        &mut viewer,
        "fit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        refusal.0,
        format!("Cannot fit {item}: 1 observations are in, and the track stage needs two or more")
    );
    assert!(
        state.background_task().is_none(),
        "a refusal started a task"
    );
    assert_eq!(version_count(&state), before, "a refusal pushed a version");

    let read = worked(
        &mut state,
        &mut viewer,
        "evaluate_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        read["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Evaluated {item}")),
        "a reading of one sighting is a measurement, not a refusal: {read}"
    );
}

// ── What a commit names ─────────────────────────────────────────────────

/// A commit names the point it wrote, by index and by the id a later call can
/// address it with, so an agent can `get_point` it without hunting through the
/// counts for whichever row is new.
#[test]
fn a_commit_names_the_point_it_wrote() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let reply = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let index = reply["point"]["index"]
        .as_u64()
        .unwrap_or_else(|| panic!("the commit named no point: {reply}"));
    let id = reply["point"]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("the commit minted no id: {reply}"))
        .to_string();
    // This track came off a point, so the commit replaced that point rather
    // than creating one, and the reply says which.
    assert_eq!(reply["point"]["replaced"], json!(BENCH_POINT), "{reply}");

    // Both handles resolve, which is the whole of what naming it is for.
    let by_index = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": index }),
    );
    assert_eq!(by_index["id"], json!(id), "{by_index}");
    let by_id = call(&mut state, &mut viewer, "get_point", json!({ "point": id }));
    assert_eq!(by_id["index"], json!(index), "{by_id}");
}

// ── The hash a version's point ids are minted from ──────────────────────

/// `get_scene`'s `content_hash` is the hash the version's point ids carry, so
/// an agent holding the field and an agent holding an id read off a point are
/// holding the same digits -- and an edit that mints a new one moves it.
///
/// The failure this pins down is a field that never moved: it reported the
/// *base*'s hash, which a point edit leaves exactly where it was, while the
/// point that edit created was already being named by the edit's own.
#[test]
fn the_scene_s_content_hash_is_the_hash_its_point_ids_carry() {
    let (mut state, mut viewer) = benchable();
    let hash_of = |state: &mut AppState, viewer: &mut Viewer3D| -> String {
        call(state, viewer, "get_scene", json!({}))["scene"][0]["content_hash"]
            .as_str()
            .expect("a content hash")
            .to_string()
    };
    let before = hash_of(&mut state, &mut viewer);
    assert_eq!(before.len(), 8, "a content hash is the id's eight digits");

    // A point the reconstruction had no row for: it is named by the hash of
    // the edit that made it, and that is what the node now reports. Pushed
    // directly, because what is under test is the field rather than any one
    // step, and every step that creates a point pushes this shape.
    let record = state.scene[0]
        .edited()
        .point(BENCH_POINT)
        .expect("a live point")
        .to_record();
    let node = &mut state.scene[0];
    let mut next = node.history.current().clone();
    let hash = next
        .point_edit_hash(std::slice::from_ref(&record))
        .expect("a hashable base");
    let created = next.add_point(record).expect("a well-formed record");
    node.history.push_creating(
        next,
        crate::document::PointMap::Removed(Vec::new()),
        "Created a point",
        Some(crate::document::CreatedPoints {
            hash,
            indexes: vec![created],
        }),
    );
    let id = state.scene[0].id;
    state.select_point(crate::scene::PointRef::new(id, created as usize));
    let after = hash_of(&mut state, &mut viewer);
    assert_ne!(
        after, before,
        "an edit that minted a new hash did not move it"
    );

    let selected = call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["point"]
        ["id"]
        .as_str()
        .expect("the created point is selected")
        .to_string();
    assert!(
        selected.starts_with(&format!("pt3d_{after}_")),
        "{selected} is not named by the hash get_scene reports ({after})"
    );

    // And an undo takes the node back to the hash it had, so the field tracks
    // the version rather than accumulating.
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(hash_of(&mut state, &mut viewer), before);
}

// ── What a cursor move does to the selection ────────────────────────────

/// Undo, redo and a jump are steps through a node's history, not statements
/// about what the person is looking at: the photograph on screen stays on
/// screen, and so does the point.
///
/// The failure this pins down is Image Detail going to "No image selected" on
/// every step of a run of bench steps, none of which touches the image table.
#[test]
fn a_history_move_keeps_the_photograph_and_the_point_selected() {
    let (mut state, mut viewer) = benchable();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 3 }),
    );
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": BENCH_POINT }),
    );
    let item = on_the_bench(&mut state, &mut viewer);
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": 1, "verdict": "out" }),
    );

    let selection = |state: &mut AppState, viewer: &mut Viewer3D| -> Value {
        call(state, viewer, "get_scene", json!({}))["selection"].clone()
    };
    let standing = selection(&mut state, &mut viewer);
    assert_eq!(standing["camera_image"]["index"], json!(3), "{standing}");

    for step in ["undo", "undo", "redo", "redo"] {
        call(
            &mut state,
            &mut viewer,
            step,
            json!({ "reconstruction_label": "run_a" }),
        );
        let now = selection(&mut state, &mut viewer);
        assert_eq!(
            now["camera_image"], standing["camera_image"],
            "{step} dropped the photograph"
        );
        assert_eq!(
            now["point"]["index"], standing["point"]["index"],
            "{step} dropped the point"
        );
    }

    // And a jump, which walks several steps at once.
    let history = call(
        &mut state,
        &mut viewer,
        "get_history",
        json!({ "reconstruction_label": "run_a" }),
    );
    let first = history["versions"][0]["serial"]
        .as_str()
        .expect("a serial")
        .to_string();
    call(
        &mut state,
        &mut viewer,
        "jump_to_version",
        json!({ "reconstruction_label": "run_a", "serial": first }),
    );
    let jumped = selection(&mut state, &mut viewer);
    assert_eq!(
        jumped["camera_image"], standing["camera_image"],
        "a jump dropped the photograph"
    );
    let _ = item;
}

/// The one move that *can* take the photograph away takes it away: undoing a
/// `delete_camera_image` puts the image back under a different index, and
/// redoing the delete leaves nothing to show.
#[test]
fn a_move_across_a_deleted_camera_image_follows_it_by_name() {
    let (mut state, mut viewer) = two_reconstructions();
    let name = call(
        &mut state,
        &mut viewer,
        "get_camera_image",
        json!({ "camera_image": 5 }),
    )["name"]
        .as_str()
        .expect("a name")
        .to_string();
    call(
        &mut state,
        &mut viewer,
        "select_camera_image",
        json!({ "camera_image": 5 }),
    );
    // Deleting an earlier image renumbers the one being looked at.
    call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 1 }),
    );
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "alpha" }),
    );
    let back =
        call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["camera_image"].clone();
    assert_eq!(
        back["name"],
        json!(name),
        "the undo did not put the selection back on the same photograph"
    );
    assert_eq!(back["index"], json!(5), "{back}");

    // Deleting the selected photograph itself leaves nothing to show, so the
    // selection clears -- and an undo does not resurrect it, because a
    // selection is not part of a version.
    call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": 5 }),
    );
    let gone =
        call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["camera_image"].clone();
    assert_eq!(gone, Value::Null, "the deleted photograph stayed selected");
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "alpha" }),
    );
    let after =
        call(&mut state, &mut viewer, "get_scene", json!({}))["selection"]["camera_image"].clone();
    assert_eq!(after, Value::Null, "{after}");
}

// ── The SIFT index and the search through it ────────────────────────────

/// `get_bench` reports the index a search would query, the two index tools give
/// a node one, and the search itself is a background task an agent polls for.
///
/// Over the workspace fixture rather than [`benchable`]: a search needs real
/// `.sift` files and a real `.kdf`, which is what
/// [`crate::sift_index::tests::searchable`] builds.
#[test]
fn the_sift_index_and_the_search_are_on_the_wire() {
    use crate::sift_index::tests as fixture;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, item) = fixture::searchable(dir.path());
    let label = state.node(id).expect("loaded").label.clone();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": label }),
    );
    let index = &bench["sift_index"];
    assert_eq!(index["state"], json!("current"), "{bench}");
    assert!(
        index["path"]
            .as_str()
            .expect("a path")
            .ends_with("demo-sift-index.kdf"),
        "{bench}"
    );
    assert!(
        index["descriptors"].as_u64().expect("a count") > 0,
        "{bench}"
    );

    let searched = worked(
        &mut state,
        &mut viewer,
        "search_bench_track_descriptors",
        json!({ "reconstruction_label": label, "track": item, "observation": 0 }),
    );
    let report = searched["report"].as_str().expect("a report");
    assert!(
        report.contains("Searched from observation 0 of 3") && report.contains("images matched"),
        "{searched}"
    );

    // The operation an agent polls for, under the name the panel shows.
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["operation"], json!("Search descriptors"), "{task}");

    // And the candidate it added is on the wire under the search's provenance.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": label, "track": item }),
    );
    let observations = track["observations"].as_array().expect("a list");
    assert_eq!(observations.len(), 4, "{track}");
    let added = observations.last().expect("the search added one");
    assert_eq!(
        added["camera_image"],
        json!(fixture::FOUND_IMAGE),
        "{track}"
    );
    assert_eq!(added["provenance"]["kind"], json!("search"), "{track}");
    assert!(added["provenance"]["inliers"].is_number(), "{track}");

    // Opening the same file again by path is the same index, and pushes no
    // version: an index is a file beside the workspace, not a value.
    let path = index["path"].as_str().expect("a path").to_string();
    let before = state.node(id).expect("loaded").history.versions().len();
    let opened = call(
        &mut state,
        &mut viewer,
        "open_sift_index",
        json!({ "reconstruction_label": label, "path": path }),
    );
    assert_eq!(opened["sift_index"]["state"], json!("current"), "{opened}");
    assert_eq!(
        state.node(id).expect("loaded").history.versions().len(),
        before,
        "an index is not a version"
    );
}

/// A step reports **its own** sentence, even when it sets something else off.
///
/// Putting the first item on a bench looks for the node's default descriptor
/// index, and finding one writes an Action Log row of its own -- after the
/// step's. The reply reads the log back for what the call did, so that row is
/// the viewer's rather than the caller's, and the reply skips it.
#[test]
fn a_create_that_opens_the_default_index_still_reports_the_create() {
    use crate::sift_index::tests as fixture;

    // One session builds the index; a second opens the same workspace with
    // nothing open yet, which is the state the first create meets.
    let dir = tempfile::tempdir().unwrap();
    let (_built, _, _) = fixture::searchable(dir.path());
    let (mut state, id) = fixture::state_in(dir.path());
    assert!(
        state.sift_index(id).is_none(),
        "the fresh session has opened nothing yet"
    );
    let label = state.node(id).expect("loaded").label.clone();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": label, "point": fixture::POINT }),
    );
    let item = made["item"].as_str().expect("a create names its item");
    let report = made["report"].as_str().expect("a create reports itself");
    assert!(
        report.starts_with("Put point") && report.contains(item),
        "the reply carries the step's own sentence: {made}"
    );
    // The index really did open in the middle of it, and said so in a row of
    // the viewer's own.
    assert!(state.sift_index(id).is_some());
    let opened = state
        .action_log
        .entries()
        .find(|entry| entry.text.starts_with("Opened the SIFT index"))
        .expect("the open wrote a row");
    assert_eq!(opened.actor, crate::action_log::Actor::Viewer);
}

/// The geometry search over the wire: it needs no SIFT index, reports under an
/// operation name of its own, pushes one version for its own sentence, and is
/// refused at the cluster stage in front of the worker rather than from inside
/// it. That a view clearing the gates arrives as a `sweep` candidate at the
/// patch's projection is held by core, over a scene with real texture
/// (`sfmtool-core/src/bench/tests.rs`); this fixture's cameras admit nothing,
/// which is what makes it the honest test of the wire rather than of the
/// kernel.
#[test]
fn search_bench_track_geometry_needs_no_index_and_reports_as_itself() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    )["observations"]
        .as_array()
        .expect("a list")
        .len();

    // No index is open on this node, and the search answers all the same --
    // which is the difference between it and search_bench_track_descriptors.
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["sift_index"]["state"], json!("none"), "{bench}");

    let versions = version_count(&state);
    let searched = worked(
        &mut state,
        &mut viewer,
        "search_bench_track_geometry",
        json!({ "reconstruction_label": "run_a", "track": item, "observation": 0 }),
    );
    let report = searched["report"].as_str().expect("a report");
    assert!(
        report.contains(&format!("Geometry search from observation 0 of {before}")),
        "the reply carries some other step's sentence: {searched}"
    );
    assert_eq!(version_count(&state), versions + 1, "{searched}");
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["operation"], json!("Geometry search"), "{task}");

    // Nothing this fixture offers clears the gates, and a search that admits
    // nothing moves no row it did not add.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let observations = track["observations"].as_array().expect("a list");
    assert_eq!(observations.len(), before, "{track}");
    for row in observations {
        assert_eq!(row["provenance"]["kind"], json!("origin"), "{track}");
    }

    // A cluster carries no geometry to project, and the refusal says so in
    // front of the worker rather than from inside it.
    worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "track": item, "stage": "cluster" }),
    );
    let command = tools::parse(
        "search_bench_track_geometry",
        Some(
            &json!({ "reconstruction_label": "run_a", "track": item, "observation": 0 })
                .as_object()
                .cloned()
                .expect("an object"),
        ),
    )
    .expect("a valid call");
    let error = refused(&mut state, &mut viewer, command);
    assert!(error.to_string().contains("track stage"), "{error}");
}

/// With no index, `get_bench` says so and still names where a build would put
/// one, and the search is refused in the step's own sentence rather than
/// deferring to a worker that could never answer.
#[test]
fn a_search_with_no_index_is_refused_and_get_bench_names_the_default_path() {
    use crate::sift_index::tests as fixture;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = fixture::state_in(dir.path());
    let label = state.node(id).expect("loaded").label.clone();
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    let item = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": label, "point": fixture::POINT }),
    )["item"]
        .as_str()
        .expect("a create names the item it made")
        .to_string();

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": label }),
    );
    assert_eq!(bench["sift_index"]["state"], json!("none"), "{bench}");
    assert!(
        bench["sift_index"]["path"]
            .as_str()
            .expect("the default path is named even when nothing is open")
            .ends_with("demo-sift-index.kdf"),
        "{bench}"
    );
    assert_eq!(bench["sift_index"]["descriptors"], Value::Null);

    let command = tools::parse(
        "search_bench_track_descriptors",
        Some(
            &json!({ "reconstruction_label": label, "track": item, "observation": 0 })
                .as_object()
                .cloned()
                .expect("an object"),
        ),
    )
    .expect("a valid call");
    let error = refused(&mut state, &mut viewer, command);
    assert!(
        error.to_string().contains("No SIFT index is open"),
        "{error}"
    );

    // And a build over a node whose images have no `.sift` companion is refused
    // the same way, in front of the worker.
    let command = tools::parse(
        "build_sift_index",
        Some(
            &json!({ "reconstruction_label": label })
                .as_object()
                .cloned()
                .expect("an object"),
        ),
    )
    .expect("a valid call");
    let error = refused(&mut state, &mut viewer, command);
    assert!(error.to_string().contains("No .sift file"), "{error}");
}

/// [`crate::state::edits::tests::projected_embedded_demo`] with [`BENCH_POINT`]
/// stored as a bearing: `w = 0`, a unit direction and a zero normal, which is
/// the row the format states for a point at infinity.
fn bearing_demo() -> sfmtool_core::SfmrReconstruction {
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    let point = &mut recon.point_set.points[BENCH_POINT as usize];
    point.position = nalgebra::Point3::from(point.position.coords.normalize());
    point.w = 0.0;
    point.normal = nalgebra::Vector3::zeros();
    recon.rebuild_derived_fields();
    recon
}

/// A bearing and a position are the same three numbers under different rules,
/// so the wire publishes the coordinate under the name of whichever it is, with
/// the flag beside it. An agent that read `position` off a `w = 0` track would
/// be holding a place one unit from the world origin.
#[test]
fn get_bench_track_publishes_a_bearing_as_a_direction() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(stage["at_infinity"], json!(true), "{track}");
    let direction = stage["direction"].as_array().expect("a bearing");
    assert_eq!(direction.len(), 3, "{track}");
    let norm: f64 = direction
        .iter()
        .map(|c| c.as_f64().expect("a number"))
        .map(|c| c * c)
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-9, "a unit direction: {track}");
    assert!(stage["position"].is_null(), "{track}");

    // And the finite point of the same fixture comes back the other way round.
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(stage["at_infinity"], json!(false), "{track}");
    assert!(stage["position"].is_array(), "{track}");
    assert!(stage["direction"].is_null(), "{track}");
}

/// A fit says which representation the rays earned and why, so an agent driving
/// the bench reads the decision rather than inferring it from a coordinate.
///
/// The demo's cameras sit on an arc forty-five degrees apart, so the three
/// sightings of a track on it resolve a depth easily: the stored bearing is
/// promoted, which is the answer, and the wire carries it.
#[test]
fn a_fit_of_a_bearing_reports_the_classification_and_can_promote_it() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    let item = on_the_bench(&mut state, &mut viewer);
    let fitted = worked(
        &mut state,
        &mut viewer,
        "fit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let report = fitted["report"].as_str().expect("a report");
    assert!(
        report.contains("finite at (") && report.contains("rays up to"),
        "the report should name the call and the evidence: {report}"
    );

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(
        stage["at_infinity"],
        json!(false),
        "the promotion is on the wire: {track}"
    );
    assert!(stage["position"].is_array(), "{track}");
}

// ── The no-effect contract, the clamp, and a frameless bearing ─────────────

/// A step told to do what has already been done answers successfully and says
/// so: no version, `changed: false`, the cursor where it was, and the step's
/// **own** sentence under `report` -- not the previous step's label, which is
/// what a reply assembled from the cursor alone echoes.
#[test]
fn a_bench_step_with_no_effect_pushes_nothing_and_says_so() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let pixel = track["observations"][0]["pixel"]
        .as_array()
        .expect("a sighting has a pixel")
        .iter()
        .map(|c| c.as_f64().expect("a number"))
        .collect::<Vec<f64>>();

    let before = version_count(&state);
    let cursor = call(
        &mut state,
        &mut viewer,
        "get_history",
        json!({ "reconstruction_label": "run_a" }),
    )["cursor"]
        .as_str()
        .expect("a cursor")
        .to_string();

    // The patch slid to the pixel its centre already sits under.
    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "pixel": [pixel[0], pixel[1]],
        }),
    );
    assert_eq!(moved["changed"], json!(false), "{moved}");
    assert_eq!(moved["serial"], json!(cursor), "{moved}");
    assert_eq!(moved["cursor"], json!(cursor), "{moved}");
    let report = moved["report"].as_str().expect("a sentence");
    assert!(report.contains("no effect"), "{report}");
    assert!(report.starts_with("Moved"), "{report}");
    assert!(!report.contains("by 0.000 units"), "{report}");

    // A turn of nothing, and a verdict the observation already carries.
    let turned = call(
        &mut state,
        &mut viewer,
        "spin_bench_patch",
        json!({ "reconstruction_label": "run_a", "track": item, "degrees": 0.0 }),
    );
    assert_eq!(turned["changed"], json!(false), "{turned}");
    assert!(
        turned["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{turned}"
    );

    let verdict = call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "verdict": "in",
        }),
    );
    assert_eq!(verdict["changed"], json!(false), "{verdict}");
    assert!(
        verdict["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{verdict}"
    );

    // A painting that proposes the verdicts the track already carries.
    let painted = call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(painted["changed"], json!(false), "{painted}");
    assert!(
        painted["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{painted}"
    );

    // A resize repeated: the second drag names the size the first left.
    let target = [pixel[0] + 3.0, pixel[1] + 1.0];
    let first = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "edge": "+u", "pixel": target,
        }),
    );
    assert_eq!(first["changed"], json!(true), "{first}");
    let again = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "edge": "+u", "pixel": target,
        }),
    );
    assert_eq!(again["changed"], json!(false), "{again}");
    assert!(
        again["report"]
            .as_str()
            .expect("a sentence")
            .contains("no effect"),
        "{again}"
    );

    // One version for the one drag that did something, and none for the rest.
    assert_eq!(
        version_count(&state),
        before + 1,
        "a no-effect step pushed a version"
    );
}

/// The commit is in the contract too: a second call on a track the point it
/// wrote already holds answers `changed: false`, names that point as any commit
/// names the one it wrote, and pushes no version.
#[test]
fn a_second_commit_of_the_same_track_answers_changed_false() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let first = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(first["changed"], json!(true), "{first}");
    let point = first["point"]["index"].clone();
    let cursor = first["cursor"].as_str().expect("a cursor").to_string();
    let before = version_count(&state);
    let points = state.scene[0].point_count();

    let again = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );

    assert_eq!(again["changed"], json!(false), "{again}");
    assert_eq!(again["serial"], json!(cursor), "{again}");
    assert_eq!(again["cursor"], json!(cursor), "{again}");
    assert_eq!(again["item"], json!(item), "{again}");
    assert_eq!(again["point"]["index"], point, "{again}");
    assert_eq!(again["point"]["replaced"], json!(null), "{again}");
    assert_eq!(again["point"]["id"], first["point"]["id"], "{again}");
    let report = again["report"].as_str().expect("a sentence");
    assert!(report.contains("no effect"), "{report}");
    assert!(report.starts_with("Committed "), "{report}");
    assert_eq!(
        version_count(&state),
        before,
        "a second commit is a version"
    );
    assert_eq!(state.scene[0].point_count(), points, "{again}");
}

/// A pixel off the photograph names no place on it, and the patch whose centre
/// was slid to meet that pixel's ray was flung across the reconstruction. The
/// step takes the nearest place the photograph does name, and the reply says so.
#[test]
fn a_bench_pixel_off_the_photograph_is_clamped_and_the_reply_says_so() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a", "track": item,
            "observation": 0, "pixel": [-500.0, -500.0],
        }),
    );
    assert_eq!(moved["clamped"], json!(true), "{moved}");
    assert_eq!(moved["clamped_from"], json!([-500.0, -500.0]), "{moved}");
    // `pixel` is where the moved centre now **projects**, not the clamp's own
    // output: the clamped place is unprojected onto the patch's plane, the patch
    // is carried there and the centre is projected back, and that round trip is
    // good to the arithmetic's last bits rather than bit for bit. So a corner
    // pixel comes back a rounding either side of the corner.
    let used = moved["pixel"].as_array().expect("the pixel it used");
    let camera = &state.scene[0].recon().image_table.cameras[0];
    for (value, extent) in used.iter().zip([camera.width, camera.height]) {
        let value = value.as_f64().expect("a number");
        assert!(
            value >= -1e-6 && value < f64::from(extent),
            "the centre landed at {value}, off a {extent} px axis: {moved}"
        );
    }
    let report = moved["report"].as_str().expect("a sentence");
    assert!(
        report.contains("clamped to the photograph from"),
        "{report}"
    );

    // A pixel on the photograph is left where it was named. Two px in on both
    // axes, and not along the edge the clamp above put the sighting on: the
    // number read back is a reprojection, so it sits a rounding either side of
    // that edge and a pixel named along it would be clamped for the rounding
    // rather than for anything the call said.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let pixel = track["observations"][0]["pixel"].clone();
    let at = pixel.as_array().expect("a pixel");
    let inside = call(
        &mut state,
        &mut viewer,
        "sight_bench_observation",
        json!({
            "reconstruction_label": "run_a", "track": item, "observation": 0,
            "pixel": [
                at[0].as_f64().expect("x") + 2.0,
                at[1].as_f64().expect("y") + 2.0,
            ],
        }),
    );
    assert_eq!(inside["clamped"], json!(false), "{inside}");
    assert_eq!(inside["clamped_from"], Value::Null, "{inside}");
}

/// [`bearing_demo`] with the patch-frame columns stripped: the bearing a node
/// with no `patch_u_halfvec` holds, which is every row of a `sift_files` value.
fn frameless_bearing_demo() -> sfmtool_core::SfmrReconstruction {
    let mut recon = bearing_demo();
    recon.point_set.patch_u_halfvec_xyz = None;
    recon.point_set.patch_v_halfvec_xyz = None;
    recon.rebuild_derived_fields();
    recon
}

/// The flag is the **track's** and not its patch's `w`, so a bearing put on the
/// bench from a node with no patch frames publishes the bearing it is. Reading
/// the frame answered `at_infinity: false` and put a unit direction under
/// `position`, which is a place one unit from the world origin.
#[test]
fn a_frameless_bearing_is_published_as_a_bearing_on_the_wire() {
    let (mut state, mut viewer) = benchable_with(frameless_bearing_demo());

    // The point itself says so, through `get_point`.
    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": BENCH_POINT }),
    );
    assert_eq!(point["at_infinity"], json!(true), "{point}");

    // And so does the track put on the bench from it.
    on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let stage = &track["stage_data"];
    assert_eq!(stage["frame_fitted"], json!(false), "{track}");
    assert_eq!(stage["at_infinity"], json!(true), "{track}");
    assert!(stage["position"].is_null(), "{track}");
    let direction = stage["direction"].as_array().expect("a bearing: {track}");
    let norm: f64 = direction
        .iter()
        .map(|c| c.as_f64().expect("a number"))
        .map(|c| c * c)
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-9, "a unit direction: {track}");
}

/// A view request is answered by the frame that draws it whenever the standing
/// reading is about a panel the dock no longer has. Answering from the stale
/// reading reported a `panel_size_points` the very next `get_image_detail_view`
/// contradicted, with the arithmetic done in a panel body that had gone.
#[test]
fn a_view_set_after_a_layout_change_reports_the_panel_as_it_now_is() {
    let (mut state, mut viewer) = two_reconstructions();
    let image = crate::scene::ImageRef::new(state.scene[0].id, 0);
    state.select_image(Some(image));

    // A frame draws the panel and publishes what it measured.
    panel_draws(&mut state, image);
    let view = call(&mut state, &mut viewer, "get_image_detail_view", json!({}));
    assert_eq!(
        view["image_detail_view"]["panel_size_points"],
        json!([VIEW_PANEL[0] as f64, VIEW_PANEL[1] as f64]),
        "{view}"
    );

    // The dock is re-laid out under it: the panel the reading describes is gone.
    lay_out(
        &mut state,
        Tab::ImageDetail,
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1366.0, 720.0)),
    );
    let pending = deferred_view(
        &mut state,
        &mut viewer,
        json!({ "pixel": [300.0, 200.0], "zoom": 2.0 }),
    );
    assert_eq!(pending.image, image);
    assert!(
        super::display::pending_view_reply(&state, &pending).is_none(),
        "answered from the reading the layout change invalidated"
    );

    // The frame that draws in the new layout is the one that answers.
    let after = [1366.0f32, 680.0f32];
    panel_draws_at(&mut state, image, after);
    let landed = view_reply(&state, &pending);
    assert_eq!(
        landed["image_detail_view"]["panel_size_points"],
        json!([after[0] as f64, after[1] as f64]),
        "{landed}"
    );
    assert_eq!(landed["image_detail_view"]["zoom"], json!(2.0), "{landed}");
    assert_about(
        view_centre(&landed),
        [300.0, 200.0],
        "the pixel is off centre",
    );
}
