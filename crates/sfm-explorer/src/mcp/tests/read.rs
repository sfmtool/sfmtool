// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
    assert_eq!(by_index["camera_intrinsics_index"], 1);
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
    assert_eq!(lens["camera_intrinsics_index"], 1);
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
    // The fixture sits beside no .sift file, so nothing is detected and a
    // sift_files value has no observed pixel either; the field is still there.
    let outermost = &lens["outermost_keypoint"];
    assert_eq!(outermost["detected"], Value::Null, "{outermost}");
    assert_eq!(outermost["detected_camera_images"], 0, "{outermost}");
}

/// An intrinsics handle copied from any reply is the argument for the next
/// tool, without digging into a nested record or translating a bare `index`.
#[test]
fn camera_intrinsics_has_one_handle_across_replies() {
    let (mut state, mut viewer) = two_reconstructions();
    let page = call(
        &mut state,
        &mut viewer,
        "list_camera_images",
        json!({ "reconstruction_label": "alpha", "offset": 5, "limit": 1 }),
    );
    let row = &page["camera_images"][0];
    assert_eq!(row["index"], 5, "the row's index names the image");
    let lens_index = row["camera_intrinsics_index"].clone();

    let image = call(
        &mut state,
        &mut viewer,
        "get_camera_image",
        json!({ "reconstruction_label": "alpha", "camera_image": row["name"] }),
    );
    assert_eq!(image["camera_intrinsics_index"], lens_index);
    assert!(image["camera_intrinsics"].get("index").is_none());

    let lens = call(
        &mut state,
        &mut viewer,
        "get_camera_intrinsics",
        json!({ "reconstruction_label": "alpha", "camera_intrinsics_index": lens_index }),
    );
    assert_eq!(
        lens["camera_intrinsics_index"],
        image["camera_intrinsics_index"]
    );
    assert!(lens.get("index").is_none());

    let selected = call(
        &mut state,
        &mut viewer,
        "select_camera_intrinsics",
        json!({ "reconstruction_label": "alpha", "camera_intrinsics_index": lens["camera_intrinsics_index"] }),
    );
    let selection = &selected["selection"]["camera_intrinsics"];
    assert_eq!(
        selection["camera_intrinsics_index"],
        lens["camera_intrinsics_index"]
    );
    assert!(selection.get("index").is_none());

    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    assert_eq!(scene["selection"]["camera_intrinsics"], *selection);
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
    assert_eq!(selection["camera_intrinsics"]["camera_intrinsics_index"], 1);
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
    assert_eq!(selection["camera_intrinsics"]["camera_intrinsics_index"], 0);
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
    assert_eq!(selection["camera_intrinsics"]["camera_intrinsics_index"], 1);
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

/// `get_camera_image` reports the image's intrinsic adjustment and the focal
/// and principal point it was taken with, and `get_camera_intrinsics` which
/// parameters its camera varies per image.
#[test]
fn get_camera_image_reports_the_image_s_own_intrinsics() {
    use crate::test_support::{with_adjustment, ADJUSTMENT};

    let (_, mut viewer) = two_reconstructions();
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/alpha.sfmr"),
        with_adjustment(recon(8, "A"), 5, ADJUSTMENT),
    ));
    let image = |state: &mut AppState, viewer: &mut Viewer3D, index: usize| {
        call(
            state,
            viewer,
            "get_camera_image",
            json!({ "camera_image": index }),
        )
    };

    // The demo camera: focal 1000 px, principal point (960, 540), 1920 × 1080.
    let adjusted = image(&mut state, &mut viewer, 5);
    assert_eq!(
        adjusted["intrinsic_adjustment"],
        json!({ "focal": 0.01, "principal_point": [0.02, -0.01] })
    );
    let intrinsics = &adjusted["image_intrinsics"];
    let number = |value: &Value| value.as_f64().expect("a number");
    for (k, expected) in [(0, 1010.0), (1, 1010.0)] {
        let got = number(&intrinsics["focal_lengths"][k]);
        assert!((got - expected).abs() < 1e-9, "{intrinsics}");
    }
    for (k, expected) in [(0, 960.0 + 21.6), (1, 540.0 - 10.8)] {
        let got = number(&intrinsics["principal_point"][k]);
        assert!((got - expected).abs() < 1e-9, "{intrinsics}");
    }
    // The shared camera's block is unchanged.
    assert_eq!(adjusted["camera_intrinsics"]["camera_model"], "PINHOLE");

    // An image of the same camera with no adjustment reads the camera as it is.
    let plain = image(&mut state, &mut viewer, 4);
    assert_eq!(
        plain["intrinsic_adjustment"],
        json!({ "focal": 0.0, "principal_point": [0.0, 0.0] })
    );
    assert_eq!(
        plain["image_intrinsics"],
        json!({ "focal_lengths": [1000.0, 1000.0], "principal_point": [960.0, 540.0] })
    );

    let lens = |state: &mut AppState, viewer: &mut Viewer3D, index: usize| {
        call(
            state,
            viewer,
            "get_camera_intrinsics",
            json!({ "camera_intrinsics_index": index }),
        )
    };
    assert_eq!(
        lens(&mut state, &mut viewer, 1)["per_image_adjustments"],
        json!(["focal", "principal_point"])
    );
    assert_eq!(
        lens(&mut state, &mut viewer, 0)["per_image_adjustments"],
        json!([])
    );
}
