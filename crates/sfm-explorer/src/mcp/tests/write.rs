// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
                camera_image: super::super::CameraImageSel::Index(99),
            },
            "out of range",
        ),
        (
            Command::SelectCameraImage {
                reconstruction_label: None,
                camera_image: super::super::CameraImageSel::Index(99),
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
            camera_image: super::super::CameraImageSel::Name("A_003.jpg".into()),
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
