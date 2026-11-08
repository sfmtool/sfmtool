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
    // The image and camera lists are dense and are refused by their count; a
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
            Command::GetIntrinsicAdjustments {
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

// ── Quitting ────────────────────────────────────────────────────────────

/// Parse a `quit` call and apply it as the agent.
fn quit(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Outcome {
    let map = arguments.as_object().cloned().expect("an object");
    let command = tools::parse("quit", Some(&map)).expect("a quit call parses");
    agent(state, viewer, command)
}

/// Delete a point of `alpha`, which leaves it with changes not on disk.
fn dirty_alpha(state: &mut AppState, viewer: &mut Viewer3D) {
    call(
        state,
        viewer,
        "delete_point",
        json!({ "reconstruction_label": "alpha", "point": 3 }),
    );
    assert!(state.any_dirty(), "the delete left alpha dirty");
}

#[test]
fn quit_parses_its_force_argument_and_refuses_a_non_boolean() {
    let parse =
        |arguments: Value| tools::parse("quit", Some(arguments.as_object().expect("an object")));
    assert_eq!(
        parse(json!({})),
        Ok(Command::Quit {
            discard_unsaved: false
        })
    );
    assert_eq!(
        parse(json!({ "force_quit_no_confirmation_no_save": true })),
        Ok(Command::Quit {
            discard_unsaved: true
        })
    );
    let error = parse(json!({ "force_quit_no_confirmation_no_save": "yes" }))
        .expect_err("a string is not a boolean");
    assert!(
        error.0.contains("force_quit_no_confirmation_no_save"),
        "{error}"
    );
}

#[test]
fn quit_on_a_clean_scene_says_the_viewer_is_quitting() {
    let (mut state, mut viewer) = two_reconstructions();
    match quit(&mut state, &mut viewer, json!({})) {
        Outcome::Quit(reply) => {
            assert_eq!(reply["quitting"], json!(true), "{reply}");
            assert_eq!(reply["discarded"], json!([]), "{reply}");
            let message = reply["message"].as_str().expect("a message");
            assert!(message.contains("The viewer is quitting"), "{message}");
        }
        _ => panic!("a quit with nothing unsaved must quit"),
    }
}

/// File ▸ Quit puts up the close prompt where a node is dirty; the tool,
/// which cannot answer a prompt, refuses in the prompt's words and leaves
/// everything where it was.
#[test]
fn quit_with_unsaved_changes_is_refused_naming_them() {
    let (mut state, mut viewer) = two_reconstructions();
    dirty_alpha(&mut state, &mut viewer);
    let map = json!({}).as_object().cloned().expect("an object");
    let command = tools::parse("quit", Some(&map)).expect("a quit call parses");
    let error = refused(&mut state, &mut viewer, command);
    assert!(
        error.0.contains("alpha has changes that are not on disk."),
        "{error}"
    );
    assert!(!error.0.contains("beta"), "{error}");
    assert!(error.0.contains("save_reconstruction"), "{error}");
    assert!(
        error.0.contains("force_quit_no_confirmation_no_save: true"),
        "{error}"
    );
    assert_eq!(state.scene.len(), 2);
    assert!(state.is_dirty(state.scene[0].id), "the refusal kept alpha");
    let last = state.action_log.entries().last().expect("an entry");
    assert!(last.failed, "{last:?}");
    assert_eq!(last.actor, Actor::Mcp);
    assert_eq!(last.kind, Kind::Session);
    assert!(last.text.starts_with("quit failed: "), "{}", last.text);
}

#[test]
fn a_forced_quit_with_unsaved_changes_quits_and_names_what_it_lost() {
    let (mut state, mut viewer) = two_reconstructions();
    dirty_alpha(&mut state, &mut viewer);
    match quit(
        &mut state,
        &mut viewer,
        json!({ "force_quit_no_confirmation_no_save": true }),
    ) {
        Outcome::Quit(reply) => {
            assert_eq!(reply["quitting"], json!(true), "{reply}");
            assert_eq!(reply["discarded"], json!(["alpha"]), "{reply}");
        }
        _ => panic!("a forced quit must quit"),
    }
}
