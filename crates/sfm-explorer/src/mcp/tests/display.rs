// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
            change: super::super::DisplayChange {
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
            change: super::super::ImageDetailDisplayChange {
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
