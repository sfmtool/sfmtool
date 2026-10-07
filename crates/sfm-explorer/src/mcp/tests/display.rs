// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::viewer_3d::display::{Control, Field, FieldValue, Viewer3dDisplay};

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

// ── The 3D viewport's display controls ──────────────────────────────────

/// The document `get_viewer_3d_display` hands back, unwrapped.
#[track_caller]
fn viewer_3d_display(state: &mut AppState, viewer: &mut Viewer3D) -> Value {
    ok(state, viewer, Command::GetViewer3dDisplay)["viewer_3d_display"].clone()
}

/// Parse and apply a `set_viewer_3d_display`, returning its document.
#[track_caller]
fn set_viewer_3d(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Value {
    call(state, viewer, "set_viewer_3d_display", arguments)["viewer_3d_display"].clone()
}

/// A fresh viewer reports the HUD's defaults, every field under the name of
/// the field it is stored in.
#[test]
fn get_viewer_3d_display_returns_the_defaults() {
    let (mut state, mut viewer) = two_reconstructions();
    let document = viewer_3d_display(&mut state, &mut viewer);
    let mut names: Vec<&str> = document
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    let mut expected: Vec<&str> = Field::ALL.iter().map(|field| field.wire_name()).collect();
    names.sort_unstable();
    expected.sort_unstable();
    assert_eq!(names, expected);

    assert_eq!(document["show_points"], true);
    assert_eq!(document["show_grid"], true);
    assert_eq!(document["show_target_indicator"], false);
    assert_eq!(document["point_size_log2"], 0.0);
    assert_eq!(document["infinity_point_px"], 3.0);
    assert_eq!(document["edl_line_thickness"], 2.4);
    assert_eq!(document["maintain_z_up"], true);
    assert_eq!(document["show_fps"], true);
}

/// A call writes what it names and nothing else, and the reply is the whole
/// document as the read would return it.
#[test]
fn set_viewer_3d_display_changes_only_what_it_names() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = viewer_3d_display(&mut state, &mut viewer);
    let named = [
        "point_size_log2",
        "show_target_indicator",
        "show_grid",
        "maintain_z_up",
    ];
    let reply = set_viewer_3d(
        &mut state,
        &mut viewer,
        json!({
            "point_size_log2": 1.5,
            "show_target_indicator": true,
            "show_grid": false,
            "maintain_z_up": false,
        }),
    );
    assert_eq!(reply, viewer_3d_display(&mut state, &mut viewer));
    assert_eq!(state.point_size_log2, 1.5);
    assert!(state.show_target_indicator);
    assert!(!state.show_grid);
    assert!(!viewer.maintain_z_up, "maintain_z_up lives on the viewer");
    for (name, value) in before.as_object().expect("an object") {
        if !named.contains(&name.as_str()) {
            assert_eq!(&reply[name], value, "{name} moved");
        }
    }

    // And back on, which set_view cannot do.
    let reply = set_viewer_3d(&mut state, &mut viewer, json!({ "maintain_z_up": true }));
    assert_eq!(reply["maintain_z_up"], true);
    assert!(viewer.maintain_z_up);
}

/// A number inside the range is rounded to the decimals its slider shows, so
/// the HUD drawing it changes nothing; the wire reads the stored value.
#[test]
fn set_viewer_3d_display_rounds_to_the_slider_s_decimals() {
    let (mut state, mut viewer) = two_reconstructions();
    let reply = set_viewer_3d(
        &mut state,
        &mut viewer,
        json!({ "point_size_log2": 1.53, "patch_opacity": 0.333, "length_scale": 0.12345 }),
    );
    assert_eq!(reply["point_size_log2"], 1.5);
    assert_eq!(reply["patch_opacity"], 0.33);
    assert_eq!(reply["length_scale"], 0.123);
    assert_eq!(state.point_size_log2, 1.5);
}

/// Every number the viewer stores without a person or an agent choosing it —
/// the defaults, and the scene scale measured from the points at load — is
/// one the slider can hold: inside its range and at its decimals. So the value
/// the read reports is accepted when written back, and drawing the HUD does
/// not change it.
#[test]
fn a_seeded_scene_scale_reads_back_as_a_value_the_set_accepts() {
    use crate::viewer_3d::display::seed_length_scale;

    // The defaults first, before anything is seeded.
    let (state, viewer) = two_reconstructions();
    for (field, value) in Viewer3dDisplay::snapshot(&state, &viewer).fields() {
        if let (Control::Slider(range), FieldValue::Number(number)) = (field.control(), value) {
            assert_eq!(
                number,
                range.clamp_round(f64::from(number)),
                "{}'s default is not one its slider holds",
                field.wire_name()
            );
        }
    }

    // A scene in millimetres measures far above the slider's top, one in
    // kilometres below its bottom, and any scene to more decimals than it shows.
    for (seed, expected) in [(312.4_f32, 100.0_f32), (0.0002, 0.001), (0.65537, 0.655)] {
        let (mut state, mut viewer) = quiet_scene();
        seed_length_scale(&mut state, seed);
        assert_eq!(state.length_scale, expected, "seeded from {seed}");

        let read = viewer_3d_display(&mut state, &mut viewer)["length_scale"].clone();
        let reply = set_viewer_3d(
            &mut state,
            &mut viewer,
            json!({ "length_scale": read.clone() }),
        );
        assert_eq!(reply["length_scale"], read, "seeded from {seed}");
        // Writing back the value read is not a change, so it leaves no row.
        let displayed = |state: &AppState| {
            state
                .action_log
                .entries()
                .filter(|entry| entry.kind == Kind::Display)
                .count()
        };
        assert_eq!(displayed(&state), 0, "seeded from {seed}");

        // The HUD's slider clamps and rounds what it is handed on every frame
        // it draws; the seeded value is already what it would hold.
        let ctx = eframe::egui::Context::default();
        for _ in 0..3 {
            let input = eframe::egui::RawInput {
                screen_rect: Some(eframe::egui::Rect::from_min_size(
                    eframe::egui::Pos2::ZERO,
                    eframe::egui::vec2(1200.0, 800.0),
                )),
                ..Default::default()
            };
            crate::test_support::run_frame_headless(&ctx, input, |ui| {
                eframe::egui::CentralPanel::default().show(ui, |ui| {
                    viewer.show_hud(ui, &mut state, None, true);
                });
            });
        }
        assert!(viewer.hud_open, "the HUD was not drawn open");
        assert_eq!(state.length_scale, expected, "the HUD moved {seed}'s seed");
        assert_eq!(displayed(&state), 0, "the HUD recorded {seed}'s seed");
    }
}

/// `show_patches` is a master switch, and it is set whether or not anything
/// loaded carries patches; without them it draws nothing.
#[test]
fn show_patches_is_settable_without_patch_data() {
    let (mut state, mut viewer) = two_reconstructions();
    assert!(state.scene.iter().all(|node| !node.has_patch_data()));
    let reply = set_viewer_3d(&mut state, &mut viewer, json!({ "show_patches": false }));
    assert_eq!(reply["show_patches"], false);
    assert!(!state.show_patches);
}

/// Every refusal is at the parse, names the range, and leaves the call's good
/// fields unapplied.
#[test]
fn set_viewer_3d_display_refuses_a_value_its_slider_cannot_hold() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = viewer_3d_display(&mut state, &mut viewer);

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "show_grid": false, "point_size_log2": 4.0 }),
    );
    assert!(error.0.contains("point_size_log2"), "{error}");
    assert!(error.0.contains("-3 to 3"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "infinity_point_px": 0.5 }),
    );
    assert!(error.0.contains("1 to 16 px"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "target_fog_multiplier": 1000.0 }),
    );
    assert!(error.0.contains("0.5 to 100"), "{error}");

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "show_points": "yes" }),
    );
    assert!(error.0.contains("show_points"), "{error}");

    // The field of view is the view's, and stays with set_view.
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "fov_short_axis_deg": 60.0 }),
    );
    assert!(error.0.contains("has no argument"), "{error}");

    let error = refused_call(&mut state, &mut viewer, "set_viewer_3d_display", json!({}));
    assert!(error.0.contains("nothing to change"), "{error}");

    assert_eq!(viewer_3d_display(&mut state, &mut viewer), before);
}

/// A value that is not finite is refused like one outside the range. JSON
/// cannot carry one, so the range check is asked directly.
#[test]
fn a_slider_range_holds_no_value_that_is_not_finite() {
    for field in Field::ALL {
        if let Control::Slider(range) = field.control() {
            assert!(!range.contains(f64::NAN), "{}", field.wire_name());
            assert!(!range.contains(f64::INFINITY), "{}", field.wire_name());
            let (min, max) = range.decimal_ends();
            assert!(range.contains(min));
            assert!(range.contains(max));
        }
    }
}

/// A small negative value that rounds to zero is stored as `0.0`, not `-0.0`,
/// so it reads back and logs without a minus sign.
#[test]
fn a_slider_rounds_a_small_negative_value_to_positive_zero() {
    let rounded = Field::PointSizeLog2.range().round(-0.04);
    assert_eq!(rounded, 0.0);
    assert!(rounded.is_sign_positive());
    assert_eq!(
        Field::PointSizeLog2.text(FieldValue::Number(rounded)),
        "Point size 0.0"
    );
}

/// Both ends of every slider are accepted as the decimals the schema and the
/// refusal write them as, including ends such as `0.001` and `0.05` that no
/// `f32` holds exactly, and each reads back as that decimal.
#[test]
fn set_viewer_3d_display_accepts_each_end_of_every_slider() {
    let (mut state, mut viewer) = two_reconstructions();
    for field in Field::ALL {
        if let Control::Slider(range) = field.control() {
            for end in [range.min, range.max] {
                let written: f64 = end.to_string().parse().expect("an f32 prints as a number");
                let name = field.wire_name();
                let reply = set_viewer_3d(&mut state, &mut viewer, json!({ name: written }));
                assert_eq!(reply[name], written, "{name} at {written}");
            }
        }
    }
}

/// One `Display` entry per field the call changed, as the agent, in the words
/// the HUD records, and nothing for a field set to the value it had.
#[test]
fn set_viewer_3d_display_records_one_display_entry_per_changed_field() {
    let (mut state, mut viewer) = quiet_scene();
    call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({
            "show_grid": false,
            "point_size_log2": 1.5,
            "infinity_point_px": 6.0,
            "show_target_indicator": true,
            "maintain_z_up": false,
            "show_points": true,
        }),
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>(),
        [
            "Grid off",
            "Target indicator on",
            "Point size 1.5",
            "∞ point size 6.0 px",
            "Maintain Z-up off",
        ]
    );
    for entry in &entries {
        assert_eq!(entry.kind, Kind::Display);
        assert_eq!(entry.actor, Actor::Mcp);
    }

    // A repeat of the newest row's field folds into it.
    call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "maintain_z_up": true }),
    );
    assert_eq!(
        state.action_log.entries().last().map(|e| e.text.as_str()),
        Some("Maintain Z-up on")
    );
    assert_eq!(state.action_log.entries().count(), 5);

    // A field set to the value it had is not a change.
    let (mut state, mut viewer) = quiet_scene();
    call(
        &mut state,
        &mut viewer,
        "set_viewer_3d_display",
        json!({ "show_grid": true, "point_size_log2": 0.0 }),
    );
    assert_eq!(state.action_log.entries().count(), 0);
}

/// Every field's text is in the HUD's words: the HUD writes its entries
/// through the same `Field::text`, and these are the words it wrote before the
/// list existed.
#[test]
fn every_viewer_3d_display_field_records_the_hud_s_words() {
    let texts: Vec<String> = Field::ALL
        .iter()
        .map(|field| match field.control() {
            Control::Checkbox => field.text(FieldValue::Flag(false)),
            Control::Slider(_) => field.text(FieldValue::Number(1.0)),
        })
        .collect();
    assert_eq!(
        texts,
        [
            "Points off",
            "Camera Images off",
            "Grid off",
            "Patches off",
            "Points at ∞ off",
            "Target indicator off",
            "Point size 1.0",
            "∞ point size 1.0 px",
            "Scene scale 1.000",
            "Patch opacity 1.00",
            "Patch size 1.0",
            "Patch edge cutoff 1.00",
            "Maintain Z-up off",
            "EDL width 1.0",
            "Frustum size 1.00",
            "Target size 1.00",
            "Target fog 1.0",
            "Controls help off",
            "Frame rate off",
        ]
    );
}

/// The schema advertises each slider's range, from the constant the parse
/// checks against.
#[test]
fn set_viewer_3d_display_advertises_the_slider_ranges() {
    let spec = tools::catalog()
        .iter()
        .find(|spec| spec.name == "set_viewer_3d_display")
        .expect("in the catalog");
    let properties = &spec.schema["properties"];
    let point_size = &properties["point_size_log2"];
    assert_eq!(point_size["minimum"], -3.0);
    assert_eq!(point_size["maximum"], 3.0);
    // Ends that no `f32` holds exactly are advertised as their decimals.
    assert_eq!(properties["length_scale"]["minimum"], 0.001);
    assert_eq!(properties["frustum_size_multiplier"]["minimum"], 0.05);
    // Every slider's schema ends are the decimals the refusal writes, and the
    // ends the parse accepts.
    let written = |end: f32| -> f64 { end.to_string().parse().expect("a number") };
    for field in Field::ALL {
        if let Control::Slider(range) = field.control() {
            let name = field.wire_name();
            let (min, max) = (written(range.min), written(range.max));
            assert_eq!(properties[name]["minimum"], min, "{name}");
            assert_eq!(properties[name]["maximum"], max, "{name}");
            assert_eq!(range.decimal_ends(), (min, max), "{name}");
        }
    }
    assert_eq!(
        spec.schema["properties"]["show_target_indicator"]["type"],
        "boolean"
    );
}

/// The read is a query, like every other read on the surface, and a refused
/// set is a protocol error that never reaches the viewer.
#[test]
fn get_viewer_3d_display_is_a_query() {
    let (mut state, mut viewer) = quiet_scene();
    tools::parse(
        "set_viewer_3d_display",
        json!({ "point_size_log2": 9.0 }).as_object(),
    )
    .expect_err("off the slider");
    assert_eq!(state.action_log.entries().count(), 0);

    ok(&mut state, &mut viewer, Command::GetViewer3dDisplay);
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, Kind::Query("get_viewer_3d_display"));
    assert_eq!(entries[0].text, "get_viewer_3d_display");
}
