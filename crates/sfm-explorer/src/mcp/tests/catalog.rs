// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
        ("get_viewer_3d_display", json!({})),
        (
            "set_viewer_3d_display",
            json!({ "point_size_log2": 1.5, "show_target_indicator": true }),
        ),
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
        (
            "set_reconstruction_transform",
            json!({
                "reconstruction_label": "alpha",
                "transform": {
                    "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                    "scale": 1.0,
                },
            }),
        ),
        (
            "set_reconstruction_transform_from_patch",
            json!({ "reconstruction_label": "alpha", "mode": "set_to_origin" }),
        ),
        (
            "bake_reconstruction_transform",
            json!({ "reconstruction_label": "alpha" }),
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
        (
            "add_camera_image_to_tracks",
            json!({ "reconstruction_label": "alpha", "camera_image": 0 }),
        ),
        ("bundle_adjust", json!({ "reconstruction_label": "alpha" })),
        (
            "switch_camera_model",
            json!({ "reconstruction_label": "alpha", "camera_intrinsics_index": 0 }),
        ),
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
            "create_track_at_pixel",
            json!({
                "reconstruction_label": "alpha",
                "camera_image": 0,
                "pixel": [142.0, 197.5],
            }),
        ),
        (
            "find_nearby_tracks",
            json!({
                "reconstruction_label": "alpha",
                "camera_image": 0,
                "pixel": [142.0, 197.5],
                "commit": false,
                "label": "nose",
            }),
        ),
        (
            "focus_bench_item",
            json!({ "reconstruction_label": "alpha", "item": "bull-nose" }),
        ),
        ("unfocus_bench_item", json!({})),
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
            "select_bench_observations",
            json!({ "reconstruction_label": "alpha", "observations": [1] }),
        ),
        (
            "commit_bench_track",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "fit_bench_track",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "fit_bench_track_normal",
            json!({ "reconstruction_label": "alpha", "method": "photometric" }),
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
            "open_index_files",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "build_index_files",
            json!({ "reconstruction_label": "alpha" }),
        ),
        (
            "close_index_files",
            json!({ "reconstruction_label": "alpha" }),
        ),
        ("get_background_task", json!({})),
        ("cancel_background_task", json!({})),
        ("screenshot", json!({})),
        (
            "get_widgets",
            json!({ "panel_name": "scene", "crop_px": [0, 0, 100, 50] }),
        ),
        (
            "click",
            json!({ "panel_name": "scene", "at_px": [10, 10], "mouse_button": "right" }),
        ),
        ("hover", json!({ "widget": "003233f928812d8a" })),
        (
            "press_key",
            json!({ "key": "Z", "modifiers": ["command"], "panel_name": "viewer_3d" }),
        ),
        (
            "type_text",
            json!({ "text": "pt3d", "widget": "003233f928812d8a" }),
        ),
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

/// Every representative call carries every argument its schema marks
/// required, so a client that validates against the schema can form at least
/// that call. A tool whose alternative arguments are all listed as required
/// fails here, since no single valid call can carry both alternatives.
#[test]
fn every_representative_call_satisfies_its_schema_s_required_list() {
    let calls = representative_tool_calls();
    for spec in tools::catalog() {
        let arguments = calls
            .iter()
            .find_map(|(name, arguments)| (*name == spec.name).then_some(arguments))
            .expect("every tool has a representative call");
        let required = spec.schema["required"].as_array().cloned();
        for required in required.iter().flatten() {
            let required = required.as_str().expect("a property name");
            assert!(
                arguments.get(required).is_some(),
                "{} marks {required:?} required, but its valid call {arguments} has none",
                spec.name
            );
        }
    }
}

/// `translate_bench_patch` and `resize_bench_patch` each take one of two
/// forms. The schema requires only the reconstruction, each form parses on its
/// own, and the parser refuses a call that gives both forms, neither, or an
/// edge argument that belongs to the other form.
#[test]
fn the_bench_patch_forms_are_optional_in_the_schema_and_exclusive_in_the_parser() {
    for tool in ["translate_bench_patch", "resize_bench_patch"] {
        let spec = tools::catalog()
            .iter()
            .find(|spec| spec.name == tool)
            .expect("advertised");
        assert_eq!(
            spec.schema["required"],
            json!(["reconstruction_label"]),
            "{tool}"
        );
    }

    let parse = |tool: &str, arguments: Value| {
        tools::parse(tool, Some(arguments.as_object().expect("an object")))
    };
    for (tool, arguments) in [
        (
            "translate_bench_patch",
            json!({ "reconstruction_label": "a", "by": [0.1, 0.0, 0.0] }),
        ),
        (
            "translate_bench_patch",
            json!({ "reconstruction_label": "a", "camera_image": 2, "pixel": [1.0, 2.0] }),
        ),
        (
            "resize_bench_patch",
            json!({ "reconstruction_label": "a", "half_length": 0.5 }),
        ),
        (
            "resize_bench_patch",
            json!({ "reconstruction_label": "a", "half_length": 0.5, "moved_edge": "-v" }),
        ),
    ] {
        parse(tool, arguments.clone())
            .unwrap_or_else(|error| panic!("{tool} {arguments}: {error}"));
    }

    for (tool, arguments, expected) in [
        (
            "translate_bench_patch",
            json!({
                "reconstruction_label": "a", "by": [0.1, 0.0, 0.0],
                "observation": 0, "pixel": [1.0, 2.0],
            }),
            "give one",
        ),
        (
            "translate_bench_patch",
            json!({ "reconstruction_label": "a" }),
            "needs either by",
        ),
        (
            "resize_bench_patch",
            json!({ "reconstruction_label": "a" }),
            "needs either half_length",
        ),
        (
            "resize_bench_patch",
            json!({
                "reconstruction_label": "a", "half_length": 0.5,
                "observation": 0, "edge": "+u", "pixel": [1.0, 2.0],
            }),
            "give one",
        ),
        (
            "resize_bench_patch",
            json!({ "reconstruction_label": "a", "half_length": 0.5, "edge": "+u" }),
            "moved_edge",
        ),
        (
            "resize_bench_patch",
            json!({
                "reconstruction_label": "a", "observation": 0, "edge": "+u",
                "moved_edge": "+u", "pixel": [1.0, 2.0],
            }),
            "moved_edge belongs to the half_length form",
        ),
    ] {
        let error = parse(tool, arguments.clone()).expect_err("refused");
        assert!(error.0.contains(expected), "{tool} {arguments}: {error}");
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

/// Exercise the nested objects parsed by `Args` using the valid catalog
/// calls. The schema supplies the probe names; the parser must recognize each
/// advertised name and reject a name absent from the schema.
#[test]
fn nested_argument_names_agree_with_the_catalog() {
    let calls = representative_tool_calls();
    for (tool, field) in [
        ("set_reconstruction_transform", "transform"),
        ("move_camera_image", "world_from_camera"),
        ("set_view", "look_through"),
        ("set_view", "move"),
        ("set_view", "turn"),
        ("set_view", "orbit"),
    ] {
        let spec = tools::catalog()
            .iter()
            .find(|spec| spec.name == tool)
            .expect("advertised tool");
        let schema = &spec.schema["properties"][field];
        assert_eq!(schema["additionalProperties"], false, "{tool}.{field}");
        let properties = schema["properties"].as_object().expect("nested object");
        let base = if tool == "set_view" {
            match field {
                "look_through" => json!({ "look_through": { "camera_image": 0 } }),
                "move" => json!({ "move": { "forward": 1.0 } }),
                _ => json!({ field: { "yaw_deg": 1.0 } }),
            }
        } else {
            calls
                .iter()
                .find(|(name, _)| *name == tool)
                .expect("representative call")
                .1
                .clone()
        };

        for name in properties.keys() {
            let mut arguments = base.clone();
            arguments[field][name] = json!("<probe>");
            if let Err(error) = tools::parse(tool, arguments.as_object()) {
                assert!(
                    !error.0.contains("has no argument"),
                    "{tool}.{field} advertises {name:?} but rejects it: {error}"
                );
            }
        }

        let mut arguments = base;
        arguments[field]["unknown_argument"] = json!(true);
        let error = tools::parse(tool, arguments.as_object()).expect_err("unknown nested key");
        assert!(
            error.0.starts_with(&format!(
                "{tool}.{field} has no argument \"unknown_argument\" — "
            )),
            "{tool}.{field}: {error}"
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

    for (tool, arguments, expected) in [
        (
            "set_reconstruction_transform",
            json!({
                "transform": { "unknown_argument": true },
            }),
            "set_reconstruction_transform.transform has no argument \"unknown_argument\" — \
             it takes rotation_wxyz, translation, scale.",
        ),
        (
            "move_camera_image",
            json!({
                "world_from_camera": { "unknown_argument": true },
            }),
            "move_camera_image.world_from_camera has no argument \"unknown_argument\" — \
             it takes quaternion_wxyz, translation.",
        ),
    ] {
        assert_eq!(
            tools::parse(tool, arguments.as_object())
                .expect_err("rejected")
                .0,
            expected
        );
    }

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
            // Off a camera, a bare `model` could be any model; the wire says
            // which (specs/GLOSSARY.md, `camera_model`).
            assert!(
                property != "model",
                "{}: a camera model argument is camera_model",
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
    assert_eq!(
        keys,
        ["crop_px", "hud", "max_dimension", "panel_name", "widgets"]
    );
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
        Outcome::Deferred(super::super::Deferred::Screenshot { caption, .. }) => {
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
            "get_viewer_3d_display",
            "get_history",
            "get_bench",
            "get_bench_track",
            "get_background_task",
            "screenshot",
            "get_widgets",
        ]
    );
    // Sixteen JSON reads, the one read that hands back a picture, sixty-six
    // writes, four input tools, and the one that writes a file.
    assert_eq!(catalog.len(), 88, "the catalog has grown or shrunk");
    assert_eq!(
        catalog
            .iter()
            .filter(|spec| spec.kind == ToolKind::Write)
            .count(),
        66
    );
    // One tool can overwrite a file by name, and four can press what a person
    // can press, File ▸ Save included; those five are the ones annotated
    // destructive.
    let saves: Vec<&str> = catalog
        .iter()
        .filter(|spec| spec.kind == ToolKind::Save)
        .map(|spec| spec.name)
        .collect();
    assert_eq!(saves, ["save_reconstruction"]);
    let inputs: Vec<&str> = catalog
        .iter()
        .filter(|spec| spec.kind == ToolKind::Input)
        .map(|spec| spec.name)
        .collect();
    assert_eq!(inputs, ["click", "hover", "press_key", "type_text"]);
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
        format!(
            "{} write, four send input, and one writes a file",
            spelled(writes)
        ),
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
    // `create_track_at_pixel` and `find_nearby_tracks` are of the family
    // without the word: what they produce is points, and the bench is where
    // the tracks wait on them.
    let bench = catalog
        .iter()
        .filter(|spec| {
            spec.name.contains("bench")
                || spec.name.contains("index_files")
                || spec.name == "create_track_at_pixel"
                || spec.name == "find_nearby_tracks"
        })
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

/// Every description text in `value`, at any depth of a schema.
fn schema_descriptions(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, inner) in map {
                match (key.as_str(), inner) {
                    ("description", Value::String(text)) => out.push(text.clone()),
                    _ => schema_descriptions(inner, out),
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|item| schema_descriptions(item, out)),
        _ => {}
    }
}

/// `delete_camera_image` renumbers the bench's observations with the image
/// table, so no tool tells an agent that an observation's position is never
/// renumbered, and the delete's own description says what it does to the
/// bench. Bug bash of PR #672.
#[test]
fn no_description_promises_observations_are_never_renumbered() {
    for spec in tools::catalog() {
        let mut texts = vec![spec.description.to_string()];
        schema_descriptions(&spec.schema, &mut texts);
        for text in texts {
            assert!(
                !text.contains("never renumbered") && !text.contains("life of the track"),
                "{} says observations are never renumbered: {text}",
                spec.name
            );
        }
    }
    let delete = tools::catalog()
        .iter()
        .find(|spec| spec.name == "delete_camera_image")
        .expect("delete_camera_image is in the catalog");
    assert!(
        delete.description.contains("bench"),
        "delete_camera_image says nothing about the bench: {}",
        delete.description
    );
}
