// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `get_track_table_query` / `set_track_table_query`: the Track Table panel's
//! query as one document, merged field by field, refused whole, logged in the
//! panel's words, and taken by the panel on its next frame.

use super::edit::{editable, rows};
use super::*;
use crate::track_table_panel::TrackTablePanel;

/// The query document a reply carries.
fn query(reply: &Value) -> Value {
    reply["track_table_query"].clone()
}

/// The document `get_track_table_query` answers with.
fn get(state: &mut AppState, viewer: &mut Viewer3D) -> Value {
    ok(state, viewer, Command::GetTrackTableQuery)
}

/// One `set_track_table_query` call, parsed as the transport would.
#[track_caller]
fn set(state: &mut AppState, viewer: &mut Viewer3D, arguments: Value) -> Value {
    call(state, viewer, "set_track_table_query", arguments)
}

/// One frame of the panel, drawn, followed by the frame's after-dock step.
fn frame(ctx: &egui::Context, panel: &mut TrackTablePanel, state: &mut AppState) {
    crate::test_support::run_frame_headless(ctx, egui::RawInput::default(), |ui| {
        panel.show(ui, state)
    });
    let open = state.is_panel_open(Tab::TrackTable);
    panel.after_dock(state, open);
}

#[test]
fn get_returns_the_default_query_and_the_selected_reconstruction() {
    let (mut state, mut viewer) = two_reconstructions();
    let reply = get(&mut state, &mut viewer);
    assert_eq!(
        reply,
        json!({
            "track_table_query": {
                "bounds": [],
                "order_by": null,
                "columns": [
                    "crop", "patch", "track_length", "rms_error_px", "centroid_distance",
                    "max_ray_angle_deg",
                ],
                "seen_in_selected_camera_image": false,
                "filter_3d": "off",
            },
            "reconstruction_label": "alpha",
        })
    );
    let entries: Vec<_> = state.action_log.entries().collect();
    assert_eq!(
        entries.last().unwrap().kind,
        Kind::Query("get_track_table_query")
    );
}

/// Each call changes the fields it names and leaves the rest, and the reply is
/// what a get returns.
#[test]
fn set_merges_the_fields_it_names() {
    let (mut state, mut viewer) = two_reconstructions();
    let first = set(
        &mut state,
        &mut viewer,
        json!({
            "bounds": [
                { "column": "track_length", "min": 3 },
                { "column": "centroid_distance", "min": "inf" },
            ],
            "order_by": { "column": "rms_error_px", "descending": true },
        }),
    );
    assert_eq!(
        query(&first)["bounds"],
        json!([
            { "column": "track_length", "min": 3.0 },
            { "column": "centroid_distance", "min": "inf" },
        ])
    );
    assert_eq!(
        query(&first)["order_by"],
        json!({ "column": "rms_error_px", "descending": true })
    );
    let second = set(
        &mut state,
        &mut viewer,
        json!({ "filter_3d": "hide", "seen_in_selected_camera_image": true }),
    );
    assert_eq!(query(&second)["bounds"], query(&first)["bounds"]);
    assert_eq!(query(&second)["order_by"], query(&first)["order_by"]);
    assert_eq!(query(&second)["filter_3d"], json!("hide"));
    assert_eq!(query(&second)["seen_in_selected_camera_image"], json!(true));
    assert_eq!(second, get(&mut state, &mut viewer));

    // bounds replaces every box, a bound with neither end clears its column,
    // and a null order_by puts the rows back in point order.
    let third = set(
        &mut state,
        &mut viewer,
        json!({ "bounds": [{ "column": "rms_error_px", "max": 1.5 }], "order_by": null }),
    );
    assert_eq!(
        query(&third)["bounds"],
        json!([{ "column": "rms_error_px", "max": 1.5 }])
    );
    assert_eq!(query(&third)["order_by"], Value::Null);
    let cleared = set(&mut state, &mut viewer, json!({ "bounds": [] }));
    assert_eq!(query(&cleared)["bounds"], json!([]));
}

#[test]
fn filter_3d_takes_off_dim_and_hide() {
    let (mut state, mut viewer) = two_reconstructions();
    for choice in ["dim", "hide", "off", "hide"] {
        let reply = set(&mut state, &mut viewer, json!({ "filter_3d": choice }));
        assert_eq!(query(&reply)["filter_3d"], json!(choice));
    }
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_track_table_query",
        json!({ "filter_3d": "faint" }),
    );
    assert_eq!(
        error.0,
        "set_track_table_query wants filter_3d to be one of \"off\", \"dim\", \"hide\" — got \
         \"faint\"."
    );
}

/// `columns` names the columns shown, the picture columns among them; a
/// column holding a bound or the sort cannot be left out, and without
/// `columns` a bound on a hidden column shows it.
#[test]
fn columns_names_the_columns_shown() {
    let (mut state, mut viewer) = two_reconstructions();
    let reply = set(
        &mut state,
        &mut viewer,
        json!({ "columns": ["max_error_px", "patch", "track_length"] }),
    );
    assert_eq!(
        query(&reply)["columns"],
        json!(["patch", "track_length", "max_error_px"])
    );
    let shown = set(
        &mut state,
        &mut viewer,
        json!({ "bounds": [{ "column": "condition_number", "max": 100 }] }),
    );
    assert_eq!(
        query(&shown)["columns"],
        json!(["patch", "track_length", "max_error_px", "condition_number"])
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_track_table_query",
        json!({ "columns": ["track_length"], "filter_3d": "dim" }),
    );
    assert_eq!(
        error.0,
        "set_track_table_query cannot leave the column condition_number out of columns: it \
         holds a bound or the sort. List it in columns, or clear its bound and the sort, so \
         that a hidden column never filters or orders the rows."
    );
    // Refused whole: the filter in the same call was not applied either.
    assert_eq!(
        query(&get(&mut state, &mut viewer))["filter_3d"],
        json!("off")
    );
}

/// A name that is not a column, a picture column as a bound or a sort, two
/// bounds on one column, a min above its max and an empty call are refused
/// with find_points' words where it has them.
#[test]
fn set_refuses_a_column_that_does_not_exist() {
    let (mut state, mut viewer) = two_reconstructions();
    let mut refusal = |arguments: Value| {
        refused_call(&mut state, &mut viewer, "set_track_table_query", arguments).0
    };
    let columns = "track_length, repeat_observations, back_facing_observations";
    let error = refusal(json!({ "bounds": [{ "column": "error", "min": 1 }] }));
    assert!(
        error.starts_with(&format!(
            "set_track_table_query does not know the column \"error\" — the columns are \
             {columns}"
        )),
        "{error}"
    );
    let error = refusal(json!({ "order_by": { "column": "error" } }));
    assert!(
        error.starts_with("set_track_table_query does not know the column \"error\""),
        "{error}"
    );
    let error = refusal(json!({ "columns": ["track_length", "error"] }));
    assert!(
        error.starts_with(
            "set_track_table_query does not know the column \"error\" — the Track Table's \
             columns are crop, patch, track_length, "
        ),
        "{error}"
    );
    let error = refusal(json!({ "order_by": { "column": "crop" } }));
    assert!(
        error.starts_with(&format!(
            "set_track_table_query does not bound or sort by the column \"crop\": a picture \
             column holds no numbers, so it can only be shown, in columns. The columns it \
             bounds and sorts by are {columns}"
        )),
        "{error}"
    );
    let error = refusal(json!({ "bounds": [{ "column": "patch", "min": 1 }] }));
    assert!(
        error.starts_with("set_track_table_query does not bound or sort by the column \"patch\""),
        "{error}"
    );
    let error = refusal(json!({
        "bounds": [
            { "column": "track_length", "min": 2 },
            { "column": "track_length", "max": 4 },
        ],
    }));
    assert_eq!(
        error,
        "set_track_table_query names the column track_length in two bounds — the panel holds \
         one minimum and one maximum per column."
    );
    let error = refusal(json!({ "bounds": [{ "column": "track_length", "min": 5, "max": 3 }] }));
    assert_eq!(
        error,
        "set_track_table_query wants each bound's min to be at most its max — got min 5 and max \
         3 for track_length."
    );
    let error = refusal(json!({ "bounds": [{ "column": "track_length", "min": "infinity" }] }));
    assert_eq!(
        error,
        "set_track_table_query.bounds wants min to be a number or the string \"inf\" — got the \
         string \"infinity\"."
    );
    assert_eq!(
        refusal(json!({})),
        "set_track_table_query was given nothing to change."
    );
    // Nothing a refusal named was kept.
    assert_eq!(query(&get(&mut state, &mut viewer))["bounds"], json!([]),);
}

/// Every control the call changed is one Action Log row, as the agent, in the
/// panel's words; a field that did not change writes nothing.
#[test]
fn set_logs_each_changed_control_in_the_panel_s_words() {
    let (mut state, mut viewer) = two_reconstructions();
    let before = rows(&state).len();
    set(
        &mut state,
        &mut viewer,
        json!({
            "bounds": [
                { "column": "rms_error_px", "max": 1.5 },
                { "column": "centroid_distance", "min": "inf" },
            ],
            "order_by": { "column": "rms_error_px", "descending": true },
            "columns": ["patch", "track_length", "rms_error_px", "centroid_distance"],
            "seen_in_selected_camera_image": true,
            "filter_3d": "hide",
        }),
    );
    let written: Vec<(Actor, bool, String)> = rows(&state).split_off(before);
    let texts: Vec<&str> = written.iter().map(|(_, _, text)| text.as_str()).collect();
    assert_eq!(
        texts,
        [
            "Track Table: maximum Error (px) 1.5",
            "Track Table: minimum Distance inf",
            "Track Table: sorted by Error (px), decreasing",
            "Track Table: Columns Patch, Track, Error (px), Distance",
            "Track Table: Seen in selected image on",
            "Track Table: Filter the 3D view on, Hide",
        ]
    );
    assert!(written
        .iter()
        .all(|(actor, failed, _)| *actor == Actor::Mcp && !failed));
    let display = state
        .action_log
        .entries()
        .filter(|entry| entry.kind == Kind::Display)
        .count();
    assert_eq!(display, 6);

    // The same call again changes nothing and writes nothing.
    let again = rows(&state).len();
    set(&mut state, &mut viewer, json!({ "filter_3d": "hide" }));
    assert_eq!(rows(&state).len(), again);
    set(
        &mut state,
        &mut viewer,
        json!({ "bounds": [], "order_by": null }),
    );
    let texts: Vec<String> = rows(&state)
        .split_off(again)
        .into_iter()
        .map(|r| r.2)
        .collect();
    assert_eq!(
        texts,
        [
            "Track Table: maximum Error (px) cleared",
            "Track Table: minimum Distance cleared",
            "Track Table: sorted by point",
        ]
    );
}

/// The panel does not have to be open: a closed one is opened, and takes the
/// query on its next frame; an open one stays where it is.
#[test]
fn set_opens_a_closed_panel_which_takes_the_query() {
    let (mut state, mut viewer) = two_reconstructions();
    state.hide_panel(Tab::TrackTable);
    assert!(!state.is_panel_open(Tab::TrackTable));
    set(&mut state, &mut viewer, json!({ "filter_3d": "dim" }));
    assert!(state.is_panel_open(Tab::TrackTable));
    assert!(rows(&state)
        .iter()
        .any(|(_, _, text)| text.contains("Track Table panel")));

    // The panel takes it on its next frame, and keeps it on the frames after.
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    frame(&ctx, &mut panel, &mut state);
    assert!(state.filter_3d.is_some(), "the panel took the filter");
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(
        query(&get(&mut state, &mut viewer))["filter_3d"],
        json!("dim")
    );

    // A call while the panel is open behind another tab leaves the layout
    // alone.
    state.show_panel(Tab::Viewer3D);
    let layout = state.window_layout();
    set(&mut state, &mut viewer, json!({ "filter_3d": "off" }));
    assert_eq!(state.window_layout(), layout);
    frame(&ctx, &mut panel, &mut state);
    assert!(state.filter_3d.is_none(), "the panel took the cleared box");
}

/// A query set over the wire and a `find_points` call with the same bounds and
/// sort agree: the panel's rows, which *Filter the 3D view* keeps, are the
/// points the call matches.
#[test]
fn the_panel_and_find_points_agree_on_a_query_set_over_the_wire() {
    let (mut state, mut viewer) = editable();
    let id = state.scene[0].id;
    // A maximum error at the median, so the bound drops about half the points.
    let mut errors: Vec<f64> = state.scene[0]
        .recon()
        .point_set
        .points
        .iter()
        .map(|point| f64::from(point.rms_reprojection_error))
        .collect();
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    let bounds = json!([
        { "column": "track_length", "min": 2 },
        { "column": "rms_error_px", "max": median },
    ]);
    set(
        &mut state,
        &mut viewer,
        json!({ "bounds": bounds, "filter_3d": "dim" }),
    );
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    frame(&ctx, &mut panel, &mut state);
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let kept = state
        .filter_3d
        .as_ref()
        .and_then(|filter| filter.kept.clone())
        .expect("the panel listed its rows")
        .1;

    let command = tools::parse(
        "find_points",
        Some(
            json!({ "bounds": bounds, "limit": 500 })
                .as_object()
                .unwrap(),
        ),
    )
    .unwrap();
    let reply = ok(&mut state, &mut viewer, command);
    let matched: Vec<u32> = reply["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["index"].as_u64().unwrap() as u32)
        .collect();
    assert!(!matched.is_empty());
    let live: Vec<u32> = state.node(id).unwrap().edited().live_indexes().collect();
    assert!(matched.len() < live.len(), "the bounds drop some points");
    for point in live {
        assert_eq!(
            kept.contains(point),
            matched.contains(&point),
            "point {point}"
        );
    }
}
