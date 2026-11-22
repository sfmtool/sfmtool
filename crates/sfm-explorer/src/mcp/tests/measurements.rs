// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurement tools on the fixture in `crate::measurements::fixture`:
//! `get_scene`'s counts, `list_measurements`, `get_measurement` and
//! `select_measurement`.

use super::*;
use crate::measurements::{EndpointSide, MeasurementRef};

fn measured() -> (AppState, crate::scene::ReconId, Viewer3D) {
    let (state, id) = crate::measurements::fixture::measured_state(Some("m"));
    (state, id, Viewer3D::new())
}

/// `get_scene` counts the measurements of each kind on every reconstruction.
#[test]
fn get_scene_counts_each_kind_of_measurement() {
    let (mut state, _, mut viewer) = measured();
    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    assert_eq!(
        scene["scene"][0]["measurement_counts"],
        json!({ "distance": 5 })
    );
}

/// `list_measurements` gives each measurement the row the Scene tree draws
/// and the result's numbers.
#[test]
fn list_measurements_gives_each_row_and_reading() {
    let (mut state, _, mut viewer) = measured();
    let listed = call(&mut state, &mut viewer, "list_measurements", json!({}));
    let rows = listed["measurements"].as_array().expect("an array");
    assert_eq!(rows.len(), 5);
    assert_eq!(listed["world_space_unit"], "m");
    let first = &rows[0];
    assert_eq!(first["measurement_kind"], "distance");
    assert_eq!(first["measurement_index"], 0);
    assert_eq!(first["label"], "ear to ear");
    assert_eq!(first["unit"], "cm");
    assert_eq!(
        first["row_text"],
        "Distance #0  ear to ear  153 cm  \u{2192} 150.0 cm  \u{2212}2.0%"
    );
    assert_eq!(first["result"]["state"], "with_unit");
    let reconstructed = first["result"]["reconstructed_distance"].as_f64().unwrap();
    assert!((reconstructed - 150.0).abs() < 1e-3, "{reconstructed}");
    assert_eq!(
        rows[2]["result"],
        json!({ "state": "refused", "endpoint": "b" })
    );
    assert_eq!(rows[2]["endpoints"][1]["resolved"], false);
    assert_eq!(
        rows[3]["result"],
        json!({ "state": "at_infinity", "endpoint": "b" })
    );
}

/// `get_measurement` gives each endpoint's resolution with its candidates, a
/// refusal's sentence, and the median line.
#[test]
fn get_measurement_gives_the_resolutions_and_the_median_line() {
    let (mut state, _, mut viewer) = measured();
    let one = call(
        &mut state,
        &mut viewer,
        "get_measurement",
        json!({ "measurement_index": 0 }),
    );
    let a = &one["endpoints"][0];
    assert_eq!(a["endpoint"], "a");
    assert_eq!(a["resolution"]["chosen"], 0);
    let candidate = &a["resolution"]["candidates"][0];
    assert_eq!(candidate["point"], 0);
    assert_eq!(candidate["standing"], "alone");
    assert_eq!(a["refusal"], Value::Null);
    assert_eq!(one["median"]["count"], 3);
    assert_eq!(
        one["median"]["text"],
        "Median over 3: 1 su = 102 cm; this one +0.0%"
    );

    let refused = call(
        &mut state,
        &mut viewer,
        "get_measurement",
        json!({ "measurement_kind": "distance", "measurement_index": 2 }),
    );
    let b = &refused["endpoints"][1];
    assert_eq!(b["resolution"], Value::Null);
    assert_eq!(b["refusal"]["short"], "no patch holds the pixel");
    assert!(b["refusal"]["sentence"]
        .as_str()
        .unwrap()
        .contains("Create Track Here"));
}

/// An index past the rows is refused with the count.
#[test]
fn a_measurement_past_the_rows_is_refused_with_the_count() {
    let (mut state, _, mut viewer) = measured();
    for tool in ["get_measurement", "select_measurement"] {
        let error = refused_call(
            &mut state,
            &mut viewer,
            tool,
            json!({ "measurement_index": 5 }),
        );
        assert!(
            error.0.contains("stores 5 distance measurements"),
            "{tool}: {error}"
        );
    }
    let error = refused_call(
        &mut state,
        &mut viewer,
        "select_measurement",
        json!({ "measurement_index": 0, "endpoint": "c" }),
    );
    assert!(error.0.contains("expected a or b"), "{error}");
}

/// `select_measurement` selects as a row click does, and with an endpoint as
/// a block click does; the selection block reports both.
#[test]
fn select_measurement_selects_the_measurement_and_its_endpoint() {
    let (mut state, id, mut viewer) = measured();
    let reply = call(
        &mut state,
        &mut viewer,
        "select_measurement",
        json!({ "measurement_index": 4 }),
    );
    assert_eq!(
        reply["selection"]["measurement"],
        json!({
            "reconstruction_label": "measured",
            "measurement_kind": "distance",
            "measurement_index": 4,
        })
    );
    assert_eq!(reply["selection"]["measurement_endpoint"], Value::Null);
    assert_eq!(reply["selection"]["camera_image"], Value::Null);

    let reply = call(
        &mut state,
        &mut viewer,
        "select_measurement",
        json!({ "measurement_index": 4, "endpoint": "a" }),
    );
    assert_eq!(reply["selection"]["measurement_endpoint"], "a");
    assert_eq!(reply["selection"]["camera_image"]["index"], 1);
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 4))
    );
    assert_eq!(state.selected_endpoint, Some(EndpointSide::A));
}

// ── The bench ────────────────────────────────────────────────────────────

/// `put_measurement_on_bench` puts the measurement on and focuses it, a second
/// call focuses the same item with no version, and `get_bench` lists it.
#[test]
fn put_measurement_on_bench_puts_on_once_and_get_bench_lists_it() {
    let (mut state, _, mut viewer) = measured();
    let put = call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 3 }),
    );
    assert_eq!(put["item"], "distance #3");
    assert_eq!(put["changed"], true);
    let again = call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 3 }),
    );
    assert_eq!(again["item"], "distance #3");
    assert_eq!(again["changed"], false);
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "measured" }),
    );
    assert_eq!(bench["focused_item"], "distance #3");
    let items = bench["items"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["kind"], "measurement");
    assert_eq!(items[0]["origin"], json!({ "measurement_index": 3 }));
    assert_eq!(items[0]["counts"]["endpoints"], 2);
    // A track tool on it says what it is.
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "measured" }),
    );
    assert!(
        error
            .0
            .contains("distance #3 is a measurement, not a track"),
        "{error}"
    );
}

/// `get_measurement` naming an item reads it live, with its origin.
#[test]
fn get_measurement_reads_an_item_live() {
    let (mut state, _, mut viewer) = measured();
    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 0 }),
    );
    let item = call(
        &mut state,
        &mut viewer,
        "get_measurement",
        json!({ "reconstruction_label": "measured", "item": "distance #0" }),
    );
    assert_eq!(item["item"], "distance #0");
    assert_eq!(item["focused"], true);
    assert_eq!(item["origin"], json!({ "measurement_index": 0 }));
    assert_eq!(item["label"], "ear to ear");
    assert_eq!(item["result"]["state"], "with_unit");
    assert_eq!(
        item["endpoints"][0]["resolution"]["candidates"][0]["point"],
        0
    );
    assert_eq!(
        item["median"]["text"],
        "Median over 3: 1 su = 102 cm; this one +0.0%"
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_measurement",
        json!({ "reconstruction_label": "measured" }),
    );
    assert!(error.0.contains("measurement_index"), "{error}");
}

/// `delete_measurement` drops one row as an edit, and
/// `commit_bench_measurement` puts an item whose row was deleted back as a
/// new row; an unchanged commit answers changed: false.
#[test]
fn delete_and_commit_a_measurement_over_the_wire() {
    let (mut state, id, mut viewer) = measured();
    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 4 }),
    );
    let unchanged = call(
        &mut state,
        &mut viewer,
        "commit_bench_measurement",
        json!({ "reconstruction_label": "measured" }),
    );
    assert_eq!(unchanged["changed"], false);
    assert_eq!(unchanged["measurement"]["measurement_index"], 4);
    let deleted = call(
        &mut state,
        &mut viewer,
        "delete_measurement",
        json!({ "reconstruction_label": "measured", "measurement_index": 4 }),
    );
    assert_eq!(deleted["changed"], true);
    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    assert_eq!(
        scene["scene"][0]["measurement_counts"],
        json!({ "distance": 4 })
    );
    let committed = call(
        &mut state,
        &mut viewer,
        "commit_bench_measurement",
        json!({ "reconstruction_label": "measured", "item": "distance #4" }),
    );
    assert_eq!(committed["changed"], true);
    assert_eq!(
        committed["measurement"],
        json!({
            "measurement_kind": "distance",
            "measurement_index": 4,
            "replaced": false,
        })
    );
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 4))
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "delete_measurement",
        json!({ "reconstruction_label": "measured", "measurement_index": 7 }),
    );
    assert!(
        error.0.contains("stores 5 distance measurements"),
        "{error}"
    );
}
