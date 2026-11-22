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
