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
    // The reply's sentence is about the item this call focused, not the
    // version an earlier step wrote.
    assert_eq!(
        again["report"],
        "Put distance #3 on the bench: no effect, it is on the bench already as distance #3, \
         now the focused item"
    );
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

/// A track tool that names a measurement item by label says it is a
/// measurement, rather than that nothing on the bench has that name.
#[test]
fn a_track_tool_naming_a_measurement_says_what_it_is() {
    let (mut state, _, mut viewer) = measured();
    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 1 }),
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "measured", "track": "distance #1",
                "min_zncc": 0.5 }),
    );
    assert!(
        error
            .0
            .contains("distance #1 is a measurement, not a track"),
        "{error}"
    );
    assert!(!error.0.contains("Nothing on the bench"), "{error}");
}

/// `set_bench_measurement_label` sets the focused item's label as one bench
/// version, says so in its reply, and answers changed: false for the label
/// it holds.
#[test]
fn set_bench_measurement_label_sets_the_focused_item_s_label() {
    let (mut state, id, mut viewer) = measured();
    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 2 }),
    );
    let set = call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_label",
        json!({ "reconstruction_label": "measured", "label": "paw width" }),
    );
    assert_eq!(set["changed"], true);
    assert_eq!(set["item"], "distance #2");
    assert_eq!(set["measurement_label"], "paw width");
    assert_eq!(
        set["label"],
        "Set the label of distance #2 to \"paw width\""
    );
    assert_eq!(set["dirty"], false);
    assert_eq!(
        state.bench_measurement(id, "distance #2").unwrap().label,
        "paw width"
    );
    let again = call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_label",
        json!({ "reconstruction_label": "measured", "item": "distance #2",
                "label": "paw width" }),
    );
    assert_eq!(again["changed"], false);
    assert!(
        again["report"]
            .as_str()
            .unwrap()
            .ends_with("no effect, it has that label already"),
        "{again}"
    );
}

/// `set_bench_measurement_distance` reads text and a JSON number, a bare
/// number in the file's unit, and refuses a distance that is not positive in
/// core's sentence.
#[test]
fn set_bench_measurement_distance_reads_text_and_numbers() {
    let (mut state, id, mut viewer) = measured();
    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 2 }),
    );
    let set = call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_distance",
        json!({ "reconstruction_label": "measured", "distance": "12in" }),
    );
    assert_eq!(set["changed"], true);
    assert_eq!(set["item"], "distance #2");
    assert_eq!(set["stated_distance"], 12.0);
    assert_eq!(set["unit"], "in");
    assert_eq!(
        set["label"],
        "Set the stated distance of distance #2 to 12 in"
    );
    let bare = call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_distance",
        json!({ "reconstruction_label": "measured", "distance": 0.5 }),
    );
    assert_eq!(bare["stated_distance"], 0.5);
    assert_eq!(bare["unit"], "m");
    assert_eq!(
        state
            .bench_measurement(id, "distance #2")
            .unwrap()
            .as_distance()
            .unwrap()
            .distance,
        Some((0.5, sfmtool_core::LengthUnit::Metre))
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_distance",
        json!({ "reconstruction_label": "measured", "distance": "-3cm" }),
    );
    assert!(
        error
            .0
            .contains("the stated distance is -3, which must be finite and strictly positive"),
        "{error}"
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_distance",
        json!({ "reconstruction_label": "measured", "distance": true }),
    );
    assert!(error.0.contains("a string or a number"), "{error}");
}

/// `get_measurement_endpoint_views` lists the popup's offers for an
/// endpoint of the focused item, and `set_bench_measurement_endpoint` moved
/// to one of them is the switch: one bench version, the endpoint on the
/// same point there.
#[test]
fn an_endpoint_switched_over_the_wire_lands_on_the_same_point() {
    let (mut state, id, mut viewer) = measured();
    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 1 }),
    );
    let offered = call(
        &mut state,
        &mut viewer,
        "get_measurement_endpoint_views",
        json!({ "reconstruction_label": "measured", "endpoint": "a" }),
    );
    assert_eq!(offered["item"], "distance #1");
    assert_eq!(offered["endpoint"], "a");
    assert_eq!(offered["camera_image"], 0);
    let views = offered["views"].as_array().expect("an array");
    assert_eq!(views.len(), 1, "{offered}");
    let view = &views[0];
    assert_eq!(view["point"], 1);
    assert_eq!(view["camera_image"], 1);
    assert_eq!(view["camera_image_name"], "image_001.jpg");
    assert_eq!(view["resolves_to"], 1);
    assert_eq!(view["resolves_elsewhere"], false);
    assert_eq!(view["localization"], Value::Null);
    assert!(view["viewing_angle_deg"].as_f64().is_some(), "{view}");

    let set = call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_endpoint",
        json!({
            "reconstruction_label": "measured",
            "endpoint": "a",
            "camera_image": "image_001.jpg",
            "pixel": view["pixel"],
        }),
    );
    assert_eq!(set["changed"], true, "{set}");
    assert_eq!(set["item"], "distance #1");
    assert_eq!(set["camera_image"], 1);
    assert!(
        set["label"]
            .as_str()
            .unwrap()
            .starts_with("Moved endpoint A of distance #1 to image_001.jpg ("),
        "{set}"
    );
    let item = call(
        &mut state,
        &mut viewer,
        "get_measurement",
        json!({ "reconstruction_label": "measured", "item": "distance #1" }),
    );
    let a = &item["endpoints"][0];
    assert_eq!(a["camera_image"], 1);
    let resolution = &a["resolution"];
    let chosen = resolution["chosen"].as_u64().unwrap() as usize;
    assert_eq!(resolution["candidates"][chosen]["point"], 1);
    assert_eq!(
        crate::measurements::distance_rows(state.node(id).unwrap().edited().measurements())[1]
            .images,
        [0, 1],
        "the stored row changed"
    );
}

/// The views are refused in the button's words, and the step refuses a
/// pixel outside its image in the commit's.
#[test]
fn the_endpoint_tools_refuse_in_the_panel_s_words() {
    let (mut state, _, mut viewer) = measured();
    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 2 }),
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_measurement_endpoint_views",
        json!({ "reconstruction_label": "measured", "endpoint": "b" }),
    );
    assert!(
        error
            .0
            .starts_with("Endpoint B has no position, so no other image can be offered:"),
        "{error}"
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_endpoint",
        json!({
            "reconstruction_label": "measured",
            "endpoint": "b",
            "camera_image": 0,
            "pixel": [3.0, 480.0],
        }),
    );
    assert_eq!(
        error.0,
        "Cannot move endpoint B of distance #2: endpoint B's pixel (3, 480) is outside image \
         0, which is [0, 640) x [0, 480)."
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "get_measurement_endpoint_views",
        json!({ "reconstruction_label": "measured", "endpoint": "c" }),
    );
    assert!(error.0.contains("expected a or b"), "{error}");
}

/// `get_point_reference_pixel` gives a point's reference pixel, named by
/// index or by id, which `set_bench_measurement_endpoint` then places an
/// endpoint at as the 3D view's menu does for a right-click off the point's
/// drawn patch; a point with no reference pixel is refused in the menu
/// entry's sentence.
#[test]
fn get_point_reference_pixel_gives_the_pixel_the_point_menu_takes() {
    use crate::measurements::{AT_INFINITY_FOR_PIXEL, NO_REFERENCE_FOR_PIXEL};
    let (mut state, id) = crate::measurements::fixture::referenced_state(Some("m"));
    let mut viewer = Viewer3D::new();
    let reference = call(
        &mut state,
        &mut viewer,
        "get_point_reference_pixel",
        json!({ "reconstruction_label": "measured", "point": 0 }),
    );
    let keypoint = state
        .node(id)
        .unwrap()
        .edited()
        .point(0)
        .unwrap()
        .keypoint_xy(1)
        .unwrap();
    let point_id = crate::scene::point_id(state.node(id).unwrap(), 0);
    assert_eq!(
        reference,
        json!({
            "reconstruction_label": "measured",
            "point": 0,
            "point_id": point_id,
            "observation": 1,
            "camera_image": 1,
            "camera_image_name": "image_001.jpg",
            "pixel": [f64::from(keypoint[0]), f64::from(keypoint[1])],
        })
    );
    let by_id = call(
        &mut state,
        &mut viewer,
        "get_point_reference_pixel",
        json!({ "reconstruction_label": "measured", "point": point_id }),
    );
    assert_eq!(by_id, reference);

    call(
        &mut state,
        &mut viewer,
        "put_measurement_on_bench",
        json!({ "reconstruction_label": "measured", "measurement_index": 0 }),
    );
    let set = call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_endpoint",
        json!({
            "reconstruction_label": "measured",
            "endpoint": "b",
            "camera_image": reference["camera_image"],
            "pixel": reference["pixel"],
        }),
    );
    assert_eq!(set["changed"], true, "{set}");
    let item = call(
        &mut state,
        &mut viewer,
        "get_measurement",
        json!({ "reconstruction_label": "measured", "item": "distance #0" }),
    );
    let resolution = &item["endpoints"][1]["resolution"];
    let chosen = resolution["chosen"].as_u64().unwrap() as usize;
    assert_eq!(resolution["candidates"][chosen]["point"], 0);

    for (point, sentence) in [(1, NO_REFERENCE_FOR_PIXEL), (3, AT_INFINITY_FOR_PIXEL)] {
        let error = refused_call(
            &mut state,
            &mut viewer,
            "get_point_reference_pixel",
            json!({ "reconstruction_label": "measured", "point": point }),
        );
        assert_eq!(error.0, sentence);
    }
}

/// `create_bench_measurement` puts a one-endpoint item on the bench and
/// focuses it, as *New Measurement from Here* does, without raising a panel;
/// the item is then finished and committed with the other tools, appending a
/// row. A pixel outside the image is refused in the commit's words.
#[test]
fn create_bench_measurement_starts_an_item_the_other_tools_finish() {
    let (mut state, id) = crate::measurements::fixture::measured_state(Some("m"));
    let mut viewer = Viewer3D::new();
    state.hide_panel(crate::dock::Tab::MeasurementView);
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_measurement",
        json!({
            "reconstruction_label": "measured",
            "camera_image": 1,
            "pixel": [141.6, 198.4],
        }),
    );
    let label = "distance from image_001@142,198";
    assert_eq!(made["item"], label, "{made}");
    assert_eq!(made["changed"], true);
    assert_eq!(made["camera_image"], 1);
    assert_eq!(made["pixel"], json!([141.6, 198.4]));
    assert_eq!(state.focused_measurement_label(id), Some(label));
    assert!(!state.is_panel_open(crate::dock::Tab::MeasurementView));

    let item = call(
        &mut state,
        &mut viewer,
        "get_measurement",
        json!({ "reconstruction_label": "measured", "item": label }),
    );
    assert_eq!(item["origin"], Value::Null, "{item}");
    let refused = refused_call(
        &mut state,
        &mut viewer,
        "commit_bench_measurement",
        json!({ "reconstruction_label": "measured" }),
    );
    assert_eq!(
        refused.0,
        format!(
            "Cannot commit {label}: endpoint B is not placed, and a distance needs both \
             endpoints."
        )
    );
    call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_endpoint",
        json!({
            "reconstruction_label": "measured",
            "endpoint": "b",
            "camera_image": 0,
            "pixel": [300.0, 200.5],
        }),
    );
    call(
        &mut state,
        &mut viewer,
        "set_bench_measurement_distance",
        json!({ "reconstruction_label": "measured", "distance": "42 cm" }),
    );
    let stored = state
        .node(id)
        .unwrap()
        .edited()
        .measurements()
        .distance_measurement_count();
    let committed = call(
        &mut state,
        &mut viewer,
        "commit_bench_measurement",
        json!({ "reconstruction_label": "measured" }),
    );
    assert_eq!(committed["changed"], true, "{committed}");
    assert_eq!(
        state
            .node(id)
            .unwrap()
            .edited()
            .measurements()
            .distance_measurement_count(),
        stored + 1
    );

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_measurement",
        json!({
            "reconstruction_label": "measured",
            "camera_image": 0,
            "pixel": [640.0, 3.0],
        }),
    );
    assert_eq!(
        refused.0,
        "Cannot start a measurement there: endpoint A's pixel (640, 3) is outside image 0, \
         which is [0, 640) x [0, 480)."
    );
}

// ── The measurement scale ───────────────────────────────────────────────

/// `get_measurement_scale` gives the dialog's numbers: the median factor in
/// the unit, its statistics, each usable measurement's factor and deviation,
/// and each excluded one with its reason.
#[test]
fn get_measurement_scale_gives_the_factor_the_statistics_and_the_exclusions() {
    let (mut state, _, mut viewer) = measured();
    let scale = call(
        &mut state,
        &mut viewer,
        "get_measurement_scale",
        json!({ "unit": "mm" }),
    );
    assert_eq!(scale["unit"], "mm");
    assert_eq!(scale["world_space_unit"], "m");
    assert_eq!(scale["refusal"], Value::Null);
    assert_eq!(scale["apply_refusal"], Value::Null);
    let factor = scale["factor"].as_f64().expect("a factor");
    assert!((factor - 1020.0).abs() < 1e-3, "{factor}");
    assert_eq!(scale["scale_text"], "1 scene unit = 1020 mm");
    let correction = scale["correction"].as_f64().expect("a correction");
    assert!((correction - 1.02).abs() < 1e-6, "{correction}");
    assert_eq!(scale["statistics"]["count"], 3);
    assert_eq!(scale["statistics"]["total"], 5);
    assert_eq!(scale["statistics"]["disagrees"], true);
    let indexes: Vec<u64> = scale["factors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["measurement_index"].as_u64().unwrap())
        .collect();
    assert_eq!(indexes, [0, 1, 4]);
    let bad = &scale["factors"][2];
    assert_eq!(bad["label"], "paw width");
    assert_eq!(bad["stated_unit"], "m");
    assert!((bad["stated_in_unit"].as_f64().unwrap() - 1800.0).abs() < 1e-3);
    let deviation = bad["deviation_percent"].as_f64().unwrap();
    assert!(
        (deviation - 100.0 * (1.2 / 1.02 - 1.0)).abs() < 1e-4,
        "{deviation}"
    );
    let excluded = scale["excluded"].as_array().unwrap();
    assert_eq!(excluded.len(), 2);
    assert_eq!(excluded[1]["measurement_index"], 3);
    assert_eq!(
        excluded[1]["reason"],
        "Endpoint B lies on a point at infinity, so no distance can be measured to it."
    );
}

/// Naming no unit reads the one the dialog opens on, the file's own.
#[test]
fn get_measurement_scale_defaults_to_the_dialogs_unit() {
    let (mut state, _, mut viewer) = measured();
    let scale = call(&mut state, &mut viewer, "get_measurement_scale", json!({}));
    assert_eq!(scale["unit"], "m");
    assert_eq!(scale["scale_text"], "1 scene unit = 1.020 m");
}

/// `scale_reconstruction_to_median_measurement` is the dialog's Apply: one
/// edit, the unit declared, and the scale's own reading afterwards agreeing
/// with it.
#[test]
fn scaling_to_the_median_measurement_is_one_edit() {
    let (mut state, id, mut viewer) = measured();
    let reply = call(
        &mut state,
        &mut viewer,
        "scale_reconstruction_to_median_measurement",
        json!({ "reconstruction_label": "measured", "unit": "mm" }),
    );
    assert_eq!(reply["changed"], true);
    assert!(
        reply["report"]
            .as_str()
            .is_some_and(|r| r.starts_with("Scaled measured to mm by the median of 3")),
        "{reply}"
    );
    let node = state.node(id).expect("loaded");
    assert_eq!(
        node.recon().metadata.world_space_unit.as_deref(),
        Some("mm")
    );
    let after = call(
        &mut state,
        &mut viewer,
        "get_measurement_scale",
        json!({ "unit": "mm" }),
    );
    assert!(
        (after["factor"].as_f64().unwrap() - 1.0).abs() < 1e-6,
        "{after}"
    );
}

/// A reconstruction with no measurement is refused in the greyed menu
/// entry's words, and nothing is pushed.
#[test]
fn scaling_a_reconstruction_with_no_measurement_is_refused_in_the_menus_words() {
    let mut state = AppState::new();
    let id = state.append_node(crate::scene::SceneNode::from_path(
        std::path::Path::new("plain.sfmr"),
        sfmtool_core::SfmrReconstruction::demo(10),
    ));
    let mut viewer = Viewer3D::new();
    let versions = state.node(id).unwrap().history.versions().len();
    let why = refused_call(
        &mut state,
        &mut viewer,
        "scale_reconstruction_to_median_measurement",
        json!({ "reconstruction_label": "plain" }),
    );
    let menu = crate::measurement_scale_prompt::refusal(state.node(id).unwrap())
        .expect("the entry is greyed");
    assert_eq!(why.to_string(), menu);
    assert_eq!(state.node(id).unwrap().history.versions().len(), versions);
    let read = call(&mut state, &mut viewer, "get_measurement_scale", json!({}));
    assert_eq!(read["refusal"], json!(menu));
    assert_eq!(read["factor"], Value::Null);
}
