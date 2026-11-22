// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The readings, their text, the median line, the cache and the selection,
//! on the fixture in [`super::fixture`], whose numbers are known by
//! construction.

use std::sync::Arc;

use sfmtool_core::LengthUnit;

use super::fixture::{measured_reconstruction, measured_state, BAD_SCALE, GOOD_SCALE, POINTS, SKY};
use super::*;
use crate::scene::ImageRef;

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} against {b}");
}

/// The rows and readings of the fixture, its file declaring `unit`.
fn read(unit: Option<&str>) -> (Vec<DistanceRow>, MeasurementReadings) {
    let recon = measured_reconstruction(unit);
    let rows = distance_rows(&recon);
    let readings = MeasurementReadings::of(&EditedReconstruction::new(Arc::new(recon)));
    (rows, readings)
}

// ── The result block's numbers ──────────────────────────────────────────

/// With a unit, the reconstructed distance is in the measurement's own unit,
/// the error is reconstructed less stated, and the implied scale is stated
/// over reconstructed: hand-computed from the fixture's 1.5 m between points
/// 0 and 1 and its stated 153 cm.
#[test]
fn a_reading_with_a_unit_gives_the_error_and_the_scale_in_the_measurements_unit() {
    let (rows, readings) = read(Some("m"));
    assert_eq!(rows.len(), 5);
    assert_eq!(readings.distance.len(), 5);
    assert_eq!(rows[0].unit, LengthUnit::Centimetre);
    let DistanceResult::WithUnit {
        reconstructed,
        error,
        error_percent,
        implied_scale,
    } = distance_result(&rows[0], &readings.distance[0], Some("m"))
    else {
        panic!("not a reading with a unit");
    };
    assert_close(reconstructed, 150.0, 1e-3, "reconstructed cm");
    assert_close(error, 150.0 - 153.0, 1e-3, "error cm");
    assert_close(error_percent, -300.0 / 153.0, 1e-3, "error percent");
    assert_close(implied_scale, GOOD_SCALE, 1e-5, "implied scale");

    // Endpoints in two images read the same way.
    let DistanceResult::WithUnit {
        reconstructed,
        implied_scale,
        ..
    } = distance_result(&rows[1], &readings.distance[1], Some("m"))
    else {
        panic!("not a reading with a unit");
    };
    assert_close(reconstructed, 3.0, 1e-5, "reconstructed m");
    assert_close(implied_scale, GOOD_SCALE, 1e-5, "implied scale");
}

/// Without a unit, the reading is the distance in scene units and the length
/// one scene unit stands for in the measurement's unit.
#[test]
fn a_reading_without_a_unit_is_in_scene_units() {
    let (rows, readings) = read(None);
    let DistanceResult::SceneUnits {
        scene_distance,
        unit_per_scene_unit,
    } = distance_result(&rows[0], &readings.distance[0], None)
    else {
        panic!("not a reading in scene units");
    };
    assert_close(scene_distance, 1.5, 1e-5, "scene distance");
    assert_close(unit_per_scene_unit, 153.0 / 1.5, 1e-3, "cm per scene unit");
}

/// A file in metres and a measurement in inches: the reading goes through
/// metres into inches.
#[test]
fn a_measurement_in_inches_reads_a_file_in_metres_in_inches() {
    let (mut rows, readings) = read(Some("m"));
    rows[0].unit = LengthUnit::Inch;
    rows[0].stated = 60.0;
    let DistanceResult::WithUnit {
        reconstructed,
        error,
        ..
    } = distance_result(&rows[0], &readings.distance[0], Some("m"))
    else {
        panic!("not a reading with a unit");
    };
    assert_close(reconstructed, 1.5 / 0.0254, 1e-3, "reconstructed in");
    assert_close(error, 1.5 / 0.0254 - 60.0, 1e-3, "error in");
}

/// A refused endpoint and one at infinity each name their side.
#[test]
fn a_refused_endpoint_and_one_at_infinity_name_their_side() {
    let (rows, readings) = read(Some("m"));
    assert_eq!(
        distance_result(&rows[2], &readings.distance[2], Some("m")),
        DistanceResult::Refused(EndpointSide::B)
    );
    assert!(matches!(
        readings.distance[2].endpoints[1],
        Err(EndpointRefusal::NoCoveringSquare { .. })
    ));
    assert_eq!(
        distance_result(&rows[3], &readings.distance[3], Some("m")),
        DistanceResult::AtInfinity(EndpointSide::B)
    );
    let sky = readings.distance[3].endpoints[1]
        .as_ref()
        .expect("the endpoint on the sky resolves");
    assert_eq!(sky.chosen().point, SKY);
}

/// A node drawn under a scaled display transform reads the same as the node
/// at the identity: the reading is the file's own. The drawn endpoints move
/// with the transform.
#[test]
fn a_scaled_display_transform_never_scales_a_reading() {
    let (mut state, id) = measured_state(Some("m"));
    let measurement = MeasurementRef::distance(id, 0);
    let before = world_endpoints(&state.scene, measurement);
    let node = crate::scene::node_by_id(&state.scene, id).unwrap();
    let plain = readings(node);
    let index = state.scene.iter().position(|n| n.id == id).unwrap();
    state.scene[index].history.transform_mut().scale = 3.0;
    let node = crate::scene::node_by_id(&state.scene, id).unwrap();
    let scaled = readings(node);
    assert_eq!(plain.distance, scaled.distance);
    let rows = distance_rows(node.recon());
    assert_eq!(
        distance_result(&rows[0], &plain.distance[0], Some("m")),
        distance_result(&rows[0], &scaled.distance[0], Some("m"))
    );
    let after = world_endpoints(&state.scene, measurement);
    assert_eq!(before.len(), 2);
    assert_close(
        (after[0] - after[1]).norm(),
        3.0 * (before[0] - before[1]).norm(),
        1e-6,
        "drawn length",
    );
}

// ── The row text ─────────────────────────────────────────────────────────

/// The Scene tree's rows: a reading with a unit, a refused endpoint, an
/// endpoint at infinity, and a reading in scene units.
#[test]
fn the_rows_read_as_the_scene_tree_draws_them() {
    let id = crate::scene::ReconId::from_raw(7);
    let (rows, readings) = read(Some("m"));
    let text = |i: usize, unit| {
        row_text(
            MeasurementRef::distance(id, i),
            &rows[i],
            &readings.distance[i],
            unit,
        )
    };
    assert_eq!(
        text(0, Some("m")),
        "Distance #0  ear to ear  153 cm  \u{2192} 150.0 cm  \u{2212}2.0%"
    );
    assert_eq!(
        text(2, Some("m")),
        "Distance #2  1 m  \u{26a0} endpoint B: no patch holds the pixel"
    );
    assert_eq!(
        text(3, Some("m")),
        "Distance #3  to the sky  1 m  \u{2192} \u{221e}"
    );
    let (rows, readings) = read(None);
    assert_eq!(
        row_text(
            MeasurementRef::distance(id, 0),
            &rows[0],
            &readings.distance[0],
            None
        ),
        "Distance #0  ear to ear  153 cm  \u{2192} 1.50 su"
    );
}

/// A value that rounds to zero is printed with a plus sign, and a negative
/// one with U+2212.
#[test]
fn signed_values_use_the_minus_sign_and_never_print_minus_zero() {
    assert_eq!(signed(-0.04, 1), "+0.0");
    assert_eq!(signed(-0.5, 1), "\u{2212}0.5");
    assert_eq!(signed(0.7, 1), "+0.7");
    assert_eq!(significant(11.937, 4), "11.94");
    assert_eq!(significant(0.31749, 3), "0.317");
}

// ── The median line ──────────────────────────────────────────────────────

/// Over the fixture's three readings the median scale is the good one, so a
/// good measurement deviates by nothing and the bad one by its own error, and
/// the text says so in the measurement's own unit.
#[test]
fn the_median_line_flags_the_bad_measurement_among_good_ones() {
    let (rows, readings) = read(Some("m"));
    let good = median_line(&rows, &readings.distance, 0).expect("three readings");
    assert_eq!(good.count, 3);
    assert_close(
        good.median_unit_per_scene_unit,
        100.0 * GOOD_SCALE,
        1e-3,
        "cm per su",
    );
    assert_close(good.deviation_percent.unwrap(), 0.0, 1e-4, "good deviation");
    let bad = median_line(&rows, &readings.distance, 4).expect("three readings");
    assert_close(
        bad.deviation_percent.unwrap(),
        100.0 * (BAD_SCALE / GOOD_SCALE - 1.0),
        1e-3,
        "bad deviation",
    );
    assert_eq!(
        median_text(&bad, rows[4].unit),
        "Median over 3: 1 su = 1.02 m; this one +17.6%"
    );
    // A measurement with no reading still gets the line, saying so.
    let refused = median_line(&rows, &readings.distance, 2).expect("three readings");
    assert_eq!(refused.deviation_percent, None);
}

/// With only one reading there is nothing to compare with, and no line.
#[test]
fn the_median_line_is_absent_with_one_reading() {
    let (rows, readings) = read(Some("m"));
    let one_rows = vec![rows[0].clone(), rows[2].clone(), rows[3].clone()];
    let one = vec![
        readings.distance[0].clone(),
        readings.distance[2].clone(),
        readings.distance[3].clone(),
    ];
    assert_eq!(median_line(&one_rows, &one, 0), None);
}

// ── The cache ────────────────────────────────────────────────────────────

/// The readings are resolved once per version: asked twice they are the same
/// value, and an edit's version is resolved afresh, and undoing back to the
/// first version resolves it afresh too, with the same numbers.
#[test]
fn the_readings_are_resolved_once_per_version() {
    let (mut state, id) = measured_state(Some("m"));
    let node = crate::scene::node_by_id(&state.scene, id).unwrap();
    let first = readings(node);
    assert!(Arc::ptr_eq(&first, &readings(node)));
    state
        .delete_image(ImageRef::new(id, 2))
        .expect("image 2 observes nothing and can go");
    let node = crate::scene::node_by_id(&state.scene, id).unwrap();
    let second = readings(node);
    assert!(!Arc::ptr_eq(&first, &second));
    assert!(Arc::ptr_eq(&second, &readings(node)));
    assert_eq!(first.distance.len(), second.distance.len());
}

// ── The selection ────────────────────────────────────────────────────────

/// Selecting a measurement selects nothing else: no image, and the selected
/// point stays as it was. It clears the selected endpoint, and logs one row.
#[test]
fn selecting_a_measurement_selects_nothing_else_and_clears_the_endpoint() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_point(crate::scene::PointRef::new(id, 1));
    state.select_endpoint(MeasurementRef::distance(id, 0), EndpointSide::B);
    assert_eq!(state.selected_endpoint, Some(EndpointSide::B));
    let image_before = state.selected_image;
    state.select_measurement(MeasurementRef::distance(id, 1));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 1))
    );
    assert_eq!(state.selected_endpoint, None);
    assert_eq!(state.selected_image, image_before);
    assert_eq!(
        state.selected_point,
        Some(crate::scene::PointRef::new(id, 1))
    );
    assert_eq!(
        state.action_log.entries().last().map(|e| e.text.as_str()),
        Some("Selected distance #1 in measured")
    );
}

/// Selecting an endpoint selects its measurement, its image with its pixel
/// for Image Detail to reveal, and leaves the selected point alone.
#[test]
fn selecting_an_endpoint_selects_its_image_and_asks_image_detail_to_reveal_it() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_point(crate::scene::PointRef::new(id, 2));
    let measurement = MeasurementRef::distance(id, 1);
    state.select_endpoint(measurement, EndpointSide::B);
    let row = state.distance_row(measurement).unwrap();
    assert_eq!(row.image(EndpointSide::B), 1);
    let image = ImageRef::new(id, 1);
    assert_eq!(state.selected_measurement, Some(measurement));
    assert_eq!(state.selected_endpoint, Some(EndpointSide::B));
    assert_eq!(state.selected_image, Some(image));
    let [x, y] = row.pixel(EndpointSide::B);
    assert_eq!(
        state.look,
        Some((
            image,
            crate::image_detail::Look::Reveal {
                pixel: [x as f32, y as f32]
            }
        ))
    );
    assert_eq!(
        state.selected_point,
        Some(crate::scene::PointRef::new(id, 2))
    );
    // Selecting another image leaves the endpoint, which is a pixel of the
    // image it was selected with.
    state.select_image(Some(ImageRef::new(id, 0)));
    assert_eq!(state.selected_endpoint, None);
    assert_eq!(state.selected_measurement, Some(measurement));
}

/// Deleting an image drops the measurements with an endpoint in it and
/// renumbers the rest: a selection on a later row moves with its measurement,
/// and one on a dropped row is cleared.
#[test]
fn a_deleted_image_moves_the_selection_with_its_measurement_or_clears_it() {
    let (mut state, id) = measured_state(Some("m"));
    // Image 1 holds an endpoint of measurements 1 and 4, so measurement 2
    // becomes row 1.
    state.select_measurement(MeasurementRef::distance(id, 2));
    state
        .delete_image(ImageRef::new(id, 1))
        .expect("the delete runs");
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 1))
    );
    let node = crate::scene::node_by_id(&state.scene, id).unwrap();
    assert_eq!(measurement_count(node.recon()), 3);
    assert_eq!(distance_rows(node.recon())[1].label, "");

    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 4));
    state
        .delete_image(ImageRef::new(id, 1))
        .expect("the delete runs");
    assert_eq!(state.selected_measurement, None);
}

/// Undo and redo keep the selection where the version arrived at has its row,
/// and clear it where it does not; a redo that brings the row back leaves it
/// cleared.
#[test]
fn undo_and_redo_keep_the_selection_only_where_the_row_is() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .delete_image(ImageRef::new(id, 1))
        .expect("the delete runs");
    // Row 2 of the three rows left.
    state.select_measurement(MeasurementRef::distance(id, 2));
    state.undo(id).expect("an undo");
    state.prune_measurement_selection();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 2)),
        "the version before the delete has five rows"
    );
    state.select_measurement(MeasurementRef::distance(id, 4));
    state.redo(id).expect("a redo");
    state.prune_measurement_selection();
    assert_eq!(state.selected_measurement, None);
    state.undo(id).expect("an undo");
    state.prune_measurement_selection();
    assert_eq!(state.selected_measurement, None);
}

/// The fixture's measurement endpoints land on its points: the first two
/// endpoints of measurement 0 resolve to points 0 and 1, whose positions are
/// the fixture's.
#[test]
fn the_fixtures_endpoints_resolve_to_its_points() {
    let (_, readings) = read(Some("m"));
    let resolved = &readings.distance[0];
    for (side, point) in [(EndpointSide::A, 0), (EndpointSide::B, 1)] {
        let resolution = endpoint(resolved, side).as_ref().expect("it resolves");
        assert_eq!(resolution.chosen().point, point);
        let position = resolution.position().finite().expect("finite");
        for c in 0..3 {
            assert_close(position[c], POINTS[point as usize][c], 1e-4, "position");
        }
    }
}
