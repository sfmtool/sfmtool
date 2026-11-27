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
    let rows = distance_rows(&recon.image_set.measurements);
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
    rows[0].stated = Some(60.0);
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
    let rows = distance_rows(node.edited().measurements());
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
        "Distance #2  bare pixel  1 m  \u{26a0} endpoint B: no patch holds the pixel"
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
    let good = scale_comparison(&rows, &readings.distance, 0).expect("three readings");
    assert_eq!(good.count, 3);
    assert_close(
        good.median_unit_per_scene_unit,
        100.0 * GOOD_SCALE,
        1e-3,
        "cm per su",
    );
    assert_close(good.deviation_percent.unwrap(), 0.0, 1e-4, "good deviation");
    let bad = scale_comparison(&rows, &readings.distance, 4).expect("three readings");
    assert_close(
        bad.deviation_percent.unwrap(),
        100.0 * (BAD_SCALE / GOOD_SCALE - 1.0),
        1e-3,
        "bad deviation",
    );
    assert_eq!(
        scale_comparison_text(&bad, rows[4].unit),
        "Median over 3: 1 su = 1.02 m; this one +17.6%"
    );
    // A measurement with no reading still gets the line, saying so.
    let refused = scale_comparison(&rows, &readings.distance, 2).expect("three readings");
    assert_eq!(refused.deviation_percent, None);
}

/// The fixture with measurement 0 stating no distance (a ruler) and
/// measurement 1's endpoint B not placed, its file declaring `unit`.
fn with_ruler_and_unplaced(unit: Option<&str>) -> (Vec<DistanceRow>, MeasurementReadings) {
    let mut recon = measured_reconstruction(unit);
    let m = recon.image_set.measurements.distance.as_mut().unwrap();
    m.distances[0] = f32::NAN;
    m.image_indexes[[1, 1]] = u32::MAX;
    m.endpoints_xy[[1, 1, 0]] = f32::NAN;
    m.endpoints_xy[[1, 1, 1]] = f32::NAN;
    let rows = distance_rows(&recon.image_set.measurements);
    let readings = MeasurementReadings::of(&EditedReconstruction::new(Arc::new(recon)));
    (rows, readings)
}

/// A measurement that states no distance reads the reconstructed distance
/// alone, in the file's unit or in scene units, and its row prints no stated
/// distance; one with an endpoint not placed says so in its result and row.
#[test]
fn a_ruler_reads_its_distance_and_an_unplaced_endpoint_says_so() {
    let (rows, readings) = with_ruler_and_unplaced(Some("cm"));
    assert_eq!(rows[0].stated, None);
    assert_eq!(rows[0].stated_text(), None);
    let result = distance_result(&rows[0], &readings.distance[0], Some("cm"));
    let DistanceResult::Unstated {
        reconstructed,
        unit,
    } = result
    else {
        panic!("not a ruler's reading: {result:?}");
    };
    // The file is in centimetres, and the reading is in the file's unit
    // whatever the row's own unit is.
    assert_close(reconstructed, 1.5, 1e-5, "reconstructed");
    assert_eq!(unit, Some(LengthUnit::Centimetre));
    let measurement = MeasurementRef::distance(crate::scene::ReconId::from_raw(0), 0);
    assert_eq!(
        row_text(measurement, &rows[0], &readings.distance[0], Some("cm")),
        format!("Distance #0  ear to ear  {ARROW} 1.500 cm")
    );
    assert_eq!(
        distance_result(&rows[0], &readings.distance[0], None),
        DistanceResult::Unstated {
            reconstructed,
            unit: None
        }
    );
    assert_eq!(
        reading_text(
            &rows[0],
            &readings.distance[0],
            distance_result(&rows[0], &readings.distance[0], None)
        ),
        format!("{ARROW} 1.50 su")
    );

    assert_eq!(rows[1].endpoint(EndpointSide::B), None);
    assert!(rows[1].endpoint(EndpointSide::A).is_some());
    let result = distance_result(&rows[1], &readings.distance[1], Some("cm"));
    assert_eq!(result, DistanceResult::NotPlaced(EndpointSide::B));
    assert_eq!(
        reading_text(&rows[1], &readings.distance[1], result),
        "endpoint B: not placed"
    );
}

/// The median line counts only measurements that state a distance and have a
/// reading, and a ruler's line says it states none.
#[test]
fn the_median_line_leaves_out_a_ruler() {
    let (rows, readings) = with_ruler_and_unplaced(Some("m"));
    // Of the five, 0 states nothing, 1 has B not placed, 2 is refused and 3 is
    // at infinity: only 4 is left, so there is no line.
    assert_eq!(scale_comparison(&rows, &readings.distance, 4), None);
    let (full_rows, full) = read(Some("m"));
    let mut rows = full_rows;
    rows[1].stated = None;
    let line = scale_comparison(&rows, &full.distance, 1).expect("two readings left");
    assert_eq!(line.count, 2);
    assert_eq!(line.deviation_percent, None);
    assert!(!line.states_distance);
    assert!(scale_comparison_text(&line, rows[1].unit).ends_with("; this one states no distance"));
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
    assert_eq!(scale_comparison(&one_rows, &one, 0), None);
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
    assert_eq!(measurement_count(node.edited().measurements()), 3);
    assert_eq!(
        distance_rows(node.edited().measurements())[1].label,
        "bare pixel"
    );

    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 4));
    state
        .delete_image(ImageRef::new(id, 1))
        .expect("the delete runs");
    assert_eq!(state.selected_measurement, None);
}

/// Undo and redo carry the selection across the delete that renumbered the
/// rows, back onto the measurement that was selected; a redo that drops the
/// selected row clears the selection, and an undo that brings the row back
/// leaves it cleared.
#[test]
fn undo_and_redo_carry_the_selection_with_its_measurement() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .delete_image(ImageRef::new(id, 1))
        .expect("the delete runs");
    // Row 2 of the three rows left, which was row 3 before the delete.
    state.select_measurement(MeasurementRef::distance(id, 2));
    state.undo(id).expect("an undo");
    state.prune_measurement_selection();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 3)),
        "the selection follows its measurement back to its own row"
    );
    // Row 4 has an endpoint in image 1, so the redo drops it.
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

/// Deleting an image of a value built by hand whose measurement columns
/// disagree in length does not panic, and the selection follows the rows the
/// subset keeps: the row with no image index reads as one whose endpoints are
/// not placed, which no delete drops.
#[test]
fn deleting_an_image_of_mismatched_measurement_columns_does_not_panic() {
    let mut recon = crate::measurements::fixture::measured_reconstruction(Some("m"));
    let m = recon.image_set.measurements.distance.as_mut().unwrap();
    // Five distances, but image indexes and units for the first four only.
    m.image_indexes = m.image_indexes.slice(ndarray::s![..4, ..]).to_owned();
    m.units = m.units.slice(ndarray::s![..4]).to_owned();
    let mut state = crate::state::AppState::new();
    let id = state.append_node(crate::scene::SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        recon,
    ));
    state.select_measurement(MeasurementRef::distance(id, 3));
    state
        .delete_image(ImageRef::new(id, 2))
        .expect("the delete runs");
    // Rows 0 to 3 name images 0 and 1 only and survive, and row 4 names
    // none.
    let node = state.node(id).unwrap();
    assert_eq!(measurement_count(node.edited().measurements()), 5);
    assert_eq!(
        distance_rows(node.edited().measurements())[4].endpoints,
        [None, None]
    );
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 3))
    );
}

// ── A point's reference pixel ───────────────────────────────────────────

/// A point's reference pixel is the keypoint of its reference observation in
/// that observation's image, and an endpoint placed there resolves alone to
/// texel `(0, 0)` of the point's own patch, the point itself.
#[test]
fn a_reference_pixel_resolves_to_the_centre_of_the_point_s_own_patch() {
    use sfmtool_core::reconstruction::measurement_endpoint::{
        resolve_measurement_endpoint, CandidateStanding, MeasurementEndpointOptions,
    };
    let recon = super::fixture::referenced_reconstruction(Some("m"));
    let edited = EditedReconstruction::new(Arc::new(recon));
    // Point 0's reference is its observation in image 1, so the pixel is in
    // image 1, not in image 0, where its track starts.
    let reference = reference_pixel(&edited, 0).expect("point 0 has a reference");
    assert_eq!(reference.observation, 1);
    assert_eq!(reference.image, 1);
    let keypoint = edited.point(0).unwrap().keypoint_xy(1).unwrap();
    assert_eq!(
        reference.pixel,
        [f64::from(keypoint[0]), f64::from(keypoint[1])]
    );
    let resolved = resolve_measurement_endpoint(
        &edited,
        reference.image,
        reference.pixel,
        &MeasurementEndpointOptions::default(),
    )
    .expect("the pixel resolves");
    let chosen = resolved.chosen();
    assert_eq!(chosen.point, 0);
    assert_eq!(chosen.standing, CandidateStanding::Alone);
    let [s, t] = chosen.patch_coords;
    assert!(s.abs() < 1e-4 && t.abs() < 1e-4, "texel ({s}, {t})");

    let reference = reference_pixel(&edited, 2).expect("point 2 has a reference");
    assert_eq!((reference.observation, reference.image), (0, 0));
}

/// The three refusals, each in its own sentence: no patch frame, no reference
/// observation (`-1`, or no column at all), and a point at infinity.
#[test]
fn a_point_without_a_reference_pixel_says_why() {
    let referenced = EditedReconstruction::new(Arc::new(
        super::fixture::referenced_reconstruction(Some("m")),
    ));
    assert_eq!(
        reference_pixel(&referenced, 1),
        Err(NO_REFERENCE_FOR_PIXEL.to_string())
    );
    assert_eq!(
        reference_pixel(&referenced, SKY),
        Err(AT_INFINITY_FOR_PIXEL.to_string())
    );
    // The fixture without the column has frames and no reference to read.
    let unreferenced = EditedReconstruction::new(Arc::new(measured_reconstruction(Some("m"))));
    assert_eq!(
        reference_pixel(&unreferenced, 0),
        Err(NO_REFERENCE_FOR_PIXEL.to_string())
    );
    // The demo reconstruction stores no patch frames at all.
    let demo = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
    assert_eq!(
        reference_pixel(&demo, 0),
        Err(NO_PATCH_FOR_PIXEL.to_string())
    );
}

// ── The spot under the pointer ──────────────────────────────────────────

/// A square tilted out of every axis plane, its centre off the origin, with
/// half-extents that differ, so a reading that mixes up `u` and `v` or drops
/// a half-extent fails.
fn tilted_square() -> sfmtool_core::patch::cloud::OrientedPatch {
    let turn = nalgebra::UnitQuaternion::from_euler_angles(0.3, -0.2, 0.7);
    sfmtool_core::patch::cloud::OrientedPatch::new(
        nalgebra::Point3::new(0.5, -1.0, 2.0),
        turn * nalgebra::Vector3::x(),
        turn * nalgebra::Vector3::y(),
        [0.3, 0.2],
    )
}

/// An eye `distance` in front of `square`'s centre, off its normal by
/// `angle_deg` towards `u`: the side the surfel is drawn on.
fn eye_before(
    square: &sfmtool_core::patch::cloud::OrientedPatch,
    distance: f64,
    angle_deg: f64,
) -> nalgebra::Point3<f64> {
    let a = angle_deg.to_radians();
    square.center + (square.normal() * a.cos() + square.u_axis * a.sin()) * distance
}

/// The ray from `eye` through texel `(s, t)` of `square` drawn at `scale`
/// meets the square at that texel, at any scale: the surfel at scale 2 shows
/// texel `(s, t)` where the unscaled one shows `(2s, 2t)`.
#[test]
fn texel_under_ray_reads_the_texel_the_ray_meets() {
    let square = tilted_square();
    let eye = eye_before(&square, 3.0, 25.0);
    for scale in [1.0, 2.0, 0.5] {
        for [s, t] in [[0.6, -0.35], [-0.9, 0.8], [0.0, 0.0]] {
            let aim = square.center
                + square.u_axis * (s * square.half_extent[0] * scale)
                + square.v_axis * (t * square.half_extent[1] * scale);
            let [hs, ht] = texel_under_ray(&square, eye, (aim - eye) * 7.0, scale)
                .unwrap_or_else(|| panic!("({s}, {t}) at scale {scale} is on the square"));
            assert!(
                (hs - s).abs() < 1e-9 && (ht - t).abs() < 1e-9,
                "({hs}, {ht}) against ({s}, {t}) at scale {scale}"
            );
        }
    }
}

/// No texel where the ray misses the square, sees its back, points away from
/// it, grazes its plane under the bench's bar, or the point is at infinity.
#[test]
fn texel_under_ray_answers_none_off_the_drawn_square() {
    let square = tilted_square();
    let eye = eye_before(&square, 3.0, 25.0);
    let through = |s: f64, t: f64| square.to_world(s, t) - eye;
    assert!(texel_under_ray(&square, eye, through(0.95, -0.95), 1.0).is_some());
    // Outside the square: the dot of a point may stick out past a small one.
    assert_eq!(texel_under_ray(&square, eye, through(1.05, 0.0), 1.0), None);
    assert_eq!(texel_under_ray(&square, eye, through(0.0, -1.2), 1.0), None);
    // The same texel is inside the square drawn twice as large.
    assert!(texel_under_ray(&square, eye, through(1.05, 0.0), 2.0).is_some());
    // Pointing away from the square.
    assert_eq!(texel_under_ray(&square, eye, -through(0.1, 0.1), 1.0), None);
    // From behind: the viewport culls the surfel's back.
    let behind = square.center - (eye - square.center);
    assert!((behind - square.center).dot(&square.normal()) < 0.0);
    let from_behind = square.to_world(0.2, 0.1) - behind;
    assert_eq!(texel_under_ray(&square, behind, from_behind, 1.0), None);
    // A point at infinity has no place to measure to.
    let mut sky = square.clone();
    sky.w = 0.0;
    assert_eq!(texel_under_ray(&sky, eye, through(0.2, 0.1), 1.0), None);
}

/// The grazing bar is the bench's [`MIN_PLANE_ANGLE_DEG`]: a ray at a degree
/// more than it to the plane reads a texel, and one a degree less does not.
#[test]
fn texel_under_ray_refuses_a_ray_grazing_the_square() {
    use crate::bench::geometry::MIN_PLANE_ANGLE_DEG;
    let square = tilted_square();
    // The angle off the normal is 90° less the angle to the plane; the ray
    // runs to the centre, so it meets the square wherever the bar allows.
    for (to_plane, reads) in [
        (MIN_PLANE_ANGLE_DEG + 1.0, true),
        (MIN_PLANE_ANGLE_DEG - 1.0, false),
    ] {
        let eye = eye_before(&square, 3.0, 90.0 - to_plane);
        let hit = texel_under_ray(&square, eye, square.center - eye, 1.0);
        assert_eq!(hit.is_some(), reads, "{to_plane}° to the plane: {hit:?}");
    }
}

/// Degenerate input answers `None` and never panics: a ray or an eye that is
/// not a number, a ray of no length, a scale that is not a positive number,
/// and a square of no extent.
#[test]
fn texel_under_ray_answers_none_on_degenerate_input() {
    let square = tilted_square();
    let eye = eye_before(&square, 3.0, 25.0);
    let ray = square.center - eye;
    let nan = nalgebra::Vector3::new(f64::NAN, 0.0, 0.0);
    assert!(texel_under_ray(&square, eye, ray, 1.0).is_some());
    assert_eq!(texel_under_ray(&square, eye, nan, 1.0), None);
    assert_eq!(
        texel_under_ray(&square, eye, nalgebra::Vector3::zeros(), 1.0),
        None
    );
    assert_eq!(
        texel_under_ray(&square, nalgebra::Point3::from(nan), ray, 1.0),
        None
    );
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(texel_under_ray(&square, eye, ray, scale), None, "{scale}");
    }
    let mut flat = square.clone();
    flat.half_extent = [0.0, 0.0];
    assert_eq!(texel_under_ray(&flat, eye, ray, 1.0), None);
    let mut no_axes = square.clone();
    no_axes.u_axis = nalgebra::Vector3::zeros();
    assert_eq!(texel_under_ray(&no_axes, eye, ray, 1.0), None);
}

/// The pixel for a spot on point 0 is in its reference image, image 1, away
/// from the keypoint, and an endpoint there resolves alone to point 0 at that
/// texel; with no spot it is the reference pixel itself.
#[test]
fn a_spot_pixel_resolves_to_its_point_at_its_texel() {
    use sfmtool_core::reconstruction::measurement_endpoint::{
        resolve_measurement_endpoint, CandidateStanding, MeasurementEndpointOptions,
    };
    let edited = EditedReconstruction::new(Arc::new(super::fixture::referenced_reconstruction(
        Some("m"),
    )));
    let reference = reference_pixel(&edited, 0).unwrap();
    assert_eq!(spot_pixel(&edited, 0, None), Ok(reference));
    let (s, t) = (0.6, -0.4);
    let spot = spot_pixel(&edited, 0, Some([s, t])).expect("point 0 has a reference");
    assert_eq!((spot.observation, spot.image), (1, 1));
    let off = (spot.pixel[0] - reference.pixel[0]).hypot(spot.pixel[1] - reference.pixel[1]);
    assert!(off > 5.0, "the spot is {off} px from the keypoint");
    let resolved = resolve_measurement_endpoint(
        &edited,
        spot.image,
        spot.pixel,
        &MeasurementEndpointOptions::default(),
    )
    .expect("the pixel resolves");
    let chosen = resolved.chosen();
    assert_eq!(chosen.point, 0);
    assert_eq!(chosen.standing, CandidateStanding::Alone);
    let [rs, rt] = chosen.patch_coords;
    assert!(
        (rs - s).abs() < 1e-6 && (rt - t).abs() < 1e-6,
        "texel ({rs}, {rt})"
    );
}

/// A spot whose place falls outside the reference image, or on a square that
/// faces away from the reference camera, takes the reference pixel; a texel
/// outside the square does too. The three refusals stand with a spot as
/// without one.
#[test]
fn a_spot_pixel_falls_back_to_the_reference_pixel() {
    let referenced = super::fixture::referenced_reconstruction(Some("m"));
    // Point 0's patch 20 units across: texel `(0.9, 0)` is 9 units to the
    // side, well off image 1 at a depth of 8 and a focal length of 500.
    let mut wide = referenced.clone();
    wide.point_set.patch_u_halfvec_xyz.as_mut().unwrap()[[0, 0]] = 10.0;
    let wide = EditedReconstruction::new(Arc::new(wide));
    let reference = reference_pixel(&wide, 0).unwrap();
    assert_ne!(
        spot_pixel(&wide, 0, Some([0.01, 0.0])).unwrap().pixel,
        reference.pixel,
        "a spot near the centre is on the image"
    );
    assert_eq!(spot_pixel(&wide, 0, Some([0.9, 0.0])), Ok(reference));

    // Point 0's patch turned to face away from the cameras.
    let mut away = referenced.clone();
    let set = &mut away.point_set;
    std::mem::swap(
        set.patch_u_halfvec_xyz.as_mut().unwrap(),
        set.patch_v_halfvec_xyz.as_mut().unwrap(),
    );
    let away = EditedReconstruction::new(Arc::new(away));
    let reference = reference_pixel(&away, 0).unwrap();
    assert_eq!(spot_pixel(&away, 0, Some([0.6, -0.4])), Ok(reference));

    let edited = EditedReconstruction::new(Arc::new(referenced));
    let reference = reference_pixel(&edited, 0).unwrap();
    assert_eq!(spot_pixel(&edited, 0, Some([1.5, 0.0])), Ok(reference));
    assert_eq!(spot_pixel(&edited, 0, Some([f64::NAN, 0.0])), Ok(reference));

    let spot = Some([0.6, -0.4]);
    assert_eq!(
        spot_pixel(&edited, 1, spot),
        Err(NO_REFERENCE_FOR_PIXEL.to_string())
    );
    assert_eq!(
        spot_pixel(&edited, SKY, spot),
        Err(AT_INFINITY_FOR_PIXEL.to_string())
    );
    let demo = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
    assert_eq!(
        spot_pixel(&demo, 0, spot),
        Err(NO_PATCH_FOR_PIXEL.to_string())
    );
}
