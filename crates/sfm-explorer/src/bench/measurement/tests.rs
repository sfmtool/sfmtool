// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A measurement on the bench, on the fixture in
//! [`crate::measurements::fixture`]: putting it on and focusing it, following
//! its origin across the edits that renumber rows, the commit, the direct
//! delete, and the selection rules a focused measurement keeps.

use std::sync::Arc;

use sfmtool_core::bench::{BenchItem, EditableMeasurement, EditableMeasurementBody, ItemKind};
use sfmtool_core::LengthUnit;

use crate::measurements::fixture::{measured_reconstruction, measured_state};
use crate::measurements::{EndpointSide, MeasurementRef};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;

fn version_count(state: &AppState, id: ReconId) -> usize {
    state.node(id).unwrap().history.versions().len()
}

fn item(state: &AppState, id: ReconId, label: &str) -> Arc<EditableMeasurement> {
    Arc::clone(state.bench_measurement(id, label).expect("on the bench"))
}

/// The row an item's origin names at the cursor.
fn origin_row(state: &AppState, id: ReconId, label: &str) -> Option<u32> {
    let node = state.node(id).unwrap();
    state.resolved_measurement_origin(node, &item(state, id, label))
}

/// Install `edit` of the item `label` as a bench step, as the later steps'
/// fields will.
fn edit_item(
    state: &mut AppState,
    id: ReconId,
    label: &str,
    edit: impl FnOnce(&mut EditableMeasurement),
) {
    let mut next = (*item(state, id, label)).clone();
    edit(&mut next);
    let index = state.node_index(id).unwrap();
    let bench = state.scene[index]
        .history
        .current_bench()
        .replace(label, BenchItem::Measurement(Arc::new(next)))
        .unwrap();
    state.scene[index]
        .history
        .push_bench(Arc::new(bench), "edited the item");
}

fn set_stated(item: &mut EditableMeasurement, stated: f64, unit: LengthUnit) {
    let EditableMeasurementBody::Distance(d) = &mut item.body;
    d.distance = Some((stated, unit));
}

fn stated_of(state: &AppState, id: ReconId, row: usize) -> f64 {
    crate::measurements::distance_rows(state.node(id).unwrap().edited().measurements())[row].stated
}

/// A put copies the row as the file holds it, labels the item by its kind
/// and row, focuses it, selects its row and leaves the point selection.
#[test]
fn a_put_copies_the_row_focuses_the_item_and_selects_its_row() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_point(PointRef::new(id, 1));
    let before = version_count(&state, id);
    let put = state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    assert_eq!(put.label, "distance #1");
    assert!(put.made);
    assert_eq!(version_count(&state, id), before + 1);
    let held = item(&state, id, "distance #1");
    assert_eq!(held.label, "nose to tail");
    let d = held.as_distance().unwrap();
    let rows = crate::measurements::distance_rows(state.node(id).unwrap().edited().measurements());
    let row = &rows[1];
    assert_eq!(d.distance, Some((row.stated, LengthUnit::Metre)));
    assert_eq!(d.endpoints[0].unwrap().image, row.images[0]);
    assert_eq!(d.endpoints[1].unwrap().pixel, row.pixels[1]);
    assert_eq!(state.focused_measurement_label(id), Some("distance #1"));
    assert_eq!(state.focused_track_label(id), None);
    assert_eq!(state.focused_item_kind(), Some(ItemKind::Measurement));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 1))
    );
    assert_eq!(state.selected_point, Some(PointRef::new(id, 1)));
    assert!(state
        .action_log
        .entries()
        .any(|e| e.text.starts_with("Put distance #1 on the bench")));
    // The bench step leaves the document clean.
    assert!(!state.node(id).unwrap().history.is_dirty());
}

/// Putting the same measurement on twice focuses the item already there and
/// pushes no version.
#[test]
fn a_second_put_focuses_the_item_already_there() {
    let (mut state, id) = measured_state(Some("m"));
    let measurement = MeasurementRef::distance(id, 0);
    state.put_measurement_on_bench(measurement).unwrap();
    state.unfocus_bench_item();
    let before = version_count(&state, id);
    let again = state.put_measurement_on_bench(measurement).unwrap();
    assert_eq!(again.label, "distance #0");
    assert!(!again.made);
    assert_eq!(version_count(&state, id), before);
    assert_eq!(state.bench(id).unwrap().len(), 1);
    assert_eq!(state.focused_measurement_label(id), Some("distance #0"));
}

/// While a measurement is focused, a point selection on its node keeps it
/// focused, and another measurement's unfocuses it.
#[test]
fn a_focused_measurement_keeps_the_point_selection_and_leaves_with_another_row() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    state.select_point(PointRef::new(id, 2));
    assert_eq!(state.focused_measurement_label(id), Some("distance #0"));
    state.select_measurement(MeasurementRef::distance(id, 0));
    assert_eq!(state.focused_measurement_label(id), Some("distance #0"));
    state.select_measurement(MeasurementRef::distance(id, 3));
    assert_eq!(state.focused_item(), None);
    // Unfocusing a measurement leaves the point selection alone.
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    state.unfocus_bench_item();
    assert_eq!(state.selected_point, Some(PointRef::new(id, 2)));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 0))
    );
}

/// Track View's box is the track's and Measurement View's the measurement's:
/// focusing one kind leaves the other's label empty.
#[test]
fn each_panel_s_label_is_its_own_kind_s() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_point_on_bench(PointRef::new(id, 0), None)
        .unwrap();
    let track = state.focused_item_label(id).unwrap().to_string();
    assert_eq!(state.focused_track_label(id), Some(track.as_str()));
    assert_eq!(state.focused_measurement_label(id), None);
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    assert_eq!(state.focused_track_label(id), None);
    assert_eq!(state.focused_measurement_label(id), Some("distance #0"));
    // A tick of Track View's box with no point selected goes back to the
    // track, and Measurement View's with no measurement selected to the
    // measurement.
    assert_eq!(
        state.most_recent_item_of(ItemKind::Track),
        Some((id, track))
    );
    assert_eq!(
        state.most_recent_item_of(ItemKind::Measurement),
        Some((id, "distance #0".to_string()))
    );
}

/// The Edit box: ticked with a measurement selected it puts it on; cleared it
/// unfocuses; ticked with none selected it goes back to the last item; with
/// no item to go back to it is refused.
#[test]
fn the_edit_box_puts_focuses_and_unfocuses() {
    let (mut state, id) = measured_state(Some("m"));
    assert_eq!(
        state.set_measurement_editing(id, true),
        Err(super::nothing_to_edit())
    );
    state.select_measurement(MeasurementRef::distance(id, 2));
    state.set_measurement_editing(id, true).unwrap();
    assert_eq!(state.focused_measurement_label(id), Some("distance #2"));
    state.set_measurement_editing(id, false).unwrap();
    assert_eq!(state.focused_item(), None);
    assert_eq!(state.bench(id).unwrap().len(), 1);
    state.selected_measurement = None;
    state.set_measurement_editing(id, true).unwrap();
    assert_eq!(state.focused_measurement_label(id), Some("distance #2"));
}

/// Deleting an earlier measurement moves the origin down a row; deleting the
/// origin's own row leaves the item with none, and its commit appends.
#[test]
fn the_origin_follows_a_delete_and_a_deleted_row_commits_as_a_new_one() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 3))
        .unwrap();
    state
        .delete_measurement(MeasurementRef::distance(id, 1))
        .unwrap();
    assert_eq!(origin_row(&state, id, "distance #3"), Some(2));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 2))
    );
    assert!(state
        .action_log
        .entries()
        .any(|e| e.text.starts_with("Deleted distance #1 in measured")));
    // An undo of the delete brings the row back where it was.
    state.undo(id).unwrap();
    assert_eq!(origin_row(&state, id, "distance #3"), Some(3));
    state.redo(id).unwrap();
    state
        .delete_measurement(MeasurementRef::distance(id, 2))
        .unwrap();
    assert_eq!(origin_row(&state, id, "distance #3"), None);
    assert_eq!(state.selected_measurement, None);
    let committed = state.commit_bench_measurement(id, "distance #3").unwrap();
    assert!(committed.changed);
    assert!(!committed.replaced);
    assert_eq!(committed.measurement, MeasurementRef::distance(id, 3));
    let rows = crate::measurements::distance_rows(state.node(id).unwrap().edited().measurements());
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[3].label, "to the sky");
    // The commit seats the item on the row it added and selects it.
    assert_eq!(origin_row(&state, id, "distance #3"), Some(3));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 3))
    );
}

/// Deleting an image that drops an earlier measurement moves the origin with
/// the rows, and the bench's endpoints in later images move down with it; an
/// endpoint in the deleted image is cleared.
#[test]
fn the_origin_and_the_endpoints_follow_an_image_delete() {
    let (mut state, id) = measured_state(Some("m"));
    // Row 2 is in image 0 alone; row 4 has endpoint A in image 1.
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 2))
        .unwrap();
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 4))
        .unwrap();
    state.delete_image(ImageRef::new(id, 1)).unwrap();
    // Rows 1 and 4 had an endpoint in image 1, so row 2 is now row 1.
    assert_eq!(origin_row(&state, id, "distance #2"), Some(1));
    assert_eq!(origin_row(&state, id, "distance #4"), None);
    let moved = item(&state, id, "distance #4");
    let d = moved.as_distance().unwrap();
    assert_eq!(d.endpoints[0], None);
    assert_eq!(d.endpoints[1].unwrap().image, 0);
    assert!(state.action_log.entries().any(|e| e
        .text
        .contains("cleared 1 bench measurement endpoint in it")));
}

/// A commit replaces the row its origin names, shares the base's point
/// columns, marks the node dirty, keeps the item focused and selects the row;
/// an undo restores the row and keeps the item.
#[test]
fn a_commit_replaces_the_row_and_an_undo_restores_it() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    edit_item(&mut state, id, "distance #1", |m| {
        set_stated(m, 3.5, LengthUnit::Metre)
    });
    let base_before = Arc::clone(&state.node(id).unwrap().edited().base);
    let committed = state.commit_bench_measurement(id, "distance #1").unwrap();
    assert!(committed.changed);
    assert!(committed.replaced);
    assert_eq!(committed.measurement, MeasurementRef::distance(id, 1));
    let node = state.node(id).unwrap();
    assert!(Arc::ptr_eq(&node.edited().base, &base_before));
    assert!(node.history.is_dirty());
    assert_eq!(stated_of(&state, id, 1), 3.5);
    assert_eq!(state.focused_measurement_label(id), Some("distance #1"));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 1))
    );
    assert!(state
        .action_log
        .entries()
        .any(|e| e.text.starts_with("Committed distance #1 in measured")));
    state.undo(id).unwrap();
    assert!((stated_of(&state, id, 1) - 3.06).abs() < 1e-6);
    assert_eq!(
        item(&state, id, "distance #1")
            .as_distance()
            .unwrap()
            .distance,
        Some((3.5, LengthUnit::Metre))
    );
}

/// A commit of what the row already holds pushes no version and still
/// selects the row.
#[test]
fn a_commit_that_changes_nothing_pushes_no_version() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    state.selected_measurement = None;
    let before = version_count(&state, id);
    let committed = state.commit_bench_measurement(id, "distance #0").unwrap();
    assert!(!committed.changed);
    assert_eq!(version_count(&state, id), before);
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 0))
    );
    assert!(state
        .action_log
        .entries()
        .any(|e| e.text.contains("no effect, the row already holds it")));
}

/// The commit's refusals are the button's: an endpoint not placed, and a
/// track named as the item.
#[test]
fn the_commit_refusal_is_the_button_s() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    assert_eq!(state.measurement_commit_refusal(id, "distance #0"), None);
    edit_item(&mut state, id, "distance #0", |m| {
        let EditableMeasurementBody::Distance(d) = &mut m.body;
        d.endpoints[1] = None;
    });
    let greyed = state.measurement_commit_refusal(id, "distance #0");
    let refused = state
        .commit_bench_measurement(id, "distance #0")
        .unwrap_err();
    assert_eq!(greyed, Some(refused.clone()));
    assert_eq!(
        refused,
        "Cannot commit distance #0: endpoint B is not placed, and a distance needs both \
         endpoints."
    );
    state
        .put_point_on_bench(PointRef::new(id, 0), None)
        .unwrap();
    let track = state.focused_track_label(id).unwrap().to_string();
    assert_eq!(
        state.commit_bench_measurement(id, &track),
        Err(super::not_a_measurement(&track))
    );
}

/// On a `sift_files` node the bench is view-only: a measurement goes on, is
/// focused, renamed and discarded, and its commit is refused with the bench's
/// sentence.
#[test]
fn a_view_only_bench_takes_a_measurement_and_refuses_its_commit() {
    let mut recon = measured_reconstruction(Some("m"));
    let keypoints = recon.point_set.keypoints_xy().cloned();
    let count = recon.point_set.tracks.len();
    recon.point_set.observations = sfmtool_core::ObservationSource::SiftFiles {
        feature_indexes: vec![0; count],
        keypoints_xy: keypoints,
        feature_tool_hashes: vec![[0u8; 16]; 3],
        sift_content_hashes: vec![[0u8; 16]; 3],
    };
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        recon,
    ));
    let sentence = state.bench_view_only_refusal(id).expect("view-only");
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    assert_eq!(
        state.commit_bench_measurement(id, "distance #0"),
        Err(sentence.clone())
    );
    assert_eq!(
        state.measurement_commit_refusal(id, "distance #0"),
        Some(sentence)
    );
    state
        .rename_bench_item(id, "distance #0", "ear width")
        .unwrap();
    assert_eq!(state.focused_measurement_label(id), Some("ear width"));
    state.discard_bench_item(id, "ear width").unwrap();
    assert!(state.bench(id).unwrap().is_empty());
    assert_eq!(state.focused_item(), None);
}

/// A click on an item's endpoint block selects the endpoint and its image.
#[test]
fn an_item_endpoint_click_selects_its_image_and_the_endpoint() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    state.select_item_endpoint(id, EndpointSide::B);
    assert_eq!(state.selected_image, Some(ImageRef::new(id, 1)));
    assert_eq!(state.selected_endpoint, Some(EndpointSide::B));
    assert_eq!(state.focused_measurement_label(id), Some("distance #1"));
}
