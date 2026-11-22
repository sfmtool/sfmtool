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
    let count = recon.point_set.tracks.len();
    recon.point_set.observations = sfmtool_core::ObservationSource::SiftFiles {
        feature_indexes: vec![0; count],
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

/// An undo of a measurement delete puts the selection back on the measurement
/// that was selected, not on the row that now has its index, and a redo moves
/// it down again; the same across an image delete that drops rows.
#[test]
fn undo_and_redo_carry_the_selected_measurement_across_a_renumbering() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 2));
    state
        .delete_measurement(MeasurementRef::distance(id, 1))
        .unwrap();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 1))
    );
    state.undo(id).unwrap();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 2))
    );
    state.redo(id).unwrap();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 1))
    );
    // Undo the delete again, then delete image 1, which drops rows 1 and 4.
    state.undo(id).unwrap();
    state.select_measurement(MeasurementRef::distance(id, 3));
    state.delete_image(ImageRef::new(id, 1)).unwrap();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 2))
    );
    state.undo(id).unwrap();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 3))
    );
    state.redo(id).unwrap();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 2))
    );
    // A selected row the redone step drops is cleared.
    state.undo(id).unwrap();
    state.select_measurement(MeasurementRef::distance(id, 4));
    state.redo(id).unwrap();
    assert_eq!(state.selected_measurement, None);
}

/// `tab` moved into the Scene tree's node, behind it, and the Scene tree
/// brought to the front there: an arrangement a person makes by dragging.
fn docked_behind_the_scene_tree(state: &mut AppState, tab: crate::dock::Tab) {
    use crate::dock::Tab;
    state.hide_panel(tab);
    let at = state.dock.find_tab(&Tab::SceneGraph).expect("docked");
    state
        .dock
        .leaf_mut(at.node_path())
        .expect("a tab sits in a leaf")
        .append_tab(tab);
    let at = state.dock.find_tab(&Tab::SceneGraph).expect("docked");
    let _ = state.dock.set_active_tab(at);
    assert!(state.panel_is_in_front(Tab::SceneGraph));
}

/// A double-click on a Bench row raises the item's panel beside the Scene
/// tree, never over it (G2, G9): with the panel docked behind the tree,
/// the tree stays in front, for a measurement and for a track.
#[test]
fn a_bench_row_s_double_click_never_covers_the_scene_tree() {
    use crate::dock::Tab;
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    state
        .put_point_on_bench(PointRef::new(id, 0), None)
        .unwrap();
    state.unfocus_bench_item();
    for (position, tab) in [(0, Tab::MeasurementView), (1, Tab::TrackView)] {
        docked_behind_the_scene_tree(&mut state, tab);
        state.edit_bench_item_at(id, position);
        assert!(state.panel_is_in_front(Tab::SceneGraph), "{tab:?}");
        assert!(!state.panel_is_in_front(tab), "{tab:?}");
        assert!(state.focused_item().is_some());
    }
    // From a tree with nothing else in its node, it is a raise.
    state.hide_panel(Tab::MeasurementView);
    state.show_panel(Tab::MeasurementView);
    state.unfocus_bench_item();
    state.edit_bench_item_at(id, 0);
    assert!(state.panel_is_in_front(Tab::MeasurementView));
}

/// While a measurement item is focused, an undo or a redo that brings its
/// origin's row back selects that row again: after an undo of the delete of
/// its row, and after a redo of a commit that appended it.
#[test]
fn undo_and_redo_reselect_a_focused_item_s_origin() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 3))
        .unwrap();
    state
        .delete_measurement(MeasurementRef::distance(id, 3))
        .unwrap();
    assert_eq!(state.selected_measurement, None);
    state.undo(id).unwrap();
    assert_eq!(state.focused_measurement_label(id), Some("distance #3"));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 3))
    );

    // The delete again, then a commit that appends the row as row 4.
    state.redo(id).unwrap();
    assert_eq!(state.selected_measurement, None);
    let committed = state.commit_bench_measurement(id, "distance #3").unwrap();
    assert_eq!(committed.measurement, MeasurementRef::distance(id, 4));
    state.undo(id).unwrap();
    // The version before the commit holds four rows, so nothing is selected.
    assert_eq!(state.selected_measurement, None);
    state.redo(id).unwrap();
    assert_eq!(state.focused_measurement_label(id), Some("distance #3"));
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 4))
    );
}

/// A selection the person cleared while a measurement item stays focused
/// stays cleared across an undo: the move reselects the origin only where it
/// was the selection before.
#[test]
fn a_cleared_selection_stays_cleared_across_an_undo() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 3))
        .unwrap();
    state
        .delete_measurement(MeasurementRef::distance(id, 0))
        .unwrap();
    assert_eq!(
        state.selected_measurement,
        Some(MeasurementRef::distance(id, 2))
    );
    state.clear_selection();
    assert_eq!(state.focused_measurement_label(id), Some("distance #3"));
    state.undo(id).unwrap();
    assert_eq!(state.selected_measurement, None);
    assert_eq!(state.focused_measurement_label(id), Some("distance #3"));
}

// ---- Setting the label and the stated distance ----------------------------

fn distance_of(state: &AppState, id: ReconId, label: &str) -> Option<(f64, LengthUnit)> {
    item(state, id, label).as_distance().unwrap().distance
}

/// The newest Action Log row's text, and whether it is a failure.
fn last_row(state: &AppState) -> (String, bool) {
    let entry = state.action_log.entries().last().expect("a row");
    (entry.text.clone(), entry.failed)
}

/// The fixture on a `sift_files` node, whose bench is view-only.
fn view_only_state() -> (AppState, ReconId) {
    let mut recon = measured_reconstruction(Some("m"));
    let count = recon.point_set.tracks.len();
    recon.point_set.observations = sfmtool_core::ObservationSource::SiftFiles {
        feature_indexes: vec![0; count],
        feature_tool_hashes: vec![[0u8; 16]; 3],
        sift_content_hashes: vec![[0u8; 16]; 3],
    };
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        recon,
    ));
    (state, id)
}

/// Setting the label is one bench step with its label in the Edit History,
/// leaves the reconstruction and the stored row alone, keeps the item's ID
/// and focus, and is undone by an undo.
#[test]
fn setting_the_label_is_one_bench_step_and_undoes() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 2))
        .unwrap();
    let before = version_count(&state, id);
    let document = state
        .node(id)
        .unwrap()
        .history
        .current_version()
        .document_serial;
    let item_id = state.focused_item().unwrap().item;
    assert!(state
        .set_bench_measurement_label(id, "distance #2", "paw width")
        .unwrap());
    assert_eq!(version_count(&state, id), before + 1);
    let node = state.node(id).unwrap();
    assert_eq!(
        node.history.current_version().label,
        "Set the label of distance #2 to \"paw width\""
    );
    assert_eq!(node.history.current_version().document_serial, document);
    assert!(!node.history.is_dirty());
    assert_eq!(item(&state, id, "distance #2").label, "paw width");
    assert_eq!(state.focused_item().unwrap().item, item_id);
    let rows = crate::measurements::distance_rows(state.node(id).unwrap().edited().measurements());
    assert_eq!(rows[2].label, "", "the stored row changed");
    assert!(last_row(&state)
        .0
        .starts_with("Set the label of distance #2 to \"paw width\""));
    state.undo(id).unwrap();
    assert_eq!(item(&state, id, "distance #2").label, "");
}

/// The label an item holds already pushes no version and writes the
/// no-effect row.
#[test]
fn setting_the_label_it_has_is_no_step() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let before = version_count(&state, id);
    assert!(!state
        .set_bench_measurement_label(id, "distance #0", "ear to ear")
        .unwrap());
    assert_eq!(version_count(&state, id), before);
    assert_eq!(
        last_row(&state),
        (
            "Set the label of distance #0 to \"ear to ear\": no effect, it has that label \
             already"
                .to_string(),
            false
        )
    );
}

/// The stated distance is read as the CLI reads one: a suffix names the
/// unit, a bare number takes the reconstruction's unit, and the step's label
/// names the distance it set. Undo and redo move between them, and the
/// commit writes the one at the cursor.
#[test]
fn setting_the_stated_distance_reads_the_unit_and_is_one_bench_step() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 2))
        .unwrap();
    let before = version_count(&state, id);
    let set = state
        .set_bench_measurement_distance(id, "distance #2", "4.2 cm")
        .unwrap();
    assert!(set.changed);
    assert_eq!(set.distance, (4.2, LengthUnit::Centimetre));
    assert_eq!(
        distance_of(&state, id, "distance #2"),
        Some((4.2, LengthUnit::Centimetre))
    );
    assert_eq!(version_count(&state, id), before + 1);
    assert_eq!(
        state.node(id).unwrap().history.current_version().label,
        "Set the stated distance of distance #2 to 4.2 cm"
    );
    assert_eq!(stated_of(&state, id, 2), 1.0, "the stored row changed");
    // A bare number is in the file's unit, metres here.
    let set = state
        .set_bench_measurement_distance(id, "distance #2", "0.25")
        .unwrap();
    assert_eq!(set.distance, (0.25, LengthUnit::Metre));
    state.undo(id).unwrap();
    assert_eq!(
        distance_of(&state, id, "distance #2"),
        Some((4.2, LengthUnit::Centimetre))
    );
    state.undo(id).unwrap();
    assert_eq!(
        distance_of(&state, id, "distance #2"),
        Some((1.0, LengthUnit::Metre))
    );
    state.redo(id).unwrap();
    let committed = state.commit_bench_measurement(id, "distance #2").unwrap();
    assert!(committed.changed);
    let rows = crate::measurements::distance_rows(state.node(id).unwrap().edited().measurements());
    assert_eq!(
        (rows[2].stated as f32, rows[2].unit),
        (4.2, LengthUnit::Centimetre)
    );
}

/// With no `world_space_unit` a bare number is in metres, and with another
/// unit it is in that one. A distance the item states already, as the file
/// holds it, is no step.
#[test]
fn a_bare_number_takes_the_file_s_unit_or_metres() {
    let (mut state, id) = measured_state(None);
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 2))
        .unwrap();
    let set = state
        .set_bench_measurement_distance(id, "distance #2", "3")
        .unwrap();
    assert_eq!(set.distance, (3.0, LengthUnit::Metre));
    let (mut state, id) = measured_state(Some("ft"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 2))
        .unwrap();
    let set = state
        .set_bench_measurement_distance(id, "distance #2", "3")
        .unwrap();
    assert_eq!(set.distance, (3.0, LengthUnit::Foot));
    let before = version_count(&state, id);
    let again = state
        .set_bench_measurement_distance(id, "distance #2", "3.0ft")
        .unwrap();
    assert!(!again.changed);
    assert_eq!(version_count(&state, id), before);
    assert_eq!(
        last_row(&state).0,
        "Set the stated distance of distance #2 to 3 ft: no effect, it states that already"
    );
}

/// Text that does not read, an unknown unit and a value that is not finite
/// and positive are each refused in core's sentence, and push nothing.
#[test]
fn a_stated_distance_that_does_not_read_is_refused_in_core_s_words() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let before = version_count(&state, id);
    assert_eq!(
        state
            .set_bench_measurement_distance(id, "distance #0", "-2cm")
            .map(|_| ()),
        Err(
            "Cannot set the stated distance of distance #0: the stated distance is -2, which \
             must be finite and strictly positive."
                .to_string()
        )
    );
    assert_eq!(
        state
            .set_bench_measurement_distance(id, "distance #0", "12yd")
            .map(|_| ()),
        Err(
            "Cannot set the stated distance of distance #0: \"yd\" is not a unit: the units \
             are mm, cm, m, in, ft."
                .to_string()
        )
    );
    assert!(state
        .set_bench_measurement_distance(id, "distance #0", "twelve")
        .unwrap_err()
        .contains("\"twelve\" is not a distance"));
    assert_eq!(version_count(&state, id), before);
    assert_eq!(
        distance_of(&state, id, "distance #0"),
        Some((153.0, LengthUnit::Centimetre))
    );
}

/// Both steps are refused while a background task holds the node and on a
/// view-only bench, in the shared sentence, and for a track named as the
/// item.
#[test]
fn both_steps_are_refused_while_busy_on_a_view_only_bench_and_for_a_track() {
    use crate::background::{Finished, Operation};
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    state
        .start_background_task(
            Operation::BENCH_FIT,
            id,
            Box::new(move |_| Finished::Cancelled),
        )
        .expect("nothing else is running");
    let busy = state.busy_refusal(id).expect("the task holds the node");
    let before = version_count(&state, id);
    assert_eq!(
        state.set_bench_measurement_label(id, "distance #0", "x"),
        Err(busy.clone())
    );
    assert_eq!(
        state
            .set_bench_measurement_distance(id, "distance #0", "1m")
            .map(|_| ()),
        Err(busy)
    );
    assert_eq!(version_count(&state, id), before);

    let (mut state, id) = view_only_state();
    let sentence = state.bench_view_only_refusal(id).expect("view-only");
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    assert_eq!(
        state.set_bench_measurement_label(id, "distance #0", "x"),
        Err(sentence.clone())
    );
    assert_eq!(
        state
            .set_bench_measurement_distance(id, "distance #0", "1m")
            .map(|_| ()),
        Err(sentence)
    );
    assert_eq!(item(&state, id, "distance #0").label, "ear to ear");

    let (mut state, id) = measured_state(Some("m"));
    state
        .put_point_on_bench(PointRef::new(id, 0), None)
        .unwrap();
    let track = state.focused_track_label(id).unwrap().to_string();
    assert_eq!(
        state.set_bench_measurement_label(id, &track, "x"),
        Err(super::not_a_measurement(&track))
    );
}

// ---- Moving an endpoint ---------------------------------------------------

fn endpoints_of(
    state: &AppState,
    id: ReconId,
    label: &str,
) -> [Option<sfmtool_core::bench::MeasurementEndpoint>; 2] {
    item(state, id, label).as_distance().unwrap().endpoints
}

/// The offers of endpoint A of `distance #1`, on point 1 in image 0: its
/// observation in image 1, at its keypoint.
fn offers_of_a(
    state: &AppState,
    id: ReconId,
) -> Vec<sfmtool_core::reconstruction::measurement_endpoint::EndpointView> {
    let node = state.node(id).unwrap();
    let a = endpoints_of(state, id, "distance #1")[0].unwrap();
    sfmtool_core::reconstruction::measurement_endpoint::endpoint_views(
        node.edited(),
        a.image,
        a.pixel,
        &Default::default(),
    )
    .unwrap()
}

/// Switching an endpoint to an offered view is one bench step, labelled by
/// the two images, which moves the endpoint, leaves the stored row and the
/// dirty flag alone, keeps the endpoint on the same point and texel, and is
/// undone and redone. The same switch again is no step.
#[test]
fn switching_an_endpoint_is_one_bench_step_and_undoes() {
    use crate::bench::measurement::EndpointMove;
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    let before = version_count(&state, id);
    let was = endpoints_of(&state, id, "distance #1");
    let offers = offers_of_a(&state, id);
    assert_eq!(offers.len(), 1);
    let view = &offers[0];
    assert_eq!((view.point, view.image), (1, 1));

    let moved = state
        .set_bench_measurement_endpoint(
            id,
            "distance #1",
            EndpointSide::A,
            view.image,
            view.pixel,
            EndpointMove::Switch,
        )
        .unwrap();
    assert!(moved);
    assert_eq!(version_count(&state, id), before + 1);
    let node = state.node(id).unwrap();
    assert_eq!(
        node.history.current_version().label,
        "Moved endpoint A of distance #1 from image_000.jpg to image_001.jpg"
    );
    assert!(!node.is_dirty());
    let now = endpoints_of(&state, id, "distance #1");
    assert_eq!(now[0].map(|e| (e.image, e.pixel)), Some((1, view.pixel)));
    assert_eq!(now[1], was[1]);
    let rows = crate::measurements::distance_rows(node.edited().measurements());
    assert_eq!(rows[1].images, [0, 1], "the stored row changed");
    let reading =
        crate::measurements::item_reading(node.edited(), &item(&state, id, "distance #1"));
    let a = reading.endpoints[0].as_ref().unwrap().as_ref().unwrap();
    assert_eq!(a.chosen().point, 1);
    for c in a.chosen().patch_coords {
        assert!(c.abs() < 1e-4, "the texel moved: {c}");
    }
    assert_eq!(state.focused_measurement_label(id), Some("distance #1"));

    state.undo(id).unwrap();
    assert_eq!(endpoints_of(&state, id, "distance #1"), was);
    state.redo(id).unwrap();
    assert_eq!(endpoints_of(&state, id, "distance #1")[0].unwrap().image, 1);

    let again = state
        .set_bench_measurement_endpoint(
            id,
            "distance #1",
            EndpointSide::A,
            view.image,
            view.pixel,
            EndpointMove::Switch,
        )
        .unwrap();
    assert!(!again);
    assert_eq!(version_count(&state, id), before + 1);
    assert_eq!(
        last_row(&state),
        (
            "Moved endpoint A of distance #1 from image_001.jpg to image_001.jpg: no effect, \
             the endpoint is there already"
                .to_string(),
            false
        )
    );
}

/// Setting an endpoint outright is labelled by the pixel, and placing one not
/// placed says so; a switch of an endpoint not placed, and a pixel outside its
/// image, are refused and push nothing.
#[test]
fn setting_an_endpoint_names_the_pixel_and_refuses_one_outside_the_image() {
    use crate::bench::measurement::EndpointMove;
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    assert!(state
        .set_bench_measurement_endpoint(
            id,
            "distance #1",
            EndpointSide::A,
            2,
            [10.0, 20.25],
            EndpointMove::Set,
        )
        .unwrap());
    assert_eq!(
        state.node(id).unwrap().history.current_version().label,
        "Moved endpoint A of distance #1 to image_002.jpg (10.0, 20.2)"
    );
    edit_item(&mut state, id, "distance #1", |item| {
        let EditableMeasurementBody::Distance(d) = &mut item.body;
        d.endpoints[1] = None;
    });
    let before = version_count(&state, id);
    assert_eq!(
        state.set_bench_measurement_endpoint(
            id,
            "distance #1",
            EndpointSide::B,
            0,
            [1.0, 1.0],
            EndpointMove::Switch,
        ),
        Err(
            "Cannot switch endpoint B of distance #1: it is not placed, so it has no image to \
             switch from."
                .to_string()
        )
    );
    assert_eq!(
        state.set_bench_measurement_endpoint(
            id,
            "distance #1",
            EndpointSide::A,
            0,
            [640.0, 3.0],
            EndpointMove::Set,
        ),
        Err(
            "Cannot move endpoint A of distance #1: endpoint A's pixel (640, 3) is outside \
             image 0, which is [0, 640) x [0, 480)."
                .to_string()
        )
    );
    assert_eq!(version_count(&state, id), before);
    assert!(state
        .set_bench_measurement_endpoint(
            id,
            "distance #1",
            EndpointSide::B,
            0,
            [1.0, 1.0],
            EndpointMove::Set,
        )
        .unwrap());
    assert_eq!(
        state.node(id).unwrap().history.current_version().label,
        "Set endpoint B of distance #1 to image_000.jpg (1.0, 1.0)"
    );
}

/// Moving an endpoint is refused while a background task holds the node and
/// on a view-only bench, in the shared sentence.
#[test]
fn moving_an_endpoint_is_refused_while_busy_and_on_a_view_only_bench() {
    use crate::background::{Finished, Operation};
    use crate::bench::measurement::EndpointMove;
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    let view = offers_of_a(&state, id).remove(0);
    state
        .start_background_task(
            Operation::BENCH_FIT,
            id,
            Box::new(move |_| Finished::Cancelled),
        )
        .expect("nothing else is running");
    let busy = state.busy_refusal(id).expect("the task holds the node");
    let switch = |state: &mut AppState, id| {
        state.set_bench_measurement_endpoint(
            id,
            "distance #1",
            EndpointSide::A,
            view.image,
            view.pixel,
            EndpointMove::Switch,
        )
    };
    assert_eq!(switch(&mut state, id), Err(busy));

    let (mut state, id) = view_only_state();
    let sentence = state.bench_view_only_refusal(id).expect("view-only");
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    assert_eq!(switch(&mut state, id), Err(sentence));
    assert_eq!(endpoints_of(&state, id, "distance #1")[0].unwrap().image, 0);
}

// ---- Setting an endpoint from a menu --------------------------------------

/// The keypoint of point `point`'s observation `k` in the fixture, as the
/// pixel an endpoint is set to.
fn keypoint_of(state: &AppState, id: ReconId, point: u32, k: usize) -> [f64; 2] {
    let kp = state
        .node(id)
        .unwrap()
        .edited()
        .point(point)
        .unwrap()
        .keypoint_xy(k)
        .unwrap();
    [f64::from(kp[0]), f64::from(kp[1])]
}

/// The 3D view's point menu: a right-click selects the point and keeps the
/// measurement item focused, *Set Endpoint B Here* moves B to the point's
/// reference pixel in one bench step, *Set Endpoint A Here* places an
/// endpoint that is not placed, and a point with no reference pixel is one
/// failed row in the entry's sentence, with no step.
#[test]
fn the_point_menu_sets_an_endpoint_to_the_point_s_reference_pixel() {
    use crate::measurements::fixture::referenced_state;
    use crate::state::edits::PointGesture;
    let (mut state, id) = referenced_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let from = crate::dock::Tab::Viewer3D;
    state.apply_point_gesture(PointGesture::Opened(PointRef::new(id, 2)), from);
    assert_eq!(state.selected_point, Some(PointRef::new(id, 2)));
    assert_eq!(state.focused_measurement_label(id), Some("distance #0"));

    let before = version_count(&state, id);
    state.apply_point_gesture(
        PointGesture::SetEndpointHere(PointRef::new(id, 0), EndpointSide::B),
        from,
    );
    assert_eq!(version_count(&state, id), before + 1);
    let b = endpoints_of(&state, id, "distance #0")[1].unwrap();
    // Point 0's reference is its observation in image 1.
    assert_eq!((b.image, b.pixel), (1, keypoint_of(&state, id, 0, 1)));
    let label = state
        .node(id)
        .unwrap()
        .history
        .current_version()
        .label
        .clone();
    assert!(
        label.starts_with("Moved endpoint B of distance #0 to image_001.jpg ("),
        "{label}"
    );
    assert_eq!(state.selected_point, Some(PointRef::new(id, 0)));
    assert_eq!(state.focused_measurement_label(id), Some("distance #0"));
    assert!(
        !state.node(id).unwrap().history.is_dirty(),
        "a bench step dirtied the node"
    );

    edit_item(&mut state, id, "distance #0", |item| {
        let EditableMeasurementBody::Distance(d) = &mut item.body;
        d.endpoints[0] = None;
    });
    state.apply_point_gesture(
        PointGesture::SetEndpointHere(PointRef::new(id, 2), EndpointSide::A),
        from,
    );
    let a = endpoints_of(&state, id, "distance #0")[0].unwrap();
    assert_eq!((a.image, a.pixel), (0, keypoint_of(&state, id, 2, 0)));
    let label = state
        .node(id)
        .unwrap()
        .history
        .current_version()
        .label
        .clone();
    assert!(
        label.starts_with("Set endpoint A of distance #0 to image_000.jpg ("),
        "{label}"
    );

    let before = version_count(&state, id);
    state.apply_point_gesture(
        PointGesture::SetEndpointHere(PointRef::new(id, 1), EndpointSide::A),
        from,
    );
    assert_eq!(version_count(&state, id), before);
    assert_eq!(
        last_row(&state),
        (
            crate::measurements::NO_REFERENCE_FOR_PIXEL.to_string(),
            true
        )
    );
}

/// What the two menus are told: nothing with no measurement item focused;
/// the item and its node while one is; and the refusals the entries are
/// greyed with, a node other than the item's naming the item's node, a busy
/// node and a view-only bench.
#[test]
fn the_endpoint_target_is_the_focused_measurement_with_its_refusals() {
    use crate::background::{Finished, Operation};
    use crate::measurements::fixture::referenced_reconstruction;
    let (mut state, id) = measured_state(Some("m"));
    assert_eq!(state.endpoint_target(), None);
    let other = state.append_node(SceneNode::from_path(
        std::path::Path::new("other.sfmr"),
        referenced_reconstruction(Some("m")),
    ));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let target = state.endpoint_target().expect("a measurement is focused");
    assert_eq!((target.node, target.item.as_str()), (id, "distance #0"));
    assert_eq!(target.refusal_on(id), None);
    let label = state.node(id).unwrap().label.clone();
    assert_eq!(
        target.refusal_on(other),
        Some(format!(
            "The measurement being edited, distance #0, is on {label}, so its endpoints are \
             set from that reconstruction's images and points."
        ))
    );
    // A point of the other node is refused for its node before its pixel is
    // asked about, though it has one.
    assert_eq!(
        target.at_point(&state.scene, PointRef::new(other, 0)),
        Err(target.refusal_on(other).unwrap())
    );
    state
        .start_background_task(
            Operation::BENCH_FIT,
            id,
            Box::new(move |_| Finished::Cancelled),
        )
        .expect("nothing else is running");
    let busy = state.busy_refusal(id).expect("the task holds the node");
    assert_eq!(state.endpoint_target().unwrap().refusal_on(id), Some(busy));

    let (mut state, id) = view_only_state();
    let sentence = state.bench_view_only_refusal(id).expect("view-only");
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    assert_eq!(
        state.endpoint_target().unwrap().refusal_on(id),
        Some(sentence)
    );
    // A track focused is not a measurement, and offers the entries nothing.
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_point_on_bench(PointRef::new(id, 0), None)
        .unwrap();
    assert_eq!(state.endpoint_target(), None);
}
