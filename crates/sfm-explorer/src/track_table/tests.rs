// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The track table cache's keying, and *Find points* filling it.

use sfmtool_core::analysis::track_table::{
    ColumnBounds, SortOrder, TrackTableColumn, TrackTableQueryOptions,
};

use crate::background::{Operation, TaskState};
use crate::scene::{PointRef, ReconId};
use crate::state::AppState;

/// A scene holding one node of 120 points, selected.
fn one_node() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    state.select_recon(id);
    (state, id)
}

/// Run *Find points* for `columns` on `id` to its end and answer with its id.
fn find(
    state: &mut AppState,
    id: ReconId,
    columns: &[TrackTableColumn],
    options: TrackTableQueryOptions,
) -> u64 {
    let task = state
        .start_find_points(id, columns.to_vec(), options)
        .expect("nothing refuses a read");
    state.finish_background_tasks();
    task
}

fn finished(state: &AppState, task: u64) -> &crate::background::FinishedTask {
    match state.task(task) {
        Some(TaskState::Finished(finished)) => finished,
        _ => panic!("task {task} has not finished"),
    }
}

#[test]
fn find_points_fills_the_cache_for_the_version_at_the_cursor() {
    let (mut state, id) = one_node();
    assert!(state.track_table(id).is_none());
    let task = find(
        &mut state,
        id,
        &[TrackTableColumn::TrackLength],
        TrackTableQueryOptions::default(),
    );
    let table = state.track_table(id).expect("cached");
    assert!(table.has(TrackTableColumn::TrackLength));
    assert!(!table.has(TrackTableColumn::MaxRayAngleDeg));
    let live = table.point_indexes().len();

    let record = finished(&state, task);
    assert_eq!(record.operation, Operation::FIND_POINTS);
    assert!(!record.changed, "a read pushes no version");
    let found = record.found_points.as_ref().expect("found");
    assert_eq!(found.matched.len(), live);
    assert_eq!(
        record.outcome.as_deref(),
        Ok(format!("Found {live} of {live} points in run_a").as_str())
    );
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

#[test]
fn a_second_run_merges_its_columns_beside_the_first() {
    let (mut state, id) = one_node();
    find(
        &mut state,
        id,
        &[TrackTableColumn::TrackLength],
        TrackTableQueryOptions::default(),
    );
    let kept = state
        .track_table(id)
        .unwrap()
        .values(TrackTableColumn::TrackLength)
        .unwrap()
        .as_ptr();
    let task = find(
        &mut state,
        id,
        &[TrackTableColumn::MaxRayAngleDeg],
        TrackTableQueryOptions {
            bounds: vec![ColumnBounds {
                column: TrackTableColumn::TrackLength,
                min: Some(3.0),
                max: None,
            }],
            order_by: Some((TrackTableColumn::MaxRayAngleDeg, SortOrder::Descending)),
            ..Default::default()
        },
    );
    let table = state.track_table(id).unwrap();
    assert!(table.has(TrackTableColumn::MaxRayAngleDeg));
    assert_eq!(
        table
            .values(TrackTableColumn::TrackLength)
            .unwrap()
            .as_ptr(),
        kept,
        "the column already held is kept, not replaced"
    );
    let found = finished(&state, task).found_points.as_ref().unwrap();
    let lengths = table.values(TrackTableColumn::TrackLength).unwrap();
    let row = |index: u32| table.point_indexes().binary_search(&index).unwrap();
    assert!(found.matched.iter().all(|&p| lengths[row(p)] >= 3.0));
}

#[test]
fn an_edit_moves_the_key_and_a_new_table_replaces_the_old() {
    let (mut state, id) = one_node();
    find(
        &mut state,
        id,
        &[TrackTableColumn::TrackLength],
        TrackTableQueryOptions::default(),
    );
    let before = state.track_table(id).unwrap().point_indexes().len();
    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    assert!(
        state.track_table(id).is_none(),
        "the table of the version before the edit is not the one at the cursor"
    );
    find(
        &mut state,
        id,
        &[TrackTableColumn::RmsErrorPx],
        TrackTableQueryOptions::default(),
    );
    let table = state.track_table(id).unwrap();
    assert_eq!(table.point_indexes().len(), before - 1);
    assert!(
        !table.has(TrackTableColumn::TrackLength),
        "the old version's columns are not carried over"
    );
    assert_eq!(state.track_tables.entries.len(), 1, "one entry per node");
}

#[test]
fn a_run_that_lands_after_an_edit_is_dropped_with_a_failed_row() {
    let (mut state, id) = one_node();
    // The fast queue held by a fake task, so *Find points* waits; the edit
    // then moves the serial before it runs.
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_task(
            Operation::OPEN,
            None,
            "held".to_string(),
            Box::new(move |_| {
                let _ = held.recv();
                crate::background::Finished::Failed("held".to_string())
            }),
        )
        .unwrap();
    let task = state
        .start_find_points(
            id,
            vec![TrackTableColumn::TrackLength],
            TrackTableQueryOptions::default(),
        )
        .unwrap();
    assert!(matches!(state.task(task), Some(TaskState::Waiting(_))));
    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("a read locks nothing");
    open.send(()).unwrap();
    state.finish_background_tasks();
    let record = finished(&state, task);
    let message = record.outcome.as_ref().unwrap_err();
    assert!(
        message.contains("has changed since it started"),
        "{message}"
    );
    assert!(record.found_points.is_none());
    assert!(state.track_tables.entries.is_empty());
}

#[test]
fn closing_the_node_drops_its_table() {
    let (mut state, id) = one_node();
    find(
        &mut state,
        id,
        &[TrackTableColumn::TrackLength],
        TrackTableQueryOptions::default(),
    );
    assert!(state.track_table(id).is_some());
    state.close_node(id).expect("nothing locks it");
    assert!(state.track_tables.entries.is_empty());
}

#[test]
fn a_cancelled_run_caches_nothing() {
    let (mut state, id) = one_node();
    let task = state
        .start_find_points(
            id,
            vec![TrackTableColumn::MaxRayAngleDeg],
            TrackTableQueryOptions::default(),
        )
        .unwrap();
    state.cancel_background_task(task);
    state.finish_background_tasks();
    let record = finished(&state, task);
    // A run that met the flag says so; one that finished before it read the
    // flag landed whole. Either way the cache describes the version at the
    // cursor.
    if record.cancelled {
        assert_eq!(
            record.outcome.as_ref().unwrap_err(),
            "Find points of run_a cancelled"
        );
        assert!(state.track_table(id).is_none());
    } else {
        assert!(state.track_table(id).is_some());
    }
}
