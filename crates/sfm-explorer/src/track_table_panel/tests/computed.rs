// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The computed columns in the Track Table panel: the ray angle, the condition
//! number and the inverse-depth z, which the table job computes as a second
//! run after the stored columns, and only once a visible column, a bound or
//! the sort needs them.

use std::sync::mpsc;

use sfmtool_core::analysis::track_table::{
    query_points, ColumnBounds, ColumnCost, SortOrder, TrackTableColumn, TrackTableQueryOptions,
};

use super::{frame, held, input, state, BEARING, POINTS};
use crate::scene::{PointRef, ReconId};
use crate::state::AppState;
use crate::table_sort::{SortRule, TableSort};
use crate::test_support::painted_texts;
use crate::track_table_panel::{
    cell_text, compact, sort_hover, worst_first_ascending, TrackTablePanel,
};

/// The ellipsis a cell of a column not computed yet shows.
const PENDING: &str = "\u{2026}";

/// The computed columns the cache holds for `id` at its cursor.
fn computed_held(state: &AppState, id: ReconId) -> Vec<TrackTableColumn> {
    held(state, id)
        .into_iter()
        .filter(|c| c.cost() == ColumnCost::Computed)
        .collect()
}

/// A panel with the stored columns in the cache and its rows on screen, and
/// no computed column computed: the first run has landed and the second is
/// not queued yet.
fn stored_only(state: &mut AppState) -> (egui::Context, TrackTablePanel) {
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    frame(&ctx, &mut panel, state);
    state.drive_track_table_columns();
    state.wait_for_track_table_job();
    (ctx, panel)
}

/// The texts one frame of the panel paints.
fn texts(ctx: &egui::Context, panel: &mut TrackTablePanel, state: &mut AppState) -> Vec<String> {
    painted_texts(ctx, input(1200.0), |ui| panel.show(ui, state))
}

/// With no computed column shown, bounded or sorted by, none is computed: the
/// job stops after the stored columns. Ticking one queues a second run that
/// computes it alone and merges it beside the stored ones, whose arrays are
/// kept.
#[test]
fn a_computed_column_is_computed_only_once_something_needs_it() {
    let (mut state, id) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    panel
        .query
        .set_shown(TrackTableColumn::MaxRayAngleDeg, false);
    frame(&ctx, &mut panel, &mut state);
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    assert!(computed_held(&state, id).is_empty());
    assert!(!state.drive_track_table_columns(), "nothing more is queued");
    assert_eq!(state.track_table_jobs.queued_task(), None);

    let stored = state
        .track_table(id)
        .unwrap()
        .values(TrackTableColumn::TrackLength)
        .unwrap()
        .as_ptr();
    panel
        .query
        .set_shown(TrackTableColumn::ConditionNumber, true);
    frame(&ctx, &mut panel, &mut state);
    assert!(state.drive_track_table_columns());
    assert_eq!(
        state.track_table_jobs.queued_columns(),
        Some([TrackTableColumn::ConditionNumber].as_slice())
    );
    state.wait_for_track_table_job();
    assert_eq!(
        computed_held(&state, id),
        [TrackTableColumn::ConditionNumber]
    );
    let table = state.track_table(id).unwrap();
    assert_eq!(
        table
            .values(TrackTableColumn::TrackLength)
            .unwrap()
            .as_ptr(),
        stored,
        "the stored arrays are kept"
    );
}

/// The default columns include Ray angle: the first run computes the stored
/// columns and the second the ray angle alone.
#[test]
fn the_stored_columns_land_first_and_the_ray_angle_after() {
    let (mut state, id) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    frame(&ctx, &mut panel, &mut state);
    state.drive_track_table_columns();
    let first = state.track_table_jobs.queued_columns().unwrap().to_vec();
    assert!(first.iter().all(|c| c.cost() == ColumnCost::Stored));
    state.wait_for_track_table_job();
    assert!(computed_held(&state, id).is_empty());

    assert!(state.drive_track_table_columns(), "the second run");
    assert_eq!(
        state.track_table_jobs.queued_columns(),
        Some([TrackTableColumn::MaxRayAngleDeg].as_slice())
    );
    state.wait_for_track_table_job();
    assert_eq!(
        computed_held(&state, id),
        [TrackTableColumn::MaxRayAngleDeg]
    );
}

/// While the second run computes the ray angle, the rows are on screen with
/// the stored values, and every Ray angle cell shows `…`; once it lands they
/// show the values.
#[test]
fn a_column_not_computed_yet_shows_an_ellipsis_in_every_cell() {
    let (mut state, id) = state();
    let (ctx, mut panel) = stored_only(&mut state);
    let (open, hold) = mpsc::channel();
    state.track_table_jobs.hold_next = Some(hold);
    frame(&ctx, &mut panel, &mut state);
    state.drive_track_table_columns();
    let task = state.track_table_jobs.queued_task().expect("queued");
    assert!(state.quiet_task_running(task));

    let pending = texts(&ctx, &mut panel, &mut state);
    assert_eq!(panel.drawn().len(), POINTS, "the rows are usable meanwhile");
    assert_eq!(
        pending.iter().filter(|t| *t == PENDING).count(),
        POINTS,
        "{pending:?}"
    );
    assert!(pending.iter().any(|t| t == "#5"), "{pending:?}");

    open.send(()).expect("the job waits at its gate");
    state.wait_for_track_table_job();
    let landed = texts(&ctx, &mut panel, &mut state);
    assert!(!landed.iter().any(|t| t == PENDING), "{landed:?}");
    let table = state.track_table(id).unwrap();
    let row = table.point_indexes().binary_search(&5).unwrap();
    let angle = cell_text(table, TrackTableColumn::MaxRayAngleDeg, row);
    assert!(landed.contains(&angle), "{angle} in {landed:?}");
}

/// A sort or a bound on a column not computed yet leaves the rows as they
/// are and says what waits for which column, a change of a stored bound
/// waits with it, and the whole query applies once the columns land.
#[test]
fn a_sort_or_bound_on_a_column_not_computed_yet_leaves_the_rows_as_they_are() {
    let (mut state, id) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    panel
        .query
        .set_shown(TrackTableColumn::MaxRayAngleDeg, false);
    frame(&ctx, &mut panel, &mut state);
    state.settle_track_table_columns();
    panel.query.click_heading(TrackTableColumn::RmsErrorPx);
    frame(&ctx, &mut panel, &mut state);
    let by_error: Vec<u32> = (0..POINTS as u32).rev().collect();
    assert_eq!(panel.rows(), by_error.as_slice());

    // Sorted by the ray angle, which the cache does not hold.
    let (open, hold) = mpsc::channel();
    state.track_table_jobs.hold_next = Some(hold);
    panel
        .query
        .set_shown(TrackTableColumn::MaxRayAngleDeg, true);
    panel.query.click_heading(TrackTableColumn::MaxRayAngleDeg);
    frame(&ctx, &mut panel, &mut state);
    state.drive_track_table_columns();
    let painted = texts(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows(), by_error.as_slice(), "the order is left");
    assert!(
        painted
            .iter()
            .any(|t| t == "Sorting when Ray angle (\u{b0}) is ready."),
        "{painted:?}"
    );

    // A bound on the condition number waits too, and so does a bound on a
    // stored column set meanwhile.
    panel
        .query
        .set_shown(TrackTableColumn::ConditionNumber, true);
    panel.query.bound_mut(TrackTableColumn::ConditionNumber).min = "1".into();
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "2".into();
    frame(&ctx, &mut panel, &mut state);
    let painted = texts(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows(), by_error.as_slice());
    assert!(
        painted
            .iter()
            .any(|t| t == "Filtering and sorting when Ray angle (\u{b0}), Cond. are ready."),
        "{painted:?}"
    );

    // The ray angle lands; the bound still waits for the condition number.
    open.send(()).expect("the job waits at its gate");
    state.wait_for_track_table_job();
    let painted = texts(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows(), by_error.as_slice());
    assert!(
        painted
            .iter()
            .any(|t| t == "Filtering when Cond. is ready."),
        "{painted:?}"
    );

    // Both landed: the whole query applies.
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let painted = texts(&ctx, &mut panel, &mut state);
    assert!(
        !painted.iter().any(|t| t.ends_with(" ready.")),
        "{painted:?}"
    );
    let edited = state.scene[0].edited().clone();
    let expected = query_points(
        &edited,
        state.track_table(id).unwrap(),
        &TrackTableQueryOptions {
            bounds: vec![
                ColumnBounds {
                    column: TrackTableColumn::RmsErrorPx,
                    min: None,
                    max: Some(2.0),
                },
                ColumnBounds {
                    column: TrackTableColumn::ConditionNumber,
                    min: Some(1.0),
                    max: None,
                },
            ],
            seen_in_image: None,
            order_by: Some((TrackTableColumn::MaxRayAngleDeg, SortOrder::Ascending)),
        },
    )
    .unwrap();
    assert!(!expected.is_empty() && expected.len() < POINTS);
    assert_eq!(panel.rows(), expected.as_slice());
}

/// The first time a table lands, with nothing listed yet for its version, a
/// sort on a column not computed yet is said and the rest of the query
/// applies.
#[test]
fn a_first_table_lists_its_rows_by_the_rest_of_the_query() {
    let (mut state, _) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    panel.query.click_heading(TrackTableColumn::MaxRayAngleDeg);
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "0.95".into();
    frame(&ctx, &mut panel, &mut state);
    state.drive_track_table_columns();
    state.wait_for_track_table_job();
    let painted = texts(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows(), (0..=9).collect::<Vec<u32>>().as_slice());
    assert!(
        painted
            .iter()
            .any(|t| t == "Sorting when Ray angle (\u{b0}) is ready."),
        "{painted:?}"
    );
}

/// An edit while the second run computes asks it to stop and merges nothing
/// for the serial the node left. Meanwhile the rows of the version before keep
/// the computed values they had, the next job computes the new version's
/// stored columns before any computed one, and the computed ones follow.
#[test]
fn an_edit_cancels_the_second_run_and_the_old_rows_keep_their_values() {
    let (mut state, id) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    frame(&ctx, &mut panel, &mut state);
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let old = state.scene[0].history.current_version().document_serial;
    let old_angle = {
        let table = state.track_table(id).unwrap();
        let row = table.point_indexes().binary_search(&5).unwrap();
        cell_text(table, TrackTableColumn::MaxRayAngleDeg, row)
    };

    let (open, hold) = mpsc::channel();
    state.track_table_jobs.hold_next = Some(hold);
    panel
        .query
        .set_shown(TrackTableColumn::ConditionNumber, true);
    frame(&ctx, &mut panel, &mut state);
    state.drive_track_table_columns();
    let task = state.track_table_jobs.queued_task().expect("queued");
    assert!(state.quiet_task_running(task));

    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    let painted = texts(&ctx, &mut panel, &mut state);
    assert!(
        painted.iter().any(|t| t == "Updating\u{2026}"),
        "{painted:?}"
    );
    assert!(
        painted.contains(&old_angle),
        "the old rows keep their ray angle, {old_angle}: {painted:?}"
    );
    state.drive_track_table_columns();
    assert!(state.quiet_task_cancelled(task), "the edit asks it to stop");

    open.send(()).expect("the job waits at its gate");
    state.wait_for_track_table_job();
    assert!(
        !state
            .track_tables
            .latest(id)
            .is_some_and(|(_, table)| table.has(TrackTableColumn::ConditionNumber)),
        "nothing was merged for the serial the node left"
    );
    assert!(
        state.track_tables.get(id, old).is_some(),
        "the old rows' table"
    );

    frame(&ctx, &mut panel, &mut state);
    state.drive_track_table_columns();
    let next = state.track_table_jobs.queued_columns().unwrap().to_vec();
    assert!(
        next.iter().all(|c| c.cost() == ColumnCost::Stored),
        "the new version's stored columns first: {next:?}"
    );
    state.settle_track_table_columns();
    assert_eq!(
        computed_held(&state, id),
        [
            TrackTableColumn::MaxRayAngleDeg,
            TrackTableColumn::ConditionNumber
        ]
    );
    assert_eq!(
        state.track_table(id).unwrap().point_indexes().len(),
        POINTS - 1
    );
}

/// A column *Find points* computed is not computed again for the panel.
#[test]
fn a_computed_column_find_points_cached_is_not_computed_again() {
    let (mut state, id) = state();
    state
        .start_find_points(
            id,
            vec![TrackTableColumn::InverseDepthZ],
            TrackTableQueryOptions::default(),
        )
        .expect("a read");
    state.finish_background_tasks();
    let (ctx, mut panel) = stored_only(&mut state);
    let kept = state
        .track_table(id)
        .unwrap()
        .values(TrackTableColumn::InverseDepthZ)
        .unwrap()
        .as_ptr();
    panel
        .query
        .set_shown(TrackTableColumn::MaxRayAngleDeg, false);
    panel.query.set_shown(TrackTableColumn::InverseDepthZ, true);
    frame(&ctx, &mut panel, &mut state);
    assert!(!state.drive_track_table_columns(), "nothing to compute");
    let painted = texts(&ctx, &mut panel, &mut state);
    assert!(!painted.iter().any(|t| t == PENDING), "{painted:?}");
    assert_eq!(
        state
            .track_table(id)
            .unwrap()
            .values(TrackTableColumn::InverseDepthZ)
            .unwrap()
            .as_ptr(),
        kept
    );
}

/// The first click sorts the bad end first: the smallest ray angle and the
/// smallest inverse-depth z, and the largest condition number.
#[test]
fn the_first_click_on_a_computed_column_puts_the_bad_end_first() {
    for (column, descending) in [
        (TrackTableColumn::MaxRayAngleDeg, false),
        (TrackTableColumn::ConditionNumber, true),
        (TrackTableColumn::InverseDepthZ, false),
    ] {
        assert_eq!(
            TableSort::first_click(column),
            TableSort { column, descending },
            "{column:?}"
        );
        assert_eq!(column.worst_first_descending(), descending);
        assert_eq!(worst_first_ascending(column), !descending);
        assert!(
            sort_hover(None, column).ends_with("worst first."),
            "{column:?}"
        );
    }
}

/// A point at infinity's computed cells: a ray angle of 0, and no condition
/// number or inverse-depth z. Large values are written in scientific
/// notation.
#[test]
fn a_point_at_infinity_prints_a_zero_angle_and_no_triangulation_values() {
    let (mut state, id) = state();
    state
        .start_find_points(
            id,
            TrackTableColumn::ALL.to_vec(),
            TrackTableQueryOptions::default(),
        )
        .expect("a read");
    state.finish_background_tasks();
    let table = state.track_table(id).unwrap();
    let row = table.point_indexes().binary_search(&BEARING).unwrap();
    assert_eq!(cell_text(table, TrackTableColumn::MaxRayAngleDeg, row), "0");
    assert_eq!(
        cell_text(table, TrackTableColumn::ConditionNumber, row),
        "-"
    );
    assert_eq!(cell_text(table, TrackTableColumn::InverseDepthZ, row), "-");
    let finite = table.point_indexes().binary_search(&3).unwrap();
    for column in [
        TrackTableColumn::MaxRayAngleDeg,
        TrackTableColumn::ConditionNumber,
        TrackTableColumn::InverseDepthZ,
    ] {
        let text = cell_text(table, column, finite);
        assert!(text.parse::<f64>().is_ok(), "{column:?}: {text}");
    }

    assert_eq!(compact(12.345), "12.3");
    assert_eq!(compact(99_999.0), "99999");
    assert_eq!(compact(123_456.0), "1.23e5");
    assert_eq!(compact(-2.5e7), "-2.50e7");
    assert_eq!(compact(f64::INFINITY), "\u{221e}");
}
