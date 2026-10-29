// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Track Table panel and its table job.
//!
//! egui lays a frame out with no GPU behind it, so `show` really does run the
//! query and paint the visible rows; the assertions read the rows the panel
//! drew, the strings it painted and the state it changed. The table job runs
//! on a real worker through the fast queue.

use std::sync::mpsc;

use sfmtool_core::analysis::track_table::{TrackTableColumn, TrackTableQueryOptions};
use sfmtool_core::progress::Progress;
use sfmtool_core::SfmrReconstruction;

use super::{cell_text, grouped, significant, TrackTablePanel, TrackTableQuery};
use crate::background::{Finished, Operation};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::table_sort::TableSort;
use crate::test_support::{painted_text_rects, painted_texts, run_frame_headless};

mod computed;
mod crop;
mod filter_3d;
mod pictures;
mod quality;

/// How many points the fixture holds.
const POINTS: usize = 40;

/// The point the fixture makes a point at infinity, which has no `x`.
const BEARING: u32 = 7;

/// A state holding one demo node of [`POINTS`] points, selected. Point `p`'s
/// error is `p / 10` px, so sorting by error orders the points by index, and
/// point [`BEARING`] is a point at infinity.
fn state() -> (AppState, ReconId) {
    let mut recon = SfmrReconstruction::demo(POINTS);
    for (p, point) in recon.point_set.points.iter_mut().enumerate() {
        point.rms_reprojection_error = p as f32 / 10.0;
    }
    let point = &mut recon.point_set.points[BEARING as usize];
    point.position = nalgebra::Point3::from(point.position.coords.normalize());
    point.w = 0.0;
    recon.rebuild_derived_fields();
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.scene[0].id;
    state.select_recon(id);
    (state, id)
}

/// A window of 800 × `height` points.
fn input(height: f32) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, height),
        )),
        ..Default::default()
    }
}

/// One frame of the panel, in a window tall enough for every row.
fn frame(ctx: &egui::Context, panel: &mut TrackTablePanel, state: &mut AppState) {
    run_frame_headless(ctx, input(1200.0), |ui| panel.show(ui, state));
}

/// A panel drawn once, its table job run to its end, and drawn again, so the
/// rows are on screen.
fn shown(state: &mut AppState) -> (egui::Context, TrackTablePanel) {
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    frame(&ctx, &mut panel, state);
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, state);
    (ctx, panel)
}

/// Press and release the primary button at `at`, over two frames.
fn click(ctx: &egui::Context, panel: &mut TrackTablePanel, state: &mut AppState, at: egui::Pos2) {
    for pressed in [true, false] {
        let mut raw = input(1200.0);
        raw.events = vec![
            egui::Event::PointerMoved(at),
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            },
        ];
        run_frame_headless(ctx, raw, |ui| panel.show(ui, state));
    }
}

/// Where the panel painted `text`, the first time.
fn painted_at(
    ctx: &egui::Context,
    panel: &mut TrackTablePanel,
    state: &mut AppState,
    text: &str,
) -> egui::Pos2 {
    painted_text_rects(ctx, input(1200.0), |ui| panel.show(ui, state))
        .into_iter()
        .find(|painted| painted.text == text)
        .unwrap_or_else(|| panic!("{text:?} was not painted"))
        .rect
        .center()
}

/// The columns the cache holds for `id` at its cursor.
fn held(state: &AppState, id: ReconId) -> Vec<TrackTableColumn> {
    let table = state.track_table(id).expect("a cached table");
    TrackTableColumn::ALL
        .into_iter()
        .filter(|&c| table.has(c))
        .collect()
}

// -- Registration and placement ---------------------------------------------

#[test]
fn the_panel_is_the_tenth_tab_titled_track_table() {
    use crate::dock::Tab;
    assert_eq!(Tab::ALL.len(), 10);
    assert_eq!(Tab::ALL[9], Tab::TrackTable);
    assert_eq!(Tab::TrackTable.title(), "Track Table");
    assert_eq!(Tab::TrackTable.wire_name(), "track_table");
    assert_eq!(Tab::from_wire_name("track_table"), Some(Tab::TrackTable));
}

/// Its home is the centre group, behind the 3D Viewer and Image Detail, and
/// the 3D Viewer stays the raised tab, so the default layout opens showing the
/// panels it did before.
#[test]
fn the_default_layout_puts_the_table_behind_the_viewport_and_image_detail() {
    use crate::dock::Tab;
    use crate::layout::{Layout, LayoutNode};
    fn leaves(node: &LayoutNode, out: &mut Vec<(Vec<Tab>, Tab)>) {
        match node {
            LayoutNode::Leaf { tabs, active } => out.push((tabs.clone(), *active)),
            LayoutNode::Split { first, second, .. } => {
                leaves(first, out);
                leaves(second, out);
            }
        }
    }
    let mut out = Vec::new();
    leaves(Layout::default().main.as_ref().expect("docked"), &mut out);
    let centre = out
        .iter()
        .find(|(tabs, _)| tabs.contains(&Tab::TrackTable))
        .expect("the table is docked");
    assert_eq!(
        centre.0,
        vec![Tab::Viewer3D, Tab::ImageDetail, Tab::TrackTable]
    );
    assert_eq!(centre.1, Tab::Viewer3D);
}

#[test]
fn with_no_node_selected_the_panel_says_so() {
    let mut state = AppState::new();
    let mut panel = TrackTablePanel::new();
    let texts = painted_texts(&egui::Context::default(), input(600.0), |ui| {
        panel.show(ui, &mut state)
    });
    assert!(
        texts.iter().any(|t| t == "No reconstruction loaded"),
        "{texts:?}"
    );
    assert_eq!(state.track_table_jobs.queued_task(), None);
}

// -- The table job ----------------------------------------------------------

/// The panel's first frame queues the job, and its first table fills the
/// cache with every stored column the node can supply and none of the
/// computed ones.
#[test]
fn the_table_job_fills_the_cache_with_the_stored_columns() {
    let (mut state, id) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    let texts = painted_texts(&ctx, input(600.0), |ui| panel.show(ui, &mut state));
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("Computing the track table of")),
        "{texts:?}"
    );
    assert!(
        !texts
            .iter()
            .any(|t| t == "Updating\u{2026}" || t.ends_with(" points")),
        "the first open has no rows to keep on show: {texts:?}"
    );
    assert!(
        state.drive_track_table_columns(),
        "the first frame queues it"
    );
    let edited = state.scene[0].edited().clone();
    let expected: Vec<TrackTableColumn> = super::job::missing_stored_columns(&edited, None);
    assert!(!expected.is_empty());
    assert_eq!(
        state.track_table_jobs.queued_columns(),
        Some(expected.as_slice())
    );
    state.wait_for_track_table_job();
    assert_eq!(held(&state, id), expected);
    // The default Ray angle column is computed by a second run.
    state.settle_track_table_columns();
    assert!(
        !state.drive_track_table_columns(),
        "nothing is queued once every column needed is held"
    );
    assert_eq!(state.track_table_jobs.queued_task(), None);
    assert_eq!(state.scene[0].history.versions().len(), 1, "a read");
}

/// A column *Find points* already cached is kept, not computed again: the job
/// computes only the columns the cache lacks and merges them beside it.
#[test]
fn the_job_merges_beside_what_find_points_cached() {
    let (mut state, id) = state();
    state
        .start_find_points(
            id,
            vec![
                TrackTableColumn::TrackLength,
                TrackTableColumn::MaxRayAngleDeg,
            ],
            TrackTableQueryOptions::default(),
        )
        .expect("a read");
    state.finish_background_tasks();
    let kept = state
        .track_table(id)
        .unwrap()
        .values(TrackTableColumn::TrackLength)
        .unwrap()
        .as_ptr();
    let (_ctx, _panel) = shown(&mut state);
    let table = state.track_table(id).unwrap();
    assert_eq!(
        table
            .values(TrackTableColumn::TrackLength)
            .unwrap()
            .as_ptr(),
        kept,
        "the column already held is kept"
    );
    assert!(table.has(TrackTableColumn::RmsErrorPx));
    assert!(
        table.has(TrackTableColumn::MaxRayAngleDeg),
        "the panel and find_points share one cache"
    );
}

/// A frame in which the panel does not draw asks for nothing.
#[test]
fn a_hidden_panel_queues_no_job() {
    let (mut state, id) = state();
    state.want_track_table_columns(id, Vec::new(), Vec::new());
    state.hide_track_table();
    assert!(!state.drive_track_table_columns());
    assert_eq!(state.track_table_jobs.queued_task(), None);
    assert!(state.track_table(id).is_none());
}

/// An edit while the job runs asks it to stop, and a table it brings back
/// for the serial the node has left is dropped rather than merged.
#[test]
fn an_edit_cancels_the_job_and_its_table_for_the_old_serial_is_dropped() {
    let (mut state, id) = state();
    let (open, hold) = mpsc::channel();
    state.track_table_jobs.hold_next = Some(hold);
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    frame(&ctx, &mut panel, &mut state);
    state.drive_track_table_columns();
    let task = state.track_table_jobs.queued_task().expect("queued");
    assert!(
        state.quiet_task_running(task),
        "an idle fast queue starts it"
    );
    let old = state.scene[0].history.current_version().document_serial;

    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    state.drive_track_table_columns();
    assert!(state.quiet_task_cancelled(task), "the edit asks it to stop");

    open.send(()).expect("the job waits at its gate");
    state.wait_for_track_table_job();
    assert!(
        state.track_tables.get(id, old).is_none(),
        "nothing was merged for the serial the node left"
    );
    assert!(state.track_table(id).is_none());

    // The next drive queues the job for the version now at the cursor.
    state.settle_track_table_columns();
    let table = state.track_table(id).expect("computed for the new serial");
    assert_eq!(table.point_indexes().len(), POINTS - 1);
}

/// A table that lands after the cursor moved is dropped even when nothing
/// cancelled the job.
#[test]
fn a_table_that_lands_for_an_old_serial_is_dropped() {
    let (mut state, id) = state();
    let (open, hold) = mpsc::channel();
    state.track_table_jobs.hold_next = Some(hold);
    state.want_track_table_columns(id, Vec::new(), Vec::new());
    state.drive_track_table_columns();
    let old = state.scene[0].history.current_version().document_serial;
    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    // No drive between the edit and the landing, so the job is not asked to
    // stop and brings back the table of the old version.
    open.send(()).expect("the job waits at its gate");
    state.wait_for_track_table_job();
    assert!(state.track_tables.get(id, old).is_none());
    assert!(state.track_table(id).is_none());
}

/// The job locks nothing and reads the version on screen, so it runs while a
/// slow task holds the node.
#[test]
fn the_job_runs_on_a_node_a_slow_task_holds() {
    let (mut state, id) = state();
    let (open, hold) = mpsc::channel::<()>();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |_progress: &Progress<'_>| {
                let _ = hold.recv();
                Finished::Failed("held for the test".to_string())
            }),
        )
        .expect("nothing else runs");
    assert!(state.busy_refusal(id).is_some(), "the node is locked");
    let (_ctx, _panel) = shown(&mut state);
    assert!(
        state.track_table(id).is_some(),
        "the table was computed beside the slow task"
    );
    open.send(()).expect("the slow task waits at its gate");
    state.finish_background_tasks();
}

/// A locking task queued on the job's node in the other queue does not ask
/// the running job to stop: the job does not wait for its node, so it gives
/// way only in its own queue.
#[test]
fn a_slow_task_queued_on_its_node_leaves_the_running_job_alone() {
    let (mut state, id) = state();
    let (open_job, hold_job) = mpsc::channel();
    state.track_table_jobs.hold_next = Some(hold_job);
    state.want_track_table_columns(id, Vec::new(), Vec::new());
    state.drive_track_table_columns();
    let task = state.track_table_jobs.queued_task().expect("queued");
    assert!(
        state.quiet_task_running(task),
        "an idle fast queue starts it"
    );

    let (open_slow, hold_slow) = mpsc::channel::<()>();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |_progress: &Progress<'_>| {
                let _ = hold_slow.recv();
                Finished::Failed("held for the test".to_string())
            }),
        )
        .expect("the slow queue is idle");
    assert!(state.busy_refusal(id).is_some(), "the node is locked");
    assert!(
        !state.quiet_task_cancelled(task),
        "a locking task in the slow queue asked the table job to stop"
    );

    open_job.send(()).expect("the job waits at its gate");
    state.wait_for_track_table_job();
    assert!(
        state.track_table(id).is_some(),
        "the job finished beside the slow task"
    );
    open_slow.send(()).expect("the slow task waits at its gate");
    state.finish_background_tasks();
}

/// *Find points* queued in the fast queue while the job runs asks it to stop,
/// starts once it has, and the job is queued again behind it.
#[test]
fn find_points_queued_while_the_job_runs_cancels_it_and_it_runs_again_behind() {
    let (mut state, id) = state();
    let (open_job, hold_job) = mpsc::channel();
    state.track_table_jobs.hold_next = Some(hold_job);
    state.want_track_table_columns(id, Vec::new(), Vec::new());
    state.drive_track_table_columns();
    let task = state.track_table_jobs.queued_task().expect("queued");
    assert!(state.quiet_task_running(task));

    let find = state
        .start_find_points(
            id,
            vec![TrackTableColumn::MaxRayAngleDeg],
            TrackTableQueryOptions::default(),
        )
        .expect("a read");
    assert!(
        state.quiet_task_cancelled(task),
        "an asked-for task in the fast queue asks the job to stop"
    );
    assert!(
        state.waiting_task(find).is_some(),
        "Find points waits for it"
    );

    // The cancelled job computes nothing and leaves the queue, and Find
    // points starts in its place.
    open_job.send(()).expect("the job waits at its gate");
    state.wait_for_track_table_job();
    assert!(
        state.track_table(id).is_none(),
        "the cancelled job merged nothing"
    );
    assert!(
        state.running_task(find).is_some(),
        "Find points runs once the job has stopped"
    );

    // The driver queues the job again, behind Find points.
    state.drive_track_table_columns();
    let again = state.track_table_jobs.queued_task().expect("queued again");
    assert_ne!(again, task);
    assert!(
        !state.quiet_task_running(again),
        "the job waits behind Find points"
    );

    state.settle_track_table_columns();
    let table = state.track_table(id).expect("both landed");
    assert!(table.has(TrackTableColumn::MaxRayAngleDeg));
    assert!(table.has(TrackTableColumn::RmsErrorPx));
    assert!(matches!(
        state.task(find),
        Some(crate::background::TaskState::Finished(_))
    ));
}

/// Holding the declaration `cancellable: true` to its claim: the kernel the
/// job runs stops when its flag is set.
#[test]
fn the_job_stops_when_cancelled() {
    assert!(super::job::job_stops_when_cancelled());
}

// -- The row list -------------------------------------------------------------

/// A bound keeps the rows inside it, both ends inclusive, and a row with no
/// value fails a bound on its column.
#[test]
fn bounds_filter_the_rows_and_a_row_with_no_value_fails_them() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    assert_eq!(panel.rows().len(), POINTS);

    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).min = "1".into();
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "2".into();
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows(), (10..=20).collect::<Vec<u32>>().as_slice());

    // X: every finite point has a value; the bearing has none.
    panel.query = TrackTableQuery::default();
    panel.query.set_shown(TrackTableColumn::X, true);
    panel.query.bound_mut(TrackTableColumn::X).min = "-1e9".into();
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows().len(), POINTS - 1);
    assert!(!panel.rows().contains(&BEARING));
}

/// A box that does not hold a number sets no bound, and says so.
#[test]
fn a_box_that_is_not_a_number_sets_no_bound() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "one".into();
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows().len(), POINTS);
}

/// The rows sort by the clicked column, and the rows with no value go last
/// in either order.
#[test]
fn sorting_puts_the_rows_with_no_value_last_in_either_order() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.set_shown(TrackTableColumn::X, true);
    for descending in [false, true] {
        panel.query.sort = Some(TableSort {
            column: TrackTableColumn::X,
            descending,
        });
        frame(&ctx, &mut panel, &mut state);
        assert_eq!(panel.rows().len(), POINTS);
        assert_eq!(*panel.rows().last().unwrap(), BEARING, "{descending}");
    }
}

/// The row list is computed once per change of its inputs: a frame with an
/// unchanged query, version and cache does not compute it again, and a change
/// of any of the three does.
#[test]
fn a_frame_with_an_unchanged_query_does_not_recompute_the_row_list() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    let first = panel.recomputed();
    assert_eq!(first, 1, "computed once, when the table first arrived");
    for _ in 0..5 {
        frame(&ctx, &mut panel, &mut state);
    }
    assert_eq!(panel.recomputed(), first, "five unchanged frames");

    // Moving the pointer, selecting an image and hovering change nothing it
    // reads.
    state.select_image(Some(ImageRef::new(id, 0)));
    state.hovered_point = Some(PointRef {
        recon: id,
        point: 3,
    });
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.recomputed(), first);

    // The query.
    panel.query.click_heading(TrackTableColumn::RmsErrorPx);
    frame(&ctx, &mut panel, &mut state);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.recomputed(), first + 1);

    // The version: an edit moves the serial, and the rows are computed again
    // once the job has brought the new version's columns.
    state
        .delete_point(PointRef {
            recon: id,
            point: 1,
        })
        .expect("deletable");
    frame(&ctx, &mut panel, &mut state);
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.recomputed(), first + 2);
    assert!(!panel.rows().contains(&1));

    // The cache: a column merged in moves the inputs.
    state
        .start_find_points(
            id,
            vec![TrackTableColumn::ConditionNumber],
            TrackTableQueryOptions::default(),
        )
        .expect("a read");
    state.finish_background_tasks();
    frame(&ctx, &mut panel, &mut state);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.recomputed(), first + 3);
}

/// With the box ticked, only the points the selected image observes are
/// kept; with no image selected it filters nothing.
#[test]
fn the_selected_image_box_keeps_the_points_that_image_sees() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.seen_in_selected_image = true;
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows().len(), POINTS, "no image selected");

    state.select_image(Some(ImageRef::new(id, 0)));
    frame(&ctx, &mut panel, &mut state);
    let recon = state.scene[0].recon();
    let seen: Vec<u32> = (0..POINTS as u32)
        .filter(|&p| {
            recon
                .point_set
                .tracks
                .iter()
                .any(|t| t.point_index == p && t.image_index == 0)
        })
        .collect();
    assert!(!seen.is_empty() && seen.len() < POINTS);
    assert_eq!(panel.rows(), seen.as_slice());
}

/// The box's label is the same whether or not an image is selected, so the
/// header does not grow with an image's name; its hover text names the
/// selected image, or says that none is selected and the box filters nothing.
#[test]
fn the_selected_image_box_names_the_image_in_its_hover_text() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    // A tooltip waits out `tooltip_delay`, which a headless frame has no wall
    // clock to pass.
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let mut hovered = |state: &mut AppState| {
        let at = painted_at(&ctx, &mut panel, state, "Seen in selected image");
        let mut texts = Vec::new();
        for _ in 0..3 {
            let mut raw = input(1200.0);
            raw.events = vec![egui::Event::PointerMoved(at)];
            texts = painted_texts(&ctx, raw, |ui| panel.show(ui, state));
        }
        texts
    };
    let texts = hovered(&mut state);
    assert!(
        texts.iter().any(|t| t
            == "Keep only the points seen in the selected image. No image of this \
                reconstruction is selected, so this filters nothing."),
        "{texts:?}"
    );

    state.select_image(Some(ImageRef::new(id, 0)));
    let name = state.scene[0].recon().image_set.images[0].name.clone();
    let texts = hovered(&mut state);
    assert!(
        texts.iter().any(|t| t == "Seen in selected image"),
        "{texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("Only points seen")),
        "{texts:?}"
    );
    let hover = format!("Keep only the points seen in the selected image, {name}.");
    assert!(texts.contains(&hover), "{texts:?}");
}

// -- The heading row and the columns ------------------------------------------

/// A first click on Error puts the worst rows first; a second reverses them;
/// a click on Point puts them back in point order.
#[test]
fn clicking_a_heading_sorts_worst_first_and_a_second_click_reverses() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    let heading = painted_at(&ctx, &mut panel, &mut state, "Error (px)");
    click(&ctx, &mut panel, &mut state, heading);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(
        panel.query.sort,
        Some(TableSort {
            column: TrackTableColumn::RmsErrorPx,
            descending: true
        })
    );
    assert_eq!(panel.rows()[0], POINTS as u32 - 1, "the worst error first");

    click(&ctx, &mut panel, &mut state, heading);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows()[0], 0);

    let point = painted_at(&ctx, &mut panel, &mut state, "Point");
    click(&ctx, &mut panel, &mut state, point);
    assert_eq!(panel.query.sort, None);
}

/// Each heading that sorts is named *Sort by* and the text it shows, so an
/// agent finds the **Track** heading by the word a person reads on it, and a
/// click on that node sorts by the track length.
#[test]
fn a_heading_is_named_after_the_title_it_shows() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    ctx.enable_accesskit();
    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    let mut output = ctx.run_ui(input(1200.0), |ui| panel.show(ui, &mut state));
    output.textures_delta.clear();
    let nodes: Vec<egui::accesskit::Node> = output
        .platform_output
        .accesskit_update
        .expect("AccessKit is on")
        .nodes
        .into_iter()
        .map(|(_, node)| node)
        .collect();
    let sort_buttons: Vec<&egui::accesskit::Node> = nodes
        .iter()
        .filter(|node| {
            node.role() == egui::accesskit::Role::Button
                && node.label().is_some_and(|l| l.starts_with("Sort by "))
        })
        .collect();
    let mut named: Vec<&str> = sort_buttons
        .iter()
        .map(|node| &node.label().unwrap()["Sort by ".len()..])
        .collect();
    named.sort_unstable();
    let mut shown_titles = headings(&texts);
    shown_titles.push("Point");
    shown_titles.sort_unstable();
    assert_eq!(named, shown_titles, "{texts:?}");

    let track = sort_buttons
        .iter()
        .find(|node| node.label() == Some("Sort by Track"))
        .expect("the Track heading");
    let bounds = track.bounds().expect("the node has a rectangle");
    let at = egui::pos2(
        ((bounds.x0 + bounds.x1) / 2.0) as f32,
        ((bounds.y0 + bounds.y1) / 2.0) as f32,
    );
    click(&ctx, &mut panel, &mut state, at);
    assert_eq!(
        panel.query.sort.map(|sort| sort.column),
        Some(TrackTableColumn::TrackLength)
    );
}

/// A column holding a bound or the sort cannot be hidden, so a hidden column
/// never filters or orders the rows unseen.
#[test]
fn a_column_holding_a_bound_or_the_sort_cannot_be_hidden() {
    let mut query = TrackTableQuery::default();
    query.bound_mut(TrackTableColumn::RmsErrorPx).max = "2".into();
    query.set_shown(TrackTableColumn::RmsErrorPx, false);
    assert!(query.shown.contains(&TrackTableColumn::RmsErrorPx));
    query.click_heading(TrackTableColumn::TrackLength);
    query.set_shown(TrackTableColumn::TrackLength, false);
    assert!(query.shown.contains(&TrackTableColumn::TrackLength));
    query.set_shown(TrackTableColumn::CentroidDistance, false);
    assert!(!query.shown.contains(&TrackTableColumn::CentroidDistance));
    query.set_shown(TrackTableColumn::Z, true);
    query.set_shown(TrackTableColumn::X, true);
    let at = |c| query.shown.iter().position(|&s| s == c).unwrap();
    assert!(
        at(TrackTableColumn::X) < at(TrackTableColumn::Z),
        "ALL order"
    );
}

/// The fixture with each of `points` holding its second observation in the
/// image of its first, so that its track has one repeat observation.
fn state_with_repeats(points: &[u32]) -> (AppState, ReconId) {
    let (mut state, id) = state();
    let recon = state.scene[0].recon_mut();
    for &point in points {
        let rows: Vec<usize> = (0..recon.point_set.tracks.len())
            .filter(|&k| recon.point_set.tracks[k].point_index == point)
            .collect();
        assert!(rows.len() >= 2, "point {point} has two observations");
        recon.point_set.tracks[rows[1]].image_index = recon.point_set.tracks[rows[0]].image_index;
    }
    recon.rebuild_derived_fields();
    (state, id)
}

/// The value columns' headings the panel painted.
fn headings(texts: &[String]) -> Vec<&str> {
    [
        "Track",
        "Repeat obs.",
        "Back-facing obs.",
        "Error (px)",
        "Distance",
        "Ray angle (\u{b0})",
        "Cond.",
        "Inv. depth z",
        "X",
    ]
    .into_iter()
    .filter(|title| texts.iter().any(|t| t == title))
    .collect()
}

/// The default columns are Track, Error, Distance and Ray angle; Repeat obs.
/// stays out while no point has a repeat observation, and the header says
/// nothing of them.
#[test]
fn the_default_columns_are_track_error_distance_and_ray_angle() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    assert_eq!(
        headings(&texts),
        ["Track", "Error (px)", "Distance", "Ray angle (\u{b0})"],
        "{texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("repeat observations")),
        "{texts:?}"
    );
}

/// Distance holds `∞` for the point at infinity: it sorts above every finite
/// distance, a finite maximum drops it, and a minimum of inf keeps it alone.
#[test]
fn a_point_at_infinity_sorts_and_filters_as_the_largest_distance() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    let heading = painted_at(&ctx, &mut panel, &mut state, "Distance");
    click(&ctx, &mut panel, &mut state, heading);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(
        panel.query.sort,
        Some(TableSort {
            column: TrackTableColumn::CentroidDistance,
            descending: false
        })
    );
    assert_eq!(*panel.rows().last().unwrap(), BEARING, "∞ is the largest");
    click(&ctx, &mut panel, &mut state, heading);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows()[0], BEARING, "∞ first in decreasing order");

    panel
        .query
        .bound_mut(TrackTableColumn::CentroidDistance)
        .max = "1e9".into();
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows().len(), POINTS - 1);
    assert!(!panel.rows().contains(&BEARING));
    *panel.query.bound_mut(TrackTableColumn::CentroidDistance) = Default::default();
    panel
        .query
        .bound_mut(TrackTableColumn::CentroidDistance)
        .min = "inf".into();
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows(), [BEARING]);
}

/// Repeat obs. is shown by itself while the reconstruction has a point with
/// repeat observations, the header counts those points, and its first click
/// puts the worst first.
#[test]
fn repeat_observations_show_themselves_and_their_count() {
    let (mut state, _) = state_with_repeats(&[3, 12]);
    let (ctx, mut panel) = shown(&mut state);
    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    assert_eq!(
        headings(&texts),
        [
            "Track",
            "Repeat obs.",
            "Error (px)",
            "Distance",
            "Ray angle (\u{b0})"
        ],
        "{texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t == "\u{b7} 2 with repeat observations"),
        "{texts:?}"
    );
    assert!(
        !panel
            .query
            .shown
            .contains(&TrackTableColumn::RepeatObservations),
        "shown without being added to the person's columns"
    );
    let heading = painted_at(&ctx, &mut panel, &mut state, "Repeat obs.");
    click(&ctx, &mut panel, &mut state, heading);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(
        panel.query.sort,
        Some(TableSort {
            column: TrackTableColumn::RepeatObservations,
            descending: true
        })
    );
    assert_eq!(&panel.rows()[..2], [3, 12], "the defects first");
}

/// Showing Repeat obs. by itself never undoes the person's choice: unticked,
/// it stays hidden though the reconstruction has repeats; ticked, it stays
/// shown though it has none. While it holds the sort it is not hidden.
#[test]
fn the_persons_choice_of_repeat_observations_wins() {
    let column = TrackTableColumn::RepeatObservations;
    let repeats = [column];
    let mut query = TrackTableQuery::default();
    assert!(query.is_shown(column, &repeats));
    assert!(!query.is_shown(column, &[]));
    query.choose_shown(column, false, &repeats);
    assert!(!query.is_shown(column, &repeats), "unticked stays hidden");

    let mut query = TrackTableQuery::default();
    query.choose_shown(column, true, &[]);
    assert!(query.is_shown(column, &[]), "ticked stays shown");
    assert_eq!(
        query.visible(&[]),
        [
            TrackTableColumn::TrackLength,
            column,
            TrackTableColumn::RmsErrorPx,
            TrackTableColumn::CentroidDistance,
            TrackTableColumn::MaxRayAngleDeg,
        ]
    );

    // Shown by itself and sorted by, it stays while it orders the rows, on a
    // reconstruction with no repeats and against an untick.
    let mut query = TrackTableQuery::default();
    query.click_heading(column);
    assert!(query.is_shown(column, &[]));
    query.choose_shown(column, false, &repeats);
    assert!(query.is_shown(column, &repeats));
}

/// Each defect count shows itself by its own defects and keeps its own
/// choice: back-facing observations show Back-facing obs. and not Repeat
/// obs., and unticking one leaves the other showing itself.
#[test]
fn each_defect_count_shows_itself_by_its_own_defects() {
    let back = TrackTableColumn::BackFacingObservations;
    let repeat = TrackTableColumn::RepeatObservations;
    let mut query = TrackTableQuery::default();
    assert!(query.is_shown(back, &[back]));
    assert!(!query.is_shown(repeat, &[back]));
    assert!(!query.is_shown(back, &[repeat]));
    // The table job computes it while it may show itself.
    assert!(query.needed_computed().contains(&back));
    query.choose_shown(back, false, &[back, repeat]);
    assert!(!query.is_shown(back, &[back, repeat]));
    assert!(query.is_shown(repeat, &[back, repeat]));
    // Unticked, it is no longer computed for the panel.
    assert!(!query.needed_computed().contains(&back));
}

/// A column the reconstruction cannot supply is left out, and a bound on it
/// is kept, said, and filters nothing.
#[test]
fn a_column_the_reconstruction_cannot_supply_is_left_out() {
    let (mut state, _) = state();
    let edited = state.scene[0].edited().clone();
    assert!(!TrackTableColumn::NormalConfidence.is_available(&edited));
    let (ctx, mut panel) = shown(&mut state);
    panel
        .query
        .set_shown(TrackTableColumn::NormalConfidence, true);
    panel
        .query
        .bound_mut(TrackTableColumn::NormalConfidence)
        .min = "100".into();
    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    assert!(!texts.iter().any(|t| t == "Normal conf."), "{texts:?}");
    // The note opens with the sentence `find_points` refuses such a column
    // with, which lists the columns the reconstruction can supply.
    let label = &state.scene[0].label;
    let refusal = crate::track_table::cannot_supply_sentence(
        label,
        &[TrackTableColumn::NormalConfidence],
        &edited,
    );
    assert!(
        refusal.starts_with(&format!(
            "{label} cannot supply the column normal_confidence: the columns it can supply are \
             track_length, "
        )),
        "{refusal}"
    );
    let note = format!(
        "{refusal} Not filtering or sorting by Normal conf.; the bounds are kept for a \
         reconstruction that can."
    );
    assert!(texts.contains(&note), "{texts:?}");
    assert_eq!(panel.rows().len(), POINTS);
}

/// The panel publishes its query where the wire reads it, and takes the one
/// a `set_track_table_query` call left there on its next frame, drawn or not.
#[cfg(feature = "mcp")]
#[test]
fn the_panel_publishes_its_query_and_takes_one_set_over_the_wire() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "0.95".into();
    frame(&ctx, &mut panel, &mut state);
    let document = state.track_table_query.document();
    assert_eq!(
        document["bounds"],
        serde_json::json!([{ "column": "rms_error_px", "max": 0.95 }])
    );
    let change = crate::track_table_panel::TrackTableQueryChange {
        seen_in_selected_camera_image: Some(true),
        ..Default::default()
    };
    state
        .track_table_query
        .set("set_track_table_query", &change)
        .unwrap();
    // Not drawn: the frame's after-dock step takes it.
    panel.after_dock(&mut state, true);
    assert!(panel.query.seen_in_selected_image);
    assert_eq!(panel.query.bound(TrackTableColumn::RmsErrorPx).max, "0.95");
    // A person's edit after it is published again, not undone by the call.
    panel.query.seen_in_selected_image = false;
    frame(&ctx, &mut panel, &mut state);
    assert!(!panel.query.seen_in_selected_image);
    assert_eq!(
        state.track_table_query.document()["seen_in_selected_camera_image"],
        serde_json::json!(false)
    );
}

/// A defect count that shows itself is not in the get's `columns`, and a
/// `columns` that leaves it out decides it, as an untick in the Columns menu
/// does, so it no longer shows itself.
#[cfg(feature = "mcp")]
#[test]
fn a_set_columns_decides_the_defect_counts() {
    let back = TrackTableColumn::BackFacingObservations;
    let mut published = super::PublishedQuery::default();
    assert!(published.query.is_shown(back, &[back]));
    assert!(!published.document()["columns"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!(back.wire_name())));
    let change = crate::track_table_panel::TrackTableQueryChange {
        columns: Some(vec![crate::track_table_panel::PanelColumn::from_wire_name(
            "track_length",
        )
        .unwrap()]),
        ..Default::default()
    };
    published.set("set_track_table_query", &change).unwrap();
    assert!(!published.query.is_shown(back, &[back]));
}

/// The header names the reconstruction and counts matched rows against live
/// points.
#[test]
fn the_header_counts_matched_rows_against_live_points() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "0.95".into();
    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    assert!(texts.iter().any(|t| t == "10 of 40 points"), "{texts:?}");
    let label = state.scene[0].label.clone();
    assert!(texts.contains(&label), "{texts:?}");
}

// -- Selection and hover ------------------------------------------------------

/// Clicking a row selects its point, as any other selection does.
#[test]
fn clicking_a_row_selects_its_point() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    let &(point, rect) = panel
        .drawn()
        .iter()
        .find(|(point, _)| *point == 5)
        .expect("row 5 is drawn");
    click(&ctx, &mut panel, &mut state, rect.center());
    assert_eq!(state.selected_point, Some(PointRef { recon: id, point }));
}

/// Hovering a row hovers its point, and leaving the rows clears it once.
#[test]
fn hovering_a_row_hovers_its_point_and_leaving_clears_it() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    let &(point, rect) = panel.drawn().iter().find(|(p, _)| *p == 4).unwrap();
    let mut raw = input(1200.0);
    raw.events = vec![egui::Event::PointerMoved(rect.center())];
    run_frame_headless(&ctx, raw, |ui| panel.show(ui, &mut state));
    assert_eq!(state.hovered_point, Some(PointRef { recon: id, point }));

    let mut raw = input(1200.0);
    raw.events = vec![egui::Event::PointerGone];
    run_frame_headless(&ctx, raw, |ui| panel.show(ui, &mut state));
    assert_eq!(state.hovered_point, None);

    // Another panel's hover is left alone once the pointer is away.
    let other = PointRef {
        recon: id,
        point: 9,
    };
    state.hovered_point = Some(other);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(state.hovered_point, Some(other));
}

/// A point selected elsewhere has its row scrolled into view, when it is
/// among the matched rows.
#[test]
fn a_selection_made_elsewhere_scrolls_its_row_into_view() {
    let (mut state, id) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    let short = |ctx: &egui::Context, panel: &mut TrackTablePanel, state: &mut AppState| {
        run_frame_headless(ctx, input(240.0), |ui| panel.show(ui, state));
    };
    short(&ctx, &mut panel, &mut state);
    state.settle_track_table_columns();
    short(&ctx, &mut panel, &mut state);
    let far = POINTS as u32 - 2;
    assert!(
        !panel.drawn().iter().any(|(p, _)| *p == far),
        "the window is short enough to leave the row out"
    );
    state.select_point(PointRef {
        recon: id,
        point: far,
    });
    short(&ctx, &mut panel, &mut state);
    short(&ctx, &mut panel, &mut state);
    assert!(
        panel.drawn().iter().any(|(p, _)| *p == far),
        "{:?}",
        panel.drawn().iter().map(|(p, _)| p).collect::<Vec<_>>()
    );
}

// -- Cell text ----------------------------------------------------------------

#[test]
fn counts_are_grouped_by_thousands() {
    assert_eq!(grouped(0), "0");
    assert_eq!(grouped(999), "999");
    assert_eq!(grouped(9_941), "9,941");
    assert_eq!(grouped(1_234_567), "1,234,567");
}

#[test]
fn values_print_to_three_significant_digits() {
    assert_eq!(significant(0.8213, 3, 4), "0.821");
    assert_eq!(significant(41.24, 3, 4), "41.2");
    assert_eq!(significant(1234.6, 3, 4), "1235");
    assert_eq!(significant(0.0, 3, 4), "0");
    assert_eq!(significant(1_234_567.25, 4, 3), "1234567");
    assert_eq!(significant(1.23456, 4, 3), "1.235");
    assert_eq!(significant(f64::INFINITY, 3, 3), "\u{221e}");
    assert_eq!(significant(f64::NEG_INFINITY, 3, 3), "-\u{221e}");
}

/// The bearing's cells: no position, a distance of `∞`, and a count still.
#[test]
fn a_point_at_infinity_prints_no_position_and_an_infinite_distance() {
    let (mut state, id) = state();
    let _ = shown(&mut state);
    let table = state.track_table(id).unwrap();
    let row = table.point_indexes().binary_search(&BEARING).unwrap();
    assert_eq!(cell_text(table, TrackTableColumn::X, row), "-");
    assert_eq!(
        cell_text(table, TrackTableColumn::CentroidDistance, row),
        "\u{221e}"
    );
    let finite = cell_text(table, TrackTableColumn::CentroidDistance, 0);
    assert!(finite.parse::<f64>().is_ok_and(f64::is_finite), "{finite}");
    assert_eq!(cell_text(table, TrackTableColumn::TrackLength, row), "2");
    assert_eq!(
        cell_text(table, TrackTableColumn::RepeatObservations, row),
        "0"
    );
    assert_eq!(
        cell_text(table, TrackTableColumn::ConditionNumber, row),
        "\u{2026}",
        "a column the cache does not hold yet"
    );
}

/// A new sort starts the rows at the top, where the rows it puts first are,
/// even after a selection scrolled the table down.
#[test]
fn a_new_sort_scrolls_back_to_the_top() {
    let (mut state, id) = state();
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    let short = |ctx: &egui::Context, panel: &mut TrackTablePanel, state: &mut AppState| {
        run_frame_headless(ctx, input(240.0), |ui| panel.show(ui, state));
    };
    short(&ctx, &mut panel, &mut state);
    state.settle_track_table_columns();
    short(&ctx, &mut panel, &mut state);
    state.select_point(PointRef {
        recon: id,
        point: POINTS as u32 - 2,
    });
    short(&ctx, &mut panel, &mut state);
    short(&ctx, &mut panel, &mut state);
    assert!(!panel.drawn().iter().any(|(p, _)| *p == 0));

    panel.query.click_heading(TrackTableColumn::RmsErrorPx);
    short(&ctx, &mut panel, &mut state);
    short(&ctx, &mut panel, &mut state);
    let first = panel.rows()[0];
    assert_eq!(first, POINTS as u32 - 1);
    assert!(
        panel.drawn().iter().any(|(p, _)| *p == first),
        "{:?}",
        panel.drawn().iter().map(|(p, _)| p).collect::<Vec<_>>()
    );
}

// -- After an edit --------------------------------------------------------------

/// After an edit, until the new version's table lands, the panel keeps the
/// rows, the count and the headings of the version before, marked *Updating…*,
/// and a change of the bounds waits for the new table. Then the new rows and
/// count replace them.
#[test]
fn an_edit_keeps_the_old_rows_on_show_until_the_new_table_lands() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    assert!(
        state.track_table(id).is_none(),
        "the new table is not ready"
    );

    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    assert!(texts.iter().any(|t| t == "40 of 40 points"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "Updating\u{2026}"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "Error (px)"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "#0"), "{texts:?}");
    assert!(
        !texts.iter().any(|t| t.starts_with("Computing")),
        "{texts:?}"
    );
    assert_eq!(panel.rows().len(), POINTS);

    // A bound set while the rows are out of date waits for the new table.
    let computed = panel.recomputed();
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "0.95".into();
    frame(&ctx, &mut panel, &mut state);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows().len(), POINTS);
    assert_eq!(panel.recomputed(), computed);

    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    // Points 1 to 9 have an error under 0.95 px; point 0 is gone.
    assert!(texts.iter().any(|t| t == "9 of 39 points"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "Updating\u{2026}"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "#0"), "{texts:?}");
    assert_eq!(panel.rows(), (1..=9).collect::<Vec<u32>>().as_slice());
}

/// An out-of-date row takes no click and no hover, the deleted point's row
/// and a surviving point's row alike, since a row of the version before may
/// name a point the new version no longer has. Once the new table lands a
/// click selects again.
#[test]
fn an_out_of_date_row_cannot_be_selected_or_hovered() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    frame(&ctx, &mut panel, &mut state);
    for point in [0, 5] {
        let &(_, rect) = panel
            .drawn()
            .iter()
            .find(|(p, _)| *p == point)
            .expect("the old row is drawn");
        click(&ctx, &mut panel, &mut state, rect.center());
        assert_eq!(state.selected_point, None, "row #{point}");
        assert_eq!(state.hovered_point, None, "row #{point}");
    }

    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let &(point, rect) = panel.drawn().iter().find(|(p, _)| *p == 5).unwrap();
    click(&ctx, &mut panel, &mut state, rect.center());
    assert_eq!(state.selected_point, Some(PointRef { recon: id, point }));
}

// -- The pinned Point column ------------------------------------------------------

/// Scrolled sideways, the Point heading and the Point cells stay at the left
/// edge while the value columns' headings and cells move.
#[test]
fn the_point_column_stays_put_when_the_table_scrolls_sideways() {
    let (mut state, _) = state();
    let narrow = |time: f64| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(300.0, 600.0),
        )),
        time: Some(time),
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    for column in [
        TrackTableColumn::X,
        TrackTableColumn::Y,
        TrackTableColumn::Z,
        TrackTableColumn::CentroidDistance,
    ] {
        panel.query.set_shown(column, true);
    }
    run_frame_headless(&ctx, narrow(0.0), |ui| panel.show(ui, &mut state));
    state.settle_track_table_columns();
    run_frame_headless(&ctx, narrow(0.1), |ui| panel.show(ui, &mut state));
    let left_of = |texts: &[crate::test_support::PaintedText], text: &str| {
        texts
            .iter()
            .find(|t| t.text == text)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
            .rect
            .left()
    };
    let before = painted_text_rects(&ctx, narrow(0.2), |ui| panel.show(ui, &mut state));
    let &(_, row) = panel.drawn().iter().find(|(p, _)| *p == 2).unwrap();

    let mut time = 0.3;
    for _ in 0..20 {
        let mut raw = narrow(time);
        raw.events = vec![
            egui::Event::PointerMoved(egui::pos2(150.0, row.center().y)),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(-40.0, 0.0),
                phase: egui::TouchPhase::Move,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        run_frame_headless(&ctx, raw, |ui| panel.show(ui, &mut state));
        time += 0.1;
    }
    let after = painted_text_rects(&ctx, narrow(time), |ui| panel.show(ui, &mut state));

    for pinned in ["Point", "#2"] {
        assert_eq!(
            left_of(&before, pinned),
            left_of(&after, pinned),
            "{pinned} moved"
        );
    }
    let moved = left_of(&before, "Track") - left_of(&after, "Track");
    assert!(moved > 50.0, "the Track heading moved {moved}");
    assert!(
        left_of(&after, "Distance") < left_of(&before, "Distance") - 50.0,
        "the Distance heading did not scroll into view"
    );
}
