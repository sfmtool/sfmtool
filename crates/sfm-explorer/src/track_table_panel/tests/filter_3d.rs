// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The *Filter the 3D view* line: what the panel publishes to the 3D view
//! from its row list, when, and what the viewport's stats line makes of it.

use std::sync::Arc;

use sfmtool_core::analysis::track_table::TrackTableColumn;

use super::super::filter::{DIM, FILTER_3D, HIDE};
use super::{frame, shown, state, POINTS};
use crate::filter_3d::{Filter3dMode, KeptPoints};
use crate::scene::{ImageRef, PointRef};
use crate::state::AppState;
use crate::track_table_panel::TrackTablePanel;

/// The kept set the panel published, for the version at the node's cursor.
fn kept(state: &AppState) -> Arc<KeptPoints> {
    let filter = state.filter_3d.as_ref().expect("the filter is published");
    let document = state.scene[0].history.current_version().document_serial;
    Arc::clone(
        filter
            .kept_for(document)
            .expect("a set for the version in view"),
    )
}

/// Whether `kept` holds exactly the panel's matched rows.
fn holds_the_rows(kept: &KeptPoints, panel: &TrackTablePanel) -> bool {
    let rows = panel.rows();
    kept.count() == rows.len()
        && (0..POINTS as u32 + 4).all(|i| kept.contains(i) == rows.contains(&i))
}

/// The box starts clear and the panel publishes nothing; ticked, it publishes
/// the matched rows as a set for the version in view, Dim by default.
#[test]
fn the_box_publishes_the_matched_rows_while_it_is_ticked() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    assert!(state.filter_3d.is_none());

    panel.query.filter_3d = true;
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "1".into();
    frame(&ctx, &mut panel, &mut state);
    let filter = state.filter_3d.as_ref().expect("published");
    assert_eq!(filter.node, id);
    assert_eq!(filter.mode, Filter3dMode::Dim);
    let set = kept(&state);
    assert_eq!(
        set.count(),
        11,
        "points 0 to 10 have an error of at most 1 px"
    );
    assert!(holds_the_rows(&set, &panel));

    panel.query.filter_3d = false;
    frame(&ctx, &mut panel, &mut state);
    assert!(state.filter_3d.is_none(), "a cleared box drops the filter");
    assert!(panel.kept.is_none());
}

/// The set is the rows the bounds and the selected-image box keep, not their
/// order: a sort leaves the set as it was, the same `Arc`, while a new bound
/// or a new selected image builds another, and a frame with nothing changed
/// builds none.
#[test]
fn the_set_follows_the_bounds_and_the_selected_image_but_not_the_sort() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.filter_3d = true;
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).min = "2".into();
    frame(&ctx, &mut panel, &mut state);
    let first = kept(&state);
    assert_eq!(first.count(), 20, "points 20 to 39");
    frame(&ctx, &mut panel, &mut state);
    assert!(
        Arc::ptr_eq(&first, &kept(&state)),
        "an unchanged frame builds no set"
    );

    panel.query.click_heading(TrackTableColumn::RmsErrorPx);
    frame(&ctx, &mut panel, &mut state);
    let sorted = kept(&state);
    assert_eq!(*sorted, *first, "the sort moves no point in or out");
    assert!(holds_the_rows(&sorted, &panel));

    panel.query.seen_in_selected_image = true;
    state.select_image(Some(ImageRef::new(id, 0)));
    frame(&ctx, &mut panel, &mut state);
    let seen = kept(&state);
    assert!(holds_the_rows(&seen, &panel));
    assert!(
        seen.count() < first.count(),
        "the image does not see every point"
    );
    state.select_image(Some(ImageRef::new(id, 1)));
    frame(&ctx, &mut panel, &mut state);
    assert!(holds_the_rows(&kept(&state), &panel));
    assert!(!Arc::ptr_eq(&seen, &kept(&state)));
}

/// Dim or Hide is published as the mode and builds no new set.
#[test]
fn the_mode_changes_no_set() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.filter_3d = true;
    frame(&ctx, &mut panel, &mut state);
    let dimmed = kept(&state);
    panel.query.filter_3d_mode = Filter3dMode::Hide;
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(state.filter_3d.as_ref().unwrap().mode, Filter3dMode::Hide);
    assert!(Arc::ptr_eq(&dimmed, &kept(&state)));
}

/// After an edit the rows on show are the version before's, so the panel
/// publishes no set until the new version's rows are listed, and the 3D view
/// draws every point meanwhile; then the set is the new version's, without
/// the deleted point.
#[test]
fn after_an_edit_no_set_is_published_until_the_new_rows_land() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.filter_3d = true;
    frame(&ctx, &mut panel, &mut state);
    let before = kept(&state);
    assert!(before.contains(3));

    state
        .delete_point(PointRef {
            recon: id,
            point: 3,
        })
        .expect("deletable");
    frame(&ctx, &mut panel, &mut state);
    let filter = state.filter_3d.as_ref().expect("still on");
    let document = state.scene[0].history.current_version().document_serial;
    assert!(
        filter.kept.is_none(),
        "no set while the rows are out of date"
    );
    assert!(filter.kept_for(document).is_none());

    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let after = kept(&state);
    assert_eq!(after.count(), POINTS - 1);
    assert!(!after.contains(3));
}

/// A set built for one version is never applied to another: `kept_for` with
/// any serial but the one the rows were computed for answers nothing.
#[test]
fn a_set_answers_only_for_its_own_version() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.filter_3d = true;
    frame(&ctx, &mut panel, &mut state);
    let filter = state.filter_3d.clone().expect("published");
    let old = state.scene[0].history.current_version().document_serial;
    assert!(filter.kept_for(old).is_some());
    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    let new = state.scene[0].history.current_version().document_serial;
    assert!(filter.kept_for(new).is_none());
}

/// Open behind another tab, the panel still keeps the filter current: the
/// frame's call after the dock computes the rows again and asks the table job
/// for the columns. Closed, it filters nothing, and its box stays ticked.
#[test]
fn a_hidden_panel_keeps_the_filter_and_a_closed_one_drops_it() {
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.filter_3d = true;
    panel.query.seen_in_selected_image = true;
    frame(&ctx, &mut panel, &mut state);
    panel.after_dock(&mut state, true);

    // Hidden: an image selected elsewhere moves the rows and the set.
    state.select_image(Some(ImageRef::new(id, 0)));
    state.hide_track_table();
    panel.after_dock(&mut state, true);
    assert!(holds_the_rows(&kept(&state), &panel));
    assert!(kept(&state).count() < POINTS);

    // Hidden through an edit: the job computes the new version, and the set
    // follows once it lands, with no frame of the panel drawn.
    state
        .delete_point(PointRef {
            recon: id,
            point: 1,
        })
        .expect("deletable");
    state.hide_track_table();
    panel.after_dock(&mut state, true);
    assert!(state.filter_3d.as_ref().unwrap().kept.is_none());
    state.drive_track_table_columns();
    assert!(
        state.track_table_jobs.queued_task().is_some(),
        "the hidden panel names its node to the table job"
    );
    state.settle_track_table_columns();
    state.hide_track_table();
    panel.after_dock(&mut state, true);
    assert!(!kept(&state).contains(1));
    assert!(holds_the_rows(&kept(&state), &panel));

    panel.after_dock(&mut state, false);
    assert!(state.filter_3d.is_none(), "a closed panel filters nothing");
    assert!(panel.query.filter_3d, "the box stays ticked");
}

/// The stats line says how many points the filter leaves at full strength,
/// and while the table has no rows for the version in view that every point
/// is shown until they land.
#[test]
fn the_stats_line_counts_the_points_shown() {
    use crate::viewer_3d::overlay::scene_stats_text;
    let (mut state, id) = state();
    let (ctx, mut panel) = shown(&mut state);
    panel.query.filter_3d = true;
    panel.query.bound_mut(TrackTableColumn::RmsErrorPx).max = "1".into();
    frame(&ctx, &mut panel, &mut state);
    let text = scene_stats_text(&state.scene, None, state.filter_3d.as_ref(), false, 0.0);
    assert!(
        text.starts_with("40 points (1 at infinity) (11 shown) | "),
        "{text}"
    );
    assert!(!scene_stats_text(&state.scene, None, None, false, 0.0).contains("shown"));

    state
        .delete_point(PointRef {
            recon: id,
            point: 30,
        })
        .expect("deletable");
    frame(&ctx, &mut panel, &mut state);
    let text = scene_stats_text(&state.scene, None, state.filter_3d.as_ref(), false, 0.0);
    assert!(
        text.starts_with("39 points (1 at infinity) (all shown until the Track Table updates) | "),
        "{text}"
    );
}

/// The box and the two choices are named by their labels in the AccessKit
/// tree, which is what `get_widgets` lists them under, and a click on each
/// sets the query.
#[test]
fn the_box_and_the_choices_are_named_and_clicked() {
    use egui::accesskit::{Role, Toggled};
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    ctx.enable_accesskit();
    let nodes = |ctx: &egui::Context,
                 panel: &mut TrackTablePanel,
                 state: &mut AppState,
                 events: Vec<egui::Event>| {
        let raw = egui::RawInput {
            events,
            ..super::input(1200.0)
        };
        let mut output = ctx.run_ui(raw, |ui| panel.show(ui, state));
        output.textures_delta.clear();
        output
            .platform_output
            .accesskit_update
            .expect("AccessKit is on")
            .nodes
            .into_iter()
            .map(|(_, node)| node)
            .collect::<Vec<_>>()
    };
    nodes(&ctx, &mut panel, &mut state, Vec::new());
    let tree = nodes(&ctx, &mut panel, &mut state, Vec::new());
    let named = |role: Role, name: &str| {
        tree.iter()
            .find(|node| node.role() == role && node.label() == Some(name))
            .unwrap_or_else(|| panic!("no {role:?} named {name}"))
            .clone()
    };
    let checkbox = named(Role::CheckBox, FILTER_3D);
    assert_eq!(checkbox.toggled(), Some(Toggled::False));
    let dim = named(Role::RadioButton, DIM);
    let hide = named(Role::RadioButton, HIDE);
    assert_eq!(dim.toggled(), Some(Toggled::True));
    assert_eq!(hide.toggled(), Some(Toggled::False));

    let centre = |node: &egui::accesskit::Node| {
        let b = node.bounds().expect("a rectangle");
        egui::pos2(((b.x0 + b.x1) / 2.0) as f32, ((b.y0 + b.y1) / 2.0) as f32)
    };
    for target in [centre(&checkbox), centre(&hide)] {
        for pressed in [true, false] {
            nodes(
                &ctx,
                &mut panel,
                &mut state,
                vec![
                    egui::Event::PointerMoved(target),
                    egui::Event::PointerButton {
                        pos: target,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
            );
        }
    }
    assert!(panel.query.filter_3d);
    assert_eq!(panel.query.filter_3d_mode, Filter3dMode::Hide);
    assert_eq!(state.filter_3d.as_ref().unwrap().mode, Filter3dMode::Hide);
}
