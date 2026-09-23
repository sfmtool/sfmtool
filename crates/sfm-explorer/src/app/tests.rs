// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a frame decides before it draws. Headless: these are questions about
//! the state, asked the way the frame asks them.

use super::{selected_point_source, track_ray_source};
use crate::scene::{ImageRef, PointRef};
use crate::state::edits::tests as edits;
use crate::state::AppState;

/// Run the adjustment the way the viewer does, and wait for it.
fn adjust(state: &mut AppState, id: crate::scene::ReconId) {
    state
        .start_bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("the fixture is well posed");
    state.finish_background_task();
}

/// A node with a point worth selecting, and its id.
fn selected(index: u32) -> (AppState, PointRef) {
    let (mut state, id) = edits::adjustable_state();
    let point = PointRef::new(id, index as usize);
    state.select_point(point);
    (state, point)
}

/// The rays are built from a point *and* the value it was read out of, so a
/// new version of the same point is a different source.
///
/// This is the bug it is here for: a bundle adjustment that renumbers nothing
/// leaves the selection exactly where it was and replaces every position under
/// it. Comparing the selection alone said nothing had changed, and the rays
/// stayed drawn from the version before.
#[test]
fn a_new_version_of_the_selected_point_is_a_new_ray_source() {
    let (mut state, point) = selected(0);
    let before = track_ray_source(&state).expect("a selection with a live point");
    assert_eq!(before.0, point);

    let id = point.recon;
    adjust(&mut state, id);

    let after = track_ray_source(&state).expect("the point survived");
    assert_ne!(
        before, after,
        "the rays would have been left on the version before",
    );
    assert_eq!(
        after.0, before.0,
        "this case is the one where the selection does not move",
    );
}

/// Undo is the same question in reverse, and was the same bug.
#[test]
fn undoing_back_to_a_version_is_a_new_ray_source_again() {
    let (mut state, point) = selected(0);
    let id = point.recon;
    adjust(&mut state, id);
    let adjusted = track_ray_source(&state).expect("a live point");

    state.undo(id).expect("there is something to undo");
    let undone = track_ray_source(&state).expect("the point is back");
    assert_ne!(adjusted, undone, "undo left the rays where they were");

    state.redo(id).expect("there is something to redo");
    let redone = track_ray_source(&state).expect("a live point");
    assert_ne!(undone, redone, "redo left the rays where they were");
    assert_eq!(
        adjusted, redone,
        "the same version is the same source, so nothing is rebuilt twice",
    );
}

/// The frustums of the images that observe the selected point are lit from that
/// point's **track**, which is the version's rather than the index's, so they
/// are gated on the same source the rays are.
///
/// The adjustment is the case that makes the difference visible: it renumbers
/// nothing, so a comparison of the selection alone reads as no change at all and
/// the colours would be left describing the version before.
#[test]
fn a_new_version_under_the_selection_is_a_new_point_source() {
    let (mut state, point) = selected(0);
    let before = selected_point_source(&state).expect("a selection in a loaded node");

    adjust(&mut state, point.recon);

    assert_eq!(
        state.selected_point,
        Some(point),
        "this case is the one where the selection does not move",
    );
    let after = selected_point_source(&state).expect("the node is still loaded");
    assert_ne!(before, after, "the colours would have been left behind");
}

/// A commit puts the rays on the point it wrote, wherever they were.
///
/// Two changes at once, and each on its own would leave them wrong: the
/// selection moves to the written point, and the version under it is the one the
/// commit pushed.
#[test]
fn a_commit_puts_the_ray_source_on_the_point_it_wrote() {
    let (mut state, id) = crate::bench::tests::state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    // Somewhere other than the track's origin, which is where the map would
    // have left it.
    state.select_point(PointRef::new(id, 0));
    let before = track_ray_source(&state).expect("a live point");

    let written = state.commit_bench_track(id, &label).expect("a track stage");

    let after = track_ray_source(&state).expect("the written point is live");
    assert_ne!(before, after, "the rays were left on the point before");
    assert_eq!(
        after.0,
        PointRef::new(id, written.point as usize),
        "the rays are not on what the commit wrote",
    );

    // Undo takes the written point away again, and the rays with it: the
    // selection lands back on the point the commit replaced.
    state.undo(id).expect("the commit");
    let undone = track_ray_source(&state).expect("the replaced point is back");
    assert_ne!(undone, after);
    assert_eq!(
        undone.0,
        PointRef::new(id, written.replaced.expect("this one replaces") as usize),
    );
}

/// Nothing to draw is a source of its own, so the rays are cleared rather than
/// left behind: no selection, and a point this version does not have.
#[test]
fn a_point_that_is_not_there_has_no_ray_source() {
    let (mut state, point) = selected(0);
    assert!(track_ray_source(&state).is_some());

    state.deselect_point();
    assert_eq!(track_ray_source(&state), None, "a cleared selection");

    state.select_point(point);
    state.delete_point(point).expect("it deletes");
    assert_eq!(
        track_ray_source(&state),
        None,
        "a deleted point still draws rays",
    );
}

/// A node that is not on screen draws no rays.
///
/// `render_track_rays` draws whatever the buffer holds and has no node of its
/// own in the draw loop, so nothing else stops a hidden node's rays hanging in
/// the air over the node that is still shown.
#[test]
fn a_hidden_node_has_no_ray_source() {
    // The second node first: a node arriving clears the point selection, which
    // is a file taking focus rather than anything this is about.
    let (mut state, _) = edits::adjustable_state();
    let other = state.append_node(crate::scene::SceneNode::demo(
        sfmtool_core::SfmrReconstruction::demo(8),
    ));
    let point = PointRef::new(state.scene[0].id, 0);
    state.select_point(point);
    assert!(track_ray_source(&state).is_some(), "the point is selected");

    // The node's own eye.
    state.scene[0].visible = false;
    assert_eq!(track_ray_source(&state), None, "the eye was closed");
    state.scene[0].visible = true;
    assert!(track_ray_source(&state).is_some(), "and opened again");

    // Solo somewhere else, which hides this one without touching its eye.
    state.solo = Some(other);
    assert_eq!(track_ray_source(&state), None, "solo is elsewhere");
    state.solo = None;
    assert!(track_ray_source(&state).is_some(), "and released");
}

// ── What a click on a picked point does ─────────────────────────────────

/// The Action Log's texts, oldest first.
fn texts(state: &AppState) -> Vec<String> {
    state
        .action_log
        .entries()
        .map(|entry| entry.text.clone())
        .collect()
}

/// A double-click reaches this twice, because the second click of one is a
/// click in its own right: the first selects, the second stages.
///
/// What it has to leave behind is one bench item, Track View raised,
/// and one Action Log row per thing that actually happened -- one selection,
/// however many clicks named the same point.
#[test]
fn a_double_click_on_a_point_stages_it_once_and_raises_track_view() {
    let (mut state, id) = edits::adjustable_state();
    let point = PointRef::new(id, 11);
    state.hide_panel(crate::dock::Tab::TrackView);

    super::apply_point_click(&mut state, point, false);
    super::apply_point_click(&mut state, point, true);

    assert_eq!(state.selected_point, Some(point));
    let bench = state.scene[0].history.current_bench();
    assert_eq!(bench.entries().len(), 1, "the double-click staged twice");
    assert!(state.is_panel_open(crate::dock::Tab::TrackView));

    let rows = texts(&state);
    let selections = rows.iter().filter(|t| t.starts_with("Selected ")).count();
    assert_eq!(
        selections, 1,
        "the second click logged a selection: {rows:?}"
    );
    let staged = rows
        .iter()
        .filter(|t| t.contains("on the bench as"))
        .count();
    assert_eq!(staged, 1, "the point was staged twice: {rows:?}");
}

/// A single click is the selection and nothing else: the gesture that stages a
/// track is the double, and a reader clicking through a cloud collects no
/// bench items.
#[test]
fn a_single_click_on_a_point_only_selects_it() {
    let (mut state, id) = edits::adjustable_state();
    let point = PointRef::new(id, 11);

    super::apply_point_click(&mut state, point, false);

    assert_eq!(state.selected_point, Some(point));
    assert!(state.scene[0].history.current_bench().entries().is_empty());
}

// ── The `,` / `.` image-step keys ───────────────────────────────────────

/// A demo node, selected, with the panel structs a frame's shortcuts need.
///
/// `Parts` owns them so a test can hand `menu::shortcuts` the same `UiParts`
/// the frame builds.
struct Parts {
    state: AppState,
    viewer_3d: crate::viewer_3d::Viewer3D,
    image_browser: crate::image_browser::ImageBrowser,
    image_detail: crate::image_detail::ImageDetail,
    track_view: crate::track_view::TrackView,
    intrinsics_detail: crate::intrinsics_detail::IntrinsicsDetail,
}

impl Parts {
    fn new() -> Self {
        let mut state = AppState::new();
        state.append_node(crate::scene::SceneNode::demo(
            sfmtool_core::SfmrReconstruction::demo(8),
        ));
        Self::with_state(state)
    }

    fn with_state(state: AppState) -> Self {
        Parts {
            state,
            viewer_3d: crate::viewer_3d::Viewer3D::new(),
            image_browser: crate::image_browser::ImageBrowser::new(),
            image_detail: crate::image_detail::ImageDetail::new(),
            track_view: crate::track_view::TrackView::new(),
            intrinsics_detail: crate::intrinsics_detail::IntrinsicsDetail::new(),
        }
    }

    fn recon(&self) -> crate::scene::ReconId {
        self.state
            .selected_recon
            .expect("the demo node is selected")
    }

    /// Run one frame's worth of accelerators with `key` pressed, the way
    /// `App::run_egui_pass` runs them: at the top of the frame, before the dock
    /// is drawn.
    ///
    /// `typing` puts the keyboard where a text field would, which is the state
    /// `egui_wants_keyboard_input` reports and every accelerator is gated on.
    fn press(&mut self, key: egui::Key, typing: bool) {
        self.press_with(key, egui::Modifiers::NONE, typing);
    }

    /// [`Parts::press`] with `modifiers` held.
    fn press_with(&mut self, key: egui::Key, modifiers: egui::Modifiers, typing: bool) {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            }],
            ..Default::default()
        };
        let Parts {
            state,
            viewer_3d,
            image_browser,
            image_detail,
            track_view,
            intrinsics_detail,
        } = self;
        crate::test_support::run_frame_headless(&ctx, input, |ui| {
            if typing {
                ui.ctx()
                    .memory_mut(|m| m.request_focus(egui::Id::new("a_text_field")));
            }
            let mut parts = super::UiParts {
                app_state: state,
                viewer_3d,
                image_browser,
                image_detail,
                track_view,
                intrinsics_detail,
            };
            super::menu::shortcuts(ui, &mut parts);
        });
    }
}

/// The plain case: the stock layout, the 3D Viewer in front, `.` forward and
/// `,` back, wrapping at both ends.
#[test]
fn the_step_keys_walk_the_image_table_and_wrap() {
    let mut parts = Parts::new();
    let id = parts.recon();
    let n = parts.state.scene[0].recon().image_table.images.len();
    assert!(n >= 3, "the fixture has {n} images");
    parts.state.select_image(Some(ImageRef::new(id, 0)));

    parts.press(egui::Key::Period, false);
    assert_eq!(parts.state.selected_image, Some(ImageRef::new(id, 1)));
    parts.press(egui::Key::Comma, false);
    assert_eq!(parts.state.selected_image, Some(ImageRef::new(id, 0)));
    // Back off the front wraps to the end, and forward off the end to the
    // front.
    parts.press(egui::Key::Comma, false);
    assert_eq!(parts.state.selected_image, Some(ImageRef::new(id, n - 1)));
    parts.press(egui::Key::Period, false);
    assert_eq!(parts.state.selected_image, Some(ImageRef::new(id, 0)));
}

/// The bug this exists for: the keys used to be handled inside the 3D Viewer's
/// tab body, so they went silent whenever another tab was in front of it -- and
/// the stock layout puts Image Detail one click away in that very node.
#[test]
fn the_step_keys_work_with_image_detail_in_front_of_the_viewport() {
    let mut parts = Parts::new();
    let id = parts.recon();
    parts.state.select_image(Some(ImageRef::new(id, 0)));
    // Image Detail and the 3D Viewer share a node in the stock layout, so
    // raising one hides the other.
    parts.state.show_panel(crate::dock::Tab::ImageDetail);
    assert!(
        !parts.state.panel_is_in_front(crate::dock::Tab::Viewer3D),
        "the viewport is still the tab in front"
    );

    parts.press(egui::Key::Period, false);
    assert_eq!(parts.state.selected_image, Some(ImageRef::new(id, 1)));
    parts.press(egui::Key::Comma, false);
    assert_eq!(parts.state.selected_image, Some(ImageRef::new(id, 0)));
}

/// Typed into a text field, a comma is a comma.
#[test]
fn the_step_keys_do_nothing_while_a_text_field_holds_the_keyboard() {
    let mut parts = Parts::new();
    let id = parts.recon();
    parts.state.select_image(Some(ImageRef::new(id, 0)));

    parts.press(egui::Key::Period, true);
    parts.press(egui::Key::Comma, true);
    assert_eq!(
        parts.state.selected_image,
        Some(ImageRef::new(id, 0)),
        "a typed key stepped the selection"
    );
}

// ── Ctrl+D, Track View's *Duplicate* ────────────────────────────────────

/// With a track active on the bench, Ctrl+D puts a copy beside it and makes
/// the copy active, as the toolbar's *Duplicate* does.
#[test]
fn ctrl_d_duplicates_the_active_track() {
    let (mut state, id) = edits::adjustable_state();
    let point = PointRef::new(id, 11);
    super::apply_point_click(&mut state, point, false);
    super::apply_point_click(&mut state, point, true);
    let original = crate::bench::active_track_label(state.bench(id).unwrap())
        .expect("the double-click made the staged track active")
        .to_string();
    let mut parts = Parts::with_state(state);

    parts.press_with(egui::Key::D, egui::Modifiers::COMMAND, false);

    let bench = parts.state.bench(id).unwrap();
    assert_eq!(bench.entries().len(), 2, "Ctrl+D put no copy on the bench");
    let copy = crate::bench::active_track_label(bench).unwrap().to_string();
    assert_ne!(copy, original, "the copy is not the active track");
    let rows = texts(&parts.state);
    let row = format!("Duplicated {original} as {copy} ");
    assert!(rows.iter().any(|t| t.starts_with(&row)), "{rows:?}");
}

/// With nothing active on the bench there is nothing to duplicate, and Ctrl+D
/// writes nothing.
#[test]
fn ctrl_d_does_nothing_without_an_active_track() {
    let mut parts = Parts::new();
    let before = texts(&parts.state);

    parts.press_with(egui::Key::D, egui::Modifiers::COMMAND, false);

    assert_eq!(texts(&parts.state), before);
}
