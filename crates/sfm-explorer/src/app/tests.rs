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
    state.finish_background_tasks();
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
fn the_step_keys_walk_the_image_set_and_wrap() {
    let mut parts = Parts::new();
    let id = parts.recon();
    let n = parts.state.scene[0].recon().image_set.images.len();
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

/// With an item focused, Ctrl+D puts a copy beside it and focuses the copy,
/// as the toolbar's *Duplicate* does.
#[test]
fn ctrl_d_duplicates_the_focused_item() {
    let (mut state, id) = edits::adjustable_state();
    let point = PointRef::new(id, 11);
    super::apply_point_click(&mut state, point, false);
    super::apply_point_click(&mut state, point, true);
    let original = state
        .focused_item_label(id)
        .expect("the double-click focused the staged track")
        .to_string();
    let mut parts = Parts::with_state(state);

    parts.press_with(egui::Key::D, egui::Modifiers::COMMAND, false);

    let bench = parts.state.bench(id).unwrap();
    assert_eq!(bench.entries().len(), 2, "Ctrl+D put no copy on the bench");
    let copy = parts.state.focused_item_label(id).unwrap().to_string();
    assert_ne!(copy, original, "the copy is not the focused item");
    let rows = texts(&parts.state);
    let row = format!("Duplicated {original} as {copy} ");
    assert!(rows.iter().any(|t| t.starts_with(&row)), "{rows:?}");
}

/// With nothing focused on the bench there is nothing to duplicate, and Ctrl+D
/// writes nothing.
#[test]
fn ctrl_d_does_nothing_without_a_focused_item() {
    let mut parts = Parts::new();
    let before = texts(&parts.state);

    parts.press_with(egui::Key::D, egui::Modifiers::COMMAND, false);

    assert_eq!(texts(&parts.state), before);
}

/// In Surface Tiling Mode, Ctrl+D is the mode's duplicate: the copy's verdicts
/// are handed back to the thresholds, and once its evaluation has settled the
/// rows they turned `out` are dropped, as with the toolbar's button.
#[test]
fn ctrl_d_in_surface_tiling_mode_unpins_the_copy_and_drops_its_out_rows() {
    use crate::bench::tests::{put_with_rows_the_bars_refuse, Refused};
    let (mut state, id) = crate::bench::tests::state();
    let (label, refused) = put_with_rows_the_bars_refuse(&mut state, id, Refused::Some);
    assert_eq!(state.focused_item_label(id), Some(label.as_str()));
    let mut parts = Parts::with_state(state);
    parts.track_view.set_surface_tiling(true);

    parts.press_with(egui::Key::D, egui::Modifiers::COMMAND, false);
    // The frames after it: the copy's evaluation settles, and the drop runs.
    parts.state.settle_bench_evaluation();
    parts.state.drive_surface_tiling();

    let copy = parts.state.focused_item_label(id).unwrap().to_string();
    assert_ne!(copy, label);
    let rows = texts(&parts.state);
    let track = parts.state.bench_track(id, &copy).expect("the copy");
    assert!(track.observations.iter().all(|o| !o.pinned), "{rows:?}");
    assert_eq!(track.observations.len(), 3 - refused.len(), "{rows:?}");
    assert!(
        track
            .observations
            .iter()
            .all(|o| !refused.contains(&o.image)),
        "a row the thresholds turned out was kept: {rows:?}"
    );
    let handed = rows
        .iter()
        .position(|t| t.starts_with("Handed ") && t.contains(&copy))
        .unwrap_or_else(|| panic!("no unpin row: {rows:?}"));
    assert!(
        rows[handed + 1..]
            .iter()
            .any(|t| t.starts_with(&format!("Dropped {} out ", refused.len()))),
        "no drop after the unpin: {rows:?}"
    );
}

// ── What a double-click on a Bench row does to the 3D view ──────────────

/// The similarity the bench fixture's node is drawn under here, so a target
/// read in the node's own coordinates misses where the point is drawn.
fn framing() -> sfmtool_core::Se3Transform {
    sfmtool_core::Se3Transform::new(
        sfmtool_core::RotQuaternion::from_axis_angle(nalgebra::Vector3::new(0.3, -0.7, 0.6), 0.9)
            .unwrap(),
        nalgebra::Vector3::new(4.0, -2.5, 1.25),
        2.0,
    )
}

/// The bench fixture with its point staged and its node drawn under
/// [`framing`], and the label the point took.
fn staged_and_framed() -> (AppState, crate::scene::ReconId, String) {
    let (mut state, id) = crate::bench::tests::state();
    state.set_node_transform(id, framing()).expect("a reframe");
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    (state, id, label)
}

/// A viewer looking at `point` from a few units away, with `point` a little
/// off the middle of the view: inside the middle where a pan alone reaches
/// it, and off the target so that moving onto it is a motion.
fn looking_near(point: nalgebra::Point3<f64>) -> crate::viewer_3d::Viewer3D {
    let mut viewer = crate::viewer_3d::Viewer3D::new();
    viewer.panel_size = [1600, 900];
    viewer.camera.world_up = nalgebra::Vector3::z();
    let eye = point + nalgebra::Vector3::new(0.0, -6.0, 0.5);
    let aim = point + nalgebra::Vector3::new(0.4, 0.0, -0.3);
    viewer.camera.camera = sfmtool_core::Camera::look_at(eye, aim, nalgebra::Vector3::z());
    viewer
}

/// The track's own point on `id`'s bench, in the node's coordinates.
fn bench_position(
    state: &AppState,
    id: crate::scene::ReconId,
    label: &str,
) -> nalgebra::Point3<f64> {
    match &state.bench_track(id, label).expect("on the bench").stage {
        sfmtool_core::bench::Stage::Track(payload) => payload.position.expect("triangulated"),
        sfmtool_core::bench::Stage::Cluster(_) => panic!("{label} is a cluster"),
    }
}

/// A double-click on a staged point's Bench row ends the 3D view where a
/// double-click on that point in the viewport ends it -- the same position,
/// orientation and orbit target -- with the point put through the node's
/// transform, and still focuses the item and raises Track View on it.
#[test]
fn a_double_click_on_a_bench_row_moves_the_view_as_one_on_its_point_does() {
    let (mut state, id, label) = staged_and_framed();
    state.unfocus_bench_item();
    state.hide_panel(crate::dock::Tab::TrackView);
    let point = PointRef::new(id, crate::bench::tests::POINT as usize);
    let Some(crate::scene::WorldPoint::At(drawn)) = crate::scene::world_point(&state.scene, point)
    else {
        panic!("the fixture's point is a place");
    };

    // The viewport's double-click on the point.
    let mut in_viewport = looking_near(drawn);
    let start = in_viewport.camera.camera.clone();
    assert!(in_viewport.move_target_to(drawn, 0.0), "no viewport move");
    in_viewport.finish_transition();

    // The Scene tree's double-click on the point's Bench row.
    let mut from_row = looking_near(drawn);
    let position = state.bench(id).unwrap().entries().len() - 1;
    super::apply_bench_edit(&mut state, &mut from_row, id, position, 0.0);
    from_row.finish_transition();

    let (a, b) = (&in_viewport.camera.camera, &from_row.camera.camera);
    assert!(
        (b.position - start.position).norm() > 1e-3,
        "the view did not move"
    );
    assert!(
        (b.target() - drawn).norm() < 1e-9,
        "target {:?}",
        b.target()
    );
    // With the target, these two are the whole end state: the orbit distance
    // is the target's distance along the view direction.
    assert!(b.orientation.angle_to(&a.orientation) < 1e-9, "turned");
    assert!((b.position - a.position).norm() < 1e-9, "moved elsewhere");

    assert_eq!(state.focused_item_label(id), Some(label.as_str()));
    assert!(state.is_panel_open(crate::dock::Tab::TrackView));
}

/// The view goes to the bench track's own point, where the bench layer draws
/// its patch, and not to the stored point it came from once a hand edit has
/// moved the two apart.
#[test]
fn a_double_click_on_an_edited_bench_row_goes_to_the_moved_patch() {
    let (mut state, id, label) = staged_and_framed();
    let track = state.bench_track(id, &label).unwrap().clone();
    let site = track.observations[0].site().expect("a sighting");
    let edit = crate::bench::PatchEdit::TranslateToPixel {
        viewpoint: sfmtool_core::bench::Viewpoint::Observation(0),
        pixel: [site[0] + 4.0, site[1] - 3.0],
    };
    assert!(state.edit_bench_patch(id, &label, &edit).unwrap().changed);
    let moved = framing().apply_to_point(&bench_position(&state, id, &label));
    let point = PointRef::new(id, crate::bench::tests::POINT as usize);
    let Some(crate::scene::WorldPoint::At(stored)) = crate::scene::world_point(&state.scene, point)
    else {
        panic!("the fixture's point is a place");
    };
    assert!(
        (moved - stored).norm() > 1e-3,
        "the edit did not move the patch"
    );

    let mut viewer = looking_near(moved);
    let position = state.bench(id).unwrap().entries().len() - 1;
    super::apply_bench_edit(&mut state, &mut viewer, id, position, 0.0);
    viewer.finish_transition();

    assert!((viewer.camera.camera.target() - moved).norm() < 1e-9);
}

/// A cluster-stage item has no place in the world, so a double-click on its
/// row focuses it and leaves the 3D view where it was.
#[test]
fn a_double_click_on_a_cluster_row_leaves_the_view_alone() {
    let (mut state, id) = crate::bench::tests::state();
    let label = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
            None,
        )
        .expect("a pixel on the sensor")
        .label;
    state.unfocus_bench_item();
    let mut viewer = looking_near(nalgebra::Point3::origin());
    let start = viewer.camera.camera.clone();

    super::apply_bench_edit(&mut state, &mut viewer, id, 0, 0.0);
    viewer.finish_transition();

    assert_eq!(
        viewer.camera.camera.position, start.position,
        "the view moved"
    );
    assert_eq!(
        viewer.camera.camera.orientation, start.orientation,
        "the view turned"
    );
    assert_eq!(state.focused_item_label(id), Some(label.as_str()));
}
