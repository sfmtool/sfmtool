// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Scene Graph panel and the multi-node state machine
//! behind it.
//!
//! egui needs no GPU to lay out a frame, so the panel tests run the real thing
//! through `Context::run_ui` (the `point_track_detail/tests.rs` pattern): the
//! tree is really built, `CollapsingState` really stores its expansion under
//! [`row_id`], and clicks are really delivered by pointer events. Clicks aim at
//! the rects the panel recorded on the previous frame
//! ([`SceneGraphPanel::hit_rect`]), because a collapsible virtualized tree has
//! no geometry a test could predict from outside.
//!
//! The rest — cache purge, selection fallback, the finer-selection invariant,
//! `[` / `]` stepping, label disambiguation — is `AppState` behaviour and needs
//! no frame at all.

use eframe::egui;
use nalgebra::Vector3;
use sfmtool_core::reconstruction::ObservationSource;
use sfmtool_core::{RotQuaternion, Se3Transform, SfmrReconstruction};

use super::{row_id, SceneGraphPanel};
use crate::align::{AlignOptions, AlignSource};
use crate::scene::{ImageRef, NodeTint, PointRef, SceneNode, TINT_PALETTE};
use crate::state::{AppState, CachedSiftFeatures};
use crate::viewer_3d::Viewer3D;

const VIEWPORT: egui::Vec2 = egui::vec2(320.0, 900.0);

// ── Fixtures ────────────────────────────────────────────────────────────

/// A demo reconstruction padded to `images` images named `<prefix>_<i>.jpg`, so
/// two nodes can be made to share image names or not, at will, and so a camera
/// list can be made longer than the panel shows at once.
///
/// Only the image *list* grows: nothing exercised here reads the per-image side
/// tables, and `SfmrReconstruction::demo` fixes the camera ring at 8.
fn recon_named(points: usize, images: usize, prefix: &str) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(points);
    let template = recon.images[0].clone();
    while recon.images.len() < images {
        recon.images.push(template.clone());
    }
    recon.images.truncate(images);
    for (i, image) in recon.images.iter_mut().enumerate() {
        image.name = format!("{prefix}_{i:03}.jpg");
    }
    recon
}

/// A node loaded from `path`, the way `File > Open` would build it.
fn file_node(path: &str, images: usize, prefix: &str) -> SceneNode {
    SceneNode::from_path(std::path::Path::new(path), recon_named(32, images, prefix))
}

/// State holding `n` file-backed nodes that all share image names — the
/// comparison case: the same shoot solved several times.
fn shared_shoot(n: usize) -> AppState {
    let mut state = AppState::new();
    for i in 0..n {
        state.append_node(file_node(&format!("/runs/run_{i}.sfmr"), 8, "IMG"));
    }
    // Land on the first node, as if the user had clicked back to it.
    let first = state.scene[0].id;
    state.select_recon(first);
    state
}

/// A node whose reconstruction has *distinct* camera poses — unlike
/// [`recon_named`], which clones one image's pose for every row. Alignment fits
/// on camera centres, so it needs a real camera ring.
fn posed_node(path: &str) -> SceneNode {
    SceneNode::from_path(std::path::Path::new(path), SfmrReconstruction::demo(64))
}

/// A similarity with a rotation, a translation and a scale change, so a fit
/// that only handles some of the three cannot pass.
fn known_similarity() -> Se3Transform {
    Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.3, -0.7, 0.6), 0.9).unwrap(),
        Vector3::new(4.0, -2.5, 1.25),
        2.0,
    )
}

/// `recon` with every point and camera pose put through `t`.
fn transformed(recon: &SfmrReconstruction, t: &Se3Transform) -> SfmrReconstruction {
    let mut out = recon.clone();
    for p in &mut out.points {
        p.position = t.apply_to_point(&p.position);
    }
    for image in &mut out.images {
        let (rotation, translation) = t.apply_to_camera_pose(
            &RotQuaternion::from_nalgebra(image.quaternion_wxyz),
            &image.translation_xyz,
        );
        image.quaternion_wxyz = *rotation.as_nalgebra();
        image.translation_xyz = translation;
    }
    out
}

/// Two nodes: `run_a` as loaded, `run_b` the same scene under
/// [`known_similarity`]. The comparison case the whole feature exists for.
fn misaligned_pair() -> AppState {
    let mut state = AppState::new();
    let a = posed_node("/runs/run_a.sfmr");
    let b_recon = transformed(&a.recon, &known_similarity());
    state.append_node(a);
    let mut b = SceneNode::from_path(std::path::Path::new("/runs/run_b.sfmr"), b_recon);
    b.label = "run_b".to_string();
    state.append_node(b);
    state
}

/// Worst distance between the node's points *as displayed* and `target`'s.
fn worst_display_error(node: &SceneNode, target: &SceneNode, target_frame: &Se3Transform) -> f64 {
    node.recon
        .points
        .iter()
        .zip(target.recon.points.iter())
        .map(|(s, t)| {
            (node.transform.apply_to_point(&s.position) - target_frame.apply_to_point(&t.position))
                .norm()
        })
        .fold(0.0, f64::max)
}

// ── Frame driving ───────────────────────────────────────────────────────

/// Run one frame of the panel with `events` delivered, returning its response.
fn run_frame(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
) -> super::SceneGraphResponse {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    };
    let mut response = None;
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            response = Some(panel.show(ui, state));
        });
    });
    response.expect("the panel ran")
}

/// A panel + context that have already settled: egui resolves hover and clicks
/// against the widget rects registered on the *previous* pass, so nothing can
/// be clicked until at least one frame has been laid out.
fn settled(state: &mut AppState) -> (SceneGraphPanel, egui::Context) {
    let mut panel = SceneGraphPanel::new();
    let ctx = egui::Context::default();
    for _ in 0..2 {
        run_frame(&mut panel, &ctx, state, Vec::new());
    }
    (panel, ctx)
}

fn press(pos: egui::Pos2, pressed: bool) -> egui::Event {
    button(pos, egui::PointerButton::Primary, pressed)
}

fn button(pos: egui::Pos2, button: egui::PointerButton, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::default(),
    }
}

/// Right-click the element recorded under `id`, opening its context menu. The
/// menu's own rows are laid out (and recorded) on the frames that follow.
fn open_context_menu(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    id: egui::Id,
) {
    let pos = panel
        .hit_rect(id)
        .unwrap_or_else(|| panic!("{id:?} was not drawn on the previous frame"))
        .center();
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, true)],
    );
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, false)],
    );
    // One more settling frame so the menu's contents register their rects.
    run_frame(panel, ctx, state, Vec::new());
}

/// Right-click a reconstruction row and open its `Align to ▸` submenu, leaving
/// the target buttons and the option radios laid out and clickable.
fn open_align_menu(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    node: crate::scene::ReconId,
) {
    open_context_menu(panel, ctx, state, row_id(node, "node_label"));
    click(panel, ctx, state, row_id(node, "align_menu"));
    run_frame(panel, ctx, state, Vec::new());
}

/// Right-click at an explicit position rather than at a recorded element, for
/// the parts of a row that are not a widget of their own.
fn right_click_at(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    pos: egui::Pos2,
) {
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, true)],
    );
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, false)],
    );
    run_frame(panel, ctx, state, Vec::new());
}

/// Whether a reconstruction row's context menu is on screen: its entries lay
/// their own rects out only on the frames the menu is actually shown.
fn context_menu_open(panel: &SceneGraphPanel, node: crate::scene::ReconId) -> bool {
    panel.hit_rect(row_id(node, "reset_transform")).is_some()
}

/// Hover, press, release on the element recorded under `id` — the three frames
/// egui needs to register a click. Returns the response of the frame the click
/// landed in.
fn click(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    id: egui::Id,
) -> super::SceneGraphResponse {
    let pos = panel
        .hit_rect(id)
        .unwrap_or_else(|| panic!("{id:?} was not drawn on the previous frame"))
        .center();
    click_at(panel, ctx, state, pos)
}

fn click_at(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    pos: egui::Pos2,
) -> super::SceneGraphResponse {
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    run_frame(panel, ctx, state, vec![press(pos, true)]);
    run_frame(panel, ctx, state, vec![press(pos, false)])
}

/// Whether a row was drawn at all: `CollapsingState` is only stored for a
/// header that actually ran.
fn drawn(ctx: &egui::Context, id: egui::Id) -> bool {
    egui::collapsing_header::CollapsingState::load(ctx, id).is_some()
}

fn set_open(ctx: &egui::Context, id: egui::Id, open: bool) {
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ctx, id, false);
    state.set_open(open);
    state.store(ctx);
}

// ── Tree structure ──────────────────────────────────────────────────────

#[test]
fn every_loaded_node_gets_a_row_with_its_camera_and_point_groups() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let (_panel, ctx) = settled(&mut state);

    for id in ids {
        assert!(drawn(&ctx, row_id(id, "node")), "node row missing");
        assert!(drawn(&ctx, row_id(id, "cameras")), "Cameras row missing");
        assert!(drawn(&ctx, row_id(id, "points")), "Points row missing");
    }
}

#[test]
fn the_camera_rows_appear_only_once_the_cameras_group_is_expanded() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    assert!(
        panel.hit_rect(row_id(id, "camera_0")).is_none(),
        "the collapsed Cameras group still laid out its rows"
    );
    set_open(&ctx, row_id(id, "cameras"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "camera_0")).is_some(),
        "expanding the Cameras group did not draw any camera rows"
    );
}

#[test]
fn the_camera_list_lays_out_only_the_visible_rows() {
    // 400 images against a 220px-tall list: a non-virtualized list would lay
    // out every one of them.
    let mut state = AppState::new();
    state.append_node(file_node("/runs/big.sfmr", 400, "IMG"));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "cameras"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let laid_out = (0..400)
        .filter(|i| panel.hit_rect(row_id(id, &format!("camera_{i}"))).is_some())
        .count();
    assert!(
        laid_out > 0 && laid_out < 60,
        "{laid_out} of 400 camera rows were laid out; the list is not virtualized"
    );
}

#[test]
fn the_patches_row_appears_only_for_a_node_that_carries_patch_data() {
    use ndarray::{Array2, Array4};

    let mut state = shared_shoot(2);
    let plain = state.scene[0].id;
    let patched = state.scene[1].id;
    {
        let recon = &mut state.scene[1].recon;
        let n = recon.points.len();
        recon.patch_u_halfvec_xyz = Some(Array2::<f32>::from_elem((n, 3), 0.1));
        recon.patch_v_halfvec_xyz = Some(Array2::<f32>::from_elem((n, 3), 0.1));
        recon.patch_bitmaps_y_x_rgba = Some(Array4::<u8>::from_elem((n, 8, 8, 4), 200));
    }
    let (panel, _ctx) = settled(&mut state);

    assert!(
        panel.hit_rect(row_id(patched, "patches_eye")).is_some(),
        "a node with patch bitmaps got no Patches row"
    );
    assert!(
        panel.hit_rect(row_id(plain, "patches_eye")).is_none(),
        "a node without patch bitmaps got a Patches row anyway"
    );
}

#[test]
fn the_infinity_mini_toggle_appears_only_when_the_node_has_points_at_infinity() {
    let mut state = shared_shoot(2);
    let none = state.scene[0].id;
    let some = state.scene[1].id;
    state.scene[1].recon.metadata.infinity_point_count = 12;
    let (panel, _ctx) = settled(&mut state);

    assert!(panel.hit_rect(row_id(some, "points_infinity")).is_some());
    assert!(panel.hit_rect(row_id(none, "points_infinity")).is_none());
}

#[test]
fn the_panel_glyphs_are_available_in_the_bundled_fonts() {
    // A glyph egui does not bundle renders as a replacement box rather than
    // failing, so nothing else here would notice.
    let ctx = egui::Context::default();
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("warm the font atlas");
    });
    let font = egui::FontId::proportional(14.0);
    // The selection accent bar is deliberately absent: it is painted, not
    // written, precisely because no bundled proportional glyph would do.
    for glyph in [
        super::EYE_GLYPH,
        super::CURSOR_GLYPH,
        super::INFINITY_GLYPH,
        // A plain letter, so this one cannot fail — kept in the list because
        // the reason it is a letter is precisely that no bundled pictograph
        // says "only this one" (see `SOLO_GLYPH`).
        super::SOLO_GLYPH,
    ] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&font, glyph)),
            "{glyph:?} is not in egui's bundled fonts and would render as a box"
        );
    }
}

// ── Clicks and toggles ──────────────────────────────────────────────────

#[test]
fn clicking_a_reconstruction_row_reports_it_as_the_selection() {
    let mut state = shared_shoot(2);
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "node_label"));
    assert_eq!(response.select_recon, Some(second));
    // The panel reports; `dock.rs` applies. Nothing was selected behind its back.
    assert_ne!(state.selected_recon, Some(second));
}

#[test]
fn clicking_a_camera_row_reports_the_image_it_names() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "cameras"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "camera_2"));
    assert_eq!(response.select_image, Some(ImageRef::new(id, 2)));
}

/// A camera row keeps working once the virtualized list has scrolled off its
/// first slice — its identity is the image index, not its place in whatever
/// slice is currently rendered.
#[test]
fn a_camera_row_still_selects_after_the_list_has_scrolled() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/long.sfmr", 200, "IMG"));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "cameras"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    // Scroll row 150 into view the only way the panel offers: a selection made
    // elsewhere.
    state.select_image(ImageRef::new(id, 150));
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "camera_150"));
    assert_eq!(response.select_image, Some(ImageRef::new(id, 150)));
}

#[test]
fn the_eye_and_cursor_toggles_write_through_to_the_node() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    assert!(state.scene[0].visible && state.scene[0].interactive);

    click(&mut panel, &ctx, &mut state, row_id(id, "node_eye"));
    assert!(!state.scene[0].visible, "the eye did not hide the node");
    assert!(
        state.scene[0].interactive,
        "the eye also flipped the interaction cursor"
    );

    click(&mut panel, &ctx, &mut state, row_id(id, "node_cursor"));
    assert!(
        !state.scene[0].interactive,
        "the cursor toggle did not turn interaction off"
    );
    assert!(
        !state.scene[0].visible,
        "the cursor toggle also moved the eye"
    );

    // And back again — a toggle that only worked once would still pass above.
    click(&mut panel, &ctx, &mut state, row_id(id, "node_eye"));
    assert!(state.scene[0].visible);
}

#[test]
fn the_group_eyes_drive_their_own_layers_only() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    click(&mut panel, &ctx, &mut state, row_id(id, "cameras_eye"));
    assert!(!state.scene[0].show_cameras);
    assert!(state.scene[0].show_points, "the Cameras eye hid the points");

    click(&mut panel, &ctx, &mut state, row_id(id, "points_eye"));
    assert!(!state.scene[0].show_points);
    assert!(
        state.scene[0].visible,
        "a group eye should not touch the master eye"
    );
}

#[test]
fn the_infinity_mini_toggle_drives_only_the_infinity_points() {
    let mut state = shared_shoot(1);
    state.scene[0].recon.metadata.infinity_point_count = 12;
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    click(&mut panel, &ctx, &mut state, row_id(id, "points_infinity"));
    assert!(!state.scene[0].show_points_at_infinity);
    assert!(
        state.scene[0].show_points,
        "the ∞ toggle also hid the finite points"
    );
}

#[test]
fn the_selected_reconstruction_is_marked_in_the_tree() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    // The accent bar is painted, not written, so what is recorded for it is the
    // rect it was painted into — present only on the row that carries it.
    assert!(
        panel.hit_rect(row_id(first, "node_selected_bar")).is_some(),
        "the selected node got no marker"
    );
    assert!(
        panel
            .hit_rect(row_id(second, "node_selected_bar"))
            .is_none(),
        "an unselected node was marked too"
    );

    state.select_recon(second);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel
            .hit_rect(row_id(second, "node_selected_bar"))
            .is_some()
            && panel.hit_rect(row_id(first, "node_selected_bar")).is_none(),
        "the marker did not follow the selection"
    );
}

/// The space the bar occupies is reserved on every row, marked or not, so the
/// names line up down the tree and the selection does not shove one sideways.
#[test]
fn the_selection_marker_takes_no_room_from_the_row_it_is_not_on() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    let (panel, _ctx) = settled(&mut state);

    assert_eq!(
        panel.hit_rect(row_id(first, "node_label")).unwrap().left(),
        panel.hit_rect(row_id(second, "node_label")).unwrap().left(),
        "the selected row starts at a different x from the unselected one",
    );
}

#[test]
fn the_points_group_shows_the_selected_point_id_and_never_a_listing() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "points"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "point_selected")).is_none(),
        "a selection row appeared with nothing selected"
    );

    state.select_point(PointRef::new(id, 7));
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "point_selected")).is_some(),
        "the selected point got no row"
    );

    // Clicking it re-selects the same point (useful after selecting elsewhere).
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "point_selected"));
    assert_eq!(response.select_point, Some(PointRef::new(id, 7)));
}

#[test]
fn hovering_a_camera_row_reports_it_for_cross_panel_hover() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "cameras"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let pos = panel.hit_rect(row_id(id, "camera_3")).unwrap().center();
    run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(pos)],
    );
    let response = run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(pos)],
    );
    assert!(response.has_pointer, "the panel did not claim the pointer");
    assert_eq!(response.hovered_image, Some(ImageRef::new(id, 3)));
}

#[test]
fn the_camera_list_scrolls_to_a_selection_made_elsewhere_but_not_to_its_own() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/long.sfmr", 200, "IMG"));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "cameras"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "camera_150")).is_none(),
        "row 150 is visible without scrolling, so this test proves nothing"
    );

    // A selection change from another panel scrolls the row into view.
    state.select_image(ImageRef::new(id, 150));
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "camera_150")).is_some(),
        "the selected row was not scrolled into view"
    );

    // With the selection unchanged the list stays where the user left it: a
    // second frame must not re-apply the scroll.
    let before = panel.hit_rect(row_id(id, "camera_150")).unwrap();
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(
        panel.hit_rect(row_id(id, "camera_150")),
        Some(before),
        "the list kept scrolling itself with the selection unchanged"
    );
}

/// The other half of the interaction-cursor contract: switching a node's picks
/// off must not make it uninspectable. The tree is the control surface, so a
/// display-only node can always be selected deliberately.
///
/// (The GPU half — `pickable == 0` produces no hover or selection from the
/// readback — is asserted in `scene_renderer/upload/tests.rs`.)
#[test]
fn a_non_interactive_node_can_still_be_selected_from_the_tree() {
    let mut state = shared_shoot(2);
    let second = state.scene[1].id;
    state.scene[1].interactive = false;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(second, "cameras"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "node_label"));
    assert_eq!(response.select_recon, Some(second));

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "camera_1"));
    assert_eq!(response.select_image, Some(ImageRef::new(second, 1)));
    assert!(
        !state.scene[1].interactive,
        "selecting from the tree quietly re-armed the node's picks"
    );
}

#[test]
fn an_empty_scene_draws_no_rows() {
    let mut state = AppState::new();
    let (panel, _ctx) = settled(&mut state);
    assert!(panel.hit_rect(egui::Id::new("anything")).is_none());
}

// ── The reconstruction row's context menu ───────────────────────────────

/// The row is one target all the way across, not just the name's own glyphs.
/// A right-click in the gap between the name and the counts is the natural one,
/// and it used to land on nothing at all.
#[test]
fn the_context_menu_opens_from_anywhere_along_the_reconstruction_row() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(b, "node_label")).expect("the row");
    // Far enough right to be past the name and short of the counts — bare row.
    let gap = egui::pos2(row.right() - row.width() / 3.0, row.center().y);
    right_click_at(&mut panel, &ctx, &mut state, gap);

    assert!(
        context_menu_open(&panel, b),
        "right-clicking the row away from its name opened no menu"
    );
}

/// And on the name itself, which is the part of the row a user actually aims
/// at. Worth its own test because a click there does *not* reach the row by
/// default: egui gives a bare label `Sense::click_and_drag()` so its text can be
/// selected, and a label drawn after the row wins every hit that lands on a
/// glyph. Aiming at the row's centre — as the tests around this one do — sails
/// straight past that, because the name is at its left end.
#[test]
fn the_context_menu_opens_on_the_reconstructions_name() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(b, "node_label")).expect("the row");
    // The name is drawn from the row's left edge, past the selection marker.
    let on_the_name = egui::pos2(row.left() + 12.0, row.center().y);

    // The same spot selects on a left click — checked first, because the menu
    // the right click opens lands under the pointer and would eat this one.
    let response = click_at(&mut panel, &ctx, &mut state, on_the_name);
    assert_eq!(
        response.select_recon,
        Some(b),
        "left-clicking the name did not select the reconstruction"
    );

    right_click_at(&mut panel, &ctx, &mut state, on_the_name);
    assert!(
        context_menu_open(&panel, b),
        "right-clicking the reconstruction's name opened no menu"
    );
}

/// The row spans the panel rather than wrapping the name, so there is no dead
/// strip along it for a click to fall into.
#[test]
fn the_reconstruction_row_is_as_wide_as_the_panel() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (panel, _ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(id, "node_label")).expect("the row");
    assert!(
        row.right() >= VIEWPORT.x - 20.0,
        "the row stops at {} in a {}-wide panel",
        row.right(),
        VIEWPORT.x,
    );
}

/// egui keys a popup on the id of the widget it hangs off. The row used to
/// carry an auto-generated one — a count of what had been laid out before it —
/// which the accent bar shifted the moment the row became the selection, so an
/// open menu lost its identity and silently stopped being drawn.
#[test]
fn the_context_menu_survives_the_rows_selection_changing_under_it() {
    let mut state = misaligned_pair();
    let a = state.scene[0].id;
    let b = state.scene[1].id;
    state.select_recon(a);
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    assert!(context_menu_open(&panel, b), "the menu never opened");

    // `Select` from this very menu does exactly this, so the menu has to be able
    // to outlive it.
    state.select_recon(b);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        context_menu_open(&panel, b),
        "the menu vanished when the row it belongs to became the selection"
    );
}

/// The other order: the row's selected state changed, and only *then* is the
/// menu asked for.
#[test]
fn the_context_menu_opens_on_a_row_whose_selection_just_changed() {
    let mut state = misaligned_pair();
    let a = state.scene[0].id;
    let b = state.scene[1].id;
    state.select_recon(a);
    let (mut panel, ctx) = settled(&mut state);

    // Click the row and apply the selection the way `dock.rs` does.
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    assert_eq!(response.select_recon, Some(b), "the row did not select");
    state.select_recon(b);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    assert!(
        context_menu_open(&panel, b),
        "no menu on a row that had just been selected"
    );
}

/// Double-click still frames the node, and a right-click is not a selection.
#[test]
fn the_row_keeps_select_on_click_and_zoom_on_double_click() {
    let mut state = shared_shoot(2);
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let pos = panel
        .hit_rect(row_id(second, "node_label"))
        .expect("the row")
        .center();
    click_at(&mut panel, &ctx, &mut state, pos);
    let response = click_at(&mut panel, &ctx, &mut state, pos);
    assert_eq!(
        response.zoom_to_node,
        Some(second),
        "double-clicking the row did not ask for a zoom-to-fit"
    );

    let response = run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![
            button(pos, egui::PointerButton::Secondary, true),
            button(pos, egui::PointerButton::Secondary, false),
        ],
    );
    assert_eq!(
        response.select_recon, None,
        "a right-click on the row selected it as well as opening its menu"
    );
}

/// The toggles keep their own clicks: the row-wide target must not swallow
/// them, and they are not a place the node's menu comes from either.
#[test]
fn the_eye_and_cursor_toggles_are_not_part_of_the_rows_target() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "node_eye"));
    assert!(!state.scene[0].visible, "the eye stopped hiding the node");
    assert_eq!(
        response.select_recon, None,
        "the row underneath the eye took the click too"
    );

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_cursor"));
    assert!(
        !context_menu_open(&panel, id),
        "right-clicking a toggle opened the node's menu"
    );
}

// ── Align to… ───────────────────────────────────────────────────────────

#[test]
fn the_align_menu_lists_every_other_loaded_node() {
    let mut state = misaligned_pair();
    let a = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, a);

    assert!(
        panel.hit_rect(row_id(a, "align_to_run_b")).is_some(),
        "the other node is not offered as a target"
    );
    assert!(
        panel.hit_rect(row_id(a, "align_to_run_a")).is_none(),
        "a node was offered as a target for itself"
    );
}

#[test]
fn picking_a_target_reports_the_align_with_the_chosen_options() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, b);
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "align_to_run_a"));

    assert_eq!(
        response.align_node,
        Some((b, a, AlignOptions::default())),
        "the source, target or options did not reach the response"
    );
    // The panel reports; `dock.rs` applies. Nothing was fitted behind its back.
    assert!(!state.scene[1].has_transform());
}

#[test]
fn the_align_options_are_remembered_and_travel_with_the_request() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, b);
    click(&mut panel, &ctx, &mut state, row_id(b, "align_points"));
    click(&mut panel, &ctx, &mut state, row_id(b, "align_rigid"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "align_to_run_a"));

    assert_eq!(
        response.align_node,
        Some((
            b,
            a,
            AlignOptions {
                source: AlignSource::Points,
                estimate_scale: false,
            }
        )),
    );
}

#[test]
fn the_point_mode_is_disabled_without_feature_indexes_in_both_nodes() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    // Node A carries embedded keypoints, so there is no feature index for a
    // point correspondence to be keyed on.
    let tracks = state.scene[0].recon.tracks.len();
    let images = state.scene[0].recon.images.len();
    state.scene[0].recon.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: ndarray::Array2::<f32>::from_elem((tracks, 2), 100.0),
        image_file_hashes: vec![[0u8; 16]; images],
    };
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, b);
    click(&mut panel, &ctx, &mut state, row_id(b, "align_points"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "align_to_run_a"));

    let (.., options) = response.align_node.expect("the target is still clickable");
    assert_eq!(
        options.source,
        AlignSource::Cameras,
        "the disabled Points radio still switched the mode"
    );
}

#[test]
fn reset_transform_is_offered_only_once_a_node_has_one() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "reset_transform"));
    assert_eq!(
        response.reset_transform, None,
        "an untransformed node offered a reset that would do nothing"
    );

    state.align_node(b, a, AlignOptions::default());
    let (mut panel, ctx) = settled(&mut state);
    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "reset_transform"));
    assert_eq!(response.reset_transform, Some(b));
}

// ── Tint ────────────────────────────────────────────────────────────────

/// Open a reconstruction row's `Tint ▸` submenu, leaving its entries laid out
/// and clickable.
fn open_tint_menu(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    node: crate::scene::ReconId,
) {
    open_context_menu(panel, ctx, state, row_id(node, "node_label"));
    click(panel, ctx, state, row_id(node, "tint_menu"));
    run_frame(panel, ctx, state, Vec::new());
}

#[test]
fn the_tint_menu_offers_original_and_the_whole_palette() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    open_tint_menu(&mut panel, &ctx, &mut state, id);

    assert!(
        panel.hit_rect(row_id(id, "tint_original")).is_some(),
        "the Tint menu offers no way back to the original colors"
    );
    for color in TINT_PALETTE.iter() {
        assert!(
            panel
                .hit_rect(row_id(id, &format!("tint_{}", color.name)))
                .is_some(),
            "the palette entry {:?} was not drawn",
            color.name,
        );
    }
}

#[test]
fn picking_a_tint_writes_it_to_the_node_and_picking_original_takes_it_off() {
    let mut state = shared_shoot(2);
    let id = state.scene[0].id;
    let chosen = &TINT_PALETTE[5]; // Vermillion
    let (mut panel, ctx) = settled(&mut state);
    assert_eq!(state.scene[0].tint, NodeTint::Original);

    open_tint_menu(&mut panel, &ctx, &mut state, id);
    click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, &format!("tint_{}", chosen.name)),
    );

    // A tint is per-node display state, so unlike `Solo` it is written straight
    // into the node rather than reported for `dock.rs` to apply.
    assert_eq!(state.scene[0].tint, NodeTint::Tint(chosen));
    assert_eq!(
        state.scene[1].tint,
        NodeTint::Original,
        "tinting one node tinted its neighbour"
    );

    // The menu is still standing, which is the point of it: a color is chosen
    // by looking at the viewport, so the next entry has to be one click away.
    click(&mut panel, &ctx, &mut state, row_id(id, "tint_original"));
    assert_eq!(state.scene[0].tint, NodeTint::Original);
}

/// The swatch is how a tinted node is identifiable in the tree rather than only
/// in the viewport. Painted, not written, so what is recorded is the rect.
#[test]
fn only_a_tinted_row_gets_a_swatch() {
    let mut state = shared_shoot(2);
    let (first, second) = (state.scene[0].id, state.scene[1].id);
    state.scene[0].tint = NodeTint::Tint(&TINT_PALETTE[1]);
    let (panel, _ctx) = settled(&mut state);

    assert!(
        panel.hit_rect(row_id(first, "node_tint_swatch")).is_some(),
        "the tinted node got no swatch"
    );
    assert!(
        panel.hit_rect(row_id(second, "node_tint_swatch")).is_none(),
        "an untinted node was given a swatch anyway"
    );
    // Reserved on every row, marked or not, so a tint does not shove the name
    // sideways (the accent bar's rule).
    assert_eq!(
        panel.hit_rect(row_id(first, "node_label")).unwrap().left(),
        panel.hit_rect(row_id(second, "node_label")).unwrap().left(),
    );
}

/// Tinting is a display change and nothing else: walking through the menu must
/// not move the selection, least of all onto the row the menu belongs to.
#[test]
fn working_the_tint_menu_leaves_the_selection_where_it_was() {
    let mut state = shared_shoot(2);
    let (first, second) = (state.scene[0].id, state.scene[1].id);
    state.select_image(ImageRef::new(first, 2));
    let (mut panel, ctx) = settled(&mut state);

    open_tint_menu(&mut panel, &ctx, &mut state, second);
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(second, &format!("tint_{}", TINT_PALETTE[0].name)),
    );

    assert_eq!(response.select_recon, None, "the tint menu selected a node");
    assert_eq!(state.selected_recon, Some(first));
    assert_eq!(state.selected_image, Some(ImageRef::new(first, 2)));
    assert_eq!(state.scene[1].tint, NodeTint::Tint(&TINT_PALETTE[0]));
}

#[test]
fn a_reload_carries_the_tint_like_the_rest_of_the_display_state() {
    let mut state = shared_shoot(1);
    state.scene[0].tint = NodeTint::Tint(&TINT_PALETTE[2]);

    let mut reloaded = file_node("/runs/run_0.sfmr", 8, "IMG");
    reloaded.copy_display_from(&state.scene[0]);

    // The tint is how the user was telling this file apart from the one beside
    // it; a refresh that dropped it would undo that silently.
    assert_eq!(reloaded.tint, NodeTint::Tint(&TINT_PALETTE[2]));
}

#[test]
fn an_untinted_node_writes_the_original_colors_convention() {
    // `a == 0` is what every scene shader reads as "leave my colors alone".
    assert_eq!(NodeTint::Original.to_uniform(), [0.0; 4]);
    assert_eq!(NodeTint::Original.rgb(), None);

    let tint = NodeTint::Tint(&TINT_PALETTE[0]);
    let uniform = tint.to_uniform();
    assert_eq!(uniform[3], crate::scene::TINT_STRENGTH);
    assert!(
        uniform[3] > 0.0 && uniform[3] < 1.0,
        "a tint is a mix, not a repaint"
    );
    assert_eq!(tint.rgb(), Some(TINT_PALETTE[0].rgb));
}

/// The palette is meant to be told apart at a glance, so no two entries may be
/// near-identical — and none of them may be the background this viewer paints
/// on, which is what an eighth, black Okabe–Ito entry would have amounted to.
///
/// The floor is the palette's own closest pair, Orange and Vermillion, 82 apart
/// in summed channel distance; a crude metric, but enough to catch a duplicate
/// or a near-black entry added later.
#[test]
fn the_palette_entries_are_mutually_distinguishable() {
    for (i, a) in TINT_PALETTE.iter().enumerate() {
        let sum: u32 = a.rgb.iter().map(|&c| c as u32).sum();
        assert!(sum > 120, "{:?} is too dark to read as a tint", a.name);
        for b in TINT_PALETTE.iter().skip(i + 1) {
            let distance: u32 = (0..3)
                .map(|k| (a.rgb[k] as i32 - b.rgb[k] as i32).unsigned_abs())
                .sum();
            assert!(
                distance >= 82,
                "{:?} and {:?} are only {distance} apart",
                a.name,
                b.name,
            );
        }
    }
}

// ── Solo ────────────────────────────────────────────────────────────────

#[test]
fn the_solo_toggle_reports_the_row_it_sits_on() {
    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "node_solo"));
    assert_eq!(response.toggle_solo, Some(second));
    // The panel reports; `dock.rs` applies. Solo is app state, not node state,
    // so nothing was soloed behind its back — and no eye moved either.
    assert_eq!(state.solo, None);
    assert!(state.scene.iter().all(|n| n.visible));
}

/// Solo lives on the row, so the row-wide click target must not swallow it and
/// it must not be a place the node's menu comes from — the eye's contract.
#[test]
fn the_solo_toggle_is_not_part_of_the_rows_target() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "node_solo"));
    assert_eq!(response.toggle_solo, Some(id));
    assert_eq!(
        response.select_recon, None,
        "the row underneath the solo toggle took the click too"
    );

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_solo"));
    assert!(
        !context_menu_open(&panel, id),
        "right-clicking the solo toggle opened the node's menu"
    );
}

/// Clicking `S` on the soloed node ends the solo rather than re-soloing it.
#[test]
fn the_solo_toggle_reports_the_soloed_node_again_to_switch_it_off() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    state.toggle_solo(first);
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(first, "node_solo"));
    assert_eq!(response.toggle_solo, Some(first));
    state.toggle_solo(first); // what `dock.rs` does with it
    assert_eq!(state.solo, None);
}

// ── Solo state (no frame needed) ────────────────────────────────────────

/// Effective visibility is one rule — eye AND solo — and every consumer reads
/// it. Here it is directly.
#[test]
fn soloing_hides_every_other_node_without_touching_their_eyes() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    // One node the user had already hidden by hand, which is what makes
    // "restore what was there" a real requirement rather than "show everything".
    state.scene[2].visible = false;

    state.toggle_solo(second);
    assert!(!is_visible(&state.scene[0], state.solo));
    assert!(is_visible(&state.scene[1], state.solo));
    assert!(!is_visible(&state.scene[2], state.solo));
    // The eyes are untouched: solo overlays them rather than editing them.
    assert!(state.scene[0].visible && state.scene[1].visible && !state.scene[2].visible);

    state.toggle_solo(second);
    assert_eq!(state.solo, None);
    assert!(is_visible(&state.scene[0], state.solo));
    assert!(is_visible(&state.scene[1], state.solo));
    assert!(
        !is_visible(&state.scene[2], state.solo),
        "un-soloing revealed a node the user had hidden before the solo"
    );
}

#[test]
fn soloing_a_second_node_moves_the_solo_rather_than_adding_to_it() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(3);
    let (a, b) = (state.scene[0].id, state.scene[1].id);

    state.toggle_solo(a);
    state.toggle_solo(b);

    assert_eq!(state.solo, Some(b));
    assert!(!is_visible(&state.scene[0], state.solo));
    assert!(is_visible(&state.scene[1], state.solo));
}

#[test]
fn an_eye_toggled_while_soloed_takes_effect_when_the_solo_ends() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(2);
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    state.toggle_solo(a);

    // Switching the hidden node's eye off while it is soloed away: it changes
    // nothing on screen now, and everything the moment the solo ends.
    state.scene[1].visible = false;
    assert!(!is_visible(&state.scene[1], state.solo));

    state.toggle_solo(a);
    assert!(is_visible(&state.scene[0], state.solo));
    assert!(!is_visible(&state.scene[1], state.solo));

    // And the soloed node's own eye still applies while it is soloed: solo says
    // "hide the others", not "force this one on".
    state.toggle_solo(b);
    assert!(!is_visible(&state.scene[1], state.solo));
}

#[test]
fn closing_the_soloed_node_ends_the_solo() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    state.toggle_solo(second);

    state.close_node(second);

    // A solo naming a node that is gone would hide the whole scene, with
    // nothing left on screen to explain why.
    assert_eq!(state.solo, None);
    assert!(state.scene.iter().all(|n| is_visible(n, state.solo)));
}

#[test]
fn closing_another_node_leaves_the_solo_alone() {
    let mut state = shared_shoot(3);
    let (first, second) = (state.scene[0].id, state.scene[1].id);
    state.toggle_solo(second);

    state.close_node(first);
    assert_eq!(state.solo, Some(second));

    state.close_all();
    assert_eq!(state.solo, None);
}

#[test]
fn opening_a_file_ends_a_solo_so_the_new_node_is_not_born_hidden() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    state.toggle_solo(first);

    state.append_node(file_node("/runs/new.sfmr", 8, "IMG"));

    assert_eq!(state.solo, None);
}

#[test]
fn stepping_carries_an_active_solo_to_the_node_it_lands_on() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    // No solo, no solo: stepping never starts one.
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.solo, None);

    state.toggle_solo(ids[1]);
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[2]));
    assert_eq!(
        state.solo,
        Some(ids[2]),
        "the solo stayed behind on the node we stepped away from, so `]` \
         appeared to do nothing at all"
    );
}

// ── Transforms (no frame needed) ────────────────────────────────────────

#[test]
fn every_node_loads_in_its_own_frame() {
    let state = misaligned_pair();
    assert!(state.scene.iter().all(|n| !n.has_transform()));
}

#[test]
fn aligning_puts_the_source_node_where_the_target_is_drawn() {
    for source in [AlignSource::Cameras, AlignSource::Points] {
        let mut state = misaligned_pair();
        let (a, b) = (state.scene[0].id, state.scene[1].id);

        state.align_node(
            b,
            a,
            AlignOptions {
                source,
                estimate_scale: true,
            },
        );

        let error =
            worst_display_error(&state.scene[1], &state.scene[0], &Se3Transform::identity());
        assert!(
            error < 1e-9,
            "{source:?}: worst displayed error {error} after aligning run_b onto run_a"
        );
        assert!(state.scene[1].has_transform());
        // The target is never modified.
        assert!(!state.scene[0].has_transform());
    }
}

#[test]
fn an_align_lands_in_the_targets_currently_displayed_frame() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    // Pretend run_a was itself aligned onto something earlier, so it is already
    // displayed somewhere other than its own coordinates.
    let target_frame = Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), 1.1).unwrap(),
        Vector3::new(-7.0, 3.0, 0.5),
        0.25,
    );
    state.scene[0].transform = target_frame.clone();

    state.align_node(b, a, AlignOptions::default());

    // `source.transform = target.transform ∘ T_fit`: run_b lands on top of run_a
    // *as drawn*, so aligning C→B after B→A chains as expected.
    let error = worst_display_error(&state.scene[1], &state.scene[0], &target_frame);
    assert!(error < 1e-9, "worst displayed error {error} after chaining");
}

#[test]
fn an_aligned_nodes_cameras_are_looked_through_where_they_are_drawn() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    state.align_node(b, a, AlignOptions::default());

    // The pose `enter_camera_view` builds its end state from. run_b's camera 3,
    // through run_b's transform, has to land on run_a's camera 3 — same photo,
    // two solves, one viewpoint — or "look through this camera" would show the
    // transformed scene from an untransformed viewpoint.
    let (rotation, centre) = crate::viewer_3d::transformed_pose(
        &state.scene[1].recon.images[3],
        &state.scene[1].transform,
    );
    let expected = &state.scene[0].recon.images[3];
    assert!(
        (centre - expected.camera_center()).norm() < 1e-9,
        "camera centre {centre:?} is not the target's {:?}",
        expected.camera_center(),
    );
    assert!(
        rotation.angle_to(&expected.quaternion_wxyz) < 1e-9,
        "the composed orientation is off by {} rad",
        rotation.angle_to(&expected.quaternion_wxyz),
    );
}

#[test]
fn the_status_message_reports_the_fit() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);

    state.align_node(b, a, AlignOptions::default());

    let status = state.status_message.as_deref().expect("a status message");
    assert!(
        status.starts_with("Aligned run_b → run_a: ") && status.contains(" cameras, RMS "),
        "unexpected status line: {status}"
    );
}

#[test]
fn a_failed_align_leaves_the_transform_alone_and_says_why() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    for (i, image) in state.scene[1].recon.images.iter_mut().enumerate() {
        image.name = format!("unrelated_{i:03}.jpg");
    }

    state.align_node(b, a, AlignOptions::default());

    assert!(
        !state.scene[1].has_transform(),
        "a failed fit still moved the node"
    );
    let status = state.status_message.as_deref().expect("a status message");
    assert!(
        status.starts_with("Align run_b → run_a failed: "),
        "unexpected status line: {status}"
    );
}

#[test]
fn aligning_a_node_to_itself_does_nothing() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    state.align_node(b, b, AlignOptions::default());
    assert!(!state.scene[1].has_transform());
    assert_eq!(state.transform_epoch, 0);
}

#[test]
fn resetting_a_transform_returns_the_node_to_its_own_frame() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    state.align_node(b, a, AlignOptions::default());
    assert_eq!(state.transform_epoch, 1);

    state.reset_node_transform(b);

    assert!(!state.scene[1].has_transform());
    assert_eq!(state.scene[1].transform.scale, 1.0);
    // Both directions bump the epoch: the bounds and `length_scale` have to be
    // re-derived on the way back too.
    assert_eq!(state.transform_epoch, 2);
}

#[test]
fn a_reload_carries_the_nodes_transform_like_the_rest_of_its_display_state() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    state.align_node(b, a, AlignOptions::default());
    state.scene[1].show_points = false;

    // What `AppState::reload_node` does with the freshly-read node, minus the
    // disk read a test has no file for.
    let mut reloaded = posed_node("/runs/run_b.sfmr");
    reloaded.copy_display_from(&state.scene[1]);

    assert!(reloaded.has_transform(), "the reload reset the alignment");
    assert_eq!(
        reloaded.transform.scale, state.scene[1].transform.scale,
        "the reload changed the alignment"
    );
    assert!(!reloaded.show_points);
}

// ── Node lifecycle (no frame needed) ────────────────────────────────────

#[test]
fn labels_are_disambiguated_when_two_files_share_a_stem() {
    let mut state = AppState::new();
    state.append_node(file_node("/a/run.sfmr", 8, "IMG"));
    state.append_node(file_node("/b/run.sfmr", 8, "IMG"));
    state.append_node(file_node("/c/run.sfmr", 8, "IMG"));
    state.append_node(file_node("/d/other.sfmr", 8, "IMG"));

    let labels: Vec<_> = state.scene.iter().map(|n| n.label.as_str()).collect();
    assert_eq!(labels, ["run", "run (2)", "run (3)", "other"]);
}

#[test]
fn closing_a_node_purges_its_caches_and_selection() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    for id in [first, second] {
        state.sift_cache.insert(
            ImageRef::new(id, 0),
            CachedSiftFeatures {
                positions_xy: vec![[0.0, 0.0]],
                affine_shapes: vec![[[1.0, 0.0], [0.0, 1.0]]],
                read_count: 1,
            },
        );
        state.full_res_cache.insert(ImageRef::new(id, 0), None);
    }
    state.select_image(ImageRef::new(first, 3));
    state.hovered_point = Some(PointRef::new(first, 1));

    state.close_node(first);

    assert_eq!(state.scene.len(), 1);
    assert!(
        state.sift_cache.keys().all(|k| k.recon == second),
        "the SIFT cache kept entries for the closed node"
    );
    assert!(
        state.full_res_cache.keys().all(|k| k.recon == second),
        "the full-res cache kept entries for the closed node"
    );
    assert_eq!(state.selected_image, None, "selection outlived its node");
    assert_eq!(state.hovered_point, None, "hover outlived its node");
    // The other node's entries are untouched.
    assert!(state.sift_cache.contains_key(&ImageRef::new(second, 0)));
}

#[test]
fn closing_the_selected_node_falls_back_to_the_first_remaining() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    state.select_recon(ids[1]);

    state.close_node(ids[1]);
    assert_eq!(state.selected_recon, Some(ids[0]));

    // Closing an unselected node leaves the selection alone.
    state.close_node(ids[2]);
    assert_eq!(state.selected_recon, Some(ids[0]));

    // An empty scene means no selection at all.
    state.close_node(ids[0]);
    assert_eq!(state.selected_recon, None);
    assert!(state.scene.is_empty());
}

#[test]
fn close_all_empties_the_scene_and_every_shared_cache() {
    let mut state = shared_shoot(2);
    let id = state.scene[0].id;
    state.full_res_cache.insert(ImageRef::new(id, 0), None);
    state.select_image(ImageRef::new(id, 0));

    state.close_all();
    assert!(state.scene.is_empty());
    assert_eq!(state.selected_recon, None);
    assert_eq!(state.selected_image, None);
    assert!(state.full_res_cache.is_empty());
}

#[test]
fn selecting_a_reconstruction_clears_finer_selection_from_other_nodes() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    state.select_image(ImageRef::new(first, 2));
    state.selected_point = Some(PointRef::new(first, 5));
    // Hover is exempt — it is transient and may touch any visible node.
    state.hovered_image = Some(ImageRef::new(first, 9));

    state.select_recon(second);
    assert_eq!(state.selected_recon, Some(second));
    assert_eq!(state.selected_image, None);
    assert_eq!(state.selected_point, None);
    assert_eq!(
        state.hovered_image,
        Some(ImageRef::new(first, 9)),
        "hover was cleared, but it is exempt from the invariant"
    );

    // Re-selecting the node the selection already lives in changes nothing.
    state.select_image(ImageRef::new(second, 1));
    state.select_recon(second);
    assert_eq!(state.selected_image, Some(ImageRef::new(second, 1)));
}

#[test]
fn selecting_an_image_or_point_selects_its_reconstruction_too() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    state.select_recon(first);

    state.select_image(ImageRef::new(second, 4));
    assert_eq!(state.selected_recon, Some(second));

    state.select_point(PointRef::new(first, 4));
    assert_eq!(state.selected_recon, Some(first));
    assert_eq!(
        state.selected_image, None,
        "the image selection belonged to the node we just left"
    );
}

#[test]
fn opening_a_file_appends_a_node_rather_than_replacing_the_scene() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/a.sfmr", 8, "IMG"));
    state.append_node(file_node("/runs/b.sfmr", 8, "IMG"));
    assert_eq!(state.scene.len(), 2);
    assert_eq!(state.selected_recon, Some(state.scene[1].id));
}

// ── `[` / `]` stepping ──────────────────────────────────────────────────

/// Deliver one `[` or `]` press to `handle_recon_step`, the way `dock.rs` does.
fn step(viewer: &mut Viewer3D, ctx: &egui::Context, state: &mut AppState, forward: bool) {
    let key = if forward {
        egui::Key::CloseBracket
    } else {
        egui::Key::OpenBracket
    };
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events: vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }],
        ..Default::default()
    };
    crate::test_support::run_frame_headless(ctx, input, |ui| viewer.handle_recon_step(ui, state));
}

#[test]
fn bracket_keys_step_the_selected_reconstruction_in_tree_order() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[1]));
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[2]));
    // Wraps.
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[0]));
    step(&mut viewer, &ctx, &mut state, false);
    assert_eq!(state.selected_recon, Some(ids[2]));
}

#[test]
fn stepping_carries_the_selection_to_the_same_named_image() {
    let mut state = shared_shoot(2);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    // Shift the second node's images by one, so the same name sits at a
    // different index — an index-based carry-over would land on the wrong photo.
    state.scene[1].recon.images.rotate_left(1);
    let name = state.scene[0].recon.images[3].name.clone();
    let expected = state.scene[1]
        .recon
        .images
        .iter()
        .position(|i| i.name == name)
        .expect("the name is present in the second node");
    assert_ne!(
        expected, 3,
        "the fixture did not actually shift the indices"
    );

    state.select_image(ImageRef::new(ids[0], 3));
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    step(&mut viewer, &ctx, &mut state, true);

    assert_eq!(state.selected_recon, Some(ids[1]));
    assert_eq!(state.selected_image, Some(ImageRef::new(ids[1], expected)));
}

#[test]
fn stepping_clears_the_finer_selection_when_the_name_is_absent() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/a.sfmr", 16, "LEFT"));
    state.append_node(file_node("/runs/b.sfmr", 16, "RIGHT"));
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    state.select_recon(ids[0]);
    state.select_image(ImageRef::new(ids[0], 2));
    state.selected_point = Some(PointRef::new(ids[0], 5));

    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    step(&mut viewer, &ctx, &mut state, true);

    assert_eq!(state.selected_recon, Some(ids[1]));
    assert_eq!(state.selected_image, None);
    assert_eq!(state.selected_point, None);
}

#[test]
fn stepping_carries_the_camera_view_to_the_same_named_image() {
    let mut state = shared_shoot(2);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    // Enter camera view on the first node's image 3, without the animation:
    // `switch_camera_view` needs an active view, so seed one directly.
    let image = ImageRef::new(ids[0], 3);
    viewer.camera_view = Some(crate::viewer_3d::CameraViewMode {
        image,
        r_world_from_cam: state.scene[0].recon.images[3].quaternion_wxyz.inverse(),
    });
    state.select_image(image);

    step(&mut viewer, &ctx, &mut state, true);
    let now = viewer.camera_view.as_ref().expect("still in camera view");
    assert_eq!(now.image.recon, ids[1], "the camera view stayed behind");
    assert_eq!(
        state.scene[1].recon.images[now.image.index()].name,
        state.scene[0].recon.images[3].name,
    );
}

#[test]
fn stepping_drops_a_camera_view_whose_image_has_no_counterpart() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/a.sfmr", 16, "LEFT"));
    state.append_node(file_node("/runs/b.sfmr", 16, "RIGHT"));
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    state.select_recon(ids[0]);
    let image = ImageRef::new(ids[0], 1);
    state.select_image(image);
    let mut viewer = Viewer3D::new();
    viewer.camera_view = Some(crate::viewer_3d::CameraViewMode {
        image,
        r_world_from_cam: state.scene[0].recon.images[1].quaternion_wxyz.inverse(),
    });

    let ctx = egui::Context::default();
    step(&mut viewer, &ctx, &mut state, true);
    assert!(
        viewer.camera_view.is_none(),
        "the camera view kept pointing into the node we stepped away from"
    );
}

#[test]
fn stepping_does_nothing_with_a_single_node_loaded() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    state.select_image(ImageRef::new(id, 2));
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(id));
    assert_eq!(
        state.selected_image,
        Some(ImageRef::new(id, 2)),
        "stepping to the only node cleared the selection"
    );
}

// ── Window title and stats ──────────────────────────────────────────────

#[test]
fn the_window_title_names_the_first_file_and_counts_the_rest() {
    let mut state = AppState::new();
    assert_eq!(state.window_title(), "SfM Explorer");

    state.append_node(file_node("/runs/run_a.sfmr", 8, "IMG"));
    assert_eq!(state.window_title(), "SfM Explorer - run_a.sfmr");

    state.append_node(file_node("/runs/run_b.sfmr", 8, "IMG"));
    state.append_node(file_node("/runs/run_c.sfmr", 8, "IMG"));
    assert_eq!(state.window_title(), "SfM Explorer - run_a.sfmr (+2)");
}

#[test]
fn demo_data_first_leaves_the_base_window_title_alone() {
    // `ui_basic` attaches to the window by this exact name on Windows.
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(8)));
    assert_eq!(state.window_title(), "SfM Explorer");
    state.append_node(file_node("/runs/run_a.sfmr", 8, "IMG"));
    assert_eq!(state.window_title(), "SfM Explorer");
}

#[test]
fn the_stats_overlay_sums_visible_nodes_and_leads_with_the_count() {
    use crate::viewer_3d::overlay::scene_stats_text;

    let mut state = shared_shoot(2);
    state.scene[0].recon.metadata.infinity_point_count = 3;
    let one_node_points = state.scene[0].recon.points.len();
    let one_node_images = state.scene[0].recon.images.len();

    let text = scene_stats_text(&state.scene, None, false, 60.0);
    assert_eq!(
        text,
        format!(
            "2 reconstructions | {} points (3 at infinity) | {} images",
            2 * one_node_points,
            2 * one_node_images
        )
    );

    // Hiding a node takes it out of the totals — and with one left, the
    // reconstruction count leads no more.
    state.scene[1].visible = false;
    let text = scene_stats_text(&state.scene, None, false, 60.0);
    assert_eq!(
        text,
        format!("{one_node_points} points (3 at infinity) | {one_node_images} images")
    );
}

/// Solo reaches the totals the same way an eye does — the overlay describes
/// what is on screen, and the two switches are composed by one rule.
#[test]
fn the_stats_overlay_counts_only_the_soloed_node() {
    use crate::viewer_3d::overlay::scene_stats_text;

    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    let points = state.scene[1].recon.points.len();
    let images = state.scene[1].recon.images.len();

    state.toggle_solo(second);
    assert_eq!(
        scene_stats_text(&state.scene, state.solo, false, 60.0),
        format!("{points} points | {images} images"),
        "the stats line still counted the nodes the solo is hiding"
    );

    // A soloed node whose own eye is off is drawn by nobody.
    state.scene[1].visible = false;
    assert_eq!(
        scene_stats_text(&state.scene, state.solo, false, 60.0),
        "0 points | 0 images"
    );
}

#[test]
fn the_hover_overlay_names_the_reconstruction_only_when_several_are_loaded() {
    use crate::scene_renderer::PickTarget;
    use crate::viewer_3d::overlay::hover_overlay_text;

    let mut state = shared_shoot(1);
    let first = state.scene[0].id;
    let name = state.scene[0].recon.images[1].name.clone();

    let image_pick = Some(PickTarget::Image(ImageRef::new(first, 1)));
    let point_pick = Some(PickTarget::Point(PointRef::new(first, 88)));
    assert_eq!(
        hover_overlay_text(&state.scene, image_pick, None),
        format!("Camera: {name}")
    );
    assert_eq!(
        hover_overlay_text(&state.scene, point_pick, None),
        "Point3D #88"
    );

    state.append_node(file_node("/runs/run_b.sfmr", 8, "IMG"));
    assert_eq!(
        hover_overlay_text(&state.scene, image_pick, None),
        format!("Camera: run_0 / {name}")
    );
    assert_eq!(
        hover_overlay_text(&state.scene, point_pick, None),
        "Point3D run_0 #88"
    );
}

// ── Formatting helpers ──────────────────────────────────────────────────

#[test]
fn counts_are_formatted_compactly_on_the_node_row() {
    assert_eq!(super::compact_count(0), "0");
    assert_eq!(super::compact_count(999), "999");
    assert_eq!(super::compact_count(1_000), "1.0K");
    assert_eq!(super::compact_count(12_345), "12.3K");
    assert_eq!(super::compact_count(1_204_551), "1.2M");
}

#[test]
fn exact_counts_carry_thousands_separators_on_the_group_rows() {
    assert_eq!(super::with_thousands(0), "0");
    assert_eq!(super::with_thousands(999), "999");
    assert_eq!(super::with_thousands(1_000), "1,000");
    assert_eq!(super::with_thousands(1_204_551), "1,204,551");
}
