// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the viewport HUD and the input arbitration around it.
//!
//! egui needs no GPU to lay out a frame, so these drive the real thing:
//! `show_hud` followed by `Viewer3D::show`, through `Context::run_ui`, exactly
//! as `dock.rs` calls them. The assertions target what the HUD *decides* — is
//! it expanded, which sections did it draw, what rect did it claim — and what
//! the viewport does with pointer and keyboard input while the HUD is in the
//! way.
//!
//! Note that egui resolves hover, clicks and drags against the widget rects
//! registered on the *previous* pass, and an `Area` only settles at its
//! anchored position on its second frame. Every helper here therefore runs
//! several frames and only trusts the state left behind by the last one.

use eframe::egui;
use ndarray::{Array2, Array4};
use sfmtool_core::SfmrReconstruction;

use super::{section_id, HUD_COLLAPSE_GLYPH, HUD_EXPAND_GLYPH};
use crate::platform::ScrollInput;
use crate::scene::{ImageRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

const VIEWPORT: egui::Vec2 = egui::vec2(1200.0, 800.0);

// ── Fixtures ────────────────────────────────────────────────────────────

/// App state holding a plain demo reconstruction — no patch frames, so the
/// Patches section must stay away.
fn demo_state() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(64)));
    state
}

/// The id of the one node `demo_state` loads.
fn recon_id(state: &AppState) -> ReconId {
    state.selected_recon.expect("a selected reconstruction")
}

/// The same, plus the patch half-vectors and bitmaps the surfel pass needs, so
/// the Patches section turns on. The values are arbitrary — nothing here
/// renders them, only `has_patch_data` looks.
fn patch_state() -> AppState {
    let mut state = demo_state();
    let recon = &mut state.scene[0].recon;
    let n = recon.points.len();
    recon.patch_u_halfvec_xyz = Some(Array2::<f32>::from_elem((n, 3), 0.1));
    recon.patch_v_halfvec_xyz = Some(Array2::<f32>::from_elem((n, 3), 0.1));
    recon.patch_bitmaps_y_x_rgba = Some(Array4::<u8>::from_elem((n, 8, 8, 4), 200));
    state
}

/// Somewhere no widget and no viewport rect reaches, so a frame that does not
/// say where the pointer is is a frame with the pointer out of the way.
const POINTER_PARKED: egui::Pos2 = egui::pos2(-100.0, -100.0);

/// One frame of input.
struct Frame {
    events: Vec<egui::Event>,
    pointer: egui::Pos2,
    /// Simulate a widget holding the keyboard (a `DragValue` being typed into).
    grab_keyboard: bool,
    viewport: Option<egui::Vec2>,
}

impl Frame {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            pointer: POINTER_PARKED,
            grab_keyboard: false,
            viewport: None,
        }
    }

    fn at(mut self, pos: egui::Pos2) -> Self {
        self.pointer = pos;
        self.events.push(egui::Event::PointerMoved(pos));
        self
    }

    fn button(mut self, pos: egui::Pos2, pressed: bool) -> Self {
        self.pointer = pos;
        self.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        });
        self
    }

    fn wheel(mut self, delta: egui::Vec2) -> Self {
        self.events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta,
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::default(),
        });
        self
    }

    fn key_down(mut self, key: egui::Key) -> Self {
        self.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        self
    }

    fn grabbing_keyboard(mut self) -> Self {
        self.grab_keyboard = true;
        self
    }

    fn sized(mut self, size: egui::Vec2) -> Self {
        self.viewport = Some(size);
        self
    }
}

/// Run one frame of `show_hud` + `Viewer3D::show`, wired the way `dock.rs`
/// wires them.
fn run_frame(viewer: &mut Viewer3D, ctx: &egui::Context, state: &mut AppState, frame: Frame) {
    let size = frame.viewport.unwrap_or(VIEWPORT);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), size)),
        events: frame.events,
        ..Default::default()
    };
    // `platform::pointer_in_rect` does not read egui's pointer — on Windows it
    // reads the position the window's pointer messages last recorded — so the
    // test has to place it there too, or every geometric gate would see the
    // pointer parked at the origin.
    crate::platform::set_test_pointer_pos(Some(frame.pointer));
    let grab_keyboard = frame.grab_keyboard;
    let _ = ctx.run_ui(input, |ui| {
        if grab_keyboard {
            ui.ctx()
                .memory_mut(|m| m.request_focus(egui::Id::new("a_widget_being_typed_into")));
        }
        let scroll_input = ScrollInput::from_ctx(ui.ctx(), false);
        egui::CentralPanel::default().show_inside(ui, |ui| {
            viewer.show_hud(ui, state, Some((1, 2, 3, 4)), true);
            // The node borrows only `state.scene`, so the rest of `AppState`
            // stays reachable alongside it — the same split `dock.rs` relies on.
            let node = &state.scene[0];
            viewer.show(
                ui,
                &node.recon,
                node.id,
                &state.scene,
                &mut state.selected_image,
                state.show_grid,
                state.length_scale,
                &[],
                &scroll_input,
                state.show_controls_help,
                state.show_fps,
                None,
                None,
                None,
            );
        });
    });
}

/// Two settling frames: one to register widget rects, one for the HUD's `Area`
/// to reach its anchored position. Also absorbs the initial zoom-to-fit, so a
/// camera snapshot taken afterwards is stable.
fn warm_up(viewer: &mut Viewer3D, ctx: &egui::Context, state: &mut AppState) {
    for _ in 0..2 {
        run_frame(viewer, ctx, state, Frame::new());
    }
}

/// Hover, press, release at `pos` — the three frames egui needs to register a
/// click on a widget.
fn click_at(viewer: &mut Viewer3D, ctx: &egui::Context, state: &mut AppState, pos: egui::Pos2) {
    run_frame(viewer, ctx, state, Frame::new().at(pos));
    run_frame(viewer, ctx, state, Frame::new().button(pos, true));
    run_frame(viewer, ctx, state, Frame::new().button(pos, false));
}

/// A viewer + context + state that have already settled.
fn settled(state: &mut AppState) -> (Viewer3D, egui::Context) {
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    warm_up(&mut viewer, &ctx, state);
    (viewer, ctx)
}

/// Everything about the camera that any input path in this file can move.
#[derive(PartialEq, Debug)]
struct CameraSnapshot {
    position: [f64; 3],
    orientation: [f64; 4],
    distance: f64,
    fov: f64,
}

fn snapshot(viewer: &Viewer3D) -> CameraSnapshot {
    let p = viewer.camera.camera.position;
    let q = viewer.camera.camera.orientation;
    CameraSnapshot {
        position: [p.x, p.y, p.z],
        orientation: [q.w, q.i, q.j, q.k],
        distance: viewer.camera.camera.target_distance,
        fov: viewer.camera.fov,
    }
}

/// A point comfortably inside the HUD.
fn inside_hud(viewer: &Viewer3D) -> egui::Pos2 {
    viewer.hud_rect.expect("the HUD was built").center()
}

/// A point in the viewport well clear of the HUD and of the axis gizmo.
fn inside_viewport() -> egui::Pos2 {
    egui::pos2(400.0, 400.0)
}

// ── The egui behaviour the arbitration rests on ─────────────────────────

/// The spec leaves one rule resting on egui internals rather than our own
/// geometry: an `Area` on a higher layer is supposed to claim the pointer, so
/// the `allocate_painter` response underneath reports neither drag nor click.
/// It does hold in egui 0.34 — this pins it, because if it ever stops holding
/// the viewport would start orbiting while the user drags a HUD slider.
#[test]
fn an_area_on_a_higher_layer_swallows_drag_and_click_from_the_painter_beneath() {
    struct Probe {
        dragged: bool,
        clicked: bool,
        hovered: bool,
        area: egui::Rect,
    }

    fn probe_frame(ctx: &egui::Context, events: Vec<egui::Event>, with_area: bool) -> Probe {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            events,
            ..Default::default()
        };
        let mut probe = Probe {
            dragged: false,
            clicked: false,
            hovered: false,
            area: egui::Rect::NOTHING,
        };
        let _ = ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                if with_area {
                    let area = egui::Area::new(egui::Id::new("probe"))
                        .order(egui::Order::Middle)
                        .fixed_pos(egui::pos2(900.0, 20.0))
                        .show(ui.ctx(), |ui| {
                            ui.set_width(180.0);
                            egui::Frame::popup(ui.style()).show(ui, |ui| ui.label("hud"));
                        });
                    probe.area = area.response.rect;
                }
                let (response, _) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
                probe.dragged = response.dragged();
                probe.clicked = response.clicked();
                probe.hovered = response.hovered();
            });
        });
        probe
    }

    fn press(pos: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        }
    }

    for with_area in [false, true] {
        let ctx = egui::Context::default();
        probe_frame(&ctx, vec![], with_area);
        let probe = probe_frame(&ctx, vec![], with_area);
        // Aim at the middle of where the area settled (or the same spot with no
        // area at all, as the control).
        let start = if with_area {
            probe.area.center()
        } else {
            egui::pos2(950.0, 40.0)
        };
        let end = start + egui::vec2(40.0, 40.0);

        probe_frame(&ctx, vec![egui::Event::PointerMoved(start)], with_area);
        probe_frame(
            &ctx,
            vec![egui::Event::PointerMoved(start), press(start, true)],
            with_area,
        );
        let dragging = probe_frame(&ctx, vec![egui::Event::PointerMoved(end)], with_area);
        probe_frame(&ctx, vec![press(end, false)], with_area);

        let ctx = egui::Context::default();
        probe_frame(&ctx, vec![], with_area);
        probe_frame(&ctx, vec![egui::Event::PointerMoved(start)], with_area);
        probe_frame(&ctx, vec![press(start, true)], with_area);
        let clicking = probe_frame(&ctx, vec![press(start, false)], with_area);

        if with_area {
            assert!(!dragging.dragged, "the area let a drag through");
            assert!(!dragging.hovered, "the area let hover through");
            assert!(!clicking.clicked, "the area let a click through");
        } else {
            // The control: without the area the very same input does reach the
            // painter, so the assertions above are testing the layering and not
            // a mistake in how the frames are driven.
            assert!(dragging.dragged, "the bare painter should have dragged");
            assert!(
                clicking.clicked,
                "the bare painter should have been clicked"
            );
        }
    }
}

// ── Collapsed / expanded ────────────────────────────────────────────────

#[test]
fn the_hud_starts_open_with_its_sections_drawn() {
    let mut state = demo_state();
    let (viewer, ctx) = settled(&mut state);

    assert!(viewer.hud_open, "the HUD should open expanded");
    let rect = viewer.hud_rect.expect("the panel claims a rect");
    assert!(
        rect.width() > 200.0,
        "HUD opened at {} wide, which is a gear rather than the panel",
        rect.width()
    );
    assert!(
        egui::collapsing_header::CollapsingState::load(&ctx, section_id("layers")).is_some(),
        "an expanded HUD did not draw its sections"
    );
}

/// Collapsed, the sections are not merely closed but never drawn — so nothing
/// inside them can be hit, and their widgets cost nothing to skip. Starts from
/// a HUD that was never expanded, since egui remembers a section's open state
/// once drawn and collapsing the panel does not erase that memory.
#[test]
fn a_collapsed_hud_never_draws_its_sections() {
    let mut state = demo_state();
    let mut viewer = Viewer3D::new();
    viewer.hud_open = false;
    let ctx = egui::Context::default();
    warm_up(&mut viewer, &ctx, &mut state);

    assert!(
        egui::collapsing_header::CollapsingState::load(&ctx, section_id("layers")).is_none(),
        "a collapsed HUD drew its sections"
    );
}

/// Clicks the close button, then the gear it collapses to. Both directions in
/// one test because each needs the other's end state to start from.
#[test]
fn the_close_button_collapses_the_hud_and_the_gear_expands_it_again() {
    let mut state = demo_state();
    let (mut viewer, ctx) = settled(&mut state);

    // The close button sits at the panel's top-right, inside the frame margin.
    let expanded = viewer.hud_rect.expect("the panel claims a rect");
    let close = egui::pos2(expanded.right() - 14.0, expanded.top() + 16.0);
    click_at(&mut viewer, &ctx, &mut state, close);
    assert!(
        !viewer.hud_open,
        "the close button did not collapse the HUD"
    );

    // Let the gear settle, then confirm it claims no more than a gear's worth.
    warm_up(&mut viewer, &ctx, &mut state);
    let collapsed = viewer.hud_rect.expect("the gear still claims a rect");
    assert!(
        collapsed.width() < 60.0 && collapsed.height() < 60.0,
        "collapsed HUD claimed {collapsed:?}, which is more than a gear"
    );
    click_at(&mut viewer, &ctx, &mut state, collapsed.center());
    assert!(viewer.hud_open, "clicking the gear did not expand the HUD");
    warm_up(&mut viewer, &ctx, &mut state);
    assert!(
        viewer.hud_rect.expect("the panel is back").width() > 200.0,
        "the HUD did not return to the full panel"
    );
}

#[test]
fn a_viewport_too_small_for_the_panel_keeps_the_hud_collapsed() {
    let mut state = demo_state();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    viewer.hud_open = true;

    let tiny = egui::vec2(320.0, 240.0);
    for _ in 0..3 {
        run_frame(&mut viewer, &ctx, &mut state, Frame::new().sized(tiny));
    }

    let rect = viewer.hud_rect.expect("the gear is still there");
    assert!(
        rect.width() < 60.0,
        "the panel expanded into a {tiny:?} viewport and claimed {rect:?}"
    );
    assert!(
        viewer.hud_open,
        "the request to open should be remembered, only refused"
    );
}

// ── Anchoring ───────────────────────────────────────────────────────────

#[test]
fn the_hud_anchors_to_the_top_right_of_whatever_viewport_it_is_given() {
    let mut state = demo_state();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    let mut corner_for = |size: egui::Vec2| {
        for _ in 0..3 {
            run_frame(&mut viewer, &ctx, &mut state, Frame::new().sized(size));
        }
        viewer.hud_rect.expect("the HUD was built")
    };

    let wide = corner_for(egui::vec2(1200.0, 800.0));
    let narrow = corner_for(egui::vec2(700.0, 800.0));

    assert!(
        wide.right() > 1100.0,
        "HUD did not follow the wide viewport's right edge: {wide:?}"
    );
    assert!(
        narrow.right() < 700.0 && narrow.right() > 600.0,
        "HUD did not re-anchor when the viewport narrowed: {narrow:?}"
    );
    assert!(
        narrow.top() < 60.0,
        "HUD drifted away from the top edge: {narrow:?}"
    );
}

// ── Sections ────────────────────────────────────────────────────────────

#[test]
fn the_patches_section_is_omitted_when_the_reconstruction_has_no_patch_bitmaps() {
    for (name, mut state, expected) in [
        ("plain", demo_state(), false),
        ("with patches", patch_state(), true),
    ] {
        let mut viewer = Viewer3D::new();
        let ctx = egui::Context::default();
        viewer.hud_open = true;
        for _ in 0..3 {
            run_frame(&mut viewer, &ctx, &mut state, Frame::new());
        }

        // Every other section is drawn either way — otherwise this would pass
        // just as happily on a HUD that drew nothing at all.
        for always in ["layers", "size", "camera", "advanced", "debug"] {
            assert!(
                egui::collapsing_header::CollapsingState::load(&ctx, section_id(always)).is_some(),
                "{name}: the {always} section was missing"
            );
        }
        let drawn =
            egui::collapsing_header::CollapsingState::load(&ctx, section_id("patches")).is_some();
        assert_eq!(drawn, expected, "{name}: Patches section visibility");
    }
}

#[test]
fn the_size_sliders_write_through_to_app_state() {
    let mut state = patch_state();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    viewer.hud_open = true;
    for _ in 0..3 {
        run_frame(&mut viewer, &ctx, &mut state, Frame::new());
    }

    // The HUD reads and writes `AppState` in place rather than keeping its own
    // copies, so values set outside it survive a frame untouched.
    state.point_size_log2 = 1.5;
    state.patch_opacity = 0.25;
    state.edl_line_thickness = 4.0;
    state.target_fog_multiplier = 20.0;
    run_frame(&mut viewer, &ctx, &mut state, Frame::new());
    assert_eq!(state.point_size_log2, 1.5);
    assert_eq!(state.patch_opacity, 0.25);
    // The Advanced sliders had no widget at all before the HUD; a stale copy
    // in the HUD would quietly reset them every frame.
    assert_eq!(state.edl_line_thickness, 4.0);
    assert_eq!(state.target_fog_multiplier, 20.0);
}

#[test]
fn the_hud_glyphs_are_available_in_the_bundled_fonts() {
    // A glyph egui does not bundle renders as a replacement box rather than
    // failing, so nothing else here would notice. U+2715 MULTIPLICATION X — the
    // obvious close-button character — is one of those; this is the check that
    // says so.
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        ui.label("warm the font atlas");
    });
    let font = egui::FontId::proportional(14.0);
    for glyph in [HUD_EXPAND_GLYPH, HUD_COLLAPSE_GLYPH, "∞", "°"] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&font, glyph)),
            "{glyph:?} is not in egui's bundled fonts and would render as a box"
        );
    }
}

#[test]
fn the_defaults_the_debug_section_toggles_start_on() {
    // The two overlays the Debug section governs were unconditionally painted
    // before the HUD existed, so their toggles have to default to on or the
    // move would silently remove them.
    let state = AppState::new();
    assert!(state.show_controls_help);
    assert!(state.show_fps);
    assert!(state.show_points_at_infinity);
}

// ── Input arbitration ───────────────────────────────────────────────────

#[test]
fn scrolling_over_the_hud_does_not_zoom_the_viewport() {
    let mut state = demo_state();
    let (mut viewer, ctx) = settled(&mut state);
    // Expand first, so the HUD is a panel-sized obstacle rather than a gear.
    viewer.hud_open = true;
    warm_up(&mut viewer, &ctx, &mut state);

    let before = snapshot(&viewer);
    let over_hud = inside_hud(&viewer);
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().at(over_hud).wheel(egui::vec2(0.0, 6.0)),
    );
    assert_eq!(
        snapshot(&viewer),
        before,
        "a wheel event over the HUD moved the camera"
    );

    // Control: the same wheel event over bare viewport does zoom, so the
    // assertion above is about the exclusion and not about the event never
    // arriving.
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new()
            .at(inside_viewport())
            .wheel(egui::vec2(0.0, 6.0)),
    );
    assert_ne!(
        snapshot(&viewer),
        before,
        "the same wheel event over the viewport should have zoomed"
    );
}

#[test]
fn dragging_over_the_hud_does_not_orbit_the_viewport() {
    let mut state = demo_state();
    let (mut viewer, ctx) = settled(&mut state);
    viewer.hud_open = true;
    warm_up(&mut viewer, &ctx, &mut state);

    let before = snapshot(&viewer);
    let start = inside_hud(&viewer);
    let end = start + egui::vec2(30.0, 30.0);
    run_frame(&mut viewer, &ctx, &mut state, Frame::new().at(start));
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().button(start, true),
    );
    run_frame(&mut viewer, &ctx, &mut state, Frame::new().at(end));
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().button(end, false),
    );
    assert_eq!(
        snapshot(&viewer),
        before,
        "a drag that started on the HUD orbited the camera"
    );

    // Control: the same gesture on bare viewport does orbit.
    let start = inside_viewport();
    let end = start + egui::vec2(30.0, 30.0);
    run_frame(&mut viewer, &ctx, &mut state, Frame::new().at(start));
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().button(start, true),
    );
    run_frame(&mut viewer, &ctx, &mut state, Frame::new().at(end));
    assert_ne!(
        snapshot(&viewer),
        before,
        "the same drag over the viewport should have orbited"
    );
}

#[test]
fn clicking_the_hud_does_not_reach_the_pick_buffer() {
    let mut state = demo_state();
    let (mut viewer, ctx) = settled(&mut state);
    viewer.hud_open = true;
    warm_up(&mut viewer, &ctx, &mut state);

    let target = inside_hud(&viewer);
    run_frame(&mut viewer, &ctx, &mut state, Frame::new().at(target));
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().button(target, true),
    );
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().button(target, false),
    );
    assert!(
        viewer.pending_click.is_none(),
        "a click on the HUD queued a pick, which would have deselected"
    );
    assert!(
        viewer.hover_pixel.is_none(),
        "the HUD leaked a hover position into the depth readback"
    );

    // Control: a click on bare viewport still queues a pick.
    let target = inside_viewport();
    run_frame(&mut viewer, &ctx, &mut state, Frame::new().at(target));
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().button(target, true),
    );
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().button(target, false),
    );
    assert!(
        viewer.pending_click.is_some(),
        "a click on the viewport should still request a pick"
    );
}

#[test]
fn fly_keys_are_disarmed_while_a_widget_holds_the_keyboard() {
    let mut state = demo_state();
    let (mut viewer, ctx) = settled(&mut state);

    // Baseline: W alone flies.
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().key_down(egui::Key::W),
    );
    assert!(viewer.fly_keys_held, "W should arm the fly keys");
    let before = snapshot(&viewer);

    // Typing the same W into a HUD `DragValue` must not.
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().key_down(egui::Key::W).grabbing_keyboard(),
    );
    assert!(
        !viewer.fly_keys_held,
        "W typed into a widget still armed the fly keys"
    );
    assert_eq!(
        snapshot(&viewer),
        before,
        "W typed into a widget moved the camera"
    );
}

#[test]
fn viewport_shortcuts_are_suppressed_while_a_widget_holds_the_keyboard() {
    let mut state = demo_state();
    let (mut viewer, ctx) = settled(&mut state);
    let id = recon_id(&state);
    state.selected_image = Some(ImageRef::new(id, 0));

    // Baseline: `.` steps the selected image.
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().key_down(egui::Key::Period),
    );
    assert_eq!(
        state.selected_image,
        Some(ImageRef::new(id, 1)),
        "'.' should step the image selection"
    );

    // The same keystroke typed into a widget must not.
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        Frame::new().key_down(egui::Key::Period).grabbing_keyboard(),
    );
    assert_eq!(
        state.selected_image,
        Some(ImageRef::new(id, 1)),
        "a typed '.' stepped the image selection"
    );
}
