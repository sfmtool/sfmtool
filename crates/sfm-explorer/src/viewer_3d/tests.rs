// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the viewport's point context menu.
//!
//! egui needs no GPU to lay out a frame, so these drive the real thing:
//! `Viewer3D::show` through `Context::run_ui`, with the pick the GPU would have
//! reported handed in as the parameter `dock.rs` hands it in as. What they
//! assert is what the menu *decides* -- whether it opened, on which point, and
//! which entry was chosen -- rather than anything about pixels.
//!
//! egui resolves clicks against the widget rects registered on the *previous*
//! pass, so every gesture here is several frames and only the state the last
//! one left behind is trusted.

use eframe::egui;
use sfmtool_core::SfmrReconstruction;

use super::{Viewer3D, EDIT_ON_BENCH_LABEL, RETRIANGULATE_POINT_LABEL};
use crate::platform::ScrollInput;
use crate::scene::{PointRef, SceneNode};
use crate::scene_renderer::PickTarget;
use crate::state::edits::PointGesture;
use crate::state::AppState;

const VIEWPORT: egui::Vec2 = egui::vec2(1200.0, 800.0);
/// Well inside the viewport, and far from the HUD's top-right corner.
const OVER_A_POINT: egui::Pos2 = egui::pos2(400.0, 500.0);

fn demo_state() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(64)));
    state
}

/// The point the pick reports under the cursor for these tests.
fn picked(state: &AppState) -> PointRef {
    PointRef::new(state.selected_recon.expect("a selected reconstruction"), 7)
}

/// One frame of `Viewer3D::show`, wired the way `dock.rs` wires it.
fn run_frame(
    viewer: &mut Viewer3D,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
    pointer: egui::Pos2,
    pick: Option<PickTarget>,
    busy: Option<&str>,
) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    };
    // `platform::pointer_in_rect` does not read egui's pointer, so the test has
    // to place it where the window would have.
    crate::platform::set_test_pointer_pos(Some(pointer));
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        let scroll_input = ScrollInput::from_ctx(ui.ctx(), false);
        egui::CentralPanel::default().show(ui, |ui| {
            let node = &state.scene[0];
            viewer.show(
                ui,
                node,
                &state.scene,
                state.solo,
                state.selected_image,
                state.show_grid,
                state.length_scale,
                None,
                &[],
                &scroll_input,
                state.show_controls_help,
                state.show_fps,
                None,
                None,
                pick,
                busy,
                &mut state.action_log,
            );
        });
    });
}

/// Move the pointer there and settle, so the viewport's own rect is registered.
fn settled() -> (Viewer3D, egui::Context, AppState) {
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    let mut state = demo_state();
    for _ in 0..2 {
        run_frame(
            &mut viewer,
            &ctx,
            &mut state,
            vec![egui::Event::PointerMoved(OVER_A_POINT)],
            OVER_A_POINT,
            None,
            None,
        );
    }
    viewer.point_menu = None;
    (viewer, ctx, state)
}

/// One secondary button event.
fn secondary(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Secondary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

/// Press and release the right button at one place: a click.
fn right_click(
    viewer: &mut Viewer3D,
    ctx: &egui::Context,
    state: &mut AppState,
    pick: Option<PickTarget>,
    busy: Option<&str>,
) {
    for events in [
        vec![egui::Event::PointerMoved(OVER_A_POINT)],
        vec![secondary(OVER_A_POINT, true)],
        vec![secondary(OVER_A_POINT, false)],
        Vec::new(),
    ] {
        run_frame(viewer, ctx, state, events, OVER_A_POINT, pick, busy);
    }
}

/// Click the menu entry `label`, which has to be on screen already.
fn click_entry(
    viewer: &mut Viewer3D,
    ctx: &egui::Context,
    state: &mut AppState,
    pick: Option<PickTarget>,
    busy: Option<&str>,
    label: &str,
) {
    let rect = viewer
        .menu_entry_rects
        .iter()
        .find(|(text, _)| *text == label)
        .map(|(_, rect)| *rect)
        .unwrap_or_else(|| panic!("the menu offered no {label}"));
    let at = rect.center();
    for events in [
        vec![egui::Event::PointerMoved(at)],
        vec![egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        }],
        vec![egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    ] {
        run_frame(viewer, ctx, state, events, at, pick, busy);
    }
}

#[test]
fn a_right_click_on_a_point_opens_its_menu_and_names_the_point() {
    let (mut viewer, ctx, mut state) = settled();
    let point = picked(&state);
    right_click(
        &mut viewer,
        &ctx,
        &mut state,
        Some(PickTarget::Point(point)),
        None,
    );
    assert_eq!(viewer.menu_point, Some(point));
    // The gesture asks for the point to be selected, which is `dock.rs`'s to
    // carry out, and it asks before any entry is chosen.
    assert_eq!(viewer.point_menu, Some(PointGesture::Opened(point)));
    let offered: Vec<&str> = viewer
        .menu_entry_rects
        .iter()
        .map(|(text, _)| *text)
        .collect();
    assert_eq!(offered, [EDIT_ON_BENCH_LABEL, RETRIANGULATE_POINT_LABEL]);
}

#[test]
fn a_right_click_on_nothing_opens_no_menu() {
    let (mut viewer, ctx, mut state) = settled();
    right_click(&mut viewer, &ctx, &mut state, None, None);
    assert_eq!(viewer.menu_point, None);
    assert_eq!(viewer.point_menu, None);
    assert!(viewer.menu_entry_rects.is_empty());
}

#[test]
fn a_right_drag_is_the_zoom_and_puts_no_menu_up() {
    let (mut viewer, ctx, mut state) = settled();
    let point = picked(&state);
    let pick = Some(PickTarget::Point(point));
    // Press here, release well past egui's own click threshold: a drag, which
    // is the viewport's zoom, and never a menu.
    let away = OVER_A_POINT + egui::vec2(0.0, 80.0);
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        vec![secondary(OVER_A_POINT, true)],
        OVER_A_POINT,
        pick,
        None,
    );
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(away)],
        away,
        pick,
        None,
    );
    run_frame(
        &mut viewer,
        &ctx,
        &mut state,
        vec![secondary(away, false)],
        away,
        pick,
        None,
    );
    run_frame(&mut viewer, &ctx, &mut state, Vec::new(), away, pick, None);
    assert_eq!(viewer.menu_point, None);
    assert_eq!(viewer.point_menu, None);
}

#[test]
fn choosing_edit_on_bench_reports_the_point_the_menu_opened_on() {
    let (mut viewer, ctx, mut state) = settled();
    let point = picked(&state);
    let pick = Some(PickTarget::Point(point));
    right_click(&mut viewer, &ctx, &mut state, pick, None);
    viewer.point_menu = None;
    click_entry(
        &mut viewer,
        &ctx,
        &mut state,
        pick,
        None,
        EDIT_ON_BENCH_LABEL,
    );
    assert_eq!(viewer.point_menu, Some(PointGesture::EditOnBench(point)));
}

#[test]
fn choosing_retriangulate_reports_the_point_the_menu_opened_on() {
    let (mut viewer, ctx, mut state) = settled();
    let point = picked(&state);
    let pick = Some(PickTarget::Point(point));
    right_click(&mut viewer, &ctx, &mut state, pick, None);
    viewer.point_menu = None;
    click_entry(
        &mut viewer,
        &ctx,
        &mut state,
        pick,
        None,
        RETRIANGULATE_POINT_LABEL,
    );
    assert_eq!(viewer.point_menu, Some(PointGesture::Retriangulate(point)));
}

/// The viewport's half of the double-click: the click is recorded with the
/// flag that says it was one, and the pick that decides what it lands on
/// arrives a frame later through the GPU readback. `app.rs` resolves the two
/// together (`apply_point_click`), which is where Edit on Bench is asked for.
#[test]
fn a_double_click_is_recorded_on_the_pending_click() {
    let (mut viewer, ctx, mut state) = settled();

    primary_clicks(&mut viewer, &ctx, &mut state, 1);
    assert!(viewer.pending_click.is_some(), "the click was not recorded");
    assert!(!viewer.pending_click_is_double, "one click read as two");

    // Taken the way the readback takes it, so the second gesture is recorded
    // into an empty slot exactly as it is in the viewer.
    viewer.pending_click = None;
    primary_clicks(&mut viewer, &ctx, &mut state, 2);
    assert!(viewer.pending_click.is_some(), "the click was not recorded");
    assert!(viewer.pending_click_is_double, "two clicks read as one");
}

/// `count` primary press/release pairs at one place, in one frame: what egui
/// counts as a single click, a double-click, and so on.
fn primary_clicks(viewer: &mut Viewer3D, ctx: &egui::Context, state: &mut AppState, count: usize) {
    let mut events = vec![egui::Event::PointerMoved(OVER_A_POINT)];
    for _ in 0..count {
        for pressed in [true, false] {
            events.push(egui::Event::PointerButton {
                pos: OVER_A_POINT,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
        }
    }
    run_frame(viewer, ctx, state, events, OVER_A_POINT, None, None);
}

#[test]
fn a_busy_node_greys_both_entries() {
    let (mut viewer, ctx, mut state) = settled();
    let point = picked(&state);
    let pick = Some(PickTarget::Point(point));
    let busy = Some("run_a is busy: Bundle adjust is still running.");
    right_click(&mut viewer, &ctx, &mut state, pick, busy);
    // Drawn rather than hidden, so a reader can see the operation exists and
    // read why it cannot run right now.
    let offered: Vec<&str> = viewer
        .menu_entry_rects
        .iter()
        .map(|(text, _)| *text)
        .collect();
    assert_eq!(offered, [EDIT_ON_BENCH_LABEL, RETRIANGULATE_POINT_LABEL]);
    viewer.point_menu = None;
    click_entry(
        &mut viewer,
        &ctx,
        &mut state,
        pick,
        busy,
        EDIT_ON_BENCH_LABEL,
    );
    assert_eq!(viewer.point_menu, None, "a greyed entry cannot be chosen");
}
