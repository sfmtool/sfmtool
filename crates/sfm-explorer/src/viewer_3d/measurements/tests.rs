// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless frames of the 3D view drawing the fixture's measurements
//! ([`crate::measurements::fixture`]): where the marks and lines land, what a
//! hover, a click and a double-click ask for, that a press on a mark is the
//! mark's, and the off-screen indicator.
//!
//! The camera looks along `+Y` from ten units back, with `+X` to the right
//! and `+Z` up, so the fixture's points on the `x` axis lie across the middle
//! of the view.

use eframe::egui;
use nalgebra::{Point3, Vector3};
use sfmtool_core::Camera;

use crate::measurement_marks::{DrawnId, MeasurementGesture};
use crate::measurements::fixture::{measured_state, POINTS};
use crate::measurements::{EndpointSide, MeasurementRef};
use crate::platform::ScrollInput;
use crate::scene::ReconId;
use crate::state::AppState;

use super::super::Viewer3D;

const VIEWPORT: egui::Vec2 = egui::vec2(1200.0, 800.0);

/// What one frame left behind: the strings it painted, the viewport's rect
/// and the cursor it asked for.
struct Frame {
    texts: Vec<String>,
    rect: egui::Rect,
    cursor: egui::CursorIcon,
}

/// One frame of the viewport on the fixture's node, its layer built as the
/// dock builds it.
fn frame(
    viewer: &mut Viewer3D,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
    pointer: egui::Pos2,
) -> Frame {
    viewer.measurements = state.measurement_layer(state.show_all_measurements);
    crate::platform::set_test_pointer_pos(Some(pointer));
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT)),
        events,
        // Half a second a frame, so clicks on separate frames are never a
        // double-click and the two of one frame are.
        time: Some(ctx.input(|i| i.time) + 0.5),
        ..Default::default()
    };
    let mut rect = egui::Rect::NOTHING;
    let mut output = ctx.run_ui(input, |ui| {
        let scroll_input = ScrollInput::from_ctx(ui.ctx(), false);
        egui::CentralPanel::default().show(ui, |ui| {
            let node = &state.scene[0];
            rect = ui.available_rect_before_wrap();
            viewer.show(
                ui,
                node,
                &state.scene,
                state.solo,
                None,
                state.selected_image,
                false,
                state.length_scale,
                None,
                &[],
                &scroll_input,
                false,
                false,
                false,
                None,
                None,
                None,
                None,
                None,
                &mut state.action_log,
            );
        });
    });
    output.textures_delta.clear();
    Frame {
        texts: crate::test_support::output_texts(&output),
        rect,
        cursor: output.platform_output.cursor_icon,
    }
}

/// A viewer looking along `+Y` from `eye`, with Z up, and a context that has
/// laid out twice with the pointer at `pointer`.
fn looking_from(
    state: &mut AppState,
    eye: Point3<f64>,
    pointer: egui::Pos2,
) -> (Viewer3D, egui::Context) {
    let mut viewer = Viewer3D::new();
    viewer.view_initialized = true;
    viewer.maintain_z_up = false;
    viewer.camera.camera = Camera::look_at(eye, eye + Vector3::y(), Vector3::z());
    let ctx = egui::Context::default();
    for _ in 0..2 {
        frame(
            &mut viewer,
            &ctx,
            state,
            vec![egui::Event::PointerMoved(pointer)],
            pointer,
        );
    }
    (viewer, ctx)
}

fn press(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

/// Click `times` times at `pos`, all in one frame as a double-click arrives.
fn click(
    viewer: &mut Viewer3D,
    ctx: &egui::Context,
    state: &mut AppState,
    pos: egui::Pos2,
    times: usize,
) {
    frame(
        viewer,
        ctx,
        state,
        vec![egui::Event::PointerMoved(pos)],
        pos,
    );
    let events = (0..times)
        .flat_map(|_| [press(pos, true), press(pos, false)])
        .collect();
    frame(viewer, ctx, state, events, pos);
}

/// Where the viewport projects `p`, in the shared world space.
fn projected(viewer: &Viewer3D, rect: egui::Rect, p: Point3<f64>) -> egui::Pos2 {
    viewer
        .camera
        .project_homogeneous(p.coords, 1.0, rect)
        .expect("in front of the eye")
}

fn eye() -> Point3<f64> {
    Point3::new(0.0, -10.0, 0.0)
}

fn select(state: &mut AppState, id: ReconId, index: usize) -> MeasurementRef {
    let m = MeasurementRef::distance(id, index);
    state.select_measurement(m);
    m
}

/// The marks and the line are where the viewport projects the endpoints'
/// positions put through the node's display transform, not their own.
#[test]
fn marks_and_lines_are_projected_through_the_display_transform() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    state.scene[0].history.transform_mut().scale = 2.0;
    state.scene[0].history.transform_mut().translation = Vector3::new(0.5, 0.0, 1.0);
    let (viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let mut rect = egui::Rect::NOTHING;
    let mut screen = None;
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);
        screen = Some(viewer.measurement_screen(ui.painter(), rect));
    });
    let screen = screen.unwrap();
    let world = |p: [f64; 3]| Point3::new(2.0 * p[0] + 0.5, 2.0 * p[1], 2.0 * p[2] + 1.0);
    let (a, b) = (
        projected(&viewer, rect, world(POINTS[0])),
        projected(&viewer, rect, world(POINTS[1])),
    );
    let drawn = &screen.measurements[0];
    assert_eq!(drawn.marks.len(), 2);
    assert!(
        drawn.marks[0].pos.distance(a) < 1e-3,
        "{:?} != {a:?}",
        drawn.marks[0].pos
    );
    assert!(drawn.marks[1].pos.distance(b) < 1e-3);
    let [la, lb] = drawn.line.expect("both ends are finite");
    assert!(la.distance(a) < 1e-3 && lb.distance(b) < 1e-3);
    let own = projected(&viewer, rect, Point3::from(POINTS[1]));
    assert!(
        own.distance(b) > 20.0,
        "not drawn where the untransformed point is"
    );
    assert!(screen.indicator.is_none(), "in view: no indicator");
}

/// A refused endpoint and one at infinity draw nothing, and the other
/// endpoint still draws its mark, with no line.
#[test]
fn refused_and_infinite_endpoints_draw_nothing() {
    let (mut state, id) = measured_state(Some("m"));
    for index in [2, 3] {
        select(&mut state, id, index);
        let (viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
        let mut screen = None;
        crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);
            screen = Some(viewer.measurement_screen(ui.painter(), rect));
        });
        let drawn = &screen.unwrap().measurements[0];
        let sides: Vec<EndpointSide> = drawn.marks.iter().map(|m| m.side).collect();
        assert_eq!(sides, vec![EndpointSide::A], "measurement {index}");
        assert_eq!(drawn.line, None, "measurement {index}");
    }
}

/// Hovering a mark or the line shows the row text and the pointing hand;
/// off them, neither.
#[test]
fn hovering_a_mark_or_the_line_shows_the_row_text() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    let hover = state.measurement_layer(false).drawn[0].hover.clone();
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let rect = frame(
        &mut viewer,
        &ctx,
        &mut state,
        Vec::new(),
        egui::pos2(5.0, 5.0),
    )
    .rect;
    let a = projected(&viewer, rect, Point3::from(POINTS[0]));
    let b = projected(&viewer, rect, Point3::from(POINTS[1]));
    for at in [a, a.lerp(b, 0.5)] {
        frame(
            &mut viewer,
            &ctx,
            &mut state,
            vec![egui::Event::PointerMoved(at)],
            at,
        );
        let shown = frame(&mut viewer, &ctx, &mut state, Vec::new(), at);
        assert!(
            shown.texts.contains(&hover),
            "{hover:?} not in {:?}",
            shown.texts
        );
        assert_eq!(shown.cursor, egui::CursorIcon::PointingHand);
    }
    let off = a + egui::vec2(0.0, 60.0);
    frame(
        &mut viewer,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(off)],
        off,
    );
    let shown = frame(&mut viewer, &ctx, &mut state, Vec::new(), off);
    assert!(!shown.texts.contains(&hover));
}

/// A click on a mark asks to select that endpoint and requests no point pick
/// under it (G7); a click on the line asks to select the measurement; a
/// double-click on a mark asks to look through it.
#[test]
fn a_click_on_a_mark_or_line_is_the_measurement_s() {
    let (mut state, id) = measured_state(Some("m"));
    let m = select(&mut state, id, 0);
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let rect = frame(
        &mut viewer,
        &ctx,
        &mut state,
        Vec::new(),
        egui::pos2(5.0, 5.0),
    )
    .rect;
    let a = projected(&viewer, rect, Point3::from(POINTS[0]));
    let b = projected(&viewer, rect, Point3::from(POINTS[1]));

    click(&mut viewer, &ctx, &mut state, b, 1);
    assert_eq!(
        viewer.measurement_gesture.take(),
        Some(MeasurementGesture::SelectEndpoint(
            DrawnId::Stored(m),
            EndpointSide::B
        ))
    );
    assert_eq!(viewer.pending_click, None, "no pick under the mark");

    click(&mut viewer, &ctx, &mut state, a.lerp(b, 0.5), 1);
    assert_eq!(
        viewer.measurement_gesture.take(),
        Some(MeasurementGesture::Select(DrawnId::Stored(m)))
    );
    assert_eq!(viewer.pending_click, None);

    click(&mut viewer, &ctx, &mut state, a, 2);
    assert_eq!(
        viewer.measurement_gesture.take(),
        Some(MeasurementGesture::LookThrough(
            DrawnId::Stored(m),
            EndpointSide::A
        ))
    );

    // The same click off the measurement is the viewport's pick.
    let off = a + egui::vec2(0.0, 80.0);
    click(&mut viewer, &ctx, &mut state, off, 1);
    assert_eq!(viewer.measurement_gesture, None);
    assert!(
        viewer.pending_click.is_some(),
        "a click off the marks picks"
    );
}

/// A drag that starts on a mark is the mark's: the view does not orbit,
/// while the same drag off it does.
#[test]
fn a_drag_from_a_mark_does_not_orbit() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let rect = frame(
        &mut viewer,
        &ctx,
        &mut state,
        Vec::new(),
        egui::pos2(5.0, 5.0),
    )
    .rect;
    let a = projected(&viewer, rect, Point3::from(POINTS[0]));
    let drag = |viewer: &mut Viewer3D, state: &mut AppState, from: egui::Pos2| {
        let before = viewer.camera.camera.orientation;
        frame(
            viewer,
            &ctx,
            state,
            vec![egui::Event::PointerMoved(from)],
            from,
        );
        frame(viewer, &ctx, state, vec![press(from, true)], from);
        let mut at = from;
        for _ in 0..6 {
            at += egui::vec2(20.0, 10.0);
            frame(viewer, &ctx, state, vec![egui::Event::PointerMoved(at)], at);
        }
        frame(viewer, &ctx, state, vec![press(at, false)], at);
        viewer.finish_transition();
        before.angle_to(&viewer.camera.camera.orientation)
    };
    assert_eq!(
        drag(&mut viewer, &mut state, a),
        0.0,
        "a drag from a mark orbited"
    );
    assert_eq!(viewer.measurement_gesture, None, "a drag is not a click");
    let off = a + egui::vec2(0.0, 120.0);
    assert!(
        drag(&mut viewer, &mut state, off) > 1e-3,
        "a drag off the marks orbits"
    );
}

/// The indicator for an eye at `eye` looking along `+Y` at the fixture's
/// measurement 0, whose midpoint is at `(0.75, 0, 0)`.
fn indicator_from(eye: Point3<f64>) -> Option<crate::measurement_marks::Indicator> {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    let (viewer, ctx) = looking_from(&mut state, eye, egui::pos2(5.0, 5.0));
    let mut screen = None;
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);
        screen = Some(viewer.measurement_screen(ui.painter(), rect));
    });
    screen.unwrap().indicator
}

/// The indicator is absent while any part of the measurement is in view,
/// and points toward the midpoint to the right, the left, below and above,
/// and for a midpoint behind the eye, the way to turn.
#[test]
fn the_indicator_points_toward_the_measurement() {
    assert!(indicator_from(eye()).is_none());
    // Only the line in view: the endpoints are off both sides, the line
    // crosses the middle.
    assert!(indicator_from(Point3::new(0.75, -0.3, 0.0)).is_none());
    let cases = [
        (
            Point3::new(-40.0, -10.0, 0.0),
            egui::vec2(1.0, 0.0),
            "right",
        ),
        (Point3::new(40.0, -10.0, 0.0), egui::vec2(-1.0, 0.0), "left"),
        (
            Point3::new(0.75, -10.0, 30.0),
            egui::vec2(0.0, 1.0),
            "below",
        ),
        (
            Point3::new(0.75, -10.0, -30.0),
            egui::vec2(0.0, -1.0),
            "above",
        ),
        // Behind the eye and to its right: the projection lands on the left,
        // and the arrow points right, the way to turn.
        (
            Point3::new(-5.75, 10.0, 0.0),
            egui::vec2(1.0, 0.0),
            "behind, to the right",
        ),
    ];
    for (eye, want, what) in cases {
        let indicator = indicator_from(eye).unwrap_or_else(|| panic!("{what}: no indicator"));
        assert!(
            indicator.direction.dot(want) > 0.99,
            "{what}: points {:?}",
            indicator.direction
        );
        assert_eq!(indicator.label, "Distance #0");
    }
}

/// A click on the indicator frames the measurement: after the transition
/// both endpoints project inside the view, and the indicator is gone.
#[test]
fn a_click_on_the_indicator_frames_the_measurement() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    let start = egui::pos2(5.0, 5.0);
    let (mut viewer, ctx) = looking_from(&mut state, Point3::new(-40.0, -10.0, 0.0), start);
    let shown = frame(&mut viewer, &ctx, &mut state, Vec::new(), start);
    let mut screen = None;
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        screen = Some(viewer.measurement_screen(ui.painter(), shown.rect));
    });
    let indicator = screen.unwrap().indicator.expect("off screen");
    assert!(
        shown.texts.contains(&"Distance #0".to_string()),
        "the label is drawn"
    );
    let orientation = viewer.camera.camera.orientation;
    click(
        &mut viewer,
        &ctx,
        &mut state,
        indicator.tip - indicator.direction * 6.0,
        1,
    );
    assert_eq!(viewer.measurement_gesture, None);
    viewer.finish_transition();
    assert!(
        orientation.angle_to(&viewer.camera.camera.orientation) < 1e-9,
        "framing keeps the orientation"
    );
    for p in [POINTS[0], POINTS[1]] {
        let at = projected(&viewer, shown.rect, Point3::from(p));
        assert!(shown.rect.contains(at), "{p:?} at {at:?}");
    }
    let shown = frame(&mut viewer, &ctx, &mut state, Vec::new(), start);
    assert!(
        !shown.texts.contains(&"Distance #0".to_string()),
        "the indicator is gone"
    );
}

/// With the option on every measurement is drawn, and a hidden
/// reconstruction draws none.
#[test]
fn the_option_draws_every_measurement_and_a_hidden_node_none() {
    let (mut state, _) = measured_state(Some("m"));
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let count = |viewer: &mut Viewer3D, state: &mut AppState| {
        frame(viewer, &ctx, state, Vec::new(), egui::pos2(5.0, 5.0));
        let mut screen = None;
        crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);
            screen = Some(viewer.measurement_screen(ui.painter(), rect));
        });
        screen.unwrap().measurements.len()
    };
    assert_eq!(count(&mut viewer, &mut state), 0);
    state.show_all_measurements = true;
    assert_eq!(count(&mut viewer, &mut state), 5);
    state.scene[0].visible = false;
    assert_eq!(count(&mut viewer, &mut state), 0);
}
