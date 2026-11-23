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

/// A line with one end behind the eye is clipped at the near plane: the end
/// in front keeps its mark and the line runs from it, and the end behind has
/// no mark.
#[test]
fn a_line_with_one_end_behind_the_eye_is_clipped() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    // Between the two points, looking along +X: B ahead, A behind.
    let eye = Point3::new(0.75, -0.2, 0.0);
    let mut viewer = Viewer3D::new();
    viewer.view_initialized = true;
    viewer.maintain_z_up = false;
    viewer.camera.camera = Camera::look_at(eye, eye + Vector3::x(), Vector3::z());
    let ctx = egui::Context::default();
    for _ in 0..2 {
        frame(
            &mut viewer,
            &ctx,
            &mut state,
            Vec::new(),
            egui::pos2(5.0, 5.0),
        );
    }
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);
    let mut screen = None;
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        screen = Some(viewer.measurement_screen(ui.painter(), rect));
    });
    let drawn = &screen.unwrap().measurements[0];
    let sides: Vec<EndpointSide> = drawn.marks.iter().map(|m| m.side).collect();
    assert_eq!(sides, vec![EndpointSide::B], "A is behind the eye");
    let [a, b] = drawn.line.expect("the line is clipped, not dropped");
    assert!(a.x.is_finite() && a.y.is_finite());
    assert!(b.distance(projected(&viewer, rect, Point3::from(POINTS[1]))) < 1e-3);
    assert!(
        viewer
            .camera
            .project_homogeneous(Point3::from(POINTS[0]).coords, 1.0, rect)
            .is_none(),
        "A projects nowhere"
    );
}

/// Ribbons a mark draws: a ring and its halo, of `RING_SEGMENTS` chords each,
/// and the dot.
const MARK_RIBBONS: usize = 2 * super::RING_SEGMENTS + 1;

/// `p` moved [`super::toward_eye_fraction`] of the way toward the viewer's eye
/// in `rect`, as the
/// pass takes it.
fn toward_eye(viewer: &Viewer3D, rect: egui::Rect, p: Point3<f64>) -> [f32; 4] {
    let eye = viewer.camera.position();
    let q = eye + (p - eye) * (1.0 - super::toward_eye_fraction(&viewer.camera, rect));
    [q.x as f32, q.y as f32, q.z as f32, 1.0]
}

/// Where the viewport projects a ribbon's endpoint.
fn on_screen(viewer: &Viewer3D, rect: egui::Rect, p: [f32; 4]) -> egui::Pos2 {
    projected(
        viewer,
        rect,
        Point3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])),
    )
}

fn assert_close(a: [f32; 4], b: [f32; 4], what: &str) {
    assert!(
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4),
        "{what}: {a:?} != {b:?}"
    );
}

/// The selected measurement's ribbons: the line and its halo between the two
/// endpoints, then each mark's ring and halo at its radius on screen, the
/// selected endpoint's larger, and its dot; in the node's colour and the
/// halo's, at the stroke's width in physical pixels. The frame builds the
/// same ribbons for the upload.
#[test]
fn a_measurement_draws_its_line_and_marks_as_ribbons() {
    use crate::measurement_marks::{
        HALO, LINE_WIDTH, MARK_DOT_RADIUS, MARK_RADIUS, SELECTED_MARK_SCALE, STORED_RGB,
    };
    let (mut state, id) = measured_state(Some("m"));
    let m = select(&mut state, id, 0);
    state.select_endpoint(m, EndpointSide::B);
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let shown = frame(
        &mut viewer,
        &ctx,
        &mut state,
        Vec::new(),
        egui::pos2(5.0, 5.0),
    );
    let rect = shown.rect;
    assert_eq!(
        viewer.measurement_ribbons,
        super::ribbons(&viewer.measurements, &viewer.camera, rect, 1.0),
        "the frame builds the ribbons the upload draws"
    );

    let ppp = 2.0;
    let ribbons = super::ribbons(&viewer.measurements, &viewer.camera, rect, ppp);
    assert_eq!(ribbons.len(), 2 + 2 * MARK_RIBBONS);
    let grey = super::linear_rgba(egui::Color32::from_rgb(
        STORED_RGB[0],
        STORED_RGB[1],
        STORED_RGB[2],
    ));
    let halo = super::linear_rgba(HALO);
    assert!((halo[3] - 170.0 / 255.0).abs() < 1e-6 && halo[..3] == [0.0; 3]);
    assert!(
        grey[3] == 1.0 && grey[0] > 0.9 && grey[0] < 0.92,
        "{grey:?}"
    );

    let (a, b) = (Point3::from(POINTS[0]), Point3::from(POINTS[1]));
    let [line_halo, line] = [ribbons[0], ribbons[1]];
    for (ribbon, color, width) in [
        (line_halo, halo, LINE_WIDTH + 2.0),
        (line, grey, LINE_WIDTH),
    ] {
        assert_close(ribbon.a, toward_eye(&viewer, rect, a), "line end A");
        assert_close(ribbon.b, toward_eye(&viewer, rect, b), "line end B");
        assert_eq!(ribbon.color, color);
        assert_eq!(ribbon.half_width, width / 2.0 * ppp);
    }

    for (k, (side, p)) in [(EndpointSide::A, a), (EndpointSide::B, b)]
        .into_iter()
        .enumerate()
    {
        let mark = &ribbons[2 + k * MARK_RIBBONS..2 + (k + 1) * MARK_RIBBONS];
        let centre = projected(&viewer, rect, p);
        let radius = if side == EndpointSide::B {
            MARK_RADIUS * SELECTED_MARK_SCALE
        } else {
            MARK_RADIUS
        };
        let (rings, dot) = mark.split_at(2 * super::RING_SEGMENTS);
        let (ring_halo, ring) = rings.split_at(super::RING_SEGMENTS);
        for (chords, color, width) in [
            (ring_halo, halo, LINE_WIDTH + 2.0),
            (ring, grey, LINE_WIDTH),
        ] {
            for chord in chords {
                for end in [chord.a, chord.b] {
                    let r = on_screen(&viewer, rect, end).distance(centre);
                    assert!((r - radius).abs() < 1e-2, "{side:?}: {r} != {radius}");
                }
                assert_eq!(chord.color, color);
                assert_eq!(chord.half_width, width / 2.0 * ppp);
            }
        }
        // The ring is closed: its last chord ends where its first begins.
        assert_close(ring[0].a, ring[super::RING_SEGMENTS - 1].b, "closed ring");
        let dot = dot[0];
        assert_eq!(dot.color, grey);
        assert_eq!(dot.half_width, MARK_DOT_RADIUS * ppp);
        for end in [dot.a, dot.b] {
            let r = on_screen(&viewer, rect, end).distance(centre);
            assert!((r - MARK_DOT_RADIUS).abs() < 1e-2, "dot: {r}");
        }
        // The mark is drawn toward the eye, at the depth its line's end is.
        let depth = |q: [f32; 4]| {
            -viewer
                .camera
                .world_to_view(&Point3::new(
                    f64::from(q[0]),
                    f64::from(q[1]),
                    f64::from(q[2]),
                ))
                .z
        };
        let expected = -viewer.camera.world_to_view(&p).z
            * (1.0 - super::toward_eye_fraction(&viewer.camera, rect));
        assert!((depth(ring[0].a) - expected).abs() < 1e-4);
    }
}

/// With *Show all measurements* on, the selected measurement's line is the
/// emphasised width and the others are at the plain width and 55% opacity;
/// the focused item is the bench violet at the bench's stroke.
#[test]
fn show_all_emphasises_the_selected_measurement_and_the_item_is_violet() {
    use crate::measurement_marks::{
        Role, EMPHASISED_LINE_WIDTH, ITEM_LINE_WIDTH, LINE_WIDTH, OTHER_OPACITY,
    };
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    state.show_all_measurements = true;
    let (viewer, _ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);
    let layer = &viewer.measurements;
    assert!(layer.drawn.len() > 1, "every measurement is drawn");
    let mut roles = Vec::new();
    for drawn in &layer.drawn {
        let one = crate::measurement_marks::MeasurementLayer {
            drawn: vec![drawn.clone()],
        };
        let ribbons = super::ribbons(&one, &viewer.camera, rect, 1.0);
        // A measurement with no finite endpoint draws nothing.
        let Some(stroke) = ribbons
            .iter()
            .find(|r| r.color != super::linear_rgba(crate::measurement_marks::HALO))
        else {
            continue;
        };
        roles.push(drawn.role);
        match drawn.role {
            Role::Selected => {
                assert_eq!(stroke.half_width, EMPHASISED_LINE_WIDTH / 2.0);
                assert_eq!(stroke.color[3], 1.0);
            }
            Role::Other => {
                assert_eq!(stroke.half_width, LINE_WIDTH / 2.0);
                assert!((stroke.color[3] - OTHER_OPACITY).abs() < 0.01, "{stroke:?}");
            }
            Role::Item => panic!("no item is focused"),
        }
    }
    assert!(roles.contains(&Role::Selected) && roles.contains(&Role::Other));

    // The item: the bench violet at the bench's stroke.
    let mut item = layer.drawn[0].clone();
    item.role = Role::Item;
    item.color = crate::bench::verdict_color(sfmtool_core::bench::Verdict::In);
    item.width = ITEM_LINE_WIDTH;
    let ribbons = super::ribbons(
        &crate::measurement_marks::MeasurementLayer { drawn: vec![item] },
        &viewer.camera,
        rect,
        1.0,
    );
    let violet = super::linear_rgba(crate::bench::verdict_color(
        sfmtool_core::bench::Verdict::In,
    ));
    assert_eq!(ribbons[1].color, violet);
    assert_eq!(ribbons[1].half_width, ITEM_LINE_WIDTH / 2.0);
}

/// A refused endpoint and one at infinity draw no ribbon, and the other
/// endpoint still draws its mark, with no line.
#[test]
fn refused_and_infinite_endpoints_draw_no_ribbons() {
    let (mut state, id) = measured_state(Some("m"));
    for index in [2, 3] {
        select(&mut state, id, index);
        let (viewer, _ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);
        let drawn = &viewer.measurements.drawn[0];
        let crate::measurement_marks::Place::Finite(a) = drawn.endpoints[0].place else {
            panic!("measurement {index}: A is finite");
        };
        assert!(
            !matches!(
                drawn.endpoints[1].place,
                crate::measurement_marks::Place::Finite(_)
            ),
            "measurement {index}: B is not"
        );
        let ribbons = super::ribbons(&viewer.measurements, &viewer.camera, rect, 1.0);
        assert_eq!(ribbons.len(), MARK_RIBBONS, "measurement {index}");
        let centre = projected(&viewer, rect, a);
        for ribbon in &ribbons {
            for end in [ribbon.a, ribbon.b] {
                assert!(
                    on_screen(&viewer, rect, end).distance(centre) < 10.0,
                    "measurement {index}: every piece is A's mark"
                );
            }
        }
    }
}

/// A line with one end behind the eye keeps that end, for the pass to clip
/// at the near plane, and the end behind draws no mark.
#[test]
fn a_line_with_one_end_behind_the_eye_is_handed_to_the_pass_whole() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    // Between the two points, looking along +X: B ahead, A behind.
    let eye = Point3::new(0.75, -0.2, 0.0);
    let mut viewer = Viewer3D::new();
    viewer.view_initialized = true;
    viewer.maintain_z_up = false;
    viewer.camera.camera = Camera::look_at(eye, eye + Vector3::x(), Vector3::z());
    viewer.measurements = state.measurement_layer(false);
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, VIEWPORT);

    let ribbons = super::ribbons(&viewer.measurements, &viewer.camera, rect, 1.0);
    assert_eq!(ribbons.len(), 2 + MARK_RIBBONS, "the line and B's mark");
    let (a, b) = (Point3::from(POINTS[0]), Point3::from(POINTS[1]));
    assert_close(
        ribbons[1].a,
        toward_eye(&viewer, rect, a),
        "A, behind the eye",
    );
    assert_close(ribbons[1].b, toward_eye(&viewer, rect, b), "B");
    assert!(
        viewer.camera.world_to_view(&a).z > 0.0,
        "A is behind the eye"
    );
    let centre = projected(&viewer, rect, b);
    for ribbon in &ribbons[2..] {
        assert!(on_screen(&viewer, rect, ribbon.a).distance(centre) < 10.0);
    }
}

/// The pull toward the eye is `PULL_POINTS` panel points at each piece's
/// depth, whatever the field of view and the panel's shape: measured against
/// the world length of one point at that depth, it is the same number of
/// points at 5, 45 and 160 degrees, in a short and a tall landscape panel and
/// in a portrait one, which a fixed fraction of the distance is not. It is at
/// most a quarter of the depth, and none in a panel with no height.
#[test]
fn the_pull_toward_the_eye_is_a_fixed_number_of_points_on_screen() {
    let mut camera = crate::viewer_3d::ViewportCamera::default();
    let depth = 7.0;
    let mut fractions = Vec::new();
    for fov_degrees in [5.0_f64, 45.0, 160.0] {
        camera.fov = fov_degrees.to_radians();
        for (width, height) in [(1200.0_f32, 800.0_f32), (2400.0, 1600.0), (1200.0, 1600.0)] {
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
            let c = rect.center();
            let point_length = (camera.unproject(c.x, c.y + 1.0, depth, rect)
                - camera.unproject(c.x, c.y, depth, rect))
            .norm();
            let fraction = super::toward_eye_fraction(&camera, rect);
            let points = fraction * depth / point_length;
            assert!(
                (points - super::PULL_POINTS).abs() < 1e-3,
                "{fov_degrees} degrees, {width} by {height} points: {points} points"
            );
            assert!(fraction > 0.0 && fraction < 0.25);
            fractions.push(fraction);
        }
    }
    // Wider fields and shorter panels pull further, as a share of the depth.
    assert!(fractions.windows(2).all(|w| w[0] != w[1]));
    assert!(fractions[0] < fractions[3] && fractions[3] < fractions[6]);
    assert!(fractions[1] < fractions[0], "a taller panel pulls less");

    // At 160 degrees in a panel 100 points tall, PULL_POINTS would be more
    // than a quarter of the depth.
    let small = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(150.0, 100.0));
    assert_eq!(super::toward_eye_fraction(&camera, small), 0.25);
    let flat = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(150.0, 0.0));
    assert_eq!(super::toward_eye_fraction(&camera, flat), 0.0);
}
