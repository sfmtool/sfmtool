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
use crate::scene_renderer::{pack_rgba8, OverlayInstance};
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
    frame_sized(viewer, ctx, state, events, pointer, VIEWPORT)
}

/// [`frame`] in a window `size` points across.
fn frame_sized(
    viewer: &mut Viewer3D,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
    pointer: egui::Pos2,
    size: egui::Vec2,
) -> Frame {
    viewer.measurement_key = state.measurement_layer_key(state.show_all_measurements);
    viewer.measurements = state.measurement_layer(state.show_all_measurements);
    crate::platform::set_test_pointer_pos(Some(pointer));
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
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

/// How every piece of a measurement is drawn.
const FLAGS: u32 = OverlayInstance::HOLD | OverlayInstance::PULL_TOWARD_EYE;

/// A finite world point as an instance holds it.
fn at(p: Point3<f64>) -> [f32; 4] {
    [p.x as f32, p.y as f32, p.z as f32, 1.0]
}

/// The selected measurement's instances: the halos of its line and of each
/// mark's ring, then the line and each ring, then each dot; each ring and dot
/// one instance at the endpoint's own position with its radius in points, the
/// selected endpoint's ring larger; in the node's colour and the halo's, at
/// the strokes' widths in points, held behind the scene and pulled toward the
/// eye. The frame builds the same instances for the upload.
#[test]
fn a_measurement_draws_its_line_and_marks_as_instances() {
    use crate::measurement_marks::{
        HALO, HALO_EXTRA_WIDTH, LINE_WIDTH, MARK_DOT_RADIUS, MARK_RADIUS, SELECTED_MARK_SCALE,
        STORED_RGB,
    };
    let (mut state, id) = measured_state(Some("m"));
    let m = select(&mut state, id, 0);
    state.select_endpoint(m, EndpointSide::B);
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    frame(
        &mut viewer,
        &ctx,
        &mut state,
        Vec::new(),
        egui::pos2(5.0, 5.0),
    );
    let pieces = super::instances(&viewer.measurements);
    assert_eq!(
        viewer.measurement_upload.as_ref(),
        Some(&pieces),
        "the frame builds the instances the upload draws"
    );
    let grey = super::encoded_rgba(egui::Color32::from_rgb(
        STORED_RGB[0],
        STORED_RGB[1],
        STORED_RGB[2],
    ));
    let halo = super::encoded_rgba(HALO);
    // egui's premultiplied bytes as they are: the halo's black at alpha 170,
    // and the opaque grey.
    assert_eq!(halo, [0, 0, 0, 170]);
    assert_eq!(grey, [STORED_RGB[0], STORED_RGB[1], STORED_RGB[2], 255]);

    let (a, b) = (at(Point3::from(POINTS[0])), at(Point3::from(POINTS[1])));
    let (half, halo_half) = (LINE_WIDTH / 2.0, (LINE_WIDTH + HALO_EXTRA_WIDTH) / 2.0);
    let selected = MARK_RADIUS * SELECTED_MARK_SCALE;
    // The line and its halo stop at the outer edge of each ring's stroke.
    let line = |color, half_width| {
        OverlayInstance::segment(a, b, color, half_width, FLAGS)
            .inset(MARK_RADIUS + half, selected + half)
    };
    let ring =
        |p, color, radius, half_width| OverlayInstance::ring(p, color, radius, half_width, FLAGS);
    let dot = |p| OverlayInstance::disc(p, grey, MARK_DOT_RADIUS, FLAGS);
    assert_eq!(
        pieces,
        vec![
            line(halo, halo_half),
            ring(a, halo, MARK_RADIUS, halo_half),
            ring(b, halo, selected, halo_half),
            line(grey, half),
            ring(a, grey, MARK_RADIUS, half),
            ring(b, grey, selected, half),
            dot(a),
            dot(b),
        ]
    );
    for piece in &pieces {
        assert_eq!(piece.flags, FLAGS);
    }
}

/// A ring and a dot are one instance each, whatever the distance of the
/// endpoint: their radius is in points and their centre is the endpoint, so
/// the same instances are built from far away and from close up.
#[test]
fn a_ring_or_a_dot_is_one_instance_with_its_radius_in_points() {
    use crate::measurement_marks::{MARK_DOT_RADIUS, MARK_RADIUS};
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    let near = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0)).0;
    let far = looking_from(
        &mut state,
        Point3::new(0.0, -200.0, 0.0),
        egui::pos2(5.0, 5.0),
    )
    .0;
    let pieces = super::instances(&near.measurements);
    assert_eq!(pieces, super::instances(&far.measurements));
    let rings: Vec<_> = pieces
        .iter()
        .filter(|p| p.kind == OverlayInstance::RING)
        .collect();
    let discs: Vec<_> = pieces
        .iter()
        .filter(|p| p.kind == OverlayInstance::DISC)
        .collect();
    assert_eq!(
        (rings.len(), discs.len()),
        (4, 2),
        "a halo and a ring a mark"
    );
    for ring in rings {
        assert_eq!(ring.radius, MARK_RADIUS);
    }
    for disc in discs {
        assert_eq!((disc.radius, disc.half_width), (MARK_DOT_RADIUS, 0.0));
    }
}

/// With *Show all measurements* on, every halo comes before every stroke, and
/// every dot after both, across the measurements: one measurement's halo never
/// crosses another's stroke.
#[test]
fn halos_come_before_strokes_across_measurements() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    state.show_all_measurements = true;
    let (viewer, _ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    assert!(viewer.measurements.drawn.len() > 1);
    let halo = pack_rgba8(super::encoded_rgba(crate::measurement_marks::HALO));
    let pieces = super::instances(&viewer.measurements);
    let rank = |p: &OverlayInstance| match (p.color == halo, p.kind) {
        (true, _) => 0,
        (false, OverlayInstance::DISC) => 2,
        (false, _) => 1,
    };
    let ranks: Vec<u8> = pieces.iter().map(rank).collect();
    assert!(ranks.windows(2).all(|w| w[0] <= w[1]), "{ranks:?}");
    let halos = ranks.iter().filter(|r| **r == 0).count();
    let strokes = ranks.iter().filter(|r| **r == 1).count();
    assert_eq!(halos, strokes, "a halo under every stroke");
    assert!(ranks.contains(&2));
    // Each measurement's stroke is in the layer's order.
    let lines: Vec<u32> = pieces
        .iter()
        .filter(|p| p.kind == OverlayInstance::SEGMENT && p.color != halo)
        .map(|p| p.color)
        .collect();
    let expected: Vec<u32> = viewer
        .measurements
        .drawn
        .iter()
        .filter(|d| {
            d.endpoints
                .iter()
                .filter(|e| matches!(e.place, crate::measurement_marks::Place::Finite(_)))
                .count()
                == 2
        })
        .map(|d| pack_rgba8(super::encoded_rgba(d.color)))
        .collect();
    assert_eq!(lines, expected);
}

/// Every measurement's line and its halo stop at the outer edge of the stroke
/// of the ring at each of their ends, the selected endpoint's larger ring and
/// the selected measurement's wider stroke included, so no line draws over a
/// ring's stroke or inside it. The rings, dots and every other piece are not
/// inset.
#[test]
fn a_line_stops_at_the_outer_edge_of_each_end_s_ring() {
    let (mut state, id) = measured_state(Some("m"));
    let m = select(&mut state, id, 0);
    state.select_endpoint(m, EndpointSide::A);
    state.show_all_measurements = true;
    let (viewer, _ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    let halo = pack_rgba8(super::encoded_rgba(crate::measurement_marks::HALO));
    let pieces = super::instances(&viewer.measurements);
    let segments: Vec<_> = pieces
        .iter()
        .filter(|p| p.kind == OverlayInstance::SEGMENT)
        .collect();
    assert!(segments.len() >= 4, "two lines, each with its halo");
    let mut radii = Vec::new();
    for segment in &segments {
        // The stroke ring of the same colour at each end: a halo's line is
        // inset as its measurement's stroke is.
        let stroke = segments
            .iter()
            .find(|s| s.a == segment.a && s.b == segment.b && s.color != halo)
            .expect("each halo's line has its stroke");
        let ring_at = |p: [f32; 4]| {
            pieces
                .iter()
                .find(|r| r.kind == OverlayInstance::RING && r.a == p && r.color == stroke.color)
                .expect("a ring at each end of a line")
        };
        let (ring_a, ring_b) = (ring_at(segment.a), ring_at(segment.b));
        assert_eq!(
            segment.inset,
            [
                ring_a.radius + ring_a.half_width,
                ring_b.radius + ring_b.half_width
            ]
        );
        radii.extend([ring_a.radius, ring_b.radius]);
    }
    // The selected endpoint's ring is the larger one, and its line's end
    // stops further out.
    assert!(radii
        .iter()
        .any(|r| *r > crate::measurement_marks::MARK_RADIUS));
    for piece in pieces.iter().filter(|p| p.kind != OverlayInstance::SEGMENT) {
        assert_eq!(piece.inset, [0.0; 2]);
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
    let layer = &viewer.measurements;
    assert!(layer.drawn.len() > 1, "every measurement is drawn");
    let halo = pack_rgba8(super::encoded_rgba(crate::measurement_marks::HALO));
    let mut roles = Vec::new();
    for drawn in &layer.drawn {
        let one = crate::measurement_marks::MeasurementLayer {
            drawn: vec![drawn.clone()],
        };
        let pieces = super::instances(&one);
        // A measurement with no finite endpoint draws nothing.
        let Some(stroke) = pieces.iter().find(|p| p.color != halo) else {
            continue;
        };
        roles.push(drawn.role);
        match drawn.role {
            Role::Selected => {
                assert_eq!(stroke.half_width, EMPHASISED_LINE_WIDTH / 2.0);
                assert_eq!(stroke.color.to_le_bytes()[3], 255);
            }
            Role::Other => {
                assert_eq!(stroke.half_width, LINE_WIDTH / 2.0);
                let alpha = f32::from(stroke.color.to_le_bytes()[3]) / 255.0;
                assert!((alpha - OTHER_OPACITY).abs() < 0.01, "{stroke:?}");
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
    let pieces =
        super::instances(&crate::measurement_marks::MeasurementLayer { drawn: vec![item] });
    let violet = pack_rgba8(super::encoded_rgba(crate::bench::verdict_color(
        sfmtool_core::bench::Verdict::In,
    )));
    let line = pieces
        .iter()
        .find(|p| p.kind == OverlayInstance::SEGMENT && p.color != halo)
        .expect("the item's line");
    assert_eq!(line.color, violet);
    assert_eq!(line.half_width, ITEM_LINE_WIDTH / 2.0);
}

/// A refused endpoint and one at infinity draw no instance, and the other
/// endpoint still draws its mark, with no line.
#[test]
fn refused_and_infinite_endpoints_draw_no_instances() {
    let (mut state, id) = measured_state(Some("m"));
    for index in [2, 3] {
        select(&mut state, id, index);
        let (viewer, _ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
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
        let pieces = super::instances(&viewer.measurements);
        let kinds: Vec<u32> = pieces.iter().map(|p| p.kind).collect();
        assert_eq!(
            kinds,
            [
                OverlayInstance::RING,
                OverlayInstance::RING,
                OverlayInstance::DISC
            ],
            "measurement {index}: A's halo, ring and dot"
        );
        for piece in &pieces {
            assert_eq!(piece.a, at(a), "measurement {index}: every piece is A's");
        }
    }
}

/// A line with one end behind the eye keeps that end, for the pass to clip
/// at the near plane, and both ends keep their marks: the instances do not
/// depend on the camera, and the pass draws nothing of a ring or a dot whose
/// centre is behind the near plane.
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
    let ctx = egui::Context::default();
    frame(
        &mut viewer,
        &ctx,
        &mut state,
        Vec::new(),
        egui::pos2(5.0, 5.0),
    );
    let (a, b) = (Point3::from(POINTS[0]), Point3::from(POINTS[1]));
    assert!(
        viewer.camera.world_to_view(&a).z > 0.0,
        "A is behind the eye"
    );
    let pieces = viewer.measurement_upload.expect("built");
    let lines: Vec<_> = pieces
        .iter()
        .filter(|p| p.kind == OverlayInstance::SEGMENT)
        .collect();
    assert_eq!(lines.len(), 2, "the line and its halo");
    for line in lines {
        assert_eq!((line.a, line.b), (at(a), at(b)));
    }
    let marks: Vec<_> = pieces
        .iter()
        .filter(|p| p.kind != OverlayInstance::SEGMENT)
        .map(|p| p.a)
        .collect();
    assert_eq!(
        marks,
        [at(a), at(b), at(a), at(b), at(a), at(b)],
        "each end's halo, ring and dot"
    );
}

/// The pull the pass draws the measurements toward the eye by is the one the
/// CPU drew them by before the shader did: [`super::toward_eye_fraction`] of
/// the camera and the panel of the frame, handed over with the frame's pixels
/// per point. It follows the field of view and the panel's size.
#[test]
fn the_frame_hands_the_pass_the_pull_and_the_scale() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    ctx.set_pixels_per_point(2.0);
    let mut seen = Vec::new();
    for (fov, size) in [
        (0.8, VIEWPORT),
        (0.3, VIEWPORT),
        (0.3, egui::vec2(900.0, 400.0)),
    ] {
        viewer.camera.fov = fov;
        let shown = frame_sized(
            &mut viewer,
            &ctx,
            &mut state,
            Vec::new(),
            egui::pos2(5.0, 5.0),
            size,
        );
        let expected = super::toward_eye_fraction(&viewer.camera, shown.rect) as f32;
        assert_eq!(viewer.overlay_view.pull_fraction, expected);
        assert_eq!(viewer.overlay_view.pixels_per_point, 2.0);
        seen.push(expected);
    }
    assert!(seen[0] != seen[1] && seen[1] != seen[2], "{seen:?}");
}

/// The instances are built once, and a frame that changes only the camera, the
/// field of view or the panel's size builds and hands over nothing. Each change
/// to what they are drawn from builds them once more: a step on the focused
/// item, an undo, another measurement or an endpoint selected, *Show all
/// measurements* toggled, the node hidden and shown, its display transform
/// changed, and its tint.
#[test]
fn the_instances_are_built_when_their_key_changes_and_not_for_the_camera() {
    let (mut state, id) = measured_state(Some("m"));
    select(&mut state, id, 0);
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    assert_eq!(viewer.measurement_builds, 1, "built once over two frames");
    assert!(viewer.measurement_upload.take().is_some());
    let pointer = egui::pos2(5.0, 5.0);

    // The camera alone.
    let still = |viewer: &mut Viewer3D, state: &mut AppState, size: egui::Vec2| {
        frame_sized(viewer, &ctx, state, Vec::new(), pointer, size);
        assert_eq!(viewer.measurement_builds, 1);
        assert!(viewer.measurement_upload.is_none());
    };
    viewer.camera.orbit(40.0, 15.0);
    still(&mut viewer, &mut state, VIEWPORT);
    viewer.camera.fov = 0.3;
    still(&mut viewer, &mut state, VIEWPORT);
    still(&mut viewer, &mut state, egui::vec2(700.0, 500.0));

    // Each change to the key builds once, and the frame after it nothing.
    let mut builds = 1;
    let mut changed = |viewer: &mut Viewer3D, state: &mut AppState, what: &str| {
        frame(viewer, &ctx, state, Vec::new(), pointer);
        builds += 1;
        assert_eq!(viewer.measurement_builds, builds, "{what}");
        assert_eq!(
            viewer.measurement_upload.take(),
            Some(super::instances(&viewer.measurements)),
            "{what}"
        );
        frame(viewer, &ctx, state, Vec::new(), pointer);
        assert_eq!(viewer.measurement_builds, builds, "{what}, a frame later");
        assert!(viewer.measurement_upload.is_none(), "{what}, a frame later");
    };
    let put = state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    changed(&mut viewer, &mut state, "the item focused");
    state
        .set_bench_measurement_label(id, &put.label, "door")
        .unwrap();
    changed(&mut viewer, &mut state, "a step on the item");
    state.undo(id).unwrap();
    changed(&mut viewer, &mut state, "an undo");
    select(&mut state, id, 1);
    changed(&mut viewer, &mut state, "another measurement selected");
    state.select_endpoint(MeasurementRef::distance(id, 1), EndpointSide::A);
    changed(&mut viewer, &mut state, "an endpoint selected");
    state.show_all_measurements = true;
    changed(&mut viewer, &mut state, "show all toggled");
    state.scene[0].visible = false;
    changed(&mut viewer, &mut state, "the node hidden");
    state.scene[0].visible = true;
    changed(&mut viewer, &mut state, "the node shown");
    state.scene[0].history.transform_mut().scale = 2.0;
    changed(&mut viewer, &mut state, "the display transform changed");
    state.scene[0].tint = crate::scene::NodeTint::Tint(&crate::scene::TINT_PALETTE[0]);
    changed(&mut viewer, &mut state, "the node tinted");
}

/// With a thousand measurements and *Show all measurements* on, a frame while
/// the camera orbits builds no instance and hands nothing to the upload.
#[test]
fn a_thousand_measurements_orbited_build_and_upload_nothing() {
    let (mut state, id) = measured_state(Some("m"));
    {
        let distance = state.scene[0]
            .recon_mut()
            .image_set
            .measurements
            .distance
            .as_mut()
            .expect("the fixture's measurements");
        let n = distance.distances.len();
        let rows: Vec<usize> = (0..1000).map(|i| i % n).collect();
        distance.distances = distance.distances.select(ndarray::Axis(0), &rows);
        distance.units = distance.units.select(ndarray::Axis(0), &rows);
        distance.image_indexes = distance.image_indexes.select(ndarray::Axis(0), &rows);
        distance.endpoints_xy = distance.endpoints_xy.select(ndarray::Axis(0), &rows);
        distance.labels = rows.iter().map(|&i| distance.labels[i].clone()).collect();
    }
    select(&mut state, id, 0);
    state.show_all_measurements = true;
    let (mut viewer, ctx) = looking_from(&mut state, eye(), egui::pos2(5.0, 5.0));
    assert_eq!(viewer.measurements.drawn.len(), 1000);
    assert_eq!(viewer.measurement_builds, 1);
    let built = viewer.measurement_upload.take().expect("built once");
    assert!(built.len() > 5000, "{} instances", built.len());
    for _ in 0..5 {
        viewer.camera.orbit(25.0, 5.0);
        frame(
            &mut viewer,
            &ctx,
            &mut state,
            Vec::new(),
            egui::pos2(5.0, 5.0),
        );
        assert_eq!(viewer.measurement_builds, 1);
        assert!(viewer.measurement_upload.is_none());
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
