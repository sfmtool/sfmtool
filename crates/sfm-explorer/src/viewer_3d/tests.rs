// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the viewport's point context menu and for the bench
//! figure's handles.
//!
//! egui needs no GPU to lay out a frame, so these drive the real thing:
//! `Viewer3D::show` through `Context::run_ui`, with the pick the GPU would have
//! reported handed in as the parameter `dock.rs` hands it in as. What they
//! assert is what the viewport *decides* -- whether the menu opened and on
//! which point, which handle a press took and what the release asked for --
//! rather than anything about pixels.
//!
//! egui resolves clicks against the widget rects registered on the *previous*
//! pass, so every gesture here is several frames and only the state the last
//! one left behind is trusted. The handle gestures aim at the figure the last
//! frame built, projected back through the camera it was drawn with, which is
//! the very picture a person would have pressed on.

use eframe::egui;
use nalgebra::{Point3, Vector3};
use sfmtool_core::bench::{EditableTrack, Stage};
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::{Camera, SfmrReconstruction};

use super::bench_track::{self, BenchGesture, HANDLE_HIT_RADIUS};
use super::{
    MenuTarget, Viewer3D, ALIGN_NORMAL_TO_Z_LABEL, EDIT_ON_BENCH_LABEL, RETRIANGULATE_POINT_LABEL,
    SET_TO_ORIGIN_LABEL, TRANSLATE_TO_ORIGIN_LABEL, TRANSLATE_TO_XY_PLANE_LABEL,
};
use crate::bench::geometry::{self, PatchEdit};
use crate::platform::ScrollInput;
use crate::scene::{PointRef, ReconId, SceneNode};
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
    let mut nowhere = egui::Rect::NOTHING;
    run_bench_frame(
        viewer,
        ctx,
        state,
        events,
        pointer,
        pick,
        busy,
        None,
        &mut nowhere,
    );
}

/// One frame with the node's bench handed in as well, and the rect the viewport
/// occupied reported back.
///
/// The rect is what a test projects the figure through to aim at a handle: the
/// viewport takes `ui.available_size()` at the cursor, so this is the very rect
/// `show` allocated.
#[allow(clippy::too_many_arguments)]
fn run_bench_frame(
    viewer: &mut Viewer3D,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
    pointer: egui::Pos2,
    pick: Option<PickTarget>,
    busy: Option<&str>,
    bench: Option<(&EditableTrack, Option<usize>, bool)>,
    rect: &mut egui::Rect,
) -> egui::CursorIcon {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    };
    // `platform::pointer_in_rect` does not read egui's pointer, so the test has
    // to place it where the window would have.
    crate::platform::set_test_pointer_pos(Some(pointer));
    let mut output = ctx.run_ui(input, |ui| {
        let scroll_input = ScrollInput::from_ctx(ui.ctx(), false);
        egui::CentralPanel::default().show(ui, |ui| {
            let node = &state.scene[0];
            *rect = ui.available_rect_before_wrap();
            let bench = bench.map(|(track, selected, busy)| bench_track::BenchTrack {
                node: node.id,
                track,
                edited: node.edited(),
                transform: node.transform(),
                selected,
                busy,
            });
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
                bench,
                &mut state.action_log,
            );
        });
    });
    output.textures_delta.clear();
    output.platform_output.cursor_icon
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
    assert_eq!(viewer.menu_target, Some(MenuTarget::Point(point)));
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
    assert_eq!(viewer.menu_target, None);
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
    assert_eq!(viewer.menu_target, None);
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

// ---- The bench figure's handles --------------------------------------------

/// How many half-lengths the eye stands off the patch: far enough that the whole
/// figure is on screen, near enough that the square is a couple of hundred panel
/// pixels across and its handles are well apart.
const STANDOFF: f64 = 12.0;

/// The tangent half-length a track at infinity is given here, which at the
/// viewport's own field of view puts its square about a hundred panel pixels
/// from its centre.
const BEARING_HALF: f64 = 0.1;

/// Somewhere in the viewport well clear of the figure, which is drawn about the
/// middle.
const EMPTY: egui::Pos2 = egui::pos2(120.0, 700.0);

/// The staged bench track, and a viewer looking square at its patch with two
/// frames already drawn -- the second against the first's figure, which is what
/// a press is hit-tested against.
struct Staged {
    viewer: Viewer3D,
    ctx: egui::Context,
    state: AppState,
    id: ReconId,
    label: String,
    /// The rect the viewport occupied, for projecting the figure back onto it.
    rect: egui::Rect,
}

impl Staged {
    /// The track as the bench now holds it.
    fn track(&self) -> EditableTrack {
        (**self
            .state
            .bench_track(self.id, &self.label)
            .expect("a track"))
        .clone()
    }

    /// Settle the view on `track` again, so the figure a press is aimed at is
    /// the one this track draws.
    fn settle(&mut self, track: &EditableTrack) {
        for _ in 0..2 {
            run_bench_frame(
                &mut self.viewer,
                &self.ctx,
                &mut self.state,
                Vec::new(),
                EMPTY,
                None,
                None,
                Some((track, None, false)),
                &mut self.rect,
            );
        }
    }

    /// The version labels the node holds, oldest first.
    fn versions(&self) -> Vec<String> {
        self.state
            .node(self.id)
            .expect("loaded")
            .history
            .versions()
            .iter()
            .map(|version| version.label.clone())
            .collect()
    }
}

/// A viewer with the staged track on its bench, looking square at the patch.
fn staged() -> Staged {
    let (state, id, label) = bench_track::tests::staged();
    let mut staged = Staged {
        viewer: Viewer3D::new(),
        ctx: egui::Context::default(),
        state,
        id,
        label,
        rect: egui::Rect::NOTHING,
    };
    let track = staged.track();
    look_square_at(&mut staged.viewer, &placement_of(&track));
    staged.settle(&track);
    staged
}

/// The track's patch, as an owned value.
fn placement_of(track: &EditableTrack) -> OrientedPatch {
    track
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a track from a point carries the stored patch")
}

/// Point the viewport square at the patch, [`STANDOFF`] half-lengths off it.
///
/// The frame's own `v` is the viewport's up, so the square's `u` runs along the
/// panel's `x` and a drag along the screen is a drag along an axis of the patch.
fn look_square_at(viewer: &mut Viewer3D, frame: &OrientedPatch) {
    let eye = frame.center + frame.normal() * (frame.half_extent[0] * STANDOFF);
    viewer.camera.world_up = frame.v_axis;
    viewer.camera.camera = Camera::look_at(eye, frame.center, frame.v_axis);
    viewer.view_initialized = true;
}

/// Point it along the patch's own `u` instead, which is the view in which the
/// plane is edge-on and a pixel of pointer motion is an unbounded distance
/// along it.
fn look_edge_on(viewer: &mut Viewer3D, frame: &OrientedPatch) {
    let eye = frame.center + frame.u_axis * (frame.half_extent[0] * STANDOFF);
    viewer.camera.world_up = frame.v_axis;
    viewer.camera.camera = Camera::look_at(eye, frame.center, frame.v_axis);
    viewer.view_initialized = true;
}

/// Point it `degrees` off the patch's own normal, leaning toward `+u`, and
/// `standoff` half-lengths out.
///
/// The view the **normal's** handle is read in: at zero it is
/// [`look_square_at`], where the segment projects to a point, and the further
/// off it goes the longer the segment lies across the screen -- while the plane,
/// which the three other handles read, foreshortens by the same turn. `standoff`
/// is the second half of that, a nearer eye making the segment larger on screen
/// without making the view any less end-on.
fn look_off_normal(viewer: &mut Viewer3D, frame: &OrientedPatch, degrees: f64, standoff: f64) {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let out = frame.normal() * cos + frame.u_axis * sin;
    let eye = frame.center + out * (frame.half_extent[0] * standoff);
    viewer.camera.world_up = frame.v_axis;
    viewer.camera.camera = Camera::look_at(eye, frame.center, frame.v_axis);
    viewer.view_initialized = true;
}

/// The figure the last frame built, which is the one on screen.
fn figure(viewer: &Viewer3D) -> &bench_track::Figure {
    viewer.bench_figure.as_ref().expect("a figure was built")
}

/// Where a `(xyz, w)` endpoint of the figure lands in the panel.
fn on_panel(staged: &Staged, at: [f32; 4]) -> egui::Pos2 {
    staged
        .viewer
        .camera
        .project_homogeneous(
            Vector3::new(f64::from(at[0]), f64::from(at[1]), f64::from(at[2])),
            f64::from(at[3]),
            staged.rect,
        )
        .expect("the figure is in front of the eye")
}

/// The centre dot, in panel px.
fn dot(staged: &Staged) -> egui::Pos2 {
    on_panel(staged, figure(&staged.viewer).centre)
}

/// Corner `k` of the square, in panel px.
fn corner(staged: &Staged, k: usize) -> egui::Pos2 {
    on_panel(staged, figure(&staged.viewer).frame[k].a)
}

/// The midpoint of edge `k`, which runs from corner `k` to corner `k + 1`.
fn edge_mid(staged: &Staged, k: usize) -> egui::Pos2 {
    let (a, b) = (corner(staged, k), corner(staged, (k + 1) % 4));
    a + (b - a) * 0.5
}

/// The normal's segment, centre first and arrow tip second, in panel px.
fn normal_segment(staged: &Staged) -> (egui::Pos2, egui::Pos2) {
    let arrow = figure(&staged.viewer)
        .normal
        .expect("a finite frame carries its normal");
    (on_panel(staged, arrow[0].a), on_panel(staged, arrow[0].b))
}

/// Where observation `observation`'s circle sits, in panel px.
fn circle(staged: &Staged, observation: usize) -> egui::Pos2 {
    let mark = figure(&staged.viewer)
        .marks
        .iter()
        .find(|mark| mark.observation == observation)
        .expect("that observation draws a mark");
    on_panel(staged, mark.segment.b)
}

/// What one gesture over the bench figure produced.
struct Dragged {
    /// What the release asked of the app, if the press had hold of a handle.
    gesture: Option<BenchGesture>,
    /// The cursor the viewport asked for while the pointer hovered the handle,
    /// before any button went down.
    cursor: egui::CursorIcon,
    /// How far the **eye** moved over the gesture, in world units. Zero is the
    /// claim a handle drag makes: the scene holds still for the whole of it.
    orbited: f64,
}

/// Drive one press-move-release over the figure and report what it produced.
///
/// `steps` are pointer positions after the press, as offsets from it in panel
/// px, so a test can put one below egui's own drag threshold and the next above
/// it. That distinction is the whole of what the press-decides-the-handle rule
/// is about: the viewport orbits on the first pixels of motion egui calls a
/// drag, and a handle taken only then would already have lost the gesture.
#[allow(clippy::too_many_arguments)]
fn gesture(
    staged: &mut Staged,
    track: &EditableTrack,
    busy: bool,
    press: egui::Pos2,
    steps: &[egui::Vec2],
    escape: bool,
) -> Dragged {
    let button = |pos: egui::Pos2, pressed: bool| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let escape_key = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let last = steps.last().map_or(press, |step| press + *step);
    let mut frames: Vec<Vec<egui::Event>> = vec![
        vec![egui::Event::PointerMoved(press)],
        vec![egui::Event::PointerMoved(press), button(press, true)],
    ];
    for step in steps {
        frames.push(vec![egui::Event::PointerMoved(press + *step)]);
    }
    frames.push(if escape {
        vec![egui::Event::PointerMoved(last), escape_key]
    } else {
        vec![egui::Event::PointerMoved(last)]
    });
    frames.push(vec![egui::Event::PointerMoved(last), button(last, false)]);

    let was = staged.viewer.camera.position();
    staged.viewer.bench_gesture = None;
    let mut cursor = egui::CursorIcon::Default;
    for (index, events) in frames.into_iter().enumerate() {
        let pointer = match events.first() {
            Some(egui::Event::PointerMoved(pos)) => *pos,
            _ => press,
        };
        let output = run_bench_frame(
            &mut staged.viewer,
            &staged.ctx,
            &mut staged.state,
            events,
            pointer,
            None,
            None,
            Some((track, None, busy)),
            &mut staged.rect,
        );
        // The hover frame, before any button is down: what the viewport asks
        // for there is the cursor a person sees over the handle.
        if index == 0 {
            cursor = output;
        }
    }
    Dragged {
        gesture: staged.viewer.bench_gesture.take(),
        cursor,
        orbited: (staged.viewer.camera.position() - was).norm(),
    }
}

/// The resize cursor that points **across** `direction`, worked out here rather
/// than called out of the viewport so a test says what it wants independently of
/// what the code does.
fn cursor_across(direction: egui::Vec2) -> egui::CursorIcon {
    let angle = direction
        .y
        .atan2(direction.x)
        .to_degrees()
        .rem_euclid(180.0);
    match angle {
        a if a < 22.5 || a >= 157.5 => egui::CursorIcon::ResizeVertical,
        a if a < 67.5 => egui::CursorIcon::ResizeNeSw,
        a if a < 112.5 => egui::CursorIcon::ResizeHorizontal,
        _ => egui::CursorIcon::ResizeNwSe,
    }
}

/// The axis one resize cursor lies along, as a panel direction.
fn cursor_axis(cursor: egui::CursorIcon) -> egui::Vec2 {
    match cursor {
        egui::CursorIcon::ResizeVertical => egui::vec2(0.0, 1.0),
        egui::CursorIcon::ResizeHorizontal => egui::vec2(1.0, 0.0),
        // The raster's `y` runs downward, so north-west to south-east is `x`
        // and `y` growing together.
        egui::CursorIcon::ResizeNwSe => egui::vec2(1.0, 1.0).normalized(),
        egui::CursorIcon::ResizeNeSw => egui::vec2(1.0, -1.0).normalized(),
        other => panic!("{other:?} is not a resize cursor"),
    }
}

/// The edit a gesture asked for, or a panic naming what it asked for instead.
fn edit_of(dragged: &Dragged) -> PatchEdit {
    match &dragged.gesture {
        Some(BenchGesture::Edit(edit)) => *edit,
        other => panic!("the gesture named something else: {other:?}"),
    }
}

#[test]
fn a_press_on_an_edge_resizes_and_orbits_nothing_while_the_same_motion_off_it_orbits() {
    let mut staged = staged();
    let track = staged.track();
    // Two panel pixels, then thirty: the first is under egui's drag threshold
    // and is exactly what would otherwise be spent orbiting the scene.
    let steps = [egui::vec2(2.0, 0.0), egui::vec2(30.0, 0.0)];
    let edge = edge_mid(&staged, 1);

    let dragged = gesture(&mut staged, &track, false, edge, &steps, false);
    assert_eq!(
        dragged.orbited, 0.0,
        "the scene orbited under a handle drag",
    );
    assert!(
        matches!(
            edit_of(&dragged),
            PatchEdit::Resize {
                moved_edge: Some(sfmtool_core::bench::Edge::PlusU),
                ..
            }
        ),
        "the press did not take the `+u` edge",
    );

    // The same motion from empty viewport is the orbit it always was.
    let dragged = gesture(&mut staged, &track, false, EMPTY, &steps, false);
    assert!(dragged.gesture.is_none(), "empty viewport edited the track");
    assert!(
        dragged.orbited > 0.0,
        "a press off the handles should still orbit",
    );
}

#[test]
fn an_edge_dragged_lands_under_the_release_point_with_the_far_edge_held() {
    let mut staged = staged();
    let track = staged.track();
    let press = edge_mid(&staged, 1);
    let far_before = edge_mid(&staged, 3);
    // Out along the square's own `+u`, which the view puts along the panel's x.
    let step = egui::vec2((press.x - dot(&staged).x) * 0.8, 0.0);

    let dragged = gesture(&mut staged, &track, false, press, &[step], false);
    let edit = edit_of(&dragged);
    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit)
        .expect("a place the ray reaches");
    let labels = staged.versions();
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    let sentence = labels.last().expect("a version");
    assert!(
        sentence.starts_with(&format!("Resized {} to a half-length of", staged.label)),
        "the version's label does not name the resize in world units: {sentence}",
    );

    // Redraw and read the square back off the figure: the dragged edge is under
    // the release point and the far one has not moved.
    let resized = staged.track();
    staged.settle(&resized);
    let landed = edge_mid(&staged, 1);
    let wanted = press + step;
    assert!(
        (landed - wanted).length() < 1.0,
        "the dragged edge should land on {wanted:?}, it landed on {landed:?}",
    );
    let far_after = edge_mid(&staged, 3);
    assert!(
        (far_after - far_before).length() < 1.0,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );
    let after = placement_of(&resized);
    assert_eq!(
        after.half_extent[0], after.half_extent[1],
        "a patch frame is square"
    );
}

#[test]
fn a_corner_dragged_onto_its_neighbour_is_a_quarter_turn_and_one_version() {
    let mut staged = staged();
    let track = staged.track();
    let press = corner(&staged, 2);
    let step = corner(&staged, 3) - press;

    let dragged = gesture(&mut staged, &track, false, press, &[step], false);
    assert_eq!(
        dragged.cursor,
        cursor_across(corner(&staged, 2) - dot(&staged)),
        "a corner's cursor lies along the arc it turns on",
    );
    let PatchEdit::Spin { angle_rad } = edit_of(&dragged) else {
        panic!("the press did not take a corner");
    };
    assert!(
        (angle_rad.to_degrees().abs() - 90.0).abs() < 0.5,
        "a corner dragged onto its neighbour is a quarter turn, not {}",
        angle_rad.to_degrees(),
    );

    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit_of(&dragged))
        .expect("a finite angle");
    let labels = staged.versions();
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    assert!(
        labels
            .last()
            .expect("a version")
            .starts_with(&format!("Spun {} by ", staged.label)),
        "{:?}",
        labels.last(),
    );
    let was = placement_of(&track);
    let turned = placement_of(&staged.track());
    assert_eq!(turned.center, was.center, "a turn moves the patch nowhere");
    assert!((turned.normal() - was.normal()).norm() < 1e-12);
}

/// How far off the normal the eye stands for the **dot's** own tests.
///
/// A view exactly down the normal collapses the whole arrow onto the centre, so
/// the arrowhead -- which is hit-tested ahead of everything, sitting as it does
/// at the far end of the segment that leaves the centre -- takes the dot's
/// presses. That is the arrangement rather than an accident: the aim is at its
/// best in exactly the view where the arrow is shortest, and a few degrees of
/// lean pulls the head clear.
const DOT_LEAN_DEG: f64 = 15.0;

/// Lean the view off the normal by [`DOT_LEAN_DEG`] and check the head really
/// has come clear of the dot, so a test that means the dot presses the dot.
fn look_past_the_arrowhead(staged: &mut Staged, track: &EditableTrack) {
    look_off_normal(
        &mut staged.viewer,
        &placement_of(track),
        DOT_LEAN_DEG,
        STANDOFF,
    );
    staged.settle(track);
    let (_, tip) = normal_segment(staged);
    assert!(
        (tip - dot(staged)).length() > 2.0 * HANDLE_HIT_RADIUS,
        "the lean should carry the arrowhead clear of the dot's own reach",
    );
}

#[test]
fn a_dot_drag_is_one_version_whose_label_names_the_move() {
    let mut staged = staged();
    let track = staged.track();
    look_past_the_arrowhead(&mut staged, &track);
    let press = dot(&staged);

    let dragged = gesture(
        &mut staged,
        &track,
        false,
        press,
        &[egui::vec2(24.0, -16.0)],
        false,
    );
    assert_eq!(dragged.cursor, egui::CursorIcon::Move, "the dot slides it");
    // The dot's own constraint: the travel is read on `u` and `v` alone, so
    // the displacement has no component along the normal at all.
    let PatchEdit::Translate { by } = edit_of(&dragged) else {
        panic!("the press did not take the dot");
    };
    assert_eq!(by[2], 0.0, "the dot drifted the patch off its own plane");

    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit_of(&dragged))
        .expect("a place the ray reaches");
    let labels = staged.versions();
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    assert!(
        labels
            .last()
            .expect("a version")
            .starts_with(&format!("Moved {} by ", staged.label)),
        "{:?}",
        labels.last(),
    );
    // In the patch's own plane, with the axes and the size untouched.
    let was = placement_of(&track);
    let now = placement_of(&staged.track());
    assert_eq!(now.half_extent, was.half_extent);
    assert!((now.normal() - was.normal()).norm() < 1e-12);
    let offset = now.center - was.center;
    assert!(offset.norm() > 0.0, "the patch did not move");
    assert!(
        offset.dot(&was.normal()).abs() < 1e-12,
        "the patch left its own plane: {offset:?}",
    );
}

#[test]
fn escape_leaves_no_edit_and_a_drag_that_ends_where_it_started_pushes_nothing() {
    let mut staged = staged();
    let track = staged.track();
    look_past_the_arrowhead(&mut staged, &track);
    let press = dot(&staged);

    let cancelled = gesture(
        &mut staged,
        &track,
        false,
        press,
        &[egui::vec2(24.0, -16.0)],
        true,
    );
    assert!(
        cancelled.gesture.is_none(),
        "escape left a gesture behind: {:?}",
        cancelled.gesture,
    );

    // And a gesture that ends where it began is a step that changes nothing,
    // which pushes no version -- the way a verdict an observation already holds
    // does.
    let steps = [egui::vec2(24.0, -16.0), egui::vec2(0.0, 0.0)];
    let still = gesture(&mut staged, &track, false, press, &steps, false);
    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit_of(&still))
        .expect("the place it already sits at");
    assert_eq!(
        staged.versions().len(),
        before,
        "a drag that ended where it started pushed a version",
    );
}

#[test]
fn an_edge_on_view_takes_no_press_and_a_busy_node_takes_none_either() {
    let mut staged = staged();
    let track = staged.track();
    let steps = [egui::vec2(2.0, 0.0), egui::vec2(30.0, 0.0)];

    // Busy first, while the view is still square on and the press would
    // otherwise land on the dot.
    let press = dot(&staged);
    let held = gesture(&mut staged, &track, true, press, &steps, false);
    assert!(
        held.gesture.is_none(),
        "a handle took a press while a task held the node: {:?}",
        held.gesture,
    );
    assert!(held.orbited > 0.0, "the viewport should navigate instead");

    // Then edge-on, where a pixel of pointer motion is an unbounded distance
    // along the plane.
    look_edge_on(&mut staged.viewer, &placement_of(&track));
    staged.settle(&track);
    let press = dot(&staged);
    let flat = gesture(&mut staged, &track, false, press, &steps, false);
    assert!(
        flat.gesture.is_none(),
        "a handle took a press on an edge-on plane: {:?}",
        flat.gesture,
    );
    assert!(flat.orbited > 0.0, "the viewport should navigate instead");
    assert_eq!(
        flat.cursor,
        egui::CursorIcon::Default,
        "an edge-on plane offers no handle, so it offers no cursor",
    );
}

#[test]
fn a_click_on_an_observations_circle_selects_its_row() {
    let mut staged = staged();
    // The fixture's keypoints are their point's exact projections, so every
    // mark sits on the centre dot. Move one off it, which is what a mark says
    // when the photograph and the geometry disagree.
    let track = staged.track();
    let site = track.observations[0].site().expect("a sighting");
    staged
        .state
        .edit_bench_patch(
            staged.id,
            &staged.label,
            &PatchEdit::Sight {
                observation: 0,
                pixel: [site[0] + 12.0, site[1] + 9.0],
            },
        )
        .expect("a pixel on the sensor");
    let track = staged.track();
    staged.settle(&track);

    let at = circle(&staged, 0);
    assert!(
        (at - dot(&staged)).length() > 12.0,
        "the fixture's mark should be well clear of the centre dot",
    );
    let clicked = gesture(&mut staged, &track, false, at, &[], false);
    assert_eq!(clicked.cursor, egui::CursorIcon::PointingHand);
    assert_eq!(clicked.gesture, Some(BenchGesture::SelectRow(0)));
    // And the click never reached the points under it.
    assert!(
        staged.viewer.pending_click.is_none(),
        "a click the figure caught also asked for a pick",
    );
}

/// The staged track taken to the sky, with a proper tangent frame about the
/// bearing the first observation looks along -- so its rays still meet the
/// tangent plane and the figure is the one a track at infinity draws.
fn as_bearing(state: &AppState, id: ReconId, track: &EditableTrack) -> EditableTrack {
    let seen_from = state
        .node(id)
        .expect("a loaded node")
        .edited()
        .base
        .image_table
        .images[track.observations[0].image as usize]
        .camera_center();
    let mut track = track.clone();
    let Stage::Track(payload) = &mut track.stage else {
        unreachable!("the staged track is at the track stage");
    };
    payload.at_infinity = true;
    payload.position = None;
    let frame = payload.placement.as_mut().expect("a patch");
    let direction = (frame.center - seen_from).normalize();
    // The world axis the bearing leans on least, so the tangent frame it builds
    // is well conditioned.
    let up = [Vector3::x(), Vector3::y(), Vector3::z()]
        .into_iter()
        .min_by(|a, b| direction.dot(a).abs().total_cmp(&direction.dot(b).abs()))
        .expect("three axes");
    *frame = OrientedPatch::from_infinity_direction(
        Point3::from(direction),
        up,
        [BEARING_HALF, BEARING_HALF],
    );
    track
}

/// The three handles of a track at infinity work through the same steps: the
/// dot moves the bearing, an edge changes the tangent half-length, and a corner
/// spins the square about the bearing.
#[test]
fn the_three_handles_of_a_track_at_infinity_move_the_bearing_the_size_and_the_spin() {
    let mut staged = staged();
    let track = as_bearing(&staged.state, staged.id, &staged.track());
    let was = placement_of(&track);
    // Looking along the bearing: a direction is projected rotation-only, so
    // where the eye stands does not enter into it.
    staged.viewer.camera.world_up = was.v_axis;
    staged.viewer.camera.camera = Camera::look_at(
        Point3::origin(),
        Point3::from(was.center.coords),
        was.v_axis,
    );
    staged.viewer.view_initialized = true;
    staged.settle(&track);

    let edited = staged
        .state
        .node(staged.id)
        .expect("a loaded node")
        .edited()
        .clone();
    let applied = |edit: &PatchEdit| {
        crate::bench::geometry::apply(&track, &edited, edit)
            .expect("the step takes a bearing")
            .0
    };

    // The dot: the bearing moves and comes back onto the unit sphere.
    let press = dot(&staged);
    let dragged = gesture(
        &mut staged,
        &track,
        false,
        press,
        &[egui::vec2(30.0, 0.0)],
        false,
    );
    let slid = placement_of(&applied(&edit_of(&dragged)));
    assert_eq!(slid.w, 0.0, "a bearing stays a bearing");
    assert!((slid.center.coords.norm() - 1.0).abs() < 1e-12);
    assert!(
        (slid.center - was.center).norm() > 1e-6,
        "the bearing did not move",
    );

    // An edge: the tangent half-length changes and the square stays square.
    let press = edge_mid(&staged, 1);
    let step = egui::vec2((press.x - dot(&staged).x) * 0.6, 0.0);
    let dragged = gesture(&mut staged, &track, false, press, &[step], false);
    let bigger = placement_of(&applied(&edit_of(&dragged)));
    assert_eq!(bigger.w, 0.0);
    assert_eq!(bigger.half_extent[0], bigger.half_extent[1]);
    assert!(
        bigger.half_extent[0] > was.half_extent[0] * 1.1,
        "the tangent half-length did not grow: {} from {}",
        bigger.half_extent[0],
        was.half_extent[0],
    );

    // A corner: the square spins about the bearing, which moves nowhere.
    let press = corner(&staged, 2);
    let step = corner(&staged, 3) - press;
    let dragged = gesture(&mut staged, &track, false, press, &[step], false);
    let spun = placement_of(&applied(&edit_of(&dragged)));
    assert_eq!(spun.center, was.center, "a turn moves the bearing nowhere");
    assert_eq!(spun.half_extent, was.half_extent);
    assert!(
        (spun.u_axis - was.u_axis).norm() > 1e-3,
        "the square did not spin",
    );
}

/// A corner's cursor lies **along** the arc it turns on and an edge's lies
/// **across** the edge, so running the pointer down an edge and onto the corner
/// turns the cursor by the difference between the two gestures.
///
/// There is no rotation cursor in egui's set to give a corner, so the direction
/// is the whole of what says a corner turns. It is checked against the figure's
/// own geometry on screen rather than against the call the viewport makes: what
/// is claimed is a fact about the picture.
#[test]
fn a_corner_turns_the_cursor_along_its_arc_where_an_edge_lies_across_itself() {
    let mut staged = staged();
    let track = staged.track();
    let centre = dot(&staged);

    // Within the 22.5-degree quantization a resize cursor has, two directions
    // count as square to each other when their cosine is under sin 22.5.
    let square_to = |cursor: egui::CursorIcon, direction: egui::Vec2| {
        cursor_axis(cursor).dot(direction.normalized()).abs() < 22.5_f32.to_radians().sin()
    };

    for k in 0..4 {
        let at = corner(&staged, k);
        let radius = at - centre;
        let cursor = gesture(&mut staged, &track, false, at, &[], false).cursor;
        assert!(
            square_to(cursor, radius),
            "corner {k} stands {radius:?} off the centre, so it travels across that; \
             the cursor was {cursor:?}, which lies {:?}",
            cursor_axis(cursor),
        );

        // And the edge running out of that same corner, which is dragged across
        // itself: the two cursors differ, which is the switch a person sees.
        let along = corner(&staged, (k + 1) % 4) - at;
        let midpoint = edge_mid(&staged, k);
        let on_edge = gesture(&mut staged, &track, false, midpoint, &[], false).cursor;
        assert!(
            square_to(on_edge, along),
            "edge {k} runs {along:?} and is moved across itself, not {on_edge:?}",
        );
        assert_ne!(
            cursor, on_edge,
            "a square's corner and the edge leaving it are 45 degrees apart, \
             so the cursor has to change between them",
        );
    }
}

// ---- The normal's segment ---------------------------------------------------

/// How far off the normal the eye stands for the segment's own tests: far
/// enough that the segment lies well across the screen and the press is nowhere
/// near the dot, near enough that the plane is still square-on and the three
/// other handles are live beside it.
const OFF_NORMAL_DEG: f64 = 60.0;

/// The eye's place for [`OFF_NORMAL_DEG`], as a world point.
fn off_normal_eye(frame: &OrientedPatch) -> Point3<f64> {
    let (sin, cos) = OFF_NORMAL_DEG.to_radians().sin_cos();
    frame.center + (frame.normal() * cos + frame.u_axis * sin) * (frame.half_extent[0] * STANDOFF)
}

/// A press on the normal's segment moves the patch along the normal, and the
/// point the press had hold of follows the pointer.
///
/// That is the claim in the picture: the segment keeps its length, so the
/// material point at the press's own fraction of it lands where the release
/// was. Read off the **redrawn** figure, which is the one a person would see.
#[test]
fn a_normal_drag_moves_the_patch_along_its_normal_and_orbits_nothing() {
    let mut staged = staged();
    let track = staged.track();
    let was = placement_of(&track);
    look_off_normal(&mut staged.viewer, &was, OFF_NORMAL_DEG, STANDOFF);
    staged.settle(&track);

    let (centre, tip) = normal_segment(&staged);
    let along = tip - centre;
    assert!(
        along.length() > 4.0 * HANDLE_HIT_RADIUS,
        "the fixture's view should draw a segment long enough to press away from the dot",
    );
    let press = centre + along * 0.5;
    let release = tip;

    let dragged = gesture(&mut staged, &track, false, press, &[release - press], false);
    assert_eq!(
        dragged.orbited, 0.0,
        "the scene orbited under a handle drag",
    );
    let PatchEdit::Translate { by } = edit_of(&dragged) else {
        panic!("the press did not take the normal's segment");
    };
    assert_eq!(
        [by[0], by[1]],
        [0.0, 0.0],
        "the normal's segment moved the patch across its own plane",
    );
    assert!(by[2].abs() > 0.0, "the drag asked for no distance");

    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit_of(&dragged))
        .expect("a finite distance");
    let labels = staged.versions();
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    let sentence = labels.last().expect("a version");
    assert!(
        sentence.starts_with(&format!("Moved {} by ", staged.label))
            && sentence.contains(" units along its normal to ("),
        "the version's label does not name the offset: {sentence}",
    );

    // The patch moved along its normal and nowhere else, keeping its axes and
    // its size.
    let now = placement_of(&staged.track());
    assert_eq!(now.half_extent, was.half_extent);
    assert!((now.normal() - was.normal()).norm() < 1e-12);
    let moved = now.center - was.center;
    assert!(
        (moved - was.normal() * moved.dot(&was.normal())).norm() < 1e-9 * moved.norm().max(1.0),
        "the patch left its own normal: {moved:?}",
    );

    // And the picture: the half-way point of the segment is the material point
    // the press had hold of, so the redrawn figure puts it under the release.
    let moved_track = staged.track();
    staged.settle(&moved_track);
    let (centre, tip) = normal_segment(&staged);
    let landed = centre + (tip - centre) * 0.5;
    assert!(
        (landed - release).length() < 1.0,
        "the place the press had hold of should land on {release:?}, it landed on {landed:?}",
    );

    // The same motion from empty viewport is the orbit it always was.
    let dragged = gesture(
        &mut staged,
        &moved_track,
        false,
        EMPTY,
        &[release - press],
        false,
    );
    assert!(dragged.gesture.is_none(), "empty viewport edited the track");
    assert!(
        dragged.orbited > 0.0,
        "a press off the handles should orbit"
    );
}

/// Escape abandons the gesture, and a drag that ends where it started asks for
/// an offset the step reads as none.
#[test]
fn escape_leaves_no_offset_and_a_normal_drag_that_ends_where_it_started_pushes_nothing() {
    let mut staged = staged();
    let track = staged.track();
    look_off_normal(
        &mut staged.viewer,
        &placement_of(&track),
        OFF_NORMAL_DEG,
        STANDOFF,
    );
    staged.settle(&track);

    let (centre, tip) = normal_segment(&staged);
    let press = centre + (tip - centre) * 0.5;
    let step = (tip - centre) * 0.4;

    let cancelled = gesture(&mut staged, &track, false, press, &[step], true);
    assert!(
        cancelled.gesture.is_none(),
        "escape left a gesture behind: {:?}",
        cancelled.gesture,
    );

    let still = gesture(
        &mut staged,
        &track,
        false,
        press,
        &[step, egui::vec2(0.0, 0.0)],
        false,
    );
    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit_of(&still))
        .expect("the depth it already stands at");
    assert_eq!(
        staged.versions().len(),
        before,
        "a drag that ended where it started pushed a version",
    );
}

/// The two degenerate views are one bar read from its two ends, so this is one
/// test: seen almost down the normal the segment takes no press, and that is
/// exactly the view in which the plane handles are at their best.
#[test]
fn a_view_down_the_normal_refuses_the_segment_where_the_plane_handles_are_at_their_best() {
    let mut staged = staged();
    let track = staged.track();
    // Inside the bar, and near enough that the segment is still tens of pixels
    // long: what refuses it is the angle and not its size on screen.
    look_off_normal(
        &mut staged.viewer,
        &placement_of(&track),
        crate::bench::geometry::MIN_PLANE_ANGLE_DEG - 1.0,
        3.0,
    );
    staged.settle(&track);

    let (centre, tip) = normal_segment(&staged);
    let press = centre + (tip - centre) * 0.6;
    assert!(
        (press - dot(&staged)).length() > HANDLE_HIT_RADIUS,
        "the press should be out of the dot's own reach, on the segment alone",
    );

    let steps = [egui::vec2(2.0, 0.0), egui::vec2(30.0, 0.0)];
    let refused = gesture(&mut staged, &track, false, press, &steps, false);
    assert!(
        refused.gesture.is_none(),
        "the segment took a press seen end-on: {:?}",
        refused.gesture,
    );
    assert!(
        refused.orbited > 0.0,
        "the viewport should navigate instead"
    );
    assert_eq!(
        refused.cursor,
        egui::CursorIcon::Default,
        "a refused handle offers no cursor",
    );

    // The same view, square on the plane: the dot is exactly where it should be
    // used, which is what makes the pair complementary.
    let centre = dot(&staged);
    let slid = gesture(&mut staged, &track, false, centre, &steps, false);
    assert!(
        matches!(edit_of(&slid), PatchEdit::Translate { .. }),
        "the plane handles should be at their best in this view",
    );
    assert_eq!(slid.orbited, 0.0);
}

/// The normal's segment is dragged **along** itself where an edge is dragged
/// across itself, so its cursor lies on the segment rather than square to it.
///
/// Checked against the figure's own geometry on screen, at several rolls of the
/// camera, so the claim is about the direction the segment runs in and not
/// about the panel's own axes.
#[test]
fn the_normal_segments_cursor_lies_along_it_where_an_edges_lies_across_itself() {
    let mut staged = staged();
    let track = staged.track();
    let frame = placement_of(&track);
    let eye = off_normal_eye(&frame);
    let forward = (frame.center - eye).normalize();
    // The patch's `v` taken into the image plane, which is the unrolled up, and
    // the axis a roll about the view direction turns it toward.
    let flat_up = (frame.v_axis - forward * frame.v_axis.dot(&forward)).normalize();
    let right = forward.cross(&flat_up).normalize();

    // Within the 22.5-degree quantization a resize cursor has, two directions
    // count as the same when their cosine is over cos 22.5 and as square to
    // each other when it is under sin 22.5.
    let cosine = |cursor: egui::CursorIcon, direction: egui::Vec2| {
        cursor_axis(cursor).dot(direction.normalized()).abs()
    };

    for roll in [0.0_f64, 35.0, 70.0, 110.0] {
        let (sin, cos) = roll.to_radians().sin_cos();
        let up = flat_up * cos + right * sin;
        staged.viewer.camera.world_up = up;
        staged.viewer.camera.camera = Camera::look_at(eye, frame.center, up);
        staged.viewer.view_initialized = true;
        staged.settle(&track);

        let (centre, tip) = normal_segment(&staged);
        let along = tip - centre;
        let cursor = gesture(&mut staged, &track, false, centre + along * 0.5, &[], false).cursor;
        assert!(
            cosine(cursor, along) > 22.5_f32.to_radians().cos(),
            "the segment runs {along:?} and is dragged along itself, so its cursor should \
             lie on it; at roll {roll} it was {cursor:?}, which lies {:?}",
            cursor_axis(cursor),
        );
        // An edge running the same way is dragged across itself, so the two
        // cursors are the ones a person would tell apart.
        let across = cursor_across(along);
        assert_ne!(
            cursor, across,
            "at roll {roll} the segment took the cursor an edge of the same slope would",
        );
        assert!(cosine(across, along) < 22.5_f32.to_radians().sin());
    }
}

// ---- The arrowhead ----------------------------------------------------------

/// How far off the normal the eye stands for the arrowhead's **aiming** tests:
/// inside [`geometry::AIM_ANGLE_DEG`], so the gesture is the aim, and far
/// enough off it that the head is drawn well clear of the dot at the centre.
const AIMING_DEG: f64 = 25.0;

/// The arrowhead, in panel px: the far end of the very segment the normal
/// handle is.
fn arrowhead(staged: &Staged) -> egui::Pos2 {
    normal_segment(staged).1
}

/// One primary button event.
fn primary(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

/// A press on the arrowhead turns the patch's normal, and the viewport does not
/// orbit while it is held.
#[test]
fn an_arrowhead_drag_tilts_the_patch_and_orbits_nothing() {
    let mut staged = staged();
    let track = staged.track();
    let was = placement_of(&track);
    look_off_normal(&mut staged.viewer, &was, AIMING_DEG, STANDOFF);
    staged.settle(&track);

    let head = arrowhead(&staged);
    assert!(
        (head - dot(&staged)).length() > 2.0 * HANDLE_HIT_RADIUS,
        "this view should draw the head clear of the dot",
    );
    let step = egui::vec2(0.0, 40.0);

    let dragged = gesture(&mut staged, &track, false, head, &[step], false);
    assert_eq!(
        dragged.orbited, 0.0,
        "the scene orbited under a handle drag",
    );
    let PatchEdit::Tilt { normal } = edit_of(&dragged) else {
        panic!("the press did not take the arrowhead");
    };
    assert!(
        Vector3::from(normal).norm() > 0.0,
        "the drag named no direction",
    );

    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit_of(&dragged))
        .expect("a finite direction");
    let labels = staged.versions();
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    let sentence = labels.last().expect("a version");
    assert!(
        sentence.starts_with(&format!("Tilted {} by ", staged.label))
            && sentence.contains(" degrees"),
        "the version's label does not name the tilt: {sentence}",
    );

    // The square turned about its own centre and nowhere else: the centre and
    // the size are where they were, and the normal is not.
    let tilted = staged.track();
    let now = placement_of(&tilted);
    assert_eq!(now.center, was.center);
    assert_eq!(now.half_extent, was.half_extent);
    assert!(
        (now.normal() - was.normal()).norm() > 1e-3,
        "the patch did not turn",
    );

    // The same motion from empty viewport is the orbit it always was.
    let dragged = gesture(&mut staged, &tilted, false, EMPTY, &[step], false);
    assert!(dragged.gesture.is_none(), "empty viewport edited the track");
    assert!(
        dragged.orbited > 0.0,
        "a press off the handles should orbit"
    );
}

/// How near the patch this test stands the eye, in half-lengths: inside
/// [`geometry::AIM_LEVER`], and outside [`geometry::NORMAL_LENGTH`] so the
/// arrowhead is still in front of the camera to be pressed at all.
const CLOSE_STANDOFF: f64 = 3.0;

/// The standoff is only the regression it claims to be from inside the lever,
/// and both are constants, so the claim is settled when this compiles.
const _: () = assert!(CLOSE_STANDOFF < geometry::AIM_LEVER);

/// The press is taken from **inside** the aim's own lever, where the patch
/// nearly fills the window.
///
/// The regression for a plane stood `AIM_LEVER` half-lengths out along the
/// normal: toward an eye that close it lay *behind* the camera, so the ray met
/// nothing, no drag was taken, and the press fell through to the viewport's
/// orbit -- while the hit test and the cursor, which read no plane, went on
/// showing the arrowhead as live. A handle that shows its cursor and then
/// navigates is worse than one that refuses, so what is asserted is the whole
/// path: the press is taken, it tilts, and the scene does not turn under it.
#[test]
fn an_arrowhead_close_to_the_eye_takes_the_press_rather_than_orbiting() {
    let mut staged = staged();
    let track = staged.track();
    let was = placement_of(&track);
    look_off_normal(&mut staged.viewer, &was, 10.0, CLOSE_STANDOFF);
    staged.settle(&track);

    let head = arrowhead(&staged);
    let dragged = gesture(
        &mut staged,
        &track,
        false,
        head,
        &[egui::vec2(0.0, 25.0)],
        false,
    );
    assert_eq!(
        dragged.orbited, 0.0,
        "the press fell through to the viewport's navigation",
    );
    let PatchEdit::Tilt { normal } = edit_of(&dragged) else {
        panic!("the press did not take the arrowhead");
    };
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &PatchEdit::Tilt { normal })
        .expect("a finite direction");
    assert!(
        (placement_of(&staged.track()).normal() - was.normal()).norm() > 1e-3,
        "the patch did not turn",
    );
}

/// Escape abandons the gesture, and a drag that ends where it started asks for
/// the normal the patch already faces, which the step reads as no turn.
#[test]
fn escape_leaves_no_tilt_and_an_arrowhead_drag_that_ends_where_it_started_pushes_nothing() {
    let mut staged = staged();
    let track = staged.track();
    look_off_normal(
        &mut staged.viewer,
        &placement_of(&track),
        AIMING_DEG,
        STANDOFF,
    );
    staged.settle(&track);

    let head = arrowhead(&staged);
    let step = egui::vec2(0.0, 30.0);

    let cancelled = gesture(&mut staged, &track, false, head, &[step], true);
    assert!(
        cancelled.gesture.is_none(),
        "escape left a gesture behind: {:?}",
        cancelled.gesture,
    );

    let still = gesture(
        &mut staged,
        &track,
        false,
        head,
        &[step, egui::vec2(0.0, 0.0)],
        false,
    );
    let before = staged.versions().len();
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit_of(&still))
        .expect("the direction it already faces");
    assert_eq!(
        staged.versions().len(),
        before,
        "a drag that ended where it started pushed a version",
    );
}

/// One frame of the viewport over `track`, for a test that has to look at the
/// drag between the press and the release rather than only at what it produced.
fn bench_frame(
    staged: &mut Staged,
    track: &EditableTrack,
    events: Vec<egui::Event>,
    pointer: egui::Pos2,
) {
    run_bench_frame(
        &mut staged.viewer,
        &staged.ctx,
        &mut staged.state,
        events,
        pointer,
        None,
        None,
        Some((track, None, false)),
        &mut staged.rect,
    );
}

/// The gesture is decided at the **press** and held for the whole drag.
///
/// Driven frame by frame rather than through [`gesture`], because what is
/// claimed is a fact about the middle of the drag: the aim carries the normal
/// well past [`geometry::AIM_ANGLE_DEG`], so a press made at the end of it
/// would swing, and the drag that made it is an aim from the first frame to the
/// last.
#[test]
fn the_arrowheads_gesture_is_chosen_at_the_press_and_does_not_change_under_it() {
    let mut staged = staged();
    let track = staged.track();
    let frame = placement_of(&track);
    look_off_normal(&mut staged.viewer, &frame, AIMING_DEG, STANDOFF);
    staged.settle(&track);

    let head = arrowhead(&staged);
    // Well out along the head's own radius from the centre, which is the way
    // the aim's plane carries the normal away from the eye.
    let far = head + (head - dot(&staged)).normalized() * 400.0;
    staged.viewer.bench_gesture = None;

    bench_frame(
        &mut staged,
        &track,
        vec![egui::Event::PointerMoved(head)],
        head,
    );
    bench_frame(
        &mut staged,
        &track,
        vec![egui::Event::PointerMoved(head), primary(head, true)],
        head,
    );
    let took = staged
        .viewer
        .bench_drag
        .expect("the press should have taken the arrowhead")
        .handle;
    assert_eq!(
        took,
        bench_track::Handle::Arrowhead(geometry::Tilt::Aim),
        "this view is the aim's",
    );
    bench_frame(
        &mut staged,
        &track,
        vec![egui::Event::PointerMoved(far)],
        far,
    );
    assert_eq!(
        staged
            .viewer
            .bench_drag
            .expect("the drag should still be held")
            .handle,
        took,
        "the gesture changed character halfway through",
    );
    bench_frame(
        &mut staged,
        &track,
        vec![egui::Event::PointerMoved(far), primary(far, false)],
        far,
    );

    let edit = match staged.viewer.bench_gesture.take() {
        Some(BenchGesture::Edit(edit)) => edit,
        other => panic!("the release named something else: {other:?}"),
    };
    staged
        .state
        .edit_bench_patch(staged.id, &staged.label, &edit)
        .expect("a finite direction");

    // And the normal it reached is past the bar, so a press made now would
    // swing: the aim ran the normal through the place the two gestures part.
    let eye = staged.viewer.camera.position();
    assert!(
        matches!(
            geometry::tilt_gesture(&placement_of(&staged.track()), eye),
            Some(geometry::Tilt::Swing(_)),
        ),
        "the drag should have carried the normal past the bar it was chosen by",
    );
}

/// The arrowhead takes the cursor of the gesture it is about to make, and the
/// swing's is checked against **the picture**: the head travels along the arc
/// it turns on, so its cursor lies square to its own radius from the centre --
/// the corner's reading, and that arc's tangent.
#[test]
fn the_arrowheads_cursor_is_the_gesture_it_is_about_to_make() {
    let mut staged = staged();
    let track = staged.track();
    let frame = placement_of(&track);

    // Near the line of sight, where the aim is free in two directions at once.
    look_off_normal(&mut staged.viewer, &frame, AIMING_DEG, STANDOFF);
    staged.settle(&track);
    let head = arrowhead(&staged);
    assert_eq!(
        gesture(&mut staged, &track, false, head, &[], false).cursor,
        egui::CursorIcon::AllScroll,
        "an aim is free in two directions, which is the scroll-all cursor",
    );

    // Across it, where the head swings about one axis. Read at several rolls of
    // the camera, so the claim is about the direction the radius runs in and
    // not about the panel's own axes.
    let eye = off_normal_eye(&frame);
    let forward = (frame.center - eye).normalize();
    let flat_up = (frame.v_axis - forward * frame.v_axis.dot(&forward)).normalize();
    let right = forward.cross(&flat_up).normalize();
    for roll in [0.0_f64, 35.0, 70.0, 110.0] {
        let (sin, cos) = roll.to_radians().sin_cos();
        let up = flat_up * cos + right * sin;
        staged.viewer.camera.world_up = up;
        staged.viewer.camera.camera = Camera::look_at(eye, frame.center, up);
        staged.viewer.view_initialized = true;
        staged.settle(&track);

        let head = arrowhead(&staged);
        let radius = head - dot(&staged);
        let cursor = gesture(&mut staged, &track, false, head, &[], false).cursor;
        assert!(
            cursor_axis(cursor).dot(radius.normalized()).abs() < 22.5_f32.to_radians().sin(),
            "the head stands {radius:?} off the centre and travels across that, so its \
             cursor should be the tangent; at roll {roll} it was {cursor:?}, which lies {:?}",
            cursor_axis(cursor),
        );
        // A corner of the square at the same slope reads the same way, which is
        // the point: the two are one gesture about two axes.
        assert_eq!(cursor, cursor_across(radius));
    }
}

// ---- The patch menu --------------------------------------------------------

/// Somewhere inside the square and on none of its handles: halfway from the
/// centre to a corner, which on a square this size is tens of pixels from the
/// dot, the corners, the edges and the circles clustered at the centre.
fn inside_the_square(staged: &Staged) -> egui::Pos2 {
    let (centre, corner) = (dot(staged), corner(staged, 0));
    centre + (corner - centre) * 0.5
}

/// Right-click at `at` with the staged track handed in and `pick` reported
/// under the cursor, then settle a frame with the pointer somewhere else, so
/// what is asserted is what the latch kept rather than what is under the
/// pointer.
fn right_click_on_bench(
    staged: &mut Staged,
    track: Option<&EditableTrack>,
    at: egui::Pos2,
    pick: Option<PickTarget>,
    busy: Option<&str>,
) {
    staged.viewer.point_menu = None;
    for (events, pointer) in [
        (vec![egui::Event::PointerMoved(at)], at),
        (vec![secondary(at, true)], at),
        (vec![secondary(at, false)], at),
        (vec![egui::Event::PointerMoved(EMPTY)], EMPTY),
        (Vec::new(), EMPTY),
    ] {
        run_bench_frame(
            &mut staged.viewer,
            &staged.ctx,
            &mut staged.state,
            events,
            pointer,
            pick,
            busy,
            track.map(|track| (track, None, busy.is_some())),
            &mut staged.rect,
        );
    }
}

/// The entries the menu laid out on the last frame, by label.
fn offered(viewer: &Viewer3D) -> Vec<&'static str> {
    viewer
        .menu_entry_rects
        .iter()
        .map(|(text, _)| *text)
        .collect()
}

#[test]
fn a_right_click_on_the_square_opens_the_patch_menu_even_over_a_point() {
    let mut staged = staged();
    let track = staged.track();
    let at = inside_the_square(&staged);
    let point = PointRef::new(staged.id, 7);
    right_click_on_bench(
        &mut staged,
        Some(&track),
        at,
        Some(PickTarget::Point(point)),
        None,
    );
    assert_eq!(
        staged.viewer.menu_target,
        Some(MenuTarget::Patch(staged.id))
    );
    assert_eq!(
        staged.viewer.point_menu, None,
        "the point under it was not named"
    );
    // Laid out with the pointer long gone from the square: the latch held.
    assert_eq!(
        offered(&staged.viewer),
        [
            SET_TO_ORIGIN_LABEL,
            ALIGN_NORMAL_TO_Z_LABEL,
            TRANSLATE_TO_ORIGIN_LABEL,
            TRANSLATE_TO_XY_PLANE_LABEL,
        ]
    );
}

#[test]
fn a_right_click_on_a_point_away_from_the_square_opens_the_point_menu() {
    let mut staged = staged();
    let track = staged.track();
    let point = PointRef::new(staged.id, 7);
    right_click_on_bench(
        &mut staged,
        Some(&track),
        EMPTY,
        Some(PickTarget::Point(point)),
        None,
    );
    assert_eq!(staged.viewer.menu_target, Some(MenuTarget::Point(point)));
    assert_eq!(
        offered(&staged.viewer),
        [EDIT_ON_BENCH_LABEL, RETRIANGULATE_POINT_LABEL]
    );
}

#[test]
fn a_right_click_on_empty_space_beside_the_square_opens_nothing() {
    let mut staged = staged();
    let track = staged.track();
    right_click_on_bench(&mut staged, Some(&track), EMPTY, None, None);
    assert_eq!(staged.viewer.menu_target, None);
    assert!(staged.viewer.menu_entry_rects.is_empty());
}

#[test]
fn with_no_active_track_the_square_is_not_there_to_click() {
    let mut staged = staged();
    let at = inside_the_square(&staged);
    staged.viewer.bench_figure = None;
    right_click_on_bench(&mut staged, None, at, None, None);
    assert_eq!(staged.viewer.menu_target, None);
    assert!(staged.viewer.menu_entry_rects.is_empty());
}

#[test]
fn choosing_a_patch_entry_reports_the_node_and_the_reframe() {
    let mut staged = staged();
    let track = staged.track();
    let at = inside_the_square(&staged);
    right_click_on_bench(&mut staged, Some(&track), at, None, None);
    let rect = staged
        .viewer
        .menu_entry_rects
        .iter()
        .find(|(text, _)| *text == ALIGN_NORMAL_TO_Z_LABEL)
        .map(|(_, rect)| *rect)
        .expect("the entry is up");
    let on = rect.center();
    for events in [
        vec![egui::Event::PointerMoved(on)],
        vec![primary(on, true)],
        vec![primary(on, false)],
    ] {
        run_bench_frame(
            &mut staged.viewer,
            &staged.ctx,
            &mut staged.state,
            events,
            on,
            None,
            None,
            Some((&track, None, false)),
            &mut staged.rect,
        );
    }
    assert_eq!(
        staged.viewer.patch_menu,
        Some((
            staged.id,
            crate::display_transform::PatchReframe::AlignNormalToZ
        ))
    );
}

#[test]
fn a_busy_node_greys_the_patch_entries() {
    let mut staged = staged();
    let track = staged.track();
    let at = inside_the_square(&staged);
    right_click_on_bench(&mut staged, Some(&track), at, None, Some("busy"));
    assert_eq!(
        staged.viewer.menu_target,
        Some(MenuTarget::Patch(staged.id))
    );
    assert_eq!(offered(&staged.viewer).len(), 4, "greyed, not hidden");
}
