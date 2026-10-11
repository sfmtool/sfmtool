// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests of the delete brush: the preview, the stroke, its
//! cancellation and its commit, and the tool strip.
//!
//! `Viewer3D::show` runs through `Context::run_ui` as it does in
//! `viewer_3d/tests.rs`. The GPU is replaced by [`Gpu`], which answers each
//! region request from a synthetic picture of which point every pixel shows,
//! with the latency the real frame has: the request a frame makes is copied at
//! the start of the next frame and read at that frame's end.

use std::collections::HashSet;

use eframe::egui;
use sfmtool_core::SfmrReconstruction;

use super::{BrushGesture, Tool, BRUSH_RADIUS};
use crate::platform::ScrollInput;
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::scene_renderer::{PickRegion, PixelRect};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

const VIEWPORT: egui::Vec2 = egui::vec2(1200.0, 800.0);

/// Side of the square of pixels one synthetic point covers.
const BLOCK: u32 = 20;
/// Texture column where the first reconstruction's half of the picture ends.
const SPLIT: u32 = 600;
/// Texture row below which the picture is background.
const GROUND: u32 = 700;

/// The point the synthetic picture shows at texture pixel `(x, y)`: the left
/// half is `first`'s points, the right half `second`'s, one point to each
/// square of [`BLOCK`] pixels, and below [`GROUND`] nothing.
fn shown(first: ReconId, second: ReconId, x: u32, y: u32) -> Option<PointRef> {
    if y >= GROUND {
        return None;
    }
    let recon = if x < SPLIT { first } else { second };
    let index = (x % SPLIT) / BLOCK + 30 * (y / BLOCK);
    Some(PointRef::new(recon, index as usize))
}

/// Two demo reconstructions, each with enough points for every index the
/// picture shows, the first selected.
fn two_nodes() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(1200)));
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(1200)));
    let first = state.scene[0].id;
    state.select_recon(first);
    state
}

/// The renderer's half of a region read: the request a frame leaves is copied
/// when the next frame starts and read when it ends.
#[derive(Default)]
struct Gpu {
    copied: Option<PixelRect>,
}

/// One running viewport, with what a test reads back.
struct Rig {
    viewer: Viewer3D,
    ctx: egui::Context,
    state: AppState,
    gpu: Gpu,
    /// The rect `show` allocated on the last frame.
    rect: egui::Rect,
}

impl Rig {
    fn new() -> Self {
        let mut rig = Self {
            viewer: Viewer3D::new(),
            ctx: egui::Context::default(),
            state: two_nodes(),
            gpu: Gpu::default(),
            rect: egui::Rect::NOTHING,
        };
        rig.viewer.tool = Tool::DeleteBrush;
        rig.frames(2, egui::pos2(300.0, 300.0));
        rig
    }

    fn first(&self) -> ReconId {
        self.state.scene[0].id
    }

    fn second(&self) -> ReconId {
        self.state.scene[1].id
    }

    /// One frame with `events`, the pointer at `pointer` and `modifiers` held.
    fn frame_with(
        &mut self,
        events: Vec<egui::Event>,
        pointer: egui::Pos2,
        modifiers: egui::Modifiers,
    ) {
        self.gpu.copied = self.viewer.pick_region_request.take();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            events: std::iter::once(egui::Event::ModifiersChanged(modifiers))
                .chain(events)
                .collect(),
            ..Default::default()
        };
        crate::platform::set_test_pointer_pos(Some(pointer));
        let viewer = &mut self.viewer;
        let state = &mut self.state;
        let rect = &mut self.rect;
        let mut output = self.ctx.run_ui(input, |ui| {
            let scroll_input = ScrollInput::from_ctx(ui.ctx(), false);
            egui::CentralPanel::default().show(ui, |ui| {
                viewer.show_tool_strip(ui, &mut state.action_log);
                let node = crate::scene::selected_node(&state.scene, state.selected_recon)
                    .expect("a selected node");
                *rect = ui.available_rect_before_wrap();
                viewer.show(
                    ui,
                    node,
                    &state.scene,
                    state.solo,
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
        if let Some(asked) = self.gpu.copied.take() {
            let (first, second) = (self.first(), self.second());
            let mut points = Vec::new();
            for y in asked.y..asked.y + asked.height {
                for x in asked.x..asked.x + asked.width {
                    points.push(shown(first, second, x, y));
                }
            }
            self.viewer.pick_region = Some(PickRegion::new(asked, points));
        }
    }

    fn frame(&mut self, events: Vec<egui::Event>, pointer: egui::Pos2) {
        self.frame_with(events, pointer, egui::Modifiers::NONE);
    }

    /// `count` frames with the pointer resting at `pointer`.
    fn frames(&mut self, count: usize, pointer: egui::Pos2) {
        for _ in 0..count {
            self.frame(vec![egui::Event::PointerMoved(pointer)], pointer);
        }
    }

    /// Press the left button at `path[0]`, move through the rest one point a
    /// frame, release at the last, and let the release's answers arrive.
    fn stroke(&mut self, path: &[egui::Pos2], modifiers: egui::Modifiers) {
        let first = path[0];
        self.frame_with(vec![egui::Event::PointerMoved(first)], first, modifiers);
        self.frame_with(vec![button(first, true, modifiers)], first, modifiers);
        for &pos in &path[1..] {
            self.frame_with(vec![egui::Event::PointerMoved(pos)], pos, modifiers);
        }
        let last = *path.last().expect("a path");
        self.frame_with(vec![button(last, false, modifiers)], last, modifiers);
        self.frames(4, last);
    }

    /// A screen position as a texture pixel position.
    fn texture(&self, pos: egui::Pos2) -> [f32; 2] {
        [pos.x - self.rect.left(), pos.y - self.rect.top()]
    }

    /// The `recon` points whose pixels' centres lie within the brush's radius
    /// of the polyline through `path`: what a stroke along it paints.
    fn expected(&self, recon: ReconId, path: &[egui::Pos2]) -> HashSet<u32> {
        let (first, second) = (self.first(), self.second());
        let points: Vec<[f32; 2]> = path.iter().map(|p| self.texture(*p)).collect();
        let segments: Vec<([f32; 2], [f32; 2])> = if points.len() == 1 {
            vec![(points[0], points[0])]
        } else {
            points.windows(2).map(|w| (w[0], w[1])).collect()
        };
        let mut found = HashSet::new();
        for y in 0..800u32 {
            for x in 0..1200u32 {
                let Some(point) = shown(first, second, x, y).filter(|p| p.recon == recon) else {
                    continue;
                };
                let c = [x as f32 + 0.5, y as f32 + 0.5];
                let near = segments.iter().any(|&(a, b)| {
                    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                    let length_sq = dx * dx + dy * dy;
                    let t = if length_sq > 0.0 {
                        (((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / length_sq).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let (ex, ey) = (c[0] - (a[0] + t * dx), c[1] - (a[1] + t * dy));
                    ex * ex + ey * ey <= BRUSH_RADIUS * BRUSH_RADIUS
                });
                if near {
                    found.insert(point.point);
                }
            }
        }
        found
    }

    fn version_count(&self, index: usize) -> usize {
        self.state.scene[index].history.versions().len()
    }
}

fn button(pos: egui::Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers,
    }
}

/// Where a circle straddles the line between the two reconstructions' halves.
fn on_the_split(rig: &Rig) -> egui::Pos2 {
    rig.rect.min + egui::vec2(SPLIT as f32 - 5.0, 300.0)
}

fn sorted(set: &HashSet<u32>) -> Vec<u32> {
    let mut points: Vec<u32> = set.iter().copied().collect();
    points.sort_unstable();
    points
}

/// The preview marks the selected reconstruction's points under the circle and
/// none of the other's, and a press and release there paints exactly that set
/// and reports it as one gesture.
#[test]
fn the_preview_is_what_a_press_there_paints_and_only_of_the_selected_reconstruction() {
    let mut rig = Rig::new();
    let at = on_the_split(&rig);
    rig.frames(4, at);
    let (first, second) = (rig.first(), rig.second());
    let expected = rig.expected(first, &[at]);
    assert!(!expected.is_empty(), "the circle covers no point");
    let (painted, under) = rig.viewer.brush_marks(first);
    assert!(painted.is_none(), "nothing is painted before a press");
    assert_eq!(under, Some(&expected), "the preview");
    let (_, other) = rig.viewer.brush_marks(second);
    assert!(other.is_none(), "the other reconstruction is previewed");

    let before = rig.version_count(0);
    rig.stroke(&[at], egui::Modifiers::NONE);
    let gesture = rig.viewer.brush_gesture.take().expect("a gesture");
    assert_eq!(
        gesture,
        BrushGesture {
            node: first,
            points: sorted(&expected)
        }
    );
    assert!(rig.viewer.brush_stroke.is_none());
    gesture.apply(&mut rig.state);
    assert_eq!(rig.version_count(0), before + 1, "one version");
    assert_eq!(rig.version_count(1), 1, "the other node was edited");
    let label = rig.state.scene[0].label.clone();
    assert!(
        rig.state.action_log.entries().any(|e| e
            .text
            .starts_with(&format!("Deleted {} points in {label}", expected.len()))),
        "no Deleted entry"
    );
}

/// A stroke paints every point the circle swept, with no gap between the
/// frames of a fast drag, marks them painted while it is held, and never paints
/// the other reconstruction's points even where it crosses them.
#[test]
fn a_stroke_paints_the_swept_capsules_of_the_selected_reconstruction() {
    let mut rig = Rig::new();
    let start = rig.rect.min + egui::vec2(100.0, 300.0);
    let path = [
        start,
        start + egui::vec2(250.0, 0.0),
        start + egui::vec2(650.0, 40.0),
    ];
    let first = rig.first();
    rig.frame(vec![egui::Event::PointerMoved(path[0])], path[0]);
    rig.frame(vec![button(path[0], true, egui::Modifiers::NONE)], path[0]);
    for &pos in &path[1..] {
        rig.frame(vec![egui::Event::PointerMoved(pos)], pos);
    }
    rig.frames(3, path[2]);
    let expected = rig.expected(first, &path);
    let (painted, under) = rig.viewer.brush_marks(first);
    assert_eq!(painted, Some(&expected), "the painted marks while held");
    assert!(under.is_none(), "no preview while a stroke is held");
    assert!(rig.viewer.brush_marks(rig.second()).0.is_none());
    rig.frame(vec![button(path[2], false, egui::Modifiers::NONE)], path[2]);
    rig.frames(4, path[2]);
    let gesture = rig.viewer.brush_gesture.take().expect("a gesture");
    assert_eq!(gesture.points, sorted(&expected));
    assert!(
        rig.viewer.brush_marks(first).0.is_none(),
        "the marks outlived the stroke"
    );
}

/// The release waits for the last capsule's answer, so the final segment of a
/// stroke released straight after its last move is not lost.
#[test]
fn the_release_waits_for_the_last_capsule() {
    let mut rig = Rig::new();
    let start = rig.rect.min + egui::vec2(100.0, 200.0);
    let end = start + egui::vec2(300.0, 0.0);
    rig.frame(vec![egui::Event::PointerMoved(start)], start);
    rig.frame(vec![button(start, true, egui::Modifiers::NONE)], start);
    rig.frame(
        vec![
            egui::Event::PointerMoved(end),
            button(end, false, egui::Modifiers::NONE),
        ],
        end,
    );
    assert!(
        rig.viewer.brush_gesture.is_none(),
        "reported before the answers"
    );
    rig.frames(4, end);
    let gesture = rig.viewer.brush_gesture.take().expect("a gesture");
    assert_eq!(
        gesture.points,
        sorted(&rig.expected(rig.first(), &[start, end]))
    );
}

/// Esc while the button is held cancels the stroke: nothing stays painted,
/// nothing is reported, and the rest of the drag does not orbit.
#[test]
fn escape_cancels_a_stroke_and_the_drag_does_not_orbit() {
    let mut rig = Rig::new();
    let start = rig.rect.min + egui::vec2(100.0, 300.0);
    rig.frame(vec![egui::Event::PointerMoved(start)], start);
    rig.frame(vec![button(start, true, egui::Modifiers::NONE)], start);
    let mid = start + egui::vec2(100.0, 0.0);
    rig.frame(vec![egui::Event::PointerMoved(mid)], mid);
    rig.frames(3, mid);
    assert!(rig
        .viewer
        .brush_marks(rig.first())
        .0
        .is_some_and(|p| !p.is_empty()));
    let orientation = rig.viewer.camera.camera.orientation;
    rig.frame(
        vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        mid,
    );
    assert!(
        rig.viewer
            .brush_marks(rig.first())
            .0
            .is_none_or(HashSet::is_empty),
        "the painted points kept their mark"
    );
    let end = mid + egui::vec2(100.0, 50.0);
    rig.frame(vec![egui::Event::PointerMoved(end)], end);
    rig.frame(vec![button(end, false, egui::Modifiers::NONE)], end);
    rig.frames(4, end);
    assert!(rig.viewer.brush_gesture.is_none());
    assert!(rig.viewer.brush_stroke.is_none());
    assert_eq!(
        rig.viewer.camera.camera.orientation, orientation,
        "it orbited"
    );
    assert_eq!(rig.version_count(0), 1);
}

/// A selection change while the button is held cancels the stroke.
#[test]
fn a_selection_change_while_held_cancels_the_stroke() {
    let mut rig = Rig::new();
    let start = rig.rect.min + egui::vec2(100.0, 300.0);
    rig.frame(vec![egui::Event::PointerMoved(start)], start);
    rig.frame(vec![button(start, true, egui::Modifiers::NONE)], start);
    rig.frames(3, start);
    let second = rig.second();
    rig.state.select_recon(second);
    let end = start + egui::vec2(50.0, 0.0);
    rig.frame(vec![egui::Event::PointerMoved(end)], end);
    rig.frame(vec![button(end, false, egui::Modifiers::NONE)], end);
    rig.frames(4, end);
    assert!(rig.viewer.brush_gesture.is_none());
    assert!(rig.viewer.brush_stroke.is_none());
}

/// A stroke over nothing but background reports nothing, so no version is
/// pushed.
#[test]
fn a_stroke_that_painted_nothing_reports_nothing() {
    let mut rig = Rig::new();
    let at = rig.rect.min + egui::vec2(300.0, GROUND as f32 + 60.0);
    rig.stroke(&[at, at + egui::vec2(80.0, 0.0)], egui::Modifiers::NONE);
    assert!(rig.viewer.brush_gesture.is_none());
    assert_eq!(rig.version_count(0), 1);
}

/// A press while the platform says the right or the middle button is down is
/// that button's, as Windows reports every button as the primary one, so it
/// starts no stroke.
#[test]
fn a_press_with_another_button_down_starts_no_stroke() {
    let mut rig = Rig::new();
    let at = rig.rect.min + egui::vec2(300.0, 300.0);
    crate::platform::set_test_other_button_down(true);
    rig.frame(vec![egui::Event::PointerMoved(at)], at);
    rig.frame(vec![button(at, true, egui::Modifiers::NONE)], at);
    let held = rig.viewer.brush_stroke.is_some();
    rig.frame(vec![button(at, false, egui::Modifiers::NONE)], at);
    crate::platform::set_test_other_button_down(false);
    rig.frames(4, at);
    assert!(!held, "a stroke started");
    assert!(rig.viewer.brush_gesture.is_none());
}

fn alt() -> egui::Modifiers {
    egui::Modifiers {
        alt: true,
        ..egui::Modifiers::NONE
    }
}

/// With the brush on, Alt + left drag orbits in free view, which in Navigate is
/// the nodal pan that keeps the camera where it stands, and paints nothing.
#[test]
fn with_the_brush_on_alt_drag_orbits_and_paints_nothing() {
    let drag = |tool: Tool| {
        let mut rig = Rig::new();
        rig.viewer.tool = tool;
        let start = rig.rect.min + egui::vec2(300.0, 300.0);
        let position = rig.viewer.camera.camera.position;
        let path: Vec<egui::Pos2> = (0..5)
            .map(|i| start + egui::vec2(30.0 * i as f32, 0.0))
            .collect();
        rig.stroke(&path, alt());
        assert!(rig.viewer.brush_gesture.is_none(), "{tool:?} painted");
        (rig.viewer.camera.camera.position - position).norm()
    };
    assert!(
        drag(Tool::DeleteBrush) > 1e-6,
        "the brush's Alt drag did not orbit"
    );
    assert!(
        drag(Tool::Navigate) < 1e-9,
        "Navigate's Alt drag moved the camera"
    );
}

/// In camera view, Alt + left drag orbits out of it with the brush on, as it
/// does in Navigate, and paints nothing.
#[test]
fn in_camera_view_alt_drag_orbits_with_the_brush_on() {
    let mut rig = Rig::new();
    let image = ImageRef::new(rig.first(), 0);
    rig.viewer.jump_to_camera_view(image, &rig.state.scene[0]);
    let position = rig.viewer.camera.camera.position;
    let start = rig.rect.min + egui::vec2(300.0, 300.0);
    let path: Vec<egui::Pos2> = (0..5)
        .map(|i| start + egui::vec2(30.0 * i as f32, 0.0))
        .collect();
    rig.stroke(&path, alt());
    assert!(rig.viewer.camera_view.is_none(), "still in camera view");
    assert!((rig.viewer.camera.camera.position - position).norm() > 1e-6);
    assert!(rig.viewer.brush_gesture.is_none());
}

/// Choosing Navigate ends the preview, and a stroke held then is cancelled.
#[test]
fn leaving_the_brush_cancels_the_stroke() {
    let mut rig = Rig::new();
    let start = rig.rect.min + egui::vec2(100.0, 300.0);
    rig.frame(vec![egui::Event::PointerMoved(start)], start);
    rig.frame(vec![button(start, true, egui::Modifiers::NONE)], start);
    rig.frames(3, start);
    rig.viewer.tool = Tool::Navigate;
    rig.frame(vec![button(start, false, egui::Modifiers::NONE)], start);
    rig.frames(4, start);
    assert!(rig.viewer.brush_gesture.is_none());
    assert_eq!(rig.viewer.brush_marks(rig.first()), (None, None));
}

/// Taking a camera in hand switches the tool to Navigate and refuses the
/// brush until the lock ends.
#[test]
fn the_move_camera_lock_leaves_the_brush_and_refuses_it() {
    let mut rig = Rig::new();
    let image = ImageRef::new(rig.first(), 0);
    rig.viewer.jump_to_camera_view(image, &rig.state.scene[0]);
    crate::camera_lock::enter(&mut rig.viewer, &mut rig.state).expect("the lock");
    assert_eq!(rig.viewer.tool, Tool::Navigate);
    assert!(rig.viewer.brush_refusal().is_some());
}

/// The tool strip's lower button chooses the brush and records the choice; the
/// upper one goes back to Navigate.
#[test]
fn the_tool_strip_chooses_the_tool() {
    let mut rig = Rig::new();
    rig.viewer.tool = Tool::Navigate;
    let strip = rig.viewer.tool_strip_rect.expect("the strip was drawn");
    let lower = egui::pos2(strip.center().x, strip.bottom() - 19.0);
    let upper = egui::pos2(strip.center().x, strip.top() + 19.0);
    for (at, tool) in [(lower, Tool::DeleteBrush), (upper, Tool::Navigate)] {
        rig.frame(vec![egui::Event::PointerMoved(at)], at);
        rig.frame(vec![button(at, true, egui::Modifiers::NONE)], at);
        rig.frame(vec![button(at, false, egui::Modifiers::NONE)], at);
        assert_eq!(rig.viewer.tool, tool);
        assert!(
            rig.viewer.brush_stroke.is_none(),
            "the click reached the viewport"
        );
        // Two choices less than a second apart fold into one row holding the
        // last.
        let last = rig
            .state
            .action_log
            .entries()
            .last()
            .map(|e| e.text.clone());
        assert_eq!(
            last.as_deref(),
            Some(format!("{} tool", tool.label()).as_str())
        );
    }
}
