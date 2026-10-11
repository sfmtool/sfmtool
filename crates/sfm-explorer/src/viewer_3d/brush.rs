// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The eraser brush: the tool strip that chooses it, the preview of the points
//! under its circle, and the stroke that paints the points it passes over and
//! deletes them on release as one version.
//!
//! See `specs/gui/edits/eraser-brush.md`. The brush reads what is on screen
//! through the renderer's region read (`specs/gui/camera-views.md` § "Region
//! reads"): a frame that wants one names a rectangle in
//! [`Viewer3D::pick_region_request`], and the answer arrives in
//! [`Viewer3D::pick_region`] two frames later. Each request is kept with the
//! capsule it was made for until its answer arrives, and the answer is matched
//! to it by the rectangle it named.

use std::collections::HashSet;

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};

use super::Viewer3D;
use crate::action_log::{ActionLog, Kind};
use crate::document::VersionSerial;
use crate::scene::{ReconId, SceneNode};
use crate::scene_renderer::{points_in_capsule, PixelRect};
use crate::state::AppState;

/// The radius of the brush's circle, in logical pixels.
pub(crate) const BRUSH_RADIUS: f32 = 24.0;

/// How many frames after a region request its answer may still arrive. The
/// answer to a request made in frame N is read after frame N + 1's present and
/// seen in frame N + 2; a request that names nothing left on the texture (a
/// stroke dragged off the viewport) is never answered, and is given up after
/// this many frames so the release does not wait on it.
const ANSWER_FRAMES: u64 = 3;

/// Side of one tool strip button, in logical pixels.
const TOOL_BUTTON_SIZE: f32 = 28.0;

/// Inset of the tool strip from the viewport's left edge.
const TOOL_STRIP_INSET: f32 = 8.0;

/// Why the brush cannot be chosen while a camera is in hand: the tooltip of
/// the greyed button and the end of the wire's refusal.
pub(crate) const BRUSH_LOCK_REFUSAL: &str =
    "A camera is being moved; M or Enter commits it, Esc cancels it.";

/// What an unmodified left drag in the 3D viewport does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Tool {
    /// Orbit, pan, zoom and click to select: the viewport's navigation.
    #[default]
    Navigate,
    /// Paint points and delete them on release.
    EraserBrush,
}

impl Tool {
    /// Every tool, in the strip's order from the top.
    pub(crate) const ALL: [Tool; 2] = [Tool::Navigate, Tool::EraserBrush];

    /// The name on the wire.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Tool::Navigate => "navigate",
            Tool::EraserBrush => "eraser_brush",
        }
    }

    /// The tool a wire name names, or `None`.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn from_wire_name(name: &str) -> Option<Tool> {
        Tool::ALL.into_iter().find(|tool| tool.wire_name() == name)
    }

    /// The tool's name: its button's accessible name and the start of its
    /// Action Log entry.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Tool::Navigate => "Navigate",
            Tool::EraserBrush => "Eraser brush",
        }
    }

    /// The button's tooltip.
    fn tooltip(self) -> &'static str {
        match self {
            Tool::Navigate => "Navigate: orbit, pan and zoom the view, and click to select.",
            Tool::EraserBrush => {
                "Eraser brush: drag over points to delete them from the selected reconstruction."
            }
        }
    }
}

/// What the view looked like when a region was asked for. A read is repeated
/// only when this or the pointer changed, since otherwise the answer would be
/// the one already held.
///
/// The node's version is part of it because an edit or an undo changes which
/// points are drawn under a still pointer, and the camera is the whole of what
/// moves the picture otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ViewKey {
    position: [f64; 3],
    orientation: [f64; 4],
    fov: f64,
    rect: Rect,
    pixels_per_point: f32,
    node: ReconId,
    version: VersionSerial,
}

/// One region request, kept with the capsule it was made for until its answer
/// arrives. Positions and the radius are in physical pixels of the scene
/// texture.
#[derive(Debug, Clone, Copy)]
struct Asked {
    rect: PixelRect,
    from: [f32; 2],
    to: [f32; 2],
    radius: f32,
    frame: u64,
}

/// A stroke of the eraser brush, from the press until the edit is reported or
/// the stroke is cancelled.
///
/// Held on the viewport, as the bench handle drag is, because what the stroke
/// edits was decided when the button went down: the node selected then.
pub(crate) struct BrushStroke {
    /// The reconstruction selected at the press, the only one the stroke paints.
    pub(crate) node: ReconId,
    /// The edited indexes the stroke has painted so far.
    pub(crate) painted: HashSet<u32>,
    /// The cursor position the last capsule ended at, in texture pixels.
    last: [f32; 2],
    /// The view the last capsule was read in, `None` before the first.
    view: Option<ViewKey>,
    /// The capsules asked for and not yet answered.
    asked: Vec<Asked>,
    /// The button came up; the stroke waits for its last answers, then reports.
    released: bool,
    /// Esc, a selection change or a change of tool ended it. The pointer stays
    /// the stroke's until the button comes up, so the drag does not turn into
    /// an orbit halfway through.
    cancelled: bool,
}

/// The points under the brush's circle before a press.
#[derive(Default)]
pub(crate) struct BrushPreview {
    /// The node the points belong to.
    node: Option<ReconId>,
    /// The edited indexes under the circle, as of the last answer.
    points: HashSet<u32>,
    /// Where the circle was and what the view was at the last request.
    at: Option<([f32; 2], ViewKey)>,
    /// The circles asked for and not yet answered.
    asked: Vec<Asked>,
}

/// What a released stroke asks of the app: delete these points of this node
/// as one version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BrushGesture {
    pub(crate) node: ReconId,
    pub(crate) points: Vec<u32>,
}

impl BrushGesture {
    /// Delete the painted points through [`AppState::delete_points`], which
    /// writes the version's Action Log entry. A refusal, a busy node above all,
    /// is the status line's.
    pub(crate) fn apply(self, state: &mut AppState) {
        if let Err(why) = state.delete_points(self.node, &self.points) {
            state.action_log.fail(Kind::Edit, why);
        }
    }
}

/// Record the tool chosen, folding into the last entry when that one was a
/// tool choice made less than a second before.
fn record_tool(log: &mut ActionLog, tool: Tool) {
    log.record_run(Kind::Display, "Tool", format!("{} tool", tool.label()));
}

/// `pos`, a point of the viewport in logical pixels, in physical pixels of the
/// scene texture.
fn texture_px(pos: Pos2, rect: Rect, pixels_per_point: f32) -> [f32; 2] {
    [
        (pos.x - rect.left()) * pixels_per_point,
        (pos.y - rect.top()) * pixels_per_point,
    ]
}

impl Viewer3D {
    /// Choose `tool`, and record the choice when it changed it.
    ///
    /// A stroke held when the tool leaves the brush is cancelled on the next
    /// frame. The caller checks [`Self::brush_refusal`] first where the brush
    /// can be refused.
    pub(crate) fn set_tool(&mut self, tool: Tool, log: &mut ActionLog) {
        if self.tool != tool {
            self.tool = tool;
            record_tool(log, tool);
        }
    }

    /// Why the brush cannot be chosen now, or `None`: a camera is in hand, and
    /// the lock is an edit of its own in progress.
    pub(crate) fn brush_refusal(&self) -> Option<&'static str> {
        self.camera_lock.is_some().then_some(BRUSH_LOCK_REFUSAL)
    }

    /// The brush's marks on `id`'s points: the painted set of a stroke on `id`,
    /// and the preview's set when it is of `id`. What the frame hands
    /// `SceneRenderer::update_point_mask` beside the deleted set.
    pub(crate) fn brush_marks(
        &self,
        id: ReconId,
    ) -> (Option<&HashSet<u32>>, Option<&HashSet<u32>>) {
        let painted = self
            .brush_stroke
            .as_ref()
            .filter(|stroke| stroke.node == id)
            .map(|stroke| &stroke.painted);
        let under = (self.tool == Tool::EraserBrush && self.brush_stroke.is_none())
            .then_some(&self.brush_preview)
            .filter(|preview| preview.node == Some(id))
            .map(|preview| &preview.points);
        (painted, under)
    }

    /// Draw the tool strip at the left edge of the viewport and record the rect
    /// it occupies in [`Viewer3D::tool_strip_rect`].
    ///
    /// Called beside [`Viewer3D::show_hud`], before [`Viewer3D::show`], for the
    /// HUD's reason: it is an `egui::Area` on a layer above the viewport, so a
    /// click on it never reaches the viewport, and its rect joins the HUD's in
    /// the gates of the input paths that read the pointer geometrically.
    pub(crate) fn show_tool_strip(&mut self, ui: &mut egui::Ui, log: &mut ActionLog) {
        let viewport = ui.available_rect_before_wrap();
        if viewport.width() <= 0.0 || viewport.height() <= 0.0 {
            self.tool_strip_rect = None;
            return;
        }
        let refusal = self.brush_refusal();
        let anchor = egui::pos2(viewport.left() + TOOL_STRIP_INSET, viewport.center().y);
        let ctx = ui.ctx().clone();
        let area = egui::Area::new(egui::Id::new("viewport_tool_strip"))
            .order(egui::Order::Middle)
            .pivot(egui::Align2::LEFT_CENTER)
            .fixed_pos(anchor)
            .constrain_to(viewport)
            .show(&ctx, |ui| {
                let style = ui.style();
                let frame = egui::Frame::popup(style)
                    .fill(style.visuals.window_fill.gamma_multiply(0.88))
                    .inner_margin(4.0);
                frame.show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
                    ui.vertical(|ui| {
                        for tool in Tool::ALL {
                            let enabled = tool == Tool::Navigate || refusal.is_none();
                            let selected = self.tool == tool;
                            let response = ui
                                .add_enabled_ui(enabled, |ui| tool_button(ui, tool, selected))
                                .inner;
                            let response = match refusal.filter(|_| !enabled) {
                                Some(why) => response.on_disabled_hover_text(why),
                                None => response.on_hover_text(tool.tooltip()),
                            };
                            if response.clicked() {
                                self.set_tool(tool, log);
                            }
                        }
                    });
                });
            });
        self.tool_strip_rect = Some(area.response.rect);
    }

    /// The first half of the brush's frame, before the viewport's own input:
    /// take an answer that arrived, start a stroke on an unmodified left press,
    /// and cancel one that Esc, a selection change or a change of tool ended.
    ///
    /// Returns whether the pointer is the stroke's, in which case the drag and
    /// the click handlers are skipped, as they are for a bench handle drag.
    pub(super) fn begin_brush_frame(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        node: ReconId,
    ) -> bool {
        self.brush_frame += 1;
        let frame = self.brush_frame;
        self.take_brush_answer();
        let answerable = |asked: &Asked| asked.frame + ANSWER_FRAMES >= frame;
        self.brush_preview.asked.retain(answerable);
        if let Some(stroke) = &mut self.brush_stroke {
            stroke.asked.retain(answerable);
        }

        let (pressed, escape, modifiers) = ui.input(|i| {
            (
                i.pointer.primary_pressed(),
                i.key_pressed(egui::Key::Escape),
                i.modifiers,
            )
        });
        if let Some(stroke) = &mut self.brush_stroke {
            if !stroke.cancelled
                && (escape || stroke.node != node || self.tool != Tool::EraserBrush)
            {
                stroke.cancelled = true;
                stroke.painted.clear();
                stroke.asked.clear();
            }
        }

        // Only an unmodified press paints: Alt, Ctrl and Shift drags and the
        // middle and right buttons stay navigation's. On Windows every button
        // reaches egui as the primary one, so the platform's own button state
        // says whether this press was the left button alone.
        let plain = !(modifiers.alt || modifiers.ctrl || modifiers.command || modifiers.shift)
            && !crate::platform::other_mouse_button_down();
        if self.tool == Tool::EraserBrush
            && self.brush_stroke.is_none()
            && pressed
            && plain
            && response.contains_pointer()
        {
            if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                let at = texture_px(pos, response.rect, ui.ctx().pixels_per_point());
                // An easing camera would move the picture under the stroke.
                self.cancel_transition();
                self.brush_preview = BrushPreview::default();
                self.brush_stroke = Some(BrushStroke {
                    node,
                    painted: HashSet::new(),
                    last: at,
                    view: None,
                    asked: Vec::new(),
                    released: false,
                    cancelled: false,
                });
            }
        }
        if self.tool != Tool::EraserBrush {
            self.brush_preview = BrushPreview::default();
        }
        self.brush_stroke.is_some()
    }

    /// The second half of the brush's frame, after the viewport's own input has
    /// moved the view: ask for the capsule the circle swept since the last
    /// request, or for the circle the preview reads, and report a released
    /// stroke once its last answer is in.
    ///
    /// A region is asked for only when the pointer or the view changed since
    /// the last request; otherwise the last answer stands.
    pub(super) fn end_brush_frame(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        node: &SceneNode,
    ) {
        let rect = response.rect;
        let ppp = ui.ctx().pixels_per_point();
        let radius = BRUSH_RADIUS * ppp;
        let frame = self.brush_frame;
        let view = ViewKey {
            position: self.camera.camera.position.coords.into(),
            orientation: self.camera.camera.orientation.coords.into(),
            fov: self.camera.fov,
            rect,
            pixels_per_point: ppp,
            node: node.id,
            version: node.history.current_version().serial,
        };
        let down = ui.input(|i| i.pointer.primary_down());

        if let Some(stroke) = &mut self.brush_stroke {
            if !stroke.cancelled && !stroke.released {
                let at = ui
                    .input(|i| i.pointer.latest_pos())
                    .map(|pos| texture_px(pos, rect, ppp))
                    .unwrap_or(stroke.last);
                // The first frame reads the circle at the press, since no view
                // has been read yet.
                if stroke.view != Some(view) || at != stroke.last {
                    let asked = Asked {
                        rect: PixelRect::around_capsule(stroke.last, at, radius),
                        from: stroke.last,
                        to: at,
                        radius,
                        frame,
                    };
                    self.pick_region_request = Some(asked.rect);
                    stroke.asked.push(asked);
                    stroke.last = at;
                    stroke.view = Some(view);
                }
            }
            if !down {
                stroke.released = true;
            }
            if stroke.released && (stroke.cancelled || stroke.asked.is_empty()) {
                let stroke = self.brush_stroke.take().expect("held above");
                if !stroke.cancelled && !stroke.painted.is_empty() {
                    let mut points: Vec<u32> = stroke.painted.into_iter().collect();
                    points.sort_unstable();
                    self.brush_gesture = Some(BrushGesture {
                        node: stroke.node,
                        points,
                    });
                }
            }
        } else if self.tool == Tool::EraserBrush {
            match response.hover_pos() {
                Some(pos) => {
                    let at = texture_px(pos, rect, ppp);
                    let preview = &mut self.brush_preview;
                    if preview.node != Some(node.id) {
                        preview.points.clear();
                        preview.node = Some(node.id);
                    }
                    if preview.at != Some((at, view)) {
                        let asked = Asked {
                            rect: PixelRect::around_capsule(at, at, radius),
                            from: at,
                            to: at,
                            radius,
                            frame,
                        };
                        self.pick_region_request = Some(asked.rect);
                        preview.asked.push(asked);
                        preview.at = Some((at, view));
                    }
                }
                // Off the viewport, or over the tool strip or the HUD, the
                // circle is not over anything.
                None => self.brush_preview = BrushPreview::default(),
            }
        }

        let waiting = self
            .brush_stroke
            .as_ref()
            .is_some_and(|stroke| !stroke.asked.is_empty())
            || !self.brush_preview.asked.is_empty();
        if waiting {
            ui.ctx().request_repaint();
        }
    }

    /// Hand the region that arrived to the request it answers: a stroke's
    /// capsule adds what it covers to the painted set, a preview's circle
    /// replaces the preview. Requests older than the one answered were lost
    /// and are dropped with it.
    fn take_brush_answer(&mut self) {
        let Some(region) = self.pick_region.take() else {
            return;
        };
        let requested = region.requested();
        if let Some(stroke) = &mut self.brush_stroke {
            if let Some(i) = stroke.asked.iter().position(|a| a.rect == requested) {
                let asked = stroke.asked[i];
                stroke.asked.drain(..=i);
                if !stroke.cancelled {
                    stroke.painted.extend(points_in_capsule(
                        &region,
                        stroke.node,
                        asked.from,
                        asked.to,
                        asked.radius,
                    ));
                }
            }
            return;
        }
        let preview = &mut self.brush_preview;
        let (Some(node), Some(i)) = (
            preview.node,
            preview.asked.iter().position(|a| a.rect == requested),
        ) else {
            return;
        };
        let asked = preview.asked[i];
        preview.asked.drain(..=i);
        preview.points = points_in_capsule(&region, node, asked.from, asked.to, asked.radius);
    }

    /// Draw the brush's circle at the pointer, and show the crosshair at its
    /// centre, while the brush is on and the pointer is over the viewport or a
    /// stroke is held. Painted on the viewport's painter, as the lock banner
    /// is: egui has no circle cursor.
    pub(super) fn draw_brush(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        response: &egui::Response,
    ) {
        if self.tool != Tool::EraserBrush {
            return;
        }
        let pos = match &self.brush_stroke {
            Some(_) => ui.input(|i| i.pointer.latest_pos()),
            None => response.hover_pos(),
        };
        let Some(pos) = pos else {
            return;
        };
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        let painting = self
            .brush_stroke
            .as_ref()
            .is_some_and(|stroke| !stroke.cancelled);
        let ring = if painting {
            Color32::from_rgb(255, 110, 110)
        } else {
            Color32::from_rgb(235, 235, 235)
        };
        painter.circle_stroke(
            pos,
            BRUSH_RADIUS,
            Stroke::new(3.0, Color32::from_black_alpha(160)),
        );
        painter.circle_stroke(pos, BRUSH_RADIUS, Stroke::new(1.5, ring));
    }
}

/// One button of the tool strip: a square with the tool's icon, drawn as the
/// selected one when `selected`. Reported to accessibility as a radio button,
/// since exactly one tool is on.
fn tool_button(ui: &mut egui::Ui, tool: Tool, selected: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE),
        egui::Sense::click(),
    );
    let enabled = ui.is_enabled();
    response.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::RadioButton,
            enabled,
            selected,
            tool.label(),
        )
    });
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact_selectable(&response, selected);
        let painter = ui.painter();
        painter.rect(
            rect,
            visuals.corner_radius,
            visuals.weak_bg_fill,
            visuals.bg_stroke,
            egui::StrokeKind::Inside,
        );
        let color = visuals.fg_stroke.color;
        match tool {
            Tool::Navigate => paint_cursor_icon(painter, rect, color),
            Tool::EraserBrush => paint_eraser_icon(painter, rect, color),
        }
    }
    response
}

/// A mouse pointer arrow, in the middle of `rect`.
fn paint_cursor_icon(painter: &egui::Painter, rect: Rect, color: Color32) {
    // The arrow's outline on a 16-unit grid, tip at the top left.
    const OUTLINE: [[f32; 2]; 7] = [
        [0.0, 0.0],
        [0.0, 13.0],
        [3.5, 9.8],
        [6.0, 15.0],
        [8.2, 14.0],
        [5.8, 8.8],
        [10.0, 8.8],
    ];
    let origin = rect.center() - egui::vec2(5.0, 7.5);
    let points = OUTLINE
        .iter()
        .map(|[x, y]| origin + egui::vec2(*x, *y))
        .collect();
    painter.add(egui::Shape::closed_line(points, Stroke::new(1.5, color)));
}

/// An eraser held at an angle: a filled tip and an outlined body.
fn paint_eraser_icon(painter: &egui::Painter, rect: Rect, color: Color32) {
    let c = rect.center();
    // Along the eraser and across it, at 45 degrees.
    let along = egui::vec2(1.0, -1.0) / std::f32::consts::SQRT_2;
    let across = egui::vec2(1.0, 1.0) / std::f32::consts::SQRT_2;
    let (length, width) = (16.0, 7.0);
    let corner = |a: f32, b: f32| c + along * (a * length) + across * (b * width);
    let body = vec![
        corner(-0.5, -0.5),
        corner(0.5, -0.5),
        corner(0.5, 0.5),
        corner(-0.5, 0.5),
    ];
    let tip = vec![
        corner(-0.5, -0.5),
        corner(-0.1, -0.5),
        corner(-0.1, 0.5),
        corner(-0.5, 0.5),
    ];
    painter.add(egui::Shape::convex_polygon(tip, color, Stroke::NONE));
    painter.add(egui::Shape::closed_line(body, Stroke::new(1.5, color)));
    // The line it rubs along.
    let base_y = rect.bottom() - 5.0;
    painter.line_segment(
        [
            egui::pos2(rect.left() + 6.0, base_y),
            egui::pos2(rect.right() - 6.0, base_y),
        ],
        Stroke::new(1.0, color),
    );
}

#[cfg(test)]
mod tests;
