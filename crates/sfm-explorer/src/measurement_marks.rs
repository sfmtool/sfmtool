// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurements the 3D view and Image Detail draw: which ones, the marks
//! and lines each puts on screen, what a press on one of them asks for, and
//! the off-screen indicator that points to the selected one.
//!
//! The two views share everything here but where a mark lands. The 3D view
//! projects an endpoint's resolved position through its camera and Image
//! Detail puts it at the endpoint's pixel; each builds a [`Screen`] of panel
//! positions from the [`MeasurementLayer`] the dock hands it, and from there
//! the hit test, the hover text and the press are this module's, so a mark
//! means the same thing in both views (G8). Image Detail paints the marks and
//! lines with [`Screen::draw`]; the 3D view draws the same marks and lines, at
//! the same sizes and in the same colours, in its scene pass
//! (`viewer_3d::measurements::instances`), and paints only the indicator
//! ([`Screen::draw_indicator`]).
//!
//! See `specs/gui/measurement-view.md` § "Drawing measurements".

use std::sync::Arc;

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

use crate::measurements::{self, EndpointSide, MeasurementRef};
use crate::scene::ReconId;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// The colour a stored measurement of a node drawn in its own colours is
/// drawn in. A tinted node's measurements take its tint.
pub(crate) const STORED_RGB: [u8; 3] = [245, 245, 245];

/// Radius of an endpoint's mark, in panel points.
pub(crate) const MARK_RADIUS: f32 = 5.0;

/// How much larger the selected endpoint's mark is drawn, as the 3D view
/// draws the selected observation's circle larger.
pub(crate) const SELECTED_MARK_SCALE: f32 = 1.6;

/// Stroke width of a drawn measurement's line and marks.
pub(crate) const LINE_WIDTH: f32 = 2.0;

/// Stroke width of the selected measurement while every measurement is drawn.
pub(crate) const EMPHASISED_LINE_WIDTH: f32 = 3.5;

/// Stroke width of the focused item, the bench layers' stroke.
pub(crate) const ITEM_LINE_WIDTH: f32 = 2.5;

/// How opaque the measurements other than the selected one are while every
/// measurement is drawn, so the selected one reads as brighter.
pub(crate) const OTHER_OPACITY: f32 = 0.55;

/// How far from a mark's centre the pointer still takes it, in panel points.
pub(crate) const MARK_REACH: f32 = 9.0;

/// How far from a line the pointer still takes it, in panel points.
pub(crate) const LINE_REACH: f32 = 6.0;

/// How far the pointer may travel from its press before the press is a drag
/// rather than a click, in panel points.
pub(crate) const CLICK_TRAVEL: f32 = 3.0;

/// How far inside the view's edge the off-screen indicator's arrow tip sits.
pub(crate) const INDICATOR_INSET: f32 = 24.0;

/// Length of the indicator's arrowhead, in panel points.
const ARROW_LENGTH: f32 = 14.0;

/// Which measurement a drawn one is: a stored row, or the focused item on its
/// node's bench, named by its label there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DrawnId {
    Stored(MeasurementRef),
    Item { node: ReconId, label: String },
}

/// Where an endpoint resolves, as the 3D view needs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Place {
    /// A finite position in the shared world space: the resolved position put
    /// through the node's display transform, as its points are.
    Finite(nalgebra::Point3<f64>),
    /// The endpoint lies on a point at infinity.
    AtInfinity,
    /// The resolver refused the endpoint.
    Refused,
}

/// One placed endpoint of a drawn measurement.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DrawnEndpoint {
    pub(crate) side: EndpointSide,
    /// The image the pixel is in.
    pub(crate) image: u32,
    /// The pixel, which is what the measurement stores.
    pub(crate) pixel: [f64; 2],
    pub(crate) place: Place,
}

/// Why a measurement is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// Drawn only because *Show all measurements* is on.
    Other,
    /// The selected measurement.
    Selected,
    /// The focused measurement item, drawn as the bench's layer.
    Item,
}

/// One measurement as both views draw it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DrawnMeasurement {
    pub(crate) id: DrawnId,
    pub(crate) node: ReconId,
    pub(crate) role: Role,
    /// `Distance #2`, or the item's label on the bench: the indicator's label.
    pub(crate) title: String,
    /// What hovering a mark or the line shows: the Scene tree row's text.
    pub(crate) hover: String,
    /// The placed endpoints, `A` before `B`.
    pub(crate) endpoints: Vec<DrawnEndpoint>,
    pub(crate) color: Color32,
    /// Its line's width.
    pub(crate) width: f32,
    /// The endpoint drawn larger: the selected endpoint, on the item while
    /// one is focused and on the selected measurement otherwise.
    pub(crate) selected_endpoint: Option<EndpointSide>,
    /// Whether its marks in Image Detail are handles: the focused item, on a
    /// bench that takes steps.
    pub(crate) handles: bool,
}

impl DrawnMeasurement {
    /// The finite positions of its endpoints, in the shared world space.
    pub(crate) fn world_points(&self) -> Vec<nalgebra::Point3<f64>> {
        self.endpoints
            .iter()
            .filter_map(|e| match e.place {
                Place::Finite(p) => Some(p),
                _ => None,
            })
            .collect()
    }
}

/// The measurements a view draws on one frame, in the order they are drawn:
/// the others, then the selected one, then the focused item on top.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct MeasurementLayer {
    pub(crate) drawn: Vec<DrawnMeasurement>,
}

impl MeasurementLayer {
    /// The selected measurement, where it is drawn: what the off-screen
    /// indicator follows.
    pub(crate) fn selected(&self) -> Option<&DrawnMeasurement> {
        self.drawn.iter().find(|d| d.role == Role::Selected)
    }

    /// The drawn measurement `id` names.
    pub(crate) fn find(&self, id: &DrawnId) -> Option<&DrawnMeasurement> {
        self.drawn.iter().find(|d| &d.id == id)
    }
}

/// Everything [`AppState::measurement_layer`] builds the 3D view's
/// measurements from that can change their rings, dots and lines: what the
/// 3D view keys the instances it builds from a layer on, so it builds them
/// again only when this changes and never for a camera move.
///
/// The focused item is compared by its `Arc`, which a step on it replaces,
/// as [`crate::scene::SceneNode::item_reading`] compares it.
#[derive(Debug, Clone, Default)]
pub(crate) struct MeasurementLayerKey {
    show_all: bool,
    selected: Option<MeasurementRef>,
    selected_endpoint: Option<EndpointSide>,
    item: Option<Arc<sfmtool_core::bench::EditableMeasurement>>,
    /// Every visible node and the focused item's, in scene order.
    nodes: Vec<NodeKey>,
}

/// What [`MeasurementLayerKey`] holds of one node.
#[derive(Debug, Clone, PartialEq)]
struct NodeKey {
    id: ReconId,
    visible: bool,
    /// The version's document serial, which the stored measurements, their
    /// readings and the item's reading follow.
    document: crate::document::VersionSerial,
    /// The display transform as its quaternion, translation and scale.
    transform: [f64; 8],
    /// Which colour its stored measurements are drawn in.
    tint: crate::scene::NodeTint,
}

impl PartialEq for MeasurementLayerKey {
    fn eq(&self, other: &Self) -> bool {
        let item = match (&self.item, &other.item) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        item && self.show_all == other.show_all
            && self.selected == other.selected
            && self.selected_endpoint == other.selected_endpoint
            && self.nodes == other.nodes
    }
}

impl AppState {
    /// What [`Self::measurement_layer`] builds the 3D view's layer from, with
    /// *Show all measurements* `show_all`, as far as it changes how the
    /// measurements are drawn ([`MeasurementLayerKey`]).
    pub(crate) fn measurement_layer_key(&self, show_all: bool) -> MeasurementLayerKey {
        let item = self.focused_item().and_then(|focused| {
            let label = self.focused_measurement_label(focused.node)?;
            Some((
                focused.node,
                Arc::clone(self.bench_measurement(focused.node, label)?),
            ))
        });
        let nodes = self
            .scene
            .iter()
            .filter_map(|node| {
                let visible = crate::scene::is_visible(node, self.solo);
                if !visible && item.as_ref().is_none_or(|(id, _)| *id != node.id) {
                    return None;
                }
                let transform = node.transform();
                let [qw, qx, qy, qz] = transform.rotation.to_wxyz_array();
                let t = transform.translation;
                Some(NodeKey {
                    id: node.id,
                    visible,
                    document: node.history.current_version().document_serial,
                    transform: [qw, qx, qy, qz, t.x, t.y, t.z, transform.scale],
                    tint: node.tint,
                })
            })
            .collect();
        MeasurementLayerKey {
            show_all,
            selected: self.selected_measurement,
            selected_endpoint: self.selected_endpoint,
            item: item.map(|(_, item)| item),
            nodes,
        }
    }

    /// What the 3D view or Image Detail draws this frame, with that view's
    /// *Show all measurements* option `show_all`.
    ///
    /// The selected measurement where its reconstruction is visible, every
    /// stored measurement of every visible reconstruction with `show_all` on,
    /// and the focused measurement item, which the bench layers draw whatever
    /// the node's eye says. A stored measurement's endpoints are read from the
    /// node's cached readings ([`measurements::readings`]) and an item's live
    /// ([`crate::scene::SceneNode::item_reading`], shared with Measurement
    /// View), each finite position put through the
    /// node's display transform.
    pub(crate) fn measurement_layer(&self, show_all: bool) -> MeasurementLayer {
        let item_focused = self
            .focused_item()
            .is_some_and(|f| self.focused_measurement_label(f.node).is_some());
        let mut others = Vec::new();
        let mut selected = Vec::new();
        for node in &self.scene {
            // A node that draws nothing here is not resolved for it.
            let selected_here = self
                .selected_measurement
                .is_some_and(|m| m.recon == node.id);
            if !crate::scene::is_visible(node, self.solo) || !(show_all || selected_here) {
                continue;
            }
            let rows = measurements::distance_rows(node.edited().measurements());
            if rows.is_empty() {
                continue;
            }
            let readings = measurements::readings(node);
            let unit = node.recon().metadata.world_space_unit.as_deref();
            let color = node.tint.rgb().unwrap_or(STORED_RGB);
            let color = Color32::from_rgb(color[0], color[1], color[2]);
            for (index, row) in rows.iter().enumerate() {
                let measurement = MeasurementRef::distance(node.id, index);
                let is_selected = self.selected_measurement == Some(measurement);
                if !is_selected && !show_all {
                    continue;
                }
                let Some(resolved) = readings.distance.get(index) else {
                    continue;
                };
                let endpoints = EndpointSide::BOTH
                    .into_iter()
                    .map(|side| DrawnEndpoint {
                        side,
                        image: row.image(side),
                        pixel: row.pixel(side),
                        place: place(node, Some(measurements::endpoint(resolved, side))),
                    })
                    .collect();
                let (role, color, width) = match (is_selected, show_all) {
                    (true, true) => (Role::Selected, color, EMPHASISED_LINE_WIDTH),
                    (true, false) => (Role::Selected, color, LINE_WIDTH),
                    (false, _) => (Role::Other, color.gamma_multiply(OTHER_OPACITY), LINE_WIDTH),
                };
                let drawn = DrawnMeasurement {
                    id: DrawnId::Stored(measurement),
                    node: node.id,
                    role,
                    title: measurement.title(),
                    hover: measurements::row_text(measurement, row, resolved, unit),
                    endpoints,
                    color,
                    width,
                    selected_endpoint: self
                        .selected_endpoint
                        .filter(|_| is_selected && !item_focused),
                    handles: false,
                };
                if is_selected {
                    selected.push(drawn);
                } else {
                    others.push(drawn);
                }
            }
        }
        let mut drawn = others;
        drawn.append(&mut selected);
        if let Some(item) = self.drawn_item() {
            drawn.push(item);
        }
        MeasurementLayer { drawn }
    }

    /// The focused measurement item as the bench layers draw it, or `None`
    /// with no measurement item focused.
    fn drawn_item(&self) -> Option<DrawnMeasurement> {
        let id = self.focused_item()?.node;
        let label = self.focused_measurement_label(id)?;
        let item = self.bench_measurement(id, label)?;
        let node = self.node(id)?;
        let distance = item.as_distance()?;
        let reading = node.item_reading(item);
        let endpoints = EndpointSide::BOTH
            .into_iter()
            .filter_map(|side| {
                let endpoint = distance.endpoints[side.index()]?;
                Some(DrawnEndpoint {
                    side,
                    image: endpoint.image,
                    pixel: endpoint.pixel,
                    place: place(node, reading.endpoints[side.index()].as_ref()),
                })
            })
            .collect();
        Some(DrawnMeasurement {
            id: DrawnId::Item {
                node: id,
                label: label.to_string(),
            },
            node: id,
            role: Role::Item,
            title: label.to_string(),
            hover: item_hover(node, label, item, &reading),
            endpoints,
            color: crate::bench::verdict_color(sfmtool_core::bench::Verdict::In),
            width: ITEM_LINE_WIDTH,
            selected_endpoint: self.selected_endpoint,
            handles: self.bench_edit_refusal(id).is_none(),
        })
    }
}

/// Where an endpoint resolution puts the endpoint in the shared world space.
fn place(
    node: &crate::scene::SceneNode,
    resolution: Option<
        &Result<
            sfmtool_core::reconstruction::measurement_endpoint::EndpointResolution,
            sfmtool_core::reconstruction::measurement_endpoint::EndpointRefusal,
        >,
    >,
) -> Place {
    match resolution {
        Some(Ok(resolution)) => match resolution.position().finite() {
            Some(p) => Place::Finite(node.transform().apply_to_point(&nalgebra::Point3::from(p))),
            None => Place::AtInfinity,
        },
        Some(Err(_)) | None => Place::Refused,
    }
}

/// The focused item's hover text: its label, then, where both endpoints and a
/// stated distance are given, the stated distance and the reading a Scene tree
/// row prints; otherwise where each endpoint is.
fn item_hover(
    node: &crate::scene::SceneNode,
    label: &str,
    item: &sfmtool_core::bench::EditableMeasurement,
    reading: &measurements::ItemReading,
) -> String {
    let unit = node.recon().metadata.world_space_unit.as_deref();
    if let (Some(row), Some(resolved)) = (measurements::item_row(item), &reading.resolved) {
        let result = measurements::distance_result(&row, resolved, unit);
        return format!(
            "{label}  {}  {}",
            row.stated_text(),
            measurements::reading_text(&row, resolved, result)
        );
    }
    let endpoints = item.as_distance().map_or([None, None], |d| d.endpoints);
    let parts: Vec<String> = EndpointSide::BOTH
        .into_iter()
        .map(|side| match endpoints[side.index()] {
            Some(e) => format!(
                "endpoint {} in {}",
                side.letter(),
                measurements::image_name(node.recon(), e.image)
            ),
            None => format!("endpoint {} not placed", side.letter()),
        })
        .collect();
    format!("{label}  {}", parts.join(", "))
}

/// An endpoint's mark on screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScreenMark {
    pub(crate) side: EndpointSide,
    pub(crate) pos: Pos2,
    /// Drawn in the warning colour: a refused endpoint in Image Detail.
    pub(crate) warning: bool,
}

/// One drawn measurement on screen: its marks and its line, in panel points.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScreenMeasurement {
    /// Its index in the layer's `drawn`.
    pub(crate) index: usize,
    pub(crate) marks: Vec<ScreenMark>,
    pub(crate) line: Option<[Pos2; 2]>,
}

impl ScreenMeasurement {
    /// Whether any part of its marks and line is inside `rect`.
    pub(crate) fn in_view(&self, rect: Rect) -> bool {
        self.marks.iter().any(|m| rect.contains(m.pos))
            || self
                .line
                .is_some_and(|[a, b]| segment_meets_rect(a, b, rect))
    }
}

/// What a press over the measurements can land on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Target {
    /// An endpoint's mark.
    Mark(DrawnId, EndpointSide),
    /// The line between the two marks.
    Line(DrawnId),
    /// The off-screen indicator.
    Indicator,
}

/// The off-screen indicator: an arrow on the view's border pointing toward the
/// selected measurement, with its title beside it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Indicator {
    /// The arrow's point.
    pub(crate) tip: Pos2,
    /// The unit direction it points in.
    pub(crate) direction: Vec2,
    pub(crate) label: String,
    /// Where the label is drawn.
    pub(crate) label_rect: Rect,
}

impl Indicator {
    /// The indicator in `rect` pointing along `toward`, a direction on screen
    /// from the view's centre, labelled `label` laid out at `label_size`.
    /// `None` for a direction of no length.
    pub(crate) fn new(rect: Rect, toward: Vec2, label: &str, label_size: Vec2) -> Option<Self> {
        let length = toward.length();
        if !length.is_finite() || length <= f32::EPSILON {
            return None;
        }
        let direction = toward / length;
        let half = (rect.size() / 2.0 - Vec2::splat(INDICATOR_INSET)).max(Vec2::ZERO);
        let reach = |half: f32, d: f32| {
            if d.abs() <= f32::EPSILON {
                f32::INFINITY
            } else {
                half / d.abs()
            }
        };
        let t = reach(half.x, direction.x).min(reach(half.y, direction.y));
        let tip = rect.center() + direction * t;
        // The label sits behind the arrow, toward the view's centre, far
        // enough along the direction that its own extent clears the arrow.
        let along = (label_size.x * direction.x.abs() + label_size.y * direction.y.abs()) / 2.0;
        let centre = tip - direction * (ARROW_LENGTH + 6.0 + along);
        Some(Self {
            tip,
            direction,
            label: label.to_string(),
            label_rect: Rect::from_center_size(centre, label_size),
        })
    }

    /// Whether `pos` is on the arrow or its label.
    pub(crate) fn contains(&self, pos: Pos2) -> bool {
        let back = self.tip - self.direction * ARROW_LENGTH;
        crate::bench::distance_to_segment(back, self.tip, pos) <= MARK_REACH
            || self.label_rect.expand(4.0).contains(pos)
    }

    /// Draw the arrow and its label in `color`.
    pub(crate) fn draw(&self, painter: &egui::Painter, color: Color32) {
        let back = self.tip - self.direction * ARROW_LENGTH;
        let side = self.direction.rot90() * (ARROW_LENGTH * 0.5);
        painter.add(egui::Shape::convex_polygon(
            vec![self.tip, back + side, back - side],
            color,
            Stroke::new(1.5, HALO),
        ));
        painter.rect_filled(self.label_rect.expand(3.0), 3.0, LABEL_GROUND);
        painter.text(
            self.label_rect.center(),
            egui::Align2::CENTER_CENTER,
            &self.label,
            label_font(),
            color,
        );
    }
}

/// The dark stroke under every mark and line, so they read over a bright
/// photograph or a bright cloud.
pub(crate) const HALO: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 170);

/// How much wider a mark's or a line's halo is than its stroke, in panel
/// points.
pub(crate) const HALO_EXTRA_WIDTH: f32 = 2.0;

/// Radius of the dot at the centre of an endpoint's mark, in panel points.
pub(crate) const MARK_DOT_RADIUS: f32 = 1.5;

/// The ground the hover text and the indicator's label are drawn on.
const LABEL_GROUND: Color32 = Color32::from_rgba_premultiplied(20, 20, 24, 215);

/// The font of the hover text and the indicator's label.
fn label_font() -> egui::FontId {
    egui::FontId::proportional(12.0)
}

/// The size `text` takes in the label font.
pub(crate) fn label_size(painter: &egui::Painter, text: &str) -> Vec2 {
    painter
        .layout_no_wrap(text.to_string(), label_font(), Color32::WHITE)
        .size()
}

/// The measurements of `layer` on screen, in its draw order.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Screen {
    pub(crate) measurements: Vec<ScreenMeasurement>,
    pub(crate) indicator: Option<Indicator>,
}

impl Screen {
    /// What `pos` lands on: the indicator first, then a mark, the nearest of
    /// the topmost measurement's, then a line, topmost first. A mark is tried
    /// before any line because it sits on a line's end and is the smaller
    /// target.
    pub(crate) fn hit(&self, layer: &MeasurementLayer, pos: Pos2) -> Option<Target> {
        if self.indicator.as_ref().is_some_and(|i| i.contains(pos)) {
            return Some(Target::Indicator);
        }
        for m in self.measurements.iter().rev() {
            let nearest = m
                .marks
                .iter()
                .map(|mark| (mark.side, mark.pos.distance(pos)))
                .filter(|(_, d)| *d <= MARK_REACH)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((side, _)) = nearest {
                return Some(Target::Mark(layer.drawn[m.index].id.clone(), side));
            }
        }
        for m in self.measurements.iter().rev() {
            if let Some([a, b]) = m.line {
                if crate::bench::distance_to_segment(a, b, pos) <= LINE_REACH {
                    return Some(Target::Line(layer.drawn[m.index].id.clone()));
                }
            }
        }
        None
    }

    /// The screen measurement drawn for `id`.
    pub(crate) fn of(&self, layer: &MeasurementLayer, id: &DrawnId) -> Option<&ScreenMeasurement> {
        self.measurements
            .iter()
            .find(|m| layer.drawn.get(m.index).is_some_and(|d| &d.id == id))
    }

    /// Move `side`'s mark of `id` to `pos`, and its line's end with it: the
    /// preview of a handle drag.
    pub(crate) fn move_mark(
        &mut self,
        layer: &MeasurementLayer,
        id: &DrawnId,
        side: EndpointSide,
        pos: Pos2,
    ) {
        for m in &mut self.measurements {
            if layer.drawn.get(m.index).is_none_or(|d| &d.id != id) {
                continue;
            }
            for mark in &mut m.marks {
                if mark.side == side {
                    mark.pos = pos;
                }
            }
            if let Some(line) = &mut m.line {
                line[side.index()] = pos;
            }
        }
    }

    /// Draw every measurement, then the indicator. `warning` is the colour a
    /// refused endpoint's mark takes.
    pub(crate) fn draw(&self, painter: &egui::Painter, layer: &MeasurementLayer, warning: Color32) {
        for m in &self.measurements {
            let Some(drawn) = layer.drawn.get(m.index) else {
                continue;
            };
            let halo = drawn.width + HALO_EXTRA_WIDTH;
            if let Some([a, b]) = m.line {
                painter.line_segment([a, b], Stroke::new(halo, HALO));
                painter.line_segment([a, b], Stroke::new(drawn.width, drawn.color));
            }
            for mark in &m.marks {
                let color = if mark.warning { warning } else { drawn.color };
                let radius = mark_radius(drawn, mark.side);
                painter.circle_stroke(mark.pos, radius, Stroke::new(halo, HALO));
                painter.circle_stroke(mark.pos, radius, Stroke::new(drawn.width, color));
                painter.circle_filled(mark.pos, MARK_DOT_RADIUS, color);
            }
        }
        self.draw_indicator(painter, layer);
    }

    /// Draw the indicator alone, in the selected measurement's colour: what
    /// the 3D view puts over the scene, whose pass draws the marks and lines.
    pub(crate) fn draw_indicator(&self, painter: &egui::Painter, layer: &MeasurementLayer) {
        if let Some(indicator) = &self.indicator {
            let color = layer.selected().map_or(Color32::WHITE, |d| d.color);
            indicator.draw(painter, color);
        }
    }
}

/// The radius `drawn`'s mark on `side` is drawn at.
pub(crate) fn mark_radius(drawn: &DrawnMeasurement, side: EndpointSide) -> f32 {
    if drawn.selected_endpoint == Some(side) {
        MARK_RADIUS * SELECTED_MARK_SCALE
    } else {
        MARK_RADIUS
    }
}

/// Draw `text` beside the pointer at `pos`, inside `bounds` and off the
/// pointer: the hover text of a mark or a line. Its ground's corner is
/// [`HOVER_TEXT_OFFSET`] right of and below the pointer where that fits, and
/// [`place_beside_pointer`](crate::pointer_label::place_beside_pointer) picks
/// another side where it does not.
pub(crate) fn draw_hover_text(painter: &egui::Painter, pos: Pos2, text: &str, bounds: Rect) {
    /// The space between the text and the edge of its ground.
    const MARGIN: f32 = 4.0;
    let galley = painter.layout_no_wrap(text.to_string(), label_font(), Color32::WHITE);
    let ground = crate::pointer_label::place_beside_pointer(
        pos,
        galley.size() + Vec2::splat(2.0 * MARGIN),
        bounds,
        HOVER_TEXT_OFFSET,
    );
    painter.rect_filled(ground, 3.0, LABEL_GROUND);
    painter.galley(ground.min + Vec2::splat(MARGIN), galley, Color32::WHITE);
}

/// From the pointer to the top-left corner of a measurement's hover text's
/// ground, where it fits: 10 points right and 10 below, so the text itself
/// starts 14 points right of and below the pointer.
const HOVER_TEXT_OFFSET: Vec2 = Vec2::new(10.0, 10.0);

/// Whether the segment `a`–`b` has any part inside `rect`.
pub(crate) fn segment_meets_rect(a: Pos2, b: Pos2, rect: Rect) -> bool {
    if rect.contains(a) || rect.contains(b) {
        return true;
    }
    // Liang–Barsky: clip the segment's parameter range to each slab.
    let d = b - a;
    let mut t0 = 0.0_f32;
    let mut t1 = 1.0_f32;
    for (p, q) in [
        (-d.x, a.x - rect.left()),
        (d.x, rect.right() - a.x),
        (-d.y, a.y - rect.top()),
        (d.y, rect.bottom() - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return false;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    t0 <= t1
}

/// A press the measurements took: what it landed on and where the pointer
/// has been since.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Press {
    pub(crate) target: Target,
    pub(crate) from: Pos2,
    pub(crate) to: Pos2,
    /// Whether the pointer has travelled more than [`CLICK_TRAVEL`].
    pub(crate) moved: bool,
    /// Escape abandoned it.
    pub(crate) cancelled: bool,
}

/// What [`follow`] made of one frame's pointer.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Followed {
    /// A press is held, or came up this frame: the pointer is the
    /// measurements' and nothing under them may act on it (G7).
    pub(crate) owns: bool,
    /// The press that came up this frame, and whether egui calls the release
    /// the second click of a double-click.
    pub(crate) released: Option<(Press, bool)>,
    /// What the pointer is over, or holds.
    pub(crate) hovered: Option<Target>,
}

/// Follow a press over the measurements for one frame: take it at an
/// unmodified primary press that `hit` lands, follow the pointer, let Escape
/// abandon it, and end it when the button comes up.
///
/// **The press decides the target, not the drag**, as for the bench layers'
/// handles: the view's own pan or orbit would otherwise start on the motion
/// before egui calls it a drag, and the target under the press would move.
pub(crate) fn follow(
    press: &mut Option<Press>,
    ui: &egui::Ui,
    response: &egui::Response,
    hit: impl Fn(Pos2) -> Option<Target>,
) -> Followed {
    let (pressed, down, pointer, hover, modifiers, escape) = ui.input(|i| {
        (
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.interact_pos().or(i.pointer.hover_pos()),
            i.pointer.hover_pos(),
            i.modifiers,
            i.key_pressed(egui::Key::Escape),
        )
    });
    let plain = !(modifiers.alt || modifiers.ctrl || modifiers.command || modifiers.shift)
        && !crate::platform::other_mouse_button_down();
    if press.is_none() && pressed && plain {
        if let Some(pos) = pointer.filter(|_| response.contains_pointer()) {
            if let Some(target) = hit(pos) {
                *press = Some(Press {
                    target,
                    from: pos,
                    to: pos,
                    moved: false,
                    cancelled: false,
                });
            }
        }
    }
    if let Some(held) = press.as_mut() {
        if let Some(pos) = response.interact_pointer_pos().or(pointer) {
            held.to = pos;
            held.moved |= pos.distance(held.from) > CLICK_TRAVEL;
        }
        held.cancelled |= escape;
    }
    let mut released = None;
    if !down {
        if let Some(ended) = press.take() {
            released = Some((ended, response.double_clicked()));
        }
    }
    let hovered = press
        .as_ref()
        .map(|p| p.target.clone())
        .or_else(|| hover.filter(|_| response.contains_pointer()).and_then(&hit));
    Followed {
        owns: press.is_some() || released.is_some(),
        released,
        hovered,
    }
}

/// What a gesture on a drawn measurement asks of the app, applied by the dock.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MeasurementGesture {
    /// A click on a line: select the stored measurement. The focused item is
    /// focused already, so a click on its line asks for nothing more.
    Select(DrawnId),
    /// A click on a mark: select the measurement and that endpoint, as a
    /// click on its endpoint block does.
    SelectEndpoint(DrawnId, EndpointSide),
    /// A double-click on a mark: what a double-click on its endpoint block
    /// does.
    LookThrough(DrawnId, EndpointSide),
    /// A released drag of the focused item's handle in Image Detail: the
    /// *Set an endpoint's image and pixel* step.
    SetEndpoint(crate::bench::measurement::EndpointSet),
}

/// What a click on `target` asks for, where it is a mark or a line.
pub(crate) fn click_gesture(target: &Target, double: bool) -> Option<MeasurementGesture> {
    match target {
        Target::Mark(id, side) if double => {
            Some(MeasurementGesture::LookThrough(id.clone(), *side))
        }
        Target::Mark(id, side) => Some(MeasurementGesture::SelectEndpoint(id.clone(), *side)),
        Target::Line(id) => Some(MeasurementGesture::Select(id.clone())),
        Target::Indicator => None,
    }
}

/// The cursor the pointer takes over `target`, `handle` where it is a handle
/// and `held` while a press holds it.
pub(crate) fn cursor(handle: bool, held: bool) -> egui::CursorIcon {
    match (handle, held) {
        (true, true) => egui::CursorIcon::Grabbing,
        (true, false) => egui::CursorIcon::Grab,
        (false, _) => egui::CursorIcon::PointingHand,
    }
}

/// Whether `target` is a handle: a mark of a drawn measurement whose marks
/// are handles.
pub(crate) fn is_handle(layer: &MeasurementLayer, target: &Target) -> bool {
    match target {
        Target::Mark(id, _) => layer.find(id).is_some_and(|d| d.handles),
        _ => false,
    }
}

/// The hover text for `target`.
pub(crate) fn hover_text<'a>(layer: &'a MeasurementLayer, target: &Target) -> Option<&'a str> {
    match target {
        Target::Mark(id, _) | Target::Line(id) => layer.find(id).map(|d| d.hover.as_str()),
        Target::Indicator => None,
    }
}
