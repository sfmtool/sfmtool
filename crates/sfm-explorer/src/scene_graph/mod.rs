// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph panel — the tree of loaded reconstructions.
//!
//! A fifth dock tab titled **"Scene"**, and the control surface for everything
//! the scene graph adds: which reconstructions are loaded, which of their layers
//! are drawn, which of them capture pointer picks, which one the sequence-shaped
//! panels follow, and how a node leaves. See `specs/gui/gui-scene-graph.md`
//! ("Scene Graph Panel").
//!
//! ## Shape of the tree
//!
//! ```text
//! ▾ 👁 🖱 run_a                    1.2M pts · 243 cams
//!   ▾ 👁 Cameras (243)
//!       IMG_0001.jpg              ← virtualized, `ScrollArea::show_rows`
//!   ▾ 👁 Points (1,204,551 · 12 at ∞) ∞
//!       selected: pt3d_a1b2c3_88231   ← selection / hover rows only
//!     👁 Patches                   ← only when the node carries patch data
//! ```
//!
//! Expansion state lives in [`egui::collapsing_header::CollapsingState`] under
//! **explicit ids** ([`row_id`]) rather than `CollapsingHeader`'s
//! id-salt-plus-parent scheme, following the viewport HUD: that keeps a row's
//! open/closed state addressable from outside the panel, which is how the tests
//! tell an expanded subtree from a collapsed one.
//!
//! ## What the panel does not do
//!
//! The Points group expands to the **selection and hover rows only**, never a
//! full per-point listing: millions of rows are not navigable, and past about
//! 16.7M row-pixels egui's `f32` scroll coordinates lose integer precision, so
//! such a list would misbehave mechanically as well as ergonomically.

use eframe::egui;

use crate::scene::{point_id, ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;

/// Height of one tree row. Fixed so the camera list can be virtualized, and so
/// the eye/cursor toggles line up down the tree regardless of label height.
const ROW_HEIGHT: f32 = 18.0;

/// Height of the virtualized camera list inside an expanded Cameras group.
/// Bounded rather than unbounded so a 50K-camera node cannot push every node
/// below it off the panel.
const CAMERA_LIST_HEIGHT: f32 = 220.0;

/// Width reserved for each toggle glyph, so labels align across rows whether or
/// not a row has an interaction cursor.
const TOGGLE_WIDTH: f32 = 18.0;

/// Master-eye glyph. Dimmed rather than swapped when off — egui's bundled fonts
/// have no closed-eye counterpart, and `scene_graph/tests.rs` guards that every
/// glyph here is actually present.
const EYE_GLYPH: &str = "👁";

/// Interaction-cursor glyph (the Blender-outliner eye/cursor pairing): whether
/// pointer picks in the 3D viewport reach this node.
const CURSOR_GLYPH: &str = "🖱";

/// The ∞ mini-toggle on the Points group row.
const INFINITY_GLYPH: &str = "∞";

/// Width of the accent bar marking the selected reconstruction.
///
/// Painted rather than written: egui's bundled *proportional* family has no
/// vertical-bar glyph (U+2502 and U+258D are both absent — Hack has them, but
/// the tree is not monospaced), and a filled rect is the better bar anyway.
const SELECTED_BAR_WIDTH: f32 = 3.0;

/// Stable id of one tree row's expansion state, independent of the `Ui` that
/// hosts it.
///
/// Keyed by the node's [`ReconId`], which is never reused — so a closed node
/// cannot hand its expansion state to a later one, and a reload (which mints a
/// fresh id) comes back collapsed.
pub(crate) fn row_id(recon: ReconId, key: &str) -> egui::Id {
    egui::Id::new(("scene_graph_row", recon, key))
}

/// What the Scene panel reports back to `dock.rs`.
///
/// The spec's `align_node` / `reset_transform` fields are absent: they belong to
/// node transforms (phase 4), and a response field with no producer is a
/// promise the panel cannot keep.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SceneGraphResponse {
    /// A camera row was clicked.
    pub select_image: Option<ImageRef>,
    /// The Points group's `selected:` row was clicked.
    pub select_point: Option<PointRef>,
    /// A camera row was double-clicked — enter/switch camera view.
    pub request_camera_view: Option<ImageRef>,
    /// Image under the pointer, for cross-panel hover.
    pub hovered_image: Option<ImageRef>,
    /// Point under the pointer, for cross-panel hover.
    pub hovered_point: Option<PointRef>,
    /// Whether the pointer is inside this panel — which makes it the owner of
    /// both hover fields for the frame.
    pub has_pointer: bool,
    /// A reconstruction row was clicked, or `Select` chosen from its menu.
    pub select_recon: Option<ReconId>,
    /// A reconstruction row was double-clicked, or `Zoom to Fit` chosen.
    pub zoom_to_node: Option<ReconId>,
    /// `Close` chosen from a reconstruction's context menu.
    pub close_node: Option<ReconId>,
    /// `Reload from Disk` chosen from a reconstruction's context menu.
    pub reload_node: Option<ReconId>,
}

/// Scene Graph panel state.
///
/// Only what the tree has to remember between frames; expansion lives in egui's
/// own memory under [`row_id`], and everything else is read from `AppState`.
#[derive(Default)]
pub struct SceneGraphPanel {
    /// The image selection as of the previous frame. A camera list scrolls its
    /// selected row into view only when this *changes*, so the panel does not
    /// fight the user's own scrolling.
    prev_selected_image: Option<ImageRef>,
    /// Screen rect of every row and toggle drawn on the last frame, keyed by
    /// [`row_id`]. A collapsible, virtualized tree has no geometry an outside
    /// caller can predict, so this is how anything that needs to point *at* a
    /// row finds it — the panel tests today, keyboard navigation later.
    hits: std::collections::HashMap<egui::Id, egui::Rect>,
}

impl SceneGraphPanel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Where a row or toggle was drawn on the last frame, if it was drawn.
    ///
    /// Test-only for now — the recording itself is unconditional so that what
    /// the tests aim at is the very layout the app produces, with no
    /// test-specific code path to drift out of step.
    #[cfg(test)]
    pub fn hit_rect(&self, id: egui::Id) -> Option<egui::Rect> {
        self.hits.get(&id).copied()
    }

    /// Draw the tree and report what the user did with it.
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) -> SceneGraphResponse {
        let panel_rect = ui.available_rect_before_wrap();
        let mut out = TreeOutput {
            response: SceneGraphResponse::default(),
            hits: &mut self.hits,
        };
        out.hits.clear();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            out.response.has_pointer = panel_rect.contains(pos);
        }

        if state.scene.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No reconstruction loaded");
            });
            self.prev_selected_image = None;
            return out.response;
        }

        // Read the selection out before the mutable walk over `scene` below:
        // every field of `AppState` is one borrow, and the node eyes need `&mut`.
        let selected_recon = state.selected_recon;
        let selected_image = state.selected_image;
        let selected_point = state.selected_point;
        let hovered_image = state.hovered_image;
        let hovered_point = state.hovered_point;
        // Auto-scroll fires on a selection change from *another* panel; a click
        // inside this one lands in the response and is applied after the frame,
        // so by construction it cannot also trigger a scroll.
        let selection_moved = self.prev_selected_image != selected_image;
        self.prev_selected_image = selected_image;

        egui::ScrollArea::vertical()
            .id_salt("scene_graph_tree")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                for node in state.scene.iter_mut() {
                    let ctx = NodeContext {
                        selected: selected_recon == Some(node.id),
                        selected_image,
                        selected_point,
                        hovered_image,
                        hovered_point,
                        selection_moved,
                    };
                    show_node(ui, node, &ctx, &mut out);
                }
            });

        out.response
    }
}

/// What one frame of the tree produced.
struct TreeOutput<'a> {
    response: SceneGraphResponse,
    hits: &'a mut std::collections::HashMap<egui::Id, egui::Rect>,
}

impl TreeOutput<'_> {
    /// Record where a row or toggle landed, and hand the response back through.
    fn hit(&mut self, id: egui::Id, response: egui::Response) -> egui::Response {
        self.hits.insert(id, response.rect);
        response
    }
}

/// The selection/hover context one node's subtree is drawn against.
struct NodeContext {
    selected: bool,
    selected_image: Option<ImageRef>,
    selected_point: Option<PointRef>,
    hovered_image: Option<ImageRef>,
    hovered_point: Option<PointRef>,
    selection_moved: bool,
}

/// One reconstruction row and, when expanded, its group rows.
fn show_node(ui: &mut egui::Ui, node: &mut SceneNode, ctx: &NodeContext, out: &mut TreeOutput) {
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        row_id(node.id, "node"),
        // Open by default: with one file loaded that is the whole panel, and
        // with a handful it is still the view that answers "what is in here".
        true,
    );
    let header = state.show_header(ui, |ui| {
        show_node_header(ui, node, ctx, out);
    });
    header.body(|ui| {
        show_cameras_group(ui, node, ctx, out);
        show_points_group(ui, node, ctx, out);
        if node.has_patch_data() {
            ui.horizontal(|ui| {
                ui.set_height(ROW_HEIGHT);
                let eye = eye_toggle(ui, &mut node.show_patches, "Show this node's patches");
                out.hit(row_id(node.id, "patches_eye"), eye);
                ui.label("Patches");
            });
        }
    });
}

/// `[👁] [🖱] label    1.2M pts · 243 cams` — the reconstruction row's content,
/// drawn to the right of the collapsing triangle.
fn show_node_header(
    ui: &mut egui::Ui,
    node: &mut SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    ui.set_height(ROW_HEIGHT);
    let eye = eye_toggle(ui, &mut node.visible, "Show this reconstruction");
    out.hit(row_id(node.id, "node_eye"), eye);
    let cursor = glyph_toggle(
        ui,
        &mut node.interactive,
        CURSOR_GLYPH,
        "Let hover and clicks in the 3D viewport reach this reconstruction",
    );
    out.hit(row_id(node.id, "node_cursor"), cursor);

    if ctx.selected {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(SELECTED_BAR_WIDTH, ROW_HEIGHT),
            egui::Sense::hover(),
        );
        ui.painter()
            .rect_filled(rect, 1.0, ui.visuals().selection.bg_fill);
    }

    let mut label = egui::RichText::new(&node.label);
    if ctx.selected {
        label = label.strong();
    }
    let row = ui.add(egui::Label::new(label).sense(egui::Sense::click()));
    let row = out.hit(row_id(node.id, "node_label"), row);
    if row.clicked() {
        out.response.select_recon = Some(node.id);
    }
    if row.double_clicked() {
        out.response.zoom_to_node = Some(node.id);
    }
    row.context_menu(|ui| node_context_menu(ui, node, &mut out.response));

    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.label(
            egui::RichText::new(format!(
                "{} pts · {} cams",
                compact_count(node.recon.points.len()),
                compact_count(node.recon.images.len()),
            ))
            .weak()
            .small(),
        );
    });
}

/// The reconstruction row's context menu.
///
/// `Align to ▸`, `Reset Transform` and `Tint ▸` are deliberately absent: they
/// operate on the node transform and tint, which arrive in phases 4–5. A menu
/// entry that silently does nothing is worse than no entry.
fn node_context_menu(ui: &mut egui::Ui, node: &SceneNode, response: &mut SceneGraphResponse) {
    if ui.button("Select").clicked() {
        response.select_recon = Some(node.id);
        ui.close();
    }
    if ui.button("Zoom to Fit").clicked() {
        response.zoom_to_node = Some(node.id);
        ui.close();
    }
    ui.separator();
    // Demo data came from no file, so there is nothing to re-read.
    if ui
        .add_enabled(node.path.is_some(), egui::Button::new("Reload from Disk"))
        .on_disabled_hover_text("This reconstruction was generated, not loaded from a file")
        .clicked()
    {
        response.reload_node = Some(node.id);
        ui.close();
    }
    if ui.button("Close").clicked() {
        response.close_node = Some(node.id);
        ui.close();
    }
}

/// `[▸] [👁] Cameras (243)`, expanding to a virtualized per-image list.
fn show_cameras_group(
    ui: &mut egui::Ui,
    node: &mut SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        row_id(node.id, "cameras"),
        // Collapsed: the image list is the Image Browser's job, and an
        // expanded-by-default list would bury every node below this one.
        false,
    );
    let id = node.id;
    let header = state.show_header(ui, |ui| {
        ui.set_height(ROW_HEIGHT);
        let eye = eye_toggle(ui, &mut node.show_cameras, "Show this node's cameras");
        out.hit(row_id(id, "cameras_eye"), eye);
        ui.label(format!("Cameras ({})", node.recon.images.len()));
    });
    header.body(|ui| show_camera_rows(ui, node, ctx, out));
}

/// The per-image rows, laid out only for the visible slice of the list.
fn show_camera_rows(ui: &mut egui::Ui, node: &SceneNode, ctx: &NodeContext, out: &mut TreeOutput) {
    let count = node.recon.images.len();
    if count == 0 {
        ui.weak("No images");
        return;
    }

    // Scroll the selected row into view only when the selection moved and it
    // belongs to this node. Driven by an explicit offset rather than
    // `scroll_to_me`, because a virtualized row outside the rendered slice is
    // never built and so could never ask to be scrolled to.
    let scroll_target = (ctx.selection_moved)
        .then(|| ctx.selected_image?.index_in(node.id))
        .flatten()
        .filter(|&row| row < count);

    let row_height = ROW_HEIGHT + ui.spacing().item_spacing.y;
    let mut area = egui::ScrollArea::vertical()
        .id_salt(row_id(node.id, "camera_list"))
        .max_height(CAMERA_LIST_HEIGHT)
        .auto_shrink([false, false]);
    if let Some(row) = scroll_target {
        let centered = row as f32 * row_height - CAMERA_LIST_HEIGHT / 2.0;
        area = area.vertical_scroll_offset(centered.max(0.0));
    }
    area.show_rows(ui, ROW_HEIGHT, count, |ui, range| {
        for index in range {
            let image = ImageRef::new(node.id, index);
            let name = node
                .recon
                .images
                .get(index)
                .map(|i| i.name.as_str())
                .unwrap_or("?");
            let selected = ctx.selected_image == Some(image);
            // Hover highlight comes from the shared hover state, not from
            // egui's own, so a hover raised in the 3D viewport or the browser
            // lights this row up too.
            let hovered = ctx.hovered_image == Some(image);
            let mut text = egui::RichText::new(name);
            if hovered && !selected {
                text = text.color(ui.visuals().strong_text_color());
            }
            let row = ui.add(egui::Button::selectable(selected, text));
            let row = out.hit(row_id(node.id, &format!("camera_{index}")), row);
            if row.clicked() {
                out.response.select_image = Some(image);
            }
            if row.double_clicked() {
                out.response.request_camera_view = Some(image);
            }
            if row.hovered() {
                out.response.hovered_image = Some(image);
            }
        }
    });
}

/// `[▸] [👁] Points (1,204,551 · 12 at ∞) [∞]`, expanding to the selection and
/// hover rows.
fn show_points_group(
    ui: &mut egui::Ui,
    node: &mut SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    let at_infinity = node.recon.metadata.infinity_point_count as usize;
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        row_id(node.id, "points"),
        false,
    );
    let id = node.id;
    let header = state.show_header(ui, |ui| {
        ui.set_height(ROW_HEIGHT);
        let eye = eye_toggle(ui, &mut node.show_points, "Show this node's 3D points");
        out.hit(row_id(id, "points_eye"), eye);
        let count = with_thousands(node.recon.points.len());
        let label = if at_infinity > 0 {
            format!("Points ({count} · {} at ∞)", with_thousands(at_infinity))
        } else {
            format!("Points ({count})")
        };
        ui.label(label);
        if at_infinity > 0 {
            let infinity = glyph_toggle(
                ui,
                &mut node.show_points_at_infinity,
                INFINITY_GLYPH,
                "Draw this node's w = 0 points — directions with no parallax",
            );
            out.hit(row_id(id, "points_infinity"), infinity);
        }
    });
    header.body(|ui| {
        let mut any = false;
        if let Some(point) = ctx.selected_point.filter(|p| p.recon == id) {
            any = true;
            let row = ui.add(egui::Button::selectable(
                true,
                format!("selected: {}", point_id(&node.recon, point.index())),
            ));
            if out.hit(row_id(id, "point_selected"), row).clicked() {
                out.response.select_point = Some(point);
            }
        }
        if let Some(point) = ctx
            .hovered_point
            .filter(|p| p.recon == id && Some(*p) != ctx.selected_point)
        {
            any = true;
            let row = ui.add(egui::Button::selectable(
                false,
                egui::RichText::new(format!("hovered: {}", point_id(&node.recon, point.index())))
                    .weak(),
            ));
            if out.hit(row_id(id, "point_hovered"), row).clicked() {
                out.response.select_point = Some(point);
            }
        }
        if !any {
            // Deliberately not a listing: see the module docs.
            ui.weak("No point selected");
        }
    });
}

/// The master/group eye. Dimmed when off; the tooltip says what it governs.
fn eye_toggle(ui: &mut egui::Ui, on: &mut bool, tooltip: &str) -> egui::Response {
    glyph_toggle(ui, on, EYE_GLYPH, tooltip)
}

/// A one-glyph toggle button: full-strength when on, weak when off.
fn glyph_toggle(ui: &mut egui::Ui, on: &mut bool, glyph: &str, tooltip: &str) -> egui::Response {
    let color = if *on {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    let button = egui::Button::new(egui::RichText::new(glyph).color(color))
        .frame(false)
        .min_size(egui::vec2(TOGGLE_WIDTH, ROW_HEIGHT));
    let response = ui.add(button).on_hover_text(tooltip);
    if response.clicked() {
        *on = !*on;
    }
    response
}

/// `1234567` → `"1.2M"`, `12345` → `"12.3K"`, `999` → `"999"`.
///
/// The reconstruction row has to stay one line at an 18%-wide panel, and an
/// exact count is available one row down on the group rows.
fn compact_count(n: usize) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}K", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// `1204551` → `"1,204,551"`.
fn with_thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests;
