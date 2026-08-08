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

use crate::align::{AlignOptions, AlignSource};
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
    /// `Align to ▸ <node>` chosen: `(source, target, options)`. The source is
    /// the row the menu was opened on; the target is the node picked from the
    /// submenu, and is never itself modified.
    pub align_node: Option<(ReconId, ReconId, AlignOptions)>,
    /// `Reset Transform` chosen — return this node to its own frame.
    pub reset_transform: Option<ReconId>,
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
    /// The `Align to ▸` popup's two settings. Panel state rather than app
    /// state: they configure the *next* fit and nothing outside the menu reads
    /// them, but they have to survive the frame the popup is open.
    align_options: AlignOptions,
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
        // What every *other* node offers as an alignment target, taken before
        // the mutable walk below: the tree holds one node at a time, and the
        // `Align to ▸` submenu on that node has to name all the rest.
        let targets: Vec<AlignTarget> = state
            .scene
            .iter()
            .map(|n| AlignTarget {
                id: n.id,
                label: n.label.clone(),
                feature_indexed: n.recon.feature_indexes().is_some(),
            })
            .collect();
        let mut out = TreeOutput {
            response: SceneGraphResponse::default(),
            hits: &mut self.hits,
            align_options: &mut self.align_options,
            targets: &targets,
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

/// One loaded node as an `Align to ▸` menu entry.
struct AlignTarget {
    id: ReconId,
    label: String,
    /// Whether it carries `sift_files` observations — the feature indexes the
    /// point mode matches on.
    feature_indexed: bool,
}

/// What one frame of the tree produced.
struct TreeOutput<'a> {
    response: SceneGraphResponse,
    hits: &'a mut std::collections::HashMap<egui::Id, egui::Rect>,
    align_options: &'a mut AlignOptions,
    /// Every loaded node, including the one being drawn (filtered out where the
    /// menu is built).
    targets: &'a [AlignTarget],
}

impl TreeOutput<'_> {
    /// Record where a row or toggle landed, and hand the response back through.
    fn hit(&mut self, id: egui::Id, response: egui::Response) -> egui::Response {
        self.mark(id, response.rect);
        response
    }

    /// Record where something the panel *painted* landed. The selection accent
    /// bar is no widget, so it has no response to record — but whether it was
    /// drawn is exactly what a test asking "is this row marked selected?" wants.
    fn mark(&mut self, id: egui::Id, rect: egui::Rect) {
        self.hits.insert(id, rect);
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
///
/// Everything past the two toggles is a single click target spanning the row:
/// select, zoom-to-fit and the context menu all hang off it, so none of them
/// depends on the user finding the name's own few pixels.
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

    // Everything from here to the right edge is one target, claimed *before*
    // its contents are drawn: a tree row should answer a click anywhere along
    // it — on the name, in the gap, on the counts — the way any outliner does,
    // and hunting for the few pixels the name happens to occupy is how the
    // context menu came to look like it did not exist.
    //
    // Under an explicit id, not the auto-generated one a bare widget would
    // carry. egui keys a popup's open state on `response.id`, and an auto id is
    // a count of what was allocated before it in this `Ui` — so anything that
    // changed the row's contents would move the menu's identity out from under
    // it. The contents below are all hover-only, so none of them competes with
    // this rect for the click; the toggles above sit outside it.
    let available = ui.available_rect_before_wrap();
    let row_rect =
        egui::Rect::from_min_size(available.min, egui::vec2(available.width(), ROW_HEIGHT));
    let row = ui.interact(
        row_rect,
        row_id(node.id, "node_label"),
        egui::Sense::click(),
    );
    let row = out.hit(row_id(node.id, "node_label"), row);
    if row.clicked() {
        out.response.select_recon = Some(node.id);
    }
    if row.double_clicked() {
        out.response.zoom_to_node = Some(node.id);
    }
    // Built from `Popup` rather than `Response::context_menu` for the sake of
    // one setting: egui's default menu closes on *any* click inside it, which
    // would tear the whole menu down the moment the user set one of the two
    // `Align to` radio buttons. `CloseOnClickOutside` leaves closing to the
    // explicit `ui.close()` on each item that actually does something.
    egui::Popup::context_menu(&row)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| node_context_menu(ui, node, out));

    // Reserved whether or not it is painted, so the name does not jump sideways
    // as the selection moves and so every id downstream of it holds still.
    let (bar, _) = ui.allocate_exact_size(
        egui::vec2(SELECTED_BAR_WIDTH, ROW_HEIGHT),
        egui::Sense::hover(),
    );
    if ctx.selected {
        ui.painter()
            .rect_filled(bar, 1.0, ui.visuals().selection.bg_fill);
        out.mark(row_id(node.id, "node_selected_bar"), bar);
    }

    let mut label = egui::RichText::new(&node.label);
    if ctx.selected {
        label = label.strong();
    }
    // `selectable(false)` on both texts, and it is load-bearing rather than
    // cosmetic: egui's default `selectable_labels` gives a bare label
    // `Sense::click_and_drag()` so its text can be dragged out, and being drawn
    // *after* the row it would win every pointer hit that landed on a glyph.
    // That is how the name — the one part of the row a user actually aims at —
    // came to be the one part that answered nothing.
    ui.add(egui::Label::new(label).selectable(false));

    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.add(
            egui::Label::new(
                egui::RichText::new(format!(
                    "{} pts · {} cams",
                    compact_count(node.recon.points.len()),
                    compact_count(node.recon.images.len()),
                ))
                .weak()
                .small(),
            )
            .selectable(false),
        );
    });
}

/// The reconstruction row's context menu.
///
/// `Tint ▸` is deliberately absent: it operates on the node tint, which arrives
/// in phase 5. A menu entry that silently does nothing is worse than no entry.
fn node_context_menu(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    if ui.button("Select").clicked() {
        out.response.select_recon = Some(node.id);
        ui.close();
    }
    if ui.button("Zoom to Fit").clicked() {
        out.response.zoom_to_node = Some(node.id);
        ui.close();
    }
    ui.separator();
    show_align_menu(ui, node, out);
    let reset = ui
        .add_enabled(node.has_transform(), egui::Button::new("Reset Transform"))
        .on_disabled_hover_text("This reconstruction is already in its own frame");
    if out.hit(row_id(node.id, "reset_transform"), reset).clicked() {
        out.response.reset_transform = Some(node.id);
        ui.close();
    }
    ui.separator();
    // Demo data came from no file, so there is nothing to re-read.
    if ui
        .add_enabled(node.path.is_some(), egui::Button::new("Reload from Disk"))
        .on_disabled_hover_text("This reconstruction was generated, not loaded from a file")
        .clicked()
    {
        out.response.reload_node = Some(node.id);
        ui.close();
    }
    if ui.button("Close").clicked() {
        out.response.close_node = Some(node.id);
        ui.close();
    }
}

/// Why the point mode is unavailable, shown on hover over the greyed option.
const POINTS_DISABLED_HINT: &str =
    "Point correspondences are matched by feature index, so both reconstructions \
     need `sift_files` observations. One of these carries embedded patches instead.";

/// `Align to ▸`: the fit's two options, then one entry per other loaded node.
///
/// Options above targets rather than a popup per target: there are two of them,
/// they persist between opens, and a submenu three levels deep to set a radio
/// button would cost more than it explains.
fn show_align_menu(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    let others: Vec<&AlignTarget> = out.targets.iter().filter(|t| t.id != node.id).collect();
    if others.is_empty() {
        // Kept visible but dead: the operation exists, it just has nothing to
        // align to yet, and hiding it would make it look unimplemented.
        ui.add_enabled(false, egui::Button::new("Align to"))
            .on_disabled_hover_text("Load a second reconstruction to align this one to it");
        return;
    }

    let source_indexed = node.recon.feature_indexes().is_some();
    let any_target_indexed = others.iter().any(|t| t.feature_indexed);
    let points_available = source_indexed && any_target_indexed;
    let id = node.id;

    let menu = ui.menu_button("Align to", |ui| {
        ui.label(egui::RichText::new("Correspondences").weak().small());
        let cameras = ui.radio_value(
            &mut out.align_options.source,
            AlignSource::Cameras,
            "Cameras",
        );
        out.hit(row_id(id, "align_cameras"), cameras);
        let points = ui
            .add_enabled_ui(points_available, |ui| {
                ui.radio_value(&mut out.align_options.source, AlignSource::Points, "Points")
            })
            .inner
            .on_disabled_hover_text(POINTS_DISABLED_HINT);
        out.hit(row_id(id, "align_points"), points);

        ui.separator();
        ui.label(egui::RichText::new("Fit").weak().small());
        let similarity = ui.radio_value(&mut out.align_options.estimate_scale, true, "Similarity");
        out.hit(row_id(id, "align_similarity"), similarity);
        let rigid = ui.radio_value(&mut out.align_options.estimate_scale, false, "Rigid");
        out.hit(row_id(id, "align_rigid"), rigid);

        ui.separator();
        let by_points = out.align_options.source == AlignSource::Points;
        for target in &others {
            // A target can be individually unusable even when the mode is
            // selectable: one other node may carry feature indexes and another
            // not.
            let usable = !by_points || (source_indexed && target.feature_indexed);
            let button = ui
                .add_enabled(usable, egui::Button::new(&target.label))
                .on_disabled_hover_text(POINTS_DISABLED_HINT);
            let button = out.hit(row_id(id, &format!("align_to_{}", target.label)), button);
            if button.clicked() {
                out.response.align_node = Some((id, target.id, *out.align_options));
                // Both levels: `ui.close()` here would dismiss this submenu and
                // leave the reconstruction row's menu standing open behind it.
                egui::Popup::close_all(ui.ctx());
            }
        }
    });
    out.hit(row_id(id, "align_menu"), menu.response);
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
            // Salted by image index, so a row keeps one id however far the
            // virtualized list has scrolled. Left to egui's auto ids a row's
            // identity would be its position in the *rendered slice*, which
            // moves under it on every scroll.
            let row = ui
                .push_id(index, |ui| ui.add(egui::Button::selectable(selected, text)))
                .inner;
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
        // Both rows are conditional, so both are salted: without that the hover
        // row's identity would depend on whether a selection row happened to be
        // drawn above it.
        if let Some(point) = ctx.selected_point.filter(|p| p.recon == id) {
            any = true;
            let row = ui
                .push_id("point_selected", |ui| {
                    ui.add(egui::Button::selectable(
                        true,
                        format!("selected: {}", point_id(&node.recon, point.index())),
                    ))
                })
                .inner;
            if out.hit(row_id(id, "point_selected"), row).clicked() {
                out.response.select_point = Some(point);
            }
        }
        if let Some(point) = ctx
            .hovered_point
            .filter(|p| p.recon == id && Some(*p) != ctx.selected_point)
        {
            any = true;
            let row = ui
                .push_id("point_hovered", |ui| {
                    ui.add(egui::Button::selectable(
                        false,
                        egui::RichText::new(format!(
                            "hovered: {}",
                            point_id(&node.recon, point.index())
                        ))
                        .weak(),
                    ))
                })
                .inner;
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
