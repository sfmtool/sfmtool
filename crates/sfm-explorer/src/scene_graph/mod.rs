// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph panel — the tree of loaded reconstructions.
//!
//! A fifth dock tab titled **"Scene"**, and the control surface for everything
//! the scene graph adds: which reconstructions are loaded, which of their layers
//! are drawn, which of them capture pointer picks, which one the sequence-shaped
//! panels follow, and how a node leaves. See `specs/gui/scene-graph.md`
//! ("Scene Graph Panel").
//!
//! ## Shape of the tree
//!
//! ```text
//! ▾ 👁 S 🖱 ▪ run_a       1.2M pts · 243 imgs · 2 cams   ← S = solo, ▪ = tint
//!   ▾   Camera Intrinsics (2)      ← no eye: intrinsics draw nothing
//!         #0  OPENCV_FISHEYE  480×480  f 240.1  26 images
//!   ▾ 👁 Camera Images (243)
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
use sfmtool_core::CameraIntrinsics;

use crate::action_log::{interactive_text, tint_text, visibility_text, ActionLog, Kind, Layer};
use crate::align::{AlignOptions, AlignSource};
use crate::resect::ResectFrom;
use crate::scene::{
    point_id, CameraRef, ImageRef, NodeTint, PointRef, ReconId, SceneNode, TINT_PALETTE,
};
use crate::state::AppState;
use sfmtool_core::geometry::MIN_OTHER_POSED_IMAGES;

/// Height of one tree row. Fixed so the image list can be virtualized, and so
/// the eye/cursor toggles line up down the tree regardless of label height.
const ROW_HEIGHT: f32 = 18.0;

/// Height cap on a list drawn inside an expanded group: the virtualized image
/// list under Camera Images, and the camera list under Camera Intrinsics.
/// Bounded rather than unbounded so a 50K-image node cannot push every node
/// below it off the panel.
const LIST_MAX_HEIGHT: f32 = 220.0;

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

/// Solo toggle — "show only this reconstruction".
///
/// A letter rather than a pictograph, and deliberately: `S` is what a mixer's
/// solo button has said for fifty years, it needs no bundled emoji to exist,
/// and no glyph in egui's fonts says "only this one" any more clearly. It dims
/// when off exactly like the eye and the cursor beside it.
const SOLO_GLYPH: &str = "S";

/// Side of the square tint swatch painted on a tinted node's row.
const SWATCH_SIZE: f32 = 8.0;

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
    /// An image row was clicked.
    pub select_image: Option<ImageRef>,
    /// A Camera Intrinsics row was clicked.
    pub select_camera: Option<CameraRef>,
    /// A Camera Intrinsics row was double-clicked — frame every image taken
    /// through it in the 3D viewport.
    pub zoom_to_camera: Option<CameraRef>,
    /// The Points group's `selected:` row was clicked.
    pub select_point: Option<PointRef>,
    /// An image row was double-clicked — enter/switch camera view.
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
    /// The row's `S` was clicked: solo this node, or end the solo if it is
    /// already the soloed one. A view mode rather than node state, so unlike
    /// the eyes it travels through the response to `AppState::toggle_solo`
    /// instead of being written into the node.
    pub toggle_solo: Option<ReconId>,
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
    /// `Resect Image` / `Resect Image from Matches…` chosen on an image row:
    /// the image to resect and which correspondence source was asked for. The
    /// `.matches` file itself is chosen a layer up, where the file dialog and
    /// the per-node memory of the last path live — see [`crate::resect`].
    pub resect_image: Option<(ImageRef, ResectFrom)>,
}

/// Scene Graph panel state.
///
/// Only what the tree has to remember between frames; expansion lives in egui's
/// own memory under [`row_id`], and everything else is read from `AppState`.
#[derive(Default)]
pub struct SceneGraphPanel {
    /// The image selection as of the previous frame. An image list scrolls its
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
        self.hits.clear();
        let mut response = SceneGraphResponse::default();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            response.has_pointer = panel_rect.contains(pos);
        }

        // Before the empty-scene bail: an agent may be about to open the first
        // file, and the endpoint is exactly what a human wants to read off the
        // window at that moment.
        show_mcp_header(ui, state);

        if state.scene.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No reconstruction loaded");
            });
            self.prev_selected_image = None;
            return response;
        }

        // Read the selection out before the mutable walk over `scene` below:
        // every field of `AppState` is one borrow, and the node eyes need `&mut`.
        let selected_recon = state.selected_recon;
        let solo = state.solo;
        let selected_image = state.selected_image;
        let selected_camera = state.selected_camera;
        let selected_point = state.selected_point;
        let hovered_image = state.hovered_image;
        let hovered_point = state.hovered_point;
        // Auto-scroll fires on a selection change from *another* panel; a click
        // inside this one lands in the response and is applied after the frame,
        // so by construction it cannot also trigger a scroll.
        let selection_moved = self.prev_selected_image != selected_image;
        self.prev_selected_image = selected_image;

        // The eyes, the interaction cursor and the tint are written straight
        // into the node — they *are* per-node display state, with nothing for
        // `dock.rs` to arbitrate — so the log has to travel alongside the nodes
        // rather than being reached through `state`, which is borrowed for the
        // walk. See `AppState::scene_and_log`.
        let (scene, log) = state.scene_and_log();
        let mut out = TreeOutput {
            response,
            hits: &mut self.hits,
            align_options: &mut self.align_options,
            targets: &targets,
            log,
        };

        egui::ScrollArea::vertical()
            .id_salt("scene_graph_tree")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                for node in scene.iter_mut() {
                    let ctx = NodeContext {
                        selected: selected_recon == Some(node.id),
                        soloed: solo == Some(node.id),
                        visible: crate::scene::is_visible(node, solo),
                        selected_image,
                        selected_camera,
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

/// The header line naming the live MCP endpoint, and how many tool calls it
/// has served.
///
/// Draws nothing when no endpoint is running, which is the usual case. The
/// counter is live so a human can see the agent working, and the endpoint is
/// selectable text because pasting it into a client config is the thing people
/// want to do with it.
#[cfg(feature = "mcp")]
fn show_mcp_header(ui: &mut egui::Ui, state: &AppState) {
    let Some(mcp) = &state.mcp else {
        return;
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("MCP").strong());
        ui.add(egui::Label::new(egui::RichText::new(mcp.endpoint()).monospace()).selectable(true))
            .on_hover_text(
                "An agent can drive this window through this endpoint. Started with --mcp.",
            );
        ui.weak(match mcp.requests {
            1 => "· 1 call".to_string(),
            n => format!("· {n} calls"),
        });
    });
    ui.separator();
}

#[cfg(not(feature = "mcp"))]
fn show_mcp_header(_ui: &mut egui::Ui, _state: &AppState) {}

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
    /// Where the toggles that write straight into a node record what they did.
    log: &'a mut ActionLog,
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

    /// Record a toggle in the tree, and hand its response back through, so a
    /// row stays one statement: draw it, note where it landed, log what it did.
    fn logged(
        &mut self,
        id: egui::Id,
        response: egui::Response,
        text: impl FnOnce() -> String,
    ) -> egui::Response {
        let response = self.hit(id, response);
        // `clicked()` rather than `changed()`: a glyph toggle flips the flag
        // itself and reports no change, being an `egui::Button`.
        if response.clicked() {
            self.log.record(Kind::Scene, text());
        }
        response
    }
}

/// The selection/hover context one node's subtree is drawn against.
struct NodeContext {
    selected: bool,
    /// This node is the soloed one.
    soloed: bool,
    /// Whether the node is actually drawn — its own eye AND the solo override.
    /// The master eye is lit from *this* rather than from `node.visible`, so a
    /// node another node's solo is hiding reads as dark without its own flag
    /// having been touched.
    visible: bool,
    selected_image: Option<ImageRef>,
    /// The selected intrinsics. By the coupling invariant this is `Some`
    /// whenever `selected_image` is, and names that image's camera — so one
    /// highlight rule on the camera rows covers both "the camera you picked"
    /// and "the camera the image you picked was taken through".
    selected_camera: Option<CameraRef>,
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
        show_camera_intrinsics_group(ui, node, ctx, out);
        show_camera_images_group(ui, node, ctx, out);
        show_points_group(ui, node, ctx, out);
        if node.has_patch_data() {
            ui.horizontal(|ui| {
                ui.set_height(ROW_HEIGHT);
                let eye = eye_toggle(
                    ui,
                    row_id(node.id, "patches_eye"),
                    &mut node.show_patches,
                    None,
                    "Show this node's patches",
                );
                let shown = node.show_patches;
                out.logged(row_id(node.id, "patches_eye"), eye, || {
                    visibility_text(&node.label, Layer::Patches, shown)
                });
                ui.label("Patches");
            });
        }
    });
}

/// `[👁] [S] [🖱] ▪ label   1.2M pts · 243 imgs · 2 cams` — the reconstruction
/// row's content, drawn to the right of the collapsing triangle.
///
/// Everything past the three toggles is a single click target spanning the row:
/// select, zoom-to-fit and the context menu all hang off it, so none of them
/// depends on the user finding the name's own few pixels.
fn show_node_header(
    ui: &mut egui::Ui,
    node: &mut SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    ui.set_height(ROW_HEIGHT);
    let id = node.id;
    let eye = eye_toggle(
        ui,
        row_id(id, "node_eye"),
        &mut node.visible,
        Some(ctx.visible),
        "Show this reconstruction",
    );
    let visible = node.visible;
    out.logged(row_id(id, "node_eye"), eye, || {
        visibility_text(&node.label, Layer::Node, visible)
    });

    // Solo lives on the row rather than in the context menu (the spec left that
    // open): it is a *transient* view mode used over and over while comparing —
    // solo A, look, solo B, look, off — and a menu would cost two clicks and a
    // popup over the viewport every time. Tint, set once per node, stays in the
    // menu. Painted lit only on the soloed node, so the row is also where you
    // read *which* node is soloed.
    //
    // The flip lands in a throwaway local: solo is app state, not node state,
    // so what counts is the request in the response — `dock.rs` applies it, and
    // `ctx.soloed` is what the glyph is drawn from next frame.
    let mut soloed = ctx.soloed;
    let solo = glyph_toggle(
        ui,
        row_id(id, "node_solo"),
        &mut soloed,
        None,
        SOLO_GLYPH,
        "Solo: draw only this reconstruction. Click again to show them all — \
         the other nodes' eyes are left exactly as you set them.",
    );
    if out.hit(row_id(id, "node_solo"), solo).clicked() {
        out.response.toggle_solo = Some(id);
    }

    let cursor = glyph_toggle(
        ui,
        row_id(id, "node_cursor"),
        &mut node.interactive,
        None,
        CURSOR_GLYPH,
        "Let hover and clicks in the 3D viewport reach this reconstruction",
    );
    let interactive = node.interactive;
    out.logged(row_id(id, "node_cursor"), cursor, || {
        interactive_text(&node.label, interactive)
    });

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

    // The tint swatch: a tinted node has to be identifiable in the tree as well
    // as in the viewport, or the only way to find out which node is the orange
    // one is to hide the others. Painted rather than a widget, like the accent
    // bar — it answers nothing, and a widget here would compete with the
    // row-wide target for the click. Its space is reserved on every row for the
    // same reason the bar's is.
    let (swatch, _) =
        ui.allocate_exact_size(egui::vec2(SWATCH_SIZE, ROW_HEIGHT), egui::Sense::hover());
    if let Some([r, g, b]) = node.tint.rgb() {
        let square =
            egui::Rect::from_center_size(swatch.center(), egui::vec2(SWATCH_SIZE, SWATCH_SIZE));
        ui.painter()
            .rect_filled(square, 2.0, egui::Color32::from_rgb(r, g, b));
        out.mark(row_id(node.id, "node_tint_swatch"), square);
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
        let counts = counts_text(ui, node);
        ui.add(egui::Label::new(egui::RichText::new(counts).weak().small()).selectable(false));
    });
}

/// The reconstruction row's context menu.
///
/// `Solo` is deliberately absent: it lives on the row itself, where a view mode
/// toggled this often belongs (see [`show_node_header`]).
fn node_context_menu(ui: &mut egui::Ui, node: &mut SceneNode, out: &mut TreeOutput) {
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
    show_tint_menu(ui, node, out);
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

/// `Tint ▸`: `Original`, then the palette.
///
/// Written straight into the node, like the eyes it sits beside and unlike
/// `Solo`: a tint *is* per-node display state, so there is nothing for
/// `dock.rs` to arbitrate. It reaches the GPU on the next frame's display
/// mirror, so the menu can stay open while the user tries colors and watches
/// the viewport — which is why the entries close nothing.
fn show_tint_menu(ui: &mut egui::Ui, node: &mut SceneNode, out: &mut TreeOutput) {
    let id = node.id;
    let menu = ui.menu_button("Tint", |ui| {
        let original = ui.radio_value(&mut node.tint, NodeTint::Original, "Original");
        let original = out.hit(row_id(id, "tint_original"), original);
        // `changed()`, not `clicked()`: these are radios, so re-choosing the
        // colour a node already wears is not a change and writes no entry.
        if original.changed() {
            out.log
                .record(Kind::Scene, tint_text(&node.label, node.tint));
        }
        ui.separator();
        for color in TINT_PALETTE.iter() {
            let [r, g, b] = color.rgb;
            // The entry is written in its own color: a palette is only useful
            // if you can see what you are choosing, and a colored name needs no
            // glyph that egui's bundled fonts might not have.
            let label = egui::RichText::new(color.name).color(egui::Color32::from_rgb(r, g, b));
            let entry = ui.radio_value(&mut node.tint, NodeTint::Tint(color), label);
            let entry = out.hit(row_id(id, &format!("tint_{}", color.name)), entry);
            if entry.changed() {
                out.log
                    .record(Kind::Scene, tint_text(&node.label, node.tint));
            }
        }
    });
    out.hit(row_id(id, "tint_menu"), menu.response);
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
        // "Camera Poses", not "Cameras": this fits the two clouds' *poses*
        // onto one another, and under the tree's vocabulary a bare "Cameras"
        // now reads as the intrinsics the Camera Images group is drawn from.
        let cameras = ui.radio_value(
            &mut out.align_options.source,
            AlignSource::Cameras,
            "Camera Poses",
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

/// Camera count up to which the Camera Intrinsics group opens by itself.
///
/// A typical reconstruction has one camera and a rig has two or three: for
/// those the list *is* the answer, and hiding it behind a triangle costs a
/// click for nothing. Beyond a handful — a per-image-intrinsics solve can
/// produce hundreds — it behaves like the image list and stays out of the way.
const INTRINSICS_AUTO_EXPAND_MAX: usize = 4;

/// `[▸] Camera Intrinsics (2)`, expanding to one row per intrinsics record.
///
/// **No eye.** Every other group row's eye drives a visibility flag on the
/// node, and intrinsics have no geometry of their own to hide; the column is
/// left blank rather than filled with a disabled glyph that would answer
/// nothing, and the label indents to match the groups above and below it.
///
/// Placed above Camera Images because it is the coarser of the two: an image
/// row denotes one posed view, a camera row denotes the whole set of them that
/// share a lens.
fn show_camera_intrinsics_group(
    ui: &mut egui::Ui,
    node: &SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    let count = node.recon.cameras.len();
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        row_id(node.id, "intrinsics"),
        count <= INTRINSICS_AUTO_EXPAND_MAX,
    );
    let header = state.show_header(ui, |ui| {
        ui.set_height(ROW_HEIGHT);
        // The blank eye column, *allocated* rather than skipped: an eye is a
        // widget, so only allocating the same box keeps this label on the same
        // x as the ones that have one.
        ui.allocate_exact_size(egui::vec2(TOGGLE_WIDTH, ROW_HEIGHT), egui::Sense::hover());
        ui.label(format!("Camera Intrinsics ({count})"));
    });
    header.body(|ui| show_camera_rows(ui, node, ctx, out));
}

/// The per-camera rows: `#0  OPENCV_FISHEYE  480×480  f 240.1  26 images`.
///
/// Laid out plainly rather than virtualized — the count is bounded by the
/// number of *distinct* intrinsics, which is small even in the pathological
/// per-image case — but capped at [`LIST_MAX_HEIGHT`] behind a scroll area like
/// the image list, so that pathological case cannot bury the node below it
/// either.
fn show_camera_rows(ui: &mut egui::Ui, node: &SceneNode, ctx: &NodeContext, out: &mut TreeOutput) {
    let cameras = &node.recon.cameras;
    if cameras.is_empty() {
        ui.weak("No cameras");
        return;
    }
    // One pass over the images rather than one per row: the rows want a count
    // each, and the image list is the long one.
    let mut uses = vec![0usize; cameras.len()];
    for image in &node.recon.images {
        if let Some(count) = uses.get_mut(image.camera_index as usize) {
            *count += 1;
        }
    }

    egui::ScrollArea::vertical()
        .id_salt(row_id(node.id, "intrinsics_list"))
        .max_height(LIST_MAX_HEIGHT)
        // Shrink vertically: two cameras should take two rows, not the 220px
        // box the image list always fills.
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for (index, camera) in cameras.iter().enumerate() {
                let reference = CameraRef::new(node.id, index);
                let selected = ctx.selected_camera == Some(reference);
                let mut text = egui::RichText::new(camera_row_text(index, camera, uses[index]));
                if uses[index] == 0 && !selected {
                    // A camera no image references is legal in a `.sfmr` and
                    // worth seeing rather than hiding — but it describes
                    // nothing that is on screen, so it reads weak.
                    text = text.weak();
                }
                // Salted like the image rows, so a row's identity is its camera
                // index rather than its position in whatever slice is drawn.
                let row = ui
                    .push_id(index, |ui| ui.add(egui::Button::selectable(selected, text)))
                    .inner
                    .on_hover_ui(|ui| camera_tooltip(ui, camera));
                let row = out.hit(row_id(node.id, &format!("intrinsics_{index}")), row);
                if row.clicked() {
                    out.response.select_camera = Some(reference);
                }
                if row.double_clicked() {
                    out.response.zoom_to_camera = Some(reference);
                }
                // No hover channel, deliberately: cross-panel hover is a
                // two-field protocol with an ownership rule per panel, and a
                // third field threaded through every panel would buy a preview
                // of a selection that is one click away.
            }
        });
}

/// `#0  OPENCV_FISHEYE  480×480  f 240.1  26 images` — one intrinsics record on
/// one line, with a `β` on a model whose parameterization is not yet frozen.
fn camera_row_text(index: usize, camera: &CameraIntrinsics, uses: usize) -> String {
    let beta = if camera.model.beta_note().is_some() {
        " β"
    } else {
        ""
    };
    let images = if uses == 1 {
        "1 image".to_string()
    } else {
        format!("{uses} images")
    };
    format!(
        "#{index}  {}{beta}  {}×{}  {}  {images}",
        camera.model.model_name(),
        camera.width,
        camera.height,
        focal_text(camera),
    )
}

/// `f 240.1`, or `f 240.1/239.7` when the model carries two focal lengths and
/// they differ.
///
/// One decimal, because the row is a summary — the exact values are in the
/// hover tooltip one pointer-rest away.
fn focal_text(camera: &CameraIntrinsics) -> String {
    let (fx, fy) = camera.focal_lengths();
    if fx == fy {
        format!("f {fx:.1}")
    } else {
        format!("f {fx:.1}/{fy:.1}")
    }
}

/// The hover tooltip: the model's whole parameter list, in the order
/// [`sfmtool_core::CameraModel::parameter_names`] declares.
///
/// That order is the point — it is the one `sfm inspect` prints, so the tree
/// and the CLI can be read side by side rather than diffed against a
/// `BTreeMap`'s lexicographic order, which separates related terms and puts
/// `bspline_c10` before `bspline_c2`.
///
/// A beta model's note lands here too, under a separator, rather than on the
/// `β` itself: egui hangs a tooltip off a whole widget, and the row is one
/// button — so a sub-span of its label has nowhere of its own to put one.
fn camera_tooltip(ui: &mut egui::Ui, camera: &CameraIntrinsics) {
    ui.label(format!(
        "{} · {}×{}",
        camera.model.model_name(),
        camera.width,
        camera.height
    ));
    let parameters = camera.parameters();
    let width = parameters
        .iter()
        .map(|(name, _)| name.chars().count())
        .max()
        .unwrap_or(0);
    let mut table = String::new();
    for (name, value) in &parameters {
        table.push_str(&format!("{name:<width$}  {value:>14.6}\n"));
    }
    ui.label(egui::RichText::new(table.trim_end()).monospace());
    if let Some(note) = camera.model.beta_note() {
        ui.separator();
        ui.label(egui::RichText::new(note).weak());
    }
}

/// `[▸] [👁] Camera Images (243)`, expanding to a virtualized per-image list.
///
/// Counts `recon.images.len()`: a `.sfmr` *image* is one posed view, and the
/// intrinsics several of them share is a *camera* — a distinction the row used
/// to lose by calling itself `Cameras`.
fn show_camera_images_group(
    ui: &mut egui::Ui,
    node: &mut SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        row_id(node.id, "camera_images"),
        // Collapsed: the image list is the Image Browser's job, and an
        // expanded-by-default list would bury every node below this one.
        false,
    );
    let id = node.id;
    let header = state.show_header(ui, |ui| {
        ui.set_height(ROW_HEIGHT);
        let eye = eye_toggle(
            ui,
            row_id(id, "camera_images_eye"),
            &mut node.show_camera_images,
            None,
            "Show this node's camera frustums and image quads",
        );
        let shown = node.show_camera_images;
        out.logged(row_id(id, "camera_images_eye"), eye, || {
            visibility_text(&node.label, Layer::CameraImages, shown)
        });
        ui.label(format!("Camera Images ({})", node.recon.images.len()));
    });
    header.body(|ui| show_camera_image_rows(ui, node, ctx, out));
}

/// What an image row's `Resect Image` entries need to know about the node they
/// belong to, computed once for the whole list rather than once per row.
struct ResectAvailability {
    /// Whether each image carries a pose at all. A `.sfmr` row always has the
    /// fields; a non-finite one is a placeholder rather than a registration.
    posed: Vec<bool>,
    /// How many images of the node are posed.
    posed_count: usize,
    /// Whether the node's observations carry feature indexes — what the match
    /// rows are joined through, and so what the matches variant needs.
    feature_indexed: bool,
}

impl ResectAvailability {
    fn of(node: &SceneNode) -> Self {
        let posed: Vec<bool> = node
            .recon
            .images
            .iter()
            .map(|image| {
                image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
                    && image.translation_xyz.iter().all(|c| c.is_finite())
            })
            .collect();
        Self {
            posed_count: posed.iter().filter(|&&p| p).count(),
            posed,
            feature_indexed: node.recon.feature_indexes().is_some(),
        }
    }

    /// Why `Resect Image` is unavailable for image `index`, or `None` when it
    /// is available.
    fn refusal(&self, index: usize) -> Option<&'static str> {
        if !self.posed.get(index).copied().unwrap_or(false) {
            return Some(NOT_POSED_HINT);
        }
        // The target itself is one of the posed images, so "three others" is
        // four in total.
        (self.posed_count < MIN_OTHER_POSED_IMAGES + 1).then_some(TOO_FEW_POSED_HINT)
    }
}

/// Why `Resect Image` is greyed on an image with no pose.
const NOT_POSED_HINT: &str =
    "This image is not posed, so there is no pose to re-estimate against the rest.";

/// Why `Resect Image` is greyed on a node with too little of a reconstruction
/// to hold anything out from.
const TOO_FEW_POSED_HINT: &str =
    "Fewer than three other images of this reconstruction are posed. Two cameras fix \
     structure only up to their own degenerate freedoms, so re-estimating a pose \
     against them would measure the pair rather than the scene.";

/// Why `Resect Image from Matches…` is greyed on an embedded-patches node.
const MATCHES_DISABLED_HINT: &str =
    "Match rows are joined to observations by feature index, and this reconstruction \
     carries embedded patches instead — there is no feature index to join on.";

/// The per-image rows, laid out only for the visible slice of the list.
fn show_camera_image_rows(
    ui: &mut egui::Ui,
    node: &SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    let count = node.recon.images.len();
    if count == 0 {
        ui.weak("No images");
        return;
    }
    let resect = ResectAvailability::of(node);

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
        .id_salt(row_id(node.id, "image_list"))
        .max_height(LIST_MAX_HEIGHT)
        .auto_shrink([false, false]);
    if let Some(row) = scroll_target {
        let centered = row as f32 * row_height - LIST_MAX_HEIGHT / 2.0;
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
            let mut text = egui::RichText::new(name.to_string());
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
            let row = out.hit(row_id(node.id, &format!("image_{index}")), row);
            if row.clicked() {
                out.response.select_image = Some(image);
            }
            if row.double_clicked() {
                out.response.request_camera_view = Some(image);
            }
            if row.hovered() {
                out.response.hovered_image = Some(image);
            }
            // Built from `Popup` for the same reason the reconstruction row's
            // menu is: `Response::context_menu` closes on any click inside it,
            // and a greyed entry the user clicks to read its explanation would
            // take the menu down with it.
            egui::Popup::context_menu(&row)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .show(|ui| image_context_menu(ui, node.id, index, image, &resect, out));
        }
    });
}

/// The image row's context menu: the two `Resect Image` entries.
///
/// Both are kept visible and greyed rather than hidden when unavailable — the
/// action exists on every image row, and an entry that vanishes reads as an
/// action that was never implemented. The hover text says which of the two
/// reasons applies.
fn image_context_menu(
    ui: &mut egui::Ui,
    node: ReconId,
    index: usize,
    image: ImageRef,
    resect: &ResectAvailability,
    out: &mut TreeOutput,
) {
    let refusal = resect.refusal(index);
    let observations = ui
        .add_enabled(refusal.is_none(), egui::Button::new("Resect Image"))
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Re-estimate this image's pose against structure re-triangulated without it, \
             and show the answer as a new node beside this one.",
        );
    if out
        .hit(row_id(node, &format!("resect_{index}")), observations)
        .clicked()
    {
        out.response.resect_image = Some((image, ResectFrom::Observations));
        ui.close();
    }

    let matches_hint = refusal.or((!resect.feature_indexed).then_some(MATCHES_DISABLED_HINT));
    let matches = ui
        .add_enabled(
            matches_hint.is_none(),
            egui::Button::new("Resect Image from Matches…"),
        )
        .on_disabled_hover_text(matches_hint.unwrap_or_default())
        .on_hover_text(
            "The same, with the 2D-3D pairs taken from a .matches file — which admits \
             points this reconstruction never assigned to the image.",
        );
    if out
        .hit(row_id(node, &format!("resect_matches_{index}")), matches)
        .clicked()
    {
        out.response.resect_image = Some((image, ResectFrom::Matches));
        ui.close();
    }
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
        let eye = eye_toggle(
            ui,
            row_id(id, "points_eye"),
            &mut node.show_points,
            None,
            "Show this node's 3D points",
        );
        let shown = node.show_points;
        out.logged(row_id(id, "points_eye"), eye, || {
            visibility_text(&node.label, Layer::Points, shown)
        });
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
                row_id(id, "points_infinity"),
                &mut node.show_points_at_infinity,
                None,
                INFINITY_GLYPH,
                "Draw this node's w = 0 points — directions with no parallax",
            );
            let shown = node.show_points_at_infinity;
            out.logged(row_id(id, "points_infinity"), infinity, || {
                visibility_text(&node.label, Layer::PointsAtInfinity, shown)
            });
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
fn eye_toggle(
    ui: &mut egui::Ui,
    id: egui::Id,
    on: &mut bool,
    lit: Option<bool>,
    tooltip: &str,
) -> egui::Response {
    glyph_toggle(ui, id, on, lit, EYE_GLYPH, tooltip)
}

/// A one-glyph toggle button: full-strength when lit, weak otherwise.
///
/// `lit` is normally `None` — paint the glyph from the flag it toggles. The
/// master eye passes `Some(effective visibility)` instead, so a node hidden by
/// another node's solo reads as dark without its own flag being touched: the
/// eye still says what it will do the moment the solo ends.
///
/// Under an explicit id rather than the auto id a bare `ui.add` would take. An
/// auto id is a count of what was allocated before the widget in this `Ui`, so
/// adding or removing anything earlier in the row moves every later widget's
/// identity — and with it the hover, click and tooltip state egui keys on it.
fn glyph_toggle(
    ui: &mut egui::Ui,
    id: egui::Id,
    on: &mut bool,
    lit: Option<bool>,
    glyph: &str,
    tooltip: &str,
) -> egui::Response {
    let color = if lit.unwrap_or(*on) {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    let button = egui::Button::new(egui::RichText::new(glyph).color(color))
        .frame(false)
        .min_size(egui::vec2(TOGGLE_WIDTH, ROW_HEIGHT));
    let response = ui
        .push_id(id, |ui| ui.add(button))
        .inner
        .on_hover_text(tooltip);
    if response.clicked() {
        *on = !*on;
    }
    response
}

/// `1.2M pts · 243 imgs · 2 cams`, elided to what is left of the row.
///
/// Three counts make this the longest row in a panel that defaults to 18% of
/// the window, so rather than truncate or wrap it drops a count at a time: the
/// camera count first (it is also on its own group row, one line down), then
/// the image count, leaving the point count — which has no other home in the
/// tree — last to go.
///
/// Measured against the width actually left after the label rather than
/// against a character budget, so widening the panel brings the counts back.
fn counts_text(ui: &egui::Ui, node: &SceneNode) -> String {
    let points = format!("{} pts", compact_count(node.recon.points.len()));
    let images = format!("{} imgs", compact_count(node.recon.images.len()));
    let cameras = format!("{} cams", compact_count(node.recon.cameras.len()));
    let available = ui.available_width();
    let font = egui::TextStyle::Small.resolve(ui.style());
    let fits = |text: &str| {
        let galley =
            ui.painter()
                .layout_no_wrap(text.to_owned(), font.clone(), egui::Color32::PLACEHOLDER);
        galley.size().x <= available
    };
    let all_three = format!("{points} · {images} · {cameras}");
    if fits(&all_three) {
        return all_three;
    }
    let without_cameras = format!("{points} · {images}");
    if fits(&without_cameras) {
        return without_cameras;
    }
    points
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
