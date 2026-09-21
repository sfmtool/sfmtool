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
//!       SIFT Index  1.2M descriptors   ← current / stale / none
//! ```
//!
//! Expansion state lives in [`egui::collapsing_header::CollapsingState`] under
//! **explicit ids** ([`row_id`]) rather than `CollapsingHeader`'s
//! id-salt-plus-parent scheme, following the viewport HUD: that keeps a row's
//! open/closed state addressable from outside the panel, which is how the tests
//! tell an expanded subtree from a collapsed one.
//!
//! ## How the module is laid out
//!
//! This file owns the tree walk and the panel entry point: the collapsing
//! header per node, the reconstruction row, the Points group and the response
//! the dock reads back. The rest is one module per kind of thing a row is made
//! of:
//!
//! - [`mod@cameras`] — the Camera Intrinsics and Camera Images groups, the
//!   virtualized image list, and [`cameras::ResectAvailability`], which decides
//!   whether a resection can be asked for at all.
//! - [`mod@menus`] — the three context menus, one per row kind.
//! - [`mod@widgets`] — the eye and glyph toggles every row is built from, and
//!   the two spellings of a count.
//!
//! Everything is private to `scene_graph`; the cut is about where to look, not
//! about a boundary anyone outside crosses.
//!
//! ## What the panel does not do
//!
//! The Points group expands to the **selection and hover rows only**, never a
//! full per-point listing: millions of rows are not navigable, and past about
//! 16.7M row-pixels egui's `f32` scroll coordinates lose integer precision, so
//! such a list would misbehave mechanically as well as ergonomically.

use eframe::egui;
use sfmtool_core::bench::{BenchEntry, StageKind};

use crate::action_log::{interactive_text, visibility_text, ActionLog, Kind, Layer};
use crate::align::AlignOptions;
use crate::resect::ResectFrom;
use crate::scene::{point_id, CameraRef, ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;

mod cameras;
mod menus;
mod widgets;

use cameras::{show_camera_images_group, show_camera_intrinsics_group};
use menus::{node_context_menu, sift_index_menu};
use widgets::{counts_text, eye_toggle, glyph_toggle, with_thousands};

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
/// cannot hand its expansion state to a later one, and the same file opened
/// twice is two nodes with expansion state of their own.
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
    /// `Convert to Embedded Patches` chosen from a reconstruction's context
    /// menu: give every point a patch frame and carry each observation's
    /// keypoint inline, as the node's next version. A bulk edit, and a long
    /// one, so `AppState::start_convert_to_embedded_patches` sends it to a
    /// worker after the frame.
    pub convert_to_embedded_patches: Option<ReconId>,
    /// `Bundle Adjust...` chosen from a reconstruction's context menu: ask,
    /// through `AppState::open_bundle_adjust`, whether the shared focal is
    /// released, and on `Run` refine every pose and point of this node as its
    /// next version.
    pub bundle_adjust: Option<ReconId>,
    /// `Retriangulate All Points` chosen from a reconstruction's context menu:
    /// re-solve every point from its own observations at the poses and the lens
    /// the value already holds, as the node's next version. A bulk edit, and a
    /// long one, so `AppState::start_retriangulate_all_points` sends it to a
    /// worker after the frame.
    pub retriangulate_all_points: Option<ReconId>,
    /// `Prune Covered Observations` chosen from a reconstruction's context
    /// menu: retire every observation a finer tracked one covers, as the node's
    /// next version. A bulk edit, and a long one, so
    /// `AppState::start_prune_covered_observations` sends it to a worker after
    /// the frame.
    pub prune_covered_observations: Option<ReconId>,
    /// `Build SIFT Index` / `Rebuild SIFT Index` chosen, from the SIFT Index
    /// row's menu or from the reconstruction row's. Reads every `.sift` file of
    /// the node, so `AppState::start_build_sift_index` sends it to a worker
    /// after the frame.
    pub build_sift_index: Option<ReconId>,
    /// `Open...` chosen on the SIFT Index row, which asks the dock for a file
    /// chooser. The panel names no path: a chooser is not an egui widget, so it
    /// lives where the other file questions do.
    pub open_sift_index: Option<ReconId>,
    /// `Close Index` chosen on the SIFT Index row.
    pub close_sift_index: Option<ReconId>,
    /// `Close` chosen from a reconstruction's context menu.
    pub close_node: Option<ReconId>,
    /// A Bench row was clicked: make that item the active one of its kind. The
    /// item is named by its position on the bench rather than by its label, so
    /// this response stays a `Copy` value; the dock reads the label off the
    /// bench at that position.
    pub activate_bench_item: Option<(ReconId, usize)>,
    /// `Discard` chosen on a Bench row, the item named the same way.
    pub discard_bench_item: Option<(ReconId, usize)>,
    /// `Resect Image` / `Resect Image from Matches…` chosen on an image row:
    /// the image to resect and which correspondence source was asked for. The
    /// `.matches` file itself is chosen a layer up, where the file dialog and
    /// the per-node memory of the last path live — see [`crate::resect`]. A bulk
    /// edit, carried out by `AppState::resect_image` after the frame.
    pub resect_image: Option<(ImageRef, ResectFrom)>,
    /// `Delete Image` chosen on an image row. A bulk edit on the node it
    /// belongs to, carried out by `AppState::delete_image` after the frame.
    pub delete_image: Option<ImageRef>,
    /// `Move Camera` chosen on an image row: look through that image and take
    /// its camera in hand. Not an edit yet -- the lock is viewport state, and
    /// the edit is what committing it pushes. See `crate::camera_lock`.
    pub move_camera: Option<ImageRef>,
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
        // The one node an operation locks, and the sentence that refuses an
        // edit of it: taken here for the same reason `targets` is, since the
        // walk below borrows the scene mutably.
        let busy = state
            .background_task()
            .map(|task| task.node)
            .and_then(|node| state.busy_refusal(node).map(|why| (node, why)));
        // The SIFT Index row is truthful on a node whose bench is empty, so the
        // look that the Track Edit panel and the bench do happens here too --
        // once per node per frame, and remembered, so a reconstruction with no
        // index beside it is not stat-ed again (`specs/gui/sift-index.md`).
        let ids: Vec<ReconId> = state.scene.iter().map(|node| node.id).collect();
        for id in ids {
            state.refresh_sift_index(id);
        }
        // What each node's row and menu say about its index, read out before
        // the mutable walk below for the reason `targets` is.
        let indexes: std::collections::HashMap<ReconId, SiftIndexRow> = state
            .scene
            .iter()
            .map(|node| (node.id, SiftIndexRow::of(state, node.id)))
            .collect();
        let targets: Vec<AlignTarget> = state
            .scene
            .iter()
            .map(|n| AlignTarget {
                id: n.id,
                label: n.label.clone(),
                feature_indexed: n.recon().feature_indexes().is_some(),
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
            busy,
            indexes: &indexes,
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
    /// The node a background operation is running on and the sentence that
    /// refuses an edit of it, or `None` when nothing is running. Read out
    /// before the walk, because the walk holds the scene and `AppState` is one
    /// borrow.
    busy: Option<(ReconId, String)>,
    /// What each node's SIFT Index row says, read out before the walk for the
    /// same reason.
    indexes: &'a std::collections::HashMap<ReconId, SiftIndexRow>,
    /// Where the toggles that write straight into a node record what they did.
    log: &'a mut ActionLog,
}

impl TreeOutput<'_> {
    /// Why an edit of `id` is refused right now, or `None` --
    /// `AppState::busy_refusal`'s answer, carried in.
    fn busy_refusal(&self, id: ReconId) -> Option<&str> {
        self.busy
            .as_ref()
            .filter(|(busy, _)| *busy == id)
            .map(|(_, why)| why.as_str())
    }

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
        show_bench_groups(ui, node, out);
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
        show_sift_index_row(ui, node, out);
    });
}

/// What one node's SIFT Index row says, taken off `AppState` before the tree
/// walk borrows the scene.
///
/// A row rather than a question asked while drawing, because the answer is
/// derived when a file is opened or built and when a version moves the image
/// table, never per frame (`specs/gui/sift-index.md`).
pub(super) struct SiftIndexRow {
    /// Which of the three states the index is in.
    state: crate::sift_index::SiftIndexState,
    /// The file that is open, or where a build would put one. `None` on a node
    /// with no path on disk.
    path: Option<String>,
    /// How many descriptors and images the open index holds.
    counts: Option<(usize, usize)>,
    /// The sentence naming the first discrepancy, when it is stale.
    stale_reason: Option<String>,
    /// Why a build cannot start, or `None` when it can.
    build_refusal: Option<String>,
    /// Why closing would do nothing, or `None` when there is one to close.
    close_refusal: Option<String>,
    /// The sentence a node with nowhere to put an index is told, which is also
    /// the whole of what its hover can say: there is no path to name.
    save_first: Option<String>,
    /// Whether the build is the operation running right now.
    building: bool,
}

impl SiftIndexRow {
    fn of(state: &AppState, id: ReconId) -> Self {
        let index = state.sift_index(id);
        Self {
            state: state.sift_index_state(id),
            path: index
                .map(|index| index.path.display().to_string())
                .or_else(|| {
                    state
                        .sift_index_path(id)
                        .map(|path| path.display().to_string())
                }),
            counts: index.map(|index| (index.feature_count(), index.images)),
            stale_reason: index.and_then(|index| index.stale_reason().map(str::to_string)),
            build_refusal: state.build_sift_index_refusal(id),
            close_refusal: index
                .is_none()
                .then(|| "No SIFT index is open beside this reconstruction.".to_string()),
            save_first: state.sift_index_home_refusal(id),
            building: state.building_sift_index(id),
        }
    }

    /// What the row reads to the right of its name.
    fn status(&self) -> String {
        if self.building {
            return "building...".to_string();
        }
        match (self.state, self.counts) {
            (crate::sift_index::SiftIndexState::Current, Some((descriptors, _))) => {
                format!("{} descriptors", with_thousands(descriptors))
            }
            (state, _) => state.name().to_string(),
        }
    }

    /// The whole of what hovering the row says.
    ///
    /// The path first, because the one question a row of three words cannot
    /// answer is *which file*. It is the file that is open, or -- when none is
    /// -- where a build would put one, said in words that do not read as a
    /// file that is there. A node with nowhere to put one has no path to name
    /// at all, so it gets the sentence its menu entries are greyed with.
    fn hover(&self) -> String {
        let mut text = match (&self.path, self.state) {
            (Some(path), crate::sift_index::SiftIndexState::None) => {
                format!("No SIFT index file is there.\nA build writes {path}")
            }
            (Some(path), _) => path.clone(),
            (None, _) => self
                .save_first
                .clone()
                .unwrap_or_else(|| "This reconstruction came from no file.".to_string()),
        };
        if let Some((descriptors, images)) = self.counts {
            text.push_str(&format!(
                "\n{} descriptors of {} images",
                with_thousands(descriptors),
                with_thousands(images)
            ));
        }
        if let Some(why) = &self.stale_reason {
            text.push('\n');
            text.push_str(why);
        }
        text
    }

    /// What the build entry is called: a first one, or one over again.
    fn build_label(&self) -> &'static str {
        match self.state {
            crate::sift_index::SiftIndexState::None => BUILD_SIFT_INDEX,
            _ => REBUILD_SIFT_INDEX,
        }
    }
}

/// What the entry that indexes a node's `.sift` files is called when the node
/// has no index, in the two menus that offer it and the tests that aim at them.
pub(crate) const BUILD_SIFT_INDEX: &str = "Build SIFT Index";

/// What the same entry is called when one is already open.
pub(crate) const REBUILD_SIFT_INDEX: &str = "Rebuild SIFT Index";

/// `SIFT Index   1.2M descriptors` — the forest a bench search queries, as a
/// row of the node it belongs to.
///
/// Last among the group rows, after Points and after Patches: it is the one
/// part of a node that is a *file beside it* rather than something the node
/// holds. No eye and no children — nothing here is drawn in the viewport and
/// there is nothing under it to list — and clicking it selects nothing. The
/// three states are told apart by the status text and by its colour, so the row
/// answers "is there one, and is it still good" without a hover.
fn show_sift_index_row(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    let Some(index) = out.indexes.get(&node.id) else {
        return;
    };
    let id = node.id;
    ui.horizontal(|ui| {
        ui.set_height(ROW_HEIGHT);
        // The whole row is one target, claimed before its contents for the
        // reason the reconstruction row's is: a right-click anywhere along it
        // opens the menu, and an explicit id keeps the popup's identity off a
        // count of what was allocated before it.
        let available = ui.available_rect_before_wrap();
        let rect =
            egui::Rect::from_min_size(available.min, egui::vec2(available.width(), ROW_HEIGHT));
        let row = ui.interact(rect, row_id(id, "sift_index"), egui::Sense::click());
        let row = out
            .hit(row_id(id, "sift_index"), row)
            .on_hover_text(index.hover());
        crate::context_menu::on_secondary_click(&row)
            .show(|ui| sift_index_menu(ui, node, index, out));
        // `selectable(false)` on both texts, for the reason the reconstruction
        // row's carry it: egui's default `selectable_labels` gives a bare label
        // `Sense::click_and_drag()` so its text can be dragged out, and being
        // drawn *after* the row it wins every pointer hit that lands on it --
        // which left the name the one part of the row with no menu and no
        // hover.
        ui.add(egui::Label::new("SIFT Index").selectable(false));
        let status = egui::RichText::new(index.status()).small();
        let status = match index.state {
            crate::sift_index::SiftIndexState::Stale => status.color(ui.visuals().warn_fg_color),
            crate::sift_index::SiftIndexState::None => status.weak(),
            crate::sift_index::SiftIndexState::Current => status,
        };
        ui.add(egui::Label::new(status).selectable(false));
    });
}

/// What hovering a reconstruction row says: the `.sfmr` it came from, spelled
/// in full.
///
/// A node that came from no file says so in the vocabulary the `File > Save`
/// item's own refusal uses (`specs/gui/saving.md`), rather than naming a path
/// it does not have.
fn node_hover(node: &SceneNode) -> String {
    match &node.path {
        Some(path) => path.display().to_string(),
        None => "This reconstruction came from no file.".to_string(),
    }
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
    // The file the node is attached to, on the target that spans the row, so it
    // comes up over the name as well as over the gap. The label says the stem
    // and the window title says the file name, and neither says which directory
    // -- which is the question that arises the moment two runs of one capture
    // are loaded together.
    let row = out
        .hit(row_id(node.id, "node_label"), row)
        .on_hover_text(node_hover(node));
    if row.clicked() {
        out.response.select_recon = Some(node.id);
    }
    if row.double_clicked() {
        out.response.zoom_to_node = Some(node.id);
    }
    // Built with an explicit `close_behavior` for the sake of one setting:
    // egui's default menu closes on *any* click inside it, which would tear the
    // whole menu down the moment the user set one of the two `Align to` radio
    // buttons. `CloseOnClickOutside` leaves closing to the explicit
    // `ui.close()` on each item that actually does something.
    crate::context_menu::on_secondary_click(&row)
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

    // A leading `*` while the node is not at the version its file holds -- the
    // same marker the window title carries, on the row that names the node it
    // is about, which is the one that matters when several are loaded.
    let mut label = egui::RichText::new(if node.is_dirty() {
        format!("*{}", node.label)
    } else {
        node.label.clone()
    });
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

/// The bench, as two groups: `[▸] Bench Points (2)` and
/// `[▸] Bench Clusters (1)`.
///
/// Split by stage because the stages are two different things -- a track-stage
/// item stands at a position and a cluster-stage one is a set of image patches
/// with no geometry behind them (`specs/gui/bench.md`) -- and one mixed list
/// said which only in a row's hover text. An item that is taken up or down
/// moves between the groups, which is the whole of what the split is for.
///
/// A group with nothing in it is not drawn, which is what the one bench group
/// did when the bench was empty. The two are one bench all the same: there is
/// one active item across both, because the bench's kinds are items rather than
/// stages.
fn show_bench_groups(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    show_bench_stage_group(ui, node, out, StageKind::Track, "bench_points");
    show_bench_stage_group(ui, node, out, StageKind::Cluster, "bench_clusters");
}

/// One of the two bench groups: the items at `stage`, in the bench's own order,
/// the active one marked.
///
/// No eye: nothing on the bench is drawn from this row, and an item is not part
/// of the reconstruction. The groups are here because the tree is where a
/// node's parts are listed, and they are per node because an item names that
/// node's images and poses (`specs/gui/bench.md`).
///
/// Clicking a row makes that item active, which is a step like any other; the
/// panel that edits the item is [`crate::track_edit`], and a secondary click
/// offers *Discard*. An item is named to the dock by its **position on the
/// bench**, which is the position it holds in the whole list rather than in the
/// group it is drawn under, so both groups' rows reach the same item.
fn show_bench_stage_group(
    ui: &mut egui::Ui,
    node: &SceneNode,
    out: &mut TreeOutput,
    stage: StageKind,
    key: &str,
) {
    let bench = node.history.current_bench();
    let items: Vec<(usize, &BenchEntry)> = bench
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            entry
                .item
                .as_track()
                .is_some_and(|track| track.stage_kind() == stage)
        })
        .collect();
    if items.is_empty() {
        return;
    }
    let id = node.id;
    let active = crate::bench::active_track_label(bench).map(str::to_string);
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        row_id(id, key),
        true,
    );
    let title = match stage {
        StageKind::Track => "Bench Points",
        StageKind::Cluster => "Bench Clusters",
    };
    let header = state.show_header(ui, |ui| {
        ui.set_height(ROW_HEIGHT);
        ui.label(format!("{title} ({})", items.len()));
    });
    header.body(|ui| {
        for (position, entry) in items {
            let track = entry.item.as_track().expect("filtered to tracks above");
            let selected = active.as_deref() == Some(entry.label.as_str());
            let (kept, candidates, out_count) = track.verdict_counts();
            let text = format!("{}  {kept} in", entry.label);
            let row = ui
                .push_id(("bench_item", entry.label.as_str()), |ui| {
                    ui.add(egui::Button::selectable(selected, text))
                })
                .inner
                .on_hover_text(format!(
                    "{} · the {} stage · {kept} in, {candidates} candidates, {out_count} out",
                    entry.label,
                    track.stage_kind()
                ));
            let row = out.hit(row_id(id, &format!("bench_item_{}", entry.label)), row);
            if row.clicked() {
                out.response.activate_bench_item = Some((id, position));
            }
            crate::context_menu::on_secondary_click(&row).show(|ui| {
                let discard = ui.button("Discard");
                let key = format!("bench_discard_{}", entry.label);
                if out.hit(row_id(id, &key), discard).clicked() {
                    out.response.discard_bench_item = Some((id, position));
                    ui.close();
                }
            });
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
    let at_infinity = node.infinity_point_count();
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
        let count = with_thousands(node.point_count());
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
                        format!("selected: {}", point_id(node, point.index())),
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
                        egui::RichText::new(format!("hovered: {}", point_id(node, point.index())))
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

#[cfg(test)]
pub(crate) mod tests;
