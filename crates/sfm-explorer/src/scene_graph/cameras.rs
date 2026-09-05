// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph's camera and image rows: the two groups under a
//! reconstruction that list what it was shot with and what it was shot from.
//!
//! `Camera Intrinsics` lists the models — one row per camera, with the
//! parameters that fit on a line and the rest in a tooltip. `Camera Images`
//! lists the photographs, virtualized because a reconstruction can carry
//! thousands of them, and is where a resection is asked for.
//!
//! [`ResectAvailability`] is here rather than beside the menu it greys out,
//! because deciding it needs what this module already has in hand: the node's
//! pose count, the image's own pose, and whether a `.matches` file has been
//! chosen. The menu takes the answer and the hint text that goes with it.

use eframe::egui;
use sfmtool_core::geometry::MIN_OTHER_POSED_IMAGES;
use sfmtool_core::CameraIntrinsics;

use crate::action_log::{visibility_text, Layer};
use crate::scene::{CameraRef, ImageRef, SceneNode};

use super::menus::image_context_menu;
use super::widgets::eye_toggle;
use super::{row_id, NodeContext, TreeOutput, LIST_MAX_HEIGHT, ROW_HEIGHT, TOGGLE_WIDTH};

#[cfg(test)]
mod tests;

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
pub(super) fn show_camera_intrinsics_group(
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
pub(super) fn show_camera_images_group(
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
pub(super) struct ResectAvailability {
    /// Whether each image carries a pose at all. A `.sfmr` row always has the
    /// fields; a non-finite one is a placeholder rather than a registration.
    posed: Vec<bool>,
    /// How many images of the node are posed.
    posed_count: usize,
    /// Whether the node's observations carry feature indexes — what the match
    /// rows are joined through, and so what the matches variant needs.
    pub(super) feature_indexed: bool,
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
    pub(super) fn refusal(&self, index: usize) -> Option<&'static str> {
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
pub(super) const MATCHES_DISABLED_HINT: &str =
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
