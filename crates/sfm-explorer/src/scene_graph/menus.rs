// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph's context menus: everything that opens on a right-click and
//! closes when it is used.
//!
//! Two menus, one per row kind. The reconstruction row's
//! ([`node_context_menu`]) carries the whole-node actions — visibility, tint,
//! align, reload, reset, close — and the image row's ([`image_context_menu`])
//! carries the two `Resect Image` entries. They are together because a menu is
//! the one place in the panel where an item is *described* rather than drawn:
//! each entry has a verb, an availability rule and a hover text explaining a
//! refusal, and those three read as a set.
//!
//! Nothing here mutates the scene beyond a node's own display state. An action
//! that touches a reconstruction is reported on [`TreeOutput::response`] and
//! carried out by `AppState` after the frame, so a menu item cannot leave the
//! tree half-walked.

use eframe::egui;

use crate::action_log::{tint_text, Kind};
use crate::align::AlignSource;
use crate::resect::ResectFrom;
use crate::scene::{ImageRef, NodeTint, ReconId, SceneNode, TINT_PALETTE};

use super::cameras::{ResectAvailability, MATCHES_DISABLED_HINT};
use super::{row_id, AlignTarget, TreeOutput};

/// The reconstruction row's context menu.
///
/// `Solo` is deliberately absent: it lives on the row itself, where a view mode
/// toggled this often belongs (see [`super::show_node_header`]).
pub(super) fn node_context_menu(ui: &mut egui::Ui, node: &mut SceneNode, out: &mut TreeOutput) {
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
/// The image row's context menu: the two `Resect Image` entries.
///
/// Both are kept visible and greyed rather than hidden when unavailable — the
/// action exists on every image row, and an entry that vanishes reads as an
/// action that was never implemented. The hover text says which of the two
/// reasons applies.
pub(super) fn image_context_menu(
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
