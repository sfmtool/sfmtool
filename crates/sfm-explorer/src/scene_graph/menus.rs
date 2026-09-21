// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph's context menus: everything that opens on a right-click and
//! closes when it is used.
//!
//! Three menus, one per row kind. The reconstruction row's
//! ([`node_context_menu`]) carries the whole-node actions (select, zoom to fit,
//! align, reset transform, tint, bundle adjust, retriangulate, prune covered
//! observations, build the SIFT index, convert to embedded patches, close), the
//! SIFT Index row's ([`sift_index_menu`]) carries the three ways to give a node
//! an index or take one away, and the image row's ([`image_context_menu`])
//! carries the resection, the camera move and the image deletion. They are
//! together because a menu is the one place in the panel where an item is
//! *described* rather than drawn:
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
use super::{row_id, AlignTarget, SiftIndexRow, TreeOutput};

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
    show_bundle_adjust_entry(ui, node, out);
    show_retriangulate_entry(ui, node, out);
    show_prune_covered_entry(ui, node, out);
    // Above the conversion because it is where a person looks first: the
    // reconstruction row is the row they have in hand when they find a search
    // greyed, and the SIFT Index row below it offers the same entry.
    show_build_index_entry(ui, node.id, out);
    show_convert_entry(ui, node, out);
    ui.separator();
    if ui.button("Close").clicked() {
        out.response.close_node = Some(node.id);
        ui.close();
    }
}

/// What the entry that converts a node's observations is called, in the menu
/// and in the tests that aim at it.
pub(crate) const CONVERT_TO_EMBEDDED_PATCHES: &str = "Convert to Embedded Patches";

/// What the entry that refines every pose and point of a node is called, in the
/// menu and in the tests that aim at it.
pub(crate) const BUNDLE_ADJUST: &str = "Bundle Adjust...";

/// `Bundle Adjust...`: every pose and point of the node refined against its
/// observations, behind a dialog that asks whether the shared focal is
/// released.
///
/// First of the whole-value edits because it is the widest of them: it moves
/// the cameras as well as the points the entries under it re-read. Greyed
/// rather than hidden when it cannot run, on the edit's own gate
/// (`bundle_adjust_prompt::refusal`), so the entry and the edit cannot disagree
/// about when the adjustment can run. A busy node is refused first, as it is on
/// every sibling entry.
fn show_bundle_adjust_entry(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    let refusal = out
        .busy_refusal(node.id)
        .map(str::to_string)
        .or_else(|| crate::bundle_adjust_prompt::refusal(node.edited()));
    let entry = ui
        .add_enabled(refusal.is_none(), egui::Button::new(BUNDLE_ADJUST))
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Refine every pose and point of this reconstruction against its observations, as \
             one version. Asks first whether the shared focal length is released. Runs on a \
             worker thread and can be cancelled; Undo (Ctrl+Z) puts the geometry back.",
        );
    if out.hit(row_id(node.id, "bundle_adjust"), entry).clicked() {
        out.response.bundle_adjust = Some(node.id);
        ui.close();
    }
}

/// What the entry that re-solves every point of a node is called, in the menu
/// and in the tests that aim at it.
pub(crate) const RETRIANGULATE_ALL_POINTS: &str = "Retriangulate All Points";

/// `Retriangulate All Points`: every point of the node re-read from its own
/// observations, at the poses and the lens the value already holds.
///
/// Greyed rather than hidden when it cannot run, for the reason
/// [`show_convert_entry`] is, and on `AppState`'s own sentence so the greyed
/// entry and a call that asks anyway give one answer.
fn show_retriangulate_entry(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    let refusal = crate::state::edits::retriangulate_refusal(node, out.busy_refusal(node.id));
    let entry = ui
        .add_enabled(
            refusal.is_none(),
            egui::Button::new(RETRIANGULATE_ALL_POINTS),
        )
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Re-solve every point from its own observations at these poses and this lens, as \
             one version. Moves no camera. Runs on a worker thread and can be cancelled; Undo \
             (Ctrl+Z) puts the geometry back.",
        );
    if out
        .hit(row_id(node.id, "retriangulate_all_points"), entry)
        .clicked()
    {
        out.response.retriangulate_all_points = Some(node.id);
        ui.close();
    }
}

/// What the entry that retires a node's covered observations is called, in the
/// menu and in the tests that aim at it.
pub(crate) const PRUNE_COVERED_OBSERVATIONS: &str = "Prune Covered Observations";

/// `Prune Covered Observations`: every observation of the node a finer tracked
/// one covers, handed over to the feature that supersedes it.
///
/// Sits directly under [`show_retriangulate_entry`] because the two are the
/// pair a reviewer reaches for together: one re-reads what each track says, the
/// other decides which tracks should still be saying it. Greyed rather than
/// hidden when it cannot run, on `AppState`'s own sentence.
fn show_prune_covered_entry(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    let refusal = crate::state::edits::prune_covered_refusal(node, out.busy_refusal(node.id));
    let entry = ui
        .add_enabled(
            refusal.is_none(),
            egui::Button::new(PRUNE_COVERED_OBSERVATIONS),
        )
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Retire every observation a finer tracked one covers in the same image, and drop \
             the points left with fewer than two, as one version. Moves nothing and re-solves \
             nothing. Pinned points are spared. Runs on a worker thread and can be cancelled; \
             Undo (Ctrl+Z) puts the observations back.",
        );
    if out
        .hit(row_id(node.id, "prune_covered_observations"), entry)
        .clicked()
    {
        out.response.prune_covered_observations = Some(node.id);
        ui.close();
    }
}

/// `Convert to Embedded Patches`: the one entry on this menu that is a bulk
/// edit of the reconstruction rather than display state.
///
/// Kept visible and greyed rather than hidden when it cannot run, like every
/// other refusable entry in the panel: the operation exists on every
/// reconstruction row, and an entry that vanishes reads as one that was never
/// built. The hover text is `AppState`'s own sentence, so the greyed entry and
/// a call that asks anyway give one answer.
fn show_convert_entry(ui: &mut egui::Ui, node: &SceneNode, out: &mut TreeOutput) {
    let refusal = crate::state::edits::convert_refusal(node, out.busy_refusal(node.id));
    let entry = ui
        .add_enabled(
            refusal.is_none(),
            egui::Button::new(CONVERT_TO_EMBEDDED_PATCHES),
        )
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Give every point a patch frame from its mean viewing direction and carry each \
             observation's .sift keypoint inline, as one version. Runs on a worker thread, and \
             reads the workspace's .sift files. Undo (Ctrl+Z) puts the sift_files version back.",
        );
    if out
        .hit(row_id(node.id, "to_embedded_patches"), entry)
        .clicked()
    {
        out.response.convert_to_embedded_patches = Some(node.id);
        ui.close();
    }
}

/// `Build SIFT Index` / `Rebuild SIFT Index`: every `.sift` file of the node
/// indexed into a `.kdf` beside its `.sfmr`.
///
/// On both menus that name the index, under one function, so the two cannot
/// drift apart in what they are called or in when they are live. Greyed rather
/// than hidden on its own sentence, like every other refusable entry here: an
/// entry that vanishes reads as one that was never built.
fn show_build_index_entry(ui: &mut egui::Ui, id: ReconId, out: &mut TreeOutput) {
    let Some(index) = out.indexes.get(&id) else {
        return;
    };
    let refusal = index.build_refusal.clone();
    let entry = ui
        .add_enabled(refusal.is_none(), egui::Button::new(index.build_label()))
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Index every .sift file of this reconstruction into a .kdf beside its .sfmr, and \
             open it. What a bench search queries. Runs on a worker thread.",
        );
    if out.hit(row_id(id, "build_sift_index"), entry).clicked() {
        out.response.build_sift_index = Some(id);
        ui.close();
    }
}

/// What the entry that lets go of an open index is called, in the menu and in
/// the tests that aim at it.
pub(crate) const CLOSE_SIFT_INDEX: &str = "Close Index";

/// The SIFT Index row's own menu: the build, a file of the person's choosing,
/// and letting go of what is open.
pub(super) fn sift_index_menu(
    ui: &mut egui::Ui,
    node: &SceneNode,
    index: &SiftIndexRow,
    out: &mut TreeOutput,
) {
    let id = node.id;
    show_build_index_entry(ui, id, out);
    let busy = out.busy_refusal(id).map(str::to_string);
    let open = ui
        .add_enabled(busy.is_none(), egui::Button::new("Open..."))
        .on_disabled_hover_text(busy.clone().unwrap_or_default())
        .on_hover_text(
            "Search a .kdf of your own choosing. One that is not over this \
                        reconstruction's images opens all the same, and the row says why it \
                        will not do.",
        );
    if out.hit(row_id(id, "open_sift_index"), open).clicked() {
        out.response.open_sift_index = Some(id);
        ui.close();
    }
    let close = index.close_refusal.clone().or(busy);
    let entry = ui
        .add_enabled(close.is_none(), egui::Button::new(CLOSE_SIFT_INDEX))
        .on_disabled_hover_text(close.unwrap_or_default())
        .on_hover_text("Let go of the open index, leaving the file where it is.");
    if out.hit(row_id(id, "close_sift_index"), entry).clicked() {
        out.response.close_sift_index = Some(id);
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

    let source_indexed = node.recon().feature_indexes().is_some();
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
/// The image row's context menu: the two `Resect Image` entries, one per
/// correspondence source.
///
/// They share their greying rules, because what a resection needs of an image is
/// the same question whichever correspondences answer it.
///
/// Both are kept visible and greyed rather than hidden when unavailable: the
/// action exists on every image row, and an entry that vanishes reads as an
/// action that was never implemented. The hover text says which of the
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
             and keep the answer as a version of this reconstruction. Undo (Ctrl+Z) puts \
             the stored pose back.",
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
            "The same, with the 2D-3D pairs taken from a .matches file, which admits \
             points this reconstruction never assigned to the image.",
        );
    if out
        .hit(row_id(node, &format!("resect_matches_{index}")), matches)
        .clicked()
    {
        out.response.resect_image = Some((image, ResectFrom::Matches));
        ui.close();
    }

    ui.separator();
    // The hand, beside the two estimators: where a resection re-computes a
    // pose from correspondences, this hands the camera to the reviewer. It
    // enters camera view first, because the lock *is* camera view with the
    // camera coming along.
    let move_camera = ui.add(egui::Button::new("Move Camera")).on_hover_text(
        "Look through this image and take its camera in hand: every navigation \
         input moves it, and M or Enter keeps the pose as a version of this \
         reconstruction.",
    );
    if out
        .hit(row_id(node, &format!("move_camera_{index}")), move_camera)
        .clicked()
    {
        out.response.move_camera = Some(image);
        ui.close();
    }

    ui.separator();
    // No confirmation: this is an edit with a history behind it, and Undo is
    // the answer to a mis-click, as it is for the entries above.
    let delete = ui.add(egui::Button::new("Delete Image")).on_hover_text(
        "Remove this image from the reconstruction, with its observations and any \
         track left with none. Undo (Ctrl+Z) puts it back.",
    );
    if out
        .hit(row_id(node, &format!("delete_image_{index}")), delete)
        .clicked()
    {
        out.response.delete_image = Some(image);
        ui.close();
    }
}
