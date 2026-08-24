// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The identity line — which node, which camera, which model, how big, how
//! many images — and the `Copy ▾` menu that gets the numbers out.
//!
//! The reconstruction's name is in the line because several nodes can be loaded
//! at once and a [`crate::scene::CameraRef`] carries a `ReconId`; without it
//! the panel would be ambiguous in exactly the session where it matters.
//!
//! A viewer whose numbers cannot leave it makes users retype them, so the menu
//! offers the parameter table twice (as text and as JSON), `K`, and — when the
//! extrinsics block is showing — the pose matrix in whichever frame that block
//! is showing it in.

use sfmtool_core::camera::CameraIntrinsics;

use super::extrinsics::Pose;
use super::format;
use crate::scene::SceneNode;

/// `kerry_park · Camera #0 · OPENCV_FISHEYE · 480×480 · 26 images    [Copy ▾]`
pub(super) fn show_header(
    ui: &mut egui::Ui,
    node: &SceneNode,
    index: usize,
    camera: &CameraIntrinsics,
    pose: Option<&Pose>,
) {
    let uses = node
        .recon
        .images
        .iter()
        .filter(|image| image.camera_index as usize == index)
        .count();
    let images = if uses == 1 {
        "1 image".to_string()
    } else {
        format!("{uses} images")
    };

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(&node.label).strong());
        ui.label("·");
        ui.label(egui::RichText::new(format!("Camera #{index}")).strong());
        ui.label("·");
        ui.label(camera.model.model_name());
        // The beta note gets its own label rather than riding the model name,
        // because egui hangs a tooltip off a whole *widget*: a sub-span of one
        // label has nowhere to put one (which is why the Scene panel's camera
        // row, being a single button, appends the note to the row tooltip
        // instead). Here the header is a run of separate labels, so `(beta)`
        // can be a widget in its own right and carry the note directly.
        if let Some(note) = camera.model.beta_note() {
            ui.label(egui::RichText::new("(beta)").weak())
                .on_hover_text(note);
        }
        ui.label("·");
        ui.label(format!("{}×{}", camera.width, camera.height));
        ui.label("·");
        ui.label(images);

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            show_copy_menu(ui, camera, pose);
        });
    });
}

/// The `Copy ▾` menu.
fn show_copy_menu(ui: &mut egui::Ui, camera: &CameraIntrinsics, pose: Option<&Pose>) {
    ui.menu_button("Copy", |ui| {
        if ui.button("Parameters (text)").clicked() {
            ui.ctx().copy_text(parameters_text(camera));
            ui.close();
        }
        if ui.button("Parameters (JSON)").clicked() {
            ui.ctx().copy_text(parameters_json(camera));
            ui.close();
        }
        if ui.button("K matrix").clicked() {
            let k = camera.intrinsic_matrix();
            let rows: Vec<Vec<f64>> = (0..3)
                .map(|r| (0..3).map(|c| k[(r, c)]).collect())
                .collect();
            ui.ctx().copy_text(format::matrix_text(&rows));
            ui.close();
        }
        // Offered only while the extrinsics block is on screen, so the menu
        // never hands out a pose the panel is not showing.
        if let Some(pose) = pose {
            if ui.button("Pose matrix").clicked() {
                ui.ctx().copy_text(format::matrix_text(&pose.pose_matrix()));
                ui.close();
            }
        }
    });
}

/// The parameter table as plain text — one `name value` per line, in
/// declaration order and to the same six decimals the table shows, so a paste
/// lines up with `sfm inspect` output beside it.
pub(super) fn parameters_text(camera: &CameraIntrinsics) -> String {
    let parameters = camera.parameters();
    let width = parameters
        .iter()
        .map(|(name, _)| name.chars().count())
        .max()
        .unwrap_or(0);
    parameters
        .iter()
        .map(|(name, value)| {
            let padded = format::value(*value);
            format!("{name:<width$}{padded}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The same table as JSON, at full precision.
///
/// Written out rather than serialized: the parameter order is the point, and
/// the values want every digit they have — a pasted `k1` rounded to six places
/// is a different lens. Every key here is a model or parameter name, which the
/// registry constrains to `[A-Za-z0-9_]`, so nothing needs escaping.
pub(super) fn parameters_json(camera: &CameraIntrinsics) -> String {
    let parameters = camera
        .parameters()
        .iter()
        .map(|(name, value)| format!("    \"{name}\": {value}"))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "{{\n  \"model\": \"{}\",\n  \"width\": {},\n  \"height\": {},\n  \"parameters\": {{\n{}\n  }}\n}}",
        camera.model.model_name(),
        camera.width,
        camera.height,
        parameters
    )
}
