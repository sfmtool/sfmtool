// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The identity line — which node, which camera, which model, how big, how
//! many images — the `Switch Model…` button that opens a camera model
//! proposal, and the `Copy ▾` menu that gets the numbers out.
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

/// `kerry_park · Camera #0 · OPENCV_FISHEYE · 480×480 · 26 images
/// [Switch Model…] [Copy ▾]`
///
/// Returns whether `Switch Model…` was clicked. The button is greyed, with
/// `refusal` as its hover text, where no proposal can be opened on the camera
/// (`AppState::switch_model_refusal`), or while this camera's proposal is
/// already open below it.
pub(super) fn show_header(
    ui: &mut egui::Ui,
    node: &SceneNode,
    index: usize,
    camera: &CameraIntrinsics,
    pose: Option<&Pose>,
    refusal: Option<&str>,
) -> bool {
    let mut switch = false;
    let uses = node
        .recon()
        .image_table
        .images
        .iter()
        .filter(|image| image.camera_index as usize == index)
        .count();
    let images = if uses == 1 {
        "1 image".to_string()
    } else {
        format!("{uses} images")
    };

    // The buttons are laid out first, from the right edge, and the identity
    // line takes the width they leave, wrapping between its parts rather than
    // running under them: a narrow panel puts "480×480 · 24 images" on a
    // second line instead of behind `Switch Model…`.
    ui.horizontal_top(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            show_copy_menu(ui, camera, pose);
            switch = ui
                .add_enabled(refusal.is_none(), egui::Button::new(SWITCH_MODEL))
                .on_disabled_hover_text(refusal.unwrap_or_default())
                .on_hover_text(
                    "Fit this camera to another camera model, or refit its spline to another                      coefficient count or domain end, and compare the two before applying.                      Apply switches the camera and retriangulates the tracks its images                      observe, as one version.",
                )
                .clicked();
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::Min).with_main_wrap(true),
                |ui| {
                    part(ui, egui::RichText::new(&node.label).strong());
                    part(ui, "·");
                    part(ui, egui::RichText::new(format!("Camera #{index}")).strong());
                    part(ui, "·");
                    part(ui, camera.model.model_name());
                    // The beta note gets its own label rather than riding the
                    // model name, because egui hangs a tooltip off a whole
                    // *widget*: a sub-span of one label has nowhere to put one
                    // (which is why the Scene panel's camera row, being a single
                    // button, appends the note to the row tooltip instead). Here
                    // the header is a run of separate labels, so `(beta)` can be
                    // a widget in its own right and carry the note directly.
                    if let Some(note) = camera.model.beta_note() {
                        part(ui, egui::RichText::new("(beta)").weak()).on_hover_text(note);
                    }
                    part(ui, "·");
                    part(ui, format!("{}×{}", camera.width, camera.height));
                    part(ui, "·");
                    part(ui, images);
                },
            );
        });
    });
    switch
}

/// One part of the identity line, kept on one line of its own: the line wraps
/// between its parts, never inside one.
fn part(ui: &mut egui::Ui, text: impl Into<egui::WidgetText>) -> egui::Response {
    ui.add(egui::Label::new(text).wrap_mode(egui::TextWrapMode::Extend))
}

/// What the header's button that opens a camera model proposal is called.
pub(super) const SWITCH_MODEL: &str = "Switch Model…";

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
            let padded = format::parameter(name, *value);
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
