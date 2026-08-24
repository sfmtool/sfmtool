// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The three tables: the model's stored parameters, what they mean, and `K`.
//!
//! The parameter table is in `parameter_names()` order — declaration order,
//! which is the order `sfm inspect` prints and which a `BTreeMap` cannot offer:
//! lexicographic order separates related terms and puts `bspline_c10` before
//! `bspline_c2`. The whole point of the table is that it can be diffed against
//! the CLI, so the order and the six decimals are both load-bearing.

use sfmtool_core::camera::CameraIntrinsics;

use super::derived::Derived;
use super::format;

/// Table 1: the model's parameters, in declaration order.
pub(super) fn show_parameters(ui: &mut egui::Ui, camera: &CameraIntrinsics) {
    ui.label(egui::RichText::new("Parameters").strong());
    egui::Grid::new("intrinsics_parameters")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .striped(true)
        .show(ui, |ui| {
            for (name, value) in camera.parameters() {
                ui.monospace(name.as_ref());
                ui.monospace(format::value(value));
                ui.end_row();
            }
        });
}

/// Table 2: what the parameters *mean*.
pub(super) fn show_derived(ui: &mut egui::Ui, camera: &CameraIntrinsics, derived: &Derived) {
    ui.label(egui::RichText::new("Derived").strong());
    egui::Grid::new("intrinsics_derived")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .striped(true)
        .show(ui, |ui| {
            let (fx, fy) = camera.focal_lengths();
            let unit = focal_unit(camera);
            ui.label("fx, fy");
            ui.monospace(format!("{fx:.3}, {fy:.3} {unit}"))
                .on_hover_text(focal_unit_note(camera));
            ui.end_row();

            // Hidden for a model that carries a single focal length: there is
            // no aspect ratio to report when the file cannot express one, and
            // a row reading 1.0000 for every SIMPLE_RADIAL in the world would
            // be noise.
            if has_two_focal_lengths(camera) && fx != 0.0 {
                ui.label("aspect fy/fx");
                ui.monospace(format!("{:.4}", fy / fx));
                ui.end_row();
            }

            let (cx, cy) = camera.principal_point();
            let (dx, dy) = (
                cx - f64::from(camera.width) / 2.0,
                cy - f64::from(camera.height) / 2.0,
            );
            let half_diagonal = f64::from(camera.width).hypot(f64::from(camera.height)) / 2.0;
            ui.label("principal point offset");
            let percent = if half_diagonal > 0.0 {
                format!(
                    " · {:.2}% of half-diagonal",
                    100.0 * dx.hypot(dy) / half_diagonal
                )
            } else {
                String::new()
            };
            ui.monospace(format!("({dx:+.3}, {dy:+.3}) px{percent}"))
                .on_hover_text(
                    "From the image centre (w/2, h/2), which is generally not the principal point",
                );
            ui.end_row();

            if let Some(fov) = &derived.fov {
                for (label, value) in [
                    ("horizontal FOV", fov.horizontal),
                    ("vertical FOV", fov.vertical),
                    ("diagonal FOV", fov.diagonal),
                ] {
                    ui.label(label);
                    ui.monospace(format!("{value:.1}°"));
                    ui.end_row();
                }
                ui.label("max off-axis angle");
                ui.monospace(format!("{:.1}°", fov.max_off_axis))
                    .on_hover_text(
                        "The largest incidence angle over the four image corners — a half-angle, \
                         so roughly half the diagonal span. For an equirectangular panorama the \
                         corners are the poles, so it reads 90° whatever the panorama covers.",
                    );
                ui.end_row();
            }

            if let Some(mm) = derived.equiv_35mm {
                ui.label("35 mm equivalent");
                ui.monospace(format!("{mm:.1} mm")).on_hover_text(
                    "f_px · 43.267 / diagonal_px — sensor-independent by construction, since \
                         both are in pixels. Undefined for a model whose focal length is pixels \
                         per radian.",
                );
                ui.end_row();
            }

            ui.label("distortion");
            match &derived.max_distortion {
                Some(extent) => match extent.limit_deg {
                    Some(limit) => {
                        let (dropped, total) = extent.excluded;
                        ui.monospace(format!(
                            "yes — max {:.1} px inside {limit:.1}°",
                            extent.max_px
                        ))
                        .on_hover_text(format!(
                            "The largest |model − ideal| displacement over a grid of the image, \
                             taken inside {limit:.1}° off-axis. Past that angle this model's own \
                             inverse stops inverting its distortion polynomial and slews toward \
                             the identity ray, so neither map describes the lens any more — \
                             {dropped} of the grid's {total} nodes look further out than that \
                             and are excluded. For a circular fisheye those nodes are the black \
                             corners outside the lens's image circle.",
                        ));
                    }
                    None => {
                        ui.monospace(format!("yes — max {:.1} px over the image", extent.max_px))
                            .on_hover_text(
                                "The largest |model − ideal| displacement over a grid of the \
                                 image. This model is trustworthy at every angle its frame \
                                 reaches, so the whole rectangle counts.",
                            );
                    }
                },
                None => {
                    ui.monospace("none").on_hover_text(
                        "Every distortion coefficient is zero, so this model is exactly its own \
                         ideal map.",
                    );
                }
            }
            ui.end_row();
        });
}

/// Table 3: `K`, and the two sentences that keep it from being pasted into the
/// wrong frame.
pub(super) fn show_k(ui: &mut egui::Ui, camera: &CameraIntrinsics) {
    ui.label(egui::RichText::new("K").strong());
    let k = camera.intrinsic_matrix();
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|r| (0..3).map(|c| k[(r, c)]).collect())
        .collect();
    format::matrix_grid(ui, "intrinsics_k", &rows);
    ui.label(
        egui::RichText::new(
            "K is in the optical frame (+Z forward, +Y down). Stored poses are canonical \
             (−Z forward, +Y up), so the projection matrix is P = K · S · [R|t] with \
             S = diag(1, −1, −1) — not K · [R|t].",
        )
        .weak()
        .small(),
    );
}

/// `px`, or `px/rad` for the models whose focal length is pixels per radian.
///
/// Mislabelling this is the single easiest way to make a fisheye's focal length
/// look absurd: 129 px would be a catastrophic lens and 129 px/rad is a 200°
/// fisheye on a 480-pixel frame.
pub(super) fn focal_unit(camera: &CameraIntrinsics) -> &'static str {
    if camera.model.needs_ray_path() {
        "px/rad"
    } else {
        "px"
    }
}

/// The tooltip behind [`focal_unit`].
fn focal_unit_note(camera: &CameraIntrinsics) -> &'static str {
    if camera.model.needs_ray_path() {
        "This model maps an incidence angle to a radius, so its focal length is pixels per radian."
    } else {
        "This model maps an image-plane coordinate to a pixel, so its focal length is pixels."
    }
}

/// Whether the model carries `fx` and `fy` separately.
///
/// Read off `parameter_names()` rather than compared numerically: a two-focal
/// model whose two values happen to be equal still has an aspect ratio the file
/// can express, and a single-focal model never does.
fn has_two_focal_lengths(camera: &CameraIntrinsics) -> bool {
    camera
        .model
        .parameter_names()
        .iter()
        .any(|name| name == "focal_length_y")
}
