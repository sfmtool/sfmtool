// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The layer's toolbar controls: one checkbox and, behind it, a settings popup.
//!
//! The Image Detail toolbar row is already the widest thing in the panel, and
//! four more inline controls would wrap it at any reasonable panel width. So the
//! layer contributes exactly two widgets to that row — a checkbox and a gear —
//! and the sub-toggles live in a popup the gear opens, enabled only while the
//! checkbox is ticked.

use super::CameraLayer;
use crate::state::IntrinsicsDisplaySettings;
use sfmtool_core::camera::CameraIntrinsics;

/// The gear that opens the settings popup. Spelled once so the glyph test can
/// pin it: a glyph egui does not bundle renders as a replacement box rather
/// than failing, and nothing else would notice.
pub(crate) const GEAR: &str = "⚙";

/// Draw the layer's contribution to the Image Detail toolbar.
///
/// `layer` is the cached report for the camera currently on screen, or `None`
/// when no image is selected; it is what the popup's footer reads. Because the
/// toolbar is drawn above the image, the arrow scale it quotes is the one the
/// *previous* frame resolved — invisible at any frame rate, and the only
/// alternative would be laying the panel out twice.
pub(crate) fn show_intrinsics_controls(
    ui: &mut egui::Ui,
    settings: &mut IntrinsicsDisplaySettings,
    camera: Option<&CameraIntrinsics>,
    layer: Option<&CameraLayer>,
) {
    ui.separator();
    ui.checkbox(&mut settings.enabled, "Intrinsics");
    ui.add_enabled_ui(settings.enabled, |ui| {
        ui.menu_button(GEAR, |ui| settings_popup(ui, settings, camera, layer));
    });
}

/// The popup's body: the three sub-toggles, the two ladders, and a footer
/// stating what the arrows are measuring.
pub(super) fn settings_popup(
    ui: &mut egui::Ui,
    settings: &mut IntrinsicsDisplaySettings,
    camera: Option<&CameraIntrinsics>,
    layer: Option<&CameraLayer>,
) {
    ui.set_min_width(230.0);
    ui.checkbox(&mut settings.axes, "Axes");
    ui.checkbox(&mut settings.rings, "Iso-angle rings");

    // A model that is its own ideal map has an identically-zero field, so the
    // control never sits there inviting a click that would do nothing.
    let has_distortion = camera.is_some_and(|camera| camera.has_distortion());
    if !has_distortion {
        ui.add_enabled(false, egui::Label::new("No distortion"));
        return;
    }

    ui.horizontal(|ui| {
        ui.checkbox(&mut settings.distortion, "Distortion field");
        let auto_label = layer.map_or_else(
            || "Auto".to_owned(),
            |layer| format!("Auto ({TIMES}{})", scale_text(layer.auto_scale)),
        );
        egui::ComboBox::from_id_salt("intrinsics_distortion_scale")
            .selected_text(match settings.distortion_scale {
                Some(scale) => format!("{TIMES}{}", scale_text(scale)),
                None => auto_label.clone(),
            })
            .width(96.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut settings.distortion_scale, None, auto_label);
                for scale in IntrinsicsDisplaySettings::SCALE_LADDER {
                    ui.selectable_value(
                        &mut settings.distortion_scale,
                        Some(scale),
                        format!("{TIMES}{}", scale_text(scale)),
                    );
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label("Grid density");
        egui::ComboBox::from_id_salt("intrinsics_grid_cols")
            .selected_text(format!("{}", settings.grid_cols))
            .width(64.0)
            .show_ui(ui, |ui| {
                for cols in IntrinsicsDisplaySettings::GRID_LADDER {
                    ui.selectable_value(&mut settings.grid_cols, cols, format!("{cols}"));
                }
            });
    });

    let Some(layer) = layer else {
        return;
    };
    ui.separator();
    ui.label(format!("max displacement {:.1} px", layer.max_px));
    // The qualifier is the whole reason the number above is believable: on a
    // circular fisheye the unfiltered maximum is about the folded polynomial in
    // the black corners rather than about the lens.
    if let Some(limit) = layer.limit_deg {
        let (outside, total) = layer.extrapolated;
        if outside > 0 {
            ui.label(
                egui::RichText::new(format!(
                    "inside {limit:.1}°; {outside} of {total} nodes extrapolated"
                ))
                .weak(),
            );
        }
    }
}

/// U+00D7 MULTIPLICATION SIGN, the `×` of `×3`. Spelled once for the glyph test.
pub(crate) const TIMES: &str = "×";

/// An exaggeration without its decimal point: the ladder is whole numbers, and
/// `×3` is what the spec's legend says where `×3.0` would read as a measurement.
pub(crate) fn scale_text(scale: f32) -> String {
    if scale.fract() == 0.0 {
        format!("{}", scale as i64)
    } else {
        format!("{scale}")
    }
}
