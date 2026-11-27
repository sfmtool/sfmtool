// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Measurements group: one row per measurement the version on screen
//! stores, after the Points group and before the Bench groups.
//!
//! ```text
//! ▾ Measurements (4)
//!     Distance #0  ear to ear      12 cm  → 11.94 cm  −0.5%
//!     Distance #2  paw width      4.2 cm  ⚠ endpoint B: no patch holds the pixel
//!     Distance #3                   8 cm  → 0.317 su
//! ```
//!
//! The rows sit directly under the group while distance is the one kind. Each
//! row's reading is the node's cached one ([`crate::measurements::readings`]),
//! read in the file's own coordinates.

use eframe::egui;

use super::{row_id, NodeContext, TreeOutput, ROW_HEIGHT};
use crate::measurements::{self, MeasurementRef};
use crate::scene::SceneNode;

/// `[▸] Measurements (4)` and, expanded, one row per measurement.
///
/// Absent on a node that stores none. No eye: a measurement is drawn while it
/// is selected, or while a view's *Show all measurements* is on, and the
/// node's own eye hides them all.
///
/// A click on a row selects the measurement and nothing else, as a click on a
/// Bench row does. A double-click puts it on the bench and focuses it (or
/// focuses its item where it is on the bench already), raises Measurement
/// View beside the tree and frames the measurement's finite endpoints in the
/// 3D view (`app::apply_measurement_open`). A right-click selects nothing and
/// opens the row's menu: *Edit on Bench*, which is the double-click, and
/// *Delete*, greyed while the node is busy. The selected measurement's row is
/// drawn selected.
pub(super) fn show_measurements_group(
    ui: &mut egui::Ui,
    node: &SceneNode,
    ctx: &NodeContext,
    out: &mut TreeOutput,
) {
    let recon = node.recon();
    let rows = measurements::distance_rows(node.edited().measurements());
    if rows.is_empty() {
        return;
    }
    let id = node.id;
    let readings = measurements::readings(node);
    let unit = recon.metadata.world_space_unit.as_deref();
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        row_id(id, "measurements"),
        true,
    );
    let header = state.show_header(ui, |ui| {
        ui.set_height(ROW_HEIGHT);
        let title = ui.add(
            egui::Label::new(format!(
                "Measurements ({})",
                measurements::measurement_count(node.edited().measurements())
            ))
            .selectable(false),
        );
        out.hit(row_id(id, "measurements_header"), title);
    });
    header.body(|ui| {
        for (index, row) in rows.iter().enumerate() {
            let measurement = MeasurementRef::distance(id, index);
            let Some(resolved) = readings.distance.get(index) else {
                continue;
            };
            let text = measurements::row_text(measurement, row, resolved, unit);
            let selected = ctx.selected_measurement == Some(measurement);
            let response = ui
                .push_id(("measurement", index), |ui| {
                    ui.add(egui::Button::selectable(selected, text))
                })
                .inner
                .on_hover_text(row_hover(recon, row, resolved, unit));
            let response = out.hit(
                row_id(id, &format!("measurement_distance_{index}")),
                response,
            );
            if response.clicked() {
                out.response.select_measurement = Some(measurement);
            }
            if response.double_clicked() {
                out.response.open_measurement = Some(measurement);
            }
            crate::context_menu::on_secondary_click(&response).show(|ui| {
                let edit = ui
                    .button(EDIT_ON_BENCH_LABEL)
                    .on_hover_text("Put this measurement on the bench and edit it");
                let key = format!("measurement_distance_{index}_edit");
                if out.hit(row_id(id, &key), edit).clicked() {
                    out.response.open_measurement = Some(measurement);
                    ui.close();
                }
                let refusal = out.edit_refusal(id).map(str::to_string);
                let delete = ui.add_enabled(refusal.is_none(), egui::Button::new("Delete"));
                let delete = match refusal {
                    Some(why) => delete.on_disabled_hover_text(why),
                    None => delete.on_hover_text(
                        "Delete this measurement from the reconstruction; each row after it \
                         takes the index one lower",
                    ),
                };
                let key = format!("measurement_distance_{index}_delete");
                if out.hit(row_id(id, &key), delete).clicked() {
                    out.response.delete_measurement = Some(measurement);
                    ui.close();
                }
            });
        }
    });
}

/// A row's hover text: the endpoints' images and pixels, then the refusal's
/// whole sentence where an endpoint has none, then the gestures.
fn row_hover(
    recon: &sfmtool_core::SfmrReconstruction,
    row: &measurements::DistanceRow,
    resolved: &sfmtool_core::reconstruction::measurement_endpoint::ResolvedDistanceMeasurement,
    unit: Option<&str>,
) -> String {
    let mut lines = Vec::new();
    for side in measurements::EndpointSide::BOTH {
        let Some(endpoint) = row.endpoint(side) else {
            lines.push(format!("Endpoint {} is not placed", side.letter()));
            continue;
        };
        let [x, y] = endpoint.pixel;
        lines.push(format!(
            "Endpoint {}: {} at ({x:.1}, {y:.1})",
            side.letter(),
            measurements::image_name(recon, endpoint.image)
        ));
        if let Err(refusal) = measurements::endpoint(resolved, side) {
            lines.push(format!("  No position: {refusal}"));
        }
    }
    if unit.is_none() {
        lines.push("su: the reconstruction's own scene units".to_string());
    }
    lines
        .push("Click to select, double-click to edit on the bench in Measurement View".to_string());
    lines.join("\n")
}

/// The row menu's entry that puts the measurement on the bench.
pub(crate) const EDIT_ON_BENCH_LABEL: &str = "Edit on Bench";
