// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Edit History panel: the selected node's versions, oldest first, with the
//! cursor on one of them.
//!
//! See `specs/gui/edit-history.md` § "The Edit History panel". The panel reads
//! [`crate::document::History`] and decides nothing: a click reports the serial
//! it wants, and [`crate::state::AppState::jump_to_version`] is what moves the
//! cursor. A row whose value the budget released is listed and disabled, since
//! the history still knows what happened there while having nothing to show.

use crate::document::{Version, VersionSerial};
use crate::scene::ReconId;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// What the panel reports back to the dock.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EditHistoryResponse {
    /// The version a click asked for, and the node it belongs to.
    pub jump: Option<(ReconId, VersionSerial)>,
}

/// The mark on the row the cursor is on.
const CURSOR_MARK: &str = "\u{25b6}";

/// The mark on the row the file on disk holds.
const DISK_MARK: &str = "\u{25cf}";

/// The panel body.
pub(crate) fn show(ui: &mut egui::Ui, state: &AppState) -> EditHistoryResponse {
    let mut response = EditHistoryResponse::default();
    let Some(node) = crate::scene::selected_node(&state.scene, state.selected_recon) else {
        ui.centered_and_justified(|ui| {
            ui.label("No reconstruction loaded");
        });
        return response;
    };
    let history = &node.history;
    let cursor = history.cursor();
    let disk = history.disk_serial();

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(&node.label).strong());
        ui.label(
            egui::RichText::new(match history.versions().len() {
                1 => "1 version".to_string(),
                n => format!("{n} versions"),
            })
            .weak()
            .small(),
        );
    });
    ui.separator();

    // A node that has never been edited has one version, which is the file as
    // it was opened. Saying so is what keeps the panel from reading as broken.
    if history.versions().len() == 1 {
        ui.label(
            egui::RichText::new(format!(
                "{} has not been edited; its one version is the file as it was opened.",
                node.label
            ))
            .weak(),
        );
    }

    egui::ScrollArea::vertical()
        .id_salt("edit_history_versions")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new("edit_history_version_table")
                .num_columns(4)
                .striped(true)
                .show(ui, |ui| {
                    ui.label("");
                    ui.label(egui::RichText::new("Version").strong().small());
                    ui.label(egui::RichText::new("Description").strong().small());
                    ui.label(egui::RichText::new("Size").strong().small());
                    ui.end_row();
                    for (index, version) in history.versions().iter().enumerate() {
                        if show_version_row(
                            ui,
                            state,
                            version,
                            index == cursor,
                            version.serial == disk,
                        ) {
                            response.jump = Some((node.id, version.serial));
                        }
                    }
                });
        });
    response
}

/// Draw one selectable table row. Every cell participates in the same action,
/// so the compact serial column is as clickable as the description and size.
fn show_version_row(
    ui: &mut egui::Ui,
    state: &AppState,
    version: &Version,
    at_cursor: bool,
    at_disk: bool,
) -> bool {
    let released = version.value.is_none();
    let marks = format!(
        "{}{}",
        if at_cursor { CURSOR_MARK } else { " " },
        if at_disk { DISK_MARK } else { " " }
    );
    let enabled = !released && !at_cursor;
    let mut row = ui.add_enabled(enabled, egui::Button::selectable(at_cursor, marks));
    row = row.union(ui.add_enabled(
        enabled,
        egui::Button::selectable(at_cursor, version.serial.to_string()),
    ));
    row = row.union(ui.add_enabled(
        enabled,
        egui::Button::selectable(
            at_cursor,
            format!(
                "{}{}",
                version.label,
                if released { "  (released)" } else { "" },
            ),
        ),
    ));
    row = row.union(ui.add_enabled(
        enabled,
        egui::Button::selectable(at_cursor, format_bytes(version.unshared_bytes)),
    ));
    ui.end_row();

    let when = state.action_log.format(version.at, "%H:%M:%S");
    let hover = if released {
        format!(
            "{} at {when}. Its value was released to keep the history inside its memory budget, so there is nothing to go back to.",
            version.serial
        )
    } else if at_cursor {
        format!("{} at {when}. This is what the node shows.", version.serial)
    } else {
        format!("{} at {when}. Click to go here.", version.serial)
    };
    // A released row and the cursor's own row are disabled, and a disabled
    // widget answers a hover only through the disabled channel.
    let row = if released || at_cursor {
        row.on_disabled_hover_text(hover)
    } else {
        row.on_hover_text(hover)
    };
    row.clicked()
}

/// Bytes, as a row states them: three significant figures and a binary unit.
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if value < 10.0 {
        format!("{value:.2} {}", UNITS[unit])
    } else if value < 100.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.0} {}", UNITS[unit])
    }
}
