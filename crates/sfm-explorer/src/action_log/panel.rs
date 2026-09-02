// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Action Log panel: a toolbar and a scrolling, terminal-style list.
//!
//! The list is virtualized on a **uniform row height**
//! ([`egui::ScrollArea::show_rows`], as the Scene panel's image list is), which
//! is what makes ten thousand entries free to scroll — and what forbids
//! wrapping, so a text wider than the panel is truncated and shown whole in the
//! row's tooltip instead.

use super::{ActionLog, Actor, Entry, Kind};

/// Width of the time column: `00:00:00` in the monospace font plus a clear gap
/// before the actor column, at any reasonable text size.
const TIME_WIDTH: f32 = 76.0;

/// Width of the actor column: `Viewer` plus the same clear gap before the text.
const ACTOR_WIDTH: f32 = 62.0;

/// The panel body. Draws the toolbar and the virtualized list into `ui`.
pub(crate) fn show(ui: &mut egui::Ui, log: &mut ActionLog) {
    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);

    let mut clear = false;
    let mut latest = false;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(match log.len() {
                1 => "1 entry".to_string(),
                n => format!("{n} entries"),
            })
            .weak(),
        );
        if log.dropped() > 0 {
            ui.label(egui::RichText::new(format!("({} dropped)", log.dropped())).weak())
                .on_hover_text(format!(
                    "The log keeps the most recent {} entries.",
                    ActionLog::CAPACITY
                ));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("Clear")
                .on_hover_text("Empty the log. This also clears the viewport status line.")
                .clicked()
            {
                clear = true;
            }
            if ui
                .button("Copy")
                .on_hover_text("Put the whole log on the clipboard as plain text")
                .clicked()
            {
                ui.ctx().copy_text(log.to_clipboard_text());
            }
            if ui
                .button("Latest")
                .on_hover_text("Scroll to the newest entry")
                .clicked()
            {
                latest = true;
            }
        });
    });
    ui.separator();

    if clear {
        log.clear();
    }

    let total = log.len();
    let mut area = egui::ScrollArea::vertical()
        .id_salt("action_log_list")
        .auto_shrink([false, false])
        // The list follows the tail while it is *at* the tail and holds still
        // the moment the user scrolls up, so an entry can be read while an
        // agent keeps working. The panel keeps no follow state of its own.
        .stick_to_bottom(true);
    if latest {
        // Clamped by the scroll area, so an offset past the end simply lands on
        // it — which is the whole of what "Latest" has to do.
        area = area.vertical_scroll_offset(total as f32 * row_height);
    }
    area.show_rows(ui, row_height, total, |ui, range| {
        ui.spacing_mut().item_spacing.y = 0.0;
        for index in range {
            let Some(entry) = log.get(index) else {
                continue;
            };
            show_row(ui, log, entry, row_height);
        }
    });
}

/// One row: the local time of day, the actor, and the text.
fn show_row(ui: &mut egui::Ui, log: &ActionLog, entry: &Entry, row_height: f32) {
    // Colour carries the entry's shape so the columns stay plain: an MCP row is
    // distinguished by its actor rather than by a prefix in the text, because
    // the text of an action never depends on who took it.
    let query = matches!(entry.kind, Kind::Query(_));
    let weak = ui.visuals().weak_text_color();
    let actor_color = match entry.actor {
        _ if query => weak,
        Actor::Mcp => ui.visuals().hyperlink_color,
        Actor::User => ui.visuals().text_color(),
        Actor::Viewer => weak,
    };
    let text_color = if entry.failed {
        ui.visuals().error_fg_color
    } else if query {
        weak
    } else {
        ui.visuals().text_color()
    };

    let tooltip = format!(
        "{}\n{}",
        log.format(entry.at, "%Y-%m-%d %H:%M:%S %:z"),
        entry.text
    );
    ui.horizontal(|ui| {
        ui.set_height(row_height);
        ui.spacing_mut().item_spacing.x = 0.0;
        monospace(ui, TIME_WIDTH, &log.format(entry.at, "%H:%M:%S"), weak);
        monospace(ui, ACTOR_WIDTH, entry.actor.label(), actor_color);
        ui.add(
            egui::Label::new(
                egui::RichText::new(&entry.text)
                    .monospace()
                    .color(text_color),
            )
            .truncate()
            .selectable(false),
        );
    })
    .response
    .on_hover_text(tooltip);
}

/// One fixed-width monospace cell, so the three columns line up down the list
/// however wide their contents are.
fn monospace(ui: &mut egui::Ui, width: f32, text: &str, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(width, ui.available_height()),
        egui::Sense::hover(),
    );
    ui.painter().text(
        rect.left_center(),
        egui::Align2::LEFT_CENTER,
        text,
        egui::TextStyle::Monospace.resolve(ui.style()),
        color,
    );
}
