// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Action Log panel: a toolbar and a scrolling, terminal-style list.
//!
//! The list is virtualized on a **uniform row height**
//! ([`egui::ScrollArea::show_rows`], as the Scene panel's image list is), which
//! is what makes ten thousand entries free to scroll — and what forbids
//! wrapping, so a text wider than the panel is truncated and shown whole in the
//! row's tooltip instead.
//!
//! ## Expansion inserts rows, it does not make one tall
//!
//! An entry that carries detail expands into the rows under it rather than
//! growing, because a variable row height would cost the virtualization the
//! uniform height it is built on: `show_rows` would have to lay every entry out
//! to know where the visible ones start. So the rows of the list are not the
//! entries of the log, and [`rows`] is the mapping between them: a small table
//! built once a frame, one line per expanded entry still held, saying where it
//! expands and how many rows its detail adds. A lookup binary-searches that
//! table and falls through to the entry at the remaining offset. Expansions are
//! a handful at most, so the table is cheaper to build than the layout it
//! avoids.

use sfmtool_core::progress::Level;

use super::{ActionLog, Actor, Entry, Kind};
use crate::progress::Detail;

/// Width of the toggle column: one glyph, plus the gap before the time.
const TOGGLE_WIDTH: f32 = 14.0;

/// Width of the time column: `00:00:00` in the monospace font plus a clear gap
/// before the actor column, at any reasonable text size.
const TIME_WIDTH: f32 = 76.0;

/// Width of the actor column: `Viewer` plus the same clear gap before the text.
const ACTOR_WIDTH: f32 = 62.0;

/// Width of the duration column: `<1 ms` through `99.99 s` in the monospace
/// font, plus the gap before the text.
const TOOK_WIDTH: f32 = 64.0;

/// The panel body. Draws the toolbar and the virtualized list into `ui`.
pub(crate) fn show(ui: &mut egui::Ui, log: &mut ActionLog) {
    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
    let space = space_width(ui);

    let mut clear = false;
    let mut latest = false;
    let mut detailed = log.detailed_timing();
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
            // Beside the buttons because this is where the detail is read,
            // which is where somebody decides they want more of it.
            ui.checkbox(&mut detailed, "Detailed timing").on_hover_text(
                "Record the finer stages inside each operation.\n\
                     Takes effect on the next operation: nothing already \
                     recorded is re-timed.",
            );
        });
    });
    ui.separator();

    // Before the rows are counted: both of these change what there is to draw.
    if detailed != log.detailed_timing() {
        log.set_detailed_timing(detailed);
    }
    if clear {
        log.clear();
    }

    let rows = rows(log);
    let mut area = egui::ScrollArea::vertical()
        .id_salt("action_log_list")
        .auto_shrink([false, false])
        // The list follows the tail while it is *at* the tail and holds still
        // the moment the user scrolls up, so an entry can be read while an
        // agent keeps working. The panel keeps no follow state of its own.
        .stick_to_bottom(true);
    if latest {
        // Clamped by the scroll area, so an offset past the end simply lands on
        // it — which is the whole of what "Latest" has to do. The count is the
        // expanded one, since that is what the area is scrolling through.
        area = area.vertical_scroll_offset(rows.total as f32 * row_height);
    }
    let mut toggled = None;
    area.show_rows(ui, row_height, rows.total, |ui, range| {
        ui.spacing_mut().item_spacing.y = 0.0;
        for row in range {
            match rows.at(row) {
                Row::Entry(index) => {
                    let Some(entry) = log.get(index) else {
                        continue;
                    };
                    if show_row(ui, log, entry, row_height) {
                        toggled = Some(entry.revision);
                    }
                }
                Row::Detail(index, within) => {
                    let Some(entry) = log.get(index) else {
                        continue;
                    };
                    show_detail_row(ui, entry, within, row_height, space);
                }
            }
        }
    });
    if let Some(revision) = toggled {
        log.toggle_expanded(revision);
    }
}

// ── The rows of the list, which are not the entries of the log ──────────

/// One expanded entry, and where its detail lands in the list.
struct Expansion {
    /// Which entry of the log it is.
    index: usize,
    /// The row the entry itself is drawn at.
    start: usize,
    /// How many rows its detail adds under it. Never zero: an entry with
    /// nothing to show is not in the table at all.
    extra: usize,
    /// Rows added by every expansion before this one, which is what turns a
    /// row past this expansion back into an entry index.
    before: usize,
}

/// What a row of the list is: an entry, or one row of an expanded entry's
/// detail.
enum Row {
    /// The entry at this index of the log.
    Entry(usize),
    /// A detail row of the entry at this index: the offset within its
    /// breakdown, one past the end being the `elsewhere` line.
    Detail(usize, usize),
}

/// The mapping from the list's rows to the log's entries.
struct Rows {
    /// The expanded entries, ordered by where they sit.
    table: Vec<Expansion>,
    /// How many rows the list has.
    total: usize,
}

impl Rows {
    /// What `row` is. Binary search over the expansions, then arithmetic.
    fn at(&self, row: usize) -> Row {
        let after = self.table.partition_point(|open| open.start <= row);
        let Some(open) = after.checked_sub(1).map(|i| &self.table[i]) else {
            // Before the first expansion, a row is its own entry.
            return Row::Entry(row);
        };
        if row == open.start {
            Row::Entry(open.index)
        } else if row <= open.start + open.extra {
            Row::Detail(open.index, row - open.start - 1)
        } else {
            Row::Entry(row - (open.before + open.extra))
        }
    }
}

/// The list's rows for the log as it stands.
///
/// Built from the expansion set rather than by walking the entries, so the cost
/// is one binary search per expanded revision rather than a pass over ten
/// thousand rows a frame. A revision whose entry has dropped off the front, and
/// an entry expanded before it turned out to carry nothing, are both left out:
/// an expansion that adds no rows is not an expansion.
fn rows(log: &ActionLog) -> Rows {
    let mut indexes: Vec<usize> = log
        .expanded_revisions()
        .filter_map(|revision| log.index_of(revision))
        .filter(|index| log.get(*index).is_some_and(|entry| detail_rows(entry) > 0))
        .collect();
    indexes.sort_unstable();
    let mut table = Vec::with_capacity(indexes.len());
    let mut before = 0;
    for index in indexes {
        let extra = log.get(index).map_or(0, detail_rows);
        table.push(Expansion {
            index,
            start: index + before,
            extra,
            before,
        });
        before += extra;
    }
    Rows {
        table,
        total: log.len() + before,
    }
}

/// How many rows an entry's detail draws: its events, and the `elsewhere` line
/// that makes them add up.
pub(super) fn detail_rows(entry: &Entry) -> usize {
    entry.detail.len() + usize::from(ActionLog::elsewhere(entry).is_some())
}

/// How many rows the list has, entries and expanded detail together. For the
/// tests, which hold the property that expanding adds exactly one row per
/// event.
#[cfg(test)]
pub(super) fn row_count(log: &ActionLog) -> usize {
    rows(log).total
}

// ── The rows themselves ─────────────────────────────────────────────────

/// One row: the toggle, the local time of day, the actor, the cost and the
/// text. Returns whether the click that expands it happened.
fn show_row(ui: &mut egui::Ui, log: &ActionLog, entry: &Entry, row_height: f32) -> bool {
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

    // The kind rides on the tooltip rather than in a column of its own: it is
    // what a row is *about*, which the text usually says already, and the one
    // time it is worth asking is the one time a hover costs nothing.
    let tooltip = format!(
        "{}  {}\n{}",
        log.format(entry.at, "%Y-%m-%d %H:%M:%S %:z"),
        entry.kind.label(),
        entry.text
    );
    let expandable = !entry.detail.is_empty();
    let toggle = match (expandable, log.is_expanded(entry.revision)) {
        (false, _) => "",
        (true, false) => "+",
        (true, true) => "-",
    };
    let row = ui.horizontal(|ui| {
        ui.set_height(row_height);
        ui.spacing_mut().item_spacing.x = 0.0;
        let toggle_rect = monospace(ui, TOGGLE_WIDTH, toggle, weak);
        let time_rect = monospace(ui, TIME_WIDTH, &log.format(entry.at, "%H:%M:%S"), weak);
        monospace(ui, ACTOR_WIDTH, entry.actor.label(), actor_color);
        // Right-aligned, so the slow rows stand out of a column of small ones
        // without anyone having to read the numbers. Weak, because it is the
        // one column that is about the viewer rather than about the action.
        monospace_right(
            ui,
            TOOK_WIDTH,
            &entry.took.map(ActionLog::format_took).unwrap_or_default(),
            weak,
        );
        ui.add(
            egui::Label::new(
                egui::RichText::new(&entry.text)
                    .monospace()
                    .color(text_color),
            )
            .truncate()
            .selectable(false),
        );
        toggle_rect.union(time_rect)
    });
    // The *row* is what senses the click, and where the pointer was is what
    // says whether it was on the toggle or the time. A cell that sensed clicks
    // of its own would take the hover off the row and the tooltip with it:
    // egui hovers a non-interactive widget only when it sits above the topmost
    // interactive one, and a row's own widget is registered when the row
    // opens, before any of the cells inside it.
    let response = if expandable {
        row.response.interact(egui::Sense::click())
    } else {
        row.response
    };
    let clicked = response.clicked()
        && response
            .interact_pointer_pos()
            .is_some_and(|pos| row.inner.contains(pos));
    response.on_hover_text(tooltip);
    clicked
}

/// One row of an expanded entry's breakdown: a phase, a message, or the
/// `elsewhere` line that closes it.
///
/// The toggle, time and actor columns are blank here, and the CPU figure is
/// drawn right-aligned across them: it wants a column of its own, and the
/// alternative is a gutter between the cost and the text that every row in the
/// list would pay for a figure that only a kernel reporting thread-summed time
/// ever fills.
fn show_detail_row(ui: &mut egui::Ui, entry: &Entry, within: usize, row_height: f32, space: f32) {
    let row = detail_row(entry, within);
    let weak = ui.visuals().weak_text_color();
    // The rule is painted along the top of the overhead row's own rect rather
    // than given a row of its own, so the list keeps the uniform row height its
    // virtualization depends on.
    let rule = row.rules_above;
    let marker_color = if row.warn {
        ui.visuals().error_fg_color
    } else {
        ui.visuals().text_color()
    };
    let drawn = ui
        .horizontal(|ui| {
            ui.set_height(row_height);
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.add_space(TOGGLE_WIDTH);
            monospace_right(ui, TIME_WIDTH + ACTOR_WIDTH, &row.cpu, weak);
            monospace_right(ui, TOOK_WIDTH, &row.cost, weak);
            // The indent and the marker share one cell, so a phase name and a
            // message text at the same depth start at the same place and the
            // marker sits in the column before them, as it reads in a transcript.
            let (lead, glyphs) = if row.marker.is_empty() {
                (String::new(), row.indent)
            } else {
                (
                    format!("{}{}", " ".repeat(row.indent), row.marker),
                    row.indent + 2,
                )
            };
            monospace(ui, glyphs as f32 * space, &lead, marker_color);
            ui.add(
                egui::Label::new(
                    egui::RichText::new(&row.text)
                        .monospace()
                        .color(ui.visuals().text_color()),
                )
                .truncate()
                .selectable(false),
            );
        })
        .response
        .on_hover_text(&row.text);
    if rule {
        let rect = drawn.rect;
        ui.painter().hline(
            rect.x_range(),
            rect.top(),
            ui.visuals().widgets.noninteractive.bg_stroke,
        );
    }
}

/// What one row of an expanded entry says.
///
/// Built here rather than at each of the two places that draw it, so that the
/// panel and the clipboard export cannot disagree about what a row reads.
pub(super) struct DetailRow {
    /// The cost column: `412 ms`, or `--` for a stage that cost nothing worth
    /// printing.
    pub cost: String,
    /// The CPU column, empty unless the stage reported thread-summed time.
    pub cpu: String,
    /// Spaces before the marker: two per level of nesting.
    pub indent: usize,
    /// What marks a message, empty for a phase.
    pub marker: &'static str,
    /// Whether the marker is a warning's, and so wants the error colour.
    pub warn: bool,
    /// Whether a rule is drawn above this row, which divides what the
    /// operation accounted for from the overhead of showing its result.
    pub rules_above: bool,
    /// The row's text, with neither the indent nor the marker in it.
    pub text: String,
}

/// Where an entry's own account ends and the frame's overhead begins, as an
/// index into its detail.
///
/// The overhead row and everything under it are appended when the frame is
/// charged, so they are a suffix, and the operation's own account is what comes
/// before them.
fn overhead_at(entry: &Entry) -> usize {
    entry
        .detail
        .iter()
        .position(|row| {
            matches!(
                row,
                Detail::Phase {
                    name: ActionLog::OVERHEAD,
                    ..
                }
            )
        })
        .unwrap_or(entry.detail.len())
}

/// The row `within` of `entry`'s breakdown.
///
/// `elsewhere` closes the operation's own account rather than the whole entry:
/// it is what makes the breakdown reconcile with the number in the entry's cost
/// column, so work nobody has named shows up as a gap rather than as silence,
/// and the work that put the result on the screen is not the operation's to
/// answer for. So the order is the operation's stages, then `elsewhere`, then
/// the overhead under a rule.
pub(super) fn detail_row(entry: &Entry, within: usize) -> DetailRow {
    let split = overhead_at(entry);
    let elsewhere = ActionLog::elsewhere(entry).is_some();
    let index = if within < split {
        Some(within)
    } else if within == split && elsewhere {
        None
    } else {
        Some(within - usize::from(elsewhere))
    };
    match index.and_then(|index| entry.detail.get(index)) {
        Some(Detail::Phase {
            name,
            depth,
            took,
            cpu,
            note,
            note_last,
            runs,
        }) => DetailRow {
            cost: ActionLog::format_took(*took),
            cpu: cpu
                .map(|cpu| format!("cpu {}", ActionLog::format_took(cpu)))
                .unwrap_or_default(),
            indent: 2 * usize::from(*depth),
            marker: "",
            warn: false,
            rules_above: *name == ActionLog::OVERHEAD,
            text: format!(
                "{name}{}{}",
                // A stage that ran once is drawn as it would have been anyway.
                if *runs > 1 {
                    format!(" x{runs}")
                } else {
                    String::new()
                },
                // `reused` is the useful one: it is how a reader tells a stage
                // that was skipped from one that was merely fast. A folded row
                // whose runs said different things shows both ends, since
                // neither end on its own is true of the row.
                match (note.as_deref(), note_last.as_deref()) {
                    (Some(first), Some(last)) => format!("  {first} ... {last}"),
                    (Some(only), None) => format!("  {only}"),
                    (None, _) => String::new(),
                },
            ),
        },
        Some(Detail::Message { level, depth, text }) => DetailRow {
            cost: String::new(),
            cpu: String::new(),
            indent: 2 * usize::from(*depth),
            marker: match level {
                Level::Info => "\u{2022}",
                Level::Warn => "!",
            },
            warn: matches!(level, Level::Warn),
            rules_above: false,
            text: text.clone(),
        },
        None => DetailRow {
            cost: ActionLog::format_took(ActionLog::elsewhere(entry).unwrap_or_default()),
            cpu: String::new(),
            indent: 0,
            marker: "",
            warn: false,
            rules_above: false,
            text: "elsewhere".to_string(),
        },
    }
}

/// The clipboard spelling of one detail row's text: its indent, its marker and
/// what follows, with the CPU figure after it since the clipboard has no
/// columns to give it.
pub(super) fn detail_text(row: &DetailRow) -> String {
    let marker = if row.marker.is_empty() {
        String::new()
    } else {
        format!("{} ", row.marker)
    };
    let cpu = if row.cpu.is_empty() {
        String::new()
    } else {
        format!("  {}", row.cpu)
    };
    format!("{}{marker}{}{cpu}", " ".repeat(row.indent), row.text)
}

// ── Cells ───────────────────────────────────────────────────────────────

/// The width of one space in the list's font, which is what an indent is
/// counted in.
fn space_width(ui: &egui::Ui) -> f32 {
    let font = egui::TextStyle::Monospace.resolve(ui.style());
    ui.ctx().fonts_mut(|fonts| fonts.glyph_width(&font, ' '))
}

/// The same cell, with its text against the right edge and a gap after it.
fn monospace_right(ui: &mut egui::Ui, width: f32, text: &str, color: egui::Color32) -> egui::Rect {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(width, ui.available_height()),
        egui::Sense::hover(),
    );
    ui.painter().text(
        rect.right_center() - egui::vec2(8.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        text,
        egui::TextStyle::Monospace.resolve(ui.style()),
        color,
    );
    rect
}

/// One fixed-width monospace cell, so the columns line up down the list
/// however wide their contents are.
fn monospace(ui: &mut egui::Ui, width: f32, text: &str, color: egui::Color32) -> egui::Rect {
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
    rect
}
