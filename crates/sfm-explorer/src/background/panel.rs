// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Background panel: what is running, on which node, and what it has spent
//! its time on so far.
//!
//! See `specs/drafts/background-process-panel.md`, "What the user sees". The
//! panel decides nothing: it reads [`AppState::background`] and the collector
//! that process shares with its worker, and the one button it has calls
//! [`AppState::cancel_background`].
//!
//! ## Two forms, and neither of them is blank
//!
//! Running, it draws the operation, the node, a bar or a spinner, the elapsed,
//! a Cancel button and the live phase table. Idle, it draws the last operation
//! of the session greyed, with its phases under a toggle, because the question
//! a reader brings here after a long solve is "what did that cost" and the
//! answer is still in reach. A session that has run nothing says
//! `Nothing running` and no more.
//!
//! ## The phase table is the Action Log's rows
//!
//! A row here is [`crate::action_log::detail_row`] over a
//! [`Breakdown::running`], which is the function the Action Log draws a
//! finished entry's breakdown with. The panel and the entry are two views of
//! one collector, and a second spelling of a row would be a second thing that
//! can be wrong.

use std::time::Duration;

use crate::action_log::{detail_row, detail_text, ActionLog, Breakdown};
use crate::progress::{Collector, Count, Detail, Live};
use crate::state::AppState;

/// Marks the phase that has not closed yet, beside the time it has been open.
///
/// The Edit History panel's cursor mark, and the same shape for the same
/// reason: it points at the row the viewer is on. The smaller `\u{25b8}` reads
/// better at this size and is not in the fonts egui bundles, so it draws as an
/// empty box.
pub(super) const OPEN_MARK: &str = "\u{25b6}";

/// Width of the cost column: [`ActionLog::format_took`]'s widest ordinary
/// output, `999.9 s`, and a gap before the panel's edge.
const COST_WIDTH: f32 = 56.0;

/// How often a frame is asked for while an operation runs.
///
/// The elapsed counts up, and a worker deep in a silent stage sends no report
/// for the frame to ride on, so an idle event loop would leave the number
/// frozen: a viewer that looks stopped is the thing this whole panel exists to
/// prevent. Ten frames a second is enough for a reader watching seconds, and it
/// is asked for only where there is a live process.
const TICK: Duration = Duration::from_millis(100);

/// The panel body.
pub(crate) fn show(ui: &mut egui::Ui, state: &mut AppState) {
    if state.background().is_some() {
        show_running(ui, state);
    } else {
        show_idle(ui, state);
    }
}

// -- Running ---------------------------------------------------------------

/// What the panel draws where the progress goes.
///
/// Two cases, and no third that invents a number: a bar moving at a rate nobody
/// measured makes a promise about the finish. A stage that reports nothing gets
/// a spinner and its own name rather than a bar creeping across it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Bar {
    /// Something underneath reported a count, so there is a real fraction of
    /// the whole to draw and the kernel's own count to put beside it.
    Measured {
        /// How much of the whole operation is behind, in `0.0..=1.0`.
        fraction: f32,
        /// The count in the kernel's own unit, `round 2/3`, where it gave one.
        count: Option<String>,
    },
    /// Nothing underneath has reported a number, so the open phase's name is
    /// all the panel can honestly say.
    Spinner {
        /// The innermost open phase, or `None` before anything has opened one.
        phase: Option<&'static str>,
    },
}

/// Which of the two the operation has earned.
///
/// The fraction is the collector's, which is the mapped sum of what the stages
/// reported and is therefore measured rather than synthesised: a stage that
/// reports counts moves it smoothly across its own range, a stage that reports
/// nothing moves it not at all, and the step at a boundary is the next stage's
/// range beginning rather than anything interpolated here.
pub(crate) fn bar(collector: &Collector, live: &Live) -> Bar {
    match collector.fraction() {
        Some(fraction) => Bar::Measured {
            fraction,
            count: collector.count().map(count_text),
        },
        None => Bar::Spinner {
            phase: live.open.last().and_then(|&row| phase_name(live, row)),
        },
    }
}

/// A count in words: `round 2/3`, or `iteration 12` where the total is unknown.
///
/// The unit leads because it is singular: a bare `12 iteration` reads as a
/// typo where `iteration 12` reads as a label.
fn count_text(count: Count) -> String {
    match count.total {
        Some(total) => format!("{} {}/{total}", count.unit, count.done),
        None => format!("{} {}", count.unit, count.done),
    }
}

/// The name of the phase at `row`, or `None` where that row is a message.
fn phase_name(live: &Live, row: usize) -> Option<&'static str> {
    match live.rows.get(row) {
        Some(Detail::Phase { name, .. }) => Some(name),
        _ => None,
    }
}

/// The running form: the operation, the node, the progress, the elapsed, the
/// Cancel button and the live phase table.
fn show_running(ui: &mut egui::Ui, state: &mut AppState) {
    // Everything is read off the process first: the Cancel below needs
    // `&mut AppState`, and these reads borrow it.
    let refusal = state.cancel_refusal();
    let process = state.background().expect("just checked");
    let name = process.operation.name;
    let label = process.label.clone();
    let elapsed = process.started.elapsed();
    let live = process.collector.live();
    let status = process.collector.status();
    let bar = bar(&process.collector, &live);

    ui.label(egui::RichText::new(name).strong());
    ui.label(egui::RichText::new(label).weak());
    show_bar(ui, &bar);

    // Absent rather than blank: an operation that says nothing about what it is
    // doing right now leaves no row, where an empty one would read as a stage
    // that had failed to name itself.
    if let Some(status) = status {
        ui.label(egui::RichText::new(status).weak());
    }

    let mut cancel = false;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("{} elapsed", ActionLog::format_took(elapsed))).weak(),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Present always, so a reader never wonders whether they missed it,
            // and live only where asking would do something.
            let button = ui.add_enabled(refusal.is_none(), egui::Button::new("Cancel"));
            let button = match refusal.as_deref() {
                Some(refusal) => button.on_disabled_hover_text(refusal),
                None => button.on_hover_text("Ask the operation to stop."),
            };
            if button.clicked() {
                cancel = true;
            }
        });
    });
    ui.separator();
    show_phases(ui, &live.rows, &live.open);
    ui.ctx().request_repaint_after(TICK);

    if cancel {
        state.cancel_background();
    }
}

/// The bar, or the spinner that stands in for one.
fn show_bar(ui: &mut egui::Ui, bar: &Bar) {
    match bar {
        Bar::Measured { fraction, count } => {
            let mut widget = egui::ProgressBar::new(*fraction);
            if let Some(count) = count {
                widget = widget.text(count.clone());
            }
            ui.add(widget);
        }
        Bar::Spinner { phase } => {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new());
                if let Some(phase) = phase {
                    ui.label(egui::RichText::new(*phase).weak());
                }
            });
        }
    }
}

// -- Idle ------------------------------------------------------------------

/// The idle form: the last operation of the session, greyed, with its phases
/// under a toggle.
fn show_idle(ui: &mut egui::Ui, state: &mut AppState) {
    let Some(last) = state.last_background.as_ref() else {
        ui.label(egui::RichText::new("Nothing running").weak());
        return;
    };
    let expandable = !last.detail.is_empty();
    let expanded = state.background_detail_expanded && expandable;
    let name = last.operation.name;
    let label = last.label.clone();
    let took = ActionLog::format_took(last.took);

    let mut toggled = false;
    ui.horizontal(|ui| {
        // The Action Log's glyphs and the Action Log's meaning: `+` opens, `-`
        // closes, and nothing at all where there is nothing to open.
        if expandable {
            let toggle = if expanded { "-" } else { "+" };
            let response =
                ui.add(egui::Button::new(egui::RichText::new(toggle).monospace()).frame(false));
            if response.clicked() {
                toggled = true;
            }
        }
        // The cost is reserved before the names are drawn, for the reason a
        // phase row reserves it: a node with a long label would otherwise push
        // the number off the panel, and the number is what a reader came back
        // to this panel to find.
        let width = (ui.available_width() - COST_WIDTH).max(0.0);
        ui.allocate_ui_with_layout(
            egui::vec2(width, ui.text_style_height(&egui::TextStyle::Body)),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(egui::RichText::new(name).weak());
                ui.add(
                    egui::Label::new(egui::RichText::new(&label).weak())
                        .truncate()
                        .selectable(false),
                )
                .on_hover_text(label);
            },
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(took).weak());
        });
    });
    if expanded {
        ui.separator();
        // Nothing is open: the operation is over, so every row is a run that
        // closed.
        show_phases(ui, &last.detail, &[]);
    }
    if toggled {
        state.background_detail_expanded = !state.background_detail_expanded;
    }
}

// -- The phase table -------------------------------------------------------

/// The transcript, one row per run of a stage and one per message, with `open`
/// naming the runs that have not closed.
///
/// Virtualized on a uniform row height, as the Action Log's list is, because
/// nothing folds here: a three-round, sixty-iteration adjustment with detailed
/// timing on opens `linearise` and its two siblings five hundred and forty
/// times, and every one of those is a row. Drawing only the range in view is
/// what keeps that affordable at ten frames a second.
fn show_phases(ui: &mut egui::Ui, rows: &[Detail], open: &[usize]) {
    let breakdown = Breakdown::running(rows);
    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
    egui::ScrollArea::vertical()
        .id_salt("background_phases")
        .auto_shrink([false, false])
        // The stage that is running is the newest row, and this panel is narrow
        // enough that the stages which finished early fill it: without this, a
        // reader watching a long solve sees the prologue for the whole of it and
        // has to scroll to find out what it is doing now. The Action Log follows
        // its own tail for the same reason, and this holds still the moment the
        // reader scrolls up, so an early stage can be read while the operation
        // keeps going.
        .stick_to_bottom(true)
        .show_rows(ui, row_height, rows.len(), |ui, range| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for index in range {
                show_phase_row(ui, &breakdown, index, open.contains(&index));
            }
        });
}

/// One row: the stage at its indent on the left, what it has cost on the right,
/// and the open mark between them where the run is still going.
fn show_phase_row(ui: &mut egui::Ui, breakdown: &Breakdown<'_>, index: usize, open: bool) {
    let row = detail_row(breakdown, index);
    let text = detail_text(&row);
    let color = if row.warn {
        ui.visuals().error_fg_color
    } else {
        ui.visuals().text_color()
    };
    let weak = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        // The name is given what is left over rather than allowed to claim it:
        // a long note would otherwise push the cost off the panel, and the cost
        // is the column a reader came for.
        let width = (ui.available_width() - COST_WIDTH).max(0.0);
        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        ui.allocate_ui_with_layout(
            egui::vec2(width, height),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(&text).monospace().color(color))
                        .truncate()
                        .selectable(false),
                );
            },
        )
        // A note or a message runs past this column far more often than it does
        // in the Action Log, which has the width of the window to spend, so the
        // truncated half is read the way the Action Log's is read.
        .response
        .on_hover_text(text);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(&row.cost).monospace().color(weak));
            if open {
                ui.label(egui::RichText::new(OPEN_MARK).color(color));
            }
        });
    });
}
