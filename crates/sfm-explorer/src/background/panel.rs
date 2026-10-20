// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Background Tasks panel: what each queue is running, on which node, and
//! what one task has spent its time on so far.
//!
//! See `specs/gui/background-tasks.md`, "What the user sees". The
//! panel decides nothing: it reads each queue's running task
//! ([`AppState::queue`]), the record of finished tasks and the collector a
//! running task shares with its worker, and its Cancel buttons call
//! [`AppState::cancel_background_task`].
//!
//! ## A line per queue, and neither form is blank
//!
//! Each queue has one line. Running, it draws the operation, the node, the
//! elapsed and a Cancel button over a fill showing how far along the task is,
//! or a sliding band where nothing is measured. Idle, it draws the last
//! task that finished from that queue greyed, with its node and its cost,
//! because the question a reader brings here after a long solve is "what did
//! that cost" and the answer is still in reach. A queue that has run nothing
//! says `Nothing running` and no more. A queue running a quiet task has one
//! dim line more, under its own, naming it, with no fill and no Cancel. A queue
//! with tasks waiting has one line more for the next of them, with a cancel
//! button and a count of the rest; quiet tasks waiting are not named or
//! counted there. Below the lines is one phase table, for
//! the task a person last clicked the name of, or else the slow queue's running
//! task, the fast queue's, or the last task to finish, under a title line that
//! carries a running task's counts and status, or a finished one's ending.
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
use crate::background::{Ended, FinishedTask, Queue, TaskState};
use crate::progress::{Collector, Count, Detail, Live, StageCount};
use crate::state::AppState;

/// Marks the phase that has not closed yet, beside the time it has been open.
///
/// The Edit History panel's cursor mark, and the same shape for the same
/// reason: it points at the row the viewer is on. The smaller `\u{25b8}` reads
/// better at this size and is not in the fonts egui bundles, so it draws as an
/// empty box.
pub(super) const OPEN_MARK: &str = "\u{25b6}";

/// The least share of the title line's room, after the count, that the title
/// keeps when a status follows it.
const TITLE_SHARE: f32 = 0.6;

/// How to read nested counts, under the whole of them on the title line's
/// hover: see [`in_progress`].
pub(super) const COUNTS_HOVER: &str =
    "The outer counts name the unit in progress; the last counts the units done.";

/// The most of the title line's width the counts take before they are written
/// shorter, so the title keeps the rest.
const COUNT_SHARE: f32 = 0.5;

/// The share of a queue line's room, after its right-hand items, that the
/// node's name keeps when the operation's name does not fit beside it whole.
const NODE_SHARE: f32 = 0.4;

/// How many of the operation's letters a queue line keeps, before an
/// ellipsis, however little room the node leaves it: `Bund…`.
const NAME_KEEPS: usize = 4;

/// The gap between the items of a queue's line.
const LINE_SPACING: f32 = 4.0;

/// A running task's Cancel button: egui's own close glyph, since the word
/// would take a quarter of a line the panel's width has no room for. Its
/// tooltip says what it does.
pub(super) const CANCEL: &str = "\u{1f5d9}";

/// What leads a waiting task's line, where a queue's line has its name.
pub(super) const WAITING: &str = "waiting";

/// What a running task's title line shows in place of its status once it has
/// been asked to stop and until it lands, and its queue's line in place of its
/// bar where the title line is another task's and the word leaves the
/// operation's first letters room.
pub(super) const STOPPING: &str = "stopping\u{2026}";

/// The tooltip of [`STOPPING`] and of the Cancel button it disables.
pub(super) const STOPPING_HOVER: &str =
    "Asked to stop: it stops the next time it checks whether it should.";

/// What an idle line and a finished task's title line say of a task that was
/// cancelled, running or waiting.
pub(super) const CANCELLED: &str = "cancelled";

/// What they say of a task whose kernel refused, or whose answer could not be
/// installed.
pub(super) const FAILED: &str = "failed";

/// What leads a running quiet task's line.
pub(super) const QUIET: &str = "quiet";

/// Width of the cost column: [`ActionLog::format_took`]'s widest ordinary
/// output, `999.9 s`, and a gap before the panel's edge.
const COST_WIDTH: f32 = 56.0;

/// How often a frame is asked for while an operation runs.
///
/// The elapsed counts up, and a worker deep in a silent stage sends no report
/// for the frame to ride on, so an idle event loop would leave the number
/// frozen: a viewer that looks stopped is the thing this whole panel exists to
/// prevent. Ten frames a second is enough for a reader watching seconds, and it
/// is asked for only where there is a live task.
const TICK: Duration = Duration::from_millis(100);

/// The panel body: one line per queue, then the phases of one task.
///
/// A line rather than a block per queue, because the panel sits at the bottom
/// of the left column, about 170 px tall in a 720 px window, and the phase
/// table under the lines is what a reader watches: two blocks of four rows each
/// would leave it one row, and not the row that is running.
pub(crate) fn show(ui: &mut egui::Ui, state: &mut AppState) {
    let mut asked = Asked::default();
    for queue in Queue::ALL {
        let running = state.running_in(queue).map(|task| task.id);
        // A place in the paint order before the line's widgets, so the fill
        // drawn into it once the line's height is known lies behind them.
        let behind = ui.painter().add(egui::Shape::Noop);
        let row = ui
            .horizontal(|ui| {
                // Closer than egui's default, so the node's name keeps some of
                // a line that also holds the queue, the operation, the
                // elapsed and the Cancel button.
                ui.spacing_mut().item_spacing.x = LINE_SPACING;
                ui.label(egui::RichText::new(queue.title()).strong());
                match running {
                    Some(id) => show_running(ui, state, id, &mut asked),
                    None => show_idle(ui, state, queue, &mut asked),
                }
            })
            .response
            .rect;
        if let Some(task) = running.and_then(|id| state.running_task(id)) {
            let line = egui::Rect::from_x_y_ranges(ui.max_rect().x_range(), row.y_range());
            let live = task.collector.live();
            let form = Fill {
                bar: bar(&task.collector, &live),
                stopping: task.stopping(),
                chosen: chosen_task(state) == Some(task.id),
            };
            let time = ui.input(|input| input.time);
            let moving = ui.style().animation_time > 0.0;
            ui.painter()
                .set(behind, fill_shape(ui.visuals(), &form, line, time, moving));
        }
        if let Some(task) = state.quiet_running_in(queue) {
            let (name, label) = (task.operation.name, task.label.clone());
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = LINE_SPACING;
                show_quiet(ui, name, &label);
            });
        }
        if state.waiting_in(queue).next().is_some() {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = LINE_SPACING;
                show_waiting(ui, state, queue, &mut asked);
            });
        }
    }
    if state.running_tasks().next().is_some() {
        ui.ctx().request_repaint_after(TICK);
    }
    ui.separator();
    show_chosen_phases(ui, state);

    if let Some(id) = asked.shown {
        state.background_phases_of = Some(id);
    }
    if let Some(id) = asked.cancel {
        state.cancel_background_task(id);
    }
}

/// What a click in the sections asked for, acted on once the sections are
/// drawn: both need `&mut AppState`, which the drawing borrows.
#[derive(Default)]
struct Asked {
    /// The task whose name was clicked, whose phases the panel shows next.
    shown: Option<u64>,
    /// The task whose Cancel was clicked.
    cancel: Option<u64>,
}

/// The task whose phases the panel shows: the one a person last clicked while
/// it is still running or kept, and otherwise the slow queue's running task,
/// the fast queue's, or the last task to finish, in that order.
///
/// One phase list for one task, so the panel stays the height it was with one
/// queue.
pub(crate) fn chosen_task(state: &AppState) -> Option<u64> {
    state
        .background_phases_of
        .filter(|&id| {
            matches!(
                state.task(id),
                Some(TaskState::Running(_) | TaskState::Finished(_))
            )
        })
        .or_else(|| state.running_tasks().next().map(|task| task.id))
        .or_else(|| state.last_finished_task().map(|task| task.id))
}

/// The task's name, as a label that picks its phases when clicked, cut to the
/// room left on its line.
///
/// `selected` draws it as egui draws a selected button. A running line passes
/// false and is outlined whole instead ([`fill_shape`]), because its fill is
/// the selection's colour too and the two would run together.
fn name_button(
    ui: &mut egui::Ui,
    text: egui::RichText,
    id: u64,
    selected: bool,
    asked: &mut Asked,
) {
    let name = text.text().to_string();
    if ui
        .add(egui::Button::selectable(selected, text).truncate())
        .on_hover_text(format!("{name}: show its phases below."))
        .clicked()
    {
        asked.shown = Some(id);
    }
}

// -- Running ---------------------------------------------------------------

/// What a running line's fill measures ([`fill_shape`]).
///
/// Two cases, and no third that invents a number: a bar moving at a rate nobody
/// measured makes a promise about the finish. A stage that reports nothing gets
/// a band sliding along the line, and its own name as the marked row of the
/// phase table, rather than a fill creeping across it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Bar {
    /// Something underneath reported a count, so there is a real fraction of
    /// the whole to draw and the kernel's own count to put beside it.
    Measured {
        /// How much of the whole operation is behind, in `0.0..=1.0`.
        fraction: f32,
        /// The counts in the kernel's own units, outermost first, `round 2/3`
        /// or `round 2/3 · iteration 4/60`, where it gave any.
        count: Option<String>,
    },
    /// Nothing underneath has reported a number, so the open phase's name is
    /// all the panel can honestly say.
    Unmeasured {
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
            count: counts_text(&in_progress(&collector.counts()), true),
        },
        None => Bar::Unmeasured {
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

/// The counts as the title line reads them: each outer count names the unit
/// in progress, the innermost the units done.
///
/// A kernel counts what is behind it, so a solve in its fifth round has
/// reported `round 4/9`. Read beside `iteration 4/60`, that says the fourth
/// round is running. So an outer count whose stage has a stage open inside it,
/// which is work on the next unit begun, reads one more than it reported,
/// capped at its total: `round 5/9`. With nothing open inside it, between
/// rounds, it reads what it reported. The innermost count is left as reported
/// in every case, so its number does not step between a unit's stages when
/// detailed timing opens phases inside it.
fn in_progress(counts: &[StageCount]) -> Vec<Count> {
    let innermost = counts.len().saturating_sub(1);
    counts
        .iter()
        .enumerate()
        .map(|(at, staged)| {
            let mut count = staged.count;
            if at < innermost && staged.under_way {
                count.done = match count.total {
                    Some(total) => (count.done + 1).min(total),
                    None => count.done + 1,
                };
            }
            count
        })
        .collect()
}

/// The counts of the stages still open, outermost first, joined by ` · `:
/// `round 4/9 · iteration 4/60`. With `units` false every count after the
/// first is its numbers alone, `round 4/9 · 4/60`, which is the short form the
/// title line falls back to. `None` where nothing has reported a count.
fn counts_text(counts: &[Count], units: bool) -> Option<String> {
    let mut parts = counts.iter().enumerate().map(|(at, &count)| {
        if units || at == 0 {
            count_text(count)
        } else {
            match count.total {
                Some(total) => format!("{}/{total}", count.done),
                None => count.done.to_string(),
            }
        }
    });
    let first = parts.next()?;
    Some(parts.fold(first, |line, part| format!("{line} \u{b7} {part}")))
}

/// What the title line shows of the counts in `room`, the whole line's width.
///
/// The whole of them where that takes no more than [`COUNT_SHARE`] of the
/// line, then the short form, and otherwise the innermost count alone, which
/// always shows, as the one count did before nested counts were drawn: it is
/// the number a reader watches move.
fn counts_in(ui: &egui::Ui, counts: &[Count], room: f32) -> Option<String> {
    let innermost = count_text(*counts.last()?);
    let fits = |text: &String| text_width(ui, text) <= room * COUNT_SHARE;
    [counts_text(counts, true), counts_text(counts, false)]
        .into_iter()
        .flatten()
        .find(fits)
        .or(Some(innermost))
}

/// The name of the phase at `row`, or `None` where that row is a message.
fn phase_name(live: &Live, row: usize) -> Option<&'static str> {
    match live.rows.get(row) {
        Some(Detail::Phase { name, .. }) => Some(name),
        _ => None,
    }
}

/// A queue's running task, on its queue's line: the operation, the node, the
/// elapsed and the Cancel button, over the fill that shows how far along it is
/// ([`fill_shape`], drawn by [`show`] once the line's height is known).
///
/// The count and the status line are the phase list's title's, for the task
/// whose phases are shown ([`show_chosen_phases`]), so the line stays one
/// line.
fn show_running(ui: &mut egui::Ui, state: &AppState, id: u64, asked: &mut Asked) {
    let refusal = state.cancel_refusal(id);
    let task = state.running_task(id).expect("the queue's running task");
    let name = task.operation.name;
    let label = task.label.clone();
    let elapsed = task.started.elapsed();
    let live = task.collector.live();
    let count = match bar(&task.collector, &live) {
        Bar::Measured { count, .. } => count,
        Bar::Unmeasured { .. } => None,
    };
    let stopping = task.stopping();

    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        // Present always, so a reader never wonders whether they missed it,
        // and live only where asking would do something: not where the
        // operation never asks, and not once it has been asked.
        let button = ui.add_enabled(
            refusal.is_none() && !stopping,
            egui::Button::new(CANCEL).small(),
        );
        let button = match (refusal.as_deref(), stopping) {
            (Some(refusal), _) => button.on_disabled_hover_text(refusal),
            (None, true) => button.on_disabled_hover_text(STOPPING_HOVER),
            (None, false) => button.on_hover_text("Ask the operation to stop."),
        };
        if button.clicked() {
            asked.cancel = Some(id);
        }
        // The elapsed is reserved before the node's name is drawn, as an idle
        // line's cost is: it is the number a reader is watching. The counts
        // the fill measures are on its hover, as on the title line.
        let elapsed = ui.label(egui::RichText::new(ActionLog::format_took(elapsed)).weak());
        match count {
            Some(count) => elapsed.on_hover_text(format!("elapsed; {count}")),
            None => elapsed.on_hover_text("elapsed"),
        };
        // Asked to stop, the line says so in words where they leave the names
        // what they would have beside the word anyway: the operation's first
        // letters and the node's share. Where they do not, the grey fill, the
        // title line when it is the chosen task and the disabled button's
        // tooltip say it. The names say what is stopping, so it is the word
        // that gives way.
        let gap = ui.spacing().item_spacing.x;
        let room = ui.available_width();
        let node_keeps = text_width(ui, &label).min(room * NODE_SHARE);
        let word_fits = room - text_width(ui, STOPPING) - gap
            >= name_widths(ui, name, Name::Button).1 + gap + node_keeps;
        if stopping && word_fits {
            ui.add(egui::Label::new(egui::RichText::new(STOPPING).weak()).selectable(false))
                .on_hover_text(STOPPING_HOVER);
        }
        // The operation and its node from the left, in what the right-hand
        // items leave, so a long operation name is cut rather than drawn
        // under the elapsed.
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
            let text = egui::RichText::new(name).strong();
            name_and_node(ui, name, Name::Button, &label, |ui| {
                name_button(ui, text, id, false, asked)
            });
        });
    });
}

/// What a running line's fill draws: how far along the task is, whether it
/// has been asked to stop, and whether it is the chosen task.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Fill {
    /// The measured fraction, or nothing measured.
    pub(crate) bar: Bar,
    /// Asked to stop and not yet landed.
    pub(crate) stopping: bool,
    /// The task whose phases are shown below.
    pub(crate) chosen: bool,
}

/// The share of the line the band of the indeterminate fill covers.
const BAND_SHARE: f32 = 0.3;

/// How long the band takes to cross the line, in seconds: slow enough to read
/// as "working" rather than as a measure of anything.
const BAND_PERIOD: f64 = 2.4;

/// The tint a running line's fill is drawn in: the selection's colour, faint
/// enough that the line's text keeps its contrast over it in the dark theme
/// and in the light one.
pub(crate) fn fill_color(visuals: &egui::Visuals, stopping: bool) -> egui::Color32 {
    match stopping {
        // Grey rather than the selection's hue: the work it measured is no
        // longer what the task is doing.
        true => visuals.weak_text_color().gamma_multiply(0.2),
        false => visuals.selection.bg_fill.gamma_multiply(0.35),
    }
}

/// The span of `line` the fill covers.
///
/// Measured, from the line's left edge to `fraction` of its width. With
/// nothing measured, a band of [`BAND_SHARE`] of the width that slides across
/// the line once every [`BAND_PERIOD`] seconds, or stands at its middle where
/// `moving` is false (egui's animations turned off) or the task is stopping;
/// a stopping task with nothing measured is filled across the whole line, so
/// it reads as held rather than working.
pub(crate) fn fill_span(form: &Fill, line: egui::Rect, time: f64, moving: bool) -> egui::Rect {
    let width = line.width();
    let (left, right) = match (&form.bar, form.stopping) {
        (Bar::Measured { fraction, .. }, _) => (0.0, fraction.clamp(0.0, 1.0) * width),
        (Bar::Unmeasured { .. }, true) => (0.0, width),
        (Bar::Unmeasured { .. }, false) => {
            let band = BAND_SHARE * width;
            let at = match moving {
                true => ((time / BAND_PERIOD).fract() as f32) * (width + band) - band,
                false => (width - band) / 2.0,
            };
            (at.max(0.0), (at + band).min(width))
        }
    };
    egui::Rect::from_x_y_ranges(line.left() + left..=line.left() + right, line.y_range())
}

/// The fill behind a running line, its leading edge and, for the chosen task,
/// an outline round the whole line.
///
/// Behind rather than beside the text, so the 28 px a bar once took on the
/// line is the names' again and the measure is the line's whole width. The
/// chosen task is outlined rather than filled, as an idle line's selected name
/// is, because its fill is already the selection's colour.
pub(crate) fn fill_shape(
    visuals: &egui::Visuals,
    form: &Fill,
    line: egui::Rect,
    time: f64,
    moving: bool,
) -> egui::Shape {
    let span = fill_span(form, line, time, moving);
    let mut shapes = vec![egui::Shape::rect_filled(
        span,
        0.0,
        fill_color(visuals, form.stopping),
    )];
    // Where a measured fill ends, a thin stronger line, so the end can be read
    // against text of about the same brightness.
    if let (Bar::Measured { fraction, .. }, false) = (&form.bar, form.stopping) {
        if *fraction > 0.0 && *fraction < 1.0 {
            shapes.push(egui::Shape::line_segment(
                [span.right_top(), span.right_bottom()],
                egui::Stroke::new(1.5, visuals.selection.bg_fill),
            ));
        }
    }
    if form.chosen {
        shapes.push(egui::Shape::rect_stroke(
            line.shrink(0.5),
            2.0,
            egui::Stroke::new(1.0, visuals.selection.stroke.color),
            egui::StrokeKind::Inside,
        ));
    }
    egui::Shape::Vec(shapes)
}

/// The node's label, cut out of the middle to the room left on the line, with
/// the whole of it on hover.
fn show_node(ui: &mut egui::Ui, label: &str) {
    // The cut here is ours, out of the middle, so `Label` is handed a string
    // that already fits and would tooltip nothing: its own elision tooltip is
    // turned off and the full name given directly. That is the one tooltip,
    // rather than the two a `Label` that elided for itself would stack.
    ui.add(
        egui::Label::new(egui::RichText::new(elide_to_fit(ui, label)).weak())
            .show_tooltip_when_elided(false)
            .truncate()
            .selectable(false),
    )
    .on_hover_text(label);
}

/// How a line draws the operation's name, which decides how wide it is drawn.
#[derive(Clone, Copy)]
enum Name {
    /// A button that picks the task's phases, with a button's padding.
    Button,
    /// A plain label.
    Label,
}

/// The operation's name and then the node's, in the room left on the line,
/// the node keeping some of it whatever the operation is called.
///
/// Both whole where they fit. Where they do not, the operation's name is cut
/// first, down to what leaves the node its whole name or [`NODE_SHARE`] of the
/// room, whichever is less, and the node takes the rest, cut out of the middle.
/// The node is what says which reconstruction is busy, and the operation's name
/// is repeated in full on its tooltip and, for the chosen task, on the title
/// line; a long operation name taking the whole line left the node none of it.
/// The operation's name is never cut below its first [`NAME_KEEPS`] letters
/// and an ellipsis, though, so a narrow line still says what is running: a
/// stopping adjustment at 227 px once read `Slow … dino`.
fn name_and_node(
    ui: &mut egui::Ui,
    name: &str,
    drawn_as: Name,
    label: &str,
    draw_name: impl FnOnce(&mut egui::Ui),
) {
    let room = ui.available_width();
    let gap = ui.spacing().item_spacing.x;
    let (wanted, least) = name_widths(ui, name, drawn_as);
    let node_keeps = text_width(ui, label).min(room * NODE_SHARE);
    // The operation keeps its first few letters before the node takes its
    // share: a name cut to a bare ellipsis says nothing about what is running.
    let name_room = wanted
        .min((room - gap - node_keeps).max(least))
        .min(room.max(0.0));
    let height = ui.spacing().interact_size.y;
    ui.allocate_ui_with_layout(
        egui::vec2(name_room, height),
        egui::Layout::left_to_right(egui::Align::Center),
        draw_name,
    );
    show_node(ui, label);
}

/// How wide the operation's name is drawn whole, and the least it is drawn
/// at: its first [`NAME_KEEPS`] letters and an ellipsis, or the whole of it
/// where that is narrower.
fn name_widths(ui: &egui::Ui, name: &str, drawn_as: Name) -> (f32, f32) {
    let padding = match drawn_as {
        Name::Button => 2.0 * ui.spacing().button_padding.x,
        Name::Label => 0.0,
    };
    // A point of slack, so a name that fits is not cut by a rounding.
    let wanted = text_width(ui, name) + padding + 1.0;
    let prefix: String = name.chars().take(NAME_KEEPS).collect();
    let least = wanted.min(text_width(ui, &format!("{prefix}\u{2026}")) + padding + 1.0);
    (wanted, least)
}

/// `text` cut to the width left in `ui`, out of the middle.
///
/// Both ends of a node's name carry meaning and the middle does not: the front
/// is the date every run of that day shares, and the back is the range and the
/// `-embedded` suffix. See [`crate::elide`].
fn elide_to_fit(ui: &egui::Ui, text: &str) -> String {
    elide_to(ui, text, ui.available_width())
}

/// `text` cut out of the middle to `room` points.
fn elide_to(ui: &egui::Ui, text: &str, room: f32) -> String {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let color = ui.visuals().weak_text_color();
    crate::elide::middle(text, room, |candidate| {
        ui.ctx().fonts_mut(|fonts| {
            fonts
                .layout_no_wrap(candidate.to_owned(), font.clone(), color)
                .rect
                .width()
        })
    })
}

// -- Waiting ---------------------------------------------------------------

/// The line under a busy queue's that names the task it runs next: the word
/// `waiting`, the operation, its node and a cancel button, and `+{k}` before
/// the button when `k` more wait behind it.
///
/// One line per queue however many wait, because the panel is about 170 px
/// tall and a line per waiting task would push the phase table out of it: the
/// table is what a reader watches, and the tasks behind the next one are
/// named in the `+{k}`'s tooltip, in order, and reach the front of the line
/// one at a time. A waiting task has no phases, so its name is not a button
/// that picks them.
fn show_waiting(ui: &mut egui::Ui, state: &AppState, queue: Queue, asked: &mut Asked) {
    let mut waiting = state.waiting_in(queue);
    let Some(next) = waiting.next() else {
        return;
    };
    let id = next.id;
    let name = next.operation.name;
    let label = next.label.clone();
    let behind: Vec<String> = waiting
        .map(|task| format!("{} on {}", task.operation.name, task.label))
        .collect();

    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if ui
            .add(egui::Button::new(CANCEL).small())
            .on_hover_text("Take it out of the queue before it starts.")
            .clicked()
        {
            asked.cancel = Some(id);
        }
        if !behind.is_empty() {
            ui.label(egui::RichText::new(format!("+{}", behind.len())).weak())
                .on_hover_text(format!("Also waiting, in order: {}.", behind.join("; ")));
        }
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(WAITING).weak());
            name_and_node(ui, name, Name::Label, &label, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(name))
                        .truncate()
                        .selectable(false),
                );
            });
        });
    });
}

// -- Quiet -----------------------------------------------------------------

/// The line under a queue's that names the quiet task it runs: the word
/// `quiet`, the operation and its subject, all dim, with no bar, no elapsed
/// and no Cancel.
///
/// A quiet task is work the viewer started by itself, and it stops on its own
/// whenever someone asks for a task, so there is nothing for a reader to wait
/// on or to stop. It runs only while its queue runs nothing someone asked for,
/// so the queue's line above it is in its idle form. It has no phases the
/// panel shows, so its name is not a button that picks them.
fn show_quiet(ui: &mut egui::Ui, name: &str, label: &str) {
    ui.label(egui::RichText::new(QUIET).weak());
    name_and_node(ui, name, Name::Label, label, |ui| {
        ui.add(
            egui::Label::new(egui::RichText::new(name).weak())
                .truncate()
                .selectable(false),
        );
    });
}

// -- Idle ------------------------------------------------------------------

/// An idle queue's line: the last task that finished from it, greyed, with
/// its node and what it cost, or `Nothing running` when it has run nothing.
fn show_idle(ui: &mut egui::Ui, state: &AppState, queue: Queue, asked: &mut Asked) {
    let Some(last) = state.last_finished_in(queue) else {
        ui.label(egui::RichText::new("Nothing running").weak());
        return;
    };
    let chosen = chosen_task(state);
    let id = last.id;
    let name = last.operation.name;
    let label = last.label.clone();
    let took = ActionLog::format_took(last.took);
    // The cost is drawn before the node, from the right: a long name would
    // otherwise push the number off the panel, and the number is what a
    // reader came back to this panel to find. How it ended, where it did not
    // finish, is next, for the same reason. The node takes what is left.
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.label(egui::RichText::new(took).weak());
        show_ending(ui, last);
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
            let text = egui::RichText::new(name).weak();
            name_and_node(ui, name, Name::Button, &label, |ui| {
                name_button(ui, text, id, chosen == Some(id), asked)
            });
        });
    });
}

/// How a finished task ended, where it did not run to its end: the weak word
/// `cancelled`, or `failed` in the error colour, with the Action Log's
/// sentence on hover. Nothing for a task that finished.
///
/// A word rather than a mark, because the panel's other marks are a close
/// glyph that cancels and a triangle that points at the open stage, and a
/// third glyph would be one more thing to learn; the word is short enough to
/// leave the node its share of the line.
fn show_ending(ui: &mut egui::Ui, task: &FinishedTask) {
    let (word, color) = match task.ended() {
        Ended::Done => return,
        Ended::Cancelled => (CANCELLED, ui.visuals().weak_text_color()),
        Ended::Failed => (FAILED, ui.visuals().error_fg_color),
    };
    let sentence = match &task.outcome {
        Ok(text) | Err(text) => text.clone(),
    };
    ui.add(egui::Label::new(egui::RichText::new(word).color(color)).selectable(false))
        .on_hover_text(sentence);
}

/// The Action Log sentence a finished task wrote, wrapped to the panel's
/// width, in the error colour for a task that failed and weak otherwise: what
/// is shown under the title line of a task that reported no stages.
fn show_sentence(ui: &mut egui::Ui, task: &FinishedTask) {
    let sentence = match &task.outcome {
        Ok(text) | Err(text) => text.as_str(),
    };
    let color = match task.ended() {
        Ended::Failed => ui.visuals().error_fg_color,
        Ended::Done | Ended::Cancelled => ui.visuals().weak_text_color(),
    };
    ui.add(
        egui::Label::new(egui::RichText::new(sentence).color(color))
            .wrap()
            .selectable(false),
    );
}

/// The phase list's title: the task and its node, whose whole name the queue
/// line may have had to cut.
pub(crate) fn phases_title(operation: &str, label: &str) -> String {
    format!("{operation} on {label}")
}

/// The phase list's title in `room`, with the whole of it on hover.
///
/// `{operation} on {node}` where that fits. Where it does not, the node's name
/// alone, cut out of the middle as a queue line cuts it, so its start and its
/// end both stay: the operation is named on its queue's line, which may have
/// cut the node's name to a letter, and this line is where a reader finds which
/// run it is.
fn show_title(ui: &mut egui::Ui, operation: &str, label: &str, room: f32) {
    let title = phases_title(operation, label);
    let shown = if text_width(ui, &title) <= room {
        title.clone()
    } else {
        elide_to(ui, label, room)
    };
    ui.add(
        egui::Label::new(egui::RichText::new(shown).weak())
            .show_tooltip_when_elided(false)
            .truncate()
            .selectable(false),
    )
    .on_hover_text(title);
}

/// How wide `text` is drawn in the panel's body font.
fn text_width(ui: &egui::Ui, text: &str) -> f32 {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let color = ui.visuals().weak_text_color();
    ui.ctx().fonts_mut(|fonts| {
        fonts
            .layout_no_wrap(text.to_owned(), font, color)
            .rect
            .width()
    })
}

/// The phases of the task [`chosen_task`] names, under a line naming it.
///
/// For a running task the title line also carries the kernel's counts, `round
/// 2/3 · iteration 4/60`, which are laid out first from the right so they
/// always show ([`counts_in`]), and the status line, the one thing the
/// operation says it is doing right now, cut to what the title leaves: the
/// queue lines have no room for either. A finished task's title line carries
/// how it ended, where it did not finish.
///
/// Always showing, for a finished task as for a running one. The Action Log
/// hides a breakdown behind a toggle because it has a row for every action of
/// the session and expanding them all would bury the list; this panel shows
/// one task's, and the breakdown is the main thing it has to say. A toggle
/// there would be a click between a reader and the thing they opened the panel
/// for.
fn show_chosen_phases(ui: &mut egui::Ui, state: &AppState) {
    let Some(id) = chosen_task(state) else {
        return;
    };
    match state.task(id) {
        Some(TaskState::Running(task)) => {
            let live = task.collector.live();
            let counts = in_progress(&task.collector.counts());
            // Asked to stop, what it last said it was doing is no longer the
            // news: that it is stopping is.
            let status = match task.stopping() {
                true => Some(STOPPING.to_string()),
                false => task.collector.status(),
            };
            ui.horizontal(|ui| {
                // The counts from the right and first, so they are on screen
                // whatever the node is called: they are the numbers on this
                // line a reader watches move. All of them on hover, where the
                // line may have room for fewer, with how to read them where
                // there is more than one.
                let room = ui.available_width();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(shown) = counts_in(ui, &counts, room) {
                        let whole = counts_text(&counts, true).unwrap_or_default();
                        let label = ui.add(
                            egui::Label::new(egui::RichText::new(shown.clone()).weak())
                                .selectable(false),
                        );
                        if counts.len() > 1 {
                            label.on_hover_text(format!("{whole}\n{COUNTS_HOVER}"));
                        } else if shown != whole {
                            label.on_hover_text(whole);
                        }
                    }
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        // The title before the status: what the status needs,
                        // but never less than a share of the line.
                        let room = ui.available_width();
                        let room = match status.as_deref() {
                            Some(status) => (room - text_width(ui, status) - LINE_SPACING)
                                .max(room * TITLE_SHARE),
                            None => room,
                        };
                        show_title(ui, task.operation.name, &task.label, room);
                        // Absent rather than blank: an operation that says
                        // nothing about what it is doing right now leaves
                        // nothing here, where an empty label would read as a
                        // stage that had failed to name itself.
                        if let Some(status) = status {
                            ui.add(
                                egui::Label::new(egui::RichText::new(status).weak())
                                    .truncate()
                                    .selectable(false),
                            );
                        }
                    });
                });
            });
            show_phases(ui, id, &live.rows, &live.open, true);
        }
        Some(TaskState::Finished(task)) => {
            ui.horizontal(|ui| {
                // How it ended from the right and first, where the running
                // form has its count, so it shows whatever the node is called.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    show_ending(ui, task);
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        let room = ui.available_width();
                        show_title(ui, task.operation.name, &task.label, room);
                    });
                });
            });
            if task.detail.is_empty() {
                // A task that reported no stages, such as a refusal before the
                // kernel began or a task cancelled while it waited, has no
                // table. Its Action Log sentence goes where the table would,
                // because for a refusal it is the reason, and an empty space
                // under the title would say nothing at all.
                show_sentence(ui, task);
            } else {
                // Nothing is open: the operation is over, so every row is a
                // run that closed.
                show_phases(ui, id, &task.detail, &[], false);
            }
        }
        _ => {}
    }
}

// -- The phase table -------------------------------------------------------

/// The transcript, one row per run of a stage and one per message, with `open`
/// naming the runs that have not closed.
///
/// `running` is whether the table belongs to a task that is still reporting.
/// A live one is read at its tail, because the stage that is running is the
/// newest row and this panel is narrow enough that the stages which finished
/// early fill it: without that, a reader watching a long solve sees the
/// prologue for the whole of it. The Action Log follows its own tail for the
/// same reason, and this holds still the moment the reader scrolls up, so an
/// early stage can be read while the task keeps going.
///
/// A finished task keeps the scroll position its live table had, and goes on
/// following its tail where the reader had left it there, so a task that lands
/// or stops while it is watched leaves the reader at the stage it ended on
/// rather than jumping back to its first row. A finished task whose table this
/// panel never drew while it ran has no position to keep, and is read from the
/// top, where it started.
///
/// Virtualized on a uniform row height, as the Action Log's list is, because
/// nothing folds here: a three-round, sixty-iteration adjustment with detailed
/// timing on opens `linearise` and its two siblings five hundred and forty
/// times, and every one of those is a row. Drawing only the range in view is
/// what keeps that affordable at ten frames a second.
fn show_phases(ui: &mut egui::Ui, id: u64, rows: &[Detail], open: &[usize], running: bool) {
    let breakdown = Breakdown::running(rows);
    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
    // Whether this task's table was drawn while it ran, which is what decides
    // whether its finished table follows the tail or opens at the top.
    let watched = egui::Id::new(("background_tasks_watched", id));
    let follow = running || ui.data(|data| data.get_temp::<bool>(watched).unwrap_or(false));
    if running {
        ui.data_mut(|data| data.insert_temp(watched, true));
    }
    egui::ScrollArea::vertical()
        // A scroll position of its own for each task, one for both of its
        // forms: another task's table opens wherever that task's was left, and
        // a task that finishes stays where its live table was.
        .id_salt(("background_tasks_phases", id))
        .auto_shrink([false, false])
        .stick_to_bottom(follow)
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
                // The whole of a cut row is read on hover, and `Label` is what
                // does it: it puts an elided text in a tooltip of its own, in
                // this label's font, and only when the text was actually cut. A
                // second `on_hover_text` of the same string stacks a
                // proportional-font copy over that one.
                ui.add(
                    egui::Label::new(egui::RichText::new(text).monospace().color(color))
                        .truncate()
                        .selectable(false),
                );
            },
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(&row.cost).monospace().color(weak));
            if open {
                ui.label(egui::RichText::new(OPEN_MARK).color(color));
            }
        });
    });
}
