// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where a long operation's progress lands in the viewer.
//!
//! `sfmtool_core` hands a slow function a [`Progress`], which is a sink and
//! nothing else: the caller decides where the events go. This is the viewer's
//! sink. A [`Collector`] is made when an operation starts, reported into while
//! it runs, and emptied into the Action Log entry the operation writes, so
//! there is nothing to attribute afterwards: the collector *is* that entry's
//! detail.
//!
//! See `specs/drafts/operation-progress.md`.
//!
//! ## Shared by `&`, never by `&mut`
//!
//! Every method here takes `&self`, and the events land behind a `Mutex`. That
//! is not a style preference. `TabContext` hands out seven simultaneous `&mut`
//! borrows and `AppState::scene_and_log` exists only because two things needed
//! borrowing at once, so a collector reached through `&mut AppState` would
//! conflict with every panel closure that already holds one. Taking it by `&`
//! also makes the worker case free: a background thread and the GUI thread hold
//! the same collector, and nothing crosses a channel.
//!
//! ## What is kept, and what is only the latest
//!
//! Phases and messages are the entry's detail, in the order they happened.
//! [`Event::Status`], [`Event::Count`] and [`Event::Fraction`] are live state:
//! they say what the operation is doing *now* and how far along it is, so only
//! the newest of each is kept and none of them reaches [`Collector::take`]. A
//! status in particular is never part of an entry, because by the time the
//! entry exists the answer to "what is it doing" is "finished".
//!
//! ## Repeated phases fold
//!
//! A guard goes where the code already has a boundary, and for a loop body that
//! means once per trip: a bundle adjustment of three rounds at sixty iterations
//! opens `linearise` and its two siblings five hundred and forty times. Drawn
//! one row each that is a transcript of the solve rather than a breakdown of
//! it, and it would exhaust `ActionLog::DETAIL_EVENTS` before the first round
//! finished. So within one enclosing phase every child sharing a name is one
//! row, whose time is the sum and which carries the count, ordered by first
//! appearance. Folding happens as the events arrive, which is what makes the
//! cap count rows kept rather than phases opened.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sfmtool_core::progress::{Event, Level, Phase, Progress};

#[cfg(test)]
mod tests;

/// One row of an Action Log entry's breakdown.
///
/// A phase is a stage that ran and what it cost; a message is something the
/// operation said. Both carry the depth they were reported at, which the panel
/// draws as an indent, and both sit in one sequence so that an expanded entry
/// reads as a short transcript rather than as a table beside a list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Detail {
    /// A stage, and what the runs of it under this parent cost between them.
    Phase {
        /// The name the stage was opened with, which after folding is a key as
        /// well as a label: two unrelated stages under one parent must not
        /// share one.
        name: &'static str,
        /// How deep it sits, counted from 0 at the operation's top.
        depth: u8,
        /// Wall time, summed over `runs`.
        took: Duration,
        /// Thread-summed CPU time, when the stage reports one. Never folded
        /// into a wall-clock total: eight seconds of CPU inside one second of
        /// wall is not eight seconds of anybody's wait.
        cpu: Option<Duration>,
        /// What the stage said it did, for the column beside its time, from
        /// the most recent run that said anything.
        note: Option<String>,
        /// How many times the stage ran under this parent. 1 for a stage that
        /// ran once, which is drawn with no count at all.
        runs: u32,
    },
    /// Something the operation said, at the depth of the phase it was said
    /// inside.
    Message {
        /// How much it wants to be noticed.
        level: Level,
        /// The depth of the phase it was emitted inside.
        depth: u8,
        /// The formatted text.
        text: String,
    },
}

/// How far along an operation said it was, in its own unit.
///
/// Live state, like the status and the fraction beside it: only the newest is
/// kept, and it never reaches an Action Log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Count {
    /// How many are behind it.
    pub done: u64,
    /// How many there are in total, when the operation knows.
    pub total: Option<u64>,
    /// What is being counted, singular: `"image"`, `"iteration"`.
    pub unit: &'static str,
}

/// Somewhere for one operation's events to land. Shared, never borrowed
/// mutably.
pub(crate) struct Collector {
    /// Everything the sink writes. An `Arc` because the sink closure below
    /// holds the other end of it, which is what lets [`Collector::progress`]
    /// hand out a `Progress` borrowing only `&self`.
    state: Arc<Mutex<State>>,
    /// The closure [`Progress::to`] is handed. Owned here rather than built per
    /// call, so that it can be lent for as long as the collector lives.
    sink: Box<dyn Fn(Event<'_>) + Send + Sync>,
    /// Whether `detail_phase` records, carried on every `Progress` this hands
    /// out.
    ///
    /// A field the collector is built with rather than something read from the
    /// environment: the viewer sets it per operation, so two callers in one
    /// process can legitimately differ. `SFMTOOL_PROFILE` keeps the meaning it
    /// has always had, which is the `prof` modules.
    detail: bool,
}

impl Collector {
    /// A collector for one operation, recording detailed phases or not.
    pub(crate) fn new(detail: bool) -> Self {
        let state = Arc::new(Mutex::new(State::default()));
        let written = Arc::clone(&state);
        Collector {
            state,
            sink: Box::new(move |event| lock(&written).record(event)),
            detail,
        }
    }

    /// What the operation reports through: a `Progress` at depth 0, over the
    /// whole `0..=1` range, at this collector's detail level.
    pub(crate) fn progress(&self) -> Progress<'_> {
        let sink: &(dyn Fn(Event<'_>) + Sync) = &*self.sink;
        Progress::to(sink).detailed(self.detail)
    }

    /// Time a phase of the viewer's own work, around code that takes no
    /// `Progress` of its own.
    pub(crate) fn phase(&self, name: &'static str) -> Phase<'_> {
        self.progress().phase(name)
    }

    /// Everything reported so far, in order, leaving the collector empty.
    ///
    /// The live state is left alone: it is the newest thing the operation said,
    /// not part of what it did.
    pub(crate) fn take(&self) -> Vec<Detail> {
        lock(&self.state).take()
    }

    /// What the operation last said it was doing, if anything.
    // Read by the Background panel, which is not built yet.
    #[allow(dead_code)]
    pub(crate) fn status(&self) -> Option<String> {
        lock(&self.state).status.clone()
    }

    /// The newest count the operation reported, if any.
    // Read by the Background panel, which is not built yet.
    #[allow(dead_code)]
    pub(crate) fn count(&self) -> Option<Count> {
        lock(&self.state).count
    }

    /// How far along the operation is, in `0.0..=1.0` of the whole, if it has
    /// said.
    // Read by the Background panel, which is not built yet.
    #[allow(dead_code)]
    pub(crate) fn fraction(&self) -> Option<f32> {
        lock(&self.state).fraction
    }
}

/// Take the lock, poisoned or not.
///
/// The state is rows of text and durations, so a panic part-way through one
/// event leaves it stale at worst; losing the rest of an operation's breakdown
/// to a second panic would be the larger failure.
fn lock(state: &Mutex<State>) -> std::sync::MutexGuard<'_, State> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One phase that has been entered and not yet left.
///
/// The stream is flat and carries its own depth, and a parent's `Leave` arrives
/// *after* its children's, so the enclosing phase has to be tracked as the
/// events are consumed rather than worked out afterwards. This stack is that.
///
/// The depth is never *counted* here, only compared: an event's own depth is
/// what says where it sits, and a phase still open at or below the depth of the
/// one arriving cannot be enclosing it, so it is dropped. That keeps the stack
/// honest across a `Phase::cancel`, which opens without closing. What it cannot
/// keep honest is two rayon workers opening phases at once, where one stack
/// cannot say which of them a third event is inside; the events carry no thread
/// of their own, so a parallel region's children can fold under the wrong
/// parent. The depths they are drawn at stay right, which is what the spec
/// makes the guarantee about.
struct Open {
    /// The depth the opening [`Event::Enter`] carried.
    depth: u8,
    /// Which row of [`State::rows`] the phase folds into.
    row: usize,
}

/// What fold a phase belongs to: the row of the phase enclosing it, and its
/// name.
///
/// The enclosing *row* rather than the enclosing name, because the parent has
/// already been folded by the time a child arrives. Three `round` guards are
/// one row, so the `linearise` inside each of them shares one parent and folds
/// into one row too, while the same name under two genuinely different parents
/// keys differently and stays two rows.
type Fold = (Option<usize>, &'static str);

#[derive(Default)]
struct State {
    /// The rows, in the order they first appeared.
    rows: Vec<Detail>,
    /// The phases entered and not yet left, innermost last.
    open: Vec<Open>,
    /// Which row each fold has already claimed.
    folds: HashMap<Fold, usize>,
    status: Option<String>,
    count: Option<Count>,
    fraction: Option<f32>,
}

impl State {
    fn record(&mut self, event: Event<'_>) {
        match event {
            Event::Enter { phase, depth } => self.enter(phase, depth),
            Event::Leave {
                phase,
                depth,
                took,
                note,
            } => self.leave(phase, depth, took, note),
            Event::Message { level, depth, text } => {
                // Mirrored unconditionally, as Action Log entries already are:
                // a `RUST_LOG` capture of a session is the stream of what
                // happened, and is the piece that makes these messages useful
                // outside the window.
                match level {
                    Level::Info => log::info!(target: "sfm_explorer::progress", "{text}"),
                    Level::Warn => log::warn!(target: "sfm_explorer::progress", "{text}"),
                }
                self.rows.push(Detail::Message {
                    level,
                    depth,
                    text: text.to_string(),
                });
            }
            Event::Status { text } => self.status = Some(text.to_string()),
            Event::Count { done, total, unit } => {
                self.count = Some(Count { done, total, unit });
            }
            Event::Fraction { of_whole } => {
                // The kernels guarantee only that a call cannot report outside
                // the range it was given. That the bar never runs backwards is
                // the collector's, because only the collector sees the whole
                // operation.
                let seen = self.fraction.unwrap_or(0.0);
                self.fraction = Some(of_whole.max(seen));
            }
        }
    }

    /// Open a phase, claiming the row its fold already has or starting one.
    ///
    /// The row is pushed here rather than when the phase closes, which is what
    /// puts a parent above its children: a `Leave` knows the cost but arrives
    /// after everything it contained.
    fn enter(&mut self, name: &'static str, depth: u8) {
        // Anything still open at or below this depth cannot enclose this
        // phase, so it was abandoned: `Phase::cancel` emits no `Leave`.
        while self.open.last().is_some_and(|open| open.depth >= depth) {
            self.open.pop();
        }
        let parent = self.open.last().map(|open| open.row);
        let rows = &mut self.rows;
        let row = *self.folds.entry((parent, name)).or_insert_with(|| {
            rows.push(Detail::Phase {
                name,
                depth,
                took: Duration::ZERO,
                cpu: None,
                note: None,
                runs: 0,
            });
            rows.len() - 1
        });
        self.open.push(Open { depth, row });
    }

    /// Close a phase, adding its cost to the row its fold owns.
    fn leave(&mut self, name: &'static str, depth: u8, took: Duration, note: Option<&str>) {
        // Children that never closed, for the same reason as in `enter`.
        while self.open.last().is_some_and(|open| open.depth > depth) {
            self.open.pop();
        }
        let Some(open) = self.open.last() else { return };
        if open.depth != depth {
            return;
        }
        let row = open.row;
        self.open.pop();
        let Some(Detail::Phase {
            name: folded,
            took: total,
            note: kept,
            runs,
            ..
        }) = self.rows.get_mut(row)
        else {
            return;
        };
        debug_assert_eq!(*folded, name, "a phase closed under another's row");
        *total += took;
        *runs += 1;
        // A run that says nothing leaves standing what an earlier run said, so
        // a `reused` is not erased by the next trip through the loop.
        if let Some(note) = note {
            *kept = Some(note.to_string());
        }
    }

    /// Everything recorded, leaving the state empty.
    ///
    /// A phase that was entered and never left is dropped. The row exists
    /// because a row is claimed on the way in, which is what puts a parent
    /// above the children it contains, but `Phase::cancel` says the phase
    /// did not run and a row reading zero runs in zero time is noise rather
    /// than a finding. Anything it did contain keeps its own row, which is
    /// the honest reading: that work happened.
    fn take(&mut self) -> Vec<Detail> {
        self.open.clear();
        self.folds.clear();
        let mut rows = std::mem::take(&mut self.rows);
        rows.retain(|row| !matches!(row, Detail::Phase { runs: 0, .. }));
        rows
    }
}
