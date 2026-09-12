// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Action Log: a timestamped record of every action taken in the viewer,
//! by whoever took it.
//!
//! See `specs/gui/action-log.md`. The buffer lives on [`crate::state::AppState`]
//! as `action_log`, so every place that already holds the state to mutate holds
//! the log to write; the panel that shows it is [`show`], docked as
//! `Tab::ActionLog` beside the Image Browser.
//!
//! Three things about the shape here are worth knowing before reading it:
//!
//! - **The actor is ambient, not an argument.** Every mutating `AppState`
//!   method would otherwise take an [`Actor`] for the benefit of exactly one
//!   caller that is not the user. Instead the MCP drain sets [`Actor::Mcp`]
//!   before applying a frame's commands and restores [`Actor::User`] after.
//! - **Muting is a depth counter.** Composite actions nest — `close_all` over
//!   `close_node`, `resect_image` over `append_node` — and each layer only
//!   knows about itself.
//! - **Coalescing happens at record time and is not reversible.** Successive
//!   values of one [`Run`] — a control, a selection slot, a polled query tool —
//!   fold into one line, so that a scrub or a slider drag stays one line. An
//!   entry with no run is a discrete act and never folds, and neither does a
//!   failure, which is why the log is a readable record rather than an audit
//!   trail.

// Several things here exist for the MCP surface, which is a Cargo feature: the
// wire spellings of a kind and an actor, the revision clock `get_action_log`
// reads, and the `query` entry the drain writes. In a `--no-default-features`
// build nothing calls them, which is a property of that build rather than a
// loose end here.
#![cfg_attr(not(feature = "mcp"), allow(dead_code))]

use std::collections::{HashSet, VecDeque};

use std::time::{Duration, Instant};

use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use sfmtool_core::progress::Level;

use crate::progress::Detail;

mod panel;

#[cfg(test)]
mod tests;

pub(crate) use panel::show;
// The row vocabulary of a breakdown, which the Background panel draws too: the
// live form of an entry's detail and the entry's own are one thing said once.
pub(crate) use panel::{detail_row, detail_text, Breakdown};

/// Who took an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Actor {
    /// Someone at the window: a click, a key, a menu item, the command line.
    User,
    /// An agent, through the `--mcp` endpoint.
    Mcp,
    /// The viewer on its own: startup, an animation reaching its end.
    Viewer,
}

impl Actor {
    /// Every actor, in the order an error message lists them.
    pub(crate) const ALL: [Actor; 3] = [Actor::User, Actor::Mcp, Actor::Viewer];

    /// The word the actor column shows, and the clipboard export writes.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Actor::User => "User",
            Actor::Mcp => "MCP",
            Actor::Viewer => "Viewer",
        }
    }

    /// The actor's name on the wire: its column word, lower-cased.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Actor::User => "user",
            Actor::Mcp => "mcp",
            Actor::Viewer => "viewer",
        }
    }

    /// The actor `name` spells, or `None`. Exact: `"User"` is not an actor.
    pub(crate) fn from_wire_name(name: &str) -> Option<Actor> {
        Actor::ALL
            .into_iter()
            .find(|actor| actor.wire_name() == name)
    }

    /// Every actor name, as an error message lists them.
    pub(crate) fn all_wire_names() -> String {
        Actor::ALL
            .iter()
            .map(|actor| actor.wire_name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What an entry is about: the panel's tooltip word and the wire's `kind`.
///
/// Says nothing about folding — that is [`Run`]'s job, and the run is finer
/// than the kind: two `Display` entries are the same kind whether or not they
/// are the same control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The viewer starting, and the endpoints it brought up with it.
    Session,
    /// What is loaded: opened, closed.
    File,
    /// Which reconstruction, image, camera or point is selected.
    Selection,
    /// The scene graph: visibility, tint, solo, alignment, resection.
    Scene,
    /// An edit to a reconstruction, and the undo or redo of one. See
    /// [`crate::document`].
    Edit,
    /// Where the 3D camera is looking.
    View,
    /// A viewport HUD control.
    Display,
    /// Image-strip playback.
    Animation,
    /// Which panels are docked where: a panel opened, closed or raised, and
    /// the layout reset, saved or loaded. See [`crate::layout`].
    Layout,
    /// The window itself: the state, size, position and focus one `set_window`
    /// call changed.
    Window,
    /// A read-only MCP tool, by name.
    Query(&'static str),
}

impl Kind {
    /// The word the row tooltip shows, which for a query is the tool's name.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Kind::Session => "Session",
            Kind::File => "File",
            Kind::Selection => "Selection",
            Kind::Scene => "Scene",
            Kind::Edit => "Edit",
            Kind::View => "View",
            Kind::Display => "Display",
            Kind::Animation => "Animation",
            Kind::Layout => "Layout",
            Kind::Window => "Window",
            Kind::Query(tool) => tool,
        }
    }

    /// The kind's name on the wire: [`Kind::label`] lower-cased, with every
    /// query under the one word `query`.
    ///
    /// A query's tool travels beside the kind rather than inside it — a reply
    /// row carries `"kind": "query"` and `"tool": "get_scene"` — so that an
    /// agent filtering or grouping by kind has a closed vocabulary rather than
    /// one that grows with the tool table.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Kind::Session => "session",
            Kind::File => "file",
            Kind::Selection => "selection",
            Kind::Scene => "scene",
            Kind::Edit => "edit",
            Kind::View => "view",
            Kind::Display => "display",
            Kind::Animation => "animation",
            Kind::Layout => "layout",
            Kind::Window => "window",
            Kind::Query(_) => "query",
        }
    }
}

/// The one thing an entry is a successive value *of* — a control's label, a
/// selection slot, a query tool — or `None` for a discrete act.
///
/// Two entries with the same run, kind and actor inside
/// [`ActionLog::COALESCE_WINDOW`] fold into one line; an entry with no run
/// never folds, in either direction. The run is finer than the [`Kind`] on
/// purpose: folding by kind alone took every `Display` entry within a second
/// to be the same gesture, so a call that changed three fields kept only the
/// third field's row. A fold is for a widget or a slot being dragged through
/// intermediate values, and two different controls are two acts however close
/// together. The catalogue in `specs/gui/action-log.md` gives every row its
/// run.
///
/// A `&'static str` because every run is a literal at its call site — the
/// HUD's control labels, the selection slots, the tool table's names — and the
/// log compares them, never composes them.
pub(crate) type Run = Option<&'static str>;

/// What a write is recording, past the fields every entry carries.
///
/// The two halves of "this already happened": when it began, and what it
/// reported while it ran. An action the viewer performs as it records it has
/// nothing to report and began just now, which is [`Work::just_now`].
struct Work {
    /// When the work being recorded began, which is what the entry's cost is
    /// measured from.
    started: Instant,
    /// What it reported, before the cap at [`ActionLog::DETAIL_EVENTS`].
    detail: Vec<Detail>,
}

impl Work {
    /// An action with no breakdown, happening as it is recorded.
    fn just_now() -> Self {
        Work {
            started: Instant::now(),
            detail: Vec::new(),
        }
    }
}

/// One line of the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    /// The revision at which this entry was written or last replaced.
    ///
    /// Strictly increasing along the buffer, because the only two writes are a
    /// push of a fresh highest revision and a coalescing replacement of the
    /// newest entry with one. That ordering is what lets
    /// [`ActionLog::since`] stop as soon as it reaches an entry the caller has
    /// already seen.
    pub revision: u64,
    /// When it was recorded. Wall clock, because the log's job is to be read
    /// next to something else — an agent's transcript, a CI log, a screen
    /// recording — and only wall-clock time lines those up.
    pub at: Timestamp,
    pub actor: Actor,
    pub kind: Kind,
    /// What this entry folds with, or `None` for a discrete act.
    ///
    /// Not on the wire: an agent reads rows, and which rows folded is already
    /// visible in the revisions it did not see.
    pub run: Run,
    /// Whether the action was refused or failed. A failed entry never
    /// coalesces, in either direction.
    pub failed: bool,
    pub text: String,
    /// How long the viewer took to show this action's result, or `None` while
    /// it has not been shown yet.
    ///
    /// Wall time from the moment the action was recorded to the end of the
    /// first frame drawn *after* the upload phase saw it -- which is the wait
    /// the person who took it actually sits through, and not the cost of the
    /// state change alone. A command an agent sent is applied before that
    /// phase and so settles in its own frame; a click handled in the egui pass
    /// runs after it and settles in the next one. Either way the number covers
    /// the GPU uploads and the draw the action caused.
    ///
    /// `None` also for an entry whose run folded before a frame could settle
    /// it: the fold gives the row a new revision, and the timing follows the
    /// row that survived.
    pub took: Option<Duration>,
    /// What the operation reported, in the order it reported it.
    ///
    /// Empty for the great majority of entries, which are a state change too
    /// small to have stages. An operation that took long enough to be worth
    /// breaking down hands over the [`crate::progress::Collector`] it reported
    /// into, and these are its rows, already folded and capped at
    /// [`ActionLog::DETAIL_EVENTS`].
    pub detail: Vec<Detail>,
}

/// The buffer, the recording rules, and the local zone the panel formats in.
pub(crate) struct ActionLog {
    entries: VecDeque<Entry>,
    /// Who subsequent entries are attributed to. `User` unless the MCP drain
    /// has said otherwise for the duration of a frame's commands.
    actor: Actor,
    /// How many nested `mute`s are outstanding. Zero means recording.
    mute: usize,
    /// The system zone, resolved once at construction rather than per row: a
    /// lookup per entry per frame would re-read `/etc/localtime` on Linux.
    zone: TimeZone,
    /// How many entries have been dropped off the front at [`ActionLog::CAPACITY`].
    dropped: usize,
    /// Entries written but not yet timed, as `(revision, when it was written)`.
    ///
    /// A frame settles the ones the upload phase had already seen, so this holds
    /// at most the writes of one frame in the steady state. It is still bounded
    /// ([`ActionLog::PENDING_CAPACITY`]) because nothing outside the viewer's
    /// frame loop calls [`ActionLog::settle`] -- a headless test, or a
    /// `--no-default-features` build with no window, would otherwise grow it
    /// without limit.
    pending: Vec<(u64, Instant)>,
    /// The revisions whose detail the panel is showing.
    ///
    /// Here rather than in egui's memory because it is the one piece of panel
    /// state a headless test needs to drive, and the panel already takes a
    /// `&mut ActionLog`. Keyed on the revision, so an expansion survives
    /// entries dropping at [`ActionLog::CAPACITY`] and means nothing once its
    /// entry is gone.
    expanded: HashSet<u64>,
    /// Whether the next operation records its detailed phases.
    ///
    /// Here beside the expansion set, for the same reason: it is
    /// panel-adjacent state a headless test has to drive, and keeping it here
    /// is what leaves [`show`] taking a `&mut ActionLog` and nothing else. It
    /// is read when an operation builds its collector, so a change takes
    /// effect on the next operation and nothing already recorded is re-timed.
    detailed_timing: bool,
    /// The log's clock: one tick per write, whether the write appended an
    /// entry or folded into the newest one.
    ///
    /// A timestamp could not stand in for it — two entries can share an
    /// instant, and a fold moves an entry's time to the newest of the run,
    /// which is precisely the change a reader has to be told about. Never
    /// reset, [`ActionLog::clear`] included, so a revision an agent is holding
    /// stays comparable for the life of the session.
    revision: u64,
}

impl ActionLog {
    /// Entries kept. Past this the oldest goes and the toolbar reports how many
    /// have been dropped.
    pub(crate) const CAPACITY: usize = 10_000;

    /// Writes held waiting to be timed. A frame settles what it saw, so this is
    /// a bound on a queue that is otherwise a frame deep -- it matters only when
    /// nothing is calling [`ActionLog::settle`] at all.
    const PENDING_CAPACITY: usize = 1_024;

    /// Largest gap between two like entries for the newer to replace the older.
    pub(crate) const COALESCE_WINDOW: SignedDuration = SignedDuration::from_secs(1);

    /// Detail rows kept per entry. Past this the remainder is dropped and a
    /// final line says how many.
    ///
    /// Applied after the collector has folded, so it bounds the rows an entry
    /// carries rather than the phases the operation opened: a kernel that
    /// opens one guard six hundred times leaves one row and cannot push the
    /// rest of the operation out of its own entry.
    pub(crate) const DETAIL_EVENTS: usize = 128;

    /// The row an entry gathers the frame's own stages under.
    ///
    /// Named for what it is from the operation's point of view. Putting a
    /// reconstruction on the screen is not part of reading it off the disk,
    /// and a reader who takes the upload for the load draws the wrong
    /// conclusion about where a slow action went. It is the last thing an
    /// expanded entry shows, under a rule, after everything the operation
    /// itself accounted for.
    pub(crate) const OVERHEAD: &'static str = "overhead: uploading and drawing";

    /// A log formatting in the system's local time zone.
    pub(crate) fn new() -> Self {
        Self::with_zone(TimeZone::system())
    }

    /// A log formatting in `zone`. The tests use a fixed one so that what a row
    /// reads is not a property of the machine running them.
    pub(crate) fn with_zone(zone: TimeZone) -> Self {
        Self {
            entries: VecDeque::new(),
            actor: Actor::User,
            mute: 0,
            zone,
            dropped: 0,
            pending: Vec::new(),
            expanded: HashSet::new(),
            detailed_timing: false,
            revision: 0,
        }
    }

    /// Record a discrete action as the current actor, now. Never coalesces.
    pub(crate) fn record(&mut self, kind: Kind, text: impl Into<String>) {
        self.record_at(Timestamp::now(), kind, None, false, text);
    }

    /// Record one value of `run` as the current actor, now: folds into the
    /// newest entry when that entry is the same run.
    pub(crate) fn record_run(&mut self, kind: Kind, run: &'static str, text: impl Into<String>) {
        self.record_at(Timestamp::now(), kind, Some(run), false, text);
    }

    /// Record a failed action as the current actor, now. Never coalesces.
    pub(crate) fn fail(&mut self, kind: Kind, text: impl Into<String>) {
        self.record_at(Timestamp::now(), kind, None, true, text);
    }

    /// Record one entry as `actor`, restoring the standing one afterwards.
    ///
    /// The viewer's own entries — startup, the endpoint coming up, an animation
    /// running out of images — are the only ones that are not the standing
    /// actor's, and they are all single lines, so they say so here rather than
    /// by moving the ambient actor and having to remember to move it back.
    pub(crate) fn record_as(&mut self, actor: Actor, kind: Kind, text: impl Into<String>) {
        let standing = self.actor;
        self.actor = actor;
        self.record(kind, text);
        self.actor = standing;
    }

    /// Record a failed action as `actor`, restoring the standing one
    /// afterwards.
    ///
    /// A background operation's refusal, which lands frames or minutes after
    /// the call that asked for it: whoever asked is the actor of the row,
    /// whoever happens to be standing when it arrives is not.
    /// A refusal an operation reached after doing the work, as `actor` and
    /// timed from `started`.
    ///
    /// The failing counterpart of [`ActionLog::record_done_as`], and it exists
    /// for the same reason: a solve cancelled seventeen seconds in spent those
    /// seventeen seconds, and a row costing it at the two milliseconds of the
    /// frame that gave up is the sort of number that discredits a whole column.
    /// A refusal reached without doing any work is [`ActionLog::fail`].
    pub(crate) fn fail_done_as(
        &mut self,
        actor: Actor,
        kind: Kind,
        started: Instant,
        text: impl Into<String>,
        detail: Vec<Detail>,
    ) {
        let standing = self.actor;
        self.actor = actor;
        self.write(
            Timestamp::now(),
            kind,
            None,
            true,
            text,
            Work { started, detail },
        );
        self.actor = standing;
    }

    /// [`ActionLog::record_done`] as `actor`, restoring the standing one
    /// afterwards.
    ///
    /// The pair a background operation writes its outcome with: the cost is
    /// the operation's, measured from `started`, and the actor is the one who
    /// asked for it however long ago.
    pub(crate) fn record_done_as(
        &mut self,
        actor: Actor,
        kind: Kind,
        started: Instant,
        text: impl Into<String>,
        detail: Vec<Detail>,
    ) {
        let standing = self.actor;
        self.actor = actor;
        self.record_done(kind, started, text, detail);
        self.actor = standing;
    }

    /// Record a read-only MCP tool call, coalescing per tool: the tool names
    /// both the kind and the run, so a poll is one row however often it asks.
    ///
    /// `screenshot` does not come through here — it is recorded with
    /// [`ActionLog::record`], as the discrete act it is.
    pub(crate) fn query(&mut self, tool: &'static str, text: impl Into<String>) {
        self.record_at(Timestamp::now(), Kind::Query(tool), Some(tool), false, text);
    }

    /// Record a widget's change as one value of `run` — the control's label —
    /// building the text only then.
    ///
    /// The whole of what a HUD checkbox or slider needs: those write straight
    /// into the state they govern, so `changed()` is the only signal that a
    /// frame's value is a new one.
    ///
    /// **`changed()` alone is not enough.** An `egui::Slider` clamps and rounds
    /// the value it was handed, and reports that as a change — so the Scene
    /// slider, first drawn over a `length_scale` the upload phase had just
    /// derived, announced `Scene scale 0.655` before anyone had touched it. An
    /// action needs an actor, so a change arriving with neither the pointer nor
    /// the keyboard on the widget is not one.
    pub(crate) fn changed(
        &mut self,
        response: &egui::Response,
        kind: Kind,
        run: &'static str,
        text: impl FnOnce() -> String,
    ) {
        let touched = response.is_pointer_button_down_on()
            || response.clicked()
            || response.drag_stopped()
            || response.has_focus();
        if response.changed() && touched {
            self.record_run(kind, run, text());
        }
    }

    /// The `record` / `record_run` / `fail` primitive with an explicit instant.
    ///
    /// Public to the crate for the tests, which drive the coalescing window and
    /// the formatting with fixed instants rather than the wall clock.
    pub(crate) fn record_at(
        &mut self,
        at: Timestamp,
        kind: Kind,
        run: Run,
        failed: bool,
        text: impl Into<String>,
    ) {
        self.write(at, kind, run, failed, text, Work::just_now());
    }

    /// Record an entry for work that has already happened: `started` is when
    /// that work began, `detail` what it reported.
    ///
    /// The one call a long operation uses. `started` is what makes a two-minute
    /// solve report two minutes rather than the six milliseconds of the frame
    /// that installed it; the entry still settles on the frame that draws the
    /// result, like every other.
    ///
    /// A discrete act, like [`ActionLog::record`]: an operation big enough to
    /// have a breakdown is not a value of a run.
    pub(crate) fn record_done(
        &mut self,
        kind: Kind,
        started: Instant,
        text: impl Into<String>,
        detail: Vec<Detail>,
    ) {
        self.write(
            Timestamp::now(),
            kind,
            None,
            false,
            text,
            Work { started, detail },
        );
    }

    /// The one write. Every recording method above arrives here.
    fn write(
        &mut self,
        at: Timestamp,
        kind: Kind,
        run: Run,
        failed: bool,
        text: impl Into<String>,
        work: Work,
    ) {
        let Work { started, detail } = work;
        if self.mute > 0 {
            return;
        }
        // One tick per write, in both branches below: an entry a run folded
        // into is as new to a reader as one that was appended.
        self.revision += 1;
        let entry = Entry {
            revision: self.revision,
            at,
            actor: self.actor,
            kind,
            run,
            failed,
            text: text.into(),
            took: None,
            detail: Self::capped(detail),
        };
        // The mirror is unconditional and precedes coalescing: a `RUST_LOG`
        // capture is the stream of what happened, and folding a run away is a
        // property of the panel's readability, not of the session.
        log::info!(target: "sfm_explorer::action_log", "{}", self.line(&entry));
        if self.coalesce(&entry) {
            // The whole entry is replaced, detail included: a folded row is the
            // newest value of the run, so it carries what that value reported
            // and not what the value it replaced did.
            *self
                .entries
                .back_mut()
                .expect("coalesce found a last entry") = entry;
            self.await_timing(started);
            return;
        }
        if self.entries.len() >= Self::CAPACITY {
            self.entries.pop_front();
            self.dropped += 1;
        }
        self.entries.push_back(entry);
        self.await_timing(started);
    }

    /// At most [`ActionLog::DETAIL_EVENTS`] rows, plus a line saying how many
    /// were dropped.
    ///
    /// The first rows rather than the last: an operation's shape is in the
    /// order it did things, and a breakdown truncated at the front would say
    /// nothing about where it started.
    fn capped(mut detail: Vec<Detail>) -> Vec<Detail> {
        if detail.len() <= Self::DETAIL_EVENTS {
            return detail;
        }
        let dropped = detail.len() - Self::DETAIL_EVENTS;
        detail.truncate(Self::DETAIL_EVENTS);
        detail.push(Detail::Message {
            level: Level::Info,
            depth: 0,
            text: format!("{dropped} more events dropped"),
        });
        detail
    }

    /// Put the entry just written in the queue waiting to be timed, from
    /// `started`.
    ///
    /// `started` is the write itself for an action the viewer performs as it
    /// records it, and the moment the work began for one recorded with
    /// [`ActionLog::record_done`] after the fact. Either way what is stamped is
    /// the wait, from the first thing that happened to the frame that showed
    /// the result.
    ///
    /// After the write in both branches of `write`, coalescing included: a
    /// fold mints a new revision, so the folded row is the one that gets timed
    /// and the value it replaced drops out of the queue unstamped.
    fn await_timing(&mut self, started: Instant) {
        if self.pending.len() >= Self::PENDING_CAPACITY {
            self.pending.remove(0);
        }
        self.pending.push((self.revision, started));
    }

    /// How many writes are waiting to be timed. For the test that holds the
    /// bound on a log nothing is settling.
    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Stamp every entry the frame that is ending had already written before
    /// its upload phase, with the time from the write to now.
    ///
    /// `uploads_began` is when this frame's upload phase started. An entry
    /// written before it has had its GPU work done and its pixels drawn by the
    /// time this is called, so the elapsed time is the whole wait. One written
    /// after it -- anything the egui pass handled, which is every click and
    /// keystroke -- has not, and waits for the next frame.
    ///
    /// `frame` is what the frame itself reported: the uploads, the two draws
    /// and the present. They go to the newest entry this stamps, which is the
    /// action whose effect the frame was uploading, and that entry says how
    /// many others waited alongside it. A frame that stamps nothing has nobody
    /// to charge and drops them.
    ///
    /// Called once per frame from the viewer's frame loop, at the end.
    ///
    /// One consequence looks like a bug and is not. An entry written during the
    /// egui pass (which is every click, every key and every menu item) does
    /// not carry the tail of the frame it was written in: it was written after
    /// that frame's upload phase, so it is not stamped until the *next* frame,
    /// and the next frame is the one that shows its result. The millisecond or
    /// two of its own frame lands in `elsewhere`.
    pub(crate) fn settle(&mut self, uploads_began: Instant, frame: Vec<Detail>) {
        if self.pending.is_empty() {
            return;
        }
        let now = Instant::now();
        let mut still_waiting = Vec::new();
        let mut stamped = Vec::new();
        for (revision, written) in std::mem::take(&mut self.pending) {
            if written > uploads_began {
                still_waiting.push((revision, written));
                continue;
            }
            // Found by revision rather than by position: entries drop off the
            // front at capacity, and a row whose run folded is gone under a
            // revision of its own.
            if let Some(entry) = self
                .entries
                .iter_mut()
                .rev()
                .find(|entry| entry.revision == revision)
            {
                entry.took = Some(now.duration_since(written));
                stamped.push(revision);
            }
        }
        self.pending = still_waiting;
        self.share_frame(&stamped, frame);
    }

    /// Give the frame's own events to the one entry they belong to.
    ///
    /// That is the newest entry the frame stamped, because the uploads reflect
    /// the state as of the last action recorded before they began: the earlier
    /// entries a frame settles did not cause its work, they waited through it.
    /// Copying the table onto all of them instead would make a log of a
    /// startup read as though the file had been opened three times, once per
    /// row that happened to be pending. Those rows keep an honest `took`, which
    /// is the wait they really had, and say nothing about work that was not
    /// theirs; this one says how many waited with it.
    fn share_frame(&mut self, stamped: &[u64], frame: Vec<Detail>) {
        let Some(&newest) = stamped.iter().max() else {
            return;
        };
        if frame.is_empty() {
            return;
        }
        let waited = stamped.len() - 1;
        // From the back: what a frame stamps was written in it, so the entry
        // it belongs to is among the newest there are.
        let Some(entry) = self
            .entries
            .iter_mut()
            .rev()
            .find(|entry| entry.revision == newest)
        else {
            return;
        };
        let mut detail = std::mem::take(&mut entry.detail);
        detail.extend(Self::under_one_row(frame, waited));
        // Capped again rather than trusted: the write capped what the
        // operation reported, and the frame's events arrive after it.
        entry.detail = Self::capped(detail);
    }

    /// The frame's rows, gathered under one [`Self::OVERHEAD`] row a level above
    /// them.
    ///
    /// Uploading a reconstruction to the GPU is not part of reading it off the
    /// disk. It happens because the document changed and it happens in the next
    /// frame, since that is where uploads happen, and it is inside the wait the
    /// entry reports, which is why one entry carries both. It is still not the
    /// operation, so it is named as the overhead it is and the panel rules a
    /// line above it.
    ///
    /// That row costs what its own stages cost between them, so `elsewhere` is
    /// unchanged by the gathering and the entry still reconciles with its own
    /// headline.
    ///
    /// How many entries only waited for this frame is the row's note rather
    /// than a line of its own: it is a fact about the frame, and it belongs
    /// beside it.
    fn under_one_row(frame: Vec<Detail>, waited: usize) -> Vec<Detail> {
        let took = frame
            .iter()
            .filter_map(|row| match row {
                Detail::Phase { depth: 0, took, .. } => Some(*took),
                _ => None,
            })
            .sum();
        let note = (waited > 0).then(|| {
            let entries = if waited == 1 { "entry" } else { "entries" };
            format!("also settled {waited} earlier {entries}")
        });
        let mut rows = Vec::with_capacity(frame.len() + 1);
        rows.push(Detail::Phase {
            name: Self::OVERHEAD,
            depth: 0,
            took,
            cpu: None,
            note,
            note_last: None,
            runs: 1,
        });
        rows.extend(frame.into_iter().map(|row| match row {
            Detail::Phase {
                name,
                depth,
                took,
                cpu,
                note,
                note_last,
                runs,
            } => Detail::Phase {
                name,
                depth: depth.saturating_add(1),
                took,
                cpu,
                note,
                note_last,
                runs,
            },
            Detail::Message { level, depth, text } => Detail::Message {
                level,
                depth: depth.saturating_add(1),
                text,
            },
        }));
        rows
    }

    /// Whether `entry` should replace the newest entry rather than follow it.
    ///
    /// The [`Run`] decides, not the kind: both entries must name the same one,
    /// so an entry with no run neither folds nor is folded into.
    ///
    /// The window is measured from the entry being *replaced*, which is itself
    /// the time of the last replacement — so an unbroken run folds
    /// indefinitely, and a pause longer than the window starts a new line.
    fn coalesce(&self, entry: &Entry) -> bool {
        let Some(last) = self.entries.back() else {
            return false;
        };
        !entry.failed
            && !last.failed
            && entry.run.is_some()
            && last.run == entry.run
            && last.kind == entry.kind
            && last.actor == entry.actor
            && entry.at >= last.at
            && entry.at.duration_since(last.at) <= Self::COALESCE_WINDOW
    }

    /// Who subsequent entries are attributed to. [`Actor::User`] by default.
    pub(crate) fn set_actor(&mut self, actor: Actor) {
        self.actor = actor;
    }

    /// Who entries are being attributed to right now.
    pub(crate) fn actor(&self) -> Actor {
        self.actor
    }

    /// Suppress recording until the matching [`ActionLog::unmute`]. Nests.
    pub(crate) fn mute(&mut self) {
        self.mute += 1;
    }

    /// Undo one [`ActionLog::mute`].
    pub(crate) fn unmute(&mut self) {
        self.mute = self.mute.saturating_sub(1);
    }

    /// Every entry, oldest first. Reversible, which is how the status line
    /// finds the newest one that is not a query.
    pub(crate) fn entries(&self) -> impl ExactSizeIterator<Item = &Entry> + DoubleEndedIterator {
        self.entries.iter()
    }

    /// The counter now: the revision of the newest write.
    ///
    /// Sent to an agent as `revision` and handed back as `since_revision`, so
    /// that a reader can ask for exactly what it has not seen.
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// The entries whose revision is above `since`, oldest first.
    ///
    /// Walks from the back and stops at the first entry the caller has already
    /// seen, which entry order makes sound and which makes a poll that finds
    /// nothing O(1).
    pub(crate) fn since(&self, since: u64) -> impl ExactSizeIterator<Item = &Entry> {
        let fresh = self
            .entries
            .iter()
            .rev()
            .take_while(|entry| entry.revision > since)
            .count();
        self.entries.iter().skip(self.entries.len() - fresh)
    }

    /// The revision of the oldest entry still held, or [`ActionLog::revision`]
    /// when the log is empty.
    ///
    /// A reader whose `since_revision` is below this has missed entries — to
    /// [`ActionLog::CAPACITY`], or to the toolbar's Clear — and can say so
    /// rather than assume the gap was quiet.
    pub(crate) fn oldest_revision(&self) -> u64 {
        self.entries
            .front()
            .map(|entry| entry.revision)
            .unwrap_or(self.revision)
    }

    /// One entry by position, oldest first — what the virtualized list asks
    /// for, since it lays out a slice of the rows and not all of them.
    pub(crate) fn get(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index)
    }

    /// How many entries are held.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// How many entries have been dropped off the front at capacity.
    pub(crate) fn dropped(&self) -> usize {
        self.dropped
    }

    /// Empty the log. Also empties the viewport status line, which is a view of
    /// it.
    ///
    /// The revision counter is left alone: an agent holding a revision from
    /// before a Clear should learn that entries went missing, which
    /// [`ActionLog::oldest_revision`] tells it, rather than be handed a
    /// rewound clock that makes the gap look quiet.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.dropped = 0;
        // The entries those expansions were holding open are gone.
        self.expanded.clear();
    }

    /// Whether the panel is showing `revision`'s detail.
    pub(crate) fn is_expanded(&self, revision: u64) -> bool {
        self.expanded.contains(&revision)
    }

    /// Show `revision`'s detail, or stop showing it.
    pub(crate) fn toggle_expanded(&mut self, revision: u64) {
        if !self.expanded.remove(&revision) {
            self.expanded.insert(revision);
        }
    }

    /// Every revision the panel is showing the detail of, in no order.
    ///
    /// Some of them may name entries that are no longer held, since an
    /// expansion outlives its entry dropping at [`ActionLog::CAPACITY`], so a
    /// reader puts them through [`ActionLog::index_of`] rather than trusting
    /// the set.
    pub(crate) fn expanded_revisions(&self) -> impl Iterator<Item = u64> + '_ {
        self.expanded.iter().copied()
    }

    /// Where `revision` sits in the buffer, or `None` when it is no longer
    /// held.
    ///
    /// A binary search, which is sound because revisions increase strictly
    /// along the buffer, the same property [`ActionLog::since`] stops on.
    pub(crate) fn index_of(&self, revision: u64) -> Option<usize> {
        self.entries
            .binary_search_by_key(&revision, |entry| entry.revision)
            .ok()
    }

    /// Whether the next operation records its detailed phases.
    pub(crate) fn detailed_timing(&self) -> bool {
        self.detailed_timing
    }

    /// Turn detailed phases on or off for the operations to come, recording
    /// the change.
    ///
    /// A `Display` entry like any other control's, so the log says when the
    /// level changed and who changed it, and nothing at all when the value
    /// handed over is the one already standing.
    pub(crate) fn set_detailed_timing(&mut self, on: bool) {
        if on == self.detailed_timing {
            return;
        }
        self.detailed_timing = on;
        let state = if on { "on" } else { "off" };
        self.record_run(
            Kind::Display,
            "Detailed timing",
            format!("Detailed timing {state}"),
        );
    }

    /// `took` minus the entry's top-level wall-clock phases, or `None` when it
    /// has no cost yet or no phases.
    ///
    /// The line that makes an expanded entry add up: work nobody has named
    /// shows as a gap rather than as silence, so the breakdown reconciles with
    /// the number in the entry's own cost column by construction.
    ///
    /// Only depth 0 is subtracted, because a nested phase's cost is already
    /// inside its parent's. A `cpu` figure is not subtracted at all: eight
    /// seconds of thread-summed CPU inside one second of wall is not eight
    /// seconds of anybody's wait, and folding it in would be lying about the
    /// sum. The result is clamped at zero rather than allowed to go negative:
    /// the phases and the settle read the clock at different points, and a
    /// folded row sums wall times that may have overlapped on different rayon
    /// threads.
    pub(crate) fn elsewhere(entry: &Entry) -> Option<Duration> {
        let took = entry.took?;
        let mut named = Duration::ZERO;
        let mut any = false;
        for detail in &entry.detail {
            if let Detail::Phase { depth: 0, took, .. } = detail {
                named += *took;
                any = true;
            }
        }
        any.then(|| took.saturating_sub(named))
    }

    /// An entry's breakdown in the order the panel draws it, which is not the
    /// order it was recorded in.
    ///
    /// [`ActionLog::elsewhere`] closes the operation's own account before the
    /// overhead the frame charged, so `panel::detail_row` reorders, and a
    /// reader that carries the breakdown without drawing it has to reorder the
    /// same way. Reading the order off the panel rather than restating it is
    /// what keeps the window and the wire from ever disagreeing about what an
    /// operation did. `elsewhere` itself is not in here: it is a row of the
    /// breakdown rather than an event of it, and the wire carries it as a
    /// field of its own.
    pub(crate) fn detail_in_draw_order(entry: &Entry) -> impl Iterator<Item = &Detail> + '_ {
        let breakdown = panel::Breakdown::of(entry);
        (0..panel::detail_rows(&breakdown))
            .filter_map(|row| panel::detail_at(&breakdown, row))
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|index| entry.detail.get(index))
    }

    /// Every row of an entry's breakdown as the panel draws it, indent and
    /// marker included, `elsewhere` among them.
    ///
    /// For the tests that hold the MCP surface to what the window shows: the
    /// wire carries the same events in the same order, and the two disagreeing
    /// would mean one of them is lying to somebody.
    #[cfg(test)]
    pub(crate) fn drawn_detail(entry: &Entry) -> Vec<String> {
        let breakdown = panel::Breakdown::of(entry);
        (0..panel::detail_rows(&breakdown))
            .map(|row| panel::detail_text(&panel::detail_row(&breakdown, row)))
            .collect()
    }

    /// What a phase's note reads as: what it said, or both ends where a folded
    /// row's runs disagreed, `first ... last`.
    ///
    /// One spelling, because the panel draws it and the MCP surface carries
    /// it: a reader comparing the two should not find two different claims
    /// about what a stage did.
    pub(crate) fn note_text(note: Option<&str>, note_last: Option<&str>) -> Option<String> {
        match (note, note_last) {
            (Some(first), Some(last)) => Some(format!("{first} ... {last}")),
            (Some(only), None) => Some(only.to_string()),
            (None, _) => None,
        }
    }

    /// The most recent entry that is not a successful query, as the viewport
    /// status line shows it: prefixed `MCP: ` when its actor is [`Actor::Mcp`].
    ///
    /// A successful read is skipped so that an agent polling `get_scene` does
    /// not read its own polling back as the viewer's status; a *failed* read is
    /// not, because a refusal is something the person at the window should see
    /// whichever tool it came from.
    ///
    /// A fresh `String` rather than a stored one: it is built once a frame and
    /// prefixes at most five bytes, where keeping the prefixed form would mean
    /// two copies of every text.
    pub(crate) fn status_line(&self) -> Option<String> {
        let entry = self
            .entries()
            .rev()
            .find(|entry| entry.failed || !matches!(entry.kind, Kind::Query(_)))?;
        Some(match entry.actor {
            Actor::Mcp => format!("MCP: {}", entry.text),
            _ => entry.text.clone(),
        })
    }

    /// The whole log as clipboard text, one line per entry, with the detail of
    /// every expanded entry indented under it.
    ///
    /// A collapsed entry writes its own line and nothing else, so what Copy
    /// puts on the clipboard is what the panel is showing.
    pub(crate) fn to_clipboard_text(&self) -> String {
        let mut out = String::new();
        for entry in self.entries() {
            out.push_str(&self.line(entry));
            out.push('\n');
            if !self.is_expanded(entry.revision) {
                continue;
            }
            for row in 0..panel::detail_rows(&panel::Breakdown::of(entry)) {
                out.push_str(&Self::detail_line(entry, row));
                out.push('\n');
            }
        }
        out
    }

    /// One row of an expanded entry's breakdown, in the columns
    /// [`ActionLog::line`] lays out, so that a phase's cost lands under its
    /// entry's on the clipboard as it does in the panel.
    fn detail_line(entry: &Entry, row: usize) -> String {
        let row = panel::detail_row(&panel::Breakdown::of(entry), row);
        // The rule the panel paints above the overhead, in the one spelling a
        // text buffer has for it. Without it a pasted breakdown reads as though
        // the operation did the uploading.
        let rule = if row.rules_above {
            format!("{:19}  {:<6}  {:>7}  ---\n", "", "", "")
        } else {
            String::new()
        };
        format!(
            "{rule}{:19}  {:<6}  {:>7}  {}",
            "",
            "",
            row.cost,
            panel::detail_text(&row),
        )
    }

    /// One entry as the clipboard and the `log::info!` mirror render it: the
    /// full date, the actor, a `!` where colour would have said "failed", how
    /// long it took, and the text.
    ///
    /// The duration column is blank in the mirror and filled in the clipboard,
    /// and that is not an inconsistency: the mirror is written at the moment of
    /// the action, when what it cost is still in the future.
    pub(crate) fn line(&self, entry: &Entry) -> String {
        format!(
            "{}  {:<6}{} {:>7}  {}",
            self.format(entry.at, "%Y-%m-%d %H:%M:%S"),
            entry.actor.label(),
            if entry.failed { '!' } else { ' ' },
            entry.took.map(Self::format_took).unwrap_or_default(),
            entry.text,
        )
    }

    /// How long an action took, for a reader: `4 ms`, `1.24 s`.
    ///
    /// Milliseconds up to a second and seconds past it, because the two
    /// questions a reader has are different on either side of that line. Below
    /// a millisecond the answer is `<1 ms` rather than `0 ms`: the action did
    /// happen, and a rounded zero reads like a measurement that failed.
    pub(crate) fn format_took(took: std::time::Duration) -> String {
        let ms = took.as_secs_f64() * 1000.0;
        if ms >= 1000.0 {
            // One decimal, because this column is read while it moves: the
            // Background panel redraws a running operation ten times a second,
            // and a hundredths digit there is a digit that only ever spins.
            format!("{:.1} s", ms / 1000.0)
        } else if ms >= 1.0 {
            format!("{ms:.0} ms")
        } else {
            "<1 ms".to_string()
        }
    }

    /// `at` in this log's zone, through `jiff`'s `strftime`.
    pub(crate) fn format(&self, at: Timestamp, fmt: &str) -> String {
        at.to_zoned(self.zone.clone()).strftime(fmt).to_string()
    }

    /// `at` as RFC 3339 in this log's zone, to the millisecond:
    /// `2026-09-02T12:41:07.123-07:00`.
    ///
    /// The zone the panel formats in rather than UTC, so a time an agent reads
    /// off the wire is the time the human beside it is reading off the row.
    pub(crate) fn format_rfc3339(&self, at: Timestamp) -> String {
        self.format(at, "%Y-%m-%dT%H:%M:%S%.3f%:z")
    }
}

impl Default for ActionLog {
    fn default() -> Self {
        Self::new()
    }
}

// ── The texts shared by the GUI and the MCP surface ──────────────────────
//
// One text per action, whoever took it: two rows that read the same did the
// same thing, which is what makes the actor column trustworthy. So the wording
// of a scene-graph change lives here rather than being composed twice.

/// Which of a node's layers a visibility toggle governs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layer {
    /// The node's master eye.
    Node,
    Points,
    CameraImages,
    Patches,
    PointsAtInfinity,
}

/// `run_a hidden`, `Points of run_a shown`.
pub(crate) fn visibility_text(label: &str, layer: Layer, shown: bool) -> String {
    let state = if shown { "shown" } else { "hidden" };
    match layer {
        Layer::Node => format!("{label} {state}"),
        Layer::Points => format!("Points of {label} {state}"),
        Layer::CameraImages => format!("Camera images of {label} {state}"),
        Layer::Patches => format!("Patches of {label} {state}"),
        Layer::PointsAtInfinity => format!("Points at infinity of {label} {state}"),
    }
}

/// `run_a made non-interactive`.
pub(crate) fn interactive_text(label: &str, interactive: bool) -> String {
    if interactive {
        format!("{label} made interactive")
    } else {
        format!("{label} made non-interactive")
    }
}

/// `Tint of run_a: Orange`, or `: None` for the node's own colors.
pub(crate) fn tint_text(label: &str, tint: crate::scene::NodeTint) -> String {
    let name = match tint {
        crate::scene::NodeTint::Original => "None",
        crate::scene::NodeTint::Tint(color) => color.name,
    };
    format!("Tint of {label}: {name}")
}
