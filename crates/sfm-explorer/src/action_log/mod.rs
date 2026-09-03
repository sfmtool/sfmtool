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

use std::collections::VecDeque;

use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};

mod panel;

#[cfg(test)]
mod tests;

pub(crate) use panel::show;

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
    /// What is loaded: opened, reloaded, closed.
    File,
    /// Which reconstruction, image, camera or point is selected.
    Selection,
    /// The scene graph: visibility, tint, solo, alignment, resection.
    Scene,
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

    /// Largest gap between two like entries for the newer to replace the older.
    pub(crate) const COALESCE_WINDOW: SignedDuration = SignedDuration::from_secs(1);

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
        };
        // The mirror is unconditional and precedes coalescing: a `RUST_LOG`
        // capture is the stream of what happened, and folding a run away is a
        // property of the panel's readability, not of the session.
        log::info!(target: "sfm_explorer::action_log", "{}", self.line(&entry));
        if self.coalesce(&entry) {
            *self
                .entries
                .back_mut()
                .expect("coalesce found a last entry") = entry;
            return;
        }
        if self.entries.len() >= Self::CAPACITY {
            self.entries.pop_front();
            self.dropped += 1;
        }
        self.entries.push_back(entry);
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

    /// The whole log as clipboard text, one line per entry.
    pub(crate) fn to_clipboard_text(&self) -> String {
        let mut out = String::new();
        for entry in self.entries() {
            out.push_str(&self.line(entry));
            out.push('\n');
        }
        out
    }

    /// One entry as the clipboard and the `log::info!` mirror render it: the
    /// full date, the actor, a `!` where colour would have said "failed", and
    /// the text.
    pub(crate) fn line(&self, entry: &Entry) -> String {
        format!(
            "{}  {:<6}{} {}",
            self.format(entry.at, "%Y-%m-%d %H:%M:%S"),
            entry.actor.label(),
            if entry.failed { '!' } else { ' ' },
            entry.text,
        )
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
