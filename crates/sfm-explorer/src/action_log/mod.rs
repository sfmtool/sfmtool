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
//! - **Coalescing happens at record time and is not reversible.** A run of like
//!   entries folds into one so that a scrub, a slider drag or an agent polling
//!   for screenshots stays one line. Failures are exempt, which is why the log
//!   is a readable record rather than an audit trail.

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
    /// The word the actor column shows, and the clipboard export writes.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Actor::User => "User",
            Actor::Mcp => "MCP",
            Actor::Viewer => "Viewer",
        }
    }
}

/// What an entry is about; decides whether a run of them coalesces.
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
    /// A read-only MCP tool, by name. Coalesces per tool.
    Query(&'static str),
}

impl Kind {
    /// Whether a run of entries of this kind folds into one.
    ///
    /// The continuous kinds do: a selection scrub, a camera framing, a slider
    /// drag and an agent's polling are all things that happen many times a
    /// second and mean one thing. The discrete kinds do not — two files opened
    /// in a row are two events, not one.
    pub(crate) fn coalesces(self) -> bool {
        match self {
            Kind::Session | Kind::File | Kind::Scene | Kind::Animation => false,
            Kind::Selection | Kind::View | Kind::Display | Kind::Query(_) => true,
        }
    }
}

/// One line of the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    /// When it was recorded. Wall clock, because the log's job is to be read
    /// next to something else — an agent's transcript, a CI log, a screen
    /// recording — and only wall-clock time lines those up.
    pub at: Timestamp,
    pub actor: Actor,
    pub kind: Kind,
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
        }
    }

    /// Record a successful action as the current actor, now.
    pub(crate) fn record(&mut self, kind: Kind, text: impl Into<String>) {
        self.record_at(Timestamp::now(), kind, false, text);
    }

    /// Record a failed action as the current actor, now. Never coalesces.
    pub(crate) fn fail(&mut self, kind: Kind, text: impl Into<String>) {
        self.record_at(Timestamp::now(), kind, true, text);
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

    /// Record a read-only MCP tool call, coalescing per tool.
    pub(crate) fn query(&mut self, tool: &'static str, text: impl Into<String>) {
        self.record_at(Timestamp::now(), Kind::Query(tool), false, text);
    }

    /// Record a widget's change as one entry, building the text only then.
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
        text: impl FnOnce() -> String,
    ) {
        let touched = response.is_pointer_button_down_on()
            || response.clicked()
            || response.drag_stopped()
            || response.has_focus();
        if response.changed() && touched {
            self.record(kind, text());
        }
    }

    /// The `record` / `fail` primitive with an explicit instant.
    ///
    /// Public to the crate for the tests, which drive the coalescing window and
    /// the formatting with fixed instants rather than the wall clock.
    pub(crate) fn record_at(
        &mut self,
        at: Timestamp,
        kind: Kind,
        failed: bool,
        text: impl Into<String>,
    ) {
        if self.mute > 0 {
            return;
        }
        let entry = Entry {
            at,
            actor: self.actor,
            kind,
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
    /// The window is measured from the entry being *replaced*, which is itself
    /// the time of the last replacement — so an unbroken run folds
    /// indefinitely, and a pause longer than the window starts a new line.
    fn coalesce(&self, entry: &Entry) -> bool {
        let Some(last) = self.entries.back() else {
            return false;
        };
        !entry.failed
            && !last.failed
            && entry.kind.coalesces()
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
