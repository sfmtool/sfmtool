// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What a long computation tells its caller, and how it is told to stop.
//!
//! A kernel here runs for anywhere between a millisecond and several minutes,
//! and for that whole time it has no way to say anything. Five things are
//! missing and they are one problem: which stage it is in and what each stage
//! cost (**phases**), a remark worth keeping (**messages**), what it is doing
//! at this instant (**status**), how far along it is (**counts**), and, in the
//! other direction, stop (**cancellation**).
//!
//! [`Progress`] is the one parameter carrying all five. A function that can be
//! slow takes a `&Progress`, passes it down, and reports through it; a caller
//! that wants none of this passes `&Progress::none()`, whose every method is a
//! branch on a null sink. Nothing here decides where output goes: the sink is
//! the caller's closure, so the viewer routes it to a panel, a command-line run
//! to stderr, a test to a `Vec`. That is deliberate, because this crate carries
//! no logging dependency and is not about to grow one.
//!
//! ## Why a parameter rather than an ambient collector
//!
//! The obvious alternative is a thread-local the way `tracing` spans work, with
//! nothing threaded through any signature. It does not survive contact with
//! these kernels, because they are rayon-parallel. Work-stealing means the
//! calling thread runs some of the work and the pool runs the rest, so a
//! thread-local collector would capture a nondeterministic subset of what
//! happened, which is worse than capturing nothing at all. A `Sync` parameter
//! captured by the closure does not have the problem: every worker writes to
//! the same sink, and the sink is the caller's.
//!
//! Two smaller reasons point the same way. A process-global counter cannot say
//! what *one* call cost, and per-call attribution is the only question anybody
//! asks of it; and Rust runs tests across threads, so a global collector is
//! state that works until a test spawns a thread.
//!
//! ## Nesting
//!
//! The two things a `Progress` carries nest differently.
//!
//! A **phase** is about time already spent, so it composes by addition: a
//! child's cost is part of its parent's. [`Progress::phase`] hands back a
//! [`Phase`] guard that derefs to a `Progress` one depth deeper over the same
//! range, so a nested call takes `&p` and reports underneath it without being
//! told anything. The depth is carried in the value and never counted at the
//! sink, which is what keeps it right when two rayon threads interleave: a
//! collector matching `Enter` against `Leave` would be wrong the moment they
//! did.
//!
//! A **fraction** is about work remaining and does not compose by addition. If
//! every nested call reported 0..1 on its own terms the bar would run to full
//! and snap back once per stage, so a caller carves a share of its own range
//! for each child with [`Progress::split`], and the child reports within that
//! share without ever learning it is a child.
//!
//! A phase and a split are independent: a phase narrows the depth and keeps the
//! range, a split narrows the range and keeps the depth, and either may contain
//! the other.
//!
//! ## Two levels
//!
//! Overview is always on: [`Progress::phase`] always records, which costs two
//! clock reads and is cheap enough to be a coverage requirement rather than a
//! budget. Detail is a switch, off by default and set per `Progress` value with
//! [`Progress::detailed`], so two callers in one process can legitimately
//! differ and nothing has to be reset. [`Progress::detail_phase`] is completely
//! inert when it is off, which is what makes it affordable to leave those call
//! sites in place.
//!
//! ## Two guarantees, and only one of them lives here
//!
//! A child's range is a subinterval of its parent's, and
//! [`Progress::set_fraction`] clamps its argument into `0.0..=1.0` before
//! mapping it, so a child cannot move the global fraction outside its own
//! slice however wrong its arithmetic is. That guarantee is this module's.
//!
//! The other one, that the fraction a caller *sees* never runs backwards, is
//! the collector's: whoever owns the sink clamps what it reports to the
//! greatest fraction seen so far in the operation. It is not implemented here,
//! and there is nothing to look for. A kernel reporting nonsense can stall a
//! bar; only the collector can stop it running the bar backwards, because only
//! the collector sees the whole operation.
//!
//! ## Reporting without paying for it
//!
//! [`Progress::message`] takes [`fmt::Arguments`], which defers the formatting
//! but not the evaluation of what is being formatted: `format_args!("{}",
//! expensive())` still calls `expensive()`. So messages are written through the
//! [`crate::progress_info!`], [`crate::progress_warn!`],
//! [`crate::progress_status!`] and [`crate::progress_note!`] macros, which
//! wrap the whole thing in a test of whether anybody is listening. The methods
//! stay public for a caller that has already asked.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests;

/// How much a message wants to be noticed.
///
/// There is no per-level filtering yet: [`Progress::wants`] answers the same
/// for both. The level is carried so that a collector can display or mirror it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Worth reading.
    Info,
    /// Worth reading, and something was not as it should be.
    Warn,
}

/// The caller asked the operation to stop.
///
/// Each kernel error type carries this, as a variant or a `From`, so that
/// `progress.check_cancel()?` propagates a cancellation by the same mechanism
/// as every other failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the operation was cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// One thing a computation said, as the sink sees it.
///
/// The borrowed text is valid only for the duration of the sink call: a
/// collector that keeps an event owns what it keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event<'a> {
    /// A phase opened. What a live panel needs to mark the stage that is
    /// running, before it is over.
    Enter {
        /// The name the phase was opened with.
        phase: &'static str,
        /// How deep it sits, counted from 0 at the operation's top.
        depth: u8,
    },
    /// A phase closed, with what it cost.
    Leave {
        /// The name the matching [`Event::Enter`] carried.
        phase: &'static str,
        /// The same depth the matching [`Event::Enter`] carried.
        depth: u8,
        /// Wall time between the guard being made and being dropped.
        took: Duration,
        /// What [`Phase::note`] said this phase did, if anything.
        note: Option<&'a str>,
    },
    /// Something worth reading, in words. Appends.
    Message {
        /// How much it wants to be noticed.
        level: Level,
        /// The depth of the phase it was emitted inside.
        depth: u8,
        /// The formatted text.
        text: &'a str,
    },
    /// What the operation is doing right now. Replaces whatever it last said,
    /// rather than appending.
    Status {
        /// The formatted text.
        text: &'a str,
    },
    /// How far along, in the reporting call's own unit, for a panel to print
    /// in words: `(7, Some(20), "iteration")`.
    Count {
        /// How many are behind it.
        done: u64,
        /// How many there are in total, when the caller knows.
        total: Option<u64>,
        /// What is being counted, singular: `"image"`, `"iteration"`.
        unit: &'static str,
    },
    /// Where the bar sits, already mapped out of the reporting call's range
    /// into the whole operation's.
    Fraction {
        /// A fraction of the whole operation, in `0.0..=1.0`.
        of_whole: f32,
    },
}

/// What a long computation tells its caller, and how it is told to stop.
///
/// `Copy`, and small: a shared reference to the sink, a shared reference to the
/// cancel flag, the detail bit, the phase depth, and the `(offset, extent)` of
/// the range this call owns. A child from [`Progress::split`] or
/// [`Progress::phase`] borrows nothing from the `Progress` it came from, which
/// is what keeps the lifetimes trivial and lets a child outlive the expression
/// that made it.
///
/// It is `Sync` so that a rayon closure can capture it, and it is taken by `&`
/// rather than `&mut` so that it never conflicts with another borrow.
///
/// ```
/// use sfmtool_core::progress::{Event, Progress};
/// use std::sync::Mutex;
///
/// let seen = Mutex::new(Vec::new());
/// let sink = |event: Event<'_>| {
///     if let Event::Leave { phase, .. } = event {
///         seen.lock().unwrap().push(phase);
///     }
/// };
/// let progress = Progress::to(&sink);
///
/// {
///     let outer = progress.phase("solve");
///     let _inner = outer.phase("linearise");
/// }
/// assert_eq!(*seen.lock().unwrap(), ["linearise", "solve"]);
/// ```
#[derive(Clone, Copy)]
pub struct Progress<'a> {
    sink: Option<&'a (dyn Fn(Event<'_>) + Sync)>,
    cancel: Option<&'a AtomicBool>,
    detail: bool,
    depth: u8,
    /// Where this call's range starts within the whole operation's `0..=1`.
    offset: f32,
    /// How much of that range this call owns.
    extent: f32,
}

impl Progress<'static> {
    /// Reports nothing, cancels never, detail off.
    ///
    /// What every call site that does not care passes. Every method on it is a
    /// branch on a null sink: nothing is formatted, nothing is allocated, and
    /// the clock is never read.
    pub const fn none() -> Self {
        Progress {
            sink: None,
            cancel: None,
            detail: false,
            depth: 0,
            offset: 0.0,
            extent: 1.0,
        }
    }
}

impl<'a> Progress<'a> {
    /// Report everything to `sink`, which owns whatever it keeps.
    ///
    /// The sink is called from whichever thread reported, including rayon
    /// workers, which is why it is `Sync` rather than `FnMut`.
    pub fn to(sink: &'a (dyn Fn(Event<'_>) + Sync)) -> Self {
        Progress {
            sink: Some(sink),
            ..Progress::none()
        }
    }

    /// Stop when `flag` is set.
    pub fn cancelled_by(self, flag: &'a AtomicBool) -> Self {
        Progress {
            cancel: Some(flag),
            ..self
        }
    }

    /// Record the phases opened with [`Self::detail_phase`], or not.
    pub fn detailed(self, on: bool) -> Self {
        Progress { detail: on, ..self }
    }

    /// Has the caller asked this to stop? One relaxed atomic load, and `false`
    /// when no flag is attached.
    ///
    /// For a loop that wants to stop and hand back what it has. A loop that
    /// wants to give up instead should use [`Self::check_cancel`].
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    /// The same question in `?` form, which is how most callers should ask it:
    /// `progress.check_cancel()?` propagates without anything to remember.
    ///
    /// # Errors
    ///
    /// [`Cancelled`] when the attached flag is set.
    pub fn check_cancel(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    /// Whether detailed phases are being recorded, so a caller can skip
    /// building something only a detailed run would read.
    pub fn is_detailed(&self) -> bool {
        self.detail
    }

    /// Whether anything is listening at this level, so a caller can skip work
    /// nobody will read. The [`crate::progress_info!`] family checks this first.
    ///
    /// True whenever a sink is attached. There is no per-level filter: both
    /// levels answer the same, and the level is a property of the message
    /// rather than a threshold.
    pub fn wants(&self, level: Level) -> bool {
        let _ = level;
        self.sink.is_some()
    }

    /// Append a line.
    ///
    /// Prefer [`crate::progress_info!`] and [`crate::progress_warn!`], which do
    /// not evaluate their arguments when nothing is listening. The message
    /// carries the depth of the phase it was emitted inside.
    pub fn message(&self, level: Level, text: fmt::Arguments<'_>) {
        let Some(sink) = self.sink else { return };
        let depth = self.depth;
        match text.as_str() {
            Some(text) => sink(Event::Message { level, depth, text }),
            None => sink(Event::Message {
                level,
                depth,
                text: &fmt::format(text),
            }),
        }
    }

    /// Replace the one-line statement of what this call is doing right now.
    ///
    /// A status is live state, so it is worth setting where a loop would
    /// otherwise emit one message per item: a name that changes, not a line
    /// that accumulates. Prefer [`crate::progress_status!`].
    pub fn set_status_message(&self, text: fmt::Arguments<'_>) {
        let Some(sink) = self.sink else { return };
        match text.as_str() {
            Some(text) => sink(Event::Status { text }),
            None => sink(Event::Status {
                text: &fmt::format(text),
            }),
        }
    }

    /// How far along, in this call's own unit.
    ///
    /// A known total also moves the bar, within this call's range: the emitted
    /// [`Event::Count`] is followed by the [`Event::Fraction`] that
    /// `done / total` maps to. `count(done, None, unit)` reports the number and
    /// moves nothing, and a `total` of zero moves nothing either.
    ///
    /// This is what a loop over many items reports. Splitting a range per item
    /// would be the wrong shape and the wrong cost.
    pub fn count(&self, done: u64, total: Option<u64>, unit: &'static str) {
        if self.sink.is_none() {
            return;
        }
        self.emit(Event::Count { done, total, unit });
        if let Some(total) = total.filter(|total| *total > 0) {
            self.set_fraction(done as f32 / total as f32);
        }
    }

    /// Where this call has got to within its own range, `0.0..=1.0`.
    ///
    /// The argument is clamped into `0.0..=1.0` before it is mapped (and a NaN
    /// is read as `0.0`), so a call cannot report its way out of the range it
    /// was given and into a sibling's.
    pub fn set_fraction(&self, f: f32) {
        if self.sink.is_none() {
            return;
        }
        let f = if f.is_nan() { 0.0 } else { f.clamp(0.0, 1.0) };
        self.emit(Event::Fraction {
            of_whole: self.offset + f * self.extent,
        });
    }

    /// Carve this call's range into `N` pieces, one per nested call, in
    /// proportion to `weights`.
    ///
    /// The weights are relative: they are normalised to their sum, so
    /// `[1.0, 9.0]` and `[0.1, 0.9]` are the same split. They are constants
    /// chosen by whoever wrote the call, an estimate of how the time divides,
    /// and one that is wrong makes the bar uneven rather than untrue.
    ///
    /// Split for stages, not for items: a handful of distinct steps get a
    /// `split`, and a loop over a hundred thousand of anything reports
    /// [`Self::count`] instead.
    ///
    /// A weight that is negative, zero or not finite counts as zero. If that
    /// leaves nothing positive to normalise by, the range is split evenly,
    /// since a caller that says nothing usable about the division is better
    /// served by an even bar than by a panic or a dead one.
    ///
    /// ```
    /// use sfmtool_core::progress::Progress;
    ///
    /// let progress = Progress::none();
    /// let [_materialise, solve, _row_map] = progress.split([0.05, 0.90, 0.05]);
    /// // `solve` reports its own 0..1, and it lands in 0.05..0.95.
    /// solve.set_fraction(0.5);
    /// ```
    pub fn split<const N: usize>(&self, weights: [f32; N]) -> [Progress<'a>; N] {
        let mut share = [0.0f32; N];
        for (share, weight) in share.iter_mut().zip(weights.iter()) {
            *share = if weight.is_finite() && *weight > 0.0 {
                *weight
            } else {
                0.0
            };
        }
        // Every share is finite and non-negative by now, so the sum is either
        // usable or one of two degenerate cases: nothing positive was given, or
        // the weights were large enough to overflow.
        let mut total: f32 = share.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            share = [1.0f32; N];
            total = N as f32;
        }

        let mut start = [0.0f32; N];
        let mut acc = 0.0f32;
        for (start, share) in start.iter_mut().zip(share.iter()) {
            *start = acc;
            acc += *share;
        }

        std::array::from_fn(|i| Progress {
            offset: self.offset + self.extent * (start[i] / total),
            extent: self.extent * (share[i] / total),
            ..*self
        })
    }

    /// `n` equal pieces, for a loop with a known trip count whose body is big
    /// enough to report progress of its own.
    pub fn split_evenly(&self, n: usize) -> impl Iterator<Item = Progress<'a>> {
        let parent = *self;
        let n_f = n as f32;
        (0..n).map(move |i| Progress {
            offset: parent.offset + parent.extent * (i as f32 / n_f),
            extent: parent.extent / n_f,
            ..parent
        })
    }

    /// Open a phase, which closes when the guard drops.
    ///
    /// Always recorded, when there is a sink to record it to. This is the
    /// overview level, and it is a coverage requirement rather than a budget:
    /// every stage of an operation that can exceed a frame should have one, so
    /// that the first question about a surprising number is never
    /// unanswerable.
    pub fn phase(&self, name: &'static str) -> Phase<'a> {
        self.open(name, true)
    }

    /// The same, recorded only when [`Self::is_detailed`], and inert
    /// otherwise: no clock read, no allocation, no event, and the guard's inner
    /// `Progress` stays at this depth, because no phase was opened.
    pub fn detail_phase(&self, name: &'static str) -> Phase<'a> {
        self.open(name, self.detail)
    }

    fn open(&self, name: &'static str, enabled: bool) -> Phase<'a> {
        if !enabled || self.sink.is_none() {
            return Phase {
                inner: *self,
                name,
                depth: self.depth,
                started: None,
                note: None,
                recording: false,
            };
        }
        self.emit(Event::Enter {
            phase: name,
            depth: self.depth,
        });
        Phase {
            inner: Progress {
                depth: self.depth.saturating_add(1),
                ..*self
            },
            name,
            depth: self.depth,
            started: Some(Instant::now()),
            note: None,
            recording: true,
        }
    }

    #[inline]
    fn emit(&self, event: Event<'_>) {
        if let Some(sink) = self.sink {
            sink(event);
        }
    }
}

impl fmt::Debug for Progress<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Progress")
            .field("reporting", &self.sink.is_some())
            .field("cancellable", &self.cancel.is_some())
            .field("detail", &self.detail)
            .field("depth", &self.depth)
            .field("offset", &self.offset)
            .field("extent", &self.extent)
            .finish()
    }
}

/// An open phase, and the `Progress` for whatever runs inside it.
///
/// Derefs to a [`Progress`] one depth deeper, over the same range and the same
/// sink, so a nested call is `inner(&phase)?` and its phases nest under this
/// one without anything being passed but the guard.
///
/// Dropping the guard emits [`Event::Leave`] with the depth its
/// [`Event::Enter`] used, the wall time it stood for, and whatever
/// [`Phase::note`] was told.
pub struct Phase<'a> {
    inner: Progress<'a>,
    name: &'static str,
    /// The depth the opening [`Event::Enter`] carried, which is one less than
    /// `inner`'s whenever this guard is recording. Kept rather than recomputed
    /// so that a saturated depth still closes what it opened.
    depth: u8,
    started: Option<Instant>,
    note: Option<String>,
    recording: bool,
}

impl Phase<'_> {
    /// Say what this phase did, for the column beside its time: `reused`,
    /// `46 231 points`.
    ///
    /// Replaces whatever the phase was last told. Prefer [`crate::progress_note!`],
    /// which does not evaluate its arguments for a phase that is not
    /// recording.
    pub fn note(&mut self, note: fmt::Arguments<'_>) {
        if !self.recording {
            return;
        }
        self.note = Some(fmt::format(note));
    }

    /// Whether this guard will record anything at all.
    ///
    /// False for a [`Progress::detail_phase`] on a `Progress` whose detail is
    /// off, and false for any phase on a `Progress` with no sink.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Drop the guard without recording: the phase did not run.
    ///
    /// The opening [`Event::Enter`] has already been sent, so a collector
    /// tracking what is open should treat a phase that never closes as one that
    /// was abandoned.
    pub fn cancel(mut self) {
        self.recording = false;
    }
}

impl<'a> std::ops::Deref for Phase<'a> {
    type Target = Progress<'a>;

    fn deref(&self) -> &Progress<'a> {
        &self.inner
    }
}

impl Drop for Phase<'_> {
    fn drop(&mut self) {
        if !self.recording {
            return;
        }
        let took = self
            .started
            .map_or(Duration::ZERO, |started| started.elapsed());
        self.inner.emit(Event::Leave {
            phase: self.name,
            depth: self.depth,
            took,
            note: self.note.as_deref(),
        });
    }
}

impl fmt::Debug for Phase<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Phase")
            .field("name", &self.name)
            .field("depth", &self.depth)
            .field("recording", &self.recording)
            .field("note", &self.note)
            .finish()
    }
}

/// Append a [`Level::Info`] message to a [`Progress`], evaluating nothing when
/// nothing is listening.
///
/// The first argument is anything that derefs to a `Progress`, which includes a
/// [`Phase`] guard; the rest is a `format!` call.
///
/// ```
/// use sfmtool_core::progress::{Event, Progress};
/// use sfmtool_core::progress_info;
/// use std::sync::Mutex;
///
/// fn expensive() -> usize {
///     unreachable!("nobody is listening, so this is never called")
/// }
///
/// let quiet = Progress::none();
/// progress_info!(quiet, "{} points", expensive());
///
/// let seen = Mutex::new(Vec::new());
/// let sink = |event: Event<'_>| {
///     if let Event::Message { text, .. } = event {
///         seen.lock().unwrap().push(text.to_string());
///     }
/// };
/// progress_info!(Progress::to(&sink), "{} points", 17);
/// assert_eq!(*seen.lock().unwrap(), ["17 points"]);
/// ```
#[macro_export]
macro_rules! progress_info {
    ($progress:expr, $($arg:tt)*) => {{
        let progress = &$progress;
        if progress.wants($crate::progress::Level::Info) {
            progress.message(
                $crate::progress::Level::Info,
                ::core::format_args!($($arg)*),
            );
        }
    }};
}

/// Append a [`Level::Warn`] message to a [`Progress`], evaluating nothing when
/// nothing is listening.
///
/// ```
/// use sfmtool_core::progress::{Event, Level, Progress};
/// use sfmtool_core::progress_warn;
/// use std::sync::Mutex;
///
/// let seen = Mutex::new(Vec::new());
/// let sink = |event: Event<'_>| {
///     if let Event::Message { level, text, .. } = event {
///         seen.lock().unwrap().push((level, text.to_string()));
///     }
/// };
/// let progress = Progress::to(&sink);
/// progress_warn!(progress, "{} views below the ZNCC floor", 3);
///
/// let seen = seen.lock().unwrap();
/// assert_eq!(seen[0].0, Level::Warn);
/// assert_eq!(seen[0].1, "3 views below the ZNCC floor");
/// ```
#[macro_export]
macro_rules! progress_warn {
    ($progress:expr, $($arg:tt)*) => {{
        let progress = &$progress;
        if progress.wants($crate::progress::Level::Warn) {
            progress.message(
                $crate::progress::Level::Warn,
                ::core::format_args!($($arg)*),
            );
        }
    }};
}

/// Replace what a [`Progress`] is saying it is doing right now, evaluating
/// nothing when nothing is listening.
///
/// This is what a loop over many named things reports: one line that changes,
/// rather than one line per item that accumulates.
///
/// ```
/// use sfmtool_core::progress::{Event, Progress};
/// use sfmtool_core::progress_status;
/// use std::sync::Mutex;
///
/// let latest = Mutex::new(String::new());
/// let sink = |event: Event<'_>| {
///     if let Event::Status { text } = event {
///         *latest.lock().unwrap() = text.to_string();
///     }
/// };
/// let progress = Progress::to(&sink);
/// for name in ["dino_42.jpg", "dino_43.jpg"] {
///     progress_status!(progress, "{name}");
/// }
/// assert_eq!(*latest.lock().unwrap(), "dino_43.jpg");
/// ```
#[macro_export]
macro_rules! progress_status {
    ($progress:expr, $($arg:tt)*) => {{
        let progress = &$progress;
        if progress.wants($crate::progress::Level::Info) {
            progress.set_status_message(::core::format_args!($($arg)*));
        }
    }};
}

/// Say what an open [`Phase`] did, evaluating nothing when that phase is not
/// recording.
///
/// The first argument is a mutable [`Phase`] binding, since the note is carried
/// to the phase's own [`Event::Leave`].
///
/// ```
/// use sfmtool_core::progress::{Event, Progress};
/// use sfmtool_core::progress_note;
/// use std::sync::Mutex;
///
/// let seen = Mutex::new(Vec::new());
/// let sink = |event: Event<'_>| {
///     if let Event::Leave { phase, note, .. } = event {
///         seen.lock().unwrap().push((phase, note.map(str::to_string)));
///     }
/// };
/// let progress = Progress::to(&sink);
/// {
///     let mut packing = progress.phase("patch atlas");
///     progress_note!(packing, "{} tiles", 46_231);
/// }
/// assert_eq!(
///     *seen.lock().unwrap(),
///     [("patch atlas", Some("46231 tiles".to_string()))]
/// );
/// ```
#[macro_export]
macro_rules! progress_note {
    ($phase:expr, $($arg:tt)*) => {{
        let phase = &mut $phase;
        if phase.is_recording() {
            phase.note(::core::format_args!($($arg)*));
        }
    }};
}
