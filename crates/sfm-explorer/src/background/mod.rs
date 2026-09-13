// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One long operation, running off the GUI thread.
//!
//! See `specs/gui/background-tasks.md`. A bulk edit is already a
//! pure function from the value at a node's cursor to the next value, so a
//! worker can be handed a clone of the `Arc` that value is behind and read it
//! while the GUI thread keeps drawing the same allocation. Nothing here wraps
//! [`AppState`] in a lock or hands a worker a reference into the scene, and the
//! reason it does not have to is the document's value semantics
//! (`specs/gui/document-model.md`).
//!
//! Two small things *are* shared mutably, deliberately and under a lock, and
//! neither is the document: the [`Collector`] the worker reports into and the
//! flag it is asked to stop through. Both are a few words of state that exist
//! to be written by one thread and read by another.
//!
//! ## The shape
//!
//! - [`Operation`] is what a background operation is and what it claims about
//!   itself; [`Job`] is the work, as a function of the [`Progress`] it reports
//!   through.
//! - [`BackgroundTask`] is the one that is running: a field of [`AppState`],
//!   so the busy check is where every method that needs it already is.
//! - [`Report`] is what crosses the channel, and carries no progress: phases,
//!   messages and counts are in the collector both threads hold.
//!
//! ## What the GUI thread does with it
//!
//! [`AppState::poll_background_task`] runs in phase 0 of the frame, before the MCP
//! drain, so a completed operation's version is on screen in the frame it
//! landed and an agent's call in that same frame reads the new value. What
//! remains of the freeze is the upload the next phase does, which for an
//! adjustment is single-digit milliseconds.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sfmtool_core::progress::Progress;
use sfmtool_core::{EditedReconstruction, SfmrReconstruction};

use crate::action_log::{Actor, Kind};
use crate::document::PointMap;
use crate::progress::{Collector, Detail};
use crate::scene::ReconId;
use crate::state::AppState;

pub(crate) mod panel;

#[cfg(test)]
mod tests;

/// One kind of background operation: what it is called, and what it claims
/// about itself.
///
/// `cancellable` is a **declaration**, not something discovered: a kernel that
/// never polls the flag is indistinguishable from one that has not reached a
/// poll yet, so nothing can infer this. The wrapper knows which kernel it
/// calls, so the wrapper states it, and a test holds it to the claim by
/// cancelling each operation that says `true` and asserting it stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Operation {
    /// What the panel, the refusals and the Action Log call it.
    pub(crate) name: &'static str,
    /// Whether asking it to stop does anything.
    pub(crate) cancellable: bool,
}

impl Operation {
    /// The bundle adjustment: the one that freezes the window for minutes, and
    /// the first one moved off the thread.
    ///
    /// Cancellable because `sfmtool_core::bundle_adjust` polls the flag between
    /// rounds and between iterations and hands back
    /// `BundleAdjustError::Cancelled` rather than a half-solved state.
    pub(crate) const BUNDLE_ADJUST: Operation = Operation {
        name: "Bundle adjust",
        cancellable: true,
    };

    /// Every operation that can go to the background.
    ///
    /// A list rather than a set of constants used one at a time, so the test
    /// that holds each `cancellable: true` to its claim has something to walk:
    /// a declaration nothing checks is a declaration that rots.
    // Read by that test alone, which is what it is for.
    #[cfg(test)]
    pub(crate) const ALL: [Operation; 1] = [Operation::BUNDLE_ADJUST];
}

/// The work one background operation does, as a function of the [`Progress`]
/// it reports through.
///
/// A boxed `FnOnce` rather than a kernel call written into the worker, because
/// the value it reads and the options it was given are the caller's to capture:
/// what crosses the thread boundary is one closure that owns everything it
/// touches, which is what keeps the worker from needing a reference into the
/// scene.
pub(crate) type Job = Box<dyn for<'a> FnOnce(&Progress<'a>) -> Finished + Send>;

/// A long operation running off the GUI thread.
pub(crate) struct BackgroundTask {
    /// Which operation this is, monotonic across the session and never reused.
    ///
    /// A caller that was handed this id while the operation was running comes
    /// back for the answer after it has finished, by which time the task is
    /// gone and another may have taken its place: the id is what says whether
    /// [`AppState::last_background_task`] is about the operation being asked after
    /// or about a later one.
    pub(crate) id: u64,
    /// What it is, for the refusals, and what it claims about itself.
    pub(crate) operation: Operation,
    /// The node it will install its answer into, and the node it locks.
    pub(crate) node: ReconId,
    /// That node's label, for the sentences that name it. Kept here rather
    /// than read back from the scene, so a refusal reads the same whatever has
    /// happened to the node in the meantime.
    pub(crate) label: String,
    /// When it started, which is what the Action Log entry's cost is measured
    /// from.
    pub(crate) started: Instant,
    /// Who asked for it. The entry the operation writes is theirs, however
    /// many minutes later it lands: an agent that started a solve is the actor
    /// of the version it produced.
    pub(crate) actor: Actor,
    /// Where the worker reports, and where the panel will read. Shared, taken
    /// by `&`, never borrowed mutably (`specs/gui/operation-progress.md`,
    /// "In the viewer").
    pub(crate) collector: Arc<Collector>,
    /// Set to ask the operation to stop. The kernel polls it through the
    /// `Progress` built over `collector`; one that never polls is not
    /// cancellable, and [`Operation::cancellable`] says so.
    pub(crate) cancel: Arc<AtomicBool>,
    /// What the worker sends, drained by [`AppState::poll_background_task`].
    reports: Receiver<Report>,
    /// Whether a [`Report::Progressed`] the GUI thread has not drained yet is
    /// already in the queue.
    ///
    /// A report says only "something changed", so a second one that says the
    /// same thing while the first is still waiting is not worth sending: the
    /// GUI thread reads the collector once either way, and the wake it costs is
    /// a cross-thread wake of the event loop. So a thousand reports in a frame
    /// leave one message and one wake rather than a thousand of each.
    queued: Arc<AtomicBool>,
}

/// What crosses the channel, worker to GUI thread.
///
/// Phases, messages and counts do not: they are in the collector both sides
/// hold, which is what lets a report carry nothing at all.
pub(crate) enum Report {
    /// Something was reported. Carries nothing; it exists to wake the loop.
    Progressed,
    /// The operation is over, one way or another.
    Done(Box<Finished>),
}

/// How an operation ended, and what it produced if it produced anything.
///
/// Three outcomes rather than a `Result` with a string, because a cancellation
/// is not a failure of the kernel and the log says so in different words: only
/// the code that knows the kernel's error type can tell the two apart, and that
/// code is the job.
// The produced variant is a whole reconstruction, and the other two are a
// string at most. That difference costs two moves on the worker and nothing
// anywhere else: a `Finished` reaches the GUI thread inside the `Box` that
// [`Report::Done`] puts it in, so the channel carries a pointer either way.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Finished {
    /// The next value, and everything the GUI thread needs to install it.
    Produced {
        /// What the kernel returned, which becomes the next version's base.
        value: SfmrReconstruction,
        /// The map from the input's rows to that value's, which is what a
        /// selection follows across the operation.
        map: PointMap,
        /// The version's label, as the Edit History panel lists it.
        version_label: String,
        /// The Action Log sentence, up to the serials, which only the GUI
        /// thread can know because only it can push the version.
        text: String,
    },
    /// The operation was asked to stop, and did.
    Cancelled,
    /// The kernel refused, in its own words.
    Failed(String),
}

/// What became of the operation that ran most recently.
///
/// Kept after the task is gone because the question outlives it: a tool call
/// that was handed a handle while the operation was running comes back for the
/// answer, and by then there is no task to ask. The Background panel asks
/// the same question with nothing running, which is why what it cost and what
/// it spent the time on are here rather than only in the Action Log: the panel
/// would otherwise have to find its own entry in a buffer that drops and
/// clears.
pub(crate) struct FinishedTask {
    /// Which operation this is about.
    pub(crate) id: u64,
    /// What it was, by the name the panel and the refusals call it.
    pub(crate) operation: Operation,
    /// The label of the node it ran on, as it read when the operation started.
    pub(crate) label: String,
    /// What the whole operation cost, measured from the instant it started.
    ///
    /// Not the entry's `took`, which the frame stamps on settling and which
    /// therefore also carries the upload that put the answer on screen.
    pub(crate) took: Duration,
    /// What it reported, unfolded, which is what the panel drew while it ran.
    ///
    /// Not the entry's breakdown, which folds repeated stages: the Background
    /// panel does not fold, and an operation that collapsed into a summary the
    /// moment it finished would be a panel that changed its mind.
    pub(crate) detail: Vec<Detail>,
    /// The Action Log sentence it wrote, or the refusal or cancellation that
    /// ended it.
    pub(crate) outcome: Result<String, String>,
}

/// What is running, for a reader that cannot reach [`AppState`].
///
/// The MCP server answers a timed-out call on its own thread, where the GUI
/// thread is by definition not answering, so the one fact that message needs is
/// the one fact it cannot ask for. Two fields written twice per operation and
/// read once per timeout, which is why a `Mutex` is the whole of the
/// machinery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Busy {
    /// What the operation is called, as the panel and the refusals call it.
    pub(crate) operation: &'static str,
    /// The node it is running on.
    pub(crate) label: String,
}

/// Where [`Busy`] is kept: shared with whoever has to read it off the GUI
/// thread, and `None` whenever nothing is running.
pub(crate) type BusyNotice = Arc<Mutex<Option<Busy>>>;

/// What is running, or `None`. Poisoned or not, because a stale sentence in an
/// error message is a smaller failure than a panic inside one.
pub(crate) fn busy(notice: &BusyNotice) -> Option<Busy> {
    notice
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Say what is running, or that nothing is.
fn set_busy(notice: &BusyNotice, running: Option<Busy>) {
    *notice
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = running;
}

/// What one poll did, for the frame that called it.
///
/// Two answers rather than the one the draft's `bool` gives, because the frame
/// asks two different questions of a poll: whether to repaint, which is true of
/// every report, and whether to drop the panel caches describing a geometry the
/// node no longer holds, which is true only of a completed operation. A frame
/// that dropped those caches per report would throw away its textures a
/// thousand times during one solve.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Polled {
    /// Whether anything arrived, so the frame knows to redraw.
    pub(crate) changed: bool,
    /// The node a finished operation gave a new version to, whose panel caches
    /// are now about a value it no longer holds.
    pub(crate) installed: Option<ReconId>,
}

impl AppState {
    /// The operation running off the GUI thread, if there is one.
    pub(crate) fn background_task(&self) -> Option<&BackgroundTask> {
        self.background_task.as_ref()
    }

    /// Why an edit of `id` is refused right now, or `None`.
    ///
    /// Every method that gives a node a new version, moves its cursor, writes
    /// it out or closes it calls this first, so one sentence covers the
    /// method's refusal, the menu's greying and the wire's. Other nodes are
    /// untouched and stay fully editable: the operation is a function of the
    /// value it was handed, and an edit underneath it would produce a version
    /// whose parent is not the version it was computed from.
    pub(crate) fn busy_refusal(&self, id: ReconId) -> Option<String> {
        let task = self.background_task.as_ref()?;
        (task.node == id).then(|| {
            format!(
                "{} is busy: {} is still running.",
                task.label, task.operation.name
            )
        })
    }

    /// Why cancelling would do nothing, or `None` when it would stop something.
    ///
    /// The sentence a Cancel button's tooltip carries and the wire's refusal
    /// are the same sentence, for the same reason every other refusal here has
    /// one spelling.
    pub(crate) fn cancel_refusal(&self) -> Option<String> {
        match self.background_task.as_ref() {
            None => Some("Nothing is running in the background.".to_string()),
            Some(task) if !task.operation.cancellable => Some(format!(
                "{} cannot be cancelled: it never asks whether it should stop.",
                task.operation.name
            )),
            Some(_) => None,
        }
    }

    /// Start `operation` on `id`, running `job` on a worker thread.
    ///
    /// Refuses when anything is already running, anywhere in the viewer: these
    /// operations saturate the machine, so two at once would make both slower
    /// and neither would finish sooner.
    ///
    /// **Nothing is logged here.** The log records outcomes, not intentions
    /// (`specs/gui/action-log.md`), so the entry is written when the operation
    /// ends, by [`AppState::poll_background_task`].
    pub(crate) fn start_background_task(
        &mut self,
        operation: Operation,
        id: ReconId,
        job: Job,
    ) -> Result<(), String> {
        if let Some(running) = self.background_task.as_ref() {
            return Err(format!(
                "{} is still running on {}.",
                running.operation.name, running.label
            ));
        }
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;

        let (reports, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let queued = Arc::new(AtomicBool::new(false));
        // The level the Action Log toolbar's checkbox last left, read as the
        // operation starts so that a change to it takes effect on the next one.
        let collector = Arc::new(Collector::reporting(
            self.action_log.detailed_timing(),
            progressed(reports.clone(), Arc::clone(&queued), self.wake.clone()),
        ));

        let worker = Worker {
            operation,
            job,
            collector: Arc::clone(&collector),
            cancel: Arc::clone(&cancel),
            reports,
            wake: self.wake.clone(),
        };
        std::thread::Builder::new()
            .name(format!("sfm-explorer {}", operation.name))
            .spawn(move || worker.run())
            .map_err(|e| {
                format!(
                    "Could not start a worker thread for {}: {e}",
                    operation.name
                )
            })?;

        let operation_id = self.next_operation_id;
        self.next_operation_id += 1;
        // Said here and unsaid in `finish`, which are the two instants the
        // answer changes at. Anything reading it is off the GUI thread and
        // cannot ask the state.
        set_busy(
            &self.busy_notice,
            Some(Busy {
                operation: operation.name,
                label: label.clone(),
            }),
        );
        self.background_task = Some(BackgroundTask {
            id: operation_id,
            operation,
            node: id,
            label,
            started: Instant::now(),
            actor: self.action_log.actor(),
            collector,
            cancel,
            reports: rx,
            queued,
        });
        Ok(())
    }

    /// Apply every report the worker has sent.
    ///
    /// Runs in phase 0 of the frame, before the MCP drain: a completed
    /// operation's version is then on screen in the frame it landed, and an
    /// agent's call in that same frame reads the new value rather than the old
    /// one.
    pub(crate) fn poll_background_task(&mut self) -> Polled {
        self.apply_reports(None)
    }

    /// The drain both readers share: `first` is a report already taken off the
    /// channel, and the rest are whatever is waiting behind it.
    ///
    /// The parameter is there because a blocking read consumes the message it
    /// waited for, and a message consumed outside this loop would be a message
    /// nothing acted on.
    fn apply_reports(&mut self, first: Option<Report>) -> Polled {
        let mut polled = Polled::default();
        let Some(task) = self.background_task.as_ref() else {
            return polled;
        };
        // Cleared before the drain rather than after it, so a report made
        // between the two is sent rather than swallowed by a latch the GUI
        // thread was about to clear.
        task.queued.store(false, Ordering::Release);
        let mut done = None;
        let mut next = first;
        loop {
            let report = match next.take() {
                Some(report) => report,
                None => match task.reports.try_recv() {
                    Ok(report) => report,
                    Err(TryRecvError::Empty) => break,
                    // Not reachable while the viewer holds the task: the
                    // collector it shares with the worker owns a sender of its
                    // own, and a job that panics is caught and reported
                    // ([`Worker::run`]). Kept because the alternative to
                    // answering a disconnected channel is a node locked
                    // forever by an operation that will never speak again.
                    Err(TryRecvError::Disconnected) => {
                        polled.changed = true;
                        done = Some(Finished::Failed(format!(
                            "{} of {} stopped without an answer.",
                            task.operation.name, task.label
                        )));
                        break;
                    }
                },
            };
            polled.changed = true;
            match report {
                Report::Progressed => {}
                Report::Done(finished) => {
                    done = Some(*finished);
                    break;
                }
            }
        }
        if let Some(finished) = done {
            let task = self.background_task.take().expect("just borrowed");
            set_busy(&self.busy_notice, None);
            polled.installed = self.finish(task, finished);
        }
        polled
    }

    /// Ask the operation to stop. Silently does nothing when it cannot, which
    /// is what [`AppState::cancel_refusal`] is for saying out loud.
    pub(crate) fn cancel_background_task(&mut self) {
        if let Some(task) = self.background_task.as_ref() {
            if task.operation.cancellable {
                task.cancel.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Install what the operation produced, or record why it produced nothing.
    ///
    /// The entry is the one the synchronous edit wrote, in the same words and
    /// with the same stages under it, with two differences that are the whole
    /// point: its actor is whoever asked rather than whoever happens to be
    /// standing when it lands, and its cost is measured from the instant the
    /// operation started rather than from the frame that installed it. Without
    /// that instant a ninety-five second solve would report the six
    /// milliseconds of the frame that pushed its version.
    fn finish(&mut self, task: BackgroundTask, finished: Finished) -> Option<ReconId> {
        let BackgroundTask {
            id,
            operation,
            node,
            label,
            started,
            actor,
            collector,
            ..
        } = task;
        let mut installed = None;
        let outcome = match finished {
            Finished::Produced {
                value,
                map,
                version_label,
                text,
            } => match self.scene.iter().position(|n| n.id == node) {
                // Closing the busy node is refused, so this is a bug rather
                // than a race; it is still said out loud rather than dropped,
                // because a solve that produced an answer nobody can see is
                // worth a row.
                None => Err(format!(
                    "{} of {label} finished, but it is no longer loaded.",
                    operation.name
                )),
                Some(index) => {
                    // Timed on this thread because it happens on this thread:
                    // the push is the one stage of the operation a worker
                    // cannot run, and it belongs in the same breakdown as the
                    // stages that preceded it.
                    let serial = {
                        let _phase = collector.phase("push version");
                        self.scene[index].history.push(
                            EditedReconstruction::new(Arc::new(value)),
                            map,
                            version_label,
                        )
                    };
                    let parent = crate::state::edits::version_before(&self.scene[index], serial);
                    self.follow_selection_forward(node);
                    installed = Some(node);
                    Ok(format!("{text} ({parent} → {serial})"))
                }
            },
            // No elapsed in the sentence: the entry is timed from `started`
            // like the successful one, so the cost column already says how
            // long it ran, and two spellings of one number can only disagree.
            Finished::Cancelled => Err(format!("{} of {label} cancelled", operation.name)),
            Finished::Failed(message) => Err(message),
        };
        // Timed from `started` whichever way it went. A cancelled solve spent
        // the seconds it ran for, and a kernel that refused spent whatever it
        // spent before deciding to; the frame that collected the answer is not
        // what either of them cost. What it reported before stopping is kept
        // too, since a refusal is a thing a reader wants the breakdown of.
        // The transcript, taken before `take` clears it: the panel goes on
        // showing this operation after the entry is written, and shows it the
        // way it showed it while it ran. Folding it here instead would collapse
        // the rounds a reader had just watched arrive, at the instant the
        // operation finished. A transcript is a few hundred rows of names and
        // durations, once per operation.
        let kept = collector.live().rows;
        let detail = collector.take();
        let took = started.elapsed();
        match &outcome {
            Ok(text) => self
                .action_log
                .record_done_as(actor, Kind::Edit, started, text, detail),
            Err(message) => {
                self.action_log
                    .fail_done_as(actor, Kind::Edit, started, message.clone(), detail)
            }
        }
        self.last_background_task = Some(FinishedTask {
            id,
            operation,
            label,
            took,
            detail: kept,
            outcome,
        });
        installed
    }

    /// Drive the running operation to its end, applying what it reports.
    ///
    /// The test seam, and the only blocking read of the channel: a frame never
    /// waits, so the production drain is [`AppState::poll_background_task`]'s
    /// `try_recv`. Waiting on the worker's own messages is what makes a test
    /// deterministic without a sleep or a poll loop.
    #[cfg(test)]
    pub(crate) fn finish_background_task(&mut self) -> Polled {
        use std::time::Duration;

        /// Long enough that a loaded machine does not fail a passing test,
        /// short enough that a wedged worker fails rather than hangs.
        const PATIENCE: Duration = Duration::from_secs(120);

        let mut polled = Polled::default();
        while self.background_task.is_some() {
            let waited = {
                let task = self.background_task.as_ref().expect("just checked");
                task.reports.recv_timeout(PATIENCE)
            };
            let first = match waited {
                Ok(report) => Some(report),
                // The worker went without a word; the drain says so in its own
                // words rather than this doing it twice.
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => None,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    panic!("the background operation said nothing for {PATIENCE:?}")
                }
            };
            let step = self.apply_reports(first);
            polled.changed |= step.changed;
            polled.installed = polled.installed.or(step.installed);
        }
        polled
    }
}

/// Everything the worker thread owns.
///
/// A struct rather than five captures, so the `spawn` closure is one line and
/// what crosses the boundary is written down in one place.
struct Worker {
    operation: Operation,
    job: Job,
    collector: Arc<Collector>,
    cancel: Arc<AtomicBool>,
    reports: Sender<Report>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Worker {
    /// Run the job and say what it came back with.
    ///
    /// The `Progress` is built here, on the worker, over the collector the GUI
    /// thread also holds: nothing about a phase or a message crosses the
    /// channel, because both threads are looking at the same collector.
    fn run(self) {
        let Worker {
            operation,
            job,
            collector,
            cancel,
            reports,
            wake,
        } = self;
        let progress = collector.progress().cancelled_by(&cancel);
        // A job that panics still has to say something. The GUI thread cannot
        // notice the thread going on its own -- the collector it shares holds
        // the other end of this channel, so nothing here is ever dropped while
        // the viewer holds the task -- and a node locked by an operation
        // that will never answer is the worst outcome available.
        let finished = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            job(&progress)
        })) {
            Ok(finished) => finished,
            Err(_) => Finished::Failed(format!("{} stopped without an answer.", operation.name)),
        };
        // Unconditional, unlike a `Progressed`: a frame that misses this one
        // leaves the viewer busy forever.
        let _ = reports.send(Report::Done(Box::new(finished)));
        if let Some(wake) = wake.as_ref() {
            wake();
        }
    }
}

/// The collector's hook: tell the GUI thread that something was reported.
///
/// Coalesced against `queued`, which the GUI thread clears as it drains. What
/// a report says is "read the collector again", and a second copy of that
/// sentence while the first is still unread says nothing more.
fn progressed(
    reports: Sender<Report>,
    queued: Arc<AtomicBool>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
) -> Box<dyn Fn() + Send + Sync> {
    Box::new(move || {
        if queued.swap(true, Ordering::AcqRel) {
            return;
        }
        if reports.send(Report::Progressed).is_ok() {
            if let Some(wake) = wake.as_ref() {
                wake();
            }
        }
    })
}
