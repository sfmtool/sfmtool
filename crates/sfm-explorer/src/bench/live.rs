// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench's live evaluation: every track on a node's bench, and the viewed
//! track, is kept evaluated against its current inputs.
//!
//! See `specs/gui/bench.md` § "Live evaluation". The invariant is that the
//! measurements a track shows are the evaluation of its current inputs, or an
//! evaluation of those inputs is running or about to start. Nobody asks for an
//! evaluation: [`AppState::drive_bench_evaluation`] runs once per frame, finds
//! a track whose inputs have no evaluation, and starts one.
//!
//! **What an evaluation reads** is exactly what
//! [`AppState::bench_evaluate_job`] captures, and [`Inputs`] holds the same
//! things: the track value and the document half of the version it stands in
//! (the poses and the camera intrinsics the kernels project with). The search
//! radius is the track's own `max_shift_px`, so it is part of the track value.
//! Every edit to a track gives it a new `Arc`, and every edit to the document
//! half gives the version a new document serial. So there is no list of steps
//! that have to remember to ask for an evaluation: a step changes one of the
//! two and the next frame sees that no evaluation matches. Undo and redo land on another
//! version, which is the same case.
//!
//! **One evaluation runs at a time.** When an input changes while one runs, it
//! is cancelled, and the next starts only once the cancelled one has reported
//! back. A slider drag that changes the inputs on every frame therefore has at
//! most one worker behind it, and the result a worker brings back is installed
//! only when its inputs are still the track's.
//!
//! **An evaluation pushes no version and writes no Action Log row.** It fills
//! measurement slots and sets each unpinned verdict to what the bars propose,
//! moving nothing a person put there, so it is installed
//! into the version at the cursor with
//! [`crate::document::History::replace_current_bench`]. The node is not made
//! busy by it either: every step stays available while it runs, and a step
//! taken during it is what cancels it.
//!
//! **The viewed track is the second kind of subject** ([`Subject::Viewed`]):
//! the selected point read as an editable track off every bench, which Track
//! View draws while no item is focused ([`crate::bench::viewed`]). It follows
//! the same freshness rule, and it is evaluated before any bench track, since
//! it is what the panel shows. Its result is installed into the viewed-track
//! cache, never into a version. A change of the viewed track (another point,
//! another node, a document edit or an undo that moves the document serial)
//! cancels its running evaluation, as a step on a bench track cancels that
//! track's.
//!
//! **A repaint that moves a verdict is read once more.** The readings it
//! returns were taken under the `in` set before the repaint, so the track it
//! lands carries core's repaint mark
//! (`sfmtool_core::bench::EditableTrack::repainted`) and is not recorded as
//! settled; the next frame evaluates it again, and core reads a marked track
//! without repainting, so that second evaluation settles it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

use sfmtool_core::bench::{BenchItem, EditableTrack, ItemId};
use sfmtool_core::progress::Progress;

use crate::document::VersionSerial;
use crate::scene::ReconId;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// Which track an evaluation is of: an item on a node's bench, or the viewed
/// track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Subject {
    /// An item on `node`'s bench, by the ID a rename keeps.
    Item { node: ReconId, item: ItemId },
    /// The viewed track of `point` on `node` ([`crate::bench::viewed`]).
    Viewed { node: ReconId, point: u32 },
}

impl Subject {
    /// The node the track is read against.
    fn node(self) -> ReconId {
        match self {
            Subject::Item { node, .. } | Subject::Viewed { node, .. } => node,
        }
    }
}

/// What one evaluation reads, and so what says whether the measurements a
/// track carries are current.
#[derive(Clone)]
pub(crate) struct Inputs {
    /// The track the evaluation is of.
    subject: Subject,
    /// The track value. Compared by address: every step on a track gives it a
    /// new `Arc`, and holding this one keeps the address from being reused.
    track: Arc<EditableTrack>,
    /// The version whose document half the track is read against: the poses
    /// and the camera intrinsics.
    document: VersionSerial,
}

impl Inputs {
    /// Whether an evaluation of `self` is an evaluation of `other`.
    fn same(&self, other: &Inputs) -> bool {
        self.subject == other.subject
            && Arc::ptr_eq(&self.track, &other.track)
            && self.document == other.document
    }
}

/// Where a track's evaluation stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Evaluation {
    /// The measurements the track carries are the evaluation of its current
    /// inputs.
    Current,
    /// The track's inputs have changed since it was last evaluated, and an
    /// evaluation of the current inputs is running or starts on the next frame.
    /// It waits while an operation holds the node, since that operation's
    /// answer would replace the inputs it read.
    Evaluating,
    /// The track cannot be evaluated as it stands, in
    /// `sfmtool_core::bench::evaluate_preconditions`'s words. Nothing runs
    /// until a step changes that.
    Refused(String),
    /// The evaluation of the current inputs failed, in its own words. It is
    /// not retried until an input changes.
    Failed(String),
}

impl Evaluation {
    /// The word the wire and the panel use for the state.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Evaluation::Current => "current",
            Evaluation::Evaluating => "evaluating",
            Evaluation::Refused(_) => "refused",
            Evaluation::Failed(_) => "failed",
        }
    }

    /// The sentence a refusal or a failure carries.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn reason(&self) -> Option<&str> {
        match self {
            Evaluation::Refused(why) | Evaluation::Failed(why) => Some(why),
            Evaluation::Current | Evaluation::Evaluating => None,
        }
    }
}

/// How one evaluation ended.
pub(crate) enum Measured {
    /// The track with its measurement slots filled, and how many unpinned
    /// rows its repaint turned `in` and `out` (`EvaluateReport::turned_in` and
    /// `turned_out`).
    Track(Box<EditableTrack>, (usize, usize)),
    /// Asked to stop, and did.
    Cancelled,
    /// The kernel refused, in its own words.
    Failed(String),
}

/// The work one evaluation does, as a function of the `Progress` it polls its
/// cancellation through.
pub(crate) type EvaluationJob = Box<dyn for<'a> FnOnce(&Progress<'a>) -> Measured + Send>;

/// The evaluation that is running.
struct Running {
    /// What it was started on.
    inputs: Inputs,
    /// Set to ask it to stop.
    cancel: Arc<AtomicBool>,
    /// Where the worker sends how it ended.
    done: Receiver<Measured>,
}

/// The live evaluation's state: the one running, and how the last one of each
/// bench track ended. How the viewed track's last evaluation ended is kept on
/// the viewed track itself ([`crate::bench::viewed::ViewedTrack::evaluation`]).
#[derive(Default)]
pub(crate) struct Evaluations {
    /// The evaluation on a worker, if there is one.
    running: Option<Running>,
    /// For each bench track, by node and item, the inputs its last evaluation
    /// was installed for and whether it failed. The inputs hold the track
    /// value that was installed, so a track that has not moved since compares
    /// equal.
    settled: HashMap<(ReconId, ItemId), (Inputs, Result<(), String>)>,
}

impl AppState {
    /// Where the evaluation of `item` on `id`'s bench stands, or `None` when
    /// there is no such track.
    pub(crate) fn bench_evaluation(&self, id: ReconId, item: &str) -> Option<Evaluation> {
        let item = self.bench(id)?.id(item)?;
        let inputs = self.evaluation_inputs(Subject::Item { node: id, item })?;
        Some(self.evaluation_of(&inputs))
    }

    /// Whether an evaluation of `item`'s current inputs is on a worker right
    /// now, as against waiting for the node or for another track's evaluation
    /// to finish.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn bench_evaluation_running(&self, id: ReconId, item: &str) -> bool {
        let Some(running) = self.bench_evaluations.running.as_ref() else {
            return false;
        };
        self.bench(id)
            .and_then(|bench| bench.id(item))
            .and_then(|item| self.evaluation_inputs(Subject::Item { node: id, item }))
            .is_some_and(|current| current.same(&running.inputs))
    }

    /// Whether an evaluation of the viewed track's current inputs is on a
    /// worker right now.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn viewed_evaluation_running(&self) -> bool {
        let Some(running) = self.bench_evaluations.running.as_ref() else {
            return false;
        };
        self.viewed_track()
            .and_then(|viewed| {
                self.evaluation_inputs(Subject::Viewed {
                    node: viewed.node,
                    point: viewed.point,
                })
            })
            .is_some_and(|current| current.same(&running.inputs))
    }

    /// Install a finished evaluation, and start or stop one so that the
    /// evaluation running is of current inputs.
    ///
    /// Called once per frame, after the steps the frame applied. Returns
    /// whether anything changed that the next frame has to draw.
    pub(crate) fn drive_bench_evaluation(&mut self) -> bool {
        let landed = self.poll_bench_evaluation();
        if let Some(running) = self.bench_evaluations.running.as_ref() {
            let current = self.evaluation_inputs(running.inputs.subject);
            if !current.is_some_and(|current| current.same(&running.inputs)) {
                running.cancel.store(true, Ordering::Relaxed);
            }
            return landed;
        }
        self.forget_settled_tracks();
        let Some(inputs) = self.next_unevaluated() else {
            return landed;
        };
        let started = self
            .evaluate_job(inputs.subject)
            .and_then(|job| self.spawn_evaluation(inputs.clone(), job));
        if let Err(message) = started {
            self.record_failure(inputs, message);
        }
        true
    }

    /// The work an evaluation of `subject` does, read from its current inputs.
    fn evaluate_job(&mut self, subject: Subject) -> Result<EvaluationJob, String> {
        match subject {
            Subject::Item { node, item } => {
                let label = self.item_label(node, item);
                self.bench_evaluate_job(node, &label)
            }
            Subject::Viewed { .. } => self.viewed_evaluate_job(),
        }
    }

    /// Record that the evaluation of `inputs` failed with `message`, so it is
    /// not retried until an input changes.
    fn record_failure(&mut self, inputs: Inputs, message: String) {
        match inputs.subject {
            Subject::Item { node, item } => {
                self.bench_evaluations
                    .settled
                    .insert((node, item), (inputs, Err(message)));
            }
            Subject::Viewed { .. } => {
                self.install_viewed_evaluation(&inputs.track, None, Evaluation::Failed(message));
            }
        }
    }

    /// Take the running evaluation's answer if it has one, and install it.
    fn poll_bench_evaluation(&mut self) -> bool {
        let Some(running) = self.bench_evaluations.running.as_ref() else {
            return false;
        };
        let measured = match running.done.try_recv() {
            Ok(measured) => measured,
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => {
                Measured::Failed("The evaluation stopped without an answer.".to_string())
            }
        };
        let running = self
            .bench_evaluations
            .running
            .take()
            .expect("just borrowed");
        self.land_evaluation(running.inputs, measured);
        true
    }

    /// Install what an evaluation brought back, if its inputs are still the
    /// track's, and drop it otherwise.
    fn land_evaluation(&mut self, inputs: Inputs, measured: Measured) {
        let current = self.evaluation_inputs(inputs.subject);
        if !current.is_some_and(|current| current.same(&inputs)) {
            return;
        }
        let item = match inputs.subject {
            Subject::Item { item, .. } => item,
            Subject::Viewed { .. } => {
                self.land_viewed_evaluation(inputs, measured);
                return;
            }
        };
        let key = (inputs.subject.node(), item);
        match measured {
            // Cancelled with the inputs current again -- an edit and its undo
            // inside one evaluation -- leaves the track unevaluated, and the
            // next frame starts it over.
            Measured::Cancelled => {}
            Measured::Failed(message) => self.record_failure(inputs, message),
            Measured::Track(track, turned) => {
                let Ok(index) = self.node_index(key.0) else {
                    return;
                };
                let track = Arc::new(*track);
                let history = &mut self.scene[index].history;
                let bench = history.current_bench();
                let Some(label) = bench.label_of(item) else {
                    return;
                };
                let label = label.to_string();
                let Ok(next) = bench.replace(&label, BenchItem::Track(Arc::clone(&track))) else {
                    return;
                };
                history.replace_current_bench(Arc::new(next));
                // The repaint judged the rows against the bitmap the track
                // now has, which a step that handed the reference back to the
                // rule could not: the log says what the bars moved.
                if turned != (0, 0) {
                    self.action_log.record(
                        crate::action_log::Kind::Bench,
                        format!(
                            "Evaluated {label}: the bars turned {} in and {} out",
                            turned.0, turned.1
                        ),
                    );
                }
                // A repaint that moved a verdict left readings taken under the
                // old `in` set, so the track is not settled: the next frame
                // reads it again, and that evaluation, of a track carrying the
                // repaint's mark, does not repaint.
                if track.repainted() {
                    self.bench_evaluations.settled.remove(&key);
                    return;
                }
                self.bench_evaluations
                    .settled
                    .insert(key, (Inputs { track, ..inputs }, Ok(())));
            }
        }
    }

    /// Install what an evaluation of the viewed track brought back into the
    /// viewed-track cache. Nothing goes into a version and no row is written.
    fn land_viewed_evaluation(&mut self, inputs: Inputs, measured: Measured) {
        match measured {
            Measured::Cancelled => {}
            Measured::Failed(message) => self.record_failure(inputs, message),
            // The viewed track's rows are pinned, so a repaint moves nothing
            // on it; the rule is still the bench's, so a track carrying the
            // repaint mark is read once more like any other.
            Measured::Track(track, _) => {
                let track = Arc::new(*track);
                let evaluation = if track.repainted() {
                    Evaluation::Evaluating
                } else {
                    Evaluation::Current
                };
                self.install_viewed_evaluation(&inputs.track, Some(track), evaluation);
            }
        }
    }

    /// Run `job` on a worker thread as the evaluation of `inputs`.
    fn spawn_evaluation(&mut self, inputs: Inputs, job: EvaluationJob) -> Result<(), String> {
        let cancel = Arc::new(AtomicBool::new(false));
        let (sender, done) = std::sync::mpsc::channel();
        let flag = Arc::clone(&cancel);
        let wake = self.wake.clone();
        std::thread::Builder::new()
            .name("sfm-explorer bench evaluation".to_string())
            .spawn(move || {
                let progress = Progress::none().cancelled_by(&flag);
                // A job that panics still answers, or the track would read
                // "evaluating" for the rest of the session.
                let measured =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&progress)))
                        .unwrap_or_else(|_| {
                            Measured::Failed(
                                "The evaluation stopped without an answer.".to_string(),
                            )
                        });
                let _ = sender.send(measured);
                if let Some(wake) = wake.as_ref() {
                    wake();
                }
            })
            .map_err(|e| format!("Could not start a worker thread for the evaluation: {e}"))?;
        self.bench_evaluations.running = Some(Running {
            inputs,
            cancel,
            done,
        });
        Ok(())
    }

    /// What an evaluation of `subject` would read right now: for a bench item,
    /// the track at its node's cursor; for the viewed track, the current
    /// viewed track when it is of that point.
    fn evaluation_inputs(&self, subject: Subject) -> Option<Inputs> {
        match subject {
            Subject::Item { node: id, item } => {
                let node = self.node(id)?;
                let bench = node.history.current_bench();
                let track = bench.track(bench.label_of(item)?)?;
                Some(Inputs {
                    subject,
                    track: Arc::clone(track),
                    document: node.history.current_version().document_serial,
                })
            }
            Subject::Viewed { node, point } => {
                let viewed = self
                    .viewed_track()
                    .filter(|viewed| viewed.node == node && viewed.point == point)?;
                Some(Inputs {
                    subject,
                    track: Arc::clone(&viewed.track),
                    document: viewed.document,
                })
            }
        }
    }

    /// Where the evaluation of `inputs` stands.
    fn evaluation_of(&self, inputs: &Inputs) -> Evaluation {
        let (node, item) = match inputs.subject {
            Subject::Item { node, item } => (node, item),
            // The viewed track carries its own state, set when it is built and
            // when an evaluation of it lands.
            Subject::Viewed { .. } => {
                return self
                    .viewed_track()
                    .filter(|viewed| Arc::ptr_eq(&viewed.track, &inputs.track))
                    .map_or(Evaluation::Evaluating, |viewed| viewed.evaluation.clone());
            }
        };
        if let Err(why) = sfmtool_core::bench::evaluate_preconditions(&inputs.track) {
            let label = self.item_label(node, item);
            return Evaluation::Refused(format!("Cannot evaluate {label}: {why}"));
        }
        match self.bench_evaluations.settled.get(&(node, item)) {
            Some((done, outcome)) if done.same(inputs) => match outcome {
                Ok(()) => Evaluation::Current,
                Err(message) => Evaluation::Failed(message.clone()),
            },
            _ => Evaluation::Evaluating,
        }
    }

    /// The next track to evaluate whose evaluation stands at
    /// [`Evaluation::Evaluating`]: the viewed track, then the focused item,
    /// since whichever of the two there is is what Track View shows, and then
    /// every other item in scene order, skipping any node an operation holds.
    fn next_unevaluated(&self) -> Option<Inputs> {
        let viewed = self.viewed_track().map(|viewed| Subject::Viewed {
            node: viewed.node,
            point: viewed.point,
        });
        let focused = self.focused_item().map(|f| (f.node, f.item));
        let others = self.scene.iter().flat_map(|node| {
            node.history
                .current_bench()
                .entries()
                .iter()
                .map(move |entry| (node.id, entry.id))
        });
        let items = focused
            .into_iter()
            .chain(others.filter(|key| Some(*key) != focused))
            .map(|(node, item)| Subject::Item { node, item });
        viewed
            .into_iter()
            .chain(items)
            .filter(|subject| self.busy_refusal(subject.node()).is_none())
            .filter_map(|subject| self.evaluation_inputs(subject))
            .find(|inputs| self.evaluation_of(inputs) == Evaluation::Evaluating)
    }

    /// The label `item` carries on `id`'s bench at the cursor, or an empty
    /// string when it is not there.
    fn item_label(&self, id: ReconId, item: ItemId) -> String {
        self.bench(id)
            .and_then(|bench| bench.label_of(item))
            .unwrap_or_default()
            .to_string()
    }

    /// Drop what is remembered about tracks that are no longer on any bench.
    fn forget_settled_tracks(&mut self) {
        let scene = &self.scene;
        self.bench_evaluations.settled.retain(|(node, item), _| {
            scene
                .iter()
                .find(|n| n.id == *node)
                .is_some_and(|n| n.history.current_bench().label_of(*item).is_some())
        });
    }

    /// Run the live evaluation until every track is current, refused or
    /// failed, blocking on the worker: the test seam, as
    /// [`AppState::finish_background_task`] is for an operation.
    #[cfg(test)]
    pub(crate) fn settle_bench_evaluation(&mut self) {
        const PATIENCE: std::time::Duration = std::time::Duration::from_secs(120);
        loop {
            self.drive_bench_evaluation();
            let Some(running) = self.bench_evaluations.running.as_ref() else {
                return;
            };
            let measured = running
                .done
                .recv_timeout(PATIENCE)
                .expect("the evaluation answered");
            let running = self
                .bench_evaluations
                .running
                .take()
                .expect("just borrowed");
            self.land_evaluation(running.inputs, measured);
        }
    }

    /// The track the running evaluation read, if one is running: for the tests
    /// that edit a track while an evaluation of it runs.
    #[cfg(test)]
    pub(crate) fn running_evaluation_track(&self) -> Option<Arc<EditableTrack>> {
        self.bench_evaluations
            .running
            .as_ref()
            .map(|running| Arc::clone(&running.inputs.track))
    }

    /// Whether the running evaluation has been asked to stop, for the tests
    /// that move its inputs while it runs.
    #[cfg(test)]
    pub(crate) fn running_evaluation_cancelled(&self) -> bool {
        self.bench_evaluations
            .running
            .as_ref()
            .is_some_and(|running| running.cancel.load(Ordering::Relaxed))
    }

    /// Wait for the running evaluation's answer and land it, as a frame would
    /// once the worker has answered.
    #[cfg(test)]
    pub(crate) fn land_running_evaluation(&mut self) {
        let Some(running) = self.bench_evaluations.running.take() else {
            return;
        };
        let measured = running
            .done
            .recv_timeout(std::time::Duration::from_secs(120))
            .expect("the evaluation answered");
        self.land_evaluation(running.inputs, measured);
    }
}
