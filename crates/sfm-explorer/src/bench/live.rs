// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench's live evaluation: every track on a node's bench is kept
//! evaluated against its current inputs.
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
//! measurement slots and moves nothing a person put there, so it is installed
//! into the version at the cursor with
//! [`crate::document::History::replace_current_bench`]. The node is not made
//! busy by it either: every step stays available while it runs, and a step
//! taken during it is what cancels it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

use sfmtool_core::bench::{BenchItem, EditableTrack};
use sfmtool_core::progress::Progress;

use crate::document::VersionSerial;
use crate::scene::ReconId;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// What one evaluation reads, and so what says whether the measurements a
/// track carries are current.
#[derive(Clone)]
pub(crate) struct Inputs {
    /// The node whose bench holds the track.
    node: ReconId,
    /// The item's label on that bench.
    item: String,
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
        self.node == other.node
            && self.item == other.item
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
    /// The track with its measurement slots filled.
    Track(Box<EditableTrack>),
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
/// track ended.
#[derive(Default)]
pub(crate) struct Evaluations {
    /// The evaluation on a worker, if there is one.
    running: Option<Running>,
    /// For each track, by node and label, the inputs its last evaluation was
    /// installed for and whether it failed. The inputs hold the track value
    /// that was installed, so a track that has not moved since compares equal.
    settled: HashMap<(ReconId, String), (Inputs, Result<(), String>)>,
}

impl AppState {
    /// Where the evaluation of `item` on `id`'s bench stands, or `None` when
    /// there is no such track.
    pub(crate) fn bench_evaluation(&self, id: ReconId, item: &str) -> Option<Evaluation> {
        let inputs = self.evaluation_inputs(id, item)?;
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
        self.evaluation_inputs(id, item)
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
            let current = self.evaluation_inputs(running.inputs.node, &running.inputs.item);
            if !current.is_some_and(|current| current.same(&running.inputs)) {
                running.cancel.store(true, Ordering::Relaxed);
            }
            return landed;
        }
        self.forget_settled_tracks();
        let Some(inputs) = self.next_unevaluated() else {
            return landed;
        };
        let key = (inputs.node, inputs.item.clone());
        let started = self
            .bench_evaluate_job(inputs.node, &inputs.item)
            .and_then(|job| self.spawn_evaluation(inputs.clone(), job));
        if let Err(message) = started {
            self.bench_evaluations
                .settled
                .insert(key, (inputs, Err(message)));
        }
        true
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
        let current = self.evaluation_inputs(inputs.node, &inputs.item);
        if !current.is_some_and(|current| current.same(&inputs)) {
            return;
        }
        let key = (inputs.node, inputs.item.clone());
        match measured {
            // Cancelled with the inputs current again -- an edit and its undo
            // inside one evaluation -- leaves the track unevaluated, and the
            // next frame starts it over.
            Measured::Cancelled => {}
            Measured::Failed(message) => {
                self.bench_evaluations
                    .settled
                    .insert(key, (inputs, Err(message)));
            }
            Measured::Track(track) => {
                let Ok(index) = self.node_index(inputs.node) else {
                    return;
                };
                let track = Arc::new(*track);
                let history = &mut self.scene[index].history;
                let Ok(next) = history
                    .current_bench()
                    .replace(&inputs.item, BenchItem::Track(Arc::clone(&track)))
                else {
                    return;
                };
                history.replace_current_bench(Arc::new(next));
                self.bench_evaluations
                    .settled
                    .insert(key, (Inputs { track, ..inputs }, Ok(())));
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

    /// What an evaluation of `item` on `id`'s bench would read right now.
    fn evaluation_inputs(&self, id: ReconId, item: &str) -> Option<Inputs> {
        let node = self.node(id)?;
        let track = node.history.current_bench().track(item)?;
        Some(Inputs {
            node: id,
            item: item.to_string(),
            track: Arc::clone(track),
            document: node.history.current_version().document_serial,
        })
    }

    /// Where the evaluation of `inputs` stands.
    fn evaluation_of(&self, inputs: &Inputs) -> Evaluation {
        if let Err(why) = sfmtool_core::bench::evaluate_preconditions(&inputs.track) {
            return Evaluation::Refused(format!("Cannot evaluate {}: {why}", inputs.item));
        }
        let key = (inputs.node, inputs.item.clone());
        match self.bench_evaluations.settled.get(&key) {
            Some((done, outcome)) if done.same(inputs) => match outcome {
                Ok(()) => Evaluation::Current,
                Err(message) => Evaluation::Failed(message.clone()),
            },
            _ => Evaluation::Evaluating,
        }
    }

    /// The next track to evaluate: the first in scene order whose evaluation
    /// stands at [`Evaluation::Evaluating`], the active track of each bench
    /// ahead of its other items, skipping any node an operation holds.
    fn next_unevaluated(&self) -> Option<Inputs> {
        for node in &self.scene {
            if self.busy_refusal(node.id).is_some() {
                continue;
            }
            let bench = node.history.current_bench();
            let active = super::active_track_label(bench);
            let items = active
                .into_iter()
                .chain(bench.labels().filter(|label| Some(*label) != active));
            for item in items {
                let Some(inputs) = self.evaluation_inputs(node.id, item) else {
                    continue;
                };
                if self.evaluation_of(&inputs) == Evaluation::Evaluating {
                    return Some(inputs);
                }
            }
        }
        None
    }

    /// Drop what is remembered about tracks that are no longer on any bench.
    fn forget_settled_tracks(&mut self) {
        let scene = &self.scene;
        self.bench_evaluations.settled.retain(|(node, item), _| {
            scene
                .iter()
                .find(|n| n.id == *node)
                .is_some_and(|n| n.history.current_bench().track(item).is_some())
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
