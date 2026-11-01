// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench's live evaluation: every track on a node's bench, and the viewed
//! track, is kept evaluated against its current inputs.
//!
//! See `specs/gui/bench.md` § "Live evaluation". The invariant is that the
//! measurements a track shows are the evaluation of its current inputs, or an
//! evaluation of those inputs is running, waiting in the fast queue or about
//! to be queued. Nobody asks for an evaluation:
//! [`AppState::drive_bench_evaluation`] runs once per frame, finds a track
//! whose inputs have no evaluation, and queues one.
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
//! **An evaluation is a quiet task in the fast queue**
//! ([`Operation::LIVE_EVALUATION`], `specs/gui/background-tasks.md` § "Quiet
//! tasks"). The driver owns no thread: it chooses the subject and queues a
//! *Live evaluation* quiet task for it, which answers on a channel of the
//! driver's. It gives way to every task someone asks for: a task queued in the
//! fast queue, or a locking task queued on its node, asks it to stop, starts
//! once it has, and the driver queues it again, behind that task, when its
//! cancelled answer comes back. A request waiting in the queue that the queue
//! drops -- its node became locked, or a newer request replaced it -- answers
//! that it was dropped, which is not a failure: the subject is picked again
//! once its node is free.
//!
//! **One evaluation is queued at a time.** When an input changes while one
//! runs, it is cancelled, and the next is queued only once the cancelled one
//! has reported back; one that is still waiting is replaced by the request for
//! the next subject. A slider drag that changes the inputs on every frame
//! therefore has at most one worker behind it, and the result a worker brings
//! back is installed only when its inputs are still the track's and its node
//! is not locked: no evaluation lands into a locked node, so a step built from
//! the bench when it was queued still finds that bench when it starts.
//!
//! **An evaluation pushes no version, and writes no Action Log row for
//! running.** It fills measurement slots and sets each unpinned verdict to
//! what the bars propose, moving nothing a person put there, so it is
//! installed into the version at the cursor with
//! [`crate::document::History::replace_current_bench`]. The one row it writes
//! reports a change to the bench, not that it ran: when its repaint moved a
//! verdict of a bench item, *Evaluated {label}: the bars turned {n} in and
//! {m} out*. The node is not locked by it: every step stays available while
//! it runs, and a step taken during it is what cancels it.
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
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

use sfmtool_core::bench::{BenchItem, EditableTrack, ItemId};
use sfmtool_core::progress::Progress;

use crate::background::{Operation, QuietJob, QuietRank, QuietSource};
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
    /// evaluation of the current inputs is running, waits in the fast queue,
    /// or is queued on the next frame. It waits while an operation holds the
    /// node, since that operation's answer would replace the inputs it read.
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

/// What comes back on an evaluation's channel.
enum Answer {
    /// The evaluation ran, and this is how it ended.
    Measured(Measured),
    /// The queue dropped the evaluation before it started, because its node
    /// became locked while it waited or because a newer request replaced it.
    /// It never ran, so it is not a failure.
    Dropped,
}

/// The evaluation the driver has queued: waiting in the fast queue or running
/// from it.
struct QueuedEvaluation {
    /// What it was queued on.
    inputs: Inputs,
    /// The quiet task it runs as.
    task: u64,
    /// Where it answers.
    answers: Receiver<Answer>,
}

/// The live evaluation's state: the evaluation queued, and how the last one of
/// each bench track ended. How the viewed track's last evaluation ended is kept
/// on the viewed track itself ([`crate::bench::viewed::ViewedTrack::evaluation`]).
#[derive(Default)]
pub(crate) struct Evaluations {
    /// The evaluation queued as a quiet task, waiting or running, if there is
    /// one.
    queued: Option<QueuedEvaluation>,
    /// For each bench track, by node and item, the inputs its last evaluation
    /// was installed for and whether it failed. The inputs hold the track
    /// value that was installed, so a track that has not moved since compares
    /// equal.
    settled: HashMap<(ReconId, ItemId), (Inputs, Result<(), String>)>,
    /// The gate the next evaluation queued waits on before it reads anything,
    /// set by [`AppState::hold_next_evaluation`].
    #[cfg(test)]
    held: Option<std::sync::mpsc::Receiver<()>>,
}

impl AppState {
    /// Where the evaluation of `item` on `id`'s bench stands, or `None` when
    /// there is no such track.
    pub(crate) fn bench_evaluation(&self, id: ReconId, item: &str) -> Option<Evaluation> {
        let item = self.bench(id)?.id(item)?;
        let inputs = self.evaluation_inputs(Subject::Item { node: id, item })?;
        Some(self.evaluation_of(&inputs))
    }

    /// The evaluation the driver has queued when it is running from the fast
    /// queue, as against waiting there.
    fn running_evaluation(&self) -> Option<&QueuedEvaluation> {
        self.bench_evaluations
            .queued
            .as_ref()
            .filter(|queued| self.quiet_task_running(queued.task))
    }

    /// Whether an evaluation of `item`'s current inputs is on a worker right
    /// now, as against waiting in the fast queue, for the node or for another
    /// track's evaluation to finish.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn bench_evaluation_running(&self, id: ReconId, item: &str) -> bool {
        let Some(running) = self.running_evaluation() else {
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
        let Some(running) = self.running_evaluation() else {
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

    /// Install a finished evaluation, and queue or stop one so that the
    /// evaluation queued is of current inputs.
    ///
    /// Called once per frame, after the steps the frame applied. Returns
    /// whether anything changed that the next frame has to draw.
    pub(crate) fn drive_bench_evaluation(&mut self) -> bool {
        let mut changed = self.poll_bench_evaluation();
        if let Some(queued) = self.bench_evaluations.queued.as_ref() {
            let current = self.evaluation_inputs(queued.inputs.subject);
            if current.is_some_and(|current| current.same(&queued.inputs)) {
                return changed;
            }
            // A stale evaluation that is running is asked to stop, and the
            // next is queued only once it has answered: one worker at a time,
            // however many steps land. One still waiting is replaced below by
            // the request for the next subject, or taken out when there is
            // none.
            if self.quiet_task_running(queued.task) {
                let task = queued.task;
                self.cancel_quiet_task(task);
                return changed;
            }
        }
        self.forget_settled_tracks();
        let stale = self.bench_evaluations.queued.take();
        let Some((inputs, rank)) = self.next_unevaluated() else {
            if let Some(stale) = stale {
                self.cancel_quiet_task(stale.task);
                changed = true;
            }
            return changed;
        };
        if let Err(message) = self.queue_evaluation(inputs.clone(), rank) {
            if let Some(stale) = stale {
                self.cancel_quiet_task(stale.task);
            }
            self.record_failure(inputs, message);
        }
        true
    }

    /// Queue the evaluation of `inputs` as a *Live evaluation* quiet task in
    /// the fast queue, replacing the one the driver has waiting there, if any.
    fn queue_evaluation(&mut self, inputs: Inputs, rank: QuietRank) -> Result<(), String> {
        let job = self.evaluate_job(inputs.subject)?;
        #[cfg(test)]
        let job: EvaluationJob = match self.bench_evaluations.held.take() {
            Some(gate) => Box::new(move |progress: &Progress<'_>| {
                let _ = gate.recv();
                job(progress)
            }),
            None => job,
        };
        let label = self.subject_label(inputs.subject);
        let (sender, answers) = std::sync::mpsc::channel();
        let dropped = sender.clone();
        let run = move |progress: &Progress<'_>| {
            // A job that panics still answers, or the track would read
            // "evaluating" for the rest of the session.
            let measured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(progress)))
                .unwrap_or_else(|_| {
                    Measured::Failed("The evaluation stopped without an answer.".to_string())
                });
            let _ = sender.send(Answer::Measured(measured));
        };
        let task = self.start_quiet_task(
            Operation::LIVE_EVALUATION,
            QuietSource::LiveEvaluation,
            rank,
            inputs.subject.node(),
            label,
            QuietJob::new(run, move || {
                let _ = dropped.send(Answer::Dropped);
            }),
        )?;
        self.bench_evaluations.queued = Some(QueuedEvaluation {
            inputs,
            task,
            answers,
        });
        Ok(())
    }

    /// What the Background Tasks panel names an evaluation of `subject` by:
    /// the node and the item, or the node and the viewed point.
    fn subject_label(&self, subject: Subject) -> String {
        let node = self
            .node(subject.node())
            .map(|node| node.label.as_str())
            .unwrap_or_default();
        match subject {
            Subject::Item { node: id, item } => format!("{node}: {}", self.item_label(id, item)),
            Subject::Viewed { .. } => {
                let point = self
                    .viewed_track()
                    .map(|viewed| viewed.label.as_str())
                    .unwrap_or_default();
                format!("{node}: viewed {point}")
            }
        }
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

    /// Take the queued evaluation's answer if it has one, and install it.
    fn poll_bench_evaluation(&mut self) -> bool {
        let Some(queued) = self.bench_evaluations.queued.as_ref() else {
            return false;
        };
        let answer = match queued.answers.try_recv() {
            Ok(answer) => answer,
            Err(TryRecvError::Empty) => return false,
            // The job was dropped without a word, which only a worker thread
            // that could not be spawned does.
            Err(TryRecvError::Disconnected) => Answer::Measured(Measured::Failed(
                "The evaluation stopped without an answer.".to_string(),
            )),
        };
        let queued = self.bench_evaluations.queued.take().expect("just borrowed");
        self.land_answer(queued.inputs, answer);
        true
    }

    /// Act on what came back for an evaluation of `inputs`. A dropped one is
    /// not run and not a failure: the driver picks its subject again once its
    /// node is free.
    fn land_answer(&mut self, inputs: Inputs, answer: Answer) {
        match answer {
            Answer::Measured(measured) => self.land_evaluation(inputs, measured),
            Answer::Dropped => {}
        }
    }

    /// Install what an evaluation brought back, if its inputs are still the
    /// track's and its node is not locked, and drop it otherwise.
    fn land_evaluation(&mut self, inputs: Inputs, measured: Measured) {
        let current = self.evaluation_inputs(inputs.subject);
        if !current.is_some_and(|current| current.same(&inputs)) {
            return;
        }
        // No evaluation lands into a locked node. A task queued on it built
        // its job from the bench as it stood, and cancelled this evaluation
        // then; an evaluation that finished before it read the flag still
        // answers, and installing that answer would change the bench under
        // the job. The track stays unevaluated, and the driver picks it again
        // once the node is free.
        if self.busy_refusal(inputs.subject.node()).is_some() {
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
            // Cancelled with the inputs current -- an asked-for task displaced
            // it, or an edit and its undo landed inside one evaluation --
            // leaves the track unevaluated, and the driver queues it again.
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
    /// every other item in scene order, skipping any node an operation holds,
    /// with the place among quiet tasks its evaluation takes.
    fn next_unevaluated(&self) -> Option<(Inputs, QuietRank)> {
        let viewed = self.viewed_track().map(|viewed| {
            (
                Subject::Viewed {
                    node: viewed.node,
                    point: viewed.point,
                },
                QuietRank::ViewedTrack,
            )
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
            .map(|key| (key, QuietRank::FocusedItem))
            .into_iter()
            .chain(
                others
                    .filter(|key| Some(*key) != focused)
                    .map(|key| (key, QuietRank::BenchItem)),
            )
            .map(|((node, item), rank)| (Subject::Item { node, item }, rank));
        viewed
            .into_iter()
            .chain(items)
            .filter(|(subject, _)| self.busy_refusal(subject.node()).is_none())
            .filter_map(|(subject, rank)| Some((self.evaluation_inputs(subject)?, rank)))
            .find(|(inputs, _)| self.evaluation_of(inputs) == Evaluation::Evaluating)
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
    /// failed, blocking on the fast queue: the test seam, as
    /// [`AppState::finish_background_task`] is for an operation.
    ///
    /// While an evaluation is queued, the fast queue's running task -- the
    /// evaluation, or a task it waits behind -- is driven to its next report
    /// and applied. Once nothing is queued, the quiet task whose answer was
    /// read last is landed too, so the fast queue is idle for the test's next
    /// start.
    #[cfg(test)]
    pub(crate) fn settle_bench_evaluation(&mut self) {
        loop {
            self.drive_bench_evaluation();
            if self.bench_evaluations.queued.is_none() {
                self.land_quiet_tasks();
                return;
            }
            self.wait_for_evaluation();
        }
    }

    /// Block until the queued evaluation has answered, driving the fast queue
    /// meanwhile, and act on the answer.
    #[cfg(test)]
    fn wait_for_evaluation(&mut self) {
        loop {
            let Some(queued) = self.bench_evaluations.queued.as_ref() else {
                return;
            };
            match queued.answers.try_recv() {
                Ok(answer) => {
                    let queued = self.bench_evaluations.queued.take().expect("just read");
                    self.land_answer(queued.inputs, answer);
                    return;
                }
                // The drive's own poll answers a channel's end, in its words.
                Err(TryRecvError::Disconnected) => {
                    self.poll_bench_evaluation();
                    return;
                }
                Err(TryRecvError::Empty) => {}
            }
            assert!(
                self.step_queue(crate::background::Queue::Fast),
                "an evaluation is queued in an idle fast queue"
            );
        }
    }

    /// The track the queued evaluation read, if one is queued: for the tests
    /// that edit a track while an evaluation of it runs.
    #[cfg(test)]
    pub(crate) fn running_evaluation_track(&self) -> Option<Arc<EditableTrack>> {
        self.bench_evaluations
            .queued
            .as_ref()
            .map(|queued| Arc::clone(&queued.inputs.track))
    }

    /// Make the next evaluation queued wait, once it starts, until the
    /// returned sender sends or is dropped, for the tests that move its inputs
    /// while it runs: without the hold, a worker that has already answered by
    /// the next drive has its answer landed and the next evaluation queued.
    #[cfg(test)]
    pub(crate) fn hold_next_evaluation(&mut self) -> std::sync::mpsc::Sender<()> {
        let (open, gate) = std::sync::mpsc::channel();
        self.bench_evaluations.held = Some(gate);
        open
    }

    /// Whether the running evaluation has been asked to stop, for the tests
    /// that move its inputs while it runs.
    #[cfg(test)]
    pub(crate) fn running_evaluation_cancelled(&self) -> bool {
        self.bench_evaluations
            .queued
            .as_ref()
            .is_some_and(|queued| self.quiet_task_cancelled(queued.task))
    }

    /// The id of the quiet task the queued evaluation runs as, if one is
    /// queued.
    #[cfg(test)]
    pub(crate) fn queued_evaluation_task(&self) -> Option<u64> {
        self.bench_evaluations
            .queued
            .as_ref()
            .map(|queued| queued.task)
    }

    /// Wait for the queued evaluation's answer and land it, as a frame would
    /// once the worker has answered, and land its quiet task in the queue.
    #[cfg(test)]
    pub(crate) fn land_running_evaluation(&mut self) {
        self.wait_for_evaluation();
        self.land_quiet_tasks();
    }
}
