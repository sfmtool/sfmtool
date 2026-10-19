// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The proposal's **job**: the switch and, while the preview is asked for, the
//! retriangulation of the proposal's tracks, run off the GUI thread while a
//! proposal is open.
//!
//! See `specs/gui/edits/switch-camera-model.md` § "The proposal's job". A job
//! is a **quiet task** in the fast queue ([`Operation::CAMERA_MODEL_PROPOSAL`],
//! `specs/gui/background-tasks.md` § "Quiet tasks"): it locks nothing, pushes
//! no version, writes no Action Log row and is not reported on the wire. The
//! driver owns no thread, and follows the bench's live evaluation
//! (`crate::bench::live`):
//!
//! - **Its inputs** are the proposal's generation and the document half of the
//!   version it opened on, and whether the retriangulation is asked for.
//!   Nobody asks for a job: once per frame
//!   [`AppState::drive_camera_model_proposal_job`] finds a proposal whose
//!   inputs have no answer, and queues one, which answers on a channel of the
//!   driver's.
//! - **One job is queued at a time.** When the proposal's generation or version
//!   moves on while one runs, it is cancelled through its `Progress`, and the
//!   next is queued once the cancelled one has reported back; one that is still
//!   waiting is replaced by the request for the new inputs. Clearing the
//!   preview lets a job finish, so ticking it again finds the answer rather
//!   than starting over. A message is kept only when it is for the proposal's
//!   current generation.
//! - **It gives way to every task someone asks for.** A task queued in the fast
//!   queue asks it to stop and starts once it has; the driver then queues it
//!   again, behind that task, and it starts over. Among quiet tasks it goes
//!   first ([`QuietRank::CameraModelProposal`]), ahead of the viewed track's
//!   evaluation. A waiting job that the queue drops or replaces answers that it
//!   was dropped, which is neither an answer nor a failure: the driver queues
//!   it again when it next wants the job.
//!
//! One job runs [`switch_and_retriangulate`], the computation Apply's
//! background task runs, so the version it holds is the version the task would
//! push. It sends the switch's report, which carries the observation
//! comparison, as soon as the switch returns, and the answer when the
//! retriangulation does. A job asked only for the comparison stops after the
//! switch.

use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::switch_camera_model::{switch_camera_model, CameraSwitch};

use sfmtool_core::{RowMap, SfmrReconstruction};

use super::preview::{PreviewCounts, PreviewTargets};
use crate::background::{Operation, QuietJob, QuietRank, QuietSource, SwitchedCamera};
use crate::document::VersionSerial;
use crate::state::edits::{
    switch_and_retriangulate, switch_refusal_in_version_numbering, ResolvedSwitch, SwitchEnd,
};
use crate::state::AppState;

/// What a job's answer holds: the version Apply installs, and the preview
/// built from the retriangulation's report.
pub(crate) struct ProposalAnswer {
    /// The version, its label and its Action Log sentence, as the `Switch
    /// camera model` task would push them.
    pub(crate) run: SwitchedCamera,
    /// The preview buffers' rows.
    pub(crate) targets: Arc<PreviewTargets>,
    /// The banner's counts.
    pub(crate) counts: PreviewCounts,
}

/// What comes back on a job's channel.
enum Message {
    /// The switch returned: its report for the camera, or its refusal.
    Switched(Result<Box<CameraSwitch>, String>),
    /// The retriangulation returned: the answer, or the refusal.
    Answered(Result<Box<ProposalAnswer>, String>),
    /// The queue dropped the job before it started, because a newer request
    /// replaced it or its node became locked while it waited. It never ran,
    /// so it carries neither an answer nor a failure.
    Dropped,
}

/// The work one job does, as a function of the `Progress` it polls its
/// cancellation through and the channel it sends what it found on.
type Work = Box<dyn FnOnce(&Progress<'_>, &dyn Fn(Message)) + Send>;

/// What one job was queued on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Inputs {
    /// The proposal's generation.
    generation: u64,
    /// The document half of the version the proposal opened on.
    document: VersionSerial,
    /// Whether the job runs the retriangulation after the switch.
    triangulate: bool,
}

/// The job the driver has queued: waiting in the fast queue or running from
/// it.
struct Queued {
    /// What it was queued on.
    inputs: Inputs,
    /// The quiet task it runs as.
    task: u64,
    /// Where it sends what it found. The channel ends when the task's work has
    /// returned or been dropped, which is the job's end.
    messages: Receiver<Message>,
}

/// The proposal job's state on [`AppState`]: the one queued, if any. What it
/// found is kept on the proposal itself, so it goes with the proposal.
#[derive(Default)]
pub(crate) struct ProposalJobs {
    /// The job queued as a quiet task, waiting or running, if there is one.
    queued: Option<Queued>,
    /// A gate the next job's work waits at before it starts, so a test can
    /// queue a task while the job is known to be running and unfinished.
    #[cfg(test)]
    pub(crate) hold_next: Option<Receiver<()>>,
}

impl ProposalJobs {
    /// Whether a job is queued, waiting or running, cancelled or not.
    #[cfg(test)]
    pub(crate) fn is_queued(&self) -> bool {
        self.queued.is_some()
    }

    /// The id of the quiet task the queued job runs as, if one is queued.
    #[cfg(test)]
    pub(crate) fn queued_task(&self) -> Option<u64> {
        self.queued.as_ref().map(|queued| queued.task)
    }
}

impl AppState {
    /// Take what the queued job has found, and queue or stop one so that the
    /// job queued is of the open proposal's current inputs.
    ///
    /// Called once per frame. Returns whether anything changed that the next
    /// frame has to draw.
    pub(crate) fn drive_camera_model_proposal_job(&mut self) -> bool {
        let mut changed = self.poll_camera_model_proposal_job();
        let wanted = self.proposal_job_inputs();
        if let Some(queued) = self.proposal_jobs.queued.as_ref() {
            let current = self.camera_model_proposal.as_ref().is_some_and(|proposal| {
                proposal.generation == queued.inputs.generation
                    && proposal.opened_on == queued.inputs.document
            });
            if self.quiet_task_running(queued.task) {
                // Still the open proposal's inputs: let it finish, even where
                // the preview was cleared meanwhile, so ticking it again finds
                // the answer rather than starting over. Otherwise it is asked
                // to stop, and the next is queued once it has reported back.
                if !current {
                    let task = queued.task;
                    self.cancel_quiet_task(task);
                }
                return changed;
            }
            // A waiting job is kept while it is for the current inputs and
            // does at least what is wanted. Otherwise it is replaced below by
            // the request for the inputs wanted, or taken out when there are
            // none.
            let covers =
                wanted.is_none_or(|wanted| queued.inputs.triangulate || !wanted.triangulate);
            if current && covers {
                return changed;
            }
        }
        let stale = self.proposal_jobs.queued.take();
        if wanted.is_some_and(|inputs| self.queue_proposal_job(inputs)) {
            return true;
        }
        if let Some(stale) = stale {
            self.cancel_quiet_task(stale.task);
            changed = true;
        }
        changed
    }

    /// The inputs of the job the open proposal needs, or `None` when it has
    /// all it can have: the comparison for its generation, and the answer too
    /// while the preview is asked for on a node that can be retriangulated.
    fn proposal_job_inputs(&self) -> Option<Inputs> {
        let proposal = self.camera_model_proposal.as_ref()?;
        proposal.fit.as_ref().ok()?;
        let generation = proposal.generation;
        let has_comparison = proposal
            .comparison
            .as_ref()
            .is_some_and(|(g, _)| *g == generation);
        let triangulate = proposal.wants_triangulation();
        let has_answer = proposal
            .triangulation
            .as_ref()
            .is_some_and(|(g, _)| *g == generation);
        // A switch the job found refused has no retriangulation to follow.
        let switch_refused = proposal
            .comparison
            .as_ref()
            .is_some_and(|(g, found)| *g == generation && found.is_err());
        let needs = !has_comparison || (triangulate && !has_answer && !switch_refused);
        needs.then_some(Inputs {
            generation,
            document: proposal.opened_on,
            triangulate,
        })
    }

    /// The work a job of `inputs` does, read from the open proposal, with the
    /// node it reads and the label the Background Tasks panel's quiet line
    /// names it by.
    fn proposal_job(&self, inputs: Inputs) -> Option<(Work, crate::scene::ReconId, String)> {
        let proposal = self.camera_model_proposal.as_ref()?;
        let id = proposal.recon;
        let resolved = match self.resolve_switch(id, &proposal.request()) {
            Ok(resolved) => resolved,
            // A busy node: the job waits for it, as the bench's evaluation
            // waits for an operation holding its node.
            Err(_) => return None,
        };
        let label = format!("{}: camera {}", proposal.label, proposal.camera);
        let value = Arc::clone(&proposal.value);
        let rows = proposal.rows.clone();
        let (base_count, addition_count) = (proposal.base_count, proposal.addition_count);
        let work: Work = Box::new(move |progress, send| {
            run_job(
                &value,
                rows,
                &resolved,
                inputs,
                (base_count, addition_count),
                progress,
                send,
            );
        });
        Some((work, id, label))
    }

    /// Queue the job of `inputs` as a *Camera model proposal* quiet task in
    /// the fast queue, replacing the one the driver has waiting there, if any.
    /// False when there is no job to run, because the node is busy, or its
    /// worker could not be spawned; the next frame tries again.
    fn queue_proposal_job(&mut self, inputs: Inputs) -> bool {
        let Some((work, node, label)) = self.proposal_job(inputs) else {
            return false;
        };
        let (sender, messages) = std::sync::mpsc::channel();
        let dropped = sender.clone();
        let wake = self.wake.clone();
        #[cfg(test)]
        let hold = self.proposal_jobs.hold_next.take();
        let run = move |progress: &Progress<'_>| {
            #[cfg(test)]
            if let Some(hold) = hold {
                let _ = hold.recv();
            }
            // Each message wakes the window, so the comparison is shown as
            // soon as the switch returns rather than when the job ends. The
            // queue's worker wakes it once more when the work has returned.
            let send = |message: Message| {
                let _ = sender.send(message);
                if let Some(wake) = wake.as_ref() {
                    wake();
                }
            };
            work(progress, &send);
        };
        let started = self.start_quiet_task(
            Operation::CAMERA_MODEL_PROPOSAL,
            QuietSource::CameraModelProposal,
            QuietRank::CameraModelProposal,
            node,
            label,
            QuietJob::new(run, move || {
                let _ = dropped.send(Message::Dropped);
            }),
        );
        let Ok(task) = started else {
            return false;
        };
        self.proposal_jobs.queued = Some(Queued {
            inputs,
            task,
            messages,
        });
        true
    }

    /// Take every message the queued job has sent, keeping each that is for
    /// the open proposal's current generation, and forget the job once it has
    /// ended or was dropped.
    fn poll_camera_model_proposal_job(&mut self) -> bool {
        let mut changed = false;
        loop {
            let Some(queued) = self.proposal_jobs.queued.as_ref() else {
                return changed;
            };
            let inputs = queued.inputs;
            match queued.messages.try_recv() {
                Ok(message) => {
                    self.take_proposal_message(inputs, message);
                    changed = true;
                }
                Err(TryRecvError::Empty) => return changed,
                // The work returned, cancelled or not, or was dropped with its
                // task: either way the job is over, and what it sent has been
                // read.
                Err(TryRecvError::Disconnected) => {
                    self.proposal_jobs.queued = None;
                    return true;
                }
            }
        }
    }

    /// Act on one message from the queued job of `inputs`. One that says the
    /// job was dropped unrun forgets the job, keeping nothing on the proposal,
    /// so the driver queues it again when it next wants it.
    fn take_proposal_message(&mut self, inputs: Inputs, message: Message) {
        if let Message::Dropped = message {
            self.proposal_jobs.queued = None;
            return;
        }
        self.keep_proposal_message(inputs, message);
    }

    /// Keep what a job of `inputs` sent on the open proposal, when it is still
    /// for that proposal's generation, and drop it otherwise.
    fn keep_proposal_message(&mut self, inputs: Inputs, message: Message) {
        let Some(proposal) = self.camera_model_proposal.as_mut() else {
            return;
        };
        if proposal.generation != inputs.generation || proposal.opened_on != inputs.document {
            return;
        }
        match message {
            Message::Switched(found) => proposal.comparison = Some((inputs.generation, found)),
            Message::Answered(found) => {
                proposal.triangulation = Some((inputs.generation, found));
            }
            Message::Dropped => {}
        }
    }

    /// Run the open proposal's jobs until it has all it asks for, blocking on
    /// the fast queue: the test seam, as `settle_bench_evaluation` is for the
    /// bench.
    ///
    /// While a job is queued, the fast queue's running task -- the job, or a
    /// task it waits behind -- is driven to its next report and applied. Once
    /// nothing is queued, the quiet task whose messages were read last is
    /// landed too, so the fast queue is idle for the test's next start.
    #[cfg(test)]
    pub(crate) fn settle_camera_model_proposal_job(&mut self) {
        loop {
            self.drive_camera_model_proposal_job();
            if self.proposal_jobs.queued.is_none() {
                self.land_quiet_tasks();
                return;
            }
            self.wait_for_proposal_job();
        }
    }

    /// Whether a proposal's job on camera 1 of `self`, run with the cancel
    /// flag already set, stops without sending anything, with the preview and
    /// without it: the test of [`Operation::CAMERA_MODEL_PROPOSAL`]'s claim to
    /// be cancellable.
    #[cfg(test)]
    pub(crate) fn camera_model_proposal_job_stops_when_cancelled(mut self) -> bool {
        let id = self.scene[0].id;
        self.open_camera_model_proposal(crate::scene::CameraRef::new(id, 1), None)
            .expect("opens");
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let progress = Progress::none().cancelled_by(&cancel);
        [false, true].into_iter().all(|triangulate| {
            let proposal = self.camera_model_proposal.as_ref().expect("open");
            let inputs = Inputs {
                generation: proposal.generation,
                document: proposal.opened_on,
                triangulate,
            };
            let (work, ..) = self.proposal_job(inputs).expect("a job");
            let sent = std::cell::Cell::new(0);
            work(&progress, &|_| sent.set(sent.get() + 1));
            sent.get() == 0
        })
    }

    /// Block until the queued job has ended or was dropped, driving the fast
    /// queue meanwhile and keeping what it sends.
    #[cfg(test)]
    pub(crate) fn wait_for_proposal_job(&mut self) {
        loop {
            let Some(queued) = self.proposal_jobs.queued.as_ref() else {
                return;
            };
            let inputs = queued.inputs;
            match queued.messages.try_recv() {
                Ok(message) => {
                    self.take_proposal_message(inputs, message);
                    continue;
                }
                Err(TryRecvError::Disconnected) => {
                    self.proposal_jobs.queued = None;
                    return;
                }
                Err(TryRecvError::Empty) => {}
            }
            assert!(
                self.step_queue(crate::background::Queue::Fast),
                "a proposal's job is queued in an idle fast queue"
            );
        }
    }
}

/// One job: the switch over `value` (the version's base, or its
/// materialisation with `rows` its map), its report sent at once, then, when
/// `inputs` asks for it, the retriangulation and the preview built from it.
fn run_job(
    value: &SfmrReconstruction,
    rows: Option<RowMap>,
    resolved: &ResolvedSwitch,
    inputs: Inputs,
    (base_count, addition_count): (usize, usize),
    progress: &Progress<'_>,
    send: &dyn Fn(Message),
) {
    if !inputs.triangulate {
        // The comparison alone: the switch, and nothing after it. The switch
        // takes no `Progress`, so the flag is read once, before it.
        if progress.check_cancel().is_err() {
            return;
        }
        let found = switch_camera_model(
            value,
            &[resolved.camera],
            &resolved.target,
            &resolved.options,
        )
        .map(|(_, report)| {
            Box::new(
                report
                    .cameras
                    .into_iter()
                    .next()
                    .expect("one camera was named"),
            )
        })
        .map_err(|e| {
            format!(
                "Switch camera model of {} refused: {}",
                resolved.label,
                switch_refusal_in_version_numbering(e, rows.as_ref())
            )
        });
        send(Message::Switched(found));
        return;
    }
    let switched_sent = std::cell::Cell::new(false);
    let ran = switch_and_retriangulate(value, rows.clone(), resolved, progress, |entry| {
        send(Message::Switched(Ok(Box::new(entry.clone()))));
        switched_sent.set(true);
    });
    let (run, report) = match ran {
        Ok(ran) => ran,
        Err(SwitchEnd::Cancelled) => return,
        Err(SwitchEnd::Failed(why)) => {
            // Refused by the switch, the comparison carries the sentence; by
            // the retriangulation, the triangulation does.
            let message = if switched_sent.get() {
                Message::Answered(Err(why))
            } else {
                Message::Switched(Err(why))
            };
            send(message);
            return;
        }
    };
    if progress.check_cancel().is_err() {
        return;
    }
    let (targets, counts) = PreviewTargets::build(
        inputs.generation,
        value,
        rows.as_ref(),
        base_count,
        addition_count,
        &report,
        &run.value,
    );
    send(Message::Answered(Ok(Box::new(ProposalAnswer {
        run,
        targets: Arc::new(targets),
        counts,
    }))));
}

#[cfg(test)]
mod tests;
