// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The proposal's **job**: the switch and, while the preview is asked for, the
//! retriangulation of the proposal's tracks, run off the GUI thread while a
//! proposal is open.
//!
//! See `specs/gui/edits/switch-camera-model.md` § "The proposal's job". It is
//! not a background task: it locks nothing, pushes no version, writes no
//! Action Log row and is not shown in the Background Task panel, so it neither
//! holds off an operation on another node nor is held off by one. It follows
//! the bench's live evaluation (`crate::bench::live`) instead:
//!
//! - **Its inputs** are the proposal's generation and the document half of the
//!   version it opened on, and whether the retriangulation is asked for.
//!   Nobody asks for a job: once per frame
//!   [`AppState::drive_camera_model_proposal_job`] finds a proposal whose
//!   inputs have no answer, and starts one.
//! - **One job runs at a time.** When the inputs change while one runs, it is
//!   cancelled through its `Progress`, and the next starts once the cancelled
//!   one has reported back. An answer is kept only when it is for the
//!   proposal's current generation.
//!
//! One job runs [`switch_and_retriangulate`], the computation Apply's
//! background task runs, so the version it holds is the version the task would
//! push. It sends the switch's report, which carries the observation
//! comparison, as soon as the switch returns, and the answer when the
//! retriangulation does. A job asked only for the comparison stops after the
//! switch.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::switch_camera_model::{switch_camera_model, CameraSwitch};

use sfmtool_core::{RowMap, SfmrReconstruction};

use super::preview::{PreviewCounts, PreviewTargets};
use crate::background::SwitchedCamera;
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

/// What a job sends back.
enum Message {
    /// The switch returned: its report for the camera, or its refusal.
    Switched(Result<Box<CameraSwitch>, String>),
    /// The retriangulation returned: the answer, or the refusal.
    Answered(Result<Box<ProposalAnswer>, String>),
}

/// The work one job does, as a function of the `Progress` it polls its
/// cancellation through and the channel it sends what it found on.
type Work = Box<dyn FnOnce(&Progress<'_>, &dyn Fn(Message)) + Send>;

/// What one job was started on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Inputs {
    /// The proposal's generation.
    generation: u64,
    /// The document half of the version the proposal opened on.
    document: VersionSerial,
    /// Whether the job runs the retriangulation after the switch.
    triangulate: bool,
}

/// The job that is running.
struct Running {
    inputs: Inputs,
    /// Set to ask it to stop.
    cancel: Arc<AtomicBool>,
    /// Where it sends what it found.
    done: Receiver<Message>,
}

/// The proposal job's state on [`AppState`]: the one running, if any. What it
/// found is kept on the proposal itself, so it goes with the proposal.
#[derive(Default)]
pub(crate) struct ProposalJobs {
    running: Option<Running>,
}

impl ProposalJobs {
    /// Whether a job is running, cancelled or not.
    #[cfg(test)]
    pub(crate) fn is_running(&self) -> bool {
        self.running.is_some()
    }
}

impl AppState {
    /// Take what the running job has found, and start or stop one so that the
    /// job running is of the open proposal's current inputs.
    ///
    /// Called once per frame. Returns whether anything changed that the next
    /// frame has to draw.
    pub(crate) fn drive_camera_model_proposal_job(&mut self) -> bool {
        let landed = self.poll_camera_model_proposal_job();
        let wanted = self.proposal_job_inputs();
        if let Some(running) = self.proposal_jobs.running.as_ref() {
            let current = wanted.is_some_and(|wanted| {
                wanted.generation == running.inputs.generation
                    && wanted.document == running.inputs.document
            });
            if !current {
                running.cancel.store(true, Ordering::Relaxed);
            }
            return landed;
        }
        let Some(inputs) = wanted else {
            return landed;
        };
        let Some(job) = self.proposal_job(inputs) else {
            return landed;
        };
        self.spawn_proposal_job(inputs, job);
        true
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

    /// The work a job of `inputs` does, read from the open proposal.
    fn proposal_job(&self, inputs: Inputs) -> Option<Work> {
        let proposal = self.camera_model_proposal.as_ref()?;
        let id = proposal.recon;
        let resolved = match self.resolve_switch(id, &proposal.request()) {
            Ok(resolved) => resolved,
            // A busy node: the job waits for it, as the bench's evaluation
            // waits for an operation holding its node.
            Err(_) => return None,
        };
        let value = Arc::clone(&proposal.value);
        let rows = proposal.rows.clone();
        let (base_count, addition_count) = (proposal.base_count, proposal.addition_count);
        Some(Box::new(move |progress, send| {
            run_job(
                &value,
                rows,
                &resolved,
                inputs,
                (base_count, addition_count),
                progress,
                send,
            );
        }))
    }

    /// Run `job` on a worker thread as the job of `inputs`.
    fn spawn_proposal_job(&mut self, inputs: Inputs, job: Work) {
        let cancel = Arc::new(AtomicBool::new(false));
        let (sender, done) = std::sync::mpsc::channel();
        let flag = Arc::clone(&cancel);
        let wake = self.wake.clone();
        let spawned = std::thread::Builder::new()
            .name("sfm-explorer camera model proposal".to_string())
            .spawn(move || {
                let progress = Progress::none().cancelled_by(&flag);
                // Each message wakes the window, so the comparison is shown as
                // soon as the switch returns rather than when the job ends.
                let send = |message: Message| {
                    let _ = sender.send(message);
                    if let Some(wake) = wake.as_ref() {
                        wake();
                    }
                };
                // A job that panics still ends, so the next one can start.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    job(&progress, &send)
                }));
                drop(sender);
                if let Some(wake) = wake.as_ref() {
                    wake();
                }
            });
        if spawned.is_ok() {
            self.proposal_jobs.running = Some(Running {
                inputs,
                cancel,
                done,
            });
        }
    }

    /// Take every message the running job has sent, keeping each that is for
    /// the open proposal's current generation, and forget the job once it has
    /// ended.
    fn poll_camera_model_proposal_job(&mut self) -> bool {
        let mut changed = false;
        loop {
            let Some(running) = self.proposal_jobs.running.as_ref() else {
                return changed;
            };
            let inputs = running.inputs;
            match running.done.try_recv() {
                Ok(message) => {
                    self.keep_proposal_message(inputs, message);
                    changed = true;
                }
                Err(TryRecvError::Empty) => return changed,
                Err(TryRecvError::Disconnected) => {
                    self.proposal_jobs.running = None;
                    return true;
                }
            }
        }
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
        }
    }

    /// Run the open proposal's jobs until it has all it asks for, blocking on
    /// the worker: the test seam, as `settle_bench_evaluation` is for the
    /// bench.
    #[cfg(test)]
    pub(crate) fn settle_camera_model_proposal_job(&mut self) {
        const PATIENCE: std::time::Duration = std::time::Duration::from_secs(120);
        loop {
            self.drive_camera_model_proposal_job();
            let Some(running) = self.proposal_jobs.running.as_ref() else {
                return;
            };
            let inputs = running.inputs;
            match running.done.recv_timeout(PATIENCE) {
                Ok(message) => self.keep_proposal_message(inputs, message),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    self.proposal_jobs.running = None;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    panic!("the proposal's job said nothing for {PATIENCE:?}")
                }
            }
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
        // The comparison alone: the switch, and nothing after it.
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
