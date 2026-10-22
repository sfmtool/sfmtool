// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Table panel's **table job**: the stored columns of the track
//! table, computed off the GUI thread for the node the panel shows.
//!
//! See `specs/gui/track-table.md` § "Computing the columns". The job is a
//! **quiet task** in the fast queue ([`Operation::TRACK_TABLE_COLUMNS`],
//! `specs/gui/background-tasks.md` § "Quiet tasks"): it locks nothing, pushes
//! no version, writes no Action Log row and is not reported on the wire. Its
//! driver owns no thread and follows the camera model proposal's
//! (`crate::camera_model_proposal::job`):
//!
//! - **Its inputs** are the node the panel showed in this frame and the
//!   document serial at that node's cursor. The panel says which node it shows
//!   each time it draws ([`AppState::want_track_table_columns`]), and the frame
//!   clears that before the dock draws ([`AppState::hide_track_table`]), so a
//!   hidden panel asks for nothing. Twice per frame, before the MCP drain and
//!   after the egui pass, [`AppState::drive_track_table_columns`] queues a job when the track table
//!   cache lacks a stored column the node can supply.
//! - **One job is queued at a time.** A job runs [`TrackTable::compute`] for
//!   the missing columns of the value at the cursor when it was queued, and
//!   sends the table back. The GUI thread merges it into the cache with
//!   [`TrackTable::merge`], as *Find points* does, so a column either of them
//!   computed is kept. A table for a serial the node has left is dropped.
//! - **An edit cancels it.** When the document serial at its node's cursor
//!   moves while the job runs, or the node is closed, the job is asked to stop
//!   through its `Progress`, and the next is queued once it has reported back.
//!   A waiting job that is no longer wanted is taken out of the queue.
//! - **It does not wait for its node.** The answer is keyed by the serial it
//!   read, so the job runs on a node a slow task holds, against the version on
//!   screen ([`crate::background::QuietSource::waits_for_its_node`]). It gives
//!   way to any task queued in the fast queue, and the driver queues it again
//!   behind that task.

use std::sync::mpsc::{Receiver, TryRecvError};

use sfmtool_core::analysis::track_table::{ColumnCost, TrackTable, TrackTableColumn};
use sfmtool_core::progress::Progress;

use crate::background::{Operation, QuietJob, QuietRank, QuietSource};
use crate::document::VersionSerial;
use crate::scene::ReconId;
use crate::state::AppState;

/// What one job was queued on: the node and the document serial at its
/// cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Inputs {
    node: ReconId,
    document: VersionSerial,
}

/// What comes back on a job's channel.
enum Message {
    /// The columns, computed for the job's inputs.
    Computed(Box<TrackTable>),
    /// The queue dropped the job before it started, because a newer request
    /// replaced it. It never ran.
    Dropped,
}

/// The job the driver has queued: waiting in the fast queue or running from
/// it.
struct Queued {
    /// What it was queued on.
    inputs: Inputs,
    /// The quiet task it runs as.
    task: u64,
    /// Where it sends the table. The channel ends when the task's work has
    /// returned or been dropped, which is the job's end.
    messages: Receiver<Message>,
}

/// The table job's state on [`AppState`].
#[derive(Default)]
pub(crate) struct TrackTableJobs {
    /// The node the Track Table panel showed in this frame, or `None` while it
    /// is hidden or shows no reconstruction.
    wanted: Option<ReconId>,
    /// The job queued as a quiet task, waiting or running, if there is one.
    queued: Option<Queued>,
    /// A gate the next job's work waits at before it starts, so a test can
    /// act while the job is known to be running and unfinished.
    #[cfg(test)]
    pub(super) hold_next: Option<Receiver<()>>,
}

impl TrackTableJobs {
    /// The id of the quiet task the queued job runs as, if one is queued.
    #[cfg(test)]
    pub(super) fn queued_task(&self) -> Option<u64> {
        self.queued.as_ref().map(|queued| queued.task)
    }
}

/// The stored columns `node`'s value at its cursor can supply that `held`
/// does not hold, in [`TrackTableColumn::ALL`] order.
pub(super) fn missing_stored_columns(
    edited: &sfmtool_core::EditedReconstruction,
    held: Option<&TrackTable>,
) -> Vec<TrackTableColumn> {
    TrackTableColumn::ALL
        .into_iter()
        .filter(|c| c.cost() == ColumnCost::Stored)
        .filter(|c| c.is_available(edited))
        .filter(|&c| !held.is_some_and(|table| table.has(c)))
        .collect()
}

/// The work one job does: `columns` computed for `edited`, sent on `sender`
/// with a wake of the window. A cancelled compute sends nothing, and the
/// channel's end tells the driver the job is over.
fn work(
    edited: sfmtool_core::EditedReconstruction,
    columns: Vec<TrackTableColumn>,
    sender: std::sync::mpsc::Sender<Message>,
    wake: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
) -> impl for<'a> FnOnce(&Progress<'a>) + Send + 'static {
    move |progress: &Progress<'_>| {
        if let Ok(table) = TrackTable::compute(&edited, &columns, progress) {
            let _ = sender.send(Message::Computed(Box::new(table)));
            if let Some(wake) = wake.as_ref() {
                wake();
            }
        }
    }
}

/// Whether a job's work, run with the cancel flag already set over a
/// reconstruction of several blocks of points, stops without sending a
/// table: the test of [`Operation::TRACK_TABLE_COLUMNS`]'s claim to be
/// cancellable.
#[cfg(test)]
pub(crate) fn job_stops_when_cancelled() -> bool {
    let edited = sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(
        sfmtool_core::SfmrReconstruction::demo(20_000),
    ));
    let columns = missing_stored_columns(&edited, None);
    let (sender, messages) = std::sync::mpsc::channel();
    let cancel = std::sync::atomic::AtomicBool::new(true);
    work(edited, columns, sender, None)(&Progress::none().cancelled_by(&cancel));
    messages.try_recv().is_err()
}

impl AppState {
    /// Say that the Track Table panel shows `node` in this frame, so the table
    /// job keeps its columns current.
    pub(super) fn want_track_table_columns(&mut self, node: ReconId) {
        self.track_table_jobs.wanted = Some(node);
    }

    /// Forget which node the Track Table panel showed: called by the frame
    /// before the dock draws, so a frame in which the panel is hidden asks for
    /// nothing.
    pub(crate) fn hide_track_table(&mut self) {
        self.track_table_jobs.wanted = None;
    }

    /// Take what the queued job has computed, and queue or stop one so that
    /// the job queued is for the node the panel shows, at its cursor.
    ///
    /// Called twice per frame, before the MCP drain and after the egui pass.
    /// Returns whether anything changed that the next
    /// frame has to draw.
    pub(crate) fn drive_track_table_columns(&mut self) -> bool {
        let mut changed = self.poll_track_table_job();
        let wanted = self.track_table_job_inputs();
        if let Some(queued) = self.track_table_jobs.queued.as_ref() {
            if self.quiet_task_running(queued.task) {
                // Still the version at its node's cursor: let it finish, even
                // where the panel has moved on, since what it computes is kept.
                // An edit, an undo or a close asks it to stop.
                let current = self
                    .node(queued.inputs.node)
                    .map(|node| node.history.current_version().document_serial)
                    == Some(queued.inputs.document);
                if !current {
                    let task = queued.task;
                    self.cancel_quiet_task(task);
                }
                return changed;
            }
            if wanted.as_ref().map(|(inputs, _)| *inputs) == Some(queued.inputs) {
                return changed;
            }
        }
        let stale = self.track_table_jobs.queued.take();
        if let Some((inputs, columns)) = wanted {
            if self.queue_track_table_job(inputs, columns) {
                return true;
            }
        }
        if let Some(stale) = stale {
            self.cancel_quiet_task(stale.task);
            changed = true;
        }
        changed
    }

    /// The inputs of the job the panel needs, with the columns it computes,
    /// or `None` when the panel is hidden or the cache already holds every
    /// stored column the node can supply.
    fn track_table_job_inputs(&self) -> Option<(Inputs, Vec<TrackTableColumn>)> {
        let node = self.node(self.track_table_jobs.wanted?)?;
        let columns = missing_stored_columns(node.edited(), self.track_table(node.id));
        if columns.is_empty() {
            return None;
        }
        Some((
            Inputs {
                node: node.id,
                document: node.history.current_version().document_serial,
            },
            columns,
        ))
    }

    /// Queue the job of `inputs`, computing `columns`, as a *Track Table
    /// columns* quiet task in the fast queue, replacing the one the driver has
    /// waiting there, if any. False when its worker could not be spawned; the
    /// next frame tries again.
    fn queue_track_table_job(&mut self, inputs: Inputs, columns: Vec<TrackTableColumn>) -> bool {
        let Some(node) = self.node(inputs.node) else {
            return false;
        };
        let edited = node.edited().clone();
        let label = node.label.clone();
        let (sender, messages) = std::sync::mpsc::channel();
        let dropped = sender.clone();
        let work = work(edited, columns, sender, self.wake.clone());
        #[cfg(test)]
        let hold = self.track_table_jobs.hold_next.take();
        let run = move |progress: &Progress<'_>| {
            #[cfg(test)]
            if let Some(hold) = hold {
                let _ = hold.recv();
            }
            work(progress);
        };
        let started = self.start_quiet_task(
            Operation::TRACK_TABLE_COLUMNS,
            QuietSource::TrackTableColumns,
            QuietRank::TrackTableColumns,
            inputs.node,
            label,
            QuietJob::new(run, move || {
                let _ = dropped.send(Message::Dropped);
            }),
        );
        let Ok(task) = started else {
            return false;
        };
        self.track_table_jobs.queued = Some(Queued {
            inputs,
            task,
            messages,
        });
        true
    }

    /// Take every message the queued job has sent, merging a table that is
    /// for its node's current serial into the cache, and forget the job once
    /// it has ended or was dropped.
    fn poll_track_table_job(&mut self) -> bool {
        let mut changed = false;
        loop {
            let Some(queued) = self.track_table_jobs.queued.as_ref() else {
                return changed;
            };
            let inputs = queued.inputs;
            match queued.messages.try_recv() {
                Ok(Message::Computed(table)) => {
                    self.land_track_table_columns(inputs, *table);
                    changed = true;
                }
                Ok(Message::Dropped) | Err(TryRecvError::Disconnected) => {
                    self.track_table_jobs.queued = None;
                    return true;
                }
                Err(TryRecvError::Empty) => return changed,
            }
        }
    }

    /// Merge a table the job computed for `inputs` into the cache, or drop it
    /// when its node has been closed or its cursor has moved since.
    fn land_track_table_columns(&mut self, inputs: Inputs, table: TrackTable) {
        let current = self
            .node(inputs.node)
            .map(|node| node.history.current_version().document_serial);
        if current != Some(inputs.document) {
            return;
        }
        if let Err(error) = self.track_tables.merge(inputs.node, inputs.document, table) {
            // Equal serials hold equal rows, so this is a bug, not a state a
            // person can reach; the next frame queues the job again.
            log::warn!("Track Table columns: {error}");
        }
    }

    /// Run the table job until the cache holds every stored column of the
    /// node the panel last asked for, blocking on the fast queue: the test
    /// seam, as `settle_camera_model_proposal_job` is for the proposal.
    #[cfg(test)]
    pub(super) fn settle_track_table_columns(&mut self) {
        loop {
            self.drive_track_table_columns();
            if self.track_table_jobs.queued.is_none() {
                self.land_quiet_tasks();
                return;
            }
            self.wait_for_track_table_job();
        }
    }

    /// Block until the queued job has ended or was dropped, driving the fast
    /// queue meanwhile and landing what it sends.
    #[cfg(test)]
    pub(super) fn wait_for_track_table_job(&mut self) {
        loop {
            let Some(queued) = self.track_table_jobs.queued.as_ref() else {
                return;
            };
            let inputs = queued.inputs;
            match queued.messages.try_recv() {
                Ok(Message::Computed(table)) => self.land_track_table_columns(inputs, *table),
                Ok(Message::Dropped) => {
                    self.track_table_jobs.queued = None;
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    let task = queued.task;
                    self.track_table_jobs.queued = None;
                    while self.quiet_task_running(task) {
                        self.step_queue(crate::background::Queue::Fast);
                    }
                    return;
                }
                Err(TryRecvError::Empty) => {
                    if !self.step_queue(crate::background::Queue::Fast) {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }
}
