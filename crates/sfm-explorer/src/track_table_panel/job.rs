// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Table panel's **table job**: the stored columns of the track
//! table, and then the computed columns the panel needs, computed off the GUI
//! thread for the node the panel shows.
//!
//! See `specs/gui/track-table.md` § "Computing the columns". The job is a
//! **quiet task** in the fast queue ([`Operation::TRACK_TABLE_COLUMNS`],
//! `specs/gui/background-tasks.md` § "Quiet tasks"): it locks nothing, pushes
//! no version, writes no Action Log row and is not reported on the wire. Its
//! driver owns no thread and follows the camera model proposal's
//! (`crate::camera_model_proposal::job`):
//!
//! - **Its inputs** are the node the panel showed in this frame, the document
//!   serial at that node's cursor and the columns the job computes. Each time
//!   it draws, the panel says which node it shows and which computed columns a
//!   visible column, a bound or the sort needs
//!   ([`AppState::want_track_table_columns`]), and the frame clears that
//!   before the dock draws ([`AppState::hide_track_table`]), so a hidden panel
//!   asks for nothing. Twice per frame, before the MCP drain and after the
//!   egui pass, [`AppState::drive_track_table_columns`] queues a job when the
//!   track table cache lacks a stored column the node can supply, or, once it
//!   holds them all, a computed column the panel needs.
//! - **The stored columns come first.** A job computes the missing stored
//!   columns before the missing computed ones the panel needs
//!   ([`columns_to_compute`]), so the table is usable while the computed
//!   columns, which cost a pass over every pair of rays or a triangulation per
//!   point, are still being computed. A computed column that no visible
//!   column, bound or sort needs is not computed. After an edit, while the
//!   panel keeps the old version's rows on show, the computed columns a bound
//!   or the sort names are computed in the same run as the stored ones, so the
//!   old rows stay until the whole query can apply to the new version rather
//!   than the rows falling back to another order in between.
//! - **One job is queued at a time.** A job runs [`TrackTable::compute`] for
//!   its columns of the value at the cursor when it was queued, and sends the
//!   table back. The GUI thread merges it into the cache with
//!   [`TrackTable::merge`], as *Find points* does, so a column either of them
//!   computed is kept. A table for a serial the node has left is dropped. A
//!   waiting job whose columns are no longer the ones needed is replaced.
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

/// What the Track Table panel asked for in this frame: the node it shows and
/// the computed columns a visible column, a bound or the sort needs, in
/// [`TrackTableColumn::ALL`] order.
struct Wanted {
    node: ReconId,
    computed: Vec<TrackTableColumn>,
    /// The computed columns among `computed` that hold a bound or the sort.
    ordering: Vec<TrackTableColumn>,
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
    /// The columns it computes.
    columns: Vec<TrackTableColumn>,
    /// The quiet task it runs as.
    task: u64,
    /// Where it sends the table. The channel ends when the task's work has
    /// returned or been dropped, which is the job's end.
    messages: Receiver<Message>,
}

/// The table job's state on [`AppState`].
#[derive(Default)]
pub(crate) struct TrackTableJobs {
    /// The node the Track Table panel showed in this frame, with the computed
    /// columns it needs, or `None` while it is hidden or shows no
    /// reconstruction.
    wanted: Option<Wanted>,
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

    /// The columns the queued job computes, if one is queued.
    #[cfg(test)]
    pub(super) fn queued_columns(&self) -> Option<&[TrackTableColumn]> {
        self.queued.as_ref().map(|queued| queued.columns.as_slice())
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

/// The columns the next job computes for `edited`, given the table `held`
/// for its version and the computed columns the panel needs, `computed`: the
/// stored columns it can supply that `held` lacks, with the columns of
/// `with_stored` it lacks, or, once it holds every stored column, the columns
/// of `computed` it lacks, in [`TrackTableColumn::ALL`] order. Empty when there
/// is nothing to compute.
///
/// The stored columns go first so that the table is usable while the
/// computed ones, which cost far more per point, are still being computed.
/// `with_stored` names the computed columns that go in the same run as the
/// stored ones: the ones a bound or the sort names while the panel keeps an
/// earlier version's rows on show, which stay until the whole query applies.
pub(super) fn columns_to_compute(
    edited: &sfmtool_core::EditedReconstruction,
    held: Option<&TrackTable>,
    computed: &[TrackTableColumn],
    with_stored: &[TrackTableColumn],
) -> Vec<TrackTableColumn> {
    let missing_computed = |names: &[TrackTableColumn]| {
        TrackTableColumn::ALL
            .into_iter()
            .filter(|c| names.contains(c))
            .filter(|c| c.cost() == ColumnCost::Computed && c.is_available(edited))
            .filter(|&c| !held.is_some_and(|table| table.has(c)))
            .collect::<Vec<_>>()
    };
    let mut stored = missing_stored_columns(edited, held);
    if !stored.is_empty() {
        stored.extend(missing_computed(with_stored));
        return stored;
    }
    missing_computed(computed)
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
    /// Say that the Track Table panel shows `node` in this frame and needs the
    /// computed columns `computed`, of which `ordering` hold a bound or the
    /// sort, so the table job keeps the stored columns and those current.
    pub(super) fn want_track_table_columns(
        &mut self,
        node: ReconId,
        computed: Vec<TrackTableColumn>,
        ordering: Vec<TrackTableColumn>,
    ) {
        self.track_table_jobs.wanted = Some(Wanted {
            node,
            computed,
            ordering,
        });
    }

    /// Forget which node the Track Table panel showed, and which points it
    /// wanted picked: called by the frame before the dock draws, so a frame in
    /// which the panel is hidden asks for nothing.
    pub(crate) fn hide_track_table(&mut self) {
        self.track_table_jobs.wanted = None;
        self.hide_reference_picks();
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
            // Waiting: kept while it is still the job needed, replaced when
            // the node, its serial or the columns needed have changed.
            if wanted.as_ref().is_some_and(|(inputs, columns)| {
                *inputs == queued.inputs && *columns == queued.columns
            }) {
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
    /// stored column the node can supply and every computed column the panel
    /// needs.
    fn track_table_job_inputs(&self) -> Option<(Inputs, Vec<TrackTableColumn>)> {
        let wanted = self.track_table_jobs.wanted.as_ref()?;
        let node = self.node(wanted.node)?;
        let document = node.history.current_version().document_serial;
        // An earlier version's table still cached is what the panel keeps on
        // show after an edit: its rows stay until the bounds and the sort can
        // apply to the new version, so the computed columns they name come
        // with the stored ones.
        let earlier_on_show = self
            .track_tables
            .latest(node.id)
            .is_some_and(|(cached, _)| cached != document);
        let with_stored: &[TrackTableColumn] = if earlier_on_show {
            &wanted.ordering
        } else {
            &[]
        };
        let columns = columns_to_compute(
            node.edited(),
            self.track_table(node.id),
            &wanted.computed,
            with_stored,
        );
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
        let work = work(edited, columns.clone(), sender, self.wake.clone());
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
            columns,
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
    /// node the panel last asked for and every computed column it needed,
    /// blocking on the fast queue: the test
    /// seam, as `settle_camera_model_proposal_job` is for the proposal.
    #[cfg(test)]
    pub(crate) fn settle_track_table_columns(&mut self) {
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
