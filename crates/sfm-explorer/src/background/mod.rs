// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Long operations, running off the GUI thread from two queues.
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
//! - [`Queue`] is where an operation runs from, slow or fast. Each queue runs
//!   one task at a time and the two run at the same time; a task started while
//!   its queue is busy waits behind the ones already waiting there.
//!   [`QueueState`] is one queue's, held by [`AppState`] so the busy check is
//!   where every method that needs it already is.
//! - [`WaitingTask`] is one queued and not yet started, [`BackgroundTask`] one
//!   that is running, and [`FinishedTask`] the record of one that has finished
//!   or was cancelled before it started, kept in a ring of the last
//!   [`FINISHED_KEPT`]. All three are found by id ([`AppState::task`]).
//! - A **quiet task** ([`Operation::quiet`]) is work the viewer starts by
//!   itself to keep a panel current: the bench's live evaluation, or a camera
//!   model proposal's job. It is
//!   queued by its source with [`AppState::start_quiet_task`] as a [`QuietJob`]
//!   that answers on the source's own channel. It gives way to every task
//!   someone asks for: a task queued in its queue, or a locking task queued on
//!   its node, asks it to stop and starts once it has; it waits behind every
//!   such task; a source has at most one waiting entry; and it is dropped
//!   rather than started on a locked node. It writes no row, leaves no record
//!   and is left out of [`AppState::running_tasks`], [`AppState::waiting_tasks`],
//!   [`AppState::task`] and the [`BusyNotice`].
//! - [`Report`] is what crosses the channel, and carries no progress: phases,
//!   messages and counts are in the collector both threads hold.
//!
//! ## What the GUI thread does with it
//!
//! [`AppState::poll_background_tasks`] runs in phase 0 of the frame, before the
//! MCP drain, so a completed operation's version is on screen in the frame it
//! landed and an agent's call in that same frame reads the new value. What
//! remains of the freeze is the upload the next phase does, which for an
//! adjustment is single-digit milliseconds.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sfmtool_core::bench::BenchItem;
use sfmtool_core::progress::Progress;
use sfmtool_core::{EditedReconstruction, SfmrReconstruction};

use crate::action_log::{version_step_text, Actor, Kind};
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
    /// The kind the Action Log row it ends with carries.
    ///
    /// Stated per operation because only the wrapper knows what its answer
    /// touches: an adjustment writes the reconstruction and is an `Edit`, and a
    /// bench fit writes an item beside it and is a [`Kind::Bench`].
    pub(crate) kind: Kind,
    /// The queue it runs from. Fixed per operation rather than estimated from
    /// the reconstruction's size, so the same operation always waits behind
    /// the same work and nothing has to be measured before the queue is known.
    pub(crate) queue: Queue,
    /// Whether it locks its node from the moment it is queued until it lands.
    /// Every operation that pushes a version of its node locks it; an open
    /// makes a new node and locks none.
    pub(crate) locks: bool,
    /// Whether it is work the viewer starts by itself to keep a panel
    /// current, rather than work someone asked for: a **quiet task**
    /// (`specs/gui/background-tasks.md` § "Quiet tasks"). A quiet task gives
    /// way to every task someone asks for, writes no Action Log row for
    /// running, leaves no record in the ring and is not reported on the wire.
    /// It is started with [`AppState::start_quiet_task`], never with
    /// [`AppState::start_task`].
    pub(crate) quiet: bool,
}

/// One of the two queues a background task runs from.
///
/// Each queue runs one task at a time, and the two run at the same time
/// (`specs/gui/background-tasks.md` § "Two queues"). The slow queue holds the
/// operations whose cost grows with the whole reconstruction and that use
/// every core: the whole-reconstruction edits and Build index files. The fast
/// queue holds opening files, the other bench steps and the quiet tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Queue {
    /// The whole-reconstruction edits and Build index files.
    Slow,
    /// Opening files, the other bench steps and the quiet tasks.
    Fast,
}

impl Queue {
    /// Both queues, in the order the panel and the wire list them.
    pub(crate) const ALL: [Queue; 2] = [Queue::Slow, Queue::Fast];

    /// The queue's name on the wire.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Queue::Slow => "slow",
            Queue::Fast => "fast",
        }
    }

    /// The queue's heading in the Background Tasks panel.
    pub(crate) fn title(self) -> &'static str {
        match self {
            Queue::Slow => "Slow",
            Queue::Fast => "Fast",
        }
    }
}

/// What one queue is doing: the task it runs, if any, and the tasks waiting
/// behind it, in the order they were started.
///
/// A queue with tasks waiting always has one running: a task waits only while
/// its queue runs another, and the next one starts in the poll that lands it.
#[derive(Default)]
pub(crate) struct QueueState {
    /// The task running from this queue, or `None` while the queue is idle.
    pub(crate) running: Option<BackgroundTask>,
    /// The tasks waiting to run from this queue, the next one first.
    pub(crate) waiting: VecDeque<WaitingTask>,
}

/// A task queued behind its queue's running task and not yet started.
///
/// Its job is already built, from the value at the cursor when it was queued,
/// and its node, when it locks one, is busy from that moment: nothing can move
/// the cursor before the job starts, so the value it captured is still the one
/// at the cursor then.
pub(crate) struct WaitingTask {
    /// The id it keeps when it starts, and in the ring once it is over.
    pub(crate) id: u64,
    /// What it is.
    pub(crate) operation: Operation,
    /// The node it will install into and, when the operation locks, the node
    /// it already locks. `None` for an open.
    pub(crate) node: Option<ReconId>,
    /// That node's label as it read when the task was queued.
    pub(crate) label: String,
    /// When it was queued, which is what a tool call's wait for a direct
    /// answer is measured from. Not the Action Log row's start, which is the
    /// moment the task starts.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) queued: Instant,
    /// Who asked for it, whose row the task writes.
    pub(crate) actor: Actor,
    /// The work, built when the task was queued.
    job: WaitingJob,
}

/// The work a waiting task will do: a task someone asked for, which lands
/// through the queue, or a quiet one, which answers its source itself.
enum WaitingJob {
    /// A task someone asked for, which lands through [`AppState::finish`].
    Asked(Job),
    /// A quiet task, which answers on its source's channel.
    Quiet {
        /// Which producer of quiet work queued it. A source has at most one
        /// waiting entry.
        source: QuietSource,
        /// Its place in the fixed order among waiting quiet tasks.
        rank: QuietRank,
        /// The work.
        job: QuietJob,
    },
}

/// One producer of quiet work.
///
/// A source keeps at most one waiting entry: a new request replaces the one it
/// has waiting, because a quiet task's input is the state of a panel and only
/// the latest state matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuietSource {
    /// The camera model proposal's job ([`crate::camera_model_proposal::job`]).
    CameraModelProposal,
    /// The bench's live evaluation ([`crate::bench::live`]).
    LiveEvaluation,
}

/// A quiet task's place in the fixed order among waiting quiet tasks, the
/// first first (`specs/gui/background-tasks.md` § "Quiet tasks").
///
/// The camera model proposal's job goes first, because its comparison and its
/// preview are on screen while a person decides whether to apply it. The live
/// evaluation's place depends on its subject: the viewed track and the focused
/// item are what Track View shows, so they go ahead of the other bench items,
/// in the order the evaluation driver picks them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum QuietRank {
    /// The camera model proposal's job.
    CameraModelProposal,
    /// The live evaluation of the viewed track.
    ViewedTrack,
    /// The live evaluation of the focused bench item.
    FocusedItem,
    /// The live evaluation of any other bench item.
    BenchItem,
}

/// The work one quiet task does, and what it says when it never runs.
///
/// A quiet task's answer goes to its source on the source's own channel, not
/// through the queue, so a job with more than one answer sends each when it
/// has it and the queue carries only that the task has finished. `dropped` is
/// called instead of `run`, on the GUI thread, when the task is dropped before
/// it starts (its node became locked) or replaced by a newer request from its
/// source; it sends the source's *dropped* message, so the source does not read
/// the end of its channel as a failure.
pub(crate) struct QuietJob {
    /// The work, run on the worker.
    run: Box<dyn for<'a> FnOnce(&Progress<'a>) + Send>,
    /// What tells the source the work will not run.
    dropped: Box<dyn FnOnce() + Send>,
}

impl QuietJob {
    /// A quiet job that does `run` on its worker, or calls `dropped` when it
    /// never starts.
    pub(crate) fn new(
        run: impl for<'a> FnOnce(&Progress<'a>) + Send + 'static,
        dropped: impl FnOnce() + Send + 'static,
    ) -> Self {
        QuietJob {
            run: Box::new(run),
            dropped: Box::new(dropped),
        }
    }
}

impl WaitingTask {
    /// Drop a quiet task that will never run, telling its source so. A task
    /// someone asked for is dropped with its job, unrun.
    fn drop_unrun(self) {
        if let WaitingJob::Quiet { job, .. } = self.job {
            (job.dropped)();
        }
    }

    /// The source that queued it, when it is quiet.
    fn quiet_source(&self) -> Option<QuietSource> {
        match &self.job {
            WaitingJob::Quiet { source, .. } => Some(*source),
            WaitingJob::Asked(_) => None,
        }
    }
}

/// How many finished tasks [`AppState`] keeps, newest last.
///
/// A deferred reply and Surface Tiling's wait find their task here by id once
/// it has landed, and two tasks can land in one frame now that two run at a
/// time, so one slot would let a fast task push a slow task's record out
/// between its landing and the poll that reads it.
pub(crate) const FINISHED_KEPT: usize = 16;

/// What [`AppState::task`] found for an id: a task waiting to start, one
/// running, or the record of one that has finished.
pub(crate) enum TaskState<'a> {
    /// The task is queued behind its queue's running task.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    Waiting(&'a WaitingTask),
    /// The task is running.
    Running(&'a BackgroundTask),
    /// The task has finished, and this is what it came to.
    Finished(&'a FinishedTask),
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
        kind: Kind::Edit,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// One node converted from `sift_files` to `embedded_patches`, the minimal
    /// conversion `sfm xform --to-embedded-patches` runs
    /// (`specs/core/patch/sift-to-patch-reconstruction.md`).
    ///
    /// Cancellable because `SfmrReconstruction::to_embedded_patches` polls the
    /// flag between its three stages, between the images of both its `.sift`
    /// walks, and at intervals through each pass over the points and the
    /// observations, and hands back `ReconstructionError::Cancelled` rather
    /// than a half-converted value. Each of those stages reports a fraction
    /// too, so the panel draws a bar for the whole of it rather than the
    /// spinner it falls back to for an operation that has reported no number.
    pub(crate) const TO_EMBEDDED_PATCHES: Operation = Operation {
        name: "Convert to embedded patches",
        cancellable: true,
        kind: Kind::Edit,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// Every point of one node re-solved from its own observations, at the
    /// poses and the lenses the value already holds
    /// (`specs/gui/edits/retriangulate-point.md`).
    ///
    /// Cancellable because `sfmtool_core::reconstruction::retriangulate_points`
    /// polls the flag between its stages and hands back
    /// `RetriangulateError::Cancelled` rather than a value whose points were
    /// half re-solved.
    pub(crate) const RETRIANGULATE_ALL_POINTS: Operation = Operation {
        name: "Retriangulate all points",
        cancellable: true,
        kind: Kind::Edit,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// Every point of one node fitted to the render of its reference
    /// observation, as Track View's *Fit* and *Commit* do for one track
    /// (`specs/gui/edits/refit-all-points.md`).
    ///
    /// Cancellable because the photographs are decoded with the flag polled
    /// between them, and `sfmtool_core::bench::refit_points` polls it in front
    /// of each stage and of each point's fit and hands back
    /// `RefitPointsError::Cancelled` rather than a value some of whose points
    /// were refit.
    pub(crate) const REFIT_ALL_POINTS: Operation = Operation {
        name: "Refit all points",
        cancellable: true,
        kind: Kind::Edit,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// One camera of one node switched to the camera model its proposal was
    /// fitted to, and the tracks its images observe retriangulated through the
    /// new model, as one version: the camera model proposal's Apply
    /// (`specs/gui/edits/switch-camera-model.md`).
    ///
    /// Cancellable on either side of the switch, which takes no `Progress`
    /// and is a fit and a pass over the camera's observations, and inside
    /// `sfmtool_core::reconstruction::retriangulate_points`, which polls the
    /// flag between its stages and hands back `RetriangulateError::Cancelled`
    /// rather than a value with some of the tracks re-solved.
    pub(crate) const SWITCH_CAMERA_MODEL: Operation = Operation {
        name: "Switch camera model",
        cancellable: true,
        kind: Kind::Edit,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// One node's coarse observations retired where a finer tracked one covers
    /// them (`specs/gui/edits/prune-covered-observations.md`).
    ///
    /// Cancellable because
    /// `sfmtool_core::reconstruction::prune_covered_observations` polls the
    /// flag between its stages and hands back `PruneCoveredError::Cancelled`
    /// rather than a value some of whose tracks were shortened.
    pub(crate) const PRUNE_COVERED_OBSERVATIONS: Operation = Operation {
        name: "Prune covered observations",
        cancellable: true,
        kind: Kind::Edit,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// One image added to the tracks of its node it can see
    /// (`specs/gui/edits/add-image-to-tracks.md`).
    ///
    /// Cancellable because the photographs are decoded with the flag polled
    /// between them, and `sfmtool_core::reconstruction::add_image_to_tracks`
    /// polls it while it measures the candidates and hands back
    /// `AddImageToTracksError::Cancelled` rather than a value with some of
    /// the observations added.
    pub(crate) const ADD_IMAGE_TO_TRACKS: Operation = Operation {
        name: "Add image to tracks",
        cancellable: true,
        kind: Kind::Edit,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// One bench track fitted: localized, re-triangulated, its reference
    /// view's tile stored as the patch bitmap, and read back
    /// (`specs/gui/track-view.md`).
    ///
    /// Cancellable: the fit polls the flag on either side of the decode,
    /// between its rounds, and inside the localizer between the views it
    /// renders -- which is where a widened search spends its time -- and in
    /// the reading it ends with, and hands back `FitError::Cancelled` rather
    /// than a half-fitted track.
    pub(crate) const BENCH_FIT: Operation = Operation {
        name: "Fit track",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// One bench track moved between the cluster and the track stage.
    /// Cancellable for the reason [`Operation::BENCH_FIT`] is: an upgrade is a
    /// fit with a triangulation in front of it, and it stops where the fit
    /// stops, leaving the track at the stage it was in.
    pub(crate) const BENCH_SET_STAGE: Operation = Operation {
        name: "Set track stage",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// One bench track's patch turned to the normal the photographs agree on
    /// most (`sfmtool_core::bench::fit_normal`).
    ///
    /// Cancellable on either side of the decode, in front of the search and
    /// of the turn, and inside the reading it ends with. The search itself is
    /// one kernel call over a handful of views and runs to its end once
    /// entered.
    pub(crate) const BENCH_FIT_NORMAL: Operation = Operation {
        name: "Fit normal",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// One bench track's patch turned to the plane its pieces fit on
    /// (`sfmtool_core::bench::finite_difference_normal`).
    ///
    /// Cancellable for the reason [`Operation::BENCH_FIT`] is, since each
    /// piece is a fit: the step polls the flag in front of every piece and
    /// stops where the piece's fit stops, leaving the track as it was.
    pub(crate) const BENCH_FINITE_DIFFERENCE_NORMAL: Operation = Operation {
        name: "Finite difference normal",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// One bench track's patch turned to the plane through a grid of its
    /// fitted pieces (`sfmtool_core::bench::finite_difference_normal` with
    /// `PieceLayout::Grid`). Cancellable for the reason
    /// [`Operation::BENCH_FINITE_DIFFERENCE_NORMAL`] is.
    pub(crate) const BENCH_GRID_PLANE_NORMAL: Operation = Operation {
        name: "Grid plane normal",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// One bench track searched from, in the node's SIFT index
    /// (`specs/core/bench/editable-track.md` § "Searching the descriptor
    /// index").
    ///
    /// Cancellable in front of the forest query and between the candidates it
    /// found. The query itself is one call into the forest and runs to its end
    /// once entered, so a cancel lands either before it or after it -- which is
    /// still the difference between a step that stops and a button that does
    /// nothing.
    pub(crate) const BENCH_SEARCH: Operation = Operation {
        name: "Search descriptors",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// One track-stage bench patch projected and photometrically vetted in
    /// every image of its node (`specs/core/bench/editable-track.md` section
    /// "Searching by geometry").
    ///
    /// Cancellable around image decoding, reference construction, and between
    /// the views scored and candidates added. A stop returns no partially
    /// grown track.
    pub(crate) const BENCH_GEOMETRY_SEARCH: Operation = Operation {
        name: "Geometry search",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// A node's index files built and written beside its `.sfmr`: the SIFT
    /// index over its `.sift` files, then the cluster-patches file made from
    /// that index ([`crate::index_files`]).
    ///
    /// Cancellable in every phase. In the index: the read polls the flag
    /// between images, `KdForestU8::build` polls it as each leaf is placed, and
    /// `sfmtool_kdf_format::write_kdf` polls it between batches of blocks and
    /// hands back `KdfError::Cancelled`. In the cluster patches: the self-join
    /// polls it in front of every query, the decode between photographs, and
    /// the refinement between batches of clusters; the write is the last step
    /// and runs to its end. A file that was there is left standing, because
    /// both writes stream into a temporary sibling and rename over the target
    /// only once they have a whole file.
    pub(crate) const BUILD_INDEX_FILES: Operation = Operation {
        name: "Build index files",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Slow,
        locks: true,
        quiet: false,
    };

    /// A track built at one pixel of a posed photograph by the track-at-pixel
    /// cascade, then put on the node's bench and committed
    /// ([`crate::bench::track_at_pixel`]).
    ///
    /// Cancellable on either side of the decode and of the reads of the
    /// keypoints and the cluster patches, and inside the cascade, which polls
    /// the flag in front of each member and hands back
    /// `TrackAtPixelError::Cancelled`. A member once entered runs to its end,
    /// so a cancel lands between two of them. A cancelled run puts nothing on
    /// the bench.
    pub(crate) const CREATE_TRACK_AT_PIXEL: Operation = Operation {
        name: "Create track at pixel",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// The tracks near one pixel of a posed photograph found, put on the
    /// node's bench and the new ones committed, as one version
    /// ([`crate::bench::nearby_tracks`]).
    ///
    /// Cancellable on either side of the decode and of the reads of the
    /// `.sift` files and the cluster patches, and inside the search, which
    /// polls the flag between its sources, in the far-field sweep and in
    /// front of building the tracks, and hands back
    /// `NearbyTracksError::Cancelled`. A cancelled run puts nothing on the
    /// bench.
    ///
    /// Its kind is `Bench`, the kind of its refusals; a run that committed
    /// points writes its row as an `Edit`, since the version it pushed wrote
    /// the reconstruction.
    pub(crate) const FIND_NEARBY_TRACKS: Operation = Operation {
        name: "Find nearby tracks",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: true,
        quiet: false,
    };

    /// One or more `.sfmr` files read and made nodes, with the columns a file
    /// does not carry filled in for display (`specs/gui/background-tasks.md`
    /// section "Opening a file").
    ///
    /// The one operation that runs on no node: the node it makes does not exist
    /// until it finishes, so it locks nothing and every loaded node stays
    /// editable while it runs. Cancellable because the open polls the flag
    /// between files and between the stages of each, before every thumbnail it
    /// builds and every photograph it decodes, and before every patch bitmap it
    /// renders, and a cancelled open appends no node at all.
    pub(crate) const OPEN: Operation = Operation {
        name: "Open",
        cancellable: true,
        kind: Kind::File,
        queue: Queue::Fast,
        locks: false,
        quiet: false,
    };

    /// One track evaluated against its current inputs by the bench's live
    /// evaluation ([`crate::bench::live`]): a quiet task.
    ///
    /// It locks nothing, since it pushes no version, and it does not start on
    /// a locked node: a waiting one whose node has become locked is dropped
    /// when it would start, and an answer that comes back to a locked node is
    /// not installed. Cancellable because the evaluation polls the flag on
    /// either side of the decode, between its rounds and between the views it
    /// renders, and hands back `EvaluateError::Cancelled`.
    pub(crate) const LIVE_EVALUATION: Operation = Operation {
        name: "Live evaluation",
        cancellable: true,
        kind: Kind::Bench,
        queue: Queue::Fast,
        locks: false,
        quiet: true,
    };

    /// A camera model proposal's job ([`crate::camera_model_proposal::job`]):
    /// the switch of the proposal's camera and, while the preview is asked
    /// for, the retriangulation of its tracks. A quiet task.
    ///
    /// It locks nothing, since it pushes no version: Apply installs its answer
    /// on the GUI thread. An open proposal refuses every task start on its
    /// node, and a proposal cannot be opened on a busy node, so it and a
    /// locking task on its node do not meet. Cancellable because the job
    /// polls the flag before the switch and after it, and the retriangulation
    /// polls it as it runs. The switch itself takes no `Progress`, so a cancel
    /// that arrives during it is read when it returns.
    pub(crate) const CAMERA_MODEL_PROPOSAL: Operation = Operation {
        name: "Camera model proposal",
        cancellable: true,
        kind: Kind::Edit,
        queue: Queue::Fast,
        locks: false,
        quiet: true,
    };

    /// Every operation that can go to the background.
    ///
    /// A list rather than a set of constants used one at a time, so the tests
    /// that hold each `cancellable: true` to its claim, and each queue, lock
    /// and quiet value to the spec's table, have something to walk: a
    /// declaration nothing checks is a declaration that rots.
    // Read by those tests alone, which is what it is for.
    #[cfg(test)]
    pub(crate) const ALL: [Operation; 20] = [
        Operation::OPEN,
        Operation::BUNDLE_ADJUST,
        Operation::TO_EMBEDDED_PATCHES,
        Operation::RETRIANGULATE_ALL_POINTS,
        Operation::REFIT_ALL_POINTS,
        Operation::SWITCH_CAMERA_MODEL,
        Operation::PRUNE_COVERED_OBSERVATIONS,
        Operation::ADD_IMAGE_TO_TRACKS,
        Operation::BENCH_FIT,
        Operation::BENCH_SET_STAGE,
        Operation::BENCH_FIT_NORMAL,
        Operation::BENCH_FINITE_DIFFERENCE_NORMAL,
        Operation::BENCH_GRID_PLANE_NORMAL,
        Operation::BENCH_SEARCH,
        Operation::BENCH_GEOMETRY_SEARCH,
        Operation::BUILD_INDEX_FILES,
        Operation::CREATE_TRACK_AT_PIXEL,
        Operation::FIND_NEARBY_TRACKS,
        Operation::LIVE_EVALUATION,
        Operation::CAMERA_MODEL_PROPOSAL,
    ];
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
    /// gone and others may have taken its place: the id is what
    /// [`AppState::task`] finds it by, running or finished.
    pub(crate) id: u64,
    /// What it is, for the refusals, and what it claims about itself.
    pub(crate) operation: Operation,
    /// The node it will install its answer into, and the node it locks when
    /// [`Operation::locks`] says it does: `None` for an open, whose node does
    /// not exist until it finishes.
    pub(crate) node: Option<ReconId>,
    /// That node's label, for the sentences that name it, or for an open the
    /// label the file suggests. Kept here rather
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
    /// What the worker sends, drained by [`AppState::poll_background_tasks`].
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
    /// A camera model proposal's Apply: the next value as
    /// [`Finished::Produced`] carries it, and the switch's report for the
    /// camera, which the wire's reply carries as its `fit` object.
    SwitchedCamera(Box<SwitchedCamera>),
    /// One bench item's next value, and everything the GUI thread needs to
    /// install it.
    ///
    /// The document half is untouched, so there is no map and no selection to
    /// follow: what the GUI thread does with this is put the track back on the
    /// bench under its own label and push one version. A report that comes
    /// home to a bench the item has left is discarded, because there is
    /// nothing left for it to describe.
    BenchTrack {
        /// The item on the node's bench the report is about.
        label: String,
        /// The measured track, which replaces the value under that label.
        track: Box<sfmtool_core::bench::EditableTrack>,
        /// The version's label, as the Edit History panel lists it.
        version_label: String,
        /// The Action Log sentence, up to the serials.
        text: String,
    },
    /// A node's index files built, for the node the task ran on: the index
    /// it wrote and reopened, if it wrote one, and the cluster-patches file, if
    /// it got that far.
    ///
    /// Not a version: the files sit beside the node's `.sfmr`, and nothing
    /// about the reconstruction or the bench moved. So the GUI thread installs
    /// what was written and writes the row, and Undo has nothing to take back.
    /// A build that wrote its index and then stopped still hands the index
    /// back, so the node opens the file that is now on disk.
    IndexFiles {
        /// The `.kdf` that was written, and it opened.
        index: Option<(
            std::path::PathBuf,
            Arc<sfmtool_core::features::kdforest::LazyKdForestU8>,
        )>,
        /// The `.matches` that was written.
        cluster_patches: Option<std::path::PathBuf>,
        /// How the build ended.
        end: IndexFilesEnd,
    },
    /// A track-at-pixel run: the track the cascade built, or every member's
    /// refusal.
    ///
    /// Two versions when it built one, and neither is pushed on the worker:
    /// the GUI thread puts the track on the bench, writes that row, and then
    /// commits it through the bench's own commit, which writes its own
    /// ([`crate::bench::track_at_pixel`]). A refusal is the failed row alone.
    TrackAtPixel(Box<crate::bench::track_at_pixel::TrackAtPixelRun>),
    /// A *Find Nearby Tracks* run: the tracks core found near the pixel.
    ///
    /// One version when anything usable was found, pushed on the GUI thread:
    /// every track put on the bench and the new ones committed together
    /// ([`crate::bench::nearby_tracks`]). A find with nothing usable is its
    /// row alone.
    NearbyTracks(Box<crate::bench::nearby_tracks::NearbyTracksRun>),
    /// The files an open read, each with what it filled in for display, and
    /// the ones it could not read.
    ///
    /// Not a version of anything: each file becomes a node of its own, whose
    /// first version is the file as read. The GUI thread appends them in the
    /// order they were asked for and writes one row.
    Opened(Vec<crate::state::open::OpenedFile>),
    /// The kernel ran to its end and found nothing to change.
    ///
    /// Not a refusal and not a version: an edit whose answer is "there was
    /// nothing here to do" has run successfully, and pushing a version that
    /// changes nothing would put a row in the history a reader cannot tell from
    /// one that did something. So the sentence is recorded and the document is
    /// left alone.
    NoChange(String),
    /// The operation was asked to stop, and did.
    Cancelled,
    /// The kernel refused, in its own words.
    Failed(String),
    /// A quiet task's job returned. Its answers went to its source on the
    /// source's own channel, so this carries nothing; it exists so the queue
    /// knows the task has finished.
    Quiet,
}

/// What a camera model proposal's Apply produced: the value, installed as
/// [`Finished::Produced`] installs one, and the switch's report.
pub(crate) struct SwitchedCamera {
    /// What the switch and the retriangulation returned.
    pub(crate) value: Arc<SfmrReconstruction>,
    /// The map from the input's rows to that value's.
    pub(crate) map: PointMap,
    /// The version's label.
    pub(crate) version_label: String,
    /// The Action Log sentence, up to the serials.
    pub(crate) text: String,
    /// The switch's report for the camera.
    pub(crate) switch: sfmtool_core::reconstruction::switch_camera_model::CameraSwitch,
}

/// How a index-files build ended, beside whatever it wrote.
///
/// The three endings of [`Finished`] itself, for a build that may have written
/// one of its two files before it stopped.
pub(crate) enum IndexFilesEnd {
    /// Both files were written; the Action Log sentence.
    Built(String),
    /// Asked to stop, and did.
    Cancelled,
    /// A stage refused, in its own words.
    Failed(String),
}

/// What became of one task that has finished.
///
/// Kept after the task is gone, in a ring of the last [`FINISHED_KEPT`],
/// because the question outlives it: a tool call that was handed a handle while
/// the operation was running comes back for the answer, and by then there is no
/// task to ask. The Background Tasks panel asks the same question of an idle
/// queue, which is why what it cost and what it spent the time on are here
/// rather than only in the Action Log: the panel would otherwise have to find
/// its own entry in a buffer that drops and clears.
pub(crate) struct FinishedTask {
    /// Which task this is about.
    pub(crate) id: u64,
    /// What it was, by the name the panel and the refusals call it.
    pub(crate) operation: Operation,
    /// The label of the node it ran on, as it read when the operation started.
    pub(crate) label: String,
    /// The first node an open made, which is what `open_reconstruction`
    /// answers with once it lands. `None` for every other operation.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) opened: Option<ReconId>,
    /// What a *Create Track Here* left behind: the point it committed, the
    /// commit's refusal, or every member's refusal. What `create_track_at_pixel`
    /// answers with once it lands. `None` for every other operation.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) created_track: Option<crate::bench::track_at_pixel::CreatedTrack>,
    /// What a *Find Nearby Tracks* found and what became of each track. What
    /// `find_nearby_tracks` answers with once it lands. `None` for every other
    /// operation, and for a run that was refused or cancelled.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) found_nearby: Option<Box<crate::bench::nearby_tracks::FoundNearby>>,
    /// What a camera model proposal's Apply switched: the switch's report for
    /// the camera, which `switch_camera_model { proposal: true }` answers with
    /// once it lands. `None` for every other operation, and for a run that was
    /// refused or cancelled.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) switched_camera:
        Option<Box<sfmtool_core::reconstruction::switch_camera_model::CameraSwitch>>,
    /// Whether it pushed a version of the node it ran on.
    ///
    /// False for a run that ended in [`Finished::NoChange`] as well as for one
    /// that was refused or cancelled, so a reply can say `changed: false` for a
    /// run that succeeded without changing anything.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) changed: bool,
    /// What the whole operation cost, measured from the instant it started,
    /// not from when it was queued. Zero for a task cancelled before it
    /// started.
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
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) outcome: Result<String, String>,
}

/// One running task, for a reader that cannot reach [`AppState`].
///
/// The MCP server answers a timed-out call on its own thread, where the GUI
/// thread is by definition not answering, so the one fact that message needs is
/// the one fact it cannot ask for. A list written twice per task and read once
/// per timeout, which is why a `Mutex` is the whole of the machinery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Busy {
    /// What the operation is called, as the panel and the refusals call it.
    pub(crate) operation: &'static str,
    /// The node it is running on.
    pub(crate) label: String,
}

/// Where [`Busy`] is kept: shared with whoever has to read it off the GUI
/// thread, one entry per running task, slow queue first, and empty whenever
/// nothing is running.
pub(crate) type BusyNotice = Arc<Mutex<Vec<Busy>>>;

/// What is running, empty when nothing is. Poisoned or not, because a stale
/// sentence in an error message is a smaller failure than a panic inside one.
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
pub(crate) fn busy(notice: &BusyNotice) -> Vec<Busy> {
    notice
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Say what is running.
fn set_busy(notice: &BusyNotice, running: Vec<Busy>) {
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
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Polled {
    /// Whether anything arrived, so the frame knows to redraw.
    pub(crate) changed: bool,
    /// The nodes finished operations gave a new version to, whose panel caches
    /// are now about a value they no longer hold. Two at most, since each
    /// queue lands at most one task per poll.
    pub(crate) installed: Vec<ReconId>,
}

impl AppState {
    /// One queue's state.
    pub(crate) fn queue(&self, queue: Queue) -> &QueueState {
        match queue {
            Queue::Slow => &self.slow_queue,
            Queue::Fast => &self.fast_queue,
        }
    }

    fn queue_mut(&mut self, queue: Queue) -> &mut QueueState {
        match queue {
            Queue::Slow => &mut self.slow_queue,
            Queue::Fast => &mut self.fast_queue,
        }
    }

    /// The task someone asked for that `queue` is running, or `None` while it
    /// is idle or runs a quiet task.
    pub(crate) fn running_in(&self, queue: Queue) -> Option<&BackgroundTask> {
        self.queue(queue)
            .running
            .as_ref()
            .filter(|task| !task.operation.quiet)
    }

    /// The quiet task `queue` is running, if it runs one.
    pub(crate) fn quiet_running_in(&self, queue: Queue) -> Option<&BackgroundTask> {
        self.queue(queue)
            .running
            .as_ref()
            .filter(|task| task.operation.quiet)
    }

    /// The tasks someone asked for that wait in `queue`, in the order they
    /// will start. Quiet tasks waiting behind them are left out.
    pub(crate) fn waiting_in(&self, queue: Queue) -> impl Iterator<Item = &WaitingTask> {
        self.queue(queue)
            .waiting
            .iter()
            .filter(|task| !task.operation.quiet)
    }

    /// Every task someone asked for running off the GUI thread, the slow
    /// queue's first. Quiet tasks are left out: this is "someone's work is
    /// running", which is what every reader of it asks.
    pub(crate) fn running_tasks(&self) -> impl Iterator<Item = &BackgroundTask> {
        Queue::ALL
            .into_iter()
            .filter_map(|queue| self.running_in(queue))
    }

    /// The running task with this id, or `None` when no running task has it.
    pub(crate) fn running_task(&self, id: u64) -> Option<&BackgroundTask> {
        self.running_tasks().find(|task| task.id == id)
    }

    /// Every task someone asked for waiting to start, the slow queue's first
    /// and each queue's in the order they will start. Quiet tasks are left
    /// out.
    pub(crate) fn waiting_tasks(&self) -> impl Iterator<Item = &WaitingTask> {
        Queue::ALL
            .into_iter()
            .flat_map(|queue| self.waiting_in(queue))
    }

    /// Whether the quiet task `id` is running, as against waiting or over.
    pub(crate) fn quiet_task_running(&self, id: u64) -> bool {
        Queue::ALL
            .into_iter()
            .filter_map(|queue| self.quiet_running_in(queue))
            .any(|task| task.id == id)
    }

    /// The waiting task with this id, or `None` when no waiting task has it.
    pub(crate) fn waiting_task(&self, id: u64) -> Option<&WaitingTask> {
        self.waiting_tasks().find(|task| task.id == id)
    }

    /// The nodes a task holds locked, running or waiting, each with the
    /// sentence that refuses an edit of it.
    ///
    /// A node appears at most once, since a second locking task on a locked
    /// node is refused.
    pub(crate) fn locked_nodes(&self) -> Vec<(ReconId, String)> {
        self.running_tasks()
            .filter(|task| task.operation.locks)
            .filter_map(|task| task.node)
            .chain(
                self.waiting_tasks()
                    .filter(|task| task.operation.locks)
                    .filter_map(|task| task.node),
            )
            .filter_map(|node| self.busy_refusal(node).map(|why| (node, why)))
            .collect()
    }

    /// The record of the finished task with this id, or `None` when it is
    /// still running, never ran, or has been pushed out of the ring.
    pub(crate) fn finished_task(&self, id: u64) -> Option<&FinishedTask> {
        self.finished_tasks.iter().rev().find(|task| task.id == id)
    }

    /// The task with this id, waiting, running or finished, or `None` when no
    /// task has it or its record is no longer kept. Quiet tasks have ids of
    /// their own ([`AppState::start_quiet_task`]) and are never found here.
    ///
    /// What a caller that was handed an id asks, rather than whether "the"
    /// running task is its own: with two queues, the task that is running may
    /// be another caller's, and the task that finished last may be too.
    pub(crate) fn task(&self, id: u64) -> Option<TaskState<'_>> {
        if let Some(task) = self.running_task(id) {
            return Some(TaskState::Running(task));
        }
        if let Some(task) = self.waiting_task(id) {
            return Some(TaskState::Waiting(task));
        }
        self.finished_task(id).map(TaskState::Finished)
    }

    /// The task that finished most recently, from either queue.
    pub(crate) fn last_finished_task(&self) -> Option<&FinishedTask> {
        self.finished_tasks.back()
    }

    /// The task that finished most recently from `queue`.
    pub(crate) fn last_finished_in(&self, queue: Queue) -> Option<&FinishedTask> {
        self.finished_tasks
            .iter()
            .rev()
            .find(|task| task.operation.queue == queue)
    }

    /// Why an edit of `id` is refused right now, or `None`.
    ///
    /// A node is busy while a task that locks it runs or waits to run, in
    /// either queue. Every method that gives a node a new version, moves its
    /// cursor, writes it out or closes it calls this first, so one sentence
    /// covers the method's refusal, the menu's greying and the wire's. Other
    /// nodes are untouched and stay fully editable: the operation is a function
    /// of the value it was handed when it was queued, and an edit underneath it
    /// would produce a version whose parent is not the version it was computed
    /// from.
    pub(crate) fn busy_refusal(&self, id: ReconId) -> Option<String> {
        let locks =
            |operation: Operation, node: Option<ReconId>| operation.locks && node == Some(id);
        if let Some(task) = self
            .running_tasks()
            .find(|task| locks(task.operation, task.node))
        {
            return Some(format!(
                "{} is busy: {} is still running.",
                task.label, task.operation.name
            ));
        }
        let task = self
            .waiting_tasks()
            .find(|task| locks(task.operation, task.node))?;
        Some(format!(
            "{} is busy: {} is waiting to run.",
            task.label, task.operation.name
        ))
    }

    /// Why cancelling the task `id` would do nothing, or `None` when it would
    /// stop it.
    ///
    /// The sentence a Cancel button's tooltip carries and the wire's refusal
    /// are the same sentence, for the same reason every other refusal here has
    /// one spelling. A waiting task is never refused: whatever its operation,
    /// taking it out of its queue stops it.
    pub(crate) fn cancel_refusal(&self, id: u64) -> Option<String> {
        if self.waiting_task(id).is_some() {
            return None;
        }
        match self.running_task(id) {
            Some(task) if !task.operation.cancellable => Some(format!(
                "{} cannot be cancelled: it never asks whether it should stop.",
                task.operation.name
            )),
            Some(_) => None,
            None => Some(match self.finished_task(id) {
                Some(task) => format!(
                    "{} of {} has already finished.",
                    task.operation.name, task.label
                ),
                None => format!("No background task with operation_id {id} is running."),
            }),
        }
    }

    /// Start `operation` on `id`, running `job` on a worker thread, or queue it
    /// behind its queue's running task, and answer with the task's id.
    ///
    /// Refused when `operation` locks its node and the node is busy
    /// ([`Self::start_refusal`]), or while a camera model proposal is open on
    /// it. A start while the operation's queue runs another task is not
    /// refused: the task waits, its node locked from now, and starts when the
    /// tasks ahead of it have landed. `job` was built by the caller from the
    /// value at the cursor, after the caller's own busy checks, which is why a
    /// queued task's own lock never refuses it.
    ///
    /// **Nothing is logged here.** The log records outcomes, not intentions
    /// (`specs/gui/action-log.md`), so the entry is written when the operation
    /// ends, by [`AppState::poll_background_tasks`].
    pub(crate) fn start_background_task(
        &mut self,
        operation: Operation,
        id: ReconId,
        job: Job,
    ) -> Result<u64, String> {
        if let Some(why) = self.start_refusal(operation, Some(id)) {
            return Err(why);
        }
        // The one task that can start on a node with a proposal open is the
        // proposal's own Apply, which ends the proposal before it starts.
        if let Some(why) = self.proposal_refusal(id) {
            return Err(why);
        }
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        self.start_task(operation, Some(id), label, job)
    }

    /// Why starting `operation` on `node` is refused right now, or `None`.
    ///
    /// A task that locks its node is refused while a locking task of either
    /// queue runs on that node or waits to, with the busy sentence, so a node
    /// has at most one locking task at any time. A busy queue refuses nothing:
    /// the task waits in it.
    pub(crate) fn start_refusal(
        &self,
        operation: Operation,
        node: Option<ReconId>,
    ) -> Option<String> {
        if !operation.locks {
            return None;
        }
        node.and_then(|node| self.busy_refusal(node))
    }

    /// Start `operation`, named `label`, running `job` on a worker thread, or
    /// queue it behind its queue's running task, and answer with the task's
    /// id either way.
    ///
    /// `node` is the node it installs into, and locks from now when the
    /// operation says it does, or `None` for an open, whose node does not
    /// exist yet. The body [`Self::start_background_task`] and
    /// [`Self::start_open`](crate::state::AppState::start_open) share.
    pub(crate) fn start_task(
        &mut self,
        operation: Operation,
        node: Option<ReconId>,
        label: String,
        job: Job,
    ) -> Result<u64, String> {
        debug_assert!(
            !operation.quiet,
            "a quiet task starts through start_quiet_task"
        );
        if let Some(why) = self.start_refusal(operation, node) {
            return Err(why);
        }
        self.give_way_to(operation, node);
        let task = WaitingTask {
            id: self.next_operation_id,
            operation,
            node,
            label,
            queued: Instant::now(),
            actor: self.action_log.actor(),
            job: WaitingJob::Asked(job),
        };
        let id = task.id;
        if self.queue(operation.queue).running.is_some() {
            self.next_operation_id += 1;
            // Behind the tasks someone asked for and ahead of every quiet one.
            let waiting = &mut self.queue_mut(operation.queue).waiting;
            let at = waiting
                .iter()
                .position(|task| task.operation.quiet)
                .unwrap_or(waiting.len());
            waiting.insert(at, task);
            return Ok(id);
        }
        // A thread that could not be spawned is the caller's refusal, so the
        // id is not spent on a task that never existed.
        self.run_task(task)?;
        self.next_operation_id += 1;
        Ok(id)
    }

    /// Ask the quiet tasks a task someone asked for displaces to stop: the
    /// quiet task running in its queue, which it would otherwise wait behind,
    /// and, for a task that locks `node`, a quiet task running on that node in
    /// either queue, whose answer would land into the bench the new task's
    /// job has just captured.
    ///
    /// The quiet task is not taken off its queue: it runs until it reads the
    /// flag, so a queue still runs one task at a time, and the task that
    /// displaced it starts once it reports that it has stopped. Its source
    /// queues it again, behind the task, when its cancelled answer comes back.
    fn give_way_to(&self, operation: Operation, node: Option<ReconId>) {
        for queue in Queue::ALL {
            let Some(task) = self.quiet_running_in(queue) else {
                continue;
            };
            let on_its_node = operation.locks && node.is_some() && task.node == node;
            if queue == operation.queue || on_its_node {
                task.cancel.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Queue a quiet task for `source`, named `label`, running `job` on a
    /// worker thread, and answer with its id.
    ///
    /// A quiet task waits behind every task someone asked for and behind the
    /// quiet tasks ahead of it in `rank`, and the source's own waiting entry,
    /// if it has one, is replaced, its job told it was dropped. It starts at
    /// once only when its queue is idle, and then only when `node` is not
    /// locked: a quiet task does not start on a locked node, and is dropped,
    /// its source told so, rather than run. The id is from a count of its own
    /// ([`AppState::next_quiet_id`]), so the ids the wire hands out stay
    /// consecutive. An error is a worker thread that could not be spawned.
    ///
    /// Nothing is logged, here or when it finishes: a quiet task writes no row
    /// for running and leaves no record in the ring of finished tasks.
    pub(crate) fn start_quiet_task(
        &mut self,
        operation: Operation,
        source: QuietSource,
        rank: QuietRank,
        node: ReconId,
        label: String,
        job: QuietJob,
    ) -> Result<u64, String> {
        debug_assert!(operation.quiet && !operation.locks);
        let queue = operation.queue;
        if let Some(at) = self
            .queue(queue)
            .waiting
            .iter()
            .position(|task| task.quiet_source() == Some(source))
        {
            let replaced = self
                .queue_mut(queue)
                .waiting
                .remove(at)
                .expect("just found");
            replaced.drop_unrun();
        }
        let id = self.next_quiet_id;
        self.next_quiet_id += 1;
        let task = WaitingTask {
            id,
            operation,
            node: Some(node),
            label,
            queued: Instant::now(),
            actor: self.action_log.actor(),
            job: WaitingJob::Quiet { source, rank, job },
        };
        if self.queue(queue).running.is_some() {
            let waiting = &mut self.queue_mut(queue).waiting;
            let at = waiting
                .iter()
                .position(|task| match &task.job {
                    WaitingJob::Quiet { rank: ahead, .. } => *ahead > rank,
                    WaitingJob::Asked(_) => false,
                })
                .unwrap_or(waiting.len());
            waiting.insert(at, task);
            return Ok(id);
        }
        self.begin(task)?;
        Ok(id)
    }

    /// Ask the quiet task `id` to stop when it is running, or take it out of
    /// its queue, telling its source it was dropped, when it is waiting. Does
    /// nothing when no quiet task has that id.
    pub(crate) fn cancel_quiet_task(&mut self, id: u64) {
        for queue in Queue::ALL {
            if let Some(task) = self.quiet_running_in(queue).filter(|task| task.id == id) {
                task.cancel.store(true, Ordering::Relaxed);
                return;
            }
            let waiting = &mut self.queue_mut(queue).waiting;
            if let Some(at) = waiting
                .iter()
                .position(|task| task.operation.quiet && task.id == id)
            {
                let task = waiting.remove(at).expect("just found");
                task.drop_unrun();
                return;
            }
        }
    }

    /// Start `task` from its idle queue, or, for a quiet task whose node has
    /// become locked since it was queued, drop it and tell its source.
    fn begin(&mut self, task: WaitingTask) -> Result<(), String> {
        if task.operation.quiet
            && task
                .node
                .is_some_and(|node| self.busy_refusal(node).is_some())
        {
            task.drop_unrun();
            return Ok(());
        }
        self.run_task(task)
    }

    /// Spawn the worker for `task` and make it its queue's running task.
    ///
    /// The queue must be idle. `started`, which the Action Log row's cost is
    /// measured from, is now, however long the task waited.
    fn run_task(&mut self, task: WaitingTask) -> Result<(), String> {
        let WaitingTask {
            id,
            operation,
            node,
            label,
            actor,
            job,
            ..
        } = task;
        debug_assert!(self.queue(operation.queue).running.is_none());
        let (reports, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let queued = Arc::new(AtomicBool::new(false));
        let (job, collector): (Job, Collector) = match job {
            // The level the Action Log toolbar's checkbox last left, read as
            // the operation starts so that a change to it takes effect on the
            // next one.
            WaitingJob::Asked(job) => (
                job,
                Collector::reporting(
                    self.action_log.detailed_timing(),
                    progressed(reports.clone(), Arc::clone(&queued), self.wake.clone()),
                ),
            ),
            // Nothing draws a quiet task's phases, so its collector wakes
            // nobody: a burst of evaluations during a slider drag costs no
            // frame per phase it opens.
            WaitingJob::Quiet { job, .. } => {
                let QuietJob { run, dropped } = job;
                drop(dropped);
                (
                    Box::new(move |progress: &Progress<'_>| {
                        run(progress);
                        Finished::Quiet
                    }),
                    Collector::new(false),
                )
            }
        };
        let collector = Arc::new(collector);

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

        self.queue_mut(operation.queue).running = Some(BackgroundTask {
            id,
            operation,
            node,
            label,
            started: Instant::now(),
            actor,
            collector,
            cancel,
            reports: rx,
            queued,
        });
        // Said here and in `apply_reports`, which are the two instants the
        // answer changes at. Anything reading it is off the GUI thread and
        // cannot ask the state.
        self.publish_busy();
        Ok(())
    }

    /// Start `queue`'s next waiting task, if it is idle and one waits.
    ///
    /// Called in the poll that lands the queue's running task, so the next one
    /// starts in that same poll. A task whose thread cannot be spawned ends
    /// there, with a failed row and a record in the ring, and the one behind
    /// it is tried. A quiet task is checked again here: one whose node has
    /// become locked since it was queued is dropped, its source told so, and
    /// the one behind it is tried. A quiet task whose thread cannot be spawned
    /// ends with no row, and its source reads its channel's end as a failure.
    fn start_next(&mut self, queue: Queue) {
        while self.queue(queue).running.is_none() {
            let Some(task) = self.queue_mut(queue).waiting.pop_front() else {
                return;
            };
            let (id, operation, label, actor) =
                (task.id, task.operation, task.label.clone(), task.actor);
            if let Err(message) = self.begin(task) {
                if !operation.quiet {
                    self.end_before_start(id, operation, label, actor, message);
                }
            }
        }
    }

    /// Take the waiting task `id` out of its queue, writing its *cancelled
    /// before it started* row and keeping its record. False when no task with
    /// that id is waiting.
    fn cancel_waiting(&mut self, id: u64) -> bool {
        for queue in Queue::ALL {
            let waiting = &mut self.queue_mut(queue).waiting;
            let Some(index) = waiting
                .iter()
                .position(|task| !task.operation.quiet && task.id == id)
            else {
                continue;
            };
            let task = waiting.remove(index).expect("just found");
            let message = format!(
                "{} of {} cancelled before it started",
                task.operation.name, task.label
            );
            // The job is dropped here, unrun, with whatever value it captured.
            let WaitingTask {
                operation,
                label,
                actor,
                ..
            } = task;
            self.end_before_start(id, operation, label, actor, message);
            return true;
        }
        false
    }

    /// Record a task that ended without starting: a failed row as whoever
    /// asked for it, costing nothing, and its record in the ring.
    fn end_before_start(
        &mut self,
        id: u64,
        operation: Operation,
        label: String,
        actor: Actor,
        message: String,
    ) {
        self.action_log.fail_done_as(
            actor,
            operation.kind,
            Instant::now(),
            message.clone(),
            Vec::new(),
        );
        self.keep_finished(FinishedTask {
            id,
            operation,
            label,
            opened: None,
            created_track: None,
            found_nearby: None,
            switched_camera: None,
            changed: false,
            took: Duration::ZERO,
            detail: Vec::new(),
            outcome: Err(message),
        });
    }

    /// Put `task` in the ring of finished tasks, letting the oldest go when it
    /// is full.
    fn keep_finished(&mut self, task: FinishedTask) {
        if self.finished_tasks.len() == FINISHED_KEPT {
            self.finished_tasks.pop_front();
        }
        self.finished_tasks.push_back(task);
    }

    /// Write every running task into the notice the MCP server reads.
    fn publish_busy(&self) {
        set_busy(
            &self.busy_notice,
            self.running_tasks()
                .map(|task| Busy {
                    operation: task.operation.name,
                    label: task.label.clone(),
                })
                .collect(),
        );
    }

    /// Apply every report the workers of both queues have sent.
    ///
    /// Runs in phase 0 of the frame, before the MCP drain: a completed
    /// operation's version is then on screen in the frame it landed, and an
    /// agent's call in that same frame reads the new value rather than the old
    /// one. Both queues are polled here, so two tasks can land in one frame.
    pub(crate) fn poll_background_tasks(&mut self) -> Polled {
        let mut polled = Polled::default();
        for queue in Queue::ALL {
            let step = self.apply_reports(queue, None);
            polled.changed |= step.changed;
            polled.installed.extend(step.installed);
        }
        polled
    }

    /// The drain both readers share, for one queue: `first` is a report
    /// already taken off that queue's channel, and the rest are whatever is
    /// waiting behind it.
    ///
    /// The parameter is there because a blocking read consumes the message it
    /// waited for, and a message consumed outside this loop would be a message
    /// nothing acted on.
    fn apply_reports(&mut self, queue: Queue, first: Option<Report>) -> Polled {
        let mut polled = Polled::default();
        let Some(task) = self.queue(queue).running.as_ref() else {
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
            let task = self.queue_mut(queue).running.take().expect("just borrowed");
            self.publish_busy();
            // A quiet task's answers are on its source's channel already, sent
            // before its worker reported this, so there is nothing to install,
            // no row to write and no record to keep.
            if !task.operation.quiet {
                polled.installed.extend(self.finish(task, finished));
            }
            // After the landed task's row, so the log reads in the order the
            // two happened. The task that starts locks its own node, which is
            // never the one that just landed.
            self.start_next(queue);
        }
        polled
    }

    /// Ask the task `id` to stop, or, for a task still waiting, take it out of
    /// its queue at once. Silently does nothing when it cannot, which is what
    /// [`AppState::cancel_refusal`] is for saying out loud.
    ///
    /// A waiting task that is cancelled writes *"{operation} of {label}
    /// cancelled before it started"*, as whoever asked for it, and its record
    /// goes in the ring, so a call holding its id is answered.
    pub(crate) fn cancel_background_task(&mut self, id: u64) {
        if let Some(task) = self.running_task(id) {
            if task.operation.cancellable {
                task.cancel.store(true, Ordering::Relaxed);
            }
            return;
        }
        self.cancel_waiting(id);
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
            node: locked,
            label,
            started,
            actor,
            collector,
            ..
        } = task;
        let mut opened = None;
        let mut installed = None;
        let mut created_track = None;
        // A track-at-pixel run that landed a track, to commit once its own row
        // is written: the node, the item and the member that built it.
        let mut to_commit: Option<(ReconId, String, &'static str)> = None;
        // What a *Find Nearby Tracks* found, the kind its row is written as,
        // and the point it selects once the row is written.
        let mut found_nearby = None;
        let mut kind = operation.kind;
        let mut to_select = None;
        // A proposal's Apply installs as any produced value does; its switch
        // report is kept for the reply.
        let mut switched_camera = None;
        let finished = match finished {
            Finished::SwitchedCamera(run) => {
                let SwitchedCamera {
                    value,
                    map,
                    version_label,
                    text,
                    switch,
                } = *run;
                switched_camera = Some(Box::new(switch));
                Finished::Produced {
                    // The task's answer is shared with nothing, so this is a move.
                    value: Arc::unwrap_or_clone(value),
                    map,
                    version_label,
                    text,
                }
            }
            other => other,
        };
        let outcome = match finished {
            Finished::Produced {
                value,
                map,
                version_label,
                text,
            } => match self.scene.iter().position(|n| Some(n.id) == locked) {
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
                    let node = self.scene[index].id;
                    self.follow_selection_forward(node);
                    installed = Some(node);
                    Ok(version_step_text(&text, parent, serial))
                }
            },
            // A report lands on the item it measured, wherever the cursor has
            // gone in the meantime: measurements are keyed by observation index
            // and no bench step renumbers observations (deleting an image, which
            // does, is refused while the task holds the node), so a verdict set
            // while the task ran is shown over the number the run produced. What the
            // report cannot survive is the item leaving the bench at the
            // cursor, and then it is dropped with the row that says so.
            Finished::BenchTrack {
                label: item,
                track,
                version_label,
                text,
            } => match self.scene.iter().position(|n| Some(n.id) == locked) {
                None => Err(format!(
                    "{} of {label} finished, but it is no longer loaded.",
                    operation.name
                )),
                Some(index) => {
                    let bench = self.scene[index].history.current_bench();
                    match bench.replace(&item, BenchItem::Track(Arc::new(*track))) {
                        Err(_) => Err(format!(
                            "{} of {item} finished, and {item} is no longer on {label}'s bench.",
                            operation.name
                        )),
                        Ok(next) => {
                            let serial = {
                                let _phase = collector.phase("push version");
                                self.scene[index]
                                    .history
                                    .push_bench(Arc::new(next), version_label)
                            };
                            let parent =
                                crate::state::edits::version_before(&self.scene[index], serial);
                            installed = Some(self.scene[index].id);
                            Ok(version_step_text(&text, parent, serial))
                        }
                    }
                }
            },
            // The files are the node's and not the version's, so nothing is
            // pushed: what was written is opened and the row says what was
            // built. A node that has left the scene in the meantime leaves the
            // files on disk, which the next session opens.
            Finished::IndexFiles {
                index,
                cluster_patches,
                end,
            } => match locked.filter(|&node| self.scene.iter().any(|n| n.id == node)) {
                None => Err(format!(
                    "{} of {label} finished, but it is no longer loaded.",
                    operation.name
                )),
                Some(node) => {
                    let wrote_index = index.as_ref().map(|(path, _)| path.display().to_string());
                    self.install_index_files(node, index, cluster_patches);
                    match end {
                        IndexFilesEnd::Built(text) => Ok(text),
                        IndexFilesEnd::Failed(message) => Err(message),
                        IndexFilesEnd::Cancelled => Err(match wrote_index {
                            None => format!("{} of {label} cancelled", operation.name),
                            Some(path) => format!(
                                "{} of {label} cancelled after it wrote the SIFT index {path}",
                                operation.name
                            ),
                        }),
                    }
                }
            },
            // The track goes on the bench here, as one bench version; the
            // commit that follows is written after this row, below.
            Finished::TrackAtPixel(run) => {
                match self.scene.iter().position(|n| Some(n.id) == locked) {
                    None => Err(format!(
                        "{} of {label} finished, but it is no longer loaded.",
                        operation.name
                    )),
                    Some(index) => {
                        let member = match &run.result {
                            Ok((_, report)) => report.member.name(),
                            Err(_) => "",
                        };
                        let (outcome, created, item) = {
                            let _phase = collector.phase("push version");
                            self.land_track_at_pixel(index, *run)
                        };
                        let node = self.scene[index].id;
                        if let Some(item) = item {
                            installed = Some(node);
                            to_commit = Some((node, item, member));
                        }
                        created_track = Some(created);
                        outcome
                    }
                }
            }
            // Every track on the bench and the new ones committed, as one
            // version; the selection follows after the row, below.
            Finished::NearbyTracks(run) => {
                match self.scene.iter().position(|n| Some(n.id) == locked) {
                    None => Err(format!(
                        "{} of {label} finished, but it is no longer loaded.",
                        operation.name
                    )),
                    Some(index) => {
                        let landed = {
                            let _phase = collector.phase("push version");
                            self.land_nearby_tracks(index, *run)
                        };
                        if landed.found.changed {
                            installed = Some(self.scene[index].id);
                        }
                        kind = landed.kind;
                        to_select = landed.select;
                        found_nearby = Some(Box::new(landed.found));
                        landed.outcome
                    }
                }
            }
            // Each file becomes a node in the order it was asked for, and the
            // row names every one; a file the worker could not read is a failed
            // row of its own under the same actor. Nothing is installed into
            // an existing node, so no panel cache describes a value that has
            // changed.
            Finished::Opened(files) => {
                let _phase = collector.phase("append nodes");
                let (made, refused) = self.append_opened(files, actor);
                opened = made.first().copied();
                match (made.is_empty(), refused) {
                    (true, Some(message)) => Err(message),
                    _ => Ok(crate::state::open::opened_sentence(self, &made)),
                }
            }
            Finished::SwitchedCamera(_) => unreachable!("unwrapped into Produced above"),
            Finished::Quiet => unreachable!("a quiet task does not land through finish"),
            // Nothing to install and nothing to push, so the row is the whole
            // of it: the node's caches still describe the value it holds.
            Finished::NoChange(text) => Ok(text),
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
                .record_done_as(actor, kind, started, text, detail),
            Err(message) => {
                self.action_log
                    .fail_done_as(actor, kind, started, message.clone(), detail)
            }
        }
        // The commit a landed track ends in, as its own version and its own
        // row, after the row that put the track on the bench because that is
        // the order the two happened in, and as whoever asked for the run.
        if let Some((node, item, member)) = to_commit {
            created_track = Some(self.commit_created_track(actor, node, item, member));
        }
        // The focused nearby track's point, selected after the row that put it
        // on the bench, as a commit selects the point it wrote, and as whoever
        // asked for the run.
        if let Some(point) = to_select {
            let standing = self.action_log.actor();
            self.action_log.set_actor(actor);
            self.select_point(point);
            self.action_log.set_actor(standing);
        }
        self.keep_finished(FinishedTask {
            id,
            operation,
            label,
            opened,
            created_track,
            found_nearby,
            switched_camera: switched_camera.filter(|_| outcome.is_ok()),
            changed: installed.is_some(),
            took,
            detail: kept,
            outcome,
        });
        installed
    }

    /// Drive the task `id` to its end, applying what it reports.
    ///
    /// The test seam, and the only blocking read of a channel: a frame never
    /// waits, so the production drain is [`AppState::poll_background_tasks`]'s
    /// `try_recv`. Waiting on the worker's own messages is what makes a test
    /// deterministic without a sleep or a poll loop. Only `id`'s queue is
    /// drained; a task in the other queue lands when it is driven or polled.
    /// A task that is waiting is driven through the tasks ahead of it, which
    /// land first.
    #[cfg(test)]
    pub(crate) fn finish_background_task(&mut self, id: u64) -> Polled {
        use std::time::Duration;

        /// Long enough that a loaded machine does not fail a passing test,
        /// short enough that a wedged worker fails rather than hangs.
        const PATIENCE: Duration = Duration::from_secs(120);

        let mut polled = Polled::default();
        loop {
            let queue = match self.task(id) {
                Some(TaskState::Running(task)) => task.operation.queue,
                Some(TaskState::Waiting(task)) => task.operation.queue,
                Some(TaskState::Finished(_)) | None => break,
            };
            let waited = {
                let task = self
                    .queue(queue)
                    .running
                    .as_ref()
                    .expect("a queue with a task waiting runs one");
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
            let step = self.apply_reports(queue, first);
            polled.changed |= step.changed;
            polled.installed.extend(step.installed);
        }
        polled
    }

    /// Wait for the next report of `queue`'s running task, quiet or not, and
    /// apply it with whatever follows it, as a frame would. False when the
    /// queue is idle.
    ///
    /// The seam for a test that waits on a quiet task, which has no id the
    /// asked-for seams can name.
    #[cfg(test)]
    pub(crate) fn step_queue(&mut self, queue: Queue) -> bool {
        const PATIENCE: Duration = Duration::from_secs(120);
        let Some(task) = self.queue(queue).running.as_ref() else {
            return false;
        };
        let first = match task.reports.recv_timeout(PATIENCE) {
            Ok(report) => Some(report),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => None,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                panic!("the background operation said nothing for {PATIENCE:?}")
            }
        };
        self.apply_reports(queue, first);
        true
    }

    /// Drive every running quiet task to its end, so that a queue a test reads
    /// next is not held by a quiet task its source has already heard from.
    #[cfg(test)]
    pub(crate) fn land_quiet_tasks(&mut self) {
        while let Some(queue) = Queue::ALL
            .into_iter()
            .find(|&queue| self.quiet_running_in(queue).is_some())
        {
            self.step_queue(queue);
        }
    }

    /// Whether the running quiet task `id` has been asked to stop.
    #[cfg(test)]
    pub(crate) fn quiet_task_cancelled(&self, id: u64) -> bool {
        Queue::ALL
            .into_iter()
            .filter_map(|queue| self.quiet_running_in(queue))
            .any(|task| task.id == id && task.cancel.load(Ordering::Relaxed))
    }

    /// Ask every running task that can stop to stop.
    #[cfg(test)]
    pub(crate) fn cancel_background_tasks(&mut self) {
        let ids: Vec<u64> = self.running_tasks().map(|task| task.id).collect();
        for id in ids {
            self.cancel_background_task(id);
        }
    }

    /// Drive every running task to its end, the slow queue's first.
    ///
    /// The seam for a test that started its work through a call that does not
    /// hand back the id, such as a wire tool or a menu entry.
    #[cfg(test)]
    pub(crate) fn finish_background_tasks(&mut self) -> Polled {
        let mut polled = Polled::default();
        loop {
            let Some(id) = self.running_tasks().next().map(|task| task.id) else {
                break;
            };
            let step = self.finish_background_task(id);
            polled.changed |= step.changed;
            polled.installed.extend(step.installed);
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
