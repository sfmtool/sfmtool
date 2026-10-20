// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The seam between the GUI thread and a worker: what an operation produces,
//! what it writes when it does not, and what the rest of the viewer is allowed
//! to do while it runs.
//!
//! Headless, and deterministic without waiting on a clock: the one sleep, in
//! the test of a waiting task's row, only makes the wait long enough to tell
//! apart from the work. Two seams make that possible, and both are the
//! mechanism rather than a fake of it:
//!
//! - [`AppState::start_background_task`] takes the work as a [`Job`], so a test can
//!   hand it one that waits on a gate the test opens. Everything about the
//!   state machine can then be asserted at an instant of the test's choosing,
//!   with a real worker really running.
//! - [`AppState::finish_background_task`] blocks on the operation's own channel,
//!   applying each report as it arrives, so "wait until it is done" is the
//!   worker's own message rather than a poll loop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use sfmtool_core::bench::StageKind;
use sfmtool_core::progress::Progress;
use sfmtool_core::{BundleAdjustOptions, SfmrReconstruction};

use crate::action_log::{ActionLog, Actor, Entry};
use crate::bench::NormalStep;
use crate::progress::Detail;
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;

use super::{Finished, Job, Operation, Queue, QuietJob, QuietRank, QuietSource, TaskState};

/// A state holding one node an adjustment can actually run on, and its id.
///
/// The resection fixture with image 1 pushed a few pixels off the pose its own
/// keypoints were computed at, which is [`crate::state::edits::tests`]'s
/// `adjustable_state`: the same value, because these tests are about the move
/// to a worker and not about the solve.
fn adjustable() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    let image = &mut state.scene[0].recon_mut().image_table.images[1];
    image.translation_xyz += nalgebra::Vector3::new(0.02, -0.015, 0.01);
    (state, id)
}

/// A second node beside the first, for the half of the refusal rule that says
/// other nodes are untouched.
fn second_node(state: &mut AppState) -> ReconId {
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(32)));
    state.scene[1].id
}

/// The newest Action Log entry.
fn newest(state: &AppState) -> &Entry {
    state.action_log.entries().next_back().expect("an entry")
}

/// A gate a job waits at, so a test can hold an operation open for as long as
/// it needs to ask questions of a running one.
struct Gate {
    open: mpsc::Sender<()>,
    wait: Option<mpsc::Receiver<()>>,
}

impl Gate {
    fn new() -> Self {
        let (open, wait) = mpsc::channel();
        Gate {
            open,
            wait: Some(wait),
        }
    }

    /// Take the waiting half, for a job to block on.
    fn held(&mut self) -> mpsc::Receiver<()> {
        self.wait.take().expect("the gate is taken once")
    }

    /// Let the job through.
    fn open(&self) {
        self.open.send(()).expect("the job is still waiting");
    }
}

/// A job that waits at `gate` and then produces nothing, successfully.
///
/// The [`Finished::Failed`] is deliberate: a fake worker has no reconstruction
/// to produce, and every test that uses this is about the machinery around the
/// value rather than about the value.
fn waiting_job(gate: mpsc::Receiver<()>) -> Job {
    Box::new(move |_progress| {
        let _ = gate.recv();
        Finished::Failed("the fake worker produced nothing".to_string())
    })
}

// -- What a finished operation installs -----------------------------------

/// The one that says the move was safe: what the worker installs is the
/// kernel's own answer, and the entry reads as the synchronous edit's did.
#[test]
fn a_backgrounded_adjustment_installs_the_kernel_s_own_answer() {
    let (mut state, id) = adjustable();
    let options = BundleAdjustOptions::default();
    // The same operation, run here on this thread, which is what the edit used
    // to do inside the frame that asked for it.
    let source = Arc::clone(&state.scene[0].edited().base);
    let (expected, _) = sfmtool_core::bundle_adjust(&source, &options, &Progress::none())
        .expect("the fixture is well posed");

    state.start_bundle_adjust(id, &options).expect("well posed");
    state.finish_background_tasks();

    let installed = state.scene[0].recon();
    assert_eq!(installed.image_count(), expected.image_count());
    assert_eq!(installed.point_count(), expected.point_count());
    // Compared to a tolerance rather than bit for bit: what is under test is
    // that the worker installed the answer the kernel gave it, not that a
    // rayon-parallel solve reduces its floats in the same order twice.
    for (i, (a, b)) in installed
        .image_table
        .images
        .iter()
        .zip(&expected.image_table.images)
        .enumerate()
    {
        let moved = (a.camera_center() - b.camera_center()).norm();
        assert!(
            moved < 1e-9,
            "image {i} is {moved} from the kernel's answer"
        );
    }
    // And the sentence is the one the synchronous edit wrote, serials and all.
    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert!(
        entry.text.starts_with("Bundle adjusted run_a: 8 images, "),
        "{}",
        entry.text
    );
    let serials = state.scene[0].history.versions();
    assert!(
        newest(&state)
            .text
            .ends_with(&format!("({} → {})", serials[0].serial, serials[1].serial)),
        "{}",
        newest(&state).text
    );
}

/// The entry's cost is the operation's, and its actor is whoever asked.
#[test]
fn the_entry_is_the_operation_s_cost_and_the_asker_s() {
    let (mut state, id) = adjustable();
    // Whoever asked: an agent, which is what the MCP drain sets while it
    // applies a batch, and which has moved on long before the answer lands.
    state.action_log.set_actor(Actor::Mcp);
    state
        .start_bundle_adjust(id, &BundleAdjustOptions::default())
        .expect("well posed");
    state.action_log.set_actor(Actor::User);

    state.finish_background_tasks();

    let entry = newest(&state);
    assert_eq!(entry.actor, Actor::Mcp, "{}", entry.text);
    // The cost covers the whole operation rather than the frame that installed
    // it: every stage the solve reported ran inside it, so an entry timed from
    // the write could not contain them.
    crate::test_support::assert_timed_from_the_work(&mut state.action_log);
    let entry = newest(&state);
    let named: std::time::Duration = entry
        .detail
        .iter()
        .filter_map(|row| match row {
            crate::progress::Detail::Phase { depth: 0, took, .. } => Some(*took),
            _ => None,
        })
        .sum();
    assert!(
        named > std::time::Duration::ZERO,
        "the operation reported no stages at all: {:?}",
        entry.detail
    );
}

/// The log records outcomes, not intentions.
#[test]
fn nothing_is_logged_when_an_operation_starts() {
    let (mut state, id) = adjustable();
    let mut gate = Gate::new();
    let before = state.action_log.revision();

    state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");

    assert_eq!(
        state.action_log.revision(),
        before,
        "starting an operation wrote {:?}",
        newest(&state).text,
    );
    gate.open();
    state.finish_background_tasks();
    // And the row that does arrive is the outcome's.
    assert_eq!(state.action_log.revision(), before + 1);
}

// -- What the rest of the viewer may do meanwhile --------------------------

/// The busy node refuses every change to itself, in one sentence; another node
/// accepts all of them.
#[test]
fn the_busy_node_refuses_and_another_node_does_not() {
    let (mut state, busy) = adjustable();
    let other = second_node(&mut state);
    // Something to undo, redo and jump to on the free node.
    state
        .delete_point(PointRef::new(other, 3))
        .expect("a live point");
    let target = state.scene[1].history.versions()[0].serial;

    let mut gate = Gate::new();
    state
        .start_background_task(Operation::BUNDLE_ADJUST, busy, waiting_job(gate.held()))
        .expect("nothing else is running");
    let expected = "run_a is busy: Bundle adjust is still running.";

    // An edit.
    assert_eq!(
        state.delete_point(PointRef::new(busy, 3)),
        Err(expected.to_string())
    );
    // An undo, a redo and a jump.
    assert_eq!(state.undo(busy), Err(expected.to_string()));
    assert_eq!(state.redo(busy), Err(expected.to_string()));
    assert_eq!(
        state.jump_to_version(busy, state.scene[0].history.versions()[0].serial),
        Err(expected.to_string())
    );
    // A save, and a close.
    assert_eq!(
        state.save_node_as(busy, std::path::Path::new("/runs/never.sfmr"), None),
        Err(expected.to_string())
    );
    assert_eq!(state.close_node(busy), Err(expected.to_string()));
    assert_eq!(state.close_all(), Err(expected.to_string()));
    assert_eq!(state.scene.len(), 2, "a refused close closed something");

    // The other node is untouched: it is not the value the worker was handed.
    state
        .delete_point(PointRef::new(other, 4))
        .expect("an edit of a free node");
    state.undo(other).expect("an undo of a free node");
    state.redo(other).expect("a redo of a free node");
    state
        .jump_to_version(other, target)
        .expect("a jump of a free node");
    // Saving writes a file, so this one is asked of the refusal rather than of
    // the disk: what a free node's save must not be is busy.
    assert_eq!(state.busy_refusal(other), None);
    state.close_node(other).expect("a close of a free node");

    gate.open();
    state.finish_background_tasks();
}

/// A third node beside the first two, for a queue with two tasks waiting.
fn third_node(state: &mut AppState) -> ReconId {
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(24)));
    state.scene.last().expect("just appended").id
}

/// A job that waits at `gate`, counts that it ran, and produces nothing.
fn noted_job(gate: mpsc::Receiver<()>, ran: Arc<AtomicUsize>) -> Job {
    Box::new(move |_progress| {
        let _ = gate.recv();
        ran.fetch_add(1, Ordering::SeqCst);
        Finished::Failed("the fake worker produced nothing".to_string())
    })
}

/// The ids of the tasks waiting in the slow queue, the next one first.
fn slow_waiting(state: &AppState) -> Vec<u64> {
    state
        .queue(super::Queue::Slow)
        .waiting
        .iter()
        .map(|task| task.id)
        .collect()
}

/// Each queue runs one task at a time: a second and a third start in the same
/// queue wait, in the order they were started, each under the id its start
/// returned, and each starts in the poll that lands the one ahead of it.
#[test]
fn two_operations_in_the_same_queue_run_one_after_the_other() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let third = third_node(&mut state);
    let mut gates = [Gate::new(), Gate::new(), Gate::new()];
    let first = state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(gates[0].held()))
        .expect("nothing else is running");
    let second = state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            other,
            waiting_job(gates[1].held()),
        )
        .expect("a busy queue queues the task");
    let last = state
        .start_background_task(
            Operation::REFIT_ALL_POINTS,
            third,
            waiting_job(gates[2].held()),
        )
        .expect("a busy queue queues the task");
    assert_eq!(slow_waiting(&state), [second, last]);
    assert!(matches!(
        state.task(second),
        Some(super::TaskState::Waiting(task)) if task.operation.name == "Retriangulate all points"
    ));
    assert!(
        state.running_task(first).is_some(),
        "the first is not running"
    );

    gates[0].open();
    state.finish_background_task(first);
    // Started in the poll that landed the first, with the third still behind.
    assert!(
        state.running_task(second).is_some(),
        "the second did not start"
    );
    assert_eq!(slow_waiting(&state), [last]);

    gates[1].open();
    state.finish_background_task(second);
    assert!(
        state.running_task(last).is_some(),
        "the third did not start"
    );
    assert_eq!(slow_waiting(&state), Vec::<u64>::new());
    gates[2].open();
    state.finish_background_task(last);

    // The records are in the order the tasks ran.
    let order: Vec<u64> = state.finished_tasks.iter().map(|task| task.id).collect();
    assert_eq!(order, [first, second, last]);
    assert_eq!(state.action_log.entries().count(), 3, "one row each");
}

/// A locking task locks its node from the moment it is queued: an edit, a
/// close and a second locking task in either queue are refused with the
/// waiting sentence, and with the running sentence once it starts. The task
/// itself is not refused by its own lock: it starts and lands, and the node is
/// free again after.
#[test]
fn a_waiting_task_locks_its_node_from_the_moment_it_is_queued() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut ahead = Gate::new();
    let first = state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(ahead.held()))
        .expect("nothing else is running");
    let mut queued = Gate::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let task = state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            other,
            noted_job(queued.held(), Arc::clone(&ran)),
        )
        .expect("a busy queue queues the task");
    let waiting = "demo is busy: Retriangulate all points is waiting to run.";

    assert_eq!(state.busy_refusal(other), Some(waiting.to_string()));
    assert_eq!(
        state.delete_point(PointRef::new(other, 3)),
        Err(waiting.to_string())
    );
    assert_eq!(state.close_node(other), Err(waiting.to_string()));
    assert_eq!(state.scene.len(), 2, "a refused close closed something");
    let mut refused = Gate::new();
    assert_eq!(
        state.start_background_task(Operation::BENCH_FIT, other, waiting_job(refused.held())),
        Err(waiting.to_string()),
        "a locking task in the other queue",
    );
    let mut refused = Gate::new();
    assert_eq!(
        state.start_background_task(
            Operation::PRUNE_COVERED_OBSERVATIONS,
            other,
            waiting_job(refused.held())
        ),
        Err(waiting.to_string()),
        "a locking task in the same queue",
    );

    ahead.open();
    state.finish_background_task(first);
    assert_eq!(
        state.busy_refusal(other),
        Some("demo is busy: Retriangulate all points is still running.".to_string())
    );
    queued.open();
    state.finish_background_task(task);
    assert_eq!(ran.load(Ordering::SeqCst), 1, "the queued job ran once");
    assert_eq!(state.busy_refusal(other), None);
    state
        .delete_point(PointRef::new(other, 3))
        .expect("an edit of a free node");
}

/// The id a start function returns names the task when it waits: an
/// adjustment queued behind another node's task is found by that id while it
/// waits, then starts and installs the kernel's answer.
#[test]
fn a_start_function_s_id_names_its_task_while_it_waits() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut ahead = Gate::new();
    let first = state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            other,
            waiting_job(ahead.held()),
        )
        .expect("nothing else is running");
    let task = state
        .start_bundle_adjust(id, &BundleAdjustOptions::default())
        .expect("queued rather than refused")
        .expect("a task");
    match state.task(task) {
        Some(super::TaskState::Waiting(waiting)) => {
            assert_eq!(waiting.operation.name, "Bundle adjust");
            assert_eq!(waiting.label, "run_a");
            assert_eq!(waiting.node, Some(id));
        }
        _ => panic!("the adjustment is not waiting"),
    }

    ahead.open();
    state.finish_background_task(first);
    state.finish_background_task(task);
    assert_eq!(
        state.scene[0].history.versions().len(),
        2,
        "it did not land"
    );
    let landed = state.finished_task(task).expect("its record");
    assert!(landed.outcome.is_ok(), "{:?}", landed.outcome);
    assert!(landed.changed);
}

/// Cancelling a waiting task takes it out of its queue before its job runs,
/// writes its own failed row as whoever asked, keeps its record in the ring
/// and leaves the running task alone.
#[test]
fn cancelling_a_waiting_task_removes_it_and_leaves_the_running_one() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut ahead = Gate::new();
    let first = state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(ahead.held()))
        .expect("nothing else is running");
    let ran = Arc::new(AtomicUsize::new(0));
    let (_never, gate) = mpsc::channel();
    state.action_log.set_actor(Actor::Mcp);
    let task = state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            other,
            noted_job(gate, Arc::clone(&ran)),
        )
        .expect("a busy queue queues the task");
    state.action_log.set_actor(Actor::User);
    assert_eq!(
        state.cancel_refusal(task),
        None,
        "a waiting task is refused"
    );

    state.cancel_background_task(task);
    assert!(state.queue(super::Queue::Slow).waiting.is_empty());
    let row = newest(&state);
    assert!(row.failed);
    assert_eq!(
        row.text,
        "Retriangulate all points of demo cancelled before it started"
    );
    assert_eq!(row.actor, Actor::Mcp, "the row is not the asker's");
    let record = state.finished_task(task).expect("kept in the ring");
    assert_eq!(
        record.outcome,
        Err("Retriangulate all points of demo cancelled before it started".to_string())
    );
    assert_eq!(record.took, std::time::Duration::ZERO);
    assert!(!record.changed);
    assert_eq!(state.busy_refusal(other), None, "the node is still locked");
    let running = state.running_task(first).expect("the running task");
    assert!(
        !running.cancel.load(std::sync::atomic::Ordering::Relaxed),
        "the running task was asked to stop"
    );

    ahead.open();
    state.finish_background_task(first);
    assert_eq!(ran.load(Ordering::SeqCst), 0, "the cancelled job ran");
    assert!(state.running_tasks().next().is_none());
}

/// A waiting task's row is timed from when it started rather than from when
/// it was queued, so its cost is the work's and not the wait's.
///
/// The one sleep in this module: the task ahead is held long enough that the
/// wait is measurable, and nothing is asserted about how long anything took
/// beyond that floor.
#[test]
fn a_waiting_task_s_row_is_timed_from_when_it_started() {
    const WAIT: std::time::Duration = std::time::Duration::from_millis(50);
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut ahead = Gate::new();
    let first = state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(ahead.held()))
        .expect("nothing else is running");
    let task = state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            other,
            Box::new(|_progress| Finished::Failed("quick".to_string())),
        )
        .expect("a busy queue queues the task");
    let queued = state.waiting_task(task).expect("waiting").queued;
    std::thread::sleep(WAIT);

    ahead.open();
    state.finish_background_task(first);
    let started = state.running_task(task).expect("started").started;
    assert!(
        started >= queued + WAIT,
        "it started before the task ahead landed"
    );
    state.finish_background_task(task);
    state
        .action_log
        .settle(std::time::Instant::now(), Vec::new());
    // Read after the settle, so it is no earlier than the instant the settle
    // stamped the row at.
    let now = std::time::Instant::now();
    let took = newest(&state).took.expect("settled");
    assert!(
        took <= now.duration_since(started),
        "the row's {took:?} runs from before the task started"
    );
    assert!(
        state.finished_task(task).expect("kept").took <= now.duration_since(started),
        "the record's cost runs from before the task started"
    );
}

/// The two queues are independent: a fast task starts and lands while a slow
/// one runs, and the slow one is still running after it.
#[test]
fn two_operations_in_different_queues_run_at_the_same_time() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut slow_gate = Gate::new();
    let slow = state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(slow_gate.held()))
        .expect("nothing else is running");
    let mut fast_gate = Gate::new();
    let fast = state
        .start_background_task(Operation::BENCH_FIT, other, waiting_job(fast_gate.held()))
        .expect("the fast queue is idle");
    assert_ne!(slow, fast);
    assert_eq!(
        state
            .running_tasks()
            .map(|task| task.id)
            .collect::<Vec<_>>(),
        [slow, fast],
        "both run, the slow queue's first",
    );

    fast_gate.open();
    state.finish_background_task(fast);
    assert!(state.running_task(slow).is_some(), "the slow task stopped");
    assert_eq!(
        state.finished_task(fast).map(|task| task.operation.name),
        Some("Fit track")
    );

    slow_gate.open();
    state.finish_background_task(slow);
    // Both records are kept, and each queue's last finished task is its own.
    assert_eq!(
        state
            .last_finished_in(super::Queue::Slow)
            .map(|task| task.id),
        Some(slow)
    );
    assert_eq!(
        state
            .last_finished_in(super::Queue::Fast)
            .map(|task| task.id),
        Some(fast)
    );
}

/// A node has at most one locking task: a locking task in the other queue is
/// refused on the busy node with the busy sentence, and accepted on another.
#[test]
fn a_locking_task_in_the_other_queue_is_refused_on_the_busy_node() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut gate = Gate::new();
    let task = state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");

    let mut refused = Gate::new();
    assert_eq!(
        state.start_background_task(Operation::BENCH_FIT, id, waiting_job(refused.held())),
        Err("run_a is busy: Bundle adjust is still running.".to_string()),
    );
    let mut beside = Gate::new();
    let fit = state
        .start_background_task(Operation::BENCH_FIT, other, waiting_job(beside.held()))
        .expect("another node is free");
    // A fast task locks its node as a slow one does.
    assert_eq!(
        state.busy_refusal(other),
        Some("demo is busy: Fit track is still running.".to_string())
    );

    beside.open();
    state.finish_background_task(fit);
    gate.open();
    state.finish_background_task(task);
    assert_eq!(state.busy_refusal(id), None);
}

/// The record of finished tasks keeps the last `FINISHED_KEPT`, newest last,
/// and a task pushed out of it is found by nothing.
#[test]
fn the_finished_tasks_are_kept_in_a_ring() {
    let (mut state, id) = adjustable();
    let mut ids = Vec::new();
    for _ in 0..super::FINISHED_KEPT + 2 {
        let task = state
            .start_background_task(
                Operation::BUNDLE_ADJUST,
                id,
                Box::new(|_progress| Finished::Failed("nothing".to_string())),
            )
            .expect("the last one is over");
        state.finish_background_task(task);
        ids.push(task);
    }
    assert_eq!(state.finished_tasks.len(), super::FINISHED_KEPT);
    assert!(state.task(ids[0]).is_none(), "the oldest was kept");
    assert!(state.task(ids[1]).is_none(), "the second oldest was kept");
    assert!(matches!(
        state.task(ids[2]),
        Some(super::TaskState::Finished(_))
    ));
    assert_eq!(
        state.last_finished_task().map(|task| task.id),
        ids.last().copied()
    );
}

/// Each operation's declared queue, lock and quiet value are the ones the
/// spec's table gives (`specs/gui/background-tasks.md` § "Two queues").
#[test]
fn each_operation_declares_the_spec_s_queue_lock_and_quiet_value() {
    use super::Queue::{Fast, Slow};
    let table = [
        ("Bundle adjust", Slow, true, false),
        ("Convert to embedded patches", Slow, true, false),
        ("Retriangulate all points", Slow, true, false),
        ("Refit all points", Slow, true, false),
        ("Switch camera model", Slow, true, false),
        ("Prune covered observations", Slow, true, false),
        ("Add image to tracks", Slow, true, false),
        ("Build index files", Slow, true, false),
        ("Open", Fast, false, false),
        ("Fit track", Fast, true, false),
        ("Set track stage", Fast, true, false),
        ("Fit normal", Fast, true, false),
        ("Finite difference normal", Fast, true, false),
        ("Grid plane normal", Fast, true, false),
        ("Search descriptors", Fast, true, false),
        ("Geometry search", Fast, true, false),
        ("Create track at pixel", Fast, true, false),
        ("Find nearby tracks", Fast, true, false),
        ("Live evaluation", Fast, false, true),
        ("Camera model proposal", Fast, false, true),
    ];
    assert_eq!(
        table.len(),
        Operation::ALL.len(),
        "the table names every operation"
    );
    for operation in Operation::ALL {
        let row = table
            .iter()
            .find(|(name, ..)| *name == operation.name)
            .unwrap_or_else(|| panic!("{} is not in the table", operation.name));
        assert_eq!(
            (operation.queue, operation.locks, operation.quiet),
            (row.1, row.2, row.3),
            "{}",
            operation.name
        );
    }
}

// -- Cancellation ----------------------------------------------------------

/// A cancelled operation writes a failed entry, pushes no version, and leaves
/// the value it was reading exactly where it was.
#[test]
fn a_cancelled_operation_writes_a_failed_entry_and_pushes_nothing() {
    let (mut state, id) = adjustable();
    let versions = state.scene[0].history.versions().len();
    let before = Arc::clone(&state.scene[0].edited().base);
    let centres: Vec<nalgebra::Point3<f64>> = state.scene[0]
        .recon()
        .image_table
        .images
        .iter()
        .map(|image| image.camera_center())
        .collect();

    let job = state
        .bundle_adjust_job(id, &BundleAdjustOptions::default())
        .expect("the fixture is well posed");
    cancel_before_it_runs(&mut state, id, Operation::BUNDLE_ADJUST, job);

    let entry = newest(&state);
    assert!(entry.failed, "{}", entry.text);
    assert!(
        entry.text == "Bundle adjust of run_a cancelled",
        "{}",
        entry.text
    );
    // A cancelled operation spent the time it ran for, so it is timed from its
    // start like a successful one. An entry costing it at the frame that
    // collected the answer would read as milliseconds for a solve somebody sat
    // through, and the sentence no longer carries an elapsed of its own to
    // disagree with the column.
    crate::test_support::assert_timed_from_the_work(&mut state.action_log);
    assert_eq!(state.scene[0].history.versions().len(), versions);
    assert!(
        Arc::ptr_eq(&before, &state.scene[0].edited().base),
        "the input value was disturbed by an operation that read it",
    );
    let after: Vec<nalgebra::Point3<f64>> = state.scene[0]
        .recon()
        .image_table
        .images
        .iter()
        .map(|image| image.camera_center())
        .collect();
    assert_eq!(centres, after);
}

/// `cancellable` is a declaration, so every operation that makes it is held to
/// it: cancelled, each one stops.
#[test]
fn every_operation_that_says_it_is_cancellable_really_is() {
    for operation in Operation::ALL {
        if !operation.cancellable {
            continue;
        }
        // A quiet task writes no row, so its claim is held on its answer.
        if operation.quiet {
            assert!(
                quiet_job_stops(operation),
                "{} claims to be cancellable and did not stop",
                operation.name
            );
            continue;
        }
        let mut task = real_task(operation);
        let versions = task.state.scene[0].history.versions().len();
        let nodes = task.state.scene.len();
        let job = task.job.take().expect("the starter built one");
        cancel_before_it_runs(&mut task.state, task.id, operation, job);
        let entry = newest(&task.state);
        assert!(
            entry.failed && entry.text.ends_with("cancelled"),
            "{} claims to be cancellable and did not stop: {}",
            operation.name,
            entry.text,
        );
        assert_eq!(
            task.state.scene[0].history.versions().len(),
            versions,
            "{} pushed a version after being cancelled",
            operation.name,
        );
        assert_eq!(
            task.state.scene.len(),
            nodes,
            "{} appended a node after being cancelled",
            operation.name,
        );
    }
}

/// Whether the quiet `operation`'s real work, run with the cancel flag already
/// set, stops rather than answering.
///
/// Run through the job builder rather than the queue, since the queue hands a
/// quiet task's answer to its source and keeps nothing of it.
fn quiet_job_stops(operation: Operation) -> bool {
    match operation.name {
        // An evaluation of a track on the bench fixture, whose photographs it
        // would otherwise decode and render.
        "Live evaluation" => {
            let (mut state, id) = crate::bench::tests::state();
            let label = crate::bench::tests::put_on_bench(&mut state, id);
            let job = state
                .bench_evaluate_job(id, &label)
                .expect("the fixture's track is readable");
            let cancel = std::sync::atomic::AtomicBool::new(true);
            let progress = Progress::none().cancelled_by(&cancel);
            matches!(job(&progress), crate::bench::live::Measured::Cancelled)
        }
        // A proposal's job on a spline camera, with and without the preview,
        // which would otherwise switch the camera and retriangulate its tracks.
        "Camera model proposal" => {
            let (state, _) = crate::camera_model_proposal::tests::spline_state();
            state.camera_model_proposal_job_stops_when_cancelled()
        }
        other => panic!("{other} has no starter here; add one"),
    }
}

/// Start `operation`'s real work with the cancel flag already set, and drive it
/// to its end.
///
/// The gate is what removes the race: the flag is set while the job is still
/// waiting to start, so the kernel meets it at its first poll however fast the
/// fixture solves. Cancelling *after* a start would be a bet on the machine
/// being slower than the test.
#[track_caller]
fn cancel_before_it_runs(state: &mut AppState, id: ReconId, operation: Operation, job: Job) {
    let mut gate = Gate::new();
    let held = gate.held();
    state
        .start_background_task(
            operation,
            id,
            Box::new(move |progress| {
                let _ = held.recv();
                job(progress)
            }),
        )
        .expect("nothing else is running");
    state.cancel_background_tasks();
    gate.open();
    state.finish_background_tasks();
}

/// One operation's real work, with everything it needs to run: a test of the
/// declaration then tests the kernel rather than a stand-in for it.
struct RealTask {
    /// The state the job was built from and runs against.
    state: AppState,
    /// The node it runs on.
    id: ReconId,
    /// The work itself, taken out when it is handed to the worker.
    job: Option<Job>,
    /// The directory the node's files live in, where it has any: a search
    /// reads `.sift` files and a `.kdf` off the disk, and they have to outlive
    /// the task.
    _workspace: Option<tempfile::TempDir>,
}

/// The work `operation` really does, over a fixture that can actually do it.
///
/// The `match` is exhaustive on purpose: a new [`Operation`] fails here until
/// somebody says how to start it, which is what keeps the walk over
/// [`Operation::ALL`] honest.
fn real_task(operation: Operation) -> RealTask {
    match operation.name {
        // The open, over a saved file with photographs and patch frames: its
        // job is the one File > Open starts, reading, building thumbnails and
        // rendering bitmaps. It is started here on the fixture's node, which a
        // real open does not lock; what is held to the claim is the job.
        "Open" => {
            let workspace = tempfile::tempdir().expect("a temporary directory");
            let path = crate::state::open::tests::saved_with_photographs(workspace.path());
            let mut state = AppState::new();
            state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));
            let id = state.scene[0].id;
            let job =
                crate::state::open::open_job(vec![path], std::sync::Arc::clone(&state.photographs));
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: Some(workspace),
            }
        }
        "Bundle adjust" => {
            let (state, id) = adjustable();
            let job = state
                .bundle_adjust_job(id, &BundleAdjustOptions::default())
                .expect("the fixture is well posed");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // The conversion, over a workspace fixture: it reads a real `.sift`
        // file per image, which the demo node has none of.
        "Convert to embedded patches" => {
            let workspace = tempfile::tempdir().expect("a temporary directory");
            let (state, id) = crate::state::edits::tests::convertible_state(workspace.path());
            let job = state
                .convert_to_embedded_patches_job(id)
                .expect("the fixture is a sift_files node with its companions");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: Some(workspace),
            }
        }
        // Every point re-solved, over the adjustment's own fixture: it needs
        // exactly what the adjustment needs -- a pixel per observation and at
        // least one posed image.
        "Retriangulate all points" => {
            let (state, id) = adjustable();
            let job = state
                .retriangulate_all_points_job(id)
                .expect("the fixture is well posed");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // Every point refit, over the bench's fixture: an embedded_patches
        // node with a patch frame per point and a textured photograph cached
        // for every image.
        "Refit all points" => {
            let (state, id) = crate::bench::tests::state();
            let job = state
                .refit_all_points_job(id)
                .expect("the fixture carries keypoints, frames and poses");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // The proposal's Apply, over the retriangulation's fixture: its pinhole
        // camera switched to SFMTOOL_PINHOLE and its tracks retriangulated.
        "Switch camera model" => {
            let (state, id) = adjustable();
            let request = crate::state::edits::SwitchCameraModelRequest {
                camera: 0,
                camera_model: Some("SFMTOOL_PINHOLE".to_string()),
                ..Default::default()
            };
            let job = state
                .switch_camera_model_job(id, &request)
                .expect("the fixture is well posed");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // The prune, over the retriangulation's fixture with a patch frame per
        // point: an observation's footprint is that frame projected into the
        // image that saw it.
        "Prune covered observations" => {
            let mut state = AppState::new();
            state.append_node(crate::scene_graph::tests::prunable_node("/runs/run_a.sfmr"));
            let id = state.scene[0].id;
            let job = state
                .prune_covered_observations_job(
                    id,
                    &sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions::default(),
                )
                .expect("the fixture carries frames, pixels and poses");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // Add Image to Tracks, over the plane capture with image 0's
        // observations taken out: it decodes the photographs and measures
        // every point against them.
        "Add image to tracks" => {
            let (state, id) = crate::add_image_to_tracks::tests::untracked_state();
            let job = state
                .add_image_to_tracks_job(crate::scene::ImageRef::new(id, 0))
                .expect("a posed image of an embedded_patches node with frames");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // The two steps that read photographs, over the bench fixture: a point
        // on the bench and a textured photograph cached for every image.
        "Fit track"
        | "Set track stage"
        | "Fit normal"
        | "Finite difference normal"
        | "Grid plane normal" => {
            let (mut state, id) = crate::bench::tests::state();
            let label = crate::bench::tests::put_on_bench(&mut state, id);
            let job = match operation.name {
                "Fit track" => state.bench_fit_job(id, &label, None),
                "Fit normal" => state.bench_normal_job(id, &label, NormalStep::Photometric),
                "Finite difference normal" => state.bench_normal_job(
                    id,
                    &label,
                    NormalStep::FiniteDifference(Default::default()),
                ),
                "Grid plane normal" => {
                    state.bench_normal_job(id, &label, NormalStep::GridPlane(Default::default()))
                }
                _ => state.bench_stage_job(id, &label, StageKind::Cluster, None),
            }
            .expect("the fixture's track is readable");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // The search, over the workspace fixture: it queries a real `.kdf` and
        // reads a real `.sift` file, neither of which the demo node has.
        "Search descriptors" => {
            let workspace = tempfile::tempdir().expect("a temporary directory");
            let (mut state, id, label) = crate::sift_index::tests::searchable(workspace.path());
            let job = state
                .bench_search_job(id, &label, 0, None, None)
                .expect("the fixture has an index and a searchable observation");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: Some(workspace),
            }
        }
        "Geometry search" => {
            let (mut state, id) = crate::bench::tests::state();
            let label = crate::bench::tests::put_on_bench(&mut state, id);
            let job = state
                .bench_geometry_search_job(id, &label, 0)
                .expect("the fixture has a framed track and decoded photographs");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        // The index-files build, over the same workspace fixture: it reads
        // every `.sift` file and photograph of the node and writes a `.kdf`
        // and a `.matches` beside its `.sfmr`.
        "Build index files" => {
            let workspace = tempfile::tempdir().expect("a temporary directory");
            let (state, id) = crate::sift_index::tests::state_in(workspace.path());
            crate::sift_index::tests::with_sift_files(&state, id, [900.0, 500.0]);
            let job = state
                .build_index_files_job(id)
                .expect("the fixture is a saved node with .sift files");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: Some(workspace),
            }
        }
        // Create Track Here, over the plane capture its own tests use: a grid
        // of points on a textured plane and a pixel between them, which the
        // cascade builds a track at when it is let run.
        "Create track at pixel" => {
            let (mut state, id) = crate::bench::track_at_pixel::tests::plane_state();
            let job = state
                .create_track_at_pixel_job(
                    crate::scene::ImageRef::new(id, 0),
                    crate::bench::track_at_pixel::tests::textured_pixel(),
                    None,
                )
                .expect("the fixture is a posed embedded_patches node");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        "Find nearby tracks" => {
            let (mut state, id) = crate::bench::track_at_pixel::tests::plane_state();
            let job = state
                .find_nearby_tracks_job(
                    crate::scene::ImageRef::new(id, 0),
                    crate::bench::track_at_pixel::tests::textured_pixel(),
                    true,
                    None,
                )
                .expect("the fixture is a posed embedded_patches node");
            RealTask {
                state,
                id,
                job: Some(job),
                _workspace: None,
            }
        }
        other => panic!("{other} has no starter here; add one"),
    }
}

// -- Reports ---------------------------------------------------------------

/// Reports do not pile up: however many an operation makes between two frames,
/// the GUI thread drains them once and reads the collector once.
#[test]
fn many_reports_in_one_frame_are_one_drain() {
    let (mut state, id) = adjustable();
    let reported = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&reported);
    // The worker says it has reported, then waits: what the test asserts is
    // the state of the queue at an instant when a thousand reports are
    // certainly in the past.
    let (said, heard) = mpsc::channel();
    let mut gate = Gate::new();
    let held = gate.held();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |progress| {
                for i in 0..1000u64 {
                    progress.count(i, Some(1000), "iteration");
                    counted.fetch_add(1, Ordering::Relaxed);
                }
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                Finished::Failed("the fake worker produced nothing".to_string())
            }),
        )
        .expect("nothing else is running");

    heard.recv().expect("the worker reported");
    assert_eq!(reported.load(Ordering::Relaxed), 1000);

    let polled = state.poll_background_tasks();
    assert!(polled.changed, "a thousand reports moved nothing");
    assert!(polled.installed.is_empty(), "nothing has finished");
    // One drain took all of them: the queue is empty, and the collector holds
    // the newest count rather than a thousand of them.
    assert_eq!(
        state.poll_background_tasks(),
        super::Polled::default(),
        "reports were left in the queue",
    );
    let count = state
        .running_tasks()
        .next()
        .expect("still running")
        .collector
        .count()
        .expect("the operation counted");
    assert_eq!(
        (count.done, count.total, count.unit),
        (999, Some(1000), "iteration")
    );

    gate.open();
    state.finish_background_tasks();
}

/// A job that panics frees the node rather than wedging it.
///
/// The panic this provokes is printed by the runtime's hook and is expected:
/// what is under test is that it comes back as an outcome rather than as a
/// node locked by an operation that will never answer.
#[test]
fn a_worker_that_panics_frees_the_node() {
    let (mut state, id) = adjustable();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(|_progress| panic!("a job that fails on the worker")),
        )
        .expect("nothing else is running");

    state.finish_background_tasks();

    assert!(
        state.running_tasks().next().is_none(),
        "the node is still locked"
    );
    assert!(newest(&state).failed);
    assert_eq!(state.busy_refusal(id), None);
    // And the node can be edited again.
    state
        .delete_point(PointRef::new(id, 0))
        .expect("a freed node");
}

// -- The operation's identity ---------------------------------------------

/// An id names one operation, and goes on naming it after it has finished.
#[test]
fn an_operation_id_outlives_the_operation() {
    let (mut state, id) = adjustable();
    let mut gate = Gate::new();
    state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");
    let first = state.running_tasks().next().expect("running").id;
    gate.open();
    state.finish_background_tasks();

    let last = state.last_finished_task().expect("it finished");
    assert_eq!(last.id, first, "the answer is about another operation");

    // The next one is a different operation, and says so.
    let mut gate = Gate::new();
    state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("the first one is over");
    assert_ne!(state.running_tasks().next().expect("running").id, first);
    gate.open();
    state.finish_background_tasks();
}

// -- The value the worker was handed --------------------------------------

/// The value was not copied to run in the background: the base the worker reads
/// is the very allocation the node goes on drawing.
#[test]
fn the_worker_reads_the_arc_the_node_is_still_drawing() {
    let (mut state, id) = adjustable();
    let drawn = Arc::clone(&state.scene[0].edited().base);
    let strong = Arc::strong_count(&drawn);

    let job = state
        .bundle_adjust_job(id, &BundleAdjustOptions::default())
        .expect("well posed");
    // The job holds a clone of the value at the cursor, whose base is that same
    // `Arc`: one more owner of one allocation, not a second reconstruction.
    assert_eq!(
        Arc::strong_count(&drawn),
        strong + 1,
        "the job copied the value instead of sharing it",
    );
    state
        .start_background_task(Operation::BUNDLE_ADJUST, id, job)
        .expect("nothing else is running");
    // And the node is still drawing it while the worker reads it.
    assert!(Arc::ptr_eq(&drawn, &state.scene[0].edited().base));

    state.finish_background_tasks();
    // The next version is a new base, as every bulk edit's is.
    assert!(!Arc::ptr_eq(&drawn, &state.scene[0].edited().base));
}

/// An image of the busy node cannot be deleted either, which is the bulk edit
/// the refusal list names first.
#[test]
fn a_bulk_edit_of_the_busy_node_is_refused_too() {
    let (mut state, id) = adjustable();
    let mut gate = Gate::new();
    state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");

    assert_eq!(
        state.delete_image(ImageRef::new(id, 2)),
        Err("run_a is busy: Bundle adjust is still running.".to_string())
    );

    gate.open();
    state.finish_background_tasks();
}

// -- The panel -------------------------------------------------------------

/// The input one headless frame of the panel is given: narrow and tall, which
/// is the shape of the bottom of the left column.
fn input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(260.0, 400.0),
        )),
        ..Default::default()
    }
}

/// Every string one headless frame of the panel painted, with where it landed.
///
/// The positions are what a hover test needs: a tooltip only appears over a
/// widget the context already knows about, so the click-through of
/// `action_log::tests` applies here too and the caller owns the context.
fn painted_at(
    ctx: &egui::Context,
    state: &mut AppState,
    input: egui::RawInput,
) -> Vec<(String, egui::Pos2)> {
    fn walk(shape: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
        match shape {
            egui::Shape::Text(text) => out.push((text.galley.text().to_owned(), text.pos)),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
            _ => {}
        }
    }
    let mut output = ctx.run_ui(input, |ui| super::panel::show(ui, state));
    output.textures_delta.clear();
    let mut out = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut out);
    }
    out
}

/// Everything one frame of the panel painted, in a context of its own.
fn painted(state: &mut AppState) -> Vec<String> {
    let ctx = egui::Context::default();
    crate::test_support::painted_texts(&ctx, input(), |ui| super::panel::show(ui, state))
}

/// A job that opens a phase and leaves it open at the gate, having reported no
/// number at all: the case the panel has nothing to draw a bar from.
fn open_phase_job(gate: mpsc::Receiver<()>, said: mpsc::Sender<()>) -> Job {
    Box::new(move |progress| {
        let _open = progress.phase("gather arrays");
        said.send(()).expect("the test is listening");
        let _ = gate.recv();
        Finished::Failed("the fake worker produced nothing".to_string())
    })
}

/// A job that reports a folded phase table, a message and a count, closes all
/// of it, and then waits at the gate.
///
/// Closed before the wait on purpose: what the panel draws then is exactly what
/// the entry will hold, so the two can be compared row for row rather than
/// allowing for a run that was still going when one of them was read.
fn counted_job(gate: mpsc::Receiver<()>, said: mpsc::Sender<()>) -> Job {
    Box::new(move |progress| {
        drop(progress.phase("gather arrays"));
        {
            let solve = progress.phase("solve");
            for round in 1..=2u64 {
                let mut phase = solve.phase("round");
                phase.note(format_args!("median 1.{round}0 px"));
                drop(phase);
                solve.count(round, Some(2), "round");
            }
            solve.message(
                sfmtool_core::progress::Level::Info,
                format_args!("median 0.412 px"),
            );
        }
        said.send(()).expect("the test is listening");
        let _ = gate.recv();
        Finished::Failed("the fake worker produced nothing".to_string())
    })
}

/// Start the fake work for `operation` on `id` and come back once it has
/// reported everything it means to, so the panel is read at an instant the test
/// chose rather than one the scheduler did.
fn running(state: &mut AppState, id: ReconId, operation: Operation) -> Gate {
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    let job: Job = if operation.cancellable {
        counted_job(gate.held(), said)
    } else {
        open_phase_job(gate.held(), said)
    };
    state
        .start_background_task(operation, id, job)
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_tasks();
    gate
}

/// An operation whose kernels never ask whether they should stop, for the half
/// of the Cancel rule that is about a button nobody can press.
const NOT_CANCELLABLE: Operation = Operation {
    name: "Fake solve",
    cancellable: false,
    kind: crate::action_log::Kind::Edit,
    queue: super::Queue::Slow,
    locks: true,
    quiet: false,
};

/// A session that has run nothing says so for each queue and says nothing
/// else: one greyed line a queue is a panel a reader can skip, where a blank
/// one is a panel that looks broken.
#[test]
fn an_idle_panel_with_nothing_run_says_only_that() {
    let (mut state, _) = adjustable();
    assert_eq!(
        painted(&mut state),
        ["Slow", "Nothing running", "Fast", "Nothing running"]
    );
}

/// A section per queue: each queue's running task in the running form, and
/// one phase list below them, the slow task's until a person clicks the fast
/// task's name. An idle queue shows the last task that finished from it.
#[test]
fn each_queue_has_a_section_and_one_task_s_phases_show_below() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let slow = running(&mut state, id, Operation::BUNDLE_ADJUST);
    let mut fast_gate = Gate::new();
    let (said, heard) = mpsc::channel();
    let held = fast_gate.held();
    let fast = state
        .start_background_task(
            Operation::BENCH_FIT,
            other,
            Box::new(move |progress| {
                drop(progress.phase("decode images"));
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                Finished::Failed("the fake fit produced nothing".to_string())
            }),
        )
        .expect("the fast queue is idle");
    heard.recv().expect("the fit reported");
    state.poll_background_tasks();

    let texts = painted(&mut state);
    let at = |text: &str| texts.iter().position(|t| t == text);
    let order = [
        at("Slow"),
        at("Bundle adjust"),
        at("Fast"),
        at("Fit track"),
        at("Bundle adjust on run_a"),
        at("gather arrays"),
    ];
    assert!(order.iter().all(Option::is_some), "{texts:?}");
    assert!(order.is_sorted(), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "decode images"), "{texts:?}");

    // Clicking the fit's name is what this stands for.
    state.background_phases_of = Some(fast);
    let texts = painted(&mut state);
    assert!(texts.iter().any(|t| t == "Fit track on demo"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "decode images"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "gather arrays"), "{texts:?}");

    // Finished, the fit is the fast section's idle form, and still the task
    // whose phases are shown.
    fast_gate.open();
    state.finish_background_task(fast);
    let texts = painted(&mut state);
    assert!(texts.iter().any(|t| t == "Fit track on demo"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "demo"), "{texts:?}");
    assert!(
        !texts.iter().any(|t| t == "Nothing running"),
        "a queue that ran a task said it had run nothing: {texts:?}"
    );

    slow.open();
    state.finish_background_tasks();
}

/// Idle after an operation: the name, the node, the cost and the phases, all
/// of it showing at once.
///
/// No toggle. The Action Log hides a breakdown behind one because it has a row
/// per action and expanding them all would bury the list; this panel holds one
/// operation, and the breakdown is the only thing it has to say.
#[test]
fn an_idle_panel_shows_the_last_operation_and_its_phases_at_once() {
    let (mut state, id) = adjustable();
    running(&mut state, id, Operation::BUNDLE_ADJUST).open();
    state.finish_background_tasks();

    let drawn = painted(&mut state);
    assert!(
        drawn.iter().any(|text| text == "Bundle adjust"),
        "{drawn:?}"
    );
    assert!(drawn.iter().any(|text| text == "run_a"), "{drawn:?}");
    // What it cost, in the Action Log's spelling of a duration.
    let took = ActionLog::format_took(state.last_finished_task().expect("it finished").took);
    assert!(drawn.contains(&took), "{took:?} is missing from {drawn:?}");

    // The phases, with nothing asked for first.
    assert!(drawn.iter().any(|text| text == "solve"), "{drawn:?}");
    // Unfolded, as the running panel drew them: two runs, two rows.
    assert_eq!(
        drawn
            .iter()
            .filter(|text| text.starts_with("  round"))
            .count(),
        2,
        "{drawn:?}"
    );

    // And nothing to click: a toggle would be a click between a reader and the
    // thing they opened the panel for.
    for glyph in ["+", "-"] {
        assert!(
            !drawn.iter().any(|text| text == glyph),
            "{glyph:?} is still drawn: {drawn:?}",
        );
    }
}

/// The panel and the entry are two views of one collector, and the entry is
/// the *summary* of what the panel showed: every run the panel drew is counted
/// in the row the entry folds it into, and the costs add up. They present it
/// differently on purpose (see
/// `a_running_phase_table_keeps_every_run_and_every_note_apart`); what they
/// must never do is disagree about what happened.
#[test]
fn the_entry_is_the_summary_of_what_the_running_panel_showed() {
    let (mut state, id) = adjustable();
    let gate = running(&mut state, id, Operation::BUNDLE_ADJUST);

    let live = state
        .running_tasks()
        .next()
        .expect("running")
        .collector
        .live()
        .rows;
    let while_running = painted(&mut state);
    assert!(
        while_running.iter().any(|text| text == "Bundle adjust"),
        "{while_running:?}"
    );
    assert!(while_running.iter().any(|text| text == "run_a"));
    assert!(
        while_running
            .iter()
            .any(|text| text.ends_with(" s") || text.ends_with(" ms")),
        "the elapsed is missing from {while_running:?}"
    );

    gate.open();
    state.finish_background_tasks();

    let entry = newest(&state);
    // Every phase the entry folded, against the runs of it the panel drew.
    // `runs` is the count of them and `took` is their sum, which is the whole
    // of what folding claims.
    for (name, depth, runs) in crate::test_support::phase_rows(&entry.detail) {
        let drawn: Vec<&Detail> = live
            .iter()
            .filter(|row| {
                matches!(row, Detail::Phase { name: n, depth: d, .. } if *n == name && *d == depth)
            })
            .collect();
        assert_eq!(
            drawn.len() as u32,
            runs,
            "the entry folded {runs} runs of {name:?} and the panel drew {}",
            drawn.len(),
        );
    }
    // And every message the entry kept is a message the panel painted, since a
    // message is never folded in either view.
    for row in &entry.detail {
        if let Detail::Message { text, .. } = row {
            assert!(
                while_running
                    .iter()
                    .any(|drawn| drawn.contains(text.as_str())),
                "{text:?} is in the entry and was not painted: {while_running:?}",
            );
        }
    }
}

/// What the panel shows is the operation happening, not a summary of it: each
/// run of a stage is its own row with its own cost and its own note, and two
/// runs that said different things say both, separately. The entry folds the
/// same two runs into `round x2` with the ends of the note joined, which is the
/// right answer to a different question.
#[test]
fn a_running_phase_table_keeps_every_run_and_every_note_apart() {
    let (mut state, id) = adjustable();
    let gate = running(&mut state, id, Operation::BUNDLE_ADJUST);

    let texts = painted(&mut state);
    let rounds = texts
        .iter()
        .filter(|text| text.starts_with("  round"))
        .count();
    assert_eq!(rounds, 2, "the two rounds were not drawn apart: {texts:?}");
    for note in ["  round  median 1.10 px", "  round  median 1.20 px"] {
        assert!(
            texts.iter().any(|text| text == note),
            "{note:?} was not painted on its own row: {texts:?}",
        );
    }
    assert!(
        !texts.iter().any(|text| text.contains(" x2")),
        "the panel folded a stage: {texts:?}",
    );
    assert!(
        !texts.iter().any(|text| text.contains("...")),
        "the panel joined two notes: {texts:?}",
    );

    gate.open();
    state.finish_background_tasks();

    // The entry, of the same two runs, folds them and joins the ends.
    let drawn = ActionLog::drawn_detail(newest(&state));
    assert!(
        drawn
            .iter()
            .any(|row| row.contains("round x2") && row.contains("...")),
        "the entry did not fold the rounds: {drawn:?}",
    );
}

/// A bar only where something underneath reported a number, and the open
/// phase's name beside a spinner where nothing did. Nothing interpolates across
/// a silent stage, because a bar moving at a rate nobody measured makes a
/// promise about the finish.
#[test]
fn the_bar_is_drawn_only_where_a_count_was_reported() {
    let (mut state, id) = adjustable();
    let gate = running(&mut state, id, Operation::BUNDLE_ADJUST);
    let task = state.running_tasks().next().expect("running");
    let live = task.collector.live();
    assert_eq!(
        super::panel::bar(&task.collector, &live),
        super::panel::Bar::Measured {
            fraction: 1.0,
            count: Some("round 2/2".to_string()),
        },
    );
    let texts = painted(&mut state);
    assert!(texts.iter().any(|text| text == "round 2/2"), "{texts:?}");
    gate.open();
    state.finish_background_tasks();

    let gate = running(&mut state, id, NOT_CANCELLABLE);
    let task = state.running_tasks().next().expect("running");
    let live = task.collector.live();
    assert_eq!(
        super::panel::bar(&task.collector, &live),
        super::panel::Bar::Spinner {
            phase: Some("gather arrays"),
        },
    );
    let texts = painted(&mut state);
    assert!(
        texts.iter().any(|text| text == "gather arrays"),
        "{texts:?}"
    );
    // The open phase is marked, so a reader can tell the stage that is running
    // from the ones that are over.
    assert!(
        texts.iter().any(|text| text == super::panel::OPEN_MARK),
        "the open phase is not marked: {texts:?}"
    );
    gate.open();
    state.finish_background_tasks();
}

/// Cancel is there either way, and says why when it would do nothing. Hiding it
/// leaves the reader wondering whether they missed it.
#[test]
fn cancel_is_live_for_one_operation_and_explained_for_the_other() {
    for operation in [Operation::BUNDLE_ADJUST, NOT_CANCELLABLE] {
        let (mut state, id) = adjustable();
        let gate = running(&mut state, id, operation);
        let task = state.running_tasks().next().expect("running").id;
        let refusal = state.cancel_refusal(task);
        assert_eq!(refusal.is_some(), !operation.cancellable, "{refusal:?}");

        let ctx = egui::Context::default();
        // A tooltip waits out `tooltip_delay` before it shows, which a headless
        // frame has no wall clock to pass; what is under test is which sentence
        // the button carries, not how long egui makes a reader wait for it.
        ctx.all_styles_mut(|style| {
            style.interaction.tooltip_delay = 0.0;
            style.interaction.tooltip_grace_time = 0.0;
        });
        let first = painted_at(&ctx, &mut state, input());
        let at = first
            .iter()
            .find(|(text, _)| text == super::panel::CANCEL)
            .unwrap_or_else(|| panic!("no Cancel button: {first:?}"))
            .1;
        // Into the middle of the glyphs rather than at their corner, which is
        // the one point of the button's rect a rounding could put outside it.
        let over = at + egui::vec2(8.0, 4.0);
        let mut hovering = input();
        hovering.events = vec![egui::Event::PointerMoved(over)];
        painted_at(&ctx, &mut state, hovering);
        // A second frame with the pointer where it was: egui wants it to have
        // stopped moving before it puts a tooltip up, and the frame that
        // carries the move is the frame it was still moving in.
        let hovered: Vec<String> = painted_at(&ctx, &mut state, input())
            .into_iter()
            .map(|(text, _)| text)
            .collect();
        // Which sentence comes up is also what says whether the button is live:
        // egui shows a disabled hover text only for a disabled widget and an
        // ordinary one only for an enabled widget, so reading the refusal off
        // the tooltip is reading the disablement off it too.
        let expected = refusal.unwrap_or_else(|| "Ask the operation to stop.".to_string());
        assert!(
            hovered.contains(&expected),
            "{expected:?} was not the tooltip: {hovered:?}",
        );

        gate.open();
        state.finish_background_tasks();
    }
}

/// A job that opens more phases than a short panel can show, and waits at the
/// gate with the last of them named.
fn many_phases_job(gate: mpsc::Receiver<()>, said: mpsc::Sender<()>) -> Job {
    Box::new(move |progress| {
        for name in PHASES {
            drop(progress.phase(name));
        }
        said.send(()).expect("the test is listening");
        let _ = gate.recv();
        Finished::Failed("the fake worker produced nothing".to_string())
    })
}

/// Enough stages to overflow the panel, the last of which is what a reader
/// watching a running operation needs to see.
const PHASES: [&str; 12] = [
    "gather arrays",
    "residuals before",
    "solve",
    "round",
    "linearise",
    "normal equations",
    "damping ladder",
    "write back",
    "row map",
    "reindex",
    "push version",
    "settle",
];

/// The table follows its tail, because the stage that is running is the newest
/// row and the stages that finished first are the ones that fit.
///
/// Found by watching a 102 second solve: the panel showed `gather arrays`,
/// which cost 2 ms, for the whole of it, while the stage actually spending the
/// time was a scroll below the fold.
#[test]
fn a_running_phase_table_shows_the_newest_stage_not_the_first() {
    let (mut state, id) = adjustable();
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            many_phases_job(gate.held(), said),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_tasks();

    // Short enough that the table cannot hold all twelve, which is the shape of
    // the panel at the bottom of the left column.
    for height in [170.0, 190.0] {
        let mut short = input();
        short.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(260.0, height),
        ));
        let ctx = egui::Context::default();
        // The first frame lays the table out; the scroll offset it computes
        // from that is applied to the next one.
        crate::test_support::painted_texts(&ctx, short.clone(), |ui| {
            super::panel::show(ui, &mut state)
        });
        let texts = crate::test_support::painted_texts(&ctx, short, |ui| {
            super::panel::show(ui, &mut state)
        });
        let last = PHASES.last().expect("a stage");
        assert!(
            texts.iter().any(|text| text == last),
            "{last:?} was below the fold at {height} px: {texts:?}",
        );
        assert!(
            !texts.iter().any(|text| text == PHASES[0]),
            "the table did not scroll at all at {height} px: {texts:?}",
        );
    }

    gate.open();
    state.finish_background_tasks();
}

/// With a task in each queue, the panel at the height it has at the bottom
/// of the left column still shows the chosen task's newest stage: the queue
/// lines are one line each, so the phase table keeps the room.
#[test]
fn with_both_queues_running_the_newest_stage_is_still_shown() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            many_phases_job(gate.held(), said),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    let fast = running(&mut state, other, Operation::BENCH_FIT);
    state.poll_background_tasks();

    for height in [170.0, 190.0] {
        let mut short = input();
        short.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(260.0, height),
        ));
        let ctx = egui::Context::default();
        crate::test_support::painted_texts(&ctx, short.clone(), |ui| {
            super::panel::show(ui, &mut state)
        });
        let texts = crate::test_support::painted_texts(&ctx, short, |ui| {
            super::panel::show(ui, &mut state)
        });
        for line in ["Slow", "Bundle adjust", "Fast", "Fit track"] {
            assert!(
                texts.iter().any(|text| text == line),
                "{line:?} at {height}: {texts:?}"
            );
        }
        let last = PHASES.last().expect("a stage");
        assert!(
            texts.iter().any(|text| text == last),
            "{last:?} was below the fold at {height} px: {texts:?}",
        );
    }

    fast.open();
    gate.open();
    state.finish_background_tasks();
}

/// With a task running and one waiting in each queue, the panel at the height
/// it has at the bottom of the left column still shows the chosen task's
/// newest stage: each queue's waiting tasks take one line under its own,
/// however many there are, with `+{k}` for the ones behind the next.
#[test]
fn waiting_tasks_take_one_line_a_queue_and_leave_the_newest_stage_shown() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            many_phases_job(gate.held(), said),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    let fast = running(&mut state, other, Operation::BENCH_FIT);
    let mut held = Vec::new();
    let mut queue = |state: &mut AppState, operation: Operation| {
        let node = third_node(state);
        let mut gate = Gate::new();
        state
            .start_background_task(operation, node, waiting_job(gate.held()))
            .expect("a busy queue queues the task");
        held.push(gate);
    };
    queue(&mut state, Operation::RETRIANGULATE_ALL_POINTS);
    queue(&mut state, Operation::BENCH_FIT_NORMAL);
    state.poll_background_tasks();

    let frame = |state: &mut AppState, height: f32| {
        let mut short = input();
        short.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(260.0, height),
        ));
        let ctx = egui::Context::default();
        crate::test_support::painted_texts(&ctx, short.clone(), |ui| super::panel::show(ui, state));
        crate::test_support::painted_texts(&ctx, short, |ui| super::panel::show(ui, state))
    };
    for height in [170.0, 190.0] {
        let texts = frame(&mut state, height);
        for line in [
            "Slow",
            "Bundle adjust",
            "Retriangulate all points",
            "Fast",
            "Fit track",
            "Fit normal",
        ] {
            assert!(
                texts.iter().any(|text| text == line),
                "{line:?} at {height}: {texts:?}"
            );
        }
        assert_eq!(
            texts
                .iter()
                .filter(|text| *text == super::panel::WAITING)
                .count(),
            2,
            "a waiting line a queue at {height}: {texts:?}"
        );
        // A Cancel on each running line and on each waiting line.
        assert_eq!(
            texts
                .iter()
                .filter(|text| *text == super::panel::CANCEL)
                .count(),
            4,
            "{texts:?}"
        );
        let last = PHASES.last().expect("a stage");
        assert!(
            texts.iter().any(|text| text == last),
            "{last:?} was below the fold at {height} px: {texts:?}",
        );
    }

    // Two more behind the slow queue's next: still one line, counting them.
    queue(&mut state, Operation::REFIT_ALL_POINTS);
    queue(&mut state, Operation::PRUNE_COVERED_OBSERVATIONS);
    let texts = frame(&mut state, 170.0);
    assert!(texts.iter().any(|text| text == "+2"), "{texts:?}");
    assert!(
        !texts.iter().any(|text| text == "Refit all points"),
        "a task behind the next has a line of its own: {texts:?}"
    );
    assert_eq!(
        texts
            .iter()
            .filter(|text| *text == super::panel::WAITING)
            .count(),
        2,
        "{texts:?}"
    );
    let last = PHASES.last().expect("a stage");
    assert!(
        texts.iter().any(|text| text == last),
        "{last:?} was below the fold with four waiting: {texts:?}",
    );

    for gate in &held {
        gate.open();
    }
    fast.open();
    gate.open();
    state.finish_background_tasks();
}

/// Where a painted string actually lies: left edge, right edge, and the top
/// of its line, since two things on different rows cannot collide however
/// their columns overlap.
///
/// Not its position: a right-aligned galley reports its *anchor*, with the
/// glyphs running back from it, so a cost pinned to the right edge of a 260px
/// panel is drawn at `pos.x = 260`. The span is what says whether two things
/// overlap, and overlapping is the failure this is for.
fn painted_spans(state: &mut AppState, width: f32) -> Vec<(String, f32, f32, f32)> {
    fn walk(shape: &egui::Shape, out: &mut Vec<(String, f32, f32, f32)>) {
        match shape {
            egui::Shape::Text(text) => out.push((
                text.galley.text().to_owned(),
                text.pos.x + text.galley.rect.min.x,
                text.pos.x + text.galley.rect.max.x,
                text.pos.y,
            )),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
            _ => {}
        }
    }
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width, 400.0),
        )),
        ..Default::default()
    };
    let mut output = ctx.run_ui(input, |ui| super::panel::show(ui, state));
    output.textures_delta.clear();
    let mut out = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut out);
    }
    out
}

/// The cost survives a long label, because it is the column a reader came back
/// to an idle panel for.
///
/// Found by looking at the panel with `dino_dog_toy-embedded` in it: the label
/// claimed the whole row and ran under the number. The name is given what is
/// left after the cost rather than allowed to take it, which is the rule the
/// phase rows one function below already follow.
#[test]
fn an_idle_panel_keeps_the_cost_clear_of_a_long_label() {
    let (mut state, id) = adjustable();
    running(&mut state, id, Operation::BUNDLE_ADJUST).open();
    state.finish_background_tasks();
    let last = state.finished_tasks.back_mut().expect("it finished");
    last.label = "a_reconstruction_with_a_name_nobody_would_shorten".to_string();
    let took = ActionLog::format_took(last.took);

    let width = 260.0;
    let spans = painted_spans(&mut state, width);
    let (_, cost_left, cost_right, cost_row) = *spans
        .iter()
        .find(|(text, _, _, _)| *text == took)
        .unwrap_or_else(|| panic!("{took:?} was not painted: {spans:?}"));
    assert!(
        cost_right <= width,
        "the cost ran to {cost_right} past the {width}px edge: {spans:?}",
    );
    // Nothing sharing its line reaches it. Asserted over the row rather than
    // by finding one text, because what the names read as is the eliding
    // rule's business rather than this test's.
    for (text, _, right, row) in &spans {
        if *text == took || (*row - cost_row).abs() > 1.0 {
            continue;
        }
        assert!(
            *right <= cost_left,
            "{text:?} ran to {right}, under a cost starting at {cost_left}",
        );
    }
    // The node is still there to be read, shortened out of the middle on the
    // queue's line, in the room the cost leaves it.
    spans
        .iter()
        .find(|(text, _, _, _)| text.starts_with("a_rec") && text.ends_with("en"))
        .unwrap_or_else(|| panic!("the node was not drawn: {spans:?}"));
}

/// A running task's line holds a long operation name in the default panel
/// width by cutting it: nothing on the line is drawn over anything else.
#[test]
fn a_running_line_cuts_a_long_name_rather_than_overlapping() {
    let (mut state, id) = adjustable();
    let gate = running(&mut state, id, Operation::PRUNE_COVERED_OBSERVATIONS);
    // The left column's width in a 1280 px window.
    let spans = painted_spans(&mut state, 230.0);
    let row = spans
        .iter()
        .find(|(text, _, _, _)| text == "Slow")
        .unwrap_or_else(|| panic!("no slow line: {spans:?}"))
        .3;
    let mut line: Vec<(f32, f32, &str)> = spans
        .iter()
        .filter(|(text, _, _, y)| (*y - row).abs() <= 2.0 && !text.is_empty())
        .map(|(text, left, right, _)| (*left, *right, text.as_str()))
        .collect();
    line.sort_by(|a, b| a.0.total_cmp(&b.0));
    for pair in line.windows(2) {
        assert!(
            pair[0].1 <= pair[1].0 + 0.5,
            "{:?} runs under {:?}: {line:?}",
            pair[0].2,
            pair[1].2,
        );
    }
    assert!(line.iter().all(|(_, right, _)| *right <= 230.5), "{line:?}");
    gate.open();
    state.finish_background_tasks();
}

/// Run `operation` on `id` to its end with a job that opens one phase and
/// ends as `ended` says.
fn ended_as(state: &mut AppState, id: ReconId, operation: Operation, ended: fn() -> Finished) {
    let task = state
        .start_background_task(
            operation,
            id,
            Box::new(move |progress| {
                drop(progress.phase("solve"));
                ended()
            }),
        )
        .expect("nothing else is running");
    state.finish_background_task(task);
}

/// An idle line says when its task was cancelled or failed, and the title line
/// says it again when that task's phases are shown: greyed, a cancelled task
/// looked exactly like one that had finished, and only the Action Log said
/// otherwise.
#[test]
fn an_idle_line_says_whether_its_task_was_cancelled_or_failed() {
    use super::panel::{CANCELLED, FAILED};
    let (mut state, id) = adjustable();
    let count = |texts: &[String], word: &str| texts.iter().filter(|t| *t == word).count();

    ended_as(&mut state, id, Operation::BUNDLE_ADJUST, || {
        Finished::NoChange("nothing to adjust".to_string())
    });
    let texts = painted(&mut state);
    assert_eq!(
        count(&texts, CANCELLED) + count(&texts, FAILED),
        0,
        "{texts:?}"
    );

    ended_as(&mut state, id, Operation::BUNDLE_ADJUST, || {
        Finished::Cancelled
    });
    assert_eq!(
        state.last_finished_task().map(|task| task.ended()),
        Some(super::Ended::Cancelled)
    );
    let texts = painted(&mut state);
    // Once on the idle line and once on the title line above its phases.
    assert_eq!(count(&texts, CANCELLED), 2, "{texts:?}");
    assert_eq!(count(&texts, FAILED), 0, "{texts:?}");

    ended_as(&mut state, id, Operation::BUNDLE_ADJUST, || {
        Finished::Failed("the kernel refused".to_string())
    });
    let texts = painted(&mut state);
    assert_eq!(count(&texts, FAILED), 2, "{texts:?}");
    assert_eq!(count(&texts, CANCELLED), 0, "{texts:?}");

    // A task taken out of its queue is cancelled too. It has no phases, so it
    // has no title line, and the word is on its queue's idle line alone.
    let mut gate = Gate::new();
    let other = second_node(&mut state);
    state
        .start_background_task(Operation::BUNDLE_ADJUST, other, waiting_job(gate.held()))
        .expect("the slow queue is idle");
    let waiting = state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            id,
            waiting_job(Gate::new().held()),
        )
        .expect("a busy queue queues the task");
    state.cancel_background_task(waiting);
    assert_eq!(
        state.finished_task(waiting).map(|task| task.ended()),
        Some(super::Ended::Cancelled)
    );
    gate.open();
    state.finish_background_tasks();

    // At the default width the word, the cost and the node all fit.
    ended_as(&mut state, id, Operation::BUNDLE_ADJUST, || {
        Finished::Cancelled
    });
    state.finished_tasks.back_mut().expect("it finished").label = "dino".to_string();
    let width = 227.0;
    let spans = painted_spans(&mut state, width);
    let line = line_led_by(&spans, "Slow", width);
    assert!(line.iter().any(|t| t == CANCELLED), "{line:?}");
    assert!(line.iter().any(|t| t == "dino"), "{line:?}");
}

/// Once a cancel is asked for, a running task says it is stopping until it
/// lands: its line shows `stopping…` where its bar was, its title line shows it
/// where the status was, and its Cancel button is disabled with a tooltip
/// saying why.
///
/// Found on dino's bundle adjustment, which reaches its next check of the flag
/// about half a second after the click: until then the line looked exactly as
/// it had before it, and a reader could not tell the click had registered.
#[test]
fn a_task_asked_to_stop_says_it_is_stopping_until_it_lands() {
    use super::panel::{STOPPING, STOPPING_HOVER};
    let (mut state, id) = adjustable();
    let gate = running(&mut state, id, Operation::BUNDLE_ADJUST);
    let task = state.running_tasks().next().expect("running").id;
    assert!(
        !painted(&mut state).iter().any(|t| t == STOPPING),
        "a task nobody asked to stop said it was stopping"
    );

    state.cancel_background_task(task);
    assert!(state.running_task(task).expect("still running").stopping());
    let texts = painted(&mut state);
    // On its queue's line and on its title line.
    assert_eq!(
        texts.iter().filter(|t| *t == STOPPING).count(),
        2,
        "{texts:?}"
    );
    // The count is still on the title line; the bar is gone from the queue's.
    assert!(texts.iter().any(|t| t == "round 2/2"), "{texts:?}");

    // Cancel is disabled, and says why on hover.
    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let first = painted_at(&ctx, &mut state, input());
    let at = first
        .iter()
        .find(|(text, _)| text == super::panel::CANCEL)
        .unwrap_or_else(|| panic!("no Cancel button: {first:?}"))
        .1;
    let mut hovering = input();
    hovering.events = vec![egui::Event::PointerMoved(at + egui::vec2(8.0, 4.0))];
    painted_at(&ctx, &mut state, hovering);
    let hovered: Vec<String> = painted_at(&ctx, &mut state, input())
        .into_iter()
        .map(|(text, _)| text)
        .collect();
    assert!(
        hovered.iter().any(|t| t == STOPPING_HOVER),
        "{STOPPING_HOVER:?} was not the tooltip: {hovered:?}"
    );

    gate.open();
    state.finish_background_tasks();
    assert!(
        !painted(&mut state).iter().any(|t| t == STOPPING),
        "a task that landed still said it was stopping"
    );
}

/// The texts painted on the line `lead` is painted on, left to right, after
/// asserting that none of them runs under the next or past `width`.
fn line_led_by(spans: &[(String, f32, f32, f32)], lead: &str, width: f32) -> Vec<String> {
    let row = spans
        .iter()
        .find(|(text, _, _, _)| text == lead)
        .unwrap_or_else(|| panic!("no {lead:?} line: {spans:?}"))
        .3;
    let mut line: Vec<(f32, f32, &str)> = spans
        .iter()
        .filter(|(text, _, _, y)| (*y - row).abs() <= 2.0 && !text.is_empty())
        .map(|(text, left, right, _)| (*left, *right, text.as_str()))
        .collect();
    line.sort_by(|a, b| a.0.total_cmp(&b.0));
    for pair in line.windows(2) {
        assert!(
            pair[0].1 <= pair[1].0 + 0.5,
            "{:?} runs under {:?}: {line:?}",
            pair[0].2,
            pair[1].2,
        );
    }
    assert!(
        line.iter().all(|(_, right, _)| *right <= width + 0.5),
        "{line:?}"
    );
    line.into_iter()
        .map(|(_, _, text)| text.to_string())
        .collect()
}

/// At the panel's default width the node stays on a running line and on a
/// waiting line, however long the operation's name: it is the node that says
/// which reconstruction is busy, and the operation's name is cut first.
///
/// Found on a 227 px panel, where a running line read `Slow Bundle adjust`,
/// the bar, the elapsed and Cancel with no `dino`, and a waiting line
/// `waiting Retriangulate all points` and Cancel with no `bull`.
#[test]
fn the_node_keeps_its_place_on_running_and_waiting_lines() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut fast = Gate::new();
    state
        .start_background_task(Operation::BENCH_FIT, other, waiting_job(fast.held()))
        .expect("the fast queue is idle");
    let slow = running(&mut state, id, Operation::BUNDLE_ADJUST);
    let mut waiting = Gate::new();
    let node = third_node(&mut state);
    state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            node,
            waiting_job(waiting.held()),
        )
        .expect("a busy queue queues the task");
    state.poll_background_tasks();
    state.slow_queue.running.as_mut().expect("running").label = "dino".to_string();
    state.slow_queue.waiting[0].label = "bull".to_string();
    state.fast_queue.running.as_mut().expect("running").label =
        "20260628-01-solve-seattle_backyard_1-26-embedded".to_string();

    let width = 227.0;
    let spans = painted_spans(&mut state, width);
    let running_line = line_led_by(&spans, "Slow", width);
    assert!(running_line.iter().any(|t| t == "dino"), "{running_line:?}");
    assert!(
        running_line.iter().any(|t| t.starts_with("Bundle")),
        "{running_line:?}"
    );
    let waiting_line = line_led_by(&spans, super::panel::WAITING, width);
    assert!(waiting_line.iter().any(|t| t == "bull"), "{waiting_line:?}");
    assert!(
        waiting_line.iter().any(|t| t.starts_with("Retri")),
        "{waiting_line:?}"
    );
    // A long node name beside an operation name that fits whole is cut out of
    // the middle, so both its ends stay, to what the operation leaves.
    let fast_line = line_led_by(&spans, "Fast", width);
    assert!(fast_line.iter().any(|t| t == "Fit track"), "{fast_line:?}");
    assert!(
        fast_line
            .iter()
            .any(|t| t.starts_with("20") && t.contains('\u{2026}') && t.ends_with('d')),
        "{fast_line:?}"
    );

    waiting.open();
    slow.open();
    fast.open();
    state.finish_background_tasks();
}

/// The phase list's title line keeps the count on screen however long the
/// node's name is, and cuts the title to the room left, with the whole of it on
/// hover: the queue line has cut the same name to almost nothing, so this line
/// is where the run is named.
#[test]
fn a_long_node_name_leaves_the_count_on_the_title_line() {
    let (mut state, id) = adjustable();
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    let held = gate.held();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |progress| {
                let solve = progress.phase("solve");
                solve.count(1, Some(3), "round");
                progress.set_status_message(format_args!("refining member 4"));
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                Finished::Failed("the fake worker produced nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_tasks();
    let whole = "20260628-01-solve-seattle_backyard_1-26-embedded";
    state.slow_queue.running.as_mut().expect("running").label = whole.to_string();
    let title = super::panel::phases_title("Bundle adjust", whole);

    let width = 230.0;
    let spans = painted_spans(&mut state, width);
    let (_, count_left, _, row) = *spans
        .iter()
        .find(|(text, _, _, _)| text == "round 1/3")
        .unwrap_or_else(|| panic!("the count was not painted: {spans:?}"));
    // Nothing leaves the panel, and nothing on the title line runs under
    // anything else.
    for (text, _, right, _) in &spans {
        assert!(*right <= width + 0.5, "{text:?} ran to {right}: {spans:?}");
    }
    let mut line: Vec<(f32, f32, &str)> = spans
        .iter()
        .filter(|(text, _, _, y)| (*y - row).abs() <= 2.0 && !text.is_empty())
        .map(|(text, left, right, _)| (*left, *right, text.as_str()))
        .collect();
    line.sort_by(|a, b| a.0.total_cmp(&b.0));
    for pair in line.windows(2) {
        assert!(
            pair[0].1 <= pair[1].0 + 0.5,
            "{:?} runs under {:?}: {line:?}",
            pair[0].2,
            pair[1].2,
        );
    }
    // Where the whole title does not fit, it is the node's name, cut out of
    // the middle so its start and its end both stay, left of the count.
    let (title_left, _, _) = *line
        .iter()
        .find(|(_, _, text)| text.starts_with("2026") && text.ends_with("dded") && *text != whole)
        .unwrap_or_else(|| panic!("the cut node was not the title: {line:?}"));
    assert!(title_left < count_left);
    // The status keeps the rest of the line, and shows.
    assert!(
        line.iter().any(|(_, _, text)| text.starts_with("refin")),
        "{line:?}"
    );

    // Its whole is one hover away.
    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let narrow = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width, 400.0),
        )),
        ..Default::default()
    };
    let first = painted_at(&ctx, &mut state, narrow());
    let at = first
        .iter()
        .find(|(text, _)| text.starts_with("2026") && text.ends_with("dded") && *text != whole)
        .unwrap_or_else(|| panic!("the title was not painted: {first:?}"))
        .1;
    let mut hovering = narrow();
    hovering.events = vec![egui::Event::PointerMoved(at + egui::vec2(8.0, 4.0))];
    painted_at(&ctx, &mut state, hovering);
    let hovered: Vec<String> = painted_at(&ctx, &mut state, narrow())
        .into_iter()
        .map(|(text, _)| text)
        .collect();
    assert!(
        hovered.contains(&title),
        "{title:?} is not on hover: {hovered:?}"
    );

    gate.open();
    state.finish_background_tasks();
}

/// A scroll wheel event over the phase table.
///
/// Positive `delta.y` moves the content down, which is what a reader does to
/// look back at what has already happened. `phase` is what
/// `platform::gesture_scroll_events` sends for a touchpad pan, and a mouse
/// wheel is read the same way.
fn wheel(unit: egui::MouseWheelUnit, amount: f32) -> egui::Event {
    egui::Event::MouseWheel {
        unit,
        delta: egui::vec2(0.0, amount),
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::NONE,
    }
}

/// One frame with the pointer inside the phase table, so a wheel event has
/// somewhere to land.
fn over_the_table(events: Vec<egui::Event>) -> egui::RawInput {
    let mut input = input();
    let mut all = vec![egui::Event::PointerMoved(egui::pos2(130.0, 320.0))];
    all.extend(events);
    input.events = all;
    input
}

/// The whole log is kept, however long it runs: nothing in the panel caps the
/// rows, and every stage a long operation reported is there to scroll back to.
///
/// The Action Log entry caps at `DETAIL_EVENTS` and so does the wire
/// (`specs/gui/mcp-server.md`), and the panel is the one view of the three that
/// does not, because it is where a reader goes to watch the whole thing.
#[test]
fn the_phase_table_keeps_every_row_of_a_long_operation() {
    let (mut state, id) = adjustable();
    let reported = ActionLog::DETAIL_EVENTS * 3 + 7;
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    let held = gate.held();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |progress| {
                for i in 0..reported {
                    progress.message(
                        sfmtool_core::progress::Level::Info,
                        format_args!("event {i}"),
                    );
                }
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                Finished::Failed("the fake worker produced nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_tasks();

    let live = state
        .running_tasks()
        .next()
        .expect("running")
        .collector
        .live();
    assert_eq!(
        live.rows.len(),
        reported,
        "the panel's own rows were capped"
    );

    gate.open();
    state.finish_background_tasks();

    // And the idle form keeps them too, where the entry it wrote did not.
    let last = state.last_finished_task().expect("it finished");
    assert_eq!(
        last.detail.len(),
        reported,
        "the idle panel's rows were capped"
    );
    assert_eq!(
        newest(&state).detail.len(),
        ActionLog::DETAIL_EVENTS + 1,
        "the entry was not capped, so the two are not different views after all",
    );
}

/// The table follows the tail while it is at the tail, and holds still the
/// moment the reader scrolls up, so an early stage can be read while the
/// operation keeps going.
///
/// Run for both wheel units. A Windows precision touchpad never reaches a
/// `ScrollArea` as a wheel of its own: DirectManipulation claims the contacts
/// for the whole window, so `platform::gesture_scroll_events` feeds the pan
/// back in as a **`Point`**-unit `MouseWheel`, where a mouse sends `Line`. This
/// panel has no gesture handling of its own and is scrolled entirely by that
/// path, so the unit is the only difference and the behaviour must not depend
/// on it.
#[test]
fn scrolling_up_holds_the_table_while_the_operation_keeps_reporting() {
    for unit in [egui::MouseWheelUnit::Line, egui::MouseWheelUnit::Point] {
        // A line is a row; a point is a pixel, so the same distance is roughly
        // a row height more of them.
        let up = match unit {
            egui::MouseWheelUnit::Point => 240.0,
            _ => 16.0,
        };
        let (mut state, id) = adjustable();
        let mut gate = Gate::new();
        let (said, heard) = mpsc::channel();
        let held = gate.held();
        let (more, report_more) = mpsc::channel::<usize>();
        state
            .start_background_task(
                Operation::BUNDLE_ADJUST,
                id,
                Box::new(move |progress| {
                    for i in 0..40usize {
                        progress.message(
                            sfmtool_core::progress::Level::Info,
                            format_args!("event {i}"),
                        );
                    }
                    said.send(()).expect("the test is listening");
                    // More only when the test asks, so what the panel drew is
                    // decided by the test rather than by the scheduler.
                    if let Ok(n) = report_more.recv() {
                        for i in 0..n {
                            progress.message(
                                sfmtool_core::progress::Level::Info,
                                format_args!("later {i}"),
                            );
                        }
                    }
                    let _ = held.recv();
                    Finished::Failed("the fake worker produced nothing".to_string())
                }),
            )
            .expect("nothing else is running");
        heard.recv().expect("the worker reported");
        state.poll_background_tasks();

        let ctx = egui::Context::default();
        let frame = |state: &mut AppState, input: egui::RawInput| {
            crate::test_support::painted_texts(&ctx, input, |ui| super::panel::show(ui, state))
        };
        let shows = |rows: &[String], needle: &str| rows.iter().any(|row| row.ends_with(needle));

        // At the tail: the newest is on screen and the oldest is not.
        frame(&mut state, input());
        let tail = frame(&mut state, input());
        assert!(shows(&tail, "event 39"), "{unit:?}: {tail:?}");
        assert!(!shows(&tail, "event 0"), "{unit:?}: {tail:?}");

        // Scrolled up. egui smooths a wheel over several frames, so the scroll
        // is drawn out rather than read on the frame that carried it.
        frame(&mut state, over_the_table(vec![wheel(unit, up)]));
        for _ in 0..6 {
            frame(&mut state, input());
        }
        let looking_back = frame(&mut state, input());
        assert!(
            !shows(&looking_back, "event 39"),
            "{unit:?}: the wheel did not scroll the table: {looking_back:?}",
        );
        // Whatever it landed on is what must still be there afterwards.
        let anchor = looking_back
            .iter()
            .find(|row| row.contains("event "))
            .expect("some event is on screen")
            .clone();

        // Twenty more arrive, and the view does not move under the reader.
        more.send(20).expect("the worker is waiting");
        frame(&mut state, input());
        let after = frame(&mut state, input());
        assert!(
            shows(&after, anchor.trim_start_matches("• ")),
            "{unit:?}: the table jumped out from under the reader: {after:?}",
        );
        assert!(
            !shows(&after, "later 19"),
            "{unit:?}: the table followed the tail while the reader was scrolled up: {after:?}",
        );

        gate.open();
        state.finish_background_tasks();
    }
}

/// Start a task that reports forty messages, `event 0` to `event 39`, and
/// waits at the gate: more rows than a short panel holds.
fn forty_events(state: &mut AppState, id: ReconId) -> Gate {
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    let held = gate.held();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |progress| {
                for i in 0..40usize {
                    progress.message(
                        sfmtool_core::progress::Level::Info,
                        format_args!("event {i}"),
                    );
                }
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                Finished::Failed("the fake worker produced nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_tasks();
    gate
}

/// A panel 200 px tall, too short for forty rows.
fn short_input() -> egui::RawInput {
    let mut short = input();
    short.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(260.0, 200.0),
    ));
    short
}

/// A task that lands while its table is watched leaves the table where it was,
/// at the stage it ended on, rather than jumping back to its first row.
///
/// Found by watching a bundle adjustment of dino land: the table had followed
/// its tail for the whole solve, and on the frame it landed it showed `gather
/// arrays` and the rows after it, which nobody had scrolled to.
#[test]
fn a_watched_task_that_lands_keeps_its_last_stage_shown() {
    let (mut state, id) = adjustable();
    let gate = forty_events(&mut state, id);
    let ctx = egui::Context::default();
    let frame = |state: &mut AppState| {
        crate::test_support::painted_texts(&ctx, short_input(), |ui| super::panel::show(ui, state))
    };
    let shows = |rows: &[String], needle: &str| rows.iter().any(|row| row.ends_with(needle));

    // Running, it is at the tail.
    frame(&mut state);
    let live = frame(&mut state);
    assert!(shows(&live, "event 39"), "{live:?}");

    gate.open();
    state.finish_background_tasks();

    // Finished, it is still at the tail, on the first frame and after it.
    for _ in 0..2 {
        let idle = frame(&mut state);
        assert!(
            shows(&idle, "event 39"),
            "the finished task jumped away from its last stage: {idle:?}",
        );
        assert!(
            !shows(&idle, "event 0"),
            "the finished task jumped to its first stage: {idle:?}",
        );
    }
}

/// A finished task the panel never drew while it ran is read from the top,
/// where it started: there is no tail a reader was following, and the first
/// stage is where its account begins.
#[test]
fn an_unwatched_finished_task_opens_at_its_first_stage() {
    let (mut state, id) = adjustable();
    forty_events(&mut state, id).open();
    state.finish_background_tasks();

    let ctx = egui::Context::default();
    let frame = |state: &mut AppState| {
        crate::test_support::painted_texts(&ctx, short_input(), |ui| super::panel::show(ui, state))
    };
    let shows = |rows: &[String], needle: &str| rows.iter().any(|row| row.ends_with(needle));
    frame(&mut state);
    let idle = frame(&mut state);
    assert!(
        shows(&idle, "event 0"),
        "the finished task did not open at its first stage: {idle:?}",
    );
    assert!(
        !shows(&idle, "event 39"),
        "the finished task opened at its last stage: {idle:?}",
    );
}

/// A row too wide for its column says the whole of itself once, not twice.
///
/// `egui::Label` already puts an elided text in a tooltip, in the label's own
/// font, and only when it is actually elided. A second `on_hover_text` of the
/// same string stacks a proportional-font copy over that one: two tooltips,
/// same words, different typefaces.
#[test]
fn a_truncated_row_shows_one_tooltip_rather_than_two() {
    let (mut state, id) = adjustable();
    let mut gate = Gate::new();
    let (said, heard) = mpsc::channel();
    let held = gate.held();
    // Wider than any column this panel has, so the row is certainly elided.
    let long = "median 0.171 px over 32205 observations, and a good deal more \
               besides so that nothing can fit it";
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |progress| {
                progress.message(sfmtool_core::progress::Level::Info, format_args!("{long}"));
                said.send(()).expect("the test is listening");
                let _ = held.recv();
                Finished::Failed("the fake worker produced nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background_tasks();

    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    // Where the row landed, so the hover is over it rather than near it.
    let first = painted_at(&ctx, &mut state, input());
    let at = first
        .iter()
        .find(|(text, _)| text.contains("median 0.171"))
        .unwrap_or_else(|| panic!("the row was not painted: {first:?}"))
        .1;
    let over = at + egui::vec2(8.0, 4.0);
    let mut hovering = input();
    hovering.events = vec![egui::Event::PointerMoved(over)];
    painted_at(&ctx, &mut state, hovering);
    // A second frame with the pointer at rest, which is when egui puts a
    // tooltip up.
    let hovered = painted_at(&ctx, &mut state, input());

    // The row's own galley carries the whole string however few glyphs it drew,
    // so the row counts once and each tooltip counts once more.
    let copies = hovered
        .iter()
        .filter(|(text, _)| text.contains("median 0.171"))
        .count();
    assert_eq!(
        copies, 2,
        "the row and one tooltip make two; {copies} means a second tooltip: {hovered:?}",
    );

    gate.open();
    state.finish_background_tasks();
}

/// The node's whole name is one hover away, once.
///
/// The cut there is ours, out of the middle, so `Label` is handed a string that
/// already fits: it would tooltip nothing, and removing the tooltip we add
/// would leave the full name unreadable. Its own elision tooltip is off and
/// ours is the only one.
#[test]
fn the_idle_node_says_its_whole_name_on_hover_once() {
    let (mut state, id) = adjustable();
    running(&mut state, id, Operation::BUNDLE_ADJUST).open();
    state.finish_background_tasks();
    let whole = "a_reconstruction_with_a_name_nobody_would_shorten".to_string();
    state
        .finished_tasks
        .back_mut()
        .expect("it finished")
        .label
        .clone_from(&whole);

    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let first = painted_at(&ctx, &mut state, input());
    // The idle line's, which is painted before the title line's.
    let at = first
        .iter()
        .find(|(text, _)| text.starts_with("a_re"))
        .unwrap_or_else(|| panic!("the node was not painted: {first:?}"))
        .1;
    let over = at + egui::vec2(8.0, 4.0);
    let mut hovering = input();
    hovering.events = vec![egui::Event::PointerMoved(over)];
    painted_at(&ctx, &mut state, hovering);
    let hovered: Vec<String> = painted_at(&ctx, &mut state, input())
        .into_iter()
        .map(|(text, _)| text)
        .collect();

    assert!(
        hovered.contains(&whole),
        "the whole name is not reachable on hover: {hovered:?}",
    );
    assert_eq!(
        hovered.iter().filter(|text| **text == whole).count(),
        1,
        "more than one tooltip carries it: {hovered:?}",
    );
}

// -- Quiet tasks -----------------------------------------------------------

/// A quiet job that waits at `gate` and then says on `said` whether it was
/// asked to stop, or says `dropped` when the queue drops it unrun: the source's
/// channel, as the live evaluation's is.
fn quiet_job(gate: mpsc::Receiver<()>, said: mpsc::Sender<&'static str>) -> QuietJob {
    let dropped = said.clone();
    QuietJob::new(
        move |progress| {
            let _ = gate.recv();
            let _ = said.send(if progress.is_cancelled() {
                "cancelled"
            } else {
                "ran"
            });
        },
        move || {
            let _ = dropped.send("dropped");
        },
    )
}

/// Queue a fake quiet task of the live evaluation's source on `node`.
fn start_quiet(
    state: &mut AppState,
    node: ReconId,
    gate: mpsc::Receiver<()>,
    said: mpsc::Sender<&'static str>,
) -> u64 {
    state
        .start_quiet_task(
            Operation::LIVE_EVALUATION,
            QuietSource::LiveEvaluation,
            QuietRank::BenchItem,
            node,
            "fake subject".to_string(),
            quiet_job(gate, said),
        )
        .expect("a worker thread")
}

/// The ids waiting in the fast queue, quiet ones included, in the order they
/// will start, each with whether it is quiet.
fn fast_waiting(state: &AppState) -> Vec<(u64, bool)> {
    state
        .queue(Queue::Fast)
        .waiting
        .iter()
        .map(|task| (task.id, task.operation.quiet))
        .collect()
}

/// A task someone asks for in the fast queue asks the running quiet task to
/// stop and waits for it; it starts once the quiet task reports that it has
/// stopped, and the quiet task queued again waits behind it.
#[test]
fn a_running_quiet_task_gives_way_to_an_asked_for_task_and_runs_after_it() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let (said, heard) = mpsc::channel();
    let mut quiet_gate = Gate::new();
    let quiet = start_quiet(&mut state, other, quiet_gate.held(), said.clone());
    assert!(state.quiet_task_running(quiet), "an idle queue ran nothing");
    assert!(!state.quiet_task_cancelled(quiet));

    let mut fit_gate = Gate::new();
    let fit = state
        .start_background_task(Operation::BENCH_FIT, id, waiting_job(fit_gate.held()))
        .expect("a busy queue queues the task");
    assert!(
        state.quiet_task_cancelled(quiet),
        "the asked-for task did not ask the quiet one to stop"
    );
    assert!(
        matches!(state.task(fit), Some(TaskState::Waiting(_))),
        "the asked-for task started beside the quiet one"
    );

    quiet_gate.open();
    assert_eq!(heard.recv().expect("the quiet job answered"), "cancelled");
    state.land_quiet_tasks();
    assert!(
        state.running_task(fit).is_some(),
        "the asked-for task did not start once the quiet one stopped"
    );

    // Its source queues it again, and it waits behind the asked-for task.
    let mut again_gate = Gate::new();
    let again = start_quiet(&mut state, other, again_gate.held(), said);
    assert_eq!(fast_waiting(&state), [(again, true)]);
    fit_gate.open();
    state.finish_background_task(fit);
    assert!(
        state.quiet_task_running(again),
        "the quiet task did not run after the asked-for one"
    );
    again_gate.open();
    assert_eq!(heard.recv().expect("the quiet job answered"), "ran");
    state.land_quiet_tasks();
}

/// Waiting tasks someone asked for start ahead of a waiting quiet one, and a
/// source's new request replaces the entry it has waiting, which is told it
/// was dropped.
#[test]
fn asked_for_tasks_start_ahead_of_quiet_ones_and_a_source_keeps_one_entry() {
    let (mut state, id) = adjustable();
    let b = second_node(&mut state);
    let c = third_node(&mut state);
    let mut first_gate = Gate::new();
    let first = state
        .start_background_task(Operation::BENCH_FIT, id, waiting_job(first_gate.held()))
        .expect("an idle queue");

    let (said, heard) = mpsc::channel();
    let mut replaced_gate = Gate::new();
    let replaced = start_quiet(&mut state, b, replaced_gate.held(), said.clone());
    assert_eq!(fast_waiting(&state), [(replaced, true)]);
    let mut quiet_gate = Gate::new();
    let quiet = start_quiet(&mut state, b, quiet_gate.held(), said);
    assert_eq!(
        fast_waiting(&state),
        [(quiet, true)],
        "the source has two entries waiting"
    );
    assert_eq!(
        heard.try_recv(),
        Ok("dropped"),
        "the replaced entry did not tell its source"
    );

    let mut second_gate = Gate::new();
    let second = state
        .start_background_task(
            Operation::BENCH_FIT_NORMAL,
            c,
            waiting_job(second_gate.held()),
        )
        .expect("a busy queue queues the task");
    assert_eq!(
        fast_waiting(&state),
        [(second, false), (quiet, true)],
        "the asked-for task waits behind the quiet one"
    );

    first_gate.open();
    state.finish_background_task(first);
    assert!(state.running_task(second).is_some());
    second_gate.open();
    state.finish_background_task(second);
    assert!(state.quiet_task_running(quiet));
    quiet_gate.open();
    assert_eq!(heard.recv().expect("the quiet job answered"), "ran");
    state.land_quiet_tasks();
    drop(replaced_gate);
}

/// A waiting quiet task whose node has become locked since it was queued is
/// dropped when it would start, and its source is told so; the queue is then
/// idle rather than running work whose answer could not land.
#[test]
fn a_waiting_quiet_task_whose_node_became_locked_is_dropped_at_start() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut fit_gate = Gate::new();
    let fit = state
        .start_background_task(Operation::BENCH_FIT, id, waiting_job(fit_gate.held()))
        .expect("an idle queue");
    let (said, heard) = mpsc::channel();
    let mut quiet_gate = Gate::new();
    start_quiet(&mut state, other, quiet_gate.held(), said);
    let mut solve_gate = Gate::new();
    let solve = state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            other,
            waiting_job(solve_gate.held()),
        )
        .expect("the slow queue is idle");

    fit_gate.open();
    state.finish_background_task(fit);
    assert_eq!(heard.try_recv(), Ok("dropped"));
    assert!(
        state.queue(Queue::Fast).running.is_none() && fast_waiting(&state).is_empty(),
        "the quiet task started on a locked node"
    );

    solve_gate.open();
    state.finish_background_task(solve);
    drop(quiet_gate);
}

/// A locking task queued on a node asks a quiet task running on that node to
/// stop, from the other queue and while it only waits there. One on another
/// node does not.
#[test]
fn a_locking_task_queued_on_its_node_cancels_a_running_quiet_task() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let (said, heard) = mpsc::channel();
    let mut quiet_gate = Gate::new();
    let quiet = start_quiet(&mut state, other, quiet_gate.held(), said);

    let mut first_gate = Gate::new();
    state
        .start_background_task(Operation::BUNDLE_ADJUST, id, waiting_job(first_gate.held()))
        .expect("the slow queue is idle");
    assert!(
        !state.quiet_task_cancelled(quiet),
        "a task on another node cancelled it"
    );
    let mut second_gate = Gate::new();
    let second = state
        .start_background_task(
            Operation::RETRIANGULATE_ALL_POINTS,
            other,
            waiting_job(second_gate.held()),
        )
        .expect("a busy queue queues the task");
    assert!(matches!(state.task(second), Some(TaskState::Waiting(_))));
    assert!(
        state.quiet_task_cancelled(quiet),
        "a task queued on its node did not cancel it"
    );

    quiet_gate.open();
    assert_eq!(heard.recv().expect("the quiet job answered"), "cancelled");
    state.land_quiet_tasks();
    first_gate.open();
    second_gate.open();
    state.finish_background_task(second);
}

/// A quiet task writes no Action Log row, leaves no record in the ring, and
/// is not what `running_tasks`, `waiting_tasks`, the lookup by id or the busy
/// notice report. The panel names it on a dim line under its queue's.
#[test]
fn a_quiet_task_writes_no_row_leaves_no_record_and_is_not_reported() {
    let (mut state, id) = adjustable();
    state.action_log.clear();
    let (said, heard) = mpsc::channel();
    let mut gate = Gate::new();
    let quiet = start_quiet(&mut state, id, gate.held(), said);
    assert!(state.quiet_task_running(quiet));
    assert!(state.running_tasks().next().is_none());
    assert!(state.waiting_tasks().next().is_none());
    assert!(state.task(quiet).is_none());
    assert!(super::busy(&state.busy_notice).is_empty());
    assert!(
        state.busy_refusal(id).is_none(),
        "a quiet task locked its node"
    );
    assert_eq!(
        painted(&mut state),
        [
            "Slow",
            "Nothing running",
            "Fast",
            "Nothing running",
            super::panel::QUIET,
            "Live evaluation",
            "fake subject",
        ],
        "no bar, no elapsed and no Cancel on the quiet line"
    );

    gate.open();
    assert_eq!(heard.recv().expect("the quiet job answered"), "ran");
    state.land_quiet_tasks();
    assert_eq!(
        state.action_log.entries().count(),
        0,
        "a quiet task wrote a row"
    );
    assert!(
        state.finished_tasks.is_empty(),
        "a quiet task left a record"
    );
    assert_eq!(
        painted(&mut state),
        ["Slow", "Nothing running", "Fast", "Nothing running"],
        "the fast queue's line shows a quiet task as its last finished"
    );
}

/// With a quiet task running and a task waiting in the fast queue beside a
/// slow task's phases, the panel at the height it has at the bottom of the
/// left column still shows the newest stage: the quiet line is one line.
#[test]
fn a_quiet_line_leaves_the_newest_stage_shown() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut gate = Gate::new();
    let (said_phases, heard_phases) = mpsc::channel();
    state
        .start_background_task(
            Operation::BUNDLE_ADJUST,
            id,
            many_phases_job(gate.held(), said_phases),
        )
        .expect("nothing else is running");
    heard_phases.recv().expect("the worker reported");
    let (said, heard) = mpsc::channel();
    let mut quiet_gate = Gate::new();
    start_quiet(&mut state, other, quiet_gate.held(), said);
    let third = third_node(&mut state);
    let mut fit_gate = Gate::new();
    state
        .start_background_task(Operation::BENCH_FIT, third, waiting_job(fit_gate.held()))
        .expect("a busy queue queues the task");
    state.poll_background_tasks();

    for height in [170.0, 190.0] {
        let mut short = input();
        short.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(260.0, height),
        ));
        let ctx = egui::Context::default();
        crate::test_support::painted_texts(&ctx, short.clone(), |ui| {
            super::panel::show(ui, &mut state)
        });
        let texts = crate::test_support::painted_texts(&ctx, short, |ui| {
            super::panel::show(ui, &mut state)
        });
        for line in [
            "Slow",
            "Bundle adjust",
            "Fast",
            super::panel::QUIET,
            "Live evaluation",
            super::panel::WAITING,
            "Fit track",
        ] {
            assert!(
                texts.iter().any(|text| text == line),
                "{line:?} at {height}: {texts:?}"
            );
        }
        // The slow task's Cancel and the waiting task's; none on the quiet line.
        assert_eq!(
            texts
                .iter()
                .filter(|text| *text == super::panel::CANCEL)
                .count(),
            2,
            "{texts:?}"
        );
        let last = PHASES.last().expect("a stage");
        assert!(
            texts.iter().any(|text| text == last),
            "{last:?} was below the fold at {height} px: {texts:?}",
        );
    }

    quiet_gate.open();
    assert_eq!(heard.recv().expect("the quiet job answered"), "cancelled");
    fit_gate.open();
    gate.open();
    state.land_quiet_tasks();
    state.finish_background_tasks();
}
