// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The seam between the GUI thread and a worker: what an operation produces,
//! what it writes when it does not, and what the rest of the viewer is allowed
//! to do while it runs.
//!
//! Headless, and deterministic without a sleep anywhere. Two seams make that
//! possible, and both are the mechanism rather than a fake of it:
//!
//! - [`AppState::start_background`] takes the work as a [`Job`], so a test can
//!   hand it one that waits on a gate the test opens. Everything about the
//!   state machine can then be asserted at an instant of the test's choosing,
//!   with a real worker really running.
//! - [`AppState::finish_background`] blocks on the operation's own channel,
//!   applying each report as it arrives, so "wait until it is done" is the
//!   worker's own message rather than a poll loop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use sfmtool_core::progress::Progress;
use sfmtool_core::{BundleAdjustOptions, SfmrReconstruction};

use crate::action_log::{Actor, Entry};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;

use super::{Finished, Job, Operation};

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
    state.finish_background();

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

    state.finish_background();

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
        .start_background(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");

    assert_eq!(
        state.action_log.revision(),
        before,
        "starting an operation wrote {:?}",
        newest(&state).text,
    );
    gate.open();
    state.finish_background();
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
        .start_background(Operation::BUNDLE_ADJUST, busy, waiting_job(gate.held()))
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
        state.save_node_as(busy, std::path::Path::new("/runs/never.sfmr")),
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
    state.finish_background();
}

/// One at a time, viewer-wide.
#[test]
fn a_second_operation_is_refused_while_one_runs() {
    let (mut state, id) = adjustable();
    let other = second_node(&mut state);
    let mut gate = Gate::new();
    state
        .start_background(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");

    let mut second = Gate::new();
    assert_eq!(
        state.start_background(Operation::BUNDLE_ADJUST, other, waiting_job(second.held())),
        Err("Bundle adjust is still running on run_a.".to_string()),
        "a second operation started on another node",
    );

    gate.open();
    state.finish_background();
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

    cancel_before_it_runs(&mut state, id, Operation::BUNDLE_ADJUST);

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
        let (mut state, id) = adjustable();
        cancel_before_it_runs(&mut state, id, operation);
        let entry = newest(&state);
        assert!(
            entry.failed && entry.text.ends_with("cancelled"),
            "{} claims to be cancellable and did not stop: {}",
            operation.name,
            entry.text,
        );
        assert_eq!(
            state.scene[0].history.versions().len(),
            1,
            "{} pushed a version after being cancelled",
            operation.name,
        );
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
fn cancel_before_it_runs(state: &mut AppState, id: ReconId, operation: Operation) {
    let mut gate = Gate::new();
    let held = gate.held();
    let job = real_job(state, id, operation);
    state
        .start_background(
            operation,
            id,
            Box::new(move |progress| {
                let _ = held.recv();
                job(progress)
            }),
        )
        .expect("nothing else is running");
    state.cancel_background();
    gate.open();
    state.finish_background();
}

/// The work `operation` really does, so a test of the declaration tests the
/// kernel rather than a stand-in for it.
///
/// The `match` is exhaustive on purpose: a new [`Operation`] fails here until
/// somebody says how to start it, which is what keeps the walk over
/// [`Operation::ALL`] honest.
fn real_job(state: &AppState, id: ReconId, operation: Operation) -> Job {
    match operation.name {
        "Bundle adjust" => state
            .bundle_adjust_job(id, &BundleAdjustOptions::default())
            .expect("the fixture is well posed"),
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
        .start_background(
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

    let polled = state.poll_background();
    assert!(polled.changed, "a thousand reports moved nothing");
    assert_eq!(polled.installed, None, "nothing has finished");
    // One drain took all of them: the queue is empty, and the collector holds
    // the newest count rather than a thousand of them.
    assert_eq!(
        state.poll_background(),
        super::Polled::default(),
        "reports were left in the queue",
    );
    let count = state
        .background()
        .expect("still running")
        .collector
        .count()
        .expect("the operation counted");
    assert_eq!(
        (count.done, count.total, count.unit),
        (999, Some(1000), "iteration")
    );

    gate.open();
    state.finish_background();
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
        .start_background(
            Operation::BUNDLE_ADJUST,
            id,
            Box::new(|_progress| panic!("a job that fails on the worker")),
        )
        .expect("nothing else is running");

    state.finish_background();

    assert!(state.background().is_none(), "the node is still locked");
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
        .start_background(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");
    let first = state.background().expect("running").id;
    gate.open();
    state.finish_background();

    let last = state.last_background.as_ref().expect("it finished");
    assert_eq!(last.id, first, "the answer is about another operation");

    // The next one is a different operation, and says so.
    let mut gate = Gate::new();
    state
        .start_background(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("the first one is over");
    assert_ne!(state.background().expect("running").id, first);
    gate.open();
    state.finish_background();
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
        .start_background(Operation::BUNDLE_ADJUST, id, job)
        .expect("nothing else is running");
    // And the node is still drawing it while the worker reads it.
    assert!(Arc::ptr_eq(&drawn, &state.scene[0].edited().base));

    state.finish_background();
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
        .start_background(Operation::BUNDLE_ADJUST, id, waiting_job(gate.held()))
        .expect("nothing else is running");

    assert_eq!(
        state.delete_image(ImageRef::new(id, 2)),
        Err("run_a is busy: Bundle adjust is still running.".to_string())
    );

    gate.open();
    state.finish_background();
}
