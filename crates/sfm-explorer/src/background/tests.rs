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

use crate::action_log::{ActionLog, Actor, Entry};
use crate::progress::Detail;
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
        .start_background(operation, id, job)
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background();
    gate
}

/// An operation whose kernels never ask whether they should stop, for the half
/// of the Cancel rule that is about a button nobody can press.
const NOT_CANCELLABLE: Operation = Operation {
    name: "Fake solve",
    cancellable: false,
};

/// A session that has run nothing says so and says nothing else: one greyed
/// line is a panel a reader can skip, where a blank one is a panel that looks
/// broken.
#[test]
fn an_idle_panel_with_nothing_run_says_only_that() {
    let (mut state, _) = adjustable();
    assert_eq!(painted(&mut state), ["Nothing running"]);
}

/// Idle after an operation: the name, the node and the cost, with the phases
/// behind a toggle that works as the Action Log's does.
#[test]
fn an_idle_panel_shows_the_last_operation_and_expands_its_phases() {
    let (mut state, id) = adjustable();
    running(&mut state, id, Operation::BUNDLE_ADJUST).open();
    state.finish_background();

    let collapsed = painted(&mut state);
    assert!(collapsed.iter().any(|text| text == "Bundle adjust"));
    assert!(collapsed.iter().any(|text| text == "run_a"));
    assert!(collapsed.iter().any(|text| text == "+"), "{collapsed:?}");
    assert!(
        !collapsed.iter().any(|text| text == "solve"),
        "a collapsed panel drew its phases: {collapsed:?}"
    );
    // What it cost, in the Action Log's spelling of a duration.
    let took = ActionLog::format_took(state.last_background.as_ref().expect("it finished").took);
    assert!(
        collapsed.contains(&took),
        "{took:?} is missing from {collapsed:?}"
    );

    state.background_detail_expanded = true;
    let expanded = painted(&mut state);
    assert!(expanded.iter().any(|text| text == "-"), "{expanded:?}");
    assert!(expanded.iter().any(|text| text == "solve"), "{expanded:?}");
    // Unfolded, as the running panel drew them: two runs, two rows.
    assert_eq!(
        expanded
            .iter()
            .filter(|text| text.starts_with("  round"))
            .count(),
        2,
        "{expanded:?}"
    );
}

/// And the toggle is what a click reaches, rather than a flag only a test can
/// move.
#[test]
fn clicking_the_idle_toggle_opens_and_closes_the_phases() {
    let (mut state, id) = adjustable();
    running(&mut state, id, Operation::BUNDLE_ADJUST).open();
    state.finish_background();

    let ctx = egui::Context::default();
    let first = painted_at(&ctx, &mut state, input());
    let at = first
        .iter()
        .find(|(text, _)| text == "+")
        .unwrap_or_else(|| panic!("no toggle: {first:?}"))
        .1;
    let button = |pressed| egui::Event::PointerButton {
        pos: at + egui::vec2(2.0, 4.0),
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let mut click = input();
    click.events = vec![
        egui::Event::PointerMoved(at + egui::vec2(2.0, 4.0)),
        button(true),
        button(false),
    ];
    painted_at(&ctx, &mut state, click);
    assert!(
        state.background_detail_expanded,
        "the click did not open the phases"
    );
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

    let live = state.background().expect("running").collector.live().rows;
    let while_running = painted(&mut state);
    assert!(
        while_running.iter().any(|text| text == "Bundle adjust"),
        "{while_running:?}"
    );
    assert!(while_running.iter().any(|text| text == "run_a"));
    assert!(
        while_running.iter().any(|text| text.ends_with(" elapsed")),
        "the elapsed is missing from {while_running:?}"
    );

    gate.open();
    state.finish_background();

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
    state.finish_background();

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
    let process = state.background().expect("running");
    let live = process.collector.live();
    assert_eq!(
        super::panel::bar(&process.collector, &live),
        super::panel::Bar::Measured {
            fraction: 1.0,
            count: Some("round 2/2".to_string()),
        },
    );
    let texts = painted(&mut state);
    assert!(texts.iter().any(|text| text == "round 2/2"), "{texts:?}");
    gate.open();
    state.finish_background();

    let gate = running(&mut state, id, NOT_CANCELLABLE);
    let process = state.background().expect("running");
    let live = process.collector.live();
    assert_eq!(
        super::panel::bar(&process.collector, &live),
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
    state.finish_background();
}

/// Cancel is there either way, and says why when it would do nothing. Hiding it
/// leaves the reader wondering whether they missed it.
#[test]
fn cancel_is_live_for_one_operation_and_explained_for_the_other() {
    for operation in [Operation::BUNDLE_ADJUST, NOT_CANCELLABLE] {
        let (mut state, id) = adjustable();
        let gate = running(&mut state, id, operation);
        let refusal = state.cancel_refusal();
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
            .find(|(text, _)| text == "Cancel")
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
        state.finish_background();
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
        .start_background(
            Operation::BUNDLE_ADJUST,
            id,
            many_phases_job(gate.held(), said),
        )
        .expect("nothing else is running");
    heard.recv().expect("the worker reported");
    state.poll_background();

    // Short enough that the table cannot hold all twelve, which is the shape of
    // the panel at the bottom of the left column.
    let mut short = input();
    short.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(260.0, 190.0),
    ));
    let ctx = egui::Context::default();
    // The first frame lays the table out; the scroll offset it computes from
    // that is applied to the next one.
    crate::test_support::painted_texts(&ctx, short.clone(), |ui| {
        super::panel::show(ui, &mut state)
    });
    let texts =
        crate::test_support::painted_texts(&ctx, short, |ui| super::panel::show(ui, &mut state));
    let last = PHASES.last().expect("a stage");
    assert!(
        texts.iter().any(|text| text == last),
        "{last:?} was below the fold: {texts:?}",
    );
    assert!(
        !texts.iter().any(|text| text == PHASES[0]),
        "the table did not scroll at all: {texts:?}",
    );

    gate.open();
    state.finish_background();
}

/// Where a painted string actually lies, left edge to right edge.
///
/// Not its position: a right-aligned galley reports its *anchor*, with the
/// glyphs running back from it, so a cost pinned to the right edge of a 260px
/// panel is drawn at `pos.x = 260`. The span is what says whether two things
/// overlap, and overlapping is the failure this is for.
fn painted_spans(state: &mut AppState, width: f32) -> Vec<(String, f32, f32)> {
    fn walk(shape: &egui::Shape, out: &mut Vec<(String, f32, f32)>) {
        match shape {
            egui::Shape::Text(text) => out.push((
                text.galley.text().to_owned(),
                text.pos.x + text.galley.rect.min.x,
                text.pos.x + text.galley.rect.max.x,
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
    state.finish_background();
    let last = state.last_background.as_mut().expect("it finished");
    last.label = "a_reconstruction_with_a_name_nobody_would_shorten".to_string();
    let took = ActionLog::format_took(last.took);
    let label = last.label.clone();

    let width = 260.0;
    let spans = painted_spans(&mut state, width);
    let span = |needle: &str| {
        spans
            .iter()
            .find(|(text, _, _)| text == needle)
            .unwrap_or_else(|| panic!("{needle:?} was not painted: {spans:?}"))
    };
    let (_, cost_left, cost_right) = *span(&took);
    let (_, _, label_right) = *span(&label);
    assert!(
        cost_right <= width,
        "the cost ran to {cost_right} past the {width}px edge",
    );
    assert!(
        label_right <= cost_left,
        "the label ran to {label_right}, under a cost starting at {cost_left}",
    );
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
        .start_background(
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
    state.poll_background();

    let live = state.background().expect("running").collector.live();
    assert_eq!(
        live.rows.len(),
        reported,
        "the panel's own rows were capped"
    );

    gate.open();
    state.finish_background();

    // And the idle form keeps them too, where the entry it wrote did not.
    let last = state.last_background.as_ref().expect("it finished");
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
            .start_background(
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
        state.poll_background();

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
        state.finish_background();
    }
}
