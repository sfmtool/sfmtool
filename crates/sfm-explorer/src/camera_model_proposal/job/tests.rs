// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The proposal's job: the comparison it sends, the answer it holds while the
//! preview is asked for, which is the version Apply pushes, the generation
//! that keeps an older answer from being used, and the preview ending with
//! every ending of the proposal; and the job as a quiet task in the fast
//! queue, which gives way to asked-for tasks and goes ahead of the live
//! evaluation.

use crate::background::{Operation, Queue, TaskState};
use crate::camera_model_proposal::preview::{FLAG_MOVES, FLAG_STILL};
use crate::camera_model_proposal::tests::{newest, positions, spline_state, versions};
use crate::camera_model_proposal::TriangulationState;
use crate::scene::{CameraRef, PointRef, ReconId};
use crate::state::edits::ProposalApplied;
use crate::state::AppState;

/// `spline_state` with its points nudged off where their rays meet, so the
/// retriangulation moves them, and a proposal open on camera 1 at six
/// coefficients.
fn proposed() -> (AppState, ReconId) {
    let (mut state, id) = spline_state();
    for point in state.scene[0].recon_mut().point_set.points.iter_mut() {
        point.position.x += 0.01;
    }
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let proposal = state.camera_model_proposal.as_mut().unwrap();
    let mut inputs = proposal.inputs.clone();
    inputs.coeff_count = Some(6);
    proposal.set_inputs(inputs);
    (state, id)
}

/// Tick Preview retriangulation and let the job run to its answer.
fn previewed() -> (AppState, ReconId) {
    let (mut state, id) = proposed();
    state
        .camera_model_proposal
        .as_mut()
        .unwrap()
        .preview_retriangulation = true;
    state.settle_camera_model_proposal_job();
    (state, id)
}

/// Without the preview the job runs the switch alone and sends its
/// comparison; nothing is pushed and no row is written.
#[test]
fn the_job_sends_the_comparison_without_the_preview() {
    let (mut state, id) = proposed();
    let rows = state.action_log.entries().count();
    state.settle_camera_model_proposal_job();
    let proposal = state.camera_model_proposal_on(id).unwrap();
    let entry = proposal
        .current_comparison()
        .expect("the comparison arrived")
        .expect("the switch succeeded");
    assert!(entry.observations.observations > 0);
    assert!(
        proposal.triangulation.is_none(),
        "it retriangulated unasked"
    );
    assert!(matches!(
        proposal.triangulation_state(),
        TriangulationState::Off
    ));
    assert_eq!(versions(&state), 1);
    assert_eq!(state.action_log.entries().count(), rows);
    assert!(state.running_tasks().next().is_none());
}

/// Apply with the job's answer in hand installs it on the GUI thread, with no
/// background task, and pushes the version, label and positions the `Switch
/// camera model` task pushes for the same proposal.
#[test]
fn apply_with_the_answer_installs_the_version_the_task_would_push() {
    let (mut held, id) = previewed();
    assert!(matches!(
        held.camera_model_proposal_on(id)
            .unwrap()
            .triangulation_state(),
        TriangulationState::Ready(_)
    ));
    let applied = held.apply_camera_model_proposal(id).expect("applies");
    assert!(
        matches!(applied, ProposalApplied::Pushed(node, _) if node == id),
        "{applied:?}"
    );
    assert!(held.running_tasks().next().is_none(), "it solved again");
    assert!(held.camera_model_proposal.is_none());
    assert_eq!(versions(&held), 2);

    let (mut solved, solved_id) = proposed();
    assert!(matches!(
        solved
            .apply_camera_model_proposal(solved_id)
            .expect("applies"),
        ProposalApplied::Started(_)
    ));
    solved.finish_background_tasks();

    let (a, b) = (held.scene[0].recon(), solved.scene[0].recon());
    assert_eq!(a.image_table.cameras, b.image_table.cameras);
    assert_eq!(positions(&held), positions(&solved));
    assert_eq!(
        held.scene[0].history.current_version().label,
        solved.scene[0].history.current_version().label
    );
    // The same sentence, up to the serials, which one counter numbers for
    // every node of the session.
    let words = |state: &AppState| {
        let text = newest(state).text.clone();
        text[..text.rfind(" (v").expect("the serials")].to_string()
    };
    assert_eq!(words(&held), words(&solved));
    assert!(!newest(&held).failed);

    held.undo(id).expect("one edit to undo");
    assert_eq!(versions(&held), 2);
    assert_eq!(held.scene[0].history.cursor(), 0);
}

/// An answer is the version only for the generation it was computed for: a
/// change of a control leaves the triangulation pending, and Apply then runs
/// the task rather than installing the older answer.
#[test]
fn an_answer_for_an_older_generation_is_not_installed() {
    let (mut state, id) = previewed();
    let proposal = state.camera_model_proposal.as_mut().unwrap();
    let mut inputs = proposal.inputs.clone();
    inputs.coeff_count = Some(8);
    assert!(proposal.set_inputs(inputs));
    assert!(proposal.current_answer().is_none());
    assert!(matches!(
        proposal.triangulation_state(),
        TriangulationState::Pending
    ));
    assert!(proposal.current_comparison().is_none());
    assert!(matches!(
        state.apply_camera_model_proposal(id).expect("applies"),
        ProposalApplied::Started(_)
    ));
    state.finish_background_tasks();
    assert_eq!(versions(&state), 2);
    let camera = &state.scene[0].recon().image_table.cameras[1];
    assert_eq!(camera.model.radial_spline().expect("a spline").0.len(), 8);
}

/// A job whose generation moved on while it ran is cancelled, and what it
/// sends after that is not kept.
#[test]
fn a_running_job_for_an_older_generation_is_cancelled() {
    let (mut state, _) = proposed();
    state
        .camera_model_proposal
        .as_mut()
        .unwrap()
        .preview_retriangulation = true;
    assert!(state.drive_camera_model_proposal_job());
    assert!(state.proposal_jobs.is_queued());
    let proposal = state.camera_model_proposal.as_mut().unwrap();
    let mut inputs = proposal.inputs.clone();
    inputs.coeff_count = Some(7);
    proposal.set_inputs(inputs);
    let generation = proposal.generation;
    state.settle_camera_model_proposal_job();
    let proposal = state.camera_model_proposal.as_ref().unwrap();
    assert!(matches!(
        &proposal.triangulation,
        Some((g, Ok(_))) if *g == generation
    ));
    assert!(!state.proposal_jobs.is_queued());
}

/// On a node whose observations carry no pixel the preview is greyed with
/// `retriangulate_points`' sentence, and the job still runs its switch, which
/// reads the `.sift` files; with them gone the comparison carries the
/// switch's refusal.
#[test]
fn a_node_without_pixels_greys_the_preview_and_still_compares() {
    let workspace = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(workspace.path());
    state.select_recon(id);
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    let proposal = state.camera_model_proposal.as_mut().unwrap();
    let why = proposal
        .retriangulate_refusal
        .clone()
        .expect("greyed on a node without pixels");
    assert!(why.contains("no inline keypoints"), "{why}");
    proposal.preview_retriangulation = true;
    state.settle_camera_model_proposal_job();
    let proposal = state.camera_model_proposal_on(id).unwrap();
    assert!(matches!(
        proposal.triangulation_state(),
        TriangulationState::Refused(sentence) if sentence == why
    ));
    assert!(matches!(proposal.current_comparison(), Some(Ok(_))));
    assert!(proposal.triangulation.is_none());

    std::fs::remove_dir_all(workspace.path().join("features")).expect("the features");
    state.drop_camera_model_proposal(id, crate::camera_model_proposal::Dropped::Cancelled);
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    state.settle_camera_model_proposal_job();
    let proposal = state.camera_model_proposal_on(id).unwrap();
    match proposal.current_comparison() {
        Some(Err(why)) => assert!(why.contains("could not be read"), "{why}"),
        other => panic!("expected the switch's refusal, got {:?}", other.is_some()),
    }
}

/// The preview moves to the answer as the blend runs, every ending of the
/// proposal drops it in the frame it ends, and hiding the node leaves it.
#[test]
fn every_ending_drops_the_preview_and_hiding_the_node_leaves_it() {
    let (mut state, id) = previewed();
    let mut frames = 0;
    while state.step_camera_model_preview(1.0 / 60.0) {
        frames += 1;
        assert!(frames < 200, "the blend never settled");
    }
    let draw = state
        .camera_model_preview_draw()
        .expect("a preview is drawn");
    assert_eq!(draw.recon, id);
    assert_eq!(draw.s, 1.0);
    assert!(draw.targets.base.iter().any(|row| row[3] == FLAG_MOVES));

    state.scene[0].visible = false;
    assert!(!state.step_camera_model_preview(1.0 / 60.0));
    assert_eq!(state.camera_model_preview_draw().map(|d| d.s), Some(1.0));
    state.scene[0].visible = true;

    state.drop_camera_model_proposal(id, crate::camera_model_proposal::Dropped::Cancelled);
    assert!(state.camera_model_preview_draw().is_none());

    let (mut state, id) = previewed();
    state.step_camera_model_preview(1.0 / 60.0);
    assert!(state.camera_model_preview_draw().is_some());
    state.apply_camera_model_proposal(id).expect("applies");
    assert!(state.camera_model_preview_draw().is_none());

    let (mut state, id) = previewed();
    state.step_camera_model_preview(1.0 / 60.0);
    // An implicit ending applies the proposal, installing its answer.
    assert_eq!(state.end_camera_model_proposal_implicitly(), Some(id));
    assert!(state.camera_model_preview_draw().is_none());
    assert!(state.running_tasks().next().is_none());
}

/// On a version with an overlay the job reads one materialisation, and its
/// targets are laid out by the node's two buffers: a deleted point's row does
/// not move, and a moving added point's target lands in the additions
/// buffer's row.
#[test]
fn the_targets_follow_the_overlay_into_the_base_and_additions_buffers() {
    let (mut state, id) = spline_state();
    for point in state.scene[0].recon_mut().point_set.points.iter_mut() {
        point.position.x += 0.01;
    }
    let tracks = crate::state::edits::tracks_of_camera(state.scene[0].recon(), 1);
    let (deleted, rewritten) = (tracks[0], tracks[1]);
    state
        .delete_point(PointRef::new(id, deleted as usize))
        .expect("deletes");
    // A retriangulation of one point rewrites it as an addition.
    state
        .retriangulate_point(PointRef::new(id, rewritten as usize))
        .expect("retriangulates");
    let edited = state.scene[0].edited().clone();
    let base_count = edited.base_point_count();
    let added = edited.added.points.len();
    assert!(added > 0, "the fixture holds no addition");

    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let proposal = state.camera_model_proposal.as_mut().unwrap();
    let mut inputs = proposal.inputs.clone();
    inputs.coeff_count = Some(6);
    proposal.set_inputs(inputs);
    proposal.preview_retriangulation = true;
    state.settle_camera_model_proposal_job();
    let proposal = state.camera_model_proposal_on(id).unwrap();
    let answer = proposal.current_answer().expect("the job answered");
    let targets = &answer.targets;
    assert_eq!(targets.base.len(), base_count);
    assert_eq!(targets.additions.len(), added);
    assert_eq!(targets.base[deleted as usize][3], FLAG_STILL);
    assert_eq!(
        targets.base[rewritten as usize][3], FLAG_STILL,
        "the replaced base row of a rewritten point moved"
    );
    assert!(
        targets.additions.iter().any(|row| row[3] == FLAG_MOVES),
        "the added point did not move"
    );

    // Apply writes what the preview moved to: each moving base row's target
    // is where the new version holds that point.
    let moving: Vec<(usize, [f32; 4])> = targets
        .base
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, row)| row[3] == FLAG_MOVES)
        .collect();
    let map = state.scene[0].edited().clone();
    state.apply_camera_model_proposal(id).expect("applies");
    let after = state.scene[0].recon();
    let rows = map.materialize().1;
    for (index, row) in moving {
        let new = rows.forward(index as u32).expect("a live row") as usize;
        let p = after.point_set.points[new].position;
        for (k, v) in [p.x, p.y, p.z].into_iter().enumerate() {
            assert!((v as f32 - row[k]).abs() < 1e-5, "point {index}");
        }
    }
}

/// The preview's end state is the version Apply installs, for every point:
/// each moving row's target is the installed position to the last bit of the
/// `f32` the 3D view draws, each surfel's half-vectors times its size factor
/// are the installed ones to the bit, a still row's point and frame are
/// unchanged, and a lost point is installed as a direction.
#[test]
fn the_preview_ends_exactly_where_apply_puts_every_point_and_surfel() {
    let (mut state, id) = spline_state();
    {
        let points = &mut state.scene[0].recon_mut().point_set;
        let n = points.points.len();
        for (k, point) in points.points.iter_mut().enumerate() {
            point.position.x += 0.01 * (k % 7) as f64;
        }
        points.patch_u_halfvec_xyz = Some(ndarray::Array2::from_elem((n, 3), 0.05));
        points.patch_v_halfvec_xyz = Some(ndarray::Array2::from_shape_fn((n, 3), |(_, c)| {
            if c == 1 {
                0.05
            } else {
                -0.02
            }
        }));
    }
    let stored = state.scene[0].recon().clone();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let proposal = state.camera_model_proposal.as_mut().unwrap();
    let mut inputs = proposal.inputs.clone();
    inputs.coeff_count = Some(6);
    proposal.set_inputs(inputs);
    proposal.preview_retriangulation = true;
    state.settle_camera_model_proposal_job();
    let targets = std::sync::Arc::clone(
        &state
            .camera_model_proposal_on(id)
            .unwrap()
            .current_answer()
            .expect("the job answered")
            .targets,
    );
    state.apply_camera_model_proposal(id).expect("applies");
    let applied = state.scene[0].recon();

    let f32s = |p: &nalgebra::Point3<f64>| [p.x as f32, p.y as f32, p.z as f32];
    let half = |recon: &sfmtool_core::SfmrReconstruction, i: usize| {
        let (u, v) = (
            recon.point_set.patch_u_halfvec_xyz.as_ref().unwrap(),
            recon.point_set.patch_v_halfvec_xyz.as_ref().unwrap(),
        );
        [
            u[[i, 0]],
            u[[i, 1]],
            u[[i, 2]],
            v[[i, 0]],
            v[[i, 1]],
            v[[i, 2]],
        ]
    };
    let mut moving = 0;
    for (i, row) in targets.base.iter().enumerate() {
        let (before, after) = (&stored.point_set.points[i], &applied.point_set.points[i]);
        let size = targets.base_size[i];
        match row[3] {
            flag if flag == FLAG_MOVES => {
                moving += 1;
                assert_eq!(f32s(&after.position), [row[0], row[1], row[2]], "point {i}");
                let scaled = half(&stored, i).map(|h| h * size);
                assert_eq!(scaled, half(applied, i), "surfel {i}");
            }
            flag if flag == FLAG_STILL => {
                assert_eq!(before.position, after.position, "point {i} moved");
                assert_eq!(half(&stored, i), half(applied, i), "surfel {i} resized");
            }
            _ => assert!(after.is_at_infinity(), "lost point {i} kept a position"),
        }
    }
    assert!(moving > 0);
}

// -- As a quiet task in the fast queue -------------------------------------

/// A gated job a task's place in the fast queue is held by, which produces
/// nothing.
fn held_step(gate: std::sync::mpsc::Receiver<()>) -> crate::background::Job {
    Box::new(move |_| {
        let _ = gate.recv();
        crate::background::Finished::NoChange("Nothing to fit".to_string())
    })
}

/// A node beside the proposal's, which a fast task can lock while the
/// proposal is open: an open proposal refuses every task on its own node.
fn other_node(state: &mut AppState) -> ReconId {
    state.append_node(crate::scene::SceneNode::demo(
        sfmtool_core::SfmrReconstruction::demo(8),
    ))
}

/// The quiet tasks waiting in the fast queue, by id, in the order they will
/// start.
fn quiet_waiting(state: &AppState) -> Vec<u64> {
    state
        .queue(Queue::Fast)
        .waiting
        .iter()
        .filter(|task| task.operation.quiet)
        .map(|task| task.id)
        .collect()
}

/// The job runs as a *Camera model proposal* quiet task in the fast queue:
/// nothing that reports someone's work lists it, it locks nothing, it writes
/// no row and leaves no record, and the Background Tasks panel names it on its
/// quiet line.
#[test]
fn the_job_runs_as_a_quiet_task_that_nothing_reports() {
    let (mut state, id) = proposed();
    state.action_log.clear();
    assert!(state.drive_camera_model_proposal_job());
    let task = state.proposal_jobs.queued_task().expect("queued");
    assert!(
        state.quiet_task_running(task),
        "an idle fast queue ran nothing"
    );
    assert!(state.running_tasks().next().is_none());
    assert!(state.waiting_tasks().next().is_none());
    assert!(state.task(task).is_none());
    assert!(crate::background::busy(&state.busy_notice).is_empty());
    assert!(state.busy_refusal(id).is_none(), "the job locked its node");

    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(600.0, 400.0),
        )),
        ..Default::default()
    };
    let texts = crate::test_support::painted_texts(&egui::Context::default(), input, |ui| {
        crate::background::panel::show(ui, &mut state)
    });
    assert_eq!(
        texts,
        [
            "Slow",
            "Nothing running",
            "Fast",
            "Nothing running",
            "quiet",
            "Camera model proposal",
            "run_a: camera 1",
        ],
        "the quiet line does not name the proposal"
    );

    state.settle_camera_model_proposal_job();
    assert!(state
        .camera_model_proposal_on(id)
        .unwrap()
        .current_comparison()
        .is_some());
    assert_eq!(state.action_log.entries().count(), 0, "the job wrote a row");
    assert!(state.finished_tasks.is_empty(), "the job left a record");
}

/// A task someone asks for in the fast queue asks a running job to stop and
/// starts once it has; the job, which kept nothing of the cancelled run, is
/// queued again behind the task, and once that task has landed it still
/// delivers its comparison and its answer.
#[test]
fn an_asked_for_fast_task_cancels_the_job_which_answers_after_it() {
    let (mut state, id) = proposed();
    let other = other_node(&mut state);
    state
        .camera_model_proposal
        .as_mut()
        .unwrap()
        .preview_retriangulation = true;
    let (release, hold) = std::sync::mpsc::channel();
    state.proposal_jobs.hold_next = Some(hold);
    assert!(state.drive_camera_model_proposal_job());
    let first = state.proposal_jobs.queued_task().expect("queued");
    assert!(state.quiet_task_running(first));

    let (open, gate) = std::sync::mpsc::channel();
    let fit = state
        .start_background_task(Operation::BENCH_FIT, other, held_step(gate))
        .expect("a busy fast queue queues the task");
    assert!(
        matches!(state.task(fit), Some(TaskState::Waiting(_))),
        "the task started beside the job"
    );
    assert!(
        state.quiet_task_cancelled(first),
        "the task did not ask the job to stop"
    );
    release.send(()).expect("the job is held");
    state.wait_for_proposal_job();
    assert!(
        matches!(state.task(fit), Some(TaskState::Running(_))),
        "the task did not start once the job had stopped"
    );
    let proposal = state.camera_model_proposal_on(id).unwrap();
    assert!(
        proposal.comparison.is_none(),
        "the cancelled run sent its switch"
    );
    assert!(proposal.triangulation.is_none());

    // Queued again, it waits behind the task.
    state.drive_camera_model_proposal_job();
    let again = state.proposal_jobs.queued_task().expect("queued again");
    assert_ne!(again, first);
    assert_eq!(quiet_waiting(&state), [again]);

    open.send(()).expect("the task is waiting");
    state.finish_background_task(fit);
    state.settle_camera_model_proposal_job();
    let proposal = state.camera_model_proposal_on(id).unwrap();
    assert!(matches!(proposal.current_comparison(), Some(Ok(_))));
    assert!(matches!(
        proposal.triangulation_state(),
        TriangulationState::Ready(_)
    ));
}

/// A job that waits in the fast queue and is replaced by a newer request, or
/// dropped unrun, never ran: neither is kept on the proposal as an answer or a
/// failure, the source keeps one entry waiting, and a dropped one is queued
/// again on the driver's next frame and delivers once it runs.
#[test]
fn a_replaced_or_dropped_job_is_neither_an_answer_nor_a_failure() {
    let (mut state, id) = proposed();
    let other = other_node(&mut state);
    let (open, gate) = std::sync::mpsc::channel();
    let fit = state
        .start_background_task(Operation::BENCH_FIT, other, held_step(gate))
        .expect("the fast queue is idle");
    assert!(state.drive_camera_model_proposal_job());
    let first = state.proposal_jobs.queued_task().expect("queued");
    assert!(!state.quiet_task_running(first), "it ran beside the task");

    // Replaced: a change of a control while it waits.
    let proposal = state.camera_model_proposal.as_mut().unwrap();
    let mut inputs = proposal.inputs.clone();
    inputs.coeff_count = Some(7);
    assert!(proposal.set_inputs(inputs));
    state.drive_camera_model_proposal_job();
    let second = state.proposal_jobs.queued_task().expect("queued");
    assert_ne!(second, first, "not replaced");
    assert_eq!(quiet_waiting(&state), [second], "two entries wait");
    state.drive_camera_model_proposal_job();
    let proposal = state.camera_model_proposal_on(id).unwrap();
    assert!(proposal.comparison.is_none() && proposal.triangulation.is_none());

    // Dropped: the queue takes it out unrun, which its source hears of on the
    // next frame and queues it again.
    state.cancel_quiet_task(second);
    assert!(quiet_waiting(&state).is_empty());
    state.drive_camera_model_proposal_job();
    let third = state.proposal_jobs.queued_task().expect("queued again");
    assert_ne!(third, second);
    assert_eq!(quiet_waiting(&state), [third]);
    let proposal = state.camera_model_proposal_on(id).unwrap();
    assert!(
        proposal.comparison.is_none() && proposal.triangulation.is_none(),
        "a dropped job was kept as an answer or a failure"
    );

    open.send(()).expect("the task is waiting");
    state.finish_background_task(fit);
    state.settle_camera_model_proposal_job();
    assert!(matches!(
        state
            .camera_model_proposal_on(id)
            .unwrap()
            .current_comparison(),
        Some(Ok(_))
    ));
}

/// Among quiet tasks the proposal's job goes first: queued behind a live
/// evaluation that waits in the fast queue, it starts ahead of it.
#[test]
fn the_job_goes_ahead_of_a_waiting_live_evaluation() {
    let (mut state, _) = proposed();
    let other = other_node(&mut state);
    let bench = state.append_node(crate::scene::SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    crate::bench::tests::put_on_bench(&mut state, bench);
    let (open, gate) = std::sync::mpsc::channel();
    let fit = state
        .start_background_task(Operation::BENCH_FIT, other, held_step(gate))
        .expect("the fast queue is idle");
    state.drive_bench_evaluation();
    let evaluation = state.queued_evaluation_task().expect("an evaluation waits");
    state.drive_camera_model_proposal_job();
    let job = state.proposal_jobs.queued_task().expect("the job waits");
    assert_eq!(quiet_waiting(&state), [job, evaluation]);

    open.send(()).expect("the task is waiting");
    state.finish_background_task(fit);
    assert!(
        state.quiet_task_running(job),
        "the evaluation started ahead of the proposal's job"
    );
    state.settle_camera_model_proposal_job();
    state.settle_bench_evaluation();
}
