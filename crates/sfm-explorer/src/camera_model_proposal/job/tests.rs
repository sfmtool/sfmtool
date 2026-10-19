// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The proposal's job: the comparison it sends, the answer it holds while the
//! preview is asked for, which is the version Apply pushes, the generation
//! that keeps an older answer from being used, and the preview ending with
//! every ending of the proposal.

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
    assert!(state.proposal_jobs.is_running());
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
    assert!(!state.proposal_jobs.is_running());
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
