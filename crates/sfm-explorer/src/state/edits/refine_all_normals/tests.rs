// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refine All Normals in the viewer: one version per run of either step, Undo
//! back out of it, the overlay folded in with the selection following it, the
//! failed entries of a cancel and of an unreadable photograph, the refusal in
//! the menu's words, and the Action Log's sentence.
//!
//! The step on each point is core's, and is tested there
//! (`crates/sfmtool-core/src/bench/tests/refine_normals.rs`). The fixture is
//! Refit All Points', whose photographs are a pattern in screen space rather
//! than views of one surface, so which outcome each point gets is not
//! asserted here, only that a pairwise-flow run reports outcomes that only
//! that step has.

use std::sync::Arc;

use sfmtool_core::bench::{
    NormalError, NormalLeftUnchanged, NormalTurned, RefineNormalsOutcome, RefineNormalsReport,
    RefinedNormal,
};
use sfmtool_core::SfmrReconstruction;

use crate::action_log::Entry;
use crate::scene::{PointRef, SceneNode};
use crate::state::edits::refit_all_points::tests::refit_state;
use crate::state::edits::RefineAllNormalsStep;
use crate::state::AppState;

/// The newest Action Log entry.
fn newest(state: &AppState) -> &Entry {
    state.action_log.entries().next_back().expect("an entry")
}

fn versions(state: &AppState) -> usize {
    state.scene[0].history.versions().len()
}

#[test]
fn a_refinement_pushes_one_version_with_a_new_base_and_undo_restores_the_old_one() {
    let (mut state, id) = refit_state(None);
    let before = Arc::clone(&state.scene[0].edited().base);
    let count = state.scene[0].point_count();
    state.selected_point = Some(PointRef::new(id, 5));

    state
        .start_refine_all_normals(id, RefineAllNormalsStep::GrownPlane)
        .expect("the fixture carries keypoints, frames and poses");
    state.finish_background_tasks();

    let node = &state.scene[0];
    assert_eq!(versions(&state), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert!(node.edited().deleted_points.is_empty() && node.edited().added.points.is_empty());
    assert_eq!(node.point_count(), count, "no point was deleted or created");
    assert_eq!(
        node.history.current_version().label,
        "Refined the normals of demo with the grown-plane normal"
    );
    assert_eq!(state.selected_point, Some(PointRef::new(id, 5)));
    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert!(
        entry
            .text
            .starts_with("Refined the normals of demo with the grown-plane normal: "),
        "{}",
        entry.text
    );
    assert!(
        entry.text.contains(" finite points turned"),
        "{}",
        entry.text
    );
    // A third of the fixture's points have one observation, which the step
    // refuses, and the entry says so.
    assert!(
        entry.text.contains("with too few observations"),
        "{}",
        entry.text
    );

    state.undo(id).expect("one version to take back");
    assert!(
        Arc::ptr_eq(&before, &state.scene[0].edited().base),
        "Undo did not return to the value before the refinement"
    );
}

/// The submenu's *Pairwise Flow* runs the pairwise-flow normal on every
/// point: one version, labelled with that step, whose entry counts the points.
#[test]
fn a_pairwise_flow_refinement_pushes_one_version_named_for_its_step() {
    let (mut state, id) = refit_state(None);
    let before = Arc::clone(&state.scene[0].edited().base);
    state
        .start_refine_all_normals(id, RefineAllNormalsStep::PairwiseFlow)
        .expect("the fixture carries keypoints, frames and poses");
    state.finish_background_tasks();

    assert_eq!(versions(&state), 2);
    assert_eq!(
        state.scene[0].history.current_version().label,
        "Refined the normals of demo with the pairwise-flow normal"
    );
    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert!(
        entry
            .text
            .starts_with("Refined the normals of demo with the pairwise-flow normal: "),
        "{}",
        entry.text
    );
    assert!(
        entry.text.contains("with too few observations"),
        "{}",
        entry.text
    );
    // Words only the pairwise-flow normal's outcomes bring, so the run was
    // that step's and not the grown plane's under another label.
    assert!(
        ["from a single pair", "undetermined", "unstable"]
            .iter()
            .any(|words| entry.text.contains(words)),
        "{}",
        entry.text
    );
    state.undo(id).expect("one version to take back");
    assert!(Arc::ptr_eq(&before, &state.scene[0].edited().base));
}

#[test]
fn a_pending_point_edit_is_folded_in_and_the_selection_follows_it() {
    let (mut state, id) = refit_state(None);
    let count = state.scene[0].point_count();
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state.selected_point = Some(PointRef::new(id, 5));

    state
        .start_refine_all_normals(id, RefineAllNormalsStep::GrownPlane)
        .expect("the fixture carries keypoints, frames and poses");
    state.finish_background_tasks();

    let node = &state.scene[0];
    assert_eq!(versions(&state), 3);
    assert!(
        node.edited().deleted_points.is_empty(),
        "the overlay was kept"
    );
    assert_eq!(node.edited().base.point_count(), count - 1);
    assert_eq!(state.selected_point, Some(PointRef::new(id, 4)));
    assert!(!newest(&state).failed, "{}", newest(&state).text);
}

/// Of either step.
#[test]
fn a_cancelled_refinement_pushes_no_version_and_writes_a_failed_entry() {
    for step in [
        RefineAllNormalsStep::GrownPlane,
        RefineAllNormalsStep::PairwiseFlow,
    ] {
        let (mut state, id) = refit_state(None);
        let before = Arc::clone(&state.scene[0].edited().base);
        state
            .start_refine_all_normals(id, step)
            .expect("the fixture carries keypoints, frames and poses");
        state.cancel_background_tasks();
        state.finish_background_tasks();

        assert_eq!(versions(&state), 1, "{step:?}");
        assert!(Arc::ptr_eq(&before, &state.scene[0].edited().base));
        let entry = newest(&state);
        assert!(entry.failed, "{}", entry.text);
        assert!(
            entry.text.contains("Refine all normals of demo cancelled"),
            "{}",
            entry.text
        );
    }
}

#[test]
fn a_photograph_that_cannot_be_read_fails_the_run_and_names_the_image() {
    let (mut state, id) = refit_state(Some(1));
    let name = state.scene[0].recon().image_set.images[1].name.clone();
    state
        .start_refine_all_normals(id, RefineAllNormalsStep::GrownPlane)
        .expect("the gate reads no photograph");
    state.finish_background_tasks();

    assert_eq!(versions(&state), 1);
    let entry = newest(&state);
    assert!(entry.failed, "{}", entry.text);
    assert!(
        entry.text.contains(&format!("Cannot read {name}.")),
        "{}",
        entry.text
    );
}

#[test]
fn a_reconstruction_with_no_keypoints_is_refused_in_the_menu_s_own_words() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));
    let id = state.selected_recon.expect("a selected reconstruction");
    let why = state
        .refine_refusal(id)
        .expect("the demo value carries no keypoints");
    assert!(why.contains("Convert to Embedded Patches"), "{why}");
    assert!(why.contains("refining a normal"), "{why}");
    let refused = state
        .start_refine_all_normals(id, RefineAllNormalsStep::GrownPlane)
        .expect_err("with no keypoints there is nothing to refine");
    assert!(
        refused.starts_with("Refine all normals of demo refused: "),
        "{refused}"
    );
    assert!(refused.contains(&why), "{refused}");
    assert_eq!(versions(&state), 1);
    assert!(newest(&state).failed);
}

#[test]
fn the_entry_counts_what_happened_to_each_point() {
    let turned = |turn_deg, low_confidence, unchanged| RefineNormalsOutcome::Turned {
        turn_deg,
        unchanged,
        by: NormalTurned::GrownPlane {
            joined: 3,
            low_confidence,
            agreement: 0.8,
        },
    };
    let outcomes = [
        turned(4.0, false, false),
        turned(10.0, true, false),
        turned(2.0, false, true),
        RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::NoneJoined),
        RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::WeakTurn {
            turn_deg: 80.0,
            agreement: 0.7,
        }),
        RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::WeakTurn {
            turn_deg: 75.0,
            agreement: 0.7,
        }),
        RefineNormalsOutcome::NotRefined(NormalError::TooFewObservations(1)),
        RefineNormalsOutcome::NoBitmap,
        RefineNormalsOutcome::AtInfinity,
        RefineNormalsOutcome::AtInfinity,
    ];
    let report = RefineNormalsReport {
        points: outcomes
            .into_iter()
            .enumerate()
            .map(|(k, outcome)| RefinedNormal {
                index: k as u32,
                outcome,
            })
            .collect(),
    };
    assert_eq!(
        super::refine_summary(&report),
        "3 of 8 finite points turned, 1 of them as stored, 1 of them with low confidence, \
         3 points left their normal unchanged (1 with no view joined, \
         2 with a large turn on weak evidence), 2 points left as they were (1 with too few \
         observations, 1 with no bitmap to render), 2 points at infinity skipped, median \
         turn 4.0\u{b0}"
    );
}

#[test]
fn a_run_that_turned_nothing_says_so() {
    let report = RefineNormalsReport {
        points: vec![
            RefinedNormal {
                index: 0,
                outcome: RefineNormalsOutcome::AtInfinity,
            },
            RefinedNormal {
                index: 1,
                outcome: RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::NoneJoined),
            },
        ],
    };
    assert_eq!(
        super::refine_summary(&report),
        "0 of 1 finite points turned, 1 point left its normal unchanged (1 with no view \
         joined), 1 point at infinity skipped"
    );
}

/// A pairwise-flow run's sentence names the points read from a single pair
/// among the turned ones, and its undetermined and unstable points among
/// those whose normal was left unchanged.
#[test]
fn the_entry_names_the_pairwise_flow_outcomes() {
    let turned = |single_pair| RefineNormalsOutcome::Turned {
        turn_deg: 6.0,
        unchanged: false,
        by: NormalTurned::PairwiseFlow {
            single_pair,
            chosen_radius: 1.0,
            sigma_deg: 2.0,
        },
    };
    let outcomes = [
        turned(false),
        turned(true),
        RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::Undetermined),
        RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::Undetermined),
        RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::Unstable),
    ];
    let report = RefineNormalsReport {
        points: outcomes
            .into_iter()
            .enumerate()
            .map(|(k, outcome)| RefinedNormal {
                index: k as u32,
                outcome,
            })
            .collect(),
    };
    assert_eq!(
        super::refine_summary(&report),
        "2 of 5 finite points turned, 1 of them from a single pair, 3 points left their \
         normal unchanged (2 undetermined, 1 unstable), median turn 6.0\u{b0}"
    );
}
