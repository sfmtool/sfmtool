// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refit All Points in the viewer: one version per run, the overlay folded in
//! with the selection following it, the failed entries of a cancel and of an
//! unreadable photograph, and the Action Log's sentence.
//!
//! The fit of each point is core's, and is tested there
//! (`crates/sfmtool-core/src/bench/tests/refit_points.rs`).

use std::sync::Arc;

use sfmtool_core::bench::{FitError, RefitOutcome, RefitPoint, RefitPointsReport, RefitRange};
use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::SfmrReconstruction;

use crate::action_log::Entry;
use crate::scene::{PointRef, ReconId, SceneNode};
use crate::state::edits::tests::projected_embedded_demo;
use crate::state::AppState;

/// The newest Action Log entry.
fn newest(state: &AppState) -> &Entry {
    state.action_log.entries().next_back().expect("an entry")
}

/// A state holding one `embedded_patches` node of twelve framed points, with a
/// textured photograph cached for every image but `missing`, which then has
/// no file to be read from either.
fn refit_state(missing: Option<usize>) -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    for image in 0..state.scene[0].image_count() {
        if Some(image) == missing {
            continue;
        }
        // The pattern the bench's fixture uses: periods of a few pixels that
        // differ between the axes, so the kernels have something to register.
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| {
                let p = i / 3;
                ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
            })
            .collect();
        state.insert_photograph(id, image, ImageU8::new(w, h, 3, data));
    }
    (state, id)
}

fn versions(state: &AppState) -> usize {
    state.scene[0].history.versions().len()
}

#[test]
fn a_refit_pushes_one_version_with_a_new_base_and_undo_restores_the_old_one() {
    let (mut state, id) = refit_state(None);
    let before = Arc::clone(&state.scene[0].edited().base);
    let count = state.scene[0].point_count();
    state.selected_point = Some(PointRef::new(id, 5));

    state
        .start_refit_all_points(id)
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
    assert_eq!(node.history.current_version().label, "Refit demo");
    // No index moved, so the selection is where it was.
    assert_eq!(state.selected_point, Some(PointRef::new(id, 5)));
    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert!(entry.text.starts_with("Refit demo: "), "{}", entry.text);
    assert!(entry.text.contains(" points refit"), "{}", entry.text);
    assert!(
        !entry.text.starts_with("Refit demo: 0 of"),
        "no point was refit: {}",
        entry.text
    ); // A third of the fixture's points have one observation, which the fit
       // refuses, and the entry says so.
    assert!(
        entry.text.contains("with too few observations"),
        "{}",
        entry.text
    );

    state.undo(id).expect("one version to take back");
    assert!(
        Arc::ptr_eq(&before, &state.scene[0].edited().base),
        "Undo did not return to the value before the refit"
    );
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
        .start_refit_all_points(id)
        .expect("the fixture carries keypoints, frames and poses");
    state.finish_background_tasks();

    let node = &state.scene[0];
    assert_eq!(versions(&state), 3);
    assert!(
        node.edited().deleted_points.is_empty(),
        "the overlay was kept"
    );
    assert_eq!(node.edited().base.point_count(), count - 1);
    // Point 1's slot closed up, so point 5 is now point 4.
    assert_eq!(state.selected_point, Some(PointRef::new(id, 4)));
    assert!(!newest(&state).failed, "{}", newest(&state).text);
}

#[test]
fn a_cancelled_refit_pushes_no_version_and_writes_a_failed_entry() {
    let (mut state, id) = refit_state(None);
    let before = Arc::clone(&state.scene[0].edited().base);
    state
        .start_refit_all_points(id)
        .expect("the fixture carries keypoints, frames and poses");
    state.cancel_background_tasks();
    state.finish_background_tasks();

    assert_eq!(versions(&state), 1);
    assert!(Arc::ptr_eq(&before, &state.scene[0].edited().base));
    let entry = newest(&state);
    assert!(entry.failed, "{}", entry.text);
    assert!(
        entry.text.contains("Refit all points of demo cancelled"),
        "{}",
        entry.text
    );
}

#[test]
fn a_photograph_that_cannot_be_read_fails_the_run_and_names_the_image() {
    let (mut state, id) = refit_state(Some(1));
    let name = state.scene[0].recon().image_table.images[1].name.clone();
    state
        .start_refit_all_points(id)
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
        .refit_refusal(id)
        .expect("the demo value carries no keypoints");
    assert!(why.contains("Convert to Embedded Patches"), "{why}");
    let refused = state
        .start_refit_all_points(id)
        .expect_err("with no keypoints there is nothing to refit");
    assert!(refused.contains(&why), "{refused}");
    assert_eq!(versions(&state), 1);
    assert!(newest(&state).failed);
}

#[test]
fn the_entry_counts_what_happened_to_each_point() {
    let refit = |crossed, shift, range, unchanged, kept_at_seed| RefitOutcome::Refit {
        placed: 3,
        kept_at_seed,
        crossed,
        shift,
        classification: None,
        range,
        unchanged,
    };
    let outcomes = [
        refit(false, Some(0.5), RefitRange::Free, false, 2),
        refit(false, Some(1.5), RefitRange::Free, true, 0),
        refit(false, Some(1.0), RefitRange::Applied, false, 1),
        refit(
            true,
            None,
            RefitRange::Unresolved { released: true },
            false,
            0,
        ),
        RefitOutcome::NotFitted(FitError::TooFewRays {
            usable: 1,
            sightings: 2,
        }),
        RefitOutcome::NotFitted(FitError::TooFewObservations(1)),
        RefitOutcome::NotFitted(FitError::TooFewObservations(1)),
        RefitOutcome::NoBitmap,
        RefitOutcome::Held,
    ];
    let report = RefitPointsReport {
        points: outcomes
            .into_iter()
            .enumerate()
            .map(|(k, outcome)| RefitPoint {
                index: k as u32,
                outcome,
            })
            .collect(),
    };
    assert_eq!(
        super::refit_summary(&report),
        "4 of 8 points refit, 1 of them as stored, 1 of them ranged at their distance, \
         1 ranged point released: its reference image has no pose and it became a direction, \
         1 crossed to or from infinity, 3 observations kept their keypoint past the shift \
         bound, 4 left as they were (2 with too few observations, 1 with too few rays, \
         1 with no bitmap to render), 1 held, median shift 1.0000"
    );
}

#[test]
fn a_run_that_refit_nothing_says_so() {
    let report = RefitPointsReport {
        points: vec![RefitPoint {
            index: 0,
            outcome: RefitOutcome::Held,
        }],
    };
    assert_eq!(super::refit_summary(&report), "0 of 0 points refit, 1 held");
}

#[test]
fn the_photographs_read_are_the_images_some_live_point_observes() {
    let (mut state, id) = refit_state(None);
    let edited = state.scene[0].edited().clone();
    let every: Vec<usize> = (0..3).collect();
    assert_eq!(super::observed_images(&edited), every);
    // The fixture's points are seen from images 0..=2 only; a deleted point
    // takes nothing away while another point still sees the image.
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    assert_eq!(super::observed_images(state.scene[0].edited()), every);
}
