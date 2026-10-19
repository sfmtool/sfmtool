// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The camera model proposal: what it opens on, what it asks the switch for,
//! the outermost keypoint beside the domain, Apply and its version, Cancel,
//! the implicit endings, and the node's other edits greyed while it is open.

use sfmtool_core::camera::refit_intrinsics::{RefitTarget, ThetaFitSource};
use sfmtool_core::camera::{CameraIntrinsics, CameraModel};

use crate::action_log::Kind;
use crate::scene::{CameraRef, ImageRef, ReconId};
use crate::state::edits::ProposalApplied;
use crate::state::AppState;

use super::*;

/// The resectable node with its images split over two cameras, camera 1 a
/// four-coefficient `SFMTOOL_PINHOLE` on a 45° domain whose spline is flat, so
/// it is still the pinhole the observations were made through.
pub(crate) fn spline_state() -> (AppState, ReconId) {
    let (mut state, id) = crate::state::edits::tests::adjustable_state();
    {
        let recon = state.scene[0].recon_mut();
        let camera = &mut recon.image_table.cameras[0];
        let (f, _) = camera.focal_lengths();
        let (cx, cy) = camera.principal_point();
        camera.model = CameraModel::SimplePinhole {
            focal_length: f,
            principal_point_x: cx,
            principal_point_y: cy,
        };
        let mut second = recon.image_table.cameras[0].clone();
        second.model = CameraModel::SfmtoolPinhole {
            focal_length: f,
            principal_point_x: cx,
            principal_point_y: cy,
            bspline_rho_max: 1.0,
            bspline: vec![0.0; 4],
        };
        recon.image_table.cameras.push(second);
        for image in recon.image_table.images.iter_mut().skip(4) {
            image.camera_index = 1;
        }
    }
    (state, id)
}

/// The newest Action Log row.
pub(crate) fn newest(state: &AppState) -> &crate::action_log::Entry {
    state.action_log.entries().next_back().expect("an entry")
}

pub(crate) fn versions(state: &AppState) -> usize {
    state.scene[0].history.versions().len()
}

/// Every point position the node holds, for a before-and-after comparison.
pub(crate) fn positions(state: &AppState) -> Vec<[f64; 3]> {
    state.scene[0]
        .recon()
        .point_set
        .points
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect()
}

fn reach(theta_deg: f64) -> sfmtool_core::reconstruction::outermost_keypoint::KeypointReach {
    sfmtool_core::reconstruction::outermost_keypoint::KeypointReach {
        radius_px: 2.0 * theta_deg,
        theta_deg,
        image: 0,
        xy: [0.0, 0.0],
    }
}

// ── What it opens on ────────────────────────────────────────────────────

/// Every model the list offers is one the switch takes.
#[test]
fn every_target_model_is_one_the_switch_takes() {
    for name in TARGET_MODELS {
        let target = RefitTarget::from_name(name, None).expect(name);
        assert_eq!(target.model_name(), name);
    }
}

/// A fisheye lists `SFMTOOL_FISHEYE` first and leaves its own polynomial model
/// out; a perspective camera lists `SFMTOOL_PINHOLE` first; a spline camera
/// keeps its own model, which is the refit.
#[test]
fn the_model_list_orders_the_targets_by_the_source() {
    let fisheye = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 129.1,
            focal_length_y: 129.3,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.03,
            radial_distortion_k2: -0.008,
            radial_distortion_k3: 0.008,
            radial_distortion_k4: -0.002,
        },
        width: 480,
        height: 480,
    };
    let targets = targets_for(&fisheye);
    assert_eq!(targets[0], "SFMTOOL_FISHEYE");
    assert!(!targets.contains(&"OPENCV_FISHEYE"), "{targets:?}");

    let (state, _) = spline_state();
    let cameras = &state.scene[0].recon().image_table.cameras;
    let pinhole = targets_for(&cameras[0]);
    assert_eq!(pinhole[0], "SFMTOOL_PINHOLE");
    assert!(!pinhole.contains(&"SIMPLE_PINHOLE"), "{pinhole:?}");
    assert!(targets_for(&cameras[1]).contains(&"SFMTOOL_PINHOLE"));
}

/// A spline camera's proposal opens on its own model, the refit, at its own
/// count and domain with no fit angle, and asks the switch for what the Refit
/// Spline dialog asked for: no model named, no fit angle, the count, and no
/// domain while the field is untouched.
#[test]
fn a_spline_cameras_proposal_opens_as_the_refit_of_its_spline() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let proposal = state.camera_model_proposal_on(id).expect("open");
    assert_eq!(proposal.inputs.camera_model, "SFMTOOL_PINHOLE");
    assert!(proposal.is_spline_refit("SFMTOOL_PINHOLE"));
    assert_eq!(
        proposal.target_label("SFMTOOL_PINHOLE"),
        "SFMTOOL_PINHOLE (refit spline)"
    );
    assert_eq!(proposal.inputs.coeff_count, Some(4));
    assert_eq!(proposal.inputs.theta_fit_deg, None);
    assert_eq!(proposal.inputs.spline_domain_deg, None);
    let shown = proposal.shown_domain_deg().expect("a domain");
    assert!((shown - 45.0).abs() < 1e-9, "{shown}");
    let refit = proposal.fit.as_ref().expect("a fit");
    assert_eq!(refit.theta_fit_source, ThetaFitSource::SplineDomain);
    assert_eq!(
        proposal.request(),
        crate::state::edits::SwitchCameraModelRequest {
            camera: 1,
            camera_model: None,
            coeff_count: Some(4),
            spline_domain_deg: None,
            theta_fit_deg: None,
        }
    );
    // The domain field set back to the camera's own is untouched.
    let proposal = state.camera_model_proposal.as_mut().expect("open");
    let mut inputs = proposal.inputs.clone();
    inputs.spline_domain_deg = Some(shown);
    assert!(!proposal.set_inputs(inputs), "the same fit refitted");
    assert_eq!(proposal.request().spline_domain_deg, None);
    // The selection follows the proposal to its camera.
    assert_eq!(state.selected_camera, Some(CameraRef::new(id, 1)));
}

/// Any other camera opens on the list's first target, with the default fit
/// angle shown, and its own model is not in the list.
#[test]
fn a_polynomial_camera_opens_on_the_first_target() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    let proposal = state.camera_model_proposal_on(id).expect("open");
    assert_eq!(proposal.inputs.camera_model, proposal.targets[0]);
    assert!(!proposal.targets.contains(&"SIMPLE_PINHOLE"));
    assert_eq!(proposal.inputs.coeff_count, Some(8));
    assert!(proposal.fit.is_ok(), "{:?}", proposal.fit.as_ref().err());
    assert!(proposal.theta_fit_default.is_some());
    assert!(proposal.request().camera_model.is_some());
}

/// A change of a control advances the generation; one that asks for the same
/// fit keeps it.
#[test]
fn the_generation_advances_only_when_the_fit_inputs_change() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let proposal = state.camera_model_proposal.as_mut().expect("open");
    let generation = proposal.generation;
    let same = proposal.inputs.clone();
    assert!(!proposal.set_inputs(same));
    assert_eq!(proposal.generation, generation);
    let mut more = proposal.inputs.clone();
    more.coeff_count = Some(6);
    assert!(proposal.set_inputs(more));
    assert!(proposal.generation > generation);
}

/// The domain row's button takes the detected keypoint, the observed one
/// without a detected, and nothing without either.
#[test]
fn the_keypoint_button_takes_the_detected_angle_and_else_the_observed_one() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let proposal = state.camera_model_proposal.as_mut().expect("open");
    for (observed, detected, expected) in [
        (Some(41.2), Some(43.9), Some(43.9)),
        (Some(41.2), None, Some(41.2)),
        (None, None, None),
    ] {
        proposal.keypoint_extent = KeypointExtent {
            observed: observed.map(reach),
            detected: detected.map(reach),
        };
        let mut inputs = proposal.inputs.clone();
        inputs.spline_domain_deg = None;
        crate::intrinsics_detail::proposal::use_outermost_keypoint(proposal, &mut inputs);
        assert_eq!(inputs.spline_domain_deg, expected);
    }
}

#[test]
fn the_outermost_keypoint_is_labelled_by_its_source() {
    let both = KeypointExtent {
        observed: Some(reach(101.2)),
        detected: Some(reach(107.9)),
    };
    assert_eq!(
        both.describe().as_deref(),
        Some("outermost keypoint: 202.4 px, 101.2° observed; 215.8 px, 107.9° detected")
    );
    let observed_only = KeypointExtent {
        observed: Some(reach(101.2)),
        detected: None,
    };
    assert_eq!(
        observed_only.describe().as_deref(),
        Some("outermost keypoint: 202.4 px, 101.2° observed")
    );
    assert_eq!(KeypointExtent::default().describe(), None);
}

// ── Apply and Cancel ────────────────────────────────────────────────────

/// A proposal pushes nothing; Apply starts the `Switch camera model` task,
/// which pushes one version holding the switched camera and the
/// retriangulated tracks, labelled as the switch with the retriangulation
/// added; an undo restores both the camera and the points.
#[test]
fn apply_pushes_the_switch_and_the_retriangulation_as_one_version() {
    let (mut state, id) = spline_state();
    // Nudge the points so the retriangulation has something to move.
    for point in state.scene[0].recon_mut().point_set.points.iter_mut() {
        point.position.x += 0.01;
    }
    let before = positions(&state);
    let camera_before = state.scene[0].recon().image_table.cameras[1].clone();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let mut inputs = state.camera_model_proposal_on(id).unwrap().inputs.clone();
    inputs.coeff_count = Some(6);
    state
        .camera_model_proposal
        .as_mut()
        .unwrap()
        .set_inputs(inputs);
    assert_eq!(versions(&state), 1, "a proposal pushed a version");
    assert_eq!(newest(&state).kind, Kind::View);

    let applied = state.apply_camera_model_proposal(id).expect("applies");
    assert!(
        matches!(applied, ProposalApplied::Started(_)),
        "{applied:?}"
    );
    assert!(
        state.camera_model_proposal.is_none(),
        "the proposal outlived Apply"
    );
    assert_eq!(
        state.running_tasks().next().map(|task| task.operation.name),
        Some("Switch camera model")
    );
    state.finish_background_tasks();

    assert_eq!(versions(&state), 2);
    let label = &state.scene[0].history.current_version().label;
    assert_eq!(
        label,
        "Refit spline of camera 1 of run_a: 4 → 6 coefficients, domain 45.0° kept and \
         retriangulated its tracks"
    );
    let camera = &state.scene[0].recon().image_table.cameras[1];
    assert_eq!(camera.model.radial_spline().expect("a spline").0.len(), 6);
    let row = newest(&state);
    assert!(!row.failed, "{}", row.text);
    assert!(
        row.text.contains("fit rms ") && row.text.contains("points moved"),
        "{}",
        row.text
    );
    // Only the camera's tracks moved: a point camera 1's images do not
    // observe keeps its position.
    let after = positions(&state);
    let tracks = crate::state::edits::tracks_of_camera(state.scene[0].recon(), 1);
    let moved: Vec<usize> = (0..after.len())
        .filter(|&p| after[p] != before[p])
        .collect();
    assert!(!moved.is_empty(), "nothing was retriangulated");
    assert!(
        moved.iter().all(|&p| tracks.contains(&(p as u32))),
        "a point outside the camera's tracks moved"
    );

    state.undo(id).expect("one edit to undo");
    assert_eq!(state.scene[0].recon().image_table.cameras[1], camera_before);
    assert_eq!(positions(&state), before);
}

/// On a node whose observations carry no pixel, Apply is the switch alone on
/// the GUI thread, with the reason the tracks were not moved in its row; with
/// the `.sift` files gone the switch refuses and the proposal stays open.
#[test]
fn apply_without_pixels_is_the_switch_alone() {
    let workspace = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(workspace.path());
    state.select_recon(id);
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    let applied = state.apply_camera_model_proposal(id).expect("applies");
    assert_eq!(applied.pushed(), Some(id));
    assert_eq!(versions(&state), 2);
    let row = newest(&state);
    assert!(
        row.text
            .contains("its tracks were not retriangulated: This reconstruction's observations"),
        "{}",
        row.text
    );
    assert!(
        !state.scene[0]
            .history
            .current_version()
            .label
            .contains("retriangulated"),
        "the switch alone named a retriangulation"
    );

    // An implicit ending that pushes on the GUI thread tells the frame, once,
    // so the panels drop what they cached about the node.
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    assert_eq!(state.end_camera_model_proposal_implicitly(), Some(id));
    assert_eq!(state.tend_camera_model_proposal(), Some(id));
    assert_eq!(state.tend_camera_model_proposal(), None);
    assert_eq!(versions(&state), 3);

    // Without the `.sift` files the switch cannot read the pixels.
    std::fs::remove_dir_all(workspace.path().join("features")).expect("the features");
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    let why = state
        .apply_camera_model_proposal(id)
        .expect_err("no pixels");
    assert!(why.contains("could not be read"), "{why}");
    assert!(
        state.camera_model_proposal_on(id).is_some(),
        "the proposal was dropped"
    );
    assert_eq!(versions(&state), 3);
}

/// Cancel drops the proposal, writes one `View` row, and changes nothing.
#[test]
fn cancel_drops_the_proposal_and_changes_nothing() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    assert!(state.drop_camera_model_proposal(id, Dropped::Cancelled));
    assert!(state.camera_model_proposal.is_none());
    assert_eq!(
        newest(&state).text,
        "Cancelled the camera model proposal for camera 1 of run_a"
    );
    assert_eq!(versions(&state), 1);
    assert!(!state.drop_camera_model_proposal(id, Dropped::Cancelled));
}

/// A proposal whose fit was refused cannot be applied.
#[test]
fn a_refused_fit_is_shown_and_cannot_be_applied() {
    let (mut state, id) = spline_state();
    let inputs = ProposalInputs {
        camera_model: "SFMTOOL_PINHOLE",
        coeff_count: Some(4),
        theta_fit_deg: None,
        spline_domain_deg: Some(95.0),
    };
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), Some(inputs))
        .expect("opens with the refusal");
    let proposal = state.camera_model_proposal_on(id).expect("open");
    assert!(proposal.fit.is_err());
    assert!(
        proposal.fit_line().starts_with("Refused: camera 1"),
        "{}",
        proposal.fit_line()
    );
    let why = state.apply_camera_model_proposal(id).expect_err("refused");
    assert!(why.contains("camera 1"), "{why}");
    assert!(state.camera_model_proposal.is_some());
    assert_eq!(versions(&state), 1);
}

// ── The implicit endings ────────────────────────────────────────────────

/// Selecting another node applies the proposal; a refused fit is dropped
/// instead.
#[test]
fn selecting_another_node_applies_the_proposal_and_drops_a_refused_one() {
    let (mut state, id) = spline_state();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_b.sfmr",
    ));
    let other = state.scene[1].id;
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    // Nothing moves while the selection stays.
    assert_eq!(state.tend_camera_model_proposal(), None);
    assert!(state.camera_model_proposal.is_some());

    state.select_recon(other);
    assert_eq!(state.tend_camera_model_proposal(), None);
    assert!(state.camera_model_proposal.is_none());
    state.finish_background_tasks();
    assert_eq!(versions(&state), 2, "the step away did not apply");

    let inputs = ProposalInputs {
        camera_model: "SFMTOOL_PINHOLE",
        coeff_count: Some(4),
        theta_fit_deg: None,
        spline_domain_deg: Some(95.0),
    };
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), Some(inputs))
        .expect("opens");
    state.select_recon(other);
    state.tend_camera_model_proposal();
    assert!(state.camera_model_proposal.is_none());
    assert_eq!(
        newest(&state).text,
        "Dropped the camera model proposal for camera 1 of run_a: its fit was refused"
    );
    assert_eq!(versions(&state), 2);
}

/// Closing the node applies the proposal first, which makes it busy, so the
/// close is refused with the busy sentence.
#[test]
fn closing_applies_first_and_is_then_refused_while_busy() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    assert_eq!(state.end_camera_model_proposal_on(id), None);
    let why = state.close_node(id).expect_err("busy");
    assert!(
        why.contains("Switch camera model is still running"),
        "{why}"
    );
    state.finish_background_tasks();
    assert_eq!(versions(&state), 2);
    state
        .close_node(id)
        .expect("closes once the version landed");
}

/// Undo, redo and a history jump drop an open proposal with one `View` row,
/// and move the cursor as asked.
#[test]
fn undo_redo_and_a_jump_drop_the_proposal() {
    let (mut state, id) = spline_state();
    state.delete_points(id, &[0]).expect("an edit to undo");
    for step in 0..3 {
        state
            .open_camera_model_proposal(CameraRef::new(id, 1), None)
            .expect("opens");
        let before = state.action_log.revision();
        match step {
            0 => state.undo(id).expect("undoes"),
            1 => state.redo(id).expect("redoes"),
            _ => {
                let first = state.scene[0].history.versions()[0].serial;
                state.jump_to_version(id, first).expect("jumps");
            }
        }
        assert!(state.camera_model_proposal.is_none(), "step {step}");
        let dropped: Vec<_> = state
            .action_log
            .since(before)
            .filter(|entry| entry.text.starts_with("Dropped the camera model proposal"))
            .collect();
        assert_eq!(dropped.len(), 1, "step {step}");
        assert_eq!(dropped[0].kind, Kind::View);
    }
}

/// A version that changes under the proposal by some other way drops it.
#[test]
fn a_version_changed_under_the_proposal_drops_it() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    // Pushed past the greyed menus, as a path this module does not know of
    // would.
    let node = &mut state.scene[0];
    let value = node.history.current().clone();
    node.history.push(
        value,
        crate::document::PointMap::Removed(Vec::new()),
        "something else".to_string(),
    );
    state.tend_camera_model_proposal();
    assert!(state.camera_model_proposal.is_none());
    assert!(newest(&state)
        .text
        .ends_with("the reconstruction changed under it"));
}

/// A bench step shares the version's document half, so it leaves the proposal
/// open; the commit, the one bench step that writes the reconstruction, is
/// refused.
#[test]
fn a_bench_step_leaves_the_proposal_and_a_commit_is_refused() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let label = state
        .put_point_on_bench(crate::scene::PointRef::new(id, 0), None)
        .expect("a live point");
    assert!(versions(&state) > 1, "the bench step pushed no version");
    state.tend_camera_model_proposal();
    assert!(
        state.camera_model_proposal_on(id).is_some(),
        "a bench step dropped the proposal"
    );
    let why = state.commit_bench_track(id, &label).expect_err("refused");
    assert_eq!(why, PROPOSAL_OPEN_HINT);
}

/// Hiding the node leaves the proposal open.
#[test]
fn hiding_the_node_leaves_the_proposal() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    state.scene[0].visible = false;
    state.tend_camera_model_proposal();
    assert!(state.camera_model_proposal_on(id).is_some());
}

// ── What is greyed ──────────────────────────────────────────────────────

/// While a proposal is open the node's other edits are refused with one
/// sentence, Move Camera included, and the only task that can start on it is
/// Apply's.
#[test]
fn the_nodes_other_edits_are_refused_while_a_proposal_is_open() {
    let (mut state, id) = spline_state();
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    assert_eq!(state.edit_refusal(id).as_deref(), Some(PROPOSAL_OPEN_HINT));
    let why = state.delete_points(id, &[0]).expect_err("greyed");
    assert_eq!(why, PROPOSAL_OPEN_HINT);
    let why = state
        .start_retriangulate_all_points(id)
        .expect_err("greyed");
    assert_eq!(why, PROPOSAL_OPEN_HINT);
    let menu = state.image_menu(id).expect("a menu");
    assert_eq!(menu.resect_refusal(0), Some(PROPOSAL_OPEN_HINT));
    assert_eq!(menu.add_to_tracks_refusal(0), Some(PROPOSAL_OPEN_HINT));

    let mut viewer = crate::viewer_3d::Viewer3D::new();
    viewer.panel_size = [800, 600];
    viewer.jump_to_camera_view(ImageRef::new(id, 0), &state.scene[0]);
    let why = crate::camera_lock::refusal(&state, &viewer).expect("greyed");
    assert!(why.contains("camera model proposal"), "{why}");
}

/// Switch Model… is greyed on a busy node, on another camera of a node with a
/// proposal open, and on a node with no posed image.
#[test]
fn switch_model_is_greyed_for_the_three_reasons() {
    let (mut state, id) = spline_state();
    state.start_retriangulate_all_points(id).expect("starts");
    let why = state
        .switch_model_refusal(CameraRef::new(id, 1))
        .expect("busy");
    assert!(why.contains("is still running"), "{why}");
    state.finish_background_tasks();

    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let why = state
        .switch_model_refusal(CameraRef::new(id, 0))
        .expect("another camera");
    assert!(why.contains("open on camera 1 of run_a"), "{why}");
    assert_eq!(state.switch_model_refusal(CameraRef::new(id, 1)), None);
    state.drop_camera_model_proposal(id, Dropped::Cancelled);

    for image in state.scene[0].recon_mut().image_table.images.iter_mut() {
        image.translation_xyz.x = f64::NAN;
    }
    let why = state
        .switch_model_refusal(CameraRef::new(id, 0))
        .expect("no pose");
    assert!(why.contains("carries a pose"), "{why}");
}

/// Opening a proposal ends a Move Camera lock held on the node, by the lock's
/// own rule: unmoved, it is dropped silently.
#[test]
fn opening_a_proposal_ends_a_held_lock() {
    let (mut state, id) = spline_state();
    let mut viewer = crate::viewer_3d::Viewer3D::new();
    viewer.panel_size = [800, 600];
    let image = ImageRef::new(id, 0);
    state.select_image(Some(image));
    viewer.jump_to_camera_view(image, &state.scene[0]);
    crate::camera_lock::enter(&mut viewer, &mut state).expect("takes the camera in hand");
    assert!(viewer.camera_lock.is_some());

    let moved = open_in_window(&mut viewer, &mut state, CameraRef::new(id, 0), false);
    assert_eq!(moved, None, "an unmoved lock pushed a version");
    assert!(
        viewer.camera_lock.is_none(),
        "the lock outlived the proposal"
    );
    assert!(state.camera_model_proposal_on(id).is_some());
    assert_eq!(versions(&state), 1);
}

/// An implicit ending whose Apply is refused, here because the `.sift` files
/// the switch reads its pixels from are gone, drops the proposal with a `View`
/// row of its own after the refusal's failed row.
#[test]
fn an_implicit_ending_whose_apply_is_refused_drops_with_a_row() {
    let workspace = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(workspace.path());
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_b.sfmr",
    ));
    let other = state.scene[1].id;
    std::fs::remove_dir_all(workspace.path().join("features")).expect("the features");
    state.select_recon(id);
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    let label = state.scene[0].label.clone();
    let before = state.action_log.revision();
    state.select_recon(other);
    state.tend_camera_model_proposal();
    assert!(state.camera_model_proposal.is_none());
    let rows: Vec<(bool, Kind, String)> = state
        .action_log
        .since(before)
        .map(|entry| (entry.failed, entry.kind, entry.text.clone()))
        .collect();
    let refused = rows
        .iter()
        .position(|(failed, kind, _)| *failed && *kind == Kind::Edit)
        .expect("the refusal's row");
    let dropped = rows
        .iter()
        .position(|(failed, kind, text)| {
            !failed
                && *kind == Kind::View
                && *text
                    == format!(
                        "Dropped the camera model proposal for camera 0 of {label}: its \
                         Apply was refused"
                    )
        })
        .expect("the drop's row");
    assert!(refused < dropped, "{rows:?}");
    assert_eq!(versions(&state), 1);
}

/// An implicit ending while the slow queue runs a task on another node is not
/// refused: its Apply waits behind that task, the proposal's node busy with
/// the waiting sentence, and the switch starts and lands once the task ahead
/// of it has.
#[test]
fn an_implicit_ending_behind_a_busy_queue_waits_to_apply() {
    let (mut state, id) = spline_state();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_b.sfmr",
    ));
    let other = state.scene[1].id;
    state
        .open_camera_model_proposal(CameraRef::new(id, 1), None)
        .expect("opens");
    let (release, held) = std::sync::mpsc::channel::<()>();
    let ahead = state
        .start_background_task(
            crate::background::Operation::RETRIANGULATE_ALL_POINTS,
            other,
            Box::new(move |_progress| {
                let _ = held.recv();
                crate::background::Finished::Failed("nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    state.select_recon(other);
    state.tend_camera_model_proposal();
    assert!(state.camera_model_proposal.is_none());
    let switch = state
        .waiting_tasks()
        .next()
        .map(|task| (task.id, task.operation.name))
        .expect("the Apply waits");
    assert_eq!(switch.1, "Switch camera model");
    assert_eq!(
        state.busy_refusal(id),
        Some("run_a is busy: Switch camera model is waiting to run.".to_string())
    );
    release.send(()).expect("the worker is waiting");
    state.finish_background_task(ahead);
    assert!(state.running_task(switch.0).is_some(), "the switch started");
    state.finish_background_task(switch.0);
    assert_eq!(versions(&state), 2);
}

// ── The comparison's observations ───────────────────────────────────────

/// `kerry_park`'s first `OPENCV_FISHEYE`, 480 × 480, trusted to about 84°
/// and with `fx ≠ fy`.
pub(crate) fn kerry_camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 129.1499937015594,
            focal_length_y: 129.2573627423474,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.038113353966529886,
            radial_distortion_k2: -0.00800851799065643,
            radial_distortion_k3: 0.008329720504707577,
            radial_distortion_k4: -0.0026901578801066814,
        },
        width: 480,
        height: 480,
    }
}

/// How far off its image's axis each of the fixture's wide points is, past
/// the camera's trusted bound.
pub(crate) const WIDE_DEG: f64 = 95.0;

/// The demo reconstruction seen through [`kerry_camera`], with one more point
/// per image [`WIDE_DEG`] off that image's axis, past the camera's trusted
/// bound. Every point is observed, at its exact projection, by each image it
/// projects into, with the keypoints inline.
pub(crate) fn fisheye_node(path: &str) -> crate::scene::SceneNode {
    use sfmtool_core::reconstruction::{ObservationSource, TrackObservation};

    let mut recon = sfmtool_core::SfmrReconstruction::demo(120);
    let camera = kerry_camera();
    recon.image_table.cameras = vec![camera.clone()];
    let wide: Vec<sfmtool_core::reconstruction::Point3D> = recon
        .image_table
        .images
        .iter()
        .map(|image| {
            let (sin, cos) = WIDE_DEG.to_radians().sin_cos();
            let in_camera = nalgebra::Vector3::new(sin, 0.0, -cos) * 3.0;
            let world = image.quaternion_wxyz.inverse() * (in_camera - image.translation_xyz);
            sfmtool_core::reconstruction::Point3D {
                position: nalgebra::Point3::from(world),
                w: 1.0,
                color: [200, 40, 40],
                error: 0.5,
                normal: nalgebra::Vector3::zeros(),
            }
        })
        .collect();
    recon.point_set.points.extend(wide);

    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for (p, point) in recon.point_set.points.iter().enumerate() {
        let mut count = 0u32;
        for (i, image) in recon.image_table.images.iter().enumerate() {
            let local = image.quaternion_wxyz * point.position.coords + image.translation_xyz;
            // Inside the fold, so the keypoint is the exact projection of a
            // ray the lens images.
            let theta = local.x.hypot(local.y).atan2(-local.z).to_degrees();
            if theta > 100.0 {
                continue;
            }
            let Some((u, v)) = camera.ray_to_pixel([local.x, local.y, local.z]) else {
                continue;
            };
            if u < 0.0 || v < 0.0 || u >= 480.0 || v >= 480.0 {
                continue;
            }
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push([u as f32, v as f32]);
            count += 1;
        }
        counts.push(count);
    }
    let images = recon.image_table.images.len();
    let mut keypoints_xy = ndarray::Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; images],
    };
    recon.metadata.feature_source = "embedded_patches".to_string();
    recon.point_set.tracks = tracks;
    recon.point_set.observation_counts = counts;
    recon.rebuild_derived_fields();
    crate::scene::SceneNode::from_path(std::path::Path::new(path), recon)
}

/// A state holding [`fisheye_node`], with a proposal of `SFMTOOL_FISHEYE`
/// open on its camera.
pub(crate) fn fisheye_proposal_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(fisheye_node("/runs/kerry.sfmr"));
    let id = state.scene[0].id;
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    (state, id)
}

/// The rug has one angle per observation, ascending, under the current camera,
/// and the wide points are past its trusted bound.
#[test]
fn the_rug_has_one_tick_per_observation() {
    let (state, _) = fisheye_proposal_state();
    let proposal = state.camera_model_proposal.as_ref().expect("open");
    assert_eq!(proposal.inputs.camera_model, "SFMTOOL_FISHEYE");
    let observations = state.scene[0].recon().point_set.tracks.len();
    assert_eq!(proposal.rug().len(), observations);
    assert!(proposal.rug().windows(2).all(|w| w[0] <= w[1]));
    let limit = sfmtool_core::camera::report::trustworthy_max_theta_deg(&kerry_camera()).unwrap();
    let past = proposal.rug().iter().filter(|&&t| t > limit).count();
    assert!(past >= 8, "the wide points: {past}");
    // Under the current camera, whose inverse is not to be trusted there, so
    // not at the angle the points were placed at.
    let widest = proposal.rug().last().copied().unwrap();
    assert!(widest > limit + 5.0, "{widest}");
}

/// Each observation of an image is reprojected under both cameras: the
/// current camera puts it on its keypoint, and the proposed one reprojects it
/// too, the wide point past the bound included.
#[test]
fn an_images_observations_are_reprojected_under_both_cameras() {
    let (state, _) = fisheye_proposal_state();
    let proposal = state.camera_model_proposal.as_ref().expect("open");
    let recon = state.scene[0].recon();
    let observed = recon
        .point_set
        .tracks
        .iter()
        .filter(|t| t.image_index == 2)
        .count();
    let layer = proposal.image_reprojections(2);
    assert_eq!(layer.len(), observed);
    for observation in &layer {
        let current = observation.current.expect("the current camera projects it");
        let error =
            (current[0] - observation.keypoint[0]).hypot(current[1] - observation.keypoint[1]);
        assert!(error < 1e-3, "{error}");
        assert!(observation.proposed.is_some());
    }
    assert!(layer.iter().any(|o| {
        let (dx, dy) = (o.keypoint[0] - 240.0, o.keypoint[1] - 240.0);
        dx.hypot(dy) > 200.0
    }));
    // An image the table does not have gets nothing.
    assert!(proposal.image_reprojections(999).is_empty());
}

/// A point at infinity is reprojected as the direction it is.
#[test]
fn a_point_at_infinity_is_reprojected_as_a_direction() {
    let mut node = fisheye_node("/runs/kerry.sfmr");
    let recon = node.recon_mut();
    let row = recon
        .point_set
        .tracks
        .iter()
        .position(|t| t.image_index == 2)
        .expect("an observation of image 2");
    let point = recon.point_set.tracks[row].point_index as usize;
    // The direction from image 2's centre through the point: the keypoint
    // stays the projection of that direction.
    let centre = recon.image_table.images[2].camera_center();
    let direction = (recon.point_set.points[point].position - centre).normalize();
    recon.point_set.points[point].position = nalgebra::Point3::from(direction);
    recon.point_set.points[point].w = 0.0;
    let mut state = AppState::new();
    state.append_node(node);
    let id = state.scene[0].id;
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    let proposal = state.camera_model_proposal.as_ref().expect("open");
    let layer = proposal.image_reprojections(2);
    let keypoint = {
        let xy = state.scene[0].recon().keypoints_xy().unwrap();
        [f64::from(xy[[row, 0]]), f64::from(xy[[row, 1]])]
    };
    let observation = layer
        .iter()
        .find(|o| o.keypoint == keypoint)
        .expect("the observation");
    let current = observation.current.expect("a direction projects");
    assert!((current[0] - keypoint[0]).hypot(current[1] - keypoint[1]) < 1e-3);
}

/// Opening a proposal selects the Change to proposed field, and any ending
/// restores the mode the person had.
#[test]
fn the_change_field_is_selected_while_a_proposal_is_open() {
    use crate::state::FieldMode;
    let mut state = AppState::new();
    state.append_node(fisheye_node("/runs/kerry.sfmr"));
    let id = state.scene[0].id;
    assert_eq!(state.intrinsics_display.field, FieldMode::Distortion);
    // Opened over a mode of its own, so the mode the proposal restores
    // (`ChangeToProposed`) is not the one picked by hand while it is open
    // (`Distortion`), and the two can be told apart.
    state.intrinsics_display.field = FieldMode::ChangeToProposed;
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    assert_eq!(state.intrinsics_display.field, FieldMode::ChangeToProposed);
    // Chosen by hand while it is open, the mode stands across frames until it
    // ends, and the ending restores the mode from before the proposal.
    state.intrinsics_display.field = FieldMode::Distortion;
    state.tend_camera_model_proposal();
    state.tend_camera_model_proposal();
    assert_eq!(state.intrinsics_display.field, FieldMode::Distortion);
    state.drop_camera_model_proposal(id, Dropped::Cancelled);
    assert_eq!(state.intrinsics_display.field, FieldMode::ChangeToProposed);
    state.intrinsics_display.field = FieldMode::Distortion;

    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    assert_eq!(state.intrinsics_display.field, FieldMode::ChangeToProposed);
    state.drop_camera_model_proposal(id, Dropped::Cancelled);
    assert_eq!(state.intrinsics_display.field, FieldMode::Distortion);

    // Apply ends it too; the frame after restores the mode.
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    state.apply_camera_model_proposal(id).expect("applies");
    state.finish_background_tasks();
    state.tend_camera_model_proposal();
    assert!(state.camera_model_proposal.is_none());
    assert_eq!(state.intrinsics_display.field, FieldMode::Distortion);
}

/// No two proposals share a generation, so a cache keyed by it cannot serve
/// one proposal's camera for another's.
#[test]
fn a_reopened_proposal_takes_a_new_generation() {
    let (mut state, id) = fisheye_proposal_state();
    let first = state.camera_model_proposal.as_ref().unwrap().generation;
    state.drop_camera_model_proposal(id, Dropped::Cancelled);
    state
        .open_camera_model_proposal(CameraRef::new(id, 0), None)
        .expect("opens");
    assert_ne!(
        state.camera_model_proposal.as_ref().unwrap().generation,
        first
    );
}

/// The one pass over the `.sift` files gives the outermost extent the core's
/// `outermost_keypoints` gives over the base, on a node whose keypoints are in
/// its `.sift` files and on one with them inline, and reads each image's file
/// once.
#[test]
fn one_pass_gives_the_outermost_extent_and_reads_each_sift_file_once() {
    use sfmtool_core::reconstruction::outermost_keypoint::outermost_keypoints;

    let workspace = tempfile::tempdir().expect("a temporary directory");
    let (sift_state, _) = crate::state::edits::tests::convertible_state(workspace.path());
    let mut inline_state = AppState::new();
    inline_state.append_node(fisheye_node("/runs/kerry.sfmr"));
    for (state, has_sift) in [(&sift_state, true), (&inline_state, false)] {
        let base = &state.scene[0].history.current().base;
        let core = outermost_keypoints(base, &[0], true).remove(0);
        let mut reads: Vec<std::path::PathBuf> = Vec::new();
        let (extent, observed) = observations::read_camera_keypoints_with(base, base, 0, |path| {
            reads.push(path.to_path_buf());
            sfmtool_sift_format::read_sift_positions(path, usize::MAX).ok()
        });
        assert_eq!(extent.observed, core.observed);
        assert_eq!(extent.detected, core.detected);
        assert_eq!(extent.detected.is_some(), has_sift);
        assert!(extent.observed.is_some());

        // One read per image of the camera, each its own file.
        let images = base
            .image_table
            .images
            .iter()
            .filter(|image| image.camera_index == 0)
            .count();
        assert_eq!(reads.len(), images);
        let mut distinct = reads.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), reads.len());

        // Every observation of the camera's images has its keypoint.
        let observations = base
            .point_set
            .tracks
            .iter()
            .filter(|t| base.image_table.images[t.image_index as usize].camera_index == 0)
            .count();
        assert_eq!(observed.len(), observations);
        assert_eq!(
            (extent, observed),
            observations::read_camera_keypoints(base, base, 0)
        );
    }
}
