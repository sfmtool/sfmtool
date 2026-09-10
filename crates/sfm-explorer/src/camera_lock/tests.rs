// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Move Camera lock, headlessly: the snap that enters it, the pose the
//! viewport reads back as, what each navigation input does to that pose, and
//! the four ways the lock ends.
//!
//! The demo reconstruction is `sift_files` with no inline keypoints, which is
//! the `n/a` residual case; [`with_projected_keypoints`] gives it the inline
//! column, with every keypoint the exact projection, for the tests that measure
//! one.

use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use ndarray::Array2;
use sfmtool_core::{ObservationSource, RotQuaternion, Se3Transform, SfmrReconstruction};

use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

use super::*;

/// The image every test takes in hand.
const IMAGE: usize = 3;

/// Every observation's stored pixel is the exact projection of its point at the
/// stored poses, so the residual under the stored pose is zero and any move
/// away from it is measurable.
fn with_projected_keypoints(mut recon: SfmrReconstruction) -> SfmrReconstruction {
    let mut keypoints = Array2::<f32>::zeros((recon.point_set.tracks.len(), 2));
    for (row, observation) in recon.point_set.tracks.iter().enumerate() {
        let image = &recon.image_table.images[observation.image_index as usize];
        let point = &recon.point_set.points[observation.point_index as usize];
        let camera = recon
            .image_table
            .camera_for_image(observation.image_index as usize);
        let cam = image.quaternion_wxyz * point.position.coords + image.translation_xyz;
        if let Some((u, v)) = camera.ray_to_pixel([cam.x, cam.y, cam.z]) {
            keypoints[[row, 0]] = u as f32;
            keypoints[[row, 1]] = v as f32;
        }
    }
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; recon.image_table.images.len()],
    };
    recon.metadata.feature_source = sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// A state holding one demo node, selected, and a viewport looking through
/// [`IMAGE`].
fn looking_through(recon: SfmrReconstruction) -> (AppState, Viewer3D, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.selected_recon.expect("a selected reconstruction");
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [800, 600];
    let image = ImageRef::new(id, IMAGE);
    state.select_image(Some(image));
    viewer.jump_to_camera_view(image, &state.scene[0]);
    (state, viewer, id)
}

/// The plain fixture: no inline keypoints, so no residual.
fn plain() -> (AppState, Viewer3D, ReconId) {
    looking_through(SfmrReconstruction::demo(64))
}

/// The fixture whose observations carry their exact projections.
fn measured() -> (AppState, Viewer3D, ReconId) {
    looking_through(with_projected_keypoints(SfmrReconstruction::demo(64)))
}

/// Give the node a display transform: a quarter turn about `z`, a shift and a
/// scale, which is what an `Align to…` leaves behind.
fn align(state: &mut AppState) {
    state.scene[0].transform = Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::z(), std::f64::consts::FRAC_PI_2)
            .expect("a non-zero axis"),
        Vector3::new(2.0, -1.0, 0.5),
        1.7,
    );
}

/// The node's stored pose for [`IMAGE`], in its own frame.
fn stored_pose(state: &AppState) -> Se3Transform {
    sfmtool_core::reconstruction::move_camera::pose_of(state.scene[0].recon(), IMAGE)
}

/// The Action Log's texts, oldest first.
fn texts(state: &AppState) -> Vec<String> {
    state
        .action_log
        .entries()
        .map(|entry| entry.text.clone())
        .collect()
}

/// How far two poses stand apart: `(degrees, distance)`.
fn apart(a: &Se3Transform, b: &Se3Transform) -> (f64, f64) {
    (
        a.rotation
            .as_nalgebra()
            .rotation_to(b.rotation.as_nalgebra())
            .angle()
            .to_degrees(),
        (a.translation - b.translation).norm(),
    )
}

// ── Entering ────────────────────────────────────────────────────────────

#[test]
fn entering_snaps_to_the_stored_pose_and_keeps_the_field_of_view() {
    let (mut state, mut viewer, _id) = plain();
    // A free-look offset the reviewer happened to be holding, and a zoom.
    viewer.camera.nodal_pan(60.0, -25.0);
    viewer.camera.zoom_fov(3.0);
    let fov = viewer.camera.fov;

    enter(&mut viewer, &mut state).expect("camera view of a posed image");

    let lock = viewer.camera_lock.as_ref().expect("a held lock");
    let pending = pending_pose(&viewer, &state.scene[0]);
    let (degrees, distance) = apart(&pending, &lock.stored);
    assert!(degrees < 1e-9, "the snap left {degrees} deg of free-look");
    assert!(distance < 1e-9, "the snap left {distance} of offset");
    assert_eq!(viewer.camera.fov, fov, "the snap moved the lens");
    assert!(viewer.camera_view.is_some(), "the snap left camera view");
}

#[test]
fn entering_an_aligned_node_snaps_through_its_transform() {
    let (mut state, mut viewer, _id) = plain();
    align(&mut state);
    let image = ImageRef::new(state.scene[0].id, IMAGE);
    viewer.jump_to_camera_view(image, &state.scene[0]);

    enter(&mut viewer, &mut state).expect("camera view of a posed image");

    // The viewport is where the *drawn* camera is, and reading it back divides
    // the transform out again.
    let stored = stored_pose(&state);
    let pending = pending_pose(&viewer, &state.scene[0]);
    let (degrees, distance) = apart(&pending, &stored);
    assert!(
        degrees < 1e-9 && distance < 1e-9,
        "{degrees} deg, {distance}"
    );
    let drawn = state.scene[0]
        .transform
        .apply_to_point(&Point3::from(stored.translation));
    assert!(
        (viewer.camera.camera.position - drawn).norm() < 1e-9,
        "the viewport is not at the transformed centre"
    );
}

#[test]
fn the_gate_refuses_outside_camera_view_and_while_a_lock_is_held() {
    let (state, mut viewer, _id) = plain();
    viewer.camera_view = None;
    assert!(refusal(&state, &viewer)
        .expect("no camera view")
        .contains("Look through a camera"));

    let (mut state, mut viewer, _id) = plain();
    assert_eq!(refusal(&state, &viewer), None);
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    assert!(refusal(&state, &viewer)
        .expect("a held lock")
        .contains("already being moved"));
}

#[test]
fn the_gate_refuses_an_image_with_no_pose() {
    let mut recon = SfmrReconstruction::demo(64);
    recon.image_table.images[IMAGE].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    let (state, viewer, _id) = looking_through(recon);
    assert!(refusal(&state, &viewer)
        .expect("an unposed image")
        .contains("no pose"));
}

// ── The pending pose, and what moves it ─────────────────────────────────

/// One row of the table in `specs/gui/edits/move-camera.md`: what the input is
/// called, what it does to the viewport, and whether the pose is meant to move.
type NavigationCase = (&'static str, fn(&mut Viewer3D), bool);

/// Every navigation input, and whether the spec's table says it moves the pose.
/// The two that do not are the ones that move the *lens* and the *orbit
/// target*, neither of which is part of a pose.
#[test]
fn every_navigation_input_moves_the_pending_pose_except_the_lens_and_the_target() {
    let moves: Vec<NavigationCase> = vec![
        ("nodal pan", |v| v.camera.nodal_pan(30.0, 10.0), true),
        ("orbit", |v| v.camera.orbit(30.0, 10.0), true),
        ("pan", |v| v.camera.pan(30.0, 10.0, 800.0, 600.0), true),
        ("fly", |v| v.camera.fly_move(0.5, 0.25, -0.1), true),
        ("tilt", |v| v.camera.tilt(0.2), true),
        ("zoom fov", |v| v.camera.zoom_fov(2.0), false),
        ("target push", |v| v.camera.target_push_pull(1.5), false),
    ];
    for (what, apply, expected) in moves {
        let (mut state, mut viewer, _id) = plain();
        enter(&mut viewer, &mut state).expect("camera view of a posed image");
        let before = pending_pose(&viewer, &state.scene[0]);
        apply(&mut viewer);
        let after = pending_pose(&viewer, &state.scene[0]);
        let (degrees, distance) = apart(&before, &after);
        let moved = degrees > 1e-9 || distance > 1e-9;
        assert_eq!(moved, expected, "{what}: {degrees} deg, {distance}");
        assert!(viewer.camera_view.is_some(), "{what} left camera view");
    }
}

#[test]
fn a_held_lock_keeps_camera_view_through_every_navigation_path() {
    let (mut state, mut viewer, _id) = plain();
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.leave_camera_view();
    assert!(
        viewer.camera_view.is_some(),
        "a navigation path dropped camera view while a camera was in hand"
    );
    // Every path in the viewport's input handling goes through that one funnel,
    // which is what makes the assertion above cover all of them rather than the
    // one it calls.
    let input = include_str!("../viewer_3d/input.rs");
    assert!(
        !input.contains("camera_view = None"),
        "a navigation path in input.rs clears camera_view directly, behind the lock's back"
    );
}

// ── The residual readout ────────────────────────────────────────────────

#[test]
fn the_readout_equals_a_direct_computation_and_rises_as_the_camera_moves() {
    let (mut state, mut viewer, _id) = measured();
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    let lock = viewer.camera_lock.as_ref().expect("a held lock");
    let stored = residuals(lock, &stored_pose(&state)).1.expect("keypoints");
    assert!(stored[0] < 1e-3, "the stored pose disagrees with itself");

    viewer.camera.nodal_pan(40.0, 0.0);
    let pending = pending_pose(&viewer, &state.scene[0]);
    let lock = viewer.camera_lock.as_ref().expect("a held lock");
    let (now, _) = residuals(lock, &pending);
    let now = now.expect("keypoints");
    assert!(now[0] > 1.0, "a 40 px nodal pan moved nothing: {now:?}");
    assert!(now[1] >= now[0], "p90 below the median");

    // The same numbers the core computes for that pose over the same value.
    let direct = sfmtool_core::reconstruction::move_camera::residual_quantiles_px(
        state.scene[0].recon().image_table.camera_for_image(IMAGE),
        &pending,
        &sfmtool_core::reconstruction::move_camera::edited_image_reprojection_samples(
            state.scene[0].edited(),
            IMAGE,
        ),
    )
    .expect("keypoints");
    assert_eq!(direct, now);
}

#[test]
fn a_value_with_no_keypoints_reads_n_a() {
    let (mut state, mut viewer, _id) = plain();
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    let lock = viewer.camera_lock.as_ref().expect("a held lock");
    let (now, stored) = residuals(lock, &stored_pose(&state));
    assert_eq!((now, stored), (None, None));
    let lines = banner_lines("image_003.jpg", "demo", now, stored);
    assert_eq!(lines[0], "Moving image_003.jpg (demo)");
    assert_eq!(lines[1], "residual n/a, stored n/a");
    assert_eq!(lines[2], "M or Enter commits, Esc cancels");
}

#[test]
fn the_banner_prints_both_pairs_when_there_are_keypoints() {
    let lines = banner_lines("a.jpg", "run", Some([4.125, 6.5]), Some([1.0, 2.25]));
    assert_eq!(lines[1], "residual 4.12 / 6.50 px, stored 1.00 / 2.25 px");
}

// ── Ending the lock ─────────────────────────────────────────────────────

#[test]
fn committing_pushes_one_version_whose_pose_is_the_viewport_s() {
    let (mut state, mut viewer, id) = measured();
    state.selected_point = Some(PointRef::new(id, 5));
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.camera.nodal_pan(40.0, 15.0);
    let pending = pending_pose(&viewer, &state.scene[0]);

    let moved = commit(&mut viewer, &mut state).expect("a movable camera");

    assert_eq!(moved, Some(id));
    assert!(viewer.camera_lock.is_none(), "the lock outlived its commit");
    assert_eq!(state.scene[0].history.versions().len(), 2);
    let landed = stored_pose(&state);
    let (degrees, distance) = apart(&landed, &pending);
    assert!(
        degrees < 1e-9 && distance < 1e-9,
        "{degrees} deg, {distance}"
    );
    // The move deletes and creates no points, so the selection stays where it
    // was and the value still holds it.
    assert_eq!(state.selected_point, Some(PointRef::new(id, 5)));
    let label = &state.scene[0].history.current_version().label;
    assert!(
        label.starts_with("Moved camera image_003.jpg (demo): "),
        "{label}"
    );
    assert!(label.contains(" deg, "), "{label}");
    let last = texts(&state).pop().expect("a log entry");
    assert!(last.contains("Moved camera image_003.jpg (demo)"), "{last}");
    assert!(last.contains("residual"), "{last}");
}

#[test]
fn the_committed_pose_crosses_an_aligned_node_s_transform() {
    let (mut state, mut viewer, _id) = measured();
    align(&mut state);
    let image = ImageRef::new(state.scene[0].id, IMAGE);
    viewer.jump_to_camera_view(image, &state.scene[0]);
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.camera.pan(35.0, -20.0, 800.0, 600.0);
    let drawn_centre = viewer.camera.camera.position;

    commit(&mut viewer, &mut state).expect("a movable camera");

    // What the value holds is the *node's* pose; where it is drawn is where the
    // reviewer put it.
    let landed = stored_pose(&state);
    let redrawn = state.scene[0]
        .transform
        .apply_to_point(&Point3::from(landed.translation));
    assert!(
        (redrawn - drawn_centre).norm() < 1e-9,
        "the transform was not divided out: {redrawn:?} vs {drawn_centre:?}"
    );
}

#[test]
fn cancelling_pushes_nothing_and_puts_the_viewport_back() {
    let (mut state, mut viewer, _id) = plain();
    let before_position = viewer.camera.camera.position;
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.camera.fly_move(1.0, 0.5, 0.25);

    cancel(&mut viewer, &mut state);

    assert!(viewer.camera_lock.is_none());
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!((viewer.camera.camera.position - before_position).norm() < 1e-9);
    let last = texts(&state).pop().expect("a log entry");
    assert!(last.contains("Cancelled the camera move"), "{last}");
}

#[test]
fn a_commit_inside_the_dead_band_pushes_nothing_and_says_so() {
    let (mut state, mut viewer, _id) = plain();
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    // A rotation well under a hundredth of a degree.
    viewer.camera.tilt(1e-7);

    let moved = commit(&mut viewer, &mut state).expect("a movable camera");

    assert_eq!(moved, None);
    assert_eq!(state.scene[0].history.versions().len(), 1);
    let last = texts(&state).pop().expect("a log entry");
    assert!(last.contains("was not moved"), "{last}");
}

#[test]
fn an_implicit_exit_commits_a_move_and_drops_a_dead_band() {
    let (mut state, mut viewer, id) = plain();
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    assert_eq!(exit_implicitly(&mut viewer, &mut state), None);
    assert_eq!(state.scene[0].history.versions().len(), 1);

    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.camera.nodal_pan(50.0, 0.0);
    assert_eq!(exit_implicitly(&mut viewer, &mut state), Some(id));
    assert_eq!(state.scene[0].history.versions().len(), 2);
    // Nothing held afterwards, either way.
    assert!(viewer.camera_lock.is_none());
    assert_eq!(exit_implicitly(&mut viewer, &mut state), None);
}

#[test]
fn undo_of_a_move_puts_the_viewport_back_on_the_restored_pose() {
    let (mut state, mut viewer, id) = plain();
    let before = stored_pose(&state);
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.camera.nodal_pan(60.0, 20.0);
    commit(&mut viewer, &mut state).expect("a movable camera");
    let after_commit = pending_pose(&viewer, &state.scene[0]);
    assert!(
        apart(&after_commit, &before).0 > 1.0,
        "the commit moved nothing, so the re-snap has nothing to undo"
    );

    state.undo(id).expect("one edit to undo");
    resnap_camera_view(&mut viewer, &state);

    let restored = stored_pose(&state);
    let (degrees, distance) = apart(&restored, &before);
    assert!(
        degrees < 1e-9 && distance < 1e-9,
        "the undo left the pose moved"
    );
    let pending = pending_pose(&viewer, &state.scene[0]);
    let (degrees, distance) = apart(&pending, &before);
    assert!(
        degrees < 1e-9 && distance < 1e-9,
        "the viewport stayed where the hand left it: {degrees} deg, {distance}"
    );
}

// ── The keys ────────────────────────────────────────────────────────────

/// Run one headless frame that presses `key`, and report what the lock's key
/// handler did with it.
fn press(viewer: &mut Viewer3D, state: &mut AppState, key: eframe::egui::Key) -> Option<ReconId> {
    let ctx = eframe::egui::Context::default();
    let input = eframe::egui::RawInput {
        events: vec![eframe::egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: eframe::egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let mut moved = None;
    crate::test_support::run_frame_headless(&ctx, input, |ui| {
        moved = handle_keys(ui, viewer, state);
    });
    moved
}

#[test]
fn m_takes_the_camera_in_hand_and_m_again_commits_it() {
    let (mut state, mut viewer, id) = plain();
    press(&mut viewer, &mut state, eframe::egui::Key::M);
    assert!(viewer.camera_lock.is_some(), "M did not enter the lock");

    viewer.camera.nodal_pan(45.0, 10.0);
    assert_eq!(
        press(&mut viewer, &mut state, eframe::egui::Key::M),
        Some(id)
    );
    assert!(viewer.camera_lock.is_none());
    assert_eq!(state.scene[0].history.versions().len(), 2);
}

#[test]
fn enter_commits_and_escape_cancels() {
    let (mut state, mut viewer, id) = plain();
    press(&mut viewer, &mut state, eframe::egui::Key::M);
    viewer.camera.nodal_pan(45.0, 10.0);
    assert_eq!(
        press(&mut viewer, &mut state, eframe::egui::Key::Enter),
        Some(id)
    );
    assert_eq!(state.scene[0].history.versions().len(), 2);

    press(&mut viewer, &mut state, eframe::egui::Key::M);
    viewer.camera.nodal_pan(45.0, 10.0);
    press(&mut viewer, &mut state, eframe::egui::Key::Escape);
    assert!(viewer.camera_lock.is_none());
    assert_eq!(
        state.scene[0].history.versions().len(),
        2,
        "a cancel pushed a version"
    );
}

#[test]
fn stepping_to_the_next_image_commits_first() {
    let (mut state, mut viewer, id) = plain();
    press(&mut viewer, &mut state, eframe::egui::Key::M);
    viewer.camera.nodal_pan(45.0, 10.0);
    assert_eq!(
        press(&mut viewer, &mut state, eframe::egui::Key::Period),
        Some(id)
    );
    assert!(viewer.camera_lock.is_none());
    assert_eq!(state.scene[0].history.versions().len(), 2);
}

// ── The highlight, and the photograph ───────────────────────────────────

#[test]
fn the_lock_highlights_exactly_the_points_this_image_observes() {
    let (mut state, mut viewer, _id) = plain();
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    let lock = viewer.camera_lock.as_ref().expect("a held lock");
    let expected: std::collections::HashSet<u32> = state.scene[0]
        .recon()
        .point_set
        .tracks
        .iter()
        .filter(|row| row.image_index as usize == IMAGE)
        .map(|row| row.point_index)
        .collect();
    assert!(!expected.is_empty());
    assert_eq!(lock.highlighted, expected);
}

#[test]
fn the_photograph_turns_with_the_camera_and_starts_at_the_node_s_own_transform() {
    let (mut state, mut viewer, _id) = plain();
    align(&mut state);
    let image = ImageRef::new(state.scene[0].id, IMAGE);
    viewer.jump_to_camera_view(image, &state.scene[0]);
    enter(&mut viewer, &mut state).expect("camera view of a posed image");

    // At the snap the background is drawn exactly as it is with no lock at all.
    let at_entry = background_transform(&viewer, &state.scene[0]).expect("a held lock");
    let (degrees, _) = apart(&at_entry, &state.scene[0].transform);
    assert!(
        degrees < 1e-9,
        "the entry turned the photograph by {degrees}"
    );

    viewer.camera.nodal_pan(80.0, 0.0);
    let turned = background_transform(&viewer, &state.scene[0]).expect("a held lock");
    let (degrees, _) = apart(&turned, &state.scene[0].transform);
    assert!(degrees > 1.0, "the photograph stayed behind: {degrees} deg");
}

#[test]
fn the_lock_is_viewport_state_and_leaves_the_value_alone() {
    let (mut state, mut viewer, _id) = plain();
    let before = Arc::clone(&state.scene[0].edited().base);
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.camera.nodal_pan(50.0, 20.0);
    assert!(
        Arc::ptr_eq(&before, &state.scene[0].edited().base),
        "holding the lock changed the value"
    );
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

#[test]
fn a_lock_whose_node_is_closed_under_it_is_dropped_rather_than_left_held() {
    let (mut state, mut viewer, id) = plain();
    enter(&mut viewer, &mut state).expect("camera view of a posed image");
    viewer.camera.nodal_pan(50.0, 0.0);

    state.close_node(id);

    // Nothing to commit the pose to, so the lock goes and the next `M` is free
    // to take another camera in hand.
    assert_eq!(exit_implicitly(&mut viewer, &mut state), None);
    assert!(viewer.camera_lock.is_none());
}
