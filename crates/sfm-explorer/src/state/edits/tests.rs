// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two edits end to end: what each does to the node's value, what the
//! selection does across it and back, and what the Action Log says.
//!
//! The GPU consequences are asserted separately, in
//! `scene_renderer/upload/tests.rs`, against a real `wgpu` device.

use std::sync::Arc;

use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::SfmrReconstruction;

use crate::action_log::Entry;
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::PointGesture;
use crate::state::AppState;
use crate::test_support::{assert_timed_from_the_work, phase_rows};

/// The newest Action Log entry, which is the one the operation just driven
/// wrote.
fn newest(state: &AppState) -> &Entry {
    state.action_log.entries().next_back().expect("an entry")
}

/// A state holding one demo node, selected.
fn state() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(64)));
    state
}

/// The id of the one node.
fn node(state: &AppState) -> ReconId {
    state.selected_recon.expect("a selected reconstruction")
}

/// The Action Log's texts, oldest first.
fn texts(state: &AppState) -> Vec<String> {
    state
        .action_log
        .entries()
        .map(|entry| entry.text.clone())
        .collect()
}

// ── Delete point: the point edit ────────────────────────────────────────

#[test]
fn deleting_a_point_leaves_the_base_alone_and_costs_one_index() {
    let mut state = state();
    let id = node(&state);
    let before = Arc::clone(&state.scene[0].edited().base);
    let count = state.scene[0].point_count();

    state.selected_point = Some(PointRef::new(id, 7));
    state
        .delete_point(PointRef::new(id, 7))
        .expect("a live point");

    let node = &state.scene[0];
    assert!(
        Arc::ptr_eq(&before, &node.edited().base),
        "a point edit wrote through the base"
    );
    assert_eq!(node.point_count(), count - 1);
    assert!(node.is_point_deleted(7));
    assert!(!node.is_point_deleted(8), "an index after the hole moved");
    assert_eq!(node.history.versions().len(), 2);
}

#[test]
fn deleting_the_selected_point_clears_the_selection_and_an_undo_does_not_guess() {
    let mut state = state();
    let id = node(&state);
    state.selected_point = Some(PointRef::new(id, 3));

    state.delete_selected_point().expect("a live point");
    assert_eq!(
        state.selected_point, None,
        "a deleted point stayed selected"
    );

    state.undo(id).expect("one edit to undo");
    assert!(!state.scene[0].is_point_deleted(3));
    // The selection is not restored: the map says where an index went, not
    // what the user was looking at.
    assert_eq!(state.selected_point, None);
}

#[test]
fn a_surviving_selection_keeps_its_index_across_a_point_edit() {
    let mut state = state();
    let id = node(&state);
    state.selected_point = Some(PointRef::new(id, 9));
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    assert_eq!(state.selected_point, Some(PointRef::new(id, 9)));
}

#[test]
fn deleting_a_point_twice_is_refused_and_leaves_the_history_alone() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    let versions = state.scene[0].history.versions().len();
    assert!(state.delete_point(PointRef::new(id, 1)).is_err());
    assert_eq!(state.scene[0].history.versions().len(), versions);
}

// ── Delete image: the bulk edit ─────────────────────────────────────────

#[test]
fn deleting_an_image_produces_a_new_base_with_an_empty_overlay() {
    let mut state = state();
    let id = node(&state);
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let name = state.scene[0].recon().image_table.images[1].name.clone();

    state.delete_image(ImageRef::new(id, 1)).expect("an image");

    let node = &state.scene[0];
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "the base is the same value"
    );
    assert!(node.edited().deleted_points.is_empty());
    assert!(node.edited().added.points.is_empty());
    assert_eq!(node.image_count(), images - 1);
    assert!(node
        .recon()
        .image_table
        .images
        .iter()
        .all(|image| image.name != name));
    assert_eq!(node.id, id, "a node's identity did not survive an edit");
}

#[test]
fn deleting_an_image_after_a_point_edit_folds_the_overlay_in() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 0))
        .expect("a live point");
    let after_point_edit = state.scene[0].point_count();

    state.delete_image(ImageRef::new(id, 0)).expect("an image");

    let node = &state.scene[0];
    assert!(node.edited().deleted_points.is_empty());
    // The materialisation closed the hole the point edit made, so the new
    // base holds at most what the overlay did.
    assert!(node.point_count() <= after_point_edit);
    assert_eq!(node.history.versions().len(), 3);
}

#[test]
fn the_selection_follows_the_renumbering_and_comes_back_through_the_undo() {
    let mut state = state();
    let id = node(&state);
    // A point the deleted image does not orphan, so it survives the edit.
    let survivor = surviving_point(&state, 0);
    let position = state.scene[0].recon().point_set.points[survivor as usize].position;
    state.selected_point = Some(PointRef::new(id, survivor as usize));

    state.delete_image(ImageRef::new(id, 0)).expect("an image");
    let moved = state.selected_point.expect("the point survived");
    assert_eq!(
        state.scene[0].recon().point_set.points[moved.index()].position,
        position,
        "the selection followed the map onto a different point"
    );

    state.undo(id).expect("one edit to undo");
    assert_eq!(
        state.selected_point,
        Some(PointRef::new(id, survivor as usize)),
        "the selection did not come back through the map"
    );
}

#[test]
fn the_last_image_cannot_be_deleted() {
    let mut state = AppState::new();
    let mut recon = SfmrReconstruction::demo(16);
    recon = recon
        .subset_by_image_indices(&[0], false)
        .expect("a one-image subset");
    state.append_node(SceneNode::demo(recon));
    let id = node(&state);
    assert!(state.delete_image(ImageRef::new(id, 0)).is_err());
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

/// A point of `state`'s node that image `image` does not orphan: it has an
/// observation somewhere else.
fn surviving_point(state: &AppState, image: u32) -> u32 {
    let recon = state.scene[0].recon();
    let mut elsewhere = vec![false; recon.point_count()];
    for observation in &recon.point_set.tracks {
        if observation.image_index != image {
            elsewhere[observation.point_index as usize] = true;
        }
    }
    elsewhere
        .iter()
        .position(|&alive| alive)
        .expect("some point is seen by another image") as u32
}

// ── Undo, redo and the log ──────────────────────────────────────────────

#[test]
fn undo_and_redo_walk_the_versions_and_refuse_at_the_ends() {
    let mut state = state();
    let id = node(&state);
    assert!(!state.can_undo(id) && !state.can_redo(id));

    state
        .delete_point(PointRef::new(id, 4))
        .expect("a live point");
    assert!(state.can_undo(id) && !state.can_redo(id));

    state.undo(id).expect("one edit to undo");
    assert!(!state.can_undo(id) && state.can_redo(id));
    assert!(state.undo(id).is_err());
    assert!(!state.scene[0].is_point_deleted(4));

    state.redo(id).expect("a redo tail");
    assert!(state.scene[0].is_point_deleted(4));
    assert!(state.redo(id).is_err());
}

#[test]
fn every_edit_undo_and_redo_writes_one_log_entry_naming_the_node_and_the_serials() {
    let mut state = state();
    let id = node(&state);
    state.action_log.clear();

    state
        .delete_point(PointRef::new(id, 5))
        .expect("a live point");
    state.undo(id).expect("one edit to undo");
    state.redo(id).expect("a redo tail");

    let texts = texts(&state);
    assert_eq!(texts.len(), 3, "{texts:?}");
    assert!(
        texts[0].starts_with("Deleted point 5 in demo ("),
        "{}",
        texts[0]
    );
    assert!(
        texts[1].starts_with("Undo: Deleted point 5 in demo ("),
        "{}",
        texts[1]
    );
    assert!(
        texts[2].starts_with("Redo: Deleted point 5 in demo ("),
        "{}",
        texts[2]
    );
    // Each entry names the step it took, as `vN → vM`.
    for text in &texts {
        assert!(text.contains(" → v"), "{text}");
    }
    assert!(state
        .action_log
        .entries()
        .all(|entry| entry.kind == crate::action_log::Kind::Edit));
}

/// What a cursor move is made of, under the name of the move itself.
#[test]
fn undo_and_redo_name_the_step_the_selection_and_the_caches() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 5))
        .expect("a live point");

    state.undo(id).expect("one edit to undo");
    assert_eq!(
        phase_rows(&newest(&state).detail),
        [
            ("undo", 0, 1),
            ("history step", 1, 1),
            ("selection follow", 1, 1),
            ("forget images", 1, 1),
        ],
        "{:?}",
        newest(&state).detail,
    );
    assert_timed_from_the_work(&mut state.action_log);

    state.redo(id).expect("a redo tail");
    assert_eq!(
        phase_rows(&newest(&state).detail),
        [
            ("redo", 0, 1),
            ("history step", 1, 1),
            ("selection follow", 1, 1),
            ("forget images", 1, 1),
        ],
        "{:?}",
        newest(&state).detail,
    );
}

/// A jump is a run of steps, so each stage is one row saying how many times it
/// ran rather than one row per version passed.
#[test]
fn a_jump_folds_one_row_per_stage_over_the_steps_it_walked() {
    let mut state = state();
    let id = node(&state);
    for point in [1usize, 2, 3] {
        state
            .delete_point(PointRef::new(id, point))
            .expect("a live point");
    }
    let first = serial_at(&state, 1);

    state.jump_to_version(id, first).expect("a live version");

    assert_eq!(
        phase_rows(&newest(&state).detail),
        [
            ("go to", 0, 1),
            ("history step", 1, 2),
            ("selection follow", 1, 2),
            ("forget images", 1, 1),
        ],
        "{:?}",
        newest(&state).detail,
    );
    assert_timed_from_the_work(&mut state.action_log);
}

#[test]
fn an_edit_after_an_undo_discards_the_redo_tail() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    state.undo(id).expect("one edit to undo");
    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");

    assert!(!state.can_redo(id));
    let node = &state.scene[0];
    assert!(node.is_point_deleted(1));
    assert!(!node.is_point_deleted(2), "the discarded version came back");
    assert!(node.is_point_deleted(3));
}

// ── Jumping to a version ────────────────────────────────────────────────

/// The serial of the node's version at `position`.
fn serial_at(state: &AppState, position: usize) -> crate::document::VersionSerial {
    state.scene[0].history.versions()[position].serial
}

#[test]
fn a_jump_back_lands_on_the_version_a_run_of_undos_would_have() {
    let mut state = state();
    let id = node(&state);
    for point in [1u32, 2, 3] {
        state
            .delete_point(PointRef::new(id, point as usize))
            .expect("a live point");
    }
    let first = serial_at(&state, 1);

    state.jump_to_version(id, first).expect("a live version");

    let node = &state.scene[0];
    assert_eq!(node.history.current_version().serial, first);
    assert!(node.is_point_deleted(1));
    assert!(!node.is_point_deleted(2));
    assert!(!node.is_point_deleted(3));
    // The versions themselves did not move: a jump walks the cursor.
    assert_eq!(node.history.versions().len(), 4);
    assert_eq!(node.history.cursor(), 1);
}

#[test]
fn a_jump_forward_returns_to_the_version_it_came_from() {
    let mut state = state();
    let id = node(&state);
    for point in [1u32, 2, 3] {
        state
            .delete_point(PointRef::new(id, point as usize))
            .expect("a live point");
    }
    let last = serial_at(&state, 3);
    let first = serial_at(&state, 1);
    state.jump_to_version(id, first).expect("a live version");

    state.jump_to_version(id, last).expect("a live version");

    let node = &state.scene[0];
    assert_eq!(node.history.current_version().serial, last);
    for point in [1u32, 2, 3] {
        assert!(node.is_point_deleted(point));
    }
}

#[test]
fn a_jump_takes_the_selection_through_every_step_it_passes() {
    let mut state = state();
    let id = node(&state);
    let survivor = surviving_point(&state, 0);
    let position = state.scene[0].recon().point_set.points[survivor as usize].position;
    state.selected_point = Some(PointRef::new(id, survivor as usize));
    let loaded = serial_at(&state, 0);

    // A point edit and then a bulk edit, so the jump back composes a
    // stable-index step with a renumbering one.
    state
        .delete_point(PointRef::new(id, survivor as usize + 1))
        .expect("a live point");
    state.delete_image(ImageRef::new(id, 0)).expect("an image");
    let moved = state.selected_point.expect("the point survived");
    assert_eq!(
        state.scene[0].recon().point_set.points[moved.index()].position,
        position
    );

    state.jump_to_version(id, loaded).expect("a live version");

    assert_eq!(
        state.selected_point,
        Some(PointRef::new(id, survivor as usize)),
        "the jump did not compose the maps a run of undos would have"
    );
}

#[test]
fn a_jump_writes_one_log_entry_naming_the_two_serials() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    let from = serial_at(&state, 2);
    let to = serial_at(&state, 0);
    state.action_log.clear();

    state.jump_to_version(id, to).expect("a live version");

    let texts = texts(&state);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(texts[0], format!("Go to: Opened demo ({from} → {to})"));
    assert!(state
        .action_log
        .entries()
        .all(|entry| entry.kind == crate::action_log::Kind::Edit));
}

#[test]
fn a_jump_onto_a_released_version_is_refused_whole() {
    let mut state = state();
    let id = node(&state);
    for point in [1u32, 2, 3] {
        state
            .delete_point(PointRef::new(id, point as usize))
            .expect("a live point");
    }
    let released = serial_at(&state, 1);
    let earliest = serial_at(&state, 0);
    state.scene[0].history.versions_mut_for_test()[1].value = None;
    state.action_log.clear();

    // The released version itself, and any version the walk would pass
    // through it to reach.
    assert!(state.jump_to_version(id, released).is_err());
    assert!(state.jump_to_version(id, earliest).is_err());
    assert_eq!(
        state.scene[0].history.cursor(),
        3,
        "the cursor moved anyway"
    );
    assert_eq!(texts(&state), Vec::<String>::new());
}

#[test]
fn a_jump_to_where_the_cursor_already_is_or_to_no_version_is_refused() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    let here = serial_at(&state, 1);
    assert!(state.jump_to_version(id, here).is_err());

    // A serial of another node's history is not one of this node's versions.
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(8)));
    let other = state.scene[1].history.current_version().serial;
    assert!(state.jump_to_version(id, other).is_err());
}

// ── The embedded_patches fixtures ────────────────────────────────────────

/// A demo node whose value is `embedded_patches`. No images behind it, so a
/// photometric kernel is exercised in `sfmtool-core` and through the Python
/// binding rather than here.
fn embedded_state() -> AppState {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(embedded_demo(64)));
    state
}

/// The demo reconstruction with its observations rewritten as inline keypoints
/// and a patch frame per point.
///
/// Built here rather than through `to_embedded_patches`, which reads the `.sift`
/// companions the demo value has no files for.
/// The in-plane half-extent of the patch frame the two demo fixtures carry, in
/// world units. At the demo's focal of 1000 and its cameras' distance of
/// roughly 5, it projects to a few pixels.
pub(crate) const PATCH_HALF_EXTENT: f32 = 0.02;

pub(crate) fn embedded_demo(points: usize) -> SfmrReconstruction {
    use ndarray::Array2;
    use sfmtool_core::ObservationSource;

    let mut recon = SfmrReconstruction::demo(points);
    let images = recon.image_count();
    let observations = recon.point_set.tracks.len();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::<f32>::from_shape_fn((observations, 2), |(i, c)| {
            (i % 64) as f32 + c as f32
        }),
        image_file_hashes: vec![[0u8; 16]; images],
    };
    let p = recon.point_count();
    let mut u = Array2::<f32>::zeros((p, 3));
    let mut v = Array2::<f32>::zeros((p, 3));
    for i in 0..p {
        u[[i, 0]] = PATCH_HALF_EXTENT;
        v[[i, 1]] = PATCH_HALF_EXTENT;
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.metadata.feature_source = "embedded_patches".to_string();
    recon.rebuild_derived_fields();
    recon
}

// ── The default patch radius ───────────────────────────

#[test]
fn the_default_radius_is_the_median_the_reconstruction_already_uses() {
    let state = embedded_state();
    let id = node(&state);
    let derived = state.default_patch_radius(ImageRef::new(id, 0));
    assert!(
        derived > 0.0 && derived != super::FALLBACK_PATCH_RADIUS_PX,
        "a node with patch frames derives its own radius, got {derived}"
    );
}

#[test]
fn a_node_with_no_patch_frames_falls_back_to_the_named_radius() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(8)));
    let id = node(&state);
    assert_eq!(
        state.default_patch_radius(ImageRef::new(id, 0)),
        super::FALLBACK_PATCH_RADIUS_PX
    );
}

// ── The projected embedded_patches fixture ─────────────────────

/// [`embedded_demo`] with every keypoint the exact projection of its point, and
/// track lengths of one, two and three cycling by point index.
///
/// The projections are what make the re-triangulation well-conditioned, and the
/// three lengths are the three outcomes: point 0 is deleted by the edit, point 1
/// becomes a bearing, point 2 stays finite and is re-solved.
///
/// Each patch frame is turned to face the cameras that observe it, the way
/// `to_embedded_patches` frames one. [`embedded_demo`]'s world-axis-aligned
/// frame is seen nearly edge-on from most of the demo's ring of cameras, and
/// its two columns then project almost parallel -- a footprint a few tenths of
/// a pixel across in one direction, which no photometric kernel can register
/// and which would make a refusal here the fixture's rather than the code's.
pub(crate) fn projected_embedded_demo(points: usize) -> SfmrReconstruction {
    use ndarray::Array2;
    use sfmtool_core::patch::cloud::{mean_viewing_normal, OrientedPatch};
    use sfmtool_core::{ObservationSource, TrackObservation};

    let mut recon = embedded_demo(points);
    let tracks: Vec<TrackObservation> = (0..points)
        .flat_map(|p| {
            (0..=(p % 3)).map(move |image| TrackObservation {
                image_index: image as u32,
                point_index: p as u32,
            })
        })
        .collect();
    let mut keypoints = Array2::<f32>::zeros((tracks.len(), 2));
    for (row, track) in tracks.iter().enumerate() {
        let image = &recon.image_table.images[track.image_index as usize];
        let point = recon.point_set.points[track.point_index as usize].position;
        let cam = image.quaternion_wxyz.to_rotation_matrix() * point.coords + image.translation_xyz;
        let (x, y) = recon.image_table.cameras[image.camera_index as usize]
            .ray_to_pixel([cam.x, cam.y, cam.z])
            .expect("every demo camera sees every demo point");
        keypoints[[row, 0]] = x as f32;
        keypoints[[row, 1]] = y as f32;
    }
    // The frames, turned toward the cameras each point is seen from.
    let half = f64::from(PATCH_HALF_EXTENT);
    let mut u = Array2::<f32>::zeros((points, 3));
    let mut v = Array2::<f32>::zeros((points, 3));
    for p in 0..points {
        let position = recon.point_set.points[p].position;
        let centers: Vec<_> = (0..=(p % 3))
            .map(|image| recon.image_table.images[image].camera_center())
            .collect();
        let normal = mean_viewing_normal(&position, &centers);
        let patch = OrientedPatch::from_center_normal(
            position,
            normal,
            nalgebra::Vector3::z(),
            [half, half],
        );
        for c in 0..3 {
            u[[p, c]] = (patch.u_axis[c] * half) as f32;
            v[[p, c]] = (patch.v_axis[c] * half) as f32;
        }
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);

    let images = recon.image_count();
    recon.point_set.observation_counts = (0..points).map(|p| (p % 3) as u32 + 1).collect();
    recon.point_set.tracks = tracks;
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; images],
    };
    recon.rebuild_derived_fields();
    recon
}

// ── Resect Image: the bulk edit that re-poses one image ─────────────────

/// A state holding one node a resection can run on, with image 1's stored pose
/// pushed off the truth its own keypoints were computed at.
///
/// The fixture is the Scene Graph tests' own, because "a node a resection can
/// run on" is one thing and there is no second answer to it: the demo camera
/// ring with every observation recomputed as the pixel that camera projects the
/// point to.
fn resectable_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    let image = &mut state.scene[0].recon_mut().image_table.images[1];
    image.translation_xyz += nalgebra::Vector3::new(0.30, -0.20, 0.15);
    (state, id)
}

/// How far image `index`'s camera centre stands from `centre`.
fn centre_offset(state: &AppState, index: usize, centre: nalgebra::Point3<f64>) -> f64 {
    (state.scene[0].recon().image_table.images[index].camera_center() - centre).norm()
}

#[test]
fn resecting_pushes_a_version_whose_base_is_new_and_whose_images_stay_put() {
    let (mut state, id) = resectable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr")
        .recon()
        .image_table
        .images[1]
        .camera_center();
    let moved = centre_offset(&state, 1, truth);

    state
        .resect_image(id, 1, crate::resect::ResectFrom::Observations)
        .expect("the ring corroborates image 1");

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert!(
        node.edited().deleted_points.is_empty() && node.edited().added.points.is_empty(),
        "the new version arrived with an overlay"
    );
    assert_eq!(node.image_count(), images, "the image table moved");
    let recovered = centre_offset(&state, 1, truth);
    assert!(
        recovered < moved * 0.1,
        "the resection left the camera {recovered} from the truth, having started {moved}"
    );
}

#[test]
fn the_action_log_carries_one_entry_naming_the_image_and_the_version() {
    let (mut state, id) = resectable_state();
    let entries = texts(&state).len();
    state
        .resect_image(id, 1, crate::resect::ResectFrom::Observations)
        .expect("the ring corroborates image 1");
    let logged = texts(&state);
    assert_eq!(logged.len(), entries + 1, "{logged:?}");
    let last = logged.last().expect("one entry");
    assert!(
        last.starts_with("Resected image_001.jpg (run_a): 120 pts, inliers "),
        "{last}"
    );
    let serials = state.scene[0].history.versions();
    assert!(
        last.ends_with(&format!(
            "re-triangulated ({} → {})",
            serials[0].serial, serials[1].serial
        )),
        "{last}"
    );
    assert_eq!(
        state.scene[0].history.current_version().label,
        "Resected image_001.jpg (run_a)"
    );
}

#[test]
fn an_undo_puts_the_stored_pose_and_the_selection_back() {
    let (mut state, id) = resectable_state();
    let stored = state.scene[0].recon().image_table.images[1].camera_center();
    state.selected_point = Some(PointRef::new(id, 9));

    state
        .resect_image(id, 1, crate::resect::ResectFrom::Observations)
        .expect("the ring corroborates image 1");
    let selected = state.selected_point.expect("the point survived the edit");
    assert_eq!(selected.recon, id);
    assert!(
        (state.scene[0].recon().image_table.images[1].camera_center() - stored).norm() > 1e-6,
        "the pose did not move"
    );

    state.undo(id).expect("one version to undo");
    assert_eq!(
        state.scene[0].recon().image_table.images[1].camera_center(),
        stored
    );
    assert_eq!(state.selected_point, Some(PointRef::new(id, 9)));
}

#[test]
fn a_resection_that_cannot_be_attempted_pushes_no_version_and_logs_a_failure() {
    let (mut state, id) = resectable_state();
    state.scene[0].recon_mut().image_table.images[1].translation_xyz =
        nalgebra::Vector3::new(f64::NAN, 0.0, 0.0);

    let why = state
        .resect_image(id, 1, crate::resect::ResectFrom::Observations)
        .expect_err("an unposed target has no pose to re-estimate");

    assert!(why.contains("refused"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    let last = state.action_log.entries().last().expect("one entry");
    assert!(last.failed, "the refusal was logged as a success");
    assert_eq!(last.text, why);
}

#[test]
fn a_refused_estimate_pushes_no_version_and_logs_a_failure() {
    let (mut state, id) = resectable_state();
    // Image 1's keypoints now agree with nothing: the estimate finds
    // correspondences and no consensus among them, which is the refusal that
    // must not produce a version.
    {
        let recon = state.scene[0].recon_mut();
        let rows: Vec<usize> = recon
            .point_set
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, o)| o.image_index == 1)
            .map(|(row, _)| row)
            .collect();
        let sfmtool_core::ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
            &mut recon.point_set.observations
        else {
            panic!("the fixture is embedded_patches");
        };
        for (k, row) in rows.iter().enumerate() {
            keypoints_xy[[*row, 0]] = (k % 97) as f32 * 13.0;
            keypoints_xy[[*row, 1]] = (k % 53) as f32 * 11.0;
        }
    }

    let why = state
        .resect_image(id, 1, crate::resect::ResectFrom::Observations)
        .expect_err("nothing corroborates that pose");

    assert!(why.contains("refused"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(
        state.action_log.entries().last().expect("one entry").failed,
        "the refusal was logged as a success"
    );
}

#[test]
fn the_matches_source_without_a_chosen_file_reports_itself() {
    let (mut state, id) = resectable_state();

    let why = state
        .resect_image(id, 1, crate::resect::ResectFrom::Matches)
        .expect_err("no .matches file was chosen for this node");

    assert!(why.contains(".matches"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

// ── Bundle adjust: the bulk edit that moves everything ──────────────────

/// The resectable node with image 1 pushed a few pixels off the pose its own
/// keypoints were computed at.
///
/// A few pixels rather than the resection fixture's tens: the adjustment is a
/// local refinement whose first round trims what disagrees by more than 50 px,
/// so a camera perturbed past that is one whose observations leave the solve
/// rather than one it pulls back.
pub(crate) fn adjustable_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    let image = &mut state.scene[0].recon_mut().image_table.images[1];
    image.translation_xyz += nalgebra::Vector3::new(0.02, -0.015, 0.01);
    (state, id)
}

#[test]
fn bundle_adjusting_pushes_a_version_with_a_new_base_and_keeps_the_images() {
    let (mut state, id) = adjustable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr")
        .recon()
        .image_table
        .images[1]
        .camera_center();
    let moved = centre_offset(&state, 1, truth);
    state.selected_point = Some(PointRef::new(id, 11));

    state
        .start_bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("the fixture is well posed");
    state.finish_background_task();

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert_eq!(node.image_count(), images, "the image table moved");
    assert_eq!(
        node.history.current_version().label,
        "Bundle adjusted run_a"
    );
    // The perturbed camera came back toward the pose its own observations were
    // computed at.
    assert!(centre_offset(&state, 1, truth) < moved, "{moved}");
    // The selection followed the map onto a live point.
    let selected = state.selected_point.expect("the point survived");
    assert_eq!(selected.recon, id);
    assert!(state.scene[0].edited().point(selected.point).is_some());
}

#[test]
fn the_log_entry_carries_the_counts_and_the_residuals() {
    let (mut state, id) = adjustable_state();
    let entries = texts(&state).len();

    state
        .start_bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("well posed");
    state.finish_background_task();

    let logged = texts(&state);
    assert_eq!(logged.len(), entries + 1, "{logged:?}");
    let last = logged.last().expect("one entry");
    assert!(
        last.starts_with("Bundle adjusted run_a: 8 images, "),
        "{last}"
    );
    assert!(last.contains("median residual "), "{last}");
    assert!(!last.contains("focal"), "the focal was held: {last}");
    let serials = state.scene[0].history.versions();
    assert!(
        last.ends_with(&format!("({} → {})", serials[0].serial, serials[1].serial)),
        "{last}"
    );
}

#[test]
fn a_released_focal_is_named_in_the_label_and_the_entry() {
    let (mut state, id) = adjustable_state();
    // The demo camera is a two-focal PINHOLE, which the adjustment's focal
    // column is not exact for; a single-focal one of the same geometry is.
    {
        let camera = &mut state.scene[0].recon_mut().image_table.cameras[0];
        let (fx, _) = camera.focal_lengths();
        let (cx, cy) = camera.principal_point();
        camera.model = sfmtool_core::CameraModel::SimplePinhole {
            focal_length: fx,
            principal_point_x: cx,
            principal_point_y: cy,
        };
    }
    let options = sfmtool_core::BundleAdjustOptions {
        opt_f: true,
        ..sfmtool_core::BundleAdjustOptions::default()
    };

    state.start_bundle_adjust(id, &options).expect("well posed");
    state.finish_background_task();

    assert_eq!(
        state.scene[0].history.current_version().label,
        "Bundle adjusted run_a, focal released"
    );
    let last = texts(&state).last().expect("one entry").clone();
    assert!(last.contains(", focal "), "{last}");
}

#[test]
fn an_undo_puts_every_pose_back() {
    let (mut state, id) = adjustable_state();
    let before: Vec<nalgebra::Point3<f64>> = state.scene[0]
        .recon()
        .image_table
        .images
        .iter()
        .map(|i| i.camera_center())
        .collect();

    state
        .start_bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("well posed");
    state.finish_background_task();
    state.undo(id).expect("one version to undo");

    let after: Vec<nalgebra::Point3<f64>> = state.scene[0]
        .recon()
        .image_table
        .images
        .iter()
        .map(|i| i.camera_center())
        .collect();
    assert_eq!(before, after);
}

#[test]
fn a_node_with_no_inline_keypoints_is_refused_before_anything_is_solved() {
    let mut state = state();
    let id = node(&state);

    let why = state
        .start_bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect_err("a sift_files demo carries no keypoints");

    assert!(why.contains("inline keypoints"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    let last = state.action_log.entries().last().expect("one entry");
    assert!(last.failed, "the refusal was logged as a success");
    // The menu entry says the same thing, in the same call.
    assert!(crate::bundle_adjust_prompt::refusal(state.scene[0].edited()).is_some());
}

#[test]
fn a_node_whose_images_disagree_about_the_lens_is_refused() {
    let (mut state, id) = adjustable_state();
    {
        let recon = state.scene[0].recon_mut();
        let second = recon.image_table.cameras[0].clone();
        recon.image_table.cameras.push(second);
        recon.image_table.images[2].camera_index = 1;
    }

    let why = state
        .start_bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect_err("two lenses, one solve");

    assert!(why.contains("one shared camera"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(crate::bundle_adjust_prompt::refusal(state.scene[0].edited()).is_some());
}

#[test]
fn the_gate_and_the_focal_gate_pass_on_a_node_that_can_be_adjusted() {
    let (state, _) = adjustable_state();
    let edited = state.scene[0].edited();
    assert_eq!(crate::bundle_adjust_prompt::refusal(edited), None);
    // The demo camera is a two-focal PINHOLE, whose focal the adjustment cannot
    // release: the checkbox is greyed and says so.
    assert!(crate::bundle_adjust_prompt::focal_refusal(edited).is_some());
}

// ── Move camera: the bulk edit that moves one pose ──────────────────────

#[test]
fn moving_a_camera_pushes_a_new_base_and_leaves_the_image_table_where_it_was() {
    let (mut state, id) = adjustable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let images = state.scene[0].image_count();
    let points = state.scene[0].point_count();
    state.selected_point = Some(PointRef::new(id, 11));
    // Back to the pose image 1's own keypoints were computed at, which is what
    // `adjustable_state` moved it off.
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr");
    let pose = sfmtool_core::reconstruction::move_camera::pose_of(truth.recon(), 1);

    state
        .move_camera(ImageRef::new(id, 1), &pose)
        .expect("a posed image and a finite pose");

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert_eq!(node.image_count(), images, "the image table moved");
    assert_eq!(node.point_count(), points, "the move deleted a point");
    let label = node.history.current_version().label.clone();
    assert!(label.starts_with("Moved camera "), "{label}");
    assert!(label.contains("points re-solved"), "{label}");
    // The identity map: the selection stays on the point it was on, and the
    // value still holds it.
    let selected = state.selected_point.expect("the point survived");
    assert_eq!(selected, PointRef::new(id, 11));
    assert!(state.scene[0].edited().point(11).is_some());
}

#[test]
fn a_move_of_an_image_that_is_not_there_pushes_no_version_and_logs_a_failure() {
    let (mut state, id) = adjustable_state();
    let pose = sfmtool_core::reconstruction::move_camera::pose_of(state.scene[0].recon(), 0);

    let why = state
        .move_camera(ImageRef::new(id, 99), &pose)
        .expect_err("there is no image 99");

    assert!(why.contains("no longer in the reconstruction"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(
        state.action_log.entries().last().expect("one entry").failed,
        "the refusal was logged as a success"
    );
}

#[test]
fn an_undo_of_a_move_puts_the_pose_back() {
    let (mut state, id) = adjustable_state();
    let before = state.scene[0].recon().image_table.images[1].camera_center();
    let truth = crate::scene_graph::tests::resectable_node("/runs/truth.sfmr");
    let pose = sfmtool_core::reconstruction::move_camera::pose_of(truth.recon(), 1);

    state
        .move_camera(ImageRef::new(id, 1), &pose)
        .expect("a posed image and a finite pose");
    let moved = state.scene[0].recon().image_table.images[1].camera_center();
    assert!((moved - before).norm() > 1e-6, "the move moved nothing");

    state.undo(id).expect("one edit to undo");
    let restored = state.scene[0].recon().image_table.images[1].camera_center();
    assert!((restored - before).norm() < 1e-12);
}

// ── What an edit says its time went on ─────────────────────────────────

/// The four stages every bulk edit has, over the one that is not the
/// adjustment: the overlay fold, the kernel, the row map read off its two
/// values, and the version push.
#[test]
fn a_bulk_edit_names_the_fold_the_kernel_the_map_and_the_push() {
    let (mut state, id) = adjustable_state();
    // A point edit first, so there is an overlay to fold and the fold is one of
    // the rows rather than a stage that did not run.
    state
        .delete_point(PointRef::new(id, 7))
        .expect("a live point");

    state
        .resect_image(id, 1, crate::resect::ResectFrom::Observations)
        .expect("the fixture's image 1 resects from its own observations");

    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert_eq!(
        phase_rows(&entry.detail),
        [
            ("materialise", 0, 1),
            ("resect", 0, 1),
            ("row map", 0, 1),
            ("push version", 0, 1)
        ],
        "{:?}",
        entry.detail,
    );
    assert_timed_from_the_work(&mut state.action_log);
}

/// An empty overlay materialises to its own base, so the fold does not run and
/// leaves no row: three stages rather than four.
#[test]
fn a_bulk_edit_with_nothing_to_fold_names_no_materialise() {
    let mut state = state();
    let id = node(&state);

    state.delete_image(ImageRef::new(id, 1)).expect("an image");

    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert_eq!(
        phase_rows(&entry.detail),
        [("subset", 0, 1), ("row map", 0, 1), ("push version", 0, 1)],
        "{:?}",
        entry.detail,
    );
}

// ── What a cursor move does to the image selection ──────────────────────

/// A cursor move carries the photograph on screen with it.
///
/// Undo, redo and a jump are steps through the node's *history*; none of them
/// says anything about what the person is looking at, and a point edit does not
/// touch the image table at all. The one thing that can take the photograph
/// away is a move across an edit that deleted it, and the selection follows by
/// **name**, so an image renumbered by such a move is found again rather than
/// silently swapped for its neighbour.
#[test]
fn a_cursor_move_keeps_the_photograph_that_is_on_screen() {
    let mut state = state();
    let id = node(&state);
    state.select_image(Some(ImageRef::new(id, 3)));
    let camera = state.selected_camera;

    state
        .delete_point(PointRef::new(id, 7))
        .expect("a live point");
    state.undo(id).expect("the point edit");
    assert_eq!(state.selected_image, Some(ImageRef::new(id, 3)));
    assert_eq!(state.selected_camera, camera, "the lens went with it");

    state.redo(id).expect("the redo tail");
    assert_eq!(state.selected_image, Some(ImageRef::new(id, 3)));

    let first = state.scene[0].history.versions()[0].serial;
    state.jump_to_version(id, first).expect("a live version");
    assert_eq!(state.selected_image, Some(ImageRef::new(id, 3)));
}

/// The one move that can take the photograph away: undoing a delete of an
/// *earlier* image renumbers the one on screen, and the selection follows the
/// photograph rather than the index it used to hold.
#[test]
fn a_cursor_move_across_a_deleted_image_follows_the_photograph() {
    let mut state = state();
    let id = node(&state);
    let name = state.image_name(ImageRef::new(id, 5));
    state.select_image(Some(ImageRef::new(id, 5)));

    state
        .delete_image(ImageRef::new(id, 1))
        .expect("a live image");
    assert_eq!(
        state.selected_image,
        Some(ImageRef::new(id, 4)),
        "the delete did not move the selection down with the table"
    );
    assert_eq!(state.image_name(ImageRef::new(id, 4)), name);

    state.undo(id).expect("the delete");
    assert_eq!(
        state.selected_image,
        Some(ImageRef::new(id, 5)),
        "the undo did not put the selection back on the same photograph"
    );

    // And deleting the photograph on screen leaves nothing to show.
    state
        .delete_image(ImageRef::new(id, 5))
        .expect("a live image");
    assert_eq!(state.selected_image, None);
}

// -- Convert to embedded patches: the bulk edit that changes the source ---

/// A `sift_files` node called `run_a` whose workspace is `dir`, with a real
/// `.sift` file per image.
///
/// The demo value is `sift_files` already; what this adds is the companions
/// the conversion reads -- the affine shapes its default `FeatureSize` sizing
/// needs, and the detections it copies as keypoints -- and the workspace
/// metadata saying where they are.
pub(crate) fn convertible_state(dir: &std::path::Path) -> (AppState, ReconId) {
    let mut recon = SfmrReconstruction::demo(24);
    recon.workspace_dir = dir.to_path_buf();
    recon.metadata.workspace.absolute_path = dir.display().to_string();
    recon.metadata.workspace.relative_path = ".".into();
    recon.metadata.workspace.contents.feature_prefix_dir = "features/sift-test".into();
    for image in 0..recon.image_count() {
        let count = recon.point_set.max_track_feature_index[image] as usize + 1;
        // Distinct per feature and per image, so a keypoint the conversion
        // copied can be told from a zero it did not.
        let positions: Vec<[f64; 2]> = (0..count)
            .map(|feature| [100.0 + feature as f64, 200.0 + image as f64])
            .collect();
        let path = recon.sift_path_for_image(image);
        std::fs::create_dir_all(path.parent().expect("a feature directory")).unwrap();
        crate::sift_index::tests::write_sift(
            &path,
            &recon.image_table.images[image].name,
            &vec![vec![0u8; 128]; count],
            &positions,
        );
    }
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(&dir.join("run_a.sfmr"), recon));
    let id = state.scene[0].id;
    (state, id)
}

/// The conversion pushes one version whose base is `embedded_patches`, with
/// every point and every image still there and every index where it was.
#[test]
fn converting_pushes_one_embedded_patches_version_over_the_same_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = convertible_state(dir.path());
    let points = state.scene[0].point_count();
    let images = state.scene[0].image_count();
    let observations = state.scene[0].edited().observation_count();
    let before = Arc::clone(&state.scene[0].edited().base);
    state.selected_point = Some(PointRef::new(id, 11));

    state
        .start_convert_to_embedded_patches(id)
        .expect("the fixture has a .sift file per image");
    state.finish_background_task();

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert_eq!(
        node.recon().point_set.observations.name(),
        "embedded_patches"
    );
    assert_eq!(node.recon().metadata.feature_source, "embedded_patches");
    // What the panels' embedded path needs of the value: a keypoint per
    // observation, and a frame with a real extent to derive each sighting's
    // affine shape from. A zero frame would draw a degenerate ellipse in
    // every overlay that reads one.
    let keypoints = node
        .recon()
        .point_set
        .keypoints_xy()
        .expect("an embedded_patches value carries its keypoints");
    assert_eq!(keypoints.shape(), [observations, 2]);
    let u = node
        .recon()
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("the conversion frames every point");
    let v = node
        .recon()
        .point_set
        .patch_v_halfvec_xyz
        .as_ref()
        .expect("the conversion frames every point");
    for point in 0..points {
        for half in [u, v] {
            let norm: f32 = (0..3).map(|c| half[[point, c]] * half[[point, c]]).sum();
            assert!(
                norm > 0.0,
                "point {point} was framed with a zero half-vector"
            );
        }
    }
    assert_eq!(node.point_count(), points);
    assert_eq!(node.image_count(), images);
    assert_eq!(node.edited().observation_count(), observations);
    assert_eq!(
        node.history.current_version().label,
        "Converted run_a to embedded patches"
    );
    // The map is the identity, so the selection stays on the index it was on
    // and that index is still a live point.
    assert_eq!(state.selected_point, Some(PointRef::new(id, 11)));
    assert!(state.scene[0].edited().point(11).is_some());
    let map = state.scene[0]
        .history
        .map_into_cursor()
        .expect("the version carries its map");
    for point in 0..points as u32 {
        assert_eq!(map.forward(point), Some(point), "point {point} moved");
        assert_eq!(
            map.inverse(point),
            Some(point),
            "point {point} came from elsewhere"
        );
    }
    // The frames are there and the bitmaps are not: the minimal conversion
    // fuses no reference texture, so the surfel renderer still has nothing to
    // draw.
    assert!(!state.scene[0].has_patch_data());
}

/// The entry says what it did, with the two counts, and the sentence ends on
/// the serials the frame stamped.
#[test]
fn the_conversions_entry_carries_the_counts() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = convertible_state(dir.path());
    let entries = texts(&state).len();

    state
        .start_convert_to_embedded_patches(id)
        .expect("well posed");
    state.finish_background_task();

    let logged = texts(&state);
    assert_eq!(logged.len(), entries + 1, "{logged:?}");
    let last = logged.last().expect("one entry");
    assert!(
        last.starts_with("Converted run_a to embedded patches: 24 points framed, 8 images read ("),
        "{last}"
    );
    let serials = state.scene[0].history.versions();
    assert!(
        last.ends_with(&format!(
            "({} \u{2192} {})",
            serials[0].serial, serials[1].serial
        )),
        "{last}"
    );
}

/// Undo puts the `sift_files` version back, with its feature indexes and no
/// patch frame.
#[test]
fn an_undo_restores_the_sift_files_version() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = convertible_state(dir.path());

    state
        .start_convert_to_embedded_patches(id)
        .expect("well posed");
    state.finish_background_task();
    state.undo(id).expect("one version to undo");

    let recon = state.scene[0].recon();
    assert_eq!(recon.point_set.observations.name(), "sift_files");
    assert!(recon.point_set.feature_indexes().is_some());
    assert!(recon.point_set.patch_u_halfvec_xyz.is_none());
}

/// A node that is already `embedded_patches` is refused, in the sentence the
/// greyed menu entry carries, and nothing is pushed.
#[test]
fn an_embedded_patches_node_is_refused() {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;

    assert_eq!(
        state.convert_to_embedded_patches_refusal(id).as_deref(),
        Some("run_a is already an embedded_patches reconstruction.")
    );
    let why = state
        .start_convert_to_embedded_patches(id)
        .expect_err("there is no .sift to copy from");
    assert!(why.contains("already an embedded_patches"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(newest(&state).failed);
}

/// A node with no `.sift` companion fails in the kernel's own words, and
/// pushes nothing.
#[test]
fn a_missing_sift_file_fails_the_operation() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = convertible_state(dir.path());
    // Every companion removed, so the sizing step has nothing to read.
    std::fs::remove_dir_all(dir.path().join("features")).unwrap();

    state
        .start_convert_to_embedded_patches(id)
        .expect("nothing refused it before it ran");
    state.finish_background_task();

    let entry = newest(&state);
    assert!(entry.failed, "{}", entry.text);
    assert!(
        entry
            .text
            .starts_with("Convert to embedded patches of run_a refused: ")
            && entry.text.contains("building patch frames failed"),
        "{}",
        entry.text
    );
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

// ── Retriangulate: the point edit and its whole-value sibling ───────────

/// The resectable node with one point pushed off the place its own keypoints
/// were computed at, so a retriangulation has somewhere to pull it back to.
fn nudged_point_state(point: u32) -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    state.scene[0].recon_mut().point_set.points[point as usize].position +=
        nalgebra::Vector3::new(0.05, -0.04, 0.03);
    (state, id)
}

/// Where the fixture's pixels were computed from, which is where a
/// retriangulation of that point has to head back toward.
fn truth_position(point: u32) -> nalgebra::Point3<f64> {
    crate::scene_graph::tests::resectable_node("/runs/truth.sfmr")
        .recon()
        .point_set
        .points[point as usize]
        .position
}

/// Where `state`'s node holds the point that started life at index `origin`.
fn live_position(state: &AppState, origin: u32) -> nalgebra::Point3<f64> {
    let edited = state.scene[0].edited();
    let index = edited
        .live_index_of_base(origin)
        .expect("the point is still somewhere");
    edited.point(index).expect("a live point").point().position
}

#[test]
fn retriangulating_a_point_leaves_the_base_alone_and_moves_only_that_point() {
    let (mut state, id) = nudged_point_state(11);
    let before = Arc::clone(&state.scene[0].edited().base);
    let truth = truth_position(11);
    let neighbour = before.point_set.points[12].position;
    let away = (live_position(&state, 11) - truth).norm();

    state
        .retriangulate_point(PointRef::new(id, 11))
        .expect("the fixture retriangulates");

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        Arc::ptr_eq(&before, &node.edited().base),
        "a point edit rewrote its input's base"
    );
    assert_eq!(node.point_count(), before.point_count());
    // The nudged point came back toward the place its own pixels state, and
    // nothing beside it moved.
    assert!((live_position(&state, 11) - truth).norm() < away, "{away}");
    assert_eq!(live_position(&state, 12), neighbour);
}

#[test]
fn a_retriangulated_point_takes_a_new_index_the_selection_follows() {
    let (mut state, id) = nudged_point_state(11);
    state.selected_point = Some(PointRef::new(id, 11));

    state
        .retriangulate_point(PointRef::new(id, 11))
        .expect("the fixture retriangulates");

    let selected = state.selected_point.expect("the point survived");
    assert_eq!(selected.recon, id);
    assert_ne!(selected.point, 11, "delete-and-re-add reuses no index");
    assert!(state.scene[0].edited().point(selected.point).is_some());
}

#[test]
fn the_point_entry_names_the_version_and_the_verdict() {
    let (mut state, id) = nudged_point_state(11);
    state
        .retriangulate_point(PointRef::new(id, 11))
        .expect("the fixture retriangulates");

    assert_eq!(
        state.scene[0].history.current_version().label,
        "Retriangulated point 11 in run_a"
    );
    let text = newest(&state).text.clone();
    assert!(
        text.starts_with("Retriangulated point 11 in run_a: "),
        "{text}"
    );
    assert!(text.contains("finite"), "{text}");
    // The two stages a point edit has, and the core operation's three
    // underneath the first of them.
    let rows = phase_rows(&newest(&state).detail);
    assert!(
        rows.iter().any(|&(name, ..)| name == "retriangulate"),
        "{rows:?}"
    );
    assert!(
        rows.iter().any(|&(name, ..)| name == "push version"),
        "{rows:?}"
    );
    assert_timed_from_the_work(&mut state.action_log);
}

#[test]
fn undo_puts_the_point_back_where_it_was() {
    let (mut state, id) = nudged_point_state(11);
    let before = live_position(&state, 11);
    state
        .retriangulate_point(PointRef::new(id, 11))
        .expect("the fixture retriangulates");
    assert_ne!(live_position(&state, 11), before);
    state.undo(id).expect("one version to take back");
    assert_eq!(live_position(&state, 11), before);
}

#[test]
fn retriangulating_a_point_a_reconstruction_no_longer_holds_is_refused() {
    let (mut state, id) = nudged_point_state(11);
    let count = state.scene[0].point_count();
    let why = state
        .retriangulate_point(PointRef::new(id, count + 5))
        .expect_err("that index names nothing");
    assert!(why.contains("Cannot retriangulate point"), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert!(newest(&state).failed);
}

#[test]
fn retriangulating_every_point_pushes_one_version_with_a_new_base() {
    let (mut state, id) = nudged_point_state(11);
    let before = Arc::clone(&state.scene[0].edited().base);
    let truth = truth_position(11);
    let away = (live_position(&state, 11) - truth).norm();
    let count = state.scene[0].point_count();
    state.selected_point = Some(PointRef::new(id, 11));

    state
        .start_retriangulate_all_points(id)
        .expect("the fixture is well posed");
    state.finish_background_task();

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert_eq!(node.point_count(), count, "no point was deleted or created");
    assert_eq!(node.history.current_version().label, "Retriangulated run_a");
    assert!((live_position(&state, 11) - truth).norm() < away, "{away}");
    // No index moved, so the selection is where it was.
    assert_eq!(state.selected_point, Some(PointRef::new(id, 11)));
    let entry = newest(&state);
    assert!(
        entry.text.starts_with("Retriangulated run_a: "),
        "{}",
        entry.text
    );
    assert!(entry.text.contains("points moved"), "{}", entry.text);
}

#[test]
fn a_cancelled_whole_value_retriangulation_pushes_no_version() {
    let (mut state, id) = nudged_point_state(11);
    let before = live_position(&state, 11);
    state
        .start_retriangulate_all_points(id)
        .expect("the fixture is well posed");
    state.cancel_background_task();
    state.finish_background_task();

    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert_eq!(live_position(&state, 11), before);
    let entry = newest(&state);
    assert!(entry.failed, "{}", entry.text);
    assert!(
        entry
            .text
            .contains("Retriangulate all points of run_a cancelled"),
        "{}",
        entry.text
    );
}

#[test]
fn a_reconstruction_with_no_pixels_is_refused_in_the_menu_s_own_words() {
    // The demo value states no inline keypoint, so there is no pixel to cast a
    // ray through -- and the entry's greying and the call's refusal are the
    // same sentence.
    let mut state = state();
    let id = node(&state);
    let why = state
        .retriangulate_refusal(id)
        .expect("the demo value carries no keypoints");
    assert!(why.contains("no inline"), "{why}");
    let refused = state
        .start_retriangulate_all_points(id)
        .expect_err("with no pixels there is nothing to solve from");
    assert!(refused.contains(&why), "{refused}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

// ── Prune covered observations: the bulk edit over the footprints ──────

/// A state holding one node a prune can run on: the resectable node with a
/// patch frame per point.
fn prunable_state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::prunable_node("/runs/run_a.sfmr"));
    let id = state.scene[0].id;
    (state, id)
}

fn prune_options() -> sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions {
    sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions::default()
}

#[test]
fn pruning_covered_observations_pushes_one_version_with_a_new_base() {
    let (mut state, id) = prunable_state();
    let before = Arc::clone(&state.scene[0].edited().base);
    let observations = state.scene[0].edited().observation_count();

    state
        .start_prune_covered_observations(id, &prune_options())
        .expect("the fixture carries frames, pixels and poses");
    state.finish_background_task();

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), 2);
    assert!(
        !Arc::ptr_eq(&before, &node.edited().base),
        "a bulk edit reused its input's base"
    );
    assert!(
        node.edited().observation_count() < observations,
        "the prune retired nothing on a fixture built to be covered"
    );
    assert_eq!(
        node.history.current_version().label,
        "Pruned covered observations in run_a"
    );
    let entry = newest(&state);
    assert!(
        entry
            .text
            .starts_with("Pruned covered observations in run_a: "),
        "{}",
        entry.text
    );
    assert!(
        entry.text.contains("observations retired"),
        "{}",
        entry.text
    );
}

/// A prune that finds nothing covered pushes no version and says so, rather
/// than leaving a row in the history nobody can tell from one that did
/// something.
#[test]
fn a_prune_that_retires_nothing_pushes_no_version() {
    let (mut state, id) = prunable_state();
    // A footprint of a fiftieth of the projected radius reaches nothing.
    let options = sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions {
        footprint_fraction: 0.02,
        ..prune_options()
    };
    state
        .start_prune_covered_observations(id, &options)
        .expect("the fixture carries frames, pixels and poses");
    state.finish_background_task();

    assert_eq!(state.scene[0].history.versions().len(), 1);
    let entry = newest(&state);
    assert!(!entry.failed, "{}", entry.text);
    assert!(
        entry.text.contains("no effect, no observation is covered"),
        "{}",
        entry.text
    );
}

#[test]
fn a_cancelled_prune_pushes_no_version() {
    let (mut state, id) = prunable_state();
    let observations = state.scene[0].edited().observation_count();
    state
        .start_prune_covered_observations(id, &prune_options())
        .expect("the fixture carries frames, pixels and poses");
    state.cancel_background_task();
    state.finish_background_task();

    assert_eq!(state.scene[0].history.versions().len(), 1);
    assert_eq!(state.scene[0].edited().observation_count(), observations);
    let entry = newest(&state);
    assert!(entry.failed, "{}", entry.text);
    assert!(
        entry
            .text
            .contains("Prune covered observations of run_a cancelled"),
        "{}",
        entry.text
    );
}

#[test]
fn a_reconstruction_with_no_patch_frames_is_refused_in_the_menu_s_own_words() {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    let why = state
        .prune_covered_refusal(id)
        .expect("the fixture carries no patch frame");
    assert!(why.contains("no patch frame"), "{why}");
    let refused = state
        .start_prune_covered_observations(id, &prune_options())
        .expect_err("with no frame there is no footprint");
    assert!(refused.contains(&why), "{refused}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

// ── Where a step's photographs come from ────────────────────────────────

/// A cached photograph is already a pyramid, so a step over nothing but cached
/// photographs reads no file and builds no pyramid: what reaches the kernels is
/// the very `Arc` the cache holds.
///
/// Pointer equality rather than pixel equality, because a rebuilt pyramid would
/// hold the same pixels and cost the seconds this cache exists to remove.
#[test]
fn a_step_over_cached_photographs_takes_their_pyramids_as_they_are() {
    let mut state = state();
    let id = node(&state);
    let images = state.scene[0].image_count();
    assert!(images > 1, "the demo node has images to read");

    let cached: Vec<Arc<ImageU8Pyramid>> = (0..images)
        .map(|index| {
            let pyramid = Arc::new(ImageU8Pyramid::from_image(
                ImageU8::new(8, 8, 3, vec![index as u8; 8 * 8 * 3]),
                crate::state::PYRAMID_LEVELS,
            ));
            state
                .full_res_cache
                .insert(ImageRef::new(id, index), Some(Arc::clone(&pyramid)));
            pyramid
        })
        .collect();

    let needed: Vec<usize> = (0..images).collect();
    let sources = state.view_sources_for(id, &needed).expect("a loaded node");
    let collector = crate::progress::Collector::new(false);
    let decoded = sources
        .decode(&collector.progress())
        .expect("every photograph was cached, so nothing was read");

    for (index, pyramid) in cached.iter().enumerate() {
        assert!(
            Arc::ptr_eq(&decoded.pyramids[index], pyramid),
            "image {index} was pyramided again rather than taken from the cache",
        );
    }
    assert_eq!(
        crate::test_support::phase_note(&collector.take(), "decode images"),
        Some(format!("0 read from disk, {images} reused from the cache")),
    );
}

/// The images a step does not read share one placeholder, so a node of a
/// hundred images costs a step over two of them two pyramids and not a hundred.
#[test]
fn the_images_a_step_does_not_read_share_one_placeholder() {
    let mut state = state();
    let id = node(&state);
    let images = state.scene[0].image_count();
    assert!(images > 2, "the demo node has images to leave unused");
    state.full_res_cache.insert(
        ImageRef::new(id, 0),
        Some(Arc::new(ImageU8Pyramid::from_image(
            ImageU8::new(8, 8, 3, vec![7u8; 8 * 8 * 3]),
            crate::state::PYRAMID_LEVELS,
        ))),
    );

    let sources = state.view_sources_for(id, &[0]).expect("a loaded node");
    let collector = crate::progress::Collector::new(false);
    let decoded = sources
        .decode(&collector.progress())
        .expect("the one image the step reads was cached");

    for index in 2..images {
        assert!(
            Arc::ptr_eq(&decoded.pyramids[1], &decoded.pyramids[index]),
            "unused image {index} carries a placeholder of its own",
        );
    }
    assert_eq!(
        crate::test_support::phase_note(&collector.take(), "decode images"),
        Some("0 read from disk, 1 reused from the cache".to_string()),
    );
}

// ── The point gestures ──────────────────────────────────────────────────

#[test]
fn a_menu_that_only_opened_selects_the_point_and_edits_nothing() {
    let (mut state, id) = nudged_point_state(11);
    let point = PointRef::new(id, 11);
    state.apply_point_gesture(PointGesture::Opened(point));
    assert_eq!(state.selected_point, Some(point));
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

#[test]
fn choosing_retriangulate_selects_the_point_and_pushes_its_version() {
    let (mut state, id) = nudged_point_state(11);
    let point = PointRef::new(id, 11);
    state.apply_point_gesture(PointGesture::Retriangulate(point));
    assert_eq!(state.scene[0].history.versions().len(), 2);
    assert_eq!(
        state.scene[0].history.current_version().label,
        "Retriangulated point 11 in run_a"
    );
    // The selection moved to the point first, and then followed the edit.
    let selected = state.selected_point.expect("the point survived");
    assert_eq!(selected.recon, id);
    assert!(state.scene[0].edited().point(selected.point).is_some());
}

#[test]
fn choosing_edit_on_bench_stages_the_track_and_raises_the_panel() {
    let (mut state, id) = nudged_point_state(11);
    let point = PointRef::new(id, 11);
    // Closed first, so that what the gesture does to the dock is visible: the
    // default layout already carries the panel.
    state.hide_panel(crate::dock::Tab::TrackEdit);
    assert!(!state.is_panel_open(crate::dock::Tab::TrackEdit));

    state.apply_point_gesture(PointGesture::EditOnBench(point));

    assert_eq!(state.selected_point, Some(point));
    // The same staging the Track Edit panel's own button does, and then the
    // panel itself, because this gesture was made somewhere the panel is not.
    let bench = state.scene[0].history.current_bench();
    assert_eq!(bench.entries().len(), 1, "the point is not on the bench");
    assert!(state.is_panel_open(crate::dock::Tab::TrackEdit));
}

/// A second Edit on Bench on the same point -- a double-click after a menu, or
/// one double-click after another -- activates the item that is there rather
/// than putting a second one on, and raises the panel again.
///
/// This is what makes a double-click safe: the gesture arrives as two clicks
/// and the panel is reached from two places, so the step has to be one a
/// reader can repeat without collecting duplicates.
#[test]
fn a_second_edit_on_bench_on_one_point_activates_the_item_already_there() {
    let (mut state, id) = nudged_point_state(11);
    let point = PointRef::new(id, 11);
    state.apply_point_gesture(PointGesture::EditOnBench(point));
    let label = crate::bench::active_track_label(state.bench(id).expect("the node has a bench"))
        .expect("a track is active")
        .to_string();

    state.hide_panel(crate::dock::Tab::TrackEdit);
    state.apply_point_gesture(PointGesture::EditOnBench(point));

    let bench = state.scene[0].history.current_bench();
    assert_eq!(bench.entries().len(), 1, "a second item joined the bench");
    assert_eq!(
        crate::bench::active_track_label(bench),
        Some(label.as_str()),
        "the item already there is not the active one",
    );
    assert!(state.is_panel_open(crate::dock::Tab::TrackEdit));
}

/// The raise is a layout operation, so it has to reach the dock the state
/// holds. The frame takes that dock *out* of the state while a tab body draws
/// and puts it straight back, which is why both panels' gestures are drained
/// in `app.rs` after the `DockArea` call rather than inside one.
///
/// Applied against the placeholder the swap leaves behind, the staging still
/// lands and the raise is thrown away with the placeholder -- which is the
/// regression this pins.
#[test]
fn a_gesture_applied_while_the_dock_is_swapped_out_loses_the_raise() {
    let (mut state, id) = nudged_point_state(11);
    let point = PointRef::new(id, 11);
    state.hide_panel(crate::dock::Tab::TrackEdit);

    let dock = std::mem::replace(&mut state.dock, egui_dock::DockState::new(Vec::new()));
    state.apply_point_gesture(PointGesture::EditOnBench(point));
    let placeholder = std::mem::replace(&mut state.dock, dock);

    assert!(
        placeholder.find_tab(&crate::dock::Tab::TrackEdit).is_some(),
        "the raise did not land on the placeholder, so this test proves nothing",
    );
    assert!(
        !state.is_panel_open(crate::dock::Tab::TrackEdit),
        "the raise reached the real dock from inside the swap",
    );
}
