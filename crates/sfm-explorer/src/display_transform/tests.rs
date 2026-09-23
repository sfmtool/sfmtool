// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The four patch reframes and the bake, asserted on the frame and the picture
//! they leave rather than on the arithmetic that produced them.
//!
//! The fixture is the bench's own ([`crate::bench::tests::state`]): the demo
//! rewritten as `embedded_patches` with exact keypoints, one of its points on
//! the bench. Every test that composes starts from a node already carrying a
//! similarity with a scale other than `1`, because a reversed `compose` or an
//! `M` where `Mᵀ` belongs is silent on a node at the identity.

use std::sync::Arc;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use sfmtool_core::{RotQuaternion, Se3Transform};

use super::*;
use crate::scene::{PointRef, ReconId};
use crate::viewer_3d::bench_track;

/// The similarity the fixture node is drawn under before anything here runs.
fn prior() -> Se3Transform {
    Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.3, -0.7, 0.6), 0.9).unwrap(),
        Vector3::new(4.0, -2.5, 1.25),
        2.0,
    )
}

/// The bench fixture with its point staged, drawn under [`prior`].
fn framed() -> (AppState, ReconId, String) {
    let (mut state, id, label) = bench_track::tests::staged();
    state.set_node_transform(id, prior()).expect("a reframe");
    (state, id, label)
}

/// The staged track's patch, in the reconstruction's own coordinates.
fn placement(state: &AppState, id: ReconId, label: &str) -> OrientedPatch {
    bench_track::placement_of(state.bench_track(id, label).expect("the staged track"))
        .expect("a point carries its patch")
        .clone()
}

/// The staged patch as the node draws it now.
fn drawn(state: &AppState, id: ReconId, label: &str) -> PatchFrame {
    let node = state.node(id).expect("loaded");
    PatchFrame::of(&placement(state, id, label), node.transform())
}

fn transform_of(state: &AppState, id: ReconId) -> Se3Transform {
    state.node(id).expect("loaded").transform().clone()
}

fn close(a: &Vector3<f64>, b: &Vector3<f64>, tolerance: f64) {
    assert!((a - b).norm() <= tolerance, "{a:?} is not {b:?}");
}

/// Whether two transforms are the same map, to `tolerance`; bit for bit at
/// zero, which is what a version restoring one it holds has to give back.
fn same_transform(a: &Se3Transform, b: &Se3Transform, tolerance: f64) {
    if tolerance == 0.0 {
        assert_eq!(
            (&a.rotation, a.translation, a.scale),
            (&b.rotation, b.translation, b.scale)
        );
        return;
    }
    let angle = a.rotation.as_nalgebra().angle_to(b.rotation.as_nalgebra());
    assert!(angle <= tolerance, "the rotations differ by {angle} rad");
    close(&a.translation, &b.translation, tolerance);
    assert!((a.scale - b.scale).abs() <= tolerance);
}

// ── The four reframes ─────────────────────────────────────────────────────

#[test]
fn set_to_origin_puts_the_patchs_frame_on_the_worlds() {
    let (mut state, id, label) = framed();
    state
        .reframe_on_patch(id, PatchReframe::SetToOrigin)
        .expect("a finite patch");

    let frame = drawn(&state, id, &label);
    close(&frame.centre.coords, &Vector3::zeros(), 1e-12);
    close(&frame.u, &Vector3::x(), 1e-12);
    close(&frame.v, &Vector3::y(), 1e-12);
    close(&frame.n, &Vector3::z(), 1e-12);
    assert_eq!(transform_of(&state, id).scale, prior().scale);
}

#[test]
fn align_normal_to_z_tips_about_the_patch() {
    let (mut state, id, label) = framed();
    let before = drawn(&state, id, &label);
    state
        .reframe_on_patch(id, PatchReframe::AlignNormalToZ)
        .expect("a finite patch");
    let after = drawn(&state, id, &label);
    close(&after.centre.coords, &before.centre.coords, 1e-12);
    close(&after.n, &Vector3::z(), 1e-12);
    assert_eq!(transform_of(&state, id).scale, prior().scale);
}

#[test]
fn align_normal_to_z_holds_a_patch_far_from_the_origin_where_it_is() {
    // Far enough that a turn about the world origin would carry the patch a
    // long way along its arc.
    let (mut state, id, label) = bench_track::tests::staged();
    let far = Se3Transform::new(prior().rotation, Vector3::new(900.0, -400.0, 250.0), 1.5);
    state.set_node_transform(id, far).expect("a reframe");
    let before = drawn(&state, id, &label);
    state
        .reframe_on_patch(id, PatchReframe::AlignNormalToZ)
        .expect("a finite patch");
    let after = drawn(&state, id, &label);
    close(&after.centre.coords, &before.centre.coords, 1e-9);
    close(&after.n, &Vector3::z(), 1e-12);
}

#[test]
fn a_normal_at_minus_z_turns_half_about_u() {
    // Built directly, so the input really is antiparallel and the shortest arc
    // really is undetermined.
    let frame = PatchFrame {
        centre: Point3::new(1.0, 2.0, 3.0),
        u: Vector3::x(),
        v: -Vector3::y(),
        n: -Vector3::z(),
    };
    assert!(UnitQuaternion::rotation_between(&frame.n, &Vector3::z()).is_none());
    let map = PatchReframe::AlignNormalToZ.map(&frame);
    close(&map.rotation.rotate_vector(&frame.n), &Vector3::z(), 1e-12);
    close(&map.rotation.rotate_vector(&frame.u), &frame.u, 1e-12);
    close(&map.rotation.rotate_vector(&frame.v), &-frame.v, 1e-12);
    close(
        &map.apply_to_point(&frame.centre).coords,
        &frame.centre.coords,
        1e-12,
    );
}

#[test]
fn translate_to_origin_moves_the_centre_and_turns_nothing() {
    let (mut state, id, label) = framed();
    let before = drawn(&state, id, &label);
    state
        .reframe_on_patch(id, PatchReframe::TranslateToOrigin)
        .expect("a finite patch");
    let after = drawn(&state, id, &label);
    close(&after.centre.coords, &Vector3::zeros(), 1e-12);
    close(&after.n, &before.n, 1e-12);
    close(&after.u, &before.u, 1e-12);
    assert_eq!(transform_of(&state, id).scale, prior().scale);
}

#[test]
fn translate_to_xy_plane_touches_z_alone() {
    let (mut state, id, label) = framed();
    let before = drawn(&state, id, &label);
    let rotation = transform_of(&state, id).rotation;
    state
        .reframe_on_patch(id, PatchReframe::TranslateToXyPlane)
        .expect("a finite patch");
    let after = drawn(&state, id, &label);
    assert!(after.centre.z.abs() <= 1e-12);
    assert_eq!(
        (after.centre.x, after.centre.y),
        (before.centre.x, before.centre.y)
    );
    let now = transform_of(&state, id);
    assert_eq!(now.rotation, rotation);
    assert_eq!(now.scale, prior().scale);
}

#[test]
fn level_then_drop_lays_the_patch_on_the_ground() {
    let (mut state, id, label) = framed();
    state
        .reframe_on_patch(id, PatchReframe::AlignNormalToZ)
        .expect("a finite patch");
    state
        .reframe_on_patch(id, PatchReframe::TranslateToXyPlane)
        .expect("a finite patch");
    let patch = placement(&state, id, &label);
    let transform = transform_of(&state, id);
    for (s, t) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let corner = transform.apply_to_point(&patch.to_world(s, t));
        assert!(
            corner.z.abs() <= 1e-9,
            "a corner stands {} off the plane",
            corner.z
        );
    }
}

#[test]
fn a_reframe_names_the_patch_and_records_a_scene_row() {
    let (mut state, id, label) = framed();
    state
        .reframe_on_patch(id, PatchReframe::SetToOrigin)
        .expect("a finite patch");
    let node_label = state.node(id).expect("loaded").label.clone();
    let version = state.node(id).expect("loaded").history.current_version();
    assert_eq!(
        version.label,
        format!("Set {node_label} to the frame of patch {label}")
    );
    let row = state.action_log.entries().last().expect("a row");
    assert_eq!(row.kind, Kind::Scene);
    assert!(row.text.starts_with(&version.label) && row.text.contains('→'));
}

#[test]
fn a_reframe_with_nothing_active_is_refused_and_pushes_nothing() {
    let (mut state, id, _) = framed();
    let bench = state
        .bench(id)
        .expect("a bench")
        .deactivate(sfmtool_core::bench::ItemKind::Track);
    let node = state.scene.iter_mut().find(|n| n.id == id).expect("loaded");
    node.history.push_bench(Arc::new(bench), "Deactivated");
    let versions = node.history.versions().len();
    let why = state
        .reframe_on_patch(id, PatchReframe::SetToOrigin)
        .expect_err("no patch to read");
    assert!(why.contains("active"), "{why}");
    assert_eq!(
        state.node(id).expect("loaded").history.versions().len(),
        versions
    );
    assert!(state.action_log.entries().last().expect("a row").failed);
}

// ── A reframe on the timeline ─────────────────────────────────────────────

#[test]
fn a_reframe_is_a_version_and_not_a_change() {
    let (mut state, id, _) = bench_track::tests::staged();
    let node = state.node(id).expect("loaded");
    let versions = node.history.versions().len();
    let document = node.history.current_version().document_serial;
    let hash = crate::scene::version_hash_prefix(node);
    let clean = !node.is_dirty();

    state.set_node_transform(id, prior()).expect("a reframe");

    let node = state.node(id).expect("loaded");
    assert!(node.history.can_undo());
    assert_eq!(node.is_dirty(), !clean);
    assert_eq!(node.history.versions().len(), versions + 1);
    assert_eq!(node.history.current_version().document_serial, document);
    assert_eq!(crate::scene::version_hash_prefix(node), hash);

    state.undo(id).expect("an undo");
    assert!(!state.node(id).expect("loaded").has_transform());
}

#[test]
fn every_version_records_the_framing() {
    let (mut state, id, _) = bench_track::tests::staged();
    state.set_node_transform(id, prior()).expect("a reframe");
    state
        .delete_point(PointRef::new(id, 5))
        .expect("a live point");
    same_transform(&transform_of(&state, id), &prior(), 0.0);

    state.undo(id).expect("undo the deletion");
    same_transform(&transform_of(&state, id), &prior(), 0.0);
    state.undo(id).expect("undo the reframe");
    same_transform(&transform_of(&state, id), &Se3Transform::identity(), 0.0);
}

#[test]
fn a_reframe_truncates_the_redo_tail() {
    let (mut state, id, _) = bench_track::tests::staged();
    state
        .delete_point(PointRef::new(id, 5))
        .expect("a live point");
    state.undo(id).expect("an undo");
    assert!(state.can_redo(id));
    state.set_node_transform(id, prior()).expect("a reframe");
    assert!(!state.can_redo(id));
    assert!(!state.node(id).expect("loaded").is_dirty());
}

#[test]
fn reset_transform_is_a_version_and_is_undone() {
    let (mut state, id, _) = framed();
    state.reset_node_transform(id).expect("a reset");
    let node = state.node(id).expect("loaded");
    assert!(!node.has_transform());
    assert!(node
        .history
        .current_version()
        .label
        .starts_with("Reset transform of"));
    state.undo(id).expect("an undo");
    same_transform(&transform_of(&state, id), &prior(), 0.0);
    // Nothing to reset is a refusal rather than a version.
    state.undo(id).expect("back to the identity");
    assert!(state.reset_node_transform(id).is_err());
}

// ── The bake ──────────────────────────────────────────────────────────────

/// Everything the picture is made of that a bake could move: the points and
/// every camera's centre in world space.
fn picture(state: &AppState, id: ReconId) -> Vec<Point3<f64>> {
    let node = state.node(id).expect("loaded");
    let mut at = crate::scene::world_points(node);
    for camera in 0..node.recon().image_table.cameras.len() {
        at.extend(crate::scene::camera_world_centres(node, camera));
    }
    at
}

fn same_picture(a: &[Point3<f64>], b: &[Point3<f64>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        let tolerance = 1e-9 * a.coords.norm().max(1.0);
        assert!((a - b).norm() <= tolerance, "{a} moved to {b}");
    }
}

/// Where every observation of the node's value projects, in pixels.
fn reprojections(state: &AppState, id: ReconId) -> Vec<Option<(f64, f64)>> {
    let recon = state.node(id).expect("loaded").recon();
    recon
        .point_set
        .tracks
        .iter()
        .map(|observation| {
            let image = &recon.image_table.images[observation.image_index as usize];
            let camera = &recon.image_table.cameras[image.camera_index as usize];
            let point = &recon.point_set.points[observation.point_index as usize];
            let seen = crate::scene::cam_from_world(image).transform_point(&point.position);
            camera.ray_to_pixel([seen.x, seen.y, seen.z])
        })
        .collect()
}

/// The staged track's figure from the fixture's eye.
fn figure(state: &AppState, id: ReconId, label: &str) -> bench_track::Figure {
    let track = state.bench_track(id, label).expect("the staged track");
    bench_track::tests::figure_of(state, id, track).expect("a figure")
}

/// Whether two figures are the same drawing, to the `f32` they are stored in.
fn same_figure(a: &bench_track::Figure, b: &bench_track::Figure) {
    let ends = |figure: &bench_track::Figure| -> Vec<[f32; 4]> {
        figure
            .strokes()
            .flat_map(|stroke| [stroke.a, stroke.b])
            .chain(std::iter::once(figure.centre))
            .collect()
    };
    let (a_ends, b_ends) = (ends(a), ends(b));
    assert_eq!(a_ends.len(), b_ends.len());
    for (a, b) in a_ends.iter().zip(&b_ends) {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() <= 1e-4 * a[k].abs().max(1.0),
                "{a:?} moved to {b:?}"
            );
        }
    }
    assert!((a.fog_distance - b.fog_distance).abs() <= 1e-4 * a.fog_distance.max(1.0));
}

#[test]
fn the_bake_does_not_move_the_picture() {
    let (mut state, id, _) = framed();
    let before = picture(&state, id);
    state.bake_node_transform(id).expect("a bake");
    same_picture(&before, &picture(&state, id));
    assert!(!state.node(id).expect("loaded").has_transform());
}

#[test]
fn the_bake_does_not_move_any_reprojection() {
    let (mut state, id, _) = framed();
    let before = reprojections(&state, id);
    state.bake_node_transform(id).expect("a bake");
    let after = reprojections(&state, id);
    assert!(before.iter().filter(|p| p.is_some()).count() > 10);
    for (a, b) in before.iter().zip(&after) {
        match (a, b) {
            (Some(a), Some(b)) => assert!(
                (a.0 - b.0).abs() <= 1e-9 && (a.1 - b.1).abs() <= 1e-9,
                "{a:?} moved to {b:?}"
            ),
            (a, b) => assert_eq!(a, b),
        }
    }
}

#[test]
fn the_bench_figure_does_not_move_across_the_bake() {
    let (mut state, id, label) = framed();
    let before = figure(&state, id, &label);
    let frame = drawn(&state, id, &label);
    let half = placement(&state, id, &label).half_extent[0];
    state.bake_node_transform(id).expect("a bake");
    same_figure(&before, &figure(&state, id, &label));
    let after = drawn(&state, id, &label);
    close(&after.centre.coords, &frame.centre.coords, 1e-9);
    close(&after.n, &frame.n, 1e-12);
    // The placement's size is in the value's units, which the bake rescaled.
    let patch = placement(&state, id, &label);
    assert!((patch.half_extent[0] - half * prior().scale).abs() <= 1e-12);
}

#[test]
fn undo_of_a_bake_puts_the_picture_the_value_and_the_transform_back() {
    let (mut state, id, label) = framed();
    let (picture_before, figure_before) = (picture(&state, id), figure(&state, id, &label));
    let base_before = Arc::clone(&state.node(id).expect("loaded").edited().base);
    state.bake_node_transform(id).expect("a bake");
    let node = state.node(id).expect("loaded");
    assert!(node.is_dirty());
    assert!(!Arc::ptr_eq(&node.edited().base, &base_before));
    let (picture_after, figure_after) = (picture(&state, id), figure(&state, id, &label));

    state.undo(id).expect("an undo");
    let node = state.node(id).expect("loaded");
    assert!(Arc::ptr_eq(&node.edited().base, &base_before));
    same_transform(node.transform(), &prior(), 0.0);
    same_picture(&picture_before, &picture(&state, id));
    same_figure(&figure_before, &figure(&state, id, &label));

    state.redo(id).expect("a redo");
    assert!(!state.node(id).expect("loaded").has_transform());
    same_picture(&picture_after, &picture(&state, id));
    same_figure(&figure_after, &figure(&state, id, &label));
}

#[test]
fn a_bake_folds_the_overlay_and_keeps_every_live_point() {
    let (mut state, id, _) = framed();
    state
        .delete_point(PointRef::new(id, 5))
        .expect("a live point");
    let live = state.node(id).expect("loaded").point_count();
    let before = picture(&state, id);
    state.bake_node_transform(id).expect("a bake");
    let node = state.node(id).expect("loaded");
    assert_eq!(node.point_count(), live);
    assert!(node.edited().deleted_points.is_empty());
    same_picture(&before, &picture(&state, id));
}

#[test]
fn the_bake_is_one_edit_row_with_the_size_of_the_move() {
    let (mut state, id, _) = framed();
    state.bake_node_transform(id).expect("a bake");
    let row = state.action_log.entries().last().expect("a row");
    assert_eq!(row.kind, Kind::Edit);
    let label = &state.node(id).expect("loaded").label;
    assert!(
        row.text
            .starts_with(&format!("Baked transform of {label}: "))
            && row.text.contains("deg")
            && row.text.contains("scene units")
            && row.text.contains(", x2.000"),
        "{}",
        row.text
    );
}

#[test]
fn a_node_at_the_identity_has_nothing_to_bake() {
    let (mut state, id, _) = bench_track::tests::staged();
    let versions = state.node(id).expect("loaded").history.versions().len();
    let why = state.bake_node_transform(id).expect_err("nothing to bake");
    assert!(why.contains("already in its own frame"), "{why}");
    assert_eq!(
        state.node(id).expect("loaded").history.versions().len(),
        versions
    );
}

#[test]
fn set_to_origin_after_a_bake_is_the_identity_to_rounding() {
    let (mut state, id, label) = framed();
    state
        .reframe_on_patch(id, PatchReframe::SetToOrigin)
        .expect("a finite patch");
    state.bake_node_transform(id).expect("a bake");
    state
        .reframe_on_patch(id, PatchReframe::SetToOrigin)
        .expect("a finite patch");
    // Float arithmetic, not an algebraic identity: `has_transform` compares
    // exactly and may still say yes, which is the wart the spec records.
    same_transform(&transform_of(&state, id), &Se3Transform::identity(), 1e-12);
    let frame = drawn(&state, id, &label);
    close(&frame.centre.coords, &Vector3::zeros(), 1e-12);
}
