// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::{Matrix3, Vector3};

const EPS: f64 = 1e-12;

fn assert_mat_eq(a: &Matrix3<f64>, b: &Matrix3<f64>) {
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (a[(r, c)] - b[(r, c)]).abs() < EPS,
                "matrices differ at ({r}, {c}): {} vs {}",
                a[(r, c)],
                b[(r, c)]
            );
        }
    }
}

fn assert_vec_eq(a: &Vector3<f64>, b: &Vector3<f64>) {
    assert!((a - b).norm() < EPS, "vectors differ: {a:?} vs {b:?}");
}

/// A few fixed, non-trivial world-to-camera poses for round-trip tests.
fn fixture_poses() -> Vec<(RotQuaternion, Vector3<f64>)> {
    vec![
        (RotQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0)),
        (
            RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 3.0), 0.7).unwrap(),
            Vector3::new(0.3, -1.2, 2.5),
        ),
        (
            RotQuaternion::from_axis_angle(Vector3::new(-0.4, 1.0, 0.1), 2.1).unwrap(),
            Vector3::new(-5.0, 0.25, -0.75),
        ),
        (
            RotQuaternion::new(0.2, -0.5, 0.7, 0.3),
            Vector3::new(10.0, -20.0, 30.0),
        ),
    ]
}

// ── S and W algebra ───────────────────────────────────────────────────────

#[test]
fn s_is_a_proper_involutive_rotation() {
    let s = s_matrix();
    assert!((s.determinant() - 1.0).abs() < EPS);
    assert_mat_eq(&(s.transpose() * s), &Matrix3::identity());
    // Involutive: S·S = I
    assert_mat_eq(&(s * s), &Matrix3::identity());
}

#[test]
fn w_is_a_proper_rotation() {
    let w = w_matrix();
    assert!((w.determinant() - 1.0).abs() < EPS);
    assert_mat_eq(&(w.transpose() * w), &Matrix3::identity());
}

#[test]
fn s_and_w_expected_entries() {
    let s = s_matrix();
    assert_mat_eq(
        &s,
        &Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0),
    );
    let w = w_matrix();
    assert_mat_eq(
        &w,
        &Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0),
    );
}

// ── World-vector rotation by W ────────────────────────────────────────────

#[test]
fn world_rotate_w_matches_matrix_and_hand_computed_values() {
    // Axis mapping: X fixed; COLMAP +Y (typical "down") → canonical −Z;
    // COLMAP +Z → canonical +Y. In particular COLMAP's −Y "up" → +Z up.
    assert_vec_eq(
        &world_rotate_w(&Vector3::new(1.0, 0.0, 0.0)),
        &Vector3::new(1.0, 0.0, 0.0),
    );
    assert_vec_eq(
        &world_rotate_w(&Vector3::new(0.0, 1.0, 0.0)),
        &Vector3::new(0.0, 0.0, -1.0),
    );
    assert_vec_eq(
        &world_rotate_w(&Vector3::new(0.0, 0.0, 1.0)),
        &Vector3::new(0.0, 1.0, 0.0),
    );
    assert_vec_eq(
        &world_rotate_w(&Vector3::new(0.0, -1.0, 0.0)),
        &Vector3::new(0.0, 0.0, 1.0),
    );

    let v = Vector3::new(0.5, -2.0, 3.25);
    assert_vec_eq(&world_rotate_w(&v), &(w_matrix() * v));
    assert_vec_eq(&world_rotate_w_inverse(&v), &(w_matrix().transpose() * v));
}

#[test]
fn world_rotate_w_round_trips() {
    let v = Vector3::new(-1.5, 4.0, 0.25);
    assert_vec_eq(&world_rotate_w_inverse(&world_rotate_w(&v)), &v);
    assert_vec_eq(&world_rotate_w(&world_rotate_w_inverse(&v)), &v);
}

// ── Pose conversion COLMAP ↔ canonical ────────────────────────────────────

#[test]
fn pose_conversion_round_trips() {
    for (q, t) in fixture_poses() {
        let (q_can, t_can) = pose_colmap_to_canonical(&q, &t);
        let (q_back, t_back) = pose_canonical_to_colmap(&q_can, &t_can);
        assert_eq!(q_back, q, "rotation did not round-trip");
        assert_vec_eq(&t_back, &t);

        // And the reverse order: canonical → COLMAP → canonical.
        let (q_col, t_col) = pose_canonical_to_colmap(&q, &t);
        let (q_back2, t_back2) = pose_colmap_to_canonical(&q_col, &t_col);
        assert_eq!(q_back2, q);
        assert_vec_eq(&t_back2, &t);
    }
}

#[test]
fn pose_conversion_matches_formula() {
    // R' = S·R·Wᵀ, t' = S·t — checked against explicit matrix products.
    for (q, t) in fixture_poses() {
        let (q_can, t_can) = pose_colmap_to_canonical(&q, &t);
        let expected_r = s_matrix() * q.to_rotation_matrix() * w_matrix().transpose();
        assert_mat_eq(&q_can.to_rotation_matrix(), &expected_r);
        assert_vec_eq(&t_can, &(s_matrix() * t));
    }
}

#[test]
fn identity_colmap_pose_maps_to_expected_canonical_camera() {
    // The identity COLMAP pose is a camera at the origin looking down COLMAP
    // world +Z with world +Y "down". After conversion the camera must look
    // along W·(0,0,1) = (0,1,0) with up W·(0,−1,0) = (0,0,1).
    let (q_can, t_can) =
        pose_colmap_to_canonical(&RotQuaternion::identity(), &Vector3::new(0.0, 0.0, 0.0));
    let r_can = q_can.to_rotation_matrix();

    // Canonical cameras look down camera −Z: viewing dir = R'ᵀ·(0,0,−1).
    let view_dir = r_can.transpose() * Vector3::new(0.0, 0.0, -1.0);
    assert_vec_eq(&view_dir, &Vector3::new(0.0, 1.0, 0.0));

    // Canonical camera up is +Y: up = R'ᵀ·(0,1,0) — world +Z (up).
    let up = r_can.transpose() * Vector3::new(0.0, 1.0, 0.0);
    assert_vec_eq(&up, &Vector3::new(0.0, 0.0, 1.0));

    // Image-plane right stays world +X.
    let right = r_can.transpose() * Vector3::new(1.0, 0.0, 0.0);
    assert_vec_eq(&right, &Vector3::new(1.0, 0.0, 0.0));

    assert_vec_eq(&t_can, &Vector3::new(0.0, 0.0, 0.0));
}

#[test]
fn camera_center_transforms_by_w() {
    // C' = −R'ᵀ·t' must equal W·C where C = −Rᵀ·t (plan §1 invariant).
    for (q, t) in fixture_poses() {
        let center = q.camera_center(&t);
        let (q_can, t_can) = pose_colmap_to_canonical(&q, &t);
        let center_can = q_can.camera_center(&t_can);
        assert_vec_eq(&center_can, &world_rotate_w(&center));
    }
}

#[test]
fn projected_point_is_preserved_through_conversion() {
    // A world point in front of the COLMAP camera maps to the same
    // normalized pixel after converting both pose and point: camera-space
    // coordinates conjugate by S, so (x/z, y/z) = (x'/−z', −y'/−z').
    let (q, t) = (
        RotQuaternion::from_axis_angle(Vector3::new(0.2, -0.3, 0.9), 0.4).unwrap(),
        Vector3::new(0.5, 1.0, -2.0),
    );
    let x_world = Vector3::new(0.7, -0.4, 6.0);
    let p_cam = q.to_rotation_matrix() * x_world + t;
    assert!(p_cam.z > 0.0, "fixture point must be in front (COLMAP)");

    let (q_can, t_can) = pose_colmap_to_canonical(&q, &t);
    let x_can = world_rotate_w(&x_world);
    let p_cam_can = q_can.to_rotation_matrix() * x_can + t_can;

    // Camera-space coordinates flip by S.
    assert_vec_eq(&p_cam_can, &(s_matrix() * p_cam));
    // In front in canonical terms (z < 0), same normalized coords (y up).
    assert!(p_cam_can.z < 0.0);
    assert!((p_cam.x / p_cam.z - p_cam_can.x / -p_cam_can.z).abs() < EPS);
    assert!((p_cam.y / p_cam.z - -p_cam_can.y / -p_cam_can.z).abs() < EPS);
}

// ── Relative-pose S-conjugation ───────────────────────────────────────────

#[test]
fn relative_pose_conjugate_s_is_involutive() {
    for (q, t) in fixture_poses() {
        let (q1, t1) = relative_pose_conjugate_s(&q, &t);
        let (q2, t2) = relative_pose_conjugate_s(&q1, &t1);
        assert_eq!(q2, q);
        assert_vec_eq(&t2, &t);
    }
}

#[test]
fn relative_pose_conjugate_s_matches_formula() {
    for (q, t) in fixture_poses() {
        let (q1, t1) = relative_pose_conjugate_s(&q, &t);
        let expected_r = s_matrix() * q.to_rotation_matrix() * s_matrix();
        assert_mat_eq(&q1.to_rotation_matrix(), &expected_r);
        assert_vec_eq(&t1, &(s_matrix() * t));
    }
}

#[test]
fn relative_pose_conjugate_s_consistent_with_absolute_conversion() {
    // cam2_from_cam1 = pose2 ∘ pose1⁻¹; converting the absolute poses and
    // recomposing must equal S-conjugating the relative pose directly.
    let poses = fixture_poses();
    let (q1, t1) = &poses[1];
    let (q2, t2) = &poses[2];

    let compose_relative =
        |qa: &RotQuaternion, ta: &Vector3<f64>, qb: &RotQuaternion, tb: &Vector3<f64>| {
            // b_from_a: R = Rb·Raᵀ, t = tb − R·ta
            let r = qb.to_rotation_matrix() * qa.to_rotation_matrix().transpose();
            let t = tb - r * ta;
            (RotQuaternion::from_rotation_matrix(r), t)
        };

    let (q_rel, t_rel) = compose_relative(q1, t1, q2, t2);
    let (q_rel_conj, t_rel_conj) = relative_pose_conjugate_s(&q_rel, &t_rel);

    let (q1c, t1c) = pose_colmap_to_canonical(q1, t1);
    let (q2c, t2c) = pose_colmap_to_canonical(q2, t2);
    let (q_rel_can, t_rel_can) = compose_relative(&q1c, &t1c, &q2c, &t2c);

    assert_eq!(q_rel_can, q_rel_conj);
    assert_vec_eq(&t_rel_can, &t_rel_conj);
}
