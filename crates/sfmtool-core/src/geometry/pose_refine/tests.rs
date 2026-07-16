// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::CameraModel;

fn simple_pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

/// Deterministic pseudo-random in [-1, 1] from an index (no rand dependency).
fn jitter(i: usize, salt: u64) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z ^= z >> 27;
    ((z % 20001) as f64 / 10000.0) - 1.0
}

/// A synthetic scene: camera, ground-truth pose `(R, t)`, world points, and
/// their observed pixels.
type Scene = (
    CameraIntrinsics,
    UnitQuaternion<f64>,
    Vector3<f64>,
    Vec<[f64; 3]>,
    Vec<[f64; 2]>,
);

fn make_scene() -> Scene {
    let cam = simple_pinhole();
    let r_true = UnitQuaternion::from_scaled_axis(Vector3::new(0.15, -0.1, 0.05));
    let t_true = Vector3::new(0.3, -0.2, 0.4);
    let mut points = Vec::new();
    let mut uv = Vec::new();
    for i in 0..40 {
        // World points spread in front of the camera after the pose.
        let x = [
            2.0 * jitter(i, 1),
            2.0 * jitter(i, 2),
            -5.0 + 2.0 * jitter(i, 3),
        ];
        let c = r_true * Vector3::new(x[0], x[1], x[2]) + t_true;
        if c.z >= -0.5 {
            continue; // keep well in front
        }
        let (u, v) = cam.ray_to_pixel([c.x, c.y, c.z]).unwrap();
        points.push(x);
        uv.push([u, v]);
    }
    (cam, r_true, t_true, points, uv)
}

#[test]
fn recovers_pose_from_clean_correspondences() {
    let (cam, r_true, t_true, points, uv) = make_scene();
    // Perturbed init.
    let r0 = UnitQuaternion::from_scaled_axis(Vector3::new(0.15 + 0.08, -0.1 - 0.06, 0.05 + 0.04));
    let t0 = t_true + Vector3::new(0.1, -0.08, 0.12);
    let out = refine_absolute_pose(&cam, &uv, &points, &r0, &t0, 5, 0.6, 3.0);
    assert!(out.inlier_fraction > 0.95, "inl {}", out.inlier_fraction);
    let ang = out.rotation.angle_to(&r_true);
    assert!(ang < 1e-3, "rotation error {ang} rad");
    assert!((out.translation - t_true).norm() < 1e-3, "t err");
}

#[test]
fn trims_outliers() {
    let (cam, r_true, t_true, mut points, mut uv) = make_scene();
    // Corrupt ~25% of the observations with large pixel offsets.
    for i in (0..points.len()).step_by(4) {
        uv[i][0] += 60.0 + 30.0 * jitter(i, 7);
        uv[i][1] -= 50.0 + 20.0 * jitter(i, 9);
    }
    // Also inject a couple of behind-camera junk points.
    points.push([0.0, 0.0, 10.0]);
    uv.push([10.0, 10.0]);

    let r0 = UnitQuaternion::from_scaled_axis(Vector3::new(0.15 + 0.05, -0.1 - 0.05, 0.05 + 0.03));
    let t0 = t_true + Vector3::new(0.08, -0.05, 0.09);
    let out = refine_absolute_pose(&cam, &uv, &points, &r0, &t0, 5, 0.6, 3.0);
    // The clean majority should be recovered despite the contamination.
    let ang = out.rotation.angle_to(&r_true);
    assert!(ang < 5e-2, "rotation error {ang} rad under outliers");
    assert!(out.inlier_fraction > 0.6, "inl {}", out.inlier_fraction);
}

#[test]
fn identity_stays_put_when_already_optimal() {
    let (cam, r_true, t_true, points, uv) = make_scene();
    let out = refine_absolute_pose(&cam, &uv, &points, &r_true, &t_true, 5, 0.6, 3.0);
    assert!(out.inlier_fraction > 0.99);
    assert!(out.rotation.angle_to(&r_true) < 1e-6);
}
