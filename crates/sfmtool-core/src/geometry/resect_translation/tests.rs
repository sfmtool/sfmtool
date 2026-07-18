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

fn simple_radial_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: 300.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
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

/// A synthetic scene for a fixed rotation: ground-truth translation, world
/// points, and their exact observed pixels under `cam`.
type Scene = (Vector3<f64>, Vec<[f64; 3]>, Vec<[f64; 2]>);

fn make_scene(cam: &CameraIntrinsics, rotation: &UnitQuaternion<f64>, count: usize) -> Scene {
    let t_true = Vector3::new(0.4, -0.25, 0.3);
    let mut points = Vec::new();
    let mut uv = Vec::new();
    let mut i = 0usize;
    while points.len() < count {
        let x = [
            2.0 * jitter(i, 1),
            2.0 * jitter(i, 2),
            -6.0 + 2.0 * jitter(i, 3),
        ];
        i += 1;
        let c = rotation * Vector3::new(x[0], x[1], x[2]) + t_true;
        if c.z >= -0.5 {
            continue; // keep well in front
        }
        let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) else {
            continue;
        };
        points.push(x);
        uv.push([u, v]);
    }
    (t_true, points, uv)
}

fn rotation() -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(Vector3::new(0.15, -0.1, 0.05))
}

#[test]
fn recovers_translation_pinhole() {
    let cam = simple_pinhole();
    let r = rotation();
    let (t_true, points, uv) = make_scene(&cam, &r, 30);
    let out = resect_translation(&cam, &r, &points, &uv, 8.0, 10).expect("resection");
    assert!((out.translation - t_true).norm() < 1e-9, "t err");
    assert!(out.inliers.iter().all(|&k| k));
    assert!(out.residual_norms.iter().all(|&e| e < 1e-6));
}

#[test]
fn recovers_translation_fisheye() {
    let cam = simple_radial_fisheye();
    let r = rotation();
    let (t_true, points, uv) = make_scene(&cam, &r, 30);
    let out = resect_translation(&cam, &r, &points, &uv, 8.0, 10).expect("resection");
    assert!(
        (out.translation - t_true).norm() < 1e-6,
        "t err {}",
        (out.translation - t_true).norm()
    );
    assert!(out.inliers.iter().all(|&k| k));
    assert!(out.residual_norms.iter().all(|&e| e < 1e-4));
}

#[test]
fn trims_planted_outliers() {
    let cam = simple_pinhole();
    let r = rotation();
    let (t_true, points, mut uv) = make_scene(&cam, &r, 40);
    // Corrupt ~10% of the observations with large pixel offsets (alternating
    // directions) — stragglers in a largely correct set, which is the regime
    // the trimmed gate is specified for.
    let mut corrupted = Vec::new();
    for i in (0..points.len()).step_by(10) {
        let s = if (i / 10) % 2 == 0 { 1.0 } else { -1.0 };
        uv[i][0] += s * (60.0 + 30.0 * jitter(i, 7));
        uv[i][1] -= s * (50.0 + 20.0 * jitter(i, 9));
        corrupted.push(i);
    }
    let out = resect_translation(&cam, &r, &points, &uv, 8.0, 10).expect("resection");
    assert!(
        (out.translation - t_true).norm() < 1e-6,
        "outliers biased t: err {}",
        (out.translation - t_true).norm()
    );
    for i in 0..points.len() {
        let expect = !corrupted.contains(&i);
        assert_eq!(out.inliers[i], expect, "mask at {i}");
    }
}

#[test]
fn excludes_behind_camera_points() {
    let cam = simple_pinhole();
    let r = rotation();
    let (t_true, mut points, mut uv) = make_scene(&cam, &r, 30);
    // A world point behind the canonical camera (camera-frame z > 0), paired
    // with a plausible pixel: cheirality must exclude it.
    let behind_cam = Vector3::new(0.1, -0.2, 3.0);
    let behind_world = r.inverse() * (behind_cam - t_true);
    points.push([behind_world.x, behind_world.y, behind_world.z]);
    uv.push([300.0, 250.0]);
    let out = resect_translation(&cam, &r, &points, &uv, 8.0, 10).expect("resection");
    let last = points.len() - 1;
    assert!(!out.inliers[last], "behind-camera point kept");
    assert_eq!(out.residual_norms[last], INVALID_RESIDUAL);
    assert!((out.translation - t_true).norm() < 1e-9);
}

#[test]
fn fails_below_min_inliers() {
    let cam = simple_pinhole();
    let r = rotation();
    let (_t_true, points, uv) = make_scene(&cam, &r, 8);
    // Only 8 observations against the default floor of 10.
    assert!(resect_translation(&cam, &r, &points, &uv, 8.0, 10).is_none());

    // Enough observations, but contamination trims the survivors below the
    // floor.
    let (_t_true, points, mut uv) = make_scene(&cam, &r, 12);
    for item in uv.iter_mut().take(4) {
        item[0] += 80.0;
    }
    assert!(resect_translation(&cam, &r, &points, &uv, 8.0, 10).is_none());
}

#[test]
fn mismatched_lengths_fail() {
    let cam = simple_pinhole();
    let r = rotation();
    let (_t_true, points, uv) = make_scene(&cam, &r, 12);
    assert!(resect_translation(&cam, &r, &points[..11], &uv, 8.0, 10).is_none());
}

#[test]
fn near_parallel_rays_still_solve() {
    let cam = simple_pinhole();
    let r = rotation();
    let t_true = Vector3::new(0.4, -0.25, 0.3);
    // A distant, tight bundle: every ray is within a fraction of a degree of
    // its neighbors — heavily ill-conditioned along the shared direction.
    let mut points = Vec::new();
    let mut uv = Vec::new();
    for i in 0..15 {
        let x = [
            0.05 * jitter(i, 1),
            0.05 * jitter(i, 2),
            -500.0 + 0.05 * jitter(i, 3),
        ];
        let c = r * Vector3::new(x[0], x[1], x[2]) + t_true;
        let (u, v) = cam.ray_to_pixel([c.x, c.y, c.z]).unwrap();
        points.push(x);
        uv.push([u, v]);
    }
    let out = resect_translation(&cam, &r, &points, &uv, 8.0, 10)
        .expect("degenerate bundle must still return the least-squares solve");
    // The least-squares solution reprojects consistently even where the
    // bundle leaves the along-ray component weakly constrained.
    assert!(out.inliers.iter().all(|&k| k));
    assert!(out.residual_norms.iter().all(|&e| e < 8.0));
}
