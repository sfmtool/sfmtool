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

/// A >180 deg equidistant fisheye: `theta = r/f` with the image circle well
/// past 90 deg (240 px across a 480 px sensor is theta = 240/138 = 99.6 deg).
fn equidistant_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 138.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
    }
}

/// Scene spread over the whole sphere in front of AND beside the camera, so a
/// large share of the observations sit past 90 deg off axis (canonical `z > 0`).
fn make_wide_scene(cam: &CameraIntrinsics, rotation: &UnitQuaternion<f64>, count: usize) -> Scene {
    let t_true = Vector3::new(0.4, -0.25, 0.3);
    let (mut points, mut uv) = (Vec::new(), Vec::new());
    let mut i = 0usize;
    while points.len() < count {
        // Directions covering theta up to ~110 deg at a range of distances.
        let theta = 0.15 + 1.75 * (0.5 + 0.5 * jitter(i, 11));
        let phi = std::f64::consts::PI * jitter(i, 12);
        let rho = 3.0 + 2.0 * jitter(i, 13).abs();
        let c = Vector3::new(
            rho * theta.sin() * phi.cos(),
            rho * theta.sin() * phi.sin(),
            -rho * theta.cos(),
        );
        i += 1;
        let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) else {
            continue;
        };
        if !(0.0..480.0).contains(&u) || !(0.0..480.0).contains(&v) {
            continue;
        }
        let x = rotation.inverse() * (c - t_true);
        points.push([x.x, x.y, x.z]);
        uv.push([u, v]);
    }
    (t_true, points, uv)
}

#[test]
fn equidistant_fisheye_resects_past_ninety_degrees() {
    // Regression: the trim gate used to be the perspective half-space
    // `(R.X + t)_z < 0`, which scores every past-90-degree observation as
    // INVALID_RESIDUAL — the whole periphery a >180 deg capture exists to
    // image. The gate is now positive range along the observed ray.
    let cam = equidistant_fisheye();
    let r = rotation();
    let (t_true, points, uv) = make_wide_scene(&cam, &r, 90);
    let past_90 = points
        .iter()
        .filter(|x| (r * Vector3::new(x[0], x[1], x[2]) + t_true).z > 0.0)
        .count();
    assert!(
        past_90 >= 12,
        "fixture must exercise the periphery: {past_90}"
    );

    let out = resect_translation(&cam, &r, &points, &uv, 2.0, 10).expect("resection");
    assert!(
        (out.translation - t_true).norm() < 1e-8,
        "t err {}",
        (out.translation - t_true).norm()
    );
    assert!(
        out.inliers.iter().all(|&k| k),
        "a peripheral view was trimmed"
    );
    assert!(out.residual_norms.iter().all(|&e| e < 1e-6));
}

#[test]
fn equidistant_fisheye_still_rejects_the_antipodal_reflection() {
    // The cross-product rows are sign-blind, so dropping the half-space gate
    // must not admit the point mirrored through the camera centre: its range
    // along the observed ray is negative.
    let cam = equidistant_fisheye();
    let r = rotation();
    let (t_true, mut points, mut uv) = make_wide_scene(&cam, &r, 30);
    let c = Vector3::new(0.6, -0.3, -2.0); // in front, theta ~ 17 deg
    let (u, v) = cam.ray_to_pixel([c.x, c.y, c.z]).unwrap();
    let mirrored = r.inverse() * (-c - t_true); // same line, opposite side
    points.push([mirrored.x, mirrored.y, mirrored.z]);
    uv.push([u, v]);
    let out = resect_translation(&cam, &r, &points, &uv, 2.0, 10).expect("resection");
    let last = points.len() - 1;
    assert!(!out.inliers[last], "antipodal reflection kept");
    assert_eq!(out.residual_norms[last], INVALID_RESIDUAL);
    assert!((out.translation - t_true).norm() < 1e-8);
}

#[test]
fn perspective_trim_gate_is_bit_identical_to_the_half_space() {
    // The structural claim the pinhole fleet's byte-parity rests on: for a
    // perspective camera the new branch evaluates the SAME expression the
    // half-space always did, so nothing about a pinhole solve can move.
    // Checked as an identity over the gate's own decision, on geometry that
    // straddles the plane (points both in front of and behind the camera).
    for cam in [simple_pinhole(), simple_radial_fisheye()] {
        let ray_path = cam.model.needs_ray_path();
        assert_eq!(
            ray_path,
            matches!(cam.model, CameraModel::SimpleRadialFisheye { .. }),
            "fixture expectation"
        );
        if ray_path {
            continue; // the polynomial fisheye takes the ray branch by design
        }
        let r = rotation();
        let (t_true, mut points, mut uv) = make_scene(&cam, &r, 30);
        // Add a spread of points behind the camera plane and just in front of
        // it, so the half-space decision is exercised in both directions.
        for k in 0..10 {
            let c = Vector3::new(
                0.3 * jitter(k, 21),
                0.3 * jitter(k, 22),
                -1.0 + 0.4 * k as f64,
            );
            let w = r.inverse() * (c - t_true);
            points.push([w.x, w.y, w.z]);
            uv.push([320.0 + 40.0 * jitter(k, 23), 240.0 + 40.0 * jitter(k, 24)]);
        }
        let out = resect_translation(&cam, &r, &points, &uv, 8.0, 10).expect("resection");
        for (i, x) in points.iter().enumerate() {
            let c = r * Vector3::new(x[0], x[1], x[2]) + out.translation;
            // The gate's verdict is exactly `z < 0` plus the pixel bound; an
            // observation at z >= 0 must carry INVALID_RESIDUAL, never a
            // range-based score.
            if c.z >= 0.0 {
                assert_eq!(
                    out.residual_norms[i], INVALID_RESIDUAL,
                    "perspective gate scored a behind-camera point at index {i}"
                );
                assert!(!out.inliers[i]);
            }
        }
    }
}
