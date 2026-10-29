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
        rolling_shutter: None,
    }
}

/// Project canonical camera-frame points to pixels via the camera's own
/// forward model (the ground-truth the residual must reproduce).
fn project(cam: &CameraIntrinsics, pts: &[[f64; 3]]) -> Vec<f64> {
    let mut uv = Vec::with_capacity(pts.len() * 2);
    for p in pts {
        let (u, v) = cam.ray_to_pixel(*p).expect("point in front");
        uv.push(u);
        uv.push(v);
    }
    uv
}

#[test]
fn identity_pose_zero_residual() {
    let cam = simple_pinhole();
    // Canonical frame: camera looks along −Z, in-front points have z < 0.
    let pts = [[0.0, 0.0, -5.0], [1.0, 0.5, -4.0], [-2.0, 1.5, -6.0]];
    let uv = project(&cam, &pts);
    let points: Vec<f64> = pts.iter().flatten().copied().collect();

    let quats = [1.0, 0.0, 0.0, 0.0]; // identity, one image
    let t = [0.0, 0.0, 0.0];
    let obs_img = [0u32, 0, 0];
    let obs_pt = [0u32, 1, 2];

    let res = reprojection_residuals(&cam, &quats, &t, &points, &uv, &obs_img, &obs_pt, 1e6);
    assert_eq!(res.len(), 6);
    for r in &res {
        assert!(r.abs() < 1e-9, "residual should be ~0, got {r}");
    }
    assert_eq!(inlier_fraction(&res, 1.0), 1.0);
}

#[test]
fn translated_pose_zero_residual() {
    let cam = simple_pinhole();
    // World points; a nonzero world-to-camera translation. Camera-frame point
    // is R·X + t; here R = I so cam_pt = X + t. Choose X so cam_pt is in front.
    let t = [0.3, -0.2, 2.0];
    let world = [[0.1, 0.0, -7.0], [-1.0, 0.8, -5.0]];
    // camera-frame points to derive the observed uv
    let cam_pts: Vec<[f64; 3]> = world
        .iter()
        .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
        .collect();
    let uv = project(&cam, &cam_pts);
    let points: Vec<f64> = world.iter().flatten().copied().collect();

    let quats = [1.0, 0.0, 0.0, 0.0];
    let obs_img = [0u32, 0];
    let obs_pt = [0u32, 1];
    let res = reprojection_residuals(&cam, &quats, &t, &points, &uv, &obs_img, &obs_pt, 1e6);
    for r in &res {
        assert!(r.abs() < 1e-9, "residual should be ~0, got {r}");
    }
}

#[test]
fn behind_camera_is_invalid() {
    let cam = simple_pinhole();
    // z > 0 in canonical frame is behind the camera → ray_to_pixel None.
    let points = [0.0, 0.0, 5.0];
    let uv = [320.0, 240.0];
    let quats = [1.0, 0.0, 0.0, 0.0];
    let t = [0.0, 0.0, 0.0];
    let res = reprojection_residuals(&cam, &quats, &t, &points, &uv, &[0], &[0], 1e6);
    assert_eq!(res[0], 1e6);
    assert_eq!(res[1], 0.0);
    assert_eq!(inlier_fraction(&res, 3.0), 0.0);
}

#[test]
fn non_finite_point_is_invalid() {
    let cam = simple_pinhole();
    let points = [f64::NAN, 0.0, -5.0];
    let uv = [0.0, 0.0];
    let res = reprojection_residuals(
        &cam,
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &points,
        &uv,
        &[0],
        &[0],
        f64::INFINITY,
    );
    assert!(res[0].is_infinite());
    assert_eq!(inlier_fraction(&res, 3.0), 0.0);
}

#[test]
fn known_offset_residual() {
    // Shift the observation by a known pixel amount; residual must equal it.
    let cam = simple_pinhole();
    let pts = [[0.5, -0.3, -4.0]];
    let uv_true = project(&cam, &pts);
    let uv = [uv_true[0] - 2.0, uv_true[1] + 1.5]; // observed is offset
    let points: Vec<f64> = pts.iter().flatten().copied().collect();
    let res = reprojection_residuals(
        &cam,
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &points,
        &uv,
        &[0],
        &[0],
        1e6,
    );
    assert!((res[0] - 2.0).abs() < 1e-9);
    assert!((res[1] + 1.5).abs() < 1e-9);
}

#[test]
fn multi_image_indexing() {
    let cam = simple_pinhole();
    // Two images with distinct translations, two shared points.
    let t = [0.0, 0.0, 0.0, 0.5, 0.0, 1.0];
    let world = [[0.2, 0.1, -6.0], [-0.4, 0.3, -5.0]];
    let points: Vec<f64> = world.iter().flatten().copied().collect();
    let quats = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    // observations: (img0,pt0), (img0,pt1), (img1,pt0), (img1,pt1)
    let mut uv = Vec::new();
    for (im, tt) in [[0.0, 0.0, 0.0], [0.5, 0.0, 1.0]].iter().enumerate() {
        let _ = im;
        for p in &world {
            let cp = [p[0] + tt[0], p[1] + tt[1], p[2] + tt[2]];
            let (u, v) = cam.ray_to_pixel(cp).unwrap();
            uv.push(u);
            uv.push(v);
        }
    }
    let obs_img = [0u32, 0, 1, 1];
    let obs_pt = [0u32, 1, 0, 1];
    let res = reprojection_residuals(&cam, &quats, &t, &points, &uv, &obs_img, &obs_pt, 1e6);
    for r in &res {
        assert!(r.abs() < 1e-9, "residual should be ~0, got {r}");
    }
}

// ── Model-genericity: equidistant fisheye, including θ > 90° ───────────────

/// The Phase-1 fisheye-seed camera: equidistant `θ = r/f`, `k1 = 0`.
fn equidistant_seed() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: 130.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    }
}

#[test]
fn fisheye_residuals_are_zero_past_ninety_degrees() {
    let cam = equidistant_seed();
    // Camera-frame points at θ = 30°, 90°, 100° and 140° — the last three
    // have canonical `z ≥ 0`, i.e. behind the image plane but inside a
    // >180° lens' field.
    let mut pts = Vec::new();
    for deg in [30.0f64, 90.0, 100.0, 140.0] {
        let th = deg.to_radians();
        pts.push([3.0 * th.sin(), 0.0, -3.0 * th.cos()]);
        pts.push([0.0, 4.0 * th.sin(), -4.0 * th.cos()]);
    }
    assert!(pts.iter().filter(|p| p[2] > 0.0).count() >= 4);
    let uv = project(&cam, &pts);
    let points: Vec<f64> = pts.iter().flatten().copied().collect();
    let quats = [1.0, 0.0, 0.0, 0.0];
    let t = [0.0, 0.0, 0.0];
    let obs_img = vec![0u32; pts.len()];
    let obs_pt: Vec<u32> = (0..pts.len() as u32).collect();

    let res = reprojection_residuals(&cam, &quats, &t, &points, &uv, &obs_img, &obs_pt, 1e6);
    for (k, chunk) in res.chunks(2).enumerate() {
        assert!(
            chunk[0].hypot(chunk[1]) < 1e-9,
            "observation {k} (θ past 90°) residual {chunk:?}"
        );
    }
    assert_eq!(inlier_fraction(&res, 1e-6), 1.0);
}
