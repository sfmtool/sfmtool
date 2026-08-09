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

// ── Model-genericity: equidistant fisheye, including θ > 90° ───────────────
//
// The kernel takes the camera object and never touches `z` itself: its domain
// test is whatever `ray_to_pixel` says, and its Jacobian falls back to a
// central difference for models with no analytic one. These pin both under
// the Phase-1 seed model `SimpleRadialFisheye { k1 = 0 }` with observations
// deliberately past 90° off-axis (canonical `z_cam > 0`).

/// The fisheye-seed camera: equidistant `θ = r/f`, centred principal point.
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
    }
}

/// The same map as a native `EquidistantFisheye` — closed form, and the one
/// ray-path model with an analytic pixel Jacobian.
fn equidistant_native() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 130.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
    }
}

/// Wide-FOV scene: points spread over θ ∈ [5°, 130°] around the true pose,
/// so a large share of the observations are behind the image plane. Returns
/// the scene plus the count of `z_cam > 0` observations.
fn make_fisheye_scene() -> (Scene, usize) {
    make_fisheye_scene_for(equidistant_seed())
}

/// [`make_fisheye_scene`] under an arbitrary equidistant camera.
fn make_fisheye_scene_for(cam: CameraIntrinsics) -> (Scene, usize) {
    let r_true = UnitQuaternion::from_scaled_axis(Vector3::new(0.15, -0.1, 0.05));
    let t_true = Vector3::new(0.3, -0.2, 0.4);
    let r_inv = r_true.inverse();
    let mut points = Vec::new();
    let mut uv = Vec::new();
    let mut n_behind = 0usize;
    for i in 0..60 {
        let theta = (5.0 + 125.0 * i as f64 / 59.0).to_radians();
        let phi = 2.399_963 * i as f64; // golden-angle azimuth spread
        let range = 4.0 + 2.0 * jitter(i, 5);
        let c = Vector3::new(
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            -theta.cos(),
        ) * range;
        if c.z > 0.0 {
            n_behind += 1;
        }
        let (u, v) = cam.ray_to_pixel([c.x, c.y, c.z]).expect("in domain");
        let w = r_inv * (c - t_true);
        points.push([w.x, w.y, w.z]);
        uv.push([u, v]);
    }
    ((cam, r_true, t_true, points, uv), n_behind)
}

#[test]
fn refines_a_wide_fov_fisheye_pose() {
    let ((cam, r_true, t_true, points, uv), n_behind) = make_fisheye_scene();
    assert!(
        n_behind >= 18,
        "scene not wide enough ({n_behind} past 90°)"
    );
    let r0 = UnitQuaternion::from_scaled_axis(Vector3::new(0.15 + 0.05, -0.1 - 0.04, 0.05 + 0.03));
    let t0 = t_true + Vector3::new(0.08, -0.06, 0.09);
    let out = refine_absolute_pose(&cam, &uv, &points, &r0, &t0, 5, 0.6, 3.0);
    assert!(
        out.inlier_fraction > 0.99,
        "inlier fraction {} — backward observations are being dropped",
        out.inlier_fraction
    );
    let ang = out.rotation.angle_to(&r_true);
    assert!(ang < 1e-6, "rotation error {ang} rad");
    assert!(
        (out.translation - t_true).norm() < 1e-6,
        "translation error {}",
        (out.translation - t_true).norm()
    );
}

#[test]
fn fisheye_residuals_are_finite_past_ninety_degrees() {
    // The failure mode this guards: a `z > 0` domain rejection turning every
    // backward observation into INVALID_RESIDUAL, which the trim then removes.
    let ((cam, r_true, t_true, points, uv), _) = make_fisheye_scene();
    let axis = r_true.scaled_axis();
    let p: Params = [axis.x, axis.y, axis.z, t_true.x, t_true.y, t_true.z];
    let rn = residual_norms(&cam, &uv, &points, &p);
    assert_eq!(rn.len(), uv.len());
    for (i, r) in rn.iter().enumerate() {
        assert!(*r < 1e-9, "observation {i} residual {r} at the true pose");
    }
}

// ── Analytic vs central-difference Jacobian: same answer, fewer projections ──

/// The two representations of `θ = r/f` — `SimpleRadialFisheye { k1 = 0 }`
/// (no analytic Jacobian, so `lm_fit` central-differences `ray_to_pixel`)
/// and `EquidistantFisheye` (analytic) — must converge to the same pose from
/// the same perturbed start.
///
/// The projections themselves are identical, so the only difference between
/// the two runs is how each LM step is linearized. The converged poses agree
/// far below the residual floor of the scene: `1e-10` rad of rotation and
/// `1e-10` of translation are two orders tighter than the `1e-6` accuracy
/// each arm is separately asserted to reach against the planted truth.
#[test]
fn analytic_and_central_difference_jacobians_converge_together() {
    let (legacy, n_behind) = make_fisheye_scene_for(equidistant_seed());
    let (native, _) = make_fisheye_scene_for(equidistant_native());
    assert!(
        n_behind >= 18,
        "scene not wide enough ({n_behind} past 90°)"
    );
    // The arms really do take different paths through `project_with_jac`.
    assert!(!legacy.0.model.supports_pixel_jacobian());
    assert!(native.0.model.supports_pixel_jacobian());
    // Same observations to begin with.
    for (a, b) in legacy.4.iter().zip(native.4.iter()) {
        assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
    }

    let r0 = UnitQuaternion::from_scaled_axis(Vector3::new(0.15 + 0.05, -0.1 - 0.04, 0.05 + 0.03));
    let t0 = legacy.2 + Vector3::new(0.08, -0.06, 0.09);
    let a = refine_absolute_pose(&legacy.0, &legacy.4, &legacy.3, &r0, &t0, 5, 0.6, 3.0);
    let b = refine_absolute_pose(&native.0, &native.4, &native.3, &r0, &t0, 5, 0.6, 3.0);

    assert!(
        a.rotation.angle_to(&b.rotation) < 1e-10,
        "rotation disagreement {} rad between the two Jacobian paths",
        a.rotation.angle_to(&b.rotation)
    );
    assert!(
        (a.translation - b.translation).norm() < 1e-10,
        "translation disagreement {}",
        (a.translation - b.translation).norm()
    );
    assert!((a.inlier_fraction - b.inlier_fraction).abs() < 1e-12);
    // Both are also correct against the planted truth.
    assert!(b.rotation.angle_to(&legacy.1) < 1e-6);
    assert!((b.translation - legacy.2).norm() < 1e-6);
}
