// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the ray-space two-view estimators.
//!
//! Every fixture plants a scene whose rays reach past the hemisphere
//! (`θ ≥ 90°`, i.e. `ray.z < 0`), because that is exactly the population a
//! `z = 1` normalization would silently drop.

use super::*;
use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

/// Deterministic pseudo-random unit ray, spread over the whole sphere out to
/// `theta_max`.
fn planted_rays(n: usize, theta_max: f64) -> Vec<Vector3<f64>> {
    (0..n)
        .map(|i| {
            let t = (i as f64 + 0.5) / n as f64;
            let theta = theta_max * t;
            let phi = 2.399_963_23 * i as f64;
            Vector3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            )
        })
        .collect()
}

/// A rotation of `angle` radians about a fixed slanted axis.
fn planted_rotation(angle: f64) -> Matrix3<f64> {
    let axis = Unit::new_normalize(Vector3::new(0.31, -0.87, 0.38));
    Rotation3::from_axis_angle(&axis, angle).into_inner()
}

/// A planted two-view scene: the two ray sets and the relative pose behind
/// them.
type TwoViewScene = (
    Vec<Vector3<f64>>,
    Vec<Vector3<f64>>,
    Matrix3<f64>,
    Vector3<f64>,
);

/// Two-view scene: points at planted depths along the image-1 rays, seen from
/// a second camera at `(R, t)`. Returns the two ray sets and the poses.
fn two_view_scene(n: usize, theta_max: f64, angle: f64, t: Vector3<f64>) -> TwoViewScene {
    let rot = planted_rotation(angle);
    let r1 = planted_rays(n, theta_max);
    let r2 = r1
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let depth = 3.0 + 2.0 * ((0.7 * i as f64).sin() + 1.0);
            (rot * (d * depth) + t).normalize()
        })
        .collect();
    (r1, r2, rot, t)
}

fn n_beyond_hemisphere(rays: &[Vector3<f64>]) -> usize {
    rays.iter().filter(|r| r.z <= 0.0).count()
}

// ── Epipolar cell ───────────────────────────────────────────────────────────

#[test]
fn essential_recovers_a_planted_two_view_geometry_past_the_hemisphere() {
    let (r1, r2, rot, t) = two_view_scene(160, 2.0, 0.35, Vector3::new(0.9, -0.25, 0.4));
    assert!(
        n_beyond_hemisphere(&r1) > 30,
        "fixture must exercise theta >= 90 deg"
    );
    let out = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 1e-6,
            min_inliers: 20,
            ..Default::default()
        },
    )
    .expect("a noise-free scene has an epipolar consensus");
    assert!(out.inliers.iter().all(|&b| b), "every planted ray is exact");
    assert!(
        out.essentialness < 1e-9,
        "the exact geometry is essential: {}",
        out.essentialness
    );
    // E = [t]x R up to sign and scale.
    let tx = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    let truth = (tx * rot).normalize();
    let got = out.e_matrix;
    let d = (got - truth).norm().min((got + truth).norm());
    assert!(d < 1e-9, "epipolar matrix off by {d}");
}

#[test]
fn essential_rejects_outliers_and_keeps_the_planted_inliers() {
    let (mut r1, r2, _rot, _t) = two_view_scene(200, 2.1, 0.3, Vector3::new(0.6, 0.3, -0.5));
    // Corrupt a quarter of the correspondences by rotating one side far away.
    let junk = planted_rotation(0.9);
    for i in (0..r1.len()).step_by(4) {
        r1[i] = junk * r1[i];
    }
    let out = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 1e-4,
            min_inliers: 20,
            samples: 800,
            ..Default::default()
        },
    )
    .expect("three quarters of the population is exact");
    let kept = out.inliers.iter().filter(|&&b| b).count();
    assert!(
        kept >= 145,
        "kept only {kept} of ~150 clean correspondences"
    );
    assert!(
        (0..r1.len()).step_by(4).filter(|&i| out.inliers[i]).count() <= 3,
        "outliers leaked into the consensus"
    );
}

#[test]
fn essential_is_deterministic_in_the_seed() {
    let (r1, r2, _rot, _t) = two_view_scene(120, 1.9, 0.25, Vector3::new(0.4, 0.1, 0.8));
    let opts = RayEssentialOptions {
        max_angle_rad: 1e-5,
        min_inliers: 20,
        seed: 7,
        ..Default::default()
    };
    let a = estimate_essential_rays(&r1, &r2, &opts).unwrap();
    let b = estimate_essential_rays(&r1, &r2, &opts).unwrap();
    assert_eq!(a.inliers, b.inliers);
    assert_eq!(a.e_matrix, b.e_matrix);
}

#[test]
fn essential_abstains_below_the_inlier_floor() {
    let (r1, mut r2, _rot, _t) = two_view_scene(60, 1.8, 0.3, Vector3::new(0.5, 0.5, 0.2));
    // Scrambled partners: no epipolar geometry explains them.
    r2.rotate_left(17);
    assert!(estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 1e-6,
            min_inliers: 40,
            ..Default::default()
        },
    )
    .is_none());
    // A degenerate tolerance is rejected outright rather than silently taken.
    assert!(estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 0.0,
            ..Default::default()
        },
    )
    .is_none());
}

#[test]
fn one_sided_residuals_are_different_measurements() {
    let (r1, r2, _rot, _t) = two_view_scene(150, 2.0, 0.4, Vector3::new(0.7, -0.4, 0.3));
    let base = RayEssentialOptions {
        max_angle_rad: 3e-3,
        min_inliers: 20,
        ..Default::default()
    };
    let both = estimate_essential_rays(&r1, &r2, &base).unwrap();
    let one = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            side: EpipolarSide::One,
            ..base
        },
    )
    .unwrap();
    let two = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            side: EpipolarSide::Two,
            ..base
        },
    )
    .unwrap();
    // The two-sided residual is the max of the one-sided ones, so it never
    // scores a correspondence better than either side does.
    for i in 0..r1.len() {
        assert!(both.residuals_rad[i] >= one.residuals_rad[i] - 1e-12);
        assert!(both.residuals_rad[i] >= two.residuals_rad[i] - 1e-12);
    }
}

// ── Rotation cell ───────────────────────────────────────────────────────────

#[test]
fn rotation_recovers_a_pure_ray_rotation_past_the_hemisphere() {
    let rot = planted_rotation(0.6);
    let r1 = planted_rays(180, 2.2);
    let r2: Vec<Vector3<f64>> = r1.iter().map(|d| rot * d).collect();
    assert!(n_beyond_hemisphere(&r1) > 50);
    let out = fit_ray_rotation(
        &r1,
        &r2,
        &RayRotationOptions {
            max_angle_rad: 1e-6,
            ..Default::default()
        },
    )
    .expect("a pure rotation is explained exactly");
    assert!(out.inliers.iter().all(|&b| b));
    assert!(out.rms_rad < 1e-7, "rms {}", out.rms_rad);
    assert!((out.rotation - rot).norm() < 1e-12);
}

#[test]
fn rotation_abstains_on_a_pair_with_real_parallax() {
    // Points at widely different depths under a real baseline are not a
    // rotation of each other; no rotation explains enough of them.
    let (r1, r2, _rot, _t) = two_view_scene(200, 2.0, 0.3, Vector3::new(1.6, -0.9, 0.7));
    assert!(fit_ray_rotation(
        &r1,
        &r2,
        &RayRotationOptions {
            max_angle_rad: 1e-4,
            min_inliers: 60,
            ..Default::default()
        },
    )
    .is_none());
}

#[test]
fn rotation_separates_a_far_field_from_its_near_outliers() {
    let rot = planted_rotation(0.45);
    let r1 = planted_rays(200, 2.1);
    let mut r2: Vec<Vector3<f64>> = r1.iter().map(|d| rot * d).collect();
    // A near-field minority carries genuine parallax and must be excluded.
    let t = Vector3::new(0.8, 0.2, -0.5);
    for i in (0..r1.len()).step_by(5) {
        r2[i] = (rot * (r1[i] * 1.2) + t).normalize();
    }
    let out = fit_ray_rotation(
        &r1,
        &r2,
        &RayRotationOptions {
            max_angle_rad: 1e-3,
            min_inliers: 40,
            ..Default::default()
        },
    )
    .expect("the far field dominates");
    assert!((out.rotation - rot).norm() < 1e-6);
    let near_kept = (0..r1.len()).step_by(5).filter(|&i| out.inliers[i]).count();
    assert!(near_kept <= 2, "{near_kept} near-field rays kept");
}

#[test]
fn rotation_is_deterministic_in_the_seed() {
    let rot = planted_rotation(0.5);
    let r1 = planted_rays(120, 2.0);
    let r2: Vec<Vector3<f64>> = r1.iter().map(|d| rot * d).collect();
    let opts = RayRotationOptions {
        max_angle_rad: 1e-5,
        seed: 11,
        ..Default::default()
    };
    let a = fit_ray_rotation(&r1, &r2, &opts).unwrap();
    let b = fit_ray_rotation(&r1, &r2, &opts).unwrap();
    assert_eq!(a.inliers, b.inliers);
    assert_eq!(a.rotation, b.rotation);
}
