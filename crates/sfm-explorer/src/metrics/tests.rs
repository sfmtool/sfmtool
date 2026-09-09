// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The triangulation numerics on their own, with no panel around them.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use sfmtool_core::{CameraIntrinsics, CameraModel, Point3D, SfmrImage, SfmrReconstruction};

use super::{compute_max_pairwise_angle, compute_observation_metrics, compute_point_diagnostics};

fn simple_camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 100.0,
            principal_point_x: 50.0,
            principal_point_y: 50.0,
        },
        width: 100,
        height: 100,
    }
}

fn image_at(translation: [f64; 3]) -> SfmrImage {
    SfmrImage {
        name: "img".to_string(),
        camera_index: 0,
        quaternion_wxyz: UnitQuaternion::identity(),
        translation_xyz: Vector3::new(translation[0], translation[1], translation[2]),
    }
}

fn bearing(direction: [f64; 3]) -> Point3D {
    Point3D {
        position: Point3::new(direction[0], direction[1], direction[2]),
        w: 0.0,
        color: [0, 0, 0],
        error: 0.0,
        normal: Vector3::zeros(),
    }
}

#[test]
fn observation_metrics_are_defined_for_a_point_at_infinity() {
    // A bearing down the optical axis lands on the principal point from any
    // camera position: the direction rotates but never translates.
    let camera = simple_camera();
    for t in [[0.0, 0.0, 0.0], [5.0, -7.0, 3.0]] {
        let (err, angle) = compute_observation_metrics(
            &bearing([0.0, 0.0, -1.0]),
            &image_at(t),
            &camera,
            [50.0, 50.0],
        );
        assert!(err.abs() < 1e-4, "reproj error {err}");
        assert!(angle.abs() < 1e-4, "ray angle {angle}");
    }
}

#[test]
fn observation_metrics_of_a_backward_bearing_are_undefined() {
    let (err, angle) = compute_observation_metrics(
        &bearing([0.0, 0.0, 1.0]),
        &image_at([0.0, 0.0, 0.0]),
        &simple_camera(),
        [50.0, 50.0],
    );
    assert!(err.is_nan());
    assert!(angle.is_nan());
}

#[test]
fn max_pairwise_angle_finds_the_widest_pair() {
    // Three rays: 0°, 45° and 90° from +X. The widest pair is the outer two.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let rays = [[1.0, 0.0, 0.0], [s, s, 0.0], [0.0, 1.0, 0.0]];
    let angle = compute_max_pairwise_angle(&rays);
    assert!((angle - 90.0).abs() < 1e-4, "angle was {angle}");
}

#[test]
fn max_pairwise_angle_of_fewer_than_two_rays_is_zero() {
    assert_eq!(compute_max_pairwise_angle(&[]), 0.0);
    assert_eq!(compute_max_pairwise_angle(&[[1.0, 0.0, 0.0]]), 0.0);
}

#[test]
fn point_diagnostics_are_undefined_for_a_missing_point() {
    // A missing point has no view at all, which is the caller's `None` rather
    // than a value this function returns.
    let edited =
        sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(SfmrReconstruction::demo(4)));
    assert!(edited.point(999).is_none());
}

#[test]
fn point_diagnostics_are_finite_for_a_triangulated_point() {
    let recon = SfmrReconstruction::demo(12);
    let edited = sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(recon));
    let view = edited.point(5).expect("a live point");
    let (cond, z) = compute_point_diagnostics(&edited.base.image_table, &view);
    assert!(cond.is_finite() && cond >= 1.0, "condition number {cond}");
    assert!(z.is_finite(), "inverse-depth z {z}");
}
