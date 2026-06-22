// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::intrinsics::{CameraIntrinsics, CameraModel};
use crate::camera::warp_map::WarpMap;
use nalgebra::{Point3, Vector3};

fn pinhole(f: f64, cx: f64, cy: f64, w: u32, h: u32) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: f,
            focal_length_y: f,
            principal_point_x: cx,
            principal_point_y: cy,
        },
        width: w,
        height: h,
    }
}

#[test]
fn to_world_and_normal() {
    let p = OrientedPatch::new(
        Point3::new(1.0, 2.0, 3.0),
        Vector3::x(),
        Vector3::y(),
        [2.0, 4.0],
    );
    assert!((p.to_world(0.0, 0.0) - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
    assert!((p.to_world(1.0, 1.0) - Point3::new(3.0, 6.0, 3.0)).norm() < 1e-12);
    assert!((p.normal() - Vector3::z()).norm() < 1e-12);
}

#[test]
fn from_center_normal_is_orthonormal_and_preserves_normal() {
    let normal = Vector3::new(0.3, -0.5, 0.8).normalize();
    let p = OrientedPatch::from_center_normal(Point3::origin(), normal, Vector3::y(), [1.0, 1.0]);
    assert!((p.u_axis.norm() - 1.0).abs() < 1e-9);
    assert!((p.v_axis.norm() - 1.0).abs() < 1e-9);
    assert!(p.u_axis.dot(&p.v_axis).abs() < 1e-9);
    assert!((p.normal() - normal).norm() < 1e-9);
}

#[test]
fn from_center_normal_handles_up_hint_parallel_to_normal() {
    let normal = Vector3::z();
    let p = OrientedPatch::from_center_normal(Point3::origin(), normal, Vector3::z(), [1.0, 1.0]);
    assert!(p.u_axis.dot(&p.v_axis).abs() < 1e-9);
    assert!((p.normal() - normal).norm() < 1e-9);
}

#[test]
fn is_front_facing_uses_outward_normal() {
    let pose = RigidTransform::identity(); // camera at origin looking +Z
    let front = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 5.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::y(),
        [1.0, 1.0],
    );
    assert!(front.is_front_facing(&pose));
    let back = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 5.0),
        Vector3::z(),
        Vector3::y(),
        [1.0, 1.0],
    );
    assert!(!back.is_front_facing(&pose));
}

#[test]
fn from_patch_projects_fronto_parallel_plane() {
    let (f, cx, cy) = (500.0, 320.0, 240.0);
    let cam = pinhole(f, cx, cy, 640, 480);
    let pose = RigidTransform::identity(); // world == camera frame
    let (d, h) = (4.0, 0.5);
    let patch = OrientedPatch::new(Point3::new(0.0, 0.0, d), Vector3::x(), Vector3::y(), [h, h]);
    let r = 2u32;
    let wm = WarpMap::from_patch(&patch, &cam, &pose, r);

    let step = 2.0 / r as f64;
    for col in 0..r {
        for row in 0..r {
            let s = (col as f64 + 0.5) * step - 1.0;
            let t = (row as f64 + 0.5) * step - 1.0;
            let exp_x = f * (s * h) / d + cx;
            let exp_y = f * (t * h) / d + cy;
            let (gx, gy) = wm.get(col, row);
            assert!((gx as f64 - exp_x).abs() < 1e-3, "x {gx} vs {exp_x}");
            assert!((gy as f64 - exp_y).abs() < 1e-3, "y {gy} vs {exp_y}");
        }
    }
}

#[test]
fn mean_viewing_normal_points_at_camera_cluster() {
    let center = Point3::origin();
    // Cameras symmetric about the +Z axis -> mean viewing direction is +Z.
    let cams = [
        Point3::new(1.0, 0.0, 2.0),
        Point3::new(-1.0, 0.0, 2.0),
        Point3::new(0.0, 1.0, 2.0),
        Point3::new(0.0, -1.0, 2.0),
    ];
    let n = mean_viewing_normal(&center, &cams);
    assert!((n - Vector3::z()).norm() < 1e-9);
}

#[test]
fn pca_plane_normal_of_coplanar_points() {
    // Points spread in the z = 3 plane -> normal is +/- Z.
    let pts = [
        Point3::new(0.0, 0.0, 3.0),
        Point3::new(1.0, 0.0, 3.0),
        Point3::new(0.0, 1.0, 3.0),
        Point3::new(1.0, 1.0, 3.0),
        Point3::new(-1.0, 0.5, 3.0),
    ];
    let n = pca_plane_normal(&pts);
    assert!(n.z.abs() > 1.0 - 1e-6);
    assert!(n.x.abs() < 1e-6 && n.y.abs() < 1e-6);
}

#[test]
fn reduce_across_views() {
    let base = [2.0, 5.0, 3.0, 9.0];
    assert_eq!(reduce(&mut base.to_vec(), ViewReduce::Min), 2.0);
    assert_eq!(reduce(&mut base.to_vec(), ViewReduce::Max), 9.0);
    assert!((reduce(&mut base.to_vec(), ViewReduce::Mean) - 4.75).abs() < 1e-12);
    // even count -> average of the two middle values ([2,3,5,9] -> (3+5)/2)
    assert_eq!(reduce(&mut base.to_vec(), ViewReduce::Median), 4.0);
    assert_eq!(reduce(&mut [3.0, 1.0, 2.0], ViewReduce::Median), 2.0);
}

#[test]
fn feature_size_without_sift_is_an_error() {
    // The demo reconstruction has no workspace `.sift` files, so no keypoint
    // scale is readable for any point. FeatureSize must error rather than fall
    // back to a substitute size; the other extent policies still succeed.
    let recon = SfmrReconstruction::demo(12);

    let err = PatchCloud::from_reconstruction(
        &recon,
        PatchNormal::MeanViewing,
        PatchExtent::FeatureSize {
            factor: 5.0,
            across: ViewReduce::Median,
        },
    )
    .unwrap_err();
    assert!(matches!(err, PatchCloudError::MissingFeatureScale { .. }));

    let cloud =
        PatchCloud::from_reconstruction(&recon, PatchNormal::MeanViewing, PatchExtent::Fixed(0.1))
            .expect("Fixed extent needs no sift files");
    assert_eq!(cloud.len(), 12);
}

#[test]
fn from_patch_behind_camera_is_invalid() {
    let cam = pinhole(500.0, 320.0, 240.0, 640, 480);
    let pose = RigidTransform::identity();
    let patch = OrientedPatch::new(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::x(),
        Vector3::y(),
        [0.5, 0.5],
    );
    let wm = WarpMap::from_patch(&patch, &cam, &pose, 4);
    assert!(!wm.is_valid(0, 0));
}
