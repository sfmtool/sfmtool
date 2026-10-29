// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests of the cell plane kernel through its own interface: rays built from
//! posed views and per-ray floors the caller chooses, not from a cluster file.

use nalgebra::{Matrix3, Vector3};

use super::*;
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::{RigidTransform, RotQuaternion};

fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 800.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
        rolling_shutter: None,
    }
}

/// `cam_from_world` of a camera at `centre` looking at the origin, −Z forward.
fn look_at(centre: Vector3<f64>) -> RigidTransform {
    let forward = (-centre).normalize();
    let right = forward.cross(&Vector3::y()).normalize();
    let up = right.cross(&forward);
    let r = Matrix3::from_rows(&[right.transpose(), up.transpose(), (-forward).transpose()]);
    RigidTransform::new(RotQuaternion::from_rotation_matrix(r), -(r * centre))
}

fn centres() -> Vec<Vector3<f64>> {
    vec![
        Vector3::new(0.0, 0.0, 10.0),
        Vector3::new(3.0, 0.5, 9.5),
        Vector3::new(-2.5, 2.0, 9.8),
        Vector3::new(0.5, -3.0, 9.6),
    ]
}

/// The rays of each of nine cells at `points`, from every camera at
/// `centres()`, each ray given the floor `floor_px`.
fn rays_to(points: &[Vector3<f64>; 9], floor_px: f64) -> [Vec<CellRay>; 9] {
    rays_from(&centres(), points, floor_px)
}

/// [`rays_to`] from cameras at `centres`, each looking at the origin.
fn rays_from(
    centres: &[Vector3<f64>],
    points: &[Vector3<f64>; 9],
    floor_px: f64,
) -> [Vec<CellRay>; 9] {
    let cam = camera();
    let poses: Vec<RigidTransform> = centres.iter().copied().map(look_at).collect();
    let views: Vec<PosedView<'_>> = poses.iter().map(|p| PosedView::new(&cam, p)).collect();
    std::array::from_fn(|j| {
        views
            .iter()
            .zip(&poses)
            .filter_map(|(view, pose)| {
                let p = pose.transform_point(&points[j].into());
                let (u, v) = cam.project(p.x / -p.z, p.y / -p.z);
                view.ray_through([u, v], floor_px)
            })
            .collect()
    })
}

/// Nine points of a three-by-three grid on the plane through the origin with
/// unit normal `n`, `spacing` apart along `a` and `b`, two in-plane axes.
fn grid_on_plane(n: Vector3<f64>, spacing: f64) -> [Vector3<f64>; 9] {
    let a = n.cross(&Vector3::x()).normalize();
    let b = n.cross(&a);
    std::array::from_fn(|j| {
        let (r, c) = ((j / 3) as f64 - 1.0, (j % 3) as f64 - 1.0);
        spacing * (c * a + r * b)
    })
}

fn angle_deg(a: [f64; 3], b: Vector3<f64>) -> f64 {
    let a = Vector3::new(a[0], a[1], a[2]);
    a.dot(&b).abs().min(1.0).acos().to_degrees()
}

#[test]
fn rays_from_posed_views_give_the_plane_and_both_axes() {
    let n = Vector3::new(0.2, -0.1, 1.0).normalize();
    let out = fit_cell_plane(
        &rays_to(&grid_on_plane(n, 0.1), 0.1),
        &CellPlaneParams::default(),
    );
    assert_eq!(out.determinacy, NormalDeterminacy::BothAxes);
    assert!(out
        .cell_status
        .iter()
        .all(|&s| s == CellPlaneStatus::InPlane));
    assert!(out.cell_rays.iter().all(|&k| k == 4));
    assert!(angle_deg(out.normal, n) < 1e-6, "{:?}", out.normal);
}

#[test]
fn cells_along_a_line_fix_one_axis() {
    // Nine distinct points on the world X axis.
    let points: [Vector3<f64>; 9] = std::array::from_fn(|j| {
        Vector3::new(
            0.1 * ((j % 3) as f64 - 1.0) + 0.01 * (j / 3) as f64,
            0.0,
            0.0,
        )
    });
    let out = fit_cell_plane(&rays_to(&points, 0.1), &CellPlaneParams::default());
    let NormalDeterminacy::OneAxis { free_axis } = out.determinacy else {
        panic!("expected one axis, got {:?}", out.determinacy);
    };
    assert!(angle_deg(free_axis, Vector3::x()) < 1e-6, "{free_axis:?}");
}

#[test]
fn the_caller_s_ray_floor_sets_the_cell_weights() {
    // Noise-free rays: each cell's residual is below every floor, so its
    // variance is the floor's square and doubling the floor quarters its
    // weight.
    let n = Vector3::new(0.1, 0.2, 1.0).normalize();
    let points = grid_on_plane(n, 0.1);
    let params = CellPlaneParams::default();
    let one = fit_cell_plane(&rays_to(&points, 0.5), &params);
    let two = fit_cell_plane(&rays_to(&points, 1.0), &params);
    for j in 0..9 {
        assert!(one.cell_residual_px[j] < 1e-6);
        let ratio = one.cell_weight[j] / two.cell_weight[j];
        assert!((ratio - 4.0).abs() < 1e-6, "cell {j}: {ratio}");
    }
}

#[test]
fn a_cell_with_one_ray_is_not_triangulated() {
    let n = Vector3::new(0.0, 0.0, 1.0);
    let mut rays = rays_to(&grid_on_plane(n, 0.1), 0.1);
    rays[4].truncate(1);
    let out = fit_cell_plane(&rays, &CellPlaneParams::default());
    assert_eq!(out.cell_rays[4], 1);
    assert_eq!(out.cell_status[4], CellPlaneStatus::TooFewRays);
    assert!(out.cell_positions[4].iter().all(|v| v.is_nan()));
    assert_eq!(out.determinacy, NormalDeterminacy::BothAxes);
}

/// The cameras of [`centres`] drawn toward the first one's axis by `scale`:
/// at `1` the views span about 20 degrees, at `0.15` about 3.
fn centres_scaled(scale: f64) -> Vec<Vector3<f64>> {
    centres()
        .into_iter()
        .map(|c| Vector3::new(scale * c.x, scale * c.y, c.z))
        .collect()
}

#[test]
fn the_predicted_precision_is_small_with_a_wide_baseline_and_large_with_a_narrow_one() {
    let n = Vector3::new(0.2, -0.1, 1.0).normalize();
    let points = grid_on_plane(n, 0.1);
    let params = CellPlaneParams::default();
    let wide = fit_cell_plane(&rays_from(&centres_scaled(1.0), &points, 0.1), &params);
    let narrow = fit_cell_plane(&rays_from(&centres_scaled(0.15), &points, 0.1), &params);
    assert_eq!(wide.determinacy, NormalDeterminacy::BothAxes);
    assert_eq!(narrow.determinacy, NormalDeterminacy::BothAxes);
    let [wide_weak, wide_strong] = normal_precision_deg(&wide);
    let [narrow_weak, narrow_strong] = normal_precision_deg(&narrow);
    assert!(wide_weak.is_finite() && wide_strong.is_finite());
    assert!(wide_strong <= wide_weak, "{wide_strong} {wide_weak}");
    assert!(
        wide_weak < 1.0,
        "a wide baseline pins the normal: {wide_weak}"
    );
    assert!(
        narrow_weak > 3.0 * wide_weak,
        "a narrow baseline does not: {narrow_weak} against {wide_weak}"
    );
    assert!(narrow_strong.is_finite());
}

#[test]
fn the_predicted_precision_follows_the_ray_floor() {
    // Noise-free rays: each cell's variance is its floor's square, so
    // doubling the floor doubles the predicted precision.
    let n = Vector3::new(0.1, 0.2, 1.0).normalize();
    let points = grid_on_plane(n, 0.1);
    let params = CellPlaneParams::default();
    let one = normal_precision_deg(&fit_cell_plane(&rays_to(&points, 0.5), &params));
    let two = normal_precision_deg(&fit_cell_plane(&rays_to(&points, 1.0), &params));
    for k in 0..2 {
        assert!((two[k] / one[k] - 2.0).abs() < 1e-6, "{one:?} {two:?}");
    }
}

#[test]
fn no_precision_is_predicted_where_no_axis_is_fixed() {
    let empty = fit_cell_plane(&Default::default(), &CellPlaneParams::default());
    assert!(normal_precision_deg(&empty).iter().all(|v| v.is_nan()));
}
