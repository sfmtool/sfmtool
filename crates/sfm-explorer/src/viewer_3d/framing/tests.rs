// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::scene::SceneNode;
use sfmtool_core::SfmrReconstruction;

/// A demo node whose points named by `infinite` are at infinity, each along
/// the direction `bearing(i)` gives it.
fn node_with(
    points: usize,
    infinite: impl Fn(usize) -> Option<Vector3<f64>>,
) -> (SceneNode, Vec<Point3<f64>>) {
    let mut recon = SfmrReconstruction::demo(points);
    let mut finite = Vec::new();
    for (i, point) in recon.point_set.points.iter_mut().enumerate() {
        match infinite(i) {
            Some(direction) => {
                point.position = Point3::from(direction.normalize());
                point.w = 0.0;
            }
            None => finite.push(point.position),
        }
    }
    let node = SceneNode::from_path(std::path::Path::new("/runs/framing.sfmr"), recon);
    (node, finite)
}

/// A unit vector `degrees` round from +X in the horizontal plane, raised by
/// `elevation` degrees.
fn direction(degrees: f64, elevation: f64) -> Vector3<f64> {
    let (a, e) = (degrees.to_radians(), elevation.to_radians());
    Vector3::new(a.cos() * e.cos(), a.sin() * e.cos(), e.sin())
}

fn angle_deg(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.normalize()
        .dot(&b.normalize())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn forward_of(end: &FitEnd) -> Vector3<f64> {
    end.orientation.inverse() * Vector3::new(0.0, 0.0, -1.0)
}

fn right_of(end: &FitEnd) -> Vector3<f64> {
    end.orientation.inverse() * Vector3::x()
}

/// A camera looking somewhere other than the default, rolled, so a framing
/// that leaves it alone and one that moves it cannot be confused.
fn rolled_camera() -> ViewportCamera {
    let mut camera = ViewportCamera {
        world_up: Vector3::new(0.3, 0.0, 1.0).normalize(),
        ..ViewportCamera::default()
    };
    camera.camera.position = Point3::new(7.0, 3.0, -2.0);
    camera.set_orientation_from_forward(Vector3::new(-1.0, 0.2, 0.1).normalize());
    camera
}

#[test]
fn a_mixed_node_is_framed_on_its_finite_points_alone() {
    let (node, finite) = node_with(400, |i| (i % 3 == 0).then(|| direction(i as f64, 10.0)));
    let points = FitPoints::of(&node);
    assert_eq!(points.positions.len(), finite.len());
    assert_eq!(points.bearings.len(), 400 - finite.len());
    for (got, want) in points.positions.iter().zip(&finite) {
        assert!((got - want).norm() < 1e-12, "{got} is not {want}");
    }

    let camera = rolled_camera();
    let end = camera.compute_fit(&points, 1.5, true).expect("framed");
    let (position, distance) = camera
        .compute_zoom_to_fit(&finite, 1.5)
        .expect("finite points frame");
    assert!((end.position - position).norm() < 1e-9);
    assert!((end.target_distance - distance).abs() < 1e-9);
    // The finite framing keeps the orientation and the roll.
    assert_eq!(end.orientation, camera.camera.orientation);
    assert_eq!(end.world_up, camera.world_up);
}

#[test]
fn a_panorama_with_a_clear_mean_is_looked_along_from_the_starting_position() {
    // Bearings within 25 degrees of azimuth 120, elevation 15.
    let centre = direction(120.0, 15.0);
    let (node, _) = node_with(300, |i| {
        let spread = (i % 50) as f64 - 25.0;
        Some(direction(120.0 + spread, 15.0 + (i % 7) as f64 - 3.0))
    });
    let points = FitPoints::of(&node);
    assert!(points.positions.is_empty());
    assert_eq!(points.bearings.len(), 300);

    let camera = rolled_camera();
    let end = camera.compute_fit(&points, 1.5, true).expect("framed");
    let start = ViewportCamera::default();
    assert_eq!(end.position, start.camera.position);
    assert_eq!(end.target_distance, start.camera.target_distance);
    assert!(angle_deg(&forward_of(&end), &centre) < 2.0);
    // Maintain Z-up on: level, whatever roll the camera had.
    assert_eq!(end.world_up, Vector3::z());
    assert!(right_of(&end).z.abs() < 1e-9, "the view is not level");

    // Maintain Z-up off: the current up is kept.
    let free = camera.compute_fit(&points, 1.5, false).expect("framed");
    assert_eq!(free.world_up, camera.world_up);
    assert!(angle_deg(&forward_of(&free), &centre) < 2.0);
}

#[test]
fn a_panorama_spread_around_the_sphere_looks_at_its_largest_cluster() {
    // A 360 degree ring on the horizon, plus a denser cluster at azimuth 0,
    // a smaller one at 90 and another at 180. The mean, (10, 30, 0) over the
    // clusters, points near 72 degrees, where little is.
    let mut bearings: Vec<Vector3<f64>> = (0..360).map(|a| direction(a as f64, 0.0)).collect();
    bearings.extend((0..40).map(|i| direction((i % 9) as f64 - 4.0, (i % 5) as f64 - 2.0)));
    bearings.extend((0..30).map(|i| direction(90.0 + (i % 9) as f64 - 4.0, 0.0)));
    bearings.extend((0..30).map(|i| direction(180.0 + (i % 9) as f64 - 4.0, 0.0)));
    let total = bearings.len();
    let (node, _) = node_with(total, |i| Some(bearings[i]));
    let points = FitPoints::of(&node);
    let sum = points
        .bearings
        .iter()
        .fold(Vector3::zeros(), |acc, b| acc + b);
    assert!(sum.norm() / (total as f64) < MIN_MEAN_RESULTANT_LENGTH);

    let camera = rolled_camera();
    let end = camera.compute_fit(&points, 1.5, true).expect("framed");
    assert_eq!(end.position, ViewportCamera::default().camera.position);
    assert!(
        angle_deg(&forward_of(&end), &Vector3::x()) < 5.0,
        "looked along {} rather than the cluster at +X",
        forward_of(&end)
    );
    assert!(right_of(&end).z.abs() < 1e-9, "the view is not level");
    // The same bearings frame the same way every time.
    assert_eq!(camera.compute_fit(&points, 1.5, true), Some(end));
}

#[test]
fn bearings_evenly_spread_over_the_sphere_still_frame_deterministically() {
    // A Fibonacci sphere: no direction is preferred, so the mean is short and
    // the cluster rule picks one; it must pick the same one each time.
    let n = 1000;
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let bearings: Vec<Vector3<f64>> = (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let a = golden * i as f64;
            Vector3::new(r * a.cos(), r * a.sin(), z)
        })
        .collect();
    let (node, _) = node_with(n, |i| Some(bearings[i]));
    let points = FitPoints::of(&node);
    let camera = rolled_camera();
    let end = camera.compute_fit(&points, 1.5, true).expect("framed");
    assert_eq!(end.position, ViewportCamera::default().camera.position);
    assert!((forward_of(&end).norm() - 1.0).abs() < 1e-9);
    assert_eq!(camera.compute_fit(&points, 1.5, true), Some(end));
}

/// The fit depends on how the camera is turned and not on where it stands, so
/// a camera left far away -- framing a node drawn at a scale of 1e20 -- frames
/// a node of ordinary size exactly as a camera beside it does, rather than
/// landing on a rounded place with no distance to its target.
#[test]
fn a_fit_from_a_camera_far_away_is_the_fit_from_close_by() {
    let (_, finite) = node_with(400, |_| None);
    let near = rolled_camera();
    let mut far = rolled_camera();
    far.camera.position = Point3::new(-1.9e20, -5.7e21, 3.3e21);
    let (want, want_distance) = near.compute_zoom_to_fit(&finite, 1.5).expect("framed");
    let (got, got_distance) = far.compute_zoom_to_fit(&finite, 1.5).expect("framed");
    assert!(want_distance > 0.0);
    assert!((got - want).norm() < 1e-9, "{got} is not {want}");
    assert!((got_distance - want_distance).abs() < 1e-9);
}

#[test]
fn nothing_to_frame_moves_nothing() {
    let camera = ViewportCamera::default();
    assert_eq!(camera.compute_fit(&FitPoints::default(), 1.5, true), None);
    assert_eq!(framing_bearing(&[Vector3::zeros()]), None);
}
