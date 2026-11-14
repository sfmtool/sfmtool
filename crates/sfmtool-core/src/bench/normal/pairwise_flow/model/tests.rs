// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pair model checked against the geometry it states: the round's own
//! plane predicts no flow, a predicted flow lands where camera `j` sees the
//! candidate plane, and the derivative matches finite differences.

use nalgebra::Vector3;

use super::{tangent_basis, CommonPlane, PairModel};

fn plane() -> CommonPlane {
    let normal = Vector3::new(0.1, -0.2, 1.0).normalize();
    let (x_axis, y_axis) = tangent_basis(normal, Vector3::x());
    CommonPlane {
        point: Vector3::new(0.3, -0.1, 0.2),
        normal,
        x_axis,
        y_axis,
        spacing: 0.01,
    }
}

fn model(level_scale: f64) -> PairModel {
    PairModel {
        plane: plane(),
        from: Vector3::new(-1.5, 0.4, 4.0),
        to: Vector3::new(1.8, -0.6, 3.5),
        level_scale,
    }
}

fn turned(n: Vector3<f64>, deg: f64, toward: Vector3<f64>) -> Vector3<f64> {
    let (a, _) = tangent_basis(n, toward);
    (n * deg.to_radians().cos() + a * deg.to_radians().sin()).normalize()
}

#[test]
fn the_rounds_own_plane_predicts_no_flow() {
    let m = model(1.0);
    for q in [[0.0, 0.0], [20.0, -13.0], [-40.0, 35.0]] {
        let f = m.flow(q, m.plane.normal).unwrap();
        assert!(f[0].abs() < 1e-9 && f[1].abs() < 1e-9, "{f:?} at {q:?}");
    }
}

/// The predicted flow carries a sample of crop `i` to where camera `j` sees
/// the same point of the candidate plane: that point, camera `j`'s centre
/// and the round's-plane point at the flow's end are on one line, and the
/// point is on camera `i`'s ray through the sample. The flow is in samples
/// of the level, so at level 1 it is half the grid's.
#[test]
fn a_predicted_flow_ends_where_the_second_camera_sees_the_candidate_plane() {
    let m = model(2.0);
    let p = m.plane;
    let candidate = turned(p.normal, 25.0, Vector3::new(1.0, 0.3, 0.0));
    let mut largest = 0.0f64;
    for q in [[0.0, 0.0], [30.0, -12.0], [-25.0, 40.0], [45.0, 45.0]] {
        let f = m.flow(q, candidate).unwrap();
        largest = largest.max(f[0].hypot(f[1]));
        // Where camera `i`'s ray through the sample meets the candidate.
        let start = p.world(q);
        let d = start - m.from;
        let y = m.from + d * (candidate.dot(&(p.point - m.from)) / candidate.dot(&d));
        let end = p.world([q[0] + 2.0 * f[0], q[1] + 2.0 * f[1]]);
        let off_line = (end - m.to)
            .normalize()
            .cross(&(y - m.to).normalize())
            .norm();
        assert!(off_line < 1e-12, "{off_line} at {q:?}");
        assert!((y - p.point).dot(&candidate).abs() < 1e-12);
    }
    // The candidate is far enough off for the test to mean something.
    assert!(largest > 1.0, "{largest}");
}

#[test]
fn the_derivative_matches_finite_differences() {
    let m = model(2.0);
    let n = turned(m.plane.normal, 20.0, Vector3::new(0.2, 1.0, 0.0));
    let (a, b) = tangent_basis(n, m.plane.x_axis);
    let h = 1e-6;
    for q in [[0.0, 0.0], [35.0, -20.0], [-30.0, 25.0]] {
        let (_, j) = m.flow_and_jacobian(q, n, a, b).unwrap();
        for (column, axis) in [a, b].into_iter().enumerate() {
            let plus = m.flow(q, (n + axis * h).normalize()).unwrap();
            let minus = m.flow(q, (n - axis * h).normalize()).unwrap();
            for row in 0..2 {
                let numeric = (plus[row] - minus[row]) / (2.0 * h);
                let analytic = j[row][column];
                assert!(
                    (numeric - analytic).abs() < 1e-5 * (1.0 + analytic.abs()),
                    "{numeric} vs {analytic} at {q:?}, entry {row},{column}"
                );
            }
        }
    }
}

#[test]
fn a_ray_parallel_to_the_candidate_has_no_prediction() {
    let m = model(1.0);
    let q = [10.0, 5.0];
    let d = m.plane.world(q) - m.from;
    // A candidate containing the ray's direction.
    let parallel = d.cross(&m.plane.x_axis).normalize();
    assert!(m.flow(q, parallel).is_none());
}
