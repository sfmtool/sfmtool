// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The working grid laid again at a plane, and the charts that carry a laid
//! grid's tilt and information into the starting frame's: the laid grid's
//! samples on the turned square, the template rendered there, a laid grid's
//! pixel derivatives against differences of its projection, and the chart
//! Jacobian against a central difference of the chart.

use nalgebra::{Matrix2, Rotation3, Unit, Vector3};

use super::render::{Grid, ViewImage, ViewRender};
use super::solve::carried_information;
use super::Laid;
use crate::bench::tests::scene::{texture, Scene, PLANE_Z, WORLD};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::ProjectedImage;

/// A 128-px pinhole, focal 160 px, centred.
fn camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 160.0,
            focal_length_y: 160.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
        },
        width: 128,
        height: 128,
        rolling_shutter: None,
    }
}

/// A camera centre 4 units from the point, `from` degrees off the plane's
/// normal toward the azimuth `toward` degrees.
fn centre(from: f64, toward: f64) -> [f64; 3] {
    let (t, p) = (from.to_radians(), toward.to_radians());
    [
        WORLD.x + 4.0 * t.sin() * p.cos(),
        WORLD.y + 4.0 * t.sin() * p.sin(),
        WORLD.z - 4.0 * t.cos(),
    ]
}

/// A reference facing the textured plane `z = PLANE_Z` and a view 30
/// degrees off it.
fn scene() -> Scene {
    Scene::looking_at(
        &[centre(0.0, 0.0), centre(30.0, 70.0)],
        WORLD,
        camera(),
        |c, d| {
            let lambda = (PLANE_Z - c[2]) / d[2];
            texture(c[0] + lambda * d[0], c[1] + lambda * d[1], PLANE_Z)
        },
    )
}

/// The starting patch frame: on the point, facing the reference, a half-side
/// of 0.4.
fn patch_frame() -> OrientedPatch {
    OrientedPatch::from_center_normal(
        WORLD,
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.4, 0.4],
    )
}

/// The starting normal turned `deg` degrees about the axis `axis`.
fn turned(grid: &Grid, axis: Vector3<f64>, deg: f64) -> Vector3<f64> {
    Rotation3::from_axis_angle(&Unit::new_normalize(axis), deg.to_radians()) * grid.start.normal
}

/// The starting grid of `side` samples over [`patch_frame`] seen from the
/// scene's reference.
fn starting_grid(views: &[ProjectedImage<'_>], side: usize) -> Grid {
    Grid::new(&patch_frame(), &views[0], None, side)
}

/// The grid laid at the starting normal is the starting grid: the same
/// sample points to 10⁻¹², the same frame, the same cells.
#[test]
fn a_grid_laid_at_the_starting_normal_is_the_starting_grid() {
    let scene = scene();
    let views = scene.views();
    let grid = starting_grid(&views, 24);
    let again = grid.laid_at(grid.start.normal);
    for k in 0..24 * 24 {
        let d = (again.sample_point(k) - grid.sample_point(k)).norm();
        assert!(d < 1e-12, "sample {k}: {d}");
    }
    assert_eq!(again.frame, grid.frame);
    assert_eq!(again.cells, grid.cells);
    assert_eq!(again.start, grid.start);
}

/// On a grid laid 40° from the starting normal every sample lies on the
/// turned plane through the anchor, neighbouring samples are one spacing
/// apart along the turned square's axes, the centre sample's ray passes
/// within half a sample of the anchor, and the anchor projects into a view
/// at the same pixel as on the starting grid. Its frame is the turned normal
/// and the starting axes turned by the least rotation.
#[test]
fn a_laid_grid_covers_the_turned_square() {
    let scene = scene();
    let views = scene.views();
    let side = 24;
    let grid = starting_grid(&views, side);
    let n = turned(&grid, Vector3::new(1.0, 0.3, 0.0), 40.0);
    let laid = grid.laid_at(n);
    assert_eq!(laid.frame.normal, n);
    assert!(laid.frame.u.dot(&n).abs() < 1e-12 && laid.frame.v.dot(&n).abs() < 1e-12);
    assert!((laid.frame.u.cross(&laid.frame.v) - n).norm() < 1e-12);
    let spacing = 2.0 * 0.4 / side as f64;
    for row in 0..side {
        for col in 0..side {
            let k = row * side + col;
            let p = laid.sample_point(k);
            assert!(
                (p - laid.anchor).dot(&n).abs() < 1e-12,
                "sample {k} off the plane"
            );
            if col + 1 < side {
                let step = laid.sample_point(k + 1) - p;
                assert!(
                    (step - laid.frame.u * spacing).norm() < 1e-12,
                    "{k}: {step}"
                );
            }
            if row + 1 < side {
                let step = laid.sample_point(k + side) - p;
                assert!(
                    (step + laid.frame.v * spacing).norm() < 1e-12,
                    "{k}: {step}"
                );
            }
        }
    }
    // The centre sample (W/2, W/2), whose ray meets the plane at its point on
    // the square, is half a sample right of and below the anchor on an even
    // grid: within half a sample of it along each of the square's axes.
    let centre = laid.sample_point((side / 2) * side + side / 2);
    let off = centre - laid.anchor;
    assert!(
        (off.dot(&laid.frame.u) - 0.5 * spacing).abs() < 1e-12,
        "{off}"
    );
    assert!(
        (off.dot(&laid.frame.v) + 0.5 * spacing).abs() < 1e-12,
        "{off}"
    );
    let view = ViewImage::new(
        &views[1],
        &grid.anchored_patch_frame,
        side,
        Vector3::zeros(),
    );
    let (a, b) = (
        grid.anchor_pixel(&view).expect("projects"),
        laid.anchor_pixel(&view).expect("projects"),
    );
    assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
}

/// The template on a laid grid is the reference photograph sampled where
/// each of the turned square's samples projects into it, and a view read
/// on the laid grid at no tilt sees the same points: its pixels are those
/// of the turned square's samples.
#[test]
fn the_template_is_rendered_on_the_turned_square() {
    let scene = scene();
    let views = scene.views();
    let side = 24;
    let grid = starting_grid(&views, side);
    let reference = ViewImage::new(
        &views[0],
        &grid.anchored_patch_frame,
        side,
        Vector3::zeros(),
    );
    let n = turned(&grid, Vector3::new(-0.2, 1.0, 0.0), 30.0);
    let laid = Laid::new(grid.laid_at(n), &reference);
    let start = Laid::new(starting_grid(&views, side), &reference);
    let rows = laid.grid.all();
    let pixels: Vec<Option<[f64; 2]>> = rows
        .iter()
        .map(|&k| {
            views[0].camera.project_homogeneous(
                views[0].cam_from_world,
                laid.grid.sample_point(k),
                1.0,
            )
        })
        .collect();
    let mut direct = super::Samples::new(side * side, reference.channels);
    reference.sample(&pixels, &rows, [0.0; 2], &mut direct);
    let mut differs = 0;
    for k in 0..side * side {
        assert_eq!(laid.template.valid[k], direct.valid[k], "{k}");
        for c in 0..reference.channels {
            let (a, b) = (
                laid.template.values[k * reference.channels + c],
                direct.values[k * reference.channels + c],
            );
            assert!((a - b).abs() < 1e-3, "{k}/{c}: {a} vs {b}");
            if (a - start.template.values[k * reference.channels + c]).abs() > 1.0 {
                differs += 1;
            }
        }
    }
    // The turned square is not the starting one: its template differs.
    assert!(differs > side * side / 4, "{differs}");
    // A view read on the laid grid at no tilt reads the turned square's
    // samples, moved by its offset (zero here).
    let view = ViewImage::new(
        &views[1],
        &grid.anchored_patch_frame,
        side,
        Vector3::zeros(),
    );
    let projected = laid.grid.project(&view, [0.0; 2], &rows);
    for &k in &rows {
        let want = views[1].camera.project_homogeneous(
            views[1].cam_from_world,
            laid.grid.sample_point(k),
            1.0,
        );
        match (projected[k], want) {
            (Some(a), Some(b)) => assert!((a[0] - b[0]).hypot(a[1] - b[1]) < 1e-6, "{k}"),
            (a, b) => assert_eq!(a.is_some(), b.is_some(), "{k}"),
        }
    }
    // Rendering a template counts as a render of the reference.
    let before = reference.renders();
    let _ = ViewRender::new(&laid.grid, &reference).render([0.0; 2], [0.0; 2], &rows);
    assert_eq!(reference.renders(), before + 1);
}

/// A laid grid's pixel derivatives with the tilt, in its own frame, follow a
/// central difference of its projection.
#[test]
fn a_laid_grids_pixel_derivatives_follow_its_projection() {
    let scene = scene();
    let views = scene.views();
    let side = 16;
    let grid = starting_grid(&views, side);
    let laid = grid.laid_at(turned(&grid, Vector3::new(1.0, -0.5, 0.0), 35.0));
    let view = ViewImage::new(
        &views[1],
        &grid.anchored_patch_frame,
        side,
        Vector3::zeros(),
    );
    let rows = laid.all();
    let tilt = [0.05, -0.03];
    let analytic = laid.pixel_tilt_derivatives(&view, tilt, &rows);
    let h = 1e-6;
    for axis in 0..2 {
        let mut plus = tilt;
        let mut minus = tilt;
        plus[axis] += h;
        minus[axis] -= h;
        let (a, b) = (
            laid.project(&view, plus, &rows),
            laid.project(&view, minus, &rows),
        );
        for &k in &rows {
            let (Some(a), Some(b)) = (a[k], b[k]) else {
                continue;
            };
            for xy in 0..2 {
                let difference = (a[xy] - b[xy]) / (2.0 * h);
                let got = analytic[k][xy][axis];
                assert!(
                    (got - difference).abs() < 1e-5 * difference.abs().max(1.0),
                    "{k} {xy} {axis}: {got} vs {difference}"
                );
            }
        }
    }
}

/// The chart Jacobian `J` follows a central difference of the starting
/// chart at a laid grid's plane to 10⁻⁶ and is the identity on the starting
/// grid; information carried by `J` and back by `J⁻¹` is itself; and on the
/// starting frame the chart's step from one plane to another is the
/// difference of their tilts, so a shift moved by it is the shift moved by
/// the difference of tilts.
#[test]
fn the_charts_carry_tilts_and_information_into_the_starting_frame() {
    let scene = scene();
    let views = scene.views();
    let grid = starting_grid(&views, 16);
    let j0 = grid.start_chart_jacobian();
    assert!((j0 - Matrix2::identity()).norm() < 1e-12, "{j0}");
    for (axis, deg) in [
        (Vector3::new(1.0, 0.3, 0.0), 40.0),
        (Vector3::new(-0.4, 1.0, 0.0), 70.0),
        (Vector3::new(0.2, 0.1, 0.0), 5.0),
    ] {
        let laid = grid.laid_at(turned(&grid, axis, deg));
        let j = laid.start_chart_jacobian();
        let chart = |alpha: f64, beta: f64| {
            let m = laid.frame.normal + laid.frame.u * alpha + laid.frame.v * beta;
            laid.start.tilt_of(m).expect("under 90 degrees")
        };
        let h = 1e-6;
        for (column, (da, db)) in [(h, 0.0), (0.0, h)].into_iter().enumerate() {
            let (p, q) = (chart(da, db), chart(-da, -db));
            for row in 0..2 {
                let difference = (p[row] - q[row]) / (2.0 * h);
                assert!(
                    (j[(row, column)] - difference).abs() < 1e-6 * difference.abs().max(1.0),
                    "{deg}: J[{row},{column}] {} vs {difference}",
                    j[(row, column)]
                );
            }
        }
        let information = Matrix2::new(3.0, 0.5, 0.5, 1.2);
        let there = carried_information(&information, &j);
        let back = carried_information(&there, &j.try_inverse().expect("invertible"));
        assert!((back - information).norm() < 1e-9, "{deg}: {back}");
    }
    // On the starting frame the chart's step between two planes is the
    // difference of their tilts.
    let (from, to) = ([0.1, -0.2], [-0.3, 0.25]);
    let step = grid
        .start
        .tilt_step(grid.plane_normal(from), grid.plane_normal(to))
        .expect("both face the start");
    assert!((step[0] - (to[0] - from[0])).abs() < 1e-12, "{step:?}");
    assert!((step[1] - (to[1] - from[1])).abs() < 1e-12, "{step:?}");
    // A plane at or past 90 degrees has no tilt.
    assert!(grid.start.tilt_of(-grid.start.normal).is_none());
    assert!(grid.start.tilt_of(grid.start.u).is_none());
}

/// A laid grid's tilt that turns its plane to 90° or more from the starting
/// normal does not face the start; every tilt of the starting grid does.
#[test]
fn only_a_laid_grid_reaches_the_charts_edge() {
    let scene = scene();
    let views = scene.views();
    let grid = starting_grid(&views, 16);
    assert!(grid.faces_start([1e6, -1e6]));
    let laid = grid.laid_at(turned(&grid, Vector3::x(), 80.0));
    assert!(laid.faces_start([0.0, 0.0]));
    // The laid frame's tilt toward the starting normal, unit length: a tilt
    // of 0.5 away from it (27°) takes the plane past 90°, toward it back.
    let n0 = grid.start.normal;
    let (a, b) = (laid.frame.u.dot(&n0), laid.frame.v.dot(&n0));
    let toward = [a / a.hypot(b), b / a.hypot(b)];
    assert!(!laid.faces_start([-0.5 * toward[0], -0.5 * toward[1]]));
    assert!(laid.faces_start([0.5 * toward[0], 0.5 * toward[1]]));
}
