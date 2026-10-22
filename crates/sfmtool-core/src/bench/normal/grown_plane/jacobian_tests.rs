// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The fits' analytic Jacobians against differences of real renders of a
//! synthetic plane: a view fit, a shift-only fit, a window, a pair, a grid
//! pushed partly off the photograph and lenses with distortion; and the
//! pieces they are built from, the sampler's two gradients, the derivative
//! through a block's normalisation and how a pixel moves with the tilt.

use nalgebra::{DMatrix, Vector3};

use super::combine::sigmas_deg;
use super::render::{Block, GradientRead, Gradients, Grid, Samples, ViewImage, ViewRender};
use super::solve::{
    cell_znccs, forward_difference, live_cells, normal_matrix, schur_tilt, tilt_information,
    Objective, PairFit, Reading, ViewFit,
};
use crate::bench::tests::scene::{texture, Scene, PLANE_Z, WORLD};
use crate::camera::image::ImageU8;
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::patch::cloud::OrientedPatch;

/// Every camera's distance from the point.
const DISTANCE: f64 = 4.0;

/// A 128-px camera of the given model, focal 160 px, centred.
fn camera(model: &str) -> CameraIntrinsics {
    let model = match model {
        "pinhole" => CameraModel::Pinhole {
            focal_length_x: 160.0,
            focal_length_y: 160.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
        },
        "simple_radial" => CameraModel::SimpleRadial {
            focal_length: 160.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
            radial_distortion_k1: -0.2,
        },
        "simple_radial_fisheye" => CameraModel::SimpleRadialFisheye {
            focal_length: 160.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
            radial_distortion_k1: 0.1,
        },
        other => panic!("no camera {other}"),
    };
    CameraIntrinsics {
        model,
        width: 128,
        height: 128,
    }
}

/// A camera centre `DISTANCE` from the point, `from` degrees off the plane's
/// normal toward the azimuth `toward` degrees.
fn centre(from: f64, toward: f64) -> [f64; 3] {
    let (t, p) = (from.to_radians(), toward.to_radians());
    [
        WORLD.x + DISTANCE * t.sin() * p.cos(),
        WORLD.y + DISTANCE * t.sin() * p.sin(),
        WORLD.z - DISTANCE * t.cos(),
    ]
}

/// The textured plane `z = PLANE_Z` seen along the ray from `c` along `d`.
fn plane(c: [f64; 3], d: [f64; 3]) -> f64 {
    let lambda = (PLANE_Z - c[2]) / d[2];
    texture(c[0] + lambda * d[0], c[1] + lambda * d[1], PLANE_Z)
}

/// The reference facing the plane and two views 25 and 35 degrees off it.
fn scene(model: &str) -> Scene {
    let centres = [centre(0.0, 0.0), centre(25.0, 40.0), centre(35.0, 200.0)];
    Scene::looking_at(&centres, WORLD, camera(model), plane)
}

/// The starting patch frame: on the point, its normal 10 degrees off the
/// plane's, 0.4 units a half-side, about 16 px in the reference.
fn patch_frame() -> OrientedPatch {
    let turn = nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.4, 0.0)),
        10f64.to_radians(),
    );
    OrientedPatch::from_center_normal(
        WORLD,
        turn * Vector3::new(0.0, 0.0, -1.0),
        turn * Vector3::new(0.0, 1.0, 0.0),
        [0.4, 0.4],
    )
}

/// The largest relative difference of a column of `a` from the same column
/// of `b`, `‖a_i − b_i‖ / ‖b_i‖`, over the columns.
fn worst_column(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    (0..a.ncols())
        .map(|i| (a.column(i) - b.column(i)).norm() / b.column(i).norm().max(1e-300))
        .fold(0.0, f64::max)
}

/// The central difference of `objective`'s frozen residual at `p`, with step
/// `steps[i]` along parameter `i`: the mean of the forward differences with
/// `+steps` and `−steps`.
fn central<O: Objective>(
    objective: &O,
    frozen: &O::Frozen,
    p: &[f64],
    r0: &[f64],
    steps: &[f64],
) -> DMatrix<f64> {
    let back: Vec<f64> = steps.iter().map(|h| -h).collect();
    (forward_difference(objective, frozen, p, r0, steps)
        + forward_difference(objective, frozen, p, r0, &back))
        * 0.5
}

/// How far a column of the analytic Jacobian may lie from a central
/// difference of renders, relative to the column's length. The difference
/// itself is no closer than about 1%: a sample's position is cast to `f32`
/// for the bilinear read, which near 100 px resolves about 8·10⁻⁶ px, 0.8% of
/// the 10⁻³ px shift step and of the pixel move of a 10⁻⁴ tilt step; a
/// larger step crosses more of the pixel edges where a bilinear read's slope
/// jumps, and differs more (2% at 10⁻² px). What a render-level comparison
/// has to catch, a wrong sign, a missing level scale or a lost term of the
/// normalisation, is off by tens of percent or more; the pieces are checked
/// to 10⁻⁶ on their own, below, where no photograph is read.
const RENDER_TOLERANCE: f64 = 0.03;

/// The analytic Jacobian of `objective` at `p` with the bilinear read's own
/// derivative against central differences of renders, with steps of 10⁻⁴ in
/// the tilt and 10⁻³ px in the shift, and against the forward differences the
/// fits used before, with steps of 10⁻³ in the tilt and 0.05 px in the shift;
/// and the Jacobian with the Sobel gradient against it: the worst column's
/// relative difference in each case.
fn compare<O: Objective>(objective: &O, p: &[f64], tilt_free: bool) -> (f64, f64, f64) {
    let (_, frozen) = objective.freeze(p);
    let (fz, r0) = frozen.expect("the fit reads something");
    assert!(!r0.is_empty());
    let analytic = objective.jacobian(&fz, p, &r0, GradientRead::Bilinear);
    let sobel = objective.jacobian(&fz, p, &r0, GradientRead::Sobel);
    assert_eq!(analytic.shape(), (r0.len(), p.len()));
    let steps = |tilt: f64, shift: f64| -> Vec<f64> {
        let shifts = if tilt_free { p.len() - 2 } else { p.len() };
        let mut s = vec![tilt; p.len() - shifts];
        s.extend(vec![shift; shifts]);
        s
    };
    let small = central(objective, &fz, p, &r0, &steps(1e-4, 1e-3));
    let old = forward_difference(objective, &fz, p, &r0, &steps(1e-3, 0.05));
    (
        worst_column(&analytic, &small),
        worst_column(&analytic, &old),
        worst_column(&sobel, &analytic),
    )
}

/// The analytic Jacobian follows a central difference of real renders
/// within [`RENDER_TOLERANCE`] in every column, for a view fit over the whole
/// grid, a window, a shift-only fit and a pair, through a pinhole, a radial
/// lens and a fisheye; the worst columns differ by 0.1% to 1.7%. The forward
/// differences the fits used before differ from it by 9% to 12%: their 0.05
/// px shift step and 10⁻³ tilt step cross many pixel edges, and a forward
/// difference is first-order. The Jacobian the fits step with, through the
/// Sobel gradient, differs from it by 26% to 31% on this texture, whose
/// detail is a few px across: the Sobel gradient is the slope averaged over
/// about two pixels, and the bilinear read's slope is that of one pixel. A
/// wrong sign or a missing term would differ by 100% or more.
#[test]
fn the_analytic_jacobian_follows_a_difference_of_renders() {
    for model in ["pinhole", "simple_radial", "simple_radial_fisheye"] {
        let scene = scene(model);
        let views = scene.views();
        let patch_frame = patch_frame();
        let side = 32;
        let grid = Grid::new(&patch_frame, &views[0], None, side);
        let images: Vec<ViewImage<'_>> = views
            .iter()
            .map(|v| ViewImage::new(v, &grid.anchored_patch_frame, side))
            .collect();
        let renders: Vec<ViewRender<'_, '_>> =
            images.iter().map(|i| ViewRender::new(&grid, i)).collect();
        let reference = renders[0].render([0.0; 2], [0.0; 2], &grid.all());
        let live = live_cells(&grid.cells, &reference);
        assert!(live.iter().all(|&l| l), "{model}: every cell textured");
        let fit = |reading: Reading, held: Option<[f64; 2]>| ViewFit {
            render: &renders[1],
            reference: &reference,
            live: &live,
            reading,
            held,
            min_coverage: 0.6,
        };
        let tilt = [0.06, -0.04];
        let cases: Vec<(&str, (f64, f64, f64))> = vec![
            (
                "grid",
                compare(
                    &fit(Reading::Grid, None),
                    &[tilt[0], tilt[1], 0.3, -0.2],
                    true,
                ),
            ),
            (
                "window",
                compare(
                    &fit(Reading::Window(grid.window(6)), None),
                    &[tilt[0], tilt[1], 0.3, -0.2],
                    true,
                ),
            ),
            (
                "shift-only",
                compare(&fit(Reading::Grid, Some(tilt)), &[0.3, -0.2], false),
            ),
            (
                "pair",
                compare(
                    &PairFit {
                        first: &renders[1],
                        second: &renders[2],
                        live: &live,
                        shifts: [[0.3, -0.2], [-0.4, 0.1]],
                        min_coverage: 0.6,
                    },
                    &[tilt[0], tilt[1], 0.25, 0.15],
                    true,
                ),
            ),
        ];
        for (name, (small, old, sobel)) in cases {
            assert!(small < RENDER_TOLERANCE, "{model} {name}: {small}");
            assert!(sobel > 0.2 && sobel < 0.4, "{model} {name}: {sobel}");
            assert!(old < 0.2, "{model} {name}: {old}");
        }
    }
}

/// The information about the tilt, which gives `σ`, the robust mean's
/// weights and the selection's stop, is the normal matrix of the Jacobian the
/// fits step with, read through the Sobel gradient, and not that of the
/// bilinear read's own derivative. On this texture, whose detail is a few px
/// across, the Sobel gradient is the smaller, so the information is smaller
/// and the weakest `σ` larger than the bilinear derivative would give.
#[test]
fn the_information_is_read_through_the_sobel_gradient() {
    let scene = scene("pinhole");
    let views = scene.views();
    let patch_frame = patch_frame();
    let side = 32;
    let grid = Grid::new(&patch_frame, &views[0], None, side);
    let images: Vec<ViewImage<'_>> = views
        .iter()
        .map(|v| ViewImage::new(v, &grid.anchored_patch_frame, side))
        .collect();
    let renders: Vec<ViewRender<'_, '_>> =
        images.iter().map(|i| ViewRender::new(&grid, i)).collect();
    let reference = renders[0].render([0.0; 2], [0.0; 2], &grid.all());
    let live = live_cells(&grid.cells, &reference);
    let fit = ViewFit {
        render: &renders[1],
        reference: &reference,
        live: &live,
        reading: Reading::Grid,
        held: None,
        min_coverage: 0.6,
    };
    let p = [0.06, -0.04, 0.3, -0.2];
    let (_, frozen) = fit.freeze(&p);
    let (fz, r0) = frozen.expect("the fit reads something");
    let sobel = fit.jacobian(&fz, &p, &r0, GradientRead::Sobel);
    let bilinear = fit.jacobian(&fz, &p, &r0, GradientRead::Bilinear);
    let (sobel, bilinear) = (sobel.transpose() * &sobel, bilinear.transpose() * &bilinear);
    let n = normal_matrix(&fit, &p).expect("the fit reads something");
    assert!(
        (&n - &sobel).norm() <= 1e-12 * sobel.norm(),
        "{n} vs {sobel}"
    );
    assert!(
        (&n - &bilinear).norm() > 0.1 * bilinear.norm(),
        "{n} vs {bilinear}"
    );
    let tau = 0.01;
    let sigma = sigmas_deg(&tilt_information(&fit, &p), tau)[0];
    let sobel_sigma = sigmas_deg(&schur_tilt(&sobel), tau)[0];
    let bilinear_sigma = sigmas_deg(&schur_tilt(&bilinear), tau)[0];
    assert!(
        (sigma - sobel_sigma).abs() <= 1e-9 * sobel_sigma,
        "{sigma} vs {sobel_sigma}"
    );
    assert!(sigma > 1.1 * bilinear_sigma, "{sigma} vs {bilinear_sigma}");
}

/// A view shifted until about a quarter of the grid lies off the photograph:
/// those samples are not in the residual, the Jacobian has a row for each
/// residual and none for them, and it follows the difference of renders as
/// on the photograph.
#[test]
fn a_grid_partly_off_the_photograph_differences_like_its_residual() {
    let scene = scene("pinhole");
    let views = scene.views();
    let patch_frame = patch_frame();
    let side = 32;
    let grid = Grid::new(&patch_frame, &views[0], None, side);
    let images: Vec<ViewImage<'_>> = views
        .iter()
        .map(|v| ViewImage::new(v, &grid.anchored_patch_frame, side))
        .collect();
    let renders: Vec<ViewRender<'_, '_>> =
        images.iter().map(|i| ViewRender::new(&grid, i)).collect();
    let reference = renders[0].render([0.0; 2], [0.0; 2], &grid.all());
    let live = live_cells(&grid.cells, &reference);
    let anchor = grid.anchor_pixel(&images[1]).expect("projects");
    // Shift the anchor to 8 px from the right edge: the grid's right part
    // lands past it.
    let shift = [128.0 - 8.0 - anchor[0], 0.0];
    let p = [0.03, 0.02, shift[0], shift[1]];
    let samples = renders[1].render([p[0], p[1]], shift, &grid.all());
    let off = samples.valid.iter().filter(|&&v| !v).count();
    assert!(
        off > side * side / 8 && off < side * side / 2,
        "{off} of {} off the photograph",
        side * side
    );
    let fit = ViewFit {
        render: &renders[1],
        reference: &reference,
        live: &live,
        reading: Reading::Grid,
        held: None,
        min_coverage: 0.6,
    };
    // Some cells are under the coverage and read −1; the others are scored.
    let cells = cell_znccs(&renders[1], &reference, &live, &samples, 0.6);
    assert!(cells.iter().flatten().any(|&z| z == -1.0), "{cells:?}");
    assert!(cells.iter().flatten().any(|&z| z > -1.0), "{cells:?}");
    let (_, frozen) = fit.freeze(&p);
    let (_, r0) = frozen.as_ref().expect("cells on the photograph");
    let on: usize = grid
        .cells
        .iter()
        .zip(&cells)
        .filter(|(_, z)| z.is_some_and(|z| z > -1.0))
        .map(|(rows, _)| rows.iter().filter(|&&k| samples.valid[k]).count())
        .sum();
    assert!(r0.len() <= on, "{} residuals, {on} samples on", r0.len());
    let (small, ..) = compare(&fit, &p, true);
    assert!(small < RENDER_TOLERANCE, "{small}");
}

/// The sampler's gradient is the derivative of the value it samples: inside
/// the photograph, along an axis clamped at the edge (where the value does
/// not move and the gradient is zero) and past the edge on both axes; a
/// sample with no pixel has value and gradient zero. A level-1 read reports
/// its gradient per full-resolution px.
#[test]
fn the_sampler_gradient_is_the_derivative_of_its_value_at_the_edge() {
    let (w, h) = (16u32, 12u32);
    let data: Vec<u8> = (0..w * h)
        .flat_map(|k| {
            let (x, y) = ((k % w) as f64, (k / w) as f64);
            [
                (120.0 + 60.0 * (0.7 * x).sin() + 40.0 * (0.5 * y).cos()) as u8,
                (100.0 + 3.0 * x + 7.0 * y) as u8,
            ]
        })
        .collect();
    let image = ImageU8::new(w, h, 2, data);
    let pyramid = crate::camera::image::ImageU8Pyramid::build(&image, 2);
    let camera = camera("pinhole");
    let pose = crate::geometry::RigidTransform::identity();
    let view = crate::patch::normal_refine::ProjectedImage {
        camera: &camera,
        cam_from_world: &pose,
        pyramid: &pyramid,
    };
    for level in [0usize, 1] {
        let read = ViewImage::at_level(&view, level);
        let scale = (1u64 << level) as f64;
        let height = f64::from(h >> level);
        // Inside; left of the first pixel centre in x; below the last in y;
        // past both edges; no pixel.
        let pixels: Vec<Option<[f64; 2]>> = vec![
            Some([5.3 * scale, 4.6 * scale]),
            Some([0.2 * scale, 3.3 * scale]),
            Some([7.4 * scale, (height - 0.2) * scale]),
            Some([-3.0 * scale, 40.0 * scale]),
            None,
        ];
        let rows: Vec<usize> = (0..pixels.len()).collect();
        let n = pixels.len();
        let mut samples = Samples::new(n, read.channels);
        let mut gradients = Gradients::new(n, read.channels);
        read.sample(&pixels, &rows, [0.0; 2], &mut samples);
        read.gradients(
            &pixels,
            &rows,
            [0.0; 2],
            GradientRead::Bilinear,
            &mut gradients,
        );
        let step = 1e-2 * scale;
        for axis in 0..2 {
            let mut shift = [0.0; 2];
            shift[axis] = step;
            let mut ahead = Samples::new(n, read.channels);
            read.sample(&pixels, &rows, shift, &mut ahead);
            shift[axis] = -step;
            let mut behind = Samples::new(n, read.channels);
            read.sample(&pixels, &rows, shift, &mut behind);
            for k in 0..n {
                for c in 0..read.channels {
                    let i = k * read.channels + c;
                    let difference = f64::from(ahead.values[i] - behind.values[i]) / (2.0 * step);
                    let g = gradients.at(k, c)[axis];
                    assert!(
                        (g - difference).abs() < 1e-2,
                        "level {level}, sample {k}, channel {c}, axis {axis}: {g} vs {difference}"
                    );
                }
            }
        }
        // Clamped axes read zero gradient exactly.
        for c in 0..read.channels {
            assert_eq!(gradients.at(1, c)[0], 0.0);
            assert_eq!(gradients.at(2, c)[1], 0.0);
            assert_eq!(gradients.at(3, c), [0.0; 2]);
            assert_eq!(gradients.at(4, c), [0.0; 2]);
        }
        assert!(samples.valid[0] && !samples.valid[1] && !samples.valid[4]);
        assert!(gradients.at(0, 0)[0] != 0.0);
    }
}

/// The Sobel read is the bilinear read of the level's Sobel gradient images:
/// built here pixel by pixel from the edge-clamped neighbours and read with
/// the bilinear weights, they agree to 10⁻⁴ grey levels per px, inside the
/// photograph and with taps clamped at its edges. On a linear ramp it is the
/// ramp's slope wherever its 4×4 pixels lie on the photograph, per
/// full-resolution px at level 0; and it is continuous where the bilinear
/// read's slope jumps, at a pixel centre.
#[test]
fn the_sobel_read_is_a_bilinear_read_of_the_sobel_images() {
    let (w, h) = (16u32, 12u32);
    let data: Vec<u8> = (0..w * h)
        .flat_map(|k| {
            let (x, y) = ((k % w) as f64, (k / w) as f64);
            [
                (120.0 + 60.0 * (0.7 * x).sin() + 40.0 * (0.5 * y).cos()) as u8,
                (100.0 + 3.0 * x + 7.0 * y) as u8,
            ]
        })
        .collect();
    let image = ImageU8::new(w, h, 2, data);
    let pixel = |x: i32, y: i32, c: u32| {
        f32::from(image.get_pixel(
            x.clamp(0, w as i32 - 1) as u32,
            y.clamp(0, h as i32 - 1) as u32,
            c,
        ))
    };
    let sobel_image = |x: i32, y: i32, c: u32| {
        let s = [0.25f32, 0.5, 0.25];
        let (mut dx, mut dy) = (0.0, 0.0);
        for (d, weight) in s.iter().enumerate() {
            let d = d as i32 - 1;
            dx += weight * (pixel(x + 1, y + d, c) - pixel(x - 1, y + d, c));
            dy += weight * (pixel(x + d, y + 1, c) - pixel(x + d, y - 1, c));
        }
        (0.5 * dx, 0.5 * dy)
    };
    for &(x, y) in &[
        (5.3f32, 4.6f32),
        (0.2, 3.3),
        (15.9, 11.7),
        (8.5, 6.5),
        (1.1, 0.9),
    ] {
        let (gx, gy) = (x - 0.5, y - 0.5);
        let (x0, y0) = (gx.floor() as i32, gy.floor() as i32);
        let (fx, fy) = (gx - x0 as f32, gy - y0 as f32);
        for c in 0..2 {
            let mut expected = (0.0f32, 0.0f32);
            for (dx, dy, weight) in [
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let g = sobel_image(x0 + dx, y0 + dy, c);
                expected.0 += weight * g.0;
                expected.1 += weight * g.1;
            }
            let got = super::render::sobel_gradient(&image, x, y, c);
            assert!(
                (got.0 - expected.0).abs() < 1e-4 && (got.1 - expected.1).abs() < 1e-4,
                "({x}, {y}) channel {c}: {got:?} vs {expected:?}"
            );
        }
    }
    // The ramp, channel 1: 3 per px in x and 7 in y.
    let pyramid = crate::camera::image::ImageU8Pyramid::build(&image, 1);
    let camera = camera("pinhole");
    let pose = crate::geometry::RigidTransform::identity();
    let view = crate::patch::normal_refine::ProjectedImage {
        camera: &camera,
        cam_from_world: &pose,
        pyramid: &pyramid,
    };
    let read = ViewImage::at_level(&view, 0);
    let pixels: Vec<Option<[f64; 2]>> =
        vec![Some([5.3, 4.6]), Some([8.5, 6.5]), Some([8.5001, 6.4999])];
    let rows: Vec<usize> = (0..pixels.len()).collect();
    let mut sobel = Gradients::new(pixels.len(), read.channels);
    read.gradients(&pixels, &rows, [0.0; 2], GradientRead::Sobel, &mut sobel);
    for k in 0..pixels.len() {
        let g = sobel.at(k, 1);
        assert!(
            (g[0] - 3.0).abs() < 1e-4 && (g[1] - 7.0).abs() < 1e-4,
            "{k}: {g:?}"
        );
    }
    // Across the pixel centre at (8.5, 6.5) the textured channel's Sobel read
    // moves by little; the bilinear read's slope jumps.
    let mut bilinear = Gradients::new(pixels.len(), read.channels);
    read.gradients(
        &pixels,
        &rows,
        [0.0; 2],
        GradientRead::Bilinear,
        &mut bilinear,
    );
    let (a, b) = (sobel.at(1, 0), sobel.at(2, 0));
    assert!(
        (a[0] - b[0]).abs() < 0.01 && (a[1] - b[1]).abs() < 0.01,
        "{a:?} {b:?}"
    );
    let (a, b) = (bilinear.at(1, 0), bilinear.at(2, 0));
    assert!(
        (a[0] - b[0]).abs() > 1.0 || (a[1] - b[1]).abs() > 1.0,
        "{a:?} {b:?}"
    );
}

/// A block's derivative is the chain rule through its normalisation: for
/// samples moved along a direction, it follows a central difference of the
/// block's residual, for a view against the reference (sign −1) and for a
/// pair (both sides moving), over two channels with different weights and a
/// third the view is flat in, whose normalised samples are zeros and whose
/// rows add nothing.
#[test]
fn a_blocks_derivative_goes_through_its_normalisation() {
    let n = 40;
    let channels = 3;
    let wave = |k: usize, c: usize, phase: f64| {
        let t = k as f64 * 0.37 + c as f64 * 1.3 + phase;
        // On a 1/256 lattice, so that a move by a multiple of 1/64 is exact.
        ((120.0 + 50.0 * t.sin() + 20.0 * (2.3 * t).cos()) * 256.0).round() as f32 / 256.0
    };
    let reference = Samples {
        channels,
        values: (0..n * channels)
            .map(|i| wave(i / channels, i % channels, 0.0))
            .collect(),
        valid: vec![true; n],
    };
    // The view: a different signal in channels 0 and 1, flat in 2.
    let view_at = |t: f64| Samples {
        channels,
        values: (0..n * channels)
            .map(|i| {
                let (k, c) = (i / channels, i % channels);
                if c == 2 {
                    90.0
                } else {
                    wave(k, c, 0.4) + (t * direction(k, c)) as f32
                }
            })
            .collect(),
        valid: vec![true; n],
    };
    fn direction(k: usize, c: usize) -> f64 {
        ((k * 7 + c * 3) % 11) as f64 - 5.0
    }
    let rows: Vec<usize> = (0..n).collect();
    let view = view_at(0.0);
    let block = Block::freeze(&reference, &view, &rows, 0.3, 0.6).expect("textured");
    let derivative = |k: usize, c: usize, out: &mut [f64]| {
        out[0] = if c == 2 { 0.0 } else { direction(k, c) };
    };
    let h = 1.0 / 64.0;

    let mut j = DMatrix::zeros(3 * n, 1);
    assert_eq!(
        block.add_derivative(&view, -1.0, derivative, &mut j, 0),
        3 * n
    );
    let residual = |t: f64| {
        let mut r = Vec::new();
        block.residuals(&view_at(t), &mut r);
        DMatrix::from_column_slice(r.len(), 1, &r)
    };
    let difference = (residual(h) - residual(-h)) / (2.0 * h);
    assert!(
        (&j - &difference).norm() < 1e-6 * difference.norm(),
        "{}",
        (&j - &difference).norm() / difference.norm()
    );
    assert!(
        j.rows(2 * n, n).iter().all(|&v| v == 0.0),
        "the flat channel"
    );

    // A pair: the reference's side moves too, along its own direction.
    let first_at = |t: f64| Samples {
        values: reference
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| v + (t * direction(i / channels + 3, i % channels)) as f32)
            .collect(),
        ..reference.clone()
    };
    let mut jp = DMatrix::zeros(3 * n, 1);
    block.add_derivative(
        &reference,
        1.0,
        |k, c, out| out[0] = direction(k + 3, c),
        &mut jp,
        0,
    );
    block.add_derivative(&view, -1.0, derivative, &mut jp, 0);
    let pair = |t: f64| {
        let mut r = Vec::new();
        block.pair_residuals(&first_at(t), &view_at(t), &mut r);
        DMatrix::from_column_slice(r.len(), 1, &r)
    };
    let difference = (pair(h) - pair(-h)) / (2.0 * h);
    assert!(
        (&jp - &difference).norm() < 1e-6 * difference.norm(),
        "{}",
        (&jp - &difference).norm() / difference.norm()
    );
}

/// How a sample's pixel moves with the tilt follows a central difference of
/// the projection itself, which reads no photograph and is `f64` throughout,
/// to 10⁻⁶ of its size, through a pinhole, a radial lens and a fisheye.
#[test]
fn the_pixels_tilt_derivative_follows_the_projection() {
    for model in ["pinhole", "simple_radial", "simple_radial_fisheye"] {
        let scene = scene(model);
        let views = scene.views();
        let patch_frame = patch_frame();
        let side = 32;
        let grid = Grid::new(&patch_frame, &views[0], None, side);
        let rows = grid.all();
        for view in &views[1..] {
            let image = ViewImage::new(view, &grid.anchored_patch_frame, side);
            let tilt = [0.06, -0.04];
            let analytic = grid.pixel_tilt_derivatives(&image, tilt, &rows);
            let h = 1e-6;
            let (mut worst, mut largest) = (0.0f64, 0.0f64);
            for i in 0..2 {
                let (mut ahead, mut behind) = (tilt, tilt);
                ahead[i] += h;
                behind[i] -= h;
                let a = grid.project(&image, ahead, &rows);
                let b = grid.project(&image, behind, &rows);
                for &k in &rows {
                    let (pa, pb) = (a[k].expect("projects"), b[k].expect("projects"));
                    for axis in 0..2 {
                        let difference = (pa[axis] - pb[axis]) / (2.0 * h);
                        worst = worst.max((analytic[k][axis][i] - difference).abs());
                        largest = largest.max(difference.abs());
                    }
                }
            }
            assert!(largest > 1.0, "{model}: the pixels move with the tilt");
            assert!(worst < 1e-6 * largest, "{model}: {worst} of {largest}");
        }
    }
}
