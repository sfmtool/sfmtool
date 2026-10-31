// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use approx::assert_relative_eq;
use nalgebra::{Point3, Vector3};

use super::consensus::{axpy_f32, axpy_f32_scalar, sum_sq_diff, sum_sq_diff_scalar};
use super::support::{grid_bounds, square_cells, view_render_patch, windowed_zncc_at, zncc_grid};
use super::znorm::{weighted_moments, weighted_moments_scalar, znorm_write, znorm_write_scalar};
use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;

/// Helper: build a simple pinhole camera.
fn pinhole(width: u32, height: u32, focal: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: focal,
            focal_length_y: focal,
            principal_point_x: width as f64 / 2.0,
            principal_point_y: height as f64 / 2.0,
        },
        width,
        height,
        rolling_shutter: None,
    }
}

const PLANE_Z: f64 = 4.0;
const IMG_W: u32 = 320;
const IMG_H: u32 = 240;
const FOCAL: f64 = 260.0;

/// Procedural texture on the world plane `z = PLANE_Z`, smooth and
/// multi-frequency so the photometric objective has a well-defined optimum.
fn texture(x: f64, y: f64) -> f64 {
    127.5 + 55.0 * (x * 17.0).sin() + 45.0 * (y * 23.0).cos() + 25.0 * ((x + y) * 31.0).sin()
}

/// Synthesize the image a pinhole camera at `center` (looking down world +z)
/// sees of the textured plane `z = PLANE_Z`.
fn render_plane_view(center: [f64; 3], tex: fn(f64, f64) -> f64) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            // Ray from the camera center through the pixel, intersected with
            // the plane z = PLANE_Z.
            let lambda = (PLANE_Z - center[2]) / 1.0;
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            data.push(tex(x, y).clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8::new(IMG_W, IMG_H, 1, data)
}

struct Scene {
    cams: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyrs: Vec<ImageU8Pyramid>,
}

impl Scene {
    fn new(centers: &[[f64; 3]]) -> Self {
        Self::with_textures(
            centers,
            &vec![texture as fn(f64, f64) -> f64; centers.len()],
        )
    }

    fn with_textures(centers: &[[f64; 3]], texs: &[fn(f64, f64) -> f64]) -> Self {
        let cams = centers
            .iter()
            .map(|_| pinhole(IMG_W, IMG_H, FOCAL))
            .collect();
        let poses = centers
            .iter()
            .map(|c| {
                // Rotated 180° about X (canonical −Z-forward camera looking down world
                // +z); cam_from_world translation = R · (-center).
                RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-c[0], c[1], c[2]])
            })
            .collect();
        let pyrs = centers
            .iter()
            .zip(texs)
            .map(|(c, tex)| ImageU8Pyramid::build(&render_plane_view(*c, *tex), 5))
            .collect();
        Self { cams, poses, pyrs }
    }

    fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.cams
            .iter()
            .zip(&self.poses)
            .zip(&self.pyrs)
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect()
    }
}

/// True surface normal of the synthetic plane (toward the cameras).
fn true_normal() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -1.0)
}

fn plane_patch(init_n: Vector3<f64>) -> OrientedPatch {
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        init_n,
        Vector3::new(0.0, 1.0, 0.0),
        [0.5, 0.5],
    )
}

// ---------------------------------------------------------------------------
// Robust consensus weights and their kernels
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random zero-mean unit-norm vector.
fn synthetic_normalized(seed: u64, len: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..len)
        .map(|i| ((seed as f64 + 1.3) * (i as f64 * 0.731 + 0.17)).sin())
        .collect();
    let mean = v.iter().sum::<f64>() / len as f64;
    for x in &mut v {
        *x -= mean;
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Flatten a nested `[view][channel][pixel]` test stack into the production
/// layout `data[(v*channels + c)*n + k]` as f32, returning the dims.
fn flatten_stack(xs: &[Vec<Vec<f64>>]) -> (Vec<f32>, usize, usize, usize) {
    let views = xs.len();
    let channels = xs[0].len();
    let n = xs[0][0].len();
    let mut data = vec![0f32; views * channels * n];
    for (v, per_channel) in xs.iter().enumerate() {
        for (c, col) in per_channel.iter().enumerate() {
            for (k, &x) in col.iter().enumerate() {
                data[(v * channels + c) * n + k] = x as f32;
            }
        }
    }
    (data, views, channels, n)
}

#[test]
fn sum_sq_diff_avx2_matches_scalar() {
    // The dispatched AVX2 residual reduction agrees with the scalar reference
    // (up to f32 summation order), including a length that exercises the n % 8
    // tail.
    for len in [0usize, 1, 7, 8, 31, 173, 512] {
        let row: Vec<f32> = (0..len)
            .map(|i| ((i * 7 % 53) as f32 - 26.0) * 0.13)
            .collect();
        let xbar: Vec<f32> = (0..len)
            .map(|i| ((i * 11 % 47) as f32 - 23.0) * 0.07)
            .collect();
        let want = sum_sq_diff_scalar(&row, &xbar, 0, len);
        let got = sum_sq_diff(&row, &xbar);
        assert!(
            (want - got).abs() <= 1e-3 * (1.0 + want.abs()),
            "len {len}: scalar {want} vs dispatched {got}"
        );
    }
}

#[test]
fn axpy_f32_avx2_matches_scalar() {
    // The dispatched AVX2 weighted-consensus SAXPY agrees with the scalar
    // reference, including a length that exercises the n % 8 tail.
    for len in [0usize, 1, 7, 8, 31, 173, 512] {
        let row: Vec<f32> = (0..len)
            .map(|i| ((i * 5 % 37) as f32 - 18.0) * 0.11)
            .collect();
        let w = 0.137f32;
        let init: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
        let mut want = init.clone();
        axpy_f32_scalar(&mut want, &row, w, 0, len);
        let mut got = init.clone();
        axpy_f32(&mut got, &row, w);
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}

#[test]
fn weighted_moments_avx2_matches_scalar() {
    // The dispatched AVX2 (f64-accumulate) moments agree with the scalar
    // reference, including a length that exercises the n % 4 tail.
    for len in [0usize, 1, 3, 4, 31, 173, 512] {
        let col: Vec<f32> = (0..len)
            .map(|i| ((i * 7 % 53) as f32) * 1.7 + 3.0)
            .collect();
        let w: Vec<f64> = (0..len).map(|i| 0.3 + (i % 5) as f64 * 0.13).collect();
        let (s1s, s2s) = weighted_moments_scalar(&col, &w, 0, len);
        let (s1, s2) = weighted_moments(&col, &w);
        assert!(
            (s1 - s1s).abs() <= 1e-9 * (1.0 + s1s.abs()),
            "s1 {s1s} vs {s1}"
        );
        assert!(
            (s2 - s2s).abs() <= 1e-9 * (1.0 + s2s.abs()),
            "s2 {s2s} vs {s2}"
        );
    }
}

#[test]
fn znorm_write_avx2_matches_scalar() {
    // The dispatched AVX2 normalize write agrees with the scalar reference,
    // including a length that exercises the n % 8 tail.
    for len in [0usize, 1, 7, 8, 31, 173, 512] {
        let src: Vec<f32> = (0..len)
            .map(|i| ((i * 11 % 41) as f32) * 0.9 - 7.0)
            .collect();
        let sqrt_weights: Vec<f32> = (0..len).map(|i| 0.5 + (i % 7) as f32 * 0.05).collect();
        let (mean, inv_norm) = (12.5f32, 0.031f32);
        let mut want = vec![0f32; len];
        znorm_write_scalar(&src, &sqrt_weights, mean, inv_norm, &mut want, 0, len);
        let mut got = vec![0f32; len];
        znorm_write(&src, &sqrt_weights, mean, inv_norm, &mut got);
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}

/// A unit-norm, zero-mean view that is `base` plus a small distinct perturbation.
fn consistent_view(base: &[f64], seed: u64, amount: f64) -> Vec<f64> {
    let pert = synthetic_normalized(seed, base.len());
    let mut v: Vec<f64> = base
        .iter()
        .zip(&pert)
        .map(|(b, p)| b + amount * p)
        .collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in &mut v {
        *x -= mean;
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[test]
fn irls_weights_stay_near_uniform_without_an_outlier() {
    // Distinct but mutually-consistent views (a shared signal + small per-view
    // perturbations): with no outlier the Tukey weights stay near-uniform.
    let base = synthetic_normalized(7, 24);
    let xs: Vec<Vec<Vec<f64>>> = (0..4)
        .map(|i| vec![consistent_view(&base, 100 + i as u64, 0.12)])
        .collect();
    let (d, vw, ch, nn) = flatten_stack(&xs);
    let mut sc = ConsensusScratch::default();
    irls_view_weights(&d, vw, ch, nn, 3, &mut sc);
    assert_relative_eq!(sc.w.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    for &w in &sc.w {
        assert!((w - 0.25).abs() < 0.05, "weights {:?}", sc.w);
    }
}

#[test]
fn irls_weights_reject_an_anticorrelated_outlier() {
    // Three agreeing views plus one anticorrelated outlier: IRLS drives the
    // outlier's weight to zero and shares the weight among the inliers.
    let good = synthetic_normalized(7, 24);
    let bad: Vec<f64> = good.iter().map(|x| -x).collect();
    let xs: Vec<Vec<Vec<f64>>> = vec![
        vec![good.clone()],
        vec![good.clone()],
        vec![good.clone()],
        vec![bad],
    ];
    let (d, vw, ch, nn) = flatten_stack(&xs);
    let mut sc = ConsensusScratch::default();
    irls_view_weights(&d, vw, ch, nn, 5, &mut sc);
    assert!(sc.w[3] < 1e-6, "outlier weight {}", sc.w[3]);
    for &w in &sc.w[..3] {
        assert_relative_eq!(w, 1.0 / 3.0, epsilon = 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Keypoint-anchored render patch
// ---------------------------------------------------------------------------

/// Project the patch center into a view, returning the source-image pixel — the
/// reprojection `project_i(X_p)` that the keypoint plumbing recenters against.
fn project_center(patch: &OrientedPatch, view: &ProjectedImage<'_>) -> [f64; 2] {
    let pc = view
        .cam_from_world
        .transform_point_homogeneous(patch.center.coords, patch.w);
    let (px, py) = view
        .camera
        .ray_to_pixel([pc.x, pc.y, pc.z])
        .expect("in frame");
    [px, py]
}

#[test]
fn view_render_patch_anchors_center_at_keypoint() {
    // Provable anchoring: recentering a view's patch onto a keypoint makes the
    // recentered center reproject to *exactly* that keypoint. Guards the
    // seed_offset axis order (x before y) and the round-trip math — a transposition
    // would fail since the offset is deliberately asymmetric in x and y.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(true_normal());
    for v in &views {
        let c = project_center(&patch, v);
        let kp = [c[0] + 9.0, c[1] - 6.0]; // asymmetric in x and y
        let rp = view_render_patch(&patch, v, Some(kp));
        let pc = v
            .cam_from_world
            .transform_point_homogeneous(rp.center.coords, rp.w);
        let (px, py) = v.camera.ray_to_pixel([pc.x, pc.y, pc.z]).expect("in frame");
        assert_relative_eq!(px, kp[0], epsilon = 1e-6);
        assert_relative_eq!(py, kp[1], epsilon = 1e-6);
    }
}

// ---- The middle ZNCC -----------------------------------------------------------

#[test]
fn the_middle_is_the_centred_square_half_the_grid_s_width() {
    use super::support::middle_span;
    assert_eq!(middle_span(24), 6..18);
    assert_eq!(middle_span(20), 5..15);
    // An odd grid keeps the middle symmetric about the centre.
    assert_eq!(middle_span(17), 4..13);
    let support = build_support(PatchWindow::Uniform, 24);
    assert_eq!(support.middle(24).len(), 12 * 12);
}

/// Two `R×R` single-channel grids, from `f(row, col)`, laid out over the uniform
/// support.
fn grid(r: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    (0..r * r).map(|p| f(p / r, p % r)).collect()
}

fn mean_over(values: &[f32], at: &[usize]) -> f32 {
    at.iter().map(|&k| values[k]).sum::<f32>() / at.len() as f32
}

#[test]
fn the_middle_zncc_equals_the_whole_when_the_outside_is_uniform() {
    let r = 24;
    let support = build_support(PatchWindow::Uniform, r as u32);
    let middle = support.middle(r as u32);
    let everything: Vec<usize> = (0..r * r).collect();
    let texture =
        |row: usize, col: usize| (row as f32 * 0.9).sin() * 40.0 + (col as f32 * 1.3).cos() * 30.0;
    let noise = |row: usize, col: usize| ((row * 7 + col * 13) % 11) as f32 - 5.0;
    let mut a = grid(r, |row, col| 100.0 + texture(row, col));
    let mut b = grid(r, |row, col| {
        50.0 + 2.0 * texture(row, col) + 3.0 * noise(row, col)
    });
    // Outside the middle each grid holds its own middle's mean, so the outside
    // adds nothing to either mean or either norm.
    let (fill_a, fill_b) = (mean_over(&a, &middle), mean_over(&b, &middle));
    for k in 0..r * r {
        if !middle.contains(&k) {
            a[k] = fill_a;
            b[k] = fill_b;
        }
    }
    let n = r * r;
    let whole = windowed_zncc_at(&a, &b, 1, n, &support.weights, &everything);
    let mid = windowed_zncc_at(&a, &b, 1, n, &support.weights, &middle);
    assert!(whole < 0.999, "the noise keeps it off 1: {whole}");
    assert_relative_eq!(whole, mid, epsilon = 1e-6);
}

#[test]
fn the_middle_zncc_falls_when_only_the_outside_matches() {
    let r = 24;
    let support = build_support(PatchWindow::Uniform, r as u32);
    let middle = support.middle(r as u32);
    let everything: Vec<usize> = (0..r * r).collect();
    let n = r * r;
    let shared =
        |row: usize, col: usize| (row as f32 * 0.7).sin() * 40.0 + (col as f32 * 0.5).cos() * 35.0;
    let own_a = |row: usize, col: usize| (row as f32 * 2.1 + col as f32 * 0.3).sin() * 40.0;
    let own_b = |row: usize, col: usize| (col as f32 * 1.7 - row as f32 * 0.4).cos() * 40.0;
    let span = 6..18;
    let inside = |row: usize, col: usize| span.contains(&row) && span.contains(&col);
    let a = grid(r, |row, col| {
        128.0
            + if inside(row, col) {
                own_a(row, col)
            } else {
                shared(row, col)
            }
    });
    let b = grid(r, |row, col| {
        128.0
            + if inside(row, col) {
                own_b(row, col)
            } else {
                shared(row, col)
            }
    });
    let whole = windowed_zncc_at(&a, &b, 1, n, &support.weights, &everything);
    let mid = windowed_zncc_at(&a, &b, 1, n, &support.weights, &middle);
    assert!(whole > 0.5, "the shared outside carries the whole: {whole}");
    assert!(mid.abs() < 0.3, "the middles do not agree: {mid}");
}

#[test]
fn the_middle_zncc_follows_the_whole_patch_flat_channel_conventions() {
    let r = 8;
    let support = build_support(PatchWindow::Uniform, r as u32);
    let middle = support.middle(r as u32);
    let n = r * r;
    let textured = grid(r, |row, col| (row * 3 + col * 5 % 7) as f32);
    let flat = vec![7.0f32; n];
    // A flat sample channel scores 0; a flat reference is left out, and a
    // reference with no textured channel has no reading.
    let two_channels = |x: &[f32], y: &[f32]| [x, y].concat();
    let sample = two_channels(&textured, &flat);
    let reference = two_channels(&textured, &textured);
    assert_relative_eq!(
        windowed_zncc_at(&sample, &reference, 2, n, &support.weights, &middle),
        0.5,
        epsilon = 1e-9
    );
    let reference = two_channels(&textured, &flat);
    assert_relative_eq!(
        windowed_zncc_at(
            &textured.repeat(2),
            &reference,
            2,
            n,
            &support.weights,
            &middle
        ),
        1.0,
        epsilon = 1e-9
    );
    assert!(windowed_zncc_at(&textured, &flat, 1, n, &support.weights, &middle).is_nan());
    assert!(windowed_zncc_at(&textured, &textured, 1, n, &support.weights, &[]).is_nan());
}

// ---- The ZNCC grid -------------------------------------------------------------

#[test]
fn the_zncc_grid_splits_the_grid_into_symmetric_thirds() {
    assert_eq!(grid_bounds(24), [0, 8, 16, 24]);
    // The middle third takes the remainder, so the split stays symmetric.
    assert_eq!(grid_bounds(25), [0, 8, 17, 25]);
    assert_eq!(grid_bounds(26), [0, 8, 18, 26]);

    let uniform = build_support(PatchWindow::Uniform, 24);
    let cells = uniform.grid_cells(24);
    for row in &cells {
        for cell in row {
            assert_eq!(cell.len(), 8 * 8);
        }
    }
    // Row-major from the top-left: the first support pixel is in the top-left
    // cell and the last in the bottom-right one.
    assert_eq!(cells[0][0][0], 0);
    assert_eq!(*cells[2][2].last().unwrap(), 24 * 24 - 1);

    // Under a disk the cells still partition the support, and a corner cell
    // holds only the part of its square the disk covers.
    let disk = build_support(PatchWindow::GaussianDisk { sigma: 0.6 }, 24);
    let cells = disk.grid_cells(24);
    let mut every: Vec<usize> = cells.iter().flatten().flatten().copied().collect();
    every.sort_unstable();
    assert_eq!(every, (0..disk.pixels.len()).collect::<Vec<_>>());
    assert_eq!(cells[1][1].len(), 8 * 8);
    assert!(cells[0][0].len() < cells[0][1].len());
    assert!(!cells[0][0].is_empty());
}

#[test]
fn the_zncc_grid_reads_each_cell_unweighted() {
    let r = 24;
    let n = r * r;
    // A steep window, so a weighted reading of a corner cell would differ from
    // an unweighted one.
    let support = build_support(PatchWindow::Gaussian { sigma: 0.3 }, r as u32);
    let uniform = build_support(PatchWindow::Uniform, r as u32);
    let cells = support.grid_cells(r as u32);
    let texture =
        |row: usize, col: usize| (row as f32 * 0.9).sin() * 40.0 + (col as f32 * 1.3).cos() * 30.0;
    let noise = |row: usize, col: usize| ((row * 7 + col * 13) % 11) as f32 - 5.0;
    let a = grid(r, |row, col| 100.0 + texture(row, col));
    // The bottom-right cell holds its own texture; the others are an affine
    // image of `a` with some noise.
    let b = grid(r, |row, col| {
        if row >= 16 && col >= 16 {
            100.0 + (col as f32 * 2.3 - row as f32 * 0.8).sin() * 40.0
        } else {
            50.0 + 2.0 * texture(row, col) + 3.0 * noise(row, col)
        }
    });
    let readings = zncc_grid(&a, &b, 1, n, &cells);
    for (row, readings_row) in readings.iter().enumerate() {
        for (col, &z) in readings_row.iter().enumerate() {
            let unweighted = windowed_zncc_at(&a, &b, 1, n, &uniform.weights, &cells[row][col]);
            assert_relative_eq!(z, unweighted, epsilon = 1e-9);
            if (row, col) == (2, 2) {
                assert!(z.abs() < 0.3, "the cell's own texture does not agree: {z}");
            } else {
                assert!(z > 0.9, "cell ({row}, {col}) agrees: {z}");
            }
        }
    }
    let weighted = windowed_zncc_at(&a, &b, 1, n, &support.weights, &cells[0][0]);
    assert!((weighted - readings[0][0]).abs() > 1e-6);
}

#[test]
fn a_flat_cell_has_no_zncc_grid_reading() {
    let r = 24;
    let n = r * r;
    let support = build_support(PatchWindow::Uniform, r as u32);
    let cells = support.grid_cells(r as u32);
    let textured = |row: usize, col: usize| ((row * 3 + col * 5) % 7) as f32;
    let a = grid(r, textured);
    // The reference is flat over its top-middle cell only.
    let b = grid(r, |row, col| {
        if row < 8 && (8..16).contains(&col) {
            9.0
        } else {
            textured(row, col)
        }
    });
    let readings = zncc_grid(&a, &b, 1, n, &cells);
    assert!(readings[0][1].is_nan());
    assert_relative_eq!(readings[1][1], 1.0, epsilon = 1e-9);
}

#[test]
fn the_square_s_cells_are_whole_rectangles() {
    let cells = square_cells(24);
    for row in &cells {
        for cell in row {
            assert_eq!(cell.len(), 8 * 8);
        }
    }
    // Positions of the square itself: the top-left cell starts at the top-left
    // pixel, and the bottom-right one ends at the last.
    assert_eq!(cells[0][0][..3], [0, 1, 2]);
    assert_eq!(cells[0][0][8], 24);
    assert_eq!(*cells[2][2].last().unwrap(), 24 * 24 - 1);
}

#[test]
fn a_cell_with_an_unread_pixel_has_no_zncc_grid_reading() {
    let r = 24;
    let n = r * r;
    let cells = square_cells(r as u32);
    let textured = |row: usize, col: usize| ((row * 3 + col * 5) % 7) as f32;
    let a = grid(r, textured);
    let mut b = a.clone();
    // A pixel of the bottom-left cell could not be sampled.
    b[20 * r + 2] = f32::NAN;
    let readings = zncc_grid(&a, &b, 1, n, &cells);
    assert!(readings[2][0].is_nan());
    assert_relative_eq!(readings[2][1], 1.0, epsilon = 1e-9);
}
