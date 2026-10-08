// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the piecewise refinement: recovery of a perturbed affine shape,
//! the homography's second-order term in the residuals, refusal of the cells
//! over a second surface, a flat member, the bounds of the cell search, and
//! the stage as it runs inside `refine_cluster_patches`.

use super::super::kernels::{SupportTables, TemplateKernel};
use super::super::{
    build_template, inv2, mul2, refine_cluster_patches, ClusterRefineParams, FeatureGeometry, Mat2,
    MemberGeo, MemberStatus,
};
use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::patch::normal_refine::build_support;
use ndarray::{Array2, Array3};

/// The band-limited texture of the cluster-refinement tests, with fine terms
/// so every cell carries texture.
fn texture(x: f64, y: f64) -> f64 {
    127.0
        + 40.0 * (0.11 * x + 0.06 * y + 1.3).sin()
        + 28.0 * (0.05 * x - 0.12 * y + 0.7).sin()
        + 20.0 * (0.17 * x + 0.13 * y + 2.9).sin()
        + 10.0 * (0.29 * x - 0.23 * y + 0.4).cos()
        + 15.0 * (0.83 * x + 0.47 * y + 0.2).sin()
        + 12.0 * (-0.52 * x + 0.88 * y + 1.1).sin()
}

/// A second texture, unrelated to [`texture`].
fn other_texture(x: f64, y: f64) -> f64 {
    127.0
        + 45.0 * (0.71 * x - 0.33 * y + 2.1).sin()
        + 35.0 * (-0.21 * x + 0.93 * y + 0.3).sin()
        + 25.0 * (0.47 * x + 0.61 * y + 1.7).cos()
}

fn make_image(w: u32, h: u32, f: impl Fn(f64, f64) -> f64) -> ImageU8 {
    let mut data = vec![0u8; (w * h) as usize];
    for row in 0..h {
        for col in 0..w {
            let v = f(col as f64 + 0.5, row as f64 + 0.5);
            data[(row * w + col) as usize] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    ImageU8::new(w, h, 1, data)
}

fn rot2(deg: f64) -> Mat2 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s], [s, c]]
}

fn matvec(a: &Mat2, x: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * x[0] + a[0][1] * x[1],
        a[1][0] * x[0] + a[1][1] * x[1],
    ]
}

/// Image centre: the reference keypoint, and the point the warps fix.
const C: [f64; 2] = [64.0, 64.0];
/// The reference's SIFT shape.
const A_REF: Mat2 = [[2.5, 0.0], [0.0, 2.5]];
/// The member's true affine warp about [`C`].
fn a_true() -> Mat2 {
    mul2(
        &[[0.9, 0.0], [0.0, 0.9]],
        &mul2(&rot2(12.0), &[[1.0, 0.1], [0.0, 1.0]]),
    )
}

/// The cluster's template, the cell layout and the grid constants at the
/// default parameters, cut from [`texture`] around [`C`].
struct Fixture {
    params: ClusterRefineParams,
    tmpl: TemplateKernel,
    layout: CellLayout,
    templates: CellTemplates,
    step: f64,
    off: f64,
}

fn fixture() -> Fixture {
    let params = ClusterRefineParams::default();
    let resolution = params.resolution;
    let step = 2.0 * params.radius / resolution as f64;
    let off = 0.5 * step - params.radius;
    let support = build_support(params.window, resolution);
    let tables = SupportTables::new(&support, resolution);
    let img = make_image(128, 128, texture);
    let pyr = ImageU8Pyramid::build(&img, 6);
    let geo = MemberGeo {
        k_global: 0,
        image: 0,
        pos: C,
        a: A_REF,
        scale: 2.5,
    };
    let tmpl = build_template(&pyr, &geo, &support, &tables, resolution, step, off)
        .expect("textured template");
    let pp = params.piecewise.clone().unwrap();
    let layout = CellLayout::new(resolution, pp.cell_shift_bound_px);
    let templates = CellTemplates::new(&tmpl, &layout);
    Fixture {
        params,
        tmpl,
        layout,
        templates,
        step,
        off,
    }
}

impl Fixture {
    fn pp(&self) -> PiecewiseParams {
        self.params.piecewise.clone().unwrap()
    }

    fn run(&self, member: &ImageU8, s: Mat2, p: [f64; 2]) -> MemberCells {
        let pyr = ImageU8Pyramid::build(member, 6);
        refine_member_cells(
            &pyr,
            &self.tmpl.src_channels,
            &self.templates,
            &self.layout,
            s,
            p,
            self.step,
            self.off,
            &self.pp(),
        )
    }

    /// RMSE in member-image px between two maps `p + S·u` over the template
    /// grid.
    fn map_rmse(&self, a: (Mat2, [f64; 2]), b: (Mat2, [f64; 2])) -> f64 {
        let r = self.params.resolution as usize;
        let mut sq = 0.0;
        for row in 0..r {
            for col in 0..r {
                let u = [
                    self.off + self.step * col as f64,
                    self.off + self.step * row as f64,
                ];
                let (x, y) = (matvec(&a.0, u), matvec(&b.0, u));
                sq += (a.1[0] + x[0] - b.1[0] - y[0]).powi(2)
                    + (a.1[1] + x[1] - b.1[1] - y[1]).powi(2);
            }
        }
        (sq / (r * r) as f64).sqrt()
    }
}

/// The member image of the plane whose reference-to-member map is
/// `x ↦ C + A·q / (1 + h·q)`, `q = x − C`, with `second` drawn instead of
/// [`texture`] where the reference point's `q_x` exceeds `split`.
fn member_image(a: Mat2, h: [f64; 2], split: Option<f64>) -> ImageU8 {
    let a_inv = inv2(&a);
    make_image(128, 128, move |x, y| {
        let z = matvec(&a_inv, [x - C[0], y - C[1]]);
        let den = 1.0 - (h[0] * z[0] + h[1] * z[1]);
        let q = [z[0] / den, z[1] / den];
        match split {
            Some(s) if q[0] > s => other_texture(C[0] + q[0], C[1] + q[1]),
            _ => texture(C[0] + q[0], C[1] + q[1]),
        }
    })
}

/// The starting shape: the true affine shape perturbed by a known affine
/// (scale 1.04, rotation 3°) and the position moved by `(0.5, −0.4)` px.
fn perturbed_start() -> (Mat2, [f64; 2]) {
    let s_true = mul2(&a_true(), &A_REF);
    let n = mul2(&[[1.04, 0.0], [0.0, 1.04]], &rot2(3.0));
    (mul2(&s_true, &n), [C[0] + 0.5, C[1] - 0.4])
}

#[test]
fn recovers_a_perturbed_affine_with_nine_fitted_cells() {
    let fx = fixture();
    let member = member_image(a_true(), [0.0, 0.0], None);
    let (s0, p0) = perturbed_start();
    let truth = (mul2(&a_true(), &A_REF), C);
    let before = fx.map_rmse((s0, p0), truth);
    let out = fx.run(&member, s0, p0);
    let after = fx.map_rmse((out.shape, out.position), truth);
    assert!(out.updated);
    assert_eq!(out.model, Some(UpdateModel::Affine));
    assert!(
        out.cells.iterations <= 3,
        "{} iterations",
        out.cells.iterations
    );
    for row in out.cells.status {
        for st in row {
            assert_eq!(st, CellStatus::Fitted, "{:?}", out.cells.status);
        }
    }
    assert!(before > 0.5, "the start is {before:.3} px off");
    assert!(
        after < 0.05,
        "recovered within {after:.4} px (from {before:.3})"
    );
    // An affine warp leaves no second-order term: every residual is small.
    for sh in out.cells.shift_px.iter().flatten() {
        assert!(sh[0].hypot(sh[1]) < 0.1, "{:?}", out.cells.shift_px);
    }
    for z in out.cells.zncc.iter().flatten() {
        assert!(*z > 0.95, "{:?}", out.cells.zncc);
    }
}

#[test]
fn residuals_follow_the_homography_second_order_term() {
    let fx = fixture();
    let h = [0.003, -0.002];
    let member = member_image(a_true(), h, None);
    let (s0, p0) = perturbed_start();
    let out = fx.run(&member, s0, p0);
    assert!(out.updated);
    assert!(
        out.cells.iterations <= 3,
        "{} iterations",
        out.cells.iterations
    );
    let s_inv = inv2(&out.shape);
    let (mut largest, mut sq_err) = (0.0f64, 0.0f64);
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(out.cells.status[row][col], CellStatus::Fitted);
            // Where the true map sends the cell centre, in the converged
            // map's grid coordinates.
            let c = fx.layout.centres[row][col];
            let q = matvec(&A_REF, [fx.step * c[0], fx.step * c[1]]);
            let den = 1.0 + h[0] * q[0] + h[1] * q[1];
            let aq = matvec(&a_true(), q);
            let y = [C[0] + aq[0] / den, C[1] + aq[1] / den];
            let u = matvec(&s_inv, [y[0] - out.position[0], y[1] - out.position[1]]);
            let want = [u[0] / fx.step - c[0], u[1] / fx.step - c[1]];
            let got = out.cells.shift_px[row][col];
            largest = largest.max(want[0].hypot(want[1]));
            let err = (f64::from(got[0]) - want[0]).hypot(f64::from(got[1]) - want[1]);
            sq_err += err * err;
            assert!(
                err < 0.12,
                "cell ({row}, {col}): residual {got:?}, the homography gives {want:?}"
            );
        }
    }
    assert!(
        largest > 0.2,
        "the second-order term ({largest:.3} grid px) is too small to test"
    );
    let rms = (sq_err / 9.0).sqrt();
    assert!(
        rms < 0.07,
        "residuals off the homography by {rms:.3} grid px RMS"
    );
}

#[test]
fn cells_over_a_second_surface_are_refused() {
    let fx = fixture();
    // The second plane starts a grid px and a half inside the right column
    // of cells, so the bilinear footprint of the middle column reads none of
    // it at the converged shape. `split` is in reference-image px from the
    // keypoint.
    let b = fx.layout.bounds[2] as f64 + 1.0 - (fx.layout.resolution as f64 - 1.0) / 2.0;
    let split = b * fx.step * A_REF[0][0];
    let member = member_image(a_true(), [0.0, 0.0], Some(split));
    let (s0, p0) = perturbed_start();
    let truth = (mul2(&a_true(), &A_REF), C);
    let out = fx.run(&member, s0, p0);
    assert!(out.updated);
    for row in 0..3 {
        for col in 0..3 {
            let st = out.cells.status[row][col];
            if col == 2 {
                assert_ne!(st, CellStatus::Fitted, "{:?}", out.cells.status);
            } else {
                assert_eq!(st, CellStatus::Fitted, "{:?}", out.cells.status);
            }
        }
    }
    assert_eq!(out.model, Some(UpdateModel::Affine));
    let after = fx.map_rmse((out.shape, out.position), truth);
    // The update extrapolates over the refused third, so the bound is looser
    // than with nine cells.
    assert!(
        after < 0.1,
        "recovered the first plane within {after:.4} px"
    );
}

#[test]
fn a_flat_member_is_left_at_its_cascade_shape() {
    let fx = fixture();
    let member = make_image(128, 128, |_, _| 127.0);
    let (s0, p0) = perturbed_start();
    let out = fx.run(&member, s0, p0);
    assert!(!out.updated);
    assert_eq!(out.shape, s0);
    assert_eq!(out.position, p0);
    assert_eq!(out.cells.iterations, 1);
    assert_eq!(out.cells.status, [[CellStatus::NotAttempted; 3]; 3]);
    assert!(out
        .cells
        .shift_px
        .iter()
        .flatten()
        .flatten()
        .all(|v| v.is_nan()));
}

#[test]
fn the_cell_search_reads_only_the_working_patch() {
    // Every cell, at every grid size and bound, searched over a working patch
    // of exactly the rendered size: the search's debug assertion and the
    // buffer's own bounds catch a read past the render.
    for resolution in [6u32, 12, 13, 24, 25, 26] {
        for bound in [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] {
            let layout = CellLayout::new(resolution, bound);
            let r = resolution as usize;
            let square: Vec<f32> = (0..r * r)
                .map(|p| texture((p % r) as f64 * 1.7, (p / r) as f64 * 1.3) as f32)
                .collect();
            let tmpl = TemplateKernel {
                channels: 1,
                src_channels: vec![0],
                kern: Vec::new(),
                kern_sums: Vec::new(),
                samples: Vec::new(),
                square_samples: square,
            };
            let templates = CellTemplates::new(&tmpl, &layout);
            let side = layout.patch_side();
            assert_eq!(side, r + 2 * (bound.ceil() as usize));
            let patch = WorkingPatch {
                side,
                channels: 1,
                samples: (0..side * side)
                    .map(|p| texture((p % side) as f64 * 1.7, (p / side) as f64 * 1.3) as f32)
                    .collect(),
            };
            let pp = PiecewiseParams {
                cell_shift_bound_px: bound,
                ..PiecewiseParams::default()
            };
            for row in 0..3 {
                for col in 0..3 {
                    read_cell(
                        &templates.cells[row][col],
                        &tmpl.src_channels,
                        &patch,
                        &layout,
                        &pp,
                        row,
                        col,
                    );
                }
            }
        }
    }
}

#[test]
fn a_missing_sample_leaves_its_cell_not_attempted() {
    let layout = CellLayout::new(25, 2.0);
    let r = 25;
    let square: Vec<f32> = (0..r * r)
        .map(|p| texture((p % r) as f64 * 1.7, (p / r) as f64 * 1.3) as f32)
        .collect();
    let tmpl = TemplateKernel {
        channels: 1,
        src_channels: vec![0],
        kern: Vec::new(),
        kern_sums: Vec::new(),
        samples: Vec::new(),
        square_samples: square,
    };
    let templates = CellTemplates::new(&tmpl, &layout);
    let side = layout.patch_side();
    let mut samples: Vec<f32> = (0..side * side)
        .map(|p| texture((p % side) as f64 * 1.7, (p / side) as f64 * 1.3) as f32)
        .collect();
    // The top-left corner of the render, which only the top-left cell's
    // search reaches.
    samples[0] = f32::NAN;
    let patch = WorkingPatch {
        side,
        channels: 1,
        samples,
    };
    let pp = PiecewiseParams::default();
    let read = |row: usize, col: usize| {
        read_cell(
            &templates.cells[row][col],
            &tmpl.src_channels,
            &patch,
            &layout,
            &pp,
            row,
            col,
        )
        .status
    };
    assert_eq!(read(0, 0), CellStatus::NotAttempted);
    assert_ne!(read(1, 1), CellStatus::NotAttempted);
}

/// Two images of the affine-warped plane, the member's seed perturbed, run
/// through `refine_cluster_patches` with `params`.
fn run_cluster(params: &ClusterRefineParams) -> crate::patch::cluster_refine::ClusterRefineResult {
    let img1 = make_image(128, 128, texture);
    let img2 = member_image(a_true(), [0.0, 0.0], None);
    let (s0, p0) = perturbed_start();
    let a_mem = s0;
    let mut pos = Array2::<f32>::zeros((1, 2));
    let mut aff = Array3::<f32>::zeros((1, 2, 2));
    let mut pos2 = Array2::<f32>::zeros((1, 2));
    let mut aff2 = Array3::<f32>::zeros((1, 2, 2));
    for i in 0..2 {
        pos[[0, i]] = C[i] as f32;
        pos2[[0, i]] = p0[i] as f32;
        for j in 0..2 {
            aff[[0, i, j]] = A_REF[i][j] as f32;
            // A smaller scale than the reference, so the reference is image 0.
            aff2[[0, i, j]] = a_mem[i][j] as f32;
        }
    }
    let features = [
        FeatureGeometry {
            positions_xy: pos.view(),
            affine_shapes: aff.view(),
        },
        FeatureGeometry {
            positions_xy: pos2.view(),
            affine_shapes: aff2.view(),
        },
    ];
    let pyramids = [
        ImageU8Pyramid::build(&img1, 6),
        ImageU8Pyramid::build(&img2, 6),
    ];
    refine_cluster_patches(
        &pyramids,
        &features,
        &[0, 2],
        &[0, 1],
        &[0, 0],
        params,
        None,
    )
}

#[test]
fn the_stage_refines_kept_members_inside_the_kernel() {
    let fx = fixture();
    let truth = (mul2(&a_true(), &A_REF), C);
    let shape_of = |res: &crate::patch::cluster_refine::ClusterRefineResult| {
        let s = res.member_affine_shapes.index_axis(ndarray::Axis(0), 1);
        let p = res.member_positions.index_axis(ndarray::Axis(0), 1);
        (
            [[s[[0, 0]], s[[0, 1]]], [s[[1, 0]], s[[1, 1]]]],
            [p[0], p[1]],
        )
    };

    let off = run_cluster(&ClusterRefineParams {
        piecewise: None,
        ..ClusterRefineParams::default()
    });
    assert_eq!(off.reference_members[0], 0);
    assert_eq!(off.member_status[1], MemberStatus::Kept);
    assert!(off.cells.is_empty());

    let on = run_cluster(&ClusterRefineParams::default());
    assert_eq!(on.member_status, off.member_status);
    assert_eq!(on.cells.len(), 1, "one entry per kept member");
    assert!(on.cells[0].iterations <= 3);
    assert_eq!(on.cells[0].status, [[CellStatus::Fitted; 3]; 3]);

    let cascade = fx.map_rmse(shape_of(&off), truth);
    let piecewise = fx.map_rmse(shape_of(&on), truth);
    assert!(
        piecewise <= cascade.max(0.05),
        "piecewise {piecewise:.4} px against the cascade's {cascade:.4} px"
    );
    assert!(piecewise < 0.05, "piecewise {piecewise:.4} px");
    // The readings are taken again at the refined shape.
    assert!(on.member_zncc[1] > 0.95);
    let (_, p) = shape_of(&on);
    let seed = [p0_f32()[0], p0_f32()[1]];
    let drift = (p[0] - seed[0]).hypot(p[1] - seed[1]) as f32;
    assert!((on.member_shift_px[1] - drift).abs() < 1e-5);

    // Two runs agree bit for bit.
    let again = run_cluster(&ClusterRefineParams::default());
    assert_eq!(
        on.member_affine_shapes.as_slice().unwrap(),
        again.member_affine_shapes.as_slice().unwrap()
    );
    assert_eq!(format!("{:?}", on.cells), format!("{:?}", again.cells));
}

/// The perturbed seed position as the kernel reads it, through `f32`.
fn p0_f32() -> [f64; 2] {
    let (_, p0) = perturbed_start();
    [p0[0] as f32 as f64, p0[1] as f32 as f64]
}
