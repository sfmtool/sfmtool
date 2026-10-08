// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the piecewise refinement: recovery of a perturbed affine shape,
//! the homography's second-order term in the residuals, refusal of the cells
//! over a second surface, a flat member, the bounds of the cell search, the
//! window sums against a direct pass, the update's fallback models, the revert
//! to the cascade shape when a refined shape fails a gate or leaves the
//! frame, the rejection of an update that lowers the whole-member ZNCC, the
//! early stop of an oscillating loop, the robust fit's refusal of an outlier
//! cell, and the stage as it runs inside `refine_cluster_patches`, on one
//! thread and on four. Those run the shape-moving loop
//! (`PiecewiseParams::move_shape`); the tests at the end run the default,
//! which measures the cells at the cascade's shape and leaves the member
//! bit-identical to the cascade's.

use super::super::kernels::{SupportTables, TemplateKernel};
use super::super::{
    build_template, inv2, member_zncc_at, mul2, read_member_at, refine_cluster_patches,
    refine_kept_member_cells, ClusterRefineParams, ClusterRefineResult, FeatureGeometry, Mat2,
    MemberGeo, MemberOutcome, MemberStatus,
};
use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::patch::normal_refine::build_support;
use crate::patch::normal_refine::grid_cell_centres;
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
/// default parameters with the piecewise stage on, cut from [`texture`]
/// around [`C`].
struct Fixture {
    params: ClusterRefineParams,
    tables: SupportTables,
    tmpl: TemplateKernel,
    layout: CellLayout,
    templates: CellTemplates,
    step: f64,
    off: f64,
}

/// The default piecewise settings with the shape-moving loop turned on.
fn loop_params() -> PiecewiseParams {
    PiecewiseParams {
        move_shape: true,
        ..PiecewiseParams::default()
    }
}

/// The fixture with the shape-moving loop on, which most tests here exercise.
fn fixture() -> Fixture {
    fixture_with(loop_params())
}

/// The fixture at the default settings, which measure the cells at the
/// cascade's shape and never move it.
fn measuring_fixture() -> Fixture {
    let fx = fixture_with(PiecewiseParams::default());
    assert!(!fx.pp().move_shape, "measuring is the default");
    fx
}

fn fixture_with(pp: PiecewiseParams) -> Fixture {
    let params = ClusterRefineParams {
        piecewise: Some(pp),
        ..ClusterRefineParams::default()
    };
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
        tables,
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
        self.run_scored(&pyr, s, p, self.score(&pyr))
    }

    /// [`Self::run`] on a built pyramid, with `score` in place of the
    /// cascade's objective.
    fn run_scored(
        &self,
        pyr: &ImageU8Pyramid,
        s: Mat2,
        p: [f64; 2],
        score: impl FnMut(&Mat2, [f64; 2]) -> Option<f64>,
    ) -> MemberCells {
        refine_member_cells(
            pyr,
            &self.tmpl.src_channels,
            &self.templates,
            &self.layout,
            s,
            p,
            self.step,
            self.off,
            &self.pp(),
            score,
        )
    }

    /// The cascade's objective on `pyr`: the whole-member windowed ZNCC at a
    /// shape and position.
    fn score<'a>(
        &'a self,
        pyr: &'a ImageU8Pyramid,
    ) -> impl FnMut(&Mat2, [f64; 2]) -> Option<f64> + 'a {
        move |s, p| {
            member_zncc_at(
                pyr,
                p,
                s,
                &self.tmpl,
                &self.tables,
                self.params.resolution,
                self.step,
                self.off,
            )
        }
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
    assert_eq!(out.cells.stop, LoopStop::Converged);
    assert!(out.cells.final_update_accepted);
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

/// How far the stored residuals are from the homography's second-order
/// term, for the plane whose map has second-order coefficients `h`:
/// `(largest error, RMS error, largest term)`, all in grid px over the nine
/// cells. The cells in `outliers` must be refused as outliers and every other
/// cell fitted.
fn homography_residual_errors(h: [f64; 2], outliers: &[(usize, usize)]) -> (f64, f64, f64) {
    let fx = fixture();
    let member = member_image(a_true(), h, None);
    let (s0, p0) = perturbed_start();
    let out = fx.run(&member, s0, p0);
    assert!(out.updated);
    assert!(
        out.cells.iterations <= 3,
        "{} iterations",
        out.cells.iterations
    );
    let (worst, rms, largest) = truth_residual_errors(&fx, &out, h, outliers);
    eprintln!(
        "h = {h:?}: largest error {worst:.4}, RMS {rms:.4}, term {largest:.4} grid px; {:?} {} {}",
        out.cells.stop, out.cells.iterations, out.cells.final_update_accepted
    );
    (worst, rms, largest)
}

/// How far `out`'s stored residuals are from where the plane with
/// second-order coefficients `h` puts each cell's content relative to the
/// returned shape's placement of the cell: `(largest error, RMS error,
/// largest true residual)`, in grid px over the nine cells. The cells in
/// `outliers` must be refused as outliers, which still stores their shifts,
/// and every other cell fitted.
fn truth_residual_errors(
    fx: &Fixture,
    out: &MemberCells,
    h: [f64; 2],
    outliers: &[(usize, usize)],
) -> (f64, f64, f64) {
    let s_inv = inv2(&out.shape);
    let (mut largest, mut worst, mut sq_err) = (0.0f64, 0.0f64, 0.0f64);
    for row in 0..3 {
        for col in 0..3 {
            let want_status = if outliers.contains(&(row, col)) {
                CellStatus::RefusedOutlier
            } else {
                CellStatus::Fitted
            };
            assert_eq!(
                out.cells.status[row][col], want_status,
                "{:?}",
                out.cells.status
            );
            // Where the true map sends the cell centre, in the returned
            // shape's grid coordinates.
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
            worst = worst.max(err);
            sq_err += err * err;
        }
    }
    let rms = (sq_err / 9.0).sqrt();
    (worst, rms, largest)
}

#[test]
fn residuals_follow_the_homography_second_order_term() {
    let (worst, rms, largest) = homography_residual_errors([0.003, -0.002], &[]);
    assert!(
        largest > 0.2,
        "the second-order term ({largest:.3} grid px) is too small to test"
    );
    assert!(
        worst < 0.12,
        "a residual is {worst:.3} grid px off the homography"
    );
    assert!(
        rms < 0.07,
        "residuals off the homography by {rms:.3} grid px RMS"
    );
}

#[test]
fn residuals_follow_a_doubled_second_order_term_less_closely() {
    // Twice the term of the test above. The residuals still follow it, but
    // the error grows faster than the term: measured at 0.29 grid px at worst
    // and 0.13 RMS here, against 0.093 and 0.058 at the single term. The
    // bottom-left corner cell, where the term is largest, is far enough off
    // the affine the other eight agree on that the robust fit refuses it as
    // an outlier; its shift is still stored and counted here.
    let (worst, rms, largest) = homography_residual_errors([0.006, -0.004], &[(2, 0)]);
    assert!(
        largest > 0.8,
        "the second-order term ({largest:.3} grid px) is too small to test"
    );
    assert!(
        worst < 0.35,
        "a residual is {worst:.3} grid px off the homography"
    );
    assert!(
        rms < 0.15,
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
    // than with nine cells. The whole-member ZNCC also reads the second
    // surface, so it does not peak exactly at the first plane's affine: the
    // loop stops when an update toward the first plane would lower it,
    // 0.104 px from the truth here.
    assert!(
        after < 0.12,
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
            let patch = WorkingPatch::new(
                side,
                1,
                (0..side * side)
                    .map(|p| texture((p % side) as f64 * 1.7, (p / side) as f64 * 1.3) as f32)
                    .collect(),
            );
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
    let patch = WorkingPatch::new(side, 1, samples);
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

/// Two images of the affine-warped plane run through `refine_cluster_patches`
/// with `params`, one cluster per entry of `starts`: cluster `k` is the
/// reference feature in the first image and a member in the second whose
/// seed shape and position are `starts[k]`.
fn run_clusters(params: &ClusterRefineParams, starts: &[(Mat2, [f64; 2])]) -> ClusterRefineResult {
    let img1 = make_image(128, 128, texture);
    let img2 = member_image(a_true(), [0.0, 0.0], None);
    let n = starts.len();
    let mut pos = Array2::<f32>::zeros((1, 2));
    let mut aff = Array3::<f32>::zeros((1, 2, 2));
    let mut pos2 = Array2::<f32>::zeros((n, 2));
    let mut aff2 = Array3::<f32>::zeros((n, 2, 2));
    for i in 0..2 {
        pos[[0, i]] = C[i] as f32;
        for j in 0..2 {
            aff[[0, i, j]] = A_REF[i][j] as f32;
        }
    }
    for (k, (s, p)) in starts.iter().enumerate() {
        for i in 0..2 {
            pos2[[k, i]] = p[i] as f32;
            for j in 0..2 {
                // A smaller scale than the reference, so the reference is
                // image 0.
                aff2[[k, i, j]] = s[i][j] as f32;
            }
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
    let cluster_starts: Vec<u32> = (0..=n as u32).map(|c| 2 * c).collect();
    let member_images: Vec<u32> = (0..n).flat_map(|_| [0, 1]).collect();
    let member_features: Vec<u32> = (0..n as u32).flat_map(|k| [0, k]).collect();
    refine_cluster_patches(
        &pyramids,
        &features,
        &cluster_starts,
        &member_images,
        &member_features,
        params,
        None,
    )
}

/// [`run_clusters`] with one cluster whose member starts at
/// [`perturbed_start`].
fn run_cluster(params: &ClusterRefineParams) -> ClusterRefineResult {
    run_clusters(params, &[perturbed_start()])
}

#[test]
fn the_stage_refines_kept_members_inside_the_kernel() {
    let fx = fixture();
    let truth = (mul2(&a_true(), &A_REF), C);
    let shape_of = |res: &ClusterRefineResult| {
        let s = res.member_affine_shapes.index_axis(ndarray::Axis(0), 1);
        let p = res.member_positions.index_axis(ndarray::Axis(0), 1);
        (
            [[s[[0, 0]], s[[0, 1]]], [s[[1, 0]], s[[1, 1]]]],
            [p[0], p[1]],
        )
    };

    // Off by default.
    assert_eq!(ClusterRefineParams::default().piecewise, None);
    let off = run_cluster(&ClusterRefineParams::default());
    assert_eq!(off.reference_members[0], 0);
    assert_eq!(off.member_status[1], MemberStatus::Kept);
    assert_eq!(off.cells, vec![None, None], "one entry per member");

    let on = run_cluster(&fx.params);
    assert_eq!(on.member_status, off.member_status);
    assert_eq!(on.cells.len(), 2, "one entry per member");
    assert_eq!(on.cells[0], None, "the reference has no cells");
    let cells = on.cells[1].expect("the kept member has cells");
    assert!(cells.iterations <= 3);
    assert_eq!(cells.status, [[CellStatus::Fitted; 3]; 3]);

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
    let again = run_cluster(&fx.params);
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

/// One member's cells as bits: shifts, ZNCCs, statuses and iterations.
type CellBits = (Vec<u32>, Vec<u32>, [[CellStatus; 3]; 3], u8);

#[test]
fn the_stage_gives_the_same_cells_on_one_thread_and_on_four() {
    let fx = fixture();
    let (s0, p0) = perturbed_start();
    // Clusters whose members start at different perturbations, so the
    // threads see different work.
    let starts: Vec<(Mat2, [f64; 2])> = (0..24)
        .map(|k| {
            let t = k as f64;
            let n = mul2(
                &[
                    [1.0 + 0.004 * (t - 12.0), 0.0],
                    [0.0, 1.0 + 0.004 * (t - 12.0)],
                ],
                &rot2(0.3 * (t - 12.0)),
            );
            (
                mul2(&s0, &n),
                [p0[0] + 0.05 * (t - 12.0), p0[1] - 0.03 * (t - 12.0)],
            )
        })
        .collect();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run_clusters(&fx.params, &starts))
    };
    let (one, four) = (run(1), run(4));
    assert_eq!(one.member_status, four.member_status);
    assert!(one.cells.iter().filter(|c| c.is_some()).count() >= 20);
    let bits = |r: &ClusterRefineResult| -> Vec<CellBits> {
        r.cells
            .iter()
            .flatten()
            .map(|c| {
                (
                    c.shift_px
                        .iter()
                        .flatten()
                        .flatten()
                        .map(|v| v.to_bits())
                        .collect(),
                    c.zncc.iter().flatten().map(|v| v.to_bits()).collect(),
                    c.status,
                    c.iterations,
                )
            })
            .collect()
    };
    assert_eq!(
        one.cells.iter().map(Option::is_some).collect::<Vec<_>>(),
        four.cells.iter().map(Option::is_some).collect::<Vec<_>>()
    );
    assert_eq!(bits(&one), bits(&four));
    let shapes = |r: &ClusterRefineResult| -> Vec<u64> {
        r.member_affine_shapes
            .iter()
            .chain(r.member_positions.iter())
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(shapes(&one), shapes(&four));
}

/// The correspondences a known update `c ↦ A·c + b` gives at the cell centres
/// `cells`, each of weight `1`.
fn correspondences(a: Mat2, b: [f64; 2], cells: &[(usize, usize)]) -> Vec<Correspondence> {
    let layout = CellLayout::new(25, 2.0);
    cells
        .iter()
        .map(|&(row, col)| {
            let c = layout.centres[row][col];
            let ac = matvec(&a, c);
            (c, [ac[0] + b[0] - c[0], ac[1] + b[1] - c[1]], 1.0)
        })
        .collect()
}

fn assert_update_near(got: &Update, a: Mat2, b: [f64; 2]) {
    let got_a = got.a.iter().flatten();
    for (g, w) in got_a.chain(&got.b).zip(a.iter().flatten().chain(&b)) {
        assert!((g - w).abs() < 1e-9, "{got:?}");
    }
}

#[test]
fn three_or_four_survivors_fit_a_similarity() {
    let side = CellLayout::new(25, 2.0).cell_side;
    let a = mul2(&[[1.02, 0.0], [0.0, 1.02]], &rot2(2.0));
    let b = [0.3, -0.2];
    for cells in [
        &[(0, 0), (1, 2), (2, 1)][..],
        &[(0, 0), (0, 2), (2, 0), (2, 2)][..],
    ] {
        let (update, model) =
            fit_update(&correspondences(a, b, cells), side).expect("a similarity");
        assert_eq!(model, UpdateModel::Similarity, "{cells:?}");
        assert_update_near(&update, a, b);
    }
}

#[test]
fn five_survivors_in_one_row_fit_a_similarity() {
    let side = CellLayout::new(25, 2.0).cell_side;
    let a = mul2(&[[0.98, 0.0], [0.0, 0.98]], &rot2(-1.5));
    let b = [-0.1, 0.25];
    // The middle row, and two corner cells carrying almost no weight: five
    // cells, spread along one axis only.
    let mut points = correspondences(a, b, &[(1, 0), (1, 1), (1, 2), (0, 0), (2, 2)]);
    points[3].2 = 1e-3;
    points[4].2 = 1e-3;
    let (update, model) = fit_update(&points, side).expect("a similarity");
    assert_eq!(model, UpdateModel::Similarity);
    assert_update_near(&update, a, b);
}

#[test]
fn a_singular_affine_falls_to_a_similarity_before_a_shift() {
    // Five cells on one line, with the spread bar at zero so they count as
    // spread: the affine's normal equations are singular, and the similarity
    // is fitted instead.
    let a = mul2(&[[1.01, 0.0], [0.0, 1.01]], &rot2(1.0));
    let b = [0.2, 0.1];
    let points: Vec<Correspondence> = [-8.0, -4.0, 0.0, 4.0, 8.0]
        .iter()
        .map(|&x| {
            let c = [x, 0.0];
            let ac = matvec(&a, c);
            (c, [ac[0] + b[0] - c[0], ac[1] + b[1] - c[1]], 1.0)
        })
        .collect();
    let (update, model) = fit_update(&points, 0.0).expect("a similarity");
    assert_eq!(model, UpdateModel::Similarity);
    assert_update_near(&update, a, b);
}

#[test]
fn one_or_two_survivors_fit_a_shift() {
    let side = CellLayout::new(25, 2.0).cell_side;
    let a = mul2(&[[1.02, 0.0], [0.0, 1.02]], &rot2(2.0));
    let b = [0.3, -0.2];
    for cells in [&[(1, 1)][..], &[(0, 0), (2, 2)][..]] {
        let points = correspondences(a, b, cells);
        let (update, model) = fit_update(&points, side).expect("a shift");
        assert_eq!(model, UpdateModel::Shift, "{cells:?}");
        assert_eq!(update.a, [[1.0, 0.0], [0.0, 1.0]]);
        // The mean of the survivors' shifts.
        let n = points.len() as f64;
        let mean = [
            points.iter().map(|p| p.1[0]).sum::<f64>() / n,
            points.iter().map(|p| p.1[1]).sum::<f64>() / n,
        ];
        assert!((update.b[0] - mean[0]).abs() < 1e-12);
        assert!((update.b[1] - mean[1]).abs() < 1e-12);
    }
    assert!(fit_update(&[], side).is_none());
}

/// A kept member at shape `s` and position `p`, with cascade readings.
fn kept_member(s: Mat2, p: [f64; 2]) -> MemberOutcome {
    MemberOutcome {
        status: MemberStatus::Kept,
        affine: [[s[0][0], s[0][1], p[0]], [s[1][0], s[1][1], p[1]]],
        zncc: 0.9,
        zncc_middle: 0.9,
        zncc_grid: [[0.9; 3]; 3],
        shift: 0.5,
        cells: None,
    }
}

impl Fixture {
    /// Run `refine_kept_member_cells` on a kept member of `member` at shape
    /// `s` and position `p`, with the seed at `seed` and the cascade gates of
    /// `params`.
    fn run_kept(
        &self,
        member: &ImageU8,
        s: Mat2,
        p: [f64; 2],
        seed: [f64; 2],
        params: &ClusterRefineParams,
    ) -> MemberOutcome {
        let pyr = ImageU8Pyramid::build(member, 6);
        let mut out = kept_member(s, p);
        refine_kept_member_cells(
            &mut out,
            &pyr,
            seed,
            &self.tmpl,
            &self.templates,
            &self.layout,
            &self.tables,
            self.params.resolution,
            self.step,
            self.off,
            params,
            &self.pp(),
        );
        out
    }
}

/// The member was left at its cascade shape, position and readings, and its
/// cells are all not attempted after `iterations` renders.
fn assert_reverted(out: &MemberOutcome, s: Mat2, p: [f64; 2], iterations: u8) {
    let cascade = kept_member(s, p);
    assert_eq!(out.affine, cascade.affine);
    assert_eq!(out.zncc, cascade.zncc);
    assert_eq!(out.zncc_middle, cascade.zncc_middle);
    assert_eq!(out.shift, cascade.shift);
    assert_eq!(out.status, MemberStatus::Kept);
    let cells = out.cells.expect("cells are stored");
    assert_eq!(cells.status, [[CellStatus::NotAttempted; 3]; 3]);
    assert_eq!(cells.iterations, iterations);
}

#[test]
fn a_refined_shape_that_fails_a_cascade_gate_is_reverted() {
    let fx = fixture();
    let member = member_image(a_true(), [0.0, 0.0], None);
    let (s0, p0) = perturbed_start();
    let accepted = fx.run_kept(&member, s0, p0, p0, &fx.params);
    let cells = accepted.cells.expect("cells are stored");
    assert_eq!(cells.status, [[CellStatus::Fitted; 3]; 3]);
    assert_ne!(accepted.affine, kept_member(s0, p0).affine);
    assert!(accepted.zncc < 0.9999, "{}", accepted.zncc);

    // The ZNCC read again at the refined shape is below the bar.
    let strict = ClusterRefineParams {
        min_zncc: 0.9999,
        ..fx.params.clone()
    };
    let out = fx.run_kept(&member, s0, p0, p0, &strict);
    assert_reverted(&out, s0, p0, cells.iterations);

    // The refined position is further from the seed than the bar allows.
    let far_seed = [p0[0] + 5.0, p0[1]];
    let out = fx.run_kept(&member, s0, p0, far_seed, &fx.params);
    assert_reverted(&out, s0, p0, cells.iterations);
}

/// The member image of the affine-warped plane with the keypoint at `centre`
/// rather than [`C`].
fn member_image_at(a: Mat2, centre: [f64; 2]) -> ImageU8 {
    let a_inv = inv2(&a);
    make_image(128, 128, move |x, y| {
        let z = matvec(&a_inv, [x - centre[0], y - centre[1]]);
        texture(C[0] + z[0], C[1] + z[1])
    })
}

#[test]
fn an_update_whose_support_leaves_the_frame_is_not_applied() {
    let fx = fixture();
    let s_true = mul2(&a_true(), &A_REF);
    // Start a little smaller than the truth and to its right, so the start's
    // support is in the frame.
    let s0 = mul2(&s_true, &[[0.97, 0.0], [0.0, 0.97]]);
    let in_frame = |pyr: &ImageU8Pyramid, s: &Mat2, p: [f64; 2]| {
        read_member_at(
            pyr,
            p,
            s,
            &fx.tmpl,
            &fx.tables,
            fx.params.resolution,
            fx.step,
            fx.off,
        )
        .is_some()
    };
    // Move the keypoint toward the left edge until the loop, scored by a
    // ZNCC that ignores the frame, would carry the start's support, which is
    // in the frame, out of it.
    let mut found = None;
    for k in 0..800 {
        let x = 40.0 - 0.05 * k as f64;
        let centre = [x, C[1]];
        let p0 = [x + 1.0, C[1]];
        let member = member_image_at(a_true(), centre);
        let pyr = ImageU8Pyramid::build(&member, 6);
        if !in_frame(&pyr, &s0, p0) {
            continue;
        }
        let blind = fx.run_scored(&pyr, s0, p0, |_: &Mat2, _: [f64; 2]| Some(1.0));
        if blind.updated && !in_frame(&pyr, &blind.shape, blind.position) {
            found = Some((member, pyr, p0));
            break;
        }
    }
    let (member, pyr, p0) = found.expect("a keypoint position where the loop leaves the frame");
    // With the real objective, which cannot read a shape whose support left
    // the frame, that update is rejected and the shape stays in the frame.
    let loop_out = fx.run_scored(&pyr, s0, p0, fx.score(&pyr));
    assert_eq!(loop_out.cells.stop, LoopStop::Rejected);
    assert!(!loop_out.cells.final_update_accepted);
    assert!(in_frame(&pyr, &loop_out.shape, loop_out.position));
    // The kept member keeps that shape and its readings.
    let out = fx.run_kept(&member, s0, p0, p0, &fx.params);
    let cells = out.cells.expect("cells are stored");
    // Compared as text, since the cells carry NaN.
    assert_eq!(format!("{cells:?}"), format!("{:?}", loop_out.cells));
    let sh = [
        [out.affine[0][0], out.affine[0][1]],
        [out.affine[1][0], out.affine[1][1]],
    ];
    assert_eq!(sh, loop_out.shape);
    assert_eq!([out.affine[0][2], out.affine[1][2]], loop_out.position);
}

/// The ZNCC of a cell against the working patch moved by `(dx, dy)`, by a
/// direct pass over a copy of the moved window: its mean removed, its norm
/// and its cross term with the template summed. The oracle for the
/// summed-area-table reading `cell_zncc_at`.
#[allow(clippy::too_many_arguments)]
fn direct_cell_zncc(
    tc: &CellTemplate,
    src_channels: &[usize],
    patch: &WorkingPatch,
    layout: &CellLayout,
    row: usize,
    col: usize,
    dx: i64,
    dy: i64,
) -> Option<f64> {
    let b = layout.bounds;
    let m = layout.margin as i64;
    let side = patch.side as i64;
    let (mut sum, mut scored) = (0.0, 0usize);
    for (tch, t) in tc.channels.iter().enumerate() {
        let Some((t, t_norm)) = t else {
            continue;
        };
        scored += 1;
        let src = src_channels[tch];
        if src >= patch.channels {
            continue;
        }
        let mut window = Vec::new();
        for y in b[row]..b[row + 1] {
            let py = y as i64 + dy + m;
            for x in b[col]..b[col + 1] {
                let px = x as i64 + dx + m;
                let v = patch.samples[(py * side + px) as usize * patch.channels + src];
                if !v.is_finite() {
                    return None;
                }
                window.push(f64::from(v));
            }
        }
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let (mut cross, mut norm_sq) = (0.0, 0.0);
        for (u, tv) in window.iter().zip(t) {
            let du = u - mean;
            cross += du * tv;
            norm_sq += du * du;
        }
        if norm_sq < FLAT_NORM_SQ_EPS {
            continue;
        }
        sum += cross / (norm_sq * t_norm).sqrt();
    }
    (scored > 0).then(|| sum / scored as f64)
}

#[test]
fn the_window_sums_agree_with_a_direct_pass() {
    // Two channels: a textured one, and one that is flat over the left half
    // of the patch, so some windows are flat in it. A few missing samples.
    for (resolution, bound) in [(25u32, 2.0f32), (12, 1.0), (26, 3.0)] {
        let layout = CellLayout::new(resolution, bound);
        let r = resolution as usize;
        let side = layout.patch_side();
        let tex = |x: f64, y: f64, c: usize| -> f64 {
            if c == 0 || x > side as f64 / 2.0 {
                texture(x * 1.7 + 3.0 * c as f64, y * 1.3)
            } else {
                131.0
            }
        };
        let square: Vec<f32> = (0..2)
            .flat_map(|c| (0..r * r).map(move |p| (c, p)))
            .map(|(c, p)| tex((p % r) as f64 + 1.3, (p / r) as f64 + 0.6, c) as f32)
            .collect();
        let tmpl = TemplateKernel {
            channels: 2,
            src_channels: vec![0, 1],
            kern: Vec::new(),
            kern_sums: Vec::new(),
            samples: Vec::new(),
            square_samples: square,
        };
        let templates = CellTemplates::new(&tmpl, &layout);
        let mut samples: Vec<f32> = (0..side * side)
            .flat_map(|p| {
                let (x, y) = ((p % side) as f64, (p / side) as f64);
                [
                    tex(x + 0.4, y - 0.2, 0) as f32,
                    tex(x + 0.4, y - 0.2, 1) as f32,
                ]
            })
            .collect();
        for p in [0, side * side - 1, side * (side / 2)] {
            samples[2 * p] = f32::NAN;
            samples[2 * p + 1] = f32::NAN;
        }
        let patch = WorkingPatch::new(side, 2, samples);
        let n = layout.reach as i64;
        let (mut compared, mut missing) = (0, 0);
        for row in 0..3 {
            for col in 0..3 {
                let tc = &templates.cells[row][col];
                for dy in -n..=n {
                    for dx in -n..=n {
                        let src = &tmpl.src_channels;
                        let got = cell_zncc_at(tc, src, &patch, &layout, row, col, dx, dy);
                        let want = direct_cell_zncc(tc, src, &patch, &layout, row, col, dx, dy);
                        match (got, want) {
                            (Some(g), Some(w)) => {
                                assert!(
                                    (g - w).abs() < 1e-6,
                                    "R {resolution}, cell ({row}, {col}) at ({dx}, {dy}): \
                                     {g} against {w}"
                                );
                                compared += 1;
                            }
                            (None, None) => missing += 1,
                            _ => panic!(
                                "cell ({row}, {col}) at ({dx}, {dy}): {got:?} against {want:?}"
                            ),
                        }
                    }
                }
            }
        }
        assert!(
            compared > 0 && missing > 0,
            "{compared} compared, {missing} missing"
        );
    }
}

#[test]
fn cell_statuses_share_the_matches_format_codes() {
    let all = [
        CellStatus::Fitted,
        CellStatus::RefusedCurvature,
        CellStatus::RefusedZncc,
        CellStatus::NotAttempted,
        CellStatus::RefusedBound,
        CellStatus::RefusedOutlier,
    ];
    assert_eq!(all.len(), ClusterCellStatus::ALL.len());
    for status in all {
        assert_eq!(status as u8, ClusterCellStatus::from(status) as u8);
    }
}

#[test]
fn member_cell_data_fills_kept_rows_and_leaves_the_rest_not_attempted() {
    let mut cell = CellRefinement::not_attempted(3);
    cell.shift_px[0][2] = [0.25, -0.5];
    cell.zncc[0][2] = 0.93;
    cell.status[0][2] = CellStatus::Fitted;
    cell.status[1][1] = CellStatus::RefusedBound;
    let data = member_cell_data(&[None, Some(cell), None]);

    assert_eq!(data.shift_px.shape(), &[3, 3, 3, 2]);
    assert_eq!(data.zncc.shape(), &[3, 3, 3]);
    assert_eq!(data.status.shape(), &[3, 3, 3]);
    assert_eq!(data.iterations.to_vec(), vec![0, 3, 0]);
    assert_eq!(data.shift_px[[1, 0, 2, 0]], 0.25);
    assert_eq!(data.shift_px[[1, 0, 2, 1]], -0.5);
    assert_eq!(data.zncc[[1, 0, 2]], 0.93);
    assert_eq!(data.status[[1, 0, 2]], ClusterCellStatus::Fitted as u8);
    assert_eq!(
        data.status[[1, 1, 1]],
        ClusterCellStatus::RefusedBound as u8
    );
    assert!(data.shift_px[[1, 1, 1, 0]].is_nan());
    for m in [0, 2] {
        assert!(data
            .status
            .index_axis(ndarray::Axis(0), m)
            .iter()
            .all(|&s| s == ClusterCellStatus::NotAttempted as u8));
        assert!(data
            .zncc
            .index_axis(ndarray::Axis(0), m)
            .iter()
            .all(|z| z.is_nan()));
    }
}

#[test]
fn an_update_that_lowers_the_whole_member_zncc_is_rejected() {
    let fx = fixture();
    let member = member_image(a_true(), [0.0, 0.0], None);
    let (s0, p0) = perturbed_start();
    let pyr = ImageU8Pyramid::build(&member, 6);
    // A score that is lower at every shape but the start, so the first
    // update, which the cells fit toward the truth, lowers it.
    let score = |s: &Mat2, p: [f64; 2]| Some(if *s == s0 && p == p0 { 0.99 } else { 0.5 });
    let out = fx.run_scored(&pyr, s0, p0, score);
    assert!(!out.updated);
    assert_eq!(out.shape, s0);
    assert_eq!(out.position, p0);
    assert_eq!(out.cells.iterations, 1);
    assert_eq!(out.cells.stop, LoopStop::Rejected);
    assert!(!out.cells.final_update_accepted);
    assert_eq!(out.cells.status, [[CellStatus::Fitted; 3]; 3]);
    // The stored shifts are the residual to the returned shape, the start:
    // they carry the start's whole first-order error, which the rejected
    // update would have removed.
    let (worst, rms, largest) = truth_residual_errors(&fx, &out, [0.0, 0.0], &[]);
    assert!(
        largest > 0.5,
        "the start is only {largest:.3} grid px off the truth"
    );
    // The cells are read at the perturbed start, where each cell is itself
    // scaled and rotated against the template, so they follow the truth
    // less closely than at a converged shape; a residual with the update's
    // first-order term removed would be off by the whole `largest`.
    assert!(
        worst < 0.2 && rms < 0.1,
        "a residual is {worst:.3} grid px off the truth (RMS {rms:.3})"
    );
}

/// A pass whose every cell is fitted with weight `1` and the shift `b`, and
/// whose update is the shift `b`.
fn scripted_pass(b: [f64; 2]) -> Pass {
    let reading = CellReading {
        status: CellStatus::Fitted,
        shift: b,
        zncc: 0.95,
        curvature: 0.1,
    };
    Pass {
        readings: [[reading; 3]; 3],
        update: Update {
            a: [[1.0, 0.0], [0.0, 1.0]],
            b,
        },
        model: UpdateModel::Shift,
        weights: [[1.0; 3]; 3],
    }
}

#[test]
fn an_oscillating_loop_stops_early_with_the_better_shape() {
    let layout = CellLayout::new(25, 2.0);
    let pp = loop_params();
    let step = 0.48;
    let (s0, p0) = (A_REF, C);
    // The updates alternate in sign and do not shrink: 0.3 grid px right,
    // then 0.35 back.
    let updates = [[0.3, 0.0], [-0.35, 0.0]];
    let after_one = [p0[0] + A_REF[0][0] * step * 0.3, p0[1]];
    let after_two = [p0[0] - A_REF[0][0] * step * 0.05, p0[1]];
    // The ZNCC rises at the first update; at the second it rises again, or
    // falls by less than the rejection tolerance.
    for (second_zncc, keeps_second) in [(0.9205, true), (0.92 - 0.5e-4, false)] {
        let mut k = 0;
        let pass = |_: &Mat2, _: [f64; 2]| {
            let pass = scripted_pass(updates[k % 2]);
            k += 1;
            Some(pass)
        };
        let mut scores = [0.90, 0.92, second_zncc].into_iter();
        let score = |_: &Mat2, _: [f64; 2]| scores.next();
        let out = run_loop(s0, p0, step, &layout, &pp, pass, score);
        assert_eq!(out.cells.iterations, 2, "stopped before the cap of 5");
        assert_eq!(out.cells.stop, LoopStop::Oscillation);
        assert_eq!(out.cells.final_update_accepted, keeps_second);
        assert!(out.updated);
        assert_eq!(out.shape, s0);
        let want = if keeps_second { after_two } else { after_one };
        assert!(
            (out.position[0] - want[0]).abs() < 1e-12 && (out.position[1] - want[1]).abs() < 1e-12,
            "{:?} against {want:?}",
            out.position
        );
        // The residual to the returned shape: none after the applied shift,
        // the whole measured shift when it was not applied.
        let want_shift = if keeps_second { 0.0 } else { -0.35 };
        for sh in out.cells.shift_px.iter().flatten() {
            assert!((f64::from(sh[0]) - want_shift).abs() < 1e-6, "{sh:?}");
            assert!(sh[1].abs() < 1e-6, "{sh:?}");
        }
    }
}

#[test]
fn an_outlier_cell_is_refused_and_the_fit_follows_the_inliers() {
    let layout = CellLayout::new(25, 2.0);
    let a = mul2(&[[1.01, 0.0], [0.0, 1.01]], &rot2(0.5));
    let b = [0.1, -0.05];
    let outlier = (0, 2);
    let readings: [[CellReading; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|col| {
            let c = layout.centres[row][col];
            let ac = matvec(&a, c);
            let mut shift = [ac[0] + b[0] - c[0], ac[1] + b[1] - c[1]];
            if (row, col) == outlier {
                shift[0] += 1.5;
                shift[1] -= 1.0;
            }
            CellReading {
                status: CellStatus::Fitted,
                shift,
                zncc: 0.95,
                curvature: 0.05 + 0.02 * (row * 3 + col) as f64,
            }
        })
    });
    let pass = fit_pass(readings, &layout).expect("a fit");
    for row in 0..3 {
        for col in 0..3 {
            let w = pass.weights[row][col];
            if (row, col) == outlier {
                assert_eq!(w, 0.0, "{:?}", pass.weights);
            } else {
                assert!(w > 0.0, "{:?}", pass.weights);
            }
        }
    }
    assert_eq!(pass.model, UpdateModel::Affine);
    assert_update_near(&pass.update, a, b);

    // Through the loop: the outlier is stored as refused_outlier with its
    // shift, the inliers as fitted with no residual.
    let pp = loop_params();
    let out = run_loop(
        A_REF,
        C,
        0.48,
        &layout,
        &pp,
        |_: &Mat2, _: [f64; 2]| Some(pass),
        |_: &Mat2, _: [f64; 2]| Some(0.9),
    );
    for row in 0..3 {
        for col in 0..3 {
            let (st, sh) = (out.cells.status[row][col], out.cells.shift_px[row][col]);
            if (row, col) == outlier {
                assert_eq!(st, CellStatus::RefusedOutlier);
                assert!(sh[0].hypot(sh[1]) > 1.5, "{sh:?}");
            } else {
                assert_eq!(st, CellStatus::Fitted);
                assert!(sh[0].hypot(sh[1]) < 1e-6, "{sh:?}");
            }
        }
    }
}

#[test]
fn measuring_leaves_every_cascade_output_bit_identical_inside_the_kernel() {
    let fx = measuring_fixture();
    let off = run_cluster(&ClusterRefineParams::default());
    let on = run_cluster(&fx.params);
    assert_eq!(on.member_status, off.member_status);
    assert_eq!(on.reference_members, off.reference_members);
    let bits64 = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits64(on.member_affine_shapes.as_slice().unwrap()),
        bits64(off.member_affine_shapes.as_slice().unwrap())
    );
    assert_eq!(
        bits64(on.member_positions.as_slice().unwrap()),
        bits64(off.member_positions.as_slice().unwrap())
    );
    assert_eq!(bits32(&on.member_zncc), bits32(&off.member_zncc));
    assert_eq!(
        bits32(&on.member_zncc_middle),
        bits32(&off.member_zncc_middle)
    );
    let grid = |r: &ClusterRefineResult| {
        r.member_zncc_grid
            .iter()
            .flatten()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(grid(&on), grid(&off));
    assert_eq!(bits32(&on.member_shift_px), bits32(&off.member_shift_px));

    // The kept member, whose start is perturbed, is measured once and not
    // moved, with every cell read.
    assert_eq!(on.member_status[1], MemberStatus::Kept);
    let cells = on.cells[1].expect("the kept member has cells");
    assert_eq!(cells.iterations, 1);
    assert_eq!(cells.stop, LoopStop::Measured);
    assert!(!cells.final_update_accepted);
    assert_eq!(cells.status, [[CellStatus::Fitted; 3]; 3]);
}

#[test]
fn measuring_stores_the_raw_shifts_which_carry_the_start_affine() {
    let fx = measuring_fixture();
    let member = member_image(a_true(), [0.0, 0.0], None);
    let (s0, p0) = perturbed_start();
    let pyr = ImageU8Pyramid::build(&member, 6);
    // The whole-member ZNCC is not read when the shape cannot move.
    let out = fx.run_scored(&pyr, s0, p0, |_: &Mat2, _: [f64; 2]| -> Option<f64> {
        panic!("the score is read only by the shape-moving loop")
    });
    assert!(!out.updated);
    assert_eq!(out.shape, s0);
    assert_eq!(out.position, p0);
    assert_eq!(out.cells.iterations, 1);
    assert_eq!(out.cells.stop, LoopStop::Measured);
    assert!(!out.cells.final_update_accepted);
    assert_eq!(out.model, Some(UpdateModel::Affine));
    assert_eq!(out.cells.status, [[CellStatus::Fitted; 3]; 3]);

    // The stored shifts are the displacements from the start's placement of
    // each cell, as the plane puts them: the perturbation's affine, which no
    // fitted map is removed from. Measured at the perturbed start, where each
    // cell is itself scaled and rotated against the template, they follow it
    // to within a fifth of a grid px.
    let (worst, rms, largest) = truth_residual_errors(&fx, &out, [0.0, 0.0], &[]);
    assert!(
        largest > 0.5,
        "the start is only {largest:.3} grid px off the truth"
    );
    assert!(
        worst < 0.2 && rms < 0.1,
        "a shift is {worst:.3} grid px off the start's displacement (RMS {rms:.3})"
    );

    // They are the shifts the loop reads at its first render, bit for bit:
    // the loop with its first update rejected stores those as measured.
    let lp = fixture();
    let rejected = lp.run_scored(&pyr, s0, p0, |s: &Mat2, p: [f64; 2]| {
        Some(if *s == s0 && p == p0 { 0.99 } else { 0.5 })
    });
    assert_eq!(rejected.cells.stop, LoopStop::Rejected);
    assert_eq!(
        format!(
            "{:?}",
            (out.cells.shift_px, out.cells.zncc, out.cells.status)
        ),
        format!(
            "{:?}",
            (
                rejected.cells.shift_px,
                rejected.cells.zncc,
                rejected.cells.status
            )
        )
    );
}

#[test]
fn measuring_refuses_an_outlier_and_keeps_every_raw_shift() {
    let layout = CellLayout::new(25, 2.0);
    let a = mul2(&[[1.01, 0.0], [0.0, 1.01]], &rot2(0.5));
    let b = [0.1, -0.05];
    let outlier = (0, 2);
    let readings: [[CellReading; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|col| {
            let c = layout.centres[row][col];
            let ac = matvec(&a, c);
            let mut shift = [ac[0] + b[0] - c[0], ac[1] + b[1] - c[1]];
            if (row, col) == outlier {
                shift[0] += 1.5;
                shift[1] -= 1.0;
            }
            CellReading {
                status: CellStatus::Fitted,
                shift,
                zncc: 0.95,
                curvature: 0.05 + 0.02 * (row * 3 + col) as f64,
            }
        })
    });
    let pass = fit_pass(readings, &layout).expect("a fit");
    let mut renders = 0;
    let out = run_loop(
        A_REF,
        C,
        0.48,
        &layout,
        &PiecewiseParams::default(),
        |_: &Mat2, _: [f64; 2]| {
            renders += 1;
            Some(pass)
        },
        |_: &Mat2, _: [f64; 2]| -> Option<f64> { panic!("the score is not read") },
    );
    assert_eq!(renders, 1);
    assert!(!out.updated);
    assert_eq!((out.shape, out.position), (A_REF, C));
    assert_eq!(out.cells.iterations, 1);
    assert_eq!(out.cells.stop, LoopStop::Measured);
    assert!(!out.cells.final_update_accepted);
    for (row, readings_row) in readings.iter().enumerate() {
        for (col, reading) in readings_row.iter().enumerate() {
            let want = if (row, col) == outlier {
                CellStatus::RefusedOutlier
            } else {
                CellStatus::Fitted
            };
            assert_eq!(out.cells.status[row][col], want);
            // Every shift is stored as read, the fitted map not removed.
            let rd = reading.shift;
            assert_eq!(out.cells.shift_px[row][col], [rd[0] as f32, rd[1] as f32]);
        }
    }

    // A failed render or fit leaves every cell not attempted after one
    // render, at the cascade's shape.
    let failed = run_loop(
        A_REF,
        C,
        0.48,
        &layout,
        &PiecewiseParams::default(),
        |_: &Mat2, _: [f64; 2]| None,
        |_: &Mat2, _: [f64; 2]| Some(0.9),
    );
    assert!(!failed.updated);
    assert_eq!(failed.cells.iterations, 1);
    assert_eq!(failed.cells.stop, LoopStop::NotRun);
    assert_eq!(failed.cells.status, [[CellStatus::NotAttempted; 3]; 3]);
}

#[test]
fn the_residual_scale_is_the_rayleigh_median_factor() {
    // The residuals are 2-D lengths: under isotropic Gaussian noise of
    // per-axis σ their median is σ·√(2 ln 2).
    let want = 1.0 / (2.0 * std::f64::consts::LN_2).sqrt();
    assert!(
        (MEDIAN_LENGTH_TO_SIGMA - want).abs() < 1e-15,
        "{MEDIAN_LENGTH_TO_SIGMA} vs {want}"
    );
}

#[test]
fn cell_layout_centres_are_the_shared_cell_centres() {
    // The cell plane normals cast their rays through `grid_cell_centres`; the
    // refinement must measure its cells about the same points. Each centre is
    // also the mean of the integer grid positions its cell spans.
    for resolution in [6u32, 12, 13, 24, 25] {
        let layout = CellLayout::new(resolution, 2.0);
        let shared = grid_cell_centres(resolution);
        let bounds = grid_bounds(resolution);
        let mid = (resolution as f64 - 1.0) / 2.0;
        let span_mean = |t: usize| {
            let n = (bounds[t + 1] - bounds[t]) as f64;
            (bounds[t]..bounds[t + 1]).map(|i| i as f64).sum::<f64>() / n - mid
        };
        for (row, shared_row) in shared.iter().enumerate() {
            for (col, &centre) in shared_row.iter().enumerate() {
                assert_eq!(layout.centres[row][col], centre, "R={resolution}");
                assert_eq!(
                    centre,
                    [span_mean(col), span_mean(row)],
                    "R={resolution} cell ({row}, {col})"
                );
            }
        }
    }
}
