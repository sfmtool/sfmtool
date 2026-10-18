// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pieces of the grown plane normal that need no photograph: the robust
//! cell score, the residual's link to ZNCC, the window schedule, the
//! elimination of the shifts from the information, the fit itself on a
//! residual the test writes, and the information-weighted mean.

use std::cell::RefCell;

use nalgebra::{DMatrix, Matrix2};

use super::combine::{covariance, distance, weighted_mean};
use super::render::{robust_score, zncc, Block, Samples};
use super::solve::{maximise, schur_tilt, tilt_information, Iteration, Linearised, Objective};
use super::{sigmas_deg, window_radii, GrownPlaneNormalOptions};

/// Samples of `channels` channels, every one valid, from `values` laid out
/// sample by sample.
fn samples(values: &[f32], channels: usize) -> Samples {
    let n = values.len() / channels;
    Samples {
        channels,
        values: values.to_vec(),
        valid: vec![true; n],
    }
}

/// A deterministic sequence in `0 .. 255`.
fn pseudo(k: usize, seed: u64) -> f32 {
    let mut h = (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
    h ^= h >> 29;
    h = h.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 32;
    (h % 256) as f32
}

#[test]
fn every_cell_agreeing_scores_their_mean() {
    let cells = [0.9, 0.8, 0.85, 0.9, 0.95, 0.8, 0.85, 0.9, 0.88].map(Some);
    let (score, weights) = robust_score(&cells);
    let mean = cells.iter().map(|c| c.unwrap()).sum::<f64>() / 9.0;
    // Nothing is past the cut: the lowest, 0.8, is 0.08 under the median 0.88
    // against a cut of 4.685 × 0.05 at least.
    assert!(weights.iter().all(|&w| w > 0.7), "{weights:?}");
    assert!((score - mean).abs() < 0.01, "{score} vs {mean}");
}

/// The top row of a view that does not see the reference's texture there:
/// dino 528's image 45, whose bottom two rows match at 0.67 to 0.81 and whose
/// top row reads 0.40, -0.06 and 0.06. The robust score is the bottom rows',
/// and a flat cell is not counted.
#[test]
fn cells_that_do_not_match_are_discounted() {
    let cells = [
        Some(0.40),
        Some(-0.06),
        Some(0.06),
        Some(0.81),
        Some(0.75),
        Some(0.70),
        Some(0.67),
        None,
        Some(0.78),
    ];
    let (score, weights) = robust_score(&cells);
    assert_eq!(weights[7], 0.0, "a flat cell has no weight");
    assert_eq!(weights[1], 0.0, "{weights:?}");
    assert_eq!(weights[2], 0.0, "{weights:?}");
    // The median is 0.685 and the upper cells' scale 0.119, so the cut is
    // 0.556: 0.40 is 0.285 under the median and keeps about half its weight.
    assert!(weights[0] > 0.4 && weights[0] < 0.6, "{weights:?}");
    let mean = cells.iter().flatten().sum::<f64>() / 8.0;
    assert!(score > 0.68 && score < 0.8, "{score}: {weights:?}");
    assert!(score > mean + 0.15, "{score} vs {mean}");
}

#[test]
fn no_live_cell_scores_minus_one() {
    assert_eq!(robust_score(&[None; 9]).0, -1.0);
}

/// A block's residual sum of squares is `2 · (1 − ZNCC)`, the identity that
/// lets the fit minimise a residual for the best ZNCC.
#[test]
fn a_blocks_residual_is_two_times_one_less_the_zncc() {
    for channels in [1, 3] {
        let n = 40;
        let a: Vec<f32> = (0..n * channels).map(|k| pseudo(k, 1)).collect();
        let b: Vec<f32> = (0..n * channels)
            .map(|k| 0.6 * a[k] + 0.4 * pseudo(k, 7) + 10.0)
            .collect();
        let (ra, rb) = (samples(&a, channels), samples(&b, channels));
        let rows: Vec<usize> = (0..n).collect();
        let z = zncc(&ra, &rb, &rows, 0.6);
        assert!(z > 0.3 && z < 0.99, "{z}");
        for weight in [1.0, 0.25] {
            let block = Block::freeze(&ra, &rb, &rows, weight, 0.6).expect("textured");
            let mut r = Vec::new();
            block.residuals(&rb, &mut r);
            let ss: f64 = r.iter().map(|v| v * v).sum();
            assert!(
                (ss - weight * 2.0 * (1.0 - z)).abs() < 1e-9,
                "{channels} channels, weight {weight}: {ss} vs {z}"
            );
        }
    }
}

#[test]
fn too_few_samples_on_the_photograph_score_minus_one() {
    let n = 20;
    let a: Vec<f32> = (0..n).map(|k| pseudo(k, 3)).collect();
    let ra = samples(&a, 1);
    let mut rb = samples(&a, 1);
    for k in 0..9 {
        rb.valid[k] = false;
    }
    let rows: Vec<usize> = (0..n).collect();
    // 11 of 20 is under 60%.
    assert_eq!(zncc(&ra, &rb, &rows, 0.6), -1.0);
    assert!(Block::freeze(&ra, &rb, &rows, 1.0, 0.6).is_none());
    rb.valid[0] = true;
    assert!((zncc(&ra, &rb, &rows, 0.6) - 1.0).abs() < 1e-9);
}

#[test]
fn a_flat_reference_channel_is_not_scored() {
    let n = 30;
    // Channel 0 textured, channel 1 flat in the reference.
    let mut a = Vec::new();
    let mut b = Vec::new();
    for k in 0..n {
        a.extend([pseudo(k, 5), 100.0]);
        b.extend([pseudo(k, 5), pseudo(k, 9)]);
    }
    let rows: Vec<usize> = (0..n).collect();
    let z = zncc(&samples(&a, 2), &samples(&b, 2), &rows, 0.6);
    assert!((z - 1.0).abs() < 1e-9, "only the textured channel: {z}");
}

#[test]
fn the_windows_follow_the_grid() {
    let options = GrownPlaneNormalOptions::default();
    assert_eq!(window_radii(48, &options), [3, 6, 10, 16, 24]);
    assert_eq!(window_radii(24, &options), [3, 5, 8, 12]);
    assert_eq!(window_radii(32, &options), [3, 4, 6, 11, 16]);
}

/// The shifts' elimination is the profile: for a quadratic score, minus its
/// Hessian in the tilt with the shifts fitted again at every tilt.
#[test]
fn eliminating_the_shifts_gives_the_profile_information() {
    // N = JᵀJ for a residual linear in (a, b, s, t).
    let j = DMatrix::from_row_slice(
        5,
        4,
        &[
            1.0, 0.2, 0.5, 0.0, //
            0.0, 1.0, 0.0, 0.5, //
            0.3, 0.1, 1.0, 0.2, //
            0.2, 0.4, 0.1, 1.0, //
            0.5, 0.5, 0.3, 0.3,
        ],
    );
    let n = j.transpose() * &j;
    let info = schur_tilt(&n);
    // Directly: minimise over (s, t) for tilt e_a, e_b; ‖J (δ, s)‖² is the
    // profile δᵀ I δ.
    let profile = |d: [f64; 2]| {
        let jt = j.columns(0, 2).into_owned();
        let js = j.columns(2, 2).into_owned();
        let rt = jt * nalgebra::Vector2::new(d[0], d[1]);
        let s = (js.transpose() * &js).try_inverse().unwrap() * (js.transpose() * &rt);
        (rt - js * s).norm_squared()
    };
    let e = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    for d in e {
        let v = nalgebra::Vector2::new(d[0], d[1]);
        let q = (v.transpose() * info * v)[(0, 0)];
        assert!((q - profile(d)).abs() < 1e-9, "{q} vs {}", profile(d));
    }
}

#[test]
fn sigma_reads_the_eigenvalues() {
    // λ = 2τ / σ² for σ of 1 and 2 degrees.
    let tau = 0.01;
    let lambda = |deg: f64| 2.0 * tau / deg.to_radians().powi(2);
    let info = Matrix2::new(lambda(2.0), 0.0, 0.0, lambda(1.0));
    let [weak, strong] = sigmas_deg(&info, tau);
    assert!((weak - 2.0).abs() < 1e-9 && (strong - 1.0).abs() < 1e-9);
}

/// A residual the test writes: `r = p − target`, scaled so that the score is
/// `1 − ½‖r‖²`, as a block's would be.
struct Quadratic {
    target: Vec<f64>,
}

impl Objective for Quadratic {
    type Frozen = ();

    fn freeze(&self, p: &[f64]) -> (f64, Option<Linearised<()>>) {
        let mut r = Vec::new();
        self.residuals(&(), p, &mut r);
        let ss: f64 = r.iter().map(|v| v * v).sum();
        (1.0 - 0.5 * ss, Some(((), r)))
    }

    fn residuals(&self, _: &(), p: &[f64], out: &mut Vec<f64>) {
        out.extend(p.iter().zip(&self.target).map(|(a, b)| a - b));
    }
}

fn iteration() -> Iteration {
    let o = GrownPlaneNormalOptions::default();
    Iteration {
        max_iterations: o.max_iterations,
        damping: o.initial_damping,
        min_improvement: o.min_improvement,
        tilt_step: o.jacobian_tilt_step,
        shift_step: o.jacobian_shift_step_px,
    }
}

/// The fit reaches the target of a residual it can see, and its information
/// is the residual's.
#[test]
fn the_fit_reaches_the_optimum_of_a_quadratic() {
    let target = vec![0.35, -0.2, 5.0, -3.0];
    let objective = Quadratic {
        target: target.clone(),
    };
    let (p, score, _) = maximise(&objective, &[0.0; 4], &iteration());
    for (a, b) in p.iter().zip(&target) {
        assert!((a - b).abs() < 1e-6, "{p:?}");
    }
    assert!((score - 1.0).abs() < 1e-9);
    // r = p − target has J = I, so the shifts are independent of the tilt
    // and the tilt's information is the identity.
    let info = tilt_information(&objective, &p, &iteration());
    assert!((info - Matrix2::identity()).norm() < 1e-6, "{info}");
}

/// [`Quadratic`] with a log of the points it is read at: each `freeze` and
/// each `residuals` call stands for one render of a real objective.
struct Logged {
    inner: Quadratic,
    frozen_at: RefCell<Vec<Vec<f64>>>,
    read_at: RefCell<Vec<Vec<f64>>>,
}

impl Objective for Logged {
    type Frozen = ();

    fn freeze(&self, p: &[f64]) -> (f64, Option<Linearised<()>>) {
        self.frozen_at.borrow_mut().push(p.to_vec());
        self.inner.freeze(p)
    }

    fn residuals(&self, frozen: &(), p: &[f64], out: &mut Vec<f64>) {
        self.read_at.borrow_mut().push(p.to_vec());
        self.inner.residuals(frozen, p, out);
    }
}

/// The fit reads the residual at a point it has scored from that score's
/// render: every `residuals` call is a Jacobian column, at a point no
/// `freeze` was made at, so no point is rendered twice.
#[test]
fn the_fit_does_not_render_a_scored_point_again() {
    let objective = Logged {
        inner: Quadratic {
            target: vec![0.35, -0.2, 5.0, -3.0],
        },
        frozen_at: RefCell::new(Vec::new()),
        read_at: RefCell::new(Vec::new()),
    };
    let (_, _, taken) = maximise(&objective, &[0.0; 4], &iteration());
    assert!(taken >= 2, "{taken}");
    let frozen_at = objective.frozen_at.borrow();
    let read_at = objective.read_at.borrow();
    assert_eq!(read_at.len() % 4, 0, "{}", read_at.len());
    for q in read_at.iter() {
        assert!(!frozen_at.contains(q), "{q:?} rendered twice");
    }
}

/// The information-weighted mean against a hand-computed case:
/// `I₁ = [[2, 1], [1, 2]]` at `t₁ = (1, 1)` and `I₂ = [[1, 0], [0, 3]]` at
/// `t₂ = (0, −1)` sum to `[[3, 1], [1, 5]]`, whose inverse is
/// `[[5, −1], [−1, 3]] / 14`, and `I₁t₁ + I₂t₂ = (3, 0)`, so the mean is
/// `(15, −3) / 14`. With diagonal information each axis is its own weighted
/// mean, one estimate's mean is itself, and the covariance `2τ·(Σ I)⁻¹`
/// gives the `σ` of the summed information.
#[test]
fn the_weighted_mean_matches_a_hand_computed_case() {
    let i1 = Matrix2::new(2.0, 1.0, 1.0, 2.0);
    let i2 = Matrix2::new(1.0, 0.0, 0.0, 3.0);
    let mean = weighted_mean(&[([1.0, 1.0], i1), ([0.0, -1.0], i2)]).expect("two estimates");
    assert!((mean[0] - 15.0 / 14.0).abs() < 1e-5, "{mean:?}");
    assert!((mean[1] + 3.0 / 14.0).abs() < 1e-5, "{mean:?}");

    let diagonal = weighted_mean(&[
        ([1.0, 0.0], Matrix2::new(4.0, 0.0, 0.0, 1.0)),
        ([0.0, 2.0], Matrix2::identity()),
    ])
    .expect("two estimates");
    assert!((diagonal[0] - 0.8).abs() < 1e-5 && (diagonal[1] - 1.0).abs() < 1e-5);

    // One estimate, informed along one axis only, is its own mean.
    let one = weighted_mean(&[([0.3, -0.2], Matrix2::new(5.0, 0.0, 0.0, 0.0))]).expect("one");
    assert!(
        (one[0] - 0.3).abs() < 1e-9 && (one[1] + 0.2).abs() < 1e-9,
        "{one:?}"
    );
    assert!(weighted_mean(&[]).is_none());

    let tau = 0.01;
    let sum = i1 + i2;
    let cov = covariance(&sum, tau);
    let expected = Matrix2::new(5.0, -1.0, -1.0, 3.0) * (2.0 * tau / 14.0);
    assert!((cov - expected).norm() < 1e-8, "{cov}");
    // Two estimates at the same plane are no distance apart; one the
    // summed covariance away along an axis is one apart.
    assert!(distance([0.1, 0.2], &i1, [0.1, 0.2], &i2, tau).abs() < 1e-12);
    let unit = Matrix2::identity() * (4.0 * tau);
    let d = distance([1.0, 0.0], &unit, [0.0, 0.0], &unit, tau);
    assert!((d - 1.0).abs() < 1e-4, "{d}");
}
