// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The texture information: the structure tensor, the significance
//! threshold and its simulation, the parallel sum, the flow floor, and one
//! pair's samples and disc.

use crate::features::optical_flow::{FlowField, GrayImage};

use super::super::crops::PairLevel;
use super::{
    eigen, floored, noise_threshold, parallel_sum, significant, simulate_noise_threshold,
    structure_tensors, Gaussian, PairSamples, Sym2, QUANTIZATION_VARIANCE,
};

fn close(a: Sym2, b: Sym2, tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
}

/// `[[c, −s], [s, c]] · diag(l1, l2) · its transpose`.
fn rotated(l1: f64, l2: f64, angle: f64) -> Sym2 {
    let (s, c) = angle.sin_cos();
    [
        l1 * c * c + l2 * s * s,
        (l1 - l2) * c * s,
        l1 * s * s + l2 * c * c,
    ]
}

#[test]
fn a_ramp_has_its_gradient_squared_over_the_window() {
    let (w, h) = (20u32, 16u32);
    let data = (0..h)
        .flat_map(|_| (0..w).map(|c| 0.01 * c as f32))
        .collect();
    let image = GrayImage::new(w, h, data);
    let tensors = structure_tensors(&image, &vec![true; (w * h) as usize], 5);
    let inside = tensors[8 * w as usize + 10];
    assert!((inside[0] - 25.0 * 1e-4).abs() < 1e-7, "{inside:?}");
    assert_eq!(inside[1], 0.0);
    assert_eq!(inside[2], 0.0);
    // An invalid sample gives its two neighbours no gradient: of the
    // window's columns 8 to 12, columns 10 and 12 read none beside an
    // invalid column 11, so three columns' worth is left.
    let mut valid = vec![true; (w * h) as usize];
    for r in 0..h as usize {
        valid[r * w as usize + 11] = false;
    }
    let cut = structure_tensors(&image, &valid, 5)[8 * w as usize + 10];
    assert!((cut[0] - 15.0 * 1e-4).abs() < 1e-7, "{cut:?}");
}

#[test]
fn significance_subtracts_the_threshold_from_each_eigenvalue() {
    let t = rotated(10.0, 3.0, 0.4);
    assert!(close(significant(t, 4.0), rotated(6.0, 0.0, 0.4), 1e-12));
    assert!(close(significant(t, 12.0), [0.0; 3], 1e-12));
    assert!(close(
        significant([5.0, 0.0, 5.0], 1.0),
        [4.0, 0.0, 4.0],
        1e-12
    ));
}

#[test]
fn the_parallel_sum_is_never_more_than_either_tensor() {
    let a = rotated(9.0, 1.0, 0.3);
    let b = rotated(4.0, 2.0, 1.2);
    let p = parallel_sum(a, b);
    // Equal tensors halve; a flat crop gives nothing; and `a − p` and
    // `b − p` are non-negative.
    assert!(close(parallel_sum(a, a), a.map(|v| v / 2.0), 1e-12));
    assert!(close(parallel_sum(a, [0.0; 3]), [0.0; 3], 1e-12));
    for other in [a, b] {
        let gap = [other[0] - p[0], other[1] - p[1], other[2] - p[2]];
        let ([_, least], _) = eigen(gap);
        assert!(least >= -1e-12, "{gap:?}");
    }
    // Texture along one axis in one crop and the other axis in the other
    // gives a flow fixed along neither.
    let x = [3.0, 0.0, 0.0];
    let y = [0.0, 0.0, 3.0];
    assert!(close(parallel_sum(x, y), [0.0; 3], 1e-12));
}

#[test]
fn the_floor_caps_a_correlated_squares_information() {
    let l2 = 64.0;
    let f = 0.03;
    let strong = floored([1e9, 0.0, 1e9], f, l2);
    assert!((strong[0] - 1.0 / (f * f * l2)).abs() < 1e-3 * strong[0]);
    let weak = floored([2.0, 0.0, 0.5], f, l2);
    assert!((weak[0] - 2.0 / (1.0 + f * f * 2.0) / l2).abs() < 1e-12);
    assert!(close(
        floored([2.0, 0.0, 0.5], 0.0, l2),
        [2.0 / l2, 0.0, 0.5 / l2],
        1e-15
    ));
}

#[test]
fn the_noise_threshold_is_the_same_on_every_call() {
    let tau = noise_threshold(8, 1.0, 0, 0.001);
    assert_eq!(tau, noise_threshold(8, 1.0, 0, 0.001));
    assert_eq!(tau, simulate_noise_threshold(8, 4, 0, 0.001));
    // The central difference of unit white noise has a variance of 1/2 per
    // component, so a window of 64 holds 32 on each diagonal entry on
    // average; the 99.9% quantile of the larger eigenvalue is well above it.
    assert!(tau > 40.0 && tau < 80.0, "{tau}");
    // A wider window sums more noise and needs a higher threshold. An
    // upsampled crop's noise is smooth, so its gradients are smaller beside
    // its variance and the threshold, in units of that variance, lower.
    let wider = noise_threshold(12, 1.0, 0, 0.001);
    let smooth = noise_threshold(8, 2.0, 0, 0.001);
    assert!(wider > tau && smooth < tau, "{tau} {wider} {smooth}");
    // A laxer false alarm rate lowers it.
    assert!(noise_threshold(8, 1.0, 0, 0.01) < tau);
}

/// Fresh white noise, not the simulation's, passes the threshold in about
/// one window in a thousand.
#[test]
fn white_noise_passes_the_threshold_about_as_often_as_asked() {
    let side = 200usize;
    let mut normal = Gaussian::new(77);
    let data: Vec<f32> = (0..side * side).map(|_| normal.next() as f32).collect();
    let image = GrayImage::new(side as u32, side as u32, data);
    let tensors = structure_tensors(&image, &vec![true; side * side], 8);
    let tau = noise_threshold(8, 1.0, 0, 0.001);
    let (mut over, mut windows) = (0usize, 0usize);
    for r in 5..side - 4 {
        for c in 5..side - 4 {
            windows += 1;
            let t = tensors[r * side + c].map(f64::from);
            if significant(t, tau) != [0.0; 3] {
                over += 1;
            }
        }
    }
    let share = over as f64 / windows as f64;
    assert!(share < 0.004, "{over} of {windows}");
}

#[test]
fn identical_crops_give_the_quantization_floor_and_no_disc_past_the_square() {
    let side = 32u32;
    let mut normal = Gaussian::new(3);
    let data: Vec<f32> = (0..side * side)
        .map(|_| (0.5 + 0.05 * normal.next()) as f32)
        .collect();
    let image = GrayImage::new(side, side, data);
    let valid = vec![true; (side * side) as usize];
    let flow = FlowField::new(side, side);
    let level = PairLevel::new(1.0, 1.0);
    let samples = PairSamples::new((&image, &valid), (&image, &valid), &flow, 16, &level, 8);
    // Every sample but the last row and column, whose bilinear read needs
    // a neighbour beyond the crop.
    assert_eq!(samples.positions.len(), 31 * 31);
    assert_eq!(samples.reach, 16.0);
    assert_eq!(samples.positions[0], [-15.5, -15.5]);
    let disc = samples.disc(10.0, [50.0, 50.0], 0.03).unwrap();
    assert_eq!(disc.residual_variance, QUANTIZATION_VARIANCE);
    assert_eq!(disc.l2, 64.0);
    assert!(disc.positions.iter().all(|p| p[0].hypot(p[1]) <= 10.0));
    // Textured crops carry information once noise is so low.
    assert!(disc.information.iter().any(|w| w[0] + w[2] > 0.0));
    assert!(samples.disc(16.5, [50.0, 50.0], 0.03).is_none());
}
