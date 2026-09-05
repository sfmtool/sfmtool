// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The triangulation numerics on their own, with no panel around them.

use sfmtool_core::SfmrReconstruction;

use super::{compute_max_pairwise_angle, compute_point_diagnostics};

#[test]
fn max_pairwise_angle_finds_the_widest_pair() {
    // Three rays: 0°, 45° and 90° from +X. The widest pair is the outer two.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let rays = [[1.0, 0.0, 0.0], [s, s, 0.0], [0.0, 1.0, 0.0]];
    let angle = compute_max_pairwise_angle(&rays);
    assert!((angle - 90.0).abs() < 1e-4, "angle was {angle}");
}

#[test]
fn max_pairwise_angle_of_fewer_than_two_rays_is_zero() {
    assert_eq!(compute_max_pairwise_angle(&[]), 0.0);
    assert_eq!(compute_max_pairwise_angle(&[[1.0, 0.0, 0.0]]), 0.0);
}

#[test]
fn point_diagnostics_are_undefined_for_a_missing_point() {
    let recon = SfmrReconstruction::demo(4);
    let (cond, z) = compute_point_diagnostics(&recon, 999);
    assert!(cond.is_nan());
    assert!(z.is_nan());
}

#[test]
fn point_diagnostics_are_finite_for_a_triangulated_point() {
    let recon = SfmrReconstruction::demo(12);
    let (cond, z) = compute_point_diagnostics(&recon, 5);
    assert!(cond.is_finite() && cond >= 1.0, "condition number {cond}");
    assert!(z.is_finite(), "inverse-depth z {z}");
}
