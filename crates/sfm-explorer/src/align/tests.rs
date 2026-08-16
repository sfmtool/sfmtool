// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Status-line text for an `Align to ▸` outcome.
//!
//! The fit behind these numbers is tested where it lives, in
//! `sfmtool-core`'s `analysis::alignment::reconstructions`; what is asserted
//! here is only how the viewer words the result.

use sfmtool_core::Se3Transform;

use super::*;

#[test]
fn the_success_message_reads_the_way_the_spec_writes_it() {
    let fit = AlignFit {
        transform: Se3Transform::identity(),
        correspondences: 243,
        inliers: 214,
        rms: 0.0312,
        source: AlignSource::Cameras,
    };
    assert_eq!(
        success_message("run_b", "run_a", &fit),
        "Aligned run_b → run_a: 214/243 cameras, RMS 0.031"
    );
}

#[test]
fn a_point_mode_fit_is_reported_in_points() {
    let fit = AlignFit {
        transform: Se3Transform::identity(),
        correspondences: 10_412,
        inliers: 9_988,
        rms: 0.0025,
        source: AlignSource::Points,
    };
    assert_eq!(
        success_message("b", "a", &fit),
        "Aligned b → a: 9988/10412 points, RMS 0.003"
    );
}

#[test]
fn the_failure_message_names_both_nodes_and_the_reason() {
    assert_eq!(
        failure_message(
            "run_b",
            "run_a",
            "the two reconstructions share no image names"
        ),
        "Align run_b → run_a failed: the two reconstructions share no image names"
    );
}
