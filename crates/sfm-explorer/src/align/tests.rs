// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Status-line text for an `Align to ▸` outcome.
//!
//! The fit behind these numbers is tested where it lives, in
//! `sfmtool-core`'s `analysis::alignment::reconstructions`; what is asserted
//! here is only how the viewer words the result.

use sfmtool_core::analysis::alignment::ImagePairing;
use sfmtool_core::Se3Transform;

use super::*;

/// A pairing of `n` images, each by its file hash.
fn by_hash(n: u32) -> ImagePairing {
    ImagePairing {
        pairs: (0..n).map(|i| (i, i)).collect(),
        by_file_hash: n as usize,
        by_name: 0,
        same_name_different_file: Vec::new(),
    }
}

#[test]
fn the_success_message_reads_the_way_the_spec_writes_it() {
    let fit = AlignFit {
        transform: Se3Transform::identity(),
        correspondences: 243,
        inliers: 214,
        rms: 0.0312,
        source: AlignSource::Cameras,
        point_matching: None,
        images: by_hash(243),
    };
    assert_eq!(
        success_message("run_b", "run_a", &fit),
        "Aligned run_b → run_a: 214/243 cameras, RMS 0.031; images paired: 243 by file hash, 0 by name"
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
        point_matching: Some(PointMatching::FeatureIndex),
        images: by_hash(17),
    };
    assert_eq!(
        success_message("b", "a", &fit),
        "Aligned b → a: 9988/10412 points, RMS 0.003; images paired: 17 by file hash, 0 by name"
    );
}

#[test]
fn a_fit_matched_by_keypoint_position_says_so() {
    let fit = AlignFit {
        transform: Se3Transform::identity(),
        correspondences: 223,
        inliers: 212,
        rms: 0.0041,
        source: AlignSource::Points,
        point_matching: Some(PointMatching::KeypointPosition),
        images: by_hash(17),
    };
    assert_eq!(
        success_message("v02", "v01", &fit),
        "Aligned v02 → v01: 212/223 points by keypoint position, RMS 0.004; images paired: 17 by file hash, 0 by name"
    );
}

#[test]
fn the_failure_message_names_both_nodes_and_the_reason() {
    assert_eq!(
        failure_message("run_b", "run_a", "the two reconstructions share no images"),
        "Align run_b → run_a failed: the two reconstructions share no images"
    );
}

#[test]
fn the_success_message_names_the_names_shared_by_different_files() {
    let fit = AlignFit {
        transform: Se3Transform::identity(),
        correspondences: 12,
        inliers: 10,
        rms: 0.02,
        source: AlignSource::Cameras,
        point_matching: None,
        images: ImagePairing {
            pairs: (0..12).map(|i| (i, i)).collect(),
            by_file_hash: 9,
            by_name: 3,
            same_name_different_file: vec![(12, 12), (13, 13)],
        },
    };
    assert_eq!(
        success_message("b", "a", &fit),
        "Aligned b → a: 10/12 cameras, RMS 0.020; images paired: 9 by file hash, 3 by name; 2 names shared by different files"
    );
}
