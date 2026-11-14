// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_no_outliers() {
    // All points are consistent (pure translation), non-collinear 3D points
    #[rustfmt::skip]
    let source: Vec<f64> = vec![
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,  1.0, 1.0, 0.0,  1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,  1.0, 1.0, 1.0,  2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
    ];
    let target: Vec<f64> = source.iter().map(|&x| x + 1.0).collect();
    let n = 10;

    let mask = ransac_alignment(&source, &target, n, 100, 0.5, 3, 42).unwrap();
    let inlier_count: usize = mask.iter().filter(|&&b| b).count();
    assert_eq!(inlier_count, n);
}

#[test]
fn test_known_outliers_rejected() {
    // 8 inliers with pure translation, 2 outliers with big offset
    // Use non-collinear 3D points so the least-squares fit can find a valid rotation
    let inlier_sources: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];

    let mut source = vec![0.0; 30];
    let mut target = vec![0.0; 30];

    for (i, pts) in inlier_sources.iter().enumerate() {
        let base = i * 3;
        source[base] = pts[0];
        source[base + 1] = pts[1];
        source[base + 2] = pts[2];
        // target = source + (1, 1, 1)
        target[base] = pts[0] + 1.0;
        target[base + 1] = pts[1] + 1.0;
        target[base + 2] = pts[2] + 1.0;
    }
    // Outliers: indices 8, 9 have wildly different target
    for i in 8..10 {
        let base = i * 3;
        source[base] = i as f64;
        source[base + 1] = (i as f64) * 0.5;
        source[base + 2] = (i as f64) * 0.3;
        target[base] = source[base] + 100.0;
        target[base + 1] = source[base + 1] + 100.0;
        target[base + 2] = source[base + 2] + 100.0;
    }

    let mask = ransac_alignment(&source, &target, 10, 1000, 0.5, 3, 42).unwrap();

    // Outliers should be rejected
    assert!(!mask[8]);
    assert!(!mask[9]);
    // Most inliers should be kept
    let inlier_count: usize = mask.iter().filter(|&&b| b).count();
    assert!(inlier_count >= 7);
}

#[test]
fn test_deterministic_with_seed() {
    // Non-collinear 3D points
    #[rustfmt::skip]
    let source: Vec<f64> = vec![
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,  1.0, 1.0, 0.0,  1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,  1.0, 1.0, 1.0,  2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
    ];
    let target: Vec<f64> = source.iter().map(|&x| x + 0.5).collect();

    let mask1 = ransac_alignment(&source, &target, 10, 100, 0.5, 3, 123).unwrap();
    let mask2 = ransac_alignment(&source, &target, 10, 100, 0.5, 3, 123).unwrap();
    assert_eq!(mask1, mask2);
}

#[test]
fn test_point_slice_lengths_are_errors() {
    let six = [0.0; 6];
    let nine = [0.0; 9];
    // Points of 2 values each, as from a (3, 2) array.
    assert_eq!(
        ransac_alignment(&six, &six, 3, 10, 0.5, 3, 0),
        Err(AlignmentError::PointsLength {
            array: "source_points",
            len: 6,
            n_points: 3,
        })
    );
    // A target with fewer points than the source.
    let err = ransac_alignment(&nine, &six, 3, 10, 0.5, 3, 0).unwrap_err();
    assert_eq!(
        err,
        AlignmentError::PointsLength {
            array: "target_points",
            len: 6,
            n_points: 3,
        }
    );
    assert_eq!(
        err.to_string(),
        "target_points has 6 values but 3 points of 3 values each need 9"
    );
}

#[test]
fn test_sample_size_out_of_range_is_error() {
    let six = [0.0; 6];
    for min_sample_size in [0, 3] {
        assert_eq!(
            ransac_alignment(&six, &six, 2, 10, 0.5, min_sample_size, 0),
            Err(AlignmentError::SampleSize {
                min_sample_size,
                n_points: 2,
            })
        );
    }
}
