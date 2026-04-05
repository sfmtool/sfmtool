// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Sort-and-sweep matching algorithm for feature correspondences.
//!
//! Operates on Y-sorted keypoints with a sliding window, finding the
//! best descriptor match within a local neighborhood.

use std::collections::HashMap;

use super::descriptor::find_best_match_contiguous;
use super::geometric_filter::{
    two_stage_geometric_filter, GeometricFilterConfig, StereoPairGeometry,
};

/// Result of a one-way sweep match: maps query index to (target index, distance).
pub type SweepMatches = HashMap<usize, (usize, f64)>;

/// One-way Y-sweep nearest-neighbor match on pre-sorted features.
///
/// Walks through features in image 1 (sorted by Y) and, for each one,
/// finds the best descriptor match among the `window_size` closest
/// features in image 2 by Y coordinate. A sliding window keeps this
/// linear in the number of features.
///
/// # Parameters
///
/// * `sorted_kpts1` – Keypoints in image 1, flat row-major Nx2, sorted by Y.
/// * `sorted_descs1` – Descriptors for image 1, flat row-major Nx(desc_len),
///   in the same sorted order as `sorted_kpts1`.
/// * `n1` – Number of features in image 1.
/// * `sorted_kpts2` – Keypoints in image 2, flat row-major Mx2, sorted by Y.
/// * `sorted_descs2` – Descriptors for image 2, flat row-major Mx(desc_len),
///   in the same sorted order as `sorted_kpts2`.
/// * `n2` – Number of features in image 2.
/// * `window_size` – Number of Y-neighbors to consider from image 2
///   for each feature in image 1.
/// * `threshold` – Optional L2 distance ceiling. When `Some(t)`, a match
///   is rejected if its descriptor distance exceeds `t`.
///
/// # Returns
///
/// A map from sorted index in image 1 to `(sorted_index_in_image_2, distance)`.
#[allow(clippy::too_many_arguments)]
pub fn match_one_way_sweep(
    sorted_kpts1: &[f64],
    sorted_descs1: &[u8],
    n1: usize,
    sorted_kpts2: &[f64],
    sorted_descs2: &[u8],
    n2: usize,
    window_size: usize,
    threshold: Option<f64>,
) -> SweepMatches {
    let desc_len = if n1 > 0 {
        sorted_descs1.len() / n1
    } else {
        128
    };

    let mut matches = SweepMatches::new();

    if n2 == 0 {
        return matches;
    }

    let mut win_start: usize = 0;

    for idx1 in 0..n1 {
        let query_y = sorted_kpts1[idx1 * 2 + 1];

        // Slide the window forward
        while win_start + window_size < n2 {
            let next_y = sorted_kpts2[(win_start + window_size) * 2 + 1];
            let start_y = sorted_kpts2[win_start * 2 + 1];
            if (next_y - query_y).abs() < (start_y - query_y).abs() {
                win_start += 1;
            } else {
                break;
            }
        }

        let win_end = (win_start + window_size).min(n2);
        let window_len = win_end - win_start;

        if window_len == 0 {
            continue;
        }

        // Slice the descriptors for the window
        let query_desc = &sorted_descs1[idx1 * desc_len..(idx1 + 1) * desc_len];
        let window_descs = &sorted_descs2[win_start * desc_len..win_end * desc_len];

        if let Some((rel_idx, dist)) =
            find_best_match_contiguous(query_desc, window_descs, desc_len, threshold)
        {
            matches.insert(idx1, (win_start + rel_idx, dist));
        }
    }

    matches
}

/// Full bidirectional Y-sweep matching with mutual consistency check.
///
/// Takes *unsorted* keypoints and descriptors, sorts both sets by Y
/// internally, runs forward (1→2) and backward (2→1) one-way sweeps,
/// keeps only mutual best matches, and maps results back to the
/// caller's original feature indices.
///
/// # Parameters
///
/// * `keypoints1` – Feature positions in image 1, flat row-major Nx2.
/// * `descriptors1` – Descriptors for image 1, flat row-major Nx(desc_len).
/// * `n1` – Number of features in image 1.
/// * `keypoints2` – Feature positions in image 2, flat row-major Mx2.
/// * `descriptors2` – Descriptors for image 2, flat row-major Mx(desc_len).
/// * `n2` – Number of features in image 2.
/// * `desc_len` – Number of bytes per descriptor (typically 128).
/// * `window_size` – Number of Y-neighbors to consider in each direction.
/// * `threshold` – Optional L2 distance ceiling.
///
/// # Returns
///
/// `Vec<(orig_idx1, orig_idx2, distance)>` for every pair that is each
/// other's best match.
#[allow(clippy::too_many_arguments)]
pub fn mutual_best_match_sweep(
    keypoints1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    keypoints2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
) -> Vec<(usize, usize, f64)> {
    if n1 == 0 || n2 == 0 {
        return Vec::new();
    }

    // Sort both sets by Y coordinate
    let sort_idx1 = argsort_by_y(keypoints1, n1);
    let sort_idx2 = argsort_by_y(keypoints2, n2);

    let sorted_kpts1 = gather_2d(keypoints1, &sort_idx1, 2);
    let sorted_descs1 = gather_2d(descriptors1, &sort_idx1, desc_len);
    let sorted_kpts2 = gather_2d(keypoints2, &sort_idx2, 2);
    let sorted_descs2 = gather_2d(descriptors2, &sort_idx2, desc_len);

    // Forward matching: image1 -> image2
    let forward = match_one_way_sweep(
        &sorted_kpts1,
        &sorted_descs1,
        n1,
        &sorted_kpts2,
        &sorted_descs2,
        n2,
        window_size,
        threshold,
    );

    // Backward matching: image2 -> image1
    let backward = match_one_way_sweep(
        &sorted_kpts2,
        &sorted_descs2,
        n2,
        &sorted_kpts1,
        &sorted_descs1,
        n1,
        window_size,
        threshold,
    );

    // Find mutual matches and map back to original indices
    let mut mutual = Vec::new();
    for (&s_idx1, &(s_idx2, dist)) in &forward {
        if let Some(&(back_idx1, _)) = backward.get(&s_idx2) {
            if back_idx1 == s_idx1 {
                let orig_idx1 = sort_idx1[s_idx1];
                let orig_idx2 = sort_idx2[s_idx2];
                mutual.push((orig_idx1, orig_idx2, dist));
            }
        }
    }

    mutual
}

/// One-way Y-sweep nearest-neighbor match with geometric filtering on pre-sorted features.
///
/// Like [`match_one_way_sweep`] but applies a two-stage geometric filter
/// (orientation + size consistency) to the sliding window *before* descriptor
/// comparison. Only candidates that pass the geometric check are considered
/// for the best descriptor match.
///
/// # Parameters
///
/// * `sorted_kpts1` – Keypoints in image 1, flat row-major Nx2, sorted by Y.
/// * `sorted_descs1` – Descriptors for image 1, flat row-major Nx(desc_len).
/// * `n1` – Number of features in image 1.
/// * `sorted_kpts2` – Keypoints in image 2, flat row-major Mx2, sorted by Y.
/// * `sorted_descs2` – Descriptors for image 2, flat row-major Mx(desc_len).
/// * `n2` – Number of features in image 2.
/// * `sorted_affines1` – Affine shapes for image 1, flat row-major Nx4.
/// * `sorted_affines2` – Affine shapes for image 2, flat row-major Mx4.
/// * `window_size` – Number of Y-neighbors to consider from image 2.
/// * `threshold` – Optional L2 distance ceiling.
/// * `geom` – Precomputed camera geometry.
/// * `config` – Geometric filter configuration.
///
/// # Returns
///
/// A map from sorted index in image 1 to `(sorted_index_in_image_2, distance)`.
#[allow(clippy::too_many_arguments)]
pub fn match_one_way_sweep_geometric(
    sorted_kpts1: &[f64],
    sorted_descs1: &[u8],
    n1: usize,
    sorted_kpts2: &[f64],
    sorted_descs2: &[u8],
    n2: usize,
    sorted_affines1: &[f64],
    sorted_affines2: &[f64],
    window_size: usize,
    threshold: Option<f64>,
    geom: &StereoPairGeometry,
    config: &GeometricFilterConfig,
) -> SweepMatches {
    let desc_len = if n1 > 0 {
        sorted_descs1.len() / n1
    } else {
        128
    };

    let mut matches = SweepMatches::new();

    if n2 == 0 {
        return matches;
    }

    let mut win_start: usize = 0;
    let mut filtered_descs_buf: Vec<u8> = Vec::new();

    for idx1 in 0..n1 {
        let query_y = sorted_kpts1[idx1 * 2 + 1];

        // Slide the window forward
        while win_start + window_size < n2 {
            let next_y = sorted_kpts2[(win_start + window_size) * 2 + 1];
            let start_y = sorted_kpts2[win_start * 2 + 1];
            if (next_y - query_y).abs() < (start_y - query_y).abs() {
                win_start += 1;
            } else {
                break;
            }
        }

        let win_end = (win_start + window_size).min(n2);
        let window_len = win_end - win_start;

        if window_len == 0 {
            continue;
        }

        // Extract query point and affine
        let x1 = [sorted_kpts1[idx1 * 2], sorted_kpts1[idx1 * 2 + 1]];
        let affine1 = [
            sorted_affines1[idx1 * 4],
            sorted_affines1[idx1 * 4 + 1],
            sorted_affines1[idx1 * 4 + 2],
            sorted_affines1[idx1 * 4 + 3],
        ];

        // Window positions and affines
        let window_positions = &sorted_kpts2[win_start * 2..win_end * 2];
        let window_affines = &sorted_affines2[win_start * 4..win_end * 4];

        // Apply geometric filter
        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            window_positions,
            window_affines,
            window_len,
            geom,
            config,
        );

        // Gather filtered descriptors into contiguous buffer
        filtered_descs_buf.clear();
        let mut valid_indices: Vec<usize> = Vec::new();
        for (i, &passes) in mask.iter().enumerate() {
            if passes {
                valid_indices.push(i);
                let src_idx = win_start + i;
                let src_start = src_idx * desc_len;
                filtered_descs_buf
                    .extend_from_slice(&sorted_descs2[src_start..src_start + desc_len]);
            }
        }

        if valid_indices.is_empty() {
            continue;
        }

        let query_desc = &sorted_descs1[idx1 * desc_len..(idx1 + 1) * desc_len];

        if let Some((rel_idx, dist)) =
            find_best_match_contiguous(query_desc, &filtered_descs_buf, desc_len, threshold)
        {
            // Map back: rel_idx -> valid_indices[rel_idx] -> window index -> sorted index
            let window_idx = valid_indices[rel_idx];
            matches.insert(idx1, (win_start + window_idx, dist));
        }
    }

    matches
}

/// Full bidirectional Y-sweep matching with geometric filtering and mutual consistency.
///
/// Like [`mutual_best_match_sweep`] but applies two-stage geometric filtering
/// before descriptor comparison in both directions.
///
/// # Parameters
///
/// * `keypoints1` – Feature positions in image 1, flat row-major Nx2.
/// * `descriptors1` – Descriptors for image 1, flat row-major Nx(desc_len).
/// * `n1` – Number of features in image 1.
/// * `keypoints2` – Feature positions in image 2, flat row-major Mx2.
/// * `descriptors2` – Descriptors for image 2, flat row-major Mx(desc_len).
/// * `n2` – Number of features in image 2.
/// * `affines1` – Affine shapes for image 1, flat row-major Nx4.
/// * `affines2` – Affine shapes for image 2, flat row-major Mx4.
/// * `desc_len` – Number of bytes per descriptor (typically 128).
/// * `window_size` – Number of Y-neighbors to consider in each direction.
/// * `threshold` – Optional L2 distance ceiling.
/// * `geom` – Precomputed camera geometry.
/// * `config` – Geometric filter configuration.
///
/// # Returns
///
/// `Vec<(orig_idx1, orig_idx2, distance)>` for every pair that is each
/// other's best match.
#[allow(clippy::too_many_arguments)]
pub fn mutual_best_match_sweep_geometric(
    keypoints1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    keypoints2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    affines1: &[f64],
    affines2: &[f64],
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    geom: &StereoPairGeometry,
    config: &GeometricFilterConfig,
) -> Vec<(usize, usize, f64)> {
    if n1 == 0 || n2 == 0 {
        return Vec::new();
    }

    // Sort both sets by Y coordinate
    let sort_idx1 = argsort_by_y(keypoints1, n1);
    let sort_idx2 = argsort_by_y(keypoints2, n2);

    let sorted_kpts1 = gather_2d(keypoints1, &sort_idx1, 2);
    let sorted_descs1 = gather_2d(descriptors1, &sort_idx1, desc_len);
    let sorted_affines1 = gather_2d(affines1, &sort_idx1, 4);
    let sorted_kpts2 = gather_2d(keypoints2, &sort_idx2, 2);
    let sorted_descs2 = gather_2d(descriptors2, &sort_idx2, desc_len);
    let sorted_affines2 = gather_2d(affines2, &sort_idx2, 4);

    // Forward matching: image1 -> image2
    let forward = match_one_way_sweep_geometric(
        &sorted_kpts1,
        &sorted_descs1,
        n1,
        &sorted_kpts2,
        &sorted_descs2,
        n2,
        &sorted_affines1,
        &sorted_affines2,
        window_size,
        threshold,
        geom,
        config,
    );

    // Backward matching: image2 -> image1 (swap camera geometry)
    let geom_swapped = geom.swapped();
    let backward = match_one_way_sweep_geometric(
        &sorted_kpts2,
        &sorted_descs2,
        n2,
        &sorted_kpts1,
        &sorted_descs1,
        n1,
        &sorted_affines2,
        &sorted_affines1,
        window_size,
        threshold,
        &geom_swapped,
        config,
    );

    // Find mutual matches and map back to original indices
    let mut mutual = Vec::new();
    for (&s_idx1, &(s_idx2, dist)) in &forward {
        if let Some(&(back_idx1, _)) = backward.get(&s_idx2) {
            if back_idx1 == s_idx1 {
                let orig_idx1 = sort_idx1[s_idx1];
                let orig_idx2 = sort_idx2[s_idx2];
                mutual.push((orig_idx1, orig_idx2, dist));
            }
        }
    }

    mutual
}

/// Return indices that would sort an Nx2 row-major keypoint array by Y,
/// using X as a tiebreaker for deterministic ordering.
fn argsort_by_y(kpts: &[f64], n: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let ya = kpts[a * 2 + 1];
        let yb = kpts[b * 2 + 1];
        match ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => {
                let xa = kpts[a * 2];
                let xb = kpts[b * 2];
                xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal)
            }
            ord => ord,
        }
    });
    indices
}

/// Reorder rows of a flat row-major array according to `indices`.
///
/// Each row is `cols` elements wide. Returns a new flat array with the
/// rows in the order specified by `indices`.
fn gather_2d<T: Copy>(data: &[T], indices: &[usize], cols: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(indices.len() * cols);
    for &idx in indices {
        let start = idx * cols;
        result.extend_from_slice(&data[start..start + cols]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argsort_by_y() {
        // 3 points: (0, 10), (0, 5), (0, 15)
        let kpts = [0.0, 10.0, 0.0, 5.0, 0.0, 15.0];
        let indices = argsort_by_y(&kpts, 3);
        assert_eq!(indices, vec![1, 0, 2]); // sorted by Y: 5, 10, 15
    }

    #[test]
    fn test_one_way_sweep_basic() {
        // 3 features in image 1, 3 in image 2
        // Sorted by Y already
        let kpts1 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let kpts2 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];

        // Descriptors: make feature 0 in img1 closest to feature 0 in img2, etc.
        let mut descs1 = vec![0u8; 3 * 128];
        let mut descs2 = vec![0u8; 3 * 128];
        // Feature 0: descs1[0] = [1, 0, ...], descs2[0] = [1, 0, ...]
        descs1[0] = 1;
        descs2[0] = 1;
        // Feature 1: descs1[128] = [0, 2, ...], descs2[128] = [0, 2, ...]
        descs1[129] = 2;
        descs2[129] = 2;
        // Feature 2: descs1[256] = [0, 0, 3, ...], descs2[256] = [0, 0, 3, ...]
        descs1[258] = 3;
        descs2[258] = 3;

        let matches = match_one_way_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 3, None);

        assert_eq!(matches.len(), 3);
        assert_eq!(matches[&0].0, 0);
        assert_eq!(matches[&1].0, 1);
        assert_eq!(matches[&2].0, 2);
    }

    #[test]
    fn test_mutual_match_basic() {
        // Two sets of identical features
        let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
        let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

        let mut descs1 = vec![0u8; 3 * 128];
        let mut descs2 = vec![0u8; 3 * 128];
        descs1[0] = 10;
        descs2[0] = 10;
        descs1[129] = 20;
        descs2[129] = 20;
        descs1[258] = 30;
        descs2[258] = 30;

        let mutual = mutual_best_match_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 128, 3, None);

        assert_eq!(mutual.len(), 3);
        // All should be identity matches
        for (idx1, idx2, dist) in &mutual {
            assert_eq!(idx1, idx2);
            assert_eq!(*dist, 0.0);
        }
    }

    #[test]
    fn test_empty_inputs() {
        let matches = match_one_way_sweep(&[], &[], 0, &[0.0, 1.0], &[0u8; 128], 1, 30, None);
        assert!(matches.is_empty());

        let matches = match_one_way_sweep(&[0.0, 1.0], &[0u8; 128], 1, &[], &[], 0, 30, None);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_mutual_match_geometric_basic() {
        use nalgebra::{Matrix3, Vector3};

        // Same setup as test_mutual_match_basic but with identity affines and identity cameras
        let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
        let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

        let mut descs1 = vec![0u8; 3 * 128];
        let mut descs2 = vec![0u8; 3 * 128];
        descs1[0] = 10;
        descs2[0] = 10;
        descs1[129] = 20;
        descs2[129] = 20;
        descs1[258] = 30;
        descs2[258] = 30;

        // Identity affines for all features
        let affines1 = [
            1.0, 0.0, 0.0, 1.0, // feature 0
            1.0, 0.0, 0.0, 1.0, // feature 1
            1.0, 0.0, 0.0, 1.0, // feature 2
        ];
        let affines2 = [
            1.0, 0.0, 0.0, 1.0, // feature 0
            1.0, 0.0, 0.0, 1.0, // feature 1
            1.0, 0.0, 0.0, 1.0, // feature 2
        ];

        // Identity cameras (same K, identity R, zero t)
        let mut k = Matrix3::identity();
        k[(0, 0)] = 500.0;
        k[(1, 1)] = 500.0;
        k[(0, 2)] = 320.0;
        k[(1, 2)] = 240.0;
        let r = Matrix3::identity();
        let t = Vector3::zeros();
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t, &t);
        let config = GeometricFilterConfig::default();

        let mutual = mutual_best_match_sweep_geometric(
            &kpts1, &descs1, 3, &kpts2, &descs2, 3, &affines1, &affines2, 128, 3, None, &geom,
            &config,
        );

        assert_eq!(mutual.len(), 3);
        // All should be identity matches
        for (idx1, idx2, dist) in &mutual {
            assert_eq!(idx1, idx2);
            assert_eq!(*dist, 0.0);
        }
    }

    #[test]
    fn test_one_way_sweep_with_threshold_rejects() {
        // 1 feature each, with descriptors far apart
        let kpts1 = [0.0, 1.0];
        let kpts2 = [0.0, 1.0];

        let descs1 = vec![0u8; 128];
        let descs2 = vec![100u8; 128];
        // Distance = sqrt(100^2 * 128) ≈ 1131.4

        // Tight threshold: should reject
        let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(10.0));
        assert!(
            matches.is_empty(),
            "Tight threshold should reject distant descriptors"
        );

        // Generous threshold: should accept
        let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(2000.0));
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_one_way_sweep_with_threshold_accepts() {
        // With a generous threshold, all matches should pass
        let kpts1 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let kpts2 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];

        let mut descs1 = vec![0u8; 3 * 128];
        let mut descs2 = vec![0u8; 3 * 128];
        descs1[0] = 1;
        descs2[0] = 1;
        descs1[129] = 2;
        descs2[129] = 2;
        descs1[258] = 3;
        descs2[258] = 3;

        let matches = match_one_way_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 3, Some(2000.0));
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_one_way_sweep_multiple_thresholds() {
        let kpts1 = [0.0, 1.0];
        let kpts2 = [0.0, 1.0];

        let descs1 = vec![0u8; 128];
        let mut descs2 = vec![0u8; 128];
        descs2[0] = 3;
        descs2[1] = 4;
        // Distance = sqrt(9 + 16) = 5.0

        // Threshold below distance: no match
        let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(4.0));
        assert!(matches.is_empty());

        // Threshold at distance: accepted (<=)
        let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(5.0));
        assert_eq!(matches.len(), 1);

        // Threshold above distance: accepted
        let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(100.0));
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_asymmetric_feature_counts() {
        // 5 features in img1, 3 in img2
        let n1 = 5;
        let n2 = 3;
        let mut kpts1 = vec![0.0; n1 * 2];
        let mut kpts2 = vec![0.0; n2 * 2];
        for i in 0..n1 {
            kpts1[i * 2 + 1] = i as f64;
        }
        for i in 0..n2 {
            kpts2[i * 2 + 1] = i as f64;
        }

        let mut descs1 = vec![0u8; n1 * 128];
        let mut descs2 = vec![0u8; n2 * 128];
        // Make each descriptor unique
        for i in 0..n1 {
            descs1[i * 128] = (i * 10) as u8;
        }
        for i in 0..n2 {
            descs2[i * 128] = (i * 10) as u8;
        }

        let matches = match_one_way_sweep(&kpts1, &descs1, n1, &kpts2, &descs2, n2, 5, None);
        // First 3 features in img1 should match perfectly to img2 features
        assert!(matches.len() >= 3);
        for i in 0..3 {
            assert_eq!(matches[&i].0, i);
            assert_eq!(matches[&i].1, 0.0);
        }
    }

    #[test]
    fn test_mutual_match_asymmetric() {
        // 4 features in img1, 6 in img2
        let n1 = 4;
        let n2 = 6;
        let mut kpts1 = vec![0.0; n1 * 2];
        let mut kpts2 = vec![0.0; n2 * 2];
        for i in 0..n1 {
            kpts1[i * 2 + 1] = i as f64;
        }
        for i in 0..n2 {
            kpts2[i * 2 + 1] = i as f64;
        }

        let mut descs1 = vec![0u8; n1 * 128];
        let mut descs2 = vec![0u8; n2 * 128];
        // First n1 features match between the two sets
        for i in 0..n1 {
            descs1[i * 128] = (i * 20 + 10) as u8;
            descs2[i * 128] = (i * 20 + 10) as u8;
        }
        // Extra features in img2 are distinct
        for i in n1..n2 {
            descs2[i * 128] = 200;
            descs2[i * 128 + 1] = (i * 30) as u8;
        }

        let mutual =
            mutual_best_match_sweep(&kpts1, &descs1, n1, &kpts2, &descs2, n2, 128, 10, None);

        assert_eq!(mutual.len(), n1);
        for (idx1, idx2, dist) in &mutual {
            assert_eq!(idx1, idx2);
            assert_eq!(*dist, 0.0);
        }
    }

    #[test]
    fn test_mutual_match_with_threshold() {
        let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
        let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

        // Feature 0: identical (dist=0), Feature 1: close (dist=5), Feature 2: far (dist≈1131)
        let mut descs1 = vec![0u8; 3 * 128];
        let mut descs2 = vec![0u8; 3 * 128];
        descs1[0] = 10;
        descs2[0] = 10;
        descs1[129] = 20;
        descs2[129] = 23; // diff = 3
        descs2[130] = 4; // diff = 4, distance = 5
        descs1[258] = 30;
        descs2[256..384].fill(100); // very far

        // Threshold that accepts feature 0 and 1, rejects feature 2
        let mutual =
            mutual_best_match_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 128, 3, Some(10.0));
        assert_eq!(mutual.len(), 2);
    }

    #[test]
    fn test_larger_window_sizes() {
        // 10 features, test window sizes 5, 10, 20
        let n = 10;
        let mut kpts = vec![0.0; n * 2];
        for i in 0..n {
            kpts[i * 2 + 1] = (i * 10) as f64;
        }
        let mut descs = vec![0u8; n * 128];
        for i in 0..n {
            descs[i * 128] = (i * 15) as u8;
        }

        for window in [5, 10, 20] {
            let mutual =
                mutual_best_match_sweep(&kpts, &descs, n, &kpts, &descs, n, 128, window, None);
            // All features should match themselves regardless of window size
            assert_eq!(
                mutual.len(),
                n,
                "Window size {window}: expected {n} matches, got {}",
                mutual.len()
            );
            for (idx1, idx2, dist) in &mutual {
                assert_eq!(idx1, idx2);
                assert_eq!(*dist, 0.0);
            }
        }
    }

    #[test]
    fn test_match_one_way_sweep_geometric_rejects_bad_orientation() {
        use nalgebra::{Matrix3, Vector3};

        // 3 features in image 1, 3 in image 2, sorted by Y
        let kpts1 = [320.0, 1.0, 320.0, 2.0, 320.0, 3.0];
        let kpts2 = [320.0, 1.0, 320.0, 2.0, 320.0, 3.0];

        // Make all descriptors identical so descriptor matching always succeeds
        let descs1 = vec![1u8; 3 * 128];
        let descs2 = vec![1u8; 3 * 128];

        // Query affines: identity orientation
        let affines1 = [
            5.0, 0.0, 0.0, 5.0, // feature 0
            5.0, 0.0, 0.0, 5.0, // feature 1
            5.0, 0.0, 0.0, 5.0, // feature 2
        ];
        // Target affines: perpendicular orientation (first col rotated 90 degrees)
        let affines2 = [
            0.0, 5.0, 5.0, 0.0, // feature 0: first col = (0, 5), perpendicular
            0.0, 5.0, 5.0, 0.0, // feature 1
            0.0, 5.0, 5.0, 0.0, // feature 2
        ];

        let mut k = Matrix3::identity();
        k[(0, 0)] = 500.0;
        k[(1, 1)] = 500.0;
        k[(0, 2)] = 320.0;
        k[(1, 2)] = 240.0;
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
        let config = GeometricFilterConfig::default();

        let matches = match_one_way_sweep_geometric(
            &kpts1, &descs1, 3, &kpts2, &descs2, 3, &affines1, &affines2, 3, None, &geom, &config,
        );

        // All candidates should be rejected due to bad orientation
        assert!(
            matches.is_empty(),
            "All matches should be rejected due to perpendicular orientation"
        );
    }
}
