// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! RANSAC-based outlier rejection for point alignment.
//!
//! Uses Random Sample Consensus to identify inlier correspondences
//! that fit a dominant similarity transformation.

use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;

use super::kabsch::kabsch_algorithm;

/// RANSAC alignment: identify inlier point correspondences.
///
/// Points are passed as flat slices of length `n_points * 3`, stored in
/// row-major order (x0, y0, z0, x1, y1, z1, ...).
///
/// Returns a `Vec<bool>` of length `n_points` where `true` indicates an inlier.
pub fn ransac_alignment(
    source_points: &[f64],
    target_points: &[f64],
    n_points: usize,
    max_iterations: usize,
    threshold: f64,
    min_sample_size: usize,
    seed: u64,
) -> Vec<bool> {
    let mut best_inliers = vec![true; n_points];
    let mut best_inlier_count: usize = 0;
    let mut rng = StdRng::seed_from_u64(seed);

    for _ in 0..max_iterations {
        // Sample random indices without replacement
        let indices = sample(&mut rng, n_points, min_sample_size);

        // Extract sample points into contiguous buffers
        let mut sample_src = vec![0.0f64; min_sample_size * 3];
        let mut sample_tgt = vec![0.0f64; min_sample_size * 3];
        for (j, idx) in indices.iter().enumerate() {
            let src_base = idx * 3;
            let dst_base = j * 3;
            sample_src[dst_base] = source_points[src_base];
            sample_src[dst_base + 1] = source_points[src_base + 1];
            sample_src[dst_base + 2] = source_points[src_base + 2];
            sample_tgt[dst_base] = target_points[src_base];
            sample_tgt[dst_base + 1] = target_points[src_base + 1];
            sample_tgt[dst_base + 2] = target_points[src_base + 2];
        }

        // Estimate transform from sample
        let transform = match kabsch_algorithm(&sample_src, &sample_tgt, min_sample_size) {
            Ok(result) => result,
            Err(_) => continue, // degenerate sample, skip
        };

        // Transform all source points and count inliers
        let rot = transform.rotation.to_rotation_matrix();
        let trans = &transform.translation;
        let scale = transform.scale;
        let mut inlier_count = 0usize;
        let mut inliers = vec![false; n_points];

        for (i, inlier) in inliers.iter_mut().enumerate() {
            let base = i * 3;
            let sx = source_points[base];
            let sy = source_points[base + 1];
            let sz = source_points[base + 2];

            // transformed = scale * (R @ src) + trans
            let tx = scale * (rot[(0, 0)] * sx + rot[(0, 1)] * sy + rot[(0, 2)] * sz) + trans[0];
            let ty = scale * (rot[(1, 0)] * sx + rot[(1, 1)] * sy + rot[(1, 2)] * sz) + trans[1];
            let tz = scale * (rot[(2, 0)] * sx + rot[(2, 1)] * sy + rot[(2, 2)] * sz) + trans[2];

            let dx = tx - target_points[base];
            let dy = ty - target_points[base + 1];
            let dz = tz - target_points[base + 2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < threshold {
                *inlier = true;
                inlier_count += 1;
            }
        }

        if inlier_count > best_inlier_count {
            best_inlier_count = inlier_count;
            best_inliers = inliers;
        }
    }

    best_inliers
}

#[cfg(test)]
mod tests {
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

        let mask = ransac_alignment(&source, &target, n, 100, 0.5, 3, 42);
        let inlier_count: usize = mask.iter().filter(|&&b| b).count();
        assert_eq!(inlier_count, n);
    }

    #[test]
    fn test_known_outliers_rejected() {
        // 8 inliers with pure translation, 2 outliers with big offset
        // Use non-collinear 3D points so Kabsch can find a valid rotation
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

        let mask = ransac_alignment(&source, &target, 10, 1000, 0.5, 3, 42);

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

        let mask1 = ransac_alignment(&source, &target, 10, 100, 0.5, 3, 123);
        let mask2 = ransac_alignment(&source, &target, 10, 100, 0.5, 3, 123);
        assert_eq!(mask1, mask2);
    }
}