// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! SIFT descriptor distance computation and best-match search.

/// Compute squared L2 distance between two SIFT descriptors.
///
/// Uses integer arithmetic internally to avoid per-element float conversion.
/// Returns the squared distance (no sqrt) for efficient comparisons.
///
/// # Parameters
///
/// * `a` – First descriptor (typically 128 bytes).
/// * `b` – Second descriptor, must be the same length as `a`.
pub fn descriptor_distance_l2_squared(a: &[u8], b: &[u8]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as i64 - y as i64;
            d * d
        })
        .sum()
}

/// Compute L2 distance between two SIFT descriptors.
///
/// Convenience wrapper that takes the sqrt of the squared distance.
///
/// # Parameters
///
/// * `a` – First descriptor (typically 128 bytes).
/// * `b` – Second descriptor, must be the same length as `a`.
pub fn descriptor_distance_l2(a: &[u8], b: &[u8]) -> f64 {
    (descriptor_distance_l2_squared(a, b) as f64).sqrt()
}

/// Find the best matching descriptor among `candidates` for `query`.
///
/// Scans every candidate, tracking the minimum squared L2 distance to
/// avoid per-candidate sqrt. Only takes sqrt once for the winning match.
///
/// # Parameters
///
/// * `query` – The descriptor to match against.
/// * `candidates` – Slice of descriptor slices to search.
/// * `threshold` – Optional L2 distance ceiling. When `Some(t)`, a match
///   is rejected if its L2 distance exceeds `t`.
///
/// # Returns
///
/// `Some((index, distance))` for the closest match, or `None` if
/// `candidates` is empty or the best distance exceeds `threshold`.
pub fn find_best_match(
    query: &[u8],
    candidates: &[&[u8]],
    threshold: Option<f64>,
) -> Option<(usize, f64)> {
    if candidates.is_empty() {
        return None;
    }

    // Square the threshold so we can compare in integer space without sqrt per iteration.
    let thresh_sq: i64 = match threshold {
        Some(t) => (t * t) as i64,
        None => i64::MAX,
    };

    let mut best_idx = 0;
    let mut best_dist_sq: i64 = i64::MAX;

    for (i, candidate) in candidates.iter().enumerate() {
        let dist_sq = descriptor_distance_l2_squared(query, candidate);
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = i;
        }
    }

    if best_dist_sq > thresh_sq {
        return None;
    }

    let best_dist = (best_dist_sq as f64).sqrt();
    Some((best_idx, best_dist))
}

/// Find the best matching descriptor in a contiguous flat array of candidates.
///
/// Same algorithm as [`find_best_match`], but candidates are packed into a
/// single contiguous slice instead of a `Vec<&[u8]>`. This avoids an extra
/// level of indirection and is the layout used by the sweep matchers.
///
/// # Parameters
///
/// * `query` – The descriptor to match against.
/// * `candidates` – Flat row-major array of `num_candidates * desc_len` bytes.
///   Candidate *i* occupies bytes `[i*desc_len .. (i+1)*desc_len]`.
/// * `desc_len` – Number of bytes per descriptor.
/// * `threshold` – Optional L2 distance ceiling (see [`find_best_match`]).
///
/// # Returns
///
/// `Some((index, distance))` for the closest match, or `None` if
/// `candidates` is empty or the best distance exceeds `threshold`.
pub fn find_best_match_contiguous(
    query: &[u8],
    candidates: &[u8],
    desc_len: usize,
    threshold: Option<f64>,
) -> Option<(usize, f64)> {
    if candidates.is_empty() || desc_len == 0 {
        return None;
    }

    // Square the threshold so we can compare in integer space without sqrt per iteration.
    let thresh_sq: i64 = match threshold {
        Some(t) => (t * t) as i64,
        None => i64::MAX,
    };

    let num_candidates = candidates.len() / desc_len;
    let mut best_idx = 0;
    let mut best_dist_sq: i64 = i64::MAX;

    for i in 0..num_candidates {
        let start = i * desc_len;
        let end = start + desc_len;
        let dist_sq = descriptor_distance_l2_squared(query, &candidates[start..end]);
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = i;
        }
    }

    if best_dist_sq > thresh_sq {
        return None;
    }

    let best_dist = (best_dist_sq as f64).sqrt();
    Some((best_idx, best_dist))
}

/// Batch descriptor matching with deduplication.
///
/// For each query point, examines its candidate target indices, computes
/// descriptor L2 distances, and picks the best match under the threshold.
/// If multiple source features match the same target, keeps the one with
/// the lowest descriptor distance.
///
/// Returns a sorted vector of `[src_idx, dst_idx]` pairs.
///
/// # Parameters
///
/// * `candidates` – Flat row-major `n_queries × k` array of candidate target
///   indices into `desc2`. Empty slots are marked with `u32::MAX`. A typical
///   source is [`KdTree2d::nearest_k_within_radius`](crate::spatial::KdTree2d::nearest_k_within_radius).
/// * `in_bounds_idx` – Maps query index to source feature index (`n_queries` entries).
/// * `desc1` – Flat row-major source descriptors (`n_feat1 * desc_len` bytes).
/// * `desc2` – Flat row-major target descriptors (`n_feat2 * desc_len` bytes).
/// * `n_queries` – Number of query points.
/// * `k` – Number of candidate slots per query.
/// * `desc_len` – Descriptor length in bytes (e.g. 128).
/// * `threshold` – Maximum L2 descriptor distance.
#[allow(clippy::too_many_arguments)]
pub fn match_candidates_and_deduplicate(
    candidates: &[u32],
    in_bounds_idx: &[u32],
    desc1: &[u8],
    desc2: &[u8],
    n_queries: usize,
    k: usize,
    desc_len: usize,
    threshold: f64,
) -> Vec<[u32; 2]> {
    use rayon::prelude::*;
    use std::collections::HashMap;

    let thresh_sq = (threshold * threshold) as i64;

    // Phase 1: parallel best-match search per query point
    let raw_matches: Vec<(u32, u32, i64)> = (0..n_queries)
        .into_par_iter()
        .filter_map(|qi| {
            let src_idx = in_bounds_idx[qi];
            let q_start = src_idx as usize * desc_len;
            let query_desc = &desc1[q_start..q_start + desc_len];
            let cand_row = &candidates[qi * k..(qi + 1) * k];

            let mut best_dst = 0u32;
            let mut best_dist_sq = i64::MAX;

            for &cand_idx in cand_row {
                if cand_idx == u32::MAX {
                    continue;
                }
                let c_start = cand_idx as usize * desc_len;
                let cand_desc = &desc2[c_start..c_start + desc_len];
                let dist_sq = descriptor_distance_l2_squared(query_desc, cand_desc);
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_dst = cand_idx;
                }
            }

            if best_dist_sq <= thresh_sq && best_dist_sq < i64::MAX {
                Some((src_idx, best_dst, best_dist_sq))
            } else {
                None
            }
        })
        .collect();

    // Phase 2: deduplicate — if multiple src map to same dst, keep lowest distance
    let mut best_by_dst: HashMap<u32, (u32, i64)> = HashMap::new();
    for &(src, dst, dist_sq) in &raw_matches {
        best_by_dst
            .entry(dst)
            .and_modify(|entry| {
                if dist_sq < entry.1 {
                    *entry = (src, dist_sq);
                }
            })
            .or_insert((src, dist_sq));
    }

    let mut result: Vec<[u32; 2]> = best_by_dst
        .into_iter()
        .map(|(dst, (src, _))| [src, dst])
        .collect();
    result.sort_unstable_by_key(|pair| (pair[0], pair[1]));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_identical() {
        let a = [42u8; 128];
        assert_eq!(descriptor_distance_l2(&a, &a), 0.0);
        assert_eq!(descriptor_distance_l2_squared(&a, &a), 0);
    }

    #[test]
    fn test_distance_known_values() {
        let a = [0u8; 128];
        let mut b = [0u8; 128];
        b[0] = 3;
        b[1] = 4;
        // sqrt(9 + 16) = 5.0
        assert!((descriptor_distance_l2(&a, &b) - 5.0).abs() < 1e-10);
        assert_eq!(descriptor_distance_l2_squared(&a, &b), 25);
    }

    #[test]
    fn test_distance_symmetric() {
        let a: Vec<u8> = (0..128).collect();
        let b: Vec<u8> = (128..=255).chain(0..1).take(128).collect();
        let d1 = descriptor_distance_l2(&a, &b);
        let d2 = descriptor_distance_l2(&b, &a);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_find_best_match_empty() {
        let query = [0u8; 128];
        let candidates: Vec<&[u8]> = vec![];
        assert!(find_best_match(&query, &candidates, None).is_none());
    }

    #[test]
    fn test_find_best_match_selects_closest() {
        let query = [0u8; 128];
        let far = [100u8; 128];
        let close = [1u8; 128];
        let mid = [50u8; 128];
        let candidates: Vec<&[u8]> = vec![&far, &close, &mid];
        let (idx, _dist) = find_best_match(&query, &candidates, None).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_find_best_match_threshold() {
        let query = [0u8; 128];
        let far = [100u8; 128];
        let candidates: Vec<&[u8]> = vec![&far];
        // Distance is sqrt(100^2 * 128) ≈ 1131.4
        assert!(find_best_match(&query, &candidates, Some(10.0)).is_none());
        assert!(find_best_match(&query, &candidates, Some(2000.0)).is_some());
    }

    #[test]
    fn test_distance_max_values() {
        // All-zero vs all-255: maximum possible distance
        let a = [0u8; 128];
        let b = [255u8; 128];
        // sqrt(255^2 * 128) = 255 * sqrt(128) ≈ 2884.27
        let dist = descriptor_distance_l2(&a, &b);
        let expected = 255.0 * (128.0_f64).sqrt();
        assert!((dist - expected).abs() < 0.01);
        assert_eq!(descriptor_distance_l2_squared(&a, &b), 255_i64 * 255 * 128);
    }

    #[test]
    fn test_find_best_match_multiple_thresholds() {
        let query = [0u8; 128];
        let mut candidate = [0u8; 128];
        candidate[0] = 3;
        candidate[1] = 4;
        // Distance = 5.0
        let candidates: Vec<&[u8]> = vec![&candidate];

        // Test various thresholds around the distance
        assert!(find_best_match(&query, &candidates, Some(4.0)).is_none());
        assert!(find_best_match(&query, &candidates, Some(4.9)).is_none());
        assert!(find_best_match(&query, &candidates, Some(5.0)).is_some());
        assert!(find_best_match(&query, &candidates, Some(100.0)).is_some());
        assert!(find_best_match(&query, &candidates, Some(500.0)).is_some());
        assert!(find_best_match(&query, &candidates, Some(1000.0)).is_some());
    }

    #[test]
    fn test_find_best_match_single_candidate() {
        let query = [42u8; 128];
        let candidate = [42u8; 128];
        let candidates: Vec<&[u8]> = vec![&candidate];
        let (idx, dist) = find_best_match(&query, &candidates, None).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_distance_l2_squared_matches_l2() {
        // Verify that l2 == sqrt(l2_squared) for multiple test vectors
        let test_cases: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (vec![0; 128], vec![0; 128]),
            (vec![0; 128], vec![255; 128]),
            (
                (0..128).collect(),
                (128..=255).chain(0..1).take(128).collect(),
            ),
            (vec![42; 128], vec![43; 128]),
        ];

        for (a, b) in &test_cases {
            let dist_sq = descriptor_distance_l2_squared(a, b);
            let dist = descriptor_distance_l2(a, b);
            assert!(
                (dist - (dist_sq as f64).sqrt()).abs() < 1e-10,
                "l2 and sqrt(l2_squared) should match"
            );
        }
    }

    #[test]
    fn test_find_best_match_contiguous() {
        let query = [0u8; 128];
        let mut candidates = vec![100u8; 128 * 3];
        // Make second candidate closest
        for b in &mut candidates[128..256] {
            *b = 1;
        }
        let (idx, _) = find_best_match_contiguous(&query, &candidates, 128, None).unwrap();
        assert_eq!(idx, 1);
    }
}