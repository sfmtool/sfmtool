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
mod tests;
