// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Sort-and-sweep matching algorithm for feature correspondences.
//!
//! Operates on Y-sorted keypoints with a sliding window, finding the
//! best descriptor match within a local neighborhood.
//!
//! ## Plain and geometric matching share one path
//!
//! Both pairs of public entry points — [`match_one_way_sweep`] /
//! [`match_one_way_sweep_geometric`] and [`mutual_best_match_sweep`] /
//! [`mutual_best_match_sweep_geometric`] — run the *same* sweep. The geometric
//! ones additionally take per-feature affine shapes — carried through the Y
//! sort by the bidirectional pair, supplied already sorted to the one-way pair
//! — and narrow each sliding window to the candidates passing the two-stage
//! geometric filter before any descriptor is compared. That is the whole
//! difference, and it is carried by one `Option` parameter rather than by a
//! parallel family of `_geometric` functions.

use std::collections::HashMap;

use super::descriptor::find_best_match_contiguous;
use super::gather::gather_rows;
use super::geometric_filter::{
    two_stage_geometric_filter, GeometricFilterConfig, StereoPairGeometry,
};

/// Result of a one-way sweep match: maps query index to (target index, distance).
pub type SweepMatches = HashMap<usize, (usize, f64)>;

/// The extra per-feature data the geometric path needs, already in Y order:
/// query-side and candidate-side affine shapes, and the filter to apply.
///
/// Positions are not carried here — the sweep already holds both keypoint
/// arrays, since Y *is* its sort key, and the filter reads its positions from
/// those. A second field naming a position array could later be pointed at a
/// different one than the window slider indexes. (The polar sweep's
/// same-named struct *does* carry positions, because there the sort key is the
/// angle and the positions are genuinely separate data.)
struct GeometricInputs<'a> {
    affines1: &'a [f64],
    affines2: &'a [f64],
    geom: &'a StereoPairGeometry,
    config: &'a GeometricFilterConfig,
}

/// The shared body of both one-way entry points, [`match_one_way_sweep`] and
/// [`match_one_way_sweep_geometric`], which document the parameters.
///
/// `geometric` carries the entire difference between them: `Some` narrows each
/// window to the candidates passing the two-stage orientation/size filter
/// before any descriptor is compared, `None` compares the whole window.
#[allow(clippy::too_many_arguments)]
fn match_one_way_sweep_inner(
    sorted_kpts1: &[f64],
    sorted_descs1: &[u8],
    n1: usize,
    sorted_kpts2: &[f64],
    sorted_descs2: &[u8],
    n2: usize,
    window_size: usize,
    threshold: Option<f64>,
    geometric: Option<&GeometricInputs<'_>>,
) -> SweepMatches {
    let desc_len = sorted_descs1.len().checked_div(n1).unwrap_or(128);

    let mut matches = SweepMatches::new();

    if n2 == 0 {
        return matches;
    }

    let mut win_start: usize = 0;

    // Reused across iterations on the geometric path: the descriptors that
    // survived the filter, and their offsets within the current window.
    let mut passing_descs: Vec<u8> = Vec::new();
    let mut passing_offsets: Vec<usize> = Vec::new();

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

        let query_desc = &sorted_descs1[idx1 * desc_len..(idx1 + 1) * desc_len];

        // The candidate set is either the whole window or the part of it the
        // geometric filter admits. `window_offsets` is `Some` exactly on the
        // geometric path — whether or not the filter actually dropped anything
        // — and maps a candidate back to its offset within the
        // window; deciding it here, in the arm that knows, keeps it from
        // drifting out of step with the descriptors it indexes.
        let (candidate_descs, window_offsets): (&[u8], Option<&[usize]>) = match geometric {
            Some(g) => {
                let x1 = [sorted_kpts1[idx1 * 2], sorted_kpts1[idx1 * 2 + 1]];
                let affine1 = [
                    g.affines1[idx1 * 4],
                    g.affines1[idx1 * 4 + 1],
                    g.affines1[idx1 * 4 + 2],
                    g.affines1[idx1 * 4 + 3],
                ];

                let mask = two_stage_geometric_filter(
                    x1,
                    &affine1,
                    &sorted_kpts2[win_start * 2..win_end * 2],
                    &g.affines2[win_start * 4..win_end * 4],
                    window_len,
                    g.geom,
                    g.config,
                );

                passing_descs.clear();
                passing_offsets.clear();
                for (offset, &passes) in mask.iter().enumerate() {
                    if passes {
                        passing_offsets.push(offset);
                        let start = (win_start + offset) * desc_len;
                        passing_descs.extend_from_slice(&sorted_descs2[start..start + desc_len]);
                    }
                }

                if passing_offsets.is_empty() {
                    continue;
                }
                (&passing_descs, Some(passing_offsets.as_slice()))
            }
            None => (
                &sorted_descs2[win_start * desc_len..win_end * desc_len],
                None,
            ),
        };

        if let Some((rel_idx, dist)) =
            find_best_match_contiguous(query_desc, candidate_descs, desc_len, threshold)
        {
            // `rel_idx` indexes the candidate set, which is the window itself
            // unless the filter narrowed it.
            let offset = match window_offsets {
                Some(offsets) => offsets[rel_idx],
                None => rel_idx,
            };
            matches.insert(idx1, (win_start + offset, dist));
        }
    }

    matches
}

/// The geometric filter's inputs, gathered into Y order once and held for both
/// sweep directions.
struct SortedGeometry<'a> {
    affines1: Vec<f64>,
    affines2: Vec<f64>,
    geom: &'a StereoPairGeometry,
    /// The backward sweep reverses the roles of query and target, so it needs
    /// the geometry swapped to match.
    geom_swapped: StereoPairGeometry,
    config: &'a GeometricFilterConfig,
}

impl SortedGeometry<'_> {
    fn forward(&self) -> GeometricInputs<'_> {
        GeometricInputs {
            affines1: &self.affines1,
            affines2: &self.affines2,
            geom: self.geom,
            config: self.config,
        }
    }

    fn backward(&self) -> GeometricInputs<'_> {
        GeometricInputs {
            affines1: &self.affines2,
            affines2: &self.affines1,
            geom: &self.geom_swapped,
            config: self.config,
        }
    }
}

/// The shared body of both bidirectional entry points,
/// [`mutual_best_match_sweep`] and [`mutual_best_match_sweep_geometric`], which
/// document the parameters.
///
/// `geometric` carries the only difference between them: `Some((affines1,
/// affines2, geom, config))` runs the geometric filter over each window,
/// `None` compares the whole window.
#[allow(clippy::too_many_arguments)]
fn mutual_best_match_sweep_inner(
    keypoints1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    keypoints2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    geometric: Option<(&[f64], &[f64], &StereoPairGeometry, &GeometricFilterConfig)>,
) -> Vec<(usize, usize, f64)> {
    if n1 == 0 || n2 == 0 {
        return Vec::new();
    }

    // Sort both sets by Y coordinate
    let sort_idx1 = argsort_by_y(keypoints1, n1);
    let sort_idx2 = argsort_by_y(keypoints2, n2);

    let sorted_kpts1 = gather_rows(keypoints1, 2, sort_idx1.iter().copied());
    let sorted_descs1 = gather_rows(descriptors1, desc_len, sort_idx1.iter().copied());
    let sorted_kpts2 = gather_rows(keypoints2, 2, sort_idx2.iter().copied());
    let sorted_descs2 = gather_rows(descriptors2, desc_len, sort_idx2.iter().copied());

    let sorted_geometry = geometric.map(|(affines1, affines2, geom, config)| SortedGeometry {
        affines1: gather_rows(affines1, 4, sort_idx1.iter().copied()),
        affines2: gather_rows(affines2, 4, sort_idx2.iter().copied()),
        geom,
        geom_swapped: geom.swapped(),
        config,
    });

    // Forward matching: image1 -> image2
    let forward = match_one_way_sweep_inner(
        &sorted_kpts1,
        &sorted_descs1,
        n1,
        &sorted_kpts2,
        &sorted_descs2,
        n2,
        window_size,
        threshold,
        sorted_geometry
            .as_ref()
            .map(SortedGeometry::forward)
            .as_ref(),
    );

    // Backward matching: image2 -> image1
    let backward = match_one_way_sweep_inner(
        &sorted_kpts2,
        &sorted_descs2,
        n2,
        &sorted_kpts1,
        &sorted_descs1,
        n1,
        window_size,
        threshold,
        sorted_geometry
            .as_ref()
            .map(SortedGeometry::backward)
            .as_ref(),
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
    match_one_way_sweep_inner(
        sorted_kpts1,
        sorted_descs1,
        n1,
        sorted_kpts2,
        sorted_descs2,
        n2,
        window_size,
        threshold,
        None,
    )
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
    match_one_way_sweep_inner(
        sorted_kpts1,
        sorted_descs1,
        n1,
        sorted_kpts2,
        sorted_descs2,
        n2,
        window_size,
        threshold,
        Some(&GeometricInputs {
            affines1: sorted_affines1,
            affines2: sorted_affines2,
            geom,
            config,
        }),
    )
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
    mutual_best_match_sweep_inner(
        keypoints1,
        descriptors1,
        n1,
        keypoints2,
        descriptors2,
        n2,
        desc_len,
        window_size,
        threshold,
        None,
    )
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
    mutual_best_match_sweep_inner(
        keypoints1,
        descriptors1,
        n1,
        keypoints2,
        descriptors2,
        n2,
        desc_len,
        window_size,
        threshold,
        Some((affines1, affines2, geom, config)),
    )
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

#[cfg(test)]
mod tests;
