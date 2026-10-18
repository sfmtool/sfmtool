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
//!
//! ## Sort positions and filter positions
//!
//! The geometric filter unprojects positions through the cameras in its
//! [`StereoPairGeometry`], so it must read them in those cameras' pixels. The
//! public `_geometric` functions read the filter positions from the keypoints
//! they sort by, so the geometry has to describe the cameras those keypoints
//! are pixels of. [`super::match_image_pair`] sorts by rectified keypoints but
//! holds the geometry of the original cameras, so it calls
//! `mutual_best_match_sweep_inner` with the raw positions as separate filter
//! positions, carried through the Y sort with the other per-feature arrays.

use std::collections::HashMap;

use super::gather::gather_rows;
use super::geometric_filter::{GeometricFilterConfig, StereoPairGeometry};
use super::window::{affine_row, mutual_matches, WindowFilter, WindowMatcher};

/// Result of a one-way sweep match: maps query index to (target index, distance).
pub type SweepMatches = HashMap<usize, (usize, f64)>;

/// The extra per-feature data the geometric path needs, already in Y order:
/// query-side and candidate-side filter positions and affine shapes, and the
/// filter to apply.
///
/// The filter positions are the keypoints the sweep sorts by on the public
/// entry points, and the raw pixel positions when [`super::match_image_pair`]
/// sorts by rectified keypoints (see the module doc).
struct GeometricInputs<'a> {
    positions1: &'a [f64],
    positions2: &'a [f64],
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
    let mut window_matcher = WindowMatcher::default();

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
        let filter = geometric.map(|g| WindowFilter {
            x1: [g.positions1[idx1 * 2], g.positions1[idx1 * 2 + 1]],
            affine1: affine_row(g.affines1, idx1),
            positions2: &g.positions2[win_start * 2..win_end * 2],
            affines2: &g.affines2[win_start * 4..win_end * 4],
            geom: g.geom,
            config: g.config,
        });

        if let Some((offset, dist)) = window_matcher.best_in_window(
            query_desc,
            &sorted_descs2[win_start * desc_len..win_end * desc_len],
            desc_len,
            threshold,
            filter,
        ) {
            matches.insert(idx1, (win_start + offset, dist));
        }
    }

    matches
}

/// The geometric filter's inputs, gathered into Y order once and held for both
/// sweep directions.
struct SortedGeometry<'a> {
    positions1: Vec<f64>,
    positions2: Vec<f64>,
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
            positions1: &self.positions1,
            positions2: &self.positions2,
            affines1: &self.affines1,
            affines2: &self.affines2,
            geom: self.geom,
            config: self.config,
        }
    }

    fn backward(&self) -> GeometricInputs<'_> {
        GeometricInputs {
            positions1: &self.positions2,
            positions2: &self.positions1,
            affines1: &self.affines2,
            affines2: &self.affines1,
            geom: &self.geom_swapped,
            config: self.config,
        }
    }
}

/// The geometric filter's per-feature inputs for a bidirectional sweep, in the
/// caller's feature order: the positions the filter reads (`N × 2` and
/// `M × 2`), the affine shapes (`N × 4` and `M × 4`), the camera geometry and
/// the filter settings.
///
/// `positions1` and `positions2` must be pixel positions in the cameras `geom`
/// describes. They may differ from the keypoints the sweep sorts by, as on
/// [`super::match_image_pair`]'s rectified path, where the sweep sorts by
/// rectified keypoints and the filter reads the raw ones.
pub(super) struct SweepFilterInputs<'a> {
    pub(super) positions1: &'a [f64],
    pub(super) positions2: &'a [f64],
    pub(super) affines1: &'a [f64],
    pub(super) affines2: &'a [f64],
    pub(super) geom: &'a StereoPairGeometry,
    pub(super) config: &'a GeometricFilterConfig,
}

/// The shared body of both bidirectional entry points,
/// [`mutual_best_match_sweep`] and [`mutual_best_match_sweep_geometric`], which
/// document the parameters.
///
/// `geometric` carries the only difference between them: `Some` runs the
/// geometric filter over each window, `None` compares the whole window.
#[allow(clippy::too_many_arguments)]
pub(super) fn mutual_best_match_sweep_inner(
    keypoints1: &[f64],
    descriptors1: &[u8],
    n1: usize,
    keypoints2: &[f64],
    descriptors2: &[u8],
    n2: usize,
    desc_len: usize,
    window_size: usize,
    threshold: Option<f64>,
    geometric: Option<SweepFilterInputs<'_>>,
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

    let sorted_geometry = geometric.map(|g| SortedGeometry {
        positions1: gather_rows(g.positions1, 2, sort_idx1.iter().copied()),
        positions2: gather_rows(g.positions2, 2, sort_idx2.iter().copied()),
        affines1: gather_rows(g.affines1, 4, sort_idx1.iter().copied()),
        affines2: gather_rows(g.affines2, 4, sort_idx2.iter().copied()),
        geom: g.geom,
        geom_swapped: g.geom.swapped(),
        config: g.config,
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

    mutual_matches(&forward, &backward, |s| sort_idx1[s], |s| sort_idx2[s])
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
            positions1: sorted_kpts1,
            positions2: sorted_kpts2,
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
/// other's best match, sorted by `orig_idx1`.
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
/// other's best match, sorted by `orig_idx1`.
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
        Some(SweepFilterInputs {
            positions1: keypoints1,
            positions2: keypoints2,
            affines1,
            affines2,
            geom,
            config,
        }),
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
