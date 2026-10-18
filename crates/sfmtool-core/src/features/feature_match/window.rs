// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The steps the Y sweep and the polar sweep share once each has sorted its
//! features and placed its sliding window.
//!
//! The two sweeps differ in how they order features and slide the window:
//! [`super::sweep`] sorts by Y and reads the window from the sorted keypoints
//! themselves, while [`super::polar`] sorts by angle around the epipole and
//! extends the candidate arrays across the ±π seam. Once a window is placed,
//! both do the same thing with it: optionally narrow it with the geometric
//! filter, take the nearest descriptor among what remains
//! ([`WindowMatcher::best_in_window`]), and after both directions keep the
//! pairs that are each other's nearest ([`mutual_matches`]). Those steps live
//! here so the two sweeps cannot drift apart in them.

use std::collections::HashMap;

use super::descriptor::find_best_match_contiguous;
use super::geometric_filter::{
    two_stage_geometric_filter, GeometricFilterConfig, StereoPairGeometry,
};

/// The geometric filter's inputs for one query feature and one window.
///
/// `positions2` and `affines2` hold exactly the window's rows, in the same
/// order as the window's descriptors.
pub(super) struct WindowFilter<'a> {
    /// The query feature's position.
    pub(super) x1: [f64; 2],
    /// The query feature's affine shape.
    pub(super) affine1: [f64; 4],
    /// The window's positions, flat row-major, 2 per feature.
    pub(super) positions2: &'a [f64],
    /// The window's affine shapes, flat row-major, 4 per feature.
    pub(super) affines2: &'a [f64],
    pub(super) geom: &'a StereoPairGeometry,
    pub(super) config: &'a GeometricFilterConfig,
}

/// Row `idx` of a flat row-major `N × 4` affine-shape array.
pub(super) fn affine_row(affines: &[f64], idx: usize) -> [f64; 4] {
    [
        affines[idx * 4],
        affines[idx * 4 + 1],
        affines[idx * 4 + 2],
        affines[idx * 4 + 3],
    ]
}

/// Finds the nearest descriptor in a window, holding the buffers the
/// geometric path reuses from one query to the next.
#[derive(Default)]
pub(super) struct WindowMatcher {
    /// The descriptors that passed the filter, in window order.
    passing_descs: Vec<u8>,
    /// The offset within the window of each entry in `passing_descs`.
    passing_offsets: Vec<usize>,
}

impl WindowMatcher {
    /// The nearest descriptor to `query_desc` among a window's candidates.
    ///
    /// `window_descs` holds the window's descriptors, `desc_len` bytes each.
    /// With `filter` set, only the candidates passing the two-stage
    /// orientation/size filter are compared; without it, the whole window is.
    /// Candidates are compared in window order either way, so a tie in
    /// distance goes to the earlier one.
    ///
    /// Returns the winner's offset within the window and its distance, or
    /// `None` when no candidate passes the filter or the threshold.
    pub(super) fn best_in_window(
        &mut self,
        query_desc: &[u8],
        window_descs: &[u8],
        desc_len: usize,
        threshold: Option<f64>,
        filter: Option<WindowFilter<'_>>,
    ) -> Option<(usize, f64)> {
        let Some(f) = filter else {
            return find_best_match_contiguous(query_desc, window_descs, desc_len, threshold);
        };

        // Counted from the positions rather than the descriptors, so a zero
        // `desc_len` reaches the search below (which returns `None`) instead
        // of dividing by zero.
        let window_len = f.positions2.len() / 2;
        let mask = two_stage_geometric_filter(
            f.x1,
            &f.affine1,
            f.positions2,
            f.affines2,
            window_len,
            f.geom,
            f.config,
        );

        self.passing_descs.clear();
        self.passing_offsets.clear();
        for (offset, &passes) in mask.iter().enumerate() {
            if passes {
                self.passing_offsets.push(offset);
                let start = offset * desc_len;
                self.passing_descs
                    .extend_from_slice(&window_descs[start..start + desc_len]);
            }
        }

        if self.passing_offsets.is_empty() {
            return None;
        }

        // The search indexes the passing candidates; map its winner back to
        // its offset within the window.
        find_best_match_contiguous(query_desc, &self.passing_descs, desc_len, threshold)
            .map(|(rel_idx, dist)| (self.passing_offsets[rel_idx], dist))
    }
}

/// The pairs each direction's one-way match agrees on, in original indices.
///
/// `forward` maps a sorted index in image 1 to `(sorted index in image 2,
/// distance)`, and `backward` the same from image 2 to image 1. A pair is kept
/// when the backward match of its image-2 feature is the image-1 feature it
/// came from; it is reported with the forward distance, its indices mapped
/// back through `original1` and `original2`.
///
/// The pairs are sorted by their image-1 index, so the output does not depend
/// on the order the maps iterate in, which `HashMap`'s random hasher changes
/// from run to run. The sort has no ties: `forward` has one entry per sorted
/// image-1 index and `original1` maps distinct sorted indices to distinct
/// original ones, so each image-1 index appears at most once.
pub(super) fn mutual_matches(
    forward: &HashMap<usize, (usize, f64)>,
    backward: &HashMap<usize, (usize, f64)>,
    original1: impl Fn(usize) -> usize,
    original2: impl Fn(usize) -> usize,
) -> Vec<(usize, usize, f64)> {
    let mut mutual = Vec::new();
    for (&s_idx1, &(s_idx2, dist)) in forward {
        if let Some(&(back_idx1, _)) = backward.get(&s_idx2) {
            if back_idx1 == s_idx1 {
                mutual.push((original1(s_idx1), original2(s_idx2), dist));
            }
        }
    }
    mutual.sort_unstable_by_key(|&(idx1, _, _)| idx1);
    mutual
}

#[cfg(test)]
mod tests;
