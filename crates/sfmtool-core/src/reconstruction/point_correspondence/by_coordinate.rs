// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point correspondence by keypoint position rather than by feature index.
//!
//! [`find_point_correspondences`](super::find_point_correspondences) needs the
//! two reconstructions to name the same `.sift` feature with the same index. An
//! `embedded_patches` reconstruction has no feature index, and two
//! reconstructions built with different feature extractors number the same
//! keypoint differently, yet in both cases the same scene feature lands at
//! nearly the same pixel of a shared image. This module matches on that pixel.

use std::collections::{HashMap, HashSet};

use super::PointCorrespondenceResult;

/// How [`find_point_correspondences_by_coordinate`] decides that two
/// observations of a shared image are the same keypoint, and how much
/// agreement a point pair needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordinateCorrespondenceOptions {
    /// Largest distance in pixels between a source and a target keypoint that
    /// still counts as a match. The comparison is `distance <= pixel_threshold`.
    pub pixel_threshold: f64,
    /// Fewest matched observations, one vote each, that a
    /// `(source point, target point)` pair needs to be kept. A point pair
    /// matched in only one image is weak evidence, since two unrelated points
    /// can project close together in one view.
    pub min_votes: usize,
}

impl Default for CoordinateCorrespondenceOptions {
    /// 2 px and 2 votes, the defaults of `sfm compare --by-coordinate`.
    fn default() -> Self {
        Self {
            pixel_threshold: 2.0,
            min_votes: 2,
        }
    }
}

/// Find corresponding 3D points between two reconstructions by matching the
/// pixel positions of their observations in shared images.
///
/// For each shared image pair, the source observations and the target
/// observations of that image are matched by **mutual nearest keypoint**: an
/// observation pair matches when each is the other's nearest keypoint and they
/// are at most `options.pixel_threshold` pixels apart. Each match casts one vote
/// for the `(source point, target point)` pair its two observations belong to.
/// The votes are then resolved greedily into a one-to-one mapping: pairs in
/// descending order of votes, each kept when it has at least
/// `options.min_votes` votes and neither of its points was already taken.
///
/// The result is deterministic. Among keypoints at the same distance the
/// nearest is the one listed first for that image, and pairs with the same
/// vote count are taken in the order their first vote was cast (shared image
/// pairs in the order given, source observations in row order within each).
/// The returned pairs are in the order they were accepted.
///
/// Observations whose position is not finite (such as the `NaN` rows
/// `observation_pixels` leaves for a feature index past the end of its `.sift`
/// file) match nothing. The function knows only point ids, not positions, so
/// it keeps a pair with a point at infinity; a caller fitting positions drops
/// those pairs afterwards.
///
/// The three source slices are parallel, one entry per source observation, as
/// are the three target slices and the two shared-image slices.
///
/// # Panics
///
/// If parallel slices differ in length.
///
/// # Example
///
/// ```
/// use sfmtool_core::reconstruction::point_correspondence::{
///     find_point_correspondences_by_coordinate, CoordinateCorrespondenceOptions,
/// };
///
/// // Point 7 is seen in images 0 and 1 of the source, point 3 at nearly the
/// // same pixels in the same two images of the target.
/// let result = find_point_correspondences_by_coordinate(
///     &[0, 1],
///     &[[10.0, 20.0], [30.0, 40.0]],
///     &[7, 7],
///     &[0, 1],
///     &[[10.4, 19.8], [30.1, 40.2]],
///     &[3, 3],
///     &[0, 1],
///     &[0, 1],
///     &CoordinateCorrespondenceOptions::default(),
/// );
/// assert_eq!(result.source_ids, vec![7]);
/// assert_eq!(result.target_ids, vec![3]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn find_point_correspondences_by_coordinate(
    source_track_image_indexes: &[u32],
    source_keypoints_xy: &[[f64; 2]],
    source_track_point_indexes: &[u32],
    target_track_image_indexes: &[u32],
    target_keypoints_xy: &[[f64; 2]],
    target_track_point_indexes: &[u32],
    shared_images_source: &[u32],
    shared_images_target: &[u32],
    options: &CoordinateCorrespondenceOptions,
) -> PointCorrespondenceResult {
    assert_eq!(source_track_image_indexes.len(), source_keypoints_xy.len());
    assert_eq!(
        source_track_image_indexes.len(),
        source_track_point_indexes.len()
    );
    assert_eq!(target_track_image_indexes.len(), target_keypoints_xy.len());
    assert_eq!(
        target_track_image_indexes.len(),
        target_track_point_indexes.len()
    );
    assert_eq!(shared_images_source.len(), shared_images_target.len());

    let source_rows = finite_rows_by_image(source_track_image_indexes, source_keypoints_xy);
    let target_rows = finite_rows_by_image(target_track_image_indexes, target_keypoints_xy);
    let threshold_sq = options.pixel_threshold * options.pixel_threshold;

    // (source point, target point) -> (votes, order of the first vote).
    let mut votes: HashMap<(u32, u32), (usize, usize)> = HashMap::new();
    for (&src_img, &tgt_img) in shared_images_source.iter().zip(shared_images_target) {
        let (Some(src_rows), Some(tgt_rows)) =
            (source_rows.get(&src_img), target_rows.get(&tgt_img))
        else {
            continue;
        };
        let src = ImageKeypoints::new(src_rows, source_keypoints_xy);
        let tgt = ImageKeypoints::new(tgt_rows, target_keypoints_xy);
        for (i, &src_row) in src_rows.iter().enumerate() {
            let Some(j) = tgt.nearest(src.xy[i], options.pixel_threshold, threshold_sq) else {
                continue;
            };
            if src.nearest(tgt.xy[j], options.pixel_threshold, threshold_sq) != Some(i) {
                continue;
            }
            let pair = (
                source_track_point_indexes[src_row],
                target_track_point_indexes[tgt_rows[j]],
            );
            let next_order = votes.len();
            votes.entry(pair).or_insert((0, next_order)).0 += 1;
        }
    }

    let mut ordered: Vec<((u32, u32), (usize, usize))> = votes.into_iter().collect();
    ordered.sort_by(|(_, (va, oa)), (_, (vb, ob))| vb.cmp(va).then(oa.cmp(ob)));

    let mut source_ids = Vec::new();
    let mut target_ids = Vec::new();
    let mut used_sources = HashSet::new();
    let mut used_targets = HashSet::new();
    for ((src, tgt), (count, _)) in ordered {
        if count < options.min_votes {
            break;
        }
        if used_sources.contains(&src) || used_targets.contains(&tgt) {
            continue;
        }
        used_sources.insert(src);
        used_targets.insert(tgt);
        source_ids.push(src);
        target_ids.push(tgt);
    }
    PointCorrespondenceResult {
        source_ids,
        target_ids,
    }
}

/// `image -> rows`, in row order, of the observations with a finite position.
fn finite_rows_by_image(image_indexes: &[u32], xy: &[[f64; 2]]) -> HashMap<u32, Vec<usize>> {
    let mut map: HashMap<u32, Vec<usize>> = HashMap::new();
    for (row, (&image, p)) in image_indexes.iter().zip(xy).enumerate() {
        if p[0].is_finite() && p[1].is_finite() {
            map.entry(image).or_default().push(row);
        }
    }
    map
}

/// One image's keypoints from one reconstruction, with an index sorted by `x`
/// so a nearest-keypoint query scans only the keypoints within the threshold
/// in `x` rather than all of them.
struct ImageKeypoints {
    /// Keypoint positions, in the image's row order.
    xy: Vec<[f64; 2]>,
    /// Positions into `xy`, sorted by `x`.
    by_x: Vec<usize>,
}

impl ImageKeypoints {
    fn new(rows: &[usize], all_xy: &[[f64; 2]]) -> Self {
        let xy: Vec<[f64; 2]> = rows.iter().map(|&r| all_xy[r]).collect();
        let mut by_x: Vec<usize> = (0..xy.len()).collect();
        by_x.sort_by(|&a, &b| xy[a][0].total_cmp(&xy[b][0]).then(a.cmp(&b)));
        Self { xy, by_x }
    }

    /// The keypoint nearest to `p` within the threshold, the first-listed one
    /// among equally near keypoints, or `None` when none is that close.
    ///
    /// The `x` window is widened a little past the threshold so that rounding
    /// in the window's bounds never drops a keypoint the squared-distance test
    /// below would accept; that test alone decides.
    fn nearest(&self, p: [f64; 2], threshold: f64, threshold_sq: f64) -> Option<usize> {
        let reach = threshold.abs() * (1.0 + 1e-9);
        let lo = self.by_x.partition_point(|&k| self.xy[k][0] < p[0] - reach);
        let hi = self
            .by_x
            .partition_point(|&k| self.xy[k][0] <= p[0] + reach);
        let mut best: Option<(f64, usize)> = None;
        for &k in &self.by_x[lo.min(hi)..hi] {
            let dx = self.xy[k][0] - p[0];
            let dy = self.xy[k][1] - p[1];
            let d2 = dx * dx + dy * dy;
            if d2 <= threshold_sq && best.is_none_or(|(bd, bk)| d2 < bd || (d2 == bd && k < bk)) {
                best = Some((d2, k));
            }
        }
        best.map(|(_, k)| k)
    }
}

#[cfg(test)]
mod tests;
