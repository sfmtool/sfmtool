// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

/// Result of filtering points by a boolean mask.
pub struct FilteredTracksResult {
    /// Filtered track image indexes (only observations of kept points)
    pub track_image_indexes: Vec<u32>,
    /// Filtered track feature indexes (only observations of kept points)
    pub track_feature_indexes: Vec<u32>,
    /// Remapped track point IDs (contiguous from 0)
    pub track_point_ids: Vec<u32>,
}

/// Filter tracks based on a point keep mask, and remap point IDs to be contiguous.
///
/// Given a boolean mask indicating which points to keep, this function:
/// 1. Filters track observations to only include kept points
/// 2. Remaps point IDs to be contiguous (0, 1, 2, ...)
///
/// # Arguments
/// * `points_to_keep_mask` - Boolean slice of length num_points. true = keep.
/// * `track_image_indexes` - Image index for each track observation
/// * `track_feature_indexes` - Feature index for each track observation
/// * `track_point_ids` - Point ID for each track observation
///
/// # Returns
/// FilteredTracksResult with filtered and remapped track data
pub fn filter_tracks_by_point_mask(
    points_to_keep_mask: &[bool],
    track_image_indexes: &[u32],
    track_feature_indexes: &[u32],
    track_point_ids: &[u32],
) -> FilteredTracksResult {
    let num_points = points_to_keep_mask.len();

    // Build point ID remapping: old_id -> new_id (contiguous)
    let mut point_id_mapping = vec![u32::MAX; num_points];
    let mut new_id = 0u32;
    for (old_id, &keep) in points_to_keep_mask.iter().enumerate() {
        if keep {
            point_id_mapping[old_id] = new_id;
            new_id += 1;
        }
    }

    // Filter and remap tracks in a single pass
    let n_tracks = track_point_ids.len();
    let mut new_image_indexes = Vec::with_capacity(n_tracks);
    let mut new_feature_indexes = Vec::with_capacity(n_tracks);
    let mut new_point_ids = Vec::with_capacity(n_tracks);

    for i in 0..n_tracks {
        let point_id = track_point_ids[i] as usize;
        if point_id < num_points && points_to_keep_mask[point_id] {
            new_image_indexes.push(track_image_indexes[i]);
            new_feature_indexes.push(track_feature_indexes[i]);
            new_point_ids.push(point_id_mapping[point_id]);
        }
    }

    FilteredTracksResult {
        track_image_indexes: new_image_indexes,
        track_feature_indexes: new_feature_indexes,
        track_point_ids: new_point_ids,
    }
}

#[cfg(test)]
mod tests;
