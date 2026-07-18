// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::Point3;
use rand::seq::SliceRandom;

use super::gpu_types::{FALLBACK_POINT_SIZE, NN_SUBSAMPLE_COUNT};

/// Compute an automatic point size from nearest-neighbor distances.
///
/// Builds a KD-tree of all points, then queries NN distances for a random
/// subsample of up to `NN_SUBSAMPLE_COUNT` points. Returns
/// `median_nn_distance * 0.5` as the splat radius.
pub(super) fn compute_auto_point_size(points: &[sfmtool_core::Point3D]) -> f32 {
    // Points at infinity store a unit direction, not a location, so they would
    // cluster on the unit sphere and skew NN distances — exclude them.
    let positions: Vec<[f32; 3]> = points
        .iter()
        .filter(|p| !p.is_at_infinity())
        .map(|p| {
            [
                p.position.x as f32,
                p.position.y as f32,
                p.position.z as f32,
            ]
        })
        .collect();
    if positions.len() < 2 {
        return FALLBACK_POINT_SIZE;
    }

    // Build KD-tree from finite points (f32 for speed)
    let mut tree: KdTree<f32, 3> = KdTree::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        tree.add(p, i as u64);
    }

    // Subsample indices for NN queries
    let mut rng = rand::rng();
    let query_indices: Vec<usize> = if positions.len() <= NN_SUBSAMPLE_COUNT {
        (0..positions.len()).collect()
    } else {
        let mut indices: Vec<usize> = (0..positions.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(NN_SUBSAMPLE_COUNT);
        indices
    };

    // Query nearest neighbor for each subsampled point (k=2: self + nearest)
    let mut nn_distances: Vec<f32> = Vec::with_capacity(query_indices.len());
    for &idx in &query_indices {
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&positions[idx], 2);
        // The first result is the point itself (distance 0); take the second
        if neighbors.len() >= 2 {
            let dist = neighbors[1].distance.sqrt();
            if dist > 0.0 {
                nn_distances.push(dist);
            }
        }
    }

    if nn_distances.is_empty() {
        return FALLBACK_POINT_SIZE;
    }

    // 40th percentile NN distance
    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p40 = nn_distances[nn_distances.len() * 2 / 5];

    let auto_size = p40 * 1.2;
    log::info!(
        "Auto point size: {:.4} (p40 NN dist: {:.4}, from {} queries over {} finite points)",
        auto_size,
        p40,
        nn_distances.len(),
        positions.len()
    );

    auto_size
}

/// Compute a characteristic inter-camera distance from nearest-neighbor distances.
///
/// Builds a KD-tree of all camera centers, queries the NN distance for each
/// camera, and returns the 90th percentile. The high percentile makes the
/// result robust to a few cameras that happen to sit on top of each other
/// (e.g. colocated rig cameras), which would otherwise pull the value to zero.
///
/// Returns `None` if there are fewer than 2 images.
pub(super) fn compute_camera_nn_scale(images: &[sfmtool_core::SfmrImage]) -> Option<f32> {
    if images.len() < 2 {
        return None;
    }

    let mut tree: KdTree<f32, 3> = KdTree::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        let c = img.camera_center();
        tree.add(&[c.x as f32, c.y as f32, c.z as f32], i as u64);
    }

    let mut nn_distances: Vec<f32> = Vec::with_capacity(images.len());
    for img in images {
        let c = img.camera_center();
        let query = [c.x as f32, c.y as f32, c.z as f32];
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&query, 2);
        if neighbors.len() >= 2 {
            let dist = neighbors[1].distance.sqrt();
            if dist > 0.0 {
                nn_distances.push(dist);
            }
        }
    }

    if nn_distances.is_empty() {
        return None;
    }

    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = nn_distances[nn_distances.len() * 9 / 10];

    log::info!(
        "Camera NN scale: {:.4} (p90 of {} NN distances from {} cameras)",
        p90,
        nn_distances.len(),
        images.len()
    );

    Some(p90)
}

/// Compute the bounding sphere (center, radius) for a set of 3D points.
///
/// Uses component-wise median for a robust center, then 80th percentile
/// distance from center as a robust radius. Handles outliers gracefully
/// since percentile-based statistics ignore extreme values.
///
/// Returns `(origin, 1.0)` if fewer than 2 points.
pub(super) fn compute_scene_bounds(points: &[sfmtool_core::Point3D]) -> (Point3<f64>, f64) {
    // Exclude points at infinity: their `position` is a unit direction, not a
    // location, and would pull the center toward the origin and distort the
    // radius (and hence the adaptive clip planes that depend on these bounds).
    let finite: Vec<&sfmtool_core::Point3D> =
        points.iter().filter(|p| !p.is_at_infinity()).collect();
    if finite.len() < 2 {
        return (Point3::origin(), 1.0);
    }

    // Collect coordinates
    let mut xs: Vec<f64> = finite.iter().map(|p| p.position.x).collect();
    let mut ys: Vec<f64> = finite.iter().map(|p| p.position.y).collect();
    let mut zs: Vec<f64> = finite.iter().map(|p| p.position.z).collect();

    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    zs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = xs.len();
    let center = Point3::new(xs[n / 2], ys[n / 2], zs[n / 2]);

    // Compute distances from center and take 80th percentile
    let mut dists: Vec<f64> = finite
        .iter()
        .map(|p| (p.position - center).norm())
        .collect();
    dists.sort_unstable_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    let p80 = dists[n * 4 / 5].max(0.1);

    log::info!(
        "Scene bounds: center=[{:.2}, {:.2}, {:.2}], radius={:.2} (from {} points)",
        center.x,
        center.y,
        center.z,
        p80,
        n
    );

    (center, p80)
}

#[cfg(test)]
mod tests;
