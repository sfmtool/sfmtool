// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Spatial analysis utilities using KD-trees.
//!
//! The primary types are [`PointCloud2`] and [`PointCloud3`], generic over the
//! scalar type (`f32` or `f64`).
//!
//! ```ignore
//! use sfmtool_core::spatial::PointCloud3;
//!
//! let positions: &[f32] = &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
//! let cloud = PointCloud3::<f32>::new(&positions, 2);
//! let nearest = cloud.nearest(&[0.5, 0.0, 0.0], 1);
//! ```

use kiddo::float::kdtree::Axis;
use kiddo::SquaredEuclidean;

/// A point cloud with a KD-tree for spatial queries.
///
/// Owns both the point positions and the KD-tree index, so it can be
/// built once and reused for multiple queries.
///
/// `A` is the scalar type (`f32` or `f64`), and `DIM` is the spatial
/// dimensionality (e.g. 2 or 3).
pub struct PointCloud<A: Axis, const DIM: usize> {
    positions: Vec<A>,
    tree: kiddo::KdTree<A, DIM>,
    n_points: usize,
}

impl<A: Axis, const DIM: usize> PointCloud<A, DIM> {
    /// Build from a flat slice of coordinates with `n_points` entries.
    ///
    /// For 2D: `[x, y, x, y, ...]`, for 3D: `[x, y, z, x, y, z, ...]`.
    pub fn new(positions: &[A], n_points: usize) -> Self {
        assert_eq!(
            positions.len(),
            n_points * DIM,
            "positions length must be {DIM} * n_points",
        );

        let mut tree = kiddo::KdTree::<A, DIM>::new();
        for i in 0..n_points {
            let base = i * DIM;
            let mut point = [A::default(); DIM];
            point.copy_from_slice(&positions[base..base + DIM]);
            tree.add(&point, i as u64);
        }

        Self {
            positions: positions.to_vec(),
            tree,
            n_points,
        }
    }

    /// Number of points in the cloud.
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Whether the cloud is empty.
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Get the position of point `i`.
    pub fn position(&self, i: usize) -> [A; DIM] {
        let base = i * DIM;
        let mut point = [A::default(); DIM];
        point.copy_from_slice(&self.positions[base..base + DIM]);
        point
    }

    /// Find the index of the nearest point for each of `n_queries` query points.
    ///
    /// `query` is a flat slice of length `n_queries * DIM`.
    /// Returns a `Vec<u32>` of length `n_queries`.
    pub fn nearest(&self, query: &[A], n_queries: usize) -> Vec<u32> {
        assert_eq!(query.len(), n_queries * DIM);
        let mut result = Vec::with_capacity(n_queries);
        for i in 0..n_queries {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let nn = self.tree.nearest_one::<SquaredEuclidean>(&q);
            result.push(nn.item as u32);
        }
        result
    }

    /// Find the nearest `k` point indices for each of `n_queries` query points.
    ///
    /// `query` is a flat slice of length `n_queries * DIM`.
    /// Returns a flat `Vec<u32>` of length `n_queries * k`, row-major.
    /// If fewer than `k` neighbors exist, remaining slots are filled with `u32::MAX`.
    pub fn nearest_k(&self, query: &[A], n_queries: usize, k: usize) -> Vec<u32> {
        assert_eq!(query.len(), n_queries * DIM);
        let mut result = Vec::with_capacity(n_queries * k);
        for i in 0..n_queries {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let neighbors = self.tree.nearest_n::<SquaredEuclidean>(&q, k);
            for nb in &neighbors {
                result.push(nb.item as u32);
            }
            for _ in neighbors.len()..k {
                result.push(u32::MAX);
            }
        }
        result
    }

    /// Find all points within `radius` (Euclidean) of each of `n_queries` query points.
    ///
    /// Returns `(offsets, indices)` in CSR format:
    /// - `offsets` has length `n_queries + 1`, with `offsets[0] == 0`
    ///   and `offsets[n_queries] == R` (total result count).
    /// - `indices` has length `R` with 0-based point indices.
    ///
    /// The results for query `i` are `indices[offsets[i]..offsets[i+1]]`.
    pub fn within_radius(&self, query: &[A], n_queries: usize, radius: A) -> (Vec<u32>, Vec<u32>) {
        assert_eq!(query.len(), n_queries * DIM);
        let radius_sq = radius * radius;
        let mut offsets = Vec::with_capacity(n_queries + 1);
        let mut indices = Vec::new();
        offsets.push(0u32);
        for i in 0..n_queries {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let neighbors = self.tree.within_unsorted::<SquaredEuclidean>(&q, radius_sq);
            for nb in &neighbors {
                indices.push(nb.item as u32);
            }
            offsets.push(indices.len() as u32);
        }
        (offsets, indices)
    }

    /// Find up to `k` nearest points within `radius` (Euclidean) for each query.
    ///
    /// Combines the semantics of `nearest_k` and `within_radius`: returns the
    /// closest `k` points, but only those within the given Euclidean distance.
    ///
    /// `query` is a flat slice of length `n_queries * DIM`.
    /// Returns a flat `Vec<u32>` of length `n_queries * k`, row-major.
    /// If fewer than `k` neighbors are within `radius`, remaining slots are `u32::MAX`.
    /// Results for each query are sorted by distance (nearest first).
    pub fn nearest_k_within_radius(
        &self,
        query: &[A],
        n_queries: usize,
        k: usize,
        radius: A,
    ) -> Vec<u32> {
        use rayon::prelude::*;

        assert_eq!(query.len(), n_queries * DIM);
        let radius_sq = radius * radius;
        let mut result = vec![u32::MAX; n_queries * k];
        result.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let neighbors = self
                .tree
                .nearest_n_within::<SquaredEuclidean>(&q, radius_sq, k, true);
            for (slot, nb) in row.iter_mut().zip(neighbors.iter()) {
                *slot = nb.item as u32;
            }
        });
        result
    }

    /// Find the nearest `k` neighbors (excluding self) for every point in the cloud.
    ///
    /// Returns a flat `Vec<u32>` of length `n_points * k`, row-major.
    /// If fewer than `k` other points exist, remaining slots are `u32::MAX`.
    pub fn self_nearest_k(&self, k: usize) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.n_points * k);
        for i in 0..self.n_points {
            let q = self.position(i);
            let neighbors = self.tree.nearest_n::<SquaredEuclidean>(&q, k + 1);
            let mut count = 0;
            for nb in &neighbors {
                if nb.item != i as u64 {
                    result.push(nb.item as u32);
                    count += 1;
                    if count == k {
                        break;
                    }
                }
            }
            for _ in count..k {
                result.push(u32::MAX);
            }
        }
        result
    }

    /// Compute the nearest-neighbor Euclidean distance for each point in the cloud.
    ///
    /// Returns a `Vec<A>` of length `n_points`. If there is only one point,
    /// returns `[A::infinity()]`.
    pub fn nearest_neighbor_distances(&self) -> Vec<A> {
        if self.n_points <= 1 {
            return vec![A::infinity(); self.n_points];
        }

        let mut distances = Vec::with_capacity(self.n_points);
        for i in 0..self.n_points {
            let query = self.position(i);
            let neighbors = self.tree.nearest_n::<SquaredEuclidean>(&query, 2);

            let mut min_dist = A::infinity();
            for nb in &neighbors {
                if nb.item != i as u64 {
                    // FloatCore doesn't provide sqrt, so round-trip through f64.
                    let sq = nb.distance.to_f64().unwrap();
                    min_dist = A::from(sq.sqrt()).unwrap();
                    break;
                }
            }
            distances.push(min_dist);
        }

        distances
    }
}

// ── Type aliases ────────────────────────────────────────────────────────

/// A 2D point cloud, generic over the scalar type.
pub type PointCloud2<A> = PointCloud<A, 2>;
/// A 3D point cloud, generic over the scalar type.
pub type PointCloud3<A> = PointCloud<A, 3>;

#[cfg(test)]
mod tests {
    use super::*;

    // ── f64 query tests ─────────────────────────────────────────────────

    #[test]
    fn test_2d_nearest() {
        let positions = [0.0_f64, 0.0, 3.0, 4.0, 10.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 3);

        let result = cloud.nearest(&[1.0, 1.0], 1);
        assert_eq!(result, vec![0]);

        let result = cloud.nearest(&[9.0, 0.0], 1);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_2d_nearest_k() {
        let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 4);

        let result = cloud.nearest_k(&[0.5, 0.0], 1, 2);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_2d_within_radius() {
        let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 4);

        let (offsets, indices) = cloud.within_radius(&[0.5, 0.0], 1, 2.0);
        assert_eq!(offsets, vec![0, 3]);
        assert_eq!(indices.len(), 3);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_2d_self_nearest_k() {
        let positions = [0.0_f64, 0.0, 1.0, 0.0, 5.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 3);

        let result = cloud.self_nearest_k(1);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 1);
    }

    #[test]
    fn test_3d_nearest() {
        let positions = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud3::<f64>::new(&positions, 3);

        let result = cloud.nearest(&[0.5, 0.0, 0.0], 1);
        assert_eq!(result, vec![0]);

        let result = cloud.nearest(&[0.9, 0.0, 0.0], 1);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_3d_within_radius() {
        let positions = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 0.0, 0.0];
        let cloud = PointCloud3::<f64>::new(&positions, 3);

        let (offsets, indices) = cloud.within_radius(&[0.5, 0.0, 0.0], 1, 2.0);
        assert_eq!(offsets, vec![0, 2]);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1]);
    }

    #[test]
    fn test_batch_queries() {
        let positions = [0.0_f64, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 2);

        let result = cloud.nearest(&[1.0, 0.0, 9.0, 0.0], 2);
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_self_nearest_k_greater_than_available() {
        let positions = [0.0_f64, 0.0, 1.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 2);
        let result = cloud.self_nearest_k(3);
        assert_eq!(result.len(), 6);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], u32::MAX);
        assert_eq!(result[2], u32::MAX);
        assert_eq!(result[3], 0);
        assert_eq!(result[4], u32::MAX);
        assert_eq!(result[5], u32::MAX);
    }

    // ── f32 tests ───────────────────────────────────────────────────────

    #[test]
    fn test_2d_f32_nearest() {
        let positions: [f32; 6] = [0.0, 0.0, 3.0, 4.0, 10.0, 0.0];
        let cloud = PointCloud2::<f32>::new(&positions, 3);
        assert_eq!(cloud.len(), 3);

        let result = cloud.nearest(&[1.0, 1.0], 1);
        assert_eq!(result, vec![0]);

        let result = cloud.nearest(&[9.0, 0.0], 1);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_2d_f32_nearest_k() {
        let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f32>::new(&positions, 4);

        let result = cloud.nearest_k(&[0.5_f32, 0.0], 1, 2);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_2d_f32_within_radius() {
        let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f32>::new(&positions, 4);

        let (offsets, indices) = cloud.within_radius(&[0.5_f32, 0.0], 1, 2.0);
        assert_eq!(offsets, vec![0, 3]);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_2d_f32_self_nearest_k() {
        let positions: [f32; 6] = [0.0, 0.0, 1.0, 0.0, 5.0, 0.0];
        let cloud = PointCloud2::<f32>::new(&positions, 3);

        let result = cloud.self_nearest_k(1);
        assert_eq!(result, vec![1, 0, 1]);
    }

    #[test]
    fn test_3d_f32_nearest() {
        let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud3::<f32>::new(&positions, 3);

        let result = cloud.nearest(&[0.5_f32, 0.0, 0.0], 1);
        assert_eq!(result, vec![0]);

        let result = cloud.nearest(&[0.9_f32, 0.0, 0.0], 1);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_3d_f32_self_nearest_k() {
        let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let cloud = PointCloud3::<f32>::new(&positions, 3);

        let result = cloud.self_nearest_k(1);
        assert_eq!(result, vec![1, 0, 1]);
    }

    // ── nearest_k_within_radius tests ────────────────────────────────────

    #[test]
    fn test_2d_nearest_k_within_radius() {
        // Points at x = 0, 1, 5, 10
        let positions = [0.0_f64, 0.0, 1.0, 0.0, 5.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 4);

        // Query at 0.1: within radius 2.0, k=3 → should get points 0 (dist 0.1) and 1 (dist 0.9)
        let result = cloud.nearest_k_within_radius(&[0.1, 0.0], 1, 3, 2.0);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0); // nearest (dist 0.1)
        assert_eq!(result[1], 1); // 2nd nearest (dist 0.9)
        assert_eq!(result[2], u32::MAX); // point 2 at dist 4.9, beyond radius
    }

    #[test]
    fn test_2d_nearest_k_within_radius_limits_by_k() {
        // Points at x = 0, 1, 2, 10 — query at 0.1 avoids equidistant ties
        let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 4);

        // 3 nearby points within radius 3, but k=2
        let result = cloud.nearest_k_within_radius(&[0.1, 0.0], 1, 2, 3.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0); // dist 0.1
        assert_eq!(result[1], 1); // dist 0.9
    }

    #[test]
    fn test_2d_nearest_k_within_radius_limits_by_radius() {
        let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f64>::new(&positions, 4);

        // k=5 but radius 0.6 → only point 0 (at dist 0.1) and point 1 (at dist 0.9 > 0.6)
        let result = cloud.nearest_k_within_radius(&[0.1, 0.0], 1, 5, 0.6);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], u32::MAX);
    }

    #[test]
    fn test_2d_f32_nearest_k_within_radius() {
        let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 10.0, 0.0];
        let cloud = PointCloud2::<f32>::new(&positions, 4);

        // Query at 0.1: k=3 within radius 2.0 → points 0 (dist 0.1) and 1 (dist 0.9)
        let result = cloud.nearest_k_within_radius(&[0.1_f32, 0.0], 1, 3, 2.0);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], u32::MAX);
    }

    #[test]
    fn test_3d_nearest_k_within_radius() {
        let positions = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 0.0, 0.0];
        let cloud = PointCloud3::<f64>::new(&positions, 3);

        let result = cloud.nearest_k_within_radius(&[0.1, 0.0, 0.0], 1, 5, 2.0);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 0); // dist 0.1
        assert_eq!(result[1], 1); // dist 0.9
        assert_eq!(result[2], u32::MAX);
    }

    // ── nearest_neighbor_distances on different types ────────────────────

    #[test]
    fn test_nn_distances_2d_f64() {
        let positions = [0.0_f64, 0.0, 3.0, 4.0];
        let cloud = PointCloud2::<f64>::new(&positions, 2);
        let dists = cloud.nearest_neighbor_distances();
        assert_eq!(dists.len(), 2);
        assert!((dists[0] - 5.0).abs() < 1e-10);
        assert!((dists[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nn_distances_3d_f32() {
        let positions: [f32; 6] = [0.0, 0.0, 0.0, 3.0, 4.0, 0.0];
        let cloud = PointCloud3::<f32>::new(&positions, 2);
        let dists = cloud.nearest_neighbor_distances();
        assert_eq!(dists.len(), 2);
        assert!((dists[0] - 5.0).abs() < 1e-5);
        assert!((dists[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_nn_distances_single_point_f32() {
        let positions: [f32; 2] = [1.0, 2.0];
        let cloud = PointCloud2::<f32>::new(&positions, 1);
        let dists = cloud.nearest_neighbor_distances();
        assert_eq!(dists.len(), 1);
        assert!(dists[0].is_infinite());
    }
}
