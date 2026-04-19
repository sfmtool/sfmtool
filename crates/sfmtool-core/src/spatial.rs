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
mod tests;
