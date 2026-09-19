// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Spatial proximity queries, in world units and in image pixels.
//!
//! The primary types are [`PointCloud2`] and [`PointCloud3`], KD-tree indexes
//! generic over the scalar type (`f32` or `f64`), which answer proximity
//! between points in world units at a shared radius. Alongside them
//! [`keypoint_reach`] answers proximity between the keypoints of a track set in
//! image pixels, per image, each keypoint carrying its own radius.
//!
//! See `specs/core/spatial/point-cloud-index.md` for the design.
//!
//! ```ignore
//! use sfmtool_core::spatial::PointCloud3;
//!
//! let positions: &[f32] = &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
//! let cloud = PointCloud3::<f32>::new(&positions, 2);
//! let nearest = cloud.nearest(&[0.5, 0.0, 0.0], 1);
//! ```

pub mod keypoint_reach;

use kiddo::{ImmutableKdTree, SquaredEuclidean};

/// The scalar coordinate types a [`PointCloud`] can be built over: `f32` and
/// `f64`.
///
/// Sealed, and deliberately the only name from this module that a caller ever
/// has to mention alongside [`PointCloud`]. The KD-tree backend's own trait
/// bounds live behind it, so they stay an implementation detail of this module
/// rather than something every signature that takes a cloud has to repeat.
pub trait Scalar: sealed::ScalarOps {}
impl<A: sealed::ScalarOps> Scalar for A {}

mod sealed {
    /// What [`PointCloud`](super::PointCloud) needs of its scalar type: the
    /// KD-tree's axis operations, plus the three float operations the wrapper
    /// performs itself (squaring a radius, taking a square root, and the
    /// "no neighbour" sentinel).
    ///
    /// The three KD-tree bounds are the crate's own public ones, and together
    /// they are what makes `SquaredEuclidean<Self>` a
    /// [`QueryMetric`](kiddo::dist::QueryMetric) over `Self`: `Axis` says the
    /// type can be a coordinate, `QueryDistance` that it can be a distance
    /// accumulator, and `WideningCastFrom` (with the two arithmetic ops) that
    /// coordinates can be accumulated into one. Stating them as supertraits
    /// rather than as a `where SquaredEuclidean<Self>: QueryMetric<Self>`
    /// clause is deliberate: a trait's `where` clause is not elaborated for
    /// code that is merely generic over the trait, so the clause would have to
    /// be repeated at every use.
    pub trait ScalarOps:
        kiddo::traits::Axis<Coord = Self>
        + kiddo::dist::QueryDistance
        + kiddo::dist::WideningCastFrom<Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Add<Output = Self>
        + Default
        + Send
        + Sync
    {
        /// Positive infinity: what a nearest-neighbour distance is when there
        /// is no other point to measure to.
        const INFINITY: Self;

        /// `self * self`. Radii arrive Euclidean and the metric is squared.
        fn squared(self) -> Self;

        /// The non-negative square root.
        fn sqrt(self) -> Self;
    }

    impl ScalarOps for f32 {
        const INFINITY: Self = f32::INFINITY;

        #[inline]
        fn squared(self) -> Self {
            self * self
        }

        #[inline]
        fn sqrt(self) -> Self {
            f32::sqrt(self)
        }
    }

    impl ScalarOps for f64 {
        const INFINITY: Self = f64::INFINITY;

        #[inline]
        fn squared(self) -> Self {
            self * self
        }

        #[inline]
        fn sqrt(self) -> Self {
            f64::sqrt(self)
        }
    }
}

/// A point cloud with a KD-tree for spatial queries.
///
/// Owns both the point positions and the KD-tree index, so it can be
/// built once and reused for multiple queries.
///
/// `A` is the scalar type (`f32` or `f64`), and `DIM` is the spatial
/// dimensionality (e.g. 2 or 3).
///
/// The index is kiddo's `ImmutableKdTree` — bulk-built, median-split, and
/// queried but never modified, which is exactly this type's life cycle. The
/// mutable tree is not an option even where a cloud is built once from a
/// growing list: its fixed-size leaf buckets hold at most 32 items sharing a
/// value on the split axis, and beyond that `add` rejects the item with
/// `UnsplittableBucket` and silently leaves it out of the tree. That limit is
/// reached routinely — SIFT features along a strong image edge, a rotation-only
/// reconstruction whose cameras all sit at the origin — whereas the immutable
/// tree's median splits take any number of them.
///
/// # Guarantees
///
/// These hold for every query method here and are pinned by the tests in
/// `spatial/tests.rs`:
///
/// - **Radii are Euclidean and inclusive.** Callers pass a plain distance;
///   the cloud squares it and a point exactly `radius` away is a hit. The
///   only distances the cloud itself returns, from
///   [`nearest_neighbor_distances`](Self::nearest_neighbor_distances), are
///   Euclidean too.
/// - **k-nearest results are ordered by ascending distance**, and padded with
///   `u32::MAX` when fewer than `k` neighbours qualify. The padding is always
///   a suffix.
/// - **[`within_radius`](Self::within_radius) is unordered**, as its CSR
///   shape suggests: the points in one query's span are whatever the traversal
///   found, in traversal order.
/// - **Ties are broken arbitrarily.** Two points at exactly the same distance
///   from a query may come back in either order, and which of them a
///   single-result query returns is unspecified. Nothing here promises the
///   lowest index wins.
/// - **Duplicate coordinates are fine**, including more of them than the
///   tree's bucket size — see `test_2d_construction_axis_cluster` and
///   `test_2d_construction_exact_duplicates`.
pub struct PointCloud<A: Scalar, const DIM: usize> {
    positions: Vec<A>,
    tree: ImmutableKdTree<A, DIM>,
    n_points: usize,
}

impl<A: Scalar, const DIM: usize> PointCloud<A, DIM> {
    /// Build from a flat slice of coordinates with `n_points` entries.
    ///
    /// For 2D: `[x, y, x, y, ...]`, for 3D: `[x, y, z, x, y, z, ...]`.
    ///
    /// # Panics
    ///
    /// If `positions.len()` is not `DIM * n_points`, or if `n_points` exceeds
    /// `u32::MAX` — point indices are `u32` throughout this API, so a cloud
    /// that large could not be queried anyway.
    pub fn new(positions: &[A], n_points: usize) -> Self {
        assert_eq!(
            positions.len(),
            n_points * DIM,
            "positions length must be {DIM} * n_points",
        );

        let mut points: Vec<[A; DIM]> = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let base = i * DIM;
            let mut point = [A::default(); DIM];
            point.copy_from_slice(&positions[base..base + DIM]);
            points.push(point);
        }
        let tree = ImmutableKdTree::<A, DIM>::new_from_slice(&points)
            .expect("point count fits in the tree's u32 item index");

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
    ///
    /// Which of several equidistant points is returned is unspecified.
    pub fn nearest(&self, query: &[A], n_queries: usize) -> Vec<u32> {
        assert_eq!(query.len(), n_queries * DIM);
        let mut result = Vec::with_capacity(n_queries);
        for i in 0..n_queries {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let nn = self
                .tree
                .query(&q)
                .nearest_one::<SquaredEuclidean<A>>()
                .execute();
            result.push(nn.item);
        }
        result
    }

    /// Find the nearest `k` point indices for each of `n_queries` query points.
    ///
    /// `query` is a flat slice of length `n_queries * DIM`.
    /// Returns a flat `Vec<u32>` of length `n_queries * k`, row-major.
    /// Each row is ordered by ascending distance; if fewer than `k` neighbors
    /// exist, the remaining slots are filled with `u32::MAX`.
    pub fn nearest_k(&self, query: &[A], n_queries: usize, k: usize) -> Vec<u32> {
        assert_eq!(query.len(), n_queries * DIM);
        let Some(k_nz) = std::num::NonZero::<usize>::new(k) else {
            return Vec::new();
        };
        let mut result = Vec::with_capacity(n_queries * k);
        for i in 0..n_queries {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let neighbors = self
                .tree
                .query(&q)
                .nearest_n::<SquaredEuclidean<A>>(k_nz)
                .execute();
            for nb in &neighbors {
                result.push(nb.item);
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
    /// The results for query `i` are `indices[offsets[i]..offsets[i+1]]`, in no
    /// particular order. `radius` is inclusive: a point exactly `radius` away
    /// is a hit.
    pub fn within_radius(&self, query: &[A], n_queries: usize, radius: A) -> (Vec<u32>, Vec<u32>) {
        assert_eq!(query.len(), n_queries * DIM);
        let radius_sq = radius.squared();
        let mut offsets = Vec::with_capacity(n_queries + 1);
        let mut indices = Vec::new();
        offsets.push(0u32);
        for i in 0..n_queries {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let neighbors = self
                .tree
                .query(&q)
                .within::<SquaredEuclidean<A>>(radius_sq)
                .unsorted()
                .execute();
            for nb in &neighbors {
                indices.push(nb.item);
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
    /// Results for each query are sorted by distance (nearest first), and
    /// `radius` is inclusive.
    pub fn nearest_k_within_radius(
        &self,
        query: &[A],
        n_queries: usize,
        k: usize,
        radius: A,
    ) -> Vec<u32> {
        use rayon::prelude::*;

        assert_eq!(query.len(), n_queries * DIM);
        let Some(k_nz) = std::num::NonZero::<usize>::new(k) else {
            return Vec::new();
        };
        let radius_sq = radius.squared();
        let mut result = vec![u32::MAX; n_queries * k];
        result.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            let base = i * DIM;
            let mut q = [A::default(); DIM];
            q.copy_from_slice(&query[base..base + DIM]);
            let neighbors = self
                .tree
                .query(&q)
                .nearest_n::<SquaredEuclidean<A>>(k_nz)
                .within(radius_sq)
                .execute();
            for (slot, nb) in row.iter_mut().zip(neighbors.iter()) {
                *slot = nb.item;
            }
        });
        result
    }

    /// Find the nearest `k` neighbors (excluding self) for every point in the cloud.
    ///
    /// Returns a flat `Vec<u32>` of length `n_points * k`, row-major, each row
    /// ordered by ascending distance.
    /// If fewer than `k` other points exist, remaining slots are `u32::MAX`.
    ///
    /// Only the point's own index is excluded, not points that happen to sit at
    /// the same coordinates.
    pub fn self_nearest_k(&self, k: usize) -> Vec<u32> {
        let k_plus_1 = std::num::NonZero::<usize>::new(k + 1).expect("k + 1 is non-zero");
        let mut result = Vec::with_capacity(self.n_points * k);
        for i in 0..self.n_points {
            let q = self.position(i);
            let neighbors = self
                .tree
                .query(&q)
                .nearest_n::<SquaredEuclidean<A>>(k_plus_1)
                .execute();
            let mut count = 0;
            for nb in &neighbors {
                if nb.item != i as u32 {
                    result.push(nb.item);
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
    /// The distance is Euclidean, and measured to the nearest point with a
    /// *different index* — two points at identical coordinates each report
    /// zero, not the distance to the next distinct location.
    ///
    /// Returns a `Vec<A>` of length `n_points`. If there is only one point,
    /// returns `[A::INFINITY]`.
    pub fn nearest_neighbor_distances(&self) -> Vec<A> {
        if self.n_points <= 1 {
            return vec![A::INFINITY; self.n_points];
        }

        let two = std::num::NonZero::<usize>::new(2).expect("2 is non-zero");
        let mut distances = Vec::with_capacity(self.n_points);
        for i in 0..self.n_points {
            let query = self.position(i);
            let neighbors = self
                .tree
                .query(&query)
                .nearest_n::<SquaredEuclidean<A>>(two)
                .execute();

            let mut min_dist = A::INFINITY;
            for nb in &neighbors {
                if nb.item != i as u32 {
                    min_dist = nb.distance.sqrt();
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
