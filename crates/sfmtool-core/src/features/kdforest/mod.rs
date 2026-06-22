// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Randomized kd-tree forest: a pure-Rust approximate nearest-neighbor (ANN)
//! index for high-dimensional data such as SIFT descriptors.
//!
//! Implements the *multiple randomized kd-trees* of Muja & Lowe, "Fast
//! Approximate Nearest Neighbors with Automatic Algorithm Configuration"
//! (VISAPP 2009, §3.1), building on the randomized trees of Silpa-Anan &
//! Hartley, the priority-queue search of Arya et al., and the best-bin-first
//! fixed-budget stopping criterion of Beis & Lowe.
//!
//! Exact kd-trees (see [`crate::spatial`]) degenerate to near-linear search in
//! the ~128 dimensions of a SIFT descriptor. The forest trades a small,
//! controllable loss in accuracy for one to three orders of magnitude in speed:
//! it builds `T` independent trees whose split dimensions are randomized among
//! the top-variance axes, then searches them together under a single
//! best-bin-first priority queue with a fixed budget of distance computations.
//!
//! The index is generic over a scalar [`ForestScalar`] (`u8` for descriptors,
//! `f32` for general vectors) and works on flat row-major coordinate arrays,
//! mirroring [`crate::spatial::PointCloud`]. Descriptor distances stay in the
//! integer domain (`i64` squared-L2), matching
//! [`crate::features::feature_match::descriptor`]; `sqrt` is taken only when reporting.
//!
//! # Example
//! ```
//! use sfmtool_core::features::kdforest::{KdForestU8, KdForestParams};
//!
//! // Three 4-D u8 points, row-major.
//! let points: Vec<u8> = vec![
//!     0, 0, 0, 0,
//!     10, 10, 10, 10,
//!     0, 1, 0, 1,
//! ];
//! let forest = KdForestU8::build(&points, 3, 4, KdForestParams::balanced());
//! let nbrs = forest.search(&[0, 0, 0, 0], 1, 64, None);
//! assert_eq!(nbrs[0].index, 0);
//! ```

mod build;
mod calibrate;
mod distance;
mod search;

#[cfg(test)]
mod tests;

pub use distance::{ForestScalar, OrdF32};
pub use search::Neighbor;

use build::{build_tree, Tree};
use rayon::prelude::*;

/// Print per-query search diagnostics to stderr when `SFMTOOL_KDFOREST_STATS=1`.
///
/// Mirrors the SIFT/optical-flow `*_STATS`/`*_TIMING` precedent; one cached bool
/// check per batch when unset. Reported by [`KdForest::search_batch`] et al.
static KDFOREST_STATS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_KDFOREST_STATS").is_some());

/// Tunable parameters for a [`KdForest`].
///
/// Defaults follow Muja & Lowe (2009); `max_leaf_checks` is the precision/speed
/// dial and the only value most callers vary (directly or via
/// [`KdForest::calibrate_max_leaf_checks`]).
#[derive(Clone, Copy, Debug)]
pub struct KdForestParams {
    /// `T`: number of independent randomized trees. More trees raise precision
    /// (gains saturate around ~20 in the paper) at linear memory cost.
    pub num_trees: usize,
    /// Number of top-variance dimensions the split axis is drawn from at random
    /// (the paper's `D`). Its fixed `D = 5` works well across datasets.
    pub split_dim_candidates: usize,
    /// Maximum points per leaf bucket before a node is split.
    pub leaf_size: usize,
    /// `L_max`: default budget of unique distance computations per query. The
    /// precision dial — larger is more accurate and slower.
    pub max_leaf_checks: usize,
    /// Base RNG seed for reproducible tree construction.
    pub seed: u64,
}

impl Default for KdForestParams {
    fn default() -> Self {
        Self::balanced()
    }
}

impl KdForestParams {
    /// Balanced default: 4 trees, a moderate budget. Good for SIFT-sized work.
    pub fn balanced() -> Self {
        Self {
            num_trees: 4,
            split_dim_candidates: 5,
            leaf_size: 16,
            max_leaf_checks: 128,
            seed: 0,
        }
    }

    /// Fast preset: same forest, smaller budget — lower precision, lower latency.
    pub fn fast() -> Self {
        Self {
            max_leaf_checks: 32,
            ..Self::balanced()
        }
    }

    /// Accurate preset: more trees and a larger budget — higher precision.
    pub fn accurate() -> Self {
        Self {
            num_trees: 8,
            max_leaf_checks: 512,
            ..Self::balanced()
        }
    }
}

/// A randomized kd-tree forest over `n_points` points of dimension `dim`.
///
/// Built once from a flat row-major coordinate array, then queried many times.
/// `Send`/`Sync`: queries borrow the forest immutably and own their per-query
/// scratch, so [`search_batch`](KdForest::search_batch) parallelizes freely.
pub struct KdForest<S: ForestScalar> {
    points: Vec<S>,
    n_points: usize,
    dim: usize,
    trees: Vec<Tree<S>>,
    params: KdForestParams,
}

impl<S: ForestScalar> KdForest<S> {
    /// Build a forest from a flat row-major array of `n_points * dim` values.
    ///
    /// The `T` trees are constructed in parallel, each from its own seeded RNG
    /// (`seed + tree_index`), so the result is deterministic for a given seed
    /// regardless of thread count.
    ///
    /// For floating-point scalars, coordinates must be finite — `NaN`/infinity
    /// would corrupt the distance ordering (see [`OrdF32`]).
    #[must_use]
    pub fn build(points: &[S], n_points: usize, dim: usize, params: KdForestParams) -> Self {
        assert_eq!(
            points.len(),
            n_points * dim,
            "points length must be n_points * dim",
        );
        assert!(dim > 0, "dim must be positive");
        assert!(
            dim <= u16::MAX as usize,
            "dim must fit in u16 (split dimensions are stored as u16); got {dim}",
        );
        assert!(
            n_points <= u32::MAX as usize,
            "n_points must fit in u32 (point ids are u32); got {n_points}",
        );
        assert!(params.num_trees > 0, "num_trees must be positive");
        assert!(
            params.split_dim_candidates > 0,
            "split_dim_candidates must be positive"
        );
        assert!(params.leaf_size > 0, "leaf_size must be positive");

        let trees: Vec<Tree<S>> = (0..params.num_trees)
            .into_par_iter()
            .map(|t| {
                let ids: Vec<u32> = (0..n_points as u32).collect();
                build_tree(
                    points,
                    dim,
                    ids,
                    &params,
                    params.seed.wrapping_add(t as u64),
                )
            })
            .collect();

        Self {
            points: points.to_vec(),
            n_points,
            dim,
            trees,
            params,
        }
    }

    /// Number of points in the forest.
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Whether the forest is empty.
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Dimensionality of the indexed points.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The parameters this forest was built with.
    pub fn params(&self) -> KdForestParams {
        self.params
    }

    /// Batched k-NN: flat `n_queries * dim` queries in, flat `n_queries * k`
    /// point indices out (row-major), `u32::MAX` padding where fewer than `k`
    /// neighbors are found (or fall within `max_dist`). Parallel over rows.
    ///
    /// `max_leaf_checks` is the soft per-query budget; see
    /// [`search`](KdForest::search) for its exact semantics.
    #[must_use]
    pub fn search_batch(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Vec<u32> {
        self.search_batch_inner(queries, n_queries, k, max_leaf_checks, max_dist)
            .0
    }

    /// As [`search_batch`](KdForest::search_batch) but also returns the squared
    /// distances (flat `n_queries * k`, `f32::INFINITY` padding), for ratio
    /// tests and thresholding.
    #[must_use]
    pub fn search_batch_with_distances(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> (Vec<u32>, Vec<f32>) {
        self.search_batch_inner(queries, n_queries, k, max_leaf_checks, max_dist)
    }

    fn search_batch_inner(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> (Vec<u32>, Vec<f32>) {
        assert_eq!(
            queries.len(),
            n_queries * self.dim,
            "queries length must be n_queries * dim",
        );

        let mut indices = vec![u32::MAX; n_queries * k];
        let mut distances = vec![f32::INFINITY; n_queries * k];
        if k == 0 {
            return (indices, distances);
        }

        // Optional diagnostics: accumulated lock-free, only touched when enabled.
        let stats = (*KDFOREST_STATS).then(AtomicStats::default);

        // One reusable scratch per rayon worker (allocated once, reset per query)
        // keeps the query inner loop free of per-query heap/bitset allocation.
        indices
            .par_chunks_mut(k)
            .zip(distances.par_chunks_mut(k))
            .enumerate()
            .for_each_init(
                || search::SearchScratch::new(self.n_points),
                |scratch, (i, (irow, drow))| {
                    let q = &queries[i * self.dim..(i + 1) * self.dim];
                    let mut s = search::QueryStats::default();
                    self.run_query(q, k, max_leaf_checks, max_dist, scratch, &mut s);
                    scratch.write_results_into(irow, drow);
                    if let Some(acc) = &stats {
                        acc.record(&s);
                    }
                },
            );

        if let Some(acc) = &stats {
            acc.report(n_queries, k, max_leaf_checks, self.params.num_trees);
        }

        (indices, distances)
    }
}

/// Lock-free accumulator for [`search::QueryStats`] across a batch, used only
/// when `SFMTOOL_KDFOREST_STATS` is set.
#[derive(Default)]
struct AtomicStats {
    checks: std::sync::atomic::AtomicU64,
    pushes: std::sync::atomic::AtomicU64,
    pops: std::sync::atomic::AtomicU64,
}

impl AtomicStats {
    fn record(&self, s: &search::QueryStats) {
        use std::sync::atomic::Ordering::Relaxed;
        self.checks.fetch_add(s.checks, Relaxed);
        self.pushes.fetch_add(s.pushes, Relaxed);
        self.pops.fetch_add(s.pops, Relaxed);
    }

    fn report(&self, n_queries: usize, k: usize, max_leaf_checks: usize, num_trees: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let nq = n_queries.max(1) as f64;
        eprintln!(
            "KDFOREST_STATS: {n_queries} queries, k={k}, L_max={max_leaf_checks}, T={num_trees} | \
             avg checks {:.1}, avg pushes {:.1}, avg pops {:.1}",
            self.checks.load(Relaxed) as f64 / nq,
            self.pushes.load(Relaxed) as f64 / nq,
            self.pops.load(Relaxed) as f64 / nq,
        );
    }
}

/// A forest over `u8` descriptors (integer squared-L2), e.g. SIFT.
pub type KdForestU8 = KdForest<u8>;
/// A forest over `f32` vectors (floating squared-L2). Coordinates must be finite
/// (`NaN`/infinity corrupt the distance ordering). Phase 2 will harden and
/// benchmark this path; today it is exercised far less than the `u8` index.
pub type KdForestF32 = KdForest<f32>;
