// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two forest operations a query written against "a forest" needs.

use sfmtool_kdf_format::KdfScalar;

use super::distance::ForestScalar;
use super::{KdForest, KdfError, LazyKdForest};

/// Batched approximate k-NN with distances, plus read-back of the corpus
/// vectors behind a feature ID.
///
/// The resident [`KdForest`] and the file-backed [`LazyKdForest`] already have
/// both operations and, for the same forest, answer them identically: the file
/// stores the topology, leaf order and feature IDs rather than a seed, so a
/// query against the file visits the same leaves in the same order as the
/// forest it was written from. This trait is what lets an algorithm over those
/// two operations be written once, and it is also what makes their agreement
/// testable end to end: the same function, given the same forest through both
/// paths, must return the same answer rather than merely the same neighbours.
///
/// It is deliberately small. Anything a caller wants that only one of the two
/// can supply -- I/O counters, the cache budget, the in-memory point array --
/// stays on the concrete type, so adding a method here means both paths really
/// do have it.
pub trait NeighborIndex<S: ForestScalar> {
    /// Dimensionality of the indexed vectors.
    fn dim(&self) -> usize;

    /// Flat `n_queries * dim` queries in, flat `n_queries * k` feature IDs and
    /// squared distances out, `u32::MAX` and positive infinity where fewer than
    /// `k` neighbours were found.
    fn search_batch_with_distances(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<u32>, Vec<f32>), KdfError>;

    /// Copy the indexed descriptors for `feature_ids`, in request order, as a flat
    /// `feature_ids.len() * dim` row-major array.
    fn resolve_descriptors(&self, feature_ids: &[u32]) -> Result<Vec<S>, KdfError>;
}

/// The resident forest cannot fail at either operation, so its implementation
/// is a wrapper that says so: every error a caller has to handle comes from the
/// file-backed path.
impl<S: ForestScalar> NeighborIndex<S> for KdForest<S> {
    fn dim(&self) -> usize {
        Self::dim(self)
    }

    fn search_batch_with_distances(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<u32>, Vec<f32>), KdfError> {
        Ok(Self::search_batch_with_distances(
            self,
            queries,
            n_queries,
            k,
            max_leaf_checks,
            max_dist,
        ))
    }

    fn resolve_descriptors(&self, feature_ids: &[u32]) -> Result<Vec<S>, KdfError> {
        Self::resolve_descriptors(self, feature_ids)
    }
}

impl<S: ForestScalar + KdfScalar> NeighborIndex<S> for LazyKdForest<S> {
    fn dim(&self) -> usize {
        Self::dim(self)
    }

    fn search_batch_with_distances(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<u32>, Vec<f32>), KdfError> {
        Self::search_batch_with_distances(self, queries, n_queries, k, max_leaf_checks, max_dist)
    }

    fn resolve_descriptors(&self, feature_ids: &[u32]) -> Result<Vec<S>, KdfError> {
        Self::resolve_descriptors(self, feature_ids)
    }
}
