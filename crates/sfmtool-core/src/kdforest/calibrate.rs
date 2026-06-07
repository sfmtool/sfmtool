// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Optional `max_leaf_checks` (L_max) precision calibration.
//!
//! Given a sample of queries with their *exact* nearest neighbors (one
//! brute-force pass on a subset), binary-search the smallest `max_leaf_checks`
//! whose measured precision — the fraction of sample queries whose exact NN is
//! the forest's top-1 result — meets a target. This is the narrow slice of the
//! paper's auto-tuning we support: `T`, `D`, and `leaf_size` stay fixed; only
//! the budget is fitted.

use super::distance::ForestScalar;
use super::KdForest;

impl<S: ForestScalar> KdForest<S> {
    /// Smallest `max_leaf_checks` reaching `target_precision` on the sample.
    ///
    /// `sample_queries` is a flat `n_queries * dim` array; `exact_nn[i]` is the
    /// true nearest-neighbor index of query `i` (e.g. from a brute-force pass).
    /// `target_precision` is in `[0, 1]`.
    ///
    /// Precision is monotone non-decreasing in the budget, so a binary search
    /// over `[1, n_points]` is valid. This holds despite the inadmissible search
    /// bound: raising only `max_leaf_checks` makes the BBF loop visit a *superset*
    /// of points in the *same* deterministic heap order, and the result set only
    /// improves — so a query's top-1 can never get worse. Returns `n_points` (an
    /// effectively exhaustive budget) if the target is never met.
    #[must_use]
    pub fn calibrate_max_leaf_checks(
        &self,
        sample_queries: &[S],
        exact_nn: &[u32],
        target_precision: f64,
    ) -> usize {
        let n_queries = exact_nn.len();
        assert_eq!(
            sample_queries.len(),
            n_queries * self.dim,
            "sample_queries length must be n_queries * dim"
        );
        if n_queries == 0 || self.n_points == 0 {
            return 1;
        }

        let measure = |budget: usize| -> f64 {
            let found = self.search_batch(sample_queries, n_queries, 1, budget, None);
            let hits = (0..n_queries).filter(|&i| found[i] == exact_nn[i]).count();
            hits as f64 / n_queries as f64
        };

        // Binary search the smallest budget whose precision meets the target.
        let mut lo = 1usize;
        let mut hi = self.n_points;
        if measure(hi) < target_precision {
            return hi;
        }
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if measure(mid) >= target_precision {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }
}
