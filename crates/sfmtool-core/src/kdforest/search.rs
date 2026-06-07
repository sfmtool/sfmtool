// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared best-bin-first (BBF) search across the forest.
//!
//! All `T` trees are searched together with a *single* min-priority queue keyed
//! by a lower bound on the query-to-cell distance (Beis & Lowe 1997). Each tree
//! is descended once to seed the queue; thereafter the closest unexplored
//! branch is expanded until a fixed budget of unique distance computations
//! (`max_leaf_checks`, the precision dial) is spent or nothing closer than the
//! current `k`-th best can remain. A `checked` bitset ensures each point's
//! distance is computed at most once across all trees.
//!
//! ## Approximation
//!
//! The branch lower bound is accumulated additively (`bound + (q - split)²` per
//! far turn) — the priority bound of FLANN's randomized kd-tree forest. It is
//! **not admissible**: when the same axis is re-split deeper along a far path it
//! double-counts that axis and can *over-estimate* the true query-to-cell
//! distance. The prune (`bound > prune_threshold`) can therefore skip a branch
//! that holds a true neighbor, so the forest is approximate even at an unlimited
//! budget — the same trade FLANN's `FLANN_INDEX_KDTREE` makes. An exact bound
//! would replace, not add, the per-axis component, needing per-axis state on
//! every queue entry; the only exact configuration here is a single leaf.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::build::Node;
use super::distance::ForestScalar;

/// A nearest-neighbor result: a point index and its squared distance (as `f32`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Index of the point in the forest's original point set.
    pub index: u32,
    /// Squared Euclidean distance from the query to this point.
    pub dist_sq: f32,
}

/// Per-query search counters, surfaced when `SFMTOOL_KDFOREST_STATS=1`.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct QueryStats {
    /// Unique point distances computed (the quantity `max_leaf_checks` bounds).
    pub checks: u64,
    /// Branches pushed onto the shared priority queue.
    pub pushes: u64,
    /// Branches popped from the shared priority queue.
    pub pops: u64,
}

/// A packed bitset over point ids for cross-tree dedup within one query.
///
/// Reused across queries on a rayon worker (see [`SearchScratch`]): it tracks
/// which words were touched so [`clear`](Checked::clear) costs O(words touched)
/// — i.e. O(checks) — instead of re-zeroing the whole `N/64`-word buffer every
/// query.
struct Checked {
    bits: Vec<u64>,
    touched: Vec<u32>,
}

impl Checked {
    fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; n.div_ceil(64)],
            touched: Vec::new(),
        }
    }

    /// Mark point `i` as checked; returns `true` if it was not already marked.
    #[inline]
    fn insert(&mut self, i: u32) -> bool {
        let word = i as usize / 64;
        let bit = 1u64 << (i % 64);
        if self.bits[word] == 0 {
            // First bit set in this word this query — record it for cheap reset.
            self.touched.push(word as u32);
        }
        let was_set = self.bits[word] & bit != 0;
        self.bits[word] |= bit;
        !was_set
    }

    /// Zero only the words touched since the last clear.
    fn clear(&mut self) {
        for &word in &self.touched {
            self.bits[word as usize] = 0;
        }
        self.touched.clear();
    }
}

/// Per-worker reusable scratch for the query path.
///
/// Threaded through a batch via rayon's `for_each_init` so the BBF priority
/// queue, the dedup bitset, and the result set are allocated once per worker and
/// reset (not reallocated) between queries — making the query inner loop free of
/// per-query heap allocation.
pub(super) struct SearchScratch<S: ForestScalar> {
    queue: BinaryHeap<(Reverse<S::Dist>, u32, u32)>,
    checked: Checked,
    result: BoundedResult<S>,
}

impl<S: ForestScalar> SearchScratch<S> {
    pub(super) fn new(n_points: usize) -> Self {
        Self {
            queue: BinaryHeap::new(),
            checked: Checked::new(n_points),
            result: BoundedResult::new(),
        }
    }

    /// Reset to empty for a fresh query, retaining allocated capacity.
    fn reset(&mut self, k: usize, cutoff: S::Dist) {
        self.queue.clear();
        self.checked.clear();
        self.result.reset(k, cutoff);
    }

    /// Write the current results into caller-owned output rows (length `k`),
    /// nearest first; slots past the found count are left untouched (the caller
    /// pre-fills them with `u32::MAX` / `f32::INFINITY` padding).
    pub(super) fn write_results_into(&self, out_idx: &mut [u32], out_dist: &mut [f32]) {
        self.result.write_into(out_idx, out_dist);
    }
}

/// A bounded set of the `k` nearest candidates seen so far, kept sorted
/// ascending by distance. Rejects anything beyond the cutoff or the current
/// `k`-th best, so it doubles as the `max_dist` filter. Lives in
/// [`SearchScratch`] and is reused across queries.
struct BoundedResult<S: ForestScalar> {
    k: usize,
    cutoff: S::Dist,
    items: Vec<(u32, S::Dist)>,
}

impl<S: ForestScalar> BoundedResult<S> {
    fn new() -> Self {
        Self {
            k: 0,
            cutoff: S::MAX_DIST,
            items: Vec::new(),
        }
    }

    /// Reconfigure for a fresh query, retaining the items buffer's capacity.
    fn reset(&mut self, k: usize, cutoff: S::Dist) {
        self.k = k;
        self.cutoff = cutoff;
        self.items.clear();
        self.items.reserve(k);
    }

    /// Whether the result already holds its full `k` candidates. Also guards the
    /// `items[k - 1]` reads below against `k == 0` (where `len == k == 0` would
    /// otherwise underflow) — defensive, since callers skip `k == 0` queries.
    #[inline]
    fn is_full(&self) -> bool {
        self.k != 0 && self.items.len() == self.k
    }

    /// The distance beyond which no candidate can improve the result: the
    /// `k`-th best once full, otherwise the cutoff.
    #[inline]
    fn worst_dist(&self) -> S::Dist {
        if self.is_full() {
            self.items[self.k - 1].1
        } else {
            self.cutoff
        }
    }

    /// Consider point `idx` at squared distance `d`, inserting it if it ranks.
    #[inline]
    fn consider(&mut self, idx: u32, d: S::Dist) {
        if d > self.cutoff {
            return;
        }
        if self.is_full() && d >= self.items[self.k - 1].1 {
            return;
        }
        let pos = self.items.partition_point(|&(_, dd)| dd <= d);
        self.items.insert(pos, (idx, d));
        if self.items.len() > self.k {
            self.items.pop();
        }
    }

    /// Allocate a `Vec<Neighbor>` (for the single-query public API).
    fn to_neighbors(&self) -> Vec<Neighbor> {
        self.items
            .iter()
            .map(|&(index, d)| Neighbor {
                index,
                dist_sq: S::dist_sq_to_f32(d),
            })
            .collect()
    }

    /// Write results into caller-owned slices without allocating.
    fn write_into(&self, out_idx: &mut [u32], out_dist: &mut [f32]) {
        for (i, &(index, d)) in self.items.iter().enumerate() {
            out_idx[i] = index;
            out_dist[i] = S::dist_sq_to_f32(d);
        }
    }
}

impl<S: ForestScalar> super::KdForest<S> {
    /// Find up to `k` approximate nearest neighbors of `query`.
    ///
    /// `query` has length [`dim`](super::KdForest::dim). `max_leaf_checks` is
    /// the precision dial: a *soft* budget on unique distance computations,
    /// matching FLANN's `checks`. It is enforced at leaf granularity and only
    /// after every tree has been descended once to seed the queue, so the
    /// actual count can exceed it by up to one leaf per seeded tree. `max_dist`
    /// is an optional Euclidean cutoff (`None` = unbounded). Results are sorted
    /// by ascending distance; fewer than `k` may be returned. The search is
    /// approximate (see the bound note in `descend`).
    #[must_use]
    pub fn search(
        &self,
        query: &[S],
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Vec<Neighbor> {
        let mut scratch = SearchScratch::new(self.n_points);
        let mut stats = QueryStats::default();
        self.run_query(
            query,
            k,
            max_leaf_checks,
            max_dist,
            &mut scratch,
            &mut stats,
        );
        scratch.result.to_neighbors()
    }

    /// Run one query, leaving the results in `scratch.result` and per-query
    /// counters in `stats`. The batch path supplies one reused scratch per
    /// rayon worker and reads the results via
    /// [`SearchScratch::write_results_into`].
    pub(super) fn run_query(
        &self,
        query: &[S],
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
        scratch: &mut SearchScratch<S>,
        stats: &mut QueryStats,
    ) {
        assert_eq!(query.len(), self.dim, "query length must equal dim");
        let cutoff = max_dist.map(S::cutoff_sq).unwrap_or(S::MAX_DIST);
        scratch.reset(k, cutoff);
        if k == 0 || self.n_points == 0 {
            return;
        }

        let mut search = Search {
            forest: self,
            query,
            scratch,
            stats,
        };

        // Seed: descend every non-empty tree once from its root (node 0).
        for (ti, tree) in self.trees.iter().enumerate() {
            if !tree.nodes.is_empty() {
                search.descend(ti as u32, 0, S::ZERO_DIST);
            }
        }

        // Best-bin-first expansion across all trees, popping the closest branch.
        while let Some((Reverse(branch_lb), ti, node)) = search.scratch.queue.pop() {
            search.stats.pops += 1;
            if search.stats.checks >= max_leaf_checks as u64 {
                break;
            }
            if branch_lb > search.scratch.result.worst_dist() {
                break;
            }
            search.descend(ti, node, branch_lb);
        }
    }
}

/// Mutable per-query search state: a shared forest (`&`) plus the reused
/// [`SearchScratch`] (priority queue, dedup bitset, result set) and the stats
/// sink. Bundling this keeps [`descend`](Search::descend) a two-line signature.
struct Search<'a, S: ForestScalar> {
    forest: &'a super::KdForest<S>,
    query: &'a [S],
    scratch: &'a mut SearchScratch<S>,
    stats: &'a mut QueryStats,
}

impl<S: ForestScalar> Search<'_, S> {
    /// Descend the near chain from `node` in tree `tree_idx` (entered with lower
    /// bound `node_lb`), enqueueing each far child and scanning the reached leaf.
    fn descend(&mut self, tree_idx: u32, mut node: u32, node_lb: S::Dist) {
        let tree = &self.forest.trees[tree_idx as usize];
        loop {
            match tree.nodes[node as usize] {
                Node::Internal {
                    split_dim,
                    split_val,
                    left,
                    right,
                } => {
                    let q = self.query[split_dim as usize];
                    let (near, far) = match S::coord_cmp(q, split_val) {
                        std::cmp::Ordering::Greater => (right, left),
                        _ => (left, right),
                    };
                    // Additive FLANN bound for the far child (approximate — see
                    // the module-level "Approximation" note). Skip it if it
                    // already cannot beat the current k-th best.
                    let far_child_lb = node_lb + S::axis_dist_sq(q, split_val);
                    if far_child_lb <= self.scratch.result.worst_dist() {
                        self.scratch
                            .queue
                            .push((Reverse(far_child_lb), tree_idx, far));
                        self.stats.pushes += 1;
                    }
                    node = near;
                }
                Node::Leaf { start, len } => {
                    let dim = self.forest.dim;
                    for &p in &tree.point_ids[start as usize..(start + len) as usize] {
                        if self.scratch.checked.insert(p) {
                            self.stats.checks += 1;
                            let base = p as usize * dim;
                            let d = S::dist_sq(self.query, &self.forest.points[base..base + dim]);
                            self.scratch.result.consider(p, d);
                        }
                    }
                    return;
                }
            }
        }
    }
}
