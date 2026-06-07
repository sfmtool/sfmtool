// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Randomized kd-tree construction (Muja & Lowe 2009, §3.1).
//!
//! Each tree is built independently from the full point set. At every internal
//! node the split dimension is chosen *at random from the `D` dimensions of
//! highest variance* (estimated from a bounded sample), and the split value is
//! the median along that dimension, with points equal to the median distributed
//! across both sides so the partition halves the data regardless of duplicate
//! coordinates. A per-tree seeded RNG (`StdRng`) makes builds reproducible.

use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;

use super::distance::ForestScalar;
use super::KdForestParams;

/// Upper bound on the per-node sample used to estimate coordinate variance.
/// The top-`D` selection only needs the relative ordering of variances, so a
/// bounded sample keeps construction close to `O(N log N)`.
const VARIANCE_SAMPLE: usize = 100;

/// A node in a single kd-tree.
///
/// Children are stored as arena indices (`u32`) into the tree's node vector
/// rather than boxed pointers, keeping nodes cache-friendly and the whole tree
/// trivially `Send`/`Sync` for shared queries.
#[derive(Clone, Copy, Debug)]
pub(super) enum Node<S: ForestScalar> {
    /// Internal split node.
    Internal {
        /// Dimension this node splits on.
        split_dim: u16,
        /// Split value (median along `split_dim`).
        split_val: S,
        /// Arena index of the near (`<= split_val`) child. Points exactly equal
        /// to `split_val` may live on either side (see [`partition`]), so a
        /// query equal to `split_val` must still explore both children.
        left: u32,
        /// Arena index of the far (`>= split_val`) child.
        right: u32,
    },
    /// Leaf bucket: a contiguous range `[start, start + len)` of `point_ids`.
    Leaf { start: u32, len: u32 },
}

/// A single randomized kd-tree: an arena of nodes plus a permutation of point
/// ids that the leaves index into by range.
pub(super) struct Tree<S: ForestScalar> {
    pub(super) nodes: Vec<Node<S>>,
    pub(super) point_ids: Vec<u32>,
}

/// Build a single tree over `point_ids` (a fresh `0..n_points` permutation that
/// this call rearranges in place to match the leaf layout).
///
/// Each tree gets its own `StdRng` seeded from `base_seed + tree_index`, so
/// builds are reproducible for a given seed regardless of thread count.
pub(super) fn build_tree<S: ForestScalar>(
    points: &[S],
    dim: usize,
    mut point_ids: Vec<u32>,
    params: &KdForestParams,
    seed: u64,
) -> Tree<S> {
    let mut nodes = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    if !point_ids.is_empty() {
        build_node(&mut nodes, points, dim, &mut point_ids, 0, params, &mut rng);
    }
    Tree { nodes, point_ids }
}

/// Recursively build the subtree for `ids` (the slice of point ids belonging to
/// this node), whose first element sits at `offset` within the tree's full
/// `point_ids` array. Returns the arena index of the created node.
fn build_node<S: ForestScalar>(
    nodes: &mut Vec<Node<S>>,
    points: &[S],
    dim: usize,
    ids: &mut [u32],
    offset: u32,
    params: &KdForestParams,
    rng: &mut StdRng,
) -> u32 {
    let n = ids.len();
    if n <= params.leaf_size {
        let idx = nodes.len() as u32;
        nodes.push(Node::Leaf {
            start: offset,
            len: n as u32,
        });
        return idx;
    }

    let variances = estimate_variances(points, dim, ids, rng);
    let top = top_dims(&variances, params.split_dim_candidates);
    let split_dim = top[rng.random_range(0..top.len())];
    let split_val = median_value::<S>(points, dim, split_dim, ids);
    let left_count = partition::<S>(points, dim, split_dim, split_val, ids);

    // Reserve this node's slot before building children so child arena indices
    // are assigned after it (children are appended during their recursion). The
    // sentinel is overwritten below once the children exist; it is never read
    // (and the MAX range would trip bounds checks if it ever were).
    let node_idx = nodes.len() as u32;
    nodes.push(Node::Leaf {
        start: u32::MAX,
        len: u32::MAX,
    });

    let (left_ids, right_ids) = ids.split_at_mut(left_count);
    let left = build_node(nodes, points, dim, left_ids, offset, params, rng);
    let right = build_node(
        nodes,
        points,
        dim,
        right_ids,
        offset + left_count as u32,
        params,
        rng,
    );

    nodes[node_idx as usize] = Node::Internal {
        split_dim: split_dim as u16,
        split_val,
        left,
        right,
    };
    node_idx
}

/// Estimate per-dimension variance from a bounded random sample of `ids`.
fn estimate_variances<S: ForestScalar>(
    points: &[S],
    dim: usize,
    ids: &[u32],
    rng: &mut StdRng,
) -> Vec<f64> {
    let n = ids.len();
    let take = n.min(VARIANCE_SAMPLE);
    let mut sum = vec![0.0f64; dim];
    let mut sum_sq = vec![0.0f64; dim];
    for s in 0..take {
        // Use every id when the node is small; otherwise sample (repeats are
        // harmless — we only need relative variance ordering).
        let id = if take == n {
            ids[s]
        } else {
            ids[rng.random_range(0..n)]
        };
        let base = id as usize * dim;
        for d in 0..dim {
            let v = points[base + d].to_f64();
            sum[d] += v;
            sum_sq[d] += v * v;
        }
    }
    let inv = 1.0 / take as f64;
    (0..dim)
        .map(|d| {
            let mean = sum[d] * inv;
            (sum_sq[d] * inv - mean * mean).max(0.0)
        })
        .collect()
}

/// Return the `pool` dimensions with the largest variance, ordered by
/// descending variance (ties broken by lower dimension index).
///
/// Uses `select_nth_unstable_by` to partition out the top `pool` in `O(dim)`,
/// then sorts only those few — rather than a full `O(dim log dim)` sort of every
/// dimension. The comparator is a strict total order (variance, then index), so
/// the selected set and its final order are identical to a full sort.
fn top_dims(variances: &[f64], pool: usize) -> Vec<usize> {
    let pool = pool.max(1).min(variances.len());
    let mut idx: Vec<usize> = (0..variances.len()).collect();
    let cmp = |&a: &usize, &b: &usize| {
        variances[b]
            .partial_cmp(&variances[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    };
    if pool < idx.len() {
        // Place the top `pool` indices in idx[0..pool] (unordered among themselves).
        idx.select_nth_unstable_by(pool - 1, cmp);
        idx.truncate(pool);
    }
    idx.sort_unstable_by(cmp);
    idx
}

/// Median coordinate value along `dim_idx` over the points in `ids`.
fn median_value<S: ForestScalar>(points: &[S], dim: usize, dim_idx: usize, ids: &[u32]) -> S {
    let mut vals: Vec<S> = ids
        .iter()
        .map(|&id| points[id as usize * dim + dim_idx])
        .collect();
    let mid = vals.len() / 2;
    vals.select_nth_unstable_by(mid, |&a, &b| S::coord_cmp(a, b));
    vals[mid]
}

/// Partition `ids` in place around `split_val` on `split_dim`, returning the
/// number of ids placed on the left (`<= split_val`) side.
///
/// Points equal to the median are distributed across both sides so the split
/// lands as close to `n / 2` as possible — this keeps the tree balanced even
/// when many points share the median value on the split axis (common for `u8`
/// descriptors, whose coordinates take only 256 distinct values). Both sides are
/// guaranteed non-empty for `n >= 2`.
fn partition<S: ForestScalar>(
    points: &[S],
    dim: usize,
    split_dim: usize,
    split_val: S,
    ids: &mut [u32],
) -> usize {
    let n = ids.len();
    let target = n / 2; // >= 1 for n >= 2

    let mut lt = Vec::new();
    let mut eq = Vec::new();
    let mut gt = Vec::new();
    for &id in ids.iter() {
        let c = points[id as usize * dim + split_dim];
        match S::coord_cmp(c, split_val) {
            std::cmp::Ordering::Less => lt.push(id),
            std::cmp::Ordering::Equal => eq.push(id),
            std::cmp::Ordering::Greater => gt.push(id),
        }
    }

    // Lay the ids out as [ lt | eq | gt ]. All `eq` points share `split_val`, so
    // they are interchangeable: the left/right boundary is just an index *inside*
    // the eq run, placed to send enough median-valued points left to reach
    // `target`. No need to physically split `eq`.
    let eq_to_left = target.saturating_sub(lt.len()).min(eq.len());
    let left_count = lt.len() + eq_to_left;
    for (slot, id) in ids.iter_mut().zip(lt.into_iter().chain(eq).chain(gt)) {
        *slot = id;
    }

    // Guaranteed in `[1, n-1]` for n >= 2 (target <= n/2 < n, and at least one
    // median-valued point exists to seed the left side when `lt` is empty).
    left_count.clamp(1, n - 1)
}
