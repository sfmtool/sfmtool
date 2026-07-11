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

/// Per-tree reusable scratch for construction, so the recursion allocates
/// nothing per node: the partition's class tags and reordered ids, the median
/// selection's coordinate buffer, and the variance estimator's accumulators.
/// Buffers are (re)sized inside each helper; contents never carry between
/// nodes.
struct BuildScratch<S> {
    /// Per-id comparison class from the partition's first pass.
    classes: Vec<std::cmp::Ordering>,
    /// The partition's `[lt | eq | gt]` reordering, copied back into `ids`.
    reordered: Vec<u32>,
    /// Coordinate values for median selection.
    vals: Vec<S>,
    /// Variance estimator sums, `2 × dim` (means, then squares).
    sums: Vec<f64>,
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
    let mut scratch = BuildScratch {
        classes: Vec::new(),
        reordered: Vec::new(),
        vals: Vec::new(),
        sums: vec![0.0; 2 * dim],
    };
    if !point_ids.is_empty() {
        build_node(
            &mut nodes,
            points,
            dim,
            &mut point_ids,
            0,
            params,
            &mut rng,
            &mut scratch,
        );
    }
    Tree { nodes, point_ids }
}

/// Recursively build the subtree for `ids` (the slice of point ids belonging to
/// this node), whose first element sits at `offset` within the tree's full
/// `point_ids` array. Returns the arena index of the created node.
#[allow(clippy::too_many_arguments)]
fn build_node<S: ForestScalar>(
    nodes: &mut Vec<Node<S>>,
    points: &[S],
    dim: usize,
    ids: &mut [u32],
    offset: u32,
    params: &KdForestParams,
    rng: &mut StdRng,
    scratch: &mut BuildScratch<S>,
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

    let variances = estimate_variances(points, dim, ids, rng, &mut scratch.sums);
    let top = top_dims(variances, params.split_dim_candidates);
    let split_dim = top[rng.random_range(0..top.len())];
    let split_val = median_value::<S>(points, dim, split_dim, ids, &mut scratch.vals);
    let left_count = partition::<S>(points, dim, split_dim, split_val, ids, scratch);

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
    let left = build_node(nodes, points, dim, left_ids, offset, params, rng, scratch);
    let right = build_node(
        nodes,
        points,
        dim,
        right_ids,
        offset + left_count as u32,
        params,
        rng,
        scratch,
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
/// `sums` is the caller's `2 × dim` scratch (means, then squares); the
/// returned variances overwrite its first `dim` slots.
fn estimate_variances<'a, S: ForestScalar>(
    points: &[S],
    dim: usize,
    ids: &[u32],
    rng: &mut StdRng,
    sums: &'a mut [f64],
) -> &'a [f64] {
    let n = ids.len();
    let take = n.min(VARIANCE_SAMPLE);
    let (sum, sum_sq) = sums.split_at_mut(dim);
    sum.fill(0.0);
    sum_sq.fill(0.0);
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
    for d in 0..dim {
        let mean = sum[d] * inv;
        sum[d] = (sum_sq[d] * inv - mean * mean).max(0.0);
    }
    &sums[..dim]
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

/// Median coordinate value along `dim_idx` over the points in `ids`. `vals` is
/// the caller's reusable coordinate buffer.
fn median_value<S: ForestScalar>(
    points: &[S],
    dim: usize,
    dim_idx: usize,
    ids: &[u32],
    vals: &mut Vec<S>,
) -> S {
    vals.clear();
    vals.extend(ids.iter().map(|&id| points[id as usize * dim + dim_idx]));
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
///
/// The layout is `[ lt | eq | gt ]` with each class keeping its encounter
/// order (the order matters downstream: the child recursion's variance
/// sampling indexes into it). Built in two passes over reusable scratch — one
/// coordinate-comparison pass recording each id's class, one placement pass —
/// instead of three per-node `Vec`s; with millions of ids per tree the
/// allocator churn dominated construction.
fn partition<S: ForestScalar>(
    points: &[S],
    dim: usize,
    split_dim: usize,
    split_val: S,
    ids: &mut [u32],
    scratch: &mut BuildScratch<S>,
) -> usize {
    let n = ids.len();
    let target = n / 2; // >= 1 for n >= 2

    scratch.classes.clear();
    let mut n_lt = 0usize;
    let mut n_eq = 0usize;
    for &id in ids.iter() {
        let c = points[id as usize * dim + split_dim];
        let class = S::coord_cmp(c, split_val);
        match class {
            std::cmp::Ordering::Less => n_lt += 1,
            std::cmp::Ordering::Equal => n_eq += 1,
            std::cmp::Ordering::Greater => {}
        }
        scratch.classes.push(class);
    }

    let reordered = &mut scratch.reordered;
    reordered.resize(n, 0);
    let (mut i_lt, mut i_eq, mut i_gt) = (0usize, n_lt, n_lt + n_eq);
    for (&id, &class) in ids.iter().zip(&scratch.classes) {
        match class {
            std::cmp::Ordering::Less => {
                reordered[i_lt] = id;
                i_lt += 1;
            }
            std::cmp::Ordering::Equal => {
                reordered[i_eq] = id;
                i_eq += 1;
            }
            std::cmp::Ordering::Greater => {
                reordered[i_gt] = id;
                i_gt += 1;
            }
        }
    }
    ids.copy_from_slice(&reordered[..n]);

    // All `eq` points share `split_val`, so they are interchangeable: the
    // left/right boundary is just an index *inside* the eq run, placed to send
    // enough median-valued points left to reach `target`.
    let eq_to_left = target.saturating_sub(n_lt).min(n_eq);
    let left_count = n_lt + eq_to_left;

    // Guaranteed in `[1, n-1]` for n >= 2 (target <= n/2 < n, and at least one
    // median-valued point exists to seed the left side when `lt` is empty).
    left_count.clamp(1, n - 1)
}
