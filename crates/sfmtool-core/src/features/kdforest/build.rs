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
//!
//! Because the partition halves every node exactly, a tree's shape depends on
//! its point count alone. Its draws are therefore taken up front, in the
//! depth-first order a node-by-node build takes them ([`TreeDraws`]), and the
//! subtrees of large nodes are then built in parallel, each into the stretch
//! of the node arena depth-first order gives it, so the tree is the same at
//! any thread count.
//!
//! Specified in `specs/core/features/randomized-kdtree-forest.md`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;

use super::distance::ForestScalar;
use super::KdForestParams;
use crate::progress::{Cancelled, Progress};

#[cfg(test)]
mod tests;

/// Upper bound on the per-node sample used to estimate coordinate variance.
/// The top-`D` selection only needs the relative ordering of variances, so a
/// bounded sample keeps construction close to `O(N log N)`.
const VARIANCE_SAMPLE: usize = 100;

/// How many reports a whole forest build makes, at most.
///
/// The unit of progress is a **point placed in a leaf**, of which a build makes
/// `num_trees * n_points`; a report apiece would be millions of events. So the
/// counter is shared and the report is made only by the leaf that carries the
/// count across the next boundary.
const REPORTS_PER_BUILD: u64 = 200;

/// The one counter every tree of a build reports into.
///
/// Per tree would be the natural unit and is the wrong one here: the trees are
/// built in parallel, so four of them over three seconds are four steps that
/// all land at the end. Points placed is what moves evenly, and it is cheap at
/// this granularity -- one relaxed `fetch_add` per leaf, which is one per
/// `leaf_size` points rather than one per point.
///
/// Leaves are placed on many threads at once, so two leaves can cross two
/// boundaries in one order and reach the sink in the other. The reports are
/// therefore made under a lock that keeps the count last reported, and a
/// report of a smaller count is dropped, so the fractions a caller sees never
/// fall. The lock is taken only at a boundary, at most
/// [`REPORTS_PER_BUILD`] times.
pub(super) struct BuildProgress<'p, 'a> {
    progress: &'p Progress<'a>,
    /// Points placed in a leaf so far, across every tree of the build.
    placed: AtomicU64,
    /// The count of the last report made, which the next must exceed.
    reported: Mutex<u64>,
    /// `num_trees * n_points`, which is what `placed` runs up to.
    total: u64,
    /// How many points pass between two reports.
    step: u64,
}

impl<'p, 'a> BuildProgress<'p, 'a> {
    /// The counter for a build of `num_trees` trees over `n_points` points.
    pub(super) fn new(progress: &'p Progress<'a>, num_trees: usize, n_points: usize) -> Self {
        let total = (num_trees as u64).saturating_mul(n_points as u64).max(1);
        Self {
            progress,
            placed: AtomicU64::new(0),
            reported: Mutex::new(0),
            total,
            step: (total / REPORTS_PER_BUILD).max(1),
        }
    }

    /// Count `len` points into a leaf, reporting when the count crosses a
    /// boundary.
    fn placed(&self, len: u64) {
        let before = self.placed.fetch_add(len, Ordering::Relaxed);
        let after = before + len;
        if before / self.step != after / self.step {
            let mut reported = self.reported.lock().unwrap_or_else(PoisonError::into_inner);
            if after > *reported {
                *reported = after;
                self.progress.set_fraction(after as f32 / self.total as f32);
            }
        }
    }

    /// Whether the caller has asked the build to stop. One relaxed load, read
    /// once per leaf.
    fn is_cancelled(&self) -> bool {
        self.progress.is_cancelled()
    }
}

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

/// The fewest points a node needs before its two subtrees are built at the
/// same time, as two rayon tasks.
///
/// Below it a subtree is built by the task that reached it, one node after
/// another. A subtree of this size is a few hundred microseconds of work,
/// which is large next to what a `rayon::join` costs, and a frame of 8,192
/// SIFT descriptors still splits into sixteen tasks per tree.
pub(super) const PARALLEL_MIN_POINTS: usize = 1024;

/// Reusable scratch for construction, so the recursion allocates nothing per
/// node: the partition's class tags and reordered ids, the median selection's
/// coordinate buffer, and the variance estimator's accumulators. Buffers are
/// (re)sized inside each helper; contents never carry between nodes. Each task
/// of a build has its own.
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

impl<S> BuildScratch<S> {
    fn new(dim: usize) -> Self {
        Self {
            classes: Vec::new(),
            reordered: Vec::new(),
            vals: Vec::new(),
            sums: vec![0.0; 2 * dim],
        }
    }
}

/// Every draw one tree's build takes from its RNG, made before any point is
/// read.
///
/// What an internal node draws depends only on how many points it holds, and
/// [`partition`] always sends `n / 2` of them left (see [`left_len`]), so the
/// tree's shape, and with it every draw, is known from `n_points` alone.
/// Drawing them up front, in the depth-first order a node-by-node build
/// would draw them, lets subtrees be built in any order and at the same time
/// and still see the sequence that order gives them.
struct TreeDraws {
    /// One per internal node, in depth-first order (a node, then its left
    /// subtree, then its right subtree), so a node's left child, when it is
    /// internal, is the next entry.
    nodes: Vec<NodeDraws>,
    /// The variance-sample positions of every internal node, concatenated in
    /// the same order.
    positions: Vec<u32>,
}

/// One internal node's draws.
struct NodeDraws {
    /// Where its variance-sample positions sit in [`TreeDraws::positions`]:
    /// [`VARIANCE_SAMPLE`] positions within the node's ids, drawn with
    /// repetition, or none when the node is small enough to use every id.
    positions: std::ops::Range<u32>,
    /// Index of the split dimension among the node's top-variance candidates.
    split_choice: usize,
    /// Index in [`TreeDraws::nodes`] of its right child's draws, and the
    /// number of internal nodes before it in depth-first order.
    right: u32,
}

impl TreeDraws {
    /// Draw for a tree of `n_points` points, from an RNG seeded with `seed`.
    ///
    /// Each internal node draws its variance-sample positions, then its split
    /// choice among `candidates` dimensions, before its left subtree draws,
    /// whose draws come before its right subtree's.
    fn new(n_points: usize, leaf_size: usize, candidates: usize, seed: u64) -> Self {
        fn walk(
            draws: &mut TreeDraws,
            rng: &mut StdRng,
            n: usize,
            leaf_size: usize,
            candidates: usize,
        ) {
            if n <= leaf_size {
                return;
            }
            let start = draws.positions.len() as u32;
            if n > VARIANCE_SAMPLE {
                // Repeats are harmless: only the relative variance ordering is
                // needed. Each position is below `n`, so it fits a `u32` as
                // the point ids do.
                draws
                    .positions
                    .extend((0..VARIANCE_SAMPLE).map(|_| rng.random_range(0..n) as u32));
            }
            let end = draws.positions.len() as u32;
            let k = draws.nodes.len();
            draws.nodes.push(NodeDraws {
                positions: start..end,
                split_choice: rng.random_range(0..candidates),
                right: 0,
            });
            let left = left_len(n);
            walk(draws, rng, left, leaf_size, candidates);
            draws.nodes[k].right = draws.nodes.len() as u32;
            walk(draws, rng, n - left, leaf_size, candidates);
        }

        let mut draws = TreeDraws {
            nodes: Vec::new(),
            positions: Vec::new(),
        };
        let mut rng = StdRng::seed_from_u64(seed);
        walk(&mut draws, &mut rng, n_points, leaf_size, candidates);
        draws
    }
}

/// How many of an internal node's `n` points go to its left child.
///
/// It depends on `n` alone, since [`partition`] places exactly `n / 2` there.
fn left_len(n: usize) -> usize {
    n / 2
}

/// What every node of one tree's build reads.
struct TreeBuild<'b, 'p, 'a, S> {
    points: &'b [S],
    dim: usize,
    params: &'b KdForestParams,
    draws: &'b TreeDraws,
    progress: &'b BuildProgress<'p, 'a>,
    /// The fewest points a node needs before its subtrees are built as two
    /// rayon tasks; `usize::MAX` builds the whole tree in one.
    parallel_min_points: usize,
}

/// Build a single tree over `point_ids` (a fresh `0..n_points` permutation that
/// this call rearranges in place to match the leaf layout).
///
/// Each tree's randomness comes from its own `StdRng` seeded with `seed`
/// (`base_seed + tree_index`), drawn up front in depth-first order (see
/// [`TreeDraws`]). A node of at least `parallel_min_points` points builds its
/// two subtrees as two rayon tasks, each writing into the stretch of the
/// node arena that depth-first order gives it. So the tree -- split
/// dimensions, split values, node order and leaf contents -- is the same
/// whatever `parallel_min_points` is, and so whatever the thread count.
///
/// `progress` is the counter every tree of the build shares; it is read and
/// written once per leaf, and a build that is asked to stop hands back
/// [`Cancelled`] rather than a tree whose leaves cover some of the points.
pub(super) fn build_tree<S: ForestScalar>(
    points: &[S],
    dim: usize,
    mut point_ids: Vec<u32>,
    params: &KdForestParams,
    seed: u64,
    progress: &BuildProgress<'_, '_>,
    parallel_min_points: usize,
) -> Result<Tree<S>, Cancelled> {
    let n = point_ids.len();
    if n == 0 {
        return Ok(Tree {
            nodes: Vec::new(),
            point_ids,
        });
    }
    // The number of candidates `top_dims` returns.
    let candidates = params.split_dim_candidates.max(1).min(dim);
    let draws = TreeDraws::new(n, params.leaf_size, candidates, seed);
    let build = TreeBuild {
        points,
        dim,
        params,
        draws: &draws,
        progress,
        parallel_min_points,
    };
    // A binary tree has one more leaf than it has internal nodes. Every slot
    // is overwritten by the build; the sentinel is never read (and the MAX
    // range would trip bounds checks if it ever were).
    let mut nodes = vec![
        Node::Leaf {
            start: u32::MAX,
            len: u32::MAX,
        };
        2 * draws.nodes.len() + 1
    ];
    let mut scratch = BuildScratch::new(dim);
    build_node(&build, &mut nodes, 0, &mut point_ids, 0, 0, &mut scratch)?;
    Ok(Tree { nodes, point_ids })
}

/// Recursively build the subtree for `ids` (the slice of point ids belonging to
/// this node), whose first element sits at `offset` within the tree's full
/// `point_ids` array.
///
/// `nodes` is the subtree's stretch of the arena, this node first, and `base`
/// is that first slot's arena index; `k` is the index of this node's draws in
/// [`TreeDraws::nodes`] when it is internal.
fn build_node<S: ForestScalar>(
    build: &TreeBuild<'_, '_, '_, S>,
    nodes: &mut [Node<S>],
    base: u32,
    ids: &mut [u32],
    offset: u32,
    k: usize,
    scratch: &mut BuildScratch<S>,
) -> Result<(), Cancelled> {
    let n = ids.len();
    if n <= build.params.leaf_size {
        // The leaf is where a build says where it has got to and hears whether
        // it should stop: it is the one place the recursion retires points
        // rather than passing them down, so the count it keeps is the work
        // behind it.
        if build.progress.is_cancelled() {
            return Err(Cancelled);
        }
        build.progress.placed(n as u64);
        debug_assert_eq!(nodes.len(), 1);
        nodes[0] = Node::Leaf {
            start: offset,
            len: n as u32,
        };
        return Ok(());
    }

    let (points, dim) = (build.points, build.dim);
    let draws = &build.draws.nodes[k];
    let positions =
        &build.draws.positions[draws.positions.start as usize..draws.positions.end as usize];
    let variances = estimate_variances(points, dim, ids, positions, &mut scratch.sums);
    let top = top_dims(variances, build.params.split_dim_candidates);
    let split_dim = top[draws.split_choice];
    let split_val = median_value::<S>(points, dim, split_dim, ids, &mut scratch.vals);
    let left_count = partition::<S>(points, dim, split_dim, split_val, ids, scratch);
    // The draws were made for a tree of this shape; `partition` says why it
    // holds.
    assert_eq!(left_count, left_len(n), "partition must halve the node");

    // Depth-first order puts the left subtree straight after this node and
    // the right one after that. The left subtree has `right - (k + 1)`
    // internal nodes, and so one more leaf than that.
    let right_k = draws.right as usize;
    let left_slots = 2 * (right_k - (k + 1)) + 1;
    let (this, children) = nodes.split_first_mut().expect("a node has a slot");
    let (left_nodes, right_nodes) = children.split_at_mut(left_slots);
    let left = base + 1;
    let right = left + left_slots as u32;
    *this = Node::Internal {
        split_dim: split_dim as u16,
        split_val,
        left,
        right,
    };

    let (left_ids, right_ids) = ids.split_at_mut(left_count);
    let right_offset = offset + left_count as u32;
    if n >= build.parallel_min_points {
        let (left_built, right_built) = rayon::join(
            || build_node(build, left_nodes, left, left_ids, offset, k + 1, scratch),
            || {
                let mut right_scratch = BuildScratch::new(dim);
                build_node(
                    build,
                    right_nodes,
                    right,
                    right_ids,
                    right_offset,
                    right_k,
                    &mut right_scratch,
                )
            },
        );
        left_built?;
        right_built
    } else {
        build_node(build, left_nodes, left, left_ids, offset, k + 1, scratch)?;
        build_node(
            build,
            right_nodes,
            right,
            right_ids,
            right_offset,
            right_k,
            scratch,
        )
    }
}

/// Estimate per-dimension variance over the ids of a node at `positions` (its
/// variance sample from [`TreeDraws`]), or over every id when `positions` is
/// empty. `sums` is the caller's `2 × dim` scratch (means, then squares); the
/// returned variances overwrite its first `dim` slots.
fn estimate_variances<'a, S: ForestScalar>(
    points: &[S],
    dim: usize,
    ids: &[u32],
    positions: &[u32],
    sums: &'a mut [f64],
) -> &'a [f64] {
    let (sum, sum_sq) = sums.split_at_mut(dim);
    sum.fill(0.0);
    sum_sq.fill(0.0);
    let mut accumulate = |id: u32| {
        let base = id as usize * dim;
        for d in 0..dim {
            let v = points[base + d].to_f64();
            sum[d] += v;
            sum_sq[d] += v * v;
        }
    };
    let take = if positions.is_empty() {
        ids.iter().for_each(|&id| accumulate(id));
        ids.len()
    } else {
        positions.iter().for_each(|&p| accumulate(ids[p as usize]));
        positions.len()
    };
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
/// lands on `n / 2` exactly — this keeps the tree balanced even when many
/// points share the median value on the split axis (common for `u8`
/// descriptors, whose coordinates take only 256 distinct values) — so both
/// sides are non-empty for `n >= 2`.
///
/// Exactly `n / 2` because `split_val` is the element of rank `n / 2` under
/// `coord_cmp`, a total order for both scalars: at most `n / 2` points are
/// below it and at least `n / 2 + 1` are at or below it, so the `eq` run
/// always holds enough points to bring the left side to `n / 2`.
/// [`left_len`] depends on this.
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
