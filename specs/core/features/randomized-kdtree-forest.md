# Randomized KD-Tree Forest

A randomized kd-tree forest finds the approximate nearest neighbors of a query
vector among a large set of high-dimensional vectors, such as 128-dimension SIFT
descriptors, much faster than comparing the query with every candidate. It
builds several kd-trees whose split dimensions are chosen at random among the
highest-variance axes, then searches all the trees together with one shared
priority queue, stopping once a budget of distance computations is spent. This
spec covers the in-memory forest; a forest can also be saved to a file and
searched while only part of it is held in memory, as specified in
[lazy-kdforest-query.md](lazy-kdforest-query.md).

## Motivation

The most expensive step in descriptor matching is finding, for each query
descriptor, its nearest neighbor(s) among a large set of candidates. The
exhaustive scan in `features/feature_match/descriptor.rs` (`find_best_match`,
`find_best_match_contiguous`) is fine for many use cases; the randomized
kd-tree forest is the approximate alternative for large feature counts. It also
finds image patches: the patch constellation query
([kdf-constellation-query.md](kdf-constellation-query.md)) looks up the SIFT
keypoints around a pixel to find which other images hold the patch, and the
bench's descriptor search ([editable-track.md](../bench/editable-track.md))
runs that query through the file-backed `LazyKdForestU8`.

`spatial.rs` already wraps an exact kd-tree for 2D/3D point clouds; exact
kd-trees degenerate to near-linear search in the 128 dimensions of a SIFT
descriptor. The randomized kd-tree forest is the high-dimensional approximate
counterpart, trading a small, controllable loss in accuracy for one to three
orders of magnitude in speed. A pure-Rust implementation (rather than wrapping
FLANN/nanoflann) gives us a shared, reusable ANN index that any matcher
(descriptor, sweep, polar, flow-seeded) can build once and query many times. It
mirrors the optical-flow ([optical-flow.md](optical-flow.md)) and SIFT
([sift.md](sift.md)) implementations: pure Rust, no external ANN library, an
AVX2/SSE2 SIMD inner loop, and rayon for both build and batched query.

Its first consumer is the background-floor track-cluster matcher behind
`sfm match --cluster` ([track-cluster-matching.md](track-cluster-matching.md)),
which uses the forest for per-descriptor k-NN — see
`crates/sfmtool-core/src/features/cluster_match/` and the Python wrapper
`src/sfmtool/feature_match/_cluster_matching.py`.

This spec defines library types in sfmtool-core, independent of any on-disk
layout. It follows the codebase's existing descriptor conventions: descriptors
are arbitrary-length `u8` vectors and distances are squared L2 computed in integer space
(`descriptor_distance_l2_squared`), with `sqrt` taken only for the neighbors
actually returned.

## Rust API

The forest is in
[`crates/sfmtool-core/src/features/kdforest/`](../../../crates/sfmtool-core/src/features/kdforest/):
[`mod.rs`](../../../crates/sfmtool-core/src/features/kdforest/mod.rs) holds
`KdForestParams`, `KdForest` and the batched search,
[`build.rs`](../../../crates/sfmtool-core/src/features/kdforest/build.rs) the
tree construction,
[`search.rs`](../../../crates/sfmtool-core/src/features/kdforest/search.rs) the
shared-queue search,
[`distance.rs`](../../../crates/sfmtool-core/src/features/kdforest/distance.rs)
the `ForestScalar` trait and the SIMD kernels, and
[`calibrate.rs`](../../../crates/sfmtool-core/src/features/kdforest/calibrate.rs)
the `L_max` calibration. It is bound to Python as `sfmtool.spatial.KdForest`
(see "Python bindings" below). The same directory holds code specified
elsewhere: the file-backed forest
([lazy-kdforest-query.md](lazy-kdforest-query.md)), and the patch
constellation query with the `NeighborIndex` trait both forests implement
([kdf-constellation-query.md](kdf-constellation-query.md)).

```rust
pub struct KdForestParams {
    pub num_trees: usize,            // T
    pub split_dim_candidates: usize, // D
    pub leaf_size: usize,
    pub max_leaf_checks: usize,      // L_max, the default search budget
    pub seed: u64,
}
impl KdForestParams {
    pub fn balanced() -> Self; // also `Default`: T = 4, L_max = 128
    pub fn fast() -> Self;     // T = 4, L_max = 32
    pub fn accurate() -> Self; // T = 8, L_max = 512
}

pub struct Neighbor { pub index: u32, pub dist_sq: f32 }

pub struct KdForest<S: ForestScalar> { /* … */ }
pub type KdForestU8 = KdForest<u8>;
pub type KdForestF32 = KdForest<f32>;

impl<S: ForestScalar> KdForest<S> {
    pub fn build(points: &[S], n_points: usize, dim: usize,
                 params: KdForestParams, progress: &Progress<'_>)
        -> Result<Self, Cancelled>;

    pub fn search(&self, query: &[S], k: usize, max_leaf_checks: usize,
                  max_dist: Option<f32>) -> Vec<Neighbor>;
    pub fn search_batch(&self, queries: &[S], n_queries: usize, k: usize,
                        max_leaf_checks: usize, max_dist: Option<f32>) -> Vec<u32>;
    pub fn search_batch_with_distances(&self, queries: &[S], n_queries: usize,
                                       k: usize, max_leaf_checks: usize,
                                       max_dist: Option<f32>) -> (Vec<u32>, Vec<f32>);
    pub fn search_batch_with_distances_ordered(&self, queries: &[S], n_queries: usize,
                                               k: usize, max_leaf_checks: usize,
                                               max_dist: Option<f32>, order: &[u32])
        -> (Vec<u32>, Vec<f32>);
    pub fn locality_order(&self) -> &[u32];
    pub fn tree_leaves(&self, tree: usize) -> (&[u32], Vec<u32>);
    // The `NeighborIndex` corpus read, as in lazy-kdforest-query.md.
    pub fn resolve_descriptors(&self, feature_ids: &[u32]) -> Result<Vec<S>, KdfError>;

    pub fn calibrate_max_leaf_checks(&self, sample_queries: &[S], exact_nn: &[u32],
                                     target_precision: f64) -> usize;

    pub fn params(&self) -> KdForestParams;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn dim(&self) -> usize;
    pub fn num_trees(&self) -> usize;
}
```

Why it has this shape:

- **Flat row-major arrays at the boundary**, mirroring `spatial.rs`. `points`
  holds `n_points * dim` values and `queries` holds `n_queries * dim`; the
  batch searches return `n_queries * k` values, row by row, padded with
  `u32::MAX` (and `f32::INFINITY` for the distances) where fewer than `k`
  neighbors are found or fall within `max_dist`. A caller that holds a
  `.sift` descriptor block or a NumPy array passes it without copying it into
  per-row vectors.
- **Generic over the scalar, with `dim` a runtime field.** `S` is `u8` for
  descriptors and `f32` for general vectors. Descriptors are arbitrary-length
  `u8` vectors and the Python binding infers the width from the array it is
  handed, so a `KdForest<S, const DIM: usize>` could only serve them through a
  per-width dispatch table. The aliases `KdForestU8` / `KdForestF32` parallel
  `spatial.rs`'s `PointCloud2` / `PointCloud3`.
- **The budget is an argument of every search.** `L_max` is the setting a
  caller changes most, and one forest can serve queries that need different
  precision, so each search takes `max_leaf_checks`. The value in
  `KdForestParams` is the default a caller reads back through `params()`; the
  presets differ only in it and in `num_trees`.
- **`max_dist` is a Euclidean distance**, `None` for no cutoff, as in
  `spatial.rs`; the search squares it internally (see "Distance cutoff"). The
  distances it hands back are squared, so a ratio test compares them against
  the squared ratio.
- **`build` takes a `Progress`.** It says where it has got to and stops when
  it is asked to
  ([`../../gui/operation-progress.md`](../../gui/operation-progress.md)); a
  caller with nothing to report through passes `Progress::none()`, which
  cannot cancel, so its build cannot fail. The unit it counts is a **point
  placed in a leaf**, across every tree: the trees are built in parallel, so
  per-tree reporting would be `T` steps that all land at the end, while points
  placed moves evenly from the first leaf to the last. The counter is one
  relaxed `fetch_add` per leaf and it reports only when the count crosses one
  of two hundred boundaries. A build that is cancelled hands back nothing,
  because a forest missing the trees that had not finished is not a forest.
- **`search_batch_with_distances_ordered`** *processes* the batch in the given
  permutation of `0..n_queries` but returns it in query order. Each query is
  independent and deterministic, so the schedule changes only cache behavior;
  `locality_order()` hands back tree 0's leaf layout, which keeps
  descriptor-space neighbors consecutive, so in a self-join batch (the corpus
  queried against itself) most row fetches are cache hits. The track-cluster
  matcher calls it this way.
- **`calibrate_max_leaf_checks` is a method on the built forest**, since the
  calibration measures *this* forest's trees (see "Precision calibration").

A build and a batched 2-NN query, with the ratio test on the squared
distances:

```rust
use sfmtool_core::features::kdforest::{KdForestParams, KdForestU8};
use sfmtool_core::progress::Progress;

// Three 4-D u8 points, row-major.
let points: Vec<u8> = vec![0, 0, 0, 0, 10, 10, 10, 10, 0, 1, 0, 1];
let params = KdForestParams::balanced();
let forest = KdForestU8::build(&points, 3, 4, params, &Progress::none())
    .expect("nothing asked it to stop");

let queries: Vec<u8> = vec![0, 0, 0, 1, 9, 9, 9, 9];
let (indices, dist_sq) =
    forest.search_batch_with_distances(&queries, 2, 2, params.max_leaf_checks, None);
// Row `q` is indices[2 * q..2 * q + 2], nearest first.
let passes_ratio = dist_sq[0] < 0.8 * 0.8 * dist_sq[1];
```

## Python bindings

[`crates/sfmtool-py/src/spatial/kdforest.rs`](../../../crates/sfmtool-py/src/spatial/kdforest.rs),
registered by the `spatial` binding module and imported as
`from sfmtool.spatial import KdForest`, following `flow/optical.rs` conventions
(`PyReadonlyArray2` in, `IntoPyArray` out, `py.detach(...)` around build and
query). The binding indexes `u8` descriptors only:

- `KdForest(descriptors (N,D) u8, preset=None, num_trees=None, leaf_size=None, max_leaf_checks=None, seed=None)`
  — `#[pyclass]` constructor; `D` is inferred from the array width. `preset` is
  one of `balanced` (default) / `fast` / `accurate`, with the other kwargs as
  optional overrides.
- `forest.query(descriptors (M,D) u8, k=2, max_leaf_checks=None, max_dist=None) -> (indices (M,k) u32, distances (M,k) f32)`
  — `max_leaf_checks=None` uses the build-time default. Reported distances are
  Euclidean (the `sqrt` of the internal squared value), matching the existing
  `descriptor_distance` binding, where the Rust
  `search_batch_with_distances` reports the squared value directly.
- Read-only properties `len`, `is_empty`, `dim`, `dtype` (always `"uint8"`),
  `num_trees` and `max_leaf_checks` (the build-time default budget).
- `forest.leaf_layout(tree) -> (point_ids, leaf_starts)` — one tree's
  leaf-ordered point ids and the start of each leaf within them, both `uint32`,
  for studying how a `.kdf` file should order its descriptors.
- `forest.constellation_query(…)` and `forest.constellation_at_pixel(…)` — the
  patch constellation query over this in-memory forest, specified in
  [kdf-constellation-query.md](kdf-constellation-query.md).

```python
from sfmtool.spatial import KdForest

forest = KdForest(train_descriptors, preset="accurate")  # (N, 128) uint8
indices, distances = forest.query(query_descriptors, k=2)  # (M, 2) each
passes_ratio = distances[:, 0] < 0.8 * distances[:, 1]
```

The `(indices, distances)` pair from `query` has the layout a k-NN ratio test
reads, but no module under `src/sfmtool/` uses `KdForest`. The one matcher that
uses the forest, the track-cluster matcher, calls it from Rust.

## Algorithm: Multiple Randomized KD-Trees

Reference: Marius Muja and David G. Lowe, "Fast Approximate Nearest Neighbors
with Automatic Algorithm Configuration," VISAPP 2009.
[09muja](https://www.cs.ubc.ca/~lowe/papers/09muja.pdf) (§3.1). Builds on the
randomized trees of [Silpa-Anan & Hartley (2008)](https://users.cecs.anu.edu.au/~hartley/Papers/PDF/SilpaAnan:CVPR08.pdf),
the priority queue search of
[Arya et al. (1998)](https://www.cse.ust.hk/faculty/arya/pub/JACM.pdf), and
the best-bin-first fixed-budget stopping criterion of
[Beis & Lowe (1997)](https://www.cs.ubc.ca/~lowe/papers/cvpr97.pdf).

### Overview

A classical kd-tree ([Friedman et al., 1977](https://dl.acm.org/doi/pdf/10.1145/355744.355745)) splits the data in half at each level
on the **single** dimension of greatest variance. In high dimensions this is both
fragile (one fixed partition) and ineffective. The forest fixes this two ways:

1. **Randomized construction.** Build `T` independent trees. At each node, instead
   of always splitting on the top-variance dimension, pick the split dimension
   **at random from the `D` dimensions of highest variance**. Different random
   choices make the trees diverge, so a query and its true nearest neighbor that
   are split apart in one tree are likely to share a cell in another.

2. **Shared best-bin-first priority search.** Search all `T` trees together with a
   **single priority queue** ordered by increasing distance from the query to each
   unexplored bin boundary. Approximation is controlled by stopping after a fixed
   budget of `L_max` distance computations; the best candidates found so far are
   returned.

The paper's key empirical results for the forest:

- Performance improves with the number of trees up to ~20 (§4.1, 100K SIFT),
  then is flat or decreasing; memory grows linearly with `T`.
- The fixed value `D = 5` performs well across all tested datasets.
- At 60% precision the forest reaches ~three orders of magnitude speedup over
  linear search on the sift1M dataset (§4.4, Fig. 6(b)).
- The forest is the better of the two algorithms when intrinsic dimensionality is
  much lower than the ambient dimension, except at precisions very close to 100%
  (Fig. 6(d)).

### 1. Building a tree

Each of the `T` trees is built independently from the full point set.

```
build(point_indices, depth):
    if point_indices.len() <= leaf_size:
        return Leaf(point_indices)

    var[d]    = variance of coordinate d over point_indices, for d in 0..DIM
    top       = the D indices with the largest var[d]            # D = 5
    split_dim = top[rng.gen_range(0..D)]                          # random choice
    split_val = median of { x[split_dim] : x in point_indices }  # halve the data

    (left, right) = partition point_indices by x[split_dim] <= split_val,
                     distributing equal-to-median points to keep |left| ≈ |right|
    return Internal { split_dim, split_val,
                      left:  build(left,  depth + 1),
                      right: build(right, depth + 1) }
```

- **Variance estimate.** Per-dimension variance is estimated from a bounded random
  sample of the node's points (e.g. up to ~100), since the top-`D` selection only
  needs the relative ordering of variances.
- **Split value = median** on the chosen dimension. With `u8` descriptors,
  many points may share the median value; the partition distributes duplicates
  across both sides to keep the split balanced (depth ≈ log₂ N).
- **Determinism.** Each tree gets its own seeded RNG (derived from a single
  user-visible `seed` + tree index) so builds are reproducible.
- **`leaf_size`** caps points per leaf. Small buckets (e.g. 8–16) cut tree height
  and let the leaf scan run as one tight SIMD loop.

### 2. Searching the forest

A k-NN query maintains a bounded result set of the `k` best candidates seen and a
single min-priority queue of unexplored branches across all trees, keyed by a
lower bound on the distance from the query to that branch's cell. An optional
`max_dist` cutoff (squared internally) bounds the result set from the start, so
both the leaf scan and the branch pruning ignore anything farther.

```
search(q, k, L_max, max_dist):
    result   = BoundedResult(k, max_dist²)  # keeps the k smallest dist_sq <= max_dist²
    queue    = MinHeap<(lb_dist_sq, &Node)>
    checked  = BitSet(N)                 # dedupe points shared across trees
    n_checks = 0

    for tree in forest:                  # seed: descend every tree once
        descend(tree.root, q, 0, result, queue, checked, &mut n_checks)

    while let Some((lb, node)) = queue.pop():
        if n_checks >= L_max: break
        if lb > result.worst_dist_sq(): break       # nothing closer can remain
        descend(node, q, lb, result, queue, checked, &mut n_checks)

    return result.sorted_ascending()

descend(node, q, lb, result, queue, checked, n_checks):
    while node is Internal:
        diff = q[node.split_dim] as i32 - node.split_val as i32
        (near, far) = if diff <= 0 { (node.left, node.right) }
                      else        { (node.right, node.left) }
        # lower bound for the far cell: add squared distance to this split plane
        far_lb = lb + diff*diff
        queue.push((far_lb, far))
        node = near
    for p in node.points:                # Leaf
        if checked.insert(p):            # first time this point is seen
            *n_checks += 1
            result.consider(p, dist_sq(q, points[p]))   # keeps only dist_sq <= max_dist²
```

- **Single shared queue across all trees**, ordered by `lb` (squared distance from
  `q` to the branch boundary). Popping the smallest first is best-bin-first.
- **Boundary lower bound (approximate).** We enqueue the far child with
  `lb + diff²` — the additive priority bound of FLANN's randomized kd-tree
  forest. This bound is **not admissible**: when the same axis is re-split deeper
  along a far path it double-counts that axis and can *over-estimate* the true
  query-to-cell distance. Consequently the `lb > result.worst_dist_sq()`
  early-exit (and the enqueue prune) can skip a branch that holds a true
  neighbor, so the forest is **approximate even at an unlimited budget** — the
  same speed/accuracy trade FLANN's `FLANN_INDEX_KDTREE` makes. (An exact bound
  would replace, not add, the per-axis component, which needs per-axis state on
  every queue entry; FLANN's *exact* index is the separate single-tree DFS,
  out of scope here.) The only configuration that is exact by construction is a
  single leaf (`leaf_size ≥ N`), where no pruning occurs.
- **Stopping criterion = `L_max` distance computations (soft).** `L_max` is the
  search's precision setting (Beis & Lowe's `E_max`): larger ⇒ more accurate
  and slower. It is a *soft* budget — checked at leaf granularity and only after each
  tree has been descended once to seed the queue — so the actual count can exceed
  `L_max` by up to one leaf per seeded tree. This matches FLANN's `checks`.
- **Cross-tree dedup.** A `checked` bitset ensures each point's distance is
  computed at most once, so `L_max` measures unique work.
- **Distance cutoff.** Seeding `result.worst_dist_sq()` with `max_dist²` makes the
  cutoff prune branches and reject candidates for free through the same
  `worst_dist_sq()` paths — mirroring `spatial.rs`'s `nearest_k_within_radius`.
  Fewer than `k` neighbors may be returned (padded with `u32::MAX`). For `u8`
  descriptors the squared distance is an integer, so the cutoff is `max_dist²`
  rounded down after a relative tolerance of `4 · f32::EPSILON`, which absorbs
  the rounding of `max_dist` to `f32`: `max_dist = 2.5` keeps a squared
  distance of 6 and drops 7, and `max_dist = sqrt(7)` keeps 7.

### Parameters

| Symbol | Name | Description | Default |
|--------|------|-------------|---------|
| `T` | Number of trees | Independent randomized kd-trees in the forest | 4 |
| `D` | Random-dim pool | Split dim picked at random from the top-`D` variance dims | 5 |
| `leaf_size` | Leaf bucket size | Max points per leaf before splitting | 16 |
| `L_max` | Max checks | Unique distance computations before the search stops | 128 |
| `seed` | RNG seed | Base seed for reproducible tree construction | 0 |
| `k` | Neighbors | Number of nearest neighbors per query | 2 (ratio test) |

- **`T = 4`** is a good default for SIFT-sized problems (the paper's parameter
  search sweeps `{1, 4, 8, 16, 32}`; gains saturate around ~20). Raise toward 8–16
  for higher precision at the cost of linear memory growth.
- **`D = 5`** is the paper's fixed value and should not normally be changed.
- **`L_max`** trades precision for speed. Its defaults are fixed preset values
  (32 / 128 / 512 for `fast` / `balanced` / `accurate`, see "Rust API"), and
  every caller takes a preset's value or passes its own. An optional helper (see
  below) can instead fit it on a sample to a *target precision* — a deliberately
  narrow slice of the paper's broader auto-tuning, not the full Nelder-Mead cost
  optimization.

### Precision calibration (optional helper)

Given a sample of
queries with their *exact* nearest neighbors (one brute-force pass on a subset),
binary-search the smallest `L_max` whose measured precision (fraction of sample
queries whose exact NN is returned) meets a target (e.g. 0.9). This mirrors the
paper's statement that the user "specifies only the desired search precision,
which is used during training to select the number of leaf nodes." `T`, `D`, and
`leaf_size` stay fixed; only `L_max` is fitted. Nothing outside the module's
own tests calls it.

## Parallelism & SIMD strategy

Following the optical-flow and SIFT implementations:

- **Build.** The `T` trees are independent → built with rayon (`into_par_iter`
  over tree index), each with its own seeded RNG. Recursion within a tree is
  sequential, so build parallelism is capped at `T`.
- **Tree storage.** Each tree is an arena of nodes linked by `u32` indices,
  not boxes, plus its own permutation of point ids that each leaf indexes as a
  contiguous range. A tree is then a few flat allocations, a leaf's ids are
  adjacent in memory, and the forest is `Send`/`Sync`, so one forest serves
  every rayon worker by shared reference.
- **Batched query.** `par_chunks_mut(k)` over the output rows (as in
  `spatial.rs::nearest_k_within_radius`); the forest is shared `&`. The batch
  runs under rayon's `for_each_init`, which makes one scratch (priority queue,
  `checked` bitset, result set) for each piece rayon splits the batch into, so
  a worker can make several over one batch. Within a piece the scratch is
  reset, not reallocated, between queries, so the query inner loop does no
  per-query heap allocation.
- **SIMD leaf scan.** The per-point squared-L2 distance over the 128 `u8` lanes
  is the hot loop. `distance.rs` provides hand-written AVX2 and SSE2 sum-of-
  squared-differences kernels (`|a−b|` via `max−min`, widened and squared with
  `madd_epi16`), runtime-dispatched, with the scalar fallback delegating to
  `descriptor_distance_l2_squared`. The generic `f32` index uses a scalar
  squared-L2 loop (no hand-written SIMD).
- **Integer distance domain.** Keep SIFT distances in `i64`/`u32` squared-L2 and
  take `sqrt` only when a reported distance is needed, matching
  `features/feature_match/descriptor.rs`.
- **Software prefetch in the leaf scan.** The leaf scan is memory-bound, not
  compute-bound: each point row is a random ~128-byte gather from the shared
  corpus, and the gather takes hundreds of cycles from DRAM against the SIMD
  kernel's ~25. Because a
  leaf's point ids are known before its rows are needed, the scan issues
  `_mm_prefetch` hints two rows ahead, overlapping the fetches with the distance
  computation. Measured on a 255k-descriptor corpus (32 MB, `accurate` preset):
  ~1.2× on the batched query, recovering most of what a per-tree leaf-contiguous
  coordinate copy would buy without the `T`× memory cost. Prefetch is a hint
  with no architectural effect, so results and determinism are unchanged; the
  helper is a no-op off x86_64.

### Diagnostics (environment variables)

Following the SIFT/optical-flow precedent, setting `SFMTOOL_KDFOREST_STATS`
prints the per-query average checks and queue pushes/pops for each batch an
in-memory forest searches — useful when choosing `L_max`. Any value turns it
on, including `0` and the empty string; only leaving it unset turns it off.
(Measured precision is reported separately by the tests and
`scripts/kdforest_vs_flann.py`, not by this env var.)

Setting `SFMTOOL_KDFOREST_NO_SIMD`, to any value in the same way, makes the
`u8` distance use the scalar loop instead of the AVX2/SSE2 kernels on x86_64,
for timing comparisons. The kernels return the same integers as the scalar
loop, so it changes speed only, never results.

## Testing & validation

- **Exactness ceiling.** The bound is approximate (see "Boundary lower bound"),
  so the genuine exact configuration is a *single leaf* (`T = 1`,
  `leaf_size ≥ N`): the root scans every point with no pruning, so the result is
  exact by construction. Unit-tested against `descriptor_distance_l2_squared`
  brute force (`single_leaf_search_is_exact`) and via the Python binding
  (`test_single_leaf_matches_brute_force` and `test_self_query_is_exact`, both
  built with `num_trees=1, leaf_size=len(pts)`). A deep tree at full budget is
  asserted to reach **high recall**, not exactness
  (`deep_tree_full_budget_high_recall`, ≥0.95; the Python
  `test_deep_tree_high_recall`, ≥0.9).
- **Precision vs budget curve.** A synthetic-data version is covered
  (`precision_monotone_in_budget`: precision is asserted monotone
  non-decreasing in `L_max` — valid because a larger budget checks a superset of
  points in the same heap order — and ≥0.98 at an exhaustive budget). No test
  measures the curve on real SIFT descriptors; the standalone script
  `scripts/kdforest_vs_flann.py` compares the forest's recall@1 and query time
  against OpenCV's FLANN forest on real SIFT.
- **Determinism.** Same `seed` ⇒ identical query results across runs and thread
  counts (each tree is seeded independently as `seed + tree_index` and built by
  an order-preserving parallel map). Covered by `determinism_same_seed` and the
  Python `test_determinism`.
- **Matcher parity** — verified in production via the track-cluster matcher:
  the cluster-radius scheme that consumes `KdForest`'s k-NN at the accurate
  preset is the matcher behind `sfm match --cluster`. See
  `specs/core/features/track-cluster-matching.md` for the empirical recall numbers and
  the dataset sweep results.
- **Rust unit tests:** high recall across many trees exercising cross-tree
  `checked`-set dedup (`many_trees_full_budget_high_recall`), median split
  balance under heavy duplicates (`duplicate_coordinates_build_and_query`),
  `max_dist` cutoff
  (`max_dist_cutoff_respected`), reported-distance correctness, `< k` padding,
  empty/`k = 0` edge cases, and calibration (`calibration_finds_a_budget`).
  `distance/tests.rs` covers the kernels separately: SIMD-vs-scalar parity
  (`u8_kernel_matches_scalar`, `simd_kernels_match_scalar`), saturating
  extremes, and cutoff rounding for both scalars.
- **PyO3 surface test** (`tests/rust_bindings/spatial/test_kdtree_forest_rust_bindings.py`) exercising
  build/query and comparing against a NumPy brute-force reference.
- **Criterion benchmarks** (`crates/sfmtool-core/benches/kdtree_forest.rs`), all
  on synthetic random 128-dimension `u8` descriptors: build time vs `T`, query
  throughput vs `L_max`, a batched 1-NN search against the exact brute-force
  scan, and a file-backed (`.kdf`) batch against the same batch in memory.

## Out of scope

This module's forest is memory-resident. Persistent storage and file-backed
queries are specified in [lazy-kdforest-query.md](lazy-kdforest-query.md), with
the companion [KDF format](../../formats/kdf-file-format.md).

The paper's companion priority search k-means tree and its automatic algorithm
and parameter selection are not implemented.

`KdForestF32` is a generality rather than a tuned path. It shares the build and
search machinery with the descriptor index but has no hand-written SIMD kernel,
is exercised far less than the `u8` one, and has no benchmarks on the
higher-dimensional, less-correlated data it would be used for.

## Dependencies

No new crate dependencies: `rayon` for parallel build and batched query, the
workspace's existing `rand` crate (`StdRng::seed_from_u64` per tree, as used
elsewhere in `sfmtool-core`) for deterministic seeded construction, and
`criterion` (dev) for benchmarks. SIMD via `std::arch` (hand-written AVX2/SSE2
`u8` kernels).
