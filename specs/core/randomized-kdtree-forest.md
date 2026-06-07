# Randomized KD-Tree Forest

> Status (2026-06-06): Implemented (Phase 1). Pure-Rust ANN index landed in
> `crates/sfmtool-core/src/kdforest/` (`distance.rs`, `build.rs`, `search.rs`,
> `calibrate.rs`, `mod.rs`) with PyO3 bindings in
> `crates/sfmtool-py/src/py_kdforest.rs` (`KdForest` class), Rust unit tests
> (`kdforest/tests.rs`), a Python binding test
> (`tests/rust_bindings/test_kdtree_forest_rust_bindings.py`), and Criterion benchmarks
> (`crates/sfmtool-core/benches/kdtree_forest.rs`). It mirrors the optical-flow
> (`specs/core/optical-flow.md`) and SIFT (`specs/core/sift.md`)
> implementations: pure Rust, no external ANN library, AVX2/SSE2 SIMD inner
> loop, and
> rayon for both build and batched query, covering the multiple randomized
> kd-tree forest from Muja & Lowe (2009).
>
> **Refinements from the original draft (as built):**
> - **Runtime dimensionality.** The index is `KdForest<S: ForestScalar>` with a
>   runtime `dim` field rather than `KdForest<S, const DIM>`. Descriptors are
>   "arbitrary-length `u8` vectors" and the Python binding infers `D` from the
>   array width at runtime — a const generic could not satisfy that without a
>   per-width dispatch table. Type aliases `KdForestU8` / `KdForestF32`.
> - **Presets** are `fast` / `balanced` (default) / `accurate`, differing in
>   `max_leaf_checks` (32 / 128 / 512) and `num_trees` (4 / 4 / 8). These are
>   starting points pending broader cross-validation.
> - **Python `query` returns Euclidean distances** (sqrt of the internal squared
>   value), matching the existing `descriptor_distance` binding; the Rust
>   `search_batch_with_distances` returns squared distances.
> - **Matcher CLI integration** (an approximate backend wired into
>   `src/sfmtool/feature_match/`) is left as a follow-up; the `KdForest` Python
>   class already emits the `(indices, distances)` layout that pipeline consumes.

## Motivation

The most expensive step in descriptor matching is finding, for each query
descriptor, its nearest neighbor(s) among a large set of candidates. Today
sfmtool-core matches descriptors with an exhaustive scan
(`feature_match/descriptor.rs`: `find_best_match`, `find_best_match_contiguous`),
which is fine for many use cases. We want an approximate alternative for large
feature counts, and potentially for patch matching as well.

`spatial.rs` already wraps an exact kd-tree for 2D/3D point clouds; exact
kd-trees degenerate to near-linear search in the 128 dimensions of a SIFT
descriptor. The randomized kd-tree forest is the high-dimensional approximate
counterpart, trading a small, controllable loss in accuracy for one to three
orders of magnitude in speed. A pure-Rust implementation (rather than wrapping
FLANN/nanoflann) gives us a shared, reusable ANN index that any matcher
(descriptor, sweep, polar, flow-seeded) can build once and query many times.

This spec defines library types in sfmtool-core, independent of any on-disk
layout. It follows the codebase's existing descriptor conventions: descriptors
are arbitrary-length `u8` vectors and distances are squared L2 computed in integer space
(`descriptor_distance_l2_squared`), with `sqrt` taken only for the neighbors
actually returned.

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
        (near, far) = if diff < 0 { (node.left, node.right) }
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
  single precision knob (Beis & Lowe's `E_max`): larger ⇒ more accurate and
  slower. It is a *soft* budget — checked at leaf granularity and only after each
  tree has been descended once to seed the queue — so the actual count can exceed
  `L_max` by up to one leaf per seeded tree. This matches FLANN's `checks`.
- **Cross-tree dedup.** A `checked` bitset ensures each point's distance is
  computed at most once, so `L_max` measures unique work.
- **Distance cutoff.** Seeding `result.worst_dist_sq()` with `max_dist²` makes the
  cutoff prune branches and reject candidates for free through the same
  `worst_dist_sq()` paths — mirroring `spatial.rs`'s `nearest_k_within_radius`.
  Fewer than `k` neighbors may be returned (padded with `u32::MAX`).

### Parameters

| Symbol | Name | Description | Default |
|--------|------|-------------|---------|
| `T` | Number of trees | Independent randomized kd-trees in the forest | 4 |
| `D` | Random-dim pool | Split dim picked at random from the top-`D` variance dims | 5 |
| `leaf_size` | Leaf bucket size | Max points per leaf before splitting | 16 |
| `L_max` | Max checks | Unique distance computations before the search stops | precision-tuned |
| `seed` | RNG seed | Base seed for reproducible tree construction | 0 |
| `k` | Neighbors | Number of nearest neighbors per query | 2 (ratio test) |

- **`T = 4`** is a good default for SIFT-sized problems (the paper's parameter
  search sweeps `{1, 4, 8, 16, 32}`; gains saturate around ~20). Raise toward 8–16
  for higher precision at the cost of linear memory growth.
- **`D = 5`** is the paper's fixed value and should not normally be changed.
- **`L_max`** is the precision/speed dial. Rather than have users guess it, an
  optional helper (see below) can calibrate it on a sample so a caller specifies a
  *target precision* instead — a deliberately narrow slice of the paper's broader
  auto-tuning, not the full Nelder-Mead cost optimization.

### Precision calibration (optional helper)

Given a sample of
queries with their *exact* nearest neighbors (one brute-force pass on a subset),
binary-search the smallest `L_max` whose measured precision (fraction of sample
queries whose exact NN is returned) meets a target (e.g. 0.9). This mirrors the
paper's statement that the user "specifies only the desired search precision,
which is used during training to select the number of leaf nodes." `T`, `D`, and
`leaf_size` stay fixed; only `L_max` is fitted.

## Out of scope

The companion priority search k-means tree and the paper's automatic algorithm
and parameter selection are out of scope.

## Parallelism & SIMD strategy

Following the optical-flow and SIFT implementations:

- **Build.** The `T` trees are independent → built with rayon (`into_par_iter`
  over tree index), each with its own seeded RNG. Recursion within a tree is
  sequential, so build parallelism is capped at `T`.
- **Batched query.** `par_chunks_mut(k)` over the output rows (as in
  `spatial.rs::nearest_k_within_radius`); the forest is shared `&`. Each rayon
  worker keeps one reusable scratch (priority queue, `checked` bitset, result
  set) that is reset, not reallocated, between queries, so the query inner loop
  does no per-query heap allocation.
- **SIMD leaf scan.** The per-point squared-L2 distance over the 128 `u8` lanes
  is the hot loop. `distance.rs` provides hand-written AVX2 and SSE2 sum-of-
  squared-differences kernels (`|a−b|` via `max−min`, widened and squared with
  `madd_epi16`), runtime-dispatched, with the scalar fallback delegating to
  `descriptor_distance_l2_squared`. The generic `f32` index uses a scalar
  squared-L2 loop (no hand-written SIMD).
- **Integer distance domain.** Keep SIFT distances in `i64`/`u32` squared-L2 and
  take `sqrt` only when a reported distance is needed, matching
  `feature_match/descriptor.rs`.

### Diagnostics (environment variables)

Following the SIFT/optical-flow precedent, `SFMTOOL_KDFOREST_STATS=1` prints the
per-query average checks and queue pushes/pops for a batch — useful when
calibrating `L_max`. (Measured precision is reported separately by the tests and
`scripts/kdforest_vs_flann.py`, not by this env var.)

## Architecture

### Module structure (as built)

```
sfmtool-core/src/kdforest/
├── mod.rs        # Public API: KdForestParams, KdForest, build, search, search_batch
├── build.rs      # Randomized tree construction (variance sample, top-D pick, median split)
├── search.rs     # Shared-queue BBF search, bounded result set, checked-set dedup
├── distance.rs   # SquaredL2 over u8 (SIMD) and f32; ForestScalar trait, metric-generic
├── calibrate.rs  # Optional L_max precision calibration against brute-force sample
└── tests.rs      # Exactness ceiling, determinism, monotonicity, dedup, cutoff, calibration
```

`KdForest<S: ForestScalar>` is generic over the scalar `S` (`u8` for
descriptors, `f32` for general vectors), with `dim` stored at runtime (see the
status note above). Type aliases `KdForestU8` / `KdForestF32` parallel
`spatial.rs`'s `PointCloud2` / `PointCloud3`.

### Key types

- **`KdForestParams`** — `T`, `D`, `leaf_size`, `seed`, plus the default
  `max_leaf_checks` (`L_max`). Constructor presets analogous to the optical-flow
  presets, e.g. `fast` (low `L_max`), `balanced`, `accurate`.
- **`KdForest`** — owns the points (flat row-major, like `spatial.rs`), the `T`
  tree node arenas (index-based `Vec<Node>`, no `Box` chasing), and per-dimension
  bookkeeping. Built once, queried many times.
- **`Node`** — either `Internal { split_dim: u16, split_val, left: u32, right: u32 }`
  or `Leaf { start: u32, len: u32 }` indexing a per-tree permutation of point ids.
  Index-based children keep nodes cache-friendly and `Send`/`Sync` for shared
  queries.
- **`Neighbor { index: u32, dist_sq }`** and a small **bounded result set** (a
  `k`-element max-heap or insertion-sorted array for the typical `k ≤ 8`).

### Public API (planned)

Flat row-major arrays at the boundary, mirroring `spatial.rs`:

- `KdForest::build(points: &[S], n_points, params) -> KdForest` (points length
  `n_points * DIM`).
- `forest.search(query: &[S], k, max_leaf_checks, max_dist: Option<f32>) -> Vec<Neighbor>`
  — single query; `max_dist` is an optional Euclidean distance cutoff (`None` =
  unbounded), squared internally as in `spatial.rs`.
- `forest.search_batch(queries: &[S], n_queries, k, max_leaf_checks, max_dist: Option<f32>) -> Vec<u32>`
  — flat `n_queries * k` indices (row-major), `u32::MAX` padding when fewer than
  `k` are found (or fall within `max_dist`), rayon over rows. A `_with_distances`
  variant also returns the squared distances for the ratio test / thresholding.
- `calibrate_max_leaf_checks(forest, sample_queries, exact_nn, target_precision)
  -> usize`.

### Python bindings (as built)

`crates/sfmtool-py/src/py_kdforest.rs`, registered in `sfmtool-py/src/lib.rs`,
following `py_optical_flow.rs` conventions (`PyReadonlyArray2` in, `IntoPyArray`
out, `py.detach(...)` around build and query):

- `KdForest(descriptors (N,D) u8, preset=None, num_trees=None, leaf_size=None, max_leaf_checks=None, seed=None)`
  — `#[pyclass]` constructor; `D` is inferred from the array width. `preset` is
  one of `balanced` (default) / `fast` / `accurate`, with the other kwargs as
  optional overrides.
- `forest.query(descriptors (M,D) u8, k=2, max_leaf_checks=None, max_dist=None) -> (indices (M,k) u32, distances (M,k) f32)`
  — `max_leaf_checks=None` uses the build-time default; reported distances are
  Euclidean.

This output is exactly what `src/sfmtool/feature_match/` already consumes for the
ratio test, so an approximate matcher backend slots in alongside the exact one.

## Phasing

1. **Phase 1 (this spec): CPU + SIMD + multithread.** Randomized build,
   shared-queue BBF search, SIFT-`u8` specialization, and `L_max` calibration
   helper, cross-validated against brute-force exact NN. **Done**, except the
   matcher CLI integration (the `KdForest` building block exists; wiring an
   approximate backend into `src/sfmtool/feature_match/` is a follow-up).
2. **Phase 2 (future):** generic `f32` index hardening and benchmarks on
   higher-dimensional / less-correlated data; consider the k-means tree and full
   auto-selection as separate specs if a dataset wants them.

## Testing & validation

> _Status (2026-06-06): Implemented in `kdforest/tests.rs` and
> `tests/rust_bindings/test_kdtree_forest_rust_bindings.py`, plus
> `benches/kdtree_forest.rs`. The two dataset-driven items below remain as
> future work; the listed unit tests are covered._

- **Exactness ceiling.** The bound is approximate (see "Boundary lower bound"),
  so the genuine exact configuration is a *single leaf* (`T = 1`,
  `leaf_size ≥ N`): the root scans every point with no pruning, so the result is
  exact by construction. Unit-tested against `descriptor_distance_l2_squared`
  brute force (`single_leaf_search_is_exact`) and via the Python binding
  (`test_exhaustive_budget_matches_brute_force`, which uses `num_trees=1,
  leaf_size=1` and asserts exact recovery on a fixed random set — empirically
  exact there, though not guaranteed for a deep tree in general). A deep tree at
  full budget is asserted to reach **high recall**, not exactness
  (`deep_tree_full_budget_high_recall`, ≥0.95).
- **Precision vs budget curve.** A synthetic-data version is covered
  (`precision_monotone_in_budget`: precision is asserted monotone
  non-decreasing in `L_max` — valid because a larger budget checks a superset of
  points in the same heap order — and ≥0.98 at an exhaustive budget). _Future:_
  the same measurement on the checked-in datasets
  (`seoul_bull_sculpture`, `seattle_backyard`, `dino_dog_toy`) with a speedup
  table calibrated like the optical-flow cross-validation table. A standalone
  comparison against OpenCV's FLANN forest on real SIFT already exists in
  `scripts/kdforest_vs_flann.py`.
- **Determinism.** Same `seed` ⇒ identical query results across runs and thread
  counts (each tree is seeded independently as `seed + tree_index` and built by
  an order-preserving parallel map). Covered by `determinism_same_seed` and the
  Python `test_determinism`.
- **Matcher parity.** _Future, with the matcher CLI integration:_ the
  approximate matcher's accepted matches should recover ≥0.95 of the exact
  mutual-best matches at the accurate preset. The exactness/precision tests
  above bound the index's accuracy in the meantime.
- **Rust unit tests:** high recall across many trees exercising cross-tree
  `checked`-set dedup (`many_trees_full_budget_high_recall`), median split
  balance under heavy duplicates (`duplicate_coordinates_build_and_query`),
  `max_dist` cutoff
  (`max_dist_cutoff_respected`), reported-distance correctness, `< k` padding,
  empty/`k = 0` edge cases, SSE2-vs-scalar kernel parity (`u8_kernel_matches_scalar`),
  and calibration (`calibration_finds_a_budget`).
- **PyO3 surface test** (`tests/rust_bindings/test_kdtree_forest_rust_bindings.py`) exercising
  build/query and comparing against a NumPy brute-force reference.
- **Criterion benchmarks** (`crates/sfmtool-core/benches/kdtree_forest.rs`): build
  time vs `T`, query throughput vs `L_max`, and end-to-end image-pair matching
  vs the exact scanner — same structure as `benches/optical_flow.rs`.

## Dependencies

No new crate dependencies: `rayon` for parallel build and batched query, the
workspace's existing `rand` crate (`StdRng::seed_from_u64` per tree, as used
elsewhere in `sfmtool-core`) for deterministic seeded construction, and
`criterion` (dev) for benchmarks. SIMD via `std::arch` (hand-written AVX2/SSE2
`u8` kernels).
