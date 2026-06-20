# Track-Cluster Matching

## The Idea

Traditional SfM feature matching is pair-centric. It enumerates image pairs,
matches SIFT descriptors within each pair, geometrically verifies each pair, and
only then stitches the pairwise matches into tracks.

Each track forms a cluster in descriptor space: its observations are the same
surface point seen from different images, so their descriptors are mutually
close. What if we start from that clustering property directly, instead of
building tracks up from pairwise correspondence? A descriptor index of all the
features across all images gives us a way to achieve this. We can search for
candidate clusters in the index directly.

## Approach

Querying the index for a descriptor's nearest neighbours returns the other
members of its cluster, interleaved with unrelated background. The problem is
approximating where each descriptor's cluster ends — which neighbours are genuine
co-observations and which are background. We call this process **track-cluster
matching**.

We use a nearest-neighbour query to determine this for each descriptor. Its
sorted neighbour distances rise gently across its true co-observations, then jump
up to the level of unrelated features — its *background floor*. We take a fixed
fraction of that floor as the descriptor's cluster membership radius and keep
its cross-image neighbours within that radius as candidate co-observations. A
descriptor with tight co-observations keeps them; an isolated descriptor, whose
nearest neighbour already sits at the background level, keeps nothing.

![Neighbour-distance profiles for example descriptors (seattle_backyard): the
co-observations (green) sit at the near ranks below the cluster membership radius α·B
(dashed), while unrelated background (grey) plateaus above it; the background
scale B is the d-th-nearest distance (dotted).](images/floor-profile.png)

We materialize these candidate track clusters as the matcher's primary output, so
later consumers can work with the clusters themselves. For the current consumer we
convert them into feature matches between image pairs, feeding geometric
verification and the rest of the SfM pipeline.

## Empirical Observations

### The nearest-distance distribution is bimodal

For each descriptor, let `d1` be the distance to its nearest *other* descriptor.
Across the corpus `d1` is **bimodal**: a near mode of descriptors that have a
likely match — a feature seen in more than one image, whose other observations
sit close in descriptor space — and a far mode of *isolated* descriptors seen in
only one place, with no near neighbour.

![Histogram of d1 for seattle_backyard: a near "has a near neighbour" mode and a
far "isolated" mode, with the antimode valley between them](images/d1-histogram-seattle.png)

The valley between the modes can be used as a global threshold to determine
clusters. It varies by dataset, so must be derived from the data; we found a
per-descriptor approach works better.

### The floor separates co-observations from background

A descriptor's true co-observations sit close in descriptor space, while unrelated
features pile up in a far "background" shell; the floor exploits the gap between
them. Plotting, per in-track descriptor, the distances to its co-observations
against the distances to its background neighbours, the floor `α·B` falls in the
valley between the two on seoul_bull, seattle_backyard, and kerry_park — capturing
most co-observations while excluding the shell. The exception is dino_dog_toy,
whose repetitive structure leaves the distributions overlapping — there no radius
is clean, and the interleaved background is left to geometric verification.

![Co-observation (green) vs background (red) descriptor-distance distributions per
reconstruction; the dashed line is the median floor α·B](images/floor-coobs-vs-background.png)

The background scale `B` (the d-th-nearest distance) marks that shell, so `α` sets
how far below it the cut sits. Sweeping `α` shows why `α = 0.8`: co-observation
recall climbs steadily, but the background admitted stays near zero until
`α ≈ 1.0`, where the radius reaches the shell and background floods in. `α = 0.8`
sits just below that cliff — recovering ~0.70–0.85 of each track's co-observations
while admitting little background, leaving the rest to geometric verification.

![Co-observation recall (green, left axis) and background neighbours admitted
(red, right axis) as the floor scale α is swept, with α = 0.8 marked](images/floor-alpha-sweep.png)

### Iterating on cluster membership rules

The membership rule — which neighbours to keep — is the heart of the method, so we
searched it carefully. For each descriptor in a reference reconstruction we save
its 48 nearest neighbours, each labelled as a real co-observation (a neighbour
from the same 3-D point, necessarily in another image) or not. A candidate rule —
"given a descriptor's neighbours, which ones are co-observations?" — can then be
scored against those labels in an instant, with no index to build or
reconstruction to run, so we tried a wide range of rules.

They fell into a few families:

- **Cut at a gap.** Sort a descriptor's neighbour distances and cut where they
  jump up, by various definitions of "jump."
- **A radius from a per-point scale.** Keep neighbours within a multiple of the
  nearest distance, of the second-nearest, or of the background level — the
  distance at which the neighbours flatten out into unrelated features.
- **A fixed count.** Keep the few nearest cross-image neighbours.
- **Mutual agreement.** Keep a neighbour only if the descriptor is, in turn, near
  the top of that neighbour's own list.
- **Combinations.** Pair a generous radius with one of the stricter tests above.

The background-level radius — the floor — was the clear winner on every dataset.
Mutual-agreement tests only hurt, dropping real co-observations. And no single
shared cut-off, even the best one we could find for a dataset, did as well as
letting each descriptor set its own radius from its background. Tuning settled on
keeping neighbours within 0.8× the 28th-nearest distance.

### End-to-end reconstruction vs the baseline

Fed into **incremental** SfM, the cluster matches reconstruct every image and
place the cameras where the baseline does, with a denser point cloud. They go
through COLMAP's geometric verification and the incremental mapper (both seeded),
and the result is compared to the workspace's baseline with `sfm compare`:

| Dataset          | reg   | points (base → cluster) | reproj (base → cluster) | `sfm compare` |
| ---------------- | ----- | ----------------------- | ----------------------- | ------------- |
| seoul_bull       | 17/17 | 1,080 → 1,551           | 0.46 → 0.58 px          | VERY SIMILAR  |
| seattle_backyard | 26/26 |   521 → 4,968           | 0.61 → 0.37 px          | VERY SIMILAR  |
| kerry_park       | 48/48 | 1,128 → 3,193           | 0.31 → 0.78 px          | VERY SIMILAR  |
| dino_dog_toy     | 85/85 | 5,312 → 29,571          | 1.20 → 1.13 px          | VERY SIMILAR  |

`sfm compare` rates all four VERY SIMILAR — which here means the shared cameras'
centres agree (mean position error < 0.1) after a similarity alignment; it does not
look at the point cloud. The cloud is in fact 1.4–9.5× denser than the baseline, at
reprojection error comparable to the baseline's (sub-pixel to ~1 px). COLMAP's
geometric verification filters the cluster correspondences and the incremental
mapper builds the reconstruction from what survives.

### Incremental reconstructs more reliably than global

Those same matches do not reconstruct as dependably under the *global* mapper.
Run through it instead, every image registers and this run's verdicts pass, but
the point counts are erratic — kerry_park keeps less than a third of the points
the incremental mapper recovers from the identical matches:

| Dataset          | reg   | points | `sfm compare`           |
| ---------------- | ----- | ------ | ----------------------- |
| seoul_bull       | 17/17 | 1,430  | VERY SIMILAR            |
| seattle_backyard | 26/26 | 4,903  | VERY SIMILAR            |
| kerry_park       | 48/48 |   926  | VERY SIMILAR            |
| dino_dog_toy     | 85/85 | 23,446 | VERY SIMILAR            |

The deeper problem is run-to-run instability: across repeated end-to-end runs with
slightly different match sets, the incremental mapper has passed all four every
time, while the global mapper's verdicts have ranged from two to four of four —
which datasets pass shifts with the match set and seed, and we don't yet
understand why. Incremental works better on our small set of test datasets.

## Algorithm

### Overview

1. **Index & k-NN query** — concatenate every image's descriptors into one corpus,
   build a nearest-neighbour index, and query it once for the **`d + 1` nearest**
   (self + the `d` nearest others, `d = 28`) of *every* descriptor. The resulting
   `(N, d+1)` table of neighbour ids and aligned distances is the single substrate
   everything below reads from, and the index is not touched again.
2. **Per-point threshold** — for each descriptor, read a cluster membership radius
   off its own neighbour profile (the background floor, §2) and keep the
   cross-image neighbours within it.
3. **Materialize clusters** — walk descriptors densest-first; each unclaimed
   descriptor seeds a cluster from its within-radius cross-image neighbours, one
   feature per image, and claims them (§3). The clusters are the matcher's primary
   output.
4. **Convert to matches** — expand each cluster into its cross-image feature
   pairs, bucketed by image pair (§4); this is a derived view for the pairwise
   pipeline.
5. **Verify & write** — run geometric verification on the pairs
   (`pycolmap.geometric_verification`) and write `.matches` carrying the surviving
   two-view geometry, like every other matcher.

### 1. Index and the shared k-NN query

The corpus is all descriptors `(ΣKᵢ, 128)` (uint8 SIFT). The index is the in-tree
randomized kd-tree forest (`sfmtool.KdForest`,
`crates/sfmtool-core/src/kdforest/`, spec `randomized-kdtree-forest.md`). Each row
carries its `(image_index, feature_index)` so a hit maps back to a feature.

One query drives everything. For every descriptor we fetch its `d + 1 = 29`
nearest, yielding an `(N, 29)` array of neighbour ids and an aligned distance
array, sorted ascending — column 0 is the descriptor itself at distance 0,
columns 1…28 are its 28 nearest others. The query width is exactly what the
membership rule (§2) needs: the last column is the background rank `d`, and the
candidate members are the columns before it; nothing else queries the index.
Since members necessarily lie nearer than rank `d`, this also caps a descriptor's
match degree below `d`.

### 2. Per-point threshold: the background floor

Each descriptor sets its own cluster membership radius from its neighbour
distances. For descriptor `i`, with neighbour distances `dist[i, 0…]` sorted
ascending (Euclidean L2), the **background floor** is its `d`-th-nearest distance,

```
B_i = dist[i, d]          (d = 28)
```

— far enough out to land among unrelated background, past the descriptor's few
genuine co-observations. Keep neighbour `j` of `i` as a member iff

```
dist(i, j) ≤ α · B_i    and    image(j) ≠ image(i)    and    j ≠ i      (α = 0.8)
```

A descriptor with tight co-observations has small early distances and a large
`B_i`, so it keeps them; an isolated descriptor's near distances already sit at
the background scale, so `α · B_i` admits nothing — the isolated-point prefilter
falls out for free. Since `α < 1`, every member is nearer than `B_i`, so cluster
membership only ever reaches ranks below `d`. `d` and `α` are fixed defaults.

#### Why a generous radius

`α < 1` puts the cut *below* the background floor `B_i`, deliberately on the
generous side of the co-observations. The bias is intentional: collecting a few too
many neighbours is cheap — geometric verification rejects the misfits — while
collecting too few loses real observations. The reconstruction is
robust to it, staying close to the baseline across a wide band of radii above the
data boundary; the only failures come from radii that are *too tight* and drop
whole images.

### 3. Materialize clusters

Clusters are built by density-ordered seeding over the k-NN table. Order
descriptors by how many within-radius cross-image neighbours they have, densest
first, and walk that order with a `claimed` bitset: each unclaimed descriptor `s`
seeds a cluster from `s` plus its within-radius (`α · B_s`), cross-image,
still-unclaimed neighbours, resolved to **one feature per image** (nearest `s`).
If the cluster spans at least two images, record it and mark its members claimed;
otherwise drop `s`. The result is a hard partition: each feature belongs to at
most one cluster, each cluster holds at most one feature per image, and a cluster
is a candidate track. Because membership is proximity to the seed rather than
transitive linkage, a chain A–B–C cannot merge two distinct points; the density
ordering forms the best-defined clusters first.

These clusters are the matcher's primary output, kept so later consumers can work
with candidate tracks directly; the pairwise matches below are a view derived
from them.

### 4. Convert clusters to per-image-pair matches

Each cluster of `m` members expands into its `C(m, 2)` cross-image feature pairs,
bucketed by image pair. Because clusters hold one feature per image and the
partition is hard, the resulting matches are already one-to-one per image pair —
no reconciliation pass is needed. Only image pairs that share a cluster appear,
so pair selection falls out of the clustering. The matcher then runs geometric
verification on those pairs and writes the same `.matches` artefact the existing
matchers produce, two-view geometry included; the verifier rejects the pairs that
are not tracks.

### Alternatives considered

- **A global threshold.** A single, data-derived radius `T` for the whole corpus
  in place of the per-point floor, read from the same k-NN table two ways: the
  **per-point cliff** (for each descriptor, the largest jump between consecutive
  sorted neighbour distances separates its likely co-observations from background;
  `T` is a percentile — by default the median — of the just-past-the-cliff
  distance over all descriptors), or the **`d1` bimodal antimode** (`d1` is bimodal
  over the corpus — see Empirical — and the valley between the modes is a
  label-free split, located with Otsu or a 2-component mixture on `log d1`,
  optionally scaled by `t_scale ≈ 1.0–1.25`). The fallback, not the primary path:
  one radius never fits every cluster, and on labelled neighbourhoods the
  per-point floor matches or beats even the best-possible global `T`. An optional
  mean-shift step (re-query at the cluster mean to recentre it) helped the
  global-`T` clusters slightly but does not change reconstruction outcomes.
- **Per-descriptor edges, no materialized clusters.** Keep each descriptor's
  within-radius cross-image neighbours directly as match edges and reconcile them
  to one match per feature per image pair (two passes: keep the smallest-distance
  edge per low-side feature, then per high-side feature). Produces essentially the
  same matches and reconstructions, slightly more cheaply — but leaves no cluster
  artefact for later consumers, which is why materializing clusters is the chosen
  design.
- **Transitive merge.** The connected components of a within-radius graph chain
  distinct points into mega-clusters through repeated structure; not recommended.
  Seeded clusters avoid it by construction.

### Isolated-point prefilter (optional)

The per-point floor already excludes isolated descriptors for free (their nearest
neighbour sits at the background scale, so `α · B_i` admits nothing). Under the
global-threshold alternative it does not, so a descriptor can be dropped up front
if `d1 > T_base` or `d1/d5 > 0.85` (~40–75% of descriptors, all background the
solve discarded), shrinking the problem; it only ever removes would-be singletons.

## Cost Analysis

- **Index build.** The randomized kd-tree forest is cheap to build
  (`O(n log n)` median splits) — far cheaper than a navigable-graph index, whose
  build only amortises over many queries or a persistent index.
- **Query / match.** `ΣKᵢ` queries × `k` neighbours. Exact brute force is
  `O(n²·D)` — fine at test scale (seconds–minutes), prohibitive at realistic
  scale. The forest makes each query sub-linear at a fixed search budget
  (`L_max` leaf visits, not `log n` — the index is approximate; see its spec),
  preserving the downstream match signal close to exact at a tunable precision
  budget.
- **Verification.** Per-image-pair geometric verification, only over pairs that
  have candidate edges (implicit pair selection), which is `≪ N²` on scenes with
  limited covisibility.

At the current (tiny) dataset scale, exact matching would also do and is the
oracle the matcher validates against; the forest is what carries the approach to
realistic corpus sizes. See `specs/core/randomized-kdtree-forest.md` for the
index design.

## Parameters

Primary path (the per-point background floor, §2):

| Parameter     | Default   | Effect                                                 |
| ------------- | --------- | ------------------------------------------------------ |
| `d`           | 10        | background rank: the `d`-th-nearest distance is the background floor `B_i = dist[i, d]`; the query width is `d + 1` |
| `bg_alpha` (α) | 0.8      | keep cross-image neighbours within `α · B_i`; `<1` = generous cut inside the floor |
| `min_size`    | 2         | record a cluster only if it spans at least this many images |

Global-threshold alternative (§"Alternatives considered"):

| Parameter     | Default   | Effect                                                 |
| ------------- | --------- | ------------------------------------------------------ |
| `threshold`   | `cliff`   | global radius estimator: `cliff` / `otsu` / `gmm` / fixed float |
| `cliff_pct`   | 50        | percentile of the per-point just-past-the-cliff distance for `cliff` |
| `t_scale`     | 1.0       | multiply `T_base`; higher = larger clusters; ~1.25 with otsu/gmm |
| `refine`      | 0         | mean-shift centroid steps (re-query at cluster mean); 1 usually suffices |
| `prefilter`   | off       | drop isolated points up front (falls out for free under the floor) |

## Limitations

- **Descriptor distance can't separate every track from background.** For some
  in-track descriptors a background neighbour is nearer than a true
  co-observation, so no radius clusters them cleanly; this remainder concentrates
  in high-multiplicity, repetitive structure and is left to geometric
  verification. The exact fraction is reference-relative — much of the apparent
  non-separability is the lean reference mislabelling real co-observations as
  background — so it is smaller than a single solve suggests.
- **Some members are unreachable by distance.** Wide-baseline observations sit
  beyond any reasonable radius — they have no near co-member (dino recall@5 ≈ 0.46
  even with exact search) — so they never join, and the recovered tracks are
  correspondingly fragmented (a track spans ~1.6–2.7 clusters, ~85% of members
  recovered). Conventional NN+ratio misses the scattered members too, but it is a
  ceiling worth measuring.
- **Repeated structure → false merges**, held off because membership is
  proximity to a fixed seed (no transitive chaining) and by the geometric
  verifier, but not by descriptor distance alone.
- **The hard partition is greedy and order-dependent.** A feature claimed by one
  cluster cannot join a better one later; seeding densest-first mitigates this by
  forming the best-defined clusters before the leftovers.
- **Global SfM is less reliable than incremental on these datasets** (its `sfm
  compare` verdicts range from two to four of four across runs, and which pass
  varies with the match set and seed — above), for reasons we don't yet
  understand.
- **Exact NN does not scale**; in production use the forest, which trades a few
  points of recall (absorbed downstream by geometric verification + track
  redundancy).

## Relationship to Existing Pipeline

Conceptually this matcher replaces the per-pair Lowe ratio test with a per-point,
data-derived distance radius. It runs geometric verification itself, so its
`.matches` output carries two-view geometry and plugs into `sfm solve` and `sfm
to-colmap-db` exactly like the existing matchers.

### Vocabulary trees

COLMAP's `vocab_tree_matcher` also clusters SIFT descriptors, so it is the
natural reference point. A vocabulary tree (Nistér & Stewénius 2006) clusters a
large *training* corpus of descriptors **offline** into a hierarchical k-means
tree whose centroids are coarse "visual words"; each image becomes a bag of those
words, and bag-of-words similarity **retrieves candidate image pairs**, which are
then matched and verified normally. That offline k-means is itself centroid
iteration accelerated by a randomized kd-forest (Philbin et al. 2007) — the same
index this matcher uses.

This method reuses the "cluster descriptors" idea at a different granularity and
stage. Rather than a coarse, reusable vocabulary built offline on a separate
corpus, it clusters the reconstruction's **own** descriptors online into tight,
**track-scale** groups, and those clusters *are* the candidate correspondences —
not a retrieval index. A visual word is a large cell of descriptor space shared
by many unrelated features across the world; a cluster here aims to be the
observations of a single 3-D point. And the implicit pair selection it gets for
free (only image pairs that share a cluster are verified) does the same job the
vocabulary tree does for COLMAP — avoiding `O(N²)` pair enumeration — but folded
into the matching step instead of a separate retrieval stage.

## Implementation Status

The approach was validated end to end by a Python prototype over the in-tree
`sfmtool.KdForest` index — the empirical results above are from it. The production
form is a `sfmtool-core` matcher over the kd-tree forest, exposed as a matching
method in `feature_match/` and selectable from `sfm match` / `sfm solve`; the
per-point background floor is the recommended membership rule, and its production
API is specified in [Production Implementation](#production-implementation) below.

The production implementation is **done** as specified: the Rust matcher lives in
`crates/sfmtool-core/src/cluster_match/`, the PyO3 bindings in
`crates/sfmtool-py/src/py_cluster_match.rs` (exposed as
`sfmtool.background_floor_clusters` / `sfmtool.clusters_to_pair_matches`), the
Python matcher layer in `src/sfmtool/feature_match/_cluster_matching.py` with the
`_run_cluster_matching` orchestration in `feature_match/_run.py`, and the CLI as
`sfm match --cluster` (see `specs/cli/match-command.md`). A `sfm solve --cluster`
shortcut remains future work.

The production matcher reproduces the prototype's end-to-end results: run
through `sfm match --cluster` + seeded incremental `sfm solve -i` on all four
datasets (cluster corpus at each dataset's full extraction budget), every image
registers, `sfm compare` against the baseline rates all four VERY SIMILAR, and
the point clouds land where the table above predicts — seoul_bull 17/17 at
1,550 points, seattle_backyard 26/26 at 4,980, kerry_park 48/48 at 3,153,
dino_dog_toy 85/85 at 29,644.

**Default `d` lowered 28 → 10 after a production sweep (2026-06-10).** The
prototype tuned `d = 28`; sweeping the production matcher at
`d ∈ {6, 7, 8, 9, 10, 14, 20, 28}` across all four datasets (match + seeded
incremental solve per point) showed the wide floor was paying for itself in
solve time, not quality. Findings:

- Registration is full (17/17, 26/26, 48/48, 85/85) at every `d ≥ 8`;
  kerry_park drops to 46/48 at `d ∈ {6, 7}`, locating the cliff just below 8.
- Smaller `d` is faster end to end — mostly in the *solve*, which scales with
  the candidate matches a wider floor admits (kerry 55 s at `d = 8` vs 108 s at
  28; dino 99 s vs 147 s total).
- Mean reprojection error *improves* monotonically as `d` shrinks on every
  dataset (e.g. seoul 0.47 px at 8 vs 0.58 px at 28): the extra matches a wide
  floor admits are disproportionately the weak ones.
- Total points dip on the small scenes (seoul −20%, kerry −15% at `d = 10` vs
  28) but the lost points are mostly 2-view: the fraction of points with ≥ 3
  observations is far higher at small `d` (92–97% vs 77–84%), and dino's point
  count actually rises (32,181 vs 29,657).

`d = 10` was chosen as the default: measured directly on all four datasets,
two ranks of margin above the kerry_park registration cliff, ~1.5–2.4×
end-to-end speedup vs 28, better reprojection everywhere. The original `d = 28`
remains a reasonable choice for unusually high-covisibility collections
(features co-observed in tens of images), where a small rank could read the
floor inside the track itself; pass `--cluster-d` to raise it.

## Production Implementation

This section specifies the production form of the **background-floor** matcher
across three layers — a Rust matcher in `sfmtool-core`, its PyO3 binding, and the
`sfm match` CLI wiring — in enough detail for a fresh implementation. The
algorithm and its justification are above; this section pins down the API and data
flow, centred on
[§2 Per-point threshold: the background floor](#2-per-point-threshold-the-background-floor).

### What the matcher does (one paragraph)

Given the SIFT descriptors of every image in a set, concatenate them into one
corpus, build a randomized kd-tree forest over it, and query each descriptor's
`k` nearest neighbours. For each descriptor, set a *per-point* radius from its own
background floor — `alpha ×` its `d`-th-nearest distance. Materialize clusters by
density-ordered seeding under those radii (one feature per image, hard partition)
— the clusters are the primary output, returned to the caller for downstream
consumers. A second function converts clusters to per-image-pair matches (each
cluster's `C(m,2)` cross-image pairs, already one-to-one per pair); the CLI then
runs geometric verification and writes a `.matches` file with the surviving
two-view geometry, for the existing `sfm solve` / `sfm to-colmap-db` consumers.

### Distance space (read this first)

All distances in this matcher are **Euclidean L2** (square-rooted), not squared.
This matters because the tuned defaults `alpha = 0.8`, `d = 10` were fit in L2
space (via Python `KdForest.query`, which returns L2).

The core `KdForest::search_batch_with_distances` returns **squared** L2
(`dist_sq`). **The matcher must take the square root of those distances before
computing the background floor and the radius test.** Distances written to
`match_descriptor_distances` are likewise L2, matching every other matcher's
`.matches` output.

### Layer 1 — Rust core (`sfmtool-core`)

#### Location

New module `crates/sfmtool-core/src/cluster_match/` with `mod.rs`; declare
`pub mod cluster_match;` in `crates/sfmtool-core/src/lib.rs`. Optionally re-export
the public types from `lib.rs` alongside the other `pub use` lines.

#### Public types

```rust
use ndarray::{Array1, Array2, ArrayView2};
use crate::kdforest::KdForestParams;

/// Tuning for the background-floor matcher. `Default` is the production config.
/// The k-NN query width is derived, not configured: `d + 1` (self + the `d`
/// nearest others), exactly enough that the background rank is the last column.
#[derive(Clone, Debug)]
pub struct BackgroundFloorParams {
    /// Background rank: the `d`-th-nearest distance is the background floor
    /// `B_i = dist[i, d]`. Default 10.
    pub d: usize,
    /// Radius multiplier: keep neighbours within `alpha * B_i`. Default 0.8.
    pub alpha: f32,
    /// Record a cluster only if it spans at least this many images. Default 2.
    pub min_size: usize,
    /// Index build + per-query search budget. Default `KdForestParams::accurate()`.
    pub forest: KdForestParams,
}

impl Default for BackgroundFloorParams {
    fn default() -> Self {
        Self {
            d: 10,
            alpha: 0.8,
            min_size: 2,
            forest: KdForestParams::accurate(),
        }
    }
}

/// Materialized track clusters — the matcher's primary output. CSR layout:
/// cluster `c` owns members `cluster_starts[c] .. cluster_starts[c+1]`. Within a
/// cluster, members are sorted by image index, and a cluster holds at most one
/// feature per image (so member count == image span). Clusters are disjoint (a
/// hard partition of the participating features).
pub struct Clusters {
    /// `(C + 1,)` CSR offsets into the member arrays.
    pub cluster_starts: Array1<u32>,
    /// `(M,)` member image index, aligned with `member_features`.
    pub member_images: Array1<u32>,
    /// `(M,)` member feature index (row in that image's `.sift` file).
    pub member_features: Array1<u32>,
}

/// Cross-image matches, in the parallel-array form the `.matches` writer wants.
pub struct PairMatches {
    /// `(P, 2)` image-index pairs, each `[i, j]` with `i < j`, sorted ascending
    /// by `(i, j)`.
    pub image_index_pairs: Array2<u32>,
    /// `(P,)` number of matches in each pair; `sum == M`. Aligned to
    /// `image_index_pairs`.
    pub match_counts: Array1<u32>,
    /// `(M, 2)` feature-index pairs `[feat_i, feat_j]`, grouped by pair in the
    /// same order as `image_index_pairs`. `feat_i` indexes image `i`'s `.sift`
    /// rows, `feat_j` indexes image `j`'s.
    pub match_feature_indexes: Array2<u32>,
    /// `(M,)` Euclidean L2 descriptor distance per match, aligned to
    /// `match_feature_indexes`.
    pub match_descriptor_distances: Array1<f32>,
}

#[derive(Debug, thiserror::Error)]  // or a hand-rolled enum, matching crate style
pub enum ClusterMatchError {
    #[error("descriptor corpus is empty")]
    EmptyCorpus,
    #[error("corpus has {n} descriptors; need more than d ({d}) for the floor")]
    CorpusSmallerThanFloor { n: usize, d: usize },
    #[error("image_starts must be non-decreasing, start at 0, and end at N ({n})")]
    BadOffsets { n: usize },
}
```

> Use whichever error idiom the crate already uses — check neighbouring modules
> (e.g. `reconstruction`, `camera_intrinsics`) and match it; `thiserror` above is
> illustrative.

#### Public functions

```rust
/// Background-floor track-cluster matcher: materialize the clusters.
///
/// `descriptors` is the `(N, D)` corpus of every image's SIFT descriptors,
/// concatenated image by image (uint8, D = 128). `image_starts` is a CSR-style
/// offset array of length `n_images + 1`: image `i` owns rows
/// `image_starts[i] .. image_starts[i+1]`, and row `r` of that image has
/// feature index `r - image_starts[i]`. Returns the materialized clusters —
/// the primary artefact.
pub fn background_floor_clusters(
    descriptors: ArrayView2<'_, u8>,
    image_starts: &[u32],
    params: &BackgroundFloorParams,
) -> Result<Clusters, ClusterMatchError>;

/// Derived view: expand clusters into one-to-one-per-image-pair cross-image
/// matches (each cluster's C(m,2) pairs, bucketed by image pair). `descriptors`
/// and `image_starts` must be the same arrays the clusters were built from;
/// they supply the L2 match distances.
pub fn clusters_to_pair_matches(
    clusters: &Clusters,
    descriptors: ArrayView2<'_, u8>,
    image_starts: &[u32],
) -> PairMatches;
```

#### Algorithm (exact)

Let `N = descriptors.nrows()`, `D = descriptors.ncols()` (128),
`n_images = image_starts.len() - 1`.

1. **Validate.** `N > params.d` (the floor rank must exist in the corpus);
   `image_starts[0] == 0`, non-decreasing, `image_starts[n_images] == N`. Else
   return the matching `ClusterMatchError`. Let `k = params.d + 1` — the query
   width, derived, not configured.

2. **Row → (image, feature) maps.** From `image_starts`, build `image_of[r]`
   (`u32`, the owning image) and `feature_of[r]` (`u32`, `r - image_starts[image]`)
   for every row `r`. (A binary search over `image_starts`, or a single linear
   fill, both fine.)

3. **Build the forest.** `KdForest::build` over the flat `descriptors` slice
   (row-major `N*D` `u8`), `dim = D`, with `params.forest`. The corpus must be
   contiguous; if `descriptors` is not standard layout, copy to a `Vec<u8>` first.

4. **Query.** `let (idx, dist_sq) = forest.search_batch_with_distances(corpus, N,
   k, params.forest.max_leaf_checks, None);` — flat `N*k` arrays, each row
   sorted ascending with column 0 = self at distance 0. Unfound slots are
   `u32::MAX` / `f32::INFINITY` (ruled out for the floor column by the `N > d`
   validation, modulo forest misses; treat an infinite floor as "keep nothing").

5. **L2 distances.** Materialise `dist[r*k + c] = dist_sq[...].sqrt()`.

6. **Per-point floor & radius.** For each row `i`, the background floor is its
   `d`-th-nearest distance, `B_i = dist[i, d]`. The row is already sorted ascending,
   so this is a direct index — no scan. `radius_i = alpha * B_i`.

7. **Candidate neighbours.** For each row `i`, its candidates are the neighbour
   columns `c` in `0..k` with `j = idx[i, c]` satisfying `j != u32::MAX`,
   `j != i`, `image_of[i] != image_of[j]`, and `dist[i, c] <= radius_i`. (Self at
   column 0 is dropped by `j != i`.) Record each row's candidate count for the
   density ordering.

8. **Materialize clusters (density-ordered claim).** Sort rows by candidate
   count, descending, with row index as the tie-break (a fixed, deterministic
   order). Walk that order with a `claimed: Vec<bool>`:
   - Skip `s` if already claimed.
   - Gather `s` plus its still-unclaimed candidates; resolve to **one feature per
     image**, keeping the nearest to `s` per image (`s` itself wins its own
     image at distance 0).
   - If the members span ≥ `params.min_size` images, append the cluster (members
     sorted by image index) and mark all members claimed; otherwise mark only `s`
     claimed and drop it.

   This loop is inherently sequential (claims are global state); it is cheap —
   one pass over `N` rows with ≤ `k` work each. Emit `Clusters` in CSR form as
   the function result.

9. **Convert to pairs (`clusters_to_pair_matches`).** For each cluster, emit all
   `C(m, 2)` member pairs `(img_lo, img_hi, feat_lo, feat_hi, dval)` — members
   are sorted by image so `img_lo < img_hi` directly — with `dval` the L2
   distance between the two members' descriptors (uint8 rows → f32, sqrt of the
   squared distance). Because clusters hold one feature per image and are
   disjoint, the pairs are already one-to-one per image pair; no reconciliation
   pass is needed.

10. **Assemble.** Sort the emitted pairs by `(img_lo, img_hi)`. Produce:
    `image_index_pairs` = the distinct sorted pairs; `match_counts` = per-pair
    edge counts; `match_feature_indexes` = `[feat_lo, feat_hi]` rows grouped by
    pair in that order; `match_descriptor_distances` = the aligned `dval` values.

#### Parallelism

Steps 6–7 are embarrassingly parallel over rows — use `rayon` (`par_iter` /
`par_chunks`) as elsewhere in the crate. Step 4's `search_batch_with_distances`
is already internally parallel. Step 8 is sequential by design (the claim order
defines the result); it is a single cheap pass. Step 9 parallelises per cluster,
and step 10 can use a parallel sort (`rayon`'s `par_sort_unstable_by`). Keep
memory bounded: candidates are `≤ N*d`; for dino (`N ≈ 600k`, `d = 28`) that
is ~17M before clustering — fine as flat `Vec`s of primitives, but do not build
per-edge structs with heap fields.

#### Determinism

Given a fixed `params.forest.seed`, the matcher is deterministic: the density
order's tie-break is the row index, per-image resolution prefers the smaller
distance then the smaller row index, and the conversion order is fixed by the
cluster order. Output arrays are byte-stable across runs and platforms.

#### Tests (`crates/sfmtool-core/src/cluster_match/tests.rs`)

- **Synthetic clusters.** Build a tiny corpus: a few "points" each with one
  descriptor in 3–4 images (tight intra-cluster distance) plus scattered
  background. Assert the matcher recovers each planted point as one cluster and
  that no cluster mixes two planted points.
- **Cluster invariants.** Assert clusters are disjoint (no feature in two
  clusters), each holds at most one feature per image, members are sorted by
  image, and every cluster spans ≥ `min_size` images.
- **Conversion.** Assert `clusters_to_pair_matches` emits exactly `Σ C(mᵢ, 2)`
  pairs, that within every image pair no `feat_lo` and no `feat_hi` repeats, and
  that a returned distance equals the true L2 between the two descriptors (not
  squared).
- **Validation.** A corpus with `N <= d` descriptors and malformed
  `image_starts` return the right `ClusterMatchError`.
- **Determinism.** Two runs with the same seed produce byte-identical arrays.

Run `pixi run cargo test -p sfmtool-core cluster_match` and
`pixi run cargo clippy --workspace` / `pixi run cargo fmt`.

### Layer 2 — PyO3 binding (`sfmtool-py`)

#### Location

New file `crates/sfmtool-py/src/py_cluster_match.rs`; `mod py_cluster_match;` and
registration of both functions in `crates/sfmtool-py/src/lib.rs`’s `#[pymodule]`
via `m.add_function(wrap_pyfunction!(...))`.

#### Functions

Mirror the `KdForest` binding conventions (`py_kdforest.rs`): validate the uint8
dtype, make the corpus contiguous, release the GIL around the heavy call, and
return numpy arrays via `PyArray1::from_vec(...).reshape(...).into_any().unbind()`.

```rust
/// Background-floor track-cluster matcher: materialize the clusters.
///
/// Args:
///     descriptors: (N, 128) uint8 corpus, every image's SIFT descriptors
///         concatenated image by image.
///     image_starts: (n_images + 1,) uint32 CSR offsets; image i owns rows
///         image_starts[i]:image_starts[i+1].
///     d, alpha, min_size: background-floor parameters (defaults 10, 0.8, 2);
///         the query width is derived as d + 1.
///     preset / num_trees / leaf_size / max_leaf_checks / seed: forest config,
///         same meaning as KdForest.
///
/// Returns (CSR clusters — the primary artefact):
///     (cluster_starts (C+1,) uint32, member_images (M,) uint32,
///      member_features (M,) uint32)
#[pyfunction]
#[pyo3(signature = (descriptors, image_starts, d=10, alpha=0.8, min_size=2,
                    preset=None, num_trees=None, leaf_size=None,
                    max_leaf_checks=None, seed=None))]
pub fn background_floor_clusters<'py>(...) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)>;

/// Derived view: expand clusters into per-image-pair matches.
///
/// Args:
///     cluster_starts / member_images / member_features: the arrays returned by
///         background_floor_clusters.
///     descriptors, image_starts: the same corpus the clusters were built from
///         (supplies the L2 match distances).
///
/// Returns:
///     (image_index_pairs (P,2) uint32, match_counts (P,) uint32,
///      match_feature_indexes (M,2) uint32, match_descriptor_distances (M,) float32)
#[pyfunction]
pub fn clusters_to_pair_matches<'py>(...) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)>;
```

Build `KdForestParams` from `preset` + overrides exactly as `PyKdForest::new`
does (reuse that resolution logic — consider lifting it into a shared helper).
Map `ClusterMatchError` to `PyValueError::new_err(...)`. Validate
`descriptors.ndim == 2`, `ncols == 128`, dtype `uint8`, and
`image_starts.len() == n_images + 1` with a clear message.

#### Python package surface

`KdForest` is re-exported as `sfmtool.KdForest` (see `src/sfmtool/__init__.py`).
Re-export both functions the same way so callers can `from sfmtool import
background_floor_clusters, clusters_to_pair_matches` (they live in
`sfmtool._sfmtool`).

#### Rebuild + tests

After the Rust edits, **`pixi run maturin develop --release`** (the `.so` does
not auto-rebuild). Add `tests/rust_bindings/test_cluster_match_rust_bindings.py`:

- A tiny hand-built corpus (numpy) with a couple of planted cross-image points;
  assert the cluster arrays have the documented shapes/dtypes and CSR validity
  (`cluster_starts[0] == 0`, non-decreasing, ends at `len(member_images)`), that
  planted points come back as clusters, and that feeding the clusters through
  `clusters_to_pair_matches` yields sorted pairs with `i < j`,
  `match_counts.sum() == len(match_feature_indexes)`, and the planted matches.
- Dtype/shape errors raise `ValueError`/`TypeError`.

### Layer 3 — Python matcher layer + CLI

#### Matcher module

New `src/sfmtool/feature_match/_cluster_matching.py`, mirroring
`_flow_matching.py`:

```python
def cluster_match(
    image_paths: list[Path],
    sift_paths: list[Path],
    *,
    d: int = 10,
    alpha: float = 0.8,
    min_size: int = 2,
    preset: str = "accurate",
    max_feature_count: int | None = None,
) -> tuple[ClusterSet, PairArrays]:
    """Run the background-floor matcher over every image's SIFT descriptors.

    Loads each image's descriptors (capped at max_feature_count to match the
    feature indices used downstream), concatenates them into one (N, 128) uint8
    corpus with a CSR image_starts array, and calls
    sfmtool.background_floor_clusters followed by
    sfmtool.clusters_to_pair_matches. Returns both: the clusters
    (cluster_starts, member_images, member_features — the primary artefact) and
    the four parallel pair arrays (image_index_pairs, match_counts,
    match_feature_indexes, match_descriptor_distances) for the .matches writer.
    """
```

(`ClusterSet` / `PairArrays` here are just named tuples of the numpy arrays —
use whatever lightweight container fits the module's style.)

Load descriptors with `SiftReader(sift_path).read_descriptors(count=...)` (see
`src/sfmtool/sift/file.py`). Build `image_starts` as the cumulative feature
counts. Feature indices in the result are `.sift` row indices (capped to the
first `max_feature_count` when set), consistent with how `to-colmap-db` loads
keypoints.

#### Verifying and writing the `.matches` file

Run geometric verification and embed the two-view geometry, exactly as the
existing matchers do — `_run_matching` builds a temporary COLMAP DB, matches
(which verifies), then `read_colmap_db_matches(db, include_tvg=True)` →
`write_matches`. The cluster matcher reuses the **same back half**, seeding the DB
with the cluster pairs instead of running a pycolmap matcher:

1. Populate a temporary COLMAP DB with the images' features
   (`_populate_db_features`, as `_run_matching` does).
2. Write the cluster pair arrays into the DB (`_write_matches_to_db` from
   `_colmap_db.py`, or `pycolmap.Database.write_matches` per pair).
3. `pycolmap.geometric_verification(db_path)` — populates the two-view geometries.
4. `matches_data = read_colmap_db_matches(str(db_path), include_tvg=True)` —
   reads the surviving matches **and** their TVG back, so
   `has_two_view_geometries = True`.
5. `_compute_descriptor_distances(matches_data, sift_paths, max_feature_count)`,
   fill metadata, then `write_matches(out, matches_data)`.

Set `matching_method = "cluster"`, `matching_tool = "sfmtool"`,
`matching_tool_version` = the package version, and record the parameters in
`matching_options` (`{"mode": "background-floor", "d": d, "alpha": alpha,
"min_size": min_size, "preset": preset}`). The resulting `.matches` carries
two-view geometry, like every other matcher's output.

The CLI consumes only the pair output today; the clusters are returned to the
orchestrator so a future cluster artefact (a persisted cluster file, cluster
visualisation, track seeding) can be added without touching the matcher.
Persisting clusters to disk is **out of scope** here.

> Factor the verify-and-write back half so the cluster path and the existing
> paths share it rather than duplicating the DB / TVG plumbing.

Add an orchestrator in `_run.py`, e.g. `_run_cluster_matching(image_paths,
output_path, *, d, alpha, min_size, preset, max_feature_count,
workspace_dir)`, that resolves `.sift` paths via `image_files_to_sift_files`,
calls `cluster_match`, runs the verify-and-write back half above, and returns the
output path. Because the matches now carry two-view geometry, the default output
goes under `workspace/tvg-matches/` (the same place the existing verified matchers
write), not `matches/`.

#### CLI: extend `sfm match`

Add a fourth matching **method** to `src/sfmtool/_commands/match.py`, mutually
exclusive with `--exhaustive` / `--sequential` / `--flow`:

```
--cluster                 use the background-floor track-cluster matcher
--cluster-alpha FLOAT     background-floor radius multiplier (default 0.8)
--cluster-d INT           background rank: d-th-nearest distance sets the floor (default 10)
--cluster-preset CHOICE   forest preset: accurate|balanced|fast (default accurate)
```

`--cluster` dispatches to `_run_cluster_matching` over the resolved image set and
honours `--max-features`, `--output`, and `--range` (range restricts the image
set the corpus is built from). Update the method-selection / mutual-exclusion
validation and the help text. Keep it in the **Image Feature** category (already
where `match` is registered in `cli.py`).

**Camera model / camera_config.** The clustering itself uses no intrinsics or
poses, but the geometric-verification step that turns clustered pairs into
two-view geometries does. `--camera-model` therefore stays available with
`--cluster`: it is written into the COLMAP database (exactly as for the other
matchers) and used to estimate each pair's two-view geometry. The existing
`_check_camera_model_conflict` (`src/sfmtool/camera/setup.py`) still applies
across all methods, so `--camera-model` is rejected when a `camera_config.json`
resolves for an image.

#### CLI spec

Add a short `## Cluster matching` section to `specs/cli/match-command.md`
describing `--cluster` and its options, and link back to this section.

#### Downstream (no new code)

The emitted `.matches` flows through the existing consumers unchanged:

```bash
pixi run sfm match --cluster images -o tvg-matches/cluster.matches
pixi run sfm to-colmap-db tvg-matches/cluster.matches colmap.db
pixi run sfm solve -i tvg-matches/cluster.matches                       # incremental SfM
```

Because the `.matches` already carries two-view geometry, `sfm to-colmap-db` /
`sfm solve` write the embedded TVG straight into the COLMAP DB
(`_setup_for_sfm_from_matches` / `_write_matches_to_db` in `_colmap_db.py`) — no
re-verification needed. A `sfm solve --cluster` shortcut (match then map in one
call) is a reasonable follow-on but is **out of scope** here; the
`match → .matches → solve` path is
the deliverable.

#### Tests

- **Unit**: `tests/test_cluster_matching.py` — small synthetic descriptor sets
  through `cluster_match`, asserting cluster invariants (disjoint, one feature
  per image, spans ≥ `min_size` images) and one-to-one-per-pair on the converted
  matches.
- **Integration**: using the `isolated_seoul_bull_17_images` fixture (see
  `tests/conftest.py`), run `sfm match --cluster` and assert a `.matches` file is
  produced with the expected pair/match counts > 0 and `has_two_view_geometries`
  True (geometric verification ran); optionally feed it to `to-colmap-db` and
  assert a DB is created.
- `pixi run fmt && pixi run check` for the Python changes.

### Production defaults (single source of truth)

| Parameter | Default      | Layer(s)            | Meaning                                            |
| --------- | ------------ | ------------------- | -------------------------------------------------- |
| `d`       | 10           | core/py/cli         | background rank: `B_i = dist[i, d]`; query width is `d + 1` (derived) |
| `alpha`   | 0.8          | core/py/cli         | keep neighbours within `alpha · B_i`               |
| `min_size`| 2            | core/py/cli         | record a cluster only if it spans ≥ this many images |
| `preset`  | `accurate`   | core/py/cli         | forest build + search budget                       |
| distance  | Euclidean L2 | all                 | sqrt of the forest's squared distances             |
| TVG       | embedded     | py/cli              | matcher runs geometric verification; `.matches` carries two-view geometry |

These are the tuned values from the empirical sections above (`d` re-tuned by
the 2026-06-10 production sweep — see Implementation Status); do not change them
without re-running the membership-rule bench and the end-to-end reconstructions.

### Implementation order (suggested)

1. Rust `cluster_match` module + unit tests → `cargo test`/`clippy`/`fmt`.
2. PyO3 binding + `maturin develop --release` + `tests/rust_bindings/...`.
3. `_cluster_matching.py` + `_run.py` orchestrator + Python unit test.
4. `sfm match --cluster` wiring + integration test + `specs/cli` note.
5. `fmt && check`, full `pixi run test`, update this section if anything diverged.
