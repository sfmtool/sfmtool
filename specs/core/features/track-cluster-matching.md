# Track-Cluster Matching

Track-cluster matching finds correspondences across a whole image set at once.
It puts every image's SIFT descriptors into one nearest-neighbour index and
groups each descriptor with the descriptors from other images that lie within
0.8× its distance to its 10th-nearest neighbour, forming clusters of at most one
feature per image that are candidate tracks.

Conventional SfM matching works one image pair at a time: it enumerates pairs,
matches descriptors within each pair, verifies each pair geometrically, and only
then joins the pairwise matches into tracks. This matcher starts from the
property a track already has. The observations of one surface point are the same
point seen from different images, so their descriptors are close to each other
in descriptor space, and a search over all descriptors at once can find them
together. The image pairs to verify are then the pairs that share a cluster, so
pair selection needs no separate step.

## Approach

Querying the index for a descriptor's nearest neighbours returns the other
members of its cluster, interleaved with unrelated background. The problem is
deciding where each descriptor's cluster ends: which neighbours are genuine
co-observations and which are background.

The matcher decides this per descriptor from one nearest-neighbour query. A
descriptor's sorted neighbour distances rise gently across its true
co-observations, then jump up to the level of unrelated features, its
*background floor*. A fixed fraction of that floor is the descriptor's cluster
membership radius, and its cross-image neighbours within that radius are its
candidate co-observations. A descriptor with tight co-observations keeps them;
an isolated descriptor, whose nearest neighbour already sits at the background
level, keeps nothing.

![Neighbour-distance profiles for example descriptors (seattle_backyard): the
co-observations (green) sit at the near ranks below the cluster membership radius α·B
(dashed), while unrelated background (grey) plateaus above it; the background
scale B is the d-th-nearest distance (dotted).](images/floor-profile.png)

The clusters are the matcher's output, so consumers can work with candidate
tracks directly. A derived view expands them into matches between image pairs
for the pairwise pipeline: geometric verification and COLMAP's mappers.

## Interface

The matcher lives in
[`features/cluster_match/mod.rs`](../../../crates/sfmtool-core/src/features/cluster_match/mod.rs),
declared in [`features/mod.rs`](../../../crates/sfmtool-core/src/features/mod.rs).
The same module directory holds
[`covisibility.rs`](../../../crates/sfmtool-core/src/features/cluster_match/covisibility.rs)
and [`covisibility/`](../../../crates/sfmtool-core/src/features/cluster_match/covisibility/),
which count the clusters each image pair shares; they are specified in
[cluster-covisibility.md](cluster-covisibility.md) and
[covisibility-selection.md](covisibility-selection.md), not here. The bindings
are in [`matching/cluster.rs`](../../../crates/sfmtool-py/src/matching/cluster.rs),
bound as `sfmtool.matching`. The Python matcher layer is
[`_cluster_matching.py`](../../../src/sfmtool/feature_match/_cluster_matching.py),
and the orchestration for `sfm match --cluster` is in
[`_run.py`](../../../src/sfmtool/feature_match/_run.py).

### Rust

| Item | What it is |
| ---- | ---------- |
| `BackgroundFloorParams { d, alpha, min_size, forest }` | Matcher parameters; `Default` is the production configuration (table below). The query width is not a parameter: it is always `d + 1`. |
| `background_floor_clusters(descriptors, image_starts, params)` | Builds a forest over an in-memory `(N, 128)` `u8` corpus, runs the k-NN self-join, and clusters the result. |
| `background_floor_clusters_lazy(forest, image_starts, params, progress)` | The same clustering over a `LazyKdForestU8` that stays on disk (a `.kdf` file). Reports the join's answered-query count to `progress` and stops on cancel with `LazyClusterError::Kdf(KdfError::Cancelled)`; the clustering after the join is not interrupted. |
| `background_floor_clusters_from_neighbors(n, image_starts, params, neighbors)` | The clustering stage alone, over a `NeighborTable` computed elsewhere. Both functions above end by calling it. |
| `NeighborTable { indexes, distances_sq, width }` | Row-major `N × width` k-NN table, nearest first, **squared** L2 distances, `u32::MAX` / `f32::INFINITY` padding unfilled slots. `width` must equal `d + 1`. |
| `Clusters { cluster_starts, member_images, member_features }` | The output, in CSR form: cluster `c` owns members `cluster_starts[c] .. cluster_starts[c + 1]`. |
| `clusters_to_pair_matches(clusters, descriptors, image_starts)` | The derived pairwise view, `PairMatches`: image pairs `[i, j]` with `i < j` sorted ascending, per-pair match counts, feature-index pairs grouped by image pair, and per-match L2 descriptor distances. `descriptors` and `image_starts` must be the ones the clusters came from. |
| `ClusterMatchError` | Five variants: `EmptyCorpus`; `CorpusSmallerThanFloor` (`N ≤ d`, so the floor rank does not exist); `BadOffsets` (`image_starts` does not start at 0, is not non-decreasing, or does not end at `N`); and, from `_from_neighbors` only, `BadNeighborTable` (wrong width or length) and `BadNeighborIndex` (an index that is neither a row nor `u32::MAX`). |
| `LazyClusterError` | `Cluster(ClusterMatchError)` or `Kdf(KdfError)` for a read failure or a cancel. |

`image_starts` has length `n_images + 1`: image `i` owns corpus rows
`image_starts[i] .. image_starts[i + 1]`, and row `r` of image `i` is feature
`r - image_starts[i]`, the row index in that image's `.sift` file. The errors
are a hand-written enum with `Display` and `std::error::Error`.

The clustering is split from the search because it never reads a descriptor,
only neighbour indexes and distances. That lets one clustering run over a table
from the in-memory forest, from a `.kdf` file, or from any other search, and it
lets a corpus too large to hold in memory be clustered once its neighbours are
known. The `.kdf` path holds only the file's bounded cache and the
`N × (d + 1)` table. It returns the same clusters as the in-memory matcher over
the forest the file was written from, at the same per-query budget, because
the file stores that forest's topology and leaf order. A `.kdf` stores no
search budget, so the caller always supplies `params.forest.max_leaf_checks`.

```rust
use sfmtool_core::features::cluster_match::{
    background_floor_clusters, clusters_to_pair_matches, BackgroundFloorParams,
};

// corpus: Array2<u8> of shape (N, 128); image_starts: Vec<u32> of length n_images + 1.
let clusters = background_floor_clusters(corpus.view(), &image_starts, &BackgroundFloorParams::default())?;
let pairs = clusters_to_pair_matches(&clusters, corpus.view(), &image_starts);
```

### Contract the signatures do not show

- **Distances are Euclidean L2, not squared.** The forest and `NeighborTable`
  carry squared L2; the matcher takes the square root before computing the
  floor and the radius test, and `PairMatches` reports L2. The defaults
  `alpha = 0.8`, `d = 10` were tuned in L2.
- **Rank cap.** With `alpha < 1` and a positive floor, every kept neighbour is
  strictly nearer than the floor `B_i`, so membership only reaches the columns
  before `d`. With `alpha ≥ 1` the floor neighbour itself can pass, so the
  candidate buffer is sized at the full query width `d + 1`.
- **Missing neighbours.** A padded slot is skipped. A row whose floor slot is
  padded (the search found fewer than `d + 1` neighbours) has an infinite floor,
  so the radius test passes every neighbour the search did find.
- **Deterministic order.** Seeds are visited by candidate count, descending,
  with the smaller row index first on a tie. Within a cluster, one feature per
  image is kept: the one nearest the seed, and on equal distance the smaller
  row index. Members are stored sorted by image index. For a fixed forest seed
  the output arrays are the same on every run.
- **Hard partition.** No feature is in two clusters, no cluster holds two
  features of one image, and every cluster spans at least `min_size` images.
  So the pair expansion is already one-to-one within each image pair.
- **Input checks in the pair expansion.** `clusters_to_pair_matches` in Rust
  trusts its input and panics on an out-of-range member. The binding checks the
  CSR arrays and every member's image and feature index first, and raises
  `ValueError`.

### Python

`sfmtool.matching` binds three functions, each returning numpy arrays:

| Function | Returns |
| -------- | ------- |
| `background_floor_clusters(descriptors, image_starts, d=10, alpha=0.8, min_size=2, preset=None, num_trees=None, leaf_size=None, max_leaf_checks=None, seed=None)` | `(cluster_starts, member_images, member_features)`, all `uint32`. The forest arguments mean what they mean for `KdForest`; the preset defaults to `"accurate"`. |
| `background_floor_clusters_kdf(path, image_starts, d=10, alpha=0.8, min_size=2, max_leaf_checks=128, cache_bytes=None, max_chunk_bytes=None, query_workers=None)` | The same tuple, from a `.kdf` file through `background_floor_clusters_lazy`. `image_starts` must follow the feature-ID order the file was written in. |
| `clusters_to_pair_matches(cluster_starts, member_images, member_features, descriptors, image_starts)` | `(image_index_pairs (P, 2), match_counts (P,), match_feature_indexes (M, 2), match_descriptor_distances (M,) float32)`. |

The bindings require `uint8` descriptors of width 128 and `uint32` offsets
(`TypeError` for a wrong dtype), reject `d = 0`, and map `ClusterMatchError` to
`ValueError`; the `.kdf` binding raises `OSError` for a malformed or damaged
file. They release the GIL around the core call.

`cluster_match(image_paths, sift_paths, *, d, alpha, min_size, preset,
max_feature_count)` in `_cluster_matching.py` reads each image's descriptors
(the first `max_feature_count` rows when it is set) through a thread pool,
concatenates them in the order given, and returns a `ClusterSet` and a
`PairArrays`, named tuples of the arrays above.

### Defaults

These are the only definitions of the defaults; the CLI flags and bindings use
the same values.

| Parameter | Default | Set by | Meaning |
| --------- | ------- | ------ | ------- |
| `d` | 10 | `--cluster-d` | background rank: the floor is `B_i = dist[i, d]`; the query width is `d + 1` |
| `alpha` | 0.8 | `--cluster-alpha` | keep cross-image neighbours within `alpha · B_i` |
| `min_size` | 2 | no flag | keep a cluster only if it spans at least this many images |
| forest | `accurate` preset: 8 trees, leaf size 16, 512 leaf checks | `--cluster-preset` | index build and per-query search budget |
| `max_leaf_checks` on the `.kdf` path | 128 | `background_floor_clusters_kdf` argument | per-query search budget when the forest is read from a file |

`d` was re-tuned from 28 to 10 by the sweep in [Choosing `d`](#choosing-d);
changing a default needs the membership-rule bench and the end-to-end
reconstructions below run again.

## Empirical Observations

### The nearest-distance distribution is bimodal

For each descriptor, let `d1` be the distance to its nearest *other* descriptor.
Across the corpus `d1` is **bimodal**: a near mode of descriptors that have a
likely match (a feature seen in more than one image, whose other observations
sit close in descriptor space) and a far mode of *isolated* descriptors seen in
only one place, with no near neighbour.

![Histogram of d1 for seattle_backyard: a near "has a near neighbour" mode and a
far "isolated" mode, with the antimode valley between them](images/d1-histogram-seattle.png)

The valley between the modes can serve as a global threshold for clusters. It
varies by dataset, so it must be derived from the data, and a per-descriptor
radius works better (below).

### The floor separates co-observations from background

A descriptor's true co-observations sit close in descriptor space, while
unrelated features are concentrated in a far "background" shell; the floor uses
the gap between them. Plotting, per in-track descriptor, the distances to its
co-observations against the distances to its background neighbours, the floor
`α·B` falls in the valley between the two on seoul_bull, seattle_backyard, and
kerry_park, capturing most co-observations while excluding the shell. The
exception is dino_dog_toy, whose repetitive structure leaves the distributions
overlapping: there no radius is clean, and the interleaved background is left to
geometric verification.

![Co-observation (green) vs background (red) descriptor-distance distributions per
reconstruction; the dashed line is the median floor α·B](images/floor-coobs-vs-background.png)

The background scale `B` (the d-th-nearest distance) marks that shell, so `α` sets
how far below it the cut sits. Sweeping `α` shows why `α = 0.8`: co-observation
recall climbs steadily, but the background admitted stays near zero until
`α ≈ 1.0`, where the radius reaches the shell and the count of admitted
background neighbours rises sharply. `α = 0.8` sits just below that point,
recovering about 0.70–0.85 of each track's co-observations while admitting
little background, and leaves the rest to geometric verification.

![Co-observation recall (green, left axis) and background neighbours admitted
(red, right axis) as the floor scale α is swept, with α = 0.8 marked](images/floor-alpha-sweep.png)

### Choosing a membership rule

The membership rule, which neighbours to keep, decides what the matcher
produces, so a wide range of rules was compared. For each descriptor in a
reference reconstruction, its 48 nearest neighbours are saved, each labelled as
a real co-observation (a neighbour from the same 3-D point, necessarily in
another image) or not. A candidate rule ("given a descriptor's neighbours, which
ones are co-observations?") can then be scored against those labels at once,
with no index to build or reconstruction to run.

The rules tried fell into a few families:

- **Cut at a gap.** Sort a descriptor's neighbour distances and cut where they
  jump up, by various definitions of "jump."
- **A radius from a per-point scale.** Keep neighbours within a multiple of the
  nearest distance, of the second-nearest, or of the background level — the
  distance at which the neighbours flatten out into unrelated features.
- **A fixed count.** Keep the few nearest cross-image neighbours.
- **Mutual agreement.** Keep a neighbour only if the descriptor is, in turn, near
  the top of that neighbour's own list.
- **Combinations.** Pair a generous radius with one of the stricter tests above.

The background-level radius, the floor, scored best on every dataset.
Mutual-agreement tests only hurt, dropping real co-observations. And no single
shared cut-off, even the best one found for a dataset, did as well as letting
each descriptor set its own radius from its background. This tuning settled on
keeping neighbours within 0.8× the 28th-nearest distance; the default rank was
later lowered to 10 (see [Choosing `d`](#choosing-d)).

### End-to-end reconstruction vs the baseline

Fed into **incremental** SfM, the cluster matches reconstruct every image and
place the cameras where the baseline does, with a denser point cloud. They go
through COLMAP's geometric verification and the incremental mapper (both seeded),
and the result is compared to the workspace's baseline with `sfm compare`. These
runs, and the global-mapper runs in the next section, used `d = 28`, the rank the
membership-rule tuning settled on; at the default `d = 10` the point counts
differ as [Choosing `d`](#choosing-d) reports. The Rust matcher at `d = 28`
gave the same registrations and point counts within 2% of this table.

| Dataset          | reg   | points (base → cluster) | reproj (base → cluster) | `sfm compare` |
| ---------------- | ----- | ----------------------- | ----------------------- | ------------- |
| seoul_bull       | 17/17 | 1,080 → 1,551           | 0.46 → 0.58 px          | VERY SIMILAR  |
| seattle_backyard | 26/26 |   521 → 4,968           | 0.61 → 0.37 px          | VERY SIMILAR  |
| kerry_park       | 48/48 | 1,128 → 3,193           | 0.31 → 0.78 px          | VERY SIMILAR  |
| dino_dog_toy     | 85/85 | 5,312 → 29,571          | 1.20 → 1.13 px          | VERY SIMILAR  |

`sfm compare` rates all four VERY SIMILAR, which here means the shared cameras'
centres agree (mean position error < 0.1) after a similarity alignment; it does
not look at the point cloud. The cloud is 1.4–9.5× denser than the baseline, at
reprojection error comparable to the baseline's (sub-pixel to about 1 px).
COLMAP's geometric verification filters the cluster correspondences and the
incremental mapper builds the reconstruction from what survives.

### Incremental reconstructs more reliably than global

Those same matches do not reconstruct as dependably under the *global* mapper
(`d = 28`, as above). Run through it instead, every image registers and this
run's verdicts pass, but the point counts are erratic: kerry_park keeps less than
a third of the points the incremental mapper recovers from the identical matches.

| Dataset          | reg   | points | `sfm compare`           |
| ---------------- | ----- | ------ | ----------------------- |
| seoul_bull       | 17/17 | 1,430  | VERY SIMILAR            |
| seattle_backyard | 26/26 | 4,903  | VERY SIMILAR            |
| kerry_park       | 48/48 |   926  | VERY SIMILAR            |
| dino_dog_toy     | 85/85 | 23,446 | VERY SIMILAR            |

Across repeated end-to-end runs with slightly different match sets, the
incremental mapper has passed all four every time, while the global mapper's
verdicts have ranged from two to four of four. Which datasets pass shifts with
the match set and seed, for reasons not understood. Incremental works better on
these test datasets.

## Algorithm

### Overview

1. **Index and k-NN query.** Concatenate every image's descriptors into one
   corpus, build a nearest-neighbour index, and query it once for the
   **`d + 1` nearest** (self and the `d` nearest others, `d = 10`) of *every*
   descriptor. The resulting `(N, d + 1)` table of neighbour ids and aligned
   distances is the only input to the steps below, and the index is not touched
   again.
2. **Per-point threshold.** For each descriptor, read a cluster membership
   radius off its own neighbour profile (the background floor, §2) and keep the
   cross-image neighbours within it.
3. **Materialize clusters.** Walk descriptors densest-first; each unclaimed
   descriptor seeds a cluster from its within-radius cross-image neighbours, one
   feature per image, and claims them (§3). The clusters are the matcher's
   output.
4. **Convert to matches.** Expand each cluster into its cross-image feature
   pairs, bucketed by image pair (§4). This derived view feeds the pairwise
   pipeline.

Geometric verification is not part of the matcher. `sfm match --derive-pairs`
runs it on the derived pairs (see [Pipeline](#pipeline)).

### 1. Index and the shared k-NN query

The corpus is all descriptors `(ΣKᵢ, 128)` (uint8 SIFT). The index is the in-tree
randomized kd-tree forest
([randomized-kdtree-forest.md](randomized-kdtree-forest.md)). Each row carries
its `(image_index, feature_index)` through `image_starts`, so a hit maps back to
a feature.

One query drives everything. For every descriptor the matcher fetches its
`d + 1 = 11` nearest, yielding an `(N, 11)` array of neighbour ids and an aligned
distance array, sorted ascending: column 0 is the descriptor itself at distance
0, columns 1…10 are its 10 nearest others. The query width is exactly what the
membership rule (§2) needs: the last column is the background rank `d`, and the
candidate members are the columns before it; nothing else queries the index.
With `α < 1`, members lie nearer than rank `d`, so this also caps a descriptor's
match degree below `d`.

### 2. Per-point threshold: the background floor

Each descriptor sets its own cluster membership radius from its neighbour
distances. For descriptor `i`, with neighbour distances `dist[i, 0…]` sorted
ascending (Euclidean L2), the **background floor** is its `d`-th-nearest distance,

```
B_i = dist[i, d]          (d = 10)
```

which is far enough out to land among unrelated background, past the
descriptor's few genuine co-observations. Keep neighbour `j` of `i` as a member
iff

```
dist(i, j) ≤ α · B_i    and    image(j) ≠ image(i)    and    j ≠ i      (α = 0.8)
```

A descriptor with tight co-observations has small early distances and a large
`B_i`, so it keeps them; an isolated descriptor's near distances already sit at
the background scale, so `α · B_i` admits nothing. Removing isolated
descriptors therefore needs no separate step. Since `α < 1`, every member is
nearer than `B_i`, so cluster membership only ever reaches ranks below `d`.

#### Why a generous radius

`α < 1` puts the cut *below* the background floor `B_i`, deliberately on the
generous side of the co-observations. Collecting a few too many neighbours is
cheap, because geometric verification rejects the misfits, while collecting too
few loses real observations. The reconstruction stays close to the baseline
across a wide band of radii above the data boundary; the only failures come from
radii that are *too tight* and drop whole images.

### 3. Materialize clusters

Clusters are built by density-ordered seeding over the k-NN table. Order
descriptors by how many within-radius cross-image neighbours they have, densest
first, and walk that order with a `claimed` flag per descriptor: each unclaimed
descriptor `s` seeds a cluster from `s` plus its within-radius (`α · B_s`),
cross-image, still-unclaimed neighbours, resolved to **one feature per image**
(nearest `s`). If the cluster spans at least `min_size` images, record it and
mark its members claimed; otherwise mark only `s` claimed and drop it. The
result is a hard partition: each feature belongs to at most one cluster, each
cluster holds at most one feature per image, and a cluster is a candidate track.
Because membership is proximity to the seed rather than transitive linkage, a
chain A–B–C cannot merge two distinct points; the density ordering forms the
best-defined clusters first.

### 4. Convert clusters to per-image-pair matches

Each cluster of `m` members expands into its `C(m, 2)` cross-image feature pairs,
bucketed by image pair. Because clusters hold one feature per image and the
partition is hard, the resulting matches are already one-to-one per image pair,
with no reconciliation pass. Only image pairs that share a cluster appear, so
the clustering also selects the pairs. Geometric verification of those pairs
rejects the ones that are not tracks.

### Alternatives considered

- **A global threshold.** A single, data-derived radius `T` for the whole corpus
  in place of the per-point floor, read from the same k-NN table two ways: the
  **per-point cliff** (for each descriptor, the largest jump between consecutive
  sorted neighbour distances separates its likely co-observations from background;
  `T` is a percentile, by default the median, of the just-past-the-cliff
  distance over all descriptors), or the **`d1` bimodal antimode** (the valley
  between the modes of `d1`, see above, located with Otsu's method or a
  two-component mixture on `log d1`, optionally scaled by 1.0–1.25). One radius
  never fits every cluster, and on labelled neighbourhoods the per-point floor
  matches or beats even the best-possible global `T`. Recentring each cluster by
  re-querying at its mean (mean shift) improved the global-`T` clusters slightly
  and did not change reconstruction outcomes. A global `T` also does not exclude
  isolated descriptors by itself; they would have to be dropped up front, for
  example when `d1 > T` or `d1/d5 > 0.85` (about 40–75% of descriptors, all
  background the solve discarded).
- **Per-descriptor edges, no materialized clusters.** Keep each descriptor's
  within-radius cross-image neighbours directly as match edges and reconcile them
  to one match per feature per image pair (two passes: keep the smallest-distance
  edge per low-side feature, then per high-side feature). This produces
  essentially the same matches and reconstructions, slightly more cheaply, but
  leaves no clusters for later consumers, which is why the matcher materializes
  clusters.
- **Transitive merge.** The connected components of a within-radius graph chain
  distinct points into very large clusters through repeated structure. Seeded
  clusters avoid this by construction.

### Choosing `d`

The default background rank is `d = 10`. The Python prototype the empirical
sections above are drawn from used `d = 28`; sweeping the Rust matcher at
`d ∈ {6, 7, 8, 9, 10, 14, 20, 28}` across all four datasets (match plus seeded
incremental solve per point) showed the wide floor cost solve time without
improving quality. Findings:

- Registration is full (17/17, 26/26, 48/48, 85/85) at every `d ≥ 8`;
  kerry_park drops to 46/48 at `d ∈ {6, 7}`, so the registration loss starts
  just below 8.
- Smaller `d` is faster end to end, mostly in the *solve*, which scales with
  the candidate matches a wider floor admits (kerry 55 s at `d = 8` vs 108 s at
  28; dino 99 s vs 147 s total).
- Mean reprojection error *improves* monotonically as `d` shrinks on every
  dataset (e.g. seoul 0.47 px at 8 vs 0.58 px at 28): the extra matches a wide
  floor admits are disproportionately the weak ones.
- Total points dip on the small scenes (seoul −20%, kerry −15% at `d = 10` vs
  28) but the lost points are mostly 2-view: the fraction of points with ≥ 3
  observations is far higher at small `d` (92–97% vs 77–84%), and dino's point
  count rises (32,181 vs 29,657).

`d = 10` is the default because it was measured directly on all four datasets,
sits two ranks above the rank where kerry_park loses images, runs about
1.5–2.4× faster end to end than 28, and has lower reprojection error everywhere.
`d = 28` remains a reasonable choice for unusually high-covisibility collections
(features co-observed in tens of images), where a small rank could read the
floor inside the track itself; pass `--cluster-d` to raise it.

## Pipeline

**`sfm match --cluster`** ([match-command.md](../../cli/image-feature/match-command.md))
runs the matcher and writes one file. `_run_matching` sorts the images by
workspace-relative name, so the corpus order is the order every `.matches`
reader uses; `_materialize_clusters` calls `cluster_match` and prints the
cluster, candidate-match and image-pair counts (the pair expansion is computed
for that report and not written); and `_write_clusters_matches` writes a
`.matches` file holding the clusters backbone and no pairs or two-view
geometries. The backbone carries each member's position and affine shape copied
from its `.sift` row, and the metadata records `matching_method: "cluster"` with
`matching_options` `{"mode": "background-floor", "d", "alpha", "min_size",
"preset"}`, plus `max_feature_count` when `--max-features` is given. The
command opens no COLMAP database and runs no verification, so the same corpus
and options give the same clusters backbone bit for bit on every run. It honours `--range`,
`--max-features` and `-o`; the default path is under the workspace's `matches/`
directory with a `-clusters` suffix. `min_size` has no flag and is 2.
`--camera-model` with `--cluster` is a `UsageError`, because the clustering uses
no intrinsics.

Consumers of that file:

- `sfm match --derive-pairs` expands the clusters into pairs, verifies them with
  `pycolmap.verify_matches` in a temporary COLMAP database, and writes the
  pairwise + two-view-geometry `.matches` file under `tvg-matches/`.
- `sfm solve` takes that derived file. It refuses a clusters-bearing file, since
  COLMAP's mappers read their correspondence graph from two-view geometries.
- `sfm to-colmap-db` reads a clusters file directly, deriving the pairs at read
  time (`pairs_from_matches` in
  [`_pairs.py`](../../../src/sfmtool/feature_match/_pairs.py)).
- `sfm cluster-patches` and `sfm estimate-intrinsics` read the clusters.

```bash
sfm match --cluster images -o matches/cluster-clusters.matches
sfm match --derive-pairs matches/cluster-clusters.matches -o tvg-matches/cluster.matches
sfm solve -i tvg-matches/cluster.matches
```

**In-solve matching.** `_setup_for_sfm(matching_mode="cluster")` in
[`db_setup.py`](../../../src/sfmtool/colmap/db_setup.py) calls
`_run_cluster_matching`, which clusters with the default parameters, writes the
derived pairs into the solve's own COLMAP database, drops same-frame pairs of a
multi-sensor rig (back-to-back sensors with no shared view), and verifies the
rest with `pycolmap.verify_matches`. The Python functions `run_global_sfm` and
`run_incremental_sfm` accept `matching_mode="cluster"`, and the kerry_park
`.camrig` test fixture uses it; no CLI command selects this mode.

**SfM Explorer.** The viewer's index-files build runs
`background_floor_clusters_lazy` over a reconstruction's SIFT index `.kdf`, with
`d = 10`, `alpha = 0.8`, `min_size = 2` and 128 leaf checks
([index-files.md](../../gui/index-files.md)).

## Implementation Notes

- **Parallelism.** The radius pass is parallel over rows (rayon). Seeding is
  one sequential pass, because the claim order defines the result; it does at
  most `d + 1` work per row. The pair expansion runs in parallel per cluster,
  followed by a parallel sort on `(img_lo, img_hi, feat_lo, feat_hi)`.
- **Memory.** Candidates are held in flat `N × (d + 1)` arrays of indexes and
  distances, with no per-edge heap structures.
- **Query order.** The in-memory self-join runs its queries in the forest's
  `locality_order` (tree 0's leaf layout) through
  `search_batch_with_distances_ordered`, so consecutive queries read
  overlapping, cache-resident corpus rows. Results are written back to query
  order, which is safe because each query's result does not depend on the
  order. The query is still bound by memory latency on those reads.
- **Forest build** reuses per-tree scratch buffers (partition tags and reorder
  buffer, median values, variance sums) rather than allocating per node, with
  the same arithmetic and random-number consumption, so the trees are
  bit-identical.
- **Timing.** Setting `SFMTOOL_CLUSTER_TIMING` prints `CLUSTER_TIMING` lines to
  stderr with per-stage wall-clock times: forest build and query, the radius
  pass and seeding, and the pair expansion.

On DinoLedge (1196 images × 8192 features = 9.7M descriptors, 1.7M clusters,
317K candidate pairs, on an i9-14900HX) the k-NN query took about 31 s.
Geometric verification of the derived pairs runs in `--derive-pairs` through
pycolmap, whose two-view-geometry inlier sets vary by about 0.001% between
identical runs because of its multithreaded RANSAC.

Tests: [`cluster_match/tests.rs`](../../../crates/sfmtool-core/src/features/cluster_match/tests.rs)
(planted clusters, partition invariants, pair expansion, validation errors,
determinism, external neighbour tables),
[`test_cluster_match_rust_bindings.py`](../../../tests/rust_bindings/matching/test_cluster_match_rust_bindings.py),
[`test_cluster_matching.py`](../../../tests/matching/test_cluster_matching.py).

## Cost

- **Index build.** The randomized kd-tree forest is cheap to build
  (`O(n log n)` median splits), far cheaper than a navigable-graph index, whose
  build cost is only repaid over many queries or a persistent index.
- **Query.** `ΣKᵢ` queries × `d + 1` neighbours. Exact brute force is
  `O(n²·D)`: seconds to minutes on the test datasets, prohibitive at realistic
  scale. The forest makes each query sub-linear at a fixed search budget (leaf
  visits, not `log n`; the index is approximate, see its spec), keeping the
  downstream match signal close to exact at a tunable budget. Exact search is
  the reference the forest's recall is measured against.
- **Verification.** Per-image-pair geometric verification, only over pairs that
  share a cluster, which is far fewer than `N²` on scenes with limited
  covisibility.

## Limitations

- **Descriptor distance can't separate every track from background.** For some
  in-track descriptors a background neighbour is nearer than a true
  co-observation, so no radius clusters them cleanly; this remainder concentrates
  in high-multiplicity, repetitive structure and is left to geometric
  verification. The exact fraction is relative to the reference reconstruction:
  much of the apparent non-separability is the reference labelling real
  co-observations as background, so it is smaller than a single solve suggests.
- **Some members are unreachable by distance.** Wide-baseline observations sit
  beyond any reasonable radius; they have no near co-member (dino recall@5 ≈ 0.46
  even with exact search), so they never join, and the recovered tracks are
  fragmented (a track spans about 1.6–2.7 clusters, about 85% of members
  recovered). Conventional nearest-neighbour matching with a ratio test misses
  the scattered members too.
- **Repeated structure causes false merges.** Membership by proximity to a
  fixed seed (no transitive chaining) and the geometric verifier limit them;
  descriptor distance alone does not.
- **The hard partition is greedy and order-dependent.** A feature claimed by one
  cluster cannot join a better one later; seeding densest-first reduces this by
  forming the best-defined clusters before the rest.
- **Global SfM is less reliable than incremental on these datasets** (its `sfm
  compare` verdicts range from two to four of four across runs, and which pass
  varies with the match set and seed, as above), for reasons not understood.
- **Exact nearest-neighbour search does not scale**; the forest trades a few
  points of recall, which geometric verification and track redundancy absorb.
- **There is no `sfm solve --cluster` shortcut.** The matcher is reached through
  `sfm match --cluster` followed by `sfm match --derive-pairs`.

## Relationship to Existing Pipeline

This matcher replaces the per-pair Lowe ratio test with a per-point,
data-derived distance radius. Its `.matches` output stores the clusters, and the
pairwise view every pairwise consumer wants is derived from them.

### Vocabulary trees

COLMAP's `vocab_tree_matcher` also clusters SIFT descriptors, so it is the
closest existing method. A vocabulary tree (Nistér & Stewénius 2006) clusters a
large *training* corpus of descriptors **offline** into a hierarchical k-means
tree whose centroids are coarse "visual words"; each image becomes a bag of those
words, and bag-of-words similarity **retrieves candidate image pairs**, which are
then matched and verified normally. That offline k-means is itself centroid
iteration accelerated by a randomized kd-forest (Philbin et al. 2007), the same
kind of index this matcher uses.

This method clusters descriptors at a different granularity and stage. Rather
than a coarse, reusable vocabulary built offline on a separate corpus, it
clusters the reconstruction's **own** descriptors online into tight,
**track-scale** groups, and those clusters *are* the candidate correspondences,
not a retrieval index. A visual word is a large cell of descriptor space shared
by many unrelated features; a cluster here aims to be the observations of a
single 3-D point. Selecting only the image pairs that share a cluster does the
same job the vocabulary tree does for COLMAP, avoiding `O(N²)` pair enumeration,
but inside the matching step instead of a separate retrieval stage.
