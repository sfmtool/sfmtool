# Cluster Covisibility

## Purpose

Cluster covisibility measures how many match clusters each pair of images
shares, so a caller can pick mutually-overlapping image groups and rank
candidate views before any reconstruction exists.

Given the clusters of a `.matches` file (the `clusters/` section, optionally
enriched with `cluster_patches/`), compute how many clusters each pair of
images shares, and answer the grouping queries that consumers build on top of
those counts:

- **Seed groups** — small sets of mutually covisible images, strong enough
  for a windowed weak-perspective factorization.
- **Ranking** — order a candidate set of images by covisibility with a given
  image.
- **Inspection** — raw counts for analysis and scheduling.

The counts depend only on the match clusters, so the groupings reflect
actual view overlap and apply to any image collection — video frames,
looped captures, and unordered photo sets alike.

**Naming.** This is *cluster* covisibility — a pre-reconstruction quantity
computed from match clusters. It is distinct from the post-reconstruction
covisibility of shared 3D tracks in
`sfmtool_core::analysis::image_pair_graph` (`sfm analyze --coviz`), which
requires poses and points. The two must not share a name or type.

**See also.** The selection queries layered on this type — pair displacement,
banded thinning, reach — are specified in
[covisibility-selection.md](covisibility-selection.md), and the sparse
displacement-neighborhood substrate (`DisplacementNeighborhood`: exhaustive
per-pair displacement means with `nearest` / `farthest` / pair-stats queries and
array serialization) in
[pose-verification.md](../geometry/pose-verification.md).

## Definition

For images `i ≠ j`:

```
W[i, j] = |{ c : cluster c has an accepted member in image i
                 and an accepted member in image j }|
```

- `W` is symmetric with a zero diagonal.
- **Accepted** is caller-defined via a per-member mask (see Acceptance
  below). Each cluster contributes at most 1 to any pair regardless of how
  the mask is built — the `.matches` format guarantees at most one
  status-`reference`/`kept` member per (cluster, image), and for unrefined
  clusters the per-cluster image list is deduplicated before counting.
- A cluster's **span** is its number of distinct accepted images; clusters
  with span < 2 contribute nothing.

## Acceptance and filtering

The constructor takes an optional per-member boolean mask. This mask is the
entire integration point for prioritization and filtering:

- `None` — every member counts (a raw `clusters/` section, no statuses).
- Patch-enriched default — status ∈ {`reference`, `kept`}.
- **Custom masks** — callers filter members by any properties they choose,
  and rebuild the matrix when the selection changes (construction is cheap,
  see Complexity). The properties available *from the file alone* (v4):

  | channel | source | typical filter |
  |---|---|---|
  | status classes | `member_status` | base acceptance; rejected classes stay re-gateable |
  | photometric quality | `member_zncc` | drop weakly-correlated members |
  | geometric consistency | `member_consistency_residual` | drop members that misfit their cluster's joint weak-perspective geometry |
  | feature size (ordinal) | `member_features` row index — `.sift` rows are scale-sorted descending | restrict to larger features |
  | image radius | `clusters/member_positions` + `images/image_dims` | restrict by distance from the image center |
  | feature extent | `clusters/member_affine_shapes` column norms | restrict by feature size |

## Complexity and bounds

Construction visits each cluster's accepted-image list and increments its
`span·(span−1)/2` pairs: total `Σ span²/2` increments. The cluster matcher
caps cluster size (`matcher_options.d`, currently 10), so

```
Σ span² ≤ d · (accepted members)      — linear in observations
```

Empirically, across three campaign datasets, mean span ≈ 3.3 and
`Σ span²` ≈ 4.6 × members. Ten million observations is therefore ≈ 50 M
increments — a few hundred milliseconds serial; parallelism is not required
at current scales.

Storage is dense row-major `u32`, `4·N²` bytes, and is the actual scaling
wall: ~25 MB at N = 2,500 images, ~400 MB at N = 10,000. The public API
does not expose dense-ness (see below), so a sparse/CSR backend can be added
behind the same type when a >4–5 k-image consumer appears; construction
errors with a clear message above the bound — `MAX_DENSE_IMAGES = 4096`
(64 MB dense) — until then. Long
videos — the capture style that produces such N — have banded covisibility,
so the sparse variant is compact where dense is hopeless.

The `d` coupling is an assumption worth keeping visible: raising the
matcher's cluster-size cap (e.g. to ~100 for long-track experiments) scales
construction cost by `d` and makes each mega-cluster vote on up to
`d(d−1)/2` pairs.

A positioned build runs three passes over the same arrays and returns one
object, so which pass the time went to is not visible from outside it.
`SFMTOOL_PROFILE=1` turns on per-phase wall-time counters
(`covisibility/prof.rs`, the convention of `focal_vote/prof.rs`) and prints a
summary to stderr when the build finishes: `cluster_pass` (the span dedupe, the
shared-count votes and the sampled displacement draw), `mean_fold` (the `N²`
divide-and-mirror over the sampled tables), and `neighborhood_accum` /
`neighborhood_sort` (the exhaustive cross-image member-pair accumulation and
the pair sort behind `DisplacementNeighborhood`). With the variable unset each
timer is one branch on a cached flag.

## Rust API

The counts and the grouping queries live in
[covisibility.rs](../../../crates/sfmtool-core/src/features/cluster_match/covisibility.rs)
— module `sfmtool_core::features::cluster_match::covisibility`, beside the
matcher that produces the clusters — bound as
`sfmtool._sfmtool.matching.ClusterCovisibility`. Core stays I/O-free: raw CSR
slices in, following `refine_cluster_patches`, or an already-parsed
`MatchesData`.

```rust
pub struct ClusterCovisibility { /* num_images, counts (private) */ }

impl ClusterCovisibility {
    /// `member_accepted`: parallel to `member_images`; `None` = all members.
    /// Panics/errors if `num_images` exceeds the dense bound or arrays are
    /// not parallel / CSR-consistent.
    pub fn from_clusters(
        cluster_starts: &[u32],
        member_images: &[u32],
        member_accepted: Option<&[bool]>,
        num_images: usize,
    ) -> Result<Self, CovisibilityError>;

    /// A parsed `.matches` file, with the acceptance policy its sections
    /// imply (see Acceptance). `CovisibilityError::NoClusters` when the file
    /// stores the pairwise backbone; `seed` drives the displacement
    /// sampling pass.
    pub fn from_matches(
        matches: &MatchesData,
        seed: u64,
    ) -> Result<Self, CovisibilityError>;

    pub fn num_images(&self) -> usize;
    pub fn count(&self, i: u32, j: u32) -> u32;
    pub fn row(&self, i: u32) -> &[u32];

    /// Lazy iterator of greedy mutually-covisible groups; see Seed-group
    /// algorithm.  Each `next()` produces one group; consumers take as
    /// many as they need and drop the rest unpaid.
    pub fn seed_groups(&self, params: &SeedGroupParams) -> SeedGroups<'_>;

    /// `candidates` reordered by descending covisibility with `image`
    /// (ties: ascending index); zero-covisibility candidates are dropped.
    pub fn rank_by_covisibility(&self, image: u32, candidates: &[u32]) -> Vec<u32>;
}

/// Borrows the matrix; state is an excluded-image mask (no matrix copy).
/// Each `next()` costs one scan for the strongest remaining edge plus the
/// group-extension steps.
pub struct SeedGroups<'a> { /* covis, excluded: Vec<bool>, params */ }
impl Iterator for SeedGroups<'_> { type Item = Vec<u32>; }

pub struct SeedGroupParams {
    pub group_size: usize, // default 5
    pub min_shared: u32,   // default 8 — see caveat
}
```

`min_shared = 8` is carried over from the experiments unvalidated; a
data-derived constructor (`SeedGroupParams::derive`, e.g. a fraction of the
median nonzero edge weight) is the intended replacement once evaluated.

### Seed-group algorithm

Deterministic greedy; images in already-yielded groups are excluded from
all later consideration. Each `next()`:

1. Take the strongest edge `(i, j)` among non-excluded images (ties:
   lexicographically smallest `(i, j)`); the group starts as `{i, j}`. If
   the strongest such edge is below `min_shared`, the iterator ends.
2. Repeatedly add the non-excluded image `k` maximizing
   `min over g in group of W[k, g]` (ties: smallest `k`), while that
   minimum is ≥ `min_shared` and the group is below `group_size`. The
   *minimum*-vs-group criterion keeps groups mutually covisible rather
   than hub-and-spokes.
3. Yield the group sorted ascending and mark its images excluded.

Guarantees: the sequence depends only on the input arrays (no RNG, no
iteration-order dependence); groups are disjoint; the first `k` groups are
identical however many are ultimately consumed; every within-group pair of
a yielded group has `W ≥ min_shared`.

## Bindings

`sfmtool._sfmtool.matching.ClusterCovisibility` (PyO3 class in
`crates/sfmtool-py/src/matching/covisibility.rs`); no Python wrapper layer.

```python
ClusterCovisibility.from_matches(matches_file, seed=0)   # a MatchesFile
ClusterCovisibility.from_arrays(cluster_starts, member_images, num_images,
                                member_accepted=None, positions_xy=None,
                                seed=0)

cov.num_images          # getter
cov.counts              # numpy (N, N) uint32 copy; errors above dense bound
cov.seed_groups(group_size=5, min_shared=8)   # iterator of list[int]
cov.rank_by_covisibility(image, candidates)   # numpy uint32
```

`seed_groups` returns an iterator object (holding a reference to its
`ClusterCovisibility` plus the excluded-image state), so
`for group in cov.seed_groups(...):` consumes lazily and
`list(cov.seed_groups(...))` recovers the eager behavior.

`from_matches` takes the `MatchesFile` handle (a selection included) and
forwards its parsed data to the core entry of the same name, so a file is
parsed once and the acceptance policy has one home: the mask is
status ∈ {reference, kept} when a `cluster_patches/` section is present, and
all-members otherwise. The backbone's `member_positions` pass through as the
`f32` pairs the file stores, copying nothing and widening nothing (the
displacement arithmetic is `f64` at the point of computation), so the
displacement queries and the isolation-ordered `thin` sweep answer on a
matrix built this way; a cluster file below format v6 stores no positions and
leaves them unavailable. `num_images` is the image table's length. A file
storing the pairwise backbone raises `ValueError`. Custom masks use
`read_matches` + numpy + `from_arrays`.

## Validation

- Unit tests (core): hand-built CSR fixtures — counts, mask handling,
  span-1 clusters ignored, seed-group determinism and tie-breaks, prefix
  stability (the first `k` groups match whether `k` or all are consumed),
  the hub-vs-mutual distinction (a star topology must not form a group),
  dense bound error.
- Bindings test (`tests/rust_bindings/`): array and file constructors on a
  small generated file; numpy round-trip; parity of `seed_groups` with a
  numpy reference implementation.
- First consumer: `exp_pinhole_bootstrap.py` swaps its `covisibility()` and
  `pick_seed_groups()` for the binding — campaign parity on seoul is the
  acceptance test.
- Evaluation experiments (separate from this spec's implementation): the
  filter-aware cluster budget (replace the span-only `MAX_CLUSTERS`
  selection with span + consistency + ZNCC + feature-size priority)
  measured against the campaign baselines — does dino_dog_toy's >10° camera
  count drop; does kerry's usable field grow when admission is restricted
  by image radius and then relaxed.

## Open questions

- `min_shared` derivation from the edge-weight distribution (and whether
  seed quality is sensitive to it at all on well-connected sets).
- The trigger and representation for the sparse backend (banded CSR vs
  hash-based), when a >4–5 k-image consumer exists.
- Weighted covisibility: whether any consumer actually needs graded
  weights that binary masks cannot express.
- A CLI inspection surface (`sfm analyze`-family, matches-file input) —
  cheap to add, not needed by any current consumer.
