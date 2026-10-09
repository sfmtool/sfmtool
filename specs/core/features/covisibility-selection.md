# Covisibility Selection: Displacement, Thinning, Reach

## Purpose

These are three queries for choosing a subset of images from match data alone,
without using camera poses. They run on `ClusterCovisibility`, which counts
how many match clusters each pair of images shares (see
[cluster-covisibility.md](cluster-covisibility.md)). Pair displacement gives
the mean pixel distance between matched features of two covisible images.
Thinning keeps a spread-out subset by dropping images that share too many
clusters with one already kept. Reach gives the fraction of all images that a
subset shares enough clusters with. Reconstruction growth
([reconstruction-growth.md](../geometry/reconstruction-growth.md)) calls
`thin_to` to pick the cameras of its periodic anchor bundle adjustment and of
its finishing bundle adjustment. Every result is deterministic for a given
`seed`.

## Interface

The queries are methods of `ClusterCovisibility` in
[cluster_match/covisibility.rs](../../../crates/sfmtool-core/src/features/cluster_match/covisibility.rs)
(construction and the displacement tables) and
[covisibility/selection.rs](../../../crates/sfmtool-core/src/features/cluster_match/covisibility/selection.rs)
(thinning and reach). Python reaches them as
`sfmtool.matching.ClusterCovisibility`, bound in
[matching/covisibility.rs](../../../crates/sfmtool-py/src/matching/covisibility.rs).

```rust
impl ClusterCovisibility {
    // Construction with positions; `from_clusters` is this with `None` and seed 0.
    pub fn from_clusters_with_positions(
        cluster_starts: &[u32],
        member_images: &[u32],
        member_accepted: Option<&[bool]>,
        num_images: usize,
        positions_xy: Option<&[[f32; 2]]>,
        seed: u64,
    ) -> Result<Self, CovisibilityError>;
    // Reads clusters, acceptance mask and positions from a `.matches` file.
    pub fn from_matches(matches: &MatchesData, seed: u64) -> Result<Self, CovisibilityError>;

    pub fn pair_displacement_magnitude(&self) -> Option<&[f64]>; // row-major n×n
    pub fn pair_displacement_counts(&self) -> Option<&[u32]>;    // row-major n×n
    pub fn thin(&self, tau: f64) -> Vec<u32>;
    pub fn thin_to(&self, target: usize) -> Vec<u32>;
    pub fn reach(&self, images: &[u32], min_shared: u32) -> f64;
}
```

Positions are optional because the shared-cluster counts do not need them, and
a `.matches` file below format version 6 carries none. They are `f32` pairs
parallel to `member_images`, the width the `.matches` backbone stores
keypoints at, so `from_matches` passes a file's rows through without copying.
Without positions the two displacement getters return `None` and thinning
sweeps images in index order (see Thinning). Positions do not change the
shared-cluster counts.

```rust
let covis = ClusterCovisibility::from_matches(&matches, 0)?;
let spread = covis.thin_to(120);         // about 120 spread-out images
let fraction = covis.reach(&spread, 8);  // how much of the capture they cover
```

The Python constructors are

```python
ClusterCovisibility.from_arrays(cluster_starts, member_images, num_images,
                                member_accepted=None, positions_xy=None,
                                seed=0)
ClusterCovisibility.from_matches(matches_file, seed=0)
```

`from_arrays` also accepts a `float64` `positions_xy` and casts it to `f32`.
`from_matches` takes an open `MatchesFile` and uses the status ∈ {reference,
kept} mask when the file has a `cluster_patches/` section.

## Pair displacement

Construction with positions makes one sampled pass over the clusters. Every
cluster with two or more accepted members draws one uniformly random pair of
distinct members, using a generator seeded by `seed`. When the two members lie
in the same image, the cluster contributes nothing. Otherwise the Euclidean
distance between their two positions is added to that image pair's sum. Each
distance is computed in `f64`, with each coordinate widened where it is read.
The `member_accepted` mask applies here too: only accepted members are drawn.

The tables hold magnitudes only. A caller that needs a displacement vector, or
a mean over every member pair rather than a sample, reads the
`DisplacementNeighborhood` described in
[pose-verification.md](../geometry/pose-verification.md).

- `pair_displacement_magnitude()` → `f64 [n_img, n_img]`, symmetric: the mean
  sampled displacement per covisible pair, `0` where no sample landed.
- `pair_displacement_counts()` → `u32 [n_img, n_img]`, symmetric: the number of
  samples behind each mean, for callers that require a minimum support.

## Thinning

`thin(tau)` returns a subset, sorted ascending, built by one greedy sweep over
the images. The first image swept is always kept. Each later image is kept
only when its largest shared-cluster count with any already-kept image lies in
the band `[tau/8, tau)`. An image at or above `tau` shares so many clusters with
a kept image that it adds little, and an image below `tau/8` is too weakly
connected to the kept set. The factor 8 is a fixed constant with no recorded
derivation.

The sweep order is decreasing isolation. An image's isolation is its smallest
sampled mean displacement to any partner with at least one sample, and an image
with no such partner is infinitely isolated. Ties are broken by ascending
image index. Without positions there is no isolation, and the sweep visits
images in ascending index order. So with positions the kept set does not depend
on how the images are numbered, except where isolations tie exactly; without
positions it does.

`thin_to(target)` chooses `tau` for the caller. It bisects `tau` over
`[1, m]` for 25 iterations, where `m` is the median over images of each image's
largest shared-cluster count with any other image. At each step it raises the
lower bound when the subset is smaller than `target` and lowers the upper bound
otherwise. It returns the subset whose size was closest to `target` among those
it computed, and the earlier one on a tie.

Two consequences follow. First, sizes that only a `tau` above `m` would produce
are never returned: on the 8-image chain of the tests, `thin(128)` keeps all 8
images, but `thin_to(8)` returns 4. Second, the bisection assumes that a larger
`tau` keeps more images. That is not guaranteed, because the lower band edge
`tau/8` rises with `tau` and the sweep is greedy. When it fails, `thin_to` still
returns the closest size it computed, which may not be the closest size any
`tau` would give. An empty matrix gives an empty subset.

## Reach

`reach(images, min_shared)` returns the fraction of all images that share at
least `min_shared` clusters with at least one image of the subset. Members of
the subset count as reached. An empty subset, or a matrix with no images,
returns `0.0`. A subset confined to one neighbourhood of viewpoints has low
reach however large the capture is, and a subset that spans the capture has
reach near 1. The Python default `min_shared=8` is the same value
[cluster-covisibility.md](cluster-covisibility.md) uses for seed groups, with
no separate justification.

## Bindings

The Python methods have the Rust names. `thin` and `thin_to` return `uint32`
numpy arrays and `reach` returns a float; the two displacement getters return
`(N, N)` numpy copies. Python raises `ValueError` for a displacement query on
a matrix built without `positions_xy` (where Rust returns `None`), for an
out-of-range index passed to `reach` (where Rust panics), and for a non-finite
`tau` passed to `thin`, which the binding rejects before calling Rust.

## Testing requirements

Rust tests are in
[covisibility/tests.rs](../../../crates/sfmtool-core/src/features/cluster_match/covisibility/tests.rs),
and the binding tests in
[test_cluster_covisibility_rust_bindings.py](../../../tests/rust_bindings/matching/test_cluster_covisibility_rust_bindings.py).

- Displacement: a synthetic scene with known geometry yields the expected pair
  means; results repeat for a fixed seed; the mask limits which members are
  sampled; same-image samples are skipped; construction without positions
  makes the displacement queries fail.
- Thinning: on a chain of 8 images whose shared-cluster counts halve with each
  step of separation, `thin` reproduces the expected band selection; with
  positions the sweep follows isolation, and renumbering the images renumbers
  the output the same way; `thin_to` reaches the sizes available on the chain.
- Reach: hand-built subsets on a known graph give exact fractions, and
  `min_shared` is respected at the boundary.
- The bindings return the Rust results.

## Non-goals

- Pair selection policy (which pairs to estimate geometry on). Callers compose
  it from these queries.
