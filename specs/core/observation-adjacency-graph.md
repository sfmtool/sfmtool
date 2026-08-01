# Observation adjacency graph: image-space point neighbourhoods

**Status:** Implemented (2026-08-01) — core in
`crates/sfmtool-core/src/analysis/observation_adjacency.rs`, exposed to
Python as `sfmtool._sfmtool.analysis.build_observation_adjacency`.

## Overview

Several per-point operations need to know **which reconstructed points are
next to each other on the imaged surface**: surfel-normal estimation fits a
plane through a point's image-space neighbours, duplicate-point collapse
looks for points whose observations coincide in every shared image, and
aliased-track repair looks for pairs that coincide in one image but sit far
apart in another. All three start from the same structure — a point-level
adjacency graph built from how close the points' *observations* are in the
images that see both — and differ only in how they filter and read its
edges.

This module builds that graph in one batch call. Adjacency is decided in
**image space, in units of a per-point radius** the caller supplies (for
example the feature's detection scale in pixels, or the projected extent of
the point's patch), so the notion of "next to" scales with the feature, not
with the scene.

## Adjacency criterion

Two points `p` and `q` are adjacent when all of the following hold, with
every distance measured between their keypoints in a shared image and
expressed as a ratio of the **pair radius** `r_pq = min(radius[p],
radius[q])` in that comparison:

1. **Annulus, by majority.** In at least `majority` (fraction, default 0.5)
   of the images that observe both points, the keypoint separation falls in
   `[a_lo · r_pq, b_max · r_pq]`.
2. **Support.** The pair shares at least `min_shared_images` (default 2)
   images.
3. **Range consistency.** The pair's ranges from the shared cameras agree:
   the median over shared images of `|range_p − range_q| / mean(range_p,
   range_q)` is at most `range_tol` (default 0.05). Range is the Euclidean
   distance from the observing camera's centre to the point — a
   convention-free depth proxy, so the test never depends on which way the
   camera frame points. This is what separates "next to each other on the
   surface" from "one behind the other along the viewing ray".

Setting `a_lo = 0` admits fully-overlapping observations (the
duplicate-collapse regime); setting `range_tol = ∞` disables the range vet
(the aliased-pair regime, which needs to *see* depth-inconsistent
neighbours rather than filter them). Points at infinity and points with a
non-positive radius take part in no edges.

## Algorithm

Candidate generation and vetting are separate passes so the ball query runs
once per image while every candidate is judged on **all** its shared
images:

1. **Candidates** — per image, a 2D neighbour query over that image's
   keypoints (radius `b_max · radius[point]` per query site); a pair whose
   separation lies inside the annulus in *at least one* image becomes a
   candidate. Deduplicate candidates across images.
2. **Vet** — for every candidate pair, walk every image observing both
   endpoints, accumulating: shared-image count, annulus-hit count,
   per-image separation ratios, and per-image relative range differences.
   Apply the three criteria above to the accumulated totals (medians for
   the separation and range statistics).

The candidate pass is per image and parallelizes over images. The vet is
per candidate: each pair merge-joins its two endpoints' image-sorted
observation lists, which costs the pair's own observations rather than a
sweep of every image, and parallelizes over candidates. Results are merged
in image, then candidate, order, so the output never depends on thread
scheduling.

## Output

Symmetric CSR over points, with per-directed-edge statistics parallel to
the neighbour array:

- `offsets` — `n_points + 1`; the neighbours of point `p` are
  `neighbours[offsets[p] .. offsets[p+1]]`.
- `neighbours` — point indices (u32).
- `separation_med`, `separation_min`, `separation_max` (f32) — the pair's
  median / min / max keypoint separation over the shared images where it
  landed in the annulus, in pair-radius units.
- `shared_images`, `annulus_hits` (u32) — the vet's counts.
- `range_mismatch` (f32) — the median relative range difference.

Each point's neighbour list is sorted by `(separation_med, neighbour
index)`, so the nearest surface neighbours come first and ordering is fully
deterministic.

## API

```rust
pub struct ObservationAdjacencyParams {
    pub a_lo: f64,              // annulus inner edge, pair-radius units (1.0)
    pub b_max: f64,             // annulus outer edge, pair-radius units
    pub min_shared_images: u32, // support floor (2)
    pub majority: f64,          // annulus-hit fraction (0.5)
    pub range_tol: f64,         // median relative range gate (0.05; INFINITY disables)
}

pub fn build_observation_adjacency(
    keypoints_xy: &[[f64; 2]],       // per observation
    track_point_indexes: &[u32],     // per observation
    track_image_indexes: &[u32],     // per observation
    radii_px: &[f32],                // per point; <= 0 excludes the point
    point_is_at_infinity: &[bool],   // per point
    positions: &[[f64; 3]],          // per point (range vet)
    camera_centers: &[[f64; 3]],     // per image (range vet)
    params: &ObservationAdjacencyParams,
) -> ObservationAdjacency
```

The Python binding takes the same arrays as numpy inputs (accepting what
`SfmrReconstruction` accessors return directly), the parameters as
keywords with the defaults above — `b_max` has no default, so it is a
required keyword — and returns the CSR arrays as a dict of numpy arrays
keyed as in the output list. Camera centers are
`-Rᵀ·t` per image; the binding accepts poses as `(quaternions_wxyz,
translations)` and derives the centers so callers don't repeat that
conversion.

## Testing

Sibling `tests.rs` under `analysis/observation_adjacency/` covers:
annulus inclusion at both edges; the majority vote (hit in 1 of 3 shared
images fails at `majority = 0.5`, 2 of 3 passes); the `min_shared_images`
floor; the range vet separating a surface pair from a
one-behind-the-other pair at equal image separation; `a_lo = 0` admitting
coincident observations; infinity and non-positive-radius exclusion;
CSR symmetry (`q ∈ N(p) ⇔ p ∈ N(q)`) and the documented neighbour
ordering; and empty inputs (no points, no shared images, single
observation per point).
