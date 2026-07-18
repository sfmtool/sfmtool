# Covisibility Selection: Displacement, Thinning, Reach

**Status:** Specified — not implemented. Extends `ClusterCovisibility`
(`crates/sfmtool-core/src/matching/covisibility.rs`, bound as
`sfmtool._sfmtool.matching.ClusterCovisibility`).

## Purpose

Callers working from cluster tracks need three selection primitives that
today exist only as per-caller array code: how far apart two covisible
images are in appearance (displacement), a redundancy-thinned working
subset (thinning), and how much of a capture a chosen subset connects to
(reach). All three are order-free — nothing depends on image ordering —
and deterministic given a seed.

## Construction

`ClusterCovisibility` gains an optional per-observation position input:

```
ClusterCovisibility.from_arrays(cluster_starts, member_images, n_img,
                                positions_xy=None, seed=0)
```

`positions_xy` (`f64 [n_obs, 2]`, aligned with `member_images`) enables
the displacement queries; construction without it leaves them
unavailable (calls raise). The shared-cluster counts are unchanged.

## Pair displacement

One sampled pass at construction: every cluster with two or more member
images contributes one seeded uniformly-sampled member pair (same-image
pairs skipped); squared-root pixel distances accumulate per image pair.

- `pair_displacement()` → `f64 [n_img, n_img]`: mean feature displacement
  per covisible pair, `0` where no sample landed.
- `pair_displacement_counts()` → `u32 [n_img, n_img]`: samples behind each
  mean, for callers that gate on support.

## Thinning

A redundancy-thinned subset keeps an image only when its best
shared-cluster count against the already-kept set falls in a band
`[tau/8, tau)`: images above the band duplicate a kept viewpoint, images
below it are disconnected from the skeleton. The greedy sweep visits
images in decreasing isolation (largest nearest-covisible-partner
displacement first, falling back to construction order without
positions), so the kept set is invariant to input permutation up to
exact ties.

- `thin(tau)` → sorted image indices.
- `thin_to(target)` → sorted image indices: binary-searches `tau` (the
  kept count grows monotonically with `tau`) and returns the subset whose
  size is closest to `target`.

## Reach

`reach(images, min_shared=8)` → `f64`: the fraction of all images that
share at least `min_shared` clusters with at least one image of the
given subset (subset members count as reached). A subset confined to one
viewpoint neighbourhood has low reach however large the capture; a
subset spanning the capture approaches 1.

## Bindings

All three query families on the existing Python class; arrays validated
like the constructor's. `thin`/`thin_to` return `uint32` arrays;
`reach` a float.

## Testing requirements

- Displacement: a synthetic two-cluster scene with known geometry yields
  the expected pair means; seeded determinism; construction without
  positions raises on displacement queries only.
- Thinning: on a synthetic chain with geometrically decaying covisibility,
  `thin` reproduces the expected band selection; `thin_to` hits requested
  sizes across a sweep; permuting input image ids permutes the output
  consistently (order-free).
- Reach: hand-built subsets on a known graph produce exact fractions;
  `min_shared` respected at the boundary.
- Binding parity with the Rust results.

## Non-goals

- Pair selection policy (which pairs to estimate geometry on) — callers
  compose it from these queries.
- Any use of image ordering.
