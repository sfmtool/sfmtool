# Observation coverage: which image pixels existing tracks already claim

## Overview

Several per-image decisions need to know **which parts of an image are
already accounted for by reconstructed tracks**: spawning new candidate
tracks should aim at regions no existing observation claims, expansion
around an under-determined point should reach into directions with no
coverage, and a contested region (many overlapping claims) is a duplicate /
alias signal. This module rasterizes every observation's image-space
footprint — a disk at its keypoint with a caller-supplied radius — into
per-image occupancy grids, then answers batch queries against them.

Coverage is deliberately observation-based and depth-free: a pixel is
"claimed" because some track *observes* it there, with the footprint radius
supplied per observation (typically the feature's detection scale or the
projected patch extent). Counts are kept, not booleans, so sparse coverage
and contested coverage read differently.

## Structure

Per image, a grid of saturating `u8` counts at `1/cell_px` resolution
(`cell_px` in pixels, default 4): cell `(cx, cy)` covers the pixel square
`[cx·cell_px, (cx+1)·cell_px) × [cy·cell_px, (cy+1)·cell_px)` and its
count is the number of observation footprints containing the cell's
**center** (saturating at 255). Grid dimensions are
`ceil(width / cell_px) × ceil(height / cell_px)` per image, so cells at the
right/bottom edges may extend past the image; their centers can therefore
lie outside it, and such cells simply never get covered by clipping (see
below) — queries treat them like any other in-grid cell.

## Building

```rust
pub fn build(
    image_sizes: &[[u32; 2]],          // per image: [width, height] px
    track_image_indexes: &[u32],       // per observation
    keypoints_xy: &[[f64; 2]],         // per observation
    radii_px: &[f32],                  // per observation footprint radius
    cell_px: u32,                      // default 4
) -> ObservationCoverage
```

For each observation, every cell of its image whose center lies within
`radius` of the keypoint is incremented once. Observations with
non-positive or non-finite radius, or a keypoint so far outside the image
that the disk misses every cell, contribute nothing; disks partially
outside are clipped to the grid. Counts are order-independent, so the
per-image rasterization parallelizes over images (observations grouped by
image first) with deterministic results.

## Queries

All queries are batch (parallel arrays in, array out) and read-only.

- `counts_at(image_indexes, xy) -> u8 per query` — the count of the cell
  containing the pixel coordinate; 0 for out-of-grid coordinates.
- `covered_fraction(image_indexes, xy, radius_px) -> f32 per query` — of
  the cells whose centers lie within `radius` of `xy`, the fraction with
  count ≥ 1. 0 when no cell center falls in the disk.
- `uncovered_sectors(image_indexes, xy, radius_px, n_sectors) -> u32
  bitmask per query` — divide the disk of `radius` around `xy` into
  `n_sectors` equal angular sectors (sector `k` spans
  `[k, k+1) · 2π / n_sectors`, angles by `atan2(dy, dx)` of the cell
  center relative to `xy`, mapped like the surfel kernel's sector binning);
  bit `k` is set when sector `k` contains **at least one cell with count
  0** — an uncovered direction worth reaching into. A sector none of whose
  cells fall inside the grid contributes no set bit (outside the image is
  not spawnable, so it is never reported as uncovered). Cells at exactly
  `xy` (zero displacement) are skipped — they have no direction.
- `image_covered_fraction(image_index) -> f32` — fraction of the image's
  cells with count ≥ 1, an overall claim summary.

## API surface

The grids and their queries live in
[observation_coverage.rs](../../../crates/sfmtool-core/src/analysis/observation_coverage.rs),
bound as the `sfmtool._sfmtool.analysis.ObservationCoverage` class.

Core: an `ObservationCoverage` struct owning the per-image grids, with the
builder and the four queries as methods, plus read access to a grid
(`grid(image_index) -> (&[u8], width_cells, height_cells)`) so callers can
run analyses the queries don't anticipate.

The Python binding exposes the same as a class:
`ObservationCoverage(image_sizes, track_image_indexes, keypoints_xy,
radii_px, cell_px=4)` with methods `counts_at`, `covered_fraction`,
`uncovered_sectors`, `image_covered_fraction`, and `grid(image_index)`
returning the counts as a `(height_cells, width_cells)` numpy array.
Inputs accept `SfmrReconstruction` accessor dtypes directly; batch query
methods take numpy arrays and return numpy arrays; the GIL is released
around build and batch queries.

## Testing

Sibling `tests.rs` under `analysis/observation_coverage/` covers: disk
rasterization by the cell-center rule (with radii chosen off cell-center
distance boundaries so a 1-ULP difference cannot flip a cell); count
accumulation from overlapping footprints and saturation at 255; clipping
of partially-out-of-image disks and no-op fully-outside observations;
non-positive and non-finite radii contributing nothing; `counts_at`
in-grid and out-of-grid; `covered_fraction` on empty, partial, and full
neighbourhoods; `uncovered_sectors` on an empty image (every in-grid
sector set), a fully covered disk (no bit set), a half-covered
neighbourhood (exactly the uncovered half's bits), and a query at the
image edge (sectors falling outside the grid contribute no bits);
`image_covered_fraction` bookkeeping; multi-image independence; and empty
inputs (no observations, zero-size batch queries).
