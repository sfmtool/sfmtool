# Candidate track spawning: congeal new tracks at patch-frame offsets

**Status:** Implemented (2026-08-01). Core in `crates/sfmtool-core/src/patch/spawn.rs`,
exposed to Python as `sfmtool._sfmtool.patches.spawn_candidate_tracks`.
Bundled change: the `localize_keypoints` binding gains the core's existing
`starting_keypoints` parameter (see below).

## Overview

Given reconstructed points with patch frames, spawn **candidate tracks** at
chosen in-plane offsets from those parents: place a synthetic patch, find
it photometrically in the parent's views, and triangulate what was found —
exactly the way a real track is congealed, with the same acceptance gates.
Two callers: surfel-normal expansion places candidates along uncovered
directions and uses the surviving *positions* as extra fit neighbours;
densification places candidates over unclaimed image regions
([observation coverage](../analysis/observation-coverage.md)) and assembles the
survivors into new reconstruction tracks. The primitive itself neither
picks offsets nor assembles tracks — it turns (parent, offset) requests
into vetted `(position, views, keypoints)` results, batch.

A candidate that fails any gate is reported with the stage that killed it,
not silently dropped: callers budget and diagnose on those counts.

## Candidate construction

Per request `(parent p, du, dv)`:

- Center `X_c = X_p + du · hu_p + dv · hv_p` — offsets are in units of the
  parent's half-extent vectors, so a request speaks the patch's own scale
  and stays in its plane.
- Frame: the parent's `(hu, hv)` translated to `X_c` (same orientation and
  extents — the candidate hypothesizes locally planar surface).
- View set: per candidate, the images in which the pipeline will *attempt*
  to find it — its whole search scope. Localization can only drop views
  from this set, never add any, so it is the ceiling on the spawned
  track's observations and the baseline `too_few_views` is judged
  against. It is caller-supplied because deciding which images plausibly
  see a hypothetical 3D spot is visibility knowledge the kernel does not
  have: expansion passes the parent's views (a candidate one patch
  diameter away is imaged by essentially the same cameras), while a
  densification caller can propose views from frustum or coverage
  queries. The set should stay tight — every view in it congeals against
  every other (the uncapped consensus basis below), so an inflated set
  costs quadratically and dilutes the photometric consensus with views
  that never see the spot.

Candidates are finite: the offsets displace the frame in world units and the
output is a triangulated position, neither of which a point at infinity has, so
an infinity parent is rejected rather than gated.

## Pipeline

All candidates go through the existing batch kernels as **one** batch each
(one patch cloud of all candidates, one localization call, one sub-pixel
refinement call, one triangulation call — the composition adds no
parallelism of its own):

1. **Discrete localization** over each candidate's views, seeded at the
   view's projection of `X_c`, with the kernel's `search` /
   `max_shift_px` semantics. Views the localizer rejects are gone. The
   consensus basis is uncapped: a candidate carries its parent's view set,
   which is small, so every view congeals against every other.
2. **Sub-pixel refinement** (`subpixel_sweeps`; 0 skips) seeded at the
   localized keypoints.
3. **Triangulation** from the refined keypoints' camera rays.
4. **Gates**, in order, each recording its casualty:
   - `too_few_views` — fewer than `min_views` views survived localization.
     A candidate below the floor is not refined or triangulated at all.
   - `bad_triangulation` — non-finite triangulation, a degenerate solve
     (unobservable depth, so the reported point is a minimum-norm artifact
     rather than an intersection), or a point that is not in front of every
     surviving camera.
   - `high_reproj` — RMS reprojection error of the triangulated position
     against the refined keypoints exceeds `max_reproj_rms_px`.

Survivors report status `spawned`. Results are deterministic: the
underlying kernels are deterministic and candidates are independent.

## API

```rust
pub struct SpawnParams {
    pub resolution: u32,          // sampling grid, as in localization (24)
    pub search: f64,              // localizer search half-width, grid px (6.0)
    pub max_shift_px: f64,        // localizer shift gate, image px (8.0)
    pub subpixel_sweeps: u32,     // refinement outer sweeps (1)
    pub min_views: u32,           // surviving-view floor (3)
    pub max_reproj_rms_px: f64,   // acceptance gate, image px (2.0)
}

pub fn spawn_candidate_tracks(
    views: &[ProjectedImage<'_>],     // camera + pose + pyramid per image
    cloud: &PatchCloud,               // the parents
    parents: &[u32],                  // per candidate; may repeat
    offsets_uv: &[[f64; 2]],          // per candidate (du, dv)
    view_sets: &[Vec<u32>],           // per candidate
    params: &SpawnParams,
) -> SpawnedTracks
```

`SpawnedTracks`, parallel over candidates with CSR observations:

- `status` — u8: 0 `spawned`, 1 `too_few_views`, 2 `bad_triangulation`,
  3 `high_reproj`.
- `positions` — `(n, 3)` f64, the triangulated position (`NaN` rows for
  non-`spawned`).
- `requested_centers` — `(n, 3)` f64, `X_c` (always filled; lets callers
  measure displacement).
- `reproj_rms_px`, `n_views` — per candidate. `reproj_rms_px` is `NaN` for a
  candidate that died before the reprojection stage; `n_views` counts the views
  actually carried into triangulation, so it is `0` for a candidate the view
  floor stopped.
- `obs_offsets` (`n + 1`), `obs_view_indexes`, `obs_keypoints_xy`
  (`(n_obs, 2)` f64) — the surviving observations of every candidate that
  reached triangulation, in view-index order.

The Python binding (`views, images, cloud, parents, offsets_uv, view_sets`)
takes the existing `CameraViews` / `ImagePyramidSet` / `PatchCloud` objects —
or, like its sibling patch kernels, a reconstruction and a list of images —
plus numpy arrays (`view_sets` as a list of sequences), keyword params with the
defaults above, and returns the result arrays as a dict keyed as above. It
raises `ValueError` for a parent index out of range, a parent at infinity, an
`offsets_uv` / `view_sets` length or shape mismatch, and a view index out of
range for the scene.

## `localize_keypoints` gains `starting_keypoints`

The core localization kernel accepts per-view starting keypoints and
documents that `None` seeds every view at the point's own projection; the
Python binding predates the parameter and always passes `None`. As part of
this change the binding exposes it — same shape as `refine_keypoints`'s
existing parameter (per point, keypoints parallel to that point's view
set), optional, default `None` preserving today's behaviour exactly. A point
absent from the map keeps the projection seeding, which the core batch entry
learns to express as an empty per-patch seed list — the convention its
`view_scores` already uses for "this point is unscored". This
lets a caller localize a point around its *stored observations* rather
than around its possibly-wrong projection — the seed and the shift gate
then both anchor on evidence the caller trusts.

## Testing

Sibling tests under `patch/spawn/` following the patch kernels' synthetic-
image test idiom: a textured plane seen by several cameras, where a
candidate offset inside the textured region localizes, refines,
triangulates onto the plane (status `spawned`, position on the plane
within tolerance, `n_views` full); a candidate pushed off every image
(`too_few_views`); a `min_views` floor just above the surviving count
(`too_few_views`); an unreachable `max_reproj_rms_px` (`high_reproj`);
`subpixel_sweeps = 0` still spawning (discrete-only); multiple candidates
per parent in one batch matching the same candidates spawned separately
(batch independence); and CSR bookkeeping (offsets sum to observation
count, view-index ordering). Binding tests exercise the dict surface, the
dtype acceptance, `ValueError` on malformed inputs, and the new
`starting_keypoints` parameter of `localize_keypoints` (explicit seeds at
the true keypoints reproduce the default behaviour when the projection is
already correct, and recover a point whose cloud position is displaced
while its seeds point at the true image locations — the case the default
projection seeding cannot recover).
