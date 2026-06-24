# Sift-Based → Patch-Based Reconstruction

_Status: draft for review. The pipeline behind `sfm embed-patches` (see
[embed-patches-command.md](../cli/embed-patches-command.md)): it converts an
in-memory `SfmrReconstruction` whose observations reference external `.sift`
features into one whose observations carry inline, patch-derived keypoints
(`feature_source` `sift_files` → `embedded_patches`; the modes are defined in
[sfmr-file-format.md](../formats/sfmr-file-format.md), "Observation source"). The
per-point keypoint refinement it calls is specified in
[patch-keypoint-localization.md](patch-keypoint-localization.md)._

## Overview

A `sift_files` reconstruction locates each observation by a reference into an
external `.sift` file. An `embedded_patches` reconstruction instead carries each
point's patch inline — a 3D patch frame (`(u, v)` half-vectors + normal) with a
reference bitmap — plus a 2D keypoint per observation locating where each image
sees that patch. This pipeline performs
that change of representation on a loaded reconstruction: it builds an oriented
patch per point (the `(u, v)` frame + normal), then for each point expands its
track with the other vetted views that see the surfel, refines each
observation's keypoint, drops the views that won't register, and compacts the
result into a valid `embedded_patches` reconstruction.

This is a reconstruction-in / reconstruction-out transform at the
`SfmrReconstruction` (API) level.

## Inputs

- An in-memory `SfmrReconstruction` (points, tracks, camera poses + intrinsics).
  The pipeline builds each point's **patch frame** itself — the half-vectors
  `u_p`, `v_p` and normal `n_p = normalize(u_p × v_p)` — in steps 1–2.
- Source images, to render the patches from.

## Pipeline

1. **Initialize a patch frame.** Seed each point's `(u, v)` frame with a starting
   normal from the mean viewing direction — the average of its point→camera
   directions (`initial_normals=mean_viewing`).
2. **Refine the normal photometrically.** Rotate each frame to maximize
   cross-view photometric consensus — [normal
   refinement](patch-normal-refinement.md), with `save_patches`.
3. **Select the views (per point).** Run [patch-view
   selection](patch-view-selection.md): geometric candidacy plus photometric
   vetting against a track-seeded template yields the view set `G`.
4. **Project starting keypoints (per point).** For each view in `G`, project the
   point to its naive keypoint `project_i(X_p)` — the seed the refinement starts
   from.
5. **Refine keypoints (per point).** Refine each seed with the
   [keypoint-localization algorithm](patch-keypoint-localization.md): it drops
   views that won't
   co-register (grazing, out-of-frame, large-shift `max_shift_px`, low-agreement
   `min_relative_zncc`) in-loop, and returns the kept views with their refined
   keypoints and quality signals.
6. **Cull under-supported points (per point).** Drop any point whose kept-view
   count fell below `min_views`.
7. **Compact.** Renumber the surviving points and observations into a dense,
   valid `embedded_patches` reconstruction.

## Where it lives

The producer is Python (`src/sfmtool/`) **orchestration** — the per-point loop,
point culling, and compaction — over Rust kernels reached through the PyO3
bindings:

- Building the patch frame reuses [normal
  refinement](patch-normal-refinement.md): `PatchCloud.from_reconstruction`
  with the `mean_viewing` seed + `refine_normals` + `save_patches` (steps 1–2).
- Per-point view selection is [patch-view selection](patch-view-selection.md), in
  `sfmtool-core::patch` — geometric candidacy plus photometric vetting against a
  track-seeded template.
- Per-point refinement is the [congealing
  algorithm](patch-keypoint-localization.md), which lives in
  `sfmtool-core::patch` (Rust) and reuses the same patch rendering and IRLS
  consensus.

## Parameters (defaults)

These are the pipeline-exposed knobs; most are forwarded to the algorithm that
owns them (the congealing knobs `max_iters` / `search` live entirely in the
keypoint-localization spec).

| parameter | default | forwarded to / meaning |
|---|---|---|
| `min_relative_zncc` | ~0.7 | view selection **and** keypoint refiner: a view must agree at least this fraction as well as the reference (the track's self-agreement on admission; the views' median LOO ZNCC during refinement) |
| `patch_size` | inherits `refine-normals`' `extent_value` | frame init: surfel size — full patch edge length, halved to the library half-extent (see `refine-normals`) |
| `max_shift_px` | ~3 | keypoint refiner: drop a view whose keypoint sits more than this from the point's projection (source-image px) |
| `min_views` | 2 | pipeline cull: drop a point left with fewer kept views |

## Scope

**v1:** for each point, build a patch frame (initialize + refine the normal),
select its view set (track + photometrically-vetted views), refine each
view's keypoint from its projection (the refiner drops views that won't
co-register), cull points left below `min_views`, and compact the result into a
valid `embedded_patches` reconstruction. The observation set starts from the
input track, then is expanded with vetted views and filtered by drops.

**Future work (not v1):**
- **Per-pixel robustness** and other normal/template refinements feeding a better
  consensus.
- **Re-centring** — the *common* in-plane shift across a point's views indicates
  a mis-located 3D point; re-triangulating from it is point QC, likely a separate
  `xform` step rather than part of this conversion.
- **Per-observation quality output** — v1 *uses* each observation's LOO / shift
  to prune (above) but then discards the numbers; persisting them as an ancillary
  per-observation field, so downstream tools can read each kept keypoint's
  quality, waits on the format growing optional per-observation fields (out of v4
  scope today).

## Open questions

- The discard gates (`min_relative_zncc`, `max_shift_px`) want tuning across the
  four datasets before the defaults are fixed.

_Status: the **write/compaction tail** (step 7) is implemented in
`src/sfmtool/_embed_patches.py` as `compact_to_embedded_patches(recon, cloud,
localizations, image_file_hashes, *, patch_bitmaps=None, min_views=2)`: it culls
points left below `min_views`, renumbers the survivors into a dense point set,
flattens the kept per-point keypoint-localization results into point-then-image
sorted track + `keypoints_xy` arrays, carries the surviving patch frame (via the
new `PatchCloud.from_halfvec_arrays`) and bitmaps, and emits an
`embedded_patches` `SfmrReconstruction` through `clone_with_changes` (ready to
`save()`). `image_file_hashes_from_images(recon)` supplies the per-image identity
hashes from the workspace image bytes. The writer requires the patch frame for an
`embedded_patches` file (`has_uv_frames = true`). The upstream orchestration
(steps 1–6, running the kernels) and the `sfm embed-patches` CLI are not yet
wired up — a caller runs the kernels and hands the results to the compaction._
