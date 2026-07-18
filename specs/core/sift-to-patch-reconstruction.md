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
track with the other vetted views that see the surfel and, over **`rounds`
alternating passes** (default `rounds = 2`), refines the patch normal and then
re-localizes each observation's keypoint — dropping the views that won't
register — before compacting the result into a valid `embedded_patches`
reconstruction. The photometric normal refinement **down-weights oblique views**
(and hard-drops grazing ones) and carries a **fronto (front-facing) prior** that
keeps a low-parallax normal from drifting to a tilted, photometrically-equivalent
pose.

This is a reconstruction-in / reconstruction-out transform at the
`SfmrReconstruction` (API) level.

A non-photometric **baseline** conversion also exists —
`SfmrReconstruction::to_embedded_patches` (exposed as `sfm xform
--to-embedded-patches`, see [xform-command.md](../cli/xform-command.md)). It skips
all photometric steps: it gives each finite point a mean-viewing-direction frame
(and each point at infinity a tangent-sphere frame around its direction, per the
[format's infinity-patch convention](../formats/sfmr-file-format.md)), copies each
observation's keypoint and each image's hash straight from the `.sift` files, and
emits a valid `embedded_patches` reconstruction whose keypoints are exactly the
original SIFT detections — the whole point set preserved. It runs none of the
photometric steps of the pipeline below.

## Operating contract: surfel ops require `embedded_patches`

> _Status: **shipped** (2026-06-25). **Done:** the keypoint-aware refine kernel
> (`refine_normals(use_stored_keypoints=...)`), the `embed-patches` re-layer onto
> `to_embedded_patches` with `use_stored_keypoints`, and the hard precondition
> gating `sfm xform --refine-normals` and `sfm render-patches` to
> `embedded_patches` (both reject `sift_files` with a `UsageError` naming the
> fix). **Scoped out:** `compare --strips` is intentionally left ungated — it
> stays a dual-source diagnostic that builds patch clouds from raw solves on the
> fly (its strip montage is deeply `.sift`-tied; see
> `specs/cli/compare-command.md`). See the bullets below._

The gated surfel operations — `sfm xform --refine-normals` and `sfm
render-patches` — **require** a `feature_source == "embedded_patches"`
reconstruction and **reject** `sift_files` with an error naming the fix (run
`sfm xform --to-embedded-patches` first). `compare --strips` is the deliberate
exception (ungated, dual-source). The motivation is in the keypoint-source
experiments (`reports/exp/2026-06-21-mvs-normal-refinement.md`):
an `embedded_patches` reconstruction *stores* a per-observation keypoint, so
refinement can position each view on its real detected feature instead of the
reprojected point center — which gives a cleaner cross-view consensus (and a
sharper reference bitmap), scaling with the solve's reprojection error.

Consequences of the contract:

- **The Rust `to_embedded_patches` is the one sift-consuming step.** That
  function — `SfmrReconstruction.to_embedded_patches` (the PyO3 binding, also
  surfaced as `sfm xform --to-embedded-patches`) — is the only place that reads
  `.sift` files to build patches; everything downstream is
  `embedded_patches → embedded_patches`.
- **`embed-patches` calls `to_embedded_patches` as its first pipeline step.**
  `embed_patches()` step 0 is a single call to
  `SfmrReconstruction.to_embedded_patches(extent="feature_size",
  extent_value=patch_size/2, …)`, which returns a baseline `embedded_patches`
  reconstruction: a mean-viewing `(u, v)` frame per point, each observation's
  keypoint copied verbatim from its `.sift` detection, and each image's hash from
  the `.sift` metadata. Steps 2+ (normal refinement, view selection, keypoint
  localization) then run on that `embedded_patches` reconstruction. The pipeline
  no longer builds a patch cloud directly from the `sift_files` recon — the only
  `.sift` read is inside that one `to_embedded_patches` call.
- **Normal refinement can position views from the stored keypoints.** With
  `use_stored_keypoints=True`, each view's patch on an `embedded_patches` recon is
  rendered at its stored per-observation keypoint rather than at `project_i(X_p)`.
  `embed-patches` enables this, so its refine runs over the SIFT-detection
  keypoints `to_embedded_patches` carried in. `sfm xform --refine-normals` now
  does the same: gated to `embedded_patches`, its `apply` reads the stored frame
  back (`recon.patches`) and refines with `use_stored_keypoints=True`. Because it
  reuses that frame, it has no frame-sizing / seeding (`extent` /
  `extent_value` / `initial_normals`) or `save_patches` knobs — those live on
  `to_embedded_patches`, the step that builds the frame (see
  `specs/cli/xform-refine-normals-command.md`).
- **The low-level builder stays dual-mode.** `PatchCloud::from_reconstruction`
  (and the diagnostic `_solve_strips` engine and `scripts/exp_*`/`cmp_*`) may
  still build a cloud from either source by projecting; the precondition is
  enforced at the command / `xform` transform layer, not in the kernel.

## Inputs

- An in-memory `SfmrReconstruction` (points, tracks, camera poses + intrinsics).
  The pipeline builds each point's **patch frame** itself — the half-vectors
  `u_p`, `v_p` and normal `n_p = normalize(u_p × v_p)` — in steps 1–2.
- Source images, to render the patches from.

## Pipeline

Steps 2–5 form one **round**; the pipeline runs `rounds` of them (default `2`),
alternating normal refinement and keypoint refinement so each feeds the next.
Round 1 seeds from the SIFT detections (normal-refine → localize → sub-pixel
refine); each later round re-refines every normal against the *previous* round's
keypoints, then re-localizes the keypoints against the new normals — a fixed-point
alternation. The view set is expanded once (step 3, round 1) and only ever shrinks
thereafter (the per-round obliquity drop).

1. **Initialize a patch frame.** Seed each point's `(u, v)` frame with a starting
   normal from the mean viewing direction — the average of its point→camera
   directions (`to_embedded_patches`'s `normal="mean_viewing"`).
2. **Refine the normal photometrically.** Rotate each frame to maximize
   cross-view photometric consensus — [normal
   refinement](patch-normal-refinement.md); the refined frame is re-persisted.
   The robust consensus **down-weights oblique views** by
   `|v̂·n|^obliquity_weight_power` and adds a **fronto-parallel prior**
   (`fronto_prior_weight · mean(v̂·n)²`); after each round every observation
   more than `max_obliquity_deg` off the refined normal is dropped. From round 2
   the refinement basis is capped at the `max_refine_views` most
   normal-informative views per point (output-lossless — every observation still
   registers and fuses; see
   [patch-normal-refine-view-subset.md](patch-normal-refine-view-subset.md)).
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
   keypoints and quality signals. The sub-pixel pass
   ([keypoint-subpixel-refinement](keypoint-subpixel-refinement.md)) then settles
   the final keypoints **and fuses each point's consensus bitmap at them**
   (`refine_keypoints(render_bitmaps=True)`) — points at infinity included, via
   the same `w`-aware render path — reporting per-point validity (a point with
   no cross-view consensus gets no bitmap).
6. **Cull unsupported points (per point).** Drop any point whose kept-view count
   fell below `min_views`, **and** any point the sub-pixel pass produced no
   valid consensus bitmap for — one uniform rule for finite and infinity points
   (no point is kept with an all-black bitmap).
7. **Compact.** Renumber the surviving points and observations into a dense,
   valid `embedded_patches` reconstruction carrying the fused bitmaps.

## Where it lives

The producer is Python (`src/sfmtool/`) **orchestration** — the per-point loop,
point culling, and compaction — over Rust kernels reached through the PyO3
bindings:

- The patch frame is built once by `SfmrReconstruction.to_embedded_patches`
  (mean-viewing seed, feature-size extent) and read back as the cloud via
  `recon.patches`; its normal is then refined by [normal
  refinement](patch-normal-refinement.md) (`refine_normals`,
  `use_stored_keypoints=True`) anchored on the carried-in SIFT keypoints
  (steps 0–2). The pipeline no longer asks normal refinement for bitmaps
  (`render_bitmaps` stays available there for the strips diagnostics); the
  stored reference bitmaps come from the sub-pixel keypoint refinement (step 5),
  fused at the final keypoints.
- Per-point view selection is [patch-view selection](patch-view-selection.md), in
  `sfmtool-core::patch` — geometric candidacy plus photometric vetting against a
  track-seeded template.
- Per-point refinement is the [congealing
  algorithm](patch-keypoint-localization.md), which lives in
  `sfmtool-core::patch` (Rust) and reuses the same patch rendering and IRLS
  consensus.
- The source images are decoded into per-image pyramids **once**, up front:
  `embed_patches()` builds an `ImagePyramidSet` (a PyO3 class wrapping the
  rayon-parallel `ImageU8Pyramid` build) from the numpy image list and passes it
  to every `PatchCloud` kernel call. Each kernel also still accepts the raw
  numpy list (it then builds the pyramids for that one call — the back-compat
  path); the prebuilt set is level-for-level identical, so results are unchanged
  either way.

## Parameters (defaults)

These are the pipeline-exposed knobs; most are forwarded to the algorithm that
owns them (the congealing knobs `max_iters` / `search` live entirely in the
keypoint-localization spec).

| parameter | default | forwarded to / meaning |
|---|---|---|
| `min_relative_zncc` | ~0.7 | view selection **and** keypoint refiner: a view must agree at least this fraction as well as the reference (the track's self-agreement on admission; the views' median LOO ZNCC during refinement) |
| `patch_size` | `5.0` | frame init: surfel size — full patch edge length, halved to the library half-extent and passed to `to_embedded_patches` (`extent="feature_size"`) |
| `max_shift_px` | ~3 | keypoint refiner: drop a view whose keypoint sits more than this from the point's projection (source-image px) |
| `min_views` | 2 | pipeline cull: drop a point left with fewer kept views |
| `rounds` | `2` | pipeline loop: number of alternating (normal-refine, keypoint-refine) passes (see [Pipeline](#pipeline)) |
| `max_obliquity_deg` | `80.0` | normal refinement: after each round, drop observations viewing the surfel more than this off the refined normal (`90` disables) |
| `obliquity_weight_power` | `2.0` | normal refinement: exponent `p` of the multiplicative obliquity view-weight `\|v̂·n\|^p` in the robust consensus (`0` disables; `2` = cos²θ foreshortening) |
| `fronto_prior_weight` | `0.05` | normal refinement: weight `λ` of the additive fronto-parallel prior `λ·mean(v̂·n)²` pulling a low-parallax normal toward facing the cameras (`0` disables) |
| `max_refine_views` (`--refine-max-views`) | `8` | normal refinement: cap the round-2+ refinement basis at the N most normal-informative views/point (`0` = all); output-lossless ([patch-normal-refine-view-subset.md](patch-normal-refine-view-subset.md)) |
| `subpixel` | `1` | keypoint refiner: LK/ECC sub-pixel outer-sweep count applied once per round (per-sweep consensus); `0` disables movement (render-only bitmap fuse still runs), `≥1` = that many sweeps ([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md)) |
| `localize_search_strategy` | `plus_descent` | keypoint refiner: discrete shift-grid traversal — `plus_descent` (local descent) or `exhaustive` (full grid); see [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md) |
| `search_resolution_multiplier` | `1.0` | keypoint refiner: discrete-search resolution multiplier `m` (`1.0` = no-op; `>1` = supersampled grid); see [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md) |

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

_Status: **fully wired** in `src/sfmtool/_embed_patches.py`. `embed_patches(recon,
images, *, min_relative_zncc, patch_size, max_shift_px, min_views, max_iters,
search, resolution, search_resolution_multiplier, subpixel, rounds,
max_obliquity_deg, obliquity_weight_power, fronto_prior_weight, max_refine_views,
localize_search_strategy)` runs the whole pipeline (steps 0–7, iterated over
`rounds` alternating normal/keypoint passes): a single
`recon.to_embedded_patches(...)` bridge (the only `.sift` read) builds the
mean-viewing, feature-sized frames + inline SIFT keypoints + image hashes; the
cloud is read back via `embedded.patches` and its normal refined photometrically
over the embedded recon (`use_stored_keypoints=True`), anchored on the carried-in
keypoints; then view selection, keypoint congealing, the sub-pixel refinement
(which fuses the reference textures at the final keypoints and reports per-point
validity — with `subpixel=0` it runs render-only so the bitmaps/validity are
still produced), and `compact_to_embedded_patches` (the write/compaction tail,
given the original `recon` for geometry carry-over and `embedded.image_file_hashes`
so there is no second `.sift` read; its `valid` mask drops the points with no
consensus bitmap). The `sfm embed-patches` CLI
(`src/sfmtool/_commands/embed_patches.py`) is a thin wrapper over it.
`compact_to_embedded_patches` and the `image_file_hashes_from_sift` /
`image_file_hashes_from_images` helpers live in
`src/sfmtool/_patch_compaction.py`. The writer requires the patch frame for an `embedded_patches` file
(`has_uv_frames = true`)._

_Points at infinity flow through end to end (the kernels are first-class on them
since the patch pipeline gained `w`-aware rendering/selection/localization), and
`compact_to_embedded_patches` preserves their `w = 0` via `positions_xyzw`.
Normal refinement remains finite-only (an infinity point keeps its fixed
tangent-sphere frame), but the **reference bitmap no longer depends on it**: the
sub-pixel keypoint refinement fuses every point's consensus bitmap — infinity
points included, through the same `w`-aware render path — so a surviving
infinity point carries a real texture (the former "zero `patch_bitmaps` row"
limitation is fixed). The flip side is uniform: any point (finite or infinity)
for which no valid consensus bitmap could be fused is **dropped** by the final
compaction rather than kept with an all-black bitmap._
