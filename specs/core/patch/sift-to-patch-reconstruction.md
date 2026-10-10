# Sift-Based → Patch-Based Reconstruction

## Overview

A `.sfmr` reconstruction records where each 3D point is seen in each image in
one of two ways. In the `sift_files` mode an observation is an index into the
`.sift` feature file of its image, so those files have to be kept with the
reconstruction, even when the file also keeps an inline copy of each
observation's 2D position. In the `embedded_patches` mode the file stores each
observation's 2D image position (its keypoint) with no index, gives each point a
small flat patch oriented in 3D (`(u, v)` half-vectors + normal, optionally with
an RGBA image of the patch's appearance), and records each source image's hash
itself, so it needs no `.sift` files. The pipeline in this spec, the one behind
`sfm embed-patches`, converts a loaded reconstruction from the first mode to the
second (`feature_source` `sift_files` → `embedded_patches`; the two modes are
defined in [sfmr-file-format.md](../../formats/sfmr-file-format.md),
"Observation source"). The purpose of the conversion is to make photometric
patch matching, in place of SIFT descriptor matching, decide each point's track
and keypoints: an image joins the track when the point's patch rendered in it
agrees with the other images' renders, an observation whose render does not
agree is dropped, and each kept keypoint is placed by aligning the image's
render to the point's reference render rather than kept at the SIFT detection. The `.sift`
files seed this and are not read afterwards.

In order, it builds an oriented patch per point (the `(u, v)` frame + normal),
then for each point expands its track with the other vetted views that see the
surfel and, over **`rounds` alternating passes** (default `rounds = 2`), refines
the patch normal and then re-localizes each observation's keypoint — dropping
the views that won't register — before compacting the result into a valid
`embedded_patches` reconstruction that carries an RGBA image per point. The
photometric normal refinement **down-weights oblique views** (and hard-drops
grazing ones) and carries a **fronto (front-facing) prior** that keeps a
low-parallax normal from drifting to a tilted, photometrically-equivalent pose.

This is a reconstruction-in / reconstruction-out transform at the
`SfmrReconstruction` (API) level.

A non-photometric **baseline** conversion also exists —
`SfmrReconstruction::to_embedded_patches` (exposed as `sfm xform
--to-embedded-patches`, see [xform-command.md](../../cli/reconstruction/xform/xform-command.md),
and as the SfM Explorer's `Convert to Embedded Patches`, see
[scene-graph.md](../../gui/scene-graph.md)). It skips
all photometric steps: it gives each finite point a mean-viewing-direction frame
(and each point at infinity a tangent-sphere frame around its direction, per the
[format's infinity-patch convention](../../formats/sfmr-file-format.md)), copies each
observation's keypoint and each image's hash straight from the `.sift` files, and
emits a valid `embedded_patches` reconstruction whose keypoints are exactly the
original SIFT detections — the whole point set preserved. It runs none of the
photometric steps of the pipeline below.

### What the baseline conversion reports, and how it is stopped

```rust
pub fn to_embedded_patches(
    &self,
    normal: PatchNormal,
    extent: PatchExtent,
    progress: &Progress<'_>,
) -> Result<Self, ReconstructionError>
```

It reads a `.sift` file per image -- twice over under the default
`FeatureSize` sizing, once for the keypoint scales the frames are sized from and
once for the detections and the image hashes -- so its cost grows with the image
count and it takes a `Progress` like every other kernel that can outlast a
frame ([operation-progress.md](../../gui/operation-progress.md)). Three stages
under it, sharing the bar 93/5/2:

| Phase | What it covers | What it notes |
|---|---|---|
| `patch frames` | the `PatchCloud::from_reconstruction` build, whose own passes are named underneath it ([patch-cloud.md](patch-cloud.md), "What a build reports") | the point count |
| `read keypoints` | one `.sift` per image: the detections, and the `image_file_xxh128` each image's hash is decoded from | the image count, plus one `Count` per image as it goes |
| `assemble` | the per-observation keypoint column, the output value, and its column validation | the observation count |

The weights follow the measured division. The two `.sift` walks read nearly the
same thing and neither expands a descriptor: the frame build's walk takes each
file's affine shapes, where the keypoint read takes the positions and the
metadata. What makes the first stage the largest is the sizing and framing it
does after its walk. On a 4054-image, 1.07M-point, 16.3M-observation capture the
three stages are 4.5 s, 1.9 s and 0.92 s with the files in cache, and 5.5 s,
2.5 s and 1.1 s on a colder one. **Every stage reports a fraction**: the walks count
their images, the passes over the points and the observations report on a
boundary every two-hundredth of the way through, and the bar therefore moves
from the first stage to the last rather than standing at zero for the frame
build and then jumping.

**Cancellation is polled between the stages, between the images of both walks,
and at those same boundaries within each pass**, so a stop lands within a
fraction of a percent of the stage it is asked in. A cancelled conversion
returns `ReconstructionError::Cancelled` and builds nothing: the call is a
function of its input and writes a new value, so there is no half-converted
state to leave behind. `&Progress::none()` reports nothing and never stops,
which is what the PyO3 binding passes: the Python signature is unchanged.

## Operating contract: surfel ops require `embedded_patches`

The gated surfel operations — `sfm xform --refine-normals` and `sfm
render-patches` — **require** a `feature_source == "embedded_patches"`
reconstruction and **reject** `sift_files` with an error naming the fix (run
`sfm xform --to-embedded-patches` first). `compare --strips` is the deliberate
exception: it stays an ungated dual-source diagnostic that builds patch clouds
from raw solves on the fly, because its strip montage is deeply `.sift`-tied
(see
[compare-command.md](../../cli/reconstruction/compare-command.md)). The motivation is in the keypoint-source
experiments (`reports/exp/2026-06-21-mvs-normal-refinement.md`):
an `embedded_patches` reconstruction *stores* a per-observation keypoint, so
refinement can position each view on its real detected feature instead of the
reprojected point center — which gives a cleaner cross-view consensus (and a
sharper reference bitmap), scaling with the solve's reprojection error.

Consequences of the contract:

- **The Rust `to_embedded_patches` is the one sift-consuming step.** That
  function — `SfmrReconstruction.to_embedded_patches` (the PyO3 binding, also
  surfaced as `sfm xform --to-embedded-patches` and as the viewer's
  `Convert to Embedded Patches`) -- is the only place that reads
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
  `specs/cli/reconstruction/xform/refine-normals-command.md`).
- **The low-level builder stays dual-mode.** `PatchCloud::from_reconstruction`
  (and the diagnostic `strips/_solve` engine and `scripts/exp_*`/`cmp_*`) may
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
   (`fronto_prior_weight · mean(v̂·n)²`); in each round, after the normal
   refinement and before the sub-pixel keypoint refinement, every observation
   more than `max_obliquity_deg` off the refined normal is dropped, the
   reference observation included (the sub-pixel pass's reference-view rule
   then picks a new reference from the views left). From round 2
   the refinement basis is capped at the `max_refine_views` most
   normal-informative views per point (output-lossless — every observation still
   registers and has its bitmap rendered; see
   [patch-normal-refine-view-subset.md](patch-normal-refine-view-subset.md)).
3. **Select the views (per point).** Run [patch-view
   selection](patch-view-selection.md): geometric candidacy plus photometric
   vetting against a track-seeded template yields the view set `G`.
4. **Starting keypoints (per point).** Each view of `G` the point already
   observes starts at its stored keypoint; a view step 3 added has no
   observation, so it starts at the point's projection `project_i(X_p)`.
5. **Align keypoints (per point).** Align each view to the point's reference
   render with the [keypoint-localization
   algorithm](patch-keypoint-localization.md), in one pass. The reference is
   the point's stored reference observation where an `embedded_patches` input
   stores one in `G`, or the one an earlier round recorded, and otherwise the
   [reference-view rule](reference-view.md)'s pick from the renders at the
   starting keypoints; its keypoint is not moved. The localizer drops views
   that cannot be aligned (grazing, out-of-frame, a tile that fixes no position,
   large-shift `max_shift_px`, a ZNCC against the reference below
   `min_absolute_zncc` or below `min_relative_zncc` times the median of the
   other views) and returns the kept views with their keypoints and their ZNCC
   against the reference. The sub-pixel pass
   ([keypoint-subpixel-refinement](keypoint-subpixel-refinement.md)) then
   refines the keypoints against the same reference **and renders each point's
   stored bitmap at them** (`refine_keypoints(render_bitmaps=True)`): the tile
   of the reference observation, rendered at its keypoint, or the fused mean of
   the views where there is none — points at infinity included, via the same
   `w`-aware render path — reporting per-point validity and the reference's
   image, which the output records as the point's reference observation (a point
   with fewer than two views, or with no reference and fewer than two views in
   frame for the mean, gets no bitmap).
6. **Cull unsupported points (per point).** Drop any point whose kept-view count
   fell below `min_views`, **and** any point the sub-pixel pass produced no
   valid bitmap for — one uniform rule for finite and infinity points
   (no point is kept with an all-black bitmap).
7. **Compact.** Renumber the surviving points and observations into a dense,
   valid `embedded_patches` reconstruction carrying the rendered bitmaps. Each
   surviving finite point's stored normal is re-derived from the frame being
   written — `n_p = normalize(u_p × v_p)` — so the two agree in the output; the
   incoming reconstruction's `normals_xyz` predates the refinement that rotated
   the frames and is not carried through. A point at infinity keeps its stored
   `(0, 0, 0)` row: its normal is implied by the direction, and refinement never
   moves its tangent-sphere frame.

## Where it lives

The pipeline is [_embed_patches.py](../../../src/sfmtool/_embed_patches.py),
driven by `sfm embed-patches`
([_commands/embed_patches.py](../../../src/sfmtool/_commands/embed_patches.py), a
thin wrapper); the write/compaction tail `compact_to_embedded_patches` and the
`image_file_hashes_from_sift` / `image_file_hashes_from_images` helpers live in
[_patch_compaction.py](../../../src/sfmtool/_patch_compaction.py). The producer is
Python **orchestration** — the per-point loop, point culling, and compaction —
over Rust kernels reached through the PyO3 bindings:

- The patch frame is built once by `SfmrReconstruction.to_embedded_patches`
  (mean-viewing seed, feature-size extent) and read back as the cloud via
  `recon.patches`; its normal is then refined by [normal
  refinement](patch-normal-refinement.md) (`refine_normals`,
  `use_stored_keypoints=True`) anchored on the carried-in SIFT keypoints
  (steps 0–2). The pipeline no longer asks normal refinement for bitmaps
  (`render_bitmaps` stays available there for the strips diagnostics); the
  stored bitmaps come from the sub-pixel keypoint refinement (step 5),
  rendered at the final keypoints.
- Per-point view selection is [patch-view selection](patch-view-selection.md), in
  `sfmtool-core::patch` — geometric candidacy plus photometric vetting against a
  track-seeded template.
- Per-point keypoint alignment is the [keypoint-localization
  algorithm](patch-keypoint-localization.md), which lives in
  `sfmtool-core::patch` (Rust) and reuses the same patch rendering.
- The source images are decoded into per-image pyramids **once**, up front:
  `embed_patches()` builds an `ImagePyramidSet` (a PyO3 class wrapping the
  rayon-parallel `ImageU8Pyramid` build) from the numpy image list and passes it
  to every `PatchCloud` kernel call. Each kernel also still accepts the raw
  numpy list (it then builds the pyramids for that one call — the back-compat
  path); the prebuilt set is level-for-level identical, so results are unchanged
  either way.

## Parameters (defaults)

These are the pipeline-exposed knobs; most are forwarded to the algorithm that
owns them (the localizer's `search` is described in the keypoint-localization
spec).

| parameter | default | forwarded to / meaning |
|---|---|---|
| `min_relative_zncc` | `0.7` | view selection **and** keypoint localizer: a view must agree at least this fraction as well as the reference (the track's self-agreement on admission; during alignment, the median ZNCC against the reference render of the views other than the reference) |
| `min_absolute_zncc` | `0.5` | keypoint localizer: drop a view whose ZNCC against the reference render is below this (`0` disables) |
| `max_member_zncc_self_similarity_radius` | `2.5` | keypoint localizer: drop a view whose own tile's ZNCC self-similarity radius is above this, patch-grid px (`0` disables) |
| `patch_size` | `11.0` | frame init: surfel size — full patch edge length, halved to the library half-extent and passed to `to_embedded_patches` (`extent="feature_size"`) |
| `max_shift_px` | ~3 | keypoint refiner: drop a view whose keypoint sits more than this from the point's projection (source-image px) |
| `min_views` | 2 | pipeline cull: drop a point left with fewer kept views |
| `rounds` | `2` | pipeline loop: number of alternating (normal-refine, keypoint-refine) passes (see [Pipeline](#pipeline)) |
| `max_obliquity_deg` | `80.0` | normal refinement: in each round, before the sub-pixel pass, drop observations (the reference included) viewing the surfel more than this off the refined normal (`90` disables) |
| `obliquity_weight_power` | `2.0` | normal refinement: exponent `p` of the multiplicative obliquity view-weight `\|v̂·n\|^p` in the robust consensus (`0` disables; `2` = cos²θ foreshortening) |
| `fronto_prior_weight` | `0.05` | normal refinement: weight `λ` of the additive fronto-parallel prior `λ·mean(v̂·n)²` pulling a low-parallax normal toward facing the cameras (`0` disables) |
| `max_refine_views` (`--refine-max-views`) | `8` | normal refinement: cap the round-2+ refinement basis at the N most normal-informative views/point (`0` = all); output-lossless ([patch-normal-refine-view-subset.md](patch-normal-refine-view-subset.md)) |
| `subpixel` | `True` | keypoint refiner: run the ECC sub-pixel refinement against the reference render once per round; `False` moves no keypoint after the localizer (the bitmap render still runs) ([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md)) |
| `localize_search_strategy` | `exhaustive` | keypoint localizer: discrete shift-grid traversal — `exhaustive` (full grid) or `plus_descent` (local descent); see [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md) |

## Scope

For each point, the conversion builds a patch frame (initializes and refines
the normal), selects its view set (track and photometrically vetted views),
aligns each view's keypoint to the point's reference render (the localizer
drops views that cannot be aligned), culls points left below `min_views`, and compacts the
result into a valid `embedded_patches` reconstruction. The observation set starts from the
input track, then is expanded with vetted views and filtered by drops.

The conversion does not move 3D points: where all of a point's views shift by
the same in-plane offset, which indicates a mis-located point, the point is
not re-triangulated. It uses each
observation's ZNCC against the reference render and its shift to prune and
then discards them; it does not write the format's optional
`observation_confidence` column.

Every patch is sized at `patch_size` times its SIFT feature scale; the
embedding does not choose a size per track. Choosing each track's size from a
short ladder of sizes is proposed in [patch-footprint-selection.md](../../drafts/patch-footprint-selection.md).

## Implementation notes

`embed_patches(recon, images, *, min_relative_zncc, min_absolute_zncc,
max_member_zncc_self_similarity_radius, patch_size, max_shift_px, min_views,
search, resolution, subpixel, rounds, max_obliquity_deg, obliquity_weight_power,
fronto_prior_weight, max_refine_views, max_zncc_self_similarity_radius,
localize_search_strategy, sampler, progress)` in
[_embed_patches.py](../../../src/sfmtool/_embed_patches.py) runs the whole pipeline (steps 0–7, iterated over
`rounds` alternating normal/keypoint passes): a single
`recon.to_embedded_patches(...)` bridge (the only `.sift` read) builds the
mean-viewing, feature-sized frames + inline SIFT keypoints + image hashes; the
cloud is read back via `embedded.patches` and its normal refined photometrically
over the embedded recon (`use_stored_keypoints=True`), anchored on the carried-in
keypoints; then view selection, keypoint alignment to the reference render, the
sub-pixel refinement against the same reference (which renders the stored
bitmaps at the final keypoints and reports per-point validity and the
reference's image — with `subpixel=False` it runs render-only so the bitmaps/validity are
still produced), and `compact_to_embedded_patches` (the write/compaction tail,
given the original `recon` for geometry carry-over and `embedded.image_file_hashes`
so there is no second `.sift` read; its `valid` mask drops the points with no
valid bitmap). The writer requires the patch frame for an
`embedded_patches` file (`has_uv_frames = true`).

Points at infinity flow through end to end — the kernels are first-class on them
through the `w`-aware render/selection/localization paths — and
`compact_to_embedded_patches` preserves their `w = 0` via `positions_xyzw`.
Normal refinement remains finite-only (an infinity point keeps its fixed
tangent-sphere frame), but the **reference bitmap does not depend on it**: the
sub-pixel keypoint refinement renders every point's stored bitmap — infinity
points included, through the same `w`-aware render path — so a surviving
infinity point carries a real texture rather than a zero `patch_bitmaps` row.
The flip side is uniform: any point (finite or infinity) for which no valid bitmap could be rendered is **dropped** by the final
compaction rather than kept with an all-black bitmap.

## Open questions

- The discard gates (`min_relative_zncc`, `max_shift_px`) want tuning across the
  four datasets before the defaults are fixed.
