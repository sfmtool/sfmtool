# `sfm xform --refine-keypoints` Design

The `sfm xform --refine-keypoints` operation surfaces the sub-pixel keypoint
refinement described in
[keypoint-subpixel-refinement.md](../../../core/patch/keypoint-subpixel-refinement.md)
as a reconstruction transform: it rewrites each observation's stored 2D
keypoint in place, moving it to the sub-pixel position that best matches the
point's reference render.

## What it does

Refines each observation's stored 2D keypoint to **sub-pixel** by a local
continuous photometric solve: forward-additive ECC Gauss–Newton against the
point's **reference render**, the `R×R` render of its reference observation at
its stored keypoint, in one pass. The reference observation is the one the
point stores in `tracks/reference_observations`, and where it stores none, the
reference-view rule's pick from the views' renders at their stored keypoints;
where the rule picks none it would store, the template is the views' fused mean
and no view is the reference. The reference's keypoint is not moved. The
binding does no grid search, **changes no view membership**, and is **never
worse than the seed** (a step is accepted only if it raises the ECC score and
stays in frame). Points at infinity are refined like finite ones, not skipped.

It is therefore a **pure in-place modifier** — the keypoint counterpart of
`--refine-normals`:

- The point count, positions, poses, cameras, and normals are unchanged.
- The track structure (`track_image_indexes`, `track_point_indexes`,
  `observation_counts`) is byte-identical to the input; no view or point is
  dropped.
- Only `keypoints_xy` values move (and, with `bitmaps` — on by default, the
  per-point patch textures are re-rendered at the refined keypoints; disable
  with `bitmaps=false`).

**Precondition:** requires an `embedded_patches` reconstruction and rejects
`sift_files` with a `UsageError` pointing at `sfm xform --to-embedded-patches`
(enforced per-step in `xform/_apply.py` via the
`RefineKeypointsTransform.required_feature_source` attribute, so a
`--to-embedded-patches --refine-keypoints` chain converts first and passes).
The refiner is *local*: it needs a seed already close to the optimum (≲ 1 px),
and only an `embedded_patches` recon carries real per-observation keypoints to
seed from. The refiner seeds every view from the recon's stored inline keypoint
and refines each point's full track (the binding's `view_sets=None` /
`starting_keypoints=None` default path), passing each point's stored reference
observation as `reference_images`; the stored per-point patch frame
(`recon.patches`) supplies the patch geometry and is never rebuilt.

Because the refinement is photometric it reads the workspace source images
(`workspace_dir / image_name`), exactly like `--refine-normals`; a missing
image (or unresolvable workspace) is a hard error (`FileNotFoundError`).

## Command syntax

The operation is `RefineKeypointsTransform` in
[_refine_keypoints.py](../../../../src/sfmtool/xform/_refine_keypoints.py),
wired through `parse_refine_keypoints_params` in
[_arg_parser.py](../../../../src/sfmtool/xform/_arg_parser.py) and the
`--refine-keypoints` Click option in
[xform.py](../../../../src/sfmtool/_commands/xform.py); the refinement itself
is the `PatchCloud.refine_keypoints` PyO3 binding.

```
sfm xform <input.sfmr> [<output.sfmr>] --refine-keypoints [<params>] [...]
```

`--refine-keypoints` takes an **optional** comma-separated parameter string of
`key=value` modifiers (the Click option is `is_flag=False, flag_value=""`, so
all three forms work: bare `--refine-keypoints`, space-separated
`--refine-keypoints max_gn_steps=20`, and joined
`--refine-keypoints=max_gn_steps=20` — while a following option, e.g.
`--refine-keypoints --refine-normals`, is left untouched). With no value it
runs the binding defaults.

```
--refine-keypoints
--refine-keypoints max_gn_steps=20,sampler=anisotropic
--refine-keypoints bitmaps=false
--to-embedded-patches --refine-keypoints --refine-normals
```

### `key=value` modifiers

These pass straight through to `PatchCloud.refine_keypoints`, reusing the
binding's own defaults — the "Default" column matches each binding default
exactly, so the CLI re-specifies nothing and the two layers cannot drift.

| Key                    | Default         | Forwards to                                    |
|------------------------|-----------------|------------------------------------------------|
| `resolution`           | `24`            | `refine_keypoints` (R×R patch grid)            |
| `window`               | `gaussian_disk` | `refine_keypoints` (`gaussian_disk`/`gaussian`/`uniform`) |
| `window_sigma`         | `0.6`           | `refine_keypoints`                             |
| `sampler`              | `per_view`      | `refine_keypoints` (`per_view`/`bilinear`/`bilinear_mip`/`anisotropic`; `per_view` applies the sampler rule to each view, `anisotropic` where `bilinear_mip` would read its less compressed axis at least 1.5× too coarsely and `bilinear_mip` otherwise; `bilinear_mip` takes one bilinear tap from the mip level nearest the warp's compression, bounding the aliasing `bilinear` suffers on cross-scale views at the same cost; `anisotropic` resolves oblique footprints; the value+gradient render this step reads has no AVX2 kernel and costs 2.8–7× the `bilinear_mip` one) |
| `robust_iters`         | `3`             | `refine_keypoints` (IRLS passes for the fused mean, where it is the template) |
| `max_gn_steps`         | `10`            | `refine_keypoints` (Gauss–Newton steps per view) |
| `convergence_px`       | `0.01`          | `refine_keypoints` (per-view stop, patch-grid px) |
| `max_offset_px`        | `2.0`           | `refine_keypoints` (max per-view drift from the seed, patch-grid px) |
| `bitmaps`              | `true`          | render + persist the per-point RGBA patch bitmaps (below); `bitmaps=false` skips the render |

Unknown keys, malformed `key=value` tokens (no `=`, empty key), duplicate keys,
or unparseable values raise `click.UsageError`; range/enum validation lives in
the `RefineKeypointsTransform` constructor (surfacing as `UsageError` through
the CLI), consistent with the other parsers in `xform/_arg_parser.py`.

## Write-back semantics

Because the binding changes no view membership and returns each point's
keypoints in input (track) order, the write-back **copies the recon's stored
`keypoints_xy` and overwrites only the refined observations**, scattering
through a `(point_index, image_index) → observation row` index built from the
recon's own track arrays. The track arrays and `observation_counts` are never
rebuilt or passed to `clone_with_changes` — the output's track structure is the
input's, byte for byte.

Each written keypoint is clamped to the largest in-frame f32 for its image's
camera (`np.nextafter(width, 0)` / `np.nextafter(height, 0)`): the refiner
keeps keypoints strictly in-frame in f64, but the f32 the format stores can
round a near-edge value up to exactly width/height, which the writer's
`< width` check rejects — failing the whole save. This mirrors the clamp in the
`embed-patches` pipeline (`src/sfmtool/_embed_patches.py`).

**Persisting the patch bitmaps (`bitmaps`).** With `bitmaps` (the default) the
binding additionally renders each point's stored bitmap at the **final**
refined keypoints, the tile of the reference observation the views were
aligned to, or the fused mean of the views, naming no observation, where there
is none ([reference-view.md](../../../core/patch/reference-view.md) § "The
stored bitmap"), and the command scatters them into a `(point_count, R, R, 4)`
uint8 array (zero rows where the point produced no bitmap) and records each
point's reference observation, attached via
`clone_with_changes(patches=cloud, patch_bitmaps=…,
reference_observations=…)`; the stored frame is
re-persisted alongside so the bitmaps have a frame to attach to (the frame
itself is unchanged — keypoints moved, not the surfel). A point whose
`tracks/reference_observations` entry already names an observation keeps it
where its views were aligned to it, which is wherever it renders at its
keypoint; each refined point records the reference its views were aligned
to, so a point at `-1` takes the rule's pick, and a point whose stored
reference does not render, and so aligned nothing, records `-1` and keeps the
fused mean the refiner rendered. Every point with a
reference then has its bitmap rendered again from that observation at its
refined keypoint as the file stores it, in `f32` (`render_from_references` in
[`_patch_compaction.py`](../../../../src/sfmtool/_patch_compaction.py), through
`PatchCloud.render_bitmaps(referenced_only=True)`), so dropping and adding the
bitmaps later gives the same bytes. On by default so the
refined reconstruction carries its per-point patch textures and can display them
without re-rendering; it costs a tile render and a self-similarity reading per view per
point, so a multi-stage pipeline can pass `bitmaps=false` on intermediate stages
and render once on the finalizing stage. With `bitmaps=false` the command
writes the keypoints and drops any stored bitmaps, which were rendered at the
old keypoints, as `--refine-normals bitmaps=false` does. With no bitmap
written there is nothing for the reference to agree with, so a point that
stores a reference `≥ 0` keeps it, even one whose tile does not render at its
keypoint and so aligned nothing, and a point at `-1` records the rule's pick
its views were aligned to; a later `--add-patch-bitmaps` renders each point
from the reference it then stores.

The transform prints a one-line summary in the established `xform` style over
the finitely-scored views, the reference's included (a point with fewer than
two views, or with no template rendered, carries NaN scores and keeps its
seeds):

```
  Refined N keypoints (mean |offset| 0.142 patch-grid px)
```

## Ordering and interactions

- **Invariant to global similarity.** `--rotate` / `--translate` / `--scale`
  move points and poses together, so the renders the views are aligned to are
  unchanged;
  ordering relative to those is immaterial.
- **Repeatable.** A second pass starts from the first's stored keypoints and
  can move them further (within `max_offset_px` of the *new* seed).
- **Images must be resolvable.** Fails fast if `workspace_dir` or any image is
  missing, exactly like `--refine-normals` and the `.sift`-reading ops.
- **`--localize-keypoints` is the upstream search.** That op surfaces
  `PatchCloud.localize_keypoints` — basin *search*, a heavier, structural
  operation that drops views and points
  ([localize-keypoints-command.md](localize-keypoints-command.md)) — and is
  what puts seeds inside the basin this op needs when the stored keypoints may
  be further than ~1 px from the optimum; run it first, as
  `--localize-keypoints --refine-keypoints`. The localizer renders no bitmaps,
  so a following `--refine-keypoints` (bitmaps on by default) also regenerates
  them.

## Performance and memory

Same envelope as `--refine-normals`: the binding loads **all** full-resolution
images (plus pyramids) into memory at once and releases the GIL during the
solve, parallelizing across points. Work scales with observations × GN steps,
each view refined once; there is no streaming of the image set.
