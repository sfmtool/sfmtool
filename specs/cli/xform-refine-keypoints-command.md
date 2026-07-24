# `sfm xform --refine-keypoints` Design

**Status:** Implemented (2026-07-05) in `src/sfmtool/xform/_refine_keypoints.py`
(`RefineKeypointsTransform`), wired through `xform/_arg_parser.py`
(`parse_refine_keypoints_params`) and the `--refine-keypoints` Click option in
`_commands/xform.py`. Surfaces the subpixel keypoint refinement
(`specs/core/keypoint-subpixel-refinement.md`, implemented in the
`PatchCloud.refine_keypoints` PyO3 binding) as an `sfm xform` operation that
rewrites a reconstruction's per-observation `keypoints_xy` in place.

## What it does

Refines each observation's stored 2D keypoint to **sub-pixel** by a local
continuous photometric solve: forward-additive ECC Gauss–Newton against a
robust cross-view consensus. The binding does no grid search, **changes no view
membership**, and is **never worse than the seed** (a step is accepted only if
it raises the ECC score and stays in frame). Points at infinity are refined
like finite ones, not skipped.

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
`starting_keypoints=None` default path); the stored per-point patch frame
(`recon.patches`) supplies the patch geometry and is never rebuilt.

Because the refinement is photometric it reads the workspace source images
(`workspace_dir / image_name`), exactly like `--refine-normals`; a missing
image (or unresolvable workspace) is a hard error (`FileNotFoundError`).

## Command syntax

```
sfm xform <input.sfmr> [<output.sfmr>] --refine-keypoints [<params>] [...]
```

`--refine-keypoints` takes an **optional** comma-separated parameter string of
`key=value` modifiers (the Click option is `is_flag=False, flag_value=""`, so
all three forms work: bare `--refine-keypoints`, space-separated
`--refine-keypoints max_outer_sweeps=2`, and joined
`--refine-keypoints=max_outer_sweeps=2` — while a following option, e.g.
`--refine-keypoints --refine-normals`, is left untouched). With no value it
runs the binding defaults.

```
--refine-keypoints
--refine-keypoints max_outer_sweeps=2,sampler=anisotropic
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
| `sampler`              | `bilinear`      | `refine_keypoints` (`bilinear`/`bilinear_mip`/`anisotropic`; `bilinear_mip` takes one bilinear tap from the mip level nearest the warp's compression — use it when cross-scale views alias under `bilinear` but the anisotropic cost is not warranted) |
| `robust_iters`         | `3`             | `refine_keypoints` (IRLS passes for the consensus) |
| `max_outer_sweeps`     | `1`             | `refine_keypoints` (`1` = single-pass frozen consensus; `>1` refreshes per sweep) |
| `outer_convergence_px` | `0.005`         | `refine_keypoints` (outer-loop stop, patch-grid px; ignored at 1 sweep) |
| `max_gn_steps`         | `10`            | `refine_keypoints` (Gauss–Newton steps per view per sweep) |
| `convergence_px`       | `0.01`          | `refine_keypoints` (per-view stop, patch-grid px) |
| `max_offset_px`        | `2.0`           | `refine_keypoints` (max per-view drift from the seed, patch-grid px) |
| `consensus_refresh`    | `per_sweep`     | `refine_keypoints` (`per_sweep`/`per_move`)    |
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
binding additionally fuses each point's RGBA representative texture at the
**final** refined keypoints and the command scatters them into a
`(point_count, R, R, 4)` uint8 array (zero rows where the point produced no
valid cross-view consensus) attached via
`clone_with_changes(patches=cloud, patch_bitmaps=…)`; the stored frame is
re-persisted alongside so the bitmaps have a frame to attach to (the frame
itself is unchanged — keypoints moved, not the surfel). On by default so the
refined reconstruction carries its per-point patch textures and can display them
without re-rendering; it costs one extra full-grid source render per view per
point, so a multi-stage pipeline can pass `bitmaps=false` on intermediate stages
and render once on the finalizing stage.

The transform prints a one-line summary in the established `xform` style over
the finitely-scored views (a point with fewer than two views has no consensus;
its views carry NaN scores and keep their seed):

```
  Refined N keypoints (mean |offset| 0.142 patch-grid px)
```

## Ordering and interactions

- **Invariant to global similarity.** `--rotate` / `--translate` / `--scale`
  move points and poses together, so the photometric consensus is unchanged;
  ordering relative to those is immaterial.
- **Repeatable.** A second pass starts from the first's stored keypoints and
  can move them further (within `max_offset_px` of the *new* seed).
- **Images must be resolvable.** Fails fast if `workspace_dir` or any image is
  missing, exactly like `--refine-normals` and the `.sift`-reading ops.

## Performance and memory

Same envelope as `--refine-normals`: the binding loads **all** full-resolution
images (plus pyramids) into memory at once and releases the GIL during the
solve, parallelizing across points. Work scales with observations × GN steps ×
sweeps; there is no streaming of the image set.

## Future

A `--localize-keypoints` op could later surface
`PatchCloud.localize_keypoints` — basin *search* that can drop views, a
heavier, structural (filter-like) operation — as the upstream step that puts
seeds in the basin; this op is subpixel refinement only.

> _Status (2026-07-05): Done —
> [xform-localize-keypoints-command.md](xform-localize-keypoints-command.md)._
