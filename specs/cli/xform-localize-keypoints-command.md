# `sfm xform --localize-keypoints` Design

**Status:** Implemented (2026-07-05) in `src/sfmtool/xform/_localize_keypoints.py`
(`LocalizeKeypointsTransform`), wired through `xform/_arg_parser.py`
(`parse_localize_keypoints_params`) and the `--localize-keypoints` Click option
in `_commands/xform.py`. Surfaces the discrete cross-view keypoint localization
(`specs/core/patch-keypoint-localization.md`, implemented in the
`PatchCloud.localize_keypoints` PyO3 binding) as an `sfm xform` operation that
**rebuilds** a reconstruction from the views that co-register.

## What it does

Localizes each observation's 2D keypoint by group-wise translation registration
(**congealing**): each round renders every view's patch tile at its accumulated
in-plane offset, builds the robust cross-view consensus, and searches each
view's residual shift against the **leave-one-out** consensus of the others.
Views that drift too far, leave the frame, graze the patch plane
(`min_grazing_cos`), or stop agreeing (`min_relative_zncc`) are **dropped**.
Seeds are each point's own projection (`project_i(X_p)`); the search basin is
`±search` patch-grid px around it.

It is therefore a **structural** operation — the search counterpart of the
in-place `--refine-keypoints`, with a fundamentally different shape:

- **Views are dropped.** Only the views the localizer keeps appear in the
  output; the observation count can only shrink.
- **Points can be dropped.** After localization, a point whose kept-view count
  falls below `min_views` (default 2) is culled entirely; surviving points are
  renumbered densely (ascending source order), and positions, colors, errors,
  normals, and patch frames are carried over per survivor.
- **The track structure is rebuilt.** `keypoints_xy`,
  `track_image_indexes` / `track_point_indexes` / `observation_counts` are all
  reconstructed from the kept views — nothing structural from the input is
  reused. Cameras, poses, and each surviving point's 3D geometry are unchanged
  (points at infinity stay at infinity).
- **Bitmaps are dropped.** The localizer renders no bitmaps, and any stored
  ones are stale once keypoints move and views drop, so the output carries
  patch *frames* but no bitmaps. Re-run
  `sfm xform --refine-keypoints bitmaps=true` (or
  `--refine-normals bitmaps=true`) to regenerate them (a frames-without-bitmaps
  `embedded_patches` recon is valid — see `specs/gui/gui-patch-rendering.md`).
  There is no `bitmaps` key on this op.

The write-back is `compact_to_embedded_patches`
(`src/sfmtool/_embed_patches.py`) — the **same helper the `embed-patches`
pipeline uses** to turn localizer output into a valid `embedded_patches`
reconstruction (survivor selection, dense renumbering, track rebuild, per-image
in-frame f32 keypoint clamp, culled patch frames). The op hand-rolls none of
it; `image_file_hashes` come from the recon itself (recomputed from the source
images only if absent). If **no** point survives the cull, the operation fails
with a `ValueError` (surfaced as a CLI error) rather than writing an empty
reconstruction.

Unlike the `embed-patches` pipeline's localize step, this op passes
`view_sets=None`: it localizes over each point's **full track**, exposing only
the localizer's own in-loop view dropping (no `select_views` pre-selection).

**Precondition:** requires an `embedded_patches` reconstruction and rejects
`sift_files` with a `UsageError` pointing at `sfm xform --to-embedded-patches`
(enforced per-step in `xform/_apply.py` via the
`LocalizeKeypointsTransform.required_feature_source` attribute, so a
`--to-embedded-patches --localize-keypoints` chain converts first and passes).
The localizer searches over the stored per-point patch frame
(`recon.patches`), which only that source carries; the frame is never rebuilt.

Because the search is photometric it reads the workspace source images
(`workspace_dir / image_name`), exactly like `--refine-keypoints` /
`--refine-normals`; a missing image (or unresolvable workspace) is a hard error
(`FileNotFoundError`).

## Command syntax

```
sfm xform <input.sfmr> [<output.sfmr>] --localize-keypoints [<params>] [...]
```

`--localize-keypoints` takes an **optional** comma-separated parameter string
of `key=value` modifiers (the Click option is `is_flag=False, flag_value=""`,
so all three forms work: bare `--localize-keypoints`, space-separated
`--localize-keypoints search=8`, and joined `--localize-keypoints=search=8` —
while a following option, e.g. `--localize-keypoints --refine-keypoints`, is
left untouched). With no value it runs the binding defaults plus `min_views=2`.

```
--localize-keypoints
--localize-keypoints search=8,min_views=3
--to-embedded-patches --localize-keypoints --refine-keypoints
```

### `key=value` modifiers

All keys except `min_views` pass straight through to
`PatchCloud.localize_keypoints`, reusing the binding's own defaults — the
"Default" column matches each binding default exactly, so the CLI re-specifies
nothing and the two layers cannot drift. `min_views` is the compaction cull
threshold consumed by `compact_to_embedded_patches`.

| Key                            | Default         | Forwards to                                    |
|--------------------------------|-----------------|------------------------------------------------|
| `min_views`                    | `2`             | `compact_to_embedded_patches` (drop a point with fewer kept views; `>= 1`) |
| `max_iters`                    | `5`             | `localize_keypoints` (max congealing rounds)   |
| `search`                       | `6.0`           | `localize_keypoints` (max total per-view drift, patch-grid px) |
| `max_shift_px`                 | `3.0`           | `localize_keypoints` (drop a view whose keypoint sits further than this from the point's projection, source-image px) |
| `min_relative_zncc`            | `0.7`           | `localize_keypoints` (drop a view whose leave-one-out ZNCC falls below this fraction of the median) |
| `min_grazing_cos`              | `0.1`           | `localize_keypoints` (drop a view whose ray grazes the patch plane) |
| `resolution`                   | `24`            | `localize_keypoints` (R×R patch grid)          |
| `window`                       | `gaussian_disk` | `localize_keypoints` (`gaussian_disk`/`gaussian`/`uniform`) |
| `window_sigma`                 | `0.6`           | `localize_keypoints`                           |
| `sampler`                      | `bilinear`      | `localize_keypoints` (`bilinear`/`anisotropic`) |
| `robust_iters`                 | `3`             | `localize_keypoints` (IRLS passes for the consensus) |
| `convergence_px`               | `0.05`          | `localize_keypoints` (round-level stop, patch-grid px) |
| `search_resolution_multiplier` | `1.0`           | `localize_keypoints` (supersampled search grid; `> 1` resolves sub-pixel offsets at ~m² cost) |
| `search_strategy`              | `plus_descent`  | `localize_keypoints` (`plus_descent`/`exhaustive`) |

Unknown keys, malformed `key=value` tokens (no `=`, empty key), duplicate keys,
or unparseable values raise `click.UsageError`; range/enum validation lives in
the `LocalizeKeypointsTransform` constructor (surfacing as `UsageError` through
the CLI), consistent with the other parsers in `xform/_arg_parser.py`.

The transform prints a structural summary in the established `xform` style:

```
  Localized keypoints: 1204 -> 1130 points, 5820 -> 5233 observations
  Kept 4.6 views per surviving point (mean)
```

## Ordering and interactions

- **Pairs naturally with `--refine-keypoints`.** The localizer is the discrete
  *search* that puts each view in the right photometric basin (and drops the
  ones that have none); the refiner is the *local* sub-pixel solve that needs a
  seed already near the optimum. `--localize-keypoints --refine-keypoints`
  (search, then sharpen) is the chain the `embed-patches` pipeline itself runs.
  Either op is also useful alone.
- **`--refine-normals` is an option on either side.** Refining normals first
  gives the localizer a better patch plane to search over; localizing first
  gives the normal refiner cleaner view sets. Both orderings are legitimate —
  pick per dataset rather than by rule.
- **Invariant to global similarity.** `--rotate` / `--translate` / `--scale`
  move points and poses together, so the photometric consensus is unchanged;
  ordering relative to those is immaterial.
- **Repeatable, not idempotent.** A second pass re-seeds at each point's
  projection and re-runs the search over the already-culled track; it can drop
  further views.
- **Downstream ops see the culled structure.** Track-based filters
  (`--remove-short-tracks`, `--remove-narrow-tracks`) and `--bundle-adjust`
  after this op operate on the rebuilt, smaller track set — often exactly what
  is wanted (BA on the co-registering observations only).
- **Images must be resolvable.** Fails fast if `workspace_dir` or any image is
  missing, exactly like the other photometric ops.

## Performance and memory

Same envelope as `--refine-keypoints` / `--refine-normals`: the binding loads
**all** full-resolution images (plus pyramids) into memory at once and releases
the GIL during the search, parallelizing across points. Work scales with
observations × rounds × the search area (`search²`, × `m²` under
`search_resolution_multiplier`); `plus_descent` (default) prunes the search
against `exhaustive`. There is no streaming of the image set.
