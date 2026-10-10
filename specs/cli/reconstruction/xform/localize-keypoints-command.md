# `sfm xform --localize-keypoints` Design

The `sfm xform --localize-keypoints` operation moves the 2D keypoints of an
`embedded_patches` reconstruction, which stores a patch for each 3D point (a
small oriented square of the surface around it). For each point it starts from
each observation's stored keypoint, searches for the shift at which that
image's view of the patch best matches the point's reference render (the tile
of its reference observation), and drops the views it cannot align this way. It then **rebuilds** the
reconstruction from the views that remain, removing points left with too few of
them. It is the search that places each keypoint near enough to the best match
for `--refine-keypoints` to refine it locally, and a way to remove the
observations that do not register with the rest of their track. The search
itself is specified in
[patch-keypoint-localization.md](../../../core/patch/patch-keypoint-localization.md).

## What it does

Localizes each observation's 2D keypoint by aligning it, in one pass, to the
point's **reference render**: the `R×R` render of the point's reference
observation at its stored keypoint. The reference observation is the one the
point stores in `tracks/reference_observations`, and where it stores none, the
reference-view rule's pick from the views' renders at their stored keypoints
(where the rule picks none it would store, the template is the views' fused
mean and no view is the reference). The reference's keypoint is not moved.
Every other view renders one tile around its stored keypoint and searches the
shift, within `±search` patch-grid px, at which it best matches the reference
render. Views that drift too far, leave the frame, graze the patch plane
(`min_grazing_cos`), pin no 2D position of their own
(`max_member_zncc_self_similarity_radius`, 2.5 by default), or match the
reference render too poorly — absolutely (`min_absolute_zncc`) or relative to
the other views (`min_relative_zncc`) — are **dropped**.

It is therefore a **structural** operation — the search counterpart of the
in-place `--refine-keypoints`, with a fundamentally different shape:

- **Views are dropped.** Only the views the localizer keeps appear in the
  output; the observation count can only shrink.
- **Points can be dropped.** After localization, a point whose kept-view count
  falls below `min_views` (default 2) is culled entirely; surviving points are
  renumbered densely (ascending source order). Positions, colors, errors and
  patch frames are carried over per survivor; each finite survivor's normal is
  re-derived from its patch frame (`normalize(u × v)`), and a point at
  infinity keeps its stored row.
- **The track structure is rebuilt.** `keypoints_xy`,
  `track_image_indexes` / `track_point_indexes` / `observation_counts` are all
  reconstructed from the kept views — nothing structural from the input is
  reused. Cameras, poses, and each surviving point's 3D geometry are unchanged
  (points at infinity stay at infinity).
- **Bitmaps are dropped.** The localizer renders no bitmaps, and any stored
  ones are stale once keypoints move and views drop, so the output carries
  patch *frames* but no bitmaps. With no bitmap written there is nothing for
  the reference to agree with, so each point keeps its stored reference in
  `tracks/reference_observations`, moved to the observation of that image in
  its rebuilt track, wherever that image is still in the track -- including a
  stored reference whose tile does not render at its keypoint, to which
  nothing was aligned. A point that stores none, or whose stored reference was
  dropped (a grazing one is dropped with the other grazing views), records
  the reference its views were aligned to, the rule's pick, or `-1` where they
  were aligned to the fused mean. A later render renders the point from the
  reference it then stores. Re-run
  `sfm xform --refine-keypoints bitmaps=true` (or
  `--refine-normals bitmaps=true`) to regenerate them (a frames-without-bitmaps
  `embedded_patches` recon is valid — see `specs/gui/patch-rendering.md`).
  There is no `bitmaps` key on this op.

The output is built by the same compaction step the `embed-patches` pipeline
uses after its own localization, so both produce `embedded_patches` files by
the same rules: the cull, the dense renumbering, the track rebuild and the
culled patch frames. Each stored keypoint is clamped to lie inside its image
after rounding to `float32`, because a keypoint just inside the image edge in
`f64` can round up to exactly the width or height, which the writer rejects.
The output keeps the input's image file hashes. If **no** point survives the
cull, the operation fails with a CLI error rather than writing an empty
reconstruction.

Unlike the `embed-patches` pipeline's localize step, this op localizes over
each point's **full track**: it does no view pre-selection, so the only views
dropped are the ones the localizer drops during its search.

**Precondition:** requires an `embedded_patches` reconstruction and rejects
`sift_files` with a `UsageError` pointing at `sfm xform --to-embedded-patches`.
The check is made against the reconstruction as it stands when this step runs,
so a `--to-embedded-patches --localize-keypoints` chain converts first and
passes. The localizer searches over the patch frame stored for each point,
which only that source carries; the frame is never rebuilt.

Because the search is photometric it reads the workspace source images
(`workspace_dir / image_name`), exactly like `--refine-keypoints` /
`--refine-normals`; a missing image (or unresolvable workspace) is a hard error
(`FileNotFoundError`).

## Command syntax

The operation is `LocalizeKeypointsTransform` in
[_localize_keypoints.py](../../../../src/sfmtool/xform/_localize_keypoints.py),
wired through `parse_localize_keypoints_params` in
[_arg_parser.py](../../../../src/sfmtool/xform/_arg_parser.py) and the
`--localize-keypoints` Click option in
[xform.py](../../../../src/sfmtool/_commands/xform.py); the localization itself
is the `PatchCloud.localize_keypoints` PyO3 binding, and the write-back is
`compact_to_embedded_patches` in
[_patch_compaction.py](../../../../src/sfmtool/_patch_compaction.py), which
the `embed-patches` pipeline also calls.

```
sfm xform <input.sfmr> [<output.sfmr>] --localize-keypoints [<params>] [...]
```

`--localize-keypoints` takes an **optional** comma-separated parameter string
of `key=value` modifiers. `parse_xform_args` reads the value, so all three
forms work: bare `--localize-keypoints`, space-separated
`--localize-keypoints search=8`, and joined `--localize-keypoints=search=8`; a
following option, e.g. `--localize-keypoints --refine-keypoints`, is left
untouched. The Click option is declared `is_flag=False, flag_value=""`, and
the values Click collects are checked against that walk. With no value it runs
the binding defaults plus `min_views=2`.

Because the next token is taken as the value whenever it is not an option, the
output path must come before the operation: in
`sfm xform in.sfmr --localize-keypoints out.sfmr`, `out.sfmr` is read as the
parameter string and rejected as a malformed `key=value` token.

```
--localize-keypoints
--localize-keypoints search=8,min_views=3
--to-embedded-patches --localize-keypoints --refine-keypoints
```

### `key=value` modifiers

All keys except `min_views` are `PatchCloud.localize_keypoints` keyword
arguments, and only the keys given are passed to it, so a key left out takes
the binding's own default. The "Default" column lists those binding defaults.
`min_views` is the compaction cull threshold consumed by
`compact_to_embedded_patches`.

| Key                            | Default         | Forwards to                                    |
|--------------------------------|-----------------|------------------------------------------------|
| `min_views`                    | `2`             | `compact_to_embedded_patches` (drop a point with fewer kept views; `>= 1`) |
| `search`                       | `6.0`           | `localize_keypoints` (reach of each view's search around its stored keypoint, patch-grid px) |
| `max_shift_px`                 | `3.0`           | `localize_keypoints` (drop a view whose keypoint sits further than this from the point's projection, source-image px) |
| `min_relative_zncc`            | `0.7`           | `localize_keypoints` (drop a view whose ZNCC against the reference render falls below this fraction of the median over the views other than the reference; `0` disables) |
| `min_absolute_zncc`            | `0.5`           | `localize_keypoints` (drop a view whose ZNCC against the reference render is finite and below this absolute floor, whatever the view count; `0` disables) |
| `max_member_zncc_self_similarity_radius` | `2.5` | `localize_keypoints` (drop a view whose own tile's ZNCC self-similarity radius is above this, patch-grid px — see [`specs/core/patch/patch-keypoint-localization.md`](../../../core/patch/patch-keypoint-localization.md#the-member-self-similarity-gate); `0` disables, `3` or more turns nothing out) |
| `min_grazing_cos`              | `0.1`           | `localize_keypoints` (drop a view whose ray grazes the patch plane) |
| `resolution`                   | `24`            | `localize_keypoints` (R×R patch grid)          |
| `window`                       | `gaussian_disk` | `localize_keypoints` (`gaussian_disk`/`gaussian`/`uniform`) |
| `window_sigma`                 | `0.6`           | `localize_keypoints`                           |
| `sampler`                      | `per_view`      | `localize_keypoints` (`per_view`/`bilinear`/`bilinear_mip`/`anisotropic`; `per_view` picks `anisotropic` or `bilinear_mip` for each view by the rule in [image-warping.md](../../../core/camera/image-warping.md#choosing-the-sampler-per-view), which also gives each sampler's cost) |
| `robust_iters`                 | `3`             | `localize_keypoints` (IRLS passes for the fused mean, where it is the template) |
| `search_strategy`              | `exhaustive`    | `localize_keypoints` (`exhaustive`/`plus_descent`) |

Unknown keys, malformed `key=value` tokens (no `=`, empty key), duplicate keys,
or unparseable values raise `click.UsageError`, and so does a value out of its
key's range or not among its key's choices.

The transform prints a structural summary in the established `xform` style:

```
  Localized keypoints: 1204 -> 1130 points, 5820 -> 5233 observations
  Kept 4.6 views per surviving point (mean)
```

## Ordering and interactions

- **Pairs with `--refine-keypoints`.** The localizer is the discrete *search*
  that moves each view's keypoint to the shift where its tile best matches the
  other views (and drops the views it cannot place); the refiner is the *local*
  sub-pixel solve, which needs a starting keypoint already near that shift.
  `--localize-keypoints --refine-keypoints` (search, then sub-pixel
  refinement) is the chain the `embed-patches` pipeline itself runs.
  Either op is also useful alone.
- **`--refine-normals` is an option on either side.** Refining normals first
  gives the localizer a better patch plane to search over; localizing first
  gives the normal refiner cleaner view sets. Both orderings are legitimate —
  pick per dataset rather than by rule.
- **Invariant to global similarity.** `--rotate` / `--translate` / `--scale`
  move points and poses together, so the renders the views are aligned to are
  unchanged;
  ordering relative to those is immaterial.
- **Repeatable, not idempotent.** A second pass starts from the keypoints the
  first one wrote and aligns them to the reference observation the first pass
  recorded, over the already-culled track; it can move views again and drop
  further ones.
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
observations × the search area (`search²`), with each view searched once;
`plus_descent` scores far fewer shifts than `exhaustive` (the default). There is no streaming of the image set.
