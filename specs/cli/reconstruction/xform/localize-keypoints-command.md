# `sfm xform --localize-keypoints` Design

The `sfm xform --localize-keypoints` operation moves the 2D keypoints of an
`embedded_patches` reconstruction, which stores a patch for each 3D point (a
small oriented square of the surface around it). For each point it starts from
the point's projection into every image that observes it, searches for the shift
at which that image's view of the patch best agrees with the point's other
views, and drops the views it cannot register this way. It then **rebuilds** the
reconstruction from the views that remain, removing points left with too few of
them. It is the search that places each keypoint near enough to the best match
for `--refine-keypoints` to refine it locally, and a way to remove the
observations that do not register with the rest of their track. The search
itself is specified in
[patch-keypoint-localization.md](../../../core/patch/patch-keypoint-localization.md).

## What it does

Localizes each observation's 2D keypoint by group-wise translation registration
(**congealing**): each round renders every view's patch tile at its accumulated
in-plane offset, builds the robust cross-view consensus, and searches each
view's residual shift against the **leave-one-out** consensus of the others.
Views that drift too far, leave the frame, graze the patch plane
(`min_grazing_cos`), pin no 2D position of their own
(`max_member_zncc_self_similarity_radius`, 2.5 by default), or stop agreeing
— absolutely (`min_absolute_zncc`) or relative to their peers (`min_relative_zncc`) — are
**dropped**.
Seeds are each point's own projection (`project_i(X_p)`); the search basin is
`±search` patch-grid px around it.

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
  patch *frames* but no bitmaps. Re-run
  `sfm xform --refine-keypoints bitmaps=true` (or
  `--refine-normals bitmaps=true`) to regenerate them (a frames-without-bitmaps
  `embedded_patches` recon is valid — see `specs/gui/patch-rendering.md`).
  There is no `bitmaps` key on this op.

The write-back is `compact_to_embedded_patches`
(`src/sfmtool/_patch_compaction.py`) — the **same helper the `embed-patches`
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

The operation is `LocalizeKeypointsTransform` in
[_localize_keypoints.py](../../../../src/sfmtool/xform/_localize_keypoints.py),
wired through `parse_localize_keypoints_params` in
[_arg_parser.py](../../../../src/sfmtool/xform/_arg_parser.py) and the
`--localize-keypoints` Click option in
[xform.py](../../../../src/sfmtool/_commands/xform.py); the localization itself
is the `PatchCloud.localize_keypoints` PyO3 binding.

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
| `max_iters`                    | `5`             | `localize_keypoints` (max congealing rounds)   |
| `search`                       | `6.0`           | `localize_keypoints` (max total per-view drift, patch-grid px) |
| `max_shift_px`                 | `3.0`           | `localize_keypoints` (drop a view whose keypoint sits further than this from the point's projection, source-image px) |
| `min_relative_zncc`            | `0.7`           | `localize_keypoints` (drop a view whose leave-one-out ZNCC falls below this fraction of the median — the one gate the two-view floor can undo) |
| `min_absolute_zncc`            | `0.5`           | `localize_keypoints` (drop a view whose leave-one-out ZNCC is finite and below this absolute floor, whatever the view count; `0` disables) |
| `max_member_zncc_self_similarity_radius` | `2.5` | `localize_keypoints` (drop a view whose own tile's ZNCC self-similarity radius is above this, patch-grid px — see [`specs/core/patch/patch-keypoint-localization.md`](../../../core/patch/patch-keypoint-localization.md#the-member-self-similarity-gate); `0` disables, `3` or more turns nothing out) |
| `min_grazing_cos`              | `0.1`           | `localize_keypoints` (drop a view whose ray grazes the patch plane) |
| `resolution`                   | `24`            | `localize_keypoints` (R×R patch grid)          |
| `window`                       | `gaussian_disk` | `localize_keypoints` (`gaussian_disk`/`gaussian`/`uniform`) |
| `window_sigma`                 | `0.6`           | `localize_keypoints`                           |
| `sampler`                      | `per_view`      | `localize_keypoints` (`per_view`/`bilinear`/`bilinear_mip`/`anisotropic`; `per_view` picks `anisotropic` or `bilinear_mip` for each view by the rule in [image-warping.md](../../../core/camera/image-warping.md#choosing-the-sampler-per-view), which also gives each sampler's cost) |
| `robust_iters`                 | `3`             | `localize_keypoints` (IRLS passes for the consensus) |
| `convergence_px`               | `0.05`          | `localize_keypoints` (round-level stop, patch-grid px) |
| `search_resolution_multiplier` | `1.0`           | `localize_keypoints` (supersampled search grid; `> 1` resolves sub-pixel offsets at ~m² cost) |
| `search_strategy`              | `plus_descent`  | `localize_keypoints` (`plus_descent`/`exhaustive`) |
| `basis_max_views`              | `8`             | `localize_keypoints` (consensus-basis cap `K`; `0` congeals all views — the cleanest error metrics. A track of `≤ K` views takes the uncapped path either way. This surface runs over each point's own track and supplies no per-view appearance scores, so a biting cap ranks the basis by grazing angle — see [`specs/core/patch/keypoint-localization-consensus-basis.md`](../../../core/patch/keypoint-localization-consensus-basis.md)) |

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
