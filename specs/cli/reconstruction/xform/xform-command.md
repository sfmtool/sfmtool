# `sfm xform` Command Design

## Overview

The `sfm xform` command provides a unified interface for applying transformations and
filters to SfM reconstructions. Multiple operations can be chained in a single pass,
applied sequentially in the order specified on the command line.

Densification is a separate, experimental command today (`sfm densify`); folding it into
`xform` so it can be pipelined with filtering and bundle adjustment is a possible future
direction (see [densify-command.md](../densify-command.md)).

## Command Syntax

```bash
sfm xform <input.sfmr> [<output.sfmr>] [OPTIONS...]
```

If `<output.sfmr>` is omitted, the result is written next to the input as
`<stem>-transformed.sfmr`. If that name is already taken, a numeric suffix is
appended starting at 2 (`<stem>-transformed-2.sfmr`, `-3`, ...).

### Order Matters

Transformations are applied sequentially. The output of each becomes the input to the next.

```bash
# Rotate, then translate
sfm xform in.sfmr out.sfmr --rotate 0,1,0,+25deg --translate 3,5,-2

# Translate, then rotate (different result!)
sfm xform in.sfmr out.sfmr --translate 3,5,-2 --rotate 0,1,0,+25deg
```

## Available Operations

### Geometric Transformations

#### `--rotate <axisX>,<axisY>,<axisZ>,<angle>`

Rotates all 3D points and camera poses about the origin. Axis is normalized internally.
Angle requires a unit suffix: `deg`/`degrees` or `rad`/`radians`.

```bash
--rotate 0,1,0,90deg
--rotate 1,1,0,45deg
--rotate 0,0,1,1.5708rad
```

#### `--translate <X>,<Y>,<Z>`

Translates all 3D points and camera centers. Camera orientations unchanged.

```bash
--translate 3,5,-2
```

#### `--scale <S>`

Scales all 3D points and camera positions relative to the origin. Must be positive.

```bash
--scale 2.0
```

### Filtering Operations

#### `--include-range <RANGE_EXPR>` / `--exclude-range <RANGE_EXPR>`

Filters images by file number. Supports single numbers, ranges, and comma-separated
combinations. Also removes 3D points with no remaining observations and remaps indices.
Rig metadata is pruned alongside the images: any rig frame whose images are all removed
is dropped, and partially-kept frames are retained with the surviving sensors.

```bash
--include-range 10-50
--exclude-range "23-25,47"
```

#### `--include-glob <PATTERN>` / `--exclude-glob <PATTERN>`

Filters images by filename pattern, matched against the workspace-relative image name
(e.g. `subdir/fisheye_left/0001.jpg`) using Python's `fnmatch.fnmatch`. Note that
`fnmatch` is *not* shell globbing: `*` matches across `/` boundaries. As with range
filters, points with no remaining observations are removed and rig metadata is pruned;
an error is raised if the filter would keep zero images.

```bash
--include-glob "*fisheye_left*"
--exclude-glob "*/fisheye_right/*"
```

#### `--include-by-distribution <COUNT>[,verbose]`

Keeps a small, strategically-chosen subset of `COUNT` views instead of decimating blindly. A single
greedy loop drives the choice off the reconstructed point cloud: a farthest-point step always heads
to the part of the cloud that is currently worst-covered, and at each such target it adds only the
observers that open up genuinely new viewing angles on it (so covered points end up well
triangulated and the subset solves cleanly on its own). The unit of selection is the rig frame when
rig data is present (both fisheyes of a 360° pair kept together), otherwise the individual image.
Operates on the cloud as given — clean it first if needed (e.g. `--filter-by-reprojection-error`,
`--remove-isolated`) — and typically follow with `--remove-short-tracks 1` to drop stranded
single-observation points.

Append `,verbose` for a per-step trace of the selection loop.

```bash
--include-by-distribution 16
--include-by-distribution 16,verbose
--filter-by-reprojection-error 2.0 --include-by-distribution 16 --remove-short-tracks 1
```

See [select-by-distribution-command.md](select-by-distribution-command.md) for the full
specification — the farthest-point coverage loop, angular thinning of observers, the incremental
algorithm, and parameters.

#### `--remove-isolated <factor>,<value_spec>`

Removes 3D points whose nearest neighbor distance exceeds `factor * reference_value`.
The reference value is `median` or `<N>percent`/`<N>percentile`.

```bash
--remove-isolated 3.0,median
--remove-isolated 2.0,95percent
```

#### `--remove-short-tracks <size>`

Removes 3D points with track length (observation count) <= size.

```bash
--remove-short-tracks 2
```

#### `--remove-narrow-tracks <angle>`

Removes 3D points whose maximum viewing angle span is less than the threshold.

```bash
--remove-narrow-tracks 5deg
```

#### `--remove-large-features <size>`

Removes 3D points whose largest feature in the track exceeds `size` pixels.
"Feature size" is the average of the two column norms of the feature's affine-shape
matrix (an approximate per-feature radius in source-image pixels, with no rescale to
the reconstructed-image resolution), and the per-track value is the maximum across its
observations.

The feature source determines where the affine shape comes from:

- **`sift_files`** — reads the original `.sift` files associated with the
  reconstruction's workspace (resolved as
  `<workspace_dir>/<image_parent>/<feature_prefix_dir>/<name>.sift`), so the SIFT
  artifacts must still be present where the reconstruction was created. A missing
  file raises `FileNotFoundError`.
- **`embedded_patches`** — has no external `.sift` companion (and no per-observation
  feature-index column), so each observation's shape is derived by projecting the
  point's stored patch frame into the observing camera
  (`observation_affine_shape` / `max_embedded_feature_size_per_point`) — the same
  size measure the Track View reports. No workspace `.sift` files are needed.

```bash
--remove-large-features 50
```

#### `--filter-by-reprojection-error <threshold>`

Removes 3D points with reprojection error exceeding the threshold (in pixels).
Points at infinity (`w = 0`) are scored too: a point at infinity still projects
its bearing through rotation + intrinsics, so its reprojection error is
well-defined and a high-error point at infinity is removed like any other.

```bash
--filter-by-reprojection-error 2.0
```

#### `--filter-by-keypoint-uncertainty <threshold>`

Removes 3D points whose predicted keypoint position uncertainty `σ_pos` exceeds the
threshold. `σ_pos` is the noise-normalized weak-axis structure-tensor uncertainty of
the point's cross-view consensus patch — see
[patch-localizability.md](../../../core/patch/patch-localizability.md) — and it grades
how well *conditioned* a keypoint's localization is (corner vs. edge vs. flat),
independently of whether the views agree. That is the aperture blind spot the
reprojection and agreement gates cannot see: an edge-like patch slides freely along
the edge while every view keeps agreeing.

The threshold is in **patch-grid px**, the intrinsic resolution-independent unit, not
source-image px. Grid px transfers across datasets of different resolution; the
source-px form folds in a per-point focal/depth scale, so one fixed threshold would
cull wildly different fractions of two clouds. A point that cannot be scored at all
(empty consensus, `σ_pos` = NaN) is **kept** — the filter only removes points it has
positive evidence about. `threshold` must be positive.

This requires an `embedded_patches` reconstruction *with* per-point patch bitmaps: the
consensus is what gets scored, and it is computed on demand from the stored consensus
and geometry, reading no source images. A reconstruction missing either the bitmaps or
the patch frames is rejected with a message naming `sfm embed-patches` or
`sfm xform --refine-keypoints bitmaps=true`.

```bash
--filter-by-keypoint-uncertainty 0.35
```

#### `--filter-by-patch-size <multiplier>`

Removes 3D points whose world-space patch size exceeds `multiplier` times the
median patch size across the reconstruction. A patch's characteristic world size
is the geometric mean of its two world half-extents,
`sqrt(|half_extent[0]| * |half_extent[1]|)`, so under the `feature_size` extent
policy it tracks the keypoint's SIFT scale and the largest patches are the
coarsest features. The keep threshold is data-derived per reconstruction —
`size <= multiplier * median(size)` — so it adapts to each cloud's own scale
rather than fixing an absolute world size. `multiplier` must be positive;
a non-positive value is rejected.

This requires an `embedded_patches` reconstruction, since it reads the per-point
patch frames that carry the world half-extents. A `sift_files` reconstruction
(which has no patch frames) is rejected with a message directing the user to
convert first with `sfm embed-patches` or `--to-embedded-patches`.

```bash
--filter-by-patch-size 3.0
```

### Points at Infinity

`--find-points-at-infinity` is *additive*: it appends new points and tracks, so
the point count grows. See
[find-points-at-infinity.md](find-points-at-infinity.md)
for the design.

#### `--find-points-at-infinity <eps_deg>[,<desc_thresh>[,<min_views>[,<noise_floor_px>]]]`

Discovers points at infinity (and near-infinite distant points) by clustering
keypoint world-space directions across all images within `eps_deg` on the unit
sphere, confirming each cluster with mutual SIFT descriptor matching, and
emitting each surviving track as a `w = 0` point or a finite distant point.
`desc_thresh` is the maximum L2 SIFT descriptor distance (default `200`) and
`min_views` is the minimum number of distinct images a track must span
(default `2`, must be `>= 2`); a tighter `eps_deg` demands more nearly parallel
rays (more "infinite"), and a tighter `desc_thresh` rejects weak matches (raise
it to recover more, lower it to suppress coincidental matches on feature-dense
or self-similar scenes). `noise_floor_px` (default `1.0`) is the assumed
keypoint-localisation noise the finite-vs-infinity classifier converts to
per-ray angular noise. This reads the original `.sift` files from the
reconstruction's workspace, so the SIFT artifacts must still be present.

```bash
--find-points-at-infinity 0.1
--find-points-at-infinity 0.1,200,2
--find-points-at-infinity 0.1,200,2,1.5
```

#### `--classify-points-at-infinity <noise_floor_px>`

Reclassifies *existing* finite points whose depth is unconstrained as points at
infinity (`w = 0`). A finite point is reclassified when its parallax signal
falls below the track's measurement noise (never taken below `noise_floor_px`).
It finds no new points and leaves the point count unchanged.

```bash
--classify-points-at-infinity 1.0
```

#### `--max-features <N>`

Caps the per-image keypoint set used by `--find-points-at-infinity` to the `N`
largest features (by scale). Bounds memory and runtime on dense or many-image
solves; the largest-scale features tend to be the most repeatable across the
wide viewpoint changes a distant point is seen under. This is a single global
value (not an ordered transform), shared by every `--find-points-at-infinity`
in the chain. Passing `--max-features` without any `--find-points-at-infinity`
operation is rejected with a `UsageError` rather than silently ignored.

```bash
sfm xform in.sfmr out.sfmr --find-points-at-infinity 0.1,200,2 --max-features 2000
```

### Camera Model

#### `--camera-model <NAME>`

Converts every camera in the reconstruction to a different COLMAP camera model. The
target name is case-insensitive and must be one of the models registered in
`_CAMERA_PARAM_NAMES` (e.g. `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`,
`OPENCV`, `OPENCV_FISHEYE`, ...). Each camera is converted independently, so a
reconstruction with mixed source models is fine; image width and height are preserved.
Parameter handling:

- Parameters whose names are identical in source and target are carried over as-is.
- Parameters that exist only in the target model are initialized to zero.
- Parameters that exist only in the source model are dropped.
- When the focal-length representation differs: a single `focal_length` becomes split
  `focal_length_x = focal_length_y` and vice versa; collapsing split → single averages
  `fx` and `fy` and prints a warning if they differ by more than a small relative tolerance.
- If every camera already uses the target model, all parameter names match and values
  are carried over unchanged (the operation is effectively a no-op, but is still logged).

The typical use is to widen the parameter set right before bundle adjustment — e.g.
upgrading `SIMPLE_RADIAL` to `RADIAL` so bundle adjustment has a `k2` term to refine.

```bash
--camera-model RADIAL
```

### Optimization

#### `--bundle-adjust`

Applies bundle adjustment via pycolmap to refine camera poses and 3D point positions.

```bash
--remove-short-tracks 2 --bundle-adjust
```

Works on both `sift_files` and `embedded_patches` reconstructions. The transform
round-trips through COLMAP binary files, which need a 2D keypoint per
observation: for a `sift_files` recon these are read from the workspace `.sift`
files (indexed by each observation's feature index); for an `embedded_patches`
recon they are taken from the inline `keypoints_xy` (no `.sift` companion is
required). An `embedded_patches` input is refined and written back in
`embedded_patches` mode, with the per-observation keypoints and the per-point
patch frames carried through unchanged — note that the patch `(u, v)` frames are
*not* re-fit to the adjusted geometry, so re-run `sfm embed-patches` (or
`--refine-normals`) afterward if the poses/points moved materially.

> **TODO (implementation cleanup):** the "where does an observation's 2D
> keypoint come from" distinction (`.sift` file vs inline `keypoints_xy`) is
> currently special-cased at several sites — `save_colmap_binary` and the BA
> readback in `_bundle_adjust.py`, plus two parallel reprojection-error loops in
> `reconstruction/data/recompute.rs` (`compute_observation_reprojection_errors` for
> `sift_files`, `embedded_point_reprojection_errors` for `embedded_patches`).
> These should collapse onto a single source-agnostic accessor on the
> reconstruction (an "observed keypoint for (point, observation)" lookup, plus a
> `keypoints_per_image()` helper) so a new feature source or a change to the
> reprojection/at-infinity semantics is a one-place edit rather than several
> kept-in-sync copies.

#### `--refine-normals [<params>]`

Refines each finite point's `normal` to the one that maximizes photometric
cross-view consensus, leaving the point count, positions, poses, and cameras
unchanged. Because it is photometric it reads the workspace source images, so
those must still be present where the reconstruction was created. The refined
patch cloud (per-point in-plane `u`/`v` half-extent vectors) is always re-written
beside the normals in the `.sfmr` `points3d/` section (the stored frame stays
consistent with them); `bitmaps` (on by default) additionally renders the
per-point patch textures so the output is self-contained, and `bitmaps=false`
skips that render. The optional value is a comma-separated `key=value` string;
with no value it runs the v1 defaults.

The operation requires an `embedded_patches` reconstruction and rejects
`sift_files`, so the canonical chain is `--to-embedded-patches
--refine-normals`: convert first, then refine, so the refinement positions its
views on the stored per-observation keypoints.

See [refine-normals-command.md](refine-normals-command.md) for the
full parameter list and semantics.

```bash
--refine-normals
--refine-normals angular_range_deg=25,init_steps=7
--refine-normals bitmaps=false
--to-embedded-patches --refine-normals
```

#### `--refine-keypoints [<params>]`

Refines each observation's stored 2D keypoint to sub-pixel by a local
photometric solve (forward-additive ECC Gauss–Newton against a robust
cross-view consensus; never worse than the seed). A pure in-place modifier: the
point count, positions, poses, cameras, normals, and the entire track structure
are unchanged — only `keypoints_xy` values move. It does **not** re-fit the
stored patch frames; with `bitmaps` (on by default) it additionally re-renders
the per-point patch textures at the refined keypoints (re-persisting the
unchanged frame beside them) so the output is self-contained, and `bitmaps=false`
skips that render. Because it is photometric it reads the workspace source
images, so those must still be present where the reconstruction was created.
The optional value is a comma-separated `key=value` string; with no value it
runs the binding defaults.

Like `--refine-normals` it requires an `embedded_patches` reconstruction and
rejects `sift_files` (the stored inline keypoints are the refiner's seeds), so
the canonical chain converts first.

See [refine-keypoints-command.md](refine-keypoints-command.md) for
the full parameter list and semantics.

```bash
--refine-keypoints
--refine-keypoints max_outer_sweeps=2,sampler=anisotropic
--refine-keypoints bitmaps=false
--to-embedded-patches --refine-keypoints --refine-normals
```

#### `--localize-keypoints [<params>]`

Localizes each observation's 2D keypoint by a discrete cross-view search
(congealing against a leave-one-out consensus), seeded at each point's own
projection. **Structural**, unlike the two refine ops: views that won't
co-register are dropped, points whose kept-view count falls below `min_views`
(default 2) are culled, and `keypoints_xy` plus the entire track structure are
rebuilt from the survivors (via the same compaction helper the `embed-patches`
pipeline uses). Cameras, poses, and each surviving point's 3D geometry are
unchanged. Stored patch bitmaps are dropped as stale (the frames are kept) —
re-run `--refine-keypoints` or `--refine-normals` to regenerate them (both
render bitmaps by default); there is no `bitmaps` key on this op. Because it is
photometric it reads the workspace source images, so those must still be
present where the reconstruction was created. The optional value is a
comma-separated `key=value` string; with no value it runs the binding defaults
plus `min_views=2`.

Like the refine ops it requires an `embedded_patches` reconstruction and
rejects `sift_files` (the stored per-point patch frames are the search
geometry), so the canonical chain converts first. It pairs naturally with
`--refine-keypoints` (search into the basin, then sharpen to sub-pixel);
`--refine-normals` can sit on either side of it — both orderings are
legitimate, pick per dataset.

See [localize-keypoints-command.md](localize-keypoints-command.md)
for the full parameter list and semantics.

```bash
--localize-keypoints
--localize-keypoints search=8,min_views=3
--to-embedded-patches --localize-keypoints --refine-keypoints
```

### Representation

#### `--to-embedded-patches [<params>]`

Converts a `sift_files` reconstruction into an `embedded_patches` one
([sfmr-file-format.md](../../../formats/sfmr-file-format.md), "Observation source")
**without any photometric adaptation**: each point gets a `(u, v)` patch frame
from the mean viewing direction (no normal refinement), each observation's inline
`keypoints_xy` is copied verbatim from its `.sift` feature, and each image's
`image_file_hashes` entry is read from the `.sift` metadata
(`image_file_xxh128` — a minimal metadata read). The point count, positions,
poses, and cameras are unchanged; the keypoints are exactly the original SIFT
detections. It runs none of the photometric steps of the
[sift→patch pipeline](../../../core/patch/sift-to-patch-reconstruction.md) (normal refinement
+ view selection + keypoint localization).

It reads the `.sift` files (keypoints, image hashes, and the default
`extent=feature_size` patch sizing), so they must still be present where the
reconstruction was created. After this op the reconstruction is
`embedded_patches`, so any later `.sift`-dependent operation in the same chain
will fail. The optional value is a comma-separated `key=value` string
(`normal`, `extent`, `extent_value`, `feature_reduce`, `pixel_reduce`,
`k_neighbors`); with no value it runs the defaults (`normal=mean_viewing`,
`extent=feature_size`).

```bash
--to-embedded-patches
--to-embedded-patches normal=mean_viewing,extent=feature_size,extent_value=5
--to-embedded-patches extent=fixed,extent_value=1.0
```

### Heavy Columns

A reconstruction carries, beside the geometry, two columns of pictures: a
128 x 128 thumbnail of every image and an RGBA bitmap of every surface patch.
Both are conveniences for a viewer, neither says anything about where a camera
or a point is, and together they are nearly all of a file's bytes. Four steps
remove either column from a file or put it back from the source photographs,
and `--minimal` is a shorthand over two of them. They run in command-line order
like every other step.

```bash
sfm xform in.sfmr out.sfmr --drop-thumbnails
sfm xform in.sfmr out.sfmr --drop-patch-bitmaps
sfm xform in.sfmr out.sfmr --add-thumbnails
sfm xform in.sfmr out.sfmr --add-patch-bitmaps [resolution=<R>,sampler=<S>]
sfm xform in.sfmr out.sfmr --minimal [wspath=<path>]
```

An `--add-*` step is a no-op, with one printed line, on a reconstruction that
already carries the column. To re-render at a different resolution, drop first:
`--drop-patch-bitmaps --add-patch-bitmaps resolution=32`.

**Drop, add and remove** are three different verbs here (see
[GLOSSARY.md](../../../GLOSSARY.md)): *drop* discards an optional column and
keeps every row, *add* fills an absent optional column from the source data,
and *remove* (`--remove-short-tracks` and the other point filters) deletes
points.

#### `--drop-thumbnails`

Discards the per-image thumbnail column
(`clone_with_changes(thumbnails_y_x_rgb=None)`). Reads no files and keeps every
row of everything. Valid on both feature sources; a no-op, with one printed
line, on a reconstruction without thumbnails.

#### `--drop-patch-bitmaps`

Discards the per-point patch bitmap column
(`clone_with_changes(patch_bitmaps=None)`). The patch frames
(`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`) and the normals stay, so the
patches keep their geometry and a later `--add-patch-bitmaps`,
`--refine-keypoints` or `--refine-normals` can render onto them. Valid on both
feature sources; a no-op, with one printed line, on a reconstruction without
bitmaps.

#### `--add-thumbnails`

Builds the thumbnail column from each image's **`.sift` copy first and its
source photograph second**.

**The `.sift` copy**, when it verifiably belongs to the image: in a
`sift_files` file, by its content hash against `sift_content_hashes`; in an
`embedded_patches` file, by its recorded `image_file_xxh128` against
`image_file_hashes`. An `embedded_patches` file records no `.sift` link at all,
so a `.sift` there is found only by the path `feature_prefix_dir` gives. A
`.sift` thumbnail is the row the extractor already reduced from the photograph,
so reading it is a 48 KiB decompression where the photograph is a full decode
and a resize, and the two give the same bytes. That is why it is read first.

**The photograph**, for an image whose `.sift` is missing or unverified:
`workspace_dir / name` is decoded as colour with its EXIF orientation ignored,
resized to 128 x 128 by area averaging and converted to RGB. That is the SIFT
extractors' own decode and resize
([extract_sfmtool.py](../../../../src/sfmtool/sift/extract_sfmtool.py),
`read_image_bgr` and `thumbnail_of_bgr`), so each row is byte-identical to the
`.sift` thumbnail an extractor writes for the same photograph, and to the row a
file carried before `--drop-thumbnails`. The decode deliberately does not go
through [_workspace_image.py](../../../../src/sfmtool/_workspace_image.py)
`read_workspace_image`, which applies the orientation tag. In an
`embedded_patches` file a photograph read is first checked against the image's
stored `image_file_hashes` entry, the hash that says "this is still the
photograph the reconstruction was built from"; a mismatch fails the step.

The step prints how many rows came from each source. The column is whole or
absent and the format forbids a placeholder row, so an image that yields no row
from either source fails the step, and the error names every such image.

The viewer fills in a missing column for display in the same order, with its
own decode for the photograph
([multi-panel-image-browser.md](../../../gui/multi-panel-image-browser.md)
§ "Thumbnail loading").

#### `--add-patch-bitmaps [resolution=<R>,sampler=<S>]`

Renders an RGBA bitmap for every point that has a patch frame, **at the patch's
stored frame and each observation's stored keypoint, moving nothing**:
positions, normals, frames, keypoints and tracks come out exactly as they went
in. It reads the source photographs, and requires an `embedded_patches`
reconstruction with patch frames, like the other patch steps. A point with
fewer than two observations that render in frame gets a zero row, as
`--refine-keypoints` gives it.

`resolution` defaults to 24, the default of `KeypointSubpixelParams::resolution`
([keypoint_subpixel/params.rs](../../../../crates/sfmtool-core/src/patch/keypoint_subpixel/params.rs))
and of the `resolution` key of `--refine-keypoints` and `--refine-normals`; it
cannot default to the file's `patch_bitmap_resolution`, which is `null` whenever
the step has anything to do. `sampler` takes the values and default of the
`sampler` key of `--refine-keypoints`. Those are the only two keys: the other
sub-pixel parameters tune a solve this step does not run.

The render is the one place a representative is fused for a patch whose
placement and keypoints are settled:
[`fuse_patch_bitmap`](../../../../crates/sfmtool-core/src/patch/keypoint_subpixel.rs)
runs the sub-pixel kernel with no Gauss-Newton step and a single sweep, and
`fuse_patch_cloud_bitmaps` is its whole-cloud form, parallel over points. It takes
one view per image as an `Option`, leaving a `None` view out of every patch's
view set, and a `Progress` that counts `patches` and can cancel it. It is bound
as `PatchCloud.render_bitmaps(recon, images, resolution=24,
sampler="bilinear_mip", progress=None)`, which returns the `(P, R, R, 4)` array
`clone_with_changes(patch_bitmaps=...)` takes. The bench commit
(`bench::fit::fuse_bitmap`) calls the same function for one track, and the
viewer's open runs the whole-cloud form for a file whose bitmaps are absent.

Its bitmaps therefore equal what `--refine-keypoints` renders for a patch whose
keypoints it did not move, and what the bench commits. They are not
byte-identical to `--refine-normals` bitmaps, which fuse over a different view
subset with an obliquity weight, so a file whose bitmaps came from
`--refine-normals` does not round-trip bit-exactly through
`--drop-patch-bitmaps --add-patch-bitmaps`.

#### What the four steps do not touch

None of the four moves a point, renumbers a row or changes a keypoint, so the
input's `lineage`, which maps ancestor points onto this file's rows, stays valid
across them, and they keep `xform`'s ordinary lineage behaviour: the input's
list is written into the output unchanged, and no entry is added. The
`derived/` section is recomputed by the save as always.

#### `--minimal`

Writes the smallest file that still holds the whole reconstruction. It is a
shorthand for `--drop-patch-bitmaps --drop-thumbnails`, plus a clearing of the
metadata that is machine-local or incidental, and it is what a file meant for a
repository is written with:

```bash
sfm xform in.sfmr out.sfmr --minimal
sfm xform in.sfmr out.sfmr --minimal --add-thumbnails
```

It has two parts, which take effect at different times:

- **The column part is an ordered step.** At its position in the chain it does
  what `--drop-patch-bitmaps --drop-thumbnails` would do there. A later step sees
  a reconstruction without either column, so `--minimal --add-thumbnails` writes
  a file with thumbnails and minimal metadata. An `--add-*` step after it prints
  one line saying it restores part of the shorthand, so the combination reads as
  intended.
- **The metadata part is a property of the save.** The save rewrites
  `operation`, `tool`, `tool_version`, the counts and both workspace paths after
  every step has run, so `--minimal` anywhere in the chain marks the output, and
  the save (`SfmrReconstruction.save(..., minimal=True)`) writes it minimal.
  Giving it twice is the same as giving it once. The clearing is
  `SfmrReconstruction::clear_minimal_metadata`, run after the save's
  `stamp_save`, in
  [minimal.rs](../../../../crates/sfmtool-core/src/reconstruction/minimal.rs).
  That is the one definition of a minimal file: the viewer's
  `File > Save As Minimal...` writes through `to_minimal`, which drops both
  columns and runs the same stamp and clearing
  ([saving.md](../../../gui/saving.md) § "Save As Minimal").

Thumbnails are in the shorthand as well as bitmaps because the purpose is the
smallest file, and the `--add-*` steps rebuild both: the thumbnails from the
`.sift` files or the photographs, the bitmaps from the photographs.
A caller that wants the thumbnails kept writes `--minimal --add-thumbnails`.

##### `--minimal wspath=<path>`: stating the workspace path

The save measures `workspace.relative_path` from the output's directory to the
workspace, resolving both directories to their real locations first so that a
symlinked or aliased output gives the step or two between them rather than a walk
from the filesystem root. `wspath=<path>` records `<path>` instead and measures
nothing:

```bash
sfm xform in.sfmr ws/ground_truth.sfmr --minimal wspath=.
```

It is for a file whose home is not the directory it happens to be written from. A
ground truth checked into a repository inside its own workspace records `.`,
whether it was written there or staged elsewhere and moved in afterwards. The
value is taken as written, with `\` turned into `/` because the field is POSIX, so
nothing about it is interpreted: `wspath=.` is a file beside its workspace marker,
and `wspath=` with nothing after it records an empty value, which the format reads
as no path recorded.

`wspath` is the only key `--minimal` takes. An unknown key, a token without `=`,
an empty key and a repeated key are usage errors, the same grammar the other
`[PARAMS]` options use. The statement belongs to the save, as the rest of the
metadata part does, so the last `--minimal` in the chain that names one is what
the output records, and the step's `tool_options.transforms` entry names it. The
binding argument behind it is
`SfmrReconstruction.save(..., workspace_path=<path>)`, which needs an `operation`
because the stamp is what writes the field.

**`--minimal` drops `lineage` entirely.** The output is a new file with no
ancestry: it carries none of the input's entries and gains none for the input.
Point IDs minted against the input or any earlier version therefore do not
resolve in the minimal file. A minimal file is a root, typically published
somewhere its ancestors are not, so an entry could never lead a reader to an
ancestor file.

What it clears and what it keeps:

| Field | `--minimal` | Why |
|---|---|---|
| `metadata.json` `workspace.absolute_path` | **cleared** to `""` | Names one machine's filesystem, and sits in `metadata_xxh128`, so the same reconstruction written on two machines would hash differently. An empty value means none was recorded. |
| `workspace.relative_path` | recomputed by the save, from the output's directory to the workspace, as every `xform` save does, or recorded as `wspath` states it | It is how a reader finds the workspace from where the file is. |
| `workspace.contents` | kept | States which extractor and settings the features came from; `feature_prefix_dir` is how every `.sift`-reading step and the viewer's SIFT index build locate the features. |
| `lineage` | **dropped**, and no entry added | Above. |
| `tool_options` | **replaced** by `{"transforms": [...]}`, this invocation's own step list | An ordinary save *merges* its `transforms` record into the input's options, so a file accumulates the options of every operation behind it. The inherited keys describe an ancestor, which a root does not have. |
| `operation`, `tool`, `tool_version` | rewritten by the save as for any `xform` (`xform`, `sfmtool`, its version) | They describe the operation that wrote this file. |
| `version`, `feature_source`, the counts | kept, recomputed by the save | Format facts. |
| `world_space_unit` | kept when present | It states what the coordinates mean. |
| `written.json` `timestamp` | written by the save as always | Outside every section hash, so it adds nothing to the identity. |
| `derived/` | recomputed by the save, kept | Recomputable, and outside `content_xxh128` already. |
| `images/image_file_hashes` | kept | The identity of each photograph, which is what lets `--add-thumbnails` check that the photographs found are the ones reconstructed. |
| `images/thumbnails_y_x_rgb` | **dropped** | The shorthand. |
| `points3d/patch_bitmaps_y_x_rgba` | **dropped** | The shorthand. |
| `points3d/patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`, `normals_xyz` | kept | Geometry: the patches keep their placement, so bitmaps can be rendered back onto them. |
| positions, colours, reprojection errors, tracks, keypoints, poses, cameras, image names | kept | The reconstruction itself. |

The other optional columns (`normal_confidence`, `observation_confidence` and
the point-constraint triple) are measurements or solve state and are kept when
present. Nothing beyond the table is cleared: `--minimal` removes the columns
that are the size and the metadata that describes a machine or a history, not
every byte that could be recomputed. The same input written by `--minimal`
twice, to the same place relative to its workspace, by one sfmtool version, has
the same `content_xxh128` whatever machine wrote it.

### Scaling to Physical Units

#### `--scale-by-measurements <measurements.yaml>`

Scales the reconstruction to physical units using known real-world distances between pairs
of 3D points. The measurements are specified in a YAML file with point IDs, distances, and
a target unit. With multiple measurements, the median scale factor is used.

See [scale-by-measurements-command.md](scale-by-measurements-command.md) for the full
specification, including the YAML schema, unit conversion, point ID resolution across
reconstructions, and diagnostics output.

```bash
--scale-by-measurements measurements.yaml
```

### Alignment

#### `--align-to <other.sfmr>`

Aligns the current reconstruction to match another using a similarity transformation
(rotation, translation, scale) computed from shared images and 3D point correspondences.

```bash
--align-to ground_truth.sfmr
```

#### `--align-to-input`

Aligns back to the original input reconstruction. Useful after filtering operations
to keep the result in the original coordinate frame.

```bash
sfm xform in.sfmr out.sfmr --remove-short-tracks 2 --align-to-input
```

## Architecture

The implementation uses a `Transform` protocol:

```python
class Transform(Protocol):
    def apply(self, reconstruction: SfmrReconstruction) -> SfmrReconstruction: ...
```

Each operation is a class implementing this interface. The CLI parses arguments into an
ordered list of `Transform` objects and applies them sequentially. The reconstruction is
loaded once, transformed through the pipeline, and written once.

The write is `SfmrReconstruction.save(path, operation="xform",
tool_options={"transforms": [...]})`, which stamps `operation`, `tool`,
`tool_version` and the counts, recomputes both workspace paths from the output's
location, with both directories resolved to their real locations before the
relative one is measured between them, merges `transforms` into the inherited
`tool_options`, and passes
`lineage` through unchanged. The stamp is `SfmrReconstruction::stamp_save` in
sfmtool-core. With `minimal=True`, set when `--minimal` is in the chain, it then
clears `absolute_path`, drops `lineage` and replaces `tool_options` through
`clear_minimal_metadata` (see [`--minimal`](#--minimal)). With
`workspace_path=<path>`, set by `--minimal wspath=<path>`, the stamp records that
path as `workspace.relative_path` and measures nothing
(see [`--minimal wspath=<path>`](#--minimal-wspathpath-stating-the-workspace-path)).

### Rust primitives behind the operations

Most operations bottom out in one of three editing primitives on
`SfmrReconstruction` in `crates/sfmtool-core/src/reconstruction/edit.rs`, each of
which returns a new reconstruction:

| Primitive | Used by | Semantics |
|-----------|---------|-----------|
| `apply_se3_transform` | `--rotate`, `--translate`, `--scale`, `--scale-by-measurements`, `--align-to`, `--align-to-input` (invoked as `Se3Transform @ recon` from Python) | Applies a similarity to points and camera poses. Finite points get the full rotation+translation+scale; points at infinity and per-point normals are directions, so only the rotation acts (renormalized). Rig sensor translations are scaled; per-point patch `(u, v)` half-vectors rotate and scale with their point (rotation-only at infinity); patch bitmaps are pose-invariant and pass through unchanged. |
| `subset_by_image_indices` | `--include-range`, `--exclude-range`, `--include-glob`, `--exclude-glob`, `--include-by-distribution` (via `xform/_filter_by_image_range.py`); also `sfm to-colmap-bin --range`, `sfm to-nerfstudio --range`, and `sfm panorama` source subsetting | Keeps the listed images (in order), drops observations of removed images, and prunes rig frames with no surviving image (remapping frame indices). With `drop_orphaned_points=true` (the xform filters' mode), points with zero remaining observations are removed and point IDs remapped contiguously. `sift_files` reconstructions only. |
| `filter_points_by_mask` | `--remove-isolated`, `--remove-short-tracks`, `--remove-narrow-tracks`, `--remove-large-features` (`xform/_point_filters.py`), `--filter-by-reprojection-error`, `--filter-by-keypoint-uncertainty`, `--filter-by-patch-size` | Keeps points where the boolean mask is true, filters their observations, and remaps point IDs contiguously. Images, cameras, and rig data are unchanged. Works for both `sift_files` and `embedded_patches` (inline keypoints are filtered in lockstep); per-point patch rows follow their point. |

A fourth, standalone primitive `reconstruction/filter.rs::filter_tracks_by_point_mask`
performs the same mask-filter-and-remap on bare track columns (no
reconstruction object). It is exposed to Python as
`analysis.filter_tracks_by_point_mask_py` but the xform pipeline uses the
reconstruction-level `filter_points_by_mask` instead.

## Usage Examples

```bash
# Clean and align reconstruction
sfm xform raw.sfmr clean.sfmr \
  --remove-short-tracks 2 \
  --remove-isolated 3.0,median \
  --filter-by-reprojection-error 2.0 \
  --align-to ground_truth.sfmr

# Transform to world coordinates
sfm xform model.sfmr world.sfmr \
  --rotate 1,0,0,90deg \
  --translate 0,0,-5

# Iterative refinement
sfm xform raw.sfmr refined.sfmr \
  --remove-short-tracks 2 \
  --bundle-adjust \
  --filter-by-reprojection-error 2.0 \
  --remove-isolated 2.5,median \
  --scale 0.01

# Filter by image range and clean up
sfm xform full.sfmr subset.sfmr \
  --include-range 10-50 \
  --remove-short-tracks 2

# Filter, optimize, then scale to physical units
sfm xform input.sfmr output.sfmr \
  --remove-short-tracks 2 \
  --bundle-adjust \
  --scale-by-measurements measurements.yaml

# Keep only the left fisheye of a 360° rig and clean up
sfm xform rig.sfmr left_only.sfmr \
  --include-glob "*fisheye_left*" \
  --remove-short-tracks 1

# The smallest file for a repository, with the thumbnails kept
sfm xform ground_truth.sfmr published/ground_truth.sfmr --minimal --add-thumbnails

# The same for a file checked in inside its own workspace, which records the
# workspace it sits in rather than the path it was written from
sfm xform candidate.sfmr ws/ground_truth.sfmr --minimal wspath=.

# Upgrade SIMPLE_RADIAL → RADIAL so bundle adjustment can refine k2

sfm xform input.sfmr refined.sfmr \
  --camera-model RADIAL \
  --bundle-adjust
```
