# `sfm xform` Command Design

## Overview

The `sfm xform` command provides a unified interface for applying transformations and
filters to SfM reconstructions. Multiple operations can be chained in a single pass,
applied sequentially in the order specified on the command line.

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

#### `--include-by-distribution <COUNT>`

Keeps a small, strategically-chosen subset of `COUNT` views instead of decimating blindly. A single
greedy loop drives the choice off the reconstructed point cloud: a farthest-point step always heads
to the part of the cloud that is currently worst-covered, and at each such target it adds only the
observers that open up genuinely new viewing angles on it (so covered points end up well
triangulated and the subset solves cleanly on its own). The unit of selection is the rig frame when
rig data is present (both fisheyes of a 360° pair kept together), otherwise the individual image.
Operates on the cloud as given — clean it first if needed (e.g. `--filter-by-reprojection-error`,
`--remove-isolated`) — and typically follow with `--remove-short-tracks 1` to drop stranded
single-observation points.

```bash
--include-by-distribution 16
--filter-by-reprojection-error 2.0 --include-by-distribution 16 --remove-short-tracks 1
```

See [xform-select-by-distribution-command.md](xform-select-by-distribution-command.md) for the full
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

Removes 3D points whose largest SIFT feature in the track exceeds `size` pixels.
"Feature size" is the average of the two column norms of the SIFT affine-shape matrix
(an approximate per-feature radius in source-image pixels, with no rescale to the
reconstructed-image resolution), and the per-track value is the maximum across its
observations.

This filter reads the original `.sift` files associated with the reconstruction's
workspace (resolved as `<workspace_dir>/<image_parent>/<feature_prefix_dir>/<name>.sift`),
so the SIFT artifacts must still be present where the reconstruction was created.
A missing file raises `FileNotFoundError`.

```bash
--remove-large-features 50
```

#### `--filter-by-reprojection-error <threshold>`

Removes 3D points with reprojection error exceeding the threshold (in pixels).

```bash
--filter-by-reprojection-error 2.0
```

### Points at Infinity

`--find-points-at-infinity` is *additive*: it appends new points and tracks, so
the point count grows. See
[xform-find-points-at-infinity.md](xform-find-points-at-infinity.md)
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
in the chain.

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

#### `--refine-normals [<params>]`

Refines each finite point's `estimated_normal` to the normal that maximizes
photometric cross-view consensus, leaving the point count, positions, poses, and
cameras unchanged. Because it is photometric it reads the workspace source
images (and, under the default `extent=feature_size`, the `.sift` files), so
those artifacts must still be present where the reconstruction was created. The
optional value is a comma-separated `key=value` string; with no value it runs the
v1 defaults. Recommended after `--bundle-adjust` (refine against the final
geometry).

See [xform-refine-normals-command.md](xform-refine-normals-command.md) for the
full parameter list and semantics.

```bash
--refine-normals
--refine-normals angular_range_deg=25,init_steps=7
--bundle-adjust --refine-normals
```

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

# Upgrade SIMPLE_RADIAL → RADIAL so bundle adjustment can refine k2
sfm xform input.sfmr refined.sfmr \
  --camera-model RADIAL \
  --bundle-adjust
```
