# `sfm undistort` Command

## Overview

Removes lens distortion from every image in a reconstruction, producing a new
self-contained workspace whose images use best-fit pinhole cameras with square
pixels. The output is a first-class sfmtool workspace: all downstream commands
(`sfm inspect`, `sfm solve`, the GUI viewer, etc.) can open it directly.

Per image, the command:
1. Warps the source image through the distortion model to a pinhole target.
2. Transforms `.sift` feature positions and affine shapes into the pinhole frame,
   dropping features that map outside the frame.
3. Remaps the reconstruction's tracks to the new feature indices and removes
   3D points that lose every observation.
4. Assembles a new `.sfmr` file with pinhole cameras, the original image poses,
   and the cleaned-up tracks.

## Command Syntax

```bash
sfm undistort <RECONSTRUCTION.sfmr> [OPTIONS...]
```

`RECONSTRUCTION.sfmr` must be a `.sfmr` file. The command fails otherwise.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--fit` | `inside` \| `outside` | `inside` | Pinhole fit mode. See below. |
| `--filter` | `aniso` \| `bilinear` | `aniso` | Image resampling filter. |
| `-o`, `--output` | path | `<stem>_undistorted/` next to the input | Output workspace directory. |

### `--fit`

- `inside` — the pinhole frame fits inside the distorted image. No black
  borders in the output, but corners of the distorted image are cropped and
  features near those corners get dropped.
- `outside` — the pinhole frame encloses the distorted image. No source
  features are lost, but output images can have black borders.

The tradeoff matters more for strong distortion (wide-angle / fisheye) and
less for mild distortion.

### `--filter`

- `aniso` — anisotropic resampling; slower but reduces aliasing where the warp
  compresses the source (typical at fisheye image edges).
- `bilinear` — plain bilinear; faster but can alias near image edges.

## Output Layout

```
{output_dir}/
├── .sfm-workspace.json
├── <same relative layout as the source workspace>/
│   └── image_001.jpg
└── sfmr/
    └── undistorted.sfmr
```

Each `.sift` file lives next to its image, in a `features/sift-sfmtool-undistort-{hash}/`
subdirectory of the image's parent. For example an image at
`frames/scene_001.jpg` gets a sift file at
`frames/features/sift-sfmtool-undistort-{hash}/scene_001.jpg.sift`.

Undistorted images are written at the same relative paths they occupied in the
source workspace. The `.sfmr`'s `image_names` therefore uses identical relative
paths. Each source is read with `read_image_rgb`, or with `read_image_rgba`
when `image_has_alpha` says it has an alpha channel, so a source with alpha is
written with alpha, by `write_image_rgba`, and any other by `write_image_rgb`.
The format is the one the source's extension names, and a JPEG is written at
quality 95. The images have 8 bits per channel, and a grey source is
written as a 3-channel image
([reading-and-writing-images.md](../../core/camera/reading-and-writing-images.md)).

### Workspace config (`.sfm-workspace.json`)

```json
{
  "version": 1,
  "feature_tool": "sfmtool-undistort",
  "feature_type": "sift-sfmtool-undistort",
  "feature_options": {
    "source_feature_tool": "colmap",
    "source_feature_options": { ... },
    "fit": "inside",
    "filter": "aniso"
  },
  "feature_prefix_dir": "features/sift-sfmtool-undistort-{hash}"
}
```

`feature_tool` is the sentinel `sfmtool-undistort`; `feature_type` is the
compound string `sift-sfmtool-undistort`. `source_feature_tool` and
`source_feature_options` are the `feature_tool` and `feature_options` recorded
in the input `.sfmr`'s `workspace.contents`, the features it was built from.

Features in this workspace are produced by undistortion, not extraction. The
sentinel `feature_tool` value is reserved so that extraction tooling can
distinguish an undistort workspace from a regular one. `sfm sift --extract`
does not currently recognize this value — do not run it on an undistort
workspace; the pre-existing `.sift` files are the features.

## Behavior

### Cameras

For each unique camera in the source reconstruction, the command builds a
best-fit pinhole with the source image's width and height:

- `--fit inside` uses `CameraIntrinsics.best_fit_inside_pinhole(w, h)`.
- `--fit outside` uses `CameraIntrinsics.best_fit_outside_pinhole(w, h)`.

Output images keep the source resolution. The cameras in the output
`.sfmr` are contiguous — only the cameras used by an image or by a rig sensor
are emitted — and `camera_indexes` and `rig_frame_data.sensor_camera_indexes`
are remapped accordingly. A rig sensor can name a camera no image uses, as in a
subset of a rig reconstruction that dropped all of one sensor's images. That
camera has no image to take a size from, so its pinhole is fitted at the
camera's own `width` and `height`.

Only a perspective camera model can be fitted with a pinhole. A fisheye or
equirectangular camera makes the fit raise `ValueError` ("camera model '…' is
not a perspective projection and cannot be converted to a pinhole"), and the
command fails (see [Error behavior](#error-behavior)).

An image with an intrinsic adjustment
([sfmr-file-format.md](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13))
is read through its own intrinsics, its camera's with the adjustment applied,
and written through its camera's pinhole, which has no adjustment. An
adjustment describes how one photograph departs from its camera's lens, so
undistortion removes it along with the distortion: every image of a camera
comes out through the same pinhole. That holds for all four parts of an
adjustment: the focal scale, the principal-point offset, and on an
`SFMTOOL_PINHOLE` camera the radial correction and the rolling-shutter rates.
The warp map and the keypoint remap read the source pixels through the image's
own intrinsics, whose spline has the radial correction added and whose
`project` and `unproject` apply the rolling-shutter step and its inverse
([sfmtool-pinhole-kernels.md § Per-image
corrections](../../core/camera/sfmtool-pinhole-kernels.md#per-image-corrections)),
so each output pixel and keypoint lies on the ray the corrected image saw
there, and the rolling shutter, a rotation during readout, is removed exactly.
The output has no adjustments: it stores no adjustment columns, no camera lists
a parameter in `per_image_adjustments`, and no camera keeps a
`rolling_shutter_scanlines` setting, since its images have no readout left to
describe, so it reads like any unadjusted file.

The pinhole is fitted once per camera, to the camera's shared intrinsics, not
to each image's. An image whose principal point is offset therefore loses a
sliver along one side under `--fit inside`, or shows a slightly wider black
border there under `--fit outside`, about as wide as the offset. Fitting each
camera's pinhole over all of its images' intrinsics would avoid that, but would
make the pinhole depend on which images a reconstruction holds and would widen
or shrink every image of the camera for the sake of one.
The images without an adjustment share their camera's warp map; an adjusted
image's is built for it alone, since its source intrinsics are its own.

### Image poses

Image quaternions and translations are unchanged. Undistortion affects only
the camera intrinsics; extrinsics are invariant.

### Points at infinity

Undistortion handles finite points only, so its first step is
`materialize_infinity_for_export` in
[`colmap/io.py`](../../../src/sfmtool/colmap/io.py). When the input has points
at infinity (`w = 0`), it prints
"`  Undistortion: materializing N point(s) at infinity`" and replaces the
reconstruction with `SfmrReconstruction.materialize_points_at_infinity()`,
which places each such point along its stored direction at a finite depth far
enough that its parallax falls below one pixel in every camera that observes
it. The rest of the command works on that reconstruction, and the output
`.sfmr` stores every point with `w = 1`, so it has no points at infinity.

### `.sift` files

For every image the command writes a new `.sift` file under
`{image_dir}/features/sift-sfmtool-undistort-{hash}/{image_basename}.sift`.

**Source.** Each image's source `.sift` file is found through the
`feature_prefix_dir` recorded in the input `.sfmr`'s `workspace.contents`
(`get_sift_path_from_recon` in
[`sift/file.py`](../../../src/sfmtool/sift/file.py)), not through the source
workspace's current `.sfm-workspace.json`. The reconstruction's feature indexes
refer to rows of those files, so the lookup still finds them after the
workspace has been re-extracted with another tool or other options.

**Positions.** Each keypoint `(x, y)` in the distorted image is mapped to the
pinhole frame via `pinhole.project(distorted.unproject(x, y))`, where
`distorted` is the image's own intrinsics and `pinhole` its camera's shared
pinhole (see [Cameras](#cameras)). Keypoints that
land outside `[0, pinhole.width) × [0, pinhole.height)` are dropped.

**Affine shapes.** Each keypoint's 2×2 affine shape `A` (the matrix that maps
the unit circle to the keypoint's elliptical footprint in pixel space) is
updated to `A' = J · A`, where `J` is the 2×2 Jacobian of the
distorted-to-pinhole mapping, computed by central differences with
`eps = 0.5` pixels. This is the same approach used by `WarpMap.compute_svd()`
and works for every camera model without per-model analytic derivatives.

**Order.** Kept keypoints keep their order from the source `.sift` file, and
the output `.sfmr`'s feature indices are remapped to their new row numbers.
Because `J` differs from keypoint to keypoint, the rows are in general no longer
in exact descending size; the format allows that (see
[Feature ordering](../../formats/sift-file-format.md#feature-ordering)).

**Descriptors.** SIFT descriptors are copied through unchanged. The
undistortion-induced resampling produces only small local geometric changes,
which SIFT is designed to tolerate. Descriptors are not re-extracted from the
undistorted pixels.

**Thumbnail.** Regenerated from the in-memory undistorted image (its RGB
channels resized to 128×128 with `cv2.INTER_AREA`) and embedded in the `.sift`
file. The same thumbnail is the image's thumbnail in the output `.sfmr`. The
`.sift` file is written **after** the image file; `image_file_xxh128` in the
`.sift` metadata is therefore the hash of the bytes actually on disk.

**Metadata.** `image_name` is the image basename (matching other `.sift`
writers); `image_file_xxh128` and `image_file_size` reflect the newly written
undistorted image; `image_width` / `image_height` use the pinhole camera's
dimensions. `feature_tool_metadata` records the provenance (source feature
tool and options, `fit`, `filter`).

### Track remapping

After all images are processed the command builds a per-image table
`old_feature_index → new_feature_index` and applies it to the track arrays:

1. Observations whose feature was dropped are removed.
2. 3D points with zero surviving observations are removed and remaining points
   are renumbered.
3. `observation_counts` is recomputed.
4. Track rows are sorted lexicographically by `(points3d_index, image_index)`.

### Reprojection errors and depth statistics

Per-point reprojection errors and depth statistics are computed from the
surviving points when the output `.sfmr` is built. `SfmrReconstruction.from_data`
fills the inline keypoints from the newly written `.sift` files, then
recomputes the depth statistics and scores each point's RMS reprojection error
through the pinhole cameras, so the stored errors describe the output, not the
source ([sfmr-file-format.md](../../formats/sfmr-file-format.md)). The errors
the source reconstruction stored are not kept.

### Rig data

If the source reconstruction carries rig data (`rigs/` and `frames/`
sections), it is carried through verbatim. Rigs describe *relative* sensor
poses, which are unaffected by undistortion. `sensor_camera_indexes` is
remapped to the new contiguous camera indices.

### Feature-drop summary

After processing, the command prints a summary line, followed by the image
that lost the largest share of its features when any image lost one:

```
Feature summary: 12345/13500 kept (1155 dropped, 8.6%)
  Highest drop rate: frames/scene_047.jpg (17.2%)
```

With `--fit outside`, dropped counts are typically zero (all source features
project inside the enlarged pinhole frame). With `--fit inside`, corner
features get dropped in rough proportion to the distortion magnitude.

### `.sfmr` metadata

The output `.sfmr`'s metadata block records:

```json
{
  "version": 2,
  "operation": "undistort",
  "tool": "sfmtool",
  "tool_version": "<sfmtool version>",
  "workspace": { ... },
  "timestamp": "<ISO 8601>",
  "image_count": ...,
  "point_count": ...,
  "observation_count": ...,
  "camera_count": ...,
  "tool_options": {
    "fit": "inside",
    "filter": "aniso",
    "source_sfmr": "sfmr/solve_001.sfmr"
  }
}
```

`source_sfmr` is the source `.sfmr`'s path relative to its workspace root; it
falls back to the file's basename, or to the literal string `"unknown"` if
the source workspace cannot be located.

### Error behavior

Any exception raised while the command runs is reported as a
`ClickException` carrying the exception's message, and the command exits with
status 1. The failures it can meet include a missing source image
(a `FileNotFoundError`, reported as "`Error: Image not found: …`"), a missing or unreadable source
`.sift` file, a source image that cannot be decoded, an undistorted image that
cannot be written (`write_image_rgb` / `write_image_rgba`), and a camera model
that cannot be fitted with a pinhole (see [Cameras](#cameras)). Images are
processed in order and nothing is cleaned up, so the partially written output
workspace is left on disk.

### Stdout

On startup: the materialisation line when the input has points at infinity
(see [Points at infinity](#points-at-infinity)), then one
"`Undistorting N images (fit=..., filter=...)...`" line. For each unique source camera, when
its first image is processed: a two-column table of original vs pinhole
parameters. A progress line is printed for the first three images and then
every tenth. After the per-image loop the feature-drop summary is printed, then
a table for each rig-sensor camera no image uses, and then the closing summary,
which names the output paths once:

```
Undistorted 17 images
Output directory: sfmr/solve_001_undistorted
Reconstruction: sfmr/solve_001_undistorted/sfmr/undistorted.sfmr
  1 cameras, 780 3D points, 4512 observations
```

## Usage Examples

```bash
# Default: inside fit, anisotropic filter, output next to the .sfmr.
sfm undistort sfmr/solve_001.sfmr
# -> sfmr/solve_001_undistorted/

# Keep every source feature (accepts black borders).
sfm undistort sfmr/solve_001.sfmr --fit outside

# Faster but slightly more aliased resampling.
sfm undistort sfmr/solve_001.sfmr --filter bilinear

# Explicit output directory.
sfm undistort sfmr/solve_001.sfmr -o /tmp/undistorted
```
