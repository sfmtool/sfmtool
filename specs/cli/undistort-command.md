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
paths.

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
`source_feature_options` are copied from the source workspace (falling back to
`"colmap"` with empty options when the source workspace config cannot be
located).

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

Output images keep the source resolution. The camera table in the output
`.sfmr` is contiguous — only the cameras actually used by images are emitted —
and `camera_indexes` and `rig_frame_data.sensor_camera_indexes` are remapped
accordingly.

### Image poses

Image quaternions and translations are unchanged. Undistortion affects only
the camera intrinsics; extrinsics are invariant.

### `.sift` files

For every image the command writes a new `.sift` file under
`{image_dir}/features/sift-sfmtool-undistort-{hash}/{image_basename}.sift`.

**Positions.** Each keypoint `(x, y)` in the distorted image is mapped to the
pinhole frame via `pinhole.project(distorted.unproject(x, y))`. Keypoints that
land outside `[0, pinhole.width) × [0, pinhole.height)` are dropped.

**Affine shapes.** Each keypoint's 2×2 affine shape `A` (the matrix that maps
the unit circle to the keypoint's elliptical footprint in pixel space) is
updated to `A' = J · A`, where `J` is the 2×2 Jacobian of the
distorted-to-pinhole mapping, computed by central differences with
`eps = 0.5` pixels. This is the same approach used by `WarpMap.compute_svd()`
and works for every camera model without per-model analytic derivatives.

**Descriptors.** SIFT descriptors are copied through unchanged. The
undistortion-induced resampling produces only small local geometric changes,
which SIFT is designed to tolerate; re-extracting from the undistorted pixels
was explicitly ruled out as unnecessary.

**Thumbnail.** Regenerated from the in-memory undistorted image (resized to
128×128 with `cv2.INTER_AREA`) and embedded in the `.sift` file. The
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

Per-point reprojection errors are carried through from the source
reconstruction (filtered to surviving points) — they are **not** recomputed
against the new pinhole cameras. The stored errors therefore reflect the
*source* camera model's reprojection under the *source* feature positions, not
the new pinhole reprojection. This is a known limitation; a future revision
may call `SfmrReconstruction.recompute_reprojection_errors()` after track
remapping. Depth statistics are produced by `_build_sfmr_data_dict` /
`SfmrReconstruction.from_data` with the remaining points; they are not
explicitly recomputed.

### Rig data

If the source reconstruction carries rig data (`rigs/` and `frames/`
sections), it is carried through verbatim. Rigs describe *relative* sensor
poses, which are unaffected by undistortion. `sensor_camera_indexes` is
remapped to the new contiguous camera indices.

### Feature-drop summary

After processing, the command prints a single summary line of the form:

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
  "version": 1,
  "operation": "undistort",
  "tool": "sfmtool",
  "tool_version": "<sfmtool version>",
  "workspace": { ... },
  "timestamp": "<ISO 8601>",
  "image_count": ...,
  "points3d_count": ...,
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

Missing source image files raise a fatal error (`FileNotFoundError` surfaced
through Click as a `ClickException`). The partially written output workspace
is left on disk.

### Stdout

On startup: one "`Undistorting N images (fit=..., filter=...)...`" line.
For each unique source camera: a two-column table of original vs pinhole
parameters. A progress line is printed for the first three images and then
every tenth. After the per-image loop the feature-drop summary is printed,
followed by a final counts line (image / point / observation totals and the
output paths).

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
