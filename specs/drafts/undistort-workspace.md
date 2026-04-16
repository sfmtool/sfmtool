# Undistorted Workspace and Reconstruction Output

## Status: Draft

## Overview

When `sfm undistort` processes a reconstruction, it should produce a complete,
self-contained workspace with undistorted images, transformed `.sift` feature
files, and a new `.sfmr` reconstruction file. The output workspace is a
first-class sfmtool workspace that can be used with all downstream commands
(`sfm inspect`, `sfm solve`, the GUI viewer, etc.).

## Current Behavior

`sfm undistort` produces:
- Undistorted images in an output directory
- `undistorted_cameras.json` metadata file

This is useful but limited — the output is not a workspace and has no `.sfmr`
file, so it can't be used with other sfmtool commands.

## Proposed Behavior

`sfm undistort` produces a new workspace containing:

```
{output_dir}/
├── .sfm-workspace.json
├── {image_subdir}/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── features/{feature_prefix_dir}/
│       ├── image_001.jpg.sift
│       └── image_002.jpg.sift
└── sfmr/
    └── undistorted.sfmr
```

The `.sfmr` file contains the full reconstruction with:
- Pinhole cameras (distortion removed)
- Same image poses (extrinsics are unchanged by undistortion)
- Same 3D points, colors, and tracks
- Updated feature hashes pointing to the new `.sift` files
- Updated reprojection errors (recomputed against pinhole model)
- Updated thumbnails (from undistorted images)

## Detailed Design

### 1. Workspace Initialization

The output directory is initialized as a new workspace with `sfm init`. The
workspace configuration records that features were produced by undistortion
rather than extraction:

```json
{
  "version": 1,
  "feature_tool": "sfmtool-undistort",
  "feature_type": "sift",
  "feature_options": {
    "source_workspace": "/path/to/original/workspace",
    "source_sfmr": "sfmr/solve_001.sfmr",
    "fit": "inside",
    "filter": "aniso"
  },
  "feature_prefix_dir": "features/sift-sfmtool-undistort-{hash}"
}
```

`sfm sift --extract` checks the feature tool to decide how to extract. It
should recognize `sfmtool-undistort` as a valid tool whose features already
exist and skip re-extraction (or error clearly rather than trying to
re-extract with COLMAP/OpenCV).

### 2. Image Output

Undistorted images are written to the same relative paths within the new
workspace as they had in the original workspace. For example, if the original
image was at `frames/scene_001.jpg` relative to the original workspace, the
undistorted image is at `frames/scene_001.jpg` relative to the new workspace.

This preserves the directory structure and means the `.sfmr` file's
`images/names.json.zst` can use the same relative paths.

### 3. Transformed `.sift` Files

For each image in the reconstruction, a new `.sift` file is written containing
geometrically transformed features. The `.sift` file follows the same format
as any other `.sift` file and is fully compatible with all tools that read
`.sift` files.

#### 3.1 Position Transform

Each keypoint position `(x, y)` in the distorted image is mapped to the
undistorted image:

```
(x_norm, y_norm) = distorted_camera.unproject(x, y)
(x', y') = pinhole_camera.project(x_norm, y_norm)
```

Keypoints that map outside the destination image bounds `[0, width) x [0, height)`
are dropped. This means the undistorted `.sift` file may have fewer features than
the original.

When we drop features, the feature indexes in the tracks change. We build a
remapping table per image: `old_feature_index -> new_feature_index` (or `None`
if dropped). The tracks in the `.sfmr` file must use the new indexes.
`observation_counts` must be updated, and 3D points that lose all observations
must be removed.

#### 3.2 Affine Shape Transform

Each keypoint has a 2x2 affine shape matrix `A` that maps a unit circle to the
keypoint's elliptical footprint in pixel space. Under undistortion, the affine
shape transforms as:

```
A' = J @ A
```

where `J` is the 2x2 Jacobian of the undistortion mapping at the keypoint
location. The Jacobian maps displacements in distorted pixel space to
displacements in undistorted pixel space.

The Jacobian is computed numerically via central differences at each keypoint:

```
eps = 0.5  # half-pixel step
J[0,0] = (map(x+eps, y)[0] - map(x-eps, y)[0]) / (2*eps)
J[0,1] = (map(x, y+eps)[0] - map(x, y-eps)[0]) / (2*eps)
J[1,0] = (map(x+eps, y)[1] - map(x-eps, y)[1]) / (2*eps)
J[1,1] = (map(x, y+eps)[1] - map(x, y-eps)[1]) / (2*eps)
```

where `map(x, y)` is `pinhole.project(distorted.unproject(x, y))`. This is
the same approach used by `WarpMap.compute_svd()` and works for all camera
models without needing analytic derivatives per model.

#### 3.3 Descriptors

Descriptors are kept unchanged. SIFT descriptors are designed to be robust to
small geometric transformations, and the undistortion-induced resampling
produces only minor local changes. The descriptors from the original image
remain valid for matching against the undistorted image.

#### 3.4 Thumbnail

The thumbnail is regenerated from the undistorted image (resize to 128x128
with bilinear interpolation).

#### 3.5 Metadata

The `.sift` metadata is updated:

| Field | Value |
|-------|-------|
| `image_name` | Same basename as original |
| `image_file_xxh128` | Hash of the undistorted image file |
| `image_file_size` | Size of the undistorted image file |
| `image_width` | Undistorted image width |
| `image_height` | Undistorted image height |
| `feature_count` | Number of features after dropping out-of-bounds ones |

The `feature_tool_metadata` records the transformation provenance:

```json
{
  "feature_tool": "sfmtool-undistort",
  "feature_type": "sift",
  "feature_options": {
    "source_feature_tool": "colmap",
    "source_feature_options": { ... },
    "fit": "inside",
    "filter": "aniso"
  }
}
```

### 4. Updated `.sfmr` Reconstruction

The output `.sfmr` file is derived from the input reconstruction with these
changes:

#### 4.1 Cameras

All cameras are replaced with their best-fit pinhole equivalents.

#### 4.2 Image Poses

Quaternions and translations are unchanged. Undistortion affects only the
camera intrinsics, not the extrinsics.

#### 4.3 Feature Hashes

`feature_tool_hashes` and `sift_content_hashes` are updated to reference the
new `.sift` files.

#### 4.4 Thumbnails

Regenerated from the undistorted images.

#### 4.5 Tracks

Track arrays (`image_indexes`, `feature_indexes`, `points3d_indexes`,
`observation_counts`) must be updated to account for dropped features:

1. Build a per-image feature index remapping table.
2. For each observation, look up the new feature index. If the feature was
   dropped, remove the observation.
3. If a 3D point loses all observations, remove the point (and renumber
   remaining points).
4. Recompute `observation_counts`.
5. Re-sort tracks lexicographically by `(points3d_index, image_index)`.

#### 4.6 Reprojection Errors

`SfmrReconstruction` has a `recompute_reprojection_errors()` method that reads
`.sift` files and reprojects 3D points through the camera model. Since the new
`.sift` files are written during the per-image loop (before the `.sfmr` is
assembled), we can call this method on the constructed reconstruction and it
will read the new undistorted feature positions and use the pinhole cameras.
No custom recomputation code is needed.

#### 4.7 Depth Statistics

Depth statistics and histograms must be recomputed because dropped 3D points
(those that lost all observations) change the per-image depth distributions.
`SfmrReconstruction` has a `recompute_depth_statistics()` method for this.

#### 4.8 Metadata

```json
{
  "operation": "undistort",
  "tool": "sfmtool",
  "tool_options": {
    "fit": "inside",
    "filter": "aniso",
    "source_sfmr": "sfmr/solve_001.sfmr"
  },
  "workspace": { ... }
}
```

### 5. Feature Dropping: How Much Do We Lose?

With `--fit inside`, no pinhole boundary pixels map outside the source, so
features near the center are safe. Features near the extreme corners of the
distorted image may map outside the pinhole frame and get dropped. The amount
depends on the distortion magnitude.

With `--fit outside`, the pinhole frame is wider than the source, so *all*
distorted feature positions should map to valid undistorted positions (none
dropped). The tradeoff is black borders in the undistorted images.

After all images are processed, print a summary of dropped features:
total features dropped, percentage of total, and the image with the highest
drop rate. This is informational — no threshold or warning level needed.

### 6. Rig Data

If the source reconstruction has rig data (`rigs/` and `frames/` sections),
it should be carried forward unchanged. The rig defines relative poses between
sensors, which are unaffected by undistortion. The `sensor_camera_indexes`
must be updated to point to the new (pinhole) camera indices.

### 7. Per-Image Processing Pipeline

All per-image work happens in a single pass over the image list. For each
image, the pipeline does everything that needs that image's pixel data before
moving on to the next image. This avoids re-reading images from disk:

```
for each image:
    1. Load the original image from disk
    2. Warp to produce the undistorted image (WarpMap + remap)
    3. Save the undistorted image to the output workspace
    4. Generate the 128x128 thumbnail from the in-memory undistorted image
    5. Read the original .sift file
    6. Transform feature positions and affine shapes (using the camera models)
    7. Drop features outside the pinhole frame; build the feature index remap
    8. Write the new .sift file (with transformed features and the thumbnail from step 4)
    9. Compute the image file hash (from the bytes written in step 3)
```

After all images are processed:
- Build the track remapping from the per-image feature index remaps
- Assemble the SfmrReconstruction (pinhole cameras, same poses, remapped tracks)
- Call `recompute_reprojection_errors()` (reads the new .sift files just written)
- Call `recompute_depth_statistics()` (accounts for dropped 3D points)
- Save the .sfmr file

The thumbnail in step 4 is produced by resizing the in-memory undistorted
image to 128x128 (bilinear interpolation), not by re-reading the saved file.
The `.sift` file written in step 8 embeds this thumbnail.

### 8. Implementation Phases

This could be implemented incrementally:

**Phase 1** (current): Undistort images, write JSON metadata. No workspace,
no `.sfmr`, no `.sift` files.

**Phase 2**: Add `.sfmr` output with updated cameras, same poses, same 3D
points. Zero out feature hashes. Keep tracks with original feature indexes
(stale but structurally valid — image/point associations are still correct).

**Phase 3**: Add `.sift` file generation with transformed positions and affine
shapes. Update feature hashes in `.sfmr`. Properly remap track feature indexes.
Initialize output as a full workspace.

Phase 2 is useful on its own — the `.sfmr` file can be opened in the GUI
viewer, inspected, aligned, merged, etc. The only thing that doesn't work is
looking up individual features in `.sift` files, which is a niche operation.

## Resolved Questions

1. **Feature dropping and track cleanup**: 3D points that lose all observations
   are dropped. Track arrays, observation counts, and point indexes are all
   cleaned up and renumbered.

2. **Workspace initialization**: Write `.sfm-workspace.json` directly rather
   than calling `sfm init`. This avoids a dependency on the CLI layer from the
   library layer.

3. **Implementation order**: Implement incrementally in an order that allows
   good testing and correctness assurance at each step. Each phase should be
   independently testable before moving to the next.
