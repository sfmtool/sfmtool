# `sfm to-colmap-bin` Command

## Overview

`sfm to-colmap-bin` writes a reconstruction stored in a `.sfmr` file as COLMAP's
five-file binary sparse model, so it opens in the COLMAP GUI or any tool that reads that
format; `--range` exports a subset of images.

## Coordinate Convention

This command is a convention boundary: `.sfmr` data is stored in the
canonical Z-up / −Z-forward convention, while COLMAP binary files use
COLMAP's +Z-forward, Y-down convention. On export the canonical→COLMAP
conversion is applied — the camera-frame flip `S` on every pose (including
rig `sensor_from_rig` poses) and the inverse world canonicalization `W⁻¹`
on world-space data — so the written `.bin` files are genuine
COLMAP-convention data. `sfm from-colmap-bin` applies the forward
conversion, so an export/import round trip is stable. See the "Coordinate
System Conventions" section of
[`sfmr-file-format.md`](../../formats/sfmr-file-format.md) for the transform
definitions.

## Command Syntax

```bash
sfm to-colmap-bin <INPUT.sfmr> <OUTPUT_DIR> [OPTIONS]
```

`INPUT.sfmr` must exist and have the `.sfmr` extension (compared
case-insensitively); any other extension is a usage error.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--range / -r` | range expression | (none) | Export only images whose file number matches the expression. Observations on excluded images are dropped. |
| `--filter-points` | flag | off | With `--range`, also drop 3D points that have no remaining observations. Default is to keep all 3D points. |

`--filter-points` is only meaningful together with `--range`; supplying it without
`--range` is an error. The range grammar matches `sfm sift -r`, `sfm match -r`,
`sfm solve -r`, and `sfm xform --include-range` (parsed with
`sfmtool.RangeExpr`, matched against file numbers recovered by
`number_from_filename`).

## Output Files

```
output_dir/
  cameras.bin
  images.bin
  points3D.bin
  rigs.bin       (always written; synthesized implicit values when no rig data)
  frames.bin     (always written; synthesized implicit values when no rig data)
```

## Feature sources

COLMAP's `images.bin` stores, per image, a list of 2D keypoints that the
observations index into. How the export builds that list depends on the
reconstruction's feature source:

- **`sift_files`.** The keypoint positions are read from each image's `.sift`
  file in the workspace, so those `.sift` files must exist. The observations
  keep their feature indices into the `.sift` files.
- **`embedded_patches`.** The keypoints are stored inline, one per observation,
  and no `.sift` file is read. The export gives each image's observations dense
  feature indices `0..n` that exist only in the written `images.bin`; they do
  not correspond to any `.sift` file.

## Points at infinity

COLMAP stores every 3D point as a finite `(x, y, z)`. Points at infinity
(`w = 0`) in the input are placed at a finite depth before writing, far enough
that their parallax is at most one pixel in every camera that observes them, and
the command prints how many points it moved. A reconstruction without points at
infinity is written unchanged.

## Range Semantics

With `N` images in the input and `K` kept by `--range`:

1. **Images.** Keep the `K` in their original relative order. Image IDs in
   `images.bin` are 1-based and contiguous over the kept set.
2. **Observations (tracks).** Keep every observation whose image is kept.
   For a `sift_files` reconstruction, feature indices within the kept images
   are preserved (see [Feature sources](#feature-sources)).
3. **3D points — default.** Keep every point. A point whose entire track
   referenced removed images becomes a point with zero observations in
   `points3D.bin` (track length 0, which is a normal in-format value). The
   point ID space stays 1:1 with the input, making side-by-side comparison
   against the original reconstruction straightforward.
4. **3D points — `--filter-points`.** Drop points with zero remaining
   observations, remap point IDs to be contiguous. Matches
   `xform --include-range` behavior.
5. **Cameras.** Kept unchanged. Unused cameras are not pruned.
6. **Rig / frame data.** Frames that contain no kept image are dropped.
   Rigs and sensors are unchanged.

If `--range` matches no images, the command errors out with the available file
numbers listed.

## Usage Examples

```bash
# Export the full reconstruction for the COLMAP GUI
sfm to-colmap-bin sfmr/solve_001.sfmr colmap_export/
colmap gui --import_path colmap_export/ --image_path images/

# Export only images 10-50, keeping the full 3D point cloud
sfm to-colmap-bin sfmr/solve_001.sfmr colmap_export/ -r 10-50

# Same subset, but prune 3D points that no longer have any observations
sfm to-colmap-bin sfmr/solve_001.sfmr colmap_export/ -r 10-50 --filter-points
```

## Implementation

The image-subset logic lives in Rust, on `SfmrReconstruction`:

```rust
pub fn subset_by_image_indices(
    &self,
    image_indices: &[u32],
    drop_orphaned_points: bool,
) -> Result<Self, String>;
```

exposed to Python as `PySfmrReconstruction.subset_by_image_indices`. It
handles image/thumbnail/depth-stat filtering, track filtering with image
index remapping, optional orphaned-point removal with contiguous point ID
remapping, and rig/frame filtering (dropping frames with no remaining
images and remapping frame indices).

The CLI shim, [`to_colmap_bin.py`](../../../src/sfmtool/_commands/to_colmap_bin.py),
loads the input and, when `--range` is given, calls `apply_range_filter` from
[`_range_options.py`](../../../src/sfmtool/_commands/_range_options.py), which
it shares with `sfm to-nerfstudio`. That helper parses `--range` with
`RangeExpr`, resolves file numbers to image indices via `number_from_filename`,
and calls `subset_by_image_indices`. The shim then hands the result to
`save_colmap_binary` in [`colmap/io.py`](../../../src/sfmtool/colmap/io.py),
which calls `materialize_infinity_for_export` before converting and writing.

## Testing

CLI and end-to-end tests are in
[`tests/test_colmap_interop.py`](../../../tests/test_colmap_interop.py): the
non-`.sfmr` input error, a full export, points at infinity, `--range` with and
without `--filter-points`, and the range error cases.
