# `sfm to-colmap-bin` Command

## Overview

Exports a `.sfmr` reconstruction to COLMAP binary format for use with the COLMAP GUI or
other COLMAP-compatible tools.

## Command Syntax

```bash
sfm to-colmap-bin <INPUT.sfmr> <OUTPUT_DIR> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--range / -r` | range expression | (none) | Export only images whose file number matches the expression. Observations on excluded images are dropped. |
| `--filter-points` | flag | off | With `--range`, also drop 3D points that have no remaining observations. Default is to keep all 3D points. |

`--filter-points` is only meaningful together with `--range`; supplying it without
`--range` is an error. The range grammar matches `sfm sift -r`, `sfm match -r`,
`sfm solve -r`, and `sfm xform --include-range` (parsed with
`openjd.model.IntRangeExpr`, matched against file numbers recovered by
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

## Range Semantics

With `N` images in the input and `K` kept by `--range`:

1. **Images.** Keep the `K` in their original relative order. Image IDs in
   `images.bin` are 1-based and contiguous over the kept set.
2. **Observations (tracks).** Keep every observation whose image is kept.
   Feature indices within the kept images are preserved.
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

The CLI shim (`src/sfmtool/_commands/to_colmap_bin.py`) parses `--range`
with `IntRangeExpr`, resolves file numbers to image indices via
`number_from_filename`, calls `subset_by_image_indices`, and hands the
result to `save_colmap_binary`. No changes to `_colmap_io.save_colmap_binary`
are required — it operates on whatever reconstruction it is handed.
