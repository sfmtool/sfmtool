# `sfm panorama` Command

## Overview

Renders an equirectangular panorama from a posed reconstruction. The source
images are composited onto a sphere of spherical tiles; for each tile a
photometric RANSAC selects the agreeing cluster of contributing sources and
collapses it to a per-pixel consensus, and the assembled atlas is resampled
through a full-sphere equirectangular camera.

This command is the first-class surface for the spherical-tile rendering
pipeline. The heavy lifting lives in the Rust core — see
[`specs/core/spherical-tiles-rig.md`](../core/spherical-tiles-rig.md),
[`specs/core/per-spherical-tile-source-stack.md`](../core/per-spherical-tile-source-stack.md),
[`specs/core/tile-batched-consensus-atlas.md`](../core/tile-batched-consensus-atlas.md),
and [`specs/drafts/photometric-subsets-ransac.md`](../drafts/photometric-subsets-ransac.md).

## Command Syntax

```bash
sfm panorama <RECONSTRUCTION.sfmr> --output <PANO.png> [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output panorama image path (e.g. `pano.png`) |
| `--image-dir` | path | reconstruction's workspace dir | Directory the reconstruction's image names are relative to |
| `--range / -r` | string | all | Composite only images whose file number matches this range expression (e.g. `10-50` or `0-9,20-29`) |
| `--near-image` | string | none | Composite only images spatially near this reference image (matched by exact name or path suffix). Requires `--near-count` and/or `--near-radius` |
| `--near-count` | int | none | With `--near-image`, keep the N images whose cameras are closest to the reference (reference always included) |
| `--near-radius` | float | none | With `--near-image`, keep images whose cameras are within this world-space distance of the reference |
| `--equirect-width` | int | 2160 | Output width in pixels; height is `width / 2`. Must be a positive even integer |
| `--n-tiles` | int | 320 | Number of spherical tiles in the rig |
| `--batch-size` | int | 32 | Tiles composited per batch; smaller bounds peak memory |
| `--dtype` | `float32` \| `float16` | `float32` | Per-batch stack storage; `float16` halves memory at some precision cost |
| `-k` | int | 1 | Nearest tiles blended when resampling; `k = 1` is closest-tile |
| `--seed` | int | 1234 | Rig relaxer seed |
| `--inlier-threshold` | float | 8.0 | Photometric RANSAC inlier threshold (luma units) |
| `--gamma` | float | 1.0 | Photometric RANSAC tone exponent |
| `--ransac-seed` | int | 0 | Photometric RANSAC seed |

## Pipeline

1. Build a `SphericalTileRig` whose angular resolution (`2π / equirect-width`)
   maps roughly one atlas sample per output pixel along the equator. The rig's
   patch size is rounded up to the next power of two.
2. Load each image as RGB from `image-dir / <image_name>`, paired with the
   reconstruction's per-image camera intrinsics and rotation.
3. Composite a per-tile consensus atlas in tile batches
   (`render_consensus_atlas`); the photometric RANSAC selects the agreeing
   cluster per tile.
4. Resample the atlas through a full-sphere equirectangular camera
   (`SphericalTileRig.resample_atlas`).
5. Uncovered samples are left as `NaN` through resampling and flattened to
   black at the final 8-bit image write.

Only the per-image rotation is used (rotation-only sources), so the result is a
view-direction panorama centered on the reconstruction's world frame.

## Source subsetting

By default every image in the reconstruction is composited. Two filters narrow
the source set before rendering — useful for *moving* captures, where
compositing the whole trajectory blends many viewpoints and the per-tile
photometric RANSAC only partly hides the resulting parallax:

- **`--range`** selects by image file number (same expression grammar as other
  commands), e.g. a contiguous slice of a sequential capture.
- **`--near-image REF`** keeps only images whose camera centers are spatially
  near `REF`'s camera center — `--near-radius` keeps everything within a
  world-space distance, `--near-count` keeps the N nearest. The two may be
  combined (radius first, then capped to the count); the reference image is
  always retained. `REF` is matched by exact name or by path suffix, so for a
  rig pass the sensor-qualified path (`fisheye_left/frame_000500.jpg`) to
  disambiguate frames shared across sensors.

When both filters are given, `--range` is applied first and `--near-image` then
operates on the survivors (it errors if the reference falls outside the range).
The camera center for image *i* is `C = -R(qᵢ)ᵀ · tᵢ` from the stored pose; the
filters touch only which images contribute, never the geometry.

## Usage Examples

```bash
# Render a 2160x1080 panorama next to the reconstruction
sfm panorama result.sfmr -o pano.png

# Higher resolution with an explicit image directory
sfm panorama result.sfmr -o pano.png --equirect-width 4096 --image-dir images/

# Lower memory: smaller batches and half-precision stack storage
sfm panorama result.sfmr -o pano.png --batch-size 16 --dtype float16

# Local panorama from the 200 cameras nearest one frame (less parallax)
sfm panorama result.sfmr -o local.png \
    --near-image fisheye_left/frame_000500.jpg --near-count 200

# Restrict to a contiguous slice of a sequential capture
sfm panorama result.sfmr -o slice.png --range 400-600
```

## Implementation

The CLI shim is `src/sfmtool/_commands/panorama.py`; the reusable pipeline is
`render_equirect_panorama` in `src/sfmtool/_panorama.py`, which builds the rig
(`build_panorama_rig`), loads the sources (`load_sources`), runs
`render_consensus_atlas`, and resamples via `resample_atlas_to_equirect`
(`src/sfmtool/_spherical_tile_rig.py`). Source subsetting is computed by
`select_source_indices` (also in `_panorama.py`) and applied with
`SfmrReconstruction.subset_by_image_indices` before rendering.
