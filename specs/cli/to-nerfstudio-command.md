# `sfm to-nerfstudio` Command

## Overview

Converts a pinhole `.sfmr` workspace (typically produced by `sfm undistort`)
into the directory layout expected by [Nerfstudio](https://docs.nerf.studio/),
so the output is directly trainable with `ns-train` without any further
processing:

```bash
sfm undistort source.sfmr
sfm to-nerfstudio source_undistorted/sfmr/undistorted.sfmr -o my_dataset/
ns-train nerfacto --data my_dataset/
ns-train splatfacto --data my_dataset/
```

The conversion is purely a re-packaging step. Extrinsics, intrinsics, and 3D
points are read from the input reconstruction and re-emitted in nerfstudio's
coordinate convention and JSON layout. No new SfM work is performed.

The output matches what `ns-process-data` produces for the parts
nerfstudio reads (schema, coordinate frame, PLY structure).

## Command Syntax

```bash
sfm to-nerfstudio <INPUT.sfmr> [--output <OUTPUT_DIR>] [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | `{stem}_nerfstudio/` next to input | Output dataset directory |
| `--num-downscales` | int ≥ 0 | 3 | Number of downsampled image pyramid levels (0 disables the pyramid) |
| `--jpeg-quality` | int 1–100 | 95 | JPEG quality for downsampled pyramid images |
| `--include-colmap` | flag | off | Also emit a `sparse/` directory with COLMAP `.bin` files |

Input cameras must all be pinhole (zero distortion). The command rejects any
reconstruction with a distorted camera and tells the user to run
`sfm undistort` first.

## Output Structure

```
{output_dir}/
├── transforms.json
├── sparse_pc.ply
├── images/
│   ├── <original_basename_or_frame_00001>.jpg
│   └── ...
├── images_2/        (width/2, height/2)
├── images_4/        (width/4, height/4)
├── images_8/        (width/8, height/8)
└── sparse/                  (only with --include-colmap)
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    ├── rigs.bin
    └── frames.bin
```

The minimum nerfstudio needs to train is `transforms.json` plus `images/`.
`sparse_pc.ply` seeds initial Gaussians for `splatfacto` and is used for
viewer point-cloud display. The `images_N/` pyramids let nerfstudio's
`downscale_factor` setting load pre-resampled images instead of resizing on
the fly.

## How It Works

### 1. Coordinate Conversion

COLMAP / sfmtool stores extrinsics as **camera-from-world** in OpenCV axes
(+x right, +y down, +z forward). Nerfstudio expects **world-from-camera** in
OpenGL axes (+x right, +y up, +z back).

For each image:

1. Build the 4×4 camera-from-world matrix from the WXYZ quaternion and
   translation in the reconstruction.
2. Invert to get world-from-camera.
3. Negate the Y and Z columns (OpenCV → OpenGL camera axes).
4. Left-multiply by `applied_transform` (the world-axis remap below) so the
   serialized matrix is in post-applied space, matching what
   `ns-process-data` emits.

The `applied_transform` (3×4) we emit is:

```
[[ 1,  0, 0, 0],
 [ 0,  0, 1, 0],
 [ 0, -1, 0, 0]]
```

This permutes Y↔Z and negates Y. The same transform is applied to the
point cloud positions before they are written to `sparse_pc.ply`, so cameras
and points stay aligned.

Implementation: `frame_transform_matrix` and `apply_transform_to_points` in
`src/sfmtool/_to_nerfstudio.py`. Quaternion → rotation matrix uses
`RotQuaternion.to_rotation_matrix` from the Rust bindings.

### 2. `transforms.json`

Built by `build_transforms_json` in `src/sfmtool/_to_nerfstudio.py`.

Single-camera reconstructions hoist intrinsics to the top level (matches
`ns-process-data` output):

```json
{
  "w": 2007,
  "h": 1511,
  "fl_x": 1471.812,
  "fl_y": 1471.812,
  "cx": 1003.5,
  "cy": 755.5,
  "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
  "camera_model": "OPENCV",
  "frames": [
    {
      "file_path": "images/<basename>.jpg",
      "transform_matrix": [[...], [...], [...], [0, 0, 0, 1]],
      "colmap_im_id": 1
    }
  ],
  "applied_transform": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]],
  "ply_file_path": "sparse_pc.ply"
}
```

Multi-camera reconstructions move intrinsics into each frame entry instead
of the top level. `applied_transform` and `ply_file_path` are always at the
top level.

Field notes:

- `camera_model` is always `"OPENCV"` with `k1=k2=p1=p2=0.0`. `"PINHOLE"`
  would also work but `OPENCV` matches `ns-process-data` exactly and
  round-trips through more downstream tools.
- `frames` is ordered by image index in the source reconstruction.
- `colmap_im_id` is 1-based and contiguous, matching what
  `to-colmap-bin`'s `images.bin` would assign.
- Single- vs multi-camera is auto-detected via
  `len(unique(camera_indexes)) == 1`.

### 3. `sparse_pc.ply`

ASCII PLY with one vertex element carrying x/y/z floats and r/g/b uint8
colors:

```
ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uint8 red
property uint8 green
property uint8 blue
end_header
{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}
...
```

Point positions come from `recon.positions`, colors from `recon.colors`.
Positions are pre-transformed by `applied_transform` so they share a frame
with the camera poses.

Hand-rolled writer (`write_sparse_ply`) — the format has one element type
and six fixed properties, so an external PLY library would add more
indirection than code. The reference `ns-process-data` output is also
`format ascii 1.0`, so we match it on disk.

### 4. Image Pyramid

For each level `i` in `1..=num_downscales`, the originals are downsampled by
`2**i` with `cv2.INTER_AREA` (better antialiasing than bilinear for
shrinking) and written to `images_{2**i}/` as JPEG at `--jpeg-quality`. Each
source image is read once and emits all pyramid levels in a single pass.

`--num-downscales 0` skips the pyramid entirely; nerfstudio will resample on
the fly.

### 5. Image Placement and Filenames

Originals are copied into `images/` with `shutil.copy2` (preserves mtime).
The destination basename is the source image's basename (`Path(name).name`,
dropping any subdirectory prefix).

If two source images would collapse to the same destination basename
(e.g., `cam0/img.jpg` and `cam1/img.jpg`), the command errors out. We never
duplicate files; the duplicate-naming behavior in `ns-process-data`'s
output is not replicated.

### 6. COLMAP Export (`--include-colmap`)

When set, the command writes `sparse/{cameras,images,points3D,rigs,frames}.bin`
into the output directory using the existing `_colmap_io.save_colmap_binary`
function (the same code path as `sfm to-colmap-bin`). Off by default —
nerfstudio doesn't read these files for training, and skipping them keeps
the output smaller.

## Module Layout

| File | Responsibility |
|------|----------------|
| `src/sfmtool/_commands/to_nerfstudio.py` | Click CLI shim (option parsing, .sfmr load, error wrapping) |
| `src/sfmtool/_to_nerfstudio.py` | Coordinate conversion, `transforms.json` builder, PLY writer, image placement, pyramid generation, top-level `export_to_nerfstudio` orchestrator |
| `src/sfmtool/_colmap_io.py` (existing) | `save_colmap_binary` is reused for `--include-colmap` |
| `tests/test_to_nerfstudio.py` | Unit tests for coordinate conversion, PLY writer, filename strategy |
| `tests/test_cli_to_nerfstudio.py` | End-to-end CLI tests (uses `pinhole_sfmr_17_images` fixture: undistort + to-nerfstudio over the seoul_bull dataset) |

## Out of Scope

- **Mask images.** `.sfmr` does not carry masks today. If/when it does,
  `mask_path` per frame can be added.
