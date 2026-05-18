# `sfm camrig` Command

## Overview

Builds and inspects `.camrig` camera rig files. `camrig` is a command group
with three subcommands:

- `create` — build a one-camera rig for a directory of images and write it
  to a `.camrig` file.
- `spherical-tiles` — build a spherical tile rig (a sphere discretised into
  co-centric pinhole tiles) and write it to a `.camrig` file.
- `inspect` — verify a `.camrig` file's integrity and print its metadata.

See `specs/formats/camrig-file-format.md` for the file format and
`specs/core/spherical-tiles-rig.md` for the spherical tile rig.

## `sfm camrig create`

### Syntax

```bash
sfm camrig create <OUTPUT_FILE> <IMAGE_PATTERN> [OPTIONS...]
```

Builds a single-sensor, single-camera `.camrig` for a directory of images.
The intended use is dropping a `.camrig` file alongside a folder of photos:
the rig describes one camera, and `IMAGE_PATTERN` (a `.camrig` image pattern
— globs and/or `%d`-style frame fields, see *How `.camrig` files fit into
workspaces* in the format spec) is stored verbatim as that sensor's image
pattern. `create` matches the pattern with the same semantics `sfm solve`
later applies to it, so what `create` scans is exactly what `solve` covers.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `OUTPUT_FILE` | path | required | Path of the `.camrig` file to write. Its directory is the rig root. |
| `IMAGE_PATTERN` | pattern | required | `.camrig` image pattern identifying the images, relative to the rig root (e.g. `*.jpg`, `images/**/*.png`, `cam_%04d.jpg`). Stored verbatim as the sensor's image pattern. |
| `--camera-model` | choice | — | COLMAP camera model. Overrides the EXIF-inferred model; required with `--params`. |
| `--resolution` | `WxH` | — | Image resolution as `WIDTHxHEIGHT`. Every matched image must have it. |
| `--focal-length` | float | — | Focal length in pixels; sets both `fx` and `fy`. |
| `--focal-length-x` | float | — | Focal length `fx` in pixels. |
| `--focal-length-y` | float | — | Focal length `fy` in pixels. |
| `--principal-point-x` | float | — | Principal point `cx` in pixels. |
| `--principal-point-y` | float | — | Principal point `cy` in pixels. |
| `--params` | list | — | Full parameter list in COLMAP order, comma-separated. Requires `--camera-model`; mutually exclusive with the named focal/principal options. |
| `--name` | str | output file stem | Rig name stored in the file. |

### Behaviour

The pattern is matched relative to the rig root: a frame field (`%d`, `%0Nd`)
matches digit-named frames only, and `*` / `**` respect path-segment
boundaries. The set of matched files is the sensor's images; every match must
be an image file, or the command fails and asks the caller to narrow the
pattern.

Intrinsics are derived heuristically with pycolmap's EXIF inference
(`infer_camera_from_image`), one inference per image. `--camera-model`
forces the model. `--params` skips inference entirely and takes the camera
parameters directly (focal length, principal point, distortion) in COLMAP
order. The named `--focal-length*` / `--principal-point*` options override
individual inferred values; `--resolution` overrides the resolution.

The written rig is a `generic` rig: one camera, one sensor at the identity
`sensor_from_rig` pose, and the pattern stored as the sensor's image pattern.

### Consistency checks

A `.camrig` from `create` describes **one** camera, so the matched images
must be consistent. The command fails — with a message naming the offending
images so the caller can split the set into separate rigs — when:

- the images have **mixed resolutions** (and `--resolution` is not given, or
  some image does not match the given resolution);
- pycolmap infers **different camera models** across images (resolve with
  `--camera-model`);
- EXIF-inferred **focal lengths vary** by more than 1% across images (they
  look like different cameras or zoom settings).

### Usage Examples

```bash
# Drop a .camrig beside a folder of photos, intrinsics inferred from EXIF
sfm camrig create my_images.camrig 'images/*'

# Force a fisheye model, let pycolmap infer the focal length
sfm camrig create rig.camrig '*.jpg' --camera-model OPENCV_FISHEYE

# Explicit OpenCV calibration in COLMAP parameter order
sfm camrig create rig.camrig '*.jpg' --camera-model OPENCV \
    --resolution 4000x3000 \
    --params 2800,2800,2000,1500,-0.08,0.01,0,0
```

## `sfm camrig spherical-tiles`

### Syntax

```bash
sfm camrig spherical-tiles <OUTPUT_FILE> --n <N> (--equirect-width <W> | --arc-per-pixel <R>) [OPTIONS...]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `OUTPUT_FILE` | path | required | Path of the `.camrig` file to write |
| `--n` | int | required | Number of tiles in the rig (`>= 2`) |
| `--equirect-width` | int | — | Target equirectangular width in pixels; sets the per-tile pixel size to `2π / width` |
| `--arc-per-pixel` | float | — | Angular size of one tile pixel, in radians |
| `--overlap-factor` | float | 1.15 | Tile FOV safety margin over the worst-case gap between tiles |
| `--centre` | 3 floats | `0 0 0` | Rig optical centre in world space |
| `--atlas-cols` | int | `ceil(sqrt(n))` | Atlas column count |
| `--seed` | int | unseeded | Relaxer seed for reproducible tile placement |
| `--name` | str | output file stem | Rig name stored in the file |

Exactly one of `--equirect-width` and `--arc-per-pixel` must be given — they
are two ways of expressing the same per-tile angular resolution.

### Behaviour

The tile look directions come from a Thomson-relaxed sphere-point set; the
half-FOV, patch size, and atlas packing are derived from the measured tile
spacing (see `specs/core/spherical-tiles-rig.md`). Pass `--seed` for a
bit-for-bit reproducible layout. The written `.camrig` file is a
`spherical_tiles` rig: one shared pinhole camera, all-zero translations, one
`sensor_from_rig` quaternion per tile, and the derived scalars in
`rig_attributes`.

## `sfm camrig inspect`

### Syntax

```bash
sfm camrig inspect <CAMRIG_FILE>
```

Recomputes the file's content hashes, checks its structural constraints, and
prints the rig metadata: format version, name, rig type, sensor and camera
counts, `rig_attributes`, and the content hash. Exits with an error if
verification fails. Works for any `.camrig` file, not only spherical tile
rigs.

## Usage Examples

```bash
# Build a 1280-tile rig sized for a 1024-wide equirectangular target
sfm camrig spherical-tiles tiles.camrig --n 1280 --equirect-width 1024

# Build a reproducible rig with an explicit per-tile pixel size and centre
sfm camrig spherical-tiles tiles.camrig --n 320 --arc-per-pixel 0.012 \
    --overlap-factor 1.2 --centre 0 1.5 0 --seed 42

# Verify and inspect a .camrig file
sfm camrig inspect tiles.camrig
```
