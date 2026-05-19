# `sfm camrig` Command

## Overview

Builds `.camrig` camera rig files. `camrig` is a command group with three
subcommands:

- `create` — build a one-camera rig for a directory of images and write it
  to a `.camrig` file.
- `cp` — build a `.camrig` by copying a rig, a single camera, or a subset of
  sensors out of an existing `.sfmr` reconstruction or `.camrig` file.
- `spherical-tiles` — build a spherical tile rig (a sphere discretised into
  co-centric pinhole tiles) and write it to a `.camrig` file.

`create` builds a rig from a directory of images on disk; `cp` builds one
from an existing file. Both write a `.camrig`.

To inspect a `.camrig` file, use `sfm inspect <FILE.camrig>` (see
`specs/cli/inspect-command.md`).

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

## `sfm camrig cp`

### Syntax

```bash
sfm camrig cp <SOURCE> <OUTPUT_FILE> [SELECTOR] [OPTIONS...]
```

Builds a `.camrig` by copying from an existing file. `SOURCE` is either a
`.sfmr` reconstruction or another `.camrig` file; the output is always a
`.camrig`. Where `create` reads a directory of images, `cp` reads a file that
already carries cameras (and, for a rig, sensor poses) — a solved
reconstruction or a previously built rig.

### Selectors

A *selector* says which slice of the source becomes the output rig. The
selectors available depend on the source type, and at most one may be given.

| Selector | Source | Result |
|----------|--------|--------|
| `--rig N` | `.sfmr` | The whole of rig `N` — a multi-sensor `.camrig` preserving the rig's sensors, cameras, and `sensor_from_rig` poses. |
| `--camera N` | `.sfmr` or `.camrig` | Camera `N` from the source's camera pool, on its own — a single-sensor `generic` rig at the identity pose. |
| `--sensors RANGE` | `.camrig` | The selected subset of the source's sensors — itself a `.camrig`, with the camera pool reduced to the cameras those sensors use. |
| *(none)* | `.sfmr` | Defaults to `--rig 0` when the reconstruction has exactly one rig, or `--camera 0` when it has no rig data and exactly one camera; otherwise the command asks for a selector. |
| *(none)* | `.camrig` | Copies the whole rig unchanged. |

`--rig` is rejected for a `.camrig` source (a `.camrig` holds exactly one
rig — use `--sensors`); `--sensors` is rejected for a `.sfmr` source (use
`--rig`).

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `SOURCE` | path | required | The `.sfmr` or `.camrig` file to copy from. |
| `OUTPUT_FILE` | path | required | Path of the `.camrig` file to write. Its directory is the rig root. |
| `--rig` | int | — | `.sfmr` only. Index of the rig to copy. |
| `--camera` | int | — | Index into the source's camera pool. Copies that one camera as a single-sensor rig. |
| `--sensors` | range | — | `.camrig` only. Sensor indices to keep, as an integer range expression (e.g. `0-2`, `0,2,4`). |
| `--pattern` | pattern | — | Image pattern for the output sensor. Only valid with `--camera` (single-sensor output); for multi-sensor results the per-sensor patterns are inferred (see below). |
| `--name` | str | (see below) | Rig name stored in the file. Defaults to the source rig's name, then to the output file stem. |

### Image patterns

A `.camrig` is located in a workspace by each sensor's image pattern, so `cp`
fills patterns in by default:

- **`.camrig` source** — the source already stores per-sensor patterns; the
  kept sensors keep theirs verbatim.
- **`.sfmr` source** — a reconstruction has no patterns. For each output
  sensor, `cp` collects the file names of that sensor's images and infers a
  pattern from them with the same path-sequence summariser the rest of the
  CLI uses (`summarize_paths_by_sequence`). A multi-sensor `.camrig` requires
  every pattern to carry a frame field; if inference cannot produce one for
  every sensor, the whole rig is written **geometry-only** (empty patterns,
  describing pure geometry) and a note is printed. A single-sensor result
  (`--camera`) may instead be given an explicit `--pattern`; without it the
  pattern is inferred, and `cp` fails if it cannot.

Inferred patterns are relative to whatever the source's image names are
relative to. They are only correct if the `.camrig` is placed at that same
root — normally the workspace root.

### Behaviour

The output is written with `rig_type` `generic`, except when the whole of a
`.camrig` is copied (no selector, or `--sensors` selecting every sensor), in
which case the source's `rig_type` and `rig_attributes` are preserved. A
subset of a typed rig (e.g. three faces of a `cubemap`) is no longer that
type, so it becomes `generic` with empty attributes.

`sensor_from_rig` poses are copied verbatim — the rig frame is arbitrary, so
a sensor subset needs no rebasing. `--camera` produces a lone sensor at the
identity pose.

### Usage Examples

```bash
# Harvest refined intrinsics from a solved reconstruction, pattern inferred
sfm camrig cp sfmr/solve_001.sfmr photos.camrig --camera 0

# Same, but state the pattern explicitly so the rig can be dropped elsewhere
sfm camrig cp sfmr/solve_001.sfmr rig.camrig --camera 0 --pattern '*.jpg'

# Copy a solved rig (its sensors, cameras, and poses) into a reusable .camrig
sfm camrig cp sfmr/rig_solve.sfmr studio_rig.camrig --rig 0

# Take three faces out of a six-face cubemap rig
sfm camrig cp cubemap.camrig front_faces.camrig --sensors 0-2
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

## Usage Examples

```bash
# Build a 1280-tile rig sized for a 1024-wide equirectangular target
sfm camrig spherical-tiles tiles.camrig --n 1280 --equirect-width 1024

# Build a reproducible rig with an explicit per-tile pixel size and centre
sfm camrig spherical-tiles tiles.camrig --n 320 --arc-per-pixel 0.012 \
    --overlap-factor 1.2 --centre 0 1.5 0 --seed 42

# Verify and inspect a .camrig file
sfm inspect tiles.camrig
```
