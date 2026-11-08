# `sfm camrig` Command

## Overview

Builds `.camrig` camera rig files. `camrig` is a command group with three
subcommands:

- `create` — build a one-camera rig for a directory of images and write it
  to a `.camrig` file.
- `cp` — build a `.camrig` by copying a rig, a single camera, or a subset of
  sensors out of an existing `.sfmr` reconstruction or `.camrig` file, or by
  converting a rig of a workspace's `rig_config.json`.
- `spherical-tiles` — build a spherical tile rig (a sphere discretised into
  co-centric pinhole tiles) and write it to a `.camrig` file.

`create` builds a rig from a directory of images on disk; `cp` builds one
from an existing file. Both write a `.camrig`.

To inspect a `.camrig` file, use `sfm inspect <FILE.camrig>` (see
`specs/cli/reconstruction/inspect-command.md`).

See `specs/formats/camrig-file-format.md` for the file format and
`specs/core/spherical/spherical-tiles-rig.md` for the spherical tile rig.

## `sfm camrig create`

### Syntax

```bash
sfm camrig create <IMAGE_PATTERN> <OUTPUT_FILE> [OPTIONS...]
```

The input pattern comes first and the output file second, matching the
`INPUT … OUTPUT` ordering used by `xform`, `densify`, `to-colmap-bin`, and the
sibling `camrig cp`.

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
| `IMAGE_PATTERN` | pattern | required | `.camrig` image pattern identifying the images, relative to the rig root (e.g. `*.jpg`, `images/**/*.png`, `cam_%04d.jpg`). Stored verbatim as the sensor's image pattern. |
| `OUTPUT_FILE` | path | required | Path of the `.camrig` file to write. Its directory is the rig root. |
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
sfm camrig create 'images/*' my_images.camrig

# Force a fisheye model, let pycolmap infer the focal length
sfm camrig create '*.jpg' rig.camrig --camera-model OPENCV_FISHEYE

# Explicit OpenCV calibration in COLMAP parameter order
sfm camrig create '*.jpg' rig.camrig --camera-model OPENCV \
    --resolution 4000x3000 \
    --params 2800,2800,2000,1500,-0.08,0.01,0,0
```

## `sfm camrig cp`

### Syntax

```bash
sfm camrig cp <SOURCE> <OUTPUT_FILE> [SELECTOR] [OPTIONS...]
```

Builds a `.camrig` by copying from an existing file. `SOURCE` is a `.sfmr`
reconstruction, another `.camrig` file, or a workspace's `rig_config.json`
(any `.json` file is read as one); the output is always a `.camrig`. Where
`create` reads a directory of images, `cp` reads a file that already
describes cameras (and, for a rig, sensor poses) — a solved reconstruction, a
previously built rig, or a COLMAP rig config. A `rig_config.json` source is
converted by the rules in
[*A `rig_config.json` source*](#a-rig_configjson-source).

### Selectors

A *selector* says which slice of the source becomes the output rig. The
selectors available depend on the source type, and at most one may be given.

| Selector | Source | Result |
|----------|--------|--------|
| `--rig N` | `.sfmr` | The whole of rig `N` — a multi-sensor `.camrig` preserving the rig's sensors, cameras, and `sensor_from_rig` poses. |
| `--rig N` | `rig_config.json` | Rig `N` of the file, converted to a `.camrig` that groups the same images into the same sensors and frames. |
| `--camera N` | `.sfmr` or `.camrig` | Camera `N` from the source's camera pool, on its own — a single-sensor `generic` rig at the identity pose. |
| `--sensors RANGE` | `.camrig` | The selected subset of the source's sensors — itself a `.camrig`, with the camera pool reduced to the cameras those sensors use. |
| *(none)* | `.sfmr` | Defaults to `--rig 0` when the reconstruction has exactly one rig, or `--camera 0` when it has no rig data and exactly one camera; otherwise the command asks for a selector. |
| *(none)* | `.camrig` | Copies the whole rig unchanged. |
| *(none)* | `rig_config.json` | Defaults to `--rig 0` when the file holds exactly one rig; otherwise the command asks for `--rig`. |

`--rig` is rejected for a `.camrig` source (a `.camrig` holds exactly one
rig — use `--sensors`); `--sensors` is rejected for a `.sfmr` source (use
`--rig`). A `rig_config.json` source takes `--rig` only: `--camera` and
`--sensors` are rejected for it, and so `--pattern` is too.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `SOURCE` | path | required | The `.sfmr`, `.camrig` or `rig_config.json` file to copy from. |
| `OUTPUT_FILE` | path | required | Path of the `.camrig` file to write. Its directory is the rig root. |
| `--rig` | int | — | `.sfmr` or `rig_config.json` only. Index of the rig to copy. |
| `--camera` | int | — | Index into the source's camera pool. Copies that one camera as a single-sensor rig. |
| `--sensors` | range | — | `.camrig` only. Sensor indices to keep, as an integer range expression (e.g. `0-2`, `0,2,4`). |
| `--pattern` | pattern | — | Image pattern for the output sensor. Only valid with `--camera` (single-sensor output); for multi-sensor results the per-sensor patterns are inferred (see below). |
| `--name` | str | (see below) | Rig name stored in the file. Defaults to the source rig's name, then to the output file stem (a `rig_config.json` names no rigs, so its rigs take the stem). |

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

- **`rig_config.json` source** — the patterns are derived from the image
  files on disk and checked against them; see the next section.

### A `rig_config.json` source

`cp` converts one rig of a `rig_config.json`
([rig-config.md](../../workspace/rig-config.md)) into a `.camrig` that the
commands use as they use that rig of the `rig_config.json`: the same images
for each sensor, the same frames, the same cameras and the same
`sensor_from_rig` poses. The conversion is `copy_from_rig_config` in
[`camrig/from_rig_config.py`](../../../src/sfmtool/camrig/from_rig_config.py).

- **Where the file goes.** The image prefixes are relative to the folder of
  the `rig_config.json`, which is the workspace root. `OUTPUT_FILE` must be
  in that folder or under it, so that `sfm solve` finds it, and its folder,
  the rig root, must hold every image of the rig; the patterns are written
  relative to it. Beside the `rig_config.json` is the usual place.
- **Images.** A sensor's images are the image files (`.png`, `.jpg`,
  `.jpeg`, the files `sfm solve` takes from a directory) whose path relative
  to the workspace starts with its `image_prefix`. An image whose path starts
  with the prefixes of several sensors belongs to the first of them in the
  file, as in `sfm solve`. A sensor with no images is refused: there is
  nothing to derive its pattern and camera from.
- **Patterns.** A sensor's images must form one sequence by the rule
  `sfm pick-images` groups files by: one folder, names that are the same
  apart from the file number, every image numbered and no number naming two
  files. The sequence's pattern, with a frame field for the number (`%0Nd`
  when every number is written with the same `N > 1` digits, `%d` otherwise),
  is the sensor's image pattern. The command fails, naming the images, when a
  sensor's images have no number or form more than one sequence.
- **Frames.** `rig_config.json` puts two images of a rig in one frame when the
  paths after their prefixes, their frame keys, are equal; the `.camrig` puts
  them in one frame when their frame fields capture the same number. Before
  writing, `cp` matches the derived patterns against the files from the rig
  root as `sfm solve` matches them, and fails unless they give each sensor
  exactly the images `rig_config.json` gives it, and no other file such as an
  image the file gives another sensor, and put two images in one frame
  exactly when their frame keys are equal. `a/f_7.jpg` and `b/f_07.jpg`, for
  example, are two frames in `rig_config.json` but would be one in the
  `.camrig`, so they are refused. A frame that lacks some sensors' images is
  kept as it is.
- **Sensors.** The reference sensor (`ref_sensor: true`) is sensor 0, the
  primary sensor, and the others follow in the file's order.
- **Poses.** Each `cam_from_rig` pose is in COLMAP convention; it becomes the
  canonical `sensor_from_rig` by conjugation with `S = diag(1, −1, −1)`
  (`R = S · R_colmap · S`, `t = S · t_colmap`), with the quaternion made unit.
  The reference sensor is at the identity, and a missing translation is zero.
  Written into a COLMAP database, the `.camrig` gives back the same
  `cam_from_rig` poses
  ([camrig-file-format.md](../../formats/camrig-file-format.md#relationship-to-the-colmap-rig-format)).
- **Cameras.** Each sensor's camera is built as `sfm solve` builds it from
  `rig_config.json`, on the sensor's first image in path order: with
  `camera_params`, the parameters for `camera_model_name` at that image's
  size; without them, EXIF inference with `camera_model_name`, if any, as the
  model, including the fisheye focal-length seed. A `.camrig` camera is a
  full calibration, so an inferred camera is stored as one, and `cp` prints a
  note naming each sensor whose camera was inferred. Every image of a sensor
  must have one resolution, as `sfm solve` requires of a `.camrig`. Sensors
  with equal cameras share one camera of the pool.
- **Several rigs.** A `.camrig` holds one rig, so each rig of the file is
  converted by its own run with `--rig N`, to its own `.camrig`. `sfm solve`
  uses several `.camrig` files in one solve, each for the images it covers.

The `.camrig` is written with `rig_type` `generic`. After writing, `cp`
prints the rig's sensor, frame and camera counts and, for each sensor, its
prefix, its pattern, its image count and its camera, so the conversion can be
checked, and a note naming any other `.camrig` file that already covers some
of the rig's images, since `sfm solve` refuses an image two `.camrig` files
cover.

```
$ sfm camrig cp rig_config.json kerry_park.camrig
Wrote kerry_park.camrig
  rig 0 (of 1 in the file): 2 sensors, 24 frames, 1 camera(s)
  sensor 0: fisheye_left/ -> fisheye_left/frame_%02d.jpg  (24 images, OPENCV_FISHEYE 480x480)
  sensor 1: fisheye_right/ -> fisheye_right/frame_%02d.jpg  (24 images, OPENCV_FISHEYE 480x480)
```

The COLMAP database `sfm solve` writes from the `.camrig` differs in two ways
from the one it writes from `rig_config.json`, both for a sensor without
`camera_params`:

- Its inferred camera is written with `has_prior_focal_length` set, since it
  is a `.camrig` calibration; from `rig_config.json` it is written without.
- With `--camera-model`, its camera is the stored inferred camera refit to
  the model, where from `rig_config.json` it is inferred under that model.

### Behaviour

The output is written with `rig_type` `generic`, except when the whole of a
`.camrig` is copied (no selector, or `--sensors` selecting every sensor), in
which case the source's `rig_type` and `rig_attributes` are preserved. A
subset of a typed rig (e.g. three faces of a `cubemap`) is no longer that
type, so it becomes `generic` with empty attributes.

`sensor_from_rig` poses are copied verbatim — the rig frame is arbitrary, so
a sensor subset needs no rebasing. `--camera` produces a lone sensor at the
identity pose.

A `.camrig` holds no per-image data, so a camera copied out of a `.sfmr` is
the shared camera, without its images' intrinsic adjustments
([sfmr-file-format.md](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13)).
When a copied camera has a non-zero adjustment on any of its images (a focal
scale, a principal-point offset, a radial correction or rolling-shutter
rates), `cp` prints a note naming the camera and saying that the adjustments
are not copied. A `.camrig` camera has no `rolling_shutter_scanlines` setting
either, so for a copied camera with one `cp` prints a note naming the camera
and saying the setting is not copied.

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

# Convert the second rig of a workspace's rig_config.json
sfm camrig cp rig_config.json second_rig.camrig --rig 1
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
spacing (see `specs/core/spherical/spherical-tiles-rig.md`). Pass `--seed` for a
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
