# `sfm solve` Command

## Overview

Runs Structure from Motion to produce a 3D reconstruction (`.sfmr` file) from images or
pre-computed matches. Uses pycolmap to run either COLMAP's incremental mapper
or GLOMAP's global mapper.

The command is implemented in
[`_commands/solve.py`](../../../src/sfmtool/_commands/solve.py), a Click wrapper
that validates the arguments and calls
[`_solve_driver.py`](../../../src/sfmtool/_solve_driver.py), which runs one
solve or the windows of `--seq-overlap` mode. Each solve calls
[`_incremental_sfm.py`](../../../src/sfmtool/_incremental_sfm.py) (`--incremental`)
or [`_global_sfm.py`](../../../src/sfmtool/_global_sfm.py) (`--global`). Both
prepare the COLMAP database through
[`_solve_setup.py`](../../../src/sfmtool/_solve_setup.py), which calls
[`colmap/db_setup.py`](../../../src/sfmtool/colmap/db_setup.py) to extract
features if needed, set up cameras and rigs, and run matching or load the
`.matches` file. Both write their output through `_save_reconstructions` in
`_incremental_sfm.py`, which applies the rules in
[Outputs and Multiple Models](#outputs-and-multiple-models).

## Coordinate Convention

The solvers work in COLMAP's +Z-forward, Y-down convention; the `.sfmr`
output is canonical Z-up / −Z-forward. Saving the solver's result is
therefore a convention boundary: the COLMAP→canonical conversion is
applied — the camera-frame flip `S` on every pose (including rig
`sensor_from_rig` poses) and the world canonicalization `W` on 3D points —
before the reconstruction is written, exactly as `sfm from-colmap-bin`
does for external COLMAP output. See the "Coordinate System Conventions"
section of [`sfmr-file-format.md`](../../formats/sfmr-file-format.md) for the
transform definitions.

## Command Syntax

```bash
sfm solve PATHS... --incremental | --global [OPTIONS...]
```

`PATHS` is required. It can be image files/directories (matching is run internally) or a
single `.matches` file (pre-computed matches are used directly).

## Solver Selection

Exactly one is required:

| Flag | Description |
|------|-------------|
| `--incremental / -i` | COLMAP incremental SfM (sequential image registration) |
| `--global / -g` | GLOMAP global SfM (simultaneous pose estimation) |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--seed / -s` | int | | Random seed for reproducibility |
| `--output / -o` | path | auto | Output `.sfmr` file path (default: timestamped in `sfmr/`) |
| `--colmap-dir` | path | auto | Directory for COLMAP database and intermediates |
| `--sfmr-dir` | path | `sfmr/` | Directory for `.sfmr` output files |
| `--seq-overlap` | string | | Sequential overlap mode: `WINDOW,OVERLAP` (e.g., `100,20`) |
| `--refine-rig / --no-refine-rig` | bool | `true` | Refine sensor-from-rig poses in bundle adjustment |
| `--flow-match` | flag | | Use optical flow matching instead of exhaustive |
| `--flow-preset` | `fast` \| `default` \| `high_quality` | `default` | Flow quality preset |
| `--flow-skip` | int | 5 | Sliding window size for flow matching |
| `--max-features` | int | | Maximum features per image |
| `--camera-model` | choice | auto | Camera model to start from: it replaces the model EXIF inference picks, and refits to this model the camera a `.camrig` or `camera_config.json` supplies (see [Camera Intrinsics](#camera-intrinsics)); one of the 11 supported COLMAP model names (`PINHOLE`, `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `OPENCV_FISHEYE`, `FULL_OPENCV`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`, `THIN_PRISM_FISHEYE`, `RAD_TAN_THIN_PRISM_FISHEYE`). The same vocabulary is accepted by `match` and `camrig create`. |
| `--range / -r` | string | | Range expression for file numbers, from the input directories or the images of a `.matches` file; or a named image list's name, optionally with a slice (`'NAME[:30]'`) |
| `--named-image-list` | name | | Use the images of this named image list of the workspace in place of image paths, or, with a `.matches` file, only the file's images that are among the list's images; with `-r RANGE`, only its members whose frame number is in `RANGE` |
| `--detect-infinity / --no-detect-infinity` | bool | `true` | Reclassify points whose depth the solve could not pin down as points at infinity (`w = 0`). See [sfmr file format §7](../../formats/sfmr-file-format.md). |

## Input Modes

### From images (matching + solving)

When `PATHS` are image files or directories, the command extracts features (if needed), runs
matching, and then solves. The matching strategy is exhaustive by default, or flow-based with
`--flow-match`. Extraction writes `.sift` version 2 with the workspace's options, so in a
version 1 workspace it refuses to extract a missing or stale file, naming the workspace file
and pointing at `sfm ws upgrade`
([sift-command.md § Version refusals](../image-feature/sift-command.md#version-refusals));
a version 1 workspace whose `.sift` files are all present and current solves as before.

### From a named image list

With `-r NAME` or `--named-image-list NAME` and no `PATHS`, the images are the
members of that named image list of the workspace found from the current
directory, in the list's order, keeping only the members that are image files;
a range given with the name narrows the list, and a slice after the name takes
part of it by position, in the slice's order (`-r 'walk1_sharp[:30]'` is its
first 30 images). Giving image paths as well is an error, and the list's
workspace is the solve's workspace. A list given with a `.matches` file
filters the file's images instead
([From a `.matches` file](#from-a-matches-file)). The rules and errors are in
[the shared `-r/--range` option](../README.md#the-shared--r--range-option).

### From a `.matches` file

When a single `.matches` file is provided, pre-computed matches are loaded directly into the
COLMAP database, skipping feature extraction and matching.

`-r` and `--named-image-list` choose which of the file's images to solve. The
images left out are not written to the database, and neither is any pair that
involves one of them; the kept images stay in the file's order.

- **A range** keeps the images whose file number it holds. A range that keeps
  none is an error naming the range and the file.
- **A named image list** keeps the images that are among the list's images.
  The list is read from the workspace the `.matches` file resolves to, not
  the one found from the current directory, and resolved as it is without a
  `.matches` file: its members that are image files, in the list's order,
  narrowed first by a range given beside the name, then sliced. Each of the
  file's images, named by its workspace-relative path, is kept when it is
  among the images that leaves. So a slice counts positions in the list:
  `-r 'walk1_sharp[:30]'` is the same 30 images with the `.matches` file as
  without it, and fewer when the file lacks some of them. The list's images
  the file does not hold are ignored, and the command prints how many; when
  the file holds none of them, it is an error naming the lists file, the
  list and the `.matches` file.

This lets one `.matches` file, matched once for a whole list, serve test
solves on parts of it: `-r 'walk1_sharp[:30]'`, `-r 'walk1_sharp[-30:]'`, or
`-r 'walk1_sharp[100:130]'` for 30 from the middle.

The file must carry two-view geometries: the mapper reads its correspondence
graph from the database's two-view geometry table, so a file without them
registers nothing. A clusters-bearing `.matches` file — which by format never
stores two-view geometries — is refused up front with a `UsageError` naming
`sfm match --derive-pairs`, the mode that turns it into the pairwise + TVG
file the mapper reads (see
[`../image-feature/match-command.md`](../image-feature/match-command.md)).

### Sequential overlap mode

`--seq-overlap WINDOW,OVERLAP` solves the sequence in overlapping windows of `WINDOW`
images, consecutive windows sharing `OVERLAP` images, and writes each window's
reconstruction to its own `.sfmr` file. The last window holds the images that remain and
can be shorter than `WINDOW`. It does not align or merge the windows; run `sfm align` and
`sfm merge` on the outputs for that. Useful for long sequences that fail with a single
solve. The input must contain exactly one numbered image sequence.

`--seq-overlap` cannot be combined with `--output` (each window writes its own
automatically-named output) or with a `.matches` input file (the windows drive
their own feature matching). Each window matches its images exhaustively;
`--flow-match` is not applied to the windows.

## Outputs and Multiple Models

The mapper can return more than one reconstruction (it splits a model whenever
it cannot register every image into a single one). `sfm solve` writes one
`.sfmr` per surviving model:

- **Degenerate models are skipped.** A model the mapper abandoned with zero
  3-D points is dropped with a warning rather than aborting the run — a single
  junk fragment must never discard the good models that share the output loop.
  Only when *no* model survives does the command raise
  `No 3D points found in reconstruction.`. A split is also a hint the run may be
  worth re-seeding, so it is surfaced loudly rather than swallowed silently.
- **The largest surviving model is primary.** Surviving models are ordered by
  registered image count (ties broken by observation count), and that ordering
  — not the mapper's internal model index — decides output naming: with an
  explicit `--output`, the largest model takes that exact path and the rest get
  `{stem}-{N}{suffix}` siblings; without it, each model is auto-named from the
  images it actually contains, as `<YYYYMMDD>-<NN>-solve-<descriptor>.sfmr`
  with the length-bounded image descriptor that
  [`sfm match`](../image-feature/match-command.md#the-generated-default-name)
  names its output with. This keeps a small fragment that happened to land
  at a lower model index from claiming the requested output name.
- **Each point is seen once per image.** A model is converted to `.sfmr`
  through the COLMAP import, which drops a track's observations beyond one per
  image, keeping the one with the lowest reprojection error, and prints
  `Dropped N repeat observation(s) from M point(s)` when it drops any. See
  [drop-repeat-observations.md](../../core/reconstruction/drop-repeat-observations.md).

## Rig Support

A multi-sensor `.camrig` covering images being solved makes them a rig of the solve
([Camera Rig Files](#camera-rig-files-camrig)), and the command sets up its rig
constraints. Use `--refine-rig / --no-refine-rig` to control whether sensor-from-rig poses
are refined during bundle adjustment. A `rig_config.json` at the workspace root stops the
command before it reads any image, naming the `sfm camrig cp` command that converts it
([workspace/rig-config.md § A `rig_config.json` in a workspace](../../workspace/rig-config.md#a-rig_configjson-in-a-workspace)).

## Camera Intrinsics

If an image being processed that no `.camrig` covers resolves a `camera_config.json`
(closest-ancestor walk from its parent directory up to the workspace root), the file's
intrinsics are the starting camera. Each resolved file is one camera, shared by the
images that resolve it, and the images that resolve none share one camera inferred from
EXIF. See
[`../../workspace/camera-config.md`](../../workspace/camera-config.md).

With `--camera-model`, the camera a covering `.camrig` or a `camera_config.json`
supplies is refit to that model by the
lens-only fit `CameraIntrinsics.refit`
([`../../core/camera/refit-camera-intrinsics.md`](../../core/camera/refit-camera-intrinsics.md)),
and the refit camera is the starting camera, used exactly where the supplied one would
have been. Every such camera the solve uses is refit: each camera of a multi-sensor
`.camrig`, and the camera of each single-sensor `.camrig` and of each
`camera_config.json`. The files are not changed. The fit keeps the principal point and image size, so a `SIMPLE_PINHOLE` camera
refit to `SIMPLE_RADIAL` keeps its focal and gains `radial_distortion_k1 = 0`, a term the
solve's bundle adjustment then refines. One line per refit camera names it, the two models, and the fit's rms and max in pixels, followed
by a line for each term the target model cannot represent:

```
Camera 0 of /ws/cam.camrig: SIMPLE_PINHOLE -> SIMPLE_RADIAL, fit over θ ≤ 42.5° (image_corner): rms 0.000 px, max 0.000 px
```

A fit that is refused, such as a perspective model for a fisheye camera whose image
reaches past 90°, stops the command with an error naming the camera and the reason. For
images no file covers, `--camera-model` replaces the model EXIF inference would pick.

A `.camrig` camera can be `EQUIDISTANT_FISHEYE`, the model
`sfm estimate-intrinsics --write-camrig` writes for a fisheye lens. COLMAP has no such
model, so the camera goes into the COLMAP database as `SIMPLE_RADIAL_FISHEYE` with
`radial_distortion_k1 = 0`, which is the same projection
([colmap-interop.md § Camera models](../../formats/colmap-interop.md#camera-models)).
Both `-i` and `-g` refine that `k1` along with the focal length, so the solved camera is
`SIMPLE_RADIAL_FISHEYE`, read back as `EQUIDISTANT_FISHEYE` only if `k1` comes out
exactly `0`. The solve therefore starts exactly as it does with
`--camera-model SIMPLE_RADIAL_FISHEYE`, whose refit of an `EQUIDISTANT_FISHEYE` camera
is that same `k1 = 0` camera.

## Camera Rig Files (`.camrig`)

`sfm solve` auto-discovers `.camrig` files in the workspace and uses those whose
image patterns match any of the images being solved. One rule holds for every
image: it takes its camera from the `.camrig` that covers it, else from the
`camera_config.json` it resolves, else from EXIF inference
([Camera Intrinsics](#camera-intrinsics)). Any number of `.camrig` files, single-sensor
and multi-sensor, can be used in one solve, and each image may be covered by at
most one of them. A `.camrig` takes precedence over `camera_config.json` for the
images it covers, and a note is printed when it overrides one.
`sfm estimate-intrinsics --write-camrig-per-sequence` writes one
single-sensor `.camrig` per image sequence of a capture, each estimated from
its own sequence's images, so a solve over the whole capture gives each
sequence its own camera
([estimate-intrinsics-command.md § One camera per sequence](estimate-intrinsics-command.md#one-camera-per-sequence)).

The images no `.camrig` covers take their cameras as they would with no
`.camrig`: one camera per `camera_config.json` they resolve, and one camera
inferred from EXIF for those that resolve none. Each of them is a frame of its
own, beside the rig frames of any multi-sensor `.camrig`. So a capture with a
`.camrig` for only some of its image sequences still solves the others. One line
names how many images no `.camrig` covers, with an example, and the source of
their cameras, so a pattern that misses images it was meant to match shows up:

```
Camera intrinsics: 12 of 40 images being solved are covered by no .camrig, e.g. /ws/images/shopC_00001.jpg; they use EXIF inference.
```

The source reads `camera_config.json`, `EXIF inference`, or both with their
counts, as in `camera_config.json (4) and EXIF inference (8)`.

- A **single-sensor** `.camrig` (such as one written by `sfm camrig create`)
  supplies the starting camera intrinsics for the images its stored image
  pattern matches, refit to `--camera-model` when that is given
  ([Camera Intrinsics](#camera-intrinsics)). Each single-sensor `.camrig` is
  one camera of the solve, so two videos shot with different lenses, each with
  its own `.camrig`, are solved together with a camera for each; the
  `.sfmr` holds both cameras, and each image's camera is its file's. Each of
  its images is a frame of its own.
- A **multi-sensor** `.camrig` (such as one written by `sfm insv2rig`) is one
  rig of the solve, of the images it covers. Each sensor's image pattern
  carries a frame field, and images from different sensors that share a frame
  index form one rig frame. The command builds one COLMAP camera per sensor
  present (intrinsics from the rig's camera pool, scaled to the actual image
  resolution) and one rig whose reference is the lowest-indexed sensor present
  — its `cam_from_rig` is the identity and the others are rebased relative to
  it. Two multi-sensor `.camrig` files are two rigs, each with its own cameras
  and frames. Same-frame image pairs are excluded from matching, and
  `--refine-rig / --no-refine-rig` controls whether the `sensor_from_rig` poses
  are refined during bundle adjustment.

The rig frames and the frames of single images go into one COLMAP database
(`_setup_db_with_camrigs` in
[`colmap/db_builders.py`](../../../src/sfmtool/colmap/db_builders.py)), the
rigs first and then the single images, and are solved together. When a
multi-sensor `.camrig` is used, the `.sfmr` holds its rig, and a one-sensor rig
for each camera of the single images, whose frames each hold one image.

The command fails up front, with a message naming the offending files, when the
discovered `.camrig` files cannot be used:

- an image being solved is matched by more than one `.camrig`, or by two
  sensors of one `.camrig`;
- an image being solved is matched by no `.camrig`, and a `.camrig`'s pattern
  matches it apart from the zero-padding of its number (`shot_007.jpg` for
  `shot_%04d.jpg`): a frame field matches a number only as printf writes it
  ([camrig-file-format.md](../../formats/camrig-file-format.md#how-camrig-files-fit-into-workspaces)),
  and the image would otherwise take its camera from elsewhere without a
  word;
- two images of one sensor of a multi-sensor `.camrig` share a frame index;
- the images of a `.camrig` camera have mixed resolutions, or a resolution whose
  aspect ratio differs from the camera's;
- `--camera-model` is given and the refit of a camera the solve uses is refused.

`sfm pick-images` finds its rigs by the same rule, so a list it writes is one
`sfm solve` accepts with the same `.camrig` files
([pick-images-command.md § Collecting the frames](../image-feature/pick-images-command.md#collecting-the-frames)).

See [`../../formats/camrig-file-format.md`](../../formats/camrig-file-format.md).

## Usage Examples

```bash
# Incremental SfM from images
sfm solve --incremental images/

# Global SfM from pre-computed matches
sfm solve matches/2026-01-15_match_001.matches --global

# Flow-based matching for video, with seed
sfm solve images/ --incremental --flow-match --flow-preset high_quality --seed 42

# Sequential overlap for long sequences
sfm solve images/ --global --seq-overlap 100,20

# Solve a subset of images
sfm solve images/ --incremental --range 1-200 --max-features 8192

# Solve the images of a named image list
sfm solve --incremental -r walk1_sharp

# Solve the first 30 images of the list from a .matches file of the whole list
sfm solve --global tvg-matches/walk1_sharp.matches -r 'walk1_sharp[:30]'
```
