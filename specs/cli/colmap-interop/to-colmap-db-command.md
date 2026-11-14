# `sfm to-colmap-db` Command

## Overview

Creates a COLMAP database from a `.sfmr` reconstruction or `.matches` file. Useful for
re-running COLMAP operations on existing data or for interoperability with COLMAP tools.

The `sfm` CLI uses this general approach to implement many of its operations, converting the
workspace data into a COLMAP .db, running the COLMAP operation, then converting back to `.sfmr`.

## Coordinate Convention

This command is a convention boundary: `.sfmr` and `.matches` data are
stored in the canonical Z-up / −Z-forward convention, while a COLMAP
database holds COLMAP-convention (+Z-forward, Y-down) data. On export the
canonical→COLMAP conversion is applied: relative and rig poses are
S-conjugated into COLMAP camera frames, and world-space data such as pose
priors gets the inverse world canonicalization `W⁻¹` — so a solver run
against the database, whose output is imported back through the forward
conversion (see `solve-command.md`), round-trips stably. Pixel-space
two-view geometry (fundamental/essential matrices, keypoints, matches) is
invariant under the camera-frame flip and is written unchanged. See the
"Coordinate System Conventions" section of
[`sfmr-file-format.md`](../../formats/sfmr-file-format.md) for the transform
definitions.

## Command Syntax

```bash
sfm to-colmap-db <INPUT_PATH> <DATABASE.db> [OPTIONS...]
```

`INPUT_PATH` can be a `.sfmr` or `.matches` file. The output database path is a
required positional argument (matching the positional `OUTPUT` convention of
`to-colmap-bin` and `to-nerfstudio`).

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-features` | int | all | Maximum features per image (`.sfmr` only) |
| `--no-guided-matching` | flag | | Disable two-view geometry pre-population (`.sfmr` only) |
| `--camera-model` | choice | auto | Camera model to use (`.matches` only): it replaces the model EXIF inference picks, and refits to this model the camera a `.camrig` or `camera_config.json` supplies; one of the 11 shared COLMAP model names (same vocabulary as `solve` / `match` / `camrig create`) |

> **Known gap (cross-exporter parity):** unlike `to-colmap-bin` and
> `to-nerfstudio`, this exporter has no `--range` image-subsetting option.
> Carried forward from the 2026-06-23 spec audit as a feature gap, not a
> spec/code divergence.

## Input Modes

### From `.sfmr`

Populates the database with camera intrinsics, pose priors, keypoints, descriptors, and
(by default) two-view geometries with fundamental matrices for guided matching.

The keypoints and descriptors are read from each image's `.sift` file in the
`feature_prefix_dir` the reconstruction records. An image whose `.sift` file is
missing or older than the image is extracted again with the feature tool and
options the reconstruction records, and the extraction writes `.sift` version 2
([workspace.md § Versions never mix](../../workspace/workspace.md#versions-never-mix)).
When every file is present and current, nothing is extracted.

A record with version 1 settings (no `image_to_gray`) lives in the `.sfmr`, so
`sfm ws upgrade` cannot change it. When such a record has files to extract, the
command converts it in memory as `sfm ws upgrade` converts a workspace file, the
way the viewer's *Build SIFT Files* does, extracts every file into the folder
named from the converted options, reads them from there, and prints one line:

```
Converted the version 1 SIFT extraction record of the reconstruction ‹path› to version 2 in memory, so its .sift files are read from ‹new folder›, moved from ‹old folder›; the .sfmr is not rewritten.
```

This holds for every feature tool, and for both `embedded_patches` and
`sift_files` reconstructions: the export writes keypoints and descriptors from
the `.sift` files and no matches or tracks, so it reads all its features from
one folder and needs no correspondence with the reconstruction's feature
indexes. (The viewer's *Build SIFT Files* refuses a version 1 record on a
`sift_files` node, because that node goes on using its feature indexes against
the folder it builds.) A record of `sfmtool-undistort` is refused, since only
`sfm undistort` makes those files. No refusal about a `.sfmr`'s record
suggests `sfm ws upgrade`.

The cameras follow `sfm to-colmap-bin`'s rule
([to-colmap-bin-command.md](to-colmap-bin-command.md#cameras)): a camera with a
non-zero intrinsic adjustment on any of its images is written as one camera per
image of it, each that image's intrinsics, and every other camera once. A COLMAP
solve started from the database therefore begins at each image's intrinsics, and
each fundamental matrix is computed from the two images' own intrinsics. The
database holds no rigs, so no rig is refused here. A reconstruction whose
images carry a radial correction or rolling-shutter rates is refused, naming
the camera and the correction, before the database is created, as
`sfm to-colmap-bin` refuses it.

### From `.matches`

Populates the database with cameras, keypoints, descriptors, and pre-computed matches and
two-view geometries (if included) from the matches file.

The keypoints and descriptors come from the `.sift` files the `.matches` file
was matched on, in the feature folder it records, not the workspace's current
one; a missing `.sift` file, or one whose content hash differs from what the
`.matches` file recorded, stops the command before it writes the database, as
it stops `sfm solve`
([solve-command.md § From a `.matches` file](../reconstruction/solve-command.md#from-a-matches-file)).

A clusters-bearing `.matches` file works directly: its pairs are derived at
read time through `sfmtool.feature_match.pairs_from_matches`, and the database
gets matches with no two-view geometry table. That is enough for anything that
reads correspondences, but not for COLMAP's mapper, which reads its
correspondence graph from the two-view geometry table — verify first with
`sfm match --derive-pairs` and export the file it writes.

## Camera Intrinsics (`.matches` mode only)

The cameras are set up as `sfm solve` sets them up from a `.matches` file
([solve-command.md § Camera Rig Files](../reconstruction/solve-command.md#camera-rig-files-camrig)):
each image takes its camera from the `.camrig` that covers it: each multi-sensor
`.camrig` supplies a rig, with its cameras and rig frames, of the file's images it
covers, and each single-sensor `.camrig` one camera for the images it covers. An image no
`.camrig` covers is a frame of its own and takes the camera of the
`camera_config.json` it resolves (closest-ancestor walk from its parent directory up to
the workspace root), one camera per resolved file, else one camera inferred from EXIF
that all such images share. An image more than one `.camrig` matches is refused. When
the `.camrig` files leave images uncovered, a line names how many, calling them the
`images of the .matches file`. A `rig_config.json` at the workspace root stops the
command, naming the `sfm camrig cp` command that converts it
([workspace/rig-config.md § A `rig_config.json` in a workspace](../../workspace/rig-config.md#a-rig_configjson-in-a-workspace)).
See [`../../workspace/camera-config.md`](../../workspace/camera-config.md). With
`--camera-model`, a camera one of those files supplies is refit to that model and the
refit camera is written to the database, with one report line per refit camera and an
error naming the camera when the fit is refused, exactly as
[solve-command.md § Camera Intrinsics](../reconstruction/solve-command.md#camera-intrinsics)
describes.

## Usage Examples

```bash
# Create database from reconstruction
sfm to-colmap-db sfmr/solve_001.sfmr colmap/database.db

# From matches, with camera model override
sfm to-colmap-db matches/match_001.matches colmap/database.db --camera-model OPENCV

# Without guided matching
sfm to-colmap-db solve.sfmr database.db --no-guided-matching
```
