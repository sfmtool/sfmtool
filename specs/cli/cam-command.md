# `sfm cam` Command Group

## Overview

`sfm cam` groups camera-related operations. Today the only subcommand is `cp`,
which extracts intrinsics from a `.sfmr` reconstruction and writes them to a
`camera_config.json` file. The `cp` naming sets up future siblings (`mv`, `ls`,
`print`) without committing to them.

`cam` is the first command **group** in the CLI; every other top-level command
is flat. The group has no flags of its own — all flags live on the subcommand.

## `sfm cam cp`

Copies a single camera's intrinsics from a reconstruction into a
`camera_config.json` file, ready to be committed to a workspace so that
subsequent solves start from those values instead of EXIF guesses. See
[`../workspace/camera-config.md`](../workspace/camera-config.md) for the file
format and resolution rules.

### Command Syntax

```bash
sfm cam cp <INPUT.sfmr> <OUTPUT.json> [--index N]
```

### Arguments and Options

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `INPUT.sfmr` | path | required | The reconstruction to read from |
| `OUTPUT.json` | path | required | Where to write the `camera_config.json` |
| `--index N` | integer | (auto) | Camera index to copy. Required when the reconstruction has more than one camera. Indexes match the reconstruction's camera order, starting at 0. |

### Behavior

- Reads the reconstruction, extracts the chosen camera's `CameraIntrinsics`,
  serializes to the `camera_config.json` schema (`version: 1`), writes
  `OUTPUT.json` (creating parent directories if needed).
- If the reconstruction has exactly one camera, `--index` may be omitted.
- If the reconstruction has multiple cameras and `--index` is not supplied,
  the command exits with an error listing the camera count.
- If `--index` is out of range, the command exits with an error listing the
  valid range.
- Parameters are written in the canonical order from `_CAMERA_PARAM_NAMES` so
  the output is stable across runs and easy to diff.

### Workflow

The motivating workflow ("Bootstrap from an Unknown Camera") looks like:

```bash
# Solve a small subset, clean it up, harvest intrinsics, re-solve
sfm solve -i my_project/photos --range 100:130 \
    --output my_project/sfmr/calibration_subset.sfmr
sfm xform my_project/sfmr/calibration_subset.sfmr \
    --remove-narrow-tracks 7deg --bundle-adjust --align-to-input \
    --output my_project/sfmr/calibration_subset_cleaned.sfmr
sfm cam cp my_project/sfmr/calibration_subset_cleaned.sfmr \
    my_project/camera_config.json
sfm solve -i my_project/photos    # now starts from the harvested intrinsics
```

### Usage Examples

```bash
# Single-camera reconstruction
sfm cam cp sfmr/solve_001.sfmr camera_config.json

# Multi-camera reconstruction — pick camera 1
sfm cam cp sfmr/multi.sfmr --index 1 phone/camera_config.json
```

## Implementation

`src/sfmtool/_commands/cam.py` defines the Click group `cam` and the `cp`
subcommand. The group is registered on `main` in `cli.py` under the
"Workspace" category. `cameras[i].to_dict()` on a loaded `SfmrReconstruction`
already produces the right shape — `cp` just orders the parameters according
to `_CAMERA_PARAM_NAMES` and wraps the block with `{"version": 1,
"camera_intrinsics": ...}`.
