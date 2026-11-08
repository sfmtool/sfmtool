# Rig Config Specification

## Overview

`rig_config.json` is COLMAP's
[`rig_configurator` rig config](https://colmap.github.io/rigs.html) file: a
description of one or more multi-sensor camera rigs, sets of cameras that move
together as a single physical assembly, such as a stereo pair or a 360°
back-to-back fisheye rig. Each sensor is named by the path prefix of its
images and carries its pose in the rig and, optionally, its intrinsics.

sfmtool describes a workspace's rigs only by `.camrig` files
([camrig-file-format.md](../formats/camrig-file-format.md)), and reads a
`rig_config.json` in one place: `sfm camrig cp rig_config.json OUTPUT.camrig
[--rig N]` converts one of its rigs to a `.camrig`
([camrig-command.md § A `rig_config.json` source](../cli/workspace/camrig-command.md#a-rig_configjson-source)).
This spec describes the file as that converter reads it, and the error the
other commands give when a workspace still holds one
([*A `rig_config.json` in a workspace*](#a-rig_configjson-in-a-workspace)).

The file is COLMAP's format verbatim, with no sfmtool-specific extensions,
and it stays in COLMAP's coordinate convention
([*Coordinate Convention*](#coordinate-convention)).

## File Format

```json
[
  {
    "cameras": [
      {
        "image_prefix": "fisheye_left/",
        "ref_sensor": true,
        "camera_model_name": "OPENCV_FISHEYE",
        "camera_params": [129.15, 129.26, 240.0, 240.0,
                          0.0381, -0.0080, 0.0083, -0.0027]
      },
      {
        "image_prefix": "fisheye_right/",
        "ref_sensor": false,
        "cam_from_rig_rotation": [0, 0, 1, 0],
        "cam_from_rig_translation": [0, 0, -0.0307],
        "camera_model_name": "OPENCV_FISHEYE",
        "camera_params": [129.15, 129.26, 240.0, 240.0,
                          0.0381, -0.0080, 0.0083, -0.0027]
      }
    ]
  }
]
```

This is the rig of the kerry_park dataset, which `test-data/images/kerry_park`
describes by `kerry_park.camrig`, the conversion of this file.

### Top-Level Shape

The file is a JSON **array** of rig objects, even when only one rig is
present. This matches the COLMAP rig file shape and lets a file declare
several rigs that share the same image tree.

### The Rig Object

| Field | Type | Description |
|-------|------|-------------|
| `cameras` | array | One entry per sensor in the rig. Exactly one must have `ref_sensor: true`. |

### The Camera (Sensor) Entry

| Field | Type | Description |
|-------|------|-------------|
| `image_prefix` | string | Path prefix, relative to the folder of the file, that identifies images captured by this sensor. Forward slashes only. |
| `ref_sensor` | boolean | `true` for exactly one sensor per rig. The reference sensor defines the rig coordinate frame; its `cam_from_rig` is the identity. |
| `cam_from_rig_rotation` | array of 4 floats | **WXYZ** quaternion (see *Quaternion Convention*). Required for non-reference sensors. Omitted for the reference sensor. |
| `cam_from_rig_translation` | array of 3 floats | Translation in metres. Defaults to `[0, 0, 0]` if omitted. |
| `camera_model_name` | string | Optional COLMAP camera model name (e.g. `PINHOLE`, `OPENCV_FISHEYE`). See *Per-Sensor Intrinsics* below. |
| `camera_params` | array of floats | Optional positional camera parameter array for `camera_model_name`. See *Per-Sensor Intrinsics* below. |

### Images and Frames

An image belongs to the first sensor in the file whose `image_prefix` its
path, relative to the folder of the file, starts with. Same-instant captures
are grouped into a **frame** by their shared *frame key*, the part of the
image path that remains after stripping the sensor's `image_prefix`:
`fisheye_left/frame_07.jpg` and `fisheye_right/frame_07.jpg` both strip down
to `frame_07.jpg` and are one frame. A `.camrig` groups the images of a frame
by the number its sensors' image patterns read instead, so the converter
checks that the patterns it derives give the same frames
([camrig-command.md § A `rig_config.json` source](../cli/workspace/camrig-command.md#a-rig_configjson-source)).

## Validation

`validate_rig_config` in
[`rig/config.py`](../../src/sfmtool/rig/config.py) validates the whole
document before the converter uses any of it and raises `RigConfigError` on
the first violation, which `sfm camrig cp` reports as a plain CLI error.

Because the schema is COLMAP's verbatim, **any key outside it is an error**
rather than something to ignore: a skipped key would leave intrinsics out of
the `.camrig` with nothing in the output to mark it. The message names the
offending key, its location (rig index, camera index), and the schema key that
replaces it when the intent is recognizable — a rig-level `camera_intrinsics`
block is answered with per-sensor `camera_model_name` + `camera_params`.

The rules enforced:

| Scope | Rule |
|---|---|
| file | The top level is a JSON array of rig objects, holding at least one rig. |
| rig | `cameras` is the only permitted key, is required, and holds at least one sensor entry. |
| rig | Exactly one sensor sets `ref_sensor: true`. |
| sensor | Keys are limited to the six in [*The Camera (Sensor) Entry*](#the-camera-sensor-entry). |
| sensor | `image_prefix` is required, is a non-empty string, contains no backslash, and is distinct across the whole file. |
| sensor | `ref_sensor`, when present, is a boolean; absent, it is `false`. |
| sensor | The reference sensor carries neither `cam_from_rig_rotation` nor `cam_from_rig_translation`: its `cam_from_rig` is the identity, so a pose written there would have no effect. |
| sensor | Every non-reference sensor carries `cam_from_rig_rotation` — four numbers, not all zero. |
| sensor | `cam_from_rig_translation`, when present, is three numbers. |
| sensor | `camera_model_name`, when present, is one of the COLMAP camera models in `CAMERA_MODEL_NAMES`. |
| sensor | `camera_params`, when present, is an array of numbers, requires `camera_model_name`, and has exactly the positional length that model declares. |

## Relationship to the COLMAP Rig Format

Every field documented here — the top-level array of rig objects, `cameras`,
`image_prefix`, `ref_sensor`, `cam_from_rig_rotation`,
`cam_from_rig_translation`, `camera_model_name`, `camera_params` — has the
same name, type, and semantics as in COLMAP, so a file authored for COLMAP is
read unchanged. Two consequences:

- **Intrinsics are per sensor**, given by `camera_model_name` (a COLMAP
  camera model name) and `camera_params` (a flat positional float array in
  COLMAP's parameter order). There is no rig-level intrinsics block, and the
  file carries no image `width`/`height`: the image dimensions come from the
  images themselves. A file that carries either is refused; see
  [*Validation*](#validation).
- **The WXYZ quaternion convention is COLMAP's.** COLMAP's documented
  example `[0.7071, 0, 0.7071, 0]` is a 90° rotation about Y in
  `[w, x, y, z]` order.

## Coordinate Convention

`rig_config.json` is in **COLMAP convention**: its `cam_from_rig` poses map
into COLMAP-style camera frames (camera looks down **+Z** with **Y down**),
*not* the canonical −Z-forward frames used by `.sfmr` and `.camrig` (see the
"Coordinate System Conventions" section of
[`sfmr-file-format.md`](../formats/sfmr-file-format.md)). The converter
conjugates each pose with the camera-frame flip `S = diag(1, −1, −1)` into
the canonical `sensor_from_rig` a `.camrig` holds; rig-relative poses never
touch the world frame.

## Quaternion Convention

`cam_from_rig_rotation` is stored as **WXYZ** to match the COLMAP rig file
format.

| Rotation | WXYZ |
|---|---|
| identity | `[1, 0, 0, 0]` |
| 180° about X | `[0, 1, 0, 0]` |
| 180° about Y | `[0, 0, 1, 0]` |
| 180° about Z | `[0, 0, 0, 1]` |

A back-to-back rig (right sensor pointing opposite the left) flips the
optical axis with `[0, 0, 1, 0]` (180° about Y, "look behind"), as in the
example above. (A 180° rotation about Y reads the same in both camera-frame
conventions — `S · Ry(180°) · S = Ry(180°)` — but the translation does not:
`[0, 0, -0.0307]` there is the COLMAP-convention value; see
[*Coordinate Convention*](#coordinate-convention).)

## Per-Sensor Intrinsics

Each sensor entry may carry its own intrinsics, exactly as in COLMAP:

- `camera_model_name` — a COLMAP camera model name (`PINHOLE`,
  `OPENCV_FISHEYE`, …).
- `camera_params` — the positional parameter array for that model, in
  COLMAP's parameter order. For `OPENCV_FISHEYE` that order is
  `fx, fy, cx, cy, k1, k2, k3, k4`.

The converter builds each sensor's camera on the sensor's first image:

- **Full calibration** (`camera_model_name` + `camera_params`) — the
  parameters are the camera, at that image's size.
- **Model only** (`camera_model_name`, no `camera_params`) — the camera is
  inferred from the image's EXIF with that model. For a fisheye model the
  focal length is seeded to about `min(width, height) / π` rather than the
  EXIF-derived guess, which is far too long for a ~180° projection.
- **Neither** — the camera is inferred from the image's EXIF, usually a
  rectilinear guess.

There is no distortion-only tier: COLMAP's `camera_params` is all-or-nothing
positional.

## A `rig_config.json` in a Workspace

A workspace's rigs are described only by `.camrig` files. Every command that
sets up the cameras and rig frames of a workspace's images — `sfm solve`,
`sfm match` (every mode but `--merge`), `sfm to-colmap-db` from a `.matches`
file, `sfm pick-images` and `sfm estimate-intrinsics
--write-camrig-per-sequence` — stops before it reads any image when the
workspace root holds a `rig_config.json`, rather than solving its rigs' images
as single images. The rule looks only at the workspace root and does not
look at `.camrig` files: a workspace whose rig a `.camrig` already describes
is refused too, until the `rig_config.json` is deleted. The check is
`_refuse_rig_config` in [`rig/config.py`](../../src/sfmtool/rig/config.py).

The error names the file and gives the command line that converts it, with
an output `.camrig` beside it named after the workspace folder:

```
Error: /captures/kp/rig_config.json: sfm takes a workspace's rigs only from .camrig files and does not read rig_config.json. Convert it with
  sfm camrig cp /captures/kp/rig_config.json /captures/kp/kp.camrig
and then delete it. If a .camrig already describes its rigs, delete it.
```

A file of several rigs gets one line per rig, each with `--rig N` and an
output named `<folder>_rig<N>.camrig`, and a path with a space is written in
double quotes.

## Converting to a `.camrig`

`sfm camrig cp rig_config.json OUTPUT.camrig [--rig N]` converts one rig of
the file into a `.camrig` that groups the same images into the same sensors
and frames, with the same cameras and `cam_from_rig` poses. Each sensor's
image pattern is derived from the image files under its `image_prefix`, and
the command fails, naming the images, when no pattern can reproduce the
file's sensors and frames. Each rig of a file with several rigs becomes its
own `.camrig`. See
[camrig-command.md § A `rig_config.json` source](../cli/workspace/camrig-command.md#a-rig_configjson-source).
