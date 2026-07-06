# Rig Config Specification

## Overview

`rig_config.json` is an optional file that describes multi-sensor camera rigs —
sets of cameras that move together as a single physical assembly. Common
examples: a stereo pair, a 360° back-to-back fisheye rig, a vehicle-mounted
multi-camera array, a cubemap derived from a panorama via `sfm pano2rig`.

Without `rig_config.json`, every image in a workspace is treated as an
independent free-floating camera. With it, sfmtool can:

- Use the calibrated sensor-to-rig poses as priors during reconstruction.
- Group same-instant captures into rig **frames** and skip same-frame image
  pairs during matching (their relative pose is fixed by the rig).
- Optionally refine the rig poses themselves (`refine_rig=True` is the
  default for both incremental and global solves).

A single file may declare multiple independent rigs.

The file lives at the workspace root: commands read only
`{workspace_dir}/rig_config.json`. Unlike `camera_config.json`, there is no
closest-ancestor search in subdirectories.

`rig_config.json` **is** COLMAP's
[`rig_configurator` rig config](https://colmap.github.io/rigs.html) file —
sfmtool reads and writes exactly that format, with no sfmtool-specific
extensions. See [*Relationship to the COLMAP Rig Format*](#relationship-to-the-colmap-rig-format)
below. That includes the coordinate convention: unlike sfmtool's own
formats, this file stays in **COLMAP convention** — see
[*Coordinate Convention*](#coordinate-convention).

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

### Top-Level Shape

The file is a JSON **array** of rig objects, even when only one rig is
present. This matches the COLMAP rig file shape and lets a workspace declare
multiple rigs that share the same image tree.

### The Rig Object

| Field | Type | Description |
|-------|------|-------------|
| `cameras` | array | One entry per sensor in the rig. Exactly one must have `ref_sensor: true`. |

### The Camera (Sensor) Entry

| Field | Type | Description |
|-------|------|-------------|
| `image_prefix` | string | Path prefix (relative to the workspace) that identifies images captured by this sensor. Forward slashes only. |
| `ref_sensor` | boolean | `true` for exactly one sensor per rig. The reference sensor defines the rig coordinate frame; its `cam_from_rig` is the identity. |
| `cam_from_rig_rotation` | array of 4 floats | **WXYZ** quaternion (see *Quaternion Convention*). Required for non-reference sensors. Omitted for the reference sensor. |
| `cam_from_rig_translation` | array of 3 floats | Translation in metres. Defaults to `[0, 0, 0]` if omitted. |
| `camera_model_name` | string | Optional COLMAP camera model name (e.g. `PINHOLE`, `OPENCV_FISHEYE`). See *Per-Sensor Intrinsics* below. |
| `camera_params` | array of floats | Optional positional camera parameter array for `camera_model_name`. See *Per-Sensor Intrinsics* below. |

### Frame Grouping

Same-instant captures are grouped into a **frame** by their shared
*frame key* — the part of the image path that remains after stripping the
sensor's `image_prefix`. In the kerry_park example, `fisheye_left/frame_07.jpg`
and `fisheye_right/frame_07.jpg` both strip down to `frame_07.jpg` and are
joined into one frame.

Within a frame, image pairs are excluded from feature matching: their
relative pose is fixed by the rig and the match would not contribute new
geometric information. Cross-frame pairs are matched normally.

## Relationship to the COLMAP Rig Format

`rig_config.json` is the COLMAP
[`rig_configurator` rig config](https://colmap.github.io/rigs.html) file
verbatim. Every field documented here — the top-level array of rig objects,
`cameras`, `image_prefix`, `ref_sensor`, `cam_from_rig_rotation`,
`cam_from_rig_translation`, `camera_model_name`, `camera_params` — has the
same name, type, and semantics as in COLMAP. A file authored for one tool is
accepted unchanged by the other.

Two consequences of matching COLMAP exactly:

- **Intrinsics are per sensor**, given by `camera_model_name` (a COLMAP
  camera model name) and `camera_params` (a flat positional float array in
  COLMAP's parameter order). There is no rig-level intrinsics block, and the
  config carries no image `width`/`height` — COLMAP and sfmtool both take
  image dimensions from the images themselves.
- **The WXYZ quaternion convention is COLMAP's.** COLMAP's documented
  example `[0.7071, 0, 0.7071, 0]` is a 90° rotation about Y in
  `[w, x, y, z]` order.

sfmtool does not invoke `colmap rig_configurator`; it builds rigs directly
through pycolmap. Matching the format exactly means the same
`rig_config.json` can be fed to either tool.

## Coordinate Convention

`rig_config.json` remains in **COLMAP convention** — its `cam_from_rig`
poses map into COLMAP-style camera frames (camera looks down **+Z** with
**Y down**), *not* the canonical −Z-forward frames used by `.sfmr` and
`.camrig` (see the "Coordinate System Conventions" section of
[`sfmr-file-format.md`](../formats/sfmr-file-format.md)). This is
deliberate: the file mirrors COLMAP's own rig-config schema and exists to
feed COLMAP database setup, so it is treated as a COLMAP-side artifact and
converted at ingestion like every other COLMAP input (conjugation with the
camera-frame flip `S = diag(1, −1, −1)`; rig-relative poses never touch
the world frame). Keeping the file COLMAP-convention means a
`rig_config.json` authored for COLMAP — including
`test-data/images/kerry_park/rig_config.json` — stays valid unchanged.

This is the migration plan's decision D4
(`specs/drafts/zup-camera-convention-migration.md`); contrast `.camrig`,
sfmtool's own rig format, which adopts the canonical convention (see
[`camrig-file-format.md`](../formats/camrig-file-format.md)).

## Quaternion Convention

`cam_from_rig_rotation` is stored as **WXYZ** to match the COLMAP rig file
format. sfmtool converts to **XYZW** internally when handing the value to
pycolmap. The conversion lives in `_sensor_from_rig_pose` in
`src/sfmtool/rig/config.py`.

Examples:

| Rotation | WXYZ |
|---|---|
| identity | `[1, 0, 0, 0]` |
| 180° about X | `[0, 1, 0, 0]` |
| 180° about Y | `[0, 0, 1, 0]` |
| 180° about Z | `[0, 0, 0, 1]` |

A back-to-back rig (right sensor pointing opposite the left) flips the
optical axis with `[0, 0, 1, 0]` (180° about Y, "look behind"). This is
the value used by the kerry_park dataset. (A 180° rotation about Y happens
to read the same in both camera-frame conventions — `S · Ry(180°) · S =
Ry(180°)` — but the translation does not: `[0, 0, -0.0307]` here is the
COLMAP-convention value; see [*Coordinate Convention*](#coordinate-convention).)

## Per-Sensor Intrinsics

Each sensor entry may carry its own intrinsics, exactly as in COLMAP:

- `camera_model_name` — a COLMAP camera model name (`PINHOLE`,
  `OPENCV_FISHEYE`, …).
- `camera_params` — the positional parameter array for that model, in
  COLMAP's parameter order. For `OPENCV_FISHEYE` that order is
  `fx, fy, cx, cy, k1, k2, k3, k4`.

Two tiers are supported, mirroring COLMAP:

- **Model only** (`camera_model_name`, no `camera_params`) — the model is a
  hint; focal length, principal point, and distortion are inferred from
  EXIF (or seeded — see *Fisheye and Wide-FOV Notes*).
- **Full calibration** (`camera_model_name` + `camera_params`) — the
  parameters are used directly and the camera is flagged with a prior focal
  length. `camera_params` must have the exact length COLMAP expects for the
  model, or loading fails.

Unlike `camera_config.json`, there is no distortion-only tier: COLMAP's
`camera_params` is all-or-nothing positional.

If a sensor entry has neither field, that sensor falls back to
`pycolmap.infer_camera_from_image()` on its first image — usually a
rectilinear EXIF guess, which is wrong for fisheye and wide-FOV sensors.

## Interaction with `camera_config.json`

`rig_config.json` wins for any image whose path matches one of its
`image_prefix` entries. `camera_config.json` then handles the remainder. See
[camera-config.md § Interaction with `rig_config.json`](camera-config.md#interaction-with-rig_configjson).

## Interaction with `--camera-model`

`--camera-model` is accepted alongside `rig_config.json` rather than
rejected, but when supplied it takes precedence over the config. If
`--camera-model` is given, every sensor is built by
`pycolmap.infer_camera_from_image()` under that model — even sensors that
carry a full `camera_model_name` + `camera_params` calibration, whose
`camera_params` are then ignored. With no `--camera-model`, each sensor
uses its own `camera_model_name` / `camera_params`, falling back to
inference (with `camera_model_name` as a hint) when those are absent.

This is intentionally laxer than the `camera_config.json` interaction
(which hard-rejects `--camera-model`), because rig configs commonly omit
intrinsics for cameras that don't yet have a calibration. The trade-off is
that an explicit `--camera-model` overrides calibrated rig intrinsics; pass
it only when you want to re-infer every sensor.

## Fisheye and Wide-FOV Notes

For ~180° FOV fisheye rigs (e.g. OPENCV_FISHEYE with k1–k4):

- **Undistortion blends to identity past ~90°** off-axis. See
  `crates/sfmtool-core/src/camera/distortion.rs` (search for
  `blend_fisheye_ray`). The k1–k4 polynomial is unreliable at extreme
  angles; sfmtool smoothstep-blends the recovered ray to the identity
  (small-angle) ray over 90°–100°. Solver output near the image border
  reflects this — features projected behind a sensor produce
  `Ignoring feature because it failed to project` log lines from
  pycolmap; this is benign cull-back, not a numerical error.
- **Focal-length inference is fisheye-aware.** When inferring intrinsics
  for an OPENCV_FISHEYE camera without an explicit focal length,
  `camera.setup._infer_camera` seeds the focal length to roughly
  `min(width, height) / π` instead of the EXIF-derived 1.2× image-size
  guess, which is far too long for a ~180° projection.
- **Same-frame matching is skipped.** For back-to-back fisheye rigs the
  two sensors share no field of view; matching them would be wasteful
  even without the rig pose constraint. The cross-frame pair builder in
  `rig.frames._build_cross_frame_pairs` handles this naturally.

## Examples in the Repository

- `test-data/images/kerry_park/rig_config.json` — back-to-back 180° FOV
  fisheye stereo pair (`OPENCV_FISHEYE`, 480×480, 30.7 mm baseline).

`rig_config.json` is consumed as an *input* by the solving commands, but
sfmtool no longer *produces* it. The rig builders (`sfm pano2rig`,
`sfm insv2rig`, `sfm camrig …`) all emit a `.camrig` file instead — see
`specs/formats/camrig-file-format.md` and `specs/cli/pano2rig-command.md`.
