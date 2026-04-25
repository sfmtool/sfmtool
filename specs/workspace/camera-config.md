# Camera Config Specification

## Overview

`camera_config.json` is an optional file that supplies explicit camera intrinsics
(focal length, principal point, distortion parameters) for images in a workspace.
Without it, sfmtool falls back to `pycolmap.infer_camera_from_image()`, which reads
EXIF and assumes a rectilinear projection — a guess that is fine for a first pass
but usually leaves enough error in the focal length and distortion model to hurt
reconstruction quality.

A `camera_config.json` file lets you commit known-good intrinsics — typically
recovered from a successful prior reconstruction — back to the workspace so that
subsequent solves start from the right answer instead of re-discovering it.

## File Format

```json
{
  "version": 1,
  "camera_intrinsics": {
    "model": "OPENCV",
    "width": 4096,
    "height": 3072,
    "parameters": {
      "focal_length_x": 3201.4,
      "focal_length_y": 3199.8,
      "principal_point_x": 2047.1,
      "principal_point_y": 1535.6,
      "radial_distortion_k1": -0.0214,
      "radial_distortion_k2": 0.0089,
      "tangential_distortion_p1": 0.00012,
      "tangential_distortion_p2": -0.00007
    }
  }
}
```

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | Format version. Must be `1` |
| `camera_intrinsics` | object | The intrinsics block (see below) |

### The `camera_intrinsics` Block

This is the same shape used by the optional `camera_intrinsics` block in
`rig_config.json`, so a single Python helper handles both.

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | COLMAP camera model name (e.g. `PINHOLE`, `OPENCV`, `OPENCV_FISHEYE`, `FULL_OPENCV`) |
| `width` | integer | Image width in pixels |
| `height` | integer | Image height in pixels |
| `parameters` | object | Model-specific named parameters |

The parameter names per model are defined in `_CAMERA_PARAM_NAMES` in
`src/sfmtool/_cameras.py`. For example, `PINHOLE` takes `focal_length_x`,
`focal_length_y`, `principal_point_x`, `principal_point_y`, while `OPENCV` adds
`radial_distortion_k1`, `radial_distortion_k2`, `tangential_distortion_p1`, and
`tangential_distortion_p2`.

### Partial Intrinsics

`width`, `height`, and `parameters` are all optional. The file is still
authoritative whether or not those fields are present; only the level of detail
changes. Useful tiers, from least to most specific:

- **Model only.** Sets the distortion projection (e.g. switch a fisheye lens
  out of the rectilinear default). Focal length, principal point, and any
  distortion coefficients are inferred from EXIF or left at their COLMAP
  defaults.
  ```json
  { "version": 1, "camera_intrinsics": { "model": "OPENCV_FISHEYE" } }
  ```

- **Model + distortion-only `parameters`.** Distortion coefficients are
  scale-invariant (they act on normalized image coordinates), so they can be
  committed without specifying a resolution. Focal length and principal point
  still come from EXIF.
  ```json
  {
    "version": 1,
    "camera_intrinsics": {
      "model": "OPENCV",
      "parameters": {
        "radial_distortion_k1": -0.0214,
        "radial_distortion_k2": 0.0089,
        "tangential_distortion_p1": 0.00012,
        "tangential_distortion_p2": -0.00007
      }
    }
  }
  ```

- **Full calibration.** `model + width + height + parameters`. The `width` and
  `height` describe the resolution at which the camera was calibrated, not a
  required runtime resolution. See *Resolution Mismatch* below.

When any focal length or principal point value is supplied in `parameters`,
`width` and `height` are required — those quantities are in pixels and only
make sense relative to a known image size.

### Resolution Mismatch

You may want to calibrate at one resolution (typically the camera's native
sensor resolution) and run SfM at a downsampled resolution for speed. The
file's `width` and `height` describe the *calibration* resolution; sfmtool
adapts to whatever resolution the actual image happens to be at:

1. **Actual size matches calibration size.** Intrinsics are used verbatim.
2. **Actual size differs, aspect ratio matches** (`actual_w / actual_h ==
   calib_w / calib_h` within a small tolerance). Focal length and principal
   point are scaled uniformly by `s = actual_w / calib_w`; distortion
   coefficients pass through unchanged.
3. **Aspect ratio differs.** sfmtool refuses to scale and the command exits
   with an error explaining the mismatch. Recalibrate, crop the runtime
   images to the calibration aspect, or remove `width`/`height` from the file
   and accept EXIF focal length.

```json
{
  "version": 1,
  "camera_intrinsics": {
    "model": "OPENCV",
    "width": 6000,
    "height": 4000,
    "parameters": {
      "focal_length_x": 4800.2,
      "focal_length_y": 4798.5,
      "principal_point_x": 2999.6,
      "principal_point_y": 1999.4,
      "radial_distortion_k1": -0.0214,
      "radial_distortion_k2": 0.0089,
      "tangential_distortion_p1": 0.00012,
      "tangential_distortion_p2": -0.00007
    }
  }
}
```

Applied to 1500×1000 runtime images (uniform 4× downsample), the effective
intrinsics become `fx ≈ 1200.05`, `fy ≈ 1199.6`, `cx ≈ 749.9`, `cy ≈ 499.85`,
distortion unchanged.

Within one `camera_config.json` scope, all images are expected to share the
same calibration. If individual images have different runtime resolutions
(e.g. mixed full-res and thumbnail), each is scaled independently from the
shared calibration.

## Where to Place `camera_config.json`

A `camera_config.json` file applies to all images in the directory that contains
it, and recursively to all subdirectories — unless a deeper subdirectory contains
its own `camera_config.json`, which takes precedence for images under that
subdirectory. Resolution is closest-ancestor-wins, similar to `.gitignore`.

Workspace root is the upper bound: the search never crosses out of the
workspace. If no `camera_config.json` is found between the image's parent
directory and the workspace root, sfmtool falls back to EXIF inference (and
`--camera-model` if provided).

```
my_project/
├── .sfm-workspace.json
├── camera_config.json                  # applies to everything by default
├── nikon_z6/
│   ├── DSC_0001.JPG
│   └── DSC_0002.JPG
├── gopro/
│   ├── camera_config.json              # overrides the root one for GoPro shots
│   ├── G0010001.JPG
│   └── G0010002.JPG
└── drone/
    ├── camera_config.json              # different camera again
    └── DJI_0001.JPG
```

In this layout, every image under `nikon_z6/` resolves to the workspace-root
`camera_config.json`, every image under `gopro/` resolves to
`gopro/camera_config.json`, and every image under `drone/` resolves to
`drone/camera_config.json`.

## How Commands Use `camera_config.json`

Any command that creates camera entries from images consults
`camera_config.json` during camera setup. This currently includes:

- `sfm solve` (incremental and global)
- `sfm match` (when computing two-view geometries)
- `sfm to-colmap-db`
- `sfm densify` (when adding cameras for newly-registered images)

Commands that operate on existing `.sfmr` files (`sfm xform`, `sfm inspect`,
`sfm align`, `sfm merge`, the GUI viewer) do **not** consult
`camera_config.json` — once a `.sfmr` exists, its embedded camera intrinsics are
authoritative. To apply updated intrinsics to an existing reconstruction, use
`sfm xform --switch-camera-model` or re-solve.

### One Camera per Source Directory

When several images map to the same `camera_config.json`, sfmtool creates a
single COLMAP camera and assigns every matching image to it. This matches
the typical case (one physical camera produces one `nikon_z6/` directory of
photos) and is consistent with how `infer_camera_from_image()` is used today
when the `--single-camera-per-folder` heuristic applies.

If you have two distinct cameras whose images live in the same folder, separate
them into subfolders before solving.

## Interaction with `--camera-model`

The rule is presence-based, not content-based: if any image being processed
resolves a `camera_config.json`, that file is the sole source of camera setup
for those images, and passing `--camera-model` for the same invocation is an
error.

| Situation | Behavior |
|-----------|----------|
| No image resolves a `camera_config.json` | `--camera-model` applies as today; EXIF inference fills in parameters |
| All resolved images share a `camera_config.json`, no `--camera-model` flag | File is used (full, partial, or model-only — whatever it specifies) |
| Any resolved image has a `camera_config.json` AND `--camera-model` is set | Command exits with an error before doing any work |

This makes the outcome predictable from the caller's side: looking at the
filesystem (and the input image set) is enough to know whether `--camera-model`
will apply. The two mechanisms never interleave.

To force EXIF inference with a CLI override, delete or rename the file. To
change the model permanently, edit the file. Mixing is intentionally not
supported.

If different images in one invocation resolve to different `camera_config.json`
files (e.g. multiple cameras in one workspace), each image uses its own
resolved file. `--camera-model` is still rejected if *any* of them resolves a
file — it would be ambiguous which subset the flag was meant to override.

## Interaction with `rig_config.json`

When both files are present, `rig_config.json` wins for any image whose path
matches one of its `image_prefix` entries. This preserves rig semantics:
sensor-to-rig poses, multi-camera frames, and per-sensor intrinsics are all
defined together in `rig_config.json`. `camera_config.json` then handles the
remainder — images that don't belong to any rig.

A future spec change may unify the two files. See *Future Unification* below.

## Workflows

### Workflow 1: Bootstrap from an Unknown Camera

You have a folder of images and no calibration. Solve a subset, harvest the
intrinsics, then solve the full set with the harvested values as priors.

```bash
# 1. Initialize the workspace and extract features
sfm init my_project
sfm sift --extract my_project/photos

# 2. Solve a small, well-conditioned subset (good baseline, lots of overlap)
sfm solve -i my_project/photos --range 100:130 \
    --output my_project/sfmr/calibration_subset.sfmr

# 3. Open it in SfM Explorer and visually compare the reconstruction to the
#    source images. Look for obvious wrongness: bent walls, drifting tracks,
#    points floating off surfaces. Then clean up the cloud and re-fit the
#    cameras to the surviving points.
sfm explorer my_project/sfmr/calibration_subset.sfmr
sfm xform my_project/sfmr/calibration_subset.sfmr \
    --remove-narrow-tracks 7deg \
    --remove-isolated 5,median \
    --remove-large-features 4 \
    --bundle-adjust \
    --align-to-input \
    --output my_project/sfmr/calibration_subset_cleaned.sfmr

# 4. Save the cleaned reconstruction's intrinsics into the workspace
sfm cam cp my_project/sfmr/calibration_subset_cleaned.sfmr \
    my_project/camera_config.json

# 5. Re-solve from scratch — now with priors
sfm solve -i my_project/photos
```

The point of step 3 is that bundle-adjusting after outlier removal produces
a more accurate intrinsics fit than the original solve. The first solve has
to compromise its parameters to accommodate every track it accepted, including
narrow-baseline ones, outliers, and oversized features. Filtering those out
and re-running BA lets the solver re-fit the intrinsics against the remaining
high-quality observations, which is what we want to harvest.
`--align-to-input` keeps the cleaned reconstruction in the same coordinate
frame as the original so it remains comparable.

The intent of step 4 is captured by a future `sfm cam cp` command;
until that lands, the file can be written by hand from the values in
`sfm inspect --metrics` output, or from a small script using
`pycolmap_camera_to_intrinsics`.

### Workflow 2: Reuse Intrinsics from a Previous Project

You've previously calibrated a camera and want to skip the bootstrap step.

```bash
sfm init new_project
cp ~/calibrations/nikon_z6_24mm/camera_config.json new_project/

sfm sift --extract new_project/photos
sfm solve -i new_project/photos
```

The first solve already starts from the right focal length and distortion model.
This is the cheapest way to get good results from a familiar camera.

### Workflow 3: Multiple Cameras in One Workspace

A common scenario: you've shot the same scene with a phone, a DSLR, and a drone,
and want to reconstruct from all three. Each source directory gets its own
`camera_config.json`.

```
backyard/
├── .sfm-workspace.json
├── phone/
│   ├── camera_config.json       # iPhone wide lens
│   └── IMG_*.HEIC
├── dslr/
│   ├── camera_config.json       # Nikon Z6 with 24mm prime
│   └── DSC_*.JPG
└── drone/
    ├── camera_config.json       # DJI Mavic, fisheye disabled
    └── DJI_*.JPG
```

```bash
sfm sift --extract backyard/phone backyard/dslr backyard/drone
sfm solve -i backyard
```

Each image resolves its intrinsics through the closest-ancestor rule, so a
single `sfm solve` invocation produces three camera entries in the database —
one per source directory — with the right starting parameters for each.

### Workflow 4: Locking Intrinsics for a Specific Solve

There's no flag in this spec for fixing intrinsics during bundle adjustment —
that remains a solver concern. But because the values from
`camera_config.json` are written with `has_prior_focal_length = true`, the
solver weights them more heavily. If you need them truly fixed, pair the file
with the appropriate solver flags (this is a future extension; not in scope
here).

### Open Question: Comparing Calibrations Side by Side

A natural follow-on workflow is reconstructing one image set under two
candidate calibrations and comparing the results. The closest-ancestor rule
in this spec only addresses one calibration per directory at a time, which
makes the question "how do I keep two of them addressable?" an organization
problem we haven't yet decided.

Some candidate approaches:

- **Rename in place.** Keep `calibration_a.json` and `calibration_b.json`
  archived in the workspace; rename the chosen one to `camera_config.json`
  before each solve. Cheap (the files are small), but manual and easy to
  lose track of which is active.
- **`--camera-config PATH` flag.** Let a single invocation point at an
  alternate file. Reopens the precedence question with the existing
  presence-based rule and would need its own design pass.
- **Named active calibration in `.sfm-workspace.json`.** Have the workspace
  config declare which named camera config is active, with the pool of
  candidates living next to it.

None of these are committed to in this spec. Once we have practical
experience with the basic closest-ancestor flow, we can pick the
organization that matches how A/B testing actually shows up.

## Future Unification

`rig_config.json` and `camera_config.json` overlap in their treatment of
intrinsics: the `camera_intrinsics` block defined here has the same shape as
the optional block on each rig entry. A future revision may unify them by:

1. Allowing `camera_config.json` entries to also describe rig topology
   (sensors, sensor-from-rig poses), or
2. Treating a directory-local `camera_config.json` as a degenerate
   single-sensor rig, or
3. Replacing both files with a more general `cameras.json` keyed by either
   `image_prefix` or directory containment.

This spec stays narrowly scoped to single-camera intrinsics so that we can
deploy the closest-ancestor resolution rule and the workflows above without
blocking on the larger unification design.

## Design Principles

1. **Authoritative when present.** A full `camera_config.json` is a commitment
   to use those values; CLI flags do not override it silently.
2. **Closest ancestor wins.** Per-directory placement keeps the relationship
   between an image and its camera obvious from the filesystem layout.
3. **Versioned schema.** `version: 1` lets us evolve the format without
   breaking existing files.
4. **Pairs naturally with `.sfmr` snapshots.** Intrinsics flow out of solves
   (`.sfmr` files) and back into the workspace (`camera_config.json`) without
   transformation — both store the same `{model, width, height, parameters}`
   shape.
5. **Small and human-editable.** A `camera_config.json` is short enough to
   inspect, diff, and edit by hand. No binary blobs, no lock files.
