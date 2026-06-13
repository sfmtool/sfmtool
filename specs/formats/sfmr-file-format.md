# SfM Reconstruction File Format Specification

## Overview

The `.sfmr` file format provides a portable, self-contained format for storing Structure from Motion (SfM) reconstructions. It follows the same design principles as the `.sift` file format, using ZIP archives with zstandard compression, columnar binary storage, and content hashing for integrity.

## Design Principles

1. **ZIP file with no compression** - Uses ZIP's STORE method for random access to individual files
2. **Zstandard compression** - All data files have `.zst` extension and use zstandard compression
3. **JSON metadata** - Human-readable metadata in JSON format
4. **Content hash verification** - XXH128 hashes for integrity checking
5. **Columnar storage** - One file per field, enabling selective loading
6. **Self-documenting filenames** - Include tensor shapes and data types (e.g., `positions_xyzw.2107.4.float64.zst`)
7. **Little-endian binary** - All numeric data in C/row-major order
8. **Tool agnostic** - Works with any SfM solver (COLMAP, GLOMAP, OpenMVG, etc.)

This format largely adopts the semantics of the [COLMAP Output Format](https://colmap.github.io/format.html)
as the basis for a reconstruction. It uses indexes that are always a
contiguous range from 0 to N-1, different from the potentially non-contiguous IDs in a 
COLMAP binary file starting from 1.

## File Structure

The `.sfmr` file is a ZIP archive (using STORE method) with the following structure:

```
reconstruction.sfmr (ZIP archive)
├── metadata.json.zst                          # Top-level reconstruction metadata
├── content_hash.json.zst                      # Integrity verification hashes
├── cameras/
│   └── metadata.json.zst                      # Camera intrinsics and parameters
├── rigs/                                      # (Optional) Camera rig definitions
│   ├── metadata.json.zst                      # Rig metadata
│   ├── sensor_camera_indexes.{S}.uint32.zst   # Camera index per sensor
│   ├── sensor_quaternions_wxyz.{S}.4.float64.zst  # Sensor-from-rig rotations
│   └── sensor_translations_xyz.{S}.3.float64.zst  # Sensor-from-rig translations
├── frames/                                    # (Optional) Frame groupings
│   ├── metadata.json.zst                      # Frame metadata
│   ├── rig_indexes.{F}.uint32.zst             # Which rig each frame uses
│   ├── image_sensor_indexes.{N}.uint32.zst    # Which sensor captured each image
│   └── image_frame_indexes.{N}.uint32.zst     # Which frame each image belongs to
├── images/
│   ├── names.json.zst                         # Image file paths
│   ├── camera_indexes.{N}.uint32.zst          # Camera assignment per image
│   ├── quaternions_wxyz.{N}.4.float64.zst     # Camera rotations (WXYZ format)
│   ├── translations_xyz.{N}.3.float64.zst     # Camera translations (XYZ format)
│   ├── feature_tool_hashes.{N}.uint128.zst    # Feature extraction tool identification
│   ├── sift_content_hashes.{N}.uint128.zst    # Feature file content verification
│   ├── thumbnails_y_x_rgb.{N}.128.128.3.uint8.zst # 128x128 image thumbnails (RGB)
│   ├── metadata.json.zst                      # Image metadata
│   ├── depth_statistics.json.zst                       # Per-image depth stats
│   └── observed_depth_histogram_counts.{N}.128.uint32.zst  # Observed depth histograms
├── points3d/
│   ├── positions_xyzw.{N}.4.float64.zst       # Homogeneous 3D point coordinates (w=0 = point at infinity)
│   ├── colors_rgb.{N}.3.uint8.zst             # RGB colors (0-255)
│   ├── reprojection_errors.{N}.float32.zst    # Reprojection errors
│   ├── metadata.json.zst                      # Points metadata
│   └── estimated_normals_xyz.{N}.3.float32.zst # Estimated point normals
└── tracks/
    ├── image_indexes.{M}.uint32.zst           # Image index per observation
    ├── feature_indexes.{M}.uint32.zst         # Feature index per observation
    ├── point_indexes.{M}.uint32.zst           # Point index per observation
    ├── observation_counts.{N}.uint32.zst      # Observations per point
    └── metadata.json.zst                      # Tracks metadata
```

Where:
- `{N}` = number of items (images, points, etc.)
- `{S}` = total number of sensors across all rigs
- `{F}` = number of frames (temporal instants)
- `{M}` = number of observations (track elements)

## File Format Details

### 1. Top-Level Metadata (`metadata.json.zst`)

JSON structure describing the reconstruction:

```json
{
  "version": 2,
  "operation": "sfm_solve",
  "tool": "colmap",
  "tool_version": "3.10",
  "tool_options": {
    "Mapper.ba_refine_focal_length": true,
    "Mapper.ba_refine_extra_params": true
  },
  "workspace": {
    "absolute_path": "/absolute/path/to/workspace",
    "relative_path": "../workspace",
    "contents": {
      "feature_tool": "colmap",
      "feature_type": "sift",
      "feature_options": {
        "domain_size_pooling": false,
        "max_num_features": null,
        "max_image_size": 4096,
        "estimate_affine_shape": false
      },
      "feature_prefix_dir": "features/sift-colmap-d1245b460906df27ee4730273e0aba41"
    }
  },
  "timestamp": "2025-12-21T14:32:15.123456Z",
  "image_count": 18,
  "point_count": 2107,
  "infinity_point_count": 12,
  "observation_count": 9427,
  "camera_count": 1,
  "rig_count": 1,
  "sensor_count": 6,
  "frame_count": 3
}
```

**Field descriptions:**
- `version`: Format version number. Must be `2` for this specification. See
  [Versioning and Migration](#versioning-and-migration) for the relationship
  to version `1`.
- `operation`: Type of operation that created this reconstruction
  - `"sfm_solve"`: Structure-from-Motion reconstruction
  - `"transform"`: Geometric transformation of an existing reconstruction
  - `"subset"`: Extracted subset of images from an existing reconstruction
  - `"merge"`: Merged multiple reconstructions into one
  - `"import"`: Imported from an external reconstruction format
- `tool`: Tool that performed the operation
  - For SfM operations: `"colmap"`, `"glomap"`, `"alicevision"`, `"openmvg"`, etc.
  - For manipulation operations: `"sfmtool"`
  - For imports: name of original tool, or `"unknown"`
- `tool_version`: Version string of the tool
- `tool_options`: Tool-specific configuration (key-value pairs, required, use empty object `{}` if none)
- `workspace`: SfM workspace information (embeds `.sfm-workspace.json` content)
  - `absolute_path`: Absolute path to workspace directory (at time of save)
  - `relative_path`: Path from `.sfmr` file's directory to workspace (POSIX format)
  - `contents`: Embedded workspace configuration (mirrors `.sfm-workspace.json` content)
    - `feature_tool`: Feature extraction tool used in workspace (e.g. `"colmap"`, `"opencv"`)
    - `feature_type`: Feature type (e.g. `"sift"`)
    - `feature_options`: Feature extraction options (see `.sift` format spec for details)
    - `feature_prefix_dir`: Relative path from each image's parent directory to the features subdirectory (e.g., `"features/sift-colmap-d1245b460906df27ee4730273e0aba41"`). Used to locate `.sift` files.
- `timestamp`: ISO 8601 format with timezone
- `image_count`: Number of registered images in reconstruction
- `point_count`: Number of points (finite points and points at infinity combined)
- `infinity_point_count`: Number of points at infinity (rows of `positions_xyzw`
  with `w = 0`). Derivable from the points array, but stored here so a consumer
  can read the finite/infinity split without decompressing that array. Must be
  `0` when no points are at infinity.
- `observation_count`: Total number of 2D-3D correspondences
- `camera_count`: Number of unique camera intrinsics
- `rig_count`: (Optional) Number of rig definitions. Present only when rig data exists.
- `sensor_count`: (Optional) Total sensors across all rigs. Present only when rig data exists.
- `frame_count`: (Optional) Number of frames (temporal instants). Present only when rig data exists.

**Important**: All paths in the `.sfmr` file (image paths in `images/names.json.zst`, etc.) are **relative to the workspace directory**, not relative to the `.sfmr` file itself. This ensures consistent path resolution regardless of where the `.sfmr` file is located.

#### Workspace Concept and Path Resolution

The workspace is the root directory for an SfM project, containing:
- Images or image directories
- Feature extraction configuration (`.sfm-workspace.json`)
- Optionally, the `.sfmr` reconstruction file(s)

**Why workspace-relative paths?**
1. **Portability**: The `.sfmr` file can be moved or shared independently
2. **Consistency**: All paths resolve the same way regardless of `.sfmr` file location
3. **Workspace context**: Embeds feature extraction settings used for reconstruction

**Path resolution:**
1. When reading a `.sfmr` file, first resolve the workspace location:
   - Try `workspace.relative_path` from `.sfmr` file's directory
   - Fall back to `workspace.absolute_path` if relative path fails
   - Fall back to the workspace containing the `.sfmr` file if both fail
2. All image paths are then resolved relative to the workspace directory

**Example**:
```
/projects/
├── my_workspace/
│   ├── .sfm-workspace.json
│   ├── frames/
│   │   ├── DSC_0001.JPG
│   │   └── DSC_0002.JPG
│   └── results/
│       └── reconstruction.sfmr
```

If `reconstruction.sfmr` has:
- `workspace.relative_path = ".."`
- `images/names.json.zst = ["frames/DSC_0001.JPG", ...]`

Then image path resolves to: `/projects/my_workspace/frames/DSC_0001.JPG`

If you copy the whole workspace to a different operating system, e.g. Linux to Windows,
the relative paths within the workspace will work. Similarly, if you move a `.sfmr` file within
the workspace, it will still work either falling back to the absolute path or the containing workspace directory.
If you move `reconstruction.sfmr` to `/home/user/reconstruction.sfmr`, which is outside the workspace,
it still finds the workspace because it first fails to resolve the relative path and then uses the absolute path.

### 2. Content Hash (`content_hash.json.zst`)

XXH128 hashes for integrity verification:

```json
{
  "metadata_xxh128": "1234567890abcdef1234567890abcdef",
  "cameras_xxh128": "...",
  "rigs_xxh128": "...",
  "frames_xxh128": "...",
  "images_xxh128": "...",
  "points3d_xxh128": "...",
  "tracks_xxh128": "...",
  "content_xxh128": "..."
}
```

**Field descriptions:**
- `metadata_xxh128`: Hash of the uncompressed JSON content of `metadata.json.zst`
- `cameras_xxh128`: Hash of the uncompressed JSON content of `cameras/metadata.json.zst`
- `rigs_xxh128`: (Optional) Hash of all rigs data files' uncompressed contents, fed sequentially into a streaming XXH128 hasher in lexicographic path order. Present only when `rigs/` section exists.
- `frames_xxh128`: (Optional) Hash of all frames data files' uncompressed contents, fed sequentially into a streaming XXH128 hasher in lexicographic path order. Present only when `frames/` section exists.
- `images_xxh128`: Hash of all image data files' uncompressed contents, fed sequentially into a streaming XXH128 hasher in lexicographic path order (includes depth statistics and histogram files)
- `points3d_xxh128`: Hash of all points3d data files' uncompressed contents, fed sequentially into a streaming XXH128 hasher in lexicographic path order (includes estimated normals)
- `tracks_xxh128`: Hash of all tracks data files' uncompressed contents, fed sequentially into a streaming XXH128 hasher in lexicographic path order
- `content_xxh128`: Hash of all present section hashes concatenated as raw 16-byte big-endian digests in order: metadata, cameras, rigs (if present), frames (if present), images, points3d, tracks.

**Note**: Per-section metadata files (`images/metadata.json.zst`, `points3d/metadata.json.zst`, `tracks/metadata.json.zst`) are included in their respective section hashes.

**Note**: The `metadata_xxh128` includes the workspace configuration, so changing workspace paths will invalidate the hash. This is intentional - the workspace context is part of the reconstruction's identity.

**Note**: All hashes are computed on the uncompressed content bytes. For JSON files, this is the compact JSON bytes (no pretty-printing) as originally written. Implementations verifying hashes MUST hash the raw bytes read from the archive after decompression, NOT re-serialized JSON, to avoid floating-point formatting differences across languages.

### 3. Cameras (`cameras/metadata.json.zst`)

Array of camera intrinsic parameters:

```json
[
  {
    "model": "PINHOLE",
    "width": 4624,
    "height": 3472,
    "parameters": {
      "focal_length_x": 3454.12,
      "focal_length_y": 3454.12,
      "principal_point_x": 2312.0,
      "principal_point_y": 1736.0
    }
  },
  {
    "model": "OPENCV",
    "width": 1920,
    "height": 1080,
    "parameters": {
      "focal_length_x": 1200.5,
      "focal_length_y": 1200.5,
      "principal_point_x": 960.0,
      "principal_point_y": 540.0,
      "radial_distortion_k1": -0.123,
      "radial_distortion_k2": 0.045,
      "tangential_distortion_p1": 0.001,
      "tangential_distortion_p2": -0.002
    }
  }
]
```

**Supported camera models and parameter names:**

| Model | JSON `parameters` keys (in order) | pycolmap |
|-------|-----------------------------------|----------|
| `SIMPLE_PINHOLE` | `focal_length`, `principal_point_x`, `principal_point_y` | f, cx, cy |
| `PINHOLE` | `focal_length_x`, `focal_length_y`, `principal_point_x`, `principal_point_y` | fx, fy, cx, cy |
| `SIMPLE_RADIAL` | `focal_length`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1` | f, cx, cy, k |
| `RADIAL` | `focal_length`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1`, `radial_distortion_k2` | f, cx, cy, k1, k2 |
| `OPENCV` | `focal_length_x`, `focal_length_y`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1`, `radial_distortion_k2`, `tangential_distortion_p1`, `tangential_distortion_p2` | fx, fy, cx, cy, k1, k2, p1, p2 |
| `FULL_OPENCV` | `focal_length_x`, `focal_length_y`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1`, `radial_distortion_k2`, `tangential_distortion_p1`, `tangential_distortion_p2`, `radial_distortion_k3`, `radial_distortion_k4`, `radial_distortion_k5`, `radial_distortion_k6` | fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 |
| `OPENCV_FISHEYE` | `focal_length_x`, `focal_length_y`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1`, `radial_distortion_k2`, `radial_distortion_k3`, `radial_distortion_k4` | fx, fy, cx, cy, k1, k2, k3, k4 |
| `SIMPLE_RADIAL_FISHEYE` | `focal_length`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1` | f, cx, cy, k |
| `RADIAL_FISHEYE` | `focal_length`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1`, `radial_distortion_k2` | f, cx, cy, k1, k2 |
| `THIN_PRISM_FISHEYE` | `focal_length_x`, `focal_length_y`, `principal_point_x`, `principal_point_y`, `radial_distortion_k1`, `radial_distortion_k2`, `tangential_distortion_p1`, `tangential_distortion_p2`, `radial_distortion_k3`, `radial_distortion_k4`, `thin_prism_sx1`, `thin_prism_sy1` | fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 |
| `RAD_TAN_THIN_PRISM_FISHEYE` | `focal_length_x`, `focal_length_y`, `principal_point_x`, `principal_point_y`, `radial_distortion_k0`, `radial_distortion_k1`, `radial_distortion_k2`, `radial_distortion_k3`, `radial_distortion_k4`, `radial_distortion_k5`, `tangential_distortion_p0`, `tangential_distortion_p1`, `thin_prism_s0`, `thin_prism_s1`, `thin_prism_s2`, `thin_prism_s3` | fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 |

The "pycolmap" column shows the corresponding short parameter names from COLMAP/pycolmap for reference. The parameter order in the JSON `parameters` object matches the pycolmap parameter array order. Models with a single `focal_length` use the same value for both fx and fy. All fisheye models use equidistant projection.

### 4. Rigs (Optional)

The `rigs/` section stores camera rig definitions — abstract templates describing which sensors
exist and their relative poses. This section is optional; when absent, every camera is treated as
a trivial single-sensor rig with identity `sensor_from_rig` pose (see "Implicit Rig and Frame
Values" below).

The `rigs/` and `frames/` sections must both be present or both be absent.

#### `rigs/metadata.json.zst`

```json
{
  "rig_count": 1,
  "sensor_count": 6,
  "rigs": [
    {
      "name": "cubemap_rig",
      "sensor_count": 6,
      "sensor_offset": 0,
      "ref_sensor_name": "front",
      "sensor_names": ["front", "right", "back", "left", "top", "bottom"]
    }
  ]
}
```

**Field descriptions:**
- `rig_count`: Number of rig definitions
- `sensor_count`: Total sensors across all rigs (sum of per-rig sensor counts)
- `rigs`: Array of rig definitions
  - `name`: Human-readable rig name (informational only)
  - `sensor_count`: Number of sensors in this rig
  - `sensor_offset`: Starting index into the sensor arrays for this rig's sensors
  - `ref_sensor_name`: Name of the reference sensor (must match an entry in `sensor_names`).
    The reference sensor has an identity `sensor_from_rig` pose.
  - `sensor_names`: Array of human-readable sensor names (one per sensor in this rig).
    When no meaningful names exist, use `"sensor0"`, `"sensor1"`, etc.

#### `rigs/sensor_camera_indexes.{S}.uint32.zst`

- **Shape**: `(S,)` where S = total sensor_count across all rigs
- **Data type**: `uint32` (little-endian)
- Maps each sensor to a camera intrinsics index (into `cameras/metadata.json.zst`).
  Multiple sensors can share the same camera intrinsics.

#### `rigs/sensor_quaternions_wxyz.{S}.4.float64.zst`

- **Shape**: `(S, 4)` where S = sensor_count
- **Data type**: `float64` (little-endian)
- `sensor_from_rig` rotation for each sensor as WXYZ quaternion.
  The rig coordinate frame is defined by the reference sensor, so this rotation
  expresses each sensor's orientation relative to the reference sensor.
  The reference sensor has quaternion `[1, 0, 0, 0]` (identity).

#### `rigs/sensor_translations_xyz.{S}.3.float64.zst`

- **Shape**: `(S, 3)` where S = sensor_count
- **Data type**: `float64` (little-endian)
- `sensor_from_rig` translation for each sensor, expressing each sensor's
  position relative to the reference sensor in the rig coordinate frame.
  The reference sensor has translation `[0, 0, 0]`.

### 5. Frames (Optional)

The `frames/` section stores frame instances — temporal groupings that say "these images were
captured at the same instant by this rig." This section is optional and must be present if and
only if the `rigs/` section is present.

#### `frames/metadata.json.zst`

```json
{
  "frame_count": 17
}
```

#### `frames/rig_indexes.{F}.uint32.zst`

- **Shape**: `(F,)` where F = frame_count
- **Data type**: `uint32` (little-endian)
- Maps each frame to a rig definition index (into `rigs/metadata.json.zst` rigs array)

#### `frames/image_sensor_indexes.{N}.uint32.zst`

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint32` (little-endian)
- For each image, the global sensor index that captured it.
  This determines both the camera intrinsics (via `sensor_camera_indexes`)
  and the relative pose within the rig (via `sensor_quaternions_wxyz`/`sensor_translations_xyz`).

#### `frames/image_frame_indexes.{N}.uint32.zst`

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint32` (little-endian)
- For each image, the frame index it belongs to. All images in the same frame were captured
  simultaneously by the same rig instance.

#### Partial Frames

A frame must contain at least one image but may have fewer images than the rig has sensors.
This arises naturally in practice — for example, blurry images removed during preprocessing, or
images that COLMAP/GLOMAP failed to register. The per-image mapping handles this without sentinel
values: missing sensors simply have no image entry for that frame.

#### Pose Semantics with Rigs

When rigs are present, individual image poses stored in `images/quaternions_wxyz` and
`images/translations_xyz` continue to represent the full `sensor_from_world` transformation.
The rig-from-world pose for a frame can be derived:

```
sensor_from_world = sensor_from_rig * rig_from_world
rig_from_world = inverse(sensor_from_rig) * sensor_from_world
```

This means:
- **Readers that don't understand rigs** can still use per-image poses directly (backward compatible)
- **Rig-aware readers** can recover frame poses and verify rig constraints
- The reference sensor always has identity `sensor_from_rig`, so the frame's `rig_from_world`
  equals the per-image pose of the reference sensor's image in that frame

#### Implicit Rig and Frame Values

The `rigs/` and `frames/` sections are **optional**. When absent, the implicit equivalent values
are well-defined so readers can treat every `.sfmr` file uniformly:

For a reconstruction with `C` cameras and `N` images, omitting `rigs/` and `frames/` is
equivalent to writing:

- **`rig_count`** = `C`, **`sensor_count`** = `C`
- Rig `i` (for `i` in `0..C`) has `sensor_count=1`, `sensor_offset=i`, `ref_sensor_name="sensor0"`, `sensor_names=["sensor0"]`
- `sensor_camera_indexes[i]` = `i`
- `sensor_quaternions_wxyz[i]` = `[1, 0, 0, 0]` (identity)
- `sensor_translations_xyz[i]` = `[0, 0, 0]`
- **`frame_count`** = `N`
- `rig_indexes[j]` = `camera_indexes[j]` (for `j` in `0..N`)
- `image_sensor_indexes[j]` = `camera_indexes[j]`
- `image_frame_indexes[j]` = `j`

When both sections are present, `images/camera_indexes` and `rigs/sensor_camera_indexes`
must be consistent (i.e., `camera_indexes[j]` = `sensor_camera_indexes[image_sensor_indexes[j]]`).

### 6. Images

#### `images/metadata.json.zst`

```json
{
  "image_count": 18,
  "thumbnail_size": 128
}
```

#### `images/names.json.zst`

Array of image paths **relative to workspace directory** (POSIX format):

```json
[
  "frames/DSC_0001.JPG",
  "frames/DSC_0002.JPG",
  "frames/DSC_0003.JPG"
]
```

**Note**: All image paths are relative to the workspace directory specified in `metadata.workspace.absolute_path` or `metadata.workspace.relative_path`, not relative to the `.sfmr` file location.

#### `images/camera_indexes.{N}.uint32.zst`

Binary array mapping each image to its camera:

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint32` (little-endian)
- **Format**: Index into cameras array
- **Example**: `[0, 0, 0, 1, 1, 1]` (first 3 images use camera 0, next 3 use camera 1)

#### `images/quaternions_wxyz.{N}.4.float64.zst`

Camera rotation quaternions in WXYZ format:

- **Shape**: `(N, 4)` where N = image_count
- **Data type**: `float64` (little-endian)
- **Format**: [w, x, y, z] quaternion components (unit quaternions)
- **Convention**: World-to-camera rotation (world point → camera point)

#### `images/translations_xyz.{N}.3.float64.zst`

Camera translation vectors:

- **Shape**: `(N, 3)` where N = image_count
- **Data type**: `float64` (little-endian)
- **Format**: [x, y, z] translation vector
- **Convention**: World-to-camera translation (from world coordinates to camera coordinates)

#### `images/feature_tool_hashes.{N}.uint128.zst`

XXH128 hashes identifying feature extraction tool:

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint128` (little-endian, 16 bytes per hash)
- **Format**: Hash of feature tool metadata (tool name + options)
- **Purpose**: Links to specific .sift file version

#### `images/sift_content_hashes.{N}.uint128.zst`

XXH128 hashes of feature file contents:

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint128` (little-endian, 16 bytes per hash)
- **Format**: Hash of .sift file content
- **Purpose**: Verifies feature data integrity

#### `images/thumbnails_y_x_rgb.{N}.128.128.3.uint8.zst`

Downscaled preview thumbnails for each image, embedded directly in the file:

- **Shape**: `(N, 128, 128, 3)` where N = image_count
- **Data type**: `uint8` (little-endian)
- **Format**: Row-major RGB data. For each image, 128 rows of 128 pixels, each pixel 3 bytes [R, G, B] in range [0, 255]
- **Dimension order**: `(image_index, y, x, channel)` — y is the row (top-to-bottom), x is the column (left-to-right)
- **Size**: Fixed 128×128 square, regardless of the source image aspect ratio. Source images are resized to fill the square (stretching if non-square). Consumers restore the correct aspect ratio at display time using the camera intrinsics width/height
- **Resize method**: Bilinear interpolation (triangle filter)
- **Purpose**: Enables instant thumbnail display in viewers without requiring access to the workspace source images
- **Source**: Copied from the `thumbnail_y_x_rgb.128.128.3.uint8.zst` in each image's `.sift` file during `.sfmr` creation, avoiding re-reading and re-downscaling the source images

The `thumbnail_size` field in `images/metadata.json.zst` records the dimension (currently always 128). Readers should use this value rather than hardcoding the size.

#### Depth Statistics

Per-image depth statistics and histograms computed from the 3D structure.

#### `images/depth_statistics.json.zst`

Per-image depth statistics and histogram parameters for observed points only:

```json
{
  "num_histogram_buckets": 128,
  "images": [
    {
      "histogram_min_z": 2.939,
      "histogram_max_z": 38.148,
      "observed": {
        "count": 239,
        "infinity_count": 4,
        "min_z": 2.939,
        "max_z": 38.148,
        "median_z": 5.606,
        "mean_z": 7.284
      }
    }
  ]
}
```

**Field descriptions:**
- `num_histogram_buckets`: Number of histogram buckets (fixed at 128)
- `images`: Array of per-image statistics, indexed by image order
  - `histogram_min_z`: Minimum Z depth for histogram bucket edges (`null` if no observed points)
  - `histogram_max_z`: Maximum Z depth for histogram bucket edges (`null` if no observed points)
  - `observed`: Statistics for points with actual track observations in this image
    - `count`: Number of observed finite points with positive depth (0 if none)
    - `infinity_count`: Number of observed points at infinity (`w == 0`). Omitted in version 1 files; treat a missing value as 0. Points at infinity have no finite depth, so they are excluded from `count`, the depth range, and the histogram.
    - `min_z`, `max_z`: Depth range of observed finite points (`null` if count is 0)
    - `median_z`, `mean_z`: Central tendency statistics (`null` if count is 0)

**Null handling**: When an image has no observed finite points with positive depth, all depth values are `null` and `count` is 0 (`infinity_count` may still be nonzero). The corresponding row in the histogram counts array is all zeros.

**Histogram bucket edges** can be reconstructed as:
```python
bucket_edges = np.linspace(histogram_min_z, histogram_max_z, num_histogram_buckets + 1)
```

#### `images/observed_depth_histogram_counts.{N}.128.uint32.zst`

Histogram counts for observed points (points with track observations):

- **Shape**: `(N, 128)` where N = image_count
- **Data type**: `uint32` (little-endian)
- **Format**: Row `i` contains 128 bucket counts for image `i`
- **Bucket edges**: Defined by `histogram_min_z` and `histogram_max_z` in `depth_statistics.json.zst`

### 7. Points3D

Every point — finite or at infinity — is one homogeneous coordinate
`(x, y, z, w)`:

- `w ≠ 0` — a finite point at Euclidean position `(x/w, y/w, z/w)`.
- `w = 0` — a point at infinity; `(x, y, z)` is a direction in the world frame.

`w` is the kind: the representation is self-describing, with no separate flag.
A point at infinity is the `w → 0` limit, not a special case bolted on.
A point at infinity is a feature track whose observation rays are parallel to
within feature-localisation noise — distant content (a skyline, a far building)
whose depth the SfM solve cannot pin down. Storing it as a finite `(x, y, z)`
would be lossy and misleading; the homogeneous model represents it faithfully.

#### `w` normalisation

`w` is a homogeneous coordinate, so `(x, y, z, w)` and `(λx, λy, λz, λw)` denote
the same point for any `λ ≠ 0`. The format permits any such scale; it does
**not** mandate a canonical one.

The **recommended normalised form** sets two conventions:

- finite points (`w ≠ 0`) are divided through by their own `w`, so `w = 1`;
- infinity points (`w = 0`) store a unit-length direction in `(x, y, z)`.

The first compresses well: a `w` column that is all `1`s and `0`s is a
near-constant run that zstd collapses to almost nothing. The second is the
natural canonical form for a direction. The writer in this codebase emits this
form.

Because the format does not *require* the normalised form, a consumer that
relies on `w ∈ {0, 1}` (or on unit-length directions) must normalise on read.
It cannot assume an arbitrary v2 file, possibly produced by other tooling, is
already normalised.

#### `points3d/metadata.json.zst`

```json
{
  "point_count": 2107
}
```

#### `points3d/positions_xyzw.{N}.4.float64.zst`

Homogeneous 3D point coordinates:

- **Shape**: `(N, 4)` where N = point_count
- **Data type**: `float64` (little-endian)
- **Format**: `[x, y, z, w]` in the world coordinate system. `w ≠ 0` is a finite
  point at `(x/w, y/w, z/w)`; `w = 0` is a point at infinity whose direction is
  `(x, y, z)`.
- **Constraint**: No coordinate may be `NaN` or infinite. A `w = 0` row must
  have a non-zero `(x, y, z)` direction.

#### `points3d/colors_rgb.{N}.3.uint8.zst`

RGB colors:

- **Shape**: `(N, 3)` where N = point_count
- **Data type**: `uint8` (little-endian)
- **Format**: [R, G, B] values in range [0, 255]

#### `points3d/reprojection_errors.{N}.float32.zst`

Reprojection errors:

- **Shape**: `(N,)` where N = point_count
- **Data type**: `float32` (little-endian)
- **Format**: RMS reprojection error in pixels. A `w = 0` point still projects
  (rotation + intrinsics only), so its reprojection error stays well-defined.

#### `points3d/estimated_normals_xyz.{N}.3.float32.zst`

Estimated surface normals for 3D points.

- **Shape**: `(N, 3)` where N = point_count
- **Data type**: `float32` (little-endian)
- **Format**: [x, y, z] unit normal vectors in world coordinate system. Rows for
  `w = 0` points are `(0, 0, 0)` — a direction has no surface normal.

**Computation method**: By default each finite 3D point's normal is the average
direction from the point toward all cameras that observe it (from track data),
which approximates the surface normal for points on convex surfaces.

```python
# For point P observed by cameras C1, C2, ..., Ck:
normal = normalize(sum(camera_center[i] - P for i in observers))
```

**Write-time preservation**: The writer does *not* unconditionally overwrite this
field. It **preserves** any normal already set on a point and recomputes only the
*missing* ones — the `(0, 0, 0)` rows left for points whose normal was never
estimated (and for `w = 0` points, which stay `(0, 0, 0)`). So a freshly built
reconstruction (normals start all-zero) gets a full mean-viewing set on its first
write, while a better normal a consumer has computed — e.g. the photometric
`sfm xform --refine-normals` (`specs/cli/xform-refine-normals-command.md`) —
survives subsequent saves instead of being clobbered. Depth statistics and
histograms are always recomputed regardless. (Set `skip_recompute_depth_stats` to
write every depth-derived field, normals included, exactly as supplied.)

**Use cases**:
- Visibility testing (front-facing check)
- Frustum-based covisibility estimation
- Surface orientation analysis

### 8. Tracks

Tracks link 2D feature observations to 3D points. Each observation has three components stored in separate columnar files.

#### `tracks/metadata.json.zst`

```json
{
  "observation_count": 9427
}
```

#### `tracks/image_indexes.{M}.uint32.zst`

Image index for each observation:

- **Shape**: `(M,)` where M = observation_count
- **Data type**: `uint32` (little-endian)
- **Format**: Index into images arrays for each observation
- **Constraint**: MUST be sorted lexicographically by `(point_indexes[i], image_indexes[i])`

#### `tracks/feature_indexes.{M}.uint32.zst`

Feature index for each observation:

- **Shape**: `(M,)` where M = observation_count
- **Data type**: `uint32` (little-endian)
- **Format**: Index into feature file for each observation
- **Constraint**: MUST be sorted lexicographically by `(point_indexes[i], image_indexes[i])`

#### `tracks/point_indexes.{M}.uint32.zst`

Point index for each observation:

- **Shape**: `(M,)` where M = observation_count
- **Data type**: `uint32` (little-endian)
- **Format**: Index into the points3d arrays for each observation. A track is a
  track regardless of the referenced point's `w`; the finite/infinity
  distinction is a property of the point, never of the observation.
- **Constraint**: MUST be sorted lexicographically by `(point_indexes[i], image_indexes[i])`

#### `tracks/observation_counts.{N}.uint32.zst`

Observations per point:

- **Shape**: `(N,)` where N = point_count
- **Data type**: `uint32` (little-endian)
- **Format**: Number of observations for each 3D point
- **Constraints**:
  - Sum must equal observation_count
  - Every value must be >= 1 (every 3D point must have at least one observation)

## Data Ordering and Constraints

### Critical Ordering Requirements

1. **Tracks must be sorted lexicographically**:
   - All three track arrays (`image_indexes`, `feature_indexes`, `point_indexes`) MUST be sorted lexicographically by `(point_indexes[i], image_indexes[i])`
   - Primary sort: by point index (all observations of point 0, then point 1, etc.)
   - Secondary sort: by image index within each point's observations
   - This provides deterministic ordering and enables efficient extraction of observations per point
   - Example: `[(0,5), (0,12), (0,15), (1,3), (1,8), (2,1), ...]` where tuples are `(point_index, image_index)`

2. **Observation counts must align**:
   - `observation_counts[i]` = number of entries in track arrays where `point_indexes == i`
   - `sum(observation_counts)` must equal `observation_count`

### Index Relationships

- `camera_indexes[i]` → index into `cameras` array
- `image_indexes[j]` → index into `images` arrays
- `feature_indexes[j]` → index into feature file for `images[image_indexes[j]]`
- `point_indexes[j]` → index into `points3d` arrays

## Compression Details

All `.zst` files use zstandard compression:

- **Compression level**: Default (level 3, configurable)
- **Strategy**: Optimize for compression ratio vs speed based on file size
- **JSON files**: Compact encoding (no pretty-printing)
- **Binary files**: Direct compression of raw byte stream

## File Naming Convention

Binary files use self-documenting names:

**Format**: `{field_name}.{dim1}.{dim2}...{dtype}.zst`

**Examples**:
- `positions_xyzw.2107.4.float64.zst` → 2107 points, 4 homogeneous coordinates, float64
- `camera_indexes.18.uint32.zst` → 18 images, uint32 indices
- `quaternions_wxyz.18.4.float64.zst` → 18 images, 4 components, float64

**Supported dtypes**:
- `uint8`, `uint16`, `uint32`, `uint64`
- `int8`, `int16`, `int32`, `int64`
- `float32`, `float64`
- `uint128` (for XXH128 hashes, 16 bytes)

## Integrity Verification

### Hash Computation

1. **Metadata hash**: XXH128 of the uncompressed `metadata.json.zst` content bytes
2. **Cameras hash**: XXH128 of the uncompressed `cameras/metadata.json.zst` content bytes
3. **Section hashes** (images, points3d, tracks): For each section, feed all files' uncompressed content bytes into a streaming XXH128 hasher in lexicographic path order. The final digest is the section hash.
4. **Overall hash**: Concatenate all present section hashes as raw 16-byte big-endian digests (metadata, cameras, rigs if present, frames if present, images, points3d, tracks), then compute XXH128 of the concatenation. Each 128-bit hash digest is serialized as 16 bytes in big-endian (most significant byte first) order before concatenation.

### Verification Process

1. Decompress each file and hash the raw uncompressed bytes (do NOT re-serialize JSON)
2. Recompute section and overall hashes as described above
3. Compare with stored values in `content_hash.json.zst`
4. If any hash mismatches, file is corrupted

## Usage Examples

### Reading a .sfmr file

```python
from sfmtool import SfmrFileReader

with SfmrFileReader("reconstruction.sfmr") as sfm:
    # Read metadata
    metadata = sfm.metadata
    print(f"Reconstruction with {metadata['image_count']} images")

    # Load cameras
    cameras = sfm.read_cameras()

    # Load images
    image_names = sfm.read_image_names()
    camera_indexes, quaternions, translations = sfm.read_image_data()

    # Load 3D points (positions are homogeneous: (N, 4) [x, y, z, w])
    positions_xyzw, colors_rgb, reprojection_errors = sfm.read_points3d()

    # Load tracks
    image_indexes, feature_indexes, point_indexes, obs_counts = sfm.read_tracks()
```

### Writing a .sfmr file

```python
from sfmtool import write_sfm

write_sfm(
    output_path="reconstruction.sfmr",
    cameras=cameras,
    images={
        "names": image_names,
        "camera_indexes": camera_indexes,
        "quaternions_wxyz": quaternions,
        "translations_xyz": translations,
        "feature_tool_hashes": feature_hashes,
        "sift_content_hashes": content_hashes,
    },
    points3d={
        "positions_xyzw": positions_xyzw,
        "colors_rgb": colors_rgb,
        "reprojection_errors": reprojection_errors,
    },
    tracks={
        "image_indexes": image_indexes,
        "feature_indexes": feature_indexes,
        "point_indexes": point_indexes,
        "observation_counts": observation_counts,
    },
    metadata={
        "operation": "sfm_solve",
        "tool": "colmap",
        "tool_version": "3.10",
        "tool_options": {...},
        "workspace": {
            "absolute_path": "/path/to/workspace",
            "relative_path": "../workspace",
            "contents": {
                "feature_tool": "colmap",
                "feature_type": "sift",
                "feature_options": {...},
                "feature_prefix_dir": "features/sift-colmap-d1245b460906df27ee4730273e0aba41"
            }
        },
        # ... other metadata
    }
)
```

### Verifying integrity

```python
from sfmtool import verify_sfm

is_valid, errors = verify_sfm("reconstruction.sfmr")
if is_valid:
    print("✓ File integrity verified")
else:
    print("✗ Verification failed:")
    for error in errors:
        print(f"  - {error}")
```

## Comparison with Directory Format

### Directory Format
```
20251220T204131-00/
├── metadata.json.zst
├── cameras/
│   └── metadata.json.zst
├── images/
│   ├── names.json.zst
│   ├── camera_indexes.18.uint32.zst
│   └── ...
├── points3d/
│   └── ...
└── tracks/
    └── ...
```

### .sfmr file Format
```
reconstruction.sfmr (single ZIP file)
├── metadata.json.zst
├── content_hash.json.zst
├── cameras/
│   └── metadata.json.zst
├── rigs/                        (optional)
│   ├── metadata.json.zst
│   └── ...
├── frames/                      (optional)
│   ├── metadata.json.zst
│   └── ...
├── images/
│   ├── names.json.zst
│   ├── camera_indexes.18.uint32.zst
│   └── ...
├── points3d/
│   └── ...
└── tracks/
    └── ...
```

### Key Differences

1. **Single file vs directory**: .sfmr is portable, directory is not
2. **Content hashing**: .sfmr adds `content_hash.json.zst` for integrity
3. **Random access**: .sfmr uses ZIP STORE for efficient partial loading
4. **File distribution**: Single .sfmr file is easier to share and archive

## Design Rationale

### Why ZIP with STORE?

- Enables random access to individual files without full decompression
- Standard format with wide tooling support
- Self-contained single file for easy distribution

### Why Zstandard compression?

- Better compression ratio than gzip (typically 15-30% smaller)
- Faster decompression speed
- Configurable compression levels
- Industry standard (used by Facebook, Linux kernel, etc.)

### Why columnar storage?

- Load only needed data (e.g., just positions, not colors)
- Better compression (similar data together)
- Easy to extend with new fields
- Efficient for large reconstructions

### Why XXH128 hashes?

- Extremely fast (GB/s throughput)
- 128-bit provides collision resistance
- Non-cryptographic (speed over security)
- Used consistently with .sift format

## Point ID: Portable 3D Point References

A **Point ID** is a compact, copy-pastable string that uniquely identifies a 3D
point across `.sfmr` files and sessions. Raw point indices (e.g., `#12345`) are
meaningless without knowing which reconstruction they came from. A Point ID
bundles a reconstruction fingerprint with the index, making the reference
self-contained.

The intended workflow: a user clicks to select a 3D point in the GUI viewer,
copies its Point ID from the panel header, and pastes it into a file or CLI
command alongside other Point IDs and annotations. For example, a ground-truth
constraints file might list pairs of points with known real-world distances
between them:

```
# Scale constraints for reconstruction alignment
# point_a  point_b  distance_m  description
pt3d_a1b2c3d4_4821  pt3d_a1b2c3d4_9107  0.38   "ruler endpoints"
pt3d_a1b2c3d4_331   pt3d_a1b2c3d4_12040  0.305  "tile edge"
```

Because each ID embeds a reconstruction hash, a tool processing this file can
verify that all points come from the expected reconstruction and report a clear
error if the `.sfmr` file has changed since the constraints were written. The
double-click-selectable format makes it easy to pick out individual IDs when
editing these files in a text editor or terminal.

### Format

```
pt3d_{hash}_{index}
```

| Part | Content | Example |
|------|---------|---------|
| `pt3d` | Literal prefix | `pt3d` |
| `{hash}` | First 8 hex characters of `content_xxh128` | `a1b2c3d4` |
| `{index}` | Decimal point index | `12345` |

**Example**: `pt3d_a1b2c3d4_12345`

### Why `content_xxh128`

The `content_xxh128` is the overall file integrity hash, computed from all
section hashes (metadata, cameras, images, points, tracks, etc.). It uniquely
identifies a specific `.sfmr` file.

`.sfmr` files are written once — each SfM solve, filter, or transform produces
a new file with its own content hash. This means each file gets a unique
`content_xxh128`, and a Point ID unambiguously identifies both the file and the
point within it.

### Hash Prefix Length

8 hex characters = 32 bits of hash entropy. Among a user's collection of
`.sfmr` files (realistically <1000), the birthday-problem collision probability
is approximately `n^2 / (2 * 2^32)` ≈ 1 in 10 million for n=1000. This keeps
the ID compact while being practically unique.

If programmatic disambiguation is needed, the full 32-character
`content_xxh128` is available in `content_hash.json.zst` for exact matching.

### Double-Click Selectability

The ID uses only characters in the `[a-zA-Z0-9_]` word-character class.
Terminals (Windows Terminal, iTerm2, gnome-terminal), web browsers, and most
text editors treat this class as a single "word" for double-click selection. The
entire ID is selected with one double-click in any common environment.

### Resolving a Point ID to a `.sfmr` File

Given a Point ID and a workspace directory, a program can locate the source
`.sfmr` file by matching the hash prefix against `content_xxh128` values in
available `.sfmr` files. Reading the hash requires only decompressing
`content_hash.json.zst` — a small JSON entry near the start of the ZIP archive
— not the full reconstruction data.

**Search strategy** to find `.sfmr` whose content hash has matching prefix:

1. **`sfmr/` subdirectory** — The conventional location for reconstruction
   files within a workspace. Scan all `.sfmr` files in this directory first.
   This covers the common case with minimal I/O.

2. **Workspace root** — Check any `.sfmr` files directly in the workspace
   directory. Some workflows place reconstructions alongside images.

3. **Breadth-first directory traversal** — Walk the remaining workspace tree
   breadth-first, skipping hidden directories and common non-project directories.

## World-Space Unit

The top-level `metadata.json.zst` may include an optional `world_space_unit` field that declares
the physical unit of 3D coordinates (point positions and camera translations) in the reconstruction.

```json
{
  "operation": "transform",
  "tool": "sfmtool",
  ...
  "world_space_unit": "mm"
}
```

**Field definition:**

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `world_space_unit` | No | string | Physical unit of 3D world-space coordinates. One of: `"mm"`, `"cm"`, `"m"`, `"in"`, `"ft"`. |

When absent, the reconstruction is in arbitrary (unscaled) units — the default state after an SfM
solve.

**Semantics:**

- All 3D point positions (`points3d/positions_xyzw`) are in this unit. The unit
  applies to a finite point's Euclidean position `(x/w, y/w, z/w)`; a `w = 0`
  point is a direction and is unitless.
- All camera translations (`images/translations_xyz`) are in this unit.
- Camera intrinsics (focal length, principal point) remain in pixel units and are unaffected.
- Rotations (quaternions) are dimensionless and are unaffected.

**Propagation:** When a reconstruction with `world_space_unit` set is transformed by operations
that preserve scale (rotation, translation, filtering, bundle adjustment), the output should
preserve `world_space_unit`. Operations that change scale (`--scale`) should clear the field
unless the caller explicitly sets it. The `--scale-by-measurements` transform both scales the
reconstruction and sets the field.

**Display:** The GUI and CLI inspection tools can use this field to display coordinates with
appropriate units (e.g., "xyz: (1234.0, -567.0, 2891.0) mm" instead of bare numbers).

## Future Extensions

Potential additions while maintaining backward compatibility:

1. **Optional fields**:
   - Covariance matrices: `points3d/covariances.{N}.3.3.float64.zst`

2. **Metadata extensions**:
   - GPS coordinates per image
   - Depth maps or dense reconstruction references
   - Semantic labels for points

3. **Multi-reconstruction support**:
   - Store multiple disconnected components
   - Hierarchical reconstructions (coarse-to-fine)

All extensions should:
- Be optional (readers can skip unknown fields)
- Follow naming conventions
- Include metadata describing new fields
- Update content hashes appropriately

## Versioning and Migration

The `version` field in `metadata.json.zst` is `2` for this specification.

Version 2 replaces the version 1 point representation with the unified
homogeneous model described in [Points3D](#7-points3d). The differences:

| Version 1 | Version 2 |
|-----------|-----------|
| `points3d/positions_xyz.{N}.3.float64.zst` — Euclidean `(x, y, z)` | `points3d/positions_xyzw.{N}.4.float64.zst` — homogeneous `(x, y, z, w)` |
| `tracks/points3d_indexes.{M}.uint32.zst` | `tracks/point_indexes.{M}.uint32.zst` |
| metadata `points3d_count` | metadata `point_count` |
| (no infinity points) | metadata `infinity_point_count` |
| `points3d/metadata.json.zst` key `points3d_count` | `points3d/metadata.json.zst` key `point_count` |

The `points3d/` archive directory and the `points3d_xxh128` content-hash field
keep their version 1 names in version 2.

Version 2 changes the meaning of the positions array and renames the track
index array, so it is **not** backward compatible — hence the version bump.
A version 1 file is **not** readable by version-2-only tooling, and a version 2
file is **not** readable by pre-version-2 tooling.

**Migration is mechanical and lossless.** A version 1 file upgrades to the
version 2 model by appending a `w = 1` column to every point position (every
version 1 point is finite) and renaming `points3d_indexes` to `point_indexes`;
it carries no points at infinity, so `infinity_point_count` is `0`.

The reader in this codebase accepts **both version 1 and version 2**, upgrading
version 1 on read so the rest of the codebase only ever sees the unified model.
The writer emits **version 2 only**.

## Version History

- **Version 2**: Unified homogeneous point model — points at infinity
  (`w = 0`) are first-class. Reader accepts version 1 and version 2.
- **Version 1.0rc1**: Release candidate
