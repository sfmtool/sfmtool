# COLMAP Interop Crate

The `sfmtool-colmap` crate reads and writes the two file formats of
[COLMAP](https://colmap.github.io/), the Structure-from-Motion program that
sfmtool's solvers run on: the binary reconstruction model (`cameras.bin`,
`images.bin`, `points3D.bin`, `rigs.bin`, `frames.bin` in one directory) and the
SQLite database that holds cameras, images, features, matches and two-view
geometries before a solve. It exists so that sfmtool can hand its data to COLMAP
and take COLMAP's results back without depending on COLMAP's own I/O code. The
crate translates between COLMAP's IDs and sfmtool's 0-based indexes and between
COLMAP's positional camera parameters and sfmtool's named ones. It does **not**
convert poses or points between COLMAP's camera convention and sfmtool's
canonical one, with the single exception of two-view relative poses exchanged
through a `.matches` structure (see [Coordinate conventions](#coordinate-conventions)).

Unlike the other specs in this directory, this one does not define a format.
COLMAP defines both formats; this spec describes how the crate maps them onto
sfmtool's data.

## Rust API

The crate has two public modules, and each re-exports its items at the module
root.

[`colmap_io`](../../crates/sfmtool-colmap/src/colmap_io/mod.rs), the binary
model:

| Item | What it does |
|------|--------------|
| `read_colmap_binary(dir) -> ColmapReconstruction` | Reads the three required files, plus `rigs.bin` and `frames.bin` when present. |
| `write_colmap_binary(dir, &ColmapWriteData)` | Writes all five files, creating `dir` if needed. |
| `write_rigs_bin`, `write_frames_bin` | Write one of the two rig files on its own. |
| `colmap_model_id`, `camera_params_to_array`, `claim_native_camera_model`, `EQUIDISTANT_FISHEYE`, `EQUIDISTANT_FISHEYE_CARRIER` | Camera-model translation, shared with the database writer. |
| `ColmapRig`, `ColmapFrame`, `ColmapSensor`, `ColmapRigSensor`, `ColmapDataId`, `ColmapSensorType`, `Keypoint2D` | The records the reader returns and the writer takes. |
| `ColmapIoError` | Every failure: I/O, unknown model ID or name, truncated file, inconsistent data. |

[`colmap_db`](../../crates/sfmtool-colmap/src/colmap_db/mod.rs), the SQLite
database:

| Item | What it does |
|------|--------------|
| `write_colmap_db(path, &ColmapDbWriteData) -> Vec<i64>` | Creates a database with cameras, images, keypoints, descriptors, and optionally pose priors, two-view geometries, rigs and frames. Returns the database image ID of each input image. |
| `write_colmap_db_features(path, &ColmapDbFeatureData) -> ImageIdMap` | The same without two-view geometries, returning the index-to-ID map that `write_colmap_db_matches` needs. |
| `write_colmap_db_matches(path, &MatchesData, &ImageIdMap)` | Adds the `matches` and `two_view_geometries` tables to a database made by `write_colmap_db_features`. |
| `read_colmap_db_matches(path, include_tvg) -> MatchesData` | Reads the `matches` table, and the `two_view_geometries` table when asked, into a `.matches` structure. |
| `PosePrior`, `TwoViewGeometry`, `TwoViewGeometryConfig`, `DbRig`, `DbRigSensor`, `DbSensor`, `DbSensorType`, `DbFrame`, `DbFrameDataId` | The rows these functions write. |
| `ColmapDbError` | Every failure: SQLite, I/O, unknown camera model, invalid pair ID, inconsistent data (including a missing camera parameter). |

The input structs (`ColmapWriteData`, `ColmapDbWriteData`,
`ColmapDbFeatureData`) borrow parallel slices rather than taking a
reconstruction type, which keeps the crate independent of `sfmtool-core`: it
depends only on the `.sfmr` and `.matches` format crates.

### Callers

The only caller in the workspace is the Python binding layer,
[`sfmtool-py/src/fileio/colmap_binary.rs`](../../crates/sfmtool-py/src/fileio/colmap_binary.rs)
and [`sfmtool-py/src/fileio/colmap_db.rs`](../../crates/sfmtool-py/src/fileio/colmap_db.rs),
which expose `read_colmap_binary`, `write_colmap_binary`, `write_colmap_db` and
`read_colmap_db_matches` on `sfmtool.fileio`. From Python these back
[`sfm from-colmap-bin`](../cli/colmap-interop/from-colmap-bin-command.md),
[`sfm to-colmap-bin`](../cli/colmap-interop/to-colmap-bin-command.md),
[`sfm to-colmap-db`](../cli/colmap-interop/to-colmap-db-command.md) and
[`sfm to-nerfstudio`](../cli/colmap-interop/to-nerfstudio-command.md), the
incremental solve (which reads COLMAP's binary output), `sfm match` (which reads
COLMAP's matches back out of the database), and the bundle-adjust and densify
steps (which write a binary model for COLMAP to work on).
`write_colmap_db_features` and `write_colmap_db_matches` are not bound to
Python and are called only from the crate's tests.

## IDs and ordering

COLMAP numbers cameras, images and 3D points with IDs that start at 1 and need
not be contiguous. sfmtool uses 0-based indexes into arrays.

- **Reading a binary model.** Cameras are indexed in file order. Images are
  sorted by name and indexed in that order, whatever their COLMAP IDs. 3D points
  are indexed in file order, and each keypoint's point reference and each track
  entry are remapped to those indexes. A track entry that names an image not in
  `images.bin` is dropped. Rig sensor IDs and frame data IDs are remapped to
  camera and image indexes the same way.
- **Writing a binary model.** Camera, image and point IDs are the index plus 1;
  feature indexes inside a track stay 0-based, as COLMAP stores them. Rigs and
  frames passed in are written as given, so their sensor and data IDs must
  already be COLMAP camera and image IDs (the Python binding adds 1); the
  implicit rigs and frames the writer creates itself use the camera and image
  IDs above.
- **The database.** Camera and image IDs are assigned by SQLite in input order
  (so they are also index plus 1 in a fresh database). An image pair is stored
  under COLMAP's pair ID, `(2³¹ − 1) · smaller_id + larger_id`; the writer
  refuses image IDs that are not positive or are not below `2³¹ − 1`. The reader
  sorts images by name, decodes each pair ID, and returns pairs sorted by
  `(index_i, index_j)` with `index_i < index_j`, swapping the two feature
  columns when it has to swap the images.

## Camera models

The crate translates the COLMAP models with IDs 0–6 and 8–11 (`SIMPLE_PINHOLE`,
`PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `OPENCV_FISHEYE`,
`FULL_OPENCV`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`, `THIN_PRISM_FISHEYE`,
`RAD_TAN_THIN_PRISM_FISHEYE`), mapping each positional parameter list to the
named parameters of an `SfmrCamera`. A camera with any other model ID, such as
COLMAP's `FOV` (ID 7), is refused on read with `UnknownModelId`.

One sfmtool model with no COLMAP counterpart crosses the boundary:
`EQUIDISTANT_FISHEYE` is written as `SIMPLE_RADIAL_FISHEYE` with
`radial_distortion_k1 = 0`, and on read a `SIMPLE_RADIAL_FISHEYE` whose `k1` is
exactly `0.0` becomes `EQUIDISTANT_FISHEYE` again. Every other model name that is
not in the COLMAP table, including the
[`SFMTOOL_PINHOLE` and `SFMTOOL_FISHEYE`](sfmtool-camera-models.md) models, is
refused on write with `UnknownModelName`.

## Coordinate conventions

COLMAP cameras look down +Z with +Y down; a `.sfmr` file stores the canonical
convention defined in
[sfmr-file-format.md § Coordinate System Conventions](sfmr-file-format.md#coordinate-system-conventions),
which requires that conversion happen at the I/O boundary.

The binary reader and writer and `write_colmap_db` copy poses, rig sensor poses
and point positions verbatim, in COLMAP's convention. The conversion is done by
their Python callers, in
[`src/sfmtool/colmap/io.py`](../../src/sfmtool/colmap/io.py) and
[`src/sfmtool/colmap/db_export.py`](../../src/sfmtool/colmap/db_export.py), via
the helpers in [`src/sfmtool/colmap/convention.py`](../../src/sfmtool/colmap/convention.py).
They own it because the same binary files are used for more than one purpose:
an external import or export applies both the camera-frame flip and the world
rotation, while a pipeline step that writes a model for COLMAP and reads the
result back within one operation applies only the camera-frame flip.
`sfm to-nerfstudio` also applies only the camera-frame flip to the COLMAP model
it writes, so that the model shares the world frame of its `transforms.json`.

The exception is the `.matches` path. A `MatchesData` structure always holds
canonical relative poses, so `write_colmap_db_matches` and
`read_colmap_db_matches` conjugate each two-view relative pose by
`S = diag(1, −1, −1)` on the way out and on the way in. Fundamental, essential
and homography matrices relate pixels, which do not change between conventions,
and cross unchanged. Keypoint coordinates are also copied unchanged.

## What each direction stores and drops

- **Binary write** checks that the per-image, per-point and per-observation
  slices have matching lengths, and refuses the call otherwise. When the caller
  passes neither rigs nor frames, it writes the implicit rigs and frames that
  the `.sfmr` format defines (one single-sensor rig per camera, one frame per
  image, see
  [sfmr-file-format.md § Implicit Rig and Frame Values](sfmr-file-format.md#implicit-rig-and-frame-values)),
  so `rigs.bin` and `frames.bin` are always present.
- **Binary read** keeps camera sensors only: IMU sensors are dropped from rigs
  and from frame data. `rigs` and `frames` are `None` when their file is absent.
- **Database write** replaces any existing file. Keypoints are stored as two
  `f32` columns (x, y); descriptors must have `keypoints × descriptor_dim`
  bytes per image or the call is refused. `prior_focal_length` is always 0.
  COLMAP's `images` table has no pose columns, so the per-image poses in
  `ColmapDbWriteData` are not stored. Frames are written only when rigs are.
  `write_colmap_db_matches` refuses a `MatchesData` without an image-pairs
  section (a cluster-only `.matches` must be expanded to pairs first), and
  stores an all-zero F, E or H matrix, an identity rotation and a zero
  translation as SQL `NULL`.
- **Database read** skips pairs with no match rows and pairs that name an image
  missing from the `images` table, and refuses a match blob whose length does
  not equal `rows × 8` bytes. COLMAP does not store descriptor distances, SIFT
  hashes, image sizes or workspace metadata, so the returned `MatchesData` holds
  zero distances, zero hashes, no image sizes and placeholder metadata (mostly
  empty strings; `matching_method` is `"unknown"` and `matching_tool` is
  `"colmap"`); the caller fills these in from the `.sift` files before writing
  a `.matches` file.
