# `sfm inspect` Command

## Overview

Inspects a single sfmtool file or image and prints a summary. The file type is
determined by extension; one summary printer handles each type. Without
`--verbose`, prints a compact label/value block; with `--verbose`, prints the
full detail available for that type.

For deep-analysis reports on a `.sfmr` reconstruction (covisibility graph,
frustum intersection, depth ranges, per-image metrics), use `sfm analyze`
(see `specs/cli/analyze-command.md`).

## Command Syntax

```bash
sfm inspect <PATH> [--verbose / -v]
sfm inspect <POINT_ID> [WORKSPACE] [--verbose / -v]
sfm inspect --strips <FILE.sfmr> [POINT ...] [-o OUT.png] [--strips-views N] [--context FRACTION]
```

`PATH` is a single file. Directories and multiple paths are not accepted.

`POINT_ID` is a 3D point reference of the form `pt3d_<hash>_<index>` (as shown
by the GUI and by verbose reconstruction reports — see the [Point ID section in
the sfmr format spec](../formats/sfmr-file-format.md#point-id-portable-3d-point-references)).
The optional second argument `WORKSPACE` is a directory used to locate the
source `.sfmr`; it (or its nearest ancestor containing `.sfm-workspace.json`) is
the workspace searched. `WORKSPACE` defaults to the current directory, and is
only valid with a point ID (an error with a file `PATH`).

## Supported File Types

| Extension | Type | Inspected with |
|-----------|------|----------------|
| `.sfmr` | Reconstruction | `read_sfmr_metadata` / `verify_sfmr` (default), full load (verbose) |
| `.sift` | Feature file | `read_sift_metadata` / `verify_sift` |
| `.matches` | Feature matches | `read_matches_metadata` / `verify_matches` (default), `read_matches` (verbose) |
| `.camrig` | Camera rig | `read_camrig_metadata` / `verify_camrig` (default), `read_camrig` (verbose) |
| `.png` `.jpg` `.jpeg` | Image | Image dimensions; pycolmap EXIF inference (verbose) |
| `pt3d_<hash>_<index>` | 3D point | `read_sfmr_content_hash` to resolve the `.sfmr`, then `SfmrReconstruction.load` / `inspect_point` (verbose) |

An unsupported extension is rejected with a message listing the supported
types.

## 3D Point IDs

`sfm inspect pt3d_<hash>_<index>` resolves a point reference to its 3D point.
The `.sfmr` file is located by matching `<hash>` (the first 8 hex chars of the
file's `content_xxh128`) against the content hashes of the `.sfmr` files in the
workspace — searched in the order the format spec prescribes: the `sfmr/`
subdirectory first, then the workspace root, then the rest of the tree (hidden
directories skipped). Each candidate's hash is read by decompressing only
`content_hash.json.zst` (`read_sfmr_content_hash`), not the full reconstruction.
A missing match, or an index beyond the file's point count, is a clear error.

- **Default summary** — point ID, source file, finite (`w = 1`) vs at-infinity
  (`w = 0`), position/direction, color, reprojection error, observation count.
  Uses only the loaded reconstruction (no `.sift` needed).
- **Verbose** — the full triangulation analysis from `inspect_point`, which
  re-derives the point's observation rays from the workspace `.sift` files (so
  they must be present): the re-derived classification, triangulated point and
  depth, condition number and eigenvalues, in-front flag, inverse-depth z-score
  (and σ), `resolvable_distance` vs `finite_horizon` (the camera extents) with a
  sufficient/insufficient verdict, observing-camera baseline span, ray spread,
  and a per-observation list with each ray's incidence angle off the optical
  axis (flagging the near-fisheye-edge observations).

## Point Strips (`--strips`)

`sfm inspect --strips <FILE.sfmr> [POINT ...]` renders a chosen set of 3D points
as a patch-strip montage for visually evaluating point quality. `TARGET` is the
`.sfmr`; every remaining positional argument is a **point spec**, and the points
are rendered as montage rows **in the order listed**. Each row is laid out, from
the left:

1. a **label panel** — the point's `pt3d_<hash>_<index>` id, finite (`w=1`) vs
   `infinity` (`w=0`), the mean pairwise NCC (`NCC`) of its observation patches,
   the triangulation angle (`a`, finite points only), and `shown/total` views;
2. the **reference patch** — the stored per-point patch bitmap when the recon
   carries one (the un-blended RGB patch, with the bitmap's alpha channel shown
   as a grayscale tile beside it), otherwise the cross-view consensus (the mean
   of the point's per-view core patches at the surfel orientation);
3. the **observation strip** — one tile per observing view, the point's oriented
   surfel projected into that view, padded with surrounding context and a border
   at the patch extent (see `--context`). Each tile is labeled with the image
   index (top) and, at the bottom, that observation's per-view NCC against the
   other views (`n`) and its reprojection error in pixels (`e`).

### Point specs

Each spec is either:

- a **point id** `pt3d_<hash>_<index>` — its `<hash>` must match this
  reconstruction's content hash (the first 8 hex chars of `content_xxh128`);
  a mismatch is an error (the point id belongs to a different `.sfmr`); or
- a **point-index range expression** — `5`, `5-12`, `1,4,7`, … (the same
  `RangeExpr` grammar used by `--range` elsewhere), naming point indexes directly.

Indexes are kept in listed order and de-duplicated. An index beyond the
reconstruction's point count is an error.

### Feature source

- **`sift_files`** — first converted to `embedded_patches` (the minimal
  `--to-embedded-patches` bridge: mean-viewing patch frames, keypoints copied
  from `.sift`), then a light normal refinement runs over **only the listed
  finite points** (rendering their patch bitmaps, used as the reference patch).
  The default keypoint-preserving conversion makes these good. Reads the
  workspace `.sift` files and source images.
- **`embedded_patches`** — rendered as stored (no conversion or refinement); the
  stored patch frames and bitmaps are used directly.

See `specs/core/sift-to-patch-reconstruction.md` for the conversion.

### Points at infinity

A listed point at infinity (`w=0`) is rendered via its tangent-sphere infinity
patch (`OrientedPatch.from_infinity_direction`); its reference patch is the
cross-view consensus (the refiner does not adapt infinity normals, so no bitmap
is produced).

### Options

- `-o` / `--output` — output PNG path (default: `<stem>_strips.png` in the
  current directory).
- `--strips-views N` — cap the observation tiles (views) per point (`0` = all;
  default 8). When capped, an evenly-spaced representative subset is shown and
  the label reports `shown/total`.
- `--context FRACTION` — pad each per-observation tile with `FRACTION` of extra
  field of view around the patch (default `1.0`, i.e. +100%), drawing a green
  border at the patch extent; NCC is still scored on the inner patch only. `0`
  renders tight, borderless tiles. The reference patch always renders tight (no
  context), sized to match the boxed patch region in the observation tiles.

`-o` / `--strips-views` / `--context` are rejected unless `--strips` is given.

## Integrity

For the four sfmtool formats, the matching `verify_*` function runs and the
result appears as an `Integrity:` line (`OK` or `FAILED`). When verification
fails, the error messages are printed and the command exits non-zero. Image
files have no integrity check.

## Default Summary

The default output is a compact label/value block. The fields per type:

- **`.sfmr`** — format version, operation (tool + version), image / camera /
  3D point / observation counts, rig counts (if present), integrity. When the
  reconstruction holds any points at infinity (`w == 0`), the 3D point count is
  annotated with how many are at infinity (e.g. `206,413  (105,773 at
  infinity)`), read from the format's stored `infinity_point_count`.
- **`.sift`** — image name and dimensions, feature count, feature tool,
  integrity.
- **`.matches`** — format version, matching method (tool + version), image /
  image-pair / match counts, whether two-view geometries are present,
  integrity.
- **`.camrig`** — format version, name, rig type, sensor / camera counts,
  integrity.
- **image** — format, dimensions, file size.

## Verbose Output (`--verbose` / `-v`)

- **`.sfmr`** — the full reconstruction report: metadata (including the
  operation's recorded **tool options** — the ordered `transforms` list for
  `xform`, solver flags for `solve`, etc.), workspace, per-camera parameter
  tables, rig configuration, 3D point statistics with histograms, reprojection
  error, per-point depth-reliability diagnostics (inverse-depth z-score and
  condition number), observation statistics, nearest-neighbor distances. The
  3D point count carries the same `(N at infinity)` annotation as the default
  summary when any points at infinity are present.
- **`.sift`** — adds image file size and hashes, feature tool and content
  hashes, feature tool options, and the top 5 features by size.
- **`.matches`** — adds timestamp, workspace, matching options, matches-per-pair
  and descriptor-distance histograms, and two-view-geometry inlier statistics.
- **`.camrig`** — adds `rig_attributes`, content hash, and per-camera details.
- **image** — adds the pycolmap EXIF-inferred camera (model, dimensions,
  focal length).

## Usage Examples

```bash
# Compact summary of a reconstruction
sfm inspect sfmr/solve_001.sfmr

# Full reconstruction report
sfm inspect sfmr/solve_001.sfmr --verbose

# Inspect a feature file
sfm inspect image_001.sift -v

# Inspect a matches file
sfm inspect matches/exhaustive.matches

# Inspect a camera rig
sfm inspect tiles.camrig

# Inspect an image
sfm inspect photo.jpg -v

# Render chosen points as a patch-strip montage
sfm inspect --strips sfmr/solve_001.sfmr 0-9 pt3d_220747a8_96414 -o strips.png
```
