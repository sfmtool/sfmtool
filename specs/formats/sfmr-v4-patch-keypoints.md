# Draft: `.sfmr` v4 — patch-derived keypoints as a first-class observation

_Status: draft for review. Intended to be folded into `sfmr-file-format.md` once
implemented._

## Motivation

Today every observation in `.sfmr` is a reference into an external `.sift` file:
the per-image 2D coordinate lives in the `.sift` and `.sfmr` only carries the
`(image_index, feature_index, point_index)` triple. This ties every
reconstruction to a fixed set of SIFT detections.

A patch-refinement pipeline can derive sub-pixel per-view keypoints from
oriented patches — observations at locations where SIFT has no detection. To
store those, `.sfmr` must carry the 2D coordinate itself rather than reference a
`.sift` feature. This document proposes the minimum format extension to do that.

## Design summary

A v4 `.sfmr` declares, at the **file level**, which kind of observation it
carries — `feature_source ∈ {"sift_files", "embedded_patches"}`. There is **no mixing**
within a file: a file is wholly one or the other. The two modes differ only in
the `tracks/` and `images/` sections:

| | `sift_files` (today's model) | `embedded_patches` (new) |
|---|---|---|
| `tracks/feature_indexes` | **present** (index into `.sift`) | **absent** |
| `tracks/keypoints_xy` | **absent** | **present** (the coordinate) |
| `images/feature_tool_hashes` | **present** | **absent** |
| `images/sift_content_hashes` | **present** | **absent** |
| `images/image_file_hashes` | **absent** (image hash is reachable via the `.sift`) | **present** (direct image identity) |
| 2D coordinate source | `.sift[feature_indexes[j]]` | `keypoints_xy[j]` directly |

A `sift_files` v4 file is byte-equivalent to a v3 file except for the `version`
and `feature_source` keys in the top-level metadata and the new
`has_feature_indexes` / `has_keypoints_xy` keys in `tracks/metadata.json`. An
`embedded_patches` file is self-contained — it needs no `.sift` companion.

**Image identity in `embedded_patches`.** A `sift_files` file pins each source image
*indirectly*: `.sfmr` → `sift_content_hashes[i]` → the `.sift` file → its
`image_file_xxh128` metadata field (the XXH128 of the source image bytes). With
the `.sift` link gone, an `embedded_patches` file stores that same image hash directly
in `images/image_file_hashes`, so the reconstruction still verifies which image
each observation came from without a `.sift` companion.

## Scope

In scope: a new format version (**v4**) with the `feature_source` discriminator
and the per-mode columns described above.

Out of scope for this version:
- Per-observation ancillaries (scale, quality/ZNCC).
- A `"hybrid"` mode that mixes SIFT and patch observations in one file.
- Changes to the patch *cloud* representation: the keypoints are self-contained
  2D coordinates.

## What v3 stores today

```
tracks/
├── image_indexes.{M}.uint32.zst       # parallel arrays, M = observation_count
├── feature_indexes.{M}.uint32.zst     # index into the per-image .sift file
├── point_indexes.{M}.uint32.zst
├── observation_counts.{N}.uint32.zst  # per point
└── metadata.json.zst
```

The 2D pixel coordinate of observation `j` is `sift.positions[feature_indexes[j]]`
in the `.sift` for image `image_indexes[j]`. The `.sfmr` carries the linkage
but never the coordinate itself.

## What v4 adds

### `tracks/keypoints_xy.{M}.2.float32.zst` (present iff `embedded_patches`)

- **Shape**: `(M, 2)` where M = observation_count
- **Data type**: `float32` (little-endian)
- **Format**: `(u, v)` in image pixel coordinates of `images[image_indexes[j]]`,
  origin = image top-left, half-pixel-center convention (matching the rest of
  the codebase: the pixel in column `x`, row `y` has its centre at
  `(x+0.5, y+0.5)`). Sub-pixel values are expected.
- **Constraint**: finite; the coordinate value lies within
  `[0, width) × [0, height)` of the image's camera intrinsics (the in-frame test
  the rest of the codebase uses for a projected point).
- **Ordering**: parallel to the other `tracks/*` arrays, under the same
  `(point_indexes, image_indexes)` sort.
- **Presence**: present only when `tracks/metadata.json`'s `has_keypoints_xy` is
  `true` (i.e. in an `embedded_patches` file).
- **Hash**: included in the `tracks_xxh128` section hash in lexicographic file
  order, like every other tracks file.

### `tracks/feature_indexes` becomes mode-dependent

- In a `sift_files` file it is **present** with exactly v3 semantics
  (`has_feature_indexes` is `true`).
- In an `embedded_patches` file it is **absent**. A patch observation needs no feature
  index: its coordinate is `keypoints_xy[j]`, and the 3D point it belongs to is
  given by `point_indexes[j]`.

Exactly one of `{feature_indexes, keypoints_xy}` is present in a file, selected
by `feature_source`.

### Per-image `.sift`-link arrays become mode-dependent

```
images/feature_tool_hashes.{N}.uint128.zst   # present iff sift_files
images/sift_content_hashes.{N}.uint128.zst    # present iff sift_files
```

In an `embedded_patches` file both are **absent** (there is no `.sift` to link to),
and the workspace `feature_prefix_dir` is optional.

### `images/image_file_hashes.{N}.uint128.zst` (present iff `embedded_patches`)

The direct image-identity hash that substitutes for the `.sift`-mediated link.

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint128` (little-endian, 16 bytes per hash)
- **Format**: XXH128 of the source image file bytes for `images[i]`, encoded as
  16 little-endian bytes — the same encoding as `sift_content_hashes`. This is
  the same value the image's `.sift` records in its `image_file_xxh128` metadata
  field (see `specs/formats/sift-file-format.md`), where it is a hex string; a
  producer deriving the reconstruction from `.sift` files decodes that hex to the
  16-byte form (the conversion already used for `sift_content_hashes`), while one
  working directly from images computes the XXH128 over the image bytes.
- **Purpose**: verifies that an `images/names[i]` path still resolves to the
  same image the reconstruction was built from, with no `.sift` companion
  required.
- **Presence**: only in `embedded_patches` files. In `sift_files` files it is absent —
  the hash remains reachable through `sift_content_hashes` → `.sift` →
  `image_file_xxh128`, so duplicating it would be redundant.
- **Hash**: joins the `images_xxh128` section hash in lexicographic file order,
  like the other per-image arrays.

## Top-level metadata changes

```json
{
  "version": 4,
  …,
  "feature_source": "sift_files"
}
```

- `"sift_files"` — every observation is SIFT-referenced (the v3 model).
  `feature_indexes` and the per-image `.sift`-link hashes are present;
  `keypoints_xy` is absent.
- `"embedded_patches"` — every observation is patch-derived. `keypoints_xy` and
  `images/image_file_hashes` are present; `feature_indexes` and the per-image
  `.sift`-link hashes (`feature_tool_hashes`, `sift_content_hashes`) are absent.

Legacy v1–v3 files (which have no `feature_source` key) are read as `"sift_files"`.

### Tracks metadata additions

```json
{
  "observation_count": 9427,
  "has_feature_indexes": false,
  "has_keypoints_xy": true
}
```

`has_feature_indexes` and `has_keypoints_xy` flag the presence of the two
mutually-exclusive per-observation columns: exactly one is `true`
(`has_feature_indexes` in a `sift_files` file, `has_keypoints_xy` in an
`embedded_patches` file). They are redundant with `feature_source` but kept local
to the section so a reader that loads `tracks/metadata.json` alone knows which
column to expect, mirroring the existing `points3d/metadata.json` `has_*` flags.
A missing flag is treated as `false`; a legacy v1–v3 file (no flags) is read as
`has_feature_indexes = true`, `has_keypoints_xy = false`.

## Sorting and constraints

All v3 invariants on `tracks/*` carry through:

- The present per-observation arrays remain parallel, length `M`.
- They MUST be sorted by `(point_indexes[j], image_indexes[j])`.
- `observation_counts` must align with the per-point counts derivable from
  `point_indexes`, every value ≥ 1, summing to `observation_count`.

`keypoints_xy[j]` is a parallel row under the same sort order.

## Hashing

The `tracks_xxh128` section hash is computed exactly as in v3: feed every file
present in `tracks/` into the streaming XXH128 hasher in lexicographic path
order. In an `embedded_patches` file the present files are `image_indexes`,
`keypoints_xy`, `metadata.json`, `observation_counts`, `point_indexes` (note
`feature_indexes` is absent); in a `sift_files` file they are the v3 set
(`keypoints_xy` absent). The hash is over present files only.

`images_xxh128` is computed over the per-image files present for the file's
mode: the `.sift`-link hashes (`feature_tool_hashes`, `sift_content_hashes`) in
a `sift_files` file, or `image_file_hashes` in an `embedded_patches` file — fed into the
streaming hasher in lexicographic path order with the always-present image files.

## Implementation plan (separate from this proposal)

Once the spec is agreed:

1. **Rust `sfmr-format`**
   - Accept v1–v4 on read; emit v4 on write.
   - Add `feature_source` to the metadata type, `keypoints_xy:
     Option<Array2<f32>>` and `image_file_hashes: Option<Array1<u128>>` to
     `SfmrData`; make `feature_indexes` and the per-image `.sift`-link hashes
     `Option`.
   - Read path: select required/absent columns by `feature_source`; validate
     shapes, sort, ranges (in-bounds coordinates).
   - Write path: emit the mode-appropriate columns; update naming + hashes.
   - Verify path: include `keypoints_xy` in `tracks_xxh128` when present.
2. **PyO3 bindings (`sfmtool-py`)**
   - Expose `keypoints_xy`, `image_file_hashes`, and `feature_source` on
     `SfmrReconstruction` (read).
   - A `clone_with_changes(keypoints_xy=…, image_file_hashes=…,
     feature_source="embedded_patches")` path that drops the `.sift`-link arrays when
     switching to `embedded_patches`.
3. **Python (`src/sfmtool/`)**
   - `xform/` track-touching operations preserve `keypoints_xy` (it is parallel,
     so most pass it through unchanged) and the `feature_source`.
   - A producer command (e.g. `sfm patches-to-keypoints`) that takes a `sift_files`
     reconstruction with patches, derives per-view sub-pixel keypoints from the
     consensus-aligned shifts (the congealing / group-wise sub-pixel registration
     pipeline from the exploratory experiments), copies each image's
     `image_file_xxh128` from its `.sift` into `image_file_hashes`, and writes an
     `embedded_patches` v4 file.
4. **Tests**
   - Round-trip a hand-built `embedded_patches` v4 file (write → read → byte-equal),
     including presence of `keypoints_xy` / `image_file_hashes` and absence of
     `feature_indexes` / the `.sift`-link hashes.
   - A v3 file upgraded on read has `feature_source == "sift_files"`,
     `has_feature_indexes == true`, `has_keypoints_xy == false`, `keypoints_xy`
     absent.
   - A `sift_files` v4 file is byte-equivalent to the v3 writer's output for the
     same input (modulo metadata keys).
   - Rust and Python writers produce identical content hashes for one input.
5. **Spec** — fold this file into `specs/formats/sfmr-file-format.md` in the
   same commit that ships the v4 reader/writer: drop the "What v3 stores today"
   recap (it duplicates §8 Tracks), and add a "Version 3 → Version 4" migration
   table and a "Version 4" Version-History bullet in the existing style.
