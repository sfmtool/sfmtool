# The camera rig file format

A `.camrig` file stores a **camera rig**: a set of cameras (called *sensors*)
held in fixed relative poses. It is deliberately one format for the whole
range of rigs sfmtool produces:

- a single free-floating camera (1 sensor),
- a back-to-back fisheye pair such as the one `sfm insv2rig` extracts (2 sensors),
- a six-face cubemap from `sfm pano2rig` (6 sensors),
- a spherical tile rig — the "sphere discretised as a rig of pinhole patches"
  of `specs/core/spherical-tiles-rig.md` — from a few dozen up to ~100 000
  sensors.

Having one format means every rig sfmtool builds can be saved and restored
the same way, and tooling that consumes rigs does not branch on rig type.

## Format design principles

These mirror the `.sift`, `.matches`, and `.sfmr` formats; see
`sift-file-format.md` for the original statement.

1. The file is a [ZIP file](https://en.wikipedia.org/wiki/ZIP_(file_format))
   using the STORE method (no ZIP-level compression). Any member can be
   located and its start read without decompressing the rest.
2. Every member is compressed with [zstandard](https://en.wikipedia.org/wiki/Zstd)
   and carries a `.zst` extension.
3. Metadata that occurs once per file is stored as JSON, compact (no
   pretty-printing).
4. Tables are stored columnar — one file per field, one primitive type per
   file.
5. Numeric data is little-endian raw binary in C/row-major order.
6. Data-file extensions encode the tensor shape and primitive type. E.g.
   `.100000.4.float64` is a tensor of shape `(100000, 4)` of 64-bit IEEE
   floats, and the file must hold `100000 * 4 * 8` bytes.
7. While writing, a hash summarising the whole file is computed and stored,
   so references to a `.camrig` file can use the content hash held inside it.
8. Member names, JSON field names, and table field names are chosen to be
   self-documenting.

The format is designed so that a 100 000-sensor rig stays small: sensors
that share intrinsics reference a single shared camera (see *Cameras* below),
and a co-centric rig, where every sensor has the same optical center, will
have all zero translations — both compress to almost nothing under zstd.

## Concepts

A **rig** is an ordered list of `sensor_count` **sensors**. A rig always has
at least one sensor.

Each sensor has:

- **Intrinsics** — a camera model, image dimensions, and parameters. Many
  sensors typically share intrinsics (all tiles of a spherical tile rig are
  identical; the two halves of an insv2 rig are identical), so intrinsics are
  stored once in a shared *camera pool* and each sensor references one by
  index.
- **Extrinsics** — the sensor's pose within the rig, `sensor_from_rig`: the
  rigid transform that takes a point from rig coordinates to that sensor's
  camera coordinates, `p_sensor = R · p_rig + t`. Sensor camera frames are
  canonical −Z-forward frames (see *Coordinate and quaternion conventions*).

The rig has its own coordinate frame, chosen by the producer. For a
co-centric rig the frame is conventionally placed with its origin at the
shared optical centre. Sensor `sensor_from_rig` poses are arbitrary —
**no sensor is required to sit at the identity pose**, and the rig frame
need not coincide with any sensor's frame.

**Sensor 0 is the primary sensor** — the canonical sensor of the rig, by
convention placed first in sensor order: the left camera of a stereo pair,
the front face of a cubemap, tile 0 of a spherical tile rig. It carries no
pose constraint; it is the sensor that becomes COLMAP's identity-posed
reference when exporting to the COLMAP rig format (see *Relationship to the
COLMAP rig format*). Sensor order is otherwise unconstrained — the per-sensor
tables are parallel, so a producer is free to choose whatever order makes
sensor 0 the canonical one.

Placing the whole rig in a world or reconstruction frame is out of scope for
this format — a consumer that needs that supplies the placement separately.
(Type-specific placement, such as a spherical tile rig's optical centre, may
be recorded in `rig_attributes` as a hint; see *Rig type and attributes*.)

### Coordinate and quaternion conventions

- **Sensor frames follow the canonical `.sfmr` camera convention.** Each
  sensor's frame is right-handed with the sensor **looking down −Z**: the
  sensor's optical axis is its local **−Z**, and in the image plane **+X
  points right** and **+Y points up** — the opposite of COLMAP/OpenCV, where
  the camera looks down +Z with Y down. `sensor_from_rig` poses (and any
  pose derived from them) map into frames of this convention. See the
  "Coordinate System Conventions" section of
  [`sfmr-file-format.md`](sfmr-file-format.md) for the full statement;
  `.camrig` sensor poses and `.sfmr` rig sensor poses share it, so they are
  interchangeable without conversion.
- **Quaternions are WXYZ** — stored `[w, x, y, z]`, scalar first. This
  matches `.sfmr` and the COLMAP rig format. Quaternions are unit
  quaternions.
- **`sensor_from_rig`** is the rig→sensor transform. The sensor's optical
  centre expressed in rig coordinates is `-Rᵀt`. For a co-centric rig (every
  sensor shares one optical centre, e.g. a cubemap or a spherical tile rig)
  the rig frame origin is placed at that centre, so every translation is
  `[0,0,0]` and only the rotation varies per sensor.
- Camera-model parameter conventions (image-plane vs pixel coordinates,
  principal-point origin, distortion model) follow COLMAP, as documented for
  `.sfmr` cameras — only the camera-space axes differ.

> **Migration note.** The canonical sensor-frame convention was formalized
> after the format was already in use; files written by earlier sfmtool
> releases (format version 1) hold COLMAP-convention sensor poses (sensors
> looking down +Z with Y down). See
> [Versioning and migration](#versioning-and-migration).

## Specification

A `.camrig` file is a ZIP archive (STORE method) containing exactly the
following members. `{S}` denotes `sensor_count`.

```
metadata.json.zst
cameras/metadata.json.zst
sensors/image_file_patterns.json.zst
sensors/camera_indexes.{S}.uint32.zst
sensors/quaternions_wxyz.{S}.4.float64.zst
sensors/translations_xyz.{S}.3.float64.zst
content_hash.json.zst
```

### `metadata.json.zst`

Compact JSON, zstd-compressed. Fields (consumers ignore unknown fields, for
forward-compatible extension):

* `version`: (integer) Format version. `1` or `2` (see
  [Versioning and migration](#versioning-and-migration)).
* `name`: (string) A human-readable name for the rig, e.g. `"insv2_x5"`,
  `"cubemap"`, `"spherical_tiles_n1280"`. May be empty.
* `sensor_count`: (integer) Number of sensors, `S ≥ 1`.
* `camera_count`: (integer) Number of distinct cameras in the pool, `≥ 1`.
* `rig_type`: (string) A hint describing how the rig was generated. See
  *Rig type and attributes*. Use `"generic"` when nothing more specific
  applies.
* `rig_attributes`: (object) Free-form, `rig_type`-specific key/value data.
  Not load-bearing — the generic sensor/camera tables fully describe the rig
  geometry without it. May be `{}`.

### `cameras/metadata.json.zst`

Compact JSON, zstd-compressed: a JSON **array** of exactly `camera_count`
camera objects. Each object has the same shape as a `.sfmr` camera:

* `model`: (string) COLMAP camera model name, e.g. `"PINHOLE"`,
  `"OPENCV_FISHEYE"`, `"EQUIRECTANGULAR"`.
* `width`: (integer) Image width in pixels.
* `height`: (integer) Image height in pixels.
* `parameters`: (object) Named model parameters as a string→number map. The
  key set is determined by `model` and follows the `.sfmr` naming convention
  (see [`sfmr-file-format.md`](sfmr-file-format.md)) —
  `focal_length_x`, `principal_point_x`, `radial_distortion_k1`, ….

Unlike COLMAP's `rig_config.json`, intrinsics here carry `width`/`height`:
a `.camrig` file is self-contained and does not assume the source images are
available.

### `sensors/image_file_patterns.json.zst`

Compact JSON, zstd-compressed: a JSON **array** of strings.

* If the array has `sensor_count` entries, entry `i` is sensor `i`'s image
  file pattern — the path pattern identifying that sensor's images in a
  workspace (see *How `.camrig` files fit into workspaces*). The format
  treats each entry as an opaque string and leaves pattern syntax to
  workspace tooling, with one exception: it checks each pattern's frame-field
  count — at most one in any pattern, and exactly one per pattern in a
  multi-sensor rig (see *Data ordering and constraints*).
* If the array is **empty** (`[]`), the rig is not backed by workspace
  images — it describes pure geometry, and sensors are referred to by index.
  This is the expected case for large generated rigs (a 100 000-tile
  spherical tile rig has no per-tile image files).

A non-empty array of the wrong length is invalid.

### `sensors/camera_indexes.{S}.uint32.zst`

zstd-compressed little-endian `uint32` array of length `S`. Entry `i` is the
index in `[0, camera_count)` of sensor `i`'s camera in the pool.

### `sensors/quaternions_wxyz.{S}.4.float64.zst`

zstd-compressed little-endian `float64` array of shape `(S, 4)`. Row `i` is
the WXYZ unit quaternion of sensor `i`'s `sensor_from_rig` rotation.

### `sensors/translations_xyz.{S}.3.float64.zst`

zstd-compressed little-endian `float64` array of shape `(S, 3)`. Row `i` is
the XYZ translation of sensor `i`'s `sensor_from_rig`.

### `content_hash.json.zst`

Compact JSON, zstd-compressed. Fields:

* `metadata_xxh128`: XXH128 hex digest of the uncompressed `metadata.json`
  bytes.
* `content_xxh128`: XXH128 hex digest of the concatenation of the XXH128
  digests (each as 16 big-endian bytes) of the following uncompressed
  members, in order:
  1. `metadata.json`
  2. `cameras/metadata.json`
  3. `sensors/image_file_patterns.json`
  4. `sensors/camera_indexes.{S}.uint32`
  5. `sensors/quaternions_wxyz.{S}.4.float64`
  6. `sensors/translations_xyz.{S}.3.float64`

All hashes are 32-character lowercase hex strings. XXH128 is fast and
non-cryptographic but has strong collision resistance; it matches the other
sfmtool formats.

## Data ordering and constraints

- `version` is a known version (`1` or `2` — see
  [Versioning and migration](#versioning-and-migration)).
- `sensor_count ≥ 1`, `camera_count ≥ 1`.
- `sensor_count` equals the length of every per-sensor table
  (`camera_indexes`, `quaternions_wxyz`, `translations_xyz`, and
  `image_file_patterns` unless that one is empty); `camera_count` equals the
  `cameras` array length.
- Every entry of `camera_indexes` is in `[0, camera_count)`.
- Quaternions are unit length (within tolerance).
- Every image pattern is a relative forward-slash path: no leading `/`, no
  `..` component, and every `**` stands alone as a whole path segment.
- Every image pattern contains at most one frame field (`%d` or `%0Nd`) —
  see *How `.camrig` files fit into workspaces*.
- If `sensor_count > 1` and the rig is image-backed (non-empty
  `image_file_patterns`), every pattern contains exactly one frame field. A
  frame-field-less pattern is permitted only in a single-sensor rig.

A reader must reject a file that violates any of these constraints, rather
than passing it through to a consumer — a file can be byte-intact (its
content hashes verify) yet structurally invalid.

Each camera's `parameters` key set is expected to match the key set its
`model` defines. The format layer does not enforce this — `.camrig` carries
no camera-model registry — so it is checked by the camera-model layer that
consumes the rig, not by the `.camrig` reader.

- The pool should contain only *distinct* cameras; a writer is expected to
  deduplicate intrinsics so that, e.g., a 100 000-tile rig stores one camera.
  This is a recommendation, not a hard validity rule — duplicate pool entries
  are tolerated on read.

## Rig type and attributes

`rig_type` and `rig_attributes` let a producer record how a rig was built
without changing the generic representation. A consumer that does not
recognise a `rig_type` still reads the rig correctly from the sensor and
camera tables. Defined values so far:

| `rig_type` | Meaning | Suggested `rig_attributes` |
|---|---|---|
| `generic` | No specific structure asserted. | `{}` |
| `stereo_pair` | Two side-by-side sensors with near-parallel optical axes, in a stereo-vision configuration. | `{ "baseline_m": <float> }` |
| `fisheye_360` | Two fisheye sensors facing opposite directions for full 360° coverage (e.g. `sfm insv2rig`). | `{ "baseline_m": <float> }` |
| `cubemap` | Six co-centric pinhole faces (e.g. `sfm pano2rig`). | `{}` |
| `spherical_tiles` | Co-centric pinhole tiles discretising the sphere (see `specs/core/spherical-tiles-rig.md`). | `centre` (`[x,y,z]` world placement), `measured_max_nn_angle`, `measured_max_coverage_angle`, `atlas_cols`, plus informational `half_fov_rad` and `patch_size` |

For a `spherical_tiles` rig, the generic tables already capture every tile's
intrinsics (one shared pinhole camera) and rotation (per-tile quaternion,
zero translation) — the per-tile rotations *are* the tile look directions and
tangent bases, and the shared pinhole camera *is* the authoritative tile
intrinsics, from which `patch_size` and `half_fov_rad` are recovered on read.
`rig_attributes` records the remaining derived scalars (world placement, the
measured coverage angles, and atlas packing) so the exact rig can be
reconstructed without re-running the sphere-point relaxer. It also stores
`half_fov_rad` and `patch_size`, but those are informational copies of values
already implied by the shared camera and are not consumed when the rig is
reloaded. The construction inputs (`n`, `arc_per_pixel`, `overlap_factor`,
relaxer seed) are not stored: `n` is `sensor_count`, and the rest only feed
the relaxer, which the reconstruction bypasses.

## How `.camrig` files fit into workspaces

A `.camrig` file may be placed anywhere in a workspace. The directory that
contains it is the rig's **root directory**, and every image the rig refers
to lives in that directory or one of its subdirectories. A rig is therefore
self-locating: put the `.camrig` file at the top of the image subtree it
describes and nothing else needs to point at it. Several `.camrig` files can
coexist in one workspace, each scoped to its own subtree.

A sensor that is backed by image files carries an **image pattern** — a
path, relative to the root directory, that identifies its images:

```
left/image_%04d.jpg
right/image_%04d.jpg
```

or, for a single-camera rig dropped alongside a directory of images:

```
*.jpg
```

The pattern is the sensor's entry in `sensors/image_file_patterns.json.zst`.
It is a forward-slash relative path (no leading `/`, no `..`) that may
descend into subdirectories. A pattern is built from literal text plus two
kinds of special token: **glob wildcards**, which broaden which files match,
and an optional **frame field**, which assigns frame indices.

**Glob wildcards** match path text but capture nothing. A pattern may
contain any number of them:

- `*` — matches zero or more characters within a single path segment; it
  does **not** match `/`.
- `**` — matches zero or more whole path segments, separators included
  (`a/**/b.jpg`, `**/img.png`). It must occupy a full path segment.

`*` and `**` are always wildcards — there is no escape for a literal `*`.

The **frame field** is optional and may appear **at most once**:

- `%0Nd` — a zero-padded integer `N` digits wide (`%04d` matches `0007`).
- `%d` — an unpadded integer.
- `%%` — a literal `%`.

The pattern does two things:

1. **Membership.** The files whose paths match the pattern — glob wildcards
   and frame field included — are exactly that sensor's images. Patterns for
   different sensors must not match the same file.
2. **Frame grouping.** Images from different sensors captured at the same
   instant make up one rig **frame**; their relative pose is fixed by the
   rig. Each image's **frame index** is what assigns it to a frame:
   - If the pattern has a frame field, the integer it captures is the frame
     index. Glob wildcards capture nothing and never contribute to it.
   - If the pattern has no frame field, the sensor's matched files are
     sorted by relative path and assigned frame indices `0, 1, 2, …` by
     position. A pattern with neither a frame field nor a wildcard matches a
     single file — a single-frame sensor — as the degenerate case of this
     rule.

A frame-field-less pattern is only valid in a **single-sensor** rig. With
more than one sensor, frames are paired across sensors by frame index, so
every pattern must carry a frame field; a rig with `sensor_count > 1` whose
patterns omit the frame field is invalid.

This is the `.camrig` equivalent of COLMAP's frame grouping by stripped
`image_prefix` (see *Relationship to the COLMAP rig format*), stated as an
explicit pattern rather than inferred from a prefix.

A rig whose sensors are not backed by workspace images — a spherical tile
rig, for instance — has an empty `sensors/image_file_patterns.json.zst`,
defines no patterns, and describes pure geometry.

## Relationship to the COLMAP rig format

The [**COLMAP rig-configurator format**](https://colmap.github.io/rigs.html)
is a JSON array of rigs, each a list of camera entries with `image_prefix`,
`ref_sensor`, `cam_from_rig_rotation`,
`cam_from_rig_translation`, `camera_model_name`, `camera_params`. sfmtool
supports it in workspaces as `rig_config.json` (see
`specs/workspace/rig-config.md`).

Structural differences:

- A COLMAP rig file holds **one or more** rigs; a `.camrig` file holds
  exactly one. A multi-rig COLMAP file maps to multiple `.camrig` files.
- A COLMAP entry stores intrinsics positionally (`camera_params` in COLMAP's
  parameter order); `.camrig` stores them as a named `parameters` map. The
  positional↔named mapping is the one already used for `.sfmr` cameras.
- A COLMAP rig file has no image dimensions; `.camrig` cameras carry
  `width`/`height`. `.camrig` → COLMAP drops them; COLMAP → `.camrig`
  supplies them from the images.

Field correspondences:

- Both store a rig→sensor transform with WXYZ quaternions, but in
  **different camera-frame conventions**: COLMAP poses map into
  +Z-forward / Y-down camera frames, `.camrig` poses into canonical
  −Z-forward / +Y-up frames (see *Coordinate and quaternion conventions*).
  Conversion conjugates each pose with the camera-frame flip
  `S = diag(1, −1, −1)`: `R_colmap = S · R · S`, `t_colmap = S · t`
  (`S` is involutive, so the reverse direction is the same formula).
- COLMAP defines its rig frame by the reference sensor, whose `cam_from_rig`
  is the identity; `.camrig` does not constrain any pose. `.camrig` → COLMAP
  therefore does two things: **rebase** so sensor 0 (the primary sensor) is
  the reference, and **S-conjugate** each sensor's canonical camera frame to
  COLMAP's. Rebasing is the sensor-0→sensor-i relative transform (the rig
  frame cancels), `sensor_from_rig[i] · sensor_from_rig[0]⁻¹`; S-conjugation
  re-expresses that transform's two endpoint camera frames in COLMAP
  convention. Composed:

  ```
  cam_from_rig[i] = S · sensor_from_rig[i] · sensor_from_rig[0]⁻¹ · S
  ```

  The order of the two operations does not matter: `X ↦ S · X · S` is an
  automorphism (`S · S = I`), so conjugating each pose first
  (`S · sensor_from_rig[k] · S`) and then rebasing yields the same result —
  the inner `S · S` on the reference term cancels, leaving exactly the
  expression above. COLMAP → `.camrig` inverts it: S-conjugate the COLMAP
  poses back (`sensor_from_rig[i] = S · cam_from_rig[i] · S`); no rebasing is
  needed, since `.camrig` does not require sensor 0 at identity.
- A COLMAP sensor's `image_prefix` corresponds to a `.camrig` sensor image
  file pattern.

## Using CLI tools to inspect a `.camrig` file

```bash
$ unzip spherical_tiles_n1280.camrig
Archive:  spherical_tiles_n1280.camrig
 extracting: metadata.json.zst
 extracting: cameras/metadata.json.zst
 extracting: sensors/image_file_patterns.json.zst
 extracting: sensors/camera_indexes.1280.uint32.zst
 extracting: sensors/quaternions_wxyz.1280.4.float64.zst
 extracting: sensors/translations_xyz.1280.3.float64.zst
 extracting: content_hash.json.zst

$ zstd -d --rm *.zst cameras/*.zst sensors/*.zst

$ jq . metadata.json
{
  "version": 1,
  "name": "spherical_tiles_n1280",
  "sensor_count": 1280,
  "camera_count": 1,
  "rig_type": "spherical_tiles",
  "rig_attributes": {
    "centre": [0.0, 0.0, 0.0],
    "half_fov_rad": 0.0654,
    "measured_max_nn_angle": 0.1138,
    "measured_max_coverage_angle": 0.0569,
    "patch_size": 21,
    "atlas_cols": 36
  }
}

$ jq . cameras/metadata.json
[
  {
    "model": "PINHOLE",
    "width": 21,
    "height": 21,
    "parameters": {
      "focal_length_x": 160.2,
      "focal_length_y": 160.2,
      "principal_point_x": 10.5,
      "principal_point_y": 10.5
    }
  }
]
```

## Versioning and migration

The format has two released versions (`1` and `2`). The format is versioned
(`metadata.json` `version`) precisely so that convention changes like the one
below can upgrade on load instead of breaking old files.

### Version 1 → Version 2

Version 2 makes the canonical `.sfmr` coordinate convention normative for
sensor poses (see *Coordinate and quaternion conventions*), mirroring the
`.sfmr` version 5 bump. No member is added, removed, or renamed; the change
is purely semantic:

| Version 1 | Version 2 |
|-----------|-----------|
| `sensor_from_rig` poses in COLMAP convention (sensors look down +Z with Y down) | Canonical convention: sensors look down −Z with +Y up |

**Migration is mechanical and lossless**, but the transport depends on how the
rig is anchored. Most rigs are **body-anchored** — the rig frame is a
reference-sensor frame, so both frames flip under the convention change and a
version 1 file upgrades on load by conjugating each stored `sensor_from_rig`
pose with the camera-frame flip `S = diag(1, −1, −1)`: `R' = S · R · S`,
`t' = S · t` (the rig frame never touches the world, so the `.sfmr` world
canonicalization `W` does not apply). A `spherical_tiles` rig (`rig_type ==
"spherical_tiles"`) is instead **world-anchored** — its rig frame *is* the
reconstruction world, which the convention change leaves fixed — so only the
sensor frame flips and the upgrade is a left multiply `R' = S · R`,
`t' = S · t`. Applying the body-anchored conjugation to a tile rig would
leave every sensor rotated 180° about the world X axis (the wrong hemisphere)
while still passing structural validation, so the upgrade dispatches on
`rig_type`. Saving always writes version 2. Content hashes cover the stored
bytes, so hashes verify before conversion; a converted-then-saved file is a
new version 2 file with new hashes.

As a consequence, `test-data/images/kerry_park/kerry_park.camrig` must be
regenerated in the canonical convention. The implementation roadmap is
`specs/drafts/zup-camera-convention-migration.md` (decision D5).

## Version history

- **Version 2**: Canonical coordinate convention — sensor frames
  look down −Z with +Y up, matching `.sfmr` — becomes normative; version 1
  files (COLMAP convention) upgrade on load by flipping the stored
  `sensor_from_rig` poses (`S`-conjugation for body-anchored rigs, a left `S`
  multiply for world-anchored `spherical_tiles` rigs).
- **Version 1**: Initial release.
