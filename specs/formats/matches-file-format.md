# The Matches File Format

A `.matches` file records feature correspondences among a set of images, the
data a Structure-from-Motion solve starts from. It holds them in one of two
forms: matches grouped per image pair, or clusters of features across several
images that are likely observations of the same surface point. A pairwise file
can also hold the result of geometric verification for each pair (the inlier
matches and the estimated relative pose); a cluster file can also hold the
photometric vetting of each cluster member. Feature indexes point into the
per-image `.sift` files; the file locates them through its image paths and
the feature directory it records, and stores their content hashes so a reader
can tell whether they have changed since.

## Motivation

The `.sift` format lets us extract features once and experiment with subsets. The `.sfmr` format
stores reconstructions at every stage — initial solves, filtered subsets, aligned and merged
results. The intermediate matching step — the bridge between features and reconstruction —
also needs a persistent format. Matches are otherwise ephemeral, temporarily stored in a
COLMAP SQLite database.

There are many matching strategies — COLMAP includes exhaustive, sequential, vocabulary tree,
and spatial matching. Many are costly to compute, and needing to recompute them each time
attempting a solve makes working with the pipeline less fun. It also is harder to inspect
matches independently, combine matches from multiple strategies before solving, or reuse
the same matches across different solves with different solvers or solver options.

A `.matches` file format solves this by storing:

1. **A correspondence backbone** — exactly one of:
   - **Candidate matches** (`image_pairs/`) — raw pairwise feature correspondences from a
     matching step, or
   - **Clusters** (`clusters/`) — groups of features across images that are likely
     co-observations of one surface point, the primary artifact of the cluster matcher
     (`sfm match --cluster`); pairwise matches are a derived view obtained by expansion
2. **Two-view geometries** (optional, requires `image_pairs/`) — geometrically verified
   inlier subsets with estimated relative poses such as F/E/H matrices, rotation, translation
3. **Cluster patches** (optional, requires `clusters/`) — per-cluster photometrically
   refined affine warps from a reference member to every other member, with vetting statuses

Following the approach to data files of an sfmtool workspace, `.matches` files are **write-once**.
You never edit an existing `.matches` file — instead, you write a new one. Different matching
configurations produce separate files, so they coexist without conflict. This is the same approach
used by `.sfmr` files.

There are two expected ways to produce a `.matches` file:

1. **Matches only**: Write matches immediately after a matching step that does not perform geometric
   verification. This gives an opportunity to inspect the matches, produce matches with different
   approaches to combine, or apply filters before geometric verification.

2. **Matches + two-view geometries**: Write matches that includes both candidate matches and
   geometric verification results. This data is ready to copy into a COLMAP database for solving.

A `.matches` file does not reference other `.matches` files, it only references feature files.
A process that takes several `.matches` files and combines them would copy all the data it
selects.

## Design Principles

A `.matches` file is an [archive-container](archive-container.md) file, like
`.sift`, `.sfmr` and `.camrig`: a ZIP archive of individually zstd-compressed
entries, compact JSON for metadata, little-endian columnar binary for the tables
(entry names carry their shape and data type), and XXH128 hashes over the
uncompressed bytes. That spec covers the container; everything here is what
`.matches` puts in it.

This format largely adopts the semantics of the [COLMAP Database Format](https://colmap.github.io/database.html)
as the basis for image pairs and two view geometries. It uses indexes that are always a
contiguous range from 0 to N-1, different from the potentially non-contiguous IDs in a 
COLMAP database starting from 1. One deliberate divergence is the camera-frame
convention: two-view relative poses are stored in the canonical −Z-forward camera
convention described below, **not** in COLMAP's +Z-forward, Y-down convention.

## Coordinate Conventions

Match data (feature indexes, descriptor distances) lives in pixel space and carries no
camera-frame convention. The only convention-bearing data in a `.matches` file is the
pair of relative-pose arrays in the optional two-view geometries section:

- **Relative poses follow the canonical `.sfmr` camera convention.**
  `two_view_geometries/quaternions_wxyz` and `two_view_geometries/translations_xyz`
  store the relative pose `cam2_from_cam1` — the rigid transform taking a point from
  image `idx_i`'s camera frame (cam1) to image `idx_j`'s camera frame (cam2),
  `p_2 = R · p_1 + t`, matching the COLMAP `two_view_geometries` table semantics. Both
  camera frames are right-handed with the camera **looking down −Z**: in the image
  plane **+X points right** and **+Y points up** — the opposite of COLMAP/OpenCV,
  where the camera looks down +Z with Y down. See the "Coordinate System Conventions"
  section of [`sfmr-file-format.md`](sfmr-file-format.md) for the full statement;
  `.matches` relative poses, `.sfmr` poses, and `.camrig` sensor poses share it, so
  they compose without conversion.
- **The stored F/E/H matrices are pixel-space quantities and are NOT affected by the
  camera convention.** The fundamental matrix and homography relate pixel coordinates
  directly (`x_2ᵀ F x_1 = 0`), and the essential matrix relates `K`-normalized pixel
  coordinates — all are defined by image measurements, which the camera-axis
  convention does not touch (pixel space keeps its top-left origin with y down). A
  reader might expect `E` to flip along with the poses; it does not, because the
  stored `E` is a constraint on normalized pixel coordinates, not on canonical
  camera-frame rays. A consumer that instead *derives* `E` or `F` from the stored
  relative pose plus intrinsics must first map the pose back to the COLMAP/OpenCV
  frame by conjugating with the camera-frame flip `S = diag(1, −1, −1)`.

> **Migration note.** The canonical camera convention was formalized after the format
> was already in use; files of format version 1 hold COLMAP-convention relative
> poses (cameras looking down +Z with Y down). See
> [Versioning and Migration](#versioning-and-migration).

## File Structure

```
match-output-file.matches (ZIP archive)
├── metadata.json.zst                              # Top-level metadata
├── content_hash.json.zst                          # Integrity verification
├── images/
│   ├── names.json.zst                             # Image file paths (workspace-relative)
│   ├── metadata.json.zst                          # Per-image metadata
│   ├── feature_tool_hashes.{N}.uint128.zst        # Feature tool identification
│   ├── sift_content_hashes.{N}.uint128.zst        # Feature file content verification
│   ├── feature_counts.{N}.uint32.zst              # Feature count per image (as used in matching)
│   └── image_dims.{N}.2.uint32.zst                # Per-image pixel dimensions (width, height)
├── image_pairs/                                   # (Backbone alternative A: pairwise)
│   ├── metadata.json.zst                          # Pair-level metadata
│   ├── image_index_pairs.{P}.2.uint32.zst         # (idx_i, idx_j) per pair, idx_i < idx_j
│   ├── match_counts.{P}.uint32.zst                # Number of matches per pair
│   ├── match_feature_indexes.{M}.2.uint32.zst     # (feat_idx_i, feat_idx_j) per match
│   └── match_descriptor_distances.{M}.float32.zst # L2 descriptor distance per match
├── clusters/                                      # (Backbone alternative B: clusters)
│   ├── metadata.json.zst                          # cluster_count, member_count, matcher options
│   ├── cluster_starts.{C+1}.uint32.zst            # CSR offsets: cluster c owns members starts[c]..starts[c+1]
│   ├── member_images.{K}.uint32.zst               # Index into images/names.json.zst per member
│   ├── member_features.{K}.uint32.zst             # Feature index in that image's .sift per member
│   ├── member_positions.{K}.2.float32.zst         # Keypoint position per member, at this file's stage
│   └── member_affine_shapes.{K}.2.2.float32.zst   # Affine shape per member, at this file's stage
├── cluster_patches/                               # (Optional section, requires clusters/)
│   ├── metadata.json.zst                          # Refinement options, summary counts, status legends
│   ├── reference_members.{C}.uint32.zst           # Global member index of each cluster's reference
│   ├── member_status.{K}.uint8.zst                # Status code per member, an index into the legend
│   ├── member_consistency_residual.{K}.float32.zst # Warp-consistency residual (NaN if not fitted)
│   ├── member_shift_px.{K}.float32.zst            # Translation drift from the SIFT seed (NaN if n/a)
│   ├── member_zncc.{K}.float32.zst                # Achieved windowed ZNCC vs reference (NaN if n/a)
│   ├── member_cell_shift_px.{K}.3.3.2.float32.zst # (Optional, v8+) Per-cell displacement (NaN if not measured)
│   ├── member_cell_zncc.{K}.3.3.float32.zst       # (Optional, v8+) Per-cell ZNCC (NaN if not read)
│   ├── member_cell_status.{K}.3.3.uint8.zst       # (Optional, v8+) Per-cell status, an index into its legend
│   └── member_cell_iterations.{K}.uint8.zst       # (Optional, v8+) Piecewise refinement passes per member
└── two_view_geometries/                           # (Optional section, requires image_pairs/)
    ├── metadata.json.zst                          # TVG metadata
    ├── config_types.json.zst                        # Unique TwoViewGeometryConfig type strings
    ├── config_indexes.{P}.uint8.zst                 # Index into config_types per pair
    ├── inlier_counts.{P}.uint32.zst                 # Number of inlier matches per pair
    ├── inlier_feature_indexes.{I}.2.uint32.zst      # (feat_idx_i, feat_idx_j) per inlier
    ├── f_matrices.{P}.3.3.float64.zst               # Fundamental matrices (row-major 3x3)
    ├── e_matrices.{P}.3.3.float64.zst               # Essential matrices (row-major 3x3)
    ├── h_matrices.{P}.3.3.float64.zst               # Homography matrices (row-major 3x3)
    ├── quaternions_wxyz.{P}.4.float64.zst           # Relative rotation quaternions
    └── translations_xyz.{P}.3.float64.zst           # Relative translation vectors
```

Where:
- `{N}` = number of images
- `{P}` = number of image pairs
- `{M}` = total number of matches across all pairs
- `{C}` = number of clusters
- `{K}` = total number of cluster members across all clusters
- `{I}` = total number of inlier matches across all pairs (two-view geometries)

**The backbone rule (version 3):** every `.matches` file stores **exactly one** of
`image_pairs/` and `clusters/` as its correspondence backbone. The metadata `has_clusters`
flag selects which. `two_view_geometries/` requires the pairwise backbone (its arrays are
keyed per stored pair); `cluster_patches/` requires the cluster backbone. Version ≤ 2 files
always store the pairwise backbone.

## File Format Details

### 1. Top-Level Metadata (`metadata.json.zst`)

```json
{
  "version": 9,
  "matching_method": "sequential",
  "matching_tool": "colmap",
  "matching_tool_version": "4.02",
  "matching_options": {
    "overlap": 10,
    "quadratic_overlap": true,
    "max_feature_count": 8192
  },
  "workspace": {
    "absolute_path": "/path/to/workspace",
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
      "feature_prefix_dir": "features/sift-colmap-c220a90eb516a6654748c328f3403054"
    }
  },
  "timestamp": "2026-03-29T10:00:00Z",
  "image_count": 83,
  "image_pair_count": 332,
  "match_count": 145000,
  "has_two_view_geometries": false,
  "has_clusters": false,
  "has_cluster_patches": false
}
```

A cluster-bearing file replaces the pairwise summary fields with cluster counts:

```json
{
  "...": "...",
  "image_count": 83,
  "cluster_count": 5200,
  "cluster_member_count": 14100,
  "has_two_view_geometries": false,
  "has_clusters": true,
  "has_cluster_patches": false
}
```

**Field descriptions:**
- `version`: Format version number. `1` through `9`; writers emit `9` (see
  [Versioning and Migration](#versioning-and-migration))
- `matching_method`: Type of matching used to produce these matches. The
  format does not restrict the string; these values have a defined meaning:
  - `"exhaustive"`: a pairwise file whose matcher compared the descriptors of
    every pair of images in the file
  - `"sequential"`: a pairwise file whose matcher compared the descriptors of
    each image only with images near it in capture order (the count is in
    `matching_options`)
  - `"flow"`: a pairwise file whose correspondences were found by following
    dense optical flow from one image to the other, not by comparing
    descriptors
  - `"cluster"`: a cluster-backbone file whose matcher grouped features across
    images into clusters directly, with no stored pairwise stage
  - `"merged"`: the union of several `.matches` files; the source files'
    methods are listed in `matching_options["source_methods"]`
  - A file derived from another one — its verified pairwise expansion, its
    refinement-stage counterpart, or a cluster selection — keeps its source's
    value
- `matching_tool`: Tool that produced the matches (e.g., `"colmap"`)
- `matching_tool_version`: Version string of the tool
- `matching_options`: Method-specific parameters. Contents depend on `matching_method` and
  `matching_tool`. Examples:
  - For COLMAP `"sequential"`: `overlap`, `quadratic_overlap`
  - For COLMAP `"exhaustive"`: `block_size`
  - Other methods: tool-specific key-value pairs
- `workspace`: Same structure as in `.sfmr` files — identifies the workspace and feature
  extraction configuration
- `timestamp`: ISO 8601 format with timezone
- `image_count`: Number of images referenced
- `image_pair_count`: Number of image pairs with matches. Present exactly when the file
  stores the pairwise backbone (`has_clusters` false); absent in cluster-bearing files
- `match_count`: Total number of matches across all pairs. Present exactly when the file
  stores the pairwise backbone
- `cluster_count`: Number of clusters. Present exactly when the file stores the cluster
  backbone (`has_clusters` true)
- `cluster_member_count`: Total number of cluster members. Present exactly when the file
  stores the cluster backbone
- `has_two_view_geometries`: Whether the optional two-view geometries section is present
  (pairwise backbone only)
- `has_clusters`: Whether the file stores the `clusters/` backbone instead of
  `image_pairs/`. Absent in version ≤ 2 files (readers treat absence as `false`)
- `has_cluster_patches`: Whether the optional `cluster_patches/` section is present
  (requires `has_clusters`). Absent in version ≤ 2 files

### 2. Content Hash (`content_hash.json.zst`)

```json
{
  "metadata_xxh128": "...",
  "images_xxh128": "...",
  "image_pairs_xxh128": "...",
  "clusters_xxh128": "...",
  "cluster_patches_xxh128": "...",
  "two_view_geometries_xxh128": "...",
  "content_xxh128": "..."
}
```

**Field descriptions.** Each section hash covers that section's data files in
lexicographic path order; see [archive-container.md](archive-container.md) for how
a section digest is taken and how the digests combine.

- `metadata_xxh128`: Hash of the uncompressed `metadata.json.zst` content bytes
- `images_xxh128`: The `images/` section hash
- `image_pairs_xxh128`: The `image_pairs/` section hash. Present exactly when the
  file stores the pairwise backbone.
- `clusters_xxh128`: The `clusters/` section hash. Present exactly when the file
  stores the cluster backbone.
- `cluster_patches_xxh128`: (Optional) The `cluster_patches/` section hash. Present
  only when the cluster patches section exists.
- `two_view_geometries_xxh128`: (Optional) The two-view geometries section hash.
  Present only when that section exists.
- `content_xxh128`: The whole-file digest over all present section hashes, in the
  order metadata, images, pairs, clusters, cluster_patches, two_view_geometries
  (each only if present). A pairwise file's byte stream is identical to the
  pre-version-3 layout, so version ≤ 2 hashes verify unchanged.

### 3. Images

The images section identifies which images and features the matches reference. Feature indexes
in the match data are indices into `.sift` files, so consumers need to locate those files.

The `.sift` file for an image is found by combining the workspace directory, the image's parent
directory, the `feature_prefix_dir` from the workspace contents in the top-level metadata, and the image basename:

```
{workspace}/{image_parent}/{feature_prefix_dir}/{image_basename}.sift
```

For example, with `feature_prefix_dir` of `features/sift-colmap-c220a90eb516a6654748c328f3403054`
and image path `frames/frame_0000.jpg`, the `.sift` file is at:

```
{workspace}/frames/features/sift-colmap-c220a90eb516a6654748c328f3403054/frame_0000.jpg.sift
```

The `sift_content_hashes` array can be used to verify that the `.sift` files haven't changed since the
matches were computed.

#### `images/metadata.json.zst`

```json
{
  "image_count": 83
}
```

#### `images/names.json.zst`

Array of image paths **relative to workspace directory** (POSIX format):

```json
[
  "frames/frame_0000.jpg",
  "frames/frame_0010.jpg",
  "frames/frame_0020.jpg"
]
```

Only images that participate in at least one match pair need to be listed. Image ordering
defines the index space used by `image_pairs/image_index_pairs`. The ordering is not
required to be sorted, but lexicographic ordering by name is recommended.

#### `images/feature_tool_hashes.{N}.uint128.zst`

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint128` (little-endian, 16 bytes per hash)
- XXH128 hash of the feature tool metadata, matching the value in the corresponding `.sift` file

#### `images/sift_content_hashes.{N}.uint128.zst`

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint128` (little-endian, 16 bytes per hash)
- XXH128 hash of the `.sift` file content, used to verify that the feature data the matches
  reference hasn't changed since matching was performed

#### `images/feature_counts.{N}.uint32.zst`

- **Shape**: `(N,)` where N = image_count
- **Data type**: `uint32` (little-endian)
- Number of features per image as used during matching. This may be less than the total
  feature count in the `.sift` file if `max_feature_count` was set. All `feat_idx` values
  in the match data MUST be less than the corresponding `feature_counts` entry.

#### `images/image_dims.{N}.2.uint32.zst`

- **Shape**: `(N, 2)` where N = image_count
- **Data type**: `uint32` (little-endian)
- Each row is `(width, height)` — the image's pixel dimensions, matching the
  `image_width` / `image_height` recorded in the corresponding `.sift` file's metadata
  (every writer has that metadata at hand: it already reads it for the hashes and
  feature counts)
- Makes the file self-contained for geometric consumers (principal-point priors,
  normalized coordinates, bounds checks) without opening the referenced `.sift` files
- **Constraint**: Every value MUST be ≥ 1
- Mandatory since format version 4; version ≤ 3 files never stored it

### 4. Pairs (Putative Matches — Backbone Alternative A)

Present exactly when `has_clusters` is false. Stores the pairwise correspondence
backbone: raw feature correspondences grouped per image pair.

#### `image_pairs/metadata.json.zst`

```json
{
  "image_pair_count": 332,
  "match_count": 145000
}
```

#### `image_pairs/image_index_pairs.{P}.2.uint32.zst`

- **Shape**: `(P, 2)` where P = image_pair_count
- **Data type**: `uint32` (little-endian)
- Each row is `(idx_i, idx_j)` where `idx_i < idx_j` (canonical ordering)
- Indices reference the image list in `images/names.json.zst`
- **Constraint**: MUST be sorted lexicographically by `(idx_i, idx_j)`

#### `image_pairs/match_counts.{P}.uint32.zst`

- **Shape**: `(P,)` where P = image_pair_count
- **Data type**: `uint32` (little-endian)
- Number of matches for each pair
- **Constraint**: `sum(match_counts) == match_count`
- **Constraint**: Every value must be >= 1 (pairs with zero matches are not stored)

#### `image_pairs/match_feature_indexes.{M}.2.uint32.zst`

- **Shape**: `(M, 2)` where M = match_count
- **Data type**: `uint32` (little-endian)
- Flat concatenation of all match pairs across all image pairs. Each row is
  `(feat_idx_i, feat_idx_j)` — feature index in image `idx_i` and feature index in
  image `idx_j` respectively.
- The first `match_counts[0]` rows belong to pair 0, the next `match_counts[1]` rows
  to pair 1, etc.
- **Constraint**: `feat_idx_i < feature_counts[idx_i]` and
  `feat_idx_j < feature_counts[idx_j]` for each match

#### `image_pairs/match_descriptor_distances.{M}.float32.zst`

- **Shape**: `(M,)` where M = match_count
- **Data type**: `float32` (little-endian)
- L2 descriptor distance for each match, aligned with `match_feature_indexes`
- Enables re-filtering by descriptor threshold without recomputing matches

### 5. Clusters (Backbone Alternative B)

Present exactly when `has_clusters` is true. Stores the cluster matcher's primary
artifact **in place of** the `image_pairs/` section: groups of SIFT features across
images that are likely co-observations of one surface point, in CSR layout. Cluster
`c` owns members `cluster_starts[c]..cluster_starts[c+1]` of the member-parallel
arrays, which name each member's image and feature index and state its geometry.

**Pairs are a derived view.** The pairwise view of a cluster file is its
expansion: every pair of members of one cluster that lie on different images,
grouped and sorted per the `image_pairs/` ordering rules. Because
`two_view_geometries/` arrays are keyed per stored pair, a cluster file cannot
carry TVGs directly; they belong in a separate pairwise `.matches` file
written from the expansion, with `image_pairs/` + `two_view_geometries/`. Each
file is still written once. Pair descriptor distances, which the stored
pairwise form carries, are recomputed from the referenced `.sift` files when a
consumer needs them.

A verifier that needs two-view geometries for a cluster file reads it,
expands its clusters into pairs, verifies them, and writes the result as that
pairwise file. See
[`specs/core/patch/cluster-patches.md`](../core/patch/cluster-patches.md) for the design
rationale.

#### `clusters/metadata.json.zst`

```json
{
  "cluster_count": 5200,
  "member_count": 14100,
  "matcher_options": {
    "d": 8,
    "alpha": 1.2,
    "min_size": 2,
    "preset": "default"
  }
}
```

**Field descriptions:**
- `cluster_count`: Must equal the top-level `cluster_count`
- `member_count`: Must equal the top-level `cluster_member_count`
- `matcher_options`: The cluster matcher's parameters (tool-specific key-value pairs)

#### `clusters/cluster_starts.{C+1}.uint32.zst`

- **Shape**: `(C+1,)` where C = cluster_count
- **Data type**: `uint32` (little-endian)
- CSR offsets into the member arrays: cluster `c` owns members
  `cluster_starts[c]..cluster_starts[c+1]`
- **Constraint**: `cluster_starts[0] == 0`, non-decreasing, final value equals the
  member count `K`
- **Constraint**: Every cluster has ≥ 2 members

#### `clusters/member_images.{K}.uint32.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `uint32` (little-endian)
- Index into `images/names.json.zst` per member. A cluster may contain several
  members from the same image (ambiguous detections); enrichment stages resolve
  the ambiguity (see `cluster_patches/`)
- **Constraint**: `member_images[k] < image_count`

#### `clusters/member_features.{K}.uint32.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `uint32` (little-endian)
- Feature index in that image's `.sift` file per member
- **Constraint**: `member_features[k] < feature_counts[member_images[k]]`

#### Member geometry: one position, one shape, and the file's stage

A cluster file holds exactly **one** keypoint position and **one** affine
shape per member, in the two arrays below. They are mandatory: every version 6
cluster file carries them, and a cluster file below version 6 is refused (see
[Versioning and Migration](#versioning-and-migration)). What they *mean* is the
stage the file is at, and the presence of `cluster_patches/` tells the two
apart:

- A file **without** `cluster_patches/` is at the detection stage: the
  arrays hold the detector's own values, copied verbatim — the same `float32`
  bits, gathered rather than converted — from row `member_features[k]` of the
  `.sift` `features/positions_xy` and `features/affine_shapes` arrays of image
  `member_images[k]`. See the [.sift file format](sift-file-format.md).
- A file **with** `cluster_patches/` is at the refinement stage: for every
  member the refinement **measured** (status `reference`, `kept`,
  `rejected_low_zncc`, `rejected_shift`, `rejected_unlocalizable_refined` or
  `rejected_unlocalizable_cells`, below) the arrays hold
  its answer, and for every member it never fitted they hold the detection the
  input carried, untouched. The refinement writes a **new** file (the
  write-once workflow), so the detection-stage file it read is kept beside
  it.

A matcher writes detection-stage files; a refiner writes refinement-stage
files. A writer may also cluster and refine in one step and write only the
refinement-stage file.

**No value is ever `NaN`, and `member_status` is the sole authority.** Every
row holds a real position and a real shape, so a consumer that only wants
geometry needs no join; a consumer that wants to know which reading a row
carries, or which members the vet admitted, reads
`cluster_patches/member_status`. The members whose rows the cascade measured
are those with status `reference`, `kept`, `rejected_low_zncc`,
`rejected_shift`, `rejected_unlocalizable_refined` or
`rejected_unlocalizable_cells` — the four rejected ones keep their measurement
so a consumer can re-gate without re-running. `duplicate_image`, `not_evaluated` and
`rejected_unlocalizable` were never fitted, and their rows are the detections.

The refinement's geometry lives **only** here: `cluster_patches/` carries the
vetting evidence — the reference structure, the statuses, the ZNCC, the shift
and the consistency residual — and no geometry of its own, so one file never
holds two answers for one member.

**Why `float32`.** The refinement's fit precision is on the order of 0.01 px,
while a `float32` position quantizes to 1.5e-5..6.1e-5 px on real captures,
more than a hundred times finer. A refiner may compute in `float64`; only the
write rounds. Detected values are never rounded at all — they are
`float32` in the `.sift` file and are copied bit-for-bit.

#### `clusters/member_positions.{K}.2.float32.zst`

- **Shape**: `(K, 2)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Each row is the member's `(x, y)` keypoint position in source-image pixels
  (COLMAP pixel convention, pixel centers at `+0.5`), at this file's stage
- Detected values are **bit-for-bit** the referenced `.sift` row: a writer
  gathers them, it does not round-trip them through another dtype
- A refined position is the same quantity moved: `p = detected + t` for the
  refinement's translation, whose magnitude is
  `cluster_patches/member_shift_px`. A cluster's reference member is refined
  against itself, so its refined position **is** its detected one
- **Constraint**: Never `NaN`; mandatory in every version 6 cluster file,
  alongside `member_affine_shapes`

#### `clusters/member_affine_shapes.{K}.2.2.float32.zst`

- **Shape**: `(K, 2, 2)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Each member's affine shape: the map from the detector's canonical unit frame
  onto that member's image pixels, so the member's image-space extent is the
  matrix's column norms and its radius is `sqrt(|det|)`
- At the detection stage this is the `.sift` `affine_shapes` row verbatim. A
  refined shape is `S = W · S_ref`, the refined reference→member warp `W`
  composed onto the reference member's detected shape `S_ref` — the shape the
  refinement actually sampled with. A cluster's reference member is refined
  against itself, so its refined shape **is** its detected shape, which is why
  the reference→member warp is recoverable as `W = S · S_ref⁻¹` through that
  member's row
- Because the composition runs through `S_ref`, a refined shape is a shape in
  the cluster's shared reference frame, not an independently re-measured
  ellipse for that one member
- **Constraint**: Never `NaN`; mandatory in every version 6 cluster file,
  alongside `member_positions`
- **Constraint**: A cluster's reference member's shape is non-singular — every
  consumer that wants a reference-relative warp inverts it

### 6. Cluster Patches (Optional Section)

Written by the cluster-patches operation into a **new** file that carries the source
file's images and clusters sections over (write-once workflow, same as adding TVGs).
Requires the cluster backbone. Arrays parallel the clusters' member arrays: for each
cluster, which member is its photometric reference, what became of every other, and
the signals behind those verdicts.

**The refinement's geometry is not here.** It is in the backbone's
`clusters/member_positions` and `clusters/member_affine_shapes`, which a
cluster file carries at every stage (see
[Member geometry](#member-geometry-one-position-one-shape-and-the-files-stage)),
so a consumer reads a member's position and extent from one place regardless of
the file it was handed. This section is what says which of those rows the
refinement measured and which members stand.

#### `cluster_patches/metadata.json.zst`

```json
{
  "cluster_count": 5200,
  "member_count": 14100,
  "member_status_names": [
    "reference", "kept", "rejected_low_zncc", "rejected_shift",
    "duplicate_image", "not_evaluated", "rejected_unlocalizable",
    "rejected_unlocalizable_refined", "rejected_unlocalizable_cells"
  ],
  "member_cell_status_names": [
    "fitted", "refused_curvature", "refused_zncc", "not_attempted",
    "refused_bound", "refused_outlier"
  ],
  "refine_options": {
    "patch_size": 8.0,
    "resolution": 15,
    "min_zncc": 0.85,
    "max_shift_px": 3.0
  }
}
```

**Field descriptions:**
- `cluster_count` / `member_count`: Must equal the top-level `cluster_count` /
  `cluster_member_count` (and therefore the clusters section counts)
- `member_status_names`: (version 7+) The legend `member_status` indexes, one
  name per code in code order. Always present in a version 7+ file; absent in a
  version 6 file, which is read through the canonical legend (see
  [`member_status`](#cluster_patchesmember_statuskuint8zst)). A writer always
  states the whole list in the canonical order; a reader accepts any legend and
  normalises the column onto that order. A version 7 to 9 file's legend never
  names `rejected_unlocalizable_refined` or `rejected_unlocalizable_cells`,
  which version 10 added
- `member_cell_status_names`: (version 8+, optional) The legend
  `member_cell_status` indexes, one name per code in code order. Present
  exactly when the section carries the
  [per-cell entries](#per-cell-entries-optional-version-8), and absent
  otherwise. A writer states the whole list in the canonical order; a reader
  accepts any legend and normalises the column onto that order. A version 8
  file's legend never names `refused_outlier`, which version 9 added
- `refine_options`: The refinement parameters used, present since the section
  was introduced in version 3. The patch extent appears
  under one of two keys: `patch_size` (the full
  patch edge in pixels, current) or the legacy `radius` (a half-width).
  A consumer that needs the half-width uses `patch_size / 2`, or `radius`
  as-is. The other keys record the settings for a reader to see and are not
  read back: current files also carry `resolution`, `min_zncc`, `max_shift_px`,
  `max_member_zncc_self_similarity_radius` (older files carry
  `max_keypoint_uncertainty`, the bar of an earlier member gate, in its place),
  `regate_at_refined_shape` and `max_capped_cells`, the settings of the two
  gates read at the refined shape (absent from a file written before them;
  `regate_at_refined_shape` is false whenever the whole-patch gate did not
  run, including when `max_member_zncc_self_similarity_radius` is `0`, the bar
  that gate shares),
  and `piecewise`, whether the per-cell refinement ran. When it ran, the
  piecewise refinement's settings sit beside it as flat keys: `move_shape`
  (whether the refinement was allowed to change the member's shape),
  `cell_shift_bound_px`, `min_cell_zncc`, `min_cell_curvature`,
  `update_tolerance_px` and `max_iterations`. A file written before
  `move_shape` was recorded lacks the key; its refinement could change the
  shape

  Which keys `refine_options` holds is not tied to the format version, since
  the object is a record of settings rather than a stored layout. `radius` was
  written only into version 3 and 4 files, which are refused because they are
  cluster files below version 6, so every file a reader accepts carries
  `patch_size`; a reader still takes `radius` as a half-width when it finds
  it. The member-gate key changed during version 6:
  `max_keypoint_uncertainty` is in files refined before that change and
  `max_member_zncc_self_similarity_radius` in files refined after it. A file
  rewritten at a later version, or a selection of it, keeps the keys of the
  refinement it holds, so a version 7 or later file can carry either. A reader treats
  a missing `refine_options`, or one with neither extent key, as recording no
  patch extent.

#### `cluster_patches/reference_members.{C}.uint32.zst`

- **Shape**: `(C,)` where C = cluster_count
- **Data type**: `uint32` (little-endian)
- Global member index of each cluster's reference member; `0xFFFFFFFF`
  when no reference member is present — the cluster could not be refined (no
  usable reference). Only in a derived file (one carrying the
  `matching_options["cluster_selection"]` provenance record) can the sentinel
  also mean the reference member fell outside the selection; see
  [Cluster Selection](#cluster-selection-derived-files) for the scoping
- **Constraint**: When not `0xFFFFFFFF`, `reference_members[c]` lies in cluster `c`'s
  member range and that member's status is `reference`

#### `cluster_patches/member_status.{K}.uint8.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `uint8`
- **Format**: an index into `cluster_patches/metadata.json`'s
  `member_status_names`, which is the file's own legend for this column. A code
  past the end of that list is invalid; nothing else about the numbering is
  fixed by this format, so a reader resolves every code through the list the
  file carries.
- **Names**: these are the only names this format defines, and each says what
  became of the member:
  - `reference` — the cluster's reference member. It is not refined: its
    geometry is its detection, so its reference→member warp is the identity,
    its `member_zncc` is 1.0 and its `member_shift_px` is 0
  - `kept` — refined and vetted successfully
  - `rejected_low_zncc` — achieved ZNCC below the acceptance threshold
  - `rejected_shift` — translation drifted too far from the SIFT seed
  - `duplicate_image` — outscored by another kept member in the same image, or
    shares the reference's image
  - `not_evaluated` — degenerate shape, template/seed support out of frame, or the
    cluster itself was unrefinable
  - `rejected_unlocalizable` — the member's own patch does not pin a position,
    so it was excluded before reference selection and refinement. The bar is
    `refine_options.max_member_zncc_self_similarity_radius`, in pixels of
    the patch's sampling grid (`refine_options.resolution` samples on a
    side): the member's patch, sampled at its detected position and affine
    shape, is shifted over itself by whole pixels up to 3 px in each
    direction, and its ZNCC self-similarity radius is how far it can move
    while still matching itself as well as a true match between two views
    would (the semi-major axis of the ellipse fitted to those shifts). A member
    whose radius is above the bar gets this status; a flat patch or a straight
    edge reads the maximum, 3. A bar of `0` means the gate was off. Files that
    carry `max_keypoint_uncertainty` in place of that key hold members refused
    by an earlier score of the same patch with the same status
  - `rejected_unlocalizable_refined` — (version 10+) the member passed the ZNCC
    and shift gates, but its own patch, sampled again at its refined position
    and affine shape (the ones the file stores for it), does not pin a
    position: its ZNCC self-similarity radius there, read as for
    `rejected_unlocalizable`, is above the same bar,
    `refine_options.max_member_zncc_self_similarity_radius`. Its measurement is
    kept
  - `rejected_unlocalizable_cells` — (version 10+) the member passed the ZNCC
    and shift gates, but of the nine cells of a three-by-three split of its own
    patch at its refined position and affine shape (rows and columns cut at a
    third and two thirds of the side), more than
    `refine_options.max_capped_cells` read the largest ZNCC self-similarity
    radius, 3, each cell read as a template against the rest of the patch.
    Its measurement is kept. A member that fails this and the previous rule is
    stored as `rejected_unlocalizable_refined`
- **Canonical order**: a writer always states the whole legend in the order
  listed above, so a conforming writer stores `0` reference, `1` kept, `2`
  rejected_low_zncc, `3` rejected_shift, `4` duplicate_image, `5`
  not_evaluated, `6` rejected_unlocalizable, `7`
  rejected_unlocalizable_refined, `8` rejected_unlocalizable_cells. A reader
  accepts any legend, in any
  order and naming any subset of the defined names, and **normalises the column
  onto the canonical order as it loads**, so a file's own numbering stops at the
  I/O boundary. A version 6 file carries no legend: its codes are the fixed
  numbering `0` reference through `6` rejected_unlocalizable, the first seven
  names of the canonical legend, so it is read through those.
- A patch cluster = the reference plus its `kept` members; statuses preserve the
  rejected members so consumers can re-gate without re-running (the ZNCC/shift arrays
  are the signals, mirroring how `match_descriptor_distances` enables descriptor
  re-filtering)
- **Constraint**: The legend is present (version 7+), is a non-empty list of
  names, names only statuses this format defines, and names none twice — a
  repeat would give one status two codes. A file below version 10 names
  neither `rejected_unlocalizable_refined` nor `rejected_unlocalizable_cells`
- **Constraint**: Every value is below the legend's length, and so names one of
  its entries
- **Constraint**: At most one member with status `reference` or `kept` per
  (cluster, image)
- **Integrity**: the legend lives in `cluster_patches/metadata.json`, which is
  hashed into `cluster_patches_xxh128` with the section's other files, so it
  needs no hash slot of its own: an edited legend changes the section digest
  exactly as an edited column does

#### `cluster_patches/member_zncc.{K}.float32.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Achieved windowed ZNCC vs the reference template, aligned with the member arrays;
  `NaN` where not evaluated

#### `cluster_patches/member_shift_px.{K}.float32.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Translation drift in pixels from the SIFT seed; `NaN` where not evaluated

#### `cluster_patches/member_consistency_residual.{K}.float32.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Warp-consistency residual: how far the member's warp is from a joint
  weak-perspective factorization of the fitted warps of all clusters. The
  factorization models each image `k` as a scaled-orthographic camera, a
  2×3 matrix `M_k`, and each cluster `c` as a planar patch with a 3×2
  tangent frame `T_c`, fitted by least squares over all clusters at once.
  For a member of cluster `c` on image `k`, `J` is its measured
  reference→member warp `S · S_ref⁻¹` (a 2×2 matrix; the identity for the
  reference member), and the residual is the relative misfit
  `‖M_k·T_c − J‖_F / ‖J‖_F`. See
  [`cluster-warp-consistency.md`](../core/patch/cluster-warp-consistency.md)
  for the fit.
  Lower = more consistent, 0 = perfect; `NaN` where the member did not
  enter the fit (non-reference/kept status, degenerate warp, or a cluster
  with fewer than 2 fitted members)
- A **stored signal, not a gate** — consumers choose their own threshold
  (e.g. ~0.3 as a RANSAC prefilter, ~0.1 for purity-first harvesting),
  mirroring how `member_zncc` enables re-vetting without re-running

The section records one patch extent and one sampling `resolution` for the
whole file. A second, finer resolution is proposed in
[two-tier-patch-density.md](../drafts/two-tier-patch-density.md).

#### Per-cell entries (optional, version 8+)

A member's affine shape describes its whole patch. The four entries below
describe the parts of the patch: the reference's `R × R` sampling grid
(`R` = `refine_options.resolution`) is cut into a three-by-three split of
**cells**, rows and columns cut at `⌊R/3⌋` and `R − ⌊R/3⌋`, and each cell of
the reference is registered separately against the member's image, as seen
through the member's affine shape. A cell's displacement is where its content
lies in the member's patch relative to where the member's stored shape places
it. No affine map fitted to the cells is removed from the displacements, so
they carry three things: the part one affine map over the nine cells can
express, which is how far the cells' own best fits disagree with the shape
fitted to the whole patch (in full only when the refinement left the shape
alone; see
[`member_cell_shift_px`](#cluster_patchesmember_cell_shift_pxk332float32zst));
for a planar surface, the perspective term a
surface normal is derived from once camera poses are known, the part no
affine map matches; and for a patch that spans two surfaces, the parallax of
the cells off the one the shape follows. A consumer that wants only the part
no affine map matches fits an affine map to the member's `fitted` cells'
displacements and removes it.

The four entries are present together or absent together, and present exactly
when `cluster_patches/metadata.json` carries `member_cell_status_names`. A
file without them carries no per-cell reading; a version 7 file never does.
Cells are indexed `[k, row, col]`, from the top-left cell of the grid in the
orientation the grid is sampled in (columns along the grid's first axis, rows
along its second). Only a member whose status is `kept` carries readings:
every other member's row is `NaN` displacements, `NaN` ZNCCs,
`not_attempted` throughout and `0` passes.

##### `cluster_patches/member_cell_shift_px.{K}.3.3.2.float32.zst`

- **Shape**: `(K, 3, 3, 2)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Each cell's displacement `[x, y]`, along the grid's columns and rows, from
  where the member's stored affine shape (`clusters/member_affine_shapes`,
  with its position `clusters/member_positions`) places the cell's centre, in
  pixels of the sampling grid. A displacement of `d` means the cell's content
  lies at `c + d` of the member's grid, where `c` is the cell's centre. It is
  the measured displacement, with no affine map fitted to the cells removed
- The part of the displacements one affine map over the cells can express is
  present in full only when `refine_options.move_shape` is `false`: the
  refinement then sampled the image once, through the stored shape, and left
  that shape as the whole-patch fit found it. When `move_shape` is `true`, or
  the key is absent (every version 8 file, and a version 9 file written before
  the key was recorded), the refinement may have moved the shape by the affine
  map the cells agreed on, so the stored shape may have absorbed that part and
  the displacements hold what remained. The definition above holds either
  way. When the refinement did move the shape, the cell's ZNCC and status were
  read at the sampling made before the last update that moved it, not through
  the stored shape
- `NaN` where no displacement was measured: a cell whose status is
  `refused_curvature`, `refused_bound` or `not_attempted`. A `fitted`,
  `refused_zncc` or `refused_outlier` cell carries its displacement
- The reference member's own cells would displace by zero by construction, so
  it is not attempted and carries none

##### `cluster_patches/member_cell_zncc.{K}.3.3.float32.zst`

- **Shape**: `(K, 3, 3)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Each cell's ZNCC against the reference's cell at its best displacement,
  every sample of the cell weighted equally and averaged over the template's
  textured colour channels; `NaN` where nothing was read
- Read through the stored shape when `refine_options.move_shape` is `false`;
  otherwise possibly through the shape before the last update, as
  [`member_cell_shift_px`](#cluster_patchesmember_cell_shift_pxk332float32zst)
  describes. The same holds for `member_cell_status`

##### `cluster_patches/member_cell_status.{K}.3.3.uint8.zst`

- **Shape**: `(K, 3, 3)` where K = cluster_member_count
- **Data type**: `uint8`
- **Format**: an index into `cluster_patches/metadata.json`'s
  `member_cell_status_names`, the file's own legend for this column, read the
  same way `member_status` reads its legend: a code past the end of the list
  is invalid, and nothing else about the numbering is fixed by this format
- **Names**: the only names this format defines, each saying what became of
  the cell:
  - `fitted` — the displacement was measured, and it agrees with the affine
    map that best explains the displacements of the member's fitted cells
    together (a refinement allowed to change the shape may also have moved
    the shape by that map)
  - `refused_curvature` — the cell does not pin a displacement: the reference
    is flat over it, or its ZNCC over the displacements searched is too flat
    at the best one
  - `refused_zncc` — its ZNCC at the best displacement is below the bar, so it
    lies over a different surface in this view; its displacement is measured
    but the affine map was not fitted to it
  - `not_attempted` — the cell was not registered, or its registration was
    not used: the member is not `kept`, a sample the search needs lies outside
    the image, no cell of the member survived, the image could not be sampled
    through the shape, the fitted affine map was not finite or reflected the
    patch, or, in a refinement allowed to change the shape, the refined shape
    could not be accepted — the whole-patch ZNCC or the shift from the seed
    read again at it failed the bars the member was kept on, or its support
    left the image. In every case but the first, every cell of the member is
    `not_attempted` and the member keeps the shape its whole-patch fit found
  - `refused_bound` — the best displacement lies on the edge of the range
    searched, so the optimum is at or past it and no sub-pixel displacement
    can be read
  - `refused_outlier` — (version 9+) the displacement was measured and the
    cell passed the bars above, but it disagrees with the affine map the
    member's other fitted cells agree on by so much that the fit of that map
    gave it no weight
- **Canonical order**: a writer always states the whole legend in the order
  listed above, `0` fitted, `1` refused_curvature, `2` refused_zncc, `3`
  not_attempted, `4` refused_bound, `5` refused_outlier. A reader accepts any
  legend, in any order and naming any subset of the defined names, and
  normalises the column onto the canonical order as it loads
- **Constraint**: The legend is a non-empty list of names, names only cell
  statuses this format defines, and names none twice. A version 8 file's
  legend does not name `refused_outlier`
- **Constraint**: Every value is below the legend's length

##### `cluster_patches/member_cell_iterations.{K}.uint8.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `uint8`
- How many times the member's image was sampled through a shape of the
  member; `0` for a member the refinement did not run on. A refinement that
  does not change the shape (`refine_options.move_shape` false) samples once.
  One that may change it samples once per pass, until the shape stops
  changing by more than the refinement's tolerance, a change is refused, or
  the refinement's cap on passes is reached; a member that reached the cap is
  one whose shape had not settled, and its displacements are less
  trustworthy. A member whose cells are all `not_attempted` after a failed
  pass counts the passes made, including the one that failed

**Integrity.** The four entries are files of the `cluster_patches/` section,
hashed into `cluster_patches_xxh128` in the section's lexicographic order (they
sort before `member_consistency_residual`), and the legend rides inside
`metadata.json`, which the same digest covers.

### 7. Two-View Geometries (Optional Section)

The two-view geometries section stores the results of geometric verification. It is optional —
a `.matches` file can contain only candidate matches. It requires the pairwise backbone
(`image_pairs/`): its arrays are keyed per stored pair, so a cluster-bearing file cannot
carry TVGs. To add geometric verification results,
write a new `.matches` file that includes both the candidate matches and the TVGs (see
[Writing a verified .matches file](#writing-a-verified-matches-file)). This section
parallels the `two_view_geometries` table in a COLMAP database.

When present, every pair in `image_pairs/image_index_pairs` has a corresponding entry in the
two-view geometry arrays (same count `P`, same ordering). Pairs where geometric verification
failed have config `"undefined"` or `"degenerate"` and `inlier_counts = 0`.

#### `two_view_geometries/metadata.json.zst`

```json
{
  "image_pair_count": 332,
  "inlier_count": 98000,
  "verification_tool": "colmap",
  "verification_options": {
    "min_num_inliers": 15,
    "max_error": 4.0
  }
}
```

**Field descriptions:**
- `image_pair_count`: Must equal `image_pairs/metadata.json.zst` image_pair_count
- `inlier_count`: Total inlier matches across all pairs
- `verification_tool`: Tool used for geometric verification (e.g., `"colmap"` via
  `pycolmap.verify_matches`)
- `verification_options`: Tool-specific verification parameters

#### `two_view_geometries/config_types.json.zst`

JSON array of unique configuration type strings that appear in this file. The array defines
the index space used by `config_indexes`. This avoids a hardcoded integer-to-name mapping
while keeping the per-pair data compact for large pair counts.

Valid values (corresponding to COLMAP's TwoViewGeometryConfig):
- `"undefined"` — verification not run or inconclusive
- `"degenerate"` — degenerate configuration (too few inliers, etc.)
- `"calibrated"` — calibrated pair (essential matrix estimated)
- `"uncalibrated"` — uncalibrated pair (fundamental matrix estimated)
- `"planar"` — planar scene (homography estimated)
- `"planar_or_panoramic"` — planar or panoramic
- `"panoramic"` — pure rotation (panoramic)
- `"multiple"` — multiple model types
- `"watermark_clean"` — clean of watermarks
- `"watermark_bad"` — watermark detected

Example:
```json
["calibrated", "degenerate", "planar"]
```

#### `two_view_geometries/config_indexes.{P}.uint8.zst`

- **Shape**: `(P,)` where P = pair_count
- **Data type**: `uint8` (since there are fewer than 256 config types)
- Index into `config_types.json.zst` for each pair
- **Constraint**: Every value must be a valid index into the `config_types` array

#### `two_view_geometries/inlier_counts.{P}.uint32.zst`

- **Shape**: `(P,)` where P = pair_count
- **Data type**: `uint32` (little-endian)
- Number of geometrically verified inlier matches per pair
- **Constraint**: `sum(inlier_counts) == inlier_count`
- May be 0 for image pairs where verification failed

#### `two_view_geometries/inlier_feature_indexes.{I}.2.uint32.zst`

- **Shape**: `(I, 2)` where I = inlier_count
- **Data type**: `uint32` (little-endian)
- Flat concatenation of inlier match pairs. Same layout as `image_pairs/match_feature_indexes`:
  each row is `(feat_idx_i, feat_idx_j)`, with the first `inlier_counts[0]` rows belonging
  to pair 0, etc.
- Inlier matches MUST be a subset of the candidate matches for each pair.
  Implementations SHOULD validate this constraint when writing.

#### `two_view_geometries/f_matrices.{P}.3.3.float64.zst`

- **Shape**: `(P, 3, 3)` where P = pair_count
- **Data type**: `float64` (little-endian)
- Fundamental matrix per pair, row-major 3x3
- All zeros when not applicable (e.g., when config is Degenerate or Undefined)

#### `two_view_geometries/e_matrices.{P}.3.3.float64.zst`

- **Shape**: `(P, 3, 3)` where P = pair_count
- **Data type**: `float64` (little-endian)
- Essential matrix per pair, row-major 3x3
- All zeros when not applicable

#### `two_view_geometries/h_matrices.{P}.3.3.float64.zst`

- **Shape**: `(P, 3, 3)` where P = pair_count
- **Data type**: `float64` (little-endian)
- Homography matrix per pair, row-major 3x3
- All zeros when not applicable

#### `two_view_geometries/quaternions_wxyz.{P}.4.float64.zst`

- **Shape**: `(P, 4)` where P = pair_count
- **Data type**: `float64` (little-endian)
- Relative rotation quaternion in WXYZ format — the rotation part of the
  `cam2_from_cam1` pose, in the canonical camera convention (see
  [Coordinate Conventions](#coordinate-conventions))
- `[1, 0, 0, 0]` (identity) when not applicable

#### `two_view_geometries/translations_xyz.{P}.3.float64.zst`

- **Shape**: `(P, 3)` where P = pair_count
- **Data type**: `float64` (little-endian)
- Relative translation vector — the translation part of the `cam2_from_cam1`
  pose, in the canonical camera convention (see
  [Coordinate Conventions](#coordinate-conventions))
- `[0, 0, 0]` when not applicable

## File Naming and Path Convention

`.matches` files follow the same convention as `.sfmr` files: they can be placed anywhere
within the workspace, and they embed a workspace reference for relocatability. When a command
produces a `.matches` file without an explicit output path, it writes to the `matches/`
directory at the workspace root.

When the file includes two-view geometries, the default output directory is `tvg-matches/`.
These are default conventions, not requirements — commands that consume `.matches` files
locate the workspace through the embedded workspace reference, not by assuming a fixed
location.

Example workspace layout:

```
my_project/
├── .sfm-workspace.json
├── frames/
│   └── ...
├── sfmr/
│   ├── 20260329-00-frames_1-83.sfmr
│   └── ...
├── matches/
│   ├── 20260329-00-exhaustive_1-50.matches
│   ├── 20260329-02-sequential_1-931.matches
│   └── ...
└── tvg-matches/
    ├── 20260329-01-exhaustive_1-50-verified.matches
    └── ...
```

Unlike `.sift` files (which are derived from a single image and stored relative to it),
`.matches` files don't correspond to a single entity — they cover an arbitrary subset of
images with an arbitrary choice of image pairs and matching strategy. This makes them more like
`.sfmr` files: self-contained snapshots that embed their own context, named in whatever way
is meaningful to the user.

## Data Ordering and Constraints

### Backbone Rule

Exactly one of `image_pairs/` and `clusters/` is present, selected by the metadata
`has_clusters` flag. `two_view_geometries/` requires `image_pairs/`;
`cluster_patches/` requires `clusters/`. The backbone-specific summary counts
(`image_pair_count`/`match_count` vs `cluster_count`/`cluster_member_count`) follow
the backbone — a file never carries both sets.

### Ordering Requirements

1. **Pairs sorted**: `image_index_pairs` MUST be sorted lexicographically by `(idx_i, idx_j)`
   with `idx_i < idx_j`
2. **Match counts aligned**: `match_counts[k]` = number of entries in the match arrays
   belonging to pair `k`. `sum(match_counts) == match_count`
3. **Inlier counts aligned**: Same relationship for the TVG inlier arrays
4. **Feature index bounds**: All feature indexes must be less than the corresponding
   `feature_counts` entry (pairwise match arrays and cluster `member_features` alike)
5. **Image dims positive**: Every `image_dims` value is ≥ 1

### Cluster Constraints

1. **CSR well-formed**: `cluster_starts[0] == 0`, non-decreasing, final value equals
   the member count
2. **Minimum size**: Every cluster has ≥ 2 members
3. **Member bounds**: `member_images[k] < image_count` and
   `member_features[k] < feature_counts[member_images[k]]`
4. **Member geometry present**: `member_positions` `(K, 2)` and
   `member_affine_shapes` `(K, 2, 2)` are both present, member-parallel to the
   arrays above, and free of `NaN`
5. **Cluster patches parallel**: The `cluster_patches/` arrays have lengths `C`
   (`reference_members`) and `K` (member arrays) matching the clusters section
6. **Statuses valid**: The `member_status_names` legend is well formed, every
   `member_status` value is below its length, and the rules below read each code
   through it; `reference_members[c]` is `0xFFFFFFFF` or lies in cluster `c`'s
   member range with status `reference`; at most one `reference`-or-`kept`
   member per (cluster, image)
7. **Reference shapes non-singular**: For every refinable cluster, the reference
   member's `member_affine_shapes` entry is non-singular — its value is that
   feature's own detector affine shape `S_ref`, which every reference-relative
   warp is recovered through
8. **Per-cell entries paired with their legend**: The four
   `cluster_patches/member_cell_*` entries are present exactly when
   `member_cell_status_names` is, each sized by `K` as its name states; every
   `member_cell_status` value is below the legend's length; and a member
   whose status is not `kept` has every cell `not_attempted`, every
   displacement and ZNCC `NaN`, and `0` passes

### No required ordering within a pair

Matches within a single pair (the slice of `match_feature_indexes` for that pair) have no
required ordering. They are an unordered set of correspondences.

### Index Relationships

- `image_index_pairs[k]` → `(idx_i, idx_j)` into `images/names.json.zst`
- `match_feature_indexes[m][0]` → feature index in `.sift` file for image `idx_i`
- `match_feature_indexes[m][1]` → feature index in `.sift` file for image `idx_j`
- `sift_content_hashes[i]` → verifies the `.sift` file hasn't changed

## Cluster Selection (Derived Files)

A cluster-backbone file may be a **derived file**: the result of selecting a
subset of another cluster-backbone file's clusters and members. A derived
file is an ordinary `.matches` file — every constraint in this specification
applies unchanged, and it reads back through the ordinary reader. How the
subset is chosen is not this specification's concern; the selection
operation is specified in
[cluster-selection.md](../core/features/cluster-selection.md). What this
specification defines is the file-level contract a derived file carries.

**Provenance record.** A derived file is identified by a record in the
top-level metadata under `matching_options["cluster_selection"]`:

```json
{
  "cluster_selection": {
    "source_content_xxh128": "9a51...",
    "min_span": 2,
    "restrict_images": ["frames/frame_0010.jpg", "..."],
    "accepted_statuses": ["reference", "kept"]
  }
}
```

`source_content_xxh128` names the source file (its whole-file
`content_xxh128`). The remaining keys record the selection predicate:

- `min_span` — the least number of distinct selected images a cluster's kept
  members had to span for the cluster to be kept (at least 2)
- `restrict_images` — the image names the selection was restricted to, or
  `null` when it was not restricted by image
- `accepted_statuses` — the `member_status` names (`reference`, `kept`, …)
  whose members were kept; when the source has no `cluster_patches/`, every
  member was a candidate regardless
- `restrict_cluster_ids` — present only when the selection was restricted by
  cluster: the requested cluster ids of the source, sorted and without
  duplicates

[cluster-selection.md](../core/features/cluster-selection.md) defines how
the operation applies them. All other metadata — including the timestamp — is
inherited from the source; the derived file's content hashes are its own,
computed at write time. The source file is never modified.

When the source carries a `cluster_selection` record of its own (it is itself
a selection), the new record carries a `source_selection` key holding that
record. An unwritten selection has no `content_xxh128`, so the nesting is what
keeps the chain naming the archive it started from; nesting repeats to any
depth, and the innermost `source_content_xxh128` names that archive. The key is
absent whenever the source is an ordinary file.

**Versions.** The record lives in `matching_options`, so adding it and its
keys changed no stored layout and no format version. It was first written
into version 4 files, and `restrict_cluster_ids` and `source_selection` were
first written into version 5 files. Every cluster file a reader accepts
(version 6 and later) may carry the record, with or without either key. A
reader treats a file without the record as an ordinary file, a record without
`restrict_cluster_ids` as a selection not restricted by cluster, and a record
without `source_selection` as a selection of an ordinary file.

A selection of a selection, for example, records:

```json
{
  "cluster_selection": {
    "source_content_xxh128": "",
    "min_span": 2,
    "restrict_images": ["frames/frame_0010.jpg", "..."],
    "accepted_statuses": ["reference", "kept"],
    "source_selection": {
      "source_content_xxh128": "9a51...",
      "min_span": 2,
      "restrict_images": null,
      "accepted_statuses": ["reference", "kept"]
    }
  }
}
```

**Sentinel scoping.** Only in a file carrying the `cluster_selection`
provenance record may `reference_members[c] = 0xFFFFFFFF` additionally mean
"the cluster's reference member is not present in this selection": the
cluster's members then carry real statuses, absolute positions and warps
(still expressed relative to the absent reference patch). In a file without
the record the sentinel retains its single meaning — the cluster could not
be refined. Structural constraints are identical in both cases: the sentinel
is always permitted, and a non-sentinel entry must point at an in-range
member with status `reference`.

**Working view, not an archive.** A selection drops non-accepted members, so
the per-member evidence that enables re-gating (rejected statuses and their
measurements) is absent from a derived file. Consumers needing it return to
the source named by `source_content_xxh128`.

## Design Rationale

### Why are two-view geometries optional, not separate files?

Putative matches and geometric verification are distinct pipeline stages with different
dependencies:

- **Matches** depend on: image content, features, matching method/parameters
- **Two-view geometries** depend on: matches + camera intrinsics + verification parameters

Making TVGs an optional section within the `.matches` format (rather than a separate file type)
keeps things simple: one format, one reader, one writer. A `.matches` file is always
self-contained — if TVGs are present, the matches they refer to are right there in the same file.

The write-once workflow is:

1. Run matching → write a `.matches` file with candidate matches only.
2. Run geometric verification → write a **new** `.matches` file that includes both the
   original matches and the TVG results.

Since each file gets a different content hash (the metadata records `has_two_view_geometries`),
the matches-only and matches+TVG files coexist naturally. They can live in the same directory
or in separate directories (e.g., `matches/` and `tvg-matches/`) — the workspace example shows
the latter convention. To try different verification parameters, write another new file — the
candidate matches are cheap to copy, and you never touch the original.

This means you can:
1. Write matches immediately after the matching step, inspect them before deciding to verify
2. Produce multiple verified variants with different parameters, each as a new immutable file
3. Ship a `.matches` file without TVGs and let the consumer verify

### Why is the cluster backbone exclusive with stored pairs?

A cluster-bearing file stores clusters **instead of** the pairwise expansion. The
expansion is deterministic and cheap, while storing both
roughly doubles the correspondence payload with derived values: per-pair data grows as
Σ C(k,2) over cluster sizes versus the Σ k the clusters themselves cost. Consumers
that need pairs obtain them by calling the expansion at read time; the cluster file
remains the durable primary artifact, and geometric verification writes the
COLMAP-facing pairwise derivative as a new file. See
[`specs/core/patch/cluster-patches.md`](../core/patch/cluster-patches.md) for the full design
discussion.

### Why store descriptor distances?

The descriptor distance is a useful quality signal for matches. Storing it enables
re-filtering by threshold (e.g., tighten from 250 to 150) without reloading `.sift`
files and recomputing L2 distances. At 4 bytes per match, the cost is modest.

### Why store feature counts?

The `.sift` file may contain 23,000 features, but matching may have used only the first
8,192 (via `max_feature_count`). Storing the count used during matching serves two purposes:

1. **Validation**: Feature indexes in the match data must be within bounds
2. **Reproducibility**: Documents which subset of features was used

### Why not store features directly?

Descriptors live in `.sift` files. The `.matches` file references them by index and verifies
integrity via `sift_content_hashes`. This avoids duplication and keeps the `.matches` file
focused on correspondences.

The cluster backbone does store one thing per member beyond the index: its
keypoint position and affine shape. That pair is a hundredth of a descriptor's
size, and it is what nearly every downstream consumer of a cluster file
actually wants from the `.sift` files — a member's location and extent, not
its descriptor. Storing it turns a scattered read of every referenced `.sift`
file into a column read, and it lets the same arrays state the refinement's
answer once a cluster-patches pass has produced one, so consumers stop
branching on which stage they were handed. Descriptors stay out: the matcher
consumed them and nothing downstream re-matches from a `.matches` file.

### Why columnar storage for matches?

The flat concatenated layout with per-pair counts (same pattern as tracks in `.sfmr`) enables:
- Reading just pair metadata without loading match data
- Loading matches for a specific pair by computing the offset from cumulative counts
- Better compression (similar values together)

## Compression Details

Every entry of a `.matches` file is written at the same zstandard level. The
rest is the container's; see [archive-container.md](archive-container.md).

## Integrity Verification

The hashes are the ones described under
[Content Hash](#2-content-hash-content_hashjsonzst): a `metadata_xxh128`, one
digest per present section over its data files in lexicographic path order, and a
`content_xxh128` over those in the order metadata, images, pairs, clusters,
cluster_patches, two_view_geometries.

### Verification Process

1. Check backbone/flag consistency (exactly one backbone's entries and summary counts
   present, matching the `has_*` flags); a file that fails these is reported without
   further section checks
2. Decompress each file and hash the raw uncompressed bytes
3. Recompute section and overall hashes
4. Compare with stored values in `content_hash.json.zst`
5. Check every raw array a structural check indexes against the byte length its
   declared count calls for
6. Validate structural constraints (feature index bounds, count sums, pair ordering;
   cluster CSR, member bounds, patch statuses and reference invariants)

Step 6 walks the raw arrays by index, so a section's structural checks are skipped
once one of its arrays fails step 5 — and the pairwise feature-bounds walk is
skipped as well when the per-pair `match_counts` do not sum to `match_count`,
since those are the run lengths it advances through. The length or sum error is
what reports the file as invalid, and what says why the rest of the section's
findings are absent. Verification never trusts a declared count far enough to
index past the end of an array: a truncated, over-long or hand-edited file is
reported, not a crash.

## Implementations

The code that reads, writes and verifies `.matches` files is:

- Rust: `read_matches`, `read_matches_metadata`, `write_matches` and
  `verify_matches` in
  [`sfmtool-matches-format`](../../crates/sfmtool-matches-format/src/lib.rs).
- Python: the same four functions in `sfmtool.fileio`, which take and
  return a dict of NumPy arrays and metadata
  ([bindings](../../crates/sfmtool-py/src/fileio/matches.rs)), and
  `sfmtool.fileio.MatchesFile`, which opens a file for the cluster
  queries ([bindings](../../crates/sfmtool-py/src/fileio/matches_file.rs)).

The member statuses are the `ClusterMemberStatus` enum, whose discriminants
are the canonical codes and whose `NAMES` is the canonical
`member_status_names` legend, and the `0xFFFFFFFF` reference sentinel is
`CLUSTER_REFERENCE_UNREFINABLE`, both in
[`types.rs`](../../crates/sfmtool-matches-format/src/types.rs). The Rust and
Python readers hand back `member_status` in the canonical numbering, whatever
legend the file stated. The per-cell entries are
`ClusterPatchData::member_cells`, a `MemberCellData`, `None` when the file
carries none, and the cell statuses are the `ClusterCellStatus` enum, whose
`NAMES` is the canonical `member_cell_status_names` legend, both in
[`cells.rs`](../../crates/sfmtool-matches-format/src/cells.rs); readers hand
back the cell statuses in the canonical numbering too. In Python,
`read_matches` carries them as `member_cell_shift_px`, `member_cell_zncc`,
`member_cell_status` and `member_cell_iterations` when the file has them,
`write_matches` writes the four keys together, and `MatchesFile` exposes them
under the same names beside `member_cell_status_names` and
`has_member_cells`: each is `None` when the `cluster_patches/` section carries
no cells, and raises, like the section's other getters, when the file has no
`cluster_patches/` section.
`ClusterPatchData::refine_radius` (and `MatchesFile.refine_radius` in Python)
returns the patch half-width from either `refine_options` key. The expansion of
clusters into pairs is `clusters_to_pair_matches` in
[`cluster_match`](../../crates/sfmtool-core/src/features/cluster_match/mod.rs).

`verify_matches` returns `(is_valid, error_messages)`.

`write_matches` takes the zstandard level as its `zstd_level` argument, which
the Python binding defaults to 3. A full read consumes every entry the file
holds, so `read_matches` takes the container's whole-archive path — one pass
over the ZIP, then the frames expanded in parallel — and each section reader
looks its entries up by name in that batch. `verify_matches` and
`read_matches_metadata` do not: the verifier walks the archive in the order the
hashes were taken, and a metadata-only read wants one small entry.

The writers in this repository and the `matching_method` they record:

- `sfm match -e`, `sfm match -s` and `sfm match --flow` write pairwise files
  with `"exhaustive"`, `"sequential"` and `"flow"`.
- `sfm match --cluster` writes detection-stage cluster files with `"cluster"`.
- `sfm cluster-patches` reads a detection-stage file and writes its
  refinement-stage counterpart, keeping the source's `matching_method`. The
  SfM Explorer's cluster-patches build clusters and refines in one step and
  writes a refinement-stage file with `"cluster"`.
- `sfm match --merge` writes `"merged"` files.
- [`sfm match --derive-pairs`](../cli/image-feature/match-command.md#derive-pairs)
  is the verifier that reads a cluster file and writes its verified pairwise
  file with two-view geometries, keeping the source's `matching_method`.
- `sfm to-colmap-db` (in `src/sfmtool/colmap/db_setup.py`) is the consumer that
  writes stored two-view geometries into a COLMAP database; it conjugates each
  relative pose by `S` when it builds the `pycolmap.Rigid3d`.

The reader's refusal of a cluster file below version 6 names the migration as
commands: regenerate with `sfm match --cluster`, then re-run
`sfm cluster-patches` if the file was enriched.

## As part of a Pipeline

The `.matches` file fits between `.sift` files and `.sfmr` files. The data in a collection
of `.sift` and `.matches` files can be used to populate a COLMAP database to run its algorithms
for mapping, bundle adjustment, etc. The pipeline progresses by creating new files, never by
modifying an existing file.

```
  .sift files (per-image features)
        │
        ▼
   Flow / Descriptor Matching
        │
        ▼
  .matches file(s) (candidate matches, no two view geometries)
        │
        ▼
   Geometric Verification
        │
        ▼
  .matches file (matches + two view geometries)
        │
        ▼
   COLMAP Database (populated from either .matches variant)
        │
        ▼
   SfM Solver (COLMAP / GLOMAP)
        │
        ▼
   COLMAP Binary (sparse reconstruction)
        │
        ▼
   .sfmr file (reconstruction)
```

### Writing a verified .matches file

Geometric verification does not modify the file it reads. A verifier
reads a clusters-bearing file, expands its clusters into image pairs, verifies
those pairs, and writes a new pairwise file at a separate path. The new file holds
the pairs that pass verification, their matches and the
[two-view geometries section](#7-two-view-geometries-optional-section), with
`has_two_view_geometries` set to `true` in its metadata and its own content hash.

## Versioning and Migration

The format has ten released versions (`1` through `10`). The format is versioned
(`metadata.json` `version`) precisely so that changes like the ones below can upgrade
on load instead of breaking old files. Writers always emit the current version;
readers accept any version up to it, with one exception — a cluster-backbone
file below version 6, which is refused.

### Version 9 → Version 10

| Change | Detail |
|---|---|
| `cluster_patches/metadata.json` `member_status_names` | The legend may name two more member statuses, `rejected_unlocalizable_refined` and `rejected_unlocalizable_cells`, codes `7` and `8` in the canonical order. A writer states the whole legend, so every version 10 file with `cluster_patches/` names them. |
| `cluster_patches/metadata.json` `refine_options` | A file may record `regate_at_refined_shape` and `max_capped_cells`, the settings of the two gates those statuses come from. They are recorded settings like the others and are not read back. |
| `clusters/member_positions`, `clusters/member_affine_shapes` | A member with either new status was measured, so its rows hold the refinement's answer, as a `rejected_low_zncc` or `rejected_shift` member's do. |

No entry a version 9 file stores changes its definition or layout, and a
version 9 file reads unchanged. A version 9 file whose legend names either new
status is refused, since no version 9 writer wrote those names. The bump exists
because a version 9 reader refuses a legend name it does not define: without
it, a version 9 reader would meet the new names in a file that claims a version
it reads. Integrity verification follows the same rule. A re-written file is a
new version 10 file, with new hashes. Pairwise files have no
`cluster_patches/` section, so only their metadata `version` moves.

### Version 8 → Version 9

| Change | Detail |
|---|---|
| `cluster_patches/metadata.json` `member_cell_status_names` | The legend may name a sixth cell status, `refused_outlier`, code `5` in the canonical order. A writer states the whole legend, so every version 9 file with per-cell entries names it. |
| `cluster_patches/metadata.json` `refine_options` | A file with per-cell entries may record `move_shape`, whether the refinement was allowed to change the member's shape. It is a recorded setting like the others and is not read back. |

No entry a version 8 file stores changes its definition or layout, and a
version 8 file reads unchanged. A version 8 file never records `move_shape`,
so its cells are read as those of a refinement that may have moved the shape
(see
[`member_cell_shift_px`](#cluster_patchesmember_cell_shift_pxk332float32zst)).
A version 8 file whose legend names `refused_outlier` is refused, since no
version 8 writer wrote that name. The bump exists because
a version 8 reader refuses a legend name it does not define: without it, a
version 8 reader would meet `refused_outlier` in a file that claims a version
it reads. Integrity verification follows the same rule. A re-written file is a
new version 9 file, with new hashes.

### Version 7 → Version 8

| Change | Detail |
|---|---|
| `cluster_patches/member_cell_shift_px`, `member_cell_zncc`, `member_cell_status`, `member_cell_iterations` | Four new optional entries, present together or absent together: each member's per-cell displacement, ZNCC, status and refinement passes. See [Per-cell entries](#per-cell-entries-optional-version-8). They are files of the `cluster_patches/` section and are hashed into `cluster_patches_xxh128` with it. |
| `cluster_patches/metadata.json` `member_cell_status_names` | New key, present exactly when the per-cell entries are: the legend `member_cell_status` indexes. |

Nothing that a version 7 file stores changes meaning or layout. **A version 7
file reads unchanged** and has no per-cell entries, so a reader reports that it
carries no cells. A version 7 file that carries `member_cell_status_names` or
any `member_cell_*` entry is refused, since no version 7 writer wrote one; a
version 8 file with the entries but no legend, or the legend but not all four
entries, is refused, since it would leave the column unexplained or the cells
incomplete. Integrity verification follows the same rule. A re-written file is
a new version 8 file, with new hashes. Pairwise files have no
`cluster_patches/` section, so only their metadata `version` moves.

### Version 6 → Version 7

| Change | Detail |
|---|---|
| `cluster_patches/metadata.json` `member_status_names` | New key, present exactly when `cluster_patches/` is: the legend `cluster_patches/member_status` indexes. It rides inside `metadata.json`, which is already hashed into `cluster_patches_xxh128`, so no hash slot changes. |
| `cluster_patches/member_status` | A code is now an index into that legend rather than a number the format fixes. |

Version 7 makes `member_status` follow the pattern a per-element enumeration
takes in the `.sfmr` format (`point_constraints` and its
`point_constraint_names` legend): a numeric column, and a `*_names` legend in
the same section's metadata that every code is resolved through. A writer
states the canonical legend, so the column's bytes are the same as a version 6
writer's; only the legend and the metadata `version` are new.

**A version 6 file reads unchanged.** Every version 6 writer stored the fixed
numbering `0` reference through `6` rejected_unlocalizable, which is exactly the
canonical legend, so a version 6 `cluster_patches/` section is read through that
legend and its codes keep their meaning. A version 6 file that carries
`member_status_names` is refused, since no version 6 writer wrote one; a version
7 file without it is refused, since it would leave its codes unexplained.
Integrity verification follows the same rule. A re-written file is a new
version 7 file that states the legend, with new hashes. Pairwise files have no
`cluster_patches/` section, so only their metadata `version` moves.

### Version 5 → Version 6

Version 6 gives a cluster file **one** place for its members' geometry.
`clusters/` gains a mandatory `member_positions.{K}.2.float32` and
`member_affine_shapes.{K}.2.2.float32` pair, and
`cluster_patches/member_affines.{K}.2.3.float64` is **removed**. There is now
exactly one keypoint position and one affine shape per member, and its content
is the file's stage: the `.sift` detections, verbatim, in a matcher output; the
refinement's own answer for every member its cascade measured, and the
untouched detection for every member it never fitted, in a cluster-patches
output. Nothing is `NaN`; `cluster_patches/member_status` is the sole authority
on what a row means and on exclusion. See
[Member geometry](#member-geometry-one-position-one-shape-and-the-files-stage).

The change is what removes the duplication two earlier versions had grown: a
consumer no longer branches on whether the enrichment is present, no longer
opens the `.sift` files the feature indexes name, and can no longer be handed
two answers for one member. It also halves the geometry's storage, which the
`float32` note in that section explains. A pairwise file's entries are
unchanged — only its metadata `version` moves.

**A cluster-backbone file below version 6 is refused on read**, bare or
enriched. Its geometry is only in the referenced `.sift` files, which the
format layer does not open, and its `member_affines` carries semantics the
format no longer has, so there is nothing to upgrade from. The error names the
migration: regenerate the backbone by clustering the `.sift` features again,
then refine it again if the file was enriched. Pairwise-backbone
compatibility is untouched, and integrity verification stays version-aware —
it requires the geometry pair at version 6+, forbids it before, and continues
to pass structurally sound older files.

### Versions 3 → 4 → 5

Version 4 added the mandatory `images/image_dims.{N}.2.uint32` array and gave
`cluster_patches/member_affines`' last column the member's absolute refined
keypoint position; version 5 gave its leading 2×2 the member's absolute affine
shape. Both of those affine semantics are history: version 6 removed the member
they described, and every cluster file below version 6 is refused, so no reader
carries them.

What survives for a **pairwise** file is `image_dims`: version ≤ 3 pairwise
files never stored it and load with no dimensions (in-memory `image_dims` is
absent/None); everything else loads with unchanged semantics. Integrity
verification remains version-aware and continues to pass
structurally sound older files of every version, cluster ones included — hashes
cover the stored bytes.

### Version 2 → Version 3

Version 3 introduces the cluster backbone: the `clusters/` section (the cluster
matcher's primary artifact) and the optional `cluster_patches/` enrichment, with the
`image_pairs/` section — mandatory through version 2 — becoming the stored-pairs
alternative (exactly one of the two backbones is present per file). Metadata gains
`has_clusters` / `has_cluster_patches` flags and, in cluster-bearing files,
`cluster_count` / `cluster_member_count` in place of `image_pair_count` /
`match_count`; the content hash gains `clusters_xxh128` / `cluster_patches_xxh128`.

**Version ≤ 2 files load unchanged.** They always store the pairwise backbone and
never have clusters; readers treat the absent `has_clusters` / `has_cluster_patches`
flags as `false`. No stored byte changes meaning: a pairwise version 3 file has
exactly the pre-version-3 section layout and hash byte streams (the new metadata
flags appear only in newly written files). Version 1 files additionally get the
pose S-conjugation described below.

### Version 1 → Version 2

Version 2 makes the canonical camera convention normative for the stored two-view
relative poses (see [Coordinate Conventions](#coordinate-conventions)), mirroring the
`.sfmr` version 5 and `.camrig` version 2 bumps. No member is added, removed, or
renamed; the change is purely semantic:

| Version 1 | Version 2 |
|-----------|-----------|
| `cam2_from_cam1` poses in COLMAP convention (cameras look down +Z with Y down) | Canonical convention: cameras look down −Z with +Y up |
| F/E/H matrices in pixel space | Unchanged — pixel space |

**Migration is mechanical and lossless.** A version 1 file upgrades on load by
conjugating each stored relative pose with the camera-frame flip
`S = diag(1, −1, −1)`: `R' = S · R · S`, `t' = S · t`. The F/E/H matrices are
untouched — they are pixel-space quantities, identical in both versions. Relative
poses never touch the world frame, so the `.sfmr` world canonicalization `W` does not
apply. Saving always writes the current version. Content hashes cover the stored
bytes, so hashes verify before conversion; a converted-then-saved file is a new
current-version file with new hashes.

As a consequence, a consumer that writes these relative poses into a COLMAP
database conjugates them by `S` back to COLMAP convention. The stored
F/E/H matrices are pixel-space and unchanged by the flip; see
[`sfmr-file-format.md`](sfmr-file-format.md#conversions-happen-at-the-io-boundary)
for the invariant and the `S`/`W` conversion math.

## Version History

- **Version 10**: The member statuses `rejected_unlocalizable_refined` and
  `rejected_unlocalizable_cells` — `member_status_names` may name them, and a
  writer always does — and the `refine_options` keys `regate_at_refined_shape`
  and `max_capped_cells`. Version 9 files read unchanged.
- **Version 9**: The cell status `refused_outlier` — `member_cell_status_names`
  may name it, and a writer always does — and the `refine_options` key
  `move_shape`, whether the refinement was allowed to change the member's
  shape. Version 8 files read unchanged.
- **Version 8**: Per-cell entries — `cluster_patches/` may carry
  `member_cell_shift_px`, `member_cell_zncc`, `member_cell_status` and
  `member_cell_iterations`, present together with a `member_cell_status_names`
  legend in its metadata: each member's displacement, ZNCC and status per cell
  of a three-by-three split of its patch, and its refinement passes. Version 7
  files read unchanged and carry no cells.
- **Version 7**: Status legend — `cluster_patches/metadata.json` carries
  `member_status_names`, and a `member_status` code is an index into it. A
  writer states the canonical legend; a reader accepts any legend and
  normalises the codes onto the canonical order. Version 6 files carry no
  legend and are read through the canonical one, which is the fixed numbering
  they were written with.
- **Version 6**: One geometry per member — `clusters/` gains a mandatory
  `member_positions` / `member_affine_shapes` pair and
  `cluster_patches/member_affines` is removed, so a cluster file holds exactly
  one keypoint position and one affine shape per member and its content is the
  file's stage: the `.sift` detections verbatim in a matcher output; the
  refinement's answer where its cascade measured, the untouched detection where
  it did not, in a cluster-patches output. Nothing is `NaN` and
  `member_status` is the sole authority. Cluster-backbone files below version 6
  are refused (regenerate the clusters from the `.sift` features, then refine
  again if enriched); pairwise files are unchanged.
- **Version 5**: Absolute affine shapes — `cluster_patches/member_affines`'
  leading 2×2 became the member's absolute affine shape `S = W·S_ref` rather
  than the reference-relative warp. Superseded by version 6, which removed the
  member; the shape definition itself survives in
  `clusters/member_affine_shapes`.
- **Version 4**: Self-contained geometry — mandatory per-image dimensions
  (`images/image_dims`, still current), and `cluster_patches/member_affines`'
  last column became the member's refined absolute keypoint position rather
  than the affine translation. Superseded by version 6 as above.
- **Version 3**: Cluster backbone — the `clusters/` section (CSR cluster
  membership, the cluster matcher's primary artifact) becomes the alternative
  correspondence backbone to `image_pairs/` (exactly one per file), and the optional
  `cluster_patches/` section stores photometrically refined per-member affine warps.
  Version ≤ 2 files (always pairwise) load unchanged.
- **Version 2**: Canonical camera convention — `cam2_from_cam1` relative
  poses in −Z-forward / +Y-up camera frames, matching `.sfmr` and `.camrig` — becomes
  normative; version 1 files (COLMAP convention) upgrade on load via `S`-conjugation
  of the stored poses. F/E/H matrices unchanged.
- **Version 1**: the first version, written as the integer `1`. Its two-view
  relative poses are in COLMAP convention; a reader upgrades them on load by
  `S`-conjugation (see [Version 1 → Version 2](#version-1--version-2)).