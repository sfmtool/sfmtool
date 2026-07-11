# The Matches File Format

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

See the [.sfmr file format](sfmr-file-format.md) for design principles behind this format.

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
> was already in use; files written by earlier sfmtool releases (format version 1)
> hold COLMAP-convention relative poses (cameras looking down +Z with Y down). See
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
│   └── feature_counts.{N}.uint32.zst              # Feature count per image (as used in matching)
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
│   └── member_features.{K}.uint32.zst             # Feature index in that image's .sift per member
├── cluster_patches/                               # (Optional section, requires clusters/)
│   ├── metadata.json.zst                          # Refinement options + summary counts
│   ├── reference_members.{C}.uint32.zst           # Global member index of each cluster's reference
│   ├── member_status.{K}.uint8.zst                # ClusterMemberStatus enum per member
│   ├── member_affines.{K}.2.3.float64.zst         # x_member = A·x_ref + t, pixel coords
│   ├── member_zncc.{K}.float32.zst                # Achieved windowed ZNCC vs reference (NaN if n/a)
│   └── member_shift_px.{K}.float32.zst            # Translation drift from the SIFT seed (NaN if n/a)
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
  "version": 3,
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
- `version`: Format version number. `1`, `2`, or `3` (see
  [Versioning and Migration](#versioning-and-migration))
- `matching_method`: Type of matching used to produce these matches
  - `"exhaustive"`: Exhaustive pairwise matching
  - `"sequential"`: Sequential matching with overlap
  - `"vocab_tree"`: Vocabulary tree-based matching
  - `"spatial"`: Spatial matching (GPS/location-based pair selection)
  - `"transitive"`: Transitive matching
  - `"custom"`: Any other method
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

**Field descriptions:**
- `metadata_xxh128`: Hash of uncompressed `metadata.json.zst` content bytes
- `images_xxh128`: Hash of all image data files' uncompressed contents, fed sequentially
  into a streaming XXH128 hasher in lexicographic path order
- `image_pairs_xxh128`: Hash of all image pairs data files' uncompressed contents, fed sequentially
  into a streaming XXH128 hasher in lexicographic path order. Present exactly when the file
  stores the pairwise backbone.
- `clusters_xxh128`: Hash of all clusters data files' uncompressed contents, fed sequentially
  into a streaming XXH128 hasher in lexicographic path order. Present exactly when the file
  stores the cluster backbone.
- `cluster_patches_xxh128`: (Optional) Hash of all cluster patches data files' uncompressed
  contents, fed sequentially into a streaming XXH128 hasher in lexicographic path order.
  Present only when the cluster patches section exists.
- `two_view_geometries_xxh128`: (Optional) Hash of all TVG data files' uncompressed contents,
  fed sequentially into a streaming XXH128 hasher in lexicographic path order. Present only
  when the two-view geometries section exists.
- `content_xxh128`: Hash of all present section hashes concatenated as raw 16-byte big-endian
  digests in order: metadata, images, pairs, clusters, cluster_patches, two_view_geometries
  (each only if present). A pairwise file's byte stream is identical to the pre-version-3
  layout, so version ≤ 2 hashes verify unchanged.

**Note**: All hashes are computed on uncompressed content bytes. For JSON files, hash the raw
bytes after decompression, NOT re-serialized JSON.

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
images that are likely co-observations of one surface point, in CSR layout (identical
to the in-memory `ClusterSet`). Cluster `c` owns members
`cluster_starts[c]..cluster_starts[c+1]` of the member-parallel arrays.

**Pairs are a derived view.** The canonical expansion is `clusters_to_pair_matches`
(every within-cluster cross-image member pair, grouped and sorted per the
`image_pairs/` ordering rules). Because `two_view_geometries/` arrays are keyed per
stored pair, a cluster file cannot carry TVGs directly; the geometric-verification
step materializes the expansion by writing a new pairwise `.matches` file with
`image_pairs/` + `two_view_geometries/` (the write-once workflow, unchanged). Pair
descriptor distances, which the stored pairwise form carries, are recomputed from the
referenced `.sift` files when a consumer needs them. See
[`specs/core/cluster-patches.md`](../core/cluster-patches.md) for the design
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

### 6. Cluster Patches (Optional Section)

Written by the cluster-patches operation into a **new** file that copies the source
file's images and clusters sections (write-once workflow, same as adding TVGs).
Requires the cluster backbone. Arrays parallel the clusters' member arrays: for each
cluster, a reference member plus, for every other member, a photometrically refined
affine warp mapping the reference's local patch onto that member's image, with
vetting statuses and signals.

#### `cluster_patches/metadata.json.zst`

```json
{
  "cluster_count": 5200,
  "member_count": 14100,
  "refine_options": {
    "radius": 4.0,
    "resolution": 15,
    "min_zncc": 0.85,
    "max_shift_px": 3.0
  }
}
```

**Field descriptions:**
- `cluster_count` / `member_count`: Must equal the top-level `cluster_count` /
  `cluster_member_count` (and therefore the clusters section counts)
- `refine_options`: The refinement parameters used

#### `cluster_patches/reference_members.{C}.uint32.zst`

- **Shape**: `(C,)` where C = cluster_count
- **Data type**: `uint32` (little-endian)
- Global member index of each cluster's reference member; `0xFFFFFFFF` (`u32::MAX`)
  when the cluster could not be refined (no usable reference)
- **Constraint**: When not `0xFFFFFFFF`, `reference_members[c]` lies in cluster `c`'s
  member range and that member's status is `0` (reference)

#### `cluster_patches/member_status.{K}.uint8.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `uint8`
- Per-member status:
  - `0 reference` — the cluster's reference member (identity affine, ZNCC 1.0)
  - `1 kept` — refined and vetted successfully
  - `2 rejected_low_zncc` — achieved ZNCC below the acceptance threshold
  - `3 rejected_shift` — translation drifted too far from the SIFT seed
  - `4 duplicate_image` — outscored by another kept member in the same image, or
    shares the reference's image
  - `5 not_evaluated` — degenerate shape, template/seed support out of frame, or the
    cluster itself was unrefinable
  - `6 rejected_unlocalizable` — the member's own patch scored a keypoint position
    uncertainty above the localizability threshold, so it was excluded before
    reference selection and refinement (see
    [`patch-localizability.md`](../core/patch-localizability.md))
- A patch cluster = the reference plus its `kept` members; statuses preserve the
  rejected members so consumers can re-gate without re-running (the ZNCC/shift arrays
  are the signals, mirroring how `match_descriptor_distances` enables descriptor
  re-filtering)
- **Constraint**: Every value is a valid discriminant (`0..=6`)
- **Constraint**: At most one member with status `0` or `1` per (cluster, image)

#### `cluster_patches/member_affines.{K}.2.3.float64.zst`

- **Shape**: `(K, 2, 3)` where K = cluster_member_count
- **Data type**: `float64` (little-endian)
- Absolute affine warp in pixel coordinates (COLMAP pixel convention):
  `x_member = A·x_ref + t` with `A` the leading 2×2 block and `t` the last column.
  Stored absolute (not anchored/relative) so it composes directly (`member ← ref`,
  and member↔member via the reference) without re-deriving the SIFT seed
- Identity|0 for the reference row; zeros where not evaluated

#### `cluster_patches/member_zncc.{K}.float32.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Achieved windowed ZNCC vs the reference template, aligned with the member arrays;
  `NaN` where not evaluated

#### `cluster_patches/member_shift_px.{K}.float32.zst`

- **Shape**: `(K,)` where K = cluster_member_count
- **Data type**: `float32` (little-endian)
- Translation drift in pixels from the SIFT seed; `NaN` where not evaluated

### 7. Two-View Geometries (Optional Section)

The two-view geometries section stores the results of geometric verification. It is optional —
a `.matches` file can contain only candidate matches. It requires the pairwise backbone
(`image_pairs/`): its arrays are keyed per stored pair, so a cluster-bearing file cannot
carry TVGs. To add geometric verification results,
write a new `.matches` file that includes both the candidate matches and the TVGs (see
"Writing a verified .matches file from an existing one" in Usage Examples). This section
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

### Cluster Constraints

1. **CSR well-formed**: `cluster_starts[0] == 0`, non-decreasing, final value equals
   the member count
2. **Minimum size**: Every cluster has ≥ 2 members
3. **Member bounds**: `member_images[k] < image_count` and
   `member_features[k] < feature_counts[member_images[k]]`
4. **Cluster patches parallel**: The `cluster_patches/` arrays have lengths `C`
   (`reference_members`) and `K` (member arrays) matching the clusters section
5. **Statuses valid**: Every `member_status` value is a valid discriminant (`0..=6`);
   `reference_members[c]` is `0xFFFFFFFF` or lies in cluster `c`'s member range with
   status `0`; at most one status-`0`-or-`1` member per (cluster, image)

### No required ordering within a pair

Matches within a single pair (the slice of `match_feature_indexes` for that pair) have no
required ordering. They are an unordered set of correspondences.

### Index Relationships

- `image_index_pairs[k]` → `(idx_i, idx_j)` into `images/names.json.zst`
- `match_feature_indexes[m][0]` → feature index in `.sift` file for image `idx_i`
- `match_feature_indexes[m][1]` → feature index in `.sift` file for image `idx_j`
- `sift_content_hashes[i]` → verifies the `.sift` file hasn't changed

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
expansion is deterministic and cheap (`clusters_to_pair_matches`), while storing both
roughly doubles the correspondence payload with derived values: per-pair data grows as
Σ C(k,2) over cluster sizes versus the Σ k the clusters themselves cost. Consumers
that need pairs obtain them by calling the expansion at read time; the cluster file
remains the durable primary artifact, and the geometric-verification step writes the
solver-facing pairwise derivative as a new file. See
[`specs/core/cluster-patches.md`](../core/cluster-patches.md) for the full design
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

Features live in `.sift` files. The `.matches` file references them by index and verifies
integrity via `sift_content_hashes`. This avoids duplication and keeps the `.matches` file
focused on correspondences.

### Why ZIP with STORE + zstd?

Same rationale as `.sift` and `.sfmr`: random access to individual files (read just pair
counts without loading all match data), standard tooling, efficient compression of binary
arrays.

### Why columnar storage for matches?

The flat concatenated layout with per-pair counts (same pattern as tracks in `.sfmr`) enables:
- Reading just pair metadata without loading match data
- Loading matches for a specific pair by computing the offset from cumulative counts
- Better compression (similar values together)

## Compression Details

Same as `.sift` and `.sfmr`:
- **Container**: ZIP archive using STORE method (no ZIP-level compression)
- **Compression**: zstandard on each entry, default level 3
- **JSON files**: Compact encoding (no pretty-printing)
- **Binary files**: Direct compression of raw byte stream

## Integrity Verification

### Hash Computation

1. **Metadata hash**: XXH128 of uncompressed `metadata.json.zst` content bytes
2. **Section hashes** (images, pairs, clusters, cluster_patches, two_view_geometries):
   Feed all files' uncompressed content bytes into a streaming XXH128 hasher in
   lexicographic path order
3. **Overall hash**: Concatenate all present section hashes as raw 16-byte big-endian digests
   in order (metadata, images, pairs, clusters, cluster_patches, two_view_geometries —
   each only if present), then compute XXH128

### Verification Process

1. Check backbone/flag consistency (exactly one backbone's entries and summary counts
   present, matching the `has_*` flags); a file that fails these is reported without
   further section checks
2. Decompress each file and hash the raw uncompressed bytes
3. Recompute section and overall hashes
4. Compare with stored values in `content_hash.json.zst`
5. Validate structural constraints (feature index bounds, count sums, pair ordering;
   cluster CSR, member bounds, patch statuses and reference invariants)

## Usage Examples

### Writing a .matches file after sequential matching

```python
from sfmtool.matches_file import write_matches

write_matches(
    output_path="matches/20260329-00-sequential_1-83.matches",
    images={
        "names": image_names,
        "feature_tool_hashes": feature_tool_hashes,
        "sift_content_hashes": sift_content_hashes,
        "feature_counts": feature_counts,
    },
    pairs={
        "image_index_pairs": image_index_pairs,   # (P, 2) uint32
        "match_counts": match_counts,              # (P,) uint32
        "match_feature_indexes": match_indexes,    # (M, 2) uint32
        "match_descriptor_distances": distances,   # (M,) float32
    },
    metadata={
        "matching_method": "sequential",
        "matching_tool": "colmap",
        "matching_tool_version": "4.02",
        "matching_options": {
            "overlap": 10,
            "quadratic_overlap": True,
            "max_feature_count": 8192,
        },
        "workspace": {...},
    },
)
```

### Reading matches and populating a COLMAP database

```python
from sfmtool.matches_file import MatchesReader

with MatchesReader("matches.matches") as reader:
    metadata = reader.metadata
    image_names = reader.read_image_names()
    pairs = reader.read_image_index_pairs()
    counts = reader.read_match_counts()

    # Load matches for specific pairs
    for k, (idx_i, idx_j) in enumerate(pairs):
        matches = reader.read_matches_for_pair(k)  # (count, 2) uint32
        db.write_matches(img_id_i, img_id_j, matches)

    # Or load all matches at once
    all_matches = reader.read_all_match_feature_indexes()  # (M, 2) uint32
```

### Writing a verified .matches file from an existing one

A common workflow: read a matches-only file, run geometric verification, and write a
new self-contained file that includes both matches and TVGs. The original file is not
modified — the new file is written to a separate path with its own content hash.

```python
from sfmtool.matches_file import MatchesReader, write_matches

# Read the matches-only file
with MatchesReader("matches/20260329-00-sequential_1-83.matches") as reader:
    data = reader.read_all()

# Run geometric verification (e.g., via pycolmap)
tvgs = run_geometric_verification(data)

# Write a NEW file with both matches and TVGs
write_matches(
    output_path="tvg-matches/20260329-01-sequential-verified.matches",
    images=data["images"],
    pairs=data["pairs"],
    two_view_geometries=tvgs,
    metadata={**data["metadata"], "has_two_view_geometries": True},
)
```

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

## Versioning and Migration

The format has three released versions (`1`, `2`, and `3`). The format is versioned
(`metadata.json` `version`) precisely so that changes like the ones below can upgrade
on load instead of breaking old files. Writers always emit the current version;
readers accept any version up to it.

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

As a consequence, consumers that export two-view geometries to a COLMAP database
(`sfm to-colmap-db` via `src/sfmtool/colmap/db_setup.py`) S-conjugate the canonical
poses back to COLMAP convention when building `pycolmap.Rigid3d`. The stored
F/E/H matrices are pixel-space and unchanged by the flip; see
[`sfmr-file-format.md`](sfmr-file-format.md#conversions-happen-at-the-io-boundary)
for the invariant and the `S`/`W` conversion math.

## Version History

- **Version 3**: Cluster backbone — the `clusters/` section (CSR cluster
  membership, the cluster matcher's primary artifact) becomes the alternative
  correspondence backbone to `image_pairs/` (exactly one per file), and the optional
  `cluster_patches/` section stores photometrically refined per-member affine warps.
  Version ≤ 2 files (always pairwise) load unchanged.
- **Version 2**: Canonical camera convention — `cam2_from_cam1` relative
  poses in −Z-forward / +Y-up camera frames, matching `.sfmr` and `.camrig` — becomes
  normative; version 1 files (COLMAP convention) upgrade on load via `S`-conjugation
  of the stored poses. F/E/H matrices unchanged.
- **Version 1.0rc1**: Release candidate