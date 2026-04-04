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

1. **Candidate matches** — raw feature correspondences from a matching step
2. **Two-view geometries** (optional) — geometrically verified inlier subsets with estimated
   relative poses such as F/E/H matrices, rotation, translation

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
COLMAP database starting from 1.

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
├── image_pairs/
│   ├── metadata.json.zst                          # Pair-level metadata
│   ├── image_index_pairs.{P}.2.uint32.zst         # (idx_i, idx_j) per pair, idx_i < idx_j
│   ├── match_counts.{P}.uint32.zst                # Number of matches per pair
│   ├── match_feature_indexes.{M}.2.uint32.zst     # (feat_idx_i, feat_idx_j) per match
│   └── match_descriptor_distances.{M}.float32.zst # L2 descriptor distance per match
└── two_view_geometries/                           # (Optional section)
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
- `{I}` = total number of inlier matches across all pairs (two-view geometries)

## File Format Details

### 1. Top-Level Metadata (`metadata.json.zst`)

```json
{
  "version": 1,
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
  "has_two_view_geometries": false
}
```

**Field descriptions:**
- `version`: Format version number. Must be `1` for this specification
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
- `image_pair_count`: Number of image pairs with matches
- `match_count`: Total number of matches across all pairs
- `has_two_view_geometries`: Whether the optional two-view geometries section is present

### 2. Content Hash (`content_hash.json.zst`)

```json
{
  "metadata_xxh128": "...",
  "images_xxh128": "...",
  "image_pairs_xxh128": "...",
  "two_view_geometries_xxh128": "...",
  "content_xxh128": "..."
}
```

**Field descriptions:**
- `metadata_xxh128`: Hash of uncompressed `metadata.json.zst` content bytes
- `images_xxh128`: Hash of all image data files' uncompressed contents, fed sequentially
  into a streaming XXH128 hasher in lexicographic path order
- `image_pairs_xxh128`: Hash of all image pairs data files' uncompressed contents, fed sequentially
  into a streaming XXH128 hasher in lexicographic path order
- `two_view_geometries_xxh128`: (Optional) Hash of all TVG data files' uncompressed contents,
  fed sequentially into a streaming XXH128 hasher in lexicographic path order. Present only
  when the two-view geometries section exists.
- `content_xxh128`: Hash of all present section hashes concatenated as raw 16-byte big-endian
  digests in order: metadata, images, pairs, two_view_geometries (if present)

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

### 4. Pairs (Putative Matches)

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

### 5. Two-View Geometries (Optional Section)

The two-view geometries section stores the results of geometric verification. It is optional —
a `.matches` file can contain only candidate matches. To add geometric verification results,
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
- Relative rotation quaternion in WXYZ format
- `[1, 0, 0, 0]` (identity) when not applicable

#### `two_view_geometries/translations_xyz.{P}.3.float64.zst`

- **Shape**: `(P, 3)` where P = pair_count
- **Data type**: `float64` (little-endian)
- Relative translation vector
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

### Ordering Requirements

1. **Pairs sorted**: `image_index_pairs` MUST be sorted lexicographically by `(idx_i, idx_j)`
   with `idx_i < idx_j`
2. **Match counts aligned**: `match_counts[k]` = number of entries in the match arrays
   belonging to pair `k`. `sum(match_counts) == match_count`
3. **Inlier counts aligned**: Same relationship for the TVG inlier arrays
4. **Feature index bounds**: All feature indexes must be less than the corresponding
   `feature_counts` entry

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
2. **Section hashes** (images, pairs, two_view_geometries): Feed all files' uncompressed
   content bytes into a streaming XXH128 hasher in lexicographic path order
3. **Overall hash**: Concatenate all present section hashes as raw 16-byte big-endian digests
   in order (metadata, images, pairs, two_view_geometries if present), then compute XXH128

### Verification Process

1. Decompress each file and hash the raw uncompressed bytes
2. Recompute section and overall hashes
3. Compare with stored values in `content_hash.json.zst`
4. Validate structural constraints (feature index bounds, count sums, pair ordering)

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

## Version History

- **Version 1.0rc1**: Release candidate