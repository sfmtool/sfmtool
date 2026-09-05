# `sfm match` Command

## Overview

Matches SIFT features between image pairs and writes a `.matches` file. Requires a workspace
with previously extracted SIFT features. Uses COLMAP to perform the matching, except for the
track-cluster matcher and the experimental "flow" mode, which has some promise for videos.

The command is implemented in
[`_commands/match.py`](../../../src/sfmtool/_commands/match.py), which is a thin
Click wrapper over [`feature_match/_run.py`](../../../src/sfmtool/feature_match/_run.py)
(the matching methods) and
[`feature_match/_derive_pairs.py`](../../../src/sfmtool/feature_match/_derive_pairs.py)
(the `--derive-pairs` mode).

## Command Syntax

```bash
sfm match [PATHS...] --exhaustive | --sequential | --flow | --cluster [OPTIONS...]
sfm match --derive-pairs CLUSTERS.matches [-o OUTPUT.matches]
sfm match --merge FILE1.matches FILE2.matches ... -o OUTPUT.matches
```

`PATHS` are image directories or files; when omitted, the current directory is used
(except with `--merge`, which requires explicit `.matches` paths, and `--derive-pairs`,
which requires exactly one). Exactly one matching method must be specified, or `--merge`
to combine existing `.matches` files.

## Matching Methods

| Method | Description |
|--------|-------------|
| `--exhaustive / -e` | Match every pair of images against every other |
| `--sequential / -s` | Match each image against its nearby neighbors in sequence order |
| `--flow` | Use dense optical flow to guide feature matching |
| `--cluster` | Cluster all images' descriptors at once (background-floor track-cluster matching) |
| `--derive-pairs` | Verify a clusters-bearing `.matches` file into the pairwise + two-view-geometry file COLMAP's mapper reads |
| `--merge` | Merge multiple `.matches` files into one |

Exactly one matching method must be given. Each method has its own tuning
options (`--sequential-overlap` for `--sequential`; `--flow-preset` /
`--flow-skip` for `--flow`; `--cluster-alpha` / `--cluster-d` /
`--cluster-preset` for `--cluster`). Passing a method-specific option without
its companion method is rejected with a `UsageError` rather than silently
ignored.

Every method writes exactly one `.matches` file, at `-o` or at a generated
default.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sequential-overlap` | int | 10 | Number of overlapping neighbors for sequential matching |
| `--flow-preset` | `fast` \| `default` \| `high_quality` | `default` | Optical flow quality preset |
| `--flow-skip` | int | 5 | Sliding window size for flow matching |
| `--cluster-alpha` | float | 0.8 | Background-floor radius multiplier for cluster matching |
| `--cluster-d` | int | 10 | Background rank: the d-th-nearest distance sets the floor for cluster matching |
| `--cluster-preset` | `accurate` \| `balanced` \| `fast` | `accurate` | Kd-tree forest preset for cluster matching |
| `--max-features` | int | | Maximum features per image |
| `--output / -o` | path | auto | Output `.matches` file path (default: timestamped, required for `--merge`) |
| `--range / -r` | string | | Range expression for file numbers |
| `--camera-model` | choice | auto | Camera model override (e.g., `SIMPLE_RADIAL`, `OPENCV`). Accepts the same 11 COLMAP model names as `solve` and `camrig create`. |

`--max-features` and `--range` describe an image set being matched, so
`--derive-pairs` — whose image set is fixed by the clusters file it reads —
rejects them. `--camera-model` feeds geometric verification, so `--cluster`,
which verifies nothing, rejects it too; under `--derive-pairs` it selects the
model the two-view geometries are estimated with.

## Process

1. Loads workspace config and SIFT features
2. Populates a COLMAP database with images, cameras, keypoints, and descriptors
3. Runs the selected matching strategy
4. Computes descriptor distances for matched pairs
5. Writes a timestamped `.matches` file

`--cluster` is the exception: it opens no database and runs steps 1, 3 and 5
only, because the clusters it writes carry neither descriptor distances nor
two-view geometries.

## Camera Intrinsics

If any image being processed resolves a `camera_config.json` (closest-ancestor walk from
its parent directory up to the workspace root), the file's intrinsics are used and
`--camera-model` is rejected with an error. See
[`../../workspace/camera-config.md`](../../workspace/camera-config.md).

## Cluster Matching

`--cluster` uses the background-floor track-cluster matcher: instead of
enumerating image pairs, it concatenates every image's descriptors into one
corpus, queries each descriptor's nearest neighbours over a randomized kd-tree
forest, and keeps the cross-image neighbours within `--cluster-alpha` × its
`--cluster-d`-th-nearest distance (its *background floor*). Those candidates
are materialized into track clusters, and the clusters are the output. Image
pair selection falls out of the clustering — the pairs are exactly those that
share a cluster — and is derivable from the file at any time.

### One output: the clusters-bearing `.matches`

`--cluster` writes a single file — clusters backbone, no pairs, no two-view
geometries, carrying `matching_method: "cluster"`, the matcher's options, and
`cluster_count` / `cluster_member_count` metadata. Default path: the workspace
`matches/` directory, with the usual timestamped stem plus a `-clusters`
suffix; `-o` names it explicitly.

The file feeds cluster-native consumers (`sfm cluster-patches`,
`sfm estimate-intrinsics`, inspection) directly, and pairwise consumers derive
pairs from it at read time via `sfmtool.feature_match.pairs_from_matches` —
`sfm to-colmap-db` reads it unchanged. Images are listed in lexicographic
order, which is the order every `.matches` reader uses, so indices stay
comparable with any file derived from it.

**The run is deterministic.** Nothing on this path consults a COLMAP database
or `pycolmap`; the same corpus and options produce the same backbone bit for
bit. Geometric verification, the one nondeterministic step, lives in
[`--derive-pairs`](#derive-pairs).

The clustering uses no intrinsics or poses, so `--camera-model` is rejected
with `--cluster`: it would be inert. It applies to `--derive-pairs`, which
estimates the two-view geometries.

See [`../../core/features/track-cluster-matching.md`](../../core/features/track-cluster-matching.md)
for the algorithm design, empirical justification, and the production API.

## Derive Pairs

Two-view geometries exist for COLMAP's mapper, which reads its correspondence
graph from the database's two-view geometry table. Nothing in sfmtool's own
pipeline consumes them, so verification is a COLMAP-boundary concern rather
than a matcher concern, and `--derive-pairs` produces the boundary artifact on
demand:

```bash
sfm match --derive-pairs matches/my-clusters.matches
```

The one positional argument is a clusters-bearing `.matches` file. Anything
else — image paths, several paths, or a `.matches` file that stores pairs
rather than clusters — is rejected with a `UsageError`. The output is the
verified pairwise + two-view-geometry
`.matches` file, at `-o` or, by default, under the workspace's `tvg-matches/`
directory named after the input's stem with a trailing `-clusters` removed
(so `matches/kerry_park-clusters.matches` →
`tvg-matches/kerry_park.matches`).

The derivation expands the clusters into per-image-pair matches, writes them
with the workspace's features and cameras into a throwaway COLMAP database,
runs `pycolmap.verify_matches` over exactly the derived pair list, and reads
the surviving matches back with their geometries. Verification culls pairs
below COLMAP's `min_num_matches`, so the output's pairs are a subset of the
expansion, with identical match sets on every surviving pair.

The output inherits the source's `matching_method`, matcher options and
workspace block, and records where it came from under
`matching_options["derived_pairs"]`:

```json
{
  "derived_pairs": {
    "source_path": "../matches/kerry_park-clusters.matches",
    "source_content_xxh128": "9a51..."
  }
}
```

`source_path` is relative to the output file's directory and
`source_content_xxh128` is the source's whole-file hash, the same pair of
facts a derived cluster selection records (see
[`../../formats/matches-file-format.md`](../../formats/matches-file-format.md)).

**Rig same-frame pair exclusion is not applied.** The stored clusters are the
raw matcher output, and the derivation verifies every pair they produce; for a
multi-sensor rig, the same-frame `(i, j)` exclusion (back-to-back sensors with
no shared view) is applied only by the in-solve cluster matching mode, which
matches into the solve's own database. A consumer that needs the exclusion
applies it itself.

`sfm solve` refuses a clusters-bearing `.matches` file and names this mode,
rather than handing the mapper a database with an empty correspondence graph.

## Merge

`--merge` combines multiple `.matches` files into a single file. This is useful
for combining results from different matching strategies (e.g., sequential + exhaustive)
before running a solve.

The merge process:
- Builds a unified image list from all input files
- Validates that images with the same name have identical SIFT content hashes
- Concatenates matches for each pair across all input files
- Deduplicates matches with the same feature index pair (keeps lowest descriptor distance)
- Preserves two-view geometry (TVG) data when present in any input file
  - For pairs with TVG in multiple inputs, keeps the TVG with the most inliers
  - Handles index remapping by transforming F/E/H matrices and inverting poses as needed
- Records source file names and methods in the output metadata

## Usage Examples

```bash
# Exhaustive matching (small datasets)
sfm match --exhaustive

# Sequential matching for ordered image sequences
sfm match --sequential --sequential-overlap 20

# Flow-based matching for video frames
sfm match --flow --flow-preset high_quality --flow-skip 10

# Background-floor track-cluster matching, then the COLMAP boundary file
sfm match --cluster images/ -o matches/my-clusters.matches
sfm match --derive-pairs matches/my-clusters.matches

# Match a subset of images
sfm match --exhaustive --range 1:100 --max-features 4096

# Merge matches from different strategies
sfm match --merge seq.matches exhaustive.matches -o combined.matches
```
