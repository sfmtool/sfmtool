# Cluster Patches: SIFT Clusters → Patch Clusters

_Status: draft for review. Design informed by the experiments in
`reports/2026-07-09-exp-pairwise-sift-warp.md` (harnesses
`scripts/exp_pairwise_sift_warp.py`, `scripts/exp_cluster_patch_clusters.py`)._

## The Idea

[Track-cluster matching](track-cluster-matching.md) materializes candidate
track clusters — groups of SIFT features across images that are likely
co-observations of one surface point — as its primary output, then flattens
them into pairwise matches for the legacy pipeline. The clusters themselves
carry more structure than the pairwise view: each is a proto-track, and each
member's SIFT detection carries a position and a 2×2 affine shape.

This spec turns a cluster of SIFT features into a **patch cluster**: a
reference member plus, for every other member, an affine warp that maps the
reference's local patch onto that member's image, photometrically refined and
vetted. A patch cluster is the 2D, pre-reconstruction analog of an
`embedded_patches` track — everything about the track's inter-view geometry
that can be known before any camera poses exist.

The 2026-07-09 experiments validated the approach against calibrated
`embed-patches` ground truth on seoul_bull, kerry_park (fisheye rig), and
dino_dog_toy:

- The true patch-to-patch warp at SIFT-patch scale is **affine to
  0.004–0.013 px** on all datasets, fisheye included. Perspective terms add
  nothing and overfit; the warp family stops at affine.
- Seeding from the SIFT affine shapes (`x_j = pos_j + A_j A_i⁻¹ (x_i −
  pos_i)`) and hill-climbing Gaussian-windowed ZNCC recovers the warp *shape*
  to 0.25–0.5 px and improves translation vs the detections (e.g. 0.38 →
  0.25 px against the refined tracks' consensus keypoints).
- Built on raw matcher clusters, >90% of track-covered clusters vet to a
  **pure subset of a single refined track** (median purity 1.0; 63–83%
  recover the full track). The dominant failure is over-culling, not
  contamination — the acceptance gate should stay permissive.
- Achieved ZNCC exceeds the ground-truth warp's own ZNCC; the score gates
  **match validity**, never warp correctness.
- A multi-view congealing pass adds nothing over pairwise refinement at
  raw-cluster sizes (2–5 members); consensus localization pays only after
  view expansion, which is downstream machinery's job.

## Decision: staged artifacts, one format

Two shapes were considered:

1. **A single operation** `(images + .sift files) → patch clusters`, matching
   and refinement fused.
2. **Staged**: persist the matcher's cluster output, then a separate
   operation enriches a clusters-bearing file into a patch-clusters-bearing
   file.

**This spec adopts the staged design**, with both new pieces as **optional
sections of the existing `.matches` format** rather than a new file type or a
hard "variant":

- Descriptor matching and photometric patch refinement have different inputs
  (descriptors vs image pixels), different costs, and different iteration
  cadences. The corpus-wide kd-forest matching is the expensive step; the
  experiments iterated the photometric stage dozens of times against fixed
  clusters. Fusing them forces a re-match per refinement-knob change.
- The staging mirrors the format's own two-view-geometries precedent: an
  enrichment stage that needs extra inputs is an *optional section*, written
  by producing a **new** write-once file ("one format, one reader, one
  writer" — see `specs/formats/matches-file-format.md`, Design Rationale).
- [track-cluster-matching.md](track-cluster-matching.md) already declares the
  clusters "the matcher's primary output, kept so later consumers can work
  with the clusters themselves" — but today they are dropped after pair
  expansion. Persisting them is independently valuable (inspection,
  re-expansion with different parameters, future cluster-native consumers)
  and this operation is simply its first consumer.
- A monolithic operation still needs an on-disk output format, so option 1
  saves no format work — it only hides the intermediate.

A cluster-bearing file stores **clusters instead of the pairwise
expansion** — `clusters/` and `image_pairs/` are mutually exclusive, and every
`.matches` file carries exactly one of them as its correspondence backbone.
The expansion is deterministic and cheap (`clusters_to_pair_matches` already
exists), while storing both roughly doubles the correspondence payload with
derived values; per-pair data grows as Σ C(k,2) over cluster sizes versus the
Σ k the clusters themselves cost. Existing consumers (TVG verification,
`to-colmap-db`, solve) obtain pairs by calling the expansion at read time —
the reader exposes one pairs API that returns stored pairs or expands
clusters, so consumer code has a single path. Pair descriptor distances,
which the stored pairwise form carries, are recomputed from the referenced
`.sift` files when a consumer actually needs them (the files are already
located and content-hash-verified through the images section).

## Format: two new `.matches` sections

To be folded into `specs/formats/matches-file-format.md` on implementation;
the format version bumps to **3**. Version ≤ 2 files load unchanged; the
`image_pairs/` section, mandatory through version 2, becomes the
stored-pairs alternative to `clusters/` (exactly one of the two is present).
The overall content hash covers the present sections in the order metadata,
images, pairs, clusters, cluster_patches, two_view_geometries.

### `clusters/` — the matcher's primary artifact

Written by `sfm match --cluster` **in place of** the `image_pairs/` section
(which such a file omits entirely). CSR layout identical to the in-memory
`ClusterSet`:

```
clusters/
├── metadata.json.zst                    # cluster_count, member_count,
│                                        # matcher options (d, alpha, min_size, preset)
├── cluster_starts.{C+1}.uint32.zst      # cluster c owns members starts[c]..starts[c+1]
├── member_images.{M}.uint32.zst         # index into images/names.json.zst
└── member_features.{M}.uint32.zst       # feature index in that image's .sift
```

Constraints: `cluster_starts[0] == 0`, non-decreasing, ends at `M`; every
cluster has ≥ 2 members; `member_features[k] < feature_counts[member_images[k]]`.
Top-level metadata gains `"has_clusters": true`; the pairwise summary fields
(`image_pair_count`, `match_count`) are replaced by `cluster_count` and
`cluster_member_count` in cluster-bearing files.

**Pairs are a derived view.** The canonical expansion is the existing
`clusters_to_pair_matches`: every within-cluster cross-image member pair,
grouped and sorted per the `image_pairs/` ordering rules. The reader surfaces
this behind the same pairs API used for stored-pairs files. Because
`two_view_geometries/` arrays are keyed per stored pair, a cluster file
cannot carry TVGs directly; the geometric-verification step materializes the
expansion — it reads a cluster file and writes a new pairwise
`.matches` file with `image_pairs/` + `two_view_geometries/` (the write-once
workflow, unchanged). The cluster file remains the durable primary artifact;
the verified pairwise file is the solver-facing derivative.

### `cluster_patches/` — the enrichment (requires `clusters/`)

Written by the cluster-patches operation into a **new** file that copies the
source file's images and clusters sections (write-once workflow, same as
adding TVGs). Arrays parallel the clusters' member arrays:

```
cluster_patches/
├── metadata.json.zst                    # refinement options (below) + summary counts
├── reference_members.{C}.uint32.zst     # global member index of each cluster's reference
├── member_status.{M}.uint8.zst          # enum, see below
├── member_affines.{M}.2.3.float64.zst   # x_member = A·x_ref + t, pixel coords (COLMAP
│                                        # convention); identity|0 for the reference row;
│                                        # zeros where not evaluated
├── member_zncc.{M}.float32.zst          # achieved windowed ZNCC vs reference (NaN if n/a)
└── member_shift_px.{M}.float32.zst      # translation drift from the SIFT seed (NaN if n/a)
```

`member_status` values: `0 reference`, `1 kept`, `2 rejected_low_zncc`,
`3 rejected_shift`, `4 duplicate_image` (a kept member already covers this
image with higher ZNCC), `5 not_evaluated` (degenerate shape, unusable
template, out of frame). A patch cluster = the reference plus its `kept`
members; statuses preserve the rejected members so consumers can re-gate
without re-running (the ZNCC/shift arrays are the signals, mirroring how
`match_descriptor_distances` enables descriptor re-filtering).

The affine is stored absolute (full 2×3 in pixel coordinates) rather than
anchored/relative: it composes directly (`member ← ref`, and member↔member
via the reference) and does not require the consumer to re-derive the SIFT
seed. Top-level metadata gains `"has_cluster_patches": true`.

## The operation

```
sfm cluster-patches -i clusters.matches -o patch-clusters.matches [options]
```

A new flat command (Image Feature category). Inputs: a clusters-bearing
`.matches` file plus the workspace images/`.sift` files it references.
Per cluster:

1. **Reference selection.** v1: the member with the largest SIFT scale
   (`√|det A|`). Known weakness: on kerry_park the largest-scale member is
   often an untracked feature; a smarter policy (e.g. highest median
   descriptor-space centrality or template self-agreement) is an open
   question, and the format is policy-agnostic (the reference is data).
2. **Seed.** For each other member, seed the warp from the SIFT affine
   shapes: `M₀ = A_member · A_ref⁻¹`, translation from the detections.
3. **Refine.** Hill-climb Gaussian-windowed ZNCC over a shift → similarity →
   affine cascade (never perspective) on a template of `radius` keypoint-frame
   units around the reference detection. No multi-view congealing pass (see
   findings above).
4. **Vet.** Reject members below `min_zncc` or drifting more than
   `max_shift_px` from the seed; keep at most one member per image (best
   ZNCC); record statuses and signals for the rest.

Defaults (from the experiment calibration):

| option | default | notes |
|---|---|---|
| `radius` | 4.0 | template half-width, keypoint-frame units; 2 is too small for the affine DOF, 6–8 buys nothing |
| `resolution` | 15 | template samples per axis |
| `min_zncc` | 0.85 | permissive by design — over-culling, not contamination, is the failure mode; downstream stages re-gate |
| `max_shift_px` | 3.0 | matches the `embed-patches` keypoint-localization gate |
| warp family | affine | fixed; similarity is a degenerate case the optimizer passes through |

The kernel belongs in `sfmtool-core` (a `patch::cluster_refine` sibling of
the existing keypoint kernels — it reuses the same ZNCC/rendering machinery),
exposed through `sfmtool-py`; the CLI command is thin orchestration, per the
repo's pattern. The experiment scripts stay as the reference implementation
until the Rust kernel lands.

## Consumers (future work, out of scope here)

- **Photometric verification as a TVG alternative/complement**: patch-vetted
  clusters → expanded pairs skip or soften descriptor-distance and geometric
  gates.
- **Surfel seeding**: after a solve, patch clusters seed `embed-patches`
  frames (scale/orientation per view already known) instead of re-deriving
  everything from detections.
- **Solver track seeds**: feed clusters (not pairs) to a track-native solver.

## Open questions

- Reference-selection policy (largest scale vs self-agreement/centrality) —
  measurable with the existing harness; the format does not constrain it.
- Migration timing for the derived-pairs read path: the consumers that today
  read `image_pairs/` directly (TVG verification, `to-colmap-db`, solve
  ingestion) all need to go through the expansion-aware pairs API before
  `sfm match --cluster` switches its output to cluster-bearing files.
- Whether the operation should also emit a per-cluster fused reference
  template (the 2D analog of the consensus bitmap) for downstream photometric
  gates; deferred until a consumer needs it.
