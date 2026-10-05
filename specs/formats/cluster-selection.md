# Cluster Selection

Cluster selection takes the contents of a cluster-backbone `.matches` file and
produces a new, writable set of contents holding only the clusters and members
that pass a predicate on member status, image name, source cluster id and the
number of distinct images a cluster spans. Consumers use it to get a smaller,
self-contained working set, such as the clusters seen in a chosen group of
images, without changing the source file. It is a filter: nothing is
reordered or ranked, and a consumer that needs an admission order computes it
from the selected arrays.

The output is an ordinary `.matches` file whose file-level contract —
provenance record, sentinel scoping, verifiability — is specified in
[matches-file-format.md](matches-file-format.md#cluster-selection-derived-files).
This document specifies the operation itself.

## Interface

The operation is `MatchesData::select_clusters(&ClusterSelect)` in
[`select.rs`](../../crates/sfmtool-matches-format/src/select.rs) of the
`sfmtool-matches-format` crate. It returns a new `MatchesData` and leaves the
source untouched; writing the result is the caller's choice. Python reaches it
as `MatchesFile.select_clusters` in
[`matches_file.rs`](../../crates/sfmtool-py/src/io/matches_file.rs), which
returns a new `MatchesFile` handle. `select.rs` also holds the decode
accessors described [below](#decode-accessors).

## Options

- `min_span` — the minimum number of distinct selected images a cluster's
  kept members must span (≥ 2, since every written cluster needs ≥ 2 members)
- `restrict_images` — an optional set of image **names**; every requested
  name must exist in the source file, and a repeated name counts once
- `restrict_cluster_ids` — an optional set of **source** cluster ids; every
  requested id must be a valid cluster index of the source, and a repeated id
  counts once
- `accepted_statuses` — the member statuses that survive (default
  `reference` + `kept`); ignored when the source has no `cluster_patches/`
  section (every member is then a candidate)

## Semantics

Applied in order:

1. Clusters whose `reference_members` entry is `0xFFFFFFFF` in the source are
   dropped (only when the source carries `cluster_patches/`).
2. When cluster-restricted, clusters whose source id is not in
   `restrict_cluster_ids` are dropped. The id restriction composes with the
   image restriction; each applies its own axis.
3. Per cluster, a member is kept iff its status is accepted **and**, when
   restricted, its image is in the restriction. Restriction happens before
   the span test, so span counts distinct **selected** images.
4. A cluster survives iff its kept members span ≥ `min_span` distinct
   selected images.
5. Surviving clusters and members are densely renumbered in source order
   (cluster order and within-cluster member order are preserved), and
   `reference_members` global indexes are remapped accordingly. Every
   member-parallel array is gathered by the same survival mask — the
   backbone's `member_positions` / `member_affine_shapes` included — so a
   selection is itself a writable cluster file whose members keep the values,
   and the stage, the source gave them.
6. When image-restricted, the image table becomes **exactly** the requested
   set, in source file order: requested images keep their row even if no
   member references them, all other images are dropped, and every parallel
   image array (`names`, `feature_tool_hashes`, `sift_content_hashes`,
   `feature_counts`, `image_dims`) plus `clusters/member_images` is
   renumbered consistently. A cluster-id restriction alone leaves the image
   table untouched.

Step 5 renumbers densely, so a cluster id is meaningful only inside the
selection that produced it. A consumer holding per-cluster evidence computed
against the source applies it **before** the selection, or carries it as a
per-cluster attribute compressed by the same survival mask; the derived file
records no cross-numbering correspondence.

## Absent references

A cluster can lose its reference member in two ways: an image
restriction drops it because its image is not selected, or `accepted_statuses`
leaves out `reference`, which drops every reference member. The cluster still
survives when its other kept members span `min_span` images. The derived file
does not keep the dropped reference; it records `reference_members[c] =
0xFFFFFFFF` instead, under the derived-file sentinel reading scoped by the
format specification. The kept members still carry their absolute positions
and absolute shapes, which stay valid without the reference; only the
reference-relative warp becomes unrecoverable.

## Provenance

The operation records its source and the options it was called with in the
derived file's top-level metadata, as the provenance record under
`matching_options["cluster_selection"]`. The record's keys, an example, and the
`source_selection` nesting are defined once, in
[matches-file-format.md](matches-file-format.md#cluster-selection-derived-files).

A selection may itself be selected again, narrowing a working set the caller
already holds without re-deriving it from the archive. The source is then an
unwritten derivation whose `content_xxh128` is empty, which is why the record
nests the source's own record: the chain still names the archive it started
from. A `restrict_cluster_ids` key is written only when a cluster-id
restriction was requested, so a selection without one writes the same record
as before that option existed. The counts, the section flags and the format
version describe the derived file; all other metadata, the timestamp included,
is inherited from the source. The derived file's own content hashes are
computed when it is written. The source file is never modified.

A selection is a working view, not a replacement archive: non-accepted
members are gone, so per-member evidence for re-gating is absent from the
derived file. Consumers needing it return to the source named by
`source_content_xxh128`.

## Decode accessors

Alongside the selection, the reader exposes the derived quantities consumers
otherwise re-implement:

- member positions and member affine shapes — the backbone's own arrays, whose
  content is the file's stage (detections in a matcher output, the refinement's
  answer where its cascade measured in a cluster-patches one). The
  reference→member warp, where needed, is `S·S_ref⁻¹` via the cluster's
  reference member's shape
- per-cluster worst consistency — the maximum finite
  `member_consistency_residual` over each cluster's members (`inf` when no
  member has a finite residual). The residuals live in `cluster_patches/`, so
  a file without that section has none: the Rust accessor returns `None` and
  the Python one raises `ValueError`
- `refine_radius` — the refinement patch half-width, read from either
  `refine_options` key by the rule under
  [`cluster_patches/metadata.json.zst`](matches-file-format.md#cluster_patchesmetadatajsonzst);
  `None` when `refine_options` holds neither key as a number

## Errors

The operation fails on a pairs-backbone source, on `min_span < 2`, on a
`restrict_images` name absent from the source's image table, and on a
`restrict_cluster_ids` id outside the source's cluster range.
