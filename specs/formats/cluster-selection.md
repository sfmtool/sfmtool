# Cluster Selection

Derives a smaller, self-contained working set from a cluster-backbone
`.matches` file: a predicate over members and clusters that produces a new
cluster-backbone file holding only the surviving subset. The operation is
`MatchesData::select_clusters` in the `matches-format` crate, surfaced in
Python as `MatchesFile.select_clusters`. It is a predicate, not a strategy —
nothing is reordered or ranked; consumers that need an admission order
compute it from the selected file's arrays.

The output is an ordinary `.matches` file whose file-level contract —
provenance record, sentinel scoping, verifiability — is specified in
[matches-file-format.md](matches-file-format.md#cluster-selection-derived-files).
This document specifies the operation itself.

## Options

- `min_span` — the minimum number of distinct selected images a cluster's
  kept members must span (≥ 2, since every written cluster needs ≥ 2 members)
- `restrict_images` — an optional set of image **names**; every requested
  name must exist in the source file
- `restrict_cluster_ids` — an optional set of **source** cluster ids; every
  requested id must be a valid cluster index of the source
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
   `reference_members` global indexes are remapped accordingly.
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

A restriction can drop a cluster's reference member (its image is not
selected) while the cluster itself survives on `min_span` other members. The
derived file does not keep out-of-restriction rows for such references; it
records `reference_members[c] = 0xFFFFFFFF` instead, under the derived-file
sentinel reading scoped by the format specification. The kept members still
carry their absolute positions and absolute shapes, which stay valid without
the reference; only the reference-relative warp becomes unrecoverable.

## Provenance

The operation records its predicate and its source in the derived file's
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

`source_content_xxh128` is the source file's whole-file `content_xxh128`;
`restrict_images` is `null` for an unrestricted selection. A
`restrict_cluster_ids` key holds the requested **source** ids (sorted) and is
present **only** when a cluster-id restriction was requested, so every
selection without one — including all previously written files — is
byte-identical to before the option existed. All other metadata
— including the timestamp — is inherited from the source; the derived file's
own content hashes are computed when it is written. The source file is never
modified.

A selection may itself be selected again — narrowing a working set the caller
already holds, without re-deriving it from the archive. The source is then an
unwritten derivation whose `content_xxh128` is empty, so the record nests the
source's own `cluster_selection` object under `source_selection`:

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

The key is present only when the source carries a `cluster_selection` record,
so a selection of an ordinary file is unchanged. Nesting repeats to any depth,
and the innermost `source_content_xxh128` names the archive the chain started
from.

A selection is a working view, not a replacement archive: non-accepted
members are gone, so per-member evidence for re-gating is absent from the
derived file. Consumers needing it return to the source named by
`source_content_xxh128`.

## Decode accessors

Alongside the selection, the reader exposes the derived quantities consumers
otherwise re-implement:

- member absolute positions — the `member_affines` last column
- member absolute shapes — the `member_affines` leading 2×2 block; the
  reference→member warp, where needed, is `S·S_ref⁻¹` via the cluster's
  reference row
- per-cluster worst consistency — the maximum finite
  `member_consistency_residual` over each cluster's members (`inf` when no
  member has a finite residual)
- `refine_radius` — the refinement patch half-width, normalizing the
  `refine_options` key generations (`patch_size` full edge / 2, legacy
  `radius` as-is)

## Errors

The operation fails on a pairs-backbone source, on `min_span < 2`, on a
`restrict_images` name absent from the source's image table, and on a
`restrict_cluster_ids` id outside the source's cluster range.
