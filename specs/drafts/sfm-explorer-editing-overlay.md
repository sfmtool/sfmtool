# Editing an in-memory reconstruction: what the overlay leaves open

**Status:** Draft

Answers Part 4 of [`sfm-explorer-editing.md`](sfm-explorer-editing.md), the
umbrella draft for editing a loaded reconstruction in place, and holds what is
left of that answer after the core landed.

The representation itself is built and standing:
[`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)
describes the base plus its edits, the delete-and-re-add rule for point edits,
the stable indexes, the accessor that reads one point without materialising,
the materialisation that keeps every point in its place with its row map, and
the base and point-edit hashes. This draft keeps only what that spec does not
decide: how the GPU side consumes an overlay, what a save of an addition means
on a `sift_files` reconstruction, and the open questions. The point-id version
graph, with its minting and resolution rules, is built and standing:
[`../gui/goto-point.md`](../gui/goto-point.md) § "The ID forms and the version
graph", [`../gui/edit-history.md`](../gui/edit-history.md) § "The version
graph", and
[the format spec's Point ID and Lineage sections](../formats/sfmr-file-format.md#point-id-portable-3d-point-references).

---

## The GPU side

Built and standing: [`../gui/document-model.md`](../gui/document-model.md),
"Change detection by identity". The base's buffers keep their identity, so the
upload sees no change on the base and uploads nothing for it, and the deleted
set reaches the point and patch shaders as a per-instance mask written only
where the set moved.

What that leaves open is the half an addition needs. The additions upload as a
second instance buffer drawn after the base's, with the same per-node uniforms.
The patch atlas is the one piece of GPU state the stable-index rule does not
already cover: its slot assignment is a compaction over the points that carry a
bitmap, so a point's atlas slot is not its index, and a point edit that changes
a bitmap needs the additions' patches in a second atlas, or a slot map the
base's atlas keeps across edits. The bundle already keeps a point-index-to-slot
map for the mask, which is the smaller half of that question. A materialisation
replaces both buffers with one, through the row map. No edit re-uploads a
million points.

## The file side

A save materialises, then writes as any writer does. An observation added from
a clicked pixel has a keypoint and a patch but no feature index, so the first
track edits are built on `embedded_patches` reconstructions, where that is
what an observation is. On a `sift_files` reconstruction such an edit refuses
until the format carries an observation without a feature; that format
decision is not made here.

## Open questions

- The materialisation fraction, and whether it is measured in points, in
  observations, or in unshared bytes.
