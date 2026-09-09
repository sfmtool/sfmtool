# Editing an in-memory reconstruction: what the overlay leaves open

**Status:** Draft

Answers Part 4 of [`sfm-explorer-editing.md`](sfm-explorer-editing.md), the
umbrella draft for editing a loaded reconstruction in place, and holds what is
left of that answer after the core and the first track edit landed.

The representation itself is built and standing:
[`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)
describes the base plus its edits, the delete-and-re-add rule for point edits,
the stable indexes, the accessor that reads one point without materialising, the
materialisation that keeps every point in its place with its row map, and the
base and point-edit hashes. The GPU side is built and standing too:
[`../gui/document-model.md`](../gui/document-model.md), "Change detection by
identity", covers the base's identity, the deleted mask, the additions'
instance buffers and their second patch atlas, and the pick range that spans
both;
[`../gui/point-cloud-rendering.md`](../gui/point-cloud-rendering.md) and
[`../gui/patch-rendering.md`](../gui/patch-rendering.md) carry the same answer
at the two passes. So is the file side: an added observation has a keypoint and
a patch and no feature index, so the track edits are built on
`embedded_patches` reconstructions and refuse on `sift_files`, which is
[`../gui/edits/add-observation.md`](../gui/edits/add-observation.md)'s first
refusal. The point-id version graph, with its minting and resolution rules, is
[`../gui/goto-point.md`](../gui/goto-point.md) § "The ID forms and the version
graph", [`../gui/edit-history.md`](../gui/edit-history.md) § "The version graph",
and
[the format spec's Point ID and Lineage sections](../formats/sfmr-file-format.md#point-id-portable-3d-point-references).

What is left here is one question.

---

## Open questions

- **The materialisation policy.** The overlay is materialised when a bulk edit,
  a save or a size threshold asks for it, and the first two are decided by the
  caller. The third is not: what the threshold is measured in -- points,
  observations, or unshared bytes -- and where it sits. Deliberately not decided
  until reconstructions are being edited live and the overlay's read cost can be
  measured against the materialisation's on them.

- **Whether a `sift_files` reconstruction should be able to carry an observation
  with no feature behind it.** That is a format decision, and it is what would
  let the track edits work on the other half of the corpus. Not made here.
