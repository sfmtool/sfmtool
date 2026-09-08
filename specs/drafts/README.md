# Draft Specifications

Proposals, and the parts of a standing spec that are specified but not built.
Nothing here describes code that exists.

A draft opens with `**Status:** Draft` and may use future tense and construction
language freely, because it is addressed to the people deciding. Everything
outside this directory is a standing spec: it describes what the code **is**, in
the present tense, and carries no status line — see
[../TEMPLATE.md](../TEMPLATE.md).

**Amendment drafts** are the common case. When a standing spec covers something
the code does not do, the spec states that in one present-tense sentence and
links the draft that proposes it; the draft opens by naming and linking the spec
it amends. The two point at each other, so neither can be read as describing the
other's world. When an amendment ships, its content folds into the standing spec
and the draft is deleted.

Filing a whole draft is three edits: delete the `**Status:**` line, lift the
purpose paragraph to the top, and convert the document to the present tense. Then
move the file into the area directory it belongs to and add its row to that
area's `README.md`.

| Document | Amends | Proposes |
|----------|--------|----------|
| [sift-incremental-extraction-amendment.md](sift-incremental-extraction-amendment.md) | [core/features/sift.md](../core/features/sift.md), [formats/sift-file-format.md](../formats/sift-file-format.md) | A growable `.sift` archive — detect a keypoint pool once, describe it across several commands — and the version-2 on-disk layout that carries it: append-only descriptor chunks, `described_count`, and a stable `feature_set_xxh128` that survives an append. |
| [sift-gpu-amendment.md](sift-gpu-amendment.md) | [core/features/sift.md](../core/features/sift.md) | A `wgpu` compute backend for SIFT's dense stages (blur, DoG, extrema, descriptor), and the output-parity criterion a non-bit-identical backend needs. |
| [patch-normal-refine-zncc-weighted-selection-amendment.md](patch-normal-refine-zncc-weighted-selection-amendment.md) | [core/patch/patch-normal-refine-view-subset.md](../core/patch/patch-normal-refine-view-subset.md) | Weighting the D-optimal view pick by per-view ZNCC, so the refinement basis stops preferring the most oblique — and photometrically worst — views. |
| [patch-rendering-flat-shaded-amendment.md](patch-rendering-flat-shaded-amendment.md) | [gui/patch-rendering.md](../gui/patch-rendering.md) | Drawing patches that carry a frame but no bitmap as flat-shaded oriented quads, so a reconstruction straight out of `--to-embedded-patches` shows its surfels. |
| [sfm-explorer-editing.md](sfm-explorer-editing.md) | [gui/scene-graph.md](../gui/scene-graph.md), [gui/action-log.md](../gui/action-log.md), [gui/mcp-server.md](../gui/mcp-server.md); files `gui/document-model.md`, `gui/edit-history.md`, `gui/saving.md`, `gui/edits/` | Editing a loaded reconstruction in place under value semantics: a version as a shared base plus its point edits, a per-node history of values with undo/redo and a History panel, saving, edit families, and the wire surface. The umbrella draft for the arc; steps are deleted from it as they ship. |
| [sfm-explorer-editing-overlay.md](sfm-explorer-editing-overlay.md) | Part 4 of [sfm-explorer-editing.md](sfm-explorer-editing.md); amends [gui/scene-graph.md](../gui/scene-graph.md), [gui/goto-point.md](../gui/goto-point.md), [formats/sfmr-file-format.md](../formats/sfmr-file-format.md) | What the edited reconstruction in [core/reconstruction/edited-reconstruction.md](../core/reconstruction/edited-reconstruction.md) leaves open: how the GPU consumes a deleted mask and an additions buffer, what an added observation means on a `sift_files` file, and the point-id version graph, whose ids are content-derived and minted against the earliest base or edit a point identity reaches. |
