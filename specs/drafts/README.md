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
| [kdf-shared-descriptor-reads.md](kdf-shared-descriptor-reads.md) | [core/features/lazy-kdforest-query.md](../core/features/lazy-kdforest-query.md) | Group nonconsecutive candidates by block and replay distances in original order to preserve ties; measure sorting and replay costs. |
| [sift-incremental-extraction-amendment.md](sift-incremental-extraction-amendment.md) | [core/features/sift.md](../core/features/sift.md), [formats/sift-file-format.md](../formats/sift-file-format.md) | A growable `.sift` archive — detect a keypoint pool once, describe it across several commands — and the version-2 on-disk layout that carries it: append-only descriptor chunks, `described_count`, and a stable `feature_set_xxh128` that survives an append. |
| [sift-gpu-amendment.md](sift-gpu-amendment.md) | [core/features/sift.md](../core/features/sift.md) | A `wgpu` compute backend for SIFT's dense stages (blur, DoG, extrema, descriptor), and the output-parity criterion a non-bit-identical backend needs. |
| [move-camera-preview-amendment.md](move-camera-preview-amendment.md) | [gui/edits/move-camera.md](../gui/edits/move-camera.md) | Drawing the observed points where the commit would put them while a camera is being moved by hand, under a measured cost rule that runs the preview every frame or only on rest. |
| [patch-normal-refine-zncc-weighted-selection-amendment.md](patch-normal-refine-zncc-weighted-selection-amendment.md) | [core/patch/patch-normal-refine-view-subset.md](../core/patch/patch-normal-refine-view-subset.md) | Weighting the D-optimal view pick by per-view ZNCC, so the refinement basis stops preferring the most oblique — and photometrically worst — views. |
| [patch-rendering-flat-shaded-amendment.md](patch-rendering-flat-shaded-amendment.md) | [gui/patch-rendering.md](../gui/patch-rendering.md) | Drawing patches that carry a frame but no bitmap as flat-shaded oriented quads, so a reconstruction straight out of `--to-embedded-patches` shows its surfels. |
| [bench-inconsistent-fit-amendment.md](bench-inconsistent-fit-amendment.md) | [core/bench/editable-track.md](../core/bench/editable-track.md), [gui/edits/bundle-adjust.md](../gui/edits/bundle-adjust.md) | A fit that refuses and reports (pairwise baselines and depths, per-observation residuals, a leave-one-out reading) when a track's observations disagree with the camera poses, and the open questions about feeding hand-verified tracks back into a joint solve. |
| [sfm-explorer-editing.md](sfm-explorer-editing.md) | [gui/scene-graph.md](../gui/scene-graph.md), [gui/action-log.md](../gui/action-log.md), [gui/mcp-server.md](../gui/mcp-server.md); files `gui/document-model.md`, `gui/edit-history.md`, `gui/saving.md`, `gui/edits/` | Editing a loaded reconstruction in place under value semantics: a version as a shared base plus its point edits, a per-node history of values with undo/redo and an Edit History panel, saving, edit families, and the wire surface. The umbrella draft for the arc; steps are deleted from it as they ship. |
| [sfm-explorer-editing-overlay.md](sfm-explorer-editing-overlay.md) | Part 4 of [sfm-explorer-editing.md](sfm-explorer-editing.md) | The one question the edited reconstruction in [core/reconstruction/edited-reconstruction.md](../core/reconstruction/edited-reconstruction.md) still leaves open: the materialisation policy, and what it is measured in. |
| [node-transform-from-patch-and-bake.md](node-transform-from-patch-and-bake.md) | [gui/scene-graph.md](../gui/scene-graph.md), [gui/document-model.md](../gui/document-model.md), [gui/edit-history.md](../gui/edit-history.md), [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md), [gui/mcp-server.md](../gui/mcp-server.md), [cli/reconstruction/xform/xform-command.md](../cli/reconstruction/xform/xform-command.md), [GLOSSARY.md](../GLOSSARY.md); file `gui/edits/bake-transform.md` | A context menu on the bench's active patch in the 3D viewport, with four entries that set the node's display transform from that patch's world frame, and a `Bake Transform` entry in the Scene panel that writes the display transform into the reconstruction. The display transform becomes the third half of a version, beside the value and the bench, so undo walks the framing with the geometry; it stays out of the dirty marker, the content hash and every save. Settles the open question of whether a node transform may ever reach the data: it may, through the bake and nowhere else. Corrects the per-image depth statistics `apply_se3_transform` leaves stale under a scale. |
| [sfm-explorer-track-editing.md](sfm-explorer-track-editing.md) | Part 5 of [sfm-explorer-editing.md](sfm-explorer-editing.md); [gui/track-view.md](../gui/track-view.md) | The bench: a per-node place where things are put to be worked on, held in the node's history beside the reconstruction so one Undo walks both, listed in the Scene tree, with one active item per kind the way the reconstruction has a selected point and a selected image, and a bench panel per kind. Its first kind of item, the editable track, edited in Track View: a 2D cluster stage and a 3D track stage, searches that propose observations from a `.kdf` index or a view sweep, measurements that judge them under user-set bars, and one commit that writes it into the reconstruction. The split and merge that fall out. |
