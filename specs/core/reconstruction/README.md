# Reconstruction Specifications

Operations on reconstruction data itself. Implemented in
`crates/sfmtool-core/src/reconstruction/`.

| Document | Description |
|----------|-------------|
| [edited-reconstruction.md](edited-reconstruction.md) | A reconstruction as a value: the image table, the point set, the two heavy columns they share, and the edited form that is a shared immutable base plus its point edits, with materialisation and the row map. |
| [add-observation.md](add-observation.md) | Adding one observation to a track from a pixel: the clicked pixel registered photometrically against the point's stored patch by the embed pass's own kernel, and the track re-triangulated with the new sighting in it. |
| [create-point.md](create-point.md) | Creating a 3D point from a pixel: a bearing along that pixel's ray at `w = 0`, with one observation, a patch frame at a caller-named angular size and a bitmap cut from the photograph. |
| [batch-triangulation-api.md](batch-triangulation-api.md) | Batch triangulation carrying per-point observability diagnostics, and the classifier over them. |
| [point-estimation.md](point-estimation.md) | Re-reading every point from its observations at one geometry, with the per-track rules (floor, cheirality, bar, few) held once for every caller. |
| [point-correspondence.md](point-correspondence.md) | Finding the same 3D point across two reconstructions, and merging their tracks. |
