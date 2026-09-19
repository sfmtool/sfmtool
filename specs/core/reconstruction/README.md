# Reconstruction Specifications

Operations on reconstruction data itself. Implemented in
`crates/sfmtool-core/src/reconstruction/`.

| Document | Description |
|----------|-------------|
| [edited-reconstruction.md](edited-reconstruction.md) | A reconstruction as a value: the image table, the point set, the two heavy columns they share, and the edited form that is a shared immutable base plus its point edits, with materialisation and the row map. |
| [move-camera.md](move-camera.md) | Putting one image at a pose a caller names, and settling the tracks it observes around it: the ones two or more pixels see re-triangulated, a bearing only it sees turned with the camera, everything else kept, and a residual pair measured before and after. |
| [bundle-adjust.md](bundle-adjust.md) | Bundle-adjusting a whole reconstruction: what goes into the shared-camera kernel, the constraints and representations it honours, the points it leaves unsupported and deletes, and the patch frames it rescales with their depth. |
| [batch-triangulation-api.md](batch-triangulation-api.md) | Batch triangulation carrying per-point observability diagnostics, and the classifier over them. |
| [point-estimation.md](point-estimation.md) | Re-reading every point from its observations at one geometry, with the per-track rules (floor, cheirality, bar, few) held once for every caller. |
| [point-correspondence.md](point-correspondence.md) | Finding the same 3D point across two reconstructions, and merging their tracks. |
