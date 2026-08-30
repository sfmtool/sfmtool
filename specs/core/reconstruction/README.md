# Reconstruction Specifications

Operations on reconstruction data itself. Implemented in
`crates/sfmtool-core/src/reconstruction/`.

| Document | Description |
|----------|-------------|
| [batch-triangulation-api.md](batch-triangulation-api.md) | Batch triangulation carrying per-point observability diagnostics, and the classifier over them. |
| [point-estimation.md](point-estimation.md) | Re-reading every point from its observations at one geometry, with the per-track rules (floor, cheirality, bar, few) held once for every caller. |
| [point-correspondence.md](point-correspondence.md) | Finding the same 3D point across two reconstructions, and merging their tracks. |
