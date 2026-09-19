# Spatial Index Specifications

The shared proximity index and the queries that go through it. Implemented in
`crates/sfmtool-core/src/spatial.rs` and `crates/sfmtool-core/src/spatial/`.

| Document | Description |
|----------|-------------|
| [point-cloud-index.md](point-cloud-index.md) | The KD-tree behind every world-space proximity query: its flat-array interface, what its results guarantee, and why the tree is the bulk-built one. |

The pixel-domain counterpart, `spatial/keypoint_reach.rs`, is specified in
[analysis/keypoint-reach.md](../analysis/keypoint-reach.md) beside the graphs
that consume it.
