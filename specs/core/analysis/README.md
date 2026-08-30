# Analysis Specifications

Graphs and measurements computed over a reconstruction or a `.matches` cluster
backbone. Implemented in `crates/sfmtool-core/src/analysis/`.

| Document | Description |
|----------|-------------|
| [image-pair-graph.md](image-pair-graph.md) | Which image pairs are worth matching: covisibility from shared tracks, frustum intersection from geometry alone. |
| [observation-adjacency-graph.md](observation-adjacency-graph.md) | Per-image neighbourhoods over observations in image space, the substrate the surfel and coverage passes query. |
| [observation-coverage.md](observation-coverage.md) | Which image pixels existing tracks already claim, so new candidates can be spawned where nothing is. |
| [adjacency-surfel-normals.md](adjacency-surfel-normals.md) | Surfel normals from a robust plane fit over each point's image-space neighbours, with no photometric refinement. |
| [cluster-census.md](cluster-census.md) | Match census over a cluster backbone: per-pair statistics, viewpoint groups, saturation, and the score that ranks them. |
| [reconstruction-alignment.md](reconstruction-alignment.md) | Least-squares similarity fit between two reconstructions, with RANSAC outlier rejection. |
| [source-clusters.md](source-clusters.md) | Which clusters of a selection a member's admission never held, joined by feature identity and banded by radius. |
