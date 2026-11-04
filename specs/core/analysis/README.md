# Analysis Specifications

Graphs and measurements computed over a reconstruction, a `.matches` cluster
backbone or the SIFT features of two video frames. Implemented in
`crates/sfmtool-core/src/analysis/`.

| Document | Description |
|----------|-------------|
| [image-pair-graph.md](image-pair-graph.md) | Which image pairs are worth matching: covisibility from shared tracks, frustum intersection from geometry alone. |
| [observation-adjacency-graph.md](observation-adjacency-graph.md) | Per-image neighbourhoods over observations in image space, the substrate the surfel and coverage passes query. |
| [keypoint-reach.md](keypoint-reach.md) | Per-image candidate pairs within each keypoint's own pixel reach: the enumeration under containment and same-measurement rules. |
| [covered-by-finer.md](covered-by-finer.md) | Retiring a coarse observation a finer one, on another owner, covers in the same image, and dropping the owners left with too few. |
| [observation-coverage.md](observation-coverage.md) | Which image pixels existing tracks already claim, so new candidates can be spawned where nothing is. |
| [adjacency-surfel-normals.md](adjacency-surfel-normals.md) | Surfel normals from a robust plane fit over each point's image-space neighbours, with no photometric refinement. |
| [track-table.md](track-table.md) | The track table: named per-point values (track length, error, ray angle, triangulation diagnostics, …) for every point of a reconstruction, one row per track keyed by point index and computed on demand, and the query that filters and sorts points by them. |
| [cluster-census.md](cluster-census.md) | Match census over a cluster backbone: per-pair statistics, viewpoint groups, global satisfaction, and the score that ranks them. |
| [reconstruction-alignment.md](reconstruction-alignment.md) | Least-squares similarity fit between two reconstructions, with RANSAC outlier rejection. |
| [pick-images.md](pick-images.md) | Choosing the frames of a video a solve should use: a baseline sample that judges whether each sequence is a video and whether the capture can be solved, then a walk that measures each step by SIFT matching, swaps blurred frames, fills in bad patches and records breaks. |
| [frame-pair-matching.md](frame-pair-matching.md) | The motion between two frames of a video from their SIFT features: the step, the share of matches that fit one camera motion, the share of those that fit one homography, and how close that homography is to a pure rotation. |
| [source-clusters.md](source-clusters.md) | Which clusters of a selection a member's admission never held, joined by feature identity and banded by radius; the per-cluster radius reading and the coarsest-N cut over it. |
