# Feature Specifications

Feature extraction and matching. Implemented in
`crates/sfmtool-core/src/features/`, with the pipeline-level specs driving it
from `src/sfmtool/feature_match/`. Cluster selection, which narrows the
cluster backbone that cluster matching writes, is implemented in the
`sfmtool-matches-format` crate instead.

| Document | Description |
|----------|-------------|
| [sift.md](sift.md) | The pure-Rust SIFT detector and descriptor: scale space, orientation, SIMD, and threading. |
| [randomized-kdtree-forest.md](randomized-kdtree-forest.md) | Approximate nearest-neighbour index replacing the exhaustive descriptor scan. |
| [lazy-kdforest-query.md](lazy-kdforest-query.md) | Persistent, bounded-cache queries over chunked `.kdf` forests. |
| [lazy-kdforest-query-measurements.md](lazy-kdforest-query-measurements.md) | The measurements that chose the lazy query path's layout, block, chunk, leaf and cache defaults, and which workloads a `.kdf` suits. |
| [kdf-layout-measurements.md](kdf-layout-measurements.md) | The measurements that chose the `.kdf` layout: file sizes under tree-local and shared descriptor storage, and why the corpus is compressed in blocks. |
| [kdf-constellation-query.md](kdf-constellation-query.md) | Which other images contain the patch around a pixel, by affine consensus over a descriptor index. |
| [track-cluster-matching.md](track-cluster-matching.md) | Matching a whole image set at once: every image's SIFT descriptors clustered into candidate tracks, each descriptor's radius set from its own background floor. Verification is a separate step. |
| [descriptor-matching.md](descriptor-matching.md) | Matching an image pair with known poses: a sweep along the epipolar lines (rectified or polar), mutual nearest descriptors, and an optional orientation and size filter. Used by `sfm densify`. |
| [cluster-selection.md](cluster-selection.md) | Deriving a smaller, self-contained `.matches` working set from a cluster-backbone file: the clusters and members that pass a predicate on member status, image name, source cluster id and image span (`MatchesData::select_clusters`). |
| [cluster-covisibility.md](cluster-covisibility.md) | How many clusters each image pair shares, and the grouping queries consumers build on that. |
| [covisibility-selection.md](covisibility-selection.md) | Three primitives over that structure: appearance displacement, redundancy thinning, and reach. |
| [optical-flow.md](optical-flow.md) | Pure-Rust DIS dense optical flow on the CPU, used for flow-based matching, motion analysis of image sequences and `sfm flow`. |
| [gpu-optical-flow.md](gpu-optical-flow.md) | The wgpu compute-shader implementation of the same DIS pipeline. |
| [flow-based-matching.md](flow-based-matching.md) | Matching driven by optical flow instead of descriptor search (`sfm match --flow`). Python pipeline. |
