# Feature Specifications

Feature extraction and matching. Implemented in
`crates/sfmtool-core/src/features/`, with the pipeline-level specs driving it
from `src/sfmtool/feature_match/`.

| Document | Description |
|----------|-------------|
| [sift.md](sift.md) | The pure-Rust SIFT detector and descriptor: scale space, orientation, SIMD, and threading. |
| [randomized-kdtree-forest.md](randomized-kdtree-forest.md) | Approximate nearest-neighbour index replacing the exhaustive descriptor scan. |
| [track-cluster-matching.md](track-cluster-matching.md) | Cluster-centric alternative to pair-centric matching: build track clusters directly, verify afterwards. |
| [cluster-covisibility.md](cluster-covisibility.md) | How many clusters each image pair shares, and the grouping queries consumers build on that. |
| [covisibility-selection.md](covisibility-selection.md) | Three primitives over that structure: appearance displacement, redundancy thinning, and reach. |
| [optical-flow.md](optical-flow.md) | Pure-Rust DIS dense optical flow on the CPU, used as a candidate track generator. |
| [gpu-optical-flow.md](gpu-optical-flow.md) | The wgpu compute-shader implementation of the same DIS pipeline. |
| [flow-based-matching.md](flow-based-matching.md) | Matching driven by optical flow instead of descriptor search (`sfm match --flow`). Python pipeline. |
