# Patch Specifications

Oriented patches: the geometry that renders one 3D point's surface consistently
across views, and everything refined on top of it. Implemented in
`crates/sfmtool-core/src/patch/`, driven by the `sfm embed-patches` and
`sfm cluster-patches` pipelines in `src/sfmtool/`.

## Foundations

| Document | Description |
|----------|-------------|
| [patch-cloud.md](patch-cloud.md) | Oriented patches and the patch-projected warp maps that render one point's surface the same way in every view. |
| [sift-to-patch-reconstruction.md](sift-to-patch-reconstruction.md) | The `sfm embed-patches` pipeline: converting SIFT-referencing observations into embedded patches. Python pipeline. |
| [patch-view-selection.md](patch-view-selection.md) | Which views photometrically see a point's patch. |
| [zncc-self-similarity-radius.md](zncc-self-similarity-radius.md) | How far a patch can slide over itself by whole pixels and still match itself as well as a true match between two views: the radius, its direction and the ZNCC surface, whole, middle and by ninths, read with a ring around the patch or, for a stored bitmap, over the overlap; and the reach of the contour the radius is read from, in grid px, image px and along the patch's axes, with which values are only lower bounds. |

## Normal refinement

| Document | Description |
|----------|-------------|
| [patch-normal-refinement.md](patch-normal-refinement.md) | Photometric refinement of a patch's surface normal. |
| [patch-normal-refine-view-subset.md](patch-normal-refine-view-subset.md) | The D-optimal view subset that makes that refinement cheap without losing conditioning. |
| [fronto-parallel-patch-cache.md](fronto-parallel-patch-cache.md) | The render-once fronto-parallel cache backing normal refinement, and when it is exact enough. |

## Keypoint localization

| Document | Description |
|----------|-------------|
| [patch-keypoint-localization.md](patch-keypoint-localization.md) | Congealing: refining a point's keypoint position across all its views jointly. |
| [keypoint-localization-consensus-basis.md](keypoint-localization-consensus-basis.md) | The consensus-basis cap — basis congealing, then tail registration against it. |
| [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md) | The per-view render cache and the AVX2 search kernels that make the search affordable. |
| [keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md) | Forward-additive ECC Gauss-Newton subpixel refinement, with an analytic Jacobian. |

## Clusters and tracks

| Document | Description |
|----------|-------------|
| [cluster-patches.md](cluster-patches.md) | Promoting SIFT clusters to patch clusters. |
| [cluster-patch-refinement.md](cluster-patch-refinement.md) | The refinement kernel: a windowed-ZNCC affine cascade from the reference member's patch onto every other member. |
| [cluster-warp-consistency.md](cluster-warp-consistency.md) | A reconstruction-free per-member consistency signal: the weak-perspective factorization residual. |
| [member-coherence-validation.md](member-coherence-validation.md) | Pairwise track agreement and the max-support block that decides which members belong. |
| [candidate-track-spawning.md](candidate-track-spawning.md) | Congealing new candidate tracks at offsets from an existing patch frame. |
