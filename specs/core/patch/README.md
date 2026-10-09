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
| [reference-view.md](reference-view.md) | What each view's tile can contribute to a point's patch bitmap (coverage, clipped share, viewing angle and tilt direction, agreement with the other views over the whole tile and each ninth), and the reference-view rule that picks the one view whose tile is stored as the bitmap, recorded per point in the `.sfmr` column `tracks/reference_observations`. |
| [zncc-self-similarity-radius.md](zncc-self-similarity-radius.md) | How far a patch can slide over itself by whole pixels and still match itself as well as a true match between two views: the ellipse of the shifts that match, whose semi-major axis is the radius, and the ZNCC surface, whole, middle and by ninths, read on the bitmap itself through the overlap reading; the ellipse in grid px, image px and along the patch's axes, with which lengths are only lower bounds. |
| [blur-matched-zncc.md](blur-matched-zncc.md) | The ZNCC of two views' tiles after a tile sharper than the other along every direction (its self-similarity semi-major axis shorter than the other's semi-minor axis) is blurred by a round Gaussian until its semi-major axis reaches the other's semi-minor axis, so a sharp view is counted as disagreeing less for detail a blurrier one lacks without reading grain or a one-direction blur as sharpness: each tile's blur assessment (how its semi-major axis grows when the tile is blurred by two probe widths, read once for each tile some pair blurs) and the blur to a length, the pairing rule the consumers share, the two-pass blur over the samples with data, the skip ratio, the cost, and which consumers read it by default: scoring each observation against a point's stored bitmap, with only the bitmap blurred, is the one that does. A score, never an alignment. |

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
| [keypoint-localization-consensus-basis-measurements.md](keypoint-localization-consensus-basis-measurements.md) | The runs behind the cap's default of eight views on a high-`V` capture: localizer cost, per-observation agreement with the uncapped path, and the downstream size cull and bundle adjustment. |
| [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md) | The per-view render cache and the AVX2 search kernels that make the search affordable. |
| [keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md) | Forward-additive ECC Gauss-Newton subpixel refinement, with an analytic Jacobian. |

## Clusters and tracks

| Document | Description |
|----------|-------------|
| [cluster-patches.md](cluster-patches.md) | Promoting SIFT clusters to patch clusters. |
| [cluster-patch-refinement.md](cluster-patch-refinement.md) | The refinement kernel: a windowed-ZNCC affine cascade from the reference member's patch onto every other member, and the optional piecewise refinement that registers each kept member's nine cells separately, moves the member's shape by the affine map they agree on where the ZNCC does not fall, and stores their displacements. |
| [cluster-patch-refinement-measurements.md](cluster-patch-refinement-measurements.md) | Fleet and subset measurements of the piecewise refinement (agreement with the cascade, cell statuses and gate sweeps, cost, the seed on the two ground truths, with and without the shape-moving loop), the blind human review of the loop's moved shapes, and the cell plane normals against the two checked-in ground truths. |
| [cluster-patch-refinement-human-review-2026-10-09.csv](cluster-patch-refinement-human-review-2026-10-09.csv) | The 90 cases of the blind human review of moved shapes, one row per case: entry, case type, sides, movement, ZNCCs, the reviewer's choice and note, and the one hand-placed footprint. |
| [cell-plane-normals.md](cell-plane-normals.md) | A cluster's patch normal from the stored cell displacements once poses exist: each cell's rays triangulated, a Tukey-weighted plane through the cells, and a verdict naming which of the normal's axes the cells fix. |
| [cluster-warp-consistency.md](cluster-warp-consistency.md) | A reconstruction-free per-member consistency signal: the weak-perspective factorization residual. |
| [member-coherence-validation.md](member-coherence-validation.md) | Pairwise track agreement and the max-support block that decides which members belong. |
| [candidate-track-spawning.md](candidate-track-spawning.md) | Congealing new candidate tracks at offsets from an existing patch frame. |
