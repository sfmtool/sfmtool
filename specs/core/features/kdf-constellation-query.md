# The patch constellation query

Point at a small patch of one photograph and ask which other photographs of the
same capture contain it. The features a SIFT detector found inside that patch
form a little pattern, a constellation: a few dozen keypoints at particular
places relative to one another, each with a descriptor. Another image showing
the same piece of surface holds the same pattern, moved, rotated and scaled by
the change of viewpoint. So the question can be answered by looking each
descriptor up in an index of the whole capture, sorting the hits by which image
they came from, and asking of each image whether its hits agree on one
consistent movement of the pattern. Images whose hits agree contain the patch;
images with a scattering of unrelated lookalikes do not. The answer names each
image that contains the patch, gives the transform that places the patch in it,
and hands back the matched features so a caller can build something on them.

This is the query behind "find this surface again elsewhere". It is the sparse
counterpart of matching a whole image: a patch is a few dozen to a few hundred
descriptors rather than thousands, so it reaches a small part of a large index
and is answerable from a file without loading it. It works over either form of
the [randomized kd-tree forest](randomized-kdtree-forest.md) -- resident in
memory, or read from a [`.kdf` file](../../formats/kdf-file-format.md) through
the [lazy query path](lazy-kdforest-query.md) -- and gives the same answer from
both.

## Rust API

The query lives in
[`kdforest/constellation.rs`](../../../crates/sfmtool-core/src/features/kdforest/constellation.rs),
with the forest trait it is written against in
[`kdforest/neighbor_index.rs`](../../../crates/sfmtool-core/src/features/kdforest/neighbor_index.rs),
both re-exported from `sfmtool_core::features::kdforest` and bound for Python in
[`spatial/constellation.rs`](../../../crates/sfmtool-py/src/spatial/constellation.rs).

```rust
pub trait NeighborIndex<S: ForestScalar> {
    fn dim(&self) -> usize;
    fn search_batch_with_distances(&self, queries: &[S], n_queries: usize, k: usize,
        max_leaf_checks: usize, max_dist: Option<f32>)
        -> Result<(Vec<u32>, Vec<f32>), KdfError>;
    fn resolve_vectors(&self, feature_ids: &[u32]) -> Result<Vec<S>, KdfError>;
}

pub trait FeatureSources {
    fn feature_count(&self) -> usize;
    fn resolve_origins(&self, feature_ids: &[u32]) -> Result<Vec<FeatureOrigin>, KdfError>;
    fn resolve_feature_geometry(&self, feature_ids: &[u32])
        -> Result<Vec<FeatureGeometry>, KdfError>;
    fn image_feature_ids(&self, image_index: u32) -> Result<HashMap<u32, u32>, KdfError>;
}
pub struct ResidentSources { /* origins and geometry in corpus order */ }

pub struct ConstellationParams {
    pub k: usize, pub max_leaf_checks: usize, pub threshold_px: f64,
    pub iterations: usize, pub min_correspondences: usize,
    pub min_inliers: usize, pub seed: u64,
}
pub enum ConstellationDescriptors<'a, S> { Vectors(&'a [S]), FeatureIds(&'a [u32]) }
pub struct Constellation<'a, S> {
    pub positions: &'a [[f32; 2]],
    pub descriptors: ConstellationDescriptors<'a, S>,
    pub image_index: Option<u32>,
}
pub struct ConstellationCorrespondence {
    pub query_index: u32, pub feature_id: u32,
    pub position: [f32; 2], pub affine_shape: [[f32; 2]; 2],
}
pub struct ConstellationMatch {
    pub image_index: u32, pub affine: [[f64; 3]; 2],
    pub inliers: usize, pub correspondences: usize,
    pub inlier_correspondences: Vec<ConstellationCorrespondence>,
}

pub fn constellation_query<S, I, F>(index: &I, sources: &F,
    query: &Constellation<'_, S>, params: &ConstellationParams)
    -> Result<Vec<ConstellationMatch>, KdfError>
where S: ForestScalar, I: NeighborIndex<S> + ?Sized, F: FeatureSources + ?Sized;

pub struct ImageKeypoints { pub positions: Vec<[f32; 2]>,
                            pub affine_shapes: Vec<[[f32; 2]; 2]> }
pub struct QueryImage<'a> { pub sift_path: &'a Path,
                            pub keypoints: Option<&'a ImageKeypoints>,
                            pub image_index: Option<u32> }
pub struct PatchConstellation { pub feature_rows: Vec<u32>,
                                pub feature_ids: Vec<u32>,
                                pub matches: Vec<ConstellationMatch> }

pub fn constellation_at_pixel<I, F>(index: &I, sources: &F, image: &QueryImage<'_>,
    center: [f32; 2], radius: f32, params: &ConstellationParams)
    -> Result<PatchConstellation, KdfError>
where I: NeighborIndex<u8> + ?Sized, F: FeatureSources + ?Sized;
```

**Why an index trait rather than two functions.** The resident `KdForest` and the
file-backed `LazyKdForest` already have the batch search and the vector read-back
this needs, and for one forest they answer them identically: a `.kdf` stores the
topology, leaf order and feature IDs rather than a build seed, so the file-backed
traversal visits the same leaves in the same order. `NeighborIndex` is the two
operations, and nothing else, so the query has one body and the two paths can be
compared through it end to end. A parity test that only compared neighbour lists
would leave everything downstream of them untested; this way the assertion is
that the two forests return the same warps and the same inlier sets.

**Why the sources are a separate argument.** A corpus feature ID is a row number.
Turning it into "feature 412 of image 7, at these pixels, with this affine shape"
takes the origin and geometry tables, and only the file-backed forest has them:
loading a `.kdf` into memory rebuilds the trees and the corpus and keeps no
source tables at all. So `LazyKdForest` implements `FeatureSources` from the file
it holds, a caller of the resident forest passes `ResidentSources` built from the
tables it already has, and the gap is visible in the signature instead of hidden
behind a runtime failure.

**Why one struct per candidate image, carrying its correspondences.** The caller
after this query is building something on the answer: a patch cluster seeded from
the matched features, or a track. It needs the warp to know where to look, the
inlier count to rank and threshold, and the matched features themselves with
their keypoints and affine shapes. Returning the warp alone would send every such
caller back to the index for geometry it has already been read.

**Why `constellation_at_pixel` reads the `.sift` file.** The constellation is
"the features within `radius` of this pixel", which is a question about one
image's keypoints. A `.kdf` stores geometry in corpus storage order, so one
image's keypoints are scattered across every block of the file and selecting a
radius out of them would touch most of it; the image's `.sift` file holds exactly
those keypoints, in one entry. So the keypoints come from the `.sift` file, and
only the handful inside the radius are then turned into descriptors: from the
corpus by feature ID when the image is indexed, and from the `.sift` file's
descriptor entry when it is not. In the indexed case the descriptor payload,
which is the large part of a `.sift` file, is never decompressed.

```rust
use sfmtool_core::features::kdforest::{
    constellation_at_pixel, ConstellationParams, LazyKdForestU8, QueryImage,
};
let forest = LazyKdForestU8::open("capture.kdf".as_ref(), Default::default())?;
let found = constellation_at_pixel(
    &forest,
    &forest,
    &QueryImage { sift_path: "frames/a.jpg.sift".as_ref(),
                  keypoints: None, image_index: Some(0) },
    [812.0, 430.0],
    64.0,
    &ConstellationParams::default(),
)?;
for candidate in &found.matches {
    println!("image {} with {} inliers", candidate.image_index, candidate.inliers);
}
```

## Theory

### A constellation, not a descriptor

One descriptor's nearest neighbour is a guess. SIFT descriptors of different
corners of the same building, of repeated windows, of any texture that recurs,
sit close together in descriptor space, and a 128-dimensional nearest neighbour
found under a check budget is approximate on top of that. There is no test
available to a single lookup that separates the right hit from a lookalike.

A group of features has that test, and it is geometry. If a candidate image
really contains the patch, then the patch's features and their matches in that
image are related by one transform: the pattern is rigid, and the same movement
that takes one feature to its match takes all of them. Lookalikes have no such
relationship, because each one is wherever its own accident put it. So the
question "is this the same patch" becomes "do these correspondences agree on a
transform", which is a question about a group and unanswerable about a single
feature. Everything else here follows from that: the hits are grouped by image
because the transform is per image, and an image with fewer than three of them is
dropped without being fitted because three is where the question starts having an
answer.

### Affine, and three points

Two views of a small planar patch are related exactly by a homography, and a
homography needs four correspondences. An affine transform, six parameters rather
than eight, is the first-order approximation of that homography about the centre
of the patch, and it is the right model here for two reasons. The patch is small,
so the perspective term the affine drops varies little across it and is absorbed
into the residual; and RANSAC's cost is exponential in the sample size. If a
fraction `w` of the correspondences are correct, the chance that a random sample
of `m` is all correct is `w^m`, so at a 20% inlier rate a three-point sample is
clean 0.8% of the time and a four-point sample 0.16%: five times as many trials
for the same confidence. A model that fits slightly worse but is reached five
times sooner wins on a query meant to be interactive.

Three correspondences determine the affine exactly, so each trial solves rather
than fits, and the model is scored by how many of the remaining correspondences
it places within `threshold_px` of where they actually are.

### Refusing a model that collapses the patch

A sample whose three source points are collinear determines no transform, which
every RANSAC has to detect. This query must also refuse the mirror case, where
the three *destination* points are collinear or coincident, and that one is
particular to matching a constellation.

The reason is that several constellation features can hit the same corpus
feature. They are near-duplicates of one another, or of one distinctive spot in
the other image, and `k` neighbours each means a popular corpus feature turns up
in many of the lists. Those correspondences all share one destination point. A
model fitted to three of them maps the entire patch onto that point, and then
scores every correspondence sharing that destination as an inlier -- a consensus
built out of an image that contains nothing of the patch. In a synthetic corpus
of random descriptors this reliably produced double-digit inlier counts on
unrelated images. The test is the determinant of the fitted transform's 2x2
linear part: a warp of a patch into another image is invertible, and a model that
flattens it to a line or a point is not a candidate worth scoring.

### Why `k` is larger than the matcher's

The descriptor matcher in [track-cluster-matching.md](track-cluster-matching.md)
takes 11 neighbours, because it wants each descriptor's own best matches and has
a ratio test to throw away the rest. This query takes 32. A constellation
feature's nearest neighbours are dominated by images that do not contain the
patch, simply because most of a capture does not: with a hundred images and one
patch visible in five of them, a feature's neighbour list is mostly noise no
matter how good the index is. The consensus test downstream removes wrong
candidates at almost no cost, while a candidate that never entered the list
cannot be recovered at any cost, so the asymmetry favours a generous `k`. What
sets the ceiling is the correspondence count per candidate image, which is what
RANSAC's inlier ratio, and so its iteration count, depends on.

### Seeding per candidate image

Each candidate image seeds its own generator from `seed + image_index`, and the
candidates are fitted in ascending image order.

The alternative, one generator threaded through the run, makes each image's
samples depend on how many images preceded it. Then adding a candidate, dropping
one below the correspondence floor, or changing the order they are visited in
silently changes every later fit. Worse for this query specifically: the two
forest paths are compared by running the same query through both, so any
difference in which candidates reached the fitting stage would produce different
warps from identical neighbours, and that difference would read as an index bug
rather than as the scheduling artefact it is. Per-image seeding makes a
candidate's fit a function of its own correspondences and nothing else.

## Implementation notes

Ranking is a stable sort by descending inlier count over candidates already in
ascending image order, so ties break on image identity rather than on the order a
hash map happened to yield.

The lookups are ordered to suit the corpus rather than the caller. Hits are
collected in encounter order, which is neighbour order within each constellation
feature, and origins are resolved for that whole flat list in one call; only the
surviving hits' geometry is then resolved, again in one call. Geometry and
descriptor blocks share the corpus permutation and row boundaries, so a run of
IDs that came back together tends to be one block read in each corpus.

`image_feature_ids` is the one expensive operation here, and it is unavoidable
rather than unconsidered. Origins are stored by corpus feature ID, nothing indexes
them by image, and a corpus may hold any subset of any image, so recovering one
image's feature IDs is a pass over the whole origin table. The pass is chunked at
65,536 IDs so it costs a bounded amount of memory and reads each origin block
once. It is paid once per `constellation_at_pixel` call against an indexed image,
in exchange for never decompressing that image's descriptors.

A feature inside the radius that the corpus does not index is dropped from the
constellation rather than failing the call, so an index built over a subset of a
workspace stays usable.

Reading a `.sift` file can fail, and those failures arrive as `KdfError`: an
unreadable `.sift` and an unreadable `.kdf` are the same problem to a caller, and
a second error type would buy nothing but a second mapping in the bindings. The
I/O kind is preserved, so a missing file still surfaces as a missing file.

## Parameters

All defaults are `ConstellationParams::default()` in
[`constellation.rs`](../../../crates/sfmtool-core/src/features/kdforest/constellation.rs),
and the Python bindings repeat them as their keyword defaults.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | `32` | Neighbours retrieved per constellation feature. |
| `max_leaf_checks` | `128` | Per-query budget of distance evaluations in the forest traversal. |
| `threshold_px` | `8.0` | Reprojection distance, in the candidate image's pixels, within which a correspondence agrees with a model. |
| `iterations` | `200` | Three-point samples drawn per candidate image. |
| `min_correspondences` | `3` | Fewest correspondences before an image is fitted at all; three is also the floor the model needs, so a smaller value has no effect. |
| `min_inliers` | `6` | Fewest inliers for an image to be reported. |
| `seed` | `0` | Base RNG seed; candidate image `i` draws from `seed + i`. |

## Python bindings

Both forest classes carry the same two methods on `sfmtool._sfmtool.spatial`.
`LazyKdForest` answers from its own file; `KdForest` takes a `sources` mapping,
the same one `write_kdf` accepts, because it has no source tables of its own.

```python
from sfmtool._sfmtool.spatial import LazyKdForest

lazy = LazyKdForest("capture.kdf")
found = lazy.constellation_at_pixel(
    "frames/a.jpg.sift", (812.0, 430.0), 64.0, image_index=0, k=32
)
for candidate in found["matches"]:
    print(candidate["image_index"], candidate["inliers"], candidate["affine"])
```

`constellation_query(positions, *, descriptors=None, feature_ids=None,
image_index=None, ...)` takes `(N, 2)` float32 positions and either an
`(N, D)` uint8 descriptor array or the corpus feature IDs, and returns a list of
dicts whose keys are the Rust field names: `image_index`, `affine` (a `(2, 3)`
float64 array), `inliers`, `correspondences` and `inlier_correspondences`. That
last is a dict of columns rather than a list of per-row dicts, keyed by the
correspondence field names: `query_index` and `feature_id` as `(M,)` uint32,
`position` as `(M, 2)` float32 and `affine_shape` as `(M, 2, 2)` float32. They
are consumed as arrays, and a few hundred single-row dicts would cost more to
build than the query does to answer.

`constellation_at_pixel(sift_path, center, radius, *, image_index=None, ...)`
returns one dict with `feature_rows` (the `.sift` rows the constellation was
built from), `feature_ids` (their corpus IDs, empty when the image is not
indexed) and `matches`. The `KdForest` forms take `sources` as an extra
positional argument, after `positions` and after `radius` respectively.

A budget too small for what was asked raises `MemoryError`, a damaged file
raises `OSError`, and a bad argument raises `ValueError`, matching the rest of
the `.kdf` surface. A corpus carrying no SIFT sources is a `ValueError`: its
features have no image to be grouped by.

[`scripts/kdf_patch_localize.py`](../../../scripts/kdf_patch_localize.py)
localizes patches through both paths and asserts they agree, which is where the
measurements in
[lazy-kdforest-query.md](lazy-kdforest-query.md#two-real-access-patterns-and-which-one-this-path-suits)
come from.

## Testing

[`constellation/tests.rs`](../../../crates/sfmtool-core/src/features/kdforest/constellation/tests.rs)
builds a five-image synthetic corpus: a query image, an image holding every one
of its patch features under a known affine warp, two images of unrelated random
descriptors, and one holding four of the patch's features under a different warp.
Over it:

- `the_planted_image_wins_with_the_planted_warp` asserts the warped image ranks
  first, that its inlier count is exactly the planted feature count, and that the
  recovered transform reproduces the planted one at every inlier.
- `the_query_image_and_the_thin_candidate_are_absent` asserts the query image is
  not a candidate, that the four-feature image falls below `min_inliers`, and
  that with the exclusion turned off the query image is the strongest candidate
  of all, which is what makes excluding it worth doing.
- `the_two_forests_answer_identically` writes the forest to a `.kdf` with a few
  hundred bytes per chunk and block, so the traversal crosses many boundaries,
  and asserts the resident and file-backed results are equal in full: same order,
  same inlier counts, same correspondences, same warps bit for bit. It asserts
  the same for a query handed descriptors directly rather than feature IDs.
- `a_pixel_and_a_radius_find_the_patch_through_the_sift_file` writes the query
  image as a real `.sift` file and localizes a radius around one of its
  keypoints, indexed and unindexed, and with the keypoints supplied rather than
  read.

The corpus uses a wide canvas deliberately: the odds of a wrong correspondence
landing inside an eight-pixel threshold scale with the inverse of the image area,
and an exact inlier count is only assertable when they are negligible.

[`tests/rust_bindings/test_kdf_constellation_rust_bindings.py`](../../../tests/rust_bindings/test_kdf_constellation_rust_bindings.py)
extracts SIFT once from the included Seoul Bull image and indexes those
descriptors twice, the second copy at warped positions as a second image. It
checks the dict surface key by key, that the recovered warp is the planted one,
that reported positions match the geometry the sources carry, that both forest
classes return the same thing through both entry points and both descriptor
forms, and that a `.kdf` written without sources is a `ValueError`.

## Non-goals

- No homography, and no refinement of the affine from its inliers. The transform
  is the best three-point model, not a least-squares fit to the consensus; a
  caller wanting a refined warp fits one from `inlier_correspondences`.
- No ratio test or other per-descriptor filtering before grouping. The consensus
  is the filter, and a ratio test would discard the repeated-texture matches that
  a constellation is able to keep.
- No scoring of a candidate image beyond its inlier count. Photometric agreement
  belongs to the patch refinement a caller seeds from this result.
- The `.sift` entry point is `u8` descriptors only, because that is what a
  `.sift` file holds. `constellation_query` itself is generic over the forest's
  scalar.
