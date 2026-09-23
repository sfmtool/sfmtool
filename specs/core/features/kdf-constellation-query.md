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
[`spatial/constellation_query.rs`](../../../crates/sfmtool-py/src/spatial/constellation_query.rs).

```rust
pub trait NeighborIndex<S: ForestScalar> {
    fn dim(&self) -> usize;
    fn search_batch_with_distances(&self, queries: &[S], n_queries: usize, k: usize,
        max_leaf_checks: usize, max_dist: Option<f32>)
        -> Result<(Vec<u32>, Vec<f32>), KdfError>;
    fn resolve_descriptors(&self, feature_ids: &[u32]) -> Result<Vec<S>, KdfError>;
}

pub trait FeatureSources {
    fn feature_count(&self) -> usize;
    fn resolve_origins(&self, feature_ids: &[u32]) -> Result<Vec<FeatureOrigin>, KdfError>;
    fn resolve_feature_geometry(&self, feature_ids: &[u32])
        -> Result<Vec<FeatureGeometry>, KdfError>;
    fn image_feature_ids(&self, image_index: u32) -> Result<HashMap<u32, u32>, KdfError>;
}
pub struct ResidentSources { /* origins and geometry in corpus order */ }

pub enum AffineRefit {
    None,                            // report the three-point model as drawn
    LeastSquares,                    // every inlier weighted alike
    CenterWeighted { sigma: f64 },   // Gaussian in the distance from the centre
}
pub struct ConstellationParams {
    pub k: usize, pub max_leaf_checks: usize, pub threshold_px: f64,
    pub iterations: usize, pub min_correspondences: usize,
    pub one_hit_per_image: bool, pub same_image_ratio: f32,
    pub min_inliers: usize, pub max_scale: f64,
    pub refit: AffineRefit, pub seed: u64,
}
impl ConstellationParams { pub const DEFAULT: Self; }
pub enum ConstellationDescriptors<'a, S> { Vectors(&'a [S]), FeatureIds(&'a [u32]) }
pub struct Constellation<'a, S> {
    pub positions: &'a [[f32; 2]],
    pub descriptors: ConstellationDescriptors<'a, S>,
    pub image_index: Option<u32>,
    pub center: Option<[f32; 2]>,
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

pub fn constellation_from_keypoints<I, F>(index: &I, sources: &F,
    keypoints: &ImageKeypoints, image_index: u32,
    center: [f32; 2], radius: f32, params: &ConstellationParams)
    -> Result<PatchConstellation, KdfError>
where I: NeighborIndex<u8> + ?Sized, F: FeatureSources + ?Sized;

pub fn radius_for_feature_count(image_width: u32, image_height: u32,
    keypoint_count: usize, target: usize) -> f32;
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

**Why there is a second entry point for a caller holding the keypoints.**
`constellation_from_keypoints` is `constellation_at_pixel`'s indexed half with
the file read taken out: the keypoints are the caller's and the image is one the
corpus indexes, so nothing opens a `.sift` file at all. That is the shape a
window needs -- it already holds every image's positions and shapes to draw them
over the photograph, and a read per gesture would be a second copy of what is on
screen -- and it is what the [track-editing
bench](../bench/editable-track.md) searches through.
`constellation_at_pixel` calls it for the indexed case, so the two cannot
answer differently, and keeps for itself the one thing it adds: the fallback for
an image the corpus does **not** index, whose descriptors have to come out of
the `.sift` file because the corpus has none of them.

**Why the centre is part of the constellation and not of the params.** It is a
fact about the patch, as the positions are, and the two `*_at_pixel` /
`*_from_keypoints` entry points already hold it: they select the constellation
around it and then pass it through, so the [bench's descriptor
search](../bench/editable-track.md), which calls `constellation_from_keypoints`,
gets the centre-weighted warp with no argument of its own. A caller of
`constellation_query` that assembled its positions some other way may have no
centre to give; `None` under `CenterWeighted` fits as `LeastSquares`, which is
the honest reading of "no point matters more than another" and is already most
of the gain.

**Why the refit's radius is measured and not passed.** The weight's scale is
`sigma` times the largest distance from the centre to any constellation
position. The radius a caller *asked* for can be much larger than the disc its
features fill -- a radius rule is a prediction, not a count -- and a sigma
proportional to an empty rim would flatten the weights towards `LeastSquares`
without anyone having chosen that.

**Why the refit is an enum and not a bare `sigma`.** Off, unweighted and
weighted are three behaviours, and encoding two of them as `0.0` and infinity
would put a correctness condition on a float comparison. `None` earns its place
because the resident/file-backed parity tests and anyone diagnosing RANSAC
itself want the model as it was drawn.

**Choosing the constellation size.** The query takes a radius, but what governs
the answer is how many features that radius holds, and the two are related
through the image's keypoint density. `radius_for_feature_count` is that
relation, `sqrt(target * A / (pi * K))` for image area `A` and `K` keypoints in
the image, and it is a separate function rather than a mode of the query so that
a caller who has its own radius keeps it. Fifty features is the size to ask for.
Recall rises with the constellation and never stops rising, while the share of
found images whose warp is trustworthy falls monotonically, because the affine
is the first-order approximation of a homography about the patch centre and the
term it drops grows with the patch; across five captures the two curves cross
around fifty, and past two hundred features the warp is wrong more often than
right. Keypoints cluster on texture and a patch is usually centred on one, so
the radius this predicts held 70 to 100% of the features asked for in
measurement.

**There is one size and no schedule.** Asking the nearest ten features first and
widening only when nothing matched would be the cheaper query if a small prefix
ever found an image the full fifty missed, and over 1,200 patch-stage
comparisons on five captures it never did -- two single candidates, against a
recall loss on everything else. The prefix sweep's harness is
[`kdf_constellation_progressive_eval.py`](../../../scripts/kdf_constellation_progressive_eval.py).
A feature's forest hits do not depend on which other features are in the
constellation, so stage `n` of any schedule is exactly a fresh query on the
nearest `n`, which is what lets one table of prefix queries answer for every
schedule at once. What a small prefix *did* give was a more accurate warp near
the centre, and a second round measured freezing each image's warp at the first
stage that accepted it with the same evaluation script's `measure2` and
`analyze2` modes (with `--refit none` for the original three-point baseline).
Across 13,416 (patch, image) cases on eight corpora, that lock is beaten by
refitting the fifty-feature consensus on all eight, by 0.065 to 0.148 of the
share of images placing the patch centre within 3 px, and the gap *widens* with
the baseline, so there is no capture shape where staging wins. It also costs
more: 38,966 of 38,970 candidates' correspondence lists grew between a
25-feature stage and the cap, so a staged query re-fits every candidate at every
stage, +9 to +27% of wall time against the refit's +0.7 to +3.6%. So the
constellation is fifty features fitted once, and what a caller chooses is how
the consensus is reported, not how it is gathered.

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

### Three points find a model; the consensus reports one

Three points are how a model is *found*, and a poor way to report one. The
sample size is what RANSAC's cost is exponential in, which is the whole argument
above; but a model passing exactly through three keypoints carries all three
keypoints' localisation noise, and nothing about having found the right images
says the warp through those particular three is the best account of the fifty
correspondences that agreed with it. So once the consensus is chosen, `refit`
fits the reported affine to the whole of it by least squares. It is one 3x3
solve per reported image, it changes nothing about which images are found, and
across eight corpora it moves the share of found images whose warp places the
patch centre within 3 px by +0.06 to +0.18, every interval clear of zero.

The fit is weighted towards the centre, because that is where the caller applies
the warp: for inliers `i` with constellation positions `p_i`, matched positions
`q_i` and weights `w_i`, the affine minimises `sum w_i |A p_i + t - q_i|^2`,
with `w_i = exp(-(d_i / (sigma R))^2 / 2)` in the distance `d_i` of `p_i` from
`Constellation::center` and `R` the constellation's radius about that centre.
The two rows of `[A | t]` share one 3x3 normal matrix
`sum w_i [p_i; 1][p_i; 1]^T` and differ only in the right-hand side, so it is
one factorisation and two back-solves.

**One pass, and no re-selection.** Re-selecting the inliers under the refitted
model and fitting again, up to three rounds, moved the centre share by at most
0.007 on any corpus, so the query does not. That also keeps `inliers` meaning
one thing -- within `threshold_px` of the model RANSAC chose -- at the price
that an inlier may sit slightly outside `threshold_px` of the affine reported.

**The refit is refusable and never fatal.** A refitted model goes through the
same determinant guards as a three-point one, and a normal matrix that is
singular -- which is what inliers collinear in the query image produce -- is a
third way to refuse. Any refusal reports the three-point model instead: the
consensus that admitted the candidate still stands, so nothing about the fit can
drop an image from the answer.

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

The same determinant refuses two further models that are not degenerate but are
impossible. It is the signed area ratio, so a **negative** one mirrors the patch,
and two cameras looking at one piece of surface cannot mirror it however they are
placed: a reflection is refusable on sight, with no threshold to choose. Its
square root is the geometric-mean scale of the warp, so a value far from unity is
a model that blows the patch up or shrinks it past anything a change of viewpoint
explains; `max_scale` bounds that, refusing a model whose scale leaves
`[1/max_scale, max_scale]`. Both are applied where the collapse test is, inside
the three-point solve, so a refused model is never scored and can neither win a
trial nor be reported -- and again to the refitted model, which is the other
model that can reach a caller.

Neither guard is free-floating. On four wide-baseline-stills captures, mirrored
models were 2 to 30% of all reported candidates and almost none of them were
right: 0.00 to 0.22 of them placed the ground truth's own correspondences within
3 px, where the candidates with a positive determinant on those same captures
managed 0.22 to 0.72. Candidates whose scale left `[0.5, 2]`
were 28 to 75% of the candidates at fifty features, with a correctness rate of
0.00 to 0.50 against 0.58 to 0.89 for the rest. A video walk shows almost none of
either, because consecutive frames differ by a few percent of scale; the guards
bite where a small corpus lets chaff dominate a candidate image's correspondence
list.

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

### One feature matches one place in an image

A constellation feature's `k` neighbours are spread across the corpus, and
nothing stops several of them from belonging to one image. Only one of those can
be where that feature is in that image: the patch is one piece of surface, and a
point of it appears once per photograph of it. The rest are the lookalike
problem of a single lookup, reappearing inside one candidate's correspondence
list, and they cost twice over. They dilute the inlier ratio the three-point
sampler works against, since an extra hit is an outlier of whatever the right
warp is; and they are correlated outliers rather than independent ones, because
they share a source position, so a model fitted through two of them agrees with
neither.

`one_hit_per_image` keeps the nearest hit of each (constellation feature,
candidate image) cell and drops the rest. It is the cheapest statement of "one
feature, one place": no threshold to choose, and the hit it keeps is the one the
index already ranked first. It is on by default. How much it removes is a
function of how many images the corpus holds: with a thousand images a feature's
32 neighbours land in nearly 32 distinct images and hardly any cell is crowded,
while with seventeen they cannot, and the pigeonhole alone puts two thirds of
the cells above one hit. So the collapse is nearly inert on a large capture and
removes half the correspondences per candidate on a small one, which is exactly
where the inlier ratio needs the help.

`same_image_ratio` adds Lowe's test to that choice, scoped to the same cell. The
classic ratio test compares a descriptor's nearest neighbour against its second
nearest anywhere in the corpus and refuses the match when the two are too alike.
Applied to a constellation it would throw away exactly the repeated texture the
consensus is able to keep, because a feature of a brick wall has a hundred near
neighbours across a capture and no single one of them is decisive. Scoped to one
image it asks a different and answerable question: given that this image is the
candidate, is this feature's best spot in it clearly better than its second-best
spot in it? A feature whose two best positions inside one photograph are equally
good says nothing about where the patch is in that photograph, whatever it says
elsewhere, and the cell then contributes nothing rather than contributing an
arbitrary first choice.

The ratio is a ratio of Euclidean distances and the forest reports squared ones,
so the comparison is made against the square of the ratio. A cell holding a
single hit has no runner-up and is kept, that being the shape the test is
looking for. The ratio is off by default, so by default a cell keeps its nearest
hit whatever its runner-up looks like; turning `one_hit_per_image` off as well
is what makes every hit of every cell a correspondence.

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

The per-cell collapse runs on the flat hit list, after the query image's own
hits are dropped and before the geometry lookup, rather than on the grouped
candidates. A hit already knows its cell at that point, and a hit dropped there
costs no geometry read. The survivors keep the encounter order, which is
neighbour order within each constellation feature, so nothing downstream of the
collapse can tell it happened except by the count. A cell's nearest hit is the
first slot holding its smallest distance, so two hits at one distance resolve to
the earlier, which is the earlier neighbour in the list the forest returned.

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

All defaults are `ConstellationParams::DEFAULT` in
[`constellation.rs`](../../../crates/sfmtool-core/src/features/kdforest/constellation.rs),
which is what `Default` returns and what the Python bindings name a field of in
their keyword defaults, so there is one copy of each number.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | `32` | Neighbours retrieved per constellation feature. |
| `max_leaf_checks` | `512` | Per-query budget of distance evaluations in the forest traversal. |
| `threshold_px` | `8.0` | Reprojection distance, in the candidate image's pixels, within which a correspondence agrees with a model. |
| `iterations` | `200` | Three-point samples drawn per candidate image. |
| `min_correspondences` | `3` | Fewest correspondences before an image is fitted at all; three is also the floor the model needs, so a smaller value has no effect. |
| `one_hit_per_image` | `true` | Keep only the nearest hit of each constellation feature in each candidate image, dropping the rest of that cell. |
| `same_image_ratio` | `1.0` | Lowe's ratio inside one (constellation feature, candidate image) cell. Below 1.0 the cell collapses to its nearest hit and keeps it only when that hit's distance is under this factor times the cell's runner-up; a cell with one hit is kept. 1.0 and above is off. |
| `min_inliers` | `8` | Fewest inliers for an image to be reported. |
| `max_scale` | `4.0` | Widest scale change a model may claim, as `sqrt(\|det\|)` of its 2x2 linear part; one outside `[1/max_scale, max_scale]` is refused unscored, as is any reflection. |
| `refit` | `CenterWeighted { sigma: 0.5 }` | How the reported affine is fitted to the consensus the winning three-point model collected. `None` reports that model as drawn, `LeastSquares` fits every inlier alike, and `CenterWeighted` weights an inlier by `exp(-(d / (sigma R))^2 / 2)` in its distance `d` from `Constellation::center`, `R` being the largest distance from that centre to any constellation position. |
| `seed` | `0` | Base RNG seed; candidate image `i` draws from `seed + i`. |

`max_leaf_checks` is 512 because 128 leaves a fifth to a third of the ground
truth's own correspondences outside the neighbour lists entirely: on one capture
where the budget was swept directly it carries 78% of them, against 90% at 512
and 97% at 2048, and end to end the move to 512 is +0.14 to +0.18 correspondence
recall on four of five captures and +0.04 to +0.20 image recall on all five, for
1.7 to 2.3 times the wall time, with residual medians and the correspondence
count per candidate unchanged -- unlike a larger `k`, the budget replaces chaff
rather than adding it.

`one_hit_per_image` is on because the hits it drops cannot be right: a point of
the patch's surface appears once per photograph of it, so at most one hit of a
cell is that feature's correspondence in that image and the rest are outliers of
whatever the right warp is. They are correlated outliers, sharing a source
position, so a sample drawing two of them asks for a transform sending one point
to two places and a model fitted elsewhere collects a vote from each. The index
agrees about which one to keep: where a cell holds a corroborated correspondence
at all, it is the cell's nearest hit 94 to 98% of the time, so the collapse costs
a few percent of the true correspondences and removes every crowded cell's
surplus. On a seventeen-image capture that surplus is half of everything a
candidate is offered, and removing it roughly doubles the inlier ratio, which
enters the three-point sampler's success rate cubed.

`min_inliers` is 8 because 6 is the noise floor: the median inlier count of a
candidate sharing no point at all with the query image is exactly six, on every
capture at every constellation size. Eight removes 42 to 95% of the false
candidates and 55 to 100% of the never-covisible ones, and raises the share of
trustworthy warps or leaves it flat everywhere, at the cost of image recall that
is mostly six-inlier candidates. A caller who wants a list of images to look at
rather than warps to use can set it back to 6.
The `max_scale` default only refuses the absurd: a legitimate two- or
threefold scale change between two frames exists, so the bound is loose by
default and a caller who knows its own baselines tightens it. It bounds a
refitted model exactly as it bounds a three-point one; nothing measured argues
for a different bound on the two, and over eight corpora the refit's fallback
fired too rarely to show in any column.

`refit`'s `sigma` is 0.5 because it is the knee. `0.25` gives the best warp at
the centre on all eight corpora and the worst over the disc -- 0.80 to 0.93 of
found images place the ground truth's own correspondences within 3 px across the
whole disc, against 0.91 to 0.98 for the unweighted fit -- because it all but
discards the rim. `0.5` gives back 0.01 to 0.04 of the centre share and keeps
essentially all of the disc share; at the rim its weight is `exp(-2)`, about an
eighth, so the rim still constrains the linear part. That matters because the
linear part is consumed too: the [bench's search](../bench/editable-track.md)
seeds an observation's pixel from the translation and its shape from the 2x2,
and the 2x2's error against a ground-truth local affine roughly halves at 0.5
(rotation 2.13° to 1.25° on seoul_bull). A caller that only ever maps the centre
sets 0.25. Whether the bench is such a caller is open: settling it needs the
bench's own downstream score -- whether the refinement converges from the seed --
rather than a residual, which no round has measured.

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
image_index=None, center=None, ...)` takes `(N, 2)` float32 positions and either an
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

`radius_for_feature_count(image_width, image_height, keypoint_count, target)` is
a free function on the same module, next to the forest classes, and returns the
float radius to hand `constellation_at_pixel` for a constellation of about
`target` features.

Both query methods take `refit="center_weighted" | "least_squares" | "none"` and
`refit_sigma=0.5`, whose defaults are read off `ConstellationParams::DEFAULT`
like every other keyword default here. A spelling that is none of the three is a
`ValueError`, as is a `refit_sigma` that is not finite and positive, because a
caller asking for a weighting and silently getting the three-point model is the
one outcome it could not detect from the result. `center=(x, y)` on
`constellation_query` is what the weighting measures distances from;
`constellation_at_pixel` already has the centre it was asked about and passes it
through, so the two forms of one query agree only when `constellation_query` is
handed the same one.

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
- `a_mirrored_or_an_inflated_candidate_is_refused_and_a_doubled_one_is_not`
  plants three more images holding the whole patch, under a doubling, a
  reflection and a tenfold blow-up. The doubling is reported with every planted
  feature as an inlier; the other two are absent. Lifting `max_scale` to infinity
  brings the tenfold image back at the same inlier count, which is what
  distinguishes a guard refusing the model from an index never finding the
  correspondences, and the reflection stays absent because its refusal is a sign
  test with nothing to lift.
- `a_second_hit_of_one_feature_in_one_image_collapses_to_its_nearest` plants a
  near-copy of six of the patch's descriptors inside the planted image, parked
  where the planted warp puts nothing, so six constellation features have two
  hits there. Untouched, that image is offered more correspondences than the
  constellation has features; under either knob no candidate is offered more
  than that, the decoys are gone and the twins they shadowed are still inliers.
  It then drives the collapse directly over a hand-built pair of cells, where a
  ratio of 0.8 keeps a nearest hit at nine tenths of nothing and refuses one at
  nine tenths of its runner-up.
- `the_model_solver_refuses_a_reflection_and_a_scale_far_from_unity` drives the
  three-point solve directly, over a mirrored, a tenfold, a tenth-scale and a
  doubled destination triangle, at the default bound and at a wider one.
- `a_jittered_consensus_is_refitted_nearer_the_centre_than_its_three_point_model`
  plants the warped image with a deterministic per-keypoint wobble of up to two
  pixels, so that a model through three keypoints is measurably off, and asserts
  the default refit places the patch centre nearer the planted warp than the
  `AffineRefit::None` model from the same query does, over the same candidate
  with the same inlier correspondences in the same order.
- `an_exact_consensus_refits_to_the_exact_affine` drives the fit itself over a
  disc of positions under a known affine: every mode that fits recovers it, and
  `None` returns no fit at all rather than one that happens to agree.
  `the_weighted_refit_is_truer_at_the_centre_than_the_flat_one` bends that warp
  with a quadratic term, which is what a homography looks like once the patch is
  too large for its first-order part, and asserts the weighted fit is nearer the
  truth at the centre than the flat one and nearer still at `sigma` 0.25.
  `a_collinear_or_impossible_consensus_reports_the_three_point_model` covers the
  three refusals: a consensus on one line of the query image, one whose
  least-squares model mirrors the patch, and one whose scale leaves the bound,
  the last of which comes back when the bound is widened.
- `a_constellation_with_no_centre_is_refitted_flat` asserts `CenterWeighted`
  without a centre is `LeastSquares` bit for bit, both at the fit and through the
  query, and that it is still a fit rather than the three-point model.
- `the_radius_rule_holds_the_features_it_promises` checks that the disc
  `radius_for_feature_count` returns covers `target / K` of the frame, against
  the radii measured on two real captures, and that an image with no keypoints
  gets zero.

The corpus uses a wide canvas deliberately: the odds of a wrong correspondence
landing inside an eight-pixel threshold scale with the inverse of the image area,
and an exact inlier count is only assertable when they are negligible.

[`tests/rust_bindings/test_kdf_constellation_rust_bindings.py`](../../../tests/rust_bindings/test_kdf_constellation_rust_bindings.py)
extracts SIFT once from the included Seoul Bull image and indexes those
descriptors twice, the second copy at warped positions as a second image. It
checks the dict surface key by key, that the recovered warp is the planted one,
that reported positions match the geometry the sources carry, that both forest
classes return the same thing through both entry points and both descriptor
forms, that a `.kdf` written without sources is a `ValueError`, and that the
radius rule, handed that image's own size and keypoint count, picks a disc
holding tens of features rather than a handful or most of the frame. Because
image 1 is image 0's descriptors again, a feature's neighbour list there holds
its own twin beside other features of that image, so the same file is where
`one_hit_per_image` and `same_image_ratio` are checked from Python: turning the
collapse off offers some image more correspondences than the constellation has
features, the default and the ratio each offer none, and the planted warp
survives both. The planted warp being exactly affine, it also survives all three
`refit` spellings, which is what the binding test asserts about them alongside
their leaving every candidate and every inlier count where they were; what it
tests about the mode itself is that a misspelling and a non-positive
`refit_sigma` are `ValueError`s, and that the weighted mode without a `center`
is the flat one.

## Non-goals

- No homography. The model is affine at every stage: the three-point solve draws
  one, the refit fits one, and a caller wanting the perspective term the patch
  drops fits it from `inlier_correspondences`.
- No ratio test against the neighbour list as a whole, and no other
  per-descriptor filtering of it. The consensus is the filter, and a ratio test
  over the whole corpus would discard the repeated-texture matches a
  constellation is able to keep. `same_image_ratio` is that test scoped to one
  candidate image, which is a different question, and it is off by default; what
  is on is `one_hit_per_image`, which chooses within a cell without judging it.
- No scoring of a candidate image beyond its inlier count. Photometric agreement
  belongs to the patch refinement a caller seeds from this result.
- The `.sift` entry point is `u8` descriptors only, because that is what a
  `.sift` file holds. `constellation_query` itself is generic over the forest's
  scalar.
