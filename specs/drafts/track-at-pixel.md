# Building a track at a pixel

**Status:** Draft. Decided: the operation's inputs, its output and refusal
contract, the evidence it may draw on, and the leave-one-track-out harness in
[`scripts/track_at_pixel/`](../../scripts/track_at_pixel/README.md) that judges
candidate algorithms for it. Not decided: the algorithm, its Rust signature,
and the bars a track has to clear. Those follow from what wins in the harness.

A person looking at a photograph in a reconstruction can point at a spot on a
surface and ask for the point there: which other photographs see that same
spot, where it sits in each of them, where it is in 3D, which way the surface
faces and how large a patch of it can be matched reliably. This operation
answers that question from a single image and pixel. It either returns a
high-quality track centred on the pixel, ready to be judged and committed on
the bench, or it refuses and says which step failed and what it measured on
the way. In SfM Explorer it is the gesture that turns "this spot" into a track
in one step, instead of a person assembling one sighting at a time.

## The operation

**Input.** A reconstruction with posed cameras (an
[edited reconstruction](../core/reconstruction/edited-reconstruction.md), so
the operation sees the same value the bench does), the decoded photographs, a
descriptor index over the images' SIFT keypoints (the in-memory `KdForest` or
the file-backed `LazyKdForest`, whichever the caller holds), those keypoints,
a cluster-patches `.matches` file, an image index and a pixel in that image.

The `.matches` file holds the capture's feature clusters: groups of detected
keypoints, one per image at most, that the descriptor index says are views of
one spot, each refined photometrically against its reference member into a
kept or rejected verdict with a ZNCC, a shift and an affine shape
([cluster-patches.md](../core/patch/cluster-patches.md),
[matches-file-format.md](../formats/matches-file-format.md)). It is generally
built from the same index the constellation query runs on, and it is passed as
a parameter like the index, not read from a path the operation derives. Its
images are matched to the reconstruction's by name.

**Output.** A track-stage
[editable track](../core/bench/editable-track.md) that has just been evaluated,
so each observation carries its measurements, together with a report of how it
was built. The track is not committed. It is a bench item: the person, or a
script's own bars, decides what happens to it next, and
[`commit`](../core/bench/editable-track.md) is still the only step that writes
the reconstruction.

**"Centred at the pixel"** is part of the contract. The track has an
observation in the queried image, that observation is `in`, and its keypoint
lies within a small pixel distance of the pixel asked about. A track that has
wandered onto a more distinctive feature nearby has answered a different
question. The operation refuses in that case rather than returning it.

**Refusal.** An error that names the stage that failed (building a local prior,
the constellation search, cluster refinement, the stage upgrade, the geometry
search, the final quality gate), gives the reason as a sentence a person can
act on, and carries what was measured up to that point. "The constellation of
38 keypoints within 30 px matched no other image with 6 or more inliers" tells
the person to try a more textured spot. "Refused" does not.

The sketch below shows the proposed shape in `sfmtool_core::bench`, beside the
steps it composes. It is a sketch, and the names will settle when it is filed.

```rust
pub struct TrackAtPixelOptions { /* the kernels' options, the quality bars */ }

pub struct TrackAtPixelReport {
    pub stages: Vec<StageRecord>,   // what each step did and measured, in order
}

pub enum TrackAtPixelError {
    // One variant per stage, each carrying its measurements and a Display
    // sentence, the way `SearchError` and `GeometrySearchError` do.
}

pub fn build_track_at_pixel(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    index: &dyn ConstellationIndex,      // in-memory or file-backed forest
    keypoints: &dyn Fn(u32) -> ImageKeypoints,
    clusters: &MatchesClusters,          // the cluster-patches section, by image
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, TrackAtPixelReport), TrackAtPixelError>;
```

## What makes a track high quality

These are the properties a returned track must have. The gates that enforce
them read the same measurements `evaluate` writes, so a bar here and a
threshold slider in Track View judge the same numbers.

- **Centred.** The queried observation is `in` and on the pixel, as above.
- **Supported.** Enough `in` observations that the leave-one-out consensus
  means something: two is a correspondence, and three or more is a track.
- **Photometrically consistent.** Each `in` observation's leave-one-out ZNCC
  against the consensus of the others clears the bar, and the median clears a
  higher one.
- **Geometrically consistent.** The finite-or-bearing classification
  ([editable-track.md](../core/bench/editable-track.md) § "Finite points and
  bearings") is one the sightings support, and the reprojection residuals are
  small.
- **Localizable.** The patch has enough texture that its keypoints are pinned
  to a small uncertainty ([patch-localizability.md](../core/patch/patch-localizability.md)).
- **Well framed.** The patch's size and normal are ones the photographs
  support. A patch that faces the wrong way, or that spans a depth edge,
  registers worse in the oblique views than one that fits the surface.
- **Sampled at a sensible ratio.** Each view renders the patch into a bitmap
  of fixed resolution, so the patch size sets how many image pixels one
  bitmap texel covers in each view. The Jacobian of the render, from texel
  coordinates to image pixels, gives that ratio per view as `sqrt(|det J|)`,
  and the ratio of its singular values gives the anisotropy an oblique view
  adds. A ratio well above one throws away image detail the bitmap cannot
  hold; a ratio well below one spends texels interpolating between the same
  pixels. About one texel per pixel is the starting target, but it is not a
  rule: the surface's extent, where its distinguishable features lie and the
  depth structure around the pixel can each call for a larger or smaller
  patch.

## Evidence the operation draws on

Each of these exists today as a bench step, a kernel or an index. Choosing
among them, ordering them and combining them is the algorithm, which this
draft leaves open.

| Evidence | Where it comes from |
|---|---|
| Other images that may see the pixel, before any 3D exists | the constellation query ([kdf-constellation-query.md](../core/features/kdf-constellation-query.md)), as the bench's `search_descriptors` |
| Correspondences already found and vetted across the whole capture | the cluster-patches `.matches` file: the clusters with a member near the pixel, each naming its other members' images, refined positions, affine shapes, statuses and ZNCCs. A kept cluster at the pixel is a ready-made set of sightings, and a cluster a little way off still names images and a local scale |
| Other images that see the patch, once it has a 3D frame | patch-view selection ([patch-view-selection.md](../core/patch/patch-view-selection.md)), as the bench's `search_geometry` |
| The depth, normal and size of the surface around the pixel | the reconstruction's observations near the pixel in the queried image, through a 2D index ([point-cloud-index.md](../core/spatial/point-cloud-index.md)): consistent depths on every side suggest one surface, and two depth populations suggest the pixel is near an edge |
| The same, seen from a better viewpoint | the observations near the corresponding pixel in another image, once a cluster or track places it there. This lets the operation make a lateral move to the image with the richest data around the spot |
| Size and normal priors from nearby surface | patches near the pixel in 2D in every observing image, and patches near the triangulated point in 3D even when they share no image with it; adjacency surfel normals ([adjacency-surfel-normals.md](../core/analysis/adjacency-surfel-normals.md)) |
| Registration without moving anything | `evaluate`, and patch rendering and scoring with view selection or member coherence ([member-coherence-validation.md](../core/patch/member-coherence-validation.md)) |
| Registration that moves the sightings | cluster refinement ([cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md)), keypoint localization over a chosen subset of views ([patch-keypoint-localization.md](../core/patch/patch-keypoint-localization.md)), sub-pixel refinement ([keypoint-subpixel-refinement.md](../core/patch/keypoint-subpixel-refinement.md)), and the bench's `fit` |
| The patch's normal | normal refinement ([patch-normal-refinement.md](../core/patch/patch-normal-refinement.md)), and the bench's `tilt_patch` to try a prior normal |
| Membership decisions | the thresholds' verdicts (`apply_thresholds`), and member coherence's split proposal for a track that is two surfaces |

## Evaluation

The operation is developed against ground truth, in the harness at
[`scripts/track_at_pixel/`](../../scripts/track_at_pixel/README.md). A
candidate algorithm is a Python function with the contract above,
`build_track(ctx, image, pixel, options)`, composed from the bindings. It
returns a track or raises an error that names the stage and the reason. When
one candidate wins, it moves into `sfmtool-core` under the interface above.

**Leave one track out.** For each point of a ground-truth reconstruction, the
harness removes the point, then calls the candidate at every pixel where the
point was observed. The removal is total: the point is deleted from the edited
reconstruction the candidate is handed, and every neighbourhood query filters
it out, so no candidate can find the answer by looking it up. What the
candidate may read is the context the harness hands it: the reconstruction
without the point, the photographs, the descriptor index, the keypoints, the
cluster-patches `.matches` file, the cameras, and 2D and 3D neighbourhood
queries over the remaining points and over the clusters' members. The
`.matches` file needs no filtering, because it is built from the detected
keypoints and the index alone and holds nothing of the reconstruction.

**Metrics.** Each query is scored against the removed track, using the two
tracks and the cameras alone:

- **Centring:** the queried observation's keypoint, and the point's
  projection, against the pixel.
- **Position:** error as the angle the two points subtend from the query
  camera, as a fraction of depth, and in ground-truth patch half-sizes.
  Agreement on finite versus bearing.
- **Frame:** normal error, and the ratio of patch half-sizes.
- **Membership:** the precision and recall of the image set against the
  ground truth's, and the keypoint error in the images both hold.
- **Photometry:** the median and minimum leave-one-out ZNCC, set beside the
  same `evaluate` reading taken of the ground-truth track. Localizability and
  reprojection residuals.
- **Cost:** time per query.

A refusal is scored by its stage, and the summary groups refusals by reason,
so a candidate's failure modes can be compared as well as its successes.

**Seeing the result.** Each run writes every returned track into a `.sfmr`
holding the ground truth's cameras and nothing else, with each result row
naming its point there. Loading it beside the ground truth in one SfM Explorer
window lets a person toggle between the two and see how each track changed.

**Ground truth.** A ground-truth reconstruction qualifies when a person has
inspected it, and when it carries patch frames and sits beside its images and
workspace marker. The harness prepares a writable copy with SIFT, a descriptor
index and a cluster-patches `.matches` file clustered from that index, so the
checked-in data is never written. A run can be handed a different `.matches`
file instead. Normals and patch
sizes in the ground truth are what its embedding pass produced. They are good
references rather than exact truth, and a returned track can read better than
the ground truth's.

## Consumers

- SfM Explorer's bench: an Image Detail gesture that builds a track at the
  clicked pixel and puts it on the bench as the active item, alongside the
  gesture that starts a one-sighting cluster
  ([sfm-explorer-track-editing.md](sfm-explorer-track-editing.md) § "From a
  pixel").
- The MCP wire, as a bench tool beside `create_bench_cluster`.
- Scripts that densify a reconstruction or re-derive a track a filter removed.

## Open questions

- **Which algorithm.** The harness decides this.
- **Holding the pixel.** The fit re-centres a track on whatever its consensus
  locks onto, which is often a more distinctive feature a few pixels away. The
  options are to re-anchor the patch on the pixel after fitting and refit with
  the queried sighting held, to bound that sighting's walk more tightly than
  the others', or to accept the move and report it. It is also open whether a
  small move is a refusal or a warning.
- **Near a depth edge.** When the neighbourhood holds two depth populations,
  the patch has to choose one surface and be sized not to span the edge. Which
  surface to pick, and whether to return both as two tracks, is open.
- **An existing track at the pixel.** If the pixel is already an observation of
  a committed point, the operation could return that point's track, build an
  independent one, or refuse. The harness removes the point under test, so it
  does not exercise this case.
- **Which view the sampling ratio is set in.** The views of one track see
  the patch at different sizes, so only one of them can sample at exactly one
  texel per pixel. Sizing so the view with the largest image of the patch is
  at 1:1 keeps every view at or below one pixel per texel, and no view loses
  detail to the bitmap; sizing so the smallest is at 1:1 gives the most
  distant view a full-resolution bitmap and downsamples the rest. The median
  view is between the two. The harness's `size_policy` option runs each, and
  a dataset whose tracks span a wide range of scales is needed to tell them
  apart. How the surface's content should move the size off the target is
  open as well.
- **Bearings.** Whether a track the rays classify as a bearing counts as a
  success, and what its size and normal mean for scoring.
- **The index.** `search_descriptors` takes the file-backed forest only. The
  operation should accept either, which needs a query trait the two share.
- **Clusters that disagree with the index.** A `.matches` file built from a
  different index, or before images were added, names clusters the
  constellation query would not reproduce. Whether the operation checks the
  two against each other, or treats the clusters as one more source of
  proposals, is open.
- **Kernels with no binding.** Registering one patch bitmap directly against
  another, and congealing a bare stack of bitmaps, are reachable today only
  through a one-patch cloud. A candidate that leans on either may justify a
  direct binding.
