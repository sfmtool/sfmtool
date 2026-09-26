# Building a track at a pixel

A person looking at one photograph of a reconstruction can point at a spot on
a surface and ask for the point there: which other photographs see the same
spot, where it sits in each of them, where it is in 3D, which way the surface
faces and how large a patch of it matches reliably. This operation answers that
from an image and a pixel alone. It returns a track centred on the pixel, fitted
and measured and ready to be judged and committed on the bench, or it refuses
and says which step refused, in a sentence a person can act on, together with
what was measured on the way. Nothing is written to the reconstruction.

The operation is a **cascade** of four ways of finding the other photographs'
sightings of the pixel. Each draws on different evidence and fails on different
pixels, and each judges its own result with the same gates, so the members are
tried in turn and the first track that passes is returned:

1. **Clusters.** The cluster-patches `.matches` file groups detected keypoints
   across the photographs into clusters and refines each photometrically. A
   cluster with a member near the pixel is a ready-made set of sightings of
   nearly the same piece of surface.
2. **Transfer.** The reconstruction's points around the pixel are already
   matched across the photographs. Over a small neighbourhood of one surface
   the map from the queried image to another is close to affine, so the
   neighbours' own keypoint pairs fix that map and it carries the pixel over.
3. **Sweep.** The neighbours' points say what surface the pixel is on. A plane
   through them, met by the pixel's ray, is a guess of the patch, which is
   projected into the photographs that face it; the fit then corrects it.
4. **Constellation.** The SIFT index's constellation query from the pixel names
   the other images whose keypoints around it agree on one affine warp. The
   candidates are read at the cluster stage and upgraded to a track.

The order is from the member that is least often wrong when it returns a track
to the one most often wrong. The members, their order and every threshold were
chosen in the leave-one-track-out harness in
[`scripts/track_at_pixel/`](../../../scripts/track_at_pixel/README.md), where
this cascade first ran as a Python composition of the bench bindings
([`candidates/cascade.py`](../../../scripts/track_at_pixel/candidates/cascade.py)); the Rust operation is a port of it and scores the
same there. What is still open about the operation, and the other candidates,
is in [the draft](../../drafts/track-at-pixel.md).

## Rust API

The operation lives in
[`bench/track_at_pixel.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel.rs)
with its members in
[`track_at_pixel/members.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel/members.rs),
the shared finish in
[`track_at_pixel/finish.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel/finish.rs)
and the neighbourhood queries in
[`track_at_pixel/neighbourhood.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel/neighbourhood.rs).
It is bound as `sfmtool._sfmtool.bench.build_track_at_pixel`.

```rust
pub fn build_track_at_pixel(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    sources: &TrackAtPixelSources<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, TrackAtPixelReport), TrackAtPixelError>;

#[derive(Clone, Copy, Default)]
pub struct TrackAtPixelSources<'a> {
    pub sift_index: Option<SiftIndexSource<'a>>,
    pub clusters: Option<&'a MatchesClusters>,
}

pub struct SiftIndexSource<'a> {
    pub forest: &'a LazyKdForestU8,
    pub keypoints: &'a [ImageKeypoints],   // one per image, in the reconstruction's order
}

impl MatchesClusters {
    pub fn new(matches: &MatchesData, image_names: &[&str])
        -> Result<Self, MatchesClustersError>;
    pub fn near(&self, image: u32, pixel: [f64; 2], radius_px: f64) -> Vec<NearbyCluster>;
}

pub struct TrackAtPixelOptions {
    pub members: Vec<CascadeMember>,        // Clusters, Transfer, Sweep, Constellation
    pub finish: FinishOptions,
    pub clusters: ClustersOptions,
    pub transfer: TransferOptions,
    pub sweep: SweepOptions,
    pub constellation: ConstellationOptions,
}

pub struct TrackAtPixelReport {
    pub member: CascadeMember,              // whose track was returned
    pub query_observation: usize,           // the observation in `image`; always 0
    pub refusals: Vec<MemberRefusal>,       // the members tried before it
    pub stages: Vec<StageRecord>,           // what it did and measured, in order
}

pub struct MemberRefusal {
    pub member: CascadeMember,
    pub stage: RefusalStage,
    pub reason: String,
    pub stages: Vec<StageRecord>,
}

pub enum TrackAtPixelError {
    NoSuchImage { image: u32, image_count: usize },
    PixelOffImage { pixel: [f64; 2], width: u32, height: u32 },
    InputMismatch { input: &'static str, got: usize, image_count: usize },
    Refused { refusals: Vec<MemberRefusal> },
    Cancelled,
}
```

**Why these inputs.** `edited` and `views` are what every photometric bench step
takes, so a caller that already runs `evaluate` and `fit` holds them, and
decoding and caching the photographs stay the caller's. The reconstruction is
read only through `edited`: the neighbourhood queries are built from its live
points on each call, so a point the version has deleted is invisible to the
operation, which is how the harness holds a point out, and a point an edit
added is a neighbour like any other. The two **sources** are the evidence that
holds nothing of the reconstruction's points, the SIFT index with every image's
keypoints and the `.matches` clusters, so they are built once per capture and
shared by every query and every version. Either may be `None`; the member that
reads it then refuses naming what is missing, and the others run. The clusters
are a prepared value, `MatchesClusters`, rather than the raw file, because
matching the file's images to the reconstruction's by name and building a 2D
index over each image's members is work a caller does once.

**Why the report is a list of stage records.** The four members run different
steps, so the report is the sequence of what the returning member did, each
record a variant carrying that step's measurements (`NearbyClusters`,
`DepthModes`, `Hypotheses`, `Candidates`, `LocalPrior`, `Constellation`,
`Lateral`, `ClusterEvaluate`, `Upgrade`, `PriorTilt`, `Anchor`, `NormalPrior`,
`GeometrySearch`, `Clean`, `Final`). A refusal carries the same list up to the
step that refused, so a failure is read the way a success is. The whole-query
error is `Refused` with every member's refusal in order; its `Display` names the
last, and its `stage()` is `"cascade"`. The other variants refuse a query that
names no place before any member runs.

**The track** is at the track stage and has just been evaluated, so every
observation carries its leave-one-out ZNCC, shift and projection offset.
Observation 0 is the queried sighting, `in` and pinned, with its keypoint within
`max_query_offset_px` of the pixel. The track has no origin, so a commit of it
creates a point. It **carries its consensus bitmap and colour**, fused where it
stands: the finish's last step slides the patch onto the pixel, and a patch step
drops the bitmap fused over the square as it stood, so the operation fuses once
more before it returns. The fuse is the one a fit ends with, run over the `in`
sightings at their keypoints on the reconstruction's own bitmap grid where it
stores one, and the colour is read off the tile's centre; it moves nothing, so
the position, the placement, the keypoints, the verdicts and every reported
number are what they were before it. A reconstruction that stores a bitmap per
point can therefore take the track in a commit as it is returned.

```rust
use sfmtool_core::bench::{
    build_track_at_pixel, MatchesClusters, SiftIndexSource, TrackAtPixelOptions,
    TrackAtPixelSources,
};

let names: Vec<&str> = edited.base.image_table.images.iter().map(|i| i.name.as_str()).collect();
let clusters = MatchesClusters::new(&matches, &names)?;
let sources = TrackAtPixelSources {
    sift_index: Some(SiftIndexSource { forest: &forest, keypoints: &keypoints }),
    clusters: Some(&clusters),
};
match build_track_at_pixel(&edited, &views, &sources, 3, [412.5, 280.0],
                           &TrackAtPixelOptions::default(), &Progress::none()) {
    Ok((track, report)) => { /* put `track` on the bench; report.member says who built it */ }
    Err(e) => eprintln!("{}: {e}", e.stage()),
}
```

## The members

Every member either refuses with its own stage or hands a track-stage track to
the [finish](#the-finish). The three that try several candidates (clusters,
depth modes, surface hypotheses) fit each one the same way: the pixel and the
candidate's `(image, pixel)` sightings become a one-sighting cluster with every
sighting added and set `in` by hand, the cluster is upgraded to the track stage
(which triangulates the sightings and runs the track-stage fit), the track is
given an [anchored fit](#anchoring) and the thresholds' verdicts, and it is
scored as its `in` count times its median ZNCC, floored at zero. The best score
is finished.

**Clusters** (stage `clusters`). The clusters with a member in the queried image
within `search_radius_px` of the pixel, nearest first, each by its member there
that is nearest the pixel; the first `max_clusters` are tried. The pixel's offset
from that member, `d`, is a step in the queried image. For every other member the
refinement kept (or made the reference), in an image the reconstruction holds,
`S_other · S_query⁻¹ · d` added to its position is the pixel carried into its
image, where `S` is a member's affine shape; where two members share an image,
the higher ZNCC wins. A carried pixel must land 2 px inside the frame. A member
whose shape is degenerate, or one the pixel is more than `max_offset_in_scales`
of its own scale `sqrt(|det S|)` from, carries nothing. The patch half-width is
`radius_in_scales` of that scale, clipped to `[min_radius_px, max_radius_px]`.

**Transfer** (stages `neighbourhood`, `transfer`). The observations within
`neighbour_radius_px` of the pixel with a finite positive depth are sorted by
depth and split into **depth modes** wherever one depth exceeds the one before it
by more than `depth_mode_gap`; the modes with at least `min_pairs` points are
tried, the one with the nearest point first. For each other image, the mode's
`max_neighbours` nearest points that are seen there give pairs
`(keypoint here − pixel, keypoint there)`, weighted `1 / (1 + distance)`. A
weighted least-squares affine is fitted, pairs it misses by more than
`max_residual_px` are dropped and it is fitted again, at most three fits; the map
is kept when it still rests on `min_pairs` pairs and has a positive determinant,
and its translation (the image of the pixel) must land 2 px inside the frame. The
patch half-width is the median apparent half-width of the mode's first eight
points by depth.

**Sweep** (stages `prior`, `hypothesis`). The observations within
`prior_radius_px` split into depth modes as above; a mode of fewer than
`min_mode_size` points is skipped when there are others. Each mode's `prior_k`
nearest points give a plane: through their distance-weighted centroid, with
their distance-weighted mean normal turned to face the camera. The pixel's ray
meets the plane at the hypothesis, or, where the ray grazes it (the cosine
under 0.15), runs as far as the centroid. The hypothesis is projected into every
other photograph where it lands 2 px inside the frame, faces the camera within
`max_view_angle_deg`, and is not behind the reconstruction's own surface (the
observations within 12 px of the projection do not sit, at their median depth,
nearer than 0.93 of the hypothesis's depth). The `seed_views` most nearly face-on
seed the track, which is tilted to the plane's normal and resized to the mode's
apparent half-width before its anchored fit.

**Constellation** (stages `constellation`, `cluster evaluate`, `upgrade`). A
local prior is read from the observations within `prior_radius_px`: the pixel's
nearest finite neighbour picks a depth mode, and that mode's `prior_k` nearest
points give a patch half-width and a distance-weighted mean normal. A cluster is
seeded at the pixel with that half-width, and the constellation query runs from
it at the radius a uniform keypoint density puts `constellation_target`
keypoints inside, with `min_inliers`; the `lateral_searches` images it added
with the most inliers are each searched from in turn, at their own radius. The
cluster is read (the refinement at the cluster stage), the thresholds paint its
verdicts, and it is upgraded to the track stage. The patch is then tilted toward
the prior's normal and refit, and the tilt is kept when the median ZNCC does not
fall.

## The finish

Every member ends here, with the queried sighting as observation 0.

### Anchoring

A track-stage fit localizes every sighting, the queried one included, against
the consensus of the others. Beside a feature more distinctive than the pixel's
own, the whole patch slides toward it in every photograph at once: the sightings
stay consistent with one another, but the track is no longer at the pixel.
Anchoring slides the patch back across its own plane until its centre in the
queried photograph is the pixel (`translate_patch_to_pixel`), which carries every
sighting by the same in-plane displacement, and reads the track there
(`evaluate`). An **anchored fit** is an anchor followed by `anchor_refits` rounds
of fit-then-anchor. A round is kept when the median ZNCC does not fall, and
always when an `in` sighting has no keypoint, since the fit gives it one. The
first round is also kept always after growth, whose new sightings sit at the
patch centre's projection until a fit localizes them.

### Steps

1. **Anchor.** An anchored fit and the thresholds' verdicts. A refusal here is
   the member's refusal at stage `anchor`.
2. **Neighbours' normal.** The observations within `normal_prior_radius_px` of
   the pixel whose depth is within 15% of the track's give, over the nearest
   `normal_prior_k`, a distance-weighted mean normal. The patch is tilted toward
   it (flipped to face the queried camera), refit with an anchored fit and
   thresholded, and the tilt is kept unless the median ZNCC falls by more than
   `normal_prior_tolerance`.
3. **Growth.** With two or more `in` views, the geometry search from the queried
   sighting adds the photographs the patch projects into and reads well in. When
   it adds any, the track is read, thresholded and given an anchored fit whose
   first round is always kept.
4. **Cleaning.** Up to `clean_rounds` times: every `in` view other than the
   query whose correlation peak sits more than `clean_max_shift_px` from its
   keypoint, or whose keypoint sits more than `clean_max_projection_px` from the
   point's projection, is turned `out` by hand, and the track gets an anchored
   fit.
5. **Gates** (stage `gate`), in this order: the queried sighting is `in`; its
   keypoint is within `max_query_offset_px` of the pixel; at least
   `min_in_views` views are `in`; their median ZNCC is at least
   `min_zncc_median`; and no `in` view's keypoint sits more than
   `max_projection_offset_px` from the point's projection.
6. **Bitmap.** The track that passed is given its consensus bitmap and colour
   where it stands, by the fuse a fit ends with, which moves nothing. The last
   anchor dropped the bitmap the fits before it had fused, because it slid the
   patch off the square that bitmap was fused over.
6. **Bitmap.** The track that passed is given its consensus bitmap and colour
   where it stands, by the fuse a fit ends with, which moves nothing. The last
   anchor dropped the bitmap the fits before it had fused, because it slid the
   patch off the square that bitmap was fused over.

## Parameters

Defined in the `Default` impls in
[`track_at_pixel.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel.rs).
They are the Python candidates' defaults, which the harness chose.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `members` | clusters, transfer, sweep, constellation | the members, in the order tried |
| `finish.anchor_refits` | `1` | fit-then-anchor rounds per anchored fit |
| `finish.normal_prior` | `true` | tilt toward the neighbours' normal |
| `finish.normal_prior_radius_px` | `40.0` | how far from the pixel those neighbours may be |
| `finish.normal_prior_k` | `10` | how many of them are averaged |
| `finish.normal_prior_tolerance` | `0.01` | median-ZNCC drop that still keeps the tilt |
| `finish.geometry_search` | `true` | grow by the geometry search |
| `finish.clean` | `true` | turn out the views that disagree with the geometry |
| `finish.clean_max_shift_px` | `1.5` | peak-to-keypoint distance that turns a view out |
| `finish.clean_max_projection_px` | `1.5` | keypoint-to-projection distance that turns a view out |
| `finish.clean_rounds` | `2` | cleaning rounds at most |
| `finish.min_in_views` | `3` | gate: fewest `in` views |
| `finish.min_zncc_median` | `0.8` | gate: lowest median ZNCC |
| `finish.max_query_offset_px` | `2.0` | gate: queried keypoint's distance from the pixel |
| `finish.max_projection_offset_px` | `1.5` | gate: any `in` keypoint's distance from the projection |
| `clusters.search_radius_px` | `16.0` | how far from the pixel a cluster's member may be |
| `clusters.max_offset_in_scales` | `15.0` | how far the pixel may be from that member, in its scales |
| `clusters.max_clusters` | `3` | nearest clusters tried |
| `clusters.radius_in_scales` | `2.0` | patch half-width in member scales |
| `transfer.neighbour_radius_px` | `60.0` | how far the neighbours may be |
| `transfer.max_neighbours` | `16` | nearest points of a mode used |
| `transfer.min_pairs` | `4` | fewest pairs an affine rests on, and smallest mode tried |
| `transfer.max_residual_px` | `3.0` | pair residual that drops it from the refit |
| `sweep.prior_radius_px` | `40.0` | how far the neighbours may be |
| `sweep.prior_k` | `10` | nearest points of a mode that give its plane |
| `sweep.min_mode_size` | `2` | smallest mode kept when there are others |
| `sweep.seed_views` | `5` | most nearly face-on views that seed a hypothesis |
| `sweep.max_view_angle_deg` | `70.0` | widest view angle that seeds one |
| `constellation.prior_radius_px` | `40.0` | how far the prior's neighbours may be |
| `constellation.prior_k` | `8` | nearest points of the pixel's mode in the prior |
| `constellation.constellation_target` | `50` | keypoints the search radius is sized to hold |
| `constellation.min_inliers` | `6` | fewest agreeing correspondences for an image |
| `constellation.lateral_searches` | `2` | searches from the best images found |
| `constellation.normal_prior` | `true` | tilt toward the prior's normal after the upgrade |
| `*.depth_mode_gap` | `1.15` | depth ratio that separates two surfaces |
| `*.default_radius_px` | `8.0` | patch half-width when no neighbour states one |
| `*.min_radius_px`, `*.max_radius_px` | `4.0`, `40.0` | bounds on a patch half-width |

## Implementation notes

**The neighbourhood is the version's.** The per-image observation index is
rebuilt from `edited`'s live points on every call, which costs one pass over the
reconstruction's observations; each image's 2D index is built the first time a
member asks about that image. A point observed twice in one image is listed at
its first observation there. Distances are measured to the stored keypoints,
and a neighbour's depth, apparent half-width and normal are read off its patch:
the half-width is the `u` half-vector's length projected at the neighbour's
depth through the mean focal length.

**Order is part of the behaviour.** Candidates are compared with a strict
`score > best`, cluster members sharing an image with a strict `z > best`, and
the modes, hypotheses and views are sorted stably, so where two are equal the
one met first wins. The first-met order follows the Python composition: sightings
by the order their images are first met, modes by depth, clusters by traversal
of the member index. Changing any of them changes which track ties return.

**Where the port is not bit-exact.** The operation matches the Python cascade
query for query in the harness, but not always to the last bit: the 2x2
inverse and determinant are written out rather than taken by LU, the weighted
affine is solved through an SVD with `numpy.linalg.lstsq`'s cutoff
(`eps · max(rows, 3) · s_max`) rather than by LAPACK's `gelsd`, and the cameras'
rotation matrices come from `RigidTransform`. On seoul_bull that leaves the
built tracks' ZNCC and position equal to about `1e-15`, except where a fit's
discrete peak search sits on a boundary and a last-bit change in its seed
decides which side; the harness found one such query in 1277.

## Python bindings

`sfmtool._sfmtool.bench` holds the binding
([`sfmtool-py/src/bench/track_at_pixel.rs`](../../../crates/sfmtool-py/src/bench/track_at_pixel.rs)):

- `TrackAtPixelSources(edited, forest, keypoints, matches)` builds the sources
  once: `forest` a `LazyKdForest`, `keypoints` one `(positions, affine_shapes)`
  pair of `(N, 2)` and `(N, 2, 2)` float32 arrays per image, `matches` a
  cluster-patches `MatchesFile`. `edited` supplies the image names the file's
  images are matched to.
- `build_track_at_pixel(edited, images, sources, image, pixel, *, members=None)`
  takes `images` as `evaluate` does (a list of arrays or an `ImagePyramidSet`)
  and returns `(EditableTrack, report)`. The report dict carries `member`,
  `query_observation`, `refusals` (each `member`, `stage`, `reason` and
  `diagnostics`), and one key per stage record, named as the Python candidates
  name the same step (`clusters_near`, `modes`, `hypotheses`, `tried`, `prior`,
  `radius_px`, `search_radius_px`, `constellation`, `lateral`, `cluster`,
  `upgrade`, `prior_tilt`, `anchor`, `normal_prior`, `geometry_search`,
  `clean`, `final`). `members` names the members to try, in order.
- A refusal raises `bench.TrackAtPixelError`, a `ValueError` with `stage`,
  `reason` and `diagnostics` (`{"refusals": [...]}` when every member refused).

```python
from sfmtool._sfmtool import bench

sources = bench.TrackAtPixelSources(edited, forest, keypoints, matches)
try:
    track, report = bench.build_track_at_pixel(edited, pyramids, sources, 3, (412.5, 280.0))
except bench.TrackAtPixelError as e:
    print(e.stage, e.reason)
```

The harness candidate
[`candidates/core_cascade.py`](../../../scripts/track_at_pixel/candidates/core_cascade.py) is
this call adapted to the harness contract.

## Testing

The unit tests in
[`track_at_pixel/tests.rs`](../../../crates/sfmtool-core/src/bench/track_at_pixel/tests.rs)
run on the bench's synthetic capture, a grid of points on a textured plane seen
by three pinhole cameras, with the middle point deleted and queried at its
pixel: the transfer, the sweep and a cluster each rebuild it `in` in all three
views, on the pixel and within a patch half-extent of where it was; the deleted
point is absent from the observation index; clusters are matched to images by
name and found by their nearest member; every member refusing reports each
refusal in order; a query that names no place is refused before any member
runs; and the returned track carries a bitmap on the reconstruction's own
bitmap grid, with the colour at its centre, which fusing again does not move.
The arithmetic (the weighted affine, depth modes, the median) is tested
directly.
[`test_track_at_pixel_rust_bindings.py`](../../../tests/rust_bindings/test_track_at_pixel_rust_bindings.py)
runs the
binding on the seoul_bull capture. The harness run over every point of seoul_bull
is the test of fidelity to the Python cascade.

## Non-goals

The operation does not commit, and it does not check whether the pixel is
already an observation of a committed point. It does not run the Python
candidates' options that are off by default in the cascade (the photometric
normal search, the baseline's sampling-ratio sizing and its own finish, the ray
consensus and the patch-size multiplier).
