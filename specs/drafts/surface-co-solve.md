# Solving depth and normal together over a neighbourhood of patches

**Status:** Draft. Decided: nothing yet. The first step of Use 1, finding
anchors near the pixel, has a harness mode and a finder that returns
anchors with distance ranges, grouped into depth layers (see "The first step:
anchors"). This draft sets out an approach, the
evidence for it, and the prototypes to build and measure in the
leave-one-track-out harness
([`scripts/track_at_pixel/`](../../scripts/track_at_pixel/README.md)). It
extends the open question "Framing a track" in
[track-at-pixel.md](track-at-pixel.md), which links back here, and proposes a
second, related operation: filling a surface outward from a seed patch.

A patch on a surface has a depth along each camera's ray and a normal, and the
two are hard to find separately. With the normal wrong, the patch is warped
wrongly into every other photograph, the correlation is lower, and the depth
that fits best is shifted. With the depth wrong, the normal that fits best is
tilted. The track-at-pixel operation finds the depth first and the normal
afterwards, once, and its normals are the weakest part of its output. A person
building tracks by hand on the bench does something else: they fit, tilt,
refit and repeat, and they use nearby patches, placed about half a patch
diameter apart, to see which way the surface runs. This draft describes that
process as an algorithm with one repeated step, and two uses of it: building a
track at a pixel from a small neighbourhood of patches around it, and a
"flood fill surface from here" operation that grows a sheet of patches from
one patch.

## Why the current pipeline's normals are weak

The harness scores a candidate by `S = (G + 2N) / 3`, averaged over two ground
truths (seoul_bull and the Kerry Park candidate `tk113`) and two passes: the
full pass, where the rest of the reconstruction is available, and the empty
pass, where it is not. `G` counts good tracks and `N` credits their normals.
The best candidate so far, `renormal`, scores 0.539 against the shipped
cascade's 0.403. Its gains and limits are in the harness README, § "Normals,
gates and fallbacks on two ground truths". In short:

- In the full pass, the best normal comes from copying the reconstruction's
  neighbouring normals. That works partly because the ground truths were built
  the same way, so it measures agreement with the curation as much as with the
  surface.
- In the empty pass, the only normal source is `PatchCloud.refine_normals`,
  which searches 45 degrees around the mean viewing direction with a
  fronto-parallel prior. Its median error is 16 to 23 degrees at the true
  position, and it never recovers a normal far from the mean viewing
  direction.
- Every source tried was used once, after the depth was chosen and the views
  collected. None revisited the depth or the views with the normal it found.

Every ground-truth normal was set by hand in SfM Explorer. The cases below are
patches the author of the ground truth picked out as showing how they were
made, measured against the ground truth and against the harness's runs.

## How the hand process works

On the bench, a track is refined by repeating a few gestures until nothing
changes:

1. **Fit.** The sightings are localized, the point is triangulated from them,
   and the patch is placed there.
2. **Look at the neighbours.** Patches placed about half a diameter apart on
   the same surface show which way it runs. A patch whose plane does not pass
   through its neighbours' centres has the wrong normal.
3. **Tilt, and fit again.** The refit moves the depth, the new depths show
   the next correction to the normal, and two or three rounds converge.
4. **Search for more views.** Once the pose is close, the geometry search
   finds views that were not found before, often ones with more parallax,
   and those sharpen the depth for the next round.
5. **Shape the patch.** Shrink it or move it off the query pixel to avoid a
   part of the surface that another view does not see.

## The repeated step

Everything in this draft uses one step applied to a set of patches, the
**neighbourhood**. Each patch has a centre, a normal, a size and a set of
sightings.

1. **Fit each patch.** A track-stage fit, with the queried sighting anchored
   where one exists.
2. **Search for views** from each patch whose pose changed.
3. **Estimate each patch's normal** by the estimators below, each on the axes
   it can determine.
4. **Tilt each patch** to its estimate.
5. **Stop** when no normal turns more than a small angle and no depth moves
   more than a small fraction of the patch size; otherwise repeat.

### Normal estimators

- **From the neighbours' positions.** The fitted centres of the adjacent
  patches, within about 2.5 half-sizes, should lie in the patch's plane. Where
  they spread in two directions they fix the normal. Where they lie along a
  line, as on a row of patches across a sign or along a hedge, they fix only
  the turn about that line's perpendicular; the tilt about the line itself is
  left as it was.
- **From a split patch.** For an axis the neighbours leave free, two patches
  of half the size, placed half a size apart across that axis, are fitted.
  The line between their fitted centres lies in the plane, which fixes the
  tilt about the free axis.
- **From a group photometric score.** Where positions fix nothing, the normal
  that maximises the summed correlation over several patches of one surface is
  harder to fool than one patch's score, because a wrong tilt that suits one
  patch's texture does not suit the others'.
- **The mean viewing direction.** When no tilt changes the score, there is no
  surface to find a normal for (a distant chimney seen across a wide baseline
  is the example). The normal is then the mean direction to the cameras, and
  no neighbour's normal is copied in.

No estimator caps how far the normal may turn from the mean viewing
direction. The side of a house seen at grazing angles has its true normal 62
degrees from it, and the photographs favour that normal all the way there.

### Scoring a pose

- **ZNCC and peak offset together.** The peak offset is how far each view's
  correlation peak sits from the patch's projection. It moves much more with
  depth than ZNCC does, and either one alone can be fooled: a wrong tilt can
  give a small offset at a wrong depth, and a wrong depth can give a high
  ZNCC.
- **Robust weights.** A texel that disagrees with the consensus in one view,
  such as background around a rock that some views show and others hide,
  lowers only its own weight, not the whole score. A view that correlates
  poorly at every pose, such as one at a grazing angle to grass, lowers only
  its own weight.
- **Judged at the pixel.** When a pose is chosen for the query, it is judged
  by the texels under the query pixel in the query view, not by how many views
  agree. A wrong depth can gather more views than the right one.

### Patch size, and where a surface ends

The size of a patch decides what the step can measure, and the grid's spacing
follows from it. By hand, the size is judged from two things seen in the
photograph: the scale of the texture detail on the surface, and where that
surface starts and stops. The algorithm can measure the first. It cannot see
the second directly, so it needs photometric proxies for it.

What the cases say about size:

- **Small patches fix depth, large ones fix the normal.** The window's corners
  gave good depths and poor normals; the whole window gave the better
  photometric normal. A larger patch carries more tilt information but is more
  likely to cross onto another surface.
- **The pipeline's patches are too small.** On good tracks, the apparent size
  (the texel scale in the views, against the ground truth's) has a median
  ratio of about 0.55 on seoul_bull and 0.75 on Kerry Park, and the `clusters`
  member's patches are the smallest. A photometric normal on a patch twice the
  track's size scored better in `renormal`. The rock and the sign show one
  reason patches shrink: a smaller square excludes background that some views
  show and others hide, and scores higher for it.
- **Size and position interact.** On the sign, the patches were made smaller
  *and* moved up to keep clear of an occlusion at the bottom. A patch does not
  have to be centred on the query pixel to describe it.

**Texture scale.** Proxies for "large enough to hold detail":

- The bench's per-view tile localizability, and the minimum eigenvalue of the
  image structure tensor summed over the patch. The patch grows until these
  stop improving.
- The scales of the SIFT keypoints near the pixel, which the cluster and
  constellation members already read.
- The sampling ratio (texel scale) in the views, so the patch is sampled at
  about one texel per pixel in the view that sees it largest.

**Surface extent.** Proxies for "the surface ends here":

- **The size curve.** Grow the patch in steps and read the ZNCC in the views
  with parallax. While the patch stays on one surface the ZNCC holds; once it
  crosses an edge, the part beyond the edge moves differently between views and
  the ZNCC drops. The drop marks the extent, in the direction it was grown.
  Views with little parallax do not show the edge, so the curve is read in the
  views that do.
- **Per-texel agreement.** With robust weights, the texels that disagree with
  the consensus across views form a map. Where that map has a coherent region
  of disagreement on one side of the patch, the surface ends there; the patch
  shrinks away from it or shifts off it.
- **The grid's fits.** A seeded patch that lands far off its neighbours'
  plane, or whose fit fails, marks an edge of the surface in that direction.
  So does a sudden turn in the normals across the grid.
- **Edges in the query photograph.** A strong intensity or colour edge is a
  cheap hint of a surface boundary but also appears inside a surface (the
  letters on the sign), so it is only a hint, confirmed by one of the proxies
  above.

With these, size becomes another quantity the repeated step adjusts: the
patch starts at the texture scale, and each round may grow it while the size
curve holds, or shrink and shift it away from a side where the surface ends.

## Use 1: building a track at a pixel

The operation seeds its own neighbourhood around the query pixel and runs the
repeated step on it:

1. Build the query's track as today (the cascade, or its fallbacks).
2. **Seed** a small grid of patches in the track's plane, about half a
   diameter apart: a 3 by 3 grid, or a cross of five. Each is a copy of the
   track shifted within its plane and fitted. With a wrong normal a copy lands
   above or below the surface and the fit pulls it back, and those
   corrections are what the neighbour estimator reads.
3. Prefer seed positions that localise well, such as corners, where the grid
   allows; the window case shows that a few sharp features a metre apart fix
   a plane better than one patch's photometry.
4. Keep the grid two-dimensional even when the texture runs in a line, so the
   tilt about the line is fixed by positions too.
5. Run the repeated step on the grid, then return the centre track.

In the full pass the reconstruction's own points near the pixel join the
neighbourhood, but only by their **positions**, and only when they lie near
the patch's plane in 3D (a distance measured in patch sizes, not a fraction of
depth). The chimney case shows why: a roof several metres behind it passed the
current relative-depth test and gave it a normal 60 degrees off.

The grid costs several times one track. That fits the harness's rule that a
slow Python prototype is acceptable when the result is to be written in Rust.

### The first step: anchors

Step 1 above takes the query's track as the cascade builds it. Working one
query by hand showed that this step should itself be split, and done more
carefully. Before any track is built, the pixel's depth is unknown, and the
first job is to go from knowing nothing about it to having one or more
**anchors**: 3D points near the pixel that several photographs agree on, each
with the pixel it sits at in the queried image. Later steps start from an
anchor and walk toward the pixel, solving the depth again where the reading
changes. An anchor need not be on the pixel's own surface.

The query was Kerry Park point 309, a spot of flat ground seen in
`fisheye_right/frame_04` (R04), with every point of the reconstruction
removed. Five clusters of the cluster-patches file have a member within 16 px
of the pixel. By eye, and by triangulating each cluster's members with the
posed cameras, two are real and three are spurious. The real clusters'
members meet at one point with reprojection errors under 1 px. The spurious
clusters' best points lie behind the camera, with errors of 24 to 440 px. The
two real clusters are both a bench beside the pixel, and they agree: their
points are 0.4 m apart and give the same distance along the pixel's ray
within 0.1 m. From that point the search lands within a few pixels of the
true sightings in the views that see the spot from about the query's angle.

The cascade used none of this. Its clusters member tries only the nearest
three clusters, so it never reached the second real one, which links R04 to
the close views that fix the depth. It carried the pixel through clusters
whose member in R04 the cluster refinement had rejected. Nothing in it
triangulates a cluster before using it, so a spurious cluster the refinement
kept (ZNCC 0.91 in R04) would pass too. The constellation query from the
pixel, which the cascade never reached because the clusters member answered
first, matched R05, R03 and R02: the close views again.

Anchors come from four sources, strongest first. The finder stops once it has
enough anchors close to the pixel, and otherwise goes on to the next source:

1. **Tracks.** The reconstruction's own points observed near the pixel in the
   queried image, finite, seen in two or more images, every observation close
   to the point's projection. A solver already agreed on these; they are the
   strongest readings when they are near enough and well measured.
2. **Clusters.** The cluster-patches clusters with a member near the pixel,
   vetted with the posed cameras. Every member in a posed image is a
   candidate, preferring the reference and kept members; the worst is dropped
   while three or more remain, never the queried image's, until every member
   triangulates in front of its camera with a small reprojection error. Every
   cluster that passes is used, not only the nearest three.
3. **Guided matching.** With the cameras posed, a keypoint's match in another
   image lies where that image's keypoint rays pass close to its own ray. The
   keypoints nearest the pixel are each matched by descriptor among only
   those candidates, with a ratio test and a cap on the descriptor distance,
   and the matches are triangulated. A second pass adds each match that passed
   the ratio test but not the cap, if the triangulation with it still meets
   every view. Two views are enough.
4. **Constellation queries.** The SIFT index's constellation query from the
   pixel, which looks for images that see the queried point itself. Each image
   it matches carries the pixel into its own frame by the constellation's
   affine warp; those sightings are triangulated, dropping the worst while
   three or more remain. This anchor sits at the pixel itself. The cluster
   refinement is not used to read the sightings: on point 309 it rejects every
   true match, reading grazing ground at a fixed radius. It comes last, as a
   starting hit where the clusters and the guided matches give nothing.

A second query, Kerry Park point 33 in `fisheye_left/frame_13` (L13), showed
what an anchor's position does and does not say. The pixel is in trees on the
slope below the Seattle skyline, 114 m away, and the rig moves a few metres
between frames. A cluster and a guided match both gave anchors at the pixel,
10 and 12% deeper than the ground truth. Their sightings are within half a
pixel of the true point's projection in every view: the same matches. The true
track's own seven views only fix the distance to between 0.90 and 1.08 times
its value within a pixel, and the anchors' views, which lacked the true
track's two widest, set no upper bound at all. Six other anchors were pairs of
adjacent frames, whose rays meet at under half a degree and put the point
anywhere from a quarter to one and a half times the true distance.

So each anchor carries a **range**: the distances along its pixel's ray at
which every one of its views stays within a pixel. An anchor is **bounded**
when its range is finite at both ends and no wider than a few times, far over
near. Two anchors **support** each other when both are bounded, their ranges
overlap, and their readings do not rest on the same photographs.

Point 33 also showed that nearness in the image says nothing about which
surface an anchor is on. Within 40 px of that pixel there are trees at 0.3 to
0.4 times its distance, the city at 3.3 to 3.5 times, and points at infinity,
and no other true point on the pixel's own surface. An anchor near the pixel
can be real geometry of another surface. The finder's result is therefore a
set of hypotheses: the bounded anchors grouped into **layers** of overlapping
ranges, each with its anchors and their pixels. Choosing the pixel's layer is
the walk's job.

The harness measures this step on its own (`harness.py --mode anchors`, with
[`anchors.py`](../../scripts/track_at_pixel/anchors.py)); its README has the
numbers. It scores the anchors against the ground truth in three ways, none
of which assumes the ground truth holds every surface: whether an anchor's
range meets the true point's surface along the anchor's own ray (the pixel's
layer), whether a reading at the pixel is right, and whether an anchor agrees
with a true point observed at its own pixel, where there is one. An anchor
with no true point at its pixel is unchecked, not wrong.

With no reconstructed points, 95% of Kerry Park's queries and nearly all of
seoul_bull's get an anchor, and 83% and 94% get one on the pixel's layer, up
from 66% and 82% with the first version's clusters and constellation alone.
Guided matching and the wider cluster vetting made most of that difference.
The constellation query is the one source that usually reads the pixel
itself, and it is the least accurate: on Kerry Park about as many of its
readings at the pixel are wrong as right, and on seoul_bull a quarter are
wrong.

Open for this step:

- **Readings at the pixel.** Few anchors sit at the pixel itself: 6% of
  queries with the default stopping rule. Most are a few pixels away, so the
  walk has to carry them over. How to make the constellation query's readings
  at the pixel trustworthy, or confirm them from another source, is open.
- **Enough.** When to stop looking: the finder now stops after the first
  source that leaves two bounded anchors within 20 px. Near a depth edge two
  anchors can both be on the wrong surface, so stopping might instead need
  anchors around the pixel on every side, or a reading at the pixel.
- **Choosing the layer.** Which of the layers the pixel is on. The pixel's own
  patch, compared with the photographs at each layer's distance, is the
  obvious test; it is the plane sweep restricted to the layers' ranges.
- **Walking.** How to go from an anchor to the pixel. Sliding the patch there
  in small steps, fitting at each, works while the patch stays on the anchor's
  surface, and carries the anchor's depth too far past an edge. On point 309
  the reading dropped at the edge, which is where the depth should be solved
  again. At point 33 the true track's widest view has no keypoint at the
  observation, so only a photometric step can reach it.

## Use 2: flood filling a surface

A separate operation, related through the repeated step: from one patch on a
surface, grow a sheet of patches until the surface ends. In SfM Explorer it
would be a context-menu entry on a patch, "Flood fill surface from here", run
as a background task with progress and cancel. The new patches arrive on the
bench together to be reviewed, trimmed and committed.

- **A frontier.** New patches are copies of edge patches shifted half a
  diameter outward in their plane and fitted. Each takes its first normal from
  the accepted patches beside it, so a slope is followed step by step. The
  repeated step runs on a band a ring or two behind the frontier, so a bend is
  followed rather than overshot.
- **Stopping.** A candidate is refused when its fit fails or its ZNCC falls
  below a bar; when its normal turns more than a set angle from its
  neighbours' (a wall meeting the ground, a kerb); when it lands on or beside
  an existing point; or when too few views see it.
- **Occlusion.** A candidate that loses a view to something in front is tried
  smaller or shifted before it is refused.

It would get its own draft once the prototype shows how it behaves.

## The cases

All are points of the Kerry Park candidate ground truth `tk113`. Errors are
against the ground-truth normal. "Diagnostic" results were measured at the
ground truth's positions with its views, by scripts outside the harness.

| Points | What they are | What they show |
|---|---|---|
| 322 | A rock, 6 views | Depth is sharp (1% of depth moves the peak offset from 0.1 to 1.5 px) and one tilt axis is almost invisible to ZNCC (10 to 20 degrees cost under 0.03). Background around the rock caps the correlation. The empty pass returned depths 3 to 4 cm short with normals 34 degrees off, on patches half the ground truth's size |
| 321 | A rock, 3 views, one at 70 degrees | The full pass found a track 23 cm deeper with 4 to 5 views and a ZNCC of 0.90, above the true pose's 0.81: a wrong answer that agrees with itself |
| 349, 348, 350, 351, 352 | Grass on a slope, 4 oblique views from one side | On every point the best median ZNCC on a grid of poses is at the grid's edge, above the true pose. Photometric normals are 47 to 56 degrees off on four of the five. The positions lie nearly on a line, so a plane through them is 27 degrees off. Needs the group photometric score and per-view weights |
| 12, 133 to 140 | A hedge, one curved row, 8 to 24 views | Normals from adjacent fitted positions (within 1.2 to 2 half-sizes, only on the axes they fix) keep the true normals within about 2 degrees over three rounds and bring a start at the mean viewing direction from 35 to 9 to 12 degrees in one round. A random 30 degree start does not improve, because a row leaves the tilt about itself free |
| 15, 121, 122, then 124 | Three window corners, then the whole window | The corners are sharp, high-ZNCC features whose depths the pipeline's empty pass mostly gets right to within 9 cm, but whose normals range from 1 to 33 degrees. A plane through the three corners is 5 to 8 degrees from the true normal. A photometric normal on the whole window patch is 10 degrees off, better than on any corner |
| 218 | A chimney 31 m away, 23 degrees of parallax | No surface: the true normal is the mean viewing direction. The neighbour chain gave 10 of 11 full-pass queries a normal 59 to 63 degrees off, copied from a roof |
| 20 | The side of a house, 19 views all 39 to 80 degrees oblique | Along the path from the mean viewing direction (62 degrees off) to the true normal, ZNCC and peak offset improve almost monotonically; the true pose is best by both. The 45 degree photometric search cannot reach it |
| 326 to 333 | A row of small patches across a street sign, 3 views | Depth is sharp and tilt is soft. A third view with more parallax, found by the geometry search once the pose was close, sharpened the depths. The sign's vertical tilt was set by hand; split patches recover it to within 3 to 8 degrees from starts 15 degrees off, and the row's errors average under 1 degree |
| 112 and 147 more | The ground and a hill sloping 5 to 15 degrees, 3 to 15 views each | Normals from adjacent fitted positions, from the mean viewing direction (51 degrees off), reach a median of 4.7 degrees in one round, 4.0 on the hill, with no gravity prior, and ZNCC rises from 0.78 to 0.90 as they do. On the 50 flat points around 112 and the 33 hill points (647 queries, about a sixth of Kerry Park's), the pipeline's empty pass earns a normal credit of 0.10 and 0.11 against 0.69 and 0.72 in the full pass |

Two ground-truth normals rest on a prior no algorithm here will use: the house
side (point 20) was matched knowing the front of the house, and the sign's
vertical tilt was set by eye. The tables should be read with that in mind.

An earlier attempt is recorded in [track-at-pixel.md](track-at-pixel.md):
a plane through the depths of subpatches tiling the patch did not frame
tracks better. The split patch differs in three ways: the halves are fitted,
not read at a fixed depth; they are split across the one axis the other
evidence leaves free; and the estimate is iterated with a refit.

## Prototypes

Each is a harness candidate or a standalone diagnostic, measured by `S` on
both ground truths and both passes, by its median time per query, and on the
cases above.

1. **Neighbours by position.** `renormal` with the neighbour chain's depth
   test replaced by a 3D distance to the patch's plane in patch sizes, and a
   neighbour's normal copied only when its position agrees. Cheap; it tests
   the chimney failure across the whole dataset.
2. **No cap near the mean viewing direction.** The photometric search allowed
   to follow ZNCC and peak offset as far as they improve, with the
   fronto-parallel prior applied only when no tilt changes the score. It tests
   the house side and the ground.
3. **The seeded grid.** Use 1 as a candidate: the cascade's track, a 3 by 3
   grid of shifted and fitted copies, normals from positions, two or three
   rounds, the centre returned. The main test for the empty pass.
4. **Split patches.** Added to the grid for the axis a row leaves free, and
   tried alone as a normal source on single tracks.
5. **Group photometric score.** Summed over the grid, for the axes positions
   do not fix. It tests the grass.
6. **Robust texel and view weights.** Needs a change to the correlation in
   the bench kernels; prototyped first by masking texels whose agreement with
   the consensus bitmap is low. It tests the rocks.
7. **Flood fill.** Use 2 as a script, seeded at point 112, compared with the
   ground truth's 148 patches of ground and hill.
8. **Size from texture, extent from photometry.** The patch sized by
   localizability and grown along the size curve, shrinking or shifting away
   from a side where per-texel agreement fails. Measured by the harness's
   good-track count and normal credit, and by the ratio of the track's
   apparent size to the ground truth's, which the harness should report in
   place of the world-size ratio (that one is meaningless when a track at
   infinity is compared with a finite point).

The order is by cost and by how directly the evidence supports each. Numbers
from 3 onward depend on how well a grid can be fitted where the query's own
track is weak, which only the harness will show.

## Open questions

- **Grid shape and spacing.** Half a diameter apart is what the hand process
  used. Whether a 3 by 3 grid, a cross, or seeds pulled to strong features
  works best, and at what patch size, is open.
- **When to trust the neighbourhood over the query.** If the grid converges
  on a plane that the query's own sightings disagree with, the query may be
  at a depth edge. Whether to return the query's track, the grid's plane, or
  refuse is open. It meets the open question "Near a depth edge" in
  [track-at-pixel.md](track-at-pixel.md).
- **Reporting an unresolved axis.** When neither positions nor photometry fix
  an axis, the normal could be reported with low confidence rather than
  guessed. The reconstruction has a `normal_confidence` column that could
  carry it.
- **Scene priors.** Snapping a normal that is otherwise unresolved to vertical
  or horizontal gained a little on Kerry Park and nothing on seoul_bull. Whether
  a prior of that kind is acceptable at all is open.
- **Size against spacing.** Whether the grid spacing follows the patch size
  (half a diameter) as the size changes across rounds, or is fixed at the
  first round's size, is open.
- **Cost.** The seeded grid multiplies the work per query by roughly the grid
  size. A Rust implementation could share the image reads across the grid's
  patches; how much that saves is open.
