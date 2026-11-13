# Pairwise Flow Normal Measurements

**Status:** Draft. Measurements of a Python prototype, made outside the
repository, behind [pairwise-flow-normal.md](pairwise-flow-normal.md). No code in
the repository produced these numbers, and the prototype's scripts are not
checked in.

The measurements bear on six choices the draft makes: the flow settings, the
crop's size, the rule that chooses the support, the σ and outcome rules, the
common plane and its rounds, and what the step costs. Each section states the
question, the setup and the result.

## Reference sets

Two sets of points with a reference normal each, five sets in all when split by
dataset, and a third set with none.

**(a) Curated, from the checked-in ground truths:** 59 points.

- 25 points of `dino_dog_toy_ground_truth_v01.sfmr`, 15 of
  `kerry_park_ground_truth_v02.sfmr` and 10 of
  `seoul_bull_sculpture_ground_truth_v02.sfmr`, all `ok` points: points whose
  grown-plane answer returns to within 3° from every start
  ([grown-plane-normal-measurements.md](../core/bench/grown-plane-normal-measurements.md#σ-in-the-planes-own-frame-2026-11-11)
  labels the ground truths' points `ok`, `A1` and `A2`). Their reference is the
  stored normal.
- 7 points the maintainer judged good by eye, with their stored normals.
- 2 points with normals the maintainer set by hand.

By dataset that is 18 seoul_bull points (10 `ok`, 6 judged good, 2 with hand
normals), 15 kerry_park points and 26 dino points.

**(b) Fresh solves:** seoul_bull and dino_dog_toy solved by the dataset scripts
(`scripts/init_dataset_seoul_bull.sh`, `scripts/init_dataset_dino_dog_toy.sh`),
converted to patches whose normals face the mean viewing direction (what `sfm
xform --to-embedded-patches` stores), with no normal refinement. 150 random
points from each were matched to the nearest ground-truth point, after a
similarity fitted on the camera centres maps the solve onto the ground truth; a
match is kept when it lies within the ground-truth point's patch half-extent and
the two points share at least 2 images. That left 68 seoul_bull and 26 dino
matches, each compared with the ground-truth point's normal.

**(c) `A2` points, with no reference:** 36 points of the ground truths whose
grown-plane answer depends on its start: 16 seoul_bull, 18 kerry_park and 2
dino. Only their outcomes are read.

**Caveat.** Every reference normal except the two hand normals was itself
produced by the grown-plane step, run on the ground truths' cleaner tracks. The
references therefore lean toward the grown plane's answers, and a difference
between the two steps measured against them may favour the grown plane.

**Patch half-size.** The median half-size of the points' patches in their
photographs, in pixels, which bounds what a support of one patch can see:

| Set | gt seoul | gt kerry | gt dino | fresh seoul | fresh dino |
|---|---:|---:|---:|---:|---:|
| Median half-size, px | 12.7 | 14.5 | 21.6 | 3.4 | 9.5 |

## The prototype's crops

The first prototype's runs on real photographs used two crop designs, neither
of them the one the draft proposes; the later runs of the revised design are
described in § "Real points with the revised design". (The runs on synthetic
scenes, § "Synthetic scenes", rendered their crops by casting rays onto the
true surface and test the σ and outcome rules, not the crops.)

- **The runs against the grown plane** (§ "Fresh solves against the grown
  plane", made with the prototype's `pairflow.py` and scripts like it): each
  view's crop was 3 times the patch in world space, sampled at one sample per
  photograph pixel, with a floor of 32 samples.
- **The support-rule runs** (§ "Choosing the support"): each crop was 48
  photograph pixels in half-width at the point's median scale, on one grid per
  point, sampled bilinearly with no prefilter. Views nearer than the median
  were downsampled onto the grid without a pyramid and so aliased; views
  farther than the median were upsampled.

The draft proposes a different design: every view rendered on one common plane,
iterated over rounds from the given normal, a grid of one sample per pixel of
the finest view within 24 to 128 samples per patch side, views finer than the
grid prefiltered through their photographs' pyramids, a crop floor in pixels
of the finest view, each pair read at the pyramid level of its coarser view,
radii in patch half-extents, and σ the larger of a texture σ and a residual σ.
§ "Real points with the revised design" measures the common plane, the rounds,
the grid, the σ and the crop floor, and records the other settings those runs
used; those were not measured against other values.

## Flow settings

**Question.** Does `DisFlowParams::high_quality()` measure the flow between two
crops well enough, or does it need full resolution?

**Setup.** Synthetic pairs: a real crop, warped by the flow a known plane
predicts, so the true flow is known exactly. Six points, two normals each.
`high_quality` against the same preset with `finest_scale` 0, `patch_stride` 1
and variational refinement on.

**Result.**

| Flow | Median flow error | Median normal error |
|---|---:|---:|
| `high_quality` | 0.093 px | 2.0° |
| full resolution, stride 1, variational | 0.040 px | 0.9° |

`high_quality` fixes `finest_scale` at 1, and on crops this small the automatic
coarsest scale is 0 or 1, so it solves the whole flow at half resolution and
upsamples it.

**Geometry check.** Fitting the pair model to the exact predicted flow of a
known plane, with no flow computed, recovered the normal to 0.00°.

## Fresh solves against the grown plane

**Question.** On points straight out of a solve, does the flow fit get closer
to the ground truth than the mean viewing direction and the grown-plane step?

**Setup.** Reference set (b). This run predates the support rule and the
fixed-size crops: it used OpenCV's DIS (patch 12, stride 1, finest scale 0, 16
iterations, 5 variational iterations), crops of 3 times the patch in world
space at one sample per photograph pixel with a 32-sample floor
(§ "The prototype's crops"), and a fixed
disc of 0.75, 1, 1.5 or 2 times the patch. The grown plane ran from the mean
viewing direction.

**Result.** Angle from the ground-truth normal.

| | Mean-view start | Grown plane | Flow fit 0.75× | 1× | 1.5× | 2× |
|---|---:|---:|---:|---:|---:|---:|
| seoul_bull, median | 32.7° | 23.0° | 16.8° | 15.0° | 14.0° | 13.5° |
| seoul_bull, within 20° | 15% | 43% | 59% | 63% | 71% | 69% |
| dino, median | 37.0° | 5.4° | 7.7° | 3.6° | 2.8° | 3.0° |
| dino, within 10° | 0% | 69% | 73% | 77% | 85% | 81% |

On seoul_bull, whose fresh patches are 3.4 px in half-size, the flow fit at
every size is closer than the grown plane. On dino the flow fit is closer from
1× the patch up.

## Choosing the support

**Question.** Which rule should choose the disc the normal is read over?

**Setup.** Both reference sets, split into five. sfmtool's own DIS
(`sfmtool.flow.compute_optical_flow`, `high_quality` with `finest_scale=0`,
`patch_stride=1`), crops of 48 px half-width at the point's median scale on
one grid per point, sampled bilinearly with no prefilter (§ "The prototype's
crops"), the radii 4, 6, 8, 11, 16, 22, 32
and 45 px, and the fits and combination of the draft, but with a σ that has no
texture term: the geometric Hessian's inverse times the flow residual variance,
with the correlation correction (the draft adds the texture term; § "Fitting
each pair at each radius" there). Each rule picks one radius per point.

**Result.** Median angle from the reference: the mean over the five sets, then
each set.

| Rule | Mean | gt seoul | gt kerry | gt dino | fresh seoul | fresh dino |
|---|---:|---:|---:|---:|---:|---:|
| Largest radius with mean χ² ≤ 2 | **5.4°** | 6.9° | 5.6° | 2.0° | 8.7° | 3.7° |
| Fixed 8 px (best fixed radius) | 6.3° | 7.0° | 6.0° | 3.0° | 10.4° | 5.1° |
| Fixed 16 px | 7.4° | 9.9° | 12.9° | 2.1° | 8.6° | 3.3° |
| Fixed 1× patch | 8.7° | 11.0° | 12.9° | 1.7° | 13.2° | 4.5° |
| Fixed 2× patch | 10.9° | 18.2° | 19.0° | 3.6° | 11.1° | 2.8° |
| Per-pair growth (σ ≤ 8°, then stop at a step > 3σ) | 6.9° | 9.5° | 9.6° | 2.4° | 10.4° | 2.6° |
| Radius of smallest σ | 12.7° | 25.0° | 19.8° | 2.8° | 13.2° | 2.8° |

The χ² rule has the lowest mean and is within 1.1° of the best rule on every
set. Fixed sizes in units of the patch do well on dino, whose patches are large,
and badly on seoul_bull and kerry_park. The radius of smallest σ picks large
discs, since one pair's σ keeps falling as the radius grows, across a fold
as well. The χ² threshold and the radius list were tuned on these references.

## Real points with the revised design

**Question.** On real points, how do the revised grid, the common plane and its
rounds, and the revised σ compare with the first prototype, and what outcomes
do they give where there is no reference?

**Setup.** A second Python prototype on reference sets (a), (b) and (c). Every
run used the texture-weighted σ with the 99.9% noise threshold and the two-pair
agreement rule; the rows below name what else each used. Each figure is the
median angle from the reference per set, in the order gt seoul / gt kerry / gt
dino / fresh seoul / fresh dino, and the mean of the five medians. For
comparison, the first prototype's largest radius with mean χ² ≤ 2, on its
pixel-based crops (§ "Choosing the support"), read 6.9 / 5.6 / 2.0 / 8.7 / 3.7,
mean 5.4°.

The prototype's other settings, the same in every run of this section unless a
row says otherwise:

- crop half-width 3 half-extents or 48 px, whichever is larger; the 48 px of
  the pair's coarser view, except in the last runs, which count the finest
  view's (§ "The crop floor and the bar on the mean viewing direction");
- at most 512 samples a side per render;
- radii 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8 and 12 half-extents;
- each pair read at level `floor(log2 k)` of its coarser view;
- at most 10 pairs per point, chosen at random;
- `flow_floor_px` 0.03;
- the correlated square 8 px of the coarser view at the pair's level;
- the noise threshold at the 0.999 quantile.

**A grid of 24 samples per patch side, a fronto plane per view, the texture σ
alone.** The best setting tried had a mean of the set medians of 7.7° to 9°,
the set medians 11° to 20° / 9° to 14° / 1.1° to 4° / 8° to 10° / 3°. With σ ≤
10° and χ² ≤ 2, only 40% to 44% of the curated kerry_park and seoul_bull points
turned. A larger flow floor, 0.1 or 0.2 px, did not help. On the dino ground
truth's patches, about 43 px across, the fixed grid lost detail: at a fixed
radius of 1 half-extent the error was 2.9°, against 1.7° with the first
prototype's pixel-based crops and 1.3° at native resolution.

**The texture σ alone is too small.** On good points the combined normal was
close to the reference (on seoul_bull point 157, 1.6° to 8° at every radius),
but the mean χ² was 44 to 2,500, and 600 to 50,000 on the points whose grid had
90 or more samples per patch side. σ the larger of the texture σ and the
residual σ (the draft's step 6) is the draft's answer to this, and the runs
below use it.

**The native grid and σ the larger of the two, a fronto plane per view** (σ ≤
10°, χ² ≤ 1):

| | gt seoul | gt kerry | gt dino | fresh seoul | fresh dino | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Median angle | 12.6° | 5.9° | 1.4° | 8.3° | 1.8° | 6.0° |
| Turned | 50% | 40% | 69% | 82% | 92% | |

**The common plane, 3 rounds from the mean viewing direction**, the native grid
and σ the larger of the two, before the two changes of § "The crop floor and
the bar on the mean viewing direction":

| Rule | | gt seoul | gt kerry | gt dino | fresh seoul | fresh dino | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| σ ≤ 10°, χ² ≤ 1 | median angle | 15.3° | 3.3° | 0.9° | 9.0° | 1.8° | 6.0° |
| | turned | 67% | 53% | 85% | 82% | 96% | |
| σ ≤ 10°, χ² ≤ 2 | median angle | 15.2° | 3.0° | 1.2° | 10.7° | 2.6° | 6.5° |
| | turned | 72% | 73% | 92% | 85% | 96% | |

At the same rule the common plane turns as many points as the fronto planes or
more, and is closer on kerry_park and the dino ground truth; it is further on
seoul_bull.

**By round**, the median angle from the reference at the start and after each
round. Each next round's plane was the answer by the rule at σ ≤ 10° and mean
χ² ≤ 2, or, with no answer, the fallback at 1 half-extent:

| Set | Start | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|---:|
| gt kerry | 48.9° | 16.3° | 9.1° | 4.3° |
| gt dino | 28.8° | 2.3° | 1.2° | 1.1° |
| fresh dino | 37.0° | 2.4° | 2.2° | 2.7° |
| gt seoul | 42.3° | 17.2° | 18.5° | 17.7° |
| fresh seoul | 32.7° | 10.2° | 10.8° | 12.3° |

kerry_park is still improving at round 3. seoul_bull, whose pairs are 2° to 5°
apart, stops improving after round 1; the first prototype got 6.9° on the
curated seoul_bull points, and what in the revised design costs accuracy there
is not yet isolated.

**Six points traced**, kerry_park 93, 158 and 343 and dino 1172, 44 and 204.
With a fronto plane per view, 4 were `Unstable` and 2 `Turned`, at 3.3° and
0.6° from the reference. On the common plane all 6 turned, from 13.6° to 59.9°
at the mean viewing direction to 1.1° to 4.3° after 3 rounds; kerry_park 158
went 59.9° → 1.1° → 1.3° → 1.6°, and dino 44 went 23.6° → 19.3° → 13.7° → 3.6°.
On kerry_park 93, whose pairs are 89° to 113° apart, the flow between fronto
crops was chaotic and its leftover over 1 px almost everywhere; on dino 1172,
whose pairs are 30° to 50° apart, the fronto crops were rotated and sheared by
several pixels and the flow was reliable only near the centre.

**`A2` points** (set (c)), the common plane, σ ≤ 10°, χ² ≤ 2:

| Set | Turned | Single pair | Undetermined | Unstable |
|---|---:|---:|---:|---:|
| seoul_bull | 6 | 1 | 3 | 6 |
| kerry_park | 13 | 0 | 0 | 5 |
| dino | 1 | 0 | 0 | 1 |

The outcomes do not yet tell these points from good ones: 13 of the 18
kerry_park `A2` points turned. They may be determined, with wide baselines; the
`A2` label comes from the grown plane's instability, not from the truth.

### The crop floor and the bar on the mean viewing direction

**Question.** Two failures showed in the runs above. With the crop floor in
pixels of the pair's coarser view, a point with one far view got pair squares,
and so allowed radii, that were large in the world: on seoul_bull points 70,
173 and 255 the rule picked 4 to 12 half-extents, about 76 to 152 pixels of the
finest view, where the first prototype, with crops set in pixels, never went
past 45 px. And some rounds drifted to planes the views cannot see. What do a
floor in pixels of the finest view, and a bar that makes a normal more than 80°
from the mean viewing direction of the views no answer, change?

**Setup.** As above, the common plane with 3 rounds from the mean viewing
direction, with both changes.

**Result.**

| Rule | | gt seoul | gt kerry | gt dino | fresh seoul | fresh dino | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| σ ≤ 10°, χ² ≤ 1 | median angle | 14.7° | 3.2° | 0.9° | 8.6° | 1.8° | 5.9° |
| | turned | 67% | 80% | 88% | 84% | 96% | |
| σ ≤ 10°, χ² ≤ 2 | median angle | 15.8° | 3.5° | 1.3° | 9.8° | 2.2° | 6.5° |
| | turned | 72% | 80% | 92% | 90% | 96% | |

By round, the median angle from the reference:

| Set | Start | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|---:|
| gt seoul | 42.3° | 17.4° | 17.2° | 18.3° |
| gt kerry | 48.9° | 14.2° | 5.9° | 3.7° |
| gt dino | 28.8° | 2.8° | 1.4° | 1.2° |
| fresh seoul | 32.7° | 10.4° | 11.5° | 11.0° |
| fresh dino | 37.0° | 2.7° | 2.1° | 2.4° |

`A2` points, χ² ≤ 1:

| Set | Turned | Single pair | Undetermined | Unstable |
|---|---:|---:|---:|---:|
| seoul_bull | 7 | 0 | 2 | 7 |
| kerry_park | 8 | 0 | 0 | 10 |
| dino | 1 | 0 | 0 | 1 |

kerry_park turns more of its curated points (80% against 53% to 73%) and fewer
of its `A2` points (8 of 18, against 11 to 13).

**The curated seoul_bull points, one by one.** The `ok` points 1, 2, 109, 143
and 157 end 1.2° to 7.1° from the reference, `ok` point 10 ends 16° from it,
and `ok` points 18, 120 and 167 are `Unstable`. The median is set by the judged
and hand points, 63 at 33°, 70 at 22°, 173 at 25°, 207 at 112°, 255 at 46° and
238 at 9°, and by the two-view points: 160, a single pair, at 68°, and 192 and
260 `Undetermined`. The remaining seoul_bull errors are concentrated on layered
and narrow-baseline points, not on the `ok` points.

**Drift.** The bar does not catch every drift: point 207, a bush with the bull
in front and trees behind, ends 112° from the reference while within 80° of the
mean viewing direction. Point 160, with two views and so one pair, drifted 14°
→ 40° → 55° → 68° from the reference over the rounds; with no second pair
nothing tests a round's answer, which is why the draft runs one round for a
point with one pair.

**Cost.** About 4 to 8 minutes for the 189 points with 12 processes per
configuration, more with the native grid and 3 rounds.

## Synthetic scenes

**Question.** Do the revised σ and outcome rules give `Undetermined` on flat
and weak texture, an honest σ on strong texture, and what do they give on a
crease through the point? How large should the flow floor be?

**Setup.** A Python prototype of the draft's revised σ and outcome rules: σ
weighted by the structure tensor with the 99.9% noise threshold, the floor on
the flow's precision where stated, two determined pairs needed at a radius, and
a mean χ² of at most 2. It is a prototype of those rules, not of the draft's
crop sizing: each crop was rendered by casting each sample's ray onto the true
surface, at one sample per photograph pixel.

- **Scenes.** A textured plane, or a 90° crease with the point on its edge.
- **Cameras.** 4 on a ring 8° or 2° off the axis, so pairs are up to 16° or 4°
  apart.
- **Images.** Noise of 2 grey levels; texture a smooth random field whose
  standard deviation, the contrast, is 0 to 30 grey levels.
- **Repeats.** 4 seeds for each setting.

**Flat and weak texture.** At contrast 0, 0.5, 1, 2 and 4 every run was
`Undetermined`, in all 4 seeds on both rings, and none was `Unstable`, with the
flow floor and without it. At contrast 8 the runs were `Turned`, except on the
2° ring with a floor of 0.05, where 3 were `Turned` and 1 `Undetermined`.

**Accuracy when turned.** On the 8° ring at contrast 8 to 30 the normal was
0.1° to 0.8° from the true one; on the 2° ring, 0.1° to 1.9°. The chosen radius
was 3 half-extents, the largest tried, and 2 in a few runs on the 2° ring.

**σ and the flow floor.** Without a floor, σ was too small where the texture
was strong: at contrast 16 to 30 the error was 2 to 4 times σ, for example an
error of 0.1° with σ 0.04°, and one clean plane of the 24 came out `Unstable`
because χ² was inflated. With each correlated square's covariance modelled as
`W⁻¹ + f²·I`:

| Floor `f` | Error over σ, strong texture | Clean planes `Unstable` |
|---|---|---:|
| none | 2 to 4 | 1 of 24 |
| 0.02 px | 1 to 2 | 0 |
| 0.05 px | under 1, slightly cautious | 0 |

The draft's default, 0.03, lies between the two floors measured.

**A crease through the point**, at contrast 30:

| Floor | Ring | Outcome | Distance from the faces' bisector | Mean χ² |
|---|---|---|---|---|
| none | 8° | `Unstable` | — | 2.0 to 7.2 at 0.5 half-extents, rising to thousands at 3 |
| none | 2° | `Turned` at 1 half-extent | 0.3° to 1.8° | |
| 0.03 | 8° | `Turned` at 0.75 to 1 half-extent | 2.6° to 3.2° | about 2 or less there, then 12 to 15 at 1.5 and 160 to 176 at 3 |
| 0.03 | 2° | `Turned` at 3 half-extents | 0.7° to 0.8° | 0.7 or less at every radius |

On the 2° ring the narrow pairs all see the same compromise between the faces,
so they agree at every radius. On a crease through the point the rule shrinks
the support to where the pairs still agree and turns the patch to about the
bisector of the faces; it gives `Unstable` only where no radius agrees.

**The noise estimate.** Immerkær's Laplacian estimate of the image noise
recovered the true standard deviation of 2.0 as 2.02.

## Rejected alternatives

**A boundary from a fixed pixel threshold.** Fit an affine flow or a plane to
the inner disc, and stop growing where the next ring departs from that fit by
more than a fixed number of pixels. The departure grows smoothly with the radius
(on dino's `ok` points, from 0.5 px at 0.5 times the patch to 4.0 px at 2.75
times) and faster on wider baselines, so the threshold measures the baseline
rather than a boundary. Every threshold tried scored about the same, 3.0° to
3.9° median on the curated set.

**A threshold relative to a 10° tilt.** Stop where the next ring departs by
more than the flow change a 10° tilt would make. It stopped at the first ring
every time: at small radii a 10° tilt moves the flow less than the flow's noise,
0.1 to 0.3 px.

**Iterative warp and refit.** Warp each view to the current plane, compute the
flow, fit heights to it, and repeat. It converged to the same plane from every
start, but to a biased one: on baselines of about 2°, a flow slope of 0.2 to
0.3 px across the patch turns the plane 20° to 30°. The grown plane's
photometric score preferred the stored normals to the planes it reached.

## Cost

The prototype, in Python, took about 59 s for 153 points with 12 processes,
about 4.6 CPU-seconds per point. Most of it was the fits; the flows were one
full-resolution flow on a 96×96 crop per pair.
