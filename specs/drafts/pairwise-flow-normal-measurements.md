# Pairwise Flow Normal Measurements

**Status:** Draft. Measurements of a Python prototype, made outside the
repository, behind [pairwise-flow-normal.md](pairwise-flow-normal.md). No code in
the repository produced these numbers, and the prototype's scripts are not
checked in.

The measurements bear on four choices the draft makes: the flow settings, the
crop's size, the rule that chooses the support, and what the step costs. Each
section states the question, the setup and the result.

## Reference sets

Two sets of points with a reference normal each, five sets in all when split by
dataset.

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

**(b) Fresh solves:** seoul_bull and dino_dog_toy solved by the dataset scripts
(`scripts/init_dataset_seoul_bull.sh`, `scripts/init_dataset_dino_dog_toy.sh`),
converted to patches whose normals face the mean viewing direction (what `sfm
xform --to-embedded-patches` stores), with no normal refinement. 150 random
points from each were matched to the nearest ground-truth point, after a
similarity fitted on the camera centres maps the solve onto the ground truth; a
match is kept when it lies within the ground-truth point's patch half-extent and
the two points share at least 2 images. That left 68 seoul_bull and 26 dino
matches, each compared with the ground-truth point's normal.

**Caveat.** Every reference normal except the two hand normals was itself
produced by the grown-plane step, run on the ground truths' cleaner tracks. The
references therefore lean toward the grown plane's answers, and a difference
between the two steps measured against them may favour the grown plane.

**Patch half-size.** The median half-size of the points' patches in their
photographs, in pixels, which bounds what a support of one patch can see:

| Set | gt seoul | gt kerry | gt dino | fresh seoul | fresh dino |
|---|---:|---:|---:|---:|---:|
| Median half-size, px | 12.7 | 14.5 | 21.6 | 3.4 | 9.5 |

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
iterations, 5 variational iterations), crops of 3 times the patch, and a fixed
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
`patch_stride=1`), crops of 48 px half-width, the radii 4, 6, 8, 11, 16, 22, 32
and 45 px, and the fits and combination of the draft. Each rule picks one
radius per point.

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
