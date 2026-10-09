# Patch Footprint Selection Measurements

This file records the measurements behind [patch-footprint-selection.md](patch-footprint-selection.md), the draft that chooses each track's patch half-extent from a short ladder of sizes, as the smallest size whose self-similarity radius, member coherence and sharpness readings call good. The two checked-in ground truths carry patch half-extents that the maintainer set by hand, point by point, so they are a human answer to the question the draft's rule answers. The measurements bear on four decisions the draft leaves open: whether the ladder should be in multiples of the SIFT detector scale, what the rule's gates should be, whether the rule reproduces the human choice, and at what grid density the readings must be taken.

## Setup (2026-10-09)

- **Code.** Commit `685c5cef` (branch `bootstrap-core-migration`), with the extension rebuilt by `pixi run -e test maturin develop --release`.
- **Machine.** Windows 11, Intel Core i9-14900HX, 63.7 GB RAM. Nothing here is timed.
- **Ground truths.** `test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr` (280 points, 14 at infinity) and `test-data/images/kerry_park/kerry_park_ground_truth.sfmr` (391 points, 9 at infinity). No other ground-truth `.sfmr` has been checked in: `git log --all` over `test-data/**/*.sfmr` lists only these two. Points at infinity have no world size and are left out, as are points for which the reference-view rule picks no observation. That leaves 263 points on seoul_bull and 380 on kerry_park.
- **Reference view.** Neither file stores a reference observation (`reference_observations` is `-1` throughout), so each point's reference observation is the one `PatchCloud.render_bitmaps` picks by the reference-view rule at `R = 24`.
- **Human size.** The stored half-extent `|u|`, `|v|` (their mean; the two differ by at most 0.065 m on seoul_bull and are equal on kerry_park). In reference-view photograph px, the half-extent is `(R/2)·sqrt|det J|` for the Jacobian `J` that `OrientedPatch.render_view_tile` returns at `R = 24`. This is the half-side of the square of equal area.
- **Detector scale.** `.sift` files extracted with the same feature options as the ground truths (`sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9`), copied from `C:/DataSets/SeoulBull` and `C:/DataSets/KerryPark480`. The images are byte-identical to `test-data`. `σ` is the feature size `(‖A₀‖ + ‖A₁‖)/2` of the affine shape `A`. The fleet default is a full edge of `12σ` (`cluster-patches --patch-size 12`), which is a half-extent of `6σ`. Every ratio below is half-extent over `σ`, so the fleet default reads 6.
- **Ladders.** Two ladders per point, each rendered on the reference view anchored at the stored keypoint:
  - a *human ladder*: 0.5, 0.75, 1, 1.5 and 2 times the human half-extent;
  - a *σ ladder*: half-extents of 3, 4.5, 6, 9 and 12 `σ`, the draft's `[6, 9, 12, 18, 24]` as full edges.
- **Readings at each rung**, at two grid densities:
  - *(a) fixed*: `R = 24`;
  - *(b) native*: `R = round(2·h)` for a half-extent of `h` reference-view px, clamped to 12..64, so one grid sample is about one photograph px.

  The readings are `zncc_self_similarity_parts` (whole, middle and nine cell radii, `max_radius = 3`, default tolerances), `assess_blur` on the same tile, and the median off-diagonal pairwise ZNCC of `PatchCloud.validate_member_coherence` (`return_matrix=True`, at the same `R`, the other options at their defaults) over the point's whole track.
- **The rule** is the draft's: the smallest rung where the whole and middle radii and at least 5 of the 9 cells are at or under the radius gate, and the coherence is within the drop gate of the ladder's maximum. The radius gate is a quantile (0.5 or 0.75) of the whole radii at `6σ`, or the fixed 2.5 grid px bar. The drop gate is a quantile (0.5, 0.75 or 0.9) of the positive coherence drops between adjacent rungs of the σ ladder. The draft's "quantile of the per-track drop" over all drops has a median at or below 0 on both captures (coherence rises with size more often than it falls), so a quantile of all drops makes a gate under 0, which no rung but the maximum passes. The quantiles are taken separately for each density.
- **Clusters.** The `--piecewise` cluster-patch files written by `sfm cluster-patches` at the same commit from the workspaces' `*-clusters.matches` (seoul_bull: `20261007-00-cluster-seoul_bull_sculpture_1-17-clusters-patches.matches`; kerry_park: `kerry_park-clusters-patches.matches`), `patch_size 12`, `resolution 25`.
- **Scripts.** `measure.py`, `analyze2.py` and `analyze3.py` in the session scratch directory `gt-sizes/`, run with `pixi run -e dev python`. They are not checked in.

## What the human chose

**Question.** How large are the hand-set patches, in the world, in reference-view px and in multiples of the detector scale, and does the multiple stay constant? If it does, a fixed multiple of `σ` is the right family of sizes and only its value is in question. If it does not, the ladder should be in some other unit.

**Result.**

| | seoul_bull | kerry_park |
|---|---|---|
| Half-extent, world (m), median (IQR) | 0.164 (0.085–0.327) | 0.497 (0.369–0.730) |
| Half-extent, reference px, median (IQR) | 9.9 (7.2–13.9) | 10.5 (7.3–13.9) |
| Half-extent, reference px, p5 / p95 | 5.3 / 25.1 | 4.1 / 17.9 |
| GT keypoint to nearest SIFT keypoint, median | 2.9 px | 3.6 px |
| Share with a SIFT keypoint within 1 / 2 px | 9% / 29% | 7% / 22% |
| Half-extent / `σ`, points within 2 px (n) | 75 | 84 |
| median (IQR) | 7.0 (4.7–10.2) | 4.2 (2.9–8.9) |
| p5 / p95 | 2.2 / 20.2 | 2.1 / 17.5 |
| Half-extent / `σ`, per-point median over all views (n) | 7.6 (4.8–11.5), 156 | 4.2 (2.9–6.6), 224 |
| Slope of log half-extent px on log `σ` | 0.11 | 0.02 |
| Slope of log half-extent world on log depth | 0.82 | 0.71 |

The ground-truth keypoints are not SIFT detections. Every ground-truth observation was compared with the nearest keypoint in its image's `.sift` file, and with the nearest keypoint to the same position moved 10 px in a random direction. The two distances have about the same median on seoul_bull (3.05 against 3.28 px), and on kerry_park the stored positions are only somewhat closer (3.49 against 4.10 px). A half-pixel offset or a factor of two in either coordinate does not bring them closer. So the "detector scale" of a ground-truth point is the scale of a nearby feature, not of the feature the point was found from, and the ratios above are taken only on the minority of points with a keypoint within 2 px.

On those points, the human's half-extent in photograph px does not follow `σ`: a regression of log half-extent px on log `σ` has a slope of 0.11 and 0.02, where a fixed multiple of `σ` would give 1. The ratio therefore falls as `σ` grows (Spearman −0.72 and −0.63), from a median of 10.4 and 7.6 on the smallest third of `σ` to 3.7 and 2.8 on the largest third. On the same points the spread of the half-extent in px (standard deviation of log₁₀, 0.21 and 0.24) is smaller than the spread of its ratio to `σ` (0.29 and 0.32). In the world, the half-extent grows with depth at a slope of 0.7 to 0.8 on a log-log scale, so the human sized nearer surfaces smaller in metres but not in proportion. On kerry_park the px size also shrinks with depth (Spearman −0.61), so a distant surface gets a smaller footprint. Viewing angle, the contrast of the tile and its self-similarity radius each correlate with the ratio by less than 0.5 in magnitude.

**Decision.** The human's choice is close to a fixed footprint in the reference photograph, a half-extent of about 10 px (a 20 px edge) with an IQR of 7 to 14 px, and is not a fixed multiple of the detector scale. Against the fleet's `6σ` half-extent, the human median is 1.2 times larger on seoul_bull and 0.7 times on kerry_park, with a spread of a factor of 2 either way within each capture. A ladder in multiples of `σ` therefore spends its rungs on the variation of `σ`, which the human ignored. That argues for a ladder in reference-view photograph px, as [Readings at two densities](#readings-at-two-densities) also does. Whether a rung should then be scaled by depth or by anything else is not settled by two captures.

## Whether the readings predict the human's size

**Question.** On the human ladder, does each of the draft's readings put the human's size where the draft's rule would put it: the smallest size that passes the radius gate, with the coherence still on its plateau? If it does, the rule can stand in for the hand-set sizes.

**Result.** Agreement is the share of points where the rule picks the human's rung (1×), or a rung within one step of it (0.75× to 1.5×).

| Gate (radius / drop) | Density | Rule picks 1× | Within one rung | Smaller | Larger or none |
|---|---|---|---|---|---|
| seoul_bull, q0.75 / q0.75 | fixed | 21% | 66% | 27% | 52% |
| | native | 16% | 48% | 29% | 54% |
| seoul_bull, 2.5 / q0.75 | fixed | 18% | 65% | 41% | 40% |
| | native | 17% | 63% | 52% | 30% |
| kerry_park, q0.75 / q0.75 | fixed | 20% | 59% | 39% | 41% |
| | native | 15% | 49% | 40% | 45% |
| kerry_park, 2.5 / q0.75 | fixed | 18% | 56% | 70% | 12% |
| | native | 14% | 58% | 74% | 12% |

The captured gates were 0.94 / 1.32 (seoul_bull, fixed, q0.5 / q0.75) and 0.68 / 0.97 (kerry_park) grid px for the radius, and 0.055 and 0.081 for the drop at q0.75. At native density the radius gates are about half as large (0.43 / 0.67 and 0.35 / 0.61). The q0.5 radius gate leaves 28% to 67% of points with no rung, depending on capture and density, and agrees with the human on 8% to 19%.

Taken one at a time:

- **Self-similarity radius (lower bound).** The human's size passes the radius gate on 55% to 94% of points, depending on the gate. It sits at the gate's boundary, passing at 1× and failing at 0.75×, on only 7% to 16%. Smaller sizes than the human's are localisable on most points: at the 2.5 bar the radius-only rule picks 0.5× on 65% (seoul_bull) and 72% (kerry_park) of points at fixed density, and on 59% and 61% at native density. At native density the median whole radius is smallest at 0.75× to 1× on both captures (seoul_bull 0.44 / 0.46 px, kerry_park 0.35 / 0.35 px) and rises at 1.5× and 2×. So the human did not size patches to the smallest localisable size.
- **Member coherence (upper bound).** The human's size is on the coherence plateau (within the drop gate of the ladder's maximum) for 95% to 99% of points, but the plateau ends at the human's size for only 10% to 32%; on most points coherence stays within the gate up to 2×. Per point, the rung of greatest coherence is spread over the whole ladder: at native density 1× is the most common on kerry_park (34%) and 2× on seoul_bull (35%). Across points, the median coherence peaks at 1× at native density on both captures (seoul_bull 0.851, kerry_park 0.842, falling to 0.831 and 0.748 at 2×). At fixed density it peaks at 1× on kerry_park (0.849) but keeps rising to 2× on seoul_bull (0.879).
- **Blur-matched sharpness.** `assess_blur`'s `semi_axes` on a tile are that tile's own self-similarity ellipse, so its major axis equals the whole radius on every tile measured, and the reading adds nothing to the radius as a size reading. As the draft says, it bounds the density and not the size.

**Decision.** The rule as drafted does not reproduce the human's sizes: it picks the human's rung on 14% to 21% of points and a rung within one step on 48% to 66%, whichever gate is used, and it mostly picks smaller. The radius reading is a lower bound the human stayed well above. The coherence reading has a capture-level maximum at the human's size at native density, but per point it does not cap the size within a factor of two. What would settle the upper bound is a ladder that runs past 2× at native density, to find where coherence falls per point. The rule's agreement with the human depends on the radius gate more than on anything else, so the gate cannot be left to a quantile chosen without a target. The human sizes are such a target.

## Sizes the rule picks on the σ ladder

**Question.** On the draft's own ladder, half-extents of 3 to 12 `σ`, what does the rule pick, compared with the human's size?

**Result.** The ratio of the rule's half-extent to the human's, over the points the rule picks a rung for. The last four columns are shares of all points, so they add to 100%:

| Gate (radius / drop) | Density | Median (IQR) | p5 / p95 | Within ×1.5 | Below ÷1.5 | Above ×1.5 | No rung |
|---|---|---|---|---|---|---|---|
| seoul_bull, q0.75 / q0.75 | fixed | 0.87 (0.61–1.28) | 0.34 / 2.49 | 45% | 25% | 14% | 15% |
| | native | 0.82 (0.60–1.18) | 0.31 / 1.69 | 32% | 22% | 6% | 40% |
| seoul_bull, 2.5 / q0.75 | fixed | 0.80 (0.49–1.12) | 0.28 / 2.05 | 42% | 40% | 13% | 5% |
| | native | 0.77 (0.49–1.11) | 0.28 / 1.95 | 41% | 41% | 12% | 6% |
| kerry_park, q0.75 / q0.75 | fixed | 0.85 (0.62–1.20) | 0.39 / 1.91 | 43% | 25% | 10% | 22% |
| | native | 0.75 (0.59–1.08) | 0.39 / 1.55 | 37% | 23% | 4% | 36% |
| kerry_park, 2.5 / q0.75 | fixed | 0.70 (0.47–0.99) | 0.28 / 1.84 | 44% | 44% | 7% | 4% |
| | native | 0.70 (0.49–1.00) | 0.34 / 1.83 | 46% | 44% | 4% | 4% |

The fleet's `6σ` half-extent, against the human's: on the points with a SIFT keypoint within 2 px, its median is 0.85 times the human's on seoul_bull and 1.4 times on kerry_park, the inverse of the ratios in [What the human chose](#what-the-human-chose).

**Decision.** The rule on the σ ladder picks a size within a factor of 1.5 of the human's on only a third to under half of the points, and is smaller than the human's by more than that factor on a quarter to under half. It is no closer to the human than the fleet's single `6σ` default. Per-point sizing by this rule is not supported by these data in its drafted form.

## Readings at two densities

**Question.** At a fixed `R` over a growing footprint, a reading can confound a surface with no texture with a texture the grid has averaged away: sand is textured at the scale of a photograph pixel and turns uniform when the grid is coarser than the photograph, while building siding is flat over a small footprint and gains structure once a window enters. Do the fixed grid (a) and the native grid (b) disagree on the smallest good size, and does the human's choice agree better with (b)?

**Result.** On the same gate recipe, the two densities choose the same human-ladder rung for 48% (seoul_bull) and 61% (kerry_park) of points at the captured q0.75 gates, and 58% and 66% at the 2.5 bar. The human's rung is chosen more often at fixed density (18% to 21%) than at native density (14% to 17%). Where the two disagree, the fixed density's pick is the nearer to the human on 68% and 65% of those points at the captured gates, and the two are even at the 2.5 bar (fixed nearer on 47% and 50%, native nearer on 44% and 48%).

Rung by rung, at the 2.5 bar, most disagreements are not the case the question describes. They come at small footprints, where the fixed grid is finer than the photograph and the native grid sits at its floor of `R = 12`, whose cells are 4 samples on a side. There the fixed grid passes and the native grid fails: at 0.5×, 26 against 12 points on seoul_bull and 52 against 8 on kerry_park. Where the fixed grid is coarser than the photograph, the two disagree on 1 of 547 point-rungs on seoul_bull and 21 of 785 on kerry_park, 17 of them passing at native density only, at 1.5× and 2×. The human's sizes are about a 20 px edge, so a 24-sample grid is close to native density at the human's own size, and the ladder only reaches a grid about 1.7 times coarser than the photograph at 2×.

**Texture classes.** Each point is classed by its native-density radius pass at the 2.5 bar over the human ladder:

| Class | seoul_bull | kerry_park | What it is |
|---|---|---|---|
| Scale-free | 150 | 208 | Passes at every rung from 0.5×. Lawn, gravel, foliage, the painted stripes of the sculpture: texture at every scale the ladder reaches. Median human half-extent 12.6 / 12.5 px, the largest of the classes. |
| Flat then structured | 98 | 128 | Fails at 0.5×, passes from some larger rung on. On 62 and 87 points structure arrives at 0.75×, on 20 and 24 at the human's 1×, and on 16 and 17 only past it. Dark bushes, the grey rock behind the sculpture, kerry_park's pale paving, and a blue utility box whose door edge enters the patch only at 1×. Median human half-extent 7.5 / 7.9 px. |
| Fine, native only | 0 | 12 | Passes at native density on a rung where the fixed grid is coarser than the photograph and fails. A hairline crack in concrete, and the fine leaves of a hedge. 10 of the 12 have their greatest coherence at the human's 1×. |
| Not monotone | 12 | 28 | Passes, then fails at a larger rung: a feature that is distinctive at one size and is swamped by a stronger straight edge or a uniform area at the next. |
| None | 3 | 4 | Passes at no rung: the face of a curb or wall where the only structure is one straight edge. |

The contact sheet (`gt-sizes/contact_sheet.png` in the session scratch directory, not checked in) shows 26 points across these classes, with columns for the human size at `R = 24`, the human size at native density, 0.5× and 2× at native density, the σ-ladder rule's pick at native density at the 2.5 bar, and the fleet's `6σ` at native density.

**Decision.** On these two captures the confound the question describes is real but rare: 12 of 380 kerry_park points read as textured only at native density, and none on seoul_bull. The human's choice does not agree better with native density. The larger effect of density is the opposite one: at small footprints a native grid of `R = 12` has cells too small to pass, so the native reading calls small sizes bad that the fixed, oversampled grid calls good. A rule stated in terms of footprint and density should therefore be read on a grid of about one sample per reference photograph pixel, with a floor on `R` large enough that the cells the rule counts hold enough samples. The data here does not fix that floor. `R = 12` is too small for the five-cell clause, and the human's sizes are mostly over a 14 px edge (p25 7.2 px half-extent). Testing whether the fine-texture class grows with the footprint would need a ladder that runs to grids much coarser than the photograph, past the 2× reached here.

## Clusters against the human's size

**Question.** On the same surfaces, how far is the extent the fleet's cluster-patch refinement works at from the human's? The refinement's template is a half-width of 6 keypoint-frame units under the member's refined affine shape, the `12σ` default.

**Method.** Each cluster's reference member was matched to the ground-truth observation in the same image within 3 px. The cluster's extent is `6·sqrt|det S|` for the refined shape `S`. The human's half-extent is projected into the same image at the matched observation as `(R/2)·sqrt|det J|`. That image is the cluster's reference, which is not always the ground truth's reference view.

**Result.**

| | seoul_bull | kerry_park |
|---|---|---|
| Matched clusters (distinct GT points) | 160 (114) | 309 (155) |
| Cluster half-extent px, median (IQR) | 8.1 (6.5–13.6) | 9.7 (7.7–12.9) |
| Human half-extent px in the same image, median (IQR) | 9.7 (7.3–13.7) | 6.1 (4.4–8.8) |
| Cluster / human, median (IQR) | 0.94 (0.66–1.71) | 1.79 (1.16–2.36) |
| Cluster / human, p5 / p95 | 0.34 / 3.17 | 0.50 / 3.88 |

**Decision.** On seoul_bull the fleet's cluster patches have, in the median, the human's footprint. On kerry_park they are 1.8 times larger in the median, and larger on three quarters of the matched clusters. On both, half of the clusters are more than about 1.5 times away from the human's size in one direction or the other. The spread, more than the median, is what a fixed multiple of the detector scale cannot remove. This agrees with [What the human chose](#what-the-human-chose): the human sized by footprint, and the detector's scale varies independently of it.

## A rule for the hand-set sizes (2026-10-09)

**Question.** Is there a rule, stated in readings the pipeline can compute, that reproduces the half-extents the maintainer chose by eye? Two hypotheses shaped the search. First, distinguishing texture can exist at any scale, so a footprint is good when some scale band within it carries localisable structure that the views agree on; the readings are therefore taken over a grid of footprint × scale band, with the band axis a box pyramid of the native-density tile. Second, both ends of the band axis matter: a coarse band over a large footprint can find structure such as windows on siding, and the native band can find fine texture such as sand.

**Data.**

- **Code and machine.** Commit `3d3f6ab6` (branch `bootstrap-core-migration`), with the extension built by `pixi run -e test maturin develop --release`. Windows 11, Intel Core i9-14900HX, 63.7 GB RAM.
- **Points.** The same 263 seoul_bull and 380 kerry_park points, reference views and human half-extents (reference-view px) as in [Setup](#setup-2026-10-09).
- **Footprint ladder.** Half-extents of 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 30 and 40 reference px. The patch is scaled about its stored placement, anchored at the stored keypoint, and rendered at native density, `R = 2·h` (at least 6, at most 80), so one sample covers one reference photograph px.
- **Band axis.** A box pyramid of each tile, halving while the next level has at least 10 samples across, up to four levels (for example 80, 40, 20, 10 at `h = 40`; level 0 only below `h = 10`). A coarse sample has data when all four of its children do.
- **Readings at each (footprint, level).** `zncc_self_similarity_parts` (`max_radius = 3`, default tolerances): whole, middle and nine cell radii in level samples, and converted to reference px. Member coherence: every view of the track is rendered the same way and box-reduced to the same level, and the pairwise ZNCC is taken on the support common to all views, per channel and averaged over channels; the readings are the median off-diagonal ZNCC, the share of views whose median ZNCC to the others is above 0.8, and the reference view's median. This is a reimplementation, not `validate_member_coherence`, because the kernel's coarse tables stop at a factor of 4. At `h = 10` it correlates with the kernel's median at 0.94 (seoul_bull) and 0.92 (kerry_park), and the median absolute difference is 0.02 and 0.03. Also read: mean squared gradient, mean squared Laplacian and grey standard deviation of the tile.
- **Per-point readings.** Depth; viewing angle of the reference; number of views; zoom range (largest over smallest footprint across views); `σ` of the nearest SIFT keypoint when within 2 px; SIFT keypoints within 8 and 16 px; mean squared gradient, squared Laplacian and grey variance of the photograph in disks of 4, 8, 16 and 32 px; distance to the nearest Canny edge (thresholds 100, 200); distance to the nearest and second-nearest other ground-truth point in the reference image, and counts within 10, 20 and 40 px and within 1× and 2× the human half-extent; and the texture class of [Readings at two densities](#readings-at-two-densities).
- **Scripts and outputs.** `features.py`, `analyze.py`, `rule2.py` and `residuals.py` in the session scratch directory `size-rule/`, run with `pixi run -e dev python`; the table is `size-rule/feature_table.npz`. Nothing here is checked in. scikit-learn is not in the `dev` environment, so the tree and the linear models are written in numpy.
- **Scores.** A rule's prediction is compared with the human half-extent in reference px by three numbers, written `w1.25 / w1.5 / m`: the share of points within a factor of 1.25, the share within a factor of 1.5, and the median `|log(rule / human)|`. "Cross" means fitted on one capture and scored on the other. Rules that pick a ladder rung lose little to the ladder's spacing: choosing the nearest rung to each human size scores `m = 0.05`, with 99% to 100% of points within a factor of 1.25.

### Noise floor

Two points count as on the same surface when their normals are within 20°, each centre is within `0.25·d + 0.1·(h₁ + h₂)` of the other's plane (`d` their distance, `h` the world half-extents), and they are within 3 footprint edges, `d ≤ 3·(h₁ + h₂)`. The comparison is of world half-extents, which at that range is the same as comparing px.

| | seoul_bull | kerry_park |
|---|---|---|
| Same-surface pairs | 832 | 2076 |
| Median \|log ratio\| of the pair's sizes | 0.18 | 0.076 |
| RMS log ratio | 0.45 | 0.31 |
| Per-point spread implied by the RMS, `σ = RMS/√2` | 0.32 | 0.22 |
| Best possible score at that spread, `w1.25 / w1.5 / m` | 51% / 80% / 0.22 | 69% / 94% / 0.15 |
| Nearest same-surface neighbour's size as the prediction (n) | 73% / 89% / 0.008 (224) | 84% / 92% / 0.006 (306) |
| Standard deviation of log size, all points | 0.46 | 0.47 |
| ... scale-free class | 0.40 (n 150) | 0.35 (n 208) |
| ... flat-then-structured class | 0.31 (n 98) | 0.40 (n 128) |

The human is very consistent over a surface: a point's nearest same-surface neighbour has nearly the same size as it on most points, with a median difference under 1%. The disagreements that remain are large, often a factor of 2, which is why the RMS is far larger than the median. Read as Gaussian noise, the RMS gives the floor in the fifth row: no rule should be expected to do better than about `m = 0.22` on seoul_bull and `m = 0.15` on kerry_park. Grouping points by texture class removes little of the spread. The class is also defined on the human ladder, so it is not independent of the target.

### The rule families

Each family's parameters are chosen by grid search for the lowest `m` on the training capture. "fb" is the size used when a first-passing rule finds no rung.

| Family | Fitted on seoul_bull: parameters | in-sample | cross, on kerry_park | Fitted on kerry_park: parameters | in-sample | cross, on seoul_bull |
|---|---|---|---|---|---|---|
| (a) Constant px | 11.5 px | 37% / 61% / 0.280 | **42% / 66% / 0.276** | 12.0 px | 42% / 66% / 0.263 | **37% / 59% / 0.314** |
| (b) First footprint where a level passes the radius gate and the coherence bar | levels ≥ 2 ref px per sample; whole, middle and 5 cells ≤ 2.0 samples; no coherence bar; fb 10 px | 41% / 63% / 0.305 | 36% / 59% / 0.333 | levels of 12 to 40 samples; radius ≤ 0.1 of `h`; coherence ≥ 0.85; fb 10 px (47% to 72% of points fall back) | 39% / 64% / 0.300 | 35% / 56% / 0.364 |
| (b′) as (b), fallback on ≤ 10% of points | as (b) | 41% / 63% / 0.305 | 36% / 59% / 0.333 | levels ≥ 2 ref px per sample; radius ≤ 0.3 of `h`; 5 cells; coherence ≥ 0.6 | 40% / 61% / 0.320 | 41% / 62% / 0.319 |
| (c) First footprint where coherence stops rising by more than `d` | coarsest level; `d` = 0.05; from 8 px | 39% / 67% / 0.281 | 35% / 62% / 0.320 | coarsest level; `d` = 0; from 8 px | 45% / 68% / 0.260 | 36% / 62% / 0.320 |
| (d) Footprint maximising coherence × localisability | best-level coherence × `exp(−0.4·radius)` | 32% / 54% / 0.373 | 27% / 50% / 0.403 | the same with 0.1 | 28% / 51% / 0.394 | 26% / 46% / 0.432 |
| (e) First footprint where tile contrast or gradient energy exceeds T | native-level grey std ≥ 20.4 | 26% / 44% / 0.465 | 11% / 28% / 0.735 | the same, ≥ 22.3 | 15% / 30% / 0.709 | 20% / 40% / 0.486 |
| (f) k × the smallest footprint where a cell count passes | native level; 1 cell ≤ 2.0 samples; k 2.97 | 43% / 62% / 0.276 | 35% / 60% / 0.312 | native level; 1 cell ≤ 1.5; k 2.97 | 39% / 64% / 0.276 | 41% / 61% / 0.291 |
| (g) Geometry: depth | `h ∝ depth^−0.09` | 41% / 65% / 0.294 | 41% / 69% / 0.281 | `h ∝ depth^−0.27` | 49% / 76% / 0.228 | 41% / 64% / 0.282 |
| (g) depth, zoom range, angle, views, reference zoom | log-linear | 48% / 66% / 0.251 | 17% / 31% / 0.657 | log-linear | 47% / 77% / 0.235 | 36% / 60% / 0.310 |
| (h) Tree, depth 2, on 52 per-point readings | nearest GT point distance, then photograph variance in 32 px | 51% / 79% / 0.207 | 32% / 62% / 0.331 | native gradient energy at `h = 5`, then depth | 55% / 81% / 0.199 | 36% / 58% / 0.372 |
| (h) Ridge on all 52 readings | | 69% / 91% / 0.157 | 14% / 28% / 0.610 | | 64% / 91% / 0.181 | 36% / 68% / 0.296 |
| (i) Log-linear in the top 3 readings | grey variance in 32 px, footprint of greatest native and of greatest any-level coherence | 48% / 76% / 0.233 | 40% / 67% / 0.271 | the same | 43% / 77% / 0.260 | 47% / 71% / 0.245 |
| (i′) Rounded: `h = K · s₃₂^(−1/2) · h_coh^(1/4)` | `K` = 32.7 | 50% / 72% / 0.223 | **41% / 74% / 0.273** | `K` = 35.3 | 42% / 75% / 0.259 | **47% / 73% / 0.249** |
| Noise floor (above) | | | 69% / 94% / 0.15 | | | 51% / 80% / 0.22 |

Fitting on both captures with five-fold cross-validation, which mixes the captures, the depth-3 tree scores 48% / 75% / 0.233 and the ridge 56% / 84% / 0.201. That is as far as these readings can go when the training data includes points from the same capture. Across captures, the learned models do worse than the constant, and they do not choose the same features on the two captures.

**Which scale band.** For family (b), restricted to rules that pick a rung for at least 90% of points and scored with one parameter set on both captures:

| Levels that may pass | seoul_bull | kerry_park |
|---|---|---|
| Native (level 0) only | 34% / 51% / 0.377 | 27% / 46% / 0.441 |
| Coarsest only | 32% / 51% / 0.394 | 25% / 46% / 0.467 |
| Any level | 31% / 51% / 0.392 | 28% / 50% / 0.405 |
| Levels of ≥ 2 ref px per sample | 41% / 63% / 0.313 | 38% / 59% / 0.322 |
| Levels of 20 to 80 samples | 40% / 65% / 0.326 | 37% / 62% / 0.326 |

Letting any band pass makes the rule pick small footprints, because a 3 to 6 px footprint at native density already passes on most textured points. The band rules come nearest the human when the fine bands are excluded, and then they only match the constant. Native-only and coarsest-only do no better than any band. The coherence bar does not help: the best rule in (b) without it scores the same, and coherence alone scores 40% / 65% / 0.331 and 37% / 62% / 0.326.

**Univariate correlations with log human size.** The readings with the same sign and the largest magnitude on both captures are contrast near the point, and the footprint of greatest coherence. Contrast near the point is the grey std of the native tile at `h = 20` (−0.55 / −0.49) or the variance of the photograph in a 32 px disk (−0.56 / −0.49). The footprint of greatest coherence is +0.39 / +0.43 at the native level and +0.50 / +0.36 at any level. The distance to the nearest other ground-truth point is +0.54 / +0.31, and depth is weaker. The first footprint to pass any radius gate, the detector `σ`, the distance to an edge and the viewing angle all correlate weakly.

### The best rule

**Rule (i′), in words:** make the half-extent about 34 px divided by the square root of the photograph's grey standard deviation (0 to 255) within 32 px of the point, and multiplied by the fourth root of the footprint, in px, at which the views agree best at native density. That gives about 10 px on an average surface, smaller on high-contrast texture and larger where the views keep agreeing better as the patch grows. Cross-capture it scores 41% / 74% / 0.273 on kerry_park and 47% / 73% / 0.249 on seoul_bull, against the constant's 42% / 66% / 0.276 and 37% / 59% / 0.314. A paired bootstrap over points (2000 resamples) gives the gain in `m` as 0.066 (95% interval 0.009 to 0.113) on seoul_bull and 0.002 (−0.044 to 0.046) on kerry_park. The gain in the share within ×1.5 is 14 points (7 to 21) and 8 points (1 to 14).

**What it fails on.** Its predictions span less than the human's: the slope of log prediction on log human size is 0.37 and 0.46, and the correlation is 0.61 and 0.62. By the human's size band, with signed `log(rule / human)` medians on seoul_bull / kerry_park:

| Human half-extent | n | Median signed log ratio | Rule too large by > 1.5× | Rule too small by > 1.5× |
|---|---|---|---|---|
| < 6 px | 31 / 62 | +0.53 / +0.35 | 71% / 42% | 0% / 0% |
| 6–9 px | 71 / 71 | +0.24 / −0.00 | 21% / 7% | 0% / 1% |
| 9–13 px | 88 / 126 | +0.06 / −0.14 | 7% / 8% | 1% / 12% |
| 13–18 px | 46 / 103 | −0.29 / −0.28 | 0% / 1% | 33% / 29% |
| ≥ 18 px | 27 / 18 | −0.40 / −0.45 | 0% / 0% | 41% / 61% |

By texture class, it is too small on scale-free points (median −0.11 / −0.13) and too large on flat-then-structured points (+0.22 / +0.10). On the flat-then-structured points the constant is worse (+0.47 / +0.37), and that class is where most of the gain comes from. The contact sheet of the 24 largest residuals (`size-rule/worst_residuals.png` in the session scratch directory, not checked in) shows the photograph around each point with the human's square in green and the rule's in red, and the native tile at each size. It shows two kinds of failure:

- **Too small.** These are close-range ground: lawn, gravel and pale paving at 2 to 4 m on kerry_park, and the grey rock behind the sculpture. The human chose 15 to 30 px there. The texture is fine and of moderate contrast, and the views agree at every footprint, so no reading says to grow. The recycling bin and the rainbow sign are also here.
- **Too large.** These are small features the human sized tightly: one stripe crossing on the bull's painted coils at 11 m, a distant building or treeline at 30 to 70 m, and small dark blobs in grass. The human chose 2 to 6 px there. The rule cannot go that small, because the surrounding 32 px has middling contrast.

**Decision.** No rule beats the constant-pixel baseline by more than the noise floor. On kerry_park, no family beats the constant's `m` across captures by more than the bootstrap spread. On seoul_bull, the best rule closes about two thirds of the gap between the constant (0.314) and the floor (0.22). That gain of 0.066 is a third of the floor, and its interval reaches down to 0.01. The rules built from the draft's own readings and the two hypotheses do no better than the constant: first passing the radius and coherence gates over the footprint × band grid, the coherence plateau, coherence × localisability, and the cell-count extent. Those built on any band, or on the native band alone, do worse. What the human's sizes follow is mostly not in these readings. A point's size is nearly the same as its same-surface neighbour's, so the human sized by surface, not point by point. The one consistent per-point trend is that the human made patches smaller where the local contrast is high and larger where it is low, which is the opposite of what a localisability gate does. A footprint rule for the pipeline cannot be validated against these sizes beyond "about 10 px, a little smaller on high-contrast texture". The question is better asked of the outcome the size is for, such as the normal error or the keypoint residual on a fixed observation set, as the draft's Testing section already says.

## The nine cell radii against the hand-set sizes (2026-10-09)

**Question.** The bench shows a 3×3 grid of ZNCC self-similarity radii for each patch, one per cell of the `R/3` split. Across the footprint ladder, do those nine values predict the hand-chosen half-extent? The families in [A rule for the hand-set sizes](#a-rule-for-the-hand-set-sizes-2026-10-09) used only the count of cells that pass a gate. This section tests the grid's structure: how the nine radii are spread, which cells are capped, whether the centre differs from the ring, and whether the radii vary along one axis.

**Data.**

- **Code and machine.** Commit `c7859df1` (branch `bootstrap-core-migration`). The extension is the one built at `3d3f6ab6`; the commits since change only this draft. Windows 11, Intel Core i9-14900HX, 63.7 GB RAM.
- **Points and ladder.** The same 263 seoul_bull and 380 kerry_park points, reference views, human half-extents and footprint ladder (3 to 40 reference px) as the previous section. The cell radii at native density and on its box pyramid come from that section's `feature_table.npz`. One more reading was added: the reference view rendered at `R = 24` at every rung, the bench's grid.
- **Level views.** Each rung is read on four grids: native density (`L0`); the coarsest pyramid level; the finest level with at least 2 reference px per sample (`ge2`, which exists from `h = 10`); and `R = 24`.
- **Grid statistics, per point, rung and level.** The minimum, maximum, median, mean and standard deviation of the nine radii, in grid samples and in reference px. Also: max/min; the number of cells at or under 0.75, 1, 1.5, 2 and 2.5 samples; the number at the cap of 3; the centre cell minus, and over, the mean of the eight ring cells; the share of the radii's variance carried by the row means or by the column means, whichever is larger (near 1 when the radii vary along one axis only, as on an edge); the number of cells whose radius in px is under half the footprint; and the number of cells that localise within their own cell, meaning a radius under 0.15, 0.25, 0.33 or 0.5 of the cell side and not capped.
- **Scripts.** `size-rule/grid/extra.py` (the `R = 24` and human-multiple readings), `grid_rules.py`, `g_models.py`, `diag.py`, `check.py` and `sheet.py` in the session scratch directory, run with `pixi run -e dev python`. Nothing here is checked in.
- **Scoring.** As before: `w1.25 / w1.5 / m`, each family's parameters chosen by grid search on one capture and scored on the other. Every first-passing rule also has a fallback of 10, 12 or 40 px. Every rung-picking rule may also be multiplied by a factor `k`, fitted as the median ratio on the training capture.

**Result.**

| Family | Fitted on seoul_bull, scored on kerry_park | Fitted on kerry_park, scored on seoul_bull |
|---|---|---|
| Constant px | 42% / 66% / 0.276 | 37% / 59% / 0.314 |
| (a) First footprint where all nine cells are ≤ g | `L0`, g 2.5, × 1.18: 32% / 57% / 0.350 | `ge2`, g 2.5, × 0.62: 36% / 59% / 0.332 |
| (b) First footprint where ≥ k cells localise in their own cell | `ge2`, ≤ 0.5 of a cell, k 5, × 0.92: 40% / 64% / 0.286 | `ge2`, ≤ 0.25 of a cell, k 5, × 0.73: 31% / 57% / 0.352 |
| (c) First footprint where the radii's std or max/min is under a threshold | `ge2`, max/min ≤ 4, × 0.64, 47% fall back: 29% / 53% / 0.379 | `L0`, max/min ≤ 1.5; 98% fall back to 12 px, so it is the constant: 37% / 60% / 0.312 |
| (c) Footprint where the std or max/min is least | `R = 24`, std in px, from 8 px, × 0.83: 29% / 54% / 0.383 | `R = 24`, std in px, from 12 px: 27% / 49% / 0.411 |
| (d) First footprint where the centre radius ≥ t × ring mean | `ge2`, t 1.0, × 0.68, 59% fall back: 28% / 48% / 0.435 | `L0`, t 0.8, × 3.48; picks one size for 99%: 40% / 63% / 0.321 |
| (d) Footprint where \|centre − ring mean\| is least | coarsest level, from 12 px, × 0.49: 26% / 47% / 0.439 | `ge2`, from 12 px, × 0.40: 29% / 50% / 0.404 |
| (e) First footprint with at most j capped cells | `ge2`, j 2, × 0.94: 36% / 60% / 0.321 | `ge2`, j 1, × 0.73: 37% / 64% / 0.297 |
| (f) Footprint maximising localising cells × `h^−p` | `ge2`, ≤ 0.33 of a cell, p 1, × 0.70: 41% / 67% / 0.279 | `ge2`, ≤ 0.5 of a cell, p 0.5, × 0.61: 34% / 62% / 0.334 |
| (g) Log-linear, greedy top 3 grid features at fixed rungs | cells ≤ 1 sample at 25 px (`L0`), min radius at 25 px (`ge2`), one-axis share at 17 px (`R = 24`): 39% / 62% / 0.317 | localising cells at 12 px (`ge2`), std at 14 px (`L0`), one-axis share at 3 px (`L0`): 40% / 70% / 0.264 |
| (g) Self-consistent: model of the grid at the human's own rung, applied at the rung where it predicts that rung | 32% / 49% / 0.414 | 21% / 48% / 0.433 |
| (g) Rung classifier, grid features of the rung and its two neighbours | 38% / 62% / 0.319 | 43% / 63% / 0.286 |
| *Whole-tile radius only:* first footprint where the whole radius ≤ g | `R = 24`, g 2.5, × 2.86: 29% / 52% / 0.388 | `ge2`, g 2.5: 40% / 64% / 0.328; one size for 100% |
| *Whole only:* log-linear, top 3 | 35% / 58% / 0.347 | 43% / 65% / 0.298 |
| *Whole only:* rung classifier | 34% / 58% / 0.347 | 42% / 63% / 0.305 |
| *Rung index alone:* rung classifier | 26% / 49% / 0.412 | 40% / 65% / 0.331 |
| Rule (i′) of the previous section | 41% / 74% / 0.273 | 47% / 73% / 0.249 |
| (i′) with the best one grid feature added | min radius in px at 40 px (`R = 24`): 39% / 66% / 0.290 | one-axis share at 14 px (`ge2`): 44% / 71% / 0.252 |
| Noise floor | 69% / 94% / 0.15 | 51% / 80% / 0.22 |

Most of the rung-picking rules end close to a constant. The fitted rule gives one size to between 39% and 66% of the test capture's points in (a), (b), (e) and (f), and to 99% in the two entries marked so. The best grid-only model in each direction is the log-linear (g). Fitted on kerry_park and scored on seoul_bull, it improves on the constant's `m` by 0.050 (95% bootstrap interval −0.012 to 0.094), and the share within ×1.5 by 11 points (4 to 17). Fitted on seoul_bull and scored on kerry_park, it is worse than the constant by 0.041 (interval −0.073 to −0.002). The two fits choose different features. Added to rule (i′), the best grid feature makes the cross-capture `m` worse, by 0.016 (−0.045 to 0.008) on kerry_park and 0.002 (−0.032 to 0.018) on seoul_bull.

Against the whole-tile radius, the grid features do better in every comparable family. Cross-capture `m` is 0.317 and 0.264 for the grid log-linear model against 0.347 and 0.298 for the whole-only one, and 0.319 and 0.286 for the grid classifier against 0.347 and 0.305. Neither set beats the constant in both directions.

The self-consistent model in (g) shows why the grid at the human's own size looks predictive and is not. Fitted on the grid statistics at the rung nearest the human's size, it scores 84% / 98% / 0.107 (seoul_bull) and 80% / 94% / 0.108 (kerry_park) in-sample. Those statistics include radii in reference px, whose scale is the rung's, so the model reads the footprint back from them. Applied as a rule, choosing the rung at which the model predicts that rung, it scores 0.41 and 0.43.

**The grid at half, one and two times the human's size.** Each row is the reference view at native density, with the counts at `R = 24` in brackets. The gate is 2.5 samples, the bench's bar. "Capped" means a cell at 3.

| | 0.5× | 0.75× | 1× | 1.5× | 2× |
|---|---|---|---|---|---|
| seoul_bull: all nine cells pass | 18% (36%) | 45% (62%) | 67% (78%) | 77% (78%) | 76% (74%) |
| seoul_bull: no capped cell | 22% (45%) | 51% (68%) | 71% (85%) | 83% (86%) | 81% (80%) |
| seoul_bull: median std of the nine radii, samples | 0.86 (0.50) | 0.64 (0.37) | 0.25 (0.27) | 0.26 (0.22) | 0.28 (0.26) |
| seoul_bull: median cells passing | 6 (7) | 8 (9) | 9 (9) | 9 (9) | 9 (9) |
| kerry_park: all nine cells pass | 14% (45%) | 44% (63%) | 60% (71%) | 69% (68%) | 70% (56%) |
| kerry_park: no capped cell | 18% (54%) | 47% (69%) | 65% (77%) | 74% (74%) | 77% (63%) |
| kerry_park: median std of the nine radii, samples | 0.87 (0.46) | 0.63 (0.37) | 0.37 (0.29) | 0.28 (0.35) | 0.33 (0.60) |
| kerry_park: median cells passing | 5 (8) | 8 (9) | 9 (9) | 9 (9) | 9 (9) |

The whole-tile radius passes on 98% to 100% of points at every multiple at native density, so it says nothing here. Over the population, the grid does have a recognisable shape at the human's size. At native density, the share of points with all nine cells passing climbs steeply up to 1× and levels off by 1.5×. The median spread of the nine radii falls by a factor of about 3 from 0.5× to 1× and is flat after that. So the human's size lies where the grid has just become uniform: at 1×, all nine cells pass on 67% (seoul_bull) and 60% (kerry_park) of points, against 18% and 14% at 0.5×. At `R = 24` the same shape comes at a smaller multiple, since `R = 24` oversamples below the human's ~20 px edge.

Point by point, the signature does not locate the human's size. Over the five multiples at native density, the first multiple at which all nine cells pass is 0.5× for 18% / 14% of points, 0.75× for 27% / 31%, 1× for 23% / 17%, 1.5× for 16% / 14%, and 2× for 5% / 9%; 11% / 15% never pass. The first multiple with no capped cell is spread the same way. The multiple at which the spread of the radii is least falls on 1×, 1.5× or 2× for 73% / 69% of points, against 60% if it were uniform over the five, and on 1× itself for 24% / 26%. The multiple at which the most cells localise per unit area is 0.5× for 44% / 47%. Read together: all nine cells passing at the human's size is common (about two thirds of points), so the human seldom chose a size whose grid still has a cell that fails to localise. But the human's size is not the first size at which that happens: it is reached at or below 0.75× on 45% of points, and the human went on to a larger size.

The contact sheet (`size-rule/grid/grid_contact_sheet.png` in the session scratch directory, not checked in) shows 12 points, two from each of three bands of human size on each capture. For each it shows the reference tile at 0.5×, 1× and 2× the human's size, at native density and at `R = 24`, with each cell's radius printed in it: green at or under 2.5, orange over, red at the cap. On textured ground (seoul_bull 276, kerry_park 274 and 349) all nine cells pass at every size, and nothing in the grid separates the human's size from half or double it. On dark or flat surfaces (seoul_bull 36 and 131, kerry_park 256 and 311) the capped cells at 0.5× clear by 1× or 2×, and the human's size is near where they clear. On the dark tree against sky (kerry_park 41, human half-extent 2.3 px) capped cells remain at every size.

**Decision.** The grid's structure does not predict the human's size beyond what a constant does. The rules built on it (all nine cells passing, k cells localising in their own cell, the radii becoming uniform, the centre matching the ring, the capped cells clearing, and the localising cells per area) score between 0.279 and 0.439 cross-capture, against the constant's 0.276 and 0.314. The best grid-only model beats the constant in one direction (by 0.050, interval −0.012 to 0.094) and loses in the other (by 0.041). Added to rule (i′), the grid makes it slightly worse in both directions. The grid does carry more than the whole-tile radius: in the same families it lowers the cross-capture `m` by about 0.02 to 0.03, which is within the bootstrap spread, and the whole radius alone does no better than the constant. The one finding is a population-level signature: the human's size sits where the grid has just become uniform (all nine pass on about two thirds of points, against one sixth at half the size). Point by point it holds as a lower bound, not as a choice. On the native-density ladder the first all-pass rung exists for 98% and 93% of points, and the human's size is at or above it on 70% and 61% of those, by a median factor of 1.37 and 1.51 when it is above. As a rule, "the smallest footprint whose nine cells all pass" therefore gives a floor on the footprint, not the hand-set size.
