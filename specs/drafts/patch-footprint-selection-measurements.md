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
