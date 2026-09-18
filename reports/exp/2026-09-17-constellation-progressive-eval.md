# Is a progressive patch constellation query worth building, 2026-09-17

Measurement of a *progressive* form of the patch constellation query
(`crates/sfmtool-core/src/features/kdforest/constellation.rs`, spec
`specs/core/features/kdf-constellation-query.md`) against the tracks of a
reconstruction of the same capture, on five datasets. The shipped query takes a
constellation of one fixed size and answers once; the proposal is to start from
the few features nearest the patch centre, answer, and widen only if the answer
is thin, reusing each feature's forest lookups across stages. The questions were
the start size `S`, the increment `D`, whether a small constellation ever gives a
*better* answer than the wide one, which stopping or combination rule to use, and
what acceptance bar each stage needs.

**Answer, in one line.** A progressive query is not worth building to *find
images* -- across five captures, 60 patches each, not one image that the
nearest-50 constellation misses is found by any smaller prefix, so every
early-stopping rule is a pure recall-for-latency trade that a smaller single
query already offers on better terms. It is worth building to *place* the patch,
on wide-baseline captures: the affine fitted at the stage that first accepted an
image puts the ground truth's own correspondences near the patch centre within
3 px in **0.91** of cases against 0.70 for the stage-50 affine on dino_dog_toy,
0.88 against 0.75 on kerry_park and 0.90 against 0.85 on seoul_bull over a
five-stage run, and about two thirds of that is available from a **two-stage**
run at 25 then 50 that keeps the early warp and the late inlier set (0.86, 0.92
and 0.83 respectively). The extra cost of such a run is exactly one additional
RANSAC pass, +11% to +20%, because the forest search is linear in batch size
with a negligible fixed cost (intercept −0.07 to +0.01 ms against 0.06 to
0.15 ms per feature on the four small captures) and splitting one search into
stages is therefore free. On DinoLedge, a 1,196-image video walk where
consecutive frames differ by a few percent of scale, there is no centre-warp
gain to have (20 wins / 13 losses / 7 ties at S=10, and the lock rule loses
0.86 → 0.83), so the recommendation is capture-shaped rather than universal.

The acceptance bar is a second, separable finding. The inlier distribution of a
candidate that shares no point at all with the query image barely moves with the
constellation size -- its 90th percentile is 4 to 6 inliers at n=10 and 5 to 9 at
n=50 across the five captures -- while the inlier count of a *true* candidate
roughly doubles over the same range (median 5 → 13 on DinoLedge, 5 → 8 to 5 → 12
elsewhere). So `min_inliers = 8` is not mis-set at small `n` because the noise
floor moved; it is mis-set because at n=10 a true candidate usually has five or
six inliers and eight throws almost all of them away. Lowering the bar to 5 or 6
below about n=20 recovers most of that recall at a precision *better* than the
shipped default achieves at n=50 (DinoLedge n=10 at a floor of 5: precision 0.90,
0.28 false candidates per patch, against 0.76 and 2.07 at n=50 with a floor of 8).

**An unplanned third finding, and the largest number in the report.** The
256 MiB cache budget this evaluation inherited from the 2026-09-14 one is below
the knee for DinoLedge's 1,494 MB index. Raising it to 512 MiB takes a repeated
50-feature query from **490 ms to 14 ms** and a first-touch one from 500 ms to
262 ms, and turns a bare forest search that was superlinear in batch size
(2.7 / 39 / 382 ms at 10 / 25 / 50 features) into a linear one
(2.5 / 5.7 / 11.4 ms). Almost everything the constellation query has ever been
reported to cost on a large capture is cache thrash. §6 has the sweep.

Tooling added:
[`scripts/kdf_constellation_progressive_eval.py`](../../scripts/kdf_constellation_progressive_eval.py),
which runs every prefix query once, stores the full per-candidate result, and
replays schedules, acceptance bars and stopping rules offline. It imports the
corpus, ground-truth and patch-sampling pieces of
[`scripts/kdf_constellation_eval.py`](../../scripts/kdf_constellation_eval.py),
which is unchanged. Nothing in the implementation was changed; no Rust was
touched and nothing was committed.

---

## 1. Protocol

### Why one measurement covers every schedule

A constellation feature's forest hits do not depend on which other features are
in the constellation; `one_hit_per_image` collapses a (feature, candidate image)
cell on its own; and RANSAC seeds per candidate image from `seed + image_index`
(`constellation.rs`, `StdRng::seed_from_u64(params.seed.wrapping_add(image))`).
So the stage of a progressive run that holds the nearest `n` features returns
**exactly** what a plain `constellation_query` on those `n` features returns, and
a progressive run is a subsequence of the prefix queries. Every schedule `(S, D)`
and every stopping rule is therefore recoverable offline from one table of prefix
results, exactly rather than approximately.

`min_inliers` is applied after the per-image fit -- `constellation.rs` runs
`fit_affine_ransac` and only then tests `inliers.len() < params.min_inliers` --
so it is a reporting filter and the bar can be swept offline too. Verified
empirically as well as by reading: over 20 patches × three prefixes on
seattle_backyard, the result of a query at `min_inliers = 4` filtered to
`inliers >= 8` was identical to a query at `min_inliers = 8` in **60 of 60**
cases, matching image order, inlier counts and the affines bit for bit. All
queries in this report therefore ran at `min_inliers = 4`, and every bar in every
table is applied afterwards. The bar sweep is truncated below 4; the shape of the
distribution under 4 inliers is not measured here.

### Corpora

One `.kdf` per dataset, built from that workspace's `.sift` files with SIFT
sources, four trees, leaf size 16, 2 KiB descriptor blocks, 1 MiB chunks, seed 0
-- the shape [`scripts/kdf_patch_localize.py`](../../scripts/kdf_patch_localize.py)
builds and the one the 2026-09-14 constellation evaluation used (retired from the
tree; `git show 17e532ff^:reports/exp/2026-09-14-constellation-query-eval.md`
and its dedupe-sweep sibling in the same directory). All queries go through
`LazyKdForest`, reading the file. Cache budget 256 MiB for DinoLedge and 64 MiB
for the four small captures -- which §6 shows to be the wrong choice for
DinoLedge, though it affects only the wall-clock columns.

| dataset | images | image px | descriptors | GT observations | `.kdf` |
|---|--:|---|--:|--:|--:|
| DinoLedge | 1,196 | 2160×3840 | 9,702,948 | 4,523,010 | 1,494 MB |
| dino_dog_toy | 85 | 2040×1536 | 336,478 | 36,587 | 52 MB |
| kerry_park | 48 | 480×480 | 80,964 | 4,836 | 12 MB |
| seattle_backyard | 26 | 360×640 | 79,867 | 2,672 | 12 MB |
| seoul_bull | 17 | 270×480 | 37,818 | 4,010 | 6 MB |

**DinoLedge is the same capture, the same features and the same ground truth as
the 2026-09-14 report**: `C:\DataSets\DinoLedge`, its `sfmtool`-SIFT features,
and the checked-in `20260707-00-solve-DinoLedge_1-1196.sfmr` (not the
`-embedded` sibling), which carries 4,523,010 observations exactly as that report
recorded, over a 1,494 MB index of exactly the size it recorded.

**The four small captures are not.** Their workspaces were still in the tree from
an earlier round (`seoul_bull_ws`, `dino_dog_toy_ws`, `seattle_backyard_ws`,
`kerry_park_ws`, dated May), so per the brief they were used as they stood rather
than re-bootstrapped. They carry **COLMAP** SIFT rather than `sfmtool` SIFT and
sparser solves than the ones the 2026-09-14 report measured: 36,587 observations
against 85,895 on dino_dog_toy, 2,672 against 14,388 on seattle_backyard, 4,836
against 2,957 on kerry_park, 4,010 against 3,805 on seoul_bull. Because every
"false candidate" is an image the solve does not corroborate, a sparser solve
depresses the precision column and inflates the false-candidate column on those
four; the never-covisible column, which sparsity cannot manufacture, is the one
to read there. Nothing in the comparisons below is affected, since every arm sees
the same ground truth.

### Patches

60 per dataset, seeded (numpy PCG64, seed 0): a random registered image, then a
random keypoint of that image as the centre, drawn from the same generator in the
same order as `scripts/kdf_constellation_eval.py`, so the centres are that
harness's. The constellation at stage `n` is the **nearest `n` keypoints**, taken
from one stable distance sort, so the stages nest exactly; the older harness grew
a radius instead, which differs only where keypoints tie at the radius. Prefixes
measured: 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50.

A second set of 60 patches per dataset with centres at **uniformly random pixels**
rather than on keypoints was measured and is reported separately in §7.

Parameters otherwise at their defaults: `k = 32`, `max_leaf_checks = 512`,
`threshold_px = 8`, `iterations = 200`, `min_correspondences = 3`,
`one_hit_per_image = true`, `same_image_ratio = 1.0`, `max_scale = 4`, `seed = 0`.

### What is scored

Per candidate image at every stage: inliers, correspondences offered, the warp's
geometric-mean scale, whether the image is never covisible with the query image,
and the ground truth's own residuals under that candidate's affine on **three**
reference sets --

- `res_n`, the correspondences among the stage's own `n` features. This is
  `kdf_constellation_eval.py`'s measure and reproduces its `warp ok`.
- `res_50`, the correspondences of the nearest-50 disc. A fixed set, so a warp
  fitted at n=10 and one fitted at n=50 are judged on the same points; this is
  the 10-feature affine *extrapolated* over the whole disc.
- `res_10`, the correspondences among the nearest ten only. The patch centre's
  own neighbourhood, and the thing a caller that warps the centre pixel -- the
  bench's descriptor search, for one -- actually depends on.

`warp ok`, `warp ok50` and `warp ok10` are the share of accepted candidates
carrying at least three correspondences in that set whose median residual is
within 3 px. Image recall and precision are against the images the ground truth
gives at least three correspondences with; unless a table says otherwise the
denominator is the **nearest-50** constellation's, so that stages are comparable.

### What counts as noise

60 patches. Percentile bootstrap 95% intervals on `recall≥3` at n=50, floor 8:
DinoLedge 0.67–0.81, dino_dog_toy 0.59–0.75, seattle_backyard 0.75–0.97,
kerry_park 0.65–0.89, seoul_bull 0.60–0.77. So a recall difference under about
0.07 on one capture carries no evidence by itself.

The `warp ok10` denominators are the weak point. A candidate only enters that
column when the ground truth gives it three correspondences among the patch's
*ten* nearest features, and at n=50 with a floor of 8 that is 208 candidates on
DinoLedge, 43 on dino_dog_toy, 41 on seoul_bull, 24 on kerry_park and **2** on
seattle_backyard (against 584 / 257 / 88 / 72 / 75 for the `warp ok50` column).
seattle_backyard's centre column is therefore uninformative throughout and is not
read; kerry_park's and seoul_bull's carry a standard error around 0.06 to 0.08.

---

## 2. Sanity check: the single-query numbers reproduce

DinoLedge at n = 50 with the shipped defaults, against the post-#458 baseline in
§2 of the dedupe-sweep report retired from the tree
(`git show 17e532ff^:reports/exp/2026-09-14-constellation-dedupe-sweep.md`):

| source | recall≥3 | prec≥3 | corr recall | warp ok | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|
| 2026-09-14 dedupe sweep, DinoLedge | 0.74 | 0.76 | 0.60 | 0.76 | 2.08 | 0.20 | 562 |
| this report, DinoLedge | 0.74 | 0.76 | 0.60 | 0.76 | 2.07 | 0.18 | 498 |

Every column agrees to two decimals. The four small captures are within the
expected shift for a different feature extractor and a sparser solve; their
recall is comparable (dino_dog_toy 0.67 against 0.69, seattle_backyard 0.87
against 0.69, kerry_park 0.77 against 0.82, seoul_bull 0.69 against 0.52) and
their precision is lower, as §1 says it must be.

Both `ms` figures are at the 256 MiB cache budget, and §6 shows both of them to
be measurements of that budget rather than of the query: the same query at
512 MiB is 14 ms. The residual medians are not comparable between the two
reports and are left out of the table above: this harness takes the median over
candidates of each candidate's own median residual, where the older one pooled a
patch's residuals across candidates first.

### The full prefix sweep, shipped floor of 8, stage-local denominator

**DinoLedge**

| n | found | recall≥3 | prec≥3 | corr recall | res med | warp ok | warp ok50 | warp ok10 | false | never-covis |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | 0.0 | 0.00 | n/a | 0.00 | n/a | n/a | n/a | n/a | 0.00 | 0.00 |
| 8 | 0.1 | 0.02 | 0.80 | 0.02 | 0.72 | 0.80 | 0.56 | 0.80 | 0.00 | 0.00 |
| 10 | 0.9 | 0.14 | 0.83 | 0.10 | 0.74 | 0.97 | 0.73 | 0.97 | 0.00 | 0.00 |
| 12 | 2.0 | 0.28 | 0.84 | 0.21 | 0.60 | 0.94 | 0.61 | 0.93 | 0.03 | 0.00 |
| 15 | 3.8 | 0.45 | 0.79 | 0.32 | 1.08 | 0.94 | 0.61 | 0.94 | 0.13 | 0.00 |
| 20 | 6.2 | 0.53 | 0.75 | 0.41 | 1.26 | 0.88 | 0.65 | 0.86 | 0.37 | 0.03 |
| 25 | 8.2 | 0.60 | 0.75 | 0.47 | 1.57 | 0.88 | 0.70 | 0.85 | 0.57 | 0.05 |
| 30 | 9.5 | 0.61 | 0.77 | 0.52 | 1.60 | 0.85 | 0.73 | 0.88 | 0.73 | 0.05 |
| 35 | 10.8 | 0.65 | 0.76 | 0.53 | 1.87 | 0.79 | 0.73 | 0.84 | 1.10 | 0.10 |
| 40 | 12.1 | 0.68 | 0.78 | 0.56 | 1.92 | 0.79 | 0.73 | 0.86 | 1.37 | 0.12 |
| 45 | 13.5 | 0.73 | 0.74 | 0.60 | 1.97 | 0.77 | 0.75 | 0.83 | 1.65 | 0.17 |
| 50 | 14.7 | 0.74 | 0.76 | 0.60 | 2.09 | 0.76 | 0.76 | 0.86 | 2.07 | 0.18 |

The `warp ok` / `warp ok50` pair is the whole story of the progressive idea in
one place. At n=10 the stage's own warp is right 0.97 of the time *on its own ten
points* and only 0.73 of the time when the same warp is extrapolated over the
50-feature disc. At n=50 the two are the same measurement by construction and sit
at 0.76. A small constellation gives a locally excellent, globally poor affine;
a large one gives a uniformly mediocre one.

**dino_dog_toy** (the capture where the effect is largest)

| n | found | recall≥3 | prec≥3 | corr recall | warp ok | warp ok50 | warp ok10 | false | never-covis |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 0.6 | 0.19 | 0.17 | 0.06 | 0.91 | 0.42 | 0.91 | 0.08 | 0.00 |
| 15 | 1.9 | 0.41 | 0.12 | 0.13 | 0.81 | 0.43 | 0.86 | 0.30 | 0.00 |
| 20 | 3.2 | 0.52 | 0.13 | 0.22 | 0.85 | 0.52 | 0.90 | 0.57 | 0.00 |
| 25 | 4.3 | 0.61 | 0.18 | 0.29 | 0.70 | 0.49 | 0.85 | 0.88 | 0.00 |
| 50 | 8.6 | 0.67 | 0.35 | 0.40 | 0.65 | 0.65 | 0.70 | 2.03 | 0.08 |

---

## 3. The acceptance bar as a function of `n`

### The wrong-candidate inlier distribution hardly moves with `n`

Every candidate the query reported at `min_inliers = 4`, split by whether the
ground truth corroborates it. `false` shares no correspondence with the
nearest-50 constellation; `never-covis` shares no point with the query image
anywhere in it, which a sparse solve cannot manufacture. Medians sit on the
floor of 4 by construction, so the 90th percentile and the maximum are what
matter.

**DinoLedge**

| n | false: count | p90 | max | never-covis: count | p90 | max | true: count | median inliers |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 30 | 6.1 | 7 | 5 | 5.0 | 5 | 391 | 5 |
| 15 | 71 | 8.0 | 10 | 8 | 6.3 | 7 | 574 | 7 |
| 20 | 112 | 9.0 | 14 | 11 | 8.0 | 8 | 687 | 8 |
| 25 | 166 | 10.0 | 17 | 17 | 8.0 | 8 | 769 | 9 |
| 50 | 414 | 13.0 | 34 | 58 | 9.0 | 11 | 1,010 | 13 |

**The four small captures, never-covisible p90 / max**

| n | dino_dog_toy | seattle_backyard | kerry_park | seoul_bull |
|--:|---|---|---|---|
| 10 | – (0 cases) | 4.4 / 5 | 6 / 7 | 4.0 / 5 |
| 15 | 5.0 / 5 | 4.3 / 6 | 7 / 10 | 5.0 / 6 |
| 25 | 5.0 / 6 | 4.0 / 7 | 8 / 17 | 5.0 / 7 |
| 50 | 6.0 / 11 | 5.0 / 12 | 7 / 32 | 5.0 / 8 |

From n=10 to n=50 the never-covisible 90th percentile moves 5.0 → 9.0 on
DinoLedge, 4.4 → 5.0 on seattle_backyard, 6 → 7 on kerry_park and 4.0 → 5.0 on
seoul_bull -- one inlier or less on three of the five captures, four on the
largest. Over the same range the median inlier count of a *true* candidate goes
5 → 13 (DinoLedge), 5 → 9 (dino_dog_toy), 5 → 12 (seattle_backyard), 5 → 8
(kerry_park), 5 → 8 (seoul_bull). **The floor is roughly constant and the signal
grows.** So the reason `min_inliers = 8` is
punishing at n=10 is not that the noise floor is lower there; it is that a true
candidate at n=10 usually has five or six inliers and cannot clear eight.

### What each bar buys, per stage

`c<v>` is a constant floor of `v`; `f<b>_<r>` is `max(b, ceil(r·n))`;
`s<b>_<r>_<c>` is `min(c, max(b, ceil(r·n)))`, a floor that rises with the stage
and then stops; `r<b>_<r>` adds an inlier-ratio floor to a constant. Denominator
fixed at the nearest-50 ground truth throughout.

**DinoLedge**

| n | bar | found | recall≥3 | prec≥3 | corr recall | warp ok10 | false | never-covis |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 10 | c4 | 7.0 | 0.38 | 0.86 | 0.09 | 0.93 | 0.50 | 0.08 |
| 10 | **c5** | 5.0 | 0.29 | 0.90 | 0.08 | 0.93 | 0.28 | 0.05 |
| 10 | c6 | 3.2 | 0.18 | 0.92 | 0.06 | 0.93 | 0.13 | 0.00 |
| 10 | c8 | 0.9 | 0.06 | 0.99 | 0.02 | 0.97 | 0.00 | 0.00 |
| 15 | c5 | 8.4 | 0.45 | 0.86 | 0.16 | 0.90 | 0.67 | 0.07 |
| 15 | **c6** | 6.7 | 0.37 | 0.89 | 0.14 | 0.91 | 0.38 | 0.03 |
| 15 | c8 | 3.8 | 0.23 | 0.92 | 0.10 | 0.94 | 0.13 | 0.00 |
| 20 | c6 | 9.0 | 0.49 | 0.87 | 0.20 | 0.84 | 0.72 | 0.07 |
| 20 | **c8** | 6.2 | 0.34 | 0.90 | 0.17 | 0.86 | 0.37 | 0.03 |
| 25 | c6 | 10.8 | 0.56 | 0.84 | 0.26 | 0.85 | 1.03 | 0.07 |
| 25 | **c8** | 8.2 | 0.44 | 0.88 | 0.23 | 0.85 | 0.57 | 0.05 |
| 50 | c6 | 18.4 | 0.84 | 0.71 | 0.65 | 0.85 | 3.83 | 0.33 |
| 50 | **c8** | 14.7 | 0.74 | 0.76 | 0.60 | 0.86 | 2.07 | 0.18 |
| 50 | c12 | 10.3 | 0.56 | 0.86 | 0.50 | 0.89 | 0.85 | 0.00 |

Read the n=10 rows against the n=50 ones. A floor of **5 at n=10** gives 0.90
precision and 0.28 false candidates per patch; the shipped floor of 8 at n=50
gives 0.76 and 2.07. Ten features with a low bar is a *more* precise answer than
fifty features with the shipped bar -- it is simply a much smaller one
(recall 0.29 against 0.74).

**dino_dog_toy, same cut**

| n | bar | found | recall≥3 | prec≥3 | warp ok10 | false | never-covis |
|--:|---|--:|--:|--:|--:|--:|--:|
| 10 | c5 | 3.0 | 0.31 | 0.47 | 0.91 | 0.47 | 0.00 |
| 10 | c6 | 2.0 | 0.17 | 0.56 | 0.94 | 0.25 | 0.00 |
| 10 | c8 | 0.6 | 0.05 | 0.61 | 0.91 | 0.08 | 0.00 |
| 25 | c6 | 6.4 | 0.54 | 0.38 | 0.86 | 1.48 | 0.02 |
| 25 | c8 | 4.3 | 0.37 | 0.41 | 0.85 | 0.88 | 0.00 |
| 50 | c8 | 8.6 | 0.67 | 0.35 | 0.70 | 2.03 | 0.08 |

The stepped bars `s5_0.4_8` (= `min(8, max(5, ceil(0.4 n)))`, giving 5 at n≤12,
6 at n=15, 8 from n=20 up) and `s4_0.5_8` track the hand-picked constants row for
row, which is what makes them usable as a single rule. The purely proportional
`f` bars over-tighten at the cap -- `f5_0.3` is a floor of 15 at n=50 and costs
0.30 of DinoLedge's image recall.

The ratio bars `r6_0.3` and `r8_0.3` are **identical, row for row, to their
constant halves at every stage up to n=25 on all five captures**, because a
candidate with six inliers out of the 13 to 16 correspondences a candidate is
offered at those sizes already passes a 0.3 ratio. At n=50 the ratio does bite
where a candidate is offered many correspondences -- seattle_backyard's image
recall goes 0.94 at `c6` to 0.68 at `r6_0.3` -- but it removes candidates rather
than reordering them, and `c8` gets to the same place (0.87 recall) with a better
false-candidate count. **The inlier ratio says nothing the count does not already
say**, which is worth knowing because it was the obvious alternative shape for a
stage-dependent bar.

---

## 4. Q3: does a small constellation ever beat the wide one?

For every (patch, candidate image) pair accepted at *both* a small stage and at
n=50 under the shipped floor of 8, the two affines were evaluated on the same
fixed ground-truth points: `res_10` (the centre's own neighbourhood) and `res_50`
(the whole disc). "Better" means beating the other by more than max(0.1 px, 10%).

| dataset | S | centre: small / cap / tie | median centre gain px | disc: small / cap / tie | median disc gain px | only-small (true) | only-cap (true) |
|---|--:|---|--:|---|--:|---|---|
| DinoLedge | 10 | 20 / 13 / 7 | +0.15 | 12 / 31 / 8 | −0.72 | **0 (0)** | 830 (706) |
| DinoLedge | 15 | 70 / 41 / 16 | +0.20 | 51 / 134 / 18 | −0.64 | **0 (0)** | 654 (538) |
| DinoLedge | 20 | 83 / 68 / 17 | +0.10 | 99 / 180 / 35 | −0.48 | **0 (0)** | 507 (405) |
| DinoLedge | 25 | 75 / 78 / 32 | 0.00 | 117 / 219 / 58 | −0.38 | **0 (0)** | 391 (301) |
| dino_dog_toy | 10 | 9 / 1 / 1 | +1.51 | 11 / 14 / 1 | −0.83 | **0 (0)** | 481 (364) |
| dino_dog_toy | 15 | 18 / 11 / 0 | +0.96 | 23 / 49 / 7 | −0.93 | **0 (0)** | 404 (300) |
| dino_dog_toy | 20 | 26 / 8 / 5 | +0.75 | 38 / 60 / 16 | −0.56 | **0 (0)** | 324 (236) |
| dino_dog_toy | 25 | 25 / 12 / 4 | +0.75 | 36 / 90 / 16 | −0.96 | **0 (0)** | 257 (188) |
| kerry_park | 15 | 5 / 3 / 1 | +1.38 | 5 / 9 / 2 | −1.24 | **0 (0)** | 255 (75) |
| kerry_park | 25 | 11 / 3 / 1 | +1.11 | 17 / 21 / 1 | −0.25 | **0 (0)** | 148 (44) |
| seoul_bull | 20 | 19 / 10 / 1 | +0.47 | 14 / 23 / 4 | −0.81 | **0 (0)** | 66 (48) |
| seoul_bull | 25 | 19 / 14 / 5 | +0.12 | 15 / 32 / 7 | −0.52 | 1 (1) | 49 (35) |
| seattle_backyard | 25 | 1 / 1 / 0 | −0.08 | 28 / 25 / 11 | +0.09 | 1 (0) | 82 (60) |

Three findings, in order of how firmly the data holds them.

**A small stage essentially never finds an image the cap misses.** Across five
captures, four start sizes, 60 patches each -- 1,200 patch-stage comparisons --
the "only-small" column is **0** everywhere except two single candidates
(seoul_bull at S=25, seattle_backyard at S=25). The reverse column runs into the
hundreds. This is the result that kills first-hit stopping as a way to gain
anything: the early stages have no information the last stage lacks. It also
explains why the union rule (variant C3) is worthless -- the union of images
found at any stage is 8.7 against 8.6 per patch on dino_dog_toy, 2.0 against 1.9
on seoul_bull, 6.0 against 5.9 on kerry_park, and identical on DinoLedge.

**The small stage's warp is better at the centre and worse over the disc, on the
wide-baseline captures.** dino_dog_toy at S=10 is 9 wins to 1 loss at the centre
with a median gain of 1.51 px, and 11 to 14 over the disc with a median *loss* of
0.83 px. kerry_park and seoul_bull agree in sign. This is the affine's
first-order error doing exactly what the spec says it does: the term it drops
grows with the patch, so fitting on a small disc buys accuracy at the centre and
sells it at the rim.

**On DinoLedge the centre gain is within noise.** 20/13/7 at S=10 and 75/78/32 at
S=25, with median gains of +0.15 and 0.00 px. A video walk's consecutive frames
differ by a few percent of scale, so the affine is nearly exact over the whole
disc and there is nothing for a smaller disc to improve. The progressive warp
argument is a wide-baseline argument.

---

## 5. Stopping and combination rules

All replayed offline from the same table, at the shipped floor of 8 so the
comparison is against the shipped behaviour. "Modelled ms" is §6's cost model;
`vs base` is against a single query at n=50.

The rules compared:

- **A, baseline.** One query at n=50, and at n=25 for reference.
- **B, first-hit return.** Walk the schedule; stop at the first stage where any
  image passes the bar; return that stage's candidates.
- **C, per-image lock.** Run every stage to the cap. An image enters the answer
  at the first stage that accepts it, and keeps that stage's warp (`lock-first`).
  Variants: keep the *last* accepting stage's warp (`lock-last`, which is the
  baseline by construction once the union adds nothing), or the stage with the
  highest inlier ratio (`lock-ratio`). The image set is the union over stages,
  which is variant C3.
- **D, plateau.** Stop when the best accepted image's inlier count fails to grow
  from one stage to the next.

### DinoLedge

| rule | stages | found | recall≥3 | prec≥3 | corr recall | warp ok50 | warp ok10 | false | never-covis | stop median | modelled ms | vs base |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **A single N=50** | 50 | 14.7 | 0.74 | 0.76 | 0.60 | 0.76 | 0.86 | 2.07 | 0.18 | 50 | 14.2 | 1.00 |
| A single N=25 | 25 | 8.2 | 0.44 | 0.88 | 0.23 | 0.70 | 0.85 | 0.57 | 0.05 | 25 | 7.2 | 0.51 |
| B first-hit | 10+20+30+40+50 | 4.0 | 0.27 | 0.91 | 0.13 | 0.60 | 0.88 | 0.28 | 0.03 | 20 | – | – |
| B first-hit | 15+30+50 | 4.5 | 0.32 | 0.90 | 0.15 | 0.59 | 0.95 | 0.17 | 0.00 | 15 | – | – |
| B first-hit | 25+50 | 8.6 | 0.50 | 0.86 | 0.26 | 0.69 | 0.85 | 0.60 | 0.05 | 25 | ≤ 7.2 | ≤ 0.51 |
| C lock-first | 25+50 | 14.7 | 0.74 | 0.76 | 0.39 | 0.67 | 0.83 | 2.07 | 0.18 | 50 | 15.7 | 1.11 |
| C lock-ratio | 25+50 | 14.7 | 0.74 | 0.76 | 0.43 | 0.68 | 0.84 | 2.07 | 0.18 | 50 | 15.7 | 1.11 |
| C lock-last | 25+50 | 14.7 | 0.74 | 0.76 | 0.60 | 0.76 | 0.86 | 2.07 | 0.18 | 50 | 15.7 | 1.11 |
| C lock-first | 10+20+30+40+50 | 14.7 | 0.74 | 0.76 | 0.33 | 0.61 | 0.84 | 2.07 | 0.18 | 50 | – | – |
| D plateau | 10+30+50 | 14.7 | 0.74 | 0.76 | 0.60 | 0.76 | 0.86 | 2.07 | 0.18 | 50 | – | – |

DinoLedge's cost column is given only where §6's decomposition is valid, which is
at a cache budget of 512 MiB or more and at the three prefixes 10 / 25 / 50 that
were timed there. The rows marked "–" involve prefixes not timed at that budget;
their quality columns are unaffected.

### dino_dog_toy

| rule | stages | found | recall≥3 | prec≥3 | corr recall | warp ok50 | warp ok10 | false | never-covis | stop median | modelled ms | vs base |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **A single N=50** | 50 | 8.6 | 0.67 | 0.35 | 0.40 | 0.65 | 0.70 | 2.03 | 0.08 | 50 | 12.7 | 1.00 |
| A single N=25 | 25 | 4.3 | 0.37 | 0.41 | 0.13 | 0.49 | 0.85 | 0.88 | 0.00 | 25 | 6.2 | 0.48 |
| B first-hit | 10+20+30+40+50 | 2.0 | 0.24 | 0.41 | 0.08 | 0.42 | 0.94 | 0.55 | 0.00 | 20 | 4.5 | 0.35 |
| B first-hit | 15+30+50 | 3.2 | 0.35 | 0.40 | 0.12 | 0.47 | 0.87 | 0.72 | 0.00 | 30 | 5.9 | 0.47 |
| B first-hit | 25+50 | 4.7 | 0.44 | 0.39 | 0.16 | 0.50 | 0.85 | 1.00 | 0.00 | 25 | 4.2 | 0.33 |
| C lock-first | 25+50 | 8.6 | 0.67 | 0.35 | 0.26 | 0.53 | 0.86 | 2.03 | 0.08 | 50 | 11.9 | 0.93 |
| C lock-ratio | 25+50 | 8.6 | 0.67 | 0.35 | 0.35 | 0.56 | 0.79 | 2.03 | 0.08 | 50 | 11.9 | 0.93 |
| C lock-last | 25+50 | 8.6 | 0.67 | 0.35 | 0.40 | 0.65 | 0.70 | 2.03 | 0.08 | 50 | 11.9 | 0.93 |
| C lock-first | 10+20+30+40+50 | 8.6 | 0.67 | 0.35 | 0.21 | 0.49 | **0.91** | 2.05 | 0.10 | 50 | 16.4 | 1.29 |
| D plateau | 10+30+50 | 8.6 | 0.67 | 0.35 | 0.40 | 0.65 | 0.70 | 2.03 | 0.08 | 50 | 12.8 | 1.01 |

### kerry_park and seoul_bull, the two-stage rules only

| dataset | rule | stages | found | recall≥3 | prec≥3 | corr recall | warp ok50 | warp ok10 | false | modelled ms | vs base |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| kerry_park | A single N=50 | 50 | 5.9 | 0.77 | 0.19 | 0.50 | 0.67 | 0.75 | 4.18 | 4.9 | 1.00 |
| kerry_park | B first-hit | 25+50 | 3.7 | 0.51 | 0.18 | 0.16 | 0.62 | 1.00 | 2.68 | 2.8 | 0.56 |
| kerry_park | C lock-first | 25+50 | 5.9 | 0.77 | 0.19 | 0.29 | 0.67 | **0.92** | 4.18 | 6.2 | 1.25 |
| kerry_park | C lock-first | 10+20+30+40+50 | 6.0 | 0.80 | 0.18 | 0.26 | 0.60 | **0.88** | 4.27 | 8.8 | 1.78 |
| seoul_bull | A single N=50 | 50 | 1.9 | 0.69 | 0.65 | 0.50 | 0.81 | 0.85 | 0.40 | 3.7 | 1.00 |
| seoul_bull | B first-hit | 25+50 | 1.5 | 0.55 | 0.65 | 0.28 | 0.69 | 0.84 | 0.32 | 2.0 | 0.53 |
| seoul_bull | C lock-first | 25+50 | 2.0 | 0.69 | 0.65 | 0.34 | 0.70 | 0.83 | 0.40 | 4.3 | 1.14 |
| seoul_bull | C lock-first | 10+20+30+40+50 | 2.0 | 0.70 | 0.63 | 0.31 | 0.70 | **0.90** | 0.43 | 5.8 | 1.57 |
| seattle_backyard | A single N=50 | 50 | 3.5 | 0.87 | 0.21 | 0.55 | 0.69 | (2 cases) | 0.50 | 4.6 | 1.00 |
| seattle_backyard | C lock-first | 25+50 | 3.5 | 0.87 | 0.21 | 0.39 | 0.61 | (2 cases) | 0.52 | 5.6 | 1.21 |

### What the rules say

**B, first-hit, loses recall and buys nothing but latency.** On every capture it
lands on the same frontier a single smaller query already occupies, and usually
slightly inside it: DinoLedge first-hit on 25+50 returns recall 0.50 at 0.86
precision for the price of a query at n=25, against 0.44 at 0.88 for the plain
n=25 query -- a real but tiny gain, because the patches that fail at 25 go on to
50 and a few of them succeed there. Aggressive schedules are worse: first-hit on
15+30+50 stops at n=15 for half the patches and returns recall 0.32. Given §4's
result that a small stage finds no image the cap misses, this is what had to
happen. **The one thing first-hit does buy is precision and warp quality**, for
the same reason a smaller constellation does: DinoLedge 0.86–0.91 precision and
0.85–0.95 `warp ok10` against 0.76 and 0.86 at the baseline. A caller that wants
"one image I am sure about, fast" is well served; a caller that wants the list is
not.

**C, per-image lock with the *first* accepting stage's warp, is the rule with a
real gain.** The image set is the baseline's -- the union adds 0.0 to 0.1 images
per patch anywhere -- and the precision, false-candidate and never-covisible
columns are the baseline's too. What changes is the warp. Over the five-stage
schedule `warp ok10` goes 0.70 → **0.91** on dino_dog_toy, 0.75 → **0.88** on
kerry_park and 0.85 → **0.90** on seoul_bull, while `warp ok50` goes the other
way, 0.65 → 0.49, 0.67 → 0.60 and 0.81 → 0.70. That is §4's trade applied per
image, and it costs correspondence recall heavily -- 0.40 → 0.21 on
dino_dog_toy, 0.60 → 0.33 on DinoLedge -- because a warp frozen at n=10 comes
with n=10's inlier set.

**On DinoLedge the lock rule is a pure loss**, exactly as §4 predicts for a video
walk: `warp ok10` 0.86 → 0.84 and `warp ok50` 0.76 → 0.61 over the five stages,
0.86 → 0.83 and 0.76 → 0.67 over two. There is no centre accuracy to buy there,
so freezing an earlier warp only discards evidence.

**C2: `lock-ratio` sits between `lock-first` and `lock-last` on every column and
is dominated by both.** It has no case: if the centre warp is what is wanted,
`lock-first` gives more of it; if the correspondences are what is wanted,
`lock-last` gives more of them. `lock-last` reproduces the baseline exactly,
which is the consistency check that the union is the baseline's image set.

**C3, the union of images found at any stage, is worthless**, for the reason §4
gives: 8.7 against 8.6 images per patch on dino_dog_toy, 2.0 against 1.9 on
seoul_bull, 6.0 against 5.9 on kerry_park, identical on DinoLedge and
seattle_backyard.

**D, plateau, costs more than the baseline and returns the baseline.** Its stop
stage is the cap on essentially every patch -- the best image's inlier count
almost always keeps growing -- so it pays every intermediate fit for nothing:
1.01× on dino_dog_toy over 10+30+50, 2.08× over the ten-stage schedule (2.22× on
seoul_bull), with recall 0.63 against the baseline's 0.67 because the handful of
patches that do stop early stop too early. Dropped.

**The composition that is actually wanted.** `lock-first` and `lock-last` are
measured over the *same* candidate set at the same stages, so a rule that takes
each image's **warp** from its first accepting stage and its **inlier
correspondences** from the cap stage has the `warp ok10` of the first row and the
correspondence recall of the second: on dino_dog_toy, 0.91 and 0.40 against the
baseline's 0.70 and 0.40. This composition is not itself a measured arm -- it is
two measured columns from two arms over one candidate set -- and it is what §8
recommends.

---

## 6. Cost model

Two things were timed per patch: `LazyKdForest.query` on a batch of `B`
descriptors (the forest search an increment pays for), and the whole
`constellation_query` at prefix `n` at the production floor of 8. The difference
is everything the query does besides the search -- origin and geometry
resolution, the per-cell collapse, and RANSAC per candidate image -- and is
called `fit` below.

Both were re-timed in **random order with each measurement warm for its own
prefix** (`timing` mode). The ascending sweep the measurement pass takes reports
the wrong thing on a file-backed forest: the first prefix of each patch pays the
patch's first-touch reads and every later prefix inherits a cache the earlier
ones filled, which is a *progressive run's stage cost*, not a standalone query's.

### Search is linear in batch size, with no fixed cost worth avoiding

| dataset | ms per feature | intercept ms | search @10 | search @25 | search @50 |
|---|--:|--:|--:|--:|--:|
| DinoLedge, 1 GiB cache | 0.22 | +0.28 | 2.5 | 5.7 | 11.4 |
| dino_dog_toy | 0.152 | −0.07 | 1.05 | 2.54 | 7.55 |
| kerry_park | 0.066 | +0.01 | 0.71 | 1.78 | 3.29 |
| seattle_backyard | 0.063 | −0.00 | 0.65 | 1.63 | 3.15 |
| seoul_bull | 0.059 | +0.00 | 0.60 | 1.50 | 2.96 |

**This is the answer to "is D=5 wasteful".** It is not: on the four small
captures the fitted intercept is within ±0.07 ms of zero against 0.06 to 0.15 ms
per feature, so a call costs less than one feature's worth before it looks
anything up, and ten five-feature searches cost what one fifty-feature search
costs. The whole overhead of a progressive run therefore sits in the **re-fits**:

    progressive cost of stages n₁ < … < n_k  =  single query at n_k  +  Σ_{i<k} fit(nᵢ)

### Per-stage totals and the fit term

| n | seoul_bull total / fit | seattle total / fit | kerry total / fit | dino_dog total / fit | DinoLedge (512 MiB) total / fit |
|--:|---|---|---|---|---|
| 10 | 1.00 / 0.40 | 1.27 / 0.63 | 1.40 / 0.69 | 1.94 / 0.88 | 3.2 / 0.7 |
| 20 | 1.65 / 0.47 | 2.10 / 0.82 | 2.28 / 0.87 | 3.50 / 1.48 | — |
| 25 | 1.99 / 0.50 | 2.49 / 0.86 | 2.76 / 0.98 | 4.17 / 1.62 | 7.2 / 1.5 |
| 30 | 2.35 / 0.54 | 2.89 / 0.99 | 3.06 / 0.94 | 4.75 / 1.69 | — |
| 50 | 3.73 / 0.77 | 4.59 / 1.45 | 4.94 / 1.64 | 12.69 / 5.15 | 14.2 / 2.8 |

DinoLedge's totals are the 512 MiB column of the sweep below and its fits are
those totals less the 1 GiB search times; the three budgets from 512 MiB up agree
to within the noise of 60 patches (14.2 / 15.9 / 15.6 ms at n=50), so the mix does
not matter.

So a two-stage 25+50 run costs one extra `fit(25)`: +13% on dino_dog_toy, +13%
on seoul_bull, +20% on kerry_park, +19% on seattle_backyard, **+11% on
DinoLedge**. A five-stage 10+20+30+40+50 run costs four extra fits: +55%, +77%,
+73% and +47% on seoul_bull, seattle_backyard, kerry_park and dino_dog_toy.

The `modelled ms` column in §5 differs from those percentages by a few points
because it interpolates the measured search curve rather than assuming the fitted
line, and the B=50 point sits above that line on some captures; where it does,
the model credits a split search with a saving that is not real. On
dino_dog_toy that pushes the two-stage lock to 0.93× the baseline, which should
be read as +13%, not as a saving. The percentages above are the ones to quote.

### DinoLedge's query cost is set by the cache budget, not by the constellation

This was not what the experiment set out to measure and it is the largest number
in the report. The 256 MiB cache budget that this evaluation and the 2026-09-14
one both used is **below the knee** for a 1,494 MB index, and almost all of the
per-query cost those reports attribute to the query -- 259 ms at N=50 before
#458, 562 ms after -- is cache thrash.

Bare forest search, median over 8 fresh patches, each query run twice and the
second timed:

| cache | search @10 | search @25 | search @50 |
|---|--:|--:|--:|
| 256 MiB | 2.7 ms | 39.1 ms | **381.9 ms** |
| 1 GiB | 2.5 ms | 5.7 ms | **11.4 ms** |
| 4 GiB | 2.5 ms | 5.5 ms | 11.1 ms |

Whole `constellation_query` at the production floor, all 60 patches, each query
run twice:

| cache | n=10 | n=25 | n=50 |
|---|--:|--:|--:|
| 256 MiB | 3.2 ms | 216.4 ms | **489.7 ms** |
| 512 MiB | 3.2 ms | 7.2 ms | **14.2 ms** |
| 1 GiB | 3.1 ms | 7.6 ms | 15.9 ms |
| 2 GiB | 3.1 ms | 7.3 ms | 15.6 ms |

And, separating first touch from repeat at n=50 over the same 60 patches:

| cache | first touch | repeat |
|---|--:|--:|
| 256 MiB | 500.2 ms | 489.7 ms |
| 512 MiB | 262.0 ms | 14.0 ms |

At 256 MiB the cache cannot hold even **one** 50-feature query's working set, so
repeating the identical query saves nothing and the search is wildly superlinear
in batch size. At 512 MiB it can: a fresh patch still costs 262 ms of file
reading, and a repeat costs 14 ms. Above 512 MiB nothing further changes. So:

- the per-prefix time curve of §2 and the ascending sweep -- 24 ms at n=10 rising
  to 498 ms at n=50 -- is a curve of **cache behaviour**, not of query work;
- at an adequate budget the query is 3 to 14 ms across the whole prefix range,
  which is the same order as the 85-image capture;
- and, as an accident, a progressive run at a budget *below* the knee would be
  far cheaper than a single query, because two 25-feature searches cost 78 ms
  where one 50-feature search costs 382 ms. That is an argument for fixing the
  budget, not for the progressive query, and it is not counted as a gain in §8.

### Cold against warm, on the four small captures

The measurement pass ran once cold and once warm over the same patches. On the
four small captures the two agree to within 10 to 15% at every prefix -- a 6 to
52 MB index is in the OS page cache after the first touch whatever the forest's
own budget does -- so there is no cold/warm distinction to manage there.

---

## 7. Random-pixel centres

The same protocol with each patch centre drawn uniformly over the frame instead
of landing on a keypoint -- the bench's real gesture. 60 patches per dataset,
same seed, reported separately because the constellation is then a disc around a
point with no feature at its middle.

| dataset | n=50, floor 8 | recall≥3 | prec≥3 | corr recall | warp ok50 | warp ok10 (cases) | false |
|---|---|--:|--:|--:|--:|--:|--:|
| DinoLedge | keypoint | 0.74 | 0.76 | 0.60 | 0.76 | 0.86 (208) | 2.07 |
| DinoLedge | pixel | 0.75 | 0.75 | 0.65 | 0.75 | 0.85 (218) | 1.82 |
| dino_dog_toy | keypoint | 0.67 | 0.35 | 0.40 | 0.65 | 0.70 (43) | 2.03 |
| dino_dog_toy | pixel | 0.69 | 0.36 | 0.46 | 0.67 | 0.57 (21) | 2.10 |
| seattle_backyard | keypoint | 0.87 | 0.21 | 0.55 | 0.69 | – (2) | 0.50 |
| seattle_backyard | pixel | 0.87 | 0.28 | 0.49 | 0.78 | – (6) | 0.63 |
| kerry_park | keypoint | 0.77 | 0.19 | 0.50 | 0.67 | 0.75 (24) | 4.18 |
| kerry_park | pixel | 0.91 | 0.12 | 0.70 | 0.76 | – (4) | 4.63 |
| seoul_bull | keypoint | 0.69 | 0.65 | 0.50 | 0.81 | 0.85 (41) | 0.40 |
| seoul_bull | pixel | 0.73 | 0.53 | 0.40 | 0.92 | 0.91 (22) | 0.28 |

Every recall and precision column overlaps the bootstrap interval of its
keypoint twin, and the two `warp ok10` columns that move a long way
(dino_dog_toy 0.70 → 0.57, kerry_park 0.75 → 1.00) have denominators of 21 and
4. **The progressive findings hold, and the centre-warp one is stronger.**
DinoLedge's
Q3 table, which was a wash on keypoint centres, comes out 25 wins to 7 losses at
S=10 with a median centre gain of 0.49 px on pixel centres, and 98 to 72 at
S=25. dino_dog_toy's `lock-first` on 25+50 takes `warp ok10` from 0.57 to 0.68.
The "only-small" column is 0 on 16 of the 20 dataset/start combinations and
never above 2 on the other four, six candidates in total over 1,200 patch-stage
comparisons, so §4's headline result holds on this patch set too.
That the effect is *larger* when the centre is not a keypoint is what one would
expect: a hand-placed centre is further from the features the affine is fitted
to, so where the affine is fitted matters more.

---

## 8. Recommendation

**Do not build a progressive query to find images.** Q1 and Q2 -- the best `S`
and the best `D` -- do not have useful answers, because the thing they would
govern (when to stop widening) has nothing to govern: no smaller prefix finds an
image the nearest-50 constellation misses, on 1,200 patch-stage comparisons
across five captures. Any early stop is a strict loss of recall, and it is a
worse trade than simply asking for a smaller constellation in the first place,
which costs less and returns the same thing.

**Do build a two-stage query whose product is the warp.** The recommendation, in
full:

- **S = 25, D = 25, N_max = 50.** Two stages, 25 then 50. Most of the measurable
  gain is available at two stages: five stages (10+20+30+40+50) buy another 0.05
  of `warp ok10` on dino_dog_toy and 0.07 on seoul_bull, both inside the standard
  error those denominators carry, and *lose* 0.04 on kerry_park, while costing
  four extra fits instead of one (1.29× to 1.79× the baseline against 1.11× to
  1.25×). If a capture is known to be wide-baseline and latency is not tight,
  S = 10, D = 10 is the arm with the single best measured `warp ok10`.
- **Acceptance bar: `min_inliers(n) = min(8, max(5, ceil(0.4 n)))`** -- 5 below
  n=13, 6 at n=15, 8 from n=20 up. At the recommended two stages this is simply
  8 at both, so the bar matters only if a caller wants a stage below 20; it is
  stated because the sweep supports it and because the first-hit variant needs
  it. Do **not** use a proportional bar without a cap (`f5_0.3` costs 0.30 of
  DinoLedge's image recall at n=50) and do not use an inlier-ratio bar (identical
  to its constant half below n=50 on all five captures, and only subtractive at
  the cap).
- **Combination rule: per-image lock, warp from the first accepting stage,
  correspondences from the last.** The image list, precision, false-candidate and
  never-covisible columns are the single-N=50 query's, unchanged. The warp is the
  25-feature one where 25 features sufficed.
- **Expected gain**: `warp ok10`, the share of accepted images whose affine
  places the ground truth within 3 px *near the patch centre*, 0.70 → 0.86 on
  dino_dog_toy and 0.75 → 0.92 on kerry_park over the 25+50 schedule; 0.85 →
  0.83 on seoul_bull and 0.86 → 0.83 on DinoLedge, both small losses. Only
  dino_dog_toy's and kerry_park's moves clear their standard errors
  (denominators 43 and 24; the five-stage schedule's 0.70 → 0.91 on
  dino_dog_toy is the clearest single number in the report). This is a
  wide-baseline gain and it is not universal -- see "when not to bother".
- **Expected loss**: none in image recall or precision; `warp ok50`, the same
  test over the whole 50-feature disc, falls 0.65 → 0.53 (dino_dog_toy),
  0.81 → 0.70 (seoul_bull), 0.76 → 0.67 (DinoLedge). A caller that warps the
  rim of the patch rather than its centre should keep the cap-stage affine,
  which is why both should be returned rather than one chosen.
- **Expected cost**: one extra RANSAC pass at n=25, which is `fit(25)` over
  `total(50)` -- +13% (seoul_bull), +19% (seattle_backyard), +20% (kerry_park),
  +13% (dino_dog_toy) and +11% (DinoLedge at a 512 MiB cache). Zero extra forest search,
  because the search is linear with no fixed cost and the first 25 features'
  hits are reused.

**Independent of all of the above: raise the `.kdf` cache budget.** The one
change in this report with a two-order-of-magnitude effect is not the
progressive query. 256 MiB against a 1,494 MB index costs 490 ms per repeated
50-feature query where 512 MiB costs 14 ms, and it is the budget every harness
in `scripts/` and every measurement of this query so far has used on DinoLedge.
A caller should size the budget against the corpus, and the docs and the
harnesses should say so. The knee was located between 256 and 512 MiB and was
not narrowed further.

**When not to bother.** On a video walk with small inter-frame parallax the
affine is already near-exact over the whole disc and there is nothing to gain:
DinoLedge's keypoint-centred Q3 is 75 wins to 78 losses at S=25. The gain is a
wide-baseline gain, and a caller that knows its capture is a walk should ask for
one query at 50 and stop.

---

## 9. What to change in the Rust API (sketch, not implemented)

The measurement ran the prefixes as independent `constellation_query` calls,
which re-searches every feature at every stage. A real implementation would not.
The shape that follows from §6 -- search is linear and free to split, the fit is
what costs -- is a query object that owns the hits and re-fits on demand:

```rust
pub struct ProgressiveConstellation<'a, S> { /* hits so far, per candidate image */ }

impl<'a, S: ForestScalar> ProgressiveConstellation<'a, S> {
    /// Start from the features nearest `center`, ordered by distance.
    pub fn new(query: &Constellation<'a, S>, center: [f32; 2],
               params: &ConstellationParams) -> Self;
    /// Search the next `count` features and re-fit every candidate image.
    /// Returns this stage's matches; earlier features are not searched again.
    pub fn extend<I, F>(&mut self, index: &I, sources: &F, count: usize)
        -> Result<&[ConstellationMatch], KdfError>;
    /// Every image accepted at any stage, each carrying the affine of the stage
    /// that first accepted it *and* the inlier set of the last.
    pub fn locked(&self) -> Vec<LockedMatch>;
}

pub struct LockedMatch {
    pub image_index: u32,
    pub affine: [[f64; 3]; 2],        // from the first accepting stage
    pub affine_wide: [[f64; 3]; 2],   // from the last stage
    pub accepted_at: usize,           // constellation size at that stage
    pub inliers: usize,
    pub inlier_correspondences: Vec<ConstellationCorrespondence>,  // last stage
}
```

Three things the measurement says the implementation needs. The constellation
must be **ordered by distance from the centre**, which `constellation_at_pixel`
already computes and throws away, so the order is free. The per-cell
`one_hit_per_image` collapse must be **incremental**: a cell is one (feature,
image) pair and a new feature adds new cells without disturbing old ones, so the
collapse does not have to be redone. And the re-fit is per candidate image and
seeded from `seed + image_index`, so a candidate whose correspondence list did
not grow at this stage can **keep its previous fit unchanged** -- which the cost
model does not assume and which would make the extra fits cheaper than the +11%
to +20% quoted above.

`min_inliers` should become a function of the stage size rather than a constant
if the entry point is ever used below n=20; at the recommended two stages a
constant 8 is right.

Unrelated to the progressive query and worth more than it: `LazyKdForestOptions`
gives no guidance on sizing `cache_bytes` against the corpus, and §6 shows the
consequence. A default derived from the file's own size, or at minimum a
documented rule and a warning when the budget is far below the knee, would be
worth more to a caller of this query than anything in this sketch.

---

## 10. What did not work, and what was skipped

- **The first cost decomposition was wrong and had to be redone.** Timing the
  prefixes in ascending order within a patch, as the measurement pass does,
  measures a progressive run's stage costs rather than standalone queries: the
  small prefixes pay the patch's first-touch reads and the large ones inherit a
  warm cache. On DinoLedge that produced a search curve that was non-monotonic
  and superlinear and a modelled two-stage run that came out *cheaper* than a
  single query, which looked impossible. A `timing` mode was added that shuffles
  the (patch, prefix) jobs and takes the second of two consecutive runs; §6 uses
  only those numbers. The first numbers are still in the raw JSON under
  `search_ms` / `production_ms` and should not be read.
- **The shuffled re-timing did not fix DinoLedge either**, and chasing that is
  what turned up the cache finding: the superlinearity survived the change of
  design because it is real at a 256 MiB budget. The per-stage decomposition on
  DinoLedge is therefore reported only at 512 MiB and above, and only at the
  three prefixes 10 / 25 / 50, which is all that was re-timed at those budgets.
  The full twelve-prefix sweep was **not** repeated at an adequate cache, so
  DinoLedge's `modelled ms` column in §5 is blank for the rules that need other
  prefixes.
- **The 60-patch tables for DinoLedge in §2 to §5 are all at 256 MiB.** Only
  wall-clock columns are affected by that; every quality column is a function of
  the result, not of the cache, and `LazyKdForest` returns the same answer at
  any budget.
- **The four small captures are not the 2026-09-14 report's.** Their workspaces
  were reused as found rather than re-bootstrapped, so they carry COLMAP SIFT and
  sparser solves; §1 quantifies the gap and §2 shows the effect. Only DinoLedge
  reproduces that report exactly. Re-solving the four was not attempted.
- **The bar sweep is truncated at 4 inliers.** All queries ran at
  `min_inliers = 4`, so nothing below that floor is measured and the "median
  inlier count of a wrong candidate" figure from the 2026-09-14 report cannot be
  reproduced directly; the 90th percentile and maximum are used instead.
- **`warp ok10` on seattle_backyard has two cases** and is not read. kerry_park
  (24) and seoul_bull (41) are thin. DinoLedge (208) and dino_dog_toy (43) carry
  the finding.
- **The composed rule of §5 was not run as its own arm.** It is two measured
  columns from two arms over one candidate set, which is sound but is not a
  single end-to-end measurement of the thing recommended.
- **The cache sweep of §6 was run as a one-off script, not through the
  harness**, so it is not reproducible from the Appendix's commands. It timed
  8 or 60 patches at three prefixes and four budgets, each query run twice; its
  numbers are in §6 and nowhere else. Turning it into a mode of the harness, and
  sweeping the budget finely enough to locate the knee, was not done.
- **`--multiplicity`-style crowding counts, crop dumps and parameter sweeps over
  `k`, `threshold_px` and `iterations` were not repeated.** This report changes
  only the constellation size and the acceptance bar; every other parameter sat
  at its shipped default in every arm.
- **Nothing was implemented.** No Rust was touched, no defaults changed, nothing
  committed. §9 is a sketch.

---

## Appendix: reproducing

```bash
# One .kdf per workspace with SIFT sources, four trees, leaf 16, 2 KiB blocks,
# 1 MiB chunks, seed 0 (the shape scripts/kdf_patch_localize.py builds).
pixi run -e test python scripts/kdf_constellation_progressive_eval.py measure \
    --workspace WS --kdf capture.kdf --sfmr WS/sfmr/solve.sfmr \
    --label NAME --patches 60 --cache-mib 64 \
    --out target/constellation-progressive/NAME.json
pixi run -e test python scripts/kdf_constellation_progressive_eval.py timing \
    --out target/constellation-progressive/NAME.json
pixi run -e test python scripts/kdf_constellation_progressive_eval.py analyze \
    target/constellation-progressive/*.json \
    --out target/constellation-progressive/analysis.json
```

`--centres pixel` selects the §7 patch set. Raw per-patch tables and the analysis
JSON are in `target/constellation-progressive/`, not in the repository.
