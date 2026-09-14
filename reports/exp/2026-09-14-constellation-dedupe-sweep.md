# The knobs re-swept with one hit per image, and the defaults that follow, 2026-09-14

[The collapse measurement](2026-09-14-constellation-same-image-ratio.md) ended
with a condition attached: `one_hit_per_image = true` should be turned on
together with a re-run of the `k` and `iterations` sweeps, and a look at
`min_inliers`, because those three were judged on correspondence counts per
candidate image that the collapse halves. This is that re-run, on the same five
captures and the same 60 seeded patches, and then the defaults.

**Answer, in one line.** The condition is discharged and nothing else moves:
with the collapse on, `k = 32` is no longer harmful on a small corpus -- the
`warp ok` spread across `k` = 8 to 64 falls from 0.66 / 0.32 / 0.00 to
0.93 / 0.86 / 0.93 / 0.83 on the seventeen-image capture, and from
0.81 / 0.65 / 0.32 to 0.92 / 0.91 / 0.93 / 0.89 on the twenty-six-image one --
so the one conclusion that disagreed across captures now agrees, and there is
nothing left for a change of `k` to fix. `iterations = 200` sits on a flat part
of its curve, and `min_inliers = 8` is still the knee between recall and false
candidates. So one default changes: `one_hit_per_image` is now `true`.

---

## 1. Protocol

The §1 protocol of
[the constellation query evaluation](2026-09-14-constellation-query-eval.md),
re-used unchanged: the same five `.kdf` corpora over the same workspaces, the
same `sift_files` ground-truth `.sfmr` per capture, the same 60 seeded patch
centres (numpy PCG64, seed 0: a random registered image, then a random keypoint
of it), the same scoring, and every query through `LazyKdForest`. Constellation
size is 50 features everywhere, the radius being the distance to the 50th
nearest keypoint of the query image, so every sweep point sees the same patches
and the same features. Cache budget 256 MiB for DinoLedge and 64 MiB for the
four small captures.

All 60 patches on all five captures, DinoLedge included: the full grid cost it
about 35 s per sweep point, so nothing had to be thinned.

**What is held fixed.** `one_hit_per_image = true` and `same_image_ratio = 1.0`
in every arm, and `max_leaf_checks = 512`, `threshold_px = 8`, `max_scale = 4`,
`min_correspondences = 3`, `seed = 0` throughout.

**What is swept**, one knob at a time with the other two at the current
defaults:

| knob | values | current default |
|---|---|---|
| `k` | 8, 16, 32, 64 | 32 |
| `iterations` | 100, 200, 400, 1000 | 200 |
| `min_inliers` | 6, 8, 10 | 8 |

Each knob's list contains its default, so the three default rows and the
deduped baseline are four measurements of one configuration; they came out
identical on every column of every capture, which is the consistency check that
the grid is on the patches it claims. The deduped baseline also reproduces the
"dedupe only" row of the collapse report exactly on all five captures.

`scripts/kdf_constellation_eval.py` gained `--sweep-grid`, which replaces the
built-in sweep list with knobs and values given as
`"k=8,16,32,64;iterations=100,200,400,1000;min_inliers=6,8,10"`, and its
`--one-hit-per-image` became a `--no-`able flag defaulting to the core default.

**What counts as noise.** The per-patch standard error at the deduped baseline,
over the 60 patches:

| capture | `warp ok` denominator | s.e. recall≥3 | s.e. corr recall | s.e. prec≥3 | s.e. `warp ok` |
|---|--:|--:|--:|--:|--:|
| DinoLedge | 586 | 0.036 | 0.035 | 0.038 | 0.018 |
| dino_dog_toy | 407 | 0.036 | 0.037 | 0.047 | 0.021 |
| seattle_backyard | 252 | 0.035 | 0.038 | 0.043 | 0.016 |
| kerry_park | 53 | 0.083 | 0.065 | 0.045 | 0.044 |
| seoul_bull | 42 | 0.072 | 0.052 | 0.091 | 0.040 |

So a move of less than about 0.07 on one capture is inside two standard errors
and carries no evidence on its own; on kerry_park and seoul_bull, whose `warp
ok` denominators are in the dozens, the band is 0.10 to 0.18 on the rates
averaged per patch. What is read below is therefore a column that moves the same
way on several captures at once, not a single cell.

---

## 2. The deduped baseline

`k = 32`, `iterations = 200`, `min_inliers = 8`, with the collapse on. This is
the row every sweep below is read against.

| capture | recall≥3 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 0.74 | 0.76 | 0.60 | 1.78 | 4.5 | **0.76** | 16 | 2.08 | 0.20 | 562 |
| dino_dog_toy | 0.69 | 0.67 | 0.52 | 1.56 | 5.1 | **0.76** | 24 | 0.98 | 0.00 | 9 |
| seattle_backyard | 0.69 | 0.88 | 0.60 | 1.02 | 3.4 | **0.93** | 39 | 0.05 | 0.00 | 5 |
| kerry_park | 0.82 | 0.18 | 0.67 | 1.23 | 4.2 | **0.89** | 31 | 2.42 | 0.68 | 6 |
| seoul_bull | 0.52 | 0.68 | 0.29 | 1.05 | 3.0 | **0.93** | 44 | 0.07 | 0.00 | 4 |

---

## 3. `k`: the disagreement between large and small captures is gone

| capture | k | recall≥3 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 8 | 0.53 | 0.88 | 0.48 | 1.70 | 4.6 | **0.82** | 16 | 0.35 | 0.00 | 572 |
| | 16 | 0.70 | 0.83 | 0.58 | 1.71 | 4.6 | **0.77** | 16 | 0.77 | 0.05 | 594 |
| | **32** | 0.74 | 0.76 | 0.60 | 1.78 | 4.5 | **0.76** | 16 | 2.08 | 0.20 | 628 |
| | 64 | 0.74 | 0.74 | 0.60 | 1.86 | 4.6 | **0.75** | 17 | 3.35 | 0.52 | 556 |
| dino_dog_toy | 8 | 0.52 | 0.76 | 0.44 | 1.56 | 4.7 | **0.81** | 16 | 0.17 | 0.00 | 7 |
| | 16 | 0.67 | 0.72 | 0.52 | 1.52 | 4.9 | **0.80** | 19 | 0.32 | 0.00 | 8 |
| | **32** | 0.69 | 0.67 | 0.52 | 1.56 | 5.1 | **0.76** | 24 | 0.98 | 0.00 | 8 |
| | 64 | 0.68 | 0.65 | 0.52 | 1.61 | 5.0 | **0.77** | 33 | 1.15 | 0.02 | 18 |
| seattle_backyard | 8 | 0.66 | 0.91 | 0.59 | 0.91 | 3.5 | **0.92** | 22 | 0.02 | 0.00 | 4 |
| | 16 | 0.71 | 0.90 | 0.62 | 1.14 | 3.5 | **0.91** | 30 | 0.02 | 0.00 | 5 |
| | **32** | 0.69 | 0.88 | 0.60 | 1.02 | 3.4 | **0.93** | 39 | 0.05 | 0.00 | 6 |
| | 64 | 0.66 | 0.90 | 0.58 | 1.10 | 3.6 | **0.89** | 47 | 0.03 | 0.00 | 6 |
| kerry_park | 8 | 0.78 | 0.23 | 0.59 | 0.76 | 3.7 | **0.87** | 18 | 1.80 | 0.45 | 5 |
| | 16 | 0.84 | 0.20 | 0.67 | 1.15 | 4.6 | **0.87** | 23 | 2.30 | 0.63 | 5 |
| | **32** | 0.82 | 0.18 | 0.67 | 1.23 | 4.2 | **0.89** | 31 | 2.42 | 0.68 | 6 |
| | 64 | 0.86 | 0.22 | 0.67 | 1.20 | 4.0 | **0.85** | 40 | 2.35 | 0.68 | 8 |
| seoul_bull | 8 | 0.54 | 0.65 | 0.32 | 1.20 | 3.4 | **0.93** | 26 | 0.08 | 0.00 | 3 |
| | 16 | 0.56 | 0.65 | 0.33 | 1.29 | 3.9 | **0.86** | 34 | 0.07 | 0.00 | 3 |
| | **32** | 0.52 | 0.68 | 0.29 | 1.05 | 3.0 | **0.93** | 44 | 0.07 | 0.00 | 4 |
| | 64 | 0.53 | 0.72 | 0.26 | 1.17 | 3.2 | **0.83** | 50 | 0.05 | 0.00 | 4 |

**The sign change, stated directly.** The earlier report's §3 found `warp ok`
collapsing as `k` rose on the small captures: seattle 0.81 / 0.65 / 0.32 and
seoul 0.66 / 0.32 / 0.00 at k = 16 / 32 / 64, against a flat 0.81 / 0.82 / 0.81
on DinoLedge. With the collapse on, those two columns are
0.91 / 0.93 / 0.89 and 0.86 / 0.93 / 0.83, which is flat to within the
0.04 standard error those two captures carry. The mechanism it named is visible
in `corr/img`: seoul went 51 / 96 / 189 across k = 16 / 32 / 64 and now goes
34 / 44 / 50, and seattle went 36 / 68 / 127 and now goes 30 / 39 / 47. Raising
`k` on a seventeen-image corpus mostly finds neighbours in cells that already
hold a hit, and those are exactly what the collapse drops, so the extra
neighbours no longer reach RANSAC as chaff.

**What `k` still does.** It buys images, and it buys them at the same rate on
every capture now. Correspondence recall climbs from k = 8 to k = 16 on all five
(0.48 → 0.58, 0.44 → 0.52, 0.59 → 0.62, 0.59 → 0.67, 0.32 → 0.33) and then
stops: from 16 to 64 it moves by at most 0.02 anywhere, which is inside the
noise on every capture. Image recall≥3 has the same shape, with k = 8 clearly
short on the two largest captures (DinoLedge 0.74 → 0.53, dino_dog_toy
0.69 → 0.52) and k = 16 within noise of 32.

**What it costs.** False candidates, monotonically and on every capture, because
`corr/img` still rises with `k` even after the collapse and a longer
correspondence list gives the search more chances at a spurious consensus:
DinoLedge 0.35 / 0.77 / 2.08 / 3.35 and never-covisible 0.00 / 0.05 / 0.20 /
0.52 across k = 8 / 16 / 32 / 64.

So the choice between 16 and 32 is a precision-against-recall trade of the kind
`min_inliers` already sells, with the two values inside each other's noise on
recall on four of five captures. On DinoLedge, the scale this query is aimed at,
32 is the better of the two on the recall columns (recall≥3 0.74 against 0.70,
correspondence recall 0.60 against 0.58) and worse on false candidates (2.08
against 0.77). Neither difference is decisive, and the reason to revisit `k` at
all -- that it disagreed across captures -- has been removed.

---

## 4. `iterations`: flat around 200

| capture | iterations | recall≥3 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 100 | 0.74 | 0.76 | 0.60 | 1.76 | 4.5 | **0.77** | 16 | 2.07 | 0.20 | 549 |
| | **200** | 0.74 | 0.76 | 0.60 | 1.78 | 4.5 | **0.76** | 16 | 2.08 | 0.20 | 590 |
| | 400 | 0.74 | 0.76 | 0.60 | 1.73 | 4.5 | **0.76** | 16 | 2.08 | 0.20 | 587 |
| | 1000 | 0.74 | 0.76 | 0.60 | 1.73 | 4.6 | **0.76** | 16 | 2.08 | 0.20 | 571 |
| dino_dog_toy | 100 | 0.68 | 0.68 | 0.52 | 1.56 | 5.1 | **0.78** | 24 | 0.92 | 0.00 | 7 |
| | **200** | 0.69 | 0.67 | 0.52 | 1.56 | 5.1 | **0.76** | 24 | 0.98 | 0.00 | 10 |
| | 400 | 0.70 | 0.66 | 0.53 | 1.57 | 5.1 | **0.76** | 24 | 1.02 | 0.02 | 11 |
| | 1000 | 0.71 | 0.65 | 0.54 | 1.57 | 5.2 | **0.74** | 24 | 1.08 | 0.02 | 14 |
| seattle_backyard | 100 | 0.63 | 0.90 | 0.56 | 1.08 | 3.7 | **0.90** | 40 | 0.03 | 0.00 | 5 |
| | **200** | 0.69 | 0.88 | 0.60 | 1.02 | 3.4 | **0.93** | 39 | 0.05 | 0.00 | 6 |
| | 400 | 0.72 | 0.88 | 0.63 | 0.99 | 3.3 | **0.92** | 39 | 0.05 | 0.00 | 6 |
| | 1000 | 0.74 | 0.88 | 0.64 | 1.01 | 3.4 | **0.92** | 40 | 0.05 | 0.00 | 8 |
| kerry_park | 100 | 0.82 | 0.20 | 0.65 | 1.23 | 3.5 | **0.92** | 32 | 2.28 | 0.62 | 9 |
| | **200** | 0.82 | 0.18 | 0.67 | 1.23 | 4.2 | **0.89** | 31 | 2.42 | 0.68 | 5 |
| | 400 | 0.85 | 0.20 | 0.69 | 1.23 | 4.4 | **0.87** | 31 | 2.50 | 0.70 | 9 |
| | 1000 | 0.85 | 0.20 | 0.69 | 1.32 | 4.4 | **0.83** | 30 | 2.53 | 0.72 | 9 |
| seoul_bull | 100 | 0.50 | 0.71 | 0.28 | 1.06 | 3.0 | **0.90** | 44 | 0.05 | 0.00 | 3 |
| | **200** | 0.52 | 0.68 | 0.29 | 1.05 | 3.0 | **0.93** | 44 | 0.07 | 0.00 | 4 |
| | 400 | 0.61 | 0.69 | 0.32 | 1.22 | 3.0 | **0.90** | 44 | 0.08 | 0.00 | 4 |
| | 1000 | 0.61 | 0.64 | 0.34 | 1.22 | 3.0 | **0.90** | 44 | 0.08 | 0.00 | 6 |

**DinoLedge's four rows are one row.** Every rate is identical across a tenfold
change in the sample count; only the residual medians drift by 0.05 px. A
candidate there is offered 16 correspondences of which about 16 are inliers, so
the first few draws are already clean and the rest cannot improve on them.

**The earlier report's explanation is confirmed rather than overturned.** It
said the captures where more iterations helped were the ones where `k = 32`
flooded each candidate with chaff, and that fixing the inlier ratio was the
better lever. The collapse fixes the inlier ratio, and most of the gain
disappears with it: seattle's correspondence recall went 0.36 → 0.49 → 0.52 over
200 → 1000 → 5000 before, with `warp ok` 0.65 → 0.74; it now goes
0.60 → 0.64 over 200 → 1000 with `warp ok` 0.93 → 0.92. seoul's went
0.16 → 0.21 → 0.31 with `warp ok` 0.32 → 0.29 → 0.52; it now goes 0.29 → 0.34
with `warp ok` 0.93 → 0.90.

**What is left is small and two-sided.** Going from 200 to 400 buys
correspondence recall on three captures (seattle +0.03, seoul +0.03,
kerry +0.02) and nothing on the other two, and costs `warp ok` on three
(seattle -0.01, kerry -0.02, seoul -0.03) and nothing on the other two. Every
one of those moves is inside the standard errors of §1. False candidates rise
slightly and monotonically wherever they are non-zero, which is the same
mechanism as before: more draws, more chances at a spurious consensus. Halving
to 100 costs seattle 0.06 of image recall and 0.04 of correspondence recall and
leaves the rest alone, so 200 is on a flat stretch with a soft edge just below
it.

---

## 5. `min_inliers`: still the knee, and now for a different reason

| capture | min_inliers | recall≥3 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 6 | 0.84 | 0.71 | 0.65 | 1.74 | 4.8 | **0.76** | 14 | 3.83 | 0.35 | 563 |
| | **8** | 0.74 | 0.76 | 0.60 | 1.78 | 4.5 | **0.76** | 16 | 2.08 | 0.20 | 559 |
| | 10 | 0.63 | 0.82 | 0.55 | 1.80 | 4.5 | **0.78** | 18 | 1.20 | 0.05 | 574 |
| dino_dog_toy | 6 | 0.83 | 0.54 | 0.59 | 1.53 | 5.4 | **0.76** | 22 | 2.40 | 0.12 | 12 |
| | **8** | 0.69 | 0.67 | 0.52 | 1.56 | 5.1 | **0.76** | 24 | 0.98 | 0.00 | 7 |
| | 10 | 0.59 | 0.74 | 0.47 | 1.53 | 5.1 | **0.76** | 26 | 0.48 | 0.00 | 7 |
| seattle_backyard | 6 | 0.81 | 0.73 | 0.65 | 1.01 | 4.0 | **0.91** | 39 | 0.43 | 0.10 | 5 |
| | **8** | 0.69 | 0.88 | 0.60 | 1.02 | 3.4 | **0.93** | 39 | 0.05 | 0.00 | 6 |
| | 10 | 0.59 | 0.95 | 0.54 | 1.01 | 3.5 | **0.94** | 40 | 0.02 | 0.00 | 6 |
| kerry_park | 6 | 0.87 | 0.12 | 0.73 | 1.23 | 4.5 | **0.88** | 30 | 3.63 | 1.13 | 8 |
| | **8** | 0.82 | 0.18 | 0.67 | 1.23 | 4.2 | **0.89** | 31 | 2.42 | 0.68 | 9 |
| | 10 | 0.80 | 0.21 | 0.63 | 1.23 | 3.9 | **0.88** | 32 | 1.88 | 0.48 | 8 |
| seoul_bull | 6 | 0.68 | 0.49 | 0.34 | 1.08 | 4.0 | **0.86** | 44 | 0.32 | 0.03 | 4 |
| | **8** | 0.52 | 0.68 | 0.29 | 1.05 | 3.0 | **0.93** | 44 | 0.07 | 0.00 | 4 |
| | 10 | 0.51 | 0.71 | 0.29 | 1.05 | 3.0 | **0.93** | 44 | 0.05 | 0.00 | 7 |

**The trade is the same shape as before, at a smaller scale.** Lowering to 6
raises image recall≥3 by 0.05 to 0.15 and correspondence recall by 0.05 to 0.07
on all five, and multiplies false candidates by 1.5 to 4.6 (DinoLedge
2.08 → 3.83, dino_dog_toy 0.98 → 2.40, seattle 0.05 → 0.43, kerry 2.42 → 3.63,
seoul 0.07 → 0.32), with never-covisible candidates appearing on all five where
three of them had none. Raising to 10 spends 0.02 to 0.11 of image recall for
another halving of false candidates.

**What changed is `warp ok`, which no longer moves.** Before the collapse,
6 → 8 took it from 0.65 to 0.86 on seattle and 0.32 to 0.70 on seoul; here it
goes 0.91 → 0.93 and 0.86 → 0.93, and on the three large captures it does not
move at all (0.76 / 0.76 / 0.76 and 0.76 / 0.76 / 0.76 across 6 / 8 / 10). The
collapse already removed the candidates whose warps were wrong; what the
`min_inliers` floor still removes is candidates whose warps are fine but whose
evidence is thin, and that is a precision question rather than a trust one.

**The noise floor the default was chosen against survives, with a smaller
population.** Pooling the never-covisible candidates -- those sharing no point
with the query image anywhere in it -- over the 60 patches at `min_inliers = 6`:

| capture | never-covisible candidates | median inliers | max | share below 8 |
|---|--:|--:|--:|--:|
| DinoLedge | 21 | 8 | 11 | 0.43 |
| dino_dog_toy | 7 | 6 | 7 | 1.00 |
| seattle_backyard | 6 | 6 | 6 | 1.00 |
| kerry_park | 68 | 8.5 | 26 | 0.40 |
| seoul_bull | 2 | 6 | 6 | 1.00 |

The counts themselves are the headline: before the collapse the same figure was
a per-query mean of 0.50 to 2.62 over 60 patches, and it is now a total of
between 2 and 68 candidates. Where enough of them remain to have a shape, the
median is still six, so the floor is still sitting where the noise is. kerry_park
is the exception in both direction and interpretation: its solve tracks 3.6% of
its features, so most of its "never-covisible" candidates are sky and foliage
the solve declined to track rather than errors, and its median of 8.5 is a
statement about that solve.

---

## 6. Defaults

**`one_hit_per_image`: `false` → `true`.** The condition the collapse report
attached to this change is discharged: with the collapse on, none of `k`,
`iterations` or `min_inliers` wants to move, so the default flip stands on its
own and carries the gains that report measured -- seoul_bull's share of
trustworthy warps from 0.54 to 0.93 and its correspondence recall from 0.14 to
0.29, seattle_backyard's from 0.80 to 0.93 and 0.48 to 0.60, DinoLedge unchanged
in every column. It is also the knob with no threshold in it: the hit it keeps
is the one the index already ranked first, and the hits it drops cannot be right,
since a point of the patch's surface appears once per photograph of it.

**`k`: stays 32.** The reason to reconsider it was that it disagreed across
capture sizes, and §3 shows it no longer does. Between 16 and 32 the recall
columns are inside each other's noise on four of five captures and favour 32 on
the fifth, which is DinoLedge, the scale this query exists for; against that, 16
carries a quarter to a third of 32's false candidates. That is a trade a caller
can make -- and a caller who wants it has `min_inliers` as well -- but it is not
a reason to move the default, and 8 is clearly short on the two largest
captures.

**`iterations`: stays 200.** DinoLedge is bit-identical across 100 to 1000, and
on the four small captures the 200 → 400 move is inside the standard error on
every column, positive on correspondence recall and negative on `warp ok`. What
the earlier report predicted has happened: the captures that wanted more
iterations wanted them because crowding had pushed their inlier ratio down, and
the collapse raised it instead. A caller working above fifty features still
needs more, for the reason that report gave.

**`min_inliers`: stays 8.** 6 costs 1.5 to 4.6 times the false candidates for 0.05
to 0.15 of image recall on every capture, and 10 costs recall for a further
halving. The median never-covisible candidate still carries six inliers wherever
there are enough of them to count, so 6 is still the noise floor. The difference
from the earlier report is the justification, not the number: the floor no
longer buys `warp ok`, which is flat across 6 / 8 / 10 on the three
well-populated captures, and buys precision instead.

**`same_image_ratio`: stays 1.0, off.** Nothing here changes the collapse
report's §5 finding on it: against the dedupe it is a small further gain on
precision and false candidates and a small loss on `warp ok` for two captures,
all inside this noise, and 0.9 rather than 0.8 is the tightening to reach for.

### What the earlier reports said that this supersedes

- `2026-09-14-constellation-query-eval.md` §3 and §6.4, "`k = 32` should stay,
  because this query is aimed at large captures and 32 is neutral-to-better
  there, but it is worth knowing that on a small corpus it is actively harmful
  and 16 is better on four of five metrics". The conclusion survives; its caveat
  does not. On a small corpus with the collapse on, `k` = 16 and 32 are inside
  each other's noise, and `k` = 64, which took seoul's `warp ok` to 0.00, now
  leaves it at 0.83.
- The same report's §6.5 on `iterations` stands, including its mechanism. The
  residual gain above 200 on seattle_backyard and seoul_bull is real but four to
  six times smaller than it was, which is what that mechanism predicts.
- Its §3 justification for `min_inliers = 8` -- that 8 "raises `warp ok` or
  leaves it flat everywhere" -- is now "leaves it flat almost everywhere". The
  case for 8 is false candidates.

---

## Appendix: reproducing

```bash
# The grid, one capture. Corpora and workspaces are §1 of
# reports/exp/2026-09-14-constellation-query-eval.md.
pixi run -e test python scripts/kdf_constellation_eval.py \
    --workspace WS --kdf capture.kdf --sfmr WS/sfmr/solve.sfmr \
    --label NAME --patches 60 --sizes 50 --seed 0 --cache-mib 64 \
    --sweep-size 50 --sweep-patches 60 --one-hit-per-image \
    --sweep-grid "k=8,16,32,64;iterations=100,200,400,1000;min_inliers=6,8,10" \
    --out results-NAME.json
```

`--features DIR` is needed for DinoLedge, whose `.kdf` names images by
basename, and its cache budget is 256 MiB. Since `one_hit_per_image` is now the
default, `--one-hit-per-image` is redundant above and
`--no-one-hit-per-image` is what reproduces the pre-change answer. The raw
per-patch results were written to the scratch directory, not to the repository.
