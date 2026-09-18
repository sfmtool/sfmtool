# The two-stage locked warp, decided: refit the affine instead, 2026-09-18

Round two of the patch-constellation warp evaluation
(`crates/sfmtool-core/src/features/kdforest/constellation.rs`, spec
`specs/core/features/kdf-constellation-query.md`). The
[2026-09-17 report](2026-09-17-constellation-progressive-eval.md) recommended a
two-stage query that keeps, per candidate image, the affine of the first stage
that accepted it. This round measures that proposal directly, at power, against
the thing round one never controlled for: the shipped affine is the best
*three-point* RANSAC model with no refinement, so any gain a small constellation
shows could be less noise rather than more locality.

**Answer, in one line.** It is the noise. Ship a **least-squares refit of the
affine over the correspondences the single N=50 query already returns**,
distance-weighted about the patch centre with sigma = half the constellation
radius; do **not** build a staged query. Over eight corpora, 4,800 patches and
13,416 (patch, image) cases, the refit moves the share of found images whose warp
places the patch centre within 3 px by **+0.08 to +0.22**, every interval clear
of zero on every corpus, while the best staged lock moves it by **+0.01 to
+0.08** with intervals that touch zero on two of eight. Head to head and paired,
the staged lock is **worse** than the refit by 0.07 to 0.15 on **all eight**, no
interval touching zero. The refit needs no second stage, no forest work, no new
state and no API change: it runs on `inlier_correspondences`, which the query
already returns, and costs 16 microseconds per candidate against a query that
costs milliseconds -- 0.7% to 3.6% of the query against 9% to 27% for staging.

**What this does to round one.** Its *direction* is confirmed and its
*mechanism* is overturned. A warp fitted on features near the centre really is
better at the centre -- that is why weighting the refit towards the centre beats
refitting flat, on all eight corpora. But the staged lock captures only a
fifth to a third of the available gain, because it buys locality by throwing
away the evidence a wider constellation collected, and it inherits the
three-point model's variance. Round one's headline (`warp ok10` 0.70 → 0.86 on
dino_dog_toy) survives as a much smaller effect on the direct metric
(share within 3 px 0.75 → 0.79, +0.043 [+0.027, +0.059] for its best lock arm)
and is dwarfed by the same capture's refit (0.75 → 0.90, +0.145
[+0.128, +0.161]).

---

## 1. What changed from round one

Round one's four weaknesses, each addressed:

| weakness | round two |
|---|---|
| proxy metric (`warp ok10`, residuals of ground-truth correspondences near the centre) | the caller's own quantity: the affine applied to the **patch-centre pixel**, against where the solve says that point is, with the centre feature **held out of the constellation** |
| denominators of 43 and 24 | 4,800 patches over eight corpora, 13,416 (patch, image) cases, plus 2,800 hand-placed-pixel patches, with no ">= 3 correspondences in the disc" requirement |
| sparse COLMAP-SIFT ground truth on the four small captures | re-bootstrapped and re-solved from `scripts/init_dataset_*.sh` + each workspace's `sfm_solve.sh` with `sfmtool` SIFT; observation and point counts now reproduce the 2026-09-14 report |
| the recommended rule was a composition of two arms, never itself an arm | every rule below is an arm, scored end to end, paired over identical cases |

Two further additions the brief asked for: an independent wide-baseline stills
capture, and subsampled DinoLedge at two strides as a controlled widening of the
baseline.

---

## 2. Protocol

### The metric

`bench/search.rs` takes a candidate's affine, applies it to the searched
observation's own pixel (`apply_affine(&candidate.affine, center)`) and seeds an
observation there. So the measurement is exactly that:

1. Sample a patch centre from a keypoint whose ground-truth track reaches at
   least two **other** images of the corpus (a track of length three).
2. Build the constellation from the nearest `n` keypoints **excluding that
   keypoint**. No three-point model can pass through a feature that is not in
   the constellation, so the centre is held out rather than fitted.
3. For every image the cap query accepted in which the centre's point is
   observed: `centre_err = |A(centre) − (that image's keypoint for the point)|`.

Reported per arm: median, p75, p90, the share within 2, 3 and 5 px, and the
share beyond 10 px, which is the gross-failure rate. Also a normalised error,
`centre_err / (radius × sqrt(|det A|))` -- the error as a fraction of the
constellation disc mapped through the warp -- because the corpora run from
270 px to 3840 px frames.

Secondary, for the linear part (the caller warps the observation's affine shape
with it): where a candidate image holds at least five ground-truth
correspondences among the nearest fifteen features, a least-squares "true" local
affine is fitted from them and the report gives the relative Frobenius
difference of the 2x2 parts and the rotation difference. `disc ok3`, the share
of candidates whose median ground-truth residual over the whole 50-feature disc
is within 3 px, is round one's metric, kept for continuity.

### Corpora

Eight keypoint-centred corpora, 4,800 patches, 13,416 (patch, image) cases:

| corpus | images | image px | descriptors | GT observations | patches | cases | cache |
|---|--:|---|--:|--:|--:|--:|--:|
| DinoLedge | 1,196 | 2160×3840 | 9,702,948 | 4,523,010 | 400 | 1,977 | 1 GiB |
| DinoLedge stride 10 | 120 | 2160×3840 | 972,275 | 453,451 | 400 | 649 | 1 GiB |
| DinoLedge stride 20 | 60 | 2160×3840 | 484,994 | 227,943 | 400 | 290 | 1 GiB |
| Daegu tree stump | 365 | 2160×3840 | 2,990,080 | 534,309 | 400 | 1,036 | 1 GiB |
| dino_dog_toy | 85 | 2040×1536 | 212,500 | 85,857 | 800 | 3,286 | 128 MiB |
| kerry_park | 48 | 480×480 | 83,145 | 2,374 | 800 | 1,963 | 64 MiB |
| seattle_backyard | 26 | 360×640 | 52,000 | 14,426 | 800 | 2,729 | 64 MiB |
| seoul_bull | 17 | 270×480 | 37,167 | 3,750 | 800 | 1,486 | 64 MiB |

**The four repo captures were re-bootstrapped and re-solved** into
`target/constellation-progressive/round2/ws/` with `scripts/init_dataset_*.sh`
and each workspace's own `sfm_solve.sh` (`sfmtool` SIFT). All four solved on the
first attempt and registered every image, and the descriptor counts are now
exactly the 2026-09-14 report's (37,167 / 52,000 / 83,145 / 212,500), as are the
observation and point counts to within a couple of per cent: 85,857 / 18,443
against 85,895 / 18,438 on dino_dog_toy, 14,426 / 3,286 against 14,388 / 3,274 on
seattle_backyard, 3,750 / 1,203 against 3,805 / 1,224 on seoul_bull. kerry_park
came out at 2,374 / 662 against 2,957 / 792, a fifth thinner but the same
capture and the same pipeline. **Round one's ground-truth objection is
discharged.**

**Two corpora are new.** *Daegu tree stump* is an independent wide-baseline
stills capture from `C:\DataSets` -- 365 frames of a 8,009-frame walk kept every
22nd, solved with `sfmtool` SIFT into 534,309 observations over 109,212 points,
already in the tree as `20260718-00-solve-…x22.sfmr`. *DinoLedge stride 10 and
20* are the controlled widening the brief asked for: a `.kdf` over every 10th
and every 20th registered frame of the trusted 1,196-image solve, whose own
tracks still describe the subset. Stride 20 cuts most tracks to a single
surviving observation, so its centres are required to reach one other image
rather than two and its case count is thin (290); it is read as a direction, not
a measurement.

### Arms

The image set is **always the cap query's** -- the images `constellation_query`
at N=50 reports at `min_inliers = 8`. Every arm differs in exactly one thing,
which affine it reports for those images, so every comparison is paired over
identical (patch, image) cases.

| arm | affine |
|---|---|
| `A_cap` | the shipped N=50 three-point RANSAC model |
| `B_lock{S}` | the stage-`S` model if the image passed the bar at `S`, else the cap's |
| `B_lock{S1}_{S2}` | the first of two stages that accepted it, else the cap's |
| `D_guard{S}` | between the stage-`S` and cap models, whichever explains more of the cap's correspondences among the nearest `S` within 8 px; ties to the early one |
| `D_strict{S}` | as `D_guard`, and the early model must also keep at least 0.8 of the cap's agreement over the **whole** disc |
| `C1_ls` | least squares over all the cap's inlier correspondences |
| `C1_ls_iter` | `C1_ls`, then re-select inliers under it at 8 px and refit, up to 3 rounds |
| `C2_ls{S}` | least squares over the cap's inliers among the nearest `S` only (>= 4 of them, else `C1_ls`) |
| `C3_w{sigma}` | least squares over all the cap's inliers, Gaussian weight `exp(−½(d/(sigma·R))²)` in distance `d` from the centre, `R` the disc radius |
| `C4_lock25_ls` | least squares over the **stage-25** inliers |
| `C5_hybrid25` | the stage-25 model selects, among the cap's offered correspondences over the whole disc, those within 8 px; least squares on those |

Stages measured: 15, 20, 25, 30, 35, 50. The bar is 8, easing to
`min(8, max(5, ceil(0.4 n)))` below n=20 as round one recommended, so stage 15
uses 6. Every refit applies the same guards the three-point solve does -- a
reflected model or one whose scale leaves `[1/4, 4]` is refused and the arm
falls back to the cap's affine.

Queries ran at `min_inliers = 4` and every bar was applied offline, which round
one verified is exactly equivalent (60 of 60 queries identical bit for bit).

**Two arms turned out to be definitionally empty and were changed.** The
brief's hybrid -- "cap correspondences among the nearest S within threshold of
the early warp, then least squares" -- is *identical* to `C4_lock25_ls`: a
feature's forest hits do not depend on the constellation, so the cap's offered
correspondences below row 25 are the stage-25 query's, and those of them
agreeing with the stage-25 model within 8 px are precisely that model's inlier
set. `C5_hybrid25` is therefore redefined to let the early warp select over the
**whole** disc, which is a claim it has not already made. And `D_guard` as
specified almost never fires, because the early model was fitted to maximise
agreement on exactly the correspondences the guard counts; `D_strict` is the
form that can actually refuse an early lock.

### Correctness check on the refit arms

The refit arms need the correspondence list the query fitted on, which the API
does not return -- only the inliers it kept. The harness rebuilds it from the
same batch search at the same `k` and leaf budget with the same per-cell
collapse. **In 188,961 of 188,961 cap candidates across every run the rebuilt
list had exactly the count the query reported**, so the refits operate on the
query's own data and not on an approximation of it.

### Power and pairing

Patch centres are drawn without replacement, seed 20260918. All comparisons are
paired over the same (patch, image) cases, and the bootstrap resamples
**patches**, not cases, because cases inside a patch share a constellation and a
centre and are correlated. 2,000 draws; intervals are 95% percentile.

---

## 3. Results: the arms, per corpus

Centre error in pixels, over every (patch, image) case where the centre's point
is observed. `<=3` is the headline column; `>10` is the gross-failure rate;
`norm` is the error as a fraction of the warped disc radius; `disc ok3` is round
one's metric; `lin rel` and `rot` are the linear part against a least-squares
ground-truth local affine.

### DinoLedge, 1,196 images (a video walk — round one's continuity corpus)

| arm | median | p75 | p90 | <=2 | **<=3** | <=5 | >10 | norm med | disc ok3 | lin rel | rot° |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A_cap | 1.58 | 2.79 | 4.46 | 0.61 | **0.77** | 0.93 | 0.010 | 0.0129 | 0.78 | 0.03 | 0.68 |
| B_lock25 | 1.50 | 2.71 | 4.37 | 0.62 | **0.79** | 0.93 | 0.010 | 0.0118 | 0.69 | 0.03 | 0.78 |
| D_strict25 | 1.51 | 2.71 | 4.35 | 0.62 | **0.79** | 0.94 | 0.010 | 0.0120 | 0.71 | 0.03 | 0.76 |
| C1_ls | 1.22 | 2.17 | 3.68 | 0.71 | **0.84** | 0.95 | 0.010 | 0.0097 | 0.94 | 0.02 | 0.43 |
| C2_ls25 | 1.12 | 2.08 | 3.52 | 0.74 | **0.86** | 0.95 | 0.011 | 0.0091 | 0.86 | 0.02 | 0.38 |
| **C3_w0.5** | 1.12 | 2.04 | 3.51 | 0.74 | **0.86** | 0.96 | 0.007 | 0.0090 | 0.93 | 0.02 | 0.40 |
| C3_w0.25 | 1.00 | 1.92 | 3.26 | 0.76 | **0.88** | 0.96 | 0.009 | 0.0080 | 0.82 | 0.01 | 0.33 |
| C4_lock25_ls | 1.15 | 2.17 | 3.86 | 0.72 | **0.84** | 0.94 | 0.009 | 0.0093 | 0.85 | 0.02 | 0.39 |
| C5_hybrid25 | 1.20 | 2.21 | 3.90 | 0.71 | **0.84** | 0.94 | 0.010 | 0.0096 | 0.87 | 0.02 | 0.43 |

### dino_dog_toy, 85 images (round one's strongest capture)

| arm | median | p75 | p90 | <=2 | **<=3** | <=5 | >10 | norm med | disc ok3 | lin rel | rot° |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A_cap | 1.53 | 2.99 | 5.11 | 0.60 | **0.75** | 0.90 | 0.025 | 0.0147 | 0.78 | 0.03 | 1.00 |
| B_lock20 | 1.34 | 2.65 | 4.59 | 0.65 | **0.79** | 0.91 | 0.022 | 0.0130 | 0.71 | 0.04 | 1.10 |
| B_lock25 | 1.35 | 2.66 | 4.64 | 0.65 | **0.78** | 0.91 | 0.022 | 0.0133 | 0.71 | 0.04 | 1.13 |
| D_strict25 | 1.37 | 2.71 | 4.70 | 0.64 | **0.78** | 0.91 | 0.023 | 0.0135 | 0.74 | 0.04 | 1.09 |
| C1_ls | 1.11 | 2.17 | 3.93 | 0.72 | **0.84** | 0.94 | 0.022 | 0.0110 | 0.91 | 0.02 | 0.68 |
| C2_ls25 | 0.96 | 1.87 | 3.44 | 0.77 | **0.88** | 0.94 | 0.021 | 0.0092 | 0.84 | 0.02 | 0.61 |
| **C3_w0.5** | 0.97 | 1.90 | 3.44 | 0.77 | **0.87** | 0.95 | 0.020 | 0.0094 | 0.91 | 0.02 | 0.63 |
| C3_w0.25 | 0.81 | 1.60 | 3.11 | 0.82 | **0.90** | 0.95 | 0.018 | 0.0078 | 0.82 | 0.02 | 0.50 |
| C4_lock25_ls | 1.04 | 2.11 | 3.95 | 0.73 | **0.84** | 0.93 | 0.020 | 0.0098 | 0.83 | 0.02 | 0.63 |

### The other six, headline columns only

| corpus | arm | median px | **<=3 px** | >10 px | disc ok3 | lin rel | rot° |
|---|---|--:|--:|--:|--:|--:|--:|
| Daegu | A_cap | 1.78 | **0.74** | 0.022 | 0.75 | 0.04 | 1.07 |
| | best lock (`D_guard25`) | 1.71 | **0.75** | 0.022 | 0.71 | 0.04 | 1.08 |
| | C1_ls | 1.39 | **0.83** | 0.018 | 0.91 | 0.03 | 0.72 |
| | **C3_w0.5** | 1.25 | **0.85** | 0.016 | 0.91 | 0.03 | 0.68 |
| | C3_w0.25 | 1.24 | **0.86** | 0.017 | 0.80 | 0.03 | 0.61 |
| DinoLedge s10 | A_cap | 1.82 | **0.72** | 0.025 | 0.78 | 0.03 | 0.89 |
| | best lock (`B_lock20`) | 1.76 | **0.76** | 0.022 | 0.66 | 0.04 | 0.99 |
| | C1_ls | 1.42 | **0.81** | 0.017 | 0.93 | 0.02 | 0.61 |
| | **C3_w0.5** | 1.32 | **0.84** | 0.014 | 0.93 | 0.02 | 0.54 |
| | C3_w0.25 | 1.18 | **0.87** | 0.015 | 0.87 | 0.02 | 0.47 |
| DinoLedge s20 | A_cap | 2.33 | **0.60** | 0.031 | 0.76 | 0.03 | 1.16 |
| | best lock (`B_lock20`) | 1.94 | **0.66** | 0.021 | 0.60 | 0.04 | 0.46 |
| | C1_ls | 1.77 | **0.73** | 0.024 | 0.92 | 0.03 | 0.66 |
| | **C3_w0.5** | 1.57 | **0.78** | 0.028 | 0.92 | 0.03 | 0.79 |
| | C3_w0.25 | 1.32 | **0.82** | 0.031 | 0.85 | 0.03 | 0.67 |
| kerry_park | A_cap | 1.19 | **0.83** | 0.012 | 0.87 | 0.11 | 1.74 |
| | best lock (`B_lock35`) | 1.10 | **0.84** | 0.012 | 0.83 | 0.11 | 1.94 |
| | C1_ls | 0.89 | **0.91** | 0.009 | 0.97 | 0.08 | 1.02 |
| | **C3_w0.5** | 0.74 | **0.93** | 0.009 | 0.97 | 0.07 | 0.96 |
| | C3_w0.25 | 0.56 | **0.94** | 0.009 | 0.91 | 0.06 | 0.81 |
| seattle_backyard | A_cap | 1.09 | **0.87** | 0.006 | 0.91 | 0.06 | 1.70 |
| | best lock (`B_lock20`) | 0.95 | **0.88** | 0.007 | 0.85 | 0.07 | 1.67 |
| | C1_ls | 0.78 | **0.93** | 0.003 | 0.98 | 0.04 | 1.02 |
| | **C3_w0.5** | 0.68 | **0.95** | 0.003 | 0.98 | 0.03 | 0.90 |
| | C3_w0.25 | 0.57 | **0.96** | 0.003 | 0.93 | 0.03 | 0.70 |
| seoul_bull | A_cap | 1.20 | **0.85** | 0.013 | 0.88 | 0.12 | 2.13 |
| | best lock (`D_strict30`) | 1.10 | **0.86** | 0.011 | 0.85 | 0.12 | 2.22 |
| | C1_ls | 0.86 | **0.91** | 0.013 | 0.97 | 0.07 | 1.34 |
| | **C3_w0.5** | 0.77 | **0.92** | 0.013 | 0.97 | 0.07 | 1.25 |
| | C3_w0.25 | 0.69 | **0.93** | 0.011 | 0.92 | 0.06 | 0.98 |

Four things this table says, all of them on every corpus.

**Refitting beats locking, everywhere, by three to five times the margin.** The
best staged lock moves `<=3 px` by 0.01 to 0.06; `C3_w0.25` moves it by 0.06 to
0.22. The ordering `A_cap < lock < C4 < C1 < C3_w0.5 < C3_w0.25` holds on all
eight.

**Locking *degrades* the disc metric that round one used to justify it.**
`disc ok3` falls from 0.78 to 0.69 on DinoLedge under `B_lock25`, 0.78 to 0.71
on dino_dog_toy, 0.91 to 0.85 on seattle_backyard -- while the flat refit
`C1_ls` *raises* it to 0.94, 0.91, 0.98. The lock buys centre accuracy by
throwing away evidence; the refit buys it by using more of the evidence better.

**Weighting is a real, tunable knob and `sigma = 0.5 R` is the knee.**
`C3_w0.25` always has the best centre error and always the worst `disc ok3` of
the refits (0.80 to 0.93 against `C1_ls`'s 0.91 to 0.98). `C3_w0.5` gives up 0.01
to 0.04 of the centre share and recovers essentially all of the disc share. That
is the trade a caller should be handed, and the reason to prefer 0.5 as a
default is that the disc warp is what a patch-cluster caller consumes.

**The linear part improves as much as the translation.** The relative Frobenius
error of the 2x2 against a ground-truth local affine halves (0.12 → 0.06 on
seoul_bull, 0.11 → 0.06 on kerry_park, 0.03 → 0.01 on DinoLedge) and the rotation
error roughly halves too (2.13° → 0.98°, 1.74° → 0.81°, 0.68° → 0.33°). Locking
does not move either, and on dino_dog_toy and DinoLedge it makes the rotation
slightly worse. This matters because the caller warps the observation's affine
*shape* with the same matrix.

**Iterating the refit changes nothing.** `C1_ls_iter` differs from `C1_ls` on
`<=3 px` by −0.003 to +0.007 across the eight corpora, inside the interval of
every one; re-selecting inliers at 8 px under the refitted model almost never
changes the inlier set. Do not implement it.

---

## 4. Paired comparisons

Difference in the share of cases within 3 px, paired over identical cases,
bootstrap over patches, 2,000 draws, 95% percentile intervals. Positive is
better than the shipped `A_cap`.

| corpus | cases | best lock | lock − cap | `C1_ls` − cap | `C3_w0.5` − cap | `C3_w0.25` − cap | **lock − best refit** |
|---|--:|---|---|---|---|---|---|
| DinoLedge | 1,977 | `B_lock25` | +0.019 [+0.003, +0.034] | +0.067 [+0.053, +0.082] | +0.086 [+0.071, +0.102] | +0.108 [+0.089, +0.129] | **−0.090 [−0.107, −0.073]** |
| DinoLedge s10 | 649 | `B_lock15` | +0.043 [+0.016, +0.071] | +0.089 [+0.061, +0.118] | +0.122 [+0.093, +0.150] | +0.145 [+0.115, +0.176] | **−0.102 [−0.128, −0.075]** |
| DinoLedge s20 | 290 | `D_strict20` | +0.076 [+0.041, +0.114] | +0.138 [+0.093, +0.186] | +0.183 [+0.134, +0.233] | +0.224 [+0.171, +0.281] | **−0.148 [−0.199, −0.100]** |
| Daegu | 1,036 | `D_guard25` | +0.014 [−0.002, +0.031] | +0.089 [+0.066, +0.113] | +0.115 [+0.089, +0.141] | +0.118 [+0.090, +0.146] | **−0.103 [−0.131, −0.078]** |
| dino_dog_toy | 3,286 | `B_lock20_35` | +0.043 [+0.027, +0.059] | +0.089 [+0.077, +0.101] | +0.121 [+0.107, +0.135] | +0.145 [+0.128, +0.161] | **−0.101 [−0.114, −0.088]** |
| kerry_park | 1,963 | `B_lock35` | +0.017 [+0.001, +0.036] | +0.082 [+0.064, +0.099] | +0.104 [+0.088, +0.122] | +0.114 [+0.096, +0.133] | **−0.097 [−0.114, −0.079]** |
| seattle_backyard | 2,729 | `D_strict20` | +0.015 [+0.006, +0.024] | +0.058 [+0.047, +0.069] | +0.075 [+0.064, +0.087] | +0.084 [+0.072, +0.097] | **−0.069 [−0.080, −0.058]** |
| seoul_bull | 1,486 | `D_strict30` | +0.013 [−0.003, +0.031] | +0.059 [+0.043, +0.074] | +0.076 [+0.061, +0.091] | +0.079 [+0.062, +0.095] | **−0.065 [−0.081, −0.048]** |

**Plainly, per corpus.** Against the shipped affine, the refit is
distinguishable from zero on **all eight** and by a wide margin. The best lock is
distinguishable from zero on six of eight, not on Daegu or seoul_bull, and where
it is, the effect is a fifth to a half of the refit's. Head to head the lock is
distinguishably **worse** than the refit on **all eight**, with no interval
touching zero.

Two caveats that both cut the same way. The "best lock" column is chosen post hoc
as the lock arm with the highest `<=3` share on that corpus, so its interval is
optimistically biased; a pre-registered `B_lock25` would do slightly worse. And
`C3_w0.25` was likewise picked as the best refit, but no selection is needed to
make the point: `C1_ls`, the parameter-free member, gains +0.058 to +0.138 and
beats the best lock on **every** corpus, by +0.048 (DinoLedge) to +0.075
(Daegu).

**The stride series is the controlled test of the baseline-width hypothesis, and
it confirms the hypothesis while refuting the recommendation.** As the baseline
widens (DinoLedge → stride 10 → stride 20) the lock's benefit does grow,
+0.019 → +0.043 → +0.076, exactly as round one predicted. But the refit's grows
faster over the same series, +0.108 → +0.145 → +0.224, so the gap between them
*widens* with baseline rather than closing: −0.090 → −0.102 → −0.148. There is
no corpus width at which the staged lock becomes the right answer.

**Correspondence recall is untouched.** The image set is the cap query's in
every arm and every arm reports the cap's inlier correspondences, so the
round-one cost of a staged lock -- correspondence recall falling from 0.60 to
0.33 because a frozen early warp comes with a frozen early inlier set -- is not
paid by any arm here. It remains a cost of the round-one proposal that this
round's recommendation avoids by construction.

---

## 5. Harm

| corpus | best lock | worse than cap by >5 px | gross (lock >10 px, cap <=3 px) | worse than the refit by >5 px | gross vs refit | list grew 25→50 | early-only/patch at S=25 (GT-true) |
|---|---|--:|--:|--:|--:|---|---|
| DinoLedge | `B_lock25` | 3 (0.15%) | 0 | 19 (0.96%) | 1 | 7,096/7,100 | 0.000 (0) |
| DinoLedge s10 | `B_lock15` | 1 (0.15%) | 0 | 10 (1.54%) | 1 | 1,914/1,914 | 0.003 (0) |
| DinoLedge s20 | `D_strict20` | 0 | 0 | 6 (2.07%) | 0 | 730/730 | 0.003 (0) |
| Daegu | `D_guard25` | 3 (0.29%) | 0 | 22 (2.12%) | 2 | 2,429/2,429 | 0.000 (0) |
| dino_dog_toy | `B_lock20_35` | 42 (1.28%) | 4 | 75 (2.28%) | 6 | 12,583/12,583 | 0.003 (1) |
| kerry_park | `B_lock35` | 29 (1.48%) | 4 | 53 (2.70%) | 4 | 6,714/6,714 | 0.011 (5) |
| seattle_backyard | `D_strict20` | 0 | 0 | 23 (0.84%) | 0 | 5,398/5,398 | 0.011 (9) |
| seoul_bull | `D_strict30` | 7 (0.47%) | 0 | 17 (1.14%) | 0 | 2,102/2,102 | 0.022 (17) |

**The lock's harm against the shipped affine is small.** Nowhere above 1.5% of
cases worse by more than 5 px, and gross failures -- the early lock beyond 10 px
where the cap is within 3 px -- occur only on dino_dog_toy (4) and kerry_park
(4), eight cases in 13,416. So the objection to the round-one proposal is not
that it is dangerous; it is that it barely helps. Its harm against the *refit*
is two to three times larger, 0.8% to 2.7%, which is simply the flip side of the
refit being better.

**What the eight gross cases look like.** Every one is a candidate whose early
consensus was thin in *absolute* terms and stayed thin as a share, with the cap
finding a much larger one. Their (inliers, correspondences) at the locking stage
and at the cap:

| corpus | lock err px | cap err px | at lock stage | at cap |
|---|--:|--:|---|---|
| dino_dog_toy | 87.4 | 0.00 | 7 / 13 at n=20 | 8 / 24 |
| dino_dog_toy | 26.0 | 1.12 | 4 / 10 at n=20 | 15 / 27 |
| dino_dog_toy | 22.2 | 1.58 | 8 / 15 at n=20 | 21 / 36 |
| dino_dog_toy | 10.9 | 2.50 | 5 / 10 at n=20 | 12 / 26 |
| kerry_park | 17.0 | 1.85 | 9 / 25 at n=35 | 9 / 37 |
| kerry_park | 15.5 | 0.42 | 12 / 27 at n=35 | 15 / 34 |
| kerry_park | 10.7 | 1.33 | 8 / 15 at n=35 | 11 / 20 |
| kerry_park | 10.1 | 1.07 | 12 / 27 at n=35 | 15 / 37 |

Four to twelve inliers is a three-point fit resting on very little, and the cap
overturns it. They are exactly what a guard should catch.

**Neither guard catches them.** `D_guard` almost never fires at all, for a
structural reason: the early model was fitted to maximise agreement on precisely
the correspondences the guard counts, so it wins its own test by construction --
`D_guard{S}` is within 0.002 of `B_lock{S}` on `<=3 px` on every corpus.
`D_strict`, which additionally requires the early model to retain 0.8 of the
cap's agreement over the *whole* disc, does fire and is the best lock arm on four
of eight corpora, but it moves `<=3 px` by at most +0.005 over the plain lock and
leaves the gross cases in place on dino_dog_toy and kerry_park. **A guard is not
the missing piece; the lock is the wrong mechanism.**

**Every candidate's correspondence list grows between stage 25 and the cap.**
38,966 of 38,970 across the eight corpora. The hoped-for optimisation -- skip
the re-fit for candidates whose evidence did not change -- does not exist: a
staged implementation must re-fit every candidate at every stage. This is a
cost fact, and it is why §7's staging overhead is what it is.

**Images an early stage accepts that the cap rejects are negligible**, which
confirms round one at 6× the power: 0.000 to 0.022 per patch at S=25, and the
ground truth corroborates a minority of even those. The right policy is the one
already in force -- report the cap's image set -- and no union rule is needed.
The count rises at S=15 and S=35 on the small captures (up to 0.079 per patch on
kerry_park) because the bar eases at 15 and because a 35-feature constellation
can lose a marginal candidate to a changed consensus at 50, but it never reaches
a tenth of an image per patch.

---

## 6. Hand-placed pixel centres

The bench's real centre is a pixel someone clicked, not a keypoint. Second patch
set: centres drawn uniformly over the frame, rejected unless 55 keypoints lie
within a quarter of the frame diagonal. There is no track at such a point, so
the proxy is its **five nearest keypoints, all five held out of the
constellation**, and the error is the mean residual on those five in images where
at least two of them are observed. 600 patches per small capture, 400 on Daegu.

| corpus | cases | `A_cap` <=3 | best lock − cap | `C1_ls` − cap | `C3_w0.5` − cap | `C3_w0.25` − cap | lock − best refit |
|---|--:|--:|---|---|---|---|---|
| dino_dog_toy | 1,036 | 0.66 | +0.023 [−0.001, +0.046] | +0.089 [+0.067, +0.111] | +0.112 [+0.088, +0.136] | +0.125 [+0.097, +0.153] | **−0.101 [−0.129, −0.076]** |
| seattle_backyard | 769 | 0.86 | +0.003 [−0.019, +0.024] | +0.064 [+0.045, +0.084] | +0.078 [+0.058, +0.100] | +0.069 [+0.046, +0.092] | **−0.075 [−0.098, −0.053]** |
| Daegu | 300 | 0.63 | +0.027 [+0.000, +0.058] | +0.123 [+0.084, +0.165] | +0.150 [+0.112, +0.190] | +0.170 [+0.129, +0.216] | **−0.143 [−0.185, −0.100]** |
| seoul_bull | 120 | 0.72 | +0.075 [+0.000, +0.156] | +0.100 [+0.034, +0.171] | +0.100 [+0.034, +0.171] | +0.125 [+0.065, +0.189] | −0.058 [−0.131, +0.009] |
| kerry_park | 21 | 0.62 | +0.095 [−0.118, +0.292] | +0.143 [+0.000, +0.273] | +0.143 [+0.000, +0.273] | +0.143 [+0.000, +0.273] | −0.095 [−0.353, +0.091] |

The same answer, from a different metric on a different patch set. The refit is
distinguishable from zero on four of five (kerry_park has 21 cases and says
nothing -- a fisheye frame is mostly black corners, so a uniformly placed pixel
rarely lands near five tracked keypoints). The lock is distinguishable on none of
the five at the 95% level. Head to head the lock is worse on the three corpora
with enough cases to tell.

Absolute errors are larger than on keypoint centres everywhere -- `A_cap`'s
`<=3 px` share is 0.66 against 0.75 on dino_dog_toy, 0.63 against 0.74 on Daegu
-- which is what one expects when the point being warped is further from the
features the warp was fitted to. That is also why the refit's *advantage* is if
anything slightly larger here (+0.170 on Daegu against +0.118 on keypoint
centres): there is more error available to remove.

---

## 7. Cost

Measured, not modelled. Every query and every bare forest search was timed in a
shuffled order with each measurement warm for its own prefix (the
second of two consecutive runs), which is the discipline round one had to adopt
after its ascending sweep turned out to measure cache state. `fit(n)` is
`query(n) − search(n)`: everything the query does besides the forest search,
including origin and geometry resolution and RANSAC. DinoLedge and the other
large corpora run at a 1 GiB cache, which round one established is above the
knee. 60 to 80 patches per corpus.

| corpus | query 25 | query 50 | search 25 | search 50 | fit(25) | fit(50) | **staged 25+50** | **staged overhead** | refit µs (p90) | cands/patch | **refit overhead** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 7.95 | 15.99 | 6.53 | 12.61 | 1.42 | 3.38 | 17.41 | **+9%** | 16.2 (18.3) | 19.6 | **+2.0%** |
| DinoLedge s10 | 6.40 | 11.42 | 3.55 | 7.22 | 2.85 | 4.19 | 14.27 | **+25%** | 15.6 (17.5) | 5.3 | **+0.7%** |
| Daegu | 6.88 | 14.94 | 4.24 | 8.84 | 2.63 | 6.09 | 17.57 | **+18%** | 15.7 (17.3) | 6.7 | **+0.7%** |
| dino_dog_toy | 4.30 | 7.30 | 2.35 | 4.73 | 1.94 | 2.57 | 9.24 | **+27%** | 16.4 (17.4) | 16.0 | **+3.6%** |
| kerry_park | 3.88 | 6.70 | 2.11 | 4.27 | 1.77 | 2.43 | 8.47 | **+26%** | 16.4 (18.5) | 8.8 | **+2.1%** |
| seattle_backyard | 2.66 | 4.86 | 1.57 | 3.12 | 1.09 | 1.74 | 5.95 | **+22%** | 15.8 (18.0) | 7.0 | **+2.3%** |
| seoul_bull | 2.41 | 4.53 | 1.84 | 3.53 | 0.57 | 1.00 | 5.10 | **+13%** | 16.0 (19.1) | 2.6 | **+0.9%** |

Milliseconds unless stated. "Staged 25+50" is what a two-stage implementation
would do -- search all fifty features once, fit at 25, fit again at 50 -- and its
overhead is measured against the single N=50 query.

**Staging costs 9% to 27%. The refit costs 0.7% to 3.6%.** And the refit figure
is an upper bound twice over: it is numpy's `lstsq` on a handful of rows, timed
per candidate at 15.6 to 16.4 µs median with a p90 under 19 µs, where the same
6x6 normal-equation solve in Rust is tens of nanoseconds of arithmetic plus a
pass over the inliers; and it counts one refit for every accepted candidate,
which is the whole cost, not an increment on top of another stage.

**No candidate can be skipped.** §5's 38,966-of-38,970 result means a staged
implementation must re-fit every candidate at every stage -- there is no "its
evidence did not change" shortcut -- so the overhead above is the real one.

The only thing staging would save is nothing at all on the search side: search is
linear in batch size with a negligible intercept (round one measured the
intercept at −0.07 to +0.01 ms against 0.06 to 0.15 ms per feature), so splitting
one 50-feature search into 25+25 is free but also gains nothing. **Every
millisecond of the staging overhead is the extra RANSAC pass, and it buys a fifth
of what 16 µs of least squares buys.**

---

## 8. Recommendation

**Ship the refit. Do not build the staged query.**

- **Which arm.** `C3_w0.5`: after RANSAC picks the consensus, refit the affine by
  weighted least squares over that candidate's inlier correspondences, with
  Gaussian weight `exp(−½ (d / (0.5 R))²)` in the distance `d` of each
  correspondence's source point from the patch centre, `R` the constellation
  radius. `C1_ls` -- the same thing unweighted -- is the parameter-free fallback
  and is already most of the win; it beats every staged arm on every corpus.
- **Why weighted and why 0.5.** `C3_w0.25` is the best centre warp on all eight
  corpora but is the worst refit on the disc metric (`disc ok3` 0.80 to 0.93
  against `C1_ls`'s 0.91 to 0.98), because it discards the rim. `C3_w0.5` gives
  back 0.01 to 0.04 of the centre share and recovers essentially all of the disc
  share. A caller that only ever warps the centre should use 0.25; the default
  should serve both, so 0.5.
- **No stage. No `S`.** There is no second query, so there is no start size, no
  increment, no schedule, no acceptance-bar-by-stage question. The bar stays at
  `min_inliers = 8` for the single query.
- **No guard**, because there is nothing to guard: no early model is kept.
  (`D_guard` was inert by construction and `D_strict` recovered at most +0.005.)
- **No iteration.** `C1_ls_iter` is within 0.007 of `C1_ls` everywhere.
- **Keep the existing guards on the refitted model.** A refit whose 2x2 has a
  non-positive determinant or a geometric-mean scale outside `[1/max_scale,
  max_scale]` is refused and the three-point model is reported instead; that is
  what every refit arm here does, and it fires rarely enough not to show in the
  numbers.

**Measured gain, per corpus**, as the change in the share of found images whose
affine places the patch centre within 3 px, paired, 95% CI:

| corpus | `A_cap` | `C3_w0.5` | gain [CI] | distinguishable from zero? |
|---|--:|--:|---|---|
| DinoLedge | 0.77 | 0.86 | +0.086 [+0.071, +0.102] | yes |
| DinoLedge s10 | 0.72 | 0.84 | +0.122 [+0.093, +0.150] | yes |
| DinoLedge s20 | 0.60 | 0.78 | +0.183 [+0.134, +0.233] | yes |
| Daegu | 0.74 | 0.85 | +0.115 [+0.089, +0.141] | yes |
| dino_dog_toy | 0.75 | 0.87 | +0.121 [+0.107, +0.135] | yes |
| kerry_park | 0.83 | 0.93 | +0.104 [+0.088, +0.122] | yes |
| seattle_backyard | 0.87 | 0.95 | +0.075 [+0.064, +0.087] | yes |
| seoul_bull | 0.85 | 0.92 | +0.076 [+0.061, +0.091] | yes |

Median centre error falls by 0.3 to 0.8 px on every corpus; the linear part's
relative error roughly halves; `disc ok3`, round one's metric, rises on every
corpus (0.78 → 0.93 on DinoLedge, 0.91 → 0.98 on seattle_backyard, 0.87 → 0.97 on
kerry_park). There is no metric in this report on which the refit is worse than
the shipped affine.

**Cost**: 0.7% to 3.6% of the query, numpy upper bound, against 9% to 27% for
staging.

**Where it helps and where it does not.** Everywhere, and more where the baseline
is wider. The controlled series -- full DinoLedge, every 10th frame, every 20th
-- gives +0.086 → +0.122 → +0.183 for the refit against +0.019 → +0.043 → +0.076
for the best lock. Round one's "this is a wide-baseline gain, so it is
capture-shaped" caveat applies to the *lock*; the refit helps a video walk as
much as it helps wide-baseline stills in absolute terms, and more in both as the
baseline widens.

**Round one, explicitly.** Confirmed: an affine fitted or weighted towards the
patch centre is better at the patch centre, which is why `C3_w0.25` beats
`C1_ls`. Weakened: its headline effect is a third the size on the direct metric
(dino_dog_toy 0.75 → 0.78 rather than `warp ok10` 0.70 → 0.86), because the proxy
metric conditioned on candidates having three ground-truth correspondences within
the nearest ten features, which selects for exactly the patches where locality
helps most. Overturned: its recommendation. The staged lock is not the cheapest
way to get that locality, it is not even a good way, and it degrades the disc
warp and the correspondence set while doing it.

---

## 9. What did not work, and what was skipped

- **Two arms the brief specified turned out to be definitionally empty**, and
  were replaced rather than reported as written (§2). The hybrid "select the
  cap's correspondences below row S that agree with the stage-S warp" is
  *identical* to refitting on the stage-S inlier set, because a feature's forest
  hits do not depend on the constellation. And `D_guard` as specified cannot
  refuse an early lock, because the early model was fitted to maximise agreement
  on precisely the correspondences the guard counts. Both facts are findings, but
  neither was the intended measurement.
- **`D_strict`, the guard that does bite, does not fix the harm cases.** It is
  the best lock arm on four of eight corpora and still leaves all eight gross
  failures in place. No guard was found that does fix them; the search stopped
  there because the lock lost to the refit by a wide margin regardless.
- **DinoLedge stride 20 is thin.** 290 cases, and its centres had to be allowed
  a track reaching only one other image of the subset, because a walk's tracks
  do not survive a 20× decimation. It is read as a direction and its intervals
  are wide.
- **kerry_park's pixel-centre set has 21 cases** and says nothing. A fisheye
  frame is mostly black corners, so a uniformly placed pixel rarely has five
  tracked keypoints near it.
- **kerry_park's re-solve is a fifth thinner than the 2026-09-14 report's**
  (2,374 observations over 662 points against 2,957 / 792). Same pipeline, same
  script; not chased.
- **The refit timings are numpy**, and are stated as an upper bound. No Rust was
  written, so the real cost of the recommendation is not measured, only bounded.
- **Correspondence recall and image recall were not re-measured**, because every
  arm reports the cap query's image set and the cap's inlier correspondences, so
  neither can differ between arms. Round one's numbers stand.
- **No parameter other than the constellation size and the acceptance bar was
  varied.** `k`, `threshold_px`, `iterations`, `max_scale` and
  `one_hit_per_image` sat at their shipped defaults in every arm of every run.
- **Nothing was implemented, changed or committed.** No Rust was touched, no
  default moved, no PR opened.

---

## Appendix: reproducing

```bash
S=scripts/kdf_constellation_progressive_eval.py
R=target/constellation-progressive/round2

# Fresh workspaces (test-data copied beside the init scripts so they resolve).
bash init_dataset_seoul_bull.sh   # …and the other three, then each sfm_solve.sh

pixi run -e test python $S build-kdf --workspace WS --features WS/images/features/sift-* \
    --out $R/NAME.kdf
# a wider-baseline corpus out of a solved walk:
pixi run -e test python $S build-kdf --workspace WS --features DIR --sfmr SOLVE.sfmr \
    --stride 10 --out $R/NAME-s10.kdf

pixi run -e test python $S measure2 --workspace WS --kdf $R/NAME.kdf --sfmr SOLVE.sfmr \
    --label NAME --patches 800 --cache-mib 64 --time-stages --out $R/NAME.json
pixi run -e test python $S cost2 --out $R/NAME.json --patches 80
pixi run -e test python $S analyze2 $R/*.json --out $R/analysis2.json
```

`--centres pixel` selects the §6 patch set; `--min-centre-obs 1` is needed for a
subsampled corpus. Raw per-patch tables, the two analysis JSONs and the fresh
workspaces are under `target/constellation-progressive/round2/`, not in the
repository.
