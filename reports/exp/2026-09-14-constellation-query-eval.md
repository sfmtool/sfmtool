# How big a patch constellation has to be, and whether the defaults suit it, 2026-09-14

Measurement of the patch constellation query
(`crates/sfmtool-core/src/features/kdforest/constellation.rs`, spec
`specs/core/features/kdf-constellation-query.md`) against the tracks of a
reconstruction of the same capture, on five datasets. The question asked was
whether the query needs tens or hundreds of features to find the right images
with a warp worth trusting, and whether `k=32`, `threshold_px=8`,
`min_inliers=6`, `iterations=200` are right at that size.

**Answer, in one line.** Tens. Around fifty features, about the radius that holds
fifty; recall is still climbing above that but the warp stops being usable, and
past two hundred the warp is wrong more often than right on every dataset
measured. Of the four knobs asked about, one is clearly mis-set
(`min_inliers=6` is exactly the noise floor), one is a compromise that is wrong
at both ends of the resolution range (`threshold_px=8`), and two are fine
(`k=32`, `iterations=200`). The knob that actually limits the query is a fifth
one, `max_leaf_checks=128`, which throws away a fifth to a third of the
available correspondences before RANSAC ever sees them. One defect is described
in §4: the degeneracy test refuses a warp that collapses the patch but accepts
one that inflates it thirtyfold or mirrors it, and mirrored warps are 2 to 30%
of the candidates reported on the four small datasets, essentially none of them
right.

Tooling added: [`scripts/kdf_constellation_eval.py`](../../scripts/kdf_constellation_eval.py),
which samples patches, sweeps constellation size and the parameters, and scores
every query against a `sift_files` `.sfmr`. Nothing in the implementation was
changed while measuring; the Addendum records what changed afterwards and
re-measures seoul_bull.

---

## 1. Protocol

**Corpora.** One `.kdf` per dataset, built from that workspace's `.sift` files
with SIFT sources, four trees, leaf size 16, 2 KiB descriptor blocks, 1 MiB
chunks, seed 0, the same shape `scripts/kdf_patch_localize.py` builds. All
queries go through `LazyKdForest`, so they read the file rather than a resident
copy. Cache budget was 256 MiB for DinoLedge and 64 MiB for the four small
datasets (`cache_bytes` = `max_in_flight_bytes` = the budget,
`max_chunk_bytes` = 4 MiB).

| dataset | images | image px | `.sift` features | per image | GT observations | share of features tracked | GT points | `.kdf` |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| DinoLedge | 1,196 | 2160×3840 | 9,702,948 | 8,112 | 4,523,010 | 46.6% | 792,873 | 1,494 MB |
| dino_dog_toy | 85 | 2040×1536 | 212,500 | 2,500 | 85,895 | 40.4% | 18,438 | 30 MB |
| seattle_backyard | 26 | 360×640 | 52,000 | 2,000 | 14,388 | 27.7% | 3,274 | 8 MB |
| kerry_park | 48 | 480×480 | 83,145 | 1,732 | 2,957 | 3.6% | 792 | 12 MB |
| seoul_bull | 17 | 270×480 | 37,167 | 2,186 | 3,805 | 10.2% | 1,224 | 5 MB |

DinoLedge uses the checked-in solve `20260707-00-solve-DinoLedge_1-1196.sfmr`
(not the `-embedded` sibling, whose observations carry no `.sift` feature
index). The four repo datasets were bootstrapped with their
`scripts/init_dataset_*.sh` and solved with the `sfm_solve.sh` each one writes;
all four solved on the first attempt and registered every image
(17 / 85 / 26 / 48), so nothing was dropped for a failed solve.

**Patches.** 60 per dataset, seeded (`numpy` PCG64, seed 0): a random registered
image, then a random keypoint of that image as the centre, so patches land on
texture. The same 60 centres are reused at every constellation size and at every
sweep point, so each comparison is on a fixed observation set. For a target size
N the radius is the distance to the N-th nearest keypoint, which makes the
constellation exactly N features wherever the image has enough of them (the
"features" column confirms it did at every size on every dataset).

**Ground truth per patch.** From the `.sfmr` tracks: for each constellation
feature, the point it observes, and then every other image's observation of that
point. That gives the GT correspondences `(constellation row, other image, other
feature)` and, by grouping, the GT covisible images at a floor of one shared
correspondence and of three.

**What is scored.**

- `recall≥3` / `recall≥1` / `prec≥3`: reported images against the GT covisible
  set at each floor.
- `corr recall`: the share of GT correspondences that come back as an inlier
  correspondence, matched on the exact triple.
- `res med` / `res p90`: the pixel residual of the GT correspondences under the
  affine the query fitted for their image, pooled over the found images of a
  patch, then the median over patches.
- `warp ok`: of the found images carrying at least three GT correspondences, the
  share whose median residual is within 3 px. This is the direct answer to "is
  the warp trustworthy", and it is the metric the recommendation turns on.
- `false`: images reported that share no GT correspondence with this
  constellation. `never-covis`: images reported that share no point with the
  query image *anywhere in it*, a far weaker claim that a sparse solve cannot
  inflate (see §5).
- `ms`: wall time of the `constellation_query` call, median over patches.

**Both entry points agree.** At one size per dataset, `constellation_at_pixel` was
run beside `constellation_query` on the same patch: the candidate lists and
inlier counts were identical in 15 of 15 checks across the five datasets. The
constellation membership itself matched in 9 of 15; every mismatch was a single
keypoint sitting at *exactly* the radius, kept by the harness's float32 `hypot`
comparison and dropped by the query's float64 squared-distance one. That is a
rounding artefact of the harness, not a defect, and it never changed the answer.

---

## 2. Size sweep, one table per dataset

### DinoLedge (1,196 images, 9,702,948 descriptors, 60 patches, 256 MiB cache)

| N | features | radius px | GT imgs ≥3 | found | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 10 | 52 | 2.5 | 1.0 | 0.40 | 0.17 | 0.74 | 0.23 | 0.24 | 2.7 | **0.96** | 7 | 0.1 | 0.00 | 50 |
| 25 | 25 | 87 | 7.0 | 5.0 | 0.61 | 0.36 | 0.76 | 0.42 | 1.35 | 4.1 | **0.86** | 10 | 0.9 | 0.05 | 125 |
| 50 | 50 | 128 | 11.5 | 9.0 | 0.73 | 0.47 | 0.77 | 0.51 | 1.55 | 4.6 | **0.76** | 13 | 2.0 | 0.25 | 259 |
| 100 | 100 | 180 | 17.0 | 16.5 | 0.76 | 0.58 | 0.74 | 0.57 | 2.31 | 5.9 | **0.68** | 18 | 3.9 | 0.80 | 460 |
| 200 | 200 | 260 | 21.5 | 25.0 | 0.85 | 0.66 | 0.71 | 0.60 | 2.68 | 7.4 | **0.54** | 23 | 6.5 | 2.42 | 827 |
| 400 | 400 | 375 | 29.5 | 40.5 | 0.88 | 0.73 | 0.65 | 0.60 | 3.12 | 8.6 | **0.38** | 30 | 11.8 | 5.48 | 1,705 |
| 800 | 800 | 539 | 35.0 | 59.0 | 0.90 | 0.76 | 0.59 | 0.58 | 3.57 | 13.1 | **0.23** | 44 | 17.7 | 10.45 | 3,359 |

### dino_dog_toy (85 images, 212,500 descriptors, 60 patches, 64 MiB cache)

| N | features | radius px | GT imgs ≥3 | found | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 10 | 42 | 0.5 | 1.0 | 0.35 | 0.12 | 0.58 | 0.19 | 0.08 | 2.7 | **0.93** | 10 | 0.3 | 0.00 | 3 |
| 25 | 25 | 78 | 4.5 | 3.0 | 0.54 | 0.30 | 0.52 | 0.35 | 1.07 | 4.3 | **0.88** | 17 | 1.4 | 0.18 | 3 |
| 50 | 50 | 115 | 7.0 | 10.0 | 0.73 | 0.47 | 0.54 | 0.44 | 1.59 | 5.7 | **0.75** | 26 | 3.1 | 0.45 | 4 |
| 100 | 100 | 172 | 13.0 | 17.0 | 0.73 | 0.55 | 0.54 | 0.45 | 2.48 | 15.8 | **0.58** | 46 | 5.7 | 0.98 | 6 |
| 200 | 200 | 242 | 19.0 | 27.0 | 0.72 | 0.60 | 0.51 | 0.43 | 4.28 | 38.2 | **0.33** | 81 | 9.1 | 2.18 | 11 |
| 400 | 400 | 357 | 27.0 | 38.0 | 0.75 | 0.65 | 0.54 | 0.39 | 6.10 | 106.2 | **0.18** | 154 | 12.4 | 3.75 | 19 |
| 800 | 800 | 549 | 41.0 | 50.0 | 0.76 | 0.69 | 0.61 | 0.31 | 11.11 | 153.0 | **0.06** | 294 | 13.2 | 5.43 | 35 |

### seattle_backyard (26 images, 52,000 descriptors, 60 patches, 64 MiB cache)

| N | features | radius px | GT imgs ≥3 | found | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 10 | 16 | 0.0 | 0.0 | 0.28 | 0.14 | 0.52 | 0.16 | 0.22 | 2.4 | **0.90** | 16 | 0.2 | 0.07 | 1 |
| 25 | 25 | 28 | 3.0 | 2.0 | 0.52 | 0.37 | 0.59 | 0.37 | 0.91 | 3.8 | **0.79** | 35 | 1.0 | 0.28 | 1 |
| 50 | 50 | 39 | 4.0 | 4.5 | 0.61 | 0.47 | 0.52 | 0.37 | 1.64 | 18.0 | **0.65** | 67 | 1.9 | 0.47 | 2 |
| 100 | 100 | 56 | 6.5 | 9.5 | 0.67 | 0.56 | 0.50 | 0.39 | 2.93 | 123.3 | **0.47** | 125 | 3.4 | 0.95 | 4 |
| 200 | 200 | 82 | 9.0 | 15.0 | 0.80 | 0.75 | 0.50 | 0.36 | 7.34 | 160.5 | **0.28** | 242 | 6.0 | 1.87 | 7 |
| 400 | 400 | 126 | 13.0 | 23.0 | 0.97 | 0.95 | 0.54 | 0.35 | 11.39 | 182.9 | **0.11** | 472 | 8.2 | 3.53 | 12 |
| 800 | 800 | 200 | 17.5 | 25.0 | 1.00 | 1.00 | 0.65 | 0.31 | 15.91 | 143.2 | **0.04** | 934 | 6.5 | 4.38 | 23 |

### kerry_park (48 fisheye images, 83,145 descriptors, 60 patches, 64 MiB cache)

| N | features | radius px | GT imgs ≥3 | found | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 10 | 13 | 0.0 | 0.0 | 0.51 | 0.34 | 0.14 | 0.30 | 0.23 | 0.9 | **0.83** | 12 | 0.6 | 0.18 | 1 |
| 25 | 25 | 21 | 0.0 | 2.0 | 0.85 | 0.62 | 0.12 | 0.55 | 0.80 | 2.1 | **0.88** | 23 | 2.6 | 1.02 | 2 |
| 50 | 50 | 32 | 0.0 | 6.0 | 0.87 | 0.70 | 0.09 | 0.59 | 1.63 | 5.0 | **0.89** | 40 | 5.3 | 2.55 | 3 |
| 100 | 100 | 50 | 0.0 | 12.0 | 0.88 | 0.72 | 0.10 | 0.55 | 1.69 | 8.4 | **0.65** | 70 | 10.5 | 5.58 | 5 |
| 200 | 200 | 80 | 0.0 | 26.0 | 0.96 | 0.86 | 0.10 | 0.47 | 4.99 | 50.0 | **0.37** | 132 | 20.8 | 12.1 | 9 |
| 400 | 400 | 124 | 3.0 | 40.0 | 0.98 | 0.95 | 0.14 | 0.40 | 8.46 | 82.2 | **0.18** | 256 | 30.4 | 20.0 | 15 |
| 800 | 800 | 201 | 9.0 | 47.0 | 1.00 | 1.00 | 0.22 | 0.31 | 15.54 | 118.7 | **0.07** | 504 | 30.2 | 24.3 | 28 |

### seoul_bull (17 images, 37,167 descriptors, 60 patches, 64 MiB cache)

| N | features | radius px | GT imgs ≥3 | found | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 10 | 11 | 0.0 | 0.0 | 0.08 | 0.04 | 0.05 | 0.02 | 36.62 | 236.2 | **0.00** | 21 | 0.1 | 0.02 | 1 |
| 25 | 25 | 20 | 0.0 | 1.0 | 0.44 | 0.23 | 0.25 | 0.12 | 5.65 | 13.7 | **0.44** | 50 | 0.6 | 0.17 | 1 |
| 50 | 50 | 28 | 0.0 | 3.0 | 0.60 | 0.43 | 0.19 | 0.17 | 20.19 | 157.5 | **0.33** | 95 | 2.3 | 0.58 | 2 |
| 100 | 100 | 42 | 2.0 | 8.5 | 0.81 | 0.71 | 0.18 | 0.14 | 61.85 | 203.3 | **0.14** | 187 | 5.6 | 1.50 | 3 |
| 200 | 200 | 61 | 3.0 | 15.0 | 0.99 | 0.98 | 0.20 | 0.11 | 73.86 | 213.8 | **0.06** | 366 | 10.1 | 3.25 | 6 |
| 400 | 400 | 93 | 5.0 | 16.0 | 1.00 | 1.00 | 0.28 | 0.12 | 64.98 | 205.4 | **0.02** | 727 | 9.3 | 3.57 | 11 |
| 800 | 800 | 147 | 6.0 | 16.0 | 1.00 | 1.00 | 0.39 | 0.10 | 55.28 | 181.0 | **0.01** | 1,453 | 6.8 | 3.57 | 22 |

### What the five tables say together

Two curves move in opposite directions.

**Recall rises with N and never stops rising.** At N=800 the query names every
covisible image on three of the five datasets. That is not surprising: the
constellation is by then a large fraction of the frame, and the question has
degenerated into whole-image matching.

**Warp trustworthiness falls with N, monotonically, on all five.** `warp ok`
goes 0.96 / 0.86 / 0.76 / 0.68 / 0.54 / 0.38 / 0.23 on DinoLedge and the same
shape everywhere else. By N=200 the majority of the warps the query returns put
the ground truth's own correspondences more than 3 px from where they are; by
N=800 more than three quarters do. The reason is in the model rather than in the
query: an affine is the first-order approximation of the homography about the
centre of the patch, and the term it drops grows with the patch. The residual
90th percentile tracks it exactly, from 2.7 px at N=10 to 13 px (DinoLedge, a
video walk with small inter-frame parallax) or 118 to 213 px (the wide-baseline
stills) at N=800.

The product of the two, "found the image *and* can be trusted about where",
peaks at N=50 on four datasets and at N=25 on the fifth:

| N | 10 | 25 | **50** | 100 | 200 | 400 | 800 |
|---|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 0.38 | 0.52 | **0.55** | 0.52 | 0.46 | 0.33 | 0.21 |
| dino_dog_toy | 0.33 | 0.48 | **0.55** | 0.42 | 0.24 | 0.14 | 0.05 |
| seattle_backyard | 0.25 | **0.41** | 0.40 | 0.31 | 0.22 | 0.11 | 0.04 |
| kerry_park | 0.42 | 0.75 | **0.77** | 0.57 | 0.36 | 0.18 | 0.07 |
| seoul_bull | 0.00 | 0.19 | **0.20** | 0.11 | 0.06 | 0.02 | 0.01 |

Correspondence recall, which is what a caller seeding a patch cluster from the
result actually consumes, peaks in the same place or one step later: N=50 on
kerry_park (0.59), N=100 on dino_dog_toy (0.45, against 0.44 at N=50) and on
seattle_backyard (0.39, against 0.37), N=200 on DinoLedge (0.60, against 0.51).

**Cost is linear in N.** Time per query on DinoLedge is 50 / 125 / 259 / 460 /
827 / 1,705 / 3,359 ms for N = 10 … 800 out of a 1,494 MB file at a 256 MiB
cache, so a 50-feature query is 259 ms and an 800-feature one is 3.4 s. On the
small corpora it is 1 to 35 ms. The slope is the ANN batch, which is `N × k`
lookups.

### The radius that holds fifty

The radius is the parameter the API actually takes, so the size has to be
expressed as one. Measured medians, and the radius a uniform keypoint density
would predict, `sqrt(N·A / (π·K))` for image area `A` and `K` keypoints per
image:

| dataset | radius @25 | radius @50 | radius @100 | predicted @50 | measured / predicted |
|---|--:|--:|--:|--:|--:|
| DinoLedge | 87 | 128 | 180 | 128 | 1.00 |
| dino_dog_toy | 78 | 115 | 172 | 141 | 0.82 |
| seattle_backyard | 28 | 39 | 56 | 43 | 0.91 |
| kerry_park | 21 | 32 | 50 | 46 | 0.70 |
| seoul_bull | 20 | 28 | 42 | 31 | 0.91 |

The formula is right to within 0 to 30%, always on the high side, because
patches are centred on keypoints and keypoints cluster where there is texture.
kerry_park is the furthest out at 0.70, which is what a fisheye's black corners
do to an area-based density estimate.

---

## 3. Parameter sweeps

All at N=50, one knob at a time with the rest at their defaults, 40 of the 60
patches (all 60 for DinoLedge's `max_leaf_checks` row, run separately).

### `k`: 32 is fine, and it is not where the budget should go

| dataset | k | recall≥3 | prec≥3 | corr recall | res med | warp ok | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 16 | 0.72 | 0.83 | 0.51 | 1.44 | 0.81 | 13 | 1.0 | 0.05 | 250 |
| | **32** | 0.75 | 0.77 | 0.53 | 1.41 | 0.82 | 13 | 2.4 | 0.33 | 268 |
| | 64 | 0.75 | 0.76 | 0.53 | 1.38 | 0.81 | 14.5 | 2.9 | 0.60 | 284 |
| dino_dog_toy | 16 | 0.71 | 0.55 | 0.39 | 1.39 | 0.77 | 17 | 1.3 | 0.17 | 3 |
| | **32** | 0.71 | 0.48 | 0.39 | 1.63 | 0.79 | 26 | 3.2 | 0.55 | 3 |
| | 64 | 0.58 | 0.36 | 0.32 | 2.13 | 0.65 | 43 | 5.6 | 0.85 | 4 |
| seattle_backyard | 16 | 0.73 | 0.70 | 0.49 | 1.21 | 0.81 | 36 | 0.7 | 0.20 | 2 |
| | **32** | 0.66 | 0.55 | 0.36 | 1.61 | 0.65 | 68 | 1.7 | 0.50 | 2 |
| | 64 | 0.61 | 0.39 | 0.22 | 14.78 | 0.32 | 127 | 3.8 | 0.88 | 3 |
| kerry_park | 16 | 0.89 | 0.13 | 0.62 | 1.54 | 0.80 | 25 | 3.1 | 1.23 | 2 |
| | **32** | 0.91 | 0.09 | 0.62 | 1.24 | 0.93 | 38 | 5.3 | 2.62 | 3 |
| | 64 | 0.85 | 0.07 | 0.53 | 1.25 | 0.78 | 70 | 9.0 | 4.58 | 3 |
| seoul_bull | 16 | 0.71 | 0.44 | 0.21 | 2.77 | 0.66 | 51 | 0.8 | 0.23 | 2 |
| | **32** | 0.58 | 0.19 | 0.16 | 38.92 | 0.32 | 96 | 2.2 | 0.50 | 2 |
| | 64 | 0.69 | 0.09 | 0.04 | 167.90 | 0.00 | 189 | 6.1 | 1.52 | 3 |

**This is where DinoLedge and the small datasets disagree, and the mechanism is
visible in the `corr/img` column.** On DinoLedge, going from k=16 to k=64
changes the correspondences per candidate image from 13 to 14.5: the extra
neighbours are spread across 1,196 images, so no single candidate gets more
chaff and its inlier ratio is untouched. Recall goes *up* slightly (0.72 to 0.75)
and warp quality is flat (0.81 / 0.82 / 0.81). On the 17-to-85-image corpora the
same change doubles and quadruples `corr/img` (seattle 36 → 68 → 127, seoul
51 → 96 → 189) because there are only a handful of images for the extra hits to
land in. The inlier ratio collapses, and with it RANSAC's chance of drawing a
clean sample, so warp quality falls off (seattle 0.81 → 0.65 → 0.32, seoul
0.66 → 0.32 → 0.00).

So the spec's argument for a generous `k` ("the consensus test downstream
removes wrong candidates at almost no cost") is right at DinoLedge scale and
wrong at seoul scale, for exactly the reason it identifies as the ceiling: the
correspondence count per candidate image. **k=32 should stay**, because this
query is aimed at large captures and 32 is neutral-to-better there, but it is
worth knowing that on a small corpus it is actively harmful and 16 is better on
four of five metrics.

More importantly, `k` is not the knob that gates correspondence recall at all.
Measured directly, as the share of GT correspondences that appear anywhere in
the neighbour lists (N=50, 30 patches, default budget 128):

| dataset | k=16 | k=32 | k=64 | k=128 |
|---|--:|--:|--:|--:|
| seoul_bull | 0.612 | 0.615 | 0.615 | 0.615 |
| seattle_backyard | 0.776 | 0.777 | 0.777 | 0.777 |
| dino_dog_toy | 0.716 | 0.719 | 0.719 | 0.719 |
| kerry_park | 0.754 | 0.754 | 0.754 | 0.754 |

Everything past the sixteenth neighbour is noise: k=128 buys at most 0.3
percentage points over k=16. The 22 to 39% that is missing is missing because
the traversal never reached it.

### `max_leaf_checks`: the knob that is actually mis-set

Same measurement against the leaf budget instead (seattle_backyard, N=50, k as
shown):

| budget | k=16 | k=32 |
|--:|--:|--:|
| 128 | 0.776 | 0.777 |
| 512 | 0.900 | 0.903 |
| 2048 | 0.966 | 0.973 |

End to end:

| dataset | `max_leaf_checks` | recall≥3 | prec≥3 | corr recall | res med | warp ok | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | **128** | 0.75 | 0.77 | 0.53 | 1.41 | 0.82 | 13 | 2.4 | 0.33 | 278 |
| | 512 | 0.87 | 0.72 | 0.67 | 1.51 | 0.80 | 14.5 | 4.2 | 0.47 | 505 |
| | 2048 | 0.92 | 0.65 | 0.76 | 1.66 | 0.79 | 14.5 | 5.2 | 0.55 | 1,201 |
| dino_dog_toy | **128** | 0.71 | 0.48 | 0.39 | 1.63 | 0.79 | 26 | 3.2 | 0.55 | 3 |
| | 512 | 0.83 | 0.38 | 0.54 | 1.65 | 0.76 | 25.5 | 5.8 | 0.75 | 7 |
| | 2048 | 0.89 | 0.34 | 0.63 | 1.82 | 0.74 | 26 | 7.7 | 1.02 | 18 |
| seattle_backyard | **128** | 0.66 | 0.55 | 0.36 | 1.61 | 0.65 | 68 | 1.7 | 0.50 | 2 |
| | 512 | 0.82 | 0.49 | 0.54 | 1.41 | 0.66 | 65 | 3.3 | 0.93 | 4 |
| | 2048 | 0.91 | 0.42 | 0.64 | 1.64 | 0.64 | 65 | 4.7 | 1.32 | 12 |
| kerry_park | **128** | 0.91 | 0.09 | 0.62 | 1.24 | 0.93 | 38 | 5.3 | 2.62 | 3 |
| | 512 | 0.95 | 0.08 | 0.77 | 1.20 | 0.80 | 37.5 | 8.2 | 3.98 | 5 |
| | 2048 | 0.97 | 0.07 | 0.84 | 1.05 | 0.93 | 37.5 | 10.0 | 4.85 | 14 |
| seoul_bull | **128** | 0.58 | 0.19 | 0.16 | 38.92 | 0.32 | 96 | 2.2 | 0.50 | 2 |
| | 512 | 0.78 | 0.13 | 0.17 | 46.55 | 0.24 | 95.5 | 3.7 | 0.93 | 4 |
| | 2048 | 0.74 | 0.13 | 0.24 | 21.35 | 0.39 | 95.8 | 7.0 | 1.00 | 11 |

128 → 512 raises correspondence recall by 14 to 18 points on four of five
datasets (DinoLedge 0.53 → 0.67, dino_dog_toy 0.39 → 0.54, seattle 0.36 → 0.54,
kerry 0.62 → 0.77; seoul is the exception at 0.16 → 0.17) and image recall by 4
to 20 points (+0.12, +0.12, +0.16, +0.04, +0.20), for 1.7× to 2.3× the time,
with the residual medians unchanged. The `corr/img` column is the reason this is
different in kind from raising `k`: at 128 / 512 / 2048 it stays at 13 / 14.5 /
14.5 and 26 / 25.5 / 26. The budget does not add correspondences, it *replaces
chaff with true ones*, so the inlier ratio improves rather than degrades. 2048
buys another 9 to 12 points for 4× the time again, which is the diminishing end.

### `min_inliers`: 6 is exactly the noise floor

| dataset | `min_inliers` | recall≥3 | prec≥3 | corr recall | res med | res p90 | warp ok | false | never-covis |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 4 | 0.86 | 0.47 | 0.57 | 1.38 | 4.5 | 0.81 | 9.9 | 5.22 |
| | **6** | 0.75 | 0.77 | 0.53 | 1.41 | 4.2 | 0.82 | 2.4 | 0.33 |
| | 8 | 0.63 | 0.83 | 0.48 | 1.38 | 4.0 | 0.84 | 1.4 | 0.15 |
| | 12 | 0.42 | 0.88 | 0.39 | 1.34 | 3.6 | 0.88 | 0.7 | 0.00 |
| dino_dog_toy | 4 | 0.92 | 0.15 | 0.45 | 2.60 | 482.6 | 0.70 | 33.5 | 6.65 |
| | **6** | 0.71 | 0.48 | 0.39 | 1.63 | 5.9 | 0.79 | 3.2 | 0.55 |
| | 8 | 0.50 | 0.68 | 0.31 | 1.79 | 5.2 | 0.80 | 0.8 | 0.07 |
| | 12 | 0.31 | 0.80 | 0.24 | 1.90 | 4.8 | 0.84 | 0.2 | 0.00 |
| seattle_backyard | 4 | 1.00 | 0.23 | 0.39 | 26.44 | 314.3 | 0.45 | 16.7 | 4.75 |
| | **6** | 0.66 | 0.55 | 0.36 | 1.61 | 18.0 | 0.65 | 1.7 | 0.50 |
| | 8 | 0.40 | 0.86 | 0.33 | 1.06 | 4.2 | 0.86 | 0.2 | 0.15 |
| | 12 | 0.31 | 1.00 | 0.29 | 1.01 | 2.8 | 0.90 | 0.0 | 0.00 |
| kerry_park | 4 | 1.00 | 0.03 | 0.67 | 1.96 | 7.2 | 0.83 | 42.3 | 23.00 |
| | **6** | 0.91 | 0.09 | 0.62 | 1.24 | 4.2 | 0.93 | 5.3 | 2.62 |
| | 8 | 0.89 | 0.22 | 0.59 | 1.28 | 2.9 | 0.93 | 1.7 | 0.55 |
| | 12 | 0.71 | 0.35 | 0.51 | 1.01 | 1.6 | 0.94 | 1.0 | 0.25 |
| seoul_bull | 4 | 1.00 | 0.07 | 0.18 | 118.90 | 286.0 | 0.22 | 13.3 | 3.52 |
| | **6** | 0.58 | 0.19 | 0.16 | 38.92 | 154.0 | 0.32 | 2.2 | 0.50 |
| | 8 | 0.20 | 0.64 | 0.08 | 2.93 | 5.2 | 0.70 | 0.1 | 0.00 |
| | 12 | 0.06 | 0.60 | 0.03 | 1.68 | 2.4 | 1.00 | 0.0 | 0.00 |

The decisive number is not in this table. It is the inlier count of the
candidates that are *definitely* wrong, the ones sharing no point with the query
image anywhere in it:

| dataset | N=25 | N=50 | N=100 |
|---|---|---|---|
| DinoLedge | median 6, max 7 | median 7, max 10 | median 7, max 13 |
| dino_dog_toy | median 6, max 8 | median 6, max 10 | median 6, max 9 |
| seattle_backyard | median 6, max 10 | median 6, max 11 | median 6, max 11 |
| kerry_park | median 6, max 15 | median 6, max 24 | median 6, max 39 |
| seoul_bull | median 6, max 7 | median 6, max 7 | median 6, max 8 |

The typical wrong candidate has **exactly six inliers**, on every dataset at
every size. The default sits on the noise floor, which is why `min_inliers=4`
is catastrophic (33 to 42 false candidates per query on the two datasets with
many images) and why moving to 8 removes 60 to 95% of them: seoul 2.2 → 0.1,
seattle 1.7 → 0.2, dino_dog_toy 3.2 → 0.8, kerry 5.3 → 1.7, DinoLedge 2.4 → 1.4,
and on the never-covisible floor 0.50 → 0.00, 0.50 → 0.15, 0.55 → 0.07,
2.62 → 0.55, 0.33 → 0.15.

It costs image recall: 0.75 → 0.63 (DinoLedge), 0.71 → 0.50 (dino_dog_toy),
0.66 → 0.40 (seattle), 0.91 → 0.89 (kerry), 0.58 → 0.20 (seoul). That recall is
mostly made of six-inlier candidates, which the row above says are usually
wrong, so the trade is favourable whenever a caller wants the warp; it is not
favourable if a caller only wants a list of images to look at.

### `threshold_px`: a fixed pixel count cannot be right at both ends

| dataset | image px | `threshold_px` | recall≥3 | corr recall | res med | res p90 | warp ok | false |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge | 2160×3840 | 4 | 0.74 | 0.51 | 1.33 | 3.4 | **0.90** | 2.2 |
| | | **8** | 0.75 | 0.53 | 1.41 | 4.2 | 0.82 | 2.4 |
| | | 12 | 0.76 | 0.53 | 1.54 | 5.0 | 0.76 | 2.4 |
| dino_dog_toy | 2040×1536 | 4 | 0.66 | 0.34 | 1.31 | 3.9 | **0.89** | 2.6 |
| | | **8** | 0.71 | 0.39 | 1.63 | 5.9 | 0.79 | 3.2 |
| | | 12 | 0.73 | 0.41 | 1.75 | 8.8 | 0.68 | 3.6 |
| seattle_backyard | 360×640 | 4 | 0.47 | 0.31 | 1.24 | 5.2 | **0.80** | 1.2 |
| | | **8** | 0.66 | 0.36 | 1.61 | 18.0 | 0.65 | 1.7 |
| | | 12 | 0.74 | 0.39 | 2.48 | 37.4 | 0.58 | 3.2 |
| kerry_park | 480×480 | 4 | 0.89 | 0.59 | 0.89 | 2.6 | **0.98** | 3.0 |
| | | **8** | 0.91 | 0.62 | 1.24 | 4.2 | 0.93 | 5.3 |
| | | 12 | 0.91 | 0.65 | 0.90 | 3.1 | 0.90 | 8.5 |
| seoul_bull | 270×480 | 4 | 0.30 | 0.08 | 2.95 | 279.0 | **0.53** | 0.6 |
| | | **8** | 0.58 | 0.16 | 38.92 | 154.0 | 0.32 | 2.2 |
| | | 12 | 0.83 | 0.19 | 66.52 | 184.0 | 0.18 | 6.1 |

Tightening to 4 px raises `warp ok` on all five and cuts false candidates on all
five. On the two large-image datasets it costs almost nothing in recall
(DinoLedge 0.75 → 0.74, dino_dog_toy 0.71 → 0.66) and on kerry_park nothing at
all (0.91 → 0.89, with `warp ok` going 0.93 → 0.98 and residual p90 4.2 → 2.6 px).
On the two tiny-image datasets it costs a lot (seattle 0.66 → 0.47, seoul
0.58 → 0.30), because at 270×480 with 2,186 keypoints there is a keypoint every
7.7 px and a 4 px ball is smaller than the keypoint spacing.

Put the other way: 8 px is 0.4% of DinoLedge's short side and 3.0% of seoul's.
The same constant is doing two different jobs.

### `iterations`: fine at N=50, far too few above it

| dataset | 200 | 1000 | 5000 |
|---|---|---|---|
| DinoLedge | corr 0.53, warp ok 0.82, 256 ms | corr 0.53, 0.82, 255 ms | corr 0.53, 0.82, 318 ms |
| dino_dog_toy | corr 0.39, 0.79, 4 ms | corr 0.41, 0.79, 9 ms | corr 0.44, 0.77, 34 ms |
| seattle_backyard | corr 0.36, 0.65, 2 ms | corr 0.49, 0.74, 4 ms | corr 0.52, 0.74, 15 ms |
| kerry_park | corr 0.62, 0.93, 3 ms | corr 0.63, 0.79, 6 ms | corr 0.64, 0.74, 21 ms |
| seoul_bull | corr 0.16, 0.32, 2 ms | corr 0.21, 0.29, 4 ms | corr 0.31, 0.52, 12 ms |

On DinoLedge the three rows are bit-identical, because at N=50 a candidate there
is offered 13 correspondences of which 12 are inliers and 200 samples is
overwhelming. The datasets where more iterations help are exactly the ones where
`k=32` floods each candidate with chaff: seattle at 68 correspondences and 11
inliers is a 16% inlier ratio, so a three-point sample is clean 0.4% of the time
and 200 draws give a 56% chance of one. That is a `k` problem showing up as an
`iterations` symptom, and fixing the inlier ratio is the better lever: it also
costs nothing, whereas more iterations raise the false-candidate count on every
dataset (seoul 2.2 → 9.5, kerry 5.3 → 9.9) by giving the search more chances to
find a spurious six-inlier consensus.

At N=200 and above the picture changes. seoul offers 366 correspondences per
candidate at ~7 inliers, a 2% ratio, where 200 samples have essentially no
chance. That is part of why `warp ok` collapses at large N, and a caller who
insists on a large patch should raise `iterations` with it. At the recommended
size, **keep 200**.

### Combined: `max_leaf_checks=512` with `min_inliers=8`, at N=50

| dataset | | recall≥3 | prec≥3 | corr recall | res med | res p90 | warp ok | false | never-covis | ms |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DinoLedge* | defaults | 0.75 | 0.77 | 0.53 | 1.41 | 4.2 | 0.82 | 2.4 | 0.33 | 278 |
| | combined | 0.75 | 0.77 | 0.62 | 1.51 | 4.4 | 0.80 | 2.5 | 0.30 | 508 |
| dino_dog_toy | defaults | 0.73 | 0.54 | 0.44 | 1.59 | 5.7 | 0.75 | 3.1 | 0.45 | 4 |
| | combined | 0.71 | 0.60 | 0.54 | 1.56 | 5.3 | 0.76 | 1.6 | 0.03 | 7 |
| seattle_backyard | defaults | 0.61 | 0.52 | 0.37 | 1.64 | 18.0 | 0.65 | 1.9 | 0.47 | 2 |
| | combined | 0.55 | 0.78 | 0.48 | 1.26 | 4.9 | 0.79 | 0.4 | 0.07 | 4 |
| kerry_park | defaults | 0.87 | 0.09 | 0.59 | 1.63 | 5.0 | 0.89 | 5.3 | 2.55 | 3 |
| | combined | 0.83 | 0.16 | 0.67 | 1.18 | 4.2 | 0.81 | 2.8 | 0.87 | 6 |
| seoul_bull | defaults | 0.60 | 0.19 | 0.17 | 20.19 | 157.5 | 0.33 | 2.3 | 0.58 | 2 |
| | combined | 0.31 | 0.61 | 0.15 | 2.43 | 10.6 | 0.44 | 0.2 | 0.07 | 4 |

*DinoLedge's two rows are the 40-patch runs, so that the comparison is
like-for-like; the other four are the 60-patch ones.

Correspondence recall up 8 to 11 points on four of five (DinoLedge +0.09,
dino_dog_toy +0.10, seattle +0.11, kerry +0.08; seoul -0.02), false candidates
roughly halved or better on three of five (dino_dog_toy 3.1 → 1.6, seattle
1.9 → 0.4, seoul 2.3 → 0.2) and flat on DinoLedge, warp quality flat or better
except on kerry (0.89 → 0.81), image recall within 6 points except on seoul, for
about twice the wall time. seoul is the one dataset the combination
hurts on recall, and it is the one that is hardest to take seriously: 270×480
images where a 50-feature constellation is a 28 px disc and `threshold_px=8` is
3% of the frame.

---

## 4. A defect: the degeneracy test checks that the warp is invertible, and nothing else

The spec's "Refusing a model that collapses the patch" tests the determinant of
the fitted 2×2, which rules out a model that flattens the patch onto a line or a
point. It is a test for *non-zero*, so it lets through two physically impossible
warps that the same determinant would separate: a model that inflates the patch
without bound, and a model that mirrors it.

**Reproducing case, deterministic.** seoul_bull, the workspace built by
`scripts/init_dataset_seoul_bull.sh` and solved by its `sfm_solve.sh`; a `.kdf`
over its `.sift` files as in §1. Take image 11 in `.kdf` order
(`images/seoul_bull_sculpture_12.jpg`), its keypoint **0** as the centre, and the
10 nearest keypoints as the constellation, which is a radius of 10.5 px. Call
`constellation_query` with every parameter at its default. It reports image 2
(`images/seoul_bull_sculpture_03.jpg`) with 6 inliers out of 16 correspondences,
and the affine's 2×2 has singular values **33.81 and 5.78** and determinant
**-195.4**. A 10.5 px-radius patch is stretched 33.8-fold along one axis and
5.8-fold along the other (geometric-mean scale 14.0) and mirrored, in a
270 × 480 image. The determinant is far from zero, the condition number is 5.9,
and the six inlier destinations are six distinct points, so every existing guard
passes.

Sampled from a random keypoint, the same image produces a candidate at singular
values 36.94 and 10.80 with a median ground-truth residual of 148 px.

**Reflections are never right.** Two real cameras looking at the same piece of
surface cannot mirror it, so a negative determinant is refusable on sight. Over
60 patches per dataset at N = 25, 50 and 100, taking every reported candidate
that carries at least three GT correspondences:

| dataset | det > 0: n | within 3 px | det < 0: n | share of candidates | within 3 px |
|---|--:|--:|--:|--:|--:|
| DinoLedge | 1,586 | 0.76 | 0 | 0% | n/a |
| dino_dog_toy | 1,283 | 0.67 | 23 | 2% | **0.22** |
| seattle_backyard | 472 | 0.68 | 69 | 13% | **0.00** |
| kerry_park | 132 | 0.72 | 8 | 6% | **0.12** |
| seoul_bull | 112 | 0.22 | 48 | 30% | **0.00** |

One sign test would remove 30% of seoul_bull's candidates and 13% of
seattle_backyard's, of which none at all were correct.

**How common the blow-up is.** Crossing the geometric-mean scale `sqrt(|det|)` of every
reported candidate against the median residual of its GT correspondences, over
60 patches per dataset at five sizes:

| dataset | N | scale in [0.5, 2] | | scale outside | |
|---|--:|--:|---|--:|---|
| | | n | within 3 px | n | within 3 px |
| seoul_bull | 50 | 26 | 0.58 | 22 | **0.05** |
| | 100 | 60 | 0.22 | 39 | **0.03** |
| | 200 | 119 | 0.08 | 61 | **0.00** |
| seattle_backyard | 25 | 136 | 0.88 | 15 | **0.07** |
| | 50 | 200 | 0.74 | 27 | **0.00** |
| | 100 | 261 | 0.57 | 55 | **0.00** |
| dino_dog_toy | 50 | 362 | 0.79 | 55 | **0.47** |
| | 100 | 551 | 0.63 | 100 | **0.34** |
| | 200 | 741 | 0.37 | 155 | **0.18** |
| kerry_park | 100 | 87 | 0.69 | 5 | **0.00** |
| | 200 | 148 | 0.43 | 21 | **0.00** |
| DinoLedge | 50 | 562 | 0.77 | 8 | **0.50** |
| | 200 | 1,120 | 0.55 | 47 | **0.51** |

On the four wide-baseline-stills datasets, 32 to 54% of all reported candidates
have a scale outside [0.5, 2] (at N=50 alone: 27.8% dino_dog_toy, 37.4%
seattle_backyard, 36.7% kerry_park, 75.2% seoul_bull; 75th percentile of the
scale distribution 2.02 / 2.65 / 2.54 / 3.28). Their median GT residual is 83 to
195 px on the three small-image datasets and 3.2 to 11.7 px on dino_dog_toy,
whose 2040 × 1536 frames make even a wrong warp land closer in absolute pixels.
A bound on the scale would remove a large share of the false candidates for the
price of two multiplications.

**Scope.** DinoLedge barely shows it: its scale distribution is 0.97 / 1.00 /
1.03 at the 25th / 50th / 75th percentile and only 1% of candidates exceed 2.9,
because it is a video walk in which consecutive frames differ by a few percent
of scale. The defect bites where a small corpus lets chaff dominate a candidate
image's correspondence list, which is the same condition that makes `k=32` and
`iterations=200` marginal there. The visual check in §5 confirms it: every
high-scale candidate whose crop was inspected showed unrelated content, usually
mapped mostly off the edge of the candidate image.

Not fixed here, per the terms of this evaluation.

> _Status (2026-09-14): Done. The three-point solve now refuses a negative
> determinant outright and a geometric-mean scale outside
> `[1/max_scale, max_scale]`, `max_scale` defaulting to 4.0. Measured in the
> Addendum._

---

## 5. The ground truth is a solve, and it is not always right

Every "false candidate" above is an image the query reported and the
reconstruction's tracks do not corroborate *for this patch*. That is not the
same as wrong. The solve's tracks came from the same SIFT descriptors the index
holds, so this measures agreement with one particular solve; a correspondence
the solve never made is counted against the query. How badly that bites depends
entirely on how dense the solve's tracks are, and the precision column follows
that density almost exactly:

| dataset | share of features tracked | prec≥3 at N=50 |
|---|--:|--:|
| DinoLedge | 46.6% | 0.77 |
| dino_dog_toy | 40.4% | 0.54 |
| seattle_backyard | 27.7% | 0.52 |
| seoul_bull | 10.2% | 0.19 |
| kerry_park | 3.6% | 0.09 |

kerry_park's precision of 0.09 is a statement about a solve with 2,957
observations over 48 images, not about the query. This is why the
`never-covis` column exists: it counts only candidates sharing no point with the
query image *anywhere in the frame*, which sparsity cannot manufacture, and it
is 5 to 10× smaller than the `false` column throughout.

**By eye.** Crops were dumped for false candidates at N=50: the query patch at
its own centre and radius, beside the region the fitted affine maps it into,
both resampled to 256×256 (written by `--dump-disagreements`, saved to the
scratch dir).

- `false-368-vs-369-n50-in24-scale1.00.jpg` (DinoLedge, 24 inliers): the same
  ivy leaves, the same rope-like root crossing the same stone, at the same scale
  and nearly the same position. Unambiguously the same surface. The solve simply
  carried no track through this patch between consecutive frames.
- `false-1017-vs-1032-n50-in7-scale1.03.jpg` (DinoLedge, 7 inliers, 15 frames
  apart): the same blue-grey stone above the same two rounded rocks, the same
  green leaf at the bottom edge. Correct, on the weakest evidence the default
  accepts.
- `false-6-vs-8-n51-in10-scale1.00.jpg` (dino_dog_toy, 10 inliers): the same
  corner of a room, the same floorboard joints, the same curved grey edge and
  the same pink textured object. Correct.
- `false-40-vs-38-n50-in22-scale0.87.jpg` (kerry_park, 22 inliers): the same
  cloud pattern over the same tree-and-tower skyline. Correct, and a good
  illustration of why kerry_park's precision number means nothing: sky and
  foliage are exactly what a solve declines to track.
- `false-13-vs-24-n50-in6-scale7.27.jpg` (seattle_backyard, 6 inliers, scale
  7.3): a mossy stone beside grass, mapped into a region that is mostly outside
  the candidate image and whose visible sliver is a distant garden bed. Wrong,
  and an instance of §4.
- `false-72-vs-82-n50-in6-scale12.91.jpg` (dino_dog_toy, 6 inliers, scale 12.9):
  wood grain blown up almost thirteen-fold onto a strip containing a dog's leg.
  Wrong, same cause.
- `false-4-vs-6-n50-in7-scale0.76.jpg` (seattle_backyard, 7 inliers): grass with
  fallen leaves in both crops, but no leaf in one corresponds to a leaf in the
  other. This is the genuine repeated-texture failure, at the six-to-seven-inlier
  floor, with a plausible scale. It is the case `min_inliers` has to catch,
  because no scale bound will.

Of the seven inspected, four are correct matches the solve missed and three are
genuine errors. Two of the three errors carry an absurd scale, and the third
sits at seven inliers. Both observations point the same way as §3 and §4.

---

## 6. Recommendation

**Constellation size: fifty features.** As a rule of thumb, the radius
`sqrt(50·A / (π·K))` for image area `A` and `K` keypoints per image, which came
out at 128 px on DinoLedge (2160×3840, 8,112 keypoints), 115 px on dino_dog_toy,
39 px on seattle_backyard, 32 px on kerry_park and 28 px on seoul_bull, with the
measured radius 70 to 100% of the predicted one. Twenty-five is the conservative
end and fifty the peak; a hundred is defensible if a caller cares more about
finding the image than about the warp. Above two hundred the warp is wrong more
often than right on every dataset measured, and past four hundred the query is
doing whole-image matching at whole-image cost (3.4 s per query on DinoLedge at
N=800, against 259 ms at N=50).

> _Status (2026-09-14): Done as far as a library can take it. The rule of thumb
> is now `radius_for_feature_count(width, height, keypoints, target)` beside the
> query; the size itself stays the caller's choice, since the query takes a
> radius._

The justification, in three numbers: at N=50 the fraction of found images whose
warp places the ground truth's own correspondences within 3 px is 0.76 / 0.75 /
0.65 / 0.89 / 0.33 across the five datasets; at N=200 it is 0.54 / 0.33 / 0.28 /
0.37 / 0.06; at N=800 it is 0.23 / 0.06 / 0.04 / 0.07 / 0.01. Image recall from
N=50 to N=800 goes up by 0.17 / 0.03 / 0.39 / 0.13 / 0.40 over the same five, so
the extra features buy images at the cost of knowing where they are. (Order
throughout: DinoLedge, dino_dog_toy, seattle_backyard, kerry_park, seoul_bull.)

**Defaults to change.**

> _Status (2026-09-14): Done for 1 and 2 (`max_leaf_checks` 512, `min_inliers`
> 8, both in `ConstellationParams::DEFAULT` and so in the Python keyword
> defaults). 3, 4 and 5 stand as written: nothing changed for `threshold_px`,
> `k` or `iterations`. Measured in the Addendum._

1. **`max_leaf_checks` 128 → 512.** The biggest single win and the one with the
   least downside. At 128 only 62 to 78% of the ground truth's correspondences
   are in the neighbour lists at all; on seattle_backyard, where the budget was
   swept directly, 512 takes that to 90% and 2048 to 97%. End to end at N=50 the
   move to 512 is +0.14 to +0.18 correspondence recall on four of five datasets
   and +0.04 to +0.20 image recall on all five, with residual medians and
   `corr/img` unchanged, for 1.7× to 2.3× the time (DinoLedge 278 → 505 ms; the
   small datasets 2 → 4 ms and 3 → 7 ms). Unlike `k`, the budget replaces chaff
   rather than adding it.

2. **`min_inliers` 6 → 8.** The median inlier count of a candidate that shares
   no point at all with the query image is six, on all five datasets at all
   sizes: the default is sitting on the noise floor. Moving to 8 removes 42 to
   95% of false candidates (DinoLedge 2.4 → 1.4, dino_dog_toy 3.2 → 0.8,
   seattle 1.7 → 0.2, kerry 5.3 → 1.7, seoul 2.2 → 0.1) and 55 to 100% of the
   never-covisible ones, and raises `warp ok`
   or leaves it flat everywhere. It costs image recall, most of which is
   six-inlier candidates. If a caller wants a list of images rather than warps,
   6 remains the right floor for them and this is an argument for documenting
   the trade rather than for refusing the change.

3. **`threshold_px` 8: keep the number, but it should not be a number.** Four
   pixels is better on every dataset by `warp ok` (0.90 / 0.89 / 0.80 / 0.98 /
   0.53 against 0.82 / 0.79 / 0.65 / 0.93 / 0.32) and by false candidates, and
   nearly free in recall on the three datasets whose images are 480 px or wider
   on the short side. It is expensive on the two smallest (seattle 0.66 → 0.47,
   seoul 0.58 → 0.30). A default expressed relative to the image (about 0.5% of
   the short side, which is 4 px at 800 px and 11 px at 2160 px) or to the median
   keypoint spacing would be right at both ends; 8 px is a compromise that is
   loose on a 4K frame and tight on a 270 px one. Absent such a rule, 8 stays.

4. **`k` 32: keep.** It is neutral-to-better on DinoLedge (recall 0.72 → 0.75
   from 16 to 32, `warp ok` flat) and worse on the small corpora, and the direct
   measurement shows neighbours past the sixteenth contribute under one
   percentage point of true correspondences at any leaf budget. Spending on
   `max_leaf_checks` instead is strictly better. If a future version wants to
   adapt one knob to corpus size, `k` scaled down for small image counts is the
   one with evidence behind it.

5. **`iterations` 200: keep.** No effect at all on DinoLedge at N=50 (three
   sweep points bit-identical), and where it helps it is compensating for a low
   inlier ratio that `k` and `max_leaf_checks` address more cheaply. It does
   matter above N=200, where the inlier ratio falls to a few percent; a caller
   using a large patch should raise it, and the spec could say so.

**Not a default, but the cheapest single improvement available.** Two sign-level
tests on the fitted 2×2, beyond the non-zero determinant already checked:
refuse a negative determinant, and refuse a scale far from unity. The first
would drop 30% of seoul_bull's and 13% of seattle_backyard's reported
candidates, of which none were correct, and none at all of DinoLedge's. The
second, an overlapping set, covers 27.8 to 75.2% of the candidates at N=50 on
the four small datasets and 1.4% on DinoLedge, and their correctness rate is
0.00 to 0.50 against 0.58 to 0.89 for the rest. Both are §4; neither was
implemented here.

> _Status (2026-09-14): Done. Both tests ship, the scale one as the `max_scale`
> parameter. Measured in the Addendum._

**Where DinoLedge and the small datasets disagree, and why.** Only on `k`, and
the mechanism is the correspondence count per candidate image. With 1,196 images
the neighbour lists spread thin: at k=64 a candidate on DinoLedge is offered 14.5
correspondences, of which 12 are inliers. With 17 images they pile up: the same
k=64 offers a seoul candidate 189 correspondences at 6 inliers, a 3% inlier
ratio that no 200-iteration RANSAC can work with. The spec's reasoning for a
generous `k` is correct for the scale it was written for; it just does not
survive down to a seventeen-image capture, and the defaults are one set for both.
Everything else in this report agrees across all five, including the size
recommendation, which is the question this was run to answer.

---

## Appendix: reproducing

```bash
# One .kdf per workspace, with SIFT sources (see the report's §1 for the shape).
pixi run -e test python scripts/kdf_constellation_eval.py \
    --workspace WS --kdf capture.kdf --sfmr WS/sfmr/solve.sfmr \
    --label NAME --patches 60 --sweep-size 50 --sweep-patches 40 \
    --cache-mib 64 --dump-disagreements 10 --out results-NAME.json
```

`--features DIR` is needed when the `.kdf` names images by basename rather than
by workspace-relative path, as for DinoLedge. Raw per-patch results, the crops
and the scale/ANN diagnostics were written to the scratch directory, not to the
repository.

---

## Addendum, 2026-09-14: the guards and the new defaults, measured on seoul_bull

The two findings above were applied to the implementation:
`ConstellationParams::DEFAULT` now carries `max_leaf_checks = 512` and
`min_inliers = 8` (§6, recommendations 1 and 2), and the three-point solve
refuses a model whose 2x2 linear part has a negative determinant or a
geometric-mean scale `sqrt(|det|)` outside `[1/max_scale, max_scale]`, with
`max_scale` a new parameter defaulting to 4.0 (§4). A refused model is skipped
inside the solve, so it never scores and can neither win a trial nor be
reported. `radius_for_feature_count` was added beside the query, turning §6's
`sqrt(N·A / (π·K))` rule of thumb into a function.

**Re-run.** seoul_bull only, the same workspace, `.kdf` and ground-truth `.sfmr`
as §1, the same 60 seeded patch centres, N=50, 64 MiB cache, no parameter
sweeps. DinoLedge was not re-run. The first row reproduces §2's seoul_bull N=50
row exactly on the pre-change build, so the four rows are directly comparable;
the middle two exist to separate the guards from the defaults, and were obtained
by holding the old defaults and, in the second row, lifting the scale bound to
infinity (the reflection test has no bound to lift).

| arm | `max_leaf_checks` | `min_inliers` | guards | recall≥3 | prec≥3 | corr recall | res med | res p90 | **warp ok** | false | never-covis | ms |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| before | 128 | 6 | none | 0.60 | 0.19 | 0.17 | 20.19 | 157.5 | **0.33** | 2.32 | 0.58 | 2 |
| guards only, reflection alone | 128 | 6 | sign | 0.52 | 0.23 | 0.16 | 5.70 | 138.8 | **0.38** | 1.43 | 0.33 | 1 |
| guards only, both | 128 | 6 | sign + scale 4.0 | 0.46 | 0.32 | 0.18 | 3.14 | 77.7 | **0.44** | 0.85 | 0.17 | 2 |
| after | 512 | 8 | sign + scale 4.0 | 0.26 | 0.67 | 0.14 | 2.17 | 8.8 | **0.54** | 0.05 | 0.00 | 3 |

False candidates fall 2.32 to 0.05 per query and never-covisible ones 0.58 to
0.00, the share of trustworthy warps rises 0.33 to 0.54, the residual 90th
percentile falls from 157 px to 8.8 px, and a query costs 3 ms against 2 ms.
Image recall falls 0.60 to 0.26, as §3 said it would: most of what is lost is
six-inlier candidates, and precision rises 0.19 to 0.67 over the same move. On
the dataset the §4 defect was found on, the guards alone carry more than half of
the improvement in `warp ok` and remove 63% of the false candidates, for 0.14 of the
image recall.

Two things the split says that the earlier tables could not. The reflection test
and the scale bound are not redundant: each removes candidates the other keeps
(2.32 to 1.43 to 0.85 false candidates as they are added). And at the new
defaults they overlap completely with `min_inliers = 8` on this dataset, where
the run with the scale bound lifted to infinity is identical row for row to the
one at 4.0: the absurd scales here ride on six- and seven-inlier candidates that
the higher floor already removes. Compared against §3's combined row, which is
these defaults without any guard, the reflection test still earns its place
(`warp ok` 0.44 to 0.54, false 0.2 to 0.05, residual p90 10.6 px to 8.8 px). The
scale bound's value at the default is as a cap a caller can tighten, and as
insurance for the datasets of §4 where a third to three quarters of candidates
sit outside `[0.5, 2]`.
