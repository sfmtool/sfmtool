# One feature, one place: collapsing repeated hits inside a candidate image, 2026-09-14

Measurement of what happens when a constellation feature lands two or more hits
in the same candidate image, on the five datasets of
[the constellation query evaluation](2026-09-14-constellation-query-eval.md).
The query gives each of its ~50 features `k = 32` nearest descriptors over the
whole corpus and then groups the hits by image, and nothing in it stops one
feature from contributing several correspondences to one candidate. Three
questions: how often that happens, whether Lowe's ratio test can be applied
inside such a group, and whether doing so helps.

**Answer, in one line.** It happens constantly on a small corpus and almost never
on a large one, the extra hits are nearly all wrong, and dropping them is one of
the larger free wins available: keeping only the nearest hit of each
(feature, image) pair takes seoul_bull's share of trustworthy warps from 0.54 to
0.93 and its correspondence recall from 0.14 to 0.29, takes seattle_backyard's
from 0.80 to 0.93 and 0.48 to 0.60, and changes nothing at all on DinoLedge,
where only 2% of the pairs are crowded to begin with. The ratio test on top of
that is a second-order adjustment: at 0.9 it is a small further gain on three of
five and a wash on the rest; at 0.8 it trades image recall for precision.

Implemented and measured here: `ConstellationParams::one_hit_per_image` (keep
the nearest hit of each pair) and `ConstellationParams::same_image_ratio`
(Lowe's ratio applied inside the pair). Both are off by default, so the default
answer is unchanged; §5 is the recommendation about that.

---

## 1. Protocol

Everything is the §1 protocol of the size evaluation, re-used unchanged: the
same five `.kdf` corpora built from the same workspaces, the same `sift_files`
ground-truth `.sfmr` per dataset, the same 60 seeded patch centres (numpy PCG64,
seed 0: a random registered image, then a random keypoint of it), the same
scoring, and all queries through `LazyKdForest`. Constellation size is 50
features on every dataset and in every arm, the radius being the distance to the
50th nearest keypoint of the query image, so the arms see exactly the same
patches and exactly the same features.

Parameters are the current defaults throughout: `k = 32`,
`max_leaf_checks = 512`, `threshold_px = 8`, `iterations = 200`,
`min_correspondences = 3`, `min_inliers = 8`, `max_scale = 4.0`, `seed = 0`.
Cache budget 256 MiB for DinoLedge and 64 MiB for the four small datasets.

**Four arms**, differing only in how a crowded (constellation feature, candidate
image) pair is treated:

| arm | `one_hit_per_image` | `same_image_ratio` | what it does to a pair holding two or more hits |
|---|---|---|---|
| baseline | `false` | `1.0` | nothing: every hit is a correspondence |
| dedupe only | `true` | `1.0` | keeps the nearest hit, drops the rest |
| ratio 0.8 | (implied) | `0.8` | keeps the nearest hit only if `d_best < 0.8 * d_second` |
| ratio 0.9 | (implied) | `0.9` | the same at 0.9 |

A pair holding one hit is untouched in every arm. The distances are the
forest's, which are **squared** Euclidean, so the ratio is applied against its
square; the two knobs are described in
[the spec](../../specs/core/features/kdf-constellation-query.md#one-feature-matches-one-place-in-an-image).

**Consistency with the earlier report.** The baseline arm reproduces the
Addendum's "after" row on seoul_bull exactly -- recall≥3 0.26, prec≥3 0.67, corr
recall 0.14, residual 2.17 / 8.8 px, warp ok 0.54, false 0.05, never-covis 0.00
-- so the four arms sit on the same measurement as the defaults recommendation
did.

**How the crowding was counted.** From the raw neighbour lists, before the query
filters anything. Nothing on `LazyKdForest` reads a descriptor back by feature
ID, so the harness re-runs the same batch search through `LazyKdForest.query`
(the same `search_batch_with_distances` underneath, same `k`, same leaf budget)
with the constellation's descriptors read from the query image's `.sift` file,
maps each hit to its image through the corpus offsets, drops the query image's
own hits as the query does, and groups by (constellation feature, candidate
image). A feature's neighbour list is in ascending distance, so a pair's first
hit is its nearest. Written as `--multiplicity` in
[`scripts/kdf_constellation_eval.py`](../../scripts/kdf_constellation_eval.py),
which also gained `--same-image-ratio` and `--one-hit-per-image`.

---

## 2. How often one feature hits one image twice

Pooled over the 60 patches of each dataset at 50 features, counting every
(constellation feature, candidate image) pair the neighbour lists produce.
"Crowded" means the pair holds two or more hits. "GT cells" are the crowded
pairs holding at least one hit the reconstruction's tracks corroborate.

| dataset | images | crowded pairs | extra hits, as a share of all correspondences | GT cells | nearest hit is the GT one | a later hit is |
|---|--:|--:|--:|--:|--:|--:|
| DinoLedge | 1,196 | **0.02** | 0.02 | 250 | 234 (**0.94**) | 16 |
| dino_dog_toy | 85 | **0.20** | 0.20 | 1,232 | 1,193 (**0.97**) | 39 |
| kerry_park | 48 | **0.30** | 0.28 | 175 | 171 (**0.98**) | 4 |
| seattle_backyard | 26 | **0.48** | 0.41 | 2,029 | 1,992 (**0.98**) | 37 |
| seoul_bull | 17 | **0.65** | 0.54 | 374 | 362 (**0.97**) | 12 |

Two things, and they are the whole argument.

**The crowding is a function of corpus size.** With 1,196 images, 32 neighbours
of one feature land in 32 near-distinct images and only 2% of pairs get a second
hit. With 17 images they cannot: 32 neighbours over 17 images crowd by the
pigeonhole principle before any question of similarity arises, and 65% of pairs
hold more than one hit, with 54% of all correspondences being non-nearest
members of such a pair. The ordering of the column is exactly the ordering of
the image count. This is the same mechanism §3 of the earlier report identified
behind `k = 32` being harmful on small corpora -- the correspondence count per
candidate image -- seen from the other end.

**The extra hits are nearly all wrong.** Where a crowded pair contains a
correspondence the ground truth believes in, that correspondence is the pair's
*nearest* hit 94 to 98% of the time. So collapsing a pair to its nearest costs
2 to 6% of the true correspondences inside crowded pairs, and removes the other
members, which by construction cannot also be true: one point of one surface
appears once per photograph. On seoul_bull that is 54% of every correspondence
RANSAC is offered, thrown away at a cost of 12 true ones out of 374.

---

## 3. The four arms, one table per dataset

60 patches, 50 features, medians over patches except the rates, which are pooled.
`corr/img` is the correspondences offered per candidate image, which is what
RANSAC's inlier ratio is measured against. `false` and `never-covis` are per
query.

### DinoLedge (1,196 images, 9,702,948 descriptors, 256 MiB cache)

| arm | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0.75 | 0.49 | 0.76 | 0.60 | 1.69 | 4.8 | **0.77** | 17 | 2.15 | 0.22 | 529 |
| dedupe only | 0.74 | 0.49 | 0.76 | 0.60 | 1.78 | 4.5 | **0.76** | 16 | 2.08 | 0.20 | 569 |
| ratio 0.8 | 0.74 | 0.48 | 0.76 | 0.60 | 1.71 | 4.5 | **0.77** | 16 | 2.07 | 0.20 | 585 |
| ratio 0.9 | 0.74 | 0.48 | 0.76 | 0.60 | 1.77 | 4.5 | **0.76** | 16 | 2.07 | 0.20 | 579 |

### dino_dog_toy (85 images, 212,500 descriptors, 64 MiB cache)

| arm | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0.70 | 0.47 | 0.63 | 0.54 | 1.56 | 5.2 | **0.76** | 31 | 1.43 | 0.03 | 8 |
| dedupe only | 0.69 | 0.43 | 0.67 | 0.52 | 1.56 | 5.1 | **0.76** | 24 | 0.98 | 0.00 | 8 |
| ratio 0.8 | 0.66 | 0.40 | 0.69 | 0.50 | 1.43 | 4.5 | **0.79** | 21 | 0.80 | 0.00 | 7 |
| ratio 0.9 | 0.68 | 0.41 | 0.67 | 0.51 | 1.53 | 4.7 | **0.78** | 22 | 0.88 | 0.00 | 8 |

### seattle_backyard (26 images, 52,000 descriptors, 64 MiB cache)

| arm | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0.54 | 0.39 | 0.84 | 0.48 | 1.24 | 4.8 | **0.80** | 72 | 0.20 | 0.07 | 5 |
| dedupe only | 0.69 | 0.48 | 0.88 | 0.60 | 1.02 | 3.4 | **0.93** | 39 | 0.05 | 0.00 | 6 |
| ratio 0.8 | 0.64 | 0.44 | 0.94 | 0.56 | 0.92 | 3.4 | **0.92** | 28 | 0.02 | 0.00 | 4 |
| ratio 0.9 | 0.69 | 0.48 | 0.90 | 0.60 | 1.03 | 3.4 | **0.94** | 30 | 0.02 | 0.00 | 4 |

### kerry_park (48 fisheye images, 83,145 descriptors, 64 MiB cache)

| arm | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0.83 | 0.66 | 0.16 | 0.67 | 1.18 | 4.2 | **0.81** | 46 | 2.62 | 0.80 | 6 |
| dedupe only | 0.82 | 0.62 | 0.18 | 0.67 | 1.23 | 4.2 | **0.89** | 31 | 2.42 | 0.68 | 5 |
| ratio 0.8 | 0.82 | 0.61 | 0.20 | 0.64 | 0.95 | 3.1 | **0.85** | 25 | 2.17 | 0.55 | 5 |
| ratio 0.9 | 0.85 | 0.63 | 0.21 | 0.67 | 1.32 | 2.9 | **0.87** | 27 | 2.25 | 0.58 | 6 |

### seoul_bull (17 images, 37,167 descriptors, 64 MiB cache)

| arm | recall≥3 | recall≥1 | prec≥3 | corr recall | res med | res p90 | **warp ok** | corr/img | false | never-covis | ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0.26 | 0.16 | 0.67 | 0.14 | 2.17 | 8.8 | **0.54** | 98 | 0.05 | 0.00 | 4 |
| dedupe only | 0.52 | 0.28 | 0.68 | 0.29 | 1.05 | 3.0 | **0.93** | 44 | 0.07 | 0.00 | 4 |
| ratio 0.8 | 0.49 | 0.28 | 0.71 | 0.27 | 0.90 | 2.6 | **0.89** | 23 | 0.03 | 0.00 | 3 |
| ratio 0.9 | 0.56 | 0.30 | 0.69 | 0.30 | 0.98 | 3.0 | **0.91** | 27 | 0.05 | 0.00 | 3 |

### What the five tables say together

**The `corr/img` column moves first, and everything else follows it.** The
collapse is the only thing in these arms that changes what RANSAC is offered,
and it changes it by exactly the crowding of §2: seoul 98 → 44 (a 55% cut
against a 54% extra-hit share), seattle 72 → 39, kerry 46 → 31, dino_dog_toy
31 → 24, DinoLedge 17 → 16. The inlier counts barely move with it -- seoul's
median true-inlier count is 13.0 baseline and 13.25 deduped, seattle's 16.75 and
15.5 -- so what leaves is outliers, and the inlier ratio roughly doubles on the
two smallest captures. The three-point sampler's chance of a clean draw goes
with the cube of that ratio.

**Where the crowding is, the gain is large.** seoul_bull: `warp ok` 0.54 → 0.93,
correspondence recall 0.14 → 0.29, image recall≥3 0.26 → 0.52, residual p90
8.8 px → 3.0 px. seattle_backyard: 0.80 → 0.93, 0.48 → 0.60, 0.54 → 0.69,
4.8 px → 3.4 px. This is recall and precision and warp quality moving the same
way at once, which none of the knobs in the earlier report managed: `min_inliers`
and `threshold_px` both bought warp quality with recall, and `max_leaf_checks`
bought recall with time. Here nothing is traded, because the correspondences
removed were not carrying information.

**Where it is not, nothing happens.** DinoLedge's four rows are the same row.
Every column is within one step of the baseline in either direction, which is
what 2% crowding predicts, and the arms cost 529 / 569 / 585 / 579 ms, all
within the noise of a 1.5 GB file at a 256 MiB cache. So the collapse is not a
large-capture regression waiting to happen; it is simply inert there.

**The ratio test is a second-order adjustment on top of the collapse.** Set
against the dedupe-only arm rather than the baseline: at **0.9** it is equal or
better on nine of the twelve columns that moved across the four small datasets
(kerry recall≥3 0.82 → 0.85 and residual p90 4.2 → 2.9 px, seoul recall≥3
0.52 → 0.56 and correspondence recall 0.29 → 0.30, seattle precision 0.88 → 0.90
and false candidates 0.05 → 0.02), slightly worse on `warp ok` for seoul
(0.93 → 0.91) and kerry (0.89 → 0.87), and identical on DinoLedge. At **0.8** it
tightens further and starts costing image recall (dino_dog_toy 0.69 → 0.66,
seoul 0.52 → 0.49, seattle 0.69 → 0.64) for precision (0.67 → 0.69,
0.68 → 0.71, 0.88 → 0.94) -- the same trade `min_inliers` offers, and a caller
who wants it has `min_inliers` already.

**A caution about the seoul and kerry `warp ok` figures.** That rate's
denominator is the found images carrying three or more GT correspondences, and
on those two datasets it is small: 24 baseline and 42 deduped on seoul, 53 on
kerry. The seoul move from 0.54 to 0.93 is over different and larger
populations, and its size should not be read to two digits. The direction is
corroborated by the residual percentiles, which are pooled over far more
correspondences and move the same way (p90 8.8 → 3.0 px).

---

## 4. Why the extra hits are pure loss

Three reasons, and the third is the one that makes the effect this large.

**They cannot be right.** A constellation feature is one point of one piece of
surface, and a photograph of that surface shows that point once. At most one
feature of a candidate image is its correspondence, so every member of a crowded
pair beyond the first is an outlier of whatever the right warp is. §2 confirms
the index agrees: 94 to 98% of the time the right one is the nearest.

**They are correlated outliers, not independent ones.** Every member of a pair
shares a source position. A three-point sample drawing two of them is asking for
a transform that sends one point to two places, which is not merely a bad model
but an unsolvable one, and the sampler spends the draw. Worse, a model fitted
elsewhere scores each of them separately, so a pair with six members can
contribute six votes to a wrong consensus from one point of the patch. That is
the same failure the destination-collapse guard in §4 of the earlier report was
written against, in a milder form that no determinant test catches.

**They dilute the inlier ratio cubically.** RANSAC's chance of a clean
three-point draw is `w^3` for inlier ratio `w`. On seoul_bull the collapse takes
a candidate from 98 correspondences at 13 inliers (`w` = 0.13, a clean draw 0.23% of
the time, so 200 draws find one 37% of the time) to 44 at 13 (`w` = 0.30, 2.6%,
so 200 draws find one 99.4% of the time). That is the whole of the `warp ok`
move, and it explains why the earlier report saw `iterations` help on exactly
the datasets crowding is worst on: more draws were compensating for an inlier
ratio that did not have to be that low.

---

## 5. Recommendation

**Enable the dedupe by default: `one_hit_per_image = true`.** It is the knob
with no threshold in it, it carries essentially all of the gain, and its cost is
2 to 6% of the true correspondences inside crowded pairs. On the two datasets
where crowding is worst it moves recall, precision, warp quality and residual
spread in the same direction at once, and on the one dataset where crowding is
rare it is inert in both directions. It also makes the query cheaper downstream
rather than dearer: fewer correspondences per candidate is less scoring work per
RANSAC iteration.

**Leave the ratio off by default, at `same_image_ratio = 1.0`, and document 0.9
as the tightening.** Against the dedupe it is a small further gain on precision
and false candidates and a small loss on `warp ok` for two datasets, inside the
noise of 60 patches and, on seoul and kerry, of a `warp ok` denominator in the
dozens. There is no evidence here for paying a second threshold to get it. A
caller who wants fewer and cleaner correspondences per candidate should reach
for 0.9 before 0.8: 0.8's extra precision comes out of image recall, which is
what `min_inliers` already sells more cheaply.

**What flipping the default would oblige.** The collapse halves `corr/img` on
four of five datasets, and `corr/img` is the quantity the earlier report judged
`k`, `iterations` and `min_inliers` on. Its §3 conclusion that "`k = 32` is
neutral-to-better on DinoLedge and actively harmful on a small corpus" was a
statement about crowding, and with the collapse on, most of that harm is already
removed -- k=32's extra neighbours on a 17-image corpus now mostly land in pairs
that already have a hit and are dropped. So `one_hit_per_image = true` should be
turned on together with a re-run of the `k` and `iterations` sweeps, not before
it; there is a good chance the conclusion about `k` changes sign. The same
applies to `min_inliers = 8`, whose justification was the six-inlier noise floor
of a candidate sharing nothing with the query, and that floor is partly built
out of crowded pairs.

Neither knob is turned on in this change. Both are parameters, defaulted off, so
every existing caller gets the answer it got before.

---

## Appendix: reproducing

```bash
# Four arms, one dataset. The corpora and workspaces are §1 of
# reports/exp/2026-09-14-constellation-query-eval.md.
for arm in "--multiplicity" "--one-hit-per-image" \
           "--same-image-ratio 0.8" "--same-image-ratio 0.9"; do
  pixi run -e test python scripts/kdf_constellation_eval.py \
      --workspace WS --kdf capture.kdf --sfmr WS/sfmr/solve.sfmr \
      --label NAME --patches 60 --sizes 50 --no-sweeps --seed 0 \
      --cache-mib 64 $arm --out results-NAME-$arm.json
done
```

`--multiplicity` is what produces §2's columns and is arm-independent, so it is
run once, on the baseline. `--features DIR` is needed for DinoLedge, whose
`.kdf` names images by basename. The raw per-patch results were written to the
scratch directory, not to the repository.
