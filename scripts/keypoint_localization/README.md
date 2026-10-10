# Keypoint localization: alignment to the reference render against congealing

This directory compares two ways of localizing a point's keypoints across its
track views:

- **Alignment** (this branch): every view is aligned to the point's reference
  render, scored by ZNCC against that template. Search strategy `pd`
  (`SearchStrategy::PlusDescent`, the default) or `ex`
  (`SearchStrategy::Exhaustive`).
- **Congealing** (commit `83ffb08e` on `main`, before this branch): each view is
  aligned over rounds to the IRLS mean of the others, scored by a
  leave-one-out ZNCC.

Files:

- `align_vs_congealing/` is a standalone Cargo crate (its own `[workspace]`
  table, so it is outside the main workspace) with one source file,
  `src/main.rs`. Built without features it calls this branch's
  `try_localize_patch_keypoints` (methods `pd` and `ex`); built with
  `--features congeal` it calls the congealing `try_localize_patch_keypoints`
  of `83ffb08e` (method `congeal`). It writes one JSON file per run.
- `report.py` reads those JSON files and prints the markdown tables below.

## Protocol

The poses stored in the reconstruction are taken as ground truth. For each
point with at least 3 track views (every such point; no cap), the ground-truth
(GT) keypoint of a view is the projection of the patch centre at the stored
pose. The localizer gates are off (`min_absolute_zncc = 0`,
`min_relative_zncc = 0`, `max_shift_px = 1e9`); the member gate stays at its
default, and so do all other parameters (for congealing this includes the
consensus-basis cap of 8 views).

- **Reference.** The point's stored reference observation; where the file has
  none (`-1`), the slot the branch's reference-view rule picks at the stored
  keypoints. The branch run writes `{point: reference slot}` under
  `references` in its output, and the congealing run reads that file with
  `--references`, so both use the same reference slot. Congealing does not
  align to the reference; the slot only decides the reference's seed in
  `stored` mode and which view the gate tables leave out.
- **Seeds.** At displacement 0 every view starts at its stored keypoint. At
  displacement `d > 0` a view starts at its GT keypoint plus `d` px in a
  direction from a hash of (point, displacement index, slot). With
  `--ref-mode displaced` the reference is displaced like every other view;
  with `--ref-mode stored` it keeps its stored keypoint, which gives alignment
  an undisplaced template and so favours it (the earlier protocol).
- **Errors.** `err` is the distance in px from the method's keypoint to the GT
  keypoint. `retri` is the residual in px after re-triangulating the point from
  the method's kept views at the stored poses (Gauss-Newton from the patch
  centre). In `displaced` mode every view aligned to the reference inherits
  the reference's seed offset, so the raw error of alignment is close to `d`
  even when the views agree with each other; the re-triangulated residual
  measures that agreement and is the fair measure there.
- **Side-peak share** is the share of kept views with raw error above 1.5 px.
  The last column counts tracks whose mean re-triangulated residual exceeds
  5 px, which flags Gauss-Newton failures that inflate the means.
- **Gate tables** use `stored` mode and one table per displacement (0, 0.5,
  1 px). The views are the (point, slot) pairs kept by both `congeal` and `pd`
  at that displacement, without the reference slot and without views that have
  no GT projection or whose score is missing (congealing reports NaN for a few
  views). A view is good when its raw error is below 1 px and bad above
  1.5 px, using the error of the method whose score is gated, so the good and
  bad counts differ between the two columns. The relative score is the score
  divided by the median score of the run's non-reference views of the point.
  "Plain" is `pd`'s ZNCC against the template; "LOO" is congealing's
  leave-one-out ZNCC.

## Build

`W` is the repository root, `S` a scratch directory. Build from snapshots, not
from the working tree. On Windows keep `CARGO_TARGET_DIR` short: a target
directory nested under a long scratch path fails to link with LNK1104.

```bash
H=scripts/keypoint_localization/align_vs_congealing

# Branch snapshot.
mkdir -p $S/branch-src/scripts/keypoint_localization
cp $W/Cargo.toml $W/Cargo.lock $S/branch-src/ && cp -r $W/crates $S/branch-src/
cp -r $W/$H $S/branch-src/scripts/keypoint_localization/
CARGO_TARGET_DIR=$S/tb pixi run --manifest-path $W/pixi.toml \
  cargo build --release --manifest-path $S/branch-src/$H/Cargo.toml

# Congealing snapshot, from commit 83ffb08e.
mkdir -p $S/main-src/scripts/keypoint_localization
git -C $W archive 83ffb08e Cargo.toml Cargo.lock crates | tar -x -C $S/main-src
cp -r $W/$H $S/main-src/scripts/keypoint_localization/
CARGO_TARGET_DIR=$S/tm pixi run --manifest-path $W/pixi.toml \
  cargo build --release --features congeal --manifest-path $S/main-src/$H/Cargo.toml
```

## Run

```bash
SEOUL=$W/test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr
KERRY=$W/test-data/images/kerry_park/kerry_park_ground_truth.sfmr
for ds in seoul kerry; do
  [ $ds = seoul ] && F=$SEOUL || F=$KERRY
  for mode in displaced stored; do
    $S/tb/release/align-vs-congealing $F $S/out/${ds}_${mode}_align.json \
      --ref-mode $mode --disp 0,0.5,1,2,3 --min-views 3 --threads 8
    $S/tm/release/align-vs-congealing $F $S/out/${ds}_${mode}_congeal.json \
      --ref-mode $mode --disp 0,0.5,1,2,3 --min-views 3 --threads 8 \
      --references $S/out/${ds}_${mode}_align.json
  done
done
python scripts/keypoint_localization/report.py $S/out
```

The harness writes nothing next to the input. `--workspace DIR` overrides the
directory the image names are resolved against; the two ground truths do not
need it. Each run takes 0.4 s (seoul_bull) to 1.7 s (kerry_park) with 8
threads.

## Results (2026-10-09)

Branch at commit `92ad37ef` plus the working-tree changes of that day (the 2-D
quadratic sub-pixel fit), against `83ffb08e`.

### Accuracy

#### kerry, reference displaced: 380 tracks

| disp px | method | views kept | raw mean / median | retri mean / median | side-peak share | tracks retri mean > 5 px |
|---|---|---|---|---|---|---|
| 0 | congeal | 3310/3745 | 0.250 / 0.180 | 0.207 / 0.160 | 0.6% | 0 |
| 0 | pd | 3276/3745 | 0.247 / 0.174 | 0.202 / 0.156 | 0.9% | 0 |
| 0 | ex | 3279/3745 | 0.271 / 0.175 | 0.232 / 0.157 | 1.4% | 1 |
| 0.5 | congeal | 3298/3745 | 0.365 / 0.277 | 0.216 / 0.164 | 0.9% | 0 |
| 0.5 | pd | 3255/3745 | 0.519 / 0.493 | 0.204 / 0.158 | 1.9% | 0 |
| 0.5 | ex | 3258/3745 | 0.545 / 0.495 | 0.236 / 0.159 | 2.4% | 1 |
| 1 | congeal | 3277/3745 | 0.746 / 0.659 | 0.252 / 0.177 | 7.2% | 0 |
| 1 | pd | 3251/3745 | 0.915 / 0.894 | 2.469 / 0.163 | 7.2% | 1 |
| 1 | ex | 3253/3745 | 0.931 / 0.897 | 0.240 / 0.160 | 7.2% | 0 |
| 2 | congeal | 3268/3745 | 1.580 / 1.483 | 0.672 / 0.290 | 49.2% | 2 |
| 2 | pd | 3264/3745 | 1.897 / 1.826 | 4.387 / 0.437 | 64.4% | 3 |
| 2 | ex | 3265/3745 | 1.780 / 1.733 | 0.527 / 0.254 | 60.5% | 1 |
| 3 | congeal | 3260/3745 | 2.380 / 2.255 | 2.185 / 0.751 | 73.9% | 5 |
| 3 | pd | 3220/3745 | 2.984 / 2.970 | 1.942 / 1.469 | 86.7% | 2 |
| 3 | ex | 3222/3745 | 2.781 / 2.698 | 1.297 / 0.712 | 83.8% | 1 |

#### kerry, reference stored: 380 tracks

| disp px | method | views kept | raw mean / median | retri mean / median | side-peak share | tracks retri mean > 5 px |
|---|---|---|---|---|---|---|
| 0 | congeal | 3310/3745 | 0.250 / 0.180 | 0.207 / 0.160 | 0.6% | 0 |
| 0 | pd | 3276/3745 | 0.247 / 0.174 | 0.202 / 0.156 | 0.9% | 0 |
| 0 | ex | 3279/3745 | 0.271 / 0.175 | 0.232 / 0.157 | 1.4% | 1 |
| 0.5 | congeal | 3298/3745 | 0.358 / 0.268 | 0.216 / 0.165 | 0.9% | 0 |
| 0.5 | pd | 3255/3745 | 0.253 / 0.181 | 0.203 / 0.158 | 0.9% | 0 |
| 0.5 | ex | 3258/3745 | 0.279 / 0.181 | 0.237 / 0.159 | 1.5% | 0 |
| 1 | congeal | 3278/3745 | 0.645 / 0.554 | 0.254 / 0.177 | 5.0% | 0 |
| 1 | pd | 3251/3745 | 0.263 / 0.182 | 0.210 / 0.158 | 1.0% | 0 |
| 1 | ex | 3253/3745 | 0.292 / 0.182 | 0.242 / 0.160 | 1.4% | 0 |
| 2 | congeal | 3273/3745 | 1.366 / 1.237 | 3.566 / 0.277 | 38.9% | 3 |
| 2 | pd | 3264/3745 | 0.466 / 0.210 | 3.370 / 0.223 | 7.9% | 2 |
| 2 | ex | 3265/3745 | 0.367 / 0.201 | 0.319 / 0.183 | 3.6% | 0 |
| 3 | congeal | 3264/3745 | 2.005 / 1.859 | 4.894 / 0.550 | 61.0% | 3 |
| 3 | pd | 3220/3745 | 1.156 / 0.332 | 1.205 / 0.571 | 24.6% | 1 |
| 3 | ex | 3222/3745 | 0.681 / 0.278 | 0.635 / 0.301 | 10.7% | 0 |

#### seoul, reference displaced: 259 tracks

| disp px | method | views kept | raw mean / median | retri mean / median | side-peak share | tracks retri mean > 5 px |
|---|---|---|---|---|---|---|
| 0 | congeal | 1115/1235 | 0.559 / 0.306 | 0.357 / 0.261 | 5.9% | 0 |
| 0 | pd | 1132/1235 | 0.509 / 0.284 | 0.428 / 0.263 | 5.9% | 2 |
| 0 | ex | 1133/1235 | 0.607 / 0.284 | 0.436 / 0.264 | 6.7% | 2 |
| 0.5 | congeal | 1120/1235 | 0.622 / 0.414 | 0.349 / 0.271 | 5.8% | 0 |
| 0.5 | pd | 1137/1235 | 0.740 / 0.500 | 0.340 / 0.265 | 7.2% | 0 |
| 0.5 | ex | 1139/1235 | 0.787 / 0.501 | 0.369 / 0.272 | 7.7% | 0 |
| 1 | congeal | 1113/1235 | 0.904 / 0.754 | 0.355 / 0.279 | 10.2% | 0 |
| 1 | pd | 1131/1235 | 1.135 / 1.000 | 0.346 / 0.262 | 13.1% | 0 |
| 1 | ex | 1133/1235 | 1.176 / 1.000 | 0.373 / 0.263 | 14.0% | 0 |
| 2 | congeal | 1111/1235 | 1.762 / 1.655 | 0.522 / 0.316 | 58.7% | 0 |
| 2 | pd | 1130/1235 | 2.117 / 2.000 | 0.623 / 0.331 | 87.0% | 0 |
| 2 | ex | 1137/1235 | 2.079 / 2.000 | 0.439 / 0.292 | 86.4% | 0 |
| 3 | congeal | 1111/1235 | 2.727 / 2.644 | 0.954 / 0.427 | 88.5% | 0 |
| 3 | pd | 1134/1235 | 3.158 / 3.000 | 1.276 / 0.633 | 95.9% | 2 |
| 3 | ex | 1138/1235 | 3.022 / 3.000 | 0.796 / 0.412 | 95.9% | 1 |

#### seoul, reference stored: 259 tracks

| disp px | method | views kept | raw mean / median | retri mean / median | side-peak share | tracks retri mean > 5 px |
|---|---|---|---|---|---|---|
| 0 | congeal | 1115/1235 | 0.559 / 0.306 | 0.357 / 0.261 | 5.9% | 0 |
| 0 | pd | 1132/1235 | 0.509 / 0.284 | 0.428 / 0.263 | 5.9% | 2 |
| 0 | ex | 1133/1235 | 0.607 / 0.284 | 0.436 / 0.264 | 6.7% | 2 |
| 0.5 | congeal | 1120/1235 | 0.608 / 0.377 | 0.357 / 0.267 | 6.0% | 0 |
| 0.5 | pd | 1137/1235 | 0.497 / 0.288 | 0.350 / 0.261 | 5.9% | 0 |
| 0.5 | ex | 1139/1235 | 0.596 / 0.289 | 0.358 / 0.265 | 6.6% | 0 |
| 1 | congeal | 1114/1235 | 0.796 / 0.614 | 0.363 / 0.278 | 8.3% | 0 |
| 1 | pd | 1131/1235 | 0.510 / 0.290 | 0.363 / 0.265 | 5.8% | 0 |
| 1 | ex | 1133/1235 | 0.604 / 0.289 | 0.359 / 0.269 | 6.6% | 0 |
| 2 | congeal | 1115/1235 | 1.326 / 1.139 | 0.459 / 0.291 | 31.7% | 0 |
| 2 | pd | 1130/1235 | 0.529 / 0.291 | 0.394 / 0.270 | 6.9% | 0 |
| 2 | ex | 1137/1235 | 0.612 / 0.290 | 0.391 / 0.265 | 6.9% | 0 |
| 3 | congeal | 1116/1235 | 1.945 / 1.799 | 0.720 / 0.377 | 60.3% | 0 |
| 3 | pd | 1134/1235 | 0.759 / 0.318 | 0.635 / 0.329 | 11.7% | 1 |
| 3 | ex | 1138/1235 | 0.636 / 0.306 | 0.407 / 0.275 | 7.4% | 0 |

### Gates (reference stored)

#### kerry, reference stored, disp 0 px: 2876 views

plain (pd): n 2876, good 2821, bad 28; leave-one-out (congeal): n 2876, good 2840, bad 18.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.2% | 3.6% | 0.3% | 0.0% |
| absolute 0.5 | 0.5% | 7.1% | 0.5% | 0.0% |
| absolute 0.6 | 1.3% | 14.3% | 0.8% | 11.1% |
| relative 0.6 | 0.3% | 0.0% | 0.3% | 0.0% |
| relative 0.7 | 0.5% | 7.1% | 0.6% | 5.6% |
| relative 0.8 | 1.9% | 7.1% | 1.4% | 16.7% |

#### kerry, reference stored, disp 0.5 px: 2862 views

plain (pd): n 2862, good 2804, bad 28; leave-one-out (congeal): n 2862, good 2783, bad 26.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.2% | 3.6% | 0.3% | 3.8% |
| absolute 0.5 | 0.5% | 7.1% | 0.4% | 7.7% |
| absolute 0.6 | 1.5% | 17.9% | 0.8% | 7.7% |
| relative 0.6 | 0.3% | 3.6% | 0.3% | 3.8% |
| relative 0.7 | 0.4% | 7.1% | 0.6% | 7.7% |
| relative 0.8 | 1.7% | 7.1% | 1.4% | 15.4% |

#### kerry, reference stored, disp 1 px: 2843 views

plain (pd): n 2843, good 2768, bad 32; leave-one-out (congeal): n 2843, good 2430, bad 121.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.4% | 6.2% | 0.2% | 0.8% |
| absolute 0.5 | 0.7% | 9.4% | 0.5% | 1.7% |
| absolute 0.6 | 1.7% | 18.8% | 0.8% | 6.6% |
| relative 0.6 | 0.3% | 6.2% | 0.3% | 0.8% |
| relative 0.7 | 0.5% | 9.4% | 0.5% | 2.5% |
| relative 0.8 | 1.9% | 12.5% | 1.5% | 5.0% |

#### seoul, reference stored, disp 0 px: 872 views

plain (pd): n 872, good 784, bad 59; leave-one-out (congeal): n 872, good 781, bad 49.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.4% | 11.9% | 0.0% | 0.0% |
| absolute 0.5 | 0.9% | 15.3% | 0.4% | 4.1% |
| absolute 0.6 | 4.2% | 20.3% | 2.0% | 6.1% |
| relative 0.6 | 0.3% | 1.7% | 0.1% | 2.0% |
| relative 0.7 | 0.8% | 8.5% | 0.5% | 2.0% |
| relative 0.8 | 2.7% | 13.6% | 1.9% | 4.1% |

#### seoul, reference stored, disp 0.5 px: 878 views

plain (pd): n 878, good 794, bad 59; leave-one-out (congeal): n 878, good 776, bad 49.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.6% | 10.2% | 0.0% | 2.0% |
| absolute 0.5 | 1.0% | 11.9% | 0.4% | 2.0% |
| absolute 0.6 | 4.4% | 15.3% | 1.7% | 2.0% |
| relative 0.6 | 0.3% | 0.0% | 0.1% | 0.0% |
| relative 0.7 | 0.6% | 5.1% | 0.5% | 2.0% |
| relative 0.8 | 2.5% | 6.8% | 1.9% | 2.0% |

#### seoul, reference stored, disp 1 px: 871 views

plain (pd): n 871, good 781, bad 58; leave-one-out (congeal): n 871, good 666, bad 70.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.6% | 10.3% | 0.0% | 0.0% |
| absolute 0.5 | 0.9% | 12.1% | 0.5% | 0.0% |
| absolute 0.6 | 4.4% | 19.0% | 1.8% | 1.4% |
| relative 0.6 | 0.3% | 0.0% | 0.2% | 0.0% |
| relative 0.7 | 0.6% | 3.4% | 0.6% | 0.0% |
| relative 0.8 | 2.9% | 6.9% | 2.3% | 0.0% |

### Notes on these results

- Some points have no reference slot: their stored reference is `-1` and the
  rule pick at the stored keypoints returned none (3 of 259 on seoul_bull, 6 of
  380 on kerry_park). Alignment then picks its own reference from the seeds.
- In `displaced` mode on seoul_bull, `pd` and `ex` report a NaN score for some
  views of 3 points at every displacement above 0 (never at 0, never in
  `stored` mode). Congealing reports NaN leave-one-out scores for 26 to 251
  views per file in both modes. The gate tables leave these views out.
- A few Gauss-Newton re-triangulations fail to converge and leave residuals
  well above 5 px; one such track can raise a mean by more than 2 px
  (for example `pd` on kerry_park, `displaced`, 1 px: mean 2.469 px, median
  0.163 px). Compare the medians, and read the means with the last column.
