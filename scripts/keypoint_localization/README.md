# Keypoint localization: alignment to the reference render against congealing

This directory compares two ways of localizing a point's keypoints across its
track views:

- **Alignment** (this branch): every view is aligned to the point's reference
  render, scored by ZNCC against that template. Search strategy `ex`
  (`SearchStrategy::Exhaustive`, the default) or `pd`
  (`SearchStrategy::PlusDescent`).
- **Congealing** (commit `83ffb08e` on `main`, before this branch): each view is
  aligned over rounds to the IRLS mean of the others, scored by a
  leave-one-out ZNCC.

Files:

- The harness is the `align_vs_congealing` example of `sfmtool-core`,
  [`crates/sfmtool-core/examples/align_vs_congealing.rs`](../../crates/sfmtool-core/examples/align_vs_congealing.rs).
  Built as it is, it calls this tree's `try_localize_patch_keypoints`
  (methods `pd` and `ex`). Built with `RUSTFLAGS="--cfg congeal"` in a copy
  of commit `83ffb08e` it calls that commit's congealing
  `try_localize_patch_keypoints` instead (method `congeal`). It writes one
  JSON file per run. It is an example target rather than a crate of its own
  so that `cargo test --workspace` and `cargo clippy --all-targets` in CI
  compile and lint it against the current localizer, and it builds with the
  workspace's `Cargo.lock`.
- `report.py` reads those JSON files and prints the accuracy and gate tables
  below, and, from the branch build's `pz` and `bz` (each view's plain and
  blur-matched score against the reference render, read as the bench reads a
  row against the stored bitmap), the gates set on each score side by side;
  those need no congealing run, and a table of the time per track by track length. The README leaves
  the time table out: the accuracy runs use 8 threads and are not a timing.
  The single-threaded timings quoted in
  `specs/core/patch/patch-keypoint-localization.md` come from the separate
  runs under "Timing".

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
  1 px). The views are the (point, slot) pairs kept by both `congeal` and `ex`
  at that displacement, without the reference slot and without views that have
  no GT projection or whose score is missing (congealing reports NaN for a few
  views). A view is good when its raw error is below 1 px and bad above
  1.5 px, using the error of the method whose score is gated, so the good and
  bad counts differ between the two columns. The relative score is the score
  divided by the median score of the run's non-reference views of the point.
  "Plain" is `ex`'s ZNCC against the template; "LOO" is congealing's
  leave-one-out ZNCC.

## Build

`W` is the repository root, `S` a scratch directory. Build from snapshots, not
from the working tree. On Windows keep `CARGO_TARGET_DIR` short: a target
directory nested under a long scratch path fails to link with LNK1104.

```bash
X=crates/sfmtool-core/examples/align_vs_congealing.rs

# Branch snapshot.
mkdir -p $S/branch-src
cp $W/Cargo.toml $W/Cargo.lock $S/branch-src/ && cp -r $W/crates $S/branch-src/
CARGO_TARGET_DIR=$S/tb pixi run --manifest-path $W/pixi.toml \
  cargo build --release -p sfmtool-core --example align_vs_congealing \
  --manifest-path $S/branch-src/Cargo.toml

# Congealing snapshot, from commit 83ffb08e, with this tree's harness copied in.
# RUSTFLAGS rebuilds every crate of the snapshot, which takes a few minutes.
mkdir -p $S/main-src
git -C $W archive 83ffb08e Cargo.toml Cargo.lock crates | tar -x -C $S/main-src
mkdir -p $S/main-src/crates/sfmtool-core/examples && cp $W/$X $S/main-src/$X
RUSTFLAGS="--cfg congeal --check-cfg cfg(congeal)" CARGO_TARGET_DIR=$S/tm \
  pixi run --manifest-path $W/pixi.toml \
  cargo build --release -p sfmtool-core --example align_vs_congealing \
  --manifest-path $S/main-src/Cargo.toml
```

The `--check-cfg` flag is there because the root `Cargo.toml` at `83ffb08e`
does not declare `cfg(congeal)`; without it the congealing build prints 7
`unexpected_cfgs` warnings, which are harmless.

The binaries are `$S/tb/release/examples/align_vs_congealing` and
`$S/tm/release/examples/align_vs_congealing`; below they are `$TB` and `$TM`.

## Run

```bash
SEOUL=$W/test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr
KERRY=$W/test-data/images/kerry_park/kerry_park_ground_truth.sfmr
for ds in seoul kerry; do
  [ $ds = seoul ] && F=$SEOUL || F=$KERRY
  for mode in displaced stored; do
    $TB $F $S/out/${ds}_${mode}_align.json \
      --ref-mode $mode --disp 0,0.5,1,2,3 --min-views 3 --threads 8
    $TM $F $S/out/${ds}_${mode}_congeal.json \
      --ref-mode $mode --disp 0,0.5,1,2,3 --min-views 3 --threads 8 \
      --references $S/out/${ds}_${mode}_align.json
  done
done
pixi run python scripts/keypoint_localization/report.py $S/out
```

The harness writes nothing next to the input. `--workspace DIR` overrides the
directory the image names are resolved against; the two ground truths do not
need it. Each run takes 0.4 s (seoul_bull) to 1.7 s (kerry_park) with 8
threads.

## Timing

The time per track in the spec is from single-threaded runs at displacement
0 with the reference stored, three of each, alternating the two builds:

```bash
COMMON="--disp 0 --ref-mode stored --threads 1"
for r in 1 2 3; do
  $TB $SEOUL $S/t/seoul-br-r$r.json $COMMON
  $TM $SEOUL $S/t/seoul-cg-r$r.json $COMMON
  # ... the same for $KERRY, and for the long-track files with
  # --workspace DIR --bins 20-49:10,50-99:10,100-199:10,200-399:10
done
```

`--bins LO-HI:N,...` keeps `N` tracks spread evenly over the eligible points
of each track-length range, `--top N` the `N` longest tracks, `--search S`
sets the search radius and `--methods` runs a subset of the methods; only the
images the kept tracks observe are decoded. Each run records each track's
time per method under `secs`. The DnDTabletop and DinoLedge reconstructions
are local, not checked in.

## Results (2026-10-09)

Branch at commit `0a7df104` (the 2-D quadratic sub-pixel fit is from
`92ad37ef`, and no commit between changes what the localizer computes for a
given strategy), against `83ffb08e`. The `pd` rows predate a later change to
the "+"-descent: a diagonal neighbour that beats the cell its walk stopped at
now moves the walk on. Run again after that change, 115 to 242 of each
file's `pd` runs differ, and the medians of the re-triangulated residual with
the reference displaced move by at most 0.022 px (kerry_park at 3 px, 1.469
to 1.447 px); the `ex` results are unchanged.
The gate tables read `ex`'s score.

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

#### kerry, reference stored, disp 0 px: 2877 views

plain (ex): n 2877, good 2810, bad 45; leave-one-out (congeal): n 2877, good 2841, bad 18.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.2% | 6.7% | 0.3% | 0.0% |
| absolute 0.5 | 0.4% | 15.6% | 0.5% | 0.0% |
| absolute 0.6 | 1.1% | 22.2% | 0.8% | 11.1% |
| relative 0.6 | 0.2% | 8.9% | 0.3% | 0.0% |
| relative 0.7 | 0.3% | 13.3% | 0.6% | 5.6% |
| relative 0.8 | 1.7% | 17.8% | 1.4% | 16.7% |

#### kerry, reference stored, disp 0.5 px: 2863 views

plain (ex): n 2863, good 2789, bad 46; leave-one-out (congeal): n 2863, good 2784, bad 26.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.1% | 8.7% | 0.3% | 3.8% |
| absolute 0.5 | 0.4% | 15.2% | 0.4% | 7.7% |
| absolute 0.6 | 1.3% | 23.9% | 0.8% | 7.7% |
| relative 0.6 | 0.2% | 10.9% | 0.3% | 3.8% |
| relative 0.7 | 0.3% | 17.4% | 0.6% | 7.7% |
| relative 0.8 | 1.5% | 17.4% | 1.4% | 15.4% |

#### kerry, reference stored, disp 1 px: 2843 views

plain (ex): n 2843, good 2759, bad 46; leave-one-out (congeal): n 2843, good 2430, bad 121.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.1% | 8.7% | 0.2% | 0.8% |
| absolute 0.5 | 0.4% | 15.2% | 0.5% | 1.7% |
| absolute 0.6 | 1.3% | 19.6% | 0.8% | 6.6% |
| relative 0.6 | 0.1% | 8.7% | 0.3% | 0.8% |
| relative 0.7 | 0.3% | 10.9% | 0.5% | 2.5% |
| relative 0.8 | 1.6% | 17.4% | 1.5% | 5.0% |

#### seoul, reference stored, disp 0 px: 872 views

plain (ex): n 872, good 781, bad 67; leave-one-out (congeal): n 872, good 781, bad 49.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.4% | 10.4% | 0.0% | 0.0% |
| absolute 0.5 | 0.6% | 13.4% | 0.4% | 4.1% |
| absolute 0.6 | 4.0% | 17.9% | 2.0% | 6.1% |
| relative 0.6 | 0.1% | 1.5% | 0.1% | 2.0% |
| relative 0.7 | 0.6% | 7.5% | 0.5% | 2.0% |
| relative 0.8 | 2.6% | 11.9% | 1.9% | 4.1% |

#### seoul, reference stored, disp 0.5 px: 878 views

plain (ex): n 878, good 786, bad 67; leave-one-out (congeal): n 878, good 776, bad 49.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.5% | 4.5% | 0.0% | 2.0% |
| absolute 0.5 | 0.6% | 9.0% | 0.4% | 2.0% |
| absolute 0.6 | 3.9% | 14.9% | 1.7% | 2.0% |
| relative 0.6 | 0.1% | 0.0% | 0.1% | 0.0% |
| relative 0.7 | 0.5% | 4.5% | 0.5% | 2.0% |
| relative 0.8 | 2.2% | 7.5% | 1.9% | 2.0% |

#### seoul, reference stored, disp 1 px: 871 views

plain (ex): n 871, good 775, bad 66; leave-one-out (congeal): n 871, good 666, bad 70.

| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |
|---|---|---|---|---|
| absolute 0.4 | 0.6% | 4.5% | 0.0% | 0.0% |
| absolute 0.5 | 0.6% | 7.6% | 0.5% | 0.0% |
| absolute 0.6 | 3.9% | 13.6% | 1.8% | 1.4% |
| relative 0.6 | 0.1% | 0.0% | 0.2% | 0.0% |
| relative 0.7 | 0.5% | 3.0% | 0.6% | 0.0% |
| relative 0.8 | 2.7% | 7.6% | 2.3% | 0.0% |

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

## The gates on the blur-matched score (2026-10-10)

The agreement gates now read each view's blur-matched score against the
reference render at its final keypoint. The branch build records, for every
kept view, `pz` and `bz`: its plain and blur-matched scores read as the bench
reads a row against the stored bitmap. With those in the output, `report.py`
prints a second set of gate tables, "Gates on each score", setting the
search's own score (`zncc`), `pz` and `bz` side by side at the same bars; it
needs no congealing run:

```bash
for ds in seoul kerry; do
  [ $ds = seoul ] && F=$SEOUL || F=$KERRY
  $TB $F $S/scores/${ds}_stored_align.json --ref-mode stored --disp 0,0.5,1 \
    --min-views 3 --threads 8 --methods ex
done
pixi run python scripts/keypoint_localization/report.py $S/scores
```

The tables and what they show are in
[`specs/core/patch/patch-keypoint-localization.md`](../../specs/core/patch/patch-keypoint-localization.md)
§ "The agreement gates on the blur-matched score". `--default-gates` keeps the
agreement gates at their defaults instead of off; the time per track it adds
is the cost of reading the blur-matched score, given in the same section
(single-threaded runs at `--disp 0 --ref-mode stored --methods ex`, with and
without the flag, alternated).
