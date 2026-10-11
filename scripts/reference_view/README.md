# Reference view: tuning the rule's parameters

The harness behind the values of `ReferenceViewOptions`, the parameters of
the reference-view rule, whose measurement is described in
[specs/core/patch/reference-view.md](../../specs/core/patch/reference-view.md)
§ "Each parameter and its reason" and § "How the values were measured". It
puts the tracks of the maintainer's hand picks, and a pool of sampled tracks,
on the bench, reads what the rule reads for each view, and sweeps the
parameters over a copy of the rule written in Python.

- `cases.json`: the 79 review cases of the hand picks (2026-10-06), one per
  line: `id`, `dataset`, `point` (the point's index in the file
  `datasets.py` names, as matched on 2026-10-11), `views` (each view's image
  name and stored keypoint, px, rounded to 0.01) and `pick`, the image of the
  hand-picked view, or `null` with a `verdict` (`"bad track"` for the two the
  maintainer marked so); `note` holds the maintainer's note where there is
  one. The review's tiles and the readings of that time are not kept.
- `datasets.py`: the ten reconstructions and their paths, the eight of
  `scripts/bench_bars/datasets.py` under this harness's keys plus
  `omni_temple` and `omni_coast` (the `clean-3-ba-embedded` solves of
  `OmniTemple1` and `OmniCoast`). Every file is read in place and never
  written. `xmas` is the 30-frame Christmas-tree subset of
  `bench_bars/datasets.py`, at `BENCH_BARS_XMAS`.
- `measure.py`: one dataset, one JSON line per track. It finds each case's
  point by its views (the point whose track holds the most of the case's
  image and keypoint pairs, within 0.05 px, or 1.5 px for `xmas`, which
  `embed_patches` rebuilt and so placed again), makes the bench track's rows
  the case's views at the case's keypoints, evaluates it, and samples
  `--pool` (60) more points of three or more observations with a fixed seed.
  Each row carries the readings the rule reads (coverage, clipped share,
  viewing angle, cell deficit, pair ZNCC, the self-similarity semi-axes), its
  `ray` (the unit direction from the patch's centre to the camera centre,
  which only the comparison of the angle factor's forms reads) and the rule's
  standing at this build's defaults. `--blur-matched` adds each
  row's pair ZNCC read from the rows' tiles, plain and blur-matched;
  `--localize` (the ground truths) aligns every view to each eligible row's
  tile in turn and records each view's distance from the point's projection;
  `--all` measures every point of three or more observations instead of the
  cases and a sample; `--eligible` counts the points of the whole file with no
  eligible view.
- `analyze.py`: the tables. It checks first that its Python copy of the rule
  gives the Rust rule's pick and costs on every track at the defaults, then
  splits the hand-picked cases in half by dataset with a seeded split, sweeps
  each parameter alone, picks a candidate per parameter on the tuning half
  (the middle of the best tied run by lenient then exact count), keeps it
  only where the held-out half confirms it, and prints the counts, the
  blur-matched agreement, the pool's changed picks, the ground truths'
  localization error and the counts of points with no eligible view. Its
  section 4 compares four forms of the angle factor, each swept the same
  way: the rule's `1 + k·tan θ`, `1 / cos^p θ`, `1 + k·tan φ` with `φ` the
  median angle between a row's `ray` and the other eligible rows', and the
  angle in quadrature the rule replaced, `√(r² + (c·tan θ)²) / r`; a form's
  starting value is the one whose picks differ least from the quadrature's
  at `σ_n = 3°` over the pool.

```bash
OUT=<a directory outside the repository>
for d in seoul_bull kerry_park; do
  pixi run python scripts/reference_view/measure.py $d --out $OUT --blur-matched --localize --eligible
  pixi run python scripts/reference_view/measure.py $d --out $OUT --all --localize
done
for d in dino altona badland mossy kerry480 omni_temple xmas omni_coast; do
  pixi run python scripts/reference_view/measure.py $d --out $OUT --blur-matched --eligible
done
pixi run python scripts/reference_view/analyze.py $OUT --out $OUT/analysis.txt
```

Build the extension from the tree being measured first (`pixi run maturin
develop --release`). The output directory must not be under `test-data` or
`C:\DataSets`; `measure.py` refuses those. The whole run takes about a minute.

## How the tables in reference-view.md were made

The tables and figures in reference-view.md § "How the values were measured"
and § "The form of the angle factor" (2026-10-11) are this harness's output,
run as above with the default seeds (`--seed 20261011` for the pool and for
the split), on a build whose rule has the angle factor `1 + k·tan θ`. The
readings do not depend on the rule, but section 1's check reads the rule's
costs and terms, so it needs a build whose rule is the one `analyze.py`
copies. All 79 cases were found:
72 exactly, every view at its keypoint and the point at its index of
2026-10-06; the 7 `xmas` cases in a subset rebuilt on 2026-10-11 under that
day's `embed_patches`, since the file of 2026-10-06 was not kept. There the
keypoints sit up to 1.4 px from the case's and four tracks hold one or two of
the case's views fewer, so `measure.py` added those views and sighted every
row at the case's keypoint, and the patch frame is the rebuild's; their
`point` in `cases.json` is the rebuild's index. Two cases, the
ones marked as bad tracks, are left out, so 77 are scored.

**Lenient** is one-sided, as the previous rule's tuning read it: the rule's
pick is at most 0.1 grid px blurrier (semi-major axis) and at most 0.05 less
typical (pair ZNCC) than the hand pick, a view as sharp and as typical as it.
On the same readings the previous rule of gates scores 70 of 77 lenient by
this reading, the figure reference-view.md gives for it, and 49 by a
two-sided reading; `analyze.py` prints the two-sided count beside it. The
**angle-bounded** count also asks that the two views' viewing angles differ
by at most 10° (`--angle-bound`).
