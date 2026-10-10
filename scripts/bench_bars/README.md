# Bench bars: measuring the track stage's ZNCC bars

The harness behind the track stage's ZNCC bars, `min_zncc` and
`min_zncc_middle` (`BENCH_MIN_ZNCC`, `BENCH_MIN_ZNCC_MIDDLE`), whose
measurement is described in
[specs/core/bench/editable-track.md](../../specs/core/bench/editable-track.md)
§ "Parameters". It plants wrong views and blurred members on tracks of real
reconstructions, reads every row's scores on the bench, and finds the bars
that best keep the members and turn out the wrong views. It reads each row's
plain and blur-matched score against the stored bitmap, so the bars can be set
on either score
([specs/drafts/sharper-patch-bitmap.md](../../specs/drafts/sharper-patch-bitmap.md)
Part 10).

- `datasets.py`: the eight reconstructions and their paths. Two are the
  checked-in ground truths; five are solves under `BENCH_BARS_DATASETS_DIR`
  (an environment variable, `C:\DataSets` when unset), and the sixth, `xmas`,
  is a 30-frame subset of a solve embedded with `embed_patches`, found at
  `BENCH_BARS_XMAS`; the module docstring says how it was made. Every file is
  read in place and never written. `--sfmr NAME=PATH` points `measure.py` at
  a file elsewhere, overriding both.
- `measure.py`: one dataset, one JSON line per row. For each of `--tracks`
  sampled points with at least four observations it puts the point on the
  bench, evaluates it with the bitmap rendered, and then plants rows in groups,
  each pinned `out` so that it does not change the other rows' readings:
  near misses (the keypoint moved 2 to 6 px), the true pixels at the projection
  in an image the track does not observe, substitutions (a square of another
  place in the same photograph, the most similar found or a random other
  point's keypoint, pasted unwarped over the row's keypoint or the point's
  projection) and blurred members (a Gaussian of sigma 1.5 or 3 source px).
  Each row carries `plain_zncc`, `plain_zncc_middle`, `blur_matched_zncc`,
  `blur_matched_zncc_middle`, the least ninth of each grid, `bitmap_blur_sigma`,
  `sharper_than_bitmap`, and the readings the geometry bars judge
  (`reprojection_error` or `projection_offset_px`, `seed_shift_px`,
  `zncc_self_similarity_radius`).
- `analyze.py`: the tables, plain and blur-matched side by side. It judges
  the rows that carry both scores, and counts toward the objective only the
  rows that clear every geometry bar (projection 3 px, seed shift 6 px,
  self-similarity radius 2.5). The objective is the mean over tracks of the
  average of the members kept (blurred members included) and the substitutions
  in observed images turned out. It evaluates the grid of whole bar 0.50 to
  0.85 in steps of 0.05 with the middle bar off or from 0.30 up to the whole
  bar, picks by leave-one-reconstruction-out, and prints: the counts, the
  geometry bars' shares, the turned-out and lost shares by similarity band, the
  per-reconstruction means, the held-out picks for each score, and the same
  picks on four other sets of rows (sections 4b).

```bash
OUT=<a directory outside the repository>
for d in seoul_gt kerry_gt kerry480 badland mossy altona dino xmas; do
  pixi run python scripts/bench_bars/measure.py $d --tracks 150 --out $OUT
done
pixi run python scripts/bench_bars/analyze.py $OUT --out $OUT/analysis.txt
```

Each dataset is one process, so a long run can be split across machines or
restarted per dataset. Build the extension from the tree being measured first
(`pixi run maturin develop --release`). The output directory must not be under
`test-data` or `C:\DataSets`; `measure.py` refuses those. The whole run takes
about 20 to 30 minutes on one machine; most of it is the dino and Christmas
tree reconstructions. The sampling seed is fixed (`--seed`, default
20261009), so the same build and data give the same rows.

## How the tables in editable-track.md were made

The tables in § "Parameters" (2026-10-10) are this harness's: `measure.py`
on the eight datasets at 150 tracks each with the default seed, unsharded, on
a build from before the bars were switched to the blur-matched score (the
measurement reads both scores and neither depends on the bars), then
`analyze.py`: sections 2 and 3 give the shares and per-reconstruction means,
section 4 the held-out picks and the objective at the fixed bars, section 4b
the other sets of rows, and section 5 the least member scores (the badlands
note). An earlier measurement of the same design, on the plain score beside
the leave-one-out ZNCC while the localizer still congealed, set the plain
`0.65` bar the blur-matched `0.70` replaced; its scripts were not checked in.

sfmtool has no public reader for a workspace image, so `datasets.read_image`
reads it with OpenCV as the package's own loader does (`workspace_dir /
image_name`, converted from BGR to RGB).
