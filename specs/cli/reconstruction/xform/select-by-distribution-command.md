# Select by Distribution

This document specifies the `--include-by-distribution` option for `sfm xform`, which keeps a
small, strategically-chosen subset of a reconstruction's cameras instead of decimating blindly.

## Motivation

Carving a small, committable sample dataset (for example, a fisheye example dataset for the repo)
out of a large capture. Blind decimation — every Nth frame, a random N — discards the structure we
already have. With the full 3D reconstruction in hand we can do better: spread the picks so the
*point cloud* stays covered (the subset is not a zoomed-in corner of the capture), and at each part
of it keep views from genuinely different directions so the 3D tracks stay well-triangulated and the
subset solves cleanly on its own.

The objective is geometric, expressed against the reconstructed cloud — not against the covisibility
graph. A covisibility-spread heuristic ("pick cameras that share few observed points with the ones
already picked") drifts toward thinly-reconstructed pockets and clumps picks there; it has no notion
of "this region is already covered, stop spending budget on it." Driving selection off the cloud
itself avoids that.

This is a filter in the `sfm xform` pipeline, so it composes with the existing operations: its
position on the command line determines when it runs relative to geometric transforms, point
filters, bundle adjustment, etc.

## Command Syntax

```bash
sfm xform <input.sfmr> [<output.sfmr>] --include-by-distribution <COUNT>[,verbose]
```

- `COUNT` — target number of views to keep. Must be `>= 2` (a `UsageError` otherwise). If `COUNT` is
  at least the number of selectable units present, the filter is a no-op and prints a notice.
- `verbose` — optional modifier; prints a per-step trace (see [Verbose mode](#verbose-mode)).

There is no strategy argument today. If a variant is ever wanted — e.g. a sparse "seeds only" mode,
see [Open questions](#open-questions) — it would arrive as a further modifier alongside `verbose`.

### Selection unit: images vs. rig frames

If the reconstruction carries rig frame data, the unit of selection is the rig **frame** — both
sensors of a 360° fisheye pair are kept or dropped together, never split. Otherwise the unit is the
individual image. In both cases `COUNT` counts *units*. For a rig-frame unit:

- the points it *observes* are the union of the points observed by its member images;
- its *viewing ray toward a point `p`* is the ray of whichever member image observes `p` — the
  opposite-facing sensors of a 360° rig almost never both see the same world point; if they somehow
  do, use the reference sensor's ray;
- it has no single center; the algorithm uses per-unit viewing rays toward points, never camera
  centers, so none is needed.

### The point cloud is taken as given

The filter optimizes coverage of the point cloud it receives. Outliers, floaters, and barely-seen
single-observation tracks count toward "the cloud" exactly as they are — a stray mis-triangulated
point far from everything else will look like an under-covered region and pull a pick toward it.
Cleaning the cloud is the caller's job and composes naturally in the pipeline; chain the point
filters *before* `--include-by-distribution`:

```bash
sfm xform recon.sfmr sample.sfmr \
  --filter-by-reprojection-error 2.0 --remove-isolated 5,p95 --remove-short-tracks 2 \
  --include-by-distribution 16 --remove-short-tracks 1
```

(The trailing `--remove-short-tracks 1` is separate — see [Stranded points](#stranded-points).)

### Stranded points

A point's coverage value only counts once it has two selected observers (below), so the selection
never *wants* a single-observation track — but a unit selected to cover one point will incidentally
keep its other, less-covered points, some of which can end up with one surviving observation. This
filter does not remove them; chain `--remove-short-tracks 1` afterward (consistent with
`--include-range` and friends).

### Example

```bash
# Keep 16 well-distributed rig frames, then drop single-observation points.
sfm xform sfmr/recon.sfmr sample.sfmr --include-by-distribution 16 --remove-short-tracks 1
```

## Algorithm

One greedy loop. A **farthest-point step** decides *where* to spend the next slice of budget — it
heads to whichever part of the cloud is currently worst-covered. An **angular-thinning add** decides
*which* units to take there — the observers of that target that open genuinely new viewing angles on
it, and no near-duplicates. The loop runs until `COUNT` units are selected.

The two moves pull on different axes and that is the point: the farthest-point step optimizes spatial
*extent* (no region of the cloud left uncovered), the angular thinning optimizes per-region *quality*
(every covered point sees real parallax, no budget wasted on near-identical views).

### "Well-covered"

A reconstructed point `p` is **well-covered by `S`** (the selected set) when at least two units in
`S` observe it *and* the largest angle between any two of their viewing rays toward `p` is at least
`H` (the thinning angle; default under [Parameters](#parameters)). Equivalently: `p` has two
selected observers far enough apart to triangulate it decently. Points with zero or one selected
observer, or whose selected observers are all bunched within `H`, are **not** well-covered. Let
`Cov(S)` be the set of well-covered points.

### Step 0 — seed from the widest triangulation angle

Pick the point with the widest *triangulation angle in the full reconstruction* — the largest angle
between any two of its observers' rays toward it (computed once, over all images, before any
selection). Among that point's observers, walk them in order of decreasing observation count (ties by
unit index) and greedily keep a unit if its ray toward the point is at least `H` away from the ray of
every unit kept so far. Stop when the kept set reaches `⌈COUNT / 3⌉` — a popular point seen across a
wide arc must not consume the whole budget. Add the kept units to `S`.

This bootstraps `S` with a small, angularly-spread cluster around the single point the data can
triangulate best — a guaranteed well-conditioned anchor, and (because the kept units span a wide arc
around it) not a "cluster" in the bad sense.

### Step N — farthest-point step, then thinned add

Repeat until `|S| = COUNT`:

1. **Find the target.** Consider every reconstructed point that (a) still has at least one
   *unselected* observer **and** (b) has a full triangulation angle `≥ H` — a point whose observers
   never span `H` can't be made well-covered, so visiting it would only burn budget (the verbose
   trace makes this failure mode obvious; see [Verbose mode](#verbose-mode)). Among those, take the
   one whose nearest well-covered point (Euclidean, in reconstruction coordinates) is farthest away —
   the point in the worst-covered neighborhood that selection can still do something useful about.
   (If `Cov(S)` is empty — only possible right after a degenerate seed — the target is instead the
   widest-triangulation-angle point among the candidates.) Ties → highest such distance, then lowest
   point id. If no point satisfies both (a) and (b), skip to
   [When the cloud is covered](#when-the-cloud-is-covered-before-the-budget-is-spent).
2. **Thinned add.** Walk the target point's *unselected* observers in order of decreasing observation
   count (ties by unit index) and greedily keep a unit if its ray toward the target is at least `H`
   away from the ray of every unit that *already observes the target in `S`* and every unit kept so
   far in this step. Add the kept units to `S`, but never more than the remaining budget
   (`COUNT − |S|`); if the budget allows fewer than the thinning would keep, take the highest-count
   ones (the order the walk already produces).

If thinning keeps nothing — every unselected observer is within `H` of a ray already on the target —
keep the single observer whose ray toward the target makes the *largest* angle with `S`'s existing
rays on it (ties → most observations, then lowest unit index). That is the one that moves the
target's `parallax_S` up the fastest, so the point reaches well-covered in as few revisits as
possible rather than the target being re-picked many times while observers trickle in one at a time.
The loop cannot stall: every iteration's target has an unselected observer, so `|S|` strictly
increases.

### When the cloud is covered before the budget is spent

If no point is still both improvable (has an unselected observer) and well-coverable (full
triangulation angle `≥ H`) — either everything coverable is covered, or what's left can't be
triangulated — the farthest-point step has nothing useful to do. Remaining picks then go,
deterministically, to the unselected unit that observes the most not-yet-well-covered points
(ties → most observations, then lowest unit index). This fills the budget to exactly `COUNT`; it is
not an error, and `COUNT` is always a hard target, never exceeded and never undershot (except the
no-op case where `COUNT ≥ #units`).

### Verbose mode

`--include-by-distribution <COUNT>,verbose` prints a one-row-per-step trace to stdout — useful for
seeing where the budget went and for catching pathologies (it was the verbose trace that surfaced
the "keeps re-targeting an un-triangulatable point" bug fixed by the `tri_angle ≥ H` gate above).
Columns:

| column | meaning |
|--------|---------|
| `step` | `seed`, then `1, 2, …` for each farthest-point step, then `fill` if the budget loop ran |
| `\|S\|` | selected units so far (after this step) |
| `well-cov` | points that are well-covered (≥ 2 selected observers spanning ≥ `H`) |
| `has-unsel` | points that still have at least one unselected observer |
| `targetable` | of those, how many also have full triangulation angle ≥ `H` (the candidate set the farthest-point step picks from); `--` on the seed/fill rows |
| `target_pt` | the point id chosen this step (`--` on the fill row) |
| `dist/diag` | that point's distance to the nearest well-covered point, as a fraction of the cloud's bounding-box diagonal |
| `tgt_ang` | that point's full triangulation angle, in degrees |
| `+units` | units added this step |

A header line above the table also reports `H`, the total point count, how many points have full
triangulation angle ≥ `H`, and the bounding-box diagonal used for normalization.

### Incremental state and cost

The expensive structure, built once: the observation adjacency — `obs(p)` (units observing each
point) and `pts(u)` (points each unit observes) — plus, per point, its full triangulation angle over
all images (for the seed and the `Cov(S)`-empty fallback). `O(#observations)`.

The loop's state:

- A k-d tree over `Cov(S)`'s 3D positions, plus a per-cloud-point cached "distance to nearest
  well-covered point." When a unit is added, only the points it observes can gain a second/farther
  selected observer, so only those points can newly enter `Cov(S)`; insert each new well-covered
  point into the tree, then refresh the cached distance only for cloud points whose current cached
  distance exceeds the distance to the just-inserted point (a bounded radius query). Periodically
  rebuilding the tree from scratch when `Cov(S)` has grown a lot is fine — the spec does not mandate
  an incremental k-d tree.
- Per point in `Cov(S)`-eligible state: the list of its selected observers' rays toward it and the
  current max pairwise angle (so the well-covered test is O(1) to recheck on each new observer).

The farthest-point step is one pass over the cloud's cached distances; the thinned add is a pass over
one point's observers. Over a whole run that is observation-touching work, not `#units²` — seconds,
not minutes, for a six-figure-image capture with `COUNT` in the tens to low hundreds.

### Determinism

For a given input `.sfmr` and build, selection is fully deterministic:

- The seed is the widest-triangulation-angle point (ties by point id), then its observers in
  decreasing-observation-count order (ties by unit index), thinned by `H`, capped at `⌈COUNT/3⌉`.
- Each farthest-point step is an `argmax` of cached distance over the candidate points — those with
  an unselected observer and full triangulation angle `≥ H` (ties by point id). Each thinned add
  walks observers in decreasing-observation-count order (ties by unit index); the no-new-angle
  fallback is `argmax` of the angle to `S`'s existing rays on the target (ties: most observations,
  then unit index).
- All arithmetic is on the data in the file (no RNG, no wall-clock, no order-dependent iteration over
  unordered containers).

(Floating-point distances and angles are reproducible for a given build/platform; the spec does not
promise bit-identical selection across differing BLAS/CPU configurations, only that there is no
intentional nondeterminism.)

### Approximation

No approximation guarantee. The farthest-point step is the classic 2-approximate k-center move
adapted to a coverage target that itself shifts as units are added; the angular thinning is a
greedy dispersion within a point's observer set. Both are sound, well-behaved heuristics; neither
carries a proven ratio for the combined objective (which is not submodular — a point contributes
nothing until its *second* selected observer arrives, by design).

### Parameters

Not exposed on the CLI initially; hard-coded:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `H` | `20°` | Thinning / well-covered angle — two observers of a point count as "different directions" once their rays toward it are this far apart; the same threshold gates a point becoming well-covered. |

See [Open questions](#open-questions) on whether to expose `H` and whether the well-covered gate and
the observer-thinning gate should be one knob or two.

## Implementation Notes

- `SelectByDistributionFilter(count, verbose=False)` in
  `src/sfmtool/xform/_select_by_distribution.py`, exported from `src/sfmtool/xform/__init__.py`, and
  wired into `src/sfmtool/_commands/xform.py` (a `@click.option`, the help text, and the "at least
  one transform" error list) plus a branch in `parse_transform_args`
  (`src/sfmtool/xform/_arg_parser.py`) that splits the arg on `,` — `COUNT` then optional `verbose`.
  `COUNT < 2` must surface as a `click.UsageError`. Its `description()`
  should follow precedent, e.g. `"Select N cameras by distribution"`.
- The module only computes *which images to keep*; it then delegates to `_filter_images()` from
  `_filter_by_image_range.py` for the track/point/rig-frame bookkeeping and index remapping.
- Camera centers (needed only to form viewing rays) from `quaternions_wxyz` / `translations` as
  `C = -R(q)^T t` (COLMAP world-to-camera); reuse `_compute_camera_centers`. A unit's viewing ray
  toward point `p` is `(positions[p] − center)` normalized; the angle between two rays is `arccos` of
  their clamped dot product.
- Use a k-d tree for the "nearest well-covered point" queries. The repo already does nearest-neighbor
  point queries elsewhere (`--remove-isolated`, `sfmtool-core` spatial indexing) — prefer that
  machinery over a hand-rolled tree. A from-scratch rebuild whenever `Cov(S)` grows past, say, 1.5×
  its size at last rebuild is an acceptable first implementation.
- Comfortable at hundreds of thousands of images: the per-unit/per-point incremental state is small;
  the one big structure is the static observation adjacency.

## Scope

This filter changes which images the `.sfmr` references. It does **not** modify image files or
`.sift` feature files on disk. Producing the actual small dataset on disk — downscaling images,
rescaling intrinsics and 2D observations (or re-extracting features at the new resolution), and
copying only the surviving images and features — is out of scope for this option and would be a
separate command or workflow step.

## Testing

- **Clustered fixture.** A reconstruction whose points form `K` clearly separate spatial clusters.
  With `COUNT ≥ K + 1` the selection puts well-covered points in *every* cluster (assert: each
  cluster contains at least one well-covered point, i.e. one observed by ≥2 selected units across
  ≥ `H`). This is the regression test against the "selects a clump, misses whole regions" failure
  the covisibility-spread design exhibited.
- **Video / path fixture.** Cameras densely sampled along a 1-D path, all observing one scene. The
  selection covers the whole extent of the cloud, not a localized blob, and within each visited
  stretch the kept units see shared points across real baselines (no run of adjacent near-duplicate
  frames). Regression test for the "localized zoomed-in subset" and "prefers near-duplicates" modes
  together.
- **Coverage-quality check on a volumetric fixture.** Compared with naive every-Nth selection at the
  same `COUNT`, the selection yields more well-covered points and a larger covered spatial extent
  (e.g. bounding-box volume of well-covered points, or fraction of cloud within `r` of a well-covered
  point). Comparative property, not a universal guarantee.
- **Determinism.** Running the same selection twice on the same input gives identical kept-image
  lists; a hand-checked tiny fixture gives the expected seed and the expected pick order for the
  first few farthest-point steps (asserted explicitly).
- **Rig reconstruction.** Selection keeps both sensors of every chosen frame and `rig_frame_data` is
  re-indexed correctly (covered transitively by the existing `_filter_images` rig tests); a frame's
  observed points and its ray toward a point follow the union / observing-member rules from
  [Selection unit](#selection-unit-images-vs-rig-frames).
- **Edge cases.** `COUNT ≥ #units` → no-op with notice; `COUNT < 2` → `UsageError`; tiny cloud where
  the seed alone (capped at `⌈COUNT/3⌉`) plus one farthest-point step already reaches `COUNT`; a cloud
  that is fully well-covered before the budget is spent → remaining picks fill to exactly `COUNT`
  deterministically, no error.
- **Chaining.** `--filter-by-reprojection-error V --include-by-distribution N --remove-short-tracks 1
  --bundle-adjust` runs cleanly; the result has exactly `N` images and no length-1 tracks.

## Open Questions

1. **Expose `H`?** e.g. `--include-by-distribution 16,H=...` — or keep it internal until there is a
   concrete need? It is the only knob, and the one most likely to want per-dataset tuning (a tight
   indoor scan and a wide aerial pass want different "different directions" thresholds).
2. **One angle or two?** The same `H` currently gates both "is this point well-covered" and "is this
   new observer different enough to bother keeping." A smaller well-covered threshold (a point
   triangulates acceptably at less parallax than you would demand before adding a redundant view)
   might be the more natural choice. Worth splitting?
3. **Seed cap.** `⌈COUNT/3⌉` is a guess. Too generous and one prominent feature dominates a small
   budget; too tight and the anchor is under-triangulated. Tune, or make it `min(⌈COUNT/3⌉, fixed)`?
4. **Early stop below `COUNT`?** Today `COUNT` is a hard target and a fully-covered cloud just gets
   the budget filled with the most-content remaining units. Should the filter instead stop early and
   report "the cloud was covered with M < COUNT views"? (`--include-range` and friends never undershoot
   their target, so the conservative answer is no.)
5. **A sparse "seeds only" mode?** Stopping after a few farthest-point steps yields a small,
   spatially-spread set without per-region densification — sometimes that is the goal. Worth exposing
   as a `--include-by-distribution N,seeds-only` strategy, or out of scope?
6. **Spatial vs. surface coverage.** "Nearest well-covered point, Euclidean" treats the cloud as a
   point set in `R³`. For a scene that is mostly a thin surface this is fine; for one with large empty
   interior volume (a room scanned from inside) it still behaves, since there are no cloud points in
   the empty volume to be "far" from coverage. No change anticipated, but noted.
