# Select by Distribution

This document specifies the `--include-by-distribution` option for `sfm xform`, which keeps a
small, strategically-chosen subset of a reconstruction's cameras instead of decimating blindly.

## Motivation

Carving a small, committable sample dataset (for example, a fisheye example dataset for the repo)
out of a large capture. Blind decimation — every Nth frame, a random N — discards the structure we
already have. With the full 3D reconstruction in hand we can do better: scatter picks across the
distinct parts of the capture (so the subset isn't a zoomed-in corner of it), and within each part
keep enough views from genuinely different directions that the 3D tracks stay well-triangulated and
the subset solves cleanly on its own.

This is a filter in the `sfm xform` pipeline, so it composes with the existing operations: its
position on the command line determines when it runs relative to geometric transforms, point
filters, bundle adjustment, etc.

## Command Syntax

```bash
sfm xform <input.sfmr> [<output.sfmr>] --include-by-distribution <COUNT>
```

- `COUNT` — target number of views to keep. Must be `>= 2`. If `COUNT` is at least the number of
  selectable units present, the filter is a no-op and prints a notice.

There is no strategy argument today. If a variant is ever wanted — e.g. a sparse "seeds only" mode,
see [Open questions](#open-questions) — it would arrive as `--include-by-distribution <COUNT>[,<STRATEGY>]`.

### Selection unit: images vs. rig frames

If the reconstruction carries rig frame data, the unit of selection is the rig **frame** — both
sensors of a 360° fisheye pair are kept or dropped together, never split. Otherwise the unit is the
individual image. In both cases `COUNT` counts *units*. For a rig-frame unit:

- the points it *observes* are the union of the points observed by its member images;
- its *viewing ray toward a point `p`* is the ray of whichever member image observes `p` — the
  opposite-facing sensors of a 360° rig almost never both see the same world point; if they somehow
  do, use the reference sensor's ray;
- its *center* (used only for reporting; the algorithm itself uses per-image rays, not centers) is
  the mean of its member-image centers.

### Stranded points

The objective gives a 3D point zero value until it has two selected observers, so the selection never
*wants* a single-observation track — but phase 1 (below) picks units regardless of coverage value, so
a point can still end up with one surviving observation. This filter does not remove them; chain
`--remove-short-tracks 1` afterward (consistent with `--include-range` and friends).

### Example

```bash
# Keep 16 well-distributed rig frames, then drop single-observation points.
sfm xform sfmr/recon.sfmr sample.sfmr --include-by-distribution 16 --remove-short-tracks 1
```

## Algorithm

Two phases. **Phase 1** scatters a handful of seed units across the distinct parts of the capture,
using *shared observations* as the notion of "distinct". **Phase 2** spends the rest of the budget
densifying around those seeds so every covered point ends up seen from genuinely different
directions. Both phases are deterministic greedy loops whose marginal scores are maintained
incrementally and kept exact, never lazily cached — see [Incremental, not cached](#incremental-not-cached).

### Phase 1 — regional seeding

Add units one at a time:

- **Pick** the candidate that intersects the *fewest already-selected units* — i.e. shares at least
  one observed point with the fewest of them — breaking ties toward the candidate with the *most
  observations* (so within a tier of equally-fresh candidates we take the content-rich one, not one
  staring at a blank wall), and remaining ties by lowest unit index. The first pick (nothing selected
  yet) is therefore just the most-observed unit.
- **Stop** when every remaining candidate already shares at least `T` observations with the union of
  the selected units — meaning there's no "fresh" region left to seed. (`T` is a shared-observation
  threshold; default and discussion under [Parameters](#parameters).) Also stop if the selected set
  reaches `COUNT` — the capture has more distinct regions than the budget. This is **not an error**,
  but the filter emits a prominent warning: phase 2 will not run, the `COUNT` seeds are deliberately
  non-covisible, so the subset has few or no multi-view tracks (and a following `--remove-short-tracks
  1` would gut it) — the user almost certainly wants a larger `COUNT`. The result is still those
  `COUNT` units, in case that is genuinely what was wanted.

Phase 1 typically picks a small number of units — roughly one per genuinely separate stretch of the
capture. Using *shared observations* rather than Euclidean distance as the spread metric is
deliberate: two cameras can sit centimetres apart looking at different walls, or metres apart looking
at the same one; what tells you "this is a new part of the scene" is whether they see the same 3D
points, not where they are.

### Phase 2 — densify for parallax

With the phase-1 seeds already selected, repeatedly add the unit with the largest `coverage_gain`
against the current selected set, until `COUNT` units are selected.

#### Coverage gain — reward parallax, not view count

What makes a 3D point useful in a reconstruction is being seen from genuinely different directions
(parallax), not just being seen by many cameras — five frames shot 10 cm apart triangulate a point no
better than two of them. So coverage is scored by **angular baseline**, not observer count.

For point `p` and selected set `S`, let `obs_S(p)` be the selected units observing `p`, and
`parallax_S(p)` the largest angle between any two of their viewing rays toward `p` (`0` if
`|obs_S(p)| < 2`). The point's value is

```
value_S(p) = min(parallax_S(p), A_sat) / A_sat        # in [0, 1]; 1 == "triangulated well enough"
```

where `A_sat` is a saturation angle (default under [Parameters](#parameters)): a point contributes
`0` until two selected views give it some parallax, ramps up linearly with the baseline, and stops
gaining once the baseline reaches `A_sat` — so the objective does not keep paying to pile views onto
an already-well-conditioned point. The coverage objective is

```
Coverage(S) = sum_p value_S(p)        # roughly "effective number of well-triangulated points"
```

and `coverage_gain(u | S) = Coverage(S ∪ {u}) − Coverage(S)`, which — since adding `u` only changes
`value` for the points `u` observes — is

```
coverage_gain(u | S) = sum_{p observed by u} [ value_{S ∪ {u}}(p) − value_S(p) ]
```

A near-duplicate of an already-selected view contributes ≈ 0: its rays toward shared points are
nearly parallel to ones already counted, so it barely moves any `parallax_S(p)`. A view from a fresh
direction, or the *first* second view of a point, contributes real value. This is why phase 2, even
on a video, doesn't just march along a run of adjacent frames — each consecutive frame is almost all
near-duplicate rays, so its `coverage_gain` collapses and a frame that opens new baselines wins.
(The codebase already reasons about per-track viewing-angle spans — see `RemoveNarrowTracksFilter` —
so the geometry here is familiar machinery.)

Phase 2 self-balances across the phase-1 regions: as a region's points reach `A_sat` parallax they
contribute ≈ 0, so the greedy drifts to whichever seeded region still has under-triangulated points.
The exception is a tight budget — if `COUNT` runs out, the regions phase 1 seeded last (its
lowest-content picks) get the least densification; see [Open questions](#open-questions).

### Incremental, not cached

After a unit `u` is selected, the scores that change are updated directly rather than recomputed from
scratch:

- *Phase 2 — coverage.* Keep `parallax_S(p)` and the list of selected observers' rays for every
  point, plus a running `coverage_gain[v]` for every unit. On each pick, for each point `p` that `u`
  observes: for every still-unselected co-observer `v` of `p`, subtract `v`'s current contribution
  from `p` out of `coverage_gain[v]`; fold `u`'s ray into `parallax_S(p)` and the ray list; add `v`'s
  recomputed contribution back in. (`v`'s contribution from `p` is
  `[ min(max(parallax_S(p), maxangle(ray_v, sel_rays(p))), A_sat) − min(parallax_S(p), A_sat) ] / A_sat`,
  where `maxangle(ray_v, sel_rays(p))` is the largest angle between `ray_v` and any ray in the list,
  or `0` if empty.)
- *Phase 1 — intersection / overlap counts.* Keep, per unselected unit `v`, the number of
  already-selected units it shares an observed point with, and the number of observed points of `v`
  that have at least one selected observer (its overlap with the selected union, for the stop test).
  On each pick of `u`: every unselected `v` that observes any point `u` observes gains one toward its
  intersected-units count; and for each point `p ∈ pts(u)` that had no selected observer before, every
  unselected `v` observing `p` gains one toward its overlap count.

This is deliberately *not* lazy gain-caching (CELF and its relatives), which keeps stale cached gains
and only refreshes the heap's top entry, trusting that a stale value is still an upper bound on the
true one. That trust is invalid here: a point's value is `0` until its *second* selected observer
arrives, so a candidate sharing such a point sees its `coverage_gain` *rise* the instant another pick
supplies that point's first observer — a stale gain can be an *under*-estimate, so the heap-top
candidate need not be the true maximum. The scheme above never relies on that ordering; it updates
every score that moved and leaves untouched only the scores that provably did not.

Cost is bounded by observation-touching work: each pick is
`O( Σ_{p ∈ pts(u)} |obs(p)| · (1 + |sel_obs(p)|) )` across both phases' updates — never `n_units²`.
Over a whole run that is a few million float operations for a hundred-thousand-image capture with
`COUNT` in the tens to low hundreds: seconds, not minutes.

### Approximation

No approximation guarantee is claimed for either phase. `Coverage(S)` (phase 2's objective) is
monotone — adding a view can only increase any point's max parallax — but **not** submodular: the
max-parallax of a set of viewing rays is not a submodular set function, and `value_S` is non-concave
in the observer set by design (zero for the first observer, positive once there is a second), which is
the whole point. Phase 1's min-intersection rule is a greedy dispersion heuristic, not the optimizer
of any stated objective. Both phases are sound, well-behaved heuristics; neither carries a proven
ratio.

### Parameters

Not exposed on the CLI initially; hard-coded:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `A_sat` | `15°` | Saturation parallax — the baseline angle at which a point counts as triangulated well enough; further views on it then add no value. |
| `T` | `0.15 × median per-unit observation count` | Phase-1 stop — once every remaining candidate shares at least `T` observations with the selected union, there is no fresh region left to seed. |

See [Open questions](#open-questions) on whether to expose these; `T` is the one most likely to want
per-dataset tuning.

### Determinism

For a given input `.sfmr` and build, selection is fully deterministic:

- Phase 1: each pick is `argmin` over candidates of intersected-unit count, ties broken by *most*
  observations, further ties by lowest unit index. The first pick is the most-observed unit (ties →
  lowest index). Phase 1 is integer-valued throughout.
- Phase 2: each pick is `argmax` over candidates of `coverage_gain`, ties broken by lowest unit index.
- All arithmetic is on the data in the file (no RNG, no wall-clock, no order-dependent iteration over
  unordered containers).

(Floating-point `coverage_gain` values are reproducible for a given build/platform; the spec does not
promise bit-identical phase-2 selection *across* differing BLAS/CPU configurations, only that there is
no intentional nondeterminism.)

## Implementation Notes

- New `SelectByDistributionFilter` in `src/sfmtool/xform/_select_by_distribution.py`, exported from
  `src/sfmtool/xform/__init__.py`, and wired into `src/sfmtool/_commands/xform.py` (a `@click.option`,
  a branch in `parse_transform_args`, the help text, and the "at least one transform" error list). Its
  `description()` should follow precedent, e.g. `"Select N cameras by distribution"`.
- The module only computes *which images to keep*; it then delegates to `_filter_images()` from
  `_filter_by_image_range.py` for the track/point/rig-frame bookkeeping and index remapping.
- Camera centers from `quaternions_wxyz` / `translations` as `C = -R(q)^T t` (COLMAP world-to-camera).
  A unit's viewing ray toward point `p` is `(positions[p] − center)` normalized; parallax of two
  observers is the angle between their rays (`arccos` of the clamped dot product of the unit vectors).
- The one large structure is the static observation adjacency, built once: `obs(p)` (units observing
  each point) and `pts(u)` (points each unit observes), `O(#observations)`. The incremental state on
  top of it is small: per point, its selected-observer rays and current `parallax_S(p)`; per unit, a
  running `coverage_gain` plus the phase-1 intersected-units and overlap counts. Comfortable at
  hundreds of thousands of images.

## Scope

This filter changes which images the `.sfmr` references. It does **not** modify image files or
`.sift` feature files on disk. Producing the actual small dataset on disk — downscaling images,
rescaling intrinsics and 2D observations (or re-extracting features at the new resolution), and
copying only the surviving images and features — is out of scope for this option and would be a
separate command or workflow step.

## Testing

- Clustered fixture: a reconstruction made of `K` clearly separate covisibility clusters. With
  `COUNT ≥ K` and a sane `T`, phase 1 places at least one seed in every cluster and phase 2 then
  densifies. With `COUNT < K`, phase 1 caps at `COUNT` (one seed each in the `COUNT` most-content
  clusters), phase 2 does not run, and the result is exactly those `COUNT` units.
- Video-like fixture: cameras densely sampled along a 1-D path, all observing a shared scene. Phase 1
  drops seeds spread along the path (no two seeds covisible, or sharing fewer than `T` observations);
  the final selection covers the whole path, not a localized blob, and within each seeded stretch the
  picks see shared points across real baselines. This is the regression test for the "localized
  zoomed-in subset" and "prefers near-duplicates" failure modes together.
- Coverage-quality check on a volumetric fixture: the selection yields a higher mean per-point
  parallax (and more points above any fixed parallax threshold) than naive every-Nth selection at the
  same `COUNT`. Comparative property, not a universal guarantee — the greedy provides none.
- Determinism: running the same selection twice on the same input gives identical kept-image lists; a
  hand-checked tiny fixture gives the expected pick order for both phases (asserted explicitly).
- Rig reconstruction: selection keeps both sensors of every chosen frame and `rig_frame_data` is
  re-indexed correctly (covered transitively by the existing `_filter_images` rig tests); points
  observed by a frame and a frame's ray toward a point follow the union / observing-member rules from
  [Selection unit](#selection-unit-images-vs-rig-frames).
- Edge cases: `COUNT` at least the number of units → no-op with notice; `COUNT < 2` → `UsageError`;
  `COUNT` smaller than the number of distinct regions → phase 1 fills the budget, phase 2 is skipped,
  a warning is emitted, and exactly `COUNT` units are returned (not an error).
- Chaining: `--include-by-distribution N` followed by `--remove-short-tracks 1` and `--bundle-adjust`
  runs cleanly.

## Open Questions

1. **Expose the parameters?** e.g. `--include-by-distribution 16,T=...` — or keep `A_sat` and `T`
   internal until there's a concrete need? `T` (phase-1 stop) is the one most likely to want
   per-dataset tuning.
2. **Hard floor on triangulation quality?** Should the filter ever stop *below* `COUNT` to avoid a
   subset where many points sit below some minimum parallax, or is `COUNT` always the hard target with
   the coverage term doing the work?
3. **Rig frame position.** "Mean of member-image centers" is currently only used for reporting; is
   that fine, or should anything user-visible use the reference sensor's center instead?
4. **Phase-2 fairness under a tight budget.** When `COUNT` barely exceeds the phase-1 seed count,
   phase 2's coverage greedy may pour everything into the first seeded region(s) and leave later seeds
   bare. Should phase 2 round-robin across seeds, or is the self-balancing (saturated regions stop
   paying) good enough? (The extreme — `COUNT` below the region count, so phase 2 doesn't run at all
   — is handled by warning rather than erroring; see [Phase 1](#phase-1--regional-seeding). Is the
   warning the right call, or should there be a softer note as the budget gets merely *thin*?)
5. **A sparse "seeds only" mode?** Stopping after phase 1 yields a small, covisibility-spread set that
   isn't densified — sometimes that's the goal (a quick representative scatter). Worth exposing as a
   `--include-by-distribution N,seeds-only` strategy, or out of scope?
