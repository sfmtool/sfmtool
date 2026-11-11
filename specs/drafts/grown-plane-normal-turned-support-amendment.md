# Grown Plane Normal: A Working Grid Turned With the Plane

**Status:** Draft. Nothing here is built. The problem is measured on the
ground truths; the proposal is prototyped in Python, not in the step, and is
not decided. The settings' defaults are first values.

Amends: [core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md)
§ "One plane shared by all views" (the working grid), § "Near to far within a
view", § "Combining the views' planes", § "Wide pairs", § "The selection" (the
views not picked), § "The plane's precision" and § "Parameters".

## Purpose

`grown_plane_normal` lays its working grid, `W × W` samples, once: on the
starting patch square, re-anchored on the reference keypoint. It renders the
template, the reference's samples on that grid, once. Every later plane is
read by following each sample's fixed reference ray to the plane
(`Grid::project` in
[grown_plane/render.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/render.rs)).
That reads each plane exactly, but the part of the plane it reads, its
support, is the starting square sheared onto the plane along the reference
rays, not the square the patch will have once it is turned there. After a
small turn the two are nearly the same. After a large turn they are not: the
evidence for the turn comes from surface the turned patch does not cover, and
a second run of the step, which lays its grid on the turned square, compares
different pixels and lands somewhere else.

This draft proposes laying the working grid again at the start of every fitted
stage of a growth, on the square turned to the plane the growth has reached,
with a cap on how far one stage may turn, and repeating the whole-grid stage as
rounds whose cap halves whenever a round undoes the one before it. Every other
reading of a whole grid at a plane (a growth's information, the held checks,
the pairs and the shift-only fits) reads it on the grid laid at that plane.

## The problem

**The sheared support.** The grid's samples are rays from the reference camera
through the starting square. A plane through the anchor turned by `θ` cuts
those rays over a region whose extent along the turn grows as about
`1/cos θ`, while the region the turned patch covers stays one patch side
across. On `seoul_bull_sculpture_ground_truth_v02` point 255, whose patch side
is 0.87 m, the step turns 69.6°. At that turn the grid's 24 rows still cover
the same 22 px of the reference photograph vertically, as they do at every
plane, but span about 2.4 m along the turned plane, close to the 0.87 m side
over cos 70°. The 0.87 m square turned to that plane has a footprint of only
21 × 8 px in the reference. The whole-grid score that chose the 69.6° plane
was read over nearly three patch sides of surface.

**Repeated passes do not settle.** Run again from the turned normal, the step
lays its grid on the turned square, reads different pixels and lands elsewhere.
Six passes on point 255 turned 69.6°, 15.6°, 3.9°, 4.4°, 3.5° and 8.9°. This is
one cause of the non-idempotence the ground-truth sweep found
([refine-normals.md](../core/bench/refine-normals.md) § "Parallelism and
determinism" says a refinement is not idempotent): points still turning more
than 5° on the fourth pass number 56 on seoul_bull v02, 49 on kerry_park v02
and 10 on dino_dog_toy v01.

**Point 255.** Its three views are within 4.1° of each other at 15.5 m. The
patch straddles a canopy edge in front of tree trunks. The step turns it 69.6°
with σ 36.6°, a turn of 1.9σ; the turn is just under the weak-turn rule's
`weak_turn_deg` of 70°, so the rule does not judge it and the turn is made.

## What is proposed

### Turned support per stage

A growth's fitted stages are its later windows (every window after the first)
and the whole-grid rounds below. At the start of each, the step lays the
working grid on the square turned to the plane the growth has reached:

- **The turned square** is the starting patch frame turned by the least
  rotation that takes the starting normal `n₀` to that plane's normal `n`,
  built as `tilt_patch` builds its rotation (about `n₀ × perp` by the arc's
  angle, `perp` the unit direction the arc leaves `n₀` in, the frame's own `u`
  at a half turn), without `tilt_patch`'s stop at `MAX_TILT_DEG` from a camera,
  since the square is a working grid and not the patch. It is centred on the
  same anchor, with the same side `W` and the same sample spacing in world
  units, `2·half_extent/W`. The turned axes are `u = R·u₀`, `v = R·v₀`.
  Because the rotation is taken from the starting frame each time, the grid
  laid at a plane depends on that plane alone, not on the path the growth took
  to it.
- **The rays** are the reference camera's rays through the turned square's
  samples, at the same grid positions `(col + ½ − W/2, row + ½ − W/2)`. The
  cells are the same index sets.
- **The template** is the reference rendered on those rays at no tilt and no
  shift, from the reference's pyramid level chosen once at the starting plane,
  and the live cells are read again from it.
- **The tilt** is stated in that grid's own frame: the stage fits
  `unit(n + a·u + b·v)` from `(a, b) = (0, 0)`, so the tilt parameters are
  always near zero and a tilt of `(a, b)` is a turn of `atan |(a, b)|` from the
  stage's starting plane.
- **The shift** carries over from the stage before. The shift is an image shift
  added to the view's projected positions; every plane passes through the
  anchor, whose projection does not depend on the grid, so the same value means
  the same thing on every grid. The keypoint-implied shift is unchanged.

Within a stage, the rays and the template are fixed, and a plane other than the
stage's starting plane is read as today, by sliding the samples along the
stage's rays: within a stage the support is the stage's starting square
sheared onto the plane, which a cap of 12° keeps close to the square turned
there (a factor of at most `1/cos 12° ≈ 1.02` along the turn).

**Stage 1**, the unfitted 7×7 read at the keypoint-implied shift, is read on
the grid laid at the plane the growth starts from, at no tilt in that grid's
frame. For a growth from the starting plane that grid is today's grid, so stage
1 is today's read bit for bit. For a growth from the combined plane it is the
grid the first fitted window is laid on, so stage 1 and that window compare the
same template; it costs nothing, since that window needs the grid anyway.

**The chart's edge.** A trial plane at 90° or more from the starting normal
(`n · n₀ ≤ 0`) cannot be stated in the starting frame's tilt (below), and is a
plane edge-on to or facing away from the starting square. `ViewFit` and
`PairFit` score such a trial point `−1`, so no step keeps it.

### A cap on each stage's turn

Within a stage the plane may turn at most a cap from the plane the stage began
at: in the stage's own frame, `|(a, b)| ≤ tan(cap)`. The optimiser keeps every
trial point inside that disc: a trial point whose tilt would leave it has its
tilt moved radially back onto the disc's edge, `(a, b) · tan(cap)/|(a, b)|`,
before it is rendered, and its shift is left as the step put it. The window
stages use `stage_turn_cap_deg`, 12° by default. The optimiser's per-step tilt
cap, 0.1 growing to 0.8 (`MAX_TILT_STEP`, `MAX_GROWN_TILT_STEP`), is unchanged
and applies first; the disc is applied to the point the capped step reaches.
The tilt cap doubles after a kept step it shortened, as today; a step shortened
by the disc alone does not double it.

### Whole-grid rounds with a shrinking cap

The whole-grid stage repeats as rounds. Round `r` lays the grid at the plane
the growth has reached, fits the tilt and the shift on the robust cell score
under the cap `c_r` from `(0, 0)` and the shift reached, and ends at the
plane of unit normal `p_r`. Its turn is the angle from its starting plane to
`p_r`, `atan |(a, b)|`, and its turn vector is the difference of the two
normals, `d_r = p_r − p_{r−1}` in the world frame, `p_0` being the normal of the
plane the rounds start from (the last window stage's end).

- The first round's cap is `stage_turn_cap_deg`.
- After round `r`, from the second round on, when `d_r · d_{r−1} < 0` (the
  round turned back against the one before), the cap halves: `c_{r+1} = c_r / 2`. Otherwise
  `c_{r+1} = c_r`.
- The rounds stop after a round when the next cap would be under
  `min_stage_turn_cap_deg` (0.5°), when the round turned less than half of
  `min_stage_turn_cap_deg` (a round that could not rise turns 0°), or when
  `max_grid_rounds` (16) rounds have run.

The growth's plane, shift and whole-grid stage score are the last round's. A
cap that halves on reversal makes the rounds a trust region: a round may go
12° in one direction while each round keeps finding the score higher that way,
and once the rounds start going back and forth the step shrinks until it is
too small to matter. The reason for it is in § [Measurements](#measurements):
with a fixed cap every start used all its rounds, turning 2° to 12° per round
without net drift, because laying the template again moves the optimum of a
flat score.

The early-stop rule is unchanged: it judges the window stages' scores, each
read on its own stage's grid, from index 2 of `stage_zncc` on.

### Every whole-grid reading on the grid laid at its plane

Each reading of a whole grid at a plane is taken on the grid laid at that
plane, at no tilt in its frame. One template render per plane serves every
view read there.

- **A growth's information and its own score.** After the last round, the step
  lays the grid at the growth's final plane, one more template render, and
  reads there the normal matrix over `[a, b, sx, sy]` and, from the same render,
  the robust cell score. That score is the growth's `grid_zncc`, which
  `min_held_zncc` judges (`UnderFloor`); it differs from the last round's score
  by the last round's turn, read on the turned square rather than sheared. The
  last entry of `stage_zncc` stays the last round's score.
- **The held checks** of the robust combination score each kept view on the
  grid laid at the combined plane, at its shift moved there (below).
- **The pairs** start on the grid laid at the combined plane and run the same
  rounds as a growth's whole-grid stage, the pair's tilt and relative shift
  fitted in each round's frame from `(0, 0)` and the relative shift reached.
  As today, the two views' held shifts are each view's own shift moved to the
  combined plane, for every round. The pair's information and its score
  between the two renders are read on the grid laid at its final plane, whose
  template gives the live cells; that score is the pair's `zncc`, which the
  floor judges.
- **The shift-only fits** run on the grid laid at the final plane, with the tilt
  held at `(0, 0)` in its frame: the stage-1 read, the window fit and the two
  whole-grid iterations, as today. The last held check of the combination read
  the same plane, so its template is reused.

### Combining estimates read in different frames

Each estimate's tilt and information come out in its own grid's frame. The
combination (`combine.rs`) takes `(tilt, information)` pairs, all in one
coordinate system, and computes the weighted mean, the Mahalanobis distances
and σ in it; `Prediction::model` reads each estimate's information along its
offset direction in the starting frame's `(u, v)`; and the selection, the
pairs and `GrownPlaneTry::combined_normal` pass tilts in the starting frame.
The proposal keeps all of that and carries each estimate into the starting
frame's tilt coordinates before it is combined:

```text
c₀(m) = ((m·u₀)/(m·n₀), (m·v₀)/(m·n₀))      the starting chart, m · n₀ > 0
tₑ    = c₀(nₑ)                               the estimate's plane
J     = ∂c₀(nₑ + α·uₑ + β·vₑ)/∂(α, β) at (0, 0)
      = [ (uₑ·u₀)/k − (nₑ·u₀)(uₑ·n₀)/k²    (vₑ·u₀)/k − (nₑ·u₀)(vₑ·n₀)/k² ]
        [ (uₑ·v₀)/k − (nₑ·v₀)(uₑ·n₀)/k²    (vₑ·v₀)/k − (nₑ·v₀)(vₑ·n₀)/k² ]
        with k = nₑ·n₀
Iₑ⁰   = J⁻ᵀ · Iₑ · J⁻¹                       its information about (a, b)
```

`Iₑ⁰` is the information a tilt change in the starting chart carries to first
order: a change `δ` there is `J⁻¹ δ` in the estimate's frame. This is the
rule chosen, rather than combining in a frame at the combined plane, because:

- the combination, the distance test, σ, the selection's model and the
  weak-turn rule then run unchanged, on quantities in the same coordinates
  they are in today, so `max_agreement_distance`, `target_sigma_deg`,
  `low_confidence_sigma_deg` and the weak-turn rule's `min_turn_sigma_ratio`,
  which were all set on σ and distances in this chart, keep their meaning;
- with `turned_support` off every estimate's frame is the starting frame,
  `J = I`, and the carry is skipped, so the step is today's bit for bit;
- an estimate's information about the tilt is a property of the estimate at
  its plane, and the carry is exact to first order at that plane, which is all
  today's quadratic model of each estimate claims.

The starting chart is not uniform in angle: at a turn `θ` from `n₀`, a unit of
`a` along the turn is `cos² θ` radians and across it `cos θ`, so a σ computed
in it as if a tilt unit were a radian is larger than the angular σ there, by
8.5× along the turn at 70°. That is so today, since today's information is
read in this chart at the estimate's plane; carrying the turned frame's
information into it gives σ in the same chart, with the same distortion, not
a better one. Whether σ should be
read in a frame at the combined plane is an open question below, since it moves
the weak-turn rule's calibration.

An estimate whose plane is at 90° or more from `n₀` has no starting-chart tilt.
No fitted trial point reaches such a plane (§ "The chart's edge"), so no
estimate does.

### Moving a shift to another plane

A view's shift is moved to another plane by its own normal matrix,
`s = sₑ − N_ss⁻¹ N_st δ` (`shift_response` in
[grown_plane/solve.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/solve.rs)).
`N` is read in the frame of the grid the estimate was read on, with normal
`fₑ` and axes `uₑ, vₑ`, so `δ` is the target plane's tilt in that frame less
the estimate's own:

```text
cₑ(m) = ((m·uₑ)/(m·fₑ), (m·vₑ)/(m·fₑ))
δ     = cₑ(m_target) − cₑ(nₑ)
```

With turned support that grid is laid at the estimate's plane, so `fₑ = nₑ` and
`cₑ(nₑ) = 0`; with it off the frame is the starting frame, `cₑ` is the
starting chart and `δ = t − tₑ`, today's rule. The exact chart map is used rather than `J⁻¹`
applied to the starting-chart difference, since it costs as little and is
right at any distance. A target at 90° or more from `nₑ` leaves the shift
unmoved. The shift itself needs no conversion between grids (§ "Turned support
per stage").

### What stays the same

The working side `W` and each view's pyramid level, the reference's included,
chosen once at the starting plane; every view's offset onto its slid patch
(`ViewImage::offset`), which no plane changes; the keypoint-implied shifts;
the first window; the window schedule; the early-stop rule; the robust cell
score; the selection's ladder, model and stopping; the pair choice; the robust
combination's algebra; the weak-turn rule; the final turn and readback through
`turn_and_read`; and writing no keypoint.

## The code

In [grown_plane/render.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/render.rs):

- `Grid` keeps the starting frame's `n₀, u₀, v₀` beside its own frame, and
  gains `Grid::laid_at(&self, normal: Vector3<f64>) -> Grid`, the grid on the
  square turned to `normal` with the same origin, anchor, side and spacing
  (`laid_at(n₀)` gives today's grid), and the two chart maps: the tilt of a
  normal in the grid's own frame (`None` at or past 90°) and `J` above.
  `Grid::project`, `pixel_tilt_derivatives` and `plane_normal` read the grid's
  own frame, as they do now.
- A view's render count moves off `ViewRender`, which holds one grid, onto the
  view (`ViewImage` or the candidate), so the notes' counts sum over every grid
  a view is read on. A `ViewRender` is made per view per laid grid; its
  projection cache is per grid, as now.

In [grown_plane/solve.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/solve.rs):

- `Iteration` gains the disc's radius in tilt units, `Option<f64>`, `None`
  for no disc. `maximise` moves each trial point's tilt onto the disc before
  freezing it. A fit with the tilt held has no tilt entries and ignores it.
- `ViewFit` takes the template and the live cells, and `PairFit` the live
  cells, from the laid grid they are given (today `reference` and `live` come
  from the one grid), and both score a trial plane at or past 90° from `n₀` as `−1`.
- A function that returns both the robust score and the normal matrix from one
  render, for a growth's and a pair's final reading.
- The round schedule (caps, reversal, stop) is a small pure function of the
  rounds' turn vectors, so it can be tested without a photograph.

In [grown_plane.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane.rs):

- `Sequence` holds the starting laid grid (grid, template, live cells) and lays
  others on demand through one function, `lay(normal)`, which renders the
  template and counts template renders. It keeps the last grid it laid, keyed
  by the normal's bits, so the final plane's grid is laid once for the last
  held check and the shift-only fits.
- `grow` lays a grid per window stage after the first window and per whole-grid
  round; `Grown` holds the growth's world normal and its starting-chart tilt.
- `estimate_of` lays the grid at the growth's final plane, reads the score and
  the normal matrix there, and carries the information by `J`. `Estimate`
  gains the frame its normal matrix is in (its own normal and axes), keeps
  that normal matrix in its own frame, and holds its tilt and information in
  the starting chart.
- `moved_shift` uses `δ = cₑ(m) − cₑ(nₑ)`; `score_at` scores on the grid laid
  at the plane it is given.

In [grown_plane/selected.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/selected.rs):
`fit_pairs` runs the rounds and reads the pair's information on the grid laid
at its final plane, carried by `J`; `shift_only` and `fit_shift` run on the
grid laid at the final plane. In
[grown_plane/combine.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane/combine.rs)
nothing changes but the held check's render, through `score_at`.

With `turned_support` off, none of the above lays a grid other than the
starting one, sets a disc, or runs more than one whole-grid round, and no
carry is applied: the step is today's.

## Options

`GrownPlaneNormalOptions` gains four settings; `check` refuses a value outside
the range given, naming the setting as it does the others.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `turned_support` | `true` | Lay the working grid again on the square turned to the plane at the start of every fitted stage and every whole-grid round, and read every whole grid at a plane on the grid laid there. `false` is today's step: one grid, no cap, one whole-grid stage, and the other three settings unread |
| `stage_turn_cap_deg` | `12.0` | The most a window stage, and the first whole-grid round, may turn the plane from where it began, degrees; above 0 and under 90, or infinite for no cap |
| `min_stage_turn_cap_deg` | `0.5` | The whole-grid rounds stop once their cap would fall under this, and after a round that turned less than half of it, degrees; finite, above 0, at most `stage_turn_cap_deg` |
| `max_grid_rounds` | `16` | The most whole-grid rounds a growth or a pair runs; at least 1 |

**Why a switch, and why only one.** The old step stays reachable so the
measurements can compare the two on the same build, and so a caller that
depends on today's results can keep them while the new defaults are judged.
It is a switch rather than a combination of the other three (`max_grid_rounds`
1 and an infinite cap) because the change is the laying of the grid, which no
value of the other three turns off. The cap and the rounds are not offered
without the turned support: the cap is stated as a disc in a grid's own frame
starting from `(0, 0)`, which a grid kept on the starting square does not
have, and the prototype's capped variant on the kept grid was measured and
spread 46.3° on point 255 against 13.8° with the grid re-laid.

## The report

Backward compatible: every existing field keeps its name and meaning, with the
two exceptions said below, and new fields are added.

```rust
pub struct GrownPlaneAttempt {
    // ... the fields it has
    /// The angle each fitted window stage turned the plane, degrees: one per
    /// window stage after the first, in order, as far as the growth ran.
    pub stage_turn_deg: Vec<f64>,
    /// Each whole-grid round, in order; empty for a growth that stopped
    /// early, one round with turned_support off.
    pub grid_rounds: Vec<GrownPlaneGridRound>,
}

pub struct GrownPlanePair {
    // ... the fields it has
    pub grid_rounds: Vec<GrownPlaneGridRound>,
}

/// One whole-grid round of a growth or a pair.
pub struct GrownPlaneGridRound {
    pub cap_deg: f64,   // the cap it was fitted under; infinite with no cap
    pub turn_deg: f64,  // from its starting plane to the plane it ended at
    pub zncc: f64,      // its robust score there, on its own grid
}
```

- `stage_zncc` keeps its shape: stage 1, each later window, then the whole-grid
  stage's score, which is now the last round's.
- `GrownPlaneAttempt::grid_zncc` is, with turned support, the score read on
  the grid laid at the growth's final plane, and so can differ slightly from
  the last entry of `stage_zncc`; with it off the two are equal, as today.
  `GrownPlanePair::zncc` is likewise the score on the grid laid at the pair's
  final plane.
- The `Display` sentence is unchanged.

**Python.** `sfmtool.bench.grown_plane_normal` and `sfmtool.bench.refine_normals`
take `turned_support`, `stage_turn_cap_deg`, `min_stage_turn_cap_deg` and
`max_grid_rounds` as keywords of the same names, through the shared
`GrownPlaneKeywords` in
[sfmtool-py/src/bench/normal.rs](../../crates/sfmtool-py/src/bench/normal.rs),
each `None` for the Rust default. Each attempt dict gains `stage_turn_deg` (a
float64 array) and `grid_rounds` (a list of dicts with `cap_deg`, `turn_deg`
and `zncc`); each pair dict gains `grid_rounds`. The docstrings and
[editable-track.md](../core/bench/editable-track.md) § "Estimating the normal",
which list the report's fields, gain them too.

**Phases.** The `whole-grid stage` detail phase is opened once per round,
noted with the round, its cap, its turn, its ZNCC, its iterations and renders;
each `window stage` note adds its turn. The `growth` note adds the template
renders it made. A pair fit's note adds its rounds. Nothing a phase does reaches
the result, as today.

## Cost

Per fitted stage or round, the step makes one template render: a render of the
whole grid from the reference, against the 6 to 7 view renders a stage makes
today, and one more for each growth's and pair's final reading and each held
check's plane. Measured today, the stages are 35% to 59% of the step's time
(16.9 ms a point on seoul_bull v02, 27.9 ms on kerry_park v02 and 54.9 ms on
dino_dog_toy v01, 60 random points each), so laying the grid per stage is
estimated at 6% to 9% of the step, plus the extra whole-grid rounds: 1 to 3 on
the stable points the prototype ran, and 12 or more on point 255. The
implementation's measured cost, on the same three samples of 60 points, will
replace this estimate.

## Measurements

These are the prototype's, written in numpy on the bindings. It reproduces the
step's projections to 10⁻⁶ px and its ZNCC to 4·10⁻⁷, but it is not the step:
it fits with Nelder–Mead rather than Levenberg–Marquardt, scores plain ZNCC
over the views' mean rather than the robust cell score per view, and fits the
views jointly rather than growing them one at a time and combining their
planes.

**Variants on point 255.** 17 starts: the stored normal, and 20° and 40° off it
at 8 azimuths each; then 5 repeated passes from the stored normal. The spread is
the largest angle between any two of the 17 answers; the last column is the
turn each of passes 2 to 5 made.

| Variant | Spread, all 17 starts | Spread, stored and 20° off (9) | Turns, passes 2–5 |
|---------|----------------------:|-------------------------------:|-------------------|
| Staged, grid kept on the starting square, tilt free | 117.5° | 106.6° | 52.2°, 90.0°, 90.0°, 66.2° |
| Staged, re-laid per stage, tilt free | 134.8° | 124.8° | 9.9°, 50.1°, 53.3°, 110.3° |
| Staged, kept, small windows' tilt held | 49.6° | 34.9° | 4.4°, 3.2°, 4.3°, 7.6° |
| Staged, re-laid per stage, small windows' tilt held | 83.3° | 81.7° | 1.9°, 6.3°, 3.8°, 1.3° |
| Staged, kept, 12° cap | 46.3° | 33.5° | 4.4°, 3.2°, 4.2°, 7.5° |
| Staged, re-laid per stage, 12° cap (fixed, at most 12 rounds) | 13.8° | 12.5° | 1.1°, 3.6°, 2.9°, 3.9° |
| A cost on the turn instead of a cap (12° costs 0.01 / 0.04 of score) | 66.5° / 77.2° | 41.4° / 41.6° | 36.3°, 2.6°, 0.1°, 0.1° / 10.2°, 9.5°, 8.1°, 9.5° |

With the fixed 12° cap every start used all 12 whole-grid rounds: 12° a round to
arrive, then 2° to 12° a round with no net drift, because the template laid
again moves the optimum of a flat score. That is why this proposal halves the
cap on a reversal. The cost on the turn walks about 3° (0.01) or 0.9° (0.04) a
round in one direction and stops wherever the round limit falls.

**The shrinking cap on further points**, against the step as it is:

- kerry_park v02 point 228, stable under the step today: every one of the 15
  starts the reference can see ends within 0.1° of one plane, 4.0° from the
  stored normal, and repeated passes turn 0.0°. The step's own spread over the
  same starts is 1.5°.
- seoul_bull v02 point 205, which has two basins: the prototype's spread over
  all 17 starts is 2.2°, against 53.9° for the step.
- Further points were still running when this draft was written. The
  implementation's evaluation adds them, and the measurements below.

**What was learned.**

- The small windows carry almost no information about the tilt. On point 255
  the 11×11 window's ZNCC varies only from 0.719 to 0.735 over every tilt up to
  36°, so a small window whose tilt is free wanders: to 71° to 84° in the
  prototype, and 35° in the step itself on point 255. The cap bounds that
  wandering without holding the tilt, which the "tilt held" rows show is worse
  once the grid is re-laid.
- Plain ZNCC on a support that shrinks with the tilt favours grazing planes.
  Clipping the support to the turned square's own pixels in the reference,
  scored by plain ZNCC, peaked at the 80° edge of the search, in directions
  that depended on the view. This proposal therefore keeps `W` samples on the
  turned square, interpolated where the square's footprint is under `W` px,
  rather than clipping to its pixels, and leaves a score that does not reward a
  smaller support as an open question.

## Testing

What the implementation's tests should show:

- **The laid grid.** `laid_at(n₀)` gives today's grid's rays to `10⁻¹²`. On a
  grid laid at a normal 40° from `n₀`, every sample lies on the turned plane,
  neighbouring samples are one spacing apart, the centre sample's ray passes
  within half a sample of the anchor, and the anchor projects into a view to
  the same pixel as on the starting grid.
- **The charts.** `J` against a central difference of `c₀(nₑ + α·uₑ + β·vₑ)`
  to `10⁻⁶`; `J = I` on the starting grid; an information carried by `J` and
  back gives itself; a shift moved with `δ = cₑ(m) − cₑ(nₑ)` on the starting
  grid equals today's moved shift.
- **The disc.** On a quadratic whose optimum lies outside the disc, `maximise`
  ends on the disc's edge and no trial point it renders is outside it; inside,
  it reaches the optimum as today.
- **The rounds.** The schedule on given turn vectors: turns that alternate in
  sign halve the cap each round and stop once it would fall under 0.5°; turns
  that keep one direction keep the cap and stop at `max_grid_rounds`; a round
  that turns under 0.25° stops the rounds.
- **The old step.** With `turned_support` off, every attempt that reached the
  whole-grid stage has one round with an infinite cap, `grid_zncc` equals the last entry of `stage_zncc`, and
  every test of the standing spec passes. Once, on the ground truths' sweep,
  the implementation checks that the step with `turned_support` off gives the
  same tracks and the same reports, the new fields aside, as the step before
  the change.
- **The synthetic scenes.** Every test of the standing spec passes at the
  defaults. A textured plane started 60° off its normal, seen from views out to
  40°, ends within a degree of it. On the same scene a second run from the
  turned normal turns under half a degree.
- **Idempotence on the ground truth.** On seoul_bull v02 point 255, three runs
  from the stored normal: the second and third turn under 5°. The bar is first
  set from the prototype (passes 2 to 5 turned 1.1° to 3.9°) and is moved to
  what the step measures if the step differs.
- **Options.** `check` refuses a `stage_turn_cap_deg` that is NaN, 0 or less, or
  90 or more and finite, and takes an infinite one; refuses a
  `min_stage_turn_cap_deg` that is not finite, is 0 or less, or exceeds
  `stage_turn_cap_deg`; and refuses a `max_grid_rounds` of 0.
- **Detail phases change nothing**, with the new phases at their depths.

**The evaluation** reruns, at the defaults and with `turned_support` off: the
three 60-point timing samples; the fourth-pass count of points still turning
more than 5° on seoul_bull v02, kerry_park v02 and dino_dog_toy v01 (56, 49 and
10 today); the 17-start spread on points 255, 205 and 228; and the points judged
by eye behind the weak-turn rule, since the turns the rule judges will change.

## Open questions

- **A score that does not reward a smaller support.** At a grazing plane the
  turned square's `W` rows are interpolated from fewer reference pixels: at
  70°, its 24 rows cover about 8 px of the reference. Interpolated rows are
  correlated, so the ZNCC over them overstates the evidence, and a score that
  counts pixels rather than samples might be the right one; the clipped
  variant above shows plain ZNCC is not.
- **Whether `W` should follow the turned footprint**, rather than staying the
  starting square's.
- **Whether each view's pyramid level should be chosen again** on the grid laid
  at each stage, since a view's map from the grid changes as the square turns.
- **σ in the starting chart.** σ is read as if a tilt unit of the starting
  chart were a radian, which overstates the angular σ at a large turn by up to
  `1/cos² θ`. Reading σ in a frame at the combined plane would correct it, but
  the weak-turn rule and the σ target were set on the chart's σ, so it would
  need them set again.
- **The defaults.** The cap of 12°, the floor of 0.5° and the 16 rounds are
  first values.
- **The pairs' held shifts.** Each pair holds its two views' shifts at the
  combined plane for every round, as today; moving them to each round's plane
  by their normal matrices is an alternative not measured.
