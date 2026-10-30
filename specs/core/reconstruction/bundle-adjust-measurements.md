# Reconstruction-Level Bundle Adjustment Measurements

This file records measurements behind the reconstruction-level bundle
adjustment of [bundle-adjust.md](bundle-adjust.md). Its one section measures a
rule proposed for the scanlines race, which runs one solve with rows and one
with columns for a camera whose rolling-shutter scanlines are `UNKNOWN`
([bundle-adjust.md § The rolling-shutter
scanlines](bundle-adjust.md#the-rolling-shutter-scanlines)). The race runs both
solves to the end. The rule, proposed in
[per-image-lens-corrections.md § Choosing rows or
columns](../../drafts/per-image-lens-corrections.md#choosing-rows-or-columns),
would stop the trailing solve once it cannot plausibly catch up. The rule is
not built; these measurements are for deciding whether to build it.

## Stopping the trailing solve of a scanlines race

### The rule and how it is replayed

The two solves run in lockstep, one step of each at a time. After each pair of
steps the gap is `D = (C_trailing − C_leading)/σ²`. The trailing solve is
stopped when, for `K` consecutive pairs with the same solve trailing, both
`D > T` and `D` exceeds what the trailing solve would remove in the rest of its
budget at its recent rate: the mean of its last three decreases times the steps
the schedule has left (the **budget term**). The leading solve then runs to the
end alone. The proposal is `T = 25` and `K = 3`.

The measurement runs both solves to the end and replays the rule offline on
their cost trajectories, so it can say what the rule would have done without
the rule changing anything. Two lockstep units are replayed:

- **Per LM iteration.** One iteration of each solve per pair. The two are held
  at each schedule round's end until both have finished the round, because two
  costs compare only within one round, at one loss scale and over the kept
  observations of one trim. A solve that has finished the round keeps its last
  cost. The budget is the iterations left in the round plus `max_iters` (60) for
  each round after it. The pair at a round's start, before either solve's first
  iteration in it, is skipped: it neither counts toward `K` nor breaks a run,
  since a rate cannot be read there yet.
- **Per schedule round.** One round of each solve per pair, compared at the
  round's end. The rate is the mean of the trailing solve's decrease over its
  last three rounds, each from its start to its end, and the budget is the
  rounds left.

The cost of a step is the kernel's `COST_VALUE`: the cost the damping ladder
compares, reported through `Progress` once before each round's first iteration
and once after each ([../geometry/bundle-adjustment.md](../geometry/bundle-adjustment.md)).
`σ²` is the mean of the two solves' noise levels squared in the pair's round,
each the round's `NOISE_VALUE`, measured at the round's start. In the first
round both solves start from one state, so their first costs and noise levels
are equal. The race's own comparison, the final costs over the observations
both final rounds kept at one pooled `σ_n`, decides the **winner**; a stop is
correct when it stops the other solve. The gap the replay reads at the end
agrees with the race's to within 7% on the synthetic scenes and the dino
candidate; it is 716 against 870 on the raw solve, whose two final rounds kept
7 different observations, and 4.5 against 7.4 on the seoul bull.

The tools are the `scanlines_race` example of `sfmtool-core`
([scanlines_race.rs](../../../crates/sfmtool-core/examples/scanlines_race.rs)),
which runs the race with every camera `UNKNOWN` and writes both solves'
trajectories to JSON, and
[`scripts/scanlines_race_early_stop.py`](../../../scripts/scanlines_race_early_stop.py),
which replays the rule on them. The synthetic scenes are written by the
ignored test `write_scanlines_race_scenes` in
[tests/corrections.rs](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust/tests/corrections.rs),
from the scene the race's own tests use.

### The inputs

All on one `SFMTOOL_PINHOLE` camera with its scanlines not stored, at the
default `max_iters` of 60, with the crossing on unless stated.

- **Dino candidate**: the maintainer's local
  `dino_dog_toy_ground_truth_candidate_v012.sfmr` (85 images, 24918
  observations), releasing the focal, the spline, each image's focal and
  principal point and both corrections. Once over one `(inf, 1000)` round with
  the crossing off, which a derived prior runs twice, and once over the
  default schedule.
- **Dino raw solve**: `20261003-00-solve-dino_dog_toy_1-85.sfmr` from the same
  workspace, switched to `SFMTOOL_PINHOLE` by the core's switch, the same
  releases, the default schedule (85872 observations).
- **Seoul bull**: `seoul_bull_sculpture_ground_truth_v01.sfmr` switched to
  `SFMTOOL_PINHOLE`, the same releases, the default schedule (17 images, 1277
  observations).
- **Synthetic**: the race tests' scene of 30 images and about 2400
  observations with 0.1 px of noise and perturbed poses, each image's radial
  correction drawn with spread `3·10⁻³` and its rates with spread `τ_w`, planted
  on rows or on columns, releasing both corrections, the default schedule.

| Input | Rows cost | Columns cost | Winner | Gap | Iterations, rows + columns | Seconds, rows + columns |
|---|---|---|---|---|---|---|
| Dino candidate, `(inf, 1000)` | 49326.4 | 52327.3 | rows | 3001 | 27 + 27 | 12.9 + 13.1 |
| Dino candidate, default | 47166.3 | 50184.1 | rows | 3018 | 39 + 40 | 19.8 + 19.8 |
| Dino raw solve, default | 129373.7 | 130243.8 | rows | 870 | 104 + 123 | 54.9 + 66.3 |
| Seoul bull, default | 1906.6 | 1914.0 | rows | 7.4 | 102 + 82 | 0.6 + 0.5 |
| Synthetic rows, `τ_w = 4·10⁻³` | 2685.2 | 7187.1 | rows | 4502 | 14 + 14 | 0.3 + 0.3 |
| Synthetic rows, `τ_w = 10⁻³` | 4659.7 | 5281.5 | rows | 622 | 13 + 14 | 0.3 + 0.3 |
| Synthetic rows, `τ_w = 3·10⁻⁴` | 4879.8 | 4928.1 | rows | 48.3 | 12 + 14 | 0.3 + 0.4 |
| Synthetic rows, `τ_w = 10⁻⁴` | 4864.6 | 4885.0 | rows | 20.4 | 13 + 13 | 0.3 + 0.3 |
| Synthetic columns, `τ_w = 4·10⁻³` | 7354.0 | 2335.9 | columns | 5018 | 15 + 15 | 0.4 + 0.4 |
| Synthetic columns, `τ_w = 10⁻³` | 5298.6 | 4629.2 | columns | 669 | 14 + 13 | 0.4 + 0.3 |
| Synthetic columns, `τ_w = 3·10⁻⁴` | 4985.0 | 4904.2 | columns | 80.7 | 13 + 12 | 0.3 + 0.3 |
| Synthetic columns, `τ_w = 10⁻⁴` | 4924.9 | 4903.1 | columns | 21.7 | 12 + 12 | 0.3 + 0.3 |

The costs are the race's, in units of its pooled `σ_n²`. Every synthetic race
chose the planted scanlines. On the seoul bull this release chose rows by 7.4 units; an
earlier run in SfM Explorer with other settings gave 1865.4 with rows and 1842.5
with columns, so the choice there depends on what is released.

### The trajectories

The signed gap `(C_rows − C_columns)/σ²` after a few iterations of each round
and at its end (negative while rows leads):

| Input | Round | i = 1 | i = 3 | i = 5 | i = 10 | i = 20 | End |
|---|---|---|---|---|---|---|---|
| Dino candidate, `(inf, 1000)` | 1 | −719 | −885 | −965 | −957 | | −957 |
| | 2 | −1154 | −1105 | −1064 | −1049 | | −1049 |
| Dino candidate, default | 1 | −728 | −884 | −962 | −953 | | −953 |
| | 2 | −3288 | −3159 | −3049 | −3008 | | −3008 |
| | 3 | −3140 | −3117 | −3105 | −3101 | | −3101 |
| Dino raw solve, default | 1 | −100 | −270 | −340 | −312 | −310 | −308 |
| | 2 | −1188 | −1127 | −1095 | −1071 | −1047 | −1040 |
| | 3 | −766 | −769 | −787 | −780 | −731 | −716 |
| Seoul bull, default | 1 | +18.0 | +24.1 | +22.7 | +24.3 | +21.4 | +21.4 |
| | 2 | −889 | +305 | +159 | +33.0 | −2.8 | −2.4 |
| | 3 | +4.3 | −0.4 | −3.9 | −4.5 | | −4.5 |
| Synthetic rows, `τ_w = 4·10⁻³` | 1 | −73.0 | −82.1 | −82.1 | | | −82.1 |
| | 2 | −4758 | −4758 | −4758 | | | −4758 |
| Synthetic rows, `τ_w = 10⁻³` | 1 | −9.8 | −10.2 | −10.2 | | | −10.2 |
| | 2 | −618 | −618 | −618 | | | −618 |
| Synthetic rows, `τ_w = 3·10⁻⁴` | 1 | −1.1 | −1.2 | −1.2 | | | −1.2 |
| | 2 | −37.2 | −37.2 | | | | −37.2 |
| | 3 | −49.7 | −49.7 | | | | −49.7 |
| Synthetic rows, `τ_w = 10⁻⁴` | 2 | −16.4 | −16.4 | | | | −16.4 |
| | 3 | −20.8 | −20.8 | | | | −20.8 |

The columns scenes mirror the rows ones. What the trajectories show:

- **The sign settles at the first iteration wherever the gap is more than a
  few dozen units.** On every input but the seoul bull, the gap's sign after
  the first iteration of the first round is the winner's and never changes
  again, even on the synthetic scenes whose final gap is 20 units. On the seoul bull, a near tie
  of 7.4 units, the sign is stable only from the third iteration of the last
  round (pair 95 of 107).
- **An early lead is reversed on the seoul bull.** Rows, the winner, trails by
  18 to 25 units through the whole first round (60 iterations, stopped on the
  budget) and by 305 units at the third iteration of the second round, leads
  by its twentieth, trails again by 4.3 units at the first iteration of the
  third round, and leads from the third.
- **The first round under-reads the gap from a perturbed start.** Its noise
  level is measured at the start, before the poses settle: 0.64 to 0.94 px on
  the synthetic scenes against 0.09 to 0.17 px in the later rounds, and 0.95 px
  against 0.51 px on the raw solve. Divided by it, the synthetic scenes'
  first-round gaps are 0.2 to 108 units, about 50 times smaller than the same
  cost differences divided by the second round's noise level.
- **A round's start can jump.** The re-estimation between rounds sets a new
  start: on the seoul bull the columns solve starts its second round at a cost
  of 111796 units against 2813 at the end of the first, and is back to the
  rows solve's level within three iterations. A replay that read the gap at a
  round's start would see −109017 there.
- **The kept observations agree.** Both solves' trims kept the same count in
  every round of every input except the raw solve's last round, 85746 against
  85739.

### What the rule would do

Per LM iteration, with the budget term. Each cell is where the rule fires,
which solve it stops, and how many of that solve's iterations it saves, out of
the race's total; "never" saves nothing, and a stop of the winner is marked
**wrong**.

| Input | `T = 10`, `K = 3` | `T = 25`, `K = 3` | `T = 100`, `K = 3` |
|---|---|---|---|
| Dino candidate, `(inf, 1000)` | round 1, i = 12: columns, 15 of 54 | same | same |
| Dino candidate, default | round 1, i = 12: columns, 28 of 79 | same | same |
| Dino raw solve, default | round 1, i = 16: columns, 107 of 227 | same | same |
| Seoul bull, default | round 1, i = 18: rows, **wrong** | never | never |
| Synthetic rows, `τ_w = 4·10⁻³` | round 2, i = 4: columns, 5 of 28 | same | same |
| Synthetic rows, `τ_w = 10⁻³` | round 2, i = 5: columns, 4 of 27 | same | same |
| Synthetic rows, `τ_w = 3·10⁻⁴` | round 3, i = 1: columns, 3 of 26 | same | never |
| Synthetic rows, `τ_w = 10⁻⁴` | never | never | never |
| Synthetic columns, `τ_w = 4·10⁻³` | round 2, i = 2: rows, 7 of 30 | same | same |
| Synthetic columns, `τ_w = 10⁻³` | round 2, i = 5: rows, 4 of 27 | same | same |
| Synthetic columns, `τ_w = 3·10⁻⁴` | round 3, i = 4: rows, 0 of 25 | same | never |
| Synthetic columns, `τ_w = 10⁻⁴` | round 3, i = 2: rows, 1 of 24 | never | never |

The count `K` moves the stop by two iterations a step on the dino inputs: at
`T = 25` the candidate saves 30, 28 and 26 of 79 iterations over the default
schedule at `K` = 1, 3 and 5, and the raw solve 109, 107 and 105 of 227. On the
synthetic scenes, whose rounds end after three to six iterations, `K = 5`
saves 1 to 5 iterations where it fires and does not fire at `τ_w = 10⁻⁴` or on
the columns scene at `3·10⁻⁴`, and `K = 1` saves 4 to 9, firing in the first
round at `τ_w = 4·10⁻³`. No count stops the winner at `T = 25` or `T = 100`;
at `T = 10` every count stops the winner on the seoul bull, at iterations 16,
18 and 20 of its first round.

**The budget term is what keeps the rule from stopping the winner on the seoul
bull.** Without it, at `T = 25` and `K = 3`, the rule stops rows, the winner,
at the fifth iteration of the second round, where rows trails by 159 units
after trailing by 305; the budget term holds it off because rows is still
falling fast there. In the first round it is the threshold that holds the
rule off: once rows' decrease has slowed, the budget term is met from the
sixteenth iteration on, and the gap there, 21.4 to 21.5, is under 25 by a
margin of 3.5 units and over 10. Without the budget term the rule fires
earlier where the gap is large (the dino candidate at the third iteration,
saving 37 of 79; the raw solve at the third, saving 120 of 227), because the
budget term waits until the trailing solve's own decrease has nearly stopped,
which on the dino inputs is the end of the first round's fast descent.

**Per schedule round, the rule saves almost nothing.** With three rounds and
the rate read over whole rounds, whose first round falls by thousands of units,
`K = 1` fires at the last round's end, where there is nothing left to save, on
every input it fires on but the synthetic scenes at `τ_w = 4·10⁻³`, where it
fires after the second round and saves 4 iterations; `K = 3` and `K = 5` need
more round ends than there are and never fire. A
one-round `(inf, 1000)` schedule, which a derived prior runs as two, leaves no
round to save.

### Recommendation

Adopt the rule with changes, if two full solves cost too much: lockstep per
LM iteration with each round's end as a barrier between the two solves, the
budget term kept, `T = 100` rather than 25, and `K = 3`, with `D` divided by
the mean of the two solves' round-start noise levels squared, the pair at a
round's start skipped, and no stop while the two rounds' kept observations
differ in count. On these inputs that never stops the winner and saves the
same iterations as `T = 25` on every input but the synthetic scenes at
`τ_w = 3·10⁻⁴`, where `T = 25` saves 3 or none: about a third of the race's
iterations on the dino candidate (28 of 79), almost half on the raw solve (107
of 227), and a sixth to a quarter on the synthetic scenes with clear planted
rates. `T = 100` is preferred because on the seoul bull the threshold of 25,
not the budget term, held off a wrong stop in the first round, by 3.5 units.
Do not adopt lockstep per schedule round: it saves 4 iterations on two inputs
and nothing on the others.

What the measurement does not cover: one near tie (the seoul bull) carries the
evidence that a lead can reverse, and every capture has one camera; the
savings are counted in iterations of the stopped solve, which on the dino
inputs take a nearly constant 0.5 s each, not in the wall time of two solves
run on two threads.
