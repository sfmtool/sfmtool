# Point or Bearing by Likelihood Ratio: Inverse-Depth Free Points

**Status:** Draft. Proposes solving bundle adjustment's free points in inverse
depth, so that whether a track is a finite point or a bearing becomes a storage
decision at the end of the solve rather than a representation the solve carries
between rounds. Everything else this draft once proposed is built and specified
in the standing specs: the point-or-bearing likelihood-ratio test, its theory
and the measured noise level in
[core/reconstruction/batch-triangulation-api.md](../core/reconstruction/batch-triangulation-api.md)
§ "Point or bearing" and § "The measured noise level"; reclassification,
discovery, the bench and the reports, which decide on it, in the same spec's
§ "Consumers"; and bundle adjustment's crossing between rounds, which decides
on it at the noise level each round measures, in
[core/geometry/bundle-adjustment.md](../core/geometry/bundle-adjustment.md)
§ "Free points: crossing between representations" and the `likelihood` rule of
[core/reconstruction/triangulation-rules.md](../core/reconstruction/triangulation-rules.md).
Amends bundle-adjustment.md.

## Purpose

A free point in bundle adjustment is a finite point (three Euclidean degrees of
freedom) or a direction (two on the unit sphere), and it can change between the
two only at an inter-round re-estimation, where the point-or-bearing test reads
its rays at the geometry the last round settled on. That geometry was solved
with the point in the representation it had, and the solve fits that
representation: over ten cameras and 300 points with 0.76 px of noise, a track
1,500 units out whose rays score 33 at the true poses scores 38 at the end of a
solve that started it finite and 6 at the end of one that started it as a
direction. On weaker structure the effect is larger: over six cameras and 40
points, one round's solve bends the poses by up to half a degree to fit a far
track as whichever of a point or a direction it was given, and the next
re-estimation reads that bend back, so the track never crosses. A borderline
track therefore keeps the representation it started with, and a marginal one
needs several rounds to settle (on a Kerry Park solve with 1 px of keypoint
noise, the points changing at each of the seven re-estimations of an
eight-round schedule are 137, 39, 26, 14, 7, 6 and 7). A round that stops on its
iteration budget far from convergence makes it worse: its inflated level makes
points with a depth directions, the next round's solve bends the poses to fit
them, and recovery waits for the level to fall (bundle-adjustment.md § "Free
points: crossing between representations").

Solving free points as `(u, ρ)` about a fixed anchor removes the choice from
the solve: `ρ = 0` is a value the solve can reach, so a point moves between near
and infinity within a round, and far points stay well-conditioned because the
inverse depth is close to linear in the observations there. The representation
the file stores becomes the verdict of `is_finite` at the end.

## Proposal

- **Parameter blocks.** A free point's block is `(u, ρ)` with `u` in the
  tangent plane of the unit sphere (two degrees of freedom, as a direction
  today) and `ρ ≥ 0`, three in all as a finite point today, so the Schur
  complement keeps its 3×3 blocks. The point is `a + u / ρ`, and its
  observation from camera `i` projects the direction `u + ρ (a − cᵢ)`, which is
  the direction case at `ρ = 0`. This is the parametrisation the test's point
  fit already uses (`fit_point_and_bearing`).
- **The anchor.** Fixed per point for the whole solve, so that `ρ` keeps its
  meaning across rounds: the centroid of the observing cameras' centres at the
  input poses, as the point fit's default. A ranged or held point keeps its
  own parametrisation.
- **The bound.** `ρ` is clamped at 0 inside the damping ladder, as the point
  fit clamps it: a step that would take `ρ` negative is replaced by a step in
  `u` alone.
- **Storage.** At the end of the solve each free point is scored at the final
  poses and the final round's measured noise level, and stored finite at
  `a + u / ρ` where `is_finite` says so and as the bearing `u` otherwise. The
  inter-round crossing is no longer needed for free points.

## Open questions

- **Translation observability.** A direction carries no translation Jacobian,
  which is what freezes the translation of an image observing only directions.
  At `ρ` near 0 the translation column is small but not zero; whether the
  reduced system needs the same pinning, or a damping that reads the column's
  size, is open.
- **The anchor under large pose updates.** A fixed anchor taken at the input
  poses can sit far from the observing cameras after a rough start has moved
  them; whether to re-anchor between rounds, at the cost of `ρ` changing meaning,
  is open.
- **The level at the end of an unconverged solve.** The storage decision reads
  the final round's level, which a final round that stops on its iteration
  budget inflates just as an early one does: on the Kerry Park solve with five
  iterations a round and the crossing off, the final level is 2.0 px, at which
  the test would make 453 points directions, against about 160 at the
  converged level. Whether the decision should wait for convergence, or read a
  level that discounts pose error, is open.
- **What the crossing leaves at a converged end.** From a 4° and 20% start
  with the default 60 iterations, the Kerry Park solve shows no mass flip, but
  its result still disagrees with the test at its own level (0.95 px) on 142
  points, 34 to promote and 107 to demote, with 26% of its points made
  directions: the residue of decisions taken at round boundaries, which
  inverse-depth points that cross within a round would not leave.
- **Cost.** The inverse-depth Jacobian has one more chain-rule factor than the
  Euclidean one; whether the per-iteration cost is measurable on the largest
  inputs is to be measured.

## Testing

- The crossing tests of bundle-adjustment.md hold with the crossing replaced by
  the end-of-solve verdict: a near cloud started at infinity ends finite, a far
  track started finite ends a direction at the measured noise, and a borderline
  track's verdict no longer depends on its starting representation.
- A far point started at a wrong depth converges to the same verdict and
  direction from either starting representation.
