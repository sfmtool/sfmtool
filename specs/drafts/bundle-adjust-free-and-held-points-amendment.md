# Free and held points in bundle adjustment (amendment)

**Status:** Draft. Decided: the two point kinds and their names, that a free
point's representation is chosen by the inter-round re-estimation and can cross
in either direction, that a held point contributes residuals but owns no
parameters, that "at infinity" in a file means "the estimate is a direction" and
nothing more, and that the change ships behind a switch whose off position is
bit-identical to the kernel as it stands. Not decided: the constant in the
noise-floor angle, whether a free point should later be parameterized so it can
cross inside a round rather than between rounds, and how the viewer marks a held
point.

Amends [`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
(the "Points at infinity" and "Protected observations" sections and the binding),
adds a caller setting to
[`../core/reconstruction/point-estimation.md`](../core/reconstruction/point-estimation.md),
and adds one optional column to
[`../formats/sfmr-file-format.md`](../formats/sfmr-file-format.md).

## What the kernel does today, and what that costs

The adjustment takes a per-point mask, `point_at_infinity`. A marked point is a
direction for the whole solve: its rows depend on rotation and lens only, and
the inter-round re-estimation runs the point-estimation operation with `marks`
on, so the mask is never revisited. An unmarked point is finite for the whole
solve, whatever its rays support. The mask is therefore an instruction, not an
estimate: the caller decides each point's kind before the first residual is
formed, and the kernel cannot contradict it.

Two consequences follow, and both have been measured on real captures.

- A track marked at infinity that carries parallax has no depth to absorb it.
  Its residual is weighted exactly like a finite observation's, and its only
  channel is rotation and the shared lens, so the solve bends those to fit it.
  On a hand-held pan with a 10-20 cm sway, half of the tracks the release marked
  as bearings carried 3-15 px of parallax at the poses that fit the true far
  field; on a 5 m walk, 325 far-shore tracks at 0.9-3.4 km carried 4-15 px. In
  each case the rotations moved 0.5-1.7 deg and the rim of the lens map by
  30-50 px to accommodate rows that should have carried depth.
- A point whose position is known better than the solve can know it (a
  surveyed landmark, a bearing certified by an external reference) has no
  representation. `protected` only exempts an observation from trimming and
  widens its loss scale; the point still moves.

The mask also overloads one word. In a stored reconstruction `w = 0` means
"this track's rays are parallel to within noise"; handed to the adjustment it
means "hold this at infinity". Those are different statements, and a point can
satisfy one without the other.

## The two kinds

Every point is one of two kinds, orthogonal to its current representation:

- **Free.** The solve owns the point. Its representation, finite `(x, y, z)` or
  direction `d`, is whatever its rays support at the current geometry, decided
  by the re-estimation between rounds and allowed to change in either direction
  as the poses move. `w = 0` on output reports that the final estimate is a
  direction.
- **Held.** The caller owns the point. Its homogeneous coordinate, finite or
  direction, is fixed for the whole solve. Its observations still form
  residuals and still feed the camera and lens blocks, but the point has no
  parameters and no Schur block, and re-estimation skips it. A held point is
  how a surveyed landmark, a certified bearing, or a far-field track carried at
  a depth the caller trusts more than the solve, enters the adjustment.

`protected` is unchanged and orthogonal: it is about whether an observation can
be trimmed, not about whether a point can move.

## Free points: crossing between representations

Within a round nothing changes: a finite point perturbs in three Euclidean
degrees of freedom and a direction in the two of its tangent plane, with the
Jacobian blocks the standing spec gives. The crossing happens where the
representation is already re-read, the inter-round re-estimation, which is one
call of the point-estimation operation with a new caller setting:

- `marks` **off** for free points: every free track is solved from its rays.
- `floor` at the **noise-floor angle** `θ_floor`: a track whose widest ray pair
  subtends less than it is a bearing (the normalized mean ray). `θ_floor` is
  derived, not fixed: the parallax the stage's own residual scale cannot
  distinguish from noise, `θ_floor = c · s / f` with `s` the current stage's
  loss scale in pixels, `f` the current focal, and `c` a small constant (the
  candidate default is 2). A wide-baseline stage therefore keeps more tracks
  finite than a tight one, and a track near the floor is re-decided every round.
- `cheirality` on: a solved point behind any observing camera is a bearing,
  which is what the trim would otherwise have to discover through its
  penalized residual.
- `few = absent`, as today.

A direction whose rays open past `θ_floor` as the poses settle becomes finite
at the next re-estimation; a finite point whose rays close below it becomes a
direction. Both are ordinary outcomes, not events. The first round has no
re-estimation, so the caller's initial representation (the `point_at_infinity`
input, kept for that purpose) is what the first linearization uses.

Not decided: an inverse-depth parameterization, `X = d / ρ` with `ρ ≥ 0` and
`ρ = 0` the direction, would let a point cross inside a round and make the two
representations one Jacobian. It is a larger change to the kernel and the
Schur structure, and the between-round crossing above captures the behaviour
that matters. This draft does not propose it; it records it as the natural
next step if the between-round rule turns out to lag the poses.

## Held points: residuals without parameters

A held point's observations project exactly as today for its representation:
`uv = ray_to_pixel(R_i · X + t_i)` for a held finite point, `ray_to_pixel(R_i ·
d)` for a held direction. The camera Jacobian blocks (rotation, translation for
a finite point, lens) are formed as for any observation. The point's own
Jacobian block is absent: nothing is accumulated into a point block, no Schur
complement is taken for it, and its coordinates are copied through to the
output.

Trim applies to a held point's observations as to any other's; a held point
does not make an observation `protected`, and a caller that wants both marks
both. `min_track` survival is counted for a held point too, and a held point
that loses every observation simply contributes nothing; it is still returned
unchanged.

The translation-observability rule extends: an image's translation is frozen
for a round when no surviving observation of that image carries a translation
Jacobian. Free and held directions carry none; a held finite point does, so an
image observing one held finite point and otherwise only directions keeps its
translation live, which is what a surveyed landmark is for.

## Format

`positions_xyzw` keeps its meaning exactly: `w = 0` is a direction, `w ≠ 0` a
finite point, and the value is an estimate. One optional per-point column is
added:

- `points3d/held.{N}.uint8.zst`: `1` for a held point, `0` for a free one.
  Absent means every point is free, so every existing file reads unchanged.

`infinity_point_count` and every other count are unchanged; they count
representations, not kinds.

## Binding

```python
bundle_adjust(camera, quaternions_wxyz, translations, points, uv, obs_image,
              obs_point,
              point_at_infinity=None,   # (n_pt,) bool: the INITIAL representation
              held=None,                # (n_pt,) bool: a held point keeps its
                                        # coordinate; None = all free
              free_points_cross=False,  # True: free points are re-estimated with
                                        # marks off and the noise floor; False:
                                        # the mask is honoured for the whole solve
              noise_floor_scale=2.0,    # the constant c in θ_floor = c·s/f
              protected=None, protected_loss_scale=3.0,
              opt_f=False, opt_k1=False, opt_bspline=False,
              schedule=..., max_iters=60, min_track=2, min_obs=12)
```

The result gains `point_at_infinity` (`(n_pt,)` bool), the representation each
point ended with; a held point's entry is its input value. With `held=None` and
`free_points_cross=False` the kernel is bit-identical to the standing spec,
which is the parity requirement for the transition. When the seed and release
paths have moved to `free_points_cross=True`, the default flips and the switch
stays as the kill switch.

## Consumers

- **Seed release and relaxation.** The release admits every track as free and
  lets the re-estimation classify; the bearing-admission decision that today
  happens once, before the adjustment, becomes the noise-floor rule applied
  every round at the current poses. A rotation-only opening stays as it is (its
  translations are frozen because nothing carries a translation Jacobian) and
  gains nothing but the ability to promote a track once a later stage supplies
  baseline.
- **Ground-truth construction.** Surveyed landmarks enter as held finite points
  in the world frame; the near field is free; the far field is free and reports
  whichever representation it converges to. The five GPS-pinned landmarks of
  the `south_lake_union_parallax` capture are the first acceptance case.
- **Viewer.** A held point wants a visible mark; how is not decided here.

## Testing requirements

- Parity: `held=None, free_points_cross=False` reproduces the standing kernel
  bit for bit on the existing fixtures, including the infinity and protected
  suites.
- Crossing, both directions, on synthetic scenes: a track started as a
  direction whose rays open as the poses move ends finite; a finite track whose
  rays close ends as a direction; the reported `point_at_infinity` matches the
  output `w`.
- Noise floor: the same track is finite at a wide-baseline stage and a
  direction at a tight one, and the boundary moves with `noise_floor_scale`
  and with the focal.
- Held points: coordinates are returned unchanged to the bit; their residuals
  appear in `residual_norms`; a scene whose only translation evidence is one
  held finite point solves that image's translation.
- Acceptance on real data: with the pins held, the near field of the walk
  capture stays sub-pixel and the free far field triangulates at its surveyed
  distance to within the pins' own uncertainty.

## Non-goals

Soft priors (a held depth with a variance) and per-point covariance output.
Both are natural follow-ups once held points exist; neither is needed for the
cases above.
