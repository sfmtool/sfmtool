# Free, ranged and held points in bundle adjustment (amendment)

**Status:** Draft. Decided: the three point kinds and their names, that a free
point's representation is chosen by the inter-round re-estimation and can cross
in either direction, that a ranged point keeps its distance from a reference
and lets the triangulation own its direction (infinite range being the
direction case), that a held point contributes residuals but owns no
parameters, that "at infinity" in a file means "the estimate is a direction" and
nothing more, and that the change ships behind a switch whose off position is
bit-identical to the kernel as it stands. Not decided: the constant in the
noise-floor angle, whether a free point should later be parameterized so it can
cross inside a round rather than between rounds, and how the viewer marks a
ranged or held point.

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

## The three kinds

Every point is one of three kinds, orthogonal to its current representation:

- **Free.** The solve owns the point. Its representation, finite `(x, y, z)` or
  direction `d`, is whatever its rays support at the current geometry, decided
  by the re-estimation between rounds and allowed to change in either direction
  as the poses move. `w = 0` on output reports that the final estimate is a
  direction.
- **Ranged.** The caller owns one number, the point's distance `r` from a
  reference the caller names; the solve owns the direction. The point is
  `X = O + r · d` with `d` a unit vector and `O` the reference, and `d` is the
  point's only parameter, two degrees of freedom on the sphere. `r = ∞` is
  allowed and is exactly a direction: the point contributes rotation-and-lens
  rows and nothing else. A finite `r` is how a landmark whose distance is known
  (a surveyed range, a map distance from the capture point) enters without a
  georeferenced frame, and how a far-field track can be carried at a depth the
  caller trusts more than a 5 m baseline can measure, while the solve still
  decides where on the sky it sits.
- **Held.** The caller owns the point. Its homogeneous coordinate, finite or
  direction, is fixed for the whole solve. Its observations still form
  residuals and still feed the camera and lens blocks, but the point has no
  parameters and no Schur block, and re-estimation skips it. A held point is
  how a landmark with a known position in the solve's frame enters.

The kinds nest: a held point is a ranged point that has also given up its
direction, and a ranged point at infinite range with no reference is what the
standing kernel calls a marked point. `protected` is unchanged and orthogonal:
it is about whether an observation can be trimmed, not about whether a point
can move.

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

## Ranged points: a direction at a distance

A ranged point is `X = O + r · d`, with `r` fixed by the caller and `d` the
parameter. `r = ∞` needs no reference and the point degenerates to a
direction. For a finite `r` the reference `O` is a point the solve knows the
position of *relative to the cameras*, which in an adjustment whose cameras
all move means a function of the camera poses:

- **A camera's centre**, `O = C_k = −R_kᵀ · t_k`: the distance is measured
  from camera `k` wherever the solve puts it. This is the form for "the
  landmark is 1045 m from where the photograph was taken".
- **The mean of a set of camera centres**, `O = (1/|K|) · Σ_{k∈K} C_k`: the
  distance from a capture station whose frames sit within a few metres of each
  other, where no single frame is the survey point.

A fixed world coordinate is deliberately not a reference. The adjustment has
gauge freedom, and a range measured from a point that the cameras are free to
move away from constrains nothing about the cameras: it is a held point with
its direction released, meaningful only if some other constraint has already
pinned the frame. When the frame is pinned (a held point, or the caller's own
gauge fixing), the caller can express the same thing as a camera-referenced
range, so the kernel offers only the two forms above.

Residual and derivatives, for a finite `r`:

- `uv = ray_to_pixel(R_i · X + t_i)` with `X = O + r · d`, the finite
  projection with the point's position substituted.
- **Point block.** `d` perturbs in its 2-DOF tangent plane exactly as a
  direction does, `d ← normalize(d + B(d) · δ)`, and the point's Jacobian
  block is `r · J_X · B(d)` where `J_X = ∂uv/∂X` is the finite point's
  position Jacobian; its Schur block is 2×2.
- **Observing camera's block.** The rotation and translation blocks of a
  finite point at `X`, unchanged.
- **Reference cameras' blocks.** `X` depends on every pose in the reference
  set through `O`, so each observation of the point also contributes
  `J_X · ∂O/∂(pose_k)` to camera `k`'s block for every `k ∈ K`, with
  `∂C_k/∂t_k = −R_kᵀ` and `∂C_k/∂ω_k` the derivative of `−R_kᵀ · t_k` under
  the kernel's rotation perturbation, scaled by `1/|K|` for a mean reference.
  When the observing camera is itself in `K`, both contributions land in the
  same block and add. After the Schur complement over the point's 2×2 block,
  the reduced camera system gains couplings between every observing camera and
  every reference camera; the reduced system is already dense and solved by a
  generic factorization, so this changes what is accumulated, not how it is
  solved.
- **At `r = ∞`** every one of these reduces to the standing spec's direction
  rows: `r · J_X · B(d)` becomes the direction's tangent Jacobian in the
  limit, the translation block and the reference blocks vanish.

Because `r` is held in the solve's own units, ranged points carry metric
scale into an adjustment that otherwise has none: several ranged points on
one reference set fix the scale gauge, and a range that disagrees with the
caller's other scale evidence (a walk length, say) shows up as residual
rather than being absorbed. That is the intended behaviour: a surveyed range
is better scale evidence than a paced baseline.

Re-estimation between rounds keeps `r` and re-solves `d`: at infinite range
the normalized mean of the back-rotated rays, as today; at a finite range the
direction from `O` that minimizes the reprojection error at the current
geometry, which the point-estimation operation gains as a rule (`range` with
an off position, read before `floor`). A ranged point never crosses kinds: its
representation is finite whenever `r` is, a direction when `r = ∞`.

Trim, `min_track` and `min_obs` treat a ranged point's observations exactly
like any other's.

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
finite point, and the value is an estimate. Two optional per-point columns are
added:

- `points3d/kind.{N}.uint8.zst`: `0` free, `1` ranged, `2` held. Absent means
  every point is free, so every existing file reads unchanged.
- `points3d/range.{N}.4.float64.zst`: for a ranged point, `(r, ox, oy, oz)`
  with `r = +inf` for a direction; the reference is a world point when the
  row's `ox, oy, oz` are finite and a camera index, stored in `ox`, when
  `oy = oz = NaN`. Rows of free and held points are NaN. Absent when no point
  is ranged.

`infinity_point_count` and every other count are unchanged; they count
representations, not kinds.

## Binding

```python
bundle_adjust(camera, quaternions_wxyz, translations, points, uv, obs_image,
              obs_point,
              point_at_infinity=None,   # (n_pt,) bool: the INITIAL representation
              held=None,                # (n_pt,) bool: a held point keeps its
                                        # coordinate; None = all free
              range=None,               # (n_pt,) float: a ranged point's distance,
                                        # +inf for a direction, NaN = not ranged
              range_origin=None,        # (n_pt,) int camera index, or (n_pt, 3)
                                        # world points; ignored where range is NaN
                                        # or +inf
              free_points_cross=False,  # True: free points are re-estimated with
                                        # marks off and the noise floor; False:
                                        # the mask is honoured for the whole solve
              noise_floor_scale=2.0,    # the constant c in θ_floor = c·s/f
              protected=None, protected_loss_scale=3.0,
              opt_f=False, opt_k1=False, opt_bspline=False,
              schedule=..., max_iters=60, min_track=2, min_obs=12)
```

The result gains `point_at_infinity` (`(n_pt,)` bool), the representation each
point ended with; a held point's entry is its input value, a ranged point's is
`r = +inf`. With `held=None`, `range=None` and `free_points_cross=False` the
kernel is bit-identical to the standing spec,
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
- **Ground-truth construction.** A landmark with a surveyed distance from the
  capture point enters as a ranged point referenced to the capture's camera
  (or its centroid as a world point); one with a known position in the solve's
  frame enters held; the near field is free; the far field is free and reports
  whichever representation it converges to. The five GPS-pinned landmarks of
  the `south_lake_union_parallax` capture, whose distances (0.9-3.4 km) are
  known but whose bearings are not georeferenced to the solve, are the first
  acceptance case, as ranged points.
- **Viewer.** A ranged or held point wants a visible mark; how is not decided
  here.

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
- Ranged points: the returned position sits at exactly `r` from the reference
  (camera-referenced ones re-read at the final pose); at `r = +inf` a ranged
  point reproduces a marked direction bit for bit; a planted landmark at a
  wrong initial direction but the true range converges to the true direction
  while a free point at the same start converges to a wrong depth.
- Acceptance on real data: with the pins ranged at their surveyed distances,
  the near field of the walk capture stays sub-pixel, the pins' directions
  agree with their GPS mutual angles, and the free far field triangulates near
  its surveyed distance rather than at a shell.

## Non-goals

Soft priors (a held depth with a variance) and per-point covariance output.
Both are natural follow-ups once held points exist; neither is needed for the
cases above.
