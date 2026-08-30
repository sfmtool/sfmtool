# Point Estimation

One batched operation that re-reads every point of a track set from its own
observations at one geometry and decides, per track, what the observations
support: a finite position, a bearing, or nothing. It is the decision layer
over the batch triangulation solve
([batch-triangulation-api.md](batch-triangulation-api.md)), and it is the
one implementation behind the bundle adjustment's inter-round re-estimation
and any caller that holds poses and a camera and wants the structure those
poses imply.

## What the solve leaves undecided

The midpoint solve answers one question, where the rays come closest, and
reports how well the depth was observed. It does not say whether that answer
should be used. Whether a track with parallel rays is a bearing or a defect,
whether a point behind a camera is kept for a later trim or demoted now,
whether a single observation still carries a direction, and whether a fresh
estimate has to reproject inside a bound before it counts: these are the
caller's rules, and every caller in the codebase had written its own copy of
them. This operation holds the rules once, as options, so that a caller states
its policy and the arithmetic is shared.

## Inputs

Two forms, both CSR over tracks in the order the caller supplies:

- **Rays.** Unit world rays and camera centres per observation, track offsets.
  This is the batch triangulation's own input and the form a caller uses when
  it has already built rays.
- **Observations.** Pixel positions, an image index and a track index per
  observation, the camera, and the poses (world-to-camera rotation and
  translation per image). The operation builds the world rays through the
  camera's `pixel_to_ray` and the inverse rotation, and the centres as
  `-Rᵀ t`, in observation order. An observation whose ray is not finite (a
  pixel outside the model's domain) is dropped from its track before any
  rule is read.

Per track the caller may also pass an incoming state: a position and a flag
saying the point is currently a direction. The state is read only by the
`marks` rule below and is never modified; the result is a new array.

## Rules

Each rule is an option with an off position. With every option off, the
operation is the batch triangulation solve with a finite point per track and
nothing else.

| rule | option | what it decides |
|---|---|---|
| **few** | `bearing` or `absent` | a track with fewer than two usable observations. `bearing`: its single ray is its direction, or the fixed fallback direction when it has none. `absent` (the off position): the estimate is NaN, the track is not observed at this geometry. |
| **marks** | incoming direction flags | a track flagged as a direction is not solved: its estimate is the normalized mean of its rays. Off, every track is solved. |
| **floor** | an angle | a track whose widest ray pair subtends less than the floor is THIN and becomes a bearing (the normalized mean ray). The pair angle is the minimum cosine over every ray pair of the track, read as a pairwise statistic and not from the solve's spectrum, so that a track's verdict depends on its rays alone and not on the count of them. Off, no track is thin. |
| **cheirality** | on / off | a solved point that lands behind any camera that observes it (non-positive depth along that camera's ray) is BEHIND and becomes a bearing. Off, the point is kept and the flag is reported. |
| **bar** | a pixel bound | a solved point that survives the rules above is reprojected through the camera at the same geometry; when the median finite residual over its observations exceeds the bound the track is OVER THE BAR and becomes a bearing. Requires the observation form. Off, no reprojection is read. |

Bearings are unit vectors with the point flagged as a direction. The
fallback for a track with no usable ray at all, and for a mean ray whose norm
is zero or not finite, is the camera convention's forward direction.

The rules are read in the order of the table. A track leaves at the first
rule that decides it, so a thin track is never solved, and a behind track is
never reprojected.

`few` heads the order because every rule under it needs at least two rays to
say anything: the floor needs a pair, cheirality and the bar need a solve, and
the mean of one ray is that ray, which is what `few = bearing` already returns.
The one case where the order is visible is a MARKED track with one observation:
under `few = absent` it is absent, not a bearing, which is what an adjustment's
own re-estimation does with it.

## Verdict and census

Per track the operation returns the estimate, the direction flag, and one
verdict: `finite`, `marked`, `thin`, `behind`, `over_bar`, `few`. Alongside,
a census of the counts per verdict, the number of tracks seen, and the median
triangulation angle (the widest pair angle) over the tracks that came out
finite. The census is what a caller records; the verdicts are what it filters
on.

The median angle is reported only where the `floor` is on. It is that rule's
own statistic, read once in the same pass; a caller with the floor off has not
asked for the `O(K²)` pass it costs, and gets no angle rather than paying for
one.

## Callers and their settings

- **Bundle adjustment, inter-round re-estimation.** `marks` on with the
  round's direction mask, `few = absent`, every other rule off: a direction is
  the mean back-rotated ray, a finite track is the midpoint, and a track with
  one observation is NaN in either family. The adjustment's trim, not this
  operation, decides what a behind point means.

  The adjustment holds no copy of the arithmetic: its round is this call. The
  grouping the call builds is a STABLE sort of the track index, so a track
  accumulates its observations in the order the caller listed them, and the
  adjustment's re-estimation is defined by the observation order it was handed
  rather than by a sort's tie-breaking.
- **Admitting new tracks at a settled geometry.** `floor` at the caller's
  angular bound, `cheirality` on, `bar` at the adjustment's final pixel
  bound, `few = bearing`, `marks` off. Every candidate track is estimated
  from scratch at the current poses and lens.
- **Re-estimating every point after an adjustment.** As the previous setting
  with `bar` off: the adjustment has already trimmed the observations the
  bound would cut.
- **Demoting unconstrained points in a stored reconstruction.** `floor` on,
  everything else off: finite points whose rays do not cross become
  bearings, nothing else moves.

## Determinism

Tracks are independent and may be solved in parallel; the output order is the
input order and the per-track arithmetic is sequential over that track's
observations in their given order, so the result does not depend on the
scheduling. No rule samples and nothing reads a clock. The same inputs give
the same bytes.

## Binding

`estimate_points(...)` in the reconstruction module of the bindings, taking
either form (rays and centres, or pixels with images, points, camera and
poses) as NumPy arrays, the incoming state as optional arrays, and the rules
as keyword arguments with the off position as each default. It returns the
positions as an `(n, 4)` xyzw array (w = 1 finite, w = 0 bearing, NaN rows
for `absent`), the verdict codes as an integer array with the code table
exposed as a constant, and the census as a dict. Arrays are accepted in
either memory order and returned C-contiguous.

The observation form takes the poses as world-to-camera quaternions and
translations, and the track count, since the observation indices alone do not
say how many tracks the result indexes. The quaternions are used as given
rather than renormalized, so a caller holding a unit quaternion gets its own
rotation back; the binding refuses one whose norm is not unit.

The in-front flag comes back beside the verdicts, which is what makes
`cheirality` off a reading rather than a silence.

## Testing

- **Solve parity.** With every rule off, the estimate of a track with two or
  more usable rays equals the batch triangulation's point bit for bit on the
  same rays. A shorter track is decided by `few`, whose off position is
  `absent`, so the solve has nothing to compare against there.
- **Adjustment parity.** The bundle adjustment's re-estimation is this call
  with the adjustment's settings, so its staged-loop outputs are a function of
  the observation arrays it is given: two runs on the same arrays agree bit for
  bit, and a track's estimate reads its own observations in the listed order
  whatever else the list holds.
- **Rule order.** A track that is both thin and behind reads `thin`; a track
  under the bar with a behind camera reads `behind`; a marked track is never
  solved even when its rays cross.
- **Few.** One usable ray: `bearing` returns that ray; `absent` returns NaN.
  No usable ray: the fallback direction.
- **Floor edge.** A pair exactly at the floor is not thin (strict comparison
  on the cosine), and the verdict is the same whether the track has two rays
  or twenty at that angle.
- **Bar.** The median is over finite residuals only; a track whose
  reprojections are all non-finite is over the bar.
- **Determinism.** Two runs on the same arrays are byte-identical, with and
  without parallelism.
