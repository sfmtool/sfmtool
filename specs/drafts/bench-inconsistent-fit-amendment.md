# When a track's observations disagree with the cameras (amendment)

**Status:** Draft

Amends [`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Fitting" and § "Moving between the stages", and
[`../gui/edits/bundle-adjust.md`](../gui/edits/bundle-adjust.md). Both point
back here.

A person building a track on the bench can arrive at observations that are
right, by eye and by correlation, and that no 3D point can explain, because the
camera poses they are read through are wrong. The fit has no answer for that
case today: it places the point where the rays come closest, finds that every
observation now sits outside its measuring bound, and returns a track with a
new position and no measurements. To the person it reads as the fit corrupting
a good track. This draft proposes what the fit does instead, and sets out the
larger question it opens: how hand-verified observations that contradict the
poses get back into a solve that can correct them. The first part is a
proposal. The second is a list of open questions with the options as they
stand.

## The case this is drawn from

A landmark roughly 120 m away, observed in three images of one capture. Two of
the cameras stand 0.10 m apart; the third stands 3.2 m from both.

| Pair | Baseline | Angle between rays | Depth the pair gives |
|------|----------|--------------------|----------------------|
| A, B | 0.10 m | 0.47 deg | 9 m |
| A, C | 3.20 m | 1.58 deg | 115 m |
| B, C | 3.14 m | 1.16 deg | 154 m |

With A and B alone the fit places the point at 9 m with 0.3 px residuals, and
every number on the track looks excellent. That placement rests on a 0.47 deg
angle across a 10 cm baseline, where a landmark at 120 m would subtend 0.05
deg: the angle is rotation error between the two poses, read as depth. Adding C
moves the point to about 120 m, leaves residuals of 16, 16 and 5 px, and every
observation fails its measuring bound. Roughly 0.4 deg of rotation error sits
somewhere among the three poses, and one track cannot say where.

Two things follow. The two-view placement was the fragile one and nothing said
so. And the three-view result is the more truthful and is the one that looks
like damage.

## Part 1: the fit refuses, and says what it found

### The fit is all-or-nothing

A fit either returns a track whose `in` observations are measured against the
placement it found, or returns the track it was given, unchanged, with a
report. It never returns a moved point with its measurements gone.

The refusal is decided on the placement, before anything is written: the fit
solves the position as it does now, projects it into every `in` observation,
and compares each projection with that observation's keypoint. The fit is
**inconsistent** when the observations cannot be measured at that placement,
which today means a keypoint farther from the projection than the measuring
bound. What the bar should be is open question 1.

### What the report carries

The report is the evidence a person needs to decide whether the track or the
cameras are wrong:

- **The pairwise table** above, for every pair of `in` observations: baseline,
  ray angle, the depth the pair gives, and the gap between the two rays at
  closest approach. A pair whose baseline is small against its depth is marked
  as carrying no depth information, which is what would have flagged the
  two-view placement in the case above.
- **The residual of each observation** at the best joint placement.
- **The leave-one-out reading**: the fit is re-solved with each `in`
  observation left out in turn. If exactly one omission makes the rest
  consistent, the report names that observation as the suspect. If none does,
  the report says the observations disagree as a set, which is a statement
  about the poses and not about any one sighting. With three observations the
  leave-one-out is three two-view solves and each is trivially consistent, so
  the reading is reported as undecidable below four observations, and the
  pairwise table is what the person has.

Track View shows the report in place of the measurements it could not
make, and the Action Log records its summary line. The MCP reply carries the
same structure.

### A separate defect the case exposed

The fit's refusal message read "its seed sits 71 px from the projection, beyond
the 64 px bound" for an observation whose keypoint was 16 px from the
projection. Whatever the fit measures that distance from, it is not the
keypoint. This is a defect to fix on its own, not a design question, and it is
recorded here because under the keypoint distance all three observations of the
case would have been inside the bound and measured, with honest 16 px
residuals. The refusal above is still needed for the cases that are genuinely
outside.

### Placements that rest on no baseline

Independently of refusal, a fit whose every pair is marked as carrying no depth
information reports that beside the position, so a 0.3 px two-view track at the
wrong depth stops reading as excellent. Whether such a placement should instead
be made a bearing is open question 2.

## Part 2: getting the evidence into a solve (open)

A track the fit refuses is a set of observations a person has verified as one
landmark, with no 3D position the current cameras allow. The cluster stage
already means exactly "these are the same landmark, and no claim about where it
is", so staging the track back to a cluster is the natural resting state, and
the fit's report can offer it. But a cluster cannot be committed, so on its own
this parks the evidence where no solve can read it. The questions below are
about the route from there to corrected poses.

### Open question 3: how does the evidence reach bundle adjust?

- **(a) Bundle adjust reads the bench.** The viewer's bundle adjust takes the
  bench's items, clusters included, as extra tracks for that solve, each
  initialised at its best joint placement. No format change; the evidence stays
  on the bench, visibly unresolved, until a solve absorbs it, after which the
  fit succeeds and the track commits normally. The cost is that the evidence
  lives only in the session until then, since the bench is not saved with the
  `.sfmr`.
- **(b) Commit with a mark.** The track commits at its best joint placement
  with its honest residuals, and the point carries a "verified" mark that the
  solve reads. The evidence survives a save. The cost is a format column and a
  reconstruction that, between the commit and the solve, holds a point its own
  cameras contradict.

These are not exclusive: (a) can ship first and (b) follow if evidence needs to
outlive a session.

### Open question 4: verified observations and trimming

Whichever route, the bundle adjust's staged trimming reads a 16 px observation
as an outlier and drops it, which discards precisely the evidence that
contradicts the solve. Verified observations need to be exempt from trimming and
carry a loss scale of their own; the kernel's protected loss scale is the
existing mechanism. Open: whether protection is per observation or per track,
and what stops a person's one wrong placement, now untrimmable, from bending a
solve. A candidate guard is that a protected track is still reported, never
silently dropped, when its residual after the solve stays large.

### Open question 5: how many tracks, and where?

One track cannot attribute the error to a pose. A dozen can. Once verified
tracks exist, the viewer can show, per image, the systematic residual over
verified tracks alone (mean vector and spread), which turns "somewhere among
these three poses" into "this image is rotated 0.4 deg". Open: whether this is a
column in the image list, an overlay in the 3D view, or a panel of its own, and
whether it should propose where the next verified track would be most
informative (the image pairs with the fewest verified tracks across them).

### Open question 6: does the solve belong on the bench?

A narrower alternative to a whole bundle adjust: a bench step that solves the
poses of just the images a refused track touches, against that track and the
verified tracks around it, as a preview the person accepts or discards. This
would make the loop "verify a track, watch the cameras move" immediate. It also
makes it easy to overfit a pose to a handful of tracks, which is the failure
the capture-wide solve exists to avoid.

### Open questions 1 and 2, from Part 1

1. **The bar for "inconsistent".** The measuring bound is a fixed pixel count.
   A bar derived from the track itself (the keypoint uncertainty the
   observations already carry, scaled) would say the same thing at every image
   size and focal length.
2. **A placement with no baseline.** Report it, as proposed; or place the point
   as a bearing, which is what the rays actually support; or refuse to give a
   two-view track whose pair carries no depth information a finite position at
   all.
