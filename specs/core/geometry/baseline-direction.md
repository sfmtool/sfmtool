# Baseline Direction from Ray Coplanarity

**Status:** Implemented in
`crates/sfmtool-core/src/geometry/baseline_direction.rs`, bound as
`sfmtool._sfmtool.geometry.baseline_directions`. See "What consumes it" for
the obligation the exactness of the directions places on a caller.

With both frames' rotations known, the direction between their centres is
readable from correspondences alone, with no depths and no translation solve.
The baseline `b = c_j - c_i` is coplanar with every point's two world rays:

```
b . (u_i x u_j) = 0
```

so `b` is the null space of the matrix whose rows are those normals, one row
per shared point. Every edge of a graph is one such solve, and the whole graph
is one call.

## A row is worth the baseline it saw

The normal `u_i x u_j` has norm `sin(parallax angle)`, which is literally how
much baseline that point measured. A row whose parallax falls inside an angular
bound carries none: inside the bound the two rays are the same ray and their
cross product is noise about a zero vector. Those rows are DROPPED, not
down-weighted, because a down-weighted noise row still votes.

The retained rows are normalized to unit length before the fit, so a single
wide pair cannot carry the solve on the strength of its own parallax; the
parallax has already done its work by deciding which rows are in.

An edge with fewer than three retained rows states no direction at all and
comes back as nothing. Three is the count a null space in three dimensions
needs, not a quality bar.

## The fit is refit on its own best rows

The null vector is the right singular vector of the smallest singular value of
the retained rows. It is then refit for a fixed number of rounds, each round
keeping the fraction of rows whose coplanarity residual against the current
direction is smallest, never fewer than three. The kept count is taken over the
retained rows, not over the previous round's survivors, so the trim does not
shrink round on round.

The residual that ranks rows is read over EVERY retained row, including the ones
the last round dropped, so a row the fit wandered away from can come back.

## The sign is cheirality

The null space is a line, and the two directions along it describe the two
constellations that reflect into each other. With one centre at the origin and
the other at `+d`, each retained row's point is placed at the closest approach
of its two rays and its depth along each read; the direction that puts more
points in front of both cameras is the one kept. The winning fraction is
reported beside the direction, so a caller can see whether the vote was a
majority or a coin toss.

## What comes back

Per edge: the unit direction, how many rows the edge was given and how many
cleared the bound, the conditioning of the null space (the second-smallest
singular value over the smallest, on the rows the final fit kept), the median
and maximum parallax in degrees, the cheirality fraction, and the median
absolute coplanarity residual in radians. The maximum parallax is read over
every row, the dropped ones included, so a caller can tell an edge with no
parallax from one whose parallax is concentrated in a few rows.

Conditioning is infinite where the smallest singular value is exactly zero,
which is what a noiseless edge produces. A caller reading it as a trust weight
should saturate it rather than use it raw.

## Determinism

Edges are independent and are solved in parallel; the output is in the input
order and each edge's arithmetic is sequential over its own rows. The trim
count uses banker's rounding, the same rule Python's `round` applies, so a
fraction landing exactly on a half row keeps the same count as the
implementation this was ported from. Ties in the residual ranking are broken by
row order. Two runs on the same arrays produce the same bits.

## What consumes it

The directions are exact to the last bit of the decomposition, which matters
for what consumes them. A translation averaging over a graph of such edges is
ill-posed when every edge carries the SAME direction, which is what a camera
moving along a straight line produces: the averaging minimizes the part
of each baseline perpendicular to its measured direction; with all directions
equal, that part is empty for every arrangement of the cameras along the line,
so their spacing is unconstrained and only the total scale is fixed by the gauge.
Fed exactly colinear directions such an averaging returns arbitrary positions,
and an implementation whose directions carry rounding asymmetries can appear to
resolve the spacing from those asymmetries rather than from any measurement.
A consumer therefore has to be well-posed on its own terms: recognize a graph
whose directions leave the spacing free, and resolve it from something the
directions do not carry (the pairs' own depths give the relative scale between
neighbouring edges), or refuse it.
