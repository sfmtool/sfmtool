# Translation Averaging from Pairwise Baselines

With every frame's rotation known and the direction of the baseline between
each connected pair read from correspondences
([baseline-direction.md](baseline-direction.md)), the camera centres of the
whole graph are one linear problem. This operation solves it: it turns a set
of pairwise unit directions, and optionally a relative length per pair, into
one centre per frame, and it reports what the graph did and did not determine
on the way. A second operation reads the relative lengths off the two-view
depths the pairs imply, and a third reads the one bit the directions cannot:
which of the two mirror-image constellations has the structure in front of the
cameras.

The three are separate calls because they are consumed separately: a caller
that has lengths from elsewhere needs only the first; a caller that wants to
know whether a capture is colinear needs only the reading the first returns
with the length half empty.

## Centres from directions and lengths

### The objective

Over frames `i` and edges `(i, j)` with unit direction `d_ij`, direction
weight `w_ij`, and where stated a relative length `L_ij` with weight `a_ij`:

```
sum_ij  w_ij | P_ij (c_j - c_i) |^2  +  a_ij ( d_ij . (c_j - c_i) - s L_ij )^2
```

`P_ij = I - d_ij d_ij^T` is the projector onto the plane perpendicular to the
direction, so the first term charges the part of the baseline the direction
says should not be there, and the second charges the part along it against
the stated length. The lengths are RELATIVE (they share one unknown scale
`s`), so `s` is an unknown of the fit; it is eliminated in closed form, which
leaves a homogeneous quadratic form `B` in the centres alone.

Two freedoms are invisible to pairwise readings and are gauged, not solved:
the shift (`sum_j c_j = 0`) and the overall scale
(`sum_ij w_ij d_ij . (c_j - c_i) = sum_ij w_ij`, the centres are scaled so the
weighted mean baseline projects to one). The shift gauge is applied by lifting
the three translation directions out of the spectrum of `B` (adding
`trace(B) / 3n` times the projector onto them), so that the null space read
below is never the trivial one.

### The answer is the null space

When every direction agrees with the constellation, the first term is zero at
the true centres: the constellation is what the form sends to zero. The solve
is therefore `argmin x^T B x` under the scale gauge, read from the
eigendecomposition of `B`, and not a linear system posed against `B`. Where
`B` is positive definite the two coincide and the answer is `B^{-1} v` with
`v` the gauge vector; where `B` is singular the linear solve returns the
component of the answer orthogonal to the null space, which is the one part
of the answer the measurement does not carry.

The null space is read at the numerical rank tolerance,
`eps * dim * lambda_max`, and nothing set by the caller.

### When the null space is the answer, and when it is not

A null space of exactly one dimension that no single frame accounts for is the
constellation, and the centres are read off it. Anything else is not:

- **More than one null dimension** is the graph stating that it does not
  determine the constellation. A straight camera path leaves one null
  direction per free spacing along the line, because the perpendicular part
  of every baseline is empty for any spacing along it. The count of extra
  dimensions is reported as `n_free`.
- **A loose frame** is a frame whose own edges do not fix where it sits (one
  edge and no length for it is enough). It moves freely at no cost, so it owns
  a null direction of its own. Ownership is read from the projector onto the
  null space, with no basis chosen: the trace of a frame's own 3x3 block of
  the projector is how many null dimensions are that frame moving, and a frame
  owning more than half of one is loose. Reading centres off such a null space
  would put that frame at no distance in particular.

In either case the answer is the range solution (the eigen-expansion over the
non-null part), which leaves a loose frame where the measurement leaves it,
and the census says which reading was taken (`read_off_null`).

### Reweighting

The solve runs for a fixed number of rounds. After each, every edge's
direction residual (the perpendicular part of its baseline at the current
centres) and, where a length is stated, its length slip (how far its projected
baseline sits from `s L_ij` at the fitted `s`) reweight that edge's two halves
as `w / (1 + r / floor)`. The floor of each half is the median residual of
that half over the graph, so an edge stops carrying the solve when it is worse
than the graph's own typical edge, and nothing is set in pixels or units. The
base weights are the ones the caller passed; reweighting never raises a weight
above them.

### What comes back

Per frame the centre, in the caller's rotation frame, mean-centred and at the
gauge scale. Per edge the projected baseline length `lambda_ij =
d_ij . (c_j - c_i)` and the direction residual. A census: `lambda_max`, the
two smallest eigenvalues relative to it and their ratio (the conditioning of
the constellation), `n_null`, `n_loose`, `n_free`, `read_off_null`,
`n_lengths` (edges that stated a length) and `solved`. The solve returns
nothing when the gauge vector has no component along the answer (the graph
states no baseline at all): `solved` is false, the census still carries what
the form read, and the per-frame and per-edge arrays are empty.

A negative `lambda_ij` is an edge whose baseline the constellation placed
backwards; the count and fraction are what a caller reads to judge the graph,
and the operation does not act on them.

### The direction reading

The same form at the caller's weights with the length half empty, decomposed
once and not reweighted, returns the census alone. It states what the graph's
geometry determines on its own, which is a property of the capture rather
than of a solve, and it is what tells a colinear path (`n_free > 0`) from a
general one.

## Relative lengths from two-view depths

A pair's direction fixes the pair's geometry up to how long its baseline is,
so the two-view depths that come out of the pair (each ray's depth at the
closest approach of the two rays, with `c_i` at the origin and `c_j` at
`+d_ij`) are in units of THAT baseline. A point two edges both see from the
same frame therefore has two depths for one world distance, and their ratio
is the ratio of the two baselines.

The whole graph is one fit of

```
log z(edge, frame, point) = D(frame, point) - x(edge)
```

with `x` the log baseline length and `D` the log world depth. The `D` are
eliminated rather than solved for: each is the weighted mean of its own rows,
so the operator in the edges alone is a pass over the rows (spread each edge's
value onto its rows, subtract the per-group mean, weight, gather back) and
never a matrix. The system is solved by preconditioned conjugate gradient with
the operator's diagonal as the preconditioner; the constant vector is its null
space, which is the one freedom a set of relative lengths does not have, and
the right-hand side is orthogonal to it so the iteration never leaves the
complement. The result is gauged to a median log length of zero.

A row is TIED when another edge saw the same point from the same frame; only
tied rows relate one baseline to another. An edge needs a minimum count of
tied rows (three, the baseline-direction operation's own floor: below three a
reading is the rows themselves rather than a fit of them) before it states a
length, and an edge without one comes back as NaN so that the centre solve
constrains its direction only.

Rows are reweighted by their own absolute residual against the graph's median
residual for a fixed number of rounds, so a wild depth stops carrying the fit.
Per edge the operation returns the length, the median absolute log residual
of that edge's own rows (its scatter) and its tied-row count.

Only rows with positive depth on both rays enter; the caller supplies the rows
already filtered.

## Orientation from cheirality

Pairwise directions determine the constellation only up to the point
reflection `c -> -c`: negating every centre negates every triangulated depth
and changes nothing else, because the form above is quadratic in the centres
and does not contain them. Structure in front of the cameras is the one
physical statement that separates the two.

For every point with two or more observations, the point is solved at the
current centres by the batch triangulation's least-squares midpoint (the
closest approach of its rays, and the minimum-norm point in the observable
subspace where those rays are exactly parallel, so no point drops out of the
vote for being degenerate), and each observation's depth along its own ray is
read. The reading is the parallax-weighted vote

```
angw = sum_points  theta_widest(point) * ( n_front - n_behind )
```

with `theta_widest` the point's widest ray-pair angle. A point's cheirality
statement is worth exactly the parallax it was measured with: a point inside
the caller's angular bound is a bearing whose depth sign is a coin toss, and
it contributes nothing beyond its own small angle. `angw < 0` says the
constellation should be reflected.

The reading is exactly antisymmetric under `c -> -c`, so one pass describes
both orientations. Beside `angw` the operation returns the unweighted vote
(`obs_front`, `obs_total`, the front fraction and its margin from one half)
and a census of the points it saw: inside the bound, behind at least one
camera, or in front of all.

## Determinism

Edges are processed in the order the caller gives them, and every reduction
over edges or rows is sequential in that order. The eigendecomposition is of
a symmetric matrix; the sign of an eigenvector is fixed by the gauge
projection, so the centres do not depend on the decomposition's sign
convention. Nothing samples and nothing reads a clock. The same inputs give
the same bytes.

## Binding

The operations live in
[translation_averaging.rs](../../../crates/sfmtool-core/src/geometry/translation_averaging.rs)
and are bound in the geometry module of the bindings, under
`sfmtool._sfmtool.geometry`, as four functions taking NumPy arrays and
returning NumPy arrays and a census dict:

- `average_translations(edges, directions, weights, lengths=None,
  length_weights=None, *, n_frames, rounds=...)`: `edges` an `(m, 2)` frame
  index array, `directions` `(m, 3)` unit rows, `weights` `(m,)`, `lengths`
  `(m,)` with NaN for an edge that states none and `length_weights` `(m,)`.
  `n_frames` is keyword-only and has no default, because nothing in the edges
  says how many frames the graph addresses. Returns `(centres (n, 3),
  lambda (m,), residual (m,), census)`, with all three arrays empty and the
  census's `solved` false when the solve returns nothing.
- `direction_reading(edges, directions, weights, n_frames)`: the census
  alone.
- `relative_lengths(edge_of_row, frame_of_row, point_of_row, depth_of_row,
  n_edges, rounds=..., min_tied=...)`: one row per `(edge, frame, point)`
  depth. Returns `(lengths (m,), scatter (m,), n_tied (m,))`.
- `orientation_reading(centres, rays_world, point_of_ray, frame_of_ray,
  angular_bound)`: the vote and its census.

Arrays are accepted in either memory order and returned C-contiguous. The
binding refuses a direction row that is not unit, an edge index outside
`n_frames`, an edge joining a frame to itself (its block would land twice on
one diagonal and once on nothing), and a non-positive depth.

## Testing

- **Null-space parity.** On a graph whose directions are exactly consistent
  with a general (non-colinear) constellation, the centres equal the true ones
  up to shift and scale to round-off, and `read_off_null` is true.
- **Colinear path.** Frames on a line with no lengths: `n_free` is one and the
  centres come from the range solution; with lengths stated the spacing is
  recovered and `n_free` is zero.
- **Loose frame.** A frame attached by one edge with no length is reported
  loose and the remaining constellation is unchanged by its presence.
- **Reflection.** Negating every centre negates `angw` exactly.
- **Reweighting.** One edge with a direction perpendicular to the truth comes
  back carrying the graph's largest direction residual, and the rest of the
  constellation is still recovered. The weights themselves are internal, so
  what the test reads is the outcome they produce.
- **Relative lengths.** Two edges sharing tied rows at a known depth ratio
  recover that ratio; an edge with fewer tied rows than the floor returns NaN;
  a graph with no tied rows returns all NaN.
- **Determinism.** Two runs on the same arrays are byte-identical.
