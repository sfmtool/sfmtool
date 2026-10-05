# Baseline Direction from Ray Coplanarity

Global structure from motion solves every camera's rotation first. It then
needs, for each pair of images that see the same points, the direction from
one camera centre to the other, so that translation averaging
([translation-averaging.md](translation-averaging.md)) can place the centres.
This operation computes that unit direction for every edge of an image-pair
graph in one call, from the two images' rays to their shared points alone: it
needs no depths and solves for no translation. Beside each direction it reports
how well the edge's own rows determine it. No command in `sfm` calls it today;
its callers are its tests and the Python binding.

With both rotations known, the baseline `b = c_j - c_i` is coplanar with every
point's two world rays:

```
b . (u_i x u_j) = 0
```

so `b` is the null space of the matrix whose rows are those normals, one row
per shared point. Every edge of a graph is one such solve.

## Rust API

The solve lives in
[baseline_direction.rs](../../../crates/sfmtool-core/src/geometry/baseline_direction.rs),
bound as `sfmtool._sfmtool.geometry.baseline_directions`.

```rust
pub struct BaselineTrim {
    pub tol_rad: f64,        // parallax at or below which a row is dropped
    pub rounds: usize,       // refit rounds
    pub keep_fraction: f64,  // fraction of the retained rows each round keeps
}

pub struct BaselineDirection {
    pub direction: [f64; 3],
    pub n_rows: usize,
    pub n_used: usize,
    pub condition: f64,
    pub parallax_median_deg: f64,
    pub parallax_max_deg: f64,
    pub cheiral_fraction: f64,
    pub residual_median_rad: f64,
}

pub fn baseline_directions(
    rays_i: &[f64],
    rays_j: &[f64],
    offsets: &[usize],
    trim: BaselineTrim,
) -> Vec<Option<BaselineDirection>>;
```

`rays_i` and `rays_j` hold the two frames' unit world rays, three components
per row and one row per shared point, with every edge's rows concatenated.
`offsets` has length `n_edge + 1` and gives edge `e` the rows
`offsets[e]..offsets[e + 1]`. The graph is passed in this flattened form so
that a whole covisibility graph is one call whose edges are solved in
parallel, and so that the same arrays cross the Python boundary without a
per-edge list. The function panics when the two ray arrays differ in length,
when their length is not a multiple of three, when `offsets` decreases, or when
its last entry exceeds the row count.

The result has one entry per edge, in input order. An edge with fewer than
three rows past the parallax bound comes back `None`.

```rust
use sfmtool_core::geometry::baseline_direction::{baseline_directions, BaselineTrim};

let trim = BaselineTrim { tol_rad: 0.05_f64.to_radians(), rounds: 5, keep_fraction: 0.6 };
let out = baseline_directions(&rays_i, &rays_j, &[0, n_ab, n_ab + n_bc], trim);
if let Some(edge) = out[0] {
    let d = edge.direction; // unit vector from c_a towards c_b
}
```

### Parameters

`BaselineTrim` has no defaults: the caller supplies all three fields, and the
Python binding takes all three as required arguments. The tests use
`tol_rad = 0.05°`, `rounds = 5` and `keep_fraction = 0.6`.

| Field | Meaning |
|-------|---------|
| `tol_rad` | Parallax angle, in radians, at or below which a row is dropped before the fit. |
| `rounds` | Number of trim-and-refit rounds after the first fit on all retained rows; 0 keeps that first fit. |
| `keep_fraction` | Fraction of the retained rows each refit round keeps, never fewer than three. |

### What comes back

Per edge:

- `direction`: the unit direction from the first centre to the second.
- `n_rows`: how many rows the edge was given. `n_used`: how many rows had
  parallax above `tol_rad`, which are the rows the fit reads.
- `condition`: the second-smallest singular value over the smallest, on the
  rows the final fit kept. It is infinite where the smallest singular value is
  exactly zero, which is what a noiseless edge produces, so a caller that reads
  it as a trust weight should cap it rather than use it raw.
- `parallax_median_deg`: the median parallax of the used rows, in degrees.
  `parallax_max_deg`: the widest parallax over every row, the dropped ones
  included, so a caller can tell an edge with no parallax from one whose
  parallax is concentrated in a few rows.
- `cheiral_fraction`: see [The sign is cheirality](#the-sign-is-cheirality).
- `residual_median_rad`: the median of `|n̂ . d|` over the used rows, where `n̂`
  is a row's unit normal and `d` the direction. That is the sine of the angle
  between `d` and the row's ray plane; the field name says radians because the
  two agree to first order at the small residuals of a well-determined edge.

### Python binding

`baseline_directions(rays_i, rays_j, offsets, tol_rad, rounds, keep_fraction)`
takes `(n_row, 3)` float64 ray arrays and `(n_edge + 1,)` int64 offsets, and
returns a dict of per-edge arrays with the fields above plus `stated`, a bool
that is False for an edge Rust returned `None` for. On such an edge `n_rows` is
still the edge's row count, `n_used` is 0 and the float fields are NaN. Where
the Rust function panics on malformed input, the binding raises `ValueError`.

## Rows are selected by parallax

The normal `u_i x u_j` has norm `sin(parallax angle)`, so its length says how
much the point's two rays differ. When the parallax is at or below `tol_rad`,
the two rays are, to within noise, the same ray, and their cross product is
noise around the zero vector. Those rows are dropped, not down-weighted,
because a down-weighted noise row still contributes to the fit.

The retained rows are normalized to unit length before the fit, so a single
wide pair cannot dominate the solve because of its large parallax; parallax
decides which rows are in, and nothing more.

An edge with fewer than three retained rows states no direction. Three is the
smallest count from which the null space of a three-column matrix is a fit
rather than the rows themselves; it is not a quality threshold.

## The fit is refit on its own best rows

The null vector is the right singular vector of the smallest singular value of
the retained rows. It is then refit `rounds` times, each round keeping the
`keep_fraction` of rows whose coplanarity residual against the current
direction is smallest, never fewer than three. The kept count is taken over the
retained rows, not over the previous round's survivors, so the kept set does
not shrink from round to round.

The residual that ranks rows is computed over every retained row, including the
ones the previous round dropped, so a row the fit moved away from can come
back.

## The sign is cheirality

The null space is a line, and the two directions along it describe two
constellations that are mirror images of each other. With one centre at the
origin and the other at `+d`, each retained row's point is placed at the
closest approach of its two rays, and its depth along each ray is read. The
same is done at `-d`. The sign kept is the one that puts more rows in front of
both cameras; a tie keeps the sign the decomposition produced.

`cheiral_fraction` is the share of the rows that are in front of both cameras
at one sign or the other that are so at the sign kept. Rows in front at neither
sign are not counted, and the fraction is 0 when no row is in front at either.
A value near 1 says the rows agree on the sign; a value near 0.5 says the sign
is close to arbitrary.

## Determinism

Edges are independent and are solved in parallel; the output is in input order
and each edge's arithmetic runs sequentially over its own rows. The kept count
`keep_fraction * n_used` is rounded half to even, the rule Python's `round`
applies. Ties in the residual ranking are broken by row order. Two runs on the
same arrays produce the same bits.

## What consumes it

The directions are exact to the precision of the decomposition, which matters
for what consumes them. A translation averaging over a graph of such edges is
ill-posed when every edge carries the same direction, which is what a camera
moving along a straight line produces: the averaging minimizes the part of
each baseline perpendicular to its measured direction; with all directions
equal, that part is zero for every arrangement of the cameras along the line,
so their spacing is unconstrained and only the total scale is fixed by the
gauge. Given exactly colinear directions, such an averaging returns arbitrary
positions, and an implementation whose directions carry rounding asymmetries
can appear to resolve the spacing from those asymmetries rather than from any
measurement. A consumer therefore has to handle colinear graphs itself:
recognize a graph whose directions leave the spacing free, and either resolve
it from something the directions do not carry (the pairs' own depths give the
relative scale between neighbouring edges) or refuse it.

## Testing

The Rust tests are in
[baseline_direction/tests.rs](../../../crates/sfmtool-core/src/geometry/baseline_direction/tests.rs)
and the binding tests in
[test_baseline_direction_rust_bindings.py](../../../tests/rust_bindings/geometry/test_baseline_direction_rust_bindings.py).
They check that the direction matches synthetic centres, that the sign follows
cheirality, that rows inside the bound are dropped, that an edge without
parallax states nothing, that a whole graph is one call, that the output is
bit-for-bit repeatable, and that malformed offsets are refused.
