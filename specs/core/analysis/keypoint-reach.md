# Keypoint Reach Pairs

**Status:** Implemented in
`crates/sfmtool-core/src/spatial/keypoint_reach.rs`, bound as
`sfmtool._sfmtool.analysis.keypoint_pairs_within_reach`.

One question, asked per image of a track set: which other keypoints lie inside
this keypoint's own disk? Several rules read that neighbourhood and differ only
in what they then test, so the enumeration is stated once and the tests stay
with the callers.

The domain is the image plane. A KEYPOINT here is a row of a track set: an
image index, a pixel position, and its own query radius (its REACH) in pixels.
This is the pixel-domain counterpart of the world-space queries on
`spatial::PointCloud`: that index answers proximity between 3D points in world
units at a shared radius; this operation answers proximity between 2D keypoints
in pixels, per image, each keypoint carrying its own radius. The
[observation adjacency graph](observation-adjacency-graph.md) sits above both,
deciding adjacency of points from the behaviour of their keypoints across
images.

## The enumeration

Inputs, one row per keypoint over the whole track set: the image index, the
pixel position, and the reach. Output, the candidate pairs `(i, j, d)`: every
row `j` in the same image whose centre lies within row `i`'s reach, with `d`
the pair's separation in pixels.

- The relation is per image; rows of different images are never paired.
- The reach is row `i`'s own: the relation is directed, and `(i, j)` says
  nothing about `(j, i)`. A caller wanting a symmetric relation reads both
  directions, which the enumeration already emits.
- A row is never its own candidate. The self pair answers no rule's
  question, so the enumeration leaves it out rather than hand every caller a
  pair to discard.
- A row whose reach is not finite asks nothing, and still appears as a
  candidate of other rows. Not finite means what the word says: NaN and a
  positive infinity both ask nothing. A negative reach names no disk at all
  and is refused rather than read as an empty one.
- One distance is reported per pair, so a caller testing against `reach[i]`,
  against a bound of its own, or against a function of both radii reads the
  same `d`.

## Mechanism

Within an image the rows are sorted by column. A disk of radius `reach` cannot
contain a centre whose column is further than `reach` away, so a row's
candidates are one contiguous run of the sorted order, found by binary search
at `x - reach` and `x + reach`; the run is then filtered by true Euclidean
distance. Cost is the sort plus output size: no tree, no grid, and no
quadratic pass over an image unless the answer itself is quadratic. The run is
a superset the distance test trims, so a caller's own containment test against
the same reach restates the answer rather than narrowing it.

The column order is a total one, with a column that is not a number sorting
last: such a row is found by no search and asks within no run.

The pair stream is produced in a defined order: rows in their given order,
each row's candidates in the sorted-column order of its run. Batching is a
work grain and not a threshold: the pairs are the same pairs in the same order
at any batch size, and the output is sized to the pairs rather than to the
runs. Both images and the batches of rows within one image are independent and
may be enumerated in parallel -- the intended calling pattern is one image at a
time, and parallelism that stopped at the image boundary would leave such a call
with none at all. The parts are concatenated in the order the sequential
enumeration would have produced them, so the result does not depend on the
scheduling.

## What consumes it

Nothing in the pipeline, today. The enumeration's only callers are its own
tests (`spatial/keypoint_reach/tests.rs`) and the Python binding below, through
`tests/rust_bindings/test_keypoint_reach_rust_bindings.py`. It was extracted
from two rules that ask this question, and both still expand the neighbourhood
for themselves in NumPy; see [Open questions](#open-questions).

## Binding

`keypoint_pairs_within_reach(image_of_row, xy_px, reach_px)` in the analysis
module of the bindings: `image_of_row` an `(n,)` `int64` array, `xy_px` an
`(n, 2)` `float64` array, `reach_px` an `(n,)` `float64` array. Returns
`(i, j, d_px)` as three arrays in the defined order, the row indices `int64`
and the distances `float64`. Arrays are accepted in either memory order and
returned C-contiguous. The binding refuses mismatched lengths, an `xy_px` that
is not `(n, 2)`, and a negative reach, which it names by row; a NaN reach is
the documented "asks nothing" value, not an error.

## Testing

- **Exactness.** Against a brute-force double loop on random rows: same pair
  set, same order.
- **Directedness.** A large-reach row paired with a small-reach row appears as
  `(large, small)` and not `(small, large)` when only the large reach spans
  the separation.
- **Self pair.** Never present; a NaN-reach row appears only on the
  candidate side.
- **Image isolation.** Identical pixel positions in different images produce
  no pair.
- **Order and batching.** The pair stream is identical at any batch size, and
  identical with and without parallelism.
- **Refusals.** Mismatched lengths and a negative reach are refused, and the
  refusal names the offending row.

## Open questions

**Whether the two rules this was extracted from should migrate onto it.** Both
ask the same question of the same rows and differ only in what they then test,
which is why the enumeration was stated once:

- *Retiring coarse observations a finer feature covers* would set the reach to
  the row's drawn footprint, keep the pairs at least one radius band apart, and
  retire the coarse side.
- *Reconciling points that rest on one measurement* would set the reach to a
  fraction of the row's refined unit scale, keep the pairs whose radii agree,
  and join their points into one tangle.

Each verdict would be an exact function of the pair set, so the enumeration
would carry its determinism. What a migration has to establish, and what nothing
asserts today, is that each rule's mask comes out byte for byte identical to
what its NumPy expansion produces now — including on the members the
reconciliation's tolerance was drawn from. Until that parity is measured neither
rule is a consumer of this module.
