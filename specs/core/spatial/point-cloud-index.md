# The Point Cloud Index

Most passes over a reconstruction have to ask where things are near each other:
which 3D points sit inside a radius of this one, which of a tiling's directions
is closest to a ray, how far apart neighbouring points typically are. Answering
that by scanning every point is quadratic and the clouds run to millions of
points, so those questions go through one shared spatial index — a KD-tree
built once over a fixed set of points and then only queried. The index is a
thin wrapper around a third-party KD-tree crate, and the wrapping is the point:
callers state their question in flat coordinate arrays and get flat index
arrays back, and no part of the codebase outside the wrapper names the KD-tree
library or its types.

## Rust API

The index lives in
[spatial.rs](../../../crates/sfmtool-core/src/spatial.rs), and is exposed to
Python as `sfmtool._sfmtool.spatial.KdTree2d` / `KdTree3d` (see
[Python bindings](#python-bindings)).

```rust
/// The scalar coordinate types a cloud can be built over: `f32` and `f64`.
pub trait Scalar: /* sealed */ {}

pub struct PointCloud<A: Scalar, const DIM: usize> { /* … */ }

pub type PointCloud2<A> = PointCloud<A, 2>;
pub type PointCloud3<A> = PointCloud<A, 3>;

impl<A: Scalar, const DIM: usize> PointCloud<A, DIM> {
    pub fn new(positions: &[A], n_points: usize) -> Self;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn position(&self, i: usize) -> [A; DIM];

    pub fn nearest(&self, query: &[A], n_queries: usize) -> Vec<u32>;
    pub fn nearest_k(&self, query: &[A], n_queries: usize, k: usize) -> Vec<u32>;
    pub fn within_radius(
        &self, query: &[A], n_queries: usize, radius: A,
    ) -> (Vec<u32>, Vec<u32>);
    pub fn nearest_k_within_radius(
        &self, query: &[A], n_queries: usize, k: usize, radius: A,
    ) -> Vec<u32>;
    pub fn self_nearest_k(&self, k: usize) -> Vec<u32>;
    pub fn nearest_neighbor_distances(&self) -> Vec<A>;
}
```

Everything crossing this boundary is a **flat row-major array**, never a slice
of arrays or a `Vec<Vec<_>>`. Points arrive as `[x, y, x, y, …]` with a
separate count, k-nearest results come back as one `n_queries * k` block, and
radius results come back in CSR form — an `offsets` array of length
`n_queries + 1` beside a single `indices` array. That shape is what the Python
bindings hand to numpy without copying and reshaping, and what the callers
inside the crate already hold: positions live in `Vec<f64>` columns of a
reconstruction, not in `Vec<[f64; 3]>`.

Results are **indices into the cloud**, never positions. A caller almost always
wants to reach its own parallel arrays with the answer, so returning positions
would mean it had to search for them again. `u32::MAX` is the "no such
neighbour" sentinel that pads a short k-nearest row, which is why the index type
is `u32` rather than `usize`: the sentinel has to be a value no real index can
take, and one that survives the trip into a numpy `uint32` array unchanged.

`Scalar` is sealed and implemented for `f32` and `f64`. It exists so the
KD-tree's own trait bounds — an axis type, a distance accumulator, a widening
cast — stay inside this module instead of appearing in the signature of every
function that happens to take a cloud. Callers name `PointCloud3<f32>` and
never name `Scalar` at all.

```rust
use sfmtool_core::spatial::PointCloud3;

// Two points, 3 units apart, as one flat array.
let positions: [f32; 6] = [0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
let cloud = PointCloud3::<f32>::new(&positions, 2);

// One query, its two nearest neighbours, nearest first.
assert_eq!(cloud.nearest_k(&[0.5, 0.0, 0.0], 1, 2), vec![0, 1]);

// Everything within 1 unit of the origin, in CSR form.
let (offsets, indices) = cloud.within_radius(&[0.0, 0.0, 0.0], 1, 1.0);
assert_eq!((offsets, indices), (vec![0, 1], vec![0]));
```

### What the queries guarantee

- **Radii are Euclidean and inclusive.** Callers pass a plain distance, the
  cloud squares it for the metric, and a point at exactly `radius` is a hit.
  The distances the cloud returns, from `nearest_neighbor_distances`, are
  Euclidean too. Neither direction ever exposes a squared distance.
- **k-nearest results are ordered nearest-first** and padded with `u32::MAX`
  when fewer than `k` neighbours qualify; the padding is always a suffix, so a
  caller may stop at the first sentinel.
- **`within_radius` is unordered.** Its CSR shape already says so: the members
  of one query's span are whatever the traversal found, in traversal order.
- **Ties are broken arbitrarily.** Two points at exactly the same distance from
  a query may come back in either order, and which one a single-result query
  returns is unspecified. Nothing here promises the lowest index wins, and no
  caller may depend on it.
- **Duplicate coordinates are ordinary input.** Any number of points may share
  a coordinate value, or every coordinate; all of them are indexed and all of
  them are reachable.
- **`self_nearest_k` and `nearest_neighbor_distances` exclude a point's own
  index, and nothing else.** A point with a coincident twin reports that twin
  at distance zero rather than skipping to the nearest distinct location.

## Theory

A KD-tree splits the point set by one coordinate at a time, so a query
descends to the leaf its own coordinates put it in and then backtracks only
into the sibling subtrees whose splitting plane is closer than the best
candidate found so far. Query cost is therefore output-sensitive rather than
proportional to the cloud, which is the whole reason the index exists.

Two properties of the input decide which form of tree can be used.

**Duplicate axis values are the normal case here, not a pathology.** SIFT
keypoints line up along a strong image edge and share an x to the last bit; a
rotation-only reconstruction stores every camera at the origin; a spherical
tiling is a lattice; ground points share a z. A tree whose leaves are
fixed-capacity buckets cannot represent more than a bucket's worth of points
that are identical on the splitting axis, because no split separates them. The
bulk-built tree splits on medians over the whole set instead, which places an
arbitrary number of equal values on either side of the plane, and so has no
such limit.

**The point set is complete before the first query.** Every caller here builds
from a finished array — a reconstruction's points, an image's keypoints, a
tiling's directions — and then only reads. That is exactly the case the
bulk-built, contiguous-layout tree is for, and it is faster to build and to
query than an incrementally grown one.

Those two together settle the choice: the index is always the immutable,
bulk-built tree, never the mutable one, even at call sites that assemble their
input point by point.

## Implementation notes

The cloud keeps its own copy of `positions` alongside the tree. That is
deliberate duplication: `position(i)` and the self-query methods need to read a
point back by index, the tree's internal ordering is not the caller's, and
paying one contiguous copy is cheaper than either querying the tree for its own
points or forcing every caller to keep its array alive for the cloud's
lifetime.

`nearest_k_within_radius` is the one parallel method, over rayon. It is the
only one whose per-query work is large enough and whose result rows are a
fixed width, so the output can be written into a pre-sized buffer with no
coordination — `par_chunks_mut(k)` hands each query its own row. The others
either append to a growing vector (`within_radius`, whose row lengths are not
known in advance) or are already called from inside a parallel region.

`self_nearest_k` and `nearest_neighbor_distances` ask for `k + 1` and `2`
neighbours respectively and then drop the entry whose index is the query
point's own. They cannot simply drop the first result: with coincident points
the query point need not be the first of the zero-distance group returned.

The tree's item type is `u32`, matching the index type at the boundary, so a
cloud of more than `u32::MAX` points cannot be built — `new` panics rather than
silently truncating. No caller is near that bound; the largest real clouds are
a few tens of millions of points.

## Python bindings

`KdTree2d` and `KdTree3d`, in
[spatial/kdtree.rs](../../../crates/sfmtool-py/src/spatial/kdtree.rs) and
imported from `sfmtool._sfmtool.spatial`. Each is constructed from an
`(N, DIM)` numpy array and dispatches on its dtype, holding an `f32` or an
`f64` cloud accordingly; queries take an `(M, DIM)` array of the same dtype.
`nearest` returns `(M,) uint32`, `nearest_k` and `nearest_k_within_radius`
return `(M, K) uint32` padded with `2**32 - 1`, `within_radius` returns the CSR
pair as two `uint32` arrays, `self_nearest_k` returns `(N, K) uint32`, and
`nearest_neighbor_distances` returns `(N,)` in the cloud's own dtype. `len()`
and `dtype()` report what was built.

```python
import numpy as np
from sfmtool._sfmtool.spatial import KdTree3d

tree = KdTree3d(np.asarray(points, dtype=np.float32))
offsets, indices = tree.within_radius(queries, radius=0.05)
```

## Testing

`spatial/tests.rs` covers each method in 2D and 3D, `f32` and `f64`, and pins
every guarantee above as its own test: the radius is Euclidean and inclusive at
the boundary, k-nearest rows are sorted with the padding as a suffix, `k == 0`
is a no-op rather than a panic, an empty cloud answers every query, a query
coincident with a data point finds it at distance zero, and coincident points
report zero to each other. Three construction cases stand for the
duplicate-axis argument in the theory above: 40 points sharing an x, 200 points
at one position, and a full 3D lattice — each asserting not just that
construction succeeds but that every point is reachable by a query.

[Keypoint reach pairs](../analysis/keypoint-reach.md), the pixel-domain
counterpart to these world-space queries, lives beside this index in
`spatial/keypoint_reach.rs` and has its own tests there.

## Non-goals

Approximate nearest-neighbour search: every query here is exact. The
approximate, high-dimensional case is descriptor matching, which has its own
structure in [the randomized KD-tree
forest](../features/randomized-kdtree-forest.md).

Metrics other than Euclidean, and dimensions other than 2 and 3. Nothing in the
wrapper is specific to those, but nothing needs more, and each extra
instantiation is monomorphized code.

Mutation. A cloud is built from a complete point set; a changed point set means
a new cloud.
