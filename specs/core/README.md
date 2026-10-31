# Core Algorithm Specifications

Design documents for the algorithms in `crates/sfmtool-core/`, one subdirectory
per module of `crates/sfmtool-core/src/`. Each spec's status line names the file
that implements it.

A few specs here describe Python pipelines in `src/sfmtool/` that orchestrate
these kernels rather than an `sfmtool-core` module of their own; they are filed
under the module they drive. One,
[cluster selection](features/cluster-selection.md), describes an operation of
the `sfmtool-matches-format` crate and is filed under the module whose output
it narrows.

The Python bindings to these modules are indexed by submodule in
[../python-bindings.md](../python-bindings.md), which names the spec for each.

## Median and quantile

Every median and quantile in `sfmtool-core` comes from one module,
[`sfmtool_core::numeric`](../../crates/sfmtool-core/src/numeric.rs), and
`sfm-explorer` calls the same functions rather than its own. `median` and
`median_in_place` give the median, and `quantile`, `quantile_in_place` and
`quantile_of_sorted` give the `q`-quantile for `q` in `[0, 1]`. The
`_in_place` forms reorder the caller's slice, `quantile_of_sorted` expects a
slice already sorted ascending, and the other two copy. All of them follow the
same rules:

- **Ordering is `f64::total_cmp`.** NaN sorts above every finite value (and
  `−NaN` below), so no comparison panics and a minority of NaNs leaves the
  answer finite. A NaN result means a NaN reached that rank of the population.
- **Empty input gives NaN**, never a number, so the caller's "no answer"
  path handles it.
- **The median of an even count is the mean of the two middle values**, as in
  `numpy.median`.
- **The quantile interpolates linearly between order statistics**, numpy's
  default `linear` method: the position is `q·(n − 1)`, and the interpolation
  uses numpy's own arithmetic, so whenever the gap between the two order
  statistics is finite the result equals `numpy.quantile` bit for bit. When one
  of the two is infinite the result is that infinity for any non-zero
  fraction, and only `−∞` against `+∞`, or a NaN input, gives NaN.

`numeric/tests.rs` fails on any other function named for a median, quantile or
percentile in the workspace, outside a short allowlist of different
operations. The module also holds `splitmix64`, the one RNG step behind the
crate's deterministic sampling.

## Directories

| Directory | Module | Contents |
|-----------|--------|----------|
| [analysis/](analysis/) | `analysis/` | Graphs and measurements over a reconstruction: covisibility, adjacency, coverage, alignment between reconstructions. |
| [bench/](bench/) | `bench/` | The bench beside a reconstruction and the editable track on it: the item list and its labels, the observations with their measurements and verdicts, and the commit that writes one back as a point. |
| [camera/](camera/) | `camera/` | Camera models, distortion kernels, projection derivatives, image warping, and the photograph cache. |
| [features/](features/) | `features/` | Feature extraction and matching: SIFT, descriptor search, optical flow, cluster matching. |
| [geometry/](geometry/) | `geometry/` | Pose estimation, epipolar geometry, and bundle adjustment. |
| [patch/](patch/) | `patch/` | Everything about oriented patches: normals, keypoint localization, and cluster patches. |
| [reconstruction/](reconstruction/) | `reconstruction/` | Operations on reconstruction data itself: triangulation, point correspondence. |
| [spatial/](spatial/) | `spatial/` | The shared KD-tree index behind every proximity query: nearest, k-nearest and radius, in world units. |
| [spherical/](spherical/) | `spherical/` | Spherical tiling and panorama compositing. |
