# Point Columns

A reconstruction can hold anywhere from a few hundred to a few million 3D
points, and each point comes with numbers that say how well it is supported:
how many photographs see it, how far its observations land from where it
projects, how wide the angle between its viewing rays is, and how well its
patch matches across views. Point columns give each of those numbers a fixed
name and compute it for every point of a reconstruction at once, as one array
per name. A query over the arrays then answers questions such as "which are
the worst points?" or "which points does this image see, and which of them are
badly triangulated?" by returning the matching point indexes in a chosen
order.

The definitions live in one place so that every surface that quotes a number
quotes the same one. The viewer's Track View header and its Image Detail heat
maps call the same functions for one point that the columns call for every
point, and the Python binding returns the arrays the columns hold. The draft
[track-table.md](../../drafts/track-table.md) proposes a Track Table panel and
a `find_points` wire tool that run this query.

## Rust API

The columns and the query live in
[point_columns.rs](../../../crates/sfmtool-core/src/analysis/point_columns.rs),
with the query in
[point_columns/query.rs](../../../crates/sfmtool-core/src/analysis/point_columns/query.rs)
and the per-point triangulation numbers in
[point_columns/track_geometry.rs](../../../crates/sfmtool-core/src/analysis/point_columns/track_geometry.rs),
all re-exported from `sfmtool_core::analysis::point_columns`. The columns are
bound as `sfmtool.analysis.point_columns`.

```rust
/// One named per-point value. The wire name is the stable identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PointColumn {
    TrackLength, ImageCount, RmsErrorPx, AtInfinity, X, Y, Z, CentroidDistance,
    NormalConfidence, LowestZncc, LargestZnccSelfSimilarityRadius,
    MaxRayAngleDeg, ConditionNumber, InverseDepthZ,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColumnCost { Stored, Computed }

impl PointColumn {
    pub const ALL: [PointColumn; 14];
    pub fn wire_name(self) -> &'static str;
    pub fn from_wire_name(name: &str) -> Option<Self>;
    pub fn cost(self) -> ColumnCost;
    /// `x`, `y` and `z`, read through `values_f64`.
    pub fn is_f64(self) -> bool;
    /// Whether `recon` can supply this column at all.
    pub fn is_available(self, recon: &EditedReconstruction) -> bool;
}

/// Column values for every live point of one version, in `live_indexes()`
/// order. Holds only the columns it was asked for.
pub struct PointColumns { /* … */ }

impl PointColumns {
    pub fn compute(
        recon: &EditedReconstruction,
        columns: &[PointColumn],
        progress: &Progress<'_>,
    ) -> Result<PointColumns, Cancelled>;

    pub fn point_indexes(&self) -> &[u32];
    pub fn has(&self, column: PointColumn) -> bool;
    pub fn values(&self, column: PointColumn) -> Option<&[f32]>;
    pub fn values_f64(&self, column: PointColumn) -> Option<&[f64]>;
}

pub struct ColumnBounds {
    pub column: PointColumn,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

pub enum SortOrder { Ascending, Descending }

#[derive(Clone, Debug, Default)]
pub struct PointQueryOptions {
    pub bounds: Vec<ColumnBounds>,
    /// Keep only points with an observation in this image.
    pub seen_in_image: Option<u32>,
    /// `None` keeps `live_indexes()` order.
    pub order_by: Option<(PointColumn, SortOrder)>,
}

pub enum PointQueryError {
    /// A bound or the sort names a column `columns` does not hold.
    ColumnNotComputed(PointColumn),
    ImageOutOfRange { image: u32, image_count: usize },
}

pub fn query_points(
    recon: &EditedReconstruction,
    columns: &PointColumns,
    options: &PointQueryOptions,
) -> Result<Vec<u32>, PointQueryError>;
```

The per-point functions the columns are defined by are public too, because
the viewer calls them for one point at a time:

```rust
pub fn observation_rays(
    image_table: &ImageTable,
    position: &Point3<f64>,
    at_infinity: bool,
    images: impl IntoIterator<Item = usize>,
) -> Vec<[f64; 3]>;
pub fn compute_max_pairwise_angle(rays: &[[f64; 3]]) -> f64;
pub fn compute_max_ray_angle_deg(image_table: &ImageTable, view: &PointView<'_>) -> f32;
pub fn compute_point_diagnostics(image_table: &ImageTable, view: &PointView<'_>) -> (f32, f32);
pub fn compute_position_diagnostics(
    image_table: &ImageTable,
    position: &Point3<f64>,
    error_px: f64,
    images: impl IntoIterator<Item = usize>,
) -> (f32, f32);
```

`compute_position_diagnostics` is the diagnostics of a position that has no
stored point, which Track View reads for a bench track.
`compute_max_pairwise_angle` is also the bench's nearby-track finder's
widest-pair angle, so there is one pairwise loop in the tree.

**Why it is shaped this way:**

- **Compute and query are separate calls.** A caller that shows the columns
  computes them once per version and then runs a new query each time a bound
  changes. A single call taking a query would recompute the maximum ray angle
  for every point each time a bound changed.
- **Columns are computed by name.** A caller asks only for what it shows or
  filters on, so the computed columns cost nothing for a caller that never
  reads one.
- **It works on `EditedReconstruction`.** That is what the viewer holds. A
  caller with an `SfmrReconstruction` wraps it with
  `EditedReconstruction::new(Arc::new(recon))`.
- **The result is a `Vec<u32>` of point indexes, not row positions.** A point
  index is what selecting, reading and deleting a point take, so the result
  passes to any of them without a translation. It is the whole list rather
  than a page; a caller that pages slices it.
- **It takes one `&Progress`.** `compute` can run for over a second on a
  million points, so by the interface contract it takes a `&Progress` and
  stops when it is cancelled. `query_points` is a filter and a sort over
  arrays in memory, and it takes none.
- **Positions are `f64` and read through their own accessor.** A
  georeferenced reconstruction can sit 10⁶ m from the origin, where an `f32`
  resolves only about 6 cm (§ "Determinism and precision"); every other column
  is `f32`. `values` returns `None` for a position column and `values_f64` for
  any other, so a caller cannot read a position at the wrong precision.

Example:

```rust
use sfmtool_core::analysis::point_columns::{
    query_points, ColumnBounds, PointColumn, PointColumns, PointQueryOptions, SortOrder,
};
use sfmtool_core::progress::Progress;

let columns = PointColumns::compute(
    &edited,
    &[PointColumn::TrackLength, PointColumn::RmsErrorPx],
    &Progress::none(),
)?;
// The 20 worst-error points with a track of at least 3, seen in image 2.
let worst = query_points(&edited, &columns, &PointQueryOptions {
    bounds: vec![ColumnBounds { column: PointColumn::TrackLength, min: Some(3.0), max: None }],
    seen_in_image: Some(2),
    order_by: Some((PointColumn::RmsErrorPx, SortOrder::Descending)),
})?;
let page = &worst[..worst.len().min(20)];
```

## The columns

A column belongs to one of two **cost classes**. A **stored** column is read
from data the reconstruction already holds, in O(1) or O(track length) per
point. A **computed** column costs a triangulation or a pass over every pair of
observations, per point. The class decides only where a caller runs the work;
it makes no difference to what a column means.

| Wire name | Class | Definition | Point at infinity | Absent when |
|---|---|---|---|---|
| `track_length` | stored | number of observations in the track | defined | never |
| `image_count` | stored | number of distinct images among those observations; less than `track_length` where one image holds several observations of the point, which a GLOMAP solve produces | defined | never |
| `rms_error_px` | stored | `Point3D::error`, the stored per-point RMS reprojection error in pixels, as its doc comment defines it; the viewer's `get_point` reports the same number. A file holds what its writer stored | defined | never |
| `at_infinity` | stored | 1 for a point at infinity (`w = 0`), 0 for a finite one | defined | never |
| `x`, `y`, `z` | stored | the point's position in the stored frame, in the reconstruction's world-space unit (scene units where none is declared) | no value | never |
| `centroid_distance` | stored | distance from the posed cameras' centroid, the number `ImageTable::placement_scale` gives, in the unit of `x` | no value | the reconstruction has no posed image |
| `normal_confidence` | stored | the stored `normal_confidence` byte, 0 to 255 | defined | the reconstruction has no `normal_confidence` column |
| `lowest_zncc` | stored | the lowest stored per-observation ZNCC in the track, read from `observation_confidence` as `byte / 255` over the non-zero bytes only | no value when no observation of the track was measured | the reconstruction has no `observation_confidence` column |
| `largest_zncc_self_similarity_radius` | stored | the largest ZNCC self-similarity radius among the track's observation readings: the semi-major axis of the stored ellipse, `ellipse_axes[0]`, in patch px, skipping unmeasured rows | no value when every row is unmeasured | the reconstruction has no observation readings |
| `max_ray_angle_deg` | computed | the largest angle between two of the track's viewing rays, the rays from each observing camera's centre to the stored point (`compute_max_ray_angle_deg`) | 0 | never |
| `condition_number` | computed | condition number of the point's triangulation from those rays (`compute_point_diagnostics`) | no value | never |
| `inverse_depth_z` | computed | inverse-depth z-score from the same triangulation, with per-ray angular noise `max(rms_error_px, 1) / f` | no value | never |

A column that is **absent** for a whole reconstruction is reported by
`PointColumn::is_available`, and `compute` skips it rather than failing, so
`has` is false for it. **No value** for a single point is a NaN in its
column.

**The `observation_confidence` byte.** A measured observation stores
`round(255 · clamp(z, 0, 1))` of its ZNCC `z`, so the column's range is 0 to
1 with negative scores at 0, and a `0` byte means the observation was never
measured, not that it scored 0. That is why `lowest_zncc` skips zero bytes
rather than reporting 0 for a track with one unmeasured observation.

**An unmeasured reading** is a row whose semi-major axis is NaN
(`ObservationReading::is_measured` is false), which is what
`ObservationReading::NOT_MEASURED` holds.

**Availability of the readings** is read as
`EditedReconstruction::observation_reading_options().is_some()`, which is true
when the base or an added point carries readings, rather than from the base
alone.

**The ray angle at infinity.** A point at infinity is a bearing: its stored
three numbers are a direction, and every sighting of it casts the same ray, so
its widest pair is 0° whatever the baseline. Subtracting a camera centre from
the direction would measure the rays to a point one unit from the world
origin. A track with fewer than two usable rays has an angle of 0. The
triangulation diagnostics have no depth to be uncertain about at infinity, so
they are NaN there, as they are for a track with fewer than two usable rays.

## The query

`query_points` keeps the rows that pass every bound and, with
`seen_in_image`, have an observation in that image, and returns their point
indexes:

- A row passes a bound when its value is at least `min` and at most `max`,
  both inclusive. A NaN fails every bound that sets either end, so a point
  with no value is never inside a range. A bound that sets neither end filters
  nothing.
- With `order_by`, the rows are sorted by that column, stably, so equal
  values keep ascending point index whichever way the sort runs. Rows with no
  value sort after every row with one, in either direction, which is Track
  View's rule for empty cells. `-0` and `0` compare equal. Without `order_by`
  the rows keep `live_indexes()` order, which is ascending point index.
- A bound or a sort on a column the `PointColumns` do not hold is refused with
  `ColumnNotComputed`, which covers both a column the caller did not ask for
  and one the reconstruction could not supply. A `seen_in_image` past the
  image table is refused with `ImageOutOfRange`.

The `PointColumns` must have been computed from the same version the query is
run against: the query reads the reconstruction only for `seen_in_image` and
the image count, and it does not check that the rows still describe it.

## Implementation notes

`compute` walks the live points in blocks of 8192, each block's points in
parallel under rayon, and polls cancellation between blocks, so a cancel takes
effect within one block. A point's values are written into a fixed array of
one slot per column and scattered into the column vectors after the block,
which keeps each column's rows in `live_indexes()` order without a lock. The
condition number and the inverse-depth z come from one triangulation, which a
point computes once when both are asked for.

The stored columns are filled in one pass and the computed ones in a second,
each in its own share of the progress range (a tenth and nine tenths when
both run, the whole range when one does), each reporting its count of points
in its own phase. The split reflects the cost: a computed column triangulates
or loops over every pair of rays, a stored one reads.

## Determinism and precision

Values are stored as `f32`, except the position columns, which are stored as
`f64`. The columns are shown to three or four significant digits and filtered
against bounds a person types, so `f32` is enough for every derived value. A
position is not: a georeferenced reconstruction can sit 10⁶ m from the
origin, where an `f32` resolves only about 6 cm. Eleven `f32` columns and three
`f64` ones over 10⁶ points take 68 MB, and making every column `f64` would add
another 44 MB for no gain. Every value is computed in `f64` and rounded once
when it is stored.

Each point's values depend only on that point and the image table, so the
output is bit-identical for every thread count. `query_points` sorts stably
and breaks ties by point index, so equal keys always come out in the same
order, and a test can compare whole results exactly.

## Python bindings

`sfmtool.analysis.point_columns(recon, columns=None)` returns a dict that maps
each computed column's wire name to an `(n,)` NumPy array, `float32`, or
`float64` for `x`, `y` and `z`, plus `"point_index"`, the `(n,)` `uint32`
point index of each row. `recon` is an `SfmrReconstruction` (wrapped as an
`EditedReconstruction` for the call) or an `EditedReconstruction`, read
through its edits. With `columns=None` it computes every column the
reconstruction can supply. A list of names computes those; a name that is not
a column, or names a column the reconstruction cannot supply, raises
`ValueError` saying which columns there are, rather than leaving the key out
silently.

```python
import numpy as np
from sfmtool.analysis import point_columns

cols = point_columns(recon, ["track_length", "rms_error_px"])
worst = cols["point_index"][np.argsort(-cols["rms_error_px"], kind="stable")][:20]
```

The query is not bound: a NumPy mask and an `argsort` over the arrays do the
same work, so a Python query function would add nothing. The binding exists so
that scripts and tests read exactly the numbers the viewer shows. It lives in
[analysis/point_columns.rs](../../../crates/sfmtool-py/src/analysis/point_columns.rs).

## Testing

The Rust tests in
[point_columns/tests.rs](../../../crates/sfmtool-core/src/analysis/point_columns/tests.rs)
build small reconstructions from the demo reconstruction and check each column
against the stored data or the per-point function it is defined by: a point
at infinity (no position, distance or diagnostics, a ray angle of 0), an image
holding two observations of one point, absent optional columns, unmeasured
`observation_confidence` bytes and unmeasured readings, and deleted and added
points. They check the query's inclusive bounds, NaN against a bound, a bound with
neither end, `seen_in_image`, ties in both sort orders, no-value rows last in
both orders and both refusals; that a cancelled `compute` returns `Cancelled`;
that progress runs to the end through both passes; and that the values are
bit-identical with one thread and with four. The viewer's own tests of the
per-point functions in `crates/sfm-explorer/src/metrics/tests.rs` run against
these definitions through its re-exports.

The binding's tests in
[test_point_columns_rust_bindings.py](../../../tests/rust_bindings/analysis/test_point_columns_rust_bindings.py)
check the names, dtypes and rows against the seoul_bull ground truth's own
arrays, the column selection and its refusals, and an edited reconstruction.

## Non-goals

The per-observation reprojection error (the largest error in a track, say) is
not a column: for a `sift_files` reconstruction without inline keypoints it
needs every image's `.sift` file read. It can be added as a computed column
that is absent when the reconstruction has no keypoints.

The columns hold no cache. A caller that computes them for one version keeps
them for as long as that version is current.
