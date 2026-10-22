# Track Table

A reconstruction can hold anywhere from a few hundred to a few million 3D
points, and each point comes with numbers that say how well it is supported:
how many photographs see it, how far its observations land from where it
projects, how wide the angle between its viewing rays is, and how well its
patch matches across views. The track table gives each of those numbers a
fixed name and computes it for every point of a reconstruction at once, as one
array per name. Its rows are tracks, keyed by point index, and its columns are
those named values. The table is computed on demand from one version of a
reconstruction and never stored: no file format holds it. A query over the
table then answers questions such as "which are the worst points?" or "which
points does this image see, and which of them are badly triangulated?" by
returning the matching point indexes in a chosen order.

The definitions live in one place so that every surface that quotes a number
quotes the same one. The viewer's Track View header and its Image Detail heat
maps call the same functions for one point that the table calls for every
point, and the Python binding returns the arrays the table holds. The viewer's
Track Table panel ([gui/track-table.md](../../gui/track-table.md)) shows the
table, and its `find_points` wire tool runs this query over the same cached
table ([gui/mcp-server.md](../../gui/mcp-server.md) § "`find_points`").

## Rust API

The table and the query live in
[track_table.rs](../../../crates/sfmtool-core/src/analysis/track_table.rs),
with the query in
[track_table/query.rs](../../../crates/sfmtool-core/src/analysis/track_table/query.rs)
and the per-point triangulation numbers in
[track_table/track_geometry.rs](../../../crates/sfmtool-core/src/analysis/track_table/track_geometry.rs),
all re-exported from `sfmtool_core::analysis::track_table`. The table is
bound as `sfmtool.analysis.track_table`.

```rust
/// One named per-point value. The wire name is the stable identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrackTableColumn {
    TrackLength, ImageCount, RmsErrorPx, AtInfinity, X, Y, Z, CentroidDistance,
    NormalConfidence, LowestZncc, LargestZnccSelfSimilarityRadius,
    MaxRayAngleDeg, ConditionNumber, InverseDepthZ,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColumnCost { Stored, Computed }

impl TrackTableColumn {
    pub const ALL: [TrackTableColumn; 14];
    pub fn wire_name(self) -> &'static str;
    pub fn from_wire_name(name: &str) -> Option<Self>;
    pub fn cost(self) -> ColumnCost;
    /// `x`, `y` and `z`, read through `values_f64`.
    pub fn is_f64(self) -> bool;
    /// Whether `recon` can supply this column at all.
    pub fn is_available(self, recon: &EditedReconstruction) -> bool;
}

/// Column values for every live point of one version, one row per track keyed
/// by point index, in `live_indexes()` order. Holds only the columns it was
/// asked for; computed on demand and never stored.
pub struct TrackTable { /* … */ }

impl TrackTable {
    pub fn compute(
        recon: &EditedReconstruction,
        columns: &[TrackTableColumn],
        progress: &Progress<'_>,
    ) -> Result<TrackTable, Cancelled>;

    pub fn point_indexes(&self) -> &[u32];
    pub fn has(&self, column: TrackTableColumn) -> bool;
    pub fn values(&self, column: TrackTableColumn) -> Option<&[f32]>;
    pub fn values_f64(&self, column: TrackTableColumn) -> Option<&[f64]>;
    /// Move `other`'s columns into the ones this table does not hold.
    pub fn merge(&mut self, other: TrackTable) -> Result<(), TrackTableMergeError>;
}

/// `merge` refused a table whose point indexes are not this table's.
pub struct TrackTableMergeError { pub rows: usize, pub other_rows: usize }

pub struct ColumnBounds {
    pub column: TrackTableColumn,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

pub enum SortOrder { Ascending, Descending }

#[derive(Clone, Debug, Default)]
pub struct TrackTableQueryOptions {
    pub bounds: Vec<ColumnBounds>,
    /// Keep only points with an observation in this image.
    pub seen_in_image: Option<u32>,
    /// `None` keeps `live_indexes()` order.
    pub order_by: Option<(TrackTableColumn, SortOrder)>,
}

pub enum TrackTableQueryError {
    /// A bound or the sort names a column `table` does not hold.
    ColumnNotComputed(TrackTableColumn),
    ImageOutOfRange { image: u32, image_count: usize },
}

pub fn query_points(
    recon: &EditedReconstruction,
    table: &TrackTable,
    options: &TrackTableQueryOptions,
) -> Result<Vec<u32>, TrackTableQueryError>;
```

The per-point functions the columns are defined by are public too, because
the viewer calls them for one point at a time:

```rust
pub fn observation_rays(
    image_set: &ImageSet,
    position: &Point3<f64>,
    at_infinity: bool,
    images: impl IntoIterator<Item = usize>,
) -> Vec<[f64; 3]>;
pub fn compute_max_pairwise_angle(rays: &[[f64; 3]]) -> f64;
pub fn compute_max_ray_angle_deg(image_set: &ImageSet, view: &PointView<'_>) -> f32;
pub fn compute_point_diagnostics(image_set: &ImageSet, view: &PointView<'_>) -> (f32, f32);
pub fn compute_position_diagnostics(
    image_set: &ImageSet,
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

- **Compute and query are separate calls.** A caller that shows the table
  computes it once per version and then runs a new query each time a bound
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
use sfmtool_core::analysis::track_table::{
    query_points, ColumnBounds, SortOrder, TrackTable, TrackTableColumn, TrackTableQueryOptions,
};
use sfmtool_core::progress::Progress;

let table = TrackTable::compute(
    &edited,
    &[TrackTableColumn::TrackLength, TrackTableColumn::RmsErrorPx],
    &Progress::none(),
)?;
// The 20 worst-error points with a track of at least 3, seen in image 2.
let worst = query_points(&edited, &table, &TrackTableQueryOptions {
    bounds: vec![ColumnBounds { column: TrackTableColumn::TrackLength, min: Some(3.0), max: None }],
    seen_in_image: Some(2),
    order_by: Some((TrackTableColumn::RmsErrorPx, SortOrder::Descending)),
})?;
let page = &worst[..worst.len().min(20)];
```

## Merging tables

`TrackTable::merge` fills the columns a table does not hold from another table
of the same version. A caller that keeps a table for the version on screen
computes only the columns it lacks, as a second table and off its own thread,
and merges that table in when it lands, so the columns it already holds are
not computed again. The viewer's track table cache does this
([gui/mcp-server.md](../../gui/mcp-server.md) § "`find_points`").

- **The arrays are moved, not copied.** `merge` takes `other` by value and
  moves each of its column vectors into an empty slot, so a merge costs a
  comparison of the rows and nothing per value.
- **A column this table already holds is kept.** Two computes of one column
  for one version give the same values, so when two jobs compute a column at
  the same time, the first to land is kept and the second's copy is dropped.
- **A table of another version is refused.** The two tables' `point_indexes`
  must be equal; otherwise `merge` returns `TrackTableMergeError` with both
  row counts and leaves this table unchanged. Equal rows rule out a point
  added or deleted between the two computes, but not an edit that changed
  values and kept every point. The caller rules that out by computing both
  tables from one version, which the viewer does by keying its cache on the
  document serial.

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
| `rms_error_px` | stored | `Point3D::rms_reprojection_error`, the stored per-point RMS reprojection error in pixels, as its doc comment defines it; the viewer's `get_point` reports the same number as `rms_error_px`. A file holds what its writer stored | defined | never |
| `at_infinity` | stored | 1 for a point at infinity (`w = 0`), 0 for a finite one | defined | never |
| `x`, `y`, `z` | stored | the point's position in the stored frame, in the reconstruction's world-space unit (scene units where none is declared) | no value | never |
| `centroid_distance` | stored | distance from the posed cameras' centroid, the number `ImageSet::placement_scale` gives, in the unit of `x` | no value | the reconstruction has no posed image |
| `normal_confidence` | stored | the stored `normal_confidence` byte, 0 to 255 | defined | the reconstruction has no `normal_confidence` column |
| `lowest_zncc` | stored | the lowest stored per-observation ZNCC in the track, read from `observation_confidence` as `byte / 255` over the non-zero bytes only | no value when no observation of the track was measured | the reconstruction has no `observation_confidence` column |
| `largest_zncc_self_similarity_radius` | stored | the largest ZNCC self-similarity radius among the track's observation readings: the semi-major axis of the stored ellipse, `ellipse_axes[0]`, in patch px, skipping unmeasured rows | no value when every row is unmeasured | the reconstruction has no observation readings |
| `max_ray_angle_deg` | computed | the largest angle between two of the track's viewing rays, the rays from each observing camera's centre to the stored point (`compute_max_ray_angle_deg`) | 0 | never |
| `condition_number` | computed | condition number of the point's triangulation from those rays (`compute_point_diagnostics`) | no value | never |
| `inverse_depth_z` | computed | inverse-depth z-score from the same triangulation, with per-ray angular noise `max(rms_error_px, 1) / f` | no value | never |

A column that is **absent** for a whole reconstruction is reported by
`TrackTableColumn::is_available`, and `compute` skips it rather than failing, so
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
- A bound or a sort on a column the `TrackTable` does not hold is refused with
  `ColumnNotComputed`, which covers both a column the caller did not ask for
  and one the reconstruction could not supply. A `seen_in_image` past the
  image set is refused with `ImageOutOfRange`.

The `TrackTable` must have been computed from the same version the query is
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

Each point's values depend only on that point and the image set, so the
output is bit-identical for every thread count. `query_points` sorts stably
and breaks ties by point index, so equal keys always come out in the same
order, and a test can compare whole results exactly.

## Python bindings

`sfmtool.analysis.track_table(recon, columns=None)` returns a dict that maps
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
from sfmtool.analysis import track_table

table = track_table(recon, ["track_length", "rms_error_px"])
worst = table["point_index"][np.argsort(-table["rms_error_px"], kind="stable")][:20]
```

The query is not bound: a NumPy mask and an `argsort` over the arrays do the
same work, so a Python query function would add nothing. The binding exists so
that scripts and tests read exactly the numbers the viewer shows. It lives in
[analysis/track_table.rs](../../../crates/sfmtool-py/src/analysis/track_table.rs).

## Testing

The Rust tests in
[track_table/tests.rs](../../../crates/sfmtool-core/src/analysis/track_table/tests.rs)
build small reconstructions from the demo reconstruction and check each column
against the stored data or the per-point function it is defined by: a point
at infinity (no position, distance or diagnostics, a ray angle of 0), an image
holding two observations of one point, absent optional columns, unmeasured
`observation_confidence` bytes and unmeasured readings, and deleted and added
points. They check the query's inclusive bounds, NaN against a bound, a bound with
neither end, `seen_in_image`, ties in both sort orders, no-value rows last in
both orders and both refusals; that `merge` moves the missing columns, keeps
the ones already held and refuses a table of another version; that a
cancelled `compute` returns `Cancelled`;
that progress runs to the end through both passes; and that the values are
bit-identical with one thread and with four. The viewer's own tests of the
per-point functions in `crates/sfm-explorer/src/metrics/tests.rs` run against
these definitions through its re-exports.

The binding's tests in
[test_track_table_rust_bindings.py](../../../tests/rust_bindings/analysis/test_track_table_rust_bindings.py)
check the names, dtypes and rows against the seoul_bull ground truth's own
arrays, the column selection and its refusals, and an edited reconstruction.

## Non-goals

The per-observation reprojection error (the largest error in a track, say) is
not a column: for a `sift_files` reconstruction without inline keypoints it
needs every image's `.sift` file read. It can be added as a computed column
that is absent when the reconstruction has no keypoints.

Core does not cache or store the table. A caller that computes it for one version
keeps it for as long as that version is current.
