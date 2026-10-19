# Track Table

**Status:** Draft. Amends [gui/mcp-server.md](../gui/mcp-server.md) (the
"no tool returns bulk arrays" stance, and new tools),
[gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md) (the
non-goal "Point colouring and filtering by quality") and
[gui/panel-layout.md](../gui/panel-layout.md) (a tenth panel). It will file as
two standing specs: `core/analysis/point-columns.md` (Part 1) and
`gui/track-table.md` (Parts 2 to 4, with the wire tools also described in
`gui/mcp-server.md`).

Decided:

- The panel is titled **Track Table**, wire name `track_table`. It pairs with
  Track View, which shows one track; the table lists all of them.
- The columns of the first version are the cheap ones read straight off the
  stored data, plus three computed ones: the maximum ray angle, the condition
  number and the inverse-depth z. The columns are computed by a read-only
  task on the fast background queue, which
  [background-task-queues.md](background-task-queues.md) proposes; the table
  starts no thread of its own.
- The table is a visual tool first. Its first two columns after the point id
  are pictures: **Crop**, the reference photograph cut around the patch, and
  **Patch**, the point's patch bitmap. Both are shown by default, so a
  person sorting by a number sees what each point looks like next to it.
- The table can reach the 3D viewer through an optional toggle that dims or
  hides the points its filters reject.
- Bulk edits on the matched rows are designed for but not built:
  "Remove matching points" is named as a later step, and the query's result
  is shaped so that it can feed `delete_points`.
- One core query backs the panel and the MCP `find_points` read, so a person
  and an agent filtering by the same bounds see the same points.

Not decided: see § "Open questions".

## Purpose

A reconstruction can hold anywhere from a few hundred to a few million 3D
points. Each point comes with numbers that say how well it is supported: how
many photographs see it, how far its observations land from where it
projects, how wide the angle between its viewing rays is, and how well its
patch matches across views. The Track Table lists every point of a
reconstruction as one row, with two pictures of the point first (its
reference photograph cut around it, and its patch) and those numbers as the
columns after them. A person can sort by
any column, set a minimum and a maximum on each, and read off how many points
pass. Clicking a row selects that point everywhere else in the viewer. An
agent driving the viewer runs the same query through `find_points` and gets
back point ids with the same numbers, a page at a time.

The questions this answers are the ones a person asks while judging a solve:
"what are the worst points?", "how many points have a track of two?", "which
points does this image see, and which of them are badly triangulated?".
Today each of these needs a script against the `.sfmr` file, or, for an agent,
one `get_point` call per point. The 2026-10-05 bug bash needed 9,941
`get_point` calls, taking 2 min 11 s, to find three example tracks
([reports/2026-10-05-gui-bugbash.md](../../reports/2026-10-05-gui-bugbash.md)
finding 6).

The design has three layers, and each is useful without the ones above it:

1. **Point columns** in `sfmtool-core`: named per-point values computed for
   every live point of a reconstruction, and a query that filters, sorts and
   pages them.
2. **The Track Table panel**, which shows the columns, edits a query and
   follows the selection.
3. **The wire**: `find_points`, which runs a query and returns the matched
   points, and a read/write pair for the panel's own query.

## Part 1: point columns and the point query (core)

### What a column is

A **point column** is one value per live point of a reconstruction, under a
fixed name. Every column is defined in one place, in `sfmtool-core`. The
viewer's `metrics.rs` already exists so that Track View, the Image Detail
overlay and `get_point` quote the same number. For the same reason, the
definitions the table needs for every point move from it into core, and
`metrics.rs` calls them:

- `compute_max_pairwise_angle`, together with `observation_rays`, which
  `compute_max_track_angle_deg` in `image_detail/mod.rs` copies when it builds
  its rays before calling the same angle function. Core already has a pairwise
  angle function, `ray_angle` in `bench/nearby/candidate.rs`, and the move
  reuses it rather than adding a third copy;
- `compute_point_diagnostics`.

A column belongs to one of two **cost classes**:

- **Stored** columns are read from data the reconstruction already holds,
  in O(1) or O(track length) per point.
- **Computed** columns cost a triangulation or a pass over every pair of
  observations, per point.

The class decides only where the work runs (Part 2, § "Computing the
columns"). It makes no difference to what a column means.

| Wire name | Heading | Class | Definition | Point at infinity | Absent when |
|---|---|---|---|---|---|
| `track_length` | Track | stored | number of observations in the track | defined | never |
| `image_count` | Images | stored | number of distinct images among those observations. It is less than `track_length` when one image holds several observations of the point, which a GLOMAP solve produces (bug bash finding 12) | defined | never |
| `error_px` | Error (px) | stored | `Point3D::error`, the stored per-point reprojection error, which `get_point` reports as `error` | defined | never |
| `at_infinity` | ∞ | stored | 1 for a point at infinity (`w = 0`), 0 for a finite one | defined | never |
| `x`, `y`, `z` | X, Y, Z | stored | the point's position in the stored frame, not the display frame, in the reconstruction's world-space unit (scene units where none is declared) | no value | never |
| `centroid_distance` | Distance | stored | distance from the posed cameras' centroid, the same number `ImageTable::placement_scale` gives, in the stored frame and the same unit as `x` | no value | the reconstruction has no posed image |
| `normal_confidence` | Normal conf. | stored | the stored `normal_confidence` byte, from 0 to 255 | defined | the file has no `normal_confidence` column |
| `lowest_zncc` | Lowest ZNCC | stored | the lowest stored per-observation ZNCC in the track, read from `observation_confidence` as `byte / 255` over the non-zero bytes only. The byte is `round(255 · clamp(z, 0, 1))`, so the range is 0…1 with negative scores at 0, and a `0` byte means the observation was never measured, not that it scored 0 | no value when no observation of the track was measured | the file has no `observation_confidence` column |
| `largest_zncc_self_similarity_radius` | Self-sim. (px) | stored | the largest ZNCC self-similarity radius (the semi-major axis of the stored ellipse, `ellipse_axes[0]`, in patch px) among the track's observation readings, skipping rows that are `NOT_MEASURED` | no value when every row is unmeasured | the file has no observation readings |
| `max_ray_angle_deg` | Ray angle (°) | computed | the largest angle between two of the track's viewing rays, as `metrics::compute_max_pairwise_angle` defines it today and the Image Detail `max_track_angle` heat map shows it | 0 | never |
| `condition_number` | Cond. | computed | condition number of the point's triangulation, as `metrics::compute_point_diagnostics` defines it | no value | never |
| `inverse_depth_z` | Depth z | computed | inverse-depth z-score from the same diagnostics | no value | never |

A column that is **absent** for a whole reconstruction is not offered for it:
the panel leaves it out, and `find_points` refuses a query that names it,
saying which columns that reconstruction does have. **No value** for a single
point is a NaN in the store. A no-value row fails every bound set on its
column, and it sorts after every row that has a value, whichever way the sort
runs. That is Track View's rule for empty cells.

The per-observation reprojection error (the largest error in a track, say) is
left out of the first version. For a `sift_files` reconstruction without
inline keypoints, it needs every image's `.sift` file read. It can be added as
a computed column that is absent when `has_keypoints()` is false.

### Rust API

The columns and the query live in `crates/sfmtool-core/src/analysis/point_columns.rs`,
bound as `sfmtool.analysis.point_columns`.

```rust
/// One named per-point value. The wire name is the stable identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PointColumn {
    TrackLength, ImageCount, ErrorPx, AtInfinity, X, Y, Z, CentroidDistance,
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
    /// Whether `recon` can supply this column at all (§ "Absent when").
    pub fn is_available(self, recon: &EditedReconstruction) -> bool;
}

/// Column values for every live point of one version, in `live_indexes()`
/// order. Holds only the columns it was asked for.
pub struct PointColumns { /* … */ }

impl PointColumns {
    /// Compute `columns` for every live point of `recon`. Columns the
    /// reconstruction cannot supply are skipped, not an error; `has` says
    /// which were filled.
    pub fn compute(
        recon: &EditedReconstruction,
        columns: &[PointColumn],
        progress: &Progress,
    ) -> Result<PointColumns, Cancelled>;

    pub fn point_indexes(&self) -> &[u32];
    pub fn has(&self, column: PointColumn) -> bool;
    /// `None` when the column was not computed. Position columns are read
    /// through `values_f64`.
    pub fn values(&self, column: PointColumn) -> Option<&[f32]>;
    pub fn values_f64(&self, column: PointColumn) -> Option<&[f64]>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColumnBounds {
    pub column: PointColumn,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SortOrder { Ascending, Descending }

/// What to match and in which order. Settings chosen by the caller, so
/// `*Options` by the type-name rule. (The viewer's Go to Point dialog has an
/// unrelated `goto_point::PointQuery`, the parse of what was typed; this type
/// is named apart from it.)
#[derive(Clone, Debug, Default)]
pub struct PointQueryOptions {
    pub bounds: Vec<ColumnBounds>,
    /// Keep only points with an observation in this image.
    pub seen_in_image: Option<u32>,
    /// `None` keeps `live_indexes()` order.
    pub order_by: Option<(PointColumn, SortOrder)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PointQueryError {
    /// A bound or the sort names a column `columns` does not hold.
    ColumnNotComputed(PointColumn),
    ImageOutOfRange { image: u32, image_count: usize },
}

/// The point indexes that match, in order. The full list, not a page: the
/// panel scrolls it, `find_points` slices it, and a later "Remove matching
/// points" passes it to `delete_points` unchanged.
pub fn query_points(
    recon: &EditedReconstruction,
    columns: &PointColumns,
    options: &PointQueryOptions,
) -> Result<Vec<u32>, PointQueryError>;
```

**Why it is shaped this way:**

- **Compute and query are separate calls.** The panel computes the columns
  once per version and then runs a new query at every keystroke in a filter
  box. A single `find_points(recon, options)` function would recompute the
  maximum ray angle for every point each time a bound changed.
- **Columns are computed by name.** The panel and the wire both ask for only
  what they show or filter on, so the computed columns are not computed for
  a person who never shows, sorts or bounds one.
- **It works on `EditedReconstruction`.** That is what the viewer holds. A
  caller with an `SfmrReconstruction` wraps it with
  `EditedReconstruction::new(Arc::new(recon))`, as the bench does.
- **The result is a `Vec<u32>` of point indexes, not row positions.** A point
  index is what `select_point`, `get_point` and `delete_points` take, so the
  result passes to any of them without a translation.
- **It takes one `&Progress`.** `compute` can run for over a second on a
  million points, so by the interface contract it takes a `&Progress` and
  stops when it is cancelled. `query_points` is a filter and a sort over
  arrays in memory, and it takes none.

Example:

```rust
use sfmtool_core::analysis::point_columns::{
    query_points, ColumnBounds, PointColumn, PointColumns, PointQueryOptions, SortOrder,
};
use sfmtool_core::progress::Progress;

let columns = PointColumns::compute(
    &edited,
    &[PointColumn::TrackLength, PointColumn::ErrorPx],
    &Progress::none(),
)?;
// The 20 worst-error points with a track of at least 3, seen in image 2.
let worst = query_points(&edited, &columns, &PointQueryOptions {
    bounds: vec![ColumnBounds { column: PointColumn::TrackLength, min: Some(3.0), max: None }],
    seen_in_image: Some(2),
    order_by: Some((PointColumn::ErrorPx, SortOrder::Descending)),
})?;
let page = &worst[..worst.len().min(20)];
```

### Determinism and precision

Values are stored as `f32`, except the position columns, which are stored as
`f64`. The table shows three or four significant digits and its filters
compare against bounds a person typed, so `f32` is enough for every derived
value. A position is not: a georeferenced reconstruction can sit 10⁶ m from
the origin, where an `f32` resolves only about 6 cm. Eleven `f32` columns
and three `f64` ones over 10⁶ points take 68 MB, and making every column `f64`
would add another 44 MB for no gain.

`compute` runs per point under rayon, and each point's values depend only on
that point, so the output is the same for every thread count. Every column
handles points at infinity as the table says: a value where one is defined
(`max_ray_angle_deg` is 0, since a bearing's rays are parallel), and no value
where it is not. `query_points`
sorts stably and breaks ties by point index, so equal keys always come out in
the same order. A test can compare whole results exactly.

### Python bindings

`sfmtool.analysis.point_columns(recon, columns=None)` returns a dict that maps
each wire name to a NumPy array (`float32`, or `float64` for `x`, `y` and `z`),
plus `"point_index"` (`uint32`). With `columns=None` it computes every column
the reconstruction can supply. The query is not bound: a NumPy mask and an
`argsort` over the arrays do the same work, so a Python query function would
add nothing. The binding exists so that scripts and tests read exactly the
numbers the viewer shows.

## Part 2: the Track Table panel

### Placement

`Tab::TrackTable`, titled **Track Table**, wire name `track_table`. Its home is
the bottom group, beside Image Browser, Action Log and Edit History, because
that group is wider than it is tall and the table has many columns. It is in
`Layout::default()` but is not the raised tab there, so the default layout
opens showing the same panels as it does now.

### What it shows

```
┌ Track Table ─ dino ───────────────────────────────────────────────────────────────────┐
│ 1,204 of 9,941 points  [Columns ▾]  Size [──●──] ☐ Only points seen in selected image │
│ ☐ Filter the 3D view: (•) Dim ( ) Hide                                                │
├────────┬────────┬────────┬─────────┬─────────┬────────────┬───────────────┬───────────┤
│ Point  │ Crop   │ Patch  │ Track ▼ │ Images  │ Error (px) │ Ray angle (°) │ Cond.     │
│        │        │        │ [3 ][  ]│ [  ][  ]│ [   ][1.5 ]│ [2  ][    ]   │ [  ][  ]  │
├────────┼────────┼────────┼─────────┼─────────┼────────────┼───────────────┼───────────┤
│ #9873  │ ░▓▓▓▓░ │ ▓▓▓▓▓▓ │    57   │    45   │    0.82    │     41.2      │    3.1    │
│        │ ░▓▓▓▓░ │ ▓▓▓▓▓▓ │         │         │            │               │           │
├────────┼────────┼────────┼─────────┼─────────┼────────────┼───────────────┼───────────┤
│ #830   │ ░░░░░░ │ ▓▓▓▓▓▓ │    18   │    18   │    0.61    │     22.7      │    …      │
│        │ ░░░░░░ │ ▓▓▓▓▓▓ │         │         │            │               │           │
└────────┴────────┴────────┴─────────┴─────────┴────────────┴───────────────┴───────────┘
```

The second row's crop is still an empty frame because its photograph is
still being decoded.

- **It follows the selected reconstruction**, as Image Browser and Edit
  History do. The header names that reconstruction and counts matched rows
  against live points.
- **The heading row** sorts by the column that is clicked. The first click
  sorts the bad end first where the column has one (`error_px`,
  `condition_number`, `inverse_depth_z` and
  `largest_zncc_self_similarity_radius` descending; `track_length`,
  `image_count`, `max_ray_angle_deg`, `lowest_zncc` and `normal_confidence`
  ascending), and ascending otherwise. A second click reverses the order. A
  triangle marks the sorted column, and no-value rows go last. This is
  Track View's sort rule (`track_view/body/table.rs`, `TableSort`), moved into
  a helper that both tables call.
- **The bounds row**, under the headings, has a minimum box and a maximum box
  per column, in the style of Track View's threshold row. An empty box sets no
  bound. Rows are filtered again on every edit.
- **Columns ▾** lists every available column with a checkbox. The default set
  is Point, Crop, Patch, Track, Images, Error, ∞ and Ray angle. A column that holds a bound
  or the sort cannot be hidden until those are cleared, so a hidden column
  never filters rows without the person seeing it.
- **The rows** are virtualised with `ScrollArea::show_rows` at a uniform row
  height, as the Action Log is, so a million rows cost only the visible ones
  per frame. A cell whose computed column is still being computed shows `…`.

### Picture columns

The two picture columns show what each point looks like in the
photographs, which a column of numbers does not. Crop comes before Patch, as
in Track View, so the photograph as it is comes before the picture made from
it.

| Column | Shows | Absent when |
|---|---|---|
| **Crop** (`crop`) | the reference photograph cut around the patch's outline, with the outline drawn over it, by Track View's crop rule ([track-view.md](../gui/track-view.md) § "The observation table", the paragraph "The crop beside the tile shows the patch as the photograph holds it"): the patch frame at the reference observation's keypoint, its boundary projected through the camera's own model, the bounding box widened by one pixel and made square, at most 128 texels a side. Track View strokes the outline in the row's verdict colour; the table has no verdict and strokes it white over a wider dark stroke, the way Track View draws its marks. The photograph is the point's reference observation (§ "The reference photograph") | the reconstruction has no patch frames or no stored keypoints, the usual case for a `sift_files` reconstruction |
| **Patch** (`patch`) | the point's patch bitmap (`PointView::patch_bitmap`): the bitmap the file stores, or the display patch bitmap the open renders for a file with patch frames and no bitmaps. This is the picture in Track View's header slot (`patch::stored_patch_image`), not Track View's per-row *Patch* tile, which is rendered again from each observation's own photograph. It is drawn with nearest filtering | the reconstruction has no patch bitmap column: no stored bitmaps, and no display bitmaps, because the file has no patch frames, has no stored keypoints, or none of its photographs could be read. A `sift_files` reconstruction is usually in this case |

#### The reference photograph

The Crop column's photograph is the point's reference observation
(`PointView::reference_observation`, which includes a pick that only the
open's display render made). A point with no reference, stored as `-1` or in
a file with no `reference_observations` column, gets one from the
reference-view rule, run at the point's stored keypoints. This is the same
rule Track View's evaluation uses to choose a point's reference. The
single-point step every bitmap writer shares runs it
(`stored_bitmap::render_patch_bitmap`; the open's display render calls it
through the whole-cloud `render_patch_cloud_bitmaps`), and the table calls
`stored_bitmap::render_reference`, which makes the pick without keeping the
bitmap. It is the pick an evaluation would make without localising the rows
first, so it needs the track's photographs but not a fit.

Only one kind of file reaches this path: a file with stored bitmaps and `-1`
references, such as a file older than `.sfmr` v12, or one whose bitmaps are a
mean of the views. The open renders display bitmaps only for a file with no
stored bitmaps, so for such a file no pick has been made yet. The Patch cell
still shows the stored bitmap, which need not be the render of the
photograph the crop shows. Its hover text is a new text for this case: that
the point stores no reference, so the bitmap is not the render of the
photograph in the Crop cell.

A point the open's display render left at `-1` is not queued. The display
render already ran the rule on the same keypoints and photographs and found
no eligible view (fewer than two usable views see the point, or the rule
excluded them all), and the same rule would pick nothing again. Its cell says
*no reference* straight away.

The picks run as one quiet, read-only task on the fast queue
([background-task-queues.md](background-task-queues.md)), *Track Table
reference picks*, which works through the visible rows one pick at a time
and is cancelled when the document serial changes. It locks no node, so it
runs while a slow task such as a bundle adjustment holds the node, against
the version at the cursor. Picks are queued only for rows that are visible
once the scroll has stopped (below), and a row that leaves the view is taken
off the queue. Each pick is cached by `(ReconId, point, document_serial)` and
held by the table. It is not written into the version, so it costs no undo
step and does not reach a save. A commit from Track View is still the way to
store a reference.

#### Drawing the pictures

- **The cells take values, not a track.** Track View's tile and crop helpers
  (`track_view/body/crop.rs`, `track_view/body/tile.rs`) take an
  `EditableTrack` and an observation index. The crop's outline, region, image
  and hover view are rewritten to take the values they read from it: a patch
  frame (with its `w`), the camera image, the keypoint and the point's
  projection. The helpers move to a module both panels use, and Track View
  passes its track's values. A `PointView` supplies the same values, so the
  table builds no track per row. Points at infinity are already handled by the
  helpers through the patch frame's `w` and homogeneous projection.
- **The pictures set the row height.** A **Size** slider in the header sets
  the picture size from 32 to 128 points, defaulting to 64. Every row has that
  height while a picture column is shown, and text height while neither is
  shown, so `show_rows` keeps uniform rows.
- **Hovering the crop** shows Track View's crop hover view: the crop centred
  in three times its width and height of the photograph, with the outline, a
  dot at the keypoint, the amber ring where the point projects, and the
  photograph's name above it. **Hovering the patch** shows the bitmap at three
  times the Size, with the text Track View's header slot gives a viewed point
  (`track_patch_hover`), or, for a point that stores no reference, the text
  in § "The reference photograph". A click on a picture is
  the row's click, as in Track View.
- **Picture columns cannot be sorted or bounded.** Clicking their headings
  does nothing, and they have no bounds boxes.
- **Each cell says why it is empty.** A picture cell is one of:
  - the picture;
  - an empty frame while its photograph is decoding or its reference is being
    picked;
  - *unreadable*, when the photograph cannot be read
    (`DisplayPhotograph::Unreadable`);
  - *no reference*, when the rule picked none;
  - *off image*, when no point of the outline projects into the photograph
    (`crop::region` gives no region). An outline that projects but lies
    outside the photograph still gives a crop, drawn transparent where it is
    off the photograph, as in Track View;
  - *no patch*, for an all-zero bitmap row, which the display render leaves
    for a point it could not render.
- **Only visible rows are drawn.** Patch textures come from bitmaps already in
  memory and are cached in a texture cache keyed by
  `(ReconId, point, document_serial)` and bounded by entry count, so
  scrolling back over rows already seen costs no upload. Crops need their
  photograph decoded. The table asks the shared photograph cache
  (`state::display_photograph`) for the visible rows' photographs. Track View
  only checks that cache, so the table is a new source of decodes.
- **Requests wait for the scroll to stop.** A sort by a number puts
  neighbouring rows in different photographs, so dragging the scroll bar
  through a large table would otherwise queue a decode for every photograph
  it passes. The table asks for photographs and queues reference picks only
  once the scroll position has stopped changing, and while it is still
  changing it draws the crops it already holds. The delay is to be measured
  on `dino_dog_toy` (85 photographs at 2040×1536).
- **The cost is bounded by the photograph cache.** A page of rows from a
  metric sort can touch most of a reconstruction's photographs, and a
  reference pick decodes every photograph its track observes. The cache's
  budget is a quarter of physical memory, clamped to 1–16 GiB (4 GiB when
  physical memory cannot be read, and an environment variable overrides it;
  `photograph_cache::default_budget_bytes`). All of `dino_dog_toy`, 85
  photographs at 2040×1536 with their pyramids, takes about 1.07 GB decoded,
  so it fits on any machine with more than about 4 GiB of memory. A
  reconstruction with more photographs than the budget holds evicts them and
  decodes them again as the table scrolls. That costs time but no memory
  beyond the budget.
- **The wire does not return pictures.** `find_points` returns numbers; an
  agent sees the pictures with `screenshot {panel_name: "track_table"}`, after
  `set_track_table_query` has sorted and filtered the table. The panel
  query's `columns` field names the picture columns `crop` and `patch`;
  `find_points` refuses them in its own `columns`, saying that picture columns
  belong to the panel only.

### Selection and hover

- **Clicking a row** calls `AppState::select_point`, so Track View, Image
  Detail and the 3D viewer follow it as they follow any other selection, and
  the Action Log gets its Selection row.
- **Hovering a row** sets `hovered_point`, as the Scene tree and Image Detail
  hovers do. The 3D viewer and Image Browser highlight it.
- **When the selection changes elsewhere**, the table highlights the selected
  point's row and scrolls it into view if it is among the matched rows. It
  does not change the filters to make a filtered-out point appear.

### The query is panel state

The bounds, the sort, the visible columns, the "seen in the selected image"
box and the 3D toggle are UI state of the panel, like the Image Detail display
settings. They are not held in a version, cost no undo step and are not saved
with the reconstruction. A single query is used for whichever reconstruction
is selected. A bound on a column that the newly selected reconstruction lacks
is kept but greyed out, and it does not filter until a reconstruction that has
the column is selected again.

### Computing the columns

The stored columns and the computed ones are produced by a **column job**, a
quiet, read-only task on the fast queue
([background-task-queues.md](background-task-queues.md)), *Track Table
columns*. It locks no node and writes no Action Log row, so a person can sort
and filter the version on screen while a slow task such as a bundle
adjustment holds the node. When that task installs its version, the serial
changes and the columns are computed again. It is keyed by
`(ReconId, document_serial)`. A result for an older serial is dropped, and an
edit cancels the running job through its `Progress`. The job publishes the
stored columns as soon as they are filled, and the computed ones after them,
so the table is usable while the slow columns are still being computed.

The job starts when the panel is visible, or when a `find_points` call needs a
column (Part 3). Computed columns are computed only once a visible column, a
bound or the sort needs them. A hidden panel with no wire caller costs
nothing.

A sort or a bound on a computed column that is not ready yet leaves the row
order as it is, and says *sorting when Ray angle is ready* in the header,
rather than ordering the rows by partial values.

### Filtering the 3D view

**Filter the 3D view** applies the table's matched set to the selected
reconstruction's point cloud:

- **Dim** draws the rejected points at a reduced alpha, and they can still be
  picked.
- **Hide** leaves them out of the draw and out of picking.

Selected and hovered points are always drawn at full strength, so a click in
the table never selects a point that cannot be seen. The mask is a per-point
bit buffer uploaded beside the node's point instances. It is rebuilt when the
matched set changes and dropped when the toggle is cleared. Other
reconstructions are not affected. The HUD's stats line adds *(N shown)* while
the toggle is on.

This revises the point-cloud-rendering non-goal. The 3D cloud is filtered by
the table's query. Colouring the cloud by a column stays a non-goal.

**No pictures for a `sift_files` reconstruction.** Such a reconstruction
normally has no patch frames, so there is no outline to cut a crop around and
no bitmap to show. Both columns appear after Convert to Embedded Patches.

## Part 3: the wire

### `find_points` (read)

```jsonc
// find_points
{
  "reconstruction_label": "dino",            // optional: defaults to the selected one
  "bounds": [ { "column": "track_length", "min": 3 },
              { "column": "error_px", "max": 1.5 } ],
  "seen_in_camera_image": 2,                 // optional: an index or a name, as camera_image takes
  "order_by": { "column": "error_px", "descending": true },  // optional
  "columns": ["track_length", "image_count", "error_px"],  // optional: the values to return
  "offset": 0, "limit": 50                   // as list_camera_images: default 50, at most 500
}
// reply
{
  "reconstruction_label": "dino",
  "live_points": 9941,
  "matched": 1204,
  "offset": 0,
  "points": [
    { "id": "pt3d_2db7e1f4_9873", "index": 9873,
      "track_length": 57, "image_count": 45, "error_px": 0.82 },
    …
  ]
}
```

- **It runs `query_points`** over the same column cache the panel uses, so a
  `find_points` call and the panel with the same bounds agree. The returned
  value columns default to the bound and sorted columns plus `track_length`
  and `error_px`.
- **A computed column defers the reply** until the column job has filled it,
  in the same way a deferred screenshot replies on a later frame. The UI
  thread is not blocked while it waits. `limit` and the cache keep the cost of
  a repeat call down to a slice.
- **The reply is paged scalars.** It carries no bulk arrays: at most 500 rows
  of a few numbers each. The mcp-server.md statement "no tool here returns bulk
  arrays" is narrowed to "no tool returns a whole column or a whole track
  table; `find_points` returns at most a page of summary values", and the
  `get_scene` description changes to match.
- **Errors** name the column that is not available and list the ones that
  are, or name the camera image that is out of range. They use the same
  wording as the panel's greyed bound.

This closes bug bash finding 6. The three tracks the bug bash searched for
take three calls:

- `{seen_in_camera_image: 2, bounds: [{column: "track_length", min: 16, max: 20}]}`
- `{seen_in_camera_image: 56, bounds: [{column: "track_length", min: 4, max: 4}]}`
- `{order_by: {column: "track_length", descending: true}, limit: 1}`

### `get_track_table_query` / `set_track_table_query`

These read and write the panel's query: `bounds`, `order_by`, `columns`
(visible), `seen_in_selected_camera_image` and `filter_3d`
(`"off" | "dim" | "hide"`). `set_track_table_query` merges the fields it is
given, like `set_viewer_3d_display`. It is refused when a field names a column
that does not exist, and it logs in the panel's own words. With these an agent
can set up the view a person sees, for a screenshot recipe (bug bash finding
17), without clicking through the boxes. The panel does not have to be open
for its query to be set. The panel opens with it.

## Part 4: later, Remove matching points

A **Remove matching points** button, and a `delete_matching_points` tool or a
`find_points`-then-`delete_points` recipe, would delete every matched point as
one version, through the existing `delete_points` edit. The result of
`query_points` is already the list of indexes that edit takes, valid at the
document serial it was computed for, so a removal has to refuse a match set
computed for an older serial. This is not specified here beyond that. A draft
of its own will decide on a confirmation, whether the hidden rows of a
filtered view count, and the wording in the Action Log.

## Steps

Each step is one PR through `implement-change`. Steps 2, 4 and 5 put quiet
tasks on the fast background queue, so they come after Step 3 of
[background-task-queues.md](background-task-queues.md). Step 1 does not
depend on it.

1. **Core point columns and query.** This step adds `point_columns.rs`, moves
   the max-angle and diagnostics definitions from `metrics.rs` into core (with
   `metrics.rs` calling them), adds the Python binding and tests, and files
   `core/analysis/point-columns.md`.
2. **`find_points`**, which brings the column cache and the column job into the
   viewer. Only the wire uses them at this step, and the mcp-server.md stance
   is revised.
3. **The Track Table panel** with the stored columns, the Patch column, the
   Size slider, sorting, bounds, selection and hover. This step adds the
   tenth panel to the layout, its tests and its specs, and moves
   `stored_patch_image` out of Track View's private `patch` module into the
   module both panels use.
4. **The Crop column**, moving Track View's tile and crop helpers into a shared
   module and rewriting them to take values rather than a track, with
   requests that wait for the scroll to stop and reference picks for the
   visible rows that store none.
5. **The computed columns in the panel**, and the deferred `find_points` reply
   for them.
6. **Filter the 3D view** (dim and hide).
7. **`get_track_table_query` / `set_track_table_query`.**

## Open questions

- **`error_px`'s definition and name.** `Point3D::error` is documented as an
  "RMS reprojection error", but `recompute_point_errors` stores the mean. A
  file converted from COLMAP and never recomputed may hold COLMAP's value. The
  heading has to name one definition. Settle which, and fix the doc comment.
  `get_point` calls the same number `error`. Decide whether the column takes
  that name or `get_point` gains the unit suffix.
- **Two "max ray angle" definitions.** The viewer's is the largest angle
  between two rays, and core already uses `max_ray_angle_deg` in that sense
  (`bench/nearby/candidate.rs`, `far_field.rs`, `layers.rs`).
  `analysis::inspect_point`'s `max_ray_angle_deg` is the exception: it is the
  largest angle from the mean direction. This draft keeps the pairwise
  meaning. Renaming `inspect_point`'s field, and the `sfm inspect` output that
  prints it, is the likely fix, with a glossary entry for the result.
- **Does Image Detail follow the 3D filter?** Dimming the tracked features of
  rejected points in Image Detail would extend the same toggle to a second
  panel. This draft limits the toggle to the 3D viewer.
- **Does the query persist?** It could be kept in the layout file, or
  per-session only, as drafted.
- **Multi-column sort** (Shift+click a heading for a secondary key) is not in
  the first version.
- **`lowest_zncc`'s source.** The column reads `observation_confidence`. The
  observation readings carry `blur_matched_bitmap_zncc` too, which may be the
  better reading where both exist.
- **Column names.** `lowest_zncc` and `largest_zncc_self_similarity_radius`
  avoid `min_zncc` and `max_zncc_self_similarity_radius`, which are bench bars
  on the wire. Settle the column words in a glossary entry when Step 1 is
  filed.
