# Track Table

The Track Table is the SfM Explorer panel that lists every 3D point of the
selected reconstruction as one row, with the numbers that say how well the
point is supported as its columns: how many observations its track has, how
many photographs see it, its reprojection error, whether it is at infinity,
where it is, and what its stored patch readings say. A person sorts by any
column with a click on its heading, sets a minimum and a maximum on each, and
reads off how many points pass. Clicking a row selects that point everywhere
else in the viewer. A reconstruction can hold a few million points, so the
table draws only the rows on screen and keeps the filtered, sorted list of rows
between frames.

The questions it answers are the ones a person asks while judging a solve:
"what are the worst points?", "how many points have a track of two?", "which
points does this image see, and which of them have a large error?". Without it
each of those needs a script against the `.sfmr` file.

The panel is the viewer's face of the **track table**
([core/analysis/track-table.md](../core/analysis/track-table.md)): named
per-point values computed on demand for one version of a reconstruction, one
row per track keyed by point index, and a query that filters and sorts them.
The panel and the MCP read `find_points`
([mcp-server.md](mcp-server.md) § "`find_points`") run that query over one
cache, so a person and an agent filtering by the same bounds see the same
points. Track View ([track-view.md](track-view.md)) shows one track; this panel
lists all of them.

The panel shows the stored columns, the ones read straight off the
reconstruction. The pictures of each point (its reference photograph cut
around its patch, and its patch bitmap), the computed columns (the largest ray
angle, the condition number and the inverse-depth z), applying the filter to
the 3D view and the wire tools that read and write the panel's query are
proposed in [drafts/track-table.md](../drafts/track-table.md).

## Placement

`Tab::TrackTable`, titled **Track Table**, wire name `track_table`, the tenth
panel. Its state is `TrackTablePanel`, in
[track_table_panel.rs](../../crates/sfm-explorer/src/track_table_panel.rs),
with its table job in
[track_table_panel/job.rs](../../crates/sfm-explorer/src/track_table_panel/job.rs).

Its home is the centre group, the node of the 3D Viewer and Image Detail
([panel-layout.md](panel-layout.md) § "Home positions"). That is the largest
area of the window, with the height for many rows. In `Layout::default()` it is
the last tab of that node, after the 3D Viewer and Image Detail, and the 3D
Viewer stays the raised tab, so the default layout opens showing the same
panels as it did before the table existed, with the table behind both until
its tab is clicked or `show_panel {"panel_name": "track_table"}` raises it. Its
Panels menu entry is ticked from the start, since the panel is open.

A panel opened with nothing of its group on screen goes where the 3D Viewer
goes (`Home::Root`): it becomes the root of an empty dock, or a new tab of the
root's first leaf. A layout file saved before the panel existed still loads,
and the table arrives closed, as it does for any panel a file does not name
([panel-layout.md](panel-layout.md) § "Parameters", `LAYOUT_VERSION`); the
Panels menu opens it.

## What it shows

```
┌ Track Table ──────────────────────────────────────────────────────────────────┐
│ dino  1,204 of 9,941 points  [Columns]  ☐ Only points seen in IMG_0042.jpg    │
├────────┬─────────┬─────────┬────────────┬──────┬──────────────────────────────┤
│ Point  │ Track ▼ │ Images  │ Error (px) │ ∞    │                              │
│        │[3 ][   ]│[  ][  ] │[   ][1.5]  │[ ][ ]│                              │
├────────┼─────────┼─────────┼────────────┼──────┼──────────────────────────────┤
│ #9873  │      57 │      45 │      0.820 │      │                              │
│ #830   │      18 │      18 │      0.610 │      │                              │
└────────┴─────────┴─────────┴────────────┴──────┴──────────────────────────────┘
```

- **It follows the selected reconstruction**, as Image Browser and Edit History
  do. The header names that reconstruction and counts the matched rows against
  its live points, with the thousands grouped (*1,204 of 9,941 points*). With no
  reconstruction loaded the panel says *No reconstruction loaded*. Until the
  table job has computed the columns for the version at the cursor, the panel
  says *Computing the track table of {label}…* beside a spinner.
- **The columns** are the Point column, which is always shown and is the row's
  key, `#` and the point index, and the stored track table columns the
  reconstruction can supply. A column the reconstruction cannot supply
  (`TrackTableColumn::is_available` false, such as `normal_confidence` on a
  file with no such stored column) is left out: it has no heading, no bounds
  and no entry in the Columns menu.

  | Heading | Column | Cell |
  |---|---|---|
  | Point | the point index | `#9873` |
  | Track | `track_length` | a whole number |
  | Images | `image_count` | a whole number |
  | Error (px) | `rms_error_px` | three significant digits, at most three decimals |
  | ∞ | `at_infinity` | `∞` for a point at infinity, empty for a finite one |
  | X, Y, Z | `x`, `y`, `z` | four significant digits, every whole digit kept, at most three decimals |
  | Distance | `centroid_distance` | as X |
  | Normal conf. | `normal_confidence` | a whole number, 0 to 255 |
  | Lowest ZNCC | `lowest_zncc` | two decimals |
  | Self-sim. radius | `largest_zncc_self_similarity_radius` | three significant digits, at most three decimals |

  A point with no value in a column (a NaN in the table: a point at infinity's
  `x`) shows `-`. A column the cache does not hold yet shows `…`. The number
  cells are right-aligned.
- **Columns**, a menu, lists every column the reconstruction can supply with a
  checkbox. The default set is Track, Images, Error and ∞. A column that holds
  a bound or the sort cannot be hidden until those are cleared, so a hidden
  column never filters or orders the rows without the person seeing it. Its
  checkbox is greyed, and its hover text says why. The menu stays open while
  boxes are ticked.
- **Only points seen in the selected image** keeps the points with an
  observation in the image selected in this reconstruction, and the label
  names that image. With no image of the reconstruction selected the box
  filters nothing, its label is dimmed while it is ticked, and its hover text
  says so.
- **The heading row** sorts by the column clicked. The first click on a
  column sorts the bad end first where the column has one: `rms_error_px` and
  `largest_zncc_self_similarity_radius` descending, `track_length`,
  `image_count`, `lowest_zncc` and `normal_confidence` ascending. Every other
  column sorts ascending first. A second click on the same heading reverses the
  order. A triangle after the heading marks the sorted column, pointing up for
  increasing and down for decreasing. Rows with no value go last in either
  order, which is the track table query's rule. A click on **Point** puts the
  rows back in increasing point index, the order with no sort. The click rule
  is Track View's (`crate::table_sort`, which both tables call). Each heading's
  hover text says what the column holds, its wire name and what a click on it
  does.
- **The bounds row**, under the headings, has a minimum box and a maximum box
  per column. A row passes when its value is at least the minimum and at most
  the maximum, both inclusive, and every bound must pass; a row with no value
  fails any bound that sets an end. An empty box sets no bound. A box whose
  text is not a number sets none either, and its text is drawn in the error
  colour with a hover text saying so. The rows are filtered again on the frame
  after an edit of a box. Each box is named for its column, *minimum Error
  (px)*, so `get_widgets` and a screen reader say which box it is.
- **The rows** are virtualised with `ScrollArea::show_rows` at one text line's
  height, as the Action Log is, so a million rows cost only the visible ones per
  frame. Alternate rows are shaded. The table scrolls sideways when its columns
  are wider than the panel, with the headings and the bounds moving with the
  rows; they stay put while the rows scroll up and down. A new sort, new bounds,
  the selected-image box or another reconstruction scrolls the rows back to the
  top, where the rows the query puts first are; a new version of the same
  reconstruction does not.
- **A query the reconstruction cannot honour is said, not dropped.** The query
  is one for whichever reconstruction is selected (§ "The query is panel
  state"), so it can name a column the newly selected reconstruction cannot
  supply. Its bound and its sort are kept and do not filter or order, and the
  line under the header says *Not filtering or sorting by Normal conf.: dino
  cannot supply it. The bounds are kept for a reconstruction that can.* A bound
  or the sort on a column the cache does not hold yet is not applied either, and
  the line says *Filtering and sorting when {heading} is ready.*

## Selection and hover

- **Clicking a row** calls `AppState::select_point`, so Track View, Image
  Detail and the 3D viewer follow it as they follow any other selection, and
  the Action Log gets its *Selected point* row.
- **The selected point's row** is drawn in the selection colour. When the
  selection moves elsewhere, the table scrolls the selected point's row into
  view, near the middle, if it is among the matched rows. It does not change the
  bounds to make a filtered-out point appear. A click in the table does not
  scroll it.
- **Hovering a row** sets `hovered_point`, as the Scene tree and Image Detail
  hovers do, so the 3D viewer and Image Browser highlight it. The hover text
  is the point's portable id, `pt3d_<hash>_<index>`. While the pointer is over
  the rows the panel owns the hover, and over no row it clears it; leaving the
  rows clears it once, so a hover set by another panel is left alone. A row
  another panel hovers is drawn with the hover fill, its text in the strong
  colour so it keeps its contrast over the lighter fill.

## The query is panel state

The bounds, the sort, the visible columns and the "seen in the selected image"
box are UI state of the panel (`TrackTableQuery`), like the Image Detail
display settings. They are not held in a version, cost no undo step and are not
saved with the reconstruction or in the layout file. A single query is used for
whichever reconstruction is selected.

## Computing the columns

The panel reads the **track table cache** that `find_points` reads: one core
`TrackTable` per node, for the version at its cursor, keyed by
`(ReconId, document_serial)` (`AppState::track_tables`;
[mcp-server.md](mcp-server.md) § "`find_points`"). The columns are computed by
the **table job**, a quiet, read-only task on the fast queue, *Track Table
columns* (`Operation::TRACK_TABLE_COLUMNS`;
[background-tasks.md](background-tasks.md) § "Quiet tasks").

- **It starts when the panel is drawn.** Each frame the panel draws, it names
  the node it shows (`AppState::want_track_table_columns`), and the frame
  clears that before the dock draws (`AppState::hide_track_table`), as Track
  View does for the viewed track. Once per frame, before the MCP drain and again
  after the egui pass, `AppState::drive_track_table_columns` queues the job
  when the cache lacks a stored column that node can supply. A hidden panel
  costs nothing.
- **It computes only what the cache lacks.** The job runs
  `TrackTable::compute` for the missing stored columns of the value at the
  cursor when it was queued, and the GUI thread merges that table into the
  cached one with `TrackTable::merge`, as *Find points* does
  ([core/analysis/track-table.md](../core/analysis/track-table.md) § "Merging
  tables"). A column either of them computed is kept, so a column `find_points`
  computed is not computed again for the panel, and the reverse.
- **An edit cancels it.** When the document serial at its node's cursor moves
  while the job runs (an edit, an undo, a redo, a slow task landing its
  version) or the node is closed, the job is asked to stop through its
  `Progress`. A table that comes back for a serial the node has left is dropped
  rather than merged. The next job is queued, for the version now at the
  cursor, once the cancelled one has reported back. A running job whose node is
  still at its serial finishes even when the panel has moved to another node,
  since what it computes is kept; a waiting job that is no longer wanted is
  taken out of the queue.
- **It locks nothing and does not wait for its node.** It pushes no version
  and keeps its answer in the cache keyed by the serial it read, so it runs
  while a slow task such as a bundle adjustment holds the node, against the
  version on screen, which is the value the slow task was handed. When that
  task lands its version the serial moves, and the table is computed again.
- **It is quiet.** It writes no Action Log row, is not reported on the wire,
  and gives way to any asked-for task queued in the fast queue, *Find points*
  among them; the driver queues it again behind that task. Among the quiet
  tasks it waits after the focused item's evaluation and ahead of the other
  bench items' evaluations.

## The row list

The rows that pass the query, in its order, are the **row list**: a `Vec<u32>`
of point indexes the panel keeps between frames. It is computed with
`query_points` over the cached table only when what it was computed from
changes: the node, its document serial, the set of columns the cache holds,
the parsed bounds it applies, the image the selected-image box keeps and the
sort. Each frame compares those inputs with the last ones, which costs nothing
beside a query over a million rows, so a frame with an unchanged query does not
compute the rows again. A hover, a selection, a scroll or a repaint does not
move the inputs; selecting an image moves them only while the selected-image
box is ticked.

A row's values are read from the cached table by a binary search of its point
index in `TrackTable::point_indexes`, for the visible rows only. The
scroll-into-view of a selection made elsewhere searches the row list once, on
the frame the selection changed.

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `DEFAULT_COLUMNS` | Track, Images, Error (px), ∞ | The value columns shown before any is hidden or shown. |
| `POINT_WIDTH` | 72 pt | The Point column's width. |
| `MIN_VALUE_WIDTH` | 84 pt | The narrowest a value column is, room for its two bound boxes; a wider heading widens its column. |
| Row height | the body text height + 6 pt | Every row's height. |

## Testing

[track_table_panel/tests.rs](../../crates/sfm-explorer/src/track_table_panel/tests.rs)
runs the panel through real egui frames with no GPU (`Context::run_ui`) and the
table job through the fast queue on a real worker, over a demo reconstruction
with one point at infinity:

- the panel is the tenth tab, titled Track Table, wire name `track_table`, and
  the default layout puts it last in the centre node with the 3D Viewer
  raised;
- the panel's first frame queues the job, which fills the cache with every
  stored column the node can supply and no computed one; a column *Find
  points* already cached is kept, the same array, and the panel and
  `find_points` read one table; a hidden panel queues nothing;
- an edit while the job runs asks it to stop, nothing is merged for the serial
  the node left, and the next drive computes the new version's table; a table
  that lands after the cursor moved is dropped even when nothing cancelled the
  job; the job runs while a slow task holds the node; and its kernel stops when
  its cancel flag is set;
- a bound keeps the rows inside it, both ends inclusive, and a row with no
  value fails it; a box that is not a number sets no bound; sorting puts the
  no-value row last in either order;
- a frame with an unchanged query does not compute the row list again, through
  five frames, a hover and an image selection, and a change of the query, the
  version or the cached columns does;
- the selected-image box keeps the points that image observes and filters
  nothing with no image selected;
- a click on the Error heading sorts worst first, a second reverses, and a
  click on Point restores point order; a column holding a bound or the sort
  cannot be hidden; a column the reconstruction cannot supply is left out and
  its bound is said and filters nothing; the header counts matched rows against
  live points;
- a click on a row selects its point; hovering a row hovers its point and
  leaving clears it once; a selection made elsewhere scrolls its row into
  view, and a new sort scrolls back to the top;
- the cell text of a point at infinity, the grouping of counts and the
  significant digits.

The layout tests in `layout::tests` and the MCP layout and catalog tests count
the panel among the ten.

## Non-goals

- The picture columns, the Size slider, the computed columns, Filter the 3D
  view and `get_track_table_query` / `set_track_table_query` are not built.
  They are proposed in [drafts/track-table.md](../drafts/track-table.md).
- Multi-column sort. One column orders the rows; ties keep ascending point
  index.
- The query is not saved, in the layout file or anywhere else.

## Open questions

- **The quiet order.** Whether the table's columns should go ahead of the
  focused item's evaluation while the table is the panel in view.
- **Errors in files written before #1000** hold the mean rather than the RMS
  (or, for a file converted from COLMAP and never recomputed, COLMAP's value).
  Whether the open should recompute them, or the table should say when a file's
  errors predate the change, is undecided.
- **Does the query persist?** It could be kept in the layout file, or
  per-session only, as it is now.
- **`lowest_zncc`'s source.** The column reads `observation_confidence`. The
  observation readings carry `blur_matched_bitmap_zncc` too, which may be the
  better reading where both exist.
