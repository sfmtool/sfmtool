# Track Table

The Track Table is the SfM Explorer panel that lists every 3D point of the
selected reconstruction as one row, with the numbers that say how well the
point is supported as its columns: how many observations its track has,
whether one photograph holds more than one of them, its reprojection error,
where it is and how far from the cameras (`∞` for a point at infinity), and
what its stored patch readings say, and two pictures of what each point looks
like: its reference photograph cut around its patch, and its patch bitmap. A
person sorts by any
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
reconstruction, and two picture columns: **Crop**, the point's reference
photograph cut around its patch (§ "The Crop column"), and **Patch**, its patch
bitmap (§ "The Patch column"). The computed columns (the largest ray angle, the
condition number and the inverse-depth z), applying the filter to the 3D view
and the wire tools that read and write the panel's query are proposed in
[drafts/track-table.md](../drafts/track-table.md).

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
┌ Track Table ───────────────────────────────────────────────────────────────────────┐
│ dino  1,204 of 9,941 points  · 3 with repeat observations  [Columns]               │
│   Size [──●──] 64 pt                                                               │
│ ☐ Only points seen in IMG_0042.jpg                                                 │
├────────┬────────┬────────┬─────────┬─────────────┬────────────┬──────────┬────────┤
│ Point  │ Crop   │ Patch  │ Track ▼ │ Repeat obs. │ Error (px) │ Distance │        │
│        │        │        │[3 ][   ]│[    ][    ] │[   ][1.5]  │[   ][   ]│        │
├────────┼────────┼────────┼─────────┼─────────────┼────────────┼──────────┼────────┤
│ #9873  │ ░▓▓▓▓░ │ ▓▓▓▓▓▓ │      57 │           2 │      0.820 │     4.17 │        │
│        │ ░▓▓▓▓░ │ ▓▓▓▓▓▓ │         │             │            │          │        │
├────────┼────────┼────────┼─────────┼─────────────┼────────────┼──────────┼────────┤
│ #830   │ ┌────┐ │ ▓▓▓▓▓▓ │      18 │           0 │      0.610 │        ∞ │        │
│        │ └────┘ │ ▓▓▓▓▓▓ │         │             │            │          │        │
└────────┴────────┴────────┴─────────┴─────────────┴────────────┴──────────┴────────┘
```

The second row's crop is an empty frame because its photograph is still being
decoded.

- **It follows the selected reconstruction**, as Image Browser and Edit History
  do. The header names that reconstruction and counts the matched rows against
  its live points, with the thousands grouped (*1,204 of 9,941 points*). When
  any live point has repeat observations, the count goes on with how many do,
  whatever the bounds (*· 3 with repeat observations*), since such a point is a
  defect to find and fix; with none it says nothing of them. With no
  reconstruction loaded the panel says *No reconstruction loaded*. The first
  time the panel shows a reconstruction, until the table job has computed the
  columns for the version at the cursor, the panel says *Computing the track
  table of {label}…* beside a spinner. After an edit it keeps the rows on show
  instead (§ "After an edit").
- **The columns** are the Point column, which is always shown and is the row's
  key, `#` and the point index, the Crop and Patch picture columns where the
  reconstruction can supply them (§ "The Crop column", § "The Patch column"),
  and the stored track table columns the reconstruction can supply. A column the reconstruction cannot supply
  (`TrackTableColumn::is_available` false, such as `normal_confidence` on a
  file with no such stored column) is left out: it has no heading, no bounds
  and no entry in the Columns menu.

  | Heading | Column | Cell |
  |---|---|---|
  | Point | the point index | `#9873` |
  | Track | `track_length` | a whole number |
  | Repeat obs. | `repeat_observations` | a whole number, `0` for a sound track |
  | Error (px) | `rms_error_px` | three significant digits, at most three decimals |
  | X, Y, Z | `x`, `y`, `z` | four significant digits, every whole digit kept, at most three decimals |
  | Distance | `centroid_distance` | three significant digits, at most three decimals; `∞` for a point at infinity |
  | Normal conf. | `normal_confidence` | a whole number, 0 to 255 |
  | Lowest ZNCC | `lowest_zncc` | two decimals |
  | Self-sim. radius | `largest_zncc_self_similarity_radius` | three significant digits, at most three decimals |

  A point with no value in a column (a NaN in the table: a point at infinity's
  `x`) shows `-`. A column the cache does not hold yet shows `…`. The number
  cells are right-aligned. There is no column that flags a point at infinity:
  its Distance is `∞`, so a sort by Distance gathers the points at infinity at
  one end and a minimum of `inf` keeps only them.
- **Columns**, a menu, lists every column the reconstruction can supply with a
  checkbox, the picture columns first, Crop then Patch, as they are drawn. The
  default set is Crop, Patch, Track, Error and Distance. **Repeat obs. shows
  itself**: until the person ticks or unticks it in this menu, it is shown
  exactly while the reconstruction on show has a point with repeat
  observations, so a defect is not hidden behind a column nobody thought to
  open, and a sound reconstruction does not spend a column on zeros. Once it
  has been ticked or unticked, the menu alone decides, for every
  reconstruction, so the panel never undoes a choice the person made. It is
  not added to the person's columns while it shows itself
  (`TrackTableQuery::is_shown`, `repeat_observations_chosen`), and while it
  holds a bound or the sort it stays shown whatever the reconstruction. A column that holds
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
  column sorts the bad end first where the column has one:
  `repeat_observations`, `rms_error_px` and
  `largest_zncc_self_similarity_radius` descending, `track_length`,
  `lowest_zncc` and `normal_confidence` ascending. Every other column sorts
  ascending first, Distance among them, which puts the points at infinity, at
  `∞`, after every finite distance; a second click puts them first. A second
  click on the same heading reverses the order. A triangle after the heading marks the sorted column, pointing up for
  increasing and down for decreasing. Rows with no value go last in either
  order, which is the track table query's rule. A click on **Point** puts the
  rows back in increasing point index, the order with no sort. The click rule
  is Track View's (`crate::table_sort`, which both tables call). Each heading's
  hover text says what the column holds, its wire name and what a click on it
  does.
- **The bounds row**, under the headings, has a minimum box and a maximum box
  per column. A row passes when its value is at least the minimum and at most
  the maximum, both inclusive, and every bound must pass; a row with no value
  fails any bound that sets an end. An empty box sets no bound. `inf` is a
  number here: a minimum of `inf` on Distance keeps only the points at
  infinity, and any finite maximum drops them. A box whose
  text is not a number sets none either, and its text is drawn in the error
  colour with a hover text saying so. The rows are filtered again on the frame
  after an edit of a box. Each box is named for its column, *minimum Error
  (px)*, so `get_widgets` and a screen reader say which box it is.
- **The rows** are virtualised with `ScrollArea::show_rows`, as the Action
  Log is, so a million rows cost only the visible ones per frame. `show_rows`
  needs one height for every row: the picture size and 2 pt above and below
  while a picture column is shown, and one text line's height while none is
  (§ "The Patch column"). Alternate rows are shaded. The table scrolls sideways when its columns
  are wider than the panel, with the headings and the bounds moving with the
  rows; they stay put while the rows scroll up and down. The Point column is
  pinned: its heading and its cells stay at the panel's left edge while the
  value columns scroll sideways and pass under it, so every row keeps its
  point index on screen. The picture columns are pinned with it (§ "The Patch
  column"). The rows scroll up and down as one, the pinned
  cells with them, and a row's click and hover cover the whole row, its
  Point cell included. A new sort, new bounds,
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

## The Patch column

The **Patch** column (`patch` in the query, `PictureColumn::Patch`) is a
picture column: it shows what each point looks like, which a column of numbers
does not, so a person sorting by a number sees each point's patch next to it.
The column, the *Size* slider and the texture cache are in
[track_table_panel/pictures.rs](../../crates/sfm-explorer/src/track_table_panel/pictures.rs).

- **What it shows** is the point's patch bitmap (`PointView::patch_bitmap`):
  the bitmap the file stores, or the display patch bitmap the open renders for
  a file with patch frames and no bitmaps. It is the picture in Track View's
  header slot for a viewed point, converted by the same function,
  `stored_patch_image` in
  [patch_pictures.rs](../../crates/sfm-explorer/src/patch_pictures.rs)
  ([track-view.md](track-view.md) § "The header"), and drawn with
  nearest filtering. It is not Track View's per-row *Patch* tile, which is
  rendered again from each observation's own photograph. A point whose bitmap
  is all zero, which is how a point with no stored patch is written, shows
  *no patch*.
- **It is offered only when the reconstruction's points carry a patch
  bitmap** (`EditedReconstruction::has_patch_bitmaps`), stored or rendered by
  the open for display. A reconstruction with neither, the usual case for a
  `sift_files` one, has no Patch column, no entry for it in the Columns menu
  and no *Size* slider, and its rows are text high; the column appears after
  Convert to Embedded Patches, whose open renders display bitmaps.
- **It is shown by default, after the Crop column.** It is drawn after the
  Point column and the Crop column, as in Track View the photograph comes
  before the picture made from it, and is pinned with them: while the value
  columns scroll sideways they pass under the pinned columns, so a row keeps
  its point index and its pictures on screen. The Columns menu lists it after
  Crop and can hide it.
- **It cannot be sorted or bounded.** Its heading takes no click and it has no
  bound boxes. Its heading's hover text says what the column shows and that
  it cannot be sorted or bounded.
- **Size.** A slider in the header, *Size*, sets the picture size from 32 to
  128 pt, 64 by default, in whole points, for both picture columns. Every row
  is the picture size and 2 pt above and below high while a picture column is
  shown, so `show_rows` keeps rows of one height; at 32 pt a row is still
  taller than a line of text. With both hidden the slider is greyed and the
  rows are text high. The picture is centred in a cell the picture size and 6 pt either
  side wide, or wider where the heading needs it. When the row height changes,
  by the slider or by showing or hiding the column, the vertical scroll
  offset is scaled by the new height over the old, so the row at the top stays
  at the top, unless the rows end first. The size is panel state like the
  query: it is not saved. A reconstruction that can supply neither picture
  column has no slider.
- **Hovering a picture** shows it at three times the picture size, at most
  256 pt so it stays inside a small window, with the text Track View's header
  slot gives a viewed point's stored patch (`STORED_PATCH_HOVER`) and the
  point's portable id. Over *no patch* the hover says the point has no stored
  patch. A click on a picture is the row's click.
- **A point that stores no reference** says so in its hover, where the
  reconstruction has a Crop column: the point stores no reference observation,
  so its bitmap is not the render of the photograph in the Crop cell, which is
  the reference-view rule's pick (`UNREFERENCED_PATCH_HOVER`,
  § "The reference photograph"). That is a point at `-1` in a file with stored
  bitmaps, or in a file with no `reference_observations` column, whose bitmap
  is an older file's mean of its views or the render of an observation since
  removed. A point the open's display render picked for has a reference, its
  display-only pick, and its hover is the stored-patch text.

### Drawing the pictures

- **Only the rows drawn make pictures.** A cell's picture is made, from the
  bitmap already in memory, and uploaded when its row is first drawn, so only
  for the rows `show_rows` hands the frame. Nothing is made for a row that is
  not on screen, and a frame costs work in the visible rows only.
- **The textures are cached, bounded by entry count.** Each picture is kept in
  a texture cache (`PictureTextures`) keyed by the column, the node, the point
  and the document serial of the version the rows belong to (`PictureKey`), so
  scrolling back over rows already seen uploads nothing. The cache holds up to
  1024 pictures. Once a frame leaves it holding more, it releases the pictures
  drawn longest ago, down to 768, so the pictures of rows that left the view
  stay until the cache is full and then go first. The sort that finds them is
  paid once per 256 new pictures rather than every frame, and a picture the
  frame drew is never released.
- **Out-of-date rows make none.** While the rows are of the version before an
  edit (§ "After an edit") the cells draw the pictures already made for that
  version, at half brightness as their text is dimmed, and a row that had no
  picture made for it draws an empty frame. Nothing is made for them: a row of
  the version before may name a point the edit deleted or changed, and the
  value at the cursor is the new version's. When the new rows arrive their
  pictures are made for the new serial.
- **Shared with the Crop column.** The cache key names the column, and a cell
  is a picture, an empty frame or the reason it has none (`CellPicture`), so
  the Crop column uses the same cache and the same cell drawing
  (§ "The Crop column").

## The Crop column

The **Crop** column (`crop` in the query, `PictureColumn::Crop`) shows the
point's reference photograph as it is, cut around the outline its patch has
there, so a person sorting by a number sees the pixels the point stands for
beside the patch made from them. It is in
[track_table_panel.rs](../../crates/sfm-explorer/src/track_table_panel.rs)
(`RowsView::crop_picture`), and cuts the crop with the functions Track View
cuts a row's crop with,
[patch_pictures/crop.rs](../../crates/sfm-explorer/src/patch_pictures/crop.rs)
([track-view.md](track-view.md) § "The observation table").

- **What it shows** is Track View's crop rule applied to the point's reference
  observation: the patch frame (with the point's `w`) re-anchored at that
  observation's keypoint, its boundary projected through the camera's own
  model, the bounding box of the outline widened by one photograph pixel and
  made square, read at most 128 texels a side. The outline is drawn over it.
  Track View strokes the outline in the row's verdict colour; the table has no
  verdict and strokes it white over a wider dark stroke, the way Track View
  draws the marks over its hover views. The crop takes values rather than a
  track: the patch frame, the camera and its pose, the keypoint and the point's
  position, all read off the row's `PointView` (`crop::patch_outline`), so the
  table builds no track per row. A point at infinity is cropped as any other:
  its patch frame carries `w = 0` and the outline and the projection are
  homogeneous.
- **It is offered only where the outline can be made**: where the points carry
  a patch frame and the observations a keypoint
  (`EditedReconstruction::has_patch_frames` and `has_keypoints`). A
  reconstruction without either, the usual case for a `sift_files` one, has no
  Crop column and no entry for it in the Columns menu. It appears after Convert
  to Embedded Patches.
- **It is shown by default and comes first** among the picture columns, right
  after the Point column, and is pinned with it. It shares the *Size* slider,
  the row height and the texture cache with the Patch column (§ "The Patch
  column").
- **It cannot be sorted or bounded.** Its heading takes no click and it has no
  bound boxes. A click on a crop is the row's click.
- **Each cell says why it is empty.** A Crop cell is drawn from a
  `CellPicture`, as a Patch cell is, and is one of:
  - the crop;
  - an empty frame while its photograph is decoding or its reference is being
    picked, or while the scroll has not yet stopped over it (below);
  - *unreadable*, when the photograph cannot be read
    (`DisplayPhotograph::Unreadable`);
  - *no reference*, when the point stores no reference observation and the
    reference-view rule picks none;
  - *off image*, when no sample of the outline projects into the photograph
    (`crop::region` gives no region), as for a patch behind the camera. An
    outline that projects but lies partly or wholly outside the photograph
    still gives a crop, transparent where it is off the photograph, as in
    Track View;
  - *no patch*, for a point whose patch half-vectors are zero, so it has no
    patch frame to outline.
- **Hovering a crop** shows Track View's crop hover view: the photograph's
  name, then the crop centred in three times its width and height of the
  photograph, with the outline, a dot at the keypoint, the amber ring where
  the point projects and the dashed line between them, and the bullets Track
  View gives under it, then the point's portable id. The wider picture is cut
  the first time the pointer rests on the cell and kept while it stays there
  (`PictureTextures::crop_context`). Over a cell with no crop the hover says
  why, in a sentence.

### The reference photograph

The photograph is the point's reference observation's
(`PointView::reference_observation`, which includes a pick that only the
open's display render made). A point that stores none, stored as `-1` or in a
file with no `reference_observations` column, gets one from the reference-view
rule, run at the point's stored keypoints: `stored_bitmap::render_reference`,
which makes the pick without keeping the bitmap, over the photographs of the
point's track that can be read and are the size their camera says, at the
reconstruction's patch resolution. It is the pick an evaluation would make
without localising the rows first. With fewer than two such views there is no
pick, as for the bitmap writers.

Which points reach the rule follows from whether the open rendered display
bitmaps (`PointSet::patch_bitmaps_for_display`):

- **A file with stored bitmaps** renders none at the open, so a point it stores
  at `-1` (a file older than `.sfmr` v12, one whose bitmaps are a mean of the
  views, or one whose point lost its reference observation) has had no pick
  made. Its crop waits for a pick, and its Patch cell's hover says the bitmap
  is not the render of the crop's photograph (§ "The Patch column").
- **A file the open rendered display bitmaps for** already ran the rule for
  every point it stores at `-1`, on the same keypoints and photographs, and
  kept every pick it made. A point it left at `-1` found no eligible view, and
  the same rule would pick nothing again, so its cell says *no reference* at
  once and nothing is queued for it.

The picks are a quiet task (§ "The reference picks"). Each is kept by
`(ReconId, point, document_serial)` in the panel's state on `AppState`
(`ReferencePicks`), not written into the version, so it costs no undo step and
does not reach a save. A commit from Track View is still the way to store a
reference.

### Requests wait for the scroll to stop

A crop needs its photograph decoded, and the table asks the shared photograph
cache for it (`state::display_photograph`, which starts the decode on the rayon
pool and repaints when it lands). Track View only looks in that cache, so the
table is a source of decodes of its own. A sort by a number puts neighbouring
rows in different photographs, so scrolling through a large table would queue a
decode, and a pick, for every row it passes.

So the cells ask for photographs and picks only once the rows on screen have
stopped changing: once the range of rows the frame draws has stayed the same
for `SCROLL_SETTLE_S`, 0.15 s (`ScrollMotion`). While it is still changing, a
cell draws the crop it already holds or an empty frame, looks in the photograph
cache without starting a decode, and asks for no pick; the panel asks for a
repaint at the moment the rows will count as stopped. The first frame that
draws rows counts as stopped, so a table opened or sorted shows its crops at
once. A new row count after an edit moves the rows like a scroll does.

The delay was measured on `dino_dog_toy` (85 photographs at 2040×1536, with its
bitmaps stored and half its references cleared so every second point needs a
pick), in a release build on the development machine. egui spreads one wheel
scroll over about 0.14 s of frames (an MCP `scroll` of 25 lines returned after
0.14 to 0.17 s), during which the rows change on nearly every frame. A cold
decode of one photograph through the cache took 15 to 17 ms, a pick whose track
photographs were not decoded 47 to 292 ms (3 to 18 views), and a pick whose
photographs were 1 to 12 ms. Twelve such scrolls in quick succession passed about 180 rows
sorted by error; the screenshot taken just after the last one shows the rows
it stopped on in empty frames, and the next, a few seconds later, their crops. 0.15 s is just over the time egui spreads one wheel scroll
over, so a wheel turned steadily or a scroll bar dragged never counts as
stopped, and it is shorter than one cold pick, so the wait adds less than the
work it saves.

**The cost is bounded by the photograph cache.** A page of rows from a metric
sort can touch most of a reconstruction's photographs, and a pick decodes every
photograph its track observes. The cache's budget is a quarter of physical
memory, clamped to 1–16 GiB (`photograph_cache::default_budget_bytes`). All of
`dino_dog_toy`, 85 photographs at 2040×1536 with their pyramids, takes about
1.07 GB decoded. A reconstruction with more photographs than the budget holds
evicts them and decodes them again as the table scrolls, which costs time but
no memory beyond the budget.

**Out-of-date rows ask for nothing.** While the rows are of the version before
an edit (§ "After an edit") a Crop cell draws the crop already made for that
version, dimmed, or an empty frame, as a Patch cell does; no photograph is
asked for and no pick is queued for a version the rows do not belong to.

### The reference picks

The picks are a quiet, read-only task on the fast queue, *Track Table reference
picks* (`Operation::TRACK_TABLE_REFERENCE_PICKS`;
[background-tasks.md](background-tasks.md) § "Quiet tasks"), in
[track_table_panel/picks.rs](../../crates/sfm-explorer/src/track_table_panel/picks.rs).

- **The panel names the points.** On each frame its rows have stopped changing
  and are of the version at the cursor, the panel names the points it drew
  whose crops wait for a pick (`AppState::want_reference_picks`), and the
  frame clears that before the dock draws, with the table job's node
  (`AppState::hide_track_table`). A hidden panel, or one being scrolled, names
  none. Twice per frame, after the table job's driver,
  `AppState::drive_track_table_picks` takes the picks that came back, forgets
  the picks of versions their nodes have left, and queues, updates or stops
  the task.
- **One task, one pick at a time.** The task takes the next point from a list
  it shares with the GUI thread, decodes the photographs of the point's track
  through the photograph cache, runs the rule, sends the pick back with a wake
  of the window, and takes the next. Each frame the driver replaces the list
  with the points named that have no pick, leaving out the one in hand, so a
  row that leaves the view is taken off it, and the task ends once the list is
  empty. Each pick gives the decode the first half of the task's range and the
  rule the second (`DECODE_AND_STEP`, as the bench steps do), and spans the
  whole range, since the list changes as the rows scroll; nothing draws a
  quiet task's fraction.
- **An edit cancels it.** When the document serial at its node's cursor moves
  while it runs, or the node is closed, it is asked to stop through its
  `Progress`, and a pick that comes back for a serial the node has left is
  dropped. The next task is queued once the rows are of the new version.
- **It locks nothing and does not wait for its node.** It pushes no version and
  keeps its picks keyed by the serial it read, so it runs while a slow task
  such as a bundle adjustment holds the node, against the version on screen
  (`QuietSource::TrackTableReferencePicks`, whose `waits_for_its_node` is
  false).
- **It is quiet.** It writes no Action Log row, is not reported on the wire,
  and gives way to any task queued in the fast queue, *Find points* or an Open
  among them; the driver queues it again behind that task with the points
  still to pick. Among quiet tasks it waits after the table's columns and
  ahead of the other bench items' evaluations (`QuietRank`).

## After an edit

An edit, an undo or a redo moves the document serial at the node's cursor, and
the table job computes the new version's table
(§ "Computing the columns"). On a large reconstruction that takes a while.
Until the new table lands the panel keeps showing the rows, the count, the
headings and the bound boxes it showed for the version before, and marks them
out of date: a spinner and *Updating…* in the header line, the rows' text
dimmed, and their pictures drawn at half brightness, with none made for them
and no photograph or pick asked for (§ "Drawing the pictures", § "The Crop
column"). When the new table lands the row list is computed from it and replaces
the old one in the same frame. It does not scroll back to the top unless the
query changed while the rows were out of date: a new sort, new bounds or a
change of the selected-image box set meanwhile scrolls to the top on landing,
as it does at any other time.

The old rows are read from the old version's table, which the track table
cache still holds: a node's entry is replaced only when a table for another
serial is merged (`TrackTableCache::latest` reads it whatever its serial).
The panel shows it only when its row list was computed from that table, for
the same node and serial; otherwise, as when another reconstruction was
selected meanwhile, it says *Computing…* as on a first open. `find_points` and
the job still read the cache only for the version at the cursor.

- **An out-of-date row cannot be selected or hovered.** A row of the version
  before may name a point the edit deleted, and `select_point` on it would
  select a point that no longer exists. Every out-of-date row is treated the
  same rather than only the deleted ones, since telling them apart means
  looking each one up in the new version: the rows take no click, set no
  hover and show no hover text, and a hover the panel owned is cleared once.
  The selected and hovered points of other panels are still drawn.
- **A change of the query waits.** The headings and the bound boxes stay
  live, so a sort or a bound can be set while the rows are out of date, but
  the row list is not computed again until the new table lands. Running the
  query over the old table would read the old version's values with the new
  version's observations for the selected-image box, which do not go
  together. A selection made elsewhere while the rows are out of date is
  scrolled into view once the new rows arrive.

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
  names the point's portable id, `pt3d_<hash>_<index>`. While the pointer is over
  the rows the panel owns the hover, and over no row it clears it; leaving the
  rows clears it once, so a hover set by another panel is left alone. A row
  another panel hovers is drawn with the hover fill, its text in the strong
  colour so it keeps its contrast over the lighter fill.

## The query is panel state

The bounds, the sort, the visible columns (the picture columns among them) and the
"seen in the selected image" box are UI state of the panel
(`TrackTableQuery`), like the Image Detail display settings. They are not held
in a version, cost no undo step and are not saved with the reconstruction or in
the layout file. A single query is used for whichever reconstruction is
selected. The picture size the *Size* slider sets is panel state beside the
query and is not saved either.

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
  View does for the viewed track. Twice per frame, before the MCP drain and again
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
  tasks it waits after the focused item's evaluation and ahead of the
  reference picks (§ "The reference picks") and the other bench items'
  evaluations.

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
| `DEFAULT_COLUMNS` | Track, Error (px), Distance | The value columns shown before any is hidden or shown. Repeat obs. is shown as well while the reconstruction has a point with repeat observations and the person has not ticked or unticked it. The Crop and Patch columns are shown by default too, where the reconstruction can supply them. |
| `POINT_WIDTH` | 72 pt | The Point column's width. |
| `MIN_VALUE_WIDTH` | 84 pt | The narrowest a value column is, room for its two bound boxes; a wider heading widens its column. |
| `MIN_PICTURE_SIZE`, `MAX_PICTURE_SIZE` | 32 pt, 128 pt | The *Size* slider's range. |
| `DEFAULT_PICTURE_SIZE` | 64 pt | The picture size before the slider moves. |
| `PICTURE_PAD` | 2 pt | Space above and below a picture in its row. |
| Row height | the picture size + 2 × `PICTURE_PAD` while a picture column is shown; the body text height + 6 pt while none is | Every row's height. |
| `HOVER_SCALE`, `MAX_HOVER_SIZE` | 3, 256 pt | A picture's hover view is three times the picture size, at most 256 pt. |
| `TEXTURE_CAPACITY` | 1024 | The pictures the texture cache holds before it releases the ones drawn longest ago, down to three quarters of it. Crops and patches count together. |
| `SCROLL_SETTLE_S` | 0.15 s | How long the rows on screen stay the same before the Crop cells ask for photographs and picks (§ "Requests wait for the scroll to stop"). |
| Crop outline | 1 pt light grey over 2.5 pt dark | The stroke over a crop in a cell (`CELL_OUTLINE`). |

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
- the default columns are Track, Error and Distance; the point at infinity's
  Distance is `∞`, last ascending and first descending, dropped by a finite
  maximum and kept alone by a minimum of `inf`;
- with two points holding repeat observations, Repeat obs. is shown without
  being added to the person's columns, the header says *· 2 with repeat
  observations* and the heading's first click puts those points first; an
  untick keeps it hidden though there are repeats, a tick keeps it shown
  though there are none, and while it holds the sort it is not hidden;
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
- after an edit the old rows, count and headings stay on show marked
  *Updating…*, a bound set meanwhile waits, and the new rows and count replace
  them when the job lands; an out-of-date row, a deleted point's or a
  surviving one's, takes no click and no hover, and a click selects again once
  the new table lands; the first open says *Computing…* with no count;
- scrolled sideways, the Point heading and the Point cells keep their x
  position while the value columns' headings move;
- the cell text of a point at infinity (`-` for X, `∞` for Distance), the
  grouping of counts and the significant digits.

[track_table_panel/tests/pictures.rs](../../crates/sfm-explorer/src/track_table_panel/tests/pictures.rs)
runs the Patch column over the same reconstruction given an 8 × 8 bitmap per
point, one of them all zero:

- a reconstruction with no bitmaps has no Patch column, no *Size* slider and
  text-high rows; with bitmaps the column is shown by default between Point
  and Track and the rows are the default size high, and hiding it brings text
  rows back;
- the pictures made are exactly one per drawn row, for the version at the
  cursor, a frame that draws the same rows makes none, and the all-zero row
  draws *no patch*;
- the texture cache never holds more than its capacity after a frame,
  releases down to three quarters of it, keeps a picture the frame drew and
  releases the one drawn longest ago, and a cell with nothing to make it from
  waits;
- each slider size sets the row height and keeps the top row at the top, and
  hiding the pictures keeps it on screen;
- after an edit, out-of-date rows make no pictures: a row drawn before keeps
  its picture and a row not drawn before waits, and the new version's
  pictures are made once its table lands;
- hovering a picture shows the stored-patch text, a click on it selects the
  row, and a click on the Patch heading does not sort;
- scrolled sideways, the Patch heading stays put with the Point column;
- the hover view is three times the size up to its cap, and the smallest row
  is taller than a line of text.

Track View's drawing of the same bitmap is held by
[patch_pictures/tests.rs](../../crates/sfm-explorer/src/patch_pictures/tests.rs)
and Track View's own header tests.

[track_table_panel/tests/crop.rs](../../crates/sfm-explorer/src/track_table_panel/tests/crop.rs)
runs the Crop column over an `embedded_patches` reconstruction of 60 points
with stored bitmaps, photographs in the cache, references stored for some
points and not others, a point with no patch frame, one behind its reference
camera and one at infinity:

- the column is offered only where the points carry patch frames and the
  observations keypoints, is shown by default and is drawn between Point and
  Patch; its heading does not sort;
- each cell state: a crop, a crop of the point at infinity, *no patch*, *off
  image*, an empty frame while its photograph decodes, then *unreadable* once
  the decode of a missing file fails;
- a point the display render left at `-1` says *no reference* at once and
  queues no pick; a point with stored bitmaps and no reference waits, is picked
  by the task, and draws the crop of the rule's pick, with no version pushed
  and the version's reference still `-1`;
- while the rows move under the wheel no photograph is asked for and no pick
  wanted or queued, and once they have stayed put for `SCROLL_SETTLE_S` the
  visible rows' photographs are asked for and exactly their picks wanted;
- an edit while the picks run asks them to stop and keeps no pick for the
  version left; no pick is wanted or queued while the rows are out of date,
  and the new version's are once its table lands;
- a task asked for in the fast queue asks the picks to stop and starts once
  they have, and the picks are queued again behind it; a slow task holding the
  node leaves them running and they finish beside it;
- a row that leaves the view leaves the running task's list, and with none
  wanted it picks nothing more;
- hovering a crop shows the photograph's name and Track View's crop caption;
  hovering the patch of an unreferenced point says its bitmap is not the
  render of the crop's photograph, and a referenced one gives the stored-patch
  text.

Track View's body tests hold that a crop cut from a point's values is the crop
Track View cuts from a track put on the bench from that point, pixel for pixel,
with the same region and outline. `background::tests` holds the picks'
declared queue, lock and quiet value to [background-tasks.md](background-tasks.md)
and their work to its claim to stop when cancelled; the MCP tests hold the
`find_points` refusal of the picture columns.

The layout tests in `layout::tests` and the MCP layout and catalog tests count
the panel among the ten.

## Non-goals

- The computed columns, Filter the 3D view and
  `get_track_table_query` / `set_track_table_query` are not built.
  They are proposed in [drafts/track-table.md](../drafts/track-table.md).
- The picture columns are not returned on the wire: `find_points` refuses
  `crop` and `patch` ([mcp-server.md](mcp-server.md) § "`find_points`"), and
  an agent sees them with `screenshot {"panel_name": "track_table"}`.
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
