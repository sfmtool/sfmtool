# Track Table

**Status:** Draft. Amends [gui/mcp-server.md](../gui/mcp-server.md) (the
panel-query tools),
[gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md) (the
non-goal "Point colouring and filtering by quality") and
[gui/panel-layout.md](../gui/panel-layout.md) (a tenth panel) and
[gui/background-tasks.md](../gui/background-tasks.md) § "Two queues" and
§ "Quiet tasks" (two quiet tasks).
Part 1, the track table and its query in core, is built and
filed as [core/analysis/track-table.md](../core/analysis/track-table.md).
The `find_points` read, the track table cache it reads, `TrackTable::merge`
and the asked-for *Find points* operation that fills the cache are built and
specified in [gui/mcp-server.md](../gui/mcp-server.md) § "`find_points`".
Parts 2 to 4 will file as `gui/track-table.md`, with the wire tools also
described in `gui/mcp-server.md`.

Decided:

- The panel is titled **Track Table**, wire name `track_table`. It pairs with
  Track View, which shows one track; the table lists all of them.
- The columns of the first version are the cheap ones read straight off the
  stored data, plus three computed ones: the maximum ray angle, the condition
  number and the inverse-depth z. The columns are computed by a read-only
  quiet task on the fast background queue
  ([gui/background-tasks.md](../gui/background-tasks.md) § "Quiet tasks");
  the table starts no thread of its own.
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
- The panel's home is the centre group, as a tab beside the 3D Viewer and
  Image Detail, where it has the height for rows of pictures (§ "Placement").
- The error column is `rms_error_px`, the stored
  `Point3D::rms_reprojection_error` (an RMS since #1000), which `get_point`
  also reports as `rms_error_px`. `max_ray_angle_deg` is the largest pairwise
  ray angle everywhere, since `inspect_point`'s angle from the mean direction
  was renamed `max_angle_from_mean_deg` (#1001).

Not decided: see § "Open questions".

## Purpose

A reconstruction can hold anywhere from a few hundred to a few million 3D
points. Each point comes with numbers that say how well it is supported: how
many photographs see it, how far its observations land from where it
projects, how wide the angle between its viewing rays is, and how well its
patch matches across views. The Track Table lists every point of a
reconstruction as one row, a track keyed by its point index, with two pictures of the point first (its
reference photograph cut around it, and its patch) and those numbers as the
columns after them. A person can sort by
any column, set a minimum and a maximum on each, and read off how many points
pass. Clicking a row selects that point everywhere else in the viewer. An
agent driving the viewer runs the same query through `find_points` and gets
back point ids with the same numbers, a page at a time.

The questions this answers are the ones a person asks while judging a solve:
"what are the worst points?", "how many points have a track of two?", "which
points does this image see, and which of them are badly triangulated?".
For a person each of these needs a script against the `.sfmr` file. An agent
asks `find_points`, which is built
([gui/mcp-server.md](../gui/mcp-server.md) § "`find_points`"); before it, the
2026-10-05 bug bash needed 9,941 `get_point` calls, taking 2 min 11 s, to find
three example tracks
([reports/2026-10-05-gui-bugbash.md](../../reports/2026-10-05-gui-bugbash.md)
finding 6).

The design has three layers, and each is useful without the ones above it:

1. **The track table** in `sfmtool-core`: named per-point values computed on
   demand for every live point of a reconstruction, one row per track keyed
   by point index and never stored, and a query that filters, sorts and pages
   them.
2. **The Track Table panel**, which shows the columns, edits a query and
   follows the selection.
3. **The wire**: `find_points`, which runs a query and returns the matched
   points (built), and a read/write pair for the panel's own query.

## Part 1: the track table and its query (core)

Built. The columns, their definitions, `TrackTable::compute`, `query_points`
and the `sfmtool.analysis.track_table` binding are specified in
[core/analysis/track-table.md](../core/analysis/track-table.md), and the
column words are in [GLOSSARY.md](../GLOSSARY.md) § "Track table". Parts 2
and 3 below use its wire names (`track_length`, `rms_error_px`,
`max_ray_angle_deg`, …), its "absent" and "no value" rules, and its sort rule
that rows with no value go last in either order. The error column is
`rms_error_px`, the same name the viewer's `get_point` reply gives the same
number once #1000 renames that field from `error`, so the table, `find_points`
and `get_point` use one name for it.

## Part 2: the Track Table panel

### Placement

`Tab::TrackTable`, titled **Track Table**, wire name `track_table`. Its state
is `track_table_panel::TrackTablePanel`, after `edit_history_panel`, since
`TrackTable` is the core table it shows. Its home is
the centre group, as a third tab beside the 3D Viewer and Image Detail
(`Home::Root`'s group). That is the largest area of the window: at 1280 × 720
with the stock layout it is about 690 × 470 px, enough for six or seven rows of
64 pt pictures under the heading and bounds rows. The bottom group, the first
choice, gives the table body about 100 px at that size, which is one picture
row. In `Layout::default()` it is the last tab of that group, after the 3D
Viewer and Image Detail, and the 3D Viewer stays the raised tab, so the default
layout opens showing the same panels as it does now and the table sits behind
both until its tab is clicked, or until `show_panel {"panel_name":
"track_table"}` raises it over MCP. Its Panels menu entry is ticked from the
start, since the panel is open; clicking it closes the table, as for any open
panel.

Sharing a group with the 3D Viewer means the two are not both visible until a
person drags the table out beside it. That matters for § "Filtering the 3D
view", whose effect is in the 3D Viewer: turning the filter on while the table
covers the 3D Viewer raises nothing, and the HUD's *(N shown)* says the filter
is on when the 3D Viewer is raised again.

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
  sorts the bad end first where the column has one (`rms_error_px`,
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
([gui/background-tasks.md](../gui/background-tasks.md) § "Quiet tasks"),
*Track Table reference picks*, which works through the visible rows one pick at a time
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

The panel reads the **track table cache** that `find_points` reads: one core
`TrackTable` per node, for the version at its cursor, keyed by
`(ReconId, document_serial)` ([gui/mcp-server.md](../gui/mcp-server.md)
§ "`find_points`"). The panel's columns are computed by the **table job**, a quiet, read-only task on the fast
queue ([gui/background-tasks.md](../gui/background-tasks.md) § "Quiet tasks"),
*Track Table columns*. It locks no node and writes no Action Log row, so a
person can sort and filter the version on screen while a slow task such as a
bundle adjustment holds the node. When that task installs its version, the
serial changes and the table is computed again. A result for an older serial
is dropped, and an edit cancels the running job through its `Progress`.

Each run of the job computes a `TrackTable` of only the columns the cache does
not hold yet, with `TrackTable::compute`, and the UI thread merges it into the
cached table when it lands, with `TrackTable::merge`, as *Find points* does
([core/analysis/track-table.md](../core/analysis/track-table.md) § "Merging
tables"). The job runs the stored columns first and the
computed ones as a second run, so the table is usable while the slow columns
are still being computed.

**The two quiet tasks.** *Track Table columns* and *Track Table reference
picks* each declare the fast queue, no lock and `quiet`, so they join
`Operation::ALL` and the test that holds each operation's declared values to
the table in [gui/background-tasks.md](../gui/background-tasks.md) § "Two
queues". Each is a source of its own (`QuietSource`), with one waiting entry.
They lock nothing because they push no version: they read the version at the
cursor and keep their answer in the table, so they run on a node a slow task
holds, against the version on screen, which is the value the slow task was
handed. When the slow task lands its version the document serial moves, and
each drops its stale answer and is queued again, as each does after an edit.
Among quiet tasks (`QuietRank`) they go after the focused item's evaluation,
the columns ahead of the picks, and ahead of the evaluations of the other bench
items, so the order is the camera model proposal's job, the viewed track's
evaluation, the focused item's, the table's columns, the table's picks, and
then the other bench items' evaluations.

The job starts when the panel is visible. Computed columns are computed only
once a visible column, a bound or the sort needs them. A hidden panel costs
nothing. A `find_points` call does not start the quiet job; it queues *Find
points* ([gui/mcp-server.md](../gui/mcp-server.md) § "`find_points`"), which
fills the same cache. A running quiet table job is cancelled for it and queued
again behind it, as for any asked-for task. If the two compute the same
columns at once the work is done twice, and the merge keeps the column the
cache already holds.

**Progress.** `TrackTable::compute` reports the stored columns and then the
computed ones, each with its own share of the job's range, so its fraction
follows the work. The reference picks decode photographs, so each pick gives
the decode a share of its range and the rule the rest (`DECODE_AND_STEP`, as
the bench steps do). Handing a decode the whole range is what made Refit all
points show a full bar from its first second.

**Waiting behind fast work.** Both are quiet, so they wait behind any asked-for
fast-queue task: an Open that renders display bitmaps, a Find nearby tracks,
a bench Fit. While one runs, a column or a crop that is not ready yet stays
`…` or an empty frame, as the live evaluation does in Track View.

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

Built, and specified in [gui/mcp-server.md](../gui/mcp-server.md)
§ "`find_points`", with the track table cache it reads and the *Find points*
operation that fills it. It runs `query_points` over the same cache the panel
will read, so a `find_points` call and the panel with the same bounds agree.
What later steps add to it:

- **The picture columns.** `find_points` refuses `crop` and `patch` today as
  columns it does not know. Once the panel has them, the refusal says that
  picture columns belong to the panel only.
- **The panel's greyed bound** uses the wording of `find_points`' refusal of a
  column the reconstruction cannot supply.

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

Each step is one PR through `implement-change`. Every step that changes the
panel ends with a capture run: a release build driven over MCP
(`skills/gui-bug-bash`) on real reconstructions, with screenshots of the
panel at its default size checked before the step is done. In the background
queue work, those captures found nine problems that the tests had passed:
layout, contrast, a wrong progress fraction and a missing title line.

3. **The Track Table panel** with the stored columns, the Patch column, the
   Size slider, sorting, bounds, selection and hover, and the quiet table job
   (*Track Table columns*) for the stored columns. The panel's state is
   `track_table_panel::TrackTablePanel`. This step adds the
   tenth panel to the layout, in the centre group, its tests and its specs, and moves
   `stored_patch_image` out of Track View's private `patch` module into the
   module both panels use.
4. **The Crop column**, moving Track View's tile and crop helpers into a shared
   module and rewriting them to take values rather than a track, with
   requests that wait for the scroll to stop and reference picks for the
   visible rows that store none.
5. **The computed columns in the panel**, filled by the quiet table job.
6. **Filter the 3D view** (dim and hide).
7. **`get_track_table_query` / `set_track_table_query`.**

## Open questions

- **The quiet order.** Whether the Track Table's columns should go ahead of
  the focused item's evaluation when the table is the panel in view.
- **Errors in files written before #1000** hold the mean (or, for a file
  converted from COLMAP and never recomputed, COLMAP's value). Whether the
  open should recompute them, or the table should say when a file's errors
  predate the change, is undecided.
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
