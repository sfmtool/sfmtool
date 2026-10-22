# Track Table

**Status:** Draft. Amends [gui/track-table.md](../gui/track-table.md) (the
Crop picture column, the computed columns and Filter the 3D view),
[gui/mcp-server.md](../gui/mcp-server.md) (the panel-query tools and the
picture columns' refusal in `find_points`),
[gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md) (the
non-goal "Point colouring and filtering by quality") and
[gui/background-tasks.md](../gui/background-tasks.md) § "Quiet tasks" (a
fourth quiet task, the reference picks).
Part 1, the track table and its query in core, is built and filed as
[core/analysis/track-table.md](../core/analysis/track-table.md). The
`find_points` read, the track table cache it reads, `TrackTable::merge` and
the asked-for *Find points* operation that fills the cache are built and
specified in [gui/mcp-server.md](../gui/mcp-server.md) § "`find_points`". The
Track Table panel with the stored columns, sorting, bounds, selection and
hover, and the quiet *Track Table columns* job that fills the cache with the
stored columns, is built and filed as [gui/track-table.md](../gui/track-table.md),
and so are its Patch picture column, the Size slider and the texture cache
the picture cells draw from ([gui/track-table.md](../gui/track-table.md)
§ "The Patch column"). This draft holds what remains.

Decided:

- The table is a visual tool first. Its first two columns after the point id
  are pictures: **Crop**, the reference photograph cut around the patch, and
  **Patch**, the point's patch bitmap. Both are shown by default, so a
  person sorting by a number sees what each point looks like next to it.
- The computed columns of the first version are the maximum ray angle, the
  condition number and the inverse-depth z. The quiet table job computes them
  as a second run after the stored columns
  ([gui/track-table.md](../gui/track-table.md) § "Computing the columns").
- The table can reach the 3D viewer through an optional toggle that dims or
  hides the points its filters reject.
- Bulk edits on the matched rows are designed for but not built:
  "Remove matching points" is named as a later step, and the query's result
  is shaped so that it can feed `delete_points`.

Not decided: see § "Open questions".

## Purpose

The Track Table panel ([gui/track-table.md](../gui/track-table.md)) lists
every point of a reconstruction as one row, with the stored per-point numbers
as its columns, sorted and filtered by a person. This draft adds what each
point looks like (its reference photograph cut around its patch, and the patch
itself) as the first two columns, the three computed columns that say how well
a point is triangulated, a toggle that applies the table's filter to the 3D
view, and a pair of wire tools that read and write the panel's query, so an
agent can set up the view a person sees.

## Part 2: what the panel adds

### What it shows

```
┌ Track Table ─ dino ───────────────────────────────────────────────────────────────────┐
│ 1,204 of 9,941 points  [Columns]  Size [──●──] ☐ Only points seen in selected image   │
│ ☐ Filter the 3D view: (•) Dim ( ) Hide                                                │
├────────┬────────┬────────┬─────────┬─────────┬────────────┬───────────────┬───────────┤
│ Point  │ Crop   │ Patch  │ Track ▼ │Distance │ Error (px) │ Ray angle (°) │ Cond.     │
│        │        │        │ [3 ][  ]│ [  ][  ]│ [   ][1.5 ]│ [2  ][    ]   │ [  ][  ]  │
├────────┼────────┼────────┼─────────┼─────────┼────────────┼───────────────┼───────────┤
│ #9873  │ ░▓▓▓▓░ │ ▓▓▓▓▓▓ │    57   │   4.17  │    0.820   │     41.2      │    3.1    │
│        │ ░▓▓▓▓░ │ ▓▓▓▓▓▓ │         │         │            │               │           │
├────────┼────────┼────────┼─────────┼─────────┼────────────┼───────────────┼───────────┤
│ #830   │ ░░░░░░ │ ▓▓▓▓▓▓ │    18   │     ∞   │    0.610   │     22.7      │    …      │
│        │ ░░░░░░ │ ▓▓▓▓▓▓ │         │         │            │               │           │
└────────┴────────┴────────┴─────────┴─────────┴────────────┴───────────────┴───────────┘
```

The second row's crop is still an empty frame because its photograph is
still being decoded.

- **The default columns** become Point, Crop, Patch, Track, Error, Distance
  and Ray angle, with Repeat obs. shown by itself as it is now. The Columns
  menu lists the picture columns and the computed columns beside the stored
  ones.
- **The heading rule** for the computed columns: a first click on
  `condition_number` or `inverse_depth_z` sorts descending, and on
  `max_ray_angle_deg` ascending, the bad end first. The panel's sort rule for
  `TrackTableColumn` already says so.
- **The rows** keep a uniform height for `ScrollArea::show_rows`, set by the
  Size slider while a picture column is shown, as they are for the Patch
  column now ([gui/track-table.md](../gui/track-table.md) § "The Patch
  column").
- **The query** gains the 3D toggle (§ "Filtering the 3D view") as panel
  state, beside the bounds, the sort, the visible columns and the selected-image
  box.

### Picture columns

The two picture columns show what each point looks like in the
photographs, which a column of numbers does not. Crop comes before Patch, as
in Track View, so the photograph as it is comes before the picture made from
it. The Patch column is built: what it shows, when it is offered, the Size
slider, its hover and how its pictures are cached are in
[gui/track-table.md](../gui/track-table.md) § "The Patch column". Crop joins
it among the pinned columns, between Point and Patch.

| Column | Shows | Absent when |
|---|---|---|
| **Crop** (`crop`) | the reference photograph cut around the patch's outline, with the outline drawn over it, by Track View's crop rule ([track-view.md](../gui/track-view.md) § "The observation table", the paragraph "The crop beside the tile shows the patch as the photograph holds it"): the patch frame at the reference observation's keypoint, its boundary projected through the camera's own model, the bounding box widened by one pixel and made square, at most 128 texels a side. Track View strokes the outline in the row's verdict colour; the table has no verdict and strokes it white over a wider dark stroke, the way Track View draws its marks. The photograph is the point's reference observation (§ "The reference photograph") | the reconstruction has no patch frames or no stored keypoints, the usual case for a `sift_files` reconstruction |

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
- **The pictures set the row height.** The Size slider built with the Patch
  column sets the Crop pictures' size too, and the row height while either
  picture column is shown.
- **Hovering the crop** shows Track View's crop hover view: the crop centred
  in three times its width and height of the photograph, with the outline, a
  dot at the keypoint, the amber ring where the point projects, and the
  photograph's name above it. **Hovering the patch** of a point that stores no
  reference shows the text in § "The reference photograph" in place of the
  stored-patch text the Patch cell's hover shows now. A click on a picture is
  the row's click, as it is on a Patch cell.
- **Crop cannot be sorted or bounded,** as Patch cannot.
- **Each cell says why it is empty.** A Crop cell is drawn from a
  `CellPicture`, as a Patch cell is, with further cases, and is one of:
  - the picture;
  - an empty frame while its photograph is decoding or its reference is being
    picked;
  - *unreadable*, when the photograph cannot be read
    (`DisplayPhotograph::Unreadable`);
  - *no reference*, when the rule picked none;
  - *off image*, when no point of the outline projects into the photograph
    (`crop::region` gives no region). An outline that projects but lies
    outside the photograph still gives a crop, drawn transparent where it is
    off the photograph, as in Track View.
- **Only visible rows are drawn.** Crop textures go in the texture cache the
  Patch cells use (`PictureTextures`, keyed by column, node, point and
  document serial and bounded by entry count). Crops need their photograph
  decoded. The table asks the shared photograph cache
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

### Computing the computed columns

The quiet table job, *Track Table columns*, computes the stored columns
([gui/track-table.md](../gui/track-table.md) § "Computing the columns"). It
runs the computed ones as a second run, so the table is usable while the slow
columns are still being computed. Computed columns are computed only once a
visible column, a bound or the sort needs them.

**The reference picks** are a second quiet task of the panel, *Track Table
reference picks* (§ "The reference photograph"). It declares the fast queue,
no lock and `quiet`, joins `Operation::ALL` and the test that holds each
operation's declared values to the table in
[gui/background-tasks.md](../gui/background-tasks.md) § "Two queues", and is a
source of its own (`QuietSource`), with one waiting entry. Like the table job
it does not wait for its node (`QuietSource::waits_for_its_node`): it reads
the version at the cursor and keeps its answer in the table, so it runs on a
node a slow task holds. Among quiet tasks (`QuietRank`) it goes after the
table's columns and ahead of the other bench items' evaluations, so the order
is the camera model proposal's job, the viewed track's evaluation, the focused
item's, the table's columns, the table's picks, and then the other bench items'
evaluations.

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
rather than ordering the rows by partial values. The panel today leaves a
bound or the sort on a column the cache does not hold yet unapplied, applies
the rest of the query, and says *Filtering and sorting when … is ready*; for
the stored columns, which land together, nobody sees the difference.

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
reads, so a `find_points` call and the panel with the same bounds agree.
What later steps add to it:

- **The picture columns.** `find_points` refuses `crop` and `patch` today as
  columns it does not know. Once the panel has them, the refusal says that
  picture columns belong to the panel only.
- **The panel's note for a bound it cannot apply**, on a column the
  reconstruction cannot supply, takes the wording of `find_points`' refusal of
  such a column, which lists the columns the reconstruction can supply.

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

Step 3a, the panel with the stored columns, sorting, bounds, selection and
hover and the quiet table job, and step 3b, the Patch column, the Size slider
and `stored_patch_image` moved into `patch_pictures`, a module both panels
use, are built ([gui/track-table.md](../gui/track-table.md)).

- **4. The Crop column**, moving Track View's tile and crop helpers into
  `patch_pictures`, beside `stored_patch_image`, and rewriting them to take
  values rather than a track, with requests that wait for the scroll to stop
  and reference picks for the visible rows that store none.
- **5. The computed columns in the panel**, filled by the quiet table job.
- **6. Filter the 3D view** (dim and hide).
- **7. `get_track_table_query` / `set_track_table_query`.**

## Open questions

- **Does Image Detail follow the 3D filter?** Dimming the tracked features of
  rejected points in Image Detail would extend the same toggle to a second
  panel. This draft limits the toggle to the 3D viewer.
