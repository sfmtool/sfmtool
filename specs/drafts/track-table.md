# Track Table

**Status:** Draft. Amends [gui/track-table.md](../gui/track-table.md) (the
computed columns and Filter the 3D view),
[gui/mcp-server.md](../gui/mcp-server.md) (the panel-query tools) and
[gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md) (the
non-goal "Point colouring and filtering by quality").
Part 1, the track table and its query in core, is built and filed as
[core/analysis/track-table.md](../core/analysis/track-table.md). The
`find_points` read, the track table cache it reads, `TrackTable::merge` and
the asked-for *Find points* operation that fills the cache are built and
specified in [gui/mcp-server.md](../gui/mcp-server.md) § "`find_points`". The
Track Table panel with the stored columns, sorting, bounds, selection and
hover, and the quiet *Track Table columns* job that fills the cache with the
stored columns, is built and filed as [gui/track-table.md](../gui/track-table.md),
and so are its Patch picture column, the Size dropdown and the texture cache
the picture cells draw from ([gui/track-table.md](../gui/track-table.md)
§ "The Patch column"), and so are its Crop picture column, the reference picks
that give a point with no stored reference its photograph, and the
`find_points` refusal of the picture columns
([gui/track-table.md](../gui/track-table.md) § "The Crop column"). This draft
holds what remains.

Decided:

- The table is a visual tool first. Its first two columns after the point id
  are pictures, **Crop** and **Patch**, both shown by default (built,
  [gui/track-table.md](../gui/track-table.md) § "The Crop column").
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
and two pictures of each point as its columns, sorted and filtered by a person.
This draft adds the three computed columns that say how well a point is
triangulated, a toggle that applies the table's filter to the 3D view, and a
pair of wire tools that read and write the panel's query, so an agent can set
up the view a person sees.

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

- **The default columns** become Point, Crop, Patch, Track, Error, Distance
  and Ray angle, with Repeat obs. shown by itself as it is now. The Columns
  menu lists the computed columns beside the stored ones.
- **The heading rule** for the computed columns: a first click on
  `condition_number` or `inverse_depth_z` sorts descending, and on
  `max_ray_angle_deg` ascending, the bad end first. The panel's sort rule for
  `TrackTableColumn` already says so.
- **The query** gains the 3D toggle (§ "Filtering the 3D view") as panel
  state, beside the bounds, the sort, the visible columns and the selected-image
  box.

### Computing the computed columns

The quiet table job, *Track Table columns*, computes the stored columns
([gui/track-table.md](../gui/track-table.md) § "Computing the columns"). It
runs the computed ones as a second run, so the table is usable while the slow
columns are still being computed. Computed columns are computed only once a
visible column, a bound or the sort needs them.

**Progress.** `TrackTable::compute` reports the stored columns and then the
computed ones, each with its own share of the job's range, so its fraction
follows the work.

**Waiting behind fast work.** The job is quiet, so it waits behind any
asked-for fast-queue task: an Open that renders display bitmaps, a Find nearby
tracks, a bench Fit. While one runs, a column that is not ready yet stays `…`,
as the live evaluation does in Track View.

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

## Part 3: the wire

### `find_points` (read)

Built, and specified in [gui/mcp-server.md](../gui/mcp-server.md)
§ "`find_points`", with the track table cache it reads and the *Find points*
operation that fills it. It runs `query_points` over the same cache the panel
reads, so a `find_points` call and the panel with the same bounds agree.
Its refusal of the picture columns, `crop` and `patch`, is built. What later
steps add to it:

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
hover and the quiet table job, step 3b, the Patch column, the Size slider
(since replaced by the Size dropdown) and `stored_patch_image` moved into
`patch_pictures`, a module both panels use, and step 4, the Crop column with
the crop moved into `patch_pictures` and made to take values, the requests
that wait for the scroll to stop and the reference picks, are built ([gui/track-table.md](../gui/track-table.md)).

- **5. The computed columns in the panel**, filled by the quiet table job.
- **6. Filter the 3D view** (dim and hide).
- **7. `get_track_table_query` / `set_track_table_query`.**

## Open questions

- **Does Image Detail follow the 3D filter?** Dimming the tracked features of
  rejected points in Image Detail would extend the same toggle to a second
  panel. This draft limits the toggle to the 3D viewer.
