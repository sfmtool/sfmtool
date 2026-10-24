# Track Table

**Status:** Draft. Amends [gui/mcp-server.md](../gui/mcp-server.md) (the
panel-query tools) and [gui/track-table.md](../gui/track-table.md) (their
non-goal).
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
([gui/track-table.md](../gui/track-table.md) § "The Crop column"), and so are
its computed columns, which the table job computes as a second run after the
stored ones, only once a visible column, a bound or the sort needs them
([gui/track-table.md](../gui/track-table.md) § "Computing the columns"), and
so is *Filter the 3D view*, which dims or hides the points the table does not
list ([gui/track-table.md](../gui/track-table.md) § "Filtering the 3D view",
[gui/point-cloud-rendering.md](../gui/point-cloud-rendering.md) § "The Track
Table's filter"). This draft holds what remains.

Decided:

- The table is a visual tool first. Its first two columns after the point id
  are pictures, **Crop** and **Patch**, both shown by default (built,
  [gui/track-table.md](../gui/track-table.md) § "The Crop column").
- The computed columns of the first version are the maximum ray angle, the
  condition number and the inverse-depth z (built,
  [gui/track-table.md](../gui/track-table.md) § "Computing the columns").
- The table reaches the 3D viewer through an optional toggle that dims or
  hides the points its filters reject (built,
  [gui/track-table.md](../gui/track-table.md) § "Filtering the 3D view").
- Bulk edits on the matched rows are designed for but not built:
  "Remove matching points" is named as a later step, and the query's result
  is shaped so that it can feed `delete_points`.

## Purpose

The Track Table panel ([gui/track-table.md](../gui/track-table.md)) lists
every point of a reconstruction as one row, with the stored per-point numbers
and two pictures of each point as its columns, and the computed columns that
say how well a point is triangulated, sorted and filtered by a person, and can
dim or hide in the 3D Viewer the points it does not list. This draft adds a
pair of wire tools that read and write the panel's query, so an agent can set
up the view a person sees.

## Part 2: the wire

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

## Part 3: later, Remove matching points

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
that wait for the scroll to stop and the reference picks, and step 5, the
computed columns filled by the quiet table job as a second run, and step 6,
*Filter the 3D view*, its dim and hide drawn through a bit of each point's
mask word, are built ([gui/track-table.md](../gui/track-table.md)). Whether
Image Detail follows the filter is an open question there.

- **7. `get_track_table_query` / `set_track_table_query`.**
