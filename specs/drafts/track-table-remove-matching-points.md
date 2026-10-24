# Remove matching points (amendment)

**Status:** Draft

Amends [gui/track-table.md](../gui/track-table.md), whose § "Non-goals" names
bulk edits on the matched rows and points back here.

The Track Table panel ([gui/track-table.md](../gui/track-table.md)) lists the
points of a reconstruction that pass its bounds and its selected-image box, and
`find_points` ([gui/mcp-server.md](../gui/mcp-server.md) § "`find_points`")
returns the same points over the wire. A person who has bounded the table to the
points they want gone, the points with a track of two and a large error say,
then has to delete them one at a time. This draft names the step that would
delete them all at once.

## Remove matching points

A **Remove matching points** button, and a `delete_matching_points` tool or a
`find_points`-then-`delete_points` recipe, would delete every matched point as
one version, through the existing `delete_points` edit. The result of
`query_points` is already the list of indexes that edit takes, valid at the
document serial it was computed for, so a removal has to refuse a match set
computed for an older serial.

This is not specified beyond that. Still to decide:

- Whether the removal asks for a confirmation, and in what words.
- Whether the hidden rows of a filtered view count.
- The wording of its row in the Action Log and of the version label.
