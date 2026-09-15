# Commit track

The **Commit** button in the Track Edit panel, which writes the active bench
track into the reconstruction as a point. It is the one step of the bench that
touches the file, and it is one ordinary point edit: one version, one label, one
Action Log entry of kind `Edit`, undoable like any other.

What makes it different from the one-step track edits next door is what precedes
it. Add-observation grows a track the reconstruction already holds and
create-point puts a bearing where a person pointed; a commit writes down the
result of an argument -- a set of sightings assembled, measured, some taken and
some refused -- that was held on the bench until it was settled
([`../bench.md`](../bench.md)).

Related specs: [`../bench.md`](../bench.md) (the bench and its versions),
[`../track-edit.md`](../track-edit.md) (the panel the button is in),
[`../../core/bench/editable-track.md`](../../core/bench/editable-track.md)
§ "The commit" (the core function this wraps, and every refusal),
[`../document-model.md`](../document-model.md) (the version),
[`../goto-point.md`](../goto-point.md) (the id a created point is named by),
[`add-observation.md`](add-observation.md) and
[`create-point.md`](create-point.md) (the one-step edits), and
[`../saving.md`](../saving.md).

---

## Invocation

*Commit*, in the Track Edit panel's toolbar. The viewer's half is
`AppState::commit_bench_track` in
[bench.rs](../../../crates/sfm-explorer/src/bench.rs); the write itself is
`sfmtool_core::bench::commit`. It acts on the active track, which
is what every gesture in a bench panel that names no item acts on.

The button is enabled exactly when the core commit would succeed: its greying
asks `sfmtool_core::bench::commit` of the very track it would commit and shows
that call's own refusal as the hover text, so the button and the step cannot
disagree about when it can run. The refusals are the core ones -- the track is
at the cluster stage (*"upgrade it before committing"*), the node is
`sift_files`, fewer than two observations are `in`, the track carries no
position, no frame or no bitmap where the reconstruction stores one per point,
an `in` observation has no keypoint, or one names an image past the image table
-- plus the viewer's own refusal while a background task holds the node.

No keyboard shortcut: it is the end of a piece of work, not a step in a gesture.

---

## Mechanism

Everything below the invocation is
[`../../core/bench/editable-track.md`](../../core/bench/editable-track.md)
§ "The commit": a pure function from the value at the cursor and the track to
the next value and a report carrying the index written, the index replaced and
the `PointMap`. Nothing is triangulated here -- the track commits with the
position it carries, so the record written is the one the numbers on screen
describe.

### The origin, followed to the cursor

An editable track put on the bench from a point carries an **origin**: the
version it was read out of and the index the point held there. That index is
what makes the commit a replacement rather than a creation, and any number of
edits may have moved the point since. So the viewer follows the origin through
the node's own version graph before committing -- the walk a copied point id
takes ([`../goto-point.md`](../goto-point.md)), defined across an undo, a redo
and a discarded redo tail alike -- and hands core a track seated at the cursor.
An origin that names nothing there, because the point was deleted under it,
leaves the commit creating a point rather than refusing.

### The version

A point edit: the value is the node's current one with the record written into
the overlay, so the base is the same `Arc`.

It is the one step pushed with **both halves stated**
(`History::push_pair`): the next reconstruction value, and the bench with the
committed track seated on what it just wrote. The map is the one core reported,
pushed as it stands -- a `Replaced` of one pair for a replacement, a `Created`
for a creation -- so the selection follows a replaced point exactly as it
follows an added observation, and an undo restores the pair: the point gone from
the value, and the track back to the half it had before.

A commit that creates rather than replaces is pushed with a `CreatedPoints`
carrying the point edit's own content hash and the index it took, because a
point no base has a row for cannot be named by a base's hash. That is the
machinery create-point uses, for the same reason.

**The track stays on the bench afterwards**, seated on the point just written,
so the person can keep working on it and a second commit replaces what the first
wrote rather than putting a second point on one surface. It is seated at the
version the commit was computed *from*, by the index the point held there: the
version's own map is what carries that index forward to wherever the cursor
later rests.

### The label

Core's `CommitReport::label`, which needs the name the caller knows the node by:

`Committed track: 5 observations in bull, replacing point 1207`

with `, absorbing M points` when `in` observations pulled from other points were
committed with it. A commit with no origin says only the first clause.

### The Action Log

One entry, of kind `Edit` -- the only bench step whose row is not a `Bench` --
the label plus the version serials:

`Committed track: 5 observations in bull, replacing point 1207 (v11 → v12)`

A refusal is one failed entry carrying the core sentence.

### What the viewer drops

A commit gives the node a new version, so the panels' caches describing its
points are about a value it no longer holds; the dock drops them exactly as it
does after any other edit that moves points.

---

## Testing

Core (`sfmtool-core`, headless): every commit path and every refusal, over the
synthetic textured plane whose numbers are known to the pixel. See
[`../../core/bench/editable-track.md`](../../core/bench/editable-track.md).

Explorer ([bench/tests.rs](../../../crates/sfm-explorer/src/bench/tests.rs),
headless): a commit replaces its origin point and leaves the point count alone,
the selection follows the replacement, and an undo restores the pair -- the
point the commit replaced back, and the track still on the bench; a commit's row
is an `Edit` where every other bench step's is a `Bench`; and a commit is what
makes a node dirty where a run of bench steps does not.

---

## Non-goals

- **Bundle adjustment after a commit.** The commit writes a record and nothing
  settles around it; the Edit menu's adjustment is a version of its own.
- **Committing to a `sift_files` node.** An observation the localizer placed is
  a keypoint and not a feature index. The bench can be used for inspection on
  such a node and cannot write to it.
- **Committing a cluster.** It has no position to store, and the format has no
  row for a track that is not a point.
- **Committing several tracks at once.** Each is its own version, which is what
  makes each undoable on its own.
