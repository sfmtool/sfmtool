# Commit track

The **Commit** button in Track View, which writes the active bench
track into the reconstruction as a point. It is the one step of the bench that
touches the file, and it is one ordinary point edit: one version, one label, one
Action Log entry of kind `Edit`, undoable like any other.

What makes it different from every other edit is what precedes it. A commit
writes down the result of an argument -- a set of sightings assembled, measured,
some taken and some refused -- that was held on the bench until it was settled
([`../bench.md`](../bench.md)).

Related specs: [`../bench.md`](../bench.md) (the bench and its versions),
[`../track-view.md`](../track-view.md) (the panel the button is in),
[`../../core/bench/editable-track.md`](../../core/bench/editable-track.md)
§ "The commit" (the core function this wraps, and every refusal),
[`../document-model.md`](../document-model.md) (the version),
[`../goto-point.md`](../goto-point.md) (the id a created point is named by),
and [`../saving.md`](../saving.md).

---

## Invocation

*Commit*, in Track View's toolbar. The viewer's half is
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

The button is **not** greyed on a track that would write nothing. That reading
is about what the point holds rather than about whether the step can run, the
greying's cached call is made of the track as it sits rather than of the track
seated at the cursor, and the answer a person wants is a row saying so: pressing
Commit on a track already committed is how one asks where its point went.

No keyboard shortcut: it is the end of a piece of work, not a step in a gesture.

---

## Mechanism

Everything below the invocation is
[`../../core/bench/editable-track.md`](../../core/bench/editable-track.md)
§ "The commit": a pure function from the value at the cursor and the track to
the next value and a report carrying the index written, the index replaced, the
`PointMap` and whether anything was written at all. Nothing is triangulated here
-- the track commits with the position it carries, so the record written is the
one the numbers on screen describe.

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

### The commit that writes nothing

A track whose origin already holds **exactly** the record the commit would write
-- every column, compared exactly (core § "The commit") -- has nothing to write,
and the step pushes no version, in the contract every bench step's nothing-to-do
follows ([`../bench.md`](../bench.md) § "The wire"): the history is left alone,
the bench is left alone, one Action Log row of kind `Bench` says so, and the
reply carries `changed: false`. Without it each press of *Commit* would delete
the point and re-add an identical one at a new index.

The point is selected all the same, as any commit selects what it wrote: the
gesture is a question about where the point is as much as an instruction, and
the row names the index. The panels' caches are kept, there being no new version
for them to be stale against.

### The version

A point edit: the value is the node's current one with the record written into
the overlay, so the base is the same `Arc`.

It is the one step pushed with **both halves stated**
(`History::push_pair`): the next reconstruction value, and the bench with the
committed track seated on what it just wrote. The map is the one core reported,
pushed as it stands -- a `Replaced` of one pair for a replacement, a `Created`
for a creation -- and an undo restores the pair: the point gone from the value,
and the track back to the half it had before.

### The selection

**The written point becomes the selection**, through `AppState::select_point`
like any other, so the 3D viewport puts the track rays on it, Track View
shows it in view mode once *Edit* is cleared, and the images that observe it light up. That holds
wherever the selection was standing and whether the commit replaced a point or
created one: a commit is a gesture about one point, and the index it landed at
is the one thing the person who asked for it cannot work out.

It replaces the map-following every other edit does. The map carries a selection
that was already on the origin to the same row this puts it on, and says nothing
about a selection that was elsewhere -- and `PointMap::Created` is the identity
forward, so a creation would leave the selection wherever it was.

Undo and redo are then the map's, exactly as they are for
[Retriangulate Point](retriangulate-point.md): stepping back off a replacement
puts the selection on the point it replaced, and stepping back off a creation
clears it, there being no index on the earlier version that held that point.

The selection is one Action Log row of kind `Selection` after the commit's
`Edit` row, in the order the two happened in. It is not what the step *did*, so
the wire's `report` skips it ([`../mcp-server.md`](../mcp-server.md)).

A commit that creates rather than replaces is pushed with a `CreatedPoints`
carrying the point edit's own content hash and the index it took, because a
point no base has a row for cannot be named by a base's hash
([`../edit-history.md`](../edit-history.md) § "The version graph").

**The track stays on the bench afterwards**, seated on the point just written,
so the person can keep working on it and a second commit replaces what the first
wrote rather than putting a second point on one surface. It is seated at the
version the commit was computed *from*, by the index the point held there: the
version's own map is what carries that index forward to wherever the cursor
later rests.

### What the step hands back

`AppState::commit_bench_track` answers with the point it wrote -- the index it
took, and the index it replaced where it replaced one. One row of the
reconstruction is the whole of what a commit adds, and neither index can be
recovered afterwards: a replacement writes a **new** row and deletes the one it
replaced, a creation takes whatever index the overlay had free, and the sentence
below states neither as a number a caller can use. It is what the step selects;
the wire reports it
as `{ "point": { "index": 4211, "id": "pt3d_95fe75db_0", "replaced": 1207 } }`,
the id being the one Track View shows and `get_point` takes back, so
an agent's next call names the row rather than hunting for it
([`../bench.md`](../bench.md) § "The wire").

### The label

Core's `CommitReport::label`, which needs the name the caller knows the node by:

`Committed track: 5 observations in bull, replacing point 1207`

with `, absorbing M points` when `in` observations pulled from other points were
committed with it. A commit with no origin says only the first clause.

The commit that wrote nothing has the viewer's own sentence instead, which names
the item the way every other no-effect row does:

`Committed bull-nose: no effect, point 4211 already holds this track`

### The Action Log

One entry, of kind `Edit` -- the only bench step whose row is not a `Bench` --
the label plus the version serials:

`Committed track: 5 observations in bull, replacing point 1207 (v11 → v12)`

A commit that wrote nothing is one `Bench` row instead, carrying no serials,
there being no transition to name. A refusal is one failed entry carrying the
core sentence.

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
the selection lands on the written point, and an undo restores the pair -- the
point the commit replaced back and selected again, and the track still on the
bench; a commit made with the selection standing on some other point still
selects what it wrote, and so does one that creates a point, whose undo clears
the selection and whose redo invents none; a commit's row is an `Edit` where
every other bench step's is a `Bench`, with the selection's row after it; and a
commit is what makes a node dirty where a run of bench steps does not. The
repeated press: four commits after the one that wrote the point push no version,
mint no index, leave the bench item's own `Arc` where it is, keep the point
selected and write one no-effect `Bench` row each; and the press after a sighting
is turned out, or after an undo takes the point back, writes again.

The frame's side of it is in
[app/tests.rs](../../../crates/sfm-explorer/src/app/tests.rs) and
[scene_renderer/upload/tests.rs](../../../crates/sfm-explorer/src/scene_renderer/upload/tests.rs):
a commit is a new track-ray source on the point it wrote, and the ray geometry
built for that point is the version's rather than the one the rays were standing
on ([`../architecture.md`](../architecture.md) § "Track Ray Visualization").

Wire ([mcp/tests.rs](../../../crates/sfm-explorer/src/mcp/tests.rs), headless):
the point a commit names resolves both ways -- `get_point` by the index gives
the id the reply carried, and `get_point` by that id gives the index -- a commit
onto an origin reports the index it replaced, and a second commit of the same
track answers `changed: false` with that same point named, at the cursor it was
already at.

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
