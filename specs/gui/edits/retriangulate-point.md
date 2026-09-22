# Retriangulate Point

Two actions that re-solve structure at a geometry somebody else decided: one
point, from its dot's context menu in the 3D viewport, and every point of a
reconstruction, from that reconstruction's row in the Scene Graph. Each installs
the answer as the node's next **version**, which an undo steps back out of.

Nothing about the capture moves. No camera is re-posed, no lens is re-solved,
and no observation is added, removed or walked. What each point's own pixels
support, at the poses and the lens the value already holds, is the whole of what
either action decides.

Related specs:
[../../core/reconstruction/triangulation-rules.md](../../core/reconstruction/triangulation-rules.md)
(the operation underneath, and the per-track rules it reads),
[../viewport-navigation.md](../viewport-navigation.md) (the point menu and how a
right click is told from a right drag),
[../scene-graph.md](../scene-graph.md) (the reconstruction row's menu),
[../background-tasks.md](../background-tasks.md) (the worker the whole-value
action runs on), [../track-view.md](../track-view.md) (the bench the point
menu's other entry stages onto), [README.md](README.md) (the other edit
families).

---

## Purpose

A reconstruction's points and its cameras were fit together, so a point always
agrees with the poses it was fit against -- including the poses that have since
been replaced. Every edit that moves a camera by hand leaves the points it
observes standing where the *old* pose put them unless something re-reads them:
the camera move settles the tracks of the one image it touched
([move-camera.md](move-camera.md)), and the resection settles the tracks of the
image it re-posed ([resect-image.md](resect-image.md)), but a point seen by
images neither of those touched is left describing a geometry that has moved
underneath it.

Retriangulation is that re-reading, asked for deliberately. On one point it is
the question "where do this point's own pixels put it now?", asked of a point a
reviewer is looking at and suspicious of, and answered in one sentence that
names the verdict rather than merely moving the dot. On the whole value it is
the same question asked of everything, which is what a reviewer wants after a
run of camera moves, and it is slow enough to belong on a worker.

It is deliberately **not** an adjustment. A bundle adjustment moves the poses
and the points together toward a joint optimum, which is a different claim about
the capture; this moves the points alone, and leaves a reader able to say which
of the two produced what they are looking at.

---

## Invocation

**One point.** Right-click a point in the 3D viewport -- its dot or its patch,
which the GPU pick decodes to the same point -- and choose `Retriangulate
Point`. The right click selects the point as it opens the menu, so the panels
beside the viewport are describing the point the entry will act on. A right
*drag* is the viewport's zoom and opens nothing;
[../viewport-navigation.md](../viewport-navigation.md) § "The point context
menu" carries that distinction and the menu's other entry, `Edit on Bench`.

**Every point.** Right-click a reconstruction row in the Scene Graph tree and
choose `Retriangulate All Points`.

Both are greyed, with the reason as hover text, when the node cannot be
retriangulated: its observations carry no pixel to cast a ray through, no image
carries a pose, its posed images are taken through more than one lens, or an
operation is already running on it. One function answers that question --
`state::edits::retriangulate_refusal` -- and the menu entry, the wire's two
tools and the operations themselves all ask it, so the greyed entry and a call
that asks anyway give one answer.

Over MCP the same two actions are `retriangulate_point` and
`retriangulate_all_points` ([../mcp-server.md](../mcp-server.md)).

---

## Mechanism

Everything below lives in `sfmtool-core` as one function,
`reconstruction::retriangulate_points` in
[retriangulation.rs](../../../crates/sfmtool-core/src/reconstruction/triangulation).
The viewer wraps it twice, in
[edits.rs](../../../crates/sfm-explorer/src/state/edits.rs): once synchronously
for the point, and once as a background job for the value. Both wrappers pass
the same options, so the two entries cannot disagree about what a
retriangulation is.

The core function gathers each named point's pixels, the poses and the value's
own constraint columns, runs the batch operation once, and writes each answer
back. **Which points** it was asked for also decides the shape of the edit:

- **One point** is an overlay edit. The point's whole record is re-added with
  its new geometry and the old index is deleted, so the base is the same `Arc`
  the node goes on drawing and every other index still means what it meant. The
  point itself takes a **new index**, and the version's map is a
  `PointMap::Replaced` that carries the selection, a copied point id and every
  panel's prepared state onto it.
- **Every point** is a bulk edit. The overlay is folded in, the whole point list
  is rewritten, and the next version's base is a new value with an empty
  overlay. It deletes no point and creates none, so no index moves and no panel
  cache has to be dropped for a renumbering; what is dropped is what the panels
  cached *about* the geometry, which the frame does off `Polled::installed` when
  the version lands.

What each point comes back as is the core operation's to say, and this is where
the two layers meet:

- a point whose rays still cross comes back **finite**, at the place they cross;
- a point a minority of its own observations see behind them is solved on the
  majority, which the report calls out;
- a point every ray of which passes behind a camera that sees it, or whose rays
  no longer cross at all, comes back as the **direction** they agree on -- the
  honest statement of what is left;
- a point the value carries as a direction is read as one and stays one;
- a point the value **holds** at a coordinate is never read: it is not in the
  solve and not in the answer, and asking for that point alone is refused by
  name;
- a point the value **ranges** keeps its distance, and only its direction is
  re-read, from its reference image's camera centre at these poses;
- a point fewer than two of whose observations state a usable ray keeps the
  geometry it had, because the operation has nothing to say about it and saying
  nothing is not the same as saying it is nowhere.

The patch frame of a point that moved is rescaled so the patch keeps the angular
size it had, by the same ratio the adjustment and the camera move resize theirs
by.

### On a worker

The whole-value action is a background operation, `Retriangulate all points`
([../background-tasks.md](../background-tasks.md)). It reports the core
function's three stages -- gathering the arrays, the solve, writing the answer
back -- into the Background panel, and it is **cancellable**: the flag is read at
each of those boundaries, and a cancelled run pushes no version and writes a
failed entry carrying how far it got. The array solve runs to its end once
entered, so a cancel lands between stages rather than inside one.

The single-point action is synchronous, because one track's rays are a
microsecond of arithmetic whatever the reconstruction's size.

---

## The version

| | One point | Every point |
|---|---|---|
| Version label | `Retriangulated point 1207 in seoul_bull` | `Retriangulated seoul_bull` |
| Entry | that label, then `: finite`, then the serials | that label, then the counts, then the serials |

The point entry's tail is the **verdict** the operation reached, in the words
`PointVerdict::label` holds once for the window, the Action Log and the wire:
`finite`, `finite, on the observations that agree`, `at infinity`, `at its held
distance`, `too thin to place, so at infinity`, `behind a camera that sees it,
so at infinity`, `past the reprojection bar, so at infinity`, `too few
observations to place, so left where it was`. That tail is the answer the
gesture was asked for: a point that did not move because one photograph is all
that sees it says so, rather than reporting silent success.

The whole-value entry's tail is what the run did, in the same shape every other
bulk edit reports: how many of how many points moved, how many crossed to or
from infinity, how many were too thinly seen to place, how many were held, and
the median distance a point that stayed finite travelled.

A refusal pushes no version and writes one failed entry, in the state's own
words, which are the words the greyed menu entry carries.

---

## Testing

The core function's own tests are
[triangulation-rules.md](../../core/reconstruction/triangulation-rules.md) § "Testing";
what is tested here is the wrapping.

- **The point edit leaves the base alone.** The node's base is the same `Arc`
  afterwards, the point count is unchanged, the nudged point has come back
  toward the place its own pixels state, and the point beside it has not moved.
- **It takes a new index the selection follows.** The selected point is live
  afterwards and is not the index it was.
- **The entry names the version and the verdict**, carries the two stages a
  point edit has, and is timed from the work rather than from the row.
- **Undo puts the point back**, byte for byte.
- **The bulk edit pushes one version with a new base**, the same point count,
  the label `Retriangulated <node>`, a point pulled back toward its truth, a
  selection that did not have to move, and an entry naming the counts.
- **A cancelled run pushes no version**, leaves the geometry where it was, and
  writes one failed entry saying it was cancelled.
- **A node with no pixel per observation is refused** in the sentence the greyed
  entry carries, and the refusal quotes it.
- **The menu's requests are carried out**: a menu that only opened selects the
  point and edits nothing; `Retriangulate Point` selects and pushes its version;
  `Edit on Bench` stages the track and raises Track View, which is
  asserted against a dock the test closes that panel out of first.
- **The viewport menu itself**, headless: a right click on a point opens it and
  names the point, a right click on nothing opens nothing, a right drag past
  egui's threshold opens nothing, each entry reports the point the menu was
  opened on, and a busy node draws both entries and lets neither be chosen.
- **The Scene tree entry** is live on a node with inline keypoints, drawn and
  dead without them, and drawn and dead while an operation runs on the node.
- **The wire**: `retriangulate_point` pushes a version whose report names the
  verdict, and `retriangulate_all_points` defers to the worker and comes back
  with the version.

---

## Non-goals

- **Moving a camera.** The poses are the input, not the output. An answer that
  moved them would be an adjustment, and there is one of those already
  ([bundle-adjust.md](bundle-adjust.md)).
- **Adding, removing or walking an observation.** The track is read as it
  stands. Deciding which sightings a track should have is the bench's work
  ([../track-view.md](../track-view.md)), and the point menu's other entry is
  how a reviewer gets there from the same gesture.
- **Deleting a point the observations cannot place.** A point too thinly seen
  keeps what it had and is counted, because a retriangulation asked for by hand
  should not quietly cost a reviewer their structure. Deleting is its own
  gesture.
- **A per-point choice of rules.** The options exist on the core function, and
  both entries pass its defaults. A dialog for the angular floor and the
  reprojection bar is a thing to add when somebody wants one, not before.
- **Retriangulating a rig.** The operation reads one shared camera, as the
  adjustment does; a value whose posed images are taken through more than one
  lens is refused rather than solved through one of them.
