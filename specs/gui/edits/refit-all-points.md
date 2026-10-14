# Refit All Points

An action on a reconstruction's row in the Scene Graph that fits every point of
the reconstruction to the photographs again, as Track View's *Fit* followed by
*Commit* does for one track with every observation kept: each observation's
keypoint is aligned to the render of the point's reference observation, the
point is re-triangulated (or, for a point at infinity, its direction is
re-read), its patch is placed at the answer and its stored bitmap is rendered
again, and each observation's readings are taken again. It runs on a worker
thread, can be cancelled, and installs the answer as the node's next
**version**, which an undo steps back out of. No camera and no lens moves.

Related specs:
[../../core/bench/refit-points.md](../../core/bench/refit-points.md) (the
operation underneath: the per-point steps, the rule for held and ranged points,
the write and its measured run time),
[retriangulate-point.md](retriangulate-point.md) (the entry above it, which
re-solves the points from the keypoints they have),
[../scene-graph.md](../scene-graph.md) (the reconstruction row's menu),
[../background-tasks.md](../background-tasks.md) (the worker),
[../track-view.md](../track-view.md) (the bench's *Fit* and *Commit*, for one
track), [../mcp-server.md](../mcp-server.md) § "`refit_all_points`" (the wire),
[README.md](README.md) (the other edit families).

---

## Purpose

`Retriangulate All Points` re-solves each point from the keypoints it already
has. After a run of camera moves or a bundle adjustment, the keypoints of many
points can be a pixel or two from where the photographs put the patch's
content, and re-solving from them keeps that error. Fitting each point on the
bench reads the keypoints again from the photographs, but costs one gesture per
point. This entry does that fit for every point at once.

---

## Invocation

Right-click a reconstruction's row in the Scene Graph and choose `Refit All
Points`. The whole-reconstruction entries read `Bundle Adjust...`,
`Retriangulate All Points`, `Refit All Points`, `Prune Covered Observations`.
It acts on the node the menu was opened on, not on the selection, so the panel
reports it as `SceneGraphResponse::refit_all_points` and `dock.rs` answers with
`AppState::start_refit_all_points`. Its hover text:

> Fit every point to the render of its reference observation, as Track View's
> Fit does with every observation kept: rewrite each observation's keypoint,
> re-triangulate the point and render its patch bitmap again, as one version.
> Moves no camera. Reads every photograph the points observe. Runs on a worker
> thread and can be cancelled; Undo (Ctrl+Z) puts the points back.

It is greyed, with the reason as hover text, by one function,
`state::edits::refit_refusal`
([`refit_all_points.rs`](../../../crates/sfm-explorer/src/state/edits/refit_all_points.rs)),
which the menu entry, the wire's tool and the operation itself all ask. The
reasons, in the order they are checked:

1. an operation is already running on the node (the busy sentence);
2. a camera model proposal is open on the node (*"Apply or cancel the camera
   model proposal first."*), as for every other entry that edits the node; the
   menu passes this sentence in with the busy one, as it does for
   `retriangulate_refusal`;
3. the observations carry no inline keypoint (a `sift_files` reconstruction):
   *"This reconstruction's observations are .sift feature indexes with no
   inline keypoints, and a refit rewrites a keypoint per observation. Convert
   to Embedded Patches first."*;
4. the points carry no patch frame, since a fit registers the frame into every
   view;
5. no image carries a pose.

Whether every photograph can be read and whether the reconstruction's
reprojection noise can be measured are not asked by the gate: the first reads
files and the second walks every observation, and the gate runs every frame the
menu is open. The operation refuses both on the worker, as a failed entry.

Over MCP the same action is `refit_all_points`, which defers to the worker and
comes back with the version.

---

## Mechanism

The operation is `Operation::REFIT_ALL_POINTS` (`Refit all points`,
cancellable, `Kind::Edit`), started by `AppState::start_refit_all_points` with
the job `AppState::refit_all_points_job`, both in
[`refit_all_points.rs`](../../../crates/sfm-explorer/src/state/edits/refit_all_points.rs).
On the GUI thread the job asks the gate, clones the value at the cursor (whose
base is the `Arc` the node goes on drawing), and builds a `ViewSources` naming
every image some live point observes. Nothing is read there. On the worker it:

1. **decodes the photographs** through `ViewSources::decode`, in a `decode
   images` phase whose note says how many were read from disk, reused from the
   photograph cache and how full the cache is. A photograph that cannot be read
   **fails the run** with *"Cannot read <image>."*, as the bench's *Fit* does,
   and no version is pushed. The run does not go ahead without it: refitting
   only the points that do not observe the missing image would leave a few
   points silently skipped, which is harder to read than a refusal that names
   the file;
2. **folds the overlay in** where it is not empty, in a `materialise` phase,
   with `EditedReconstruction::materialize`, which gives the plain value and a
   `RowMap`; where the overlay is empty it reads the base and folds nothing;
3. calls [`bench::refit_points`](../../core/bench/refit-points.md) on that value
   with `RefitPointsOptions::default()`, whose stages (`measure noise`, `fit
   points`, `write back`) nest under the operation's.

It answers with `Finished::Produced`, whose `value` is the refit value and whose
`map` is `PointMap::Chain` of the fold's `PointMap::Rows` (where there was a
fold) and `PointMap::Removed(Vec::new())`, the identity, since the refit moves
no index.

The cancel flag is read in front of the decode, between the photographs, after
the decode, and by the core function in front of each stage and of each point's
fit. A cancelled run pushes no version.

**Readings are written as a commit writes them.** In a reconstruction with no
readings column the refit adds one, as a bench commit of one point does, and
the rows of the points it does not write are "not measured"
([../../core/bench/refit-points.md](../../core/bench/refit-points.md)
§ "Writing the answers"). A bitmap column rendered for display keeps its
display mark, so a save writes no bitmap column the file lacked.

### Run time

On the checked-in ground truths, in a release build on a 32-thread machine, a
refit takes 0.10 s for seoul_bull (280 points, 1277 observations) and 0.40 s for
kerry_park (391 points, 3767 observations), and decoding their photographs
0.10 s and 0.17 s. One thread spends about 1 to 1.3 ms per observation, so a
capture of 250 000 observations is expected to take tens of seconds. The
measurement and its conditions are in
[../../core/bench/refit-points.md](../../core/bench/refit-points.md) § "Run
time".

---

## The version

| | Every point |
|---|---|
| Version label | `Refit seoul_bull` |
| Entry | that label, then the counts, then the serials |

One version, a bulk edit with a new base and an empty overlay, pushed by the
frame that polls the finished task, which installs a `Finished::Produced` as it
installs every one and reports the node in `Polled::installed`. The refit moves
no index (pending point edits close up when the job folds the overlay in, and
the fold's row map in `map` carries the selection across), so the selection
stays on its point, and what the panels cached about each point's geometry,
keypoints and bitmap is dropped off `Polled::installed`. The bench is not
touched: a bench track whose origin is a refit point still resolves, and
committing it replaces the refit point.

**Undo** (Ctrl+Z) steps back to the version before, which holds every point's
keypoints, bitmap and geometry as they were; Redo steps forward again.

**The entry** says what the run did, each count read off the core report's
statuses, a count of zero left out:

```text
Refit seoul_bull: 4 of 8 points refit, 1 of them as stored, 1 of them ranged at
their distance, 1 ranged point released: its reference image has no pose and it
became a direction, 1 crossed to or from infinity, 3 observations kept their
keypoint past the shift bound, 4 left as they were (2 with too few observations,
1 with too few rays, 1 with no bitmap to render), 1 held, median shift 1.0000
(v3 → v4)
```

In order: the points refit out of those read (every point but the held ones);
how many of them agree with what was stored and were not rewritten (`as
stored`); how many ranged points had their direction read at their distance; how
many ranged points were released, a sentence naming why, so a reviewer sees
which statement of theirs the run ended; the points that crossed to or from
infinity; the observations that kept their keypoint because the fit would have
moved them past the shift bound; the points left as they were, with the reason
for each group; the held points; and the median distance a point that stayed
finite moved. The words for each reason a point is left as it was come from one
exhaustive match over `FitError` in `refit_all_points.rs`, so a refusal added to
the fit does not compile until the sentence has words for it.

A **cancelled** run pushes no version and writes one failed entry, `Refit all
points of seoul_bull cancelled`, the sentence the background task writes for
every cancelled operation. A **refusal**, from the gate or on the worker,
pushes no version and writes one failed entry beginning `Refit all points of
seoul_bull refused:`. A run in which every point was left as it was, or agrees
with what is stored, still pushes its version, whose entry says so.

---

## Testing

The core function's tests are
[../../core/bench/refit-points.md](../../core/bench/refit-points.md)
§ "Testing"; what is tested here is the wrapping, in
[`refit_all_points/tests.rs`](../../../crates/sfm-explorer/src/state/edits/refit_all_points/tests.rs)
and beside the other entries' tests.

- **The run pushes one version with a new base**, an empty overlay, the same
  point count, the label `Refit <node>`, a selection that did not have to move,
  and an entry naming the counts, at least one point refit and the points left
  for too few observations; **Undo** returns the base the node had.
- **A pending point edit is folded in**: after a point is deleted, the run
  leaves no overlay, one point fewer in the base, and the selection on the
  point after the deleted one moved down by one.
- **A cancelled run** pushes no version, leaves the base as it was, and writes
  the failed `cancelled` entry.
- **A photograph that cannot be read** fails the run, pushes no version, and
  writes a failed entry naming the image.
- **A node with no inline keypoints** is refused in the sentence the greyed
  entry carries, which names `Convert to Embedded Patches`.
- **The entry's counts**: over a report holding one point of each kind, the
  sentence names every count; a run that refit nothing says so.
- **The photographs read** are the images some live point observes.
- **The Scene tree entry**
  ([`scene_graph/tests.rs`](../../../crates/sfm-explorer/src/scene_graph/tests.rs))
  is live on an embedded_patches node with frames, drawn directly under
  `Retriangulate All Points` and above `Prune Covered Observations`, and reports
  the node it was opened on while another is selected; it is drawn and dead,
  with the gate's sentence, on a `sift_files` node, a node with no frames and a
  busy node. `Prune Covered Observations` is measured against it as the entry
  directly above.
- **The operation is cancellable**, held to its claim by the test that cancels
  every operation declaring so
  ([`background/tests.rs`](../../../crates/sfm-explorer/src/background/tests.rs)).
- **The wire**: `refit_all_points` defers to the worker and comes back with the
  version, and is refused at once on a node with no patch frames.

---

## Non-goals

- **Moving a camera or a lens.** The poses and the cameras are the input, as
  for `Retriangulate All Points`; an answer that moved them would be an
  adjustment ([bundle-adjust.md](bundle-adjust.md)).
- **Changing which observations a point has.** Every observation is kept `in`.
  Deciding which sightings a track should have is the bench's work
  ([../track-view.md](../track-view.md)).
- **Refitting the selection or a list of points.** The run is over every point;
  one point is *Fit* and *Commit* on the bench.
- **Going ahead without a photograph.** A photograph that cannot be read fails
  the run rather than leaving the points that observe it out.
- **Refitting a held point.** The value owns its coordinate, so the run leaves
  it out and counts it.
