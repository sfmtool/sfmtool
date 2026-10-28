# Refine All Normals

An action on a reconstruction's row in the Scene Graph that turns the patch of
every finite point of the reconstruction to face along the surface the
photographs show, as Track View's *Grown Plane Normal* followed by *Commit* does
for one track with every observation kept: one plane is fitted to the
photographs each point is seen in, the patch is turned to it about its centre,
its stored bitmap is rendered again, and each observation's readings are taken
again. No point's centre moves, no keypoint moves, and points at infinity are
skipped. It runs on a worker thread, can be cancelled,
and installs the answer as the node's next **version**, which an undo steps back
out of. No camera and no lens moves.

Related specs:
[../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) (the
operation underneath: the per-point steps, the rule for points at infinity and
for held and ranged points, the write and its measured run time),
[refit-all-points.md](refit-all-points.md) (the entry above it, which this one
is built as), [../scene-graph.md](../scene-graph.md) (the reconstruction row's
menu), [../background-tasks.md](../background-tasks.md) (the worker),
[../track-view.md](../track-view.md) (the bench's *Grown Plane Normal* and
*Commit*, for one track), [../mcp-server.md](../mcp-server.md) §
"`refine_all_normals`" (the wire), [README.md](README.md) (the other edit
families).

---

## Purpose

A reconstruction's patches can face the wrong way: a solve may store each patch
facing its cameras rather than along the surface it lies on, and a patch that
faces the wrong way renders tiles that do not line up across the photographs.
Track View's *Grown Plane Normal* turns one patch to the plane its views agree
on, but costs one gesture per point. This entry does that for every point at
once.

---

## Invocation

Right-click a reconstruction's row in the Scene Graph and choose `Refine All
Normals`. The whole-reconstruction entries read `Bundle Adjust...`,
`Retriangulate All Points`, `Refit All Points`, `Refine All Normals`, `Prune
Covered Observations`. It sits directly under `Refit All Points` because it is
the same kind of run over the bench's steps: the refit moves each point and
keeps the way it faces, this turns each patch and keeps its centre. It acts on
the node the menu was opened on, not on the selection, so the panel reports it
as `SceneGraphResponse::refine_all_normals` and `dock.rs` answers with
`AppState::start_refine_all_normals`. Its hover text:

> Turn every finite point's patch to its grown-plane normal, as Track View's
> Grown Plane Normal does with every observation kept, and render each patch
> bitmap again, as one version. Moves no camera, no point's centre and no
> keypoint, and skips points at infinity. Reads
> every photograph the points observe. Runs on a worker thread and can be
> cancelled; Undo (Ctrl+Z) puts the points back.

It is greyed, with the reason as hover text, by one function,
`state::edits::refine_refusal`
([`refine_all_normals.rs`](../../../crates/sfm-explorer/src/state/edits/refine_all_normals.rs)),
which the menu entry, the wire's tool and the operation itself all ask. Its
reasons and their order are Refit All Points'
([refit-all-points.md](refit-all-points.md) § "Invocation"), asked through the
gate the two share (`whole_value_bench_refusal` in
[`refit_all_points.rs`](../../../crates/sfm-explorer/src/state/edits/refit_all_points.rs)),
with sentences that name this entry's work:

1. an operation is already running on the node (the busy sentence);
2. a camera model proposal is open on the node (*"Apply or cancel the camera
   model proposal first."*), which the menu passes in with the busy one;
3. the observations carry no inline keypoint (a `sift_files` reconstruction):
   *"This reconstruction's observations are .sift feature indexes with no
   inline keypoints, and refining a normal writes each point's record,
   keypoints included, as a commit does. Convert to Embedded Patches first."*;
4. the points carry no patch frame: *"This reconstruction's points carry no
   patch frame for the grown-plane normal to turn."*;
5. no image carries a pose.

Whether every photograph can be read is not asked by the gate, which runs every
frame the menu is open; the operation refuses it on the worker, as a failed
entry. A reconstruction whose points are all at infinity is not refused: the
run skips every point and says so.

Over MCP the same action is `refine_all_normals`, which defers to the worker and
comes back with the version.

---

## Mechanism

The operation is `Operation::REFINE_ALL_NORMALS` (`Refine all normals`,
cancellable, `Kind::Edit`, on the slow queue, locking the node), started by
`AppState::start_refine_all_normals` with the job
`AppState::refine_all_normals_job`, both in
[`refine_all_normals.rs`](../../../crates/sfm-explorer/src/state/edits/refine_all_normals.rs).
The job is Refit All Points' with the core call swapped
([refit-all-points.md](refit-all-points.md) § "Mechanism"): on the GUI thread
it asks the gate, clones the value at the cursor and builds a `ViewSources`
naming every image some live point observes; on the worker it

1. **decodes the photographs** in a `decode images` phase, and a photograph
   that cannot be read **fails the run** with *"Cannot read <image>."*, pushing
   no version;
2. **folds the overlay in** where it is not empty, in a `materialise` phase;
3. calls [`bench::refine_normals`](../../core/bench/refine-normals.md) on that
   value with `RefineNormalsOptions::default()`, which runs the grown-plane
   normal step at the default options Track View's *Grown Plane Normal* runs,
   and whose stages (`refine normals`, `write back`) nest under the
   operation's.

The bar splits 1 : 9 between the decode and the rest. It answers with
`Finished::Produced`, whose `map` is the fold's `PointMap::Rows` (where there
was a fold) chained with the identity, since the refinement moves no index.

The cancel flag is read in front of the decode, between the photographs, after
the decode, and by the core function in front of each stage and of each point's
step. A cancelled run pushes no version.

Readings are written as a commit writes them: a reconstruction with no
readings column gains one where a written point brings a measured reading, and
a bitmap column rendered for display keeps its display mark
([../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) §
"Writing the answers").

### Run time

On the checked-in ground truths, in a release build on a 32-thread machine, the
refinement takes 0.44 s for seoul_bull (280 points, 14 of them at infinity,
1277 observations) and 2.0 s for kerry_park (391 points, 9 at infinity, 3767
observations). One thread spends about 6 to 8 ms per observation, so a capture
of 250 000 observations is expected to take a couple of minutes. The
measurement and its conditions are in
[../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) §
"Run time".

---

## The version

| | Every point |
|---|---|
| Version label | `Refined the normals of seoul_bull` |
| Entry | that label, then the counts, then the serials |

One version, a bulk edit with a new base and an empty overlay, pushed by the
frame that polls the finished task, as for Refit All Points: the refinement
moves no index, so the selection stays on its point, and what the panels cached
about each point's geometry, keypoints and bitmap is dropped off
`Polled::installed`. The bench is not touched.

**Undo** (Ctrl+Z) steps back to the version before, which holds every point's
frame, keypoints and bitmap as they were; Redo steps forward again.

**The entry** says what the run did, each count read off the core report's
statuses, a count of zero left out:

```text
Refined the normals of seoul_bull: 3 of 8 finite points turned, 1 of them as
stored, 1 of them with low confidence, 3 points left their normal unchanged (1
with no view joined, 2 with a large turn on weak evidence), 2 points left as
they were (1 with too few observations, 1 with no bitmap to render), 2 points
at infinity skipped, median turn 4.0° (v3 → v4)
```

In order: the points turned out of the finite points; how many of them agree
with what was stored and were not rewritten (`as stored`); how many of them
were turned to a plane found with low confidence, which the step turns to
rather than rejecting; the points whose normal the step
kept, with why; the points the step refused or rendered no bitmap for, with the
reason for each group; the points at infinity skipped; and the median turn. The
words for each reason a point is left as it was come from one exhaustive match
over `NormalError` in `refine_all_normals.rs`, so a refusal added to the normal
steps does not compile until the sentence has words for it.

A **cancelled** run pushes no version and writes one failed entry, `Refine all
normals of seoul_bull cancelled`. A **refusal**, from the gate or on the
worker, pushes no version and writes one failed entry beginning `Refine all
normals of seoul_bull refused:`. A run in which no point was turned still
pushes its version, whose entry says so.

---

## Testing

The core function's tests are
[../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) §
"Testing"; what is tested here is the wrapping, in
[`refine_all_normals/tests.rs`](../../../crates/sfm-explorer/src/state/edits/refine_all_normals/tests.rs),
over Refit All Points' fixture, and beside the other entries' tests.

- **The run pushes one version with a new base**, an empty overlay, the same
  point count, the label `Refined the normals of <node>`, a selection that did
  not have to move, and an entry naming the counts and the points left for too
  few observations; **Undo** returns the base the node had.
- **A pending point edit is folded in**, and the selection follows the fold.
- **A cancelled run** pushes no version and writes the failed `cancelled`
  entry.
- **A photograph that cannot be read** fails the run and names the image.
- **A node with no inline keypoints** is refused in the sentence the greyed
  entry carries.
- **The entry's counts**: over a report holding one point of each kind, the
  sentence names every count; a run that turned nothing says so.
- **The Scene tree entry**
  ([`scene_graph/tests.rs`](../../../crates/sfm-explorer/src/scene_graph/tests.rs))
  is live on an embedded_patches node with frames, drawn directly under `Refit
  All Points` and above `Prune Covered Observations`, and reports the node it
  was opened on while another is selected; it is drawn and dead, with the
  gate's sentence, on a `sift_files` node, a node with no frames and a busy
  node.
- **The operation is cancellable**, held to its claim by the test that cancels
  every operation declaring so
  ([`background/tests.rs`](../../../crates/sfm-explorer/src/background/tests.rs)).
- **The wire**: `refine_all_normals` defers to the worker under the operation's
  name and comes back with the version, and is refused at once on a node with
  no patch frames.

---

## Non-goals

- **Moving a point, a camera or a lens.** The centres, the poses and the
  cameras are the input; moving the points is `Refit All Points`.
- **Fitting the keypoints.** The step trusts the keypoints it is given and
  moves none of them, so a second run starts from the same keypoints. A
  reviewer whose keypoints may be off runs `Refit All Points` first, and one
  who wants them relocalized on the turned patches runs it after.
- **Changing which observations a point has.** Every observation is kept `in`.
- **Refining the selection or a list of points.** The run is over every point;
  one point is *Grown Plane Normal* and *Commit* on the bench.
- **Choosing the normal step or its settings.** The run uses the grown-plane
  normal at its default options, the ones Track View's entry uses.
- **Going ahead without a photograph.** A photograph that cannot be read fails
  the run.
