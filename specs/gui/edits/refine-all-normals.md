# Refine All Normals

An action on a reconstruction's row in the Scene Graph that turns the patch of
every finite point of the reconstruction to face along the surface the
photographs show, as Track View's *Grown Plane Normal* or *Pairwise Flow
Normal* followed by *Commit* does for one track with every observation kept: a
plane is fitted to the photographs each point is seen in, by the step the
reviewer chose, the patch is turned to it about its centre, its stored bitmap
is rendered again, and each observation's readings are taken again. No point's
centre moves, no keypoint moves, and points at infinity are skipped. It runs on
a worker thread, can be cancelled, and installs the answer as the node's next **version**, which an undo steps back
out of. No camera and no lens moves.

Related specs:
[../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) (the
operation underneath: the per-point steps, the rule for points at infinity and
for held and ranged points, the write and its measured run time),
[refit-all-points.md](refit-all-points.md) (the entry above it, which this one
is built as), [../scene-graph.md](../scene-graph.md) (the reconstruction row's
menu), [../background-tasks.md](../background-tasks.md) (the worker),
[../track-view.md](../track-view.md) (the bench's *Grown Plane Normal*,
*Pairwise Flow Normal* and *Commit*, for one track),
[../mcp-server.md](../mcp-server.md) § "`refine_all_normals`" (the wire), [README.md](README.md) (the other edit
families).

---

## Purpose

A reconstruction's patches can face the wrong way: a solve may store each patch
facing its cameras rather than along the surface it lies on, and a patch that
faces the wrong way renders tiles that do not line up across the photographs.
Track View's *Grown Plane Normal* and *Pairwise Flow Normal* each turn one
patch to the plane its views agree on, but cost one gesture per point. This
entry does either for every point at once.

---

## Invocation

Right-click a reconstruction's row in the Scene Graph and open `Refine All
Normals ▸`, a submenu of two entries, one per normal step:

| Entry | Step | Wire `method` |
|---|---|---|
| `Grown Plane` | the grown-plane normal, as Track View's *Grown Plane Normal* runs it | `grown_plane` (the default) |
| `Pairwise Flow` | the pairwise-flow normal, as Track View's *Pairwise Flow Normal* runs it | `pairwise_flow` |

Each entry is named as Track View names its step, less the *Normal* the
submenu already says, and runs that step at its default options, the ones
Track View's entry runs. The whole-reconstruction entries read `Bundle
Adjust...`, `Retriangulate All Points`, `Refit All Points`, `Refine All
Normals ▸`, `Prune Covered Observations`. It sits directly under `Refit All
Points` because it is the same kind of run over the bench's steps: the refit
moves each point and keeps the way it faces, this turns each patch and keeps
its centre. It acts on the node the menu was opened on, not on the selection,
so the panel reports the entry chosen as
`SceneGraphResponse::refine_all_normals`, the node with the
`RefineAllNormalsStep` the entry names, and `dock.rs` answers with
`AppState::start_refine_all_normals`. Opening the submenu runs nothing. The
hover text of `Grown Plane`:

> Turn every finite point's patch to its grown-plane normal, as Track View's
> Grown Plane Normal does with every observation kept, and render each patch
> bitmap again, as one version. Moves no camera, no point's centre and no
> keypoint, and skips points at infinity. Reads every photograph the points
> observe. Runs on a worker thread and can be cancelled; Undo (Ctrl+Z) puts the
> points back.

That of `Pairwise Flow` names the pairwise-flow normal and Track View's
Pairwise Flow Normal in the same sentences, and ends:

> It reads the optical flow between pairs of each point's views, so it takes
> far longer than Grown Plane.

When the refinement cannot run, `Refine All Normals` is drawn as a greyed entry
rather than a submenu, as `Align to` is, since a submenu that does not open
cannot say why. It is greyed, with the reason as hover text, by one function,
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
   patch frame for a normal step to turn."*;
5. no image carries a pose.

Whether every photograph can be read is not asked by the gate, which runs every
frame the menu is open; the operation refuses it on the worker, as a failed
entry. A reconstruction whose points are all at infinity is not refused: the
run skips every point and says so.

Over MCP the same action is `refine_all_normals`, whose `method` names the
step (`grown_plane` by default, or `pairwise_flow`), which defers to the worker
and comes back with the version.

---

## Mechanism

The operation is `Operation::REFINE_ALL_NORMALS` (`Refine all normals`,
cancellable, `Kind::Edit`, on the slow queue, locking the node), one operation
for both steps, started by
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
   value with `RefineNormalsOptions::step` set to the chosen step at its
   default options, `RefineNormalsStep::GrownPlane` or
   `RefineNormalsStep::PairwiseFlow`, the options Track View's *Grown Plane
   Normal* and *Pairwise Flow Normal* run, and whose stages (`refine normals`,
   `write back`) nest under the operation's.

The bar splits 1 : 9 between the decode and the rest. It answers with
`Finished::Produced`, whose `map` is the fold's `PointMap::Rows` (where there
was a fold) chained with the identity, since the refinement moves no index.

The cancel flag is read in front of the decode, between the photographs, after
the decode, and by the core function in front of each stage and of each point's
step and inside the step ([refine-normals.md](../../core/bench/refine-normals.md)
§ "Cancellation"). A cancelled run pushes no version.

Readings are written as a commit writes them: a reconstruction with no
readings column gains one where a written point brings a measured reading, and
a bitmap column rendered for display keeps its display mark
([../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) §
"Writing the answers").

### Run time

On the checked-in ground truths, in a release build on a 32-thread machine, the
grown-plane refinement takes 0.44 s for seoul_bull (280 points, 14 of them at
infinity, 1277 observations) and 2.0 s for kerry_park (391 points, 9 at infinity, 3767
observations). One thread spends about 6 to 8 ms per observation, so a capture
of 250 000 observations is expected to take a couple of minutes. The
measurement and its conditions are in
[../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) §
"Run time".

The pairwise-flow refinement costs far more per point
([../../core/bench/refine-normals.md](../../core/bench/refine-normals.md) §
"Choosing the step"). Through the Python binding in a release build on the
same machine, it takes 90 s for seoul_bull's v01 ground truth (266 finite
points), where the grown plane took 0.6 s in the same run, and 106 s for its v02
ground truth (274 finite points). The bar counts the points as they finish, and
the cancel flag is read inside each point's step, so a cancel is answered
within a point's step rather than at the end of the run.

---

## The version

| | Every point |
|---|---|
| Version label | `Refined the normals of seoul_bull with the grown-plane normal`, or `… with the pairwise-flow normal` |
| Entry | that label, then the counts, then the serials |

The label names the step that ran, in the glossary's words for it, so the
history and the Action Log tell the two runs apart; the operation, `Refine all
normals`, is one for both.

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
Refined the normals of seoul_bull with the grown-plane normal: 3 of 8 finite
points turned, 1 of them as stored, 1 of them with low confidence, 3 points
left their normal unchanged (1 with no view joined, 2 with a large turn on weak
evidence), 2 points left as they were (1 with too few observations, 1 with no
bitmap to render), 2 points at infinity skipped, median turn 4.0° (v3 → v4)
```

```text
Refined the normals of seoul_bull with the pairwise-flow normal: 2 of 5 finite
points turned, 1 of them from a single pair, 3 points left their normal
unchanged (2 undetermined, 1 unstable), median turn 6.0° (v3 → v4)
```

In order: the points turned out of the finite points; how many of them agree
with what was stored and were not rewritten (`as stored`); how many of them
were turned to a plane the grown-plane normal found with low confidence, which
the step turns to rather than rejecting; how many the pairwise-flow normal
read from a single pair, a point with two observations whose one pair had no
agreement to check; the points whose normal the step kept, with why (for the
grown-plane normal, no view joined or a large turn on weak evidence; for the
pairwise-flow normal, undetermined or unstable, in the glossary's words for
those outcomes); the points the step refused or rendered no bitmap for, with
the reason for each group; the points at infinity skipped; and the median turn. The
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
  point count, the label `Refined the normals of <node> with the grown-plane
  normal`, a selection that did not have to move, and an entry naming the
  counts and the points left for too few observations; **Undo** returns the
  base the node had.
- **A pairwise-flow run** pushes one version labelled `… with the
  pairwise-flow normal`, whose entry names the points left for too few
  observations and one of the outcomes only that step has (single pair,
  undetermined or unstable), and Undo returns the base.
- **A pending point edit is folded in**, and the selection follows the fold.
- **A cancelled run**, of either step, pushes no version and writes the failed
  `cancelled` entry.
- **A photograph that cannot be read** fails the run and names the image.
- **A node with no inline keypoints** is refused in the sentence the greyed
  entry carries.
- **The entry's counts**: over a report holding one point of each kind, the
  sentence names every count; over one holding the pairwise-flow normal's
  outcomes, it names the single pair, undetermined and unstable counts; a run
  that turned nothing says so.
- **The Scene tree entry**
  ([`scene_graph/tests.rs`](../../../crates/sfm-explorer/src/scene_graph/tests.rs))
  is a live submenu on an embedded_patches node with frames, drawn directly
  under `Refit All Points` and above `Prune Covered Observations`; opening it
  reports nothing, it holds `Grown Plane` above `Pairwise Flow`, and each
  reports the node the menu was opened on while another is selected, with its
  own step. It is drawn and dead, with the gate's sentence, on a `sift_files`
  node, a node with no frames and a busy node.
- **The operation is cancellable**, held to its claim by the test that cancels
  every operation declaring so
  ([`background/tests.rs`](../../../crates/sfm-explorer/src/background/tests.rs)).
- **The wire**: `refine_all_normals` defers to the worker under the operation's
  name and comes back with the version, and is refused at once on a node with
  no patch frames; its `method` parses to the grown-plane step when absent or
  `grown_plane` and to the pairwise-flow step for `pairwise_flow`, and a
  `method` it does not take is refused at the parse
  ([`mcp/tests/edit.rs`](../../../crates/sfm-explorer/src/mcp/tests/edit.rs)).

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
  one point is *Grown Plane Normal* or *Pairwise Flow Normal* and *Commit* on
  the bench.
- **Choosing a step's settings, or the piece steps.** Each entry runs its step
  at its default options, the ones Track View's entry uses; the
  finite-difference and grid-plane normals are not offered.
- **Going ahead without a photograph.** A photograph that cannot be read fails
  the run.
