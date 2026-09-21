# The bench and the editable track

**Status:** Draft

A track is the record of which photographs saw one point on a surface and where
in each photograph it appears. The viewer can already delete a point, and that is
a single decisive step that lands in the reconstruction's edit history. What it
cannot do is *work on* a track: take one out of the reconstruction, or start one
that is not in it yet, try images against it, see what each would contribute,
take some and leave others, compare it with another, and only then put it back.
This draft proposes the **bench**: a place beside each loaded reconstruction
where things are put to be worked on, held in the same history as the
reconstruction so one Undo covers both. The first kind of thing that goes on it
is the **editable track**, edited in a **Track Edit panel**: a track with the
searches that suggest observations for it, the photometric and geometric measurements
that judge them, the two representations it passes through on the way from a
set of image patches to a reconstructed point, and the one step that finally
writes it into the reconstruction as a version like any other.

**Decided:** the bench is per node, holds a list of items with one active per
kind, the way the reconstruction has one selected point and one selected image,
each kind edited in a bench panel of its own of which Track Edit is the first, and
is the second half of every version so one Undo walks the reconstruction and
the bench together; the editable track is its first kind of item and the only
one this draft specifies; an editable track has two stages, a 2D cluster stage
and a 3D track stage, with transitions in both directions; every evaluation is
a pure function that produces a report, and installing the report is a step
like any other; the commit is one ordinary edit; the Point Track Detail panel
stays view-only; the bench and the editable track are values and pure
functions in `sfmtool-core`, bound to Python, and the viewer adds only the
history, the panels and the wire.

Related standing specs: [`../gui/bench.md`](../gui/bench.md),
[`../gui/track-edit.md`](../gui/track-edit.md) and
[`../gui/edits/commit-track.md`](../gui/edits/commit-track.md) (the viewer half
that is built: the bench as a half of every version, the panel, the Scene tree
group and the commit),
[`../core/bench/bench.md`](../core/bench/bench.md) and
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) (the core
half that is built: the two values, the steps, the evaluation at both stages and
the transitions between them, whose non-goals are what this draft proposes),
[`../gui/point-track-detail.md`](../gui/point-track-detail.md)
(the view-only panel the editable track is the editing counterpart of),
[`../gui/edits/commit-track.md`](../gui/edits/commit-track.md) (the one step
that reaches the reconstruction),
[`../gui/document-model.md`](../gui/document-model.md)
(the version the bench joins and the commit makes),
[`../gui/edit-history.md`](../gui/edit-history.md) (the cursor the bench is
walked by), [`../gui/background-tasks.md`](../gui/background-tasks.md)
(where an evaluation runs), [`../core/patch/cluster-patches.md`](../core/patch/cluster-patches.md)
and [`../core/patch/cluster-patch-refinement.md`](../core/patch/cluster-patch-refinement.md)
(the cluster stage's representation and kernel),
[`../core/patch/patch-keypoint-localization.md`](../core/patch/patch-keypoint-localization.md),
[`../core/patch/patch-view-selection.md`](../core/patch/patch-view-selection.md),
[`../core/patch/patch-localizability.md`](../core/patch/patch-localizability.md)
and [`../core/patch/member-coherence-validation.md`](../core/patch/member-coherence-validation.md)
(the track stage's kernels), [`../core/patch/candidate-track-spawning.md`](../core/patch/candidate-track-spawning.md)
(the pipeline an upgrade runs), [`../formats/kdf-file-format.md`](../formats/kdf-file-format.md)
and [`../core/features/lazy-kdforest-query.md`](../core/features/lazy-kdforest-query.md)
(the SIFT index), [`../gui/sift-index.md`](../gui/sift-index.md) (where
it lives and what makes one stale), and [`sfm-explorer-editing.md`](sfm-explorer-editing.md)
(the umbrella, whose "split a track and merge two" this draft absorbs).

---

## Part 1: why a bench

The one-step edits are the right shape for a gap the eye has already closed:
select the point, right-click the pixel, done. They are the wrong shape for the
work that precedes that certainty, for four reasons that each argue for the same
thing.

**Two tracks on screen.** Judging a candidate track means looking at it beside
the track it might be, might duplicate, or might be split from. The Point Track
Detail panel shows the selected point's track, and the selection is one value
the whole window agrees on. A second track needs a second place, and making
Point Track Detail able to show two would either take the selection away from
the rest of the window or turn one panel into two. A place of its own is the
smaller change, and it leaves the view-only panel exactly as clean as it is.

**A track that is not a point yet has no 3D patch.** Every observation of an
`embedded_patches` point is the same surfel re-anchored in one image, and the
kernels that place and score an observation take that surfel as given. A
candidate track assembled from image patches has none: it is a reference patch
plus, per observation, the affine warp that carries the reference onto that observation's
image, which is what a `.matches` file's cluster-patches section holds and what
`sfm cluster-patches` computes before any pose exists. So the editable track has
to carry that representation too, and be able to move to the 3D one once
there is enough to triangulate, and back when the 3D hypothesis turns out to be
the thing that was wrong.

**Suggestions need an index.** "Which other images might show this patch?" has
two good answers and the viewer can give neither today. Descriptor space
answers it across the whole workspace without a pose, through the `.kdf` forest
the matcher searches. The surfel answers it across the reconstruction's own
images, by projecting the patch into every camera that geometrically sees it and
scoring what each shows. Both produce a ranked list with numbers attached, and
the human's job is to set the bar and look at what sits near it.

**A place to put things.** A track is the first thing worth taking out of the
reconstruction to work on, and it will not be the last: a pose being re-fitted
against a hand-picked set of points, a set of images being judged together, a
cluster read out of a `.matches` file. The bench is the place, and the editable
track is what it is designed against. Giving the place its own state and its own
vocabulary is what lets the next kind of item be tried without touching the
selection model, the edit families or the view-only panels.

---

## Part 2: the bench

The bench is a value, `Bench`, one per node: a list of **items** in the order
they were put there, and, **per kind of item, the label of the active one**. An
item is an `enum` with, in this draft, one variant, `Track(Arc<EditableTrack>)`;
the point of the enum is that the list, the labels, the activation and the
history below are the bench's and not the track's, so a second kind of item
joins by adding a variant and its own active label.

The active labels are the bench's counterpart of the reconstruction's selections.
The window agrees on one selected point and one selected image, and every
panel reads them; the bench likewise has one active track, and later one active
image or whatever else is put on it, and each kind's **bench panel** shows that
kind's active item and acts on it when a gesture names no target. Track Edit is
the first bench panel. A second kind of item gets a second panel, not a mode in
this one, so each panel's vocabulary stays that of the thing it edits.

Each item has a **label**, unique on its bench, and the label is how an item is
named everywhere: in the Scene tree, on the panel's tabs, in every Action Log
row and version label, and on the wire. A label is minted from what the item
was made from, so a log line says what the thing is without a lookup: a track
from a point is its point id, `pt3d_a1b2c3d4_1207`; a cluster from a pixel is
the image stem and the pixel, `IMG_0042@142,198`; one from a `.sift` feature is
the stem and the feature index, `IMG_0042#847`; a split takes its parent's label
with `-split` appended. A collision takes ` (2)`, ` (3)` as a node's label does
([`../gui/scene-graph.md`](../gui/scene-graph.md)), and an item can be renamed,
which is a step like any other. A label is stable until renamed, and a
discarded item's label is free to be minted again, because it names nothing
then. An item belongs to one node and is not part of that node's
reconstruction: a save does not write it, the point count does not include it,
and no panel that reads the reconstruction sees it. It lives in the node's
history all the same.

Several items are held at once because judging one usually means judging it
against another that is also not settled yet: two search results that might be
one point, the two halves of a split, a track pulled apart and the pieces kept
side by side until one of them earns a commit.

### One history for the (reconstruction, bench) pair

A version of the node is today one reconstruction value, base plus overlay
([`../gui/document-model.md`](../gui/document-model.md)). Under this draft a
version is a **pair**: that value, and the bench as it stood. A document edit
produces the next version with the same bench; a step on one item, a verdict, a
search, an installed report, an upgrade, produces the next version with the
same document value and a bench in which that item is a new `Arc` and every
other is the old one; putting an item on, taking one off, or changing which of
a kind is active is a step that changes the bench's list and no item; a commit is the one
step that changes both halves at once. Every version shares whatever it did not
change, so a step on a track costs the size of that track, which is a few
hundred observations with their measurements and one small bitmap, and a document
edit costs nothing extra for the bench it carries along.

The consequences are the ones the value model already pays for, applied to one
more field:

- **Undo and redo walk the pair.** The Edit menu's Undo, the shortcuts, the Edit
  History panel and the wire's `undo`, `redo` and `jump_to_version` move the
  cursor over one list, and whichever half a step changed comes back. Turning a
  observation `out`, running a sweep, upgrading, committing, then deleting some other
  point are five versions in one order, and undo retraces them in that order.
  The person editing a track never has to know which of their steps touched
  the file.
- **Truncation is one rule.** A new step after an undo discards the redo tail,
  whichever half the discarded versions had changed.
- **Dirty is about the document half.** A version is dirty when its document
  value is not the one on disk; a run of bench steps over a clean value is
  clean, because a save of it would write the same bytes, and the `*` marker
  and the close prompt say so. The Edit History panel marks the version on disk
  as it does now, and a bench step above it is a row that does not move the
  mark.
- **The budget counts the bench.** A version's unshared bytes include the items
  it does not share with its predecessor. A track is small against a bulk edit,
  so the budget's arithmetic does not change, only what it sums.
- **The maps are trivial for a bench step.** Point indexes are untouched, so
  the step's `PointMap` is an empty `Removed`, the selection stays where it is,
  and an id copied before the step resolves after it.

What this buys over a history of the bench's own is that there is one Undo. Two
stacks would leave the person guessing which the shortcut would act on, and a
commit, which changes both, would have to appear in both or in neither. A
version that is a pair has no such question: a step is a step.

**Closing a node drops its bench with its history.** An item names images of
one node and is meaningless without it. **Discarding an item** is a
step: the list without it, and if it was active the one before it becomes
active. An undo puts it back, so discarding needs no confirmation.

---

## Part 3: the editable track

An editable track is the first kind of item: a track being worked on, with
everything that has been tried against it.

### Observations, candidates and verdicts

The track is a list of **observations**, the word the `.sfmr` format uses for a
track's entries and the one this draft uses at both stages, even though the
cluster kernel it wraps calls the same thing a *member* in its own identifiers
(`member_status`, `member_zncc`). Each names one image of the node and one
place in it, and each carries three things the document's observations do not:

- **Where it came from.** A observation's provenance is one of: the committed track
  the item was put on the bench from; a SIFT feature returned by a descriptor
  search; an image the view sweep proposed; a pixel the user pointed at; or a
  observation of another track the user pulled in. Provenance is shown, never used by
  a kernel.
- **What was measured about it**, per stage (§ "The two stages"). A
  measurement is a report and never a decision.
- **A verdict**, which is the user's: `in`, `out`, or `candidate`. A candidate is
  something a search proposed and nobody has ruled on; `out` is a observation the
  user refused, kept in the list so the next search does not propose it again
  and so the refusal is visible. The kernels run over the `in` observations and score
  the candidates against them.

**Thresholds propose, they do not decide.** The panel's sliders (minimum ZNCC,
maximum shift, maximum keypoint uncertainty, minimum triangulation angle) paint
each row as *would pass* or *would not*, and a button applies that painting as
verdicts in one step. A verdict set by hand is pinned and a slider does not move
it. This is the difference between the bench and the batch pipeline: the
pipeline's gates decide, and its reports say what they decided; here the same
numbers are shown and the person decides.

**One observation per image.** Two observations in one image is what the cluster kernel
calls `duplicate_image` and refuses, and a track cannot observe an image twice.
A second candidate in an image already held is shown, scored, and cannot be
turned `in` until the other is turned `out`.

### Origin

An editable track may have an **origin**: the point it was put on the bench
from, as a version serial and the index the point had there. The origin is what
makes a commit a replacement rather than a creation. A track started from a
pixel or from a search has none.

The origin is followed through the node's version graph the way a copied point
id is ([`../gui/goto-point.md`](../gui/goto-point.md)): a commit, an undo or
another edit does not lose it, and a version that deleted the point leaves the
track with an origin that resolves to nothing, which the header says.

### The two stages

An editable track is in exactly one of two stages, and the stage is what says
which kernels apply and what a commit can do.

**The cluster stage** is a `.matches` cluster with its cluster-patches section,
in memory: a **reference** observation, a template cut around it at
`patch_size` keypoint-frame units, and per observation a seed (position and 2×2
affine shape, in that image's pixels), the refined absolute position and shape,
the achieved ZNCC, the shift from the seed, the localizability of the observation's
own tile and a status in the `member_status` legend
([`../formats/matches-file-format.md`](../formats/matches-file-format.md)). No
pose, no position, no normal. It is what a track is when it starts from a search
hit or a pixel, and what the refinement kernel `refine_cluster_patches`
evaluates.

**The track stage** is an `embedded_patches` point that is not in the value
yet: a position, a patch frame, a consensus bitmap, and per observation a keypoint
with the localizer's leave-one-out ZNCC, the reprojection error against the
triangulated position, the ray angle and the observation's tile localizability. It
is what a track is when put on the bench from a committed point, and what the
localizer, the sub-pixel refiner, view selection and the coherence matrix
evaluate.

A track put on the bench from a committed point begins at the track stage with
the point's own frame, bitmap and keypoints, and its observations `in`. Its
measurements are the ones the Point Track Detail panel already computes, so
putting a track on the bench and doing nothing shows the same numbers the
view-only panel shows, plus the verdict column.

### Transitions

Both directions are one operation, **set the stage**, `set_bench_track_stage`
on the wire and a *Stage* toggle on the panel; setting the stage a track is
already at changes nothing and pushes no version. The two directions do
different work and are described separately.

**Upgrade, cluster to track**, needs the observations' images to have poses in the
node, at least two observations `in`, and a triangulation that is finite, in front
of every camera and not degenerate. It runs the spawn pipeline's own steps
([`../core/patch/candidate-track-spawning.md`](../core/patch/candidate-track-spawning.md)):

1. **Triangulate** the `in` observations' refined positions through their cameras.
2. **Frame** the patch at that position: the in-plane axes and half-extents from
   the reference observation's affine shape projected back through its camera at the
   triangulated depth, and the mean viewing direction as the normal, which is
   how `to_embedded_patches` frames a SIFT feature. A observation whose ray is
   near-grazing to that plane is reported as such rather than placed.
3. **Localize and refine** every `in` observation's keypoint against that surfel,
   seeded at its refined cluster position, with the same two kernel stages the
   embed pass chains, and re-triangulate from the result.
4. **Fuse** the consensus bitmap over the surviving observations.

The cluster-stage measurements are dropped with the stage: each is a registration
against a reference and a template the track no longer has, and a track carries
the measurements of its current stage and no other.

**Downgrade, track to cluster**, is always possible and is lossy on purpose. The
reference becomes the `in` observation with the largest projected patch scale; each
observation's seed becomes its keypoint and the affine shape the format derives by
projecting the frame at that observation's anchor
([`../formats/sfmr-file-format.md`](../formats/sfmr-file-format.md) § "Deriving
keypoint shape, scale, and orientation"); the position, frame and bitmap are
dropped. This is the step to take when the observations were right and the 3D
hypothesis was the problem: a point triangulated off a chimera, a frame refined
onto the wrong surface, a pose that has since been moved. The cluster kernel then
judges the observations on appearance alone, and an upgrade builds the 3D afresh from
whatever survives.

**Commit, track to the document**, is § "Part 6". A cluster cannot be committed:
it has no position to store, and the format has no row for a track that is not a
point. The header says so, and the Commit button names the upgrade as the step
that is missing.

---

## Part 4: evaluation

An evaluation is a pure function from one editable track plus the node's value
plus the decoded images to a report. (As built, this part is two steps: a
an `evaluate` that reads the track where it sits and moves nothing, and a `fit`
that localizes, re-triangulates and re-fuses and then reads its own result --
see [`../core/bench/editable-track.md`](../core/bench/editable-track.md). What
is said below about a report landing on the observations it measured holds for
both.) Installing the report is one version: the
bench at the cursor with the report's measurements written onto that track's
observations, the document half unchanged. Nothing in an evaluation writes the
document.

**A report lands on the observations it measured, wherever the cursor is.** Observations
are appended and never renumbered, and a measurement is keyed by observation index
and separate from the verdict, so a report computed against the track as it
stood when the task began still applies when it finishes: a observation the user
turned `out` meanwhile is still scored, and the panel shows the score under the
`out` verdict rather than pretending the run did not see it. A observation added
meanwhile is simply unscored until the next run. If the cursor has moved to a
version whose bench no longer holds the item, the report is discarded with one
Action Log row saying so, since there is nothing left for it to describe.

Every evaluation runs as a **background task**
([`../gui/background-tasks.md`](../gui/background-tasks.md)), with the panel
naming the phases: the decode of each image not yet in the full-resolution cache
is a phase of its own, because on a sweep over hundreds of candidates it is the
whole cost and the person watching should see that it is the photographs and not
the kernels. The task is on the node whose images and poses it reads, and the
Background Task panel shows it as it shows an adjustment. One task at a time is
the existing rule and it is right here too: two evaluations of one track racing
each other would have two reports to install.

What is measured, per stage:

| Stage | Over the `in` observations | Per candidate |
|---|---|---|
| Cluster | reference selection, template, per-observation warp, ZNCC, shift, tile σ_pos, status | the same, scored against the `in` observations' reference |
| Track | triangulation, frame, localized keypoints, LOO ZNCC, reprojection error, ray angle, tile σ_pos, coherence matrix, consensus bitmap | ZNCC against the consensus at the projection (cheap), or after a keypoint search (the localizer, one observation against the consensus) |

**Keypoint search is a switch.** At the track stage a candidate can be scored
where the surfel projects, which is view selection's affine candidate score and
costs one small render, or the localizer can be allowed to search for it within
`max_shift_px` first, which costs a
localization per candidate. The panel has the switch and the radius, because the
two answer different questions: the first asks whether the image shows the
patch where the geometry says it is, the second whether it shows it nearby. A
candidate that scores well only after a search is telling the user something
about the pose or the position, and the shift column is where that shows.

**Thresholds are the panel's, with the pipeline's defaults.** `min_zncc` 0.85
and `max_shift_px` 3.0 from cluster refinement, `max_member_keypoint_uncertainty`
0.35 grid px from the localizer, `min_relative_zncc` 0.7 from view selection.
They are read from the same constants those kernels default to, so the bench and
the batch pass start from the same bar, and moving a slider is the person
choosing to differ.

---

## Part 5: finding observations

Four ways a observation gets onto an editable track. Each produces candidates and
nothing else; the verdict is the user's.

### From a pixel

Right-click in the Image Detail panel: **Add to editable track here**, which
adds to the active track. With no track on the bench, the same entry reads
**Start editable track here...** and opens a radius prompt offering the node's
own default patch radius, so the first observation's seed has a scale; it
puts a new track on the bench and makes it active. A subsequent pixel observation
takes the reference's scale carried by the affine seed the cluster kernel
derives, and needs no prompt. The observation is a `candidate` scored on the next
evaluation. This is the gesture for the patch every detector missed.

### From the descriptor index

**Search descriptors** is a **constellation query**, not a lookup of one
descriptor. A SIFT keypoint is an extremum of the difference-of-Gaussians scale
space, and every descriptor in a `.kdf` was computed at one; a pixel someone
pointed at is not an extremum at any scale, so a descriptor computed there
corresponds to nothing any detector produced for the same surface in another
image, and its neighbours in the index are noise. What is stable is the
**neighbourhood**: the detected keypoints around the observation are extrema,
each has a descriptor in the index, and the same surface in another image
carries the same constellation under a locally affine warp.

So the query takes the detected keypoints within a radius of the observation
in its own image (a few dozen to a few hundred), asks the forest for each
one's `k` nearest neighbours (a larger `k` than the matcher's, since most of a
feature's neighbours belong to images that do not show the patch), groups the
hits by image, and fits an **affine warp per image by RANSAC** over the
correspondences, reading the hits' positions from the forest's geometry corpus
so no `.sift` file is opened. An image whose best warp carries enough inliers
is a candidate, and the warp is what makes it useful: applied to the
observation's own pixel and affine shape, it gives that image a **seed
position and shape** for the cluster stage, whether the observation was a
detected feature or a hand-placed pixel. The inlier count is shown per
candidate, ranks them, and is an admission the cluster kernel's ZNCC then
judges photometrically. This is the access pattern the file-backed index was
measured against
([`../core/features/lazy-kdforest-query.md`](../core/features/lazy-kdforest-query.md)
§ "patch constellation"): a query touches a small fraction of the corpus and
costs tens of milliseconds with a warm cache.

For a detected-feature observation the constellation still includes the
feature's own descriptor, so a single-descriptor hit is a special case of the
constellation with one correspondence, and it is the geometric consistency of
the rest that earns the candidate a warp rather than a guess.

The forest has to exist, and it belongs to the reconstruction: the node's Scene
tree carries a **SIFT Index** row naming the `.kdf` beside its `.sfmr`, with a
file chooser and a **Build SIFT Index** entry that runs the forest build over
the node's `.sift` files as a background task and writes the file
([`../gui/sift-index.md`](../gui/sift-index.md)). A node that has never been
saved, or one whose images have no `.sift` companion, has no descriptor search,
and the row says which.

### From the view sweep

**Built**, as a row's context-menu entry *Find matches by geometry* rather than
the toolbar button proposed here, and without the keypoint-search switch: the
search seeds each admitted image at the surfel's own projection and leaves the
localization to the next Evaluate or Fit. Filed as
[`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Searching by geometry" and
[`../gui/track-edit.md`](../gui/track-edit.md) § "Right-clicking a row". What
is described below is the proposal it came from.

At the track stage, **Sweep views** takes every image of the node that
geometrically sees the surfel, the front-facing candidacy test of view
selection, and scores each against the consensus of the `in` observations, with or
without the keypoint search. Every scored image not already in the track
becomes a candidate. A node of five hundred images can propose a hundred
candidates; the table sorts them by score and the threshold painting says where
the bar the user set falls in that list. This is the search the batch pipeline
runs per point in `embed-patches`, put under a person's control for one point.

### From another track

**Pull in the selected track** takes the observations of the point selected in the
Point Track Detail panel and adds them to the active track as candidates,
marked as coming from that point. **Pull in from the bench** does the same from
another track on the bench, marking them as coming from it. Either is how two
tracks that might be one are tested: pull the second in, evaluate, and read the
track's own coherence matrix, where two surfaces show as two blocks and one
surface as one (§ "Part 4"). Nothing compares two tracks as such: the
Point Track Detail panel shows the selected point's track and Track Edit shows
the active one, and looking at both is the comparison. A commit that keeps observations pulled from a point
is a merge, § "Part 6"; observations pulled from another item are just observations,
since an item is not a point and nothing has to be deleted for them. The item
they came from is left as it was, and discarding it afterwards is the
person's choice.

**How this relates to putting a point on the bench.** `create_bench_track` and
the point form of `pull_into_bench_track` read the same thing, a point's track,
and do different things with it. The create makes a **new item** whose observations
are `in` and whose origin is that point, so a commit of it *replaces* the
point. The pull adds the observations to an **existing item** as candidates, with
the point recorded as their provenance, so a commit of the receiving track
*absorbs* the point. The pull from a point is therefore the pull from a bench
track applied to a track the point would have made, without the item being
put on the bench, and the provenance is what carries the point across so the
commit knows what to delete. A observation pulled from a bench track that itself has
an origin carries that origin as its provenance too, so the merge does the same
thing whether the second track was pulled from the reconstruction directly or
had been put on the bench first.

---

## Part 6: the commit, and the document

The commit is the only step that touches the reconstruction, and it is one
ordinary point edit with one version, one label and one Action Log entry.

**What it needs.** The track stage, an `embedded_patches` node, at least two
observations `in`, and a finite triangulation from the last evaluation with no
observation changed since; a track that has been edited since its last evaluation is
re-evaluated first, as part of the same task, so the record that is written is
one the numbers on screen describe. A `sift_files` node refuses, for the reason
the one-step edits refuse: an observation the localizer moved is a keypoint and
not a `.sift` feature index.

**What it writes** is a `PointRecord`: the triangulated position, the frame, the
consensus bitmap, the colour read from the bitmap's centre, the normal the frame
states, and one observation per `in` observation with its localized keypoint and the
fit's leave-one-out ZNCC in `observation_confidence` where the column exists.

- **With no origin**, `EditedReconstruction::add_point`, pushed with
  `push_creating` so the point has an id of the `pt3d_{edit hash}_{k}` form
  ([`../gui/goto-point.md`](../gui/goto-point.md)). The map is
  `PointMap::Created`.
- **With an origin** that resolves in the version at the cursor,
  `replace_point` on that index. The map is `PointMap::Replaced`, one pair, and
  the selection follows it as it follows an added observation.
- **With `in` observations whose provenance names a point** other than the origin,
  the commit also deletes those points, because a track cannot observe an image
  twice and a reconstruction should not hold two points for one surface. The
  provenance is what a pull records (§ "Part 5"), whether the observations came from
  the point directly or through a bench track that had that point as its
  origin. The map is a `Chain` of the replacement and the removals, and the
  label says how many were absorbed. Only `in` observations count: a candidate or an
  `out` observation pulled from a point leaves that point alone. This is the merge
  the umbrella draft listed.

A **split** needs no edit of its own: put the track on the bench, select the
observations that belong to the other surface, and **Split off selected observations**
puts a second track on the bench of exactly those, at the cluster stage, beside
the first, from which they are removed. The observations are named explicitly, by
row selection in the panel and by observation index on the wire, rather than read
off the verdicts: `out` says a observation does not belong *here*, and a observation the
localizer refused and a observation that belongs to the track next door are both
`out`, so a verdict cannot say which of them the new track should take. Both
tracks are on the bench, both can be evaluated, and each is committed on its
own when it is ready: the first as a replacement of the point, the second as a
new one. The bench is what makes the split a thing to look at rather than a
thing to remember.

**The label** is `Committed track: N observations in <node label>`, with
`, replacing point <index>` when there was an origin and `, absorbing M points`
when there were pulled-in observations. **The Action Log entry** is the label plus
the fit's numbers, LOO ZNCC over the observations and the triangulation's condition
number, the way every edit's entry carries the numbers of its own family.

**After the commit the track stays on the bench**, with its origin set to the
point just written, its observations' provenance unchanged, and its measurements
those of the committing evaluation. The person can keep working on it, and a
second commit is a replacement of what the first wrote. The commit is one
version whose two halves both changed, so an undo of it restores the pair: the
point is gone from the value and the track is back to having no origin, or its
old one, exactly as it stood.

**Every bench operation is a version, and only the commit is an edit of the
file.** Putting an item on, adding a candidate, a verdict, a search's results,
an installed report, an upgrade, a downgrade and a discard each push
one version with the document half untouched, write one Action Log row
attributed to whoever asked, and appear in the Edit History as rows that do not
move the on-disk mark. The label says what the step did: `Put point 1207 on the
bench`, `Turned image_012.jpg out of pt3d_a1b2c3d4_1207`, `Swept 83 views for
IMG_0042@142,198: 41 candidates`, `Set IMG_0042@142,198 to the track
stage`, `Discarded IMG_0042@142,198 from the bench`.
A step that changes nothing, a verdict a observation already has, pushes no version.

---

## Part 7: the Track Edit panel

The bench is a place; a bench panel is for editing one kind of thing that is on
it, and shows that kind's active item (§ "Part 2"). **Track Edit** is the first
bench panel: it edits the tracks on the bench and shows the active track.

**The bench is in the Scene tree.** Each `.sfmr` node gains a **Bench** child,
beside its Camera Images and Camera Intrinsics groups
([`../gui/scene-graph.md`](../gui/scene-graph.md)). Inside it, today, is the
list of editable tracks: one row per track in the order they were put on, the
active one marked, each by its label. Clicking a row makes it active, which is a step, and raises
the Track Edit panel; a secondary click offers *Discard*. When another
kind of item exists it is listed in the same Bench child under its own kind,
and clicking it raises the panel that edits that kind. The bench is in the tree
because the tree is already where a node's parts are listed, and it is per node
because an item names that node's images and poses.

**Placement.** A tab, `Tab::TrackEdit`, titled **Track Edit**, whose home is the
top-right node beside Image Detail and Point Track, as the non-active tab
([`../gui/panel-layout.md`](../gui/panel-layout.md) § "Home positions"). It is a
panel like any other: closeable, ticked in the Panels menu. Like the Edit
History panel it has almost no state of its own: the bench is the selected
node's, at its cursor, and the panel struct holds only its tile textures and
the slider positions, which a close keeps.

**The track row.** A row of tabs along the top of the panel, one per track on
the bench in the order they were put on, the active one raised, each by its
label with `N in` beside it and a close mark that discards it. Clicking one
makes it active, which is a step; a **+** at the end puts an empty track on.
The row is tabs rather than a tree because a track is small enough to be named
in a word and the point of holding several is to flick between them. It lists
tracks only: an item of another kind is in the Scene tree's Bench group and in
its own panel, not here. Below the row, the panel shows the active track.

**Empty.** With no track on the bench: `No track on the bench`, above three
ways in, each naming its gesture: *Put selected track on the bench* (greyed
with no point selected), *Start from a pixel: right-click in Image Detail*, and
*Build a SIFT index: the Scene tree's SIFT Index row*. The Point Track Detail
panel gains the first of
these as one line under its stored-patch tile, beside the hints it already
carries, quoting the button's label from one constant. Putting a point on the
bench that already has a track with that origin makes that track active rather
than putting a second on.

**Header.** The active track's stage as a word, its origin as a point id or
`new`, `N in · M candidates · K out`, and the last evaluation's headline: at the
cluster stage the reference observation and the template size; at the track stage
the position, the condition number, the max pair angle and the consensus's own
localizability, which is the same header the view-only panel draws plus the
stage.

**Toolbar.** *Evaluate*, *Fit*, a *Stage* toggle between `cluster` and `track`, *Sweep views* at the
track stage, *Search descriptors*, *Pull in selected*, *Pull in from bench...*,
*Split off selected observations*, *Commit*, and *Discard*. No confirmation on
discarding, because it is a version and undo puts the item back. Each is greyed
with a hover text naming what is missing, in the style of the Image Detail menu
entries.

**Thresholds.** One row of sliders per stage, with the pipeline's defaults, and
*Apply as verdicts*. The table paints under them live.

**The observation table.** One row per observation, the view-only panel's columns first
so a reader who knows that panel reads this one, then what the bench adds:

| Column | Cluster stage | Track stage |
|---|---|---|
| Verdict | tri-state, click cycles `in` / `out` / `candidate` | same |
| Tile | the reference template warped onto this observation | the surfel re-rendered from this observation at its keypoint, the view-only panel's tile |
| Thumbnail, Image, Name | as Point Track Detail | as Point Track Detail |
| ZNCC | against the reference | leave-one-out against the consensus |
| Seed sh., Proj. off | from the seed, px | the peak's move from the sighting, and the sighting's distance from the projection |
| σ_pos | the observation's own tile | the same |
| Error, Angle | absent | as Point Track Detail |
| Status | the `member_status` word | `localized`, the reading's reason sentence, or `not evaluated` |
| From | provenance | provenance |

Rows are grouped `in`, then candidates by score, then `out`, each group
collapsible. Clicking a row selects the image as the view-only panel does, and
selects the row; Ctrl-click and Shift-click extend the row selection, which is
what *Split off selected observations* reads and is panel state rather than a
version. Double-clicking enters camera view; hovering sets the cross-panel
hover; a secondary click opens *Set reference* at the cluster stage, *Search
descriptors from here*, *Split off selected observations*, and the verdict entries.

**The coherence matrix**, under the table at the track stage: the pairwise
ZNCC over the `in` observations and the candidates as a heat grid, with the rows in
the table's order, so a track that is two surfaces shows as two blocks and a
pulled-in track that belongs shows as one. Collapsed to one line at the cluster
stage and before the first evaluation.

### What the other panels show

The bench is drawn wherever the reconstruction is, in one colour of its own so
it is never mistaken for committed structure, with the active track at full
strength and the others dimmed, the way the selected point's track rays stand
out from the rest:

- **Image Detail** draws every observation of every track on the bench in the image
  on screen as its own overlay layer, composing with any mode as the intrinsics
  layer does: `in` observations as filled markers with their affine ellipse,
  candidates hollow, `out` observations as a cross. Clicking a marker of the active
  track cycles its verdict; clicking one of another track makes that track
  active. The layer draws under the panel's one hover tooltip, appending the
  observation's numbers and its item.
- **The 3D Viewer** is **built**, as the active track alone drawn where it
  stands rather than every track as a ghost point through a preview buffer of
  its own, and so without the click on a ghost point that would make its track
  active. Filed as
  [`../gui/viewer-3d-bench-layer.md`](../gui/viewer-3d-bench-layer.md): the
  patch's square as depth-aware scene geometry, its normal with an arrowhead,
  and one mark per observation, every one of them a handle and two of them
  saying what no photograph can. A cluster-stage track draws nothing, because
  there is nothing in 3D to draw.
- **The Image Browser** borders the active track's `in` observations' thumbnails in
  the bench colour, beside the orange the selected point's track gets.
- **The Point Track Detail panel** is unchanged in what it shows. Its one line
  of new text is the way onto the bench.

---

## Part 8: the wire

**Built**, for every tool whose step exists, and filed as
[`../gui/bench.md`](../gui/bench.md) § "The wire" and
[`../gui/mcp-server.md`](../gui/mcp-server.md) § "The bench family": the two
creates with all three seed forms, the three item tools, the two reads, and the
eight steps on a track (`add_bench_track_observation`,
`set_bench_track_verdict`, `apply_bench_track_thresholds`,
`evaluate_bench_track`, `fit_bench_track`, `set_bench_track_stage`,
`split_bench_track` and `commit_bench_track`). The two searches are built too:
`search_bench_track_descriptors` against the SIFT index, and the sweep as
`search_bench_track_geometry`, filed in
[`../gui/mcp-server.md`](../gui/mcp-server.md) § "The bench family" and
[`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Searching by geometry". **Still proposed here** is `pull_into_bench_track`,
which arrives with the step below that builds it.

An agent gets the same bench a human does, through tools that are each one
`AppState` call, on the GUI thread at the same point in the frame as every
other tool ([`../gui/mcp-server.md`](../gui/mcp-server.md) § "Threading"). An
evaluation, a sweep, a search and a commit start background tasks and answer
through the two-level reply that spec defines, with a handle to poll when the
work is long. Every bench step answers as an edit answers, with the version it
pushed, because it is one; `undo`, `redo` and `jump_to_version` need no bench
variant, since the history they walk already holds the bench steps, and
`get_history` lists them with their labels among the rest.

```jsonc
// The bench. Each create answers with the new item's label and makes it active.
// create_bench_cluster  { "reconstruction_label": "bull", "camera_image": 4,
//                         "pixel": [142.0, 197.5], "radius_px": 7.5 }
// create_bench_cluster  { "reconstruction_label": "bull", "camera_image": 4,
//                         "pixel": [142.0, 197.5],
//                         "affine": [[7.1, -0.4], [0.4, 7.1]] }
// create_bench_cluster  { "reconstruction_label": "bull", "camera_image": 4,
//                         "feature": 847 }                 // a .sift feature
// create_bench_track    { "reconstruction_label": "bull", "point": 1207 }
// create_bench_track    { "reconstruction_label": "bull",
//                         "point": "pt3d_a1b2c3d4_1207" }
// activate_bench_item   { "item": "IMG_0042@142,198" }
// rename_bench_item     { "item": "IMG_0042@142,198", "label": "bull-nose" }
// discard_bench_item    { "item": "bull-nose" }
// get_bench             { "reconstruction_label": "bull" }
//
// An editable track on it. "track" omitted means the active track.
// add_bench_track_observation    { "track": "bull-nose", "camera_image": 4, "pixel": [142.0, 197.5] }
// add_bench_track_observation    { "track": "bull-nose", "camera_image": 4, "feature": 847 }
// set_bench_track_verdict        { "track": "bull-nose", "observation": 3, "verdict": "in" }
// search_bench_track_descriptors { "track": "bull-nose", "observation": 0, "alpha": 0.8 }
// sweep_bench_track_views        { "track": "bull-nose", "keypoint_search": true, "max_shift_px": 3.0 }
// evaluate_bench_track           { "track": "bull-nose" }
// fit_bench_track               { "track": "bull-nose" }
// set_bench_track_stage          { "track": "bull-nose", "stage": "track" }
// set_bench_track_stage          { "track": "bull-nose", "stage": "cluster" }
// pull_into_bench_track          { "track": "bull-nose", "point": 1301 }   // or "from_track": "…"
// split_bench_track              { "track": "bull-nose", "observations": [3, 5, 8] }   // answers with the new item's label
// commit_bench_track             { "track": "bull-nose" }
// get_bench_track                { "track": "bull-nose" }
```

**Every name is verb first**, as every tool on this surface is (`get_scene`,
`delete_point`, `move_camera_image`), with `bench` or `bench_track` naming
what the verb acts on. **The two creates are named by the stage they make, and where the first observation
comes from is a parameter.** `create_bench_cluster` makes a cluster-stage track
with one observation, seeded from a pixel and a radius, a pixel and a 2×2 affine
shape, or a `.sift` feature index, which carries its own; it is the tool behind
the Image Detail entry and behind an agent that has a feature in hand.
`create_bench_track` makes a track-stage track from a point of the
reconstruction, by bare index or qualified id, resolved as `get_point` resolves
one; it is the tool behind *Put selected track on the bench*. The names say
which stage the item starts at, because that is what the caller needs to know
next: a cluster wants observations and an upgrade, a track wants a sweep or a
commit. `add_bench_track_observation` takes the same seed forms as the cluster create.

Every tool takes `reconstruction_label`. `get_bench` is the tree's Bench group as JSON:
labels, kinds, origins, stages, counts and, per kind, which is active. `get_bench_track` is one
track's table: the stage, the origin, the thresholds, and every observation with its
provenance, verdict and both stages' measurements where they exist. A observation is
addressed by its position in that list, which is stable for the life of the
track: observations are appended and never renumbered, so an agent holding a observation
index after a verdict or an evaluation still holds the same observation. An item is
named by its label, exactly as a node is by `reconstruction_label`; a label that
names nothing on the bench is refused naming it, and a rename's reply carries
the new label, as `save_reconstruction`'s carries a renamed node's. `commit_bench_track` answers as every
edit answers, with the version it pushed and the sentence the Action Log
recorded.

A refusal is in the bench's own words and pushes nothing: *"IMG_0042@142,198 is
at the cluster stage; upgrade it before committing."*, *"Image 4 already has a observation
in; turn it out first."*, *"No SIFT index is open."*,
*"Nothing on the bench is called `bull-nose`."*

---

## Part 9: where it lives

**The rule: as much as possible in `sfmtool-core`, and nothing GUI-shaped in
it.** The bench and the editable track are values and pure functions over
them, and a script that wants to grow a track under its own thresholds, or
split one by a rule of its own, should be able to do so through the Python
bindings without a window. The viewer adds only what is about the window: the
history the bench is a half of, the panels, the tree, the Action Log rows and
the wire.

### In core

The module `sfmtool_core::bench`, bound as `sfmtool._sfmtool.bench`. **The two
values, every step over them, the evaluation at both stages and the transitions
between them are built**, and are filed as
[`../core/bench/bench.md`](../core/bench/bench.md) and
[`../core/bench/editable-track.md`](../core/bench/editable-track.md); what those
specs list as non-goals is what this draft still proposes.

- **`EditableTrack`**, *built*, with the evaluation and the stage change: the observations with their provenance, seeds,
  measurements and verdicts, the stage and the stage's data (reference and
  template, or position, frame and consensus bitmap), the origin, the
  thresholds. A plain value: `Clone`, no interior mutability, no handle to any
  device, cache or window. The decoded images an evaluation needs are a named
  input, one `ProjectedImage` per image of the node, exactly as every
  photometric kernel takes them, so decoding and caching stay the caller's.
- **`Bench`**, *built*: the list of items, their labels, the active label per
  kind, the label minting and the collision suffix, and the rename. Also a plain
  value. It is in core rather than the viewer because a script holding several
  candidate tracks wants the same list, the same labels in its output and the
  same "the active one" default; nothing in it is about a window.
- **The steps, as pure functions** from a value plus inputs to a new value and
  a report, never `&mut` on a shared thing. *Built*: `create_cluster` from a
  seed, `create_track` from a point of an `EditedReconstruction`,
  `add_observation`, `set_verdict`, `apply_thresholds` (the painting, as
  proposed verdicts), `split`, **`commit`**, which is a function from an
  `EditedReconstruction` and a track to the next `EditedReconstruction` and a
  report, the same shape as every other step, and the two
  that register pixels, `evaluate` and `set_stage`, which take the decoded views
  and a `Progress`. *Proposed here*: `search_descriptors` over a
  `LazyKdForest`, `sweep_views`, and `pull_in`, which reads another track. Every
  refusal is an error enum whose `Display` is the sentence the panel and the
  wire show.
- **Progress** through the one `Progress` parameter every long core function
  takes ([`../gui/operation-progress.md`](../gui/operation-progress.md)), so an
  evaluation names its phases the same way whether the Background Task panel or
  a script is watching. `evaluate` and `set_stage` name the phases the batch
  kernels carry -- `refine` and `localizability` at the cluster stage,
  `localize`, `refine`, `fuse` and `localizability` at the track stage. The
  steps that read no photograph take no `Progress`: each is decided by what the
  reconstruction and the person already say, with no phase inside it worth a
  row.

**The commit answers with a `PointMap`.** `PointMap` is core's type
([`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)
§ "The point map"), so `commit` returns the write's map, chained with a
`Removed` of the absorbed points where there are any, and the viewer pushes it as
it stands. **A split hands the half it takes off back as a cluster**, through
the same downgrade `set_stage` runs, which is why `split` takes the
reconstruction: the 3D hypothesis fitted to both halves is the thing a split
questions, and carrying it onto the new half would state it as fact.

Most of what those functions call was already there, because the batch
pipeline needed it: the cluster stage is `refine_cluster_patches` over an
in-memory cluster; the upgrade is the spawn pipeline's steps over one candidate
with a caller-supplied view set and seeds; the track-stage measurements are
`localize_patch_keypoints`, `refine_patch_keypoints` and
`score_localizability_stack`; the commit is `add_point` and `replace_point`.
What the sweep and the searches still need: `select_patch_views` for the
candidacy test, `member_coherence` for the pairwise grid, and
`LazyKdForest::search` plus `resolve_feature_geometry` and the matcher's
membership radius for the descriptor query.

Every step answers `(next value, report)`, so the line that is not yet built
reads exactly like the ones that are:

```python
from sfmtool._sfmtool.bench import (
    Bench, add_observation, apply_thresholds, commit, create_cluster, evaluate,
    set_stage, set_verdict, sweep_views,
)

edited = EditedReconstruction(recon)
bench = Bench()
bench, track = create_cluster(bench, image=4, image_stem="IMG_0042",
                              pixel=(142.0, 197.5), radius_px=3.0)
track, added = add_observation(track, 7, (88.5, 210.0))     # the same patch, another photograph
track, _ = set_verdict(track, added["observation"], "in")
track, report = evaluate(track, edited, images)             # the cluster kernel, over the seeds
print(report["measured"], "of", track.observation_count, "register")
track, staged = set_stage(track, edited, images, "track")   # triangulate, frame, localize, fuse
track, report = sweep_views(track, edited, images, keypoint_search=True)   # proposed here
track, painted = apply_thresholds(track, min_zncc=0.9)  # proposes verdicts; pinned ones stay
edited, report = commit(edited, track, node="bull")
print(report["label"])                         # the sentence the Action Log would show
```

### In the viewer

**Built**, and filed as [`../gui/bench.md`](../gui/bench.md),
[`../gui/track-edit.md`](../gui/track-edit.md) and
[`../gui/edits/commit-track.md`](../gui/edits/commit-track.md):

- `History`'s `Version` holds the bench half, a push states both halves, the
  dirty test reads the document half, and the budget sums both. This is the only
  place a bench step becomes a *version*: core has no history.
- The Track Edit panel (`track_edit/` beside `point_track_detail/`) with its
  item tabs, header, toolbar, sliders and observation table -- including the
  per-observation tile column, which renders the surfel through the view-only
  panel's own renderer at the track stage and the refinement kernel's own grid
  at the cluster stage -- and the Bench child in the Scene tree.
- The two gestures that name a pixel, as entries in the Image Detail context
  menu beside the two point edits that name one: *Start cluster on the bench
  here* and *Add observation to bench track here*.
- The Image Detail bench layer: the active track drawn over the photograph in
  the bench's own colours, the surfel's outline sampled and projected through
  the lens at the track stage and the observations' parallelograms at the
  cluster stage, with a click on a mark selecting that row in Track Edit.
- The Action Log row and version label for each step, the actor column, and the
  evaluation and the stage change as background tasks, each of which is one
  `AppState` call that decodes the images through the node's full-resolution
  cache, calls the core function, and pushes the result.
- The wire, in `mcp/bench.rs`: fifteen tools that are each one of those
  `AppState` calls, with the two reads that have no panel gesture behind them
  because a panel shows what they answer.

**Still proposed here**: the preview buffer in the 3D viewer, the Image Browser
borders, the cluster stage's template drawn beside each member's tile, and the
view sweep and the pull-in with their toolbar entries, their coherence grid and
their two wire tools.

**No second SIFT cache.** The viewer already holds each image's keypoints
(positions and affine shapes, never descriptors) in `AppState::sift_cache`, and
the constellation query takes an already-read keypoint set as its input for
exactly this reason: the constellation around an observation comes from that
cache, and the few dozen descriptors it needs come by id from the forest's own
corpus reads, under the forest's cache budget. Nothing in the viewer holds
descriptors.

Two core pieces were built ahead of the bench, and one more is needed:

1. **The constellation query** is in core, beside the forest in
   `features::kdforest` and bound to Python: the per-feature forest query, the
   grouping of hits by image, and a three-point affine RANSAC per image
   returning per image the warp, its inlier count and the correspondences, with
   the hit geometry read from the forest's own corpus. See
   [`../core/features/kdf-constellation-query.md`](../core/features/kdf-constellation-query.md).
   `describe_keypoints` (below) is **not** the search's query: a descriptor at a
   pixel nobody detected matches nothing the index holds.
2. **Describing a keypoint that was not detected.**
   `sfmtool_core::features::sift::describe_keypoints(image, params, keypoints)`
   returns the 128-byte descriptor of each caller-supplied `QueryKeypoint` (a
   position and a 2x2 affine shape, or a position, size and orientation),
   computed by the kernel the extractor describes its own detections with; the
   octave and pyramid level follow from the size
   (`ScaleSpace::octave_layer_for_scale`). It is bound beside `extract_sift` as
   `sfmtool._sfmtool.sift.describe_keypoints`, with
   `affine_shapes_from_similarity` for the size-and-angle form. It was built
   for the search and turned out not to serve it (item 1); what remains useful
   is the shape, scale and orientation correspondence it defined once, which
   the cluster stage's seeds use. See
   [`../core/features/sift.md`](../core/features/sift.md).
3. **Framing a surfel from one observation's affine shape at a depth.**
   `OrientedPatch::from_affine_shape_at_depth(camera, cam_from_world, keypoint,
   affine_shape, depth)` in
   [cloud.rs](../../crates/sfmtool-core/src/patch/cloud.rs) is the upgrade's
   step 2: the patch centre is the keypoint's ray at that depth, the normal is
   the viewing direction, and the in-plane axes and half-extents are what the
   shape's columns unproject to on that plane. It is the inverse of the
   format's frame-to-shape rule, which the downgrade uses, so the two
   directions share one statement of the relationship. Bound as
   `OrientedPatch.from_affine_shape_at_depth`. See
   [`../core/patch/patch-cloud.md`](../core/patch/patch-cloud.md).

Neither `to_embedded_patches` nor a cluster seed calls the new constructor: both
frame a patch from inputs it does not take. `to_embedded_patches` frames per
*point*, from a position that is already triangulated, with a normal averaged
over every observing view, an in-plane rotation from the first observing
camera's up axis and one isotropic half-size reduced across the views; a cluster
seed frames from a radius in one image's pixels, where there
is no depth at all. The new constructor frames per *observation*: one view, one
shape, one depth, anisotropic, normal along that view's bearing.

---

## Part 10: testing

**Core**, headless, for the two new functions: a described keypoint at a
detected keypoint's own position, scale and orientation reproduces the detected
descriptor exactly; a frame built from an affine shape at a depth projects back
to that shape.

**The bench and the track**, in core (`crates/sfmtool-core/src/bench/tests.rs`)
for everything that is a value or a pure function, and in the viewer
(`bench/tests.rs` beside `document/tests.rs`) for the history, over the
synthetic textured-plane scene the localization tests already build:

- A track put on the bench from a committed point is at the track stage with
  every observation `in` and reports the numbers Point Track Detail reports for that
  point.
- Downgrade then upgrade of that track, with nothing turned `out`, triangulates
  back to within tolerance of where it was.
- A candidate placed on the plane by pixel scores above the bar and one placed
  off every image is `not evaluated`; a verdict pinned by hand survives a
  slider move and an evaluation.
- Two observations in one image cannot both be `in`.
- Commit with no origin appends a point with the `in` observations' keypoints and
  mints an edit-hash id; commit with an origin replaces it and the selection
  follows; commit with pulled-in observations deletes their points and chains the
  map; a cluster-stage track, a `sift_files` node and a track with one observation
  each refuse, naming why.
- Putting a second item on, making it active and discarding it are three
  versions; undo of the discard puts it back active, and the first item's
  `Arc` is the same through all three.
- A track from point 1207 is labelled by its point id, a cluster from a pixel
  by its image and pixel, and a second cluster from the same pixel takes
  ` (2)`; a rename is a version and the old label then names nothing.
- Split off puts a second track on of exactly the named observations and removes
  exactly those from the first, whatever their verdicts were; an empty list, or
  every observation, is refused; each half commits on its own.
- A verdict, a sweep and an upgrade are three versions; undo retraces them in
  order and redo replays them; a document edit between two of them is a fourth,
  in its place, and undoing it leaves the bench alone.
- An undo of the committed version removes the point and restores the track to
  the pair's earlier half, with no origin.
- A run of bench steps over a clean value is clean; a commit is dirty; the
  budget charges a bench step its item and a document edit no more than today.
- A report finishing after the cursor moved lands on the observations it measured
  when the item is still there, and is discarded with one log row when it is
  not.
- Every bench operation pushes one version and writes one Action Log row; only
  the commit's row is of kind `Edit`.

**The searches**, over a small forest written from the scene's own SIFT files:
the descriptor search from a observation returns the other observations' features as
candidates with geometry that matches their `.sift` rows, and nothing from the
same image; the sweep proposes every image that sees the plane and none that
does not.

**Upload** (`scene_renderer/upload/tests.rs`): the preview buffer draws the
track-stage tracks and nothing of them reaches the base's buffers, the
additions or the mask.

**The wire** (`mcp/tests.rs`): each tool is the same call the panel makes, and
`get_bench_track` after a verdict shows the verdict under the same observation index.

No windowed `ui_basic` test for the Image Detail gestures, for the reason the
one-step edits have none. The Panels menu entry and the tab's presence are
covered by the existing layout test that walks every tab.

---

## Non-goals

- A second kind of item. The bench is shaped so one can be added; this draft
  specifies the editable track and no other.
- Evaluating two tracks in one task. One task at a time is the background rule;
  a second evaluation queues behind the first.
- A bench across nodes. The bench is one node's, and an item names that node's
  images.
- Opening a `.matches` file's clusters onto the bench. The cluster stage is that
  file's representation, so it is the natural next step, and it is not this one.
- Running an evaluation on the frame. Every one is a background task, including
  the small ones, so the panel has one path.
- ~~Editing the surfel's frame or normal by hand.~~ Superseded: the Image
  Detail panel's bench layer makes its marks handles, so the frame's **size and
  turn** are edited by dragging the outline and a sighting is placed by dragging
  its dot ([`../gui/multi-panel-image-browser.md`](../gui/multi-panel-image-browser.md)
  § "The bench layer"). The normal is still the kernels'.
- Bundle adjustment after a commit. The commit triangulates, as the one-step
  edits do, and the Edit menu's adjustment is a version of its own.
- A `sift_files` commit. The bench can be used for inspection on such a node,
  and cannot write to it.
- Persisting the bench. A save writes the document half; a commit is how bench
  work reaches the file.

## Open questions

- ~~**Where a reconstruction's `.kdf` lives.**~~ Settled:
  `<stem>-sift-index.kdf` beside the `.sfmr` it indexes, recorded in
  [`../gui/sift-index.md`](../gui/sift-index.md) and
  [`../workspace/workspace.md`](../workspace/workspace.md) § "The SIFT Index".
- **The cluster stage's images across nodes.** A observation names an image of one
  node. A workspace with two nodes loaded over the same images could in
  principle pull observations from either, and a SIFT index is per node. This
  draft holds the bench to one node and leaves the cross-node case for when a
  use turns up.
- ~~**Which observation's descriptor to search from**~~ Settled: the search
  names its observation, and the gesture is a context menu on that row, so the
  question is answered by the row the person right-clicked rather than by a
  rule. Merging the results of several observations is still open, and would be
  a step over several reports rather than a change to this one.

## Steps

1. **Done.** `sfmtool_core::bench`: the two values, and every step over them --
   `create_track`, `create_cluster`, `add_observation`, `set_verdict`,
   `apply_thresholds`, `split` and `commit`, then `evaluate` at both stages and
   `set_stage` in both directions -- as pure functions, with bindings. Filed as
   [`../core/bench/bench.md`](../core/bench/bench.md) and
   [`../core/bench/editable-track.md`](../core/bench/editable-track.md). What
   remains of the core module is what a search needs: `sweep_views`,
   `search_descriptors` and `pull_in`, which arrive with the steps below that
   need them.
2. **Done.** The bench in the history, with its items and labels; the
   evaluation and the stage change as background tasks; the Bench group in the
   Scene tree; the Track Edit panel with the item tabs, the table, the
   thresholds and the toolbar; putting a point on the bench and starting from a
   pixel; the commit with and without an origin. Filed as
   [`../gui/bench.md`](../gui/bench.md),
   [`../gui/track-edit.md`](../gui/track-edit.md) and
   [`../gui/edits/commit-track.md`](../gui/edits/commit-track.md). What remains
   of the panel is what the steps below add to it, plus the per-observation
   tile column.
3. **Done.** The constellation query in core, filed as
   [`../core/features/kdf-constellation-query.md`](../core/features/kdf-constellation-query.md);
   then `sfmtool_core::bench::search_descriptors` over it, the Scene tree's
   SIFT Index row with its *Build SIFT Index*, *Open...* and *Close Index*, and
   the search as a row's context-menu entry. Filed as
   [`../core/bench/editable-track.md`](../core/bench/editable-track.md)
   § "Searching the descriptor index",
   [`../gui/sift-index.md`](../gui/sift-index.md) and
   [`../workspace/workspace.md`](../workspace/workspace.md) § "The SIFT Index",
   which settles where a reconstruction's `.kdf` lives.
4. The view sweep, built as the geometry search: `bench::search_geometry` over
   the patch-view selector, the *Find matches by geometry* row entry, and
   `search_bench_track_geometry` on the wire. The keypoint-search switch is not
   part of it. Filed as
   [`../core/bench/editable-track.md`](../core/bench/editable-track.md)
   § "Searching by geometry" and
   [`../gui/track-edit.md`](../gui/track-edit.md).
5. Pull-in from a point and from the bench, the coherence grid, and the merging
   commit. *Split off selected observations* arrived with step 2.
6. **Done**, for the tools whose steps exist: the two creates, the three item
   tools, the two reads, the seven steps on a track, and the descriptor search
   with the two index tools it needs, and the geometry search. Filed as
   [`../gui/bench.md`](../gui/bench.md) § "The wire" and
   [`../gui/mcp-server.md`](../gui/mcp-server.md) § "The bench family". What
   remains of the wire is the pull-in tool, which arrives with the step above
   that builds it.
7. The `.matches` opener, if step 2's cluster stage earns it.
