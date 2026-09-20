# The Track Edit panel

**Track Edit** is where a track on the bench is worked on: the list of sightings
under consideration, what has been measured about each, what the person has
decided about each, and the buttons that take it from a set of image patches to
a point in the reconstruction. It is the counterpart of the Point Track Detail
panel next to it -- that panel shows the *selected point's* committed track and
stays view-only, and this one shows the *active bench item* and is where the
editing happens.

It is the first **bench panel**. A bench holds one active item per kind and each
kind is edited in a panel of its own ([`bench.md`](bench.md)), so a gesture here
that names no item means the active track, and a second kind of item would get a
second panel rather than a mode in this one.

Related specs: [`bench.md`](bench.md) (the bench, the versions its steps push
and the Scene tree groups), [`edits/commit-track.md`](edits/commit-track.md) (the
Commit button's edit), [`point-track-detail.md`](point-track-detail.md) (the
view-only panel whose columns this table carries first),
[`multi-panel-image-browser.md`](multi-panel-image-browser.md) (the Image Detail
panel, which carries the two gestures that name a pixel and draws the active
track as its bench layer),
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) (the value
it shows and every step it calls), [`panel-layout.md`](panel-layout.md) (its tab
and its home), [`background-tasks.md`](background-tasks.md) (where Evaluate, Fit,
the stage change, the descriptor search and the index build run),
[`sift-index.md`](sift-index.md) (the `.kdf` a search queries, where it lives
and what makes one stale),
[`../core/features/kdf-constellation-query.md`](../core/features/kdf-constellation-query.md)
(the query a search runs), and
[`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md)
(the searches and the remaining overlays still to come).

---

## The interface

The panel is [track_edit/](../../crates/sfm-explorer/src/track_edit/): `mod.rs`
holds the state, the tabs, the header, the toolbar and the sliders, with the
node's SIFT index and its steps in
[sift_index.rs](../../crates/sfm-explorer/src/sift_index.rs),
[table.rs](../../crates/sfm-explorer/src/track_edit/table.rs) the observation
table, and [tile.rs](../../crates/sfm-explorer/src/track_edit/tile.rs) the
per-observation tile each row draws.

```rust
pub struct TrackEdit { /* sliders, painting, row selection, rendered tiles */ }

impl TrackEdit {
    pub fn new() -> Self;
    pub fn show(&mut self, ui: &mut egui::Ui, state: &AppState) -> TrackEditResponse;
    pub fn forget_recon(&mut self, id: ReconId);
    /// Select one row from outside the panel: what the Image Detail panel's
    /// bench layer reports a click on a mark through.
    pub(crate) fn select_row(&mut self, id: ReconId, label: &str, observation: usize);
}

pub struct TrackEditResponse {
    pub activate: Option<String>,
    pub discard: Option<String>,
    pub rename: Option<(String, String)>,
    pub put_selected_point_on_bench: bool,
    pub evaluate: Option<f64>,   // the search radius the control stands at
    pub fit: Option<f64>,        // the same, for the reading a fit ends with
    pub set_stage: Option<StageKind>,
    pub apply_thresholds: Option<Thresholds>,
    pub split: Option<Vec<usize>>,
    pub commit: bool,
    pub set_verdict: Option<(usize, Verdict)>,
    pub build_sift_index: bool,          // a row's Build/Rebuild SIFT Index to Search
    pub search_descriptors: Option<usize>,  // a row's context menu, on that observation
    pub select_image: Option<usize>,
    pub hovered_image: Option<usize>,
    pub has_pointer: bool,
}
```

### Why it is shaped this way

**The panel decides nothing.** `show` takes `&AppState` and every gesture lands
in the response; [dock.rs](../../crates/sfm-explorer/src/dock.rs) applies each
one through the `AppState` method that pushes the version. That is what every
other panel does and for the same reason: the panel holds the state immutably
while it draws, and a step needs it mutably.

**Almost no state lives here.** The bench is the node's, at its cursor, so a
step taken anywhere -- this panel, the Scene tree, an undo -- is shown here on
the next frame with nothing to keep in step. What the panel does own is what is
about *looking* rather than about the track: where the sliders stand, which rows
are selected, the thumbnails it has loaded, and the painting.

**The file chooser is the dock's, not the panel's.** *Open...* reports the
gesture and nothing else, and the dock puts up `rfd::FileDialog` and calls the
step. That is what keeps `show` a pure egui function a headless frame can run,
and it is the same split the resection's `.matches` chooser takes.

**The row selection is panel state, not a version.** It is what *Split off
selected rows* reads and nothing else; a split names its observations
explicitly, because `out` says a sighting does not belong *here* and cannot say
which of two tracks it belongs to. A change of active item clears it, since
another track's observation indexes are not these.

**The painting is `apply_thresholds` run over a copy.** Rather than a second
implementation of the same rule, the panel calls the core step with the sliders'
bars and reads the verdicts it would produce. So a row can never be painted one
way and painted another when the button is pressed, and a pinned verdict comes
back unchanged from that call, which is what leaves it alone. It is recomputed
when the track's `Arc` or the bars move, and not per frame: a copy of a track
carries its consensus bitmap.

---

## Placement

A tab, `Tab::TrackEdit`, titled **Track Edit**, whose home is the top-right node
beside Point Track and Camera Intrinsics, as the last tab of that leaf
([`panel-layout.md`](panel-layout.md) § "Home positions"). It is a panel
like any other: closeable, ticked in the Panels menu, saved in the layout file
under the name `track_edit`.

---

## What it shows

**The item tabs**, along the top: one per track on the bench in the order they
were put there, each by its label with its `in` count beside it and a close mark
that discards it. The active one is raised. Clicking one makes it active, which
is a step. Tabs rather than a tree, because a track is named in a word and the
point of holding several is to flick between them.

**Empty**, with no track on the bench: `No track on the bench` above the one way
in this panel has -- *Put selected point on bench*, greyed with no point
selected -- and one line naming the other, which is the Image Detail context
menu's *Start cluster on the bench here*, quoted from that entry's own constant
so the two cannot drift.

A third way in is outside this panel: *Edit on Bench*, on a point's context menu
in the 3D viewport ([viewport-navigation.md](viewport-navigation.md) § "The
point context menu") and at the top of a feature's context menu in Image Detail
([multi-panel-image-browser.md](multi-panel-image-browser.md) § "Image Detail:
the context menu"), and in either panel by double-clicking the point or the
feature. It is the same call *Put selected point on bench* makes, whichever of
the four gestures made it, and it raises this panel afterwards, because those
gestures are made somewhere this panel is not. A point a track already came
from activates that track rather than putting a second one on, so a gesture
repeated on one point raises the panel on the item that is already there.

**The header**: the active track's label, its stage as a word, its origin as a
point index or `new`, and `N in · M candidates · K out`. Below it, the stage's
own headline: at the cluster stage the reference observation and whether a
template has been cut, and at the track stage the coordinate and the last
triangulation's condition number, or the sentence saying nothing has
triangulated it yet.

**The word in front of the coordinate says which coordinate it is.** A track at
infinity carries a unit direction where a finite one carries a place
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Finite
points and bearings"), and the same three numbers under the wrong rule read as a
point a metre from the world origin. So the line is `Bearing (x, y, z), at
infinity` for a bearing and `Position (x, y, z)` otherwise -- *at infinity* being
the word the Point Track Detail panel's own header uses for that row. Which it is
comes from the track's own `at_infinity` and not from its surfel's `w`, so a
point put on the bench from a reconstruction with no patch frames reads as the
bearing it is
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
section "The track stage"). A fit that
crosses the boundary therefore changes the header's first word, which is how a
person sees that it crossed.

**The toolbar**, in two rows. The first acts on the active track: *Evaluate*,
*Fit*, the *Stage* toggle (which names the stage it would move to), *Apply
thresholds*, *Split off N rows*, *Duplicate*, *Commit* and *Discard*. The second is the way in and the
rename: *Put selected point on bench* and *Rename*. Each entry is enabled or
greyed with a hover text naming
what is missing, in the style of the Image Detail menu entries -- and the
refusal is the core step's own sentence, asked of the very track the button would
act on, so the button and the step cannot disagree. *Commit* asks the core
commit; *Evaluate*, *Fit* and the *Stage* toggle ask `evaluate_preconditions`,
`fit_preconditions` and `set_stage_preconditions`, which are the halves of those
steps' validation that read no photograph. A track with no surfel therefore
greys *Evaluate*, *Fit* and *Stage: track -> cluster* with *"this track has no
surfel yet; fit it first"* rather than offering three buttons whose only act
would be to decode a dozen images and fail.

**Commit leaves the point it wrote selected.** The write is
[`edits/commit-track.md`](edits/commit-track.md)'s, and the panel's own part in
it is that the point the commit produced becomes the viewer's selection --
whether it replaced a point or created one -- so the 3D viewport's track rays,
the frustums of the observing images and the Point Track Detail panel next door
are all looking at what was just written rather than at wherever the selection
had been left. The dock drops what the panels cached about the node's points in
the same breath, as it does after any other edit that moves them.

**A commit of a track the point already holds writes nothing**, and the button
is not greyed for it: the press pushes no version and records the no-effect row
*"Committed bull-nose: no effect, point 4211 already holds this track"*, with
that point selected, which is the answer a person pressing Commit on an already
committed track is asking for. The greying stays what it is because it is about
whether the step can run rather than about what it would produce, and because
the call behind it is made of the track as the bench holds it rather than of the
track seated at the cursor.

**Duplicate is how a second patch over neighbouring ground is started.** It
puts a copy of the active track on the bench and makes the copy active, so the
patch just fitted to one piece of surface can be slid to the piece beside it
rather than built again from a pixel
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Duplicating"). The copy carries the geometry, the verdicts and the
measurements and drops only the origin, which is what makes its commit create a
point rather than replace the one the original came from -- so the header of the
copy reads *new* where the original's names the point it came from.

**Evaluate and Fit are two buttons because they are two questions.**
*Evaluate* measures every observation where it sits and **moves nothing** -- no
keypoint, no position, no frame -- so a person asking whether a track is right
gets an answer that does not change the thing being asked about, and nothing is
dropped from the table by a kernel's gate. *Fit* is the step that moves it:
localize, re-triangulate, re-fuse, and then read the result back, which is why
the numbers after a fit are the numbers *Evaluate* would report. *Fit* greys
with `fit_preconditions`' own sentence -- a track stage with fewer than two `in`
observations among them -- while *Evaluate* stays available for exactly that
track, because one sighting is something to report.

**The thresholds**: four sliders, one per bar of `Thresholds` -- minimum ZNCC,
maximum shift, maximum keypoint uncertainty, minimum relative ZNCC -- so there
is no bar only the wire can move. The first three are the bars the painting
reads; the fourth is the fraction of the track's own self-agreement a sweep
candidate is scored by, carried with the others because it is one of the track's
bars and is applied in the same step. Moving a slider repaints the table and
changes nothing about the track; *Apply thresholds* is what turns the painting
into verdicts, in one version carrying both the bars and the painting, since the
sliders are the panel's until the button is pressed.

**Beside them, one control that is not a threshold**: *search px*, how far from
each observation's own pixel the next reading looks for its correlation peak, in
patch-grid px, starting at `EvaluateOptions::default`'s own radius. It stands
apart from the sliders because it is an input to the measurement rather than a
bar the painting judges by: moving it repaints nothing and changes no number
until *Evaluate* or *Fit* runs, and both carry it, so a fit's numbers and a
reading's are measured in one window.

**The sliders stand where the active track's own bars are.** A track carries the
thresholds it was last applied, and that is what the panel shows: seeded from
the track when the active item changes and again whenever a step moves that
track's bars -- *Apply thresholds* here, `apply_bench_track_thresholds` over the
wire, an undo of either. What is *not* re-seeded is a drag in progress, because
a drag moves the panel's copy and leaves the track's where it is. Sliders that
said something other than the track's bars would paint the table by a rule the
track does not hold, and hand that rule to the next press of the button.

**The observation table**, one row per observation in index order. The column
headings are drawn above the scroll area rather than as its first row, so they
stay put while the rows move under them and the bottom of a long track still
says which column each number is in.

| Column | Cluster stage | Track stage |
|---|---|---|
| Verdict | a three-state control, clicked to cycle `in` / `out` / `candidate`; a dot marks a verdict set by hand | same |
| Tile | the observation's own grid: the `R x R` samples the refinement kernel reads where it sits, at its shape | the surfel re-rendered from this observation, re-anchored where it sits -- the tile Point Track Detail draws |
| Img, Name | as Point Track Detail | as Point Track Detail |
| ZNCC | against the reference template | leave-one-out against the consensus, at the correlation peak within *search px* of the observation |
| Seed sh. | how far the refinement moved off the seed, px | how far that peak sits from the observation's own keypoint, px |
| Proj. off | absent | how far the observation's keypoint sits from the point's projection, px |
| σ_pos | the observation's own tile localizability | the same |
| Error, Angle | absent | the reprojection error and the ray angle |
| Status | the kernel's `member_status` | `walked 19 px, kept at seed` where the last fit refused to move it, `localized` where the reading scored it, the reason's own sentence where it could not, `not evaluated` where nothing has been read |
| From | the provenance | the provenance |

A cell with nothing measured behind it reads `-`, which is what says the
difference between a number a round produced and a round that has not been run.

**The two distances are two columns because they are two questions.** *Seed sh.*
is the sighting's own evidence -- the correlation would rather sit this far from
where the observation is -- and is what the `max shift px` bar paints on.
*Proj. off* is a statement about the **point**: a mis-triangulated track shows a
column of large offsets beside a column of near-zero shifts, which is the
picture that says the position is what is wrong and the sightings are not. One
column could not say that, and anchoring the shift at the projection would turn
out the very observations that would pull the point back.

**The Status cell names the refusal.** A reading drops nothing, so a row without
a ZNCC always has one of core's `Unmeasured` reasons behind it, and the cell
prints that sentence -- `it sits off the photograph`, `its ray grazes the patch`,
`its seed sits 2,483 px from the projection, beyond the 64 px bound`, `nothing
to correlate against` -- elided to its column, in place of a bare "not
localized" that says only that something happened. `not evaluated` is for the
row nothing has read at all.

**And it names the walk a fit refused.** A sighting the fit's kernels wanted to
carry further than the `max shift px` bar from its seed kept the seed
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
fit's walk is bounded by the person's bar"), and that is the one thing about the
row a person reading `localized` would get wrong -- the row *is* scored, and it
is scored where it already sat. So the walk comes first among the answers a
scored row can give: `walked 19 px, kept at seed`, the number being how far the
peak sat. Only a fit sets it, so a row that says it is a row the last fit held.

**The tile is the column the numbers are about.** A ZNCC is a number; the
picture that produced it is the thing a person can judge, which is the whole
reason the bench exists. So each row draws what its stage registers, through
the code that registers it rather than a second rendering of the same idea: at
the track stage the surfel warped into this observation's view and re-anchored
where the observation sits, by the Point Track Detail panel's own warp, so a
track on the bench and the point it came from cannot show one surface two ways;
at the cluster stage the `R x R` grid the refinement kernel samples
(`sfmtool_core::patch::cluster_refine::sample_member_grid`) at that same place
and its shape, over the cluster's own radius
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "The cluster stage's units") and on the template's resolution once one has
been cut. The radius is the cluster's from the start, so the tile is the square
the person asked for before an evaluation and the square the ZNCC beside it was
measured over after one. A row with nothing to render -- no surfel yet, nothing
saying where the observation sits, or a photograph the node's cache has not
decoded -- draws an empty frame of the same size, so the columns beside it never
shift.

**Where the observation sits is the one rule**, at either stage: the keypoint a
reading wrote, else the refined cluster position, else the seed it was proposed
at (`crate::bench::observation_site`, which is also what the mark, the reveal
and the wire read). A candidate a descriptor search has just added carries only
that seed, and the tile is the whole of what says whether the search found the
right surface -- so it is cut around the seed rather than left blank or, at the
track stage, cut around wherever the bare projection of the point happens to
land in a photograph nothing has yet tied it to. Nothing has to be evaluated for
a fresh row to show its patch.

The photographs are the node's own full-resolution cache, decoded and pyramided
once for the whole viewer, and the dock fills it for the active track's images
before the panel draws, as it does for the selected point's track. A cluster
tile is the kernel's own sampler run over that cached pyramid, so the panel
builds no pyramid of its own. A tile is a warp of a
full-resolution photograph, so the rendered tiles are kept against the track's
`Arc` and rebuilt when a step moves it: every step that moves a tile gives the
track a new `Arc`.

Each row is painted by what the sliders propose for it -- green for would-pass,
red for would-not, and the panel's faint background for a row nothing has
measured. Clicking a row selects its image, as the view-only panel's rows do,
and takes the row into the split selection; Ctrl-click or Shift-click extends
that selection. Hovering a row sets the cross-panel hover.

A row is an observation, so the click names a place in that image too: it
**reveals** the observation, and the Image Detail panel pans to it if its
current view is not showing it, leaving the zoom where it is
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "Revealing a
feature named by another panel"). The pixel is the one that panel's bench layer
draws the observation's mark at -- the track stage's keypoint, else the cluster
stage's refined position or its seed -- read through the one function every
caller uses, so the mark, the tile, the view and what the wire reports cannot
disagree about where the observation is.

The verdict control is the one real widget in a row: the row rect is registered
first and the control after it, so a click that lands on the control cycles the
verdict and one anywhere else on the row selects the image.

**Right-clicking a row** opens a context menu with one entry, and what that
entry is depends on the node's SIFT index. With a current one it is *Search for
matching features*, quoted from one constant as the Image Detail menu's entries
are. It runs the descriptor search from **that** observation
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Searching the descriptor index") as a background task, pushes one version
labelled by the report's own sentence and writes one Action Log row of kind
`Bench`; the candidates it added appear in the table with `search (N)` in their
*From* column and the header's counts move. It is a row gesture and not a
toolbar one because what a search searches from is one sighting's patch, not the
track's. It greys with the sentence saying what is missing, which on this side
of the question is an image whose `.sift` file cannot be read.

With no index beside the node, or one that is out of date, the entry is the
remedy instead: it reads *Build SIFT Index to Search* or *Rebuild SIFT Index to
Search*, starts the build and runs no search
([`sift-index.md`](sift-index.md) § "The search entry"). A person who finds a
search missing has no reason to look above the table for the cure, which is why
the panel itself names no index file and offers no chooser: that is the Scene
tree's SIFT Index row.

A row is also selected from **outside** the panel: the Image Detail panel's
bench layer draws the active track over the photograph, and clicking one of its
marks selects that observation's row here
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
layer"). The mark and the row are one observation, so the two are one gesture,
and the click replaces the row selection as a plain click on a row does.

### The SIFT index behind the search

A search needs a forest of the capture's descriptors, and that forest belongs to
the reconstruction rather than to this panel: every track on the node's bench
queries the same file, written beside the node's `.sfmr` and named after it.
Where it lives, the three states it can be in, what makes one stale and the
Scene tree row that says which are all in
[`sift-index.md`](sift-index.md).

What this panel does with it is the row entry above, and one thing besides: it
asks the node to look for its index when it draws, so a session finds what the
last one built without anyone asking for it.

### The way onto the bench

The Point Track Detail panel carries one new line under its hints, on an
`embedded_patches` node: *To work on this track: press "Put selected point on
bench" in the Track Edit panel.* The button's label is quoted from the one
constant that spells it, so the two cannot drift apart. That panel is otherwise
unchanged.

### The two gestures that name a pixel

Starting a cluster and adding a candidate sighting both act at a pixel, and the
viewer's one way to name a pixel is a right-click in the **Image Detail** panel.
So both are entries in that panel's context menu -- *Start cluster on the bench
here* and *Add observation to bench track here*
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "Image Detail:
the context menu"). They are not in this panel's toolbar: there is no selected
pixel in the viewer, so a button here would act on something the person cannot
see they have chosen.

Each is one bench step and lands in the Track Edit panel's own track the moment
it is taken. The radius a new cluster's patch takes is the node's own default
(`AppState::default_patch_radius`): the median size the node's patches project
to in that image, then the median over the whole node, then a named constant, so
a cluster starts at the scale the reconstruction already works at
there. It is a half-width in that image's pixels, and the
cluster stage's own units are keypoint-frame ones, so the step converts it
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "The cluster stage's units").

---

## Testing

[track_edit/tests.rs](../../crates/sfm-explorer/src/track_edit/tests.rs),
headless: the whole panel runs through `Context::run_ui`, so `show` really does
draw the tabs, the header, the toolbar, the sliders and every row. What the
table drew is recorded unconditionally, in row order, so the assertions read the
very table the app draws rather than a second computation of it. Covered: an
empty bench offering the way in, naming the menu entry that is the other, and
drawing no rows; the panel drawing no index row of its own; a row's context menu
carrying the search entry against a current SIFT index and the build entry under
either of its other two labels otherwise, with the build label asking for the
build and for no search; a row per observation in
index order; a verdict showing under the same observation index, pinned; the
sliders painting the rows, leaving a pinned verdict where it is, and the
painting matching what applying the bars then produces; the cells following the
stage the track is in; every row drawing its own rendered tile at both stages; a
candidate a descriptor search has just added drawing its tile cut around the seed
the warp gave it rather than around the point's own projection, with nothing
evaluated;
every item named in the tabs; the sliders keeping where they were left,
following the active track's own bars when a step moves them, and re-seating
when another item becomes active; a row seeded a long way from the point's
projection saying so in its Status cell, in the reading's own sentence, while
the rows that could be read still were; and a row click reporting both the image
it selects and the observation's pixel to reveal in it.

Two assertions are on the two lines the finite/infinity boundary changes: the
header naming a `w = 0` track's coordinate `Bearing (...)` with *at infinity*
beside it and never `Position (`, and a finite one the other way round -- read
off the strings the headless frame actually painted, since the header elides
nothing and a second formatting of the same idea could disagree with it; and the
Status cell of a row carrying `walked_px` reading `walked 19 px, kept at seed`
where the same row without the flag reads `localized`.

The index the search queries has its own module and its own tests
([`sift-index.md`](sift-index.md) § "Testing").

The Panels menu entry and the tab's presence are covered by the layout test that
walks every tab. There is no windowed `ui_basic` test, for the reason the
one-step track edits have none: what the panel decides is covered headlessly,
and the accessibility tree carries no stable node for a pixel inside an image.

---

## Non-goals

- **Showing two tracks at once.** The bench holds several and the tabs flick
  between them; the comparison is this panel beside Point Track Detail.
- **Deciding anything from a number.** The sliders propose and the person
  decides; that is the whole difference between the bench and the batch pass.
- **A second tile beside the first.** Each row draws the one tile its stage
  defines; showing the cluster stage's template *and* each member warped onto
  it side by side is proposed in
  [`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md).
- **The remaining searches.** *Sweep views* and the two pull-ins are proposed in
  the same draft, as is the coherence grid under the table.
- **Keeping the index in step with the workspace.** *Build* is asked for; the
  viewer does not watch the `.sift` files and rebuild when they change.
- **Editing the surfel's frame or normal by hand.** The frame is what the
  kernels fit; a wrong frame is downgraded and refit.
