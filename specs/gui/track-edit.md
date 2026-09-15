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
and the Scene tree group), [`edits/commit-track.md`](edits/commit-track.md) (the
Commit button's edit), [`point-track-detail.md`](point-track-detail.md) (the
view-only panel whose columns this table carries first),
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) (the value
it shows and every step it calls), [`panel-layout.md`](panel-layout.md) (its tab
and its home), [`background-tasks.md`](background-tasks.md) (where Evaluate and
the stage change run), and
[`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md)
(the searches, the tiles and the overlays still to come).

---

## The interface

The panel is [track_edit/](../../crates/sfm-explorer/src/track_edit/): `mod.rs`
holds the state, the tabs, the header, the toolbar and the sliders, and
[table.rs](../../crates/sfm-explorer/src/track_edit/table.rs) the observation
table.

```rust
pub struct TrackEdit { /* sliders, painting, row selection, thumbnails */ }

impl TrackEdit {
    pub fn new() -> Self;
    pub fn show(&mut self, ui: &mut egui::Ui, state: &AppState) -> TrackEditResponse;
    pub fn forget_recon(&mut self, id: ReconId);
}

pub struct TrackEditResponse {
    pub activate: Option<String>,
    pub discard: Option<String>,
    pub rename: Option<(String, String)>,
    pub put_selected_point_on_bench: bool,
    pub start_cluster: bool,
    pub add_observation: bool,
    pub evaluate: bool,
    pub set_stage: Option<StageKind>,
    pub apply_thresholds: Option<Thresholds>,
    pub split: Option<Vec<usize>>,
    pub commit: bool,
    pub set_verdict: Option<(usize, Verdict)>,
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
beside Image Detail, Point Track and Camera Intrinsics, as the last tab of that
leaf ([`panel-layout.md`](panel-layout.md) § "Home positions"). It is a panel
like any other: closeable, ticked in the Panels menu, saved in the layout file
under the name `track_edit`.

---

## What it shows

**The item tabs**, along the top: one per track on the bench in the order they
were put there, each by its label with its `in` count beside it and a close mark
that discards it. The active one is raised. Clicking one makes it active, which
is a step. Tabs rather than a tree, because a track is named in a word and the
point of holding several is to flick between them.

**Empty**, with no track on the bench: `No track on the bench` above the two
ways in -- *Put selected point on bench*, greyed with no point selected, and
*Start cluster here*, greyed until a pixel has been named -- and one line saying
how a pixel is named.

**The header**: the active track's label, its stage as a word, its origin as a
point index or `new`, and `N in · M candidates · K out`. Below it, the stage's
own headline: at the cluster stage the reference observation and whether a
template has been cut, and at the track stage the position and the last
triangulation's condition number, or the sentence saying nothing has
triangulated it yet.

**The toolbar**, in two rows. The first acts on the active track: *Evaluate*,
the *Stage* toggle (which names the stage it would move to), *Apply thresholds*,
*Split off N rows*, *Commit* and *Discard*. The second is the ways in and the
rename: *Put selected point on bench*, *Start cluster here*, *Add observation
here* and *Rename*. Each entry is enabled or greyed with a hover text naming
what is missing, in the style of the Image Detail menu entries -- and the
Commit button's refusal is the core commit's own sentence, asked of the very
track the button would commit, so the button and the step cannot disagree.

**The thresholds**: three sliders -- minimum ZNCC, maximum shift, maximum
keypoint uncertainty -- which are exactly the bars the painting reads. Moving
one repaints the table and changes nothing about the track; *Apply thresholds*
is what turns the painting into verdicts, in one version carrying both the bars
and the painting, since the sliders are the panel's until the button is pressed.

**The observation table**, one row per observation in index order:

| Column | Cluster stage | Track stage |
|---|---|---|
| Verdict | a three-state control, clicked to cycle `in` / `out` / `candidate`; a dot marks a verdict set by hand | same |
| Thumbnail | the image, as Point Track Detail draws it | same |
| Img, Name | as Point Track Detail | as Point Track Detail |
| ZNCC | against the reference template | leave-one-out against the consensus |
| Shift | from the seed, px | from the surfel's projection, px |
| σ_pos | the observation's own tile localizability | the same |
| Error, Angle | absent | the reprojection error and the ray angle |
| Status | the kernel's `member_status` | `localized`, or `not evaluated` |
| From | the provenance | the provenance |

A cell with nothing measured behind it reads `-`, which is what says the
difference between a number a round produced and a round that has not been run.
Each row is painted by what the sliders propose for it -- green for would-pass,
red for would-not, and the panel's faint background for a row nothing has
measured. Clicking a row selects its image, as the view-only panel's rows do,
and takes the row into the split selection; Ctrl-click or Shift-click extends
that selection. Hovering a row sets the cross-panel hover.

The verdict control is the one real widget in a row: the row rect is registered
first and the control after it, so a click that lands on the control cycles the
verdict and one anywhere else on the row selects the image.

### The way onto the bench

The Point Track Detail panel carries one new line under its hints, on an
`embedded_patches` node: *To work on this track: press "Put selected point on
bench" in the Track Edit panel.* The button's label is quoted from the one
constant that spells it, so the two cannot drift apart. That panel is otherwise
unchanged.

### The pixel the two pixel entries use

*Start cluster here* and *Add observation here* act at **the pixel the Image
Detail panel's context menu was last opened at** -- the same value the Create 3D
Point prompt opens on ([`edits/create-point.md`](edits/create-point.md)). Naming
a pixel is a gesture that panel already has, and borrowing it is what lets this
panel have the entries without a second way to point at a photograph. The
entries are greyed until an image of the node is selected and a pixel has been
named, each saying which of the two is missing. The radius a new cluster's patch
takes is the one the Create 3D Point prompt would offer: the radius the last
created point was given, or the size the node's own patches project to in that
image.

---

## Testing

[track_edit/tests.rs](../../crates/sfm-explorer/src/track_edit/tests.rs),
headless: the whole panel runs through `Context::run_ui`, so `show` really does
draw the tabs, the header, the toolbar, the sliders and every row. What the
table drew is recorded unconditionally, in row order, so the assertions read the
very table the app draws rather than a second computation of it. Covered: an
empty bench offering the ways in and drawing no rows; a row per observation in
index order; a verdict showing under the same observation index, pinned; the
sliders painting the rows, leaving a pinned verdict where it is, and the
painting matching what applying the bars then produces; the cells following the
stage the track is in; every item named in the tabs; and the sliders keeping
where they were left.

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
- **The per-observation tile.** The rendered patch tile the view-only panel
  draws, and the cluster stage's warped template beside it, are proposed in
  [`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md).
- **The searches.** *Search descriptors*, *Sweep views* and the two pull-ins are
  proposed in the same draft, as is the coherence grid under the table.
- **Editing the surfel's frame or normal by hand.** The frame is what the
  kernels fit; a wrong frame is downgraded and refit.
