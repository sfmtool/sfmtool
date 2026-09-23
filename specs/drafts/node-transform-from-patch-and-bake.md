# Framing the Scene on a Patch, and Baking the Result

**Status:** Draft

Amends [gui/scene-graph.md](../gui/scene-graph.md) (§ "Node Transforms and
Alignment", and the Future Directions entry that defers baking),
[gui/document-model.md](../gui/document-model.md) and
[gui/edit-history.md](../gui/edit-history.md) (a version gains a third half),
[gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) (the figure
gains a context menu),
[gui/mcp-server.md](../gui/mcp-server.md) (three new tools and one new block on
`get_scene`),
[cli/reconstruction/xform/xform-command.md](../cli/reconstruction/xform/xform-command.md)
(the `apply_se3_transform` row gains the depth statistics), and
[GLOSSARY.md](../GLOSSARY.md) (three entries). Files a new edit-family spec,
`gui/edits/bake-transform.md`.

## Purpose

When someone is looking at a reconstruction of a wall, a table or a floor, the
thing they usually want is for that surface to be the ground: flat on the
screen, square to the axes, with its middle at the origin. A reconstruction
comes out of a solve in whatever frame the first two cameras happened to fix,
which is almost never that, and the only way to change it today is to align the
whole file against another one or to re-run `sfm xform` at the command line and
open the answer again.

This proposes two things. The first is a context menu on the oriented square
that the viewer already draws for the patch being worked on, with four entries
that put the world's frame onto that square: adopt its frame entirely, tip it
level, drop its middle on the origin, or drop its middle onto the ground plane.
Each is a change to how the reconstruction is *drawn*, which touches no data and
steps back with `Ctrl+Z`. The second is a **Bake Transform** entry in the Scene
panel, which takes whatever the node is currently drawn under and writes it into
the reconstruction, so a save carries it. The picture does not move when it is
baked; what changes is which side of the display boundary the numbers live on.

Together they are a way to hand-set a reconstruction's frame by eye, against a
surface in the scene rather than against a second file, and then keep it.

## The rule this replaces

[gui/scene-graph.md](../gui/scene-graph.md) states that
[`SceneNode::transform`](../../crates/sfm-explorer/src/scene.rs) is view state
only, and its Future Directions section defers an export that would bake it,
saying such an export "would break the invariant everything here holds to, that
a node transform is view state and never touches the reconstruction or the
disk".

The replacement rule, which the standing specs take on when this ships:

> **A node transform is view state until a bake is asked for, and the bake is
> the one door between the two.** Nothing that sets a transform touches the
> reconstruction or the disk. A transform is nonetheless part of the node's
> timeline: every version records the transform in force, so undo and redo put
> the framing back along with the geometry. `Bake Transform` is the single
> operation that crosses, and it crosses by making a version like any other
> edit: listed in the Edit History panel, undone by `Ctrl+Z`, and written only
> by a save.

The invariant that survives is the useful half of the old one, that setting a
transform costs nothing and reaches no file. What is added is a named, auditable
crossing, rather than a rule that the crossing cannot exist.

### The display transform is the third half of a version

A version today is a **pair**: the value and the bench beside it, so that one
`Ctrl+Z` walks both. The transform has to join them, and the argument is the one
that put the bench there.

Consider the bake without it. The bake pushes a version holding the transformed
data and sets the node's transform to the identity. Undo restores the value and
leaves the transform alone, so the node is drawn with the untransformed data in
the identity frame: the scene snaps back to whatever frame the solve produced,
which is neither where it was before the bake nor where it was after. The
headline invariant of this whole proposal, that the bake does not move the
picture, would not survive its own undo.

So: **a version carries the display transform alongside the value and the
bench**, and the rules follow from that.

**Every push captures the transform in force**, not only the bake. A point
deletion that recorded the identity would undo into the wrong framing, which is
the same defect one layer down. This is made structural rather than a discipline
every caller has to remember: `History` holds the transform in force as a field
of its own and `push_pair` reads it, so no call site can forget.

**Setting a transform pushes a version whose value half is `None`.** That is
already the mechanism [`History::push_pair`](../../crates/sfm-explorer/src/document.rs)
describes: "`value` is `None` for a step that left the document half alone; such
a version shares its predecessor's document serial, which is what keeps a run of
bench steps over a clean value clean." A reframe is exactly such a step, so it
shares its predecessor's document serial and **the node does not go dirty**. The
bench is carried along unchanged, the point map is the empty `Removed` identity,
and the unshared bytes are what a bench step's are.

**`Ctrl+Z` after a reframe undoes the reframe, and the Edit History panel lists
reframes.** This is at most one version per menu click: these are discrete
choices from a menu, not a drag, so there is no stream of them to coalesce. The
orbit, the pan and the zoom are viewport camera state and are not versions of
anything, which is unchanged.

**Undo, redo and `jump_to_version` restore the transform of the version they
land on**, in the same step that moves the cursor.

**The transform stays out of the dirty marker, out of the content hash and out
of every save.** It is view state that the timeline remembers, not data. A save
writes the value at the cursor and the value has never heard of the transform,
so `is_dirty`, `content_xxh128`, `to_minimal` and the `*` in the tree row and
the window title all behave exactly as they do today. This is the first thing
someone will get wrong, because "in the version" reads as "in the file", and it
is not.

**A version whose value the budget released keeps its transform.** The budget
releases values, which are megabytes; a transform is eight `f64`s and stays on
the row with the label and the map. The cursor cannot land on a released version
anyway, so nothing reads it, but nothing has to special-case it either.

**`align_node` and `reset_node_transform` become version-pushing too.** This is
a change to what `Align to…` and `Reset Transform` do today, and an implementer
should not have to discover it: both currently assign the field and record an
Action Log entry, and both must instead push a version. So must the wire's
`set_reconstruction_transform`. After this, `Reset Transform` is itself undoable,
which it is not today.

**The Action Log kind stays `Scene`.** Pushing a version does not make a step an
edit. The precedent is exact and is written into
[`action_log`](../../crates/sfm-explorer/src/action_log/mod.rs)'s own `Kind`
enum: a bench step pushes a version like any other and is recorded as
`Kind::Bench`, "a kind of its own rather than an `Kind::Edit` because nothing
here touches the file: the one bench step that does is the commit, and its row
is an `Edit` like any other." Reframing is to `Bake Transform` what a bench step
is to a commit.

### What still separates the two

| | Reframing | Baking |
|---|---|---|
| What moves | where the node is drawn | the reconstruction's own numbers |
| Version pushed | one | one |
| The value half of that version | none: shares its predecessor's document serial | a whole new base |
| Dirty marker | untouched | set |
| Stepped back by | `Ctrl+Z` | `Ctrl+Z` |
| Reaches a save | no | yes |
| Content hash | unaffected | recomputed by the save |
| Action Log kind | `Scene` | `Edit` |

The two rows that carry the distinction are the third and the sixth. Everything
else about the two is now the same, which is the point: the timeline is one
timeline.

## The patch menu

### What the active patch is

The square the viewport draws is the **placement** of the **active item** on
the **selected reconstruction's bench**, at the track stage. Every word there
is load-bearing and each is already defined:

- The bench beside each node holds items, one of which may be active
  ([GLOSSARY](../GLOSSARY.md) § "The bench").
- A track-stage item carries a placement, a centre with orthonormal in-plane
  axes and a half-extent; a cluster-stage item carries none. That is exactly
  what [`bench_track::placement_of`](../../crates/sfm-explorer/src/viewer_3d/bench_track.rs)
  answers, returning `None` for a cluster and for a track nothing has framed.
- The viewport is handed the bench of
  [`AppState::selected_recon`](../../crates/sfm-explorer/src/dock.rs) and of no
  other node, so at most one patch figure is on screen at any moment.

So **"the active patch" is singular by construction**, and *which reconstruction
each action applies to* has a one-line answer: the selected one, which is the
only node whose bench figure is drawn and the only node whose transform the
figure went out through. There is no tie to break when several files are
loaded. An action asked for while nothing is active, while the active item is
at the cluster stage, or while the active track carries no placement, has no
patch to read and is not offered.

A track at infinity is refused. Its placement is a tangent square about a
bearing with no location in the world, so "the patch centre becomes the origin"
names nothing. The frame's `w == 0` is the test, the same one
[`bench_track::figure`](../../crates/sfm-explorer/src/viewer_3d/bench_track.rs)
already branches on.

### Finding it from a right-click

Two things in the viewport already want a secondary click, and they have to be
reconciled rather than stacked.

**They cannot be two popups.** Both hang off the one `egui::Response` the
viewport panel produces, and `crate::context_menu::on_secondary_click` builds
`egui::Popup::menu(response)`, whose identity comes from that response. Two
popups built from one response are one popup asked to be two things. So this is
**one menu with two possible sets of entries**, decided when the click lands,
and `show_point_menu` becomes `show_viewport_menu` with a latched target:

```rust
/// What the viewport's context menu stands on, recorded on the frame the
/// secondary click landed and read on every later frame the menu is laid out.
enum MenuTarget {
    /// The bench's active patch, on the node whose figure was drawn.
    Patch(ReconId),
    /// A 3D point the pick reported under the cursor.
    Point(PointRef),
}
```

**The patch wins.** On the frame of the click the figure's hit test runs first;
if it claims the pointer the target is `Patch` and the GPU pick is not
consulted. This is the same precedence the primary click already keeps, stated
at the call site in
[`viewer_3d/mod.rs`](../../crates/sfm-explorer/src/viewer_3d/mod.rs): "The
figure is on top, so a click one of its marks catches does not also reach the
points under it: two selections from one click would be two answers to one
gesture." A right-click over a square drawn on top of a point cloud means the
square, for the same reason.

**The latch is kept for the same reason `menu_point` is.** The entries are laid
out on frames after the click, by which time the pointer has moved off whatever
it named; the doc comment on `show_point_menu` says so, and it applies verbatim
to a patch. A secondary click that claims neither a patch nor a point clears the
latch, so no menu opens over empty space, and that behaviour is unchanged.

**What counts as hitting the patch.** `Handles::hit` answers with a *handle*,
which is the dot, a corner, an edge, an observation's circle, the normal's
segment or the arrowhead, each within its own few pixels. That is the right
reach for a drag and the wrong one for a menu: a person right-clicking "the
patch" aims at the square, not at its furniture. So the figure gains

```rust
/// Whether `pos` is anywhere on the square: on one of its handles, or inside
/// the quad its four projected corners bound.
pub(crate) fn covers(&self, pos: Pos2) -> bool;
```

built from the existing `hit` plus a point-in-quad test over
`Handles::corners`, split into the two triangles the corner order already gives
(the corners are in `OrientedPatch::boundary` order, so consecutive pairs are
edges and no re-sorting is needed). A square seen edge-on degenerates to a
segment and the interior test finds nothing, which is correct: there is nothing
to aim at, and the handles still answer.

**A busy node refuses all four**, drawn disabled with the reason on hover rather
than hidden, as every other entry that makes a version does. A reframe pushes a
version, and a version cannot be pushed onto a node a worker thread is holding.

### The four entries

Named exactly as follows, as `pub const` label strings beside
`EDIT_ON_BENCH_LABEL` and `RETRIANGULATE_POINT_LABEL` so the menu and the tests
that aim at it cannot drift:

| Entry | What it does | Hover text |
|---|---|---|
| **Set to Origin** | The patch's frame becomes the world frame: `X` along `u`, `Y` along `v`, `Z` along the outward normal, and the centre at the origin. | "Draw the scene in this patch's own frame: its centre at the origin, its normal along +Z." |
| **Align Normal to Z** | Tips the scene about the patch's centre by the shortest rotation taking the patch's normal to `+Z`. Position and heading are otherwise untouched. | "Tip the scene so this patch faces +Z, turning about the patch so it stays where it is." |
| **Translate to Origin** | Moves the scene so the patch's centre is at the origin. Orientation untouched. | "Move the scene so this patch's centre is at the origin." |
| **Translate to XY Plane** | Moves the scene along world `Z` only, so the patch's centre lands at `z = 0`. `X`, `Y` and orientation untouched. | "Drop the scene along Z so this patch's centre sits on the ground plane." |

When the normal is already `+Z`, **Translate to XY Plane** puts the patch's
plane exactly on the `XY` plane, which is what makes **Align Normal to Z**
followed by it the two-step way to lay a surface on the ground without moving
it sideways.

Each entry composes its map onto the node's transform and pushes one version
whose value half is `None`, labelled with the sentence the Action Log records.
The node does not go dirty, nothing is written, and either `Ctrl+Z` or
`Reset Transform` steps back out of it.

## What each action computes

### The patch's world frame

Every action reads the patch **in world coordinates**, which is to say after
whatever transform the node already carries. Write the node's current transform
`T0 = (R0, t0, s0)`, the placement in the node's own coordinates as `c_own`,
`u_own`, `v_own`, and:

```
c = T0.apply_to_point(c_own)          # the centre, in world space
u = R0 · u_own                        # unit: a rotation preserves length, and
v = R0 · v_own                        #       the uniform scale is dropped
n = u × v                             # = R0 · n_own, the outward normal
```

The scale is deliberately dropped from the axes. `Se3Transform` carries a
*uniform* scale, so it stretches no direction relative to another, and `u`, `v`,
`n` stay orthonormal and right-handed under it. That is the property the four
maps rest on, and it is guaranteed on the input side too: the placement's frame
is orthonormal and right-handed with `n = u × v` by
[`OrientedPatch`](../../crates/sfmtool-core/src/patch/cloud.rs)'s own contract,
and `OrientedPatch::normal` is that cross product normalized.

Let `M = [u v n]` be the 3x3 matrix with those three vectors **as columns**.
`M` is orthonormal with `det M = +1`, so `M⁻¹ = Mᵀ`, and `Mᵀ` is itself a
rotation.

### The four maps

Each action produces an `Se3Transform` `A` that acts **in world space**. All
four carry `scale = 1`: they re-frame the scene, they do not resize it.

| Action | `rotation` | `translation` | `scale` |
|---|---|---|---|
| Set to Origin | `Mᵀ` | `−Mᵀ c` | `1` |
| Align Normal to Z | `R_z` (below) | `c − R_z c` | `1` |
| Translate to Origin | identity | `−c` | `1` |
| Translate to XY Plane | identity | `(0, 0, −c_z)` | `1` |

**Set to Origin is `Mᵀ` and not `M`.** A world point decomposes as
`p = c + a·u + b·v + d·n = c + M·(a, b, d)`, so its coordinates on the patch's
axes are `(a, b, d) = Mᵀ(p − c)`, and the map that makes the patch's frame the
world frame is

```
p' = Mᵀ (p − c) = Mᵀ p + (−Mᵀ c)
```

which under `Se3Transform`'s own form `p' = s·(R p) + t` reads off as
`R = Mᵀ`, `t = −Mᵀ c`, `s = 1`. Writing `M` instead is the plausible-looking
wrong answer: `M` takes patch coordinates to world ones, which is the inverse of
what is wanted, and it produces a rotation that is wrong by exactly the
transpose while still being a valid rotation, so nothing downstream complains.
The check that catches it is `Mᵀ u = e_x` and not `M u = e_x`.

**Align Normal to Z turns about the patch's centre.** `R_z` is the shortest arc
taking `n` to `+Z`, which is `nalgebra`'s
`UnitQuaternion::rotation_between(&n, &Vector3::z())`. Rotating about `c` rather
than about the world origin gives

```
p' = c + R_z (p − c) = R_z p + (c − R_z c)
```

The reason is what a person sees. The patch under the cursor is the thing they
are looking at and have framed the viewport on; a rotation about the world
origin moves it along an arc of radius `|c|`, which for a reconstruction whose
origin sits in the first camera is most of the scene away, and the surface they
asked to level swings out of view. About its own centre, the patch stays exactly
where it is and everything else tips around it, which is the gesture the entry
is named for. The two differ by the translation `c − R_z c` and by nothing else,
so the alternative is not a different rotation, only a worse place to put it.

**The degenerate case is a normal at exactly `−Z`**, where the shortest arc is a
half turn about an axis the two vectors do not determine.
`rotation_between` answers `None` there, checked against nalgebra 0.35:
`scaled_rotation_between_axis` returns `None` both when the cross product is
below epsilon with a negative dot product and when the cosine has rounded to
`−1`. The fallback is a **half turn about the patch's own `u` axis**:
`R_z = from_axis_angle(u, π)`. It takes `n` to `+Z` as required, leaves `u`
alone, and sends `v` to `−v`, so the heading a person can see in the square is
the one that survives. This mirrors what
[`bench::steps`](../../crates/sfmtool-core/src/bench/steps.rs) already does for
a tilt through a half turn, taking the axis from the gesture's own geometry when
`rotation_between` declines to invent one.

Near-degenerate normals, a fraction of a degree off `−Z`, are not a special
case and must not be made one: `rotation_between` answers there, the answer is a
half turn about a well-determined axis, and the result is continuous in the
input in the only sense that matters, which is that the patch ends up facing
`+Z`. The axis is what swings wildly, and the axis is not what the user asked
about.

### Composing onto a transform the node already carries

`Se3Transform::compose` applies **self first, then other**: its own doc says
`composed.apply(p) == other.apply(self.apply(p))`. The node's transform maps the
node's own coordinates into world space, and `A` acts on world space, so `A`
goes second:

```rust
let next = node.transform().compose(&action);
node.history.push_transform(next, text.clone());
```

This is the same shape as
[`AppState::align_node`](../../crates/sfm-explorer/src/state/ops.rs), which
writes `fit.transform.compose(&target.transform)` because the fit lands in the
target's own coordinates and the target's transform takes those on into the
world. Getting the order backwards here is silent for a node at identity, which
is the case every manual test starts from, so it has to be tested against a node
that already carries one.

The arithmetic works out. With `A = (Mᵀ, −Mᵀc, 1)` and `T0 = (R0, t0, s0)`,
`compose` gives `C = (Mᵀ R0, Mᵀ t0 − Mᵀ c, s0)`, and applying `C` to the
placement's own centre:

```
C(c_own) = s0 (Mᵀ R0 c_own) + Mᵀ t0 − Mᵀ c
         = Mᵀ (s0 R0 c_own + t0) − Mᵀ c
         = Mᵀ c − Mᵀ c = 0
```

and `C`'s rotation applied to `u_own` is `Mᵀ R0 u_own = Mᵀ u = e_x`. The node
keeps the scale it had, which is what "re-frame, do not resize" means when the
node arrived under an `Align to…` similarity.

## The version's third half, in code

[`History`](../../crates/sfm-explorer/src/document.rs) is where the transform
moves to, because it is the only place that can capture it on every push without
every caller cooperating.

```rust
pub struct Version {
    // … serial, label, at, value, bench, document_serial, unshared_bytes
    /// The display transform in force when this version was made.
    ///
    /// View state the timeline remembers: never written, never hashed, and no
    /// part of what `is_dirty` compares. Kept whether or not the value is,
    /// like the bench, for the same reason: it is seven floats.
    pub transform: Se3Transform,
}

pub struct History {
    // … versions, cursor, steps, disk_serial
    /// The transform in force, which is always the one on the version at the
    /// cursor. The field rather than a lookup, because it is read every frame
    /// by the upload and written by exactly two paths.
    transform: Se3Transform,
}

impl History {
    /// The display transform the node is drawn under.
    pub fn transform(&self) -> &Se3Transform;

    /// Append a version that states the display transform and nothing else.
    ///
    /// The document half is untouched, so the version shares its predecessor's
    /// document serial and a clean node stays clean; the bench at the cursor is
    /// carried along; the map is the empty `Removed`, the identity.
    pub fn push_transform(&mut self, transform: Se3Transform, label: impl Into<String>)
        -> VersionSerial;
}
```

`push_pair` gains a `transform: Option<Se3Transform>` parameter in the same
idiom its `value` parameter already uses: `None` means "carry the one in force",
`Some` means "this step states it". `push`, `push_creating` and `push_bench`
pass `None` and are otherwise unchanged, so eight of the nine existing push
sites need no edit; the ninth is the bench commit, the one place outside this
module that calls `push_pair` directly, and it gains a `None`. `push_transform` passes `Some`, and so does the bake, which is the one
step that states all three halves at once. The doc comment on `push_pair`
changes from "states **both** halves" to "states as many of the three as the
step changed", and the sentence naming the commit as the only step that states
two gains the bake as the only step that states three.

`SceneNode::transform` stops being a public field and becomes
`SceneNode::transform()`, delegating to the history. That is a mechanical change
across about thirty read sites, and it removes the two write sites entirely:
`align_node` and `reset_node_transform` call `push_transform` instead of
assigning. Making the field unreachable is most of the value, because an
assignment that skipped the push is exactly the bug this section exists to
prevent, and it would be invisible until someone pressed `Ctrl+Z`.

`undo`, `redo` and the Edit History panel's jump each set
`self.transform = self.versions[self.cursor].transform` as they move the cursor.
Because they already move the cursor in one place, this is one line in each of
`History::undo` and `History::redo` and nothing in `jump_to_version`, which is
the two of them repeated.

**One consequence to state rather than hide: a reframe truncates the redo
tail.** Today a person can undo an edit, re-aim the view, and redo. After this,
re-aiming discards the versions ahead of the cursor, because that is what every
push does. It is the price of the framing being on the timeline at all, and it
is not special-cased: a rule that some pushes truncate and others do not would
be harder to predict than the one it replaced.

It is also less novel than it sounds. `History::push_bench` reaches the same
`push_pair` and truncates the same way, so a nudge on the bench already spends a
redo tail today, for the same reason and with no complaint recorded against it.
Reframing joins an existing habit rather than starting one.

## Bake Transform

### Where it sits

A flat entry in the reconstruction row's context menu
([`scene_graph/menus.rs`](../../crates/sfm-explorer/src/scene_graph/menus.rs)),
immediately below `Reset Transform` and inside the same separator group as
`Align to ▸`, because the three are one subject: compute a transform, discard
it, keep it.

Enabled exactly when
[`SceneNode::has_transform`](../../crates/sfm-explorer/src/scene.rs) is true, the
same gate `Reset Transform` carries, with the same disabled hover text pattern:
"This reconstruction is already in its own frame". A busy node is refused first,
as on every sibling entry that makes a version.

The entry reports itself on `TreeOutput::response` and is carried out by
`AppState` after the frame, like every other menu action that touches a
reconstruction.

### What it does

```rust
impl AppState {
    /// Write `id`'s display transform into its reconstruction and return the
    /// node to its own frame, leaving the drawn scene exactly where it is.
    pub fn bake_node_transform(&mut self, id: ReconId) -> Result<(), String>;
}
```

A **bulk edit**, in the shape
[`AppState::move_camera`](../../crates/sfm-explorer/src/state/edits.rs) already
has, and for the same reasons:

1. Refuse a busy node, a node that is no longer loaded, and a node whose
   transform is the identity.
2. Fold the overlay in when there is one, taking its `PointMap`; read the base
   directly when there is not.
3. Run
   [`SfmrReconstruction::apply_se3_transform`](../../crates/sfmtool-core/src/reconstruction/edit.rs)
   with the node's transform, and put the bench's placements through the same
   transform (see the implementation notes).
4. Push one version stating all three halves: the transformed value, the
   transformed bench, and `Se3Transform::identity()` as the transform.

Step 4 is one call, and that is the point. The old failure mode, a version
pushed without the reset, drew the scene under the transform twice; the new one,
a reset that is not part of the version, would undo into the wrong frame. Both
are impossible if the version states all three halves at once, which is the
whole reason `push_pair` takes the transform rather than the caller assigning it
either side of the push.

**`apply_se3_transform` is the mechanism and there is not a second one.** It is
already the one authoritative whole-reconstruction similarity: finite point
positions, camera poses, rig sensor translations, per-point normals, patch `u`
and `v` half-vectors, points at infinity (rotation only, renormalized) and the
point constraints' distances. It is what `sfm xform --scale`, `--rotate`,
`--translate`, `--align-to` and `--scale-by-measurements` all reach through
`Se3Transform @ recon`, so the GUI and the CLI are one implementation.

**The row map is the identity**, expressed as `PointMap::Removed(Vec::new())`,
chained after the materialisation's map when there was one. This is the spelling
`History::push_bench` already uses for a version that moves no row. It is
asserted here rather than read off with `RowMap::by_scan` the way `move_camera`
does, and the difference is real: `move_camera` re-triangulates, so whether a
point survives is an outcome of the call and has to be measured, while
`apply_se3_transform` is a `map` over the point vector that preserves the count,
the order and every `..pt.clone()` field including the ids. A scan here would
spend `O(observations)` re-deriving a result the operation's construction fixes.

### What the Action Log records

`Kind::Edit`, recorded with `ActionLog::record_done` from the instant the
operation began, so the row carries the cost of the transform rather than the
cost of writing the row. The sentence names what moved and by how much, in the
vocabulary the `move_camera` row already uses, with the scale stated only when
it is not `1`:

```
Baked transform of run_b: 37.4 deg, 1.284 scene units (v3 → v4)
Baked transform of run_b: 37.4 deg, 1.284 scene units, x0.982 (v3 → v4)
```

The rotation in degrees is `RotQuaternion::angle` and the translation is the
transform's own. A refusal is an `ActionLog::fail` with the reason, as the
sibling edits write theirs.

The four patch entries record `Kind::Scene`, as `align_node` and
`reset_node_transform` do, and the Edit History panel lists the version under
the same sentence:

```
Set run_b to the frame of patch tk104
Aligned run_b's patch tk104 normal to +Z
Translated run_b's patch tk104 to the origin
Translated run_b's patch tk104 to the XY plane
Reset transform of run_b
```

The active item's label is what names the patch, which is what the Bench group
in the Scene tree and Track View both show, so a person reading the log back can
find the square the sentence is about.

## Depth statistics under a scale

`apply_se3_transform` carries `image_table.depth_statistics` and
`image_table.depth_histogram_counts` through verbatim. Under a rotation and a
translation that is exactly right: a depth is measured along the camera's own
axis and the camera travels with the scene. **Under a scale it is wrong**, and a
node transform from `Align to…` carries a scale whenever the fit was a
similarity, which is the default. A bake would then write a file whose stored
per-image depths describe the scene it used to be.

This is not new with the bake. `sfm xform --scale 0.01` has the same defect
today, which is why the fix belongs in `apply_se3_transform` rather than in the
viewer.

**Every stale field is a length, and every one of them scales by exactly `s`.**
`ObservedDepthStats` holds `min_z`, `max_z`, `median_z` and `mean_z`;
`ImageDepthStats` adds `histogram_min_z` and `histogram_max_z`, all six as
`Option<f64>`. A similarity takes a camera-frame point `R_cw(p − C)` to
`s · R_cw(p − C)`, so every observed depth is multiplied by `s`, and the minimum,
maximum, median and mean of a scaled set are the scaled minimum, maximum, median
and mean. So the fix is six multiplications per image:

```rust
// A depth is a length: a similarity scales every one of them by `scale`, and
// the order statistics of a scaled set are the scaled order statistics. The
// counts below are what that leaves alone.
let scale_depth = |z: Option<f64>| z.map(|z| z * transform.scale);
```

**The histogram counts stay as they are, and that is exact rather than
approximate.** `compute_histogram` lays `num_histogram_buckets` equal buckets
between `histogram_min_z` and `histogram_max_z` and assigns a depth by
`((v − min) / (max − min) * B)`. Scaling `v`, `min` and `max` by the same `s`
leaves that ratio unchanged, so every depth lands in the bucket it was already
in. Carrying the counts and scaling the edges is therefore the *exact* answer,
not a cheap approximation of one.

**Do not reach for `recompute_depth_statistics`.** It re-derives these numbers
from the poses and points, which would also be correct, but it does a second
thing: when `point_set.has_normals` is true it overwrites **every per-point
normal** with the mean-viewing normal. A transform must not discard normals that
were refined, set on the bench or produced by an adjacency solve, and
`apply_se3_transform` is otherwise careful to rotate them rather than replace
them. The six multiplications are exact, are `O(images)` rather than
`O(observations)`, and touch nothing else.

**`metadata.world_space_unit` is a separate problem and this does not solve
it.** It is an optional string naming the unit the world coordinates are in, and
`apply_se3_transform` passes the whole metadata block through, so a file that
said `"m"` still says `"m"` after `--scale 2.0` while every length in it has
doubled. The only code that keeps it honest is
[`_scale_by_measurements.py`](../../src/sfmtool/xform/_scale_by_measurements.py),
which sets the target unit explicitly because it knows what it scaled to. A
general scale does not know: doubling a scene in metres does not produce a scene
in any named unit. Three options, none obviously right, and the question is
recorded below rather than answered: clear the unit under any scale other than
`1`, leave it and treat it as the caller's business, or take an optional unit
alongside the transform. Clearing it silently loses information a caller may
have meant to keep; leaving it keeps a claim that is now false.

## The wire

Nothing over MCP sets, resets or bakes a node transform today, and `get_scene`
reports only `transformed`, a boolean off `SceneNode::has_transform`. An agent
can neither drive any of this nor check that a human's click did what it looked
like it did.

### `get_scene` gains a `transform` block

Beside `transformed`, which stays as it is:

```jsonc
"transformed": true,                  // SceneNode::has_transform
"transform": {                        // the display transform in force
  "rotation_wxyz": [0.924, 0.383, 0.0, 0.0],
  "translation": [0.0, 0.0, -1.42],
  "scale": 1.0
}
```

The field names are the code's, per the surface's own rule that where the GUI
has no word the code's field name wins: `rotation`, `translation` and `scale`
are `Se3Transform`'s three fields, and `rotation_wxyz` carries the quaternion
component order the surface already uses elsewhere. The block is always present,
reading as the identity for a node that carries none, so an agent that wants the
numbers never has to branch on `transformed` first. `transformed` is kept
because it is the greyed-entry rule and one boolean is cheaper to poll than a
comparison against the identity that has to decide its own tolerance.

`get_history`'s version rows gain nothing. A reframe is a version like any other
and its `label` says what it was; a row already carries enough to tell it from
an edit, and adding a per-version transform block would put seven floats on
every row of a long history to answer a question `get_scene` answers about the
only version anyone is looking at.

### Three tools

| Tool | Kind | What it does |
|---|---|---|
| `set_reconstruction_transform` | write | Set one reconstruction's display transform outright, the identity included |
| `set_reconstruction_transform_from_patch` | write | Set it from the bench's active patch, in one of the four ways the viewport menu offers |
| `bake_reconstruction_transform` | write | Write the display transform into the reconstruction and return the node to its own frame, as one version |

Each is named for the part it acts on, which is the **transform**, addressed
through the reconstruction that carries it. None of them is a bench tool: they
read the bench's patch, they do not change it, so none takes the `bench` infix
that `translate_bench_patch` and its siblings carry.

The first two push a version of the node's framing and record `Kind::Scene`; the
third pushes an edit and records `Kind::Edit`. All three are refused while a
background task holds the node, like every other tool that pushes a version, and
all three carry `destructiveHint: false`, since a bake is undoable and none of
them touches a file.

`reconstruction_label` is required on all three. It is required on the first two
even though they are not edits, because they push a version and change what a
screenshot shows, and an agent that aimed one at whatever happened to be
selected would be unable to check that it landed where it meant. That is the
same argument the surface makes for the editing family.

```jsonc
set_reconstruction_transform {
  "reconstruction_label": "run_b",
  "transform": { "rotation_wxyz": [1,0,0,0], "translation": [0,0,0], "scale": 1.0 }
}

set_reconstruction_transform_from_patch {
  "reconstruction_label": "run_b",
  "mode": "align_normal_to_z"        // set_to_origin | align_normal_to_z |
                                     // translate_to_origin | translate_to_xy_plane
}

bake_reconstruction_transform { "reconstruction_label": "run_b" }
```

**The reset is `set_reconstruction_transform` with the identity, not a tool of
its own.** "Put this node's transform at this value" is one intent, and the
identity is one of its values, the way `set_solo` with `null` ends a solo rather
than there being an `end_solo`. The tool records the Action Log sentence the
menu's `Reset Transform` writes when the value it is handed is the identity, so
the log reads the same whoever asked.

**The four patch actions are one tool with a `mode`, not four tools.** They take
identical arguments, return identical replies, refuse for identical reasons, and
differ only in how much of the patch's frame the world adopts: all of it, its
normal, its centre, or its height. That is one question with four answers rather
than four questions, which is the line the surface already draws when it says a
union of representations belongs in one tool and a union of intents does not.
Four schemas would also be four places for the refusal wording and the
availability rule to drift, on a surface that is already fifty-seven writes
long and is paid for in every agent's context on every session. The cost of the
choice is that the wire spellings of the four are values rather than tool names,
so `mode` takes the menu labels snake-cased and nothing else, which keeps a
label read off the screen directly usable in a call.

**`set_reconstruction_transform_from_patch` keeps its long name**, in preference
to the shorter `frame_reconstruction_on_patch` that this document's own
vocabulary would suggest. The verb is already taken on this surface: `set_view`
is described in the tool table as "Frame the scene, look through a camera image,
or set the viewport camera outright", where framing the scene means moving the
*viewport camera* to fit it and moves no reconstruction at all. A second wire
verb `frame_` would be the same word for the two opposite halves of a
comparison a person makes constantly, which is a worse cost than four extra
syllables. The long name also says which of the two `set_…_transform` tools it
is, which is the thing an agent picking between them needs.

Refusals name what is missing rather than answering with a no-op: no
reconstruction of that label, the node has no bench item active, the active item
is at the cluster stage, the active track carries no placement, the track is at
infinity, a background task holds the node. `bake_reconstruction_transform`
additionally refuses a node whose transform is the identity, which is the tool
half of the greyed menu entry.

This closes the loop an agent needs: `get_bench_track` says where the patch is,
`set_reconstruction_transform_from_patch` moves the world onto it, `get_scene`
reports the transform that resulted, `screenshot` shows it, and
`bake_reconstruction_transform` followed by `save_reconstruction` keeps it.
`undo` steps back out of any of them.

## Glossary entries

To add under a new **The scene's frame** heading in
[GLOSSARY.md](../GLOSSARY.md), scoped to `crates/sfm-explorer/`, the viewer
specs and the wire:

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **display transform** | the similarity a node is *drawn* under, held on the version at the cursor so undo walks it, and never written to a file | `node transform` alone, `alignment`, `pose` | now that a bake exists, "the node's transform" no longer says which side of the boundary a number is on, and the adjective is the whole distinction. It is on the timeline and still not data, which is exactly what *display* has to carry. `alignment` names how one is commonly computed and not what it is, and a node set straight from a patch was never aligned to anything. `pose` is a camera's |
| **bake** | write a node's display transform into its reconstruction and return the node to its own frame, leaving the drawn scene where it is | `apply`, `commit`, `flatten`, `freeze` | the word graphics and 3D tools already use for turning a view-time transform into stored data, so it arrives meaning the right thing. `apply` is what `apply_se3_transform` does to a value and cannot also name the version-pushing step around it; `commit` is taken by the bench, where it means writing a track into the reconstruction; `freeze` suggests something is being made read-only |
| **reframe** | set a node's display transform, by any of the ways there are to set one | `align`, `snap`, `orient`, `frame` | prose needs one noun for "a version that moved the framing and no data", and this is it. `Align to…` is taken and means fitting one reconstruction onto another, a solve over correspondences. **`frame` is taken on the wire**: `set_view` frames the scene by moving the viewport camera, which moves no reconstruction, so no function or tool here may be named `frame_…`. `snap` promises a quantized result and there is none |

The four menu labels are settled and are not glossary entries: they are strings,
and the constants that hold them are the single definition the menu and its
tests share.

## Implementation notes

**The transform is written in two places and reachable from nowhere else.**
`History::push_transform` is the reframe, `push_pair` with `Some` is the bake,
and `SceneNode::transform()` is read-only. Removing the public field is most of
what makes "every version records the framing" true rather than aspirational.

**Setting a transform already has side effects that undo and redo now inherit.**
Per [gui/scene-graph.md](../gui/scene-graph.md), setting or resetting a node
transform recomputes the union scene bounds, re-derives `length_scale`,
re-uploads frustum geometry at the new per-node scale and rebuilds the track
rays. The upload phase notices by comparing what the GPU bundle last held
against what the node now has, so a transform restored by an undo is noticed by
exactly the same comparison and needs no new signalling. Camera view is left
where it is across all of this, as `Reset Transform` leaves it today.

**The bench figure is rebuilt from the new transform on the same frame.** The
figure is built at the end of `Viewer3D::show` from `bench.transform`, which is
a borrow of the node's; the menu's choice is drained by `dock.rs` after the
frame. So the frame on which an entry is clicked still draws the old figure and
the next frame draws the new one, which is the cadence every other menu-driven
change in the viewport already runs on.

**A bake must transform the bench's placements as well as the value.** The
placement is stored on the bench item, not in the reconstruction's point set,
and the bench is the second half of the node's version. A bake that transformed
the value alone would leave the active track's square standing in the old frame,
and the figure would visibly jump. The version the bake pushes therefore states
the bench too, with each track-stage placement put through the same transform:
the centre through `apply_to_point`, the `u` and `v` half-vectors rotated and
scaled exactly as the point set's patch half-vector columns are in
`apply_se3_transform`. A cluster-stage item has nothing to move. **This is the
part of the bake most likely to be missed**, because the invariant that catches
it is about the figure rather than about the reconstruction.

**A reframe's unshared bytes are a bench step's.** The value is cloned from the
cursor, so it shares the base `Arc` and `value_bytes` charges only the overlay;
the bench is shared outright. A node carrying a large overlay is charged that
overlay again per reframe, which is exactly what a run of bench steps already
costs and is not a new hazard.

**`has_transform` compares exactly.** Its doc comment says so and gives the
reason: the only writers were a fit, which never returns an exact identity by
accident, and a reset, which assigns the identity itself. The four patch actions
do not change that, but the bake-then-repeat round trip does: see the idempotence
invariant below.

## Testing

The first three are the definition of the bake being correct, and none can be
satisfied by an implementation that has a column or a direction wrong.

**The bake does not move the picture.** For a node carrying any transform,
`scene::world_points(node)` and `scene::camera_world_centres(node, i)` for every
camera index `i` are the same before and after the bake, to `1e-9` relative on
a unit-scale fixture. These are the two CPU world-space paths the framing keys
work on, and they are exactly the pair that a bake which forgot to reset the
node transform, or which reset it without transforming the value, would fail in
opposite directions.

**The bake does not move any reprojection.** For every observation of every
image, the projected pixel is unchanged to `1e-9` px across the bake. This is
the sharp one: a similarity applied to points and camera poses together leaves
`R'_cw(p' − C') = s · R_cw(p − C)`, so the perspective divide cancels `s` and
every reprojection is fixed. A missed column, a rotation applied in the wrong
direction, or a scale applied to the points but not the camera centres all show
up here and in nothing else, because they leave the cloud looking perfectly
plausible.

**Undo of a bake puts the picture back, and redo takes it forward.** After a
bake, `Ctrl+Z` returns `world_points`, `camera_world_centres` and
`bench_track::figure` to exactly what they were before it, and a redo returns
them to what they were after. This is the test the whole third-half design
exists for, and it fails outright on an implementation that pushes the version
and resets the transform as two steps.

**The bench figure does not move across the bake either.** Same tolerance,
asserted forward as well as across the undo, which is what catches a bake that
transforms the value and leaves the placement behind.

**A reframe is a version and is not a change.** After any of the four entries:
`can_undo` is true, `is_dirty` is false, the new version's `document_serial`
equals its predecessor's, the Edit History panel gains a row, and a save writes
the same bytes it would have written before. Undo puts the node back in the
frame it was in.

**Every version records the framing.** Reframe, then delete a point, then undo
twice: the framing after the first undo is the reframed one and after the second
is the original. This is the assertion that fails if an ordinary edit records
the identity instead of the transform in force.

**Set to Origin puts the patch's frame on the world's.** After it, the patch's
world centre is the origin and its world `u`, `v`, `n` are `e_x`, `e_y`, `e_z`,
to `1e-12`. Asserted on the resulting frame rather than on the transform's
components, so the test does not restate the arithmetic it is checking, and run
from a node already carrying a transform so that `Mᵀ` versus `M` and a reversed
`compose` are both caught.

**Set to Origin is idempotent in the useful sense.** Run it, bake it, run it
again on the same patch: the second action's map is the identity to `1e-12`,
because after the bake the patch's own frame *is* the world frame. Stated with a
tolerance, because it is float arithmetic and not an algebraic identity: the
node transform that results is the identity to rounding and not bit-exactly, so
`SceneNode::has_transform`, which compares exactly, may still report `true` and
`Reset Transform` and `Bake Transform` may still be offered. That is a wart
rather than a defect, and the alternative, an epsilon in `has_transform`, costs
more than it buys. The test asserts the tolerance and the wart is written down
here.

**Align Normal to Z tips about the patch.** The patch's world centre is
unchanged to `1e-12` and its world normal is `+Z`. Separately, for a patch whose
centre is far from the origin, the centre is unchanged, which is the assertion
that fails if the rotation is applied about the world origin instead.

**The antiparallel case.** A patch whose normal is exactly `(0, 0, −1)` yields a
normal of exactly `+Z`, a `u` axis unchanged, and a `v` axis negated. Built as a
direct fixture rather than reached by a rotation, so the input really is
antiparallel and `rotation_between` really does answer `None`.

**Translate to XY Plane touches `z` alone.** The patch's world centre has
`z == 0` and its `x`, `y` are unchanged; the node transform's rotation and scale
are unchanged. Together with the previous case, a patch already facing `+Z` ends
with its plane exactly on the `XY` plane.

**Each action composes onto an existing transform.** Every one of the four runs
from a node carrying a non-identity similarity with a scale other than `1`, and
the resulting scale is the one the node had, not `1`.

**Depth statistics follow a scale.** After `apply_se3_transform` with
`scale = s`, each image's six depth lengths are `s` times what they were, and
`depth_histogram_counts` is unchanged row for row. Separately, transforming by
`s` and recomputing from scratch with `recompute_depth_statistics` agree to
`1e-9` relative on the six lengths, which is what says the multiplication is the
same answer the re-derivation would give. Rotation-only and translation-only
transforms leave all of it alone. The point normals are unchanged by any of it
beyond the rotation `apply_se3_transform` already applies, which is the
assertion that fails if somebody reaches for the recompute.

**The menu.** A lib test under `Context::run_ui`: a secondary click on the
square opens the patch menu and not the point menu; a secondary click on a point
away from the square opens the point menu; a secondary click on empty space
opens neither; the four entries are present and named exactly; the entries are
absent when the bench has no active track-stage item; the latch survives the
pointer moving off the square between the click and the layout. `ui_basic` gains
the `Bake Transform` entry to its Scene-panel context-menu check, which covers
flat entries only and so covers this one.

**`Bake Transform` availability.** Greyed on a node at identity, live once it
carries a transform, greyed again after the bake. This is the same test shape
`reset_transform_is_offered_only_once_a_node_has_one` already uses, and it is
the reason the wart above is worth knowing about.

## Non-goals

- **No numeric transform editor.** Typing a rotation and a translation into a
  panel stays a future direction of
  [gui/scene-graph.md](../gui/scene-graph.md); the wire's
  `set_reconstruction_transform` is the only way to set one by number, which is
  enough for an agent and does not owe anyone a widget.
- **No "Save Aligned Copy…".** Bake then `File > Save As` is two clicks and
  leaves the intermediate state visible and undoable, which a one-shot export
  does not.
- **No bake of a node with no transform.** Refused rather than allowed as a
  no-op, so a call that was meant to move something says that it did not.
- **The display transform is still never saved.** Being on the timeline does not
  make it data: nothing writes it, nothing hashes it, and reopening a file
  starts at the identity.
- **The world-space unit is not revisited under a scale.** See the open question
  below.

## Open questions

- **What `metadata.world_space_unit` should say after a scale.** Clearing it
  whenever the scale is not `1` is the honest default and loses information;
  leaving it keeps a claim that is false; taking an optional unit alongside the
  transform pushes the decision to callers who mostly do not know either. It
  should be settled for `apply_se3_transform` as a whole rather than for the
  bake, since `sfm xform --scale` has the question first.
- **Whether the patch menu should also offer the inverse of Set to Origin**, an
  entry that puts the *viewport* on the patch rather than the patch on the
  world. It is the same geometry read the other way and would leave the
  reconstruction's frame alone, which for a quick look is what a person often
  wants. Left out because it belongs to the viewport camera's vocabulary and
  not to the node transform's, and mixing the two in one menu would make the
  four entries above harder to read rather than easier.

## Specs to update when this ships

| Spec | Change |
|---|---|
| [gui/document-model.md](../gui/document-model.md) | A version is a triple, not a pair: the value, the bench and the display transform. The transform's rules: captured on every push, stated by a reframe and by the bake, restored by undo and redo, out of the dirty comparison, out of every save, kept when the budget releases the value. |
| [gui/edit-history.md](../gui/edit-history.md) | Undo, redo and the panel's jump restore the framing; a reframe is a version the panel lists; a reframe truncates the redo tail like any other push. |
| [gui/scene-graph.md](../gui/scene-graph.md) | § "The transform": replace "view state only ... Baking a transform into a file remains `sfm xform`'s job" with the replacement rule and the contrast table, and say that setting one pushes a version. § "Align to…": `Align to…` and `Reset Transform` now push versions; add `Bake Transform` beside `Reset Transform` with its gate and its log sentence. Future Directions: delete the "Save Aligned Copy…" clause and the invariant sentence that defers it, keeping the numeric editor and multi-way alignment. § Testing: the availability test and the reframe-is-a-version test. |
| [gui/viewer-3d-bench-layer.md](../gui/viewer-3d-bench-layer.md) | The figure gains a context menu: what the four entries are, that the patch wins a contested secondary click, and that `covers` is the reach rather than `hit`. |
| [gui/edits/bake-transform.md](../gui/edits/README.md) | New file, in the shape of `resect-image.md`: invocation, the mechanism it wraps, the version label, testing, non-goals. Add its row to `gui/edits/README.md`. |
| [gui/bench.md](../gui/bench.md) | A bake puts the bench's placements through the same transform, in the version that states all three halves. |
| [gui/mcp-server.md](../gui/mcp-server.md) | Three rows in the tool table, the tool-count sentence under it, the `transform` block in `get_scene`'s reply, and a section for the three tools beside `set_reconstruction_display`. |
| [gui/action-log.md](../gui/action-log.md) | The new `Scene` and `Edit` sentences, if that spec enumerates them. |
| [cli/reconstruction/xform/xform-command.md](../cli/reconstruction/xform/xform-command.md) | The `apply_se3_transform` row gains: per-image depth statistics scale with the transform's scale, and the histogram counts are carried because equal buckets between scaled bounds hold every depth in the bucket it was in. |
| [core/reconstruction/edited-reconstruction.md](../core/reconstruction/edited-reconstruction.md) | If it enumerates the bulk edits, add the bake. |
| [GLOSSARY.md](../GLOSSARY.md) | The three entries above, under a new heading. |
