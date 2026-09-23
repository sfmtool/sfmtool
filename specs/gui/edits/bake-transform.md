# Bake Transform

An action in the Scene Graph panel that takes the similarity a reconstruction
is currently **drawn** under and writes it into the reconstruction's own
numbers, then returns the node to its own frame. The picture does not move;
what changes is which side of the display boundary the numbers live on, so a
save afterwards carries the new frame. It is the one door from a node's
**display transform** into its data, and it is a version like any other edit:
listed in the Edit History panel and stepped back by `Ctrl+Z`.

Together with the 3D viewport's patch menu, which sets a display transform from
a surface in the scene, it is how a reconstruction's frame is set by eye and
then kept.

Related specs: [../scene-graph.md](../scene-graph.md) § "The transform" (the
display transform, the reframe, and the rule this operation is the exception
to), [../viewer-3d-bench-layer.md](../viewer-3d-bench-layer.md) § "The patch
menu" (the four ways a transform is set from a patch),
[../document-model.md](../document-model.md) and
[../edit-history.md](../edit-history.md) (the version it pushes, which states
all three of its halves), [../bench.md](../bench.md) (the bench half it
transforms too),
[../../cli/reconstruction/xform/xform-command.md](../../cli/reconstruction/xform/xform-command.md)
(the same similarity at the command line), [README.md](README.md) (the other
edit families).

---

## Invocation

Context menu on a **reconstruction row** of the Scene Graph tree:
`Bake Transform`, immediately below `Reset Transform` and in the same separator
group as `Align to ▸`, because the three are one subject: compute a transform,
discard it, keep it.

The entry is greyed exactly when `Reset Transform` is, on the same hover text,
*"This reconstruction is already in its own frame"*: when
`SceneNode::has_transform` is false. A busy node is refused first, on its own
sentence, as on every sibling entry that makes a version. The entry reports
itself on `TreeOutput::response` and is carried out by `AppState` after the
frame; the panels' caches for the node are then dropped, as after every bulk
edit.

Over MCP it is `bake_reconstruction_transform`, which refuses in the same words
([../mcp-server.md](../mcp-server.md)).

---

## Mechanism

```rust
impl AppState {
    /// Write `id`'s display transform into its reconstruction and return the
    /// node to its own frame, leaving the drawn scene exactly where it is.
    pub fn bake_node_transform(&mut self, id: ReconId) -> Result<(), String>;
}
```

In [display_transform.rs](../../../crates/sfm-explorer/src/display_transform.rs).
A **bulk edit**, in the shape `AppState::move_camera` has:

1. Refuse a busy node, a node that is no longer loaded, and a node whose
   transform is the identity.
2. Fold the overlay in when there is one, taking its `PointMap`; read the base
   directly when there is not.
3. Run `SfmrReconstruction::apply_se3_transform` with the node's transform, and
   put the bench's placements through the same transform
   (`display_transform::bake_bench`).
4. Push one version stating all three halves through `History::push_pair`: the
   transformed value, the transformed bench, and `Se3Transform::identity()` as
   the transform.

Step 4 is one call, and that is what makes the bake safe to undo. A version
pushed without the reset would draw the scene under the transform twice; a reset
that was not part of the version would undo into a frame that is neither the one
before the bake nor the one after. Stating the three halves at once makes both
impossible, and an undo puts back the value and the framing together.

**`apply_se3_transform` is the mechanism, and there is no second one.** It is
the one whole-reconstruction similarity in
[edit.rs](../../../crates/sfmtool-core/src/reconstruction/edit.rs): finite point
positions, camera poses, rig sensor translations, per-point normals, the patch
`u` and `v` half-vectors, points at infinity (rotation only, renormalised), the
point constraints' distances and the per-image depth statistics. It is what
`sfm xform --scale`, `--rotate`, `--translate`, `--align-to` and
`--scale-by-measurements` reach through `Se3Transform @ recon`, so the viewer
and the command line are one implementation.

**The bench's placements go through the same transform**, because the bench is
the version's second half and its placement is world geometry the value does
not hold. A bake that transformed the value alone would leave the active track's
square standing in the old frame, and the figure would jump. For every
track-stage item the centre and the track's position go through the whole
similarity, the in-plane axes are rotated and the half-extent is scaled, which is
what the point set's own patch half-vectors undergo; a track at infinity keeps
the rotation alone, for the reason a point at infinity does. A cluster-stage
item has nothing to move and keeps its `Arc`. This is the part of a bake most
easily missed, because the invariant that catches it is about the figure rather
than about the reconstruction.

**The row map is the identity**, `PointMap::Removed(Vec::new())`, chained after
the materialisation's map when there was one. It is asserted rather than read off
with `RowMap::by_scan` the way `move_camera` reads its map: a camera move
re-triangulates, so whether a point survives is an outcome that has to be
measured, while `apply_se3_transform` maps the point vector in place, keeping the
count, the order and every field it does not transform, ids included. A scan
would spend `O(observations)` re-deriving what the construction fixes.

**The depth statistics follow the scale.** A bake from `Align to…` usually
carries one, since a similarity fit is the default, and the file it would write
must not describe per-image depths of the scene it used to be. The rule, and why
the histogram counts are carried rather than recomputed, is
`apply_se3_transform`'s own
([xform-command.md](../../cli/reconstruction/xform/xform-command.md)); the bake
adds nothing to it.

---

## The version

Recorded as `Kind::Edit` with `ActionLog::record_done`, from the instant the
operation began, so the row carries the cost of the transform rather than the
cost of writing the row. The sentence names what moved and by how much, in the
vocabulary the camera move's row uses, with the scale only when it is not `1`,
and ends in the version transition every edit's row carries:

```
Baked transform of run_b: 37.4 deg, 1.284 scene units (v3 → v4)
Baked transform of run_b: 37.4 deg, 1.284 scene units, x0.982 (v3 → v4)
```

The rotation is `RotQuaternion::angle` in degrees and the length is the
translation's own. The version's label, which the Edit History panel lists, is
the same sentence without the transition. A refusal is an `ActionLog::fail`
with the reason.

The node goes dirty, since the value is a new base; a save writes it and
recomputes the content hash. The display transform, now the identity, is not
written: it never is.

---

## Testing

`crates/sfm-explorer/src/display_transform/tests.rs`, on the bench's own fixture
(the demo rewritten as `embedded_patches` with exact keypoints and one point on
the bench), from a node drawn under a similarity with a turn, a shift and a
scale of `2`. The first three are the definition of the bake being correct, and
none can be satisfied by an implementation with a column or a direction wrong.

- **The picture does not move.** `scene::world_points` and
  `scene::camera_world_centres` for every camera are the same before and after,
  to `1e-9` relative. A bake that forgot to reset the transform and one that
  reset it without transforming the value fail this in opposite directions.
- **No reprojection moves.** Every observation of every image projects to the
  same pixel to `1e-9` px across the bake: a similarity applied to points and
  poses together scales the camera-frame point and the perspective divide
  cancels it. A missed column, a rotation the wrong way round, or a scale
  applied to the points and not the camera centres show up here and nowhere
  else.
- **Undo puts the picture back, and redo takes it forward.** After a bake an
  undo returns the base `Arc`, the transform bit for bit, the world points, the
  camera centres and the bench figure to what they were, and a redo returns them
  to what they were after.
- **The bench figure does not move** across the bake, to the `f32` it is stored
  in, and the placement's half-extent has been scaled by the transform's scale.
- **An overlay is folded**: a bake after a point deletion keeps every live point
  and leaves an empty overlay, and still moves nothing on screen.
- **The row** is one `Edit` naming the node, the degrees, the scene units and
  the scale.
- **A node at the identity has nothing to bake**: refused, and no version.
- **Set to Origin after a bake is the identity to rounding.** Run it, bake, run
  it again on the same patch: the transform is the identity to `1e-12`, not
  bit for bit, because it is float arithmetic. `SceneNode::has_transform`
  compares exactly, so such a node may still report a transform and offer
  `Reset Transform` and `Bake Transform`; an epsilon there would cost more than
  that wart does.

The menu entry's availability, greyed at identity, live once a transform is set
and greyed again after the bake, is covered with the Scene panel's other
entries ([../scene-graph.md](../scene-graph.md) § "Testing"), and `ui_basic`
finds the entry in the reconstruction row's menu in a real window. The depth
statistics' own rule is tested in core
(`reconstruction/edit/tests.rs`), and `sfm xform --scale` carries it on the
Python side (`tests/xform/test_transforms.py`).

---

## Non-goals

- **A bake of a node with no transform.** Refused rather than done as a no-op,
  so a call that meant to move something says that it did not.
- **A one-shot "Save Aligned Copy…".** Bake, then `File > Save As`, is two
  clicks and leaves the intermediate state visible and undoable.
- **Saving the display transform.** A bake writes the value; the transform
  itself is never written or hashed, and reopening a file starts at the
  identity.
- **Revisiting `metadata.world_space_unit` under a scale.** It is an optional
  string naming the unit the world coordinates are in, and
  `apply_se3_transform` passes the metadata block through, so a file that said
  `"m"` still says it after a bake that doubled every length. What it should say
  after a scale is a question for `apply_se3_transform` as a whole, which
  `sfm xform --scale` has first: clearing it loses information a caller may have
  meant to keep, leaving it keeps a claim that is false, and taking a unit
  beside the transform hands the decision to callers who mostly do not know
  either. Only `--scale-by-measurements` sets it, because it knows the unit it
  scaled to.
