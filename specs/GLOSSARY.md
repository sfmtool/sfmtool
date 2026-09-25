# Glossary

The words this project has settled on, what each one means, and what it was
chosen over. Read it before naming a function, a wire tool, a field or a
user-facing string, and add to it whenever a naming question is settled by
argument rather than by accident.

It exists because the cheap way to settle a naming split, counting both
spellings and converging on the bigger pile, gets the answer wrong often enough
to be dangerous. A majority cannot tell a word that won on merit from a word
that won by being written first, and it cannot tell a migration in progress from
drift. Both failures have happened here, and both are recorded below.

## How to read an entry

**A glossary entry outranks a count.** Where this file names a preferred word,
that is the answer regardless of which spelling is currently more common in the
tree. The `audit-hygiene` skill's convention tallies are for finding splits this
file has not yet ruled on; they do not overturn one it has.

**Scope is part of the entry.** A word can be right in one layer and wrong in
the next, and an entry that does not say where it applies will be
over-applied. Each entry below names its scope.

**Direction of travel is part of the entry.** Some entries describe a convention
being migrated to, where the old form is still the majority. New prose follows
the entry, and a file converges as its paragraphs are rewritten for other
reasons. Nobody is expected to sweep the tree, and a one-line edit to a long
file is not an obligation to convert the rest of it.

## The bench

The vocabulary of `crates/sfmtool-core/src/bench/`, the viewer's bench layers
and the wire's bench tools.

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **patch** | the oriented square standing in the world, carrying the content the photographs see there | `surfel`, `frame` | the sense the PatchMatch family uses. The word already holds the geometry, so no second noun is needed for it |
| **placement** | the geometry alone: centre, axes, half-extent | `frame` | elsewhere in this tree `frame` means a video or rig frame, and a UI frame, so it cannot also mean this |
| **track** | the set of observations across images that are views of one point | | |
| **sighting** | one image's view of the patch, interchangeable with **observation** | | |
| **keypoint** | where one photograph sees the patch's content | | not the projection of the centre. The gap between the two is that sighting's in-plane offset |
| **bench** | where an item is held and judged before it is committed | | |
| **active item** | the one item on a bench that Track View edits and a bench tool acts on when it names none; one per bench, whether it is at the cluster or the track stage, and possibly none | a selected Scene tree row, the front tab of a dock node | "active" is the bench's word. A Scene tree row is *selected* and a dock tab is *raised*, so bench prose that says "active" means this and nothing else |
| **activate** / **deactivate** | make an item the active one / leave every item on the bench and make none active | `clear`, `unset`, `close` | the pair names what it acts on, the activation, so `deactivate_bench_item` sits beside `activate_bench_item` and needs no item argument; *discard* is the separate step that takes an item off the bench |
| **Lock** | Track View edit mode's checkbox saying what Image Detail's dot does at the track stage: ticked, it slides the patch and every sighting follows; cleared, it moves that one sighting's keypoint. A tool setting, never bench state, so it costs no version; field and accessor `lock` | `pin`, `couple`, `link` | `pin` is taken: a pinned observation is one whose verdict was set by hand, and an unlocked move pins as well, so the word cannot also name the setting. It is the user's own label for the box. Distinct from the **Move Camera lock** of camera view, which is viewport state of another panel; bench prose says *Lock* in italics or "Track View's Lock" where the two could be confused |
| **ghost outline** | Image Detail's outline of the track-stage patch as it stands, projected into an image the track has no observation in and drawn at the ghost opacity (0.8); a handle for the patch-wide edits while Track View's *Lock* is ticked, display only while it is cleared; constant `GHOST_OPACITY`, built by `Layer::ghost` | `phantom`, `preview`, `projection` | *preview* is taken by the drag's transient track and *projection* by the patch's projected centre and the *Proj. off* column. *Ghost* is the usual word in editors for a faint stand-in for the real thing, and the faintness is what stays constant: the ghost keeps its opacity whether or not it takes input, so being editable does not make it read as a sighting |
| **viewpoint** | which photograph a pixel of a gesture is in, and so which square it is read against: an observation's image, the patch re-anchored on its keypoint, or an image named as such, the patch as it stands; type `sfmtool_core::bench::Viewpoint`, wire spelling `observation` or `camera_image` | `view`, `target`, `source` | `view` is taken by the viewport and Image Detail's pan and zoom, and `target` by the wire's `TranslateTarget` and `ResizeTarget`, which say what a call wants rather than where its pixel was read. A viewpoint is the place a picture is taken from, which is the whole of what the two variants tell apart |
| **Create Track Here** | Image Detail's context-menu entry, directly above *Edit on Bench*, and its Control+Shift+Click: a track built at the pixel by the track-at-pixel cascade, put on the bench as the active item and committed as a new point. `AppState::start_create_track_at_pixel`, background operation `Create track at pixel`, wire `create_track_at_pixel` | `Build Track Here`, `Track at Pixel`, `create_bench_track_at_pixel`, `build_track_at_pixel` on the wire | the user's name for the entry. *Create* is the verb the bench's own creates use (`create_bench_track`, `create_bench_cluster`), and *Here* is what the menu's other pixel entry says (*Start cluster on the bench here*). *Build* stays core's (`build_track_at_pixel`), because core's step stops at the track and the viewer's goes on to the bench and the commit. The wire name leaves out `bench` although the tool sits in the bench family, because what it produces is a point of the reconstruction; the bench is where the track then waits, seated on that point |
| **Track View** | the panel that shows one point's track: the selected point's committed track in **view mode**, the active item in **edit mode** | `Point Track`, `Track Edit` | one panel with an explicit mode, whose **Edit** checkbox is a reading of the bench (ticked iff an item is active); wire and layout name `track_view` |

**The four gestures are named for the axis they act on**, so that no two are
synonyms: **translate** moves the centre, **resize** changes the half-length,
**spin** turns the square about its normal, and **tilt** turns the normal
itself. The pairs this replaced were unusable. `translate` and `offset` are
English synonyms that named perpendicular motions, and `rotate` is the generic
word for turning, which cannot stand beside a specific one.

**A wire tool is named for the part it acts on**, not for the item it is
addressed through. `translate_bench_patch` acts on the patch,
`sight_bench_observation` on one sighting, `commit_bench_track` on the track,
and `duplicate_bench_item` on either kind. The name that made this rule
necessary was `move_bench_track_observation`, which moves an observation and not
a track.

**Each word is said once along the path**, the enclosing namespace carrying what
it already states. A core step sits in `sfmtool_core::bench` and takes an
`&EditableTrack`, so it is `tilt_patch`. A viewer edit sits in `PatchEdit`, so
it is `PatchEdit::Tilt`. Only the wire, whose namespace is flat, spells all
three, as `tilt_bench_patch`. A name like `bench_track_frame` is three words
where the context has already supplied two.

## Files beside a reconstruction

The files the viewer builds and opens beside a node's `.sfmr`, in
`crates/sfm-explorer/`, the viewer specs and the wire.

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **index files** | the two files the viewer builds beside a reconstruction's `.sfmr` to index its capture for bench search: its SIFT index, `<stem>-sift-index.kdf`, and its cluster patches, `<stem>-cluster-patches.matches`. Built together by one operation, *Build Index Files* / *Rebuild Index Files* (`Build index files` in the Background panel), shown under one Scene tree group row *Index Files* with *Close Index Files* in its menus, and on the wire `build_index_files`, `open_index_files`, `close_index_files` and the `index_files` block. The two files keep their own names, *SIFT Index* and *Cluster Patches* | `search files`, *SIFT index* for the pair, `match files`, `sidecar files` | the name the user settled on for the pair: both files exist to index the capture so a bench search can find what it looks for, and the name says that. *SIFT index* names the `.kdf` alone (entry below), so keeping it for the operation would describe half of what it writes. `.matches` files are what `sfm match` writes into `matches/`, so *match files* would point at those. *Sidecar* says where a file sits and not what it is for |
| **cluster patches** (the file) | a node's `<stem>-cluster-patches.matches`: the SIFT index's features clustered as `sfm match --cluster` clusters them and refined as `sfm cluster-patches` refines them, with both the clusters and the cluster-patches sections. Scene tree row *Cluster Patches*, wire key `cluster_patches` | `clusters file`, `patch file` | a `.matches` with a clusters section and no refinement is a clusters file, and such a file reads stale here. *Patch file* would suggest the `.sfmr`'s own patches |

## Images and their resection

The menu an image carries in the viewer, and the correspondence sources of
*Resect Image*, in `crates/sfm-explorer/`, `sfmtool-core`'s `resect_images`, the
viewer specs and the bindings.

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **image menu** | the context menu of one image of a reconstruction (*Resect Image*, *Move Camera*, *Delete Image*), shown on a Scene tree camera image row and on an Image Browser thumbnail; one function lays it out for both, `image_menu::show`, and its entries are `ImageMenuAction` | `image row menu`, `thumbnail menu`, `image context menu` | a name for the place would make two menus of one. It is named for what it is about, an image, which is the same in both places |
| **has tracks** / **untracked** (an image, in resection) | an image with at least one track observation of the reconstruction has tracks; one with none is untracked. A cluster member in an untracked image does not count, and a cluster left with fewer than two tracked non-target images is reported as `clusters_untracked`. A track pair at infinity is a **bearing**, counted inside `track_*` as `bearing_*` | `unregistered`, `orphan`, `unsupported` | *unregistered* would say the image has no pose, and these images have one; the rule is about whether anything corroborates that pose. *Untracked* names the one fact the rule reads. A bearing stays inside `track_*` because it is a track; `bearing_*` is the part of it that constrains only the rotation |
| **tracks** / **clusters** (as resection sources) | the two sets of correspondences a resection is fit to: the reconstruction's own tracks, joined to their points' held-out positions, and the clusters of a cluster-patches file, each used as a track of its own and placed by triangulating its kept non-target members. `ResectSource::Tracks` and `ResectSource::TracksAndClusters`; report fields `track_*` and `cluster_*`; provenance `tracks` and `tracks_and_clusters` | `observations`, `stored observations`, `matches` | `observations` names only the target's half of a track and read as the whole source once clusters, which have observations too, stood beside it. `matches` named a join through a `.matches` match graph by feature index that no longer exists; the clusters are never joined to the tracks, so a name that suggests matching them would describe the wrong mechanism |
| **Add Image to Tracks** | the image menu entry, directly below *Resect Image*, that adds an image's observations of the points it sees and does not observe, nothing else moving; core `reconstruction::add_image_to_tracks`, binding `EditedReconstruction.add_image_to_tracks`, background operation `Add image to tracks`, wire `add_camera_image_to_tracks` | `Rejoin Tracks`, `Register Image`, `Add Observations`, `add_image_to_tracks` on the wire | the user's name for the entry, and it says what grows: the tracks gain the image. *Register* already means giving an image a pose, which is the resection's job; *Add Observations* hides that the observations are found, not supplied. The wire spells the image the way its siblings do (`resect_camera_image`, `delete_camera_image`), because *camera image* is the wire's noun for an image of a reconstruction, and one tool spelling it *image* would be the odd one out |
| **reference** (in adding an image to tracks) | one of a point's existing observations, rendered at its own keypoint, in the consensus the new view is searched and judged against; `ReferenceConsensus`, report fields `references` and `reference_*` | `basis`, `member` | *basis* is the localizer's word for views that congeal together and move, and these do not move; *member* is a cluster's. The references are what the new view is referred to, and stay as they are |

## The scene's frame

The vocabulary of a node's similarity transform in `crates/sfm-explorer/`, the
viewer specs and the wire.

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **display transform** | the similarity a node is *drawn* under, held on the version at the cursor so undo walks it, and never written to a file; `SceneNode::transform()` | `node transform` alone, `alignment`, `pose` | once a bake exists, "the node's transform" no longer says which side of the boundary a number is on, and the adjective is the whole distinction. It is on the timeline and still not data, which is exactly what *display* has to carry. `alignment` names how one is commonly computed and not what it is, and a node set straight from a patch was never aligned to anything. `pose` is a camera's |
| **bake** | write a node's display transform into its reconstruction and return the node to its own frame, leaving the drawn scene where it is; `Bake Transform`, `bake_node_transform`, `bake_reconstruction_transform` | `apply`, `commit`, `flatten`, `freeze` | the word graphics and 3D tools already use for turning a view-time transform into stored data, so it arrives meaning the right thing. `apply` is what `apply_se3_transform` does to a value and cannot also name the version-pushing step around it; `commit` is taken by the bench, where it means writing a track into the reconstruction; `freeze` suggests something is being made read-only |
| **reframe** | set a node's display transform, by any of the ways there are to set one: a version that moved the framing and no data; `History::push_transform`, `AppState::reframe_on_patch` | `align`, `snap`, `orient`, `frame` | prose needs one noun for such a version, and this is it. `Align to…` is taken and means fitting one reconstruction onto another, a solve over correspondences. **`frame` is taken on the wire**: `set_view` frames the scene by moving the viewport camera, which moves no reconstruction, so no function or tool here may be named `frame_…`. `snap` promises a quantized result and there is none |

The patch menu's four labels (*Set to Origin*, *Align Normal to Z*, *Translate
to Origin*, *Translate to XY Plane*) are settled and are not entries here: they
are strings, and the constants that hold them in `viewer_3d` are the single
definition the menu and the tests that aim at it share. The wire's `mode`
spells each one snake-cased.

## Optional columns

The verbs of the `sfm xform` step vocabulary and the binding keywords behind
it, for what a step does to a column rather than to a point.

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **drop** | discard an optional column and keep every row: `--drop-thumbnails`, `--drop-patch-bitmaps`, `clone_with_changes(thumbnails_y_x_rgb=None)` | `remove`, `strip`, `clear` | already the word this code used for discarding an optional column: `--localize-keypoints` drops stale patch bitmaps, and `clone_with_changes` documents `None` as the way to drop `normal_confidence`. `remove` is taken, below |
| **remove** | delete points, with their observations, and renumber the rest: `--remove-short-tracks`, `--remove-isolated` and the other point filters | `drop` | a `--remove-thumbnails` would read as the same kind of operation and is not, since it removes no row of anything. `--filter-by-*` removes points too, and `--include-*` / `--exclude-*` select images |
| **add** | fill an absent optional column from the source data: `--add-thumbnails`, `--add-patch-bitmaps` | `embed`, `render`, `restore` | drop's inverse, saying only that the column appears. `embed` collides with the `embedded_patches` feature source and `sfm embed-patches`; `render` describes how bitmaps are made but not thumbnails, which are resized |
| **display thumbnails** | the thumbnails the viewer draws for a node: the file's own column, or the rows the open builds for a file without one, from each image's `.sift` first and its photograph second; held by the node, never by a value, so none reaches a save | `thumbnails` alone, where the two could be confused | a file's thumbnails and what the viewer shows are different things once a file may carry none; naming the viewer's own keeps "the file has thumbnails" meaning what it says |
| **display patch bitmaps** | the patch bitmaps the viewer's open renders for a file with patch frames and no bitmaps; held in the value, marked `PointSet::patch_bitmaps_for_display`, and left out of every save and content hash | `synthesized bitmaps`, `display bitmaps` without `patch` | every reader of bitmaps (the bench, the edits, Track View, the renderer) looks in the value, so they live there; the mark keeps "the file has bitmaps" meaning what it says, as the node's own column does for thumbnails |
| **minimal** | the smallest file that holds the whole reconstruction: no thumbnails, no patch bitmaps, no `lineage`, no recorded absolute workspace path, `tool_options` only of the operation writing it. `--minimal`, `save(minimal=True)`, `File > Save As Minimal...`, `SfmrReconstruction::to_minimal` | `slim`, `stripped`, `lite` | one word for one definition, which `sfmtool-core` holds and every writer of such a file calls |
| **stated workspace path** | the `workspace.relative_path` a save records as the caller gives it, instead of measuring it from the output's directory to the workspace. `wspath=<path>` on `sfm xform --minimal`; the *Workspace path* field the viewer's `File > Save As Minimal...` prompts with; `workspace_path` in Rust (`SaveStamp`), Python (`save(workspace_path=…)`) and on the wire (`save_reconstruction`) | `ws_relative`, `relpath`, `workspace_rel` | one name across every surface, spelled out where there is room and abbreviated only inside the CLI's comma-separated value, where the key sits beside a path. It names the field it writes, so a reader of either spelling knows which one it is |

## Words with a boundary


**patch** (the bench) and **surfel** (the scene renderer). Both name an
`OrientedPatch`, and the split is deliberate rather than settled by a count: the
renderer is doing surfel splatting, which is what that word means in graphics,
while the bench is editing one patch in the PatchMatch sense. `surfel` is
correct in `scene_renderer/`, `track_view/`, `state*` and the specs that
describe rendering. Inside the bench it is wrong. *If this boundary is ever
removed it should be removed deliberately and in one pass, not eroded from
either side.*

## Contrast pairs that are not synonyms

Two words that look interchangeable and are not. Using either for the other
loses a distinction the code depends on.

| Pair | The distinction |
|------|-----------------|
| **retriangulate** / **direction re-estimation** | retriangulation settles a finite point from its observations. A direction re-estimation settles the **bearing** of a point at infinity, which has no depth to solve for. The specs name both in one breath, so neither word is available as a loose synonym for the other |
| **SIFT index** / descriptor index | the user-facing name for the `.kdf` structure is *SIFT index*. *Descriptor index* is not a second name for it |
| **spin** / **tilt** | a spin turns the square about its own normal and moves no sighting. A tilt turns the normal itself and rebuilds every sighting on the turned axes. In axis-angle form they look like one operation about two axes; they are not, because they do opposite things to the keypoints |
| **translate** / **offset** | not a pair but a collision. `offset` names a sighting's in-plane displacement from the patch's middle, `(a_i, b_i)`. It does **not** name a motion of the patch, which is `translate` |

## Prose

**Avoid the em-dash, and do not swap it for ` -- `.** The em-dash is a
construction rather than a character: a parenthetical break dropped into the
middle of a sentence. Replacing the character keeps the habit and changes only
its spelling, which is how prose ends up with a dashed aside every second
paragraph in a file that never once typed an em-dash.

What the sentence wants is usually something else, and usually something
different from the last time. A comma carries a light aside. A colon introduces
the thing that explains what came before. A semicolon balances two clauses that
could each stand alone. A full stop is right when the aside was really a second
thought, and parentheses when it is genuinely an aside. Often the break was not
earning its place and the sentence reads better rewritten without it. Vary these
rather than settling on one, because a single substitution repeated is the tic
again under a new name. A real dash remains available where a sentence genuinely
calls for one; it should be occasional.

*Direction of travel:* em-dash is still the majority across `specs/` by roughly
seven to one. A convention tally on this reports the wrong majority, which is
the point of recording it here. Where an em-dash is the separator in a list
whose every sibling item uses one, match the list; the entry is about prose, not
about a file's formatting furniture.

Specs are **present tense** and describe what the code *is*. A change proposal
lives in [drafts/](drafts/README.md) and is converted before filing. That rule
and the rest of the spec contract are in [README.md](README.md), not here.

## Adding an entry

Add one when a naming question is settled by argument. Record the word, what it
means, what it was chosen over, and above all **why**, since the reason is the
part that lets the next person decide a case this table does not list. Where the
answer differs by layer, say so. Where the tree has not caught up yet, say that
too, so an audit does not read the lag as the answer.
