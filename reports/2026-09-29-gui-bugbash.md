# SfM Explorer bug bash — 2026-09-29

Bugs found by driving the SfM Explorer (release build of `main` at 7043e914)
through its MCP endpoint (`--mcp 8787`, called as JSON-RPC over HTTP), with the
emphasis on what the PRs merged between 2026-09-25 and 2026-09-28 added: Find
Nearby Tracks (#642, #641), Track View (#645, #655, #659, #660), the bench
patch and verdict tools (#615, #617, #623, #629), Clear the Bench (#643),
double-click aiming (#608, #609, #610, #613), Maintain Z-up (#614), Switch
Camera Model (#605, #607), Image to Tracks (#602), and index files.

The reconstructions come from a local collection of datasets; the paths below
are relative to it. No original file was modified. Every edit and save went to
a copy in a `bugbash-20260929/` folder beside the original (inside the
dataset's `sfmr/` directory, so the workspace still resolves), or to a
temporary directory:

| label in the viewer | copy of | kind |
|---|---|---|
| `dino` (later `dino-saved`, `dino-saved2`) | `DinoDogToyWS/sfmr/20260727-00-solve-dino_dog_toy_1-85.sfmr` | 85 images, `sift_files`, then converted to embedded patches |
| `kerry` | `KerryPark480/sfmr/20260823-01-solve-frame_1-24-clean-2b-ba-embedded.sfmr` | 48 fisheye rig images, 30 points at infinity |
| `pan` | `SeedTestCaptures/lake_union_rotation/sfmr/pan-h03-embedded.sfmr` | 8-image rotation pan, 8137 of 8734 points at infinity |
| `orphan` | the `kerry` copy, placed in a temporary directory | outside its workspace |

Severity is a judgement: **high** loses or corrupts data, **medium** blocks a
feature or gives a wrong answer, **low** is a wrong message, display or
schema detail.

## Summary

| # | Severity | Area | Bug |
|--:|---|---|---|
| 1 | high | bench / delete image | Deleting a camera image leaves bench tracks pointing at the wrong photographs; committing one writes a corrupt point |
| 2 | medium | 3D view framing | Zoom-to-fit counts the unit direction of every point at infinity as a position, so a panorama frames empty space |
| 3 | medium | Find Nearby Tracks | The points source finds nothing in a `sift_files` reconstruction, so existing points are rebuilt as new tracks |
| 4 | medium | bench / Track View | A point benched from a `sift_files` reconstruction is a dead end: every remedy the refusals name is itself refused |
| 5 | medium | bench / convert | Bench items made before Convert to Embedded Patches keep no patch frame, and re-benching the point only re-activates the stale item |
| 6 | medium | MCP schema | `translate_bench_patch` and `resize_bench_patch` mark mutually exclusive arguments as all required |
| 7 | low | `tilt_bench_patch` | The reply's `normal` is the requested one, not the one applied when the tilt stops short |
| 8 | low | `create_bench_track` | `label` is silently dropped when the point is already on the bench |
| 9 | low | Image Detail | Double-clicking a tracked feature raises Track View over Image Detail when they share a dock node |
| 10 | low | verdicts | "Handed 10 verdicts back … 0 in, 4 out" counts changed verdicts but reads as totals |
| 11 | low | Track View | A track fitted at infinity shows "condition 4606326.1" in its header |
| 12 | low | messages / validation | Extreme numbers print as hundreds of digits, and a transform that overflows to infinity is accepted |
| 13 | low | wording | "Split off 1 rows", "1 images matched, 1 candidates added" |
| 14 | low | bench labels | Labels accept newlines, tabs and NUL, and draw as multi-line rows |
| 15 | low | Camera Intrinsics panel | Integer and radian parameters print as six-decimal floats (`bspline_coeff_count 8.000000`) |
| 16 | low | Switch Camera Model | The refusal's "observed at 144.91°" disagrees with `get_camera_intrinsics`' outermost observed angle of 96.85° |

## 1. Deleting a camera image leaves bench tracks pointing at the wrong photographs (high)

`delete_camera_image` renumbers every later image down by one, but the
observations of tracks already on the bench keep their old image indexes. After
the delete, each bench observation names a different photograph from the one
it was sighted in, and nothing marks the track as stale: its evaluation reports
`current`. Committing it writes the mismatched observations into the
reconstruction.

Repro on `kerry`:

1. `create_bench_track {point: 12, label: "p12"}` — observations in images
   0, 1, 2, 3, 4, 9, 32, 43 (`frame_01`, `frame_02`, `frame_03`, …).
2. `delete_camera_image {camera_image: 1}` (`fisheye_left/frame_02.jpg`).
3. `get_bench_track {track: "p12"}` — the observations still carry indexes
   0, 1, 2, 3, 4, 9, 32, 43, which now name `frame_01`, `frame_03`,
   `frame_04`, `frame_05`, `frame_06`, `frame_11`, `frame_10` (right),
   `frame_21` (right). The per-observation reprojection errors are 0.16, 26.7,
   47.7, 12.8, 215.3, 86.1, none and 42.4 px, against 0.03–1.3 px for the
   same point in the reconstruction.
4. `commit_bench_track {track: "p12"}` — "Committed track: 8 observations in
   kerry, replacing point 12". The new point 727 has an RMS error of 61.6 px
   and one observation that does not project at all.

The committed point is undoable, but nothing warns the user; the same stale
mapping would reach Fit, a geometry search, or Image Detail's bench layer.
Either remap the bench's image indexes through the deletion (dropping
observations in the deleted image), or refuse the delete while the bench holds
tracks in images at or after it.

## 2. Zoom-to-fit frames the unit directions of points at infinity (medium)

`scene::world_points` (`crates/sfm-explorer/src/scene.rs:551`) returns
`view.point().position` for every live point. For a point at infinity that
field is a unit direction, so fitting treats it as a position about 1 unit from
the origin. The function feeds `Z` zoom-to-fit, the Scene panel's per-node Zoom
to Fit, the first-show framing and the MCP `set_view {fit}`.
`compute_scene_bounds` in `sfmtool-core` already excludes these points for the
renderer's bounds; the framing path does not.

Repro on `pan` (8137 of 8734 points at infinity):

- `set_view {fit: "pan"}` puts the target at (0.38, −0.84, 0.21), which is near
  the mean of the infinity directions. The cameras are at about
  (0.1, −1.56, 1.34) and the finite points have their median at
  (0.82, −3.28, 2.33).
- A screenshot of the 3D viewer after the fit shows only the grid and axes: no
  camera, no finite point, and no point at infinity, since those bearings are
  mostly behind the camera.
- Hiding points at infinity (`show_points_at_infinity: false`) and fitting
  again gives the identical view, because the fit does not look at what is
  drawn.

## 3. Find Nearby Tracks never finds existing points in a `sift_files` reconstruction (medium)

> _Status (2026-09-29): Done — `SfmrReconstruction::load` (and the Python
> `from_data`) now fill a `sift_files` value's inline `keypoints_xy` column
> from its verified `.sift` files when the file lacks it, and every save writes
> it, so `keypoint_xy` answers for these observations and the points source
> finds them. Branch `sfmr-load-fills-sift-keypoints`._

`nearby_points` (`crates/sfmtool-core/src/bench/nearby/points.rs`) finds
candidates through `ObservationIndex`, which reads `view.keypoint_xy(k)`. A
`sift_files` reconstruction stores no keypoint pixels in the `.sfmr`, so
`keypoint_xy` returns `None` for every observation. The points source then
returns nothing without saying so, and the tracks it should have recognized
are built again as new tracks. The spec says the points source "reads only the
reconstruction and the views' cameras", and the viewer does know these pixels:
`get_point` reports every observation's `xy` for the same points.

Repro on the original `dino` copy (`feature_source: "sift_files"`):

- `find_nearby_tracks {commit: false}` at the exact pixel of an observation of
  points 500, 5000 and 12000 (and of point 100): 22 tracks come back across
  the four queries, and none is labelled `… pt <index>`.
- The query at point 100's pixel returns one new 9-view track at
  (2.412, 3.283, −1.476). Point 100 is at (2.413, 3.286, −1.469), so the new
  track is point 100 found again.
- After Convert to Embedded Patches, the same kind of query labels existing
  points (`dino_dog_toy_09@514,893 1a pt 15385`), which confirms the cause.

A commit is refused on a `sift_files` reconstruction, so no duplicate point is
written. But `commit: false` fills the bench with copies of existing points,
and the report's layer and confidence figures are computed without the
strongest source.

## 4. A point benched from a `sift_files` reconstruction is a dead end (medium)

On the unconverted `dino`, `create_bench_track {point: 100}` succeeds and
reports the item at the track stage, but it carries no patch frame:

- The evaluation is refused with "the track carries no patch frame to read
  against; upgrade it from the cluster stage to build one".
- `fit_bench_track` gives the same refusal.
- `set_bench_track_stage {stage: "cluster"}` is refused ("the track carries no
  patch frame, so there is nothing to project into each observation's image").
- `set_bench_track_stage {stage: "track"}` answers "no effect, it is at that
  stage already".

Each refusal names a remedy that is itself refused. In Track View, Fit, Commit
and the stage button are disabled, and the Crop column is empty. Every
observation in `get_bench_track` has `pixel: null`, although `get_point` gives
all ten pixels. Either refuse the create with "convert to embedded patches
first", as `find_nearby_tracks` with `commit` already does, or build the frame
from the `.sift` keypoints.

## 5. Bench items made before Convert to Embedded Patches stay frame-less (medium)

Continuing from 4: after `convert_to_embedded_patches` succeeds ("18991 points
framed"), the item `pt3d_0bdf4852_100` still reports the no-patch-frame
refusal. `create_bench_track {point: 100}` does not rebuild it. It answers
"Made pt3d_0bdf4852_100 the active track", because staging a point that
already has a track activates that track, and the item stays frame-less. Only
`discard_bench_item` followed by a fresh `create_bench_track` gives a working
item, with ZNCCs, pixels and a placement. The conversion should rebuild or mark
the bench items it made obsolete.

## 6. The MCP schema marks mutually exclusive arguments as required (medium)

`tools/list` advertises:

- `translate_bench_patch` with required
  `[reconstruction_label, by, observation, camera_image, pixel]`
- `resize_bench_patch` with required
  `[reconstruction_label, half_length, moved_edge, observation, camera_image,
  edge, pixel]`

The descriptions say to give `by` *or* a pixel with *one of* `observation` and
`camera_image`, and `half_length` *or* `moved_edge`. The server enforces the
descriptions, not the schema: `resize_bench_patch` with only `half_length`
succeeded. A client that validates arguments against the schema cannot form
any call the server accepts. The cause is in
`crates/sfm-explorer/src/mcp/tools/catalog/bench.rs` around line 283: the
alternatives are passed in `object(optional, required)`'s second (required)
list.

## 7. `tilt_bench_patch` reports the requested normal, not the applied one (low)

On `kerry`, `tilt_bench_patch {track: "k20", normal: [0.8686, 0.4269, -0.2513]}`
(a normal facing away from the cameras) replied:

> "Tilted k20 by 77.6 degrees, stopped 80.0 degrees from
> fisheye_right/frame_04.jpg", `"normal": [0.8686, 0.4269, -0.2513]`

The placement's actual normal afterwards was (0.228, −0.465, 0.855), 102° from
the reported one. The `normal` field should be the one the patch took, since
the tilt is clamped.

## 8. `create_bench_track` drops `label` when the point is already on the bench (low)

A double-click on a point in the 3D view puts it on the bench
(viewport-navigation.md). A following
`create_bench_track {point: 9921, label: "t9921"}` then replied
`"changed": false`, with the item still named `pt3d_9197bb94_9921` and the
reply's `label` field reading "Put point 9921 on the bench as
pt3d_9197bb94_9921". That label is the text of the earlier version at the
cursor, not a description of this call. The requested label is ignored
without a word. It should rename the item, or refuse and say the point is
already on the bench under another label.

## 9. Double-clicking a tracked feature in Image Detail hides Image Detail (low)

A double-click on a tracked feature is Edit on Bench, which raises Track View.
When Image Detail and Track View are tabs in the same dock node, Track View
replaces Image Detail in front of the user in the middle of the gesture. After
that double-click, `get_window_layout` reports `image_detail.active: false` and
`track_view.active: true`. The raise should skip a panel that would cover the
one the gesture came from, or open Track View in another node.

## 10. The verdict hand-back label reads as totals but counts changes (low)

`set_bench_track_verdict {verdict: "unpin", observations: "all"}` on a 10-row
track recorded "Handed 10 verdicts back to the thresholds in
pt3d_0bdf4852_100: 0 in, 4 out". The track then had 6 in and 4 out, and
`counts` said so. "0 in" counts only the verdicts that changed. The version
label, which the Edit History panel shows, should give the totals or say
"4 turned out".

## 11. Track View shows a finite-point condition number for a track at infinity (low)

After `fit_bench_track` on `pan`'s point 0 ("at infinity along … inverse-depth
z 3.43 under the 4.00 bar"), the Track View header reads "Bearing (0.392,
−0.920, 0.019), at infinity, condition 4606326.1". The fit states a bar of
10000 for the condition number, so the number reads as a failed fit. The
condition number is the reason the track went to infinity, not a measure of
the infinite placement, and should be dropped or labelled for an at-infinity
track.

## 12. Extreme numbers print in full, and an overflowing transform is accepted (low)

- `tilt_bench_patch {normal: [1e-300, 0, 0]}` is refused as expected, but the
  message spells the number out: "(0.0000…" with about 300 zeros, 368
  characters in all.
- `set_reconstruction_transform {transform: {scale: 1e300, …}}` is accepted,
  and its version label prints the scale as a 300-digit integer.
- `translation: [1e308, 0, 0]` is accepted and labelled "inf scene units".
  `set_view {fit: null}` then frames the other nodes and leaves this one out
  without saying so, because the points' squared distances overflow.

The formatting should switch to exponent notation past a sane width. The
validator, which already refuses a zero or negative scale, could refuse a
transform whose image of the node's bounds is not finite.

## 13. Pluralization (low)

- Track View's button: "Split off 1 rows"
  (`crates/sfm-explorer/src/track_view/edit/mod.rs:423`).
- `search_bench_track_geometry`'s label: "1 images matched, 1 candidates
  added".

## 14. Bench labels accept control characters (low)

`rename_bench_item {label: "line1\nline2\ttab \u0000nul"}` is accepted; only an
all-whitespace label is refused. The Scene tree then draws the row on two
lines, and the NUL goes into the Action Log and version labels. Labels are
meant as handles an agent and a person type, so newlines and control
characters should be refused.

## 15. Camera Intrinsics panel prints integer and radian parameters as floats (low)

For an `SFMTOOL_FISHEYE` camera, the Parameters table shows
`bspline_coeff_count 8.000000`, and `bspline_theta_max 2.617505` in radians,
while the plot below it labels the same domain "150.0°". The count should
print as an integer and the domain in degrees, as the rest of the panel does.

## 16. Two different "observed" angles for the same camera (low)

On `orphan`, `switch_camera_model {camera_intrinsics_index: 1, camera_model:
"SFMTOOL_PINHOLE"}` is refused with "the camera is observed at 144.91°". The
same camera's `get_camera_intrinsics.outermost_keypoint.observed.theta_deg` is
96.85° (detected: 112.73°). The refusal measures the incidence of the
triangulated points' rays (`incidence_deg` in `switch_camera_model.rs`); the
intrinsics record measures the keypoints through the lens. Either is a valid
number, but the two sit under the same word. A user who checks the refusal
against the Camera Intrinsics panel finds no observation near 145°, which
points at a badly placed point rather than a wide lens. The message should say
"a point's ray", and ideally name the point.

## Ruled out as by design

Candidates that the specs or tool descriptions settle as intended:

- The Display panel's Points size reads 0.0: the slider is `point_size_log2`,
  and 0 is the default (`viewer_3d/hud.rs`).
- A committed track's point keeps the id `pt3d_<hash>_<old index>` at a new
  index: ids are stable identities in the base content, not indexes.
- Save-as relabels the node to the new file's stem (`specs/gui/saving.md`).
- Bench steps leave the node clean; only document edits make it dirty
  (`specs/gui/bench.md`, "Dirty is about the document half").
- Opening a file ends the solo (`specs/gui/scene-graph.md`).
- The FOV set on entering camera view is kept after leaving it
  (`specs/gui/camera-views.md`).
- A double-click on a point in the 3D view puts it on the bench and raises
  Track View (`specs/gui/viewport-navigation.md`).
- Find Nearby Tracks' points source skips points at infinity and any point
  with an observation over 2 px of reprojection error
  (`specs/core/bench/nearby-sources.md`).
- `get_camera_image`'s `reproj_error` is `null` for an `embedded_patches`
  reconstruction (`specs/gui/mcp-server.md`).
- A double-click's move of the 3D target, made while the 3D viewer is a hidden
  tab, is applied when the viewer is next drawn.

## Checked and found working

These were exercised and behaved as specified, so they are not listed above:
`set_bench_track_verdict` / `apply_bench_track_thresholds` after an unpin;
`fit_bench_track`, including on a point at infinity; `commit_bench_track`
replacing a point; undo, redo and `jump_to_version` across a commit and a
conversion; save-as to a new path, which relabels the node as specified, and
reopening the copy with identical counts; Clear the Bench from the Scene tree
context menu; `find_nearby_tracks` from Image Detail's context menu, committing
one new point and benching eight existing ones; a double-click in the 3D view
landing the orbit target exactly on the point; `set_view {bench_observation}`
entering camera view; Maintain Z-up turning off on a rolled `set_view` and
righting the view when ticked again; `set_reconstruction_transform_from_patch
{set_to_origin}` putting the patch exactly at the origin with its normal on +Z;
`bake_reconstruction_transform` carrying points and bench with it;
`resect_camera_image`, `switch_camera_model` (OPENCV_FISHEYE → SFMTOOL_FISHEYE
and → EQUIDISTANT_FISHEYE), `bundle_adjust` and `add_camera_image_to_tracks` on
the fisheye rig; `build_index_files` and its cancellation; index staleness
detection by image names and order, not only count; `close_reconstruction`
refused while a task runs on the node; truncated, non-ZIP, checksum-corrupt and
missing `.sfmr` files refused with clear messages; `rename_bench_item`
collisions and blank labels; `split_bench_track` refusing all or none.
