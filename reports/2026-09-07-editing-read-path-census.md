# SfM Explorer reconstruction read-path census

_Snapshot: 2026-09-07, `crates/sfm-explorer/src` at `main` after #403._

This is step 1 of the plan in
[`specs/drafts/sfm-explorer-editing.md`](../specs/drafts/sfm-explorer-editing.md),
the "census" half: every place the viewer reads a `SfmrReconstruction`, with the
classification the overlay design in
[`specs/drafts/sfm-explorer-editing-overlay.md`](../specs/drafts/sfm-explorer-editing-overlay.md)
turns on. That draft asserts "the design holds if the per-frame readers are all
per-point or per-column, which is what the rendering path is today" and leaves as
open "how many readers in the viewer are of the first kind, and whether any hot
per-frame path in it is of the second". This report answers both, so that the
materialisation policy, the history budget and the base-identity upload can be
decided against a list rather than an impression. It also records what
`needs_upload` and `transform_epoch` do today, since the umbrella draft's Part 1
replaces both. Nothing here proposes a change; it is a survey of the code as it
stands.

Test modules (`*/tests.rs`, `test_support.rs`) are excluded throughout: they read
reconstructions the same ways the code they exercise does, and they never run in
the frame loop.

## Classification

- **per-point** -- resolves one point index, or the rows of one point's track, or
  one image's rows. Could go through an overlay accessor (base index below the
  base count and not deleted, else addition index) with no materialisation.
- **whole** -- walks all points, all tracks, or all of a point column, and so
  needs a materialised plain reconstruction (or an equivalent iterator over base
  minus deleted plus additions).
- **per-image / image-table only** -- reads `images`, `cameras`,
  `rig_frame_data`, `depth_statistics`, `thumbnails_y_x_rgb` or `workspace_dir`
  only. Untouched by point edits, since the overlay never changes the image
  table.

Two annotations recur:

- _(count only)_ -- a **whole** read that only asks for a length or a metadata
  count. Cheap to answer over an overlay without materialising (base count minus
  deleted plus additions), so it is not a materialisation forcer even though it
  is not per-point.
- _(probe)_ -- an O(1) presence check on an optional column
  (`feature_indexes().is_some()`, `patch_u_halfvec_xyz.is_some()`,
  `keypoints_xy()`). Formally a point-column read; costs nothing.

"Hot" means the site runs on every frame in which its panel is visible; "event"
means it runs on a selection change, a menu action, an MCP tool call, a cache
miss, or a load.

## Read sites

### `app.rs` -- the frame's upload phase

| Site | Reads | When | Class |
|---|---|---|---|
| [`app.rs:310`](../crates/sfm-explorer/src/app.rs) `upload_points(device, id, recon)` | whole point column | event (`needs_upload`) | whole |
| `app.rs:311` `upload_thumbnails(…, recon)` | `images`, `thumbnails_y_x_rgb` | event (`needs_upload`) | per-image |
| `app.rs:313` `upload_patches(…, recon)` | `points`, `patch_u/v_halfvec_xyz`, `patch_bitmaps_y_x_rgba` | event (`needs_upload`) | whole |
| `app.rs:394` `upload_frustums(…, &node.recon, …)` | `images`, `cameras` | hot-conditional (`geometry_changed`) | per-image |
| `app.rs:410` `node.recon.images.len()` | `images` | hot-conditional (`colors_changed`) | per-image |
| `app.rs:436` `point.index() < node.recon.points.len()` | `points.len()` | event (selection/transform change) | whole (count only) |
| `app.rs:442` `recon.feature_indexes().is_some()` | observation column | event | per-point (probe) |
| `app.rs:443` `recon.observations_for_point(point_idx)` | one track | event | per-point |
| `app.rs:445` `recon.max_track_feature_index[img_idx]` | one image's row | event | per-image |
| `app.rs:454` `upload_track_rays(device, recon, point, …)` | one track (see `upload/track_rays.rs`) | event | per-point |

### `dock.rs` -- panel bodies, one per visible tab, every frame

| Site | Reads | When | Class |
|---|---|---|---|
| [`dock.rs:206`](../crates/sfm-explorer/src/dock.rs) `recon.images.len()` | `images` | hot | per-image |
| `dock.rs:304` `recon.images.get(idx)?.camera_index` | `images` | hot | per-image |
| `dock.rs:307` `recon.cameras.get(camera_index)?` | `cameras` | hot | per-image |
| `dock.rs:334`, `:337` `recon.max_track_feature_index[idx]` | one image's row | hot | per-image |
| `dock.rs:351` `recon.feature_indexes()?` | observation column | hot | per-point (probe) |
| `dock.rs:422` `recon.feature_indexes().is_some()` | observation column | hot | per-point (probe) |
| `dock.rs:424` `pt_idx < recon.points.len()` | `points.len()` | hot | whole (count only) |
| `dock.rs:425` `recon.track_image_indices(pt_idx)` | one track | hot | per-point |
| `dock.rs:426` `recon.max_track_feature_index[img_idx]` | one image's row | hot | per-image |
| `dock.rs:442` `recon.patch_u_halfvec_xyz.is_some()` | patch column | hot | per-point (probe) |
| `dock.rs:444` `pt_idx < recon.points.len()` | `points.len()` | hot | whole (count only) |
| `dock.rs:445` `recon.track_image_indices(pt_idx)` | one track | hot | per-point |
| `dock.rs:835`, `:838` `compute_track_images` | `points.len()`, one track | hot (also from `app.rs:404`) | per-point |
| `dock.rs:853`, `:856` `compute_hover_track_images` | `points.len()`, one track | hot | per-point |

### `goto_point.rs`

| Site | Reads | When | Class |
|---|---|---|---|
| [`goto_point.rs:124`](../crates/sfm-explorer/src/goto_point.rs) `node.recon.points.len()` | `points.len()` | event (dialog submit) | whole (count only) |
| `goto_point.rs:162` `node.recon.content_hash.content_xxh128` | metadata | event | whole (metadata) |
| `goto_point.rs:176` `points.len()`, `point_id(&node.recon, idx)` | `points.len()`, metadata | event (dialog open) | whole (count only) |

### `image_browser.rs` -- the Image Browser panel

| Site | Reads | When | Class |
|---|---|---|---|
| [`image_browser.rs:45`](../crates/sfm-explorer/src/image_browser.rs) `recon.images.get(index)` | `images` | hot | per-image |
| `image_browser.rs:208` `recon.images.len()` | `images` | hot | per-image |
| `image_browser.rs:254` `recon.cameras[recon.images[i].camera_index]` | `images`, `cameras` | hot (per visible strip cell) | per-image |
| `image_browser.rs:801`--`:814` `build_barcode` | `thumbnails_y_x_rgb`, `images` | event (image count changed) | per-image |
| `image_browser.rs:842` `load_thumbnail` | `thumbnails_y_x_rgb` | event (texture cache miss) | per-image |

### `image_detail/` -- the Image Detail panel

| Site | Reads | When | Class |
|---|---|---|---|
| [`image_detail/mod.rs:382`](../crates/sfm-explorer/src/image_detail/mod.rs)`--:386` `images`, `cameras.get` | `images`, `cameras` | hot | per-image |
| `image_detail/mod.rs:464` `feature_indexes().is_none()` | observation column | event (overlay rebuild) | per-point (probe) |
| `image_detail/mod.rs:485` `recon.image_feature_to_point[img_idx]` | one image's map | event | per-image |
| `image_detail/mod.rs:550` `feature_indexes().is_none()` | observation column | event | per-point (probe) |
| `image_detail/mod.rs:602` `recon.image_feature_to_point[img_idx]` | one image's map | event | per-image |
| `image_detail/mod.rs:718`--`:744` `embedded_image_features` | `keypoints_xy`, `points.len()`, `observation_offsets`, every track, `observation_affine_shape` | event (overlay rebuild, `embedded_patches` only) | **whole** |
| `image_detail/mod.rs:749`--`:777` `populate_feature_diagnostics` | per-feature point diagnostics | event | per-point (many) |
| `image_detail/mod.rs:781`--`:804` `compute_max_track_angle_deg` | `points.get`, one track, `images.get` | event | per-point |
| [`image_detail/overlay.rs:129`](../crates/sfm-explorer/src/image_detail/overlay.rs) `recon.points.get(...)` (`ReprojError`) | one point's error | hot (draw) | per-point |
| `image_detail/overlay.rs:151` `recon.observation_counts.get(...)` (`TrackLength`) | one point's count | hot (draw) | per-point |
| `image_detail/overlay.rs:261`, `:263` `points.get`, `observation_counts` | one point | hot (draw) | per-point |
| `image_detail/overlay.rs:535` `recon.points.get(feature.point_index)` | one point | hot (tooltip) | per-point |
| `image_detail/overlay.rs:601` `observation_counts` | one point | hot (tooltip) | per-point |

`embedded_image_features` is the only whole-reconstruction walk in a panel: it
has no per-image keypoint index to go through (`image_feature_to_point` is empty
for `embedded_patches`), so it scans every point's track looking for rows landing
in one image. It runs on overlay rebuild, not per frame.

### `intrinsics_detail/`

| Site | Reads | When | Class |
|---|---|---|---|
| [`intrinsics_detail/extrinsics.rs:66`](../crates/sfm-explorer/src/intrinsics_detail/extrinsics.rs) `node.recon.images[index]` | `images` | hot | per-image |
| `intrinsics_detail/extrinsics.rs:219` `recon.rig_frame_data` | rig table | hot | per-image |
| [`intrinsics_detail/header.rs:32`](../crates/sfm-explorer/src/intrinsics_detail/header.rs) `.images` | `images` | hot | per-image |
| [`intrinsics_detail/mod.rs:120`](../crates/sfm-explorer/src/intrinsics_detail/mod.rs)`--:138` `cameras.len()`, `cameras[index]`, `images` | `cameras`, `images` | hot | per-image |

### `mcp/` -- tool calls, all event-driven

| Site | Reads | When | Class |
|---|---|---|---|
| [`mcp/mod.rs:630`](../crates/sfm-explorer/src/mcp/mod.rs) `node.recon.images.get(...)` | `images` | event | per-image |
| `mcp/mod.rs:691`--`:702` `images.len()`, `.images` | `images` | event | per-image |
| `mcp/mod.rs:726`--`:730` `cameras.len()` | `cameras` | event | per-image |
| [`mcp/read.rs:112`](../crates/sfm-explorer/src/mcp/read.rs) `recon.images.len()` | `images` | event | per-image |
| `mcp/read.rs:139`, `:140` `images[index]`, `cameras[...]` | `images`, `cameras` | event | per-image |
| `mcp/read.rs:203`, `:205` `cameras[index]`, `.images` | `cameras`, `images` | event | per-image |
| `mcp/read.rs:235`--`:247` `points[i]`, `observation_offsets[i]`, `feature_indexes()`, `keypoints_xy()`, one track, `images`, `cameras` | one point | event | per-point |
| `mcp/read.rs:291`--`:302` `feature_indexes()`, `track_image_indices`, `max_track_feature_index` | one track | event | per-point |
| [`mcp/render.rs:67`](../crates/sfm-explorer/src/mcp/render.rs)`--:74` `points.len()`, `metadata.infinity_point_count`, `images.len()`, `cameras.len()`, `tracks.len()` | counts | event (`get_scene`) | whole (count only) |
| `mcp/render.rs:124`, `:187`, `:199` `images.get`, `images[index]` | `images` | event | per-image |
| `mcp/render.rs:237`--`:244` `observations_per_image` -- one pass over `recon.tracks` | every observation | event (`list_camera_images`) | **whole** |
| [`mcp/view.rs:65`](../crates/sfm-explorer/src/mcp/view.rs) `node.recon.images[...]` | `images` | event | per-image |

### `metrics.rs`

| Site | Reads | When | Class |
|---|---|---|---|
| [`metrics.rs:93`](../crates/sfm-explorer/src/metrics.rs)`--:115` `compute_point_diagnostics`: `points.get`, one track, `images.get`, `cameras[...]` | one point | event (overlay rebuild, Point Track prepare) | per-point |

The `sfmtool_core::reconstruction::triangulation::{triangulate_batch,
depth_uncertainty_batch}` calls under it take plain slices the viewer builds from
one point's track, not `&SfmrReconstruction`.

### `point_track_detail/` -- the Point Track panel

| Site | Reads | When | Class |
|---|---|---|---|
| [`point_track_detail/mod.rs:176`](../crates/sfm-explorer/src/point_track_detail/mod.rs) `idx < recon.points.len()` | `points.len()` | hot | whole (count only) |
| `point_track_detail/mod.rs:193` `recon.content_hash.content_xxh128` | metadata | event (selection change) | whole (metadata) |
| `point_track_detail/mod.rs:201` `&recon.points[point_idx]` | one point | hot | per-point |
| [`point_track_detail/header.rs:27`](../crates/sfm-explorer/src/point_track_detail/header.rs) `recon.observation_counts[point_idx]` | one point | hot | per-point |
| [`point_track_detail/prepare.rs:46`](../crates/sfm-explorer/src/point_track_detail/prepare.rs)`--:66` `points[i]`, `feature_indexes()`, `keypoints_xy()`, `observation_offsets[i]`, one track, `images`, `cameras` | one point | event (selection change) | per-point |
| [`point_track_detail/patch.rs:64`](../crates/sfm-explorer/src/point_track_detail/patch.rs), `:65` `images[img_idx]`, `cameras[...]` | `images`, `cameras` | hot | per-image |
| `point_track_detail/patch.rs:107` `recon.keypoints_xy()?` | observation column | hot | per-point (probe) |
| `point_track_detail/patch.rs:142`--`:164` `patch_u/v_halfvec_xyz`, `points[point_idx]` | one point's frame | hot | per-point |
| `point_track_detail/patch.rs:181` `patch_bitmaps_y_x_rgba` at `point_idx` | one point's bitmap | hot | per-point |
| [`point_track_detail/table.rs:371`](../crates/sfm-explorer/src/point_track_detail/table.rs), `:372` `images[img_idx]`, `cameras[...]` | `images`, `cameras` | hot | per-image |
| `point_track_detail/table.rs:395` `thumbnails_y_x_rgb` at `idx` | one thumbnail | hot (cache fill) | per-image |

### `scene.rs`

| Site | Reads | When | Class |
|---|---|---|---|
| [`scene.rs:362`](../crates/sfm-explorer/src/scene.rs)`--:367` `has_patch_data` | patch column presence | hot | per-point (probe) |
| `scene.rs:449`--`:455` `world_points` -- every point through the node transform | every point | event (zoom to fit, first show) | **whole** |
| `scene.rs:463`--`:470` `camera_world_centres` | `images` | event | per-image |
| `scene.rs:483`--`:499` `camera_sibling_images` | `images` | hot (`app.rs:406`, `dock.rs:214`) | per-image |
| `scene.rs:537`--`:539` `visible_stats` -- `points.len()`, `metadata.infinity_point_count`, `images.len()` | counts | hot (HUD) | whole (count only) |
| `scene.rs:550`, `:562` `hash_prefix`, `point_id` | metadata | event | whole (metadata) |

### `scene_graph/` -- the Scene panel tree

| Site | Reads | When | Class |
|---|---|---|---|
| [`scene_graph/cameras.rs:55`](../crates/sfm-explorer/src/scene_graph/cameras.rs) `cameras.len()` | `cameras` | hot | per-image |
| `scene_graph/cameras.rs:80`--`:90` per-camera use counts, one pass over `images` | `cameras`, `images` | hot (when expanded) | per-image |
| `scene_graph/cameras.rs:235`, `:306` `images.len()` | `images` | hot | per-image |
| `scene_graph/cameras.rs:254`--`:268` `ResectAvailability::of` -- one pass over `images` for posedness | `images`, `feature_indexes()` probe | hot | per-image |
| `scene_graph/cameras.rs:257`, `:336` `images.get(index)` | `images` | hot (visible rows) | per-image |
| [`scene_graph/menus.rs:126`](../crates/sfm-explorer/src/scene_graph/menus.rs) `feature_indexes().is_some()` | observation column | event (menu build) | per-point (probe) |
| [`scene_graph/mod.rs:220`](../crates/sfm-explorer/src/scene_graph/mod.rs) `feature_indexes().is_some()` | observation column | hot | per-point (probe) |
| `scene_graph/mod.rs:589` `metadata.infinity_point_count` | metadata | hot | whole (count only) |
| `scene_graph/mod.rs:609` `points.len()` | `points.len()` | hot | whole (count only) |
| [`scene_graph/widgets.rs:81`](../crates/sfm-explorer/src/scene_graph/widgets.rs)`--:83` `points.len()`, `images.len()`, `cameras.len()` | counts | hot | whole (count only) |

### `scene_renderer/` -- GPU uploads

| Site | Reads | When | Class |
|---|---|---|---|
| [`scene_renderer/upload/points.rs:26`](../crates/sfm-explorer/src/scene_renderer/upload/points.rs)`--:62` instance buffer over every point | every point | event (`needs_upload`) | **whole** |
| `scene_renderer/upload/points.rs:58` `compute_auto_point_size(&recon.points)` | every point (kd-tree NN) | event | **whole** |
| `scene_renderer/upload/points.rs:59` `compute_camera_nn_scale(&recon.images)` | `images` | event | per-image |
| `scene_renderer/upload/points.rs:60` `compute_scene_bounds(&recon.points)` | every point | event | **whole** |
| [`scene_renderer/upload/patches.rs:38`](../crates/sfm-explorer/src/scene_renderer/upload/patches.rs)`--:80`, `:160` scan every point for a non-zero `u` half-vector, then per-patch instance + atlas | every point, `patch_bitmaps_y_x_rgba` | event (`needs_upload`) | **whole** |
| [`scene_renderer/upload/thumbnails.rs:26`](../crates/sfm-explorer/src/scene_renderer/upload/thumbnails.rs), `:76` atlas over `thumbnails_y_x_rgb` | `images`, thumbnails | event (`needs_upload`) | per-image |
| [`scene_renderer/upload/frustums.rs:121`](../crates/sfm-explorer/src/scene_renderer/upload/frustums.rs)`--:133`, `:309`--`:319` | `images`, `cameras` | event (`geometry_changed`) | per-image |
| [`scene_renderer/upload/track_rays.rs:19`](../crates/sfm-explorer/src/scene_renderer/upload/track_rays.rs) `camera_cloud_extent` -- one pass over `images` | `images` | event (points at infinity only) | per-image |
| `scene_renderer/upload/track_rays.rs:58`--`:89` `points[i]`, `feature_indexes()`, `keypoints_xy()`, `observation_offsets[i]`, one track, `images`, `cameras` | one point | event (selection/transform change) | per-point |
| [`scene_renderer/upload/bg_image.rs:33`](../crates/sfm-explorer/src/scene_renderer/upload/bg_image.rs), `:119` `images.get`, `cameras[...]` | `images`, `cameras` | event (camera view entry) | per-image |

`scene_renderer/picking.rs` reads no reconstruction: its `self.points` /
`self.images` are the renderer's own pick-range tables, and
`gpu_types.rs:51`/`:137` only *document* that an instance index equals the local
`recon.points` index.

### `state.rs` and `state/ops.rs`

| Site | Reads | When | Class |
|---|---|---|---|
| [`state.rs:839`](../crates/sfm-explorer/src/state.rs) `camera_of` -- `images.get(...)` | `images` | event (select image) | per-image |
| `state.rs:881` `image_name` -- `images.get(...)` | `images` | event (log entry) | per-image |
| `state.rs:987` `point_id(&node.recon, ...)` | metadata | event (log entry) | whole (metadata) |
| `state.rs:1109` `recon.sift_path_for_image(image_idx)` | `images`, `workspace_dir` | event (SIFT cache miss) | per-image |
| `state.rs:1169` `images.get(...)`, `workspace_dir` | `images` | event (full-res cache miss) | per-image |
| [`state/ops.rs:54`](../crates/sfm-explorer/src/state/ops.rs), `:55` `point_count()`, `image_count()` | counts | event (load) | whole (count only) |
| `state/ops.rs:160` `align::align_reconstructions(&recon, &recon, options)` | two whole reconstructions | event (`Align to…`) | **whole (core call)** |
| `state/ops.rs:221` `.recon.images.get(image)` | `images` | event (resect) | per-image |
| `state/ops.rs:244` `resect::resect_images(&recon, &[image], kind, opts)` | whole reconstruction | event (`Resect Image`) | **whole (core call)** |

### `viewer_3d/`

| Site | Reads | When | Class |
|---|---|---|---|
| [`viewer_3d/input.rs:75`](../crates/sfm-explorer/src/viewer_3d/input.rs), `:80` `images.get` | `images` | hot (input handling) | per-image |
| `viewer_3d/input.rs:519`, `:520` `images.is_empty()`, `images.len()` | `images` | event (image cycling keys) | per-image |
| [`viewer_3d/mod.rs:99`](../crates/sfm-explorer/src/viewer_3d/mod.rs) `images.get(...)` | `images` | event (log entry) | per-image |
| `viewer_3d/mod.rs:341` `!reconstruction.points.is_empty()` then `scene::world_points` | every point | event (first show) | **whole** |
| `viewer_3d/mod.rs:649`--`:662` `images[i]`, `cameras[...]`, `depth_statistics.images` | `images`, `cameras`, depth stats | event (enter camera view) | per-image |
| `viewer_3d/mod.rs:712`--`:720` `images[i]`, `depth_statistics.images` | `images`, depth stats | event (switch camera view) | per-image |
| [`viewer_3d/overlay.rs:248`](../crates/sfm-explorer/src/viewer_3d/overlay.rs)`--:257` `SceneStats` totals | counts (via `visible_stats`) | hot (HUD) | whole (count only) |
| `viewer_3d/overlay.rs:290` `node.recon.images.get(...)` | `images` | hot (HUD) | per-image |

## Calls from the viewer into `sfmtool-core` that take `&SfmrReconstruction`

There are exactly two, both event-driven, both classified **whole**:

| Call site | Core function | Trigger |
|---|---|---|
| [`state/ops.rs:160`](../crates/sfm-explorer/src/state/ops.rs) via [`align.rs:20`](../crates/sfm-explorer/src/align.rs) | `sfmtool_core::analysis::alignment::align_reconstructions(&SfmrReconstruction, &SfmrReconstruction, AlignOptions)` | Scene Graph `Align to ▸ <node>` |
| [`state/ops.rs:244`](../crates/sfm-explorer/src/state/ops.rs) via [`resect.rs:19`](../crates/sfm-explorer/src/resect.rs) | `sfmtool_core::geometry::resect_images(&SfmrReconstruction, &[usize], ResectSource, &ResectImageOptions)` | Scene Graph `Resect Image` / `Resect Image from Matches…` |

Every other `sfmtool_core` use in the viewer takes a camera, a point, a slice or
a transform rather than a reconstruction, so none of them force a
materialisation:

- `sfmtool_core::camera::{intrinsics, remap, report, frustum}` -- take a
  `CameraIntrinsics` (`dock.rs:716`, `image_detail/intrinsics/*`,
  `intrinsics_detail/derived.rs`, `scene_renderer/upload/frustums.rs:13`).
- `sfmtool_core::reconstruction::triangulation::{triangulate_batch,
  depth_uncertainty_batch}` -- take direction/centre/offset slices
  (`metrics.rs:91`).
- `sfmtool_core::patch::cloud::OrientedPatch` -- a value type
  (`point_track_detail/mod.rs:26`, built per point in `patch.rs`).
- `sfmtool_core::geometry::*` projection helpers in
  `point_track_detail/patch.rs:21`.
- `sfmtool_core::{Se3Transform, RotQuaternion}` throughout `scene.rs` and
  `viewer_3d/`.
- `SfmrReconstruction::load` / `::demo` (`state/ops.rs:50`, `:78`, `:105`) --
  producers, not readers.

## Summary

### Counts by class

| Class | Read sites | Of which hot |
|---|---|---|
| per-point (incl. 10 O(1) column probes) | 32 | 19 |
| whole | 29 | 8 |
| per-image / image-table only | 55 | 26 |
| **total** | **116** | **53** |

Of the 29 **whole** sites, 17 are _count only_ or _metadata only_
(`points.len()`, `tracks.len()`, `infinity_point_count`, `content_xxh128`) and
answerable over an overlay in O(1) without materialising. The remaining 12 rows
genuinely walk the point set or the track column; two pairs of them are a call
site and its implementation (`app.rs:310` with `upload/points.rs:26`, and
`app.rs:313` with `upload/patches.rs:38`), so they are ten distinct walks:

| Site | What it walks | When |
|---|---|---|
| `scene_renderer/upload/points.rs:26` (called at `app.rs:310`) | every point -> instance buffer | `needs_upload` |
| `scene_renderer/upload/points.rs:58` | every point -> kd-tree NN point size | `needs_upload` |
| `scene_renderer/upload/points.rs:60` | every point -> scene bounds | `needs_upload` |
| `scene_renderer/upload/patches.rs:38`--`:160` (called at `app.rs:313`) | every point + every patch bitmap -> instances + atlas | `needs_upload` |
| `image_detail/mod.rs:718` `embedded_image_features` | every track, all observations | overlay rebuild (embedded only) |
| `mcp/render.rs:237` `observations_per_image` | every observation | `list_camera_images` |
| `scene.rs:449` `world_points` | every point | zoom to fit |
| `viewer_3d/mod.rs:341` | every point (`is_empty` then `world_points`) | first show |
| `state/ops.rs:160` `align_reconstructions` | the whole value, twice | `Align to…` |
| `state/ops.rs:244` `resect_images` | the whole value | `Resect Image` |

All ten are event-driven. **None of them runs on a frame path.**

### Hot per-frame paths and their class

Every read that runs on a frame where its panel is visible, with its class:

- **Scene panel tree** (`scene_graph/`): `cameras.rs:55`, `:80`--`:90`, `:235`,
  `:254`--`:268`, `:257`, `:306`, `:336` -- per-image; `mod.rs:220` --
  per-point probe; `mod.rs:589`, `:609`, `widgets.rs:81`--`:83` -- whole (count
  only).
- **Image Browser** (`image_browser.rs:45`, `:208`, `:254`; `dock.rs:206`) --
  per-image.
- **Image Detail** (`dock.rs:304`--`:351`; `image_detail/mod.rs:382`--`:386`;
  `overlay.rs:129`, `:151`, `:261`, `:263`, `:535`, `:601`) -- per-image and
  per-point.
- **Point Track Detail** (`dock.rs:422`--`:445`; `point_track_detail/mod.rs:176`,
  `:201`; `header.rs:27`; `patch.rs:64`--`:181`; `table.rs:371`--`:395`) --
  per-point and per-image, plus one whole (count only) bounds check.
- **Intrinsics Detail** (`extrinsics.rs:66`, `:219`; `header.rs:32`;
  `mod.rs:120`--`:138`) -- per-image.
- **3D viewport** (`viewer_3d/input.rs:75`--`:80`; `overlay.rs:248`--`:257`,
  `:290`) -- per-image and whole (count only).
- **Track / sibling highlight sets** (`dock.rs:835`, `:853`; `scene.rs:483`) --
  per-point and per-image; these run from `app.rs` every frame regardless of
  which tabs are open.
- **Upload phase** (`app.rs:394`, `:410`; `scene.rs:362`, `:537`) -- per-image,
  per-point probe, whole (count only). The genuinely whole uploads at
  `app.rs:310`--`:313` are gated on `needs_upload` and do not run per frame.

### Assessment

**No hot per-frame path is whole-reconstruction in the materialising sense.**
The eight whole reads that do run every frame are all length or metadata reads:
`points.len()` as a selection bounds check (`dock.rs:424`, `:444`, `:835`,
`:853`, `point_track_detail/mod.rs:176`), the Scene panel's and HUD's entity
counts (`scene_graph/mod.rs:589`, `:609`, `widgets.rs:81`, `scene.rs:537`,
`viewer_3d/overlay.rs:248`), and the Point Track header's content hash
(`point_track_detail/mod.rs:193`, re-read only on a selection change). An
overlay can answer all of these from `base.point_count() - deleted.len() +
added.len()` and a cached base hash, so nothing in the frame loop forces a
materialisation. The overlay draft's stated precondition -- "the design holds if
the per-frame readers are all per-point or per-column" -- is met by the code as
it stands.

What that leaves is a redesign list rather than a blocker list:

1. **The point and patch uploads are the only real whole walks in the render
   path**, and they are already event-gated on `needs_upload`
   (`scene_renderer/upload/points.rs`, `upload/patches.rs`). They are exactly
   what the umbrella draft's base-identity upload replaces: a point edit must not
   re-run them, so the deleted mask plus an additions instance buffer has to
   cover both the point instances *and* the patch instances, including the patch
   atlas, whose slot assignment is a compaction over the surviving points
   (`upload/patches.rs:72`--`:81`) and therefore is *not* index-stable under the
   overlay. That compaction is the one piece of GPU state the "indexes are
   stable" rule does not already protect, and it needs its own answer.
2. **`compute_auto_point_size`, `compute_camera_nn_scale` and
   `compute_scene_bounds` (`upload/points.rs:58`--`:60`) are derived aggregates
   over the whole point set**, recomputed on every upload. A point edit that
   skips the instance upload also skips these, so either they are accepted as
   stale until the next materialisation, or they gain an incremental update. The
   bounds feed `length_scale` and the adaptive clip planes, so staleness there is
   visible.
3. **`embedded_image_features` (`image_detail/mod.rs:718`) is the panel read that
   will hurt.** It is O(total observations) per overlay rebuild on exactly the
   file flavour (`embedded_patches`) the first track edits target, and a track
   edit changes the overlay for the image the observation was added to. Under the
   overlay it must iterate base-minus-deleted plus additions rather than
   `0..recon.points.len()`, or gain the per-image index it currently lacks.
4. **`observations_per_image` (`mcp/render.rs:237`) and `world_points`
   (`scene.rs:449`)** are event-driven whole walks that can simply materialise,
   or be rewritten as overlay iterations; neither is on a frame path.
5. **The two core calls (`align_reconstructions`, `resect_images`) materialise by
   definition**, which is what the overlay draft already says a
   whole-reconstruction reader does. Both are user-initiated single actions, so a
   materialisation each is affordable at whatever the measured cost turns out to
   be.
6. **`feature_indexes()`, `keypoints_xy()` and `patch_u_halfvec_xyz.is_some()`
   are probed per frame** in six places (`dock.rs:351`, `:422`, `:442`,
   `point_track_detail/patch.rs:107`, `scene.rs:362`,
   `scene_graph/mod.rs:220`), and on four event paths (`app.rs:442`,
   `image_detail/mod.rs:464`, `:550`, `scene_graph/menus.rs:126`). These are
   column-presence
   questions, not point reads, and an overlay must answer them from the base
   alone -- an addition set can never introduce or remove a column.

## `needs_upload` and `transform_epoch` today

Both are the change-detection the umbrella draft's "change detection by
identity" replaces.

**`needs_upload`** is a `bool` on `SceneNode`
([`scene.rs:268`](../crates/sfm-explorer/src/scene.rs)), set `true` exactly once,
in `SceneNode::new` (`scene.rs:319`), and never set `true` again anywhere in the
crate. Its only consumer is the upload phase
([`app.rs:305`](../crates/sfm-explorer/src/app.rs)--`:316`): when set, the node's
points, thumbnails and patches all upload together and the flag is cleared
(`app.rs:314`). So today it means "this node has never been uploaded", and the
reason nothing re-sets it is that nothing mutates a node's reconstruction -- a
reload or a resection replaces the whole `SceneNode` with a fresh one carrying a
new `ReconId` (`state/ops.rs:122`, `:293`), which is also what makes
`SceneRenderer::retain_nodes` (`app.rs:297`) drop the old bundle. The flag also
drives `uploaded_any`, which re-seeds the global `length_scale` (`app.rs:362`)
and forces a frustum re-upload (`app.rs:370`).

**`transform_epoch`** is a `u64` on `AppState`
([`state.rs:544`](../crates/sfm-explorer/src/state.rs)), incremented in exactly
two places: `AppState::reset_node_transform` (`state.rs:1001`) and
`AppState::align_node` after a successful fit
([`state/ops.rs:167`](../crates/sfm-explorer/src/state/ops.rs)). It is compared
against `App::prev_transform_epoch`
([`lib.rs:301`](../crates/sfm-explorer/src/lib.rs)--`:303`, initialised at
`lib.rs:197`) once per frame at `app.rs:356`--`:357`. A difference means
`transform_changed`, which re-seeds `length_scale` (`app.rs:362`), re-uploads
frustum geometry (`app.rs:370`, `:394`) and rebuilds the CPU-space track rays
(`app.rs:430`), since those are the derived state a world-space move invalidates.

Neither carries any information about *what* changed, which is why a point edit
under today's mechanism would re-upload a million points and every patch bitmap.
