# Point Track Detail Panel

The Point Track Detail panel is where the viewer shows one 3D point up close,
together with its track — the 2D feature observations, one per image, that were
triangulated into it. It lists those observations with the numbers that say how
well each one agrees with the point, so a suspect point can be judged
observation by observation rather than as a dot in a cloud.

For the existing panels this integrates with, see
[multi-panel-image-browser.md](multi-panel-image-browser.md). For the
3D track ray visualization that complements this panel, see the Track Ray
Visualization section in [architecture.md](architecture.md).

## Motivation

When a 3D point is selected (via click in the 3D viewer or feature click in the
Image Detail panel), the GUI shows:

- **3D Viewer**: Track rays from each observing camera to the point, plus the
  point highlighted in cyan.
- **Image Browser**: Orange borders on thumbnails of images that observe the
  point.
- **Image Detail**: The selected point's feature highlighted in yellow (if it
  appears in the currently displayed image).

The Point Track Detail panel adds a **dedicated view of the track itself** —
seeing all observations of a point side by side, inspecting per-observation
reprojection error, understanding how the point was triangulated, and navigating
between observing images. It serves as the "point inspector" complement to the
Image Detail panel's "image inspector" role.

## Design

### Panel Position in Dock Layout

The Point Track Detail panel is an `egui_dock` tab in the top-right split, the
active one of the three there, with Image Detail one node to its left behind the
viewport:

```
+----------------------------------+--------------+
|  File  View                      |  (menu bar)  |
+----------------------------------+--------------+
|[3D Viewer][Image Detail]         | [Point Track]|
|                                  |              |
|           3D Viewer              |  Point Track |
|                                  |              |
|                                  |              |
+----------------------------------+--------------+
| << [img01] [img02] [img03] ...              >> |
|              Image Browser (~160px)            |
+------------------------------------------------+
```

Since we use `egui_dock`, the user can redock this panel anywhere. The narrow
right-hand column is the natural default because the panel is a table of rows
about a selection rather than a picture of one, and it reads beside the picture
rather than instead of it.

### What the Panel Shows

The panel has two states:

**No point selected**: Centered placeholder text "No point selected", above a
**Go to Point…** button. This is the state a user stares at when they have an ID
in hand and no idea how to feed it in, so it is where the way in belongs. See
[goto-point.md](goto-point.md).

**Point selected**: A header with point summary statistics, followed by a
scrollable observation table.

#### Header: Point Summary

A compact horizontal bar at the top showing key properties of the selected 3D
point:

```
pt3d_a1b2c3d4_12345 | xyzw: (1.234, -0.567, 2.891, 1) | error: 0.42px | track: 7 obs | max∠: 12.3° | [RGB]
```

| Field | Source | Description |
|-------|--------|-------------|
| Point ID | `selected_point`, minted over the node's version graph | Copy-pastable ID (see Point ID section below) |
| Position | `recon.points[idx].position` | World-space XYZ coordinates |
| Error | `recon.points[idx].error` | RMS reprojection error in pixels |
| Track length | `recon.observation_counts[idx]` | Number of observing images |
| Max angle | Computed from observation rays | Maximum angle between any pair of observation rays (triangulation quality indicator) |
| Color | `recon.points[idx].color` | RGB color swatch |

The color swatch is a small filled rectangle drawn with the point's RGB values.

#### Stored-Patch Header Tile

For an **embedded-patches** reconstruction that carries patch bitmaps
(`recon.patch_bitmaps_y_x_rgba`), a second header row below the point summary
shows the patch **stored in the reconstruction** for the selected point:

```
Stored patch: [tile]
```

The tile is the point's RGBA bitmap drawn at a fixed 64 px square with
nearest-neighbor filtering (patches are tiny; nearest shows texels). Only the
RGB channels are shown — the alpha channel (per-texel cross-view confidence)
is forced opaque. The row is hidden entirely when the reconstruction has no
bitmaps or the point's bitmap is all-zero.

#### Working on the track

On an **embedded-patches** reconstruction a line under the stored-patch tile
names the way to **work on** this track: press *Put selected point on bench* in
the Track Edit panel ([`track-edit.md`](track-edit.md)), which takes a copy of
the track out to the bench ([`bench.md`](bench.md)) where sightings can be
tried, measured and refused before any of it reaches the file. The hint lives
here because the track is what a reader is looking at when they notice the gap,
and it is a hint rather than a button because this panel stays view-only: it is
a pointer at the panel that edits, quoting that button's label from one
constant so the two cannot drift.

#### What the panel reads

Everything about the point -- its position, colour and error, its whole track,
each observation's keypoint, its patch frame and stored bitmap, its
triangulation diagnostics -- is read through the version's overlay accessor
rather than off the base, so a point an edit modified shows the track it holds
now. An index the version has deleted resolves to nothing and the panel takes its
empty state, even though the base still has a row at that index
([document-model.md](document-model.md)).

#### Point ID

The header displays a **Point ID** — a compact, copy-pastable identifier that
uniquely references this 3D point across `.sfmr` files and sessions. This
solves the problem that a raw point index (e.g., `#12345`) is meaningless
without knowing which reconstruction it came from.

The Point ID format is `pt3d_{hash}_{index}`, e.g., `pt3d_a1b2c3d4_12345`.
The hash prefix is derived from the `.sfmr` file's `content_xxh128` hash, and
the entire ID uses only `[a-zA-Z0-9_]` characters so it can be
selected with a single double-click in any terminal or browser.

The header shows, and *Copy Point ID* copies, `pt3d_{hash}_{index}`, minted over
the node's version graph by the rule **disk state first, earliest otherwise**:
the hash is the content the point sits in on disk when its identity reaches
that version, and the oldest content its identity reaches otherwise
([goto-point.md](goto-point.md) § "The ID forms and the version graph"). So the
ID copied out of this header is, in the ordinary case, one a reader of the file
on disk resolves as it stands. Nothing in it names the node it was copied from:
the point is that content's row wherever that content is loaded.

The ID is **recomputed every frame** rather than on selection change. An edit,
an undo or a save can change which content the ID names without the selection
moving at all, so an ID cached against the selection would go on showing the
content of a version the node has left.

For the full format specification, design rationale, and `.sfmr` file resolution
strategy, see the [Point ID section in the sfmr file format
spec](../formats/sfmr-file-format.md#point-id-portable-3d-point-references).

The ID is rendered in a monospace font to visually distinguish it from the other
header fields. Clicking it copies to the clipboard with visual feedback (brief
flash or "Copied!" tooltip).

Immediately right of the copy button sits a **Go to Point** button (an arrow
glyph, sized to match copy's), which opens the dialog that accepts an ID or a
bare index. The two sit together because they are the two halves of one round
trip: copy an ID out of this header, paste it back — here, or in a later
session, or after `sfm xform` produced a new file. See
[goto-point.md](goto-point.md).

#### Observation Table

Below the header, the panel shows a vertically scrollable table of observations
— one row per image that observes this point. The column headings sit above the
scroll area rather than inside it, so they stay put while the rows scroll under
them.

```
+-----+-------+-----------------+--------+-----------+--------+-------+----------------+
|     | Image | Name            | Feat # | Size      | Error  | Angle | Feature (x, y) |
+-----+-------+-----------------+--------+-----------+--------+-------+----------------+
| [t] |     3 | image_003.jpg   |    847 |   8.4x8.2 | 0.21px | 0.03° | (1024.3, 512.7)|
| [t] |    12 | image_012.jpg   |   1247 |   7.6x7.4 | 0.38px | 0.05° | ( 983.1, 498.2)|
| [t] |    15 | image_015.jpg   |    602 | 20.3x7.7  | 0.55px | 0.08° | (1051.8, 520.1)|
| [t] |    23 | image_023.jpg   |   2031 |   9.0x8.9 | 0.19px | 0.02° | ( 997.6, 505.9)|
+-----+-------+-----------------+--------+-----------+--------+-------+----------------+
```

**Columns**:

| Column | Content |
|--------|---------|
| Thumbnail | Small thumbnail of the image (from `recon.thumbnails_y_x_rgb`), with a dot overlay at the feature position, tinted by that observation's reprojection error. |
| Patch | *(embedded-patches only)* The point's patch rendered from this observation's full-res image, through the frame re-anchored on that observation's stored keypoint (see below). Omitted — and all following columns keep their original offsets — when the point has no patch frame. |
| Image | Image index in the reconstruction. |
| Name | Image filename, shortened to the column by a cut out of the **middle**, so the directory and the file name both survive: `images/seatt…yard_13.jpg`. The full path is on hover. A path with a directory above its parent keeps a leading `…/` for what was left out; one without gets none, since there is nothing for the mark to stand for. |
| Feat # | Feature index within the image's SIFT file (or the observation index for embedded-keypoint reconstructions with no SIFT file). |
| Size | The feature's full extent in pixels — see "Size column" below. For SIFT observations the affine shape comes from the cached `affine_shapes`; for embedded keypoints it is derived by projecting the point's patch frame into the image. Shows `N/A` when unavailable (zero). |
| Error | Per-observation reprojection error in pixels (`N/A` when undefined). Defined for points at infinity too: the stored unit direction rotates into camera space without translating and projects like any homogeneous coordinate, so only a point (or direction) behind the camera is undefined. |

**The thumbnail dot's colour** is `colormap::error_color` — the same
green→yellow→red ramp the Image Detail panel's reprojection-error overlay
draws — over a **fixed 0–2 px**. Fixed rather than fitted to the track, because
the dots are read against each other *and* against the absolute number in the
Error column, and a range that shrank to the best and worst of seven
observations would paint a sub-pixel track in full red. The Image Detail
overlay fits its range to the image instead, for the opposite reason: there the
question is which features in *this* frame are the bad ones. An observation
with no error to show — the point is behind the camera, so the metrics came
back `NaN` — is grey, deliberately off the ramp: "no measurement" is not a
position on a green-to-red scale.
| Angle | Angular discrepancy between observation ray and point direction, in degrees. Follows the Error column's rule for points at infinity: the direction is the point, so the angle is measured to it directly. |
| Feature (x, y) | Feature position in image pixel coordinates. |

**Size column**: the columns of an observation's affine-shape matrix are the
projected patch **half**-vectors, so each column norm is a semi-axis. The Size
column doubles them and reports the two **full** extents — the span the patch
quad drawn in the viewport actually covers (`±u ±v`), and the same diameter
convention `embed-patches --patch-size` uses.

Both extents are always printed as `<larger>x<smaller>` (`20.3x7.7`; a
circular feature reads `14.0x14.0`), so an obliquely-viewed patch reads as
foreshortened rather than as a merely smaller feature and the reader never has
to infer which display form they are looking at. One decimal place each. A
fully collapsed (edge-on) shape shows the collapse explicitly (`9.0x0.0`); a
degenerate zero shape prints `N/A`.

**Patch column** (embedded-patches reconstructions): when the reconstruction
stores patch frames (`patch_u_halfvec_xyz` / `patch_v_halfvec_xyz`) and the
selected point's `u` half-vector is non-zero, a patch tile is drawn immediately
right of each row's thumbnail. The tile is the point's oriented patch
**re-rendered from that observation's full-resolution source image**: the
stored half-vectors are split into unit axes + half-extents to build an
`OrientedPatch` (re-marked `w = 0` for points at infinity), then the full-res
image is warped through it with `WarpMap::from_patch` + `remap_bilinear` at
64×64 and displayed at thumbnail size (48 px, nearest filtering, tight at the
stored patch extent). Tiles are rendered lazily per row and cached per image
index. When the patch is not visible in that view it warps to an all-black tile,
which is still inserted into the cache and drawn as such — the tile is always
rendered (a future N/A flag may distinguish "not visible" from a genuinely dark
surface). Stored bitmaps are **not** required for this column — only the frame.

**The frame is re-anchored on the observation's stored keypoint**
(`OrientedPatch::anchored_at_keypoint`, [patch-cloud.md](../core/patch/patch-cloud.md)):
the patch centre is slid within its own plane — on its tangent sphere for a point
at infinity — so that it projects exactly onto the keypoint this observation
carries, and the tile is warped through that frame. This is a photometric
comparison, so it is anchored at the pixels the keypoint localizer aligned rather
than at the point's geometric projection: the tiles of a well-localized track then
show the same content, and match each other, whatever discrepancy the geometry
carries. That discrepancy is what the **Error** column reports, so reading a row
means reading the tiles against each other and the number beside them — a
column of matching tiles with a large error is a well-aligned patch whose 3D
position or pose is off, not a bad match. The stored geometric frame is used
unchanged when the reconstruction has no keypoints (`sift_files`, which never
reaches this column) or when the keypoint's ray cannot meet the patch.

The render itself is `render_patch_texture` in
[point_track_detail/patch.rs](../../crates/sfm-explorer/src/point_track_detail/patch.rs),
and the **Track Edit** panel calls it for a track on the bench
([track-edit.md](track-edit.md)): one renderer, so a committed track and the
editable copy of it cannot show the same surface two ways.

**Shared full-resolution image cache**: the source pixels come from
`AppState::full_res_cache`, a CPU-side cache of decoded full-res images (RGB
`ImageU8`, keyed by image index) shared with the Image Detail panel — which
builds its display texture from the same cache — so no image is decoded from
disk more than once. What an entry holds is the `ImageU8Pyramid` built when the
image was decoded, whose level 0 *is* that image: the photometric readers
(`ViewSources`, the Track Edit panel's cluster tiles) sample the lower levels,
and everything that wants the plain photograph reads level 0, so the pyramid is
built once per image rather than once per step. Decode failures are memoized
(`None`) so missing files aren't re-opened every frame. The dock pre-caches every observing image of the
selected point before the panel draws (only when the reconstruction carries
patch frames); the cache is cleared when the reconstruction changes.

> **TODO (unbounded growth):** `full_res_cache` currently retains a CPU RGB
> copy of every image ever selected or observed for the whole session, plus the
> pyramid levels under it (about a third as much again), with no
> eviction — only cleared on reconstruction load. On large datasets this can
> retain hundreds of MB (e.g. ~85 images @ 2040×1536×3 ≈ 800 MB). It should
> become an **LRU cache bounded by total memory usage** (evict the
> least-recently-used decoded image once an aggregate byte budget is
> exceeded), re-decoding on demand if an evicted image is needed again.

**Sort order**: Rows are sorted by image index (the natural sequence order),
matching the order in the Image Browser.

#### Derived Track Data

When a point is selected, the panel computes several derived quantities from the
track geometry. These are computed once per point selection change, not every
frame.

**Header-level statistics** (displayed alongside the point summary):

The header renders these alongside the point's `xyz`, `error`, and `track`
(observation count) fields, separated by `|` dividers:

| Statistic | Description |
|-----------|-------------|
| Max angle | The maximum angle between any pair of observation rays in the track (`max pair angle: X.X°`). This is the key indicator of triangulation quality — narrow tracks (small max angle) produce poorly constrained depth estimates. Shown only when it is greater than zero. |
| Depth z | Inverse-depth z-score `depth / σ_depth` of the triangulation (`depth z: X.X`). A scale-free observability diagnostic that stays correct in the near-infinity regime, complementing the max angle. Shown only when finite (undefined for points at infinity or fewer than two usable rays). |
| Cond | Condition number of the triangulation's normal matrix (`cond: X`). Large values indicate a poorly conditioned (weakly observable) triangulation. Shown only when finite. |

Both `depth z` and `cond` are computed by `compute_point_diagnostics()` and
cached on the panel, alongside `max_angle_deg`.

**Per-observation columns** (displayed in each table row):

| Column | Computation |
|--------|-------------|
| Size | The feature's full extents in pixels: twice each column norm of the observation's affine-shape matrix (the columns are half-vectors), ordered larger first, printed as one averaged number when near-circular and as `<larger>x<smaller>` when oval — see "Size column" above. Sourced from the cached SIFT `affine_shapes` for SIFT observations, or derived by projecting the point's patch frame into the image for embedded keypoints. |
| Error | Per-observation reprojection error: `\|\| project(R_i * P + t_i) - feature_xy_i \|\|`, where `P` is the 3D point, `(R_i, t_i)` is the world-to-camera transform, `project()` applies intrinsics, and `feature_xy_i` is the observed feature position. |
| Angle | Angle from this observation ray to the 3D point, measured at the camera center. For a perfectly triangulated point this equals zero; nonzero values indicate the observation ray misses the 3D point (related to reprojection error but in angular units). |

### Interactions

#### Click on Row

Clicking a row selects that image (`state.selected_image`), which propagates
to:

- **3D Viewer**: Highlights the corresponding frustum in cyan.
- **Image Browser**: Scrolls to and highlights the thumbnail.
- **Image Detail**: Loads the full-resolution image (if the Image Detail tab is
  visible, which it will be when the user switches back to it).

This is the primary navigation flow: select a point, see its track, click an
observation to inspect the full image.

A row is an *observation*, so the click names a place in that image as well as
the image itself: it **reveals the feature**. The row's feature pixel rides
along with the selection in `reveal_feature`, and the Image Detail panel pans
(never zooms) to put it at the centre of the view when its current view is not
already showing it. Zoomed in on one corner of a frame, the plain selection
would otherwise land on a part of the image the row is not about. The rule and
the state it travels through are
[multi-panel-image-browser.md](multi-panel-image-browser.md) §
"Revealing a feature named by another panel"; the Track Edit panel's rows use
the same path.

#### Double-Click on Row

Double-clicking enters camera view mode for that image (same behavior as
double-clicking a frustum in the 3D viewer or a thumbnail in the Image Browser).

#### Hover on Row

Hovering over a row sets `state.hovered_image`, producing the same cross-panel
hover feedback as hovering over a thumbnail in the Image Browser:

- **3D Viewer**: Brightness boost on the hovered frustum.
- **Image Browser**: Soft highlight border on the corresponding thumbnail.

#### Click-to-Copy in Header

The Point ID and XYZ coordinates are individually clickable. Clicking either
copies the value to the clipboard for use in external tools or scripts. Visual
feedback: the clicked text briefly flashes or a "Copied!" tooltip appears.

- **Point ID click**: Copies the full Point ID string (e.g.,
  `pt3d_a1b2c3d4_12345`).
- **Coordinates click**: Copies the XYZ coordinates (e.g.,
  `1.234, -0.567, 2.891`).

### Cross-Panel Integration

The Point Track Detail panel participates in the existing selection and hover
model:

| Event | Effect on Point Track Detail |
|-------|-------------------------------|
| Point selected (3D viewer click) | Panel populates with track data |
| Point selected (Image Detail feature click) | Panel populates with track data |
| Point selected (Go to Point dialog) | Panel populates, and the tab is raised so the jump is visible |
| Point deselected (background click) | Panel shows "No point selected" |
| `hovered_image` changes | Highlight the corresponding row |
| Reconstruction loaded | Clear panel state |

| Event from Point Track Detail | Effect on Other Panels |
|-------------------------------|------------------------|
| Row clicked | Sets `selected_image`, and reveals the row's feature in Image Detail |
| Row double-clicked | Sets `selected_image` + reveals the feature + enters camera view |
| Row hovered | Sets `hovered_image` |
| Pointer leaves panel | Clears `hovered_image` |
| Go to Point button clicked | Opens the Go to Point dialog (`state.goto_point`) |

The panel does not produce `hovered_point` (unlike Image Detail) since all
content relates to the single selected point.

### Thumbnail Column

The thumbnail column shows the existing 128x128 thumbnail from
`recon.thumbnails_y_x_rgb` with a dot overlay at the feature position.

### Panel State

The panel and its state live in
[point_track_detail/](../../crates/sfm-explorer/src/point_track_detail), split into
`prepare` (builds the per-observation data when the selection changes), `header`,
`table` and `patch`; the numbers they display come from
[metrics/](../../crates/sfm-explorer/src/metrics), at the crate root, because the
Image Detail overlay and the MCP surface read the same ones. Field-by-field
documentation is on the structs themselves; what matters at this level:

```rust
pub struct PointTrackDetail {
    prepared_point: Option<PointRef>,
    observations: Vec<TrackObservationData>,
    max_angle_deg: f32,
    inverse_depth_z: f32,
    condition_number: f32,
    thumbnail_textures: HashMap<ImageRef, egui::TextureHandle>,
    patch_frame: Option<OrientedPatch>,
    stored_patch_texture: Option<egui::TextureHandle>,
    rendered_patch_textures: HashMap<ImageRef, egui::TextureHandle>,
    point_id: String,
    scroll_offset_y: Option<f32>,
}

struct TrackObservationData {
    image_index: usize,
    feature_index: usize,
    feature_xy: [f32; 2],
    reproj_error: f32,
    ray_angle_deg: f32,
    feature_extents: [f32; 2],
    image_name: String,
    image_full_name: String,
}
```

`prepared_point` is a `PointRef` rather than a bare index on purpose: a newly opened
reconstruction can reuse the same point index for a different point, and comparing refs
is what forces a re-prepare — which is in turn what makes the texture caches, keyed by
`ImageRef` for the same reason, safe to rebuild wholesale. `point_id` is handed in by
the caller rather than derived here, because a Point ID is minted over a node's whole
version graph and this panel only ever sees one reconstruction value. Each observation
carries both a truncated `image_name` for the column and the `image_full_name` its
tooltip shows, plus `feature_extents` — the *full* widths of the affine feature shape
along its two axes, the same diameter convention the rendered patch quad spans.

**Preparation**: When `selected_point` changes and differs from
`prepared_point`, recompute `observations` from
`recon.observations_for_point(point_idx)`. This requires looking up feature
positions from the SIFT cache (same path as `upload_track_rays`). The panel
ensures SIFT data is cached for all observing images.

### Response Type

```rust
/// Response from the Point Track Detail panel.
pub struct PointTrackDetailResponse {
    /// If Some, the user clicked a row — select this image.
    pub select_image: Option<usize>,
    /// The clicked row's feature, in that image's own pixels: the place the
    /// Image Detail panel is asked to bring into view along with the image.
    pub reveal_feature: Option<[f32; 2]>,
    /// If Some, the user double-clicked a row — enter camera view for this image.
    pub request_camera_view: Option<usize>,
    /// Image index currently under the pointer (for cross-panel hover).
    pub hovered_image: Option<usize>,
    /// Whether the pointer is currently inside the panel.
    pub has_pointer: bool,
    /// The user asked for the Go to Point dialog — from the header button, or
    /// from the empty state's button when no point is selected at all.
    pub request_goto_point: bool,
    /// If Some, a row's context menu asked for that image's observation to be
    /// taken out of the selected point's track.
    pub remove_observation: Option<usize>,
}
```

The image indices are local to the reconstruction the panel was shown with; `dock.rs`
pairs them back into `ImageRef`s. A track never spans reconstructions, so every row
belongs to the selected point's own recon.

## Performance Considerations

**Track data preparation**: Computing per-observation reprojection error is
O(track_length) with one `pixel_to_ray` + projection per observation. Track
lengths are typically 2-50, so this is negligible. Done once per point selection
change.

**SIFT cache pre-population**: When a point is selected, the SIFT cache is
pre-populated for all observing images by `app.rs`. The Point Track Detail
panel relies on this.

**Thumbnail textures**: Using the existing 128x128 thumbnails from
`recon.thumbnails_y_x_rgb` avoids any image I/O. Drawing a feature dot overlay
on a 128x128 egui image is trivial.

**Scroll clipping**: Only rows within the scroll viewport need texture uploads
and rendering.

## Ideas for Future Consideration

- **Cropped feature thumbnails**: *(implemented for embedded-patches
  reconstructions as the per-observation Patch column — the point's patch
  re-rendered from each observation's full-res image; see the Observation
  Table section. Still open for `sift_files` reconstructions, which carry no
  patch frame to define the crop.)*
- **Stored-patch alpha tile**: Show the stored bitmap's alpha channel
  (per-texel cross-view confidence) as a grayscale tile beside the RGB header
  tile, like `strips/_solve.py`'s `_bitmap_ref_tile`.
- **Enhanced interactions**: Highlight the corresponding track ray in the 3D
  viewer when hovering a row (requires per-ray hover state in the track ray
  shader). Show the reprojected point position as a second dot on each
  thumbnail, next to the observed feature — the gap visualizes reprojection
  error directly.
- **3D uncertainty visualization**: Estimate positional uncertainty of the 3D
  point from the observation ray geometry and display as a confidence indicator.
  The along-ray uncertainty is approximately
  `σ_depth ≈ σ_reproj * depth / (f * sin(θ/2))`, forming an elongated ellipsoid
  for narrow tracks.
