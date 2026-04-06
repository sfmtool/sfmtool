# Point Track Detail Panel

*Status: Implemented*

This document specifies the Point Track Detail panel for the sfmtool GUI. It
provides a focused view of a selected 3D point and its track — the set of 2D
feature observations across images that were triangulated to produce it.

For the existing panels this integrates with, see
[gui-multi-panel-image-browser.md](gui-multi-panel-image-browser.md). For the
3D track ray visualization that complements this panel, see the Track Ray
Visualization section in [gui-architecture.md](gui-architecture.md).

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

The Point Track Detail panel is an `egui_dock` tab alongside Image Detail in
the top-right split, defaulting to the non-active tab:

```
+----------------------------------+--------------+
|  File  View                      |  (menu bar)  |
+----------------------------------+--------------+
|                                  | [Image Detail|
|                                  |  Point Track]|
|           3D Viewer              |              |
|                                  |              |
|                                  |              |
+----------------------------------+--------------+
| << [img01] [img02] [img03] ...              >> |
|              Image Browser (~160px)            |
+------------------------------------------------+
```

Since we use `egui_dock`, the user can redock this panel anywhere. Tabbing it
with Image Detail is the natural default because both panels display detailed
information about a selection, and the user typically wants to see either the
full image or the point track, not both simultaneously.

### What the Panel Shows

The panel has two states:

**No point selected**: Centered placeholder text: "No point selected".

**Point selected**: A header with point summary statistics, followed by a
scrollable observation table.

#### Header: Point Summary

A compact horizontal bar at the top showing key properties of the selected 3D
point:

```
pt3d_a1b2c3d4_12345 | xyz: (1.234, -0.567, 2.891) | error: 0.42px | track: 7 obs | max∠: 12.3° | [RGB]
```

| Field | Source | Description |
|-------|--------|-------------|
| Point ID | `selected_point` + `content_xxh128` | Copy-pastable ID (see Point ID section below) |
| Position | `recon.points[idx].position` | World-space XYZ coordinates |
| Error | `recon.points[idx].error` | RMS reprojection error in pixels |
| Track length | `recon.observation_counts[idx]` | Number of observing images |
| Max angle | Computed from observation rays | Maximum angle between any pair of observation rays (triangulation quality indicator) |
| Color | `recon.points[idx].color` | RGB color swatch |

The color swatch is a small filled rectangle drawn with the point's RGB values.

#### Point ID

The header displays a **Point ID** — a compact, copy-pastable identifier that
uniquely references this 3D point across `.sfmr` files and sessions. This
solves the problem that a raw point index (e.g., `#12345`) is meaningless
without knowing which reconstruction it came from.

The Point ID format is `pt3d_{hash}_{index}`, e.g., `pt3d_a1b2c3d4_12345`.
The hash prefix is derived from the `.sfmr` file's `content_xxh128` hash, and
the entire ID uses only `[a-zA-Z0-9_]` characters so it can be
selected with a single double-click in any terminal or browser.

For the full format specification, design rationale, and `.sfmr` file resolution
strategy, see the [Point ID section in the sfmr file format
spec](../formats/sfmr-file-format.md#point-id-portable-3d-point-references).

The ID is rendered in a monospace font to visually distinguish it from the other
header fields. Clicking it copies to the clipboard with visual feedback (brief
flash or "Copied!" tooltip).

#### Observation Table

Below the header, the panel shows a vertically scrollable table of observations
— one row per image that observes this point.

```
+-----+-------+-----------------+--------+--------+-------+----------------+
|     | Image | Name            | Feat # | Error  | Angle | Feature (x, y) |
+-----+-------+-----------------+--------+--------+-------+----------------+
| [t] |     3 | image_003.jpg   |    847 | 0.21px | 0.03° | (1024.3, 512.7)|
| [t] |    12 | image_012.jpg   |   1247 | 0.38px | 0.05° | ( 983.1, 498.2)|
| [t] |    15 | image_015.jpg   |    602 | 0.55px | 0.08° | (1051.8, 520.1)|
| [t] |    23 | image_023.jpg   |   2031 | 0.19px | 0.02° | ( 997.6, 505.9)|
+-----+-------+-----------------+--------+--------+-------+----------------+
```

**Columns**:

| Column | Content |
|--------|---------|
| Thumbnail | Small thumbnail of the image (from `recon.thumbnails_y_x_rgb`), with a dot overlay at the feature position. |
| Image | Image index in the reconstruction. |
| Name | Image filename (truncated with leading `…/` for long paths). |
| Feat # | Feature index within the image's SIFT file. |
| Error | Per-observation reprojection error in pixels. |
| Angle | Angular discrepancy between observation ray and point direction, in degrees. |
| Feature (x, y) | Feature position in image pixel coordinates. |

**Sort order**: Rows are sorted by image index (the natural sequence order),
matching the order in the Image Browser.

#### Derived Track Data

When a point is selected, the panel computes several derived quantities from the
track geometry. These are computed once per point selection change, not every
frame.

**Header-level statistics** (displayed alongside the point summary):

| Statistic | Description |
|-----------|-------------|
| Max angle | The maximum angle between any pair of observation rays in the track. This is the key indicator of triangulation quality — narrow tracks (small max angle) produce poorly constrained depth estimates. |

**Per-observation columns** (displayed in each table row):

| Column | Computation |
|--------|-------------|
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
| Point deselected (background click) | Panel shows "No point selected" |
| `hovered_image` changes | Highlight the corresponding row |
| Reconstruction loaded | Clear panel state |

| Event from Point Track Detail | Effect on Other Panels |
|-------------------------------|------------------------|
| Row clicked | Sets `selected_image` |
| Row double-clicked | Sets `selected_image` + enters camera view |
| Row hovered | Sets `hovered_image` |
| Pointer leaves panel | Clears `hovered_image` |

The panel does not produce `hovered_point` (unlike Image Detail) since all
content relates to the single selected point.

### Thumbnail Column

The thumbnail column shows the existing 128x128 thumbnail from
`recon.thumbnails_y_x_rgb` with a dot overlay at the feature position.

### Panel State

```rust
/// Point Track Detail panel state.
pub struct PointTrackDetail {
    /// The point index we've prepared data for, or None.
    prepared_point: Option<usize>,
    /// Precomputed observation data for the current point.
    observations: Vec<TrackObservationData>,
    /// Maximum angle (degrees) between any pair of observation rays in the track.
    max_angle_deg: f32,
    /// Cached thumbnail textures keyed by image index.
    thumbnail_textures: HashMap<usize, egui::TextureHandle>,
    /// The content_xxh128 hash prefix (first 8 hex chars) for Point IDs.
    hash_prefix: String,
}

/// Precomputed data for one observation in the track.
struct TrackObservationData {
    /// Index into `recon.images`.
    image_index: usize,
    /// Feature index within the image's SIFT file.
    feature_index: usize,
    /// Feature position in image pixel coordinates.
    feature_xy: [f32; 2],
    /// Per-observation reprojection error in pixels.
    reproj_error: f32,
    /// Angular discrepancy between observation ray and point direction, in degrees.
    ray_angle_deg: f32,
    /// Truncated display name.
    image_name: String,
}
```

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
    /// If Some, the user double-clicked a row — enter camera view for this image.
    pub request_camera_view: Option<usize>,
    /// Image index currently under the pointer (for cross-panel hover).
    pub hovered_image: Option<usize>,
    /// Whether the pointer is currently inside the panel.
    pub has_pointer: bool,
}
```

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

- **Cropped feature thumbnails**: Replace the whole-image thumbnails in the
  table with 128x128 crops centered on each feature position, loaded lazily
  from full-resolution images. Would give a much closer view of what was
  actually matched, but adds image I/O complexity.
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
