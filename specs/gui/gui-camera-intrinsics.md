# Camera Intrinsics: Scene-Graph Node, Image Overlay, Detail Panel

**Status:** design — not implemented.

The viewer can show you where a camera *is* and what it *saw*, but nothing in it
tells you what the camera *is*: which intrinsic model, what focal length, how far
the principal point sits from the image centre, how much the lens bends, how many
degrees off-axis a given pixel looks. That information exists — `sfm inspect`
prints it as a table — but it is unreachable from the one place where you are
already looking at the image it describes.

This spec adds it in three coupled pieces:

1. a **Camera Intrinsics** group in the Scene Graph, beside a renamed
   **Camera Images** group, with a two-way selection coupling between them;
2. an **Intrinsics** overlay layer in the Image Detail panel — independently
   toggled, composing with whichever feature or heatmap mode is active — drawing the
   principal point, angular axes, and the lens's distortion field *on the
   photograph*;
3. an **Intrinsics** dock panel: the parameter table, the distortion curve, and
   — when an image is selected — that image's extrinsics.

Related specs: [gui-scene-graph.md](gui-scene-graph.md) (the tree and the
selection model), [gui-multi-panel-image-browser.md](gui-multi-panel-image-browser.md)
(the Image Detail panel and its overlay modes),
[gui-camera-views.md](gui-camera-views.md) (frustum geometry, distorted frustum
rendering, the FOV maths this reuses), and
[specs/core/camera-model-registry.md](../core/camera-model-registry.md) (the one
declaration each camera model has).

---

## Motivation

Three concrete failures this closes:

- **"Why is this reconstruction warped?"** A bad focal-length prior or a
  runaway `k1` is visible in the numbers long before it is visible in the point
  cloud. Today you must quit the viewer and run `sfm inspect`.
- **"Is this fisheye actually 180°?"** The intrinsics say `f` in pixels per
  radian; what a user wants is *degrees at the image corner*, and the two are
  separated by a model-dependent projection they should not have to do in their
  head.
- **"Which images share this camera?"** In a rig dataset (`kerry_park`: 24
  frames × 2 fisheyes) the answer is structural and currently invisible — the
  tree shows 48 image rows and no hint that they resolve to two intrinsics.

---

## Terminology: the rename is a bug fix

`.sfmr` uses COLMAP's vocabulary, where a **camera** is an intrinsics record and
an **image** is a posed view that references one. The Scene Graph's group row
says `Cameras (243)` and counts `node.recon.images.len()` — it has been
labelling images as cameras since it was written, and the reconstruction row's
compact count says `243 cams` for the same quantity.

So the rename is not cosmetic:

| Before | After | Counts |
|--------|-------|--------|
| `Cameras (243)` group | `Camera Images (243)` | `recon.images.len()` |
| — | `Camera Intrinsics (2)` group | `recon.cameras.len()` |
| `1.2M pts · 243 cams` | `1.2M pts · 243 imgs · 2 cams` | points / images / cameras |

Three counts make this the longest row in a panel that defaults to 18% of the
window, so it elides rather than truncating or wrapping: when the row cannot fit
all three, the camera count goes first (it is also on the group row one line
down), then the image count, leaving the point count — which has no other home
in the tree — last to go. The elision is on available width, not on a character
budget, so a widened panel restores the counts.

Two other sites keep the old word and should **not** be swept up in the rename,
because in both of them "camera" already means something else and is correct:

- `AlignSource::Cameras` in the node context menu's `Align to ▸` submenu means
  *align using camera poses* (as against `Points`). It becomes
  **`Camera Poses`** — one word added, ambiguity removed, and it stays distinct
  from both new group names.
- Camera **view** mode (`Z`), the viewport HUD's camera section, and
  `gui-camera-views.md` throughout: all about the posed view. Unchanged.

The `SceneNode::show_cameras` field drives frustum and image-quad visibility, so
under the new vocabulary it is `show_camera_images`. It is referenced in
`gui-scene-graph.md`, `gui-camera-views.md` and `gui-viewport-hud.md`; the rename
lands in code and those three docs together, or not at all.

---

## Data model

### `CameraRef` and the selection field

A new ref type in `scene.rs`, alongside `ImageRef` and `PointRef` and shaped
exactly like them — a `ReconId` plus a local index, with `index()` and
`index_in()`:

```rust
/// A camera intrinsics record within a specific reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraRef {
    pub recon: ReconId,
    pub camera: u32,
}
```

and one new field in `AppState`:

```rust
/// The selected camera intrinsics, or `None`.
///
/// Coupled to `selected_image`: whenever `selected_image` is `Some`, this is
/// `Some` and names that image's camera. See "The selection coupling".
pub selected_camera: Option<CameraRef>,
```

`selected_camera` is **stored, not derived**. It has to be: a user can select
intrinsics with no image selected at all (that is the point of the group), and
deriving it from `selected_image` would make the Intrinsics panel go blank the
moment they clicked anything else. What is derived is the *constraint* between
the two, enforced in one place.

### The selection coupling

Every write to either field goes through two `AppState` methods, so the
invariant cannot be violated by a caller that forgot it:

```rust
/// Select an image, and with it the camera it uses.
pub fn select_image(&mut self, image: Option<ImageRef>);

/// Select a camera. Clears `selected_image` unless the selected image
/// already uses this camera.
pub fn select_camera(&mut self, camera: Option<CameraRef>);
```

**Invariant.** `selected_image == Some(i)` implies
`selected_camera == Some(camera_of(i))`.

The resulting behaviour, stated exhaustively so the tests can be read off it:

| Action | `selected_image` | `selected_camera` |
|--------|------------------|-------------------|
| Select image `i` | `i` | camera of `i` |
| Select image `i`, already selected | `i` | unchanged (same camera) |
| Select camera `c`, no image selected | `None` | `c` |
| Select camera `c`, selected image uses `c` | **kept** | `c` |
| Select camera `c`, selected image uses `c' ≠ c` | **cleared** | `c` |
| Clear image selection (click empty space, `Esc`) | `None` | **kept** |
| Select a different reconstruction | filtered to that recon | filtered to that recon |
| Close / reload the owning node | cleared | cleared |

Two of these rows are choices rather than consequences, and both go the way the
user's request implies:

- **Selecting the camera an image already uses keeps the image.** The user's
  rule is that a *different* intrinsics deselects the image; the same intrinsics
  is not different, and clearing there would make clicking the highlighted
  intrinsics row of the image you are looking at throw that image away.
- **Clearing the image keeps the camera.** Deselecting a photograph is not a
  statement about the lens, and collapsing the Intrinsics panel because the user
  dismissed an image would be a surprise. `Esc` pressed twice clears both — the
  second press, seeing no image, clears the camera.

`AppState::retain_recon` / `close_node` / `reload_node` filter `selected_camera`
by `ReconId` exactly as they already filter the other three.

### `CameraModel::parameter_names`

The parameter table needs the model's parameters **in declaration order**.
`SfmrCamera::parameters` is a `BTreeMap<String, f64>`, so it can only offer
lexicographic order, which is wrong twice over: it separates related terms, and
for the spline models it orders `bspline_c10` before `bspline_c2`. The Python
side already keeps a hand-written `_CAMERA_PARAM_NAMES` table for exactly this
reason (`camera/cameras.py`, used by `sfm inspect`).

The registry macro already has the ordered field list of every fixed-arity
model, so it generates the accessor rather than anyone writing a second copy —
this is the same "one declaration per camera model" rule
[camera-model-registry.md](../core/camera-model-registry.md) sets out:

```rust
impl CameraModel {
    /// This model's parameter names in declaration order — the order
    /// `sfm inspect` prints and the Intrinsics panel tabulates.
    ///
    /// For the two spline models the trailing `bspline_c{i}` names depend on
    /// the coefficient count, so this returns an owned `Vec<Cow<'static, str>>`
    /// rather than a `&'static [&'static str]`.
    pub fn parameter_names(&self) -> Vec<Cow<'static, str>>;
}
```

The GUI and `sfm inspect` should agree glyph for glyph, so that a user can diff
the panel against the CLI. Pointing the Python table at this accessor through
the PyO3 bindings is the obvious follow-up and is **out of scope here** — noted
so it does not get quietly forgotten.

### `camera::report` — the derived quantities

Everything the panel and overlay compute beyond raw parameters is pure maths on
a `CameraIntrinsics`, and belongs in `sfmtool-core` where it can be unit-tested
without an egui context. A new module `camera::report`:

```rust
/// Angles, in degrees, subtended by an image's edges and corners.
pub struct FieldOfView {
    pub horizontal: f64,          // mid-left pixel to mid-right pixel
    pub vertical: f64,            // mid-top to mid-bottom
    pub diagonal: f64,            // corner to opposite corner
    pub max_off_axis: f64,        // largest θ over the four corners
}

/// One sample of the model's radial map along a given azimuth.
pub struct RadialSample {
    pub theta_deg: f64,
    pub radius_px: f64,           // |project(ray(θ, φ)) − principal point|
    pub reference_px: f64,        // the same under the model's ideal map
}

/// The displacement of one pixel from where the ideal map would place it.
pub struct DistortionSample {
    pub pixel: [f64; 2],          // where the model actually projects the ray
    pub reference: [f64; 2],      // where the ideal map projects it
}

pub fn field_of_view(cam: &CameraIntrinsics) -> Option<FieldOfView>;
pub fn radial_profile(cam: &CameraIntrinsics, azimuth_deg: f64, samples: usize)
    -> Vec<RadialSample>;
pub fn distortion_field(cam: &CameraIntrinsics, cols: usize, rows: usize)
    -> Vec<DistortionSample>;
/// 35 mm-equivalent focal length: `f_px · 43.267 / diagonal_px`. `None` for
/// models whose focal length is pixels-per-radian rather than pixels.
pub fn equiv_focal_length_35mm(cam: &CameraIntrinsics) -> Option<f64>;
```

**The ideal ("reference") map** is what "distortion" is measured against, and it
is per-family, not per-model:

| Family | Ideal map | Members |
|--------|-----------|---------|
| Perspective | `r = f·tan θ` (pinhole with the same `fx, fy, cx, cy`) | Pinhole, SimplePinhole, SimpleRadial, Radial, OpenCV, FullOpenCV, SfmtoolPinhole |
| Fisheye | `r = f·θ` (equidistant with the same `f, cx, cy`) | OpenCVFisheye, SimpleRadialFisheye, RadialFisheye, ThinPrismFisheye, RadTanThinPrismFisheye, EquidistantFisheye, SfmtoolFisheye |
| Equirectangular | itself | Equirectangular |

Because the reference carries the *same* `fx, fy, cx, cy`, the displacement it
measures is pure lens distortion: neither the focal length, nor a non-square
pixel aspect, nor the principal-point offset leaks into it. A model whose
`has_distortion()` is `false` produces an identically-zero field, and the code
paths that draw it are skipped rather than drawing zero-length arrows.

### Camera-frame conventions, stated once

Both the overlay and the panel do ray maths, and both are one sign flip away
from being confidently wrong, so the conventions are pinned here and cited from
the sections that use them:

- `CameraIntrinsics::pixel_to_ray` / `ray_to_pixel` work in the **canonical
  camera frame: −Z forward, +Y up, +X right** (`camera/distortion.rs`). Forward
  is `(0, 0, −1)`.
- Consequently a ray `(0, sin ε, −cos ε)` with `ε > 0` looks **upward** and
  projects **above** the principal point, and `(sin α, 0, −cos α)` with `α > 0`
  projects to its **right**. Positive elevation is up on screen with no negation
  anywhere — which is why the overlay can label its axes with signed angles and
  have them read correctly.
- Stored poses are world-to-camera in the same canonical convention (`.sfmr`
  version ≥ 5; older files are converted on load). Camera centre is `C = −Rᵀt`;
  the world-space axis triple is `right = Rᵀe₀`, `up = Rᵀe₁`,
  `forward = −Rᵀe₂`.
- `CameraIntrinsics::intrinsic_matrix()` returns `K` in the **optical** frame
  (+Z forward, +Y down), the frame COLMAP's distortion kernels are written in.
  So the projection matrix is `P = K · S · [R | t]` with `S = diag(1, −1, −1)`,
  **not** `K · [R | t]`. The panel that displays `P` displays that product and
  says so.
- Pixel coordinates are the same continuous convention the rest of the viewer
  uses: `(0, 0)` is the top-left *corner* of the image and the centre of the
  top-left pixel is `(0.5, 0.5)`, matching `distorted_mesh.rs`'s `s·w` sampling
  and `validate_keypoints`' `[0, w) × [0, h)` bound. The image centre is
  `(w/2, h/2)`, and it is generally *not* the principal point — showing that
  gap is half the point of the overlay.

---

## Scene Graph: the Camera Intrinsics group

The node body gains a group above the images group, so its two rows read:

```
▾ 👁 S 🖱 ▪ kerry_park                      412K pts · 48 imgs · 2 cams
  ▾   Camera Intrinsics (2)
        #0  OPENCV_FISHEYE  480×480   f 240.1   26 images
        #1  OPENCV_FISHEYE  480×480   f 239.7   22 images
  ▸ 👁 Camera Images (48)
  ▸ 👁 Points (412,551 · 12 at ∞) [∞]
    👁 Patches
```

**Group row** — `[▸] Camera Intrinsics (2)`

- **No eye.** Every other group row's eye drives a visibility flag on the node;
  intrinsics have no geometry of their own to hide. The column is left blank
  rather than filled with a disabled glyph, and the label indents to match.
- Collapsed by default when the count is > 4, expanded otherwise. A typical
  reconstruction has one camera and a rig has two or three: for those the list
  *is* the answer and hiding it behind a triangle costs a click for nothing.
  Beyond a handful (a per-image-intrinsics solve can produce hundreds) it
  behaves like the images list and stays out of the way.
- Row id `row_id(node.id, "intrinsics")`, per the panel's explicit-id rule.

**Camera row** — `#0  OPENCV_FISHEYE  480×480  f 240.1  26 images`

- Fields: index, model name, `width×height`, focal length (`f` when
  `fx == fy`, `fx/fy` otherwise, one decimal), and the number of images using
  it. A camera no image references reads `0 images` and is drawn weak — that
  state is legal in a `.sfmr` and worth seeing rather than hiding.
- Models flagged beta in the registry (`SFMTOOL_FISHEYE`, `SFMTOOL_PINHOLE`)
  carry a `β` suffix on the model name with the registry's beta note as the
  hover tooltip.
- **Click**: `select_camera(Some(CameraRef))` — with the deselect-the-image
  consequence from the coupling table.
- **Double-click**: zoom the 3D viewport to fit every image using this camera —
  the same framing `zoom_to_node` does, over that subset of the node's cameras
  rather than all of them. For a rig this frames one sensor's whole trajectory
  in a single gesture, which nothing else in the viewer does. It is the tree's
  third double-click target, and consistent with the other two: a double-click
  frames what the row denotes, and a camera row denotes a set of images.
- **Selected row**: highlighted like a camera-image row. The row is *also*
  highlighted, more weakly, whenever `selected_image`'s camera is this one and
  the row is not itself the click target — but by the invariant those are the
  same row, so in practice one highlight rule covers both and the coupling is
  visible in the tree for free.
- Rows are laid out plainly, not virtualized: the count is bounded by the number
  of distinct intrinsics, which is small even in the pathological case, and the
  list is capped at `CAMERA_LIST_HEIGHT` with a scroll area like the images
  list.
- **No hover channel.** Cross-panel hover is a two-field protocol
  (`hovered_image` / `hovered_point`) with an ownership rule per panel
  ([gui-cross-panel-hover.md](gui-cross-panel-hover.md)); a third field would
  have to be threaded through every panel for a payoff — a preview of a
  selection that is one click away — that does not justify it. Hovering a camera
  row shows a tooltip with the full parameter list and nothing else.

**Response plumbing.** `SceneGraphResponse` gains two fields, applied by
`dock.rs` in the existing coarsest-first order (`select_recon`, then
`select_camera`, then `select_image` / `select_point`) so that a camera click
and the recon selection it implies land in the right order:

```rust
pub select_camera: Option<CameraRef>,
pub zoom_to_camera: Option<CameraRef>,
```

---

## Image Detail: the Intrinsics overlay layer

The intrinsics overlay is an **independent layer**, not an `OverlayMode`
variant. `OverlayMode` stays exactly as it is, with its seven mutually exclusive
feature and heatmap modes, and the intrinsics layer composes with whichever one
is active — including `None`, which gives the clean "just the camera model" view.

This is the right shape because the two answer different questions about the
same photograph and the interesting questions are the *joint* ones: do my
keypoints crowd the distorted rim? Is the reprojection-error heatmap hot
precisely where the distortion field is largest, or is the error uncorrelated
with the lens? Does the principal point sit inside the region the tracked
features actually cover? An exclusive mode makes each of those a matter of
flipping back and forth and remembering, which is exactly what an overlay exists
to avoid.

The layer's state is a sibling struct on `AppState` — not a field of
`FeatureDisplaySettings`, whose name and contents are about *feature* display:

```rust
/// Scene-level state of the intrinsics overlay layer, drawn on the Image
/// Detail panel independently of `FeatureDisplaySettings::overlay_mode`.
pub struct IntrinsicsDisplaySettings {
    /// Draw the layer at all. Off by default: it is a diagnostic, and the
    /// panel's default view is the photograph.
    pub enabled: bool,            // default false
    /// Draw the angular axes through the principal point.
    pub axes: bool,               // default true
    /// Draw iso-angle rings at the same angular ladder as the axis ticks.
    pub rings: bool,              // default false
    /// Draw the distortion displacement field. Ignored when the model has
    /// no distortion.
    pub distortion: bool,         // default true
    /// Displacement arrow exaggeration. `None` = auto (see below).
    pub distortion_scale: Option<f32>,
    /// Grid density of the arrow field, arrows across the image width.
    pub grid_cols: usize,         // default 16
}
```

The principal-point marker is always drawn when `enabled`; `axes`, `rings` and
`distortion` gate the parts of the layer that put real ink on the image.

### Toolbar

The feature filters stay where they are and keep their existing
show-when-`overlay_mode != None` rule — they are unaffected by this layer. The
layer adds one checkbox and, behind it, a settings button:

```
[Overlay: Reproj Error ▾] [Max: 500 ▾] [☐ Min/max size: 0.0 50.0] │ [☑ Intrinsics] [⚙]
```

The toolbar row is already the widest thing in the panel, and adding four more
controls inline would wrap it at any reasonable panel width. So the layer's
sub-toggles live in a popup on the `⚙` button, which is enabled only while the
checkbox is ticked:

```
  ☑ Axes
  ☐ Iso-angle rings
  ☑ Distortion field    ×3 (auto) ▾
    Grid density        16 ▾
  ─────────────────────
  max displacement 12.4 px
```

When `camera.has_distortion()` is false, the `Distortion field` row is replaced
by a disabled `No distortion` line, so the control never sits there inviting a
click that does nothing.

`I` toggles the layer while the pointer is over the Image Detail panel. It is a
control users will flip constantly once it composes — that is the whole point of
making it a layer — and `I` is free there (the panel binds only `Z`).

### Compositing with the feature layers

Two layers drawing on one image need three rules, and all three are forced by
the fact that the feature layer is the *data* and the intrinsics layer is the
*reference frame* it sits in.

**Z-order.** The intrinsics layer draws **beneath** the features, so a keypoint
is never hidden by an axis or an arrow. The one exception is the principal-point
marker, which draws last, on top of everything: it is a dozen pixels of ink
whose whole job is to be locatable, and a dense heatmap would otherwise bury it.

**Colour.** The heatmap modes sweep the full colormap, so no hue is safe to
reserve. The layer therefore draws in near-white with a 1 px dark halo on every
stroke, label and arrow — the standard treatment for annotation over arbitrary
imagery, and legible over both a bright sky and a black colormap floor. Feature
green/red and the selected-feature highlight are untouched, so nothing about the
existing modes has to change.

**Weight.** Axis polylines, rings and arrows draw at reduced opacity (~70%) so
the photograph and the features stay readable through them; the principal point,
the centre marker and all text draw fully opaque. The layer is a reference grid,
not a subject.

### Hover: one tooltip, composed

The feature layer already owns a hover tooltip, hit-testing features through its
kd-tree. Rather than two tooltips fighting for the cursor, there is one, and the
intrinsics readout is appended to whatever the feature layer produced:

- pointer within a feature's hit radius → the existing feature tooltip, with the
  intrinsics readout below a separator;
- pointer anywhere else on the image → the intrinsics readout alone.

Composed, this is better than either alone: hovering a keypoint tells you its
3D point *and* how many degrees off-axis it sits and how far the lens displaced
it — which is the natural way to ask whether a suspicious observation is a
rim-distortion artefact.

The intrinsics readout is suppressed entirely when the layer is off, so the
feature tooltip is byte-for-byte what it is today.

### What is drawn

Everything below is computed in image pixel coordinates and mapped through the
panel's existing image-to-panel transform, so it pans and zooms with the
photograph.

**Principal point.** A small cross (4 px arms) plus a 3 px open circle at
`(cx, cy)`, drawn last and fully opaque per the compositing rules above — this
is the one mark that must never be lost under a dense feature overlay.

**Image centre and the offset.** A faint `+` at `(w/2, h/2)` and a thin
connector to the principal point, labelled with the offset in pixels and as a
percentage of the half-diagonal: `Δ (−12.4, +3.1) px · 0.4%`. Suppressed when
the offset is under half a pixel, where a connector would be a smudge. This
answers "is the principal point where it should be?" at a glance, which the
number alone does not.

**Angular axes.** Two polylines through the principal point, sampled by
projecting rays rather than by drawing straight lines:

- horizontal: `ray(α) = (sin α, 0, −cos α)` for `α` on the tick ladder;
- vertical: `ray(ε) = (0, sin ε, −cos ε)`.

Each is drawn as a polyline through densely sampled `ray_to_pixel` results (one
sample per ~4 panel pixels), with tick marks and labels at the ladder values.
**They are not straight.** Under distortion they bend, and the bend *is* the
distortion, visible without the arrow field being on at all. Under a distortion-
free model they come out straight, which is itself informative.

Sampling stops at the first `ray_to_pixel` that returns `None` (outside the
model's domain) or that leaves the image by more than 5% of its diagonal.

*Tick ladder*: the coarsest of `1°, 2°, 5°, 10°, 15°, 30°, 45°` that keeps
adjacent ticks at least 48 panel pixels apart at the current zoom, recomputed
per frame — the panel zooms to 32×, and a ladder fixed at load time would be
useless at both ends. Labels are signed (`−20°`, `−10°`, `0°`, `+10°`) with `+`
to the right and `+` upward, per the frame convention above. A one-line legend
in the panel corner says `angles: off-axis, + right / + up` so nobody has to
infer it.

**Iso-angle rings** (off by default). Closed polylines at each ladder value,
sampled over azimuth, labelled once at the top. On a fisheye these are the
honest picture of the projection and make an off-centre principal point
unmissable; on a long lens they are four barely-distinguishable circles, which
is why they are opt-in.

**Distortion field.** A `grid_cols × rows` grid over the image (rows chosen to
keep cells square). For each grid pixel `u`:

1. `ray = pixel_to_ray(u)` — where this pixel actually looks;
2. `u_ref = reference_project(ray)` — where the family's ideal map would have
   put that same ray;
3. draw an arrow from `u_ref` to `u_ref + s·(u − u_ref)`, `s` the exaggeration.

Arrow direction therefore reads as "the lens moved this ray *here* from
*there*", which is the direction a rectification would undo. Arrowheads scale
with magnitude and are omitted below 1 panel pixel.

*Auto scale* picks the smallest `s` from `{1, 2, 3, 5, 10, 20, 50}` that brings
the largest displacement in the grid to at least 8 panel pixels, capped so that
no arrow exceeds one grid cell. The legend states both, always:
`max 12.4 px · shown ×3` — an exaggerated field that does not admit it is a
lie, and this is a diagnostic tool.

The scale is computed **per camera**, not per frame and not per reconstruction.
Per frame would flicker; per reconstruction would scale every camera to the most
distorted one and leave a mild lens with a field too small to read. Per camera
means switching cameras can change the scale under the user, which is accepted:
each camera is exaggerated exactly enough to show *its own* distortion, and the
legend says so every frame. See § "Decisions".

**Spline domain marker.** For `SfmtoolFisheye` / `SfmtoolPinhole`, an extra
dashed iso-contour at `bspline_theta_max` / `atan(bspline_rho_max)`, labelled
`spline domain`. Beyond it the correction is held constant and the map continues
linearly, so a lens whose image corners fall outside that contour is being
extrapolated — a fact worth seeing on the image, since it explains residuals
that appear only at the corners.

**Equirectangular.** The axes and rings are drawn from the same ray maths with
no special case (`ray_to_pixel` is defined over the whole sphere and the ticks
come out evenly spaced, which is the correct picture). The distortion field is
suppressed: the model has no distortion parameters by construction. The ladder
extends to ±180° horizontally and ±90° vertically.

**Hover readout.** The text the layer contributes to the panel's tooltip, per
§ "Hover: one tooltip, composed" above:

```
pixel  (1204.5, 733.0)
ray    (0.281, −0.104, −0.954)
off-axis 17.3°   azimuth 159.7°
distortion  +4.21 px
```

`distortion` is the same `|u − u_ref|` the arrows draw, at the exact pixel under
the cursor rather than at a grid node, and is omitted for a model with no
distortion. This is the cheapest high-value part of the whole overlay: it turns
the image into a calibrated protractor.

### Interaction with the selected point

Unchanged from
[gui-multi-panel-image-browser.md](gui-multi-panel-image-browser.md) §
"Interaction with 3D point selection": the selected 3D point's observation is
highlighted according to the active `OverlayMode`, and the intrinsics layer
neither adds to nor suppresses that. Its contribution is contextual — with the
layer on, the selected observation is visibly *somewhere* in the distortion
field and at some readable off-axis angle, which is the point of composing them.

---

## The Intrinsics panel

A sixth dock tab, `Tab::IntrinsicsDetail`, title **"Intrinsics"**, defaulting
into the same top-right tab group as Image Detail and Point Track, as the
non-active tab. It is a detail view of a selection like both of its neighbours,
and like both it is fully re-dockable.

**Empty state.** `No camera selected` centred, with a line beneath:
`Select a camera under Camera Intrinsics in the Scene panel, or select an
image.` — the second half being the discoverable route, since most users will
reach intrinsics through an image rather than the other way round.

**Populated state**, top to bottom:

### 1. Header

```
kerry_park · Camera #0 · OPENCV_FISHEYE · 480×480 · 26 images        [Copy ▾]
```

The reconstruction name is included because several nodes can be loaded at once
and `CameraRef` carries a `ReconId`; without it the panel would be ambiguous
exactly when it matters. A beta model appends `(beta)` with the registry's note
as tooltip.

`Copy ▾` offers `Parameters (text)`, `Parameters (JSON)`, `K matrix`, and —
when the extrinsics section is showing — `Pose matrix`. A viewer whose numbers
cannot leave it makes users retype them.

### 2. Parameters

The model's parameters in `parameter_names()` order — declaration order, the
same order `sfm inspect` prints:

| Parameter | Value |
|-----------|-------|
| `focal_length` | 240.104 |
| `principal_point_x` | 239.500 |
| `principal_point_y` | 240.112 |
| `radial_distortion_k1` | −0.021 |

Six decimals, right-aligned, monospaced, matching the CLI. The spline models'
`bspline_c{i}` rows are listed after the named parameters in index order (which
is why `parameter_names()` exists), preceded by their domain end.

### 3. Derived

A second table, visually separated, holding what the parameters *mean*:

| Row | Value | Notes |
|-----|-------|-------|
| `fx, fy` | `240.104, 240.104 px/rad` | unit is `px` for perspective models, `px/rad` for equidistant/equirectangular — mislabelling this is the single easiest way to make a fisheye's focal length look absurd |
| aspect `fy/fx` | `1.0000` | hidden when the model has one focal length |
| principal point offset | `(−0.500, +0.112) px · 0.15% of half-diagonal` | from the image centre |
| horizontal FOV | `176.4°` | mid-left pixel to mid-right pixel |
| vertical FOV | `176.5°` | |
| diagonal FOV | `197.2°` | corner to opposite corner |
| max off-axis angle | `98.6°` | largest θ over the four corners; the number that answers "is this really 180°?" |
| 35 mm equivalent | `19.1 mm` | perspective models only; `f_px · 43.267 / diagonal_px`, sensor-independent by construction |
| distortion | `yes — max 12.4 px at the corner` / `none` | from `has_distortion()` plus the field's maximum |

Then `K`, rendered as a 3×3 grid, with the note that it is the optical-frame
matrix and that `P = K · S · [R|t]`.

### 4. Projection plot

Two stacked plots sharing an x axis of incidence angle `θ` in degrees, from 0 to
the max off-axis angle plus 5% margin. Hand-painted with `egui::Painter`: the
crate has no plotting dependency, the app already hand-paints its other
diagnostics, and the reference curve, the azimuth band and the annotation
markers are all custom work that a general plotting widget would not shorten.

**Upper plot — the radial map**, `r(θ)` in pixels:

- solid: the model's actual `|project(ray(θ)) − c|`;
- dashed: the family's ideal map (`f·tan θ` or `f·θ`);
- a shaded band between the min and max over 32 azimuths, drawn only when the
  model is azimuth-dependent (`fx ≠ fy`, or tangential/thin-prism terms
  present). The band is how decentring distortion becomes visible at all —
  a single-azimuth curve hides it completely.

**Lower plot — the residual**, `Δr(θ) = r − r_ref` in pixels, zero line marked.
This is the one that shows the shape of the distortion, since on the upper plot
a 12-pixel departure from a 700-pixel curve is invisible.

**Markers on both**, as labelled vertical rules: θ at the mid-edges, θ at the
corner, the spline domain end where there is one, and the 90° asymptote for
perspective models.

**No distortion.** Both plots still draw — the projection curve is a fact about
the camera whether or not it is distorted, and an empty panel would be a worse
answer than a straight line. The residual plot collapses to its zero line, and a
banner across it reads
`No distortion — this model is exactly {a pinhole | equidistant}`, which is the
"says undistorted" the request asks for, stated in terms that say *what* it is
rather than only what it is not.

### 5. Extrinsics

Shown only when `selected_image` is `Some` — which, by the coupling invariant,
means the selected image uses the camera above. Header:
`Pose · IMG_0001.jpg`.

| Block | Content |
|-------|---------|
| Rotation `R` | 3×3, world-to-camera |
| Translation `t` | 3-vector, world-to-camera |
| `[R \| t]` | the 3×4 pose matrix, as one grid, `Copy`-able |
| Quaternion | `w, x, y, z` |
| Camera centre `C` | `−Rᵀt`, in world units, suffixed with `metadata.world_space_unit` when the file names one |
| Axes in world | `right`, `up`, `forward` = `Rᵀe₀`, `Rᵀe₁`, `−Rᵀe₂` |
| `P = K · S · [R \| t]` | 3×4, **perspective models only** — for fisheye and equirectangular models the row is replaced by `Not a linear projection — this model has no 3×4 P` rather than being silently omitted or, worse, printed anyway |

A caption states the convention once — *world-to-camera, canonical `.sfmr` frame:
Z-up world, camera looks down −Z with +Y up* — linking to
`specs/formats/sfmr-file-format.md`. This is the block most likely to be pasted
into someone else's code, so it says which frame it is in.

**The node transform.** A reconstruction node can carry a similarity transform
from an in-GUI `Align to…` ([gui-scene-graph.md](gui-scene-graph.md) § "Node
Transforms and Alignment"), and what the viewport draws is then *not* the stored
pose. The panel shows the **stored** pose by default, because that is what the
file holds and what a user comparing against the CLI expects. When the node's
transform is not identity, a toggle appears above the block —
`[ stored | × node transform ]` — and the header gains a `(transformed)` marker
while the second is active. Showing one silently would make the panel wrong half
the time, in a way nobody would notice.

**Rigs.** When `recon.rig_frame_data` is present, a further block:

| Row | Source |
|-----|--------|
| Rig | `rigs[rig_index].name`, and `(reference sensor)` when this image's sensor is it |
| Sensor | `sensor_names[s]`, index `s` |
| Frame | `image_frame_indexes[i]`, and the number of images in that frame |
| `sensor_from_rig` | rotation + translation, or `identity (reference sensor)` |

This is the part of "extrinsics" that a rig dataset actually needs and that
nothing in the viewer surfaces today.

### Response type

The panel is nearly read-only; it needs to report only navigation:

```rust
pub struct IntrinsicsDetailResponse {
    /// The user clicked the image name in the extrinsics header.
    pub select_image: Option<ImageRef>,
    pub has_pointer: bool,
}
```

---

## Cross-panel effects of a camera selection

Selecting a camera is a statement about a *set* of images, and the rest of the
viewer should say so:

- **3D viewport** — every frustum whose image uses the selected camera is drawn
  in the "sibling" colour, reusing the per-frustum colour channel the selected-
  track highlight already occupies (`gui-camera-views.md` § "Selection
  highlighting"). The selected image itself keeps its own stronger highlight.
  The two highlights can coexist because they are ranked, not mixed: selected
  image > selected-track member > selected-camera sibling.
- **Image Browser** — the same set gets a thin border in the same colour.
- **Scene Graph** — the camera row is highlighted, and (already, by the
  invariant) so is the selected image's row.
- **Viewport HUD** — unchanged. The HUD owns 3D *display controls*
  ([gui-viewport-hud.md](gui-viewport-hud.md)); a selection is not one.

For a single-camera reconstruction this highlights everything, which is correct
and also uninformative, so it is suppressed when every image in the node uses
the selected camera.

---

## Behaviour by camera model

The one table to check an implementation against:

| Model family | Axes / rings | Distortion field | Radial plot reference | `P` |
|--------------|--------------|------------------|-----------------------|-----|
| Pinhole, SimplePinhole | straight, to <90° | suppressed (`has_distortion` false) | `f·tan θ`, residual ≡ 0 | yes |
| SimpleRadial, Radial, OpenCV, FullOpenCV | bent, to <90° | shown | `f·tan θ` | yes |
| SfmtoolPinhole | bent, to <90° | shown; domain contour at `atan(ρ_max)` | `f·tan θ` | yes |
| SimpleRadialFisheye, RadialFisheye, OpenCVFisheye | bent, past 90° | shown | `f·θ` | no |
| ThinPrismFisheye, RadTanThinPrismFisheye | bent, past 90° | shown, azimuth band on the plot | `f·θ` | no |
| EquidistantFisheye | straight in θ | suppressed | `f·θ`, residual ≡ 0 | no |
| SfmtoolFisheye | bent; domain contour at `θ_max` | shown | `f·θ` | no |
| Equirectangular | linear in lon/lat, ladder to ±180°/±90° | suppressed | identity, residual ≡ 0 | no |

Every model with zero-valued distortion coefficients falls into its family's
"suppressed" row via `has_distortion()`, including a spline model whose spline is
inactive — the kernels short-circuit those to the exact base arithmetic, so
reporting distortion for them would be a lie the projection does not tell.

---

## Performance and caching

Nothing here is per-frame work if it is cached correctly, and all of it is
expensive enough to matter if it is not: `undistort` is iterative for the OpenCV
family, and the arrow field is a few hundred round trips.

| Product | Cost | Cached on | Invalidated by |
|---------|------|-----------|----------------|
| Distortion field | `cols × rows` ray round trips | `(CameraRef, cols, rows)` | camera change, grid density change, node reload |
| Radial profile | `samples × (1 + 32 azimuths)` | `(CameraRef, samples)` | camera change, node reload |
| FOV / derived rows | 4 corners + 4 edges | `CameraRef` | camera change, node reload |
| Axis polylines | ~1 sample per 4 panel px | `(CameraRef, zoom bucket)` | camera change, zoom crossing a bucket |
| Hover readout | 1 round trip | not cached | — |

Axis polylines are the only product that depends on the *view*, and they are
bucketed by zoom (powers of two) rather than recomputed continuously, so a pinch
gesture does not rebuild them 60 times a second. Caches are dropped by
`forget_recon` alongside the panels' textures.

Because the layer is independent of `OverlayMode`, its cost is now *additive* to
a feature mode's rather than replacing it — the worst case is 5000 heatmap
circles plus the intrinsics layer in one frame. That is fine, and it is fine for
a structural reason worth stating: the layer's draw cost depends only on grid
density and polyline sampling, both bounded by the panel's pixel count and
neither by the feature count. There is no combination of settings in which
enabling the layer scales with the data.

The largest realistic camera count is a per-image-intrinsics solve, so caches
are keyed by `CameraRef` and hold at most a handful of entries — a plain
`HashMap` with no eviction, cleared on node close.

---

## Testing

Following the crate's existing split — pure maths in `sfmtool-core`, headless
egui frames for panels, real windows only for what genuinely needs one:

**`sfmtool-core`, `camera::report` unit tests** (no GUI, runs everywhere):
- `field_of_view` on a known pinhole matches the closed form
  `2·atan(w / 2fx)`; on `EquidistantFisheye` with `f = w/π` gives 180°
  edge-to-edge.
- `radial_profile`'s reference curve equals the actual curve, to `1e-12`, for
  every model whose `has_distortion()` is false — one case per registry variant,
  asserted complete against `MODEL_COUNT` the way the registry's own corpus is.
- `distortion_field` is identically zero for those same models, and non-zero
  and radially symmetric for `SimpleRadial` with `k1 < 0`.
- Round trip: `ray_to_pixel(pixel_to_ray(u)) ≈ u` at every grid node used by the
  field, which is what makes the arrows meaningful at all.
- `equiv_focal_length_35mm` returns `None` for every fisheye and equirectangular
  model.
- `parameter_names()` is a permutation of the keys `SfmrCamera::from` writes,
  for every variant — the property that keeps the table from silently dropping a
  parameter when a model gains one.

**`sfm-explorer` lib tests** (headless, `Context::run_ui`, the
`point_track_detail/tests.rs` pattern):
- The selection coupling, one test per row of the truth table, driven through
  `AppState::select_image` / `select_camera` with no UI at all.
- `scene_graph/tests.rs`: the group renders with the right label and count; a
  click on a camera row emits `select_camera` and a double-click emits
  `zoom_to_camera`; the group is expanded by default at 2 cameras and collapsed
  at 5; a camera used by no image renders `0 images`.
- The reconstruction row shows all three counts at a comfortable panel width and
  elides them in the specified order — cameras, then images — as the width
  shrinks, restoring them when it grows back.
- The Intrinsics panel renders its empty state with no selection, its populated
  state with one, and the extrinsics block only when an image is selected;
  the `P` row is absent for a fisheye fixture.
- The intrinsics layer composes: with `overlay_mode: Features` and
  `intrinsics.enabled`, one frame contains both the feature ellipses and the
  principal-point marker, and turning the layer off leaves the feature draw
  calls unchanged.
- The layer's settings popup opens only when the checkbox is ticked, and shows
  the `No distortion` line instead of the distortion row for a pinhole fixture.
- With the layer off, the hover tooltip over a feature is identical to the
  tooltip the panel produces today — the regression that a composed tooltip
  most plausibly breaks.
- The node-transform toggle appears only when the node transform is
  non-identity.

**Fixtures.** `kerry_park` is the rig case (2 fisheyes, 24 frames) and
`seoul_bull_sculpture` the ordinary single-camera one; the model-family coverage
comes from synthetic `CameraIntrinsics` values in the core tests rather than
from datasets, since no checked-in dataset exercises thin-prism or
equirectangular.

No new windowed (`ui_basic`) tests: nothing here depends on real OS input the
way the context menu does.

---

## Implementation phases

Each phase leaves the viewer in a shippable state.

1. **Vocabulary and data model.** `CameraRef`, `selected_camera`, the two
   `AppState` setters and the coupling truth table with its tests; the
   `Cameras` → `Camera Images` rename with `show_cameras` →
   `show_camera_images` and the three spec docs that name it;
   `AlignSource::Cameras` → `Camera Poses`; the reconstruction row's counts.
   No new UI surface — this phase is only visible as better labels.
2. **`camera::report` and `parameter_names()`.** Pure core work, fully
   unit-tested, with no consumer yet.
3. **Scene Graph group.** The Camera Intrinsics rows, click and double-click,
   the cross-panel sibling highlight.
4. **Intrinsics panel.** Header, parameters, derived table, `K`, extrinsics,
   rig block, copy menu. No plot yet — the tables alone already replace the
   `sfm inspect` round trip.
5. **Projection plot.** Both stacked plots, reference curve, azimuth band,
   markers, the no-distortion banner.
6. **Image Detail overlay layer.** The checkbox, the settings popup and the
   compositing rules first, with only the principal point and centre offset
   drawn — that is the smallest thing that proves the layer composes correctly
   over every existing mode. Then axes and ticks, then the distortion field,
   then the composed hover readout.

Phases 3–6 are independent of one another once 1 and 2 land, so they can be
reordered or dropped without stranding anything.

---

## Deliberately out of scope

- **Editing intrinsics.** The viewer is a viewer. Refining a camera is
  `sfm xform bundle-adjust`'s job, and a panel that let you type a focal length
  would immediately raise the question of what it means for the loaded file.
- **Comparing two cameras side by side.** Real for rig work (how do the two
  kerry_park fisheyes differ?), but it needs a second selection and a diff
  presentation, and it is a poor reason to complicate the first version.
- **Undistorted preview** in the Image Detail panel. `sfm undistort` exists, the
  distorted-frustum path already renders the corrected geometry in 3D, and a
  full remap per frame in a 2D panel is a different feature with a different
  cost.
- **A per-camera image list** inside the intrinsics group. The sibling highlight
  in the browser answers "which images" better than a nested list would, and
  nesting a virtualized list inside a group inside a node is a layout the tree
  does not currently do.
- **Wiring the Python `_CAMERA_PARAM_NAMES` table to `parameter_names()`.**
  Right to do, unrelated to this feature's UI, and it touches the PyO3 surface —
  a separate change.

---

## Decisions

**Independent layer, not an `OverlayMode` variant** (2026-08-23). The first
draft made the intrinsics overlay an eighth exclusive mode, matching the
existing toolbar. Settled the other way: the questions worth asking are the
joint ones — whether keypoints crowd the distorted rim, whether the
reprojection-error heatmap is hot where the distortion field is largest — and an
exclusive mode turns each of those into flipping back and forth from memory. The
cost is the compositing the layer now has to specify (z-order, a colour that
survives an arbitrary colormap underneath, one composed tooltip), which
§ "Compositing with the feature layers" and § "Hover: one tooltip, composed"
carry. `OverlayMode` is left untouched.

**Three counts on the reconstruction row** (2026-08-23).
`1.2M pts · 243 imgs · 2 cams` rather than today's two. The objection was width
— it is the longest row in a panel that defaults to 18% of the window — and the
answer is that the camera count is the first thing dropped when the row cannot
fit it, not that it goes unsaid. See § "Terminology" for the elision order.

**The distortion auto-scale stays per camera** (2026-08-23). The exaggeration is
computed per camera rather than fixed across the reconstruction, so switching
between two cameras with different distortion magnitudes does change the arrow
scale under the user. That is the intended behaviour: each camera is scaled to
show *its own* distortion legibly, which is what the arrows are for, and the
legend states the scale on every frame so the change is never silent. Fixing one
scale across a reconstruction would make the milder camera's field invisible in
order to keep a comparison nobody asked for. It is computed per camera and not
per frame, so it never flickers.

**Double-click on a camera row zooms the viewport** (2026-08-23). It fits every
image using that camera — for a rig, one sensor's whole trajectory in a single
gesture, which nothing else in the viewer does. The objection was that it makes
a third double-click target in one tree (node → zoom to node, image → camera
view); accepted anyway, because the three targets are consistent rather than
arbitrary: double-clicking a row frames what that row denotes, and a camera row
denotes a set of images.

---

## Open questions

None outstanding. Resolved questions move to § "Decisions" above with the
reasoning that settled them.
