# Camera Intrinsics: Scene-Graph Node, Image Overlay, Detail Panel

**Status:** implemented — phase 1 (vocabulary and data model), phase 2
(`camera::report` and `parameter_names()`), phase 3 (the Scene Graph group),
phase 4 (the Camera Intrinsics panel), phase 5 (the projection plot, with the
trustworthy domain it needed) and phase 6 (the Image Detail overlay layer).

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
3. a **Camera Intrinsics** dock panel: the parameter table, the distortion
   curve, and — when an image is selected — that image's extrinsics.

Related specs: [scene-graph.md](scene-graph.md) (the tree and the
selection model), [multi-panel-image-browser.md](multi-panel-image-browser.md)
(the Image Detail panel and its overlay modes),
[camera-views.md](camera-views.md) (frustum geometry, distorted frustum
rendering, the FOV maths this reuses), and
[specs/core/camera/camera-model-registry.md](../core/camera/camera-model-registry.md) (the one
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
  `camera-views.md` throughout: all about the posed view. Unchanged.

The `SceneNode::show_cameras` field drives frustum and image-quad visibility, so
under the new vocabulary it is `show_camera_images`. It is referenced in
`scene-graph.md`, `camera-views.md` and `viewport-hud.md`; the rename
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
deriving it from `selected_image` would make the Camera Intrinsics panel go
blank the moment they clicked anything else. What is derived is the
*constraint* between the two, enforced in one place.

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
| Clear camera selection (`Esc` again) | **cleared** | `None` |
| Select a different reconstruction | filtered to that recon | filtered to that recon |
| Close / reload the owning node | cleared | cleared |

The "clear camera" row is the invariant closing itself: an image implies its
camera, so clearing the camera has to take the image with it, whatever order a
caller happens to use. The `Esc` sequence never reaches it — the first press
clears the image, the second finds none and clears the camera — but the
guarantee is that no caller *can* reach the forbidden state, not that none
currently tries.

Two other rows are choices rather than consequences, and both go the way the
user's request implies:

- **Selecting the camera an image already uses keeps the image.** The user's
  rule is that a *different* intrinsics deselects the image; the same intrinsics
  is not different, and clearing there would make clicking the highlighted
  intrinsics row of the image you are looking at throw that image away.
- **Clearing the image keeps the camera.** Deselecting a photograph is not a
  statement about the lens, and collapsing the Camera Intrinsics panel because
  the user dismissed an image would be a surprise. `Esc` pressed twice clears
  both — the second press, seeing no image, clears the camera.

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
[camera-model-registry.md](../core/camera/camera-model-registry.md) sets out:

```rust
impl CameraModel {
    /// This model's parameter names in declaration order — the order
    /// `sfm inspect` prints and the Camera Intrinsics panel tabulates.
    ///
    /// For the two spline models the trailing `bspline_c{i}` names depend on
    /// the coefficient count, so this returns an owned `Vec<Cow<'static, str>>`
    /// rather than a `&'static [&'static str]`.
    pub fn parameter_names(&self) -> Vec<Cow<'static, str>>;
}
```

Values come with them: `CameraIntrinsics::parameters()` returns
`(name, value)` in that same order, pairing `parameter_names()` with what
`SfmrCamera::from` writes, so a table can never show a parameter the file does
not carry or lose one it does. That is what the tree's hover tooltip and the
panel's table are both built from.

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
    pub horizontal: f64,          // left edge to right edge, through the centre
    pub vertical: f64,            // top edge to bottom edge
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
    pub theta_deg: f64,           // incidence angle of the sampled ray
}

pub fn field_of_view(cam: &CameraIntrinsics) -> Option<FieldOfView>;
pub fn radial_profile(cam: &CameraIntrinsics, azimuth_deg: f64, samples: usize)
    -> Vec<RadialSample>;
pub fn distortion_field(cam: &CameraIntrinsics, cols: usize, rows: usize)
    -> Vec<DistortionSample>;
/// The same displacement at **one** pixel rather than at a grid node —
/// what the overlay's hover readout wants. `distortion_field` is defined in
/// terms of it, so the two cannot disagree about the ideal map.
pub fn displacement_at(cam: &CameraIntrinsics, u: f64, v: f64)
    -> Option<DistortionSample>;
/// 35 mm-equivalent focal length: `f_px · 43.267 / diagonal_px`. `None` for
/// models whose focal length is pixels-per-radian rather than pixels.
pub fn equiv_focal_length_35mm(cam: &CameraIntrinsics) -> Option<f64>;
/// The incidence angle of the ray through a pixel — the frame convention,
/// spelled once, for the overlay's hover readout and the plot's edge markers.
pub fn off_axis_angle_deg(cam: &CameraIntrinsics, u: f64, v: f64) -> f64;
/// The largest incidence angle at which the model still describes a lens,
/// or `None` when it does so at every angle. See "The trustworthy domain".
pub fn trustworthy_max_theta_deg(cam: &CameraIntrinsics) -> Option<f64>;
```

### The trustworthy domain

Every function above is defined over the whole image **rectangle**, and phase 4
found out the hard way what that costs: on `kerry_park`'s real `OPENCV_FISHEYE`
— a circular fisheye in a square 480 × 480 frame — the displacement field's
maximum over a 16 × 16 grid is **272.7 px**, twenty times the 13 px the lens
actually displaces anything. The image rectangle's corners sit 150° off-axis,
outside the lens's image circle, where the `k1..k4` polynomial is evaluated
with nothing constraining it and has folded: its forward map takes θ = 132.7°
to a radius of 8.8 px, where the equidistant ideal puts it at 299 px. The
number was a true statement about two forward maps and a false one about a
camera.

`trustworthy_max_theta_deg` is how a consumer finds out where that starts, and
`DistortionSample::theta_deg` is what lets it filter — a sample's source pixel
is recoverable from its index, but the angle it looks at is not recoverable
from anything.

This is a property of the **parameterization**, not of fisheyes:

| Models | Bound |
|--------|-------|
| `OPENCV_FISHEYE`, `RADIAL_FISHEYE`, `THIN_PRISM_FISHEYE`, `RAD_TAN_THIN_PRISM_FISHEYE` | the first θ at which the *distorted* radius reaches `FISHEYE_BLEND_START_RAD`, or the radius's own peak if it turns over first |
| everything else | `None` — trustworthy at every angle |

The four bounded models are exactly the three call sites of `blend_fisheye_ray`
in `camera/distortion.rs` (`OPENCV_FISHEYE` and `RADIAL_FISHEYE` share one).
That function exists precisely because "high-order distortion polynomials
become unreliable approaching their peak": past `FISHEYE_BLEND_START_RAD` of
distorted radius, the model's own inverse stops inverting the polynomial and
slews toward the identity ray, so above that radius the forward and inverse
maps are no longer each other's inverse and neither describes the lens.

The unbounded classifications are each their own argument, not a default:

- `SIMPLE_RADIAL_FISHEYE` is **already** excluded from the blend, deliberately
  and with the reasoning written on `simple_radial_fisheye_to_ray`: with one
  coefficient `θ_d = θ·(1 + k1·θ²)` there is nothing to distrust. This section
  is that argument generalised.
- The two spline models hold `δ` constant past `bspline_*_max`, so their radial
  map continues linearly with slope `f`, and they enforce `1 + δ'(θ) > 0` as a
  construction invariant. There is no peak to approach at any angle.
- The exact maps — the plain pinholes, `EQUIDISTANT_FISHEYE`, `EQUIRECTANGULAR`
  — have no polynomial, and the perspective polynomials are already hard
  bounded at the 90° their projective divide refuses.
- A camera whose `has_distortion()` is `false` is `None` whatever its model: it
  **is** its family's ideal map, the blend interpolates between two identical
  rays, and there is no polynomial to fold. This is the first gate the function
  applies, so a zeroed `OPENCV_FISHEYE` is exact everywhere, as it should be.

The bound is found by walking the camera's own `ray_to_pixel` — a coarse sweep
in θ over eight azimuths brackets the first step at which the distorted radius
either reaches the blend radius or stops increasing, then the bracket is
refined — rather than by reading coefficients, because a second spelling of
four different polynomials is a second thing to keep in step. Eight azimuths
because the thin-prism models are not radially symmetric and the bound is the
*first* azimuth to go.

The function's own `match` is exhaustive with no `_` arm, so a newly registered
model does not build until somebody classifies it; the test corpus asserts the
same classification independently against `MODEL_COUNT`.

**The ideal ("reference") map** is what "distortion" is measured against, and it
is per-family, not per-model:

| Family | Ideal map | Members |
|--------|-----------|---------|
| Perspective | `r = f·tan θ` (pinhole with the same `fx, fy, cx, cy`) | Pinhole, SimplePinhole, SimpleRadial, Radial, OpenCV, FullOpenCV, SfmtoolPinhole |
| Fisheye | `r = f·θ` (equidistant with the same `fx, fy, cx, cy`) | OpenCVFisheye, SimpleRadialFisheye, RadialFisheye, ThinPrismFisheye, RadTanThinPrismFisheye, EquidistantFisheye, SfmtoolFisheye |
| Equirectangular | itself | Equirectangular |

Because the reference carries the *same* `fx, fy, cx, cy`, the displacement it
measures is pure lens distortion: neither the focal length, nor a non-square
pixel aspect, nor the principal-point offset leaks into it. A model whose
`has_distortion()` is `false` produces an identically-zero field, and the code
paths that draw it are skipped rather than drawing zero-length arrows.

The fisheye ideal therefore keeps `fx` and `fy` **separate**, even though
`CameraModel::EquidistantFisheye` carries only one focal length: three of the
seven fisheye models carry two, and collapsing them would fold the pixel aspect
ratio into a measurement that is supposed to be about the lens alone. It is
implemented as arithmetic rather than as a substitute `CameraIntrinsics` for
exactly that reason.

**Field of view is swept, not subtracted.** Each of `horizontal`, `vertical`
and `diagonal` is the sum of two hops — boundary to image centre, image centre
to opposite boundary — rather than the single angle between the two boundary
rays. The two agree below 180° and only there: the angle between two directions
saturates at 180° and folds back, so a 183° fisheye measured end to end reports
177°, and the two vertical edges of a full equirectangular panorama, being *the
same ray*, report 0°. Both are wrong for exactly the lenses this panel exists to
explain. The sum is exact whenever the three rays are coplanar, which they are
for a perspective or equidistant model with a centred principal point; an
off-centre principal point puts them on a shallow cone and the sum runs slightly
over, which is a far smaller error than the fold it replaces.

`max_off_axis` stays a maximum over the four corners, and for an
equirectangular panorama that means it reads 90° whatever the panorama covers —
the corners of an equirectangular image are the two poles. The `horizontal` /
`vertical` pair is the honest reading there.

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
  carry a `β` suffix on the model name, and the registry's beta note
  (`CameraModel::beta_note`) is appended to the row's hover tooltip under a
  separator. Not a tooltip on the `β` itself: egui hangs a tooltip off a whole
  widget, and the row is one button, so a sub-span of its label has nowhere to
  put one.
- **Click**: `select_camera(Some(CameraRef))` — with the deselect-the-image
  consequence from the coupling table.
- **Double-click**: zoom the 3D viewport to fit every image using this camera —
  the same `zoom_to_fit_points` call `zoom_to_node` makes, over that subset of
  the node's **camera centres** rather than over the node's points. (The two
  frame different things, which the first draft of this line elided: a node's
  zoom-to-fit frames its point cloud, and a camera has no points of its own.)
  For a rig this frames one sensor's whole trajectory
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
  list is capped at `LIST_MAX_HEIGHT` — the same cap the images list uses,
  renamed from `CAMERA_LIST_HEIGHT` in this phase because it is now the height
  of two different lists and was never a list of cameras.
- **No hover channel.** Cross-panel hover is a two-field protocol
  (`hovered_image` / `hovered_point`) with an ownership rule per panel
  ([cross-panel-hover.md](cross-panel-hover.md)); a third field would
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
    /// Draw the layer at all. On by default: the layer is the reference
    /// frame the features sit in, and the first look at a photograph in this
    /// panel is a diagnostic one — where the principal point is, how the lens
    /// bends the rim — whether a human or an agent (via `screenshot`) is
    /// looking. `I` turns it off in one keystroke for the clean view.
    pub enabled: bool,            // default true
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

**Principal point.** A reticle at `(cx, cy)` — a 3.5 px open ring with four
arms reaching outward from just outside it — drawn last and fully opaque per the
compositing rules above. This is the one mark that must never be lost under a
dense feature overlay.

> _Correction (2026-08-23, phase 6). This said "a small cross (4 px arms) plus a
> 3 px open circle". Drawn that way in the real viewer, with the halo every
> stroke of this layer carries, the cross and the ring merge into a filled dark
> disc a few pixels across with no readable shape in it. Leaving the middle open
> is what makes it a mark you can put on a pixel. The halo also narrowed from
> two pixels wider than its stroke to one and a half, for the same reason._

**Image centre and the offset.** A faint `+` at `(w/2, h/2)` and a thin
connector to the principal point, labelled with the offset in pixels and as a
percentage of the half-diagonal: `Δ (−12.4, +3.1) px · 0.4%`. The whole clause —
the `+`, the connector and the label together — is suppressed when the offset is
under half a pixel: there the two marks land on each other and a second reticle
inside the first one's halo reads as one blob rather than as two coincident
facts. Both checked-in fixtures are in exactly that case. This answers "is the
principal point where it should be?" at a glance, which the number alone does
not.

**Angular axes.** Two polylines through the principal point, sampled by
projecting rays rather than by drawing straight lines:

- horizontal: `ray(α) = (sin α, 0, −cos α)` for `α` on the tick ladder;
- vertical: `ray(ε) = (0, sin ε, −cos ε)`.

Each is drawn as a polyline through densely sampled `ray_to_pixel` results (one
sample per ~4 panel pixels), with tick marks and labels at the ladder values.

> _Correction (2026-08-23, phase 6). This section said "**they are not
> straight** — under distortion they bend, and the bend *is* the distortion".
> That holds only for a model with decentring terms. A purely **radial** model —
> most of the registry, and both checked-in fixtures — moves every point along
> its own radius from the principal point, so a line *through* the principal
> point stays exactly straight however violent the lens: on `kerry_park`'s
> `OPENCV_FISHEYE` the sampled vertical axis is straight to under a hundredth of
> a pixel. What a radial lens does to these axes is bunch the **ticks** along
> them — evenly spaced angles landing at unevenly spaced pixels — and that is
> the reading the ticks are for. Sampling rather than drawing two straight lines
> is still the right implementation, because it is what makes the thin-prism and
> tangential models' real curvature appear with no special case._

Sampling stops at the first `ray_to_pixel` that returns `None` (outside the
model's domain), that leaves the image by more than 5% of its diagonal, or —
phase 6 — at which the **radius from the principal point stops increasing**. An
axis is a scale only where it is monotone: `kerry_park`'s polynomial turns over
near 130° and the radius crashes from 191 px back to 6 px, which without the
check draws the axis back through everything it has already drawn and lands a
confident `−120°` tick between the `−60°` and `−30°` ones.

*Tick ladder*: the **finest** of `1°, 2°, 5°, 10°, 15°, 30°, 45°` that keeps
adjacent ticks at least 48 panel pixels apart at the current zoom — the panel
zooms to 32×, and a ladder fixed at load time would be useless at both ends.
(This section originally said "coarsest", which is the wrong end of the ladder:
the coarsest step always clears the spacing and would put three ticks on a
zoomed-in long lens.) The whole grid is resampled when the scale crosses a
half-octave bucket rather than every frame, per § "Performance and caching".
Labels are signed (`−20°`, `−10°`, `0°`, `+10°`) with `+` to the right and `+`
upward, per the frame convention above; `0°` is labelled once, on the horizontal
axis, since both axes cross there and the principal-point marker is already
sitting on it. A one-line legend in the panel corner says
`angles: off-axis, + right / + up` so nobody has to infer it.

**Iso-angle rings** (off by default). Closed polylines at each ladder value,
sampled over azimuth, labelled once — on the up-and-left diagonal rather than at
the top, because the top of a `30°` ring is exactly where the vertical axis's
own `+30°` tick lands and the two labels come out stacked saying the same thing
twice. On a fisheye these are the honest picture of the projection and make an
off-centre principal point unmissable; on a long lens they are four
barely-distinguishable circles, which is why they are opt-in.

**Ticks and rings stop at the trustworthy bound; the axes do not.** A tick
claims "this pixel is 60° off axis" and a ring carries the same claim, so
neither is drawn past `trustworthy_max_theta_deg`. The axes themselves continue
past it **dashed** — the same treatment the projection plot gives its curve — so
a reader still sees how much frame lies outside the modelled domain. On
`kerry_park` the frame's mid-edge is past 100° against an 84° bound, so that is
most of the outer third of the picture.

**Distortion field.** A `grid_cols × rows` grid over the image (rows chosen to
keep cells square). For each grid pixel `u`:

1. `ray = pixel_to_ray(u)` — where this pixel actually looks;
2. `u_ref = reference_project(ray)` — where the family's ideal map would have
   put that same ray;
3. draw an arrow from `u` to `u + s·(u_ref − u)`, `s` the exaggeration.

Arrow direction therefore reads as "the content under *this* pixel belongs
*there*" — the correction itself, drawn on the pixel it corrects.

> _Correction (2026-08-24, phase 6). Step 3 said to draw from `u_ref` to
> `u_ref + s·(u − u_ref)`, and the sentence after it justified that as "the
> direction a rectification would undo" — an instruction and a rationale
> naming opposite conventions. The rationale was the right one. Both directions
> are arithmetically true and the old one had the correct sign (a positive `k1`
> puts `u` outside `u_ref`, so those arrows pointed outward, which is faithfully
> how a pixel moves from undistorted to distorted); it is nonetheless the wrong
> thing to draw **on a photograph**. The field is painted on the distorted
> image, where every pixel on screen is an actual pixel, so an arrow tailed at
> `u_ref` starts at a point that does not exist in the picture being looked at
> — and an arrow on a photograph is read as "this content moves that way",
> which is only true when the tail is the real pixel. Tailing at `u` also puts
> every tail on the exact grid lattice rather than the warped one `u_ref` forms.
> Found by looking at `seoul_bull_sculpture` in the viewer: every automated check
> passed on a field pointing the wrong way, because nothing asserted direction._

Arrowheads scale
with magnitude and are omitted below **3** panel pixels — this section said 1,
but at two or three pixels two barbs on a short shaft render as a blob, which is
what the first draft put across the middle of `kerry_park` where the lens
displaces almost nothing. Below three quarters of a pixel no arrow is drawn at
all: that is the absence of a measurement, not a small one.

**Only the trustworthy half of the grid is drawn as arrows.** This section's
auto scale fits "the largest displacement in the grid", and on a circular
fisheye that is not a lens at all: `kerry_park`'s frame has corners 150°
off-axis, outside the lens's image circle, where the `k1..k4` polynomial folds
and reports 273 px against the 13 px the lens actually applies. Fitting to that
picks ×1 in a narrow panel and leaves every real arrow invisible, and the
legend's `max N px` would be quoting the artefact. So the field is split by
`DistortionSample::theta_deg` against `trustworthy_max_theta_deg`, and the two
halves differ in kind rather than in degree:

- **inside the bound** — an arrow: scaled to, counted, and the maximum the
  legend quotes;
- **outside it** — a small open dot at the grid node and nothing else. The node
  was sampled and there is no measurement there, which is a different statement
  from "the lens displaces this ray by 240 pixels". Drawn as arrows at the
  trustworthy scale they would also throw a dozen frame-crossing strokes across
  the picture.

The plot solved the same problem for a curve by shading, dotting and excluding
from the range; a field is not a curve — there is no continuous path to dot, and
the region is a ring outside the frame rather than a tail — so what carries over
is the principle: the extrapolated part stays visible, is distinguished in kind,
and is out of every number. Its boundary is the labelled dashed contour the axes
draw.

*Auto scale* picks the smallest `s` from `{1, 2, 3, 5, 10, 20, 50}` that brings
the largest **trustworthy** displacement to at least 8 panel pixels, capped so
that no arrow exceeds one grid cell. The legend states both, always:
`distortion max 12.4 px · shown ×3` — an exaggerated field that does not admit
it is a lie, and this is a diagnostic tool — with a second line
`96 of 244 nodes past 84.5° · marked, not measured` whenever the bound excluded
anything.

The scale is computed **per camera**, not per frame and not per reconstruction.
Per frame would flicker; per reconstruction would scale every camera to the most
distorted one and leave a mild lens with a field too small to read. Per camera
means switching cameras can change the scale under the user, which is accepted:
each camera is exaggerated exactly enough to show *its own* distortion, and the
legend says so every frame. See § "Decisions".

The 8-panel-pixel floor is evaluated at the panel's **fit** scale, not at the
live one. "At least 8 panel pixels" and "computed per camera, not per frame" are
in tension, since panel pixels depend on the zoom: a scale honestly recomputed
against the live view steps down the ladder as the user zooms in, which is the
flicker this section is ruling out. Fixing it at the fit scale resolves it the
way the section asks — the multiplier moves only when the camera or the panel
size does, and zooming in enlarges the arrows along with the photograph. The
one-cell cap is in image pixels on both sides and is view-independent already.

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
distortion  4.21 px
```

`distortion` is the same `|u − u_ref|` the arrows draw, at the exact pixel under
the cursor rather than at a grid node — `camera::report::displacement_at`, which
phase 6 made public for exactly this and which `distortion_field` is now defined
in terms of, so the two cannot disagree about the ideal map. It carries no sign:
it is a magnitude, and the `+` this section originally showed would be reporting
a direction the figure does not have.

The line is omitted entirely for a model with no distortion, and past
`trustworthy_max_theta_deg` it reads `distortion  not modelled past 84.5°` with
**no figure** — beside "off-axis 137.2°" a number there would be read as a
measurement, and it is a fold in a polynomial. Same call as the arrows and the
legend, from the same flag.

Azimuth is measured in the frame `radial_profile` sweeps: `0°` is `+X` (right),
`90°` is `+Y` (up). It is omitted within 0.05° of the optical axis, where "which
way round the axis" has no answer.

This is the cheapest high-value part of the whole overlay: it turns the image
into a calibrated protractor.

### Interaction with the selected point

Unchanged from
[multi-panel-image-browser.md](multi-panel-image-browser.md) §
"Interaction with 3D point selection": the selected 3D point's observation is
highlighted according to the active `OverlayMode`, and the intrinsics layer
neither adds to nor suppresses that. Its contribution is contextual — with the
layer on, the selected observation is visibly *somewhere* in the distortion
field and at some readable off-axis angle, which is the point of composing them.

---

## The Camera Intrinsics panel

A sixth dock tab, `Tab::IntrinsicsDetail`, title **"Camera Intrinsics"**,
defaulting into the same top-right tab group as Image Detail and Point Track,
as the non-active tab. It is a detail view of a selection like both of its
neighbours, and like both it is fully re-dockable.

**Empty state.** `No camera selected` centred, with a line beneath:
`Select a camera under Camera Intrinsics in the Scene panel, or select an
image.` — the second half being the discoverable route, since most users will
reach intrinsics through an image rather than the other way round. With no file
loaded at all the panel says `No reconstruction loaded`, which is what its three
dock neighbours say and is a truer answer than pointing at a tree with nothing
in it.

**Populated state**, top to bottom:

### 1. Header

```
kerry_park · Camera #0 · OPENCV_FISHEYE · 480×480 · 26 images        [Copy ▾]
```

The reconstruction name is included because several nodes can be loaded at once
and `CameraRef` carries a `ReconId`; without it the panel would be ambiguous
exactly when it matters. A beta model appends `(beta)` with the registry's note
as tooltip — and here that really is a tooltip on the `(beta)` itself, unlike
the tree's `β`. The constraint phase 3 hit is that egui hangs a tooltip off a
whole *widget*, so a sub-span of one label has nowhere to put one; a tree row is
a single button, but this header is a run of separate labels, so `(beta)` is a
widget in its own right.

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
| distortion | `yes — max 13.0 px inside 84.1°` / `yes — max 12.4 px over the image` / `none` | from `has_distortion()` plus the field's maximum, bounded by § "The trustworthy domain" |

Then `K`, rendered as a 3×3 grid, with the note that it is the optical-frame
matrix and that `P = K · S · [R|t]`.

> _Status (2026-08-23): the distortion row does not say "at the corner". Two
> reasons, both found against the real fixtures. The grid `distortion_field`
> samples is cell *centres*, so the corner is never one of the nodes; and the
> maximum is not guaranteed to be at a corner anyway — a mustache polynomial or
> a thin-prism term can put it elsewhere._
>
> _Status (2026-08-23, phase 5): the row now bounds its own domain. Phase 4
> printed the maximum **over the image** with a tooltip explaining that the
> number might be about the frame's corners rather than about the lens, because
> it had no way of saying anything narrower; on `kerry_park` that number was
> **272.7 px**. With `trustworthy_max_theta_deg` it can be precise instead: the
> maximum is taken over the grid nodes inside the model's trustworthy domain
> and the row names the bound — `yes — max 13.0 px inside 84.1°` — with the
> tooltip saying how many of the grid's nodes were excluded and why. A model
> that is trustworthy everywhere keeps the plain `over the image` phrasing,
> since a qualifier that excludes nothing is the same statement one clause
> longer. See § "The trustworthy domain"._

### 4. Projection plot

Two stacked plots sharing an x axis of incidence angle `θ` in degrees, from 0 to
the max off-axis angle plus 5% margin. Hand-painted with `egui::Painter`: the
crate has no plotting dependency, the app already hand-paints its other
diagnostics, and the reference curve, the azimuth band and the annotation
markers are all custom work that a general plotting widget would not shorten.

**Upper plot — the radial map**, `r(θ)` in pixels:

- solid: the model's actual `|project(ray(θ)) − c|`;
- dashed: the family's ideal map (`f·tan θ` or `f·θ`);
- a band between the min and max over 32 azimuths, drawn on **both** plots and
  only when the model is azimuth-dependent (`fx ≠ fy`, or tangential /
  thin-prism terms present). The band is how decentring distortion becomes
  visible at all — a single-azimuth curve hides it completely. The condition is
  *measured*, not looked up: the band is drawn when it is wider than 0.05 px
  somewhere, which is that condition evaluated rather than a second per-model
  table to keep in step with the first.

**Lower plot — the residual**, `Δr(θ) = r − r_ref` in pixels, zero line marked.
This is the one that shows the shape of the distortion, since on the upper plot
a 12-pixel departure from a 700-pixel curve is invisible.

**Markers on both**, as labelled vertical rules: θ at the mid-edges, θ at the
corner, the spline domain end where there is one, and the 90° asymptote for
perspective models. A rule the axis does not reach is not drawn rather than
clamped to the border, where it would read as a fact about the last angle
plotted — which in practice means the 90° asymptote appears only for a lens
wide enough to be getting near it.

**The extrapolated region.** The x axis runs to the frame's own corner angle,
and on a circular fisheye most of that is outside the lens's image circle,
where § "The trustworthy domain" says the model has stopped describing
anything. Drawing that stretch as the same kind of fact as the rest would be
the plot's version of the number phase 4 had to hedge in a tooltip, so
everything past `trustworthy_max_theta_deg` is drawn as extrapolation and says
so three ways at once:

- the region is washed over, bounded by a coloured rule, and labelled
  `extrapolated past 84.1°`;
- the model's curve continues into it **dotted** rather than solid, so a reader
  following the curve is told again at the moment they cross;
- both y axes are scaled to the **trustworthy samples alone**, so the fold does
  not flatten the part of the plot that means something. The dotted curve then
  dives out through the frame, clipped, which is a fair picture of what the
  polynomial is doing.

A caption under the plots states both numbers together — the bound and how far
the frame reaches past it — because that gap is itself the diagnostic.

The axis is deliberately **not** cut short at the bound. How much of a frame
falls outside the lens's modelled domain is exactly what a reader wants to know
about a circular fisheye, and it is only visible if the axis still reaches the
corner. On `kerry_park` that means roughly half the plot's width is shaded,
which is the honest proportion.

**No distortion.** Both plots still draw — the projection curve is a fact about
the camera whether or not it is distorted, and an empty panel would be a worse
answer than a straight line. The residual plot collapses to its zero line (with
a symmetric range, so the zero line lands in the middle rather than on a
border), and a banner across it reads `No distortion — this model is exactly
{a pinhole | an equidistant fisheye | its own reference map}`, which is the
"says undistorted" the request asks for, stated in terms that say *what* it is
rather than only what it is not. The third branch is `EQUIRECTANGULAR`, which
the first draft of this line left out: it is its own reference and is neither
of the other two.

**The key is words, not glyphs.** `solid: model · dashed: ideal r = f·θ` rather
than a `──`/`╌╌`/`▨` sample key: egui's default font has no box-drawing or
geometric-shape coverage, and the first draft rendered `▸` as tofu in the real
viewer.

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
Z-up world, camera looks down −Z with +Y up* — naming
`specs/formats/sfmr-file-format.md`. This is the block most likely to be pasted
into someone else's code, so it says which frame it is in. (Named, not linked:
egui's `hyperlink_to` opens a URL, and a repo-relative spec path is not one.)

**The node transform.** A reconstruction node can carry a similarity transform
from an in-GUI `Align to…` ([scene-graph.md](scene-graph.md) § "Node
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
| Rig | `rigs[rig_index].name` |
| Sensor | `sensor_names[s − sensor_offset]`, index `s`, and `(reference sensor)` when this image's sensor is it |
| Frame | `image_frame_indexes[i]`, and the number of images in that frame |
| `sensor_from_rig` | rotation + translation, or `identity (reference sensor)` |

This is the part of "extrinsics" that a rig dataset actually needs and that
nothing in the viewer surfaces today.

> _Status (2026-08-23): two corrections against the real arrays, both in the
> table above. `image_sensor_indexes` is a **global** sensor index while
> `sensor_names` is per rig, so the name is at `s − sensor_offset` of the rig
> whose sensor span contains `s` — the same arithmetic `sfm inspect`'s rig
> section does (`analyze/summary.py`). And the `(reference sensor)` marker rides
> the **Sensor** row, not the Rig row: it is a statement about which sensor this
> image came from, and `kerry_park · (reference sensor)` on a row that names the
> rig reads as a claim about the rig. The `sensor_from_rig` row says
> `identity (reference sensor)` only when the stored quaternion and translation
> really are the identity — a file storing something else for its reference
> sensor gets the numbers, since claiming an identity that is not there would
> hide exactly the corruption worth seeing. The section heading is "Rig and
> frame" so that it and the row beneath it are not the same word twice._

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
  in the "sibling" colour — a violet, `scene::SIBLING_HIGHLIGHT_RGB`, declared
  once and shared with the browser — reusing the per-frustum colour channel the
  selected-track highlight already occupies (`camera-views.md` § "Selection
  highlighting"). Like every other highlight it is written at full alpha, which
  is what keeps a node tint from dragging it toward the tint colour.
  The selected image itself keeps its own stronger highlight.
  The two highlights can coexist because they are ranked, not mixed: selected
  image > selected-track member > selected-camera sibling.
- **Image Browser** — the same set gets a thin border in the same colour.
- **Scene Graph** — the camera row is highlighted, and (already, by the
  invariant) so is the selected image's row.
- **Viewport HUD** — unchanged. The HUD owns 3D *display controls*
  ([viewport-hud.md](viewport-hud.md)); a selection is not one.

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

> _Note (2026-08-23), from phase 4 reading real fixtures and corrected by phase
> 5: past `FISHEYE_BLEND_START_RAD` the polynomial fisheye models'
> `undistort_to_ray` blends toward the identity ray rather than inverting an
> unreliable polynomial (`blend_fisheye_ray`, and the caveat § "Testing"
> already names). That threshold is **90° of distorted radius**, not 80° and
> not an incidence angle — phase 4 wrote "past 80° off-axis" from a stale doc
> comment on `blend_fisheye_ray` that contradicted the constant beside it, and
> both have been corrected. Where it lands in incidence angle is per-camera and
> is what `trustworthy_max_theta_deg` reports: **84.1°** on `kerry_park`'s
> camera 0._
>
> _Inside that angle `kerry_park`'s round trip is exact; outside it the round
> trip drifts, and the forward map itself turns over and folds — at θ = 132.7°
> it puts the ray 8.8 px from the principal point where the equidistant ideal
> puts it at 299 px, and past 132.7° it refuses the ray altogether. Every
> reading taken at the **corners** of a circular fisheye's image rectangle is
> therefore in that regime — `max_off_axis` (150.5° on `kerry_park`),
> `diagonal` (301.0°), and the distortion field's unfiltered maximum (272.7 px
> against 13.0 px inside the bound). They are the honest output of the
> definitions this spec sets out, and they describe the black corners outside
> the lens circle rather than the lens. The `horizontal` / `vertical` pair,
> swept through the mid-edges, is the reading that answers "is this fisheye
> really 180°?" for such a camera: 212.9° on `kerry_park`, against `f = 129.15`
> px/rad over a 480-pixel frame._

---

## Performance and caching

Nothing here is per-frame work if it is cached correctly, and all of it is
expensive enough to matter if it is not: `undistort` is iterative for the OpenCV
family, and the arrow field is a few hundred round trips.

| Product | Cost | Cached on | Invalidated by |
|---------|------|-----------|----------------|
| Distortion field | `cols × rows` ray round trips | `(CameraRef, cols, rows)` | camera change, grid density change, node reload |
| Radial profile | `samples × 32 azimuths` | `CameraRef` — the sample count is a constant, so it is not part of the key | camera change, node reload |
| FOV / derived rows | 4 corners + 4 edges | `CameraRef` | camera change, node reload |
| Trustworthy bound | a ~360-step sweep × 8 azimuths, refined | `CameraRef` | camera change, node reload |
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

**`sfmtool-core`, `camera::report` unit tests** (no GUI, runs everywhere) —
implemented in `camera/report/tests.rs`:
- `field_of_view` on a known pinhole matches the closed form
  `2·atan(w / 2fx)`; on `EquidistantFisheye` with `f = w/π` gives 180°
  edge-to-edge; at `f = w/2π` gives 183.35° rather than the folded 176.65°,
  and on a full panorama 360° × 180°.
- `radial_profile`'s reference curve equals the actual curve, to `1e-12`, for
  every model whose `has_distortion()` is false — one case per registry variant,
  asserted complete against `MODEL_COUNT` the way the registry's own corpus is.
- `distortion_field` is identically zero for those same models, and non-zero
  and radially symmetric for `SimpleRadial` with `k1 < 0`.
- `displacement_at` reproduces `distortion_field` exactly at every node of a
  grid — the two are one arithmetic, since the field is defined in terms of it —
  and is zero for every model that is its own ideal map.
- Round trip: `ray_to_pixel(pixel_to_ray(u)) ≈ u` at every grid node used by the
  field, which is what makes the arrows meaningful at all.
- `equiv_focal_length_35mm` returns `None` for every fisheye and equirectangular
  model.
- `parameter_names()` is a permutation of the keys `SfmrCamera::from` writes,
  for every variant — the property that keeps the table from silently dropping a
  parameter when a model gains one.
- `trustworthy_max_theta_deg` is `None` for every model in the undistorted
  corpus (the same `MODEL_COUNT`-complete corpus), and `Some` for exactly the
  four polynomial fisheye models in the distorted one — the classification
  written down twice, once as the function's exhaustive `match` and once as a
  list of model names, and compared. The bound itself is checked two ways: on a
  monotone lens the distorted radius at the reported angle is
  `FISHEYE_BLEND_START_RAD` and the round trip is exact just inside it and
  wrong outside; on a `k1 < 0` lens that folds first it is the closed-form peak
  `θ = 1/√(3|k1|)`.
- `off_axis_angle_deg` at the four corners is `FieldOfView::max_off_axis`, is
  zero at the principal point rather than at the image centre, and matches
  `atan(r/f)` on a pinhole; every `DistortionSample::theta_deg` equals it at
  that sample's grid node.

No model is exempt from either property. `THIN_PRISM_FISHEYE` and
`RAD_TAN_THIN_PRISM_FISHEYE` were, for a while: `CameraModel::distort_ray`
handed the equidistant `(θ·dx, θ·dy)` to a kernel whose input is the
*perspective* `(tan θ·dx, …)` and which converted again, so the forward map
came out off by an `atan` and zero coefficients displaced a grid node by 135 px
on a 640×480 fixture. Fixed by giving each kernel a theta-space core that both
entry points call with what they actually hold; the exclusions and the
regression test that pinned them are gone.

Two caveats remain, both named in the tests rather than left implicit:

- The round trip holds for the polynomial fisheye models only inside
  `trustworthy_max_theta_deg`, because `undistort_to_ray` blends them toward
  the identity ray past `FISHEYE_BLEND_START_RAD` (`blend_fisheye_ray`) rather
  than inverting an unreliable polynomial. That is a deliberate, documented
  approximation, so the round-trip test uses a narrower fisheye fixture instead
  of asserting something the code does not claim.
- The zero residual is to `1e-12` px, not bit for bit: the fisheye kernels do
  not all group the `θ · direction · f` product the same way, so no single
  spelling of the ideal map can match every one of them exactly.

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
- The camera rows raise no cross-panel hover — the two hover fields stay `None`
  with the pointer resting on one — and the group row draws no eye (counted as
  glyphs painted, so an eye added by accident is caught rather than merely
  unasserted).
- The sibling set is the images sharing the selected camera, empty for a camera
  in another node, and empty when *every* image in the node uses it; and on the
  renderer side `frustum_colors` resolves an image that is selected *and* in
  the track *and* a sibling to the strongest of the three, with all three
  written at the full alpha the tint exempts.
- The Camera Intrinsics panel renders its empty state with no selection, its
  populated state with one, and the extrinsics block only when an image is
  selected; the `P` row is absent for a fisheye fixture, replaced by the
  statement.
  Two of its assertions are numeric rather than textual, because the two places
  a sign error produces a plausible-looking matrix are exactly there: `P` is
  checked by projecting a real point through it and comparing against the
  camera's own `ray_to_pixel` (and against `K · [R|t]`, which must *not* match),
  and the transformed pose against `Se3Transform::apply_to_camera_pose` and the
  transform's own action on the camera centre.
- The projection plot, in two places. What it *says* comes off the frame's
  galleys like everything else above: both axis titles, the ideal map named by
  family (`f·tan θ` / `f·θ` / itself), the edge and corner rules carrying the
  same angles the derived table does, the extrapolation label and caption on a
  bounded model and neither on an unbounded one, the band's legend only when
  there is a band, and the no-distortion banner naming each of its three
  families. What it *decides* is tested without a frame, in
  `projection_plot/tests.rs`: which rules it drew (one `edge` on a square
  frame, `h edge` / `v edge` on a portrait one, the 90° asymptote only when
  the axis reaches it, a spline model's domain end in the axis's own units),
  that the radial scale bounded to the trustworthy samples is materially
  smaller than the unbounded one, that the residual's range always contains
  zero and is never zero-width, and that a tick at zero never renders as `-0`.
- The intrinsics layer composes, over **every** `OverlayMode` rather than over
  `Features` alone: a whole `ImageDetail::show` frame is run twice per mode, and
  every shape the mode painted on its own is still there, unmoved, with the
  layer on underneath it — while the frame as a whole has gained shapes.
  Two things that test found out the hard way, both worth knowing before writing
  another like it: `SfmrReconstruction::demo` relaxes its points from a random
  start, so both frames must come from **one** reconstruction or the heatmap
  modes' value ranges differ for reasons having nothing to do with the layer;
  and a `Shape::Text`'s debug form carries its glyphs' atlas UVs, which the
  layer moves by rasterizing `°`, `·` and `×` before the colorbar's labels reach
  the atlas — so text is compared by position and string, not by debug form.
- `I` over the panel toggles the layer, is a toggle rather than a latch, and
  does nothing with the pointer outside the panel. Driven through the same whole
  frame, which needs a warm-up pass first: egui resolves hover against the
  previous pass's widget rects, so on a fresh context's first frame nothing is
  hovered and neither `I` nor the panel's existing `Z` would fire.
- The same frame with the dock's before/after snapshot around it records
  `Intrinsics off` as `User`, and a frame that changed nothing records nothing —
  the differ decides, not the widget
  ([mcp-server.md](mcp-server.md) § "`get_image_detail_display` /
  `set_image_detail_display`").
- The layer's settings popup shows the `No distortion` line instead of the
  distortion row for a pinhole fixture, and its footer names the domain its
  maximum was taken over for a bounded model and not for an unbounded one.
- The composed tooltip, all four ways: the feature line alone with the layer
  off — which must be byte for byte the tooltip the panel produces today, the
  regression a composed tooltip most plausibly breaks — the readout below it
  with the layer on, the readout alone off a feature, and nothing at all with
  neither.
- What the layer *decides*, without a frame, since no painted string shows any
  of it: the tick ladder coarsens with the camera and refines with the zoom; a
  radial model's axes stay straight while its ticks bunch; the axis sweep stays
  monotone in radius across a fold; ticks and rings stay inside the trustworthy
  bound while the axes carry a flagged tail past it; the auto scale fitted to
  the lens exceeds the one fitted to the fold, holds still under zoom, and never
  lets an arrow outgrow its cell; and the offset label appears only off centre.
- Every glyph the layer writes — `Δ`, `−`, `·`, `°`, `×`, `⚙` — is in egui's
  bundled fonts, each spelled once as a named constant. `scene_graph`'s own
  glyph test, for the same reason.
- The node-transform toggle appears only when the node transform is
  non-identity.

**Fixtures.** `kerry_park` is the rig case (2 fisheyes, 24 frames) and
`seoul_bull_sculpture` the ordinary single-camera one; the model-family coverage
comes from synthetic `CameraIntrinsics` values in the core tests rather than
from datasets, since no checked-in dataset exercises thin-prism or
equirectangular.

No new windowed (`ui_basic`) tests: nothing here depends on real OS input the
way the context menu does. One existing windowed assertion moves — the HUD
Layers checkbox is matched by name, and phase 3 renames it to
`Camera Images`.

---

## Implementation phases

Each phase leaves the viewer in a shippable state.

1. **Vocabulary and data model** — *done.* `CameraRef`, `selected_camera`,
   the two `AppState` setters and the coupling truth table with its tests; the
   `Cameras` → `Camera Images` rename with `show_cameras` →
   `show_camera_images`; `AlignSource::Cameras` → `Camera Poses`; the
   reconstruction row's counts. No new UI surface — this phase is only visible
   as better labels.

   Of the "three spec docs that name it", only
   [scene-graph.md](scene-graph.md) actually did.
   [camera-views.md](camera-views.md) and
   [viewport-hud.md](viewport-hud.md) name `AppState`'s *global*
   `show_camera_images` layer switch, which already carried the new name.
   What those two left behind was one label: the HUD's Layers checkbox read
   `Cameras` (`viewer_3d/hud.rs`, asserted by name in the windowed `ui_basic`
   tests) where the HUD spec's own section table already called it
   `Show Camera Images`. Renaming it is the same bug fix; it moved a windowed
   test, so it was deferred to phase 3, which did it.
2. **`camera::report` and `parameter_names()`** — *done.* Pure core work, fully
   unit-tested, with no consumer yet. `CameraModel::parameter_names()` is
   generated by the registry macro for the thirteen fixed-arity models and
   intercepted for the two spline models, as every other registry-derived
   accessor is; `camera::report` is `crates/sfmtool-core/src/camera/report.rs`.

   Three things this phase settled against the real code, all recorded above:
   the fisheye ideal map keeps two focal lengths, the field-of-view spans are
   swept through the image centre rather than measured end to end, and the two
   thin-prism fisheye models have a pre-existing forward/inverse disagreement
   that keeps them out of two of the properties.

   Wiring the Python `_CAMERA_PARAM_NAMES` table to this accessor stays out of
   scope (§ "Deliberately out of scope"), and the two are **not** in sync
   today: the Python table has no entry for `EQUIRECTANGULAR`,
   `EQUIDISTANT_FISHEYE`, `SFMTOOL_FISHEYE` or `SFMTOOL_PINHOLE`, so
   `sfm inspect` falls back to the map's lexicographic order for those four —
   including the `bspline_c10`-before-`bspline_c2` ordering this accessor
   exists to avoid. "The GUI and `sfm inspect` agree glyph for glyph" is
   therefore a goal of that follow-up, not a fact today.
3. **Scene Graph group** — *done.* The Camera Intrinsics rows, click and
   double-click, the cross-panel sibling highlight. It also cleared the two
   residues phase 1 left: the HUD's Layers checkbox now reads
   `Camera Images` (with its windowed `ui_basic` assertion), and the per-image
   row ids and helpers are named for images (`image_list`, `image_{index}`,
   `LIST_MAX_HEIGHT`) rather than for cameras — a vocabulary this phase
   collides with head-on, since the tree now has real camera rows.

   Three things this phase settled against the real code:

   - The registry had **no beta flag** to read: "flagged beta in the registry"
     described two doc comments and a paragraph in
     [sfmtool-camera-models.md](../formats/sfmtool-camera-models.md), nothing a
     panel could query. `CameraModel::beta_note()` is that flag, one note
     shared by the two spline models, with the corpus test asserting no other
     model acquires it.
   - The tooltip and the phase-4 table need parameter **values** in declaration
     order, and phase 2 built only the names. `CameraIntrinsics::parameters()`
     pairs `parameter_names()` with what `SfmrCamera::from` writes, so the two
     cannot disagree about which parameters a model has.
   - Double-click frames camera *centres*; see the row's own bullet above.

   The sibling highlight's colour is `scene::SIBLING_HIGHLIGHT_RGB`, declared
   once because the frustum colour buffer and the Image Browser's border both
   draw it, and the ranking is enforced in `frustum_colors` (weakest written
   first, each stronger one overwriting) and mirrored by the browser's
   `!selected && !in_track` guard.
4. **Camera Intrinsics panel** — *done.* Header, parameters, derived table, `K`,
   extrinsics, rig block, copy menu. No plot yet — the tables alone already
   replace the `sfm inspect` round trip. `crates/sfm-explorer/src/intrinsics_detail/`,
   split the way `point_track_detail/` is: `mod.rs` owns the state and the one
   frame, with `derived`, `header`, `parameters`, `extrinsics` and `format`
   under it.

   Four things this phase settled against the real code, all recorded above:

   - The `(beta)` tooltip *is* reachable here, unlike the tree's `β` — see
     § "1. Header".
   - The derived table's distortion row reads "over the image", because the
     field's maximum on a real circular fisheye is a statement about the black
     corners outside the lens circle — see § "3. Derived".
   - The rig block indexes `sensor_names` by `s − sensor_offset` and puts the
     reference marker on the sensor row — see § "5. Extrinsics".
   - `sfmr_format::{RigFrameData, RigsMetadata, RigDefinition, FramesMetadata}`
     are now re-exported from `sfmtool-core`, beside `THUMBNAIL_SIZE` and for
     the same reason: `SfmrReconstruction::rig_frame_data` is a public field
     whose type nothing downstream could name, so reading — or building — a rig
     meant depending on `sfmr-format` directly.
5. **Projection plot** — *done.* Both stacked plots, reference curve, azimuth
   band, markers, the no-distortion banner, in
   `crates/sfm-explorer/src/intrinsics_detail/projection_plot.rs` with its own
   `projection_plot/tests.rs` for the parts a painted string cannot show.

   It came with a core change it could not be honest without:
   `camera::report::trustworthy_max_theta_deg` and
   `DistortionSample::theta_deg`, § "The trustworthy domain". Phase 4 found the
   symptom — 272.7 px of "distortion" on a lens that displaces 13 px — and
   hedged it in a tooltip; the plot could not hedge, because its x axis runs to
   `max_off_axis` and would otherwise have drawn `kerry_park`'s fold as though
   it were the lens.

   Five things this phase settled against the real code:

   - `blend_fisheye_ray`'s doc comment said it blends "over 80°–90° of `r_d`"
     while the constants beside it said 90° to 100°. Phase 4's note here
     inherited the 80°. Both corrected.
   - That threshold is on the **distorted** radius, not on the incidence angle.
     The two coincide only for a zero-coefficient model, and converting one to
     the other is per-camera — which is what the new function does.
   - The no-distortion banner needed a third branch for `EQUIRECTANGULAR`; the
     spec's two-way brace had no place to put it.
   - egui's default font has no box-drawing or geometric-shape glyphs, so a
     `──`/`╌╌`/`▨` key and an `extrapolated ▸` label render as tofu. Words
     instead. (Found by running the viewer on `kerry_park` and looking at it.)
   - The band's condition — "azimuth-dependent" — is measured off the sampled
     envelope rather than read from a per-model table, so it cannot drift from
     what the projection actually does.
6. **Image Detail overlay layer** — *done.* The checkbox, the settings popup and
   the compositing rules first, with only the principal point and centre offset
   drawn — that is the smallest thing that proves the layer composes correctly
   over every existing mode. Then axes and ticks, then the distortion field,
   then the composed hover readout; four commits in that order, in
   `crates/sfm-explorer/src/image_detail/intrinsics/` with `controls`, `axes`,
   `field` and `hover` under it.

   Everything it settled against the real code and the real viewer is corrected
   in place above rather than listed here; the short version is eight findings:

   - The axes **do not bend** under a purely radial model, which is most of the
     registry and both fixtures. The ticks bunch instead.
   - The axis sweep has to stop at a **fold**, or a circular fisheye's axis
     doubles back and scrambles its own tick order.
   - Ticks and rings are numeric claims, so they **stop at the trustworthy
     bound**; the axes carry on past it dashed.
   - The arrow field's auto scale and legend must be **fitted to the
     trustworthy half of the grid**, and the extrapolated half becomes marked
     nodes rather than arrows.
   - The 8-panel-pixel floor is evaluated at the panel's **fit** scale, which is
     how "at least 8 panel pixels" and "per camera, not per frame" are both
     satisfied.
   - The tick ladder is the **finest** step that clears the spacing, not the
     coarsest.
   - The principal point is a **reticle** with an open middle, not a cross
     through a ring, which haloed comes out as a blob.
   - `camera::report` had **no way to ask for the displacement at a pixel**;
     `displacement_at` is now public and `distortion_field` is defined in terms
     of it.

   Plus two smaller ones: a ring's label goes on the diagonal rather than at the
   top, where the vertical axis's own tick for the same angle already is; and an
   arrowhead needs three panel pixels, not one, before it reads as an arrow.

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
