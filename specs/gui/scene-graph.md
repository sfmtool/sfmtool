# Scene Graph and Multi-Reconstruction Support

The viewer holds more than one reconstruction at a time. Several `.sfmr` files
load together as nodes of a scene graph, and a **Scene Graph panel** — a tree
view — is where they are browsed, toggled, transformed and selected. The point
is comparison: two or more reconstructions of the same scene sharing one 3D
space, each with its own transform, tint and visibility, so a reviewer can see
where they agree and where they do not.

Related specs: [architecture.md](architecture.md) (rendering pipeline
this extends), [cross-panel-hover.md](cross-panel-hover.md) (selection
and hover model this generalizes),
[camera-views.md](camera-views.md) (pick buffer),
[viewport-hud.md](viewport-hud.md) (global display controls).

---

## Motivation

Today the viewer holds exactly one reconstruction
(`AppState.reconstruction: Option<SfmrReconstruction>`, `state.rs`), and that
assumption runs through every layer:

- Selection and hover are bare `usize` indices into "the" reconstruction.
- The pick buffer encodes `8-bit entity tag | 24-bit index` with no
  reconstruction identity and a 16.7M index ceiling.
- `SceneRenderer` owns one point instance buffer, one frustum edge buffer, one
  thumbnail atlas, and single-valued derived scalars (`auto_point_size`,
  `scene_center`, `scene_radius`).
- A long tail of caches key by bare image index: `AppState::sift_cache` /
  `full_res_cache`, `ImageBrowser::thumbnail_cache` (invalidated only by image
  *count*), `ImageDetail::loaded_image`, `PointTrackDetail`'s texture maps, and
  `SceneRenderer::bg_image_loaded_index`.
- `File > Open` replaces the loaded file wholesale.

Comparing solver runs, incremental vs. global solves, or pre/post-transform
outputs requires several reconstructions in one viewport at once — with a way
to see what is loaded, hide and show each one, and tell entities apart. That
structure is a scene graph, and it needs a panel of its own.

**Scale requirement:** the design must hold at the existing performance targets
*per reconstruction* — millions of points and tens of thousands of cameras —
multiplied by a modest number of simultaneously loaded files (design point:
2–8 typical, tens as the upper bound).

---

## Concepts and Data Model

### The scene graph

The scene is a two-level tree (with fixed group nodes one level below each
reconstruction):

```
Scene (root, implicit)
├── Reconstruction "run_a"        ← one node per loaded .sfmr
│   ├── Camera Intrinsics (2)     ← fixed group node, no eye
│   │   └── #0  OPENCV_FISHEYE  480×480  f 240.1  26 images
│   ├── Camera Images (243)       ← fixed group node
│   │   ├── IMG_0001.jpg          ← per-image rows (virtualized)
│   │   └── …
│   ├── Points (1,204,551 · 12 at ∞)  ← fixed group node
│   │   ├── ▸ selected: pt3d_a1b2c3_88231   ← selection/hover rows only
│   │   └── ▸ hovered:  pt3d_a1b2c3_10442
│   └── Patches                   ← toggle-only row, present when the recon
│                                    carries patch data; not expandable
├── Reconstruction "run_b"
│   └── …
```

This is deliberately *not* a general transform hierarchy — there is no
user-created nesting, grouping, or reparenting. Each reconstruction is a node;
its cameras and points are its children. A deeper hierarchy (rig groups, image
sequences, user folders) can be added later without changing the model below.

### Identity

```rust
/// Identity of one loaded reconstruction. Monotonically assigned per session,
/// never reused — so stale cache entries can never alias a new load.
pub struct ReconId(u32);

/// A camera/image within a specific reconstruction.
pub struct ImageRef { pub recon: ReconId, pub image: u32 }

/// A 3D point within a specific reconstruction.
pub struct PointRef { pub recon: ReconId, pub point: u32 }

/// A camera intrinsics record within a specific reconstruction.
pub struct CameraRef { pub recon: ReconId, pub camera: u32 }
```

All selection and hover state moves from `Option<usize>` to these typed refs:

```rust
pub selected_image:  Option<ImageRef>,
pub selected_camera: Option<CameraRef>,
pub selected_point:  Option<PointRef>,
pub hovered_image:   Option<ImageRef>,
pub hovered_point:   Option<PointRef>,
```

`selected_camera` is coupled to `selected_image` rather than independent of it:
selecting an image selects the camera it was taken through, and both fields are
written only through `AppState::select_image` and `AppState::select_camera`, so
no panel can leave one of them naming another image's lens. The rule is stated
exhaustively in [camera-intrinsics.md](camera-intrinsics.md) § "The
selection coupling", which owns it. Everything the recon-scoping rules below say
about the image and point selections holds for the camera selection too — it is
filtered by `ReconId` in exactly the same places.

There is still exactly **one** selected image and **one** selected point
globally — the cross-panel selection model of
[cross-panel-hover.md](cross-panel-hover.md) is unchanged; only the
key type widens.

### The node

```rust
pub struct SceneNode {
    pub id: ReconId,
    /// Display label: file stem, disambiguated with " (2)", " (3)"… on
    /// collision. "demo" for demo data.
    pub label: String,
    /// Source path; None for demo data.
    pub path: Option<PathBuf>,
    pub recon: SfmrReconstruction,

    // Per-node display state
    pub visible: bool,               // master eye for the whole node
    /// Whether pointer interaction (hover + click pick) reaches this node in
    /// the 3D viewport. Off = the node is display-only: overlay it as a
    /// reference without it stealing hovers and clicks from the node you are
    /// working with. Explicit selection from the Scene panel still works.
    pub interactive: bool,
    pub show_points: bool,           // group eyes
    pub show_camera_images: bool,
    pub show_patches: bool,
    pub show_points_at_infinity: bool,
    pub tint: NodeTint,              // Original | Tint(&'static TintColor)
    /// Similarity transform (uniform scale · rotation · translation) mapping
    /// this node's native coordinates into the shared world space. Identity on
    /// load. Set by the "Align to…" operation; see "Node Transforms and
    /// Alignment" below.
    pub transform: Se3Transform,

    /// This node's data needs (re-)upload to the GPU.
    pub needs_upload: bool,
}
```

The renderer does not read `SceneNode` directly: each frame `app.rs` mirrors the
five display flags plus `interactive` and `tint` onto the node's GPU bundle as a
`NodeDisplay`, and the `transform` alongside them; the draw loop and per-recon
uniform write consult only the bundle. `transform` and `tint` are also carried
across a `Reload from Disk`, alongside the display flags — a refreshed file
should come back where the user put it, in the color they were telling it apart
by.

`AppState` replaces its single slot with:

```rust
pub scene: Vec<SceneNode>,           // tree order = load order (reorderable later)
pub selected_recon: Option<ReconId>, // see "The selected reconstruction" below
pub solo: Option<ReconId>,           // see "Comparison Affordances" below
```

The per-node `needs_upload` flag replaces the global `points_need_upload`
bool; closing a node additionally enqueues a resource-release for its
`ReconId` (see Rendering).

**Effective visibility** of a layer in node *n* is the AND of four switches:
the global HUD Layers toggle (unchanged, now acting as a master switch across
all nodes), the node's `visible` eye, the scene's solo override (nothing soloed,
or *n* is what is soloed — see "Comparison Affordances"), and the node's
per-group eye. The Grid stays global-only.

The middle two are one question — *is this node drawn at all?* — and are
composed in exactly one place, `scene::is_visible(node, solo)`. `app.rs` folds
it into the `NodeDisplay::visible` it mirrors onto the bundle each frame, so the
draw loop and the bounds union read a single flag; the stats overlay calls the
same function. Nothing else is allowed to spell the rule out again.

**Interactivity** is a separate per-node switch (`interactive`, default on),
following the Blender-outliner eye/cursor pairing. With it off, the node's
entities stop producing pick results in the 3D viewport — no hover highlight,
no hover overlay text, no click selection, no accidental double-click into its
cameras — while remaining fully rendered. The scope is deliberately
*pick-based interaction only*:

- **Explicit selection still works**: clicking the node's rows in the Scene
  panel, and `[` / `]` stepping, can still select the node and its images —
  the tree is the control surface, so a display-only node can always be
  deliberately inspected.
- **Alt+click depth targeting still works** on its geometry: the orbit target
  comes from the depth readback, not the pick ID, so a reference node remains
  navigable "terrain" even when non-interactive.
- Outbound highlights (e.g. a browser-thumbnail hover brightening one of its
  frustums) are unaffected — the toggle gates what the node *captures*, not
  what it displays.

The intended workflow: overlay a reference reconstruction, switch its cursor
off, and work with the primary one without the reference stealing picks.

### The selected reconstruction

Selection gains a third, coarsest level: alongside the selected image and the
selected point there is a **selected reconstruction**
(`selected_recon: Option<ReconId>`). UI that is inherently sequence- or
file-shaped and cannot meaningfully show several reconstructions at once — the
Image Browser strip, animation playback, `,` / `.` camera stepping — follows
it, exactly as Image Detail follows the selected image and Point Track follows
the selected point. This matters most when the loaded reconstructions have
*different image sets*: the strip shows the selected reconstruction's own
sequence, never a merge.

Rules:

- Selecting an image or point anywhere selects its owning reconstruction.
- Clicking a reconstruction row in the Scene panel selects it directly. If the
  current image/point selection belongs to a different node, it is cleared —
  the invariant is that **all finer selection state lives inside the selected
  reconstruction**, so no two panels ever show different files' selections.
  Hover is exempt: it is transient and may touch any visible node.
- The selected reconstruction is marked in the Scene panel (bold label +
  accent bar) and named in the Image Browser's header.
- `[` / `]` step the selected reconstruction back/forward in tree order — the
  reconstruction analogue of `,` / `.` for images (handled alongside them in
  `viewer_3d/input.rs`). When an image is selected — and especially in camera
  view — stepping carries the selection to the **same-named image** in the new
  reconstruction when one exists (selection, and camera view, follow it);
  otherwise the finer selection clears per the invariant above. Flipping
  `[` / `]` while looking through a camera is the core comparison move: same
  photo, two solves. An active **solo** travels with the step (see "Comparison
  Affordances"), which sharpens the same move to one reconstruction on screen at
  a time.
- Closing the selected node falls back to selecting the first remaining node;
  an empty scene means no selection (panels show their existing empty-state
  text).

`compare_track_images` and frustum track-highlighting operate within the
reconstruction that owns the selected/hovered point; tracks never span
reconstructions.

---

## Scene Graph Panel

A fifth dock tab, `Tab::SceneGraph`, title **"Scene"**. Default layout: a new
left split of the root (~18% width), with the existing 3D Viewer / Image
Detail / Point Track / Image Browser arrangement to its right. Like the other
tabs it can be re-docked freely and closed, and **Panels ▸ Scene** brings it
back ([panel-layout.md](panel-layout.md)).

### Tree rows

The tree is drawn with `egui::collapsing_header::CollapsingState` with
explicit IDs (the viewport HUD's pattern, `viewer_3d/hud.rs`) — *not*
`CollapsingHeader` — so expansion state is addressable and testable. Rows are
fixed-height for virtualization.

**Reconstruction row** — `[▸] [👁] [S] [🖱] ▪ run_a 1.2M pts · 243 imgs · 2 cams`
- Expand triangle, visibility eye (node master), solo toggle, interaction cursor
  toggle (`interactive` — greyed when off), tint swatch, label, compact counts.
- All three glyph toggles carry **explicit ids** rather than egui's auto ids: an
  auto id is a count of what was allocated before the widget, so anything added
  to the row ahead of it would move the hover/click state of everything after.
- Bold + accent bar when selected. Click: select this reconstruction.
  Double-click: zoom-to-fit this node.
- Everything past the two toggles is **one click target spanning the row** —
  the name, the gap, the counts — carrying select, zoom-to-fit and the context
  menu alike, under an explicit id rather than an auto-generated one (egui keys
  a popup's open state on the widget id, and an auto id shifts with whatever
  was laid out before it). Its contents are drawn non-interactive on top:
  `Label::selectable(false)` on both texts, because egui's default
  `selectable_labels` gives a bare label `Sense::click_and_drag()` for text
  selection, and a label drawn after the row would win every pointer hit that
  landed on a glyph — leaving the name the one part of the row that answered
  nothing. The accent bar's width is reserved on every row whether it is
  painted or not, so the name does not shift as the selection moves.
- Context menu: `Select`, `Zoom to Fit`, `Align to ▸` (one entry per other
  loaded node — see "Node Transforms and Alignment"), `Reset Transform`,
  `Tint ▸` (Original / palette of distinguishable colors), `Reload from Disk`,
  `Close`. **`Solo` is not in the menu** — it is the row's `S` (see "Comparison
  Affordances").
- Expanded by default: with one file loaded the node's groups are the whole
  panel, and with a handful the tree is still what answers "what is in here".
  Its Camera Images and Points groups start *collapsed* — the image list is the
  Image Browser's job, and an expanded 50K-row list would bury every node below
  it.
- Both eyes and the cursor are drawn as dimmable glyph buttons (U+1F441 EYE,
  U+1F5B1 TRACKBALL, U+221E INFINITY — all in egui's bundled proportional
  fonts, which `scene_graph/tests.rs` pins); solo is the letter `S`, dimmed the
  same way, because no bundled pictograph says "only this one" and a mixer's
  solo button has said `S` for fifty years. The selection accent bar and the
  tint swatch are *painted* rather than written: no bundled proportional glyph
  is a vertical bar, and a painted rect answers no pointer hit — which is what
  keeps it from competing with the row-wide click target.
- The master eye is lit from **effective** visibility, so a node another node's
  solo is hiding reads as dark without its own flag having moved; clicking it
  still writes `visible`, which takes effect the moment the solo ends.
- The compact counts are `1.2M` / `12.3K` / `999`; the exact figure, with
  thousands separators, is one row down on the group rows. Three counts —
  points, images, cameras — make this the longest row in a panel that defaults
  to 18% of the window, so it **elides** rather than truncating or wrapping:
  when the row cannot fit all three the camera count goes first (it is also on
  the Camera Images group row, one line down), then the image count, leaving
  the point count, which has no other home in the tree, last to go. The
  measurement is against the width actually left after the label, so widening
  the panel brings the counts back.

**Camera Images group row** — `[▸] [👁] Camera Images (243)`
- Eye drives `show_camera_images` for the node.
- Named for what it counts: `recon.images.len()`, one row per *posed view*. In
  `.sfmr` — COLMAP's vocabulary — a camera is an intrinsics record that any
  number of images can share, so a group labelled `Cameras` and counting images
  was mislabelled rather than merely terse. The intrinsics themselves get their
  own group above this one; see
  [camera-intrinsics.md](camera-intrinsics.md) § "Terminology" and
  § "Scene Graph: the Camera Intrinsics group", which own that definition.
- Expands to one row per image, **in image order, virtualized** via
  `ScrollArea::show_rows` — only visible rows are laid out, so 50K images
  cost tens of rows per frame.

**Camera image row** — `IMG_0001.jpg`
- Click: `select_image(Some(ImageRef))`, which also selects the camera that
  image was taken through (the coupling above). Double-click: enter/switch
  camera view (same semantics as a browser thumbnail double-click).
- Hover: sets `hovered_image` (participates in cross-panel hover exactly like
  a browser thumbnail; the row highlights when the same image is hovered
  elsewhere, e.g. from the 3D viewport pick).
- Context menu: `Resect Image` and `Resect Image from Matches…`, which
  re-estimate this one image's pose against the rest of its reconstruction and
  show the answer as a derived node beside the original — see
  [resect-image.md](resect-image.md), which owns them.
- Selected row: highlight + auto-scroll into view when the selection changes
  from another panel (scroll-to happens only on selection *change*, so the
  user's manual scrolling isn't fought).

**Points group row** — `[▸] [👁] Points (1,204,551 · 12 at ∞)`
- Eye drives `show_points`; an inline `∞` mini-toggle drives
  `show_points_at_infinity` when the recon has infinity points.
- Expands to **selection and hover rows only**, not a full listing:
  - `selected: pt3d_<hash>_<index>` — the same copyable ID the Point Track
    panel shows; click re-selects (useful after selecting elsewhere), and the
    row doubles as where selection is *visible* in the tree.
  - `hovered: pt3d_<hash>_<index>` — transient, present only while a point of
    this recon is hovered.

  A full per-point listing is deliberately out: millions of rows are not
  navigable, and beyond ~16.7M row-pixels egui's `f32` scroll coordinates lose
  integer precision, so a virtualized list of that size would misbehave
  mechanically as well as ergonomically. If per-point browsing is ever wanted,
  it should arrive as a filtered/query view (e.g. "worst reprojection error",
  "longest tracks"), not a raw list.

**Patches row** — `[👁] Patches` — eye only, shown when the node carries patch
data (mirrors the HUD's greyed-when-absent convention).

### Panel plumbing

Following the existing per-panel response pattern threaded through `dock.rs`:

```rust
pub struct SceneGraphResponse {
    pub select_image: Option<ImageRef>,
    // The two camera-row requests; see camera-intrinsics.md, which owns
    // them.
    pub select_camera: Option<CameraRef>,
    pub zoom_to_camera: Option<CameraRef>,
    pub select_point: Option<PointRef>,
    pub request_camera_view: Option<ImageRef>,
    pub hovered_image: Option<ImageRef>,
    pub hovered_point: Option<PointRef>,
    pub has_pointer: bool,
    pub select_recon: Option<ReconId>,
    pub align_node: Option<(ReconId, ReconId, AlignOptions)>, // source, target
    pub reset_transform: Option<ReconId>,
    pub zoom_to_node: Option<ReconId>,
    pub toggle_solo: Option<ReconId>,
    pub close_node: Option<ReconId>,
    pub reload_node: Option<ReconId>,
}
```

Note what is *not* in the response: the tint. A tint is per-node display state,
like the eyes beside it in the same tree, so `Tint ▸` writes it straight into
the `SceneNode` and the next frame's display mirror carries it to the GPU —
there is nothing for `dock.rs` to arbitrate. Solo is app-level view state, so it
travels through the response like every other request the panel makes.

The reconstruction row's menu is built from `egui::Popup::context_menu` with
`PopupCloseBehavior::CloseOnClickOutside` rather than from
`Response::context_menu`, whose default closes the menu on **any** click inside
it — which would tear the whole thing down the moment the user set one of the
`Align to` radio buttons. Closing is therefore explicit: each item that acts
calls `ui.close()`, and picking an alignment target calls `Popup::close_all`,
since `ui.close()` inside the submenu would leave the row's own menu standing.

For the menu to open at all on Windows, the secondary button has to survive the
trip from the OS. `platform::windows::create_manager` enables
`EnableMouseInPointer` so DirectManipulation can see precision-touchpad
contacts; the side effect is that every mouse button arrives as a `WM_POINTER*`
message, which winit 0.30 renders as a `Touch` event and egui's touch emulation
reads as the **primary** button. Unhandled, nothing in the app is ever
`secondary_clicked`: no context menu can open anywhere, and a right-click on a
tree row selects it exactly as a left-click does.
`platform::windows::restore_mouse_button` therefore rewrites the secondary and
middle contacts back into real `MouseInput` events (preceded by the move that
positions them), reading the button off the pointer flags the window procedure
already decodes. Left clicks keep the touch-emulation path they have always
taken, which the 3D viewport's drag handling is built against.

Nothing above the window can observe any of this — the panel behaves correctly
under `Context::run_ui` — so it is guarded by a windowed test that drives real
synthetic mouse input (`ui_basic.rs`,
`a_real_right_click_opens_the_reconstruction_rows_context_menu`).

`has_pointer` ownership of hover state follows
[cross-panel-hover.md](cross-panel-hover.md) unchanged: when the
pointer is over the Scene panel, it owns both hover fields.

`dock.rs` applies the response coarsest-first — `select_recon`, then
`select_camera`, then `select_image` / `select_point` — so a finer selection
reported in the same frame wins over the coarser click that would otherwise
have cleared it.

The panel additionally records the screen rect of every row and toggle it drew,
keyed by the same explicit ids as the expansion state. A collapsible,
virtualized tree has no geometry an outside caller can predict, so this is how
anything that needs to point *at* a row finds it: the panel tests today (which
click through it rather than guessing pixel offsets), keyboard navigation later.

---

## Rendering: Per-Reconstruction GPU Resources

### Resource bundles

`SceneRenderer`'s per-reconstruction buffers move into a bundle, keyed by
`ReconId`:

```rust
struct ReconResources {
    // points
    point_instance_buffer: wgpu::Buffer,
    point_count: u32,
    // frustums + image quads
    frustum_edge_buffer: wgpu::Buffer,
    frustum_edge_count: u32,
    frustum_color_buffer: wgpu::Buffer,      // per-image ABGR, cheap write path
    image_quad_instance_buffer: wgpu::Buffer,
    distorted_quad_vertex_buffer: wgpu::Buffer,
    distorted_quad_index_buffer: wgpu::Buffer,
    // thumbnails: per-recon atlas + bind group
    thumbnail_texture: wgpu::Texture,
    // patches (optional)
    patch: Option<PatchResources>,
    // per-recon derived scalars (formerly singletons on SceneRenderer)
    auto_point_size: f32,
    camera_nn_scale: f32,
    bounds: (Vector3<f64>, f64),             // center, radius (pre-transform)
    // pick bases, see Picking
    point_pick_base: u32,
    image_pick_base: u32,
}

// SceneRenderer
recons: HashMap<ReconId, ReconResources>,
```

The bundle also owns the **bind groups** built from those resources, and the
uniform buffers whose contents are per-recon (the block below, plus the
thumbnail- and patch-atlas grid blocks) — a bind group naming a node's atlas
and color buffer cannot outlive them, so it belongs to the same lifetime.

Shared and unchanged: pipelines, render targets, samplers, the unit quad
vertex buffer, EDL/target-indicator/track-ray/bg-image resources (all
singletons by nature — track rays and bg image serve *the* selection, which is
single). The atlas samplers are singletons too: every node's atlas is sampled
identically, so only the textures are per-recon.

Loading a node uploads one bundle; closing a node drops one bundle. No other
node's GPU data is touched — this is the reason for per-recon buffers rather
than one concatenated buffer that would need rebuilding on every membership
change.

### Draw loop

Each scene pass (points, frustum wireframes, image quads, patches) becomes a
loop over visible nodes: bind the per-recon uniform slice + per-recon bind
group (thumbnail atlas, patch atlas), then the existing instanced draw. At the
design-point scale this is a handful of draw calls per pass — tens of draws
total per frame, negligible.

Pass order, blending, and depth behavior are unchanged; all nodes share the
one depth buffer, so mutual occlusion between reconstructions is automatic.

### Per-recon uniforms

A new per-recon uniform block:

```
model: mat4x4<f32>        // node.transform as a matrix
point_size: f32           // per-recon auto size × global 2^point_size_log2
point_pick_base: u32
image_pick_base: u32
pickable: u32             // 0 → emit PICK_TAG_NONE (node.interactive off)
tint_color: vec4<f32>     // rgb = tint, a = strength; a = 0 → original colors
show_infinity: f32        // global HUD ∞ toggle AND node.show_points_at_infinity
```

`show_infinity` is per-recon rather than global (where it started) because the
∞ mini-toggle is per node, and points at infinity are not a separate draw — they
ride in the same instance buffer, culled in the vertex shader so instance
indices, and therefore pick ids, stay unfiltered. The four whole-layer toggles
*are* separate draws, so effective visibility for them is resolved on the CPU by
skipping that node in that pass.

**Binding mechanism (implemented): one small uniform buffer per node, bound in
a per-recon bind group** — not one buffer sliced by dynamic offsets. Three of
the four scene pipelines need a per-recon bind group regardless (frustum
colors, thumbnail atlas, patch atlas), so the bundle owns bind groups either
way; a dynamic-offset buffer would have added a second mechanism, plus
256-byte alignment padding, for nothing. The block is appended as an extra
binding on each pipeline's existing group 0.

The atlas-grid uniform blocks move into the bundle for the same reason:
`ImageQuadUniforms` and `PatchUniforms` carry per-recon grid dimensions, so
each node allocates its own and the per-frame write loops over nodes. What
stays global is what is genuinely one per frame: the camera/selection block,
EDL, target indicator, track rays, background image.

Vertex shaders apply `model` before `view_proj`. Homogeneous points at
infinity need no special-casing: a direction transforms as
`(model × vec4(dir, 0)).xyz` — the linear part rotates it, translation drops
out, and uniform scale is irrelevant to a direction.

Point splats are billboarded *after* the model transform, in world space, at the
node's own `point_size` — so the transform's scale is folded into `point_size`
on the CPU (`uniforms.rs`), and a magnified node's points grow with it instead
of shrinking to pinpricks.

**Frustum geometry works the other way.** Frustum stubs and image quads are
built in the node's own coordinates and *then* scaled by `model`, so `app.rs`
divides the global `length_scale` by the node's scale before uploading them.
What reaches the screen is `length_scale` in world units whatever frame the node
was solved in — which is the other half of what makes the shared-frustum-size
compromise below dissolve on alignment.

The existing global uniforms keep working with one change of meaning:
`selected_point_index` / `hovered_point_index` / `hovered_image_index` become
**global pick indices** (base + local, see below), so the shader compare
`instance_index + pick_base == selected_point_index` stays a single u32
comparison. Sentinel remains `0xFFFFFFFF`.

### Derived scalars and framing

- `auto_point_size`, `camera_nn_scale`: computed **per recon** at upload, as
  today's functions already do — they just stop being singletons. Point splat
  world size is per-recon; the HUD size slider is a global multiplier on top.
  The EDL pass is the one consumer that cannot be per-recon (it shades the
  whole frame in a single fullscreen draw); it takes the **maximum**
  `auto_point_size` across loaded nodes, the value that covers the largest
  splats it has to smooth over.
- **Scene bounds** (adaptive clip planes, `Z` zoom-to-fit, supernova/grid
  scaling): the union of the *effectively* visible nodes' bounding spheres
  (eye AND solo, the one flag on the bundle), each transformed by
  its node transform — the smallest sphere enclosing them, not a bounding box.
  Recomputed when the visible set, a node transform, or a node's data changes.
  A node contributes bounds only once its points are uploaded, so an empty
  bundle cannot drag the union toward the origin. With *every* node hidden the
  union falls back to all loaded nodes rather than collapsing to a unit sphere
  at the origin, so switching the last eye off does not fling the camera.
- `length_scale` (drives frustum stub depth, target indicator): re-derived
  from the visible union whenever the node set **or any node transform**
  changes, exactly as it is re-derived on load today; still one global, still
  user-adjustable. The union rule is the **minimum** of each node's own seed
  (`min(10 × auto_point_size, camera_nn_scale)`, times that node's transform
  scale, since both inputs are measured in the node's own units) — the finest
  scale present,
  so the smallest reconstruction's frustums stay legible. This is a known
  compromise — two reconstructions at wildly different scales will share one
  frustum size until they're aligned (see "Node Transforms and Alignment").
- Per-node `Zoom to Fit` frames that node's transformed bounds.

### Track rays and background image

Track rays are CPU-built per selection from the owning recon; the builder
additionally applies `node.transform` to camera centers and ray points (a
similarity is affine, so mapping the two endpoints maps the whole segment). They
are rebuilt when the selection changes *or* when a transform does.
The background image (camera view mode) is keyed by `ImageRef` instead of
bare index, fixing the latent "same index, wrong reconstruction" aliasing.

---

## Picking

The current encoding — `8-bit tag | 24-bit index` — has no reconstruction
field and caps points at 16.7M. New scheme, still a single `R32Uint` target:

```
bits 31..30  tag: 0 = none, 1 = frustum/camera, 2 = point   (3 reserved)
bits 29..0   global index: recon pick base + local index    (2^30 ≈ 1.07B)
```

- Each `ReconResources` is assigned contiguous ranges `point_pick_base` and
  `image_pick_base` in two global index spaces (points and images allocated
  independently). Shaders emit `tag << 30 | (base + instance_index)` — the
  base arrives via the per-recon uniform block, so **instance buffers store
  nothing new** and never need rewriting when bases move.
- Bases are (re)assigned whenever a node is added or removed, **or its entity
  counts change** (an upload of new data re-cuts the space) — in `ReconId`
  order, which is also load order. Nothing is rewritten but the tables; the
  bases reach the GPU on the next frame's per-recon uniform write, which
  happens unconditionally anyway. Reassignment therefore costs one walk over
  the nodes and no buffer traffic at all.
- CPU decode: binary search a sorted `(base, ReconId)` table per entity kind →
  `(ReconId, local index)` → `ImageRef` / `PointRef`. All decode sites
  (`app.rs` pick dispatch, hover overlay text) go through one helper; the
  readback returns an already-decoded ref rather than a raw id. An index
  outside every assigned range decodes to "nothing", which is what makes a
  readback from a node released one frame earlier harmless.
- The two index spaces are allocated independently, so exceeding 2^30 in
  either is possible in principle; it is logged rather than clamped, being
  three orders of magnitude past the design point.
- `pick_id == 0` remains "nothing"; tag 1 with base 0, local 0 encodes as
  `1 << 30`, so there is no collision with the none value.
- **Non-interactive nodes** (`interactive` off): the per-recon `pickable`
  uniform makes the node's shaders emit `PICK_TAG_NONE` instead of a pick ID.
  Depth and color output are unchanged, so the node still renders and still
  occludes — and where it occludes an interactive node, the pick reads as
  background rather than passing through to hidden geometry (consistent with
  what the user sees). The depth readback is untouched, which is what keeps
  Alt+click orbit targeting working on non-interactive geometry.

Total capacity: ~1.07B points and ~1.07B images summed across loaded
reconstructions — comfortably beyond the design point (tens of files × 10M+
points).

The hover overlay gains recon context when more than one file is loaded:
`Camera: run_a / IMG_0001.jpg`, `Point3D run_a #88231`.

---

## Node Transforms and Alignment

Comparing reconstructions usually means putting them into one frame first.
Every node therefore carries a similarity transform, and the Scene panel
exposes an **Align to…** operation that computes it.

### The transform

`SceneNode::transform` is an `Se3Transform`
(`sfmtool-core::geometry::se3_transform`, applied as
`p' = scale · (R · p) + t`) — the same type the core alignment estimator
returns. It maps the node's native coordinates into the shared world space,
and is identity on load.

Where it applies:

- **GPU**: converted to the per-recon `model` matrix in the uniform block
  above — every pass, every entity of the node.
- **CPU world-space paths** of the owning node: track-ray construction, node
  bounds (union framing and per-node zoom-to-fit), and camera-view entry —
  `enter_camera_view` composes the node transform into the viewport pose so
  "look through this camera" shows the transformed scene from the transformed
  camera. Camera-view entry composes three things: the camera centre through
  the transform, the world-to-camera rotation losing the transform's rotation
  (`q' = q · conj(q_world)`, as `Se3Transform::apply_to_camera_pose` derives
  it), and the median-depth start distance scaled by the transform's scale. The
  same composition serves `,` / `.` camera switching, so flipping between
  cameras of an aligned node stays in its displayed frame.
  Two further CPU consumers the list above does not name explicitly but which
  need it for the same reason: the viewport's **first-show framing** and the
  `Z` **zoom-to-fit** key, both of which frame point positions rather than GPU
  geometry (`scene::world_points`).
- **The camera-view background image**: its mesh is ray directions in the
  node's own coordinates, so it reaches world space through the same `model`
  matrix, written into the BG uniform block from the viewed camera's node (see
  [camera-views.md](camera-views.md), "Background mesh shader"). It is
  the one geometry that would otherwise be left behind by an alignment while
  the viewpoint moved with it.
- **Points at infinity**: directions rotate; translation drops out and uniform
  scale is irrelevant to a direction. The `model × vec4(dir, 0)` path already
  handles this with no special-casing.
- **Picking** is unaffected — pick IDs are index-based, not position-based.

The transform is **view state only**: it never mutates the
`SfmrReconstruction` in memory nor the `.sfmr` on disk. Baking a transform
into a file remains `sfm xform`'s job; a "Save Aligned Copy…" export from the
GUI is a future direction.

### Align to…

Context menu on a reconstruction row: `Align to ▸ <other loaded node>`.
Computes the similarity mapping this node (source) onto the chosen node
(target), then sets `source.transform = target.transform ∘ T_fit` — the fit
lands in the target's *currently displayed* frame, so aligning C→B after B→A
chains as expected. The target node is never modified.

The whole fit lives in
`sfmtool-core::analysis::alignment::reconstructions::align_reconstructions` —
correspondence gathering included — so the GUI adds only the popup and the
status line, and any other caller aligning two loaded reconstructions gets the
same answer. See
[reconstruction-alignment.md](../core/analysis/reconstruction-alignment.md). What that
fit does:

- **Correspondences by cameras** (default): images matched by `name` across
  the two nodes; corresponded camera centers feed the fit. Works whenever the
  reconstructions were solved from overlapping image sets — the typical
  comparison case (two solver runs over the same shoot) — and mirrors
  `sfm align`'s by-cameras mode. Only the shared subset matters; disjoint
  extra images on either side are simply not correspondences. A repeated name
  on either side pairs once, first occurrence winning, as `sfm align` does.
  Fewer than 3 shared images is refused rather than fitted: two points leave
  the rotation about the line joining them free.
  Note the deviation from `sfm align`'s by-cameras mode, which averages
  per-image *orientations* (`estimate_similarity_with_orientations`) as well as
  positions. Here the fit is over camera **centres** only, per this spec — the
  same `estimate_alignment` the point mode uses, so the two modes differ in
  what they correspond and in nothing else.
- **Correspondences by points**: `find_point_correspondences` matches 3D
  points through shared feature observations in shared images, yielding a much
  larger set; points at infinity are excluded from the fit (their stored
  position is a bearing, not a location — the same `_finite_pair_mask` rule
  `sfm align` applies). Requires feature-indexed observations (`sift_files`
  source) in both nodes — the menu option is disabled, with a hover
  explanation, otherwise. Fewer than 10 correspondences, before or after
  RANSAC, is refused (`sfm align`'s `min_points`).
- **Estimation**: `ransac_alignment` for inlier selection on point
  correspondences, then `estimate_alignment` with trimmed refit
  (`AlignmentParams { rounds: 3, keep_fraction: 0.8, estimate_scale }`);
  camera-mode fits, with far fewer correspondences, use trimming alone. A
  Rigid / Similarity choice maps to `estimate_scale`. The RANSAC threshold is
  the 95th percentile of a preliminary all-correspondence fit's residuals, as
  in `sfm align` — floored at 1e-9 × the target cloud's extent so an exactly
  corresponding pair (a synthetic fixture, or a file aligned to a copy of
  itself) is not rejected as all-outlier by `f64` rounding noise. 200 RANSAC
  rounds rather than `sfm align`'s 1000: the loop is scalar Rust on the UI
  thread rather than numpy, and the preliminary fit has already done most of
  the work. Full detail in
  [reconstruction-alignment.md](../core/analysis/reconstruction-alignment.md).

Options are deliberately few (a small popup): correspondence source
(Camera Poses / Points) and Similarity vs Rigid. Defaults: camera poses,
similarity. The first is spelled `Camera Poses` rather than `Cameras` because
it fits the two clouds' *poses* onto one another, which under the tree's
vocabulary a bare "Cameras" — an intrinsics record — no longer says. They
live on the panel and persist between opens, and they sit *above* the target
list in the one `Align to ▸` submenu rather than in a popup per target — two
radio pairs do not earn a third level of nesting. A target whose node lacks
feature indexes is greyed individually when Points is selected, so a scene
mixing `sift_files` and `embedded_patches` nodes stays usable.
The fit runs synchronously — by-cameras is trivially small, and by-points
RANSAC over ~10⁵ correspondences is tens of milliseconds; if datasets outgrow
that it moves to a background thread without UI change.

Outcome feedback is recorded in the Action Log and shown in the status line,
with correspondence count, inliers, and post-fit RMS residual:
`Aligned run_b → run_a: 214/243 cameras, RMS 0.031`. On failure (no shared
images, too few correspondences, degenerate geometry, SVD failure) the
transform is left untouched and the reason recorded as a **failed** entry:
`Align run_b → run_a failed: <reason>`. The status line is the most recent
non-query entry (see [action-log.md](action-log.md)), painted in the viewport
overlay under the scene stats — `dock.rs`'s empty-state text only shows it when
*nothing* is loaded, and an alignment by definition happens with two files open.
Unlike a fit's outcome, the per-node eyes, the interaction cursor, the tint and
the solo record entries of their own too, so a comparison session reads back as
a list of what was toggled and when.

"Inliers" means the RANSAC inlier count in point mode and the trimmed-refit's
kept count in camera mode (which runs no RANSAC); the RMS is over that same
subset, recovered under the final transform, and is in the **target's** units.

`Reset Transform` returns a node to identity, and is greyed out for a node that
is already in its own frame. Setting or resetting a transform recomputes the
union scene bounds, re-derives `length_scale`, re-uploads frustum geometry at
the new per-node scale, and rebuilds the track rays — which also dissolves the
shared-frustum-size compromise noted under Rendering once the nodes' scales
agree. `AppState` carries a `transform_epoch` counter that every transform
change bumps; comparing it against the previous frame's is how the upload phase
notices, without diffing a `Vec<Se3Transform>`.

---

## Comparison Affordances

Aligning two reconstructions puts them in one space; these two make that space
readable. Both are **display-only**: neither touches picking, selection,
alignment, the `SfmrReconstruction` in memory, or anything on disk.

### Per-node tint

`SceneNode::tint` is `Original | Tint(&'static TintColor)`, set from the
reconstruction row's `Tint ▸` submenu: `Original`, then a fixed palette, each
entry written in its own color so the choice is visible while making it.

**The palette is the Okabe–Ito colorblind-safe qualitative set** (Okabe & Ito,
*Color Universal Design*, 2002), minus its eighth entry, black: a black tint is
not an identity color but a way to make a node vanish into this viewer's dark
background, which the eye toggle already does honestly. A *fixed* set rather
than a free color picker (resolving the draft's open question): the job of a
tint is telling two nodes apart, which a pre-vetted mutually-distinguishable set
does by construction and a picker leaves to a user who can, and eventually will,
pick two blues.

**Composition rule.** `tint_color` is the palette color in `rgb` and a strength
in `a`, and every scene shader composites the same way:

```wgsl
color = mix(color, tint_color.rgb, tint_color.a)
```

so `a = 0` is the identity — the block's stated "original colors" convention —
and `a = 1` would flatten the node to one flat color. The strength is a constant
0.7: far enough that a node reads as "the orange one" at a glance, short enough
that photo-derived point colors keep the shading that says what you are looking
at. Points, frustum wireframes, image quads, distorted image quads and patch
surfels all apply it, so a tinted node is tinted everywhere.

**What the tint must not swallow.** The highlight colors exist to be
distinguishable; a tint that could drag them toward itself would cost exactly
the legibility they are for. So they are exempt, by two different mechanisms:

- **Point selection (yellow) and hover (cyan), and frustum hover (white)** are
  applied in the *fragment* stage, after the vertex stage has tinted the base
  color, and they replace it outright rather than mixing with it. Nothing extra
  is needed — the ordering is the exemption.
- **Frustum selection (cyan) and the track orange** arrive through the per-image
  color storage buffer, which the vertex shader cannot otherwise tell apart from
  the ordinary frustum white. The convention that separates them: **the default
  frustum color is semi-transparent (alpha 180) and every highlight is written
  at full alpha**, so `a < 1` means "this is the node's own color" and is the
  tint's gate. That is a contract with the CPU-side writer
  (`scene_renderer::upload::frustums`), stated in named constants on that side
  and asserted by a test.
- Track rays are not tinted either: like the background image they are a
  singleton serving *the* selection, not a node's own geometry.

**In the tree**, a tinted node's row carries a small painted swatch in its tint
color — otherwise the only way to find out which node is the orange one would be
to hide the others. Its space is reserved on every row, painted or not, for the
same reason the selection accent bar's is.

### Solo

`AppState::solo: Option<ReconId>` — while `Some`, only that node is drawn.

**Placement (resolving the draft's open question): the row, not the context
menu.** Solo is a transient view mode used over and over while comparing — solo
A, look, solo B, look, off — and a context menu would cost two clicks and a
popup over the viewport every single time. Tint, set once per node, stays in the
menu. Clicking `S` on the soloed node ends the solo.

**State model: an overlay on the eyes, never an edit of them.** Soloing writes
one `Option<ReconId>` and never touches any `SceneNode::visible`, so:

- Un-soloing restores *exactly* the visibility the user had, including nodes
  they had already hidden by hand. The alternative — hide the others and
  remember what to restore — is lossy in that case and needs a saved copy that
  any other path touching `visible` can desync from. One `Option` cannot desync
  with anything.
- Soloing B while A is soloed *moves* the solo: "show only this one" has one
  answer, so the field is a single id rather than a set.
- An eye toggled while soloed changes nothing on screen and everything the
  moment the solo ends, which is the honest reading of "solo hides the others".
  The soloed node's own eye still applies: solo says *hide the others*, not
  *force this one on*, and switching it off leaves an empty viewport (with the
  all-hidden bounds fallback keeping the camera where it was).
- Closing the soloed node ends the solo rather than promoting the next one: a
  solo naming a node that is gone would hide the whole scene with nothing left
  on screen to explain why. A **reload** re-points it at the node's new
  `ReconId`, like the selection. **Opening a file** ends it — you opened that
  file to look at it.

Solo is not selection: it neither selects the node nor is cleared by selecting
another. The one place they meet is `[` / `]`, which **carries an active solo
to the node it lands on** (a solo left behind would leave the viewport showing a
reconstruction no panel is talking about, and the next `]` would appear to do
nothing). Soloed stepping is A/B comparison in its sharpest form: one
reconstruction on screen at a time, the same photo, one keystroke apart.

---

## Loading, CLI, Window Title

- **File > Open…** uses `rfd`'s `pick_files()` (multi-select) and **appends**
  one node per chosen file. Opening a path that is already loaded reloads that
  node in place (fresh read from disk), keeping its position in tree order, its
  label and its display settings — the predictable interpretation, and it
  doubles as a refresh. A reload mints a **new** `ReconId`: re-reading a file
  can change every entity count, so every index-keyed cache entry for the old id
  is wrong, and a new id makes all of them unreachable rather than merely stale.
  The cost is that the reloaded node's image/point selection clears.
- Arriving is also a selection change: an appended node becomes the selected
  reconstruction, which by the invariant below clears the image and point
  selection. You opened the file to look at it, and no panel should be left
  showing another file's row.
- **File > Close All** clears the scene. Individual close lives in the Scene
  panel.
- **Demo data** becomes a node labeled `demo` (`path: None`) and appends like
  any other load. This also fixes the current demo-load path that skips the
  cache/selection resets `load_file` performs — node lifecycle is now one code
  path. `Reload from Disk` is disabled for it: there is no file to re-read.
- **CLI**: `sfm explorer` accepts multiple paths
  (`@click.argument("sfmr_files", nargs=-1)`), and `lib.rs` loads every
  trailing argument instead of only `args[1]`.
- **Window title**: unchanged for zero or one file (`SfM Explorer`,
  `SfM Explorer - run_a.sfmr`); with N > 1 files:
  `SfM Explorer - run_a.sfmr (+2)`. The exact base title is load-bearing for
  the `ui_basic` Windows attach path and keeps its current value — which is also
  why a *nameless* first node (demo data) leaves the title at the bare base
  however many files follow it, rather than inventing a name for the count.

The scene-stats overlay (top-left) sums across the effectively visible nodes
(the same eye-AND-solo rule the draw loop uses) and leads with the file count
when more than one is contributing:
`2 reconstructions | 1.4M points (12 at ∞) | 421 images | 60 fps`.

---

## Panel-by-Panel Impact

| Panel | Change |
|-------|--------|
| **3D Viewer** | Renders all visible nodes (per-recon draws). Camera view mode stores an `ImageRef`; `,` / `.` step within that recon's images, `[` / `]` step across reconstructions with same-named-image carry-over. Hover text gains the recon label. |
| **Scene Graph** | New (this spec). |
| **Image Browser** | Bound to the **selected** reconstruction; a small header names it, shown only once more than one file is loaded (with a single one it would be chrome in an already-short panel). Thumbnail cache guarded by the owning `ReconId` (fixing the count-only invalidation bug; index keys stay local since the strip only ever shows one reconstruction, so a recon switch drops the old textures instead of accumulating them). Animation and the color barcode are per-selected-recon. |
| **Image Detail** | Selection-driven — works via `ImageRef` naturally. `loaded_image` and overlay state re-keyed by `ImageRef`. |
| **Point Track Detail** | Selection-driven via `PointRef`. Its `pt3d_<hash>_<index>` IDs already embed the per-recon content hash, so displayed IDs are already unambiguous across files. Texture maps re-keyed by `ImageRef`. |

Cross-panel selection semantics are otherwise untouched: clicking a frustum in
the 3D view selects that image, which selects its reconstruction (switching
the browser's strip if needed), loads in Image Detail, and highlights both the
reconstruction row and the camera row in the Scene panel.

### Caches

Every image/point-keyed cache re-keys by ref:

- `AppState::sift_cache: HashMap<ImageRef, CachedSiftFeatures>`
- `AppState::full_res_cache: HashMap<ImageRef, Option<ImageU8>>`
- panel-local texture caches as listed above
- `SceneRenderer` bg image: `Option<ImageRef>`

Closing a node purges its entries from every cache (a `retain` on
`ref.recon != id`), releases its `ReconResources`, clears any
selection/hover/camera-view state pointing into it, reassigns pick bases, and
recomputes union bounds. The camera-view rule holds for *any* way a node
leaves the scene, not just explicit close — replacing the scene by opening a
file drops a camera view whose node is gone, rather than letting it point at
the same raw index in an unrelated reconstruction. Because `ReconId` is never
reused, a missed purge can go stale but can never alias.

The purge runs in three places, which is what the split of ownership costs:
`AppState` drops its own caches and selection, `dock.rs` asks each panel to drop
its private texture caches (`forget_recon`), and the renderer releases the GPU
bundle from `retain_nodes` on the next frame. `Reload from Disk` runs the same
three against the *old* id.

---

## Performance

- **Draw calls**: passes × visible nodes ≈ 4 × N. Trivial for N ≤ tens.
- **Uploads**: unchanged per node (one-time on load); membership changes touch
  only the affected node's buffers plus one small uniform write per node for
  pick bases. The existing cheap paths survive per node: frustum *color*
  writes on selection change, geometry re-upload only on
  `length_scale`/frustum-size change (now per affected node).
- **GPU memory**: linear in loaded data, same per-item budget as
  [architecture.md](architecture.md). Thumbnail atlases dominate
  (~6.25 KB × 128² × … — 625 MB at 10K cameras); N large reconstructions
  multiply that, so the already-planned LRU/async thumbnail loading becomes
  more pressing but is not a prerequisite at the 2–8 file design point.
- **Scene panel**: camera lists are virtualized (`show_rows`); per-frame cost
  is proportional to visible rows, not camera count. Point listings are
  bounded (selection/hover rows only) by design.
- **Pick decode**: binary search over ≤ tens of ranges, per readback — noise.

---

## Testing

- **Scene panel** lib tests run whole egui frames via `Context::run_ui`
  (the `point_track_detail/tests.rs` pattern): tree structure, expansion
  state via explicit `CollapsingState` IDs, row click → response mapping,
  selection auto-scroll, eye and interaction-cursor toggles,
  selected-reconstruction marking, `Align to` menu gating (point mode
  disabled without feature indexes).
- **Upload tests** on the `noop` wgpu backend
  (`scene_renderer/upload/tests.rs` pattern): two-node upload produces two
  bundles; close releases one; pick bases are contiguous, non-overlapping,
  and stable under add/remove churn.
- **Pick encode/decode** round-trip unit tests, incl. boundary values
  (`base + count - 1`, sentinel, none) and the non-interactive case: a
  node with `interactive` off produces no hover/selection from readback
  dispatch, while Scene-panel selection of the same node still works.
- **State tests**: cache purge on close, selection clearing when the owning
  node closes, selected-reconstruction fallback and the
  finer-selection-clears-on-recon-switch invariant, `[` / `]` stepping with
  same-named-image carry-over (and clearing when the name is absent), label
  disambiguation.
- **Alignment round-trip**: node B built as node A under a known similarity
  (points *and* camera poses); `Align to` recovers it (camera and point modes),
  asserted on where B's points land rather than on the transform's components,
  which sidesteps the quaternion sign ambiguity. The composed
  `target.transform ∘ T_fit` chaining is verified with a pre-transformed
  target, and the same fixture checks that Rigid refuses to absorb a scale,
  that points at infinity are left out, that every failure path leaves the
  transform untouched, and that an aligned node's camera pose (what
  `enter_camera_view` builds from) lands on the target's.
- **Transform plumbing**: the `model` matrix read back the way a vertex shader
  would (noop backend), splat size scaling with a scaled node, union bounds and
  the `length_scale` seed following a transform and returning on reset, track
  rays built through it, and the transform carried across a reload.
- **Tint**: the `Tint ▸` submenu offers `Original` plus every palette entry;
  picking one writes it to that node and no other, and the menu survives the
  pick so a second color is one click away; the swatch appears on the tinted row
  only and takes no room from the others; working the menu moves no selection;
  the tint survives a reload; an untinted node writes the all-zero `a = 0`
  uniform and a tinted one writes its color at the strength constant; the
  palette entries are mutually distinguishable and none is near-black. On the
  renderer side the tint reaches that node's `ReconUniforms` and nobody else's,
  and `frustum_colors` is asserted to write every highlight at full alpha and
  the default below it — the invariant `frustum.wgsl` gates the tint on.
- **Solo**: the row's toggle reports the node (and reports it again to switch
  off) without writing any eye, and is not swallowed by the row-wide target nor
  a source of its context menu; soloing hides every other node while leaving
  their eyes alone, and un-soloing restores them *including* one hidden before
  the solo; soloing a second node moves the solo; an eye toggled while soloed
  applies when the solo ends; closing the soloed node ends the solo, closing
  another leaves it; opening a file ends it; `[` / `]` carry it. On the renderer
  side, a node that is not effectively visible drops out of every pass and out
  of the bounds union, the all-hidden fallback still frames the loaded nodes,
  and the stats overlay counts the soloed node alone.
- `ui_basic` keeps matching the base window title, and gains a check that the
  Scene panel's rows reach a real window — including the row's solo toggle,
  since a third glyph button squeezed onto a row is exactly the kind of thing
  that lays out correctly under `Context::run_ui` and not in a window. The
  context-menu check covers the flat entries only: a submenu button
  (`Align to ▸`, `Tint ▸`) does not surface under the accessibility `button`
  role, and its contents exist only once opened. The multi-file title case stays
  a lib test: driving it through `ui_basic` would need two real `.sfmr` fixtures
  on disk and a way past the file dialog, for a string `window_title` already
  decides on its own.

---

## Implementation Phases

1. **Typed refs, single recon** *(done)* (mechanical, no behavior change):
   introduce `ReconId`/`ImageRef`/`PointRef`, `Vec<SceneNode>` with the
   invariant len ≤ 1, re-key all caches, route panels through the selected
   reconstruction. Everything still loads/replaces as today.
2. **Renderer bundles + picking** *(done)*: `ReconResources`, per-recon uniform
   block with model matrix and pick bases, new pick encoding, per-recon draws.
   Still one node loaded — but the machinery is multi-ready and covered by
   noop-backend tests. The model matrix is identity and `pickable` is always 1
   until phases 3–4 supply the node transform and the interaction toggle;
   `tint_color` is carried but not yet read by any shader (phase 5).
3. **Multi-load + Scene Graph panel** *(done)*: append-on-open, multi-select
   dialog, multi-path CLI, node close/reload, the Scene tab with tree, per-node
   eye and interaction-cursor toggles, selected-reconstruction handling,
   window title and stats overlay. The feature is on.
4. **Transforms + alignment** *(done)*: the per-recon model matrix exercised
   end to end, `Align to…` (by cameras and by points), `Reset Transform`,
   status feedback.
5. **Comparison affordances** *(done)*: per-node tint palette and solo mode —
   the last display-side pieces of the original design, and what makes two
   aligned reconstructions readable once they occupy the same space.

The transform editor, "Save Aligned Copy…" and multi-way align were briefly
folded into phase 5 and have been returned to Future Directions: each is new
capability rather than a finishing touch, and the export in particular would
break the invariant every phase has held — that a node transform is view state
that never touches the reconstruction or the disk.

Phases 1–2 are refactors shippable behind unchanged UX; the feature turns on
in phase 3.

---

## Future Directions

- **Transform tooling beyond `Align to…`**: a numeric transform editor
  (inspect/tweak the fitted `Se3Transform`), a "Save Aligned Copy…" export
  that bakes `node.transform` through the `sfm xform` machinery, and multi-way
  alignment (align every node to one reference in a single action, mirroring
  `sfm align`'s multi-way mode).
- **Cross-reconstruction correspondence**: images with the same name/path in
  two nodes are "the same photo" — hover/selection echo across nodes
  (highlight the sibling frustum), side-by-side pose deltas, per-camera
  distance overlays.
- **Difference visualization**: per-point nearest-neighbor distance coloring
  between two aligned nodes; per-camera pose-error glyphs.
- **Deeper hierarchy**: rig groups under Camera Images, user grouping/reordering,
  per-node point-size overrides.
- **Session persistence**: save/restore the scene (paths, transforms, tints,
  visibility) as a small project file.
- **Filtered point views**: query-driven point listings in the Scene panel
  (worst error, longest tracks) in place of the impossible full listing.

---

## Open Questions

- Should the Image Browser optionally show *all* reconstructions as grouped
  sections instead of only the selected one? (Deferred: playback and stepping
  are per-sequence concepts; grouped display complicates both for unclear
  benefit at N ≤ 8.)
- Per-node opacity (ghosting a reference reconstruction) — cheap to add to the
  per-recon uniform block; UI slider placement is the only question.
- Should soloing more than one node at a time be possible (a set rather than an
  `Option`)? Deferred: "show only this one" has one answer, and per-node eyes
  already express "show these three". Worth revisiting only if comparing three
  or more aligned nodes turns out to be common.
