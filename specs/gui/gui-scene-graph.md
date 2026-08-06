# Scene Graph and Multi-Reconstruction Support

*Status: Draft — design proposal, not yet implemented.*

This document specifies multi-reconstruction support for SfM Explorer: loading
several `.sfmr` files at once, organizing them as nodes in a scene graph, and a
new **Scene Graph panel** — a tree view for browsing, toggling, and selecting
what is loaded. The motivating use case is *comparison*: looking at two or more
reconstructions of the same scene side by side in one 3D space.

Related specs: [gui-architecture.md](gui-architecture.md) (rendering pipeline
this extends), [gui-cross-panel-hover.md](gui-cross-panel-hover.md) (selection
and hover model this generalizes),
[gui-camera-views.md](gui-camera-views.md) (pick buffer),
[gui-viewport-hud.md](gui-viewport-hud.md) (global display controls).

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
│   ├── Cameras (243)             ← fixed group node
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
```

All selection and hover state moves from `Option<usize>` to these typed refs:

```rust
pub selected_image: Option<ImageRef>,
pub selected_point: Option<PointRef>,
pub hovered_image:  Option<ImageRef>,
pub hovered_point:  Option<PointRef>,
```

There is still exactly **one** selected image and **one** selected point
globally — the cross-panel selection model of
[gui-cross-panel-hover.md](gui-cross-panel-hover.md) is unchanged; only the
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
    pub show_cameras: bool,
    pub show_patches: bool,
    pub show_points_at_infinity: bool,
    pub tint: NodeTint,              // Original | Tint(color)
    /// Similarity transform (uniform scale · rotation · translation) mapping
    /// this node's native coordinates into the shared world space. Identity on
    /// load. Set by the "Align to…" operation; see "Node Transforms and
    /// Alignment" below.
    pub transform: Se3Transform,

    /// This node's data needs (re-)upload to the GPU.
    pub needs_upload: bool,
}
```

`AppState` replaces its single slot with:

```rust
pub scene: Vec<SceneNode>,           // tree order = load order (reorderable later)
pub selected_recon: Option<ReconId>, // see "The selected reconstruction" below
```

The per-node `needs_upload` flag replaces the global `points_need_upload`
bool; closing a node additionally enqueues a resource-release for its
`ReconId` (see Rendering).

**Effective visibility** of a layer in node *n* is the AND of three switches:
the global HUD Layers toggle (unchanged, now acting as a master switch across
all nodes), the node's `visible` eye, and the node's per-group eye. The Grid
stays global-only.

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
  photo, two solves.
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
tabs it is not closeable, and can be re-docked freely.

### Tree rows

The tree is drawn with `egui::collapsing_header::CollapsingState` with
explicit IDs (the viewport HUD's pattern, `viewer_3d/hud.rs`) — *not*
`CollapsingHeader` — so expansion state is addressable and testable. Rows are
fixed-height for virtualization.

**Reconstruction row** — `[▸] [👁] [🖱] run_a   1.2M pts · 243 cams`
- Expand triangle, visibility eye (node master), interaction cursor toggle
  (`interactive` — greyed when off), label, compact counts.
- Bold + accent bar when selected. Click: select this reconstruction.
  Double-click: zoom-to-fit this node.
- Context menu: `Select`, `Zoom to Fit`, `Align to ▸` (one entry per other
  loaded node — see "Node Transforms and Alignment"), `Reset Transform`,
  `Tint ▸` (Original / palette of distinguishable colors), `Reload from Disk`,
  `Close`. `Solo` (hide all others) is a nice-to-have.

**Cameras group row** — `[▸] [👁] Cameras (243)`
- Eye drives `show_cameras` for the node.
- Expands to one row per image, **in image order, virtualized** via
  `ScrollArea::show_rows` — only visible rows are laid out, so 50K cameras
  cost tens of rows per frame.

**Camera row** — `IMG_0001.jpg`
- Click: `selected_image = Some(ImageRef)`. Double-click: enter/switch camera
  view (same semantics as a browser thumbnail double-click).
- Hover: sets `hovered_image` (participates in cross-panel hover exactly like
  a browser thumbnail; the row highlights when the same image is hovered
  elsewhere, e.g. from the 3D viewport pick).
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
    pub select_point: Option<PointRef>,
    pub request_camera_view: Option<ImageRef>,
    pub hovered_image: Option<ImageRef>,
    pub hovered_point: Option<PointRef>,
    pub has_pointer: bool,
    pub select_recon: Option<ReconId>,
    pub align_node: Option<(ReconId, ReconId, AlignOptions)>, // source, target
    pub reset_transform: Option<ReconId>,
    pub zoom_to_node: Option<ReconId>,
    pub close_node: Option<ReconId>,
    pub reload_node: Option<ReconId>,
}
```

`has_pointer` ownership of hover state follows
[gui-cross-panel-hover.md](gui-cross-panel-hover.md) unchanged: when the
pointer is over the Scene panel, it owns both hover fields.

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

Shared and unchanged: pipelines, render targets, samplers, the unit quad
vertex buffer, EDL/target-indicator/track-ray/bg-image resources (all
singletons by nature — track rays and bg image serve *the* selection, which is
single).

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

A new per-recon uniform block, one 256-byte-aligned slice per node in a single
buffer bound with dynamic offsets (or per-recon bind groups — implementation's
choice):

```
model: mat4x4<f32>        // node.transform as a matrix
point_size: f32           // per-recon auto size × global 2^point_size_log2
point_pick_base: u32
image_pick_base: u32
pickable: u32             // 0 → emit PICK_TAG_NONE (node.interactive off)
tint_color: vec4<f32>     // a = 0 → original colors
```

Vertex shaders apply `model` before `view_proj`. Homogeneous points at
infinity need no special-casing: a direction transforms as
`(model × vec4(dir, 0)).xyz` — the linear part rotates it, translation drops
out, and uniform scale is irrelevant to a direction.

The existing global uniforms keep working with one change of meaning:
`selected_point_index` / `hovered_point_index` / `hovered_image_index` become
**global pick indices** (base + local, see below), so the shader compare
`instance_index + pick_base == selected_point_index` stays a single u32
comparison. Sentinel remains `0xFFFFFFFF`.

### Derived scalars and framing

- `auto_point_size`, `camera_nn_scale`: computed **per recon** at upload, as
  today's functions already do — they just stop being singletons. Point splat
  world size is per-recon; the HUD size slider is a global multiplier on top.
- **Scene bounds** (adaptive clip planes, `Z` zoom-to-fit, supernova/grid
  scaling): the union of visible nodes' bounding spheres, each transformed by
  its node transform. Recomputed when the visible set, a node transform, or a
  node's data changes.
- `length_scale` (drives frustum stub depth, target indicator): re-derived
  from the visible union whenever the node set changes, exactly as it is
  re-derived on load today; still one global, still user-adjustable. This is a
  known compromise — two reconstructions at wildly different scales will share
  one frustum size until they're aligned (see "Node Transforms and
  Alignment").
- Per-node `Zoom to Fit` frames that node's transformed bounds.

### Track rays and background image

Track rays are CPU-built per selection from the owning recon; the builder
additionally applies `node.transform` to camera centers and ray points.
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
- Bases are (re)assigned whenever a node is added or removed — a uniform
  rewrite per node, nothing else.
- CPU decode: binary search a sorted `(base, ReconId)` table per entity kind →
  `(ReconId, local index)` → `ImageRef` / `PointRef`. All decode sites
  (`app.rs` pick dispatch, hover overlay text) go through one helper.
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
  camera.
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

The estimation machinery already exists in
`sfmtool-core::analysis::alignment` and
`sfmtool-core::reconstruction::point_correspondence`; the GUI adds only
correspondence gathering and UI:

- **Correspondences by cameras** (default): images matched by `name` across
  the two nodes; corresponded camera centers feed the fit. Works whenever the
  reconstructions were solved from overlapping image sets — the typical
  comparison case (two solver runs over the same shoot) — and mirrors
  `sfm align`'s by-cameras mode. Only the shared subset matters; disjoint
  extra images on either side are simply not correspondences.
- **Correspondences by points**: `find_point_correspondences` matches 3D
  points through shared feature observations in shared images, yielding a much
  larger set; points at infinity are excluded from the fit. Requires
  feature-indexed observations (`sift_files` source) in both nodes — the menu
  option is disabled otherwise.
- **Estimation**: `ransac_alignment` for inlier selection on point
  correspondences, then `estimate_alignment` with trimmed refit
  (`AlignmentParams { rounds, keep_fraction, estimate_scale }`); camera-mode
  fits, with far fewer correspondences, use trimming alone. A Rigid /
  Similarity choice maps to `estimate_scale`.

Options are deliberately few (a small popup): correspondence source
(Cameras / Points) and Similarity vs Rigid. Defaults: cameras, similarity.
The fit runs synchronously — by-cameras is trivially small, and by-points
RANSAC over ~10⁵ correspondences is tens of milliseconds; if datasets outgrow
that it moves to a background thread without UI change.

Outcome feedback lands in the status message with correspondence count,
inliers, and post-fit RMS residual:
`Aligned run_b → run_a: 214/243 cameras, RMS 0.031`. On failure (no shared
images, degenerate geometry, SVD failure) the transform is left untouched and
the reason reported.

`Reset Transform` returns a node to identity. Setting or resetting a
transform recomputes the union scene bounds and re-derives `length_scale` —
which also dissolves the shared-frustum-size compromise noted under Rendering
once the nodes' scales agree.

---

## Loading, CLI, Window Title

- **File > Open…** uses `rfd`'s `pick_files()` (multi-select) and **appends**
  one node per chosen file. Opening a path that is already loaded reloads that
  node in place (fresh read from disk), keeping its node settings — the
  predictable interpretation, and it doubles as a refresh.
- **File > Close All** clears the scene. Individual close lives in the Scene
  panel.
- **Demo data** becomes a node labeled `demo` (`path: None`) and appends like
  any other load. This also fixes the current demo-load path that skips the
  cache/selection resets `load_file` performs — node lifecycle is now one code
  path.
- **CLI**: `sfm explorer` accepts multiple paths
  (`@click.argument("sfmr_files", nargs=-1)`), and `lib.rs` loads every
  trailing argument instead of only `args[1]`.
- **Window title**: unchanged for zero or one file (`SfM Explorer`,
  `SfM Explorer - run_a.sfmr`); with N > 1 files:
  `SfM Explorer - run_a.sfmr (+2)`. The exact base title is load-bearing for
  the `ui_basic` Windows attach path and keeps its current value.

The scene-stats overlay (top-left) sums across visible nodes and leads with
the file count when more than one is loaded:
`2 reconstructions | 1.4M points (12 at ∞) | 421 images | 60 fps`.

---

## Panel-by-Panel Impact

| Panel | Change |
|-------|--------|
| **3D Viewer** | Renders all visible nodes (per-recon draws). Camera view mode stores an `ImageRef`; `,` / `.` step within that recon's images, `[` / `]` step across reconstructions with same-named-image carry-over. Hover text gains the recon label. |
| **Scene Graph** | New (this spec). |
| **Image Browser** | Bound to the **selected** reconstruction; a small header names it. Thumbnail cache re-keyed by `ImageRef` (fixing the count-only invalidation bug). Animation and the color barcode are per-selected-recon. |
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
recomputes union bounds. Because `ReconId` is never reused, a missed purge can
go stale but can never alias.

---

## Performance

- **Draw calls**: passes × visible nodes ≈ 4 × N. Trivial for N ≤ tens.
- **Uploads**: unchanged per node (one-time on load); membership changes touch
  only the affected node's buffers plus one small uniform write per node for
  pick bases. The existing cheap paths survive per node: frustum *color*
  writes on selection change, geometry re-upload only on
  `length_scale`/frustum-size change (now per affected node).
- **GPU memory**: linear in loaded data, same per-item budget as
  [gui-architecture.md](gui-architecture.md). Thumbnail atlases dominate
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
- **Alignment round-trip**: node B built as node A under a known similarity;
  `Align to` recovers it (camera and point modes), and the composed
  `target.transform ∘ T_fit` chaining is verified with a pre-transformed
  target.
- `ui_basic` keeps matching the base window title; a multi-file title case is
  added.

---

## Implementation Phases

1. **Typed refs, single recon** (mechanical, no behavior change):
   introduce `ReconId`/`ImageRef`/`PointRef`, `Vec<SceneNode>` with the
   invariant len ≤ 1, re-key all caches, route panels through the selected
   reconstruction. Everything still loads/replaces as today.
2. **Renderer bundles + picking**: `ReconResources`, per-recon uniform block
   with model matrix and pick bases, new pick encoding, per-recon draws.
   Still one node loaded — but the machinery is multi-ready and covered by
   noop-backend tests.
3. **Multi-load + Scene Graph panel**: append-on-open, multi-select dialog,
   multi-path CLI, node close/reload, the Scene tab with tree, per-node
   eye and interaction-cursor toggles, selected-reconstruction handling,
   window title and stats overlay.
4. **Transforms + alignment**: the per-recon model matrix exercised end to
   end, `Align to…` (by cameras, then by points), `Reset Transform`, status
   feedback.
5. **Comparison affordances**: per-node tint palette, per-node zoom-to-fit,
   solo mode.

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
- **Deeper hierarchy**: rig groups under Cameras, user grouping/reordering,
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
- Whether `Solo` (hide all others) earns a spot on the reconstruction row
  itself rather than the context menu.
- Tint palette: fixed distinguishable set (Okabe-Ito-like) vs. free color
  picker. Draft says fixed set first.
