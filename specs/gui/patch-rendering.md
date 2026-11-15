# Patch (Surfel) Rendering in the 3D Viewport

The SfM Explorer draws a reconstruction's **embedded patches** in the 3D
viewport as small textured, oriented quads ("surfels"). An embedded patch is a
small square of surface stored with a 3D point in the `.sfmr` file: a frame (the
point's position and two in-plane half-extent vectors, whose cross product is
the normal) and an RGBA bitmap of what the surface looks like there. The viewer
draws one surfel for each point that has both. A reconstruction that carries
frames but no bitmaps draws no patches.

The point-splat renderer draws each 3D point as a camera-facing round billboard
(see [point-cloud-rendering.md](point-cloud-rendering.md)). A patch adds the
surface information a splat lacks: instead of a view-facing dot, each point is
drawn as a **world-oriented rectangle** textured with the point's patch bitmap,
so the viewport shows the reconstructed *surface* — orientation, texture, and
local appearance — not just a cloud of colored specks.

This is the 3D-viewport counterpart to the 2D patch-strip inspectors
(`sfm inspect --strips`, `sfm compare --strips`), which show a point's patch
and its views of it as rows of 2D tiles.

---

## Data source (already present)

Everything this feature needs is already carried on the loaded
`SfmrReconstruction` (`crates/sfmtool-core/src/reconstruction/data.rs:203-212`),
as columnar per-3D-point arrays parallel to `points`:

| Field | Type | Meaning |
|-------|------|---------|
| `patch_u_halfvec_xyz` | `Option<Array2<f32>>` `(P, 3)` | in-plane axis **u** × half-extent, world space |
| `patch_v_halfvec_xyz` | `Option<Array2<f32>>` `(P, 3)` | in-plane axis **v** × half-extent, world space |
| `patch_bitmaps_y_x_rgba` | `Option<Array4<u8>>` `(P, R, R, 4)` | per-point RGBA tile; **alpha = per-pixel cross-view agreement confidence** |
| `points[i].position` | `Point3<f64>` | patch **center** (not stored on the patch — it *is* the point) |
| `points[i].w` | `f64` | `1.0` finite, `0.0` at infinity |
| `points[i].normal` | `Vector3<f32>` | outward normal (`= normalize(u × v)`) |

Key facts that shape the design:

- **The center is the point position; the normal is derived** from `u × v` (a
  right-handed frame — `u × v` points back toward the camera for a front-facing
  patch). The raster reverses `v` (steps rows along `−v`) to render un-mirrored,
  which is why the vertex shader flips the bitmap's `v` texture coordinate. There
  is no separate patch struct at rest — the frame is the two half-vectors.
- **A point with no patch is an all-zero row**, of the frame or of the
  bitmaps. A point has a patch to draw when its `u` row is nonzero and its
  bitmap row is not all zero (`patch_bitmap_is_stored` in `sfmtool_core`, the
  rule the Track Table's *Patch* cell and the web export read too). The
  uploader skips every other point, so no surfel is drawn as a black square.
- **`u`/`v` are present-or-absent together.** Bitmaps require the frame but are
  independent of it: a reconstruction can carry frames with **no** bitmaps (e.g.
  a `to-embedded-patches` result before `refine_normals(render_bitmaps=True)`).
  See [What gets drawn](#what-gets-drawn-by-what-the-reconstruction-carries).
- **Four corners are trivial:** `center ± u_halfvec ± v_halfvec`. This is exactly
  `OrientedPatch::to_world(±1, ±1)` (`patch/cloud.rs:92-98`), and the `(s, t) ∈
  [-1, 1]²` patch frame maps to bitmap UV identically to the existing camera
  image-quad shader (`u = (x+1)·0.5`, `v = (y+1)·0.5`,
  `shaders/image_quad.wgsl:24,40-42`).

No new format work, no new bindings, no `maturin` rebuild: the renderer reads
fields the reconstruction already exposes.

---

## The pipeline mirrors the camera image-quad pipeline

The viewport already has a textured-quad pipeline — the **image quad** that
projects a camera thumbnail onto a frustum's far plane
(`scene_renderer/pipelines/image_quad.rs`, `shaders/image_quad.wgsl`, drawn in
`scene_renderer/render.rs:270-283`). The patch pipeline is that pipeline with two
changes: per-instance orientation vectors (instead of four precomputed corners),
and a per-instance atlas layer. It lives in **Pass 1** (the MRT geometry pass)
alongside points and image quads, so patches occlude and are occluded by both.

### Per-instance data

Compute corners in the vertex shader (smaller buffer, mirrors the point-splat
billboard expansion) rather than precomputing four corners CPU-side:

```rust
#[repr(C)]
struct PatchInstance {
    center: [f32; 3],       // world position (unit direction when w == 0)
    w: f32,                  // 1.0 finite, 0.0 at infinity
    u_halfvec: [f32; 3],     // world-space u axis × half-extent
    _pad0: f32,
    v_halfvec: [f32; 3],     // world-space v axis × half-extent
    atlas_layer: u32,        // page-packed index into the patch texture array
    point_index: u32,        // global recon.points index — for picking (see below)
}
```

`atlas_layer` and `point_index` differ because zero-row points get no patch: the
uploader compacts present patches into the instance/atlas buffers, so the
instance's atlas slot is *not* its point index. `point_index` is carried
explicitly so picking resolves to the original point.

### Vertex shader (`shaders/patch.wgsl`)

Given the unit quad the point pipeline also reads (the splat corner buffer,
`SplatCorner { position: [f32; 2], dimmed: u32 }` in slot 0, corners at `±1`,
whose `dimmed` marks the Track Table filter's dimmed draw; see
[point-cloud-rendering.md](point-cloud-rendering.md) § "The Track Table's
filter"):

```wgsl
// s, t in [-1, 1] from the unit quad corner
let corner_world = center + s * u_halfvec + t * v_halfvec;   // finite (w == 1)
let clip = view_proj * vec4(corner_world, 1.0);
// uv for the bitmap: (s, t) -> [0, 1]
let uv = vec2((s + 1.0) * 0.5, (t + 1.0) * 0.5);
```

For **points at infinity** (`w == 0`), the corner is a *direction*
`center + s·u + t·v` and must be transformed with the same `w = 0` homogeneous
path the point splats and background mesh already use — `view_proj * vec4(dir,
0.0)`, translation drops out, depth biased to `INF_EPSILON` (see
[point-cloud-rendering.md §Points at Infinity](point-cloud-rendering.md#points-at-infinity)
and `OrientedPatch::corner_homogeneous`, `patch/cloud.rs:100-110`). An infinity
patch is a tangent-to-the-sky quad; it holds angular position under camera
translation, exactly like its point splat.

The same vertex shader also applies the front-face cull (see
[Blending and face culling](#blending-and-face-culling)): a back-facing patch
returns a clipped vertex before corner expansion, so its quad drops out.

### Fragment shader

1. Measure the fragment's distance to the quad's nearest edge in screen pixels,
   `edge_px`: each patch coordinate's distance to its nearer edge over the
   length of its screen-space gradient (`dpdx`, `dpdy`). It comes first because
   WGSL allows derivatives only in uniform control flow, before any `discard`.
2. Sample the atlas layer at `uv` → `texel` (RGBA).
3. **Outline.** A fragment of a highlighted surfel (its point selected, hovered
   or called out) with `edge_px < 2.5` is in the outline band; see
   [Picking, hover, selection](#picking-hover-selection).
4. **Coverage from bitmap alpha.** `texel.a` is per-pixel cross-view confidence,
   not opacity. Treat it as coverage: `discard` when `texel.a < alpha_cutoff`
   so ragged patch edges and unfilled corners can drop out and the quad reads as
   a fitted surfel, not a hard rectangle. **The default is `0.0`** — discard
   nothing, render every texel opaque — which reads better in practice than a
   carved edge; raise the slider to trim ragged borders. A fragment in the
   outline band is never discarded, so the outline traces the whole square.
5. Colour: the texel, tinted, then the brush's red or pale red for mask word
   `3` or `4`, else the highlight colour in the outline band. Output
   premultiplied RGBA to the color target, at the global `patch_opacity` times
   the filter's dim alpha, or opaque in the outline band.
6. Depth targets — see below.
7. Pick target: `PICK_TAG_POINT | point_index`, in the outline band too.

### Render targets — reuse the image-quad convention

The three-MRT + hardware-depth layout is identical to points/image-quads
(`point-cloud-rendering.md §Three Render Targets`). Patches follow the
**image-quad** convention, not the point convention:

| Target | Patch writes | Rationale |
|--------|-------------|-----------|
| Color `Rgba8UnormSrgb` | premultiplied patch color × opacity | visible surfel |
| Linear depth `R32Float` | **`0.0`** (EDL passthrough sentinel) | a textured surface shouldn't get EDL edge-darkening; matches image quads |
| Pick `R32Uint` | `PICK_TAG_POINT \| point_index` | clicking a patch selects its point |
| HW depth `Depth32Float` | **real** reversed-Z depth, `depth_write = true`, `depth_compare = Greater` | correct occlusion against points, frustums, and other patches |

Writing real hardware depth but `0.0` linear depth means patches **occlude
correctly** yet are treated by the EDL pass and the target-indicator occlusion
test as "surface" pixels (the existing `depth == 0` branch), which is the
behavior we want. See [Open questions](#open-questions) for the EDL-on-patches
alternative.

`PICK_TAG_POINT | point_index` is deliberate: a patch and its point are the same
entity, so a patch click reuses all existing point selection/hover/track-detail
wiring for free — no `PICK_TAG_PATCH`, no new pick decode branch.

### Blending and face culling

- **Blending:** `PREMULTIPLIED_ALPHA_BLENDING` on the color target, as points and
  image quads use. With `depth_write = true`, back-to-front ordering isn't
  guaranteed, so soft-alpha edges between overlapping patches can composite
  imperfectly; the `alpha_cutoff` discard keeps this from being visible in
  practice (hard coverage instead of translucency). Order-independent
  transparency is out of scope for v1.
- **Culling:** render **front-face only** — a patch is drawn only from the side
  its outward normal faces. This is a facing test in the vertex shader, not
  hardware winding culling (the pipeline keeps `cull_mode: None`): the
  reversed-Z, −Z-forward viewport camera has no other culled geometry to anchor
  a winding convention to, so a per-instance geometric test is unambiguous. Each patch
  computes `outward = cross(u_halfvec, v_halfvec)` (a positive multiple of the
  normal — no normalize needed) and collapses its quad to a clipped vertex when
  `dot(outward, camera_pos − center) ≤ 0`. A point at infinity is never culled
  (its normal faces every observer), and neither is the surfel of a selected,
  hovered or called-out point, so its outline and texture show from behind as
  well (see [Picking, hover, selection](#picking-hover-selection)). The camera world position is supplied in
  `PatchUniforms.camera_pos`. Dimming/tinting back faces instead of hiding them
  is a possible later refinement.

---

## Texture atlas (mirror the thumbnail atlas)

All patch bitmaps share one resolution `R` (`patch_bitmap_resolution` in
`points3d/metadata.json`), so they pack naturally into a
**`texture_2d_array<f32>`** (`Rgba8UnormSrgb`), exactly like the camera-thumbnail
atlas (`scene_renderer/upload/thumbnails.rs`). Because `R` is small (typ. 16–64) and
the patch count `P` can exceed the array-layer limit (~2048), reuse the
thumbnail atlas's **page-grid packing** (multiple patches tiled per layer,
`atlas_layer` decodes to `(layer, cell)` — `upload/thumbnails.rs`), or a plain
one-patch-per-layer array when `P` is small.

Upload path (new `SceneRenderer::upload_patches` in `upload/patches.rs`, gated on the
frame arrays **and** the bitmaps being `Some`):

1. Return early unless both the `(u, v)` frame arrays and `patch_bitmaps_y_x_rgba`
   are present (a frame-only reconstruction renders nothing in v1 — flat-shaded
   fallback deferred). Walk the points and skip each whose `u` row is zero or
   whose bitmap row is all zero (points without a patch).
2. Fill the atlas a row of cells at a time: each present patch's `(R, R, 4)`
   tile is scattered into a band spanning the atlas, and each filled band goes
   up in one `write_texture`. Then push a `PatchInstance` per patch (center =
   `points[i].position`, `w = points[i].w`, `u/v` from the half-vec arrays,
   `atlas_layer`, `point_index = i`).

   **The band is what keeps the upload proportional to the pixels rather than
   to the patch count.** The bitmaps are tile-major and the atlas is grid-major,
   so one of the two has to scatter into the other. A `write_texture` per tile
   asks the driver to, and costs one call per patch; the call dominates a tile
   this small, so a 2 KiB patch tile cost about what a 64 KiB thumbnail did and
   an atlas was priced by how many tiles it held. Scattering into a band first
   makes one row of cells contiguous and its upload one call. A row of cells
   rather than a whole page, because a page is bounded only by the GPU's 2D
   texture limit while a band is `atlas_width` by `R`. See
   `upload/atlas.rs`.
3. Build the bind group (uniform + `texture_2d_array` + sampler), mirroring
   `rebuild_frustum_bind_group` (`upload/frustums.rs`).

Called from `App::prepare_uploads` (`app.rs:224-388`) beside `upload_thumbnails`.

Each instance also has an entry in a liveness buffer, its point's mask word
([point-cloud-rendering.md](point-cloud-rendering.md) § "Per-Point Data"), which
is how an edit's deleted set reaches this pass. A deleted point's surfel is a
clipped vertex, and the surfel of a point the eraser brush has painted, or has
under its circle, is drawn in the brush's red or pale red in place of its texels,
as the point's splat is. A surfel whose point takes the highlight (`2`) is
outlined in the hover's cyan. Because the
instances are compacted, a point's slot is not its point index, so the bundle
keeps the point-index-to-slot map the mask write needs.

### The overlay's additions get a second atlas

A node draws its base's atlas and then, when its current version added a
patch-bearing point, a **second, small atlas** of its own, packed by the same
code over a shorter list. Two draws in one pass; nothing in the shader tells them
apart, because each `PatchResources` already carries its own grid dimensions in
its own uniform block.

**The frame writes a block per atlas it draws**, and the two read one list:
`ReconResources::patch_atlases` yields the base's atlas and then the additions',
and both the per-frame uniform write (`scene_renderer/uniforms.rs`) and the patch
pass (`scene_renderer/render.rs`) walk it. They have to agree, because a block
carries the view-projection as well as the grid: an atlas whose block is never
written reads it as zeros, and a zero `view_proj` takes every corner of every
surfel in that atlas to `vec4(0, 0, 0, 0)`. Such an atlas draws nothing at all
rather than drawing wrongly, so the failure looks like a point that has no patch
-- which is what a point committed from the bench is. That list is what
[scene_renderer/upload/tests.rs](../../crates/sfm-explorer/src/scene_renderer/upload/tests.rs)
asserts on, through the draw loop's own filter: a version that added a
patch-bearing point is two atlases with two uniform blocks, and neither is
covered by writing the other's.

**A rebuilt atlas is born with the version's deleted set already in it.** The
frame's mask write is a *difference* -- it walks the indexes that entered or
left the set since the last frame -- so it has nothing to say on a frame that
rebuilt the buffer it would have corrected. The additions' buffers are rebuilt
whenever the addition set moves, and an edit that **creates** a point moves that
set without moving the deleted set: an atlas built all-alive then goes on
drawing the surfel of an addition an earlier edit replaced, at the position that
point used to have, and clicking it does nothing, because the click path drops a
ref the version has deleted. So the liveness buffer is a function of the deleted
set where it is built. The base's atlas passes an empty set, which is not an
exception: a new base arrives with an empty overlay, and the point upload before
it clears the mask the frame compares against, so the version's own set is
written over the top on that same frame.

A second atlas rather than slots appended to the base's: the base's atlas is
exactly what a run of point edits shares, so appending to it would mean building
a new texture per edit and re-uploading every tile. An addition's instance
carries its **edited** index as `point_index`, so one deleted-mask write finds
either atlas's slot with the same lookup. A materialisation replaces both with
one. See [document-model.md](document-model.md), "Change detection by identity".

### The camera model proposal's preview

While a camera model proposal's preview is drawn on the node
([edits/switch-camera-model.md](edits/switch-camera-model.md) § "The
preview"), each surfel buffer has a **surfel preview buffer** beside it
([`upload/preview.rs`](../../crates/sfm-explorer/src/scene_renderer/upload/preview.rs)),
written by slot rather than by point, through `slot_of_point`, the way the
liveness buffer is: one `SurfelPreview` per slot, the target centre with the
flag in a `vec4<f32>` (the flags are the point preview buffer's) and the size
factor in an `f32`. The additions' atlas gets its own, through its own slot
map, keyed by the version's index past the base's rows.

The node's surfels are drawn with the patch pipeline's `vs_preview` variant,
which reads it in a fourth vertex slot. A moving surfel's centre is blended as
its point is, keeping its orientation, and its half-vectors are scaled by
`mix(1, size_factor, s)`; the factor is `patch_frame_factor` for the move,
measured from the switched value's camera centroid taken once, which is the
factor the retriangulation rescales the frame by, so the surfel keeps its
angular size and ends at the size Apply writes. A point at infinity keeps its
size. A surfel whose point loses its position stands where it is and takes
`mix(texel, FAILURE_COLOR, s)`, after the tint and before the eraser brush's
reds. The instance buffer is never written.

### Memory

The atlas is `P × R × R × 4` bytes. At `R = 32`, 100k patches ≈ 400 MB — real,
and larger than the point buffer. v1 uploads every present patch (and logs the
atlas size in MiB) and relies on the fact that patch-bearing reconstructions are
usually far smaller than raw point clouds. If the count exceeds what the GPU's
texture-array limits can hold across all atlas pages, the surplus is dropped
with a `log::warn` — only reachable at extreme counts.
[LOD / streaming / distance culling](#open-questions) is deferred but flagged.

---

## UI controls

Whether a node's points are drawn as patches is its **point style**, the
Dots/Patches selector under the Scene panel's Points row
([scene-graph.md](scene-graph.md#point-style)): under Patches the patch pass
draws each point that has a surfel, and the point pass leaves out that point's
splat, so each point is drawn once, as its patch or as its dot. A node opened
from a file that carries a patch frame **and** bitmaps starts in Patches. There
is no patches toggle; the HUD's and the node's Points switches govern patches
as they govern dots.

A **Patches** section on the [viewport HUD](viewport-hud.md#sections)
(`state.rs` + `viewer_3d/hud.rs`) holds the controls below. The whole section
is *hidden* while no node draws patches.

| Control | Default | Effect |
|---------|---------|--------|
| **Patch opacity** (slider 0–1) | 1.0 | global multiply on patch color alpha; at 0 the patch pass is skipped and the points that have a patch are drawn as dots |
| **Patch size** (`patch_size_log2`, −3…+3) | 0 | scale factor on `u/v` half-vecs, mirroring `point_size_log2`; lets you shrink surfels to see between them or grow them to close the surface |
| **Edge cutoff** (slider 0–1) | 0.0 | `alpha_cutoff` for the coverage discard (0 = fully opaque, nothing discarded) |

A shrunk surfel, or one the edge cutoff carves, leaves no dot behind it: the
point is drawn as its patch, and what the patch does not cover shows what is
behind it.

---

## What gets drawn, by what the reconstruction carries

Patches are drawn only when the reconstruction has **both** the per-point frame
and `patch_bitmaps_y_x_rgba`. That is the textured case specified above, and it
is the only one the renderer handles: `upload_patches` returns early when the
bitmaps are absent, so such a node marks no point as having a surfel and draws
every point as a dot whatever its style, it opens in the Dots style, and its
Points row has no point style selector under it.

So a **frame-only** reconstruction — which is what `sfm xform
--to-embedded-patches` produces before any bitmap render — shows no patches,
exactly like a `sift_files` reconstruction that has no frames either. Drawing
those patches flat-shaded, so their orientation is legible without a texture, is
proposed in
[`../drafts/patch-rendering-flat-shaded-amendment.md`](../drafts/patch-rendering-flat-shaded-amendment.md).

---

## Points at infinity

Handled by the `w`-branch already threaded through the point renderer and the
`OrientedPatch` corner math (`corner_homogeneous`). An infinity patch is a
tangent-sphere quad around the point's direction: transform corners with `w = 0`
(rotation only), bias depth like infinity splats, write `0.0` linear depth (EDL
passthrough). It renders at a fixed angular size and holds position under camera
translation — the same no-parallax guarantee as the infinity point splats and
the background image mesh. No special data handling: `points[i].w` already flags
these rows and `recon.patches` re-marks them on load.

The "Points at ∞" toggle covers these surfels as well as the splats. `patch.wgsl`
reads the same per-node `ReconUniforms::show_infinity` that `points.wgsl` does
and, when it is 0, collapses a `w = 0` surfel to a clipped vertex, so the
surfel neither draws nor writes its point's pick id. Finite surfels are
unaffected.

---

## Picking, hover, selection

Because patches write `PICK_TAG_POINT | point_index`, the existing pick pipeline
(`app.rs::process_pick_readback`) resolves a patch click to its 3D point with **no
new code** — selection highlight, cross-panel hover, and Track View all work through the point they already key on.

The surfel shows its point's highlight itself, as an **outline**: a band
2.5 screen pixels wide just inside the quad's edge, in the selection's yellow
for the selected point, else the hover's cyan for the hovered point or a point
the viewport calls out (mask word `2`). That is the precedence and the colours
the point's splat takes (`SELECTED_COLOR`, `HOVERED_COLOR` in `common.wgsl`),
and the eraser brush's reds come before it as they do there. The point has no
dot while it is highlighted, so it is still drawn once
([scene-graph.md](scene-graph.md#point-style)). The shader reads the two
indexes from `PatchUniforms::selected_point_index` and `hovered_point_index`
and the call-out from the mask word.

- **Width.** 2.5 px is the stroke the bench draws the patch being edited with
  (twice `BENCH_LINE_HALF_WIDTH`), so the two read as the same kind of mark.
  It is constant on screen because it is measured with screen-space
  derivatives, and the band lies inside the quad, so the vertex stage does not
  grow it. A surfel under about 5 px across is all band, a solid square in the
  highlight colour, so a far-away highlighted point still reads.
- **Edge cutoff.** The band is exempt from the coverage discard, so the
  outline traces the whole square at any edge cutoff, while the texels inside
  it are carved as usual.
- **Back faces.** A highlighted surfel is exempt from the front-face cull, so
  a point selected from behind its patch still shows where it is.
- **Opacity.** The band is drawn opaque, whatever `patch_opacity` and the
  Track Table filter's dim alpha, and the texels inside it keep both. At
  opacity 0 the node draws dots, and the dot takes the highlight colour.
- **Depth and picking.** The band writes depth and the point's pick id like the
  rest of the surfel, so an occluded highlighted surfel shows no outline and a
  click on the outline picks the point.
- A point at infinity is outlined the same way: its surfel is never culled, and
  the derivatives measure the band on its quad as on any other.

---

## Open questions

- **EDL on patches.** v1 writes `0.0` linear depth (no EDL) for a clean textured
  look. The alternative — write real positive linear depth so EDL edge-darkens
  patch boundaries — could improve solidity/depth reading of a dense surfel
  field but may fight the texture. Prototype both; the choice may become a
  toggle.
- **Back-face dimming.** Back faces are **culled** in v1 (front-face only — see
  [Blending and face culling](#blending-and-face-culling)). An alternative is to
  dim/tint them rather than hide them; deferred.
- **Transparency order.** Depth-write + alpha-cutoff avoids the problem for hard
  edges. If soft edges matter, weighted-OIT or depth-peeling is a later add.
- **LOD / memory.** `P × R × R × 4` atlas can dominate GPU memory. Distance-based
  culling, a mip'd/downsampled atlas for far patches, or streaming the atlas are
  all deferred; v1 uploads everything and logs the size.
- **Size source.** Half-extents come from `PatchExtent::FeatureSize` at build
  time; the `patch_size_log2` slider only scales them uniformly. Whether the
  viewer should offer alternate extent policies (or re-derive per-view) is out of
  scope.

Known limitations (inherited or minor, not yet addressed):

- **Atlas cell bleed.** Tiles are packed edge-to-edge with no gutter (same as the
  thumbnail atlas), so under linear filtering the outermost half-texel of a
  magnified patch blends with its neighbor's tile (color and alpha). A half-texel
  UV inset or a 1 px gutter would fix it.
- **Infinity patches at the clip plane.** The `w = 0` cull is per corner, so an
  infinity patch straddling the camera's side plane (only reachable at large
  `patch_size_log2`) can smear toward screen center for a frame instead of
  clipping cleanly. A per-instance cull on the center direction would fix it.
