# Patch (Surfel) Rendering in the 3D Viewport

**Status:** Implemented (v1). The SfM Explorer renders **embedded patches** as
small textured, oriented quads ("surfels") in the 3D viewport, one per
reconstruction 3D point that carries a patch frame. See
[Implementation Status](#implementation-status).

The point-splat renderer draws each 3D point as a camera-facing round billboard
(see [gui-point-cloud-rendering.md](gui-point-cloud-rendering.md)). A patch adds
the missing surface information: instead of a view-facing dot, each point is
drawn as a **world-oriented rectangle** textured with the point's rendered patch
bitmap, so the viewport shows the reconstructed *surface* — orientation, texture,
and local appearance — not just a cloud of colored specks.

This is the 3D-viewport counterpart to the existing 2D patch-strip inspectors
(`sfm inspect --strips`, `sfm compare --strips`); it reuses the same
`OrientedPatch` geometry and the same `(P, R, R, 4)` RGBA bitmaps already stored
in the `.sfmr`.

---

## Data source (already present)

Everything this feature needs is already carried on the loaded
`SfmrReconstruction` (`crates/sfmtool-core/src/reconstruction/data.rs:204-213`),
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
- **A point with no patch is an all-zero row** (present iff `u` is non-zero). The
  uploader skips zero rows.
- **`u`/`v` are present-or-absent together.** Bitmaps require the frame but are
  independent of it: a reconstruction can carry frames with **no** bitmaps (e.g.
  a `to-embedded-patches` result before `refine_normals(render_bitmaps=True)`).
  See [Textured vs. flat surfels](#textured-vs-flat-surfels).
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

Given the static unit quad already used by the other instanced pipelines
(`QuadVertex { position: [f32; 2] }` in slot 0, corners at `±1`,
`pipelines/points.rs:124-143`):

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
[gui-point-cloud-rendering.md §Points at Infinity](gui-point-cloud-rendering.md#points-at-infinity)
and `OrientedPatch::corner_homogeneous`, `patch/cloud.rs:100-110`). An infinity
patch is a tangent-to-the-sky quad; it holds angular position under camera
translation, exactly like its point splat.

The same vertex shader also applies the front-face cull (see
[Blending and face culling](#blending-and-face-culling)): a back-facing patch
returns a clipped vertex before corner expansion, so its quad drops out.

### Fragment shader

1. Sample the atlas layer at `uv` → `texel` (RGBA).
2. **Coverage from bitmap alpha.** `texel.a` is per-pixel cross-view confidence,
   not opacity. Treat it as coverage: `discard` when `texel.a < alpha_cutoff`
   so ragged patch edges and unfilled corners can drop out and the quad reads as
   a fitted surfel, not a hard rectangle. **The default is `0.0`** — discard
   nothing, render every texel opaque — which reads better in practice than a
   carved edge; raise the slider to trim ragged borders. Optionally modulate
   final alpha by `texel.a` for a soft edge.
3. Output premultiplied RGBA to the color target (apply the global
   `patch_opacity` uniform).
4. Depth targets — see below.
5. Pick target: `PICK_TAG_POINT | point_index`.

### Render targets — reuse the image-quad convention

The three-MRT + hardware-depth layout is identical to points/image-quads
(`gui-point-cloud-rendering.md §Three Render Targets`). Patches follow the
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
  reversed-Z, +Z-forward camera has no other culled geometry to anchor a winding
  convention to, so a per-instance geometric test is unambiguous. Each patch
  computes `outward = cross(u_halfvec, v_halfvec)` (a positive multiple of the
  normal — no normalize needed) and collapses its quad to a clipped vertex when
  `dot(outward, camera_pos − center) ≤ 0`. A point at infinity is never culled
  (its normal faces every observer). The camera world position is supplied in
  `PatchUniforms.camera_pos`. Dimming/tinting back faces instead of hiding them
  is a possible later refinement.

---

## Texture atlas (mirror the thumbnail atlas)

All patch bitmaps share one resolution `R` (`patch_bitmap_resolution` in
`points3d/metadata.json`), so they pack naturally into a
**`texture_2d_array<f32>`** (`Rgba8UnormSrgb`), exactly like the camera-thumbnail
atlas (`scene_renderer/upload.rs:460-596`). Because `R` is small (typ. 16–64) and
the patch count `P` can exceed the array-layer limit (~2048), reuse the
thumbnail atlas's **page-grid packing** (multiple patches tiled per layer,
`atlas_layer` decodes to `(layer, cell)` — `upload.rs:502-515`), or a plain
one-patch-per-layer array when `P` is small.

Upload path (new `SceneRenderer::upload_patches` in `upload.rs`, gated on the
frame arrays **and** the bitmaps being `Some`):

1. Return early unless both the `(u, v)` frame arrays and `patch_bitmaps_y_x_rgba`
   are present (a frame-only reconstruction renders nothing in v1 — flat-shaded
   fallback deferred). Walk `patch_u_halfvec_xyz` rows and skip zero rows
   (points without a patch).
2. For each present patch, `write_texture` its `(R, R, 4)` tile into the atlas
   and push a `PatchInstance` (center = `points[i].position`, `w = points[i].w`,
   `u/v` from the half-vec arrays, `atlas_layer`, `point_index = i`).
3. Build the bind group (uniform + `texture_2d_array` + sampler), mirroring
   `rebuild_frustum_bind_group` (`upload.rs:422-452`).

Called from `App::prepare_uploads` (`app.rs:224-388`) beside `upload_thumbnails`.

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

New controls in the **View menu**, alongside the point-size / EDL controls
(`state.rs` + `app.rs`), all disabled unless the reconstruction carries a patch
frame **and** bitmaps:

| Control | Default | Effect |
|---------|---------|--------|
| **Show patches** (toggle) | on when patches present | draw the patch pass |
| **Patch opacity** (slider 0–1) | 1.0 | global multiply on patch color alpha |
| **Patch size** (`patch_size_log2`, −3…+3) | 0 | scale factor on `u/v` half-vecs, mirroring `point_size_log2`; lets you shrink surfels to see between them or grow them to close the surface |
| **Edge cutoff** (slider 0–1) | 0.0 | `alpha_cutoff` for the coverage discard (0 = fully opaque, nothing discarded) |
| **Points ↔ patches** | both | independent toggles; a natural default is to **auto-hide point splats where a patch exists** so the surface reads cleanly, keeping splats only for patch-less points — see [open questions](#open-questions) |

Patches occlude points via shared hardware depth, so "show both" already looks
right without hiding points; the auto-hide is a clarity refinement, not a
correctness requirement.

---

## Textured vs. flat surfels

Two cases, by which `Option`s are present:

- **Frame + bitmaps** (`patch_bitmaps_y_x_rgba` present): the primary case —
  textured quads as specified above.
- **Frame only** (no bitmaps): render the same oriented quad **flat-shaded** —
  fill with the point color, optionally Lambert-shaded by the normal against a
  fixed key light — so orientation is still visible without a texture. This is a
  small shader variant (skip the atlas sample); worth including in v1 since
  `to-embedded-patches` produces frames without bitmaps. If descoped, the
  Show-patches toggle simply stays disabled until bitmaps exist.

Reconstructions with **neither** frame nor bitmaps (the common `sift_files`
case) show no patches and disable the control entirely.

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

---

## Picking, hover, selection

Because patches write `PICK_TAG_POINT | point_index`, the existing pick pipeline
(`app.rs::process_pick_readback`) resolves a patch click to its 3D point with **no
new code** — selection highlight, cross-panel hover, and the Point Track Detail
panel all work through the point they already key on. Hover highlight of the
patch itself (e.g. a rim) can reuse the point-splat `selection/hover index`
uniforms if desired, but is not required for v1.

---

## Open questions

- **EDL on patches.** v1 writes `0.0` linear depth (no EDL) for a clean textured
  look. The alternative — write real positive linear depth so EDL edge-darkens
  patch boundaries — could improve solidity/depth reading of a dense surfel
  field but may fight the texture. Prototype both; the choice may become a
  toggle.
- **Point auto-hide.** Should splats be suppressed exactly where a patch exists
  (cleanest surface), always drawn (patch as an overlay accent), or user-chosen?
  Leaning toward auto-hide-with-override.
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

---

## Implementation sketch (v1)

Ordered, each step compiles:

1. `shaders/patch.wgsl` — copy `image_quad.wgsl`; corner expansion from
   `center/u/v` (+ `w` branch); front-face cull (facing test); UV from
   `(s, t)`; alpha-cutoff discard; MRT outputs (linear depth `0.0`, pick
   `PICK_TAG_POINT | point_index`).
2. `scene_renderer/pipelines/patch.rs` — copy `image_quad.rs`; same 3-target MRT
   + `Depth32Float`/`Greater`, `cull_mode: None` (culling is done in the shader).
   Register in `pipelines/mod.rs` and `SceneRenderer::ensure_pipelines`
   (`mod.rs:270-333`).
3. `gpu_types.rs` — add `PatchInstance` (and its vertex-buffer layout) + a
   `PatchUniforms` (view_proj, atlas grid, size/opacity/cutoff, `camera_pos`).
4. `SceneRenderer` fields (`mod.rs:37-158`) + `new()` init — pipeline, instance
   buffer, patch atlas texture/view/sampler, bind group, patch count.
5. `scene_renderer/upload.rs::upload_patches` — mirror `upload_thumbnails`; build
   atlas + instances from the three reconstruction arrays; skip when `None`.
6. `scene_renderer/render.rs` — draw block in Pass 1 near the image-quad draw
   (`render.rs:270-283`), gated on `show_patches`.
7. `app.rs::prepare_uploads` — call `upload_patches`; add `show_patches`,
   `patch_opacity`, `patch_size_log2`, `alpha_cutoff` to state + uniforms.
8. UI — controls in the point-cloud panel section; disable when no patch frame.
9. (optional) flat-shaded variant for frame-without-bitmap reconstructions.

Picking, hover, and selection require **no** new work — they ride the point
pick tag.

---

## Implementation Status

### Implemented

- [x] Patch instance + atlas upload from `recon.patch_*` arrays
      (`scene_renderer/upload.rs::upload_patches`, page-grid packed
      `texture_2d_array`, atlas byte size logged)
- [x] `patch.wgsl` textured oriented-quad pipeline in Pass 1
      (`pipelines/patch.rs`, premultiplied alpha)
- [x] Front-face-only culling via a vertex-shader facing test
      (`dot(u × v, camera_pos − center) ≤ 0`; infinity patches always render)
- [x] Bitmap `v` texture coordinate flipped in the vertex shader (rows are baked
      along `−v`, so `uv.y = (1 − t) / 2`) — keeps the textured quad un-mirrored
- [x] Real HW depth / `0.0` linear depth / `PICK_TAG_POINT` targets
- [x] Points-at-infinity `w = 0` corner path (direction transform +
      `INF_DEPTH` bias, matching the infinity point splats)
- [x] Show-patches / opacity / size / edge-cutoff UI (View menu, alongside
      the point-size controls; disabled unless frames **and** bitmaps exist)

### Planned (v1)

- [ ] Flat-shaded fallback for frame-without-bitmap patches (descoped from
      the first cut: `upload_patches` returns early without bitmaps and the
      Show-patches controls stay disabled until bitmaps exist)

### Future

- [ ] EDL-on-patches toggle
- [ ] Point auto-hide where patches exist
- [ ] Back-face dim/tint (culling is done — this is the softer alternative)
- [ ] LOD / atlas streaming / distance culling for large patch clouds
- [ ] Order-independent transparency for soft patch edges
