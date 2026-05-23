# Rendering points at infinity in SfM Explorer

**Status:** Draft proposal. The `.sfmr` v2 homogeneous point model and the core
`classify_points_at_infinity` / `materialize_points_at_infinity` conversions are
already implemented; the GUI does not yet render `w = 0` points. This spec
designs that rendering.

Related: [`sfmr-v2-points-at-infinity.md`](sfmr-v2-points-at-infinity.md) (the
format model and classifier), [`gui-point-cloud-rendering.md`](../gui/gui-point-cloud-rendering.md)
(the splat pipeline this extends), [`gui-camera-views.md`](../gui/gui-camera-views.md)
(the `w = 0` direction-transform trick already used for background images).

## Goal

Render `w = 0` points — directions, not locations — in the 3D viewer so that a
distant skyline/ridge/cloud track contributes to the visual reconstruction
instead of being dropped or scattered to arbitrary far coordinates. The
defining property: **a point at infinity has no parallax.** As the camera
translates, a `w = 0` point holds its angular position; only camera *rotation*
moves it. This is exactly the `w = 0` homogeneous transform the background-image
mesh already relies on.

### Non-goals

- No changes to the format, the classifier, or the conversions (all done).
- No new "color by quality" / filtering modes (tracked separately in
  [`gui-point-cloud-rendering.md`](../gui/gui-point-cloud-rendering.md)).
- Materialising infinity points for display is explicitly rejected — placing
  them at a finite far distance reintroduces parallax and is wrong.

## Current state (what we build on)

- `SfmrReconstruction.points: Vec<Point3D>` is **normalised** on read
  (`reconstruction.rs` `From<SfmrData>`): a finite point has `w == 1.0` and a
  Euclidean `position`; an infinity point has `w == 0.0` and a **unit-length
  direction** in `position`. The renderer can rely on `w ∈ {0, 1}`.
- `upload_points` (`scene_renderer/upload.rs`) flattens every point to a
  `PointInstance { position: [f32;3], color: u32 }` (16 B) and draws them as
  instanced camera-facing billboard quads (`shaders/points.wgsl`,
  `pipelines/points.rs`).
- Projection is **reversed-Z with an infinite far plane** (`viewer_3d/camera.rs`
  `projection_matrix`): `z_view = -near → ndc_z = 1`, `z_view = -∞ → ndc_z = 0`,
  depth buffer cleared to `0.0`, `depth_compare = Greater`. A pure direction
  (`w = 0`) is the `z_view → -∞` limit, so it projects to `ndc_z = 0`.
- The fragment shader writes three targets: color, **linear view depth** (drives
  EDL; `0.0` is the "not a point — skip EDL" sentinel used by frustums/quads),
  and **pick id** = `PICK_TAG_POINT | instance_index`. `instance_index` is used
  directly as the global `recon.points` index for selection, hover, and the
  Point Track Detail panel.

## Design

### 1. Carry `w` to the GPU in one buffer

Keep **a single instance buffer for all points** so `instance_index` stays equal
to the global `recon.points` index — picking, hover, and selection highlighting
then work unchanged for infinity points. (Splitting finite/infinity into
separate draw calls would reset `instance_index` per call and break the pick
id → point index mapping.)

The shader needs to know each point's kind. **Pack the flag into the unused
alpha byte of `color`**, keeping `PointInstance` at 16 B (preserves the "10M
points = 160 MB" budget). `color`'s alpha is currently always `255` and the
fragment shader hardcodes output alpha to `1.0`, so the byte is free:

- finite → alpha `255`, infinity → alpha `0`;
- shader: `is_inf = ((color >> 24) & 0xFFu) == 0u`.

For an infinity point, `position` already holds the unit direction (the
normalised in-memory form), so `upload_points` just clears the alpha bit; no
other change.

### 2. Shader: branch on kind

In `points.wgsl` `vs_main`, after unpacking the flag:

- **Finite (today's path):** unchanged. World-space billboard offset, `clip =
  view_proj * vec4(world, 1)`, `view_depth = -(view * world).z`.
- **Infinity:** treat `position` as a direction `d`.
  1. Project the direction: `clip_c = view_proj * vec4(d, 0.0)`. Translation
     drops out — no parallax.
  2. **Behind-camera cull:** if `clip_c.w <= 0` the direction points away from
     the camera; emit a degenerate/off-screen vertex so the splat is discarded
     (mirrors the near-plane handling the frustum line shader does with
     `FrustumUniforms.near`).
  3. Perspective-divide to `ndc = clip_c.xyz / clip_c.w`, then expand the quad in
     **screen space**: `ndc.xy += quad_pos * splat_ndc` where `splat_ndc` is a
     screen-space radius (see §3). World-space `point_size` is meaningless for a
     direction (no distance), so infinity splats are a fixed on-screen size.
  4. **Depth bias to the far plane:** set `ndc_z = INF_EPSILON` (a tiny positive
     value, e.g. `1e-6`). A raw `w = 0` transform gives `ndc_z = 0`, which fails
     `Greater` against the `0.0` depth clear and would make infinity points
     invisible. Biasing just above zero keeps them visible while losing to all
     finite geometry (whose `ndc_z ≫ 0`), so any finite point along a similar
     ray correctly occludes the infinity point. Reconstruct
     `clip_pos = vec4(ndc.xy * clip_c.w, INF_EPSILON * clip_c.w, clip_c.w)`.
  5. **EDL passthrough:** write `view_depth = 0.0` — the existing
     "skip EDL" sentinel. EDL on a near-infinite depth would compute huge
     responses and produce black halos; treating infinity points like the
     frustum/quad layer (composited, unshaded) is both correct and free.

The fragment shader is unchanged: circle clip, selection/hover color override,
`pick_id = PICK_TAG_POINT | index` all apply as-is.

### 3. Uniform additions

`PointUniforms` (and the WGSL `Uniforms`) gain:

- `screen_size: vec2<f32>` — to convert a pixel radius into `splat_ndc`.
- `infinity_point_px: f32` — on-screen splat **radius in pixels** for infinity
  points, driven by a UI slider (see §5). A direction has no distance, so a
  fixed pixel size is the only meaningful sizing and there is no data-derived
  default to fall back on — hence a user control rather than an auto value.
  Default 3 px.

### 4. Data-pipeline fixes (the real trouble spots)

These currently iterate `p.position` over **all** points and silently corrupt
once `position` can be a unit direction near the origin. Each must filter to
finite points (`!p.is_at_infinity()`):

| Site | Bug if unfixed | Fix |
|------|----------------|-----|
| `auto_point_size.rs::compute_scene_bounds` | Unit-direction points cluster at the origin (‖d‖ = 1), dragging the median **scene center** toward origin and distorting the p80 **radius**. This feeds `update_clip_planes`, so near/far go wrong. | Skip `w = 0` points. |
| `auto_point_size.rs::compute_auto_point_size` | KD-tree NN distances polluted by origin-clustered directions → wrong auto splat size. | Skip `w = 0` points. |
| `upload.rs::upload_track_rays` | Uses `points[idx].position` as a 3-D location and projects observations onto it; for a direction this geometry is meaningless. | For an infinity point, draw each observation as a ray from its camera center along the camera's **bearing** (all roughly parallel to `d`), with a fixed display length — there is no finite endpoint to converge on. |
| `point_track_detail.rs` header + `compute_observation_metrics` | Header prints `xyz: (…)` of a direction; reprojection/ray-angle do `R·p + t` then divide by `z`, which is wrong for a direction (translation must drop out). | Branch on `is_at_infinity()`: show **direction** (and a "∞ / at infinity" badge) instead of xyz; project as rotation+intrinsics only (`R·d`), skip the `p_cam.z` depth divide; compute `max pair angle` from the stored direction, not from `position − camera_center`. |
| `app.rs` Alt+click target-set | Infinity points write `view_depth = 0.0`, the same sentinel as frustums/bg; `apply_pick_result` would set the orbit target at a degenerate (near-camera) depth. | Treat a `0.0` depth pick on an infinity point like a background/frustum pick — do not move the orbit target (a point with no location has no pivot). Selection/hover of the point itself still works via the pick id. |

**No visual distinction.** Infinity points render exactly like finite points —
their stored color, the same circular splat. The only unavoidable difference is
that they receive no EDL edge-shading (they have no finite depth; see §2,
step 5), so they read as flat splats rather than a deliberate style choice.

- **Infinity point size** slider — `infinity_point_px`, a pixel radius
  (e.g. range 1–16 px, default 3). Separate from the finite `point_size_log2`
  slider because finite splats are sized in world units and infinity splats in
  pixels.
- **Visibility toggle** "Show points at infinity" (default **on**). Some scenes
  generate many; a quick hide is valuable.
- **Count** in the points readout: `N points (M at infinity)` — `M` is already in
  the metadata as `infinity_point_count`, no recomputation.

### 6. Camera-view interaction (nice property + one ordering check)

In camera-view mode the background image mesh is itself placed by `w = 0`
direction transforms, so an infinity point's splat lands on the **same pixel as
its observed feature** regardless of the viewed camera's translation — they line
up for free. One thing to verify: depth ordering between the background image
(far-plane sentinel) and infinity splats so the splats sit *in front of* the
backdrop. The `INF_EPSILON` bias should put them just ahead; confirm against the
bg-image pipeline's depth handling.

## Testing

There are no checked-in `.sfmr` fixtures; tests regenerate from
`test-data/images/`. To get infinity points to look at:

1. Build a workspace from a dataset (`seoul_bull_sculpture` is small), solve,
   and run `classify_points_at_infinity` (exposed via the PyO3 bindings /
   `xform`) to label distant tracks `w = 0`. A dataset with genuine far content
   (e.g. `seattle_backyard`/`kerry_park`) yields more.
2. Or synthesise a tiny `.sfmr` with a handful of `w = 0` rows for a
   deterministic visual check.
3. Manual checks (per CLAUDE.md, UI changes must be exercised in the running
   app — `pixi run gui -- path.sfmr`):
   - Infinity splats are visible and **do not parallax** when flying/panning,
     but **do** swing with camera rotation.
   - A finite point in front of an infinity point in the same direction occludes
     it; infinity points never occlude finite geometry.
   - Pick/hover/select highlight an infinity point and open its Track Detail
     with the direction + ∞ badge, not bogus xyz.
   - Auto point size and clip planes are unchanged by the presence of infinity
     points (regression: load the same recon with/without classification).
   - In camera view, infinity splats sit on their features over the bg image.

## Open questions

1. **Default infinity point size.** Starting at 3 px; is 1–16 px the right
   slider range?

## File-by-file change list

| File | Change |
|------|--------|
| `crates/sfm-explorer/src/scene_renderer/upload.rs` | Set the infinity flag (alpha byte) per point in `upload_points`; fix `upload_track_rays` for `w = 0`. |
| `crates/sfm-explorer/src/scene_renderer/auto_point_size.rs` | Exclude `w = 0` points from `compute_scene_bounds` and `compute_auto_point_size`. |
| `crates/sfm-explorer/src/shaders/points.wgsl` | Branch `vs_main` on the flag: direction transform, behind-camera cull, screen-space billboard, `INF_EPSILON` depth bias, `view_depth = 0.0`. |
| `crates/sfm-explorer/src/scene_renderer/gpu_types.rs` | Add `screen_size` + `infinity_point_px` to `PointUniforms` (and pad). |
| `crates/sfm-explorer/src/scene_renderer/uniforms.rs` | Populate the new uniform fields. |
| `crates/sfm-explorer/src/point_track_detail.rs` | Infinity-aware header (direction + ∞ badge) and `compute_observation_metrics` (rotation-only projection, direction-based max angle). |
| `crates/sfm-explorer/src/app.rs` | Don't set the orbit target from an infinity-point Alt+click. |
| `crates/sfm-explorer/src/state.rs` + dock/overlay | "Show points at infinity" toggle; `N points (M at infinity)` readout. |
| `specs/gui/gui-point-cloud-rendering.md` | Once implemented, fold this in and move out of `drafts/`. |
