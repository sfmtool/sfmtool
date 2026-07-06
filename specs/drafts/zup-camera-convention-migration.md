# Z-Up / −Z-Forward Coordinate Convention — Migration Plan

**Status**: **IMPLEMENTED** (2026-07-04, branch `zup-core-flip`). All steps 1–6
complete and reviewed; the tree is green in both languages and end-to-end
acceptance passes (see the §12 step status notes). The only open item is the
human-in-the-loop GUI visual confirmation (§12 step 4/6). Originally authored
as a draft roadmap; the per-step status notes below record what landed.
**Date**: 2026-07-03 (authored); 2026-07-04 (implemented). **Branch**: `zup-core-flip`
(cut from `sfmr-z-up`).
**Companion spec change**: `specs/formats/sfmr-file-format.md` § "Coordinate
System Conventions" (added in the same change as this document).

This document records every code change required to make sfmtool follow the
canonical `.sfmr` coordinate convention end-to-end:

1. **World space** is right-handed, **Z-up**; the X-Y plane is the ground plane.
2. **All cameras look down −Z**, with image-plane **+X right and +Y up**
   (OpenGL-style), instead of COLMAP/OpenCV's +Z-forward / Y-down.

Today, COLMAP/pycolmap solutions are copied into `.sfmr` **verbatim**, so the
entire codebase currently operates in COLMAP convention. The migration flips
the in-memory/on-disk convention and pushes all conversion to the I/O
boundaries.

---

## 1. The conversion math

Fixed matrices (both are proper rotations, det = +1, involutive `S·S = I`):

```
S = diag(1, −1, −1)                # camera-frame flip: 180° about camera X
W = [[1, 0, 0],                    # world canonicalization: (x, y, z) → (x, z, −y)
     [0, 0, 1],                    # maps COLMAP's typical −Y-up worlds to +Z-up;
     [0, −1, 0]]                   # identical to Nerfstudio's applied_transform
```

**COLMAP → canonical** (import), for world-to-camera poses `(R, t)`:

```
R' = S · R · Wᵀ          t' = S · t
X' = W · X               # finite point xyz (w carried through unchanged),
                         # infinity directions, normals, patch u/v half-vectors
sensor_from_rig:  R' = S · R · S        t' = S · t     # rig-relative poses: W cancels
cam2_from_cam1 (relative pose):  R' = S · R · S        t' = S · t
```

**Canonical → COLMAP** (export) is the inverse: `R = Sᵀ·R'·W = S·R'·W`,
`t = S·t'`, `X = Wᵀ·X'`.

Useful invariants (why some things do *not* change):

- **Pixel-space epipolar geometry is invariant.** The fundamental matrix `F`
  relates *pixels*, and pixels don't change. Equivalently, since `S` is a
  rotation, `[S·t]× = S·[t]×·Sᵀ`, so
  `E' = [S t]× (S R S) = S·E·S` and camera rays flip by the same `S`; the
  constraint `r₂ᵀ E r₁ = 0` is preserved. Consequence: stored F/E/H matrices
  in `.matches` files and COLMAP DBs are unaffected; only code that *derives*
  `E`/`F` from stored poses plus `K` matrices must first map poses back to
  OpenCV camera frames (or equivalently conjugate `E` by `S`).
- **World-space geometry that never touches a camera axis is invariant**:
  camera centers `C = −Rᵀt` (formula unchanged), ray-toward-point
  computations (`X − C`), triangulation from world rays, Kabsch/similarity
  alignment, kd-trees, patch `u × v` normals. These transport correctly under
  the `W` rotation with no code change.
- **Internal round trips need only `S`.** When a pipeline exports to
  pycolmap/COLMAP and re-imports its own output within one operation (bundle
  adjust, densify triangulation, merge PnP, DB-mediated solves against
  DB-exported priors), applying `S` on the camera frames both ways — leaving
  the world frame untouched — is self-consistent and avoids double-rotating
  the world. `W` is reserved for *external* import/export (fresh solver
  output, `from-colmap-bin`/`to-colmap-bin`/`to-colmap-db`), applied on
  import and inverted on export so round trips through external files are
  stable.

---

## 2. Recommended design decisions

These are the choices the rest of the plan assumes. Each is revisitable, but
pick before coding — they determine signatures.

- **D1 — Format version 5, upgrade on load.** Bump `.sfmr` to version 5 (a
  purely semantic bump; no arrays change). Readers apply the COLMAP→canonical
  conversion (`S`, `W`) to any version ≤ 4 file at load; `save` always writes
  v5. This keeps every existing `.sfmr` readable. (Alternative rejected: a
  per-file convention flag — perpetuates mixed conventions internally.)
- **D2 — One conversion implementation, in Rust core.** Add a
  `convention` module to `crates/sfmtool-core` (e.g.
  `src/geometry/convention.rs`) holding `S`, `W`, and functions:
  `pose_colmap_to_canonical`, `pose_canonical_to_colmap` (quat + translation),
  `relative_pose_conjugate_s`, `world_rotate_w` (points/directions/normals/
  patch frames), plus whole-`SfmrReconstruction` converters. Expose through
  `sfmtool-py`; add a thin Python helper module
  (suggest `src/sfmtool/colmap/convention.py`) wrapping the bindings for
  `pycolmap.Rigid3d` and raw numpy pose arrays so every Python boundary site
  calls the same math.
- **D3 — `S` is unconditional; `W` only at external boundaries.** See §1
  invariants. `from-colmap-bin` and the solve pipelines apply `S` + `W`;
  `to-colmap-bin` / `to-colmap-db` apply `S` + `W⁻¹`; in-pipeline pycolmap
  round trips use `S` only.
- **D4 — `rig_config.json` stays in COLMAP convention.** It mirrors COLMAP's
  own rig-config schema and exists to feed COLMAP DB setup; treat it as a
  COLMAP-side artifact and convert at ingestion like every other COLMAP
  input. Benefit: `test-data/images/kerry_park/rig_config.json` and user
  configs remain valid. Document this in `specs/workspace/rig-config.md`.
- **D5 — `.camrig` adopts the canonical convention.** It is our own format
  ("matches `.sfmr`"), so its `sensor_from_rig` poses flip with `.sfmr`
  (S-conjugation). Regenerate `test-data/images/kerry_park/kerry_park.camrig`
  and update `specs/formats/camrig-file-format.md`. Decide whether `.camrig`
  gets its own version bump/on-load upgrade (recommended: yes, mirroring D1).
- **D6 — `.matches` two-view poses adopt the canonical convention.** The
  `two_view_geometries/quaternions_wxyz` / `translations_xyz` relative poses
  (`cam2_from_cam1`) S-conjugate; the stored F/E/H matrices are pixel-space
  and unchanged (§1). Update `specs/formats/matches-file-format.md` and
  convert at the DB-export consumer.
- **D7 — Camera-model kernels keep an internal "optical frame".** Rather than
  rewriting every distortion kernel, apply `S` once at the camera-model
  boundary: `project`/`distort_ray` map the canonical camera-space input
  through `S` into the legacy y-down/+Z optical frame the kernels use;
  `unproject`/`undistort_to_ray`/`pixel_to_ray` map kernel output back
  through `S` (rays become `(x_n, −y_n, −1)`-style, unit-normalized).
  Cheirality gates flip from `z > 0` to `z < 0`; "depth" becomes `−z`.

**Atomicity.** The convention is global: core, format, boundaries, Python,
GUI, and tests must flip together. Phases in §10 are a *build order within
one coordinated change* (the tree can compile at each step, but the full test
suite is only green at the end), not independently shippable increments.

---

## 3. Already done on this branch (prerequisites)

Commits ahead of `main` on `sfmr-z-up`:

- `1011a0f` — fix(patch): image-raster frame handedness so patch renders are
  not mirrored (#159).
- `3a5eb3c` — fix(patch): store right-handed frame, reverse v in the raster.

Net effect: the stored per-point patch frame is genuinely **right-handed in
world space** (`u × v` = outward normal; `u = v × n`), and the image raster
steps rows along `−v` to reconcile with y-down pixel rows. This matters here
because a right-handed world-space frame **transports correctly under the `W`
rotation** — patch `u`/`v` half-vectors just rotate with the world and the
GUI cull / renderers stay correct with no re-signing. The raster `−v`
reversal is an image-space (pixel rows grow down) property and **survives the
camera-axis flip unchanged** — guarded by the source→grid Jacobian-determinant
regression test added in `1011a0f`.

Nothing of the camera −Z-forward flip or the Z-up world is implemented yet.

---

## 4. Work area A — Rust core camera-space flip (`crates/sfmtool-core`)

The heart of the change. All paths below are relative to
`C:\Dev\prod\sfmtool\crates\sfmtool-core\src\`.

### A.1 Camera models — `camera/distortion.rs` (per D7)

- [ ] Module doc (~L11–35): rewrite the convention description (image-plane
  coords, ray convention) for −Z forward / +Y up camera space with y-down
  pixels.
- [ ] `CameraModel::distort_ray` (~L417–524): gate `if rz <= 0.0` (~L435)
  becomes "reject `rz >= 0.0`"; map input ray through `S` before the
  perspective/fisheye kernels. Equirectangular branch (~L421–426):
  `longitude = rx.atan2(−rz)`, latitude sign re-derived (camera +Y up, pixel
  v down).
- [ ] `CameraModel::undistort_to_ray` (~L535–632): perspective rays
  `[x/len, y/len, 1/len]` (~L556–559) become `[x/len, −y/len, −1/len]`
  (i.e. `S ·` old ray); degenerate fallbacks `[0,0,1]` (~L595, 622) become
  `[0,0,−1]`; equirect branch (~L538–547) re-derived.
- [ ] `CameraIntrinsics::project` / `unproject` (~L644–663) and
  `pixel_to_ray` (~L700), `ray_to_pixel`: doc + boundary flip.
- [ ] `project_ray_node` / grid projection paths (~L759–969): inherit the
  gate change; verify NaN-out of behind-camera nodes.
- [ ] `best_fit_inside_pinhole` / `best_fit_outside_pinhole` (~L1002–1143):
  ride on `project`/`unproject`; verify only.
- Kernels in `camera/distortion/kernels.rs` are untouched (they receive
  optical-frame values after the `S` boundary).

### A.2 Frustums — `camera/frustum.rs`

- [ ] `compute_frustum_corners` (~L22–83): ray `Vector3::new(x_norm, y_norm,
  1.0)` (~L61) → canonical ray (via `pixel_to_ray` or `(x_norm, −y_norm,
  −1)`); plane intersection `t = near_z / dir[2]` (~L63–64) → `t = −near_z /
  dir[2]` (keep `near_z`/`far_z` as positive *depths*).
- [ ] `compute_distorted_frustum_grid` (~L382–440): same at ~L424.
- [ ] `compute_frustum_volume` (~L151–172): unchanged math (widths from
  positive depths); verify docs.
- `compute_frustum_planes` / `points_in_frustum` / `frustums_can_intersect`
  (~L95–266): world-space, no change.

### A.3 Epipolar & rectification — `camera/epipolar.rs`, `camera/rectification.rs`

- [ ] `plot_epipolar_curve` (`epipolar.rs` ~L153–230): cheirality `xc.z <=
  0.0` (~L178) flips to `>= 0.0`; back-projection rides on `pixel_to_ray`.
- [ ] `compute_stereo_rectification` (`rectification.rs` ~L104–176): the
  rectified basis is built around the optical axis (`raw_e2 = [0,0,1] × e1`,
  ~L128–136); re-derive for −Z optical axis (or convert to the optical frame
  at entry, per D7).
- `compute_fundamental_matrix`, `compute_epipole*` — pure linear algebra over
  supplied poses; the *callers* must supply OpenCV-frame relative poses or the
  S-conjugated equivalents consistently (see §1 invariant). Audit callers.

> _Status (2026-07-04, step 2): chose **caller-converts** for the whole
> pixel-space pipeline — `compute_fundamental_matrix`, `check_rectification_safe`,
> `compute_stereo_rectification`, `rectify_points`, and
> `features/feature_match/mod.rs::match_image_pair` keep their math unchanged and
> now **document an OpenCV/optical-frame pose contract** (callers S-premultiply /
> S-conjugate first, via `geometry::convention`). Rationale: F → epipoles →
> rectification homographies → sweep/polar matching is one pixel-space K-matrix
> system; converting inside `compute_stereo_rectification` only would split one
> pipeline across conventions and double-convert `match_image_pair`. Consistent
> with §1 and C.3 (Python callers convert in step 5). **Confirm before wiring the
> Python side (step 5).**_

### A.4 Reconstruction data — `reconstruction/data.rs`

- [ ] `observation_reprojection_error` (~L1348–1368): `p_cam.z <= 0` gate →
  `>= 0`; projection divide switches to `−z` (inside `project` per D7).
- [ ] Patch anchor / `project_point`-style projector (~L330–385): rides on
  `pixel_to_ray`/`ray_to_pixel`; verify.
- [ ] Synthetic look-at builder (~L1158–1174, used by `demo()` fixtures):
  currently documents/builds COLMAP rows `[right, −up, forward]`; rebuild as
  canonical rows `[right, up, −forward]` and update comments.
- [ ] `recompute_depth_statistics` (~L700–751): delegates to `sfmr-format`
  (see area C.3).

### A.5 Patch / surfel — `patch/`

- [ ] `view_selection.rs` `is_in_front` (~L262–267): `.z > 0.0` → `.z < 0.0`.
- [ ] `keypoint_localize.rs` `project` (~L248–256): `pc.z <= 0.0` → `>= 0.0`
  (or delegate to the shared camera-model boundary).
- [ ] `cloud.rs` `first_view_up` (~L647–654): image-up in camera space
  `(0, −1, 0)` → `(0, 1, 0)` (camera +Y is now up).
- [ ] `cloud.rs` `PatchExtent::PixelRadius` (~L419): `p_cam.z.abs()` —
  sign-agnostic, verify only.
- `from_center_normal`, `from_infinity_direction` (normal = `−d`),
  `is_front_facing`, `mean_viewing_normal`, `warp_map.rs` `from_patch`
  raster `−v` stepping (~L319), `keypoint_subpixel`, `normal_refine` — all
  world-space or already image-raster-correct; **no change**, verified by the
  Jacobian regression test and existing photometric tests (§3).

> _Status (2026-07-04, step 2): plan gap — `normal_refine/fronto_cache.rs`
> `corner_norm_pts` had a real camera-frame cheirality gate (`p.z <= 1e-9`)
> despite this section declaring `normal_refine` unchanged; without flipping it
> (now rejects `p.z >= −1e-9`) every fronto base is rejected and the cached
> normal-refine path silently dies. The `(x/z, y/z)` divide is left as-is: it
> differs from true normalized coords by a fixed mirror that provably cancels in
> the composed `a0_inv · ap` same-view grid→grid affine (comment added in code)._

### A.6 Analysis — `analysis/`

- [ ] `image_pair_graph.rs` `compute_camera_directions` (~L32–69): code uses
  `dir = −column(2)` of `R_world_from_cam` with a "camera looks down −Z"
  comment — under today's COLMAP data that is the *negated* viewing
  direction, which self-cancels in pairwise angle comparisons. After the
  flip, `−column(2)` becomes genuinely correct: **keep the code, fix the
  narrative**, and add a direction-sign unit test so it can't silently
  regress again.
- [ ] `analysis/infinity/{convert,discover}.rs`: world-space direction math
  (stored direction points from cameras toward content) — no change, but
  rides on `pixel_to_ray`; verify classification tests still pass.
- `geometry/viewing_angle.rs`, `spatial.rs`: world-space; no change.

> _Status (2026-07-04, step 2): plan gap — `analysis/point_inspect.rs`
> incidence angle was `acos(ray.z)`; flipped to `acos(−ray.z)` (angle off the
> −Z optical axis). Not enumerated above but a baked camera-axis sign. The A.6
> direction-sign lock test (`test_camera_direction_sign_matches_look_at_forward`)
> was added as planned._

### A.7 Spherical tiles — `spherical/tile_rig.rs`

- [ ] Tile frames are equirect/pinhole camera frames: flip the cheirality
  gates `tz <= 1e-9` (~L806, ~L926) and re-derive the tile forward axis for
  −Z-forward tile cameras; `sphere_points.rs` / `per_tile_source_stack.rs`
  ride along.

> _Status (2026-07-04, step 2): implemented via a new
> `SphericalTileRig::tile_camera_rotation(idx)` = columns
> `[e_right | −e_up | −direction]` (content-preserving: tile raster stays
> bit-identical to pre-flip); `warp_to_atlas_with_rotation`, `resample_atlas`
> (gate rejects `tz >= −1e-9`), `warp_from_atlas_with_rotation`, and
> `per_tile_source_stack.rs` use it. The stored tangent bases, `build_basis`,
> `tile_rotation()`, and the `.camrig` round trip are **left byte-stable** —
> the `.camrig` flip is D5/step 3. Note: `build_basis`'s world-up reference is
> still ±Y, so under the Z-up world tiles are internally rolled vs. gravity
> (harmless, write/read-symmetric) — **revisit the up reference in step 3
> alongside D5.**_

### A.8 Convention-agnostic core (verify, don't touch)

`geometry/se3_transform.rs` (`apply_to_camera_pose` uses `C = −Rᵀt` —
convention-free), `rot_quaternion.rs::camera_center`,
`rigid_transform.rs` (`inverse_translation_origin`,
`transform_point_homogeneous`), `reconstruction/triangulation.rs`
(world-ray midpoint solve; cheirality = `(X − C)·dir > 0` on world rays),
`features/optical_flow/**` (pure 2D), F-matrix algebra in `epipolar.rs`.

---

## 5. Work area B — Format layer (`crates/sfmr-format`, version 5)

- [ ] `crates/sfmr-format/src/types.rs`: bump the current format version
  constant to 5. No new fields (D1).
- [ ] Reader: on load of version ≤ 4, apply COLMAP→canonical (`S` on poses
  and rig sensor poses; `W` on points, infinity directions, normals, patch
  `u`/`v` half-vectors). Content hashes are of the *stored* bytes, so verify
  hashes before converting; a converted-then-saved file is a new v5 file with
  new hashes.
- [ ] Writer: always writes v5.
- [ ] `crates/sfmr-format/src/depth_stats.rs`:
  `compute_depth_statistics` (~L204) and its per-image helper (~L112–135)
  compute depth as camera-space `z = R[2]·(X − C)`; change to `depth = −z`
  (row-2 dot, negated). Histogram/percentile logic unchanged (depths stay
  positive-in-front).
- [ ] `crates/matches-format`: per D6, S-conjugate stored relative poses on
  load of old files if the format is versioned; otherwise document the break
  and regenerate. F/E/H matrices unchanged.
- [ ] `crates/camrig-format`: per D5, flip `sensor_from_rig` convention;
  update the rig builder helpers (`shortest_arc_from_z` tile orientation,
  insv2 back-to-back, cubemap faces, stereo baseline) to the −Z-forward
  sensor axis; version/upgrade handling mirroring D1 if feasible.

---

## 6. Work area C — COLMAP I/O boundary

The Rust `sfmr-colmap` readers/writers (`colmap_io/read.rs`,
`colmap_io/write.rs`, `colmap_db/write.rs`) stay **byte-verbatim** — they
model COLMAP-side data. The conversion applies where COLMAP-side data meets
`SfmrReconstruction` / numpy arrays, which today is Python:

### C.1 Primary converters — `src/sfmtool/colmap/io.py`

- [ ] `colmap_binary_to_rust_sfmr` (~L269): apply `S` + `W` to quats,
  translations, and points before `SfmrReconstruction.from_data` (~L301–321).
  Add an `apply_world_rotation=True` parameter so in-pipeline callers can
  request `S`-only (D3).
- [ ] `pycolmap_to_rust_sfmr` (~L327): same, at the `cam_from_world()`
  extraction (~L371–375) and point copy (~L399).
- [ ] `_extract_rig_frame_data` (~L450): S-conjugate `sensor_from_rig`
  (~L508–511).
- [ ] `save_colmap_binary` (~L578): inverse conversion (`S`, `W⁻¹`; flag for
  `S`-only) on quats/translations (~L642–667) and points.

### C.2 Solve pipeline

- [ ] `src/sfmtool/_incremental_sfm.py` `_save_reconstructions` (~L135,
  branches at ~L245/253): both paths go through the C.1 converters with
  `W` applied (fresh external solve).
- [ ] `src/sfmtool/_global_sfm.py` `run_global_sfm` (~L19, reuses
  `_save_reconstructions` ~L114): same.
- Note: these solves start from DBs we exported (C.4). With D3 (`W` at
  export-to-DB inverse + `W` at import), priors and re-imported poses stay
  consistent.

### C.3 In-pipeline pycolmap round trips (S-only, per D3)

- [ ] `src/sfmtool/xform/_bundle_adjust.py` `BundleAdjustTransform.apply`
  (~L28) / `_reconstruction_to_data` (~L71, quat reorder ~L101–109): convert
  out and back with `S`-only helpers.
- [ ] `src/sfmtool/_densify.py`:
  - `_match_single_pair` (~L90, Rigid3d built ~L130–138) — poses to OpenCV
    frames for pycolmap epipolar guidance.
  - `triangulate_new_tracks` (~L199; relative pose ~L278–300) — `S`-conjugate
    relative poses; export/import via C.1 flags.
  - `_align_to_original` (~L467; centers ~L495–508, Sim3d ~L547–553) —
    verify: operates on camera centers (invariant) but builds the Sim3d in
    pycolmap space; keep both sides in one convention.
  - final `pycolmap_to_rust_sfmr` (~L742) — `S`-only variant.
- [ ] `src/sfmtool/merge/pose_refinement.py` `_refine_single_camera_pose`
  (~L41; PnP ~L114; store ~L119–137): points fed to pycolmap and returned
  poses must both pass through the `S` helpers.
- [ ] `src/sfmtool/feature_match/_core.py` (~L157–161) and the sweep modules
  (`_rectified_sweep.py` ~L38–39, `_polar_sweep.py` ~L99–100): `.sfmr` poses
  → `pycolmap.Rigid3d` for epipolar-guided matching — convert to OpenCV
  frames.

### C.4 COLMAP database export/import

- [ ] `src/sfmtool/colmap/db_export.py`
  `create_colmap_db_from_reconstruction` (~L80): pose priors are world
  camera centers (~L168–182) — apply `W⁻¹` so the DB world matches what the
  solver import will canonicalize back; `_compute_two_view_geometry` (~L21,
  Rigid3d ~L43–77): relative poses via `S`-conjugation (resulting pixel-space
  E/F are then identical to today, per §1).
- [ ] `src/sfmtool/colmap/db_setup.py`: matches-file two-view geometry import
  (~L474–489) — `.matches` poses are canonical after D6; S-conjugate when
  building `pycolmap.Rigid3d`. `_rigid3d_sensor_from_rig` (~L283) +
  `_setup_db_with_camrig` (~L297): `.camrig` is canonical after D5 —
  S-conjugate to COLMAP for the DB.
- [ ] `src/sfmtool/rig/config.py` `_sensor_from_rig_pose` (~L44–63): per D4
  `rig_config.json` stays COLMAP-convention and this loader feeds COLMAP DB
  setup — **no math change**, but document the convention at the loader and
  in `specs/workspace/rig-config.md`. If any consumer starts using these
  poses on the `.sfmr` side, it must convert.

### C.5 CLI shims (no math of their own — verify only)

`src/sfmtool/_commands/to_colmap_bin.py`, `from_colmap_bin.py`,
`to_colmap_db.py` delegate to C.1/C.4. Add `--help`/spec notes that these
are convention boundaries.

### C.6 Undistort

`src/sfmtool/_undistort_images.py` copies poses/points verbatim
(`.sfmr`→`.sfmr`, ~L552–554) and only warps pixels through
`unproject_batch`/`project_batch` — **no boundary conversion needed**; it
rides on the camera-model flip (A.1). Verify with its tests.

---

## 7. Work area D — Other convention boundaries

### D.1 Nerfstudio exporter — `src/sfmtool/_to_nerfstudio.py`

Input assumption inverts: the `.sfmr` is already OpenGL-camera / Z-up-world.

- [ ] `frame_transform_matrix` (~L37–63): delete the two column negations
  (~L60–61) — `world_from_cam` is already OpenGL — and drop the
  `_APPLIED_TRANSFORM_3x4` pre-rotation (~L25–32): the world is already Z-up,
  so `applied_transform` becomes identity (still write the field, as
  identity, for Nerfstudio compatibility).
- [ ] `apply_transform_to_points` (~L66–69): becomes identity / is removed.
- [ ] `--include-colmap` path (~L263–267): rides on the converted
  `save_colmap_binary`.
- [ ] Update `specs/cli/to-nerfstudio-command.md` § "Coordinate Conversion"
  and `tests/test_to_nerfstudio.py` expectations.

### D.2 Pano2rig — `src/sfmtool/rig/pano2rig.py`

- [ ] `_cubemap_rotations` (~L44–59): re-derive the six `rig_from_sensor`
  rotations for −Z-forward sensors (front = identity means "looks along rig
  −Z"; right/back/left rotate about the rig up axis, top/bottom about rig X).
- [ ] `extract_perspective_face` (~L62–116): already builds a **y-up ray
  frame** internally (`rays = [uu/f, −vv/f, 1]`, ~L86); re-express with a −Z
  forward axis so the internal frame *is* the canonical camera frame (the
  equirect lon/lat mapping ~L96–103 keeps pano-up = camera +Y).
- [ ] `write_pano_camrig` (~L207–262): delete the y-flip conjugation to
  COLMAP (`[q.w, −q.x, q.y, −q.z]`, ~L247–249) — with D5 the `.camrig`
  stores canonical sensor poses directly.
- [ ] Update `specs/cli/pano2rig-command.md` and `tests/test_pano2rig.py`.

### D.3 Insv2rig — `src/sfmtool/_commands/insv2rig.py`

- [ ] X5 rig constants (~L16–47): S-conjugate the sensor poses. Note
  `S·Ry(180°)·S = Ry(180°)` — the rotation is **unchanged**; the baseline
  translation flips sign: `[0, 0, −0.0307]` → `[0, 0, +0.0307]` (the rear
  lens sits along +Z when forward is −Z). Rewrite the derivation comments
  (~L26–30) and update `tests/test_fisheye_rig.py`.
- `src/sfmtool/rig/insv2rig.py` `write_insv_camrig` (~L223–277) is a
  passthrough; update its docstring ("COLMAP's WXYZ convention" → canonical).

### D.4 Test data regeneration

- [ ] `test-data/images/kerry_park/kerry_park.camrig` — regenerate in the
  canonical convention (D5).
- [ ] `test-data/images/kerry_park/rig_config.json` — **unchanged** (D4).
- `scripts/init_dataset_*.sh` download nothing convention-dependent; local
  workspaces (`*.sfmr` artifacts) upgrade on load via D1.

---

## 8. Work area E — Python internal consumers

Mostly ride on the Rust flip; the sites below have their own baked signs.

- [ ] `src/sfmtool/visualization/_patch_renderer.py` `_render_image`
  (~L110–249): `depth = cam_pts[..., 2]` and `depth > 0` cheirality (~L160–167)
  → `depth = −cam_pts[..., 2]`; back-face test (~L169–171) and painter's sort
  (~L185) then keep their comparisons against the new positive depth.
- [ ] `src/sfmtool/visualization/_epipolar_display.py` `_curve_anchor_depths`
  (~L77–110): `track_depths = points @ R_from[2, :] + t_from[2]`,
  `in_front = depths > 0` (~L105–106) → negate. `_compute_fundamental_matrix`
  (~L21–40) and the `R_rel/t_rel` recomputations: route through the D2 helper
  to OpenCV frames before `K`-based F construction (§1 invariant).
- [ ] `src/sfmtool/feature_match/_geometry.py` (`get_essential_matrix`
  ~L18–43, `get_fundamental_matrix` ~L46–68): same helper routing.
- [ ] `src/sfmtool/_image_pair_graph.py` `compute_camera_directions` (~L33):
  `−R_world_from_cam[:, 2]` — same story as A.6: correct *after* the flip;
  update comments and add a sign test. (Feeds `analyze/images.py`
  `compute_view_angle` ~L349–353.)
- [ ] `src/sfmtool/_solve_strips.py` (~L262, ~L409): image-up in camera space
  `rot.T @ [0, −1, 0]` → `rot.T @ [0, 1, 0]`.
- [ ] `src/sfmtool/_embed_patches.py`: `_camera_centers` (~L437) and
  `_drop_grazing_observations` (~L441–491) are world-space — verify only.
- [ ] `src/sfmtool/analyze/depth.py`, `analyze/graphs.py`,
  `_histogram_utils.py`, `analyze/summary.py`: display Rust depth stats;
  semantics unchanged (depth stays positive-in-front) — wording/docs only.
- [ ] `src/sfmtool/_rectification.py` (~L78–136): feeds `R_rel`/`t_rel` to
  `cv2.stereoRectify`, which expects OpenCV frames — route through the D2
  helper.
- Convention-agnostic (verify only): `align/*`, `merge/*` (apart from C.3),
  `motion/*`, `visualization/_heatmap_renderer.py`, `camera/*` (EXIF
  orientation deliberately unused — `camera/setup.py` ~L208), `xform`
  rotate/translate/similarity/align, `xform/_find_points_at_infinity.py`,
  `xform/_refine_normals.py`, `xform/_to_embedded_patches.py`.

---

## 9. Work area F — GUI viewer (`crates/sfm-explorer`)

The viewport is **already** the target convention (world Z-up right-handed,
OpenGL camera): `viewer_3d/camera.rs` `world_up = Vector3::z()` (~L36–49),
Home-key reset (`input.rs` ~L462–466), ground grid on the Z=0 XY plane
(`viewer_3d/overlay.rs` `draw_grid` ~L18–73), reversed-Z GL projection
(`camera.rs` ~L356–380), and no axis flips in `scene_renderer/uniforms.rs`.
The format change *fixes* today's sideways-looking scenes rather than
breaking the GUI. Remaining edits:

- [ ] `viewer_3d/mod.rs` `enter_camera_view` (~L464–522): **remove the
  COLMAP→GL bridge** — `flip = from_axis_angle(x_axis, π)` and
  `end_orientation = flip * image.quaternion_wxyz` (~L476–488) become
  `end_orientation = image.quaternion_wxyz`. The `end_world_up` derivation
  (`orientation⁻¹ · (0, 1, 0)`) is already correct for +Y-up camera frames.
  `compute_switch_camera_view` (~L527–564) is purely relative and follows.
- [ ] Frustum gizmos and the camera-view background mesh
  (`scene_renderer/upload.rs` `upload_frustums` ~L100–309;
  `scene_renderer/distorted_mesh.rs` ~L23–71) ride on the core
  `compute_frustum_corners` / `pixel_to_ray` changes (A.1, A.2) — no GUI
  edits, but verify frusta and background images render un-mirrored and
  right-side-up.
- [ ] `shaders/patch.wgsl` cull `cross(u_halfvec, v_halfvec)` (~L71): world
  frames rotate rigidly under `W` — **no change** (see §3); verify.
- [ ] Specs: `specs/gui/gui-camera-views.md` (its "Convention Bridge" section
  and "COLMAP convention" labels, e.g. ~L29, ~L72–104) must be rewritten;
  navigation/grid specs already state Z-up and need no change.

---

## 10. Work area G — Tests and fixtures

No checked-in `.sfmr`/`.bin`/`.db` golden files exist; Python E2E tests
re-solve from raw images and assert statistics, so they are largely
convention-robust. The exposure is synthetic-geometry unit tests plus the two
kerry_park rig artifacts (D.4).

**Rust tests to update (hardcoded +Z / Y-down expectations):**

- [ ] `camera/frustum/tests.rs` — identity camera "looking along +Z",
  corners/planes at +z.
- [ ] `camera/distortion/tests.rs` — `ray == (0,0,1)` at principal point,
  ±z in-front assertions, `ray_to_pixel([0,0,−1]) → None`.
- [ ] `camera/epipolar/tests.rs`, `camera/rectification/tests.rs` — forward
  motion `t = (0,0,1)`, points at `(0,0,10)`.
- [ ] `camera/warp_map/tests.rs`, `camera/remap/tests.rs` — shared +Z
  projection helpers.
- [ ] `patch/{cloud,view_selection,normal_refine,keypoint_localize,keypoint_subpixel}/tests.rs`
  — cameras "looking +Z", front-facing normals `(0,0,−1)`, behind-camera
  placements.
- [ ] `reconstruction/data/tests.rs` — identity world→camera "looking down
  +Z"; the `demo()`/look-at fixture builder (A.4) update cascades here.
- [ ] `geometry/viewing_angle/tests.rs` — "+Z in front" framing.
- [ ] `camrig-format/src/tests.rs` — +Z-forward sensor poses (cubemap, insv2
  back-to-back, stereo baselines, `shortest_arc_from_z`).
- Convention-agnostic (leave): `geometry/{se3_transform,rot_quaternion,rigid_transform}`,
  `reconstruction/{triangulation,point_correspondence}`, `analysis/infinity`,
  `sfmr-colmap` DB round trips, `sfmr-format` serialization,
  `camera/viewport/tests.rs` (already OpenGL), `sfm-explorer/tests/ui_basic.rs`.

**Python tests to update:**

- [ ] `tests/test_to_nerfstudio.py` (flip matrices become identity),
  `tests/test_fisheye_rig.py`, `tests/test_pano2rig.py` (rig poses),
  `tests/test_oriented_patch.py`, `tests/test_cli_inspect_strips.py`,
  `tests/test_densify.py`, `tests/test_epipolar.py`,
  `tests/test_patch_view_selection.py`,
  `tests/test_patch_keypoint_localization.py`,
  `tests/test_patch_normal_refine.py`, `tests/test_refine_normals_keypoints.py`,
  `tests/xform/test_refine_normals.py`, `tests/test_camrig.py`,
  `tests/test_spherical_tile_rig.py`,
  `tests/test_per_spherical_tile_source_stack.py`,
  `tests/rust_bindings/test_distortion_rust_bindings.py`.
- [ ] Mild touch-ups: infinity-direction placeholders `[0,0,1,0]`
  (`tests/test_colmap_interop.py`, `tests/test_undistort.py`,
  `tests/xform/test_infinity_points.py`,
  `tests/xform/test_find_points_at_infinity.py`) — arbitrary directions,
  usually fine as-is.
- [ ] Add **new** tests: (a) v4→v5 upgrade-on-load round trip; (b)
  COLMAP-export→import identity (poses/points byte-close); (c) a
  known-geometry projection test in *both* conventions at the boundary; (d)
  camera-direction sign tests (A.6/E).

---

## 11. Work area H — Specs and docs to update alongside the code

- [x] `specs/formats/sfmr-file-format.md` — conventions section, pose/depth
  field notes, v5 migration (done with this plan).
- [x] `specs/formats/camrig-file-format.md` — sensor-frame convention (D5).
- [x] `specs/formats/matches-file-format.md` — relative-pose convention (D6).
- [x] `specs/workspace/rig-config.md` — explicitly COLMAP-convention (D4).
- [x] `specs/cli/to-nerfstudio-command.md` — "Coordinate Conversion" section.
- [x] `specs/cli/{to,from}-colmap-bin-command.md`,
  `specs/cli/to-colmap-db-command.md`, `specs/cli/solve-command.md` — note
  the boundary conversion.
- [x] `specs/gui/gui-camera-views.md` — remove the COLMAP bridge description.
- [x] `specs/cli/pano2rig-command.md`, `specs/cli/insv2rig-command.md`.
- [x] Sweep remaining specs for "+Z", "Y down", "COLMAP convention"
  (`specs/core/epipolar-curves.md`, `ray-grid-projection.md`,
  `spherical-tiles-rig.md`, `image-warping.md`, `patch-*.md`, …) and fix
  narratives.
- [x] `docs/` tutorials if any mention axes.

> _Status (2026-07-04): Done — all Area-H spec/doc updates landed in **#163**
> (`d5dcfa8`, 16 files). `docs/` checked: no tutorial mentions camera/world
> axes, so no docs changes were needed. Three items the sweep flagged for
> confirmation as later steps land the code:_
> 1. _**`spherical-tiles-rig.md` vs. Step-2 code — real divergence, resolve in
>    Step 3 (D5).** #163 documents the stored tangent basis as
>    `e_right × e_up = −direction` with `R_world_from_tile = [e_right | e_up |
>    −direction]` (a canonical −Z-forward frame, stored basis flipped). But the
>    landed Step-2 code (A.7) kept the stored basis **byte-stable** at
>    `e_right × e_up = direction` (`tile_rig.rs:148,289`), left `tile_rotation()`
>    as `[e_right | e_up | direction]`, and added a separate
>    `tile_camera_rotation()` = `[e_right | −e_up | −direction]`. The spec is
>    ahead of the code. **Recommended resolution:** in Step 3, flip the stored
>    basis to match the spec (makes `R_world_from_tile` the canonical camera
>    rotation directly, retires `tile_camera_rotation()`, and aligns with the
>    `.camrig` flip D5 performs anyway); regenerate the tile-rig `.camrig` test
>    data with it. Alternatively keep byte-stable and walk the spec back — but
>    the flip is cleaner and D5 touches this data regardless._
> 2. _**`image-warping.md` equirect formulas** (`longitude = atan2(rx, −rz)`,
>    positive-up latitude, `v = fy·(−latitude)+cy`) were written self-consistent
>    but ahead of code; A.1 has since landed — treat the code as source of truth
>    and confirm the spec matches during Step-3 verification._
> 3. _**`epipolar-curves.md` cheirality** flipped to `Xc.z < 0` mirrors the
>    landed A.3 gate; the >180°-FOV exclusion it now states explicitly is
>    pre-existing, not introduced by the migration._

---

## 12. Ordering, dependencies, and verification

Recommended build order (single coordinated branch; steps keep the tree
compiling, full green only at the end):

1. **Primitives** — add `sfmtool-core` `convention` module (D2) + unit tests
   for `S`/`W` algebra (pose/point/relative-pose round trips). *Verify:*
   `pixi run cargo test -p sfmtool-core`.

   > _Status (2026-07-03): Done — `crates/sfmtool-core/src/geometry/convention.rs`
   > implements `s_matrix`/`w_matrix`, `pose_colmap_to_canonical`,
   > `pose_canonical_to_colmap`, `relative_pose_conjugate_s`,
   > `world_rotate_w`/`world_rotate_w_inverse` with 13 unit tests; exposed as
   > batch functions on `_sfmtool.geometry` and wrapped by
   > `src/sfmtool/colmap/convention.py` (single/batch numpy +
   > `pycolmap.Rigid3d` helpers), tests in
   > `tests/rust_bindings/test_convention_rust_bindings.py`. Additive only —
   > no boundary site converted yet._
2. **Core flip** — areas A.1–A.7 and the `sfmr-format` depth change (B),
   updating Rust unit tests (G) in the same commits, module by module
   (camera models → frustum/epipolar/rectification → reconstruction →
   patch → analysis → spherical). *Verify:* `pixi run cargo test --workspace`
   (GUI compiles; its data still looks wrong until step 4).

   > _Status (2026-07-04): Done on branch `zup-core-flip` (commits `40b4848`,
   > `b88e58f`, `edb6455`, `d80530b`). D7 `S`-boundary applied in
   > `camera/distortion.rs` (`project`/`unproject`/`distort_ray`/
   > `undistort_to_ray`), cheirality gates flipped (`z < 0` in front, depth
   > `−z`), `demo()` look-at rebuilt to canonical rows `[right, up, −forward]`,
   > `depth_stats.rs` depth negated. `cargo test --workspace` green
   > (932 passed / 0 failed); `fmt`/`clippy --workspace` clean. Python
   > intentionally left inconsistent (its suite is red until steps 3–5).
   > Four judgement calls beyond the plan text — see the dated notes at A.3,
   > A.5, A.6, A.7 — need confirmation before the dependent later steps._
3. **Format v5 + boundaries** — upgrade-on-load (B), COLMAP boundary (C),
   camrig/matches (B, D5/D6), nerfstudio/pano2rig/insv2rig (D). Rebuild
   bindings: `pixi run maturin develop --release`. *Verify:*
   `pixi run test -- tests/test_colmap_interop.py tests/test_solve.py
   tests/test_to_nerfstudio.py tests/test_camrig.py tests/test_pano2rig.py
   tests/test_fisheye_rig.py`.

   > _Status (2026-07-04): Done. **3a (Rust format layer)** commits `944eca8`
   > (`.sfmr`→v5 upgrade-on-load in `SfmrReconstruction::load` — the conversion
   > lives in `sfmtool-core`, not `sfmr-format`, per crate layering), `ef74c66`
   > (`.matches`→v2 + `sfmr-colmap` DB↔matches conjugation moved into Rust),
   > `7a1d454` (`.camrig`→v2), `caadde1` (tile-basis flip; `tile_camera_rotation`
   > retired — the §11 A.7 divergence resolved by adopting the spec's stored
   > basis). **3b (Python boundaries)** commits `04f721f`/`1af1d80`/`32f7fe2`/
   > `d442913`: `colmap/io.py` S+W import/export with `apply_world_rotation`
   > flag; S-only in-pipeline pycolmap round trips (BA, densify, merge-PnP,
   > sweeps); DB export/setup convert only where Rust doesn't (NOT around
   > `*_colmap_db_matches`); nerfstudio identity, pano2rig/insv2rig, kerry_park
   > `.camrig` regenerated to v2. Both reviewed (approve / approve-with-nits);
   > primary gate 127 passed. Key correction: merge-PnP feeds world points
   > **as-is**, S-flips only the returned pose (world is convention-agnostic)._
4. **GUI** — area F. *Verify:* `pixi run gui -- <freshly solved .sfmr>` —
   scene sits on the grid right-side-up; Z-key camera view matches the
   photo (un-mirrored, upright); frusta point at the point cloud.

   > _Status (2026-07-04): Done — commit `5bfcc24`. Removed the COLMAP→GL
   > 180°-about-X bridge in `viewer_3d/mod.rs` `enter_camera_view`
   > (`end_orientation = image.quaternion_wxyz` directly); frustum/background/
   > patch-cull passenger paths ride the Step-2 core changes unchanged.
   > `cargo test -p sfm-explorer` green; reviewed. **The live visual check
   > above still needs a human with a display — deferred to step 6/human.**_
5. **Python internal + full sweep** — area E, remaining tests (G), specs (H).
   *Verify:* `pixi run test`, `pixi run cargo test --workspace`,
   `pixi run fmt && pixi run check`,
   `pixi run cargo fmt && pixi run cargo clippy --workspace`.

   > _Status (2026-07-04): Done — commits `35fc804` (Area-E sign flips:
   > `_patch_renderer`, `_epipolar_display`, `feature_match/_geometry`,
   > `_image_pair_graph`, `_solve_strips`, `_rectification` — camera-space
   > flips only; world-space sites verified untouched) and `d9758cc` (test
   > fixtures corrected to canonical truth + new `compute_camera_directions`
   > sign test; `_inject_infinity` now uses a world mean-look-direction
   > bearing). Specs (H) were already done in #163. Full green:
   > `pixi run test` **1579 passed / 0 failed**, `cargo test --workspace` 0
   > failed, `fmt`/`check`/`clippy` clean. Reviewed — approve, no bugs._
6. **End-to-end acceptance** — re-run `scripts/init_dataset_seoul_bull.sh`
   and `init_dataset_kerry_park.sh` into fresh workspaces; `sfm solve` both;
   inspect in the GUI; `sfm to-colmap-bin` → COLMAP GUI/`from-colmap-bin`
   round trip; `sfm to-nerfstudio` and check `applied_transform` is identity
   and the PLY matches the old exporter's output on the same scene
   (old-pipeline export of an old file should be *numerically identical* to
   new-pipeline export of the upgraded file — a strong global check, since
   Nerfstudio's target convention equals our new one).

   > _Status (2026-07-04): PASS (automated checks). Full suite green
   > (1579 Python + Rust workspace, 0 failed; fmt/check/clippy clean).
   > seoul_bull incremental solve → v5 `.sfmr`, cameras upright
   > (mean up_z +0.797), all depths positive. `to-colmap-bin`→`from-colmap-bin`
   > round trip is identity to float noise (max |Δquat| 2.2e-16, |Δt| 0,
   > |Δpoint| 0). `to-nerfstudio` `applied_transform` = exact identity 3×4,
   > PLY 852 finite verts. kerry_park fisheye rig solves to v5 with
   > `rig_frame_data`, rig upright (up_z 0.993). **Open:** the human GUI
   > visual confirmation (scene upright on grid, Z-key view matches photo
   > un-mirrored, frusta at the cloud) — needs a display._

Key dependencies:

- A (core flip) before everything else; C and F depend on A; B's
  upgrade-on-load depends on A's converters.
- The two patch-handedness commits (§3) must stay beneath the flip (already
  merged into this branch's history).
- After any Rust change re-exported through `sfmtool-py`, rerun
  `pixi run maturin develop --release` before Python tests.

---

## 13. Risks and open questions

- **Old embedded-patch files**: v≤4 upgrade rotates patch frames with `W`,
  which is correct for files written after commit `3a5eb3c`; files from the
  brief #159 window have flipped `v` handedness on disk and were already
  declared regenerate-only — the v5 upgrade does not try to repair them.
- **`W` heuristic quality**: scenes whose COLMAP world was not −Y-up (e.g.
  aerial, or already-transformed reconstructions) will not land Z-up; that is
  acceptable by design (the convention defines meaning; `xform` fixes
  orientation). Consider a follow-up `xform --level-horizon` that estimates
  up from camera Y axes.
- **pycolmap pose priors** (C.4): verify COLMAP's prior-based solvers treat
  the DB world consistently with the two-view geometries after the `W⁻¹`
  export choice; the kerry_park E2E solve is the canary.
- **Numerical churn**: converting on import/export multiplies by exact
  rotations (entries in {0, ±1}), so float error is negligible, but expect
  last-ulp diffs in regression comparisons against pre-migration outputs.
- **Format-version bump for `.camrig`/`.matches`** (D5/D6): if those formats
  lack a version field today, decide between adding one and declaring a
  breaking regeneration (both formats are cheap to regenerate).
- **CUDA/pycolmap variants**: the `cuda` pixi environment uses a different
  pycolmap build; boundary code is shared, but run the solve tests there
  once.
