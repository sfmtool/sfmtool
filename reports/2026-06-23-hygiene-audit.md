# Codebase Hygiene Audit — 2026-06-23

Read-only structural survey of the sfmtool repository: Python (`src/sfmtool/`,
`tests/`), the Rust workspace (`crates/`), and top-level layout (`scripts/`,
`specs/`, `docs/`, `reports/`). Totals (approx): ~23k lines Python in `src/`,
~21k lines Python tests, ~83k lines Rust across 8 crates.

This snapshot **supersedes `2026-06-09-hygiene-audit.md`** (retire decision
documented below at the end of Finding 5). That report's headline item (#2 —
regroup `sfmtool-core` into topic groups) is fully landed across PRs #107,
#109, #110, and #111, and is annotated in place. The other still-open findings
from that report (#3 `optical_flow/gpu/mod.rs`, #5 PyO3 submodule restructure,
#6 dtype-error residual, #7 `image_detail.rs::show()`, #8 GUI `panels/`
nicety, #9 `test_camrig.py` resolver split) are re-verified here with fresh
line numbers, joined by new findings that the freshly-extracted code (e.g.
`patch/normal_refine.rs`, `reconstruction/data.rs`) and the format-crate
re-survey turned up.

The headline of this run: the `sfmtool-core` regroup landed cleanly — every
group `mod.rs` is now thin wiring (9–19 lines), zero cycles, no
`#[allow(dead_code)]`, no TODO/FIXME, no `_old`-prefixed files — and the
remaining work has shifted away from that crate's root toward (a) the
`patch/normal_refine.rs` interior (the new single largest production file at
1812 lines), (b) two sf-explorer mega-methods, (c) cross-crate duplication
between the four format crates, and (d) two inline-test regressions the
previous extraction sweep didn't reach (`geometry/`, `analysis/alignment/`).

---

## Rust — `sfmtool-core`

> **Surveyed and cleared:** `reconstruction/data.rs` (1215, newly extracted in
> PR #111) — single-theme; length driven by three unavoidable methods
> (`from_sfmr_data` 182, `to_sfmr_data` 148, `demo` 202) that form a tight
> conversion story. `spherical/tile_rig.rs` (1062) — one `SphericalTileRig`
> impl, the CamRig conversion already peeled off to `tile_rig/camrig.rs`;
> watch-item if it grows further. `features/optical_flow/mod.rs` (958),
> `spherical/per_tile_source_stack.rs` (862), `camera/intrinsics.rs` (905),
> `camera/distortion.rs` (871) + `distortion/kernels.rs` (793),
> `features/feature_match/polar.rs` (844), `spherical/photometric_ransac.rs`
> (839) — each a single algorithm or a clean dispatch-vs-kernels split.
> Cross-cutting cleanliness: zero `#[allow(dead_code)]`, zero TODO/FIXME, zero
> commented-out blocks, no `_old`/`_new`/`_v2` filenames.

### 1. `patch/normal_refine.rs` — 1812 lines, six author-banner'd concerns

- Location: `crates/sfmtool-core/src/patch/normal_refine.rs` (1812 lines —
  the single largest production file in the crate; `fronto_cache.rs` 595 and
  `prof.rs` are already siblings under `patch/normal_refine/`)
- Problem: Despite genuinely clean hygiene (no dead code, no TODOs, tests
  already in sibling `normal_refine/tests.rs`), the file mixes six distinct
  concerns marked by the author's own `// ---` banner headers: public types
  (26–225), parameterization (248–323), level context (324–422), the
  objective-math kernel band (423–1099 — SIMD primitives, ZNCC z-normalization,
  IRLS/consensus scoring), coarse-to-fine search (1100–1364), top-level refine
  (1365–1519), and a multi-view render substrate `PatchViewStack` (1520–1737).
  The banners already telegraph natural seams; `fronto_cache.rs` is precedent
  for sibling extraction under `normal_refine/`.
- Proposed fix: split into the `patch/normal_refine/` subdir following
  existing precedent. Suggested layout — `params.rs` (26–225),
  `parameterization.rs` (226–323), `level.rs` (324–422), `znorm.rs` (423–910,
  the numeric kernels), `consensus.rs` (911–1099), `search.rs` (1100–1364),
  `view_stack.rs` (1520–1737). `normal_refine.rs` keeps `refine_patch_normal`,
  `refine_patch_normal_impl`, `refine_patch_cloud_normals`, `view_indices_from_reconstruction`
  (~250 lines of orchestration).
- Effort: medium — mechanical move + `pub(super)` threading on `LevelContext`,
  `ProjectedImage`, `Objective`, `PatchWindow`.
- Risk: low — internal restructure; tests already sibling-extracted, so the
  test surface is unaffected.

### 2. `features/optical_flow/gpu/mod.rs` — still 1438 lines, still three concerns (carried forward, #3)

- Location: `crates/sfmtool-core/src/features/optical_flow/gpu/mod.rs` (1438
  lines, unchanged since 2026-06-09)
- Problem: Three concerns still tangled. wgpu plumbing — 5 POD `*Params`
  structs (42–85), `GpuContext` (88–132), helper fns (608–741). Variational —
  `VariationalBufferPool` (138–316) + `GpuVariationalRefiner` (318–606). DIS
  orchestration — `GpuFlowContext` (743–1435), inside which a single
  `run_gpu_levels_prebuilt` spans 951–1434 (~484 lines). Sibling
  `dis_pipeline.rs` (569) and `pyramid_pipeline.rs` (336) already exist, so the
  orchestration in `mod.rs` is doubly out of step with the established
  pattern. Every other group `mod.rs` in this crate is ≤19 lines.
- Proposed fix: three-way split — `gpu/context.rs` (POD params, `GpuContext`,
  layout/pipeline helpers from 608–741), `gpu/variational.rs`
  (`VariationalBufferPool` + `GpuVariationalRefiner`), keep `GpuFlowContext` in
  `mod.rs` (~700 lines). Sub-finding: `run_gpu_levels_prebuilt` (484 lines) is
  itself a decomposition opportunity (per-level bind-group build vs. encode
  loop vs. final-upsample setup), worth tackling in the same pass.
- Effort: medium-high — `pub(super)` threading on `GpuContext` fields,
  helpers, `WG_SIZE`, the 5 `*Params` structs, and `VariationalBufferPool`
  fields (consumed from `mod.rs` at 852–855, 997–1000, 1241, 1363–1366). Watch
  `include_str!` shader paths.
- Risk: medium — GPU code is excluded from `pixi run test-rust` llvm-cov and
  is hardware-gated; regressions can slip past the default test path. Cover
  with `pixi run cargo test --workspace`.

### 3. Inline-`mod tests;` regression — `geometry/` and `analysis/alignment/` missed by the 2026-06-10 sweep

- Location: 19 `.rs` files crate-wide still embed `#[cfg(test)] mod tests
  { … }`. The substantial ones are concentrated in two subtrees the prior
  cleanup did not touch:
  - `geometry/transform.rs:49` (~365 test lines), `geometry/rot_quaternion.rs:218`
    (~215), `geometry/viewing_angle.rs:124` (~197),
    `geometry/rigid_transform.rs:126` (~158), `geometry/rotation.rs:63` (~142)
    — **all 5 unpaired `.rs` files in `geometry/` are inline.**
  - `analysis/alignment/kabsch.rs:150` (~176), `analysis/alignment/ransac.rs:96`
    (~88).
  - Mid-size stragglers in other groups (each ≤165 lines of inline tests):
    `reconstruction/point_correspondence.rs`, `features/sift/orientation.rs`,
    `spherical/sphere_points.rs`, `features/optical_flow/interp.rs`,
    `features/feature_match/descriptor.rs`, `reconstruction/filter.rs`,
    `features/sift/gray.rs`, `features/optical_flow/pyramid.rs`,
    `features/feature_match/mod.rs`, `features/kdforest/distance.rs`,
    `patch/normal_refine/fronto_cache.rs`, plus `camera/viewport.rs:191` (~58
    lines — the lone inline-tests holdout inside `camera/`, introduced by the
    PR #111 viewport split).
- Problem: The 2026-06-10 pass standardized on sibling `tests.rs` files (20
  such files now exist) but stopped before `geometry/` and `analysis/alignment/`.
  Two conventions coexist again — worse in `geometry/` (100% inline) than
  anywhere else. The viewport relapse is the smallest but symbolically the
  worst: a regroup commit re-introduced the very pattern the prior pass had
  cleaned.
- Proposed fix: mechanical pass to extract each `mod tests { … }` into a
  sibling `tests.rs` (or `foo/tests.rs` where the file is paired with a
  subdir). Start with `geometry/` (5 files, ~1077 test lines moved) and
  `analysis/alignment/` (2 files, ~264) — those are the biggest navigability
  wins. Roll in the smaller stragglers and `camera/viewport.rs` for uniformity.
- Effort: low — purely mechanical per file.
- Risk: low — test-only; `cargo test --workspace` covers.
> _Status (2026-06-23): Done — all 18 in-scope files lifted into sibling
> `tests.rs` (geometry ×5, analysis/alignment ×2, camera/viewport, plus the
> 10 stragglers). `patch/normal_refine/fronto_cache.rs` deliberately excluded
> per the user's scope direction (slated for the normal_refine split). See
> the `cleanup: inline-tests sweep + post-regroup spec drift` commit._

---

## Rust — `sfmtool-py`

### 4. `_sfmtool` registers 80 names on one flat module (carried forward, #5; grew from 74)

- Location: `crates/sfmtool-py/src/lib.rs` (322 lines; `#[pymodule]` block at
  121–322) + 33 flat `py_*.rs` siblings
- Problem: Re-verified and **grown**: now **80** `add_function`/`add_class`
  registrations on one `m: &Bound<'_, PyModule>` (was 74 in the prior audit).
  `src/sfmtool/__init__.py:4` still does `from sfmtool._sfmtool import *`,
  dumping the entire surface into `sfmtool`'s top level. Recent additions
  (`py_patch_cloud` with `PyOrientedPatch`/`PyPatchCloud`/`refine_normals`,
  `py_consensus_atlas`, `py_photometric_ransac`) widened the public Python API
  along the exact thematic seams (`io`, `match`, `geometry`, `flow`, `patch`,
  `spherical`) the prior audit identified. Clear filename groups today: I/O
  (`sfmr_io`, `sift_io`, `matches_io`, `camrig_io`, `colmap_binary`,
  `colmap_db`), match (`descriptor_match`, `image_match`, `sweep_match`,
  `cluster_match`), geometry (`rigid_transform`, `rot_quaternion`,
  `se3_transform`, `camera_intrinsics`, `sphere_points`), flow (`optical_flow`,
  `warp_map`), patch (`patch_cloud`, `consensus_atlas`, `photometric_ransac`),
  kd-spatial (`kdtree`, `kdforest`).
- Proposed fix: expose child modules via `PyModule::new` + `add_submodule`
  (`_sfmtool.io`, `.match` — shipped as `.matching` to dodge the Rust
  keyword, `.geometry`, `.flow`, `.patch`, `.spherical`); group the
  `py_*.rs` into subdirs; replace `from sfmtool._sfmtool import *` with
  deliberate re-exports.
- Effort: high · Risk: medium — every Python consumer of `sfmtool.<symbol>`
  must keep resolving; PyO3 submodules have `sys.modules` quirks; this is the
  only flatness reaching the public Python API.

> _Status (2026-06-23): Partially done — the `io` slice landed. 24
> file-format I/O entries (5 `.sfmr` + 2 COLMAP-binary + 2 COLMAP-db + 4
> `.matches` + 4 `.camrig` + 4 camrig-pattern + 5 `.sift` +
> `SiftWriteQueue`) are now on a `_sfmtool.io` PyO3 submodule with
> `__name__ == "sfmtool.io"` so binding objects' `__module__` reads the
> public location; the real import path stays `sfmtool._sfmtool.io` via
> a pre-`register` `sys.modules` insert through the new
> `helpers::install_submodule(parent, public_name, register)`. The
> remaining seams (`.match`, `.geometry`, `.flow`, `.patch`,
> `.spherical`) and the `from sfmtool._sfmtool import *` wildcard in
> `src/sfmtool/__init__.py` still need the same treatment. Commit
> (branch `claude/pyo3-io-submodule`)._
>
> _Status (2026-06-23): Partially done — the `matching` slice landed
> (shipped as `.matching` because Rust `match` is a keyword). 12 entries
> (2 cluster + 2 descriptor + 2 image-pair + 6 sweep) follow the io
> recipe: each source file owns its `pub fn register`,
> `matching::register` chains them, `lib.rs` installs via
> `helpers::install_submodule(m, "sfmtool.matching", matching::register)`.
> Binding objects' `__module__` reads `sfmtool.matching`. The remaining
> seams (`.geometry`, `.flow`, `.patch`, `.spherical`) and the
> `__init__.py` wildcard still pending. Commit (branch
> `claude/pyo3-matching-submodule`)._
>
> _Status (2026-06-23): Partially done — the `flow` slice landed. 9
> entries (7 optical-flow functions + `WarpMap` + `ImagePyramid`) move
> to `flow/{optical,warp}.rs` and register on `_sfmtool.flow` via the
> shared `helpers::install_submodule`. Both classes carry
> `#[pyclass(module = "sfmtool.flow")]` so their `__module__` reads
> `sfmtool.flow`. Two intra-crate cross-references
> (`py_patch_cloud.rs`, `py_spherical_tile_rig.rs`) repath to
> `crate::flow::warp::…`. The remaining seams (`.geometry`, `.patch`,
> `.spherical`) and the `__init__.py` wildcard still pending. Commit
> (branch `claude/pyo3-flow-submodule`)._
>
> _Status (2026-06-24): Partially done — the `spherical` slice landed.
> 3 entries (`evenly_distributed_sphere_points` + `SphericalTileRig` +
> `PerSphericalTileSourceStack`) move to `spherical/{sphere_points,
> tile_rig,tile_source_stack}.rs` and register on `_sfmtool.spherical`
> via `helpers::install_submodule`. Both classes carry
> `#[pyclass(module = "sfmtool.spherical")]`. Two intra-crate
> cross-references (`py_consensus_atlas.rs`, `py_photometric_ransac.rs`)
> repath to `crate::spherical::…`. The `__init__.py` wildcard grew to
> five lines (`_sfmtool` + `.io` + `.matching` + `.flow` + `.spherical`)
> — the audit's stated replace-with-deliberate-re-exports cleanup is
> still deferred. The remaining seams (`.geometry`, `.patch`) and that
> wildcard cleanup remain pending. Commit (branch
> `claude/pyo3-spherical-submodule`)._
>
> _Status (2026-06-24): Partially done — the `spatial` slice landed. 3
> entries (`KdTree2d` + `KdTree3d` + `KdForest`) move to
> `spatial/{kdtree,kdforest}.rs` and register on `_sfmtool.spatial` via
> `helpers::install_submodule`. All three classes carry
> `#[pyclass(module = "sfmtool.spatial")]`. One intra-crate
> cross-reference (`matching/cluster.rs`'s use of `extract_u8_2d` /
> `resolve_forest_params`) repaths to `crate::spatial::kdforest::…`. The
> `__init__.py` wildcard grew to six lines (`_sfmtool` + `.io` +
> `.matching` + `.flow` + `.spatial` + `.spherical`) — the audit's stated
> replace-with-deliberate-re-exports cleanup is still deferred. The
> remaining seams (`.geometry`, `.patch`) and that wildcard cleanup
> remain pending. Commit (branch `claude/intelligent-archimedes-7i8ruz`)._
>
> _Status (2026-06-24): Partially done — the `geometry` slice landed. 4
> value types (`CameraIntrinsics` + `RigidTransform` + `RotQuaternion` +
> `Se3Transform`) move to `geometry/{camera_intrinsics,rigid_transform,
> rot_quaternion,se3_transform}.rs` and register on `_sfmtool.geometry`
> via `helpers::install_submodule`; all four carry
> `#[pyclass(module = "sfmtool.geometry")]`. The crate-root `pub use`
> re-exports are kept (repointed to `geometry::…`) so the ~7 intra-crate
> users of `crate::PyCameraIntrinsics`/`crate::PySe3Transform` are
> untouched; only full-path `crate::py_<type>::` references (in `flow/`,
> `spherical/`, `py_patch_cloud`, `py_analysis`, and the two sibling
> transform files) repath to `crate::geometry::…` / `super::…`. ~50
> Python import sites (single-line + parenthesized) across `src`, `tests`,
> and `scripts` repath to `_sfmtool.geometry`. The `__init__.py` wildcard
> grew to seven lines — the replace-with-deliberate-re-exports cleanup is
> still deferred. The remaining seam (`.patch`) and that wildcard cleanup
> remain pending. Commit (branch `claude/intelligent-archimedes-7i8ruz`)._
>
> _Status (2026-06-24): Partially done — the `analysis` and `sift` slices
> landed together. `analysis` collects 11 functions from four files into
> `analysis/{core,triangulation,epipolar,image_pair_graph}.rs` (pose/track
> ops, Kabsch + RANSAC alignment, point correspondence, batch
> triangulation, epipolar curves, covisibility/frustum graphs) on
> `_sfmtool.analysis`. `sift` moves `py_sift.rs` →
> `sift/extract.rs` (`detect_sift_keypoints` + `extract_sift`) onto
> `_sfmtool.sift`. Because `sfmtool.sift` is a real Python subpackage,
> the bindings are re-exported through `src/sfmtool/sift/__init__.py`
> (not just the top-level wildcard) so their `__module__ == "sfmtool.sift"`
> is truthful and `from sfmtool.sift import extract_sift` resolves. No
> intra-crate cross-references needed repathing (all moved files use
> absolute `crate::`/`sfmtool_core::` paths). `image_dimensions`
> (`py_image.rs`) stays flat — it is image inspection, not a SIFT or
> analysis op. The `__init__.py` wildcard grew to nine lines. Remaining:
> the `.patch` seam and the deliberate-re-export cleanup of that wildcard.
> Commit (branch `claude/pyo3-analysis-sift-submodules`)._

### 5. First cut of #4: move the six `py_*_io.rs` into `crates/sfmtool-py/src/io/`

- Location: `crates/sfmtool-py/src/` — `py_sfmr_io.rs` (309),
  `py_sift_io.rs` (241), `py_matches_io.rs` (182), `py_camrig_io.rs` (192),
  `py_colmap_binary.rs` (680), `py_colmap_db.rs` (271)
- Problem: Six files dedicated to file-format I/O sit flat among 33
  `py_*.rs`. `lib.rs:56-63` already groups them under a `// ── File I/O ──`
  comment block (acknowledging the cluster) but they're 6 of the strongest
  carriers of the #4 submodule split — registering 22 of the 80 entry points
  (`read_sfmr`/`write_sfmr`/`verify_sfmr` and the equivalent triples per
  format).
- Proposed fix: as the first step of #4, move the six files into
  `crates/sfmtool-py/src/io/`. Optionally pair with exposing `_sfmtool.io` as a
  child submodule (medium effort) or land just the file move first as a no-op
  validation step (low effort) and keep `pub use` shims in `lib.rs` so existing
  `import *` consumers don't break.
- Effort: low (file move) / medium (with submodule registration)
- Risk: low — `pub use` shims keep `sfmtool._sfmtool.read_sfmr` resolving
  during the transition.
> _Status (2026-06-23): Done — six file moves to `crates/sfmtool-py/src/io/`
> with redundant `py_` prefix / `_io` suffix dropped (`io/sfmr.rs`,
> `io/sift.rs`, `io/matches.rs`, `io/camrig.rs`, `io/colmap_binary.rs`,
> `io/colmap_db.rs`); 23 `wrap_pyfunction!` call sites + 2 internal
> cross-references updated; new `io/mod.rs`. Python-facing surface
> unchanged — every function still registers flat under `_sfmtool`.
> The `_sfmtool.io` Python submodule split (the medium-effort half) is
> still open as part of the larger #4. Commit (branch
> `claude/next-step-eval`)._

### 6. Dtype-tag / array-extraction helper duplicated three ways (carried forward, #6 residual)

- Location: `crates/sfmtool-py/src/py_kdtree.rs:19-23`,
  `py_kdforest.rs:23-35`, `recon_clone.rs:28-72`
- Problem: Three independent implementations of "extract a numpy array,
  format an error mentioning the offending dtype" still coexist:
  `py_kdtree::dtype_tag` (one-line wrapper around
  `arr.getattr("dtype").getattr("name")`), `py_kdforest::extract_u8_2d` (with
  explicit fallback when getattr fails), and `recon_clone`'s
  `extract_ndarray!`/`extract_array1!`/`extract_array2!` macros (the richer
  "type + dtype" form). Each emits a different error message style for the
  same class of mistake, so a Python user gets cosmetically inconsistent
  diagnostics. `helpers.rs` (191 lines, 8 utilities) still has no
  array-extraction helper despite being the obvious home. The 2026-06-09 audit
  deliberately declined the macros — but the kd-tree/kd-forest pair is the
  residual it called out, and it remains untouched.
- Proposed fix: hoist a single `helpers::extract_array_nd<T, D>(arr, param,
  shape, dtype)` (or two thin 1D/2D fns) covering the `recon_clone` rich-message
  form; replace `dtype_tag` in `py_kdtree.rs` and `extract_u8_2d` in
  `py_kdforest.rs`. Macros in `recon_clone.rs` can stay or delegate.
- Effort: low — three call-site rewrites.
- Risk: low — Python error-message text changes (covered by
  `tests/rust_bindings/`).

> _Status (2026-06-24): Not done — still open, but the two kd files moved
> under the new `spatial` submodule (finding #4): `dtype_tag` now lives at
> `crates/sfmtool-py/src/spatial/kdtree.rs:19-23` and `extract_u8_2d` at
> `spatial/kdforest.rs:23-35`. The duplication and proposed
> `helpers::extract_array_nd` consolidation are unchanged._

> **Surveyed and cleared:** `recon_clone.rs` (736) — coherent
> `clone_with_changes` + `rebuild_observation_source` + tests.
> `py_sfmr_reconstruction.rs` (891) — `#[pymethods]` getters / per-method
> facade; the 14 trivial column accessors are an inevitable shape for a numpy
> view surface. `io/colmap_binary.rs` (680) — bidirectional COLMAP binary I/O,
> single concern. `py_patch_cloud.rs` (543), `py_optical_flow.rs` (474),
> `py_kdtree.rs` (582) — each single concern. Module-path consistency post
> the sfmtool-core regroup: facades are uniformly adopted (e.g.
> `sfmtool_core::CameraIntrinsics`, `sfmtool_core::WarpMap`); longer forms
> remain only for things that have no group-level facade today, which is
> appropriate.

---

## Rust — `sfm-explorer`

### 7. `app.rs::run_ui_and_paint` — 566-line single-method per-frame pipeline (NEW)

- Location: `crates/sfm-explorer/src/app.rs:67-632` (file 633 lines; `App`
  has only three methods — `window_hwnd`, `try_init_gesture_handler`,
  `run_ui_and_paint` — and the third dwarfs the file)
- Problem: One method holds the entire frame pipeline as inline phases — six
  "ensure/upload" steps (point cloud, frustum geometry, frustum colors, track
  rays, bg image, clip planes), camera uniform updates, scene encoder + render
  passes (scene → target indicator → track rays → depth/pick readback),
  gesture-event gathering, the egui pass with the dock UI, AccessKit
  propagation, tessellate + texture-set deltas, then surface acquire + present
  + pick-result dispatch. The comment skeleton itself (24 inline `//` section
  headers from 90–625) is the strongest signal — these are the natural
  extraction seams. The file already has no module-level structure beyond
  `impl App`, so each new GPU subsystem widens the single method instead of
  finding a home.
- Proposed fix: extract three private methods on `App` along the existing
  comment boundaries — `prepare_uploads(&mut self, device, queue, renderer)`
  (the six `upload_*`/`update_*` blocks at 96–220),
  `render_scene(&mut self, device, queue, encoder, ...)` (the wgpu pass cluster
  at 261–310), and `run_egui_pass(&mut self, window, ...)` (the egui + dock UI
  at 312–474). Leaves `run_ui_and_paint` as a ~100-line orchestrator wiring
  surface acquire/present and pick-result dispatch.
- Effort: medium · Risk: low — single binary, no public API; the borrow
  ergonomics of `&mut self` across upload/render phases need care but the data
  is already separable (`scene_renderer`, `viewer_3d`, `state`).

### 8. `image_detail.rs::show()` — still a 625-line method in a 1190-line file (carried forward, #7)

- Location: `crates/sfm-explorer/src/image_detail.rs:125-749`
- Problem: Re-verified, **unchanged**. `ImageDetail::show` still spans
  125–749 (625 lines): panel sizing/fit (205–224), pointer / drag / keyboard /
  scroll / `GestureEvent` handling (236–380), then a 7-branch `OverlayMode`
  match (401–646) — `None`/`Features`/`ReprojError`/`TrackLength`/
  `MaxTrackAngle`/`DepthReliability`/`ConditionNumber`. Below it sit 8 free
  metric/coloring helpers (`draw_feature_ellipse` 982–1042 through
  `compute_track_length_range` 1165–1188) plus `find_nearest_tracked_feature`,
  `feature_size`, `compute_error_range`, `log10_condition`,
  `compute_finite_value_range`, `compute_max_track_angle_deg` — a self-contained
  cluster.
- Proposed fix: extract `image_detail/input.rs` (drag-pan, scroll, pinch,
  keyboard, gesture handling — mirroring `viewer_3d/input.rs`),
  `image_detail/overlay.rs` (the 7-branch overlay match + `draw_feature_ellipse`
  + metric/coloring helpers); leave `show` in `image_detail/mod.rs` as
  orchestration plus `load_image`/`load_*_features`.
- Effort: medium · Risk: low — intra-crate, no public API; covered by
  `tests/ui_basic.rs`.

> **Surveyed and cleared:** `scene_renderer/upload.rs` (852) — six independent
> `upload_*` methods on `SceneRenderer`, each topical. `viewer_3d/mod.rs`
> (647) — `Viewer3D::show` is a tight 143-line orchestrator.
> `point_track_detail.rs` (844) — already delegates to
> `show_header`/`show_observation_table`/`draw_thumbnail`/`prepare_observations`.
> `image_browser.rs` (735) — `show` is 539 lines but reads as one cohesive
> scroll/animate/draw loop without `image_detail`'s overlay-mode fanout.
> `platform/windows.rs` (665) — DirectManipulation + WM_POINTER plumbing is
> irreducibly intertwined. `dock.rs` (362) — clean
> `egui_dock::TabViewer` impl plus two compute helpers.
> Carry-forward #8 (`panels/` group) judged not worth the churn given how flat
> the GUI crate now is: 10 top-level modules, three of them already-grouped
> subtrees; the four panel-shaped files would each grow a subdirectory anyway
> once #7/`app::run_ui_and_paint` land.

---

## Rust — Format crates

### 9. `archive_io.rs` duplicated across four format crates, with the cast-slice fix copy-pasted four times

- Location: `crates/sfmr-format/src/archive_io.rs` (163),
  `sift-format/src/archive_io.rs` (161),
  `matches-format/src/archive_io.rs` (164),
  `camrig-format/src/archive_io.rs` (140). The cast-slice fast/slow-path
  block is at `sfmr-format:79-92`, `sift-format:76-89`, `matches-format:83-96`,
  `camrig-format:79-92` — same body, same comment, with one-line drift (sift
  is missing the `bytes.is_empty()` early return; camrig has a different
  `write_binary_entry` signature returning `()` instead of `Vec<u8>`; sfmr
  alone owns `read_uint128_array`). The shared module-doc lines
  `//! Shared ZIP + zstd I/O utilities for `.sift` and `.sfmr` file formats.`
  appear verbatim in two crates whose name they don't match.
- Problem: This is the "shared alignment-safe array reader" duplication the
  prior cleanup brief asked about. The cast-slice fix was applied independently
  to each copy, with at least one already drifted (sift missing the empty-bytes
  guard). Any future format-level I/O change (zstd level handling, hash format,
  new error variant) has to be made in 4 places.
- Proposed fix: extract a small `archive-format-common` crate (or
  `archive_io` module in a thin shared crate) holding `read_zst_entry`,
  `read_json_entry`, `read_binary_array`, `read_uint128_array`,
  `zstd_compress`, `write_json_entry`, `write_binary_entry`, `format_hash`, and
  `ArchiveIoError`. Each of the four format crates re-exports what it needs.
  Reconcile the empty-bytes guard, write-entry return type, and dead-code
  annotation as part of the merge.
- Effort: medium — new crate in workspace + 4 import updates + Cargo.toml
  editing; tests already cover the behavior.
- Risk: low — pure refactor; the implementations are already byte-similar.

### 10. ~1200-line `#[cfg(test)] mod tests` block still inline in `sfmr-format/src/lib.rs`

- Location: `crates/sfmr-format/src/lib.rs:30-1302` (1272 lines of inline
  tests under the `lib.rs` re-export shell). Smaller-but-similar in
  `camrig-format/src/lib.rs:35-671` (637 lines),
  `matches-format/src/lib.rs:28-543` (516 lines),
  `sift-format/src/lib.rs:28-326` (299 lines).
- Problem: The 2026-06-10 inline-test extraction pass ran across
  `sfmtool-core` and lifted tests into sibling `tests.rs` files following the
  `distortion.rs` convention. The format crates were not in scope and still
  carry the tests inline — `sfmr-format/lib.rs` reads as 30 lines of crate
  root plus 1272 lines of fixture builders and round-trip tests. By the bar
  set in that pass (extraction threshold ~700 inline lines), `sfmr-format/lib.rs`
  and `camrig-format/lib.rs` qualify; `matches-format` is borderline;
  `sift-format` is below threshold.
- Proposed fix: for each crate, lift the `#[cfg(test)] mod tests { … }` block
  into a sibling `tests.rs` (declared `#[cfg(test)] mod tests;` from `lib.rs`),
  matching the `sfmtool-core` convention. Skip `sift-format`. Net `lib.rs`
  reductions: `sfmr-format` 1302→30, `camrig-format` 671→35, `matches-format`
  543→28.
- Effort: low — mechanical, mirrors the prior in-tree precedent.
- Risk: low.
> _Status (2026-06-23): Done — `sfmr-format/src/lib.rs` 1302 → 31,
> `camrig-format/src/lib.rs` 671 → 35, `matches-format/src/lib.rs` 543 → 28.
> `sift-format` left alone (below threshold). Same commit as Finding 3._

> **Surveyed and cleared:** `crates/sfmr-format/src/write.rs` (1024) and
> `depth_stats.rs` (500) — single coherent purpose, well-sectioned.
> `crates/sfmr-colmap/` — already split into `colmap_db/` + `colmap_io/`
> submodules with `lib.rs` at 11 lines; `colmap_db/tests.rs` (1207) already
> follows the sibling-tests convention.

---

## Python — `src/sfmtool/` and `tests/`

> The Python side held up to its 2026-06-09 "clean" baseline. The full
> re-survey turned up one low-effort finding (test-helper retry-loop
> duplication in `conftest.py`) plus one carried-forward optional split. Two
> watch-items (`xform/` flatness, `_commands/` size) judged acceptable.

### 11. Solve-retry loop duplicated between `build_cluster_reconstruction` and `kerry_park_camrig_workspace_once`

- Location: `tests/conftest.py:214-249` (`build_cluster_reconstruction` retry
  loop) and `tests/conftest.py:517-559` (`kerry_park_camrig_workspace_once`
  retry loop)
- Problem: Both implement the same retry pattern: max-attempts loop,
  attempt-1 uses fixed seed 42 and retries pass `seed=None`, clear
  `colmap_dir` / stale `.sfmr` siblings before each try, track best by
  `(image_count, point_count)`, copy the best to a `_best*` stash, then
  canonicalize. The kerry-camrig fixture (~75 lines of solve-and-retry inside
  an otherwise straightforward fixture) is essentially the same algorithm as
  the helper above it; the only meaningful difference is that the camrig
  version calls `run_global_sfm(..., matching_mode="cluster")` directly while
  `build_cluster_reconstruction` does match-then-solve in two steps. The two
  loops also differ in small ways (`>= 200` vs `>= min_point_count`,
  `image_count >= expected` vs `==`) — that's drift, not design.
- Proposed fix: extract a `_solve_with_retry(solve_fn, output_sfm_file, *,
  expected_image_count, min_point_count, max_attempts, random_seed=42)` helper
  that takes a callable doing one solve attempt and returns the `Path` to the
  canonicalized best. `build_cluster_reconstruction` keeps its match step and
  calls the helper for the solve; the camrig fixture passes its
  `run_global_sfm(..., matching_mode="cluster")` call as the `solve_fn`. Drop
  the `_best` / `_best_camrig` stash names in favor of a single convention.
- Effort: low.
- Risk: low — pure test-fixture refactor; both fixtures are session-scoped
  and have explicit pass/fail floors so a regression would surface immediately.

### 12. `tests/test_camrig.py` (712) still covers six unrelated areas (carried forward, #9; narrowed further)

- Location: `tests/test_camrig.py:57-687`
- Problem: After the 2026-06-09 extraction of `cam cp` (commit 0d42d92), the
  file still spans `.camrig` format round-trip (57–90), spherical-tile-rig CLI
  (92–153), PyO3 `write/read_camrig` bindings (166–220, 420–441), `camrig
  create` CLI (225–415), resolver behavior (474–634, 15 tests), and
  pattern-matching bindings (639–677). The 2026-06-09 audit flagged this same
  "optional resolver/pattern split" and explicitly left it as-is. Re-surveying:
  the resolver block is by far the largest sub-cluster (160 lines, 15 tests,
  all rooted in `_make_camrig` + `_touch_images` helpers at 444–472) and shares
  zero fixtures with the CLI tests above it — it would be a clean lift.
- Proposed fix: optional — move the 15 resolver tests + the two helpers to
  `tests/test_camrig_resolve.py` (~190 lines). Leaves `test_camrig.py` at
  ~520 lines covering format / bindings / `camrig create` CLI / spherical-tile
  CLI / pattern-matching, all of which sit much closer to the same surface.
- Effort: low · Risk: low.

> **Surveyed and cleared:** Largest source files all single-concern —
> `sift/file.py` (877), `motion/recon_discontinuity.py` (799),
> `feature_match/_run.py` (776), `_densify.py` (745), `_compare.py` (745),
> `visualization/_flow_display.py` (707), `colmap/io.py` (674),
> `visualization/_epipolar_display.py` (625), `analyze/summary.py` (623),
> `_undistort_images.py` (595). Largest test files all single-concern —
> `test_sift_file.py` (711), `test_densify.py` (686), `test_undistort.py`
> (648), `test_epipolar.py` (598), `test_merge.py` (537). No `*_old.py` /
> `*_legacy.py` / `*_deprecated.py` anywhere; no commented-out blocks.
> Rust-path references in Python comments are current after the regroup
> (`rig/panorama.py:41` cites `sfmtool_core::spherical::tile_rig::MIN_PATCH_SIZE`;
> `analyze/{depth,summary}.py:19` both cite
> `sfmtool_core::analysis::infinity::DEFAULT_INVERSE_DEPTH_Z_CUTOFF` — all
> three resolve). `xform/` (19 modules at one level) was considered for
> sub-grouping and judged not worth the churn; same call as 2026-06-09.

---

## Top-level — `specs/`, `reports/`, `scripts/`, `docs/`

### 13. Several `specs/core/*` files reference pre-regroup paths that no longer exist

- Location:
  - `specs/core/image-warping.md:3,498,506,681,724,726` — Status line says
    `warp_map.rs` / `py_warp_map.rs`; actual locations are
    `crates/sfmtool-core/src/camera/warp_map.rs` (and the `warp_map/` subdir)
    and `crates/sfmtool-py/src/py_warp_map.rs`.
  - `specs/core/photometric-subsets-ransac.md:3-5` — Status line says
    `crates/sfmtool-core/src/photometric_ransac.rs` / `py_photometric_ransac.rs`;
    actual location is `crates/sfmtool-core/src/spherical/photometric_ransac/`
    (directory module). The `py_*` path is correct.
  - `specs/core/fronto-parallel-patch-cache.md:18` —
    "Refinement (`coarse_to_fine` in `patch_normal_refine.rs`)" — the file no
    longer exists at the crate root; the code is now under
    `crates/sfmtool-core/src/patch/normal_refine/`. The same file at line 166
    correctly cites `patch_normal_refine/fronto_cache.rs`, so the spec is
    internally inconsistent.
- Problem: Exactly the post-regroup spec drift the brief called out. Status
  lines and "implemented in `<path>`" pointers were written against the flat
  layout that existed before the `patch/`, `camera/`, `spherical/` thematic
  groupings landed.
- Proposed fix: three small spec edits to retarget the paths.
- Effort: low · Risk: low — doc-only.

### 14. `specs/formats/sfmr-v4-patch-keypoints.md` is a partially-implemented Draft mismarked as draft

- Location: `specs/formats/sfmr-v4-patch-keypoints.md:1-10`
- Problem: Header still reads `# Draft:` with `Status: draft for review`, but
  the embedded "Implementation status" block says Stages 1 and 2 are landed
  (Rust crate + PyO3 bindings + in-memory `SfmrReconstruction` support). The
  same pattern triggered the 2026-06-09 spec audit's "stale Draft marker"
  finding for `photometric-subsets-ransac.md` — a partially-implemented draft
  is misleading. Either Status should flip to `Partially implemented` listing
  which stages shipped, or the implemented portion should fold into
  `sfmr-file-format.md` now and the draft pared back to the remaining items
  (Stage 3 Python producer, round-trip tests, fold-in).
- Proposed fix: flip Status to "Partially implemented", cross-reference the
  implementing modules; defer the fold-in until Stage 3 ships.
- Effort: low · Risk: low.

### 15. `reports/2026-06-09-spec-audit.md` (501 lines) is ~99% actioned and ready to retire

- Location: `reports/2026-06-09-spec-audit.md`
- Problem: A line-by-line scan against the AGENTS.md retire criteria shows
  every finding either has a `Status (2026-06-10): Done` marker or is
  explicitly a "no action" verdict. The only substantive open item is the
  `FULL_OPENCV` cross-command Choice-list discussion (`match-command.md`
  section, ~line 117) — a "discuss" recommendation, not a Done/Not-done action.
  By AGENTS.md: *"The substantive findings are resolved and only minor or
  discussion-grade items remain (carry those forward — fold them into a
  related report, the next regenerated snapshot, or an issue — rather than
  keeping a near-empty report alive for them)."* This report fits exactly that
  criterion. By contrast, the prior `2026-06-09-hygiene-audit.md` is now also
  fully superseded by this snapshot.
- Proposed fix: retire `2026-06-09-spec-audit.md` (carry the `FULL_OPENCV`
  discussion item forward into the next-steps report or into the running
  `audit-specs` re-run that follows this one). Retire `2026-06-09-hygiene-audit.md`
  in the same commit — superseded by this snapshot, headline #2 closed, the
  remaining items (#3/#5/#6/#7/#9) all carried forward into the findings above
  with refreshed line numbers.
- Effort: low · Risk: low — git preserves history.

> **Surveyed and cleared:** `scripts/` — every entry has a clear purpose; no
> dead scripts, no near-duplicates beyond the intentional dataset shells.
> `specs/drafts/` — already empty after the 2026-06-10 spec retirements.
> `docs/` — minimal, no leftover content.

---

## Top 3 (best effort-to-value)

1. **Inline-tests regression sweep across the workspace (#3 + #10).** `geometry/`
   (5 files, ~1077 test lines) and `analysis/alignment/` (2 files, ~264) in
   `sfmtool-core`, plus `camera/viewport.rs` (~58); paired with lifting the
   inline tests out of `sfmr-format/lib.rs` (~1272), `camrig-format/lib.rs`
   (~637), and `matches-format/lib.rs` (~516). All mechanical, all low risk,
   and the workspace ends up with a single sibling-`tests.rs` convention again.
   Includes the `viewport.rs` relapse — symbolically the worst miss since a
   regroup commit re-introduced the pattern.

2. **Split `patch/normal_refine.rs` along its existing banner headers (#1).**
   The single largest production file in the crate (1812 lines) — its author
   already drew the six cut lines with `// ---` banners, and `fronto_cache.rs`
   is precedent for sibling extraction under `normal_refine/`. Biggest
   navigability win in `sfmtool-core` now that the regroup has landed.

3. **Consolidate `archive_io.rs` across the four format crates (#9).** A
   small `archive-format-common` crate eliminates 4-way duplication of an
   alignment-sensitive primitive, fixes one already-drifted copy (sift missing
   the empty-bytes guard), and converts future cast-slice / zstd / hash
   changes into a single edit. Plus it gives the workspace its first proper
   shared utility crate.

> Larger items still open and worth scheduling beyond the Top 3: the
> `optical_flow/gpu/mod.rs` split (#2, medium-high effort, hardware-gated test
> risk), the `image_detail.rs::show()` + `app.rs::run_ui_and_paint` mega-method
> decompositions in `sfm-explorer` (#7, #8), the PyO3 submodule restructure
> (#4, the highest-effort item; first cut via #5 is the I/O subdir move),
> the dtype-extractor consolidation (#6), the `test_camrig.py` resolver split
> (#12), and the post-regroup spec drift cleanup (#13, #14). The
> `archive-format-common` crate (#9) plus the inline-tests sweep would also
> set up the format-crate trio (`sfmr-format`, `matches-format`,
> `camrig-format`) for a `verify.rs` cross-crate consolidation in a later
> pass.
