# Hygiene audit — 2026-07-07

Read-only structural survey of the whole codebase (Python `src/sfmtool/` + `tests/`,
Rust `crates/`, top-level layout) for oversized multi-concern files, fragmentation,
misleading names, directory smells, and dead code. Produced by the `audit-hygiene`
skill; supersedes the 2026-06-23 snapshot (retired in the same commit — its resolved
findings are history, its open items are carried forward below with status notes).

**Headline:** the 2026-06-23 backlog was worked down well — normal_refine/, gpu/,
app.rs, image_detail/ splits all held, and the `_sfmtool` submodule migration is
nearly complete (80 flat registrations → 10). The new growth areas are the two big
keypoint files that landed with #174/#175 (`keypoint_localize.rs` at 2042 lines is
now the largest source file in the workspace) and a handful of GUI/test files that
crossed the multi-concern threshold. Carried forward still-open: conftest retry-loop
duplication (#11), test_camrig split (#12), archive_io duplication (#9, deferred —
drift now affects 2 of 4 copies).

---

## Rust — `sfmtool-core`

**keypoint_localize.rs mixes params, cache, search, SIMD kernels, and orchestration**
> _Status (2026-07-08): Done — extracted `keypoint_localize/{params,search,kernels}.rs`
> (top module now 1007 lines, holding cache + orchestration); AVX2 kernels moved
> intact with SAFETY blocks. Pure code motion, `cargo clippy`/patch tests pass._
- Location: `crates/sfmtool-core/src/patch/keypoint_localize.rs` (2042 lines)
- Problem: One flat file holding 6 clearly distinct top-level concerns, well past the point where the sibling `normal_refine/` was broken up:
  1. Params/strategy config: `SearchStrategy`, `KeypointLocalizeParams`, `KeypointLocalization` (lines 48-207, ~110 lines)
  2. Per-view render cache: `ContextTile`, `cache_istride`, `render_context` (208-368, ~160)
  3. Scalar scoring primitives: `extract_core`, `znorm_core`, `template_zncc`, `parabolic`, `median` (369-483)
  4. Search strategies: `SearchScratch`, `ShiftResult`, `search_shift`, `search_shift_plus_descent` (484-1016, ~530)
  5. SIMD/scalar cell kernels: `compute_channel_grids` + `_scalar` + `_avx2`, `accumulate_count`, `score_cell_one_channel` + `_scalar` + `_avx2`, `count_invalid_at_cell` (1017-1449, ~430 lines, ~200 of them raw `_mm256_*` AVX2/FMA intrinsics with SAFETY blocks)
  6. Orchestration: `ViewState`, `retain_states_and_caches`, `localize_patch_keypoints`, `finalize`, `localize_patch_cloud_keypoints`, `track_views_from_reconstruction` (1450-2041, ~530)
- Proposed fix: Convert to a `keypoint_localize/` module (the sibling dir already exists holding `tests.rs`) following the `normal_refine/` precedent — extract `kernels.rs` (concern 5, direct analog of `camera/distortion/kernels.rs`), `search.rs` (concern 4), and `params.rs` (concern 1). Leave cache + orchestration in the top module.
- Effort: medium
- Risk: low — pure move of private free functions within one crate module; the AVX2 `#[target_feature]`/`unsafe` blocks must move intact with their SAFETY comments, and `mod tests;` stays pointing at the same sibling tests.

**reconstruction/data.rs bundles the type model, SfmrData conversion, and error-recompute logic in one 1200-line impl**
- Location: `crates/sfmtool-core/src/reconstruction/data.rs` (1505 lines)
- Problem: A single `impl SfmrReconstruction` (lines 243-1454, ~1210 lines, ~30 methods) spans three separable concerns: (a) serialization/round-trip conversion — `from_sfmr_data` (888-1076) and `to_sfmr_data` (1098-1251), ~340 lines together; (b) reprojection/error recompute — `compute_observation_reprojection_errors`, `embedded_point_reprojection_errors`, `recompute_point_errors`, `recompute_infinity_point_errors`, `recompute_depth_statistics` plus free helpers (~450 lines); (c) the actual data model + small accessors. `data.rs` is now a misleading catch-all name for what is really "types + serde + error-math".
- Proposed fix: The sibling `data/` dir already exists (holds `tests.rs`). Extract `data/conversion.rs` for the `SfmrData` <-> `SfmrReconstruction` round-trip and `data/errors.rs` for the reprojection/depth recompute methods, leaving type definitions and accessors in `data.rs`.
- Effort: medium
- Risk: medium — the conversion methods touch many private struct fields (same-module privacy applies to child modules, so workable); tests call through the public type and need no change.

**keypoint_subpixel.rs repeats the localize file's params+kernels+orchestration mix (smaller, same shape)**
> _Status (2026-07-08): Done — extracted `keypoint_subpixel/{params,kernels}.rs`
> (top module now 861 lines, holding the GN orchestration); the tile render +
> scoring kernels went to `kernels.rs`. Done in the same pass as keypoint_localize
> so the two stay symmetric. Pure code motion, `cargo clippy`/patch tests pass._
- Location: `crates/sfmtool-core/src/patch/keypoint_subpixel.rs` (1220 lines)
- Problem: Structurally parallel to keypoint_localize with the same three concerns interleaved: params/config `ConsensusRefresh`+`KeypointSubpixelParams`+`KeypointRefinement` (97-277, ~150); render/jacobian math `render_core`, `render_core_with_jg`, `znorm_core`, `ecc_score`, `view_jacobian`, `solve_2x2` (278-554, ~260); and Gauss-Newton orchestration `ViewState`, `refine_patch_keypoints`, `render_representative`, `RunningConsensus`, `GnScratch`, `refine_one_view`, `finalize`, `refine_patch_cloud_keypoints` (555-1219, ~640). Crossing the same threshold; benefits from being split consistently with keypoint_localize.
- Proposed fix: Extract `keypoint_subpixel/params.rs` and `keypoint_subpixel/kernels.rs` (render + `view_jacobian` + `ecc_score` + `solve_2x2`) into the existing sibling dir, keeping the GN orchestration in the top file. Best done in the same pass as keypoint_localize so the two stay symmetric.
- Effort: medium
- Risk: low — private free functions moving within the module; no public API change.

**Minor: three files keep inline test modules against the sibling-`tests.rs` convention**
- Location: `crates/sfmtool-core/src/patch/normal_refine/view_subset.rs` (inline `mod tests` at line 118, ~141 test lines), `.../normal_refine/fronto_cache.rs` (inline at line 504, ~158 test lines), `features/sift/simd.rs` (inline at 111)
- Problem: The crate convention is a sibling `mod tests;` file (56 files follow it); these three keep inline `#[cfg(test)] mod tests`. Tests don't dwarf their files — a consistency nit, not a size problem.
- Proposed fix: Optionally hoist to sibling `tests.rs` for convention parity, or leave as-is (low value).
- Effort: low
- Risk: low — mechanical test move.

> Surveyed and cleared: `spherical/tile_rig.rs` (1091) — one cohesive type; `camera/distortion.rs` (1219) — kernels already extracted; `camera/remap.rs` (1086), `camera/intrinsics.rs` (905), `patch/cloud.rs` (851), `features/optical_flow/mod.rs` (958) + `variational.rs` (764) — already split into subdirs per the prior audit; `spherical/per_tile_source_stack.rs` (866), `photometric_ransac.rs` (839), `features/feature_match/polar.rs` (844), `features/sift/scale_space.rs` (810) — each a single named algorithm with sibling `tests.rs`. Large `*/tests.rs` files are the convention, not a smell.

## Rust — `sfmtool-py`

**`_sfmtool` submodule split (carried forward from 2026-06-23 #4) — current status: substantially progressed, residual remains**
- Location: `crates/sfmtool-py/src/lib.rs`, `src/sfmtool/__init__.py`
- Problem: The 8 slices are now genuinely registered as real Python submodules — all eight (`geometry`, `io`, `sift`, `matching`, `analysis`, `flow`, `spatial`, `spherical`) go through `helpers::install_submodule` (helpers.rs:224), which does the full wiring: `PyModule::new` with the public `__name__`, a `sys.modules` entry at the real dotted path, and `parent.add_submodule`. Flat registrations remaining on `_sfmtool` are down to **10** (from the original 80): `build_profile`, `py_image::image_dimensions`, `refine_photometric_ransac_py`, `render_consensus_atlas_py`, and 6 classes (`PySfmrReconstruction`, `PyRangeExpr`, `PyRansacPhotometricOutput`, `PyOrientedPatch`, `PyPatchCloud`, `ProgressCounter`). However, `src/sfmtool/__init__.py:4` **still does** `from sfmtool._sfmtool import *`, so those 10 flat names still leak via wildcard, and **8 binding files remain flat at `src/` top level** (`py_consensus_atlas.rs`, `py_image.rs`, `py_patch_cloud.rs`, `py_photometric_ransac.rs`, `py_progress.rs`, `py_range_expr.rs`, `py_sfmr_reconstruction.rs`, `recon_clone.rs`) plus shared `helpers.rs`.
- Proposed fix: Group the residual flat bindings into 2-3 more submodules (e.g. `patches` for patch-cloud/oriented-patch/photometric/consensus, `reconstruction` for reconstruction+range-expr+recon_clone, `image` for image_dimensions) via the same `install_submodule` path; then replace the `import *` in `__init__.py` with explicit re-exports.
- Effort: medium
- Risk: medium — moving classes between Python submodules changes `__module__` (pickle paths); the ~10 `from sfmtool._sfmtool import ...` call sites (rig/panorama.py, _embed_patches.py, scripts/*) would need updating.

**py_patch_cloud.rs mixes a value type with four heavy refinement algorithms**
- Location: `crates/sfmtool-py/src/py_patch_cloud.rs` (1535 lines)
- Problem: One file holds the small `PyOrientedPatch` value type (lines 42-153) plus `PyPatchCloud`, whose single `#[pymethods]` impl bundles constructors/accessors with four large, independent photometric algorithms: `refine_normals` (~478-774, ~300 lines), `select_views` (~797-918), `localize_keypoints` (~972-1143), and `refine_keypoints` (~1248-1535, ~290 lines). These four are distinct concerns. ("Consensus" is a shared internal notion woven through all four, not a separable fifth.)
- Proposed fix: split into a `py_patch_cloud/` dir: `mod.rs` (both types + constructors/accessors), `refine_normals.rs`, `select_views.rs`, `localize_keypoints.rs`, `refine_keypoints.rs`, each an additional `#[pymethods] impl PyPatchCloud` block (Rust permits multiple pymethods impls across files).
- Effort: medium
- Risk: low-medium — pure code motion; shared private helpers (`build_pyramids_and_poses`, `parse_reduce`) move to `mod.rs` as `pub(super)`.

## Rust — format crates

**archive_io.rs duplication (carried forward from 2026-06-23 #9, deferred) — current status: unchanged, drift slightly worse**
- Location: `sfmr-format/src/archive_io.rs` (163), `sift-format/src/archive_io.rs` (161), `matches-format/src/archive_io.rs` (164), `camrig-format/src/archive_io.rs` (140)
- Problem: Still 4 separate copies, all with distinct hashes. The `if bytes.is_empty()` early-return guard ahead of the `try_cast_slice` block is present in sfmr-format (line 76) and matches-format, but **missing in sift-format AND camrig-format** — the new camrig-format crate was created without the guard too, so the drift now affects 2 of 4 copies.
- Proposed fix: (stays deferred per maintainer direction) extract to a shared `archive-io` helper crate; grab the missing empty-bytes guards opportunistically if either crate is touched.
- Effort: medium
- Risk: low — behavior-preserving if the guard is unified in.

**sift-format keeps its entire test suite inline in lib.rs**
> _Status (2026-07-18): Done — extracted to `crates/sift-format/src/tests.rs`
> (324 lines); lib.rs now declares `#[cfg(test)] mod tests;`. Test-only code
> motion; `cargo test -p sift-format` green._
- Location: `crates/sift-format/src/lib.rs` (lines 28-354, ~326-line inline `#[cfg(test)] mod tests`)
- Problem: sift-format has no `tests.rs` and stuffs ~326 lines of tests inline at the bottom of lib.rs. Its sibling format crates (sfmr-format, matches-format, camrig-format) all declare `mod tests;` pointing at a dedicated sibling `tests.rs`. sift-format is the odd one out.
- Proposed fix: extract to `sift-format/src/tests.rs` and change lib.rs to `#[cfg(test)] mod tests;`.
- Effort: low
- Risk: low — test-only move.

**colmap_io subdir is inconsistent with colmap_db on test placement**
> _Status (2026-07-18): Done — extracted the mod.rs block to
> `crates/sfmr-colmap/src/colmap_io/tests.rs` (665 lines) via `mod tests;`,
> matching colmap_db. The tiny `read.rs` inline test (~17 lines) left in place
> per the finding's "optional" note. `cargo test -p sfmr-colmap` green._
- Location: `crates/sfmr-colmap/src/colmap_io/mod.rs` (~667-line inline `mod tests`, lines 22-688), `colmap_io/read.rs` (~17-line inline test at line 512)
- Problem: Sibling subdir `colmap_db/` uses a dedicated `colmap_db/tests.rs` (1237 lines) via `mod tests;`. But `colmap_io/mod.rs` (a module-wiring file) carries a ~667-line inline test module — larger than the actual module code. Directly inconsistent within the same crate.
- Proposed fix: extract to `sfmr-colmap/src/colmap_io/tests.rs` and declare `mod tests;`, matching colmap_db.
- Effort: low
- Risk: low — test-only move.

**Scattered inline test modules in format crates that otherwise use sibling tests.rs**
> _Status (2026-07-18): Done (the two large blocks) — extracted
> `sfmr-format/src/depth_stats/tests.rs` (151 lines) and
> `camrig-format/src/pattern/tests.rs` (113 lines), each declared via
> `#[cfg(test)] mod tests;` (single-file-module → sibling-dir pattern, matching
> sfmtool-core). The two tiny blocks (`verify.rs` ~26, `colmap_io/read.rs` ~17)
> left inline per the finding's "optional" note. `cargo test` green._
- Location: `sfmr-format/src/depth_stats.rs` (~153-line inline test at line 373), `camrig-format/src/pattern.rs` (~115-line inline test at line 304), plus small ones in `sfmr-format/src/verify.rs` (~26 lines) and `sfmr-colmap/src/colmap_io/read.rs` (~17 lines)
- Problem: These crates already establish the sibling-`tests.rs` convention, but individual modules keep sizeable per-module test blocks inline. The two large ones (depth_stats 153, pattern 115) are the real smell.
- Proposed fix: move the larger inline blocks into the crate's existing `tests.rs` (or a per-module sibling) per repo convention; the tiny ones are optional.
- Effort: low
- Risk: low — test-only move.

**write_sfmr_with_options is a single ~537-line function**
- Location: `crates/sfmr-format/src/write.rs` (`write_sfmr_with_options`, lines 96-633)
- Problem: The file is single-concern (sfmr serialization), but one function spans ~537 lines and dominates the 1031-line file, doing pyramid/normal merging, dimension validation dispatch, and section-by-section archive writing inline. Validation helpers already exist as siblings, showing the pattern; the core writer body was not decomposed similarly.
- Proposed fix: extract cohesive stages (per-section writers) into private helpers within write.rs; no new file needed.
- Effort: medium
- Risk: medium — it's the core write path; must preserve exact byte layout/section ordering.

> Surveyed and cleared: `py_sfmr_reconstruction.rs` (1015) — single-concern surface of one type, ~50 mostly-thin accessors; `sfmr-format/src/tests.rs` (1302), `colmap_db/tests.rs` (1237), `camrig-format/src/tests.rs` (758) — convention-following test files; `camrig-format` layout consistent with sibling format crates (same lib/types/read/write/verify/archive_io/tests skeleton + domain `pattern.rs`); submodule subdirs (`io/`, `matching/`, `geometry/`, `analysis/`, `flow/`, `spatial/`, `spherical/`, `sift/`) well-grouped. No dead modules or misleading names found.

## Rust — `sfm-explorer`

**Grab-bag of unrelated GPU uploads in one file**
- Location: `crates/sfm-explorer/src/scene_renderer/upload.rs` (1089 lines)
- Problem: Ten `SceneRenderer` methods each uploading a *different* GPU resource with no shared state between them: `upload_points` (~59 lines), `upload_frustums` + `update_frustum_colors` + `rebuild_frustum_bind_group` (~296), `upload_thumbnails` (~147), `upload_patches` (~221), `upload_bg_image` (~132), `upload_track_rays` + `clear_track_rays` + `clear_bg_image` (~120). Six distinct concerns glued together only by the word "upload." Sibling `render.rs` already splits the render side per-resource, so this file is the odd one out.
- Proposed fix: regroup under `scene_renderer/upload/` with `{mod, points, frustums, thumbnails, patches, bg_image, track_rays}.rs`, each an `impl SceneRenderer` block (matches the existing `pipelines/` per-resource layout).
- Effort: medium
- Risk: low — pure code-motion of `impl` methods; no signature or logic changes.

**Multi-concern point-track detail panel + 283-line method**
- Location: `crates/sfm-explorer/src/point_track_detail.rs` (1085 lines)
- Problem: Three concerns: (1) the egui panel UI (`show`, `show_header`, `show_observation_table`, `draw_thumbnail`), (2) patch/texture construction (`ensure_rendered_patch`, `build_patch_frame`, `build_stored_patch_texture`), (3) numeric analysis (`compute_observation_metrics`, angle math, `error_color`). `show_observation_table` alone is ~283 lines (292–575), a single method doing per-row layout, hit-testing, hover, and thumbnail draw.
- Proposed fix: split into `point_track_detail/{mod,table,patch,metrics}.rs`; extract the per-row body of `show_observation_table` into a `draw_observation_row` helper.
- Effort: medium
- Risk: medium — `show_observation_table` threads many `&mut` params (`response`, `full_res_cache`, gesture/scroll input); extracting rows requires careful borrow plumbing.

**539-line `show()` god-method in image browser**
- Location: `crates/sfm-explorer/src/image_browser.rs` (735 lines)
- Problem: `ImageBrowser::show` runs 141–680 (~539 lines) — over 70% of the file. It self-documents four distinct phases with banner comments: input handling, animation keyboard controls / frame advance, rendering (thumbnail culling + highlight borders), and navigation minibar. The rest of the file is only tiny helpers.
- Proposed fix: extract private methods `handle_input`, `advance_animation`, `draw_thumbnails`, `draw_minibar` from the marked sections; keep `show` as the orchestrator. Optionally move minibar drawing to a sibling module (mirrors the `image_detail/` split done last audit).
- Effort: medium
- Risk: medium — lots of shared locals (`offset_x`, `thumb_positions`, `response`, gesture events) cross section boundaries; needs a small extracted state struct or explicit params.

**Inline `#[cfg(test)] mod tests` violates repo sibling-`tests.rs` convention**
> _Status (2026-07-18): Done — extracted to
> `crates/sfm-explorer/src/scene_renderer/auto_point_size/tests.rs` (210 lines)
> via `#[cfg(test)] mod tests;`. `cargo test -p sfm-explorer auto_point_size`:
> 10 passed._
- Location: `crates/sfm-explorer/src/scene_renderer/auto_point_size.rs` (lines 182–393, ~211-line inline `mod tests`)
- Problem: The only inline test module in the crate, contradicting the dominant repo convention (sibling `tests.rs` via `mod tests;`, used 20+ places in sfmtool-core).
- Proposed fix: move to `scene_renderer/auto_point_size_tests.rs` (or promote to a dir with `tests.rs`).
- Effort: low
- Risk: low — test-only move; `use super::*;` already present.

**Blanket module-level `#![allow(dead_code)]` masking dead code**
- Location: `crates/sfm-explorer/src/state.rs` (line 6) and `crates/sfm-explorer/src/viewer_3d/mod.rs` (line 9)
- Problem: Both files suppress dead-code warnings for the *entire module* rather than per-item. `state.rs` is the shared-state hub and `viewer_3d/mod.rs` covers the whole 3D viewer subtree — blanket suppression means genuinely-unused fields/methods accumulate invisibly (a clean build emits zero dead-code warnings, confirming everything is hidden). Other files in the crate already use targeted per-item `#[allow(dead_code)]` (`image_browser.rs:34`, `point_track_detail.rs:812`, `image_detail/mod.rs:531`), which is the right pattern.
- Proposed fix: remove the two `#![allow(dead_code)]` lines, then either delete what the compiler flags or move the `#[allow]` onto the individual items legitimately kept-for-API.
- Effort: low
- Risk: low — worst case is re-adding a few targeted attributes; may reveal real dead fields to delete.

**Compass geometry generation living in a GPU-types file (minor)**
- Location: `crates/sfm-explorer/src/scene_renderer/gpu_types.rs` (445 lines)
- Problem: Lines ~12–300 are pure bytemuck layout structs (the file's stated purpose), but `create_compass_edge_instances` (315) and `create_compass_star_mesh` (360) are ~90 lines of procedural *mesh geometry* generation — a different concern (data generation vs data layout).
- Proposed fix: move the two `create_compass_*` functions to a small `scene_renderer/compass.rs` (or into the target-indicator pipeline module), leaving `gpu_types.rs` as pure layout + tiny converters.
- Effort: low
- Risk: low — two `pub(super)` fns; move plus import fix.

> Surveyed and cleared: `app.rs` (735) — last audit's `run_ui_and_paint` decomposition held (four cohesive phase methods). `render.rs` (357) already per-resource. `viewer_3d/camera.rs` (561), `viewer_3d/mod.rs` (649) + `input.rs` (487), `platform/windows.rs` (665) — long but single-purpose. `image_detail/{mod,input,overlay}.rs` split holding. The recently-added patch rendering landed cleanly (`pipelines/patch.rs` + `shaders/patch.wgsl` + one upload method — no smearing). `dock.rs` (400), `colormap.rs` (206) fine.

## Python — `src/sfmtool/`

**`_embed_patches.py` mixes orchestration with shared compaction helpers that another subpackage reaches up to import**
- Location: `src/sfmtool/_embed_patches.py` (915 lines, largest Python file)
- Problem: The file bundles four distinct concerns: (a) generic progress/timing utilities (`_timed_step`, `_progress_poll_loop`, `_poll_progress`, lines 38-97), (b) reconstruction **compaction/hashing** (`compact_to_embedded_patches` lines 148-322 (~175 lines), `image_file_hashes_from_images`, `image_file_hashes_from_sift`, lines 100-146), (c) **subpixel wiring** (`_refine_subpixel`, lines 322-449), and (d) the `embed_patches` **pipeline orchestrator + summary printing** (lines 585-915, ~330 lines). The strongest signal is coupling, not just size: `xform/_localize_keypoints.py` (lines 132-186) reaches *up* into this top-level orchestration module to import `compact_to_embedded_patches` and `image_file_hashes_from_images` — a subpackage op depending on a sibling top-level pipeline driver purely for two utilities. Those compaction helpers are a shared concern, not an `embed_patches` internal.
- Proposed fix: Extract the compaction + hash helpers into a shared module (e.g. `_patch_compaction.py`); both `_embed_patches.py` and `xform/_localize_keypoints.py` import from it. Optionally also lift the 3 progress/timing utils to a small shared helper. Leaves `_embed_patches.py` as orchestration + `_refine_subpixel` (~500 lines, single-purpose) and removes the awkward xform→top-level dependency.
- Effort: medium
- Risk: low — pure move of self-contained functions with two known import sites; no behavior change.

**Strip modules form a coherent cluster that earns a `strips/` subpackage**
- Location: `src/sfmtool/_strip_montage.py` (210), `_patch_ncc.py` (178), `_solve_strips.py` (489), `_compare_strips.py` (479), `_inspect_strips.py` (241) — ~1600 lines across 5 top-level modules
- Problem: These five form a tight, self-contained dependency web with only two external entry points. Internal wiring: `_patch_ncc` and `_strip_montage` are leaves; `_solve_strips` imports `_patch_ncc`; `_compare_strips` imports `_solve_strips` + `_strip_montage` + `_point_correspondence`; `_inspect_strips` imports `_solve_strips` + `_strip_montage`. External consumers are only `_compare.py` and `_commands/inspect.py`. This is the flat-should-be-grouped smell: 5 of the ~24 top-level `_*.py` modules are one feature. Note `_point_correspondence.py` is **not** part of the cluster — it is shared by `_compare.py`, `align/by_points.py`, and the strips — so it stays at top level.
- Proposed fix: Regroup the five modules under `strips/` (e.g. `strips/montage.py`, `strips/patch_ncc.py`, `strips/solve.py`, `strips/compare.py`, `strips/inspect.py`), updating the two external import sites.
- Effort: medium
- Risk: low — internal imports are relative and move together; only two external import lines change. Mechanical.

**`xform/_arg_parser.py` per-op param parsers are borderline co-location candidates (low priority)**
- Location: `src/sfmtool/xform/_arg_parser.py` (785 lines)
- Problem: The module's core concern (ordered `sys.argv` walking that Click's kwargs can't express) is genuinely single-purpose. The growth is four per-op param-string parsers — `parse_refine_normals_params` (123-168), `parse_refine_keypoints_params` (192-237), `parse_localize_keypoints_params` (264-323), `parse_to_embedded_patches_params` (325-368) — each ~50-60 lines, each tightly bound to one transform's private key vocabulary and constructor.
- Proposed fix: Optional — move each `parse_*_params` next to its transform module and have `_arg_parser.py` dispatch to them; or leave as-is (all-parsing-in-one-place is a legitimate reading, and the prior audit cleared `xform/` flatness). Flagged for a decision, not urging a split.
- Effort: low
- Risk: low — the per-op parsers have single call sites in `parse_transform_args`.

> Surveyed and cleared: `_commands/solve.py` (508) — single concern, ~115 lines are option decorators. `colmap/io.py` (873) — previously cleared, shrank. `_workspace_image.py`/`_feature_source.py` — small but correctly single-purpose boundaries. `_global_sfm.py` (127) + `_incremental_sfm.py` (292) — too thin to justify a subpackage. Aside from the strips cluster, the top level is **not** a junk drawer (`_image_pair_graph` alone feeds `analyze/`, `colmap/`, `motion/`, `_densify`). Dead-code scan: 0 TODO/FIXME/XXX, no commented-out blocks, no dead imports.

## Python — `tests/`

**Solve-retry loop duplicated in conftest (carried forward from 2026-06-23 #11) — current status: still present, unchanged**
- Location: `tests/conftest.py` — `build_cluster_reconstruction` (lines 214-253) and `kerry_park_camrig_workspace_once` (lines 517-560)
- Problem: The solve-retry loop is still duplicated. Both carry the identical pattern: `max_attempts=10`, `seed = 42 if attempt == 1 else None` (lines 222 vs 525), clear-colmap-dir + unlink-stale-sfmr preamble (216-219 vs 521-524), best-by-point-count stash/copy tracking (234-239 vs 539-541), the `>= 200` break with a `< 150` hard-fail guard, and the canonicalize-then-unlink-stash tail (245-249 vs 556-559). The only real drift: `build_cluster_reconstruction` ranks by `(image_count, point_count)` tuples while the camrig fixture calls `run_global_sfm` directly and ranks by point_count alone with a `try/except RuntimeError` continue.
- Proposed fix: extract the shared attempt loop into a `_solve_with_retries(*, attempt_fn, expected_count, min_points, max_attempts)` helper taking a per-attempt callable, and have both call sites supply their solve closure.
- Effort: medium
- Risk: medium — both fixtures are session-scoped and feed most of the patch/densify/solve suite; a refactor bug (seed handling, best-tracking) would silently degrade every downstream reconstruction fixture. Needs a full-suite run to validate.

**`test_camrig.py` resolver/pattern block split (carried forward from 2026-06-23 #12) — current status: still present, not split**
> _Status (2026-07-18): Done — lifted the resolver + pattern-matching block (2
> local helpers + 19 tests) into `tests/test_camrig_resolve.py` (712→474 lines).
> The block was not fully self-contained as the finding assumed: it shares
> `_copy_images`/`_camera`/`_pinhole_camera`/`_IMAGE_DATA` with the remaining
> tests, so those went to a scoped `tests/_camrig_helpers.py` imported by both
> (matching the repo's `from .conftest import <helper>` sharing convention).
> `ruff check` clean._
- Location: `tests/test_camrig.py` (712 lines); proposed `tests/test_camrig_resolve.py` does not exist
- Problem: The module still spans ~six areas: spherical-tile↔camrig round-trip (57-155), `write_camrig` binding (166-224), `camrig create` CLI (225-419), `read_camrig` binding (420-443), the resolver block (444-637), and solve integration (682-end). The resolver block is the self-contained candidate: helpers `_touch_images` (444) + `_make_camrig` (455) plus 15 `test_resolve_camrig_*` tests (474-635) and 5 `test_pattern_matches_*` tests (639-681) — ~20 tests / ~240 lines sharing only those two helpers with nothing else in the file.
- Proposed fix: lift lines 444-681 (both helpers + the 20 tests) into `tests/test_camrig_resolve.py`, leaving test_camrig.py at ~470 lines covering create/binding/round-trip/solve.
- Effort: low
- Risk: low — pure move of a self-contained block with local-only helpers; no conftest fixture coupling.

**`test_embed_patches_compaction.py` mixes compaction, multi-round refinement, and progress-utility tests**
- Location: `tests/test_embed_patches_compaction.py` (562 lines)
- Problem: Despite the "compaction" name it holds three concerns: (a) compaction round-trip/culling — `test_compact_*` (5 tests, 85-352); (b) embed-patches multi-round refinement — `test_embed_patches_*` + `test_drop_grazing_observations_*` (4 tests, 353-509); and (c) a progress-instrumentation unit block — `_run_poll_once` (512) plus `test_progress_poll_loop_*` / `test_poll_progress_*` / `test_progress_counter_*` (5 tests, 526-570). The progress block tests a polling utility and shares no fixtures with the rest (they use `seoul_bull_workspace`; the progress tests are pure). A sibling `test_embed_patches_command.py` (415 lines) already exists, so the naming boundary is already muddy.
- Proposed fix: move the progress block (lines 512-570) into a small `tests/test_embed_progress.py` (or fold into the command module); optionally rename the remainder to `test_embed_patches_core.py`.
- Effort: low
- Risk: low — the progress block is helper-isolated and fixture-free.

**tests/ top level has grown enough to warrant a patch/ subdir**
- Location: `tests/` (~53 top-level test modules; existing subdirs `tests/xform/` (19 files) and `tests/rust_bindings/` (18 files) already demonstrate the grouping pattern)
- Problem: The flat top level now carries a large, coherent patch cluster — `test_patch_cloud.py`, `test_oriented_patch.py`, `test_patch_keypoint_localization.py`, `test_patch_keypoint_subpixel.py`, `test_patch_normal_refine.py`, `test_patch_view_selection.py`, `test_render_patches.py`, `test_embed_patches_command.py`, `test_embed_patches_compaction.py` (9 modules, ~3,600 lines) — that would navigate better as a group, mirroring xform/ and rust_bindings/.
- Proposed fix: regroup the patch modules under `tests/patch/` (conftest fixtures are inherited from the parent so no fixture move needed). A camrig/camera group is a weaker secondary candidate.
- Effort: medium
- Risk: low — file moves only; check any hard-coded test paths in CI (`scripts/coverage.sh`, `.github/workflows/ci.yml`).

## Top level — `scripts/`, `specs/`, `reports/`

**Implemented migration doc parked in specs/drafts/**
- Location: `specs/drafts/zup-camera-convention-migration.md`
- Problem: The only file under `specs/drafts/`, but its header reads `**Status**: **IMPLEMENTED** (2026-07-04)`. A drafts/ directory holding a fully-implemented, landed migration is a directory smell — "drafts" implies not-yet-built. The only genuinely open item is a human GUI visual confirmation. (The 2026-07-07 spec audit reached the same conclusion independently and also suggests promoting the convention contract to a stable home.)
- Proposed fix: move to a permanent home (e.g. `specs/migrations/` or alongside `specs/formats/sfmr-file-format.md`'s convention section), or retire it now that the convention is documented in the format specs; leave `specs/drafts/` empty afterward.
- Effort: low
- Risk: low — doc move only; check for inbound links from other specs/reports first.

**reports/2026-06-11-dataset-init-scripts.md is a stale reference snapshot, not a live backlog**
- Location: `reports/2026-06-11-dataset-init-scripts.md`
- Problem: A pure config/results table (script parameters + old-vs-new timings) with zero findings or status annotations — nothing to "mark off." Its numbers already drift from the current tree (it lists `seoul_bull d=28`, but `conftest.build_cluster_reconstruction` now defaults `cluster_d=10`). Per the AGENTS.md retirement policy, it isn't tracking open items. By contrast `2026-06-13-perf-patch-normal-refinement.md` is NOT a retirement candidate: §5 opportunities #2, #4, #5-#7, #10 remain open while only #1 [landed] and #3 (partial) are marked — still doing real backlog work.
- Proposed fix: retire (delete; git preserves history), noting "superseded — reference-only snapshot, no open items" in the commit.
- Effort: low
- Risk: low — deletion of a non-authoritative snapshot; confirm no inbound links first.

> _Status (2026-07-07): Done — retired in the same commit as this report; the only
> inbound links were from the simultaneously-retired 2026-06-23 next-steps snapshot._

**scripts/ has accumulated apparently-orphaned one-off utilities**
- Location: `scripts/` (flat, 20 files)
- Problem: Several scripts are referenced by no spec/report/CI/other file: `exp_plus_descent_localize_compare.py`, `benchmark_advect.py`, `viz_keypoint_localization_strips.py`, `viz_view_selection_strips.py`. They read as one-off experiment/visualization tooling. (By contrast `patch_crossval.py`, `sift_crossval.py`, `solve_crossval.py`, `kdforest_vs_flann.py`, `validate_refine_subset.py`, `bench_normal_refine.py` are cited by specs, and `init_dataset_*.sh` / `coverage.sh` are load-bearing.)
- Proposed fix: confirm with the author and either delete the orphaned experiment/viz scripts or, if kept as living dev tools, move viz utilities under `scripts/viz/` and benchmarks under `scripts/bench/` to make the directory legible.
- Effort: low
- Risk: low — standalone dev scripts with no import consumers; risk is only losing an ad-hoc tool someone still values, hence "confirm first."

> Surveyed and cleared: `test_patch_keypoint_subpixel.py` (612) — single-concern. No duplicated conftest helpers across test modules. `tests/xform/` and `tests/rust_bindings/` well-scoped. Root-level files conventional. `specs/` cli/core/formats/gui/workspace grouping clean; only `drafts/` anomalous.

## Top 3 (best effort-to-value)

1. **Split `patch/keypoint_localize.rs` (2042 lines) into a `keypoint_localize/` module**
   — the largest source file in the workspace, six distinct concerns, in the most
   actively-developed area of the codebase, with a proven in-repo precedent
   (`normal_refine/`) and the sibling dir already existing. Do
   `keypoint_subpixel.rs` in the same pass so the two stay symmetric.
   Medium effort, low risk.
   > _Status (2026-07-08): Done — both files split (`keypoint_localize` 2238→1007,
   > `keypoint_subpixel` 1960→861); see the two findings above._

2. **Batch the mechanical inline-test-convention sweep** — five test-only moves
   clear five findings in one low-risk commit: `sift-format/src/lib.rs` (326-line
   inline suite), `sfmr-colmap/src/colmap_io/mod.rs` (667 inline test lines, larger
   than the module's code), `sfmr-format/src/depth_stats.rs` (153),
   `camrig-format/src/pattern.rs` (115), `sfm-explorer/src/scene_renderer/auto_point_size.rs`
   (211). Add the carried-forward `test_camrig.py` resolver split (#12, also a pure
   move) to the same batch. Low effort, low risk.
   > _Status (2026-07-18): Done — all five inline-test extractions plus the
   > `test_camrig.py` split landed in one commit (the two tiny `verify.rs`/
   > `colmap_io/read.rs` blocks left inline per their "optional" note). See the
   > six findings above._

3. **Extract the shared compaction helpers out of `_embed_patches.py`** — fixes a
   real dependency smell (a `xform/` op reaching up into a top-level pipeline driver
   for `compact_to_embedded_patches` / `image_file_hashes_from_images`) and brings the
   largest Python file back to single-purpose. Medium effort, low risk.

Also queued but needing an owner decision rather than mechanical work: finishing the
`_sfmtool` submodule migration (10 flat names + `import *` remain; touching class
`__module__` paths), the `scene_renderer/upload.rs` regroup, and whether to delete or
regroup the four orphaned `scripts/` utilities.
