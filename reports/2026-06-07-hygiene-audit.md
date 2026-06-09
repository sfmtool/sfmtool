# Codebase Hygiene Audit — 2026-06-07

Read-only structural survey of the sfmtool repository: Python (`src/sfmtool/`,
`tests/`), the Rust workspace (`crates/`), and top-level layout. Totals (approx,
excluding sibling `tests.rs`/test modules where noted): ~21.5k lines Python in
`src/`, ~17.9k lines Python tests, ~74k lines Rust across 8 crates.

This snapshot **consolidates and supersedes** the two prior hygiene reports
(`2026-05-22-hygiene-audit.md` and `2026-05-24-hygiene-audit.md`, both retired in
the same commit). Their resolved findings — the `align/ analyze/ camrig/ merge/
motion/ sift/` package regroupings, the module renames (`_isfm`→`_incremental_sfm`,
`_gsfm`→`_global_sfm`, `_sfm_reconstruction`→`_sfm_filenames`), thinning
`match.py`, and the size-threshold Rust test extraction — are done and drop off.
Their still-open items are carried forward below, re-verified against the current
tree, alongside new findings. Findings are grouped by area and ranked within each;
the Top 3 by effort-to-value are called out at the end.

---

## Python — directory grouping (carried forward from prior audits)

### 1. COLMAP cluster still flat — regroup under `colmap/`
- Location: `src/sfmtool/_colmap_db.py` (860), `_colmap_io.py` (674), `_to_colmap_db.py` (247)
- Problem: ~1,780 lines of COLMAP interop sit flat at the package root as one coherent family — DB setup for the solvers (`_colmap_db`), pycolmap/binary ↔ `.sfmr` conversion (`_colmap_io`: `colmap_binary_to_rust_sfmr`, `pycolmap_to_rust_sfmr`, `save_colmap_binary`), and DB-from-reconstruction export (`_to_colmap_db`). This is the last unbuilt half of prior rec #6 (the `sift/` half landed; `colmap/` did not). All callers are internal (`_incremental_sfm.py`, `_global_sfm.py`, `feature_match/_db_populate.py`).
- Proposed fix: regroup under `colmap/` → `colmap/db_setup.py`, `colmap/io.py`, `colmap/db_export.py`.
- Effort: medium
- Risk: low — purely internal imports; symbols are `_`-prefixed/internal-only, mechanical call-site updates.
> _Status (2026-06-08): Done — built the `colmap/` package: `_colmap_io.py`→`colmap/io.py`, `_to_colmap_db.py`→`colmap/db_export.py` (both via `git mv`), and `_colmap_db.py` split into `colmap/db_setup.py` + `colmap/db_builders.py` (see #2). Updated all 13 internal call sites + 2 test modules; spec references refreshed (this commit)._

### 2. `_colmap_db.py` mixes orchestration with the three DB builders
- Location: `src/sfmtool/_colmap_db.py` (860)
- Problem: The prior "five parallel entry points" flag was partly a false alarm — there is a clean two-tier structure: `_setup_for_sfm` (29) and `_setup_for_sfm_from_matches` (148) are orchestrators that also drive feature matching, dispatching (78-108) to three lower-level builders `_setup_db_single_camera` (487), `_setup_db_with_rigs` (589), `_setup_db_with_camrig` (754). The three builders are genuinely parallel (one per camera-config source) and inherent, not accidental. But at 860 lines the file still mixes orchestration + match-writing (`_write_matches_to_db` 346) with the builders + sensor helpers (`_camera_from_sensor_entry` 553, `_rigid3d_sensor_from_rig` 740).
- Proposed fix: when moving into `colmap/`, split along the existing dispatch boundary — `colmap/db_setup.py` (orchestrators + match writing) and `colmap/db_builders.py` (the three `_setup_db_*` + sensor helpers).
- Effort: medium
- Risk: low — dispatch call sites are all within `_setup_for_sfm`.
> _Status (2026-06-08): Done — `colmap/db_setup.py` (481) holds `_setup_for_sfm`, `_setup_for_sfm_from_matches`, `_write_matches_to_db`; `colmap/db_builders.py` (403) holds the three `_setup_db_*` builders + `_camera_from_sensor_entry` / `_rigid3d_sensor_from_rig`. `db_setup` imports the builders from `db_builders` (this commit)._

### 3. Camera/rig/panorama cluster still flat — regroup under `camera/` + `rig/`
- Location: `_cameras.py` (226), `_camera_setup.py` (261), `_camera_config.py` (169), `_rig_config.py`, `_rig_frames.py` (100), `_insv2rig.py` (320), `_pano2rig.py` (261), `_panorama.py` (304), `_spherical_tile_rig.py` (86) — 9 modules
- Problem: The single largest flat conceptual cluster (prior rec #5, still open). Clean split lines: intrinsics/config/setup (`_cameras` → `get_intrinsic_matrix`, `colmap_camera_from_intrinsics`; `_camera_setup` → `_infer_camera`, `intrinsics_for_image`; `_camera_config` → `CameraConfigResolver`) vs rig ingestion/rendering (`_rig_config`, `_rig_frames`, `_insv2rig` insta360→camrig, `_pano2rig` pano→camrig, `_panorama` equirect render + its tightly-coupled partner `_spherical_tile_rig`). With `camrig/` already a package, ~13 camera/rig units sit flat.
- Proposed fix: `camera/` (`cameras.py`, `setup.py`, `config.py`) and `rig/` (`config.py`, `frames.py`, `insv2rig.py`, `pano2rig.py`, `panorama.py`, `spherical_tile.py`). Keep the `_panorama`/`_spherical_tile_rig` render pair together.
- Effort: medium
- Risk: medium — `CameraConfigResolver` and `_camera_setup` symbols are imported widely (`_colmap_db`, `_incremental_sfm`, `_global_sfm`, `feature_match/`); more call sites than the COLMAP cluster. (`camera/` coexists with Rust `sfmtool_core::camera` — naming only, no conflict.)
> _Status (2026-06-08): Done — built `camera/` (`cameras.py`, `setup.py`, `config.py`) and `rig/` (`config.py`, `frames.py`, `insv2rig.py`, `pano2rig.py`, `panorama.py`, `spherical_tile.py`) via `git mv`; the `panorama`/`spherical_tile` render pair stayed together. Adjusted intra-package imports and ~45 external call sites across src + tests; refreshed spec/AGENTS references. All 122 modules import clean; ruff clean; 259 tests pass across the affected areas (this commit)._

---

## Python — file & name smells

### 4. `xform.py` order-preserving arg parser could live in `xform/` (low value)
- Location: `src/sfmtool/_commands/xform.py` (699)
- Problem: `parse_transform_args` (86-419, ~333 lines) walks `sys.argv` by hand to build the ordered transform pipeline. This is **necessary, not duplication**: `xform` is an ordered pipeline of repeatable, interleaved heterogeneous options (`--rotate … --scale … --rotate …`), and Click's `kwargs` collapses each option into a per-option tuple that loses the cross-option ordering the pipeline depends on — there is no Click-native way to express it. The `@click.option(..., multiple=True)` decorators (427-538) are not redundant either: they provide `--help`, completion, and unknown-option rejection. The only residual hygiene point is file size/locality — the 333-line parser sits in the command module rather than beside the transforms it constructs.
- Proposed fix (optional): move `parse_transform_args` + `parse_angle` + `_auto_output_path` into `xform/_arg_parser.py`, leaving `xform.py` as the Click shell. Purely a relocation; do **not** attempt to "deduplicate" the option names against Click — the two layers serve different purposes.
- Effort: low
- Risk: low — pure relocation, covered by xform tests.
- Note: the `sys.argv.index("xform")` slice (646-650) assumes the subcommand token isn't also an option value — a minor robustness nit, not a structural smell.
> _Status (2026-06-09): Done — moved `parse_transform_args`, `parse_angle`, and `_auto_output_path` (now public `auto_output_path`) into `xform/_arg_parser.py`, leaving `xform.py` as the Click shell (700→~290 lines). No deduplication against Click was attempted, per the finding. The `sys.argv` slice robustness nit is left as-is. Spec note in `xform-select-by-distribution-command.md` refreshed to point at the new location (this commit)._

### 5. `motion/reconstruction.py` — 534-line function with flag logic duplicated in `report.py`
- Location: `src/sfmtool/motion/reconstruction.py` (`analyze_reconstruction`, 420-954) and `motion/report.py` (163-186)
- Problem: `analyze_reconstruction` interleaves per-frame signal computation, threshold/flag determination (629-658), console table rendering (544-588, 662-667), edge clustering/core-edge selection (687-764), and a second console "Summary" pass (792-952). Worse, the per-frame flag rules are computed inline here AND re-derived in `report.py:163-186` — whose own comment admits it must "recompute here so the JSON faithfully mirrors what was printed." Two copies of the same thresholding logic that can silently diverge (this is the prior audit's finding, plus a newly-spotted duplication risk).
- Proposed fix: extract a pure `_flag_frame(...) -> list[str]` (and `_select_core_edges(...)`) called by both the console loop and `report.py`; peel the two console-render passes into `motion/_recon_console.py`, leaving `analyze_reconstruction` compute-only.
- Effort: medium
- Risk: medium — both the printed table and JSON are observable; tests must pin both, and the clustering/tie-break (753-764) is subtle.

### 6. `motion/reconstruction.py` is misnamed for its contents
- Location: `src/sfmtool/motion/reconstruction.py`
- Problem: The name reads as reconstruction data handling, but it is the pose-discontinuity *analysis + console report* for `sfm motion`. Its siblings are descriptively named (`image_sequence.py` for the other input mode, `report.py` for JSON); this is the odd one out.
- Proposed fix: rename → `recon_discontinuity.py` (or `reconstruction_analysis.py`), best done with finding #5. Importers: `motion.py` plus `report.py`'s `from .reconstruction import _rotation_angle_deg`.
- Effort: low
- Risk: low — two internal import sites.
> _Status (2026-06-07): Done — renamed to `motion/recon_discontinuity.py`; updated the four import sites. The finding #5 split (console/JSON flag dedup) is still open (this commit)._

### 7. `_flow_analysis.py` — misleading name, orphaned from its only consumer
- Location: `src/sfmtool/_flow_analysis.py` (194)
- Problem: Name collides conceptually with `feature_match/_flow_matching.py`, but the concerns are distinct: `_flow_analysis` is motion-analysis flow statistics (`_flow_histogram_6x6`, `_flow_tile_means`) used **only** by `motion/image_sequence.py` (line 11), while `_flow_matching` is descriptor matching for the solver. The module sits at the root while its sole consumer is the `motion/` package.
- Proposed fix: move into `motion/` as `motion/flow_stats.py` — disambiguates the name and co-locates with its caller.
- Effort: low
- Risk: low — one import site.
> _Status (2026-06-07): Done — moved to `motion/flow_stats.py`; updated the single import in `motion/image_sequence.py` (this commit)._

### 8. `_sfm_filenames.py` vs `_filenames.py` — confusable names (low priority)
- Location: `_sfm_filenames.py` (116), `_filenames.py` (208)
- Problem: Genuinely distinct (`_filenames` = generic path utils; `_sfm_filenames` = `.sfmr` output naming), correctly separated, but the near-identical names invite confusion — `_incremental_sfm.py` even aliases on import.
- Proposed fix: optional rename `_sfm_filenames.py` → `_sfmr_naming.py`.
- Effort: low · Risk: low
> _Status (2026-06-09): Done — `git mv`'d to `_sfmr_naming.py`; updated the four import sites (`_incremental_sfm.py`, `_commands/epipolar.py`, `visualization/_epipolar_display.py`, `tests/test_epipolar.py`). The `_incremental_sfm.py` import alias is retained but is no longer load-bearing for disambiguation (this commit)._

> Assessed and cleared (no action): `_densify.py` (745), `_undistort_images.py` (595), `_compare.py` (576) are each large but single-purpose pipelines, not mixed concerns. The `_incremental_sfm`/`_global_sfm` pair (~420 lines combined) could form a `solve/` package but the value is marginal today.

---

## Rust — `sfmtool-core`

### 9. `lib.rs` still declares 33 flat `pub mod` — no thematic grouping
- Location: `crates/sfmtool-core/src/lib.rs` (13-45)
- Problem: All 33 modules flat at the crate root (prior rec #7, still open; the count *grew* from 29 as `kdforest`, `sift`, `triangulation`, `point_inspect`, `find_infinity` were added flat). Verified clusters: transforms (`rigid_transform`, `rot_quaternion`, `rotation`, `se3_transform`, `transform`, `viewing_angle` — 6); camera/lens (`camera`, `camera_intrinsics`, `distortion`, `frustum`, `rectification`, `remap`, `warp_map` — 7); spherical (`sphere_points`, `spherical_tile_rig`, `per_spherical_tile_source_stack`, `consensus_atlas` — 4, confirmed cross-referencing). Only `alignment/`, `feature_match/`, `optical_flow/`, `sift/`, `kdforest/` are nested today.
- Proposed fix: regroup into `geometry/`, `camera/`, `spherical/`; the new modules slot near `reconstruction`/`viewing_angle`.
- Effort: medium — mechanical mod moves + `pub use` re-export updates across the workspace (sfmtool-py, sfm-explorer, sfmr-* import `sfmtool_core::<flat>`).
- Risk: medium — external `use sfmtool_core::transform::…` paths break (compiler-caught); lib.rs re-export shims mitigate.

### 10. `find_infinity` + `infinity` — two adjacent modules for one concept
- Location: `find_infinity.rs` (767), `infinity.rs` (574)
- Problem: Both named for "infinity", easy to confuse, living side by side: `infinity.rs` *converts* finite↔infinity tracks; `find_infinity.rs` *discovers* new infinite tracks via sphere clustering (prior rec #7's pairing note).
- Proposed fix: merge into an `infinity/` directory module with `convert` (current `infinity.rs`) and `discover` (current `find_infinity.rs`).
- Effort: low · Risk: low — `find_infinity` is consumed by xform/point-inspect paths only.
> _Status (2026-06-07): Done — merged into `infinity/` (`convert.rs` + `discover.rs` + `mod.rs` re-export shims keep `infinity::*` paths stable; inherent methods unaffected). Inline tests extracted to `infinity/{convert,discover}/tests.rs` (this commit)._

### 11. `distortion.rs` mixes the public projection API with 17 private per-model kernels
- Location: `crates/sfmtool-core/src/distortion.rs` (1633 prod lines)
- Problem: Public coordinate-conversion API (`impl CameraModel` 57-578, `impl CameraIntrinsics` 579-871: `distort`/`undistort`/`project`/`unproject`/`pixel_to_ray`/`best_fit_inside_pinhole`) sits above 17 private model-specific math kernels (872-1632): `distort_opencv`, `distort_fisheye`, `distort_thin_prism_fisheye`, `newton_thin_prism`, `newton_rad_tan_thin_prism`, `blend_fisheye_ray`, etc. The fisheye/thin-prism Newton-solver kernels (1102-1632, ~530 lines) are a self-contained sub-concern.
- Proposed fix: move the private kernels into `distortion/kernels.rs` (or `thin_prism.rs` + `fisheye.rs`), keeping the two public `impl` blocks in `distortion.rs`. (`distortion/tests.rs` already exists.)
- Effort: medium · Risk: low — private, single-crate, compiler-caught.
> _Status (2026-06-08): Done — moved the 21 per-model kernel functions into `distortion/kernels.rs` (as `pub(super)`); `distortion.rs` now holds only the two public `impl` blocks + dispatch and `use kernels::*;` (1634→871 lines, kernels 793). `distortion/tests.rs` unchanged (still reaches the kernels via `super::*`). 67 distortion tests pass (this commit)._

### 12. `reconstruction.rs` bundles data types, file I/O, and edit/subset operations
- Location: `crates/sfmtool-core/src/reconstruction.rs` (1343 prod lines)
- Problem: One `impl SfmrReconstruction` (185-1251) carries three concerns: (a) types + accessors (`Point3D`, `SfmrImage`, `TrackObservation`, `observations_for_point`); (b) file I/O + columnar conversion (`load`, `save`, `from_sfmr_data`, `to_sfmr_data`, `rebuild_derived_fields`); (c) mutating ops (`apply_se3_transform`, `subset_by_image_indices` ~168 lines, `filter_points_by_mask`, `recompute_*`) + `demo` (~180 lines). The subset/filter/transform group (~340 lines) is the cleanest extraction.
- Proposed fix: keep types + accessors + I/O in `reconstruction.rs`; move `subset_by_image_indices`/`filter_points_by_mask`/`apply_se3_transform` (+ helpers) into `reconstruction/edit.rs` as a separate `impl` block. (`reconstruction/tests.rs` already exists.)
- Effort: medium · Risk: low — same-crate impl split; re-exported types stay put.
> _Status (2026-06-08): Done — the three editing methods (+ the `subset_rig_frame_data` helper) now live in a separate `impl SfmrReconstruction` block in `reconstruction/edit.rs` (`use super::*;`); types, accessors, I/O, and the shared `count_points_at_infinity`/`compute_observation_offsets` helpers stay in `reconstruction.rs` (1343→922 lines, edit 436). Methods are inherent so call sites are unchanged; 614 sfmtool-core tests pass (this commit)._

### 13. `optical_flow/gpu/mod.rs` mixes wgpu plumbing, the variational refiner, and DIS orchestration
- Location: `crates/sfmtool-core/src/optical_flow/gpu/mod.rs` (1438 prod lines)
- Problem: Three concerns: (1) wgpu boilerplate — `GpuContext` (88-137), `VariationalBufferPool` (138-321), 8 helper fns + 5 `*Params` POD structs (610-747, 44-87); (2) `GpuVariationalRefiner` (322-609, ~290 lines); (3) `GpuFlowContext` (748-1438, ~690 lines) DIS orchestration. Sibling files `dis_pipeline.rs` (569) and `pyramid_pipeline.rs` (336) already exist, so the orchestration in `mod.rs` duplicates that axis.
- Proposed fix: move boilerplate → `gpu/context.rs`, `GpuVariationalRefiner` → `gpu/variational.rs`, leave/fold `GpuFlowContext` orchestration. (`gpu/tests.rs` exists.)
- Effort: medium-high — shared private types need `pub(super)` threading.
- Risk: medium — GPU code is excluded from `test-rust` llvm-cov and is hardware-gated; regressions can slip past the default test path.

### 14. Inline-test standardization incomplete — 7 large files still embed `mod tests`
- Location: `crates/sfmtool-core/src/` (multiple)
- Problem: The prior size-threshold extraction left a mixed convention. Files >~700 prod lines still keeping tests inline: `sift/scale_space.rs` (1011, ~201 test lines), `optical_flow/mod.rs` (1248, ~291), `feature_match/geometric_filter.rs` (944, ~455 — the largest offender), `optical_flow/dis.rs` (831, ~295), `feature_match/sweep.rs` (818, ~360), `find_infinity.rs` (767, ~254), `remap.rs` (766, ~383). Meanwhile comparably-sized `distortion.rs`, `reconstruction.rs`, `camera_intrinsics.rs`, `spherical_tile_rig.rs`, `feature_match/polar.rs` use sibling `tests.rs` — two conventions coexist even within `feature_match/` (polar sibling vs geometric_filter/sweep inline).
- Proposed fix: extract the inline `mod tests` from the 7 files into sibling `tests.rs`, matching the established convention.
- Effort: low — mechanical per file.
- Risk: low — test-only; `use super::*` access preserved by the sibling convention already in use.
> _Status (2026-06-07): Done — all 7 extracted to sibling `tests.rs` (`find_infinity` handled as part of #10's merge). Two-convention inconsistency within `feature_match/` resolved. `cargo test -p sfmtool-core` 614 passed (this commit)._

---

## Rust — `sfmtool-py`

### 15. Everything registers on one flat module — no PyO3 submodules
- Location: `crates/sfmtool-py/src/lib.rs` + the ~32 flat `py_*.rs` files
- Problem: `add_submodule` count = **0**; `#[pymodule] _sfmtool` makes ~60 `add_function` + ~14 `add_class` (74 names) onto a single `m`, and `__init__.py` does `from sfmtool._sfmtool import *`, dumping 74 names flat into `sfmtool`'s top level (prior rec #8, still open; file count grew 30→32 with `py_kdforest.rs` + split `py_sift*.rs`). Clear filename groups exist: I/O (`py_sfmr_io`, `py_sift_io`, `py_matches_io`, `py_camrig_io`, `py_colmap_binary`, `py_colmap_db`), matching (`py_descriptor_match`, `py_image_match`, `py_sweep_match`), geometry (`py_rigid_transform`, `py_rot_quaternion`, `py_se3_transform`, `py_camera_intrinsics`, `py_sphere_points`), flow (`py_optical_flow`, `py_warp_map`).
- Proposed fix: expose child modules via `add_submodule` (`_sfmtool.io`/`.match`/`.geometry`/`.flow`), group the `py_*.rs` into subdirs, replace `import *` with deliberate re-exports.
- Effort: high
- Risk: medium — all `import *` consumers + 74 call sites move; PyO3 submodules have `sys.modules` sharp edges (child modules need manual `sys.modules` insertion to import as `sfmtool._sfmtool.io`). This is the only flatness that reaches the public Python API.

### 16. `py_sfmr_reconstruction.rs` dominated by one ~500-line method
- Location: `crates/sfmtool-py/src/py_sfmr_reconstruction.rs` (1304)
- Problem: One `#[pymethods] impl` with ~45 methods (mostly small accessors), but `clone_with_changes` (796-1296, ~500 lines, 38% of the file) hand-extracts every mutable field from `kwargs` and defines two near-identical local `macro_rules!` (`extract_array1`/`extract_array2`, 807-854). The smell is one oversized method + duplicated extraction macros, not heterogeneous concerns.
- Proposed fix: extract the kwargs/array parsing into a private `recon_clone.rs` (a `CloneChanges` struct + `from_kwargs`); hoist `extract_array1/2` into a shared `helpers.rs`.
- Effort: medium · Risk: low — `clone_with_changes`'s Python signature is unchanged; covered by `*_rust_bindings.py`.

---

## Rust — `sfm-explorer`

### 17. `image_detail.rs::show()` is a ~626-line method spanning input, overlays, rendering
- Location: `crates/sfm-explorer/src/image_detail.rs` (1190)
- Problem: `ImageDetail::show` (125-751, 11 parameters) sequentially handles image fit/placement, input handling (drag-pan/scroll/pinch + Windows DirectManipulation `GestureEvent`, ~236-370), and feature overlays (the `overlay_mode` match calling `draw_feature_ellipse`). Below it sit 8 free helpers (`draw_feature_ellipse`, `find_nearest_tracked_feature`, `feature_size`, `compute_error_range`, `log10_condition`, `compute_finite_value_range`, `compute_max_track_angle_deg`, `compute_track_length_range`) — a self-contained feature-metric/coloring cluster.
- Proposed fix: extract input handling → `image_detail/input.rs` (mirroring `viewer_3d/input.rs`) and the overlay drawing + 8 metric helpers → `image_detail/features.rs`, leaving `show` as orchestration.
- Effort: medium · Risk: low — intra-crate GUI refactor, no public API.

### 18. Top-level GUI panel/shell files form a flat cluster
- Location: `crates/sfm-explorer/src/` — `image_detail.rs` (1190), `image_browser.rs` (735), `point_track_detail.rs` (839), `dock.rs` (362), `app.rs` (620), `state.rs` (315), `colormap.rs` (206)
- Problem: `lib.rs` (15-24) declares 10 sibling modules with no grouping, mixing app shell (`app`/`state`/`dock`), the three dockable panels, and a utility (`colormap`) at the same level as the well-organized `scene_renderer/`, `viewer_3d/`, `platform/` subtrees. The three panels are clearly a family (peers in `app.rs:350-353`, routed through `dock::TabContext`, each with the same `new`/`show`/`clear` + `*Response` shape).
- Proposed fix: introduce `panels/` (`image_detail`, `image_browser`, `point_track_detail`), optionally an `app/` shell module for `app`/`state`/`dock`.
- Effort: low · Risk: low — intra-crate moves; `sfm-explorer` is a binary so no public API.

> The `scene_renderer/` (+ `pipelines/`), `viewer_3d/`, and `platform/` subtrees are well-factored and should not be touched. `scene_renderer/upload.rs` (847) and `point_track_detail.rs` (839) are large but cohesive.

---

## Tests

### 19. Inconsistent `test_cli_*` vs flat naming, plus a duplicate `to_nerfstudio` pair
- Location: `tests/` (top level)
- Problem: 11 `test_cli_*.py` modules (align, match, solve, xform, analyze, inspect, motion, sift_extract, sift_workspace, to_nerfstudio, ws_init) coexist with ~8 equally CLI-driven flat modules (`test_compare`, `test_heatmap`, `test_merge`, `test_pano2rig`, `test_undistort` 678, `test_panorama`, `test_densify` 686, `test_camrig` 1015) with no rule distinguishing them — the flat files contain `TestXxxCLI`/`TestXxxE2E` classes too. Direct collision: both `test_to_nerfstudio.py` (93) and `test_cli_to_nerfstudio.py` (303) exist for one feature.
- Proposed fix: drop the `test_cli_` prefix on the 11 modules (matching the flat majority) and merge the two `to_nerfstudio` files.
- Effort: low (renames + one merge)
- Risk: low — glob-based discovery; only explicit `pixi run test -- tests/test_cli_x.py` invocations in docs/scripts would need updating.
> _Status (2026-06-07): Done — dropped the `test_cli_` prefix on all 10 remaining modules and merged the `to_nerfstudio` pair into `test_to_nerfstudio.py` (this commit)._

### 20. `*_rust_bindings.py` — coherent cluster suitable for `tests/rust_bindings/`
- Location: 9 modules (`test_descriptor_`, `test_distortion_`, `test_kdtree_forest_`, `test_kdtree_`, `test_range_expr_`, `test_rot_quaternion_`, `test_sfmr_reconstruction_`, `test_sift_extract_`, `test_triangulation_rust_bindings.py`) — ~1,573 lines
- Problem: All exercise the `sfmtool._sfmtool` PyO3 surface, share a uniform suffix, and are the tests most sensitive to a stale `.so` (the `maturin develop` gotcha). `tests/xform/` already sets the grouping precedent.
- Proposed fix: regroup into `tests/rust_bindings/` (with `__init__.py`); enables a targeted `pixi run test -- tests/rust_bindings` after `maturin develop`.
- Effort: low · Risk: low — `conftest.py` still applies via upward discovery.
> _Status (2026-06-07): Done — all 9 moved into `tests/rust_bindings/` with `__init__.py` (this commit)._

### 21. `test_camrig.py` (1015) and `test_camera_config.py` (766) mix several concerns
- Location: `tests/test_camrig.py`, `tests/test_camera_config.py`
- Problem: `test_camrig.py` (59 tests) spans format round-trip, spherical-tile CLI, PyO3 bindings, `camrig create` CLI, resolver, pattern matching, AND a ~300-line `cam cp` block testing a *different* command. `test_camera_config.py` is mostly `_camera_config` unit tests but its tail (607-766) is end-to-end `--camera-model` rejection tests across solve/match/to-colmap-db.
- Proposed fix: extract the `cam cp` tests → `test_cam_cp.py` (and optionally resolver/pattern → `test_camrig_resolve.py`); move the four cross-command CLI rejection tests out of `test_camera_config.py` into the respective command tests or `test_camera_config_cli.py`.
- Effort: low–medium · Risk: low — shared fixtures come from `conftest.py`, no duplication.

> The format crates (`camrig-format`, `matches-format`, `sfmr-format`, `sift-format`, `sfmr-colmap`) are **clean** — consistent `read`/`write`/`types`/`verify`/`archive_io` decomposition, no oversized non-test file, no misleading names, no dead code.

---

## Top 3 (best effort-to-value)

1. **Finish the Rust inline-test standardization (#14).** Seven `sfmtool-core`
   files over ~700 lines still embed `mod tests` while their peers use sibling
   `tests.rs` — a mechanical, test-only extraction (low/low) that completes a
   half-done prior effort, removes a two-convention inconsistency *within the same
   directory* (`feature_match/`), and makes the largest "files" honest about
   production size (`geometric_filter.rs` 944→~489, `remap.rs` 766→~383).
   > _Status (2026-06-07): Done — see #14 (this commit)._

2. **Build the `colmap/` package (#1, with the #2 split).** The last unbuilt
   package-grouping from the prior audits — medium effort, low risk, all-internal
   callers — that retires ~1,780 flat lines into a coherent subpackage and splits
   the 860-line `_colmap_db.py` along its existing orchestrator/builder seam. The
   `camera/`+`rig/` regroup (#3) is the natural, slightly higher-risk follow-on.
   > _Status (2026-06-08): Done — see #1 and #2 (this commit). The `camera/`+`rig/`
   > regroup (#3) remains the natural follow-on._

3. **Tidy `tests/` naming and grouping (#19 + #20).** Drop the inconsistent
   `test_cli_` prefix, merge the duplicate `to_nerfstudio` pair (a real
   collision), and move the 9 `*_rust_bindings.py` into `tests/rust_bindings/`.
   Pure low-risk navigability win that also gives a clean target for the
   post-`maturin-develop` binding tests.
   > _Status (2026-06-07): Done — see #19 and #20 (this commit)._

> Carried-forward larger items still open and worth scheduling beyond the Top 3:
> the `sfmtool-core` `geometry/`/`camera/`/`spherical/` regroup (#9) and the
> `sfmtool-py` PyO3 submodule restructure (#15) — the latter being the only
> flatness that reaches the public Python API, and the highest-effort/highest-risk
> item in this report.
