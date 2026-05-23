# Codebase Hygiene Audit — 2026-05-22

Read-only structural survey of the sfmtool repository: Python (`src/sfmtool/`,
`tests/`), Rust crates (`crates/`), and top-level layout (`scripts/`, `specs/`,
`docs/`). Totals: ~25.5k lines Python in `src/`, ~16.6k lines Python tests,
~66.8k lines Rust.

Method: per-directory line counts, then structural skim of every file over
~500 lines plus all file-name clusters, counting distinct concerns and
test-vs-production splits. Findings below are ranked roughly by effort-to-value;
the Top 3 are called out at the end.

---

## Recommendations

### 1. Inconsistent Rust unit-test placement across `sfmtool-core`

- Location: `crates/sfmtool-core/src/` (multiple files)
- Problem: The crate uses **two** conventions for unit tests with no clear rule.
  Seven modules extract tests to a sibling `module/tests.rs`
  (`distortion/tests.rs` 1493 lines, `per_spherical_tile_source_stack/tests.rs`
  1324, `spherical_tile_rig/tests.rs` 1078, plus `consensus_atlas`,
  `photometric_ransac`, `se3_transform`, `spatial`). Meanwhile the largest
  source files keep tests inline, which inflates their apparent size and buries
  production code:
  - `optical_flow/gpu/mod.rs` — 2071 lines, of which ~634 (lines 1437–2071) are
    inline tests; real code ~1437.
  - `camera_intrinsics.rs` — 1596 lines, ~692 inline tests (904–1596); real
    code ~904.
  - `feature_match/polar.rs` — 1280 lines, ~437 inline tests (843–1280).
  - `warp_map.rs` — 1092 lines, ~607 inline tests (485–1092); real code ~485.
  - `epipolar.rs` — 893 lines, ~473 inline tests (420–893).
  - `frustum.rs` — 902 lines, ~460 inline tests (442–902).
  - `reconstruction.rs` — 1463 lines, ~149 inline tests (1314–1463).
  The mixed convention makes "which files are actually big?" hard to answer and
  forces readers of production code to scroll past test code (or vice-versa).
- Proposed fix: Pick one convention. Given the existing `module/tests.rs`
  pattern, extract the inline `#[cfg(test)] mod tests` blocks from the files
  above into sibling `tests.rs` files (`camera_intrinsics/tests.rs`,
  `feature_match/polar` would need restructuring, etc.). Mechanical move; tests
  already `use super::*`.
- Effort: medium (several files, but pure cut/paste plus a `mod tests;` line)
- Risk: low — no behavior change; `cargo test` validates immediately. Watch for
  test-only helpers that reference private items (they remain accessible via
  `use super::*`).

### 2. `_commands/match.py` mixes CLI wiring with matching orchestration

- Location: `src/sfmtool/_commands/match.py` (940 lines)
- Problem: This is by far the heaviest command module (next is `xform.py` at
  600; most commands are 90–250 lines and delegate to a sibling implementation
  module). Beyond the `@click` command, it carries ~700 lines of business
  logic in 8 helpers: `_run_matching` (216), `_populate_db_features` (383),
  `_compute_descriptor_distances` (426), `_fill_sift_hashes` (466),
  `_generate_output_path` (486), `_run_flow_matching` (531), `_run_merge` (606).
  This is database population, descriptor-distance computation, flow-matching
  orchestration, and merge — none of which is CLI concern. The package already
  has a `feature_match/` subpackage (`_core.py`, `_flow_matching.py`, etc.) that
  is the natural home, and other commands (`solve`, `densify`, `undistort`)
  follow the thin-command pattern.
- Proposed fix: Move the orchestration helpers into `feature_match/` (e.g. a new
  `feature_match/_run.py` for `_run_matching`/`_run_flow_matching`/`_run_merge`
  and `feature_match/_db_populate.py` for the DB/descriptor helpers), leaving
  `match.py` as a thin Click wrapper like its peers.
- Effort: medium
- Risk: medium — these helpers touch pycolmap DB state and the matches-file
  output path; `tests/test_cli_match.py` and `tests/test_colmap_interop.py`
  cover the surface, run them after the move.

### 3. `analyze_reconstruction` is one 534-line function mixing signal computation and report rendering

- Location: `src/sfmtool/_discontinuity_reconstruction.py` (954 lines), function
  `analyze_reconstruction` (lines 420–954)
- Problem: The module's top-level helpers (`_compute_extrapolation_errors`,
  `_compute_step_ratios`, `_compute_overlap_drops`, `_compute_obs_z_scores`,
  etc.) are clean and single-purpose, but the entry point bundles everything
  into one 534-line function that both computes the discontinuity signals AND
  renders the full text report (sequence detection, the "Summary section" table
  at internal line ~373, and the "Footer" confidence-partitioning logic at
  ~498). Computation and presentation are tangled in a single scope, and the
  table-formatting helpers (`return "<{s}>" if ...`) are nested closures.
- Proposed fix: Split into a compute layer that returns structured per-sequence
  results and a render layer that formats them. The structured results already
  exist (`all_sequence_results`); extract the summary-table and footer rendering
  into module-level functions (or move rendering alongside `_discontinuity_json.py`
  which already serializes the same results).
- Effort: medium
- Risk: low–medium — text output is asserted in `tests/test_cli_discontinuity.py`;
  keep the rendered string byte-identical or update the test.

### 4. Flat `src/sfmtool/` root: 52 modules with obvious unused subpackage clusters

- Location: `src/sfmtool/` (52 top-level `*.py` modules)
- Problem: The package already demonstrates the subpackage pattern
  (`feature_match/`, `xform/`, `visualization/`), but the root has grown to 52
  flat modules containing several tight clusters that read like they want to be
  packages:
  - `_align.py`, `_align_by_cameras.py`, `_align_by_points.py`, `_multi_align.py`
    (alignment).
  - `_analyze_depth.py`, `_analyze_graphs.py`, `_analyze_images.py`,
    `_analyze_metrics.py` (all back `sfm analyze`; `_analyze_images.py` is 501
    lines).
  - `_camrig_cp.py`, `_camrig_create.py`, `_camrig_pattern.py`,
    `_camrig_resolver.py` (`.camrig` handling).
  - `_discontinuity_constants.py`, `_discontinuity_image_sequence.py`,
    `_discontinuity_json.py`, `_discontinuity_reconstruction.py`.
  - `_merge.py`, `_merge_correspondences.py`, `_merge_pose_refinement.py`.
  - `_colmap_db.py`, `_colmap_io.py`, `_extract_sift_colmap.py`, `_to_colmap_db.py`.
- Proposed fix: Group the clear clusters under subpackages — `align/`,
  `analyze/`, `camrig/`, `discontinuity/`, `merge/`, `colmap/` — mirroring the
  existing `feature_match/`/`xform/`/`visualization/` style. Do it cluster by
  cluster, not all at once.
- Effort: high (many import-path updates across `_commands/` and tests)
- Risk: medium — private modules so no external API breakage, but lots of
  relative imports to fix; do one cluster per commit and lean on the test suite.

### 5. Misleading name: `_sfm_reconstruction.py` is filename-generation utilities

- Location: `src/sfmtool/_sfm_reconstruction.py` (about 90 lines)
- Problem: The name reads as "the SfM reconstruction model/type" (and there *is*
  an `SfmrReconstruction` type in Rust/bindings), but the file only contains
  output-filename helpers: `get_next_sfm_filename`, `_generate_image_descriptor`,
  `get_image_hint_message`. A reader looking for reconstruction logic lands here
  and finds string/path utilities.
- Proposed fix: Rename to `_sfm_filenames.py` (compare the existing
  `_filenames.py`) or fold into `_filenames.py` if the concerns align.
- Effort: low
- Risk: low — rename plus import updates; private module.

### 6. Cryptic abbreviated module names `_isfm.py` / `_gsfm.py`

- Location: `src/sfmtool/_isfm.py`, `src/sfmtool/_gsfm.py`
- Problem: The names are non-obvious abbreviations (incremental SfM via COLMAP /
  global SfM via GLOMAP). They are imported lazily inside `solve.py`
  (`from .._isfm import run_incremental_sfm`, `from .._gsfm import
  run_global_sfm`). The abbreviations don't match the descriptive naming used
  elsewhere in the package.
- Proposed fix: Rename to `_incremental_sfm.py` / `_global_sfm.py`. If a `solve/`
  or reconstruction subpackage is created (see #4) they belong there.
- Effort: low
- Risk: low — two import sites (`solve.py`, and `_gsfm.py` imports a helper from
  `_isfm.py`).

### 7. `visualization/_flow_display.py` bundles flow rendering, two display modes, legend, and I/O

- Location: `src/sfmtool/visualization/_flow_display.py` (722 lines)
- Problem: Largest visualization module. `draw_flow_visualization` (lines
  143–393, ~250 lines) plus two private renderers `_draw_flow_only_mode`
  (393) and `_draw_comparison_mode` (468), color mapping (`_flow_to_color`,
  `_direction_color_bgr`), legend drawing (`_draw_flow_legend`), and file
  output (`_save_flow_color_image`, `_save_output`) all live together. The
  sibling `_epipolar_display.py` (641) is comparably structured but tighter; the
  flow file mixes a large dispatch function with two full rendering modes and
  save helpers.
- Proposed fix: Extract the color/legend helpers into a small
  `_flow_color.py` and consider separating the two display modes; keep
  `draw_flow_visualization` as the dispatcher.
- Effort: medium
- Risk: low — image output is golden-compared in `tests/test_flow.py`; keep
  pixel output identical.

### 8. `_colmap_db.py` setup-path proliferation

- Location: `src/sfmtool/_colmap_db.py` (860 lines)
- Problem: Holds five parallel DB-setup entry points that branch by camera
  configuration — `_setup_for_sfm` (29), `_setup_for_sfm_from_matches` (148),
  `_setup_db_single_camera` (487), `_setup_db_with_rigs` (589),
  `_setup_db_with_camrig` (754) — plus `_write_matches_to_db` (346). These share
  significant structure (camera/sensor construction, index remapping) and the
  file reads as several near-parallel procedures rather than one concern.
- Proposed fix: Factor the shared camera/sensor/remap construction into helpers
  (some already exist: `_camera_from_sensor_entry`, `_rigid3d_sensor_from_rig`)
  and consider splitting the single-camera vs rig vs camrig setup paths into a
  `colmap/` subpackage (ties into #4).
- Effort: medium
- Risk: medium — central to `solve`/`match`/`to-colmap-db`; covered by
  `tests/test_colmap_interop.py` and CLI tests.

### 9. `optical_flow/gpu/mod.rs` is a 2071-line module gateway

- Location: `crates/sfmtool-core/src/optical_flow/gpu/mod.rs`
- Problem: Even discounting the ~634 inline test lines (see #1), the module is
  ~1437 production lines holding multiple GPU concerns in one file: uniform
  param structs (`WarpParams`/`CoeffParams`/`JacobiParams`/`UpdateParams`/
  `UpsampleParams`), `GpuContext`, `VariationalBufferPool` (170–322),
  `GpuVariationalRefiner` (322–610, the refinement loop), a block of
  bind-group/pipeline factory free functions (610–748), and `GpuFlowContext`
  (748–1437, ~690 lines). It already has submodules (`dis_pipeline`,
  `pyramid_pipeline`), so the splitting pattern exists.
- Proposed fix: Move `VariationalBufferPool` + `GpuVariationalRefiner` into a
  `variational_gpu.rs` submodule and the bind-group/pipeline factory helpers
  into a `wgpu_util.rs` submodule, leaving `mod.rs` as the `GpuContext` /
  `GpuFlowContext` orchestration surface.
- Effort: medium–high
- Risk: medium — GPU pipeline code is exercised by `tests/test_flow.py` and the
  Rust GPU tests; `pixi run cargo test --workspace` (gpu excluded from llvm-cov)
  is the gate.

### 10. `_camera_setup.py` vs `_cameras.py` vs `_camera_config.py` — overlapping camera-utility names

- Location: `src/sfmtool/_camera_setup.py` (261), `src/sfmtool/_cameras.py`
  (226), `src/sfmtool/_camera_config.py` (169)
- Problem: Three similarly-named root modules cover adjacent concerns —
  "Camera model inference and descriptor wrapping for COLMAP database setup"
  (`_camera_setup`), "Camera conversion utilities for pycolmap interop"
  (`_cameras`), and `camera_config.json` resolution (`_camera_config`). The
  names don't make the boundary obvious; `_cameras.py` vs `_camera_setup.py` in
  particular both relate to pycolmap camera construction.
- Proposed fix: Either merge `_cameras.py` into `_camera_setup.py` if the
  pycolmap-conversion concern is shared, or rename for a clearer split (e.g.
  `_pycolmap_cameras.py`). Group with `_colmap_db.py` under a `colmap/`
  subpackage per #4.
- Effort: low–medium
- Risk: low — verify all import sites; widely used so run the full Python suite.

### 11. CLI-test file naming is inconsistent (`test_cli_*` vs flat)

- Location: `tests/` (top level)
- Problem: Some command tests use the `test_cli_<cmd>.py` prefix (11 files:
  `test_cli_align`, `test_cli_match`, `test_cli_solve`, `test_cli_xform`, …)
  while other command/feature tests are flat (`test_compare.py`,
  `test_heatmap.py`, `test_merge.py`, `test_pano2rig.py`, `test_undistort.py`).
  Meanwhile `tests/xform/` is a proper subdirectory for the xform transforms.
  There's no consistent rule for when a test gets the `cli` prefix or a subdir.
- Proposed fix: Adopt one convention — either prefix all command-level tests
  `test_cli_*` or none, and decide whether the rig/camrig and discontinuity
  test groups deserve subdirectories like `tests/xform/`. Low urgency, mostly a
  navigability win.
- Effort: low (renames) to medium (subdir reorg)
- Risk: low — test-only; pytest discovers by pattern regardless.

### 12. `_extract_sift_colmap.py` / `_extract_sift_opencv.py` split vs `_sift_file.py`

- Location: `src/sfmtool/_extract_sift_colmap.py` (245),
  `src/sfmtool/_extract_sift_opencv.py`, `src/sfmtool/_sift_file.py` (753)
- Problem: SIFT extraction is split by backend into two `_extract_sift_*` files
  while `_sift_file.py` (the read/write format module, 753 lines) already
  contains `image_files_to_sift_files` AND `image_files_to_sift_files_opencv`
  plus `draw_sift_features`. The extraction logic spans three files with the
  backend boundary cutting across them, and `_sift_file.py` itself mixes the
  on-disk format I/O with extraction-orchestration entry points.
- Proposed fix: Group the SIFT concern under a `sift/` subpackage
  (`sift/format.py` for read/write, `sift/extract_colmap.py`,
  `sift/extract_opencv.py`), pulling the extraction entry points out of
  `_sift_file.py` so it is purely the file format.
- Effort: medium
- Risk: low–medium — `tests/test_sift_file.py` (711) and the `test_cli_sift_*`
  tests cover this; many import sites for `image_files_to_sift_files`.

---

## Top 3 (best effort-to-value)

1. **Standardize Rust unit-test placement (#1).** A mostly-mechanical
   cut/paste, validated instantly by `cargo test`, that immediately makes the
   biggest "files" in the core crate honest about their production size
   (camera_intrinsics 1596→~904, gpu/mod 2071→~1437, warp_map 1092→~485) and
   removes a confusing two-convention inconsistency. Highest value per unit
   risk.

2. **Thin out `_commands/match.py` (#2).** Moving ~700 lines of matching/DB
   orchestration into the existing `feature_match/` subpackage makes the
   heaviest command module match its lightweight peers and puts logic next to
   the code it belongs with. Medium effort, well covered by existing CLI tests.

3. **Rename `_sfm_reconstruction.py` and `_isfm.py`/`_gsfm.py` (#5 + #6).** Three
   cheap renames that kill genuinely misleading/cryptic names with only a handful
   of internal import sites and near-zero risk — pure clarity wins.
