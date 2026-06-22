# Codebase Hygiene Audit — 2026-06-09

Read-only structural survey of the sfmtool repository: Python (`src/sfmtool/`,
`tests/`), the Rust workspace (`crates/`), and top-level layout. Totals (approx):
~21.7k lines Python in `src/`, ~18k lines Python tests, ~76k lines Rust across
8 crates.

This snapshot **supersedes `2026-06-07-hygiene-audit.md`** (retired in the same
commit). That report's findings were almost entirely actioned in the intervening
two days (commits #49–#54): the `colmap/`, `camera/`, and `rig/` package
regroupings, the `motion/` dedup + console split, the `distortion.rs` /
`reconstruction.rs` / `infinity/` Rust splits, the first inline-test extraction
pass, the `xform/_arg_parser.py` extraction, the `_sfmr_naming.py` rename, and
the full `tests/` naming + `tests/rust_bindings/` regroup. Its still-open items
(#9, #13, #15–#18, #21) are carried forward below, re-verified against the
current tree, alongside new findings.

The headline of this run: the Python package is now **clean** — a thorough
re-survey of `src/sfmtool/` (largest files, flat root modules, `_commands/`
wrappers, dead-code sweep) produced no structural findings beyond one trivial
duplicated helper. The remaining work is concentrated in the Rust crates.

---

## Python

> **Surveyed and cleared (no findings):** `sift/file.py` (855 — coherent SIFT
> I/O + extraction, cleanly sectioned), `motion/recon_discontinuity.py` (799 —
> now compute-only after the #54 split; `_recon_console.py` holds all
> rendering), `visualization/_flow_display.py` (722) and `_epipolar_display.py`
> (641 — both single-purpose visualization), `feature_match/_run.py` (650 —
> pure orchestration; `_commands/match.py` is a thin 212-line wrapper),
> `analyze/summary.py` (623) and `analyze/images.py` (501), `_commands/solve.py`
> (521) and `_commands/camrig.py` (383 — both thin Click wrappers delegating to
> library modules). The remaining flat root modules (`_densify`, `_undistort_images`,
> `_compare`, `_incremental_sfm`/`_global_sfm`, and the small utilities) are
> each single-purpose; the incremental/global SfM pair are alternative solvers,
> not a package candidate. No dead modules, no `*_old` leftovers, no
> commented-out blocks found.

### 1. `_get_color_palette()` duplicated across two visualization modules
- Location: `src/sfmtool/visualization/_flow_display.py:57` and `visualization/_epipolar_display.py:21`
- Problem: Near-identical palette helpers (trivial differences: docstring style; `max(n_colors, 1)` vs `n_colors` edge-case handling) — the only duplication the Python sweep turned up.
- Proposed fix: extract into a shared `visualization/_common.py` (or fold into an existing utility module) and import from both.
- Effort: low
- Risk: low — two internal call sites.
> _Status (2026-06-22): Done — extracted to `visualization/_common.py::get_color_palette`
> (with the more defensive `max(n_colors, 1)` divisor); both modules now import it._

---

## Rust — `sfmtool-core`

### 2. `lib.rs` still declares 32 flat `pub mod` — no thematic grouping (carried forward, #9)
- Location: `crates/sfmtool-core/src/lib.rs:13-44`
- Problem: 32 modules flat at the crate root (down one from 33 after the `infinity/` merge; otherwise untouched). The verified clusters still hold: **geometry** (`rigid_transform` 283, `rot_quaternion` 432, `rotation` 204, `se3_transform`, `transform` 413, `viewing_angle` 320 — 6 modules); **camera/lens** (`camera` 248, `camera_intrinsics` 905, `distortion` 871, `frustum` 443, `rectification` 585, `remap` 384, `warp_map` 486 — 7); **spherical** (`sphere_points` 345, `spherical_tile_rig` 1060, `per_spherical_tile_source_stack` 862, `consensus_atlas` — 4). Only `alignment/`, `feature_match/`, `optical_flow/`, `sift/`, `kdforest/`, `infinity/` are nested today.
- Proposed fix: regroup into `geometry/`, `camera/`, `spherical/` with `lib.rs` re-export shims for a transition period.
- Effort: medium — mechanical mod moves + path updates across the workspace (`sfmtool-py`, `sfm-explorer`, `sfmr-*` import `sfmtool_core::<flat>`).
- Risk: medium — external `use sfmtool_core::transform::…` paths break (compiler-caught); re-export shims mitigate.

### 3. `optical_flow/gpu/mod.rs` mixes wgpu plumbing, the variational refiner, and DIS orchestration (carried forward, #13)
- Location: `crates/sfmtool-core/src/optical_flow/gpu/mod.rs` (1438 lines, unchanged)
- Problem: Re-verified; three concerns remain tangled: (1) wgpu boilerplate — 5 `*Params` POD structs (44-85), `GpuContext` (88-137), `VariationalBufferPool` (138-321); (2) `GpuVariationalRefiner` (322-747); (3) `GpuFlowContext` DIS orchestration (748-1438, ~690 lines). Sibling `dis_pipeline.rs` (569) and `pyramid_pipeline.rs` (336) already exist, so the orchestration in `mod.rs` duplicates that axis.
- Proposed fix: extract `gpu/context.rs` (GpuContext + buffer pool + params) and `gpu/variational.rs` (`GpuVariationalRefiner`), keeping `GpuFlowContext` in `mod.rs` with re-exports. Mind the `include_str!` shader paths when moving code.
- Effort: medium-high — shared private types need `pub(super)` threading.
- Risk: medium — GPU code is excluded from `test-rust` llvm-cov and is hardware-gated; regressions can slip past the default test path.

### 4. Inline-test extraction stragglers below the prior ~700-line threshold
- Location: `crates/sfmtool-core/src/` (8 files)
- Problem: The 2026-06-07 extraction pass converted everything over ~700 prod lines to sibling `tests.rs`, but a band of mid-size files still embeds `mod tests`, several test-dominated: `rectification.rs` (585 total, **356 test lines — 61% of the file**), `triangulation.rs` (557, 255 — 46%), `image_pair_graph.rs` (656, 264 — 40%), `sift/mod.rs` (641, 209), `sift/descriptor.rs` (644, 142), `sift/detect.rs` (566, 97), `spherical_tile_rig/camrig.rs` (542, 159), `optical_flow/variational.rs` (824, 61 — the one file actually above the prior threshold, missed in that pass). Two conventions again coexist within `sift/` (scale_space sibling vs mod/descriptor/detect inline).
- Proposed fix: extract the inline `mod tests` into sibling `tests.rs` for the 8 files, matching the established convention. (Genuinely small files — `camera.rs` 248, `rotation.rs` 204, etc. — are fine inline; no need to chase those.)
- Effort: low — mechanical per file.
- Risk: low — test-only.
> _Status (2026-06-10): Done — all 8 extracted to sibling `tests.rs` following the `distortion.rs` convention (net −1545 inline lines). Test counts identical before/after (614 passed, 2 ignored); `cargo fmt` and `clippy --workspace` clean (this commit)._

> **Surveyed and cleared:** `spherical_tile_rig.rs` (1060 — coherent tile-rig
> primitive; its `camrig.rs` helper is properly private), `camera_intrinsics.rs`
> (905), `photometric_ransac.rs` (839), `per_spherical_tile_source_stack.rs`
> (862), `optical_flow/mod.rs` (958 — coordinator + public API + doc examples;
> the `bench` sub-module is appropriately doc-hidden). No `#[allow(dead_code)]`,
> no cargo warnings, all 32 pub modules referenced. A subagent-reported
> "production code after `mod tests`" smell in 9 small files was **verified to
> be a false positive** — those test blocks run to end-of-file as usual.

---

## Rust — `sfmtool-py`

### 5. Everything registers on one flat module — no PyO3 submodules (carried forward, #15)
- Location: `crates/sfmtool-py/src/lib.rs:101-287` + the 30 flat `py_*.rs` files
- Problem: Re-verified; unchanged. A single `#[pymodule] _sfmtool` makes 74 `add_function`/`add_class` registrations onto one `m`, and `src/sfmtool/__init__.py:4` still does `from sfmtool._sfmtool import *`, dumping all 74 names flat into `sfmtool`'s top level. Clear filename groups exist: I/O (`py_sfmr_io`, `py_sift_io`, `py_matches_io`, `py_camrig_io`, `py_colmap_binary`, `py_colmap_db`), matching (`py_descriptor_match`, `py_image_match`, `py_sweep_match`), geometry (`py_rigid_transform`, `py_rot_quaternion`, `py_se3_transform`, `py_camera_intrinsics`, `py_sphere_points`), flow (`py_optical_flow`, `py_warp_map`).
- Proposed fix: expose child modules via `add_submodule` (`_sfmtool.io`/`.match`/`.geometry`/`.flow`), group the `py_*.rs` into subdirs, replace `import *` with deliberate re-exports.
- Effort: high
- Risk: medium — all `import *` consumers + 74 call sites move; PyO3 submodules have `sys.modules` sharp edges. This is the only flatness that reaches the public Python API.

### 6. `py_sfmr_reconstruction.rs` dominated by one ~500-line method, with extraction patterns duplicated crate-wide (carried forward, #16; widened)
- Location: `crates/sfmtool-py/src/py_sfmr_reconstruction.rs` (1304); also `py_kdtree.rs`, `py_kdforest.rs`
- Problem: Re-verified; `clone_with_changes` spans 796-1294 (~498 lines, 38% of the file) with two local `macro_rules!` (`extract_array1` 807-831, `extract_array2` 832-854) used 7+ times. New this run: the dtype-inspection error-reporting pattern those macros implement (`.getattr("dtype")` → `.getattr("name")` → formatted message) is independently re-implemented in `py_kdtree.rs:19-22` and `py_kdforest.rs`, while the existing `helpers.rs` (179 lines) has no array-extraction helper.
- Proposed fix: hoist `extract_array1/2` (or an equivalent fn-based helper) into `helpers.rs` for crate-wide reuse; split `clone_with_changes` kwargs parsing into a private `recon_clone.rs` (`CloneChanges::from_kwargs`).
- Effort: medium · Risk: low — Python signatures unchanged; covered by `tests/rust_bindings/`.
> _Status (2026-06-09): Done (method half) — commit 0d42d92 (#55) moved the ~500-line body into `recon_clone.rs::clone_with_changes`, leaving a thin 4-line `#[pymethods]` wrapper (`py_sfmr_reconstruction.rs` 1304→814), and collapsed the two macros into one shared `extract_ndarray!` base with 1D/2D wrappers. Two deliberate deviations from the proposal: the logic stays a free function (not a `CloneChanges` struct — per-field validation is interleaved with running recon state) and the macros stay in `recon_clone.rs` rather than `helpers.rs` (no cross-module consumer; messages are clone-specific). The crate-wide consolidation half is therefore declined for the macros; the only residual is the small dtype-error-message pattern shared with `py_kdtree.rs:19-22`/`py_kdforest.rs` — demoted to a take-it-or-leave-it nicety._

---

## Rust — `sfm-explorer`

### 7. `image_detail.rs::show()` is a ~625-line method spanning input, overlays, rendering (carried forward, #17)
- Location: `crates/sfm-explorer/src/image_detail.rs` (1190)
- Problem: Re-verified; `ImageDetail::show` spans 125-749: fit/placement (205-224), input handling — drag-pan/scroll/pinch/keyboard + Windows `GestureEvent` (236-375), and a 7-branch `OverlayMode` match (401-646). Below it sit the 8 free metric/coloring helpers (`draw_feature_ellipse` 982-1042 through `compute_track_length_range` 1165-1188) — a self-contained cluster.
- Proposed fix: extract input handling → `image_detail/input.rs` (mirroring `viewer_3d/input.rs`) and the overlay drawing + 8 metric helpers → `image_detail/features.rs`, leaving `show` as orchestration.
- Effort: medium · Risk: low — intra-crate GUI refactor, no public API.

### 8. Top-level GUI panel/shell files form a flat cluster (carried forward, #18; downgraded)
- Location: `crates/sfm-explorer/src/lib.rs:15-24` — `app`, `state`, `dock`, `image_detail` (1190), `image_browser` (735), `point_track_detail` (839), `colormap` (206) flat beside `scene_renderer/`, `viewer_3d/`, `platform/`
- Problem: Re-verified as still flat (10 top-level modules, none added since). This run's re-survey judged the structure acceptable — the subtrees that warrant grouping are grouped, and the panel trio is discoverable. Kept as an optional nicety: a `panels/` module would mirror the three panels' shared `new`/`show`/`clear` + `*Response` shape.
- Proposed fix (optional): introduce `panels/` (`image_detail`, `image_browser`, `point_track_detail`).
- Effort: low · Risk: low — intra-crate moves; binary crate, no public API.

> **Surveyed and cleared:** `point_track_detail.rs` (839), `image_browser.rs`
> (735), `app.rs` (620), `viewer_3d/mod.rs` (647), `platform/windows.rs` (665),
> `scene_renderer/upload.rs` (847) — all large but coherent. The handful of
> `#[allow(dead_code)]` attributes (platform-gated gesture types, panel
> `clear()` methods) are justified; clippy is clean on both crates.

---

## Tests

### 9. `test_camrig.py` (1015) spans seven concerns including a 296-line `cam cp` block (carried forward, #21; narrowed)
- Location: `tests/test_camrig.py`
- Problem: Re-verified; the file covers format round-trip (57-90), spherical-tile CLI (92-153), PyO3 binding I/O (166-220), `camrig create` CLI (225-415), resolver (474-634, 15 tests), pattern-matching bindings (639-677), and an 18-test `cam cp` block (720-1015, 296 lines) exercising a *different command*. The `test_camera_config.py` half of the original finding is **dropped**: re-inspection found its unit/E2E split intentional — the cross-command `--camera-model` rejection tail (607-766) documents one feature contract across solve/match/to-colmap-db and belongs together.
- Proposed fix: extract the `cam cp` block → `tests/test_cam_cp.py`; optionally resolver/pattern → `test_camrig_resolve.py`.
- Effort: low · Risk: low — pure test relocation; fixtures come from `conftest.py`.
> _Status (2026-06-09): Done — commit 0d42d92 (#55) extracted the block into `tests/test_camrig_cp.py` (317 lines; the command is `sfm camrig cp`, not `cam cp` as this snapshot guessed); `test_camrig.py` 1015→712. It also went further than this report's "keep together" call on `test_camera_config.py`: the four cross-command CLI tests + shared helper moved to `tests/test_camera_config_cli.py` (199 lines), keeping `test_camera_config.py` (766→580) to unit tests — the contract-documentation rationale is preserved by keeping the four together in one file. The optional resolver/pattern split was left as-is._

> **Surveyed and cleared:** `test_sift_file.py` (711), `test_densify.py` (686),
> `test_undistort.py` (678), `test_epipolar.py` (598), `test_merge.py` (549),
> `test_warp_map_pose.py` (519) — all single-concern. `conftest.py` (326, 11
> fixtures, all used). Test-module naming is consistent with the regrouped
> source tree. `scripts/` has no dead or duplicate scripts (the four
> `init_dataset_*.sh` are intentionally parameterized per dataset). Repo root
> is clean.

---

## Top 3 (best effort-to-value)

1. **Finish the inline-test extraction (#4).** Eight `sfmtool-core` files still
   embed `mod tests` — three of them test-dominated (`rectification.rs` is 61%
   tests, `triangulation.rs` 46%, `image_pair_graph.rs` 40%) — and one
   (`optical_flow/variational.rs`) was simply missed by the prior pass. Low/low,
   mechanical, and it removes the residual two-convention split inside `sift/`.
   > _Status (2026-06-10): Done — see #4 (this commit)._

2. **Extract the `cam cp` block from `test_camrig.py` (#9) and consolidate the
   PyO3 array-extraction helpers (#6, helper half).** Two independent low/low
   cleanups: a 296-line block testing a different command moves to its own
   module, and the thrice-duplicated dtype-error pattern lands in
   `sfmtool-py/helpers.rs`.
   > _Status (2026-06-09): Done/declined — commit 0d42d92 (#55) did the test
   > extraction (and more; see #9) and the `clone_with_changes` split, but
   > deliberately kept the macros local to `recon_clone.rs` (see #6). Only the
   > minor dtype-error nicety remains._

3. **Regroup `sfmtool-core::lib.rs` into `geometry/`, `camera/`, `spherical/`
   (#2).** The largest remaining navigability win — 17 of the 32 flat modules
   fall into three verified clusters — and with the Python-side regroupings all
   landed, it is now the biggest flat surface left in the repo. Medium/medium;
   re-export shims keep downstream crates compiling during the transition.

> Larger items still open and worth scheduling beyond the Top 3: the
> `optical_flow/gpu/mod.rs` split (#3, medium-high effort, hardware-gated test
> risk), the `image_detail.rs::show` decomposition (#7), the
> `clone_with_changes` split (#6, method half), and — the only flatness that
> reaches the public Python API, and the highest-effort item in this report —
> the `sfmtool-py` PyO3 submodule restructure (#5).
