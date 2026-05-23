# Spec / Code Consistency Audit — 2026-05-22

Bidirectional audit of every spec under `specs/` and `docs/` against the
implementing code, plus a code→spec sweep of every CLI command, crate, and
user-facing file format. Read-only analysis; nothing was modified.

Scope covered: 13 CLI command specs, 2 xform sub-option specs, 4 format specs,
3 workspace specs, 9 core specs, 11 GUI specs, 4 drafts, 2 docs pages — and
code-side: all 24 `_commands/*.py` modules, 8 crates, the workspace/camera
config modules, and the four on-disk formats.

---

## CLI Command Specs

### specs/cli/align-command.md
**Summary:** Aligns multiple `.sfmr` reconstructions to a reference frame via `points` or `cameras` methods, with RANSAC options for the point method.
**Implementing code:** `src/sfmtool/_commands/align.py` (`align`), `src/sfmtool/_multi_align.py` (`align_command`).
**Inconsistencies:**
  - Spec marks `--confidence` as "cameras only" (table, line 31). The code (`align.py:36-40`) always accepts and forwards `--confidence` regardless of method and only gates the RANSAC trio (`align.py:115-122`). No hard error for `--confidence` with `--method points`. Minor.
**Recommendation:** update spec — clarify `--confidence` is simply ignored for `points`, not rejected.
**Unclear / incorrect / suspicious:** None material.

### specs/cli/compare-command.md
**Summary:** Compares two reconstructions: alignment transform, intrinsics, poses, feature usage.
**Implementing code:** `src/sfmtool/_commands/compare.py`, `src/sfmtool/_compare.py` (`compare_reconstructions`).
**Inconsistencies:** None found; spec matches the four documented comparison steps.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/densify-command.md
**Summary:** Densifies a point cloud via sweep matching between covisible (and optionally frustum) pairs, with BA, point filters, and geometric filtering.
**Implementing code:** `src/sfmtool/_commands/densify.py`, `src/sfmtool/_densify.py`.
**Inconsistencies:** All options in the spec tables match the `@click.option` declarations (max-features, sweep-window-size, distance-threshold, the BA trio, the four point filters, pair-selection options, frustum, geometric trio). No drift found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** Spec self-flags the command as experimental; that note is honest and current.

### specs/cli/discontinuity-command.md
**Summary:** Image-sequence and reconstruction discontinuity analysis with adaptive-stride optical flow, four reconstruction signals, and a versioned `--json` report.
**Implementing code:** `src/sfmtool/_commands/discontinuity.py`, `_discontinuity_image_sequence.py`, `_discontinuity_reconstruction.py`, `_discontinuity_json.py`, `_discontinuity_constants.py`.
**Inconsistencies:**
  - The "Code and patterns to build on" table (spec lines 614-625) cites `_inspect_images.py:_compute_camera_centers()`, `_inspect_images.py:_compute_rotation_angle()`, and `_inspect_metrics.py` for reprojection errors. Those module names no longer exist — they are now `_analyze_images.py` / `_analyze_metrics.py` (confirmed: only `_analyze_*` and `_discontinuity_reconstruction.py` reference these symbols). Stale path references.
**Recommendation:** update spec — rename `_inspect_images.py`→`_analyze_images.py` and `_inspect_metrics.py`→`_analyze_metrics.py` in the references table.
**Unclear / incorrect / suspicious:** The detailed JSON schema (v1) and CLI options match the implementation precisely; only the build-on table is stale.

### specs/cli/epipolar-command.md
**Summary:** Visualizes epipolar lines/curves between image pairs (single or `--pairs-dir`), with rectify/undistort and sweep-matching overlay.
**Implementing code:** `src/sfmtool/_commands/epipolar.py`, `src/sfmtool/visualization/_epipolar_display.py`.
**Inconsistencies:** Options match (`--draw`, `--max-features`, `--line-thickness`, `--feature-size`, `--rectify`, `--undistort`, `--draw-lines`, `--side-by-side`, `--sweep-with-max-features`, `--sweep-window-size`, `--pairs-dir`). Mutual exclusions and defaults match.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/flow-command.md
**Summary:** Dense DIS optical flow visualization with optional reconstruction comparison and adjacent-pair batch mode.
**Implementing code:** `src/sfmtool/_commands/flow.py`, `src/sfmtool/visualization/_flow_display.py`.
**Inconsistencies:** All options present and consistent.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/from-colmap-bin-command.md
**Summary:** Imports a COLMAP binary reconstruction into `.sfmr`.
**Implementing code:** `src/sfmtool/_commands/from_colmap_bin.py`, `src/sfmtool/_colmap_io.py`.
**Inconsistencies:**
  - The code has a `--detect-infinity / --no-detect-infinity` option (default on, `from_colmap_bin.py:38-44`) that reclassifies ill-conditioned points as points at infinity. The spec's option table (lines 16-20) lists only `--image-dir`, `--output`, `--tool-name`. Missing option.
**Recommendation:** update spec — add `--detect-infinity / --no-detect-infinity` to the options table; cross-link to `specs/formats/sfmr-file-format.md` §7 / `specs/drafts/sfmr-v2-points-at-infinity.md`.
**Unclear / incorrect / suspicious:** None beyond the missing flag.

### specs/cli/heatmap-command.md
**Summary:** Per-feature quality-metric heatmaps (reproj/tracks/angle/all) overlaid on images.
**Implementing code:** `src/sfmtool/_commands/heatmap.py`, `src/sfmtool/visualization/_heatmap_renderer.py`.
**Inconsistencies:** Options and colormap choices match. Output naming (`{stem}_{metric}`) is implemented via `_insert_metric_before_number` which inserts the metric *before* a trailing number (e.g. `image_reproj_001.png`); the spec example says `image_001_reproj.png`. The spec's stated naming order is the reverse of the code.
**Recommendation:** update spec — the implemented filename inserts the metric before the trailing number (`image_001` + `reproj` → `image_reproj_001`), not after.
**Unclear / incorrect / suspicious:** Spec output example `image_001_reproj.png` contradicts `_insert_metric_before_number` docstring examples in `heatmap.py:20-40`.

### specs/cli/match-command.md
**Summary:** Feature matching (exhaustive/sequential/flow) producing `.matches`, plus `--merge`. Documents camera_config interaction and detailed TVG-preserving merge semantics.
**Implementing code:** `src/sfmtool/_commands/match.py`.
**Inconsistencies:**
  - Spec is correct that merge preserves TVG (matches code `match.py:681-862, 911-928`). However the in-code docstring of `_run_merge` (`match.py:606-613`) says "Two-view geometry data is dropped since it is invalidated by the merge" — directly contradicting the code below it. Stale internal docstring (code-side, not spec-side).
  - `--camera-model` Choice list (`match.py:88-105`) omits `FULL_OPENCV`, which is a registered model (`_cameras.py:_CAMERA_PARAM_NAMES`). `sfm xform --camera-model` accepts the full registry; `match`/`solve` accept a fixed 10-model subset. Cross-command inconsistency, not flagged by the spec.
**Recommendation:** update code — fix the `_run_merge` docstring; discuss whether `FULL_OPENCV` should be in the `match`/`solve` choice lists.
**Unclear / incorrect / suspicious:** The stale docstring is a real trap for a reader skimming `_run_merge`.

### specs/cli/merge-command.md
**Summary:** Merges pre-aligned reconstructions: dedup cameras, merge images, union-find correspondences, percentile filter, average points, PnP+RANSAC pose refine.
**Implementing code:** `src/sfmtool/_commands/merge.py`, `src/sfmtool/_merge*.py`.
**Inconsistencies:** Options (`--output`, `--merge-percentile`) match; the documented 7-step process matches the command docstring and `merge_reconstructions` call.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/scale-by-measurements-command.md
**Summary:** `--scale-by-measurements` xform option: YAML measurements file, unit conversion, median scale factor, cross-reconstruction Point ID resolution, `world_space_unit` metadata.
**Implementing code:** `src/sfmtool/xform/_scale_by_measurements.py`, wired in `xform.py:279-291`.
**Inconsistencies:** The CLI wiring, parse (`--scale-by-measurements FILE`), and existence check match. (Detailed YAML/resolution semantics not exhaustively traced into `_scale_by_measurements.py`, but the option surface is consistent.)
**Recommendation:** none at the CLI surface.
**Unclear / incorrect / suspicious:** None at surface level.

### specs/cli/sift-command.md
**Summary:** SIFT extract/draw with workspace integration, `--filter-sfm`, `--tool`/`--dsp` overrides.
**Implementing code:** `src/sfmtool/_commands/sift.py`.
**Inconsistencies:** Options match (`--extract`, `--draw`, `--filter-sfm`, `--range`, `--num-threads`, `--tool`, `--dsp`). `--dsp` requiring `--tool` is enforced (`sift.py:122-126`), matching the spec table note "requires `--tool`".
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/solve-command.md
**Summary:** Incremental/global SfM from images or `.matches`, with rig support, flow matching, seq-overlap, camera_config and `.camrig` precedence.
**Implementing code:** `src/sfmtool/_commands/solve.py`, `_isfm.py`, `_gsfm.py`.
**Inconsistencies:**
  - Code has `--detect-infinity / --no-detect-infinity` (default on, `solve.py:128-134`) that is absent from the spec's options table (lines 30-42).
**Recommendation:** update spec — add `--detect-infinity / --no-detect-infinity` and link to the points-at-infinity model.
**Unclear / incorrect / suspicious:** `--camera-model` Choice (solve.py:108-127) omits `FULL_OPENCV` (same cross-command inconsistency as `match`).

### specs/cli/to-colmap-bin-command.md
**Summary:** Exports `.sfmr` to COLMAP binary, with `--range`/`--filter-points` subsetting; documents the Rust `subset_by_image_indices` semantics.
**Implementing code:** `src/sfmtool/_commands/to_colmap_bin.py`, `_colmap_io.save_colmap_binary`, `PySfmrReconstruction.subset_by_image_indices`.
**Inconsistencies:** Options, range semantics, and `--filter-points` requires `--range` (`to_colmap_bin.py:75-76`) all match. The "Implementation" section's signature and the CLI shim description match the code exactly.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/to-colmap-db-command.md
**Summary:** Builds a COLMAP DB from `.sfmr` or `.matches`, with guided-matching and camera-model options scoped to the right input mode.
**Implementing code:** `src/sfmtool/_commands/to_colmap_db.py`, `_to_colmap_db.py`, `_colmap_db.py`.
**Inconsistencies:** Options match input-mode scoping (`--max-features`/`--no-guided-matching` for `.sfmr`; `--camera-model` for `.matches`). Behavior matches `_from_sfmr` / `_from_matches`.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** Spec doesn't note that `--max-features`/`--no-guided-matching` are silently ignored for `.matches` input (no validation error); acceptable but undocumented.

### specs/cli/to-nerfstudio-command.md
**Summary:** Repackages a pinhole `.sfmr` into a Nerfstudio dataset (transforms.json, sparse_pc.ply, image pyramid, optional COLMAP sparse/), with `--range`/`--filter-points`.
**Implementing code:** `src/sfmtool/_commands/to_nerfstudio.py`, `src/sfmtool/_to_nerfstudio.py`.
**Inconsistencies:** Options and module-layout table match. `--filter-points requires --range` enforced (`to_nerfstudio.py:96-97`). The named functions (`frame_transform_matrix`, `build_transforms_json`, `write_sparse_ply`, `export_to_nerfstudio`) match the spec's module-layout table.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/ws-init-command.md
**Summary:** Initializes a workspace `.sfm-workspace.json` with feature-tool settings and validation rules.
**Implementing code:** `src/sfmtool/_commands/ws.py` (`init`), `src/sfmtool/_workspace.py` (`init_workspace`).
**Inconsistencies:**
  - The `--gpu / --no-gpu` option (spec table line 24, "GPU acceleration for SIFT") is validated against `--feature-tool opencv` and against `--affine-shape` (`ws.py:93-105`), **but `use_gpu` is never passed to `init_workspace`** (`ws.py:130-136`) and `init_workspace`/`get_colmap_feature_options` carry no `use_gpu` parameter (confirmed: no `use_gpu` key in `_extract_sift_colmap.py` options). The flag therefore has **no persisted effect** — the workspace config records no GPU setting. The spec presents `--gpu / --no-gpu` as a stored setting; in practice it only participates in the affine-shape conflict check.
**Recommendation:** discuss / update code or spec — either persist `use_gpu` into the workspace config (and have extraction honor it) or document that `--gpu/--no-gpu` is validation-only and GPU is chosen at extraction time.
**Unclear / incorrect / suspicious:** The dangling `--gpu` flag is a genuine spec/code mismatch worth a decision.

### specs/cli/camrig-command.md
**Summary:** `camrig` group with `create` (one-camera rig from images), `cp` (copy rig/camera/sensors from `.sfmr`/`.camrig`), `spherical-tiles`.
**Implementing code:** `src/sfmtool/_commands/camrig.py`, `_camrig_create.py`, `_camrig_cp.py`, `SphericalTileRig` (Rust).
**Inconsistencies:** All three subcommands' options, selectors, mutual-exclusion rules, and the `--equirect-width`/`--arc-per-pixel` exactly-one rule match the code. `--camera-model` Choice uses `_CAMERA_PARAM_NAMES` directly (camrig.py:26), so it can't drift.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/pano2rig-command.md
**Summary:** Converts equirectangular panoramas to a 6-face cubemap `.camrig` rig.
**Implementing code:** `src/sfmtool/_commands/pano2rig.py`, `src/sfmtool/_pano2rig.py`.
**Inconsistencies:** Options match; the spec correctly states the output is a `cubemap.camrig` (consistent with recent commit 8a3957b). Confirmed no `rig_config.json` is written anywhere in `_pano2rig.py`.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/insv2rig-command.md
**Summary:** Extracts dual-fisheye frames from Insta360 `.insv` video and writes a 2-sensor `fisheye_360` `.camrig`.
**Implementing code:** `src/sfmtool/_commands/insv2rig.py`, `src/sfmtool/_insv2rig.py`.
**Inconsistencies:** Output (`<stem>.camrig`, `fisheye_left/`, `fisheye_right/`), the 180°-about-Y right sensor, and the OPENCV_FISHEYE camera all match. Spec says baseline "29 mm" (line 33) while code uses `_X5_BASELINE_M = 0.0307` (= 30.7 mm; `insv2rig.py:25`). Numeric mismatch.
**Recommendation:** update spec — change "29 mm" to "30.7 mm" (matches `rig-config.md`, which already says 30.7 mm).
**Unclear / incorrect / suspicious:** The "29 mm" figure in the prose conflicts with both the code constant and the rig-config spec.

### specs/cli/inspect-command.md
**Summary:** Unified per-file inspection for `.sfmr`/`.sift`/`.matches`/`.camrig`/images, default vs `--verbose`, integrity line.
**Implementing code:** `src/sfmtool/_commands/inspect.py`, `src/sfmtool/_inspect_summary.py`.
**Inconsistencies:** Dispatch table, supported extensions, and `--verbose` flag match. No drift found at the command surface.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/analyze-command.md
**Summary:** Deep-analysis modes (`--coviz`/`--z-range`/`--frustum`/`--images`/`--metrics`), mutually exclusive, with frustum percentile/sample options.
**Implementing code:** `src/sfmtool/_commands/analyze.py`, `_analyze_graphs.py`, `_analyze_depth.py`, `_analyze_images.py`, `_analyze_metrics.py`.
**Inconsistencies:** Modes, mutual exclusion, `--range` only-with-`--metrics`, and frustum-only option gating all match. The per-image metrics columns and outlier flags match `_analyze_metrics.py`.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/xform-command.md
**Summary:** Ordered transform/filter pipeline (rotate/translate/scale/filters/camera-model/bundle-adjust/align/scale-by-measurements/include-by-distribution).
**Implementing code:** `src/sfmtool/_commands/xform.py`, `src/sfmtool/xform/*`.
**Inconsistencies:**
  - All documented options are wired in `parse_transform_args`. One naming note: the spec refers to the camera-model conversion under "`--camera-model`" which matches the CLI, but `camera-config.md` line 202 references it as `sfm xform --switch-camera-model` (the *class* is `SwitchCameraModelTransform`; the *flag* is `--camera-model`). The cross-reference in camera-config.md is wrong (see below).
**Recommendation:** none for xform-command.md itself.
**Unclear / incorrect / suspicious:** None in this spec.

### specs/cli/xform-select-by-distribution-command.md
**Summary:** `--include-by-distribution COUNT[,verbose]` — farthest-point + angular-thinning camera/rig-frame subset selection.
**Implementing code:** `src/sfmtool/xform/_select_by_distribution.py`, wired `xform.py:305-332`.
**Inconsistencies:** CLI parsing (split on `,`, `COUNT >= 2` UsageError, `verbose` modifier, unknown-modifier error) matches `xform.py:311-330`. Module path, export, and `SelectByDistributionFilter(count, verbose=...)` signature match the spec's Implementation Notes.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** Algorithm internals (k-d tree, seed cap) not line-verified, but the surface and named symbols are consistent.

---

## Format Specs

### specs/formats/sfmr-file-format.md
**Summary:** Full `.sfmr` ZIP+zstd columnar format at version 2 (homogeneous `(x,y,z,w)` points, points at infinity), with rigs/frames/depth-stats sections, Point IDs, and `world_space_unit`.
**Implementing code:** `crates/sfmr-format`, `crates/sfmtool-py` bindings, `src/sfmtool/_colmap_io.py`, `_sfmr` reader/writer.
**Inconsistencies:** Camera model parameter tables match `_cameras.py:_CAMERA_PARAM_NAMES` (PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, OPENCV_FISHEYE, the *_FISHEYE family, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE). Version-2 homogeneous model and migration notes are internally consistent. No drift found in the audited fields.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None — this spec is detailed and current (commit c80b80c "Fix the world space unit in .sfmr" aligns with the `world_space_unit` section).

### specs/formats/camrig-file-format.md
**Summary:** `.camrig` ZIP+zstd format (version 1): camera pool, per-sensor camera_indexes/quaternions/translations, image patterns, rig_type/rig_attributes, COLMAP rig relationship.
**Implementing code:** `crates/camrig-format`, `SphericalTileRig.write_camrig`, `_camrig_create.py`, `_camrig_cp.py`, `_insv2rig.py`/`_pano2rig.py` writers.
**Inconsistencies:** rig_type values (`generic`, `stereo_pair`, `fisheye_360`, `cubemap`, `spherical_tiles`) match the producers (`insv2rig`→`fisheye_360`, `pano2rig`→`cubemap`, `spherical-tiles`→`spherical_tiles`). Sensor-0-as-reference and COLMAP rebasing match `rig-config.md`. No drift found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/formats/matches-file-format.md
**Summary:** `.matches` ZIP+zstd format (version 1): images, image_pairs, optional two_view_geometries; write-once workflow.
**Implementing code:** `crates/matches-format`, `read_matches`/`write_matches` bindings, consumed in `match.py`, `_colmap_db.py`.
**Inconsistencies:** Field layout, TVG section optionality, and `has_two_view_geometries` metadata are consistent with how `match.py` builds `matches_data`. No drift found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/formats/sift-file-format.md
**Summary:** `.sift` ZIP+zstd feature format with `feature_type`-based naming convention.
**Implementing code:** `crates/sift-format`, `_sift_file.py`, `_extract_sift_colmap.py`, `_extract_sift_opencv.py`.
**Inconsistencies:** `feature_type` naming (`sift-colmap`, `sift-colmap-dsp`, `sift-colmap-max{N}`, `sift-opencv`) consistent with `get_feature_type_for_tool`. (Only the first 60 lines read in detail; surface convention matches.)
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None observed.

---

## Workspace Specs

### specs/workspace/workspace.md
**Summary:** Workspace concept, `.sfm-workspace.json` schema (version 1), discovery, path resolution, lifecycle.
**Implementing code:** `src/sfmtool/_workspace.py`, `_sift_file.py`.
**Inconsistencies:** Config fields (`version`, `feature_tool`, `feature_type`, `feature_options`, `feature_prefix_dir`) match `init_workspace`. The example config (lines 70-86) shows a richer COLMAP `feature_options` block (`peak_threshold`, `edge_threshold`, `upright`, `normalization`) than the `.sfmr`/matches spec examples — consistent with `get_colmap_feature_options`. No `use_gpu` key, matching code (and corroborating the ws-init `--gpu` finding above).
**Recommendation:** none for this spec.
**Unclear / incorrect / suspicious:** None.

### specs/workspace/camera-config.md
**Summary:** Optional per-directory `camera_config.json` intrinsics (version 1), closest-ancestor resolution, `--camera-model` rejection, rig-config interaction.
**Implementing code:** `src/sfmtool/_camera_config.py`, `_camera_setup.py` (`_check_camera_model_conflict`), consumed by `solve`/`match`/`to-colmap-db`/`densify`.
**Inconsistencies:**
  - Line 202 says "use `sfm xform --switch-camera-model`". There is no `--switch-camera-model` flag — the CLI option is `--camera-model` (`xform.py:444-452`); only the internal class is `SwitchCameraModelTransform`. Wrong flag name in cross-reference.
**Recommendation:** update spec — change `--switch-camera-model` to `--camera-model`.
**Unclear / incorrect / suspicious:** The "How Commands Use" list (solve/match/to-colmap-db/densify) is consistent with `_check_camera_model_conflict` call sites; only the flag-name reference is wrong.

### specs/workspace/rig-config.md
**Summary:** Optional `rig_config.json` (COLMAP rig_configurator format verbatim), frame grouping, per-sensor intrinsics, `--camera-model` interaction.
**Implementing code:** `src/sfmtool/_rig_config.py`, `_rig_frames.py`, distortion blending in `crates/sfmtool-core/src/distortion.rs`.
**Inconsistencies:**
  - "Examples in the Repository" (lines 210-211): "`sfm pano2rig` — generates a rig_config.json for the six rectilinear faces of a panoramic cubemap." This is stale — `pano2rig` now writes a `.camrig` (confirmed: `_pano2rig.py` writes no `rig_config.json`; commit 8a3957b). The rig_config.json format remains a valid *input* sfmtool consumes, but it is no longer *produced* by pano2rig.
**Recommendation:** update spec — replace the pano2rig example with the `.camrig` reality, or point to `specs/cli/pano2rig-command.md`.
**Unclear / incorrect / suspicious:** Otherwise the COLMAP-format mapping and quaternion conventions match the code.

---

## Core Specs

All core specs carry explicit Status markers and matched code where checked.

### specs/core/spherical-tiles-rig.md — Status: Implemented
**Implementing code:** `crates/sfmtool-core/src/spherical_tile_rig.rs`, `SphericalTileRig` (Python), `src/sfmtool/_spherical_tile_rig.py`. Consumed by `camrig spherical-tiles`. Symbols named in the spec exist. No drift found.

### specs/core/optical-flow.md, gpu-optical-flow.md, image-warping.md, epipolar-curves.md, flow-based-matching.md, per-spherical-tile-source-stack.md (Implemented), tile-batched-consensus-atlas.md (Implemented)
**Implementing code:** `crates/sfmtool-core/src/{optical_flow,distortion,epipolar,...}.rs`, `feature_match/_flow_matching.py`, `visualization/_epipolar_display.py`. These are algorithm design specs; the ones with explicit "Implemented" status (`per-spherical-tile-source-stack`, `tile-batched-consensus-atlas`, `spherical-tiles-rig`) name concrete Rust files. Not exhaustively line-verified against Rust; spot checks (epipolar-curves → `crates/sfmtool-core/src/epipolar.rs` + `_epipolar_display.py`) are consistent.
**Recommendation:** none from this audit; a Rust-side deep dive is out of scope here.

---

## GUI Specs

### specs/gui/* (README + 10 docs)
**Summary:** Design specs for the SfM Explorer (`crates/sfm-explorer`). Most carry "Implemented" status (`gui-cross-panel-hover`, `gui-point-track-detail`, `gui-adaptive-clip-and-grid` (2026-04-05)); `gui-plan.md` has a "Current Implementation Status" section dated 2026-03-27.
**Implementing code:** `crates/sfm-explorer` (winit + wgpu + egui), launched via `sfm explorer` (`_commands/explorer.py` → `launch-sfm-explorer`).
**Inconsistencies:** Not deeply verified against Rust GUI source (large surface, out of scope for line-level checks here). `gui-plan.md`'s status snapshot is dated and may lag current GUI state.
**Recommendation:** discuss — a dedicated GUI-vs-Rust pass would be a separate effort; refresh `gui-plan.md`'s 2026-03-27 status if the GUI has advanced since.
**Unclear / incorrect / suspicious:** None at the spec-structure level.

---

## Draft Specs (forward-looking — not "missing implementations")

- **specs/drafts/sfmr-v2-points-at-infinity.md** — Draft proposal, but **largely implemented**: the homogeneous v2 model it proposes is now the authoritative content of `specs/formats/sfmr-file-format.md` §7 and is wired through `solve`/`from-colmap-bin` `--detect-infinity`. **Recommendation: discuss** promoting/retiring this draft since the format spec now owns the design.
- **specs/drafts/warpmap-pose-extension.md** — Status: Implemented (in `crates/sfmtool-core`). A draft marked implemented; consider moving to `specs/core/`.
- **specs/drafts/photometric-subsets-ransac.md** — Status: Draft, genuinely forward-looking (no production consumer yet). Correctly a draft.

---

## Code Without Specs

### `sfm explorer` (`src/sfmtool/_commands/explorer.py`)
**What it does:** Launches the native SfM Explorer GUI binary (`launch-sfm-explorer`), optionally opening a `.sfmr`. Thin subprocess shim.
**Why it matters:** User-facing CLI command in the Visualization category.
**Recommendation:** acceptable as unspecced at the CLI level — the GUI itself is covered by `specs/gui/*`. A one-paragraph `specs/cli/explorer-command.md` (mirroring `inspect`/`analyze`) would complete the per-command spec set.

### `sfm version` (`cli.py:81-84`)
**What it does:** Prints `sfmtool 0.1`.
**Why it matters:** Small utility command; hardcoded version string (does not read package metadata).
**Recommendation:** acceptable as unspecced.

### crate `sfmr-colmap` (`crates/sfmr-colmap`)
**What it does:** COLMAP binary + SQLite interop (the read/write paths behind `from-colmap-bin`, `to-colmap-bin`, `to-colmap-db`).
**Why it matters:** Internal-but-load-bearing; the interop boundary for all COLMAP commands.
**Recommendation:** acceptable as unspecced — its behavior is documented indirectly by the COLMAP-interop command specs and the format specs; a short `specs/core/` note on the COLMAP interop boundary would help but isn't required.

### crate `sfmtool-py` (`crates/sfmtool-py`)
**What it does:** PyO3 bindings compiled as `sfmtool._sfmtool` — the Python↔Rust surface.
**Why it matters:** Internal-but-load-bearing; every Python module that imports `_sfmtool` depends on it.
**Recommendation:** acceptable as unspecced (it re-exports algorithms specced elsewhere).

### crate `sfmtool-core` (`crates/sfmtool-core`)
**What it does:** The algorithm crate (camera, distortion, epipolar, optical flow, frustum, transforms, spatial indexing, spherical tiles).
**Why it matters:** Internal-but-load-bearing; the algorithmic heart.
**Recommendation:** mostly covered by `specs/core/*` (optical-flow, epipolar-curves, image-warping, spherical-tiles, etc.). Camera/distortion model details are documented within the sfmr/rig format specs. No dedicated camera-model algorithm spec exists, but coverage is adequate.

### `sfm xform` sub-transforms without dedicated specs
**What it does:** `--rotate`, `--translate`, `--scale`, `--remove-isolated`, `--remove-short-tracks`, `--remove-narrow-tracks`, `--remove-large-features`, `--filter-by-reprojection-error`, `--include-glob`/`--exclude-glob`, `--align-to`/`--align-to-input`, `--camera-model`, `--bundle-adjust` (`src/sfmtool/xform/*`).
**Why it matters:** User-facing transforms; only `--scale-by-measurements` and `--include-by-distribution` have dedicated sub-specs.
**Recommendation:** acceptable as unspecced — they are documented inline in `xform-command.md`, which is sufficient for their complexity.

### Workspace bootstrap scripts (`scripts/init_dataset_*.sh`, `scripts/coverage.sh`)
**What it does:** Dataset bootstrap and combined coverage tooling.
**Why it matters:** Small developer utilities.
**Recommendation:** acceptable as unspecced (CLAUDE.md references them).

---

## Top Priorities

1. **`sfm ws init --gpu/--no-gpu` has no persisted effect.** Validated but never passed to `init_workspace`; the workspace config records no GPU setting (`ws.py:130-136`, `_extract_sift_colmap.py`). Either persist/honor it or document it as validation-only. (specs/cli/ws-init-command.md) — *real behavioral gap.*

2. **`--detect-infinity` missing from `solve` and `from-colmap-bin` specs.** Both commands ship the flag (default on) but neither options table lists it (`solve.py:128-134`, `from_colmap_bin.py:38-44`). Add it and link to the points-at-infinity model. (specs/cli/solve-command.md, from-colmap-bin-command.md)

3. **Stale `rig-config.md` pano2rig example.** Lines 210-211 claim `pano2rig` generates `rig_config.json`; it now writes `.camrig` (commit 8a3957b; confirmed no `rig_config.json` writer in code). Update the example. (specs/workspace/rig-config.md)

4. **Discontinuity spec references dead module names.** The "build on" table cites `_inspect_images.py` / `_inspect_metrics.py`, renamed to `_analyze_images.py` / `_analyze_metrics.py` (commit 1a8b76a era). Update the references table. (specs/cli/discontinuity-command.md, lines 614-625)

5. **Two smaller doc/code fixes:** (a) `heatmap` filename order — spec says `image_001_reproj.png` but code emits `image_reproj_001.png` (`heatmap.py:_insert_metric_before_number`); (b) the `_run_merge` docstring in `match.py:606-613` says TVG is dropped while the code preserves it (and the spec correctly says preserved) — fix the misleading docstring. Also: `insv2rig.md` says "29 mm" baseline vs code's 30.7 mm.
