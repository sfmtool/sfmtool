# Spec / Code Consistency Audit — 2026-06-09

Bidirectional audit of every spec under `specs/` and `docs/` against the
implementing code, plus a code→spec sweep of every CLI command, crate, and
user-facing file format. Read-only analysis; nothing was modified.

Scope covered: 27 CLI command/sub-option specs, 10 core specs, 4 format specs,
3 workspace specs, 13 GUI specs + 2 docs pages, 6 draft specs — and code-side:
all 26 `_commands/*.py` modules, 8 crates, the workspace/camera config modules,
and the four on-disk formats.

This snapshot **supersedes `2026-06-07-spec-audit.md`** (retired in the same
commit). The two intervening days were spent on structural hygiene (commits
#49–#54), not spec fixes, so **nearly every finding carries forward**,
re-verified against the current tree with fresh line numbers. What did change:

- **Resolved:** the `sfm motion` console/JSON flag-divergence risk — commit
  707498e introduced a shared `_flag_frame` (now `motion/recon_discontinuity.py:420`)
  used by both the console table and the JSON report, and fixed a latent
  Cov-flag divergence in the JSON path. The motion spec now matches a single
  source of truth.
- **Path references held up:** the `colmap/`, `camera/`, `rig/`, and `motion/`
  regroupings refreshed spec file references in the same commits; a sweep for
  stale pre-move paths across `specs/` and `docs/` found none.
- **Refined this run:** the `epipolar` batch-naming finding is now precise
  (`_other` suffix vs the spec's `_A`/`_B`), the `gui-multi-panel` stale
  reference resolves to `viewer_3d/overlay.rs`, and the
  `photometric-subsets-ransac` draft's own "Draft" status marker is itself
  identified as stale (the code shipped and is consumed in production).

---

## CLI Command Specs

### specs/cli/align-command.md
**Summary:** Aligns multiple `.sfmr` reconstructions to a reference frame via `points` or `cameras` methods, with RANSAC tuning.
**Implementing code:** `src/sfmtool/_commands/align.py`; `align/{multi,by_cameras,by_points}.py`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/analyze-command.md
**Summary:** Deep-analysis modes, mutually exclusive, with per-image metrics and frustum/depth options.
**Implementing code:** `src/sfmtool/_commands/analyze.py`; `analyze/{graphs,depth,images,metrics}.py`.
**Inconsistencies:** (carried forward, still present)
  - `--depth-reliability` is fully implemented (`analyze.py:52-57,216`, in the mutual-exclusion set) and even appears in the command docstring (`analyze.py:120-122`), but remains absent from the spec's Analysis Modes table and syntax line — the spec lists 5 modes, the code has 6.
  - `--samples` is `click.IntRange(min=100)` (`analyze.py:81-84`); the spec's option table documents a plain `int` default 100 with no minimum.
**Recommendation:** update spec — add the `--depth-reliability` row and the `--samples` minimum.
**Unclear / incorrect / suspicious:** None.

### specs/cli/camrig-command.md
**Summary:** `camrig` group with `create`, `cp`, `spherical-tiles`; options, selectors, and mutual-exclusion rules all match.
**Implementing code:** `src/sfmtool/_commands/camrig.py` (`create` 29-100, `cp` 169-270, `spherical_tiles` 271-380); `camrig/{create,cp,resolver,pattern}.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/compare-command.md
**Summary:** Compares two reconstructions (first as reference): alignment, intrinsics, poses, feature usage.
**Implementing code:** `src/sfmtool/_commands/compare.py`, `_compare.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/densify-command.md
**Summary:** Densifies a point cloud via sweep matching with BA, point filters, geometric filtering. Self-flagged experimental.
**Implementing code:** `src/sfmtool/_commands/densify.py`, `_densify.py`.
**Inconsistencies:** None found — every option name and default matches.
**Recommendation:** none.

### specs/cli/epipolar-command.md
**Summary:** Visualizes epipolar lines/curves between image pairs (single or `--pairs-dir`), with rectify/undistort and sweep overlay.
**Implementing code:** `src/sfmtool/_commands/epipolar.py`; `visualization/_epipolar_display.py`.
**Inconsistencies:** (carried forward; sharpened this run)
  - Spec line 32 says `--separate` produces two files suffixed `_A`/`_B`; the code writes `{stem}.png` + `{stem}_other.png` (`_epipolar_display.py:617-625`), and in `--pairs-dir` batch mode saves only the first image when not side-by-side (`save_which="first"`).
**Recommendation:** update spec — document the actual `_other` suffix and batch-mode single-side behavior.
**Unclear / incorrect / suspicious:** None.

### specs/cli/flow-command.md
**Summary:** Dense DIS optical-flow visualization with optional reconstruction comparison and a documented `--pairs-dir` batch mode.
**Implementing code:** `src/sfmtool/_commands/flow.py`; `visualization/_flow_display.py`.
**Inconsistencies:** (carried forward, still present — **the one behavioral gap in the CLI set**)
  - **`--pairs-dir` is a dead option.** Declared (`flow.py:80-84`) and in the signature (`:97`), but never referenced in the body (`:138-176` use only `image1`/`image2`/`reconstruction_path`). The spec documents batch processing of "all adjacent pairs" (lines 28, 56-58) which the code does not implement.
**Recommendation:** update code (wire up batch mode, mirroring `epipolar`'s working `--pairs-dir`) **or** update spec (remove until implemented).
**Unclear / incorrect / suspicious:** The option looks vestigial next to `epipolar`'s functional twin.

### specs/cli/from-colmap-bin-command.md
**Summary:** Imports a COLMAP binary reconstruction into `.sfmr`, including `--detect-infinity`.
**Implementing code:** `src/sfmtool/_commands/from_colmap_bin.py`; `colmap/io.py` (`colmap_binary_to_rust_sfmr`).
**Inconsistencies:** None found (post-regroup imports verified).
**Recommendation:** none.

### specs/cli/heatmap-command.md
**Summary:** Per-feature quality-metric heatmaps overlaid on images.
**Implementing code:** `src/sfmtool/_commands/heatmap.py`; `visualization/_heatmap_renderer.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/inspect-command.md
**Summary:** Unified per-file/point inspection with content-hash `.sfmr` search order.
**Implementing code:** `src/sfmtool/_commands/inspect.py`; `analyze/summary.py`.
**Inconsistencies:** (carried forward) Cosmetic: spec names the optional second argument `WORKSPACE`; code declares it `location` (`inspect.py:38`). Same semantics.
**Recommendation:** none (or trivially align the name).

### specs/cli/insv2rig-command.md
**Summary:** Extracts dual-fisheye frames from Insta360 `.insv` and writes a `fisheye_360` `.camrig` (30.7 mm baseline).
**Implementing code:** `src/sfmtool/_commands/insv2rig.py`; `rig/insv2rig.py`.
**Inconsistencies:** None in the spec itself.
**Recommendation:** update code (trivial, carried forward) — the comment `rig/insv2rig.py:18` still says "~29mm" while `_X5_BASELINE_M = 0.0307` (30.7 mm). Internal comment typo, not spec divergence.

### specs/cli/match-command.md
**Summary:** Feature matching (exhaustive/sequential/flow) producing `.matches`, plus `--merge`, with the camera_config/`--camera-model` conflict rule.
**Implementing code:** `src/sfmtool/_commands/match.py`; `feature_match/_run.py`; `camera/setup.py`.
**Inconsistencies:** (carried forward, still present)
  - `--camera-model` is a 10-value `click.Choice` (`match.py:82-100`) omitting `FULL_OPENCV`, which is registered in `camera/cameras.py:66` / `camera/setup.py:85`; `sfm xform --camera-model` accepts the full registry. Cross-command inconsistency persists; still needs a decision.
**Recommendation:** discuss — whether `FULL_OPENCV` belongs in the `match`/`solve` Choice lists (or whether both should derive from the registry).

### specs/cli/merge-command.md
**Summary:** Merges pre-aligned reconstructions via the documented 7-step process.
**Implementing code:** `src/sfmtool/_commands/merge.py`; `merge/reconstructions.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/motion-command.md
**Summary:** Image-sequence and reconstruction discontinuity analysis with adaptive-stride flow and a versioned `--json` report.
**Implementing code:** `src/sfmtool/_commands/motion.py`; `motion/{recon_discontinuity,_recon_console,image_sequence,report,flow_stats}.py`.
**Inconsistencies:** None found. Strengthened since the prior audit: `_flag_frame` (`recon_discontinuity.py:420`) is now the single source of truth consumed by both the console (`_recon_console.py:107`) and the JSON report (`report.py:168`), and the latent JSON Cov-flag divergence was fixed (commit 707498e) — the code now honors the spec's "always flags" rule on total covisibility breaks.
**Recommendation:** none.

### specs/cli/pano2rig-command.md
**Summary:** Converts equirectangular panoramas to a 6-face `cubemap.camrig`.
**Implementing code:** `src/sfmtool/_commands/pano2rig.py`; `rig/pano2rig.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/panorama-command.md
**Summary:** Renders an equirectangular panorama via the spherical-tile consensus pipeline, with RANSAC knobs and `--camrig` rig source.
**Implementing code:** `src/sfmtool/_commands/panorama.py`; `rig/panorama.py`, `rig/spherical_tile.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/scale-by-measurements-command.md
**Summary:** `--scale-by-measurements` xform sub-option: YAML measurements, unit conversion, median scale, cross-reconstruction Point ID resolution.
**Implementing code:** `src/sfmtool/xform/_scale_by_measurements.py`; wired via `xform/_arg_parser.py`.
**Inconsistencies:** (carried forward, still present — re-verified directly this run)
  - The cross-reconstruction diagnostics example (spec lines 212-216) differs from the printed format: spec shows `Resolving Point IDs from sfmr/calib_x04.sfmr -> input.sfmr`; code prints `Resolving Point IDs from source ({prefix}...) -> input ({hash}...):` (`_scale_by_measurements.py:255`).
  - Example Point IDs in that block use non-hex characters (`pt3d_e5f6g7h8_…`, lines 213-216) while the parser requires `[0-9a-f]{8}` (`:30`) — the examples could not parse. (The *primary* diagnostics example earlier in the spec matches the code; only this block is wrong.)
**Recommendation:** update spec — refresh the cross-reconstruction diagnostics block and use valid hex IDs.

### specs/cli/sift-command.md
**Summary:** SIFT extract/draw with workspace integration, `--filter-sfm`, `--tool`/`--dsp` overrides.
**Implementing code:** `src/sfmtool/_commands/sift.py`; `sift/file.py`, `sift/extract_*.py`.
**Inconsistencies:** (carried forward, still present)
  - Spec line 16 says PATHS "If omitted, the workspace root is used", but the code requires it (`sift.py:98` raises `UsageError`).
**Recommendation:** discuss — drop the workspace-root-default claim, or implement it.

### specs/cli/solve-command.md
**Summary:** Incremental/global SfM from images or `.matches`, with rig support, flow matching, seq-overlap, camera_config/`.camrig` precedence, `--detect-infinity`.
**Implementing code:** `src/sfmtool/_commands/solve.py`; `_incremental_sfm.py`, `_global_sfm.py`.
**Inconsistencies:** (carried forward, still present)
  - `--camera-model` documented as free-form "string … auto" but is a 10-value `click.Choice` (`solve.py:110-122`), omitting `FULL_OPENCV` (same question as `match`).
  - `--seq-overlap` mutual exclusions — cannot combine with `--output` (`solve.py:186-189`) or `.matches` input (`:208-209`) — remain undocumented.
**Recommendation:** update spec — note the enumerated Choice and the `--seq-overlap` exclusions; fold the `FULL_OPENCV` question in with `match`.

### specs/cli/to-colmap-bin-command.md
**Summary:** Exports `.sfmr` to COLMAP binary with `--range`/`--filter-points`.
**Implementing code:** `src/sfmtool/_commands/to_colmap_bin.py`; `colmap/io.py::save_colmap_binary`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/to-colmap-db-command.md
**Summary:** Builds a COLMAP DB from `.sfmr` or `.matches`, with input-mode-scoped options.
**Implementing code:** `src/sfmtool/_commands/to_colmap_db.py`; `colmap/{db_export,db_setup,db_builders}.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/to-nerfstudio-command.md
**Summary:** Repackages a pinhole `.sfmr` into a Nerfstudio dataset.
**Implementing code:** `src/sfmtool/_commands/to_nerfstudio.py`; `_to_nerfstudio.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/undistort-command.md
**Summary:** Undistorts reconstruction images with `--fit`/`--filter`/`--output`, writing a workspace layout.
**Implementing code:** `src/sfmtool/_commands/undistort.py`; `_undistort_images.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/cli/ws-init-command.md
**Summary:** Initializes a workspace `.sfm-workspace.json` with feature-tool settings and validation rules.
**Implementing code:** `src/sfmtool/_commands/ws.py` (`init`); `_workspace.py`.
**Inconsistencies:** (carried forward, still present)
  - The `--feature-tool` Choice in the spec lists `colmap | opencv`; the code accepts `colmap | opencv | sfmtool` (`ws.py:24`). The COLMAP-only-option validation also errors for any non-`colmap` tool, not just `opencv`.
**Recommendation:** update spec — add `sfmtool` and generalize the validation wording.

### specs/cli/xform-command.md
**Summary:** Ordered transform/filter pipeline including points-at-infinity ops and `--include-by-distribution`.
**Implementing code:** `src/sfmtool/_commands/xform.py` (Click shell); `xform/_arg_parser.py` (ordered parser, since #53); `xform/*`.
**Inconsistencies:** (carried forward, still present)
  - `--find-points-at-infinity` accepts an optional 4th comma part `noise_floor_px` (`xform/_arg_parser.py:367-378`), but the spec's signature (line 168) documents only 3 parts.
**Recommendation:** update spec (minor) — document the 4th parameter (default 1.0).

### specs/cli/xform-select-by-distribution-command.md
**Summary:** `--include-by-distribution COUNT[,verbose]` — farthest-point + angular-thinning subset selection.
**Implementing code:** `src/sfmtool/xform/_select_by_distribution.py`; wired in `xform/_arg_parser.py:318-345`.
**Inconsistencies:** None material. (Wiring note was refreshed to `_arg_parser.py` in #53.)
**Recommendation:** none. Cosmetic slug/name mix (`--include-by-distribution` vs `xform-select-by-distribution`) persists; harmless.

---

## Core Specs

### specs/core/epipolar-curves.md
**Summary:** Distortion-aware epipolar "curve" plotting via ray→world→reproject sampling with adaptive subdivision. Implemented; the documented API still lags the per-feature-anchor design that shipped.
**Implementing code:** `crates/sfmtool-core/src/epipolar.rs` (`EpipolarCurveOptions` 115-122, `plot_epipolar_curve` 158, `plot_epipolar_curves_batch`); `py_epipolar.rs`; `visualization/_epipolar_display.py`.
**Inconsistencies:** (carried forward, still present)
  - The spec lists `anchor_depth` as an `EpipolarCurveOptions` field (default 1.0); the struct has only `curvature_tolerance` + `max_vertices` — anchor depth is a function parameter (`anchor_depth: f64` / `anchor_depths: &[f64]`), and the PyO3 binding takes a required positional `anchor_depths` array, not a scalar kwarg.
  - "Out of Scope" (line 287) lists `_rectification.py` among the `feature_match/` matchers; the matcher there is `_rectified_sweep.py` (`_rectification.py` lives at the package root).
  - No Status marker despite being fully implemented.
**Recommendation:** update spec — align the options struct, signatures, and PyO3 block with the per-feature-anchor implementation; fix the file path; add a Status marker.

### specs/core/flow-based-matching.md
**Summary:** Design/empirical rationale for flow-based feature matching (sliding-window advection + descriptor filtering).
**Implementing code:** `src/sfmtool/feature_match/_flow_matching.py`; `crates/sfmtool-core/src/optical_flow/`.
**Inconsistencies:** (carried forward) Internal narrative inconsistency — descriptor threshold `L2 <= 100` in the table header (line 39) vs `L2 <= 250` chosen elsewhere (lines 50/98/112).
**Recommendation:** update spec — reconcile the table header with the chosen threshold (confirm the code default in `_flow_matching.py` while doing so); optionally add a Status marker.

### specs/core/gpu-optical-flow.md
**Summary:** wgpu compute-shader DIS pipeline (5 stages, hybrid CPU/GPU routing, persistent buffer pools).
**Implementing code:** `crates/sfmtool-core/src/optical_flow/gpu/` + `dis.rs` routing.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/core/image-warping.md
**Summary:** WarpMap generation + resampling (bilinear/anisotropic), `ray_to_pixel`, `Equirectangular` camera model. All building blocks shipped, but the spec still uses "proposes" framing.
**Implementing code:** `warp_map.rs`, `remap.rs`, `camera_intrinsics.rs`/`distortion.rs` (`ray_to_pixel`), `py_warp_map.rs`.
**Inconsistencies:** (carried forward) No divergence of substance; the framing ("proposes", "New method needed") is stale and there is no Status marker.
**Recommendation:** update spec — flip to present tense + Status: Implemented; fold in the implemented `warpmap-pose-extension.md` draft (see Drafts).

### specs/core/optical-flow.md
**Summary:** Core DIS optical-flow spec (algorithm, parameters, presets, module structure, bindings).
**Implementing code:** `crates/sfmtool-core/src/optical_flow/`; `py_optical_flow.rs`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/core/per-spherical-tile-source-stack.md — Status: Implemented
**Summary:** CSR-flat per-tile multi-source pyramid stack (rotation-only build).
**Implementing code:** `per_spherical_tile_source_stack.rs`; `py_per_spherical_tile_source_stack.rs`.
**Inconsistencies:** None found — status marker accurate.
**Recommendation:** none.

### specs/core/randomized-kdtree-forest.md — Status: Implemented (Phase 1)
**Summary:** Randomized kd-tree forest ANN index for descriptors; highly accurate "as built" spec.
**Implementing code:** `crates/sfmtool-core/src/kdforest/`; `py_kdforest.rs`.
**Inconsistencies:** None found.
**Recommendation:** none — still the model status-marker style for other core specs.

### specs/core/sift.md — Status: DRAFT (stale)
**Summary:** "Planned pure-Rust SIFT detector" — but the detector fully shipped; only the status framing is wrong. API, params, defaults (`s=3`, `σ=1.6`, `contrast_threshold=0.0067`, `r=10`), SIMD symbols, and module layout all match.
**Implementing code:** `crates/sfmtool-core/src/sift/`; `py_sift.rs`, `py_sift_io.rs`.
**Inconsistencies:** (carried forward, still present) Header still declares "Status: draft … *planned*"; module-structure and bindings sections still labeled "(planned)" though they exist.
**Recommendation:** update spec — flip Status to Implemented (Phase 1), keeping the on-disk incremental extraction genuinely-future sections marked as such.

### specs/core/spherical-tiles-rig.md — Status: Implemented
**Summary:** Sphere-as-pinhole-tile-rig with atlas packing and resampling.
**Implementing code:** `spherical_tile_rig.rs` (+ `camrig.rs`); `sphere_points.rs`; `py_spherical_tile_rig.rs`; `rig/spherical_tile.py`.
**Inconsistencies:** (carried forward, still present; code-side confirmed this run)
  - Line 364's "Why closest-tile" note states `half_fov_rad = 0.5 · measured_max_nn_angle · overlap_factor`, contradicting the authoritative definition (lines 103-104, 174) and the code (`spherical_tile_rig.rs:248`: `measured_max_coverage_angle * params.overlap_factor` — no 0.5, coverage-angle).
**Recommendation:** update spec — correct line 364 to the coverage-angle form. Code is right; this is an isolated doc bug.

### specs/core/tile-batched-consensus-atlas.md — Status: Implemented
**Summary:** Bounded-memory batched panorama compositing orchestrator.
**Implementing code:** `consensus_atlas.rs`; `tiles_subset`; `consensus_patches_per_tile`; `photometric_ransac.rs`; `py_consensus_atlas.rs`; `rig/panorama.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

---

## Format Specs

### specs/formats/sfmr-file-format.md
**Summary:** `.sfmr` ZIP+zstd columnar format at version 2 (homogeneous points, points at infinity) with v1 read compatibility.
**Implementing code:** `crates/sfmr-format/src/`; `camera_intrinsics.rs`; `camera/cameras.py`.
**Inconsistencies:** (carried forward, still present)
  - The camera-model table (lines 268-282) omits `EQUIRECTANGULAR`, defined at `camera_intrinsics.rs:135` (string name `:161`). Writable/readable through the Rust core but undocumented here, and absent from Python `_CAMERA_PARAM_NAMES` — see camrig.
**Recommendation:** update spec — add an `EQUIRECTANGULAR` row or state it is Rust-core-only.

### specs/formats/camrig-file-format.md
**Summary:** `.camrig` ZIP+zstd format (version 1): camera pool, per-sensor poses, image patterns.
**Implementing code:** `crates/camrig-format/src/`; `camrig/create.py`, `rig/{pano2rig,insv2rig}.py`.
**Inconsistencies:** (carried forward, still present — **real code gap**)
  - The spec's example uses `model: EQUIRECTANGULAR` (line 134), but `_CAMERA_PARAM_NAMES` (`camera/cameras.py:21-125`) has no entry — a Python-built `.camrig` with that model would `KeyError`. Rust-side handling is fine; the gap is Python-only and untested (the kerry_park fixtures use `OPENCV_FISHEYE`).
**Recommendation:** discuss — add `EQUIRECTANGULAR` to `_CAMERA_PARAM_NAMES` (aligning Python with Rust and the spec) or change the example to a Python-supported model.

### specs/formats/matches-file-format.md
**Summary:** `.matches` ZIP+zstd format (version 1): images, image_pairs, optional two_view_geometries.
**Implementing code:** `crates/matches-format/src/`; `feature_match/`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/formats/sift-file-format.md
**Summary:** `.sift` ZIP+zstd feature format — v1 plus a clearly-marked v2 draft with an honest implementation-status banner.
**Implementing code:** `crates/sift-format/src/`; `sift/file.py`.
**Inconsistencies:** None found.
**Recommendation:** none — the `[v2]` tagging + status banner remains the right pattern.

---

## Workspace Specs

### specs/workspace/workspace.md
**Summary:** Workspace concept, `.sfm-workspace.json` schema (v1), discovery, path resolution.
**Implementing code:** `src/sfmtool/_workspace.py`.
**Inconsistencies:** None found.
**Recommendation:** none.

### specs/workspace/camera-config.md
**Summary:** Optional per-directory `camera_config.json` intrinsics (v1), closest-ancestor resolution capped at root, presence-based `--camera-model` rejection.
**Implementing code:** `src/sfmtool/camera/config.py` (`CameraConfigResolver`), `camera/{setup,cameras}.py` — path references current post-regroup.
**Inconsistencies:** None in the spec itself. (An `EQUIRECTANGULAR` entry in a `camera_config.json` would hit the same `_CAMERA_PARAM_NAMES` gap flagged under camrig — fix there.)
**Recommendation:** none here.

### specs/workspace/rig-config.md
**Summary:** Optional `rig_config.json` (COLMAP rig_configurator format verbatim), frame grouping, per-sensor intrinsics.
**Implementing code:** `rig/config.py` (`_sensor_from_rig_pose`, ref at spec line 119 current), `rig/frames.py` (`_build_cross_frame_pairs`).
**Inconsistencies:** None found.
**Recommendation:** none.

---

## GUI Specs

Feature-level consistency remains good; the recurring problem is unchanged from
the prior audit — **status hygiene** on specs written as proposals that have
since shipped, with `gui-plan.md` still carrying the oldest snapshot.

### specs/gui/README.md — index, accurate. No action.
### specs/gui/blender-viewport-navigation-implementation-overview.md — external reference doc; no consistency obligation. No action.
### specs/gui/gui-architecture.md — modules/pipelines all match. No action (a Status/date marker would help future audits).
### specs/gui/gui-user-experience.md — vision/principles doc. No action.
### specs/gui/gui-adaptive-clip-and-grid.md — Status: Implemented (2026-04-05); verified. No action.
### specs/gui/gui-camera-views.md — implemented features present; unchecked `[ ]` items verified genuinely absent. No action.
### specs/gui/gui-cross-panel-hover.md — Status: Implemented; verified. No action.
### specs/gui/gui-point-cloud-rendering.md — checkbox state verified accurate (unchecked items genuinely unimplemented). No action.
### specs/gui/gui-point-track-detail.md — Status: Implemented; verified. No action.

### specs/gui/gui-image-animation.md
**Inconsistencies:** (carried forward, still present) Proposal framing with an "Implementation Plan" and no Status marker, but `AnimationState` (`image_browser.rs:41-69`), `PlayDirection` (:32-38), keyboard controls, minibar transport, and camera-switch sync all exist.
**Recommendation:** update spec — add `Status: Implemented (date)`, convert plan framing to present tense.

### specs/gui/gui-multi-panel-image-browser.md
**Inconsistencies:** (carried forward; refined) "Plan:" framing though implemented; the stale reference at line 134 — "`viewer_3d.rs` (line ~1280)" — now resolves to `viewer_3d/overlay.rs` (the status-text overlay).
**Recommendation:** update spec — add Status: Implemented, fix the reference to `viewer_3d/overlay.rs`.

### specs/gui/gui-plan.md — "Current Implementation Status" *Updated: 2026-03-27* (oldest GUI snapshot)
**Inconsistencies:** (carried forward, still present, plus one addition)
  - Status list omits shipped features: image animation/playback, cross-panel hover, Point Track Detail (listed only as "Phase A complete" though fully implemented), and — new this run — camera-view FOV zoom (Ctrl+drag) appears nowhere in the navigation feature list.
  - "Image Browser → Planned enhancements: Animation mode" still listed as planned but implemented ("Grid mode" remains genuinely unimplemented).
  - "Next Steps" items overlapping `gui-adaptive-clip-and-grid.md` (Implemented 2026-04-05) still unreconciled.
**Recommendation:** update spec — refresh the snapshot date and reconcile the lists.

### specs/gui/gui-viewport-navigation.md
**Inconsistencies:** (carried forward, still present) Self-contradiction: the Camera View Mode Override subsection (lines 70-77) documents that all zoom controls adjust FOV in camera view — matching `viewer_3d/input.rs:105-106` and the user docs — while "Future Enhancements" line 647 keeps "FOV zoom" as a single unchecked `[ ] (Planned)` item. Only the *free-navigation* FOV-zoom binding is missing. Other unchecked items verified genuinely pending.
**Recommendation:** update spec — split the item: camera-view Ctrl+drag (done) vs free-navigation gesture (planned).

### docs/index.md
**Inconsistencies:** None found.
**Recommendation:** none.

### docs/tutorials/getting-started.md
**Inconsistencies:** None material — commands and GUI controls match the code.
**Recommendation:** none.

---

## Draft Specs

Unchanged from the prior audit — **the drafts directory remains the single
biggest source of status drift**, and none of the recommended graduations have
happened yet. Five of six drafts are implemented; one draft's own status
marker is now itself stale.

### specs/drafts/sfmr-v2-points-at-infinity.md — RETIRE (carried forward)
Fully implemented and canonicalized as v2 in `specs/formats/sfmr-file-format.md` §7. The draft is a redundant duplicate whose classification math (`α_max·f_max`) is the *old* approach superseded by `batch-triangulation-api.md`'s `inverse_depth_z`.
**Recommendation:** retire — the format spec is the source of truth.

### specs/drafts/batch-triangulation-api.md — PROMOTE to specs/core (carried forward)
Status marker reads "Implemented (all four migration phases landed)" and verification confirms it (`triangulation.rs`, `infinity/convert.rs::classify_rays_at_infinity`, bindings, GUI overlays, `analyze --depth-reliability`). Threshold calibration sub-questions remain appropriately flagged as deferred tuning.
**Recommendation:** promote to `specs/core/` as the triangulation/observability design of record.

### specs/drafts/photometric-subsets-ransac.md — PROMOTE to specs/core (carried forward; its own Status marker is now the drift)
The header still reads "**Status:** Draft … implementable from scratch", but the algorithm shipped: `photometric_ransac.rs` (839 lines) exists with PyO3 bindings and is consumed in production by `consensus_atlas.rs` and `rig/panorama.py` — the spec's own promotion trigger ("folds into specs/core once the production pipeline consumes its outputs") fired some time ago. This stale marker misled even this audit's first-pass review.
**Recommendation:** promote to specs/core and flip the Status marker; cross-link `consensus_atlas` as the production consumer.

### specs/drafts/warpmap-pose-extension.md — FOLD into image-warping (carried forward)
Status "Implemented" is accurate (`from_cameras_with_rotation`/`with_pose` at `warp_map.rs:40-78`, bindings, `tests/test_warp_map_pose.py`).
**Recommendation:** fold into `specs/core/image-warping.md` (which it explicitly extends) and remove from drafts/.

### specs/drafts/xform-find-points-at-infinity.md — PROMOTE to specs/cli (carried forward)
Status "Implemented" is accurate (`infinity/discover.rs`, `xform/_find_points_at_infinity.py`, CLI surface). Step 6 still describes the legacy `α_max·f_max` cut rather than the `inverse_depth_z`/`indeterminate` classifier that `batch-triangulation-api.md` installed — update on promotion so the two specs don't contradict.
**Recommendation:** promote into `specs/cli/`; update step 6's classifier description.

### specs/drafts/gui-points-at-infinity.md — UPDATE then fold into gui-point-cloud-rendering (carried forward)
Header still "Draft proposal" though rendering landed (commit c3c2805; `points.wgsl`, `scene_renderer/{gpu_types,uniforms,upload,auto_point_size}.rs`, `point_track_detail.rs`). The §5 UI controls ("Show points at infinity" toggle + `N points (M at infinity)` count) remain the only unbuilt piece — no such strings exist in `state.rs`/dock.
**Recommendation:** update status to "Implemented (minus §5 toggle/count)", fold into `specs/gui/gui-point-cloud-rendering.md`, and either implement or explicitly defer the toggle/count.

---

## Code Without Specs

26 of 27 user-facing CLI surfaces have a spec. Unchanged from the prior audit:

### `sfm explorer` (src/sfmtool/_commands/explorer.py)
**What it does:** Launches the native SfM Explorer GUI (same binary as `pixi run gui`), optionally opening a `.sfmr`.
**Why it matters:** user-facing — the CLI front door to the viewer.
**Recommendation:** a short `specs/cli/explorer-command.md` cross-linking the GUI specs (argument, executable discovery/error behavior, relationship to `pixi run gui`) would close the per-command spec set.

Other surfaces remain adequately covered or acceptably unspecced: `sfm version`
(small utility), `sfmtool-py` (binding glue; surface = union of core/format
specs), `sfm-explorer` (13 GUI specs), the four format crates (their format
specs), `sfmr-colmap` (the COLMAP-interop command specs), `sfmtool-core`
(the core specs).

---

## Top Priorities

Carried forward intact — none of the prior audit's priorities were addressed
in the interval (which was spent on structural hygiene), and re-verification
confirmed all of them. In unchanged priority order:

1. **Graduate the drafts directory.** Five of six drafts are implemented:
   retire `sfmr-v2-points-at-infinity.md`; promote `batch-triangulation-api.md`
   and `photometric-subsets-ransac.md` (whose own "Draft" marker is now
   actively misleading) to `specs/core/`; promote
   `xform-find-points-at-infinity.md` to `specs/cli/` (updating its step-6
   classifier); fold `warpmap-pose-extension.md` into `image-warping.md` and
   `gui-points-at-infinity.md` into `gui-point-cloud-rendering.md`.

2. **`sfm flow --pairs-dir` is a documented dead option.** Declared
   (`flow.py:80-84,97`) but never consumed; the spec's batch mode does nothing.
   Implement it (mirroring `epipolar`) or remove it from the spec. **Real
   behavioral gap.**

3. **`EQUIRECTANGULAR` camera model is Rust-only.** Defined at
   `camera_intrinsics.rs:135` and referenced by both format specs, but
   `_CAMERA_PARAM_NAMES` (`camera/cameras.py:21-125`) lacks an entry, so a
   Python-built `.camrig`/`camera_config.json` using it would `KeyError`, and
   no test covers the path. **Real code gap.**

4. **Concrete CLI spec/code mismatches.** (a) `analyze --depth-reliability`
   mode + `--samples` min missing from the analyze spec; (b) `ws init
   --feature-tool` accepts `sfmtool` but the spec lists `colmap|opencv`;
   (c) `sift` PATHS required in code vs optional in spec; (d) `solve`
   `--seq-overlap` exclusions and enumerated `--camera-model` undocumented.

5. **Stale/missing status markers.** `specs/core/sift.md` still "draft/planned"
   though shipped; `image-warping.md` and `epipolar-curves.md` (which also needs
   its `anchor_depth` API blocks and file path corrected) lack markers;
   `gui-plan.md`'s 2026-03-27 snapshot and the proposal framing in
   `gui-image-animation.md` / `gui-multi-panel-image-browser.md` need refreshing;
   `gui-viewport-navigation.md`'s FOV-zoom checkbox contradicts its own §
   "Camera View Mode Override" and the shipped code.

Lower-priority carry-forwards: `FULL_OPENCV` omitted from the `match`/`solve`
Choice lists (needs a decision), the `xform --find-points-at-infinity` 4th
`noise_floor_px` parameter undocumented, the `scale-by-measurements`
cross-reconstruction diagnostics example + invalid-hex Point IDs, the
`spherical-tiles-rig.md` line-364 `half_fov_rad` formula (code confirmed
correct at `spherical_tile_rig.rs:248`), the `epipolar` `_other`-suffix output
naming, the `rig/insv2rig.py:18` "~29mm" comment, and the one-paragraph
`explorer-command.md`.
