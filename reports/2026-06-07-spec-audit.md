# Spec / Code Consistency Audit — 2026-06-07

Bidirectional audit of every spec under `specs/` and `docs/` against the
implementing code, plus a code→spec sweep of every CLI command, crate, and
user-facing file format. Read-only analysis; nothing was modified.

Scope covered: 25 CLI command/sub-option specs, 10 core specs, 4 format specs,
3 workspace specs, 13 GUI specs + 2 docs pages, 6 draft specs — and code-side:
all 25 `_commands/*.py` modules, 8 crates, the workspace/camera config modules,
and the four on-disk formats.

This snapshot supersedes `2026-05-22-spec-audit.md` (retired in the same commit):
every concrete inconsistency that report flagged has since been fixed — the
`ws init --gpu` persistence, `--detect-infinity` in the `solve`/`from-colmap-bin`
specs, the `rig-config.md` pano2rig example, the `camera-config.md`
`--camera-model` cross-reference, the `heatmap` filename order, the `_run_merge`
docstring, and the insv2rig baseline were all resolved (largely in commit
c61cd86 and the 2026-06-05 hygiene pass). The findings below are fresh against
the current tree.

---

## CLI Command Specs

### specs/cli/align-command.md
**Summary:** Aligns multiple `.sfmr` reconstructions to a reference frame via `points` or `cameras` methods, with RANSAC tuning. Spec, defaults, Choice values, and mutual-exclusion notes all match.
**Implementing code:** `src/sfmtool/_commands/align.py` (`align`), `src/sfmtool/align/multi.py` (`align_command`, `estimate_pairwise_alignment`).
**Inconsistencies:** None found. The "confidence accepted but ignored for points" note is consistent with the code (no rejection under `points`); RANSAC-only-with-points gate at `align.py:115-122` matches.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/analyze-command.md
**Summary:** Deep-analysis modes, mutually exclusive, with per-image metrics and frustum/depth options.
**Implementing code:** `src/sfmtool/_commands/analyze.py` (`analyze`); `analyze/{graphs,depth,images,metrics}.py`.
**Inconsistencies:**
  - `--depth-reliability` mode exists in code (`analyze.py:52-57,121-123,215-216`, in the mutual-exclusion set at `:147-167`) but is entirely absent from the spec's mode table (spec lines 24-30) and syntax line (15). The spec lists 5 modes; the code has 6.
  - `--samples` is `click.IntRange(min=100)` (`analyze.py:81`); the spec (line 39) documents a plain int with default 100 and no minimum.
**Recommendation:** update spec — add the `--depth-reliability` mode and document the `--samples` minimum of 100.
**Unclear / incorrect / suspicious:** None.

### specs/cli/camrig-command.md
**Summary:** `camrig` group with `create`, `cp`, `spherical-tiles`. All three subcommands' options, selectors, mutual-exclusion rules match.
**Implementing code:** `src/sfmtool/_commands/camrig.py`; `camrig/{create,cp}.py`, `SphericalTileRig`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/compare-command.md
**Summary:** Compares two reconstructions (first as reference): alignment, intrinsics, poses, feature usage. Takes no options.
**Implementing code:** `src/sfmtool/_commands/compare.py`, `src/sfmtool/_compare.py` (`compare_reconstructions`).
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/densify-command.md
**Summary:** Densifies a point cloud via sweep matching with BA, point filters, and geometric filtering. Self-flagged experimental.
**Implementing code:** `src/sfmtool/_commands/densify.py`, `src/sfmtool/_densify.py`, `feature_match.GeometricFilterConfig`.
**Inconsistencies:** None found — every option name and default matches.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/epipolar-command.md
**Summary:** Visualizes epipolar lines/curves between image pairs (single or `--pairs-dir`), with rectify/undistort and sweep overlay.
**Implementing code:** `src/sfmtool/_commands/epipolar.py` (`epipolar`, `resolve_image_name`); `visualization/_epipolar_display.py`.
**Inconsistencies:**
  - Minor: in `--pairs-dir` batch mode the code names each output by image stem `<stem>.png` and only saves the first image when not side-by-side (`epipolar.py:283-307`, `save_which="first"`). The spec's `_A`/`_B` suffix description (line 32) and "one visualization per pair" (47) gloss this batch-mode naming/single-side behavior.
**Recommendation:** discuss — optionally clarify batch-mode output naming; otherwise consistent.
**Unclear / incorrect / suspicious:** None.

### specs/cli/flow-command.md
**Summary:** Dense DIS optical-flow visualization with optional reconstruction comparison and a documented `--pairs-dir` adjacent-pair batch mode.
**Implementing code:** `src/sfmtool/_commands/flow.py` (`flow`); `visualization/_flow_display.py`.
**Inconsistencies:**
  - **`--pairs-dir` is a dead option.** It is declared as an option and function parameter (`flow.py:79-84,97`) but never referenced in the function body — the command always processes only `image1`/`image2`. The spec documents batch processing of "all adjacent pairs" (lines 28, 56-58), which the code does not implement. Even the documented invocation still requires the two positional image args.
**Recommendation:** update code (wire up batch mode, mirroring `epipolar`'s working `--pairs-dir`) **or** update spec (remove `--pairs-dir` until implemented). Current state is a documented-but-dead option.
**Unclear / incorrect / suspicious:** The option looks vestigial relative to the functional `--pairs-dir` in `epipolar`.

### specs/cli/from-colmap-bin-command.md
**Summary:** Imports a COLMAP binary reconstruction into `.sfmr`, including `--detect-infinity`.
**Implementing code:** `src/sfmtool/_commands/from_colmap_bin.py`; `_colmap_io.py`, `_sfmtool.read_colmap_binary`.
**Inconsistencies:** None found. `--detect-infinity/--no-detect-infinity` (default True → `classify_infinity`) is now in the spec, resolving the prior audit's finding.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/heatmap-command.md
**Summary:** Per-feature quality-metric heatmaps overlaid on images.
**Implementing code:** `src/sfmtool/_commands/heatmap.py` (`heatmap`, `_insert_metric_before_number`); `visualization/_heatmap_renderer.py`, `sift/file.py`.
**Inconsistencies:** None found — the metric-before-trailing-number naming now matches (prior audit finding resolved).
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/inspect-command.md
**Summary:** Unified per-file/point inspection with content-hash `.sfmr` search order and default/verbose fields.
**Implementing code:** `src/sfmtool/_commands/inspect.py` (`inspect`, `_inspect_point`, `_find_sfmr_by_content_hash`); `analyze/summary.py`.
**Inconsistencies:**
  - Cosmetic: the spec names the optional second argument `WORKSPACE` (lines 18, 27-29); the code declares it `location` / help text `LOCATION` (`inspect.py:38,60-64`). Same semantics.
**Recommendation:** none (or trivially align the arg name).
**Unclear / incorrect / suspicious:** None.

### specs/cli/insv2rig-command.md
**Summary:** Extracts dual-fisheye frames from Insta360 `.insv` and writes a `fisheye_360` `.camrig`.
**Implementing code:** `src/sfmtool/_commands/insv2rig.py`; `src/sfmtool/_insv2rig.py`.
**Inconsistencies:** None found in the spec — baseline now reads 30.7 mm (matches `_X5_BASELINE_M = 0.0307`).
**Recommendation:** update code (trivial) — the in-code comment `insv2rig.py:19` still says "~29mm" while the constant is 30.7 mm. Internal comment typo, not a spec divergence.
**Unclear / incorrect / suspicious:** None.

### specs/cli/match-command.md
**Summary:** Feature matching (exhaustive/sequential/flow) producing `.matches`, plus `--merge`, with the camera_config/`--camera-model` conflict rule.
**Implementing code:** `src/sfmtool/_commands/match.py`; `feature_match/_run.py` (`_run_matching`, `_run_merge`), `_camera_setup._check_camera_model_conflict`.
**Inconsistencies:**
  - Carried forward: `--camera-model` is a 10-value `click.Choice` (`match.py:84-98`) that omits `FULL_OPENCV`, a registered model. `sfm xform --camera-model` accepts the full registry. The spec's examples stay within the subset, so no functional break, but the cross-command inconsistency persists and still needs a decision.
**Recommendation:** discuss — whether `FULL_OPENCV` belongs in the `match`/`solve` Choice lists.
**Unclear / incorrect / suspicious:** None (the `_run_merge` docstring is now correct).

### specs/cli/merge-command.md
**Summary:** Merges pre-aligned reconstructions via the documented 7-step process.
**Implementing code:** `src/sfmtool/_commands/merge.py`; `merge/reconstructions.py` (`merge_reconstructions`).
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/motion-command.md
**Summary:** Image-sequence and reconstruction discontinuity analysis with adaptive-stride flow and a versioned `--json` report. (Formerly `discontinuity`.)
**Implementing code:** `src/sfmtool/_commands/motion.py`; `motion/{image_sequence,reconstruction,report}.py`.
**Inconsistencies:** None material. The build-on table references resolve (`analyze/images.py`, `analyze/metrics.py`, `visualization/_flow_display.py`, `_image_pair_graph.py` all exist); the discontinuity→motion rename does not leak. Prior audit finding resolved.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/pano2rig-command.md
**Summary:** Converts equirectangular panoramas to a 6-face `cubemap.camrig`.
**Implementing code:** `src/sfmtool/_commands/pano2rig.py`; `_pano2rig.py`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/panorama-command.md
**Summary:** Renders an equirectangular panorama from a reconstruction via the spherical-tile consensus pipeline, with RANSAC knobs and `--camrig` rig source.
**Implementing code:** `src/sfmtool/_commands/panorama.py`; `_panorama.py` (`render_equirect_panorama`, `resolve_panorama_rig`, `select_source_indices`), `_spherical_tile_rig.py`.
**Inconsistencies:** None found — all 16 options/defaults and validation rules match.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/scale-by-measurements-command.md
**Summary:** `--scale-by-measurements` xform sub-option: YAML measurements, unit conversion, median scale, cross-reconstruction Point ID resolution, `world_space_unit` metadata.
**Implementing code:** `src/sfmtool/xform/_scale_by_measurements.py`; wired `xform.py:286-298`.
**Inconsistencies:**
  - The example diagnostics block (spec line 212) differs from the actual printed format (`_scale_by_measurements.py:255,274`): real output is `Resolving Point IDs from source ({prefix}…) -> input ({hash}…):` and per-line `pt3d_… -> point {idx} (via …)`. Cosmetic.
  - Illustrative example Point IDs use non-hex characters (e.g. `pt3d_e5f6g7h8`) while the parser requires `[0-9a-f]{8}` (`:30`) — the examples could not actually parse.
**Recommendation:** update spec — refresh the two diagnostics examples and use valid hex in example Point IDs.
**Unclear / incorrect / suspicious:** None behavioral.

### specs/cli/sift-command.md
**Summary:** SIFT extract/draw with workspace integration, `--filter-sfm`, `--tool`/`--dsp` overrides.
**Implementing code:** `src/sfmtool/_commands/sift.py`; `sift/file.py`, `sift/extract_*.py`.
**Inconsistencies:**
  - Spec line 16 ("If omitted, the workspace root is used.") and the syntax `sfm sift [PATHS...]` present PATHS as optional, but the code requires it: `if not paths: raise click.UsageError("Must provide a list of paths to process.")` (`sift.py:97-98`).
**Recommendation:** discuss — either drop the "workspace root default" claim and mark PATHS required, or implement the default-to-workspace-root behavior the spec promises.
**Unclear / incorrect / suspicious:** None beyond the above.

### specs/cli/solve-command.md
**Summary:** Incremental/global SfM from images or `.matches`, with rig support, flow matching, seq-overlap, camera_config/`.camrig` precedence, and `--detect-infinity`.
**Implementing code:** `src/sfmtool/_commands/solve.py`; `_incremental_sfm.py`, `_global_sfm.py`.
**Inconsistencies:**
  - `--camera-model` is documented as a free-form "string … auto" (line 41) but is a 10-value `click.Choice` (`solve.py:110-124`) — and, as with `match`, omits `FULL_OPENCV`.
  - The `--seq-overlap` mutual-exclusion rules (cannot combine with `--output`/`--flow-match`/`.matches`, enforced at `solve.py:186-211`) are undocumented; the spec is silent rather than wrong.
**Recommendation:** update spec — note `--camera-model` is an enumerated Choice and document the `--seq-overlap` exclusions; fold the `FULL_OPENCV` question in with the `match` finding.
**Unclear / incorrect / suspicious:** None blocking.

### specs/cli/to-colmap-bin-command.md
**Summary:** Exports `.sfmr` to COLMAP binary with `--range`/`--filter-points` subsetting.
**Implementing code:** `src/sfmtool/_commands/to_colmap_bin.py`; `_colmap_io.save_colmap_binary`, `SfmrReconstruction.subset_by_image_indices`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/to-colmap-db-command.md
**Summary:** Builds a COLMAP DB from `.sfmr` or `.matches`, with input-mode-scoped options.
**Implementing code:** `src/sfmtool/_commands/to_colmap_db.py`; `_to_colmap_db.py`, `_colmap_db.py`.
**Inconsistencies:** None functional — the `.matches` camera_config rejection is delegated into `_setup_for_sfm_from_matches`, consistent with the spec.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/to-nerfstudio-command.md
**Summary:** Repackages a pinhole `.sfmr` into a Nerfstudio dataset, with `--range`/`--filter-points`.
**Implementing code:** `src/sfmtool/_commands/to_nerfstudio.py`; `_to_nerfstudio.export_to_nerfstudio`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/undistort-command.md
**Summary:** Undistorts reconstruction images with `--fit`/`--filter`/`--output`, writing a workspace layout.
**Implementing code:** `src/sfmtool/_commands/undistort.py`; `_undistort_images.undistort_reconstruction_images`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/cli/ws-init-command.md
**Summary:** Initializes a workspace `.sfm-workspace.json` with feature-tool settings and validation rules.
**Implementing code:** `src/sfmtool/_commands/ws.py` (`init`); `_workspace.init_workspace` → `sift/extract_colmap.get_colmap_feature_options`, `sift/file.get_feature_tool_xxh128`.
**Inconsistencies:**
  - The `--feature-tool` Choice in the spec (line 21) lists `colmap | opencv` only, but the code accepts `colmap | opencv | sfmtool` (`ws.py:24`, `_workspace.py:53`). The `sfmtool` backend is missing from the spec.
  - The COLMAP-only-option validation errors against any non-`colmap` tool (`ws.py:86-101`), not just `opencv` as the spec frames it.
**Recommendation:** update spec — add `sfmtool` to the `--feature-tool` Choice and generalize the validation wording. (Note: `use_gpu` persistence — the prior audit's top finding — is now correctly implemented and excluded from the cache hash via `_NON_DEFINING_OPTION_KEYS`.)
**Unclear / incorrect / suspicious:** None.

### specs/cli/xform-command.md
**Summary:** Ordered transform/filter pipeline including points-at-infinity ops and `--include-by-distribution`.
**Implementing code:** `src/sfmtool/_commands/xform.py` (`parse_transform_args`, `_auto_output_path`); `xform/*`.
**Inconsistencies:**
  - `--find-points-at-infinity` accepts an optional 4th comma part `noise_floor_px` (`xform.py:360-372`, shown in the CLI help at `:526`), but the spec's signature documents only 3: `<eps_deg>[,<desc_thresh>[,<min_views>]]` (line 168).
**Recommendation:** update spec (minor) — document the optional 4th `noise_floor_px` parameter.
**Unclear / incorrect / suspicious:** None.

### specs/cli/xform-select-by-distribution-command.md
**Summary:** `--include-by-distribution COUNT[,verbose]` — farthest-point + angular-thinning subset selection.
**Implementing code:** `src/sfmtool/xform/_select_by_distribution.py`; wired `xform.py:312-339`.
**Inconsistencies:** None material.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** Cosmetic naming mix — CLI/class are `--include-by-distribution`/`SelectByDistributionFilter` while the spec slug is `xform-select-by-distribution`. No broken cross-references.

---

## Core Specs

### specs/core/epipolar-curves.md
**Summary:** Distortion-aware epipolar "curve" plotting via ray→world→reproject sampling with adaptive subdivision. Implemented, but the documented API has drifted from the per-feature-anchor design that shipped.
**Implementing code:** `crates/sfmtool-core/src/epipolar.rs` (`plot_epipolar_curve`, `plot_epipolar_curves_batch`, `EpipolarCurveOptions`); `crates/sfmtool-py/src/py_epipolar.rs` (`epipolar_curves`); `visualization/_epipolar_display.py`.
**Inconsistencies:**
  - `EpipolarCurveOptions` (spec lines 54-69) lists an `anchor_depth` field (default 1.0); the actual struct (`epipolar.rs:116-123`) has only `curvature_tolerance` and `max_vertices`. `anchor_depth` is a function parameter, not a struct field.
  - The Rust signatures (spec 87-94, 97-104) omit the anchor argument; real signatures take `anchor_depth: f64` (`epipolar.rs:158`) and `anchor_depths: &[f64]` (`:401`).
  - The PyO3 block (spec 250-258) shows a scalar kwarg `anchor_depth: float = 1.0`; the binding (`py_epipolar.rs:33-34`) takes a required positional `anchor_depths` `(N,)` array with no such kwarg.
  - "Out of Scope" (line 288) cites `feature_match/_rectification.py`; the actual file is `feature_match/_rectified_sweep.py` (no `_rectification.py` exists).
**Recommendation:** update spec — bring the options struct, both Rust signatures, the PyO3 block, and the file path in line with the per-feature-anchor implementation; add a Status marker (it has none).
**Unclear / incorrect / suspicious:** No top-level Status marker despite being fully implemented.

### specs/core/flow-based-matching.md
**Summary:** Design/empirical rationale for flow-based feature matching (sliding-window advection + descriptor filtering). Prose-heavy, no concrete Rust symbols.
**Implementing code:** `src/sfmtool/feature_match/_flow_matching.py` (+ `_core.py`, `_run.py`); optical flow in `crates/sfmtool-core/src/optical_flow/`.
**Inconsistencies:** None at the symbol/file level.
**Recommendation:** none (optionally add a Status marker).
**Unclear / incorrect / suspicious:** Internal narrative inconsistency — a descriptor threshold of `L2 <= 100` in the table (line 39) vs "L2 <= 250" chosen elsewhere (lines 50/98/112). Worth confirming the code default.

### specs/core/gpu-optical-flow.md
**Summary:** wgpu compute-shader DIS pipeline (5 stages, hybrid CPU/GPU routing, persistent buffer pools, single-submission encoding).
**Implementing code:** `crates/sfmtool-core/src/optical_flow/gpu/` (`mod.rs`, `dis_pipeline.rs`, `pyramid_pipeline.rs`, `shaders/`). All 9 cited shaders present; `gpu_min_pixels` default 50,000 confirmed.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/core/image-warping.md
**Summary:** WarpMap generation + resampling (bilinear/anisotropic), `ray_to_pixel`, and an `Equirectangular` camera model. Building blocks are all implemented but the spec still uses "proposes" framing.
**Implementing code:** `crates/sfmtool-core/src/warp_map.rs` (`WarpMap`, `WarpMapSvd`, `from_cameras`, `compute_svd`); `remap.rs` (`ImageU8`, `ImageU8Pyramid`, `remap_bilinear`, `remap_aniso`); `distortion.rs` (`CameraModel::Equirectangular`, `ray_to_pixel`, `ray_to_pixel_batch`); `py_warp_map.rs`.
**Inconsistencies:** No divergence of substance; the symbols all landed.
**Recommendation:** update spec — change the "proposes two building blocks" / "New method needed" framing to present tense and add a Status marker. This is also where the implemented `warpmap-pose-extension.md` draft should fold in (see Drafts).
**Unclear / incorrect / suspicious:** None.

### specs/core/optical-flow.md
**Summary:** Core DIS optical-flow spec (algorithm, parameters, presets, module structure, bindings).
**Implementing code:** `crates/sfmtool-core/src/optical_flow/` (`mod.rs` presets `fast`/`default_quality`/`high_quality`, `FlowField`, `DisFlowParams`, `ImagePyramid`, `compose_flow`; `dis.rs`, `pyramid.rs`, `variational.rs`, `interp.rs`, `gpu/`); `py_optical_flow.rs`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/core/per-spherical-tile-source-stack.md — Status: Implemented
**Summary:** CSR-flat per-tile multi-source pyramid stack (rotation-only build).
**Implementing code:** `crates/sfmtool-core/src/per_spherical_tile_source_stack.rs` (`PerSphericalTileSourceStack`, `build_rotation_only`, `PatchPixel`, `BuildParams`, `BuildError::{…}`, `primary_consensus_atlas`, `consensus_patches_per_tile`); `py_per_spherical_tile_source_stack.rs`.
**Inconsistencies:** None found — status marker accurate.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/core/randomized-kdtree-forest.md — Status: Implemented (Phase 1, 2026-06-06)
**Summary:** Randomized kd-tree forest ANN index for descriptors. Highly accurate "as built" spec.
**Implementing code:** `crates/sfmtool-core/src/kdforest/` (`KdForest`, `KdForestParams` `fast`/`balanced`/`accurate`, `KdForestU8`/`F32`, `build`, `search_batch[_with_distances]`; `build.rs`, `search.rs`, `distance.rs`, `calibrate.rs`, `tests.rs`); `py_kdforest.rs`; tests + bench.
**Inconsistencies:** None found — presets, defaults, and `SFMTOOL_KDFOREST_STATS` env var all match. (Code also has an undocumented `SFMTOOL_KDFOREST_NO_SIMD` at `distance.rs:187` — extra, not a contradiction.)
**Recommendation:** none — this is the model status-marker style other core specs should follow.
**Unclear / incorrect / suspicious:** None.

### specs/core/sift.md — Status: DRAFT (stale)
**Summary:** "Planned pure-Rust SIFT detector" — but the detector has fully landed; the draft/planned marker is now inaccurate. Public API, params/keypoint fields, numeric defaults, SIMD symbols, and module layout all match the shipped code.
**Implementing code:** `crates/sfmtool-core/src/sift/` (`detect_keypoints`, `compute_descriptors`, `extract_sift`, `extract_sift_partial`, `SiftParams`, `SiftKeypoint`, `Detection`, `SiftFeatures`; `scale_space.rs`, `detect.rs`, `orientation.rs`, `descriptor.rs`, `gray.rs`, `simd.rs`); `py_sift.rs`, `py_sift_io.rs`; `tests/test_sift_rust_bindings.py`.
**Inconsistencies:**
  - Status marker stale — header (lines 1-9) declares "Status: draft … a *planned* SIFT detector"; the module-structure and Python-bindings sections are labeled "(planned)" though they exist. Defaults (`s=3`, `σ=1.6`, `contrast_threshold=0.0067`, `r=10`, `max_num_features=8192`, 36 bins, 4×4×8) and SIMD symbols (`HAS_AVX2_FMA`, `atan2_approx`, `exp_approx`, `l2_norm_sse2`/`scale_sse2`) all confirmed present.
**Recommendation:** update spec — flip Status to Implemented (Phase 1), as was done for `randomized-kdtree-forest.md`. Keep the on-disk incremental extraction (`--detect`/`--describe`/`--top-k`, chunked descriptors) marked as future work — it is genuinely unimplemented.
**Unclear / incorrect / suspicious:** None.

### specs/core/spherical-tiles-rig.md — Status: Implemented
**Summary:** Sphere-as-pinhole-tile-rig with atlas packing and resampling.
**Implementing code:** `crates/sfmtool-core/src/spherical_tile_rig.rs` (`SphericalTileRig`, `set_patch_size`, `tiles_subset`, `resample_atlas`, `warp_to/from_atlas_with_rotation`, `tile_camera`); `sphere_points.rs`; `py_spherical_tile_rig.rs`; `_spherical_tile_rig.py`.
**Inconsistencies:**
  - Internal doc-comment contradiction: the `warp_from_atlas_with_rotation` "Why closest-tile" note (line 364) states `half_fov_rad = 0.5 · measured_max_nn_angle · overlap_factor`, while the authoritative definition (struct field line 174, constructor step 4 line 103) is `half_fov_rad = measured_max_coverage_angle · overlap_factor` (no 0.5, coverage-angle not nn-angle).
**Recommendation:** update spec — reconcile the line-364 formula with the coverage-angle definition; confirm code uses the coverage-angle form.
**Unclear / incorrect / suspicious:** None (both `from_cameras_with_pose`/`with_rotation` exist).

### specs/core/tile-batched-consensus-atlas.md — Status: Implemented
**Summary:** Bounded-memory batched panorama compositing orchestrator.
**Implementing code:** `crates/sfmtool-core/src/consensus_atlas.rs` (`render_consensus_atlas`); `spherical_tile_rig.rs::tiles_subset`; `per_spherical_tile_source_stack.rs::consensus_patches_per_tile`; `photometric_ransac.rs` (`tile_index_base`, RNG derivations); `py_consensus_atlas.rs`; `_panorama.py`.
**Inconsistencies:** None found — determinism plumbing matches.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

---

## Format Specs

### specs/formats/sfmr-file-format.md
**Summary:** `.sfmr` ZIP+zstd columnar format at version 2 (homogeneous `(x,y,z,w)` points, points at infinity) with v1 read compatibility. Camera parameter tables match `_CAMERA_PARAM_NAMES`.
**Implementing code:** `crates/sfmr-format/src/{write,read,verify,types}.rs` (`write.rs:65` forces v2), `crates/sfmtool-core/src/camera_intrinsics.rs` (`CameraModel`), `_cameras.py` (`_CAMERA_PARAM_NAMES`), `_colmap_io.py`.
**Inconsistencies:**
  - The camera-model table (lines 271-281) omits `EQUIRECTANGULAR`, but `camera_intrinsics.rs:135` defines a `CameraModel::Equirectangular` variant (string name `"EQUIRECTANGULAR"` at `:161`). It is writable/readable through the Rust core but undocumented here (and absent from Python `_CAMERA_PARAM_NAMES` — see camrig).
**Recommendation:** update spec — add an `EQUIRECTANGULAR` row (note no-distortion semantics) or explicitly state it is Rust-core-only.
**Unclear / incorrect / suspicious:** None.

### specs/formats/camrig-file-format.md
**Summary:** `.camrig` ZIP+zstd format (version 1): camera pool, per-sensor poses, image patterns, rig_type/attributes.
**Implementing code:** `crates/camrig-format/src/{lib,types}.rs` (writes v1, rejects others); `camrig/create.py`, `_pano2rig.py`, `_insv2rig.py`.
**Inconsistencies:**
  - The spec cites `EQUIRECTANGULAR` as an example camera model (line 134) and shows it as a valid `.camrig` camera, but `camrig/create.py:164,235` builds cameras via `_CAMERA_PARAM_NAMES[model]` (`_cameras.py:21`), which has **no** `EQUIRECTANGULAR` entry — a Python-built `.camrig` with `model: "EQUIRECTANGULAR"` would `KeyError`. The model is only supported on the Rust side.
**Recommendation:** discuss — either add `EQUIRECTANGULAR` to `_CAMERA_PARAM_NAMES` (aligning the Python camrig/sfmr paths with Rust and the spec example) or change the spec example to a Python-supported model. This is a real code gap, not just a doc issue.
**Unclear / incorrect / suspicious:** None beyond the above.

### specs/formats/matches-file-format.md
**Summary:** `.matches` ZIP+zstd format (version 1): images, image_pairs, optional two_view_geometries.
**Implementing code:** `crates/matches-format/src/{lib,types}.rs` (writes v1); `feature_match/`, `_colmap_db.py`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/formats/sift-file-format.md
**Summary:** `.sift` ZIP+zstd feature format — v1 plus a clearly-marked v2 draft, with an explicit "Implementation status" note that the codebase reads/writes v1 only.
**Implementing code:** `crates/sift-format/src/{lib,types}.rs` (writes v1); `sift/{file,extract_*}.py`.
**Inconsistencies:** None found — the v2 divergence is intentional draft, honestly flagged.
**Recommendation:** none — the `[v2]` tagging + status banner is the right pattern.
**Unclear / incorrect / suspicious:** None.

---

## Workspace Specs

### specs/workspace/workspace.md
**Summary:** Workspace concept, `.sfm-workspace.json` schema (version 1), discovery, path resolution, lifecycle.
**Implementing code:** `src/sfmtool/_workspace.py`, `_camera_config.py`.
**Inconsistencies:** None found.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/workspace/camera-config.md
**Summary:** Optional per-directory `camera_config.json` intrinsics (v1), closest-ancestor resolution capped at root, presence-based `--camera-model` rejection.
**Implementing code:** `src/sfmtool/_camera_config.py` (`CameraConfigResolver`, `load_camera_config`), `_camera_setup.py`, `_cameras.py`.
**Inconsistencies:** None found — the prior `--switch-camera-model` cross-reference bug is gone (grep finds no occurrences; spec consistently uses `--camera-model`).
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

### specs/workspace/rig-config.md
**Summary:** Optional `rig_config.json` (COLMAP rig_configurator format verbatim), frame grouping, per-sensor intrinsics.
**Implementing code:** `src/sfmtool/_rig_config.py` (`_sensor_from_rig_pose` WXYZ→XYZW), `_rig_frames.py` (`_build_cross_frame_pairs`), `_camera_setup.py` (`_infer_camera` `min(w,h)/π`), `crates/sfmtool-core/src/distortion.rs` (`blend_fisheye_ray`).
**Inconsistencies:** None found — the pano2rig→`.camrig` correction is confirmed accurate; all cited symbols verified.
**Recommendation:** none.
**Unclear / incorrect / suspicious:** None.

---

## GUI Specs

Most GUI specs match the code at the feature level. The recurring problem is
**status hygiene**, not behavioral drift: several specs written as forward-looking
"Plan/Design" proposals have shipped without being re-marked, and `gui-plan.md`'s
snapshot is the oldest in the tree.

### specs/gui/README.md — index, accurate and complete. No action.
### specs/gui/blender-viewport-navigation-implementation-overview.md — external reference doc (Blender); no consistency obligation. No action.
### specs/gui/gui-architecture.md — every pipeline/pass described has a matching module under `scene_renderer/pipelines/` and `shaders/`. No drift; add a Status/date marker for future audits.
### specs/gui/gui-user-experience.md — vision/principles doc, no concrete feature claims. No action.

### specs/gui/gui-adaptive-clip-and-grid.md — Status: Implemented (2026-04-05)
**Inconsistencies:** None found — reversed-Z infinite-far projection, adaptive near plane, adaptive grid all present (`viewer_3d/camera.rs`, `scene_renderer/uniforms.rs`).
**Recommendation:** none.

### specs/gui/gui-camera-views.md
**Inconsistencies:** Unchecked `[ ]` items confirmed absent from code (camera up-indicator, async thumbnails, FULL_OPENCV distortion, BC7/ASTC textures, fisheye viewport) — correctly pending, not drift. Implemented features (frustum/image-quad/distorted-quad pipelines, GPU pick, view-through-camera) all present.
**Recommendation:** none.

### specs/gui/gui-cross-panel-hover.md — Status: Implemented
**Inconsistencies:** None found — `state.rs` `hovered_image`/`hovered_point` exist as specified.
**Recommendation:** none.

### specs/gui/gui-image-animation.md
**Inconsistencies:** Stale framing — reads as an unimplemented proposal (future tense, "Implementation Plan" Steps 1-4) but `AnimationState`, `PlayDirection`, keyboard controls, minibar transport, and camera-switch sync all exist (`image_browser.rs`, `dock.rs`). No "Status: Implemented" marker.
**Recommendation:** update spec — add a `Status: Implemented (date)` marker and convert plan/proposal framing to present tense.

### specs/gui/gui-multi-panel-image-browser.md
**Inconsistencies:** "Plan:" / proposal framing despite the multi-panel dock, browser strip, and detail pane being implemented (`dock.rs`, `image_browser.rs`, `image_detail.rs`). Stale file reference — points to "`viewer_3d.rs` (line ~1280)" but the viewer is now the `viewer_3d/` directory (no `viewer_3d.rs`).
**Recommendation:** update spec — add Status: Implemented, drop "Plan:" framing, fix the `viewer_3d.rs` path to the current `viewer_3d/` modules.

### specs/gui/gui-plan.md — "Current Implementation Status" *Updated: 2026-03-27* (oldest GUI snapshot, now lagging)
**Inconsistencies:**
  - The status list omits shipped features: image animation/playback (`image_browser.rs` `AnimationState`), cross-panel hover (`state.rs`), and lists Point Track Detail only as "Phase A complete" though `gui-point-track-detail.md` is marked fully Implemented.
  - "Image Browser → Planned enhancements: Animation mode" is listed as planned but is implemented. ("Grid mode" multi-row layout remains genuinely unimplemented — `image_browser.rs` is strip-only.)
  - "Next Steps" items 1 (grid depth occlusion) and 4 (adaptive clip/grid) overlap with `gui-adaptive-clip-and-grid.md` (Implemented 2026-04-05), which already delivered depth-occluding grid.
**Recommendation:** update spec — refresh the status date, move animation/hover/Point-Track-Detail into the implemented list, and reconcile the Next-Steps items already delivered.

### specs/gui/gui-point-cloud-rendering.md
**Inconsistencies:** 12 `[x]` / 6 `[ ]`; unchecked items (adaptive length_scale, target visibility over bright bg, color-by-metric for the 3D cloud, quality filtering, LOD for 10M+, distance attenuation) confirmed unimplemented — correct gaps.
**Recommendation:** none.

### specs/gui/gui-point-track-detail.md — Status: Implemented
**Inconsistencies:** None at feature level (`point_track_detail.rs`, `Tab::PointTrackDetail`, `track_ray` pipeline). Only the cross-reference in `gui-plan.md` (which calls it "Phase A complete") needs reconciling — fix there, not here.
**Recommendation:** none for this spec.

### specs/gui/gui-viewport-navigation.md
**Inconsistencies:** `[ ] FOV zoom` is marked fully unchecked with a "(Planned)" section, but FOV zoom via Ctrl+drag in camera view **is** implemented (`viewer_3d/input.rs:105`) and documented to users in `docs/tutorials/getting-started.md` ("hold Control to zoom the field of view"). Only the *free-navigation* FOV-zoom binding is missing. Genuinely-unimplemented `[ ]` items (mouse-drag tilt/roll, configurable sensitivity, zoom-to-cursor, inertial scrolling, save/restore camera positions) are correctly pending — no `bookmark`/`saved_view` symbols exist.
**Recommendation:** update spec — split the FOV-zoom item into "camera-view Ctrl+drag (done)" vs "free-navigation gesture (planned)"; the as-shipped, user-documented behavior contradicts the single unchecked box and the "(Planned)" section's "not needed in camera view" framing.

### docs/index.md
**Inconsistencies:** None found — all shown commands (`ws init`, `solve -g`, `inspect`, `analyze --metrics`) exist.
**Recommendation:** none.

### docs/tutorials/getting-started.md
**Inconsistencies:** None material — the discontinuity→motion rename is correctly reflected (`sfm motion`), and the GUI controls shown match `viewer_3d/input.rs`. The "hold Control to zoom the FOV" line is accurate to the code (and is what exposes the `gui-viewport-navigation.md` checklist contradiction above).
**Recommendation:** none.

---

## Draft Specs

The drafts directory is the single biggest source of status drift: **five of six
drafts are implemented and ready to graduate or retire.** Only one is still
genuinely forward-looking.

### specs/drafts/sfmr-v2-points-at-infinity.md — RETIRE
Fully implemented and already canonicalized as version 2 in the non-draft
`specs/formats/sfmr-file-format.md` §7 (+ v1→v2 appendix). Implemented in
`crates/sfmtool-core/src/infinity.rs`, `reconstruction.rs`, `py_sfmr_reconstruction.rs`,
and wired via `--detect-infinity` on `solve`/`from-colmap-bin`. The draft is now a
redundant duplicate, and its classification math (`α_max·f_max < noise`) is the
*old* approach that `batch-triangulation-api.md` replaced with `inverse_depth_z`.
**Recommendation:** retire (git preserves history) — the format spec is the source of truth.

### specs/drafts/batch-triangulation-api.md — PROMOTE to specs/core
Status marker already reads "Implemented (all four migration phases landed)."
Batch triangulation, observability diagnostics, the scene-relative `indeterminate`
gate, bindings, and GUI overlays all exist (`triangulation.rs`, `infinity.rs`,
`py_triangulation.rs`; GUI `point_track_detail.rs`, `image_detail.rs` overlay modes,
`colormap.rs`; CLI `analyze --depth-reliability`). Only the threshold *calibration*
sub-questions remain (the spec flags them as deferred tuning).
**Recommendation:** promote to specs/core as the triangulation/observability design of record; keep calibration as an Open question.

### specs/drafts/photometric-subsets-ransac.md — PROMOTE to specs/core
The spec's own promotion trigger ("folds into specs/core once the production
pipeline consumes its outputs") has fired: `_panorama.py` consumes
`render_consensus_atlas`, built on `photometric_ransac.rs` (with PyO3 bindings).
Algorithm, knobs, primary+secondary per-tile RANSAC, fallback, and MAD reporting all map to code.
**Recommendation:** promote to specs/core; cross-link `consensus_atlas` as the production consumer.

### specs/drafts/warpmap-pose-extension.md — PROMOTE / fold into image-warping
Status marker reads "Implemented." Both construction paths
(`from_cameras_with_rotation`, `from_cameras_with_pose`, shared `build_with_pose_impl`),
the bindings, and the test suite exist (`warp_map.rs`, `py_warp_map.rs`,
`tests/test_warp_map_pose.py`).
**Recommendation:** fold into `specs/core/image-warping.md` (which it explicitly extends) and remove from drafts/.

### specs/drafts/xform-find-points-at-infinity.md — PROMOTE to specs/cli
Status marker reads "Implemented." The discovery core (`find_infinity.rs`), the
PyO3 `find_points_at_infinity`, and the CLI surface
(`xform --find-points-at-infinity`, `--classify-points-at-infinity`, `--max-features`)
all exist (`_commands/xform.py`, `xform/_find_points_at_infinity.py`).
**Recommendation:** promote into `specs/cli/` alongside `xform-command.md`; update step 6 to point at the `inverse_depth_z`/`indeterminate` classifier rather than the legacy `α_max·f_max` cut, so it doesn't contradict `batch-triangulation-api.md`.

### specs/drafts/gui-points-at-infinity.md — UPDATE then fold into gui-point-cloud-rendering
Status marker ("the GUI does not yet render `w = 0` points") is **stale**: rendering
landed in commit c3c2805. Implemented across `shaders/points.wgsl`,
`scene_renderer/{gpu_types,uniforms,upload,auto_point_size}.rs`, `point_track_detail.rs`,
`app.rs`/`readback.rs`. The one unbuilt piece is §5's UI controls — the
"Show points at infinity" visibility toggle and the `N points (M at infinity)`
count readout (no such strings found).
**Recommendation:** update status to "Implemented (minus §5 toggle/count)", fold into `specs/gui/gui-point-cloud-rendering.md` per the draft's own closing line, and either implement or explicitly defer the toggle/count.

### specs/drafts/photometric-subsets-ransac.md and the rest aside, the only still-speculative draft:
*(none)* — every remaining draft is implemented. There is no purely forward-looking draft left in `specs/drafts/`.

---

## Code Without Specs

24 of 25 CLI commands have a `specs/cli/<name>-command.md`. The single
user-facing gap:

### `sfm explorer` (src/sfmtool/_commands/explorer.py)
**What it does:** Launches the native SfM Explorer GUI by shelling out to `launch-sfm-explorer`, optionally opening a `.sfmr`. The CLI front door to the viewer (same binary as `pixi run gui`).
**Why it matters:** user-facing.
**Recommendation:** add a note to existing spec — the GUI is covered by 13 `specs/gui/*` files, but nothing documents the `sfm explorer` CLI entry point (its single optional argument, executable-discovery/error behavior, relationship to `pixi run gui`). A short `specs/cli/explorer-command.md` cross-linking the GUI specs, or a CLI section in `specs/gui/README.md`, would close the per-command spec set.

Other surfaces are adequately covered or acceptably unspecced:
- `sfm version` (`cli.py`) — small utility; acceptable (string is hardcoded, a hygiene note not a spec gap).
- `crates/sfmtool-py` — internal binding glue; surface is the union of the core/format specs. Acceptable.
- `crates/sfm-explorer` — covered indirectly by the 13 `specs/gui/*` specs (the only thin spot is the `explorer` CLI entry above).
- `sift-format`/`matches-format`/`sfmr-format`/`camrig-format` → their format specs; `sfmr-colmap` → the COLMAP-interop command specs; `sfmtool-core` → the `specs/core/*` specs.

---

## Top Priorities

1. **Graduate the drafts directory.** Five of six drafts are implemented:
   **retire** `sfmr-v2-points-at-infinity.md` (superseded by the format spec §7);
   **promote** `batch-triangulation-api.md` and `photometric-subsets-ransac.md`
   to `specs/core/`, `xform-find-points-at-infinity.md` to `specs/cli/`, and
   **fold** `warpmap-pose-extension.md` into `image-warping.md` and
   `gui-points-at-infinity.md` into `gui-point-cloud-rendering.md`. This is the
   largest, cleanest cleanup. *(specs/drafts/*)*

2. **`sfm flow --pairs-dir` is a documented dead option.** Declared but never
   consumed (`flow.py:79-84,97`); the spec's batch mode (lines 28, 56-58) does
   nothing. Either implement it (mirroring `epipolar`) or remove it from the spec.
   *(specs/cli/flow-command.md)* — **real behavioral gap.**

3. **`EQUIRECTANGULAR` camera model is Rust-only.** `camera_intrinsics.rs:135`
   defines it and both the `.sfmr` and `.camrig` specs reference it, but Python
   `_CAMERA_PARAM_NAMES` (`_cameras.py:21`) lacks an entry, so a Python-built
   `.camrig`/`.sfmr` with `model: "EQUIRECTANGULAR"` would `KeyError`. Either add
   it to `_CAMERA_PARAM_NAMES` or document it as Rust-core-only and change the
   spec examples. *(specs/formats/{camrig,sfmr}-file-format.md)* — **real code gap.**

4. **Three concrete CLI spec/code mismatches.** (a) `analyze --depth-reliability`
   mode and the `--samples` min=100 are missing from the analyze spec;
   (b) `ws init --feature-tool` accepts `sfmtool` but the spec lists only
   `colmap|opencv`; (c) `sift` PATHS is required in code but the spec says it
   defaults to the workspace root. *(specs/cli/{analyze,ws-init,sift}-command.md)*

5. **Stale/missing core status markers.** `specs/core/sift.md` still says
   "draft/planned" though the Rust SIFT detector fully shipped — flip it to
   Implemented (as done for `randomized-kdtree-forest.md`). Add Status markers to
   `epipolar-curves.md`, `image-warping.md`, `optical-flow.md`, `gpu-optical-flow.md`,
   and refresh `gui-plan.md`'s 2026-03-27 snapshot plus the "Plan/proposal"
   framing in `gui-image-animation.md` / `gui-multi-panel-image-browser.md`.
   `epipolar-curves.md` additionally needs its `anchor_depth` API blocks and the
   `_rectification.py` path corrected.

Lower-priority carry-forwards: the `FULL_OPENCV` omission from the `match`/`solve`
`--camera-model` Choice lists (needs a decision), the `xform --find-points-at-infinity`
4th `noise_floor_px` parameter undocumented, `scale-by-measurements` diagnostics
examples + invalid-hex Point IDs, the `spherical-tiles-rig.md` `half_fov_rad`
formula contradiction, the `insv2rig.py:19` "~29mm" code comment, and a
one-paragraph `explorer-command.md`.
