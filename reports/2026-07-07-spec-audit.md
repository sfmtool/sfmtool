# Spec audit — 2026-07-07

Bidirectional audit of every file under `specs/` and `docs/` against the implementing
code, plus a code→spec sweep for significant unspecced surfaces. Produced by the
`audit-specs` skill; read-only analysis of the tree at commit `a95c72d`.

Method: 11 parallel subagent passes — CLI specs (×4 groups), core specs (×3 groups),
formats/workspace/drafts, GUI specs, user docs, and a code-without-specs sweep. Every
spec file was read in full and compared against its implementing code with concrete
flag/default/constant/symbol checks.

**Headline:** the spec tree is in very good shape — the 2026-06-23 audit backlog work
(#176, a95c72d) paid off. Of 45 spec/doc files audited, 32 are fully in sync. The
remaining divergences cluster in two areas: (1) the fast-moving patch/keypoint pipeline
(#174/#175 shipped features whose home specs still describe pre-landing designs), and
(2) GUI specs carrying stale internal-contradiction and renamed-symbol references. Two
CLI specs document behavior the code doesn't deliver (dead `align` options;
`scale-by-measurements` resolution fallbacks).

---

## CLI command specs

### specs/cli/align-command.md
**Summary:** Aligns multiple `.sfmr` reconstructions into a reference coordinate frame via point-based (RANSAC) or camera-pose-based methods, with multi-reconstruction connectivity handling and basename-collision rejection on output.
**Implementing code:** `src/sfmtool/_commands/align.py::align` (CLI); `src/sfmtool/align/multi.py::align_command`, `align_reconstructions`, `_align_with_points`, `_align_with_cameras`, `_build_connectivity_graph`.
**Inconsistencies:**
  - `--max-error` (spec line 32, "Maximum acceptable alignment error", default 0.1), `--iterative` (line 33), and `--visualize` (line 34) are documented as functional but are **dead options**. `align.py:75-87` accepts them and passes them to `align_command`, but `multi.py:315-317` receives `max_error`/`iterative`/`visualize` and never references them (confirmed: the only occurrences in the whole `align/` package are the parameter declarations themselves). Nothing is refined iteratively, no error threshold is enforced, no visualization is produced.
  - Spec line 42-43 says the connectivity graph determines alignment order so each reconstruction is "aligned through the **shortest path** to the reference." The implementation (`multi.py:225-268`) does greedy iterative expansion, aligning each not-yet-aligned reconstruction to its **first** aligned neighbor (`first_neighbor = aligned_neighbors[0]`, line 270), not a shortest-path computation. Behaviorally similar for simple topologies but not literally shortest-path.
**Recommendation:** update code or spec — decide whether `--max-error`/`--iterative`/`--visualize` should be implemented or removed; until then the spec overstates the tool. The "shortest path" wording should be softened to "iterative/greedy connectivity expansion."
**Unclear / incorrect / suspicious:** The three no-op options are the suspicious item — they are accepted silently with no warning, so a user passing `--iterative` gets no error and no effect.

### specs/cli/analyze-command.md
**Summary:** Deep-analysis report on a `.sfmr` file with six mutually-exclusive modes (`--coviz`, `--z-range`, `--frustum`, `--images`, `--metrics`, `--depth-reliability`), plus frustum percentile/sample options and per-image reprojection metrics.
**Implementing code:** `src/sfmtool/_commands/analyze.py::analyze`; `src/sfmtool/analyze/metrics.py::print_metrics_analysis`, `_compute_per_image_metrics`; `analyze/graphs.py`, `analyze/depth.py`, `analyze/images.py`.
**Inconsistencies:**
  - None found. Mode exclusivity (`analyze.py:147-167`), `--range` gated to `--metrics` (169-170), frustum-only gating of `--near/--far-percentile`/`--samples` (172-178), and `near < far` validation (180-184) all match the spec. Metrics outlier flags match exactly: `!!` at >2× median, `!` at >1.5× median, `--` at zero observations (`metrics.py:203-208, 191-196`), sorted by mean error descending (`metrics.py:120-124`), zero-point early return with message (`metrics.py:106-111`), and N/A error metrics for zero-obs images (`metrics.py:191-196`).
**Recommendation:** in sync — spec and code agree on flags, defaults, gating, and metric semantics.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/camrig-command.md
**Summary:** `camrig` command group with `create` (build a one-camera rig from an image directory), `cp` (copy a rig/camera/sensor-subset out of a `.sfmr` or `.camrig`), and `spherical-tiles` (build a discretised-sphere tile rig).
**Implementing code:** `src/sfmtool/_commands/camrig.py::camrig`, `create`, `cp`, `spherical_tiles`; delegates to `camrig/create.py::build_camrig_from_images`, `camrig/cp.py::copy_from_sfmr`/`copy_from_camrig`, `_sfmtool/spherical.py::SphericalTileRig`.
**Inconsistencies:**
  - None found in the CLI surface. All `create` options (`--camera-model`, `--resolution`, `--focal-length[-x/-y]`, `--principal-point-x/-y`, `--params`, `--name`) match spec lines 47-58. All `cp` selectors/options match (`--rig`/`--camera`/`--sensors`/`--pattern`/`--name`) including the mutual-exclusion rejections at `camrig.py:212-221` and the `.sfmr`-vs-`.camrig` selector-type rejections at `226-244`, matching spec lines 133-135. `spherical-tiles` options and the "exactly one of `--equirect-width`/`--arc-per-pixel`" rule (`camrig.py:353-356`) match spec lines 220-221; `--overlap-factor` 1.15, `--centre` `0 0 0`, `--atlas-cols` default `ceil(sqrt(n))` all match.
**Recommendation:** in sync — CLI, defaults, and validation match the spec.
**Unclear / incorrect / suspicious:** Spec line 212 documents `--n` as `>= 2`, and the help text says the same, but `camrig.py:276-281` types it as a plain `int` with no `IntRange(min=2)` — the `>= 2` bound is presumably enforced inside `SphericalTileRig`. Not a divergence if the constructor validates it, but the bound is not enforced at the Click layer as it is for other integer options.

### specs/cli/compare-command.md
**Summary:** Compares two `.sfmr` reconstructions across alignment, intrinsics, poses, feature usage, and 3D points (with scale-relative metrics), plus an extensive `--strips*` side-by-side patch-strip montage diagnostic.
**Implementing code:** `src/sfmtool/_commands/compare.py::compare`, `_parse_labels`; `src/sfmtool/_compare.py::compare_reconstructions`, `_characteristic_scene_scale`, `_compare_3d_points`.
**Inconsistencies:**
  - None found. Every `--strips*` option and default matches (`--strips-num` 16, `--strips-views` 8, `--strips-context` 96, `--strips-rank` overview with the full 8-choice list, `--strips-end`, `--strips-refine/--strips-no-refine` default true, `--strips-labels` `reference,target`). `--pixel-threshold` default 2.0 matches. The auto-select default for point correspondence (`_compare.py:172-178`: coordinate matching iff all shared images use different SIFT content hashes) matches spec lines 58-59. Scene-scale RMS-radius normalization, `w==0` infinity exclusion, and the `IDENTICAL`/`VERY SIMILAR`/`SIGNIFICANT DIFFERENCES` scale-relative conclusion all appear in `_compare.py` (lines 21-29, 264-301, 696-721) matching spec steps 3 and 5.
**Recommendation:** in sync — the detailed prose behaviors and every CLI option match the implementation.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/densify-command.md
**Summary:** Experimental point-cloud densification via sweep matching over covisible (and optionally frustum-intersecting) image pairs, with bundle-adjustment, point-filtering, and geometric-filtering options.
**Implementing code:** `src/sfmtool/_commands/densify.py::densify`; `src/sfmtool/_densify.py::densify_reconstruction`; `feature_match.GeometricFilterConfig`.
**Inconsistencies:**
  - None found. Every documented option and default matches the code: `--sweep-window-size` 30, `--close-pair-threshold` 4, `--max-distant-pairs` 5000, `--distant-pair-search-multiplier` 3, `--filter-max-reproj-error` 4.0, `--filter-min-track-length` 3, `--filter-min-tri-angle` 1.5, `--filter-isolated-median-ratio` 2.0, `--geometric-size-ratio-max` 1.25, `--geometric-angle-diff-max` 15.0 (`densify.py:27-124`). The `--max-features`/`--max-close-pairs`/`--distance-threshold` "all/None" defaults match. `--enable-geometric-filtering` derives `size_ratio_min = 1/size_ratio_max` (`densify.py:186`), consistent with the "motion-invariant geometric filtering" description.
**Recommendation:** in sync — including the spec's own "experimental / not well tuned" caveat, which the CLI docstring echoes.
**Unclear / incorrect / suspicious:** Nothing (the spec's step 6 "Align back to original frame" and step 4 "Bundle adjust" are internal to `densify_reconstruction`; not separately audited here but the CLI-visible surface is fully consistent).

### specs/cli/embed-patches-command.md
**Summary:** Converts a `sift_files` reconstruction to an `embedded_patches` `.sfmr` via a Rust `to_embedded_patches` bridge plus normal-refinement / view-selection / keypoint-localization kernels, with a large set of photometric tuning options.
**Implementing code:** `src/sfmtool/_commands/embed_patches.py::embed_patches_command`; `src/sfmtool/_embed_patches.py::embed_patches`; output naming via `xform/_arg_parser.py::auto_output_path`.
**Inconsistencies:**
  - None found. All 15 options match spec defaults exactly: `--min-relative-zncc` 0.7, `--max-iters` 5, `--search` 6.0, `--max-shift-px` 3.0, `--min-views` 2, `--patch-size` 5.0, `--search-resolution-multiplier` 1.0, `--subpixel` 1, `--rounds` 2, `--max-obliquity-deg` 80.0, `--obliquity-weight-power` 2.0, `--fronto-prior-weight` 0.05, `--refine-max-views` 8, `--localize-search-strategy` `plus_descent` (`embed_patches.py:19-183`). Output defaulting to `<stem>-embedded.sfmr` with numeric suffix from 2 is confirmed in `auto_output_path` (`_arg_parser.py:370-386`, called at `embed_patches.py:244`). The "input already embedded → error" (spec line 119) is enforced at `embed_patches.py:249-253`.
**Recommendation:** in sync — this spec is unusually detailed and the code tracks it option-for-option.
**Unclear / incorrect / suspicious:** Minor: spec's `--rounds` help (line 72) describes a richer per-round mean-normal-change/keypoint-shift progress print; the CLI help text (`embed_patches.py:98-110`) is a condensed paraphrase. Behavior is delegated to `_embed_patches.embed_patches` with `progress=click.echo`, so this is presentational only, not a divergence.

### specs/cli/epipolar-command.md
**Summary:** Visualizes epipolar geometry (lines or, for wide-FOV cameras, curves) between an image pair, with single-pair and `--pairs-dir` batch modes, plus rectification/undistortion and optional sweep-matching overlay.
**Implementing code:** `src/sfmtool/_commands/epipolar.py::epipolar`, `resolve_image_name`; `visualization/_epipolar_display.py::draw_epipolar_visualization`.
**Inconsistencies:**
  - None found. Options and defaults match: `--line-thickness` 1, `--feature-size` 3, `--rectify/--no-rectify` false, `--undistort` flag with mutual-exclusion vs `--rectify` (`epipolar.py:217-218`), `--draw-lines/--no-lines` true, `--side-by-side/--separate` false, `--sweep-window-size` default applied as 30 only when `--sweep-with-max-features` is set (`epipolar.py:225-226`). Batch-mode naming `<stem>.png` per image and the "last image paired backward with predecessor" behavior match spec lines 46-52 (`epipolar.py:281-335`), and `save_which="first" if not side_by_side else "both"` implements "only the named image's frame is saved when not side-by-side" (lines 304, 330).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Spec option table (line 33) documents the separate-mode second file as suffixed `_other`, and the `--draw` help repeats it, but the exact suffixing lives inside `draw_epipolar_visualization` (not re-audited line-by-line). No conflict observed.

### specs/cli/flow-command.md
**Summary:** Computes dense DIS optical flow between two images, visualizes SIFT keypoint advection (Middlebury color wheel), and optionally compares flow correspondences against a reconstruction's matches (green/red/yellow coding).
**Implementing code:** `src/sfmtool/_commands/flow.py::flow`; `visualization/_flow_display.py::draw_flow_visualization`.
**Inconsistencies:**
  - None found. Options and defaults match: `--preset` choices `fast|default|high_quality` default `default`, `--tolerance` 3.0, `--descriptor-threshold` 100.0, `--feature-size` 4, `--line-thickness` 1, `--side-by-side/--separate` false (`flow.py:29-81`). Comparison-mode color semantics (green=agreement, red=reconstruction-only, yellow=flow-only) match spec lines 40-42 (`flow.py:107-112`). The spec's "no batch mode" note is consistent — there is no `--pairs-dir` on `flow`.
**Recommendation:** update spec (minor) — the code writes derived output files (`<stem>_flow`, `<stem>_A`, `<stem>_B` per `flow.py:24-28`) that the spec's `--draw` row (line 20) and Output prose do not mention; a reader would not learn that `--draw output.png` (without `--side-by-side`) produces multiple files. Otherwise in sync.
**Unclear / incorrect / suspicious:** Nothing beyond the undocumented derived-file naming noted above.

### specs/cli/from-colmap-bin-command.md
**Summary:** Documents `sfm from-colmap-bin`, which imports a COLMAP binary reconstruction (cameras.bin/images.bin/points3D.bin) into a single `.sfmr` file, applying the COLMAP→canonical coordinate conversion, with a required `--image-dir`, required `-o`, `--tool-name`, and `--detect-infinity` toggle.
**Implementing code:** `src/sfmtool/_commands/from_colmap_bin.py` (`from_colmap_bin`), `src/sfmtool/colmap/io.py` (`build_metadata`, `colmap_binary_to_rust_sfmr`).
**Inconsistencies:**
  - None found. All four options and their defaults match (`--tool-name` default `unknown` at `from_colmap_bin.py:36`; `--detect-infinity/--no-detect-infinity` default `True` at `:39-44`, wired to `classify_infinity=detect_infinity` at `:135`).
**Recommendation:** in sync — flags, defaults, and the import/convention behavior all agree.
**Unclear / incorrect / suspicious:** The code enforces a `.sfmr` output extension (`from_colmap_bin.py:84-87`) and records `operation="import"` (`:117`); neither is mentioned in the spec, but both are benign details, not divergences.

### specs/cli/heatmap-command.md
**Summary:** Documents `sfm heatmap`, which draws per-feature quality metrics (reproj/tracks/angle/all) as colored circle overlays on images, with per-metric default colormaps, and describes the metric-before-trailing-number output naming with path flattening.
**Implementing code:** `src/sfmtool/_commands/heatmap.py` (`heatmap`, `_insert_metric_before_number`, `_output_stem`), `src/sfmtool/visualization/_heatmap_renderer.py` (`render_heatmap_overlay`, `compute_triangulation_angles`).
**Inconsistencies:**
  - None found. Options and defaults match (`--metric` default `all` at `heatmap.py:70`, `--colormap` default `None`→auto at `:75-81`, `--radius` 5 at `:85`, `--alpha` 0.7 at `:93`). Default colormaps `reproj→error`, `tracks→tracks`, `angle→viridis` match spec table (`:148-152`). Output naming and `/`→`__` flattening implemented exactly as specified (`_output_stem` at `:43-52`, `_insert_metric_before_number` at `:20-40`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/inspect-command.md
**Summary:** Documents `sfm inspect` for single files/images/3D-point-IDs plus the `--strips` patch-montage mode. Covers per-type summary/verbose printers, integrity checks, point-ID hash resolution and search order, and the strips point-spec grammar and options.
**Implementing code:** `src/sfmtool/_commands/inspect.py` (`inspect`, `_inspect_strips_cmd`, `_inspect_point`, `_find_sfmr_by_content_hash`), `src/sfmtool/analyze/summary.py`, `src/sfmtool/_inspect_strips.py` (`parse_point_specs`, `render_inspect_strips`).
**Inconsistencies:**
  - None found. Option defaults match (`--strips-views` default 8 at `inspect.py:60-63`, `--context` default 1.0 at `:65-72`). Point-ID regex `pt3d_<8hex>_<index>` matches spec (`:31`). Search order sfmr/ → root → rest (hidden skipped) matches spec §"3D Point IDs" (`_find_sfmr_by_content_hash` at `:244-278`). Default strips output `<stem>_strips.png` in cwd matches (`:197`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** The guard rejecting `-o`/`--strips-views`/`--context` without `--strips` compares against default *values* (`if output is not None or strips_views != 8 or context != 1.0`, `inspect.py:129`) rather than using `ctx.get_parameter_source(...)` (as `match.py` does). So explicitly passing the default (e.g. `--strips-views 8`) without `--strips` is silently accepted instead of rejected. Minor; the spec's "rejected unless --strips is given" is slightly stronger than the implementation.

### specs/cli/insv2rig-command.md
**Summary:** Documents `sfm insv2rig`, which extracts dual-fisheye frames from Insta360 `.insv` video (dual-stream or side-by-side) into `fisheye_left/` and `fisheye_right/`, and writes a two-sensor `fisheye_360` `.camrig` with the calibrated X5 geometry (identity left, 180° about Y right, 30.7 mm baseline at `+Z`).
**Implementing code:** `src/sfmtool/_commands/insv2rig.py` (`insv2rig`, X5 constants), `src/sfmtool/rig/insv2rig.py` (`extract_insv_frames`, `write_insv_camrig`, `_INSV_FRAME_PATTERN`).
**Inconsistencies:**
  - None found. Single `--output` option matches. Baseline `_X5_BASELINE_M = 0.0307`, right translation `[0,0,+0.0307]`, 180°-about-Y rotation all match spec §"Rig Geometry" (`insv2rig.py:28-30`, `:96-100`). Rig type `fisheye_360`, shared `OPENCV_FISHEYE` camera, sensor patterns `fisheye_left/frame_%06d.jpg` / `fisheye_right/frame_%06d.jpg` match spec §"Output" (`rig/insv2rig.py:268-278`, `:17`, `:22`). `.camrig` named `<INPUT_STEM>.camrig` in the rig root matches (`insv2rig.py:102`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing. (Rig name is `insv2_x5`; the spec doesn't pin a name, so no conflict.)

### specs/cli/match-command.md
**Summary:** Documents `sfm match` with four mutually-exclusive matching methods (exhaustive/sequential/flow/cluster) plus `--merge`, per-method tuning options, `--camera-model` override, camera-config precedence, cluster background-floor matching semantics, and the merge algorithm.
**Implementing code:** `src/sfmtool/_commands/match.py` (`match`, `_reject_stray_mode_options`), `src/sfmtool/feature_match/_run.py` (`_run_matching`, `_run_merge`), `src/sfmtool/camera/cameras.py` (`CAMERA_MODEL_NAMES`).
**Inconsistencies:**
  - None found. All method flags and per-method options match with correct defaults: `--sequential-overlap` 10 (`match.py:90-96`), `--flow-preset` default (`:104-110`), `--flow-skip` 5 (`:111-117`), `--cluster-alpha` 0.8 (`:125-131`), `--cluster-d` 10 (`:132-139`), `--cluster-preset` accurate (`:140-146`). Spec's "11 COLMAP model names" is exactly right — `_CAMERA_PARAM_NAMES` has 11 entries. Cluster/TVG output under `tvg-matches/` matches spec §"Cluster Matching" (`_run.py:200-205`). Stray-mode-option rejection matches spec (`_reject_stray_mode_options` at `match.py:33-49`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/merge-command.md
**Summary:** Documents `sfm merge`, which merges ≥2 pre-aligned `.sfmr` files via camera dedup, image merge, union-find point correspondences, percentile filtering, point/track averaging, and parallel PnP+RANSAC pose refinement.
**Implementing code:** `src/sfmtool/_commands/merge.py` (`merge`), `src/sfmtool/merge/reconstructions.py` (`merge_reconstructions`), `src/sfmtool/merge/correspondences.py` (union-find + percentile), `src/sfmtool/merge/pose_refinement.py` (`refine_camera_poses`).
**Inconsistencies:**
  - None found. `--merge-percentile` default 95.0 (`merge.py:29-34`); ≥2-reconstruction requirement enforced (`:56-57`). Every documented process step is present: camera dedup via union-find (`reconstructions.py:141-158`), union-find correspondence grouping + percentile filtering (`correspondences.py:45,141-156`), parallel PnP+RANSAC pose refinement via `ProcessPoolExecutor` (`pose_refinement.py:6-7,54,234`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/motion-command.md
**Summary:** A large spec covering `sfm motion` in two modes (image-sequence optical-flow discontinuity analysis with adaptive stride, and reconstruction analysis using pose-extrapolation + step-ratio + covisibility-drop + obs-outlier signals), plus a versioned `--json` schema.
**Implementing code:** `src/sfmtool/_commands/motion.py` (`motion`), `src/sfmtool/motion/image_sequence.py` (`analyze_image_sequence`), `src/sfmtool/motion/recon_discontinuity.py` (`analyze_reconstruction`), `src/sfmtool/motion/constants.py`, `src/sfmtool/motion/report.py`, `src/sfmtool/motion/_recon_console.py`.
**Inconsistencies:**
  - None found. CLI defaults match (`--initial-stride` 1, `--min-stride` 1, `--max-stride` 32 with min=2, `--no-adaptive` false; `motion.py:27-53`). Every threshold constant matches the spec exactly (`constants.py:21-33`): `STEP_RATIO_THRESHOLD=1.5`, `OVERLAP_DROP_THRESHOLD=1.8`, `OBS_Z_THRESHOLD=2.5`, `STEP_RATIO_WINDOW=8`, `OVERLAP_WINDOW=16`, `OVERLAP_BASELINE_WINDOW=24`, `OBS_WINDOW=24`, `POSE_TRANS_FACTOR=3.0`, `POSE_ROT_DEG=15.0`. JSON `thresholds` object populated from those constants (`report.py:235-239`). Error behavior matches spec §"Error behavior".
**Recommendation:** in sync — the "Open Questions" are explicitly forward-looking, not divergences.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/pano2rig-command.md
**Summary:** Documents `sfm pano2rig`, which renders equirectangular panoramas into a 6-face 90°-FOV cubemap (front/right/back/left/top/bottom) with per-face subdirectories sharing a frame index, and writes a six-sensor `cubemap` `.camrig` with a single shared PINHOLE camera (zero translations, front=identity).
**Implementing code:** `src/sfmtool/_commands/pano2rig.py` (`pano2rig`), `src/sfmtool/rig/pano2rig.py` (`convert_panoramas`, `write_pano_camrig`, `_cubemap_rotations`, `extract_perspective_face`).
**Inconsistencies:**
  - None found. `--face-size` default None→`pano_width // 4` matches spec (`pano2rig.py:34-36`, `rig/pano2rig.py:122-127,176-177`); `--jpeg-quality` default 95 matches (`pano2rig.py:37-41`). Rig type `cubemap`, single PINHOLE camera with `fx=fy=cx=cy=face_size/2`, zero translations, patterns `<face>/frame_%06d.jpg` all match spec §"Output Structure" (`rig/pano2rig.py:234-264`). Face rotations match the spec's axis description (`_cubemap_rotations` at `:47-60`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing. (Spec's own NOTE flags the command as under-validated across datasets — a caveat, not a code divergence.)

### specs/cli/panorama-command.md
**Summary:** Documents `sfm panorama`, which renders an equirectangular panorama from a posed reconstruction via spherical-tile consensus rendering, with many options (width, tile count, rig source, batch/dtype/k memory controls, photometric-RANSAC params) and range/near-image source subsetting.
**Implementing code:** `src/sfmtool/_commands/panorama.py` (`panorama`), `src/sfmtool/rig/panorama.py` (`render_equirect_panorama`, `resolve_panorama_rig`, `select_source_indices`).
**Inconsistencies:**
  - None found. Every option and default matches: `--equirect-width` 2160 (`panorama.py:65-71`), `--n-tiles` 320 (`:72-79`), `--batch-size` 32 (`:89-95`), `--dtype` float32 (`:96-103`), `-k` 1 (`:104-110`), `--seed` 1234 (`:111-117`), `--inlier-threshold` 8.0 (`:118-124`), `--gamma` 1.0 (`:125-131`), `--ransac-seed` 0 (`:132-138`). Validation matches spec: even-positive width (`:205-206`), near-image requiring near-count/near-radius (`:207-216`), `--camrig` must be `.camrig` with the "n-tiles ignored" note printed only when n-tiles came from the command line (`:197-204`). Range-then-near ordering via `select_source_indices` + `subset_by_image_indices` matches spec §"Source subsetting" (`:223-235`). NaN→black flattening matches spec pipeline step 5 (`:268-270`).
**Recommendation:** in sync — the spec even cites the exact implementing symbols, and they match.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/render-patches-command.md
**Summary:** Documents `sfm render-patches`, which projects each 3D point's oriented patch quad onto the source images and composites them, with fill modes (texture/normal/flat/wire), border/alpha/scale/upscale/backface-cull controls, and an image-substring filter. Requires an `embedded_patches` reconstruction.
**Implementing code:** `src/sfmtool/_commands/render_patches.py` (`render_patches_command`), `src/sfmtool/visualization/_patch_renderer.py` (`render_patches`, `_render_image`, `MODES`), `src/sfmtool/_feature_source.py` (`require_embedded_patches`).
**Inconsistencies:**
  - None found. Every option name/default matches (`--mode` default `texture`, `--border` default off at render_patches.py:47, `--border-color` `0,255,0`, `--border-thickness` 1, `--alpha` 1.0, bare `--opaque` ⇒ `0.1` via `flag_value="0.1"` at :63-74, `--scale`/`--upscale` 1.0, `--backface-cull` default on at :90, repeatable `--images`). Output naming `<stem>_<mode>.png` with `/`→`__` matches `_patch_renderer.py:344-345`. The embedded_patches gate (render_patches.py:159) and texture-needs-bitmaps error (`_patch_renderer.py:304-308`) match the spec precondition.
**Recommendation:** in sync — spec and code agree on all flags, defaults, and output naming.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/scale-by-measurements-command.md
**Summary:** Documents the `--scale-by-measurements` option on `sfm xform`, scaling a reconstruction to physical units from a YAML file of known point-pair distances, including unit parsing, median scale computation, a histogram, and cross-reconstruction Point ID resolution.
**Implementing code:** `src/sfmtool/xform/_scale_by_measurements.py` (`ScaleByMeasurementsTransform`, `_parse_distance`, `_resolve_point_cross_recon`, `_print_histogram`), wired via `src/sfmtool/xform/_arg_parser.py` and `_commands/xform.py:139-143`.
**Inconsistencies:**
  - **Workspace hash-search fallback not implemented.** The spec (Point ID Resolution step 1; "Failure modes → Source `.sfmr` not found") says the source `.sfmr` is either the `sfmr` field *or* "found by searching for a file whose `content_xxh128` matches the Point ID hash prefix (using the workspace search strategy)." The code only supports the explicit `sfmr` field: `_load_source` (`_scale_by_measurements.py:356-377`) raises immediately when `sfmr_path is None`, telling the user to add the `sfmr` field. No workspace search occurs.
  - **"Most common point index" on disagreement not implemented.** Spec (Cross-reconstruction match): "If they disagree ... warn and use the most common point index." The code warns (`_scale_by_measurements.py:126-130`) but then does `pt_idx = next(iter(resolved_points))` (:132) — the first dict-insertion key, not the most common. `resolved_points` is keyed by `input_pt_idx` (:114), so per-point occurrence counts are structurally discarded and "most common" cannot be computed. The warning text literally says "Using most common." while the code does not.
  - Minor/cosmetic: the spec's histogram example shows counts aligned in a right-hand column, but `_print_histogram` appends the count directly after the bar (`:174-175`), so counts are not column-aligned. Also "5 by default" (Diagnostics) implies a configurable bin count, but `num_bins = 5` is hardcoded (:148) with no option.
**Recommendation:** discuss / update code — the two resolution-fallback behaviors are spec'd functionality the code does not deliver; decide whether to implement the workspace hash search + most-common tally or downgrade those spec sections to forward-looking.
**Unclear / incorrect / suspicious:** The "Using most common." diagnostic string at `_scale_by_measurements.py:129` is actively misleading given the code picks an arbitrary first match.

### specs/cli/sift-command.md
**Summary:** Documents `sfm sift` with two mutually-exclusive action modes (`--extract`, `--draw <DIR>`) plus `--filter-sfm`, `--range`, `--num-threads`, `--tool`, and `--dsp` options; defaults come from the workspace unless overridden.
**Implementing code:** `src/sfmtool/_commands/sift.py` (`sift`), `src/sfmtool/sift/file.py`, `sift/extract_{colmap,opencv,sfmtool}.py`.
**Inconsistencies:**
  - None found. Exactly-one-mode enforcement (sift.py:90-92), `--filter-sfm` only with `--draw` (:95-96), `--range/-r` (:49-54), `--num-threads/-t` default -1 (:55-61), `--tool` choice with workspace default (:62-67), and `--dsp/--no-dsp` requiring `--tool` and COLMAP (:124-138) all match the spec table.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/solve-command.md
**Summary:** Documents `sfm solve` (incremental/global SfM), its option surface, input modes (images / `.matches` / seq-overlap), multi-model output naming, rig/`.camrig`/`camera_config.json` handling, and the COLMAP→canonical convention boundary.
**Implementing code:** `src/sfmtool/_commands/solve.py` (`solve`, `_run_sfm`, `_run_sequential_overlap_sfm`), `_incremental_sfm.py` (`run_incremental_sfm`, `_save_reconstructions`), `_global_sfm.py` (`run_global_sfm`).
**Inconsistencies:**
  - None found. All option names/defaults match: `--seed/-s`, `--output/-o`, `--colmap-dir`, `--sfmr-dir` (:56-60), `--seq-overlap`, `--refine-rig` default true (:82-87), `--flow-match`, `--flow-preset` default `default` (:94-100), `--flow-skip` default 5 (:101-107), `--max-features`, `--camera-model` (11-name `CAMERA_MODEL_NAMES`), `--range/-r`, `--detect-infinity` default true (:115-121). Mutual-exclusion guards (`--seq-overlap` vs `--output` at :173-177, vs `.matches` at :195-196) match. The multi-model behavior is implemented at `_incremental_sfm.py:152-211` and shared by the global path via `_save_reconstructions`.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/to-colmap-bin-command.md
**Summary:** Documents `sfm to-colmap-bin <INPUT.sfmr> <OUTPUT_DIR>` exporting to COLMAP `.bin` files, with `--range` image subsetting and `--filter-points`, range semantics, and the canonical→COLMAP convention boundary.
**Implementing code:** `src/sfmtool/_commands/to_colmap_bin.py` (`to_colmap_bin`, `_apply_range_filter`), `colmap/io.py` (`save_colmap_binary`), Rust `subset_by_image_indices`.
**Inconsistencies:**
  - None found. `--range/-r` and `--filter-points` (with the "`--filter-points` requires `--range`" guard at to_colmap_bin.py:75-76) match. `_apply_range_filter` (:89-119) resolves file numbers via `number_from_filename`, errors listing available numbers when no images match (:100-109), and calls `subset_by_image_indices(..., drop_orphaned_points=filter_points)` exactly as the Implementation section describes.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing (the spec's claim that `rigs.bin`/`frames.bin` are "always written" lives in `save_colmap_binary`, not the shim; not independently re-verified here but nothing contradicts it).

### specs/cli/to-colmap-db-command.md
**Summary:** Documents `sfm to-colmap-db <INPUT_PATH> <DATABASE.db>` building a COLMAP database from a `.sfmr` or `.matches` file, with `--max-features`, `--no-guided-matching`, and `--camera-model` options, and a documented "known gap" that it has no `--range`.
**Implementing code:** `src/sfmtool/_commands/to_colmap_db.py` (`to_colmap_db`, `_from_sfmr`, `_from_matches`), `colmap/db_export.py`, `colmap/db_setup.py` (`_setup_for_sfm_from_matches`).
**Inconsistencies:**
  - None found. `--max-features` (sfmr-only), `--no-guided-matching` mapping to `populate_two_view_geometries=not no_guided_matching` (to_colmap_db.py:97), and `--camera-model` (matches-only, `CAMERA_MODEL_NAMES`) all match. The documented "known gap" (no `--range`) is accurate. The `.matches`-mode `camera_config.json` rejection is implemented downstream via `_check_camera_model_conflict` in `db_setup.py:306-308`.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/to-nerfstudio-command.md
**Summary:** Documents `sfm to-nerfstudio <INPUT.sfmr> [<OUTPUT_DIR>]` repackaging a pinhole reconstruction into a Nerfstudio dataset (transforms.json, sparse_pc.ply, image pyramids), with identity `applied_transform`, OPENCV camera model, and `--num-downscales/--jpeg-quality/--include-colmap/--range/--filter-points`.
**Implementing code:** `src/sfmtool/_commands/to_nerfstudio.py` (`to_nerfstudio`, `_apply_range_filter`), `src/sfmtool/_to_nerfstudio.py` (`export_to_nerfstudio`, `build_transforms_json`, `write_sparse_ply`).
**Inconsistencies:**
  - None found. Optional positional `OUTPUT_DIR` defaulting to `{stem}_nerfstudio` (to_nerfstudio.py:99-102), `--num-downscales` default 3 (:26-32), `--jpeg-quality` default 95 (:33-39), `--include-colmap` (:40-45), `--range`/`--filter-points` with the require-`--range` guard (:96-97) all match. Identity `applied_transform` (`_to_nerfstudio.py:25-26,157`), `camera_model="OPENCV"` (:114), and `cv2.INTER_AREA` downsampling (:182) match the How-It-Works section.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/undistort-command.md
**Summary:** Documents `sfm undistort <RECONSTRUCTION.sfmr>` producing a new pinhole workspace: warps images, transforms `.sift` positions/affine shapes, remaps tracks, and writes a new `.sfmr`. Options `--fit`, `--filter`, `-o/--output`.
**Implementing code:** `src/sfmtool/_commands/undistort.py` (`undistort`), `src/sfmtool/_undistort_images.py` (`undistort_reconstruction_images`, Jacobian via central differences).
**Inconsistencies:**
  - None found. `--fit inside|outside` default inside (undistort.py:18-27), `--filter aniso|bilinear` default aniso (:28-38), `-o/--output` default `<stem>_undistorted/` (:39-46,89-92). Deep behavior matches too: central-difference Jacobian `eps=0.5` (`_undistort_images.py:55,97`), 128×128 `INTER_AREA` thumbnail (:350), `sfmtool-undistort` sentinel feature tool (:260), source-fallback to `"colmap"` (:246-249), `source_sfmr` fallback to `"unknown"` (:543), and the `Feature summary` / `Highest drop rate` stdout lines (:468-473).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/ws-init-command.md
**Summary:** Documents `sfm ws init [WORKSPACE_DIR]` creating `.sfm-workspace.json`, with `--feature-tool`, `--dsp`, `--max-features`, `--gpu`, `--affine-shape`, `--force`, plus validation rules (COLMAP-only knobs, gpu/affine mutual exclusion) and persisted-settings notes.
**Implementing code:** `src/sfmtool/_commands/ws.py` (`init`), `src/sfmtool/_workspace.py` (`init_workspace`).
**Inconsistencies:**
  - None found. Defaults match: `--feature-tool` sfmtool (ws.py:23-27), `--dsp` false (:28-33), `--max-features` default surfaced as 8192 via backend default (ws.py:34-40 → `init_workspace`/`get_default_sfmtool_feature_options`, `_workspace.py:27-31`), `--gpu` true (:41-46), `--affine-shape` false (:47-53), `--force` (:54-59). Validation matches spec exactly: DSP/gpu/affine COLMAP-only (:86-102), `--max-features` allowed for colmap+sfmtool but rejected for opencv (:90-94), and gpu+affine-shape mutual exclusion (:103-107). The `use_gpu`-excluded-from-cache-hash note is corroborated by `init_workspace`'s docstring (`_workspace.py:30-33`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/xform-command.md
**Summary:** The umbrella spec for the `sfm xform` command family — describes every operation (geometric transforms, filters, points-at-infinity, camera-model conversion, optimization ops including the three patch ops, `--to-embedded-patches`, scaling/alignment), the sequential ordering semantics, auto output naming, and the `Transform` protocol plus the Rust editing primitives.
**Implementing code:** `src/sfmtool/_commands/xform.py` (Click options + help), `src/sfmtool/xform/_arg_parser.py` (`parse_transform_args`, ordered `sys.argv` walk), `src/sfmtool/xform/_apply.py` (`apply_transforms`, per-step `required_feature_source` gate), `src/sfmtool/xform/__init__.py` (transform registry).
**Inconsistencies:**
  - None found. Every operation named in the spec has a Click option and an `_arg_parser.py` branch, and every Click option/`_arg_parser` branch is documented in the spec. Auto-output naming (`{stem}-transformed[-N].sfmr`) matches `auto_output_path` (`_arg_parser.py:370-385`). `--max-features` rejection when no `--find-points-at-infinity` is present matches `xform.py:340-348`. The `--to-embedded-patches` key list in the spec (line 404) matches `_TO_EMBEDDED_PATCHES_KEYS` (`_arg_parser.py:315-322`).
**Recommendation:** in sync — the overview accurately enumerates the implemented surface.
**Unclear / incorrect / suspicious:** The long "TODO (implementation cleanup)" block (lines 281-291) about collapsing the `.sift`-vs-inline-keypoint special-casing onto a single accessor is a forward-looking note, not a divergence.

### specs/cli/xform-find-points-at-infinity.md
**Summary:** Design + algorithm for `--find-points-at-infinity` (and companions `--classify-points-at-infinity`, `--max-features`): un-project untracked keypoints to world directions, KD-tree cluster within `eps_deg`, confirm clusters with mutual SIFT descriptor matching, triangulate/classify each track as `w=0` or finite-distant, and append. Status: Implemented.
**Implementing code:** `src/sfmtool/xform/_find_points_at_infinity.py` (`FindPointsAtInfinityTransform`, `ClassifyPointsAtInfinityTransform`), PyO3 `SfmrReconstruction.find_points_at_infinity` (`crates/sfmtool-py/src/py_sfmr_reconstruction.rs:924`), core `crates/sfmtool-core/src/analysis/infinity/discover.rs:363`.
**Inconsistencies:**
  - The `ratio` (Lowe ratio) parameter exists in code but is undocumented as a parameter/default in the spec. The transform takes `ratio: float = 0.8` (`_find_points_at_infinity.py:32`) and passes it to the binding (which defaults `ratio=0.8`, `py_sfmr_reconstruction.rs:924`), but the CLI surface `eps_deg[,desc_thresh[,min_views[,noise_floor_px]]]` has no slot for it and `_arg_parser.py:734-757` never sets it, so it is hard-wired to 0.8. Minor — a fixed internal constant, not user-facing.
**Recommendation:** update spec — add a one-line note that the Lowe `ratio` is fixed at 0.8 internally and not CLI-exposed, so the parameter table is complete.
**Unclear / incorrect / suspicious:** The document still contains large "Prototype findings" / "Motivation" sections written as a proposal even though Status is "Implemented"; the `tmp/infinity_search_prototype.py` reference (line 201) is gitignored and unverifiable.

### specs/cli/xform-localize-keypoints-command.md
**Summary:** `--localize-keypoints` — a structural op that congeals each observation's keypoint by discrete cross-view search, drops non-co-registering views, culls points below `min_views`, and rebuilds the track structure via `compact_to_embedded_patches`; drops stale bitmaps. Status: Implemented (2026-07-05).
**Implementing code:** `src/sfmtool/xform/_localize_keypoints.py` (`LocalizeKeypointsTransform`), `parse_localize_keypoints_params` (`_arg_parser.py:264`), PyO3 `PatchCloud.localize_keypoints` (`py_patch_cloud.rs:964-992`), write-back `compact_to_embedded_patches` in `src/sfmtool/_embed_patches.py`.
**Inconsistencies:**
  - None found. The parameter table (spec lines 98-113) matches both the constructor defaults (`_localize_keypoints.py:57-75`) and the binding signature defaults exactly: `min_views=2`, `max_iters=5`, `search=6.0`, `max_shift_px=3.0`, `min_relative_zncc=0.7`, `min_grazing_cos=0.1`, `resolution=24`, `window=gaussian_disk`, `window_sigma=0.6`, `sampler=bilinear`, `robust_iters=3`, `convergence_px=0.05`, `search_resolution_multiplier=1.0`, `search_strategy=plus_descent` (`py_patch_cloud.rs:964-970`). Precondition gate via `required_feature_source = "embedded_patches"` matches.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/cli/xform-refine-keypoints-command.md
**Summary:** `--refine-keypoints` — pure in-place sub-pixel keypoint refiner (forward-additive ECC Gauss–Newton against robust consensus), changes no view membership or track structure, only rewrites `keypoints_xy`; optional `bitmaps=true` re-renders patch textures. Status: Implemented (2026-07-05, #174).
**Implementing code:** `src/sfmtool/xform/_refine_keypoints.py` (`RefineKeypointsTransform`), `parse_refine_keypoints_params` (`_arg_parser.py:192`), PyO3 `PatchCloud.refine_keypoints` (`py_patch_cloud.rs:1240-1268`).
**Inconsistencies:**
  - None found. Parameter table (spec lines 73-86) matches constructor (`_refine_keypoints.py:54-71`) and binding defaults (`py_patch_cloud.rs:1240-1245`) exactly: `resolution=24`, `window=gaussian_disk`, `window_sigma=0.6`, `sampler=bilinear`, `robust_iters=3`, `max_outer_sweeps=1`, `outer_convergence_px=0.005`, `max_gn_steps=10`, `convergence_px=0.01`, `max_offset_px=2.0`, `consensus_refresh=per_sweep`, `bitmaps=false`. The vectorized scatter write-back, near-edge f32 clamp, and summary format all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** The "Future" section (lines 146-153) proposing `--localize-keypoints` already carries a "Status (2026-07-05): Done" note pointing at the localize spec — correctly marked, not a divergence.

### specs/cli/xform-refine-normals-command.md
**Summary:** `--refine-normals` — photometric per-point normal refinement that rewrites `normals` in place (finite points only), always re-persists the patch frame, optional `bitmaps=true`. Requires `embedded_patches`. Status: Implemented (2026-06-13), precondition shipped 2026-06-25.
**Implementing code:** `src/sfmtool/xform/_refine_normals.py` (`RefineNormalsTransform`), `parse_refine_normals_params` + `_REFINE_NORMALS_KEYS` (`_arg_parser.py:102-168`), PyO3 `PatchCloud.refine_normals` (`py_patch_cloud.rs:469-477`).
**Inconsistencies:**
  - None found. Parameter table (spec lines 108-126) matches constructor defaults (`_refine_normals.py:77-98`) and binding signature (`py_patch_cloud.rs:469-476`) exactly, including quality-preset overrides, finite-only masking, `use_stored_keypoints=True`, low-confidence threshold 0.1, and summary format.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** The #177 fix ("build oriented-patch frame upright, not rotated 90°") touches `OrientedPatch::from_center_normal`/`from_infinity_direction`, not this spec's surface. The spec's only frame claim — `normalize(u × v)` is the per-point normal (line 251) — remains consistent with the binding. No divergence introduced by #177 in this spec.

### specs/cli/xform-select-by-distribution-command.md
**Summary:** `--include-by-distribution <COUNT>[,verbose]` — greedy farthest-point + angular-thinning selection of a well-distributed camera/rig-frame subset off the reconstructed cloud; `H=20°` hard-coded; deterministic; delegates the actual image bookkeeping to `_filter_images`.
**Implementing code:** `src/sfmtool/xform/_select_by_distribution.py` (`SelectByDistributionFilter`, `_select_images`), parsed in `_arg_parser.py:678-705`, delegates to `_filter_images` (`_filter_by_image_range.py`); `_H_RAD = radians(20)` (`_select_by_distribution.py:38`).
**Inconsistencies:**
  - Code excludes points at infinity from the cloud before selection; the spec is silent on this. `_select_images` drops all infinity-point observations (`_select_by_distribution.py:126-136`), raises `ValueError` if only infinity points remain, and computes the bbox diagonal over finite points only (lines 191-194). The spec's "The point cloud is taken as given" section (lines 52-65) never mentions the infinity exclusion. Real behavior-not-in-spec; benign but should be documented.
  - Seed/target/fill tie-breaks, `⌈COUNT/3⌉` seed cap, verbose columns/header, and the k-d-tree farthest-point loop all match the spec.
**Recommendation:** update spec — add one sentence noting points at infinity are dropped up front (finite cloud only), and that an all-infinity reconstruction is an error.
**Unclear / incorrect / suspicious:** The spec's Parameters section says `H` is "Not exposed on the CLI initially; hard-coded" — consistent with code (no CLI knob).

## Core algorithm specs

### specs/core/batch-triangulation-api.md
**Summary:** Consolidates triangulation into a batch API in `reconstruction/triangulation.rs` returning each track's midpoint point plus observability diagnostics (eigenvalues, condition number, optional depth-uncertainty z-score), and re-bases the points-at-infinity classifiers on `inverse_depth_z` with a `resolvable_distance ≥ finite_horizon` gate yielding a three-state (finite / infinity / indeterminate) decision.
**Implementing code:** `crates/sfmtool-core/src/reconstruction/triangulation.rs` (`Triangulation`, `DepthUncertainty`, `triangulate_batch`, `depth_uncertainty_batch`); `analysis/infinity/convert.rs` (`classify_rays_at_infinity`, `classify_points_at_infinity`, constants `DEFAULT_INVERSE_DEPTH_Z_CUTOFF=4.0`, `CONDITION_NUMBER_PREFILTER=1e4`); `analysis/infinity/discover.rs` (`classify_track`, `find_points_at_infinity`); py bindings `crates/sfmtool-py/src/analysis/triangulation.rs::triangulate_batch`, `py_sfmr_reconstruction.rs::triangulation_diagnostics`; CLI parsing `src/sfmtool/xform/_arg_parser.py:730-751`.
**Inconsistencies:**
  - Spec Open questions (line 363) states "discovered points currently carry `error = 0`." The code no longer does this: `discover.rs:470-497,537` computes a real mean reprojection error per discovered point via `observation_reprojection_error` and stores it. Stale spec note.
  - Minor, additive: code documents/handles the empty-track case (`in_front_of_all_cameras=false` for empty tracks, `triangulation.rs:44-46,149`) which the spec does not mention.
**Recommendation:** update spec — drop/adjust the "discovered points carry error=0" line to reflect the inline reprojection-error computation; otherwise in sync.
**Unclear / incorrect / suspicious:** `depth_uncertainty_batch` computes `b_perp = 2.0 * RMS(perp offset)` (`triangulation.rs:251`) — the factor 2 (to match a 2-view baseline) is a real modeling choice absent from the spec's `B⊥ / σ` formula (spec lines 191, 259). The code docstring explains it, but the spec should note it since it shifts `resolvable_distance` by 2× and thus the `finite_horizon` gate calibration.

### specs/core/epipolar-curves.md
**Summary:** Model-agnostic epipolar curves for non-perspective cameras: back-project `p1`, bracket the in-image depth interval in log-depth, then adaptively subdivide in `t=1/λ` worst-first to a pixel curvature tolerance, reprojecting through the full destination camera model. Exposed via PyO3 `epipolar_curves` and consumed by `sfm epipolar`.
**Implementing code:** `crates/sfmtool-core/src/camera/epipolar.rs` (`plot_epipolar_curve`, `plot_epipolar_curves_batch`, `EpipolarCurveOptions`, `find_inimage_seed`, `bisect_boundary`, `subdivide_worst_first`, constants `LOG_STEP=LN_2`, `BRACKET_MAX_STEPS=24`, `BRACKET_LOG_TOL=1e-3`); `crates/sfmtool-py/src/analysis/epipolar.rs::epipolar_curves_py`; `src/sfmtool/visualization/_epipolar_display.py`.
**Inconsistencies:**
  - None found. Constants, the cheirality predicate, per-feature `anchor_depths` signature, ±octave alternating seed search, `t=1/λ` worst-first subdivision, and the no-clip Python display all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/flow-based-matching.md
**Summary:** Flow-based feature matching for video: sliding-window advection of SIFT keypoints through dense DIS flow, spatially matched to target keypoints and validated by descriptor L2 distance (production threshold 250), producing multi-baseline matches in an O(N) sweep.
**Implementing code:** `src/sfmtool/feature_match/_flow_matching.py` (`flow_match_sequential`, `_flow_match_from_advected`, `_flow_match_pair`); Rust `crates/sfmtool-py/src/matching/descriptor.rs::match_candidates_by_descriptor` and `spatial::KdTree2d::nearest_k_within_radius`.
**Inconsistencies:**
  - **`spatial_tolerance` parameter is dead.** `flow_match_sequential` defaults `spatial_tolerance=3.0` (`_flow_matching.py:180`) and threads it into `_flow_match_from_advected` / `_flow_match_pair`, but neither function uses it — the actual spatial radius is the module constant `_SPATIAL_CANDIDATES_RADIUS = 10.0` (`_flow_matching.py:41`, used at lines 102-103 and 146-147). The spec repeatedly states the spatial tolerance is "3px default" (spec lines 91, 104, 126). Effective radius is 10px and not tunable.
  - **Matching strategy differs from spec.** Spec per-pair step 3 says "Find nearest keypoint in target within spatial tolerance" (single nearest) then descriptor-filter. Code instead retrieves `_SPATIAL_CANDIDATES_K = 5` nearest within the radius (`_flow_matching.py:40,99-103`) and lets `match_candidates_by_descriptor` pick the best-descriptor candidate among the 5.
**Recommendation:** update spec (and consider updating code) — document the 10px candidate radius + K=5 best-descriptor selection, or remove/wire the misleading dead `spatial_tolerance` argument. Descriptor threshold 250, `window_size=5`, and dedup all match.
**Unclear / incorrect / suspicious:** The public `spatial_tolerance` argument being silently ignored is a latent foot-gun: a caller tightening it to 3px gets no effect. Worth either wiring it into `nearest_k_within_radius` or deleting it.

### specs/core/fronto-parallel-patch-cache.md
**Summary:** A fronto-parallel patch cache that renders one supersampled base per view up front and affine-resamples every candidate normal from it (undistorted-normalized corners so distortion cancels), replacing per-candidate source re-rendering; scalar Phase 1 defines the numbers, Phase 2 adds a runtime-dispatched AVX2 kernel. Now the default.
**Implementing code:** `crates/sfmtool-core/src/patch/normal_refine/fronto_cache.rs` (`resample_support_avx2`, `resample_support_scalar` at lines 273-387); `params.rs` (`CacheMode {Off, FrontoParallel}` lines 76-89, `Default` sets `cache: FrontoParallel, cache_supersample: 2.0` lines 204-205).
**Inconsistencies:**
  - Internal spec tension (not a code divergence): the Status header correctly states the cache is now the default, matching `params.rs:204-205`. But the Phase-1 implementation-plan text still says "Default `Off` so existing behavior is unchanged until opted in" (spec lines 172, 189). Historical-narrative artifact.
**Recommendation:** update spec — add a one-line note in the Phase-1 block that the default was subsequently flipped to `FrontoParallel`, so the two sections don't read as contradictory.
**Unclear / incorrect / suspicious:** Nothing material.

### specs/core/gpu-optical-flow.md
**Summary:** wgpu compute-shader implementation of the 5-stage DIS optical-flow pipeline with hybrid per-level CPU/GPU routing (`gpu_min_pixels` threshold), minimized CPU↔GPU transfers, persistent buffer pools, and a WGSL Jacobi kernel transliterated from the CPU path.
**Implementing code:** `crates/sfmtool-core/src/features/optical_flow/gpu/` (`context.rs`, `dis_pipeline.rs`, `pyramid_pipeline.rs`, `variational.rs`, `shaders/`); `optical_flow/mod.rs:503,522,541` (`gpu_min_pixels: 50_000`), `dis.rs:69` and `mod.rs:654` (per-level routing).
**Inconsistencies:**
  - None found. All 9 shaders named in the spec exist exactly. `gpu_min_pixels` default 50,000 and the per-level threshold routing match spec lines 56-60.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/image-pair-graph.md
**Summary:** Builds image-pair graphs from a posed reconstruction two ways — covisibility (shared tracks) and frustum intersection (Monte Carlo volume overlap) — both filtered by a camera viewing-angle threshold, exposed to Python and consumed by `sfm analyze`/`densify`/export.
**Implementing code:** `crates/sfmtool-core/src/analysis/image_pair_graph.rs` (`compute_camera_directions`, `build_covisibility_pairs`, `estimate_z_from_histogram`, `build_frustum_intersection_pairs`); geometry in `camera/frustum.rs`.
**Inconsistencies:**
  - None found. Direction convention `−column(2)`, covisibility per-point dedup + descending-count sort, `near≥far → [min_z,max_z]` fallback, separating-plane pre-test, both-direction averaged Monte Carlo estimate, and per-row deterministic RNG `seed + i` all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/image-warping.md
**Summary:** Warp-map generation (`from_cameras` plus rotation- and pose-aware variants) and multi-channel `u8` resampling (bilinear + GPU-style anisotropic) for distortion/undistortion/model-conversion, adding `ray_to_pixel[_batch/_grid]` and an `Equirectangular` camera model; CPU/rayon, with GPU deferred.
**Implementing code:** `crates/sfmtool-core/src/camera/warp_map.rs` (`WarpMap`, `WarpMapSvd`, `from_cameras`, `from_cameras_with_rotation`, `from_cameras_with_pose`, `compute_svd`, `from_patch`); `camera/remap.rs` (`ImageU8`, `ImageU8Pyramid`, `remap_bilinear`, `remap_aniso`); `camera/distortion.rs` + `intrinsics.rs` (`ray_to_pixel`, `needs_ray_path`, `Equirectangular`); py `crates/sfmtool-py/src/flow/warp.rs`.
**Inconsistencies:**
  - None found at the API/structure level. The spec explicitly flags GPU resampling as deferred/open — forward-looking, not a divergence.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** The spec (line 555) says to add `ray_to_pixel` / `distort_ray` to `CameraModel`; `ray_to_pixel` confirmed but no `distort_ray` symbol located — if never added (folded into `ray_to_pixel`), the spec line is slightly aspirational. Low priority.

### specs/core/keypoint-localization-search-cache.md
**Summary:** Specifies a per-view render-once cache plus a hand-rolled AVX2 windowed-ZNCC kernel to accelerate the integer cross-view shift search inside keypoint localization (congealing). Covers the centered-f32 planar cache, register-blocked AVX2 accumulator, the `search_resolution_multiplier` knob, and the `SearchStrategy` (Exhaustive vs PlusDescent) axis.
**Implementing code:** `crates/sfmtool-core/src/patch/keypoint_localize.rs` — `SearchStrategy` (line 51), `KeypointLocalizeParams.search_resolution_multiplier`/`search_strategy` (lines 132–135), `ContextTile` planar centered-f32 cache (line 208), `compute_channel_grids_avx2` (line 1137), `score_cell_one_channel_avx2` (line 1351), `search_shift_plus_descent` (~line 1700); sub-phase timers in `keypoint_localize/prof.rs`.
**Inconsistencies:**
  - Stale status header. The doc header says `_Status: **proposed** (design)_` and the Phase-1 blockquote says "Still scalar — the centered-`f32`/planar AVX2 kernel and the `i16` path are later phases" (line 36). The AVX2 stage-1 kernel is in fact fully landed: `compute_channel_grids_avx2` (keypoint_localize.rs:1137) is exactly the 6-YMM register-blocked loop the spec's "Search kernel (stage 1)" section describes, and the PlusDescent per-cell gather kernel `score_cell_one_channel_avx2` is also present. The header contradicts both the code and the spec's own later PlusDescent/i16 sections.
  - Everything concrete matches: PlusDescent is the default (keypoint_localize.rs:73, 152), `search_resolution_multiplier` default 1.0, R=24/search=6 defaults, and the prof timers.
**Recommendation:** update spec — bump the header status to implemented and drop the "later phases / still scalar" language.
**Unclear / incorrect / suspicious:** The spec is internally inconsistent (header says scalar/proposed; body presents measured AVX2 tables as history). A reader can't tell current state from the header alone.

### specs/core/keypoint-subpixel-refinement.md
**Summary:** Standalone forward-additive ECC Gauss–Newton sub-pixel keypoint refiner with shared-T running consensus, three refresh granularities, analytic sampler Jacobian, and opt-in representative-bitmap fusion; exposed as `PatchCloud.refine_keypoints`.
**Implementing code:** `crates/sfmtool-core/src/patch/keypoint_subpixel.rs` — `KeypointSubpixelParams` (line 140, `max_outer_sweeps` default 1, `consensus_refresh` default `PerSweep`), `ConsensusRefresh` (line 97), `RunningConsensus` (line 868). Bindings: `crates/sfmtool-py/src/py_patch_cloud.rs::refine_keypoints` (line 1248). Wiring: `src/sfmtool/_embed_patches.py::_refine_subpixel` (line 322).
**Inconsistencies:**
  - The status prose (lines 22–25) claims `embed_patches` exposes LK behind a `subpixel: str = "none"` kwarg with `"lk"` / `"lk_per_move"` values, production default `"none"` (all variants opt-in). The actual pipeline knob is an **integer**: `_embed_patches.py:597` `subpixel: int = 1` and CLI `_commands/embed_patches.py:85-96` `--subpixel` `IntRange(min=0)` default **1** — i.e. LK is **on by default** (1 sweep), not opt-in, and there is no string enum.
  - The `"lk_per_move"` variant is **not reachable from embed_patches**: `_refine_subpixel` hardcodes `consensus_refresh="per_sweep"` (`_embed_patches.py:359`). Per-move is only reachable via the direct `PatchCloud.refine_keypoints(consensus_refresh="per_move")` binding.
  - All other claims verified correct: shared-T `RunningConsensus` (not LOO), `max_outer_sweeps` default 1, PerSweep default, `ValueError` on sift_files recon without `starting_keypoints` (py_patch_cloud.rs:1293).
**Recommendation:** update spec — rewrite the `embed_patches` wiring paragraph to describe the integer `subpixel` sweep-count knob (default 1, `0` disables), and correct the claim that `lk_per_move` is exposed through the pipeline (it is core/binding-only).
**Unclear / incorrect / suspicious:** The spec's "production default is `subpixel="none"`" implies LK is off in production; the code ships it **on** at 1 sweep. This is a behavior-level claim worth reconciling deliberately, not just a wording fix.

### specs/core/optical-flow.md
**Summary:** Pure-Rust DIS (Dense Inverse Search) optical flow with CPU (SSE2/rayon) and GPU (wgpu) paths, Jacobi variational refinement, and three presets, plus Python bindings.
**Implementing code:** `crates/sfmtool-core/src/features/optical_flow/mod.rs` — `DisFlowParams::{default_quality,fast,high_quality}` (lines 489/508/527).
**Inconsistencies:**
  - None found. Preset defaults match exactly: default_quality (patch 8, overlap 0.4, iters 12), fast (8, 0.3, 16), high_quality (12, 0.75, 16); jacobi_iterations 7; alpha 10 / gamma 10 / delta 5; `gpu_min_pixels` 50,000.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/patch-cloud.md
**Summary:** Defines `OrientedPatch` (world-space surfel), `PatchCloud` (SoA), the `PatchNormal`/`PatchExtent`/`ViewReduce` policies, `WarpMap::from_patch` projection, infinity-patch handling, and `.sfmr` serialization.
**Implementing code:** `crates/sfmtool-core/src/patch/cloud.rs` — `OrientedPatch::from_center_normal` (line 164), `from_infinity_direction` (line 194), `PatchCloud::from_reconstruction` (~line 470); `WarpMap::from_patch`.
**Inconsistencies:**
  - None found. The upright-frame convention (#177) is correctly reflected: `from_center_normal` computes `v = up_hint` projected onto the plane and `u = v.cross(&n)` (cloud.rs:171-179), matching the spec's "render is upright" text (patch-cloud.md:99-110). No leftover "rotated 90°" language remains.
**Recommendation:** in sync — spec matches code including the #177 upright-frame fix.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/patch-keypoint-localization.md
**Summary:** The congealing keypoint-localization algorithm: per-round render, robust LOO consensus, per-view windowed-ZNCC shift search, in-loop view drops, exposed as `PatchCloud.localize_keypoints`.
**Implementing code:** `crates/sfmtool-core/src/patch/keypoint_localize.rs` (`localize_patch_keypoints`, defaults line 138); binding `crates/sfmtool-py/src/py_patch_cloud.rs::localize_keypoints` (line 972).
**Inconsistencies:**
  - Algorithm step 3 (and the status section, lines 185-186) describe the per-view search as "a full-res integer search then a separable parabolic sub-pixel fit" — i.e. exhaustive. But the shipped default is `SearchStrategy::PlusDescent` (keypoint_localize.rs:152; binding default `search_strategy="plus_descent"`), a "+"-descent that is **not** a full-resolution exhaustive scan. The exhaustive path is now the non-default variant. This default-strategy change is documented in keypoint-localization-search-cache.md but not reflected here.
  - The binding signature quoted in the status block (lines 178-181) predates the `search_resolution_multiplier` and `search_strategy` kwargs (py_patch_cloud.rs:968-969); the defaults section makes no mention of either.
  - All parameter defaults match the table (max_iters 5, search 6, max_shift_px 3, min_relative_zncc 0.7, min_grazing_cos 0.1, resolution 24, robust_iters 3, convergence_px 0.05). The "binding does not expose starting_keypoints" claim still holds.
**Recommendation:** update spec — note in step 3 / the status block that the shipped default search strategy is PlusDescent (cross-linking the cache spec) and add the two new kwargs.
**Unclear / incorrect / suspicious:** A reader following only this spec would assume every per-view search is an exhaustive full-res scan; in production it is a hill-climb with a documented accuracy tail (p99 ~3 px). Worth surfacing here since this is the algorithm's home spec.

### specs/core/patch-normal-refine-view-subset.md
**Summary:** D-optimal geometric view-subset selection capping the round-2+ normal-refinement basis at K views (`select_refine_subset`), on by default at K=8 in the pipeline.
**Implementing code:** `crates/sfmtool-core/src/patch/normal_refine/view_subset.rs::select_refine_subset` (line 40); `NormalRefineParams::max_refine_views` (params.rs:176, default 0); `_embed_patches.py:602` `max_refine_views: int = 8`; CLI `--refine-max-views` default 8.
**Inconsistencies:**
  - None found. Least-oblique anchor, greedy `det(M + wᵢwᵢᵀ)` fill, back-facing skip, and the removed conditioning fallback (#170) all match. Low-level default 0, pipeline default 8, applied only to the round-2+ `refine_normals` call (`_embed_patches.py:846-856`).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/patch-normal-refinement.md
**Summary:** Photometric 2-DOF patch-normal refinement via exp-map coarse-to-fine grid over a robust IRLS consensus, with obliquity/fronto priors, Hessian confidence, bilinear default sampler, and opt-in representative-bitmap output.
**Implementing code:** `crates/sfmtool-core/src/patch/normal_refine/` — `NormalRefineParams` (params.rs, defaults line 196), `refine_patch_normal`/`refine_patch_cloud_normals` (mod.rs), `view_stack.rs`.
**Inconsistencies:**
  - None found. Defaults match; `Sampler::Bilinear` is the default; the "bitmaps now sourced from `refine_keypoints`, not normal refinement" claim matches `_embed_patches.py`.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing. (Items 1-7 under "Improvements" are explicitly forward-looking.)

### specs/core/patch-view-selection.md
**Summary:** Per-point view-selection: expand a track with geometrically-visible candidates that photometrically agree (windowed ZNCC vs the track's IRLS consensus), gated by a self-agreement trust floor; exposed as `PatchCloud.select_views`.
**Implementing code:** `crates/sfmtool-core/src/patch/view_selection.rs::select_patch_views` (defaults line 66: min_relative_zncc 0.7, min_valid_fraction 0.6, min_track_views 2, min_self_agreement 0.3).
**Inconsistencies:**
  - None found. Defaults match; trust-gate logic matches (below `min_self_agreement` admit verbatim, else bar = `min_relative_zncc × self_agreement`); two-view floor `min_track_views.max(2)` matches.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** `min_valid_fraction` 0.6 exists in code but isn't in the spec's parameter table — an unlisted shared validity knob, not a contradiction.

### specs/core/per-spherical-tile-source-stack.md
**Summary:** Specifies `PerSphericalTileSourceStack<T>`, a CSR-flat, per-tile multi-source image-pyramid store built by projecting source images into each spherical tile's pinhole frame (rotation-only warp, scene-at-infinity). Covers the `PatchPixel` storage trait (u8/f16/f32), the two-pass parallel build, visibility cull, all-four valid-mask downsample rule, and accessors.
**Implementing code:** `crates/sfmtool-core/src/spherical/per_tile_source_stack.rs` — `PatchPixel` (l.61-127), `PatchLevel<T>` (l.140), `BuildParams{max_in_flight_sources}` (l.178), `BuildError` (l.190-200), `build_rotation_only` (~l.268), `consensus_patches_per_tile`/`primary_consensus_atlas` (l.579-639). PyO3 in `crates/sfmtool-py/src/spherical/tile_source_stack.rs`.
**Inconsistencies:**
  - None found. Trait methods, error variants, `BuildParams::max_in_flight_sources` (reserved no-op), and CSR layout all match the spec's API block.
**Recommendation:** in sync — pose-aware `build_with_pose` is explicitly future work.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/photometric-subsets-ransac.md
**Summary:** Per-tile RANSAC that partitions each tile's source contributions into a primary and secondary agreeing cluster via validity-weighted patch-L1 scoring, plus per-tile counts and luminance MADs.
**Implementing code:** `crates/sfmtool-core/src/spherical/photometric_ransac.rs` — `RansacPhotometricParams` defaults (l.63-74), `refine_flat` (l.283), seed derivation (l.477-478, 544-547), tie-break (l.652-659), `per_pixel_median`/`consensus_patch`/`patch_l1_score` (l.666-738).
**Inconsistencies:**
  - None found. Defaults (`inlier_threshold=8.0`, `gamma=1.0`, `target_patch_size=4`, `scoring_patch_size=2`, `subset_size=2`, `max_subsets_per_tile=64`, `min_inliers=2`, `saturation_threshold=254`) match exactly. Seed derivation and tie-break match. Fully-masked-row `+inf` guard matches.
**Recommendation:** in sync — algorithm, knobs and numerics agree line-for-line.
**Unclear / incorrect / suspicious:** Minor: `consensus_patch` uses a `1e-3` denominator floor (l.703) whereas the spec pseudocode writes `max(denom, 1.0)` only for `patch_l1_score`; immaterial to results.

### specs/core/point-correspondence.md
**Summary:** Finding same-3D-point correspondences across reconstructions via shared feature-index observations (first-occurrence semantics), a coordinate-based fallback for different feature backends, multi-way grouping and union-find merging for `sfm merge`.
**Implementing code:** `crates/sfmtool-core/src/reconstruction/point_correspondence.rs`; `crates/sfmtool-py/src/analysis/core.rs`; `src/sfmtool/_point_correspondence.py` (`pixel_threshold=2.0`, `min_votes=2`, l.199-204); `src/sfmtool/merge/correspondences.py`.
**Inconsistencies:**
  - None found. Infinity-filtering, `ValueError` on zero correspondences, coordinate-fallback defaults (2.0 px / 2 votes), and merge percentile default (95.0) all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/randomized-kdtree-forest.md
**Summary:** Pure-Rust randomized kd-tree forest ANN (Muja & Lowe 2009): randomized top-D-variance splits, shared best-bin-first priority search with an `L_max` budget, u8/f32 metrics, SIMD leaf scan, precision calibration.
**Implementing code:** `crates/sfmtool-core/src/features/kdforest/` — `KdForestParams` presets (mod.rs l.85-114), `build.rs`, `search.rs`, `distance.rs`, `calibrate.rs`; `crates/sfmtool-py/src/py_kdforest.rs`.
**Inconsistencies:**
  - None found. Presets `fast`/`balanced`/`accurate` with `max_leaf_checks` 32/128/512 and `num_trees` 4/4/8 match; `split_dim_candidates` = 5, `leaf_size` = 16, `seed` = 0, per-tree seed `seed + tree_index` all match.
**Recommendation:** in sync — Phase 2 (f32 hardening, k-means tree) correctly marked future.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/ray-grid-projection.md
**Summary:** Splits `WarpMap::from_patch` into a model-free affine ray-grid geometry stage and a camera-owned `CameraIntrinsics::ray_to_pixel_grid` projection stage; perspective is exact per-node, fisheye/equirect uses a probe-bounded coarse grid.
**Implementing code:** `crates/sfmtool-core/src/camera/distortion.rs` — `COARSE_GRID_STRIDE=8` (l.82), `COARSE_GRID_TOL_PX=0.02` (l.89), `GridProj` (l.101); `crates/sfmtool-core/src/camera/warp_map.rs`.
**Inconsistencies:**
  - None found. Both named constants match; the two-path structure and `GridProj` hoist are present as described.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/reconstruction-alignment.md
**Summary:** Kabsch similarity fit (SVD, det correction, rank-deficient handling) + RANSAC outlier rejection, with Python wiring for data-derived thresholds in `sfm align --method points`.
**Implementing code:** `crates/sfmtool-core/src/analysis/alignment/{kabsch.rs,ransac.rs}`; binding defaults `crates/sfmtool-py/src/analysis/core.rs:139`; `src/sfmtool/align/core.py::kabsch_algorithm`.
**Inconsistencies:**
  - None found. Rank threshold `1e-10`, rank-0→`R=I`, rank-1 nullspace fixing, det<0 correction, zero-variance scale→1.0, RANSAC "keep-all on no consensus", and the Python `n≥2` vs Rust `n≥1` split all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/sift-to-patch-reconstruction.md
**Summary:** The `sfm embed-patches` pipeline converting a `sift_files` reconstruction to `embedded_patches`: build a patch frame, refine normals photometrically, select views, refine keypoints (congealing + subpixel, fusing consensus bitmaps), cull, compact.
**Implementing code:** `src/sfmtool/_embed_patches.py::embed_patches` (l.585-605); `src/sfmtool/_commands/embed_patches.py`; Rust kernels via `sfmtool-core::patch` + `SfmrReconstruction.to_embedded_patches`.
**Inconsistencies:**
  - The spec's parameter table (l.163-168) documents only `min_relative_zncc`, `patch_size`, `max_shift_px`, `min_views` (plus `max_iters`/`search`/`resolution` in the status footer). The implemented `embed_patches()` signature (l.585-604) additionally exposes **`rounds=2`**, **`max_obliquity_deg=80.0`**, **`obliquity_weight_power=2.0`**, **`fronto_prior_weight=0.05`**, **`max_refine_views=8`**, **`localize_search_strategy="plus_descent"`**, **`search_resolution_multiplier`**, and **`subpixel`** — none in the spec.
  - Structural divergence: the spec's Pipeline (steps 1–7) describes a **single** normal-refine → view-select → keypoint-refine pass, but the code runs **`rounds=2`** iterations alternating normal refinement and keypoint refinement (docstring l.653), and applies obliquity down-weighting / a fronto-parallel prior the spec's normal-refinement description does not mention.
**Recommendation:** update spec — the code has grown a multi-round loop and obliquity/fronto priors beyond the single-pass, 4-knob pipeline the spec (still labelled "draft for review") documents.
**Unclear / incorrect / suspicious:** The spec header says "draft for review" while the footer says "fully wired in `_embed_patches.py`" — the footer is accurate for the plumbing but the pipeline/params sections are behind the code.

### specs/core/sift.md
**Summary:** Pure-Rust SIFT detector+descriptor with COLMAP conventions; scale-space pyramid, DoG, extrema, subpixel localization, orientation, 128-D descriptor; tiled DoG/detect fusion; detect/describe split API; forward-looking on-disk incremental extraction.
**Implementing code:** `crates/sfmtool-core/src/features/sift/` (`SiftParams::default` l.126-137); `crates/sfmtool-py/src/sift/extract.rs`.
**Inconsistencies:**
  - None found. Defaults match exactly: `octave_layers=3`, `sigma=1.6`, `blur_radius_factor=2.25`, `input_sigma=0.5`, `contrast_threshold=0.0067`, `edge_threshold=10.0`, `max_num_features=Some(8192)`, `orientation_bins=36`, `peak_ratio=0.8`, descriptor clamp 0.2, `m_descr=3`.
**Recommendation:** in sync — on-disk incremental extraction, Tier-2 blur fusion, and GPU are explicitly future work.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/spherical-tiles-rig.md
**Summary:** `SphericalTileRig` — a rig of pinhole tiles over the sphere via Thomson-relaxed points; measured-coverage-driven FOV/patch sizing, atlas packing, atlas↔camera warps, `resample_atlas` k-nearest blend, `.camrig` persistence, `set_patch_size`, `tiles_subset`.
**Implementing code:** `crates/sfmtool-core/src/spherical/tile_rig.rs` (`MIN_PATCH_SIZE=5` l.40, `COVERAGE_PROBE_N=50_000` l.51, methods l.298-917); `spherical/sphere_points.rs` (`RelaxConfig` defaults 50/0.05/5.0); `spherical/tile_rig/camrig.rs`.
**Inconsistencies:**
  - None found. All named constants, `RelaxConfig` defaults, and the full method surface are present and match.
**Recommendation:** in sync — the "Open questions" are marked Resolved and match the camrig implementation.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/tile-batched-consensus-atlas.md
**Summary:** Bounded-memory panorama compositing that processes rig tiles in batches, building/dropping one `PerSphericalTileSourceStack` per batch, running photometric RANSAC, and blitting consensus patches into a persistent atlas — byte-identical to the monolithic path via `tile_index_base` RNG re-seeding.
**Implementing code:** `crates/sfmtool-core/src/spherical/consensus_atlas.rs` — `ConsensusAtlasBatchParams` default batch_size=32 (l.36-58), `ConsensusAtlasBatchError` (l.86+), `render_consensus_atlas` (l.146); `RansacPhotometricParams::tile_index_base` (photometric_ransac.rs l.59); `crates/sfmtool-py/src/py_consensus_atlas.rs`; `src/sfmtool/rig/panorama.py`.
**Inconsistencies:**
  - None found. Params struct, error enum variants, sequential batch loop with `div_ceil`, and `tile_index_base` plumbing all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/core/track-cluster-matching.md
**Summary:** Descriptor-clustering matcher: build a kd-forest over all descriptors, query k-NN, set a per-descriptor background-floor radius (`α·B_i`, `B_i` = d-th-nearest distance), materialize density-ordered clusters (one feature/image, hard partition), convert to per-image-pair matches, verify and write `.matches`.
**Implementing code:** `crates/sfmtool-core/src/features/cluster_match/mod.rs` (`BackgroundFloorParams` default `d=10, alpha=0.8, min_size=2, forest=accurate`, l.53-56; algorithm l.183-320); `crates/sfmtool-py/src/matching/cluster.rs`; `src/sfmtool/feature_match/_cluster_matching.py`; `src/sfmtool/_commands/match.py`.
**Inconsistencies:**
  - Spec-internal staleness against the code default: the algorithm section §2 states the floor rank literally as `B_i = dist[i, d]   (d = 28)` (l.198) and "keeping neighbours within 0.8× the **28th**-nearest distance" (l.109); §"Iterating on cluster membership rules" says "its **48** nearest neighbours" (l.87); §1 gives query width "`d + 1 = 29`" (l.181-186). The shipped default is `d=10` (query width 11), which the later "Implementation Status" and Production/Parameter tables correctly reflect. The algorithm prose was not updated when `d` was lowered 28→10.
**Recommendation:** update spec — reconcile the §1/§2 prose (`d=28`, "28th-nearest", "29", "48 nearest") with the production default `d=10`, or explicitly mark those numbers as the original prototype tuning.
**Unclear / incorrect / suspicious:** Because the spec both keeps the prototype numbers inline and separately documents `d=10` as production default, a reader hits contradictory rank numbers depending on section. Code is unambiguously `d=10`.

## Format specs

### specs/formats/camrig-file-format.md
**Summary:** Defines the `.camrig` ZIP format storing a camera rig (sensors sharing a camera pool, `sensor_from_rig` poses as WXYZ quats + XYZ translations), its JSON metadata/content-hash members, image-pattern rules, and the v1→v2 canonical-convention migration (S-conjugation for body-anchored rigs, left-S multiply for `spherical_tiles`).
**Implementing code:** `crates/camrig-format/src/types.rs` (`CAMRIG_FORMAT_VERSION = 2`, `s_conjugate_sensor_pose`, `s_premultiply_sensor_pose`, `CamRigData::upgrade_sensor_poses_from_v1`, `CamRigData::validate`), `read.rs` (v1 upgrade-on-load, lines 44-46), `write.rs` (member names/order).
**Inconsistencies:**
  - None found. Member names/order (`write.rs:57-115`), content-hash fields, version 2, and the two-path v1 upgrade all match the spec.
  - Minor (code stricter than spec): `validate()` rejects zero camera width/height (`types.rs:334-342`); the spec's constraints list does not mention this rule.
**Recommendation:** in sync — spec and code agree on every field, member name, version, and migration rule.
**Unclear / incorrect / suspicious:** The CLI-inspection example (camrig-file-format.md:419-434) shows `"version": 1` in `metadata.json`, but the writer now always emits version 2. Consider bumping the example to `2` to avoid implying new files are v1.

### specs/formats/matches-file-format.md
**Summary:** Defines the `.matches` ZIP format: candidate matches (columnar image-pair/feature-index/descriptor-distance arrays) plus an optional two-view-geometries section (F/E/H matrices, `cam2_from_cam1` relative poses), content-hash scheme, and the v1→v2 canonical-pose migration (F/E/H unchanged).
**Implementing code:** `crates/matches-format/src/types.rs` (`MATCHES_FORMAT_VERSION = 2`, `s_conjugate_relative_pose`, `TwoViewGeometryConfig`, `MatchesContentHash`), `read.rs` (v1→v2 upgrade at lines 47-54, 254-265).
**Inconsistencies:**
  - None found. The 10 `TwoViewGeometryConfig` strings exactly match the spec's `config_types` list; content-hash fields match; `config_indexes` is `uint8`; v1 poses S-conjugate on load while F/E/H are untouched.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/formats/sfmr-file-format.md
**Summary:** The large `.sfmr` reconstruction format spec — versions 1–5, homogeneous points, optional rigs/frames, `feature_source` (`sift_files`/`embedded_patches`), 13-model camera table incl. the `EQUIRECTANGULAR` extension, and the v5 canonical Z-up/−Z-forward convention.
**Implementing code:** `crates/sfmr-format/src/types.rs` (`SFMR_FORMAT_VERSION = 5`, `feature_source` lines 151-218), `crates/sfmtool-core/src/camera/intrinsics.rs` (all 13 models incl. `Equirectangular`, lines 150-161), v5 upgrade-on-load in `sfmtool-core`.
**Inconsistencies:**
  - None found. Version constant is 5; the Rust `CameraModel` enum enumerates exactly the spec's 13 models; `feature_source` discriminator with `sift_files` default matches.
**Recommendation:** in sync — the v5 semantic bump is implemented per the migration draft.
**Unclear / incorrect / suspicious:** Nothing material.

### specs/formats/sift-file-format.md
**Summary:** The `.sift` per-image feature format — v1 (single descriptor array) and a **draft** v2 (chunked append-only descriptors, `described_count`, hash fields, `image_to_gray` formula). The spec itself flags v2 as not-yet-implemented.
**Implementing code:** `crates/sift-format/src/types.rs` (`SIFT_FORMAT_VERSION = 1`; `SiftData` has a single `descriptors: Array2<u8>`), `read.rs::check_version` (rejects `version > 1`, lines 24-31).
**Inconsistencies:**
  - None found — and this is the intended state. The spec's implementation-status callout (lines 45-52) says the codebase "reads and writes version 1 only"; code matches exactly, and no v2 machinery exists.
**Recommendation:** in sync — the code correctly implements v1 only, and the spec accurately labels v2 items as a draft.
**Unclear / incorrect / suspicious:** Nothing — this spec self-documents its unimplemented portion accurately.

## Workspace specs

### specs/workspace/camera-config.md
**Summary:** `camera_config.json` schema (`version:1`, `camera_intrinsics` with `model`/`width`/`height`/`parameters`), partial-intrinsics tiers (model-only, distortion-only, full), closest-ancestor resolution, resolution-mismatch scaling rules, and the presence-based `--camera-model` rejection.
**Implementing code:** `src/sfmtool/camera/config.py` (`load_camera_config`, `find_camera_config_for_directory`, `CameraConfigResolver`), `src/sfmtool/camera/cameras.py`, `src/sfmtool/camera/setup.py` (`build_intrinsics_from_camera_config` tiers + scaling lines 128-200, `_check_camera_model_conflict`).
**Inconsistencies:**
  - None found. The three tiers, the "focal/principal ⇒ width+height required" rule (`config.py:83-91`), uniform focal+principal scaling with distortion pass-through (`setup.py:182-188`), aspect-ratio-mismatch error (`setup.py:172-179`), and `--camera-model` hard-reject (`setup.py:240-261`) all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/workspace/rig-config.md
**Summary:** `rig_config.json` is COLMAP's rig-configurator format verbatim (array of rigs; per-sensor `image_prefix`/`ref_sensor`/`cam_from_rig_rotation` WXYZ/`cam_from_rig_translation`/`camera_model_name`/`camera_params`), stays in COLMAP convention (D4), read only from the workspace root, WXYZ→XYZW conversion in `_sensor_from_rig_pose`.
**Implementing code:** `src/sfmtool/rig/config.py` (`_load_rig_config`, `_match_image_to_sensor`, `_infer_frame_key`, `_sensor_from_rig_pose` with WXYZ→XYZW at lines 60-63 and explicit "no S-conjugation" D4 comment), `rig/frames.py`, `camera/setup.py:53-61` (fisheye focal seed `min(w,h)/π`), `crates/sfmtool-core/src/camera/distortion.rs:649` (`blend_fisheye_ray`).
**Inconsistencies:**
  - None found. Every code symbol the spec cites by name exists and behaves as described.
**Recommendation:** in sync — code matches the COLMAP-native schema and the D4 convention decision.
**Unclear / incorrect / suspicious:** Nothing.

### specs/workspace/workspace.md
**Summary:** Defines `.sfm-workspace.json` (`version:1`, `feature_tool`/`feature_type`/`feature_options`/`feature_prefix_dir`), `sfm ws init` behavior, feature-prefix hashing (`use_gpu` excluded), nesting protection, upward workspace discovery, and `.sfmr` workspace-reference path resolution.
**Implementing code:** `src/sfmtool/_workspace.py` (`init_workspace` writes `version:1`, lines 68-74; `find_workspace_for_path`), `src/sfmtool/_commands/ws.py` (`ws init` guards lines 117-130).
**Inconsistencies:**
  - Default feature tool: code defaults to `"sfmtool"` (`ws.py:25`, `_workspace.py:12`), but the spec never states a default and every File-Format example plus the `.sfmr`-embedding example uses `"feature_tool": "colmap"` (workspace.md:72, 233). Not a hard contradiction, but the examples imply COLMAP is primary while bare `sfm ws init` now produces a sfmtool workspace.
**Recommendation:** update spec — add one sentence naming `sfmtool` as the default tool (and/or switch a representative example to a sfmtool-shaped `feature_options` block).
**Unclear / incorrect / suspicious:** A reader running bare `sfm ws init` gets a sfmtool-shaped config (`octave_layers`, `contrast_threshold`, …) instead of the COLMAP-shaped example. Version, nesting protection, and `use_gpu` hash-exclusion are all correctly implemented.

## Drafts

### specs/drafts/zup-camera-convention-migration.md
**Summary:** The roadmap for the global Z-up world / −Z-forward camera convention flip (`S`/`W` matrices, format bumps `.sfmr`→v5, `.camrig`→v2, `.matches`→v2, D1–D7 decisions, per-area Rust/Python/GUI/test work).
**Implementation status: DONE (fully implemented).** The draft's own header declares "**Status: IMPLEMENTED** (2026-07-04, branch `zup-core-flip`)", independently confirmed in code:
  - D1/D5/D6 version bumps landed: `SFMR_FORMAT_VERSION = 5` (sfmr-format/types.rs:210), `CAMRIG_FORMAT_VERSION = 2` (camrig-format/types.rs:66), `MATCHES_FORMAT_VERSION = 2` (matches-format/types.rs:88).
  - Upgrade-on-load present in all three readers.
  - D2 convention module exists (`crates/sfmtool-core/src/geometry/convention.rs`; Python wrapper `src/sfmtool/colmap/convention.py`).
  - D4 honored: `rig/config.py::_sensor_from_rig_pose` keeps COLMAP convention with no S-conjugation.
**Inconsistencies:** None between the draft and code — it is a completed record.
**Recommendation:** discuss (housekeeping) — the document lives in `specs/drafts/` but is fully implemented; consider moving it out of `drafts/` or marking it archival. Consider also promoting the coordinate-convention contract to a stable home (e.g. a short `specs/core/` conventions note or a section of `sfmr-file-format.md`) so the canonical convention doesn't live only in a migration record.
**Unclear / incorrect / suspicious:** The single acknowledged open item is non-code: the human-in-the-loop GUI visual confirmation (§12 step 4/6) is deferred "needs a display."

## User docs

### docs/index.md
**Summary:** Landing page introducing the SfM toolkit, the SfM Explorer GUI, and the `sfm` CLI. Shows a minimal end-to-end workflow (`mkdir`, `sfm ws init`, `sfm solve -g`, `sfm inspect`, `sfm analyze --metrics`) and a `pip install sfmtool` install instruction.
**Implementing code:** `src/sfmtool/cli.py`, `_commands/ws.py`, `solve.py`, `_global_sfm.py`, `_incremental_sfm.py`, `_workspace.py`, `sift/extract_sfmtool.py`, `pyproject.toml`, `pixi.toml`.
**Inconsistencies:**
  - None found. Command surface, printed output strings, and config defaults (`contrast_threshold: 0.0067`, `octave_layers: 3`, `max_num_features: 8192`) all match. `pip install sfmtool` matches the package name/entrypoint.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** `sfm version` prints `sfmtool 0.1` (cli.py:90) while `pyproject.toml`/`pixi.toml` declare version `0.2`. Not a divergence for this doc, but the stale version string is worth fixing for whoever owns release metadata.

### docs/tutorials/getting-started.md
**Summary:** Full walkthrough: install, create/init a workspace, run the global solver, explore in the GUI, convert video to frames with ffmpeg, view point tracks in SfM Explorer, and inspect a reconstruction via `sfm inspect -v`, `sfm analyze --metrics`, `sfm analyze --z-range`, and `sfm motion`.
**Implementing code:** `_commands/ws.py`, `solve.py`, `explorer.py`, `inspect.py`, `analyze.py`, `motion.py`, `_global_sfm.py`, `_incremental_sfm.py`, `pixi.toml`.
**Inconsistencies:**
  - None found. Every command, flag, and output line in the tutorial resolves to real code, including `sfm motion <file>.sfmr` (single-`.sfmr` branch at motion.py:104-125).
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** The note at getting-started.md:64-67 says "The `sfm explorer` command is not yet included in the PyPI package" — slightly imprecise: the *command* is registered and ships (cli.py:71); what's missing from the wheel is the `launch-sfm-explorer` GUI binary it invokes (explorer.py:15-21). The suggested `pixi run gui` workaround runs the `sfm-explorer` crate binary directly, which is functionally valid. Also "uses COLMAP and GLOMAP under the hood" (line 6) elides that the default feature extractor is sfmtool's own Rust SIFT — a minor simplification, not a hard error.

## GUI specs

Cross-cutting: the **"octahedron" → compass rose** rename is stale in four specs
(gui-plan, gui-architecture, gui-viewport-navigation, gui-point-cloud-rendering), and
the **`main.rs` → `app.rs`/`dock.rs`** + **`read_back_pick` → `read_readback_result`**
location drift recurs in gui-camera-views and gui-multi-panel-image-browser. Fixing
those two patterns clears the bulk of the GUI divergences; the items needing an actual
decision are the camera-views per-frame-FOV contradiction, the multi-panel deselect
contradiction, and the point-track all-black patch-tile behavior.

### specs/gui/README.md
**Summary:** Index/table-of-contents for the GUI spec set, with a one-line description of each of the other 13 documents.
**Implementing code:** N/A (navigational index).
**Inconsistencies:**
  - None found. Every linked document exists and the one-line descriptions match each doc's content.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/gui/blender-viewport-navigation-implementation-overview.md
**Summary:** Reference analysis of how Blender implements precision-touchpad navigation on Windows via DirectManipulation; background material for `platform/windows.rs`.
**Implementing code:** Informational only; the design it informed is specified in `gui-viewport-navigation.md`.
**Inconsistencies:**
  - None found — external reference material, no direct code contract.
**Recommendation:** in sync (reference document).
**Unclear / incorrect / suspicious:** Nothing.

### specs/gui/gui-adaptive-clip-and-grid.md
**Summary:** Reversed-Z infinite-far projection, adaptive near plane from scene bounding sphere, and adaptive ground grid scaled to `length_scale`.
**Implementing code:** `viewer_3d/camera.rs` (`projection_matrix`, `update_clip_planes`), `viewer_3d/overlay.rs` (`draw_grid`), `scene_renderer/auto_point_size.rs`.
**Inconsistencies:**
  - None found. The near-plane formula `((d + scene_radius)/1000).max(0.0001)` with `alpha = 1 - exp(-dt*8)` matches `camera.rs:330-338` verbatim; the reversed-Z matrix matches `camera.rs:356-380`; grid step/extent/axis sizing matches `overlay.rs:27-34`.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing — the spec correctly notes the grid is still an egui overlay (no depth occlusion).

### specs/gui/gui-architecture.md
**Summary:** Technology stack, crate/module layout, multi-pass render pipeline, GPU readback, build system, performance budget, platform notes.
**Implementing code:** `crates/sfm-explorer/src/lib.rs`, `app.rs`, `dock.rs`, `scene_renderer/mod.rs` + `render.rs`, `viewer_3d/`.
**Inconsistencies:**
  - Module map lists `image_detail.rs` as a single file (lines 131, 167), but it is a directory: `image_detail/{mod,input,overlay}.rs`.
  - The `shaders/` listing (lines 137-145) omits `patch.wgsl`, which exists and is central to the implemented patch pipeline.
  - `target_indicator.wgsl # Rotating octahedron at target` — stale; geometry is a compass rose (star + spikes + ring).
  - Module table describes `bg_image.rs` as "Background image pipeline (pinhole)" (line 125), but there is no distinct pinhole background pipeline anymore — `scene_renderer/mod.rs` holds only `bg_image_distorted_pipeline`; per gui-camera-views Step 8 the separate pinhole BG pipeline/shader was removed.
**Recommendation:** update spec — fix the `image_detail` directory, add `patch.wgsl`, correct the octahedron and pinhole-BG descriptions. Core prose (pass order, four-phase `run_ui_and_paint`, 5×5 readback, render targets) is accurate.
**Unclear / incorrect / suspicious:** Nothing beyond the above.

### specs/gui/gui-camera-views.md
**Summary:** Frustum wireframes, image-quad texturing, GPU pick buffer, selection/hover, camera-view mode with full-res background, distorted + fisheye frustum tessellation, and Step-9 persistent-camera-view free-look.
**Implementing code:** `scene_renderer/upload.rs` (`upload_frustums`, `update_frustum_colors`, `upload_bg_image`), `pipelines/{frustum,image_quad,distorted_quad,bg_distorted}.rs`, `scene_renderer/distorted_mesh.rs`, `viewer_3d/mod.rs` (`enter_camera_view`, `compute_switch_camera_view`), `viewer_3d/camera.rs` (`best_fit_fov`).
**Inconsistencies:**
  - **Internal contradiction on FOV recompute.** §FOV best-fit (lines 514-520) and Step-5 checklist item 4 (line 1501) state the viewport FOV is recomputed every frame; Step 9 (lines 1619-1625, 1708 "Stopped per-frame FOV override") states the per-frame override was removed. The **code follows Step 9**: `enter_camera_view` computes `best_fit_fov` once and stores it in the transition (`viewer_3d/mod.rs:492-524`); there is no per-frame recompute. The earlier prose/checklist item is stale — a reader following §FOV best-fit would implement the opposite of shipped behavior.
  - Crate name error: line 55 says `ViewportCamera` is "in `sfmtool-gui`". The GUI crate is `sfm-explorer`; there is no `sfmtool-gui`.
  - Location drift: pick/click handling described as living in `main.rs` is actually in `app.rs::process_pick_readback`; the readback method is `read_readback_result`, not `read_back_pick`.
  - Addition not in spec: `update_frustum_colors` (`upload.rs:366-388`) has a `color_hidden` (alpha 0) state for the frustum being viewed-through, absent from the selection-color table (lines 172-176).
**Recommendation:** update spec — reconcile the per-frame-FOV prose with Step 9 (FOV set once on entry), fix the `sfmtool-gui` crate name, refresh `main.rs` references, and document the hidden-frustum color.
**Unclear / incorrect / suspicious:** The self-contradiction on per-frame FOV is the main hazard.

### specs/gui/gui-cross-panel-hover.md
**Summary:** Transient cross-panel hover of images/points via `AppState::hovered_image`/`hovered_point`, panel `has_pointer` ownership, GPU uniform highlighting with `0xFFFFFFFF` sentinel, one-frame readback gating.
**Implementing code:** `state.rs`, `dock.rs` (per-panel `has_pointer` ownership at 114-120, 209-215, 286-292), `app.rs::process_pick_readback`, `app.rs:379-386` (suppress hover == selection).
**Inconsistencies:**
  - None found.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/gui/gui-image-animation.md
**Summary:** Image-sequence playback: `AnimationState`/`PlayDirection`, keyboard Space/arrows/brackets, minibar play button + FPS label, camera-view flipbook via `request_camera_switch`.
**Implementing code:** `image_browser.rs` (`AnimationState`, keyboard block :267-317, frame advance :319-360, play button :563-627), `dock.rs:135-139`.
**Inconsistencies:**
  - None found. Default fps 10, range 1–60, `looping: true`, wall-clock timing, pause-on-manual-interaction, `<2 images` no-op, reset on reconstruction change — all match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/gui/gui-multi-panel-image-browser.md
**Summary:** egui_dock 4-panel layout, image/point selection model, cross-panel hover, image browser, image detail with 7 overlay modes + feature filtering, track ray viz, navigation minibar, image-detail 2D pan/zoom.
**Implementing code:** `dock.rs`, `image_browser.rs`, `image_detail/`, `state.rs` (`OverlayMode`, `FeatureDisplaySettings`), `scene_renderer/upload.rs::upload_track_rays`.
**Inconsistencies:**
  - **Internal contradiction on deselect.** Overview line 70 says "clicking the selected thumbnail again clears `selected_image`," but the detailed Image-Browser section (line 307) says "Clicking an already-selected thumbnail keeps it selected (no toggle)." The **code matches line 307**: `image_browser.rs:534` always sets `selection_changed = Some(Some(i))` on click — no toggle-off. Line 70 is wrong.
  - Location/method drift: "Integration in main.rs", `read_back_pick()`, and the "around line 654" snippet all refer to `main.rs`; the logic is in `app.rs` (`run_egui_pass`, `process_pick_readback`) and the method is `read_readback_result`. `main.rs` is a 6-line shim.
  - Phase B (evaluation) and Step 12 (co-track points) correctly marked NOT STARTED / deferred — forward-looking.
**Recommendation:** update spec — fix the line-70 deselect contradiction and refresh `main.rs`/`read_back_pick` references. Panel behaviors, 7 overlay modes, filtering, minibar, and 2D pan/zoom (`MAX_ZOOM=32`, `PAN_MARGIN=50`) are all in sync.
**Unclear / incorrect / suspicious:** The two-places-disagree deselect behavior is the notable hazard.

### specs/gui/gui-patch-rendering.md
**Summary:** Embedded-patch surfel rendering — per-instance oriented textured quads in Pass 1, front-face-culled in the vertex shader, page-grid texture-array atlas, `PICK_TAG_POINT` picking, View-menu controls.
**Implementing code:** `pipelines/patch.rs`, `shaders/patch.wgsl`, `scene_renderer/upload.rs::upload_patches`, `gpu_types.rs` (`PatchInstance`/`PatchUniforms`), `app.rs:539-564`, `state.rs:135-149`.
**Inconsistencies:**
  - None material. `show_patches`/`patch_opacity`/`patch_size_log2`/`patch_alpha_cutoff` defaults (on/1.0/0.0/0.0) match `state.rs:228-231`; UI disabled unless frame+bitmaps present; `patch_opacity > 0.0` skip-draw; front-face cull, v-flip, MRT behavior all match. The flat-shaded frame-without-bitmaps fallback is correctly documented as Planned.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/gui/gui-plan.md
**Summary:** Roadmap and implementation-status tracker for the GUI. Lists completed 3D-viewer features, core-data status, PyO3 status, and next steps.
**Implementing code:** Whole `sfm-explorer` crate; status cross-checked against `app.rs`, `viewer_3d/`, `scene_renderer/`.
**Inconsistencies:**
  - The roadmap has **no mention of patch/surfel rendering** or the Point Track Detail panel's patch tiles, both implemented (`scene_renderer/upload.rs::upload_patches`, `pipelines/patch.rs`; gui-patch-rendering.md marked "Implemented v1"). The "Current Implementation Status" list should include patches.
  - Stale wording: line 47 "Target indicator (rotating octahedron)" — the indicator is now a compass rose (`CompassEdgeInstance` in `scene_renderer/mod.rs:69-72`, `shaders/target_indicator.wgsl`).
  - Next Steps #1 (grid depth occlusion) and #2 (free-nav FOV gesture) correctly still pending; #3/#4 correctly marked Done.
**Recommendation:** update spec — add patches to the status list and fix the octahedron wording; otherwise the roadmap assessment is accurate.
**Unclear / incorrect / suspicious:** Nothing.

### specs/gui/gui-point-cloud-rendering.md
**Summary:** Point-splat billboards, `PointInstance` layout, auto/user point sizing, EDL post-process, compass target indicator, supernova effect, points-at-infinity.
**Implementing code:** `pipelines/points.rs` + `shaders/points.wgsl`, `pipelines/edl.rs` + `shaders/edl.wgsl`, `pipelines/target.rs`, `scene_renderer/gpu_types.rs`, `state.rs`.
**Inconsistencies:**
  - Stale wording in the Implementation-Status list: line 443 "[x] Target indicator (rotating **wireframe octahedron**)" contradicts line 457 in the same list ("[x] Target indicator redesign: 3D compass shape…"), which is what the code implements.
  - Points-at-infinity rendering + `infinity_point_px` slider shipped (`app.rs:532-537`, `state.rs:235` default 3.0, range 1–16) — matches. The "Show points at infinity" toggle and `N points (M at infinity)` readout are correctly still marked future.
**Recommendation:** update spec — change the line-443 checklist item to the compass. Otherwise in sync.
**Unclear / incorrect / suspicious:** Nothing beyond the recurring octahedron wording.

### specs/gui/gui-point-track-detail.md
**Summary:** Per-point track inspector — summary header with copy-able Point ID + XYZ, stored-patch header tile, scrollable observation table with per-observation thumbnail/patch/error/angle, cross-panel select/hover.
**Implementing code:** `point_track_detail.rs`, cached data in `state.rs` (`sift_cache`, `full_res_cache`), dock wiring `dock.rs:222-292`.
**Inconsistencies:**
  - **All-black patch-tile behavior diverged.** Spec §Observation Table (line 164) and the panel-state struct (`rendered_patch_textures: HashMap<usize, Option<egui::TextureHandle>>`) say an all-black render "is memoized and the tile is simply not drawn." The code uses `HashMap<usize, egui::TextureHandle>` (no `Option`) and `ensure_rendered_patch` (`point_track_detail.rs:652-695`) always inserts and draws the tile even when the warp is all-black — its own doc comment admits "warps to an all-black tile and is drawn as such."
  - **Code is ahead of spec on columns/diagnostics.** The observation table has a "Size" column (`feature_size`, `point_track_detail.rs:349, 517-529`) not in the spec's column table. The header shows "depth z" (`inverse_depth_z`) and "cond" (`condition_number`) diagnostics (:279-286), whereas the spec's header only documents Max angle. `TrackObservationData` also carries `feature_size`/`image_full_name` beyond the spec's struct.
**Recommendation:** update spec — document the "Size" column and the depth-z/condition-number header diagnostics, and reconcile the all-black tile behavior (spec says "not drawn"; code draws it).
**Unclear / incorrect / suspicious:** The all-black tile is a genuine behavior question — the code even flags "a future N/A flag may distinguish 'not visible' from a genuinely dark surface"; worth a deliberate decision rather than silent drift.

### specs/gui/gui-user-experience.md
**Summary:** Vision, design principles, the interaction cheat-sheet (trackpad/mouse), overlays, View-menu controls, four-panel layout.
**Implementing code:** `app.rs` (View menu :517-586), `viewer_3d/input.rs` + `overlay.rs`, `dock.rs`.
**Inconsistencies:**
  - None material. The Core-Interactions table matches `viewer_3d/input.rs` (Alt+drag free-look, Alt+scroll target push/pull, WASD+R/F fly, Z fit / view-through, double-click frustum → camera view). View-menu controls (Show Points/Camera Images/Grid, Point Size log₂ −3..+3, Length Scale, FOV 10–120) match.
**Recommendation:** in sync.
**Unclear / incorrect / suspicious:** Nothing.

### specs/gui/gui-viewport-navigation.md
**Summary:** Orbit-camera model, full mouse/trackpad/keyboard control mapping, Alt-mode target control (dual-orbit, nodal pan, push/pull, depth-pick), zoom-to-fit, fly nav, FOV convention, Windows DirectManipulation.
**Implementing code:** `viewer_3d/camera.rs`, `viewer_3d/input.rs`, `viewer_3d/mod.rs`, `platform/windows.rs`.
**Inconsistencies:**
  - Stale implementation note: lines 607-609 "Render as a small set of instanced lines (wireframe octahedron = 12 edges)" — the indicator is now the compass rose. The status checklist further down (line 652) records the compass redesign as done, so prose and checklist disagree within the same file.
  - Everything else verified concretely: orbit sensitivity 0.01 (`camera.rs:115-116`), zoom min-distance clamp 0.1 (`camera.rs:141`), Alt double-tap 300 ms (`mod.rs:220`), 200 ms transitions, Home/Shift+Home (`input.rs:455-468`), `,`/`.` behavior (`input.rs:424-453`). Open questions all correctly still unimplemented.
**Recommendation:** update spec — replace the octahedron implementation note with the compass geometry. Substantively in sync.
**Unclear / incorrect / suspicious:** Nothing.

## Code without specs

Coverage is near-total: all 27 registered CLI commands map to `specs/cli/*` (with
`explorer` the one thin exception below), every crate's algorithm surface maps to
`specs/core/*` or `specs/formats/*`, and no user-facing file format is unspecced.
The `xform` sub-operations are all covered between the umbrella spec and the six
dedicated `xform-*` specs. Confirmations from the sweep:

- **AGENTS.md `cam` group reference is stale.** There is no `cam` group; the group is
  registered as **`camrig`** (`src/sfmtool/cli.py:80`, `_commands/camrig.py:15`) and now
  has **three** subcommands (`create`, `cp`, `spherical-tiles`), not the single `cam cp`
  AGENTS.md line 89 describes. AGENTS.md also says "7 crates" but `crates/` now has 8
  (`camrig-format` was split out). Worth an AGENTS.md touch-up.
- **No forward-looking spec is missing code.** The only `specs/drafts/` entry
  (zup migration) is implemented; every `specs/core/*` file spot-checked has
  corresponding code.

### CLI command: `sfm explorer` (src/sfmtool/_commands/explorer.py)
**What it does:** A thin launcher that shells out to the `launch-sfm-explorer` binary (the Rust `sfm-explorer` crate) to open the 3D viewer on an optional `.sfmr` file. Registered under "Visualization" (cli.py:71). The only registered CLI command with no `specs/cli/*-command.md`.
**Why it matters:** user-facing (the GUI entry point), but behaviorally trivial — all substance lives in the GUI, extensively documented across `specs/gui/`.
**Recommendation:** acceptable as unspecced; add a one-line cross-reference in `specs/gui/README.md` or `gui-architecture.md` stating that `sfm explorer [FILE.sfmr]` is the launch entry point. A full CLI spec would not earn its place.

### Crate: `sfmtool-py` (crates/sfmtool-py/src/)
**What it does:** The PyO3 binding layer exposing `sfmtool-core` to Python (`py_sfmr_reconstruction.rs`, `py_patch_cloud.rs`, `py_consensus_atlas.rs`, etc.) — the load-bearing bridge every Python command reaches core algorithms through.
**Why it matters:** internal-but-load-bearing glue; the binding *surface* is nowhere described as a contract, only implicitly across the per-algorithm core specs.
**Recommendation:** acceptable as unspecced — a binding-layer spec would mostly duplicate PyO3 signatures that the per-algorithm specs already document.

### Module: `crates/sfmtool-core/src/geometry/` (transform & convention primitives)
**What it does:** Foundational math: `rigid_transform.rs`, `se3_transform.rs`, `rot_quaternion.rs`, `rotation.rs`, `viewing_angle.rs`, `transform.rs`, and `convention.rs` (the canonical Z-up / −Z-forward coordinate conventions). Used pervasively.
**Why it matters:** internal-but-load-bearing; the *conventions* piece is subtle and error-prone (sign/handedness) and is currently documented only inside the implemented migration draft and the `.sfmr` format's conventions section.
**Recommendation:** acceptable as unspecced for the pure transform math, but promote the coordinate-convention contract out of `drafts/` — e.g. a "Coordinate & Transform Conventions" note in `specs/core/` or a pointer section in `specs/formats/sfmr-file-format.md` — so the canonical convention doesn't live only in a historical migration record.

---

## Top priorities

1. **Reconcile `specs/core/keypoint-subpixel-refinement.md` with the shipped `subpixel` default** — the spec says sub-pixel LK ships opt-in behind `subpixel: str = "none"`; the code ships an integer sweep count defaulting to **1 (LK on by default)** (`_embed_patches.py:597`, `_commands/embed_patches.py:85-96`), and `lk_per_move` is not reachable from the pipeline. This is a behavior-level claim about what production runs, so decide deliberately (is on-by-default intended?) and rewrite the wiring paragraph.

2. **Update `specs/core/sift-to-patch-reconstruction.md` for the multi-round pipeline** — the code runs `rounds=2` alternating normal/keypoint refinement with obliquity down-weighting and a fronto prior, plus ~8 knobs (`rounds`, `max_obliquity_deg`, `obliquity_weight_power`, `fronto_prior_weight`, `max_refine_views`, `localize_search_strategy`, `search_resolution_multiplier`, `subpixel`) absent from the spec's single-pass, 4-knob description. This is the home spec for `embed-patches`; it materially understates the shipped algorithm. Related smaller fixes in the same area: `patch-keypoint-localization.md` should note the PlusDescent default (not exhaustive search), and `keypoint-localization-search-cache.md` needs its stale "proposed / still scalar" header bumped to implemented.

3. **Decide the fate of the three dead `sfm align` options** — `--max-error`, `--iterative`, and `--visualize` are accepted and documented as functional but never used (`align/multi.py:315-317`). Implement them or remove them (and update the spec either way); today users get silent no-ops.

4. **Close the `scale-by-measurements` spec/code gap** — two spec'd behaviors are unimplemented: the workspace hash-prefix search for the source `.sfmr` (`_scale_by_measurements.py:356-377` errors immediately without an explicit `sfmr` field) and "use the most common point index" on ambiguous matches (code takes an arbitrary first match at `:132` while printing "Using most common." — an actively misleading message). Implement or downgrade the spec sections to forward-looking; fix the message regardless.

5. **Sweep the GUI specs for the recurring staleness + three real contradictions** — mechanical: "rotating octahedron" → compass rose (4 files), `main.rs`/`read_back_pick` → `app.rs`/`read_readback_result` (2 files), `sfmtool-gui` → `sfm-explorer`, missing `patch.wgsl`/patch-rendering mentions in gui-architecture/gui-plan. Decisions: per-frame FOV recompute contradiction in `gui-camera-views.md` (code sets FOV once on entry, per Step 9), thumbnail deselect contradiction in `gui-multi-panel-image-browser.md` (code: no toggle-off), and the all-black patch-tile behavior in `gui-point-track-detail.md` (spec: not drawn; code: drawn).

Honorable mentions: the dead `spatial_tolerance` parameter in flow matching (`_flow_matching.py:180` — real radius is the hard-coded 10 px, K=5 best-descriptor, not the spec's 3 px single-nearest; wire it or delete it), the `d=28`-era prose in `track-cluster-matching.md` vs the shipped `d=10`, the workspace spec's COLMAP-shaped examples vs the `sfmtool` default tool, AGENTS.md's stale `cam`-group/7-crates text, `sfm version` printing `0.1` vs the declared `0.2`, and relocating the implemented zup migration doc out of `specs/drafts/`.
