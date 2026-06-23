# Spec / Code Consistency Audit — 2026-06-23

Bidirectional audit of every spec under `specs/` and `docs/` against the
implementing code, plus a code → spec sweep of every significant code
surface. Read-only analysis; nothing was modified.

Scope covered: 29 CLI specs in `specs/cli/`, 16 core specs in `specs/core/`,
5 format specs in `specs/formats/`, 13 GUI specs in `specs/gui/`, 3 workspace
specs in `specs/workspace/`, 2 docs pages — and code-side: all 26
`_commands/*.py` modules, 8 crates, the workspace/camera-config modules, and
the four on-disk formats.

This snapshot **supersedes `2026-06-09-spec-audit.md`** (retire decision
documented in the 2026-06-23 hygiene audit's Finding 15). The two intervening
weeks landed PRs #107/#109/#110/#111 (the sfmtool-core regroup arc), which
moved nearly every file in `crates/sfmtool-core/src/` into a topic group
(`geometry/`, `camera/`, `reconstruction/`, `features/`, `analysis/`,
`spherical/`, `patch/`). As a result, the dominant theme of this snapshot is
**path drift**: 14 of 16 core specs and several format / GUI / CLI specs
still cite pre-regroup flat paths. Most are mechanical doc-only fixes. What
the rebaseline also surfaced is a smaller set of genuine spec/code
divergences (some real bugs in CLI handlers, a `.sfmr` format spec that is
one version stale, three GUI specs that miss the 4th panel that shipped, and
several code surfaces that have no spec at all).

The canonical relocations callers need to remember for the spec edits:
`epipolar.rs` → `camera/epipolar.rs`; `warp_map.rs` → `camera/warp_map.rs`;
`remap.rs` → `camera/remap.rs`; `triangulation.rs` →
`reconstruction/triangulation.rs`; `viewing_angle.rs` →
`geometry/viewing_angle.rs`; `geometric_filter.rs` →
`features/feature_match/geometric_filter.rs`; `infinity/*` →
`analysis/infinity/*`; `optical_flow/*` → `features/optical_flow/*`; `sift/*`
→ `features/sift/*`; `kdforest/*` → `features/kdforest/*`; `cluster_match/*`
→ `features/cluster_match/*`; `patch_cloud.rs` → `patch/cloud.rs`;
`patch_normal_refine.rs` → `patch/normal_refine.rs`; `spherical_tile_rig.rs`
→ `spherical/tile_rig.rs`; `per_spherical_tile_source_stack.rs` →
`spherical/per_tile_source_stack.rs`; `photometric_ransac.rs` →
`spherical/photometric_ransac.rs`; `consensus_atlas.rs` →
`spherical/consensus_atlas.rs`; `sphere_points.rs` →
`spherical/sphere_points.rs`.

---

## CLI Command Specs

### specs/cli/align-command.md
**Summary:** Aligns multiple `.sfmr` reconstructions to a reference using `points` or `cameras` matching, with optional RANSAC and iterative refinement.
**Implementing code:** `src/sfmtool/_commands/align.py` → `align.multi.align_command`.
**Inconsistencies:** Basename-collision and inside-output-dir rejection promised at spec L46-53 not visible at the Click handler — presumably in `align/multi.py`.
**Recommendation:** discuss — verify checks live in `align_command`.

### specs/cli/analyze-command.md
**Summary:** Deep-analysis report on a `.sfmr` with six mutually exclusive modes (`--coviz`, `--z-range`, `--frustum`, `--images`, `--metrics`, `--depth-reliability`).
**Implementing code:** `src/sfmtool/_commands/analyze.py::analyze()`.
**Inconsistencies:** Spec doesn't note `FloatRange(0.0, 100.0)` constraint on `--near-percentile`/`--far-percentile` (code L67) nor the `near < far` precondition (L180-184).
**Recommendation:** update spec.

### specs/cli/camrig-command.md
**Summary:** `camrig` group with `create`, `cp`, `spherical-tiles`. Documents the arg-order swap: `<IMAGE_PATTERN> <OUTPUT_FILE>` on `create`.
**Implementing code:** `_commands/camrig.py`.
**Inconsistencies:** Arg-order swap correct. `cp`'s geometry-only fallback output line (camrig.py L264-268) not documented. Minor wording drift around "rig root" vs OUTPUT_FILE directory.
**Recommendation:** update spec — polish wording, document the geometry-only fallback.

### specs/cli/compare-command.md
**Summary:** Compares two `.sfmr` reconstructions across alignment, intrinsics, poses, features, 3D points; rich `--strips*` family for patch-strip montages.
**Implementing code:** `_commands/compare.py::compare()` → `_compare.compare_reconstructions`.
**Inconsistencies:** Spec syntax block (L11-13) omits all `--strips*` options (section is broken out later but the syntax block undersells the surface).
**Recommendation:** update spec — fold `--strips*` into the syntax block or add a "see below" pointer.

### specs/cli/densify-command.md
**Summary:** Densifies a `.sfmr` via sweep-matching covisible image pairs, triangulation, BA, filtering.
**Implementing code:** `_commands/densify.py::densify()`.
**Inconsistencies:** Spec marks command experimental but code emits no startup warning. Spec's "Process" step 6 ("Align result back to original frame") not visible at Click layer; presumably in `densify_reconstruction`.
**Recommendation:** update code (surface the experimental warning) or spec (drop the claim). The "merge into xform" note is a design TODO, not a divergence today.

### specs/cli/epipolar-command.md
**Summary:** Epipolar visualization between image pairs (single or batch via `--pairs-dir`), with rectification, line/feature styling, sweep-matching overlay.
**Implementing code:** `_commands/epipolar.py::epipolar()` → `visualization._epipolar_display.draw_epipolar_visualization`.
**Inconsistencies:**
  - **Internal help-text inconsistency**: `--side-by-side` help says outputs are `_A`/`_B` suffixed (L132); `--draw` help and spec say `_other`.
  - **Adjacent-pairs batch mode emits one file per image (N files), not one per pair (N-1)** because of the trailing `last_left` extra (L310-334). Spec implies per-pair output.
**Recommendation:** update code (or spec) — reconcile `_A`/`_B` vs `_other`; clarify the adjacent-pairs output count.

### specs/cli/flow-command.md
**Summary:** DIS optical-flow visualization between two image files, optionally compared with `.sfmr` correspondences; supports `--pairs-dir` batch and side-by-side output.
**Implementing code:** `_commands/flow.py::flow()` → `visualization._flow_display.draw_flow_visualization`.
**Inconsistencies:**
  - **`--pairs-dir` is declared and bound but never used** in `flow.py` — no batch loop, the parameter is dropped on the floor. The spec example `sfm flow image_001.jpg image_002.jpg --pairs-dir flow_viz/ ...` (L58) would silently process only the single explicit pair.
**Recommendation:** update code — wire `--pairs-dir` through or remove the option. **Real bug.**

### specs/cli/from-colmap-bin-command.md
**Summary:** Imports COLMAP binary reconstruction into `.sfmr`. Options: `--image-dir`, `--output`, `--tool-name`, `--detect-infinity`.
**Inconsistencies:** None.
**Recommendation:** none.

### specs/cli/heatmap-command.md
**Summary:** Visualizes per-point reprojection error / track length / triangulation angle as colored circles. `--metric all` emits one image per metric.
**Implementing code:** `_commands/heatmap.py`.
**Inconsistencies:** None — all 5 options, choices, defaults (5, 0.7, `all`), and per-metric default colormaps match.
**Recommendation:** none.

### specs/cli/inspect-command.md
**Summary:** Suffix-dispatched inspector for `.sfmr`/`.sift`/`.matches`/`.camrig`/image with `--verbose`; resolves `pt3d_<hash>_<index>` by searching workspace `.sfmr` files in spec order.
**Inconsistencies:** Minor — spec names the 2nd arg `WORKSPACE`; code names it `location`. Behavior matches.
**Recommendation:** none (or rename Click arg purely for cosmetics).

### specs/cli/insv2rig-command.md
**Summary:** Splits Insta360 `.insv` dual-fisheye into `fisheye_left/`/`fisheye_right/` and writes a two-sensor `.camrig`.
**Implementing code:** `_commands/insv2rig.py`.
**Inconsistencies:** Spec says rig type is `fisheye_360`; code passes `rig_name="insv2_x5"` (L97). "Rig type" vs "rig name" not clarified.
**Recommendation:** discuss — clarify the type/name distinction; verify `fisheye_360` is actually emitted as the rig type.

### specs/cli/match-command.md
**Summary:** Feature matcher with four mutually exclusive modes (`--exhaustive`, `--sequential`, `--flow`, `--cluster`) plus `--merge`; method-specific options rejected if paired with wrong method.
**Implementing code:** `_commands/match.py::match()`, `_reject_stray_mode_options()`.
**Inconsistencies:** No `--camera-config PATH` flag exists — closest-ancestor `camera_config.json` resolution is automatic via `CameraConfigResolver`. Matches `specs/workspace/camera-config.md`.
**Recommendation:** none.

### specs/cli/merge-command.md
**Summary:** Merges ≥2 aligned `.sfmr` files via dedup, union-find correspondence, percentile-filtered point merging, PnP+RANSAC pose refinement.
**Implementing code:** `_commands/merge.py::merge()`.
**Inconsistencies:** None at the CLI layer.
**Recommendation:** none.

### specs/cli/motion-command.md
**Summary:** Motion-discontinuity analysis over image sequence or `.sfmr` reconstruction; emits human-readable + optional JSON.
**Inconsistencies:** All seven options + defaults match.
**Recommendation:** none.

### specs/cli/pano2rig-command.md
**Summary:** Equirect → 6-face cubemap, writes `cubemap.camrig`. Output dir must be inside a workspace.
**Inconsistencies:** `--jpeg-quality` documented as 1-100 but Click doesn't bound-check at CLI layer; relies on encoder.
**Recommendation:** none (or add `IntRange(1,100)` for friendlier errors).

### specs/cli/panorama-command.md
**Summary:** Renders an equirect panorama from a posed `.sfmr` using a spherical-tile rig; rich source subsetting and per-tile photometric RANSAC.
**Inconsistencies:** None — all 15 documented options + defaults match exactly.
**Recommendation:** none.

### specs/cli/render-patches-command.md
**Summary:** Projects oriented per-point patches onto source images (texture/normal/flat/wire fill modes).
**Inconsistencies:** None at the CLI layer.
**Recommendation:** none.

### specs/cli/scale-by-measurements-command.md
**Summary:** Documents `--scale-by-measurements` as an `xform` option: YAML schema, point-ID resolution, median-of-per-measurement scale.
**Implementing code:** `_commands/xform.py` (`--scale-by-measurements` declared with `multiple=True` ~L93).
**Inconsistencies:** **Click is `multiple=True` but spec describes a single-file invocation.** Repeating `--scale-by-measurements a.yaml --scale-by-measurements b.yaml` is allowed by the CLI but the spec doesn't describe combined-measurements vs sequential-transforms vs last-wins semantics.
**Recommendation:** discuss — clarify intended repeat semantics in spec, or change Click to single.

### specs/cli/sift-command.md
**Summary:** `sfm sift` for extracting/drawing SIFT features; workspace-driven tool selection with overrides for tool, DSP, threading, range.
**Inconsistencies:** `--dsp` help text says "default: disabled" but actual default is `None` (workspace-driven). `--filter-sfm` help says ".sfm reconstruction file" — wording drift from `.sfmr`.
**Recommendation:** update code — fix `--dsp` help string and `--filter-sfm` wording.
**Suspicious:** `sift.py` imports `deadline.job_attachments.api.summarize_path_list` (AWS Deadline) for a path-summary helper — heavy/odd dep; worth checking if it's actually used.

### specs/cli/solve-command.md
**Summary:** Incremental (COLMAP) or global (GLOMAP) SfM from images or `.matches`; rig refinement, sequential overlap, camera model, range, points-at-infinity detection, `.camrig`/`camera_config.json` integration.
**Implementing code:** `_commands/solve.py::solve()`.
**Inconsistencies:**
  - **Camera-config-vs-`--camera-model` rejection only fires on the image-paths branch** (L262); the `.matches` branch (L194-238) does not invoke `_check_camera_model_conflict`. Per AGENTS.md and the spec's "Camera Intrinsics" section, the rule should apply to any image being processed.
**Recommendation:** update code — extend `_check_camera_model_conflict` to the `.matches` branch.

### specs/cli/to-colmap-bin-command.md
**Summary:** Exports `.sfmr` → COLMAP binary with `--range`/`--filter-points` subsetting.
**Inconsistencies:** None.
**Recommendation:** none.

### specs/cli/to-colmap-db-command.md
**Summary:** Builds a COLMAP `.db` from either `.sfmr` or `.matches`; positional `OUTPUT_DB_PATH`.
**Inconsistencies:**
  - **Camera-config rejection invisible at CLI shim** — spec promises rejection (L41-46), but the CLI calls `_setup_for_sfm_from_matches` without an explicit `_check_camera_model_conflict`. Unverified whether `db_setup._setup_for_sfm_from_matches` enforces it internally.
  - **Dead code:** `_from_sfmr` (L86-105) calls `find_workspace_for_path` + `load_workspace_config` and discards the result.
  - No `--range` here despite `to-colmap-bin` and `to-nerfstudio` having one — cross-exporter inconsistency.
**Recommendation:** discuss — verify (or add) the camera-config rejection; clean up the dead workspace load.

### specs/cli/to-nerfstudio-command.md
**Summary:** `.sfmr` → Nerfstudio dataset (`transforms.json`, `sparse_pc.ply`, etc.).
**Inconsistencies:** None at CLI layer.
**Recommendation:** none.

### specs/cli/undistort-command.md
**Summary:** Converts every image in a `.sfmr` to pinhole, writing a self-contained workspace.
**Inconsistencies:** None at CLI layer.
**Recommendation:** none.

### specs/cli/ws-init-command.md
**Summary:** Writes `.sfm-workspace.json`. Documents feature-tool choices, COLMAP knobs, conflict rules, `--force`.
**Inconsistencies:**
  - **`--max-features` scope drift:** spec table (L23) says "COLMAP only"; code allows `colmap` *and* `sfmtool` (L90-94), and the option help text agrees. Spec is out of date.
  - **`--max-features` default:** spec says `8192`, Click defaults to `None`; the `8192` fallback is applied inside `init_workspace`. Behaviorally consistent.
**Recommendation:** update spec — change the `--max-features` row to "COLMAP and sfmtool".

### specs/cli/xform-command.md
**Summary:** Top-level overview of the `sfm xform` ordered pipeline.
**Inconsistencies:**
  - **No densify discussion** — the densify-vs-xform overlap is only mentioned in `densify-command.md`'s NOTE.
  - Spec's `--include-by-distribution` summary entry omits the `[,verbose]` modifier that exists in code.
**Recommendation:** update spec — add a brief note on the densify relationship and the `[,verbose]` modifier.

### specs/cli/xform-find-points-at-infinity.md
**Summary:** `--find-points-at-infinity ...` + `--classify-points-at-infinity ...` + global `--max-features <N>` cap.
**Inconsistencies:** None significant.
**Recommendation:** none.

### specs/cli/xform-refine-normals-command.md
**Summary:** `--refine-normals` photometric per-point normal refinement.
**Inconsistencies:** None — all 19+ documented keys appear in `_REFINE_NORMALS_KEYS`.
**Recommendation:** none.

### specs/cli/xform-select-by-distribution-command.md
**Summary:** `--include-by-distribution <COUNT>[,verbose]` greedy farthest-point + angular-thinning selection.
**Inconsistencies:** None at the CLI/parser surface.
**Recommendation:** none.

---

## Core Algorithm Specs

> **Path drift dominates this section.** 14 of 16 core specs cite
> pre-regroup flat paths. The lib.rs facade does re-export the headline
> types so symbol names still resolve, but every absolute file path needs
> the topic-group prefix. Findings call out only the *additional* issues
> on top of path drift; the path-drift line is implicit unless noted
> otherwise.

### specs/core/batch-triangulation-api.md
**Summary:** Batch triangulation API returning per-track point + observability diagnostics. Phases 1–4 marked done.
**Implementing code:** `reconstruction/triangulation.rs`, `analysis/infinity/{discover,convert}.rs`, `geometry/viewing_angle.rs`, `features/feature_match/geometric_filter.rs`, `sfmtool-py/src/py_triangulation.rs`.
**Inconsistencies:** Heavy path drift (`triangulation.rs` → `reconstruction/triangulation.rs`; `viewing_angle.rs:28` → `geometry/viewing_angle.rs:28`; `geometric_filter.rs:252` → `features/feature_match/geometric_filter.rs:252`; `infinity/{discover,convert}.rs` → `analysis/infinity/...`). Line numbers (`:28`, `:330`, `:275`, `:50`, `:78`, `:252`, `:375`) likely stale post-regroup; verify or drop.
**Recommendation:** update spec — repath; Status accurate.

### specs/core/epipolar-curves.md
**Summary:** `plot_epipolar_curve` / `plot_epipolar_curves_batch` — model-agnostic epipolar polylines. Marked Implemented.
**Implementing code:** `camera/epipolar.rs`, `sfmtool-py/src/py_epipolar.rs`, `src/sfmtool/visualization/_epipolar_display.py`.
**Inconsistencies:** Paths `epipolar.rs` (L3, L38, L58) → `camera/epipolar.rs`; `rectification.rs` (L299) → `camera/rectification.rs`.
**Recommendation:** update spec — path-only edits.

### specs/core/flow-based-matching.md
**Summary:** Sliding-window flow-based feature matching. Marked Implemented.
**Implementing code:** `src/sfmtool/feature_match/_flow_matching.py`; Rust at `features/optical_flow/`.
**Inconsistencies:** None — spec only references the Python module by relative name.
**Recommendation:** none.

### specs/core/fronto-parallel-patch-cache.md
**Summary:** Fronto-parallel patch cache for normal refinement. `CacheMode::FrontoParallel` is the default with `cache_supersample = 2`.
**Implementing code:** `patch/normal_refine/fronto_cache.rs`, `patch/normal_refine.rs`.
**Inconsistencies:** L13 cites `sfmtool-core/src/patch_normal_refine/fronto_cache.rs` → `patch/normal_refine/fronto_cache.rs`. L18 says "(`coarse_to_fine` in `patch_normal_refine.rs`)" → `patch/normal_refine.rs`. L167 Phase-1 status block drifts same way.
**Recommendation:** update spec — path-only.

### specs/core/gpu-optical-flow.md
**Summary:** GPU compute-shader pipeline for DIS optical flow with hybrid CPU/GPU per-level routing and persistent buffer pools.
**Implementing code:** `features/optical_flow/gpu/`.
**Inconsistencies:** L318-319 references `optical_flow/` and `optical_flow/gpu/` → `features/optical_flow/...`. **No explicit Status block** — should add one for parity with peer specs.
**Recommendation:** update spec — repath, add Status line.

### specs/core/image-warping.md
**Summary:** WarpMap (`from_cameras`, pose-aware constructors), ImageU8 / ImageU8Pyramid, `remap_bilinear` / `remap_aniso`, `ray_to_pixel`, Equirectangular model. Marked Implemented.
**Implementing code:** `camera/warp_map.rs`, `camera/remap.rs`, `camera/distortion.rs`, `sfmtool-py/src/py_warp_map.rs`.
**Inconsistencies:** L3-7 Status; L494-501 Module Organization layout box; L509-510 "Add ray_to_pixel..." all cite flat paths.
**Recommendation:** update spec — paths only.

### specs/core/optical-flow.md
**Summary:** Pure-Rust DIS optical-flow implementation; SSE2 + rayon + GPU.
**Implementing code:** `features/optical_flow/{mod,dis,pyramid,variational,interp}.rs`, `features/optical_flow/gpu/`.
**Inconsistencies:** Module-tree diagram (L119-139) rooted at `sfmtool-core/src/optical_flow/` → `features/optical_flow/`. No formal Status block.
**Recommendation:** update spec — repath the module tree and add a Status line.

### specs/core/patch-cloud.md
**Summary:** OrientedPatch + PatchCloud + WarpMap::from_patch. Status block (L204) says Implemented as of 2026-06-11.
**Implementing code:** `patch/cloud.rs`, `camera/warp_map.rs`, `sfmtool-py/src/py_patch_cloud.rs`.
**Inconsistencies:** Status block (L204) cites `patch_cloud.rs` → `patch/cloud.rs`.
**Recommendation:** update spec — one-line path fix.

### specs/core/patch-normal-refinement.md
**Summary:** Photometric refinement of patch normals; status blocks at L7, L20, L31 mark v1 implemented.
**Implementing code:** `patch/normal_refine.rs`, `crates/sfmtool-py/src/py_patch_cloud.rs`.
**Inconsistencies:** **None** — this is the one core spec that already uses the post-regroup layout. L7 Status cites `sfmtool-core/src/patch/normal_refine.rs` (correct); L11 names `view_indices_from_reconstruction` (matches the rename in the code).
**Recommendation:** none.

### specs/core/per-spherical-tile-source-stack.md
**Summary:** Per-(tile, source) CSR-packed image-pyramid stack. Implemented.
**Implementing code:** `spherical/per_tile_source_stack.rs`, `sfmtool-py/src/py_per_spherical_tile_source_stack.rs`.
**Inconsistencies:** L4 Status cites `crates/sfmtool-core/src/per_spherical_tile_source_stack.rs` → `spherical/per_tile_source_stack.rs`.
**Recommendation:** update spec — one-line path fix.

### specs/core/photometric-subsets-ransac.md
**Summary:** Per-tile RANSAC clustering over the source stack. Status: Implemented; previously mislabeled Draft. Promotion already recorded in the spec.
**Implementing code:** `spherical/photometric_ransac.rs`, `sfmtool-py/src/py_photometric_ransac.rs`.
**Inconsistencies:** L3 Status cites `photometric_ransac.rs` → `spherical/photometric_ransac.rs`. L7 cites `consensus_atlas.rs` → `spherical/consensus_atlas.rs`.
**Recommendation:** update spec — path-only.

### specs/core/randomized-kdtree-forest.md
**Summary:** Multi-randomized kd-tree forest for descriptor ANN search. Status: Implemented (Phase 1).
**Implementing code:** `features/kdforest/`, `sfmtool-py/src/py_kdforest.rs`, `sfmtool-core/benches/kdtree_forest.rs`.
**Inconsistencies:** Status block (L4) and module-tree diagram (L272) cite `kdforest/` → `features/kdforest/`. **Status carries the stale "matcher CLI integration left as a follow-up" caveat** — the cluster matcher has been built on top of KdForest (see `track-cluster-matching.md`'s "production matcher is done").
**Recommendation:** update spec — repath + retire the stale follow-up note.

### specs/core/sift.md
**Summary:** Pure-Rust SIFT detector + descriptor. Status: Implemented; on-disk incremental extension still future.
**Implementing code:** `features/sift/`, `sfmtool-py/src/py_sift.rs`, `py_sift_io.rs`.
**Inconsistencies:** L4 Status, L463, L686 cite `sfmtool-core/src/sift/` → `features/sift/`.
**Recommendation:** update spec — repath.

### specs/core/spherical-tiles-rig.md
**Summary:** SphericalTileRig discretization. Status: Implemented.
**Implementing code:** `spherical/tile_rig.rs`, `spherical/tile_rig/camrig.rs`, `spherical/sphere_points.rs`, `sfmtool-py/src/py_spherical_tile_rig.rs`.
**Inconsistencies:** L4 Status and L69 cite `spherical_tile_rig.rs` and `sphere_points.rs` → `spherical/tile_rig.rs` and `spherical/sphere_points.rs`.
**Recommendation:** update spec — repath.

### specs/core/tile-batched-consensus-atlas.md
**Summary:** Bounded-memory tile-batched orchestrator. Status: Implemented.
**Implementing code:** `spherical/consensus_atlas.rs`, `sfmtool-py/src/py_consensus_atlas.rs`.
**Inconsistencies:** L7 cites `crates/sfmtool-core/src/consensus_atlas.rs` → `spherical/consensus_atlas.rs`. L195-199 cite the flat names for the three sibling files.
**Recommendation:** update spec — repath.

### specs/core/track-cluster-matching.md
**Summary:** Background-floor track-cluster matcher. Status: "production form done".
**Implementing code:** `features/cluster_match/`, `sfmtool-py/src/py_cluster_match.rs`, `src/sfmtool/feature_match/_cluster_matching.py`, CLI `sfm match --cluster`.
**Inconsistencies:** L395, L475, L669 cite `cluster_match/` → `features/cluster_match/`. L179 references `kdforest/` → `features/kdforest/`.
**Recommendation:** update spec — repath.

---

## Format Specs

### specs/formats/sfmr-file-format.md
**Summary:** Specifies `.sfmr` v3 as the current version (ZIP+zstd, columnar, XXH128). Covers metadata, content hashes, cameras, optional rigs/frames, images, points3d, tracks; includes a Versioning section ending at v1→v2→v3.
**Implementing code:** `crates/sfmr-format/src/{types,read,write,verify}.rs`. Writer always emits version 4; reader accepts versions 1-4.
**Inconsistencies:**
  - **Version drift — major.** Spec headlines "version 3" everywhere. Code writes v4 on every write (`write.rs:110`, `read.rs:475`) and accepts v1-v4 (`read.rs:57`). The v4 changes from `sfmr-v4-patch-keypoints.md` (Stages 1+2 landed: `feature_source`, `keypoints_xy`, `image_file_hashes`, `has_feature_indexes`, `has_keypoints_xy`) are entirely missing from this spec — the only authoritative description lives in the still-marked-Draft v4 patch doc.
  - Missing `feature_source` field in top-level metadata description; missing `tracks/metadata.json` keys `has_feature_indexes`/`has_keypoints_xy`; missing `images/image_file_hashes.{N}.uint128.zst` and `tracks/keypoints_xy.{M}.2.float32.zst` archive entries.
  - `tool_options` typing claim ("use empty object `{}` if none"): code types it as `HashMap<String, serde_json::Value>` without `#[serde(default)]`, so an absent key fails to deserialize. Worth flagging that "use empty object" is enforced as "must be present".
  - `EQUIRECTANGULAR` camera-model omitted from the spec's camera-model table (used by camrig).
**Recommendation:** **Update spec.** Fold `sfmr-v4-patch-keypoints.md` into this file per the patch doc's "Stage 5" plan: bump headlined version to 4, add `feature_source` to the top-level metadata, add the new archive entries, document the `has_feature_indexes`/`has_keypoints_xy` keys, add a v3→v4 migration table and v4 history bullet. Code is already a strict superset of the spec.

### specs/formats/sfmr-v4-patch-keypoints.md
**Summary:** Draft spec for `.sfmr` v4 introducing `feature_source ∈ {sift_files, embedded_patches}` plus per-mode columns.
**Implementing code:** `crates/sfmr-format/` — Stages 1+2 landed; PyO3 surface in `sfmtool-py/`.
**Inconsistencies:**
  - **Implementation-status banner is stale.** Says round-trip / cross-hash tests still pending; they're in (`test_embedded_patches_round_trip`, `test_write_rejects_contradictory_columns`, `test_embedded_keypoints_validated_on_read_and_verify`, `test_keypoints_are_folded_into_tracks_hash`, `test_embedded_sort_reorders_keypoints_in_lockstep`). What actually remains is the fold-in (Stage 5) and the Python producer command (`sfm patches-to-keypoints`).
  - Cross-language hash claim not verified by the current test set; only Rust round-trip is exercised.
**Recommendation:** **Update spec and fold in.** Drop the "stages pending" banner; fold this file into `sfmr-file-format.md` per its own Stage 5 plan.
**Suspicious:** File-level claim "exactly one of `{feature_indexes, keypoints_xy}` is present" is enforced on write but on read a malformed `embedded_patches` file with a stray `feature_indexes` is silently tolerated. Worth tightening.

### specs/formats/sift-file-format.md
**Summary:** Specifies `.sift` v1 (current) and v2 (Draft) — ZIP+zstd columnar features.
**Implementing code:** `crates/sift-format/src/`. Writer always emits `version: 1`.
**Inconsistencies:**
  - **v2 is correctly flagged "not yet implemented"** in the spec.
  - **No version validation on read.** `read.rs` does not check that `metadata.version` is recognized — a future v2 file would deserialize as v1 with wrong hash semantics. The corresponding `.sfmr` reader rejects unsupported versions; sift reader does not.
**Recommendation:** No spec change needed today; consider adding a version-bound check in `sift-format/src/read.rs`.
**Suspicious:** Spec uses "Version 1.0rc1" in Version History but the code/JSON value is the integer `1`. Consider aligning.

### specs/formats/matches-file-format.md
**Summary:** `.matches` v1 — candidate matches + optional two-view geometries.
**Implementing code:** `crates/matches-format/src/`.
**Inconsistencies:** None — field-by-field correspondence is clean (top-level metadata, content-hash JSON shape, `TvgMetadata`, `TwoViewGeometryConfig` enum's ten string values all match). dtypes match. **WorkspaceMetadata duplication** between `matches-format` and `sfmr-format` is a code-organization smell, not a spec inconsistency.
**Recommendation:** none.

### specs/formats/camrig-file-format.md
**Summary:** `.camrig` v1 — single camera rig in ZIP+zstd.
**Implementing code:** `crates/camrig-format/src/`.
**Inconsistencies:** None of substance. Code-side additions over spec (`1e-6` quaternion tolerance, positive width/height check) are reasonable defaults the spec deliberately left unspecified.
**Recommendation:** none.
**Suspicious:** `rig_attributes` is `serde_json::Value` so no per-`rig_type` schema enforcement — worth a sentence in the spec acknowledging this is producer-honor-system.

---

## GUI Specs

### specs/gui/blender-viewport-navigation-implementation-overview.md
**Summary:** Reference document on Blender's Windows DirectManipulation implementation. Pure background research.
**Inconsistencies:** None (informational, about Blender, not sfmtool).
**Recommendation:** none.

### specs/gui/gui-adaptive-clip-and-grid.md
**Summary:** Reversed-Z infinite far projection with adaptive near plane and `length_scale`-driven adaptive ground grid. Marked "Implemented".
**Implementing code:** `viewer_3d/camera.rs`, `viewer_3d/overlay.rs::draw_grid`, `scene_renderer/auto_point_size.rs`.
**Inconsistencies:** None — spec accurately matches code.
**Recommendation:** none.

### specs/gui/gui-architecture.md
**Summary:** Documents the technology stack, crate structure, rendering pipeline, platform-specific details.
**Implementing code:** Workspace `Cargo.toml`, `crates/sfm-explorer/src/`.
**Inconsistencies:**
  - **Crate-structure tree is stale.** Spec shows `main.rs` and `viewer_3d.rs` as flat files; actually `main.rs` is a 6-line shim, the real entry point and event-loop live in `lib.rs` (399 lines) and `app.rs` (633 lines), and `viewer_3d` is now a directory (`mod.rs`/`camera.rs`/`input.rs`/`overlay.rs`). Spec also does not mention `dock.rs` or `point_track_detail.rs`, both shipped panels.
  - **Shader list is stale.** Spec lists `bg_image.wgsl` as the pinhole BG shader. The file no longer exists; only `bg_image_distorted.wgsl` does.
  - **Module Responsibilities table** attributes work to `main.rs` that now lives in `lib.rs::run` and `app.rs::run_ui_and_paint`; missing rows for `dock.rs`, `point_track_detail.rs`, `viewer_3d/{camera,input,overlay}.rs`.
**Recommendation:** **Update spec.** This is the most out-of-date doc in the directory.

### specs/gui/gui-camera-views.md
**Summary:** Frustum wireframes, image quads, pick buffer, distorted/fisheye rendering, camera view mode with free-look. 1727 lines.
**Implementing code:** `scene_renderer/*`, `shaders/*`, `viewer_3d/mod.rs::CameraViewMode`, `viewer_3d/camera.rs::best_fit_fov`, `viewer_3d/input.rs`.
**Inconsistencies:** Step 9 (persistent camera view + free-look) is implemented but the section header lacks the "— DONE" marker the other 8 steps carry. §"Implementation tasks" subsection still reads in imperative form.
**Recommendation:** update spec — mark Step 9 DONE and retire/checkmark the imperative tasks.

### specs/gui/gui-cross-panel-hover.md
**Summary:** Cross-panel hover state (`hovered_image`, `hovered_point`). Marked Implemented.
**Inconsistencies:** None.
**Recommendation:** none.

### specs/gui/gui-image-animation.md
**Summary:** Image animation playback (Space toggle, arrows, `[`/`]` fps, minibar play button). Marked Implemented.
**Inconsistencies:** None.
**Recommendation:** none.

### specs/gui/gui-multi-panel-image-browser.md
**Summary:** The egui_dock 3-panel layout, point selection, track-image highlighting, image-detail feature overlays, 2D pan/zoom, navigation minibar.
**Implementing code:** `lib.rs` (4-tab dock init), `dock.rs`, `image_browser.rs`, `image_detail.rs`, `point_track_detail.rs`.
**Inconsistencies:**
  - **Tab model lists three tabs (`Viewer3D`, `ImageBrowser`, `ImageDetail`); code has four** (`Tab::PointTrackDetail` added at `lib.rs:119`).
  - §"Hover State" pseudocode names the field `hovered_point_index`; actual field is `hovered_point` (state.rs:118).
  - §"Feature Overlays / Overlay modes" table lists 5 modes; code has **7** (state.rs:17 adds `DepthReliability` and `ConditionNumber`). The two new modes aren't documented in any spec.
**Recommendation:** **Update spec.** Add `Tab::PointTrackDetail`, rename `hovered_point_index` → `hovered_point`, document the two new overlay modes.

### specs/gui/gui-plan.md
**Summary:** Roadmap index with implemented-features list and a "Next Steps" ordered list.
**Inconsistencies:** "Image Browser / Overlay modes (features/tracks/reproj/triangulation/epipolar)" — overlays moved to Image Detail per `gui-multi-panel-image-browser.md`. Minor browser-overlay note should be updated.
**Recommendation:** minor spec update.

### specs/gui/gui-point-cloud-rendering.md
**Summary:** Point splats, EDL, target indicator (rotating compass), supernova lighting, points-at-infinity. Mostly DONE.
**Inconsistencies:**
  - §"Remaining UI work" lists the **infinity point-size slider as TODO**; `app.rs:384-389` already ships it (range 1.0-16.0 px, default 3.0 at `state.rs:205`).
**Recommendation:** minor spec update — mark the slider DONE.

### specs/gui/gui-point-track-detail.md
**Summary:** Dedicated panel showing per-observation reprojection error / ray angle / thumbnails. Marked Implemented.
**Inconsistencies:** None.
**Recommendation:** none.

### specs/gui/gui-user-experience.md
**Summary:** Vision and design principles.
**Inconsistencies:**
  - §"Multi-Panel Layout" still lists three panels (no Point Track Detail).
  - §"Future Directions" still lists "Animation playback" — already shipped per `gui-image-animation.md`.
**Recommendation:** minor spec update.

### specs/gui/gui-viewport-navigation.md
**Summary:** Most detailed navigation spec — orbit, pan, zoom, fly, target control, FOV zoom, Windows DM. Explicit `[x]`/`[ ]` checklists.
**Inconsistencies:** None — every `[ ]` item (mouse-drag tilt, free-nav FOV zoom binding, configurable sensitivity, zoom-to-cursor, inertial scrolling, save/restore camera positions) verified genuinely unimplemented.
**Recommendation:** none. **The best-maintained spec in the directory.**

---

## Workspace Specs

### specs/workspace/camera-config.md
**Summary:** `camera_config.json` format for per-directory camera intrinsics, closest-ancestor resolution, presence-based interaction with `--camera-model`.
**Implementing code:** `src/sfmtool/camera/config.py`, `src/sfmtool/camera/setup.py`, consumed by `_commands/solve.py` and `_commands/match.py`.
**Inconsistencies:**
  - Spec lists four commands that consult `camera_config.json` ("`sfm solve`, `sfm match`, `sfm to-colmap-db`, `sfm densify`"). **Only `solve.py` and `match.py` wire the resolver in.** Either the wiring is missing in two of the four commands, or the spec is overclaiming.
  - The "Open Question: Comparing Calibrations Side by Side" — `--camera-config PATH` flag undecided. Still genuinely open in the code.
**Recommendation:** discuss + update — confirm whether `to-colmap-db` and `densify` should honor `camera_config.json`.
**Suspicious:** Workflow 1 says `.camrig` takes precedence over `camera_config.json` when `sfm solve` discovers both; no `camrig` reference in `_commands/solve.py`.

### specs/workspace/rig-config.md
**Summary:** `rig_config.json` is COLMAP's `rig_configurator` format verbatim.
**Implementing code:** `src/sfmtool/rig/config.py`, `src/sfmtool/rig/frames.py`.
**Inconsistencies:** Spec is silent on rig-config location, but `_load_rig_config` only reads `{workspace_dir}/rig_config.json` — it does **not** apply the closest-ancestor walk that `camera_config.json` uses. Meaningful asymmetry worth flagging.
**Recommendation:** update spec — add a "workspace-root-only" sentence.

### specs/workspace/workspace.md
**Summary:** Workspace concept, `.sfm-workspace.json` schema, directory layout, hash-based feature cache, workspace discovery, dual relative/absolute path resolution.
**Implementing code:** `_commands/ws.py`, `_workspace.py`, `sift/file.py`.
**Inconsistencies:**
  - **Example `.sfm-workspace.json` omits `use_gpu`** from the COLMAP `feature_options` block. The sibling `ws-init-command.md` documents `use_gpu` as part of `feature_options` (excluded from the hash); the top-level workspace spec doesn't mention it.
  - `feature_tool` listed as `"colmap" | "opencv" | "sfmtool"` but the spec doesn't acknowledge per-tool option shapes for `opencv`/`sfmtool` (`contrast_threshold`, `octave_layers`, `max_num_features`).
  - No cross-reference between "Camera Intrinsics" section and `rig_config.json`.
**Recommendation:** update spec — add `use_gpu` to the example, add per-tool option shape note, cross-reference `rig-config.md`.

---

## Docs

### docs/index.md
**Summary:** Zensical home page. Project pitch + end-to-end example.
**Inconsistencies:** Captured `sfm ws init` transcript omits `use_gpu: True` (now printed by `ws.py:146-154` for COLMAP).
**Recommendation:** update docs — refresh the captured output.
**Suspicious:** "Installation: Coming soon..." conflicts with `getting-started.md`'s `pip install sfmtool`.

### docs/tutorials/getting-started.md
**Summary:** End-to-end tutorial using the `dino_dog_toy` dataset.
**Inconsistencies:** Same `use_gpu` transcript drift. `pip install sfmtool` at step 1 conflicts with `index.md`'s "Coming soon..."
**Recommendation:** update docs — refresh transcript and reconcile install instructions across the two pages.
**Suspicious:** Captured `sfm inspect -v` output shows `Feature tool: unknown` for a GLOMAP recon — worth a sanity check.

---

## Code without specs

### src/sfmtool/_commands/explorer.py
**What it does:** 29-line wrapper that launches the `launch-sfm-explorer` binary (the Rust GUI from `crates/sfm-explorer`) with an optional `.sfmr` argument.
**Why it matters:** user-facing CLI entry, but a pure shim — the GUI itself is documented under `specs/gui/`.
**Recommendation:** **acceptable as unspecced.** If anything, add a single bullet to `specs/gui/gui-architecture.md` noting that `sfm explorer` is the Python-side launcher.

### crates/sfmtool-core/src/analysis/alignment/{kabsch,ransac}.rs
**What it does:** Kabsch + RANSAC point-cloud alignment used by `sfm align`. Substantial code.
**Why it matters:** load-bearing for the `align` CLI; corresponds to a user-facing operation.
**Recommendation:** **write a spec at `specs/core/reconstruction-alignment.md`** (or expand `specs/cli/align-command.md` with an algorithm section). Currently neither spec describes the Kabsch/RANSAC method or its parameters.

### crates/sfmtool-core/src/analysis/image_pair_graph.rs (392 lines)
**What it does:** Builds the image-pair / covisibility graph used by `sfm analyze --coviz` and pair-graph visualizations.
**Why it matters:** internal-but-load-bearing for analyses surfaced in the CLI.
**Recommendation:** **write a short spec at `specs/core/image-pair-graph.md`** (or fold a section into `specs/cli/analyze-command.md`).

### crates/sfmtool-core/src/analysis/point_inspect.rs (175 lines)
**What it does:** Per-point inspection metrics used by `sfm inspect` and `sfm analyze`.
**Why it matters:** drives most of the per-point output the user sees.
**Recommendation:** add a note to the existing `specs/cli/inspect-command.md` / `specs/cli/analyze-command.md` describing what `point_inspect` computes.

### crates/sfmtool-core/src/camera/frustum.rs (443) and camera/viewport.rs (249)
**What it does:** Frustum construction + viewport/aspect handling used by visualization, frustum-cull culling in solve, and the GUI.
**Why it matters:** internal-but-load-bearing geometry shared between visualizations and the densify pipeline.
**Recommendation:** **acceptable as unspecced for now.** Add a one-paragraph mention to `specs/core/image-warping.md` or `specs/gui/gui-camera-views.md` cross-referencing where frustum geometry comes from.

### crates/sfmtool-core/src/geometry/* (se3, rigid, rot_quaternion, rotation, viewing_angle, transform)
**What it does:** Foundational geometry primitives — SE(3), rigid transforms, quaternions.
**Why it matters:** small utility library used everywhere; no algorithmic content beyond standard formulas.
**Recommendation:** **acceptable as unspecced.** Document only if/when the externalized PyO3 API stabilizes.

### crates/sfmtool-core/src/reconstruction/edit.rs (733 lines)
**What it does:** Large module of reconstruction-editing operations (filter, remove, transform points/cameras) backing `sfm xform`.
**Why it matters:** user-facing — drives a lot of `sfm xform` behavior.
**Recommendation:** add notes to `specs/cli/xform-command.md` and the sub-xform specs describing which Rust operations each invokes. **The single biggest unspecced crate module** — worth a brief inventory.

### crates/sfmtool-core/src/reconstruction/{filter.rs, point_correspondence.rs}
**What it does:** Generic reconstruction filtering primitives + point-to-point correspondence handling used in alignment/merge.
**Why it matters:** `point_correspondence.rs` (502 lines) is sizable and consumed by `sfm align`/`sfm merge`.
**Recommendation:** write a spec at `specs/core/point-correspondence.md`; fold `filter.rs` into existing reconstruction specs.

### crates/sfmtool-core/src/spatial.rs (264 lines)
**What it does:** Spatial index (kd-tree / grid) for nearest-neighbor queries.
**Why it matters:** internal utility.
**Recommendation:** **acceptable as unspecced.**

### crates/sfmtool-core/src/spherical/sphere_points.rs
**What it does:** Sphere-point sampling utilities used by the spherical tile/rig pipeline.
**Why it matters:** internal building block; `spherical-tiles-rig.md` covers the rig but doesn't drill into the sphere-points primitive.
**Recommendation:** add a short subsection to `specs/core/spherical-tiles-rig.md` describing the sampling scheme (Fibonacci? icosahedral? — worth pinning down).

### crates/sfmtool-py/ (PyO3 binding crate)
**What it does:** Compiles to `sfmtool._sfmtool`, exposes the Rust core to Python.
**Why it matters:** internal glue.
**Recommendation:** **acceptable as unspecced.**

### Two new `OverlayMode` variants in the GUI (state.rs:37,41)
**What it is:** `OverlayMode::DepthReliability` and `OverlayMode::ConditionNumber` appear in the runtime dropdown but **no spec describes them.** Neither `gui-multi-panel-image-browser.md` (overlay-mode table lists 5) nor `gui-camera-views.md` mentions them.
**Recommendation:** document in `gui-multi-panel-image-browser.md`'s overlay-mode section.

---

## Top priorities

1. **Fold `sfmr-v4-patch-keypoints.md` into `sfmr-file-format.md` and bump the headline version to 4.** The format spec is one version stale; the v4 changes (`feature_source`, `keypoints_xy`, `image_file_hashes`, the two new `tracks/metadata.json` keys, the two new archive entries) are described only in the still-marked-Draft v4 patch doc. This is the largest single spec/code divergence in the audit.

2. **Sweep post-regroup path drift across `specs/core/` (14 of 16 specs) and the three GUI specs that name files.** Mechanical, doc-only, low risk; restores grep-ability of every "implementing code" pointer.

3. **Fix the real CLI bugs:**
   - `flow.py` accepts `--pairs-dir` but ignores it (silent data-loss footgun).
   - `epipolar.py` internal `_A`/`_B` vs `_other` help-text inconsistency and N-vs-N-1 file count in adjacent-pairs mode.
   - `solve.py` does not enforce the camera-config-vs-`--camera-model` rejection on the `.matches` branch.
   - `to_colmap_db.py` camera-config rejection invisible at the CLI shim.
   - `ws-init-command.md`'s `--max-features` mis-scoped ("COLMAP only") when code allows COLMAP+sfmtool.

4. **Sync the three GUI specs that miss the 4th panel.** `gui-architecture.md`, `gui-multi-panel-image-browser.md`, and `gui-user-experience.md` still describe a 3-panel layout. `Tab::PointTrackDetail` shipped per `gui-point-track-detail.md`. Plus: the two new `OverlayMode` variants (`DepthReliability`, `ConditionNumber`) and the already-shipped infinity point-size slider need their status flipped from TODO to DONE in the relevant specs.

5. **Write three small specs for currently-undocumented but user-facing code.** `specs/core/reconstruction-alignment.md` for `analysis/alignment/{kabsch,ransac}.rs` (drives `sfm align`); a brief `specs/core/point-correspondence.md` for `reconstruction/point_correspondence.rs` (drives `sfm align`/`sfm merge`); an inventory note in `specs/cli/xform-command.md` pointing at `reconstruction/edit.rs` (drives most of `sfm xform`). These are the only substantial pieces of user-facing code with no spec coverage.
