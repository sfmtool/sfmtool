# sfmtool — Next Steps (2026-06-23)

A dated snapshot of recommended next work, drawn from the same-day
`audit-hygiene` and `audit-specs` rebaselines (`reports/2026-06-23-hygiene-audit.md`,
`reports/2026-06-23-spec-audit.md`) plus the open items still carried in
`reports/2026-06-13-perf-patch-normal-refinement.md` and
`reports/2026-05-22-next-steps.md`. Two lists: concrete implementation tasks
pulled from already-written specs and audit findings, then speculative
design topics worth a discussion before they earn specs.

This snapshot **supersedes `2026-05-22-next-steps.md`**. Of that report's
five implementation items, #1 (`sfm panorama` CLI) and #4 (render
points-at-infinity in SfM Explorer) shipped previously; #2 (FOV-zoom
gesture) is carried forward below as item 5; #3 (fold `densify` into
`xform`) and #5 (`--camera-config PATH` flag) remain open but are
deprioritized this round (covered in the design-discussion list below).

---

## ~5 implementation tasks (from existing specs and audit findings)

Ranked roughly by value × readiness — cheap, high-leverage wins first.

### 1. Land the CLI bug-fix bundle from the 2026-06-23 spec audit

- **Spec/audit reference:** `reports/2026-06-23-spec-audit.md` Top Priority #3,
  plus the per-command findings: `flow-command.md` (flow.py), `epipolar-command.md`
  (epipolar.py), `solve-command.md` (solve.py), `ws-init-command.md` (spec).
- **Current state:** Four concrete spec/code divergences with the diagnoses
  in hand — small one-shots, no design work needed.
- **Scope:**
  - `_commands/flow.py`: wire `--pairs-dir` through (today it's accepted,
    bound to a variable, and silently dropped on the floor — the spec's
    `sfm flow image_001.jpg image_002.jpg --pairs-dir flow_viz/` example
    processes only the explicit pair). Either implement the batch loop or
    remove the option.
  - `_commands/epipolar.py`: reconcile the `_A`/`_B` vs `_other` suffix
    drift between `--side-by-side` help (L132) and `--draw` help / spec.
    Adjacent-pairs batch mode writes N files (one per image) rather than
    N-1 (one per pair) — choose one and align spec + code.
  - `_commands/solve.py`: extend `_check_camera_model_conflict` to the
    `.matches` branch (L194-238). Today the check only fires on the
    image-paths branch (L262); AGENTS.md and the spec require the rule on
    every solve input.
  - `specs/cli/ws-init-command.md`: change the `--max-features` row from
    "COLMAP only" to "COLMAP and sfmtool" — `ws.py:90-94` and the option
    help string already say both.
- **Why now:** Real bugs (the flow one is a silent data-loss footgun), all
  small, no design questions, and they're already diagnosed in the audit so
  the implementation effort is just typing.

> _Status (2026-07-06): Done — flow's `--pairs-dir` was removed (spec points
> at `sfm epipolar --pairs-dir`), epipolar's help/spec now agree on `_other`
> and one-output-per-image, the solve `.matches`-branch check was refuted
> (enforced inside `_setup_for_sfm_from_matches`), and the ws-init spec row
> reads "COLMAP and sfmtool". See the annotations in
> `reports/2026-06-23-spec-audit.md`._

### 2. Quick perf wins on `PatchCloud.refine_normals`

- **Spec reference:** `reports/2026-06-13-perf-patch-normal-refinement.md`
  opportunities #6 (`compute_confidence: bool`) and #7 (subset-aware
  pyramid build). The report measured both and quantified the wins.
- **Current state:** Neither has landed. Confidence is computed
  unconditionally (~9% of total time, a fixed 9-eval stencil per patch);
  the binding builds full pyramids for every image even when `point_ids`
  selects a handful of points.
- **Scope:**
  - Add `compute_confidence: bool` (default true) to
    `refine_patch_cloud_normals` / `PatchCloud.refine_normals`. Skip the stencil
    when the caller doesn't need it (parameter sweeps, quick passes). The
    report projects ~1.1× and there are no design questions.
  - Subset-aware pyramid build in the PyO3 binding: build pyramids only
    for images referenced by the selected `point_ids`. Fixed per-call
    overhead today is **1.45 s** on dino (full pyramid build) — vanishes
    for small-subset calls. No effect on full-cloud runs. Small
    Python-visible change.
- **Why now:** Both are small, no design surface, and the second one
  unblocks interactive use on dino-scale data (every quick `refine_normals`
  on a handful of points pays 1.5 s of pure overhead today).

### 3. Fold `sfmr-v4-patch-keypoints.md` into `sfmr-file-format.md`

- **Spec reference:** `specs/formats/sfmr-v4-patch-keypoints.md` Stage 5 of
  its own plan; `reports/2026-06-23-spec-audit.md` Top Priority #1.
- **Current state:** Stages 1 (Rust `sfmr-format`) and 2 (PyO3 bindings +
  in-memory `SfmrReconstruction` support) are landed and tested — `sfmr-format`
  has `test_embedded_patches_round_trip`,
  `test_write_rejects_contradictory_columns`,
  `test_embedded_keypoints_validated_on_read_and_verify`,
  `test_keypoints_are_folded_into_tracks_hash`,
  `test_embedded_sort_reorders_keypoints_in_lockstep`. The fold-in (Stage
  5) and the Python producer command (Stage 3: `sfm patches-to-keypoints`)
  remain. The fold-in is independent of Stage 3.
- **Scope:** Spec edits only. Bump `sfmr-file-format.md` from headline v3
  to v4; add `feature_source` to the top-level metadata field list; add
  the `images/image_file_hashes.{N}.uint128.zst` and
  `tracks/keypoints_xy.{M}.2.float32.zst` archive entries; document the
  `has_feature_indexes` / `has_keypoints_xy` keys in
  `tracks/metadata.json`; add a v3 → v4 migration table and Version
  History bullet. Then delete `sfmr-v4-patch-keypoints.md` (or trim it to
  just the still-pending Stage 3 plan).
- **Why now:** Largest single spec/code divergence in the 2026-06-23
  audit. Every `.sfmr` file written today is v4 but the spec describes v3.
  Spec-only edit, but high readability value.

> _Status (2026-06-30): Done — folded in by commit 6e13efc (PR #117, landed
> the same afternoon this snapshot was taken). `sfmr-v4-patch-keypoints.md` is
> deleted; `sfmr-file-format.md` now carries the `feature_source` discriminator,
> the `images/image_file_hashes` + `tracks/keypoints_xy` archive entries (marked
> "version 4+"), the `has_feature_indexes` / `has_keypoints_xy` keys, a v3 → v4
> migration table, and a Version 4 history entry. The Stage 3 Python producer
> (`sfm patches-to-keypoints` / `embed-patches`) also shipped (PR #117 et al.),
> so this whole item is closed._

### 4. Inline-tests regression sweep across the workspace

- **Spec reference:** `reports/2026-06-23-hygiene-audit.md` Top 3 #1 (Finding
  3 in the sfmtool-core section + Finding 10 in the format-crates section).
- **Current state:** The 2026-06-10 inline-test extraction pass standardized
  on sibling `tests.rs` files (20 such files now exist), but stopped before
  `geometry/` and `analysis/alignment/`, and a PR #111 regression
  re-introduced inline tests in `camera/viewport.rs`. The format crates
  weren't in scope at all.
- **Scope:** Mechanical extraction in two passes:
  - `sfmtool-core`: `geometry/{rotation,rot_quaternion,rigid_transform,
    se3_transform,transform,viewing_angle}.rs` (5 files, ~1077 test lines),
    `analysis/alignment/{kabsch,ransac}.rs` (2 files, ~264), plus
    `camera/viewport.rs` (the PR #111 relapse, ~58). Roll in the
    smaller stragglers (`reconstruction/point_correspondence.rs`,
    `features/sift/orientation.rs`, etc.) for uniformity.
  - Format crates: lift the `#[cfg(test)] mod tests { … }` blocks out of
    `sfmr-format/src/lib.rs` (~1272 lines), `camrig-format/src/lib.rs`
    (~637), `matches-format/src/lib.rs` (~516) into sibling `tests.rs`
    files. Skip `sift-format` (below the 700-line threshold).
- **Why now:** Single sibling-`tests.rs` convention restored across the
  workspace — including the `camera/viewport.rs` relapse, which is
  symbolically the worst miss (a regroup commit re-introduced the very
  pattern the prior pass had cleaned).

### 5. Split `patch/normal_refine.rs` along its author-banner'd seams

- **Spec reference:** `reports/2026-06-23-hygiene-audit.md` Top 3 #2
  (Finding 1).
- **Current state:** The single largest production file in `sfmtool-core`
  at 1812 lines; `fronto_cache.rs` (595) and `prof.rs` are already
  siblings under `patch/normal_refine/`. The author has placed `// ---`
  banners marking six distinct concerns: public types (26-225),
  parameterization (248-323), level context (324-422), the objective-math
  kernel band (423-1099, ~675 lines), coarse-to-fine search (1100-1364),
  top-level refine (1365-1519), and a multi-view render substrate
  `PatchViewStack` (1520-1737).
- **Scope:** Split into sibling files under `patch/normal_refine/`:
  `params.rs`, `parameterization.rs`, `level.rs`, `znorm.rs`,
  `consensus.rs`, `search.rs`, `view_stack.rs`. `normal_refine.rs` keeps
  `refine_patch_normal[_impl]`, `refine_patch_cloud_normals`,
  `view_indices_from_reconstruction` (~250 lines of orchestration).
  Mostly `pub(super)` threading on `LevelContext`, `ProjectedImage`,
  `Objective`, `PatchWindow`.
- **Why now:** Biggest navigability win in `sfmtool-core` now that the
  regroup arc has landed. Banners already telegraph the cuts;
  `fronto_cache.rs` is precedent for sibling extraction under
  `normal_refine/`. Tests already sibling-extracted in
  `normal_refine/tests.rs`, so the test surface is unaffected.

> _Status (2026-06-30): Done — split into `normal_refine/{params,
> parameterization,support,level,znorm,consensus,search,view_stack}.rs`;
> `mod.rs` keeps the `refine_*` orchestration + re-exports. Largest resulting
> file is 389 lines (was 1812 at audit time, 2107 when split). Public API and
> the `crate::patch::normal_refine::*` paths are unchanged. clippy clean, 113
> patch tests green. See `reports/2026-06-23-hygiene-audit.md` Finding 1.
> Commit (branch `claude/next-steps-reports-sh73d6`)._

> Larger items still open and worth scheduling beyond the Top 5: post-regroup
> path drift sweep across 14 of 16 core specs and 3 GUI specs (mechanical
> doc-only, high readability value); `archive_io.rs` consolidation into a
> new `archive-format-common` crate (eliminates 4-way duplication with one
> already-drifted copy); PyO3 submodule restructure (grew 74 → 80
> registrations; first cut is the `io/` subdir move).
>
> _Status (2026-07-01): the three GPU/GUI splits from this list are done on
> branch `gui-gpu-work` — `optical_flow/gpu/mod.rs` (commit a1fa614),
> `app::run_ui_and_paint` (97812b1), and `image_detail::show` (1450b09), all
> verified on GPU + display hardware. See the marked-off findings 2/7/8 in
> `reports/2026-06-23-hygiene-audit.md`._

---

## 2–3 design-discussion topics (new features, not yet specced)

### A. Camera bookmarks (save/restore named viewpoints) in SfM Explorer

- **Motivation:** `specs/gui/gui-viewport-navigation.md` § "Future
  Enhancements" lists `- [ ] Save/restore camera positions` as an
  unimplemented checklist item that's been there since the spec was
  written. For inspection and before/after comparison it's invaluable to
  jump back to a saved vantage point — especially useful for the existing
  Camera View Mode (which already supports per-image-frame snapshots; this
  would generalize to user-named arbitrary viewpoints).
- **Sketch:** A small bank of bookmark slots storing the full
  orbit-camera state (`position`, `orientation`, `target_distance`, `fov`,
  `world_up`). Bind number keys to recall and a modifier+number to store,
  with an optional dock-panel listing named bookmarks (parallel to the
  existing `PointTrackDetail` panel shape). Persist per-reconstruction in
  a sidecar JSON (keyed by the `.sfmr` path) so they survive restarts.
  Recall could reuse the existing animated-transition path (slerp + ease)
  already built for Alt+click target moves and for `switch_camera_view`.
- **Where it would live:** New `specs/gui/gui-camera-bookmarks.md`; state
  in `crates/sfm-explorer/src/viewer_3d/camera.rs` and `state.rs`,
  persistence alongside the existing file-load path.
- **Open questions:** Where to persist (sidecar file vs. embed in `.sfmr`
  metadata vs. a global app-state file)? How many slots, named vs.
  numbered? Should a bookmark capture overlay-mode / selection state too,
  or pose only? Should it auto-suggest a name from the closest image (a
  "Camera 5"-style label)?

### B. `sfm xform --crop` (3D bounding-volume crop)

- **Motivation:** Today's spatial filtering in `xform`
  (`specs/cli/xform-command.md`) is by image (range/glob) or by point
  statistic (track length, reprojection error, isolation, NN distance).
  There is no way to say "keep only the points inside this region of the
  scene" — the most natural operation when you want to isolate one object
  out of a larger reconstruction, or carve out background clutter before
  exporting to NeRF/Nerfstudio. Carried over from
  `reports/2026-05-22-next-steps.md`; still no spec, still wanted.
- **Sketch:** A new `CropTransform` in `src/sfmtool/xform/` taking an
  axis-aligned box (`--crop xmin,ymin,zmin,xmax,ymax,zmax`) or a
  centre+radius sphere, dropping points outside it and remapping
  observations through the existing point-removal / remap path in
  `_point_filters.py`. Cameras stay (crop affects points only) or
  optionally drop when they observe nothing left. A future extension:
  derive the box interactively from a selection in SfM Explorer and emit
  the `--crop` argument string.
- **Where it would live:** New filter class in `src/sfmtool/xform/`, a
  new `--crop` option in `_commands/xform.py`, and a new "Filtering
  Operations" subsection in `specs/cli/xform-command.md`.
- **Open questions:** Coordinate frame (raw reconstruction units vs.
  physically-scaled — does the user know the scale?)? Should cameras with
  zero surviving observations be dropped automatically or left for a
  follow-on `--remove-short-tracks`? Box vs. sphere vs. oriented box —
  and how does a user author the numbers without a viewer? (Answer might
  be: ship box + sphere first, and let the GUI generate the numbers in
  the bookmark-adjacent design discussion above.)

### C. Adaptive wide-baseline pair selection for flow-based matching

- **Motivation:** `specs/core/flow-based-matching.md` § "Future
  Directions" sketches improvements over today's fixed-size sliding
  window. Fixed windows under-connect slow pans (lots of redundant
  near-baseline pairs) and over-connect fast motion (missed loop closure
  on fast pans). Carried over from `reports/2026-05-22-next-steps.md`;
  still no spec, still wanted.
- **Sketch:** Track cumulative median flow magnitude along the sequence;
  when it crosses a threshold (the spec suggests ~50 px) since the last
  wide-baseline computation, emit an extra wide-baseline pair — adapting
  pair density to camera speed instead of frame index. A second pass
  could build a covisibility graph from initial tracks and add flows only
  between high-but-incomplete-overlap pairs. The cheap
  accumulated-displacement trigger is the natural first cut and reuses
  flow magnitudes already computed for adjacent pairs.
- **Where it would live:** Extend `specs/core/flow-based-matching.md`
  from "Future Directions" into a committed design; implementation in the
  flow-matching modules under `src/sfmtool/feature_match/` (and core if a
  graph pass is added).
- **Open questions:** Right threshold and whether it should be absolute
  pixels or relative to image size? Interaction/ordering with the
  existing fixed window — replace or layer on top? How to cap the extra
  pairs so cost stays bounded on long sequences? Does this generalize the
  approach to handle deliberate orbital captures where the camera
  re-encounters earlier viewpoints?
