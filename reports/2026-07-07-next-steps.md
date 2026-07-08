# sfmtool — Next Steps (2026-07-07)

A dated snapshot of recommended next work, drawn from the same-day `audit-specs`
and `audit-hygiene` rebaselines (`reports/2026-07-07-spec-audit.md`,
`reports/2026-07-07-hygiene-audit.md`) plus the open items still carried in
`reports/2026-06-13-perf-patch-normal-refinement.md`.

This snapshot **supersedes `2026-06-23-next-steps.md`** (retired in the same
commit). Of that report's five implementation items: #1 (CLI bug-fix bundle),
#3 (sfmr v4 fold-in), and #5 (normal_refine split) shipped with Done
annotations; #4 (inline-tests sweep) shipped but new stragglers have since
appeared (carried forward below as item 4); #2 (refine_normals perf wins) is
half-done — `confidence: bool` landed (default false), the subset-aware pyramid
build remains in the 2026-06-13 perf report's open items. Design topics A
(camera bookmarks) and B (`xform --crop`) remain open and are carried forward;
C (adaptive wide-baseline pair selection) is dropped this round — still valid,
but the patch pipeline has clearly become the project's center of gravity and
it wasn't pulling demand.

---

## ~5 implementation tasks (from existing specs and audit findings)

Ranked roughly by value × readiness — cheap, high-leverage wins first.

### 1. Land the 2026-07-07 spec-sync bundle

> _Status (2026-07-08): Done. Decisions (all resolved as keep-shipped-behavior, update-spec): subpixel is on-by-default (integer knob, default 1) — `keypoint-subpixel-refinement.md` rewritten; the three GUI contradictions (camera-view per-frame FOV → set-once-on-entry, browser thumbnail deselect → no toggle-off, all-black patch tile → drawn) reconciled to code. Specs synced: `sift-to-patch-reconstruction.md` (rounds=2 loop + obliquity/fronto priors + 8 knobs), `patch-keypoint-localization.md` (PlusDescent default), `keypoint-localization-search-cache.md` (status → implemented, AVX2 landed), `track-cluster-matching.md` (operative prose → d=10, prototype history kept), `gui-point-track-detail.md` (Size column + Depth-z/Cond header diagnostics). GUI mechanical sweep: octahedron→compass ×4, main.rs→app.rs/dock.rs/lib.rs, sfmtool-gui→sfm-explorer, patch.wgsl/patch pipeline added to architecture+plan, bg_image pinhole→distorted note. `workspace.md` names sfmtool as default tool; AGENTS.md fixed (cam→camrig group with 3 subcommands, 7→8 crates); `sfm version` now reads package metadata (prints 0.2). The only code change was the version string; everything else docs. Follow-up (done same day, commit e176f74): the zup-migration doc was retired from `specs/drafts/` — the migration is actually merged to main (#162/#164), so the doc was a completed record; its unique §1/§2 content (S/W conversion math, invariants, D-decisions) was promoted into `sfmr-file-format.md` and its 7 code/spec references repointed there._

- **Spec reference:** `reports/2026-07-07-spec-audit.md` Top Priorities #1, #2,
  #5 and the honorable mentions — per-spec diagnoses with file:line cites.
- **Current state:** 13 of 45 specs carry divergences, all diagnosed. Two need
  a *decision*, the rest are typing: (a) `keypoint-subpixel-refinement.md`
  claims sub-pixel LK ships opt-in (`subpixel="none"`) but the code ships an
  integer sweep count defaulting to **1 = on** — confirm on-by-default is
  intended, then rewrite the wiring paragraph; (b) three GUI contradictions
  where spec sections disagree internally and the code picked a side
  (camera-view per-frame FOV, browser thumbnail deselect, all-black patch
  tile).
- **Scope:** Update `sift-to-patch-reconstruction.md` for the `rounds=2` loop +
  obliquity/fronto priors + 8 undocumented knobs; note the PlusDescent default
  in `patch-keypoint-localization.md`; bump the stale
  `keypoint-localization-search-cache.md` header; reconcile the `d=28`-era
  prose in `track-cluster-matching.md` with the shipped `d=10`; GUI mechanical
  sweep (octahedron→compass ×4, `main.rs`→`app.rs` ×2, `sfmtool-gui`→
  `sfm-explorer`, add `patch.wgsl`/patch rendering to architecture+plan) plus
  the three decisions above; name `sfmtool` as the workspace default tool in
  `workspace.md`; fix AGENTS.md (`cam` group → `camrig` with 3 subcommands;
  7 crates → 8) and the `sfm version` `0.1` string.
- **Why now:** The audit already did the hard part (diagnosis, with line
  numbers); the specs are this repo's contract and 30+ are provably exact
  right now — cheap to get back to full sync while the diagnoses are fresh.

### 2. Implement-or-remove the dead CLI surface

> _Status (2026-07-08): Done. Cluster 1 — removed `--max-error`/`--iterative`/`--visualize` from `align`. Cluster 2 — implemented most-common point selection and the workspace hash-prefix source search (reusing `find_sfmr_by_content_hash`, lifted to `_workspace.py`), with new `tests/xform/test_scale_by_measurements.py`. Cluster 3 — deleted the dead `spatial_tolerance` and rewrote `flow-based-matching.md` to document the shipped 10px/K=5 mechanism. Only the `align` "shortest path" spec prose was deferred into item #1's spec sweep._

- **Spec reference:** `reports/2026-07-07-spec-audit.md` Top Priorities #3, #4
  + the flow-matching honorable mention; `specs/cli/align-command.md`,
  `specs/cli/scale-by-measurements-command.md`, `specs/core/flow-based-matching.md`.
- **Current state:** Three clusters of documented-but-inert behavior:
  - `sfm align` accepts `--max-error`, `--iterative`, `--visualize` and drops
    them silently (`align/multi.py:315-317` never reads them).
  - `--scale-by-measurements` lacks two spec'd behaviors: the workspace
    hash-prefix search for the source `.sfmr` (`_scale_by_measurements.py:356-377`)
    and "most common point index" on ambiguous matches (`:132` takes an
    arbitrary first while printing "Using most common." — actively misleading).
  - Flow matching's public `spatial_tolerance=3.0` parameter is dead — the real
    radius is the hard-coded `_SPATIAL_CANDIDATES_RADIUS = 10.0` with K=5
    best-descriptor selection (`_flow_matching.py:40-41,180`).
- **Scope:** Per cluster, either implement the documented behavior or delete
  the parameter and update the spec. Minimum viable: fix the misleading
  message, wire-or-delete `spatial_tolerance`, drop the three align no-ops.
- **Why now:** These are user-facing silent no-ops — the worst kind of drift —
  and every one is already diagnosed to the line.

### 3. Split the two big keypoint kernel files

- **Spec reference:** `reports/2026-07-07-hygiene-audit.md` Top 3 #1;
  precedent `patch/normal_refine/` (split 2026-06-30, held up well).
- **Current state:** `patch/keypoint_localize.rs` is 2042 lines / six concerns
  (params, render cache, scalar primitives, search strategies, ~430 lines of
  AVX2 kernels, orchestration) — the largest source file in the workspace.
  `keypoint_subpixel.rs` (1220) repeats the same shape. Both sibling dirs
  already exist (holding `tests.rs`).
- **Scope:** Extract `keypoint_localize/{params,search,kernels}.rs` and
  `keypoint_subpixel/{params,kernels}.rs`, keeping orchestration in the top
  modules; move the AVX2 blocks intact with their SAFETY comments. One pass so
  the two stay symmetric. `cargo clippy` + patch tests; no public-API change.
- **Why now:** This is the most actively developed area (#174/#175/#177 all
  touched it in three weeks) — navigability pays rent immediately, and the
  in-repo precedent makes the shape uncontroversial.

### 4. Mechanical hygiene batch: inline-test sweep + test_camrig split

- **Spec reference:** `reports/2026-07-07-hygiene-audit.md` Top 3 #2 and the
  carried-forward finding #12.
- **Current state:** Five sizeable inline `#[cfg(test)] mod tests` blocks defy
  the sibling-`tests.rs` convention — `sift-format/src/lib.rs` (~326 test
  lines), `sfmr-colmap/src/colmap_io/mod.rs` (~667, larger than the module's
  own code), `sfmr-format/src/depth_stats.rs` (~153), `camrig-format/src/pattern.rs`
  (~115), `sfm-explorer/src/scene_renderer/auto_point_size.rs` (~211). And
  `tests/test_camrig.py` still carries the self-contained resolver/pattern
  block (~240 lines, 20 tests, two local-only helpers) flagged two audits ago.
- **Scope:** Pure code motion, one commit: five `tests.rs` extractions + lift
  `test_camrig.py:444-681` into `tests/test_camrig_resolve.py`.
- **Why now:** Lowest effort-to-value on the board; clears six findings and
  restores a convention that recent crates (`camrig-format`) are drifting from.

### 5. Finish the `_sfmtool` submodule migration (last 10 names + `import *`)

- **Spec reference:** `reports/2026-07-07-hygiene-audit.md` carried-forward #4
  status check.
- **Current state:** Eight submodules (`io`, `matching`, `geometry`, `flow`,
  `spatial`, `spherical`, `analysis`, `sift`) are properly registered via
  `install_submodule`; flat registrations are down 80 → 10 (4 functions + 6
  classes, incl. `PySfmrReconstruction`, `PyPatchCloud`). But
  `src/sfmtool/__init__.py:4` still wildcard-imports the flat surface, and 8
  binding files sit flat at `src/` top level.
- **Scope:** Group the residual bindings into 2–3 submodules (`patches`,
  `reconstruction`, `image`), move the files, replace `import *` with explicit
  re-exports; update the ~10 `from sfmtool._sfmtool import ...` call sites.
  Consider folding the `py_patch_cloud.rs` four-algorithm split (hygiene
  finding) into the same move since the file relocates anyway.
- **Why now:** The migration is 87% done and the remaining flat names are
  exactly the high-traffic types — finishing it locks in a deliberate public
  API before more consumers accrete on the wildcard.

> Also queued from the audits, worth scheduling behind the Top 5: extract the
> shared compaction helpers out of `_embed_patches.py` (fixes the
> `xform/_localize_keypoints.py` → top-level upward import), the `strips/`
> subpackage regroup (5 modules, 2 external import sites), the
> `scene_renderer/upload.rs` per-resource split, `tests/patch/` regroup,
> conftest solve-retry dedup (#11), the subset-aware pyramid build from the
> 2026-06-13 perf report, and the deferred `archive_io` consolidation (drift
> now affects 2 of 4 copies — the missing empty-bytes guards are worth
> grabbing even if the shared crate stays deferred).

---

## 2–3 design-discussion topics (new features, not yet specced)

### A. Camera bookmarks (save/restore named viewpoints) in SfM Explorer — carried forward

- **Motivation:** `specs/gui/gui-viewport-navigation.md` § "Future
  Enhancements" still lists `- [ ] Save/restore camera positions` (re-verified
  unimplemented in this round's spec audit). For inspection and before/after
  comparison it's invaluable to jump back to a saved vantage point —
  generalizing Camera View Mode's per-image snapshots to user-named arbitrary
  viewpoints.
- **Sketch:** A bank of bookmark slots storing full orbit-camera state
  (`position`, `orientation`, `target_distance`, `fov`, `world_up`). Number
  keys recall, modifier+number stores; optional dock panel listing named
  bookmarks. Persist per-reconstruction in a sidecar JSON keyed by `.sfmr`
  path. Recall reuses the existing animated-transition path (slerp + ease)
  built for Alt+click target moves and `switch_camera_view`.
- **Where it would live:** New `specs/gui/gui-camera-bookmarks.md`; state in
  `viewer_3d/camera.rs` + `state.rs`; persistence beside the file-load path.
- **Open questions:** Sidecar vs. `.sfmr` metadata vs. global app state? Named
  vs. numbered slots? Capture overlay/selection state or pose only?

### B. `sfm xform --crop` (3D bounding-volume crop) — carried forward

- **Motivation:** Spatial filtering in `xform` is still by image or by point
  statistic (re-verified: no crop transform exists). There's no "keep only the
  points inside this region" — the most natural operation for isolating one
  object or carving background clutter before a Nerfstudio export.
- **Sketch:** A `CropTransform` taking an axis-aligned box
  (`--crop xmin,ymin,zmin,xmax,ymax,zmax`) or centre+radius sphere, dropping
  outside points and remapping observations through the existing point-removal
  path. Cameras stay, or optionally drop when they observe nothing. Natural
  synergy with topic A: the GUI could emit the crop numbers from an
  interactive selection.
- **Where it would live:** New filter in `src/sfmtool/xform/`, `--crop` in
  `_commands/xform.py`, new subsection in `specs/cli/xform-command.md`.
- **Open questions:** Coordinate frame (raw vs. physically-scaled units)?
  Auto-drop empty cameras? Box vs. sphere vs. oriented box, and how does a
  user author the numbers without a viewer?

### C. Pose-aware per-tile source stacks (parallax-correct panoramas)

- **Motivation:** `specs/core/per-spherical-tile-source-stack.md` explicitly
  defers a pose-aware `build_with_pose` — today's `build_rotation_only`
  assumes scene-at-infinity, so `sfm panorama` is only correct for
  near-concentric captures. Real handheld rigs translate; nearby geometry
  ghosts across tile seams and the photometric RANSAC has to spend its
  clusters absorbing parallax instead of exposure/occlusion differences. This
  is the biggest quality ceiling on the panorama pipeline the last month of
  work built up.
- **Sketch:** Add a depth proxy per tile — the reconstruction already carries
  triangulated points, so a per-tile median depth (or small plane fit) from
  points inside the tile's frustum gives each tile a projection surface;
  `build_with_pose` warps each source through the existing pose-aware
  `WarpMap::from_cameras_with_pose` path onto that surface instead of the
  infinity rotation-only warp. Tiles with too few points fall back to
  rotation-only. The RANSAC/consensus and batched-atlas stages are unchanged —
  they just receive better-registered stacks.
- **Where it would live:** Extend `specs/core/per-spherical-tile-source-stack.md`
  (the "Pose-aware variant" stub) + a `--parallax` flag in
  `specs/cli/panorama-command.md`; implementation in
  `crates/sfmtool-core/src/spherical/` reusing `camera/warp_map.rs`.
- **Open questions:** Per-tile plane vs. per-tile depth constant vs. coarse
  mesh? What point density is enough, and is the fallback per-tile or global?
  Does the consensus scoring need a depth-aware validity mask where the proxy
  surface is wrong? How much of the byte-identical batching contract
  (`tile_index_base` reseeding) survives a per-tile geometry input?
