# sfmtool — Next Steps (2026-05-22)

A dated snapshot of recommended next work, drawn from existing specs/code and
verified against the actual implementation. Two lists: concrete implementation
tasks pulled from already-written specs, then speculative design topics.

> **Update 2026-05-23:** Item #4 (render points-at-infinity in SfM Explorer)
> has since shipped — commit c3c2805, with `positions_xyzw` now consumed in
> `crates/sfm-explorer/src/scene_renderer/{upload,auto_point_size}.rs`. It is
> struck through below. Remaining items are unchanged.

---

## ~5 implementation tasks (from existing specs)

Ranked roughly by value × readiness — cheap, high-leverage wins first.

### 1. Expose the equirectangular-panorama pipeline as a `sfm` CLI command

- **Spec reference:** `specs/core/spherical-tiles-rig.md`,
  `specs/core/per-spherical-tile-source-stack.md`,
  `specs/core/tile-batched-consensus-atlas.md`,
  `specs/drafts/photometric-subsets-ransac.md` (all marked **Implemented** in
  Rust). No `specs/cli/*` file describes a panorama command yet.
- **Current state:** The whole render chain exists and is wired through PyO3:
  `SphericalTileRig`, `PerSphericalTileSourceStack`, `refine_photometric_ransac`,
  `render_consensus_atlas` / `primary_consensus_atlas`, and
  `SphericalTileRig.resample_atlas`. It is driven end-to-end **only** by the
  standalone `scripts/render_equirectangular.py`. There is no `sfm` subcommand:
  `cli.py` registers nothing for it, and the only `_commands/` reference to
  `SphericalTileRig` is `camrig.py` (which writes a tiled `.camrig`, unrelated to
  rendering). `src/sfmtool/_spherical_tile_rig.py` already has a clean
  `resample_atlas_to_equirect` wrapper.
- **Scope:** Add `_commands/panorama.py` (e.g. `sfm panorama RECON.sfmr -o
  pano.png` with `--equirect-width`, `--k`, RANSAC knobs), promote the script's
  pipeline into a reusable function in `src/sfmtool/`, register it in `cli.py`
  under Visualization, and write `specs/cli/panorama-command.md`.
- **Why now:** Highest value-to-effort ratio here. A large, fully-tested,
  spec-backed Rust capability is currently reachable only by running a script by
  hand — surfacing it as a first-class command makes a marquee feature usable.

### 2. Implement the FOV-zoom gesture in SfM Explorer

- **Spec reference:** `specs/gui/gui-viewport-navigation.md` §"FOV Zoom
  (Planned)" and the Future Enhancements checklist (`- [ ] FOV zoom`);
  `specs/gui/gui-plan.md` "Next Steps" item 2.
- **Current state:** FOV is adjustable only via the View-menu slider (10°–120°).
  Grep for `fov_zoom`/`pinch.*fov` in `crates/sfm-explorer/` finds only the
  existing dolly-zoom input handling in `viewer_3d/input.rs`; no gesture binding
  changes `fov` in place.
- **Scope:** Pick an input binding distinct from dolly zoom (the spec's open
  question — survey leans toward a modifier+scroll or a dedicated key+drag), wire
  it in `viewer_3d/input.rs` to mutate `fov` while leaving `target_distance`
  fixed, respecting the camera-view-mode override already documented. Update the
  spec's open question and tick the checklist box.
- **Why now:** Small, self-contained GUI change against a spec that already did
  the design survey; closes a long-standing `[ ]` and a named roadmap item.

### 3. Fold `densify` into `sfm xform` as a chainable subcommand

- **Spec reference:** `specs/cli/densify-command.md` — its own NOTE: "I think it
  makes sense to somehow merge into the 'sfm xform' as a subcommand, where you
  can pipeline it with point filtering and bundle adjustment in a more generic
  way."
- **Current state:** `densify` is a standalone top-level command
  (`_commands/densify.py`, registered in `cli.py` line 65). `xform` already owns
  filtering + bundle-adjust + align as ordered `Transform` objects
  (`src/sfmtool/xform/`), and `densify` re-implements its own filter/BA/align
  tail — exactly the duplication the note calls out.
- **Scope:** Add a `DensifyTransform` implementing the `xform.Transform`
  protocol (matching/triangulation step only), so users can write
  `sfm xform in.sfmr out.sfmr --densify ... --filter-by-reprojection-error 2
  --bundle-adjust --align-to-input`. Keep `densify` as a thin alias or deprecate
  it. Update both command specs.
- **Why now:** Removes duplicated filter/BA/align logic, makes an experimental
  feature composable with the existing pipeline, and is the maintainer's own
  stated intent. Medium effort; mostly re-plumbing existing pieces.

### 4. ~~Render points-at-infinity (`w = 0`) in SfM Explorer~~ — DONE (2026-05-23, commit c3c2805)

- **Spec reference:** `specs/drafts/sfmr-v2-points-at-infinity.md` §"API and
  consumers": "`sfm-explorer`: reads `positions_xyzw`; renders a `w = 0` point as
  a direction on the far view sphere rather than at a finite coordinate."
- **Status:** Shipped. The GUI now consumes `positions_xyzw` in
  `crates/sfm-explorer/src/scene_renderer/upload.rs` and
  `scene_renderer/auto_point_size.rs`, projecting `w = 0` points onto the
  adaptive far sphere. No longer an open task.

### 5. Add an `--camera-config PATH` flag for explicit calibration selection

- **Spec reference:** `specs/workspace/camera-config.md` §"Open Question:
  Comparing Calibrations Side by Side" (the `--camera-config PATH` candidate).
- **Current state:** Resolution is purely presence-based closest-ancestor
  (`src/sfmtool/_camera_config.py`); no command accepts an override path
  (grep finds the string only inside `_camera_config.py`, no CLI option). A/B
  testing two calibrations on one image set currently requires renaming files.
- **Scope:** Add `--camera-config PATH` to `solve` / `match` / `to-colmap-db`
  that overrides the discovered file for the run, define precedence vs.
  `.camrig` and `rig_config.json`, and document it. The spec lists three options;
  this is the one needing a small design pass before coding.
- **Why now:** Unblocks the calibration A/B workflow the spec explicitly flags as
  missing. Lower in the ranking because it reopens a precedence question the spec
  did not commit to, so it needs a short design decision first.

---

## 2–3 design-discussion topics (new features, not yet specced)

### A. `sfm xform --crop` (3D bounding-volume crop)

- **Motivation:** Today's spatial filtering in `xform`
  (`specs/cli/xform-command.md`) is by image (range/glob) or by point statistic
  (track length, reprojection error, isolation, NN distance). There is no way to
  say "keep only the points inside this region of the scene" — the most natural
  operation when you want to isolate one object out of a larger reconstruction.
- **Sketch:** A new `CropTransform` in `src/sfmtool/xform/` taking an
  axis-aligned box (`--crop xmin,ymin,zmin,xmax,ymax,zmax`) or a centre+radius
  sphere, dropping points outside it and remapping observations exactly as the
  existing filters do (reuse the point-removal/remap path in `_point_filters.py`).
  A nice extension: derive the box interactively from a selection in SfM
  Explorer and emit the `--crop` argument string. Cameras could be kept (crop
  affects points only) or optionally pruned when they observe nothing left.
- **Where it would live:** New filter class in `src/sfmtool/xform/`, a new
  `--crop` option in `_commands/xform.py`, and a new "Filtering Operations"
  subsection in `specs/cli/xform-command.md`.
- **Open questions:** Coordinate frame (raw reconstruction units vs.
  physically-scaled)? Should cameras with zero surviving observations be dropped
  automatically or left for a follow-on `--remove-short-tracks`? Box vs. sphere
  vs. oriented box — and how does a user author the numbers without a viewer?

### B. Save / restore named viewpoints (camera bookmarks) in SfM Explorer

- **Motivation:** `specs/gui/gui-viewport-navigation.md` lists
  `- [ ] Save/restore camera positions` as an unimplemented future enhancement,
  but there is no spec describing the feature. For inspection and
  before/after comparison it is invaluable to jump back to a saved vantage point.
- **Sketch:** A small set of bookmark slots storing the full orbit-camera state
  (`position`, `orientation`, `target_distance`, `fov`, `world_up`). Bind
  number keys to recall and a modifier+number to store, with an optional dock
  panel listing named bookmarks. Persist them per-reconstruction in a sidecar
  JSON (keyed by the `.sfmr` path) so they survive restarts. Recall could reuse
  the existing animated-transition path (slerp + ease) already built for
  Alt+click target moves.
- **Where it would live:** New `specs/gui/gui-camera-bookmarks.md`; state in
  `crates/sfm-explorer/src/viewer_3d/camera.rs` and `state.rs`, persistence
  alongside the existing file-load path.
- **Open questions:** Where to persist (sidecar file vs. embed in `.sfmr`
  metadata vs. a global app-state file)? How many slots, and named vs. numbered?
  Should a bookmark capture overlay/selection state too, or pose only?

### C. Adaptive wide-baseline pair selection for flow-based matching

- **Motivation:** `specs/core/flow-based-matching.md` §"Future Directions:
  Wide-Baseline Pair Selection" sketches two improvements over the current
  fixed-size sliding window (`feature_match/` has no
  accumulated-displacement/covisibility trigger today). Fixed windows
  under-connect slow pans and over-connect fast motion, hurting loop closure.
- **Sketch:** Track cumulative median flow magnitude along the sequence; when it
  crosses a threshold (the spec suggests ~50 px) since the last wide-baseline
  computation, emit an extra wide-baseline pair — adapting pair density to
  camera speed instead of frame index. A second pass could build a covisibility
  graph from initial tracks and add flows only between high-but-incomplete-overlap
  pairs. The cheap accumulated-displacement trigger is the natural first cut and
  reuses flow magnitudes already computed for adjacent pairs.
- **Where it would live:** Extend `specs/core/flow-based-matching.md` from
  "Future Directions" into a committed design; implementation in the flow-matching
  modules under `src/sfmtool/feature_match/` (and core if a graph pass is added).
- **Open questions:** Right threshold and whether it should be absolute pixels or
  relative to image size? Interaction/ordering with the existing fixed window —
  replace it or layer on top? How to cap the extra pairs so cost stays bounded
  on long sequences?
