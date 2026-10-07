# Spec audit — 2026-10-06

**Sample:** 18 of 175 specs read against their code. Seed `25716`; the pool
was the never-audited specs (155 candidates). The random draw gave 10 of them:

- `specs/core/camera/refit-camera-intrinsics.md`
- `specs/core/features/flow-based-matching.md`
- `specs/core/geometry/absolute-pose.md`
- `specs/core/features/cluster-covisibility.md`
- `specs/core/geometry/bundle-adjustment.md`
- `specs/cli/reconstruction/xform/localize-keypoints-command.md`
- `specs/core/features/randomized-kdtree-forest.md`
- `specs/core/geometry/pose-verification.md`
- `specs/core/geometry/reprojection-residuals.md`
- `specs/cli/image-processing/flow-command.md`

Eight more were read regardless of the draw:

- **Code changed materially since the 2026-09-26 audit:**
  - `specs/gui/mcp-server.md` (about 200 file changes under `sfm-explorer/src/mcp/`, including #647, #649, #642, #804)
  - `specs/gui/track-view.md` (#645 through #809)
  - `specs/core/patch/zncc-self-similarity-radius.md` (#637 through #772)
  - `specs/core/bench/nearby-tracks.md` (moved into core, #634 through #642)
- **Format specs with a check-6 finding:** `matches-file-format`,
  `sfmr-file-format`. Both also changed in #805, #806 and #808.

The mechanical checks below cover all 175. This report replaces
`2026-09-26-spec-audit.md`: its findings that are still open are under
**Carried forward**, and the specs it and the 2026-09-05 audit read are listed
there so the sampler keeps counting them as audited. Every finding below was
checked against `main` at `dd4f6722` (#809).

Non-goals and deferral entries checked in the sampled specs: 98 in all, counting
the five negative claims in `localize-keypoints-command.md`, which has no
Non-goals section (26 in `mcp-server.md`). Seven no longer hold, one of them
only in part; they are listed under Top priority 2.

## Mechanical findings (all 175 specs)

The corpus is 175 specs under `specs/` and `docs/` (README.md and TEMPLATE.md
excluded), plus 16 drafts in `specs/drafts/`, which are counted separately where
a check reads them. The previous run (2026-09-26) counted 173. The scripts are
not in the repository.

### 1. Documented defaults vs actual defaults

| Step | Now | 2026-09-26 |
|---|---|---|
| Parameter rows extracted from specs (tables, prose `default`, fenced ` ```rust ` / ` ```python ` blocks and signatures) | 914 (7 in drafts) | 588 |
| Rows keyed to an owner and compared (spec → owning command or function/type → parameter) | 611 | 588 |
| Candidates after numeric normalization | 93 | 162 |
| Numeric rows the scan could not key to an owner (symbol tables, dotted names, GUI controls), cleared by hand | 166 | — |
| **Confirmed default drift** | **0** | 3 |

How the check was keyed. A CLI spec is keyed to its own command module
(`specs/cli/<cat>/<cmd>-command.md` → `_commands/<cmd>.py`, the `xform/*`
specs → `_commands/xform.py`), so a flag that several commands share is
compared against only the owning command. A library spec is keyed to the
functions and types it names in backticks or links to. Code defaults were read
from Click `default=`, `#[pyo3(signature = …)]`, `impl Default` bodies, `const`
items, Python function defaults and dataclass fields (2,801 in all); a
`None` that resolves downstream was followed to where it resolves.

The 93 candidates (56 from tables, 20 from fenced signatures, 10 from
`// default` comments in fenced Rust, 4 from prose, 3 from other fenced code;
8 of the 93 are CLI options) were read against the code. All of them clear,
for one of four reasons:

- The code holds the value in a named constant or a `DEFAULT` struct the scan
  did not resolve (`ConstellationParams::DEFAULT`, `BENCH_*` in
  `bench/track.rs:801-833`, `DEFAULT_MAX_SEED_OFFSET_PX` /
  `DEFAULT_MAX_CACHE_BYTES` in `bench/evaluate.rs:139-142`, `KdfWriteOptions` /
  `LazyKdForestOptions` in `sfmtool-kdf-format/src/types.rs:148-239`); every
  value matches.
- The row is an example call in a fenced block that overrides a default on
  purpose (for example `covered-by-finer.md:107`, and
  `prune-covered-observations.md:241` `footprint_fraction=0.4545` against a
  documented and coded `0.5`).
- A CLI option has `default=None` and resolves downstream to the documented
  value (`ws init --max-features` → 8192 in `sift/extract_sfmtool.py:61`;
  `epipolar --sweep-window-size` → 30 at `epipolar.py:226`;
  `analyze --depth-likelihood-ratio-threshold` → 25 through
  `DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`).
- Table-parse noise: prose rows, and enum spellings such as `PlusDescent`
  against `"plus_descent"`.

The 166 unkeyed rows were checked by group against the defining code: SIFT
parameters (`sift/mod.rs:129-147`), kd-forest (`kdforest/mod.rs:112-118`),
optical flow (`optical_flow/params.rs:50-66`, including `theta_sf = theta_ss - 2`
at `mod.rs:153`), photometric RANSAC (`photometric_ransac.rs:64-77`),
cluster refinement (`cluster_refine/params.rs:110-134`, `mod.rs:68-72`),
track-at-pixel `finish.*` / `clusters.*` / `transfer.*` / `sweep.*` /
`constellation.*`, far-field sweep, nearby tracks, depth layers, and the
viewer's HUD sliders, intrinsics overlay, camera-view subdivision constants,
bench-layer shader constants, MCP limits and default dock layout. All match.

**Two blind spots closed this run, with no hits:** the CLI cross-check keyed
on owner found every option of all 30 commands documented in its own spec and
no spec flag that no command has (the six "stale" flags it raised are external
tools' flags such as `ns-train --data` and `colmap gui --import_path`). Click
`help=` strings that state a default were compared with `default=`: 11 differ
textually, and all 11 are `None` resolving to the stated value.

**Adjacent drift the scans surfaced (not defaults):**

| Where | Says | Code | Right side |
|---|---|---|---|
| `specs/cli/README.md:76` (index, outside the corpus) | xform sub-command `--select-by-distribution` | no such option; it is `--include-by-distribution` (`src/sfmtool/_commands/xform.py`; `xform-command.md:115`) | code — rename the index row |
| `specs/formats/matches-file-format.md:223-224` | `version`: "`1` through `6`" | `MATCHES_FORMAT_VERSION = 7` (`crates/sfmtool-matches-format/src/types.rs:133`); the same spec says seven versions at :1227 | code — say `1` through `7` |

### 2. Prose duplicated between a spec and its code

212 normalized lines of 60 or more characters appear in both a spec and a doc
comment or Python source (224 last run). 13 spec↔code pairs share 5 or more
lines (13 last run) and 24 share 3 or more.

| Shared (in fence) | Spec:lines | Code:lines |
|---|---|---|
| 12 (12) | gui/operation-progress.md:63-165 | sfmtool-progress/src/lib.rs:4-545 |
| 9 (8) | core/reconstruction/batch-triangulation-api.md:93-142 | sfmtool-core/src/reconstruction/triangulation.rs:79-126 |
| 9 (9) | gui/camera-intrinsics.md:519-537 | sfm-explorer/src/state.rs:133-162 |
| 7 (7) | gui/background-tasks.md:510-537 | sfm-explorer/src/background/mod.rs:60-398 |
| 7 (7) | gui/action-log.md:444-535 | sfm-explorer/src/action_log/mod.rs:58-404 |
| 7 (7) | core/patch/patch-cloud.md:74-317 | sfmtool-core/src/patch/cloud.rs:91-1466 |
| 7 (7) | core/camera/photograph-cache.md:50-89 | sfmtool-core/src/camera/photograph_cache.rs:186-400 |
| 6 (6) | gui/multi-panel-image-browser.md:817-829 | sfm-explorer/src/state.rs:193-202 |
| 6 (6) | core/patch/zncc-self-similarity-radius.md:85-144 | sfmtool-core/src/patch/self_similarity/mod.rs:47-138 |
| 5 (2) | core/bench/editable-track.md:344-1793 | sfmtool-core/src/bench/steps.rs:1599-1891 |
| 5 (5) | core/patch/patch-normal-refinement.md:340-420 | sfmtool-core/src/patch/normal_refine/params.rs:39-217 |
| 5 (4) | gui/mcp-server.md:9-4199 | sfm-explorer/src/mcp/mod.rs:9-1040 |
| 5 (5) | gui/panel-layout.md:801-817 | sfm-explorer/src/window.rs:511-557 |

Trend: `track-cluster-matching` (23 last run) has left the list; new entries
are `photograph-cache`, `multi-panel-image-browser`, `zncc-self-similarity-radius`
and `mcp-server`. As before, nearly all the sharing is doc comments copied into
a spec's fenced API block, and the spec's copy is the one that goes stale.

Plain prose outside code fences is shared at most 3 lines per pair. The one
pair with no fenced share is a doc comment re-deriving a spec's argument (a
third copy): `core/geometry/translation-averaging.md:136-140, 159-182` and
`translation_averaging.rs:670-676` (the conjugate-gradient null-space
argument) and `:877-892` (`orientation_reading`, the `c -> -c` reflection and
parallax-weighted vote argument, nearly word for word). The doc comment is the
copy that should shrink to contract plus link.

`sfmtool-progress/src/lib.rs`, the top pair, cites no spec at all (check 5),
so nothing ties its 12 shared lines to `operation-progress.md`.

### 3. Shape of the `specs/core/` specs

| | Now | 2026-09-26 |
|---|---|---|
| `specs/core/` specs | 85 | 75 |
| with a ` ```rust ` block | 59 | 45 |
| without one | 26 | 30 |
| no code block at all | 11 | 12 |
| Rust block 50% or more of the way down | 6 | 9 |
| no usage example found by the heuristic | 46 | — |

**No code at all (11):** `analysis/image-pair-graph`, `analysis/keypoint-reach`,
`analysis/reconstruction-alignment`, `features/cluster-selection`,
`features/flow-based-matching`, `geometry/pose-verification`,
`geometry/reconstruction-growth`, `geometry/relative-pose`,
`patch/fronto-parallel-patch-cache`, `patch/keypoint-localization-consensus-basis`,
`reconstruction/point-correspondence`. Each was opened against failure 2: all
eleven name their entry points in a heading, an Interface / Binding / Inputs
section or a parameter table (for example `relative-pose.md:19-22, 44-57`
names `estimate_essential_rays` and `fit_ray_rotation` with an options table;
`keypoint-reach.md:79-93` gives the binding signature). None is confirmed as
failure 2 at this depth. `flow-based-matching` conveys its interface as CLI
flags only, which meets the requirement for a Python pipeline.

**Rust block at 50% or later:** `patch/member-coherence-validation` 71%,
`analysis/observation-adjacency-graph` 70%, `analysis/adjacency-surfel-normals`
69%, `features/sift` 63%, `geometry/rotation-locked-resection` 61%,
`geometry/affine-factorization` 59%. A reading list, not findings.

**Work-order residue, confirmed by reading:**

| Spec:line | Text | Note |
|---|---|---|
| gui/camera-views.md:196 | "`frustum_size_multiplier` (planned UI slider)" | stale: the slider exists (`viewer_3d/hud.rs:436-447`, "Frustum") |
| gui/point-cloud-rendering.md:321 | "The planned UI label for this effect is **"Target Light Echoes"**." | roadmap; no such label in the code |
| gui/architecture.md:435 | "async loading and an LRU texture cache are planned" | roadmap |
| cli/reconstruction/xform/xform-command.md:493 | "> **TODO (implementation cleanup):** …" | a work item in a standing spec |
| core/geometry/epipolar-estimation.md:307 | "a natural v2 once calibrated consumers exist" | deferral |
| cli/reconstruction/xform/refine-normals-command.md:238 | "so for now it is summarized" | deferral |
| gui/mcp-server.md:5120 | "Kept for now, alongside `window_title` … drop it" | deferral |
| core/patch/keypoint-localization-search-cache.md:388 | "## Stage 2 (follow-up): integer `i16` correlation — investigated, dropped" | experiment history |
| core/patch/keypoint-localization-consensus-basis.md:185, 209, 314 | "Validation (A/B, required before changing the pipeline default)", "Measured (2026-07-26, DnDTabletop)", "Downstream ladder evidence (recorded 2026-07-27)" | dated experiment records |
| core/spherical/spherical-tiles-rig.md:3-8 | "## Motivation … Three options were considered" | design deliberation as the opening (see check 4) |
| Future-work headings | flow-based-matching:144, patch-view-selection:324, sift-to-patch-reconstruction:271, sfmr-file-format:1945, scene-graph:1414, user-experience:261, camera-config:382 | each should become a present-tense statement plus an amendment draft, or be deleted |

Acquitted: algorithm "Step N" / "Phase N" headings (motion, select-by-distribution,
epipolar-curves, photometric-subsets-ransac), format "Versioning and Migration"
sections, present-tense "Consumers" sections (cluster-patches:237,
batch-triangulation-api:969, point-correspondence:114,
per-spherical-tile-source-stack:673), and the other 15 of 16 "Motivation" headings, which state
purpose rather than history. Last run's four grep hits are all resolved.

### 4. Opening paragraphs

174 openings read (the 175 specs less GLOSSARY.md; drafts excluded).

| Flag | Now | 2026-09-26 |
|---|---|---|
| backticked identifier or `::` in the first sentence | 57 (14 start with a code span, mostly `sfm <cmd>`) | 47 |
| link in the first sentence | 11 | 10 |
| first character is a symbol or formula | 0 | 6 |
| describes a change | 0 real (3 flagged: match, bundle-adjust, saving; all false) | 2 |
| opens "This document describes/specifies…" | 0 (`research/blender-…` says "This document details") | 8 |
| no flag | 113 | 113 |

All eight openings proposed last run have landed. Read as a list, five openings
still fail; the first three are the "never says what it is for" kind. Land
each in its own PR, because each proposed sentence below was written from the
spec and spot-checked against the code, and a person who knows the subject
should read it as a claim.

1. **`core/spherical/spherical-tiles-rig.md:5`**
   - Current: "For per-direction work on the sphere (infinity-consistency tests, parallax-from-pose depth estimation, multi-view color aggregation) we need a discretization that samples the sphere in small, nearly-distortion-free patches. Three options were considered:"
   - Problem: design deliberation under § Motivation; it never says what a tile rig is, and two of the three listed uses have no caller (the users are `sfm panorama` and `sfm camrig spherical-tiles`).
   - Proposed: *"A spherical tile rig is a camera rig of n identical small pinhole "tile" cameras that share one optical centre and look in nearly evenly spread directions, so that per-direction work on the sphere, such as the stitching behind `sfm panorama`, runs on small, nearly undistorted images held in one atlas."* (`tile_rig.rs:8-18`)
2. **`gui/edits/move-camera.md:3`**
   - Current: "An image whose pose is wrong is the commonest defect a reviewer can see and cannot fix: …  What the reviewer wants is to take hold of the camera and put it where the photograph lines up, the way a hand would."
   - Problem: the first paragraph states the problem only; what the action is appears in paragraph two, through a figure ("The viewer already has the hand").
   - Proposed: *"Move Camera is a viewer edit that locks an image's camera to the viewport in camera view, so the reviewer moves it with the ordinary navigation controls until its photograph lines up with the points; releasing the lock settles the points that image observes around the new pose and installs the result as the node's next version."*
3. **`core/analysis/keypoint-reach.md:3`**
   - Current: "One question, asked per image of a track set: which other keypoints lie inside this keypoint's own disk? Several rules read that neighbourhood and differ only in what they then test, so the enumeration is stated once and the tests stay with the callers."
   - Problem: precise and true, but it names no rule and no use; the one consumer, covered-by-finer, appears only at :65.
   - Proposed: *"Keypoint reach lists, for each image of a set of tracks, every pair of keypoints in which one lies inside the other's disk, so that a rule such as covered-by-finer, which retires a coarse observation that a finer one covers, reads those pairs instead of searching for them itself."*
4. **`research/blender-viewport-navigation-implementation-overview.md:3`**
   - Current: "This document details how Blender implements precision trackpad/touchpad navigation in the viewport on Windows."
   - Problem: says what, not why it is in this repository.
   - Proposed: *"This note records how Blender reads precision-touchpad gestures on Windows through DirectManipulation, as background for SfM Explorer's own touchpad handling in [windows-precision-touchpad.md](../gui/windows-precision-touchpad.md)."*
5. **`docs/index.md:5`** (landing page, lower weight)
   - Current: "The goal of this project is to make creating and exploring Structure from Motion (SfM) fun."
   - Problem: the page never says what the tool is before the personal note.
   - Proposed: *"SfM Tool is a command-line toolkit (`sfm`) and a desktop viewer (SfM Explorer) for building Structure-from-Motion reconstructions from photographs and inspecting them in 3D."*

Borderline and left alone: `core/bench/bench.md` (says what the bench is in its
fourth sentence), `core/geometry/translation-averaging.md` (premise first, purpose
in the second sentence), `gui/multi-panel-image-browser.md` (opens on docking,
then lists the panels).

### 5. Coverage both ways

**(a) Specs never cited from `crates/`, `src/`, `tests/` or `scripts/`:** 57 of 191
(54 of 170 last run). Matched on the last two path components, so
`edits/bundle-adjust.md` and `reconstruction/bundle-adjust.md` are told apart.

- **CLI specs: 23** (24 last run). Expected: command modules do not link their specs.
- **Drafts: 12 of 16** (8 last run).
- **Other: 22** — `core/analysis/image-pair-graph`, `core/analysis/reconstruction-alignment`,
  `core/camera/projection-jacobian`, `core/features/descriptor-matching`,
  `core/features/flow-based-matching`, `core/features/kdf-layout-measurements`,
  `core/features/optical-flow`, `core/features/randomized-kdtree-forest`,
  `core/geometry/relative-pose`, `core/geometry/reprojection-residuals`,
  `core/reconstruction/point-correspondence`, `formats/colmap-interop`,
  `gui/cross-panel-hover`, `gui/image-animation`, `gui/patch-rendering`,
  `gui/user-experience`, `gui/viewer-3d-bench-layer`,
  `gui/windows-precision-touchpad`, `research/blender-…`, `workspace/workspace`,
  `docs/index`, `docs/tutorials/getting-started`. Newly cited since last run:
  `image-warping`, `sfmtool-pinhole-kernels`, `gpu-optical-flow`,
  `gui/architecture`, `viewport-navigation`, `cluster-selection`.

**(b) Code surfaces that cite no spec** (a file in the surface contains a
`specs/` path):

| Surface kind | Surfaces | Citing no spec |
|---|---|---|
| crates | 11 | 1 — `sfmtool-progress` |
| `sfmtool-core` modules | 11 | 0 (`numeric` and `web_export` now cite) |
| `sfm-explorer` top-level modules | 48 | 14 — `cli`, `colormap`, `context_menu`, `dock`, `elide`, `image_browser`, `lib`, `main`, `metrics`, `platform`, `shaders`, `test_support`, `tests`, `texture` |
| `src/sfmtool` subpackages and top-level modules | 40 | 25 — subpackages `align`, `analyze`, `compare`, `merge`; 21 private modules |
| `src/sfmtool/_commands` modules | 32 | 28 (expected; specs link the commands) |

The private Python modules that no spec names either: `_histogram_utils`,
`_filenames`, `_cli_utils`, `_densify`, `_undistort_images`.

### 6. Format specs that depend on code names (every run)

Counted outside any heading named *Implementations*. Links are to `../core/`,
`../gui/`, `../cli/`, `../../crates/`, `../../src/`; identifier hits are the
skill's list (`::`, `fn `, `pub `, `_sfmr(`, `sfmtool._sfmtool`, "the kernel",
"the solve", "this tool", "the binding"). A wider scan for `sfm <command>`,
repo paths and CamelCase names was read as well.

| File | Impl. section | Links | Identifier hits | Verdict |
|---|---|---|---|---|
| archive-container.md | yes | 0 | 0 | clean (was 9 / 69) |
| archive-io-crate.md | no | 10 | 82 | not a format spec: a crate spec, so code names are its subject (formats/README.md says so) |
| camrig-file-format.md | no | 0 | 0 | clean; `sfm insv2rig` / `sfm pano2rig` at :8-9, :241-242 are examples of rig kinds, acquitted |
| colmap-interop.md | no | 11 | 1 | not a format spec: describes the `sfmtool-colmap` crate's mapping (formats/README.md says so) |
| kdf-file-format.md | yes | 2 | 0 | clean; :25 and :248 are background, acquitted (was 5 / 1) |
| matches-file-format.md | yes | 8 | 0 | **5 findings** (below); links :444, :667, :720, :951, :982, :1069 acquitted as background |
| sfmr-file-format.md | yes | 4 | 0 | **3 findings** (below); :451 and :992 acquitted |
| sfmtool-camera-models.md | yes | 2 | 0 | clean; :135-136 link the enforcement sites after defining the invariant in place (:110-126), acquitted |
| sift-file-format.md | yes | 1 | 1 | clean; :102 cites COLMAP's `Bitmap::CloneAsGrey` beside the stated formula, :173 links `sfm undistort` as an example, both acquitted |

Six of the nine now carry an *Implementations* section (one of seven last
run), which removed most of last run's per-sentence findings; camrig needs
none, and the two crate specs are about code.

**matches-file-format.md**
1. Versioning gap, :223-224: "`version`: Format version number. `1` through `6`". The format is at 7 (:1227; `types.rs:133`). Restate as `1` through `7`.
2. Implementation names in the format proper, :225-235: "the writers in this repository emit: `"exhaustive"`: Exhaustive pairwise matching (`sfm match -e`) … (`sfm match --cluster`, and the viewer's cluster-patches build)". Define each value by what the file holds; move the command for each value to *Implementations*.
3. :442-443: "In sfmtool, [`sfm match --derive-pairs`](../cli/image-feature/match-command.md) is the command that reads a cluster file and writes that pairwise file." Actor should be "a verifier"; the command belongs in *Implementations*.
4. :515-516: "In sfmtool, `sfm match --cluster` writes detection-stage files and `sfm cluster-patches` writes refinement-stage files." Same fix: "a matcher writes detection-stage files; a refiner writes refinement-stage files", commands under *Implementations*.
5. :1342-1344 (§ Versioning): "consumers that export two-view geometries to a COLMAP database (`sfm to-colmap-db` via `src/sfmtool/colmap/db_setup.py`) S-conjugate the canonical poses back to COLMAP convention when building `pycolmap.Rigid3d`." A repo path and library type in the normative text; say "a consumer exporting to COLMAP conjugates by `S`" and move the names. (The :1067 and :1218 uses of `sfm match --derive-pairs` as the verification step are the same pattern as item 3.)

**sfmr-file-format.md**
1. :1921-1924 (§ World-Space Unit): "The five units and their lengths in metres are [`WORLD_SPACE_UNITS`](../../crates/sfmtool-sfmr-format/src/types.rs) (with `world_space_unit_in_metres` to look one up), re-exported by `sfmtool-core`". A repo-path link and two code names in the format proper; the metre lengths are standard, so state them in place (1 mm = 0.001 m … 1 ft = 0.3048 m) and move the names to *Implementations*.
2. :859-861: "Area-averaging (OpenCV `INTER_AREA`), inherited from the `.sift` these are copied from — all four producers (the colmap, opencv and sfmtool extractors, and `sfm undistort`) use it." The resize method is defined; the producer list is implementation. Actor should be "a writer".
3. :153-162 ("Internal round trips need only `S`" under "Invariants … that internal code relies on"): names this repository's pipelines ("bundle adjust, densify, merge PnP, DB-mediated solves"). Implementation guidance in the coordinate-convention section; move to *Implementations*.
   Discussion-grade: :1807-1810 names the viewer's `Go ▸ Go to Point…` inside § Point ID.

**Other failure-6 checks:** no per-element code column without a legend
(`member_status` gained one in version 7); no optional entry without an
introducing version found by the scan; version constants agree with the specs
for sfmr (11), sift (1), camrig (2) and kdf (3).

## Sampled specs

### specs/core/camera/refit-camera-intrinsics.md
**Summary:** The lens-only fit that moves a camera's intrinsics to another model: sample 96×64 rays up to `θ_fit`, project with the source, and fit a spline target by one SVD least-squares solve under a slope-floor (monotone) constraint, or a COLMAP target by Levenberg–Marquardt. Also `refit_spline`, the report and the `CameraIntrinsics.refit` binding.
**Implementing code:** `crates/sfmtool-core/src/camera/refit_intrinsics.rs` (`refit_camera_intrinsics`, `refit_spline`, `fit_spline`, `fit_colmap`), `refit_intrinsics/constrained_lsq.rs`, `camera/report.rs` (`trustworthy_max_theta_deg`), binding `crates/sfmtool-py/src/geometry/camera_intrinsics.rs:540`, caller `reconstruction/switch_camera_model.rs:294-319`.
**Inconsistencies:**
  - Spec :196 "Bundle adjustment never frees it" and Non-goal :480 "Nothing in the toolkit frees it" (principal point) are false: `sfm densify --ba-refine-principal-point` (`_commands/densify.py:45`, `_densify.py:373-375`) frees it through pycolmap, and `BundleAdjustTransform(refine_principal_point=…)` (`xform/_bundle_adjust.py:82,212`) takes the option. Code right; say "sfmtool's bundle adjustment never frees it".
  - `DroppedTerm::FocalAspect` prints `"fx/fy aspect {fy_over_fx:.4} …"` (`refit_intrinsics.rs:311`); the spec defines the value as `fy / fx` (:368) and copies the wrong label into its example (:182). Code bug in a string that reaches Python `report["dropped"]`.
  - `fit_spline` doc says "one solve of the normal equations" (`refit_intrinsics.rs:1079-1080`); the code uses SVD (:1145-1148), as the spec says (:386). Fix the doc.
  - `SIMPLE_RADIAL_FISHEYE` is missing from the spec's list of models with no trusted bound (:202-205), though `trustworthy_max_theta_deg` returns `None` for it (`report.rs:541`), so the fold check at `refit_intrinsics.rs:649` is skipped. Spec gap and possible code gap.
  - Spec :429-431 lists five `theta_fit_source` values for `CameraIntrinsics.refit`; that method produces only three (`refit_intrinsics.rs:578-584`), and the binding docstring (`camera_intrinsics.rs:528`) lists four. The others come from `switch_camera_model`. Update the spec.
  - Quoted figures no test pins: f ≈ 129.52 (:381) vs the test's 129.56 ± 0.1 (`tests.rs:157`); tk107 rms/max and 113.2°/113.3° (:324-327) are not asserted (`tests.rs:330` checks `max_px < 5`). Not run. Minor: the `least_squares_with_inequalities` doc names a `u` argument the function does not take (`constrained_lsq.rs:32-35`).
**Third copies:** `MIN_SLOPE` doc (`refit_intrinsics.rs:75-86`, 12 lines), `refit_spline` doc (:672-691, 20 lines), `fit_spline` doc (:1076-1092, drifted) and `SMOOTHING` doc (:66-69) restate the spec; shrink them to contract plus link. The LDP/NNLS derivation is in both `constrained_lsq.rs:4-16,32-41` and spec :396-407; keep it in the code, cut the spec to the choice and its reasons. `sfm-explorer/src/image_detail/intrinsics/axes.rs:651` re-implements core's `spline_domain_deg` (`refit_intrinsics.rs:769`).
**Shape:** no failures.
**Non-goals / deferrals checked:** 4. Non-goal 1 (principal point) is false, as above; the other two Non-goals hold; the open question on the regularization weight is still open.
**Recommendation:** update spec (principal point, SIMPLE_RADIAL_FISHEYE, `theta_fit_source`) and update code (the "fx/fy" label, the `fit_spline` doc).
**Unclear / incorrect / suspicious:** a `SIMPLE_RADIAL_FISHEYE` target has no trusted bound and no monotonicity check, so a wide refit can return a camera that folds inside the fitted range; `report.rs:493-495` says this model has "nothing to distrust", but `θ·(1+k1·θ²)` folds when `k1 < 0`.

### specs/core/geometry/absolute-pose.md
**Summary:** The Lambda Twist P3P solver (`p3p_solve`), the seeded RANSAC estimator over angular residuals with local optimization (`estimate_absolute_pose`), the trimmed-LM pixel refiner (`refine_absolute_pose`), their bindings and test requirements.
**Implementing code:** `crates/sfmtool-core/src/geometry/absolute_pose.rs` (`p3p_solve`, `kabsch`, `AbsolutePoseOptions`, `estimate_absolute_pose`, `local_optimize`), `geometry/pose_refine.rs`, bindings in `crates/sfmtool-py/src/geometry/`. Consumers: `reconstruction_growth.rs:263-304`, `resect_images/finite.rs:29`, `pose_verification.rs:604`.
**Inconsistencies:**
  - Spec :138-145 says the result is "the best *refit* pose". The code keeps a refit only when it strictly grows the inlier count (`absolute_pose.rs:489`) and otherwise returns the raw P3P pose (:494-498), which is the common clean case; the comment at :495 contradicts `new_count > count`. Code bug: accept on `new_count >= count`.
  - Spec :139-140 says "squared angular residuals"; the code minimizes `sin²θ` (`absolute_pose.rs:529-536`). Update spec.
  - Spec :188-191 says only `EQUIDISTANT_FISHEYE` among fisheyes has an analytic Jacobian; `supports_pixel_jacobian` is also true for `SIMPLE_RADIAL_FISHEYE` and `SFMTOOL_FISHEYE` (`camera/intrinsics.rs:496-502`). The `project_with_jac` doc (`pose_refine.rs:88-91`) is stale too.
  - Consumers (:17-20) name a relocalization consumer that does not exist; merge uses `pycolmap.estimate_and_refine_absolute_pose` (`merge/pose_refinement.py:114`). Bindings (:232-234): the binding also derives the angular threshold from `camera` for (N, 3) input (`sfmtool-py/src/geometry/absolute_pose.rs:172-186`), and `AbsolutePoseOptions::default()` (`absolute_pose.rs:333-344`) is undocumented, and `reconstruction_growth.rs:263-266` relies on it.
  - Nits: termination is "reach", not "exceed" (:149 vs :412); the differential test is synthetic, not "real correspondence sets" (:268-270); out-of-domain residual is `(1e6, 0)`, one component (`pose_refine.rs:179`).
**Third copies:** the Kabsch rank rationale is in spec :76-85, the `KABSCH_RANK_EPS` doc (:35-39) and an inline comment (:274-278); cut the inline comment to a pointer.
**Shape:** failure 3 (no example call, no reason for the bearings/pixels split); failure 4 (:87-92 describes the body); failure 5 (imperatives at :64-74, "Non-goals (v1)"); failure 7 ("polish" :157, :279; "hopeless" :15).
**Non-goals / deferrals checked:** 3; none overtaken.
**Recommendation:** update code (local-optimization acceptance and its comment), then update spec.
**Unclear / incorrect / suspicious:** `GLOSSARY.md:118,213` defines **bearing** as a track or point at infinity, but this spec uses it for an observed unit ray throughout its API. `resect_images/finite.rs` runs its own P3P RANSAC loop, a second estimator this spec does not mention.

### specs/core/features/flow-based-matching.md
**Summary:** Sequential matching that advects SIFT keypoints through adjacent-frame DIS flow over a sliding window (default 5) and accepts the best-descriptor candidate among K=5 nearest keypoints within 10 px, L2 <= 250, with target-side deduplication. Also carries the Seoul Bull measurements, a cost table and future directions.
**Implementing code:** `src/sfmtool/feature_match/_flow_matching.py` (`flow_match_sequential` 163-375), `feature_match/_run.py` `_run_flow_matching` 422-514, `colmap/db_setup.py:117-128`, `features/feature_match/descriptor.rs:167`, `optical_flow/flow_field.rs` `advect_points`, `crates/sfmtool-py/src/flow/optical.rs` `compute_optical_flow` 59-98.
**Inconsistencies:**
  - Likely code bug: sequence order. `expand_paths` returns `Path.rglob` order unsorted (`_filenames.py:58-80`), passed through `match.py:289-291` and `solve.py:265`; on ext4 adjacent frames can be arbitrary images, and rig subdirectories are concatenated and flowed across the boundary. Sort in code; state the order rule in the spec.
  - The descriptor threshold "(default 250)" (:119) has no knob (`_run.py:456-462`); `--flow-skip` and `--flow-preset` (`match.py:120-124`) are not named. Update spec.
  - The cost table (166-167) omits GPU auto-selection (`optical.rs:37`) and the background-thread pipelining (`_flow_matching.py:253-270`). Measurements contradict: :52-53 says L2 <= 100, :63-64 and :129-130 say 250; 0.58 px median vs 0.56 px in the table.
  - Memory claim (:107-108) omits the stored original positions and (N,128) descriptors (`_flow_matching.py:235-237, 309`); "Per-pair matching" (:110-121) describes `_flow_match_pair`, which only `tests/matching/test_flow.py:11` calls.
  - Low: no descriptor "fallback" exists (:204-208; `match.py:238-246`); `sfm match --flow` writes a `.matches` file, not the database (:10-12; `_run.py:203-213`).
**Third copies:** `_flow_matching.py:4-17` module docstring (14 lines, with history residue) and :174-177 re-derive the design; shrink to contract plus link. `optical-flow.md:289` repeats the error-accumulation derivation (:139-140).
**Shape:** failure 2 (partial: no parameters, return type or CLI flags for `flow_match_sequential`); failure 5 ("Future Directions", "Relationship to Existing Pipeline").
**Non-goals / deferrals checked:** 6; none implemented or overtaken.
**Recommendation:** update code (sort the sequence) and update spec.
**Unclear / incorrect / suspicious:** `optical.rs:50-51,107-108` docstrings allow "uint8 or float32" but the signature takes `u8` only; images of different sizes fail mid-run (`optical.rs:71-75`).

### specs/cli/image-processing/flow-command.md
**Summary:** `sfm flow IMAGE1 IMAGE2` computes DIS flow, advects IMAGE1's SIFT keypoints, prints hit statistics, optionally draws images, and with `-r` compares flow hits to a reconstruction's shared tracks. All option defaults match the Click command.
**Implementing code:** `src/sfmtool/_commands/flow.py` (14-171); `src/sfmtool/visualization/_flow_display.py` (`draw_flow_visualization` 131-378, `_get_shared_feature_pairs` 83-128).
**Inconsistencies:**
  - Prerequisites undocumented: `.sift` files for both images (`_flow_display.py:199-205`) and identical dimensions (`optical.rs:71-75`).
  - Flow-only mode (spec 33-34) draws only hits with a categorical palette (`_flow_display.py:443-451`), not flow-coloured arrows; direction colour is only in the separate `<stem>_flow<ext>` image (661-677). The Click help (flow.py:103-105) has the same error.
  - `--descriptor-threshold` changes printed statistics only (`_flow_display.py:268-285`), and its default 100 differs from the matcher's 250 (`_flow_matching.py:167`).
  - `--max-features` affects drawing only; comparison mode can exceed N by up to 2 (527-533). Ranges and errors (flow.py:51, :58, :141-144; `_flow_display.py:107-110`) are undocumented.
  - Code bug: comparison mode matches images by basename, last match wins (`_flow_display.py:101-105, 288-289, 484-485`). In `kerry_park_ground_truth.sfmr`, `fisheye_left/frame_01.jpg` resolves to `fisheye_right/` and the wrong tracks are compared silently. Match the workspace-relative path; refuse ambiguous basenames.
**Third copies:** the colour legend is in four places (spec 40-42, Click help flow.py:108-112, docstrings 150-158 and 476-479); the two internal docstrings should point at the spec. Low priority.
**Shape:** failure 1. Proposed first sentence: "The flow command is a diagnostic for flow-based matching: it computes dense optical flow from one image to another, moves the first image's SIFT keypoints along it, and reports how many land near a keypoint of the second image, optionally drawing the result or comparing it with the matches in a reconstruction." It also never links `flow-based-matching.md`.
**Non-goals / deferrals checked:** 1; still true.
**Recommendation:** update code (basename lookup) and update spec (prerequisites, flow-only drawing, option scope, purpose sentence).
**Unclear / incorrect / suspicious:** "Middlebury color wheel" (spec 34, `_flow_to_color` 59) is really an HSV direction wheel (69-77); `--side-by-side` rescaling (689-699) is undocumented.

### specs/core/features/cluster-covisibility.md
**Summary:** The pre-reconstruction image-pair count matrix `W[i,j]` (clusters with an accepted member in both images), its acceptance mask, dense-storage bound, the lazy greedy seed-group iterator, candidate ranking and the PyO3 surface.
**Implementing code:** `crates/sfmtool-core/src/features/cluster_match/covisibility.rs` (`ClusterCovisibility`, `SeedImageGroup`, `MAX_DENSE_IMAGES`), binding `crates/sfmtool-py/src/matching/covisibility.rs`; consumers `reconstruction_growth.rs:398,969`, `cluster_census.rs:549`.
**Inconsistencies:**
  - Spec :323-325 and `covisibility.rs:454-455` say a pre-v6 cluster file opens without positions; `read_matches` refuses it (`sfmtool-matches-format/src/read.rs:138`). Code right.
  - The acceptance table is labelled "(v4)" (:65) but lists v6 and v7 channels (`types.rs:70-81`). The complexity bound (:79-84) should use `d + 1` (or `d + 2`) images per cluster, not `d` (`cluster_match/mod.rs:339,376-386,422-448`).
  - `cov.counts  # … errors above dense bound` (:281): the getter has no error path (py `covisibility.rs:213-224`). Minor: :51-52 "for unrefined clusters", but the code deduplicates every cluster (core :362-363).
  - The API block omits `next_seed_image_group` (core :591) and `displacement_neighborhood()` (core :561).
  - The Validation section's first consumer `exp_pinhole_bootstrap.py` (:347-349) is not in the tree; name the real consumers.
**Third copies:** `SeedImageGroup` docs (covisibility.rs:145-209, about 45 lines) restate spec :191-233 and :261-267; shrink to contract plus link. Smaller: `MAX_DENSE_IMAGES` doc (:27-32), `prof.rs:13-17`.
**Shape:** failure 1 (borderline). Proposed: "Cluster covisibility counts, for each pair of images, how many feature-match clusters (sets of matching SIFT features across images, stored in a `.matches` file) have a member in both images, so a caller can choose groups of mutually overlapping images and rank candidate views before any reconstruction exists." Failure 5: plan and experiment language at :86, :235, :347-355. Failure 7: "drop the rest unpaid" (:155), "pays nothing for it" (:197), "the actual scaling wall" (:92), "compact where dense is hopeless" (:98).
**Non-goals / deferrals checked:** 7; all still unbuilt, none overtaken.
**Recommendation:** update the spec and the core `from_matches` doc; the code is right.
**Unclear / incorrect / suspicious:** the seed-group iterator, the spec's headline query, has no consumer outside tests and its binding; worth discussing whether it is still needed.

### specs/core/features/randomized-kdtree-forest.md
**Summary:** The in-memory randomized kd-tree forest: random split among the top-D variance axes, shared best-bin-first queue with a soft `L_max` budget, AVX2/SSE2 `u8` kernels, presets, calibration, progress and cancellation, and the `KdForest` Python class.
**Implementing code:** `crates/sfmtool-core/src/features/kdforest/` (`mod.rs`, `build.rs`, `search.rs`, `distance.rs`, `calibrate.rs`); binding `crates/sfmtool-py/src/spatial/kdforest.rs`; consumer `cluster_match/mod.rs:198`.
**Inconsistencies:**
  - Code bug: the spec keeps "dist_sq <= max_dist²" (:127, :154), but `u8::cutoff_sq` rounds up (`distance.rs:78-85`), so `max_dist=2.5` admits √7 ≈ 2.65 (`search.rs:214`). Reachable from Python `KdForest.query` (py kdforest.rs:257). Floor with a small relative tolerance.
  - `L_max` default "precision-tuned" (:190) is a fixed 128 (mod.rs:104-120); `calibrate_max_leaf_checks` has no caller outside the module.
  - Pseudocode sends `diff == 0` right (:145-146); the code sends it left (search.rs:351-354), matching the spec's own rule (:99). Module tree (:271-280) omits `constellation.rs`, `neighbor_index.rs`, `persistent.rs`; "per-dimension bookkeeping" (:300) is not a field.
  - "exactly what `src/sfmtool/feature_match/` already consumes" (:359-360) is false (also py kdforest.rs:8-10); the benchmarks are synthetic, not "end-to-end image-pair matching" (:404-405).
  - `SFMTOOL_KDFOREST_STATS` is on for any value including `0` (mod.rs:79-80); `SFMTOOL_KDFOREST_NO_SIMD` (distance.rs:186-187) is undocumented; the Python class exposes more than the constructor and `query` (py :153-413).
**Third copies:** no kdforest source links this spec. search.rs:14-24 (11 lines) is nearly the spec's :159-170; the progress-unit rationale is in mod.rs:163-179 (17 lines), build.rs:28-42 and spec :315-323; also search.rs:32-39 and calibrate.rs:23-27.
**Shape:** failure 2 (partial: API at :308 of 414, no example); failure 4 (module tree, stale); failure 5 ("_Future:_" :377-381, :296-297, :22; `KdForestF32` "Phase 2" mod.rs:507-509); failure 7 ("slots in" :360, "dial"/"knob" :198/:172, "dwarfs" :251-252).
**Non-goals / deferrals checked:** 7. "Potentially for patch matching" is overtaken in part: bench searches use the forest through `LazyKdForestU8` (`bench/search.rs:30-31`, `bench/nearby/{guided,constellation}.rs`).
**Recommendation:** update code (`max_dist`), update spec (`L_max`, module tree, consumer and benchmark claims), and link the spec from the doc comments.
**Unclear / incorrect / suspicious:** "each rayon worker keeps one reusable scratch" (:237-239), but `for_each_init` (mod.rs:440-441) can initialise once per job split.

### specs/core/geometry/bundle-adjustment.md
**Summary:** The staged soft-L1 LM bundle adjustment kernel: trim/retriangulate/solve rounds, the in-front floor, per-camera lens releases, points at infinity, point constraints (free/ranged/held), inverse-depth free points with the point-or-bearing decision, and protected observations. Interface, defaults and per-camera mechanics match the code; the Non-goals section still describes the kernel as it was before #400 and #607.
**Implementing code:** `crates/sfmtool-core/src/geometry/bundle_adjust.rs` (`bundle_adjust` :824, `BaCameras` :421, `CameraRelease` :447, `Lens::new` :1757, `retriangulate_round` :1187, `bundle_adjust_staged` :2889); binding `crates/sfmtool-py/src/geometry/bundle_adjust.rs`; callers `rotation_init.rs:779`, `reconstruction_growth.rs:1209,1326`, `reconstruction/bundle_adjust.rs`, `src/sfmtool/xform/_bundle_adjust.py`.
**Inconsistencies:**
  - Non-goal "Releasing some cameras and not others" (:1773-1774) is false since #607 (`bundle_adjust.rs:437,447`, applied :2928-2939; `reconstruction/bundle_adjust.rs:73`; `--bundle-adjust cameras=0+1`, `_bundle_adjust.py:67-74`), and contradicts the spec's own :136-145, :462-467. Delete the bullet. Non-goal "Gauge fixing, covariance estimation, or constraint handling" (:1780-1781) is partly false: held and ranged constraints shipped in #400 (`PointConstraints`, :150), and :1533-1535 says ranged points fix the scale gauge.
  - "serves the bootstrap experiments" (:1782-1783) is out of date: the kernel backs `sfm xform --bundle-adjust` on spline cameras (`_bundle_adjust.py:111-112,146`), the viewer, MCP `bundle_adjust` (`mcp/tools.rs:368`) and growth.
  - Progress line: spec :939 quotes `free points decided at 0.412 px: ...`; the kernel emits `free points decided at noise {s:.3} px: ...` (:3179); the CLI prints the spec's wording (`_bundle_adjust.py:52`).
  - `free_point_decision` is also `None` on a degenerate exit under the crossing (:3058); the spec says only with the crossing off (:112, :594, :919-920).
  - Dangling references to a bootstrap spec and experiment scripts (:166, :185, :199-200; module doc :10-12).
**Third copies:** `Lens::new` (:1757-1779, 23 lines) re-derives spec :286-299; `retriangulate_round` doc (:1160-1185, 26 lines) restates :996-1025; `BundleAdjustOptions::releases` (`reconstruction/bundle_adjust.rs:64-71`) repeats :355-362. The code comments should shrink.
**Shape:** failure 4 (minor, :180-182); failure 5 (:166, :276, :162-163, history aside :734-737); failure 7 ("a traverse of a nearly-flat valley" :426-427, "buying it back with geometry" :303-304, "protection is vouching, not a safety net" :1725-1726; "rung" is not in GLOSSARY.md). About 400 lines (:1097-1496) measure a removed alternative; consider a report.
**Non-goals / deferrals checked:** 7; 3 overtaken (per-camera release #607, constraint handling and scale gauge #400, "serves the bootstrap experiments"). The kept-fraction follow-up (:1256-1257) is still unimplemented.
**Recommendation:** update spec (Non-goals bullets 2, 5, 6; the degenerate-exit `None`; the progress-line quote; the bootstrap references).
**Unclear / incorrect / suspicious:** a degenerate exit after the first round returns the state as of that round (:3053-3059), not the caller's input as :198-200 says. No code bug found; every stated default matches.

### specs/cli/reconstruction/xform/localize-keypoints-command.md
**Summary:** `sfm xform --localize-keypoints` runs the cross-view keypoint search over each point's full track on an `embedded_patches` reconstruction, drops refused views, culls points below `min_views`, and rebuilds the tracks through `compact_to_embedded_patches`. Documents the `key=value` parameter string, a 17-row key/default table, errors and the summary. Every key, caster, default, error path and the summary format match the code.
**Implementing code:** `src/sfmtool/xform/_localize_keypoints.py` (`LocalizeKeypointsTransform`); `src/sfmtool/xform/_arg_parser.py` (`_LOCALIZE_KEYPOINTS_KEYS` :277-295, `parse_xform_args` :721-780); `src/sfmtool/_commands/xform.py:95-110`; `src/sfmtool/_patch_compaction.py`; binding `crates/sfmtool-py/src/patches/localize_keypoints.rs:162-170`.
**Inconsistencies:**
  - Spec :109-111 says "the CLI re-specifies nothing and the two layers cannot drift"; `_localize_keypoints.py:69-86` hardcodes all 17 defaults and passes them at :178-198 (docstring :40-41 repeats the claim). All match the binding today. Change the code to forward only the given keys.
  - Normals "carried over per survivor" (:36-38): `_patch_compaction.py:198-213` re-derives them from the frame. Code right.
  - :93-97 credits Click with parsing the optional value; `parse_xform_args` does it (`_arg_parser.py:763-772`), Click only cross-checks (`xform.py:467`).
  - The `sampler` cell (:127) copies cost figures from `image-warping.md:886-890`, as do five other files; link instead.
**Third copies:** `_localize_keypoints.py:40-41` (false claim) and :53-59 (basis-cap rationale, 7 lines); cut to contract plus link.
**Shape:** failure 4 (minor: :52-60, :66-70 name implementation paths); failure 7 ("turns nothing out" :122, "photometric basin" :151, "sharpen" :152).
**Non-goals / deferrals checked:** no Non-goals section; 5 negative claims checked, all hold (normals re-derived, as above).
**Recommendation:** update code (forward only given keys), then the spec's normals sentence, parser mechanism and sampler cell.
**Unclear / incorrect / suspicious:** `sfm xform in.sfmr --localize-keypoints out.sfmr` takes `out.sfmr` as the parameter string (`_arg_parser.py:768-770`); the spec's syntax is correct but does not warn about this.

### specs/core/patch/zncc-self-similarity-radius.md
**Summary:** Defines the ZNCC self-similarity radius, the semi-major axis of the moment ellipse of the whole-pixel shifts where a bitmap's ZNCC with itself stays within `τ`, read from the bitmap alone; covers flags, mapping to image px and patch units, the kernels and every consumer. 500 lines. Every default, constant, threshold, panic condition, binding key and the Track View colours match.
**Implementing code:** `crates/sfmtool-core/src/patch/self_similarity/{mod.rs, overlap.rs, kernels.rs, ellipse.rs}`; `patch/keypoint_localize.rs:534`; `cluster_refine/mod.rs:318`; `bench/track.rs:833`; binding `crates/sfmtool-py/src/patches/self_similarity.rs`; `src/sfmtool/xform/_filter_by_zncc_self_similarity_radius.py:19`.
**Inconsistencies:**
  - Spec :432 "a *self-sim. px* box"; the code says "px whole" (`track_view/body/mod.rs:1851`), as `GLOSSARY.md:54` does. Fix the spec.
  - :404 "3 µs of the parts' 36 µs" was left behind when #772 changed :407 to 45 µs. Fix :404. Also, :56 and :114-115 say the surface is 1 at the centre; a flat template's is NaN throughout (`mod.rs:96-98`, :259-266).
**Third copies:** `keypoint_localize/params.rs:173-185` (13 lines) gives stale tuning figures (80.4%, 73.8%) against spec :347 (81.0%, 76.2%); `bench/track.rs:823-832`; `ellipse.rs:266-276` (`LENGTH_SLACK`) re-derives :74 nearly word for word; `overlap.rs:259-268` repeats :404. Shrink the code copies. The :432 paragraph (about 20 lines) duplicates track-view.md, mcp-server.md and editable-track.md.
**Shape:** failure 5: :347 and :417 report figures under a rule that no longer exists (:417 cull counts 59/837, 292/1,886, 69/765, 316/1,795); remeasure or delete. Failure 7: "turns out" for "rejects" at :334, :417, :430, :481. Failure 4 (mild): test fixtures at :404.
**Non-goals / deferrals checked:** 6 (3 Non-goals, 3 Open questions); none overtaken.
**Recommendation:** update spec (label, 36 µs, flat-surface exception, disk-rule figures), then shrink the doc copies.
**Unclear / incorrect / suspicious:** the :351 table was measured before the ellipse (#753 predates #772); state the reading in its caption.

### specs/core/geometry/pose-verification.md
**Summary:** The displacement-neighborhood substrate (per covisible pair: shared-cluster count, mean displacement) and two kernels: `verify_poses` (Screen A self-resection; Screen B homography vs pose-implied rotation) and `repair_poses`. Every table default matches the code.
**Implementing code:** `crates/sfmtool-core/src/geometry/pose_verification.rs`, `features/cluster_match/covisibility/displacement.rs`, `geometry/batch_resection.rs`; bindings `crates/sfmtool-py/src/geometry/pose_verification.rs`, `crates/sfmtool-py/src/matching/covisibility.rs`. Only tests call `verify_poses`/`repair_poses`.
**Inconsistencies:**
  - `INLIER_PX` (:187) "shared by the screens and repair acceptance": only the repair uses it (:188, :603, :612); Screen A uses batch_resection's 3 px (`batch_resection.rs:54-55`), Screen B `HomographyOptions::default()` (`homography_estimation.rs:193`). Doc at :40-41 says the same. Fix both texts.
  - :160-162 "or construct it on the fly": neither kernel builds the substrate (:378, :504; py :89, :186).
  - Testing line :213-214 ("flagged by both screens") contradicts :90-94 and the test (`tests.rs:304-305`). Test right.
  - Purpose :12-14 includes "thinning", which reads other tables (`covisibility/selection.rs:19-38`). Drop it. :106-108 "used everywhere in this codebase": `orthonormalized` (`rotation.rs:84-97`) is a deliberate alternative; link `polar_rotation` (:75). Repair skip conditions (:136-138) omit `polar_rotation` returning `None` (:587-589); Screen B (:99) uses `nearest_registered` (:238-253), not `nearest`.
**Third copies:** binding `verify_poses` docstring (py :127-138, 12 lines) and the repair algorithm in the Rust doc (:485-493) and binding doc (py :256-263) restate spec :116-122 and :126-144; shrink the binding copies. Orientation contract in `displacement.rs:41-52` and spec :41-53; cut the spec's restatement (:49-53).
**Shape:** failure 2 (no Rust signatures; binding inputs and dict return only in py :141-186); failure 3 (no example; shortest call is `d = cov.neighborhood_arrays()` then `verify_poses(..., d["i"], d["j"], d["count"], d["mean_magnitude"])`); failure 4 (:63-65); failure 7 ("The ruler" :7-8, "hold the current poses against it" :10, "put them back" :6).
**Non-goals / deferrals checked:** 6; none overtaken.
**Recommendation:** update spec (`INLIER_PX`, "on the fly", testing line, "thinning"; add an interface block with one call).
**Unclear / incorrect / suspicious:** the comment at :392-393 ("its own current pose available as a fallback init") is false except above 4096 images (`covisibility.rs:525-537`; `batch_resection.rs:178-183`). `inlier_fraction_of` (:171-192) duplicates growth's helper and `reprojection::inlier_fraction`. The scaling testing line (:202-203) has no test. Nothing in production calls the kernels.

### specs/core/geometry/reprojection-residuals.md
**Summary:** `reprojection_residuals` (per-observation `(projection − observed)` for images sharing one `CameraIntrinsics`, invalid observations as `(invalid_residual, 0)`) and `inlier_fraction`, with bindings. Signatures and the binding default match.
**Implementing code:** `crates/sfmtool-core/src/geometry/reprojection.rs` (:35-89, :94-103); `crates/sfmtool-py/src/geometry/reprojection.rs`. Only production caller: `analysis/cluster_census.rs:682`. No code file cites the spec.
**Inconsistencies:**
  - Purpose (:5-6) names three shared callers that each have their own code: growth (`reconstruction_growth.rs:194, :211-230, :501-530`), pose refinement (`pose_refine.rs:45-88`), pose verification (`pose_verification.rs:171-192`). Fix the spec, or move those callers onto this function (discuss).
  - Bug: zero observations give shape `(0, 0)` (`PyArray2::from_vec2`, py :90), not `(n_obs, 2)` (:83), and `inlier_fraction` then refuses it (py :105) where :62-63 says 0.0. Verified by running.
  - Bug: out-of-range `obs_image`/`obs_point` and a short `translations` array raise `PanicException` (py :59-78; core :64-76), not `ValueError`. Verified.
  - A non-finite pose gives `(NaN, NaN)`, not `(invalid_residual, 0)` (core :69-86); the spec (:46-53) should say so.
**Third copies:** invalid-observation rationale in spec :46-53, core doc :26-31, binding doc py :18-21; shrink the binding copy and link the spec from the module doc (:4-10).
**Shape:** failure 1. Proposed: "This function computes, for every observation of a world point in an image, the pixel offset between where the point projects under that image's pose and where it was observed, for a set of images that share one camera model." Failure 3: no reason for flat `&[f64]` inputs; the Python block is a signature, not a call.
**Non-goals / deferrals checked:** 2; neither overtaken.
**Recommendation:** update code (empty shape; `ValueError` on bad indexes) and spec (callers; cite it from the code). Discuss moving the other helpers onto it.
**Unclear / incorrect / suspicious:** :50-51 and :97-99 distinguish `inf` from a finite value in `inlier_fraction`, which treats both alike (:99-102). Four separate helpers project, take the norm and count below 3 px; none calls this function.

### specs/gui/mcp-server.md
**Summary:** The viewer's MCP endpoint (`sfm-explorer --mcp`): CLI and bind behaviour, the 86-tool catalog with argument and reply shapes, the GUI-thread drain, the Rust seam, transport, security, errors and tests. 5167 lines, updated with the code (#804, #807, #809). The tool table matches the catalog name for name (86 each; 16 read, 65 write, 4 input, 1 save), checked by `mcp/tests/catalog.rs:865,964`. Drift sits in prose no test reads.
**Implementing code:** `crates/sfm-explorer/src/mcp/` — `tools.rs` (`parse`), `tools/catalog/*.rs`, `mod.rs` (`Command`, `apply_with_window`), `view.rs`, `display.rs`, `edit.rs`, `bench.rs`, `layout.rs`, `frame.rs`, `logged.rs`, `server.rs` (`serve`).
**Inconsistencies:** (ranked by what an agent would get wrong)
  - Code bug: `set_view` drops `fov_short_axis_deg` beside `fit`, `look_through` or `exit_camera_view` (`tools.rs:794`, :796-848); only `point`/`bench_observation` refuse it (:815-822). Spec 1391 is right.
  - Code bug: an explicit placement with fov outside 5-160 moves the camera (`view.rs:257-282`), then refuses (`view.rs:283,505`). Spec 1541 is right. Every `set_view` also ends a held Move Camera lock before validating (`mod.rs:2288-2290`); say which is intended.
  - Code bug: a background edit ending in `Finished::NoChange` (`background/mod.rs:1043`) replies `changed: true` (`mcp/edit.rs:584-590`), against spec 3073/3276 and `mcp/edit.rs:22-24`.
  - Spec 2725-2726, 2949, 3018, 3064, 3181, 3230 and `catalog/edit.rs:121,145,244,307` say five edits "renumber nothing"; with pending point edits each calls `edited.materialize()`, which closes deleted slots (`edited.rs:1162-1175`; `state/edits.rs:1379-1386`). Code right; after `delete_point` an agent reuses stale indexes.
  - Security item 4 (4435-4436) "No tool in this surface saves an `.sfmr`" is false: `save_reconstruction` (`catalog/edit.rs:50`), as spec 203 and 223 say.
  - Panel counts: `Tab::ALL` has 9 (`layout.rs:191`); `catalog/viewer.rs:509` and `mcp/layout.rs:107` say seven, spec 2470/2486/5138 say eight.
  - Spec 3145 "Six operations run on a worker" omits add-to-tracks, prune, build_index_files, open, create_track_at_pixel, find_nearby_tracks and the bench searches (`background/mod.rs:86-302`).
  - Reply shapes: no `changed` field in 2689-2696 (`mcp/edit.rs:673,730`); bundle_adjust prose (3077-3153) and `actors` paragraphs (1751-1784) are under the wrong headings; `cancel_background_task` and `fit_bench_track_normal` have no reply section.
  - retriangulate_point verdicts (3215-3216) are a subset of `triangulation/points.rs:181-189`. Spec 3693-3699: `shape_bench_observation` has no `stage_must_be` check (`mcp/bench.rs:777-794`). The `get_bench_track` description says 24 × 24 and 8 × 8 (`catalog/read.rs:288-296`); spec 3545/3600 is right.
  - Late screenshot refusals (`frame.rs:349,438,472,514`) leave only a success `Query` row (`mod.rs:2308-2310`), against "one failure, one entry" (1950-1956, 4468).
  - Stale counts: "thirty-five" (4308/4310, is 85), "Seventy tools" (4397, is 86), "Four things" (3913, six bullets), "seven edit commands" (4179, is 14), three vs two self-wording methods (4022, 4473); `AppState::bundle_adjust` (2677, 4023) is `start_bundle_adjust`; `serve` lacks `busy` (`server.rs:125-130`); the `Command` block (4096-4162) shows about 44 of 83 variants (`mod.rs:96`).
  - Minor: `max_dimension: 0` and `limit: 0` are accepted (`frame.rs:629-631`); a blank line at 5095 splits the Parameters table.
**Third copies:** `get_bench_track` description (`catalog/read.rs:262-398`, 137 lines) repeats spec 3496-3656 and `bench.md` "The wire"; shrink to fields and units, move the #809 fields to `bench.md`. `mcp/bench.rs` doc comments, `read.rs:270-285` (which also misnames `background_task`), about ten 9-27-line blocks in `display.rs`, `view.rs`, `render.rs`, `layout.rs`, and `mcp/edit.rs:29-34,209,355-365` should shrink.
**Shape:** failure 3/4: § "The Rust seam" transcribes `Command`, `Deferred` and `serve`, all drifted; replace with rationale and links. Failure 5: 3919, 4470, the draft pointer at 5000. Failure 7: "closes the loop" (117), "defer honestly" (3927), "too blunt an instrument" (5070), "is being rude" (5123), "the representation the rays earned" (3523), "make a mess of the window" (2536).
**Non-goals / deferrals checked:** 26. One overtaken: 4994-5001 says editing intrinsics is "still not on the surface", but `switch_camera_model` and lens releases ship (`catalog/edit.rs:327,415`).
**Recommendation:** update code for the three bugs; update spec elsewhere, starting with "renumbers nothing", Security item 4, the worker list and the panel counts; replace hand-copied counts and the `Command` transcription with ones the catalog test checks.
**Unclear / incorrect / suspicious:** spec 1157 and 1194 disagree on whether `camera_image` or `bench_observation` wins (`display.rs:517-523`). `FakeWindow` clamps sizes (4499) though a real Windows window does not (`ui_basic.rs:1297-1300`). `catalog/edit.rs:504` carries runs of about 26 spaces. The widget-id parse accepts uppercase hex (`tools.rs:1717`); the schema allows only lowercase.

### specs/gui/track-view.md
**Summary:** The panel that shows one track in one body (`TrackBody`) in Viewed and Edited modes: Edit box, recent items strip, header, toolbar, six threshold boxes, the observation table, row gestures and the wire. 2030 lines; reflects most of #645 through #809. Every constant checked agrees with the code.
**Implementing code:** `crates/sfm-explorer/src/track_view/` (`mod.rs`, `recent.rs`, `header_buttons.rs`, `body/{mod,table,tile,crop,patch,reference}.rs`); `bench.rs` (`set_editing`:1587); `dock.rs` (`show_track_view`:887).
**Inconsistencies:**
  - Non-goal "Deciding anything from a number" (2015-2016) contradicts 1573-1582 and the code (`bench/evaluate.rs:430-437`): the bars set every unpinned verdict. Rewrite the bullet.
  - `TrackBodyResponse` (92-117) lacks `normal: Option<NormalStep>` (`body/mod.rs:156`, #668).
  - Module map (44-59) omits `header_buttons.rs` and `body/reference.rs`; column prose (988-991) omits Reference between Zoom and Status (`body/table.rs:306-308`).
  - Spec 765-766 attributes the Action Log labels (`bench.rs:2422,2426`) to the boxes, which say " per axis" and "% overlap" (`body/mod.rs:781,793`).
  - Cleared-Lock hover (`body/mod.rs:818`) names less than spec 779-783; fix the code text. `apply_thresholds` doc (`body/mod.rs:160`) says four boxes; there are six.
  - Spec 611 links a `metrics/` directory; the module is `metrics.rs`. Spec 1684 "its SIFT index" understates the refresh (`dock.rs:894-902`). Testing (1881-1986) omits the sort tests (`body/tests.rs:4274-4424`); none orders by Reference (948).
**Third copies:** about 220 lines of module docs in `track_view/`; `body/mod.rs` restates 170-196, 568-573; `tile.rs:4-60` restates 1347-1464. `table.rs`, `tile.rs`, `crop.rs`, `patch.rs` do not name the spec.
**Shape:** failure 1 (mild). Proposed: "Track View is the SfM Explorer panel that lists one 3D point's observations, one per image, with the measurements of each, and, with its Edit box ticked, the controls that change that track on the bench before it is committed." Failure 4: 1634-1635, 975-977, and the Testing section (1797-2002) as a prose copy of test names. Failure 7: "only as good as" (3), "a gesture with no answer" (396), "in the same breath" (725), "a hand ruling against the bars" (1543). `frame` for patch geometry (1395, 1430, 1442, 1959) conflicts with GLOSSARY.
**Non-goals / deferrals checked:** 11; one overtaken (above). `specs/drafts/sfm-explorer-track-editing.md` is stale (line 890; σ_pos removed in #654).
**Recommendation:** update spec; the code-side fixes are the Lock hover text and the four-boxes doc comment.
**Unclear / incorrect / suspicious:** code bug: the Lock hover strings contain runs of 18 spaces (`body/mod.rs:814`, `818`), as does `mcp/tools/catalog/edit.rs:504`. `AppState::set_editing` (`bench.rs:1593`) reads the raw `selected_point`; no user gesture found that reaches the bad state (`track_view/tests.rs:328`).

### specs/core/bench/nearby-tracks.md
**Summary:** `find_nearby_tracks`: for a pixel, run the matching sources in order with a stopping rule, run the far-field sweep when needed, group and rank into depth layers, build a track-stage `EditableTrack` per usable candidate, mark duplicates and label the rest. Matches the code (#634-#642) closely, including every parameter default and the example call.
**Implementing code:** `crates/sfmtool-core/src/bench/nearby/find.rs` (`find_nearby_tracks`:670), `candidate.rs`, `layers.rs`, `far_field.rs:860`; binding `crates/sfmtool-py/src/bench/nearby_tracks.rs`; viewer `crates/sfm-explorer/src/bench/nearby_tracks.rs`; wire `mcp/tools.rs:432-438`; harness `scripts/track_at_pixel/anchors.py`.
**Inconsistencies:**
  - Errors (48-49) omit `Label(BenchError)` (`find.rs:461-463`, checked first at :680-682, #702); a label with a control character is refused (`ValueError` in Python, `sfmtool-py/src/bench.rs:65`). § The labels (247) and § Python bindings (301-302) need it. No binding test.
  - "labels sort best-supported first" (256-257) holds for `bench_order()` (`find.rs:410-414`), not a string sort.
  - Spec 26-28 omits the wire's `commit` option (`mcp/tools.rs:438`). Minor. Binding `rank` is `None` when layers are unranked (`nearby_tracks.rs:298`); 304-306 does not say so. Minor.
**Third copies:** the binding docstring (`nearby_tracks.rs:141-151`) narrates the flow and defaults; shrink to contract plus link.
**Shape:** "bench" undefined in the opening's third sentence. Failure 5: 22-26 ("moved into core"), parity paragraph (353-365), timings (210-215), duplicate percentages (234-240).
**Non-goals / deferrals checked:** 5; none overtaken.
**Recommendation:** update spec (the `Label` refusal; move the experiment numbers out).
**Unclear / incorrect / suspicious:** the far-field trigger groups with `DepthLayerOptions::default()` (`find.rs:1089-1092`), not `options.layers`; harmless today, unstated.

### specs/formats/matches-file-format.md
**Summary:** The `.matches` archive: a pairwise or cluster backbone, the hashes, the cluster-selection record, and versions 1-7. The v7 status legend (#805) is described accurately; the one stale statement is the version range. The v7 legend rules, the v6 and v4 presence rules and the pre-v6 refusal (read.rs:138) match the code.
**Implementing code:** `crates/sfmtool-matches-format/src/` — `types.rs` (`MATCHES_FORMAT_VERSION`=7 :133, legend :543-619), `read.rs` (:104-152, :394-440), `write.rs` (:384), `verify.rs` (`structure_errors` :39-145, legend :896-1000), `select.rs`.
**Inconsistencies:**
  - :223-224 "`1` through `6`"; the code (types.rs:133) and spec :1227 say 7. Code right.
  - :1133-1135 requires the entries to match the `has_*` flags; `structure_errors` (verify.rs:39-145) checks stray `cluster_patches/` (:119-130) but not stray `two_view_geometries/`, so such a file passes. Spec right; small verifier gap.
**Format independence (failure 6):**
  - CONFIRMED :225-236 — `matching_method` values defined by CLI flags; define each by what a writer asserts, move the flags to Implementations.
  - CONFIRMED :442 (`sfm match --derive-pairs`), :515-516 (`sfm match --cluster`, `sfm cluster-patches`) — move to Implementations.
  - CONFIRMED :1342-1347 — repo path and `pycolmap.Rigid3d`; rewrite as "a consumer that writes these poses into a COLMAP database conjugates them by `S`".
  - NEW :1111-1121 — `write_matches`/`zstd_level`/`read_matches`/`verify_matches` API names; keep "every entry is written at one zstd level".
  - NEW :1282-1283, :1365-1366, :1300-1301 — migration and verification in command and function names; restate in format terms.
  - NEW :533-538 — focal-vote and kernel clauses in "Why `float32`".
  - NEW :664-669 — `rejected_unlocalizable` defined only through cluster-patch-refinement.md; state the bar as `refine_options.max_member_zncc_self_similarity_radius` and what the radius measures.
  - Minor: :103-106 "earlier sfmtool releases"; :760-761 acceptable; :35 and :1060-1074 acquitted as background.
  - Versioning gaps: the `cluster_selection` record, `source_selection` and `restrict_cluster_ids` (:954-997) give no version; `refine_options` keys (:621-628) are tied to "writer generations", not versions.
**Third copies:** `MATCHES_FORMAT_VERSION` doc (types.rs:67-132, about 65 lines) restates the v1-v7 history; shrink to a spec link plus the v7 line.
**Non-goals / deferrals checked:** 0 present.
**Recommendation:** update spec (version range; move CLI and API names to Implementations) and add the verifier check.
**Unclear / incorrect / suspicious:** with `has_two_view_geometries` true and the entries missing, `verify_matches` returns `Err` (verify.rs:1140) rather than reporting it in the result list (:1149-1151). The "Version 1.0rc1" line (:1385) has no migration statement.

### specs/formats/sfmr-file-format.md
**Summary:** The `.sfmr` archive (versions 1-11): the Z-up / −Z-forward convention, sections and hashes, optional columns and their presence rules, the constraint legend, thumbnails, Point IDs and `world_space_unit`. The lineage retirement (#808) and the presence and unit refusals (#806) match the code.
**Implementing code:** `crates/sfmtool-sfmr-format/src/` — `types.rs` (`SfmrMetadata` :131, `presence_violations` :524-575, `SFMR_FORMAT_VERSION`=11 :631, `WORLD_SPACE_UNITS` :663), `read.rs` (:150-157, :454-462, :803), `verify.rs`, `write.rs` (normal fill-in :152-200, :324-358).
**Inconsistencies:** none in field definitions. Lineage is skipped on read and dropped on save (spec :334-336, :2031-2043); presence rules (:1369-1380, :1592-1594), unknown-unit refusal (:1916-1917) and frames-without-rigs (:589-591) all match.
**Format independence (failure 6):**
  - CONFIRMED :1921-1925 — `WORLD_SPACE_UNITS` repo link and function names, repeating Implementations :1760-1762; delete, or add metre lengths to the :1914 table.
  - CONFIRMED :859-863 — producer list; replace with "a conforming writer resizes by area averaging; a reader that compares against its own resize must use area averaging".
  - CONFIRMED :146-161 — internal round trips (bundle adjust, densify, merge PnP, DB-mediated solves); keep the first invariant, move the second.
  - NEW :1143-1147 — "the built-in writer"; keep the coherence rule, move the fill-in to Implementations.
  - Minor: :163-165 "earlier sfmtool releases"; :545, :1103, :985-993 name implementations or a core-only test; acquitted: :449-451, :92, :566, :1810, :1955.
  - Versioning gaps: `world_space_unit` (:1898-1914) gives no version (present since e881b335, so version 1+); `infinity_point_count` (:324) lacks "(version 2+; read as 0 when absent)" (types.rs:159-162).
**Third copies:** `SFMR_FORMAT_VERSION` doc (types.rs:576-630, about 55 lines) and `SFMR_CANONICAL_CONVENTION_VERSION` doc (:641-652) restate the spec; shrink to contract plus link.
**Non-goals / deferrals checked:** 2 (Future Extensions :1945-1970; reserved `normal_confidence` values :1136); neither overtaken.
**Recommendation:** update spec (remove the four implementation-name passages; add the two introducing versions). The code matches.
**Unclear / incorrect / suspicious:** the writer replaces every all-zero normal on a finite point (write.rs:152-200, :344-358), and the format defines a zero normal only for `w = 0` rows (:1112-1113); state it. "Verification Process" (:1708-1713) lists only hash steps. Migration subsections are out of order, and 1→2 has no "how a v1 file reads" sentence (read.rs:394-407).

## Code without specs

| Surface | User-facing? | Spec |
|---|---|---|
| Workspace: `ws` (group: `ws init`), `pano2rig`, `insv2rig`, `camrig` (group: `create`, `cp`, `spherical-tiles`) | yes | one `cli/workspace/*-command.md` each (`camrig-command.md` covers all three subcommands) |
| Image Feature: `sift`, `match`, `cluster-patches` | yes | one `cli/image-feature/*-command.md` each |
| Reconstruction: `solve`, `inspect`, `analyze`, `compare`, `align`, `merge`, `densify`, `motion`, `embed-patches`, `estimate-intrinsics` | yes | one `cli/reconstruction/*-command.md` each |
| `sfm xform` (shared transforms) | yes | cli/reconstruction/xform/xform-command.md |
| xform `--refine-normals`, `--refine-keypoints`, `--localize-keypoints`, `--find-points-at-infinity` / `--classify-points-at-infinity`, `--scale-by-measurements` | yes | one `cli/reconstruction/xform/*.md` each |
| xform `--include-by-distribution` | yes | xform/select-by-distribution-command.md (the `cli/README.md:76` row misnames the flag) |
| Visualization: `explorer` (spec new since last run), `epipolar`, `heatmap`, `render-patches`, `panorama`, `web-export` | yes | one `cli/visualization/*-command.md` each |
| Image Processing: `flow`, `undistort` | yes | one `cli/image-processing/*-command.md` each |
| COLMAP Interop: `to-colmap-bin`, `to-colmap-db`, `from-colmap-bin`, `to-nerfstudio` | yes | one `cli/colmap-interop/*-command.md` each |
| `sfm version` | yes (trivial) | none (cli/README.md:11 notes it is outside `COMMANDS`); acceptable |
| `_commands/_range_options.py` | no | none dedicated; to-colmap-bin and to-nerfstudio specs describe `--range` / `--filter-points` |
| crate `sfm-explorer` | yes | gui/architecture.md plus 36 other gui specs |
| crate `sfmtool-core` | no | core/<module>/ (11 modules, all cited) |
| crate `sfmtool-py` | Python API | python-bindings.md (index to the per-module specs) |
| crate `sfmtool-progress` | no | gui/operation-progress.md (the crate links no spec) |
| crate `sfmtool-archive-io` | no | formats/archive-io-crate.md |
| crate `sfmtool-sift-format` | file format | formats/sift-file-format.md |
| crate `sfmtool-matches-format` | file format | formats/matches-file-format.md |
| crate `sfmtool-sfmr-format` | file format | formats/sfmr-file-format.md |
| crate `sfmtool-camrig-format` | file format | formats/camrig-file-format.md |
| crate `sfmtool-kdf-format` | file format | formats/kdf-file-format.md |
| crate `sfmtool-colmap` | no | formats/colmap-interop.md |
| py `align` | via CLI | cli align-command + core/analysis/reconstruction-alignment.md |
| py `analyze` | via CLI | cli analyze-command |
| py `camera` | via CLI | workspace/camera-config.md |
| py `camrig` | via CLI | cli camrig-command + formats/camrig-file-format.md |
| py `colmap` | via CLI | formats/colmap-interop.md + cli colmap-interop/* |
| py `compare` | via CLI | cli compare-command |
| py `feature_match` | via CLI | cli match-command + core/features/{descriptor-matching, flow-based-matching, track-cluster-matching} |
| py `merge` | via CLI | cli merge-command + core/reconstruction/point-correspondence.md |
| py `motion` | via CLI | cli motion-command |
| py `rig` | via CLI | workspace/rig-config.md + cli pano2rig/insv2rig/panorama + core/spherical/* |
| py `sift` | via CLI | cli sift-command + core/features/sift.md |
| py `strips` | via CLI | cli compare-command / inspect-command (`--strips`) |
| py `visualization` | via CLI | cli heatmap / motion / epipolar specs |
| py `web_export` | via CLI | cli web-export-command |
| py `xform` | via CLI | cli xform/* |
| py `_histogram_utils`, `_filenames`, `_cli_utils` | no | none (small utilities; acceptable) |
| py `_densify`, `_undistort_images` | via CLI | none names them; behaviour in densify-command / undistort-command |
| format `.sift` / `.matches` / `.sfmr` / `.camrig` / `.kdf` | yes | formats/* (one each) |
| archive container | yes | formats/archive-container.md |
| `SFMTOOL_FISHEYE` / `SFMTOOL_PINHOLE` camera models | yes | formats/sfmtool-camera-models.md |
| `.sfm-workspace.json` | yes | workspace/workspace.md |
| `camera_config.json` | yes | workspace/camera-config.md |
| `rig_config.json` | yes | workspace/rig-config.md |
| viewer layout JSON | yes | gui/panel-layout.md |
| scale-by-measurements YAML | yes | xform/scale-by-measurements-command.md |
| COLMAP binary model / database | yes | formats/colmap-interop.md |
| Nerfstudio `transforms.json` | yes | cli/colmap-interop/to-nerfstudio-command.md |
| web-export output directory | yes | cli/visualization/web-export-command.md |

Every CLI command, crate and on-disk format now has a spec; the gaps left are
the `sfmtool-progress` crate, which links none of the spec text it duplicates,
and five small private Python modules.

## Carried forward from the 2026-09-26 audit

The 2026-09-26 audit is retired in favour of this report. Every finding in it
marked **Partially done**, and every section with no status line, was re-checked
against the tree on 2026-10-06 (`main` at `dd4f6722`, #809). Only the items
still open are listed here; line numbers are current.

### Still open

- **`specs/core/features/lazy-kdforest-query.md`: about 600 lines of
  measurement history.** The spec is 1,365 lines. § Current access path and
  performance diagnosis (:488), § End-to-end held-out images (2026-09-11)
  (:588) and § What the measurements found (:621-1099) are dated benchmark
  history. § Acceptance checks (:1100-1304) still carries the cost model from
  before the build: "Keep the exact same trees" (:1147) and "**Design
  implication.**" (:1235). The fix waits on a **Needs decision**: the proposed
  TEMPLATE amendment that lets a spec keep its dated measurements in a sibling
  `<name>-measurements.md`. `specs/TEMPLATE.md` has no such rule yet, although
  `specs/core/features/kdf-layout-measurements.md` (#761) already follows that
  pattern for the `.kdf` layout study. Every other item on this spec is done
  (#742, #777).
- **`specs/TEMPLATE.md`: proposed "Determinism and precision" section for
  numerical-kernel specs** (from the focal-vote read). Not adopted: the
  template mentions determinism only as one item under § Testing
  (`TEMPLATE.md:186-187`). **Needs decision** by a maintainer. The focal-vote
  spec itself is done (#751, #778).
- **`specs/core/features/kdf-constellation-query.md:458-467`: the
  `image_feature_ids` cost is described but not measured.** The spec says the
  whole-origin-table pass is paid on every `constellation_from_keypoints` call
  and is not cached (`constellation.rs:254`), so each bench descriptor search
  pays it. Nobody has measured it at DinoLedge scale (9.7M origins), where it is
  probably the largest part of interactive search latency. This is a
  measurement to run, not a spec fix. The other items on this spec are done
  (#752).
- **`docs/index.md:23`: the screenshot was never re-checked.**
  `docs/images/sfm-explorer-with-seoul-bull-tiny-images.jpg` was last changed
  on 2026-04-04 (`cb342444`, before PR numbering), which is before most of the
  current panel layout. Retake the screenshot, or confirm it still matches the
  viewer. The text items are done (#743), and the wheel sentence at :95-96 also
  matches the later platform changes (#770, #771).

### Mechanical checks the new run replaces

- **Check 2 (prose duplicated between spec and code):** found 224 shared
  normalized lines and 13 spec↔code pairs with 5 or more. Nearly all were doc
  comments copied into fenced API blocks; the largest was
  `track-cluster-matching.md` ↔ `cluster_match/mod.rs` (23 lines, removed by
  #704). This run's check 2 replaces the measurement.
- **Check 5 (coverage both ways):** (a) found 54 of 170 specs never cited from
  code, 24 of them CLI specs. (b) The "Code without specs" table listed
  surfaces with no spec. Two still have none: `sfmtool-core/src/numeric.rs`
  (the shared median and RNG step) and `_commands/_range_options.py`, which
  only `to-colmap-bin-command.md` describes. The new run's check 5 replaces
  both lists.
- **Check 3 (shape of `specs/core/`)** was a reading list, not findings; the
  new run redoes it.

### Leftovers since fixed

The 2026-09-26 leftovers that are now fixed:
- `sfm explorer` command spec (#766, updated #787).
- Top priority 3, Implementations sections in the format specs: archive-container
  #750, sift #760, kdf #761, matches and sfmr #682 then #776, camera-models #749,
  cluster-selection moved to core #799. `camrig-file-format.md` had no code names
  and needs no Implementations section.
- sift-command third copies #744.
- editable-track third copies and Testing section #755.
- viewport-navigation: righting copies and step lists #757; DirectManipulation
  spec split out #800.
- focal-vote: opening, interface and third copies #778.
- lazy-kdforest misplaced doc comment and its other items #777.
- sfmr: presence rules and `world_space_unit` #806; lineage removed #808.
- matches: third copies #779; `member_status` legend #805.
- cluster-selection: gaps #775; move to `core/features/` #799.
- affine-factorization: kept as a library module #798.
- `sift.md` pipelining section moved to `sift-command.md` #801.
- Lower-priority openings: #781 and #782.

### Specs read by the 2026-09-05 and 2026-09-26 audits

These are the specs those two audits read against their code. Each one is kept
as a heading so the `audit-specs` sampler still counts it as audited. The
2026-09-05 audit also read `gui/point-track-detail.md`, which has since been
replaced by `gui/track-view.md`. `specs/core/features/cluster-selection.md` was
read at its old path, `specs/formats/cluster-selection.md`.
`specs/gui/mcp-server.md`, `specs/formats/matches-file-format.md` and
`specs/formats/sfmr-file-format.md` were re-read in this run and appear under
**Sampled specs**.

### docs/index.md

Read 2026-09-26; open items above.

### specs/cli/colmap-interop/from-colmap-bin-command.md
### specs/cli/colmap-interop/to-colmap-bin-command.md
### specs/cli/image-feature/match-command.md
### specs/cli/image-feature/sift-command.md
### specs/cli/reconstruction/inspect-command.md
### specs/cli/reconstruction/motion-command.md
### specs/core/analysis/cluster-census.md
### specs/core/bench/editable-track.md
### specs/core/camera/epipolar-curves.md
### specs/core/camera/ray-grid-projection.md
### specs/core/camera/sfmtool-pinhole-kernels.md
### specs/core/features/cluster-selection.md
### specs/core/features/covisibility-selection.md
### specs/core/features/gpu-optical-flow.md
### specs/core/features/kdf-constellation-query.md

Read 2026-09-26; open items above.

### specs/core/features/lazy-kdforest-query.md

Read 2026-09-26; open items above.

### specs/core/features/optical-flow.md
### specs/core/features/sift.md
### specs/core/features/track-cluster-matching.md
### specs/core/geometry/affine-factorization.md
### specs/core/geometry/baseline-direction.md
### specs/core/geometry/estimate-intrinsics.md
### specs/core/geometry/focal-vote.md

Read 2026-09-26; open items above.

### specs/core/geometry/rotation-locked-resection.md
### specs/core/patch/candidate-track-spawning.md
### specs/formats/archive-container.md
### specs/formats/kdf-file-format.md
### specs/formats/sfmtool-camera-models.md
### specs/formats/sift-file-format.md
### specs/gui/camera-intrinsics.md
### specs/gui/panel-layout.md
### specs/gui/viewport-navigation.md

## Top priorities

1. **Code bugs found by the deep reads.** Fix each in the code and add a test.
   No documented default disagrees with the code this run (check 1), so these
   are where behaviour differs from what a reader is told.
   - **Flow matching pairs frames in filesystem order.** `expand_paths`
     (`src/sfmtool/_filenames.py:60`) returns `rglob` order unsorted, and the
     flow matcher pairs adjacent entries. On ext4 that order is not the file
     name order, and rig subdirectories are joined end to end.
     (flow-based-matching)
   - **`sfm flow -r` resolves images by file name only** (`_flow_display.py:101-105`):
     on a rig, `fisheye_left/frame_01.jpg` resolves to
     `fisheye_right/frame_01.jpg`, so the wrong tracks are compared with no
     warning. (flow-command)
   - **Absolute-pose local optimization keeps the refit only on a strict gain**
     (`absolute_pose.rs:489`, `new_count > count`). On clean data it returns the
     raw P3P pose, and the comment at :495 says the opposite. (absolute-pose)
   - **`KdForest.query(max_dist=…)` returns points beyond `max_dist`**:
     `u8::cutoff_sq` rounds `max_dist²` up (`kdforest/distance.rs:78-85`), so
     2.5 admits a squared distance of 7 (2.65). (randomized-kdtree-forest)
   - **`reprojection_residuals` binding:** zero observations give shape
     `(0, 0)`, which `inlier_fraction` then refuses; out-of-range indexes and a
     short `translations` array raise `PanicException`, not `ValueError`
     (`crates/sfmtool-py/src/geometry/reprojection.rs:90, 105`).
     (reprojection-residuals)
   - **MCP `set_view`:**
     - It ignores `fov_short_axis_deg` beside `fit`, `look_through` or
       `exit_camera_view` (`mcp/tools.rs:794-848`).
     - An out-of-range field of view moves the camera and then refuses
       (`mcp/view.rs:257-283`).
     - A background edit that changes nothing still replies `changed: true`
       (`mcp/edit.rs:584-590`).
     (mcp-server)
   - **Smaller code-side text:**
     - The refit report labels `fy/fx` as "fx/fy aspect"
       (`refit_intrinsics.rs:311`).
     - The two Track View *Lock* hover texts and `mcp/tools/catalog/edit.rs:504`
       contain runs of 18 spaces.
     - A pairwise `.matches` file with stray `two_view_geometries/` entries
       passes verification (`verify.rs:39-145`).
2. **Non-goals that shipped code has overtaken.** A reader who believes them
   will not look for a feature that exists:
   - `bundle-adjustment.md:1773` says cameras cannot be released one at a
     time, but #607 added that. :1780 says there is no constraint handling,
     but #400 added it. The spec also says the kernel "serves the bootstrap
     experiments".
   - `refit-camera-intrinsics.md:196, 480` says nothing frees the principal
     point, but `sfm densify --ba-refine-principal-point` does.
   - `track-view.md:2015` gives "Deciding anything from a number" as a
     non-goal, while the bars set the verdicts.
   - `mcp-server.md:4994` says editing intrinsics is "still not on the
     surface", but `switch_camera_model` and lens releases exist.
   - `randomized-kdtree-forest.md`: "potentially for patch matching" is
     overtaken in part, because the bench searches use the forest.
3. **Spec statements that would make a caller act wrongly:**
   - `mcp-server.md` says five edits "renumber nothing", but after a
     `delete_point` they shift point indexes.
   - `pose-verification.md:187` says `INLIER_PX` bounds the screens; it bounds
     only the repair.
   - `reprojection-residuals.md` names three shared callers; there is one.
   - `localize-keypoints-command.md:109` says the CLI "re-specifies nothing";
     it hardcodes all 17 defaults.
   - `cluster-covisibility.md:323` says a pre-v6 file opens; the reader
     refuses it.
   - `randomized-kdtree-forest.md` gives `L_max` as "precision-tuned"; it is a
     fixed 128.
4. **Format specs:**
   - `matches-file-format.md:223` says versions run `1` through `6`; the code
     is at 7.
   - Both format specs still name `sfm` commands, repo paths and library types
     in the format proper: 5 confirmed passages in matches (plus 3 more found by
     the deep read) and 3 in sfmr (plus 1).
   - Three entries never say which version introduced them: `world_space_unit`
     and `infinity_point_count` in `.sfmr`, and `.matches` `refine_options`.
   - The `.sfmr` writer replaces a zero normal on a finite point, which the
     spec does not allow for or rule out.
5. **Nine opening paragraphs.** Check 4 found five:
   `spherical-tiles-rig.md`, `gui/edits/move-camera.md` and
   `keypoint-reach.md` never say what the thing is for, and
   `research/blender-…` and `docs/index.md` fail in a smaller way. The deep
   reads proposed four more: `flow-command.md`, `cluster-covisibility.md`,
   `reprojection-residuals.md` and `track-view.md`. Land them **one spec per
   PR**: each proposed sentence is a claim about the code, and a reviewer
   checks it properly only when reading it alone.
