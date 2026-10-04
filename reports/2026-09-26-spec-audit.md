# Spec audit — 2026-09-26

**Sample:** 21 of 173 specs read against their code. Seed `10351`; the pool was
the never-audited specs (157 candidates). The random draw gave 10 of them:

- `specs/core/camera/ray-grid-projection.md`
- `docs/index.md`
- `specs/core/features/covisibility-selection.md`
- `specs/core/features/lazy-kdforest-query.md`
- `specs/cli/colmap-interop/to-colmap-bin-command.md`
- `specs/cli/reconstruction/inspect-command.md`
- `specs/core/features/kdf-constellation-query.md`
- `specs/cli/image-feature/sift-command.md`
- `specs/core/geometry/baseline-direction.md`
- `specs/core/geometry/focal-vote.md`

Eleven more were read regardless of the draw:

- **Code changed materially since the 2026-09-05 audit:**
  - `specs/core/bench/editable-track.md`, because `crates/sfmtool-core/src/bench/` had heavy churn.
  - `specs/gui/viewport-navigation.md`, because of Maintain Z-up (#614), plus #610 and #613.
- **Defaults check hits:**
  - `specs/core/analysis/cluster-census.md`
  - `specs/core/features/track-cluster-matching.md`
- **Format specs with a hit in check 6** (every one except camrig):
  - `archive-container`
  - `cluster-selection`
  - `kdf-file-format`
  - `matches-file-format`
  - `sfmr-file-format`
  - `sfmtool-camera-models`
  - `sift-file-format`

The corpus-wide mechanical checks below cover all 173. A spec that was not
sampled is not acquitted by this report's silence about it. The budget was
N=10 plus 11 extra reads.

The sampled specs contain **102 Non-goals bullets and deferral phrases, and all
were checked**. 14 were stale:

- **ray-grid-projection.md:** `pixel_to_ray_grid` is written as an inline proposal.
- **covisibility-selection.md:** the "any use of image ordering" non-goal.
- **editable-track.md:** "Editing the patch's frame or normal by hand".
  > _Status (2026-10-03): **Done** — the non-goal is removed, PR #687._
- **viewport-navigation.md:**
  - "(future)" settings
  - "Planned…" FOV
  - the two duplicated open questions
  - the Alt-menu "Investigate"
  - "temporary proxy"
  - the inertia non-goal
  > _Status (2026-10-03): **Done** for viewport-navigation.md — the "(future)" sensitivity, the "Planned…" FOV text and both open questions now state what exists and point at Non-goals, the Alt-menu row describes the `SC_KEYMENU` suppression, "temporary proxy" is gone, and the inertia non-goal names the Windows touchpad exception, PR #685._
- **cluster-census.md:** two claims about callers that are not in the tree.
  > _Status (2026-10-03): **Done** — § Callers and the `flag_threshold` row are deleted, PR #684._
- **track-cluster-matching.md:** four items: persisting is out of scope, the CLI consumes pairs only, "add a section", "consider lifting".
  > _Status (2026-10-03): **Done** — all four are gone with the build-brief section; the spec's Pipeline section says `sfm match --cluster` writes the clusters file and pairs are derived from it, PR #704._
- **archive-container.md:** "No schema".

---

## Mechanical findings (all 173 specs)

### 1. Documented defaults vs actual defaults

The check compared 588 parameters: 203 CLI option rows, 4 CLI prose defaults
and 381 rows for library, GUI and xform keys. Each row was keyed on (spec →
owning command or type → parameter), and defaults written in fenced code were
read as well as those in tables. The scan produced 162 candidates, and reading
the code cleared all but 3. **No documented default differs in value from the
code.** The three that remain are parameters the spec documents but the code
lacks, or a wrong parameter name:

| Spec:line | Param | Spec | Code | Right side |
|---|---|---|---|---|
| core/features/track-cluster-matching.md:358-363 | `threshold`, `cliff_pct`, `t_scale`, `refine`, `prefilter` | cliff, 50, 1.0, 0, off | none exist (`sfmtool-py/src/matching/cluster.rs:93`, `cluster_match/mod.rs:53-59`); prototype-only in an unmerged experiment | code — delete table |
| core/features/track-cluster-matching.md:352 | `bg_alpha` | 0.8 | `alpha=0.8` | code — rename |
| core/analysis/cluster-census.md:304 | `flag_threshold` | 0.25 | absent; its callers exist only on unmerged branch `fork/bootstrap-core-migration` | code — delete row |

> _Status (2026-10-03): **Done** — the `cluster-census.md` `flag_threshold` row is deleted, PR #684._
>
> _Status (2026-10-03): **Done** — both `track-cluster-matching.md` rows: the global-threshold table is deleted and the one defaults table says `alpha`, PR #704._

Five more mismatches came from reading the sampled specs rather than from the table scan:

- **sift-command.md:36:** says `--dsp` defaults to "workspace". It is actually off, and it requires `--tool colmap`.
- **viewport-navigation.md:761:** says the indicator size multiplier defaults to 3.0. The code has 0.3.
- **viewport-navigation.md:766:** says the indicator opacity runs 50%→10%. The shader and the spec's own table say 20%→5%.
  > _Status (2026-10-03): **Done** for the two viewport-navigation.md items — 0.3 and 20%→5%, PR #685._
- **focal-vote.md:167:** says "at most 60" rotation images. The code allows up to 119.
- **kdf-constellation-query.md:593:** says a length mismatch raises `ValueError`. The code raises `OSError`.
> _Status (2026-10-02): the focal-vote and constellation items are **Done** in code, commit `b19c676` (#670); both specs already described the fixed behaviour. The `--dsp` and indicator items remain open._
> _Status (2026-10-03): the `--dsp` item is **Done** — the spec documents `--dsp` as off and requiring `--tool colmap`, PR #679._

### 2. Prose duplicated between a spec and its code

There are 224 normalized lines of 60 or more characters that appear in both a
spec and the code. Only 13 spec↔code pairs share 5 or more lines. Nearly all of
the sharing is doc comments copied into a spec's fenced API block, and the
spec's copy is the one that goes stale.

| Shared (in fence) | Spec | Code |
|---|---|---|
| 23 (23) | core/features/track-cluster-matching.md | sfmtool-core/src/features/cluster_match/mod.rs |
| 11 (11) | gui/operation-progress.md | sfmtool-progress/src/lib.rs |
| 9 (8) | core/reconstruction/batch-triangulation-api.md | reconstruction/triangulation.rs |
| 8 (8) | core/patch/patch-cloud.md | patch/cloud.rs |
| 8 (8) | gui/camera-intrinsics.md | sfm-explorer/src/state.rs |
| 7 (7) | gui/action-log.md | action_log/mod.rs |
| 7 (7) | gui/background-tasks.md | background/mod.rs |
| 6 (6) | core/features/track-cluster-matching.md | feature_match/_cluster_matching.py |
| 6 (6) | core/patch/patch-normal-refinement.md | normal_refine/params.rs |
| 5 (2) | core/bench/editable-track.md | bench/steps.rs |
| 5 (4) | core/reconstruction/edited-reconstruction.md | reconstruction/edited.rs |
| 5 (4) | core/spherical/spherical-tiles-rig.md | spherical/tile_rig.rs |
| 5 (5) | gui/panel-layout.md | sfm-explorer/src/window.rs |

Plain prose, outside code blocks, is duplicated at most 3 lines per pair. In
this corpus the third copy is usually a paraphrase rather than a verbatim
line, so the deep reads below find more of them than this check does.

### 3. Shape of the `specs/core/` specs

Of 75 `specs/core/` specs, 45 have a ` ```rust ` block and 30 do not.

**Twelve have no code at all:**
- `analysis/image-pair-graph`
- `analysis/keypoint-reach`
- `analysis/reconstruction-alignment`
- `features/flow-based-matching`
- `geometry/pose-verification`
- `geometry/reconstruction-growth`
- `geometry/relative-pose`
- `patch/fronto-parallel-patch-cache`
- `patch/keypoint-localization-consensus-basis`
- `patch/patch-normal-refine-view-subset`
- `patch/patch-view-selection`
- `reconstruction/point-correspondence`

**Nine have their Rust block 50% or more of the way down:**
- `observation-adjacency-graph` 69%
- `adjacency-surfel-normals` 68%
- `sift` 67%
- `member-coherence-validation` 70%
- `rotation-locked-resection` 59%
- `triangulation-rules` 58%
- `affine-factorization` 56%
- `track-cluster-matching` 53%
- `candidate-track-spawning` 51%

Treat these lists as a reading list, not as findings. In this sample,
`baseline-direction` and `focal-vote` both confirmed failure 2: neither lists
its Rust entry points. `covisibility-selection` lists names but no signatures.

**Work-order residue:**
- **Grep hits:**
  - `ray-grid-projection.md:163` "After this change…"
  - `candidate-track-spawning.md:134` "As part of this change…"
    > _Status (2026-10-03): **Done** — rewritten in the present tense, PR #689._
  - `patch-normal-refine-view-subset.md:228`, which is a draft in shape: § Motivation and a test plan.
  - `select-by-distribution-command.md:230` "initially".
- **Headings that matched the pattern but are not residue:**
  - Algorithm steps: `motion-command`, `select-by-distribution`, `epipolar-curves`, `photometric-subsets-ransac`.
  - Format "Versioning and migration" sections.
  - Frame phases in `operation-progress` and `action-log`.

### 4. Opening paragraphs

The check read 160 openings, excluding README, TEMPLATE, GLOSSARY and drafts.

| Flag | Hits |
|---|---|
| backticked identifier or `::` in the first sentence | 47 (about 20 are only `.sfmr` or command names) |
| link in the first sentence | 10 |
| first character is a symbol | 6 |
| describes a change | 2 real (image-warping, bundle-adjustment) |
| no flag | 113 |

The worst openings from the corpus-wide read are listed below with proposed
replacement first sentences. Land these **one spec per PR**: each replacement
was written from the spec and only spot-checked against the code, so a human
who knows the subject needs to read it as a claim before it goes in.

1. **`core/geometry/bundle-adjustment.md:5`**
   - Current: "The staged robust bundle adjustment written for the cluster pinhole bootstrap experiments, whose scripts have since been removed…"
   - Problem: it is history, and it is misleading, because this is the production optimizer.
   - Proposed: *"Staged bundle adjustment jointly refines camera poses, 3D points and optionally each camera's focal length and distortion, minimizing robust pixel reprojection error over rounds that trim outliers and retriangulate between them."*
2. **`core/camera/image-warping.md:5`**
   - Current: "The Rust codebase has complete implementations of `distort()` and `undistort()`…"
   - Problem: it describes a change, and the claim about pycolmap is stale.
   - Proposed: *"Image warping resamples a whole photograph from one camera model into another (for example, undistorting a fisheye image into a pinhole one) through a precomputed per-pixel map."*
3. **`core/spherical/photometric-subsets-ransac.md:3`**
   - Current: the first prose is a link-reference definition, followed by "The input is a [`PerSphericalTileSourceStack`]…".
   - Problem: it never says what the algorithm is for.
   - Proposed: *"Stitching a panorama needs, for each direction, the colour most source images agree on; this finds, per tile, the largest group of sources whose patches agree and marks the rest as occluders or parallax."*
4. **`core/analysis/source-clusters.md:3`**
   - Current: "A reconstruction member is drawn from a cluster selection…"
   - Problem: TEMPLATE.md already cites this as a "Not this" example, and it is still unfixed.
   - Proposed: *"A reconstruction built from a few large feature clusters leaves smaller, better-localized clusters unused; this kernel lists them, grouped by feature radius, so a caller can add them back one band at a time."*
5. **`core/features/randomized-kdtree-forest.md:3`**
   - Current: "Forests can be persisted and searched under a bounded decoded-data cache as specified in…"
   - Problem: it is a "see also" and never says what a forest is.
   - Proposed: *"A randomized kd-tree forest finds approximate nearest neighbours of SIFT descriptors much faster than an exhaustive scan, by searching several randomized trees with a shared priority queue."*
6. **`core/patch/patch-normal-refinement.md:5`**
   - Current: "A reconstructed 3D point `X` is seen by cameras `{(Kᵢ, Tᵢ)}`."
   - Problem: notation comes before any words.
   - Proposed: *"Photometric patch-normal refinement finds the surface orientation at a reconstructed point by trying candidate normals and keeping the one whose rendered patches look most alike across the images that see it."*
7. **`core/patch/sift-to-patch-reconstruction.md:5`**
   - Current: "A `sift_files` reconstruction locates each observation by a reference…"
   - Problem: it opens on a field value.
   - Proposed: *"`sfm embed-patches` converts a reconstruction whose observations point into per-image `.sift` files into a self-contained one that stores a small oriented patch per point and a 2D keypoint per observation."*
8. **`core/geometry/estimate-intrinsics.md`**
   - Current: "`estimate_intrinsics` is the high-level face of the structure-free focal vote…"
   - Proposed: *"Intrinsics estimation guesses a capture's shared focal length and camera-model family from cluster matches alone, before any reconstruction exists, and reports whether the guess is confirmed."*

The format-spec openings (kdf, matches, cluster-selection, archive-container,
camera-models) are covered in their sections below.

Eight specs open with "This document describes/specifies…":
- gpu-optical-flow
- gui/architecture
- gui/camera-views
- gui/point-cloud-rendering
- gui/user-experience
- gui/viewport-navigation
- xform/scale-by-measurements
- xform/select-by-distribution

### 5. Coverage both ways

**(a) 54 of 170 specs are never cited from `crates/`, `src/`, `tests/` or `scripts/`.** Most are CLI specs, where no citation is expected:

- **CLI specs:** 24 of the 54, which is every `cli/*` spec except those whose commands link to them.
- **Non-CLI specs:**
  - `core/analysis/image-pair-graph`
  - `core/analysis/reconstruction-alignment`
  - `core/camera/image-warping`
  - `core/camera/projection-jacobian`
  - `core/camera/sfmtool-pinhole-kernels`
  - `core/features/flow-based-matching`
  - `core/features/gpu-optical-flow`
  - `core/features/optical-flow`
  - `core/features/randomized-kdtree-forest`
  - `core/geometry/relative-pose`
  - `core/geometry/reprojection-residuals`
  - `core/reconstruction/point-correspondence`
  - `formats/cluster-selection`
  - `gui/architecture`
  - `gui/cross-panel-hover`
  - `gui/image-animation`
  - `gui/patch-rendering`
  - `gui/user-experience`
  - `gui/viewer-3d-bench-layer`
  - `gui/viewport-navigation`
  - `workspace/workspace`
  - `research/blender-…`
- **Drafts:** 8.

**(b) Code surfaces without a spec.** See **Code without specs** below.

### 6. Format specs that depend on code names (every run)

| File | Links into code, core, gui or cli | Identifier hits | Verdict after the deep read |
|---|---|---|---|
| archive-container.md | 9 | 69 (67 in the Rust API) | 5 findings; a crate spec is fused into the format spec |
| camrig-file-format.md | 0 | 0 | clean |
| cluster-selection.md | 0 | 1 | a code name in the opening; this is an operation spec filed under `formats/` |
| kdf-file-format.md | 5 | 1 | 3 findings; the only format spec with an Implementations section |
| matches-file-format.md | 5 | 3 | 10 findings; 3 of the 5 links carry a definition |
| sfmr-file-format.md | 5 | 7 | 12 findings; 2 of the 5 links are implementation links |
| sfmtool-camera-models.md | 5 | 2 | 7 findings |
| sift-file-format.md | 1 | 4 | 7 findings, plus a wrong field definition |

**Structural result:** six of seven format specs with hits have no
*Implementations* section, so crate, type, function and CLI names are spread
through the normative text. Adding that section to each one is what would fix
most of the per-sentence findings.

**Other failure-6 checks:**
- **Per-element code column with no legend in the file:** one, matches `member_status`. The spec does name all seven codes, so this is discussion-grade.
- **Optional entries with no stated introducing version:** sfmr Rigs and Frames.
- **Version-history gaps:**
  - sfmr: no v10 entry, no 7→8 migration, and the example still says v7.
    > _Status (2026-10-03): **Done** — v10 history entry, 7→8 migration and a version 11 example added, PR #677._
  - kdf: says nothing about what versions 2 and 3 changed.
    > _Status (2026-10-03): **Done** — `kdf-file-format.md` § Version now says what versions 2 and 3 changed, PR #680._

---

## Sampled specs

### specs/core/camera/ray-grid-projection.md
**Summary:** Patch warps build an affine grid of camera-frame rays (`WarpMap::from_patch`) and project the whole grid at once (`CameraIntrinsics::ray_to_pixel_grid`). The projection is exact for perspective models and uses a probe-checked coarse grid for fisheye and equirectangular models. The core design matches the code. More than half the file is before/after measurements and a separate remap optimization.
**Implementing code:** `camera/distortion/ray_grid.rs` (`ray_to_pixel_grid{,_exact,_coarse}`, `COARSE_GRID_STRIDE`, `COARSE_GRID_TOL_PX`), `camera/warp_map.rs:345` `from_patch`, `camera/remap.rs:312-424`.
**Inconsistencies:**
  - :80-83 say fisheye and equirectangular always take the coarse path. The code also takes the exact path when `cols` or `rows` < 16 (`ray_grid.rs:191-196`).
  - :143 names a "`warp_from_patch` leaf", which does not exist. The real callers are `keypoint_localize.rs:421`, `bench/evaluate.rs:1227`, the Track View `patch.rs:144` and Python `WarpMap.from_patch`.
  - :170 names a `_rectification` consumer and :167 names `pixel_to_ray_grid`. Neither exists.
  - :209 and :214 use `sfm embed-patches --subpixel lk`. `--subpixel` now takes an integer (`embed_patches.py:116-127`).
  - :36-37 write `V` for the in-plane axis, but the code walks `−v_axis` (`warp_map.rs:366`). The letter `t` is used both for the patch coordinate and for the translation.
**Third copies:** `ray_grid.rs` states "accuracy never depends on the stride" three times: at :7-10, :22-25 and :207-213. Keep one line on the constant plus a link. The tolerance rationale at `ray_grid.rs:32-33` exists only in the code; move it into the spec.
**Recommendation:** update spec. Cut the § Problem and § Impact history, move § "channel-batched bilinear gather" to `image-warping.md`, and fix the stale names.
**Unclear / incorrect / suspicious:**
  - The measured figures (18×, 0.0197 px) are quoted as if current.
  - :87-103 transcribe `GridProj` and its index arithmetic.
  - The opening is good.

### specs/core/geometry/baseline-direction.md
**Summary:** A batched solve for the unit direction between two camera centres whose rotations are known, from ray-coplanarity normals. It covers angular-bound row selection, trimmed refits, the cheirality sign and per-edge diagnostics. The algorithm text matches `baseline_direction.rs`.
**Implementing code:** `geometry/baseline_direction.rs` (`baseline_directions`, `BaselineDirection`, `BaselineTrim`); `sfmtool-py/src/geometry/baseline_direction.rs`.
**Inconsistencies:**
  - The spec never names `tol_rad`, `rounds` or `keep_fraction`. The code has no defaults for them, and all three are required in Python.
  - The `BaselineTrim::tol_rad` doc (`:61`) describes the three-row floor, not the field. **Fix the code comment.**
  - The cheirality fraction is `pos/(pos+neg)` (`:162`), which leaves out rows that vote neither way and NaN rows. The spec says "the winning fraction" and the struct doc says "of the used rows". Both are imprecise.
  - The residual `|n̂·d|` is a sine, but spec :66 calls it "radians".
  - Python returns `n_rows = n_used = 0` for an edge that states no direction (`py :114-115`), so a caller cannot tell "no rows" from "all rows under the bound".
  - Rust does not validate CSR `offsets`, so a bad offset panics. Python does validate them.
**Third copies:** The Rust module doc (`:4-24`, about 20 lines) re-derives the spec almost word for word, and the PyO3 doc (`:16-23`) does it a third time. Shrink both to a contract plus a link.
**Shape:** There is no Rust API, parameter section or example (failures 2 and 3).
  - **Opening**, proposed: *"Global SfM solves every camera's rotation first and then needs, for each pair of overlapping images, the direction from one camera centre to the other so translation averaging can place the centres; this computes that unit direction for every edge of an image-pair graph in one call."*
  - **Mannered prose:**
    - "A row is worth the baseline it saw" → "Rows are selected by parallax".
    - "a coin toss" → "close to 50%".
    - "places an obligation on whatever consumes them" → "requires the consumer to handle colinear graphs".
  - **Other:** :79-80 "the implementation this was ported from" refers to an origin the reader cannot see.
**Recommendation:** update spec, and fix the `tol_rad` doc comment in the code.
**Unclear / incorrect / suspicious:** `baseline_directions` has no production caller; only tests use it. Say in the spec whether a global-SfM pipeline is expected to use it.

### specs/core/features/lazy-kdforest-query.md
**Summary:** Approximate nearest-neighbour queries against a `.kdf` file directly through a bounded decoded-data cache, returning the same results as the in-memory forest. The interface, parity rules and option defaults match the code. About 850 of its 1,292 lines are version-1 benchmark history and planning from before implementation.
**Implementing code:** `features/kdforest/persistent.rs` (`LazyKdForest`, `write_kdf[_ordered]`, `read_kdf`); `sfmtool-kdf-format/src/{types,cache,read,verify}.rs`; `sfmtool-py/src/spatial/kdf.rs`.
**Inconsistencies:**
  - **Wrong format version.** :186-188, :989, :995 and :1261 say "Version 2… Version-1 files are rejected". The code has `KDF_FORMAT_VERSION = 3` (`types.rs:10`) and rejects version 2.
    > _Status (2026-10-03): **Done** — the spec states the format is at version 3 and rejects every other version; :376 and :995 now say version 2 introduced the single corpus and version 3 keeps it, PR #680._
  - **Shard cap.** :837 gives `cache_bytes / max_chunk_bytes`. The code (`cache.rs:185`) and :444 use the largest validated item.
  - **Block size.** :1005-1008 and :1289 recommend 4-8 KiB descriptor blocks. The default and :789-812 say 2 KiB.
  - **LRU.** :591-593 describe a counter-stamped LRU. The code uses an indexed doubly linked list, which :438-441 describe correctly.
  - **Tests.** :1192-1203 say "`persistent.rs` holds three cases". There are six, in `persistent/tests.rs`.
  - **API list.** It omits `read_kdf`, `reset_io_stats`, `content_xxh128`, `LazySearchScratch`, `search_batch_with_stats`, `search_batch_with_distances_ordered`, `self_join_with_distances[_progress]` and `kdf_summary`.
  - **Code bug:** the doc comment for `search_batch_with_stats` sits on the front of `search_batch_with_distances_ordered` (`persistent.rs` ~562-580).
  - **Encoding corruption:** at :354, :361 and :539-547, `§`, dashes and quotation marks have become `?`.
**Third copies:** Minor. `write_kdf_ordered` (`persistent.rs:60-80`) repeats spec :93-102.
**Shape:**
  - **Opening**, proposed: *"A `.kdf` file stores a randomized kd-tree forest over SIFT descriptors so that approximate nearest-neighbour queries can run against the file directly, decoding only the tree chunks and descriptor blocks a query visits into a bounded cache, with results identical to the in-memory forest."*
  - **Residue:**
    - :19 "The integration belongs beside the existing implementation".
    - Imperatives at :190, :198 and :236.
    - :379 "Start with…".
    - :403 "Do not download…".
    - :852 "An earlier version of this section…".
    - § Acceptance checks (:1036-1180) is a cost model from before the build.
  - **Mannered prose:** :690 "the balance sits far smaller than it first appeared" → "the best block size is smaller than the first measurements suggested".
  - **Proposed TEMPLATE amendment:** let an evidence-heavy spec keep its dated measurements in a sibling `<name>-measurements.md`, with the standing spec citing only the conclusions and chosen defaults. That would take this spec from about 1,300 lines to about 400.
**Recommendation:** update spec (version, shard cap, block size, encoding, split out the history). Update code only to move the misplaced doc comment.
**Unclear / incorrect / suspicious:** Python's `LazyKdForestU8` probably cannot open an f32 file, and the spec does not say so. The "provisional" labels at :33-46 are stale.

### specs/core/features/kdf-constellation-query.md
**Summary:** A well-shaped spec, with a plain opening, API rationale and measured theory. All 11 `ConstellationParams::DEFAULT` values match. There is one behavioural error-mapping divergence, and much rationale is duplicated in Rust doc comments.
**Implementing code:** `features/kdforest/constellation.rs` (`constellation_query` :403, `constellation_at_pixel` :976, `constellation_from_keypoints` :1039, `radius_for_feature_count` :944); `neighbor_index.rs`; `sfmtool-py/src/spatial/{constellation_query,kdf,kdforest}.rs`.
**Inconsistencies:**
  - **Behavioural.** The spec at :593-596 and the docstring at `kdf.rs:540` say mismatched array lengths raise `ValueError`. In fact `KdfError::ShapeMismatch` (constellation.rs:420, :429) maps to **`OSError`** (`kdf.rs:70`), and no test covers it.
    > _Status (2026-10-02): **Done** — the length checks return `KdfError::InvalidQuery`, which maps to `ValueError`, with a test, commit `b19c676` (#670). The spec was already right._
  - :459-460 say the origin-table pass is paid "once per `constellation_at_pixel` call". It is also paid on every `constellation_from_keypoints` call, which is the bench's per-gesture path (`bench/search.rs:356`), and nothing caches it.
  - A bad `positions` dtype raises `TypeError` and a missing `sources` key raises `KeyError`, not `ValueError`.
**Third copies:** These Rust doc comments in `constellation.rs` should each shrink to a contract plus a link:
  - `AffineRefit` and `sigma` (:36-73, about 35 lines).
  - Seeding (:389-402).
  - `radius_for_feature_count` (:925-943). Its per-capture numbers are not in the spec; move them there.
  - `NeighborIndex` (about 15 lines).
**Shape:** Mannered prose:
  - "chaff" (:357, :499) → "wrong correspondences".
  - "the guards bite" → "the guards take effect".
  - "earns its place" → "is kept".
**Recommendation:** update code. Map the length mismatch to `ValueError` and add a test, then shrink the doc comments.
**Unclear / incorrect / suspicious:** The per-gesture `image_feature_ids` pass is probably the largest cost in interactive latency at DinoLedge scale (9.7M origins). Measure it.

### specs/core/features/covisibility-selection.md
**Summary:** Three queries on `ClusterCovisibility`: sampled pair displacement, banded thinning (`thin`, `thin_to`) and `reach`. The behaviour largely matches the code.
**Implementing code:** `features/cluster_match/covisibility.rs`, `covisibility/selection.rs`; `sfmtool-py/src/matching/covisibility.rs`. The one consumer is `geometry/reconstruction_growth.rs:621,1031`.
**Inconsistencies:**
  - :42 says "squared-root pixel distances". The code uses a Euclidean distance (`hypot`, covisibility.rs:383).
  - :41 says clusters with "two or more member images". The code tests for two or more accepted members (:374).
  - :7-8 and :61-64 say "order-free". Without positions, `sweep_order` falls back to index order (selection.rs:21-37), and ties are broken by index. The non-goal "any use of image ordering" is stale.
  - `thin_to` searches `tau` over `[1, median peak]` in 25 iterations, so results are capped: `thin_to(8)` returns 4 images on chain8. The spec says none of this.
  - `ClusterCovisibility.from_matches` (py:198) is not mentioned.
**Third copies:** The sampling rationale is copied into `from_clusters_with_positions` (covisibility.rs:281-298, 18 lines) and into the binding's `positions_xy` Args (py:135-145). The binding copy should shrink.
**Shape:** The opening "Three queries over a set of images' shared-cluster counts…" never says what the queries are for.
  - **Proposed:** *"Before a reconstruction exists a pipeline has to choose which images to work with — which covisible pairs differ most, which images are redundant, and whether a subset still connects to the rest; these three queries on per-pair shared-cluster counts answer that."*
  - **Missing:** Rust signatures, and any rationale for `tau/8` and `min_shared=8`.
**Recommendation:** update spec.
**Unclear / incorrect / suspicious:** The binary search in `thin_to` relies on "the kept count grows monotonically with tau". The `tau/8` floor also rises with `tau`, and the sweep is greedy, so monotonicity is not guaranteed and no test checks it beyond chain8.

### specs/core/geometry/focal-vote.md
**Summary:** A thorough spec whose constants match the code almost everywhere, from the 30/16/6 pair counts through the fisheye 50°/110° bands. What is wrong: the list of environment flags, one precision claim, the rotation-image cap, and a buried interface.
**Implementing code:** `geometry/focal_vote.rs` (`focal_vote*`, `FocalVoteOptions`, `FocalVoteResult`), `focal_vote/column_scan.rs`, `homography_estimation.rs`, `simd.rs`; `sfmtool-py/src/geometry/{focal_vote,homography_estimation}.rs`.
**Inconsistencies:**
  - :167 says "at most 60" rotation images. `step = (n_img/60).max(1)` (`:1036`) visits up to 119. **This is probably a code bug; use `div_ceil`.**
    > _Status (2026-10-02): **Done** — the stride is `n_img.div_ceil(60)`, so at most 60 images are visited, with a test, commit `b19c676` (#670). The spec was already right._
  - :577-584 list "three" environment flags. The code also reads `SFMTOOL_FOCAL_VOTE_F64_EPI` and `SFMTOOL_FOCAL_VOTE_F32_ROT` (simd.rs:67,75). F32_ROT contradicts :75 "stays f64". The code also reads `_F32_AUDIT` and `SFMTOOL_PROFILE`.
  - :56 says "every computation below is `f64`". That contradicts :69, since epipolar residuals are f32 by default. There is no stated parity test for the 8-lane f32 kernel.
  - :469-471 give the wrong `estimate_homography` signature. The real one is keyword-only, with `confidence`, `max_iterations`, `min_inliers` and `local_optimization`.
  - The binding docstring (py:308) says the seed drives the pair-table pass. It does not (focal_vote.rs:897-930), so the code side is wrong.
**Third copies:**
  - The binding docstring (py:272-369, about 95 lines) restates the consensus rules. Cut it to Args, the dict keys and a link.
  - `column_scan.rs:4-38` (35 lines) re-derives the spec's rationale. Cut it to a contract plus a link.
  - The `F32_EPI` doc carries measurements the spec lacks. Move them into the spec.
**Shape:** The Rust entry points are never listed, and the code pointer is at :441 (failure 2).
  - **Opening**, proposed: *"Before any reconstruction exists, this estimates the focal length in pixels shared by every image of a single-camera capture, from keypoints matched across images, and decides whether the lens is pinhole or equidistant fisheye."*
  - **Residue:** "existing" (:144, :153), "joins the geometry module" (:195), "Output gains" (:426).
  - **Mannered prose:**
    - "costs its lane advantage back" → "is no faster than f64".
    - "centre-hugging" → "near the principal point".
  - **Proposed TEMPLATE amendment:** numerical-kernel specs get a dedicated "Determinism and precision" section, holding the flags, SIMD parity and f32 choices.
**Recommendation:** update spec. Also update code: fix the binding seed docstring and decide on the 60-image cap.

### specs/cli/colmap-interop/to-colmap-bin-command.md
> _Status (2026-10-03): **Done** — the spec has the proposed opening, the non-`.sfmr` input usage error, a Feature sources section (`.sift` files needed for `sift_files`, synthetic indices for `embedded_patches`), a Points at infinity section, an Implementation section that credits `_range_options.apply_range_filter` without the residue sentence, and a Testing section naming `tests/test_colmap_interop.py`; the `subset_by_image_indices` row in `xform-command.md` (the Unclear item) says it works for both feature sources, PR #696._
**Summary:** The options, defaults, range semantics, output files and coordinate conventions all match. Three behaviours are missing, and the Implementation section is stale.
**Implementing code:** `_commands/to_colmap_bin.py`, `_commands/_range_options.py` (shared with `to-nerfstudio`), `colmap/io.py:754` `save_colmap_binary`, `:116` `materialize_infinity_for_export`, `reconstruction/edit.rs:267` `subset_by_image_indices`.
**Inconsistencies:**
  - :107-110 say `to_colmap_bin.py` parses `--range` itself. The shared `_range_options.apply_range_filter` does it.
  - The spec does not say that points at infinity are materialized to finite depth on export (`io.py:782`).
  - It does not say a `sift_files` reconstruction needs its `.sift` files (`io.py:803-810`). The embedded-patches path assigns synthetic feature indices, so "feature indices are preserved" holds only for `sift_files`.
  - A non-`.sfmr` input gives a UsageError (`:55`), which is not documented.
**Third copies:** none.
**Shape:**
  - **Opening**, proposed: *"`sfm to-colmap-bin` writes a reconstruction stored in this toolkit's `.sfmr` file as COLMAP's five-file binary sparse model, so it opens in the COLMAP GUI or any tool that reads that format; `--range` exports a subset of images."*
  - :111 "No changes to `save_colmap_binary` are required" is residue.
  - There is no Testing section; name `tests/test_colmap_interop.py`.
**Recommendation:** update spec.
**Unclear / incorrect / suspicious:** `xform-command.md:896` says `subset_by_image_indices` works on "`sift_files` reconstructions only". The code works for both sources (`edit.rs:271-274`). The xform spec is the wrong one.

### specs/cli/image-feature/sift-command.md
**Summary:** A short spec whose flags and modes match Click. It leaves out most of the behaviour a user relies on, has a wrong `--dsp` default, and review turned up a real code bug.
**Implementing code:** `_commands/sift.py`, `sift/extract.py:21`, `sift/draw.py:16`, `sift/file.py:466` `get_sift_path_for_image`, `sift/extract_{colmap,opencv,sfmtool}.py`.
**Inconsistencies:**
  - :36 says `--dsp` defaults to "workspace". It is off, and it requires `--tool colmap` (`sift.py:124,143`).
  - `--num-threads` is ignored by the `sfmtool` backend (`extract_sfmtool.py:181-184`), which uses `SFMTOOL_SIFT_EXTRACT_WORKERS` instead.
  - **Undocumented:**
    - The output location: the workspace `feature_prefix_dir`, or `features/<type>-<xxh128>/` otherwise.
    - Images whose `.sift` is up to date are skipped (`extract.py:123-146`).
    - Only `.png` and `.jpg`/`.jpeg` are read.
    - Without a workspace and without `--tool`, the command errors.
    - `--tool` bypasses the workspace entirely.
**Third copies:** The `sfmtool` backend's concurrency rationale is copied in `extract_sfmtool.py:4-17` and `:237-258` (36 lines), and `core/features/sift.md` already covers it. Shrink both copies to the contract plus a link.
**Shape:** The opening "Extracts and visualizes SIFT features for images in a workspace." is also wrong, because the command works without a workspace when `--tool` is given.
  - **Proposed:** *"`sfm sift` detects SIFT keypoints and descriptors in a set of images and writes one `.sift` file per image for `sfm match` and `sfm solve`; with `--draw` it draws those keypoints onto copies of the images."*
**Recommendation:** update spec. **Also update code** for the bug below.
**Unclear / incorrect / suspicious:** **Bug.** `sfm sift --draw DIR --tool opencv` inside a workspace draws the *workspace* tool's features. `get_sift_path_for_image` (`file.py:488-493`) ignores `feature_tool`, while `--extract --tool` writes to `features/sift-opencv-*`. So `--draw --tool` either draws the wrong features or raises FileNotFound, and no test covers it.
> _Status (2026-10-02): **Done** for the bug — `--draw --tool` now ignores the workspace and reads the `--tool` features, with a test, commit `b19c676` (#670). The spec-side items above remain open._
> _Status (2026-10-03): **Partially done** — spec side done: the opening paragraph is the proposed one, `--dsp` is documented as off and rejected without `--tool colmap`, `--num-threads` is marked as ignored by `sfmtool` with a pointer to `SFMTOOL_SIFT_EXTRACT_WORKERS`, and new sections cover the output location, the up-to-date skip, the `.png`/`.jpg`/`.jpeg` filter, the no-workspace error and `--tool` bypassing the workspace, PR #679. Not done: the "Third copies" item (shrinking the concurrency comments in `extract_sfmtool.py`), which is a code-comment change._

### specs/cli/reconstruction/inspect-command.md
**Summary:** Accurate on file dispatch, the default and verbose fields, point-ID resolution and `--strips`. The main error is the stale claim that verbose point inspection needs `.sift` files.
**Implementing code:** `_commands/inspect.py`, `analyze/summary.py`, `strips/_inspect.py`, `analysis/point_inspect.rs` `inspect_point`, `sfmtool-py/src/reconstruction/sfmr_reconstruction.rs:1034`.
**Inconsistencies:**
  - :60-61 and :78-79 say the `.sift` files "must be present". `point_inspect.rs:80-116` reads inline `keypoints_xy` first. `embedded_patches` is still rejected, but not for the reason the spec gives.
  - :180-182 list the `.matches` fields. They leave out the backbone, cluster, member and patch counts (`summary.py:222-232`) and the verbose cluster-size histogram.
  - :155 says the strips options are rejected without `--strips`. The code compares each value to its default (`inspect.py:130`), so `--strips-views 8` is accepted silently.
  - When no requested point can be rendered, no PNG is written and the command exits 0. The spec does not say so.
  - Passing a directory gives "file not found".
  - WORKSPACE falls back to the given path when no workspace is found (`inspect.py:225`).
**Third copies:** The stale `.sift` claim is repeated in three places: the PyO3 docstring (`sfmr_reconstruction.rs:1016-1019`), `summary.py:650-653` and the Click help (`inspect.py:103-104`). Cut them back to the Rust contract. `strips/_inspect.py:4-17` repeats the spec's Feature-source paragraph. `summary.py:20-22` hand-copies `DEFAULT_INVERSE_DEPTH_Z_CUTOFF` from Rust; export it through the binding instead.
> _Status (2026-10-03): **Superseded** for the cutoff — `DEFAULT_INVERSE_DEPTH_Z_CUTOFF` is gone from Rust with the z rule, and `summary.py`'s `DEPTH_RELIABILITY_Z_CUTOFF` is now the report's own reading of the diagnostic, said so in its comment (step 6 of the point-or-bearing draft, #709). The `.sift` items remain open._
**Shape:**
  - **Opening**, proposed: *"`sfm inspect` prints a summary of one sfmtool file (`.sfmr`, `.sift`, `.matches`, `.camrig`) or image, explains a single 3D point given its `pt3d_` id, or renders chosen points as a patch-strip image for judging their quality."*
  - The "Inspected with" column and :69-79 list internal function names (failure 4).
  - There is no Testing section.
**Recommendation:** update spec, and update the three docstrings.
**Unclear / incorrect / suspicious:** The `embedded_patches` rejection may now be unnecessary. `strips/_inspect.py:34` refers to `compare`'s `--strips-context` flag.

### docs/index.md
**Summary:** The commands, `ws init` output, GLOMAP messages, `.sfmr` naming, PyPI name and tutorial link all match. The gaps are for readers.
**Implementing code:** `_commands/{ws,explorer,analyze}.py`, `_global_sfm.py`, `_sfmr_naming.py`, `pyproject.toml`.
**Inconsistencies:**
  - :17-18 say "load it into the SfM Explorer GUI" but never give the command (`sfm explorer <file>.sfmr`).
  - :91 `pip install sfmtool` leaves out that wheels are published for Linux and Windows only. macOS builds from the sdist and needs Rust ≥ 1.97.
  - :80-82 define "motion" as including intrinsics. It is the camera poses; the intrinsics are estimated alongside.
    - **Proposed:** *"The motion is each photo's camera position and orientation; the solver also estimates each camera's intrinsics, such as focal length and lens distortion."*
**Third copies:** none.
**Shape:**
  - :79 "it solves" has no antecedent.
  - Mannered prose: "cut my teeth on" → "learn". The first-person voice is fine.
**Recommendation:** update docs.
**Unclear / incorrect / suspicious:** The screenshot may predate the current UI.

### specs/core/bench/editable-track.md
> _Status (2026-10-03): **Partially done** — rechecked against the current bench code and fixed in `editable-track.md`: `SeedTooFar` added to the API block's `Unmeasured`; `CreateTrackOptions` and `GeometrySearchOptions` added to the API block; the "world-point forms" clamping sentence and Testing paragraph rewritten for `translate_patch`/`resize_patch` taking a world-unit displacement and half-length, and "The **offset**" renamed to a translation along the normal (the `tests.rs` section header too); the "only one that proposes several" claim corrected to name `search_geometry` and the track-at-pixel cascade; the classification rows now name `DEFAULT_CLASSIFY_NOISE_FLOOR_PX` / `DEFAULT_CLASSIFY_Z_CUTOFF`; `track-at-pixel.md` linked; the Python `create_track` keywords `version=`/`label=` documented; "have the last word" and "repaired by its next fit" reworded; the stale non-goal "Editing the patch's frame or normal by hand" removed. Already gone before this pass: `§ "The split"`, "parting company", "used to be refusals". Not done: shrinking the third-copy code comments and restructuring the Testing section into property lists. PR #687._
**Summary:** The signatures, thresholds, option defaults, classifier and geometry search match the code. #550 (steps folded into `translate_patch`/`resize_patch`) and the later search additions left stale text behind. Line numbers are against `main` after #615, which added the walk-acceptance fields and `BENCH_MAX_SHIFT_PX` (8 px); the spec documents both consistently with the code.
**Implementing code:** `sfmtool-core/src/bench/{track,steps,evaluate,fit,classify,stage,search,geometry_search,commit}.rs`; `sfmtool-py/src/bench.rs`.
**Inconsistencies:**
  - The API block's `enum Unmeasured` (:103-110) lacks `SeedTooFar`. The code has it at `track.rs:183`, and the spec's own prose uses it.
  - :1397-1399 and :2088-2097 describe "world-point forms", which were removed in #550. The test header at `tests.rs:3142` is stale too.
  - :2099 "The **offset** is tested" uses the old name.
  - :1403-1405 say descriptor search is "the only one that proposes several at a time". `search_geometry` and `build_track_at_pixel` do too.
  - :1734-1735 refers to `§ "The split"`, which does not exist. It should be "Splitting".
  - `GeometrySearchOptions` is not in the API block.
  - :1862 names `DEFAULT_NOISE_FLOOR_PX`. `FitOptions::default` reads `DEFAULT_CLASSIFY_*` (same values).
    > _Status (2026-10-03): **Superseded** — the bench decides on the point-or-bearing test; `DEFAULT_NOISE_FLOOR_PX`, `DEFAULT_CLASSIFY_*` and the z rule are gone, and `FitOptions` carries `sigma_px` and `depth_likelihood_ratio_threshold`, which the spec's Parameters table lists (step 6 of the point-or-bearing draft, #709)._
  - `track-at-pixel.md` is not linked.
  - The `create_track` keywords `version=` and `label=` are not documented.
**Third copies:** These code comments should shrink to a contract plus a link:
  - `open_localizer` (`evaluate.rs:57-74`, 18 lines).
  - `max_seed_offset_px` (about 16 lines).
  - `MAX_TILT_DEG` (`steps.rs:65-75`).

  For `RESIDUAL_MARGIN` (`classify.rs:62-82`), shrink the *spec* copy instead, since the spec says the constant carries the argument.
  > _Status (2026-10-03): **Superseded** — `RESIDUAL_MARGIN` and the residual check it set are gone with the z rule (step 6 of the point-or-bearing draft, #709)._
**Shape:**
  - The Testing section (:2030-2246) retells each claim at length, and that is where the staleness came from. Replace it with a list of property statements per test module.
  - **Stale non-goal:** "Editing the patch's frame or normal by hand" (:2260). `tilt_patch`, `spin_patch` and friends do exactly that.
  - **Residue:** :895 "…repaired by its next fit", :1906 "used to be refusals".
  - **Mannered prose:**
    - "parting company" → "differ".
    - "have the last word" → "decide".
**Recommendation:** update spec.

### specs/gui/viewport-navigation.md
> _Status (2026-10-03): **Partially done** — the fly and tilt paragraphs are back under Dolly / Fly and Tilt / Roll; Zoom to Fit gives the per-axis hfov/vfov distance with no clamp; indicator size 0.3 and opacity 20%→5%; the fog is described as a reversed-Z NDC difference with default 10.0; the `#camera-view-mode-override` anchor is now a heading, the "Step 9" citation links `camera-views.md#which-navigation-keeps-camera-view`, the initial distance is √29; the stale deferrals are fixed; the same `fog_distance = target_fog_multiplier × length_scale` error is corrected in `point-cloud-rendering.md` and two Rust doc comments, PR #685. Not done: the third copies in `righting.rs`/`mod.rs` (Rust doc comments) and the Shape items (opening, step lists, prose)._
**Summary:** The new Maintain Z-up material (`righting.rs`, `right_toward_z_up`, the turn-offs on Q/E and MCP `set_view`) and turn-toward-target (`TURN_INTO_FRACTION` 0.5) match the code. The insertion split the Dolly/Fly section, and several older numbers no longer match.
**Implementing code:** `sfm-explorer/src/viewer_3d/{mod.rs,righting.rs,camera.rs,input.rs,hud.rs}`, `mcp/view.rs:184-190`, `scene_renderer/gpu_types.rs`, `shaders/target_indicator.wgsl`, `platform/windows.rs`.
**Inconsistencies:**
  - **Structure (from #614):** :412-435 now sit under `### Maintain Z-up`, but they are about fly mode and tilt: the tilt-drag open question, the fly-key drag, mode locking and "This complements orbit navigation". The Non-goal link to `#tilt--roll` at :782 no longer points at them.
  - **Zoom to Fit** (:449-452) says `max(sx,sy)`, a vertical FOV and a ≥1.0 clamp. The code (`camera.rs:278-286`) computes a per-axis distance with hfov and vfov and has no clamp. The code is right.
  - **Indicator size:** :761 says 3.0; the code has `DEFAULT_TARGET_SIZE_MULTIPLIER = 0.3`. **Opacity:** :766 says 50%→10%; the shader and the spec's table say 20%→5%.
  - **Fog:** :570 describes it in "world-space depth". The shader uses a reversed-Z NDC difference. :768 says the default is "tunable, experiment"; it is 10.0.
  - **Links and numbers:**
    - The anchor `#camera-view-mode-override` at :286 is broken.
    - :158 cites "camera-views.md Step 9", which does not exist.
    - :191 gives the initial distance as "5.0". It is √29.
  - **Stale deferrals:**
    - :734 still says Alt-menu interception is to be investigated. `windows.rs:764` already suppresses it.
    - :560 still calls the point size a "temporary proxy".
    - The inertia non-goal (:789) is contradicted by DirectManipulation `TRANSLATION_INERTIA` and by the spec's own :816.
    - :234 "(future)" and :290-302 "Planned" repeat settled non-goals.
**Third copies:** The `righting.rs` module doc (:4-18) and the `right_toward_z_up` doc (`mod.rs:1168-1185`) both re-derive the speed profile and timings in spec :380-402. Shrink the module doc.
**Shape:**
  - **Opening:** "This document specifies…" describes the document, not the thing.
    - **Proposed:** *"SfM Explorer's 3D viewport moves like an orbit camera around a target point in front of it; holding Alt moves the target instead, WASD flies, and Maintain Z-up turns the horizon level again after a roll."*
  - **Failure 4:** the Pan, Nodal Pan, Zoom to Fit and near-clip step lists.
  - **Mannered prose:**
    - "without ever losing your bearings" → "while keeping track of where the camera is".
    - "lantern illumination" → "illumination that falls off with distance from the target".
**Recommendation:** update spec. **Fix the misplaced section first**; the Maintain Z-up change (#614) caused it.
**Unclear / incorrect / suspicious:** The Windows DirectManipulation section (:792-1017) is a third of the file and might be better as its own `gui/` spec.

### specs/core/analysis/cluster-census.md
**Summary:** The algorithm and every numeric default match the code: CNM grouping, the P95 screen, the Wilson bound, `sat_pct`, and the group-consistency solve (3.0 px, 1200 bridges). § Callers and `flag_threshold` describe experiment scripts that were never merged to `main`.
**Implementing code:** `analysis/cluster_census.rs` (`CensusParams` :71-100), `cluster_census/group_consistency.rs`, `sfmtool-py/src/analysis/cluster_census.rs:103-117`.
**Inconsistencies:**
> _Status (2026-10-03): **Done** — all four items: § Callers, the `flag_threshold` row and the two "arbitration callers" remarks are deleted (the callers live only on the unmerged branch, which keeps their text, so no draft was written); the spec uses `hi_parallax_deg`, `wilson_z`, `warp_percentile` and `group_a`/`group_b`; § 2 says "≥ 2 observations on posed images"; `analysis/README.md` says "global satisfaction", PR #684. The third-copies and shape items remain open._
  - :276-293 and :304 describe `_finalize_seed`, `census_echo` and `flag_threshold`. `git log -S` finds them only in `7bd2f079` and `ecfa6714` on the unmerged `fork/bootstrap-core-migration`. On `main`, `cluster_census` has test callers only.
  - :301 and :143 say `hi_para`; the code says `hi_parallax_deg`. :303 says "Wilson z"; the code says `wilson_z`. :254 says `ga`/`gb`; the code says `group_a`/`group_b`.
  - :94 says "observed by ≥ 2 posed images". The code counts observations, not distinct images (`:592`).
  - `analysis/README.md:14` says "saturation" where it means satisfaction.
**Third copies:** Paraphrases. `cluster_census.rs:4-48`, the binding docstring and `group_consistency.rs:4-45` each restate the spec. Trim them to a contract plus a link.
**Shape:** The opening "A reconstruction can be internally consistent and wrong." never says what the census is. TEMPLATE.md:88 cites it as a *good* example, which should be reconsidered.
  - **Proposed:** *"The cluster census scores a candidate reconstruction against the raw feature clusters the solve did not use, reporting for the worst pair of viewpoint groups a lower confidence bound on the fraction of well-matched spanning clusters the candidate cannot reproject within 2 px."*
  - § Core promotion notes (:323-333) is residue.
  - § Evidence cites files that are not in the repo.
**Recommendation:** update spec. Delete § Callers and the `flag_threshold` row, or move them to an amendment draft if the seed pipeline is going to merge.
**Unclear / incorrect / suspicious:** The prior audit called `flag_threshold` "forward-looking, correctly labelled". That was wrong: the label was an inline "not yet" marker.

### specs/core/features/track-cluster-matching.md (re-read; first read 2026-09-05)
**Summary:** The algorithm sections (§§ 1-4, choosing `d`, limitations) match the code. The Production Implementation section (:457-968) is still a build brief from before `239ee24`. Of the prior audit's findings, only the `pycolmap.verify_matches` fix has landed.
**Implementing code:** `features/cluster_match/mod.rs` (`BackgroundFloorParams`, `background_floor_clusters{,_lazy,_from_neighbors}`, `clusters_to_pair_matches`), `sfmtool-py/src/matching/cluster.rs`, `feature_match/_cluster_matching.py`, `feature_match/_run.py`, `_commands/match.py`.
**Inconsistencies:**
  - **Still open from 2026-09-05:**
    - `--cluster` is still said to verify and write TVGs (:173-175, :842-887). It writes a clusters-only file.
    - `--camera-model` is still said to be available with `--cluster` (:907-914). It raises UsageError.
    - "not lifted" (:785).
    - A `min_size` CLI flag (:961), which does not exist.
    - COLMAP DB timing claims (:318-337).
    - `search_batch_with_distances` should be the `_ordered` form (:640).
    - `thiserror` (:578), which the code does not use.
  - **New:**
    - The global-threshold table (:355-363) should be deleted.
    - `bg_alpha` should be `alpha`.
    - The code has 5 error variants, not 3.
    - `_lazy`, `_from_neighbors`, `NeighborTable` and `background_floor_clusters_kdf` are not documented.
    - :512 says `lib.rs`; the declaration is in `features/mod.rs`.
    - :736 cites `py_kdforest.rs`, which no longer exists.
**Third copies:** Yes, and **the spec is the copy to shrink**. :520-614, :741-774 and :817-829 are verbatim doc comments and signatures (23+6+4 lines, as found in check 2). Replace them with a linked interface summary plus the invariants: L2 distance, the tie-break and the `alpha < 1` rank cap.
**Shape:**
  - **Stale non-goals:**
    - "Persisting clusters… out of scope".
    - "CLI consumes only the pair output today". The reverse is now true.
    - "Add a `## Cluster matching` section" (done).
    - "consider lifting" (done).
  - **Opening:** "Traditional SfM feature matching is pair-centric." describes what this approach replaces.
    - **Proposed:** *"Track-cluster matching finds correspondences across a whole image set at once, grouping every image's SIFT descriptors that fall within 0.8× their 10th-nearest-neighbour distance into clusters of at most one feature per image, as candidate tracks."*
  - **Mannered prose:**
    - "falls out for free" (×4) → "needs no separate step".
    - "background floods in" → "the count of admitted background neighbours rises sharply".
**Recommendation:** update spec. Rewrite :457-968 as a present-tense contract.
**Unclear / incorrect / suspicious:** `matching_mode="cluster"` (`db_setup.py:130`) cannot be reached from any command, yet :465-468 present it as live.

> _Status (2026-10-03): **Done** — Production Implementation (:457-968) is replaced by an Interface section (linked code pointers; a summary of `background_floor_clusters{,_lazy,_from_neighbors}`, `NeighborTable`, `clusters_to_pair_matches`, the five `ClusterMatchError` variants and `background_floor_clusters_kdf`; the L2, tie-break and `alpha < 1` rank-cap invariants), one defaults table, a Pipeline section and short implementation notes; the copied doc comments and signatures are gone. Every 2026-09-05 and new inconsistency above is fixed: `--cluster` writes a clusters-only file, `--camera-model` with it is a UsageError, "not lifted" is gone (the bindings are re-exported at the `sfmtool` top level), `min_size` has no flag, the COLMAP DB timing claims are removed, the query is `search_batch_with_distances_ordered`, `thiserror`, `lib.rs` and `py_kdforest.rs` are gone. The opening is the proposed sentence, reworded so the 0.8× radius is plainly the seed descriptor's own. The mannered phrases are replaced. The in-solve mode is described as reachable only from `run_global_sfm` / `run_incremental_sfm` (a test fixture uses it), and the `_run_cluster_matching` docstring now says the same; whether to keep that mode is raised on the PR. PR #704._

### specs/formats/sfmr-file-format.md
**Summary:** Entry names, dtypes, shapes, hash order and optional flags match the crate, and every entry the code writes is documented. The version bookkeeping is stale, `derived_xxh128` is mis-described, and there is no Implementations section.
**Implementing code:** `sfmtool-sfmr-format/src/{entries,types (SFMR_FORMAT_VERSION=11 :570),read,write,verify}.rs`; `sfmtool-core/src/reconstruction/data/conversion.rs`; `sfmtool-py/src/io/sfmr.rs`.
**Inconsistencies:**
  - **Version bookkeeping:**
    - :288 says `version` is "`1` to `9`"; the current version is 11.
    - The :252 example says 7.
    - Version History (:2291-2332) has no v10 entry.
    - There is no 7→8 migration.
    > _Status (2026-10-03): **Done** — `version` reads "`1` to `11`", the example says 11, Version History has a v10 entry, and a Version 7 → Version 8 migration section is added (v8 already had a history entry), PR #677._
  - **Hash fields:**
    - :2155 calls `derived_xxh128` "optional", but the verifier requires it at v10+ (`verify.rs:278`), and the :500-510 example omits it.
    - :2014-2021 say `content_xxh128` is computed from "all section hashes". That contradicts :528, which excludes derived.
    > _Status (2026-10-03): **Done** — `derived_xxh128` is described as required from version 10 in the field list and the v9→v10 table, the example includes it, and "Why `content_xxh128`" names the sections it covers and the derived exclusion, PR #677._
  - **Presence rules the reader and verifier do not enforce** (only the writer does):
    - bitmaps ⇒ patch frame (:1480)
    - confidence ⇒ normals (:1483)
    - `embedded_patches` ⇒ patch frame (:1690)
  - **Rigs and frames:** they must appear together (:698), but only `rigs/` is checked, so a `frames/` without `rigs/` passes.
  - **Keypoint bounds:** the writer does not check them (:1572), so it can write a file its own reader rejects.
    > _Status (2026-10-02): **Done** — the writer runs `validate_keypoints`, with tests, commit `b19c676` (#670)._
  - **Usage Examples** (:1812-1897) use `SfmrFileReader`, `write_sfm` and `verify_sfm`, none of which exist.
    > _Status (2026-10-03): **Done** — the code examples are replaced by an Implementations section naming the Rust and Python read, write and verify functions with links to their source, PR #682._
**Format independence:** Twelve confirmed findings. The main ones, with who should replace the name:
  - :112-116 `SfmrReconstruction`/`conversion` → "a reader converts on load".
  - :1000-1007 `THUMBNAIL_SIZE` constants → Implementations.
  - :1099 and :1442-1444 `classify_points_at_infinity`/`materialize_points_at_infinity` → "a consumer that demotes or materialises a point…".
  - :1755 `observation_affine_shape` → delete.
  - :1782-1784 `WriteOptions::zstd_level` → "the writer's default level is 3".
  - Implementation links in normative sections: :113 and :516.
  - Smaller name leaks: :542 `verify_sfmr`, :650 `SfmrCamera`, :40 and :2094 `sfm xform` flags.

  Links to `../core/reconstruction/README.md` and `../gui/goto-point.md` are background and acquitted.
**Third copies:** The hash-slot order appears three times; the :1803 copy is incomplete. The version list appears twice, and the two copies disagree.
**Shape:**
  - **Edit-history residue:** :962-964 "(This line previously read…)", the :1009-1013 "earlier revision" block, and "before this amendment" at :1250 and :1539.
    > _Status (2026-10-03): **Done** — the thumbnail resize line states area averaging without the earlier wording, the "earlier revision" note is removed, and the two "before this amendment" phrases now name the versions (1–4 for `normal_confidence`, 1–5 for `observation_confidence`), PR #683._
  - :1899-1945 compare the format with a directory format that no code reads.
    > _Status (2026-10-03): **Done** — the "Comparison with Directory Format" section is deleted; § File Structure already shows the archive layout, PR #683._
  - The opening passes.
**Recommendation:** update spec. **Discuss** whether the reader and verifier should enforce the presence rules the writer enforces.
**Unclear / incorrect / suspicious:**
  - The lineage rules (:491-494) are not enforced anywhere.
  - `world_space_unit` is not checked against its five allowed values.

### specs/formats/matches-file-format.md
**Summary:** Entry names, dtypes, shapes, the backbone rule, hash order and version gating (`MATCHES_FORMAT_VERSION = 6`) match. The spec states rules no code enforces, lists stale `matching_method` values, gives stale Python examples, and has no Implementations section.
**Implementing code:** `sfmtool-matches-format/src/{entries,types,write,read,verify}.rs`; `sfmtool-py/src/io/{matches,matches_file}.rs`.
**Inconsistencies:**
  - :496, :530, :550 and :851 say member geometry is NaN-free. The writer and verifier do not check this.
    > _Status (2026-10-02): **Done** — the writer refuses NaN member geometry and the verifier reports it, with tests, commit `b19c676` (#670)._
  - :214-220 list `matching_method` values `vocab_tree`, `spatial`, `transitive` and `custom`. The writers emit `flow`, `cluster` and `merged`, none of which is listed.
    > _Status (2026-10-03): **Done** — the field description now lists the values the writers emit (`exhaustive`, `sequential`, `flow`, `cluster`, `merged`), says derived files keep their source's value, and drops the four that nothing writes, PR #681._
  - :577-593 leave out `refine_options.max_keypoint_uncertainty`, which is the threshold behind status 6.
    > _Status (2026-10-03): **Superseded** — the member gate now uses the ZNCC self-similarity radius. The metadata paragraph (:590-597) lists `max_member_zncc_self_similarity_radius` and notes that older files carry `max_keypoint_uncertainty`, commits `086f8e1` (#651) and `c4a9db8` (#654)._
  - :727 says a verifier checks config indices. `verify.rs` does not bound-check them.
    > _Status (2026-10-02): **Done** — the verifier reports out-of-range config indices, with a test, commit `b19c676` (#670)._
  - :907-910 give the `source_selection` nesting condition wrongly. `cluster-selection.md:121` states it correctly.
    > _Status (2026-10-03): **Done** — the paragraph now says the key is present whenever the source carries its own `cluster_selection` record, matching `select.rs` and `cluster-selection.md`, PR #681._
  - :1056-1131 Usage Examples use APIs that do not exist.
    > _Status (2026-10-03): **Done** — the code examples are replaced by an Implementations section naming the Rust and Python read, write and verify functions with links to their source; the verified-file workflow that :677 points at is kept as prose under As part of a Pipeline, PR #682._
  - :660 "Added… without a version bump" and :613 "identity affine" are stale.
    > _Status (2026-10-03): **Done** — the version-3 history note on `member_consistency_residual` is removed (cluster files below version 6 are refused), and status `0 reference` now says its reference→member warp is the identity and its geometry is its detection, PR #681._
**Format independence:** No Implementations section. Findings, with replacements:
  - `ClusterMemberStatus` (:126), `clusters_to_pair_matches` (:417, :961), `u32::MAX` (:599) and the API list at :1010-1019 → Implementations.
  - :592 "the reader's `refine_radius` accessor" → "a consumer uses `patch_size / 2`".
  - :484-494 define the stage by the command that wrote it → "without `cluster_patches/` a file holds detections; with it, rows with status 0-3 are measured".
  - :418-422 describe expansion as the CLI command → keep the rule and move the CLI sentence to background.
  - **Definitions carried by links:**
    - Status 6 (:620-624) → restate as "uncertainty above `refine_options.max_keypoint_uncertainty`".
      > _Status (2026-10-03): **Done** — status 6 is now defined inline as a ZNCC self-similarity radius above the member gate's bar (:625-627), and the link is kept only for details, commits `086f8e1` (#651) and `c4a9db8` (#654)._
    - :650-653 `M_k`, `T_c`, `J` are defined only in `core/patch/cluster-warp-consistency.md`.
    - :901 provenance keys are undefined here.
  - **Opening**, proposed: *"A `.matches` file records feature correspondences among a set of images — pairwise per image pair, or clusters across images — with optional verification or vetting results, and names the `.sift` files whose indexes it uses."*
**Third copies:** The provenance example and the `refine_radius` rule also appear in `cluster-selection.md` and `select.rs`.
**Recommendation:** update spec. **Update code** so the writer and verifier enforce NaN-free geometry and the verifier bounds-checks config indices.

### specs/formats/cluster-selection.md
**Summary:** The options, semantics, errors and provenance match `select.rs`. The document specifies an *operation and reader API*, not an on-disk format, so it is filed in the wrong place.
**Implementing code:** `sfmtool-matches-format/src/select.rs` (`ClusterSelect`, `select_clusters`, `refine_radius`, `cluster_worst_consistency`); `sfmtool-py/src/io/matches_file.rs:311,340,378`.
**Inconsistencies:**
  - :91 says "(sorted)". The code also removes duplicates.
  - :63-71 do not say that `accepted_statuses` without `reference` also produces the `0xFFFFFFFF` sentinel.
  - The worst-consistency accessor returns nothing when there is no `cluster_patches/`, and the spec does not say so.
**Format independence:** The opening names `MatchesData::select_clusters` and `MatchesFile.select_clusters`. That is wrong for a format spec and right for an operation spec.
  - **Proposed:** *"Cluster selection reads a cluster-backbone `.matches` file and writes a new one holding only the clusters and members that pass a predicate on member status, image name and source cluster id."*
  - :92-94 "byte-identical to before the option existed" is residue.
**Third copies:** The six semantic steps are copied almost verbatim into the `select.rs:112-150` doc.
**Recommendation:** discuss. Move it to `specs/core/` in template shape, and leave the provenance record's format contract in `matches-file-format.md`.

### specs/formats/kdf-file-format.md
**Summary:** The entries, metadata, chunk byte layout, hash tree, defaults and version 3 match. The reader enforces two rules the spec does not state, and about 200 lines of version-1 study still refer to a "version-2" writer.
**Implementing code:** `sfmtool-kdf-format/src/{types,read,verify,write}.rs`; `sfmtool-py/src/spatial/kdf.rs`.
**Inconsistencies:**
  - :90 allows any root address. `read.rs:1296` requires `[0, 0]` for a nonempty tree.
  - :84 says readers resolve codes through `node_kinds`. `read.rs:1265` requires exactly `["internal","leaf"]`, so state the codes as fixed.
  - "version-2" appears at :494 and :654, in `types.rs:77,113` and in `lazy-kdforest-query.md`. The writer emits version 3.
    > _Status (2026-10-03): **Done** — :494 and :654 no longer call the current layout version 2; the doc comments in `types.rs`, `summary.rs` and `sfmtool-py/src/spatial/kdf.rs` no longer name a version, PR #680._
  - The spec has no history of what versions 2 and 3 changed.
    > _Status (2026-10-03): **Done** — § Version says version 2 replaced version 1's two layouts with the single corpus and added the SIFT geometry corpus, and version 3 changed only the integrity directory to one digest per section, PR #680._
**Format independence:**
  - :360-364 `KdfFile::verify_content` and `verify_kdf` → "the digest check runs first, then the structural checks", with the links moved to § Implementations (:466).
  - :686 names the reader option `max_metadata_bytes`, and :605 links a script.
  - The other links are background (acquitted).
  - **Opening**, proposed: *"A `.kdf` file is an approximate nearest-neighbour index: a set of fixed-width vectors, such as SIFT descriptors, plus several binary spatial-partition trees over them, laid out so a reader can answer queries without loading the whole file."*
**Shape:**
  - "Sizing and tradeoffs" (:488-688) is a dated version-1 study. Move it to a rationale document.
  - **Mannered prose:**
    - "fails honestly" → "reports an error".
    - "the conventions stop being free" → "the conventions have a measurable cost".
    - "carries the weight" → "the properties a reader relies on".
**Recommendation:** update spec. Also fix the "version 2" comments in `types.rs`.

### specs/formats/archive-container.md
**Summary:** The container rules (STORE, one zstd frame per entry, JSON bytes, big-endian u128 digest fold) match the code. About 60% of the file (:146-410) is a crate spec for `sfmtool-archive-io`, placed in a format spec with no Implementations heading.
**Implementing code:** `sfmtool-archive-io/src/lib.rs` (`SectionDigests` :58, `read_zst_entry` :110, `write_json_entry` :405, `json_entry_bytes` :396, `format_hash` :552, `WorkspaceMetadata` :47).
**Inconsistencies:**
  - :150-155 say the crate is used "by the five format crates… and nothing else". `sfmtool-core` depends on it (`edited.rs:22` `format_hash`).
  - The API block leaves out `json_entry_bytes`, `WorkspaceContents` and `WorkspaceMetadata`.
  - The "No schema" non-goal is stale: the crate defines the shared `workspace` object.
  - The error list at :300-302 omits `KdfError`.
**Format independence:**
  - :56-57 "exactly what `serde_json::to_vec` produces" → a writer rule: compact UTF-8 JSON, no whitespace between tokens, no trailing newline. The reader side should also be stated: a reader must not require any particular formatting.
  - :49-52 name the `write_*` functions → "the level is the writer's choice; readers accept any level".
  - :63-64 "the crate refuses to compile for big-endian" → "a writer on a big-endian host byte-swaps".
  - :142-144 "`verify_*` function" → "a verifier".
  - **Opening**, proposed: *"The archive container is a ZIP file whose entries are all stored uncompressed at the ZIP level, each holding one zstandard frame of compact UTF-8 JSON or a raw little-endian array, with XXH128 digests kept in `content_hash.json.zst`; `.sift`, `.matches`, `.sfmr`, `.camrig` and `.kdf` are all built on it."*
**Shape:** Mannered prose:
  - "Little-endian is not negotiable" → "Numeric data is always little-endian".
  - "so that 'common' never becomes 'load-bearing'" → "so correctness does not depend on the allocator's alignment".
**Recommendation:** update spec. Split out an archive-io crate spec and leave a short Implementations section here.

### specs/formats/sift-file-format.md
**Summary:** Entry names, dtypes, shapes, hash order, version check and thumbnail rules match. **The definition of `feature_tool_xxh128` is wrong.**
**Implementing code:** `sfmtool-sift-format/src/{types,write,read,verify}.rs`; `sfmtool-py/src/io/sift.rs:106`; `src/sfmtool/sift/file.py:83,438`.
**Inconsistencies:**
  - :124-126 and :213-217 say `content_hash.feature_tool_xxh128` is "computed once during workspace initialization and propagated", with "no prescribed algorithm". In fact:
    - `write.rs:48` recomputes it as XXH128 of the stored `feature_tool_metadata.json` bytes and ignores the caller's value.
    - `verify.rs:47-54` enforces exactly that computation.
    - The workspace directory hash is a *different* computation (`file.py:96-111`).
    - A real file confirms the two differ: `…sift-colmap-dsp-max3000-a3e02aa8…/*.sift` stores `e43b2d42…`.
    - So :56-57 equate two different values.
    > _Status (2026-10-03): **Done** — the spec defines `feature_tool_xxh128` as the XXH128 of the stored `feature_tool_metadata.json` bytes, enforced by the verifier; the path convention calls the directory component the workspace's feature-cache hash and says it is a different value; § Feature tool hash computation now separates the two and states that `.sfmr` and `.matches` copy the `.sift` field per image (checked: `colmap/io.py`, `feature_match/_db_populate.py`, `_undistort_images.py`), which answers the **Check** in the recommendation. The example's hash no longer reuses the directory hash from `matches-file-format.md`. PR #678._
  - :160-163 say features are ordered by descending size. Neither the writer nor the verifier checks this; only the extractors sort.
**Format independence:**
  - :40-42 name the crate and `SIFT_FORMAT_VERSION` → "a conforming writer writes 1; a reader rejects newer versions".
  - :177-179 and :201-206 "In this repository…" → Implementations.
  - The `sfmtool` command and backends are named at :29-30, :97-99, :193-194 and :214-215.
  - Informal prose: "it's nice to avoid re-computing" (:28-31).
**Recommendation:** update spec. Define the field as XXH128 over the stored metadata bytes and separate it from the directory hash. **Check** whether `.sfmr` and `.matches` propagate the directory hash or the `.sift` field.
**Unclear / incorrect / suspicious:** Version 0 is accepted without a check.

### specs/formats/sfmtool-camera-models.md
**Summary:** The mathematics (knots, gauge, linear tail, fold gate) and serialization rules match the code, and the parameter names and order match `spline_parameter_names`. There is one internal contradiction, and the spec is heavily tied to the implementation.
**Implementing code:** `camera/intrinsics.rs:206,253`; `camera/intrinsics/registry.rs:258-259,363,409,459`; `camera/distortion/bspline.rs`; `distortion/kernels/sfmtool_{fisheye,pinhole}.rs`.
**Inconsistencies:**
  - :175-176 say the correction is "held constant beyond" `d_max`. The code continues along the end tangent (`bspline.rs:191-195`), as :90-94 say.
  - :154-158 put `get_bspline` in `intrinsics.rs`; it is in `intrinsics/registry.rs:409`.
  - :88-89 and :140-141 say a non-positive `d_max` is the identity. A reader rejects such a file (`registry.rs:445-450`), so this is an in-memory rule only.
**Format independence:** Seven findings.
  - :154-158 and :220-226 name `CameraModel`, `SfmrCamera` and the registry → Implementations.
  - :301-304 error type names → "a reader rejects the camera".
  - :307-321 are a Testing section inside a format spec.
  - :119-131 never state the file rule for a spline that violates the constraints.
  - :186 relies on a camera-frame convention defined only in `sfmr-file-format.md`. Restate it here.
  - :126 "fold-gated domain" is undefined.
  - **Opening**, proposed: *"`SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE` are camera models, named by those strings in a camera's `model` field, that project a camera-frame ray to a pixel through a one-focal base projection plus a radially symmetric cubic B-spline correction."*
**Recommendation:** update spec.
**Unclear / incorrect / suspicious:**
  - `spline_has_distortion` uses `DISTORTION_EPS = 1e-12`, but :136-137 say the zero test is exact.
  - `get_bspline` accepts key spellings such as `bspline_c01` and `bspline_c+1`.

---

## Code without specs

| Surface | User-facing? | Spec |
|---|---|---|
| 30 CLI commands (all except `explorer`) | yes | one `specs/cli/<cat>/<cmd>-command.md` each |
| CLI `explorer` | yes | **none** (mentioned at gui/architecture.md:385) |
| `_commands/_range_options.py` | no (shared helper) | none, and to-colmap-bin misattributes its logic |
| crate sfm-explorer | yes | gui/architecture.md plus 27 gui specs |
| crate sfmtool-archive-io | no | fused into formats/archive-container.md |
| crates sfmtool-{sift,matches,sfmr,camrig,kdf}-format | file formats | formats/* |
| crate sfmtool-colmap | no | **none dedicated** (cli/colmap-interop/* only) |
| crate sfmtool-core (per module) | no | core/<module>/ |
| core `numeric` | no | **none** |
| crate sfmtool-progress | no | gui/operation-progress.md |
| crate sfmtool-py (binding surface) | Python API | **none dedicated** |
| explorer `state` | no | **none dedicated** (spread across mcp-server, operation-progress) |
| explorer `elide` | no | none (acceptable) |
| explorer (all other modules) | mostly yes | a gui/* spec each (see check 5 in the agent notes) |
| py `align`/`analyze`/`merge`/`motion`/`camera`/`camrig`/`colmap`/`compare`/`feature_match`/`rig`/`sift`/`visualization`/`web_export`/`xform` | via CLI | the matching cli/* or workspace/* spec |

### `sfm explorer` (src/sfmtool/_commands/explorer.py)
**What it does:** It launches the SfM Explorer GUI on a reconstruction through the Python bindings. It is the only CLI command with no command spec.
**Why it matters:** It is user-facing. docs/index.md shows the GUI but cannot point to a command.
**Recommendation:** Write `specs/cli/visualization/explorer-command.md`, a short spec that covers the flags and links to `gui/architecture.md`.

### crates/sfmtool-py (binding surface)
**What it does:** It is the whole Python API (`sfmtool._sfmtool.*`). It is cited only piecemeal from individual core specs.
**Why it matters:** It is internal but load-bearing. This audit found three specs whose Python examples call APIs that do not exist (sfmr, matches, focal-vote's `estimate_homography`).
**Recommendation:** Add a `specs/core/README.md` note or a short index spec that maps binding submodules to their specs. A generated listing would drift less than a handwritten one.

### crates/sfmtool-colmap
**What it does:** It reads and writes the COLMAP binary files and the SQLite database.
**Why it matters:** It is internal but load-bearing for four CLI commands.
**Recommendation:** Acceptable without a spec for now. Add a paragraph to `cli/colmap-interop/README.md` covering ID and coordinate conventions.

---

## Carried forward from the 2026-09-05 audit

The 2026-09-05 audit (at `077a83d`) is retired in favour of this report. These
are its findings that were still open on 2026-10-03, each checked against the
current files; line numbers are current. 09-05 findings that this report already
tracks are not repeated: the `track-cluster-matching.md` drifts and its
Production Implementation rewrite, `cluster-census.md` `flag_threshold`, the
duplicate-prose leads, the `sfmtool-py` binding surface, the uncited flow specs,
and the opening paragraphs listed in check 4.

### Spec text that disagrees with the code (code is right unless noted)

- **`specs/core/geometry/estimate-intrinsics.md`:**
  > _Status (2026-10-03): **Done** — the spec keeps 42 captures / 6 fisheye and states the 27 strong votes that never escalate, and the `escalation_reasons` doc comment points at the spec for the counts instead of restating 40/4; the grid-step count reads 2.58; `rotation_railed` gates on the majority family `vote.family`; the binding note says `columns=None` is `Fixed` with both columns; `sfm estimate-intrinsics --json` emits `screening_vote` (test added) and `estimate-intrinsics-command.md` documents it, PR #693._
  - :237-238 say "42 captures … 6 of them fisheye"; `estimate_intrinsics.rs:107-108` says 40 captures, 4 fisheye. Settle the count where it was measured, then cut the code doc to the four cut points plus a link.
  - :224 says the ratio 1.153 is "2.2 grid steps". At step 1.0566 it is 2.58.
  - :216 says "the consensus came from the rotation family". `escalation_reasons` gates on the majority family (`vote.family`).
  - :268-271 do not say that the binding's `columns=None` means `Fixed` with both columns, not `"auto"`.
  - Command side: the `--json` payload (`_commands/estimate_intrinsics.py:459-477`) never includes `screening_vote`, so the pinhole numbers the spec tells callers to read from it cannot be reached from `sfm estimate-intrinsics`. Expose it, or say so in `estimate-intrinsics-command.md`.
- **`specs/cli/reconstruction/motion-command.md`:**
  > _Status (2026-10-03): **Done** — § Tests lists the cases in `tests/test_motion_report.py`; the build-plan table is replaced by an § Implementation table of the modules the command uses; the settled "Extrapolation order" question is removed; the KerryPark citation is dropped; `shared_points` notes the 90° angle filter; the stride options state their minimums, PR #686._
  - :592-628 § Tests describes a 10-frame fixture and a "NaN round-trips as `null`" test that do not exist; the tests are in `tests/test_motion_report.py`.
  - :630-641 "Code and patterns to build on" is a build plan, and two of its rows name helpers the motion code does not use.
  - :645-647 open question "Extrapolation order" is settled by the shipped linear+quadratic minimum.
  - :336 cites "KerryPark 831→832"; kerry_park has 24 frames.
  - :461 does not say that `shared_points` passes through the 90° angle filter in `_image_pair_graph.py:46`, so a 0 can mean "filtered".
  - :62 omits that `--max-stride` is `IntRange(min=2)` (`motion.py:44`).
- **`specs/core/patch/candidate-track-spawning.md`:**
  > _Status (2026-10-03): **Done** — the change order is rewritten in the present tense as § API › "Seeding localization with `starting_keypoints`", which gives the per-view `Option<[f64; 2]>` seed shape and says how it differs from `refine_keypoints`'s; the `high_reproj` gate now says a projection failure makes the RMS infinite; `test_parent_at_infinity` in `tests/rust_bindings/test_spawn_candidate_tracks_rust_bindings.py` covers the parent-at-infinity `ValueError`, PR #689._
  - :128-140 is a change order ("predates the parameter", "As part of this change"). Write it in the present tense inside § API.
  - :134 says "same shape as `refine_keypoints`'s"; `localize_keypoints` takes `Vec<Option<[f64;2]>>`.
  - :72-73 do not say that a projection failure forces `sum_sq = INFINITY` (`spawn.rs:367`), so it reports `high_reproj`, not `bad_triangulation`.
  - The parent-at-infinity `ValueError` (:125) has no test.
- **`specs/gui/mcp-server.md`:**
  > _Status (2026-10-03): **Done** — the spec describes `cli.rs`'s refusal of unknown `-` options and bad `--mcp=` values, says the exact `set_view` form requires `position` and `target_distance`, records the schema's `"minimum": 16` on `max_dimension` (which the parser does not check), and drops the "Before this pair existed" prior-state text. The two sizes in the screenshot text block measure different things (the returned PNG after crop and scaling, and the uncropped, unscaled target), so both stay and the spec says which is which, PR #688._
  - :51-56 say `cli.rs` "treats everything else as a path". `cli.rs:97` refuses any `-`-prefixed argument, and `--mcp=<non-number>` is an error (`:92`).
  - :1294-1307 say each piece of an explicit `set_view` call "is preserved". With `orientation_wxyz`, `tools.rs:885-887` require `position` and `target_distance`.
  - The screenshot text block states the size twice: `server.rs:321` prefixes `{w}×{h} px.` to a caption that already ends `…, {w}×{h}` (`mcp/mod.rs:1926-1931`), and the two can disagree for a panel. Drop one and say which in the spec.
  - :4770 does not mention the schema's `"minimum": 16` on `max_dimension` (`tools/catalog/background.rs:68`).
  - :1054-1058 describe the prior state ("Before this pair existed…").
- **`specs/gui/camera-intrinsics.md`:**
  > _Status (2026-10-03): **Done** — `DistortionSample`'s doc comment in `report.rs` now puts the arrow's tail at `pixel`; the spec lists the hover line's `outside the model's domain` state; the change and phase residue (opening, Terminology's Before/After table, first-draft and phase asides, "today", the thin-prism history, § Open questions) is rewritten in the present tense; § Testing names the test files that hold each group of tests; `derived.rs` says "defaults to", PR #694._
  - Code doc wrong: `sfmtool-core/src/camera/report.rs:130-132` says the arrow runs from `reference` to `pixel`; the spec and `field.rs` draw it with the tail at `pixel`.
  - :824-828 list the hover `distortion` line's states without `outside the model's domain` (`hover.rs:89`).
  - Change and phase residue at :10, :48, :58 (Before/After table), :84, :464, :478, :882, :1032, :1037, :1231 (§ Testing as a to-build list) and :1461-1463.
  - `intrinsics_detail/derived.rs:18` says "will default to" about a default that has shipped.
- **`specs/core/camera/sfmtool-pinhole-kernels.md`:**
  > _Status (2026-10-03): **Done** — the zero-spline section lists the `MIN_BSPLINE_COEFFS` case; the Inverse section states the `PINHOLE_AXIS_EPS` short-circuit, says the fisheye falls back to `equidistant_to_ray` (spec and the `undistort_sfmtool_pinhole` doc comment), and links to `sfmtool-fisheye-kernels.md` § "Inverse" for the tail and Newton solver instead of repeating them, PR #690._
  - :178 defines `bspline_is_inactive` without the `len() < MIN_BSPLINE_COEFFS` case (`bspline.rs:47-48`).
  - The Inverse section (:70-99) omits the `r_d < PINHOLE_AXIS_EPS` short-circuit (`sfmtool_pinhole.rs:72`), which makes :74 "`r_d ≤ 0` is `ρ = 0`" unreachable.
  - :80-82 and `sfmtool_pinhole.rs:60` say the identity fallback is "the policy `sfmtool_fisheye_to_ray` applies"; the fisheye falls back to `equidistant_to_ray` (`sfmtool_fisheye.rs:195-197`).
  - :84-99 repeat `sfmtool-fisheye-kernels.md:73-85`. Link to it instead.
- **`specs/core/geometry/affine-factorization.md`:**
  > _Status (2026-10-03): **Partially done** — the spec, the `metric_upgrade` doc comment and the Python docstrings say it also returns `None` when the largest eigenvalue of `Q` is not positive (smaller eigenvalues are clamped), and the spec and the `rounds` doc comment say `rounds == 0` is legal and what it returns, PR #692. The keep-or-retire question for the module is still open._
  - :150-151 and `affine_factorization.rs:414-415` say `metric_upgrade` fails only on degenerate systems. It also returns `None` when `Q` is not positive-definite (`:466`).
  - :56 does not say that `rounds == 0` is legal (`:263`).
  - Discuss: the module has no caller outside its tests. Keep it or retire it.
- **`specs/core/geometry/rotation-locked-resection.md`:**
  > _Status (2026-10-03): **Done** — the spec states the up-front `None` on a length mismatch or fewer than `max(min_inliers, 1)` observations, the exclusion of non-finite or zero-length rays and non-finite points, and names `rotation_init.rs` as the only in-repo caller, PR #691._
  - :113-114 do not state the up-front `None` when `uv.len() != n` or `n < min_inliers.max(1)` (`resect_translation.rs:159`).
  - The spec does not say that rays with a non-finite or near-zero `pixel_to_ray` are excluded (`:172-181`).
  - :11-12 present "a rig calibration, an external attitude" as callers; the only caller is `rotation_init.rs`.
- **`specs/core/features/gpu-optical-flow.md`:**
  > _Status (2026-10-03): **Done** — the inverse-search sentence gives the `patch_size²` loop (64 or 144 pixels); § Hybrid CPU/GPU Routing describes both checks, `gpu_start_scale` in `compute_optical_flow_timed` and the per-level check in `refine_flow_at_level`; "Future Work" became a present-tense sentence in § Jacobi Kernel linking the new [`drafts/gpu-optical-flow-jacobi-shared-memory-amendment.md`](../specs/drafts/gpu-optical-flow-jacobi-shared-memory-amendment.md), which links back; the "Option B from spec" comments in `inverse_search.wgsl`, `densify.wgsl` and `dis_pipeline.rs` are removed and `gpu/mod.rs` is retitled to cover the whole GPU pipeline, PR #697._
  - :131 says "64 pixels (8×8 patch)". The shader loops `params.patch_size²`, which is 12×12 under `high_quality`.
  - :56-61 place routing only in `refine_flow_at_level`; `optical_flow/mod.rs:177` also picks a `gpu_start_scale`.
  - :279 "Future Work" heading. Turn it into a present-tense sentence plus a draft.
  - In code: `inverse_search.wgsl:1` cites an "Option B from spec" that no spec has, and `gpu/mod.rs:4` titles the module "variational refinement" although it also runs DIS, the pyramid and upsampling.
- **`specs/core/features/optical-flow.md`:** the opening never says what flow is for, and :6-14 Motivation argues for building it. `variational.rs:41-43` says Jacobi needs "~1.3-2×" the SOR count while `params.rs:26-29` says "roughly 4/3×"; keep the number in one place.
  > _Status (2026-10-03): **Done** — the opening says what flow is for in sfmtool (flow-based matching, `sfm motion` on image sequences, `sfm flow`) and why it is a Rust DIS implementation, replacing § Motivation; the Jacobi-to-SOR ratio lives only on `DisFlowParams::variational_jacobi_iterations` ("roughly 4/3×"), and `VariationalParams::jacobi_iterations` links to it, PR #701._
- **`specs/core/features/sift.md`:**
  > _Status (2026-10-03): **Partially done** — the opening is a present-tense purpose paragraph, the interface section states the detect/describe split as the design, and § Lazy descriptors describes the per-octave build (`build_chain` + `extend_octave`) and on-the-fly gradients as they are, PR #703. The pipelining-section move stays open for discussion; `sift-command.md` links to the section's anchor._
  - The opening is still a conditional-voice proposal.
  - :522 "**Yes — split keypoint finding…**" answers a question from a decision memo.
  - :692-694 say the pyramid "can be rebuilt per-octave on demand … to settle with benchmarks"; `ScaleSpace::build_chain` and `extend_octave` already do this.
  - Discuss: § Extraction-orchestration pipelining (:398-510) is Python CLI content and could move to `sift-command.md`.
- **`specs/core/features/track-cluster-matching.md`** (two items not covered above): :474-480 present the `d = 28` prototype counts as what the production run reproduces, while the tables at :119-124 and :137-144 do not say they were measured at `d = 28`; § Location (:509-512) does not mention `covisibility.rs` and `covisibility/`.
  > _Status (2026-10-03): **Done** — both end-to-end tables say they were measured at `d = 28`, the production counts are given as agreeing with that table at `d = 28`, and the Interface section links `covisibility.rs` and `covisibility/` and points at their specs, PR #704._
- **`specs/cli/image-feature/match-command.md`:** `_run_matching` records `matcher_options["min_size"]` (`feature_match/_run.py:42, 95`), but no flag sets it and the Options table does not mention it. Document it as fixed at 2, or expose it.
  > _Status (2026-10-03): **Done** — `match-command.md` § Options now says `--cluster`'s minimum cluster size is fixed at 2, has no flag, and is recorded as `min_size` in the matcher options, PR #695._
- **`specs/cli/colmap-interop/from-colmap-bin-command.md`:** does not mention the `UsageError` for a non-`.sfmr` output (`from_colmap_bin.py:84-85`).
  > _Status (2026-10-03): **Done** — the `--output` row of the Options table says a non-`.sfmr` extension is a usage error, PR #696._
- **`specs/gui/panel-layout.md`:** :435-438 omit that a non-numeric `sfm_explorer_layout` value produces `Not a layout file` (`layout.rs:620-621`).
  > _Status (2026-10-03): **Done** — the version-tag rule in § Validation now says a tag that is not a non-negative integer is `Not a layout file`, PR #698._
- **`specs/core/patch/cluster-patches.md`:** :228 "## Consumers (future work, out of scope here)" is a future-work list in a standing spec. Move it to a draft.
  > _Status (2026-10-03): **Done** — § "Consumers" lists the built consumers (cluster selection, cluster covisibility, Resect Image, the track-at-pixel clusters stage, the nearby clusters source, source clusters, the viewer's index files) in the present tense with links, and says in one sentence that `--derive-pairs`, `embed-patches` and `solve` do not use patch clusters; the three unbuilt ideas moved to `specs/drafts/cluster-patches-consumers-amendment.md`, which links back. PR #699._

### Code doc comments that repeat the spec (shrink to contract + link)

- `sift/extract_sfmtool.py:238-256`, the `_extract_workers` docstring.
  > _Status (2026-10-04): **Done** — the docstring now states the contract (the default count, the memory cap, the unknown-size fallback, the `SFMTOOL_SIFT_EXTRACT_WORKERS` override) and points at `sift.md` § Extraction-orchestration pipelining for the reasoning; that section now also says an override below 1 counts as 1 and a non-integer one is ignored with a warning, branch `finding-fix-20-sift-extract-workers-docstring`._
- `sift/mod.rs:310-329`, the proof of the cap-aware walk.
- `geometry/resect_translation.rs:95-116`, the `residual_norm` chirality argument.
- `motion/recon_discontinuity.py:689-700`, the threshold rationale, which is already in `constants.py`.
- `optical_flow/gpu/mod.rs:98-103`.
- `kernels/sfmtool_pinhole.rs:87-110`.
- `image_detail/intrinsics/field.rs:4-65`, a 62-line module doc.

### Opening paragraphs not covered by check 4

Fix one spec per PR.

- `core/camera/camera-model-registry.md`. Proposed: *"A camera is represented two ways in sfmtool, a loosely-typed record mirroring the on-disk reconstruction and a closed enum the algorithms compute with; this spec says why the split is kept and how the conversion stays exhaustive and generated in one place."*
- `core/patch/cluster-patches.md`. Proposed: *"A patch cluster is a group of matched features fitted to the actual image content: one member is the reference, and every other member carries a photometrically refined affine warp mapping the reference's patch into its image."*
  > _Status (2026-10-03): **Done** — added an opening paragraph based on the proposal, saying "every other kept member" (rejected members are stored too) and naming what the spec covers. PR #699._
- `cli/reconstruction/embed-patches-command.md`. Proposed: *"Rewrites a reconstruction so it no longer depends on the .sift files it was solved from: each observation's pointer into a .sift file becomes an image patch and keypoint stored inline, producing a self-contained .sfmr."*
- `core/spherical/per-spherical-tile-source-stack.md`. Proposed: *"Panorama work divides the sphere into small tiles and asks, per tile, what each source photograph saw in that direction; this gathers exactly that, each source warped into the tile's frame as an image pyramid."*
- Lower priority: `flow-based-matching`, `affine-factorization`, `gui/action-log`, `gui/camera-intrinsics`, `gui/mcp-server`, `member-coherence-validation`, `gui/patch-rendering`, `cluster-patches-command`, `localize-keypoints-command`, `motion-command`, and the `specs/drafts/*-amendment.md` files, each of which opens "Amends [link]…".

### Surfaces without a spec

- `sfmtool-core/src/features/feature_match/` (descriptor distance, best match, ratio test, `geometric_filter.rs`). Write `specs/core/features/descriptor-matching.md` and link it from `match-command.md`.
- `sfmr-file-format.md` § "Conversions happen at the I/O boundary" (:104) does not name `sfmtool-core/src/geometry/convention/`. Put it in the Implementations section that priority 3 asks for.
- Smaller: `viewport-navigation.md` names `ViewportCamera` (:469) without linking `camera/viewport/`; `solve-command.md` does not name `_global_sfm.py` or `_incremental_sfm.py`; `match-command.md` does not state the geometric filter's model and thresholds (`feature_match/_geometric_filter.py`).
  > _Status (2026-10-03): **Done** for `match-command.md` — `sfm match` does not use `_geometric_filter.py` (only `sfm densify` does); the new § Geometric Verification says every verifying mode, `--derive-pairs` included, uses COLMAP's default `TwoViewGeometryOptions` and names the affine-shape filter's thresholds as belonging to `densify`, PR #695. The `viewport-navigation.md` and `solve-command.md` items are still open._
  > _Status (2026-10-03): **Done** for `viewport-navigation.md` — it links `ViewportCamera` to `viewer_3d/camera.rs` and `compute_fit` to `viewer_3d/framing.rs` (the type lives in `sfm-explorer`'s `viewer_3d/`; `sfmtool-core/src/camera/viewport.rs` holds the `Camera` it wraps), PR #698. The `solve-command.md` item is still open._

### Specs the 2026-09-05 audit read

The 2026-09-05 audit read the specs below against their code; their open items
are in the sections above. Each is kept as a heading so the `audit-specs`
sampler, which counts `### specs/…` headings in `reports/*-spec-audit.md`, still
treats them as audited. (`gui/point-track-detail.md`, also read then, has been
replaced by `gui/track-view.md`.)

### specs/cli/colmap-interop/from-colmap-bin-command.md

### specs/cli/image-feature/match-command.md

### specs/cli/reconstruction/motion-command.md

### specs/core/camera/epipolar-curves.md

### specs/core/camera/sfmtool-pinhole-kernels.md

### specs/core/features/gpu-optical-flow.md

### specs/core/features/optical-flow.md

### specs/core/features/sift.md

### specs/core/geometry/affine-factorization.md

### specs/core/geometry/estimate-intrinsics.md

### specs/core/geometry/rotation-locked-resection.md

### specs/core/patch/candidate-track-spawning.md

### specs/gui/camera-intrinsics.md

### specs/gui/mcp-server.md

### specs/gui/panel-layout.md

## Top priorities

1. **Code bugs and behavioural divergences found by the deep reads.** Fix each in the code and add a test:
   - `sfm sift --draw --tool X` inside a workspace draws the workspace tool's features (`sift/file.py:488-493`).
   - A length mismatch in the constellation query raises `OSError` instead of `ValueError` (`kdf.rs:70`).
   - The focal-vote rotation cap visits up to 119 images, not 60 (`focal_vote.rs:1036`).
   - The matches verifier does not bounds-check config indices, and no layer enforces NaN-free member geometry.
   - The sfmr writer can write out-of-bounds keypoints that its own reader rejects.

   > _Status (2026-10-01): **Done**, each with a test, in commit `b19c676`
   > (#670). The `--draw --tool` bug is fixed by a keyword-only
   > `ignore_workspace` on `get_sift_path_for_image` and `draw_sift_features`.
   > The constellation length checks return `KdfError::InvalidQuery`, so they
   > raise `ValueError`. The rotation-scan stride rounds up. The matches writer
   > refuses NaN member geometry, and the verifier reports it and out-of-range
   > config indexes. The sfmr writer runs `validate_keypoints`. The specs
   > already described the fixed behaviour; each finding above is marked. The
   > other spec-side items in those sections, and priorities 2–5, remain open._
2. **Format specs that define a field wrongly:**
   - `sift-file-format.md` defines `feature_tool_xxh128` as a workspace-propagated value with no fixed algorithm. It is the XXH128 of the stored metadata bytes, and verifiers enforce that.
     > _Status (2026-10-03): **Done** — field definition corrected and separated from the directory hash; see the sift-file-format section. PR #678._
   - `sfmr-file-format.md` says versions are "1 to 9" (the current version is 11), has no v10 entry, and calls `derived_xxh128` optional when the verifier requires it.
     > _Status (2026-10-03): **Done** — see the sfmr-file-format section below, PR #677._
   - The kdf version is given as 2 in `lazy-kdforest-query.md`, in two places in `kdf-file-format.md` and in `types.rs`; the code is at 3.
     > _Status (2026-10-03): **Done** — every mention now states version 3 or describes version 2 as history, PR #680._

   Another tool writing these files would get them wrong.
3. **Format specs are not independent of the code.** Six of the seven format specs with hits have no *Implementations* section. Adding one to each is the fix that resolves most of the 45 or so failure-6 findings. `archive-container.md` should split its crate API into its own spec, and `cluster-selection.md` should move out of `formats/`.
4. **Standing specs that describe code which is not there:**
   - `cluster-census.md` § Callers and `flag_threshold` refer to an unmerged branch.
     > _Status (2026-10-03): **Done** — both deleted, PR #684._
   - `track-cluster-matching.md` :457-968 is still a build brief, and 7 of the 2026-09-05 findings on it remain open.
   - The global-threshold parameter table in `track-cluster-matching.md` lists knobs that exist only in an experiment.
     > _Status (2026-10-03): **Done** for `track-cluster-matching.md` — the build brief is rewritten as a present-tense contract and the global-threshold table is deleted, PR #704._

   Delete these, or move them into drafts.
5. **Viewport navigation:** the Maintain Z-up insertion (#614) split the fly and tilt paragraphs in `viewport-navigation.md` (:412-435). Fix that, together with the zoom-to-fit formula and the indicator size and opacity numbers, which disagree with the code.
   > _Status (2026-10-03): **Done** — paragraphs moved back, zoom-to-fit formula, indicator size and opacity corrected, PR #685._

The opening-paragraph replacements (check 4 and the per-spec sections) should
land **one spec per PR**. Each one is a claim about the code, and a reviewer
checks it properly only when reading it alone.
