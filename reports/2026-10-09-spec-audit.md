# Spec audit — 2026-10-09

**Sample:** 17 of 202 specs read against their code. Seed `20883`; the pool
was the never-audited specs (153 candidates). The random draw gave 10:

- `specs/core/features/kdf-layout-measurements.md`
- `specs/core/features/descriptor-matching.md`
- `specs/core/patch/blur-matched-zncc.md`
- `specs/core/patch/cell-plane-normals.md`
- `specs/core/geometry/relative-pose.md`
- `specs/cli/reconstruction/xform/refine-normals-command.md`
- `specs/core/geometry/rotation-init.md`
- `specs/cli/image-processing/undistort-command.md`
- `specs/core/patch/fronto-parallel-patch-cache.md`
- `specs/core/patch/keypoint-localization-consensus-basis.md`

Seven more were read regardless of the draw:

- **The code changed materially since the 2026-10-06 audit (`dd4f6722`):**
  - `specs/core/patch/cluster-patch-refinement.md` (#870: piecewise refinement, refined-shape gates)
  - `specs/core/patch/reference-view.md` (#855, #857, #871)
  - `specs/core/bench/editable-track.md` (#857, #871; last read 2026-09-26)
  - `specs/python-bindings.md` (#866, #868: public binding modules, no flat root)
- **Format specs with a check-6 finding:** `sfmr-file-format` (version 12,
  #857), `matches-file-format` (versions 8-10, #870) and `camrig-file-format`
  (one example slip).

> _Status (2026-10-17): the camrig example slip is **Done** — branch `fix-round-09-camrig-example-version`._

The mechanical checks below cover all 202. This report replaces
`2026-10-06-spec-audit.md`. That report's findings that are still open are
re-measured under **Carried forward**, and the specs it and the earlier audits
read are listed there, so the sampler keeps counting them as audited. Every
finding was checked against `main` at `13c1578e` (#892).

The 2026-10-06 report's header said 18 specs were read; it has 16 sampled
headings (10 drawn and 6 others). The coverage list below uses the headings.

Non-goals and deferral entries checked in the sampled specs: 73 in all. Two
no longer hold (both in `editable-track.md`), and four point at the wrong place
or at code that does not exist (reference-view and blur-matched cite draft Part
5 or no part where #899 decided Part 9, and undistort names a method that does not
exist). The skill's figure of "roughly 14 specs with Non-goals" is out of date:
74 standing specs carry a `Non-goals` heading now.

## Mechanical findings (all 202 specs)

The corpus is 202 files under `specs/` and `docs/` (README.md and TEMPLATE.md
excluded): 182 standing specs and 20 drafts. Last run it was 175 + 16. As last
run, drafts are left out of openings (check 4), counted separately in coverage
(check 5), and their default rows (check 1) never count as drift.

### 1. Documented defaults vs actual defaults

| Step | Now | 2026-10-06 |
|---|---|---|
| Parameter rows extracted from specs (tables, py signatures, `// default` in fenced Rust, prose, Rust literals) | 852 (16 in drafts) | 914 |
| Code defaults read (`const`, `#[pyo3(signature)]`, `impl Default`, Click `default=`, py defaults, dataclass fields) | 3,437 | 2,801 |
| Rows keyed (spec → owning command or type → parameter) and compared | 528 | 611 |
| Candidates read against the code | 58 | 93 |
| Numeric rows with no owner found, cleared by hand or by a same-named default | 169 | 166 |
| **Confirmed drift found by the scan** | **0** | 0 |

The 58 candidates are example calls that override a default on purpose
(`covered-by-finer.md:233`, `prune-covered-observations.md:241`,
`lazy-kdforest-query.md:624`, …), named constants the scan did not evaluate
(`MIN_MASK_PIXELS`, `DEFAULT_SCHEDULE`, `DEFAULT_MAX_CACHE_BYTES = 256 << 20`,
…), struct literals spelled differently, and CLI `default=None` options that
resolve downstream to the documented value. All 32 command modules' Click
options appear in their own specs, and the 40 `help=` strings that state a
default different from `default=` as text are all `None` resolving to the
stated value.

**The deep reads found one documented default the scan missed.**
`relative-pose.md` gives `min_inliers` "(defaults 12 / 8)" in a table cell's
prose; the rotation estimator's default is 20 (`RayRotationOptions::default`,
`relative_pose.rs:244`, and the binding's `min_inliers=20`). The scan reads a
`| name | value |` cell, not a parenthesised pair of defaults in a description
column. That is a third blind spot for this check, after the two the skill
already names.

> _Status (2026-10-17): **Done** — `relative-pose.md`'s Inputs table now gives `min_inliers` defaults of 12 for the essential estimator and 20 for the rotation estimator, branch `fix-round-07-relative-pose-min-inliers`._

### 2. Prose duplicated between a spec and its code

Method as before: lines of 60 or more characters after normalization, from
every spec and every Rust `//` comment and Python source line. 209 normalized
lines are shared (212 last run). 12 pairs share 5 or more (13); 22 share 3 or
more (24).

| Shared (in fence) | Spec:lines | Code:lines |
|---|---|---|
| 12 (12) | gui/operation-progress.md:63-165 | sfmtool-progress/src/lib.rs:4-545 |
| 9 (9) | gui/camera-intrinsics.md:522-540 | sfm-explorer/src/state.rs:133-162 |
| 8 (7) | core/reconstruction/batch-triangulation-api.md:93-131 | reconstruction/triangulation.rs:79-126 |
| 7 (7) | core/camera/photograph-cache.md:50-89 | camera/photograph_cache.rs:186-400 |
| 7 (7) | core/patch/patch-cloud.md:74-317 | patch/cloud.rs:91-1466 |
| 7 (7) | core/patch/patch-normal-refinement.md:350-450 | normal_refine/params.rs:39-254 |
| 7 (7) | gui/action-log.md:444-535 | action_log/mod.rs:58-404 |
| 7 (7) | gui/background-tasks.md:518-545 | background/mod.rs:60-399 |
| 6 (6) | core/patch/zncc-self-similarity-radius.md:85-145 | patch/self_similarity.rs:47-138 |
| 6 (6) | gui/multi-panel-image-browser.md:817-829 | sfm-explorer/src/state.rs:193-202 |
| 5 (2) | core/bench/editable-track.md:381-2167 | bench/steps.rs:1787-2078 |
| 5 (5) | gui/panel-layout.md:801-817 | sfm-explorer/src/window.rs:513-559 |

Nearly all of it is doc comments copied into a spec's fenced API block.
`patch-normal-refinement` rose from 5 to 7; `mcp-server` left the list. The one
pair that shares prose outside fences is `translation-averaging.md:140, 159,
182` ↔ `translation_averaging.rs:675, 881, 892`, carried below.

> _Status (2026-10-31): the `patch-normal-refinement` row is **Superseded** — the spec and the settings and result types of `normal_refine/params.rs` it copied were deleted with the photometric normal search, branch `retire-photometric-search`._

### 3. Shape of the `specs/core/` specs

| | Now | 2026-10-06 |
|---|---|---|
| `specs/core/` specs | 92 | 85 |
| with a ` ```rust ` block | 63 | 59 |
| no code block at all | 13 (4 are measurement files) | 11 |
| Rust block 50% or more of the way down | 6 | 6 |
| no usage example found | 42 | 46 |

Shortlist for failure 2 (leads, not verdicts): `features/sift` (interface at
62%), `analysis/adjacency-surfel-normals` (69%), `analysis/observation-adjacency-graph`
(70%), `patch/member-coherence-validation` (70%), `geometry/rotation-locked-resection`
(61%), `geometry/affine-factorization` (59%), and with no code block
`geometry/relative-pose` (confirmed by the deep read below),
`geometry/reconstruction-growth`, `geometry/translation-averaging` and
`patch/keypoint-localization-consensus-basis` (confirmed below).

No work-order headings remain. The wider phrase grep found:

| Spec:line | Text | Note |
|---|---|---|
| core/patch/keypoint-localization-consensus-basis.md:3-5, :65, :68 | "Removing the consensus … is proposed in …"; "New fields on `KeypointLocalizeParams`"; "exactly the current behavior" | see the deep read |
| gui/camera-views.md:1274, :1652 | "identical to the current behavior"; "the current behavior is preserved" | change language |
| gui/scene-graph.md:1295 | "the already-planned LRU/async thumbnail loading becomes more pressing" | stale: `architecture.md` says the atlas evicts none |
| gui/edits/resect-image.md:24 | "`seed-candidate-evaluation` (not yet written)" | points at a spec that does not exist |
| cli/reconstruction/motion-command.md:534 | "left as future work — see specs/core/ for the … design notes" | deferral with a vague pointer |
| core/camera/image-warping.md:1209 | "uses a depth-aware resampler (future work)" | deferral |
| core/camera/projection-jacobian.md:184 | "multi-coefficient fisheye models and equirectangular — deferred" | deferral |
| formats/camrig-file-format.md:~467 | "`kerry_park.camrig` must be regenerated in the canonical convention" | done: the file is version 2 |

> _Status (2026-10-17): camrig row **Done** — sentence deleted, branch `fix-round-09-camrig-example-version`._

> _Status (2026-10-31): scene-graph and resect-image rows **Done** — `gui/scene-graph.md` § "Performance" now says in the present tense that each node's thumbnail atlas holds all its thumbnails with no eviction (no LRU or async thumbnail loading exists or is drafted); the pointer to the never-written `seed-candidate-evaluation` spec is removed from `gui/edits/resect-image.md` (no spec or draft under another name describes an offline hold-out self-resection), branch `fix-1031-21-gui-spec-stale-lines`._

Deferral language in standing specs: "not yet" 18 hits in 15 files, "future
work" 2/2, "v2" 8/5, "planned" 4/4, "deferred" 45/13; 33 files in all, most of
the "deferred" and "v2" hits technical (deferred screenshots, the "v2 model" of
homogeneous points).

### 4. Opening paragraphs

181 openings read (standing specs less GLOSSARY.md). 59 have a backticked
identifier or `::` in the first sentence (57 last run), 19 a link (11; five of
them are "This file records the measurements behind [X]", which is fine), none
starts with a formula. All nine openings proposed last run have landed.

Read as a list, three fail and three are borderline. **Land each in its own
PR**: each proposed sentence was written from the spec, so a reviewer who knows
the subject should check it as a claim about the code, one at a time.

| # | Spec | Problem | Proposed first sentence |
|---|---|---|---|
| 1 | core/patch/keypoint-localization-consensus-basis.md:3 | The opening is a pointer to a draft that proposes removing the subject; what the cap is appears at :9. | "The consensus-basis cap limits keypoint localization's congealing to at most `K` of a point's views (default 8) and registers each remaining view once against the template those views build, so that localizing a point with hundreds of views costs work linear in its view count rather than quadratic." |
| 2 | core/geometry/relative-pose.md:5 | Opens with "The setting: …", a fragment describing the problem; never says what the module is for or that nothing calls it. | "Relative pose estimates the rotation and translation direction between two calibrated images from their matched rays, with one robust estimator for general motion (`estimate_essential_rays`) and one for a camera that only rotated (`fit_ray_rotation`), offered on `sfmtool.geometry`; no pipeline in this repository calls it." |
| 3 | core/analysis/adjacency-surfel-normals.md:5 | Starts with how the fit works; needs its link to parse; never says what for. | "Adjacency surfel normals gives each selected point of a reconstruction a surface normal, and a verdict on how well that normal is determined, by fitting a plane through the point to the directions of its neighbours in the observation adjacency graph, without reading any image." |
| 4 | core/patch/cell-plane-normals.md:5 | Uses "piecewise refinement" and "reference member" as common usage; what it does is the fourth sentence. (borderline) | "The cell plane normal kernel estimates the surface normal of a cluster's patch from camera poses and the per-cell displacements that a `--piecewise` `.matches` file stores, without reading any image: it triangulates each of the patch's nine cells from its sightings and fits a plane through them." |
| 5 | cli/reconstruction/xform/refine-normals-command.md:3 | Leads with a bare flag and "surfaces"; does not say it needs an embedded-patches input. (borderline) | "`sfm xform --refine-normals` replaces the normal stored with each finite point of an embedded-patches reconstruction by the normal at which the point's patch looks most alike across the images that observe it, and rewrites the point's patch frame, and by default its RGBA patch texture, to match." |
| 6 | core/patch/reference-view.md:3 | Problem first; says what the rule is in sentence 6. (borderline; the deep read passed it) | none proposed |

> _Status (2026-10-30): row 5 **Superseded** — `refine-normals-command.md` was rewritten for the grown-plane normal, and its opening now says what the option does and that it needs an embedded-patches reconstruction, branch `grown-plane-xform-refine`._

### 5. Coverage both ways

**Specs never cited from `crates/`, `src/`, `tests/` or `scripts/`: 63 of 202**
(57 of 191). 23 are CLI specs (command modules do not link their specs), 16 are
drafts, and 4 are new measurement files that need no citation. The other 20:
`analysis/image-pair-graph`, `analysis/reconstruction-alignment`,
`camera/projection-jacobian`, `features/descriptor-matching`,
`features/flow-based-matching`, `features/kdf-layout-measurements`,
`features/optical-flow`, `geometry/relative-pose`,
`reconstruction/point-correspondence`, `formats/colmap-interop`,
`gui/cross-panel-hover`, `gui/image-animation`, `gui/patch-rendering`,
`gui/user-experience`, `gui/viewer-3d-bench-layer`,
`gui/windows-precision-touchpad`, `research/blender-…`, `workspace/workspace`,
`docs/index`, `docs/tutorials/getting-started`. `randomized-kdtree-forest` and
`reprojection-residuals` are now cited.

**Code surfaces citing no spec:** crates 1 of 11 (`sfmtool-progress`, unchanged);
`sfmtool-core` top-level modules 2 of 13 (`profiling`, `readable`); `sfm-explorer`
top-level modules 14 of 48 (same list as last run); `src/sfmtool` 36 of 51,
including the 10 binding modules from #868, which `python-bindings.md` covers
as a group; `_commands` 28 of 32 (expected).

### 6. Format specs that depend on code names (every run)

Hits outside a heading named *Implementations*. Counts in parentheses are 2026-10-06.

| File | Links | Identifier hits | Verdict |
|---|---|---|---|
| archive-container.md | 0 (0) | 0 (0) | clean |
| archive-io-crate.md | 10 (10) | 87 (82) | crate spec; code names are its subject |
| camrig-file-format.md | 0 | 0 | the `jq` example at :402 shows `"version": 1`, writers emit 2; stale "must be regenerated" sentence; joins the sample |
| colmap-interop.md | 11 (11) | 1 (1) | crate spec; `:60` acquitted |
| kdf-file-format.md | 2 (2) | 0 (0) | clean; background links |
| matches-file-format.md | 5 (8) | 0 (0) | links acquitted; **the current version is stated as 9, the code writes 10**; joins the sample |
| sfmr-file-format.md | 5 (4) | 4 (0) | **4 findings**, all in the version-12 `reference_observations` passage; joins the sample |
| sfmtool-camera-models.md | 2 (2) | 0 (0) | clean |
| sift-file-format.md | 1 (1) | 1 (1) | clean |

Version constants agree with the specs for sfmr (12), sift (1) and kdf (3), and
disagree for matches (10 vs 9) and the camrig example (2 vs 1). Details are in
the sampled sections below.

> _Status (2026-10-17): camrig row **Done** — the example shows version 2 and the "must be regenerated" sentence is deleted, branch `fix-round-09-camrig-example-version`._

## Sampled specs

### specs/core/features/kdf-layout-measurements.md
**Summary:** The version-1 measurement study behind the `.kdf` layout: node and subtree sizes, chunking, origin blocks, and the projected and measured file sizes for a 9.7M-descriptor corpus. The arithmetic re-derives (41 bytes per node, 4,737-row subtrees of 667,227 bytes, 32,940 entries, 3.82-3.85 GB).
**Implementing code:** `crates/sfmtool-kdf-format/src/write.rs` (`subtree_weight`, `pack_tree`), `types.rs` (`KdfWriteOptions::default`), `crates/sfmtool-core/src/features/kdforest/{mod,build}.rs`, `scripts/benchmark_kdf_layouts.py`.
**Inconsistencies:**
  - The reproduction formula at :163 is `33*nodes(n)+132*n <= target`; every number in the file, and `subtree_weight`, uses 41 bytes per node. Change 33 to 41.
  - The built file at :126 has 9,701,948 descriptors; the study counted 9,702,948 (`lazy-kdforest-query-measurements.md:118` also has 9,701,948). So ":138 reproduces the counts above to the byte" is false by 1,000 rows. State the difference and its cause, or drop the claim.
  - :191 says the flat array "costs about 24% more than the blocked corpus when descriptors compress to 76.4%"; 1/0.764 is 31% more. 24% is how much smaller the blocked corpus is.
  - :169 "No source files were modified and no `.kdf` was built" sits under the section reporting a built `.kdf`. Move it to the study section or say "the study itself built no `.kdf`".
  - Minor: :148 "0.223 GB" against the table's 0.2280 GB; :152 "3.33x" against 4.0257/1.212 = 3.32. No commit or PR is cited for the measured build, as TEMPLATE § "Measurement files" asks.

> _Status (2026-10-17): **Partially done** — covers the formula, corpus-count, percentage and Minor-number bullets: the reproduction formula uses 41 bytes per node; the measured section says the build indexed 9,701,948 rows because the script holds out its 1,000-query batch before building, and gives the decoded bytes at that count; the flat array is "about 31% larger" (1 / 0.764 = 1.31); 0.228 GB and 3.32x. Branch `fix-round-08-kdf-layout-numbers`. Still open: the misplaced "No source files were modified" paragraph, the missing commit or PR citation, the "Format tradeoffs" move, and the mannered prose._

> _Status (2026-10-31): **Partially done** — the misplaced paragraph and the citation: the reproduction paragraph moved to the end of the size-budget section, before the measured build, and now says the study (PR #416) read the source files only and built no `.kdf`; the measured section cites PR #421 (squash-merged as `95cb784b`), whose script and version-1 writer produced the numbers, and says the PR does not record which branch commit was run. Branch `fix-1031-15-kdf-layout-measurements-spec`. Still open: the "Format tradeoffs" move and the mannered prose._

**Third copies:** "Format tradeoffs" (:171-203) is design rationale, not measurement. Its grouped-node-column argument repeats `kdf-file-format.md:416-428`. Move the flat-versus-blocked argument into the format spec's departures section and keep only the numbers here.
**Recommendation:** update spec — fix the formula, the corpus count, the percentage, the misplaced paragraph, and cite the PR.
**Unclear / incorrect / suspicious:** Mannered prose: "genuinely attractive on paper" (:188) → "has three advantages"; "blocking wins on the axis that turned out to dominate" (:199) → "blocking needs fewer page reads, and page reads are the larger cost".
Non-goals/deferrals checked: 1; no longer hold: 0

### specs/core/features/descriptor-matching.md
**Summary:** Mutual-best descriptor matching between two posed images along epipolar lines, on a rectified (Y-sweep) or polar path, with an optional geometric filter (orientation, triangulation angle, size ratio). Well shaped: interface before theory, every default (30, 50, 10.0, 15/5/0.8/1.25, the 36-sample median offset, the wrap threshold) matches.
**Implementing code:** `crates/sfmtool-core/src/features/feature_match/{mod,descriptor,sweep,polar,window,gather,geometric_filter}.rs`, `camera/rectification.rs`, `crates/sfmtool-py/src/matching/{image,sweep,descriptor}.rs`, `src/sfmtool/feature_match/{_core,_geometric_filter}.py`, `src/sfmtool/_densify.py`.
**Inconsistencies:**
  - **Suspected code bug: the rectified path's size stage reads rectified pixels through unrectified cameras.** `match_image_pair` (`mod.rs`) passes the rectified keypoints to `mutual_best_match_sweep_geometric` together with a `StereoPairGeometry` built from the original K, R, t. The window filter takes positions from the sorted rectified keypoints (`sweep.rs` `match_one_way_sweep_inner`), and `compute_ray_angle_cosine` / `triangulate_point_dlt` unproject them with the original `K⁻¹` and `R`. Ray angles and depths are wrong, so the size stage can accept or reject the wrong candidates in `sfm densify`. The polar path passes raw positions and is correct. Spec :247-255 describes the stage as using "the two features' unprojected rays", which holds only on the polar path. Fix: carry raw positions through the Y sort, as the polar `GeometricInputs` does, and add a test comparing masks for raw and rectified inputs.
    > _Status (2026-10-17): **Done** — `match_image_pair`'s rectified path now sorts by the rectified keypoints but hands the geometric filter the raw positions, carried through the Y sort (`sweep.rs` `SweepFilterInputs`); a test in `feature_match/tests.rs` checks that every true pair passes the filter at raw positions, that some fail at rectified ones, and that `match_image_pair` keeps all of them; the spec says the rectified positions are only the sort key, branch `fix-round-01-rectified-size-filter`._
  - `match_image_pair`'s output order varies between runs: `window::mutual_matches` iterates a `std` `HashMap` with RandomState and nothing sorts afterwards. Either say "in no particular order" or sort by `index1` so `sfm densify` reproduces. Discuss.
    > _Status (2026-10-17): **Done** — `window::mutual_matches` now sorts the kept pairs by `index1`, so `match_image_pair` and the four mutual sweeps return the same order on every run; unit tests in `feature_match/window/tests.rs`, and the spec and doc comments say the output is sorted by `index1`, branch `fix-round-02-mutual-matches-order`._
  - The keyword names differ between bindings of the same module: `match_image_pair` takes `max_angle_difference`, `min_triangulation_angle`, `geometric_size_ratio_min/max` and applies the filter only when all four are given (`image.rs:79-94`); the `_geometric` sweep bindings take `max_angle_diff`, `min_tri_angle`, `size_ratio_min/max` with defaults. Document or align.
  - The parameter table omits the dataclass field `enable_geometric_filtering` (default `True`), which `_core.py` and `_densify.py` check.
    > _Status (2026-10-31): **Done** — the parameter table has an `enable_geometric_filtering` row (default `True`, Python dataclass only, what `False` skips, and that `sfm densify` builds the dataclass only under `--enable-geometric-filtering`), PR #1093._
  - :88 "its Python binding checks first" holds only for the batch binding's list lengths; mismatched row or column counts panic instead of raising `ValueError`.
    > _Status (2026-10-31): **Done** — the inputs are now validated instead of panicking: `match_image_pair` and `match_image_pairs_batch` in `sfmtool-core` check every length and index they read and return `MatchImagePairError`, and both Python bindings also check that matrices are (3, 3), translations (3,), positions (N, 2), descriptors N rows as wide as the first, and affine shapes (N, 4), raising `ValueError`; the spec describes the checks and errors, with Rust and Python tests for each bad input, PR #1093._
  - The interface never shows how to build a `StereoPairGeometry`, whose constructor orders poses `r1,r2,t1,t2` where `match_image_pair` takes `r1,t1,r2,t2`.
**Third copies:** `sweep.rs` module doc (:7-16) repeats spec :133-136; `compute_angle_offset` doc (12 lines) repeats :213-219. Minor. A stale comment at `polar.rs:164` refers to "the Python implementation's approach", which no longer exists.
> _Status (2026-10-31): **Partially done** — the `Wraparound::plan` comment in `polar.rs` no longer refers to a Python implementation, and now gives the cap as π/4 instead of "a quarter turn" (π/2), PR #1093. The `sweep.rs` and `compute_angle_offset` doc copies are still open._
**Recommendation:** update code (rectified-path filter positions, possibly the output order), then the spec.
**Unclear / incorrect / suspicious:** none beyond the above. Failures 1-5 and 7 pass.
Non-goals/deferrals checked: 3; no longer hold: 0

### specs/core/patch/blur-matched-zncc.md
**Summary:** A ZNCC that blurs the sharper of two tiles to the other's sharpness before correlating, used to score a track's views against the stored bitmap; never used for alignment. Signatures, constants, the pairing rule, `BitmapScorer` and the binding signatures all match.
**Implementing code:** `crates/sfmtool-core/src/patch/blur_matched.rs` (+ `assess.rs`, `blur.rs`, `tiles.rs`, `zncc.rs`), `patch/pair_sharpness.rs`, `patch/stored_bitmap.rs:460-731`, `patch/member_coherence/matrix.rs`, `bench/evaluate.rs` (~:2141-2156), `crates/sfmtool-py/src/patches/blur_matched.rs`.
**Inconsistencies:**
  - :440-445 cites `sharper-patch-bitmap.md` Part 6 for the member-coherence question; it is under Part 5 (only the covariance question is Part 6).
    > _Status (2026-10-31): **Done** — § "Scores against the stored bitmap" now cites the draft's Part 5 for member coherence and Part 6 for the covariance, branch `fix-1031-11-blur-matched-zncc-spec`._
  - Non-goals 3 and 4 (:771-782) explain why `loo_zncc` and the localizer's consensus are not blur-matched, with "still"; #899 decided in Part 9 to remove both, not built yet. Link Part 9.
    > _Status (2026-10-31): **Done** — already fixed: Part 9 is built; #914 removed the leave-one-out sentence from non-goal 3 and #923 rewrote that bullet, and #937 rewrote non-goal 4, so the Non-goals no longer name `loo_zncc` or the localizer's consensus; the only "still" left is normal refinement's weighted mean, which Part 9 leaves unchanged, so no Part 9 link is needed._
  - Work-order residue: § "Cost" :485-488 "When the reference-view rule read blur-matched pairs, …", and a consumer-table row "Reference view … off (removed)" while the opening says "Two consumers read it". That history belongs in reference-view § "Why the agreement is read plain", which already holds it.
  - The parameter table defines `MIN_WINDOWED_SAMPLES` as the floor for "a whole-tile reading"; it is also the floor for `zncc_middle` and each `zncc_grid` cell.
    > _Status (2026-10-31): **Done** — the parameter table now says the floor holds for every reading, the whole tile, the middle square and each ninth of the grid, as `windowed_zncc` applies it, branch `fix-1031-11-blur-matched-zncc-spec`._
  - Mannered prose: "plain ZNCC charges each of them for detail the bitmap carries" (:393) → "scores each of them lower for detail the bitmap has and they lack"; "which hands it to the rule's pick" (:419) → "which makes the rule's pick the reference".
**Third copies:** `pair_sharpness.rs` module doc (~20 lines) re-derives § "Which tile is blurred"; `blur.rs` module doc repeats the AVX2 13% measurement; `blur_matched.rs` :32-35 repeats the alignment non-goal; `MAX_MATCHED_LENGTH` and `GROWTH_PROBE_SIGMAS` docs carry 3-4 lines of rationale each. The code copies should shrink.
**Recommendation:** update spec — fix the Part citation, link Part 9 from the Non-goals, move the removed-consumer history; trim the module docs.
**Unclear / incorrect / suspicious:** About 240 of 787 lines are measurement tables; a `-measurements.md` sibling, as consensus-basis got, would keep the spec to its conclusions.
Non-goals/deferrals checked: 10; no longer hold: 0 (2 need a Part 9 link)

### specs/core/patch/cell-plane-normals.md
**Summary:** Estimates a cluster patch's surface normal from camera poses and the per-cell displacements a `--piecewise` `.matches` file stores, by triangulating the cells and fitting a plane, with a verdict on which axes are determined. Short and accurate: all nine defaults and every formula checked (`2n − 3` residual, the RMS floor, the robust scale, `det_aniso`) match, and the tests it names exist.
**Implementing code:** `crates/sfmtool-core/src/patch/cell_plane_normals.rs`, `crates/sfmtool-py/src/analysis/cell_plane_normals.rs`, `patch/normal_refine/support.rs` (`grid_cell_centres`).
**Inconsistencies:**
  - **Naming rule broken (code).** `CellPlaneParams` is a caller-chosen settings struct with `impl Default`, added in #870 after #862 recorded that new ones use `*Options`. Rename to `CellPlaneOptions`.
  - "Why `patch/`" says the kernel needs `grid_bounds`; it imports `grid_cell_centres`.
  - The Rust block omits `CellPlaneStatus` (six values, `NAMES`), which `cell_status` uses; a Python reader can only find the codes through `cell_status_names`. `NormalDeterminacy::code()` is also unlisted.
  - The panics (`resolution < 3`, mismatched lengths, out-of-range indices) and the binding's `ValueError` are undocumented, and "Why this shape" does not say why the kernel asserts rather than returning `Result`.
  - Opening: see check 4, row 4.
  - Mannered prose: "each ray pulls in pixels of its own image" (:97) → "each ray's distance to the point is measured in pixels of its own image"; "how hard each pulls on the plane" (:107) → "how much weight each has in the plane fit".

> _Status (2026-10-31): **Partially done** — the `grid_bounds`, `CellPlaneStatus` and panics bullets, and the Unclear line's missing import: "Why `patch/`" now names `grid_cell_centres`, the one helper the kernel imports; the Rust block lists `NormalDeterminacy::code()` and `CellPlaneStatus` with `NAMES`, and says how the binding returns both; `cell_plane_normals` no longer panics on its input: it checks it up front and returns a `CellPlaneNormalsError`, taking over the binding's reference-in-its-own-cluster and `patch_size` checks, and the binding raises the error as `ValueError`; "Why this shape" lists the errors and says why an unusable cluster or cell is data rather than an error; the Python example imports `analysis`. PR #1098. Still open: the rename, the opening, the mannered prose and the module doc._

**Third copies:** Module doc (:4-26) restates the weights, verdict and one-axis rule; the binding docstring (~15 lines) the ray and weight construction. Mild; the module doc should shrink.
**Recommendation:** update spec and code (rename).
**Unclear / incorrect / suspicious:** The Python example uses `analysis.` with no import; write `from sfmtool import analysis`. The "Why `<module dir>/`" paragraph is a pattern worth a TEMPLATE note.
Non-goals/deferrals checked: 5; no longer hold: 0

### specs/core/geometry/relative-pose.md
**Summary:** Two robust two-view estimators on unit rays at fixed intrinsics: `estimate_essential_rays` and `fit_ray_rotation`, sharing the focal-vote column scan's sampling and local optimization. The algorithm matches; nothing in the repo calls them except the bindings.
**Implementing code:** `crates/sfmtool-core/src/geometry/relative_pose.rs` (+ `tests.rs`), `crates/sfmtool-py/src/geometry/relative_pose.rs`, `tests/rust_bindings/geometry/test_relative_pose_rust_bindings.py`.
**Inconsistencies:**
  - **Wrong documented default.** The Inputs table says `min_inliers` "(defaults 12 / 8)"; the rotation estimator's default is 20 in both `RayRotationOptions::default` (:244) and the binding (`min_inliers=20`). Change the spec to 12 / 20.
  - `max_angle_rad` has a Rust default of 0.01 but is a required keyword in Python; `samples` (512) is not given.
  - Python takes `side` as `"both"` / `"one"` / `"two"`; the spec only gives the Rust `EpipolarSide` variants.
  - "Both take two equal-length slices": Rust silently truncates to the shorter (:148, :276); only the binding rejects a mismatch, and only the binding normalizes rays and refuses zero or non-finite ones.
  - "identical output on every platform" is stronger than the code's doc (identical for identical input and seed) and is not tested.
  - The Tests section promises a Python test of agreement with Rust; there is none.
  - Failure 2: no Rust signatures and no example call; `RayEssentialOptions`, `RayRotationOptions`, `EpipolarSide` are never named. The rationale is present.
  - Mannered prose: "gets the projection for free" → "the SVD a caller computes to decompose E also yields its projection onto the essential manifold"; "starved consensus" → "a consensus smaller than `min_inliers`".

> _Status (2026-10-17): **Partially done** — the "Wrong documented default" bullet: the Inputs table now gives `min_inliers` defaults of 12 for the essential estimator and 20 for the rotation estimator, matching `RayEssentialOptions::default`, `RayRotationOptions::default` and both bindings; the section's other bullets are still open, branch `fix-round-07-relative-pose-min-inliers`._

> _Status (2026-10-31): **Partially done** — the `max_angle_rad`, `side`, equal-length, platform and Python-test bullets: the Inputs table gives each option's Rust default and its Python form (`max_angle_rad` a required keyword, `samples` 512, `side` as `"both"` / `"one"` / `"two"`), the Inputs text says the Rust functions neither normalize nor check rays and read slices of different lengths to the shorter one while the bindings normalize and reject zero, non-finite and mismatched input, Determinism promises bit identity on one machine only and says why platforms can differ, and the Tests section lists what the Python tests check instead of promising a comparison with Rust. `estimate_essential_rays` panicked when `r1` was longer than `r2`, because the residual loops got the untrimmed slices; both estimators now trim the slices first, and `unequal_lengths_use_the_shorter_prefix` covers both orders. The Inputs text now also names `RayEssentialOptions`, `RayRotationOptions` and `EpipolarSide`. Still open: the rest of Failure 2 (Rust signatures and an example call) and the mannered-prose bullet, branch `fix-1031-08-relative-pose-spec`._

**Third copies:** The core module doc (:4-31, 28 lines) restates the spec's overview nearly word for word and does not link the spec; the binding module doc (:4-13) restates it again. Cut the core doc to the contract plus a link.
**Recommendation:** update spec — 12 / 20, the type names, one call, the Python `side` strings; soften or test the platform claim.
**Unclear / incorrect / suspicious:** Opening: see check 4, row 2. The spec could say these are bindings for experiments, as `focal-vote.md:355` implies.
Non-goals/deferrals checked: 0 (none present); no longer hold: 0

### specs/cli/reconstruction/xform/refine-normals-command.md
**Summary:** `sfm xform --refine-normals` runs photometric patch-normal refinement on an embedded-patches reconstruction and rewrites normals, patch frames and, by default, the bitmaps. All 17 key-table rows match `RefineNormalsTransform.__init__` and the binding; the gate, the finite-point mask and the writer's normal preservation exist as described.
**Implementing code:** `src/sfmtool/xform/_refine_normals.py`, `xform/_arg_parser.py` (`_REFINE_NORMALS_KEYS`), `xform/_images.py`, `_commands/xform.py`, `crates/sfmtool-py/src/patches/refine_normals.rs`, `crates/sfmtool-sfmr-format/src/write.rs`.
**Inconsistencies:**
  - **False claim:** :108-110 says the defaults reuse the binding's, "so the CLI re-specifies nothing and the two layers cannot drift". `RefineNormalsTransform.__init__` restates all 16 forwarded defaults and passes each explicitly; the class docstring repeats the claim. Either forward only the given keys (as #825 did for `--localize-keypoints`) or correct the spec and pin the defaults with a test.
  - "Not surfaced in v1" gives "the non-default window/sampler combinations" as its example, but `window` and `sampler` are exposed. The unexposed knobs are `obliquity_weight_power`, `fronto_prior_weight`, `max_refine_views`, `point_indexes`, `view_indices`, `progress`.
  - The "Write-back semantics" pseudocode transcribes `apply()` and is already stale (no `bitmaps` branch). Delete it.
  - Image loading is said to mirror `RemoveLargeFeaturesFilter`; the code uses the shared `load_workspace_images`.
  - The spec contradicts itself on a point at infinity's normal: :229-231 says `(0, 0, 0)` and then a fixed `normalize(-d)`; :317-318 says the writer leaves zero for infinity points. State it once.
  - Work-order residue: "(decided)", "used to recompute … now preserves" (:313-316), "The transform should print…" (:336, :381), "recommended v1 defaults", "Not surfaced in v1", "v1 does not gate".
  - Mannered prose: "slots in naturally" → "fits the reconstruction-in, reconstruction-out pattern"; "reaching back into the workspace" → "reading files from the workspace"; "cannot fall out of step with" → "always matches"; "deferred until a user hits it" → "not implemented; no user has asked for it".
  - Opening: see check 4, row 5.

> _Status (2026-10-17): **Partially done** — the "False claim" bullet: took the report's second option. The spec's `key=value` section and the `RefineNormalsTransform` class docstring now say the transform restates each forwarded binding default and passes it explicitly (it reads `resolution`, `sampler` and `confidence` itself, and `quality` sets the cache knobs), and the new `test_forwarded_defaults_match_the_binding` compares those defaults with the `PatchCloud.refine_normals` signature, branch `fix-round-10-refine-normals-defaults`. Still open: the section's other bullets._

> _Status (2026-10-30): **Superseded** — the spec was rewritten when `--refine-normals` moved to the grown-plane normal (`sfmtool.bench.refine_normals`, no keys): the key table, the pseudocode, the `RemoveLargeFeaturesFilter` comparison, the infinity-normal contradiction and the v1 residue are gone, branch `grown-plane-xform-refine`._

**Third copies:** `_refine_normals.py` module docstring (~22 lines) and class docstring (~16 lines) both re-derive "Why this fits `xform`". Shrink to contract plus link and drop the false "cannot drift" claim.
**Recommendation:** update spec; discuss whether the transform should stop restating the binding's defaults.

> _Status (2026-10-17): **Partially done** — the class docstring no longer makes the "cannot drift" claim, and the transform keeps restating the defaults (it uses some of the values itself), now pinned by a test, branch `fix-round-10-refine-normals-defaults`. Still open: shrinking both docstrings to the contract plus a link._

> _Status (2026-10-30): **Superseded** — the transform now runs the grown-plane normal and takes no keys, so the restated defaults, their test and the photometric docstrings are gone; both docstrings are short and point at the spec, branch `grown-plane-xform-refine`._

**Unclear / incorrect / suspicious:** none beyond the above.
Non-goals/deferrals checked: 5; no longer hold: 0

### specs/core/geometry/rotation-init.md
**Summary:** Poses the first 8-14 cameras of a weak-parallax capture: homography rotation edges, spanning-tree propagation and chordal averaging for rotations, a linear seed translation from near-field points, rotation-locked resection, and one staged bundle adjustment with far clusters at infinity. Every constant checked agrees.
**Implementing code:** `crates/sfmtool-core/src/geometry/rotation_init.rs` (+ `tests.rs`), `crates/sfmtool-py/src/geometry/rotation_init.rs`, `tests/rust_bindings/geometry/test_rotation_init_rust_bindings.py`. No other caller.
**Inconsistencies:**
  - **Undocumented input assumption.** The kernel builds `SimplePinhole` at `f0` with a centred principal point (~:607-615) and ignores distortion throughout. Inputs says only "a focal `f0`". Fisheye input gives wrong poses with no error. State it.
    > _Status (2026-10-31): **Done** — § Inputs states that every image is a distortion-free `SimplePinhole` at `f0` with a centred principal point and that distorted input gives wrong poses with no error; the Rust doc and the binding docstring say so too, branch `fix-1031-02-rotation-init-spec`._
  - Inputs refers to "the flat cluster-observation arrays (as in `focal_vote`)" without the precondition the Rust doc states (nondecreasing cluster indexes, contiguous runs).
    > _Status (2026-10-31): **Done** — § Inputs lists the arrays and the nondecreasing precondition, which the kernel does not check and the binding does, branch `fix-1031-02-rotation-init-spec`._
  - "up to 3 edges per image" is wrong: `MAX_EDGES_PER_IMAGE` caps the candidates an image proposes; it can gain more as another image's partner.
    > _Status (2026-10-31): **Done** — § 1 says each image proposes 3 candidates, can end with more edges, and a failed candidate is not replaced, branch `fix-1031-02-rotation-init-spec`._
  - The homography inlier gate `H_MAX_ERROR_PX = 3.0` px, which alone decides the far/near partition, is not stated.
    > _Status (2026-10-31): **Done** — § 1 states the 3 px symmetric-transfer gate (`H_MAX_ERROR_PX = 3.0`) and links `estimate_homography`, branch `fix-1031-02-rotation-init-spec`._
  - The `None` cases are incomplete (empty or mismatched arrays, bad `f0`, `max_images < 2`, fewer than 10 near rows on the seed edge).
    > _Status (2026-10-31): **Done** — § Output lists every `None` case, including the degenerate translation direction; the Rust doc and binding docstring list them too, branch `fix-1031-02-rotation-init-spec`._
  - The Rust signature (9 positional arguments) is never shown. `seed`, `min_images`, `max_images` are caller choices that the naming rule puts in a `RotationInitOptions`. Discuss.
    > _Status (2026-10-31): **Partially done** — § Binding shows the Rust signature. **Needs decision** — whether `seed`, `min_images` and `max_images` move into a `RotationInitOptions` struct, a public Rust API change, branch `fix-1031-02-rotation-init-spec`._
  - Mannered prose: "would starve the 25-shared gate" → "would leave too few pairs with 25 shared clusters"; "the LM walks the flat scale gauge downward" → "LM reduces the scale toward a zero baseline"; "Averaging exists to absorb tree drift" → "Averaging reduces the error that accumulates along a chain of edge rotations"; "wander harmlessly" → "change without affecting the cost".
    > _Status (2026-10-31): **Done** — all four phrases reworded along those lines, branch `fix-1031-02-rotation-init-spec`._
**Third copies:** The comment at ~:694-699 re-derives the far-mask rationale in different words from spec §4; the `point_at_infinity` doc (5 lines) and binding docstring (9 lines) repeat it. Shrink the code comment to a line plus a link.
**Recommendation:** update spec (assumptions, gate, `None` cases, edge count, prose); discuss the options struct.
**Unclear / incorrect / suspicious:** The Frame convention says a missing `S` conjugation gives a reconstruction "mirrored about the horizontal axis". `S R S = M R M` with `M = diag(−1,1,1)`, so it would be a left-right mirror, and the canonical rays would more likely fail cheirality. Verify or reword. `estimate_homography` lives in `homography_estimation.rs`, documented only in focal-vote.md.
  > _Status (2026-10-31): **Done** — the Frame convention now says `S R S = M R M` is a left-right mirror of the rotations (yaw and roll reversed, pitch kept) that disagrees with the observations, not a clean mirror of the scene, and that the finishing adjustment can correct small rotations: with the conjugation removed, every `rotation_init` test still passes. § 1 links `estimate_homography` to focal-vote.md § "Homography estimation", branch `fix-1031-02-rotation-init-spec`._
Non-goals/deferrals checked: 3; no longer hold: 0

### specs/cli/image-processing/undistort-command.md
**Summary:** `sfm undistort` warps every image of a `.sfmr` to a best-fit pinhole, transforms `.sift` positions and affine shapes, remaps tracks and writes a new workspace. The options table matches Click exactly; the opening passes.
**Implementing code:** `src/sfmtool/_commands/undistort.py`, `src/sfmtool/_undistort_images.py`, `tests/test_undistort.py`.
**Inconsistencies:**
  - **Code: source `.sift` files come from the workspace's current config.** `get_sift_path_for_image(image_path, feature_tool=…)` (:378) resolves through the current `feature_prefix_dir`, while `get_sift_path_from_recon` exists for "the features a given reconstruction was built from". After a re-extraction with another tool, undistort reads the wrong `.sift` and remaps tracks through unrelated indexes with no error. `source_feature_tool/options` also come from the workspace, as the spec documents. Fix the code, then the sentence.
    > _Status (2026-10-17): **Done** — `undistort_reconstruction_images` reads each source `.sift` through `get_sift_path_from_recon` and takes `source_feature_tool` / `source_feature_options` from the input `.sfmr`'s `workspace.contents`; its `source_workspace_config` parameter is gone, the spec states the lookup, and a test rewrites the workspace config before undistorting, branch `fix-round-03-undistort-sift-lookup`._
  - **Points at infinity are not mentioned.** The first step, `materialize_infinity_for_export` (:254), places `w = 0` points at a finite depth, so the output has none. Add a subsection.
    > _Status (2026-10-31): **Done** — the spec has a § "Points at infinity" naming `materialize_infinity_for_export` and its printed line, branch `fix-1031-07-undistort-command`._
  - Stdout: the paths are printed twice (`_undistort_images.py:605-611` and `undistort.py:119-122`), and the counts line is cameras/points/observations. "a single summary line" contradicts its own two-line example.
    > _Status (2026-10-31): **Done** — the command no longer echoes the paths after `undistort_reconstruction_images` prints them; § "Stdout" shows the closing summary with its cameras/points/observations line, § "Feature-drop summary" describes its two lines, and the CLI test checks each path is printed once, branch `fix-1031-07-undistort-command`._
  - Errors: a missing `.sift`, an unreadable image and a failed `imwrite` also become `ClickException`.
    > _Status (2026-10-10): **Not done** — the finding's `imwrite` now reads `write_image_rgb` / `write_image_rgba`; a failed write still becomes `ClickException`, branch `python-image-writes-via-rust`._
    > _Status (2026-10-31): **Done** — § "Error behavior" says every exception becomes a `ClickException` and lists the missing image or `.sift`, an undecodable image, a failed `write_image_rgb` / `write_image_rgba` and a camera model with no pinhole fit, branch `fix-1031-07-undistort-command`._
  - The thumbnail is embedded in the output `.sfmr` as well as the `.sift`.
    > _Status (2026-10-31): **Done** — § "Thumbnail" says so, branch `fix-1031-07-undistort-command`._
  - "a future revision may call `SfmrReconstruction.recompute_reprojection_errors()`" names a method that does not exist. Delete or move to `specs/drafts/`.
    > _Status (2026-10-31): **Done** — deleted with the paragraph around it, which was wrong: `SfmrReconstruction.from_data` scores every point through the pinhole cameras on the new `.sift` positions, so the section now says the errors describe the output, branch `fix-1031-07-undistort-command`._
  - Failure 4: "produced by `_build_sfmr_data_dict` / `SfmrReconstruction.from_data`" → "computed from the surviving points when the output `.sfmr` is built". Failure 5: "explicitly ruled out as unnecessary" → "Descriptors are not re-extracted".
    > _Status (2026-10-31): **Done** — both sentences rewritten, branch `fix-1031-07-undistort-command`._
**Third copies:** none of concern.
**Recommendation:** update code (feature lookup through the reconstruction, duplicated paths), then the spec.
**Unclear / incorrect / suspicious:** A rig sensor naming a camera no image uses raises `KeyError` (`cam_idx_to_new_idx[int(ci)]`). The pinhole size comes from the decoded image, not the camera's width and height. `progress_callback` is never passed.
    > _Status (2026-10-31): **Partially done** — a rig-sensor camera no image uses now gets a pinhole fitted at the camera's own size instead of a `KeyError`, with a test, and § "Cameras" says so; the pinhole size from the decoded image is what the spec already states; `progress_callback` is still never passed by the command, branch `fix-1031-07-undistort-command`._
Non-goals/deferrals checked: 3; no longer hold: 0 (one names a non-existent method)

### specs/core/patch/fronto-parallel-patch-cache.md
**Summary:** Normal refinement renders one supersampled fronto-parallel base patch per view and scores each candidate normal by an affine resample of it, instead of re-rendering from the source images. The default (`CacheMode::FrontoParallel`, supersample 2) and the named tests match.
**Implementing code:** `crates/sfmtool-core/src/patch/normal_refine/fronto_cache.rs`, `params.rs` (`CacheMode`), `search.rs:121-136`, `crates/sfmtool-py/src/patches/refine_normals.rs:134`, `src/sfmtool/xform/_refine_normals.py:45-56`.
**Inconsistencies:**
  - Keypoint anchoring is missing: `prerender` centres each base on the stored keypoint's ray (`center_offset`, :224-230), and `eval_phi` re-centres by an offset held at the seed normal. § "Idea" says the centre is fixed, and Limitations omits the approximation the code comment admits (:65-69).
  - The perspective corner formula `(x/−z, −y/−z)` is not what the code uses (`(x/z, y/z)`, :141); state the rule that any fixed linear map cancels.
  - Three speed figures: "~2.25×", "~2.3-2.5×", and "~2× faster" in `CacheMode::FrontoParallel`'s doc and the preset comment. Pick one.
  - Residue: the homography cache and `base_margin` are discussed as alternatives; neither exists in the code. Label them as development variants or drop them.
  - `quality` is described as a preset over the pair; its default `"none"` defers to the explicit knobs. The fallback threshold is `min_views.max(2)`.
  - "Surfel" is a non-preferred synonym in GLOSSARY.md.
  - Mannered prose: "Attacking that remainder is the next lever" → "Further speedup has to come from the uncached work"; "the only lever that moves accuracy" → "the only setting that changes accuracy"; "not an e2e mover" → "does not change end-to-end time".
**Third copies:** Module doc (:4-25) restates § "Idea" and § "Geometry"; `corner_norm_pts` doc (:104-112) repeats the ray-path argument word for word. Shrink the module doc.
**Recommendation:** update spec (anchoring, formula, one speed figure, historical variants), and the `CacheMode` and preset comments.
**Unclear / incorrect / suspicious:** The 3,000-point results are not tied to a named script run.
Non-goals/deferrals checked: 2; no longer hold: 0

> _Status (2026-10-31): **Superseded** — the fronto-parallel cache, `CacheMode`, the search and `PatchCloud.refine_normals` were deleted with the photometric normal search, and this spec with them, branch `retire-photometric-search`._

### specs/core/patch/keypoint-localization-consensus-basis.md
**Summary:** Caps keypoint localization's congealing to a basis of at most `K` views (default 8) and registers the rest once against the finished template. Behaviour is accurate: defaults, `select_basis` and the stride, the tail gates, the cache sizing, `is_basis`, the bindings and the CLI flag. The prior audit's shape row (dated A/B tables) is fixed by #881.
**Implementing code:** `crates/sfmtool-core/src/patch/keypoint_localize.rs` (+ `basis.rs`, `tail.rs`, `params.rs:219-221`, `prof.rs`), `crates/sfmtool-py/src/patches/localize_keypoints.rs:168-169`, `src/sfmtool/_commands/embed_patches.py:229`, `src/sfmtool/_embed_patches.py:650-699`, `src/sfmtool/xform/_localize_keypoints.py:85`.
**Inconsistencies:**
  - Opening: see check 4, row 1.
  - Work-order residue: "New fields on `KeypointLocalizeParams` (mirrored as PyO3 kwargs)" (:65), "exactly the current behavior" (:68), "the existing congealing loop … unchanged" (:36-37), "One field is added — `is_basis`" (:149-150), "the uncapped implementation" (:47). Rewrite in the present tense.
  - The Tests section links no files; the tests exist (`keypoint_localize/tests.rs:2256-2667`, `basis/tests.rs`, `tests/patch/test_patch_keypoint_localization.py`, `tests/xform/test_localize_keypoints.py`).
  - Code doc drift: `BasisPick::Strided` doc (`params.rs:53-55`) says "every `ceil(m/K)`-th entry"; the code and spec use the seats left after the track reservation. The spec is right.
  - Mannered prose: "The statistics don't want those views" → "Those views do not improve the consensus either"; "when the cap does not bite" → "when the cap does not reduce the view set".
**Third copies:** `tail.rs:130-140` repeats § "Phase B mechanics"; minor.

> _Status (2026-10-09): **Superseded** — `keypoint-localization-consensus-basis.md` is deleted with the consensus-basis cap, branch `localize-reference` (PR #914)._
**Recommendation:** update spec; retire it with its measurements sibling when Part 9 is built.
**Unclear / incorrect / suspicious:** none.
Non-goals/deferrals checked: 3; no longer hold: 0

### specs/core/patch/cluster-patch-refinement.md
**Summary:** Refines each `.matches` cluster's member warps against its reference by shift, then affine, then optionally piecewise per-cell shifts, with refined-shape gates. Accurate: every default in both parameter tables matches `ClusterRefineParams::default()` and `PiecewiseParams::default()`, the binding and the CLI, and the loop's stop rules match `piecewise.rs::run_loop`. The opening reads well.
**Implementing code:** `crates/sfmtool-core/src/patch/cluster_refine.rs` (+ `params.rs`, `piecewise.rs`, `kernels.rs`, `consistency.rs`), `crates/sfmtool-py/src/matching/cluster.rs`, `src/sfmtool/_cluster_patches.py`, `src/sfmtool/_commands/cluster_patches.py`.
**Inconsistencies:**
  - **Naming rule broken (code).** `PiecewiseParams`, a new caller-chosen settings struct from #870, should be `PiecewiseOptions`; the binding already returns it as `piecewise_options`. With `CellPlaneParams`, GLOSSARY's count of 26 legacy structs is now 28.
  - The public `refine_cluster_patches_borrowed` (:1319, the bench's entry over borrowed pyramids) is missing from the interface and from "Why this shape".
  - The binding dict reports `regate_at_refined_shape` as `refined_shape_gate_is_on()` (`cluster.rs:715`), False whenever the member bar is 0; the binding docstring says so, the spec does not.
  - Open question "Shift only, or shift and scale, per cell" conflicts with the Non-goal that refuses an affine per cell.
  - Work-order residue: "is now reverted" (:885), "the original transcription" (:952), "A first version applied…" (:1038). "What the fleet measurements decide" (:838-890) is a run log including superseded configurations that repeats figures given twice elsewhere; move it to `cluster-patch-refinement-measurements.md`.
  - Failure 4: "The optimizer allocates nothing. The simplex lives in fixed `[f64; 6]` buffers…".
  - Mannered prose: "Doing this early pays twice" (:19) → "Doing this before poses exist has two uses"; "rewarded for walking off the image" (:351) → "a warp whose support leaves the image always scores worse"; "reflect-heavy 6-dim crawl" (:389, :975) → "takes many small reflection steps until the iteration cap"; "both are dead ends at this scale" (:1262) → "neither improved the result at SIFT-patch scale".
**Third copies:** `cluster_refine.rs` module doc (:4-36) re-narrates the pipeline; `params.rs` `intermediate_convergence` and `Default` comments (:220-225) repeat sweep numbers and cite a "Performance" section that does not exist (it is "Measured cost"); the `piecewise` field doc (:197-199) carries a deferral only the code states; `PiecewiseParams::move_shape` re-argues the human review. Shrink all to a link.
**Recommendation:** update code (rename, shrink copies) and spec (add the borrowed entry, move the run log, fix residue and prose).
**Unclear / incorrect / suspicious:** "(CLI `--no-piecewise`)" for an off default; the flag is `--piecewise/--no-piecewise`. A "Why this shape" plus example per sub-API is a pattern worth a TEMPLATE note.
Non-goals/deferrals checked: 12; no longer hold: 0

> _Status (2026-10-31): **Partially done** — the spec's Rust API now lists `refine_cluster_patches_borrowed`, and "Why this shape" says why it takes borrowed pyramids; § "Python bindings" says the dict's `regate_at_refined_shape` reports whether the whole grid's gate ran, so it is False when the member gate's bar is 0; the open question "Shift only, or shift and scale, per cell" is folded into the Non-goal, now "No fit beyond a shift per cell"; the `params.rs` comment cites "Measured cost" instead of "Performance"; and § "Parameters" names the flag `--piecewise/--no-piecewise`. Still open: the `PiecewiseParams` rename, the work-order residue and the run log, the Failure 4 sentence, the mannered prose, the other third copies, and the TEMPLATE note; branch `fix-1031-14-cluster-patch-refinement-spec`._

### specs/core/patch/reference-view.md
**Summary:** The rule that picks which observation's render becomes a point's stored bitmap, its fallbacks and facing limit, and since #857/#871 the stored bitmap itself and the bench pin. Thresholds, rule, fallbacks, bindings and tests all match.
**Implementing code:** `crates/sfmtool-core/src/patch/reference_view.rs` (+ `tile.rs`, `agreement.rs`, `track.rs`), `patch/stored_bitmap.rs`, `patch/normal_refine/obliquity.rs`, `bench/evaluate.rs`, `bench/fit.rs`, `crates/sfmtool-py/src/patches/{view_tile,render_bitmaps}.rs`, `src/sfmtool/_patch_compaction.py`.
**Inconsistencies:**
  - :350-351 "A point with fewer than two views gets no bitmap" holds only for a point at `-1`; a point that stores a reference renders it whatever its view count (`render_patch_cloud_bitmaps`, ~:395-410). The binding docstring is right.
  - :442-446 and Non-goals :650-654 say the localizer's template "is decided separately (sharper-patch-bitmap.md Part 5)"; #899 decided it in Part 9, not built. Re-point, and keep one of the two copies.
  - The API block omits `stored_view(&ReferenceChoice)`, which the bench uses to apply the `without_any` rule to its own tiles (`evaluate.rs:827`); also `ReferenceChoice::standing`, the `name()` wire spellings, `ViewTile::resolution`/`channels`.
  - `render_view_tile` also raises `IndexError` for an image past the set and `ValueError` for a photograph whose size differs from the camera (`view_tile.rs:73-77`).
  - Work-order residue: § "The culls read a sharper bitmap" (:448-452) "shorter than a mean's was". Present tense: "the culls' bars were set on fused-mean bitmaps and have not been measured again on reference renders".
  - Mannered prose: "hands the reference to the rule" (:409) → "makes the rule's pick the reference".

> _Status (2026-10-31): the Part 5 bullet is **Done** — already fixed by PR #914, which aligned every view to the reference render and removed both pointers; the spec's one remaining mention of Part 5 ("Parts 4 and 5" in the opening) cites the design behind storing the view with the most detail, which those parts hold. The fewer-than-two-views and `render_view_tile` bullets are **Done** — the "Where the rule picks no view" bullet of § "The stored bitmap" now limits the fewer-than-two-views case to a point that names no reference and says a whole-cloud render gives a point with a stored reference that observation's tile however many views it has; § "Python bindings" lists the `IndexError` for an `image_index` past the set and the `ValueError`s for an unknown sampler and a photograph whose size differs from the camera's, branch `fix-1031-05-reference-view-spec`._

**Third copies:** `reference_view.rs:39-103` constants carry the tuning evidence (~20 lines); `stored_bitmap.rs` module doc (:11-41) re-derives two spec sections; `stored_reference` doc (10 lines) and `read_track` doc (6 lines) repeat rationale. Shrink the code copies.
**Recommendation:** update spec (the fewer-than-two sentence, Part 9, `stored_view`); trim the code copies.
**Unclear / incorrect / suspicious:** `zncc_grid` in the bitmap scores uses an 8-sample cell floor and the rule's `pair_zncc_grid` 16; neither spec says they differ. The "same row on all 661 tracks" figure predates the `without_any` rule.
Non-goals/deferrals checked: 6; no longer hold: 0 (one stale pointer)

### specs/core/bench/editable-track.md
**Summary:** The bench's editable track: observations whose measurements are stored apart from their verdicts, the cluster and track stages, the bars, and every step. Since #857/#871 the reference row's pin holds the reference, `set_reference` exists, and a track-stage row's `zncc` is its score against the stored bitmap. All eight `Thresholds` defaults, the five `BENCH_*` constants, `EvaluateOptions` and `FitOptions` match.
**Implementing code:** `crates/sfmtool-core/src/bench/{track,steps,evaluate,fit,normal,classify,stage,search,geometry_search,commit}.rs`, `crates/sfmtool-py/src/bench.rs`, `crates/sfm-explorer/src/bench.rs:2604`, `mcp/read.rs:573`.
**Inconsistencies:**
  - The Python section says the steps are module functions "with the same names and the same shape as the Rust ones". The Python names are `pin_verdict`/`unpin_verdict` (`bench.rs:2392-2393`), and `fit_normal`, `finite_difference_normal`, `verdicts_if_unpinned`, `bar_checks`, `render_bitmap_in_place`, `score_bitmap`, `bitmap_target`, `delete_image` have no binding.
  - The evaluate report also carries `turned_in` and `turned_out` (`bench.rs:1584-1585`).
  - The Python example says `evaluate(track, edited, images)` "measures, moves nothing"; its default `render_bitmap=True` renders and repaints unpinned verdicts.
  - Python `reference_observation` returns `None` right after `set_reference` (it is `bitmap.and(reference)`, `bench.rs:599-602`), while the Rust example asserts `held_reference() == Some(3)` at that point.
  - :1593 says `stage_data.reference_observation` never points at an `out` row; after #871 that field is the reference in use. It means `reference_view_observation`.
  - § "Growing and judging" says `bar_checks` returns `None` only without a whole-patch ZNCC; it also does without `loo_zncc` (`steps.rs:2913`), as § "The reference view" says.
  - `EditableTrack::reference_view_pick` (new in #871) and `bitmap_pending()` are not in the API block.
  - **Non-goal no longer holds:** "The pairwise coherence matrix" says evaluation scores each observation against the others' consensus and that reading the matrix in "is proposed". Evaluation now scores against the stored bitmap and computes `member_zncc_matrix_reporting` through `read_track`, storing each row's median as `pair_zncc`. Only showing the full matrix is unbuilt.
  - **Related-specs line no longer holds:** it calls `drafts/sfm-explorer-track-editing.md` "the proposal for the searches"; they are built (`search_descriptors`, `search_geometry`). Non-goals 2 and 3 say "the same draft" without naming one.
  - Work-order residue in § "Parameters": "the bars on `main` before this measurement", "Re-read on a build of this branch", "`measure.py` changed twice during the runs".
  - Mannered prose: "where the correlation would rather be" → "the shift with the highest correlation"; "lands wherever the inconsistency throws it" → "is placed at an arbitrary depth set by the inconsistency"; "what makes a found image worth anything" → "what gives a found image its seed position and shape".

> _Status (2026-10-17): **Partially done** — the first four bullets (the Python section) are fixed: § "Python bindings" now says how the Python steps follow the Rust ones, names `pin_verdict`/`unpin_verdict`, `create_track`/`create_cluster` and `evaluate` as the exceptions, and lists the Rust steps with no binding (`fit_normal` and `finite_difference_normal` have been bound since #965); the evaluate report lists `turned_in` and `turned_out`; the example's comment on `evaluate` says it renders and repaints; and `reference_observation` is not a bug but a different reading from `held_reference` — the spec and its docstring now say it reads `None` after `set_reference` until the next render. Still open: the other bullets of this section, the third copies and the recommendation's other items; branch `fix-round-04-editable-track-python`._

> _Status (2026-10-31): **Partially done** — four more bullets are fixed in the spec. § "The standing always agrees with the verdicts" now says `stage_data.reference_view_observation` (the rule's pick) never points at an `out` row. § "Growing and judging" says `bar_checks` returns `None` at the cluster stage without a whole-patch ZNCC and at the track stage without a blur-matched score against the bitmap or a `seed_shift_px`; the code gates on `seed_shift_px`, not `loo_zncc`. The coherence-matrix non-goal now says an evaluation computes member coherence's matrix through `read_track` and keeps each row's median as `pair_zncc`, and only showing the matrix is proposed. The related-specs line calls the draft the proposal for the pull-in and for showing the matrix, with the searches built, and the two non-goals name the draft. Still open: the API-block bullet, the work-order residue, the mannered prose, the third copies and the recommendation's other items; branch `fix-1031-06-editable-track-spec`._

**Third copies:** `BENCH_MIN_ZNCC` doc (`track.rs:991-1014`, 24 lines) and `BENCH_MIN_ZNCC_MIDDLE` (:1017-1024) restate the measurement; `TrackPayload::reference` (:710-748, 38 lines) copies § "The stored bitmap's reference". The spec's own 800-line Rust API block repeats its prose sections. Shrink the code copies to value plus link, and the API block to signatures plus one-line purposes.
**Recommendation:** update spec (Python section, :1593, the two stale entries, residue); shrink the code copies.
**Unclear / incorrect / suspicious:** § "Where the figures come from" says the scripts and data "are not in the repository, so the figures cannot be re-derived"; the default `BENCH_MIN_ZNCC = 0.65` rests on that. Consider checking in `measure.py` and `analyze2.py`. The "Which reading each reader takes" table works better than prose for a value with two readings; worth a TEMPLATE note.
Non-goals/deferrals checked: 10; no longer hold: 2

### specs/python-bindings.md
**Summary:** The index of the compiled extension's bindings: one public Python module per extension submodule, no flat root, and the lazy loading. Accurate after #866/#868: all 173 bound names are covered, every listed name exists, every link resolves.
**Implementing code:** `crates/sfmtool-py/src/lib.rs`, `helpers.rs` (`install_submodule`), `src/sfmtool/__init__.py`, the 10 binding modules, `src/sfmtool/sift/__init__.py`, `tests/test_module_layout.py`, `tests/rust_bindings/reconstruction/test_reconstruction_patches_registration.py`, `tests/test_python_bindings_index.py`.
**Inconsistencies:**
  - :33-36 says each public module re-exports with `from ._sfmtool.<name> import *` and takes `__all__`; `sfmtool/sift/__init__.py:9` does neither, and the registration test skips the `__all__` check for `sift` (:119) while :85 says it checks each module. Name the exception.
  - :57-59 "`from sfmtool import *` binds every public module" — it binds `_LAZY_SUBPACKAGES` (the 11 binding modules plus `rig`), not `align`, `analyze`, `camera`, …
  - `_LAZY_NAMES`, which puts about 25 Python names on the root, is never named; :61 leaves a reader thinking the root holds only the three extension names. (AGENTS.md has the converse slip: it says the root binds its public names through `_LAZY_NAMES`, but modules go through `_LAZY_SUBPACKAGES`.)
  - Gap: the array-layout rule (read every NumPy buffer through `to_contiguous!` or `.as_standard_layout()`, because Fortran-order and negative-stride arrays are otherwise transposed or panic) lives only in a 45-line doc comment in `lib.rs:43-87`. Add an "Arrays at the boundary" section and cut the comment to contract plus link.
  - :68 "the bench's steps would read as something else beside the rest" → "would sit at the root under names that do not say they belong to the bench".
**Third copies:** `lib.rs` module doc (:10-25) and the `__init__.py` docstring repeat the layout; acceptable, `lib.rs` could shrink to two lines.
**Recommendation:** update spec.
**Unclear / incorrect / suspicious:** none.
Non-goals/deferrals checked: 6; no longer hold: 0

> _Status (2026-11-01): **Partially done** — covers the first three bullets. `sfmtool.sift` now follows the rule instead of being an exception: its `__init__.py` takes `__all__` from `sfmtool._sfmtool.sift`, the registration test compares its `__all__` like the others, and § "Where Python code reads a binding" says only that its re-export is in a package's `__init__.py` beside the Python SIFT modules. The section also says `from sfmtool import *` binds the root's `__all__` (the three extension names, the `_LAZY_NAMES` names and the `_LAZY_SUBPACKAGES` modules, checked by running it: 44 names, no `align`, `analyze` or `camera`), and describes `_LAZY_NAMES`; AGENTS.md now says the binding modules and `rig` go through `_LAZY_SUBPACKAGES`. PR #1097. Still open: the array-layout gap, the :68 wording and the `lib.rs` module doc._

### specs/formats/sfmr-file-format.md
**Summary:** The `.sfmr` reconstruction archive, versions 1-12. #857 added version 12's `tracks/reference_observations` column; its entry name, `int32` type, shape, slot in the tracks hash, presence tied to `has_uv_frames`, range rule, `-1` fill for older files and migration statement all match `crates/sfmtool-sfmr-format` (`types.rs:672-684`, `read.rs:723-740`, `write.rs:939, 1109, 1386-1401`, `verify.rs:704-772`).
**Implementing code:** `crates/sfmtool-sfmr-format/src/{types,read,write,verify}.rs`.
**Inconsistencies:**
  - **The Verification Process (~:1814-1840) omits the version-12 check** that `verify.rs:756-771` performs (column length, and each value `-1` or inside its point's run of observations). An independent verifier built from the list would accept a corrupt column.
    > _Status (2026-10-31): **Done** — the Verification Process now lists the version-12 check: with `has_uv_frames` set, `reference_observations` must be `point_count` rows of 4 bytes, each `-1` or an index into its point's observations, branch `fix-1031-03-sfmr-verify-v12`._
  - **Failure 6, library API in the format proper (:1619-1636):** "In memory, `SfmrReconstruction::to_sfmr_data` writes the column … (`PointSet::saved_reference_observations`) … `validate_point_columns` … Python `clone_with_changes` keeps the old references". Move to *Implementations*.
  - **Failure 6, command names as definitions (:1578-1619):** `sfm xform --add-patch-bitmaps`, `sfm web-export`, `--convert-infinity`, `--minimal`, `--drop-patch-bitmaps`, `--localize-keypoints`, `--refine-keypoints`, `--refine-normals`, `sfm embed-patches`, "SfM Explorer's display bitmaps". The writer-role rules stay; the command lists move.
  - **Failure 6, operations as meaning (:1403-1406):** "the reference-view rule picked no view or reached its pick only through its last fallback, `without_any`; see … `ReferenceRender::stored_reference`". Restate as "a writer that cannot name a single observation the bitmap was rendered from writes `-1`".
  - **Failure 6, deferred meaning (:1575-1578):** the row is "that observation's `R×R` render … re-anchored on that observation's keypoint … with the sampler the sampler rule picks"; "re-anchored" and "the sampler rule" are defined only behind the link. Restate them, or weaken to "an `R×R` render of the patch as seen from that observation".
  - :325 and :2199 say "version 9, 10 or 11" may carry `lineage`; the reader skips it at version 12 too. Write "version 9 or later".
    > _Status (2026-10-31): **Done** — both passages (now :337 and :2906) say the key may appear in a file of version 9 or later; the retirement paragraph says a file without it is valid at its own version, branch `fix-1031-03-sfmr-verify-v12`._
  - Mannered prose: "It rode/rides inside `metadata.json`" (:2174, :2223) → "is stored in"; "without a handedness bridge" (:98) → "conversion"; "stops at the I/O boundary" (:1183) → "a caller never sees the file's own numbering".
**Third copies:** `SFMR_FORMAT_VERSION` doc (`types.rs:612-671`, 60 lines) re-tells versions 5-12 (carried); `SfmrData::reference_observations` doc (`types.rs:1033-1058`, 26 lines) restates the Format and "Keeping it true" bullets. Shrink both.
**Recommendation:** update spec — add the verifier bullet, move the § 9 implementation passages to *Implementations*, restate the bitmap terms.
**Unclear / incorrect / suspicious:** A version-12 file with `has_uv_frames: false` that still contains a `reference_observations` entry passes reader and verifier; the stray entry is outside every hash. `.matches` reports stray entries since #824; decide whether `.sfmr` should.
Non-goals/deferrals checked: 3; no longer hold: 0

### specs/formats/matches-file-format.md
**Summary:** The `.matches` archive: pairwise matches with optional two-view geometries, or clusters with an optional `cluster_patches/` section, versions 1-10. #870 added the per-cell entries (`member_cell_*` with the `member_cell_status_names` legend), version-10 member statuses, new `refine_options` keys and migrations for 8-10; legend orders, per-version gating, the "four entries together" rule, the "no readings on a not-`kept` member" rule and the hash order all match `crates/sfmtool-matches-format` (`types.rs`, `cells.rs`, `verify.rs:927-1130`).
**Implementing code:** `crates/sfmtool-matches-format/src/{types,cells,verify}.rs`; writers of `refine_options` in `src/sfmtool/_cluster_patches.py:227-241` and `crates/sfm-explorer/src/cluster_patches.rs:1030`.
**Inconsistencies:**
  - **The current version is stated as 9.** The example (:178) shows `"version": 9`; the field description (:227) says "`1` through `9`; writers emit `9`". The code writes 10 (`MATCHES_FORMAT_VERSION = 10`, `types.rs:102`), as the spec's own :1490 says. A writer following :227 emits version 9 with the 9-name legend, which readers refuse ("no version 9 writer wrote those names"). The same kind of slip as last run's 6-vs-7 finding.
  - "verifier" means two things: :452 and :1480 use it for a writer that performs geometric verification; elsewhere it is the integrity checker. Write "a writer that performs geometric verification".
  - Verification step 6 (~:1365) does not list the per-cell checks or the rejection of legend names newer than the file; both are stated elsewhere. One clause.
  - `refine_options.piecewise`: the SfM Explorer writer omits it (`cluster_patches.rs:1030-1038`); the spec gives no meaning to its absence. Say a missing key means the per-cell refinement did not run.
  - Mannered prose: "the bridge between features and reconstruction" (:18) → "the step between"; "makes working with the pipeline less fun" (:24) → "slows down repeated solves"; "rides inside" (:944, :1553) → "is stored in"; "the rows the cascade measured" (:535) → "the rows the refiner measured".

> _Status (2026-10-17): **Partially done** — the first bullet (current version stated as 9): the top-level metadata example now shows `"version": 10` and the `version` field description says "`1` through `10`; writers emit `10`", matching `MATCHES_FORMAT_VERSION`; the section's other bullets are still open, branch `fix-round-06-matches-version`._

> _Status (2026-10-31): **Partially done** — the "verifier", Verification step 6 and `refine_options.piecewise` bullets: the three places that used "verifier" for geometric verification (the cluster-file paragraph, the `--derive-pairs` writer line and "Writing a verified .matches file") now say "a writer that performs geometric verification"; step 6 now lists the per-cell checks and the rejection of legend names newer than the file's version, as `verify.rs` checks them; the `refine_options` description now says a missing `piecewise` means the per-cell refinement did not run (the SfM Explorer writer omits the key and never runs it, and files from before the refinement lack it). The mannered-prose bullet and the third copy are still open, branch `fix-1031-23-matches-format-spec`._

**Third copies:** `MATCHES_FORMAT_VERSION` doc (`types.rs:68-101`, 34 lines) re-tells versions 7-10; shrink to contract plus link, as it already does below 7.
**Recommendation:** update spec — the version line first.
**Unclear / incorrect / suspicious:** none beyond the above. All five outbound links are background.
Non-goals/deferrals checked: 1; no longer hold: 0

### specs/formats/camrig-file-format.md
**Summary:** The `.camrig` archive: a rig of sensors with shared cameras, fixed relative poses and image patterns, from a single camera up to ~100,000 spherical tiles; versions 1-2. Read in this run for its check-6 hit. The opening states what the format is for. Links and identifiers in the format proper: none outside background.
**Implementing code:** `crates/sfmtool-camrig-format/src/{types,read,write,verify,pattern}.rs` (`CAMRIG_FORMAT_VERSION = 2`, `types.rs:66`; `upgrade_sensor_poses_from_v1`, :226), `crates/sfmtool-core/src/spherical/tile_rig/camrig.rs`.
**Inconsistencies:**
  - The § "Using CLI tools" `jq` output (:402) shows `"version": 1`; writers always write 2 (:463 and `types.rs:66`). Show 2.
  - Work-order residue (~:467): "As a consequence, `test-data/images/kerry_park/kerry_park.camrig` must be regenerated in the canonical convention." It has been (`sfm inspect` reports version 2), and a repository test file has no place in the format proper. Delete the sentence.
  - Deferred meaning, minor: the `spherical_tiles` row lists `measured_max_nn_angle` and `measured_max_coverage_angle` and points at `specs/core/spherical/spherical-tiles-rig.md` (a plain path, not a link, so check 6 does not see it); the format spec never says what the two angles are. One clause each ("the largest angle from a tile centre to its nearest neighbour", "the largest angle from any direction to its nearest tile centre") would let another tool write them. "the sphere-point relaxer" is an implementation term.

> _Status (2026-10-17): **Partially done** — bullets 1 and 2: the `jq` example now shows `"version": 2`, with its `rig_attributes` keys in the alphabetical order the writer emits; the "must be regenerated" sentence is deleted (`sfm inspect` confirms `kerry_park.camrig` is version 2), branch `fix-round-09-camrig-example-version`. Still open: the third bullet (say what the two `measured_*` angles are)._

> _Status (2026-10-31): **Done** — bullet 3: § "Rig type and attributes" now defines `measured_max_nn_angle` (largest angle from a tile direction to its nearest other tile direction; diagnostic) and `measured_max_coverage_angle` (largest angle from any direction to its nearest tile direction, estimated from 50 000 random probes; sets the half-FOV and the resampling cell radius), both in radians; "the sphere-point relaxer" is now `evenly_distributed_sphere_points`, linked to `spherical-tiles-rig.md` § "Tile layout", and the row's plain path is a link, branch `fix-1031-24-camrig-format-spec`._

**Third copies:** none found.
**Recommendation:** update spec — three small edits.
**Unclear / incorrect / suspicious:** none.
Non-goals/deferrals checked: 0 (none present); no longer hold: 0

## Code without specs

Built at `13c1578e` against the 2026-10-06 table.

| Surface | User-facing? | Spec |
|---|---|---|
| 29 `COMMANDS` rows (`ws`, `camrig` groups and every flat command) | yes | one `cli/<category>/*-command.md` each; `sfm version` none (acceptable) |
| `sfm xform`: 32 options | yes | all in `cli/reconstruction/xform/*.md` |
| `_commands/_range_options.py` | no | **now cited** by to-colmap-bin-command.md |
| `_commands/_sfmr_path.py` (new, #905) | error text | **none** |
| 11 crates | mixed | all covered |
| `sfmtool-archive-io` `parse_hash` (new, #895) | no | formats/archive-io-crate.md:106-118 |
| core `analysis`, `bench`, `camera`, `features`, `geometry`, `patch`, `reconstruction`, `spatial`, `spherical`, `web_export` | no / via CLI | `specs/core/<module>/`, web-export-command.md |
| core `patch::cell_plane_normals`, `cluster_refine::piecewise`, `blur_matched`, `pair_sharpness`, `stored_bitmap` (new) | via bindings | cell-plane-normals, cluster-patch-refinement, blur-matched-zncc, reference-view |
| core `numeric` (median, `quantile*` new in #897, `splitmix64`) | no | **none** |
| core `profiling` (`SFMTOOL_PROFILE`) | env var | **none**; 10 specs mention the variable |
| core `readable` | message text | **none** |
| 11 Python binding modules (#868) | Python API | python-bindings.md |
| 15 Python subpackages | via CLI | covered |
| private py modules | no | 19 of 26 named in some spec; not `_densify`, `_undistort_images`, `_histogram_utils`, `_filenames`, `_cli_utils`, `_image_load`, `_cli_group` |
| formats and config files (15) | yes | formats/*, workspace/* |
| GUI subsystems (42) | yes | all cited by some `specs/gui/*` |
| MCP `get/set_viewer_3d_display` (#810) | yes (agents) | mcp-server.md, viewport-hud.md |
| WGSL prelude `shaders/common.wgsl` (#900) | no | scene-graph.md; **missing from gui/architecture.md's shader tree**, as is `bench_track.wgsl` |

About 147 of 157 rows are covered; 4 of the 10 uncovered are small utilities
that need none.

### crates/sfmtool-core/src/numeric.rs
**What it does:** The crate's one median, its one quantile (`quantile`, `quantile_in_place`, `quantile_of_sorted`, added in #897) and `splitmix64`.
**Why it matters:** internal but load-bearing: thresholds across geometry, analysis, patch and features depend on it, and specs such as `affine-factorization.md:71` and `absolute-pose.md:280` state quantile rules without naming it.
**Recommendation:** add a paragraph to `specs/core/README.md` naming `numeric` as the home of median and quantile and its NaN policy; have the specs that state a quantile rule link to it.

### crates/sfmtool-core/src/profiling.rs (`SFMTOOL_PROFILE`)
**What it does:** The shared phase timers and report rows that the per-algorithm `prof` modules use, switched on by `SFMTOOL_PROFILE`.
**Why it matters:** small utility, but a user-visible switch that 10 specs mention and none defines.
**Recommendation:** define `SFMTOOL_PROFILE` once in `specs/core/README.md` (values, output, which modules report) and link to it.

### src/sfmtool/_commands/_sfmr_path.py
**What it does:** `check_sfmr_path` gives every command that takes a reconstruction the same usage error.
**Why it matters:** small utility with a uniform user-facing message.
**Recommendation:** add a note to `specs/cli/README.md` beside the `_range_options` description.

### WGSL prelude `shaders/common.wgsl` (gui/architecture.md shader tree)
**What it does:** `shaders/common.wgsl` declares `ReconUniforms`, `PICK_TAG_*` and `INF_DEPTH`, and six pipelines prepend it.
**Why it matters:** internal but load-bearing: a new shader that redeclares or omits it breaks picking, and the architecture spec's file tree is where a contributor looks first.
**Recommendation:** add `common.wgsl` and `bench_track.wgsl` to the tree, with a one-line note linking `scene-graph.md:771`.

### src/sfmtool/_densify.py, src/sfmtool/_undistort_images.py (carried)
**What it does:** The pipelines behind `sfm densify` (765 lines) and `sfm undistort` (613 lines).
**Why it matters:** user-facing through the CLI; the command specs describe behaviour without linking the implementing module.
**Recommendation:** link the modules from densify-command.md and undistort-command.md.

## Carried forward from the 2026-10-06 audit

Each item was re-measured at `13c1578e`; line numbers are current.

### Still open

- **randomized-kdtree-forest third copies** (partially done: every kdforest source now cites the spec, #835). In `crates/sfmtool-core/src/features/kdforest/`: `search.rs:14-24` (11 lines, repeats spec :338-349), `mod.rs:169-185` (`build` doc, 10 lines of progress-unit rationale), `build.rs:30-34` and `:38-44` (`REPORTS_PER_BUILD`, `BuildProgress`, the same rationale; spec :145-156), `search.rs:34-41` (`PREFETCH_AHEAD`, spec :427-437). `calibrate.rs:25-30` carries an argument the spec lacks and can stay. Cut the four to contract plus a § link.
- **zncc-self-similarity-radius third copies** (partially done: the `params.rs` figures went in #840). Still open: `crates/sfmtool-core/src/bench/track.rs:1040-1048` (9 lines, `BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`, repeats spec :412 and :429 and still says "is turned out" where the spec now says "rejects"); `patch/self_similarity/ellipse.rs:264-276` (13 lines, `LENGTH_SLACK`, re-derives spec :76 nearly word for word); `patch/self_similarity/overlap.rs:259-268` (10 lines, `SAT_ROUNDING_ULPS`, repeats spec :403). The old :432 paragraph is now spec **:431**, the Track View paragraph, one line of 337 words; with :429 it restates `gui/track-view.md` (~:1091, :1132ff), `gui/mcp-server.md` (~:3905-3915) and `core/bench/editable-track.md` (:1440, :3250). Shrink the three code docs and replace spec :429-431 with a pointer to those specs.

  > _Status (2026-10-31): **Done** — the `BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`, `LENGTH_SLACK` and `SAT_ROUNDING_ULPS` docs are cut to what the constant is plus a pointer to the spec, so the "is turned out" wording is gone; the two reasons only the code gave (why a crossing's error grows on a shallow surface, why the tables' `Σ v` and `Σ v²` carry a residue of order `ε S`) moved into the spec first. In the spec's § "Where it runs", the two paragraphs on the bench's fields and bar and on Track View, `get_bench_track` and the Python dicts are replaced by one paragraph of links to `editable-track.md`, `track-view.md` and `mcp-server.md`, branch `fix-1031-18-zncc-radius-third-copies`._
- **pose-verification `inlier_fraction_of` duplicated.** Three copies of one count: `crates/sfmtool-core/src/geometry/pose_verification.rs:174-195` (used at :612), `geometry/reconstruction_growth.rs:211-228` (the same 18-line function and doc), and the public `geometry/reprojection.rs:97-110` `inlier_fraction`. Make one `pub(crate)` helper beside `reprojection::inlier_fraction`; the reprojection-residuals item about moving the other helpers onto it closes with it.
- **pose-verification third copies.** `crates/sfmtool-py/src/geometry/pose_verification.rs:127-138` (12 lines narrating `verify_poses`); `:256-263` (8 lines) and core `pose_verification.rs:494-502` (9 lines), two copies of § Repair (spec :197-221); `features/cluster_match/covisibility/displacement.rs:41-52` against spec :110-122, the orientation contract (cut the spec's copy). New since: the core module doc :8-11 keeps "The ruler" and "holds the current poses against it", which #888 removed from the spec; the repair doc :497 says `nearest` where the code uses `nearest_registered`; the binding module doc :5 names the internal `sfmtool._sfmtool.geometry` path.
  > _Status (2026-10-31): **Done** — the core module doc no longer says "The ruler" or "holds the current poses against it"; the core `repair_poses` doc and the binding `repair_poses` docstring no longer copy § Repair (the core doc points at it; the binding keeps a short summary without the `nearest` name); the binding `verify_poses` docstring is cut to one line per screen; the `DisplacementNeighborhood` orientation contract is cut to the rule plus a pointer at the spec's § Substrate, which keeps the full text (the finding said to cut the spec's copy instead; the code keeps the rule because six other doc comments cite "the type's orientation contract", and the rule is part of the interface the spec describes); the binding module doc names `sfmtool.geometry`; `t = −R·c`, the bit-for-bit pass-through of unrepaired rows and the `NaN` inlier arrays moved into § Repair. Branch `fix-1031-17-pose-verification-third-copies`._
- **refit-camera-intrinsics third copies.** `crates/sfmtool-core/src/camera/refit_intrinsics.rs`: `SMOOTHING` :66-69, `MIN_SLOPE` :75-86, `refit_spline` :672-691 (20 lines), `fit_spline` :1076-1092 (now says SVD, so no longer drifted). The LDP/NNLS derivation is in both `constrained_lsq.rs:4-16, 32-42` and spec :403-414; keep the code copy. `crates/sfm-explorer/src/image_detail/intrinsics/axes.rs:651-660` re-implements core's `spline_domain_deg` (`refit_intrinsics.rs:769`), and the two now differ on a non-finite or non-positive domain end.

  > _Status (2026-10-31): **Partially done** — the viewer's two copies of `spline_domain_deg` (`image_detail/intrinsics/axes.rs` and a fourth in `intrinsics_detail/projection_plot.rs`) are gone and both call core's. The viewer's handling was the right one, since the kernels evaluate a spline whose domain end is not positive and finite as its base model, so core's `spline_domain_deg` now returns `None` for such an end as well (spec interface line updated, test added); the plots draw the same as before. `MIN_SLOPE` and `refit_spline` docs are cut to what they do and a pointer to the spec sections that already hold the rest. `SMOOTHING` is no longer in this file: #952 moved it beside `smoothing_weight` in `camera/distortion/bspline.rs`. Branch `fix-1031-16-refit-intrinsics-third-copies`. Still open: the LDP/NNLS derivation in the spec's § "Implementation notes" ("The constrained solve reuses the SVD"), which repeats `constrained_lsq.rs`, and the `fit_spline` doc._
- **absolute-pose Kabsch rank rationale in three places:** spec :215-224, `KABSCH_RANK_EPS` doc `absolute_pose.rs:35-39`, inline comment :274-278. Cut the inline comment to a pointer.
- **absolute-pose "bearing" against GLOSSARY** (discussion-grade; the `resect_images` half is done, #836): GLOSSARY :121, :262-270 use *bearing* for a point at infinity; `absolute-pose.md` uses it 20 times for an observed ray.
- **translation-averaging third copy:** `conjugate_gradient` doc `translation_averaging.rs:670-675` repeats spec :136-140; `orientation_reading` doc :877-893 (17 lines) repeats :158-186. Cut both.
- **flow-based-matching Python third copies:** `src/sfmtool/feature_match/_flow_matching.py:4-17` (14-line module docstring, with history residue "the previous approach of composing full-resolution flow fields"), :174-177; `optical-flow.md:288-289` repeats `flow-based-matching.md:206-210`.
- **flow-command colour legend in four places** (low): spec :97-104, `_commands/flow.py:115-117`, `visualization/_flow_display.py:180-185, 508-510`.
- **cluster-covisibility third copies:** `features/cluster_match/covisibility.rs:145-217` (63 doc lines on `SeedImageGroup` and neighbours, ~45 last run); `MAX_DENSE_IMAGES` :27-32 still says a sparse backend "is the intended remedy", a deferral #884 removed from the spec; `seed_image_groups` :571 "drop the rest unpaid"; `covisibility/prof.rs:13-17`. Still unclear whether `seed_image_groups` / `next_seed_image_group` need to exist (no caller outside the module, its tests and the binding).

  > _Status (2026-10-31): **Partially done** — the `MAX_DENSE_IMAGES` doc drops the sparse-backend remedy and the "~4–5 k-image window" and points at the spec's "Complexity and bounds"; `seed_image_groups` says the groups not taken are never computed and points at "Seed-group algorithm" for the guarantees; the `SeedImageGroup` doc comments are cut from 57 lines to 18, a line or two per field plus a pointer at "Rust API"; the `prof.rs` paragraph on why the build times itself is a pointer at "Complexity and bounds", which gains the one fact found only there (each phase is timed once per build, so the timers cost nothing measurable). **Needs decision**: whether `seed_image_groups` / `next_seed_image_group` stay, given that the spec says the seed-group iterator has no consumer outside its tests and its binding, or are removed with the `sfmtool.matching` binding. Branch `fix-1031-19-cluster-covisibility-third-copies`._
- **bundle-adjustment third copies:** `geometry/bundle_adjust.rs:1757-1776` (`Lens::new` comment, 20 lines), `retriangulate_round` doc :1138-1184 (47 lines), `reconstruction/bundle_adjust.rs:46-76` (`BundleAdjustOptions::releases`, 31 lines). "rung" survives 5 times in `bundle_adjust.rs` and 25 times in its tests after #885 dropped it from the spec; :1147 "the bootstrap's post-BA refill rule" is a leftover.

  > _Status (2026-10-31): **Done** — the `Lens::new` comments, the `retriangulate_round` doc and the `BundleAdjustOptions::releases` doc are cut to the interface and a pointer at the spec section that holds each fact (every fact they carried was already in `bundle-adjustment.md`, `triangulation-rules.md` or `reconstruction/bundle-adjust.md`); the "bootstrap's post-BA refill rule" reference now points at the spec's step 1; "rung" for a distortion release is "release" in `bundle_adjust.rs`, its tests (comments, messages and four test names), `camera/distortion/tests.rs` and `specs/gui/edits/bundle-adjust.md`. "Rung" for a step of the damping ladder stays, as the spec uses it, branch `fix-1031-20-bundle-adjustment-third-copies`._
- **mcp-server third copies:** the `get_bench_track` description in `crates/sfm-explorer/src/mcp/tools/catalog/read.rs:283-462` is ~180 lines (137 last run); cut it to fields and units and move the detail to `bench.md`. The doc blocks in `mcp/bench.rs`, `display.rs`, `view.rs`, `render.rs`, `layout.rs`, `mcp/edit.rs` are unchanged.
- **mcp-server shape** (partially done: the `Command` transcription went in #831). Still: failure 7 "closes the loop" :117, "the representation the rays earned" :3845, "Screenshots defer honestly" :4323, "is being rude" :5510; failure 5 :4315 "Waking is already solved. `UserEvent` grows one variant", :4816 "where before only a success reached them"; failure 4 § "The Rust seam" (:4474) transcribes `Deferred` (:4547) and `serve` (:4628).
- **track-view module-doc third copies:** `crates/sfm-explorer/src/track_view/` carries 250 lines of `//!` (~220 last run): `body/tile.rs` 53, `body/table.rs` 44, `body/mod.rs` 37, `mod.rs` 28, `body/crop.rs` 24, `body/patch.rs` 21. Seven files do not name the spec: `body/{crop,patch,reference,surface_plot,table,tile}.rs`, `header_buttons.rs`.
- **nearby-tracks binding docstring:** `crates/sfmtool-py/src/bench/nearby_tracks.rs:140-148` narrates the source order, the 20 px stop and the far-field trigger, and the file cites no spec.
- **sfmr version-history doc:** `SFMR_FORMAT_VERSION` doc `crates/sfmtool-sfmr-format/src/types.rs:612-671`, 60 lines through version 12; `SFMR_CANONICAL_CONVENTION_VERSION` doc :693-704. Do what #779 did for matches.
- **sfmtool-progress cites no spec** and is the top duplicate pair (check 2): link `gui/operation-progress.md` from `lib.rs` and cut the shared blocks.
- **kdf-constellation-query `image_feature_ids` cost** described but not measured (`kdf-constellation-query.md:458-468`).
- **docs/index.md screenshot never re-checked:** `docs/index.md:26`; `docs/images/sfm-explorer-with-seoul-bull-tiny-images.jpg` last changed 2026-04-04.

### Needs decision

- **MCP `set_view` and a held Move Camera lock:** code and spec (`mcp-server.md:1476-1486`) agree that a leaving form ends the lock even when it is then refused. Keeping the lock would mean splitting every form into check and apply.

  > _Status (2026-10-17): **Declined** — decided by the maintainer on 2026-10-17: the current behaviour stays, in which a leaving form ends a held Move Camera lock even when the form is then refused. Code and `mcp-server.md` already agree on it, and keeping the lock would mean splitting every form into a check and an apply, which is not worth it; no code change, PR #986._
- **`frame` identifiers for patch geometry** (#891): the spec says *placement*; `crates/sfm-explorer/src/track_view/body/patch.rs:164` `render_frame` and `crates/sfm-explorer/src/bench/geometry.rs:440` `anchored_frame` still say `frame`.

  > _Status (2026-10-18): **Done** — decided by the maintainer on 2026-10-17, revised 2026-10-18: the word is *patch frame* / `patch_frame`, never bare `frame` and not *placement*. `render_frame` is now `render_patch_frame` and `anchored_frame` is `anchored_patch_frame`; with them `frame_color_image` → `patch_frame_color_image`, `AppState::rebuilt_with_frame` → `rebuilt_with_patch_frame`, `lacks_frame` → `lacks_patch_frame`, `with_frame_of` → `with_patch_frame_of`, the 3D bench `Figure::frame` field → `square`, every `frame: &OrientedPatch` parameter and local in the viewer and core's bench → `patch_frame`, core's seven `NoFrame` variants → `NoPatchFrame`, and the viewer's `PatchFrame` (centre and unit axes, no extent) → `PatchCoordinateSystem`, and the Set to Origin log sentence and hint say "coordinate system" instead of "frame". The tree's other *placement* uses for a patch's geometry say patch frame as well, which renames the wire's `placement` block of `get_point` and `get_bench_track` to `patch_frame` and Python's `EditableTrack.placement` to `patch_frame`. `GLOSSARY.md` replaces its *placement* row with *patch frame*, and `bench.md`, `track-view.md`, `multi-panel-image-browser.md`, `editable-track.md` and `mcp-server.md` follow. PR #986._
- **Patch view-set keyword** (c15214ab, #865): tracked in `reports/2026-10-08-hygiene-audit.md:164-183`; not repeated here.
  > _Status (2026-10-17): **Done** — decided by the maintainer on 2026-10-17: `track_views` and `patch_views` (with `patch_keypoints`), recorded in `specs/GLOSSARY.md` § "Per-patch view lists" and applied in core, the bindings and the specs; see the status line on the hygiene-audit finding, branch `decided-1017-04-patch-views`._

### Since fixed (dropped)

Every Inconsistencies bullet of the 16 sampled specs (#814-#836, #838, #839,
#842-#845, #872-#876, #878); all shape and opening items (#846-#854, #877,
#881-#891); the localize-keypoints and reprojection-residuals third copies
(#825, #833); the matches version-history doc (#845); flow-based-matching's
"Future Directions" (#881); pose-verification's untested scaling line and its
failures 4 and 7 (#879, #888); the 2026-09-26 carry-forwards for the
lazy-kdforest measurements (#813) and the TEMPLATE determinism section (#812);
the check-1 adjacent drift rows (#841).

### Specs read by earlier audits (2026-09-05, 2026-09-26, 2026-10-06)

Each spec those audits read is kept as a heading so the `audit-specs` sampler
still counts it as audited. `specs/core/bench/editable-track.md`,
`specs/formats/matches-file-format.md` and `specs/formats/sfmr-file-format.md`
were re-read in this run and appear under **Sampled specs**. The 2026-09-05
audit also read `gui/point-track-detail.md`, since replaced by
`gui/track-view.md`; `core/features/cluster-selection.md` was read at its old
path `formats/cluster-selection.md`.

### docs/index.md
### specs/cli/colmap-interop/from-colmap-bin-command.md
### specs/cli/colmap-interop/to-colmap-bin-command.md
### specs/cli/image-feature/match-command.md
### specs/cli/image-feature/sift-command.md
### specs/cli/image-processing/flow-command.md
### specs/cli/reconstruction/inspect-command.md
### specs/cli/reconstruction/motion-command.md
### specs/cli/reconstruction/xform/localize-keypoints-command.md
### specs/core/analysis/cluster-census.md
### specs/core/bench/nearby-tracks.md
### specs/core/camera/epipolar-curves.md
### specs/core/camera/ray-grid-projection.md
### specs/core/camera/refit-camera-intrinsics.md
### specs/core/camera/sfmtool-pinhole-kernels.md
### specs/core/features/cluster-covisibility.md
### specs/core/features/cluster-selection.md
### specs/core/features/covisibility-selection.md
### specs/core/features/flow-based-matching.md
### specs/core/features/gpu-optical-flow.md
### specs/core/features/kdf-constellation-query.md
### specs/core/features/lazy-kdforest-query.md
### specs/core/features/optical-flow.md
### specs/core/features/randomized-kdtree-forest.md
### specs/core/features/sift.md
### specs/core/features/track-cluster-matching.md
### specs/core/geometry/absolute-pose.md
### specs/core/geometry/affine-factorization.md
### specs/core/geometry/baseline-direction.md
### specs/core/geometry/bundle-adjustment.md
### specs/core/geometry/estimate-intrinsics.md
### specs/core/geometry/focal-vote.md
### specs/core/geometry/pose-verification.md
### specs/core/geometry/reprojection-residuals.md
### specs/core/geometry/rotation-locked-resection.md
### specs/core/patch/candidate-track-spawning.md
### specs/core/patch/zncc-self-similarity-radius.md
### specs/formats/archive-container.md
### specs/formats/kdf-file-format.md
### specs/formats/sfmtool-camera-models.md
### specs/formats/sift-file-format.md
### specs/gui/camera-intrinsics.md
### specs/gui/mcp-server.md
### specs/gui/panel-layout.md
### specs/gui/track-view.md
### specs/gui/viewport-navigation.md

## Top priorities

1. **Code bugs and behaviour a reader would get wrong.** Fix each in the code
   (or the stated default) and add a test.
   - **Descriptor matching, rectified path:** the geometric filter's size stage
     unprojects rectified keypoints with the original K and R
     (`feature_match/mod.rs` → `sweep.rs`), so `sfm densify` keeps or drops the
     wrong candidates. Its output order also varies between runs (`HashMap` in
     `window::mutual_matches`). (descriptor-matching)
   - **`sfm undistort` reads source `.sift` files through the workspace's current
     config** (`_undistort_images.py:378`), not the reconstruction's own, so a
     re-extracted workspace gives wrong track remapping with no error. (undistort-command)
   - **`.matches` spec says the current version is 9** (`:178`, `:227`); writers
     emit 10. A writer following the spec produces files readers refuse.
     (matches-file-format)
   - **`relative-pose.md` documents `fit_ray_rotation`'s `min_inliers` as 8;**
     the code's default is 20 in Rust and Python. The check-1 scan missed it
     because the value sits in prose inside a table cell. (relative-pose)

     > _Status (2026-10-17): **Done** — the Inputs table now gives 20 for `fit_ray_rotation`'s `min_inliers`, branch `fix-round-07-relative-pose-min-inliers`; only this item of the list is covered._

   - **`rotation-init` assumes a distortion-free pinhole with a centred
     principal point**, which no doc states; fisheye input fails without an
     error. (rotation-init)
2. **Spec statements that would make a caller act wrongly:**
   - `refine-normals-command.md:108-110` says the CLI "re-specifies nothing"
     and "cannot drift"; the transform restates all 16 defaults.

     > _Status (2026-10-17): **Done** — the spec and class docstring now say the defaults are restated, and a test pins them to the binding's signature, branch `fix-round-10-refine-normals-defaults`._

   - `editable-track.md`'s Python section says the bindings mirror the Rust
     names and shapes; two names differ, eight steps are unbound, and
     `reference_observation` is `None` after `set_reference`. Two of its
     entries (the coherence-matrix Non-goal and the "proposal for the searches"
     line) no longer hold.
   - `reference-view.md:350` says a point with fewer than two views gets no
     bitmap; that holds only for a point at `-1`.
   - The `.sfmr` Verification Process omits the version-12 column check
     (`verify.rs:756-771`).
   - `kdf-layout-measurements.md:163` gives 33 bytes per node where every
     figure uses 41, and its "to the byte" reproduction used a corpus 1,000
     rows smaller.

     > _Status (2026-10-17): **Done** — both fixed in the spec, branch `fix-round-08-kdf-layout-numbers`._

3. **The `*Options` naming rule, broken a day after it was recorded:**
   `PiecewiseParams` and `CellPlaneParams` (#870) are new caller-chosen
   settings structs. Rename them to `PiecewiseOptions` and `CellPlaneOptions`
   and correct GLOSSARY's count.
4. **Format specs:** move the in-memory API and command lists out of
   `.sfmr` § `tracks/reference_observations` into *Implementations*, and
   restate "re-anchored" and "the sampler rule" in format terms; fix the camrig
   example's version and delete its "must be regenerated" sentence.

   > _Status (2026-10-17): camrig part **Done** — branch `fix-round-09-camrig-example-version`._

5. **Openings, one spec per PR:** `keypoint-localization-consensus-basis.md`,
   `relative-pose.md` and `adjacency-surfel-normals.md` fail, and
   `cell-plane-normals.md` and `refine-normals-command.md` are borderline
   (check 4). Each proposed sentence is a claim about the code, and a reviewer
   checks it properly only when it is the only one in the PR. The carried
   third copies (kd-forest, self-similarity radius, pose verification) remain
   the largest open documentation debt.

   > _Status (2026-10-09): **Partially done** — `keypoint-localization-consensus-basis.md` is deleted with the consensus-basis cap, branch `localize-reference` (PR #914); the other openings are open._
