# Hygiene audit, 2026-09-23

Read-only whole-tree survey at `25410760`, using `skills/audit-hygiene/SKILL.md` and the vocabulary in `specs/GLOSSARY.md`. This snapshot supersedes the 2026-09-12 report, measured at `6111797c`: 132 commits and 637 changed paths made its line numbers and several proposed fixes stale. Still-open findings below were checked against current source. Resolved findings remain in Git history; the three inherited design topics are preserved at the end.

## Repeatable measurements

- 163 Python package files, 37,004 lines; 160 Python test files, 48,639 lines. Rust has 654 `.rs` files and 332,598 physical lines. File-name-based separation gives 183 Rust test files / 132,981 lines and 471 other Rust files / 199,617 lines; inline tests remain in the latter number. `specs/` has 179 Markdown files, including 148 indexed area specs (CLI 34, core 70, formats 8, GUI 33, workspace 3).
- Longest files are `mcp/tests.rs` (10,373), core `bench/tests.rs` (5,159), core `geometry/bundle_adjust/tests.rs` (4,464), and `mcp/tools.rs` (3,630). The Python source leaders are `colmap/io.py` (886), `sift/file.py` (873), and `_embed_patches.py` (833). Size by itself was not treated as a finding.
- A scan counting comment-looking lines across all Rust files, including tests, measures 26.7% for `sfmtool-progress`, 22.3% for `sfmtool-py`, 15.8% for `sfm-explorer`, 15.0% for `sfmtool-core`, and 10.9% for `sfmtool-kdf-format`. The longest contiguous documentation blocks remain 125 lines in `crates/sfmtool-py/src/patches/member_coherence.rs` and 122 in `localize_keypoints.rs`; they are parameter reference prose. A rough per-item binding scan found 201 of 679 doc/function pairs above a 0.5 ratio, led by three-line accessors with 12–16 lines of Python-visible API reference. The ratio alone does not identify repeated rationale.
- The normalized long-line duplicate scan's top production pair is `crates/sfm-explorer/examples/winit_directmanipulation.rs` versus `winit_wgpu_directmanipulation.rs` (98 distinct lines). The leading Python package pair is `_global_sfm.py` versus `_incremental_sfm.py` (46), now localized to their setup. Same-name family scanning found eight `prof.rs` modules; five patch variants share 21–26 distinct code lines of at least 30 characters pairwise. Shared wgpu pipeline descriptor syntax continues to account for the renderer pairs, and is not proposed for abstraction.
- License headers: 348/348 Python files under `src/`, `tests/`, and `scripts/`, and 654/654 Rust files under `crates/`. All 484 `specs/*.md` references in code resolve. Each of the 148 area specs is indexed by its area or subarea README. The old broken links are resolved; the remaining enforcement work is listed below.
- Glossary spot checks: zero `surfel` occurrences in core bench source and zero in the viewer bench directory, versus 38 valid renderer uses. The optional-column `drop`/`remove` distinction remains in xform; no `remove_thumbnail` name was found. `specs/` still has 177 em dashes, which the glossary describes as a gradual prose migration, not a majority vote. `frame` and `offset` in bench source are also used for their legitimate camera and sighting meanings; raw counts are not violations.
- Convention tallies: `sfmtool-core` declares 23 `*Options` and 36 `*Params` types overall; `geometry/` is 12/1 and `camera/` 1/0. In Python, `feature_match/`, `xform/`, `visualization/`, and `strips/` use underscore-private module names throughout (10/10, 27/27, 8/8, 5/5); `align/`, `analyze/`, `camera/`, `camrig/`, `colmap/`, `merge/`, `rig/`, and `sift/` use none. This split has no glossary ruling. The PyO3 surface still has both `_py` and `_rs` exported suffixes and four names for one patch view-set argument. A CLI `help=` scan across 32 modules found two implementation-word hits, both legitimate mentions of COLMAP/GLOMAP as external tools.

## Priority recommendations

**Split the MCP tool catalog and tests along their existing modules**
> _Status (2026-09-23): Done — catalog definitions and all 230 headless MCP tests are grouped by concern, with the central catalog fixtures preserved; commit `bb7130e0`._

- Location: `crates/sfm-explorer/src/mcp/tools.rs` (3,630 lines, catalog and schema helpers at 57–2417) and `crates/sfm-explorer/src/mcp/tests.rs` (10,373 lines, 230 `#[test]` cases).
- Problem: The tool catalog now has 76 entries (15 read, 60 write, one save), up from 40 in September. The test file grew by 4,946 lines while covering transport, frame, layout, view, bench, screenshot, edits, and background behavior in one place. The production module tree already separates those concerns.
- Proposed fix: Split catalog definitions and tests by the existing MCP modules, leaving the name/classification fixture in a central catalog test and sharing only common test fixtures. Preserve the exact catalog/schema/command tests added since September.
- Effort: medium. Risk: medium, because advertised wire names and schemas must remain identical.

**Derive nested MCP argument validation from its schema**
> _Status (2026-09-23): Done — the three nested parsers take accepted names from their catalog schemas, with catalog regression coverage and unchanged error wording._
- Location: `crates/sfm-explorer/src/mcp/tools.rs:2561,2686,3006` and neighboring nested `Args` parsing.
- Problem: Catalog-wide command names, classifications, and top-level unknown arguments are now checked against schemas, resolving much of the old six-edit finding. Nested object parsers still hand-list accepted keys in `reject_unknown`, beside their schema declarations. Adding one nested field can therefore fork the advertised and accepted sets.
- Proposed fix: Pass the nested schema's accepted-key set to the local parser, and extend the catalog fixture to exercise nested objects in both directions.
- Effort: medium. Risk: low, with existing compatibility tests.

**Extract the still-monolithic format readers and writers**
- Location: `crates/sfmtool-sfmr-format/src/read.rs` (750 lines, `read_sfmr` 66–687), `crates/sfmtool-sfmr-format/src/write.rs` (1,371 lines, `write_sfmr_into` 282–881), and `crates/sfmtool-matches-format/src/write.rs` (1,072 lines, `write_matches_into` 54–478).
- Problem: Each driver still owns several wire sections, unlike the per-section `verify_sfmr` helpers and the 87-line KDF writer driver completed after the last audit. The earlier recommendation's verify and KDF portions are resolved; these three remain.
- Proposed fix: Use per-section helpers while keeping `SectionDigests` folding in explicit wire order in the driver. Keep round-trip and hash-stability fixtures as the guard.
- Effort: medium. Risk: medium, because section ordering and hash bytes are format contracts.

**Make xform argument dispatch a table over the shared parsers**
- Location: `src/sfmtool/xform/_arg_parser.py` (772 lines), `parse_transform_args` at 347–772.
- Problem: Six `key=value` cases now use `_parse_kv_params` and 20 value-taking branches now use `_take_arg`, completing the two September subfindings. A 426-line conditional dispatch still repeats option selection and construction, so adding a transform continues to touch a long branch ladder.
- Proposed fix: Use a table of option spelling, value-taking rule, and constructor. Preserve option-specific errors and ordering tests.
- Effort: medium. Risk: medium, because argument order and diagnostics are CLI behavior.

## Rust: viewer and core

**Consolidate history cursor steps and node lookup**
- Location: `crates/sfm-explorer/src/state/edits.rs` (1,843 lines; `undo`, `redo`, and `jump_to_version` begin at 1446, 1491, 1543).
- Problem: The three operations still repeat the progress phases and Action Log tail. Repeated `busy_refusal` plus scene-node lookup remains in the same file; refusal wording still varies at the MCP boundary. The version transition sentence itself was shared after the previous report.
- Proposed fix: One private cursor-step helper and a shared node lookup/refusal constant. Assert the progress breakdown and wire text at all three call sites.
- Effort: medium. Risk: medium, because phase nesting is observed on the wire.

**Align one MCP tool name with its entity**
- Location: `crates/sfm-explorer/src/mcp/tools.rs:269`, `crates/sfm-explorer/src/mcp/edit.rs:58`.
- Problem: `get_history` remains the lone tool name in its family without the entity, while the wire otherwise names the object acted on.
- Proposed fix: Name it `get_reconstruction_history` with a compatibility alias if clients already call the old name; update the catalog fixture and spec together.
- Effort: low. Risk: medium, due to the public tool name.

**Build the DirectManipulation examples in CI**
- Location: three `crates/sfm-explorer/examples/*directmanipulation.rs` programs, selected only by the nondefault `directmanipulation` feature in `crates/sfm-explorer/Cargo.toml:20–33`.
- Problem: Their 1,088-line September baseline has not acquired a CI build path, and the two winit variants still share 98 normalized long code lines. The viewport spec still cites them.
- Proposed fix: Add a Windows `cargo check -p sfm-explorer --features directmanipulation --examples` job step, then decide whether the variants should remain separate reference programs.
- Effort: low. Risk: low.

**Document the `Tab::wire_name` exceptions**
- Location: `crates/sfm-explorer/src/layout.rs:203–214`.
- Problem: Its doc says names derive from panel titles, but the `IntrinsicsDetail` arm uses `camera_intrinsics`; the older `Viewer3D` exception also remains. An agent deriving `panel_name` from that sentence gets a wrong wire value.
- Proposed fix: State the exact exceptions in the doc and keep the layout strings stable.
- Effort: low. Risk: low.

**Set a visibility rule for private viewer modules**
- Location: `crates/sfm-explorer/src/document.rs` (38 bare `pub`, zero `pub(crate)`) and `crates/sfm-explorer/src/state/edits.rs` (19 bare `pub`, 17 `pub(crate)`).
- Problem: Both modules are private in `lib.rs`, while nearby newer subsystems consistently use `pub(crate)` for crate-visible items. The mixed spelling obscures the intended reach.
- Proposed fix: Write the convention in `AGENTS.md`, then converge these two modules, retaining private items where possible.
- Effort: low. Risk: low.

**Remove the runtime shadow of a const generic in bundle adjustment**
- Location: `crates/sfmtool-core/src/geometry/bundle_adjust.rs` (2,254 lines), `solve_lm` around 1376.
- Problem: `CAM_COLS == BSPLINE_CAM_COLS` still re-derives a compile-time choice as a runtime boolean inside the solver.
- Proposed fix: Select the model-specific path through const-generic helpers or an explicit compile-time trait; preserve numerical fixtures for both modes.
- Effort: medium. Risk: medium.

**Separate policy orchestration from focal voting and reconstruction growth**
- Location: `crates/sfmtool-core/src/geometry/focal_vote.rs` (1,478 lines, `focal_vote_impl` 835–1233) and `crates/sfmtool-core/src/geometry/reconstruction_growth.rs` (1,102 lines, `grow_reconstruction` 488–1099, nested `run_grow_ba` at 592).
- Problem: Both entry points still combine policy choices with their staged numerical pipelines. The old growth finding is unchanged in shape despite moved lines.
- Proposed fix: Extract named policy stages and the nested BA runner, keeping orchestration in the public entry points.
- Effort: medium. Risk: medium, due to numerical and progress behavior.

**Move image containers out of camera remapping**
- Location: `crates/sfmtool-core/src/camera/remap.rs` (1,181 lines; `ImageU8` at 52, `ImageF32WithGrad` at 824).
- Problem: Widely used image containers still live under a remapping module, so their path implies a narrower purpose than they serve.
- Proposed fix: Move the containers to a camera image module and re-export during migration.
- Effort: medium. Risk: low.

**Separate distortion projection and share Newton inversion mechanics**
- Location: `crates/sfmtool-core/src/camera/distortion.rs` (1,233 lines), `crates/sfmtool-core/src/camera/distortion/kernels/thin_prism.rs:144`, and `crates/sfmtool-core/src/camera/distortion/kernels/rad_tan.rs:182`.
- Problem: The second distortion impl block is projection, while neighboring thin-prism and radial-tangential inverse solvers still carry parallel Newton loops.
- Proposed fix: Extract projection to its proper module and factor the common Newton step without merging model-specific residuals.
- Effort: medium. Risk: medium, because convergence behavior is sensitive.

**Recheck SIMD scope in column scanning**
- Location: `crates/sfmtool-core/src/geometry/focal_vote/column_scan.rs` (1,964 lines).
- Problem: The old size acquittal is still about one concern, but the file has grown around substantial `unsafe` SIMD paths. This is a review boundary rather than a line-count-only split.
- Proposed fix: Keep public scanning here and place architecture-specific kernels in a private sibling with scalar equivalence tests.
- Effort: medium. Risk: medium.

**Extract the GPU level runner's stages**
- Location: `crates/sfmtool-core/src/features/optical_flow/gpu/mod.rs` (741 lines), `run_gpu_levels_prebuilt` from 254.
- Problem: One method still consumes most of the module and interleaves level orchestration with GPU resource handling.
- Proposed fix: Name preparation, dispatch, and readback stages as private helpers.
- Effort: medium. Risk: medium.

**Share the profiling scaffold and matching common path**
- Location: eight `crates/sfmtool-core/**/prof.rs` files; `crates/sfmtool-core/src/features/feature_match/polar.rs` (712) and `crates/sfmtool-core/src/features/feature_match/sweep.rs` (518).
- Problem: Eight sibling profilers still repeat timing scaffolding, with five patch pairs sharing 21–26 long code lines. Polar and sweep still define parallel `GeometricInputs` and differ mainly in ordering policy.
- Proposed fix: Centralize profiler counters and run logic, then pass an explicit ordering strategy to one shared matcher core.
- Effort: medium. Risk: medium, because benchmark fields and match order are observable.

**Break up the numbered cluster census and edited reconstruction internals**
- Location: `crates/sfmtool-core/src/analysis/cluster_census.rs` (737 lines, main function 425–734) and `crates/sfmtool-core/src/reconstruction/edited.rs` (1,893 lines, `RowMap` at 1396, `ScanRows` at 1423).
- Problem: Census still encodes six phases in one 310-line function. The edited reconstruction file still mixes several mutation concerns with reusable row mapping.
- Proposed fix: Extract census phase helpers and move row-map/scan-row machinery to a focused private module.
- Effort: medium. Risk: medium.

**Record the module-file and parameter-bag naming rules**
- Location: `crates/sfmtool-core/src/patch/` and `crates/sfmtool-core/src/{geometry,camera}`.
- Problem: `patch/` still mixes adjacent `foo.rs` modules with `foo/mod.rs` at the same depth. `geometry/` has 12 `*Options` to one `*Params`, while other algorithm areas favor `*Params`. The glossary is silent, and the latter split may convey estimator option versus algorithm parameter.
- Proposed fix: Decide the boundaries on semantic grounds and document them in `AGENTS.md` or the area index before renaming public types; converge only the cases that violate the chosen rule.
- Effort: low to decide, medium to rename. Risk: medium for public types.

## Rust: formats and Python bindings

**Rename the KDF options for the layer that owns them**
- Location: `crates/sfmtool-kdf-format/src/types.rs:178`, `KdfFile::open`.
- Problem: `LazyKdForestOptions` still names the algorithm in the crate above the format layer. The earlier crate-name and missing-public-doc findings were completed; this type name is the remaining mismatch.
- Proposed fix: Choose a format-facing open-options name and migrate callers with an alias where needed.
- Effort: low. Risk: medium, because the type is public.

**Unify patch binding view-set keywords**
- Location: `crates/sfmtool-py/src/patches/{localize_keypoints,refine_keypoints,refine_normals,member_coherence,select_views}.rs`.
- Problem: The shared prologue was extracted, but the same per-patch image-index input still appears as `view_sets`, `view_indices`, `member_views`, and `candidate_views`; `matching/image.rs` also retains `camera_indices` beside `camera_indexes` elsewhere.
- Proposed fix: Settle one Python-facing word, guided by glossary meaning rather than raw majority, and stage keyword aliases before removing old names.
- Effort: medium. Risk: medium, due to keyword callers.

**Remove language suffixes from exported Python function names**
- Location: `crates/sfmtool-py/src/analysis/core.rs` and `crates/sfmtool-py/src/matching/{image,sweep}.rs`.
- Problem: Functions registered by `wrap_pyfunction!` still expose `_py` and `_rs` in Python, though those suffixes only disambiguate Rust implementation symbols. The old audit counted 16 top-level names across both suffixes.
- Proposed fix: Give wrappers unsuffixed `#[pyo3(name = "...")]` names and preserve deprecated aliases for a release; update registration tests and Python callers.
- Effort: medium. Risk: medium, because the API is public.

**Split `clone_with_changes` by data family**
- Location: `crates/sfmtool-py/src/reconstruction/clone.rs` (915 lines), function 76–738 (663 lines), 27 string-keyed arms.
- Problem: A long extract/validate/assign match and fixed postprocessing tail still carry point, image, and track changes together. It grew 24 lines since the old report.
- Proposed fix: Separate point, image, and track field applicators and retain one finalization sequence with explicit dependency order.
- Effort: medium. Risk: medium, because a changed clear/rebuild order can silently alter reconstructions.

**Share the CSR argument resolver and reference prose**
- Location: `crates/sfmtool-py/src/geometry/{focal_vote,estimate_intrinsics}.rs` and `crates/sfmtool-py/src/analysis/cluster_radii.rs`.
- Problem: `radii_source` at `cluster_radii.rs:47` still duplicates `vote_source` at `focal_vote.rs:82`, as do numeric array coercers. The three binding docs share 18–20 distinct `///` lines pairwise, so one validation contract is transcribed three times.
- Proposed fix: One typed CSR/source resolver, one array coercer, and a shared included argument reference block.
- Effort: medium. Risk: low.

## Python and test layout

**Share only the duplicated solver setup**
- Location: `src/sfmtool/_global_sfm.py` (127 lines) and `src/sfmtool/_incremental_sfm.py` (292 lines).
- Problem: The old report's “88% same runner” description is obsolete after an incremental save helper was added. The first roughly 100 setup lines still duplicate source and database preparation; incremental also seeds twice (46 and 96) and uses `os.makedirs` at 91 where global uses `Path.mkdir` at 86.
- Proposed fix: Extract the common setup, then keep distinct global and incremental solve/save paths.
- Effort: low. Risk: low.

**Split SIFT file I/O from extraction and drawing**
- Location: `src/sfmtool/sift/file.py` (873 lines).
- Problem: Its hash helpers (61–110), file I/O (229–597), extraction (598–791), and drawing (799–873) remain four concerns behind a file-oriented name.
- Proposed fix: Retain file I/O here and move extraction/drawing to named siblings, preserving imports temporarily.
- Effort: medium. Risk: low.

**Split epipolar display by mode**
- Location: `src/sfmtool/visualization/_epipolar_display.py` (619 lines), `draw_epipolar_visualization` 111–619 (509 lines).
- Problem: One renderer still handles the two-by-three mode matrix with 16 parameters, unlike the decomposed flow display sibling.
- Proposed fix: Separate image preparation and mode-specific drawing behind the existing public call.
- Effort: medium. Risk: medium, because output pixels are user-visible.

**Decide the private-module naming convention**
- Location: Python subpackages listed in the convention tally above.
- Problem: Whole subpackages consistently disagree on underscore-private file names, so a new sibling has no predictable spelling. The glossary offers no ruling.
- Proposed fix: Decide whether the prefix marks import privacy or subpackage style, document the rule, and rename only after the rule is clear.
- Effort: low to decide, medium to migrate. Risk: low.

**Group binding tests by domain**
- Location: `tests/rust_bindings/`, now 60 direct modules and 19,714 lines.
- Problem: The directory is still flat and groups by the language boundary, while registration tests already divide the surface into analysis, geometry, matching, reconstruction, SIFT, spatial, spherical, and flow namespaces.
- Proposed fix: Create domain subdirectories using those registration modules as indexes.
- Effort: medium. Risk: low.

**Recheck shared test fixtures after growth**
- Location: `tests/conftest.py`, now 1,117 lines versus 841 in September.
- Problem: It crossed the old report's 1,000-line recheck mark. It has 12 fixtures; recent growth concentrates in Kerry Park setup (899–1117) and pose reconstruction helpers (404–647). The longest helpers are 124 and 113 lines. Fixtures consumed across test subpackages still belong at the common root.
- Proposed fix: Move reusable builders, not pytest fixture registration, to a `tests/_fixtures.py` helper if the next fixture repeats these blocks; keep cross-package fixtures visible in root `conftest.py`.
- Effort: low to medium. Risk: low.

**Enforce source-to-spec link integrity**
- Location: `crates/`, `src/`, `scripts/`, and `specs/` references to `specs/*.md`.
- Problem: The three broken paths from the last audit were corrected, and all 484 current code references resolve. The promised repeatable check was never added, so a future rename can recreate silent link rot. Draft prose may mention a proposed path in a code span; those mentions are not live links.
- Proposed fix: Add a small repository test that resolves cited source paths and actual Markdown links, while excluding plain code-span examples of planned files. Run it in the existing test task.
- Effort: low. Risk: low.

## Explicitly not flagged

These are measurements at `25410760` on 2026-09-23, not permanent verdicts.

- `src/sfmtool/_commands/xform.py` remains a Click option surface rather than the parser's implementation; the implementation is the 772-line `_arg_parser.py` finding above.
- `_embed_patches.py` (833 lines) retains a staged pipeline and delegates compaction; a size-only split would hide the sequence.
- The six Python helper pairs closed in September still call their shared helpers (`_range_options`, `_image_load`, `motion/ratio_band`, `_sfmr_naming`, `_pose_math`) rather than forking again.
- The PyO3 patch prologue now lives in a shared resolver; old inline copies are gone. `verify_sfmr` uses per-section helpers; KDF `write_into` is an 87-line driver. `SectionDigests`, the matches backbone check, and shared workspace metadata remain consolidated.
- License headers and spec indexes passed the counts above. `Optional[...]` under package source is zero. Bench `surfel` residue is zero in its glossary scope; renderer `surfel` is valid. The 177 em dashes in specs are a gradual prose issue under the glossary, not a bulk edit recommendation.
- `crates/sfm-explorer/src/scene_renderer/pipelines/` repeats wgpu descriptor field names, while shared G-buffer state is already a common constant. `action_log/panel.rs` and `background/panel.rs` use shared row functions. The copy is mostly declaration syntax rather than a missing abstraction.

## Top 3

1. Split the 10,373-line MCP test file and the 2,361-line catalog/schema section of `mcp/tools.rs` by their existing modules. Their growth since September is the strongest structural signal.
2. Extract the remaining monolithic SFMR and matches format drivers, following the completed verifier and KDF writer pattern while preserving explicit digest order.
3. Replace the 426-line xform argument branch ladder with a table over the already-shared parsers. Its two lower-level duplication findings are done, so the remaining boundary is now clear.

## Design topics carried forward

These are proposals inherited through older snapshots, not hygiene findings.

- **A — Camera bookmarks:** `specs/gui/viewport-navigation.md:707` still says viewpoint bookmarking is absent, and viewer source has no bookmark implementation. Re-evaluate storage alongside current versioned panel layout before a design draft.
- **B — `sfm xform --crop`:** No 3D crop transform appears in `src/sfmtool/xform/`, the xform command, or its command spec. A bounding-volume crop remains unbuilt.
- **C — Pose-aware per-tile source stacks:** `PerSphericalTileSourceStack` still exposes `build_rotation_only` at `crates/sfmtool-core/src/spherical/per_tile_source_stack.rs:271`. `WarpMap::build_with_pose_impl` exists at `camera/warp_map.rs:276`; the per-tile consumer remains to be designed.
