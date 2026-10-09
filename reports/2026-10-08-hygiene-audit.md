# Hygiene audit, 2026-10-08

This is a read-only survey of the whole tree at `18b1f970`, which is branch `hygiene-fix-1007-staged`: `main` at `7a2653f6` plus the nine `hygiene-fix-1007` branches (PRs #858–#866). It follows `skills/audit-hygiene/SKILL.md` and uses the vocabulary in `specs/GLOSSARY.md`. Two changes landed after it was taken and are recorded in status lines: PR #861 gave a leading `_` the standard meaning (internal to `sfmtool`), and PR #868 gave the bindings public modules. It supersedes the 2026-09-23 report. Every finding in that report was done, declined, or recorded as a decision or a follow-up, and the items still open are carried below with fresh measurements. Five reviewers each covered one area: core algorithms, the bench across every layer, the viewer, the bindings and format crates, and Python, tests and layout. Every finding was checked by reading the code.

## Repeatable measurements

- **Size.** Python: 169 package files with 38,992 lines, and 188 test files with 54,986 lines. Rust: 802 `.rs` files with 428,290 lines, up from 332,598 on 2026-09-23 (+29% in 15 days). Most of the growth is in the bench:
  - core `bench/tests.rs`: 5,159 → 7,171
  - viewer `bench.rs`: 1,907 → 3,076
  - `mcp/tests/bench.rs`: 1,995 → 4,411
  - the new `nearby/` module: 16 files, 6,091 lines

  Outside the bench, `geometry/bundle_adjust.rs` grew from 2,254 to 3,230 lines.

  `specs/` holds 210 Markdown files.
- **Longest non-test Rust files:**

  | File | Lines |
  |---|---|
  | `geometry/bundle_adjust.rs` | 3,230 |
  | viewer `bench.rs` | 3,076 |
  | core `bench/steps.rs` | 3,001 |
  | `track_view/body/mod.rs` | 2,589 |
  | `track_view/body/table.rs` | 2,495 |
  | `mcp/mod.rs` | 2,369 |
  | `sfmtool-py/src/bench.rs` | 2,263 |
  | `state/edits.rs` | 2,127 |

- **Longest functions:**

  | Function | Lines |
  |---|---|
  | `solve_lm` | 833 |
  | `mcp::apply_with_window` | 649 (85 `Command` arms) |
  | `mcp::tools::parse` | 610 |
  | `try_localize_patch_keypoints_with_basis` | 547 |
  | `embed_patches` (Python) | 482 |
  | `track_view::body::table::draw_row` | ~480 |
  | `undistort_reconstruction_images` (Python) | 401 |

- **Python's largest files:**

  | File | Lines |
  |---|---|
  | `colmap/io.py` | 886 |
  | `_embed_patches.py` | 837 |
  | `xform/_arg_parser.py` | 834 |
  | `visualization/_epipolar_display.py` | 821 |
  | `compare/_core.py` | 818 |

- **Comment-line share per crate** (lines starting `//`):

  | Crate | Share |
  |---|---|
  | `sfmtool-progress` | 29.7% |
  | `sfmtool-py` | 25.6% (23.2% counting `///` and `//!` only) |
  | `sfmtool-core` | 21.2% |
  | `sfm-explorer` | 19.9% |
  | `sfmtool-kdf-format` | 13.7% |
  | `sfmtool-colmap` | 9.6% |

  Bindings: 532 non-bench binding items, of which 165 have a doc-to-body ratio above 0.5, 17 above 2, and 36 have 30 or more doc lines. The longest blocks are `patches/member_coherence.rs` (145, was 125) and `localize_keypoints.rs` (133, was 122). 7.6% of the bindings' 8-word doc phrases also appear in `sfmtool-core` docs.
- **License headers:** 417 of 417 Python files and 802 of 802 Rust files carry the header.
- **Spec references:** all 617 `specs/….md` citations from code (136 distinct paths) resolve, up from 484. Every spec appears in its directory's README index.
- **Python module-name rule (AGENTS.md):** 0 violations. All 30 `_commands/` modules match the 30 `COMMANDS` rows. No test enforces the rule.
- **Rust module-file and settings-struct rules (AGENTS.md):**
  - 44 `mod.rs` files and 26 settings structs ending in `Params`/`Config` remain, matching the glossary's backlog.
  - No new violation since the rule.
  - `sfmtool-core` declares 45 `*Options`, 34 `*Params` (11 of them per-call values) and 2 `*Config`.
- **Viewer visibility:**
  - 839 bare `pub` lines in 67 private-module files, against 1,433 `pub(crate)` and 512 `pub(super)`.
  - When the rule landed (e63d538b, 2026-09-25) there were 817 bare `pub` lines.
- **MCP catalog:** 90 tools (read 14, viewer 21, edit 16, bench 31, background 4, input 4). Six nested closed objects take their keys from the schema, and three still list theirs by hand.
- **Duplicate long-line pairs:**

  | Pair | Shared lines |
  |---|---|
  | DirectManipulation examples | 106 |
  | `test_sfmtool_pinhole_rust_bindings.py` / `test_sfmtool_fisheye_rust_bindings.py` | 59 |
  | viewer `bench/nearby_tracks.rs` / `bench/track_at_pixel.rs` | 44 (17 at ≥50 characters) |
  | `_global_sfm` / `_incremental_sfm` | 7 (was 46) |

  The `prof.rs` family shares at most 10 lines per pair (was 21–26).
- **Glossary residue in scope:**
  - Bench `frame` used for the placement: about 735 candidate lines, against 87 legitimate senses. This includes 7 `NoFrame` variants.
  - "world units" (the glossary says *scene units*): 6 wire strings plus 17 bench lines.
  - "descriptor index" (the glossary says *SIFT index*): 4 bench lines plus a 16-use test fixture.
  - Bench-scope `surfel`: 0 in Rust, 1 comment in Python.
  - Zero residue for: `refit_camera`, `full_res_cache`, `aniso_ratio`, `unregistered`, `autocorrelation`, `bm_zncc`, `remove_thumbnails`, `_py`/`_rs` exported names, `active item`, `tile zoom`, `Find Anchors`.
- **`indexes` / `indices`:**

  | Where | `indexes` | `indices` |
  |---|---|---|
  | core | 812 lines in 90 files | 247 lines in 66 files |
  | binding source | 519 lines | 105 lines |
  | binding keywords | 14 names | 3 names |

  The glossary has no entry for this.

## Priority recommendations

**Unbind the eight extension names that collide with Python wrappers at the `sfmtool` root**
> _Status (2026-10-08): **Done** — the root no longer re-exports bindings: each extension submodule has a public module (`sfmtool.geometry`, `sfmtool.fileio`, …), so `sfmtool.match_image_pair` and the other seven no longer exist beside their Python namesakes, PR #868. Renaming `merge.correspondences.find_point_correspondences` to `group_point_correspondences` is still open; it now belongs with the `sfm merge` finding below._
- Location: `src/sfmtool/__init__.py:23-32`, the star-imports of `_sfmtool.analysis` and `_sfmtool.matching`.
- Problem: PR #866 removed the `_py`/`_rs` suffixes from 16 bindings. Eight of those names are now bound at the package root with a different signature from a Python function of the same name elsewhere in the package.
  - `match_image_pair`: 23 arrays at the root, against pycolmap objects in `feature_match`.
  - `find_point_correspondences`: three meanings across the root, `_point_correspondence` and `merge.correspondences`. The `merge` one returns groups.
  - `estimate_alignment`, `merge_points_and_tracks`, `build_covisibility_pairs`, `build_frustum_intersection_pairs`, `world_rotate_w` and `world_rotate_w_inverse`.

  `write_sift` already follows the rule that the Python name wins. No in-tree code reads these names through the root.
- Proposed fix:
  - Leave `analysis` and `matching` out of the root star-import, as `bench` already is.
  - Add a test that no root-bound extension name equals a public Python definition.
  - Record the rule in `specs/python-bindings.md`.
  - Rename `merge.correspondences.find_point_correspondences` to `group_point_correspondences`.

  This is best done on #866 before it merges.
- Effort: low. Risk: medium, because the root namespace is public.

**Enforce the viewer visibility rule with `unreachable_pub`**
- Location: `crates/sfm-explorer/src/`. Worst offenders: `scene_renderer/gpu_types.rs` (111 bare `pub`), `state.rs` (104), `scene.rs` (74), `scene_renderer/recon.rs` (63).
- Problem: the AGENTS.md rule is prose only, and bare `pub` grew from 817 to 839 after it landed. Code written after the rule adds them: `refit_spline_prompt.rs` (09-26) has `pub struct RefitSplinePrompt` beside 21 `pub(crate)`. 29 files mix both forms. Only `document.rs` and `state/edits.rs`, which were converged together with the rule, are clean.
- Proposed fix:
  - Add `unreachable_pub = "warn"` to the crate's lints.
  - Run `cargo fix --lib -p sfm-explorer` in one mechanical commit.
  - Point the AGENTS.md rule at the lint, so the CI clippy job enforces it.
- Effort: low. Risk: low, because rustc rejects any wrong change.

**Replace the core's copy of the `.matches` member-status enum**
- Location: `crates/sfmtool-core/src/patch/cluster_refine/params.rs:13-47`.
- Problem:
  - `MemberStatus` re-declares the seven discriminants of `sfmtool_matches_format::ClusterMemberStatus`, and `REFERENCE_UNREFINABLE` re-declares `CLUSTER_REFERENCE_UNREFINABLE`.
  - The comment justifying the copy says core "does not depend on `sfmtool-matches-format`". That is false: `sfmtool-core/Cargo.toml:39` depends on it, and `covisibility.rs:23` imports the type.
  - No test pins the discriminants, and the binding writes the raw `u8` into the file.
- Proposed fix: re-export the format crate's enum and constant, or at minimum add a `const _: () = assert!(…)` for each variant.
- Effort: low. Risk: low.

**Correct the `translate_patch_to_pixel` docstring and stop restating bench steps in four layers**
- Location: `crates/sfmtool-py/src/bench.rs:1008-1032` against `crates/sfmtool-core/src/bench/steps.rs:1520-1553`. The other copies are in `mcp/bench.rs:682` and `mcp/tools/catalog/bench.rs:271`.
- Problem:
  - The Python docstring says every keypoint "becomes the projection of the new centre". The core doc and the code carry each keypoint by the same displacement, and say that resetting to the projection "would scramble the correlation". The same reasoning is written out in the core doc, the MCP handler doc, the catalog description and the docstring, and one copy has drifted.
  - Wrapper doc-to-body ratios: Python `translate_patch_to_pixel` 26/33, `commit` 23/18; MCP `translate_bench_patch` 21/31, `commit_bench_track` 18/16.
- Proposed fix: fix the docstring now. Cut the MCP handler docs to the argument mapping plus an intra-doc link to the core step. The catalog text stays, because it is what a wire user reads.
- Effort: low. Risk: low.

## Carried forward

**Convert the remaining `mod.rs` files and rename the settings structs** (from "Record the module-file and parameter-bag naming rules", Partially done on 2026-10-08, #862)
- Location:
  - 44 `mod.rs` files: 16 in core outside the bench, 15 in `sfm-explorer`, 12 in the bindings and format crates.
  - 24 `*Params` settings structs plus `GeometricFilterConfig` and `RelaxConfig` in `sfmtool-core`, with about 1,300 references.
- Problem: the rules are written down in AGENTS.md and the glossary, and no new violation has been added since. Still to do:
  - Convert each `mod.rs` while its module is being restructured anyway. The viewer findings below for `track_view/body` and `viewer_3d` are such moments.
  - Do the struct renames as a batch.

  `ConstellationParams` cannot become `ConstellationOptions`, because a bench struct already has that name; see "Disambiguate the bench's flat re-exports" below.
- Proposed fix: rename the settings structs in two or three crate-scoped PRs, as clean breaks, updating the bindings and the Python `GeometricFilterConfig` class with them.
- Effort: medium. Risk: medium, because the types are public.

**Choose one word for the per-patch view lists in core and the bindings** (from "Unify patch binding view-set keywords", Needs decision on 2026-10-08, #865)
- Location:
  - Bindings: `crates/sfmtool-py/src/patches/*.rs`.
  - Core: `crates/sfmtool-core/src/patch/{keypoint_localize.rs:1616,1716, keypoint_subpixel.rs:983, member_coherence.rs:553,605,623, normal_refine.rs:443,506, view_selection.rs:1253,1329,1345}`.
- Problem:
  - Python usage is unchanged:

    | Word | Lines | Files |
    |---|---|---|
    | `view_sets` | 83 | 10 |
    | `view_indices` | 19 | 9 |
    | `member_views` | 6 | 1 |
    | `candidate_views` | 3 | 1 |

  - The core is the source of the split. It names the per-patch list `view_sets`, `member_views`, `patch_views` and `track_views`, and its keypoints `starting_keypoints`, `member_keypoints`, `patch_view_keypoints` and `track_keypoints`.
  - `normal_refine::view_indices_from_reconstruction` is also published as `member_views_from_reconstruction` and twice as `track_views_from_reconstruction`, each a one-line forward. `keypoint_localize::track_views_from_reconstruction` has no callers.
  - The five batch drivers repeat the same ~25-line skeleton: length assert, profiler reset, `par_iter` with a cancel check, `counter.finished()`, `expect`.
- Proposed fix:
  - Make the #865 decision (`view_sets` is recommended) for core and bindings together, and record it in the glossary.
  - Keep one `patch::view_sets_from_reconstruction` and delete the aliases.
  - Add `patch::run_over_cloud` beside `PatchCounter`.
- Effort: medium. Risk: medium, because both the Rust and Python names are public.

## Rust: core algorithms

**Share one quantile and extend the median scan test to it**
- Location:
  - `analysis/cluster_census.rs:238`
  - `geometry/affine_factorization.rs:153`
  - `geometry/focal_vote/column_scan.rs:403,411`
  - `geometry/pose_refine.rs:233` (also used by `resect_images/finite.rs`)
  - `patch/member_coherence/decide.rs:88`
  - `numeric.rs:138`
- Problem:
  - There are six private quantiles in three arithmetic forms. The `numeric.rs` doc claims numpy's convention but does not use its arithmetic.
  - `pose_refine::quantile` sorts with `partial_cmp().unwrap_or(Equal)` and returns 0.0 on empty input. That is the NaN and empty-input policy the `numeric.rs` module doc says was removed for medians.
  - The median is guarded by a function-name scan (`numeric/tests.rs:186-320`). Quantiles are not.
- Proposed fix:
  - One `numeric::quantile_of_sorted` with numpy's `_lerp` form, plus `quantile(&[f64])` using `total_cmp`.
  - Point the callers at it.
  - Extend the scan to `quantile|percentile` with an allowlist.
- Effort: low–medium. Risk: medium, because last-bit changes can flip a threshold; pin fixtures before and after.

**Share the Levenberg–Marquardt damping step**
- Location:
  - Same rule: `geometry/pose_refine.rs:196-223`, `geometry/resect_images/finite.rs:395-422`, `reconstruction/triangulation/points.rs:972-991`, `analysis/cluster_census/group_consistency.rs:422-452`, `geometry/bundle_adjust.rs:2357-2680`.
  - ×10 variants: `camera/refit_intrinsics.rs:1407`, `triangulation/point_or_bearing.rs:870`.
  - `INVALID_RESIDUAL = 1e6` defined at `bundle_adjust.rs:52`, `resect_translation.rs:39`, `pose_refine.rs:24`, `cluster_census.rs:37`.
- Problem:
  - Five copies apply the same rule: damp the diagonal by λ·max(diag, 1e-12), ×4 on failure, ÷2 on success with a floor of 1e-12, and give up above 1e12.
  - `pose_refine` and `finite` are line-for-line identical 6-DoF refits. `finite` already imports the Jacobian helpers from `pose_refine`.
- Proposed fix:
  - Collapse `finite::refit` onto `pose_refine` first.
  - Then add one `numeric::marquardt_step`.
  - Define one crate-level `INVALID_RESIDUAL`.
- Effort: medium. Risk: medium, because convergence must stay bit-identical; the `newton_2d` extraction is the precedent.

**Give the bundle-adjustment kernel an options struct and split `solve_lm`'s setup**
- Location: `crates/sfmtool-core/src/geometry/bundle_adjust.rs`, 3,230 lines.
  - `pub fn bundle_adjust` at 823 takes 20 arguments.
  - `solve_lm` runs 1899–2731 (833 lines, was 617).
- Problem:
  - Nine caller-chosen settings are positional (`free_points`, `protected_loss_scale`, `opt_f`, `opt_k1`, `opt_bspline`, `schedule`, `max_iters`, `min_track`, `min_obs`).
  - Production call sites pass runs of bare booleans, for example `reconstruction_growth.rs:1221-1223`.
  - About 300 lines at the top of `solve_lm` build the round layout before the loop.
  - The file also holds `PointConstraints` and the error types (88–410).
- Proposed fix:
  - Add a `BaOptions` struct.
  - Extract a private `RoundLayout::new`.
  - Move the constraints to `bundle_adjust/constraints.rs`.
- Effort: medium. Risk: medium; 19 test call sites, and the numerical fixtures must hold.

**Share the cluster pair tables between the focal vote and rotation initialisation**
- Location: `geometry/focal_vote.rs:430-479,979-1029` and `geometry/rotation_init.rs:132-227`.
- Problem: `PairAccum`, `ImageClusters`, `pair_correspondences` and `build_pair_tables` are the same algorithm, about 75 lines, over different input shapes:
  - `focal_vote`: CSR `cluster_starts` with f32 positions.
  - `rotation_init`: run-length `cluster_indexes` with f64 positions.

  `rotation_init` already imports `ortho_cost` from `focal_vote`.
- Proposed fix: add a private `geometry/cluster_pairs.rs` over CSR.
- Effort: low–medium. Risk: low.

**Call the remap helpers that already exist, and share the fisheye ray Jacobian**
- Location:
  - `camera/remap.rs:719-733` (against `aniso_footprint` at 511–531) and `remap.rs:402-404` (against `ImageU8Pyramid::full_levels`).
  - `camera/sampler.rs:134` against `remap.rs:315`.
  - `camera/distortion/kernels/equidistant.rs:469-517` against `sfmtool_fisheye.rs:215-267`.
- Problem:
  - `sample_aniso_with_grad` inlines the derivation that `aniso_footprint` provides, under "Mirror the value-only LOD selection".
  - `bilinear_mip_level` (f64) and `mip_level_for_sigma` (f32) are one formula at two precisions, so they can disagree near σ = 2^(k+½).
  - The two fisheye kernels share a 45-line ray-Jacobian body (17 long lines). They differ only in how θ_d is computed.
- Proposed fix:
  - Call the helpers.
  - Derive both mip levels from one function, or add a test that they agree.
  - Add `radial_ray_jacobian(rx, ry, rz, |θ| …)` in `kernels/`, as was done with `newton_2d`.
- Effort: low. Risk: low, with an undistort hash check.

**Add one `CameraModel::has_bare_focal`**
- Location: `geometry/bundle_adjust.rs:1776-1785` and `camera/intrinsics.rs:736-747`.
- Problem: the five-model list is written twice, and the two copies are tied together only by "`with_focal` mirrors this gate".
- Proposed fix: one method that both sites call.
- Effort: low. Risk: low.

## Rust: the bench

**Finish the `frame` → placement migration in the bench's public names and messages**
- Location:
  - `crates/sfmtool-core/src/bench/{commit,evaluate,fit,geometry_search,normal,stage,steps}.rs`
  - `crates/sfmtool-py/src/bench.rs`
  - `crates/sfm-explorer/src/bench/geometry.rs`
  - `mcp/tools/catalog/bench.rs`
- Problem: the glossary retires `frame` for the geometry in bench scope, and the replacement word is **placement**.
  - Seven public `NoFrame` error variants.
  - "the track carries no patch frame…" in four user-facing messages that reach Python and the wire.
  - 82 `frame: &OrientedPatch` parameters, and 83 identifiers such as `anchored_frame` and `has_patch_frames`.
  - About 120 of the lines in the sampled files were added after the 2026-09-20 ruling.

  In the same scope:
  - "world units" in 3 wire tool descriptions (`catalog/bench.rs:283,318,411`) and one docstring (py `bench.rs:1071`).
  - "descriptor index" in the user-facing `SearchError` text (`search.rs:142`).
  - `SplitSettings`, against the "fit normal" entry's ruling on *split*.
- Proposed fix:
  - Rename `NoFrame` → `NoPatch` and reword the messages.
  - Fix the user-facing "world units" and "descriptor index" strings.
  - Rename `SplitSettings` → `PieceSettings`.
  - Add a direction-of-travel note to the `placement` entry, so local variables convert when their file is touched.
- Effort: medium. Risk: low; update the error-message tests.

**Disambiguate the bench's flat re-exports**
- Location: `crates/sfmtool-core/src/bench/mod.rs:61-121`, `features/kdforest/constellation.rs:73`, `bench/search.rs`.
- Problem: about 180 names are re-exported flat into `bench::`. They include:
  - `ConstellationOptions` (cascade member), `ConstellationSeedOptions` (nearby source) and kdforest's `ConstellationParams`.
  - `ClusterMember` (a struct) and `ClusterMembers` (a policy enum).
  - Generic names that belong only to the descriptor search: `SearchOptions`, `SearchError`, `Found`, `DEFAULT_RADIUS_PX`.

  `search_geometry` lives in `geometry_search.rs` with `GeometrySearch*` types, while `search_descriptors` lives in `search.rs` with bare `Search*` types.
- Proposed fix:
  - `ConstellationParams` → `ConstellationQueryOptions`. Its doc says "Tunables for `constellation_query`", and the name is free. This settles the glossary's open note.
  - `search.rs` → `descriptor_search.rs` with `DescriptorSearch*` types.
  - Either suffix the cascade member options `*MemberOptions`, or stop re-exporting them flat.
- Effort: medium (`ConstellationParams` has 73 references). Risk: low.

**Share the constellation sizing rule, and drop a dead override**
- Location:
  - `nearby/constellation.rs:157-170`
  - `track_at_pixel/members.rs:830-846`
  - `crates/sfm-explorer/src/bench.rs:151,2253-2257,2312-2328`
  - `bench/search.rs:48`
- Problem:
  - Three callers each build `SearchOptions` with a `ConstellationParams { min_inliers, ..}` override, and `SearchOptions`' own doc says that field is not read.
  - The target of 50 features is written four times, including implicitly as `DEFAULT_RADIUS_PX = 128`.
- Proposed fix:
  - Add a `kdforest::DEFAULT_CONSTELLATION_FEATURES` constant.
  - Add a `SearchOptions::sized_for(width, height, keypoint_count, target, min_inliers)` constructor.
  - Drop the override.
- Effort: low. Risk: low.

**Move the shared bench inputs out of `track_at_pixel`, and reuse its helpers**
- Location:
  - `crates/sfmtool-core/src/bench/track_at_pixel/{neighbourhood,finish}.rs`, imported by 9 `nearby/` files.
  - `track_at_pixel/members.rs:711-790` against `depth_modes` (246) and `half_px_of` (273).
- Problem:
  - `ViewCamera`, `MatchesClusters`, `SiftIndexSource`, `seed_cluster_with` and `upgrade_sightings` are shared inputs, but they live in the cascade module.
  - `local_prior` re-implements two helpers in its own file.
  - `SweepOptions`, `ConstellationOptions` and `TransferOptions` each restate the local-prior fields with the same defaults, except `prior_k`, which is 10 in one and 8 in the others.
- Proposed fix:
  - Add a bench-level `sources.rs`.
  - Have `local_prior` call the helpers.
  - Nest a `LocalPriorOptions` in each member's options, keeping the per-member values.
- Effort: low–medium. Risk: low to medium, because Python's dotted option keys would change.

**Share the viewer's pixel-job setup and the `TrackMeasurement` serialisation contract**
- Location:
  - Viewer `bench/nearby_tracks.rs:218-298,577-684` and `bench/track_at_pixel.rs:246-335,496-560`.
  - Viewer `bench.rs:3062`.
  - `crates/sfmtool-py/src/bench/{track_at_pixel.rs:62-104, nearby.rs:64-145}`.
  - Py `bench.rs:305-450` and `mcp/bench.rs:1743-1812`.
- Problem:
  - Find Nearby Tracks and Create Track Here share 44 distinct lines. They have the same refusal, camera lookup, off-photograph check, `refresh_index_files`, index gathering, decode closure, and a verbatim 18-line cluster-patches read with its warning. They differ in how they handle a missing `.sift` file.
  - `created_points` is written twice.
  - Python's `TrackAtPixelSources` and `NearbyTrackSources` each parse the same inputs.
  - `TrackMeasurement`'s 33 fields are listed by hand in both the Python and the MCP serialiser, and no test checks that either one covers every field.
- Proposed fix:
  - Add a viewer `bench::pixel_query` module (`IndexSources`, `pixel_on_photograph`, `decode_then`, `read_cluster_patches`).
  - Choose one `.sift` policy on purpose.
  - Add `CreatedPoints::of(value, indexes)`.
  - Add a key-coverage test per serialiser.
- Effort: low–medium. Risk: low; the refusal text is pinned.

**Fold the bench error enums**
- Location: `crates/sfmtool-core/src/bench/fit.rs:126-130` against `evaluate.rs:270-330`.
- Problem: `FitError` repeats six `EvaluateError` variants with copied Display strings and a field-by-field `From`. `StageError` already wraps `Fit(FitError)`. `NoFrame` exists seven times with seven wordings.
- Proposed fix: `FitError::Evaluate(EvaluateError)` and one shared no-patch precondition error. Do this together with the `frame` rename.
- Effort: low–medium. Risk: low.

**Split the bench's largest files along their sections**
- Location and problem:
  - **Core `bench/tests.rs`: 7,171 lines, 167 tests.** Geometry-search tests sit at 94–331, and the classify tests run about 1,650 lines.
  - **`mcp/tests/bench.rs`: 4,411 lines.**
  - **Viewer `bench.rs`: 3,076 lines.** It holds focus and recent-item state, synchronous steps, and background jobs.
  - **Core `steps.rs`: 3,001 lines.** Its sections are create, verdicts, gestures, thresholds and split.
  - **`nearby/{find,layer,source}_tests.rs`.** Three of the workspace's six `*_tests.rs` files, against 212 `tests.rs`.
  - **`scripts/track_at_pixel/anchors.py`: 2,147 lines.** It keeps about 800 lines of Python reference implementations behind five `*_impl` switches, all defaulting to `"rust"`.
- Proposed fix:
  - Core: continue `bench/tests/` per production module, and move geometry-search tests to `geometry_search/tests.rs`.
  - Viewer: split `bench.rs` into `bench/focus.rs` and `bench/jobs.rs`.
  - Split `steps.rs` into `steps/{create,verdicts,gestures,thresholds,split}.rs`.
  - Move the nearby tests to `tests.rs` form.
  - Decide whether to delete the Python references in `anchors.py` or keep them under a parity test.
- Effort: medium (mechanical moves). Risk: low.

## Rust: the viewer

**Finish deriving nested MCP keys from the schema, and make the test walk the catalog**
- Location:
  - Hand-listed parsers: `mcp/display.rs:243` (`feature_size_px`), `mcp/display.rs:273` (`intrinsics`), `mcp/tools.rs:1458` (`bundle_adjust.cameras[]`).
  - Test: `mcp/tests/catalog.rs:556`.
  - `window.rs:236` and `mcp/tests/catalog.rs:1114`.
- Problem:
  - The 2026-09-23 derivation (#581) covers six nested objects. Two older objects were never converted, and `bundle_adjust.cameras` arrived three days later with a hand-written list.
  - The regression test names its six pairs by hand, unlike the top-level test, which walks the whole catalog.
  - `window` keeps two literal key lists that nothing links.
- Proposed fix:
  - Let `reject_unknown_nested` accept an `items` path, and call it from the three parsers.
  - Make the test walk every closed nested object in the catalog.
  - Share one `WINDOW_KEYS` constant.
- Effort: low. Risk: low.

**Put the shared WGSL declarations in one prelude and test the pick tags against Rust**
- Location: `src/shaders/{points,patch,frustum,image_quad,distorted_quad}.wgsl`, against `scene_renderer/picking.rs:23-32` and `gpu_types.rs`.
- Problem:
  - `PICK_TAG_NONE` is written in 5 shaders, `PICK_TAG_FRUSTUM` in 3, `PICK_TAG_POINT` in 2 and `INF_DEPTH` in 3. The Rust side derives them from `PICK_TAG_SHIFT = 30`.
  - The `ReconUniforms` block is byte-identical in all five shaders, which `points.wgsl:32` acknowledges in a comment.
  - `pipelines/tests.rs` compiles every shader but never checks the tag values or that the five blocks agree.
- Proposed fix: add a `shaders/common.wgsl` prelude, included with `concat!(include_str!…)`, and a test that parses its `PICK_TAG_*` literals against `picking::PICK_TAG_*`.
- Effort: low. Risk: low.

**Move photograph decoding and the history cursor out of `state/edits.rs`**
- Location: `crates/sfm-explorer/src/state/edits.rs`, 2,127 lines (was 1,843).
- Problem:
  - `ViewSources`, `DecodedViews` (406–595) and `view_sources_for` (1917–1965) decode photographs for any worker job, and the bench reaches into "edits" for them.
  - The undo/redo/jump cursor (1720–1916, plus `CursorMove` at 2036–2080) is a separate concern.
- Proposed fix: move them to `state/views.rs` and `state/history.rs`, following the `edits/switch_camera_model.rs` pattern.
- Effort: low. Risk: low.

**Split `track_view/body` and `viewer_3d` by concern, converting their `mod.rs`**
- Location: `track_view/body/mod.rs` (2,589 lines), `track_view/body/table.rs` (`draw_row`, ~480 lines with 12 parameters), and `viewer_3d/mod.rs` (1,727 lines).
- Problem:
  - `body/mod.rs` carries about 560 lines of cell text and colour helpers (1202–1760), imported by three other modules, and about 340 lines of header and summary drawing.
  - `draw_row` draws every column inline.
  - About 800 lines of `viewer_3d/mod.rs` (880–1720) are camera transitions and camera-view mode. The siblings `camera.rs`, `framing.rs` and `righting.rs` already show the per-concern layout.
- Proposed fix:
  - Split `body/` into `cells.rs` and `header.rs`, with one draw method per column group.
  - Split `viewer_3d/` into `transition.rs` and `camera_view.rs`.
  - Convert both `mod.rs` files in the same change.
- Effort: medium. Risk: low; the headless tests run whole frames.

**Fix the retired words on the wire and three MCP name breaks**
- Location:
  - "world units": `mcp/tools.rs:1831` (error text), `mcp/tools/catalog/viewer.rs:206`, the bench catalog strings above, and docs at `mcp/mod.rs:818` and `bench/geometry.rs:72`.
  - "bench surfel": `background/mod.rs:224`.
  - "point track table" for Track View: `platform/mod.rs:36` and `platform/tests.rs:29`.
  - Names: `spin_bench_patch`/`spin_bench_shape`'s `degrees` (`catalog/bench.rs:507,536`) and `set_solo` (`catalog/viewer.rs`).
- Problem:
  - The read tools already say "scene units", so the catalog uses both words.
  - Every other argument that carries a unit is `<quantity>_<unit>`; `degrees` is the exception.
  - `set_solo` is the only tool that addresses a reconstruction without naming it.
  - A third case needs a ruling: `set_bench_track_verdict` acts on observations, but its pin forms apply track-wide.
- Proposed fix:
  - Reword the strings.
  - `degrees` → `angle_deg`, and `set_solo` → `solo_reconstruction`, as clean breaks like `get_reconstruction_history` (#864).
  - Extend `the_wire_vocabulary_holds_across_the_catalog` to refuse "world units" and bare unit names.
  - Record the verdict-tool ruling in the glossary.
- Effort: low. Risk: medium, because these are wire names.

**Share the bench handle reach between the two views**
- Location: `image_detail/bench_track.rs:102,110,575` and `viewer_3d/bench_track.rs:92,99,676`.
- Problem: `HANDLE_HIT_RADIUS = 9.0`, `EDGE_HIT_WIDTH = 8.0`, the `nearest` closure and its doc are defined twice. The 3D copy's doc points at the other. `crate::bench` already holds the shared hit helpers.
- Proposed fix: move them to `crate::bench`. The test order deliberately differs between the two views and stays as it is.
- Effort: low. Risk: low.

## Rust: bindings and format crates

**Settle the per-image array keyword names, and range-check `match_image_pairs_batch`**
- Location:
  - `crates/sfmtool-py/src/matching/image.rs:132-185`
  - `geometry/bundle_adjust.rs`
  - `patches/views.rs:334`
  - `io/camrig.rs:96`
  - `io/sfmr.rs:153-155`
  - `geometry/convention.rs:84-150`
- Problem:
  - The camera index per image is `camera_indexes` (uint32) in `CameraViews`, `write_camrig` and the `.sfmr` getter. It is `image_camera` in `bundle_adjust`, and `camera_indices` as `int64` in `match_image_pairs_batch`.
  - `match_image_pairs_batch` casts with `as usize` without a range check, and the core then indexes `intrinsics` with it (`feature_match/mod.rs:240`). A negative or out-of-range value raises `PanicException`. Its one caller converts a uint32 column to int64 just to satisfy the binding.
  - Rotations are `quaternions_wxyz` in 15 functions and `quats_wxyz` in 4.
  - Translations are `translations` in 12 functions and `translations_xyz` in 7. One `.sfmr` column has both names: `read_sfmr()` returns `translations_xyz`, and the getter is `.translations`.
  - Observation arrays are `obs_image`/`obs_point` in some functions and `track_image_indexes`/`track_point_indexes` in others.
- Proposed fix:
  - Add a glossary entry naming `camera_indexes`, `quaternions_wxyz` and `translations_xyz`, which match the file formats.
  - Make `match_image_pairs_batch` take uint32 with a range check.
  - Add a registration test that refuses the retired spellings.
- Effort: medium. Risk: medium, because `bundle_adjust` and `CameraViews` have keyword callers.

> _Status (2026-10-09): **Partially done** — `match_image_pairs_batch` now checks list lengths, pair image indexes and camera indexes (negative or out of range) and raises `ValueError` instead of `PanicException`; it still takes `int64` camera indexes, branch `hygiene-fix-1009-03-pair-index-range-check`; the keyword spellings (`camera_indexes`/`camera_indices`/`image_camera`, `quaternions_wxyz`/`quats_wxyz`, `translations`/`translations_xyz`, `obs_*`/`track_*_indexes`) and the switch to `uint32` still need a maintainer decision._

**Map format errors to one Python exception table**
- Location: `crates/sfmtool-py/src/io/{sfmr,sift,matches,camrig}.rs`, `spatial/kdf.rs:55`, and each format crate's `types.rs`.
- Problem:
  - The five error enums share their variants, but the bindings map them differently:
    - sfmr, sift and matches raise `IOError` for everything, including bad caller data. `test_sfmr_io_rust_bindings.py:189` pins this.
    - camrig maps structural variants to `ValueError`, except `verify_camrig`, which skips the helper.
    - KDF maps structural variants to `OSError` and query errors to `ValueError`.
  - `HashMismatch` is declared in four crates and constructed nowhere. KDF calls the same concept `Integrity`.
  - The five `From<ArchiveIoError>` impls are identical.
- Proposed fix:
  - One variant-to-exception table: I/O variants → `OSError`, input-validation variants → `ValueError`.
  - Drop `HashMismatch`, or align its name with `Integrity`.
  - Add a test that triggers each variant for each format.
- Effort: low–medium. Risk: medium, because exception classes change for callers.

**Add an XXH128 hex parser to `archive-io` and delete four copies**
- Location:
  - `crates/sfmtool-kdf-format/src/verify.rs:300` (`parse_hash_bytes`)
  - `kdf-format/src/read.rs:1662-1676` (`hash_string`, `parse_hash`)
  - `crates/sfm-explorer/src/sift_index.rs:909`
  - `crates/sfmtool-core/src/reconstruction/embed.rs:258`
  - `crates/sfmtool-py/src/spatial/kdf.rs:87`
- Problem:
  - `archive-io` has `format_hash` but no inverse, and each copy restates the byte-order rule in a comment.
  - `parse_hash_bytes` skips the 32-character check, so `"ab"` decodes to 15 zero bytes followed by `0xab`.
  - `hash_string` is a byte-identical copy of `format_hash`.
  - `spatial/kdf.rs` re-implements `helpers::py_to_u128_bytes`, which is used at 12 other sites.
- Proposed fix: add `parse_hash` and `hash_bytes` to `archive-io` with a round-trip test, and use them everywhere.
- Effort: low. Risk: low; the length check gets stricter.

**Make the format binding signatures match, and decide the sweep bindings' future**
- Location: Python `spatial.write_kdf`, `verify_*`, `kdf-format/src/types.rs:130`, `crates/sfmtool-py/src/matching/sweep.rs` (404 lines), and `translation_averaging.rs:442-446`.
- Problem:
  - `write_kdf(forest, path, …)` puts the data first, while every sibling and the Rust function put the path first. It uses `compression_level` where the siblings use `zstd_level`.
  - `verify_kdf` raises and returns counts, while the other four return `(is_valid, errors)`. No spec says which shape the family follows.
  - The six sweep bindings have no callers outside `tests/matching/test_sweep_matching.py`, because their Python wrappers were deleted in #249. Their keywords (`max_angle_diff`, `min_tri_angle`, `size_ratio_min`) disagree with `match_image_pair` and the core config.
  - Five `translation_averaging` constants have no references.
- Proposed fix:
  - Change `write_kdf` to take `path` first, with `zstd_level`.
  - State the verify contract in `specs/python-bindings.md`.
  - Either drop the sweep bindings, or rename them to the `match_image_pair` spellings with one shared pair-geometry parser.
  - Drop or test the constants.
- Effort: low. Risk: low–medium (`write_kdf` has 33 references in `scripts/` and `tests/`).

**Shrink rationale in the bundle-adjust and resect binding docs, and share the error closures**
- Location: `crates/sfmtool-py/src/geometry/bundle_adjust.rs:165-291` (127 doc lines) and `resect_images.rs:109-210` (101 doc lines over a 58-line body); the error helpers are listed below.
- Problem:
  - The `protected`, `opt_f`, `opt_k1` and `distance_from` entries repeat the core's reasoning almost word for word. The two files share 191 and 110 eight-word phrases with their core files.
  - Eight of the twelve `*_err_to_py` helpers have the body `PyValueError::new_err(e.to_string())` (`analysis/cluster_radii.rs:92`, `geometry/focal_vote.rs:170`, `patches/{consensus_atlas,photometric_ransac}.rs:22`, `reconstruction/edited.rs:43`, `spherical/tile_rig.rs:21`, `spherical/tile_source_stack.rs:20,424`).
  - That closure is also written inline 80 times, plus 33 times for `IOError`. `edited.rs` defines `edit_err` and still inlines the closure five times.
- Proposed fix:
  - Keep the Args/Returns reference text and replace the reasoning with a spec pointer.
  - Add `helpers::value_err` and `helpers::os_err` taking `impl Display`, and keep only the three helpers that map by variant.
- Effort: low. Risk: low.

**Split `kdf-format/src/read.rs` by job**
- Location: `crates/sfmtool-kdf-format/src/read.rs`, 1,679 lines.
- Problem: one file holds five jobs:
  - raw ZIP I/O (27–182, 1210–1238, 1559–1660)
  - the `KdfFile` API (`open` is 217 lines)
  - content verification (867–997), beside an existing `verify.rs`
  - metadata validation (1239–1465)
  - node decoding (1466–1558)
- Proposed fix: add `read/integrity.rs`, `read/metadata.rs` and `read/raw.rs` as child modules, beside the existing `read/profiling.rs`.
- Effort: medium. Risk: low.

**Rule on `indexes` / `indices` and on count names**
- Location:
  - Core: `track_image_indices` (`reconstruction/data.rs:466`, `data/point_set.rs:285`, `edited.rs:745`), `subset_by_image_indices(image_indices)` (`reconstruction/edit.rs:267`), `spherical/per_tile_source_stack.rs:476`, `features/feature_match/mod.rs:225`.
  - Python: `SfmrReconstruction.subset_by_image_indices`, `src_indices_for_tile`, `num_images`/`n_images`/`num_trees`/`num_levels`/`n_tiles`.
- Problem:
  - `indexes` is the majority in core, in the binding source and in the binding keywords, and it is the word the file formats use.
  - The core method `track_image_indices(point)` means something different from the `.sfmr` column `track_image_indexes`.
  - Count members are `*_count` 19 times, `num_*` 3 times and `n_*` 2 times.
  - The KDF tree classes have `.len()` but no `__len__`.
- Proposed fix:
  - Add glossary entries for `indexes` and `*_count`.
  - Rename the minority names (`subset_by_image_indices` has 13 Python references in 6 files).
  - Add `__len__`.
- Effort: low to decide, medium to rename. Risk: medium, because the names are public.

## Python and test layout

**Delete the dead Python `merge_points_and_tracks`, and test merging with points at infinity**
- Location: `src/sfmtool/merge/correspondences.py`. The dead function runs from line 180 to the end; the copied binding call is `_find_pairwise_correspondences` at 18–36.
- Problem:
  - The function has no callers. `merge/reconstructions.py:78` calls the binding instead, and `specs/core/reconstruction/point-correspondence.md:109` calls it a "superseded reference implementation".
  - `_find_pairwise_correspondences` copies the binding call from `_point_correspondence.find_point_correspondences` but leaves out `_finite_pair_mask`.
  - The core `merge_points_and_tracks` stores `[f64; 3]` with no w. Points at infinity may therefore be merged as unit-distance finite points. This was not verified at runtime.
- Proposed fix:
  - Delete the function and the spec note.
  - Build the pairwise step on the shared helper.
  - Add a merge test with points at infinity.
- Effort: low. Risk: low for the deletion; any change to how infinity is handled is a behaviour change.

**Stop importing underscore names across subpackages, and enforce both naming rules with a test**
> _Status (2026-10-08): **Superseded** — the rule this finding assumed changed after the snapshot. AGENTS.md § "Python names and privacy" (PR #861) now gives `_` the standard meaning, internal to `sfmtool`, and lets code inside `src/sfmtool/` import an internal name from any subpackage, so the 42 cross-package imports are allowed. PR #868 enforces the part of the rule about the extension: `tests/test_module_layout.py` fails on a path into `sfmtool._sfmtool` from `tests/`, `scripts/` or `docs/`. Still open under the new rule: scripts that import internal Python modules (`sfmtool._workspace_image.read_workspace_image` at `scripts/add_image_to_tracks/harness.py:367`, `misses.py:47` and `scripts/track_at_pixel/context.py:160`; `sfmtool._cluster_patches._run_cluster_patches` and `feature_match._run._write_clusters_matches` at `scripts/track_at_pixel/dataset.py:218,221`; `feature_match._flow_matching` at `scripts/benchmark_flow_matching.py:114`), extending the layout test to them, and choosing which plain-named modules in module-path subpackages are helpers that should take `_` (`colmap/db_builders.py`, `merge/correspondences.py`, `merge/pose_refinement.py`, `camrig/pattern.py`, `rig/frames.py`, `sift/draw.py`, `motion/constants.py`, `flow_stats.py`, `ratio_band.py`, `analyze/point_or_bearing.py`)._
- Location: 42 import statements of 51 `_`-named functions and constants. The most common:
  - `camera/cameras.py`'s `_CAMERA_PARAM_NAMES`, imported by 8 modules even though `CAMERA_MODEL_NAMES` is public.
  - `camera/setup.py`'s `_infer_camera`, `_read_image_size` and `_check_camera_model_conflict`.
  - `rig/config.py`'s `_load_rig_config` and three siblings.
  - `colmap/io.py`'s `_build_sfmr_data_dict`.
  - `motion/image_sequence.py:22` importing three `visualization._discontinuity_display` printers.
- Problem: the new AGENTS.md rule settles what `_` means on a module name but says nothing about `_` on a function or constant. In the module-path packages these names are the cross-package API under a private spelling.
- Proposed fix:
  - Rename the cross-package names to plain names.
  - Add the rule to AGENTS.md § "Python module names".
  - Add `tests/test_module_layout.py`, which checks this rule and the module-name rule. The module-name rule has no test today.
- Effort: low–medium (about 25 sites). Risk: low.

**Share the flow display's advection hits, and choose one separate-file naming**
- Location: `src/sfmtool/visualization/_flow_display.py`, 736 lines (`draw_flow_visualization` is 255 lines).
- Problem:
  - The in-bounds mask, nearest-feature lookup and index remap appear three times (260–280, 437–455, 532–545).
  - The distance loop and green/red split appear twice (327–345, 514–529).
  - `sfm flow --separate` writes `<stem>_A`/`<stem>_B`, while `sfm epipolar --separate` writes `<path>`/`<stem>_other`.
- Proposed fix: compute one `_FlowHits` in the driver, following the epipolar split (#793), and choose one naming for both commands.
- Effort: low, plus the naming decision. Risk: low; the file naming is user-visible.

**Share the `.sfmr` path check across commands**
- Location: 24 sites in 18 `_commands/*.py` modules, for example `analyze.py:236`, `heatmap.py:134`, `xform.py:447,452`, `merge.py:62,67`.
- Problem: the same validation is written by hand with 12 different error wordings.
- Proposed fix: a `_commands/_sfmr_path.py` helper or a Click `ParamType`.
- Effort: low. Risk: low; 4 test assertions match the current strings.

**Cut implementation and benchmark detail out of `embed-patches` and `cluster-patches` help**
- Location: `src/sfmtool/_commands/embed_patches.py:224` (`--localize-search-strategy`: "AVX2 single-position vgather kernel", "~1.9× faster end-to-end on dino") and `:258` (`--sampler`), and `cluster_patches.py:39` (`--patch-size`: "passed to refine_cluster_patches").
- Problem:
  - These options carry rationale where an option reference belongs. `embed_patches.py` has 6,493 characters of help across 19 options.
  - The 2026-09-23 word list missed these lines.
- Proposed fix: say what each option does, point to the spec, and move the figures into the specs.
- Effort: low. Risk: low.

**Extract the per-image step from `undistort_reconstruction_images`**
- Location: `src/sfmtool/_undistort_images.py`, 613 lines. The function runs from line 213 to 613, and its loop (318–480) is marked "Step 1" to "Step 6".
- Problem: one function does all of this:
  - warps each image and writes it
  - makes the thumbnail
  - transforms and writes the SIFT features, and reads back their hashes
  - remaps tracks, and builds the cameras and rig frames
  - assembles the `.sfmr` through the cross-package private `_build_sfmr_data_dict`
- Proposed fix: extract a `_undistort_one_image` helper and an `.sfmr` assembly helper, guarded by byte-identical output on `seoul_bull`.
- Effort: medium. Risk: low–medium.

**Parametrize the spline-camera binding tests**
- Location: `tests/rust_bindings/geometry/test_sfmtool_pinhole_rust_bindings.py` (324 lines) and `test_sfmtool_fisheye_rust_bindings.py` (301 lines).
- Problem: they share 59 long lines. 13 parallel tests and three helpers differ only in the model name and the `rho_max`/`theta_max` key.
- Proposed fix: one module parametrized over a model description. Keep the seam tests in each file.
- Effort: low–medium. Risk: low.

**Minor, can go in one cleanup:**
- `xform/_bundle_adjust.py:81-83`'s `refine_focal_length`, `refine_principal_point` and `refine_extra_params` are pycolmap words that no caller passes, and they conflict with the glossary's **release** entry.
- The `descriptor_index` test fixture (`tests/rust_bindings/bench/test_bench_rust_bindings.py:990`, 16 uses, last changed after the ruling).
- The bench-scope `surfel` comment at line 1310.
- The core's "descriptor index" at `features/kdforest/constellation.rs:7`.

## Explicitly not flagged

These figures were measured at `18b1f970` on 2026-10-08. They are not permanent verdicts: this repo has repeatedly grown a cleared file by 30–90% within a month.

- `xform/_arg_parser.py`, 834 lines (was 772): the dispatch table stayed (`_TRANSFORM_OPTIONS`, 33 entries), and the longest function is 60 lines.
- `visualization/_epipolar_display.py`, 821 lines (was 619): this is the cost of the #793 split. The dispatcher is 192 lines and the next-longest helper 90.
- `_embed_patches.py`, 837 lines: `embed_patches` is 482 lines, still a staged pipeline that delegates compaction.
- `colmap/io.py` (886 lines), `compare/_core.py` (818), `_densify.py` (765) and `analyze/summary.py` (730): each is one concern, with its longest function at 139, 310, 272 and 166 lines respectively.
- `tests/conftest.py`, 1,269 lines: unchanged since the 2026-10-07 decline (#860). No copy of its builders exists outside it.
- `reconstruction/edited.rs` (1,693 lines; longest function ~110) and `camera/refit_intrinsics.rs` (1,461; longest ~132).
- `focal_vote.rs`, `reconstruction_growth.rs` and `cluster_census.rs`: after their stage extractions, no function is in core's top 25 (all under 193 lines).
- `try_localize_patch_keypoints_with_basis`, 547 lines: a staged function with commented phases. Recheck if it passes 600.
- The `prof.rs` family: at most 10 shared lines per pair, with `profiling.rs` at 189 lines.
- `polar.rs` / `sweep.rs`: 9 shared doc lines after the `window.rs` extraction (#859).
- GPU uniform structs marked "Must match WGSL" in core: wgpu checks binding sizes, and CI sets `SFMTOOL_REQUIRE_GPU=1`.
- The `numeric` median: enforced by its scan test, with the other 11 `…median…` names on its allowlist.
- MCP registration in three places (tool list, parser arm, command arm): the exact-set test ties them together, 90 = 90.
- The MCP `item` / `track` argument split (4 / 21 tools): deliberate per `specs/gui/mcp-server.md`.
- `get_bench`, `undo`, `redo` and `jump_to_version`, which take a `reconstruction_label` without naming the reconstruction: acquitted in #864.
- DirectManipulation examples (106 shared lines): standalone diagnostics, checked in Windows CI.
- Shader `Uniforms` blocks: 6 different layouts for distinct buffers. Renderer pipeline descriptors repeat declaration syntax, not a missing abstraction.
- `_s` and `_w` suffixes on binding names (`flip_camera_poses_s`, `world_rotate_w`): domain notation for the matrices S and W.
- `_py`, `_rs`, `_impl`, `_inner` and `_raw` suffixes on Python-visible names: 0 left.
- Shared argument text across the `PatchCloud` methods: reference text each method's `help()` needs; the CSR prose decision (#858) applies.
- `reconstruction/clone.rs`: the longest function is 208 lines (was 663).
- `sfmtool-progress` at 29.7% comments: its 97-line crate doc describes the API of a shared primitive.
- Bench `anchor` (185 uses): the anchored-fit sense the glossary keeps. The `pin_verdict` / `pin_verdicts` split is recorded in the glossary.
- `_global_sfm` / `_incremental_sfm`: 7 shared lines (was 46).

## Top 3

> _Status (2026-10-08): item 1 is done by PR #868._

1. **Unbind the colliding root names before #866 merges.** It is low effort, and it prevents an API hazard that this round's own rename created.
2. **Turn on `unreachable_pub` in `sfm-explorer`.** One lint and one `cargo fix` turn an 839-site prose rule into one that CI enforces.
3. **Replace the core's `MemberStatus` copy with the format crate's enum.** A few lines remove a false comment and a file-format contract that nothing enforces.

Close behind: the wrong `translate_patch_to_pixel` docstring (a one-paragraph fix that users read), and the possible points-at-infinity bug in `sfm merge` beside its dead Python duplicate.

## Design topics carried forward

These are proposals carried over from older snapshots, not hygiene findings.

- **A — Camera bookmarks:** viewpoint bookmarking is still absent from the viewer. Re-evaluate where to store bookmarks alongside the versioned panel layout before writing a design draft.
- **B — `sfm xform --crop`:** no 3D crop transform exists in `src/sfmtool/xform/` or the xform command spec.
- **C — Pose-aware per-tile source stacks:** `PerSphericalTileSourceStack` still builds rotation-only stacks; `WarpMap::build_with_pose_impl` exists, and the per-tile consumer has not been designed.
