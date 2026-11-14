# Hygiene audit, 2026-11-13

This is a read-only survey of the tree at `929b62f0` (`main`). It follows `skills/audit-hygiene/SKILL.md` and uses the words in `specs/GLOSSARY.md`. It replaces `2026-10-08-hygiene-audit.md`. Every still-open finding of that report is carried below with figures re-measured at `929b62f0`, and the findings it had already closed are counted under "Closed since the last snapshot". It is the first run under the budgeted sample (#1172): no earlier hygiene report had a coverage table, in the tree or in git history, so every unit started as never read and the churn pool was empty.

**Sample:** 4 of 257 units read in full, 15,258 of 725,646 lines (13,958 in units and about 1,300 in dives), against a budget of 15,000 lines. The units pulled in by mechanical hits left 1,042 lines for the dives, which read about 260 more than that. Seed `27413`; the draw's pool was never read (257 units), and no drawn unit fit in what was left. Pulled in by mechanical hits:
- `crates/sfmtool-py/src/analysis`: 7 new and 7 total kept hits (checks 12 and 13).
- `crates/sfmtool-py/src/patches`: 4 new and 4 total (check 3 pairs).
- `crates/sfmtool-matches-format/src (rest)`: 3 new and 3 total (check 3 pairs).
- `crates/sfmtool-core/src/bench/nearby/find`: 2 new and 2 total (check 4 pair).

Pulled in by churn: none (first run). Named by the user: none.

Slow-test dives:
- `bench::tests::stored_bitmap::a_viewer_spin_and_resize_then_commit_store_the_render_of_the_frame_they_store` — 60 s, CI test-linux-rust, over 60 s.
- `bench::tests::stored_bitmap::a_viewer_tilt_then_commit_stores_the_render_of_the_frame_it_stores` — 60 s, same.
- `bench::tests::stored_bitmap::refine_normals_stores_the_render_of_the_frame_it_stores` — 60 s, same.

The three share one fixture and were dived together in about 1,300 lines. Skipped as examined in the last 30 days: none.

Drawn: none.

Pulled in, not read, over budget (72 units): `crates/sfmtool-sfmr-format/src (rest)`, `crates/sfmtool-core/src/spherical (rest)`, `crates/sfm-explorer/src/state/edits (rest)`, `crates/sfmtool-core/src/analysis (rest 2)`, `crates/sfmtool-core/src/analysis/infinity`, `crates/sfmtool-core/src/bench (rest)`, `crates/sfmtool-core/src/bench/nearby (rest 1)`, `crates/sfmtool-core/src/bench/tests (rest)`, `crates/sfmtool-core/src/features (rest)`, `crates/sfmtool-core/src/geometry/bundle_adjust (own file)`, `crates/sfmtool-core/src/geometry/reconstruction_growth`, `crates/sfmtool-core/src/patch/keypoint_subpixel`, `crates/sfmtool-py/src/matching`, `crates/sfmtool-sfmr-format/src/tests (rest 2)`, `crates/sfmtool-sift-format`, `src/sfmtool (rest 2)`, `src/sfmtool/_commands (rest 2)`, `tests (rest 3)`, `tests/patch (rest 2)`, `tests/xform (rest 2)`, `crates/sfm-explorer/src/camera_model_proposal`, `crates/sfm-explorer/src/mcp (rest 2)`, `crates/sfm-explorer/src/mcp/tests/input`, `crates/sfm-explorer/src/mcp/tests/widgets`, `crates/sfm-explorer/src/track_table_panel (rest)`, `crates/sfmtool-colmap/src (rest)`, `crates/sfmtool-core/src (rest)`, `crates/sfmtool-core/src/analysis (rest 1)`, `crates/sfmtool-core/src/analysis/alignment`, `crates/sfmtool-core/src/analysis/pick_images/source`, `crates/sfmtool-core/src/features/feature_match/polar`, `crates/sfmtool-core/src/features/kdforest (rest)`, `crates/sfmtool-core/src/geometry (rest 1)`, `crates/sfmtool-core/src/geometry/bundle_adjust/tests (rest)`, `crates/sfmtool-core/src/geometry/epipolar_estimation`, `crates/sfmtool-core/src/geometry/focal_vote (rest)`, `crates/sfmtool-core/src/geometry/rotation_init`, `crates/sfmtool-core/src/patch (rest)`, `crates/sfmtool-core/src/patch/blur_matched`, `crates/sfmtool-core/src/patch/cloud`, `crates/sfmtool-core/src/patch/cluster_refine (rest)`, `crates/sfmtool-core/src/patch/keypoint_localize (rest)`, `crates/sfmtool-core/src/patch/keypoint_localize/tests`, `crates/sfmtool-core/src/patch/member_coherence`, `crates/sfmtool-core/src/patch/normal_refine`, `crates/sfmtool-core/src/patch/reference_view`, `crates/sfmtool-core/src/patch/stored_bitmap`, `crates/sfmtool-core/src/patch/view_selection`, `crates/sfmtool-core/src/reconstruction/bundle_adjust/tests`, `crates/sfmtool-matches-format/src/tests`, `crates/sfmtool-matches-format/src/verify`, `crates/sfmtool-py/src (rest)`, `crates/sfmtool-py/src/geometry (rest 1)`, `crates/sfmtool-py/src/reconstruction (rest)`, `crates/sfmtool-py/src/reconstruction/sfmr_reconstruction`, `crates/sfmtool-py/src/spatial`, `crates/sfmtool-sfmr-format/src/tests (rest 1)`, `src/sfmtool (rest 1)`, `src/sfmtool (rest 3)`, `src/sfmtool/_commands (rest 1)`, `src/sfmtool/camrig`, `src/sfmtool/sift`, `src/sfmtool/visualization`, `src/sfmtool/xform (rest 2)`, `tests (rest 2)`, `tests (rest 4)`, `tests/rig`, `tests/rust_bindings/geometry (rest 2)`, `crates/sfm-explorer/src/bench/nearby_tracks`, `crates/sfm-explorer/src/bench/track_at_pixel`, `crates/sfmtool-core/src/bench/evaluate`, `crates/sfmtool-core/src/bench/fit`.

The mechanical checks below cover all 257 units. Code in the 253 units not read is acquitted only by those numbers, not by this report's silence.

**Test environment note.** The first `pixi run test` of this run failed 77 tests and errored one module because the `sfmtool._sfmtool` extension was older than the Rust tree (`CaptureSequence.__new__() got an unexpected keyword argument 'camera'`, no `HDR_EXPOSURE_STOPS_LIMIT`). After `pixi run maturin develop --release` the suite passed: 4,138 passed, 1 skipped. The timings below are from that second run, which shared the machine with this audit's cargo builds, so they run higher than an idle machine's.

## Mechanical numbers

The tree has grown since 2026-10-08: Rust 994 files and 575,485 lines (was 802 and 428,290), Python 504 files (was 417), `specs/` 246 Markdown files (was 210). Checks 9-21 have no 2026-10-08 baseline unless one is given; their standards were added after that snapshot. Checks 2-4 count by a different rule from last time, so compare their figures as trends.

1. **Doc density** (lines starting `//`, `///`/`//!` share in brackets; 10-08 in parentheses): `sfmtool-progress` 29.8% (29.7), `sfmtool-py` 26.5% [24.2] (25.6 [23.2]), `archive-io` 22.2%, `camrig-format` 20.9%, `sfmtool-core` 20.6% (21.2), `sfm-explorer` 19.0% (19.9), `sift-format` 17.7%, `sfmr-format` 17.3%, `matches-format` 15.5%, `kdf-format` 13.8% (13.7), `colmap` 9.7% (9.6). Longest comment blocks: py `bench/normal.rs:1003` 169 lines (new), py `matching/cluster.rs:362` 154, py `reconstruction/edited.rs:804` 152, py `geometry/bundle_adjust.rs:215` 149, py `patches/member_coherence.rs:36` 145 (unchanged), core `bench/normal/grown_plane.rs:739` 128. `localize_keypoints.rs` went from 133 to 115.
2. **Doc-to-body ratio per item**, every `fn` counted: `sfmtool-py` outside the bench 1,012 functions, 256 over 0.5×, 28 over 2× (was 17), 46 with 30 or more doc lines (was 36); `sfmtool-py` bench 177, 51, 5; viewer `mcp/` 513, 84, 6, 1.
3. **Duplicate long lines** (40 or more normalized characters, in 2-4 files): 35 file pairs share 20 or more lines, 18 of them at 50 or more characters. Kept pairs are in the findings; acquitted after reading: the sfmr / matches `read` / `write` / `verify` pairs (28-43, each with its own entry list), DirectManipulation examples (76, was 106), `platform/windows.rs` (20), viewer test idioms (`image_detail` / `viewer_3d` 22, `mcp/tests` bench / edit 28, bench / render 23, `cluster_patches` / `sift_index` 24), same-module test splits (`track_table_panel` 21, matches `tests/cells` 20).
4. **Parallel families** (lines of 30 or more characters): `prof.rs` largest pair 17 (was 10); format-crate `read.rs` 11, `write.rs` 20, `verify.rs` 20, `types.rs` 32 (structure, plus the documented `relative_pose_conjugate` copy the crate graph forces), acquitted; `_global_sfm` / `_incremental_sfm` 29 at 30 characters, under 20 at 40 (was 7); new pairs `bench/nearby/{far_field,candidate,find}.rs` (27 / 24 / 20), `_commands/match.py` / `solve.py` (20), `camera_model_proposal/job.rs` / `track_table_panel/job.rs` (25), the core bundle-adjust `corrections.rs` test pair (23).
5. **Convention tallies.** `mod.rs` files 44 (was 44) against 245 modules in the `foo.rs`-beside-`foo/` form, none added since 10-08. Public settings structs in `sfmtool-core` ending in `Params` or `Config`: 27 (was 26), two of them added after the rule; `*Options` 58 (was 45). Plain-named modules in the re-exporting subpackages: 0; 33 command modules for 33 `cli.COMMANDS` rows (was 30 / 30). License headers 1,498 of 1,498. `indexes` / `indices`: core 940 lines in 122 files against 319 in 76 (was 812 / 247), binding source 554 / 102 (was 519 / 105), binding keywords 12 / 3 (`match_image_pairs_batch(camera_indices)`, `subset_by_image_indices(image_indices)`, `delete_points(indices)`; was 14 / 3). Error messages: core, `sfmtool-py` and the viewer lowercase (29 / 96 / 95, 0 capitalized); format-crate error prefixes and `sfmtool-colmap` (9 of 17) capitalized.
6. **Public-name leaks:** 0 Python-visible names end in `_py`, `_rs`, `_impl` or `_inner`; no implementation suffix in 388 CLI flag uses; five sibling-flag disagreements (finding below).
7. **Reference integrity:** all 780 `specs/…` citations in code (164 distinct) resolve (was 617 / 136); every spec is listed in its area README, and no README link is broken.
8. **Glossary conformance:** 79 probes. Residue in the finding below. Clean: focused item, recent items, walk, middle ZNCC, ZNCC grid, ZNCC self-similarity radius, distance range, depth layer, Create Track Here, Find Nearby Tracks, Track View, clipped share, blur-matched ZNCC, blur assessment, bitmap for judging, Surface Tiling Mode, refine normals, index files, Build SIFT Files, cluster patches file, image menu, untracked, HDR to sRGB, photograph cache, the view's world-space unit, `quaternion_wxyz`, ZNCC deficit, stated workspace path, sharpness, matcher grey, `min_`/`max_` columns, `back_facing_observations`, `max_error_px`, `reference_zncc_self_similarity_radius`, track table cache, Track Table panel, Filter the 3D view, refit camera intrinsics, image intrinsics, rolling shutter, eraser brush, the Background Tasks wire names, quiet task, reference pick, anisotropic threshold, piecewise refinement, rotation start, the `_py`/`_rs` exported names. Bench `surfel`: 1 deliberate test hit, 0 in Rust.
9. **Points at infinity:** Rust library `positions_xyz\b` 32, `[:, :3]` 0, `/ *w\b` 13, public signatures taking point slices 13; Python `positions_xyz` 31, `[:, :3]` 4, `/ w` 0. None of the pattern hits kept after reading; 5 kept from reading the users of the `recon.positions` getter and one core function.
10. **Progress coverage:** 31 of 230 long-running public functions (13%, bench excluded) take `&Progress`. `Progress::none()` in non-test library code: 81 (core 35, py 38, viewer 7, progress 1); 6 inside a function with its own Progress in scope (1 acquitted with a comment, 5 bindings that pass a counter beside it).
11. **Progress variants:** `_with_progress`, `_without_progress`, `_no_progress`, `_silent`, `Option<&Progress`, `verbose: bool`, `show_progress` 0 each; `_quiet` 2 (false positives). Kept: 3 pairs named `_progress` / `_reporting`, 1 forward to `Progress::none()`, and 5 drivers taking `done: Option<&AtomicUsize>` beside `&Progress`.
12. **Caller-input panics** (non-test; assert / expect / unwrap / panic! / from_row_slice / `# Panics`): core 232 / 161 / 65 / 6 / 5 / 86; py 0 / 48 / 7 / 0 / 29 / 0; sfmr-format 3 / 6 / 55 / 0 / 0 / 1; matches-format 0 / 16 / 29 / 0 / 0 / 0; kdf-format 0 / 26 / 17 / 0 / 0 / 0; sift-format 0 / 1 / 4 / 0 / 0 / 0; camrig-format 0 / 2 / 2 / 0 / 0 / 0; colmap 0 / 0 / 4 / 0 / 0 / 0; archive-io and progress 0; viewer `cli/` + `mcp/` 1 / 69 / 0 / 0 / 2 (whole viewer 2 / 145 / 19 / 1 / 2, 2 `# Panics`); viewer file parsing 16 `expect`, none kept. Python-reachable `PanicException`s confirmed by probe: 15 functions (6 sweep bindings, 8 in `sfmtool.analysis`, `MatchesFile.select_clusters` on a corrupt file). "panic(s)" on 53 lines in 27 spec files, about 18 describing a caller-input panic as the design.
13. **Python over bindings:** 324 top-level Python functions and classes use a binding name; all 25 root `_LAZY_NAMES` are Python names; 6 kept.
14. **Format I/O table:**

    | Format | read | read_metadata | write | verify |
    |---|---|---|---|---|
    | sfmr | fileio | fileio | fileio `(path, data, zstd_level)` | fileio `(bool, errors)` |
    | sift | fileio | fileio | fileio | fileio `(bool, errors)` |
    | matches | fileio | fileio | fileio | fileio `(bool, errors)` |
    | camrig | fileio | fileio | fileio, path first | fileio `(bool, errors)` |
    | kdf | `spatial` | missing (`kdf_file_summary`) | `spatial`, `(forest, path)`, `compression_level` | `spatial`, raises and returns counts |

    `SiftFile`, `MatchesFile` and `SiftWriteQueue` are in `fileio`; no Python reader or writer class exists.
15. **Documented exceptions:** 71 raw hits (AGENTS.md 1, specs 40, Rust docs 28, Python 2); 2 confirmed in the units read, 10 candidates listed in the finding.
16. **Viewer visibility:** 0 bare `pub` fields outside `CatalogTool`'s 3; 0 bare `pub` in Windows-only code (`platform/windows*` and 30 `cfg(target_os = "windows")` sites).
17. **Build and import gates:** `[lints] workspace = true` in 11 of 11 crates, passed; `cargo check -p sfm-explorer --no-default-features`, passed with 0 warnings; `tests/test_lazy_loading.py` and `tests/test_module_layout.py`, passed (18 tests); `cargo test -p sfmtool-core --test license_headers`, passed. 0 imports of `_`-prefixed `sfmtool` names in spec examples.
18. **Medians and quantiles:** `cargo test -p sfmtool-core --lib numeric` passed (26 tests). `select_nth_unstable` 11, `[n/2]` 15, fraction-to-index 7, interpolation position 1; 12 kept.
19. **Numeric precision:** counts in the finding; 5 kept (1 known).
20. **Tests that cannot fail:** Python `is not None` 168, Rust `is_some()` 180, self-comparisons 0, holds and waits 8 Python and 88 Rust; 7 tests kept plus 1 untimed wait.
21. **Test time:** 101 Python setup or call phases over 5 s, 998.1 s in all, under the load noted above. The slowest: `tests/patch/test_cluster_patches.py::test_enriched_output_states_the_refinement` 36.6 s, `tests/test_pick_images_command.py::test_a_sequence_with_no_camrig_takes_the_pilot_vote_s_camera` 30.6 s, `tests/test_camera_config_cli.py::test_solve_uses_camera_config` 24.1 s, `test_pick_images_command.py::test_a_list_with_a_slice_as_input` 23.6 s, `test_json_report_and_reuse_of_sift_files` 22.2 s. 8 Rust tests over 60 s in `test-linux-rust` of CI run 43382150438 (`ae1a20b2`): the four `bench::tests::stored_bitmap` tests and four `reconstruction::bundle_adjust::tests::corrections` tests (`a_given_or_stored_setting_runs_one_solve_on_it`, `a_stopped_race_stops_at_the_same_place_every_time`, `several_unknown_cameras_are_raced_in_order_each_from_the_last_winner`, `the_winner_of_a_stopped_race_is_the_solve_it_would_be_alone`). Run alone, the stored-bitmap tests take 0.8-8.8 s; CI's figure is the coverage build on four shared cores.

## Findings in the units read this run

### `crates/sfmtool-py/src/analysis` (read in full)

All 17 files (4,380 lines) were read in full and checked against the core functions they call; each `PanicException` below was confirmed with a probe against the rebuilt extension.

**Validate the `sfmtool.analysis` inputs that make Python panic, in core**
- Location: `crates/sfmtool-py/src/analysis/core.rs:77-126,134-166,223-248,258-296,404-434,500`, `image_pair_graph.rs:16-35,49-102`; core `geometry/transform.rs:27`, `viewing_angle.rs:56`, `alignment/ransac.rs:21`, `point_correspondence.rs:48,165`, `reconstruction/filter.rs:28`, `image_pair_graph.rs:89,210`.
- Problem: each of these raised `PanicException` in a probe:
  - `apply_se3_to_camera_poses`: translations with fewer rows than the quaternions.
  - `compute_narrow_track_mask`: short translations, or positions of shape (1, 2).
  - `ransac_alignment`: (3, 2) inputs, or a shorter target.
  - `find_point_correspondences` and `filter_tracks_by_point_mask`: track columns of different lengths.
  - `build_covisibility_pairs`: image index 5 with 1 image.
  - `build_frustum_intersection_pairs`: 1 intrinsics row for 2 images.
  - `merge_points_and_tracks`: `image_mapping={"a":[(5,0)]}` with one reconstruction indexes `reverse_mapping[recon_idx]` unchecked at `core.rs:500` ("index out of bounds: the len is 1 but the index is 5").

  `estimate_alignment` and `merge_points_and_tracks` return `Result<_, String>` rather than an error enum, one string capitalized (`least_squares.rs:63,66`). [Caller input never panics](../specs/STANDARDS.md#caller-input-never-panics).
- Proposed fix: each core function returns an error enum with a lowercase Display, and the binding maps it with `value_err`; the `image_mapping` → `reverse_mapping` construction moves into core `merge_points_and_tracks`, where the range check belongs.
- Effort: medium.
- Risk: low — core signatures change; the bindings keep theirs.

> _Status (2026-11-13): **Done** — `compute_narrow_track_mask`, `estimate_alignment`, `ransac_alignment`, `find_point_correspondences`, `build_covisibility_pairs`, `build_frustum_intersection_pairs` and `merge_points_and_tracks` return `NarrowTrackMaskError`, `AlignmentError`, `PointCorrespondenceError`, `ImagePairGraphError` and `MergePointsAndTracksError`, each with a lowercase `Display`, mapped to `ValueError` with `value_err`; `merge_points_and_tracks` takes the image mapping and builds the reverse mapping itself, checking every reconstruction, image and point index; branch `hyg-fix-1113-02-analysis-input-validation`. `apply_se3_to_camera_poses` and `filter_tracks_by_point_mask` are removed instead, on branch `hyg-fix-1113-01-delete-dead-analysis-bindings` (PR #1232)._

**Delete `apply_se3_to_camera_poses` and `filter_tracks_by_point_mask`**
- Location: `core.rs:72-126`, `core.rs:399-434`; core `geometry/transform.rs:27-47`, `reconstruction/filter.rs:28`.
- Problem: outside their crate the only caller of either binding is `tests/rust_bindings/analysis/test_analysis_registration.py`, which checks only that the name is registered, and the core functions are called only by their own `tests.rs`. `Se3Transform.apply_to_camera_poses` (`geometry/se3_transform.rs:253`) is the validated binding for the same operation, and `EditedReconstruction::filter_points_by_mask` (`edit.rs:627`) does the remap inline. Two of the panics above are in these two.
- Proposed fix: remove both bindings, both core functions, their tests and the registration entries.
- Effort: low.
- Risk: low — public `sfmtool.analysis` names, but nothing in `src/`, `scripts/` or the docs calls them.

> _Status (2026-11-13): **Done** — removed the `apply_se3_to_camera_poses` and `filter_tracks_by_point_mask` bindings from `sfmtool-py/src/analysis/core.rs`, the core modules `geometry/transform.rs` and `reconstruction/filter.rs` with their tests, the two names from `test_analysis_registration.py`, their rows in `specs/python-bindings.md` and the paragraph naming `filter_tracks_by_point_mask` in `xform-command.md`; the panics that "Validate the `sfmtool.analysis` inputs that make Python panic, in core" lists for these two functions went with them, branch `hyg-fix-1113-01-delete-dead-analysis-bindings`._

**Move the input checks of ten analysis bindings into their core functions**
- Location: `triangulation.rs:361-387` (`csr_offsets`, whose doc says "otherwise the core would ... panic"), `point_or_bearing.rs:196-204,254-269,279-310`, `source_clusters.rs:85-135`, `adjacency_surfel_normals.rs:347-393`, `observation_adjacency.rs:83-147`, `observation_coverage.rs:286-317`, `epipolar.rs:159-171`, `core.rs:351-370`, `cluster_radii.rs:278-293`, `cluster_census.rs:124-149`.
- Problem: core `triangulate_batch`, `bearing_score_batch`, `fit_point_and_bearing_batch`, `source_clusters`, `estimate_adjacency_surfel_normals`, `build_observation_adjacency`, `ObservationCoverage::build`, `plot_epipolar_curves_batch`, `find_point_correspondences_by_coordinate` and `cluster_radii` / `coarsest_cluster_ids` return plain values and hold 40 `assert!`s and 7 `# Panics` sections about caller input. `cluster_census` returns `Result`, but `CensusError` has no variant for `sat_px`, `hi_parallax_deg`, `wilson_z` or `warp_percentile`; the binding comment at `cluster_census.rs:134` says a NaN threshold reports "a clean 0.0 — the one silent failure the census must never produce", which a Rust caller still gets. Separately, `ObservationCoverage((4e9, 4e9) image)` aborted the interpreter with "memory allocation ... failed". [Caller input never panics](../specs/STANDARDS.md#caller-input-never-panics).
- Proposed fix: each core function returns a `Result` whose error enum holds the checks the binding makes now; the binding copies are deleted and mapped with `value_err`; `ObservationCoverage::build` bounds the grid size.
- Effort: medium.
- Risk: low — Rust callers gain a `?`.

> _Status (2026-11-14): **Partially done** — `source_clusters` (`SourceClustersError`), `cluster_radii` / `coarsest_cluster_ids` (new `ClusterRadiiError` variants for the CSR index and the shapes), `estimate_adjacency_surfel_normals` (`AdjacencySurfelNormalsError`), `build_observation_adjacency` (`ObservationAdjacencyError`, which now refuses an out-of-range point or image index rather than ignoring it), `ObservationCoverage::build` and its three batch queries (`ObservationCoverageError`; `build` refuses a grid over `MAX_GRID_CELLS` = 2³⁰ cells and allocates every grid fallibly, so the `(4e9, 4e9)` image is a `ValueError`), `plot_epipolar_curves_batch` (`EpipolarCurvesError`) and `find_point_correspondences_by_coordinate` (`PointCorrespondenceError::KeypointColumnLength`) return a `Result`, and `cluster_census` refuses the four thresholds with `CensusError::InvalidParam`; their binding copies are deleted and mapped with `value_err`. Two allocations sized by caller input went too: `source_clusters` sizes its frame table by the frames, not by `n_images`, and the surfel fit counts occupied sectors without an `n_sectors` table. Still open: `triangulate_batch`, `bearing_score_batch` and `fit_point_and_bearing_batch`, whose binding `csr_offsets` / `batch_rays` checks remain; their single-track siblings `bearing_score` / `fit_point_and_bearing` and `depth_uncertainty_batch` share the same length `assert!`s and have about 40 Rust callers between them, so they are one change of their own. Branch `hyg-fix2-1113-05-analysis-core-input-checks`._

**Use `read_rows` and one CSR check across the analysis bindings**
- Location: row reads at `adjacency_surfel_normals.rs:263,395-406`, `observation_adjacency.rs:149-160`, `observation_coverage.rs:319-330,487-490`, `core.rs:351-360`, `epipolar.rs:172-177`, `triangulation.rs:337-351`; CSR checks at `triangulation.rs:361-387`, `source_clusters.rs:122-135`, `adjacency_surfel_normals.rs:377-387`.
- Problem: `geometry/reconstruction_growth.rs:21` `read_rows::<W>` checks the width and returns `Vec<[f64; W]>`, but only 3 files use it (11 calls); this unit writes the same `map(|c| [c[0], c[1], ..])` 10 times and 16 more copies are elsewhere in the crate. `csr_args::check_starts_nondecreasing` / `check_starts_close` are used by 2 files while this unit has 3 more CSR validators. A camera centre from flat (q, t) is written in `observation_adjacency.rs:162-178`, core `viewing_angle.rs:67-82` and `image_pair_graph.rs:278`, beside `RotQuaternion::camera_center`. (N, k) outputs are built three ways: `from_vec2` (6 sites; `source_clusters.rs:176` says it costs more than the join), flat plus `reshape`, and `rows3`.
- Proposed fix: `read_rows` and a generic `csr_offsets` in `helpers.rs`, called everywhere; poses passed to core as `RigidTransform`, or `camera_center` called; outputs built flat with `reshape`.
- Effort: low.
- Risk: low — some error text changes.

**Take `positions_xyzw` in `compute_narrow_track_mask` and `estimate_adjacency_surfel_normals`**
- Location: binding `core.rs:134-166` with core `viewing_angle.rs:56-121`; binding `adjacency_surfel_normals.rs:331-449` with core `adjacency_surfel_normals.rs:187`; `src/sfmtool/xform/_point_filters.py:84`.
- Problem: both take `(N, 3)` positions; for a `w = 0` row, `recon.positions` gives a direction, which the narrow-track score treats as a position. `_point_filters.py:84` repairs the result afterwards with `keep_infinity_points`, Python making up for the interface. The surfel fit has no `w = 0` path, contract or count (callers: tests and 1 script). [Points at infinity are supported](../specs/STANDARDS.md#points-at-infinity-are-supported).
- Proposed fix: take `positions_xyzw`; keep `w = 0` rows in the narrow mask in core, and report them as undetermined with a count in the surfel fit; delete the Python `keep_infinity_points` wrap.
- Effort: low.
- Risk: low.

> _Status (2026-11-13): **Done** — both cores and bindings take `positions_xyzw` and divide a finite row by its `w`; `compute_narrow_track_mask` keeps every `w = 0` row in core, and `estimate_adjacency_surfel_normals` leaves a selected point at infinity `NaN` and undetermined, counts it in `selected_at_infinity`, and takes a neighbour at infinity's direction as its displacement; `keep_infinity_points` is deleted and `RemoveNarrowTracksFilter` passes `recon.positions_xyzw`, branch `hyg-fix2-1113-03-analysis-xyzw-positions`._

**Bridge a `ProgressCounter` to `track_table` and `covered_by_finer`, and give 13 analysis cores a Progress**
- Location: `track_table.rs:136`, `covered_by_finer.rs:317`, `pick_images.rs:664-690`.
- Problem: both cores take `&Progress` and loop over every point or row, but the bindings pass `Progress::none()` and take no `ProgressCounter`, so their `Cancelled` arms map an error that cannot happen. `pick_images.rs:664-690` is the one binding here that bridges counter, cancellation and signals, with its sink written inline. 13 core functions behind this unit loop over every track or pair with no `Progress` at all, among them `fit_point_and_bearing_batch` (Levenberg-Marquardt per track), `build_frustum_intersection_pairs`, `cell_plane_normals`, `build_observation_adjacency` and `cluster_census`. [Long-running functions take a Progress](../specs/STANDARDS.md#long-running-functions-take-a-progress).
- Proposed fix: move the `pick_images` sink into `py_progress.rs` as the shared bridge (the same helper the patch bindings need); `progress: Option<ProgressCounter>` on both bindings; `&Progress` on the 13 core functions.
- Effort: medium.
- Risk: low.

**Name normal determinacy codes in core**
- Location: binding `analysis/cell_plane_normals.rs:27`, `bench/normal.rs:76-82`; core `NormalDeterminacy::code` at `patch/cell_plane_normals.rs:156`.
- Problem: two bindings spell `"none"`, `"one_axis"`, `"both_axes"`, one relying on the order of `code()` with no check; `CellPlaneStatus::NAMES` (line 186) already shows the shared form.
- Proposed fix: `NormalDeterminacy::NAMES` and `name()` beside `code()`, a test that `NAMES[code()] == name()`, used in both bindings.
- Effort: low.
- Risk: low.

**Rule on the keyword names for one concept across `sfmtool.analysis`**
- Location: the sibling bindings of `crates/sfmtool-py/src/analysis/`.
- Problem: public keywords in one module: per-observation image `track_image_indexes` (4 files), `image_indexes` (2), `image_of_row` (2), `member_images` / `obs_image` (2); pixel positions `keypoints_xy`, `positions_xy`, `xy_px`, `xy`, `pixels`, `points1`, `member_positions`; rotations `quaternions_wxyz` / `cam_from_world_wxyz`; translations `translations` / `translations_xyz`. (`reach_px` and `radius_px` in `covered_by_finer` carry different meanings, so they are not drift.) The glossary has no ruling.
- Proposed fix: decide on the argument (for example `track_image_indexes`, matching the reconstruction getter, against `image_of_row`, the row model of the newer kernels), record it in `specs/GLOSSARY.md`, then rename as clean breaks while the callers are mostly tests.
- Effort: medium.
- Risk: high — public keywords; tests and scripts change.

**Have the point-correspondence bindings take the reconstructions**
- Location: binding `core.rs:258-395`; `src/sfmtool/_point_correspondence.py:120-230`.
- Problem: `_observation_xy_by_image` splits the track columns per image into a dict, and `_vote_point_correspondences.flatten` concatenates them back into the same columns for the binding, which takes eight columns. `find_point_correspondences` is defined three times (the binding, `_point_correspondence`, `merge.correspondences`). [Logic lives in Rust, not in Python wrappers](../specs/STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers).
- Proposed fix: the bindings take the two `SfmrReconstruction`s and the shared pairs and read the keypoints in Rust, as "Have the image-pair-graph bindings take the reconstruction" proposes; rename the `merge.correspondences` one to `group_point_correspondences`.
- Effort: medium.
- Risk: medium — reading `.sift` keypoints in `sift_files` mode moves into Rust.

Not flagged, read in full on 2026-11-13: 17 of 17 license headers; `track_table` cloning an `SfmrReconstruction` measured 0.0002 s against 0.0003 s for an `EditedReconstruction` on kerry_park v02 (387 points); docstring overlap with core (shared 8-word phrases: `covered_by_finer` 114 of 397 words, `source_clusters` 71, `keypoint_reach` 59, `cell_plane_normals` 32, `track_table` 4) is behaviour reference a REPL user needs; `point_or_bearing.rs` (537 lines) and `pick_images.rs` (726 lines, with a correct progress bridge) each hold one concern; the `_py` suffix on 10 of 27 Rust function names is internal only; 0 capitalized binding errors; no medians, quantiles or pixel loops; `observation_adjacency` guards positions with `point_is_at_infinity`; the `(n, 2)` shape checks in `keypoint_reach` / `covered_by_finer` must stay in the binding. `analysis/mod.rs` is one of 10 `mod.rs` files among the 11 module directories of `sfmtool-py` (only `bench.rs` is beside its directory), counted under "Convert the remaining `mod.rs` files"; error closures: 15 inline `PyValueError::new_err(e.to_string())` against 6 `map_err(value_err)` here, 132 inline crate-wide.

### `crates/sfmtool-matches-format/src (rest)` (read in full)

All seven files (`write.rs`, `read.rs`, `types.rs`, `select.rs`, `cells.rs`, `entries.rs`, `lib.rs`, 4,098 lines) were read in full. `verify.rs` was read at lines 534-620, 766-830 and 1186-1262 to confirm the duplication.

**Validate a `.matches` file's structure when it is read**
- Location: `crates/sfmtool-matches-format/src/read.rs:123-302`; panics at `select.rs:229-240, 293, 304, 313, 318, 462-468`.
- Problem: `read_matches` checks the version, entry shapes, `config_indexes` and the cells, but none of what `write_matches` checks: pair ordering and bounds, `cluster_starts` as a CSR array ending at the member count, `member_images` against `image_count`, member features against each image's feature count, the reference and kept rules, and inliers being a subset of the candidates. `select_clusters` and `cluster_worst_consistency` then index member arrays with those values. Probe: a valid 5-member cluster file re-zipped with `cluster_starts = [0,3,9]` opens without error, and `MatchesFile(f).select_clusters()` raises `PanicException` "index out of bounds: the len is 5 but the index is 5" at `select.rs:238`; with `member_images[4] = 7` and `restrict_images=[...]` it panics at `select.rs:240`. A corrupt file is caller input: [Caller input never panics](../specs/STANDARDS.md#caller-input-never-panics).
- Proposed fix: run the shared structural validation from the next finding at the end of `read_matches` (O(members)).
- Effort: low once the next finding is done.
- Risk: low — files that break the write rules stop reading, which is the intent.

> _Status (2026-11-13): **Done** — `read_matches` ends by running the shared content checks (`validate::content_errors`) and refuses a file that breaks one with the first finding as `InvalidFormat`, so both probes (`cluster_starts = [0,3,9]` over 5 members, `member_images[4] = 7` of 3 images) now fail to open, as `OSError` in Python, instead of reaching `select_clusters`; read gains the pair order and bounds, match and inlier count sums, feature bounds, inliers among their candidates, `image_dims` above 0, CSR, member bounds, member `NaN` and reference / kept checks. The checks are linear: on a 68M-match, 67M-inlier flow file they add 0.12 s to a 1.86 s read; branch `hyg-fix-1113-03-matches-shared-validation`._

**Write the `.matches` structural checks once, for write, read and verify**
- Location: `write.rs:646-1298` (`validate_structure`, `validate_dimensions`, `validate_*_constraints`, 652 lines) and `verify.rs:482-1262` (`check_pair_ordering`, `check_match_counts`, `check_match_feature_bounds`, `check_clusters_structure`, `check_cluster_references`).
- Problem: 36 shared long lines; one copy works on ndarrays and stops at the first error, the other on raw bytes and collects every error. They have drifted: the inliers-are-candidates rule (`write.rs:1110-1140`) has no counterpart in `verify.rs`, so `verify_matches` passes a file `write_matches` refuses; "sum of match_counts" (`write.rs:810`) against "Sum of match_counts" (`verify.rs:592`); the `match_counts < 1` message explains itself in write and not in verify. Validation is also the second of `write.rs`'s two concerns (streaming at 138-644, validation at 646-1298).
- Proposed fix: one `validate.rs` of slice-based checks returning `Vec<String>`; `write_matches` and `read_matches` take the first error, `verify_matches` all of them, as `cells.rs` already does with `not_kept_with_readings` and `cell_entry_errors`. Move `check_backbone_counts` (`types.rs:167-194`) there, and replace `HashMap<_, ()>` at `write.rs:1119` with `HashSet`. `write.rs` drops to about 650 lines.
- Effort: medium.
- Risk: medium — error order and message text pinned by tests change.

> _Status (2026-11-13): **Done** — new `validate.rs` holds slice-based `pair_errors`, `inlier_errors`, `config_index_errors`, `image_dims_errors`, `cluster_errors` and `cluster_patch_errors` that push every finding, plus `content_errors` over a `MatchesData`, `check_backbone_counts` (moved from `types.rs`) and the writer-only `validate_structure` / `validate_dimensions`; `write_matches` and `read_matches` take the first finding, `verify_matches` all of them. `verify_matches` gains the inliers-are-candidates rule, with byte-length checks on the two inlier arrays it now walks; the merged messages read "sum of match_counts", "sum of inlier_counts" and "match_counts contains values < 1 (pairs with zero matches should not be stored)" everywhere. The candidate set is a sorted vector rather than a `HashMap<_, ()>`, after a merge-walk that settles inliers kept in candidate order. `write.rs` is 646 lines; branch `hyg-fix-1113-03-matches-shared-validation`._

**Write caller arrays through `as_standard_layout` in the format writers**
- Location: `crates/sfmtool-matches-format/src/write.rs` (24 sites: 209, 234, 280, 289, 298, 307, 346, 360, 369, 378, 388, 467, 476, 485, 494, 523, 551, 570, 579, 588, 597, 606, 624, 633); the same pattern 39 times in `sfmtool-sfmr-format/src/write.rs`, and in `write_sift`.
- Problem: `.as_slice().unwrap()` on public `MatchesData` arrays panics on a Fortran-order array (for example from `reversed_axes()`) or a strided owned one. The same function calls `as_standard_layout()` for its cell columns only (`write.rs:429-457`), as camrig-format `write.rs:92,103` does. Python cannot reach it (the binding normalizes layout at `fileio/matches.rs:208-378`); the Rust API can. [Caller input never panics](../specs/STANDARDS.md#caller-input-never-panics).
- Proposed fix: a local `write_array_hashed(zip, name, &ArrayBase<S, D>, level, hasher)` that calls `as_standard_layout()` then `cast_slice`, used at all 28 sites; the same in sfmr-format and sift-format. It also removes most of the 500 lines of repeated `write_binary_entry_hashed(...)` calls.
- Effort: low.
- Risk: low.

> _Status (2026-11-13): **Done** — `sfmtool-archive-io` gains `row_major_bytes` and `row_major_elements`, which borrow a standard-layout array and copy any other in row-major order; the `.matches`, `.sfmr`, `.sift` and `.camrig` writers and the `.matches` cluster selection (`select_clusters`, `cluster_worst_consistency`) read every caller array through them, and tests write Fortran-order and strided values to the same section hashes as their standard-layout twins. One shared helper rather than a `write_array_hashed` per crate, because the `.sfmr` writer writes through its `EntrySink` and the `.sift` writer through `SectionDigests`, so only the array-to-bytes step is common; the `write_binary_entry_hashed(...)` call boilerplate is left as it is. This also covers the writer item of "Panics reachable from Python in the sweep bindings and from Rust callers of the solvers and writers", branch `hyg-fix-1113-10-writer-standard-layout`._

**Give each `entries.rs` dimension function one parameter type**
- Location: `entries.rs:17-28, 196` (the same rule in sfmr-format `entries.rs`).
- Problem: the module says dimension parameters are `impl Display`, then "The one exception is [`clusters_cluster_starts`], which takes a concrete `usize`"; the other 34 take `impl Display`. Breaks [A convention has no documented exceptions](../specs/STANDARDS.md#a-convention-has-no-documented-exceptions).
- Proposed fix: either all `usize` (verify casts its `u32` counts), or `clusters_cluster_starts(entry_len: impl Display)` with callers passing `n + 1`; then delete the exception paragraph. Fix sfmr-format the same way.
- Effort: low.
- Risk: low.

**State the entry-naming rationale once, not in each format crate's `entries.rs`**
- Location: matches `entries.rs:4-45` and sfmr-format `entries.rs:4-50`.
- Problem: 33 identical lines (the "single source of truth" argument, the `impl Display` rationale, "How the tests use this module").
- Proposed fix: state it once in the `sfmtool-archive-io` crate doc or `specs/formats/README.md`; each `entries.rs` keeps a 3-line pointer plus its own part (sfmr's rename table).
- Effort: low.
- Risk: low.

**Move the legend code out of `types.rs` and the accessors out of `select.rs`**
- Location: `types.rs:567-721` (`parse_legend`, `legend_names`, `normalize_codes`, `read_member_status_legend`, 155 lines), `types.rs:54-86`, `select.rs:434-493`.
- Problem: `types.rs` (867 lines) holds read and verify logic for the status legend, while the parallel cell legend is in `cells.rs:286-327`. The 34-line doc on `MATCHES_FORMAT_VERSION` restates versions 7-10, which the format spec's "Versioning and Migration" holds. `select.rs` holds four accessors unrelated to selection (`member_positions`, `member_affine_shapes`, `cluster_worst_consistency`, `ClusterPatchData::refine_radius`).
- Proposed fix: both legends and their version constants in `legend.rs`; the version doc cut to the current version and a link to the spec; the accessors beside `shared_image_dims` or in `access.rs`.
- Effort: low.
- Risk: low.

**Derive `Default` for `MatchesContentHash`**
- Location: `write.rs:116-124`, `select.rs:406-414`.
- Problem: the same 9-line all-empty hash is built twice.
- Proposed fix: `#[derive(Default)]` and `MatchesContentHash::default()`.
- Effort: low.
- Risk: low.

Not flagged, read in full on 2026-11-13: the `expect`s at `write.rs:230,356,384,1269` follow `validate_dimensions`, at `read.rs:309-368` follow `check_backbone_counts`, and at `select.rs:95,101,234,278` are internal facts; `select.rs:189-191,462` ("contiguous") act on owned arrays that only a Rust caller using `slice_move` with a step could make strided; `read.rs:568` acts on an array it built; counts are `u32` widened to `usize` with `u64` sums; indexing in `write.rs:1061-1295` always follows a check; the I/O shape is complete (`read_matches`, `read_matches_metadata`, `write_matches`, `verify_matches`, `MatchesFile`); 7 of 7 license headers; no `mod.rs`; no settings structs; `indexes` throughout, 0 `indices`; `s_conjugate_relative_pose` (`types.rs:97-108`) is a copy forced by the crate graph. Capitalized messages ("Unknown TwoViewGeometryConfig", verify's "Sum of …") are the older format-crate style the glossary has not ruled on; see "Rule on error-message capitalization".

### `crates/sfmtool-py/src/patches` (read in full)

All 16 files (4,369 lines) were read in full, with Python probes against the rebuilt extension on `seoul_bull_sculpture_ground_truth_v02.sfmr` and blank images.

**Resolve patch views and keypoints in one place for the four kernel bindings**
- Location: `crates/sfmtool-py/src/patches/localize_keypoints.rs:209-252,260-296,320-345`, `refine_keypoints.rs:202-237,259-293,362-378`, `member_coherence.rs:241-304`, `select_views.rs:139-204`; `Vec<ProjectedImage>` also built at `render_bitmaps.rs:119,213` and `spawn.rs:162`.
- Problem:
  - localize and refine share the override and `point_indexes` mask block (about 28 lines), the `patch_keypoints` validation (about 32 lines, the same three error messages) and the result dict (16 lines). member_coherence and select_views share 41 lines that differ only in names. The loop that skips points outside the mask is written 4 times.
  - `Vec<ProjectedImage>` is assembled by hand 7 times, while `bench.rs:1622 views_of` already does it, privately.
  - Each patch view's stored keypoint is found three ways: a position re-key at `member_coherence.rs:272-291` and `select_views.rs:172-191`, and a `HashMap<(pid, img)>` at `refine_keypoints.rs:300-313`.
- Proposed fix: `resolve_patch_views` and `resolve_patch_keypoints` in `patches/args.rs` beside `reference_positions`, returning the per-patch lists, the seeds and the mask; `views_of` moved to `views.rs` as `PosedViews::projected(&PyramidSet)`; a core `patch::stored_keypoints_for_views(recon, cloud, patch_views)` in `patch/track_views.rs` beside `track_keypoints_from_reconstruction`, used in all three places.
- Effort: medium.
- Risk: low — the error messages are pinned by tests and stay as they are.

**Call the core `refine_patch_cloud_keypoints` from the `refine_keypoints` binding**
- Location: `refine_keypoints.rs:315-360`.
- Problem: the binding runs its own `par_iter` with profiler `reset`/`report` and a per-patch `fetch_add`, instead of `refine_patch_cloud_keypoints` (`keypoint_subpixel.rs:900`), which takes `&Progress`. Its comment gives lazy seed construction as the reason. This is a fifth copy of the skeleton `patch::run_over_cloud` targets, and it cannot be cancelled. [Logic lives in Rust, not in Python wrappers](../specs/STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers), [Long-running functions take a Progress](../specs/STANDARDS.md#long-running-functions-take-a-progress).
- Proposed fix: build the seeds with `stored_keypoints_for_views` (above) and call `refine_patch_cloud_keypoints(..., &progress)`.
- Effort: low.
- Risk: low — peak memory rises by one `Option<[f64;2]>` per patch view.

**Let Python cancel the patch kernels, with one progress interface in the core drivers**
- Location: bindings `localize_keypoints.rs:305-317`, `member_coherence.rs:306-317`, `select_views.rs:206-217`, `render_bitmaps.rs:128-145,222-235`, `cloud.rs:83-92,285`, `spawn.rs:170`, `consensus_atlas.rs:110`, `photometric_ransac.rs:153`; core drivers `view_selection.rs:1238`, `keypoint_localize.rs:837`, `member_coherence.rs:561`, `observation_reading.rs:200`, `stored_bitmap.rs:309`.
- Problem:
  - Five binding sites pass the `ProgressCounter` atomic as `done` with `Progress::none()`, so Python sees the count but cannot cancel.
  - The core drivers they call (`select_patch_cloud_views`, `localize_patch_cloud_keypoints`, `validate_patch_cloud_member_coherence`, `render_patch_cloud_bitmaps`, `read_cloud_observations`) take both `done: Option<&AtomicUsize>` and `&Progress`; `refine_patch_cloud_keypoints` takes only `&Progress`.
  - `PatchCloud.from_reconstruction` reads every `.sift` file under `extent="feature_size"`, and it and `from_tracks` run with `Progress::none()` and no `progress` argument; `cloud.rs:83-85` justifies this with "the Python signature is the same one it has always had".
  - Core `spawn_candidate_tracks` takes no `Progress` and calls two drivers with `none()` (`spawn.rs:233,282`). The RANSAC and atlas bindings take no progress argument.
  - Related pairs found by check 11: `refine_patch_keypoints` / `refine_patch_keypoints_reporting` (`keypoint_subpixel.rs:156/179`), `member_zncc_matrix` / `member_zncc_matrix_reporting` (`member_coherence/matrix.rs:81/104`), `localize_patch_keypoints` forwarding to `try_localize_patch_keypoints(…, &Progress::none())` (`keypoint_localize.rs:709`).
- Proposed fix: one `ProgressCounter` sink helper in `py_progress.rs` following `bench/refine_normals.rs:197-204`, and `&Progress::to(&sink)` at every site; `done` removed from the five core drivers and from `PatchCounter`; each `_reporting` pair merged into one function taking `&Progress`; `progress=None` on `from_reconstruction`, `from_tracks` and `spawn_candidate_tracks`; the comment at `cloud.rs:83-85` deleted. [Long-running functions take a Progress](../specs/STANDARDS.md#long-running-functions-take-a-progress), [One interface, no progress variants](../specs/STANDARDS.md#one-interface-no-progress-variants).
- Effort: medium.
- Risk: low — the Python keywords stay.

**Read alpha the same way in the two tile bindings**
- Location: `self_similarity.rs:80-127,225-277`, `blur_matched.rs:25-55`.
- Problem: `blur_matched` reads an `(R,R,2)` tile as grey plus alpha; `zncc_self_similarity_parts` goes through `PatchTile::planes_from_interleaved`, which keeps `min(C,3)` colour planes and reads alpha only when `C >= 4`, so a 2-channel tile's alpha is correlated as a colour. Probe: a 15×15 grey tile with alpha 0 on 7 of 15 columns gives radius 0.0762 as `(R,R,2)` and 0.0423 as `(R,R,4)` with the same grey in each colour channel. The shape checks are written three times (`check_tile`, `self_similarity.rs:102-111`, `:244-253`) and the u8/f32 extraction twice.
- Proposed fix: one core function that turns an interleaved tile into colour planes plus a data mask under one alpha rule (2 or 4 channels means the last is alpha), returning an error on a bad shape; both bindings call it, and the caller-input `# Panics` at core `self_similarity.rs:140,179` goes. [Caller input never panics](../specs/STANDARDS.md#caller-input-never-panics).
- Effort: low.
- Risk: medium — the radii of 2-channel inputs change.

> _Status (2026-11-13): **Done** — core `InterleavedTile::new` splits an interleaved tile into colour planes and data flags by one rule (two channels grey and alpha, four RGB and alpha, `valid` flags combined with alpha) and returns `InterleavedTileError` for a bad shape; `planes_from_interleaved` and `data_from_interleaved`, with their `# Panics`, are gone. Both bindings read tiles through it via shared `check_square_tile` / `square_tile` helpers in `patches/args.rs`, and the u8/f32 extraction is one `grey_levels` helper. The bench, the reference-view rule and cluster refinement's member gate use it too, and `render_view_tile` makes the second of two channels opaque alpha to match. A 2-channel tile now reads the same radius as the 4-channel tile with its grey in each colour channel (core and binding tests), branch `hyg-fix2-1113-02-tile-alpha-rule`._

**Move the spherical-tile bindings to `sfmtool.spherical`**
- Location: `consensus_atlas.rs`, `photometric_ransac.rs` (322 lines), registered at `patches/mod.rs:43-51`.
- Problem: both bind `sfmtool_core::spherical::*`, take `PySphericalTileRig` / `PyPerSphericalTileSourceStack` from `crate::spherical`, and are specified under `specs/core/spherical/`, yet are exported as `sfmtool.patches.*` (callers: `src/sfmtool/rig/panorama.py:24` and two tests in `tests/patch/`). `bool_array` (`photometric_ransac.rs:87-96`) goes through a u8 view as there is "no direct bool ctor", while `blur_matched.rs:444` returns `Array1<bool>` directly.
- Proposed fix: move both files to `sfmtool-py/src/spherical/` and register them in `sfmtool.spherical` as a clean break, updating the three callers; return `PyArray1<bool>` directly; convert `patches/mod.rs` to `patches.rs` in the same change ([A Rust module with submodules is a file beside its directory](../specs/STANDARDS.md#a-rust-module-with-submodules-is-a-file-beside-its-directory)).
- Effort: low.
- Risk: medium — the public import path changes.

**Validate patch-cloud, spawn and render input in core, not only in the bindings**
- Location: binding `cloud.rs:180-266` with core `PatchCloud::from_tracks` (`patch/cloud.rs:1070`); binding `spawn.rs:98-145` with core `spawn.rs:181-205`; binding `render_bitmaps.rs:106,204` with core `stored_bitmap.rs:314` and `observation_reading.rs:205`.
- Problem: the bindings check observation grouping, index ranges and array lengths; core `from_tracks` indexes `obs_offsets` unchecked; core spawn asserts lengths, indexes `cloud.patch(p)` unchecked and does not refuse parents at infinity; the two render cores `.expect("needs a reconstruction with inline keypoints")`. Python cannot reach these; Rust callers can. [Caller input never panics](../specs/STANDARDS.md#caller-input-never-panics).
- Proposed fix: the checks move into core and return error variants (`PatchCloudError`, a new `SpawnError`, a keypoints-missing error); each binding maps the error.
- Effort: medium.
- Risk: low.

> _Status (2026-11-13): **Done** — core `PatchCloud::from_tracks` checks its array lengths, offsets and image indexes (`PatchCloudError::InputLength`, `ObservationOffsets`, `ImageIndexOutOfRange`); core `spawn_candidate_tracks` returns a new `SpawnError` for mismatched lengths, an out-of-range or infinity parent and an out-of-range patch view; `render_patch_cloud_bitmaps` and `read_cloud_observations` return a new `CloudRenderError` for missing inline keypoints, missing or out-of-range point indexes and a view count that is not one per image. The bindings raise these as `ValueError` and drop the checks core now makes; the spawn and render bindings run core's public `check_spawn_requests` and `check_cloud_render` before building the image pyramids, branch `hyg-fix2-1113-04-patch-core-input-checks`._

**Raise the method's own error for a CameraViews in `render_bitmaps` and `read_observations`**
- Location: `render_bitmaps.rs:101-105,199-203`.
- Problem: the probe raises "patch_views is required when the first argument is a CameraViews"; neither method takes `patch_views`, and their own message ("needs a reconstruction, not a CameraViews") cannot be reached.
- Proposed fix: call `resolve_scene` and the point-range check without the explicit-views guard.
- Effort: low.
- Risk: low.

> _Status (2026-11-13): **Done** — `views.rs` has a new `resolve_patch_reconstruction`, which raises "<method> needs a reconstruction, not a CameraViews" for a `CameraViews` and otherwise runs the same point-index checks as `resolve_patch_scene` (both now through a shared `check_cloud_points`); `render_bitmaps` and `read_observations` call it instead of `resolve_patch_scene` with the `patch_views` guard, and `tests/rust_bindings/patches/test_camera_views_rust_bindings.py` asserts each method's own message, branch `hyg-fix-1113-09-camera-views-error`._

**Take `centers_xyzw` in `PatchCloud.from_halfvec_arrays`**
- Location: `cloud.rs:304-337`.
- Problem: `centers` is `(N,3)` and core builds every patch with `w = 1` (its doc says the caller sets `w` afterwards; Python has no setter). `_patch_compaction.py:275` passes `recon.positions`, which holds directions for `w = 0` rows. [Points at infinity are supported](../specs/STANDARDS.md#points-at-infinity-are-supported).
- Proposed fix: take `centers_xyzw (N,4)` and set `w` in core `from_halfvec_arrays`; 3 callers.
- Effort: low.
- Risk: low.

> _Status (2026-11-13): **Done** — core `PatchCloud::from_halfvec_arrays` and the binding take `centers_xyzw` `(N, 4)`, give each `w = 0` row a patch at infinity on its unit direction, and return an error (`ValueError`) for wrong shapes, non-finite values and a zero direction; `_patch_compaction.py` and the test callers pass `positions_xyzw`, branch `hyg-fix-1113-08-halfvec-centers-xyzw`._

**Bring the sibling patch bindings' names and docs into line**
- Location: as listed.
- Problem: docstrings still say "view set" (`localize_keypoints.rs:111`, `refine_keypoints.rs:101`), which the glossary's "Per-patch view lists" entry retired, and the locals `sets` (localize 222-310, refine 210-284, `args.rs:130`) and the comment at `refine_keypoints.rs:246` use it; `select_views.rs:81` documents `admitted` as int32 where it is uint32; `progress` is missing from the Args of `member_coherence` and `select_views`; `localize_keypoints.rs:93` says "exactly today's behaviour"; "image indices" at `member_coherence.rs:143`, `select_views.rs:82`; `cloud.rs:33` says "per finite 3D point" though infinity patches are built by default; `localize` accepts `None` seeds while `refine` raises `TypeError` for them though its core takes `Option`; five inline `PyValueError::new_err(e.to_string())` closures (`cloud.rs:93,287`, `member_coherence.rs:366,370,379`) beside `helpers::value_err`, the three in member_coherence mapping an internal shape invariant to `ValueError`.
- Proposed fix: rename `sets` to `patch_views`; correct the docstrings; have `refine` take `Vec<Option<[f64;2]>>`; use `value_err`, and `expect` for the invariant.
- Effort: low.
- Risk: low.

**Cut the design argument from the `validate_member_coherence` docstring**
- Location: `member_coherence.rs:41-43,58-89` (a 145-line block); `localize_keypoints.rs:96-103`.
- Problem: the `self_bar_k` (18 lines) and `exoneration_ratio` (14 lines) entries argue the design (occluder ZNCC 0.85-0.95, "structural vs spectral"); the binding shares 92 eight-word phrases with `specs/core/patch/member-coherence-validation.md`.
- Proposed fix: keep each argument's meaning, default and what `0` does, and point to the spec section for the reasoning.
- Effort: low.
- Risk: low.

Not flagged, read in full on 2026-11-13: `oriented_patch.rs` and `view_tile.rs` (resolution guard 2..=1024; a zero-axis patch renders with coverage 0); no `PanicException` from probes with resolution 0 or 1, `search` NaN, -1 or 1e7, `window_sigma=0`, spawn with `search=NaN`, mismatched `from_halfvec_arrays`, or an empty self-similarity stack; doc overlap with core doc comments is small (member coherence 30 of 1,347 eight-word phrases, localize 13 of 1,027, blur_matched 0 of 1,079); 16 of 16 license headers; lowercase errors; every file 464 lines or fewer and one concern (`render_bitmaps.rs` also holds `read_observations`, which `mod.rs:10-13` does not list); `parse_patch_window` is no longer inlined (0 inline parsers outside `args.rs`, 9 files use the helpers); `consensus_atlas.rs` and `photometric_ransac.rs` use `value_err`.

### `crates/sfmtool-core/src/bench/nearby/find` (read in full)

`find.rs` is 1,111 lines on one concern: the types (about 430 lines), the errors (about 180), `find_nearby_tracks` (281) and its helpers (about 160). Only the error enums and query checks of its siblings `far_field.rs`, `candidate.rs`, `points.rs`, `layers.rs`, `range.rs` and `nearby.rs` were read, where a finding needed them.

**One error type for the bench queries' shared refusals**
- Location: `crates/sfmtool-core/src/bench/nearby/find.rs:433-611`, `nearby/candidate.rs:143-211`, `nearby/far_field.rs:290-344`, `bench/track_at_pixel.rs:686-773`, `nearby/range.rs:82-101`, `nearby/layers.rs:270-300`.
- Problem:
  - Four enums (`NearbyTracksError`, `NearbySourceError`, `FarFieldError`, `TrackAtPixelError`) each declare `NoSuchImage`, `PixelOffImage` and `InputMismatch` with the same Display text. `DistanceRangeError` and `DepthLayerError` also declare `NoSuchImage`.
  - "is not one of the reconstruction's {image_count} images" occurs 6 times, and "but the reconstruction has {image_count} images" 6 times.
  - `find.rs:522-611` is 90 lines of four `From` impls that copy each field from one enum's variant into the identical variant of another.
  - `TrackAtPixelError::stage()` already groups the three variants under the name "query".
- Proposed fix: add one `bench::QueryError { NoSuchImage, PixelOffImage, InputMismatch }` with a single Display, and give each enum a `Query(QueryError)` variant (`DepthLayerError::GreyMismatch` becomes `InputMismatch { input: "grey" }`). The `From` impls shrink to one line per variant. Do it together with "Fold the bench error enums" below.
- Effort: low to medium.
- Risk: low — matches on `NearbyTracksError::NoSuchImage {..}` in `find_tests.rs:449` and `layer_tests.rs:344` change; the viewer matches only `Cancelled`.

**Call the existing `check_query` instead of repeating it in three entry points**
- Location: `find.rs:683-719`, `far_field.rs:378-402`, `track_at_pixel.rs:798-830`; the helper is `candidate.rs:221-252`; the on-photograph test is `steps.rs:2895`.
- Problem:
  - `candidate::check_query` does the input-length loop, the image lookup and the on-photograph test, and the four matching sources call it. `find_nearby_tracks`, `far_field_sweep` and `build_track_at_pixel` each repeat it inline, about 30 lines each.
  - The half-open test `x >= 0 && y >= 0 && x < w && y < h` is written inline in 6 places (`find.rs:708`, `far_field.rs:392`, `candidate.rs:241`, `track_at_pixel.rs:819`, `evaluate.rs:1703`, viewer `mcp/input.rs:929`) beside the private function `steps::on_photograph`, whose doc ties it to `Unmeasured::OffSensor`.
  - The `pixel.iter().all(is_finite)` guard in the four bench copies has no effect: NaN and ±inf already fail the comparisons.
- Proposed fix: make `steps::on_photograph` public beside `clamp_to_photograph` (keep that name; it already means the sensor's half-open extent). Move `check_query` to bench level returning `QueryError`, and call it from the three entry points and `evaluate.rs`. The viewer's two `clamp_to_photograph(...).0.is_some()` checks and `mcp/input.rs` call the same function.
- Effort: low.
- Risk: low — the pixel refusals are pinned by tests and their messages do not change.

**One `image_stem` for the cluster-seed label**
- Location: `find.rs:640-646`, `track_at_pixel.rs:943-949`, `constellation.rs:152-155`, `far_field.rs:868-871`, viewer `bench/track_at_pixel.rs:294-296`.
- Problem: the same 4-line body (`Path::new(name).file_stem()`, falling back to the whole name) is written 5 times, and every copy feeds `ClusterSeed`'s label.
- Proposed fix: one `pub fn image_stem(edited: &EditedReconstruction, image: u32) -> String` beside `ClusterSeed` in `steps.rs`, called from all five places.
- Effort: low.
- Risk: low.

**Find Nearby Tracks does not see a point at infinity already in the reconstruction**
- Location: `find.rs:987-1021` (`mark_duplicates`), fed by `nearby/points.rs:78` and `candidate.rs:74` (`position: Vector3<f64>`).
- Problem: `mark_duplicates` compares built tracks only with tracks from the points source, and the points source skips `o.at_infinity` points without counting them. When the far-field sweep rebuilds an existing `w = 0` point at the queried pixel, that track is labelled new and goes on the bench, and a commit writes a second copy of the point. `NearbyCandidate` cannot hold a direction. To confirm: a reconstruction with one `w = 0` point observed at the queried pixel and `far_field_when: Always`. Breaks [Points at infinity are supported](../specs/STANDARDS.md#points-at-infinity-are-supported).
- Proposed fix: have the points source emit bearings (`[f64; 4]` xyzw, or `at_infinity` with a direction) and report how many it skipped, and add `ViewCamera::ray_distance(pixel, xyzw)` returning `INFINITY` for `w = 0`, which also replaces the copied formula at `find.rs:816-817` and `candidate.rs:113-115` and removes the `# Panics` section on `candidate.rs`'s `ray_distance`.
- Effort: medium.
- Risk: medium — candidate fields change shape for Python and the viewer.

> _Status (2026-11-13): **Done** — `nearby_points` keeps a `w = 0` point as a bearing: `NearbyCandidate` gains `at_infinity`, with `position` its unit direction, `depth` infinite and `max_ray_angle_deg` zero, as `FarFieldReading` has them, and its errors are measured at `w = 0`. A new `candidate::ray_distance` returns infinity for a bearing and replaces the formula `find.rs` copied for far-field readings. `mark_duplicates` is unchanged: the sweep's track on the point's observations now repeats the point (test `a_far_field_reading_on_an_existing_point_at_infinity_repeats_the_point`). The Python candidate dicts gain `w`, and the harness's reference `from_tracks` keeps bearings the same way. The skip count was not added: with bearings kept, what the source still drops are its filters. Branch `hyg-fix2-1113-01-nearby-infinity-duplicates`._

**Pass the query's Progress to the nearby sources, the depth layers and each track's fit**
- Location: `find.rs:748` (`run_source`), `find.rs:783,857` (`depth_layers`), `find.rs:896-910` (`bench_track` → `upgrade_sightings`, which calls `set_stage(.., &Progress::none())` at `finish.rs:276`).
- Problem: `guided_matches` (a loop over every camera), `constellation_seeds` and `depth_layers` with evidence (layers × samples × every image) take no `&Progress`. `find.rs` opens a phase around each but cannot report inside or cancel them, and each bench track's fit runs under `Progress::none()`, so cancellation is checked once per track before it starts. No timings were taken; the standard counts a loop over every image as long-running. Breaks [Long-running functions take a Progress](../specs/STANDARDS.md#long-running-functions-take-a-progress).
- Proposed fix: add `&Progress` to the four source functions and to `depth_layers`, and pass `progress` from `find.rs`, with a `split` of it to `upgrade_sightings`.
- Effort: medium.
- Risk: low — the Python and viewer call sites pass progress through.

**Sort depth-layer members with `total_cmp`**
- Location: `find.rs:871-876`.
- Problem: the only `partial_cmp(..).unwrap_or(Ordering::Equal)` sort in `bench/`; the other 24 sorts there use `total_cmp`. A NaN `distance_px` makes the order depend on the input order.
- Proposed fix: `members.sort_by(|&a, &b| tracks[a].distance_px().total_cmp(&tracks[b].distance_px()))`.
- Effort: low.
- Risk: low.

> _Status (2026-11-13): **Done** — the sort now uses `total_cmp`, as proposed. This is a consistency fix only: `distance_px` cannot be NaN or -0.0, because it is `sqrt(dx² + dy²)` or `hypot` of a sighting pixel that each source picked with a `<= radius` test or `in_frame`, and both reject NaN. For finite non-negative values the two comparisons give the same order. The finding's premise that this was the only such sort in `bench/` is wrong: `nearby/layers.rs:403-407` (`group_layers`), `:523` (`best3`) and `:636-640` (the layer ranking) and `steps.rs:3091-3095` (the painting order) also sort with `partial_cmp(..).unwrap_or(Ordering::Equal)`, and `stage.rs:437-441` and `:594-598` are two `max_by` comparisons written the same way. Those six are still open, branch `hyg-fix-1113-11-depth-layer-total-cmp`._

> _Status (2026-11-14): **Done** — the six other comparisons use `total_cmp`, and no other `partial_cmp` is left in `bench/`. All six are consistency fixes for what the finders give, since no NaN or -0.0 from them reaches the six: `best3` in `nearby/layers.rs` sorts ZNCC reads, which are finite and never -0.0; the painting order in `steps.rs` sorts scores that `finite` filtered or `-inf`; the two `max_by` in `stage.rs` compare `sqrt(|det|)` of a determinant checked finite and nonzero; and the grouping and the layer ranking in `nearby/layers.rs` see only what `depth_layers` now accepts, where a usable range has no NaN end and a key comes from a finite `distance_px` of at least 0. The `depth_layers` binding took each anchor's `range`, `bounded`, `far` and `distance_px` from a Python dict unchecked, so a usable anchor with a NaN near end reached the grouping, and a `distance_px` below -1 gave a NaN key (`ln_1p` of it) that ranked its layer first; in both, a NaN also made the other entries' order depend on where it stood. Core `depth_layers` now refuses a `distance_px` that is not finite and at least 0 (`DepthLayerError::BadDistance`) and a usable candidate with a NaN range end (`NanRange`), values no finder gives, which the binding raises as `ValueError`; `NearbyTracksError::Layers` carries them, `the_layers_refuse_a_candidate_no_finder_gives` and the binding test check them, and `specs/core/bench/depth-layers.md` and `nearby-tracks.md` list them. A -0.0 near end from that dict still reaches the grouping and now sorts before a 0.0 one rather than in the order given, branch `hyg-fix2-1113-07-bench-total-cmp`._

Not flagged, read in full on 2026-11-13: 191 doc lines and 11 plain comments in 1,111 lines, all API reference; glossary words correct ("SIFT index" at 208 and 219, no "surfel", "world units" or "descriptor index"); the `expect`s at 412, 633 and 1051-1066 and the `unreachable!` at 1071 guard invariants the function sets up itself; no medians or quantiles; no pixel loops. `group_count` (1077-1095) runs the whole `depth_layers`, including the O(n²) `support_counts`, only to count groups, and could call `group_layers` directly (minor). Only one commit (#1006, a rename) touched `find.rs` since 2026-10-08.

## Findings from the mechanical checks

Each finding here comes from a kept hit of a mechanical check, confirmed by reading the hit and its surrounding function, in units that were not read in full this run. A unit's other code was not read.

**Compute the hand-written medians and percentiles with `sfmtool_core::numeric`**
- Location (12 sites, check 18):
  - Upper-middle median, a different answer from `numeric::median_in_place` on an even count: `crates/sfmtool-core/src/geometry/bundle_adjust.rs:1725` (`in_front_floor`), `analysis/infinity/convert.rs:531`, `features/feature_match/polar.rs:138` (also panics on an empty population), `patch/cloud.rs:1334` (also panics on NaN through `partial_cmp().unwrap()`).
  - Inline median with the same answer: `analysis/reprojection_noise.rs:250` (`robust_spread`), `features/cluster_match/covisibility/selection.rs:99`.
  - Median sorted with `partial_cmp().unwrap_or(Equal)`, unspecified with a NaN present: `spherical/per_tile_source_stack.rs:711`, `spherical/photometric_ransac.rs:820` (`mad`, two medians).
  - Hand-written percentiles: `analysis/scene_scale.rs:242` (`compute_scene_bounds`, public: per-axis upper-middle median, 80th percentile at index `n*4/5`, panics on NaN), `analysis/alignment/reconstructions.rs:518` (95th by index, no interpolation), `geometry/focal_vote.rs:414` (`log_iqr`, its own interpolating closure), `crates/sfm-explorer/src/mcp/render.rs:597` (median and p95 by index).
- Problem: `cargo test -p sfmtool-core --lib numeric` passes (26 tests), because its name scan cannot see these shapes; the grep patterns found 11 `select_nth_unstable`, 15 `[n/2]`, 7 fraction-to-index and 1 interpolation-position hits. Breaks [Medians and quantiles come from one module](../specs/STANDARDS.md#medians-and-quantiles-come-from-one-module).
- Proposed fix: call `median_in_place` / `quantile_of_sorted` at each site, and extend the scan in `numeric.rs`'s test to the `select_nth_unstable` and `[len / 2]` shapes so new copies fail it.
- Effort: low.
- Risk: low — values at the even-count sites move by half a step; check any pinned test values.

> _Status (2026-11-13): **Done** — all twelve sites call `median_in_place`, `quantile_in_place` or `quantile_of_sorted`, as do two more 90th percentiles by index the list missed (`compute_camera_nn_scale` in `analysis/scene_scale.rs`, `framing_view` in `web_export.rs`); the even-count medians now average the two middle values, the percentiles interpolate as numpy does, and `compute_scene_bounds` leaves out a position with a non-finite coordinate rather than panicking on it. A new `the_workspace_selects_no_order_statistic_by_hand` in `numeric/tests.rs` fails on `select_nth_unstable`, `[n / 2]` and `[n * 9 / 10]` shapes outside an `ORDER_STATISTIC_ALLOWLIST` of top-k cuts and lines inside the allowlisted medians, and `specs/STANDARDS.md` describes it, branch `hyg-fix-1113-05-numeric-medians`._

**Make the tests that cannot fail able to fail**
- Location (check 20): `tests/rig/test_rig_config_validation.py:54,62,71,77`; `tests/test_heatmap.py:328`; `tests/test_undistort.py:239`; `crates/sfmtool-core/src/machine/tests.rs` `wait_for_stops_when_cancelled` (106-118); `crates/sfmtool-core/src/analysis/pick_images/source/tests.rs` `a_cancelled_wait_stops_the_workers_promptly` (298-322).
- Problem:
  - Four tests assert `validate_rig_config(...) is not None`, which returns its input or raises; they still pass if the validator drops or rewrites sensors, and `test_omitted_ref_sensor_flag_defaults_to_false` checks no default at all.
  - `assert ... is not None` on `read_image_rgb`, which raises rather than returning `None`; the undistort test still passes if the distorted image is copied unchanged.
  - `wait_for_stops_when_cancelled` still passes if `wait_for` waits out its 5 s probe before returning `Cancelled`.
  - `a_cancelled_wait_stops_the_workers_promptly` calls `pool.wait` with no timeout on jobs that never finish, so broken cancellation hangs the suite rather than failing the test.

  Raw counts: Python `is not None` 168, Rust `is_some()` 180, value or length compared with itself 0, holds and waits 8 Python and 88 Rust. [Tests assert something that can fail](../specs/STANDARDS.md#tests-assert-something-that-can-fail), [Tests cost what their assertions need](../specs/STANDARDS.md#tests-cost-what-their-assertions-need).
- Proposed fix: assert `== data` and check the `ref_sensor` default where it is applied; compare the image shapes with the expected width and height (for undistort, the camera recorded in the output `.sfmr`); add the `elapsed < 2 s` bound the sibling test has; run the wait on a helper thread and `recv_timeout`.
- Effort: low.
- Risk: low.

> _Status (2026-11-13): **Done** — the four rig-config tests compare the result with a deep copy of their input, and `test_omitted_ref_sensor_flag_defaults_to_false` checks that the validator writes no flag in and refuses the unflagged sensor without a pose as a non-reference sensor (the `sfm camrig cp` side of the default is checked by `test_each_rig_of_the_file_becomes_a_camrig`); the heatmap test checks the output height, and `test_basic_undistortion` checks each image's size against its camera in the output `.sfmr`. Undistortion keeps the input size, so a copied image is caught instead by `test_actually_modifies_images`, whose threshold of 0.1 a JPEG re-encode passed; it now needs three times the change a re-encode of the source makes. `wait_for_stops_when_cancelled` has the `< 2 s` bound, and `a_cancelled_wait_stops_the_workers_promptly` waits on its own thread with a 5 s `recv_timeout`, branch `hyg-fix-1113-06-tests-that-can-fail`._

**Treat points at infinity in four Python callers of `recon.positions`**
- Location (check 9): `src/sfmtool/_to_nerfstudio.py:292`, `src/sfmtool/_commands/heatmap.py:188`, `src/sfmtool/_scale_by_measurements.py:334`, `src/sfmtool/visualization/_epipolar_display.py:85`.
- Problem: `recon.positions` returns a direction for a `w = 0` row. Nerfstudio export writes those directions into `sparse_pc.ply` as points one unit from the origin; `sfm heatmap` passes them to `compute_triangulation_angles`, so the angles of those points are wrong; measurement scaling measures a distance from a direction; the epipolar display computes a depth from one. The check's own patterns found none of these (positions_xyz 32 Rust / 31 Python, `[:, :3]` 0 / 4, `/ w` 13 / 0, all handled after reading); all four came from reading the getter's users. [Points at infinity are supported](../specs/STANDARDS.md#points-at-infinity-are-supported).
- Proposed fix: call `materialize_infinity_for_export` before the Nerfstudio write; pass `positions_xyzw` to the angle computation and give `w = 0` rows NaN with a count; raise `ValueError` naming the point when a measured point is at infinity; leave `w = 0` tracks at the baseline default in the epipolar display. Add `.positions\b` users to check 9's patterns.
- Effort: low.
- Risk: low.

> _Status (2026-11-13): **Done** — the Nerfstudio export materializes points at infinity before writing `sparse_pc.ply`, as the COLMAP export does; `sfm heatmap`'s angle metric reads the track table's `max_ray_angle_deg` column (0 for a point at infinity, the number the viewer shows) instead of the Python `compute_triangulation_angles`, which is removed; `--scale-by-measurements` raises `ValueError` naming the measurement and Point ID when a measured point is at infinity; the epipolar display seeds a `w = 0` track's curve with the baseline length; check 9 lists `\.positions\b`; specs updated, branch `hyg-fix-1113-07-python-infinity-callers`._

**Give whole-file I/O and the solver entry points a `&Progress`**
- Location (check 10): `read_*`, `write_*` and `verify_*` in `sfmtool-sfmr-format`, `sfmtool-sift-format`, `sfmtool-matches-format` and `sfmtool-colmap`; core `grow_reconstruction` and `rotation_init` (bundle adjustment under `&Progress::none()`), `background_floor_clusters` (its `_lazy` sibling takes one), `find_points_at_infinity` (reads every `.sift`), `refine_cluster_patches` and `_borrowed` (`Option<&AtomicUsize>` instead, so cannot be cancelled); 32 binding sites that call a Progress-taking core function with `none()` and offer no progress argument, among them `SfmrReconstruction.load`, both `bundle_adjust` bindings, `build_sift_files`, `write_kdf`, KdForest build, `prune_covered_observations` and `drop_repeat_observations`.
- Problem: of 230 public functions that loop over a whole reconstruction, run a solver or read or write a whole file (bench excluded), 31 (13%) take `&Progress`. `Progress::none()` appears 81 times in non-test library code (core 35, py 38, viewer 7, progress 1). Only `sfmtool-kdf-format` among the format crates depends on `sfmtool-progress`. [Long-running functions take a Progress](../specs/STANDARDS.md#long-running-functions-take-a-progress).
- Proposed fix: `&Progress` on the format crates' whole-file functions and on the core functions above; an optional `ProgressCounter` on the bindings through one shared sink in `py_progress.rs` (the one the analysis and patch findings below need).
- Effort: high, across crates; best done crate by crate.
- Risk: low — signatures change at every caller.

**Merge the progress-variant pairs the check-11 suffixes miss**
- Location (check 11): `kdforest/persistent.rs:619/638` (`self_join_with_distances` / `self_join_with_distances_progress`); the patch pairs are in "Let Python cancel the patch kernels" below.
- Problem: the named suffixes (`_with_progress`, `_without_progress`, `_no_progress`, `_silent`, `Option<&Progress`, `verbose: bool`, `show_progress`) count 0 in library code, and `_quiet` 2 (the viewer's `show_quiet`, a false positive). The pairs use `_progress` and `_reporting`, which the list does not catch. [One interface, no progress variants](../specs/STANDARDS.md#one-interface-no-progress-variants).
- Proposed fix: one function taking `&Progress`; add `_reporting\b` and `_progress\b` to check 11's patterns.
- Effort: low.
- Risk: low.

**Panics reachable from Python in the sweep bindings and from Rust callers of the solvers and writers**
- Location (check 12): `crates/sfmtool-py/src/matching/sweep.rs:89-94,259-264,344-349` and core `features/feature_match/sweep.rs`; core `geometry/bundle_adjust.rs:692-744,1398-1553`; `grow_reconstruction`; `sfmtool-sfmr-format` `compute_depth_statistics` (line 284); `write_sift`, `write_sfmr`, `write_matches`.
- Problem: confirmed by probe, `match_one_way_sweep_geometric`, `mutual_best_match_sweep_geometric` and `polar_mutual_best_match_geometric` raise `PanicException` on a `k`/`r` that is not 3×3 or a `t` not of length 3 (`from_row_slice`), and `match_one_way_sweep` / `mutual_best_match_sweep` on keypoints with 1 column or fewer descriptor rows than keypoints. From Rust: `bundle_adjust` asserts and panics on an `image_camera` index past the cameras and on mismatched lengths; `grow_reconstruction` uses `assert_eq!` on its parallel arrays; `compute_depth_statistics` returns a `Result` but indexes `image_indexes` / `point_indexes` unchecked; the three writers call `.as_slice().unwrap()` on caller arrays and panic on Fortran order. 82 public core functions document a caller-input `# Panics`. Raw non-test counts (assert / expect / unwrap / panic! / from_row_slice / `# Panics`): core 232 / 161 / 65 / 6 / 5 / 86; py 0 / 48 / 7 / 0 / 29 / 0; sfmr-format 3 / 6 / 55 / 0 / 0 / 1; matches-format 0 / 16 / 29 / 0 / 0 / 0; kdf-format 0 / 26 / 17 / 0 / 0 / 0; sift-format 0 / 1 / 4 / 0 / 0 / 0; camrig-format 0 / 2 / 2 / 0 / 0 / 0; colmap 0 / 0 / 4 / 0 / 0 / 0; viewer `cli/` + `mcp/` 1 / 69 / 0 / 0 / 2 / -. About 18 lines in 9 specs describe a caller-input panic as the design (for example `bundle-adjustment.md:181,199,871,1109,1671`). [Caller input never panics](../specs/STANDARDS.md#caller-input-never-panics).
- Proposed fix: return an error from each (the sweep part is also in "Make the format binding signatures match, and decide the sweep bindings' future"); write the writers' arrays through `as_standard_layout` ("Write caller arrays through `as_standard_layout`" below); update each spec after its code fix.
- Effort: medium per area; high overall.
- Risk: low.

> _Status (2026-11-13): **Partially done** — the bench normal steps and nearby reads refuse an out-of-range image or wrong-sized data with an error instead of panicking, PR #1234. Fixed there:_
>
> - _`grid_plane_normal` and `grown_plane_normal` refuse with `NoView` instead of panicking._
> - _`finite_difference_normal` refuses with `NoView` instead of a misleading `TooFewPieces`._
> - _`render_bitmap_in_place` now returns a `Result`._
> - _`read_patch_along_ray`, `NearbyCandidate::ray_distance` and `GreyImages::get` refuse instead of panicking, and their `# Panics` sections are gone._
> - _`GreyImage::new` and `ImageDescriptors::new` no longer assert._
>
> _A survey made for that PR lists the public functions that still document a `# Panics` on caller input (about 85, almost all in `sfmtool-core`), for working through as separate changes. Line numbers are as of that PR's base:_
>
> - _**An index into views, cloud, points or images out of range:**_
>   - _`patch/` kernels: `keypoint_localize` (709, 824, `reference.rs:177`), `keypoint_subpixel` (739, 896), `member_coherence:551`, `observation_reading:190`, `stored_bitmap` (123, 196, 298), `view_selection:1228`, `spawn:168` (its binding validates) and `pair_sharpness:272`._
>   - _Geometry and reconstruction: `geometry/reprojection:35`, `adjacency_surfel_normals:184`, `point_or_bearing:112`, `feature_match/gather:29`, `reconstruction/thumbnail:297`, `image_set` (183, 268) and `point_set:476`._
>   - _`sfmtool-sfmr-format` `observation_readings:280`._
>   - _Accessors: `ImageU8::get`, `per_tile_source_stack` (474–538) and `tile_rig` (567, 614)._
> - _**Parallel slices of different lengths, or data that does not match its stated size:**_
>   - _`analysis/`: `observation_adjacency` and `observation_coverage`, with `baseline_direction`, `point_correspondence/by_coordinate`, `spatial:154` and `camera/sampler:409`._
>   - _Constructors: `ImageU8::new` (`camera/image.rs:36`), `warp_map:169`, `optical_flow/image:34`, `sift/gray` (422, 715) and `thumbnail` (42, 93)._
>   - _Patch kernels: `blur_matched/*`, `self_similarity` (and `overlap`), `member_coherence` (the matrix and 259, 272, 319), `reference_view` (`agreement`, `track`), `stored_bitmap` (425, 595, 712), `track_views`, `cluster_refine:1342`, `readings:58` and `edit:625`._
>   - _Others: `tile_rig` (471, 735) and `photograph_cache:190` (`levels` 0)._
> - _**Invalid camera values:** `intrinsics` (1009, 1311) and `rolling_shutter:49`._
> - _**A precondition on how the API is called, not on the input itself** (needs a decision on whether each is caller input):_
>   - _`warp_map` (493, 646): a method called before `compute_svd` or `compute_jacobians`._
>   - _`keypoint_localize:527`: the core outside the tile._
>   - _`keypoint_localize:709`: a search buffer that cannot be allocated._
>   - _`row_map:338`: a row with no source._
>   - _`adjusted_fixture:18`._
>
> _Not counted: the `sfm-explorer` items that are not public API (`bg_image_mesh`, `display.rs` `range` and `refit_points.rs:730`)._

> _Status (2026-11-13): **Partially done** — the writer item: `write_sift`, `write_sfmr` and `write_matches` read every caller array through `sfmtool-archive-io`'s `row_major_bytes` / `row_major_elements` and no longer panic on a Fortran-order or strided array (see "Write caller arrays through `as_standard_layout` in the format writers"), branch `hyg-fix-1113-10-writer-standard-layout`. The sweep bindings, `bundle_adjust`, `grow_reconstruction`, `compute_depth_statistics` and the specs remain._

> _Status (2026-11-14): **Partially done** — the remainder the line above names. The six core sweeps (`match_one_way_sweep`, `mutual_best_match_sweep`, `polar_mutual_best_match` and their `_geometric` variants) check their feature arrays and return `MatchImagePairError::FeatureArrayLength`; `match_one_way_sweep` takes `desc_len` rather than guessing it, and the bindings check every matrix, vector and feature-array shape under their argument names and raise `ValueError`. `bundle_adjust` returns `Result<BundleAdjustment, BaInputError>` and refuses, before moving anything, every length, index, camera, adjustment and constraint it used to assert on, plus `trans`, `obs_img` and `obs_pt` it never checked; `PointConstraints::from_arrays` also refuses an empty `ImageMean` reference. `grow_reconstruction` returns `Result<_, GrowInputError>`; `compute_depth_statistics` refuses index arrays of different lengths and indexes past the poses or points. The internal callers (`rotation_init`, the growth's adjustments, the reconstruction-level adjustment) build valid input and state why at their `expect`. `bundle-adjustment.md`, `reconstruction-growth.md` and `descriptor-matching.md` now describe refusals, not panics. Branch `hyg-fix2-1113-06-sweep-solver-panics`. Still open: the `# Panics` survey in the first status line, and the spec sentences that describe those functions' panics as the design (for example `baseline-direction.md`, `reprojection-residuals.md`, `zncc-self-similarity-radius.md` and `sfmtool-pinhole-kernels.md`)._

**Have the image-pair-graph bindings take the reconstruction, and stop repeating bindings in Python**
- Location (check 13): `src/sfmtool/_image_pair_graph.py:45,56`; `src/sfmtool/align/core.py:205`; `src/sfmtool/sift/file.py:40,61`; `src/sfmtool/_patch_compaction.py:50`; `src/sfmtool/camrig/create.py:56`, `camrig/cp.py:77`.
- Problem: of 324 top-level Python functions and classes that use a binding name, these repeat what Rust does:
  - `build_covisibility_pairs` / `build_frustum_intersection_pairs` unpack an `SfmrReconstruction` into 4 and 16 arrays (the frustum one looping over the intrinsics per image), re-check the depth statistics and convert the result to tuple lists.
  - `estimate_alignment` re-checks the shapes with a minimum of 2 points against the core's 1 (`least_squares.rs:63`) and forwards to the binding of the same name.
  - `get_feature_tool_xxh128` and the feature-type naming copy `sfmtool_core::workspace::feature_prefix_dir`, whose doc says the two "need not" agree; `xxh128_of_file` copies core `features::sift::files::xxh128_of_file`.
  - `image_file_hashes_from_sift` re-reads every `.sift` that `to_embedded_patches` reads; only tests call it.
  - Two copies of the backslash normalization around `validate_camrig_pattern`.

  All 25 root `_LAZY_NAMES` entries are Python names. [Logic lives in Rust, not in Python wrappers](../specs/STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers).
- Proposed fix: the pair bindings take the reconstruction and return the pairs; the shape check moves into core `estimate_alignment` and `align/by_points.py` calls `sfmtool.analysis.estimate_alignment`; bind `feature_prefix_dir` and a file hash and delete the Python copies; delete `image_file_hashes_from_sift` and have the tests read `to_embedded_patches(...).image_file_hashes`; the camrig format validator normalizes and returns the pattern.
- Effort: medium.
- Risk: medium — the feature prefix directory must stay byte-identical for existing workspaces.

**State or shrink the f64 accumulators in the patch ZNCC kernels**
- Location (check 19): `crates/sfmtool-core/src/patch/normal_refine/support.rs:280-306`, `patch/reference_view/agreement.rs:68-90`, `patch/blur_matched/zncc.rs:15-45`, `patch/cell_registration.rs:270-305,453`.
- Problem: f32 patch samples accumulated in f64 with no stated reason: two-pass weighted moments in `support.rs`, where f32 is probably enough; single-pass raw moments in `agreement.rs` and `zncc.rs`, where the subtraction of two large sums may justify f64; summed-area tables in `cell_registration.rs`, where differences of table entries do. Raw counts in library code (inside image, patch, tile, flow and feature directories in brackets): `f64::from(` 300 [101], `vec![0f64` 57 [17], `Vec<f64>` 530 [174], `= 0.0f64` 27 [16], `.sum::<f32>()` 5 (none over a whole image), `as f64 / 255` 0, order-dependent rayon float reductions 0. The photometric RANSAC case is the carried finding below. [Numeric precision is the smallest the use justifies](../specs/STANDARDS.md#numeric-precision-is-the-smallest-the-use-justifies).
- Proposed fix: measure `support.rs` in f32 and switch; for the other three, either state the cancellation reason at the accumulator or move to two-pass f32 moments.
- Effort: low.
- Risk: low — ZNCC values change in the last digits.

**Run the refine-normals stored-bitmap test on a working grid the size of the patch**
- Location (slow-test dive): `crates/sfmtool-core/src/bench/tests/stored_bitmap.rs:132-153` (options at 143); the clamp is documented on `working_resolution` at `bench/normal/grown_plane.rs:109-114`.
- Problem: the test runs `refine_normals` with `RefineNormalsOptions::default()` at three tilts (15, 25, 35 degrees) over 441 points. `working_resolution: None` follows the patch's footprint but clamps it to `R..=2R`; the patch is about 10 px across (half-extent 0.12 at distance 4, focal 160), but `BITMAP_R = 48` is set only so the stored bitmaps have enough samples, so the normal search runs on a 48×48 grid per view, about nine times the samples one per pixel needs. One refine run costs about 2.9 s against 0.8 s for the refit test, which renders the same 441 bitmaps at 48×48. The assertions read only the final stored bitmap and `report.turned() > 220`. Alone in the test profile on 32 threads it takes 8.8 s (23.4 s with `RAYON_NUM_THREADS=4`, 6.8 s in release); CI's coverage build reports it over 60 s. Separately, the module comment says the grid "is sized so that, without the rounding at render time, some [samples] do" cross a rounding, and no test asserts that this happens. [Tests cost what their assertions need](../specs/STANDARDS.md#tests-cost-what-their-assertions-need).
- Proposed fix: the smallest input that exercises the behaviour: `RefineNormalsOptions { normal: GrownPlaneNormalOptions { working_resolution: Some(16), ..Default::default() }, ..Default::default() }` (above `spot_window = 7`). All three tilts, the 441-point grid, the 48×48 stored bitmaps, the `turned() > half` assertion and `bytes == 0` stay. Confirm that each tilt still turns more than half the points, and that removing the `at_stored_precision` rounding still makes `bytes == 0` fail (then revert); if 16 fails either, try 24, then 32. Add an assertion that the rounding event happens. Expected: about 8.8 s to 3 s (not measured).
- Effort: low.
- Risk: low to medium — a coarser working grid lands on slightly different frames, so the rounding event must be re-confirmed.

> _Status (2026-11-14): **Done** — the test runs the grown-plane normal on a 16-sample working grid (`working_resolution: Some(16)`) and keeps all three tilts, the 441-point grid, the 48 x 48 bitmaps and both assertions (every tilt turns all 441 points; dropping the rounding in `evaluate_track` leaves 12 bytes off); a new assertion renders each refined value through frames that store as the stored ones but are not them and requires some byte to differ (6 do over 1323 bitmaps). Alone in the test profile on 32 threads, 5.7 s to 1.7 s; with `RAYON_NUM_THREADS=4`, 19.2 s to 5.9 s. Branch `hyg-fix2-1113-08-stored-bitmap-test-grid`._

**Share the copied blocks the duplicate scan found outside the units read**
- Location (checks 3 and 4): as listed.
- Problem and proposed fix, one per pair:
  - `crates/sfmtool-core/src/bench/refine_normals.rs:287-327` copies `bench/refit_points.rs:267-303` (`NoKeypoints`, `NoPatchFrames`, `NoPosedImage`, `ViewsMissing`, `ImageOutOfRange`, `Cancelled`, the same Display text, the report type and the parallel driver); 27 shared core lines, 36 in their tests, 27 in the viewer's `state/edits/{refine_all_normals,refit_all_points}/tests.rs`. Fix: one bench run-error enum and a generic run report and driver (with "Fold the bench error enums").
  - `geometry/epipolar_estimation.rs:305` `draw7` and `geometry/homography_estimation.rs:214` `draw4` are the same sampler, and the LO refit and adaptive termination are copied (23 lines). Fix: a const-generic `draw_distinct::<K>` beside `numeric::splitmix64` and one LO/termination helper.
  - `src/sfmtool/_commands/match.py:273-314` and `solve.py:178-312` hold the same selection-to-paths-to-workspace block (20 lines). Fix: a `_commands/_image_selection.py` helper.
  - `crates/sfm-explorer/src/camera_model_proposal/job.rs` and `track_table_panel/job.rs` hold the same quiet-task job driver (25 lines). Fix: a generic one in `background`.
  - Test fixtures (low priority): `patch/keypoint_{localize,subpixel}/tests.rs` (56, the plane `Scene`; a `#[cfg(test)] patch::test_scene`); `tests/test_per_image_{intrinsics,lens_corrections}.py` (49, `_undistort` and the retake helper, to `conftest.py`); sfmr-format `tests/{lens_corrections,per_image_adjustments}.rs` (40; `temp_dir` defined 5 times in `tests/`); `mcp/tests/{input,widgets}.rs` (33, the fake dock harness); `tests/patch/test_patch_keypoint_{localization,subpixel}.py` (30, `_project`); `tests/xform/test_{localize,refine}_keypoints.py` (23, grammar tests to parametrize); matches and sfmr `tests.rs` (54, the entry-pin doc and helpers); core `bundle_adjust` `tests/.../corrections.rs` pair (23, `spline_pinhole` and `correction_draws`).
- Effort: low each.
- Risk: low.

**Cut the retold algorithm from the bench normal binding docstrings**
- Location (checks 1 and 2): `crates/sfmtool-py/src/bench/normal.rs:1003-1210` (`grown_plane_normal`, 169 doc lines over a 94-line body, the longest comment block in the tree) and `grid_plane_normal` (101 / 57).
- Problem: the first ~75 lines of `grown_plane_normal`'s docstring retell the core algorithm (rungs, robust mean, round caps) that `crates/sfmtool-core/src/bench/normal/grown_plane.rs:739` already explains (a 128-line block); `grid_plane_normal` does the same. Across `sfmtool-py` outside the bench, 28 functions have docs over twice their body (was 17) and 46 have 30 or more doc lines (was 36).
- Proposed fix: an argument reference and the report keys, with a pointer to the core doc and spec for the algorithm.
- Effort: low.
- Risk: low.

**Use one name per CLI setting across sibling commands**
- Location (check 6): `src/sfmtool/_commands/`.
- Problem: `camrig spherical-tiles --n` against `panorama --n-tiles`, both binding `n_tiles`; `densify --filter-max-reproj-error` against `xform --filter-by-reprojection-error`; `explorer-call --out-dir` against `align --output-dir`; `solve --seq-overlap` (`WINDOW,OVERLAP`) against `match --sequential-overlap` (a pair count), one name for two different settings; `embed-patches --max-obliquity-deg` against the glossary's "viewing angle" used by its siblings. No implementation suffix appears among 388 flag uses, and 0 Python-visible names end in `_py`, `_rs`, `_impl` or `_inner`.
- Proposed fix: `--n-tiles`; one reprojection-error filter spelling; `--output-dir`; distinct names for the two overlap settings (for example `--seq-window` for `solve`); record the choices in `specs/GLOSSARY.md`.
- Effort: low.
- Risk: medium — CLI flags are public; keep the old spelling as a hidden alias for one release only if the maintainer wants it.

**Clear the glossary residue in the bench and patch code**
- Location (check 8): "world units" in bench scope, 12 lines (`crates/sfmtool-core/src/bench/normal/grid_plane.rs:238,242`, `grid_plane/detail.rs:81,185,232,239`, the user-facing docstring at `crates/sfmtool-py/src/bench/normal.rs:873`, `bench_track.wgsl:22,174`, 3 test comments), most written 2026-10-14 and 10-15, after the ruling; "bitmap score(s)", about 9 bench lines, including the progress phase `"bitmap scores"` at `bench/evaluate.rs:2222`, `fit.rs:387,480`, `steps.rs:3211`, py `bench.rs:1806`; "descriptor index", 3 code docs (`bench/mod.rs:20`, `mcp/edit.rs:1149`, `track_view/body/mod.rs:2149`), 1 test message, 5 lines in `scripts/track_at_pixel/`; `member_keypoints` in `patch/member_coherence/matrix.rs` (11) and `member_coherence.rs` (3), the last module on a word the 2026-10-17 ruling retired; "flat cell" ×4 against "capped cell"; py `bench.rs:1994` "infinity test".
- Problem: 79 probes over the glossary entries; the residue above is in scope, and the rest of the entries are clean (listed under the mechanical numbers).
- Proposed fix: replace each with the glossary word ("scene units", the score's glossary name, "SIFT index", `patch_keypoints`, "capped cell").
- Effort: low.
- Risk: low — the `"bitmap scores"` phase name is visible to a progress reader.

**Rule on error-message capitalization**
- Location (check 5): `crates/sfmtool-archive-io/src/lib.rs:106` and `crates/*-format/src/types.rs` (error-enum prefixes such as `"Invalid format: {0}"`, `"Shape mismatch: {0}"`); `sfmtool-colmap` (9 capitalized, 8 lowercase); verify messages such as "Sum of match_counts".
- Problem: core, `sfmtool-py` and the viewer are all lowercase (29 / 96 / 95, 0 capitalized); the format crates and COLMAP are capitalized, and are the older side. The glossary has not ruled.
- Proposed fix: lowercase, the Rust API guidelines' form and the house majority; record it in `specs/GLOSSARY.md` and change the format crates and COLMAP.
- Effort: low.
- Risk: low — tests that match message text change.

**Remove the documented exceptions the check found, once confirmed**
- Location (check 15): `crates/sfmtool-matches-format/src/entries.rs:17-28` and the same rule in sfmr-format (confirmed, in "Give each `entries.rs` dimension function one parameter type" below); `crates/sfmtool-py/src/patches/cloud.rs:83-85` (confirmed, in "Let Python cancel the patch kernels"); candidates not yet read: `specs/core/reconstruction/edited-reconstruction.md:795` ("the single exception to the index rule"), `specs/core/camera/camera-model-registry.md:15`, `specs/formats/archive-container.md:36` (`.kdf`'s `.frames` entries), `specs/gui/mcp-server.md:518,3882,5065,7024`, `crates/sfm-explorer/src/background/mod.rs:42`, `crates/sfm-explorer/src/mcp/edit.rs:31`.
- Problem: 71 raw hits (AGENTS.md 1, specs 40, Rust doc comments 28, Python 2), most using "exception" in another sense. [A convention has no documented exceptions](../specs/STANDARDS.md#a-convention-has-no-documented-exceptions).
- Proposed fix: read each candidate; where it is an exception to a convention, change the code so the convention holds and delete the sentence.
- Effort: low.
- Risk: low.

**Build `insv2rig`'s rotation inside the command**
- Location (check 17): `src/sfmtool/_commands/insv2rig.py:29`.
- Problem: the module builds a `RotQuaternion` at import, the only module-level work beyond imports, two `re.compile` constants and the command definition in the command modules. Two specs name a binding by its internal path in prose where the public path is `sfmtool.matching` / `sfmtool.patches` (`specs/core/patch/cluster-patch-refinement.md:68`, `cell-plane-normals.md:13`). [Importing sfmtool and sfm loads only what is used](../specs/STANDARDS.md#importing-sfmtool-and-sfm-loads-only-what-is-used).
- Proposed fix: build the rotation inside the command function; name the public module in the two specs.
- Effort: low.
- Risk: low.

> _Status (2026-11-14): **Done** — `insv2rig` builds the right eye's `RotQuaternion` where it writes the rig; a grep of `src/sfmtool/_commands/` confirmed it was the only module-level call beyond the two `re.compile` constants, which are now pattern strings passed to `re.match` / `re.fullmatch`; the new `test_command_modules_call_nothing_at_import` in `tests/test_lazy_loading.py` fails on any call a command module makes at import outside its decorators, and STANDARDS.md and `specs/cli/README.md` say so. The specs name the public modules: `sfmtool.matching.refine_cluster_patches` (cluster-patch-refinement.md, twice, and cluster-patches-command.md) and `sfmtool.analysis.cell_plane_normals` (cell-plane-normals.md; the binding is in `analysis`, not `patches`), branch `hyg-fix2-1113-09-insv2rig-lazy-rotation`._

## Carried findings

Still-open findings of the 2026-10-08 report. A finding whose units were not read this run carries the date of its last deep read; its figures were re-measured at `929b62f0`.

### Carried forward

**Convert the remaining `mod.rs` files and rename the settings structs** (carried, last read 2026-10-08)
- Location:
  - 44 `mod.rs` files (m1, was 44): 16 in core outside the bench plus `bench/mod.rs`, 15 in `sfm-explorer`, 10 in `sfmtool-py` and 2 in `sfmtool-colmap`. 245 modules use the `foo.rs`-beside-`foo/` form.
  - 27 settings structs in `sfmtool-core` ending in `Params` or `Config` (m1, was 26), including `GeometricFilterConfig` and `RelaxConfig`. A plain `pub struct \w+(Params|Config)` grep finds 28 names, which appear on 1,382 lines across `crates/`, `src/`, `tests/` and `scripts/` (was about 1,300). Core declares 58 `*Options` (m1, was 45).
- Problem: the rules are written down in AGENTS.md and the glossary (this finding came from "Record the module-file and parameter-bag naming rules", Partially done on 2026-10-08, #862). No `mod.rs` was added since. Two settings structs were added after the rule: `CellPlaneParams` (`patch/cell_plane_normals.rs:46`) and `PiecewiseParams` (`patch/cluster_refine/piecewise.rs:83`), both by PR #870 on 2026-10-09 (m1). `NormalRefineParams` was removed. Still to do:
  - Convert each `mod.rs` while its module is being restructured anyway. The viewer findings below for `track_view/body` and `viewer_3d` are such moments.
  - Do the struct renames as a batch, starting with the two added after the rule (`CellPlaneOptions`, `PiecewiseOptions`).

  `ConstellationParams` cannot become `ConstellationOptions`, because a bench struct already has that name; see "Disambiguate the bench's flat re-exports" below.
- Proposed fix: rename the settings structs in two or three crate-scoped PRs, as clean breaks, updating the bindings and the Python `GeometricFilterConfig` class (`src/sfmtool/feature_match/_geometric_filter.py`) with them.
- Effort: medium. Risk: medium, because the types are public.

**Choose one word for the per-patch view lists in core and the bindings**
> _Status (2026-10-17): **Partially done** — decided by the maintainer on 2026-10-17: `track_views` for a track's images read from the reconstruction, `patch_views` for the ordered per-patch list a driver evaluates against, not "set" because the lists are ordered; one `patch::track_views_from_reconstruction` and one `patch::track_keypoints_from_reconstruction` (new `patch/track_views.rs`) replace `normal_refine::view_indices_from_reconstruction`, `member_coherence::member_views_from_reconstruction` / `member_keypoints_from_reconstruction` and the `view_selection` / `keypoint_localize` forwards; the batch drivers, one-patch kernels, bench locals and bindings take `patch_views` / `patch_keypoints` (was `view_sets`, `view_set`, `member_views`, `view_indices`, `starting_keypoints`, `member_keypoints`, `patch_view_keypoints`), `select_views(candidate_views=…)` becomes `track_views=…`, "view set `G`" becomes the patch's views `G` in the specs and doc comments, and the glossary has a "Per-patch view lists" section, branch `decided-1017-04-patch-views`. Still open: `patch::run_over_cloud`._

> _Status (2026-10-31): **Partially done** — `refine_patch_cloud_normals` was deleted with the photometric normal search, so four batch drivers repeat the skeleton rather than five, PR #1082. Still open: `patch::run_over_cloud`._
- Location: the four core batch drivers, `crates/sfmtool-core/src/patch/keypoint_localize.rs:863-891`, `keypoint_subpixel.rs:930-949`, `member_coherence.rs:576-589` and `view_selection.rs:1257-1285`, beside `PatchCounter` at `patch/mod.rs:45`. The binding layer in `crates/sfmtool-py/src/patches/` repeats the matching argument setup (m1 check 3, item 1).
- Problem: `run_over_cloud` has 0 hits. The four drivers still repeat the same skeleton: length check, `PatchCounter::new`, `par_iter` with a cancel check, `counter.finished()`. Three of them also take `done: Option<&AtomicUsize>` beside `&Progress` (m2 check 11).
  Re-measured with `crates/sfmtool-py/src/patches` read in full this run: no `view_sets`, `view_indices` or `member_views` keyword remains in the bindings, and `patch_views` / `track_views` are used throughout; "view set" remains in two docstrings and the locals `sets` ("Bring the sibling patch bindings' names and docs into line"); the `refine_keypoints` binding is a fifth copy of the skeleton ("Call the core `refine_patch_cloud_keypoints` from the `refine_keypoints` binding").
- Proposed fix: add `patch::run_over_cloud` beside `PatchCounter` and have the four drivers call it, taking only `&Progress` ("Let Python cancel the patch kernels").
- Effort: medium. Risk: medium, because both the Rust and Python names are public.

### Rust: core algorithms

**State or shrink the f64 accumulators in the photometric RANSAC** (carried, last read 2026-10-08)
- Location: `crates/sfmtool-core/src/spherical/photometric_ransac.rs` (841 lines): `:341-371` (per-row mean brightness), `:643-649` (mean score), `:678-681` (the median buffer), `:701-710` and `:727-734` (validity-weighted channel sums). m3 check 19 flags the same lines.
- Problem:
  - The kernel accumulates `f32` pixel values, validities and scores of 8-bit patches in `f64` and gives no reason for it. [STANDARDS.md § "Numeric precision is the smallest the use justifies"](../specs/STANDARDS.md#numeric-precision-is-the-smallest-the-use-justifies) asks for the smallest precision the use justifies, with the reason stated where it is chosen.
  - The sums are over one patch's pixels (a few hundred terms) or over the rows of one stack (fewer), where an `f32` running sum is bounded by about `n · 2^-24`, well under one 8-bit level. The median buffer is `f64` only because `median_in_place` takes `f64`.
- Proposed fix: measure the inlier sets and gains against an `f32` version on the kerry_park and seoul_bull tile stacks, then either move the sums to `f32` or state in the module docs why `f64` is needed.
- Effort: low–medium. Risk: medium, since a last-bit change can flip a residual threshold; pin the RANSAC fixtures before and after.

**Share the Levenberg–Marquardt damping step** (carried, last read 2026-10-08)
- Location:
  - Same rule: `geometry/pose_refine.rs:196-223`, `geometry/resect_images/finite.rs:392-418`, `reconstruction/triangulation/points.rs:974-995`, `analysis/cluster_census/group_consistency.rs:422-452`, and `geometry/bundle_adjust.rs:3805-4300` (five `lambda *= 4.0` sites).
  - ×10 variants: `camera/refit_intrinsics.rs:1616-1634`, `reconstruction/triangulation/point_or_bearing.rs:872-905`.
  - `INVALID_RESIDUAL = 1e6` is still defined four times: `bundle_adjust.rs:55`, `resect_translation.rs:46`, `pose_refine.rs:25`, and `cluster_census.rs:37` (as `INVALID_RESIDUAL_PX`).
- Problem:
  - Five copies apply the same rule: damp the diagonal by λ·max(diag, 1e-12), ×4 on failure, ×0.5 on success with a floor of 1e-12, and give up above 1e12. `numeric` has no `marquardt` helper (0 hits).
  - `pose_refine` and `finite::refit` (`finite.rs:364`) are still line-for-line identical 6-DoF refits. `finite` imports the Jacobian helpers from `pose_refine`.
- Proposed fix:
  - Collapse `finite::refit` onto `pose_refine` first.
  - Then add one `numeric::marquardt_step`.
  - Define one crate-level `INVALID_RESIDUAL`.
- Effort: medium. Risk: medium, because convergence must stay bit-identical; the `newton_2d` extraction is the precedent.

**Give the bundle-adjustment kernel an options struct and split `solve_lm`'s setup** (carried, last read 2026-10-08)
- Location: `crates/sfmtool-core/src/geometry/bundle_adjust.rs`, 5,554 lines (was 3,230).
  - `pub fn bundle_adjust` at 1368 takes 26 arguments (was 20).
  - `solve_lm` runs 3056–4455, 1,400 lines (was 833). Its iteration loop starts at 3521, so about 465 lines of setup come before it.
- Problem:
  - Fifteen caller-chosen settings are positional: `free_points`, `protected_loss_scale`, `opt_f`, `opt_k1`, `opt_bspline`, `opt_pp`, `opt_image_f`, `opt_image_pp`, `opt_image_radial_bspline`, `opt_image_rolling_shutter`, `image_prior`, `schedule`, `max_iters`, `min_track`, `min_obs`.
  - Production call sites pass runs of bare booleans: `reconstruction_growth.rs:1209` passes eight `false` in a row (1222–1229). The other production callers are `reconstruction_growth.rs:1332`, `rotation_init.rs:786` and `reconstruction/bundle_adjust.rs:1481`.
  - The file also holds `PointConstraints`, its error type and `FreePointPolicy` (136–420).
  - The tests under `geometry/bundle_adjust/` have 62 lines calling `bundle_adjust(`.
- Proposed fix:
  - Add a `BaOptions` struct.
  - Extract a private `RoundLayout::new` from the setup.
  - Move the constraints to `bundle_adjust/constraints.rs`.
- Effort: medium. Risk: medium; the numerical fixtures must hold.

**Share the cluster pair tables between the focal vote and rotation initialisation** (carried, last read 2026-10-08)
- Location: `geometry/focal_vote.rs:465-520,1025-1080` and `geometry/rotation_init.rs:132-230`.
- Problem: `PairAccum`, `ImageClusters`, `pair_correspondences` and `build_pair_tables` are still defined in both files. Both `ImageClusters` aliases are now the same type, `Vec<Vec<(u32, [f64; 2])>>`, so the input-shape difference the 2026-10-08 read gave (f32 against f64 positions) no longer holds at that type. `rotation_init` still imports `ortho_cost` from `focal_vote`. No `cluster_pairs.rs` exists.
- Proposed fix: add a private `geometry/cluster_pairs.rs` and have both call it.
- Effort: low–medium. Risk: low.

### Rust: the bench

**Disambiguate the bench's flat re-exports** (carried, last read 2026-10-08)
- Location: `crates/sfmtool-core/src/bench/mod.rs:71-180` (14 `pub use` blocks), `features/kdforest/constellation.rs:73`, `bench/search.rs`.
- Problem: 263 names are re-exported flat into `bench::` (was about 180). They include:
  - `ConstellationOptions` (cascade member), `ConstellationSeedOptions` (nearby source) and kdforest's `ConstellationParams`.
  - `ClusterMember` (a struct) and `ClusterMembers` (a policy enum).
  - Generic names that belong only to the descriptor search: `SearchOptions`, `SearchError`, `Found`, `DEFAULT_RADIUS_PX`.

  `search_geometry` lives in `geometry_search.rs` with `GeometrySearch*` types, while `search_descriptors` lives in `search.rs` with bare `Search*` types.
- Proposed fix:
  - `ConstellationParams` → `ConstellationQueryOptions`. Its doc (`constellation.rs:65`) still says "Tunables for `constellation_query`", and the name is free.
  - `search.rs` → `descriptor_search.rs` with `DescriptorSearch*` types.
  - Either suffix the cascade member options `*MemberOptions`, or stop re-exporting them flat.
- Effort: medium (`ConstellationParams` has 57 references, was 73). Risk: low.

**Share the constellation sizing rule, and drop a dead override** (carried, last read 2026-10-08)
> _Status (2026-10-09): **Partially done** — dropped the dead override: `nearby/constellation.rs` and `track_at_pixel/members.rs` now pass `ConstellationParams::DEFAULT` and set the bar only on `SearchOptions::min_inliers`, and the `sfmtool-py` `search_descriptors` binding fills the unread field from the query defaults instead of its `min_inliers` argument (which still sets `SearchOptions::min_inliers`, so no public signature changed). The viewer at `sfm-explorer/src/bench.rs` already used `..SearchOptions::default()` and needed no change. Still open: the shared `DEFAULT_CONSTELLATION_FEATURES` constant and a `SearchOptions::sized_for` constructor. Branch `hygiene-fix-1009-10-constellation-min-inliers`._
- Location:
  - The target of 50 features: `bench/nearby/constellation.rs:88`, `bench/track_at_pixel.rs:373` (`constellation_target`), `crates/sfm-explorer/src/bench.rs:171` (`SEARCH_TARGET_FEATURES`), and implicitly `DEFAULT_RADIUS_PX = 128` at `bench/search.rs:49`.
  - The sizing call built by hand: `nearby/constellation.rs:157-165`, `track_at_pixel/members.rs:830-843`, viewer `bench.rs:2585-2591`.
- Problem: `DEFAULT_CONSTELLATION_FEATURES` and `sized_for` have 0 hits. Three callers each build `SearchOptions` from `radius_for_feature_count(width, height, keypoints, target)` by hand, and the target is written four times.
- Proposed fix:
  - Add a `kdforest::DEFAULT_CONSTELLATION_FEATURES` constant.
  - Add a `SearchOptions::sized_for(width, height, keypoint_count, target, min_inliers)` constructor.
- Effort: low. Risk: low.

**Move the shared bench inputs out of `track_at_pixel`, and reuse its helpers** (carried, last read 2026-10-08)
- Location:
  - `crates/sfmtool-core/src/bench/track_at_pixel/{neighbourhood,finish}.rs`, imported by 10 non-test `nearby/` files (14 counting the test files; was 9).
  - `track_at_pixel/members.rs:713-790` (`local_prior`) against `depth_modes` (246) and `half_px_of` (273).
  - Options in `bench/track_at_pixel.rs`: `TransferOptions` (258), `SweepOptions` (296), `ConstellationOptions` (337).
- Problem:
  - `ViewCamera`, `MatchesClusters`, `SiftIndexSource`, `ObservationIndex`, `seed_cluster_with` and `upgrade_sightings` are shared inputs, but they live in the cascade module. No `bench/sources.rs` exists.
  - `local_prior` still splits depths into modes inline (`members.rs:741-747`), the job `depth_modes` does.
  - `SweepOptions` and `ConstellationOptions` each restate `prior_radius_px` (40.0), `prior_k` and `depth_mode_gap` (1.15), with `prior_k` 10 in one and 8 in the other. `TransferOptions` repeats `depth_mode_gap` only.
- Proposed fix:
  - Add a bench-level `sources.rs`.
  - Have `local_prior` call the helpers.
  - Nest a `LocalPriorOptions` in each member's options, keeping the per-member values.
- Effort: low–medium. Risk: low to medium, because Python's dotted option keys would change.

**Share the viewer's pixel-job setup and the `TrackMeasurement` serialisation contract** (carried, last read 2026-10-08)
- Location:
  - Viewer `bench/nearby_tracks.rs` (732 lines) and `bench/track_at_pixel.rs` (602 lines).
  - `created_points` at viewer `bench.rs:3372` and `bench/nearby_tracks.rs:584`.
  - `crates/sfmtool-py/src/bench/{track_at_pixel.rs:55, nearby.rs:56}`.
  - Python serialiser `observation_to_dict` at py `bench.rs:309-487`; MCP `track_measurement` at `mcp/bench.rs:1880-1961`.
- Problem:
  - Find Nearby Tracks and Create Track Here share 33 long lines (26 at 50 or more characters; was 44 and 17 by the 2026-10-08 counting, m1 check 3). No `pixel_query` module exists.
  - `created_points` is still written twice.
  - Python's `TrackAtPixelSources` and `NearbyTrackSources` each parse the same inputs.
  - `TrackMeasurement` now has 40 fields (was 33), listed by hand in both the Python and the MCP serialiser, and no test checks that either one covers every field.
- Proposed fix:
  - Add a viewer `bench::pixel_query` module (`IndexSources`, `pixel_on_photograph`, `decode_then`, `read_cluster_patches`).
  - Choose one `.sift` policy on purpose.
  - Add `CreatedPoints::of(value, indexes)`.
  - Add a key-coverage test per serialiser.
- Effort: low–medium. Risk: low; the refusal text is pinned.

**Fold the bench error enums**
- Location: `crates/sfmtool-core/src/bench/fit.rs:146-255` (`FitError` and its `From<EvaluateError>` at 255) against `evaluate.rs` (`NoPatchFrame` at 235). The seven `NoPatchFrame` variants: `commit.rs:114`, `evaluate.rs:235`, `fit.rs:164`, `geometry_search.rs:50`, `normal.rs:128`, `stage.rs:38`, `steps.rs:607`.
- Problem: `FitError` still repeats `EvaluateError` variants with copied Display strings and a field-by-field `From`; `evaluate.rs` and `fit.rs` share 25 long lines (m1 check 3). The pattern has spread: `refine_normals.rs:287-327` copies `refit_points.rs:267-303` (`NoKeypoints`, `NoPatchFrames`, `NoPosedImage`, `ViewsMissing`, `ImageOutOfRange`, `Cancelled`, with identical Display, 27 shared core lines), and `nearby/{far_field,candidate,find}.rs` repeat one set of query error variants (m1 checks 3 and 4). The 2026-10-18 status on the `frame` finding left "sharing one precondition error among the seven variants" open, which this finding covers.
  Re-measured with `bench/nearby/find` read in full this run: the query refusals are now copied in six enums, with 90 lines of field-by-field `From` in `find.rs` ("One error type for the bench queries' shared refusals").
- Proposed fix: `FitError::Evaluate(EvaluateError)`, one shared no-patch-frame precondition error, one shared run error for `refine_normals` and `refit_points`, and `bench::QueryError` for the query refusals.
- Effort: low–medium. Risk: low.

**Split the bench's largest files along their sections** (carried, last read 2026-10-08)
- Location and problem:
  - **Core `bench/tests.rs`: 8,368 lines, 190 tests** (was 7,171 and 167). Geometry-search tests sit at 100–338, and the finite-points-and-bearings and classification sections run 6599–7946, about 1,350 lines. `bench/tests/` holds 10 modules.
  - **`mcp/tests/bench.rs`: 5,348 lines** (was 4,411).
  - **Viewer `bench.rs`: 3,386 lines** (was 3,076). It holds focus and recent-item state, synchronous steps, and background jobs.
  - **Core `steps.rs`: 3,577 lines** (was 3,001). Its sections are at 101 (putting a point on the bench), 376 (create), 572 (steps on one track), 1262 (placing and sizing) and 3300 (split).
  - **`nearby/{find,layer,source}_tests.rs`** (493, 352 and 684 lines). Three of the workspace's eight `*_tests.rs` files, against 257 `tests.rs`.
  - **`scripts/track_at_pixel/anchors.py`: 2,160 lines.** It still keeps Python reference implementations behind five `*_impl` switches (`range_impl`, `sources_impl`, `ff_impl`, `layers_impl`, `finder_impl`), all defaulting to `"rust"`.
- Proposed fix:
  - Core: continue `bench/tests/` per production module, and move geometry-search tests to `geometry_search/tests.rs`.
  - Viewer: split `bench.rs` into `bench/focus.rs` and `bench/jobs.rs`.
  - Split `steps.rs` into `steps/{create,verdicts,gestures,thresholds,split}.rs`.
  - Move the nearby tests to `tests.rs` form.
  - Decide whether to delete the Python references in `anchors.py` or keep them under a parity test.
- Effort: medium (mechanical moves). Risk: low.

### Rust: the viewer

**Move photograph decoding and the history cursor out of `state/edits.rs`** (carried, last read 2026-10-08)
- Location: `crates/sfm-explorer/src/state/edits.rs`, 2,458 lines (was 2,127).
- Problem:
  - `DecodedViews` and `ViewSources` (635–868) and `view_sources_for` (2248–2281) decode photographs for any worker job. 8 other files use them, including viewer `bench.rs`, `bench/nearby_tracks.rs` and `bench/track_at_pixel.rs`, so the bench still reaches into "edits" for them.
  - The undo/redo/jump cursor (2041–2247, plus `CursorMove` at 2367–2412) is a separate concern.
- Proposed fix: move them to `state/views.rs` and `state/history.rs`, following the `edits/switch_camera_model.rs` pattern.
- Effort: low. Risk: low.

**Split `track_view/body` and `viewer_3d` by concern, converting their `mod.rs`** (carried, last read 2026-10-08)
- Location: `track_view/body/mod.rs` (2,860 lines, was 2,589), `track_view/body/table.rs` (2,617 lines; `draw_row` at 1648–2166, 519 lines with 12 parameters besides `self`, was about 480), and `viewer_3d/mod.rs` (1,807 lines, was 1,727).
- Problem:
  - `body/mod.rs` carries about 590 lines of cell text and colour helpers (1376–1963) and about 345 lines of header and summary drawing (2172–2517).
  - `draw_row` draws every column inline.
  - About 815 lines of `viewer_3d/mod.rs` (953–1769) are camera transitions and camera-view mode. The siblings `camera.rs`, `framing.rs` and `righting.rs` already show the per-concern layout.
- Proposed fix:
  - Split `body/` into `cells.rs` and `header.rs`, with one draw method per column group.
  - Split `viewer_3d/` into `transition.rs` and `camera_view.rs`.
  - Convert both `mod.rs` files in the same change.
- Effort: medium. Risk: low; the headless tests run whole frames.

**Share the bench handle reach between the two views** (carried, last read 2026-10-08)
- Location: `image_detail/bench_track.rs:102,110,580` and `viewer_3d/bench_track.rs:96,103,688` (m1).
- Problem: `HANDLE_HIT_RADIUS = 9.0`, `EDGE_HIT_WIDTH = 8.0` and the `nearest` closure are still defined twice. `crate::bench` already holds the shared hit helpers and has neither constant.
- Proposed fix: move them to `crate::bench`. The test order deliberately differs between the two views and stays as it is.
- Effort: low. Risk: low.

### Rust: bindings and format crates

**Settle the per-image array keyword names, and range-check `match_image_pairs_batch`** (carried, last read 2026-10-08)
> _Status (2026-10-09): **Partially done** — `match_image_pairs_batch` now checks list lengths, pair image indexes and camera indexes (negative or out of range) and raises `ValueError` instead of `PanicException`; it still takes `int64` camera indexes, branch `hygiene-fix-1009-03-pair-index-range-check`; the keyword spellings (`camera_indexes`/`camera_indices`/`image_camera`, `quaternions_wxyz`/`quats_wxyz`, `translations`/`translations_xyz`, `obs_*`/`track_*_indexes`) and the switch to `uint32` still need a maintainer decision._
- Location (the `io/` binding directory is now `fileio/`):
  - `crates/sfmtool-py/src/matching/image.rs:170-197`
  - `geometry/bundle_adjust.rs:403-409`
  - `patches/views.rs:338`
  - `fileio/camrig.rs:50,98`
  - `fileio/sfmr.rs:204`, `reconstruction/sfmr_reconstruction.rs:702`
  - `geometry/convention.rs:38-147`
- Problem (counts are parameter lines in `sfmtool-py/src`):
  - The camera index per image is `camera_indexes` (uint32) in `CameraViews`, `write_camrig` and the `.sfmr` getter. It is `image_camera` in `bundle_adjust` and `cell_plane_normals` (2 parameters), and `camera_indices` as `int64` in `match_image_pairs_batch` (`image.rs:183`). The range check is in place (m2).
  - Rotations are `quaternions_wxyz` on about 17 parameter lines in 13 files and `quats_wxyz` on 5, all in `geometry/convention.rs`.
  - Translations are `translations` on about 15 parameter lines and `translations_xyz` on 7. One `.sfmr` column still has both names: `read_sfmr()` returns `translations_xyz`, and the getter is `.translations`.
  - Observation arrays are `obs_image` (4) and `obs_point` (3) in some functions and `track_image_indexes` (8) and `track_point_indexes` (7) in others.
- Proposed fix:
  - Add a glossary entry naming `camera_indexes`, `quaternions_wxyz` and `translations_xyz`, which match the file formats.
  - Make `match_image_pairs_batch` take uint32.
  - Add a registration test that refuses the retired spellings.
- Effort: medium. Risk: medium, because `bundle_adjust` and `CameraViews` have keyword callers.

**Make the format binding signatures match, and decide the sweep bindings' future** (carried, last read 2026-10-08)
- Location: `crates/sfmtool-py/src/spatial/kdf.rs:758-772` (`write_kdf`) and `:949` (`verify_kdf`), `crates/sfmtool-py/src/matching/sweep.rs` (404 lines, 6 functions), and `crates/sfmtool-py/src/geometry/translation_averaging.rs:442-446`.
- Problem:
  - `write_kdf(forest, path, …)` still puts the data first, while every sibling and the Rust function put the path first. It still uses `compression_level` where the siblings use `zstd_level`. m3 check 14 shows `.kdf` as the one row of the format I/O table that differs.
  - `verify_kdf` still returns a dict of counts and raises, while the other four return `(is_valid, errors)`. `specs/python-bindings.md` lists `verify_kdf` but states no verify contract.
  - The six sweep bindings still have no callers outside `tests/matching/test_sweep_matching.py`, and keep the keywords `max_angle_diff`, `min_tri_angle`, `size_ratio_min`. They are now the largest source of Python-reachable panics: `k`/`r`/`t` of the wrong shape and short keypoint or descriptor arrays raise `PanicException` (m2 check 12, items 1 and 2).
  - The five `translation_averaging` constants are no longer unreferenced: the binding exports them as module attributes and uses three of them as signature defaults. That sub-item is closed.
- Proposed fix:
  - Change `write_kdf` to take `path` first, with `zstd_level`.
  - State the verify contract in `specs/python-bindings.md`.
  - Either drop the sweep bindings, or rename them to the `match_image_pair` spellings with one shared pair-geometry parser that validates shapes.
- Effort: low. Risk: low–medium (`write_kdf` has 33 references in `scripts/`, `tests/` and `src/`).

**Shrink rationale in the bundle-adjust and resect binding docs, and share the error closures** (carried, last read 2026-10-08)
> _Status (2026-10-09): **Partially done** — the `bundle_adjust` and `resect_images` binding docstrings now give the Python contract (args, shapes, returns, raises) and point to their specs for the reasoning; `helpers::value_err` / `os_err` added; the eight single-arm `*_err_to_py` helpers are removed in favour of `value_err`, and the inline closures in those files and in `helpers.rs`, `bundle_adjust.rs`, `resect_images.rs` and `edited.rs` use the helpers. Remaining: the inline `PyValueError::new_err(e.to_string())` / `PyIOError::new_err(e.to_string())` closures in the other `sfmtool-py` files (about 140 `new_err(e.to_string())` or `new_err(format!("{e}"))` sites across 44 files, counting the variant-mapping helpers), branch `hygiene-fix-1009-14-binding-doc-helpers`._
- Location: `crates/sfmtool-py/src/`, 46 files (by directory: `geometry` 12, `analysis` 7, `fileio` 6, `reconstruction` 6, `bench` 4, `matching` 3, `sift` 3, `patches` 2, `flow`, `helpers.rs` and `workspace.rs` 1 each). `helpers::value_err` and `os_err` are at `helpers.rs:17,23`.
- Problem: 132 `new_err(e.to_string())` or `new_err(format!("{e}"))` sites remain (was about 140): 92 `PyValueError::new_err(e.to_string())` and 19 `PyIOError`/`PyOSError`, against 72 uses of `value_err`/`os_err`. The `bundle_adjust` docstring has grown back to a 149-line block at `geometry/bundle_adjust.rs:215` (m1 check 1), mostly the Args reference, which m1 does not flag.
- Proposed fix: replace the remaining inline closures with `value_err` / `os_err`, keeping only the helpers that map by variant.
- Effort: low. Risk: low.

**Split `kdf-format/src/read.rs` by job** (carried, last read 2026-10-08)
- Location: `crates/sfmtool-kdf-format/src/read.rs`, 1,663 lines (was 1,679).
- Problem: one file still holds five jobs:
  - raw ZIP I/O (28–199, 1209–1237, 1558–1663)
  - the `KdfFile` API (`impl KdfFile` at 200–1208)
  - content verification (about 866–997), beside an existing `verify.rs`
  - metadata validation (1238–1464)
  - node decoding (1465–1557)

  `read/` holds only `profiling.rs`.
- Proposed fix: add `read/integrity.rs`, `read/metadata.rs` and `read/raw.rs` as child modules, beside the existing `read/profiling.rs`.
- Effort: medium. Risk: low.

**Rule on `indexes` / `indices` and on count names** (carried, last read 2026-10-08)
- Location:
  - Core: `track_image_indices` (`reconstruction/data.rs:562`, `data/point_set.rs:356`, `edited.rs:865`), `subset_by_image_indices(image_indices)` (`reconstruction/edit.rs:288`).
  - Python: `SfmrReconstruction.subset_by_image_indices`, `src_indices_for_tile` (`spherical/tile_source_stack.rs:203`), `num_images`/`num_levels`/`num_trees`/`n_tiles`/`n_contributors`.
- Problem:
  - `indexes` is the majority (m1 check 5): core 940 lines in 122 files against 319 in 76 (was 812 and 247); binding source 554 against 102; binding keywords 12 against 3 (`match_image_pairs_batch(camera_indices)`, `subset_by_image_indices(image_indices)`, `delete_points(indices)`). It is the word the file formats use. The glossary still has no entry.
  - The core method `track_image_indices(point)` means something different from the `.sfmr` column `track_image_indexes`.
  - Binding method names: 13 `*_count`, 3 `num_*` and 2 `n_*`.
  - The KDF tree classes still have `.len()` (`spatial/kdf.rs:279`, `kdforest.rs:155`, `kdtree.rs:80,361`) but no `__len__`.
- Proposed fix:
  - Add glossary entries for `indexes` and `*_count`.
  - Rename the minority names (`subset_by_image_indices` has 16 Python references in 12 files, was 13 in 6).
  - Add `__len__`.
- Effort: low to decide, medium to rename. Risk: medium, because the names are public.

### Rust: the viewer (leftover)

**Rename `SplitSettings` to `PieceSettings`** (carried, last read 2026-10-08)
- Location: `crates/sfm-explorer/src/bench.rs:2716` (14 references).
- Problem: the 2026-10-18 Superseded status on the bench `frame` finding left this rename open, and no open finding tracked it. The core calls the unit a piece (`piecewise.rs`, glossary "piecewise refinement").
- Proposed fix: rename the struct and its references.
- Effort: low. Risk: low.

### Python and test layout

**`sfm merge` fails on two `embedded_patches` inputs** (carried, last read 2026-10-08)
- Location: `src/sfmtool/merge/reconstructions.py`, `_merge_images`: `r_ft_hashes = recon.feature_tool_hashes` and `r_sift_hashes = recon.sift_content_hashes` (353–354), then `merged_feature_tool_hashes.append(r_ft_hashes[old_idx])` and `r_sift_hashes[old_idx]` (398–399); `_create_merged_reconstruction` passes them to `clone_with_changes` (520–521).
- Problem: an `embedded_patches` reconstruction has no `feature_tool_hashes` or `sift_content_hashes` (both getters return `None`), so merging two of them raises `TypeError: 'NoneType' object is not subscriptable` before any merging is done. The lines are unchanged at `929b62f0`. `_merged_readings` (545) now handles a source with no feature indexes, but the image-hash lines do not. The merge's point correspondences and pose refinement also read `.sift` files by feature index, which an `embedded_patches` input has none of. `tests/test_merge.py` has no `embedded` test.
- Proposed fix: decide whether `sfm merge` supports `embedded_patches` inputs. If not, refuse them up front with a message naming the input; if so, carry `image_file_hashes` and `keypoints_xy` instead of the `.sift` columns and match on the inline keypoints. Add a test either way.
- Effort: low to refuse, medium to support. Risk: low.

**Stop importing underscore names across subpackages, and enforce both naming rules with a test** (carried, last read 2026-10-08)
> _Status (2026-10-08): **Superseded** — the rule this finding assumed changed after the snapshot. AGENTS.md § "Python names and privacy" (PR #861) now gives `_` the standard meaning, internal to `sfmtool`, and lets code inside `src/sfmtool/` import an internal name from any subpackage, so the 42 cross-package imports are allowed. PR #868 enforces the part of the rule about the extension: `tests/test_module_layout.py` fails on a path into `sfmtool._sfmtool` from `tests/`, `scripts/` or `docs/`. Still open under the new rule: scripts that import internal Python modules (`sfmtool._workspace_image.read_workspace_image` at `scripts/add_image_to_tracks/harness.py:367`, `misses.py:47` and `scripts/track_at_pixel/context.py:160`; `sfmtool._cluster_patches._run_cluster_patches` and `feature_match._run._write_clusters_matches` at `scripts/track_at_pixel/dataset.py:218,221`; `feature_match._flow_matching` at `scripts/benchmark_flow_matching.py:114`), extending the layout test to them, and choosing which plain-named modules in module-path subpackages are helpers that should take `_` (`colmap/db_builders.py`, `merge/correspondences.py`, `merge/pose_refinement.py`, `camrig/pattern.py`, `rig/frames.py`, `sift/draw.py`, `motion/constants.py`, `flow_stats.py`, `ratio_band.py`, `analyze/point_or_bearing.py`)._

> _Status (2026-10-10): **Partially done** — `scripts/add_image_to_tracks/harness.py`, `misses.py` and `scripts/track_at_pixel/context.py` no longer import `sfmtool._workspace_image`; they read photographs with the public `sfmtool.fileio.read_image_rgb`. Still open: `scripts/track_at_pixel/dataset.py:218,221` (`_cluster_patches._run_cluster_patches`, `feature_match._run._write_clusters_matches`), `scripts/benchmark_flow_matching.py:115` (`feature_match._flow_matching`), extending the layout test to scripts, and the module-name question above. Branch `python-images-via-rust`._

> _Status (2026-10-31): **Partially done** — the scripts part is done: `scripts/benchmark_flow_matching.py` imports `flow_match_sequential` from `sfmtool.feature_match`, which now re-exports it; `scripts/track_at_pixel/dataset.py` runs `sfm cluster-patches` on its clusters file instead of calling `_cluster_patches._run_cluster_patches`, and writes that file with the new public `sfmtool.fileio.write_clusters_matches` (`write_clusters_matches` in `sfmtool-matches-format`) instead of `feature_match._run._write_clusters_matches`, which `sfm match --cluster` now writes through as well; and `tests/test_module_layout.py` parses every script's imports and fails on an internal `sfmtool` module or name, with no exceptions. Still open: the module-name question above. PR #1088._

> _Status (2026-11-08): **Partially done** — unchanged; the module-name question above is still open. Of the examples below, `rig/config.py`'s `_load_rig_config` no longer exists, since `rig_config.json` is read only by `sfm camrig cp` and the other commands refuse it. Branch `remove-rig-config`._
- Location: the ten plain-named modules in module-path subpackages, all still present: `src/sfmtool/colmap/db_builders.py`, `merge/correspondences.py`, `merge/pose_refinement.py`, `camrig/pattern.py`, `rig/frames.py`, `sift/draw.py`, `motion/constants.py`, `motion/flow_stats.py`, `motion/ratio_band.py`, `analyze/point_or_bearing.py`.
- Problem: only the module-name question remains. m1 check 5 finds 0 plain-named modules in the re-exporting subpackages and 33 command modules matching 33 `COMMANDS` rows, so the existing rule holds; it does not decide whether these ten module-path helpers should take `_`.
- Proposed fix: choose, module by module, which of the ten are helpers that take `_` and which are public module paths, and record the choice in `specs/STANDARDS.md` § "Python names mark what is public".
- Effort: low. Risk: low, unless a module chosen as public is renamed.

**Share the flow display's advection hits, and choose one separate-file naming** (carried, last read 2026-10-08)
- Location: `src/sfmtool/visualization/_flow_display.py`, 732 lines (was 736); `draw_flow_visualization` runs 161–408.
- Problem:
  - The in-bounds mask, nearest-feature lookup and index remap still appear three times (257–275, 434–452, 528–545).
  - The distance loop and green/red split still appear twice (323–341, 510–526).
  - `sfm flow --separate` writes `<stem>_A`/`<stem>_B` (`_flow_display.py:728-729`), while `sfm epipolar --separate` writes `<path>`/`<stem>_other` (`_epipolar_display.py:613`).
- Proposed fix: compute one `_FlowHits` in the driver, following the epipolar split (#793), and choose one naming for both commands.
- Effort: low, plus the naming decision. Risk: low; the file naming is user-visible.

**Extract the per-image step from `undistort_reconstruction_images`** (carried, last read 2026-10-08)
- Location: `src/sfmtool/_undistort_images.py`, 691 lines (was 613). The function runs from line 236 to 691 (456 lines, was 401), and its loop (344–524) is marked "Step 1" to "Step 6".
- Problem: one function still does all of this:
  - warps each image and writes it
  - makes the thumbnail
  - transforms and writes the SIFT features, and reads back their hashes
  - remaps tracks, and builds the cameras and rig frames
  - assembles the `.sfmr` through `colmap.io._build_sfmr_data_dict` (632)
- Proposed fix: extract a `_undistort_one_image` helper and an `.sfmr` assembly helper, guarded by byte-identical output on `seoul_bull`.
- Effort: medium. Risk: low–medium.

**Parametrize the spline-camera binding tests** (carried, last read 2026-10-08)
- Location: `tests/rust_bindings/geometry/test_sfmtool_pinhole_rust_bindings.py` (320 lines, 18 tests) and `test_sfmtool_fisheye_rust_bindings.py` (297 lines, 16 tests).
- Problem: they share 67 long lines (m1 check 3, was 59). The parallel tests and helpers differ only in the model name and the `rho_max`/`theta_max` key.
- Proposed fix: one module parametrized over a model description. Keep the seam tests in each file.
- Effort: low–medium. Risk: low.

## Closed since the last snapshot

No open finding of the 2026-10-08 report is closed by this run's numbers. One sub-item inside a carried finding closed: the five `translation_averaging` constants are now exported by the binding and three are signature defaults (see "Make the format binding signatures match").

The 17 findings that already carried a Done or Superseded status in the 2026-10-08 report are not carried: the root-name collisions (#868), `unreachable_pub` (#1122), the `MemberStatus` copy (#870), the `translate_patch_to_pixel` docstring, one shared quantile, the remap helpers, `focal_is_releasable` (#898), the bench `frame` words (Superseded, #986), the MCP nested keys, the WGSL prelude, the retired wire words, the format error table, the XXH128 parser, the dead Python `merge_points_and_tracks`, the `.sfmr` path check, the `embed-patches` help, and the minor cleanup.

Two leftovers named in those closed statuses had no open finding; the `group_point_correspondences` rename is now in "Have the point-correspondence bindings take the reconstructions", and the other is carried as "Rename `SplitSettings` to `PieceSettings`" above.

## Design topics carried forward

These are proposals carried over from older snapshots, not hygiene findings. All three are still open.

- **A — Camera bookmarks:** `bookmark` appears nowhere in `crates/sfm-explorer/src`; `specs/gui/viewport-navigation.md:904` says "No viewpoint bookmarking exists."
- **B — `sfm xform --crop`:** no file in `src/sfmtool/xform/` mentions `crop`, and no xform spec names a `--crop` option.
- **C — Pose-aware per-tile source stacks:** `PerSphericalTileSourceStack` still builds only rotation-only stacks (`build_rotation_only`, `per_tile_source_stack.rs:273`); `WarpMap::build_with_pose_impl` exists (`camera/warp_map.rs:325`).

## Explicitly not flagged

These are measurements, not verdicts: this repo has repeatedly grown a cleared file 30-90% within a month of clearing it. Each figure was taken on 2026-11-13 unless marked; the next snapshot re-runs them.


- `xform/_arg_parser.py`: 878 lines (was 834); `_TRANSFORM_OPTIONS` 34 entries (was 33); longest function `parse_xform_args` 60 lines (measured 2026-11-13)
- `visualization/_epipolar_display.py`: 818 lines (was 821); dispatcher `draw_epipolar_visualization` 189 lines, next-longest `_sweep_matches` 88 (measured 2026-11-13)
- `_embed_patches.py`: 869 lines (was 837); `embed_patches` 440 lines (was 482) (measured 2026-11-13)
- `colmap/io.py`: 1,062 lines (was 886, +20%), longest `save_colmap_binary` 145; `compare/_core.py` 832 (was 818), longest `compare_reconstructions` 314; `_densify.py` 781 (was 765), longest `triangulate_new_tracks` 272; `analyze/summary.py` 946 (was 730, +30%), longest `print_reconstruction_summary` 184 (measured 2026-11-13)
- `tests/conftest.py`: 1,394 lines (was 1,269); longest `build_reconstruction_at_poses` 145 (measured 2026-11-13)
- `reconstruction/edited.rs`: 1,980 lines (was 1,693), longest `build_point_set` 168 (was ~110); `camera/refit_intrinsics.rs` 1,667 (was 1,461), longest `refit_radial_correction` 154 (was ~132) (measured 2026-11-13)
- `focal_vote.rs` (1,752 lines, longest `rotation_votes` 123), `reconstruction_growth.rs` (1,392, longest `grow_reconstruction` 137) and `cluster_census.rs` (886, longest `measure_clusters` 102): all under 193 lines (measured 2026-11-13)
- `try_localize_patch_keypoints_with_basis`: no longer exists; `patch/keypoint_localize.rs` is 973 lines and its longest function, `render_context`, is 122 lines (measured 2026-11-13)
- The `prof.rs` family: at most 17 shared lines per pair (`keypoint_subpixel` / `view_selection`, was 10), m1 check 4; `profiling.rs` 189 lines (measured 2026-11-13)
- `polar.rs` / `sweep.rs`: 14 shared comment lines of 40 or more characters by this run's count (9 shared doc lines on 2026-10-08 by a different count); 665 and 504 lines (measured 2026-11-13)
- GPU uniform structs marked "Must match WGSL" in core: 2 such comments (measured 2026-11-13)
- The `numeric` median: `MEDIAN_ALLOWLIST` has 19 entries and `QUANTILE_ALLOWLIST` 4; `cargo test -p sfmtool-core --lib numeric` passed, 26 tests (m3 check 18). m3 check 18 also keeps 12 inline medians and percentiles that the name scan does not see (measured 2026-11-13)
- MCP registration: 105 catalog entries (was 90: read 17, viewer 22, edit 20, bench 36, background 4, input 6), tied together by the exact-set test at `mcp/tests/catalog.rs:37` (not run in this pass) (measured 2026-11-13)
- The MCP `item` / `track` argument split: 4 / 24 tools (was 4 / 21) (measured 2026-11-13)
- `get_bench`, `undo`, `redo` and `jump_to_version` taking a `reconstruction_label`: no figure; the #864 acquittal is not re-measured
- DirectManipulation examples: 76 shared long lines (was 106), m1 check 3 (measured 2026-11-13)
- Shader uniform blocks: `ReconUniforms` once in `common.wgsl`; 6 shaders (`bench_track`, `distorted_quad`, `image_quad`, `patch`, `points`, `track_ray`) each declare their own `Uniforms`, plus `BgUniforms`, `EdlUniforms`, `FrustumUniforms` and `TargetUniforms` (measured 2026-11-13)
- `_s` and `_w` suffixes on binding names: 3 (`flip_camera_poses_s`, `relative_poses_conjugate_s`, `world_rotate_w`) (measured 2026-11-13)
- `_py`, `_rs`, `_impl`, `_inner` and `_raw` suffixes on Python-visible names: 0, m1 check 6 (measured 2026-11-13)
- Shared argument text across the `PatchCloud` methods: no figure; not re-measured
- `sfmtool-py/src/reconstruction/clone.rs`: 1,891 lines; longest `apply_image_field` 254 lines (was 208) (measured 2026-11-13)
- `sfmtool-progress`: 29.8% comment lines (was 29.7%), m1 check 1; its `lib.rs` crate doc is 110 `//!` lines (was 97) (measured 2026-11-13)
- Bench `anchor`: 140 lines matching `anchor`/`anchors`/`anchored` in core `bench/`, viewer `bench*` and py `bench*` (185 uses on 2026-10-08 by a different scope) (measured 2026-11-13)
- `_global_sfm` / `_incremental_sfm`: 29 shared lines at 30 or more characters, under 20 at 40 or more (7 on 2026-10-08), m1 check 4 (measured 2026-11-13)

## Top 3

1. **Delete `apply_se3_to_camera_poses` and `filter_tracks_by_point_mask`.** Low effort. It removes two public bindings, two core functions and their tests that nothing outside a registration test calls, and with them two of the `PanicException`s Python can reach today.
2. **Compute the hand-written medians and percentiles with `sfmtool_core::numeric`.** Low effort and mechanical: twelve sites, four of which give a different median from the shared one on an even count and three of which panic or go unspecified on NaN. Extending the name scan to these shapes stops the next copy.
3. **Validate a `.matches` file's structure when it is read, with the checks written once.** Medium effort. Today a corrupt `.matches` file opens without error and `MatchesFile.select_clusters` then raises `PanicException`. Moving the write-time checks into one `validate.rs` that write, read and verify share fixes that and the drift where `verify_matches` passes files `write_matches` refuses.

## Coverage

| Unit | Lines | Last deep read | Commit |
|---|---|---|---|
| `crates/sfm-explorer (rest)` | 3407 | never | - |
| `crates/sfm-explorer/src (rest 1)` | 4565 | never | - |
| `crates/sfm-explorer/src (rest 2)` | 4330 | never | - |
| `crates/sfm-explorer/src (rest 3)` | 4204 | never | - |
| `crates/sfm-explorer/src/action_log` | 4126 | never | - |
| `crates/sfm-explorer/src/app` | 3087 | never | - |
| `crates/sfm-explorer/src/background (rest)` | 3547 | never | - |
| `crates/sfm-explorer/src/background/tests` | 3886 | never | - |
| `crates/sfm-explorer/src/bench (rest)` | 4195 | never | - |
| `crates/sfm-explorer/src/bench/geometry` | 1681 | never | - |
| `crates/sfm-explorer/src/bench/live` | 1483 | never | - |
| `crates/sfm-explorer/src/bench/nearby_tracks` | 1064 | never | - |
| `crates/sfm-explorer/src/bench/surface_tiling` | 1273 | never | - |
| `crates/sfm-explorer/src/bench/tests` | 2009 | never | - |
| `crates/sfm-explorer/src/bench/track_at_pixel` | 1121 | never | - |
| `crates/sfm-explorer/src/bundle_adjust_prompt` | 1519 | never | - |
| `crates/sfm-explorer/src/camera_lock` | 1185 | never | - |
| `crates/sfm-explorer/src/camera_model_proposal` | 4777 | never | - |
| `crates/sfm-explorer/src/cluster_patches` | 1583 | never | - |
| `crates/sfm-explorer/src/display_transform` | 1180 | never | - |
| `crates/sfm-explorer/src/dock` | 1978 | never | - |
| `crates/sfm-explorer/src/fonts` | 1795 | never | - |
| `crates/sfm-explorer/src/image_browser` | 1520 | never | - |
| `crates/sfm-explorer/src/image_detail (rest)` | 3998 | never | - |
| `crates/sfm-explorer/src/image_detail/intrinsics` | 3931 | never | - |
| `crates/sfm-explorer/src/image_detail/tests` | 4308 | never | - |
| `crates/sfm-explorer/src/intrinsics_detail (rest)` | 4900 | never | - |
| `crates/sfm-explorer/src/intrinsics_detail/tests` | 2812 | never | - |
| `crates/sfm-explorer/src/layout` | 2864 | never | - |
| `crates/sfm-explorer/src/mcp (rest 1)` | 2958 | never | - |
| `crates/sfm-explorer/src/mcp (rest 2)` | 3316 | never | - |
| `crates/sfm-explorer/src/mcp/bench` | 2092 | never | - |
| `crates/sfm-explorer/src/mcp/edit` | 1188 | never | - |
| `crates/sfm-explorer/src/mcp/input` | 1097 | never | - |
| `crates/sfm-explorer/src/mcp/mod` | 2677 | never | - |
| `crates/sfm-explorer/src/mcp/tests (rest)` | 4535 | never | - |
| `crates/sfm-explorer/src/mcp/tests/bench` | 5348 | never | - |
| `crates/sfm-explorer/src/mcp/tests/catalog` | 1628 | never | - |
| `crates/sfm-explorer/src/mcp/tests/edit` | 4308 | never | - |
| `crates/sfm-explorer/src/mcp/tests/input` | 1678 | never | - |
| `crates/sfm-explorer/src/mcp/tests/layout` | 1019 | never | - |
| `crates/sfm-explorer/src/mcp/tests/logged` | 1129 | never | - |
| `crates/sfm-explorer/src/mcp/tests/render` | 2345 | never | - |
| `crates/sfm-explorer/src/mcp/tests/view` | 1129 | never | - |
| `crates/sfm-explorer/src/mcp/tests/widgets` | 1159 | never | - |
| `crates/sfm-explorer/src/mcp/tools (rest)` | 2523 | never | - |
| `crates/sfm-explorer/src/mcp/tools/catalog` | 4775 | never | - |
| `crates/sfm-explorer/src/mcp/widgets` | 1094 | never | - |
| `crates/sfm-explorer/src/platform` | 1699 | never | - |
| `crates/sfm-explorer/src/progress` | 1032 | never | - |
| `crates/sfm-explorer/src/scene_graph (rest)` | 2605 | never | - |
| `crates/sfm-explorer/src/scene_graph/tests` | 4209 | never | - |
| `crates/sfm-explorer/src/scene_renderer (rest)` | 4951 | never | - |
| `crates/sfm-explorer/src/scene_renderer/pipelines` | 1928 | never | - |
| `crates/sfm-explorer/src/scene_renderer/upload (rest)` | 2856 | never | - |
| `crates/sfm-explorer/src/scene_renderer/upload/tests` | 3468 | never | - |
| `crates/sfm-explorer/src/shaders` | 1622 | never | - |
| `crates/sfm-explorer/src/sift_index` | 1888 | never | - |
| `crates/sfm-explorer/src/state (rest)` | 4385 | never | - |
| `crates/sfm-explorer/src/state/edits (rest)` | 4542 | never | - |
| `crates/sfm-explorer/src/state/edits/tests` | 3505 | never | - |
| `crates/sfm-explorer/src/track_table_panel (rest)` | 4395 | never | - |
| `crates/sfm-explorer/src/track_table_panel/tests` | 4186 | never | - |
| `crates/sfm-explorer/src/track_view (rest)` | 1530 | never | - |
| `crates/sfm-explorer/src/track_view/body (rest)` | 4511 | never | - |
| `crates/sfm-explorer/src/track_view/body/mod` | 2860 | never | - |
| `crates/sfm-explorer/src/track_view/body/tests (own file)` | 6604 | never | - |
| `crates/sfm-explorer/src/track_view/body/tests (rest)` | 850 | never | - |
| `crates/sfm-explorer/src/viewer_3d (rest)` | 4221 | never | - |
| `crates/sfm-explorer/src/viewer_3d/bench_track` | 1226 | never | - |
| `crates/sfm-explorer/src/viewer_3d/hud` | 1454 | never | - |
| `crates/sfm-explorer/src/viewer_3d/mod` | 1807 | never | - |
| `crates/sfm-explorer/src/viewer_3d/tests` | 2522 | never | - |
| `crates/sfmtool-archive-io` | 2030 | never | - |
| `crates/sfmtool-camrig-format` | 2135 | never | - |
| `crates/sfmtool-colmap/src (rest)` | 2428 | never | - |
| `crates/sfmtool-colmap/src/colmap_db` | 2871 | never | - |
| `crates/sfmtool-core (rest)` | 1744 | never | - |
| `crates/sfmtool-core/src (rest)` | 4592 | never | - |
| `crates/sfmtool-core/src/analysis (rest 1)` | 2990 | never | - |
| `crates/sfmtool-core/src/analysis (rest 2)` | 3695 | never | - |
| `crates/sfmtool-core/src/analysis/alignment` | 1688 | never | - |
| `crates/sfmtool-core/src/analysis/cluster_census` | 2784 | never | - |
| `crates/sfmtool-core/src/analysis/covered_by_finer` | 1006 | never | - |
| `crates/sfmtool-core/src/analysis/infinity` | 2682 | never | - |
| `crates/sfmtool-core/src/analysis/pick_images (rest)` | 3969 | never | - |
| `crates/sfmtool-core/src/analysis/pick_images/source` | 1055 | never | - |
| `crates/sfmtool-core/src/analysis/pick_images/tests` | 2160 | never | - |
| `crates/sfmtool-core/src/analysis/track_table` | 2606 | never | - |
| `crates/sfmtool-core/src/bench (rest)` | 4104 | never | - |
| `crates/sfmtool-core/src/bench/evaluate` | 2513 | never | - |
| `crates/sfmtool-core/src/bench/fit` | 1446 | never | - |
| `crates/sfmtool-core/src/bench/nearby (rest 1)` | 2818 | never | - |
| `crates/sfmtool-core/src/bench/nearby (rest 2)` | 3108 | never | - |
| `crates/sfmtool-core/src/bench/nearby/find` | 1111 | 2026-11-13 | 929b62f0 |
| `crates/sfmtool-core/src/bench/normal (rest)` | 2704 | never | - |
| `crates/sfmtool-core/src/bench/normal/grown_plane (rest 1)` | 2930 | never | - |
| `crates/sfmtool-core/src/bench/normal/grown_plane (rest 2)` | 2639 | never | - |
| `crates/sfmtool-core/src/bench/normal/grown_plane/render` | 1119 | never | - |
| `crates/sfmtool-core/src/bench/search` | 1002 | never | - |
| `crates/sfmtool-core/src/bench/steps` | 3577 | never | - |
| `crates/sfmtool-core/src/bench/tests (own file)` | 8368 | never | - |
| `crates/sfmtool-core/src/bench/tests (rest)` | 3724 | never | - |
| `crates/sfmtool-core/src/bench/tests/grown_plane_normal` | 2017 | never | - |
| `crates/sfmtool-core/src/bench/tests/reference_view` | 2294 | never | - |
| `crates/sfmtool-core/src/bench/track` | 1540 | never | - |
| `crates/sfmtool-core/src/bench/track_at_pixel` | 3766 | never | - |
| `crates/sfmtool-core/src/camera (rest)` | 4446 | never | - |
| `crates/sfmtool-core/src/camera/distortion (rest)` | 3173 | never | - |
| `crates/sfmtool-core/src/camera/distortion/kernels` | 1836 | never | - |
| `crates/sfmtool-core/src/camera/distortion/tests` | 4258 | never | - |
| `crates/sfmtool-core/src/camera/image` | 2857 | never | - |
| `crates/sfmtool-core/src/camera/intrinsics` | 4102 | never | - |
| `crates/sfmtool-core/src/camera/refit_intrinsics` | 2555 | never | - |
| `crates/sfmtool-core/src/camera/remap` | 2926 | never | - |
| `crates/sfmtool-core/src/camera/report` | 2097 | never | - |
| `crates/sfmtool-core/src/camera/warp_map` | 2103 | never | - |
| `crates/sfmtool-core/src/features (rest)` | 3579 | never | - |
| `crates/sfmtool-core/src/features/feature_match (rest)` | 3888 | never | - |
| `crates/sfmtool-core/src/features/feature_match/polar` | 1180 | never | - |
| `crates/sfmtool-core/src/features/kdforest (rest)` | 4575 | never | - |
| `crates/sfmtool-core/src/features/kdforest/constellation` | 2174 | never | - |
| `crates/sfmtool-core/src/features/optical_flow (rest)` | 3938 | never | - |
| `crates/sfmtool-core/src/features/optical_flow/gpu` | 4138 | never | - |
| `crates/sfmtool-core/src/features/sift (rest)` | 4086 | never | - |
| `crates/sfmtool-core/src/features/sift/files` | 1196 | never | - |
| `crates/sfmtool-core/src/features/sift/gray` | 1049 | never | - |
| `crates/sfmtool-core/src/features/sift/scale_space` | 1109 | never | - |
| `crates/sfmtool-core/src/geometry (rest 1)` | 3428 | never | - |
| `crates/sfmtool-core/src/geometry (rest 2)` | 3831 | never | - |
| `crates/sfmtool-core/src/geometry/absolute_pose` | 1179 | never | - |
| `crates/sfmtool-core/src/geometry/affine_factorization` | 1027 | never | - |
| `crates/sfmtool-core/src/geometry/bundle_adjust (own file)` | 5554 | never | - |
| `crates/sfmtool-core/src/geometry/bundle_adjust/tests (own file)` | 8878 | never | - |
| `crates/sfmtool-core/src/geometry/bundle_adjust/tests (rest)` | 2076 | never | - |
| `crates/sfmtool-core/src/geometry/epipolar_estimation` | 1204 | never | - |
| `crates/sfmtool-core/src/geometry/estimate_intrinsics` | 1240 | never | - |
| `crates/sfmtool-core/src/geometry/focal_vote (rest)` | 3195 | never | - |
| `crates/sfmtool-core/src/geometry/focal_vote/column_scan` | 2835 | never | - |
| `crates/sfmtool-core/src/geometry/pose_verification` | 1170 | never | - |
| `crates/sfmtool-core/src/geometry/reconstruction_growth` | 2599 | never | - |
| `crates/sfmtool-core/src/geometry/relative_pose` | 1019 | never | - |
| `crates/sfmtool-core/src/geometry/resect_images` | 4211 | never | - |
| `crates/sfmtool-core/src/geometry/rotation_init` | 1385 | never | - |
| `crates/sfmtool-core/src/geometry/translation_averaging` | 1537 | never | - |
| `crates/sfmtool-core/src/patch (rest)` | 4857 | never | - |
| `crates/sfmtool-core/src/patch/blur_matched` | 1435 | never | - |
| `crates/sfmtool-core/src/patch/cell_plane_normals` | 1448 | never | - |
| `crates/sfmtool-core/src/patch/cloud` | 3636 | never | - |
| `crates/sfmtool-core/src/patch/cluster_refine (rest)` | 3708 | never | - |
| `crates/sfmtool-core/src/patch/cluster_refine/piecewise` | 2210 | never | - |
| `crates/sfmtool-core/src/patch/cluster_refine/tests` | 1520 | never | - |
| `crates/sfmtool-core/src/patch/keypoint_localize (rest)` | 4133 | never | - |
| `crates/sfmtool-core/src/patch/keypoint_localize/tests` | 3376 | never | - |
| `crates/sfmtool-core/src/patch/keypoint_subpixel` | 4097 | never | - |
| `crates/sfmtool-core/src/patch/member_coherence` | 3475 | never | - |
| `crates/sfmtool-core/src/patch/normal_refine` | 1889 | never | - |
| `crates/sfmtool-core/src/patch/reference_view` | 2406 | never | - |
| `crates/sfmtool-core/src/patch/self_similarity` | 3481 | never | - |
| `crates/sfmtool-core/src/patch/stored_bitmap` | 1243 | never | - |
| `crates/sfmtool-core/src/patch/view_selection` | 3208 | never | - |
| `crates/sfmtool-core/src/reconstruction (rest 1)` | 2773 | never | - |
| `crates/sfmtool-core/src/reconstruction (rest 2)` | 3073 | never | - |
| `crates/sfmtool-core/src/reconstruction/add_image_to_tracks` | 2732 | never | - |
| `crates/sfmtool-core/src/reconstruction/bundle_adjust (rest)` | 2928 | never | - |
| `crates/sfmtool-core/src/reconstruction/bundle_adjust/tests` | 3220 | never | - |
| `crates/sfmtool-core/src/reconstruction/data (rest)` | 4288 | never | - |
| `crates/sfmtool-core/src/reconstruction/data/tests` | 3291 | never | - |
| `crates/sfmtool-core/src/reconstruction/edit` | 1483 | never | - |
| `crates/sfmtool-core/src/reconstruction/edited` | 3432 | never | - |
| `crates/sfmtool-core/src/reconstruction/point_correspondence` | 1224 | never | - |
| `crates/sfmtool-core/src/reconstruction/prune_covered` | 1129 | never | - |
| `crates/sfmtool-core/src/reconstruction/switch_camera_model` | 2467 | never | - |
| `crates/sfmtool-core/src/reconstruction/triangulation (rest)` | 2828 | never | - |
| `crates/sfmtool-core/src/reconstruction/triangulation/points` | 2544 | never | - |
| `crates/sfmtool-core/src/reconstruction/triangulation/retriangulate` | 3038 | never | - |
| `crates/sfmtool-core/src/spherical (rest)` | 4417 | never | - |
| `crates/sfmtool-core/src/spherical/tile_rig` | 2730 | never | - |
| `crates/sfmtool-core/src/workspace` | 4070 | never | - |
| `crates/sfmtool-kdf-format/src (rest)` | 4796 | never | - |
| `crates/sfmtool-kdf-format/src/read` | 1869 | never | - |
| `crates/sfmtool-matches-format/src (rest)` | 4098 | 2026-11-13 | 929b62f0 |
| `crates/sfmtool-matches-format/src/tests` | 4560 | never | - |
| `crates/sfmtool-matches-format/src/verify` | 1376 | never | - |
| `crates/sfmtool-progress` | 1908 | never | - |
| `crates/sfmtool-py (rest)` | 158 | never | - |
| `crates/sfmtool-py/src (rest)` | 4669 | never | - |
| `crates/sfmtool-py/src/analysis` | 4380 | 2026-11-13 | 929b62f0 |
| `crates/sfmtool-py/src/bench (rest 1)` | 2697 | never | - |
| `crates/sfmtool-py/src/bench (rest 2)` | 2491 | never | - |
| `crates/sfmtool-py/src/bench/normal` | 1410 | never | - |
| `crates/sfmtool-py/src/fileio` | 4100 | never | - |
| `crates/sfmtool-py/src/geometry (rest 1)` | 3293 | never | - |
| `crates/sfmtool-py/src/geometry (rest 2)` | 3212 | never | - |
| `crates/sfmtool-py/src/matching` | 2361 | never | - |
| `crates/sfmtool-py/src/patches` | 4369 | 2026-11-13 | 929b62f0 |
| `crates/sfmtool-py/src/reconstruction (rest)` | 4651 | never | - |
| `crates/sfmtool-py/src/reconstruction/sfmr_reconstruction` | 2136 | never | - |
| `crates/sfmtool-py/src/spatial` | 2391 | never | - |
| `crates/sfmtool-sfmr-format/src (rest)` | 4555 | never | - |
| `crates/sfmtool-sfmr-format/src/tests (rest 1)` | 4458 | never | - |
| `crates/sfmtool-sfmr-format/src/tests (rest 2)` | 3047 | never | - |
| `crates/sfmtool-sfmr-format/src/types` | 1440 | never | - |
| `crates/sfmtool-sfmr-format/src/write` | 1845 | never | - |
| `crates/sfmtool-sift-format` | 1869 | never | - |
| `scripts (rest 1)` | 4854 | never | - |
| `scripts (rest 2)` | 4549 | never | - |
| `scripts/bench_bars` | 1134 | never | - |
| `scripts/kdf_constellation_progressive_eval` | 2232 | never | - |
| `scripts/reference_view` | 2242 | never | - |
| `scripts/track_at_pixel (rest)` | 4123 | never | - |
| `scripts/track_at_pixel/anchors` | 2160 | never | - |
| `scripts/track_at_pixel/candidates` | 2432 | never | - |
| `src/sfmtool (rest 1)` | 3470 | never | - |
| `src/sfmtool (rest 2)` | 3376 | never | - |
| `src/sfmtool (rest 3)` | 3200 | never | - |
| `src/sfmtool/_commands (rest 1)` | 4275 | never | - |
| `src/sfmtool/_commands (rest 2)` | 4310 | never | - |
| `src/sfmtool/_commands/estimate_intrinsics` | 1106 | never | - |
| `src/sfmtool/_video_frames` | 1024 | never | - |
| `src/sfmtool/analyze` | 2702 | never | - |
| `src/sfmtool/camrig` | 1866 | never | - |
| `src/sfmtool/colmap` | 2550 | never | - |
| `src/sfmtool/compare` | 1251 | never | - |
| `src/sfmtool/feature_match` | 2309 | never | - |
| `src/sfmtool/merge` | 1267 | never | - |
| `src/sfmtool/motion` | 2224 | never | - |
| `src/sfmtool/rig` | 1403 | never | - |
| `src/sfmtool/sift` | 1564 | never | - |
| `src/sfmtool/strips` | 1608 | never | - |
| `src/sfmtool/visualization` | 2828 | never | - |
| `src/sfmtool/xform (rest 1)` | 2443 | never | - |
| `src/sfmtool/xform (rest 2)` | 2562 | never | - |
| `tests (rest 1)` | 4985 | never | - |
| `tests (rest 2)` | 4366 | never | - |
| `tests (rest 3)` | 4713 | never | - |
| `tests (rest 4)` | 4641 | never | - |
| `tests/camrig` | 1861 | never | - |
| `tests/conftest` | 1394 | never | - |
| `tests/matching` | 2877 | never | - |
| `tests/patch (rest 1)` | 3144 | never | - |
| `tests/patch (rest 2)` | 3066 | never | - |
| `tests/rig` | 2318 | never | - |
| `tests/rust_bindings (rest)` | 4982 | never | - |
| `tests/rust_bindings/analysis` | 4452 | never | - |
| `tests/rust_bindings/bench` | 4383 | never | - |
| `tests/rust_bindings/fileio` | 3081 | never | - |
| `tests/rust_bindings/geometry (rest 1)` | 3337 | never | - |
| `tests/rust_bindings/geometry (rest 2)` | 3398 | never | - |
| `tests/rust_bindings/geometry/test_bundle_adjust_rust_bindings` | 1718 | never | - |
| `tests/rust_bindings/matching` | 2247 | never | - |
| `tests/rust_bindings/reconstruction` | 4273 | never | - |
| `tests/sift` | 1886 | never | - |
| `tests/test_named_image_list_cli` | 1050 | never | - |
| `tests/test_pick_images_command` | 1093 | never | - |
| `tests/xform (rest 1)` | 3837 | never | - |
| `tests/xform (rest 2)` | 3522 | never | - |

## Slow tests examined

| Test | Seconds | Measured by | Examined | Commit | Outcome |
|---|---|---|---|---|---|
| `bench::tests::stored_bitmap::a_viewer_spin_and_resize_then_commit_store_the_render_of_the_frame_they_store` | 60 | CI test-linux-rust, over 60 s | 2026-11-13 | 929b62f0 | meets the standard |
| `bench::tests::stored_bitmap::a_viewer_tilt_then_commit_stores_the_render_of_the_frame_it_stores` | 60 | CI test-linux-rust, over 60 s | 2026-11-13 | 929b62f0 | meets the standard |
| `bench::tests::stored_bitmap::refine_normals_stores_the_render_of_the_frame_it_stores` | 60 | CI test-linux-rust, over 60 s | 2026-11-13 | 929b62f0 | finding "Run the refine-normals stored-bitmap test on a working grid the size of the patch" |
