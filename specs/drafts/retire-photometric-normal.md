# Retire the Photometric Normal

**Status:** Draft. Every caller of the photometric normal refinement (`patch::normal_refine::refine_patch_normal` and its batch form `refine_patch_cloud_normals`) switches to the grown-plane normal, and the photometric search is then deleted. The work is a sequence of pull requests, one per step below.

Built: step 1, the point list (`RefineNormalsOptions::point_indexes`, outcome `NotAsked`, refusal `PointOutOfRange`) and the agreement (`GrownPlaneSequence::agreement`, carried on `Turned` and on `LeftUnchanged(WeakTurn)`, with the report's `mean_agreement` over the turned points), described in [core/bench/refine-normals.md](../core/bench/refine-normals.md); step 2, the Python binding `sfmtool.bench.refine_normals`, described in the same spec's § "Python binding"; step 3, `sfm xform --refine-normals` running `sfmtool.bench.refine_normals` and taking no keys, described in [cli/reconstruction/xform/refine-normals-command.md](../cli/reconstruction/xform/refine-normals-command.md); step 4, `inspect --strips` running `sfmtool.bench.refine_normals` on the listed finite points and rendering the turned points' bitmaps, described in [cli/reconstruction/inspect-command.md](../cli/reconstruction/inspect-command.md) § "Feature source"; step 5, `compare --strips` running `sfmtool.bench.refine_normals` on each reconstruction's listed points and printing each one's mean agreement, described in [cli/reconstruction/compare-command.md](../cli/reconstruction/compare-command.md) § "Side-by-side patch strips (`--strips`)". Not built: steps 6 to 9.

Decided (2026-10-30): `sfm xform --refine-normals` keeps its name; `compare --strips` reports the grown plane's agreement score where it reported the mean ΔΦ; `sfm embed-patches` keeps its rounds of "refine normals, select views, refine keypoints" with only the normal step swapped. Whether `embed-patches` needs the normal step in every round is left for later.

Amends: [core/bench/refine-normals.md](../core/bench/refine-normals.md) (a point subset, a Python binding and an agreement score), [cli/reconstruction/xform/refine-normals-command.md](../cli/reconstruction/xform/refine-normals-command.md), [cli/reconstruction/embed-patches-command.md](../cli/reconstruction/embed-patches-command.md), [core/patch/patch-cloud.md](../core/patch/patch-cloud.md), [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating the normal", and the specs this draft deletes (§ [Step 8](#8-delete-the-photometric-search)).

Related: [grown-plane-normal.md](grown-plane-normal.md) designs the step every caller moves to. Once this draft's steps have shipped, the grown plane is the only normal refinement, and that draft should be filed as a standing spec.

## Purpose

The grown-plane normal fits one plane to the photographs view by view, and in review it gives better normals than the photometric search. The viewer already uses only the grown plane: Track View's *Grown Plane Normal*, *Refine All Normals* and the MCP tool `refine_all_normals` all run it. The command-line tools still use the photometric search through `PatchCloud.refine_normals`. Keeping both means two normal refinements to maintain, two sets of options, and two answers to the question of which way a patch faces.

## What the photometric normal is

`refine_patch_normal` searches the 2-DOF normal, coarse to fine, for the plane whose rendered patches agree most, scored by the consensus all-pairs mean ZNCC. Its code is `search.rs`, `consensus.rs`, `level.rs`, `view_subset/`, `fronto_cache/`, `prof.rs` and the `NormalRefineParams` / `NormalRefineResult` / `CacheMode` / `Objective` types in [patch/normal_refine](../../crates/sfmtool-core/src/patch/normal_refine.rs).

The same module also holds code that other features use and that stays: `ProjectedImage`, `PatchWindow`, `Sampler` / `SamplerChoice`, `Support` / `build_support`, `grid_bounds`, `grid_cell_centres`, `window_weights`, the `znormalize_*` functions, `viewing_angle` / `ViewingAngle` / `MIN_TILT_ANGLE_DEG`, `tangent_basis` / `exp_map_normal`, `FLAT_NORM_SQ_EPS` and `MIN_MASK_PIXELS`. `irls_view_weights` and `ConsensusScratch` stay too if `keypoint_localize`, `view_selection` or `member_coherence` still use them once the search is gone.

## Callers

| Caller | Where | Use |
|---|---|---|
| `sfm embed-patches` | [_embed_patches.py](../../src/sfmtool/_embed_patches.py), round 1 and rounds 2+ | `cloud.refine_normals(..., use_stored_keypoints=True)` before `select_views` and the keypoint passes |
| `sfm xform --refine-normals` | [xform/_refine_normals.py](../../src/sfmtool/xform/_refine_normals.py), [xform/_arg_parser.py](../../src/sfmtool/xform/_arg_parser.py) | `cloud.refine_normals` with about 15 search options, then bitmaps and observation readings |
| `inspect --strips` | [strips/_inspect.py](../../src/sfmtool/strips/_inspect.py) | refines the listed finite points and renders their bitmaps |
| `compare --strips` | [strips/_solve.py](../../src/sfmtool/strips/_solve.py), [strips/_compare.py](../../src/sfmtool/strips/_compare.py) | refines the listed points at the reprojected centres and prints the mean ΔΦ |
| `bench::fit_normal` | [bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), Python `sfmtool.bench.fit_normal` | a bench step with no viewer entry; `NormalEstimate::Photometric` |
| `PatchCloud.refine_normals` | [sfmtool-py patches/refine_normals.rs](../../crates/sfmtool-py/src/patches/refine_normals.rs) | the binding every Python caller above goes through |

## Steps

Each step is one pull request through the `implement-change` skill, and each updates the specs it touches.

### 1. A point subset and an agreement score in `bench::refine_normals`

`RefineNormalsOptions` gains an optional list of point indexes. A point left out keeps every column it had and is reported as not asked for. The default, no list, refines every finite point, as now.

`RefineNormalsOutcome::Turned` and `LeftUnchanged` gain the point's **agreement**: the mean robust grid ZNCC (`grid_zncc`) at the final plane over the views that joined it or had their shift fitted (`GrownPlaneView::Joined` and `GrownPlaneView::ShiftOnly`). `NaN` when no view has a score. The report gains its mean over the turned points that have one; a `WeakTurn` point's agreement is at a plane not written, and does not count. The grown plane computes no score at the starting normal, so there is no before-and-after difference.

### 2. A Python binding for `bench::refine_normals`

`sfmtool.bench.refine_normals(recon, images, *, point_indexes=None, <grown_plane_normal's keywords>, progress=None)` returns the new reconstruction and a report dict shaped as `sfmtool.bench.refit_points`' is: the counts (`read`, `turned`, `unchanged`, `low_confidence`, `none_joined`, `weak_turn`, `kept`, `at_infinity`, `not_asked`), `median_turn_deg` and `mean_agreement`, and `points`, one dict per point with its `outcome` and, where they apply, its `turn_deg` and `agreement`. `progress` is an `sfmtool.ProgressCounter`, as `PatchCloud.refine_normals` takes, so step 6 keeps its progress bar.

### 3. `sfm xform --refine-normals`

The transform calls the binding of step 2. `--refine-normals` keeps its name and takes no keys: the photometric options (`angular_range_deg`, `init_steps`, `refine_levels`, `resolution`, `objective`, `robust_iters`, `search_robust_iters`, `window`, `window_sigma`, `sampler`, `min_valid_fraction`, `min_views`, `cache`, `cache_supersample`, `quality`, `confidence`, `bitmaps`) are removed, and a key given is an error naming the change. `bench::refine_normals` renders a bitmap where the value stores one per point and writes the records a commit would, so the transform's own bitmap and reading code goes. The summary line reports the counts, the median turn and the mean agreement.

### 4. `inspect --strips`

Calls the binding with the listed finite points.

### 5. `compare --strips`

Calls the binding with the listed points of each reconstruction and prints the mean agreement of each (`normal refinement mean agreement ref=… target=…`) where it printed the mean ΔΦ. The photometric refinement anchored every view at the reprojected centre; the grown plane needs keypoints, so a reconstruction without them keeps its stored normals, with a message, as a reconstruction with no valid photoconsistency does today. `_REFINE_RANGE_DEG` and `_REFINE_STEPS` go.

### 6. `sfm embed-patches`

Each round's normal step calls the binding instead of `cloud.refine_normals`, and the rounds stay as they are. The options that only steer the photometric search go: `--obliquity-weight-power`, `--fronto-prior-weight` and `--refine-max-views`. To check before switching:

- that the reconstructions `to_embedded_patches` and `compact_to_embedded_patches` build carry keypoints and patch frames, which `bench::refine_normals` requires;
- that in rounds 2 and later the views `select_views` added are `in` observations of the tracks the grown plane reads;
- the output on copies of the seoul_bull and kerry_park ground truths against the current output, by normal error and by the bars the readings report.

### 7. Delete `fit_normal`

`bench::fit_normal`, `FitNormalOptions`, `NormalEstimate::Photometric`, the Python `sfmtool.bench.fit_normal`, and their tests and spec text.

### 8. Delete the photometric search

`PatchCloud.refine_normals` and its binding module; `refine_patch_normal`, `refine_patch_normal_reporting` and `refine_patch_cloud_normals`; `search.rs`, `view_subset/`, `fronto_cache/` and the parts of `consensus.rs`, `level.rs` and `prof.rs` nothing else uses; the `refine_patch_normal` group in `benches/patch_render.rs`; the tests (`tests/patch/test_patch_normal_refine.py`, `tests/patch/test_refine_normals_keypoints.py`, `patch/normal_refine/tests.rs`, the cloud refinement in `patch/batch_cancel_tests.rs`, `bench/tests/reference_view.rs`'s use); and the scripts that only exercise it (`bench_normal_refine.py`, `validate_refine_subset.py`, `patch_crossval.py`, and the photometric ones under `scripts/track_at_pixel/`).

`NormalRefineParams` is then used only as the gating shim in `view_selection.rs` and `member_coherence/matrix.rs`; those get a small gating struct of their own. The module keeps the shared code listed above, and may be renamed to say what it holds.

Specs deleted: `core/patch/patch-normal-refinement.md`, `core/patch/patch-normal-refine-view-subset.md`, `core/patch/fronto-parallel-patch-cache.md`. Specs updated: `core/patch/patch-cloud.md`, `core/patch/sift-to-patch-reconstruction.md`, `python-bindings.md`, `GLOSSARY.md` (the *fit normal* and *refine* rows) and the area READMEs.

### 9. File the grown-plane draft

[grown-plane-normal.md](grown-plane-normal.md) becomes a standing spec under `core/bench/`, and this draft is deleted.

## Open questions

- Whether `embed-patches` needs the normal step in every round, or once.
