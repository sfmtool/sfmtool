# Python Bindings

The Python package reaches the Rust code through one compiled extension module,
`sfmtool._sfmtool`, built from the [`sfmtool-py`](../crates/sfmtool-py/src/lib.rs)
crate with PyO3. The extension is internal: each of its submodules has a
public module of the same name on the package, `sfmtool.<name>`, and code
outside the package imports the bindings from there
(`from sfmtool.fileio import read_sfmr`). The bindings convert NumPy arrays and
Python objects to the Rust types and back; the behaviour of each binding is the
behaviour of the Rust code it calls, and that code's spec is where the behaviour
is described. This page is the index from the bindings to those specs: for each
submodule, the source files that define its names, the main classes and
functions each file exposes, and the spec that describes them. A row whose spec
column reads *none* is a binding that no spec describes.

## How the module is assembled

[`lib.rs`](../crates/sfmtool-py/src/lib.rs) registers eleven submodules, each
through the `register` function of its source module, and sets each
submodule's `__name__` to the public `sfmtool.<name>`, so tracebacks and help
text name the public location. Three names are registered at the root for
the package, and a fourth, `run_explorer`, is the viewer that `sfm explorer`
calls ([explorer-command.md](cli/visualization/explorer-command.md)):

| Name | What it is | Spec |
|------|------------|------|
| `build_profile()` | `"debug"` or `"release"`, the Cargo profile the extension was built with; benchmarks refuse a debug build | none |
| `THUMBNAIL_SIZE` | the thumbnail edge length both on-disk formats store, which the Python SIFT extractors resize to | [sift-file-format.md](formats/sift-file-format.md), [sfmr-file-format.md](formats/sfmr-file-format.md) |
| `ProgressCounter` ([source](../crates/sfmtool-py/src/py_progress.rs)) | a thread-safe counter a long GIL-releasing kernel increments and Python polls from another thread | none (used in [cluster-patches-command.md](cli/image-feature/cluster-patches-command.md)) |

## Where Python code reads a binding

Each extension submodule `<name>` has a public module `sfmtool.<name>`, a
Python file that re-exports it with `from ._sfmtool.<name> import *` and takes
its `__all__`, so the public module exports exactly the submodule's names, as
the same objects:

| Public module | Extension submodule |
|---------------|---------------------|
| [`sfmtool.analysis`](../src/sfmtool/analysis.py) | `_sfmtool.analysis` |
| [`sfmtool.bench`](../src/sfmtool/bench.py) | `_sfmtool.bench` |
| [`sfmtool.fileio`](../src/sfmtool/fileio.py) | `_sfmtool.fileio` |
| [`sfmtool.flow`](../src/sfmtool/flow.py) | `_sfmtool.flow` |
| [`sfmtool.geometry`](../src/sfmtool/geometry.py) | `_sfmtool.geometry` |
| [`sfmtool.matching`](../src/sfmtool/matching.py) | `_sfmtool.matching` |
| [`sfmtool.patches`](../src/sfmtool/patches.py) | `_sfmtool.patches` |
| [`sfmtool.reconstruction`](../src/sfmtool/reconstruction.py) | `_sfmtool.reconstruction` |
| [`sfmtool.sift`](../src/sfmtool/sift/__init__.py) | `_sfmtool.sift` |
| [`sfmtool.spatial`](../src/sfmtool/spatial.py) | `_sfmtool.spatial` |
| [`sfmtool.spherical`](../src/sfmtool/spherical.py) | `_sfmtool.spherical` |

`sfmtool.sift` is the one subpackage among them: beside the bindings it holds
the Python SIFT file I/O and the OpenCV and COLMAP extractors. `sfmtool.analysis`
holds the analysis bindings and is a different module from `sfmtool.analyze`,
the Python reconstruction analysis that calls some of them; each keeps its
name, the one from the extension and the one from the code it holds.
`sfmtool.fileio` is named `fileio` rather than `io` because
`from sfmtool import *` binds every public module, and a module named `io`
would replace the standard library's `io` in the importing namespace.

The package root [`sfmtool/__init__.py`](../src/sfmtool/__init__.py) imports
the three root names of the extension and no binding from a submodule, so a
binding is always read through its module: `sfmtool.fileio.read_sfmr`, or
`bench.commit(…)` after `from sfmtool import bench`. One name means one thing
this way, where a flat root surface would hold the binding
`match_image_pair` and `sfmtool.feature_match` the Python
`match_image_pair` that takes different arguments, and the bench's steps
(`commit`, `split`, `fit`) would read as something else beside the rest. The
public modules are bound on the root on first use through `_LAZY_SUBPACKAGES`,
so `import sfmtool` followed by `sfmtool.fileio.read_sfmr(…)` works and `import
sfmtool` still loads nothing but the extension. Since each binding's `__module__`
is its public module, a class with a `__reduce__`, such as `CameraIntrinsics`,
pickles and unpickles through that path. `sfmtool.write_sift` on the
root is the Python function in `sfmtool.sift.file`, which checks its arguments
and then calls `sfmtool.fileio.write_sift`.

Code inside `src/sfmtool/` may import from `._sfmtool` directly, as it may
import any internal name. Tests, scripts, the docs and spec examples import
from the public modules; [`tests/test_module_layout.py`](../tests/test_module_layout.py)
fails on a path into the extension there outside its allowlist, which names
the registration tests (their subject is what each extension submodule
registers), the test of `run_explorer` (which has no public home) and two
monkeypatches that replace a binding where `src/` looks it up.
[`test_reconstruction_patches_registration.py`](../tests/rust_bindings/reconstruction/test_reconstruction_patches_registration.py)
checks that each public module exports exactly its submodule's `__all__`.

A binding's Python name carries no language suffix such as `_py` or `_rs`.
Where the Rust function behind it is named `<name>_py` to keep it apart from
the core function it calls, `#[pyo3(name = "<name>")]` gives Python the plain
name. A Python wrapper of the same name in `src/sfmtool/` imports the binding
under an alias, for example `match_image_pair as _rust_match_image_pair` in
[`feature_match/_core.py`](../src/sfmtool/feature_match/_core.py).

## Submodules

Paths in the *Source* column are relative to
[`crates/sfmtool-py/src/`](../crates/sfmtool-py/src/lib.rs).

### `geometry`

| Source | Exposes | Spec |
|--------|---------|------|
| [geometry/camera_intrinsics.rs](../crates/sfmtool-py/src/geometry/camera_intrinsics.rs) | `CameraIntrinsics`, with `CameraIntrinsics.refit` | [camera-model-registry.md](core/camera/camera-model-registry.md), [sfmtool-camera-models.md](formats/sfmtool-camera-models.md); `refit` in [refit-camera-intrinsics.md](core/camera/refit-camera-intrinsics.md) |
| [geometry/rot_quaternion.rs](../crates/sfmtool-py/src/geometry/rot_quaternion.rs), [rigid_transform.rs](../crates/sfmtool-py/src/geometry/rigid_transform.rs), [se3_transform.rs](../crates/sfmtool-py/src/geometry/se3_transform.rs) | `RotQuaternion`, `RigidTransform`, `Se3Transform` | none |
| [geometry/convention.rs](../crates/sfmtool-py/src/geometry/convention.rs) | `poses_colmap_to_canonical`, `poses_canonical_to_colmap`, `relative_poses_conjugate_s`, `flip_camera_poses_s`, `world_rotate_w`, `world_rotate_w_inverse` | [sfmr-file-format.md](formats/sfmr-file-format.md) (coordinate conventions) |
| [geometry/absolute_pose.rs](../crates/sfmtool-py/src/geometry/absolute_pose.rs), [pose_refine.rs](../crates/sfmtool-py/src/geometry/pose_refine.rs) | `p3p_solve`, `estimate_absolute_pose`, `refine_absolute_pose` | [absolute-pose.md](core/geometry/absolute-pose.md) |
| [geometry/affine_factorization.rs](../crates/sfmtool-py/src/geometry/affine_factorization.rs) | `factorize_affine`, `AffineFactorization`, `MetricHypothesis` | [affine-factorization.md](core/geometry/affine-factorization.md) |
| [geometry/epipolar_estimation.rs](../crates/sfmtool-py/src/geometry/epipolar_estimation.rs) | `estimate_fundamental`, `focal_from_fundamental` | [epipolar-estimation.md](core/geometry/epipolar-estimation.md) |
| [geometry/homography_estimation.rs](../crates/sfmtool-py/src/geometry/homography_estimation.rs), [focal_vote.rs](../crates/sfmtool-py/src/geometry/focal_vote.rs) | `estimate_homography`, `focal_vote` | [focal-vote.md](core/geometry/focal-vote.md) |
| [geometry/estimate_intrinsics.rs](../crates/sfmtool-py/src/geometry/estimate_intrinsics.rs) | `estimate_intrinsics` | [estimate-intrinsics.md](core/geometry/estimate-intrinsics.md) |
| [geometry/reprojection.rs](../crates/sfmtool-py/src/geometry/reprojection.rs) | `reprojection_residuals`, `inlier_fraction` | [reprojection-residuals.md](core/geometry/reprojection-residuals.md) |
| [geometry/resect_translation.rs](../crates/sfmtool-py/src/geometry/resect_translation.rs) | `resect_translation` | [rotation-locked-resection.md](core/geometry/rotation-locked-resection.md) |
| [geometry/rotation_init.rs](../crates/sfmtool-py/src/geometry/rotation_init.rs) | `rotation_init` | [rotation-init.md](core/geometry/rotation-init.md) |
| [geometry/bundle_adjust.rs](../crates/sfmtool-py/src/geometry/bundle_adjust.rs) | `bundle_adjust` | [bundle-adjustment.md](core/geometry/bundle-adjustment.md) |
| [geometry/baseline_direction.rs](../crates/sfmtool-py/src/geometry/baseline_direction.rs) | `baseline_directions` | [baseline-direction.md](core/geometry/baseline-direction.md) |
| [geometry/reconstruction_growth.rs](../crates/sfmtool-py/src/geometry/reconstruction_growth.rs), [resect_images.rs](../crates/sfmtool-py/src/geometry/resect_images.rs) | `grow_reconstruction`, `resect_images_batch`, `resect_images` | [reconstruction-growth.md](core/geometry/reconstruction-growth.md) |
| [geometry/pose_verification.rs](../crates/sfmtool-py/src/geometry/pose_verification.rs) | `verify_poses`, `repair_poses` | [pose-verification.md](core/geometry/pose-verification.md) |
| [geometry/relative_pose.rs](../crates/sfmtool-py/src/geometry/relative_pose.rs) | `estimate_essential_rays`, `fit_ray_rotation` | [relative-pose.md](core/geometry/relative-pose.md) |
| [geometry/translation_averaging.rs](../crates/sfmtool-py/src/geometry/translation_averaging.rs) | `average_translations`, `relative_lengths`, `direction_reading`, `orientation_reading` and their solver constants | [translation-averaging.md](core/geometry/translation-averaging.md) |

### `fileio`

| Source | Exposes | Spec |
|--------|---------|------|
| [fileio/sfmr.rs](../crates/sfmtool-py/src/fileio/sfmr.rs) | `read_sfmr`, `read_sfmr_metadata`, `read_sfmr_content_hash`, `write_sfmr`, `verify_sfmr`, `POINT_CONSTRAINT_NAMES` | [sfmr-file-format.md](formats/sfmr-file-format.md); `POINT_CONSTRAINT_NAMES` in [bundle-adjustment.md](core/geometry/bundle-adjustment.md) |
| [fileio/sift.rs](../crates/sfmtool-py/src/fileio/sift.rs) | `read_sift`, `read_sift_metadata`, `read_sift_partial`, `write_sift`, `verify_sift`, `SiftWriteQueue` | [sift-file-format.md](formats/sift-file-format.md); `SiftWriteQueue` in [sift.md](core/features/sift.md) |
| [fileio/matches.rs](../crates/sfmtool-py/src/fileio/matches.rs), [matches_file.rs](../crates/sfmtool-py/src/fileio/matches_file.rs) | `read_matches`, `read_matches_metadata`, `write_matches`, `verify_matches`, `MatchesFile` | [matches-file-format.md](formats/matches-file-format.md) |
| [fileio/camrig.rs](../crates/sfmtool-py/src/fileio/camrig.rs) | `read_camrig`, `read_camrig_metadata`, `write_camrig`, `verify_camrig`, and the image-pattern helpers `validate_camrig_pattern`, `camrig_pattern_to_glob`, `camrig_pattern_matches`, `camrig_pattern_frame_index` | [camrig-file-format.md](formats/camrig-file-format.md) |
| [fileio/colmap_binary.rs](../crates/sfmtool-py/src/fileio/colmap_binary.rs), [colmap_db.rs](../crates/sfmtool-py/src/fileio/colmap_db.rs) | `read_colmap_binary`, `write_colmap_binary`, `write_colmap_db`, `read_colmap_db_matches` | [colmap-interop.md](formats/colmap-interop.md) |
| [fileio/image.rs](../crates/sfmtool-py/src/fileio/image.rs) | `image_dimensions`, `image_has_alpha`, `read_image_rgb`, `read_image_rgba`, `write_image_rgb`, `write_image_rgba` | `image_dimensions` none; the others in [reading-and-writing-images.md](core/camera/reading-and-writing-images.md) |
| [fileio/web_export.rs](../crates/sfmtool-py/src/fileio/web_export.rs) | `write_web_export` | [web-export-command.md](cli/visualization/web-export-command.md) |

### `sift`

| Source | Exposes | Spec |
|--------|---------|------|
| [sift/extract.rs](../crates/sfmtool-py/src/sift/extract.rs) | `extract_sift`, `detect_sift_keypoints`, `describe_keypoints`, `affine_shapes_from_similarity` | [sift.md](core/features/sift.md) |

### `reconstruction`

| Source | Exposes | Spec |
|--------|---------|------|
| [reconstruction/sfmr_reconstruction.rs](../crates/sfmtool-py/src/reconstruction/sfmr_reconstruction.rs) | `SfmrReconstruction` | [sfmr-file-format.md](formats/sfmr-file-format.md) |
| | `SfmrReconstruction.to_embedded_patches` | [sift-to-patch-reconstruction.md](core/patch/sift-to-patch-reconstruction.md) |
| | `SfmrReconstruction.triangulation_diagnostics` | [batch-triangulation-api.md](core/reconstruction/batch-triangulation-api.md) |
| | `SfmrReconstruction.find_points_at_infinity` | [find-points-at-infinity.md](cli/reconstruction/xform/find-points-at-infinity.md) |
| [reconstruction/edited.rs](../crates/sfmtool-py/src/reconstruction/edited.rs) | `EditedReconstruction`, `PointMap` | [edited-reconstruction.md](core/reconstruction/edited-reconstruction.md) |
| | `EditedReconstruction.bundle_adjust` | [bundle-adjust.md](core/reconstruction/bundle-adjust.md) |
| | `EditedReconstruction.move_camera` | [move-camera.md](core/reconstruction/move-camera.md) |
| | `EditedReconstruction.prune_covered_observations` | [prune-covered-observations.md](core/reconstruction/prune-covered-observations.md) |
| | `EditedReconstruction.resect_image_in_place` | [resect-image.md](gui/edits/resect-image.md) |
| [reconstruction/bundle_adjust.rs](../crates/sfmtool-py/src/reconstruction/bundle_adjust.rs) | `focal_is_releasable`, `distortion_is_releasable` | [bundle-adjust.md](core/reconstruction/bundle-adjust.md) |
| [reconstruction/add_image_to_tracks.rs](../crates/sfmtool-py/src/reconstruction/add_image_to_tracks.rs) | `EditedReconstruction.add_image_to_tracks` | [add-image-to-tracks.md](core/reconstruction/add-image-to-tracks.md) |
| [reconstruction/switch_camera_model.rs](../crates/sfmtool-py/src/reconstruction/switch_camera_model.rs) | the `switch_camera_model` method of both reconstruction classes, `SfmrReconstruction.outermost_keypoints` | [switch-camera-model.md](core/reconstruction/switch-camera-model.md), [outermost-keypoint.md](core/reconstruction/outermost-keypoint.md) |
| [reconstruction/triangulate_points.rs](../crates/sfmtool-py/src/reconstruction/triangulate_points.rs) | `triangulate_points`, `VERDICT_CODES` | [triangulation-rules.md](core/reconstruction/triangulation-rules.md) |
| [reconstruction/range_expr.rs](../crates/sfmtool-py/src/reconstruction/range_expr.rs) | `RangeExpr` | none (its grammar is used in [inspect-command.md](cli/reconstruction/inspect-command.md) and [to-colmap-bin-command.md](cli/colmap-interop/to-colmap-bin-command.md)) |

### `patches`

| Source | Exposes | Spec |
|--------|---------|------|
| [patches/oriented_patch.rs](../crates/sfmtool-py/src/patches/oriented_patch.rs), [cloud.rs](../crates/sfmtool-py/src/patches/cloud.rs), [views.rs](../crates/sfmtool-py/src/patches/views.rs), [render_bitmaps.rs](../crates/sfmtool-py/src/patches/render_bitmaps.rs) | `OrientedPatch`, `PatchCloud`, `CameraViews`, `ImagePyramidSet`, `PatchCloud.render_bitmaps` | [patch-cloud.md](core/patch/patch-cloud.md) |
| [patches/view_tile.rs](../crates/sfmtool-py/src/patches/view_tile.rs) | `OrientedPatch.render_view_tile`: one view's `R×R` tile with its coverage, clipped share, viewing angle and tilt direction | [reference-view.md](core/patch/reference-view.md) |
| [patches/select_views.rs](../crates/sfmtool-py/src/patches/select_views.rs) | `PatchCloud.select_views` | [patch-view-selection.md](core/patch/patch-view-selection.md) |
| [patches/refine_normals.rs](../crates/sfmtool-py/src/patches/refine_normals.rs) | `PatchCloud.refine_normals` | [patch-normal-refinement.md](core/patch/patch-normal-refinement.md) |
| [patches/localize_keypoints.rs](../crates/sfmtool-py/src/patches/localize_keypoints.rs) | `PatchCloud.localize_keypoints` | [patch-keypoint-localization.md](core/patch/patch-keypoint-localization.md) |
| [patches/refine_keypoints.rs](../crates/sfmtool-py/src/patches/refine_keypoints.rs) | `PatchCloud.refine_keypoints` | [keypoint-subpixel-refinement.md](core/patch/keypoint-subpixel-refinement.md) |
| [patches/member_coherence.rs](../crates/sfmtool-py/src/patches/member_coherence.rs) | `PatchCloud.validate_member_coherence` | [member-coherence-validation.md](core/patch/member-coherence-validation.md) |
| [patches/spawn.rs](../crates/sfmtool-py/src/patches/spawn.rs) | `spawn_candidate_tracks` | [candidate-track-spawning.md](core/patch/candidate-track-spawning.md) |
| [patches/self_similarity.rs](../crates/sfmtool-py/src/patches/self_similarity.rs), [mod.rs](../crates/sfmtool-py/src/patches/mod.rs) | `zncc_self_similarity_parts`, `zncc_self_similarity_parts_stack`, `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS` | [zncc-self-similarity-radius.md](core/patch/zncc-self-similarity-radius.md) |
| [patches/blur_matched.rs](../crates/sfmtool-py/src/patches/blur_matched.rs) | `assess_blur`, `blur_sigma_to_reach`, `blur_to_length`, `score_against_bitmap` | [blur-matched-zncc.md](core/patch/blur-matched-zncc.md) |
| [patches/mod.rs](../crates/sfmtool-py/src/patches/mod.rs) | `DEFAULT_ANISOTROPIC_THRESHOLD` | [image-warping.md](core/camera/image-warping.md) § "Choosing the sampler per view" |
| [patches/photometric_ransac.rs](../crates/sfmtool-py/src/patches/photometric_ransac.rs) | `refine_photometric_ransac`, `RansacPhotometricOutput` | [photometric-subsets-ransac.md](core/spherical/photometric-subsets-ransac.md) |
| [patches/consensus_atlas.rs](../crates/sfmtool-py/src/patches/consensus_atlas.rs) | `render_consensus_atlas` | [tile-batched-consensus-atlas.md](core/spherical/tile-batched-consensus-atlas.md) |

### `bench`

| Source | Exposes | Spec |
|--------|---------|------|
| [bench.rs](../crates/sfmtool-py/src/bench.rs) | `Bench`, `EditableTrack`, and the track steps `create_track`, `create_cluster`, `add_observation`, `sight_observation`, `shape_observation`, the verdict steps (`set_verdict`, `pin_verdict`, `unpin_verdict`, `apply_thresholds`), `set_reference`, the patch steps (`translate_patch`, `translate_patch_to_pixel`, `tilt_patch`, `spin_patch`, `resize_patch`, `resize_patch_to_pixel`), `search_descriptors`, `search_geometry`, `evaluate`, `fit`, `set_stage`, `split`, `duplicate`, `commit` | [bench.md](core/bench/bench.md), [editable-track.md](core/bench/editable-track.md) |
| [bench/track_at_pixel.rs](../crates/sfmtool-py/src/bench/track_at_pixel.rs) | `build_track_at_pixel`, `TrackAtPixelSources`, `TrackAtPixelError` | [track-at-pixel.md](core/bench/track-at-pixel.md) |
| [bench/nearby.rs](../crates/sfmtool-py/src/bench/nearby.rs) | `NearbyTrackSources`, `nearby_points`, `nearby_cluster_tracks`, `guided_matches`, `constellation_seeds` | [nearby-sources.md](core/bench/nearby-sources.md) |
| [bench/nearby_tracks.rs](../crates/sfmtool-py/src/bench/nearby_tracks.rs) | `find_nearby_tracks` | [nearby-tracks.md](core/bench/nearby-tracks.md) |
| [bench/range.rs](../crates/sfmtool-py/src/bench/range.rs) | `distance_range`, `camera_spread`, `classify_range` | [distance-range.md](core/bench/distance-range.md) |
| [bench/layers.rs](../crates/sfmtool-py/src/bench/layers.rs) | `depth_layers` | [depth-layers.md](core/bench/depth-layers.md) |
| [bench/far_field.rs](../crates/sfmtool-py/src/bench/far_field.rs), [patch_read.rs](../crates/sfmtool-py/src/bench/patch_read.rs) | `far_field_sweep`, `read_patch_along_ray` | [far-field-sweep.md](core/bench/far-field-sweep.md) |

### `matching`

| Source | Exposes | Spec |
|--------|---------|------|
| [matching/descriptor.rs](../crates/sfmtool-py/src/matching/descriptor.rs), [image.rs](../crates/sfmtool-py/src/matching/image.rs), [sweep.rs](../crates/sfmtool-py/src/matching/sweep.rs) | `descriptor_distance`, `find_best_descriptor_match`, `match_candidates_by_descriptor`, `match_image_pair`, `match_image_pairs_batch`, the `*_sweep*` and `polar_mutual_best_match*` functions | [descriptor-matching.md](core/features/descriptor-matching.md) |
| [matching/cluster.rs](../crates/sfmtool-py/src/matching/cluster.rs) | `background_floor_clusters`, `background_floor_clusters_kdf`, `clusters_to_pair_matches`, `refine_cluster_patches` | [track-cluster-matching.md](core/features/track-cluster-matching.md), [cluster-patch-refinement.md](core/patch/cluster-patch-refinement.md) |
| [matching/covisibility.rs](../crates/sfmtool-py/src/matching/covisibility.rs) | `ClusterCovisibility`, `SeedImageGroup`, `ClusterCovisibilitySeedImageGroups` | [cluster-covisibility.md](core/features/cluster-covisibility.md), [covisibility-selection.md](core/features/covisibility-selection.md) |

### `analysis`

| Source | Exposes | Spec |
|--------|---------|------|
| [analysis/core.rs](../crates/sfmtool-py/src/analysis/core.rs) | `estimate_alignment`, `ransac_alignment` | [reconstruction-alignment.md](core/analysis/reconstruction-alignment.md) |
| | `find_point_correspondences`, `merge_points_and_tracks` | [point-correspondence.md](core/reconstruction/point-correspondence.md) |
| | `compute_narrow_track_mask` | [batch-triangulation-api.md](core/reconstruction/batch-triangulation-api.md) |
| | `filter_tracks_by_point_mask` | [xform-command.md](cli/reconstruction/xform/xform-command.md) |
| | `apply_se3_to_camera_poses` | none |
| [analysis/triangulation.rs](../crates/sfmtool-py/src/analysis/triangulation.rs), [point_or_bearing.rs](../crates/sfmtool-py/src/analysis/point_or_bearing.rs) | `triangulate_batch`, `fit_point_and_bearing_batch`, `bearing_score_batch`, `observed_rays` and the triangulation default constants | [batch-triangulation-api.md](core/reconstruction/batch-triangulation-api.md) |
| [analysis/epipolar.rs](../crates/sfmtool-py/src/analysis/epipolar.rs) | `epipolar_curves` | [epipolar-curves.md](core/camera/epipolar-curves.md) |
| [analysis/image_pair_graph.rs](../crates/sfmtool-py/src/analysis/image_pair_graph.rs) | `build_covisibility_pairs`, `build_frustum_intersection_pairs` | [image-pair-graph.md](core/analysis/image-pair-graph.md) |
| [analysis/keypoint_reach.rs](../crates/sfmtool-py/src/analysis/keypoint_reach.rs) | `keypoint_pairs_within_reach` | [keypoint-reach.md](core/analysis/keypoint-reach.md) |
| [analysis/covered_by_finer.rs](../crates/sfmtool-py/src/analysis/covered_by_finer.rs) | `covered_by_finer` | [covered-by-finer.md](core/analysis/covered-by-finer.md) |
| [analysis/observation_adjacency.rs](../crates/sfmtool-py/src/analysis/observation_adjacency.rs) | `build_observation_adjacency` | [observation-adjacency-graph.md](core/analysis/observation-adjacency-graph.md) |
| [analysis/observation_coverage.rs](../crates/sfmtool-py/src/analysis/observation_coverage.rs) | `ObservationCoverage` | [observation-coverage.md](core/analysis/observation-coverage.md) |
| [analysis/source_clusters.rs](../crates/sfmtool-py/src/analysis/source_clusters.rs), [cluster_radii.rs](../crates/sfmtool-py/src/analysis/cluster_radii.rs) | `source_clusters`, `assign_bands`, `cluster_radii`, `coarsest_cluster_ids` | [source-clusters.md](core/analysis/source-clusters.md) |
| [analysis/adjacency_surfel_normals.rs](../crates/sfmtool-py/src/analysis/adjacency_surfel_normals.rs) | `estimate_adjacency_surfel_normals` | [adjacency-surfel-normals.md](core/analysis/adjacency-surfel-normals.md) |
| [analysis/cell_plane_normals.rs](../crates/sfmtool-py/src/analysis/cell_plane_normals.rs) | `cell_plane_normals` | [cell-plane-normals.md](core/patch/cell-plane-normals.md) |
| [analysis/cluster_census.rs](../crates/sfmtool-py/src/analysis/cluster_census.rs) | `cluster_census` | [cluster-census.md](core/analysis/cluster-census.md) |

### `flow`

| Source | Exposes | Spec |
|--------|---------|------|
| [flow/optical.rs](../crates/sfmtool-py/src/flow/optical.rs) | `compute_optical_flow`, `compute_optical_flow_with_init`, `compute_optical_flow_timed`, `compose_flow`, `advect_points`, `gpu_available` | [optical-flow.md](core/features/optical-flow.md), [gpu-optical-flow.md](core/features/gpu-optical-flow.md) |
| [flow/warp.rs](../crates/sfmtool-py/src/flow/warp.rs) | `WarpMap`, `ImagePyramid` | [image-warping.md](core/camera/image-warping.md) |

### `spatial`

| Source | Exposes | Spec |
|--------|---------|------|
| [spatial/kdtree.rs](../crates/sfmtool-py/src/spatial/kdtree.rs) | `KdTree2d`, `KdTree3d` | [point-cloud-index.md](core/spatial/point-cloud-index.md) |
| [spatial/kdforest.rs](../crates/sfmtool-py/src/spatial/kdforest.rs) | `KdForest` | [randomized-kdtree-forest.md](core/features/randomized-kdtree-forest.md) |
| [spatial/kdf.rs](../crates/sfmtool-py/src/spatial/kdf.rs) | `LazyKdForest`, `write_kdf`, `read_kdf`, `verify_kdf`, `kdf_file_summary`, `verify_sift_sources` | [kdf-file-format.md](formats/kdf-file-format.md), [lazy-kdforest-query.md](core/features/lazy-kdforest-query.md) |
| [spatial/constellation_query.rs](../crates/sfmtool-py/src/spatial/constellation_query.rs) | `radius_for_feature_count` | [kdf-constellation-query.md](core/features/kdf-constellation-query.md) |

### `spherical`

| Source | Exposes | Spec |
|--------|---------|------|
| [spherical/tile_rig.rs](../crates/sfmtool-py/src/spherical/tile_rig.rs), [sphere_points.rs](../crates/sfmtool-py/src/spherical/sphere_points.rs) | `SphericalTileRig`, `evenly_distributed_sphere_points` | [spherical-tiles-rig.md](core/spherical/spherical-tiles-rig.md) |
| [spherical/tile_source_stack.rs](../crates/sfmtool-py/src/spherical/tile_source_stack.rs) | `PerSphericalTileSourceStack` | [per-spherical-tile-source-stack.md](core/spherical/per-spherical-tile-source-stack.md) |

## Keeping the index current

[`tests/test_python_bindings_index.py`](../tests/test_python_bindings_index.py)
fails when a source file under `crates/sfmtool-py/src/` declares a
`#[pyclass]`, `#[pyfunction]` or `#[pymethods]` item and this page does not link
it, so a new binding file has to be given a row here.
