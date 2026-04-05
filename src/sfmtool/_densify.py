# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Densify reconstructions with additional matches.

This module provides functionality to:
1. Load an existing reconstruction
2. Find image pairs to match (covisibility or frustum intersection)
3. Run guided matching using sweep algorithms (automatically picking linear/polar)
4. Triangulate new points from additional matches
5. Export updated reconstruction
"""

import tempfile
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pycolmap

from ._cameras import colmap_camera_from_intrinsics
from ._image_pair_graph import (
    build_covisibility_pairs,
    build_frustum_intersection_pairs,
)
from ._sift_file import SiftReader, get_sift_path_for_image
from ._sfmtool import KdTree3d, RotQuaternion, SfmrReconstruction
from .feature_match import GeometricFilterConfig, match_image_pair


def prune_image_pairs(
    pairs_with_scores: list[tuple[int, int, float]],
    close_pair_threshold: int = 4,
    max_close_pairs: int | None = None,
    max_distant_pairs: int = 5000,
    distant_pair_search_multiplier: int = 3,
) -> list[tuple[int, int]]:
    """Prune image pairs to a useful subset for matching.

    Strategy:
    1. Take pairs where images are close together (within close_pair_threshold)
    2. Take the best distant pairs (high score) for loop closures
    """
    if not pairs_with_scores:
        return []

    pairs_by_distance = sorted(pairs_with_scores, key=lambda p: abs(p[1] - p[0]))

    close_pairs = []
    distant_pairs = []

    for img_i, img_j, score in pairs_by_distance:
        if abs(img_j - img_i) <= close_pair_threshold:
            close_pairs.append((img_i, img_j, score))
        else:
            distant_pairs.append((img_i, img_j, score))

    if max_close_pairs is not None:
        selected_close_pairs = close_pairs[:max_close_pairs]
    else:
        selected_close_pairs = close_pairs

    search_count = max_distant_pairs * distant_pair_search_multiplier
    distant_sample = distant_pairs[:search_count]
    distant_sample_sorted = sorted(distant_sample, key=lambda p: p[2], reverse=True)
    selected_distant_pairs = distant_sample_sorted[:max_distant_pairs]

    selected_pairs = [
        (i, j) for i, j, _ in selected_close_pairs + selected_distant_pairs
    ]

    click.echo("\nPair pruning:")
    click.echo(f"  Total pairs: {len(pairs_with_scores)}")
    click.echo(
        f"  Close pairs (distance <= {close_pair_threshold}): "
        f"{len(close_pairs)} -> kept {len(selected_close_pairs)}"
    )
    click.echo(
        f"  Distant pairs: {len(distant_pairs)} -> "
        f"searched {len(distant_sample)} -> kept {len(selected_distant_pairs)}"
    )
    click.echo(f"  Final pairs to match: {len(selected_pairs)}")

    return selected_pairs


def _match_single_pair(
    img_i: int,
    img_j: int,
    workspace_dir: Path,
    image_names: list[str],
    quaternions: np.ndarray,
    translations: np.ndarray,
    camera_indexes: np.ndarray,
    cameras: list[pycolmap.Camera],
    max_features: int | None,
    window_size: int,
    distance_threshold: float | None,
    geometric_config: GeometricFilterConfig | None = None,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Match a single image pair."""
    image_path_i = workspace_dir / image_names[img_i]
    image_path_j = workspace_dir / image_names[img_j]

    sift_path_i = get_sift_path_for_image(image_path_i)
    sift_path_j = get_sift_path_for_image(image_path_j)

    with SiftReader(sift_path_i) as reader_i:
        pos_i = reader_i.read_positions(count=max_features)
        desc_i = reader_i.read_descriptors(count=max_features)
        if geometric_config is not None and geometric_config.enable_geometric_filtering:
            affine_i = reader_i.read_affine_shapes(count=max_features)
        else:
            affine_i = None
    with SiftReader(sift_path_j) as reader_j:
        pos_j = reader_j.read_positions(count=max_features)
        desc_j = reader_j.read_descriptors(count=max_features)
        if geometric_config is not None and geometric_config.enable_geometric_filtering:
            affine_j = reader_j.read_affine_shapes(count=max_features)
        else:
            affine_j = None

    cam_i = cameras[camera_indexes[img_i]]
    cam_j = cameras[camera_indexes[img_j]]

    # Convert quaternions from WXYZ to XYZW format for pycolmap
    quat_i_xyzw = np.roll(quaternions[img_i], -1)
    quat_j_xyzw = np.roll(quaternions[img_j], -1)

    img_i_cam_from_world = pycolmap.Rigid3d(
        pycolmap.Rotation3d(quat_i_xyzw), translations[img_i]
    )
    img_j_cam_from_world = pycolmap.Rigid3d(
        pycolmap.Rotation3d(quat_j_xyzw), translations[img_j]
    )

    matches = match_image_pair(
        img_i_cam_from_world,
        img_j_cam_from_world,
        cam_i,
        cam_j,
        pos_i,
        desc_i,
        pos_j,
        desc_j,
        window_size=window_size,
        distance_threshold=distance_threshold,
        affine_shapes_i=affine_i,
        affine_shapes_j=affine_j,
        geometric_config=geometric_config,
    )

    match_tuples = [(m[0], m[1]) for m in matches]
    return ((img_i, img_j), match_tuples)


def match_image_pairs(
    workspace_dir: Path,
    image_names: list[str],
    pairs: list[tuple[int, int]],
    quaternions: np.ndarray,
    translations: np.ndarray,
    camera_indexes: np.ndarray,
    cameras: list[pycolmap.Camera],
    max_features: int | None = None,
    window_size: int = 30,
    distance_threshold: float | None = None,
    geometric_config: GeometricFilterConfig | None = None,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Run sweep matching on image pairs."""
    all_matches = {}

    for idx, (img_i, img_j) in enumerate(pairs):
        if (idx + 1) % 100 == 0 or idx == len(pairs) - 1:
            click.echo(f"  Matching pair {idx + 1}/{len(pairs)}...")

        pair_key, match_tuples = _match_single_pair(
            img_i,
            img_j,
            workspace_dir,
            image_names,
            quaternions,
            translations,
            camera_indexes,
            cameras,
            max_features,
            window_size,
            distance_threshold,
            geometric_config,
        )
        all_matches[pair_key] = match_tuples

    return all_matches


def triangulate_new_tracks(
    new_matches: dict[tuple[int, int], list[tuple[int, int]]],
    max_features: int | None,
    workspace_dir: Path,
    temp_dir: Path,
    recon: SfmrReconstruction,
    ba_options: dict | None = None,
    filter_max_reproj_error: float = 4.0,
    filter_min_track_length: int = 3,
    filter_min_tri_angle: float = 1.5,
    filter_isolated_median_ratio: float = 2.0,
) -> pycolmap.Reconstruction:
    """Build tracks from new matches and triangulate using known camera poses."""
    from ._colmap_io import save_colmap_binary

    # Step 1: Export reconstruction to COLMAP binary format
    colmap_input_dir = temp_dir / "colmap_input"
    save_colmap_binary(recon, colmap_input_dir, max_features=max_features)

    # Step 2: Load the COLMAP binary reconstruction
    click.echo("  Loading reconstruction from COLMAP binary...")
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read_binary(str(colmap_input_dir))

    click.echo(
        f"  Loaded reconstruction: {len(reconstruction.cameras)} cameras, "
        f"{len(reconstruction.images)} images, "
        f"{len(reconstruction.points3D)} existing points"
    )

    # Step 3: Create database with new matches
    db_path = temp_dir / "densify.db"
    output_path = temp_dir / "output"
    output_path.mkdir(exist_ok=True)

    with pycolmap.Database.open(str(db_path)) as db:
        for cam_id, cam in reconstruction.cameras.items():
            db.write_camera(cam, use_camera_id=True)

        # Write rigs and frames before images (required by pycolmap 4.x)
        for rig_id, rig in reconstruction.rigs.items():
            db.write_rig(rig, use_rig_id=True)
        for frame_id, frame in reconstruction.frames.items():
            db.write_frame(frame, use_frame_id=True)

        for img_id, img in reconstruction.images.items():
            db.write_image(img, use_image_id=True)

            image_path = workspace_dir / img.name
            sift_path = get_sift_path_for_image(image_path)

            with SiftReader(sift_path) as reader:
                keypoints = reader.read_positions(count=max_features)
                descriptors = reader.read_descriptors(count=max_features)

            db.write_keypoints(img_id, keypoints)
            db.write_descriptors(
                img_id,
                pycolmap.FeatureDescriptors(
                    data=descriptors, type=pycolmap.FeatureExtractorType.SIFT
                ),
            )

        image_idx_to_id = {
            idx: img_id
            for idx, img_id in enumerate(sorted(reconstruction.images.keys()))
        }

        tvg_count = 0
        for (img_i, img_j), match_list in new_matches.items():
            if len(match_list) == 0:
                continue

            db_img_i = image_idx_to_id[img_i]
            db_img_j = image_idx_to_id[img_j]

            img_obj_i = reconstruction.images[db_img_i]
            img_obj_j = reconstruction.images[db_img_j]

            pose_i = img_obj_i.cam_from_world()
            pose_j = img_obj_j.cam_from_world()

            R_i = pose_i.rotation.matrix()
            t_i = pose_i.translation
            R_j = pose_j.rotation.matrix()
            t_j = pose_j.translation

            R_rel = R_j @ R_i.T
            t_rel = t_j - R_rel @ t_i

            matches_array = np.array(match_list, dtype=np.uint32)

            tvg = pycolmap.TwoViewGeometry()
            tvg.config = 2  # CALIBRATED

            rotation_rel = pycolmap.Rotation3d(R_rel)
            rigid_rel = pycolmap.Rigid3d(rotation_rel, t_rel)
            tvg.cam2_from_cam1 = rigid_rel
            tvg.inlier_matches = matches_array

            db.write_matches(db_img_i, db_img_j, matches_array)
            db.write_two_view_geometry(db_img_i, db_img_j, tvg)
            tvg_count += 1

    click.echo(f"  Wrote {tvg_count} two-view geometries to database")

    # Step 4: Triangulate new points from matches using known poses
    options = pycolmap.IncrementalPipelineOptions()
    if ba_options:
        options.ba_refine_focal_length = ba_options.get("refine_focal_length", False)
        options.ba_refine_principal_point = ba_options.get(
            "refine_principal_point", False
        )
        options.ba_refine_extra_params = ba_options.get("refine_extra_params", False)

    click.echo(
        f"  Triangulating points (preserving {len(reconstruction.points3D)} existing)..."
    )
    densified_reconstruction = pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=str(db_path),
        image_path=str(workspace_dir),
        output_path=str(output_path),
        clear_points=False,
        options=options,
        refine_intrinsics=ba_options is not None,
    )

    # Step 5: Merge duplicate points
    click.echo("  Merging duplicate points...")
    points_before_merge = len(densified_reconstruction.points3D)

    position_to_point_ids = defaultdict(list)
    for point_id, point in densified_reconstruction.points3D.items():
        pos_key = tuple(np.round(point.xyz, decimals=10))
        position_to_point_ids[pos_key].append(point_id)

    merged_count = 0
    for pos_key, point_ids in position_to_point_ids.items():
        if len(point_ids) > 1:
            base_point_id = point_ids[0]
            for other_point_id in point_ids[1:]:
                if densified_reconstruction.exists_point3D(
                    base_point_id
                ) and densified_reconstruction.exists_point3D(other_point_id):
                    base_point_id = densified_reconstruction.merge_points3D(
                        base_point_id, other_point_id
                    )
                    merged_count += 1

    points_after_merge = len(densified_reconstruction.points3D)
    if merged_count > 0:
        click.echo(
            f"  Merged {merged_count} duplicate points "
            f"({points_after_merge} remaining, "
            f"{points_before_merge - points_after_merge} duplicates removed)"
        )

    # Step 6: Run bundle adjustment on deduplicated points
    click.echo(
        f"  Running bundle adjustment on {len(densified_reconstruction.points3D)} points..."
    )

    ba_config = pycolmap.BundleAdjustmentOptions()
    if ba_options:
        ba_config.refine_focal_length = ba_options.get("refine_focal_length", True)
        ba_config.refine_principal_point = ba_options.get(
            "refine_principal_point", False
        )
        ba_config.refine_extra_params = ba_options.get("refine_extra_params", True)
    else:
        ba_config.refine_focal_length = True
        ba_config.refine_extra_params = True
        ba_config.refine_principal_point = False

    pycolmap.bundle_adjustment(densified_reconstruction, ba_config)

    # Step 7: Filter spurious points
    points_before = len(densified_reconstruction.points3D)
    densified_reconstruction.update_point_3d_errors()

    max_reproj_error = filter_max_reproj_error
    min_track_length = filter_min_track_length
    min_tri_angle = filter_min_tri_angle

    points_to_delete = []
    for point_id, point in densified_reconstruction.points3D.items():
        if point.error > max_reproj_error or point.track.length() < min_track_length:
            points_to_delete.append(point_id)
            continue

        if point.track.length() >= 2:
            track_elements = point.track.elements
            max_angle = 0.0

            for i in range(len(track_elements)):
                for j in range(i + 1, len(track_elements)):
                    img_i = densified_reconstruction.images[track_elements[i].image_id]
                    img_j = densified_reconstruction.images[track_elements[j].image_id]

                    pose_i = img_i.cam_from_world()
                    pose_j = img_j.cam_from_world()
                    center_i = pose_i.inverse().translation
                    center_j = pose_j.inverse().translation

                    angle_rad = pycolmap.calculate_triangulation_angle(
                        center_i, center_j, point.xyz
                    )
                    angle_deg = np.degrees(angle_rad)
                    max_angle = max(max_angle, angle_deg)

            if max_angle < min_tri_angle:
                points_to_delete.append(point_id)

    # Step 7b: Filter isolated points based on nearest neighbor distance
    if filter_isolated_median_ratio > 0 and len(densified_reconstruction.points3D) > 1:
        remaining_point_ids = [
            pid
            for pid in densified_reconstruction.points3D.keys()
            if pid not in points_to_delete
        ]

        if len(remaining_point_ids) > 1:
            positions = np.array(
                [
                    densified_reconstruction.points3D[pid].xyz
                    for pid in remaining_point_ids
                ]
            )

            nn_distances = KdTree3d(positions).nearest_neighbor_distances()
            nn_distance_median = np.median(nn_distances)
            threshold = nn_distance_median * filter_isolated_median_ratio

            isolated_count = 0
            for i, point_id in enumerate(remaining_point_ids):
                if nn_distances[i] > threshold:
                    points_to_delete.append(point_id)
                    isolated_count += 1

            if isolated_count > 0:
                click.echo(
                    f"  Marked {isolated_count} isolated points for deletion "
                    f"(NN distance > {threshold:.3f}, "
                    f"{filter_isolated_median_ratio:.1f}x median {nn_distance_median:.3f})"
                )

    for point_id in points_to_delete:
        densified_reconstruction.delete_point3D(point_id)

    points_after = len(densified_reconstruction.points3D)
    points_filtered = points_before - points_after

    click.echo(
        f"  Filtered {points_filtered} spurious points "
        f"({points_after} remaining, {100 * points_after / points_before:.1f}% kept)"
    )

    # Step 8: Run bundle adjustment again on filtered points
    click.echo(
        f"  Running final bundle adjustment on {points_after} filtered points..."
    )
    pycolmap.bundle_adjustment(densified_reconstruction, ba_config)
    densified_reconstruction.update_point_3d_errors()

    return densified_reconstruction


def _align_to_original(
    densified_reconstruction: pycolmap.Reconstruction,
    original_recon: SfmrReconstruction,
) -> pycolmap.Reconstruction:
    """Align densified reconstruction back to original coordinate space."""
    from ._align import ImageMatch, estimate_pairwise_alignment

    orig_quaternions_wxyz = original_recon.quaternions_wxyz
    orig_translations = original_recon.translations
    image_names = original_recon.image_names

    densified_name_to_id = {}
    for img_id, img in densified_reconstruction.images.items():
        densified_name_to_id[img.name] = img_id

    matches = []
    for orig_idx, orig_name in enumerate(image_names):
        if orig_name not in densified_name_to_id:
            continue

        densified_img_id = densified_name_to_id[orig_name]
        densified_img = densified_reconstruction.images[densified_img_id]

        # Get original camera pose
        orig_quat_wxyz = orig_quaternions_wxyz[orig_idx]
        orig_trans = orig_translations[orig_idx]
        orig_quat = RotQuaternion.from_wxyz_array(orig_quat_wxyz)
        orig_rot_matrix = orig_quat.to_rotation_matrix()
        orig_center = -orig_rot_matrix.T @ orig_trans

        # Get densified camera pose
        densified_pose = densified_img.cam_from_world()
        densified_quat_xyzw = densified_pose.rotation.quat
        densified_quat = RotQuaternion(
            densified_quat_xyzw[3],
            densified_quat_xyzw[0],
            densified_quat_xyzw[1],
            densified_quat_xyzw[2],
        )
        densified_trans = densified_pose.translation
        densified_rot_matrix = densified_quat.to_rotation_matrix()
        densified_center = -densified_rot_matrix.T @ densified_trans

        match = ImageMatch(
            image_name=orig_name,
            source_index=densified_img_id,
            target_index=orig_idx,
            source_quat=densified_quat,
            source_camera_center=densified_center,
            target_quat=orig_quat,
            target_camera_center=orig_center,
            quality=1.0,
        )
        matches.append(match)

    click.echo(f"  Found {len(matches)} matching images for alignment")

    if len(matches) < 2:
        click.echo(
            "  Warning: Fewer than 2 matching images, "
            "skipping alignment (coordinate drift may occur)"
        )
        return densified_reconstruction

    alignment_result = estimate_pairwise_alignment(
        matches=matches,
        confidence_threshold=0.0,
        source_id="densified",
        target_id="original",
    )

    click.echo(f"  Alignment RMS error: {alignment_result.total_rms_error:.6f}")
    click.echo(f"  Transform scale: {alignment_result.transform.scale:.6f}")

    # Apply transform using pycolmap's built-in method
    transform_quat = alignment_result.transform.rotation
    quat_xyzw = np.array(
        [transform_quat.x, transform_quat.y, transform_quat.z, transform_quat.w]
    )

    sim3d = pycolmap.Sim3d(
        scale=alignment_result.transform.scale,
        rotation=pycolmap.Rotation3d(quat_xyzw),
        translation=alignment_result.transform.translation,
    )

    densified_reconstruction.transform(sim3d)

    click.echo("  Alignment complete!")
    return densified_reconstruction


def densify_reconstruction(
    recon: SfmrReconstruction,
    max_features: int | None = None,
    sweep_window_size: int = 30,
    distance_threshold: float | None = None,
    ba_options: dict | None = None,
    filter_max_reproj_error: float = 4.0,
    filter_min_track_length: int = 3,
    filter_min_tri_angle: float = 1.5,
    filter_isolated_median_ratio: float = 2.0,
    close_pair_threshold: int = 4,
    max_close_pairs: int | None = None,
    max_distant_pairs: int = 5000,
    distant_pair_search_multiplier: int = 3,
    geometric_config: GeometricFilterConfig | None = None,
    include_frustum_pairs: bool = False,
) -> SfmrReconstruction:
    """Densify a reconstruction by finding additional feature correspondences.

    Default pipeline (covisibility-only):
    1. Find covisibility pairs (pairs already sharing 3D points)
    2. Prune pairs to manageable subset
    3. Sweep-match at higher features to find new correspondences
    4. Triangulate new points (preserving existing points)
    5. Merge duplicate points, bundle adjust, filter, bundle adjust again
    6. Align back to original coordinate space
    """
    image_names = recon.image_names
    camera_indexes = recon.camera_indexes
    quaternions = recon.quaternions_wxyz
    translations = recon.translations
    cameras_meta = recon.cameras
    workspace_dir = Path(recon.workspace_dir)

    cameras = [colmap_camera_from_intrinsics(cam) for cam in cameras_meta]

    # Find covisibility pairs
    click.echo("Finding covisibility pairs...")
    covisibility_pairs = build_covisibility_pairs(recon, angle_threshold_deg=90.0)
    click.echo(f"Found {len(covisibility_pairs)} covisibility pairs")

    # Prune covisibility pairs
    click.echo("\nPruning covisibility pairs...")
    covis_pairs_to_match = prune_image_pairs(
        covisibility_pairs,
        close_pair_threshold=close_pair_threshold,
        max_close_pairs=max_close_pairs,
        max_distant_pairs=max_distant_pairs,
        distant_pair_search_multiplier=distant_pair_search_multiplier,
    )

    if not covis_pairs_to_match and not include_frustum_pairs:
        click.echo("No covisibility pairs to match. Returning original reconstruction.")
        return recon

    # Sweep-match covisibility pairs
    click.echo(
        f"\nSweep-matching {len(covis_pairs_to_match)} covisibility pairs"
        f" at max-features={max_features}..."
    )
    all_matches = match_image_pairs(
        workspace_dir,
        image_names,
        covis_pairs_to_match,
        quaternions,
        translations,
        camera_indexes,
        cameras,
        max_features=max_features,
        window_size=sweep_window_size,
        distance_threshold=distance_threshold,
        geometric_config=geometric_config,
    )

    covis_total_matches = sum(len(m) for m in all_matches.values())
    covis_non_empty = sum(1 for m in all_matches.values() if len(m) > 0)
    click.echo(
        f"Found {covis_total_matches} matches across"
        f" {covis_non_empty}/{len(all_matches)} covisibility pairs"
    )

    # Optionally find and match frustum intersection pairs
    if include_frustum_pairs:
        click.echo("\nFinding frustum intersection pairs (geometric overlap)...")
        frustum_pairs = build_frustum_intersection_pairs(
            recon, near_percentile=5.0, far_percentile=95.0, num_samples=40
        )
        click.echo(f"Found {len(frustum_pairs)} frustum intersection pairs")

        covis_set = {(i, j) for i, j, _ in covisibility_pairs}
        frustum_only_pairs = [
            (i, j, score) for i, j, score in frustum_pairs if (i, j) not in covis_set
        ]
        click.echo(
            f"After removing covisibility pairs:"
            f" {len(frustum_only_pairs)} frustum-only pairs"
        )

        click.echo("\nPruning frustum-only pairs...")
        frustum_pairs_to_match = prune_image_pairs(
            frustum_only_pairs,
            close_pair_threshold=close_pair_threshold,
            max_close_pairs=max_close_pairs,
            max_distant_pairs=max_distant_pairs,
            distant_pair_search_multiplier=distant_pair_search_multiplier,
        )

        if frustum_pairs_to_match:
            click.echo(
                f"\nSweep-matching {len(frustum_pairs_to_match)} frustum-only pairs..."
            )
            frustum_matches = match_image_pairs(
                workspace_dir,
                image_names,
                frustum_pairs_to_match,
                quaternions,
                translations,
                camera_indexes,
                cameras,
                max_features=max_features,
                window_size=sweep_window_size,
                distance_threshold=distance_threshold,
                geometric_config=geometric_config,
            )

            frustum_total_matches = sum(len(m) for m in frustum_matches.values())
            click.echo(
                f"Found {frustum_total_matches} matches from"
                f" {len(frustum_matches)} frustum-only pairs"
            )

            all_matches.update(frustum_matches)

    total_matches = sum(len(m) for m in all_matches.values())
    click.echo(f"\nTotal: {total_matches} matches across {len(all_matches)} pairs")

    click.echo("\nTriangulating new points...")

    with tempfile.TemporaryDirectory() as temp_dir:
        densified_reconstruction = triangulate_new_tracks(
            new_matches=all_matches,
            max_features=max_features,
            workspace_dir=workspace_dir,
            temp_dir=Path(temp_dir),
            recon=recon,
            ba_options=ba_options,
            filter_max_reproj_error=filter_max_reproj_error,
            filter_min_track_length=filter_min_track_length,
            filter_min_tri_angle=filter_min_tri_angle,
            filter_isolated_median_ratio=filter_isolated_median_ratio,
        )

    new_points = len(densified_reconstruction.points3D)
    click.echo(f"\nDensified reconstruction: {new_points} points")

    # Align densified reconstruction back to original coordinate space
    click.echo("\nAligning to original coordinate space...")
    densified_reconstruction = _align_to_original(
        densified_reconstruction=densified_reconstruction,
        original_recon=recon,
    )

    # Convert to SfmrReconstruction
    from ._colmap_io import pycolmap_to_rust_sfmr

    workspace_dir_abs = workspace_dir.absolute()
    metadata = recon.source_metadata.copy()
    metadata["operation"] = "densify"

    # Update metadata counts to match the densified reconstruction
    num_images = len(densified_reconstruction.images)
    num_points = len(densified_reconstruction.points3D)
    num_cameras = len(densified_reconstruction.cameras)
    num_observations = sum(
        len(pt.track.elements) for pt in densified_reconstruction.points3D.values()
    )
    metadata["image_count"] = num_images
    metadata["points3d_count"] = num_points
    metadata["observation_count"] = num_observations
    metadata["camera_count"] = num_cameras

    result = pycolmap_to_rust_sfmr(
        densified_reconstruction, workspace_dir_abs, metadata
    )

    click.echo("Densification complete!")
    return result
