# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert SfM reconstruction to COLMAP database with camera and image pose priors."""

import os

import numpy as np
from pathlib import Path

from ._cameras import colmap_camera_from_intrinsics, get_intrinsic_matrix
from ._filenames import normalize_workspace_path
from ._image_pair_graph import build_covisibility_pairs
from ._sift_file import SiftReader, image_files_to_sift_files
from ._sfmtool import RotQuaternion, write_colmap_db

__all__ = ["create_colmap_db_from_reconstruction"]


def _compute_two_view_geometry(
    img_idx1: int,
    img_idx2: int,
    quaternions_wxyz: np.ndarray,
    translations: np.ndarray,
    camera_indexes: np.ndarray,
    pycolmap_cameras: dict,
) -> tuple:
    """Compute relative pose and fundamental matrix for an image pair.

    Returns:
        Tuple of (qvec_wxyz, tvec, F)
    """
    import pycolmap

    cam_idx1 = int(camera_indexes[img_idx1])
    cam_idx2 = int(camera_indexes[img_idx2])
    camera1 = pycolmap_cameras[cam_idx1]
    camera2 = pycolmap_cameras[cam_idx2]

    # Convert quaternions from WXYZ to XYZW (pycolmap format)
    quat1_wxyz = quaternions_wxyz[img_idx1]
    quat1_xyzw = quat1_wxyz[[1, 2, 3, 0]]
    quat2_wxyz = quaternions_wxyz[img_idx2]
    quat2_xyzw = quat2_wxyz[[1, 2, 3, 0]]

    img1_cam_from_world = pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(quat1_xyzw),
        translation=translations[img_idx1],
    )
    img2_cam_from_world = pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(quat2_xyzw),
        translation=translations[img_idx2],
    )

    # cam2_from_cam1 = cam2_from_world * cam1_from_world^-1
    cam2_from_cam1 = img2_cam_from_world * img1_cam_from_world.inverse()

    # Compute fundamental matrix: F = K2^-T * E * K1^-1
    K1 = get_intrinsic_matrix(camera1)
    K2 = get_intrinsic_matrix(camera2)
    R_rel = cam2_from_cam1.rotation.matrix()
    t_rel = np.array(cam2_from_cam1.translation)
    tx = np.array(
        [[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]]
    )
    E = tx @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # Extract quaternion in WXYZ order
    quat_xyzw = cam2_from_cam1.rotation.quat
    qvec_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )
    tvec = np.array(cam2_from_cam1.translation, dtype=np.float64)

    return qvec_wxyz, tvec, F


def create_colmap_db_from_reconstruction(
    recon,
    output_db_path: str | Path,
    max_feature_count: int | None = None,
    populate_two_view_geometries: bool = True,
) -> str:
    """Create a COLMAP database from an SfM reconstruction with camera and pose priors.

    The database contains camera intrinsics, images with pose priors,
    keypoints/descriptors from .sift files, and optionally two-view geometries
    with fundamental matrices computed from relative poses.

    Args:
        recon: SfmrReconstruction containing camera parameters, poses, and tracks
        output_db_path: Path where the COLMAP database should be created
        max_feature_count: Maximum features per image (None = all)
        populate_two_view_geometries: Pre-populate F matrices for guided matching

    Returns:
        Path to the created database file
    """
    workspace_dir = Path(recon.workspace_dir)
    output_db_path = Path(output_db_path)

    print("Creating COLMAP database from reconstruction...")

    cameras_meta = recon.cameras
    image_names = recon.image_names
    camera_indexes = recon.camera_indexes
    quaternions_wxyz = recon.quaternions_wxyz
    translations = recon.translations

    if populate_two_view_geometries:
        covisibility_pairs = build_covisibility_pairs(recon, angle_threshold_deg=180.0)

    print(f"Found {len(image_names)} images and {len(cameras_meta)} cameras")

    # Resolve absolute image paths
    image_paths = []
    for rel_path in image_names:
        abs_path = Path(os.path.normpath(os.path.abspath(workspace_dir / rel_path)))
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")
        image_paths.append(abs_path)

    # Determine common image directory
    try:
        image_dir = Path(os.path.commonpath([str(p) for p in image_paths]))
    except ValueError:
        image_dir = image_paths[0].parent

    if image_dir.is_file():
        image_dir = image_dir.parent

    print(f"Image directory: {image_dir}")

    # Compute relative image names for the database
    db_image_names = []
    for image_path in image_paths:
        rel_path_raw = image_path.relative_to(image_dir)
        rel_name = normalize_workspace_path(rel_path_raw)
        db_image_names.append(rel_name)

    # Load keypoints and descriptors from .sift files
    print("Loading keypoints and descriptors...")
    feature_prefix_dir = (
        recon.source_metadata.get("workspace", {})
        .get("contents", {})
        .get("feature_prefix_dir")
    )
    sift_paths = image_files_to_sift_files(
        image_paths, feature_prefix_dir=feature_prefix_dir
    )

    keypoints_per_image = []
    descriptors_per_image = []
    for i, sift_path in enumerate(sift_paths):
        with SiftReader(sift_path) as reader:
            keypoints = reader.read_positions(count=max_feature_count)
            descriptors = reader.read_descriptors(count=max_feature_count)
            keypoints_per_image.append(keypoints.astype(np.float64))
            descriptors_per_image.append(descriptors)

        if (i + 1) % 100 == 0 or i == len(sift_paths) - 1:
            print(f"  Loaded {i + 1}/{len(sift_paths)} images")

    # Compute pose priors
    print("Computing pose priors...")
    pose_priors = []
    position_covariance = 0.01 * np.eye(3)
    for i in range(len(image_names)):
        quat_wxyz = quaternions_wxyz[i]
        translation = translations[i]
        camera_center = RotQuaternion.from_wxyz_array(quat_wxyz).camera_center(
            translation
        )
        pose_priors.append(
            {
                "position": np.asarray(camera_center, dtype=np.float64),
                "position_covariance": position_covariance,
                "coordinate_system": 2,  # CARTESIAN
            }
        )

    # Build two-view geometries if requested
    two_view_geometries = None
    if populate_two_view_geometries:
        print("\nComputing two-view geometries from reconstruction...")
        unique_pairs = [(i, j) for i, j, _count in covisibility_pairs]
        print(f"  Found {len(unique_pairs)} image pairs with common 3D points")

        pycolmap_cameras = {}
        for idx, camera_meta in enumerate(cameras_meta):
            camera = colmap_camera_from_intrinsics(camera_meta)
            pycolmap_cameras[idx] = camera

        two_view_geometries = []
        for img_idx1, img_idx2 in unique_pairs:
            qvec_wxyz, tvec, F = _compute_two_view_geometry(
                img_idx1,
                img_idx2,
                quaternions_wxyz,
                translations,
                camera_indexes,
                pycolmap_cameras,
            )
            two_view_geometries.append(
                {
                    "image_idx1": int(img_idx1),
                    "image_idx2": int(img_idx2),
                    "config": 2,  # CALIBRATED
                    "f_matrix": F.astype(np.float64),
                    "qvec_wxyz": qvec_wxyz,
                    "tvec": tvec,
                }
            )
        print(f"  Computed {len(two_view_geometries)} two-view geometries")

    # Build data dict for the Rust writer
    data = {
        "cameras": cameras_meta,
        "image_names": db_image_names,
        "camera_indexes": np.asarray(camera_indexes, dtype=np.uint32),
        "quaternions_wxyz": np.asarray(quaternions_wxyz, dtype=np.float64),
        "translations_xyz": np.asarray(translations, dtype=np.float64),
        "keypoints_per_image": keypoints_per_image,
        "descriptors_per_image": descriptors_per_image,
        "descriptor_dim": 128,
        "pose_priors": pose_priors,
    }
    if two_view_geometries is not None:
        data["two_view_geometries"] = two_view_geometries

    print(f"\nWriting database: {output_db_path}")
    write_colmap_db(str(output_db_path), data)

    pairs_count = len(two_view_geometries) if two_view_geometries else 0

    print(f"\nSuccessfully created database: {output_db_path}")
    print(f"  Images: {len(image_names)}")
    print(f"  Cameras: {len(cameras_meta)}")
    if max_feature_count:
        print(f"  Max features per image: {max_feature_count}")
    if populate_two_view_geometries:
        print(
            f"  Two-view geometries: {pairs_count} (use guided_matching=True in COLMAP)"
        )

    return str(output_db_path)
