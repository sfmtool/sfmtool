# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera-based alignment for SfM reconstructions."""

import numpy as np

from ._align import ImageMatch
from ._sfmtool import RotQuaternion, SfmrReconstruction


def get_reconstruction_poses(
    recon: SfmrReconstruction,
) -> tuple[dict[int, RotQuaternion], dict[int, np.ndarray]]:
    """Get camera poses from a SfmrReconstruction.

    Returns:
        (quaternions_dict, camera_centers_dict) mapping image indices to poses.
        Camera centers are the actual camera positions in world coordinates.
    """
    quaternions = {}
    camera_centers = {}

    for i, (quat_wxyz, trans) in enumerate(
        zip(recon.quaternions_wxyz, recon.translations)
    ):
        quat = RotQuaternion.from_wxyz_array(quat_wxyz)
        quaternions[i] = quat
        camera_centers[i] = quat.camera_center(trans)

    return quaternions, camera_centers


def build_image_matches(
    shared_images: set[str],
    source_images: dict[int, str],
    source_quaternions: dict[int, RotQuaternion],
    source_camera_centers: dict[int, np.ndarray],
    target_images: dict[int, str],
    target_quaternions: dict[int, RotQuaternion],
    target_camera_centers: dict[int, np.ndarray],
) -> list[ImageMatch]:
    """Build ImageMatch objects for shared images between two reconstructions."""
    matches = []

    for img_name in shared_images:
        source_idx = None
        for idx, name in source_images.items():
            if name == img_name:
                source_idx = idx
                break

        target_idx = None
        for idx, name in target_images.items():
            if name == img_name:
                target_idx = idx
                break

        if source_idx is None or target_idx is None:
            continue

        match = ImageMatch(
            image_name=img_name,
            source_index=source_idx,
            target_index=target_idx,
            source_quat=source_quaternions[source_idx],
            source_camera_center=source_camera_centers[source_idx],
            target_quat=target_quaternions[target_idx],
            target_camera_center=target_camera_centers[target_idx],
            quality=0.95,
        )
        matches.append(match)

    return matches
