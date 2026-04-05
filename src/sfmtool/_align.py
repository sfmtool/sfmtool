# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Core alignment functionality for multiple SfM reconstructions."""

from dataclasses import dataclass

import numpy as np

from ._sfmtool import RotQuaternion, Se3Transform


@dataclass
class ImageMatch:
    """Information about a matched image between two reconstructions."""

    image_name: str
    source_index: int
    target_index: int
    source_quat: RotQuaternion
    source_camera_center: np.ndarray
    target_quat: RotQuaternion
    target_camera_center: np.ndarray
    quality: float = 1.0


class AlignmentResult:
    """Result of aligning one reconstruction to another."""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        transform: Se3Transform,
        matches: list[ImageMatch],
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.transform = transform
        self.matches = matches
        self._compute_residuals()

    def _compute_residuals(self):
        """Compute per-image residuals for matched poses."""
        self.per_image_errors = []
        self.total_rms_error = 0.0

        if not self.matches:
            return

        errors = []
        for match in self.matches:
            source_pos = match.source_camera_center
            source_rot = match.source_quat

            aligned_pos = self.transform @ source_pos
            aligned_rot_matrix = (
                self.transform.rotation.to_rotation_matrix()
                @ source_rot.to_rotation_matrix()
            )

            trans_error = np.linalg.norm(aligned_pos - match.target_camera_center)
            rot_matrix_error = (
                match.target_quat.to_rotation_matrix() - aligned_rot_matrix
            )
            rot_error = np.linalg.norm(rot_matrix_error) / np.sqrt(3)

            combined_error = trans_error + 0.1 * rot_error
            errors.append(combined_error)
            self.per_image_errors.append(combined_error)

        self.total_rms_error = np.sqrt(np.mean(np.array(errors) ** 2))
        self.confidence = max(0.0, 1.0 - self.total_rms_error / 0.1)


def kabsch_algorithm(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Se3Transform:
    """Kabsch algorithm for finding optimal SE(3) transformation.

    Finds the rotation, translation, and scale that best aligns
    source_points to target_points.

    Args:
        source_points: (N, 3) array of source points
        target_points: (N, 3) array of target points

    Returns:
        Se3Transform representing the optimal similarity transform
    """
    if source_points.shape != target_points.shape or source_points.shape[0] < 2:
        raise ValueError("Need at least 2 matching points with same shape")

    from ._sfmtool import kabsch_algorithm_rs

    return kabsch_algorithm_rs(source_points, target_points)
