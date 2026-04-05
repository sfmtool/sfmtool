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


def estimate_similarity_with_orientations(
    matches: list[ImageMatch],
    weights: np.ndarray | None = None,
) -> Se3Transform:
    """Estimate similarity transform using both camera positions AND orientations.

    Args:
        matches: List of matched cameras with positions and orientations
        weights: Optional per-match weights (default: uniform)

    Returns:
        Se3Transform representing the estimated similarity transform
    """
    n = len(matches)
    if n < 1:
        raise ValueError("Need at least 1 match")

    if weights is None:
        weights = np.ones(n)
    weights = weights / np.sum(weights)

    # Step 1: Estimate rotation from camera orientations
    if n == 1:
        match = matches[0]
        R_s = match.source_quat.to_rotation_matrix()
        R_t = match.target_quat.to_rotation_matrix()
        rotation_matrix = R_t.T @ R_s
    else:
        # Multiple matches: weighted quaternion averaging
        rotation_quats = []
        for match in matches:
            R_s = match.source_quat.to_rotation_matrix()
            R_t = match.target_quat.to_rotation_matrix()
            R_rel = R_t.T @ R_s
            q_rel = RotQuaternion.from_rotation_matrix(R_rel)
            rotation_quats.append(q_rel)

        # Weighted quaternion averaging (Markley et al.)
        Q = np.zeros((4, 4))
        for q, w in zip(rotation_quats, weights):
            q_array = np.asarray(q.to_wxyz_array())
            Q += w * np.outer(q_array, q_array)

        eigenvalues, eigenvectors = np.linalg.eigh(Q)
        avg_quat_array = eigenvectors[:, -1]
        avg_quat = RotQuaternion.from_wxyz_array(avg_quat_array)
        rotation_matrix = avg_quat.to_rotation_matrix()

    # Step 2: Given rotation R, solve for scale s and translation t
    source_positions = np.array([m.source_camera_center for m in matches])
    target_positions = np.array([m.target_camera_center for m in matches])

    source_rotated = source_positions @ rotation_matrix.T
    source_rotated_centroid = np.average(source_rotated, axis=0, weights=weights)
    target_centroid = np.average(target_positions, axis=0, weights=weights)

    source_rotated_centered = source_rotated - source_rotated_centroid
    target_centered = target_positions - target_centroid

    numerator = np.sum(
        weights[:, np.newaxis] * target_centered * source_rotated_centered
    )
    denominator = np.sum(
        weights[:, np.newaxis] * source_rotated_centered * source_rotated_centered
    )

    if denominator <= 0:
        scale = 1.0
    else:
        scale = numerator / denominator

    translation = target_centroid - scale * source_rotated_centroid
    rotation_quat = RotQuaternion.from_rotation_matrix(rotation_matrix)
    return Se3Transform(rotation=rotation_quat, translation=translation, scale=scale)


def estimate_pairwise_alignment(
    matches: list[ImageMatch],
    confidence_threshold: float = 0.7,
    source_id: str | None = None,
    target_id: str | None = None,
) -> AlignmentResult:
    """Estimate SE(3) alignment between two reconstructions.

    Args:
        matches: List of matched images between reconstructions
        confidence_threshold: Minimum quality score to include match
        source_id: ID of source reconstruction
        target_id: ID of target reconstruction

    Returns:
        AlignmentResult with estimated transform
    """
    if not matches:
        raise ValueError("No matches provided for alignment")

    valid_matches = [m for m in matches if m.quality >= confidence_threshold]
    if not valid_matches:
        raise ValueError(
            f"No matches above confidence threshold {confidence_threshold}"
        )

    if len(valid_matches) == 1:
        match = valid_matches[0]
        source_quat = match.source_quat
        target_quat = match.target_quat
        rotation_quat = target_quat.conjugate() * source_quat
        rotation_matrix = rotation_quat.to_rotation_matrix()

        source_pos = np.array(match.source_camera_center)
        target_pos = np.array(match.target_camera_center)
        translation = target_pos - rotation_matrix @ source_pos
        scale = 1.0

        transform = Se3Transform(
            rotation=rotation_quat, translation=translation, scale=scale
        )
    else:
        weights = np.array([m.quality for m in valid_matches])
        transform = estimate_similarity_with_orientations(valid_matches, weights)

    if source_id is None:
        source_id = "unknown_source"
    if target_id is None:
        target_id = "unknown_target"

    return AlignmentResult(source_id, target_id, transform, valid_matches)


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
