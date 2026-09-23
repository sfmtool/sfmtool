# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and utilities for xform tests."""

from pathlib import Path

import numpy as np

from sfmtool._sfmtool.geometry import RotQuaternion
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import apply_transforms as _apply_transforms


def apply_transforms_to_file(
    input_path: Path,
    output_path: Path,
    transforms: list,
) -> Path:
    """Helper that wraps apply_transforms with file I/O for tests."""
    recon = SfmrReconstruction.load(input_path)
    recon = _apply_transforms(recon, transforms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recon.save(output_path, operation="xform_test")
    return output_path


def load_reconstruction_data(reconstruction_path: Path) -> dict:
    """Load all data from a reconstruction for comparison."""
    recon = SfmrReconstruction.load(reconstruction_path)
    return {
        "image_names": recon.image_names,
        "positions": recon.positions,
        "colors": recon.colors,
        "errors": recon.errors,
        "quaternions_wxyz": recon.quaternions_wxyz,
        "translations": recon.translations,
        "observation_counts": recon.observation_counts,
        "image_count": recon.image_count,
        "point_count": recon.point_count,
        "observation_count": recon.observation_count,
    }


def inject_reprojection_outliers(
    sfmr_path: Path,
    output_path: Path,
    count: int,
    *,
    seed: int = 0,
    min_error_px: float = 5.0,
) -> np.ndarray:
    """Save a copy of a SIFT-backed reconstruction with ``count`` points moved off their tracks.

    The points are drawn with a fixed-seed generator. Each is moved sideways to
    the ray from the mean camera centre, by a fifth of its distance along that
    ray, so it no longer lands on its keypoints in any view. Every point's error
    is then recomputed from the ``.sift`` files, which is why ``sfmr_path`` must
    sit in its workspace. Returns the sorted indexes of the moved points. The
    function checks its own premise: each moved point reprojects worse than
    ``min_error_px``, and every other point keeps the error it had.
    """
    recon = SfmrReconstruction.load(sfmr_path)
    rng = np.random.default_rng(seed)
    indexes = np.sort(rng.choice(recon.point_count, size=count, replace=False))

    centers = np.array(
        [
            -RotQuaternion.from_wxyz_array(q).to_rotation_matrix().T @ t
            for q, t in zip(recon.quaternions_wxyz, recon.translations)
        ]
    )
    positions = np.array(recon.positions, dtype=np.float64)
    rays = positions[indexes] - centers.mean(axis=0)
    sideways = np.cross(rays, rng.normal(size=(count, 3)))
    sideways /= np.linalg.norm(sideways, axis=1, keepdims=True)
    positions[indexes] += 0.2 * np.linalg.norm(rays, axis=1, keepdims=True) * sideways

    moved = recon.clone_with_changes(positions=positions)
    moved.recompute_point_errors()
    errors_before = np.asarray(recon.errors)
    errors_after = np.asarray(moved.errors)
    untouched = np.ones(recon.point_count, dtype=bool)
    untouched[indexes] = False
    assert np.all(errors_after[indexes] > min_error_px)
    np.testing.assert_allclose(
        errors_after[untouched], errors_before[untouched], rtol=1e-5, atol=1e-5
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    moved.save(output_path, operation="xform_test")
    return indexes


def positions_are_scaled(
    original: np.ndarray, transformed: np.ndarray, scale: float, atol: float = 1e-6
) -> bool:
    """Check if transformed positions are correctly scaled."""
    expected = original * scale
    return np.allclose(transformed, expected, atol=atol)


def positions_are_translated(
    original: np.ndarray,
    transformed: np.ndarray,
    translation: np.ndarray,
    atol: float = 1e-6,
) -> bool:
    """Check if transformed positions are correctly translated."""
    expected = original + translation
    return np.allclose(transformed, expected, atol=atol)
