# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and utilities for xform tests."""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.geometry import RotQuaternion
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import apply_transforms as _apply_transforms

from ..conftest import SEOUL_BULL_POINT_STEP, every_nth_point


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
        "rms_reprojection_errors": recon.rms_reprojection_errors,
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
    errors_before = np.asarray(recon.rms_reprojection_errors)
    errors_after = np.asarray(moved.rms_reprojection_errors)
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


@pytest.fixture(scope="session")
def seoul_bull_normals_refined(seoul_bull_workspace_once: Path):
    """Every eighth point of the seoul_bull workspace (``SEOUL_BULL_POINT_STEP``)
    and point 1, converted to ``embedded_patches`` (mean-viewing normals, no
    bitmaps), and ``RefineNormalsTransform()``'s output on it, built once for
    the tests that only read them. Read them; do not modify them or their
    source workspace.

    Point 1, row 1 of the subset, is made a point at infinity in the direction
    from the mean camera centre to where it was. The refinement skips points at
    infinity, so it leaves at least that row unchanged on every platform. Which
    finite points it leaves unchanged depends on the triangulation, which
    differs between platforms, and on some it turns every finite point."""
    from sfmtool.xform import RefineNormalsTransform

    subset = every_nth_point(
        SfmrReconstruction.load(seoul_bull_workspace_once),
        SEOUL_BULL_POINT_STEP,
        keep=[1],
    )
    centers = np.array(
        [
            -RotQuaternion.from_wxyz_array(q).to_rotation_matrix().T @ t
            for q, t in zip(subset.quaternions_wxyz, subset.translations)
        ]
    )
    positions = np.array(subset.positions_xyzw, dtype=np.float64)
    direction = positions[1, :3] - centers.mean(axis=0)
    positions[1] = np.append(direction / np.linalg.norm(direction), 0.0)
    recon = subset.clone_with_changes(positions=positions).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    assert np.asarray(recon.point_is_at_infinity).nonzero()[0].tolist() == [1]
    return recon, RefineNormalsTransform().apply(recon)
