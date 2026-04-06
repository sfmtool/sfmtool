# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and utilities for xform tests."""

from pathlib import Path

import numpy as np

from sfmtool._sfmtool import SfmrReconstruction
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
