# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for geometric transforms: Rotate, Translate, Scale, Similarity."""

import numpy as np
import pytest

from sfmtool._sfmtool import RotQuaternion, Se3Transform
from sfmtool.xform import (
    RotateTransform,
    ScaleTransform,
    SimilarityTransform,
    TranslateTransform,
)

from .conftest import apply_transforms_to_file, load_reconstruction_data


# =============================================================================
# RotateTransform Tests
# =============================================================================


def test_rotate_transform(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that rotation transform works."""
    output_path = tmp_path / "rotated.sfmr"

    transforms = [RotateTransform(np.array([0, 1, 0]), np.radians(90))]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert transformed["point_count"] == original["point_count"]
    assert not np.allclose(transformed["positions"], original["positions"])


def test_zero_rotation_axis_raises_error():
    """Test that zero rotation axis raises error."""
    with pytest.raises(ValueError):
        RotateTransform(np.array([0, 0, 0]), np.radians(90))


def test_rotate_transform_identity(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that zero-angle rotation preserves positions."""
    output_path = tmp_path / "rotated_identity.sfmr"

    transforms = [RotateTransform(np.array([0, 1, 0]), 0.0)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert np.allclose(transformed["positions"], original["positions"], atol=1e-6)


def test_rotate_transform_180_degrees(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test 180 degree rotation around Z axis."""
    output_path = tmp_path / "rotated_180.sfmr"

    transforms = [RotateTransform(np.array([0, 0, 1]), np.radians(180))]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    # For 180 degree rotation around Z: (x, y, z) -> (-x, -y, z)
    expected_positions = original["positions"].copy()
    expected_positions[:, 0] = -original["positions"][:, 0]
    expected_positions[:, 1] = -original["positions"][:, 1]

    assert np.allclose(transformed["positions"], expected_positions, atol=1e-6)


# =============================================================================
# TranslateTransform Tests
# =============================================================================


def test_translate_transform(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that translation transform works."""
    output_path = tmp_path / "translated.sfmr"

    transforms = [TranslateTransform(np.array([1, 2, 3]))]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert transformed["point_count"] == original["point_count"]

    expected_positions = original["positions"] + np.array([1, 2, 3])
    assert np.allclose(transformed["positions"], expected_positions, atol=1e-6)


def test_translate_transform_zero(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that zero translation preserves positions."""
    output_path = tmp_path / "translated_zero.sfmr"

    transforms = [TranslateTransform(np.array([0, 0, 0]))]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert np.allclose(transformed["positions"], original["positions"], atol=1e-6)


def test_translate_transform_negative(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test translation with negative values."""
    output_path = tmp_path / "translated_negative.sfmr"

    translation = np.array([-5.5, -10.0, -2.5])
    transforms = [TranslateTransform(translation)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    expected_positions = original["positions"] + translation
    assert np.allclose(transformed["positions"], expected_positions, atol=1e-6)


# =============================================================================
# ScaleTransform Tests
# =============================================================================


def test_scale_transform(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that scale transform works."""
    output_path = tmp_path / "scaled.sfmr"

    transforms = [ScaleTransform(2.0)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert transformed["point_count"] == original["point_count"]

    expected_positions = original["positions"] * 2.0
    assert np.allclose(transformed["positions"], expected_positions, atol=1e-6)


def test_invalid_scale_raises_error():
    """Test that invalid scale factor raises error."""
    with pytest.raises(ValueError, match="Scale factor must be positive"):
        ScaleTransform(0.0)

    with pytest.raises(ValueError, match="Scale factor must be positive"):
        ScaleTransform(-1.0)


def test_scale_transform_unit(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that scale of 1.0 preserves positions."""
    output_path = tmp_path / "scaled_unit.sfmr"

    transforms = [ScaleTransform(1.0)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert np.allclose(transformed["positions"], original["positions"], atol=1e-6)


def test_scale_transform_fractional(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test scaling with a fractional value."""
    output_path = tmp_path / "scaled_half.sfmr"

    transforms = [ScaleTransform(0.5)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    expected_positions = original["positions"] * 0.5
    assert np.allclose(transformed["positions"], expected_positions, atol=1e-6)


# =============================================================================
# SimilarityTransform Tests
# =============================================================================


def test_similarity_transform_identity(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that identity similarity transform preserves reconstruction."""
    output_path = tmp_path / "similarity_identity.sfmr"

    identity = Se3Transform()
    transforms = [SimilarityTransform(identity)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert np.allclose(transformed["positions"], original["positions"], atol=1e-6)
    assert np.allclose(
        transformed["quaternions_wxyz"], original["quaternions_wxyz"], atol=1e-6
    )
    assert np.allclose(transformed["translations"], original["translations"], atol=1e-6)


def test_similarity_transform_scale_only(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test similarity transform with scale only."""
    output_path = tmp_path / "similarity_scaled.sfmr"

    transform = Se3Transform(scale=2.0)
    transforms = [SimilarityTransform(transform)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    expected_positions = original["positions"] * 2.0
    assert np.allclose(transformed["positions"], expected_positions, atol=1e-6)


def test_similarity_transform_translation_only(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test similarity transform with translation only."""
    output_path = tmp_path / "similarity_translated.sfmr"

    translation = np.array([10.0, -5.0, 3.0])
    transform = Se3Transform(translation=translation)
    transforms = [SimilarityTransform(transform)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    expected_positions = original["positions"] + translation
    assert np.allclose(transformed["positions"], expected_positions, atol=1e-6)


def test_similarity_transform_combined(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test similarity transform with rotation, translation, and scale combined."""
    output_path = tmp_path / "similarity_combined.sfmr"

    angle = np.pi / 2  # 90 degrees
    rotation_quat = RotQuaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))
    translation = np.array([1.0, 2.0, 3.0])
    scale = 1.5

    transform = Se3Transform(
        rotation=rotation_quat, translation=translation, scale=scale
    )
    transforms = [SimilarityTransform(transform)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert transformed["point_count"] == original["point_count"]
    assert transformed["image_count"] == original["image_count"]
    assert not np.allclose(transformed["positions"], original["positions"])


def test_similarity_transform_description():
    """Test the description method of SimilarityTransform."""
    transform = Se3Transform(
        translation=np.array([1.0, 2.0, 3.0]),
        scale=2.0,
    )
    sim_transform = SimilarityTransform(transform)
    desc = sim_transform.description()

    assert "Similarity transform" in desc
    assert "scale=2.000" in desc
    assert "translation" in desc


# =============================================================================
# Transform Order Tests
# =============================================================================


def test_order_matters(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that transformation order matters (rotate-translate vs translate-rotate)."""
    output1_path = tmp_path / "rotate_then_translate.sfmr"
    output2_path = tmp_path / "translate_then_rotate.sfmr"

    transforms1 = [
        RotateTransform(np.array([0, 1, 0]), np.radians(90)),
        TranslateTransform(np.array([5, 0, 0])),
    ]

    transforms2 = [
        TranslateTransform(np.array([5, 0, 0])),
        RotateTransform(np.array([0, 1, 0]), np.radians(90)),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output1_path, transforms1
    )
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output2_path, transforms2
    )

    result1 = load_reconstruction_data(output1_path)
    result2 = load_reconstruction_data(output2_path)

    assert not np.allclose(result1["positions"], result2["positions"])


def test_scale_then_translate_vs_translate_then_scale(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that scale-translate vs translate-scale produce different results."""
    output1_path = tmp_path / "scale_then_translate.sfmr"
    output2_path = tmp_path / "translate_then_scale.sfmr"

    transforms1 = [
        ScaleTransform(2.0),
        TranslateTransform(np.array([10, 0, 0])),
    ]

    transforms2 = [
        TranslateTransform(np.array([10, 0, 0])),
        ScaleTransform(2.0),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output1_path, transforms1
    )
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output2_path, transforms2
    )

    result1 = load_reconstruction_data(output1_path)
    result2 = load_reconstruction_data(output2_path)

    assert not np.allclose(result1["positions"], result2["positions"])
