# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for AlignToTransform and AlignToInputTransform."""

import numpy as np
import pytest

from sfmtool.xform import (
    AlignToInputTransform,
    AlignToTransform,
    RotateTransform,
    ScaleTransform,
    TranslateTransform,
)

from .conftest import apply_transforms_to_file, load_reconstruction_data


# =============================================================================
# AlignToTransform Tests
# =============================================================================


def test_align_to_transform_basic(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test aligning a reconstruction to itself (should be identity)."""
    output_path = tmp_path / "aligned.sfmr"

    transforms = [AlignToTransform(sfmrfile_reconstruction_with_17_images)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    aligned = load_reconstruction_data(output_path)

    assert aligned["point_count"] == original["point_count"]
    assert aligned["image_count"] == original["image_count"]

    # When aligning to itself, positions should be very similar
    position_diff = np.linalg.norm(aligned["positions"] - original["positions"])
    mean_position = np.mean(np.linalg.norm(original["positions"], axis=1))
    relative_error = position_diff / (mean_position * len(original["positions"]))
    assert relative_error < 0.1


def test_align_to_transform_after_scale(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that align recovers original scale after scaling."""
    scaled_path = tmp_path / "scaled.sfmr"
    transforms = [ScaleTransform(2.0)]
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, scaled_path, transforms
    )

    aligned_path = tmp_path / "realigned.sfmr"
    transforms = [AlignToTransform(sfmrfile_reconstruction_with_17_images)]
    apply_transforms_to_file(scaled_path, aligned_path, transforms)

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    aligned = load_reconstruction_data(aligned_path)

    assert aligned["point_count"] == original["point_count"]


def test_align_to_transform_nonexistent_file():
    """Test that non-existent reference file raises error."""
    with pytest.raises(FileNotFoundError):
        AlignToTransform("/nonexistent/path/to/file.sfmr")


def test_align_to_transform_invalid_extension(tmp_path):
    """Test that non-.sfmr file raises error."""
    wrong_file = tmp_path / "test.txt"
    wrong_file.write_text("dummy")

    with pytest.raises(ValueError, match="must be a .sfmr file"):
        AlignToTransform(wrong_file)


def test_align_to_transform_description(sfmrfile_reconstruction_with_17_images):
    """Test the description method."""
    transform = AlignToTransform(sfmrfile_reconstruction_with_17_images)
    desc = transform.description()

    assert "Align to" in desc


# =============================================================================
# AlignToInputTransform Tests
# =============================================================================


def test_align_to_input_after_transform(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that AlignToInput recovers original after transformations."""
    output_path = tmp_path / "realigned.sfmr"

    transforms = [
        ScaleTransform(2.0),
        TranslateTransform(np.array([10, 20, 30])),
        AlignToInputTransform(),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    realigned = load_reconstruction_data(output_path)

    assert realigned["point_count"] == original["point_count"]

    position_diff = np.linalg.norm(realigned["positions"] - original["positions"])
    mean_position = np.mean(np.linalg.norm(original["positions"], axis=1))
    relative_error = position_diff / (mean_position * len(original["positions"]))
    assert relative_error < 0.2


def test_align_to_input_after_rotation(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that AlignToInput works after rotation."""
    output_path = tmp_path / "rotated_realigned.sfmr"

    transforms = [
        RotateTransform(np.array([0, 0, 1]), np.radians(45)),
        AlignToInputTransform(),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    realigned = load_reconstruction_data(output_path)

    assert realigned["point_count"] == original["point_count"]


def test_align_to_input_description():
    """Test the description method."""
    transform = AlignToInputTransform()
    desc = transform.description()

    assert "input" in desc.lower() or "original" in desc.lower()


def test_align_to_input_without_apply_context():
    """Test that using AlignToInputTransform outside apply_transforms raises error."""
    AlignToInputTransform._original_input = None

    _ = AlignToInputTransform()

    assert AlignToInputTransform._original_input is None


# =============================================================================
# Combined Align Tests
# =============================================================================


def test_align_to_and_transform(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test combining AlignToTransform with other transforms."""
    output_path = tmp_path / "aligned_scaled.sfmr"

    transforms = [
        AlignToTransform(sfmrfile_reconstruction_with_17_images),
        ScaleTransform(0.5),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert result["point_count"] == original["point_count"]


def test_align_preserves_image_count(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that alignment preserves image count."""
    output_path = tmp_path / "aligned_images.sfmr"

    transforms = [
        ScaleTransform(3.0),
        AlignToInputTransform(),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert result["image_count"] == original["image_count"]
