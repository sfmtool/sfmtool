# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from sfmtool._sift_file import (
    SiftReader,
    compute_orientation,
    feature_size,
    feature_size_x,
    feature_size_y,
    get_feature_tool_xxh128,
    get_feature_type_for_tool,
    get_sift_path_for_image,
    write_sift,
    xxh128_of_file,
)


# ---------------------------------------------------------------------------
# write / read roundtrip tests
# ---------------------------------------------------------------------------


def test_sift_write_read_roundtrip(sample_sift_data, tmp_path):
    """Writing a .sift file and reading it back yields the original data."""
    feature_tool_metadata, metadata, position, affine_shape, descriptor, thumbnail = (
        sample_sift_data
    )

    sift_filename = tmp_path / "test.sift"

    write_sift(
        sift_filename,
        feature_tool_metadata,
        metadata,
        position,
        affine_shape,
        descriptor,
        thumbnail,
    )

    with SiftReader(sift_filename) as reader:
        assert reader.metadata == metadata

        read_position, read_affine_shape = reader.read_positions_and_shapes()
        read_descriptor = reader.read_descriptors()
        read_thumbnail = reader.read_thumbnail()

        np.testing.assert_array_equal(position, read_position)
        np.testing.assert_array_equal(affine_shape, read_affine_shape)
        np.testing.assert_array_equal(descriptor, read_descriptor)
        np.testing.assert_array_equal(thumbnail, read_thumbnail)


def test_sift_reader_partial_read(sample_sift_data, tmp_path):
    """Reading a subset of features via the count parameter works."""
    feature_tool_metadata, metadata, position, affine_shape, descriptor, thumbnail = (
        sample_sift_data
    )
    read_count = 10
    assert metadata["feature_count"] > read_count

    sift_filename = tmp_path / "test.sift"
    write_sift(
        sift_filename,
        feature_tool_metadata,
        metadata,
        position,
        affine_shape,
        descriptor,
        thumbnail,
    )

    with SiftReader(sift_filename) as reader:
        read_position, read_affine_shape = reader.read_positions_and_shapes(
            count=read_count
        )
        read_descriptor = reader.read_descriptors(count=read_count)

        assert read_position.shape[0] == read_count
        assert read_affine_shape.shape[0] == read_count
        assert read_descriptor.shape[0] == read_count

        np.testing.assert_array_equal(position[:read_count], read_position)
        np.testing.assert_array_equal(affine_shape[:read_count], read_affine_shape)
        np.testing.assert_array_equal(descriptor[:read_count], read_descriptor)


def test_sift_reader_partial_read_oversize(sample_sift_data, tmp_path):
    """Reading more features than available truncates to the total count."""
    feature_tool_metadata, metadata, position, affine_shape, descriptor, thumbnail = (
        sample_sift_data
    )
    read_count = metadata["feature_count"] + 50

    sift_filename = tmp_path / "test.sift"
    write_sift(
        sift_filename,
        feature_tool_metadata,
        metadata,
        position,
        affine_shape,
        descriptor,
        thumbnail,
    )

    with SiftReader(sift_filename) as reader:
        read_position = reader.read_positions(count=read_count)
        assert read_position.shape[0] == metadata["feature_count"]
        np.testing.assert_array_equal(position, read_position)


# ---------------------------------------------------------------------------
# write_sift validation tests
# ---------------------------------------------------------------------------


def test_write_sift_validation(sample_sift_data, tmp_path):
    """write_sift raises ValueErrors for invalid inputs."""
    feature_tool_metadata, metadata, position, affine_shape, descriptor, thumbnail = (
        sample_sift_data
    )

    tmp_sift_filename = tmp_path / "test.sift"

    # Missing metadata key
    bad_metadata = metadata.copy()
    del bad_metadata["image_width"]
    with pytest.raises(ValueError, match="Missing: image_width"):
        write_sift(
            tmp_sift_filename,
            feature_tool_metadata,
            bad_metadata,
            position,
            affine_shape,
            descriptor,
            thumbnail,
        )

    # Extra metadata key
    bad_metadata = metadata.copy()
    bad_metadata["extra_field"] = "bad"
    with pytest.raises(ValueError, match="Extra: extra_field"):
        write_sift(
            tmp_sift_filename,
            feature_tool_metadata,
            bad_metadata,
            position,
            affine_shape,
            descriptor,
            thumbnail,
        )

    # Wrong metadata type
    bad_metadata = metadata.copy()
    bad_metadata["feature_count"] = "100"
    with pytest.raises(
        ValueError, match="has type <class 'str'>, expected <class 'int'>"
    ):
        write_sift(
            tmp_sift_filename,
            feature_tool_metadata,
            bad_metadata,
            position,
            affine_shape,
            descriptor,
            thumbnail,
        )

    # Mismatched feature counts
    with pytest.raises(ValueError, match="Lengths of input arrays don't match"):
        write_sift(
            tmp_sift_filename,
            feature_tool_metadata,
            metadata,
            position[:10],
            affine_shape,
            descriptor,
            thumbnail,
        )

    # Wrong position shape
    bad_position = np.zeros((metadata["feature_count"], 1), dtype=np.float32)
    with pytest.raises(
        ValueError,
        match="Data shape .* does not match required feature count and element shape",
    ):
        write_sift(
            tmp_sift_filename,
            feature_tool_metadata,
            metadata,
            bad_position,
            affine_shape,
            descriptor,
            thumbnail,
        )

    # Wrong descriptor dtype
    with pytest.raises(
        ValueError, match="Data dtype .* does not match required element dtype"
    ):
        write_sift(
            tmp_sift_filename,
            feature_tool_metadata,
            metadata,
            position,
            affine_shape,
            descriptor.astype(np.float32),
            thumbnail,
        )


# ---------------------------------------------------------------------------
# get_feature_type_for_tool tests
# ---------------------------------------------------------------------------


class TestGetFeatureTypeForTool:
    def test_opencv(self):
        assert get_feature_type_for_tool("opencv", {}) == "sift-opencv"

    def test_colmap_default(self):
        assert get_feature_type_for_tool("colmap", {}) == "sift-colmap"

    def test_colmap_dsp(self):
        assert (
            get_feature_type_for_tool("colmap", {"domain_size_pooling": True})
            == "sift-colmap-dsp"
        )

    def test_colmap_max_features(self):
        assert (
            get_feature_type_for_tool("colmap", {"max_num_features": 500})
            == "sift-colmap-max500"
        )

    def test_colmap_dsp_and_max(self):
        opts = {"domain_size_pooling": True, "max_num_features": 500}
        assert get_feature_type_for_tool("colmap", opts) == "sift-colmap-dsp-max500"

    def test_colmap_default_max_features_omitted(self):
        # 8192 is the COLMAP default — should not appear in the name
        assert (
            get_feature_type_for_tool("colmap", {"max_num_features": 8192})
            == "sift-colmap"
        )

    def test_unknown_tool(self):
        assert get_feature_type_for_tool("superpoint", {}) == "sift-superpoint"


# ---------------------------------------------------------------------------
# get_sift_path_for_image tests
# ---------------------------------------------------------------------------


def test_get_sift_path_for_image_with_tool(tmp_path):
    """get_sift_path_for_image returns correct path when tool is specified."""
    image_path = tmp_path / "image.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # With explicit tool and options
    opts = {"domain_size_pooling": True}
    path = get_sift_path_for_image(image_path, feature_tool="colmap", feature_options=opts)
    expected_hash = get_feature_tool_xxh128("colmap", "sift-colmap-dsp", opts)
    assert f"sift-colmap-dsp-{expected_hash}" in str(path)
    assert path.name == "image.jpg.sift"


def test_get_sift_path_for_image_with_workspace(tmp_path):
    """get_sift_path_for_image uses workspace feature_prefix_dir."""
    image_path = tmp_path / "image.jpg"
    image_path.touch()

    config = {
        "version": 1,
        "feature_tool": "colmap",
        "feature_type": "sift-colmap",
        "feature_options": {},
        "feature_prefix_dir": "features/sift-colmap-abc123",
    }
    (tmp_path / ".sfm-workspace.json").write_text(json.dumps(config))

    path = get_sift_path_for_image(image_path)
    assert "features/sift-colmap-abc123" in str(path).replace("\\", "/")
    assert path.name == "image.jpg.sift"


def test_get_sift_path_for_image_default(tmp_path):
    """get_sift_path_for_image falls back to colmap defaults."""
    image_path = tmp_path / "image.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # No workspace, no tool specified — defaults to colmap with empty options
    path = get_sift_path_for_image(image_path)
    assert "sift-colmap-" in str(path)
    assert path.name == "image.jpg.sift"


# ---------------------------------------------------------------------------
# Hashing utility tests
# ---------------------------------------------------------------------------


def test_xxh128_of_file(tmp_path):
    """xxh128_of_file returns a 32-char hex string."""
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world")
    result = xxh128_of_file(f)
    assert isinstance(result, str)
    assert len(result) == 32
    # Same content → same hash
    assert xxh128_of_file(f) == result


def test_get_feature_tool_xxh128_deterministic():
    """get_feature_tool_xxh128 is deterministic and changes with inputs."""
    h1 = get_feature_tool_xxh128("colmap", "sift-colmap", {})
    h2 = get_feature_tool_xxh128("colmap", "sift-colmap", {})
    assert h1 == h2
    assert len(h1) == 32

    h3 = get_feature_tool_xxh128("opencv", "sift-opencv", {})
    assert h3 != h1


# ---------------------------------------------------------------------------
# Feature size / orientation utility tests
# ---------------------------------------------------------------------------


def test_feature_size_functions():
    """feature_size, feature_size_x, feature_size_y compute correct values."""
    # Identity-like affine shape: columns are [1,0] and [0,1] → sizes should be 1.0
    shapes = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(feature_size_x(shapes), [1.0])
    np.testing.assert_allclose(feature_size_y(shapes), [1.0])
    np.testing.assert_allclose(feature_size(shapes), [1.0])

    # Scaled: columns are [3,0] and [0,4] → sizes 3, 4, avg 3.5
    shapes2 = np.array([[[3.0, 0.0], [0.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(feature_size_x(shapes2), [3.0])
    np.testing.assert_allclose(feature_size_y(shapes2), [4.0])
    np.testing.assert_allclose(feature_size(shapes2), [3.5])


def test_compute_orientation():
    """compute_orientation returns correct angles."""
    # Identity matrix → atan2(0, 1) = 0
    shapes = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(compute_orientation(shapes), [0.0])

    # 90 degree rotation: [[0,-1],[1,0]] → atan2(1, 0) = pi/2
    shapes2 = np.array([[[0.0, -1.0], [1.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(compute_orientation(shapes2), [np.pi / 2])

    # Also works on a single (2,2) matrix
    single = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert compute_orientation(single) == pytest.approx(0.0)
