# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json

import cv2
import numpy as np
import pytest

from sfmtool._sift_file import (
    SiftExtractionError,
    SiftReader,
    compute_orientation,
    draw_sift_features,
    feature_size,
    feature_size_x,
    feature_size_y,
    get_feature_tool_xxh128,
    get_feature_type_for_tool,
    get_sift_path_for_image,
    image_files_to_sift_files,
    image_files_to_sift_files_opencv,
    write_sift,
    xxh128_of_file,
)
from sfmtool._extract_sift_colmap import get_colmap_feature_options
from sfmtool._extract_sift_opencv import (
    get_default_opencv_feature_options,
    opencv_keypoint_to_affine_shape,
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
    path = get_sift_path_for_image(
        image_path, feature_tool="colmap", feature_options=opts
    )
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


# ---------------------------------------------------------------------------
# SiftExtractionError tests
# ---------------------------------------------------------------------------


def test_sift_extraction_error():
    """SiftExtractionError is a proper exception."""
    with pytest.raises(SiftExtractionError, match="test error"):
        raise SiftExtractionError("test error")

    assert issubclass(SiftExtractionError, Exception)


# ---------------------------------------------------------------------------
# OpenCV extraction utility tests
# ---------------------------------------------------------------------------


def test_opencv_keypoint_to_affine_shape():
    """Conversion of OpenCV KeyPoint to affine shape matrix."""
    kp = cv2.KeyPoint(x=100.0, y=200.0, size=10.0, angle=45.0)

    affine_shape = opencv_keypoint_to_affine_shape(kp)

    assert affine_shape.shape == (2, 2)
    assert affine_shape.dtype == np.float32

    # At 45 degrees, cos(45) = sin(45) = sqrt(2)/2
    # OpenCV size is diameter, so radius = size / 2 = 5.0
    scale = 10.0 / 2.0
    expected_cos = np.cos(np.radians(45.0))
    expected_sin = np.sin(np.radians(45.0))

    expected = np.array(
        [
            [scale * expected_cos, -scale * expected_sin],
            [scale * expected_sin, scale * expected_cos],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_almost_equal(affine_shape, expected, decimal=5)


def test_opencv_keypoint_to_affine_shape_zero_angle():
    """Affine shape at 0 degrees is a scaled identity."""
    kp = cv2.KeyPoint(x=0.0, y=0.0, size=6.0, angle=0.0)
    affine_shape = opencv_keypoint_to_affine_shape(kp)

    expected = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(affine_shape, expected, decimal=5)


# ---------------------------------------------------------------------------
# COLMAP feature options tests
# ---------------------------------------------------------------------------


def test_colmap_feature_options_default():
    """Default COLMAP options have expected keys."""
    options = get_colmap_feature_options()
    assert options["domain_size_pooling"] is False
    assert options["max_num_features"] is None
    assert options["max_image_size"] == 4096
    assert "peak_threshold" in options
    assert "edge_threshold" in options


def test_colmap_feature_options_dsp():
    """DSP option can be enabled."""
    options = get_colmap_feature_options(domain_size_pooling=True)
    assert options["domain_size_pooling"] is True
    assert "dsp_min_scale" in options
    assert "dsp_max_scale" in options


def test_colmap_feature_options_max_features():
    """Max features can be set."""
    options = get_colmap_feature_options(max_num_features=500)
    assert options["max_num_features"] == 500


def test_opencv_feature_options_default():
    """Default OpenCV options have expected keys."""
    options = get_default_opencv_feature_options()
    assert options["nfeatures"] == 0
    assert options["nOctaveLayers"] == 3
    assert options["contrastThreshold"] == 0.04
    assert options["edgeThreshold"] == 10
    assert options["sigma"] == 1.6


# ---------------------------------------------------------------------------
# get_sift_path_for_image with tool-specific defaults
# ---------------------------------------------------------------------------


def test_get_sift_path_for_image_colmap_defaults(tmp_path):
    """get_sift_path_for_image with colmap tool uses colmap default options for hash."""
    image_path = tmp_path / "image.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    colmap_options = get_colmap_feature_options()
    path_explicit = get_sift_path_for_image(
        image_path, feature_tool="colmap", feature_options=colmap_options
    )
    # With only feature_tool, should use same defaults
    path_default = get_sift_path_for_image(image_path, feature_tool="colmap")
    assert path_explicit == path_default


def test_get_sift_path_for_image_opencv_defaults(tmp_path):
    """get_sift_path_for_image with opencv tool uses opencv default options for hash."""
    image_path = tmp_path / "image.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    opencv_options = get_default_opencv_feature_options()
    path_explicit = get_sift_path_for_image(
        image_path, feature_tool="opencv", feature_options=opencv_options
    )
    path_default = get_sift_path_for_image(image_path, feature_tool="opencv")
    assert path_explicit == path_default


# ---------------------------------------------------------------------------
# OpenCV extraction end-to-end test (synthetic image)
# ---------------------------------------------------------------------------


def _create_test_image(path, size=512):
    """Create a synthetic image with features for SIFT detection."""
    image = np.ones((size, size, 3), dtype=np.uint8) * 255
    # Checkerboard for good SIFT features
    square_size = 40
    for i in range(4, 8):
        for j in range(4, 8):
            if (i + j) % 2 == 0:
                y_start = i * square_size
                x_start = j * square_size
                image[
                    y_start : y_start + square_size, x_start : x_start + square_size
                ] = 0
    cv2.circle(image, (128, 128), 50, (100, 100, 100), -1)
    cv2.circle(image, (384, 128), 40, (80, 80, 80), -1)
    cv2.imwrite(str(path), image)
    return image


def test_opencv_sift_extraction(tmp_path):
    """OpenCV SIFT extraction produces valid feature data."""
    from sfmtool._extract_sift_opencv import extract_sift_with_opencv

    image_path = tmp_path / "test_image.jpg"
    _create_test_image(image_path)

    feature_options = get_default_opencv_feature_options()
    results = extract_sift_with_opencv([image_path], feature_options, num_threads=1)

    assert len(results) == 1
    (
        feature_tool_metadata,
        metadata,
        positions,
        affine_shapes,
        descriptors,
        thumbnail,
    ) = results[0]

    assert feature_tool_metadata["feature_tool"] == "opencv"
    assert metadata["feature_count"] == len(positions)
    assert metadata["image_width"] == 512
    assert metadata["image_height"] == 512
    assert metadata["feature_count"] > 0

    assert positions.shape == (metadata["feature_count"], 2)
    assert affine_shapes.shape == (metadata["feature_count"], 2, 2)
    assert descriptors.shape == (metadata["feature_count"], 128)
    assert descriptors.dtype == np.uint8
    assert thumbnail.shape == (128, 128, 3)
    assert thumbnail.dtype == np.uint8


def test_opencv_features_sorted_by_size(tmp_path):
    """OpenCV features are sorted by size (largest first)."""
    from sfmtool._extract_sift_opencv import extract_sift_with_opencv

    image_path = tmp_path / "test_image.jpg"
    _create_test_image(image_path)

    feature_options = get_default_opencv_feature_options()
    results = extract_sift_with_opencv([image_path], feature_options, num_threads=1)
    _, _, _, affine_shapes, _, _ = results[0]

    if len(affine_shapes) > 1:
        sizes = feature_size(affine_shapes)
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1], "Features should be sorted descending"


def test_opencv_extraction_roundtrip(tmp_path):
    """OpenCV extraction → write_sift → SiftReader roundtrip."""
    from sfmtool._extract_sift_opencv import extract_sift_with_opencv

    image_path = tmp_path / "test_image.jpg"
    _create_test_image(image_path)

    feature_options = get_default_opencv_feature_options()
    results = extract_sift_with_opencv([image_path], feature_options, num_threads=1)
    (
        feature_tool_metadata,
        metadata,
        positions,
        affine_shapes,
        descriptors,
        thumbnail,
    ) = results[0]

    sift_filename = tmp_path / "opencv_test.sift"
    write_sift(
        sift_filename,
        feature_tool_metadata,
        metadata,
        positions,
        affine_shapes,
        descriptors,
        thumbnail,
    )

    with SiftReader(sift_filename) as reader:
        assert reader.metadata == metadata
        assert reader.feature_tool_metadata["feature_tool"] == "opencv"
        read_positions, read_affine_shapes = reader.read_positions_and_shapes()
        np.testing.assert_array_equal(positions, read_positions)
        np.testing.assert_array_equal(affine_shapes, read_affine_shapes)


# ---------------------------------------------------------------------------
# image_files_to_sift_files tests
# ---------------------------------------------------------------------------


def test_image_files_to_sift_files_opencv(tmp_path):
    """Full OpenCV extraction pipeline with file caching."""
    image_path = tmp_path / "test_image.jpg"
    _create_test_image(image_path)

    sift_files = image_files_to_sift_files_opencv([image_path], num_threads=1)

    assert len(sift_files) == 1
    assert sift_files[0].exists()
    assert sift_files[0].suffix == ".sift"
    assert "sift-opencv" in str(sift_files[0].parent)

    with SiftReader(sift_files[0]) as reader:
        assert reader.feature_tool_metadata["feature_tool"] == "opencv"
        assert reader.metadata["feature_count"] > 0

    # Second extraction should skip (cached)
    original_mtime = sift_files[0].stat().st_mtime
    sift_files_2 = image_files_to_sift_files_opencv([image_path], num_threads=1)
    assert sift_files_2[0].stat().st_mtime == original_mtime


def test_image_files_to_sift_files_custom_path(tmp_path):
    """Extraction with a custom feature_path."""
    image_path = tmp_path / "test_image.jpg"
    _create_test_image(image_path)

    feature_dir = tmp_path / "custom_features"
    sift_files = image_files_to_sift_files(
        [image_path],
        feature_path=feature_dir,
        feature_tool="opencv",
        num_threads=1,
    )

    assert len(sift_files) == 1
    assert sift_files[0].parent == feature_dir
    assert sift_files[0].exists()


# ---------------------------------------------------------------------------
# draw_sift_features tests
# ---------------------------------------------------------------------------


def test_draw_sift_features(tmp_path):
    """draw_sift_features creates an output image."""
    image_path = tmp_path / "test_image.jpg"
    _create_test_image(image_path)

    # Extract features first
    image_files_to_sift_files_opencv([image_path], num_threads=1)

    output_path = tmp_path / "viz.png"
    draw_sift_features(str(image_path), str(output_path), feature_tool="opencv")

    assert output_path.exists()
    drawn = cv2.imread(str(output_path))
    assert drawn is not None
    assert drawn.shape[:2] == (512, 512)


def test_draw_sift_features_with_filter(tmp_path):
    """draw_sift_features with feature_indices filter."""
    image_path = tmp_path / "test_image.jpg"
    _create_test_image(image_path)

    image_files_to_sift_files_opencv([image_path], num_threads=1)

    output_path = tmp_path / "viz_filtered.png"
    indices = np.arange(5, dtype=np.uint32)
    draw_sift_features(
        str(image_path),
        str(output_path),
        feature_indices=indices,
        feature_tool="opencv",
    )
    assert output_path.exists()


def test_draw_sift_features_missing_image(tmp_path):
    """draw_sift_features raises FileNotFoundError for missing image."""
    with pytest.raises(FileNotFoundError, match="Image not found"):
        draw_sift_features("/nonexistent/image.png", str(tmp_path / "out.png"))


def test_draw_sift_features_missing_sift(tmp_path):
    """draw_sift_features raises FileNotFoundError when no .sift file exists."""
    image_path = tmp_path / "no_sift.png"
    _create_test_image(image_path)

    with pytest.raises(FileNotFoundError, match="SIFT file not found"):
        draw_sift_features(str(image_path), str(tmp_path / "out.png"))
