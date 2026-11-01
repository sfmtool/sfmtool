# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `SiftFile` handle binding: metadata read on open, whole-file
and first-`count` column reads, the thumbnail, the error each failure raises,
and use as a context manager."""

import threading

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool import THUMBNAIL_SIZE
from sfmtool.fileio import SiftFile, read_sift, read_sift_metadata, write_sift


@pytest.fixture
def sift_path(sample_sift_data, tmp_path):
    feature_tool_metadata, metadata, position, affine_shape, descriptor, thumbnail = (
        sample_sift_data
    )
    path = tmp_path / "image.jpg.sift"
    write_sift(
        path,
        {
            "feature_tool_metadata": feature_tool_metadata,
            "metadata": metadata,
            "positions_xy": position,
            "affine_shapes": affine_shape,
            "descriptors": descriptor,
            "thumbnail_y_x_rgb": thumbnail,
        },
    )
    return path


def test_metadata_matches_read_sift_metadata(sift_path, sample_sift_data):
    feature_tool_metadata, metadata = sample_sift_data[:2]
    f = SiftFile(sift_path)
    stored = read_sift_metadata(sift_path)
    assert f.metadata == metadata == stored["metadata"]
    assert f.feature_tool_metadata == feature_tool_metadata
    assert f.content_hash == stored["content_hash"]
    assert len(f.content_hash["content_xxh128"]) == 32
    assert f.feature_count == metadata["feature_count"]
    assert f.path == sift_path
    assert repr(f).startswith("SiftFile(") and "image.jpg.sift" in repr(f)


def test_reads_match_read_sift(sift_path):
    f = SiftFile(sift_path)
    whole = read_sift(sift_path)
    npt.assert_array_equal(f.read_positions(), whole["positions_xy"])
    npt.assert_array_equal(f.read_affine_shapes(), whole["affine_shapes"])
    npt.assert_array_equal(f.read_descriptors(), whole["descriptors"])
    positions, shapes = f.read_positions_and_shapes()
    npt.assert_array_equal(positions, whole["positions_xy"])
    npt.assert_array_equal(shapes, whole["affine_shapes"])
    thumbnail = f.read_thumbnail()
    assert thumbnail.shape == (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)
    assert thumbnail.dtype == np.uint8
    npt.assert_array_equal(thumbnail, whole["thumbnail_y_x_rgb"])
    assert f.read_positions().dtype == np.float32
    assert f.read_affine_shapes().shape == (f.feature_count, 2, 2)
    assert f.read_descriptors().dtype == np.uint8


def test_count_reads_the_first_features(sift_path, sample_sift_data):
    _, _, position, affine_shape, descriptor, _ = sample_sift_data
    f = SiftFile(sift_path)
    npt.assert_array_equal(f.read_positions(10), position[:10])
    npt.assert_array_equal(f.read_affine_shapes(count=10), affine_shape[:10])
    npt.assert_array_equal(f.read_descriptors(count=0), descriptor[:0])
    positions, shapes = f.read_positions_and_shapes(count=3)
    npt.assert_array_equal(positions, position[:3])
    npt.assert_array_equal(shapes, affine_shape[:3])
    # A count past the end reads every feature.
    assert f.read_positions(count=f.feature_count + 50).shape == position.shape


def test_returned_arrays_are_the_callers_own(sift_path):
    """Writing into a returned array does not change what the next read
    returns."""
    f = SiftFile(sift_path)
    first = f.read_positions()
    expected = first.copy()
    first[:] = -1.0
    npt.assert_array_equal(f.read_positions(), expected)


def test_context_manager_and_close(sift_path):
    with SiftFile(sift_path) as f:
        before = f.read_descriptors()
    # Closing drops the kept data, not the handle: a later read reads again.
    npt.assert_array_equal(f.read_descriptors(), before)
    f.close()
    f.close()
    assert f.metadata["feature_count"] == len(before)


def test_concurrent_reads(sift_path):
    """Threads sharing one handle all see the same data (the file read runs
    without the GIL, so they really do overlap)."""
    f = SiftFile(sift_path)
    expected = read_sift(sift_path)["descriptors"]
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(f.read_descriptors()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    for r in results:
        npt.assert_array_equal(r, expected)


def test_missing_file_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.sift"):
        SiftFile(tmp_path / "missing.sift")
    with pytest.raises(FileNotFoundError):
        read_sift(tmp_path / "missing.sift")


def test_unreadable_file_is_os_error(tmp_path):
    path = tmp_path / "not_a_zip.sift"
    path.write_bytes(b"this is not an archive")
    with pytest.raises(OSError) as excinfo:
        SiftFile(path)
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_file_removed_after_open(sift_path):
    """The metadata stays readable; a column read reports the missing file."""
    f = SiftFile(sift_path)
    sift_path.unlink()
    assert f.feature_count > 0
    with pytest.raises(FileNotFoundError):
        f.read_positions()


def test_is_immutable(sift_path):
    f = SiftFile(sift_path)
    with pytest.raises(AttributeError):
        f.metadata = {}
