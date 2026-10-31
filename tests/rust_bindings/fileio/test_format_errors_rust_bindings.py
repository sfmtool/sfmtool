# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The exception each format error raises, for each file format.

The `.sfmr`, `.sift`, `.matches` and `.camrig` bindings map their errors
through one table (`crates/sfmtool-py/src/format_errors.rs`): an I/O failure is
an `OSError` of the subclass for its kind, a file that cannot be parsed is an
`OSError`, and data a writer was handed that breaks a rule of the format is a
`ValueError`. These tests trigger each variant through each format's reader,
verifier and writer.
"""

import json
import shutil
import zipfile
from compression import zstd

import numpy as np
import pytest

from sfmtool.fileio import (
    read_camrig,
    read_camrig_metadata,
    read_matches,
    read_matches_metadata,
    read_sfmr,
    read_sfmr_content_hash,
    read_sfmr_metadata,
    read_sift,
    read_sift_metadata,
    read_sift_partial,
    verify_camrig,
    verify_matches,
    verify_sfmr,
    verify_sift,
    write_camrig,
    write_matches,
    write_sfmr,
    write_sift,
    MatchesFile,
    SiftWriteQueue,
)

from tests.rust_bindings.fileio.test_matches_file_rust_bindings import _pairwise_dict
from tests.conftest import SEOUL_BULL_GROUND_TRUTH

_SFMR = SEOUL_BULL_GROUND_TRUTH

_CAMERA = {
    "model": "PINHOLE",
    "width": 10,
    "height": 10,
    "parameters": {
        "focal_length_x": 5.0,
        "focal_length_y": 5.0,
        "principal_point_x": 5.0,
        "principal_point_y": 5.0,
    },
}


def _sift_dict(n=10):
    rng = np.random.default_rng(0)
    return {
        "feature_tool_metadata": {
            "feature_tool": "pytest",
            "feature_type": "sift",
            "feature_options": {},
        },
        "metadata": {
            "version": 1,
            "image_name": "t.jpg",
            "image_file_xxh128": "a" * 32,
            "image_file_size": 1,
            "image_width": 100,
            "image_height": 100,
            "feature_count": n,
        },
        "positions_xy": rng.random((n, 2), dtype=np.float32),
        "affine_shapes": rng.random((n, 2, 2), dtype=np.float32),
        "descriptors": rng.integers(0, 255, (n, 128), dtype=np.uint8),
        "thumbnail_y_x_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
    }


def _write_camrig(path, cameras=(_CAMERA,)):
    write_camrig(
        str(path),
        "rig",
        "generic",
        list(cameras),
        [],
        [0],
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.zeros((1, 3)),
    )


def _make_file(fmt, tmp_path):
    """A valid file of format `fmt` in `tmp_path`."""
    path = tmp_path / f"valid.{fmt}"
    if fmt == "sfmr":
        shutil.copy(_SFMR, path)
    elif fmt == "sift":
        write_sift(str(path), _sift_dict(), 3)
    elif fmt == "matches":
        write_matches(str(path), _pairwise_dict())
    else:
        _write_camrig(path)
    return path


def _rewrite_metadata(path, out, edit):
    """Copy `path` to `out` with `metadata.json.zst` replaced by `edit(bytes)`."""
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "metadata.json.zst":
                data = zstd.compress(edit(zstd.decompress(data)))
            dst.writestr(info, data)
    return out


def _bump_json(field, value):
    def edit(raw):
        metadata = json.loads(raw)
        metadata[field] = value
        return json.dumps(metadata).encode()

    return edit


# Each format's readers and its verifier. A verifier reports a damaged file in
# its result where it can, so it is only checked for the failures it raises on.
_READERS = {
    "sfmr": [read_sfmr, read_sfmr_metadata, read_sfmr_content_hash],
    "sift": [read_sift, read_sift_metadata, lambda p: read_sift_partial(p, 1)],
    "matches": [read_matches, read_matches_metadata, MatchesFile],
    "camrig": [read_camrig, read_camrig_metadata],
}
_VERIFIERS = {
    "sfmr": verify_sfmr,
    "sift": verify_sift,
    "matches": verify_matches,
    "camrig": verify_camrig,
}
# A metadata edit the full reader refuses as `InvalidFormat` or `ShapeMismatch`.
_INVALID_METADATA = {
    "sfmr": _bump_json("version", 999),
    "sift": _bump_json("version", 999),
    "matches": _bump_json("version", 999),
    "camrig": _bump_json("camera_count", 3),
}
_FORMATS = sorted(_READERS)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_a_missing_file_is_a_file_not_found_error(fmt, tmp_path):
    missing = str(tmp_path / f"missing.{fmt}")
    for call in [*_READERS[fmt], _VERIFIERS[fmt]]:
        with pytest.raises(FileNotFoundError, match="missing"):
            call(missing)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_a_file_that_is_not_a_zip_is_an_os_error(fmt, tmp_path):
    junk = tmp_path / f"junk.{fmt}"
    junk.write_bytes(b"definitely not a zip archive")
    for call in [*_READERS[fmt], _VERIFIERS[fmt]]:
        with pytest.raises(OSError, match="ZIP error") as info:
            call(str(junk))
        assert type(info.value) is OSError


@pytest.mark.parametrize("fmt", _FORMATS)
def test_metadata_that_is_not_json_is_an_os_error(fmt, tmp_path):
    bad = _rewrite_metadata(
        _make_file(fmt, tmp_path), tmp_path / f"bad.{fmt}", lambda _: b"{not json"
    )
    for call in _READERS[fmt]:
        if call is read_sfmr_content_hash:
            continue  # reads only content_hash.json.zst
        with pytest.raises(OSError, match="JSON error") as info:
            call(str(bad))
        assert type(info.value) is OSError


@pytest.mark.parametrize("fmt", _FORMATS)
def test_a_file_that_breaks_a_format_rule_is_an_os_error(fmt, tmp_path):
    bad = _rewrite_metadata(
        _make_file(fmt, tmp_path), tmp_path / f"bad.{fmt}", _INVALID_METADATA[fmt]
    )
    full_reader = _READERS[fmt][0]
    with pytest.raises(OSError, match="Invalid format|Shape mismatch") as info:
        full_reader(str(bad))
    assert type(info.value) is OSError


def test_data_a_writer_cannot_store_is_a_value_error(tmp_path):
    # Shape mismatch: three descriptors for ten features.
    sift = _sift_dict()
    sift["descriptors"] = sift["descriptors"][:3]
    with pytest.raises(ValueError, match="Shape mismatch"):
        write_sift(str(tmp_path / "bad.sift"), sift, 3)

    # The batch writer raises the same exception from its join.
    queue = SiftWriteQueue()
    queue.submit(str(tmp_path / "queued.sift"), sift)
    with pytest.raises(ValueError, match="Shape mismatch"):
        queue.join()

    # Invalid format: a rig with no cameras.
    with pytest.raises(ValueError, match="Invalid format"):
        _write_camrig(tmp_path / "bad.camrig", cameras=())

    # Shape mismatch: a match count the match table does not hold.
    matches = _pairwise_dict()
    matches["metadata"]["match_count"] = 7
    with pytest.raises(ValueError, match="Shape mismatch"):
        write_matches(str(tmp_path / "bad.matches"), matches)

    # Shape mismatch: a points table shorter than its metadata says.
    sfmr = read_sfmr(str(_SFMR))
    sfmr["metadata"]["point_count"] += 1
    with pytest.raises(ValueError, match="Shape mismatch"):
        write_sfmr(str(tmp_path / "bad.sfmr"), sfmr, skip_recompute_depth_stats=True)


def test_a_write_the_filesystem_refuses_is_an_os_error(tmp_path):
    # The parent of each output path is a file, so creating the output, or
    # the parent directory for writers that create it, fails.
    parent = tmp_path / "a_file"
    parent.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        write_sift(str(parent / "out.sift"), _sift_dict(), 3)
    with pytest.raises(NotADirectoryError):
        _write_camrig(parent / "out.camrig")
    with pytest.raises(FileExistsError):
        write_matches(str(parent / "out.matches"), _pairwise_dict())
    with pytest.raises(FileExistsError):
        write_sfmr(
            str(parent / "out.sfmr"),
            read_sfmr(str(_SFMR)),
            skip_recompute_depth_stats=True,
        )
