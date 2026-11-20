# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The `.matches` versions this build reads: version 10 only. A file of an
older version some release wrote is refused by every reader with one message,
raised as `OSError`, naming sfmtool 0.4.1 as the last release that reads it,
and a version above the current one is refused as unsupported. See
`specs/formats/matches-file-format.md` § "Format versions"."""

import json
import zipfile
from compression import zstd
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import (
    MatchesFile,
    read_matches,
    read_matches_metadata,
    verify_matches,
    write_matches,
)


def _write_current(path: Path) -> Path:
    """A valid pairwise `.matches` file of two images and one pair."""
    write_matches(
        str(path),
        {
            "metadata": {
                "version": 10,
                "matching_method": "exhaustive",
                "matching_tool": "sfmtool",
                "matching_tool_version": "0.5",
                "matching_options": {},
                "workspace": {
                    "absolute_path": "",
                    "relative_path": "",
                    "contents": {
                        "feature_tool": "sfmtool",
                        "feature_type": "sift",
                        "feature_options": {},
                        "feature_prefix_dir": "features/sift",
                    },
                },
                "timestamp": "2026-11-19T10:00:00Z",
                "image_count": 2,
                "image_pair_count": 1,
                "match_count": 1,
                "has_two_view_geometries": False,
                "has_clusters": False,
                "has_cluster_patches": False,
            },
            "image_names": ["a.jpg", "b.jpg"],
            "feature_tool_hashes": [b"\x00" * 16] * 2,
            "sift_content_hashes": [b"\x01" * 16] * 2,
            "feature_counts": np.full(2, 10, dtype=np.uint32),
            "image_dims": np.full((2, 2), [640, 480], dtype=np.uint32),
            "has_clusters": False,
            "image_index_pairs": np.array([[0, 1]], dtype=np.uint32),
            "match_counts": np.array([1], dtype=np.uint32),
            "match_feature_indexes": np.zeros((1, 2), dtype=np.uint32),
            "match_descriptor_distances": np.full(1, 100.0, dtype=np.float32),
            "has_cluster_patches": False,
            "has_two_view_geometries": False,
        },
    )
    return path


def _relabel(src: Path, dst: Path, version: int) -> Path:
    """Copy `src` to `dst` with `metadata.json`'s version set to `version`."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info)
            if info.filename == "metadata.json.zst":
                metadata = json.loads(zstd.decompress(data))
                metadata["version"] = version
                data = zstd.compress(json.dumps(metadata).encode())
            zout.writestr(info.filename, data, compress_type=zipfile.ZIP_STORED)
    return dst


_READERS = [read_matches, read_matches_metadata, MatchesFile]


def test_a_version_9_file_is_refused_naming_the_last_release_that_reads_it(
    tmp_path,
):
    """Every reader refuses a version 9 file with one message, raised as
    `OSError`, `verify_matches` reports it as the one error, and `sfm inspect`
    prints it."""
    path = _relabel(
        _write_current(tmp_path / "v10.matches"), tmp_path / "v9.matches", 9
    )
    message = (
        f"{path} is a version 9 .matches file, and this sfmtool reads .matches "
        "files of version 10 or later. sfmtool 0.4.1 is the last release that "
        "reads version 9: install it, run `sfm ws upgrade` on the workspace, "
        "then use the file with this release."
    )
    for read in _READERS:
        with pytest.raises(OSError) as raised:
            read(str(path))
        assert str(raised.value) == message, read
    assert verify_matches(str(path)) == (False, [message])
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code != 0
    assert "sfmtool 0.4.1 is the last release that reads version 9" in result.output


def test_a_version_above_the_current_one_is_refused_as_unsupported(tmp_path):
    """A version this build does not know yet is refused as unsupported,
    naming no release."""
    path = _relabel(
        _write_current(tmp_path / "v10.matches"), tmp_path / "v11.matches", 11
    )
    for read in _READERS:
        with pytest.raises(OSError, match="unsupported .matches format version 11"):
            read(str(path))
    assert verify_matches(str(path)) == (
        False,
        [
            "Invalid format: unsupported .matches format version 11 (this build "
            "reads versions 10 to 10)"
        ],
    )


def test_the_current_version_is_read(tmp_path):
    """The same file at its own version reads, so the refusals above are the
    version's alone."""
    path = _write_current(tmp_path / "v10.matches")
    assert read_matches_metadata(str(path))["version"] == 10
    assert verify_matches(str(path)) == (True, [])
