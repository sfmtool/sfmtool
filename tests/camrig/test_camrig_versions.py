# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The `.camrig` versions this build reads: version 2 only. A version 1 file,
which no release wrote, is refused by every reader with one message, raised as
`OSError`, naming sfmtool 0.4.1 as the last release that reads it, and a
version above the current one is refused as unsupported. See
`specs/formats/camrig-file-format.md` § "Format versions"."""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_camrig, read_camrig_metadata, verify_camrig
from sfmtool.spherical import SphericalTileRig

from ..conftest import TEST_DATA_DIR, relabel_camrig

KERRY_PARK_RIG = TEST_DATA_DIR / "images" / "kerry_park" / "kerry_park.camrig"

_READERS = [read_camrig, read_camrig_metadata, SphericalTileRig.read_camrig]


def _labelled(tmp_path: Path, version: int) -> Path:
    """A copy of the checked-in version 2 rig labelled `version`."""
    path = tmp_path / f"v{version}.camrig"
    shutil.copy(KERRY_PARK_RIG, path)
    relabel_camrig(path, version)
    return path


def test_a_version_1_file_is_refused_naming_the_last_release_that_reads_it(
    tmp_path,
):
    """Every reader refuses a version 1 file with one message, raised as
    `OSError`, `verify_camrig` reports it as the one error, and `sfm inspect`
    prints it."""
    path = _labelled(tmp_path, 1)
    message = (
        f"{path} is a version 1 .camrig file, and this sfmtool reads .camrig "
        "files of version 2 or later. sfmtool 0.4.1 is the last release that "
        "reads version 1: install it, run `sfm ws upgrade` on the workspace, "
        "then use the file with this release."
    )
    for read in _READERS:
        with pytest.raises(OSError) as raised:
            read(path)
        assert str(raised.value) == message, read
    assert verify_camrig(path) == (False, [message])
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code != 0
    assert "sfmtool 0.4.1 is the last release that reads version 1" in result.output


def test_a_version_above_the_current_one_is_refused_as_unsupported(tmp_path):
    """A version this build does not know yet is refused as unsupported,
    naming no release."""
    path = _labelled(tmp_path, 3)
    for read in _READERS:
        with pytest.raises(OSError, match="unsupported .camrig format version 3"):
            read(path)
    assert verify_camrig(path) == (
        False,
        [
            "Invalid format: unsupported .camrig format version 3 (this build "
            "reads versions 2 to 2)"
        ],
    )


def test_the_current_version_is_read(tmp_path):
    """The same file labelled with its own version reads, so the refusals above
    are the version's alone. (The relabelled `metadata.json` no longer matches
    its stored hash, which the readers do not check and `verify_camrig`
    does.)"""
    path = _labelled(tmp_path, 2)
    assert read_camrig_metadata(path)["metadata"]["version"] == 2
    assert read_camrig(path)["metadata"]["version"] == 2
