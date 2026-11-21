# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The `.sfmr` versions this build reads: versions 16 and 17. A file of an older
version some release wrote is refused by every reader with one message,
raised as `OSError`, naming sfmtool 0.4.1 as the last release that reads it,
and a version above the current one is refused as unsupported. See
`specs/formats/sfmr-file-format.md` § "Format versions"."""

import json
import shutil
import zipfile
from compression import zstd
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import (
    read_sfmr,
    read_sfmr_content_hash,
    read_sfmr_metadata,
    verify_sfmr,
)
from sfmtool.reconstruction import SfmrReconstruction
from tests.conftest import SEOUL_BULL_GROUND_TRUTH


def _rewrite(src: Path, dst: Path, version: int, widen: bool) -> Path:
    """Copy `src` to `dst` with `metadata.json`'s version set to `version`,
    and with `widen` each `float16` entry stored as the `float32` entry a
    version 15 writer stored (a score's deficit `w` as the score `1 - w`)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            name, data = info.filename, zin.read(info)
            if name == "metadata.json.zst":
                metadata = json.loads(zstd.decompress(data))
                metadata["version"] = version
                data = zstd.compress(json.dumps(metadata).encode())
            elif widen and ".float16." in name:
                values = np.frombuffer(zstd.decompress(data), "<f2").astype("<f4")
                if "_zncc_deficit." in name:
                    values = (1 - values).astype("<f4")
                    name = name.replace("_zncc_deficit.", "_zncc.")
                name = name.replace(".float16.", ".float32.")
                data = zstd.compress(values.tobytes())
            zout.writestr(name, data, compress_type=zipfile.ZIP_STORED)
    return dst


_READERS = [
    read_sfmr,
    read_sfmr_metadata,
    read_sfmr_content_hash,
    SfmrReconstruction.load,
]


def test_a_version_15_file_is_refused_naming_the_last_release_that_reads_it(
    tmp_path,
):
    """Every reader refuses a version 15 file with one message, raised as
    `OSError`, `verify_sfmr` reports it as the one error, and `sfm inspect`
    prints it."""
    path = _rewrite(SEOUL_BULL_GROUND_TRUTH, tmp_path / "v15.sfmr", 15, widen=True)
    with zipfile.ZipFile(path) as z:
        assert any(".float32." in n and "sharpness" in n for n in z.namelist())
    message = (
        f"{path} is a version 15 .sfmr file, and this sfmtool reads .sfmr files "
        "of version 16 or later. sfmtool 0.4.1 is the last release that reads "
        "version 15: install it, run `sfm ws upgrade` on the workspace, then use "
        "the file with this release."
    )
    for read in _READERS:
        with pytest.raises(OSError) as raised:
            read(path)
        assert str(raised.value) == message, read
    assert verify_sfmr(path) == (False, [message])
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code != 0
    assert "sfmtool 0.4.1 is the last release that reads version 15" in result.output


def test_a_version_above_the_current_one_is_refused_as_unsupported(tmp_path):
    """A version this build does not know yet is refused as unsupported,
    naming no release."""
    path = _rewrite(SEOUL_BULL_GROUND_TRUTH, tmp_path / "v18.sfmr", 18, widen=False)
    for read in _READERS:
        with pytest.raises(OSError, match="unsupported .sfmr format version 18"):
            read(path)
    ok, errors = verify_sfmr(path)
    assert not ok
    assert errors == [
        "Invalid format: unsupported .sfmr format version 18 (this build reads "
        "versions 16 to 17)"
    ]


def test_a_version_16_file_is_read(tmp_path):
    """A version 16 file (no measurements) is accepted by every reader and
    rewritten at version 17 on save."""
    shutil.copy(
        SEOUL_BULL_GROUND_TRUTH.parent / ".sfm-workspace.json",
        tmp_path / ".sfm-workspace.json",
    )
    path = _rewrite(SEOUL_BULL_GROUND_TRUTH, tmp_path / "v16.sfmr", 16, widen=False)
    assert read_sfmr_metadata(path)["version"] == 16
    assert read_sfmr(path)["metadata"]["version"] == 16
    assert len(read_sfmr_content_hash(path)) == 32
    recon = SfmrReconstruction.load(path)
    assert recon.metadata()["version"] == 16
    out = tmp_path / "v17.sfmr"
    recon.save(out)
    assert read_sfmr_metadata(out)["version"] == 17
    assert verify_sfmr(out) == (True, [])


def test_the_checked_in_file_is_read(tmp_path):
    """The same file at its own version reads, so the refusals above are the
    version's alone."""
    path = shutil.copy(SEOUL_BULL_GROUND_TRUTH, tmp_path / "v16.sfmr")
    assert read_sfmr_metadata(path)["version"] == 16
    assert verify_sfmr(path) == (True, [])
