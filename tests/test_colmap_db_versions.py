# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm to-colmap-db` on a reconstruction whose record holds version 1
settings in a version 2 workspace, as one made in a version 1 workspace and not
rewritten by sfmtool 0.4.1's `sfm ws upgrade` does: no `.sift` file is
extracted from those settings, and the refusal names sfmtool 0.4.1."""

import json
import shutil

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_sfmr, write_sfmr
from sfmtool.sift.extract import SiftExtractionError, image_files_to_sift_files

from .conftest import (
    SEOUL_BULL_DIR,
    SEOUL_BULL_GROUND_TRUTH,
    SFMTOOL_VERSION_1_WORKSPACE,
)


def test_a_version_1_record_with_files_to_extract_is_refused(tmp_path):
    """The seoul_bull ground truth (embedded patches, no .sift files) with a
    version 1 sfmtool record, in a version 2 workspace: the export refuses to
    extract from the record's settings and writes nothing."""
    for image in SEOUL_BULL_DIR.glob("seoul_bull_sculpture_*.jpg"):
        shutil.copy(image, tmp_path / image.name)
    shutil.copy(SEOUL_BULL_DIR / ".sfm-workspace.json", tmp_path)
    assert json.loads((tmp_path / ".sfm-workspace.json").read_text())["version"] == 2
    data = read_sfmr(SEOUL_BULL_GROUND_TRUTH)
    data["metadata"]["workspace"]["contents"] = {
        "feature_tool": "sfmtool",
        "feature_type": "sift",
        "feature_options": SFMTOOL_VERSION_1_WORKSPACE["feature_options"],
        "feature_prefix_dir": SFMTOOL_VERSION_1_WORKSPACE["feature_prefix_dir"],
    }
    sfmr = tmp_path / SEOUL_BULL_GROUND_TRUTH.name
    write_sfmr(sfmr, data)

    result = CliRunner().invoke(
        main,
        ["to-colmap-db", "--no-guided-matching", str(sfmr), str(tmp_path / "out.db")],
    )
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "so they are version 1 settings" in output
    assert "sfmtool 0.4.1 is the last release that converts them" in output
    assert not (tmp_path / "out.db").exists()
    assert not list(tmp_path.rglob("*.sift"))


def test_an_undistort_tool_is_refused(tmp_path):
    shutil.copy(SEOUL_BULL_DIR / "seoul_bull_sculpture_01.jpg", tmp_path)
    with pytest.raises(SiftExtractionError, match="Run 'sfm undistort' again"):
        image_files_to_sift_files(
            [tmp_path / "seoul_bull_sculpture_01.jpg"],
            feature_tool="sfmtool-undistort",
            feature_options={"image_to_gray": {"formula": "G"}},
            feature_prefix_dir="features/sift-sfmtool-undistort-x",
        )
    assert not (tmp_path / "features").exists()
