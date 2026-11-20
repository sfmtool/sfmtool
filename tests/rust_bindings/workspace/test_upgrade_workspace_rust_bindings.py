# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.workspace.upgrade_workspace`: the report it returns and the errors
it raises. The upgrade itself is tested in Rust
(`crates/sfmtool-core/src/workspace/upgrade/tests.rs`); these tests check
what the binding adds: the report as a dict, with `nothing_to_do`, the
keyword arguments, the progress counter and the exception types.
"""

import json
import os
import shutil
import sys

import pytest

from sfmtool import ProgressCounter
from sfmtool.fileio import read_camrig_metadata
from sfmtool.workspace import upgrade_workspace

from ...conftest import TEST_DATA_DIR, relabel_camrig

KERRY_PARK_RIG = TEST_DATA_DIR / "images" / "kerry_park" / "kerry_park.camrig"


def _write_workspace(directory, version=2):
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "version": version,
        "feature_tool": "sfmtool",
        "feature_type": "sift-sfmtool",
        "feature_options": {"image_to_gray": {"formula": "G"}},
        "feature_prefix_dir": "features/sift-sfmtool-x",
    }
    (directory / ".sfm-workspace.json").write_text(json.dumps(config))
    return directory / ".sfm-workspace.json"


def _write_old_rig(path):
    shutil.copy(KERRY_PARK_RIG, path)
    relabel_camrig(path, 1)


def test_dry_run_reports_each_file_of_the_tree_and_writes_nothing(tmp_path):
    _write_workspace(tmp_path / "ws")
    _write_workspace(tmp_path / "ws" / "nested")
    rig = tmp_path / "ws" / "nested" / "old.camrig"
    _write_old_rig(rig)
    before = rig.read_bytes()
    report = upgrade_workspace(tmp_path / "ws", dry_run=True)
    assert rig.read_bytes() == before
    assert report["dry_run"] is True
    assert report["workspace_dir"] == str(tmp_path / "ws")
    assert report["workspaces"] == [
        str(tmp_path / "ws"),
        str(tmp_path / "ws" / "nested"),
    ]
    assert report["files"] == [
        {"path": str(rig), "kind": "camrig", "outcome": "converted"},
    ]
    assert report["linked_directories"] == []
    assert report["notes"] == []
    assert report["nothing_to_do"] is False


def test_run_rewrites_an_older_file_and_counts_its_progress(tmp_path):
    _write_workspace(tmp_path / "ws")
    rig = tmp_path / "ws" / "old.camrig"
    _write_old_rig(rig)
    counter = ProgressCounter()
    report = upgrade_workspace(tmp_path / "ws", progress=counter)
    assert report["dry_run"] is False
    assert report["files"] == [
        {"path": str(rig), "kind": "camrig", "outcome": "converted"}
    ]
    assert read_camrig_metadata(rig)["metadata"]["version"] == 2
    # The rewrite counted the one file it went through.
    assert counter.value == 1
    again = upgrade_workspace(tmp_path / "ws")
    assert again["files"] == [
        {"path": str(rig), "kind": "camrig", "outcome": "already_current"}
    ]
    assert again["nothing_to_do"] is True


def test_a_version_1_workspace_raises_value_error_naming_sfmtool_0_4_1(tmp_path):
    config = _write_workspace(tmp_path / "ws", version=1)
    with pytest.raises(ValueError) as raised:
        upgrade_workspace(tmp_path / "ws", dry_run=True)
    assert str(raised.value).startswith(f"the workspace file {config} is version 1")
    assert "sfmtool 0.4.1 is the last release" in str(raised.value)


def test_no_workspace_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="no workspace file"):
        upgrade_workspace(tmp_path)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs Unix permissions that stop this process",
)
def test_a_folder_that_cannot_be_listed_raises_os_error(tmp_path):
    _write_workspace(tmp_path / "ws")
    hidden = tmp_path / "ws" / "hidden"
    hidden.mkdir()
    hidden.chmod(0)
    try:
        with pytest.raises(OSError, match="failed to list .*hidden"):
            upgrade_workspace(tmp_path / "ws", dry_run=True)
    finally:
        hidden.chmod(0o755)
