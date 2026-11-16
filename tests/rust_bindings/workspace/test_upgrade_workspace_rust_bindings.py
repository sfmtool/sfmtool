# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.workspace.upgrade_workspace`: the report it returns and the errors
it raises. The conversion itself is tested in Rust
(`crates/sfmtool-core/src/workspace/upgrade/tests.rs`); these tests check
what the binding adds: the report as a dict, the keyword arguments, the
progress counter and the exception types.
"""

import json

import pytest

from sfmtool import ProgressCounter
from sfmtool.workspace import (
    feature_prefix_dir,
    upgrade_feature_options,
    upgrade_workspace,
)

V1_OPTIONS = {"peak_ratio": 0.8, "gray_formula": "0.2126*R + 0.7152*G + 0.0722*B"}


def _write_v1_workspace(directory):
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "feature_tool": "sfmtool",
        "feature_type": "sift-sfmtool",
        "feature_options": V1_OPTIONS,
        "feature_prefix_dir": feature_prefix_dir("sfmtool", "sift-sfmtool", V1_OPTIONS),
        "note": "kept",
    }
    (directory / ".sfm-workspace.json").write_text(json.dumps(config))
    return directory / ".sfm-workspace.json"


def test_dry_run_reports_the_workspace_files_of_the_tree_and_writes_nothing(tmp_path):
    config = _write_v1_workspace(tmp_path / "ws")
    nested = _write_v1_workspace(tmp_path / "ws" / "nested")
    before = config.read_bytes(), nested.read_bytes()
    report = upgrade_workspace(tmp_path / "ws", dry_run=True)
    assert (config.read_bytes(), nested.read_bytes()) == before
    assert report["dry_run"] is True
    assert report["workspaces"] == [
        str(tmp_path / "ws"),
        str(tmp_path / "ws" / "nested"),
    ]
    assert report["files"] == [
        {"path": str(config), "kind": "workspace_file", "outcome": "converted"},
        {"path": str(nested), "kind": "workspace_file", "outcome": "converted"},
    ]
    assert report["linked_directories"] == []
    assert report["folders"] == []
    assert report["old_folders_deleted"] is True


def test_run_writes_the_workspace_file_as_version_2(tmp_path):
    config = _write_v1_workspace(tmp_path / "ws")
    counter = ProgressCounter()
    report = upgrade_workspace(tmp_path / "ws", progress=counter)
    assert report["dry_run"] is False
    written = json.loads(config.read_text())
    options = upgrade_feature_options("sfmtool", V1_OPTIONS)
    assert written == {
        "version": 2,
        "feature_tool": "sfmtool",
        "feature_type": "sift-sfmtool",
        "feature_options": options,
        "feature_prefix_dir": feature_prefix_dir("sfmtool", "sift-sfmtool", options),
        "note": "kept",
    }
    # The walk counted the one folder it listed.
    assert counter.value == 1
    again = upgrade_workspace(tmp_path / "ws")
    assert again["files"] == [
        {"path": str(config), "kind": "workspace_file", "outcome": "already_current"}
    ]


def test_a_nested_workspace_raises_value_error_naming_the_root(tmp_path):
    _write_v1_workspace(tmp_path / "ws")
    _write_v1_workspace(tmp_path / "ws" / "nested")
    with pytest.raises(ValueError, match="run the upgrade there"):
        upgrade_workspace(tmp_path / "ws" / "nested", dry_run=True)


def test_no_workspace_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="no workspace file"):
        upgrade_workspace(tmp_path)


def test_an_undistort_workspace_raises_value_error(tmp_path):
    directory = tmp_path / "ws"
    directory.mkdir()
    (directory / ".sfm-workspace.json").write_text(
        json.dumps(
            {
                "version": 1,
                "feature_tool": "sfmtool-undistort",
                "feature_type": "sift-sfmtool",
                "feature_options": {},
                "feature_prefix_dir": "features/x",
            }
        )
    )
    with pytest.raises(ValueError, match="sfm undistort"):
        upgrade_workspace(directory, dry_run=True)
