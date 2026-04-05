# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from sfmtool._workspace import find_workspace_for_path, load_workspace_config


def _write_workspace_config(workspace_dir: Path, config: dict | None = None) -> Path:
    """Helper to write a .sfm-workspace.json file."""
    if config is None:
        config = {
            "version": 1,
            "feature_tool": "colmap",
            "feature_type": "sift",
            "feature_options": {"domain_size_pooling": False},
            "feature_prefix_dir": "features/sift-colmap-abc123",
        }
    config_path = workspace_dir / ".sfm-workspace.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


class TestFindWorkspaceForPath:
    """Tests for find_workspace_for_path."""

    def test_finds_workspace_in_same_dir(self, tmp_path: Path):
        _write_workspace_config(tmp_path)
        assert find_workspace_for_path(tmp_path) == tmp_path

    def test_finds_workspace_from_file(self, tmp_path: Path):
        _write_workspace_config(tmp_path)
        test_file = tmp_path / "image.jpg"
        test_file.touch()
        assert find_workspace_for_path(test_file) == tmp_path

    def test_finds_workspace_from_subdirectory(self, tmp_path: Path):
        _write_workspace_config(tmp_path)
        subdir = tmp_path / "images" / "set1"
        subdir.mkdir(parents=True)
        assert find_workspace_for_path(subdir) == tmp_path

    def test_finds_workspace_from_file_in_subdirectory(self, tmp_path: Path):
        _write_workspace_config(tmp_path)
        subdir = tmp_path / "images" / "set1"
        subdir.mkdir(parents=True)
        test_file = subdir / "photo.jpg"
        test_file.touch()
        assert find_workspace_for_path(test_file) == tmp_path

    def test_returns_none_when_no_workspace(self, tmp_path: Path):
        # No .sfm-workspace.json anywhere
        subdir = tmp_path / "deep" / "nested" / "dir"
        subdir.mkdir(parents=True)
        assert find_workspace_for_path(subdir) is None

    def test_finds_nearest_workspace(self, tmp_path: Path):
        """When multiple workspaces exist, finds the nearest one."""
        _write_workspace_config(tmp_path)
        inner_ws = tmp_path / "inner"
        inner_ws.mkdir()
        _write_workspace_config(inner_ws)
        subdir = inner_ws / "images"
        subdir.mkdir()
        assert find_workspace_for_path(subdir) == inner_ws


class TestLoadWorkspaceConfig:
    """Tests for load_workspace_config."""

    def test_loads_valid_config(self, tmp_path: Path):
        config = {
            "version": 1,
            "feature_tool": "colmap",
            "feature_type": "sift",
            "feature_options": {"domain_size_pooling": False},
            "feature_prefix_dir": "features/sift-colmap-abc123",
        }
        _write_workspace_config(tmp_path, config)
        loaded = load_workspace_config(tmp_path)
        assert loaded["feature_tool"] == "colmap"
        assert loaded["feature_options"]["domain_size_pooling"] is False
        assert loaded["feature_prefix_dir"] == "features/sift-colmap-abc123"

    def test_raises_on_missing_config(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="not initialized"):
            load_workspace_config(tmp_path)

    def test_raises_on_invalid_json(self, tmp_path: Path):
        config_path = tmp_path / ".sfm-workspace.json"
        config_path.write_text("{ invalid json }")
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            load_workspace_config(tmp_path)

    def test_raises_on_missing_required_fields(self, tmp_path: Path):
        # Missing feature_options
        _write_workspace_config(tmp_path, {"version": 1, "feature_tool": "colmap"})
        with pytest.raises(RuntimeError, match="missing required fields"):
            load_workspace_config(tmp_path)

        # Missing feature_tool
        _write_workspace_config(tmp_path, {"version": 1, "feature_options": {}})
        with pytest.raises(RuntimeError, match="missing required fields"):
            load_workspace_config(tmp_path)
