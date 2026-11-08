# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

from sfmtool._workspace import load_workspace_config
from sfmtool.workspace import find_workspace_for_path


def _symlink_dir(target: Path, link: Path) -> None:
    """A directory symlink at `link`, or a skip where this process may not make
    one (Windows without Developer Mode or an administrator)."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as e:
        pytest.skip(f"cannot create a directory symlink: {e}")


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

    def test_finds_workspace_from_path_that_does_not_exist(self, tmp_path: Path):
        """An output path not written yet is searched upward as given."""
        _write_workspace_config(tmp_path)
        assert find_workspace_for_path(tmp_path / "sfmr" / "new" / "out.sfmr") == (
            tmp_path
        )

    @pytest.mark.parametrize("relative", ["x.matches", ".", "deeper/x.sfmr"])
    def test_relative_path_searches_above_current_directory(
        self, tmp_path: Path, monkeypatch, relative: str
    ):
        """A relative path is taken against the current directory, and the
        search goes on above it."""
        _write_workspace_config(tmp_path)
        current = tmp_path / "images"
        current.mkdir()
        monkeypatch.chdir(current)
        assert find_workspace_for_path(Path(relative)) == tmp_path

    def test_parent_components_stay_as_given(self, tmp_path: Path, monkeypatch):
        """A `..` in the path is walked as written, so the workspace comes back
        spelled as given and an image path built from the same `..` path is
        measured against it as `images/a.jpg`, as `colmap/io.py` measures it."""
        _write_workspace_config(tmp_path)
        (tmp_path / "colmap").mkdir()
        (tmp_path / "images").mkdir()
        monkeypatch.chdir(tmp_path / "colmap")
        image_dir = Path("../images").absolute()

        found = find_workspace_for_path(image_dir)

        assert found == Path.cwd() / ".."
        assert (image_dir / "a.jpg").relative_to(found).as_posix() == "images/a.jpg"

    @pytest.mark.skipif(sys.platform != "win32", reason="drive-relative paths")
    def test_drive_relative_path_keeps_parent_components(
        self, tmp_path: Path, monkeypatch
    ):
        """A Windows path with a drive but no root (`C:..\\images`) is taken
        against that drive's current directory, with its `..` kept as written."""
        _write_workspace_config(tmp_path)
        (tmp_path / "colmap").mkdir()
        (tmp_path / "images").mkdir()
        monkeypatch.chdir(tmp_path / "colmap")

        found = find_workspace_for_path(Path(tmp_path.drive + "..\\images"))

        assert str(found) == str(tmp_path / "colmap") + "\\.."

    def test_linked_directory_belongs_to_workspace_around_link(self, tmp_path: Path):
        """The search walks the path as given, without resolving links: a
        workspace's image folder that links to storage outside it is in the
        workspace, and a link outside any workspace into one is not."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _write_workspace_config(workspace)
        outside = tmp_path / "photos"
        outside.mkdir()
        (workspace / "frames").mkdir()
        (tmp_path / "out").mkdir()
        _symlink_dir(outside, workspace / "images")
        _symlink_dir(workspace / "frames", tmp_path / "out" / "link")

        assert find_workspace_for_path(workspace / "images" / "a.jpg") == workspace
        assert find_workspace_for_path(tmp_path / "out" / "link" / "a.jpg") is None


class TestDeduceWorkspace:
    """`sfm`'s commands find the workspace of their input images with
    `cli.deduce_workspace`, over `find_workspace_for_path`."""

    def test_images_in_a_linked_folder_belong_to_the_workspace(self, tmp_path: Path):
        """A workspace whose image folder links to storage outside it is the
        workspace of those images, spelled as the path given, so the images'
        workspace-relative names stay `images/...`."""
        import os

        from sfmtool.cli import deduce_workspace

        workspace = tmp_path / "ws"
        workspace.mkdir()
        _write_workspace_config(workspace)
        outside = tmp_path / "photos"
        outside.mkdir()
        _symlink_dir(outside, workspace / "images")

        found = deduce_workspace({workspace / "images"})

        assert found == workspace
        assert os.path.relpath(workspace / "images" / "a.jpg", found) == os.path.join(
            "images", "a.jpg"
        )


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


def _v2(feature_options: dict, **fields) -> dict:
    """A version 2 workspace config with ``feature_options``."""
    config = {
        "version": 2,
        "feature_tool": "sfmtool",
        "feature_type": "sift-sfmtool",
        "feature_options": feature_options,
        "feature_prefix_dir": "features/sift-sfmtool-abc",
    }
    config.update(fields)
    return config


class TestLoadWorkspaceVersions:
    """load_workspace_config checks the version at load."""

    def test_version_2_with_image_to_gray_loads(self, tmp_path: Path):
        config = _v2({"image_to_gray": {"formula": "G", "quantize_bits": 8}})
        _write_workspace_config(tmp_path, config)
        # The recorded folder name comes back as it is.
        assert load_workspace_config(tmp_path) == config

    def test_version_1_with_gray_formula_loads(self, tmp_path: Path):
        config = _v2({"gray_formula": "G"}, version=1)
        _write_workspace_config(tmp_path, config)
        assert load_workspace_config(tmp_path) == config

    @pytest.mark.parametrize(
        "config, message",
        [
            (
                {k: v for k, v in _v2({}).items() if k != "version"},
                "has no 'version'",
            ),
            (_v2({"image_to_gray": {"formula": "G"}}, version=3), "version 3"),
            (_v2({"image_to_gray": {"formula": "G"}}, version=0), "version 0"),
            (_v2({"image_to_gray": {"formula": "G"}}, version="2"), "version '2'"),
            (_v2({"image_to_gray": {"formula": "G"}}, version=True), "version True"),
            (
                _v2({"gray_formula": "G", "image_to_gray": {"formula": "G"}}),
                "carry 'gray_formula', which belongs to version 1",
            ),
            (_v2({"gray_formula": "G"}), "carry 'gray_formula'"),
            (_v2({}), "carry no 'image_to_gray'"),
            (
                _v2({"image_to_gray": {"formula": "R +"}}),
                "'image_to_gray' is not valid",
            ),
            (
                _v2({"image_to_gray": {"formula": "G", "quantize_bits": 17}}),
                "'image_to_gray' is not valid",
            ),
            (_v2({"image_to_gray": "G"}), "'image_to_gray' is not valid"),
            (
                _v2({"image_to_gray": {"formula": "G"}}, version=1),
                "carry 'image_to_gray', which belongs to version 2",
            ),
            (_v2([]), "'feature_options' is not a JSON object"),
        ],
    )
    def test_invalid_files_are_refused_naming_the_file(
        self, tmp_path: Path, config: dict, message: str
    ):
        path = _write_workspace_config(tmp_path, config)
        with pytest.raises(RuntimeError, match=message) as info:
            load_workspace_config(tmp_path)
        assert str(path) in str(info.value)

    def test_nested_source_options_may_hold_a_gray_formula(self, tmp_path: Path):
        """The version 2 rules apply to the top level only: an undistort
        workspace's source options are a copy of a version 1 record."""
        config = _v2(
            {
                "source_feature_tool": "sfmtool",
                "source_feature_options": {"gray_formula": "G"},
                "image_to_gray": {"formula": "G"},
            },
            feature_tool="sfmtool-undistort",
        )
        _write_workspace_config(tmp_path, config)
        assert load_workspace_config(tmp_path) == config


# The `feature_prefix_dir` `sfm ws init` writes for the sfmtool defaults. The
# Rust `BuildSiftFilesOptions::sfmtool_defaults()` test
# (crates/sfmtool-core/src/features/sift/files/tests.rs) holds the Rust side to
# the same string, so the two default folder names stay equal.
WS_INIT_SFMTOOL_PREFIX = "features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d"


class TestInitWorkspace:
    def test_writes_version_2_with_the_tool_conversion(self, tmp_path: Path):
        from sfmtool import init_workspace

        config = init_workspace(tmp_path / "s")
        assert config["version"] == 2
        assert config["feature_prefix_dir"] == WS_INIT_SFMTOOL_PREFIX
        assert config["feature_options"]["image_to_gray"] == {
            "formula": "0.2126*R + 0.7152*G + 0.0722*B"
        }
        assert "gray_formula" not in config["feature_options"]
        assert load_workspace_config(tmp_path / "s") == config

        colmap = init_workspace(tmp_path / "c", feature_tool="colmap")
        assert colmap["feature_options"]["image_to_gray"] == {
            "formula": "0.2126*R + 0.7152*G + 0.0722*B",
            "quantize_bits": 8,
        }
        opencv = init_workspace(tmp_path / "o", feature_tool="opencv")
        assert opencv["feature_options"]["image_to_gray"] == {
            "formula": "0.299*R + 0.587*G + 0.114*B",
            "quantize_bits": 8,
        }

    def test_sets_the_sfmtool_conversion_before_naming_the_folder(self, tmp_path: Path):
        from sfmtool import init_workspace

        config = init_workspace(tmp_path / "g", image_to_gray_formula="G")
        assert config["feature_options"]["image_to_gray"] == {"formula": "G"}
        assert config["feature_prefix_dir"] != WS_INIT_SFMTOOL_PREFIX
        bits = init_workspace(tmp_path / "b", quantize_bits=8)
        assert bits["feature_options"]["image_to_gray"] == {
            "formula": "0.2126*R + 0.7152*G + 0.0722*B",
            "quantize_bits": 8,
        }
        assert bits["feature_prefix_dir"] not in (
            WS_INIT_SFMTOOL_PREFIX,
            config["feature_prefix_dir"],
        )

    def test_refuses_a_conversion_for_another_tool_or_out_of_range(
        self, tmp_path: Path
    ):
        from sfmtool import init_workspace

        with pytest.raises(ValueError, match="only be set for sfmtool"):
            init_workspace(tmp_path / "c", feature_tool="colmap", quantize_bits=8)
        with pytest.raises(ValueError, match="does not parse"):
            init_workspace(tmp_path / "s", image_to_gray_formula="R +")
        with pytest.raises(ValueError, match="from 1 to 16"):
            init_workspace(tmp_path / "s", quantize_bits=0)
        assert not (tmp_path / "c").exists()
