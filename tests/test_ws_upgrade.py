# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm ws upgrade`: a version 1 workspace file becomes a version 2 one with a
new features folder, every option kept."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool import load_workspace_config
from sfmtool.cli import main
from sfmtool.sift.file import get_feature_tool_xxh128

from .conftest import SEOUL_BULL_DIR


def _write(workspace: Path, config: dict) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / ".sfm-workspace.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def _upgrade(*args):
    return CliRunner().invoke(main, ["ws", "upgrade", *map(str, args)])


def test_upgrade_keeps_hand_edited_options_and_converts_a_custom_gray_formula(
    tmp_path: Path,
):
    config = json.loads((SEOUL_BULL_DIR / ".sfm-workspace.json").read_text())
    config["feature_options"]["gray_formula"] = "0.5*R + 0.5*G"
    config["feature_options"]["peak_ratio"] = 0.7
    config["feature_options"]["max_num_features"] = 1234
    path = _write(tmp_path / "ws", config)
    old_folder = tmp_path / "ws" / "images" / config["feature_prefix_dir"]
    old_folder.mkdir(parents=True)
    (old_folder / "a.jpg.sift").write_bytes(b"kept")
    lists = tmp_path / "ws" / ".sfm-named-image-lists.json"
    lists.write_text("{}")

    # Run from a subdirectory: the workspace is found by the upward search.
    (tmp_path / "ws" / "images").mkdir(exist_ok=True)
    result = _upgrade(tmp_path / "ws" / "images")
    assert result.exit_code == 0, result.output

    new = json.loads(path.read_text())
    assert new["version"] == 2
    options = dict(config["feature_options"])
    del options["gray_formula"]
    options["image_to_gray"] = {"formula": "0.5*R + 0.5*G"}
    assert new["feature_options"] == options
    assert new["feature_tool"] == config["feature_tool"]
    assert new["feature_type"] == config["feature_type"]
    xxh128 = get_feature_tool_xxh128("sfmtool", "sift-sfmtool", options)
    assert new["feature_prefix_dir"] == f"features/sift-sfmtool-{xxh128}"
    assert new["feature_prefix_dir"] != config["feature_prefix_dir"]
    assert f"old features folder: {config['feature_prefix_dir']}" in result.output
    assert f"new features folder: {new['feature_prefix_dir']}" in result.output
    # The old folder, its files and the lists file stay.
    assert (old_folder / "a.jpg.sift").read_bytes() == b"kept"
    assert lists.read_text() == "{}"
    assert load_workspace_config(tmp_path / "ws") == new

    # A second run has nothing to do.
    result = _upgrade(tmp_path / "ws")
    assert result.exit_code == 0, result.output
    assert "nothing to do: already version 2" in result.output
    assert json.loads(path.read_text()) == new


@pytest.mark.parametrize(
    "tool, options, image_to_gray",
    [
        (
            "colmap",
            {"max_image_size": 2000, "use_gpu": False, "max_num_features": None},
            {"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8},
        ),
        (
            "opencv",
            {"nfeatures": 500, "sigma": 1.6},
            {"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8},
        ),
    ],
)
def test_upgrade_gives_colmap_and_opencv_their_tool_conversion(
    tmp_path: Path, tool, options, image_to_gray
):
    config = {
        "version": 1,
        "feature_tool": tool,
        "feature_type": f"sift-{tool}",
        "feature_options": options,
        "feature_prefix_dir": f"features/sift-{tool}-old",
    }
    path = _write(tmp_path, config)
    result = _upgrade(tmp_path)
    assert result.exit_code == 0, result.output
    new = json.loads(path.read_text())
    assert new["version"] == 2
    assert new["feature_options"] == {**options, "image_to_gray": image_to_gray}
    assert new["feature_prefix_dir"].startswith(f"features/sift-{tool}-")
    assert new["feature_prefix_dir"] != config["feature_prefix_dir"]


def test_upgrade_does_nothing_to_a_version_2_file(tmp_path: Path):
    assert CliRunner().invoke(main, ["ws", "init", str(tmp_path)]).exit_code == 0
    before = (tmp_path / ".sfm-workspace.json").read_bytes()
    result = _upgrade(tmp_path)
    assert result.exit_code == 0, result.output
    assert "nothing to do: already version 2" in result.output
    assert (tmp_path / ".sfm-workspace.json").read_bytes() == before


@pytest.mark.parametrize(
    "config, message",
    [
        ({"feature_tool": "sfmtool", "feature_options": {}}, "has no 'version'"),
        (
            {"version": 3, "feature_tool": "sfmtool", "feature_options": {}},
            "is version 3",
        ),
        (
            {
                "version": 1,
                "feature_tool": "sfmtool-undistort",
                "feature_type": "sift-sfmtool-undistort",
                "feature_options": {"source_feature_tool": "sfmtool"},
                "feature_prefix_dir": "features/x",
            },
            "run 'sfm undistort' again",
        ),
        (
            {
                "version": 1,
                "feature_tool": "mytool",
                "feature_type": "sift-mytool",
                "feature_options": {},
                "feature_prefix_dir": "features/x",
            },
            "'mytool' is not one of 'sfmtool', 'colmap' or 'opencv'",
        ),
        (
            {
                "version": 1,
                "feature_tool": "sfmtool",
                "feature_type": "sift-sfmtool",
                "feature_options": {"gray_formula": "R +"},
                "feature_prefix_dir": "features/x",
            },
            "'gray_formula' does not parse",
        ),
    ],
)
def test_upgrade_refusals_name_the_file_and_change_nothing(
    tmp_path: Path, config, message
):
    path = _write(tmp_path, config)
    before = path.read_bytes()
    result = _upgrade(tmp_path)
    assert result.exit_code != 0
    assert message in result.output
    assert str(path) in result.output.replace("\n", "")
    assert path.read_bytes() == before


def test_upgrade_without_a_workspace_is_an_error(tmp_path: Path):
    result = _upgrade(tmp_path)
    assert result.exit_code != 0
    assert "No workspace file .sfm-workspace.json was found" in result.output
