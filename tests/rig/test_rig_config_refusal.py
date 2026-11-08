# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""A `rig_config.json` at a workspace's root stops every command that sets up
the workspace's cameras and rig frames, naming the `sfm camrig cp` command
line that converts it. See specs/workspace/rig-config.md."""

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.rig.config import RigConfigError, _refuse_rig_config

from ..conftest import SEOUL_BULL_DIR

_RIG = {
    "cameras": [
        {"image_prefix": "images/", "ref_sensor": True},
        {"image_prefix": "more/", "cam_from_rig_rotation": [1, 0, 0, 0]},
    ]
}


def _sfm(*args: str):
    return CliRunner().invoke(main, [str(a) for a in args])


@pytest.fixture(scope="module")
def matched_workspace_once(tmp_path_factory) -> Path:
    """A workspace of three images in `images/`, with their `.sift` files, the
    clusters `.matches` file of `sfm match --cluster` and the pairwise one
    `sfm match --derive-pairs` derives from it, and two images in `more/` with
    no `.sift` files."""
    workspace = tmp_path_factory.mktemp("ws")
    sources = sorted(SEOUL_BULL_DIR.glob("seoul_bull_sculpture_*.jpg"))
    for folder, images in (("images", sources[:3]), ("more", sources[3:5])):
        (workspace / folder).mkdir()
        for image in images:
            shutil.copy(image, workspace / folder / image.name)
    result = _sfm("ws", "init", "--feature-tool", "sfmtool", workspace)
    assert result.exit_code == 0, result.output
    matches = workspace / "matches" / "c.matches"
    result = _sfm("match", "--cluster", workspace / "images", "-o", matches)
    assert result.exit_code == 0, result.output
    pairs = workspace / "matches" / "pairs.matches"
    result = _sfm("match", "--derive-pairs", matches, "-o", pairs)
    assert result.exit_code == 0, result.output
    return workspace


@pytest.fixture
def workspace(matched_workspace_once: Path, tmp_path: Path) -> Path:
    """A copy of `matched_workspace_once` with a `rig_config.json` at its
    root."""
    workspace = tmp_path / "ws"
    shutil.copytree(matched_workspace_once, workspace)
    (workspace / "rig_config.json").write_text(json.dumps([_RIG]))
    return workspace


def _assert_refused(result, workspace: Path) -> None:
    assert result.exit_code == 1, result.output
    rig_config = workspace / "rig_config.json"
    assert (
        f"{rig_config}: sfm takes a workspace's rigs only from .camrig files and "
        "does not read rig_config.json." in result.output
    )
    assert f"sfm camrig cp {rig_config} {workspace / 'ws.camrig'}\n" in result.output
    assert "If a .camrig already describes its rigs, delete it." in result.output


@pytest.mark.parametrize(
    "args",
    [["solve", "-i"], ["match", "--exhaustive"], ["match", "--cluster"]],
    ids=["solve", "match-exhaustive", "match-cluster"],
)
def test_commands_on_images_are_refused_before_extraction(workspace: Path, args):
    result = _sfm(*args, workspace / "more")
    _assert_refused(result, workspace)
    assert not list((workspace / "more").rglob("*.sift"))
    assert not (workspace / "sfmr").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["solve", "-i", "{pairs}"],
        ["match", "--derive-pairs", "{clusters}"],
        ["to-colmap-db", "{clusters}", "{out}"],
    ],
    ids=["solve", "match-derive-pairs", "to-colmap-db"],
)
def test_commands_on_a_matches_file_are_refused(workspace: Path, args):
    matches = workspace / "matches"
    out = workspace / "out.db"
    result = _sfm(
        *(
            a.format(
                clusters=matches / "c.matches", pairs=matches / "pairs.matches", out=out
            )
            for a in args
        )
    )
    _assert_refused(result, workspace)
    assert not out.exists()
    for folder in ("colmap", "sfmr", "tvg-matches"):
        assert not (workspace / folder).exists()


def test_a_file_of_several_rigs_gives_a_command_per_rig(tmp_path: Path):
    workspace = tmp_path / "my ws"
    workspace.mkdir()
    rig_config = workspace / "rig_config.json"
    rig_config.write_text(json.dumps([_RIG, _RIG]))
    with pytest.raises(RigConfigError) as e:
        _refuse_rig_config(workspace)
    lines = str(e.value).splitlines()
    assert lines[1:3] == [
        f'  sfm camrig cp "{rig_config}" "{workspace / f"my ws_rig{i}.camrig"}" '
        f"--rig {i}"
        for i in (0, 1)
    ]


def test_a_workspace_without_one_passes(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "rig_config.json").write_text("[]")
    assert _refuse_rig_config(tmp_path) is None
