# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""A command that uses a workspace requires version 2.

A version 1 workspace is refused before any work, with one message naming
sfmtool 0.4.1 as the last release that upgrades it, whether or not its `.sift`
files are there. Commands that only read a given file keep working, and
`sfm ws upgrade` refuses it too.
"""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main

from .conftest import (
    SEOUL_BULL_DIR,
    SEOUL_BULL_GROUND_TRUTH,
    build_reconstruction_at_poses,
    write_version_1_workspace_file,
)

_IMAGES = [f"seoul_bull_sculpture_{i:02d}.jpg" for i in range(1, 4)]


def _invoke(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


@pytest.fixture(scope="module")
def version_2_workspace(tmp_path_factory) -> Path:
    """Three seoul_bull images in a version 2 workspace, with their `.sift`
    files, a pairwise `.matches`, a clusters `.matches`, a `sift_files`
    reconstruction at the ground truth's poses and its `embedded_patches`
    conversion. Tests copy it."""
    workspace = tmp_path_factory.mktemp("v1ws") / "ws"
    images = workspace / "images"
    images.mkdir(parents=True)
    for name in _IMAGES:
        shutil.copy(SEOUL_BULL_DIR / name, images / name)
    shutil.copy(SEOUL_BULL_DIR / "camera_config.json", workspace)
    recon = build_reconstruction_at_poses(
        workspace,
        sorted(images.glob("*.jpg")),
        SEOUL_BULL_GROUND_TRUTH,
        workspace / "recon.sfmr",
        min_views=2,
    )
    assert recon == workspace / "recon.sfmr"
    # A second name for it, for the commands that take two reconstructions.
    shutil.copy(recon, workspace / "other.sfmr")
    result = _invoke(
        "match", "--exhaustive", "--output", workspace / "pairs.matches", images
    )
    assert result.exit_code == 0, result.output
    result = _invoke(
        "match", "--cluster", "--output", workspace / "clusters.matches", images
    )
    assert result.exit_code == 0, result.output
    result = _invoke(
        "xform",
        workspace / "recon.sfmr",
        workspace / "embedded.sfmr",
        "--to-embedded-patches",
    )
    assert result.exit_code == 0, result.output
    return workspace


@pytest.fixture
def version_1_workspace(version_2_workspace, tmp_path) -> Path:
    """A copy of the workspace whose `.sfm-workspace.json` is the version 1
    `sfmtool` file, with every `.sift` file, `.matches` and `.sfmr` kept."""
    workspace = tmp_path / "ws"
    shutil.copytree(version_2_workspace, workspace)
    write_version_1_workspace_file(workspace)
    return workspace


def _version_1_message(workspace: Path) -> str:
    return (
        f"The workspace file {workspace / '.sfm-workspace.json'} is version 1, and "
        "sfmtool works only in a version 2 workspace. sfmtool 0.4.1 is the last "
        "release that upgrades a version 1 workspace: install it, run "
        "`sfm ws upgrade` on the workspace, then use the workspace with this "
        "release."
    )


@pytest.mark.parametrize(
    "command",
    [
        ["sift", "--extract", "images"],
        ["sift", "--draw", "drawn", "images"],
        ["match", "--exhaustive", "images"],
        ["match", "--sequential", "images"],
        ["match", "--flow", "images"],
        ["match", "--cluster", "images"],
        ["match", "--derive-pairs", "clusters.matches"],
        ["solve", "-i", "images"],
        ["solve", "-g", "images"],
        ["solve", "-i", "pairs.matches"],
        ["to-colmap-db", "pairs.matches", "out.db"],
        ["to-colmap-db", "recon.sfmr", "out.db"],
        ["pick-images", "images"],
        ["epipolar", "recon.sfmr", "1", "2", "--draw", "epipolar.png"],
        ["epipolar", "recon.sfmr", "--pairs-dir", "pairs"],
        ["align", "recon.sfmr", "other.sfmr", "-o", "aligned"],
        ["align", "recon.sfmr", "other.sfmr", "-o", "aligned", "--method", "cameras"],
        ["densify", "recon.sfmr", "densified.sfmr"],
        ["merge", "recon.sfmr", "recon.sfmr", "-o", "merged.sfmr"],
        ["xform", "recon.sfmr", "filtered.sfmr", "--remove-large-features", "50"],
        ["xform", "recon.sfmr", "embedded.sfmr", "--to-embedded-patches"],
        ["xform", "recon.sfmr", "out.sfmr", "--find-points-at-infinity", "0.5"],
        ["xform", "recon.sfmr", "out.sfmr", "--camera-model", "PINHOLE"],
        ["xform", "recon.sfmr", "out.sfmr", "--add-thumbnails"],
        [
            "xform",
            "recon.sfmr",
            "out.sfmr",
            "--rotate",
            "0,0,1,10deg",
            "--bundle-adjust",
        ],
        ["xform", "recon.sfmr", "out.sfmr", "--rotate", "0,0,1,10deg"],
        # The input is embedded_patches, so alignment matches points by
        # keypoint position, which reads the sift_files reference's .sift files.
        ["xform", "embedded.sfmr", "out.sfmr", "--align-to", "recon.sfmr"],
        ["inspect", "--strips", "recon.sfmr", "0", "-o", "strips.png"],
        ["compare", "recon.sfmr", "recon.sfmr", "--strips", "strips.png"],
        ["compare", "recon.sfmr", "recon.sfmr", "--by-coordinate"],
        # An embedded_patches input has no .sift files, so compare chooses
        # coordinate matching, which reads the other input's.
        ["compare", "embedded.sfmr", "recon.sfmr"],
        ["embed-patches", "recon.sfmr", "embedded.sfmr"],
        ["heatmap", "recon.sfmr", "-o", "heatmaps"],
        ["undistort", "recon.sfmr", "-o", "undistorted"],
    ],
    ids=lambda command: " ".join(command[:2] + command[3:]),
)
def test_a_command_that_uses_the_workspace_refuses_version_1_before_any_work(
    version_1_workspace, monkeypatch, command
):
    """Every `.sift` file is there, so the refusal is of the workspace, not of
    a missing file; it comes before the first step, and nothing is written."""
    workspace = version_1_workspace
    monkeypatch.chdir(workspace)
    before = sorted(workspace.rglob("*"))
    result = CliRunner().invoke(main, command)
    assert result.exit_code != 0
    assert _version_1_message(workspace) in result.output + str(result.exception)
    for step in (
        "[1/",
        "Rendering",
        "Undistorting",
        "Computing",
        "Building",
        "Processing",
        "Analyzing",
    ):
        assert step not in result.output
    assert sorted(workspace.rglob("*")) == before


@pytest.mark.parametrize(
    "command",
    [
        ["inspect", "recon.sfmr"],
        ["inspect", "pairs.matches"],
        ["analyze", "--images", "recon.sfmr"],
        # Both inputs share their .sift files, so points match by feature
        # index and none is read.
        ["compare", "recon.sfmr", "recon.sfmr"],
        ["inspect", "images/seoul_bull_sculpture_01.jpg"],
        [
            "inspect",
            "images/features/sift-sfmtool-fc19c88d9c98d94ebff1b97f2a345bb6/"
            "seoul_bull_sculpture_01.jpg.sift",
        ],
        ["estimate-intrinsics", "-i", "clusters.matches"],
        ["web-export", "recon.sfmr", "-o", "web"],
        ["to-colmap-bin", "recon.sfmr", "colmap"],
        ["cluster-patches", "-i", "clusters.matches", "-o", "patches.matches"],
    ],
    ids=lambda command: " ".join(command)[:40],
)
def test_a_command_that_only_reads_a_given_file_works(
    version_1_workspace, monkeypatch, command
):
    monkeypatch.chdir(version_1_workspace)
    result = CliRunner().invoke(main, command)
    assert result.exit_code == 0, result.output


def test_ws_upgrade_refuses_a_version_1_workspace_naming_sfmtool_0_4_1(
    version_1_workspace, monkeypatch
):
    """`sfm ws upgrade` upgrades from a version 2 workspace, and refuses a
    version 1 one before it reads anything else, changing nothing."""
    workspace = version_1_workspace
    monkeypatch.chdir(workspace)
    before = {p: p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    result = _invoke("ws", "upgrade")
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert (
        f"The workspace file {workspace / '.sfm-workspace.json'} is version 1, "
        "and this sfmtool upgrades a version 2 workspace. sfmtool 0.4.1 is the "
        "last release that reads the older .sift, .sfmr, .matches and .camrig "
        "versions a version 1 workspace holds"
    ) in output
    assert {p: p.read_bytes() for p in workspace.rglob("*") if p.is_file()} == before
