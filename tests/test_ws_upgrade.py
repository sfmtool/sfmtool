# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm ws upgrade` end to end through Click: on a small version 1 workspace,
the line asking for a back-up, the report it prints, a dry run that writes
nothing, a refused run and a second run with nothing to do.

What the upgrade does to each file is tested in Rust
(`crates/sfmtool-core/src/workspace/upgrade/tests.rs`) and the binding in
`tests/rust_bindings/workspace/test_upgrade_workspace_rust_bindings.py`; these
tests check what the command adds.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._commands import ws as ws_command
from sfmtool.cli import main
from sfmtool.fileio import (
    read_image_rgb,
    read_matches,
    read_sift_metadata,
    write_matches,
    write_sift,
)
from sfmtool.sift.file import xxh128_of_file
from sfmtool.workspace import feature_prefix_dir, upgrade_feature_options

from .conftest import SEOUL_BULL_DIR

IMAGES = ("seoul_bull_sculpture_01.jpg", "seoul_bull_sculpture_02.jpg")
V1_OPTIONS = {"peak_ratio": 0.8, "gray_formula": "0.2126*R + 0.7152*G + 0.0722*B"}
OLD_PREFIX = feature_prefix_dir("sfmtool", "sift-sfmtool", V1_OPTIONS)
NEW_PREFIX = feature_prefix_dir(
    "sfmtool", "sift-sfmtool", upgrade_feature_options("sfmtool", V1_OPTIONS)
)
BACK_UP = "Back up the workspace before upgrading it"


def _write_version_1_sift(image: Path) -> Path:
    """A version 1 `.sift` file for `image` in the old folder beside it."""
    rgb = read_image_rgb(image)
    folder = image.parent / OLD_PREFIX
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{image.name}.sift"
    count = 4
    write_sift(
        path,
        {
            "feature_tool_metadata": {
                "feature_tool": "sfmtool",
                "feature_type": "sift-sfmtool",
                "feature_options": V1_OPTIONS,
            },
            "metadata": {
                "version": 1,
                "image_name": image.name,
                "image_file_xxh128": xxh128_of_file(image),
                "image_file_size": image.stat().st_size,
                "image_width": rgb.shape[1],
                "image_height": rgb.shape[0],
                "feature_count": count,
            },
            "positions_xy": np.arange(count * 2, dtype=np.float32).reshape(count, 2),
            "affine_shapes": np.ones((count, 2, 2), dtype=np.float32),
            "descriptors": np.arange(count * 128, dtype=np.uint8).reshape(count, 128),
            "thumbnail_y_x_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
        },
    )
    return path


def _write_matches(path: Path, sift_paths: list[Path]) -> None:
    """A pairwise `.matches` file of the two images, recording the old folder
    and the version 1 files' hashes."""
    hashes = [read_sift_metadata(p)["content_hash"] for p in sift_paths]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_matches(
        path,
        {
            "metadata": {
                "version": 10,
                "matching_method": "exhaustive",
                "matching_tool": "sfmtool",
                "matching_tool_version": "0.3",
                "matching_options": {},
                "workspace": {
                    "absolute_path": str(path.parent.parent),
                    "relative_path": "..",
                    "contents": {
                        "feature_tool": "sfmtool",
                        "feature_type": "sift-sfmtool",
                        "feature_options": V1_OPTIONS,
                        "feature_prefix_dir": OLD_PREFIX,
                    },
                },
                "timestamp": "2026-01-01T00:00:00Z",
                "image_count": 2,
                "image_pair_count": 1,
                "match_count": 1,
                "has_two_view_geometries": False,
                "has_clusters": False,
                "has_cluster_patches": False,
            },
            "image_names": [f"images/{name}" for name in IMAGES],
            "feature_tool_hashes": [
                bytes.fromhex(h["feature_tool_xxh128"]) for h in hashes
            ],
            "sift_content_hashes": [bytes.fromhex(h["content_xxh128"]) for h in hashes],
            "feature_counts": np.full(2, 4, dtype=np.uint32),
            "image_dims": np.array([[270, 480], [270, 480]], dtype=np.uint32),
            "has_clusters": False,
            "image_index_pairs": np.array([[0, 1]], dtype=np.uint32),
            "match_counts": np.array([1], dtype=np.uint32),
            "match_feature_indexes": np.zeros((1, 2), dtype=np.uint32),
            "match_descriptor_distances": np.array([10.0], dtype=np.float32),
            "has_cluster_patches": False,
            "has_two_view_geometries": False,
        },
    )


@pytest.fixture
def v1_workspace(tmp_path: Path) -> Path:
    """A workspace as sfmtool 0.3 leaves it: a version 1 workspace file, two
    seoul_bull images with version 1 `.sift` files, and a `.matches` file
    that names them."""
    root = tmp_path / "ws"
    (root / "images").mkdir(parents=True)
    (root / ".sfm-workspace.json").write_text(
        json.dumps(
            {
                "version": 1,
                "feature_tool": "sfmtool",
                "feature_type": "sift-sfmtool",
                "feature_options": V1_OPTIONS,
                "feature_prefix_dir": OLD_PREFIX,
            }
        )
    )
    sift_paths = []
    for name in IMAGES:
        shutil.copy(SEOUL_BULL_DIR / name, root / "images" / name)
        sift_paths.append(_write_version_1_sift(root / "images" / name))
    _write_matches(root / "matches" / "pair.matches", sift_paths)
    return root


def _upgrade(*args):
    return CliRunner().invoke(main, ["ws", "upgrade", *map(str, args)])


def _snapshot(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_dry_run_prints_the_plan_and_writes_nothing(v1_workspace: Path):
    before = _snapshot(v1_workspace)
    result = _upgrade(v1_workspace, "--dry-run")
    assert result.exit_code == 0, result.output
    assert _snapshot(v1_workspace) == before
    lines = result.output.splitlines()
    assert BACK_UP not in result.output
    assert lines[0].startswith("Dry run: nothing was written.")
    assert f"Workspace tree: {v1_workspace}" in lines
    assert "  .sfm-workspace.json: converted to version 2" in lines
    assert "  .matches: 1 converted" in lines
    assert (
        f"  images/{OLD_PREFIX} -> images/{NEW_PREFIX} (2 .sift files); deleted"
        in lines
    )
    assert "Converted .sift files: 2" in lines
    assert not any(line.startswith("Nothing to do") for line in lines)


def test_run_asks_for_a_back_up_first_then_reports_and_a_rerun_has_nothing_to_do(
    v1_workspace: Path,
):
    result = _upgrade(v1_workspace)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith(BACK_UP)
    assert lines[1] == f"Workspace tree: {v1_workspace}"
    assert "  .sfm-workspace.json: converted to version 2" in lines
    assert "  .matches: 1 converted" in lines
    assert (
        f"  images/{OLD_PREFIX} -> images/{NEW_PREFIX} (2 .sift files); deleted"
        in lines
    )
    # What it reports is what it did.
    assert json.loads((v1_workspace / ".sfm-workspace.json").read_text())[
        "feature_prefix_dir"
    ] == (NEW_PREFIX)
    assert not (v1_workspace / "images" / OLD_PREFIX).exists()
    assert sorted(p.name for p in (v1_workspace / "images" / NEW_PREFIX).iterdir()) == [
        f"{name}.sift" for name in IMAGES
    ]
    matches = read_matches(v1_workspace / "matches" / "pair.matches")
    assert (
        matches["metadata"]["workspace"]["contents"]["feature_prefix_dir"] == NEW_PREFIX
    )

    again = _upgrade(v1_workspace / "images")
    assert again.exit_code == 0, again.output
    lines = again.output.splitlines()
    assert "  .sfm-workspace.json: already current" in lines
    assert "  .matches: 1 already current" in lines
    assert lines[-1] == (
        "Nothing to do: no file is left to convert and no old features folder to delete."
    )


def test_a_nested_workspace_is_refused_naming_the_root(v1_workspace: Path):
    nested = v1_workspace / "nested"
    nested.mkdir()
    shutil.copy(v1_workspace / ".sfm-workspace.json", nested / ".sfm-workspace.json")
    before = _snapshot(v1_workspace)
    result = _upgrade(nested)
    assert result.exit_code != 0
    assert "This workspace is nested in the workspace" in result.output
    assert str(v1_workspace) in result.output
    assert "run the upgrade there" in result.output
    assert _snapshot(v1_workspace) == before


def test_an_undistort_workspace_is_told_to_run_sfm_undistort_again(tmp_path: Path):
    (tmp_path / ".sfm-workspace.json").write_text(
        json.dumps(
            {
                "version": 1,
                "feature_tool": "sfmtool-undistort",
                "feature_type": "sift-sfmtool-undistort",
                "feature_options": {"source_feature_tool": "sfmtool"},
                "feature_prefix_dir": "features/x",
            }
        )
    )
    result = _upgrade(tmp_path, "--dry-run")
    assert result.exit_code != 0
    assert "run 'sfm undistort' again" in result.output


def test_no_workspace_is_an_error(tmp_path: Path):
    result = _upgrade(tmp_path)
    assert result.exit_code != 0
    assert "No workspace file .sfm-workspace.json was found" in result.output


def _report(root: str, files: list[dict], **fields) -> dict:
    report = {
        "workspace_dir": root,
        "workspaces": [root],
        "dry_run": False,
        "files": files,
        "folders": [],
        "converted_sift_file_count": 0,
        "linked_directories": [],
        "old_folders_deleted": True,
        "notes": [],
        "nothing_to_do": False,
    }
    report.update(fields)
    return report


def test_the_report_groups_files_by_reason_and_names_folders_kept(tmp_path: Path):
    root = str(tmp_path)
    refused = "the current reader refuses it, so it is left as it is: bad camera"

    def file(name, kind, outcome, reason=None):
        row = {"path": str(tmp_path / name), "kind": kind, "outcome": outcome}
        if reason is not None:
            row["reason"] = reason
        return row

    report = _report(
        root,
        [
            file(".sfm-workspace.json", "workspace_file", "converted"),
            file("a.sfmr", "sfmr", "not_converted", refused),
            file("b.sfmr", "sfmr", "not_converted", refused),
            file("c.sfmr", "sfmr", "converted"),
            file("d.matches", "matches", "failed", "it could not be written: denied"),
            file("e.kdf", "kdf", "deleted"),
        ],
        folders=[
            {
                "old_folder": str(tmp_path / "images" / "features" / "old"),
                "new_folder": None,
                "sift_file_count": 3,
                "kept_because": "named by 2 file(s) the current reader refuses",
                "deleted": False,
                "kept_files": [],
            }
        ],
        linked_directories=[str(tmp_path / "link")],
        notes=[f"The note names {tmp_path / 'c.sfmr'}."],
    )
    lines = ws_command.report_lines(report)
    assert lines[lines.index("Files:") + 1 :][:3] == [
        "  .sfmr: 1 converted, 2 not converted",
        "  .matches: 1 failed",
        "  .kdf: 1 deleted",
    ]
    at = lines.index("Not converted:")
    assert lines[at + 1 : at + 4] == [
        f"  2 files: {refused}",
        "    a.sfmr",
        "    b.sfmr",
    ]
    at = lines.index("Failed:")
    assert lines[at + 1] == "  d.matches: it could not be written: denied"
    assert (
        "  images/features/old (3 .sift files, not converted); kept: named by 2 "
        "file(s) the current reader refuses"
    ) in lines
    at = lines.index("Linked directories, not entered:")
    assert lines[at + 1] == "  link"
    assert lines[-1] == "  The note names c.sfmr."


def test_a_failed_file_makes_the_command_fail(tmp_path: Path, monkeypatch):
    report = _report(
        str(tmp_path),
        [
            {
                "path": str(tmp_path / "a.sfmr"),
                "kind": "sfmr",
                "outcome": "failed",
                "reason": "it cannot be read: locked",
            }
        ],
    )
    monkeypatch.setattr(ws_command, "upgrade_workspace", lambda *a, **k: report)
    result = _upgrade(tmp_path)
    assert result.exit_code == 1
    assert "  a.sfmr: it cannot be read: locked" in result.output
    assert "1 file(s) failed" in result.output
