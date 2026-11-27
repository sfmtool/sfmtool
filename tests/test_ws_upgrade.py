# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm ws upgrade` end to end through Click: on a small version 2 workspace
holding a current `.camrig` file and a version 1 one, which the reader
refuses, the line asking for a back-up, the report it prints, and a dry run
that writes nothing.

What the upgrade does to each file is tested in Rust
(`crates/sfmtool-core/src/workspace/upgrade/tests.rs`) and the binding in
`tests/rust_bindings/workspace/test_upgrade_workspace_rust_bindings.py`; these
tests check what the command adds. Its refusal of a version 1 workspace is in
`tests/test_version_1_workspace.py`.
"""

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool._commands import ws as ws_command
from sfmtool.cli import main

from .conftest import TEST_DATA_DIR, relabel_camrig

KERRY_PARK_RIG = TEST_DATA_DIR / "images" / "kerry_park" / "kerry_park.camrig"
BACK_UP = "Back up the workspace before upgrading it"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A version 2 workspace holding a current `.camrig` file and a version 1
    one, which the reader refuses, under `rigs/`."""
    root = tmp_path / "ws"
    result = CliRunner().invoke(main, ["ws", "init", str(root)])
    assert result.exit_code == 0, result.output
    rigs = root / "rigs"
    rigs.mkdir()
    shutil.copy(KERRY_PARK_RIG, rigs / "current.camrig")
    shutil.copy(KERRY_PARK_RIG, rigs / "old.camrig")
    relabel_camrig(rigs / "old.camrig", 1)
    return root


def _upgrade(*args):
    return CliRunner().invoke(main, ["ws", "upgrade", *map(str, args)])


def _snapshot(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_dry_run_prints_the_plan_and_writes_nothing(workspace: Path):
    before = _snapshot(workspace)
    result = _upgrade(workspace, "--dry-run")
    assert result.exit_code == 0, result.output
    assert _snapshot(workspace) == before
    lines = result.output.splitlines()
    assert BACK_UP not in result.output
    assert lines[0].startswith("Dry run: nothing was written.")
    assert f"Workspace tree: {workspace}" in lines
    assert "  .camrig: 1 already current, 1 not converted" in lines
    assert lines[-1] == "Nothing to do: no file is left to convert."


def test_run_asks_for_a_back_up_first_then_reports_each_file(workspace: Path):
    before = _snapshot(workspace)
    result = _upgrade(workspace)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == (
        "Back up the workspace before upgrading it: this rewrites .sfmr, .matches, "
        ".camrig and .kdf files in place, which cannot be undone."
    )
    assert lines[1] == f"Workspace tree: {workspace}"
    assert "  .camrig: 1 already current, 1 not converted" in lines
    # The version 1 file is listed with the reader's message, its path relative
    # to the tree's root, and left as it is.
    at = lines.index("Not converted:")
    assert lines[at + 1] == (
        "  rigs/old.camrig: the current reader refuses it, so it is left as it "
        f"is: {Path('rigs') / 'old.camrig'} is a version 1 .camrig file, and this "
        "sfmtool reads .camrig files of version 2 or later. sfmtool 0.4.1 is the "
        "last release that reads version 1: install it, run `sfm ws upgrade` on "
        "the workspace, then use the file with this release."
    )
    assert lines[-1] == "Nothing to do: no file is left to convert."
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("with_the_object", [False, True])
def test_the_upgrade_leaves_mipmap_options_as_it_finds_them(
    tmp_path: Path, with_the_object: bool
):
    """A workspace file without `mipmap_options` is left without it, since
    adding it changes which pixels are read, and one with it keeps it: the
    file is not written."""
    config = {"version": 2, "feature_tool": "sfmtool"}
    if with_the_object:
        config["mipmap_options"] = {
            "out_of_core_long_side": 2048,
            "jpeg_quality": 90,
            "full_resolution_jpeg_quality": 98,
        }
    path = tmp_path / ".sfm-workspace.json"
    path.write_text(json.dumps(config, indent=2))
    before = path.read_bytes()
    result = _upgrade(tmp_path)
    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before


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
        "linked_directories": [],
        "notes": [],
        "nothing_to_do": False,
    }
    report.update(fields)
    return report


def test_the_report_groups_files_by_reason(tmp_path: Path):
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
            file("a.sfmr", "sfmr", "not_converted", refused),
            file("b.sfmr", "sfmr", "not_converted", refused),
            file("c.sfmr", "sfmr", "converted"),
            file("d.matches", "matches", "failed", "it could not be written: denied"),
            file("e.camrig", "camrig", "already_current"),
            file("f.kdf", "kdf", "converted"),
        ],
        workspaces=[root, str(tmp_path / "nested")],
        linked_directories=[str(tmp_path / "link")],
        notes=[f"The note names {tmp_path / 'c.sfmr'}."],
    )
    lines = ws_command.report_lines(report)
    at = lines.index("Nested workspaces:")
    assert lines[at + 1] == "  nested"
    assert lines[lines.index("Files:") + 1 :][:4] == [
        "  .sfmr: 1 converted, 2 not converted",
        "  .matches: 1 failed",
        "  .camrig: 1 already current",
        "  .kdf: 1 converted",
    ]
    at = lines.index("Not converted:")
    assert lines[at + 1 : at + 4] == [
        f"  2 files: {refused}",
        "    a.sfmr",
        "    b.sfmr",
    ]
    at = lines.index("Failed:")
    assert lines[at + 1] == "  d.matches: it could not be written: denied"
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
