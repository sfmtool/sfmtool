# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfm explorer`` that open no window.

The command is checked with ``subprocess.run`` replaced, so only the command
line it builds is looked at. The child program, ``python -m sfmtool._explorer``,
is run for real through the built extension, but only with arguments the
viewer answers before it creates a window: ``--help``, and an unknown option.
"""

import importlib
import subprocess
import sys
from types import SimpleNamespace

from click.testing import CliRunner

from sfmtool.cli import main

# `sfmtool._commands` re-exports the `explorer` command under the module's own
# name, so the module is reached through `importlib`.
explorer_module = importlib.import_module("sfmtool._commands.explorer")


def _invoke_with_fake_run(monkeypatch, args, returncode=0):
    """Run ``sfm explorer ARGS`` with ``subprocess.run`` recorded, not run."""
    calls = []

    def fake_run(argv, *a, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(explorer_module.subprocess, "run", fake_run)
    result = CliRunner().invoke(main, ["explorer", *args])
    return result, calls


def test_no_arguments_runs_viewer_module(monkeypatch):
    result, calls = _invoke_with_fake_run(monkeypatch, [])
    assert result.exit_code == 0, result.output
    assert calls == [[sys.executable, "-P", "-m", "sfmtool._explorer"]]


def test_options_then_files_in_order(monkeypatch, tmp_path):
    a = tmp_path / "a.sfmr"
    b = tmp_path / "b.sfmr"
    a.write_bytes(b"")
    b.write_bytes(b"")
    result, calls = _invoke_with_fake_run(
        monkeypatch, [str(a), str(b), "--no-default-layout", "--mcp", "9000"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        [
            sys.executable,
            "-P",
            "-m",
            "sfmtool._explorer",
            "--mcp",
            "9000",
            "--no-default-layout",
            str(a),
            str(b),
        ]
    ]


def test_mcp_port_out_of_range_is_refused(monkeypatch, tmp_path):
    a = tmp_path / "a.sfmr"
    a.write_bytes(b"")
    result, calls = _invoke_with_fake_run(monkeypatch, [str(a), "--mcp", "70000"])
    assert result.exit_code == 2
    assert "70000" in result.output
    assert calls == []


def test_mcp_without_port_passes_default(monkeypatch, tmp_path):
    a = tmp_path / "a.sfmr"
    a.write_bytes(b"")
    result, calls = _invoke_with_fake_run(monkeypatch, [str(a), "--mcp"])
    assert result.exit_code == 0, result.output
    assert calls[0][4:] == ["--mcp", str(explorer_module.DEFAULT_MCP_PORT), str(a)]


def test_exit_status_is_the_viewers(monkeypatch):
    result, calls = _invoke_with_fake_run(monkeypatch, [], returncode=3)
    assert len(calls) == 1
    assert result.exit_code == 3


def test_missing_file_is_refused_before_launch(monkeypatch, tmp_path):
    result, calls = _invoke_with_fake_run(monkeypatch, [str(tmp_path / "missing.sfmr")])
    assert result.exit_code != 0
    assert calls == []


def _run_viewer_module(*args):
    return subprocess.run(
        [sys.executable, "-P", "-m", "sfmtool._explorer", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_viewer_module_help_prints_usage_without_a_window():
    result = _run_viewer_module("--help")
    assert result.returncode == 0, result.stderr
    assert "USAGE:" in result.stdout
    assert "--no-default-layout" in result.stdout


def test_viewer_module_unknown_option_exits_2():
    result = _run_viewer_module("--no-such-option")
    assert result.returncode == 2
    assert "--no-such-option" in result.stderr
