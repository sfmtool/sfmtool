# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfm explorer`` that open no window.

The command is checked with ``run_explorer`` replaced in the command's module,
so only the viewer command line it builds and what it does with the status are
looked at. ``run_explorer`` itself is called in this process only with
arguments the viewer answers before it creates a window: ``--help``, an unknown
option, and ``--mcp 70000``, which it answers by returning a status without
ending the process. A child Python process checks which libraries ``sfm
explorer`` imports.
"""

import importlib
import signal
import subprocess
import sys
import threading

from click.testing import CliRunner

from sfmtool._sfmtool import run_explorer
from sfmtool.cli import main

explorer_module = importlib.import_module("sfmtool._commands.explorer")


def _invoke_with_fake_viewer(monkeypatch, args, status=0):
    """Run ``sfm explorer ARGS`` with ``run_explorer`` recorded, not run."""
    calls = []

    def fake_run_explorer(viewer_args):
        calls.append(list(viewer_args))
        return status

    monkeypatch.setattr(explorer_module, "run_explorer", fake_run_explorer)
    result = CliRunner().invoke(main, ["explorer", *args])
    return result, calls


def test_no_arguments_runs_viewer_with_no_arguments(monkeypatch):
    result, calls = _invoke_with_fake_viewer(monkeypatch, [])
    assert result.exit_code == 0, result.output
    assert calls == [[]]


def test_options_then_files_in_order(monkeypatch, tmp_path):
    a = tmp_path / "a.sfmr"
    b = tmp_path / "b.sfmr"
    a.write_bytes(b"")
    b.write_bytes(b"")
    result, calls = _invoke_with_fake_viewer(
        monkeypatch, [str(a), str(b), "--no-default-layout", "--mcp", "9000"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [["--mcp", "9000", "--no-default-layout", str(a), str(b)]]


def test_mcp_port_out_of_range_is_refused(monkeypatch, tmp_path):
    a = tmp_path / "a.sfmr"
    a.write_bytes(b"")
    result, calls = _invoke_with_fake_viewer(monkeypatch, [str(a), "--mcp", "70000"])
    assert result.exit_code == 2
    assert "70000" in result.output
    assert calls == []


def test_mcp_without_port_passes_default(monkeypatch, tmp_path):
    a = tmp_path / "a.sfmr"
    a.write_bytes(b"")
    result, calls = _invoke_with_fake_viewer(monkeypatch, [str(a), "--mcp"])
    assert result.exit_code == 0, result.output
    assert calls == [["--mcp", str(explorer_module.DEFAULT_MCP_PORT), str(a)]]


def test_exit_status_is_the_viewers(monkeypatch):
    result, calls = _invoke_with_fake_viewer(monkeypatch, [], status=3)
    assert len(calls) == 1
    assert result.exit_code == 3


def test_missing_file_is_refused_before_launch(monkeypatch, tmp_path):
    result, calls = _invoke_with_fake_viewer(
        monkeypatch, [str(tmp_path / "missing.sfmr")]
    )
    assert result.exit_code != 0
    assert calls == []


def test_default_sigint_while_viewer_runs_and_python_handler_after(monkeypatch):
    # While the viewer holds the main thread, SIGINT has its default action, and
    # Python's handler is back once the viewer returns.
    seen = []

    def fake_run_explorer(viewer_args):
        seen.append(signal.getsignal(signal.SIGINT))
        return 0

    monkeypatch.setattr(explorer_module, "run_explorer", fake_run_explorer)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        result = CliRunner().invoke(main, ["explorer"])
    finally:
        restored = signal.signal(signal.SIGINT, previous)
    assert result.exit_code == 0, result.output
    assert seen == [signal.SIG_DFL]
    assert restored is signal.default_int_handler


def test_off_the_main_thread_is_refused(monkeypatch):
    calls = []
    monkeypatch.setattr(explorer_module, "run_explorer", calls.append)
    results = []

    def invoke():
        results.append(CliRunner().invoke(main, ["explorer"]))

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()
    (result,) = results
    assert result.exit_code == 1
    assert "main thread" in result.output
    assert calls == []


def test_explorer_imports_no_numpy_pycolmap_or_opencv():
    # In a fresh interpreter, so this process's own imports do not count.
    code = (
        "import sys\n"
        "import sfmtool\n"
        "from sfmtool.cli import main\n"
        "try:\n"
        "    main(['explorer', '--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(sorted(m for m in ('pycolmap', 'cv2', 'numpy') if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "Launch the SfM Explorer 3D viewer." in result.stdout
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_run_explorer_returns_2_on_an_unknown_option_in_process(capfd):
    # The viewer reports the bad option and returns its status, rather than
    # ending this process; reaching the assertions is the check.
    assert run_explorer(["--no-such-option"]) == 2
    assert "--no-such-option" in capfd.readouterr().err


def test_run_explorer_returns_2_on_an_out_of_range_port_in_process(capfd):
    # A number after `--mcp` is the port, so one too large for a port is
    # refused rather than read as the default port and a file named 70000.
    assert run_explorer(["--mcp", "70000"]) == 2
    assert "70000" in capfd.readouterr().err


def test_run_explorer_returns_0_on_help_in_process(capfd):
    assert run_explorer(["--help"]) == 0
    assert "USAGE:" in capfd.readouterr().out
