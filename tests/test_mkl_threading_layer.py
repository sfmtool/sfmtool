# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that `import sfmtool` sets MKL_THREADING_LAYER on Windows.

Each check runs in a fresh interpreter, so the variable this process already
has, set by its own `import sfmtool`, does not count.
"""

import os
import subprocess
import sys


def _layer_after_import(value: str | None) -> str:
    """Run `import sfmtool` with MKL_THREADING_LAYER set to `value`, or unset
    when it is None, and return the value the import leaves."""
    env = {k: v for k, v in os.environ.items() if k != "MKL_THREADING_LAYER"}
    if value is not None:
        env["MKL_THREADING_LAYER"] = value
    script = (
        "import os, sfmtool\nprint(os.environ.get('MKL_THREADING_LAYER', '<unset>'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_sets_tbb_on_windows_only_when_unset():
    expected = "TBB" if sys.platform == "win32" else "<unset>"
    assert _layer_after_import(None) == expected


def test_import_keeps_a_value_already_set():
    assert _layer_after_import("SEQUENTIAL") == "SEQUENTIAL"
