# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Code outside the package reads the bindings from their public modules.

The compiled extension `sfmtool._sfmtool` is internal. Each of its submodules
has a public module of the same name (`sfmtool.fileio`, `sfmtool.geometry`, ...),
and the tests, the scripts and the docs import from those. This test scans
them as text for a path into the extension and fails on any it finds outside
the allowlist below.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("tests", "scripts", "docs")
SUFFIXES = {".py", ".md", ".sh", ".ps1", ".toml", ".txt", ".json", ".yml", ".yaml"}

# `sfmtool._sfmtool`, `from sfmtool import _sfmtool` and relative imports of it.
EXTENSION_PATH = re.compile(
    r"sfmtool\._sfmtool\b|from\s+sfmtool\s+import\s+_sfmtool\b|from\s+\.+_sfmtool\b"
)

ALLOWED_PATTERNS = (
    # Each `*_registration.py` test checks what an extension submodule
    # registers and that its public module matches, so it names the submodule.
    re.compile(r"^tests/rust_bindings/[a-z_]+/test_[a-z_]+_registration\.py$"),
    # This file, which names the paths it looks for.
    re.compile(r"^tests/test_module_layout\.py$"),
)

# Single files, each with the number of hits it may have and why.
ALLOWED_COUNTS = {
    # `run_explorer` is a root-level extension function with no public home,
    # since `sfm explorer` is how users reach it, and it is what this tests.
    "tests/test_explorer_command.py": 1,
    # Replaces `read_colmap_binary` where `_incremental_sfm` looks it up at
    # call time, on the extension submodule. Patching `sfmtool.fileio` would leave
    # the solve calling the real binding.
    "tests/test_solve.py": 1,
    # Replaces `estimate_intrinsics` where `sfm estimate-intrinsics` imports
    # it at call time, on the extension submodule, for the same reason.
    "tests/test_estimate_intrinsics.py": 1,
}


def _scanned_files():
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*")):
            if (
                path.is_file()
                and path.suffix in SUFFIXES
                and "__pycache__" not in path.parts
            ):
                yield path


def test_no_path_into_the_extension_outside_the_package():
    found = {}
    for path in _scanned_files():
        relative = path.relative_to(ROOT).as_posix()
        if any(pattern.match(relative) for pattern in ALLOWED_PATTERNS):
            continue
        hits = [
            f"{relative}:{number}: {line.strip()}"
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            )
            if EXTENSION_PATH.search(line)
        ]
        if len(hits) > ALLOWED_COUNTS.get(relative, 0):
            found[relative] = hits
    assert not found, (
        "These lines reach into the internal `sfmtool._sfmtool` extension. "
        "Import the binding from its public module instead, such as "
        "`from sfmtool.fileio import read_sfmr`:\n"
        + "\n".join(hit for hits in found.values() for hit in hits)
    )


def test_allowlist_has_no_stale_entries():
    """Each counted allowlist entry still has as many hits as it allows, so
    the allowance goes away with the last use."""
    for relative, allowed in ALLOWED_COUNTS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        count = sum(1 for line in text.splitlines() if EXTENSION_PATH.search(line))
        assert count == allowed, (relative, count, allowed)
