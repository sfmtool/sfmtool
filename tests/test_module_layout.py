# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Code outside the package uses public names only.

The compiled extension `sfmtool._sfmtool` is internal. Each of its submodules
has a public module of the same name (`sfmtool.fileio`, `sfmtool.geometry`, ...),
and the tests and the scripts import from those. The first test scans them as
text for a path into the extension and fails on any it finds outside
the allowlist below.

A leading `_` on any part of a name means the name is internal to `sfmtool`
(AGENTS.md § "Python names and privacy"). A test may import an internal name
when that name is what it tests, which a scan cannot tell, but a script never
does. The second test parses every script's imports and fails on one that
names an internal module or an internal name.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("tests", "scripts")
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


# Script files, each with the number of internal-name imports it may have and why.
ALLOWED_SCRIPT_IMPORTS = {
    # `_write_clusters_matches` writes a clusters `.matches` file from clusters
    # computed over the dataset's `.kdf` index. No public writer exists, and
    # `sfm match --cluster` builds its own forest; which of the two to add is
    # open in reports/2026-10-08-hygiene-audit.md.
    "scripts/track_at_pixel/dataset.py": 1,
}


def _internal_imports(path):
    """`line: dotted.name` for each import in `path` of an `sfmtool` module or
    name with a `_`-prefixed component."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "sfmtool" and any(p.startswith("_") for p in parts[1:]):
                hits.append(f"{node.lineno}: {name}")
    return hits


def _script_files():
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def test_scripts_import_no_internal_name():
    found = {}
    for path in _script_files():
        relative = path.relative_to(ROOT).as_posix()
        hits = _internal_imports(path)
        if len(hits) > ALLOWED_SCRIPT_IMPORTS.get(relative, 0):
            found[relative] = hits
    assert not found, (
        "These scripts import a name internal to `sfmtool` (a `_`-prefixed "
        "module or name). Use its public name, such as "
        "`from sfmtool.feature_match import match_image_pair`:\n"
        + "\n".join(
            f"{relative}:{hit}" for relative, hits in found.items() for hit in hits
        )
    )


def test_script_allowlist_has_no_stale_entries():
    for relative, allowed in ALLOWED_SCRIPT_IMPORTS.items():
        count = len(_internal_imports(ROOT / relative))
        assert count == allowed, (relative, count, allowed)
