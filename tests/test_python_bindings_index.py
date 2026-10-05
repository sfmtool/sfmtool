# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep specs/python-bindings.md linking every binding source file."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS_SRC = ROOT / "crates" / "sfmtool-py" / "src"
INDEX = ROOT / "specs" / "python-bindings.md"
PYO3_ITEM = re.compile(r"^\s*#\[(pyclass|pyfunction|pymethods)\b", re.MULTILINE)


def test_every_binding_source_file_is_linked_from_the_index():
    index = INDEX.read_text(encoding="utf-8")
    missing = []
    for path in sorted(BINDINGS_SRC.rglob("*.rs")):
        if not PYO3_ITEM.search(path.read_text(encoding="utf-8")):
            continue
        relative = path.relative_to(BINDINGS_SRC).as_posix()
        if f"(../crates/sfmtool-py/src/{relative})" not in index:
            missing.append(relative)
    assert not missing, (
        f"{INDEX.relative_to(ROOT)} does not link these binding source files: "
        f"{missing}. Add a row naming what each exposes and the spec that "
        "describes it."
    )
