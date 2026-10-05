# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that `sfmtool` and `sfm` load their Python submodules on first use.

`sfmtool/__init__.py` binds the names of its Python submodules on first
attribute access, and `sfm` imports a command's module only when the command is
looked up. The checks of what gets imported run in a fresh interpreter, so this
process's own imports do not count.
"""

import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import click
from click.testing import CliRunner

import sfmtool
import sfmtool._commands
from sfmtool.cli import COMMANDS, main

HEAVY = ("pycolmap", "cv2", "numpy")


def _imported_after(code: str) -> list[str]:
    """Run `code` in a fresh interpreter and return the modules among `HEAVY`
    and the `sfmtool._commands` modules that it imported."""
    script = (
        "import sys\n"
        f"{code}\n"
        f"print(sorted(m for m in sys.modules if m in {HEAVY!r} "
        "or m.startswith('sfmtool._commands.')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return eval(result.stdout.strip().splitlines()[-1])


def test_import_sfmtool_imports_no_heavy_library():
    assert _imported_after("import sfmtool") == []


def test_sfm_help_imports_no_command_module():
    code = (
        "from sfmtool.cli import main\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit:\n"
        "    pass"
    )
    assert _imported_after(code) == []


def test_running_a_command_imports_only_its_module():
    code = (
        "from sfmtool.cli import main\n"
        "try:\n"
        "    main(['ws', '--help'])\n"
        "except SystemExit:\n"
        "    pass"
    )
    assert _imported_after(code) == ["sfmtool._commands.ws"]


def test_reading_a_name_imports_its_submodule():
    assert "numpy" in _imported_after("from sfmtool import SiftReader")


def test_star_import_and_sample_public_names():
    namespace = {}
    exec("from sfmtool import *", namespace)
    for name in (
        "SfmrReconstruction",
        "ProgressCounter",
        "THUMBNAIL_SIZE",
        "build_profile",
        "image_dimensions",
        "expand_paths",
        "find_workspace_for_path",
        "SiftReader",
        "write_sift",
        "extract_sift_with_colmap",
        "extract_sift_with_opencv",
        "resample_atlas_to_equirect",
    ):
        assert name in namespace, name
        assert namespace[name] is getattr(sfmtool, name)
    assert set(sfmtool.__all__) <= set(namespace)


def test_lazy_names_resolve_to_their_submodules():
    from sfmtool.sift.file import SiftReader, write_sift
    from sfmtool._workspace import init_workspace

    assert sfmtool.SiftReader is SiftReader
    assert sfmtool.init_workspace is init_workspace
    # The Python `write_sift`, not the `_sfmtool.io` binding it calls.
    assert sfmtool.write_sift is write_sift
    assert sfmtool.sift.__name__ == "sfmtool.sift"
    assert sfmtool.rig.__name__ == "sfmtool.rig"


def test_dir_lists_every_public_name():
    listed = set(dir(sfmtool))
    assert set(sfmtool.__all__) <= listed
    assert {"SiftReader", "extract_sift_with_opencv", "sift", "rig"} <= listed


def test_unknown_name_raises_attribute_error():
    assert not hasattr(sfmtool, "no_such_name")


def test_command_table_matches_the_commands():
    """Each lazily added command is defined where the table says, under the
    table's name, and the table's one-line help is the command's own."""
    ctx = click.Context(main)
    for name, _category, _module, _attribute, short_help in COMMANDS:
        command = main.get_command(ctx, name)
        assert command is not None, name
        assert command.name == name
        for limit in (30, 45, 80, 1000):
            assert click.utils.make_default_short_help(
                short_help, limit
            ) == command.get_short_help_str(limit), (name, limit)
    assert set(main.list_commands(ctx)) == {c[0] for c in COMMANDS} | {"version"}


def test_type_checking_imports_match_the_lazy_names():
    """The `if TYPE_CHECKING:` imports in `sfmtool/__init__.py`, which type
    checkers read, name the same names from the same modules as `_LAZY_NAMES`
    and `_LAZY_SUBPACKAGES`, which `__getattr__` reads."""
    tree = ast.parse(Path(sfmtool.__file__).read_text(encoding="utf-8"))
    (block,) = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ]
    imported = set()
    for node in block.body:
        assert isinstance(node, ast.ImportFrom), ast.dump(node)
        for alias in node.names:
            assert alias.asname is None, alias.name
            imported.add((node.module, alias.name))
    expected = {
        (module, name)
        for module, names in sfmtool._LAZY_NAMES.items()
        for name in names
    } | {("sfmtool", name) for name in sfmtool._LAZY_SUBPACKAGES}
    assert imported == expected


def test_every_command_module_command_has_a_row():
    """Every Click command defined at the top level of a `sfmtool._commands`
    module is a row of `COMMANDS` or a sub-command of one, so a new command
    module without a row fails here rather than going missing from `sfm`."""
    defined = {}
    for info in pkgutil.iter_modules(sfmtool._commands.__path__):
        module = importlib.import_module(f"sfmtool._commands.{info.name}")
        for attribute, value in vars(module).items():
            if (
                isinstance(value, click.Command)
                and getattr(value.callback, "__module__", None) == module.__name__
            ):
                defined[(module.__name__, attribute)] = value
    reachable = set()
    pending = [
        getattr(importlib.import_module(f"sfmtool._commands.{c[2]}"), c[3])
        for c in COMMANDS
    ]
    while pending:
        command = pending.pop()
        reachable.add(command)
        if isinstance(command, click.Group):
            pending.extend(command.commands.values())
    missing = sorted(key for key, cmd in defined.items() if cmd not in reachable)
    assert missing == []


def test_rows_category_is_the_help_section():
    """Each command is listed in `sfm --help` under its row's category."""
    output = CliRunner().invoke(main, ["--help"], terminal_width=200).output
    section = None
    listed = {}
    for line in output.splitlines():
        if line.endswith(" Commands:"):
            section = line.removesuffix(" Commands:")
        elif section and line.startswith("  ") and line.strip():
            listed[line.split()[0]] = section
    assert listed == {c[0]: c[1] for c in COMMANDS} | {"version": "Other"}


def test_unknown_command_suggests_from_every_command():
    """The suggestion for a misspelled command comes from every command name,
    not only the ones looked up so far."""
    code = "from sfmtool.cli import main\nmain(['solv'], prog_name='sfm')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 2
    assert "Error: No such command 'solv'. Did you mean 'solve'?" in result.stderr
