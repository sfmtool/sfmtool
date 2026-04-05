# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from io import UnsupportedOperation
from pathlib import Path

import click

from ._cli_group import CategoryGroup
from ._commands import init, inspect, match, sift, solve, xform
from ._workspace import find_workspace_for_path


@click.help_option("--help", "-h")
@click.group(cls=CategoryGroup)
def main():
    """SfM Tool - Fun with Structure from Motion."""
    # Disable output buffering for real-time progress feedback
    # Use UTF-8 encoding for Unicode support (e.g., histogram block characters)
    # Skip in test environments where stdout/stderr don't have fileno()
    try:
        sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, encoding="utf-8")
        sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, encoding="utf-8")
    except (AttributeError, UnsupportedOperation):
        pass


main.add_command_with_category(init, category="Workspace")
main.add_command_with_category(sift, category="Image Feature")
main.add_command_with_category(match, category="Image Feature")
main.add_command_with_category(solve, category="Reconstruction")
main.add_command_with_category(xform, category="Reconstruction")
main.add_command_with_category(inspect, category="Reconstruction")


@main.command()
def version():
    """Print the version."""
    click.echo("sfmtool 0.1")


def deduce_workspace(paths: set[Path]) -> Path:
    """Deduce the workspace directory from a set of directory paths.

    Args:
        paths: Set of directory paths within the workspace.

    Returns:
        Path to the workspace directory

    Raises:
        click.ClickException: If no workspace is found
    """
    if not paths:
        raise click.ClickException("No input paths provided to deduce workspace.")

    common_parent = Path(os.path.commonpath([str(p.absolute()) for p in paths]))

    workspace_dir = find_workspace_for_path(common_parent)
    if workspace_dir is None:
        raise click.ClickException(
            "No workspace found for paths. "
            "Please initialize a workspace using 'sfm init' in a parent directory."
        )

    return workspace_dir
