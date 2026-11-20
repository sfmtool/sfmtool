# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace-related commands (`sfm ws ...`)."""

import os
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import click

from .._sfmtool import ProgressCounter
from .._workspace_init_options import (
    check_workspace_init_options,
    create_workspace,
    workspace_init_options,
)
from ..workspace import find_workspace_for_path, upgrade_workspace


@click.group("ws")
@click.help_option("--help", "-h")
def ws():
    """Workspace-related operations."""


@ws.command("init")
@click.help_option("--help", "-h")
@click.argument("workspace_dir", required=False, type=click.Path())
@workspace_init_options()
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Allow creating a workspace even if it's nested or already exists.",
)
def init(
    workspace_dir,
    feature_tool,
    domain_size_pooling,
    max_num_features,
    use_gpu,
    estimate_affine_shape,
    image_to_gray_formula,
    quantize_bits,
    force,
):
    """Initialize a workspace with configuration.

    Creates a .sfm-workspace.json file in the specified directory (or current
    directory if not specified) with default feature extraction settings.

    Example usage:

        sfm ws init

        sfm ws init my_workspace

        sfm ws init --feature-tool opencv opencv_workspace

        sfm ws init --dsp dsp_workspace

        sfm ws init --image-to-gray G --quantize-bits 8 green_workspace
    """
    check_workspace_init_options(
        feature_tool=feature_tool,
        domain_size_pooling=domain_size_pooling,
        max_num_features=max_num_features,
        use_gpu=use_gpu,
        estimate_affine_shape=estimate_affine_shape,
        image_to_gray_formula=image_to_gray_formula,
        quantize_bits=quantize_bits,
    )

    if workspace_dir is None:
        workspace_dir = Path(".")
    else:
        workspace_dir = Path(workspace_dir)

    workspace_dir = workspace_dir.resolve()
    config_path = workspace_dir / ".sfm-workspace.json"

    if not force:
        if config_path.exists():
            raise click.ClickException(
                f"A workspace already exists at: {workspace_dir}\n"
                "Use --force to overwrite the configuration."
            )

        existing_workspace = find_workspace_for_path(workspace_dir)
        if existing_workspace:
            raise click.ClickException(
                f"Cannot create nested workspace. Target directory is already inside "
                f"workspace: {existing_workspace}\n"
                "Use --force to create the nested workspace anyway."
            )

    create_workspace(
        workspace_dir,
        feature_tool=feature_tool,
        domain_size_pooling=domain_size_pooling,
        max_num_features=max_num_features,
        estimate_affine_shape=estimate_affine_shape,
        use_gpu=use_gpu,
        image_to_gray_formula=image_to_gray_formula,
        quantize_bits=quantize_bits,
    )


@ws.command("upgrade")
@click.help_option("--help", "-h")
@click.argument("path", required=False, type=click.Path(exists=True))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Find the files and print what the upgrade would do, writing nothing.",
)
def upgrade(path, dry_run):
    """Bring a workspace and every file in it to the current file versions.

    Back up the workspace first: the command rewrites .sfmr, .matches and
    .camrig files in place, so it cannot be undone.

    The workspace is found from PATH (or the current directory) by the upward
    search for .sfm-workspace.json, and the run upgrades it and every
    workspace nested in it. Each .sfmr, .matches and .camrig file older than
    the current version is rewritten at its own path as the current version.
    A file the current reader refuses is left as it is. A version 1 workspace
    is refused: upgrade it with sfmtool 0.4.1 first, the last release that
    reads the file versions it holds. A run that stops part way, by Ctrl+C,
    can be run again to finish.

    Example usage:

        sfm ws upgrade --dry-run

        sfm ws upgrade my_workspace
    """
    if not dry_run:
        click.echo(
            "Back up the workspace before upgrading it: this rewrites .sfmr, "
            ".matches and .camrig files in place, which cannot be undone."
        )
    try:
        with _echo_progress() as counter:
            report = upgrade_workspace(
                Path(path) if path else Path("."), dry_run=dry_run, progress=counter
            )
    except InterruptedError, KeyboardInterrupt:
        raise click.ClickException(
            "The upgrade was stopped. Run 'sfm ws upgrade' again to finish it "
            "from where it stopped."
        )
    except (ValueError, OSError) as e:
        message = str(e)
        raise click.ClickException(message[:1].upper() + message[1:])
    for line in report_lines(report):
        click.echo(line)
    failed = sum(f["outcome"] == "failed" for f in report["files"])
    if failed:
        raise click.ClickException(
            f"{failed} file(s) failed; fix them, or move them out of the tree, "
            "and run 'sfm ws upgrade' again."
        )


@contextmanager
def _echo_progress(interval: float = 10.0):
    """Yield a ``ProgressCounter`` for the upgrade, and while it runs echo
    every ``interval`` seconds how far its current step has got, from a
    thread of its own, since the upgrade runs with the GIL released."""
    counter = ProgressCounter()
    stop = threading.Event()

    def loop():
        last = 0
        while not stop.wait(interval):
            value = counter.value
            if value != last:
                last = value
                click.echo(f"  ... {value} done in the current step")

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield counter
    finally:
        stop.set()
        thread.join()


# How the report names each outcome, in the order it counts them, and each
# kind of file.
_OUTCOMES = {
    "converted": "converted",
    "already_current": "already current",
    "not_converted": "not converted",
    "failed": "failed",
}
_KINDS = {"sfmr": ".sfmr", "matches": ".matches", "camrig": ".camrig"}


def report_lines(report: dict) -> list[str]:
    """The lines ``sfm ws upgrade`` prints for the report
    ``sfmtool.workspace.upgrade_workspace`` returns, with each path in the
    tree relative to its root."""
    root = report["workspace_dir"]

    def rel(path: str) -> str:
        try:
            relative = Path(path).relative_to(root)
        except ValueError:
            return path
        return relative.as_posix()

    def plain(text: str) -> str:
        # A reason names files by their full paths; name those in the tree
        # relative to its root, as the rest of the report does.
        return text.replace(root + os.sep, "")

    lines = []
    if report["dry_run"]:
        lines.append(
            "Dry run: nothing was written. Each outcome below is what the "
            "upgrade would do."
        )
    lines.append(f"Workspace tree: {root}")
    nested = report["workspaces"][1:]
    if nested:
        lines.append("Nested workspaces:")
        lines.extend(f"  {rel(w)}" for w in nested)

    if report["files"]:
        lines.append("Files:")
        for kind, name in _KINDS.items():
            counts = Counter(f["outcome"] for f in report["files"] if f["kind"] == kind)
            if counts:
                counted = ", ".join(
                    f"{counts[o]} {label}"
                    for o, label in _OUTCOMES.items()
                    if counts[o]
                )
                lines.append(f"  {name}: {counted}")

    for outcome, heading in (("not_converted", "Not converted"), ("failed", "Failed")):
        by_reason: dict[str, list[str]] = {}
        for f in report["files"]:
            if f["outcome"] == outcome:
                by_reason.setdefault(plain(f["reason"]), []).append(rel(f["path"]))
        if by_reason:
            lines.append(f"{heading}:")
        for reason, paths in by_reason.items():
            if len(paths) == 1:
                lines.append(f"  {paths[0]}: {reason}")
            else:
                lines.append(f"  {len(paths)} files: {reason}")
                lines.extend(f"    {p}" for p in paths)

    if report["linked_directories"]:
        lines.append("Linked directories, not entered:")
        lines.extend(f"  {rel(d)}" for d in report["linked_directories"])
    if report["notes"]:
        lines.append("Notes:")
        lines.extend(f"  {plain(note)}" for note in report["notes"])
    if report["nothing_to_do"]:
        lines.append("Nothing to do: no file is left to convert.")
    return lines
