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

    Back up the workspace first: the command rewrites .sfmr, .matches,
    .camrig and .kdf files in place, so it cannot be undone.

    The workspace is found from PATH (or the current directory) by the upward
    search for .sfm-workspace.json, and the run upgrades it and every
    workspace nested in it. Each .sfmr, .matches, .camrig and .kdf file
    older than the current version is rewritten at its own path as the
    current version. A file the current reader refuses is left as it is. A
    version 1 workspace is refused: upgrade it with sfmtool 0.4.1 first, the
    last release that reads the file versions it holds. A run that stops part way, by Ctrl+C,
    can be run again to finish.

    Example usage:

        sfm ws upgrade --dry-run

        sfm ws upgrade my_workspace
    """
    if not dry_run:
        click.echo(
            "Back up the workspace before upgrading it: this rewrites .sfmr, "
            ".matches, .camrig and .kdf files in place, which cannot be undone."
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


@ws.command("mipmaps")
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, required=True, type=click.Path())
@click.option(
    "--check",
    is_flag=True,
    help="Report the mipmaps missing or stale and the files --prune would "
    "delete, writing nothing; exit with status 1 when there are any.",
)
@click.option(
    "--prune",
    is_flag=True,
    help="Delete the orphan files of the mipmap directory under the paths, "
    "and make no mipmap.",
)
def mipmaps(paths, check, prune):
    """Make the mipmaps of the large photographs under the paths.

    In a workspace whose .sfm-workspace.json has mipmap_options, a photograph
    with a side over its out_of_core_long_side is out of core, and its reduced
    levels are read from mip level files under .sfm-mipmaps/ at the workspace
    root. The command makes the mipmap of each out-of-core photograph among
    the images under PATHS (the .png, .jpg and .jpeg files sfm sift takes)
    whose mip level files are missing or stale. A photograph in core, or in a
    workspace without the object, is read no further than its header.

    --check reports instead, writing nothing: the out-of-core photographs
    whose mipmaps are missing or stale, and the files --prune would delete.

    --prune instead deletes the orphan files of the part of each workspace's
    mipmap directory that mirrors PATHS: mip level files whose photograph no
    longer exists, is in core or has changed, those of a quality the
    workspace no longer sets, and temporary files a build left over an hour
    ago. In a workspace without mipmap_options it deletes every mip level
    file. It deletes nothing else, and enters no link.

    Example usage:

        sfm ws mipmaps images

        sfm ws mipmaps --check .

        sfm ws mipmaps --prune .
    """
    from .._filenames import SIFT_IMAGE_EXTENSIONS, expand_paths
    from ..photographs import PhotographCache, check_mipmaps, prune_mipmaps

    if check and prune:
        raise click.UsageError("Give --check or --prune, not both.")
    to_do = False
    try:
        if prune:
            with _echo_progress() as counter:
                report = prune_mipmaps(list(paths), progress=counter)
            lines, failed = prune_lines(report)
        else:
            images = expand_paths(paths, extensions=SIFT_IMAGE_EXTENSIONS)
            if not images:
                raise click.UsageError("There were no image files under the paths.")
            if check:
                with _echo_progress() as counter:
                    report = check_mipmaps(images, progress=counter)
                    orphans = prune_mipmaps(list(paths), dry_run=True)
                lines, failed = check_lines(report, orphans)
                to_do = bool(report["missing"] or report["stale"] or orphans["orphans"])
            else:
                with _echo_progress() as counter:
                    report = PhotographCache(budget_bytes=0).build_mipmaps(
                        images, progress=counter
                    )
                lines, failed = build_lines(report)
    except InterruptedError, KeyboardInterrupt:
        raise click.ClickException(
            "The run was stopped. What it made or deleted before it stopped stays."
        )
    for line in lines:
        click.echo(line)
    for line in failed:
        click.echo(f"Warning: {line}", err=True)
    if failed:
        raise click.ClickException(f"{len(failed)} path(s) failed; see the warnings.")
    if to_do:
        # A mipmap is missing or stale, or a file is an orphan: the lines say
        # which.
        raise click.exceptions.Exit(1)


def _shown(path) -> str:
    """`path` relative to the current directory when it is under it, with `/`."""
    path = Path(path)
    try:
        return path.absolute().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _images_line(report: dict) -> str:
    out_of_core = sum(len(report[k]) for k in ("made", "current", "missing", "stale"))
    total = out_of_core + len(report["in_core"]) + len(report["failed"])
    return (
        f"Images: {total}, {out_of_core} out of core and "
        f"{len(report['in_core'])} in core"
    )


def build_lines(report: dict) -> tuple[list[str], list[str]]:
    """The lines ``sfm ws mipmaps`` prints for the report
    ``PhotographCache.build_mipmaps`` returns, and its warnings."""
    report = {"missing": [], "stale": [], **report}
    made, current = len(report["made"]), len(report["current"])
    lines = [
        _images_line(report),
        f"New mipmaps: {made} / {made + current} out-of-core image(s)",
    ]
    failed = [f"no mipmap was made: {message}" for _, message in report["failed"]]
    return lines, failed


def check_lines(report: dict, orphans: dict) -> tuple[list[str], list[str]]:
    """The lines ``sfm ws mipmaps --check`` prints for the reports
    ``check_mipmaps`` and a dry run of ``prune_mipmaps`` return, and its
    warnings."""
    report = {"made": [], **report}
    out_of_core = sum(len(report[k]) for k in ("current", "missing", "stale"))
    lines = [
        _images_line(report),
        f"Current mipmaps: {len(report['current'])} / {out_of_core} "
        "out-of-core image(s)",
    ]
    if report["missing"]:
        lines.append(f"Missing mipmaps: {len(report['missing'])}")
        lines.extend(f"  {_shown(p)}" for p in report["missing"])
    if report["stale"]:
        lines.append(f"Stale mipmaps: {len(report['stale'])}")
        lines.extend(f"  {_shown(p)}: {why}" for p, why in report["stale"])
    if orphans["orphans"]:
        lines.append(f"Orphan files --prune would delete: {len(orphans['orphans'])}")
        lines.extend(f"  {_shown(p)}: {why}" for p, why in orphans["orphans"])
    failed = [f"not checked: {message}" for _, message in report["failed"]]
    failed.extend(f"{_shown(p)}: {why}" for p, why in orphans["failed"])
    return lines, failed


def prune_lines(report: dict) -> tuple[list[str], list[str]]:
    """The lines ``sfm ws mipmaps --prune`` prints for the report
    ``prune_mipmaps`` returns, and its warnings."""
    lines = []
    if report["orphans"]:
        lines.append(f"Deleted {len(report['orphans'])} orphan file(s):")
        lines.extend(f"  {_shown(p)}: {why}" for p, why in report["orphans"])
    else:
        lines.append("No orphan file: nothing was deleted.")
    lines.append(f"Kept {len(report['kept'])} mip level file(s).")
    if report["links"]:
        lines.append("Links, not entered:")
        lines.extend(f"  {_shown(p)}" for p in report["links"])
    failed = [f"{_shown(p)}: {why}" for p, why in report["failed"]]
    failed.extend(
        f"{_shown(p)} is in no workspace, so there is nothing to prune"
        for p in report["outside_workspaces"]
    )
    return lines, failed


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
_KINDS = {
    "sfmr": ".sfmr",
    "matches": ".matches",
    "camrig": ".camrig",
    "kdf": ".kdf",
}


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
