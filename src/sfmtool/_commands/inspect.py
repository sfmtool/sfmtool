# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified file inspection command."""

import re
from pathlib import Path

import click

from .._cli_utils import timed_command
from ..analyze.summary import (
    print_camrig_summary,
    print_image_summary,
    print_matches_summary,
    print_point_summary,
    print_sfmr_summary,
    print_sift_summary,
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

_SUMMARY_DISPATCH = {
    ".sfmr": print_sfmr_summary,
    ".sift": print_sift_summary,
    ".matches": print_matches_summary,
    ".camrig": print_camrig_summary,
}

# A 3D point ID: pt3d_<8 hex chars of the .sfmr content hash>_<point index>.
_POINT_ID_RE = re.compile(r"^pt3d_([0-9a-fA-F]{8})_(\d+)$")


@click.command("inspect")
@timed_command
@click.help_option("--help", "-h")
@click.argument("target")
@click.argument("rest", nargs=-1)
@click.option(
    "--strips",
    "strips",
    is_flag=True,
    help="Render the listed points of a .sfmr as a patch-strip montage "
    "(labels | reference patch | per-view observation strip) for visually "
    "evaluating point quality. TARGET is the .sfmr; the remaining arguments are "
    "points to render (pt3d_<hash>_<index> ids and/or point-index range "
    "expressions like '5', '5-12', '1,4,7'), rendered in the order listed.",
)
@click.option(
    "-o",
    "--output",
    "output",
    type=click.Path(),
    default=None,
    help="Output PNG path for --strips (default: <stem>_strips.png in the "
    "current directory).",
)
@click.option(
    "--strips-views",
    type=click.IntRange(min=0),
    default=8,
    show_default=True,
    help="Cap observation tiles (views) per point with --strips (0 = all).",
)
@click.option(
    "--context",
    type=click.FloatRange(min=0),
    default=1.0,
    show_default=True,
    help="With --strips, pad each per-observation patch with this fraction of "
    "extra context around it (1.0 = +100%), drawing a border at the patch extent. "
    "0 = tight patches with no border. The reference patch always renders tight.",
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    is_flag=True,
    help="Print detailed information instead of a short summary.",
)
def inspect(target, rest, strips, output, strips_views, context, verbose):
    """Inspect an sfmtool file, image, or 3D point and print a summary.

    TARGET is a file path or a 3D point ID. A file's type is determined by
    extension:

    \b
      .sfmr     reconstruction
      .sift     feature file
      .matches  feature matches
      .camrig   camera rig
      image     .png / .jpg / .jpeg

    A 3D point ID has the form ``pt3d_<hash>_<index>`` (as shown by the GUI and
    verbose reconstruction reports). It is resolved by finding the .sfmr whose
    content hash matches ``<hash>`` within the workspace: pass the workspace
    directory (or any path inside it) as the optional second argument; the
    workspace is found by searching that directory and its parents. It defaults
    to the current directory.

    Without --verbose, prints a short summary. With --verbose, prints the full
    detail — for a point ID, the complete triangulation analysis, which reads
    the workspace ``.sift`` files.

    With --strips, TARGET is a .sfmr and the remaining arguments are points to
    render as a patch-strip montage (see --strips). A ``sift_files``
    reconstruction is converted to embedded patches with a light normal
    refinement first; an ``embedded_patches`` one is rendered as stored.

    For deep-analysis reports on a reconstruction (covisibility, frustum,
    depth, per-image metrics), use `sfm analyze`.

    Examples:

        sfm inspect reconstruction.sfmr

        sfm inspect image_001.sift --verbose

        sfm inspect pt3d_220747a8_96414

        sfm inspect pt3d_220747a8_96414 /data/KerryPark360 --verbose

        sfm inspect --strips reconstruction.sfmr 0-9 pt3d_220747a8_96414 -o strips.png
    """
    if strips:
        _inspect_strips_cmd(target, list(rest), output, strips_views, context)
        return

    if output is not None or strips_views != 8 or context != 1.0:
        raise click.UsageError(
            "--output / --strips-views / --context are only valid with --strips"
        )

    location = rest[0] if rest else None
    if len(rest) > 1:
        raise click.UsageError(
            "too many arguments; without --strips, inspect takes a single file "
            "or a pt3d_<hash>_<index> id with an optional workspace directory"
        )

    match = _POINT_ID_RE.match(target)
    if match is not None:
        _inspect_point(
            target, match.group(1).lower(), int(match.group(2)), location, verbose
        )
        return

    # Otherwise TARGET is a file path.
    if location is not None:
        raise click.UsageError(
            "the second argument is the workspace and is only valid with a "
            "pt3d_<hash>_<index> point ID"
        )
    path = Path(target)
    if not path.exists() or path.is_dir():
        raise click.UsageError(f"file not found: {target}")

    suffix = path.suffix.lower()
    summary_fn = _SUMMARY_DISPATCH.get(suffix)
    if summary_fn is None and suffix in _IMAGE_SUFFIXES:
        summary_fn = print_image_summary
    if summary_fn is None:
        raise click.UsageError(
            f"Cannot inspect '{path.name}': unsupported file type '{suffix}'. "
            "Supported types: .sfmr, .sift, .matches, .camrig, image files "
            "(.png, .jpg, .jpeg), and pt3d_<hash>_<index> point IDs."
        )

    try:
        summary_fn(path, verbose=verbose)
    except Exception as e:
        raise click.ClickException(str(e))


def _inspect_strips_cmd(target, specs, output, strips_views, context):
    """Render the listed points of a .sfmr as a patch-strip montage."""
    from .._inspect_strips import parse_point_specs, render_inspect_strips
    from .._sfmtool import SfmrReconstruction

    path = Path(target)
    if path.suffix.lower() != ".sfmr" or not path.exists() or path.is_dir():
        raise click.UsageError(
            f"--strips requires an existing .sfmr file as TARGET, got: {target}"
        )
    if not specs:
        raise click.UsageError(
            "--strips needs at least one point: a pt3d_<hash>_<index> id or a "
            "point-index range expression (e.g. '5', '5-12', '1,4,7')"
        )

    try:
        recon = SfmrReconstruction.load(path)
    except Exception as e:
        raise click.ClickException(str(e))

    point_indexes = parse_point_specs(recon, specs)
    out_path = Path(output) if output else Path.cwd() / f"{path.stem}_strips.png"

    try:
        render_inspect_strips(
            recon,
            point_indexes,
            out_path,
            max_views=strips_views,
            context=context,
        )
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


def _inspect_point(point_id, hash_prefix, point_index, location, verbose):
    """Resolve a pt3d_ point ID to its .sfmr and print the point summary."""
    from .._sfmtool import SfmrReconstruction
    from .._workspace import find_workspace_for_path

    base = Path(location) if location else Path.cwd()
    if not base.exists():
        raise click.UsageError(f"location does not exist: {base}")

    workspace = find_workspace_for_path(base) or base
    sfmr_path = _find_sfmr_by_content_hash(workspace, hash_prefix)
    if sfmr_path is None:
        raise click.ClickException(
            f"no .sfmr under '{workspace}' has content hash '{hash_prefix}' "
            f"(from {point_id}). Pass the workspace directory as the second argument."
        )

    try:
        recon = SfmrReconstruction.load(sfmr_path)
        if point_index >= recon.point_count:
            raise click.ClickException(
                f"{point_id}: point index {point_index} is out of range — "
                f"{sfmr_path.name} has {recon.point_count} points."
            )
        print_point_summary(recon, point_index, point_id, sfmr_path, verbose=verbose)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


def _find_sfmr_by_content_hash(workspace, hash_prefix):
    """First .sfmr under `workspace` whose content hash starts with `hash_prefix`.

    Search order follows the sfmr-format spec: the conventional ``sfmr/``
    subdirectory first, then the workspace root, then the rest of the tree
    (skipping hidden directories). Reading each candidate's hash decompresses
    only ``content_hash.json.zst``, not the reconstruction data.
    """
    from .._sfmtool.io import read_sfmr_content_hash

    def matches(path: Path) -> bool:
        try:
            return read_sfmr_content_hash(str(path))[:8].lower() == hash_prefix
        except Exception:
            return False

    # 1. The conventional sfmr/ subdirectory.
    sfmr_dir = workspace / "sfmr"
    if sfmr_dir.is_dir():
        for path in sorted(sfmr_dir.glob("*.sfmr")):
            if matches(path):
                return path
    # 2. The workspace root.
    for path in sorted(workspace.glob("*.sfmr")):
        if matches(path):
            return path
    # 3. The rest of the tree, skipping hidden / already-searched directories.
    for path in sorted(workspace.rglob("*.sfmr")):
        if path.parent == workspace or path.parent == sfmr_dir:
            continue
        if any(part.startswith(".") for part in path.relative_to(workspace).parts):
            continue
        if matches(path):
            return path
    return None
