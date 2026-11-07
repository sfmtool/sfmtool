# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The `sfm pick-images` command: choose a video's frames for a solve."""

import json
import os
from pathlib import Path

import click

from .._cli_utils import timed_command
from .._filenames import expand_paths
from ._range_options import parse_image_selection, range_option, resolve_list_images

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@click.command("pick-images")
@timed_command
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--write-list",
    "write_list",
    metavar="NAME",
    default=None,
    help="Write the chosen frames as the named image list NAME in the workspace's "
    ".sfm-named-image-lists.json. Without it the command only reports.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace the list NAME if the workspace already defines one.",
)
@range_option(
    "A range expression of file numbers to choose from in the input directories.",
    on="paths",
)
@click.option(
    "--target-step",
    "target_step",
    type=float,
    default=10.0,
    metavar="PERCENT",
    show_default=True,
    help="The step to aim for between two chosen frames: the median displacement "
    "of their matched features, in percent of the image's long side.",
)
@click.option(
    "--step-band",
    "step_band",
    type=float,
    default=0.3,
    metavar="FRACTION",
    show_default=True,
    help="How far a step may be from the target and still be accepted: within "
    "target x (1 +/- FRACTION).",
)
@click.option(
    "--neighbour-blur-ratio",
    "neighbour_blur_ratio",
    type=float,
    default=0.85,
    metavar="RATIO",
    show_default=True,
    help="A frame about to be chosen is blurred against its neighbours, and "
    "swapped for a sharper one, when its sharpness is below RATIO times the "
    "sharpest of the frames up to 2 positions away.",
)
@click.option(
    "--min-step-matches",
    "min_step_matches",
    type=click.IntRange(min=1),
    default=600,
    metavar="N",
    show_default=True,
    help="The match floor: a step past the next frame keeps at least N mutual "
    "matches. The step to the next frame is always taken, so a capture whose "
    "neighbouring frames match with fewer keeps every frame.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Also write the whole report, every frame read and pair matched, to this "
    "JSON file.",
)
def pick_images(
    paths,
    write_list,
    force,
    range_expr,
    named_image_list,
    target_step,
    step_band,
    neighbour_blur_ratio,
    min_step_matches,
    json_path,
):
    """Choose the frames of a video a solve should use, and write them as a named image list.

    Walks each sequence from its first frame, choosing frames about
    --target-step percent of the image's long side apart, with no step past
    the next frame keeping fewer than --min-step-matches mutual matches, and
    a frame blurred against its neighbours swapped for a sharper one. Each
    frame read, or predicted to be read, gets a .sift
    file, extracted with the workspace's settings; with a feature tool other
    than sfmtool, every input frame is extracted first. With --write-list NAME the chosen frames are written as
    the named image list NAME, for `sfm solve -r NAME`.
    """
    from .._sfmtool.workspace import is_named_image_list_name

    if write_list is not None and not is_named_image_list_name(write_list):
        raise click.UsageError(
            f"--write-list '{write_list}' is not a valid list name: a name holds "
            "only the ASCII letters, the digits, '-', '_' and '.', with at least "
            "one letter."
        )
    if force and write_list is None:
        raise click.UsageError("--force replaces a written list; give --write-list.")

    selection = parse_image_selection(range_expr, named_image_list)
    if selection.list_name is not None and paths:
        raise click.UsageError(
            "A named image list replaces the image paths; give paths or a list, "
            "not both."
        )

    if selection.list_name is not None:
        listed = resolve_list_images(selection, extensions=IMAGE_EXTENSIONS)
        filenames = listed.paths
        workspace_dir = listed.workspace_dir
        given = f"Given {selection.describe_list()}"
    else:
        from .._sfmtool.reconstruction import RangeExpr
        from .._workspace import find_workspace_for_path

        if not paths:
            paths = (".",)
        numbers = None
        if selection.range_expr is not None:
            try:
                numbers = RangeExpr(selection.range_expr)
            except ValueError as e:
                raise click.UsageError(f"--range '{selection.range_expr}': {e}")
        filenames = expand_paths(
            [Path(p) for p in paths], extensions=IMAGE_EXTENSIONS, numbers=numbers
        )
        if not filenames:
            raise click.UsageError(
                "There were no image files to process in the provided directories."
            )
        common_parent = Path(
            os.path.commonpath([os.path.abspath(f) for f in filenames])
        )
        if common_parent.is_file():
            common_parent = common_parent.parent
        workspace_dir = find_workspace_for_path(common_parent)
        if workspace_dir is None:
            raise click.ClickException(
                f"No workspace found for images at {common_parent}. sfm pick-images "
                "extracts each frame's .sift file with the workspace's settings and "
                "writes its list into the workspace; initialize one with 'sfm ws init'."
            )
        given = f"Given {len(paths)} path(s)"

    _run(
        filenames=filenames,
        workspace_dir=Path(workspace_dir),
        given=given,
        write_list=write_list,
        force=force,
        options={
            "target_step_percent": target_step,
            "step_band": step_band,
            "neighbour_blur_ratio": neighbour_blur_ratio,
            "min_step_matches": min_step_matches,
        },
        json_path=Path(json_path) if json_path is not None else None,
    )


def _run(*, filenames, workspace_dir, given, write_list, force, options, json_path):
    """Collect the sequences, run the walk, print the report and write the
    list and the JSON file."""
    from .._pick_images import (
        PickImagesInputError,
        capture_sequences,
        collect_sequences,
        echo_frames_read,
        extract_every_frame,
        list_entries,
    )
    from .._pick_images_report import report_lines, slice_suggestion
    from .._sfmtool import ProgressCounter
    from .._sfmtool.analysis import pick_images as walk
    from .._sfmtool.workspace import NamedImageLists
    from .._workspace import load_workspace_config
    from ..rig.config import RigConfigError, _load_rig_config
    from ..sift.extract import SiftExtractionError, _check_versions_never_mix

    try:
        config = load_workspace_config(workspace_dir)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    workspace_file = workspace_dir / ".sfm-workspace.json"
    settings_source = f"The workspace file {workspace_file}"
    click.echo(f"Using workspace: {workspace_dir}")
    click.echo(f"  Feature tool: {config['feature_tool']}")

    # Version 1 settings, which write no .sift file, are refused before any
    # frame is read, with the message `sfm sift --extract` gives for them; the
    # walk checks the folders themselves.
    try:
        _check_versions_never_mix(
            config["feature_tool"], config["feature_options"], (), settings_source
        )
    except SiftExtractionError as e:
        raise click.ClickException(str(e))

    if write_list is not None and not force:
        try:
            lists = NamedImageLists.read(workspace_dir)
        except (ValueError, OSError) as e:
            raise click.ClickException(str(e))
        if lists is not None and write_list in lists:
            raise click.ClickException(
                f"{lists.path}: the named image list '{write_list}' is already "
                "defined; pass --force to replace it."
            )

    try:
        rig_configs = _load_rig_config(workspace_dir)
    except RigConfigError as e:
        raise click.ClickException(str(e))
    try:
        sequences = collect_sequences(filenames, workspace_dir, rig_configs)
    except PickImagesInputError as e:
        raise click.ClickException(str(e))

    noun = "sequence" if len(sequences) == 1 else "sequences"
    click.echo(f"{given}, {len(filenames)} image file(s) in {len(sequences)} {noun}:")
    for sequence in sequences:
        span = f"{sequence.numbers[0]}-{sequence.numbers[-1]}"
        if sequence.sensors is None:
            click.echo(f"  {sequence.name}, {len(sequence.numbers)} frames ({span})")
        else:
            click.echo(
                f"  {' and '.join(sequence.patterns)}, {len(sequence.numbers)} rig "
                f"frames of {len(sequence.sensors)} sensors ({span})"
            )
    click.echo("")

    capture = capture_sequences(sequences, workspace_dir, config["feature_prefix_dir"])
    # The walk extracts the sfmtool tool's files itself, ahead of the frames
    # it reads; for another tool every frame is extracted first, and the walk
    # reads the files.
    if config["feature_tool"] == "sfmtool":
        source = {
            "feature_options": config["feature_options"],
            "feature_prefix_dir": config["feature_prefix_dir"],
        }
        written_first = None
    else:
        click.echo(
            f"Extracting the .sift files of every frame with the "
            f"{config['feature_tool']} tool before the walk:"
        )
        try:
            written_first = extract_every_frame(capture, config, settings_source)
        except (SiftExtractionError, ValueError, OSError) as e:
            raise click.ClickException(str(e))
        click.echo("")
        source = {}
    counter = ProgressCounter()
    try:
        with echo_frames_read(counter, click.echo):
            report = walk(capture, progress=counter, **source, **options)
    except (ValueError, OverflowError, OSError) as e:
        # An option too large for the binding is an OverflowError whose note
        # names the option.
        raise click.ClickException("\n".join([str(e), *getattr(e, "__notes__", [])]))

    if written_first is not None:
        # The walk extracted nothing; what this run wrote, it wrote first.
        report["frames_extracted"] = [
            f["id"]["frame"] in written_first[f["id"]["sequence"]]
            for f in report["frames"]
        ]
        report["frames_extracted_unread"] = sum(map(len, written_first)) - sum(
            report["frames_extracted"]
        )
    extracted = [{} for _ in sequences]
    for frame, written in zip(report["frames"], report["frames_extracted"]):
        extracted[frame["id"]["sequence"]][frame["id"]["frame"]] = written
    for line in report_lines(
        report,
        sequences,
        extracted,
        options["target_step_percent"],
        options["step_band"],
        options["min_step_matches"],
    ):
        click.echo(line)

    chosen = [
        [s.numbers[p] for p in r["chosen"]]
        for s, r in zip(sequences, report["sequences"])
    ]
    written_list = None
    if write_list is not None:
        entries = [
            entry
            for sequence, numbers in zip(sequences, chosen)
            for entry in list_entries(sequence, numbers)
        ]
        try:
            lists_path = NamedImageLists.write_lists(
                workspace_dir, [(write_list, entries)], replace=force
            )
        except (ValueError, OSError) as e:
            raise click.ClickException(str(e))
        image_count = sum(
            len(numbers) * len(s.patterns) for s, numbers in zip(sequences, chosen)
        )
        written_list = {
            "name": write_list,
            "path": str(lists_path),
            "images": image_count,
            "entries": [{"pattern": p, "range": str(r)} for p, r in entries],
        }
        click.echo("")
        click.echo(
            f"Wrote the named image list {write_list} ({image_count} images) to "
            f"{lists_path}"
        )
        if all(s.sensors is None for s in sequences):
            suggestion = slice_suggestion(write_list, image_count)
            if suggestion is not None:
                click.echo(suggestion)
    else:
        click.echo("")
        click.echo(
            f"Chose {sum(len(n) for n in chosen)} frames; pass --write-list NAME to "
            "write them as a named image list."
        )

    if json_path is not None:
        data = dict(report)
        data["workspace"] = str(workspace_dir)
        data["options"] = options
        data["capture"] = [
            {
                "patterns": s.patterns,
                "sensors": s.sensors,
                "numbers": s.numbers,
                "images": [[p.as_posix() for p in images] for images in s.images],
            }
            for s in sequences
        ]
        data["chosen_numbers"] = chosen
        data["named_image_list"] = written_list
        json_path.write_text(json.dumps(data, indent=1) + "\n")
        click.echo(f"Wrote the report to {json_path}")
