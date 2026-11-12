# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The `sfm extract-frames` command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._video_frames import (
    DEFAULT_FPS,
    DEFAULT_MAX_FRAME_SIZE,
    VideoFramesError,
    extract_planned,
    parse_fps,
    parse_max_frame_size,
    plan_extraction,
)


class _FullOr(click.ParamType):
    """An option value that is `full` or what `parse` accepts; `parse` raises
    `ValueError` for anything else."""

    def __init__(self, name: str, parse):
        self.name = name
        self._parse = parse

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            return value
        try:
            return self._parse(value)
        except ValueError as error:
            self.fail(str(error), param, ctx)


@click.command("extract-frames")
@timed_command
@click.help_option("--help", "-h")
@click.argument(
    "videos",
    nargs=-1,
    required=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="The directory the frame directories go in, one per video, named "
    "after the video. Created when missing; it need not be in a workspace.",
)
@click.option(
    "--fps",
    type=_FullOr("N|full", parse_fps),
    default=str(DEFAULT_FPS),
    show_default=True,
    help="Frames per second of video time to keep, or `full` for every frame.",
)
@click.option(
    "--max-frame-size",
    type=_FullOr("PX|full", parse_max_frame_size),
    default=str(DEFAULT_MAX_FRAME_SIZE),
    show_default=True,
    help="The largest long side of a written frame, in pixels (at least 64). "
    "A larger frame is scaled down, a smaller one is never scaled up; `full` "
    "writes every frame at the video's size.",
)
def extract_frames(videos, output_dir, fps, max_frame_size):
    """Extract the frames of videos into one numbered JPEG sequence per video.

    Each VIDEO becomes the directory OUTPUT/<name>/, <name> being the video's
    file name without its extension, holding frame_000001.jpg,
    frame_000002.jpg, ... in presentation order and a video-frames.json
    record of each frame's time in the video. ffmpeg 5.1 or newer and ffprobe
    must be on PATH. Frames are upright (the video's rotation metadata is
    applied), and a variable frame rate is sampled without duplicating frames.

    Every input is checked before anything is created: an Insta360 .insv
    file (use `sfm insv2rig`), an image, a file that is not a video, an HDR
    video, two videos whose names differ only in letter case or extension, and
    a frame directory that exists and is not empty are refused.

    Example usage:

        sfm extract-frames walk1.mp4 walk2.MOV -o captures/garden

        sfm extract-frames orbit.mov -o frames --fps full --max-frame-size full
    """
    try:
        plan = plan_extraction(
            list(videos),
            output_dir,
            fps=fps,
            max_frame_size=max_frame_size,
            command="sfm extract-frames",
        )
        click.echo(f"ffmpeg: {plan.programs.ffmpeg} ({plan.programs.version})")
        extracted = extract_planned(plan)
    except VideoFramesError as error:
        raise click.ClickException(str(error))
    total = sum(video.frame_count for video in extracted)
    count = len(extracted)
    click.echo(
        f"Wrote {total} frames from {count} video{'' if count == 1 else 's'} "
        f"to {output_dir}"
    )
