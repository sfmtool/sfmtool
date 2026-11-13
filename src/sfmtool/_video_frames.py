# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the frames of videos into numbered JPEG sequences with ffmpeg.

`sfm extract-frames` is the caller; specs/cli/workspace/extract-frames-command.md
describes the filters, the naming, the checks and the output, and
specs/workspace/video-frames.md the `video-frames.json` record written beside
each video's frames. The work is in two steps so a caller can check every
input before it creates anything: `plan_extraction` finds ffmpeg, probes each
video and applies every check, creating nothing, and `extract_planned`
extracts the planned videos one ffmpeg process at a time. ffmpeg writes an SDR
video's JPEG frames itself; an HDR video's frames it decodes to 16-bit Y'CbCr
on a pipe, and the binding `write_image_from_hdr` converts each to sRGB and
writes it (specs/core/camera/hdr-to-srgb.md).

The module imports only the standard library and click at the top, so `sfm
version` takes its ffmpeg discovery (`locate_program`) from here; numpy and
the extension are imported when an HDR video is extracted.
"""

import json
import math
import os
import re
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import click

# The file name pattern of the frames in each video's directory.
FRAME_PATTERN = "frame_%06d.jpg"

# The record written beside each video's frames, and its version.
RECORD_NAME = "video-frames.json"
RECORD_VERSION = 2

DEFAULT_FPS = 10
DEFAULT_MAX_FRAME_SIZE = 1920
MIN_MAX_FRAME_SIZE = 64

# The oldest ffmpeg release that has `-fps_mode`. A development build reports
# no release number, so its libavutil version is compared instead: 57.28 is
# the libavutil of release 5.1.
MIN_FFMPEG_RELEASE = (5, 1)
MIN_LIBAVUTIL = (57, 28)

# How often the progress line is printed while a video is extracted.
PROGRESS_INTERVAL_S = 10.0

# How long `ffmpeg -version` may take.
_VERSION_TIMEOUT_S = 10.0

# How long a terminated ffmpeg is given to exit before it is killed.
_TERMINATE_WAIT_S = 5.0

# How many times `run_for_output` runs a program that exits with status 0
# having printed nothing.
_EMPTY_OUTPUT_ATTEMPTS = 3

# How many of ffmpeg's last stderr lines a failure message quotes.
_STDERR_TAIL_LINES = 20

# The transfer characteristics, as ffprobe names them, of HDR video.
_HDR_TRANSFERS = {"arib-std-b67": "HLG", "smpte2084": "PQ"}

# The tags an HDR video may carry to be converted, as ffprobe names them, with
# `None` and "unknown" for a tag the video does not state: BT.2020
# non-constant-luminance Y'CbCr, BT.2020 primaries, narrow ("tv") range.
_HDR_ACCEPTED_TAGS = {
    "color_space": ("bt2020nc", None, "unknown"),
    "color_primaries": ("bt2020", None, "unknown"),
    "color_range": ("tv", None, "unknown"),
}

# The scaler flags of every scale filter on an HDR video's frames, so the
# resampling and the chroma upsampling to 4:4:4 are the ones named here rather
# than a build's default.
_HDR_SWS_FLAGS = "area+accurate_rnd+full_chroma_int"

# How many HDR frames are converted and written at once, each on its own
# thread (the conversion of one frame is itself parallel over its rows), and
# how many decoded frames may wait for a thread.
_HDR_WRITERS = 4
_HDR_PENDING_FRAMES = 8

# Image extensions the commands collect; ffmpeg would open such a file as a
# one-frame video.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

_INSTALL_HINT = (
    "Install ffmpeg 5.1 or newer (https://ffmpeg.org/download.html; "
    "`winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, "
    "the `ffmpeg` package on Linux, or `ffmpeg` from conda-forge) and put the "
    "folder that holds both ffmpeg and ffprobe on PATH. `sfm version` shows the "
    "ffmpeg it finds."
)


class VideoFramesError(Exception):
    """A check or an extraction failed; the message says which input and why."""


def parse_fps(text: str) -> float | None:
    """Parse an `--fps` value: `full` gives `None`, otherwise a finite
    positive number. Raises `ValueError` naming what is wrong."""
    if text.strip().lower() == "full":
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{text!r} is neither a number nor 'full'") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{text!r} is not a positive number")
    return value


def parse_max_frame_size(text: str) -> int | None:
    """Parse a `--max-frame-size` value: `full` gives `None`, otherwise an
    integer of at least `MIN_MAX_FRAME_SIZE`. Raises `ValueError`."""
    if text.strip().lower() == "full":
        return None
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{text!r} is neither an integer nor 'full'") from None
    if value < MIN_MAX_FRAME_SIZE:
        raise ValueError(f"{value} is less than {MIN_MAX_FRAME_SIZE}")
    return value


def run_for_output(
    args: list[str], timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Run a program for what it prints on stdout, as text.

    Some ffmpeg builds (conda-forge's win-64 ffmpeg 9.0.2 among them) now and
    then lose what they printed to stdout through the C library at exit,
    while still exiting with status 0: measured, about 3 runs in 100 of
    `ffmpeg -version` printed nothing, and about 2 in 100 of ffprobe's JSON
    description were cut short (ffprobe's `-o -` avoids that, and
    `probe_video` passes it). `ffmpeg -version` prints less than one buffer,
    so a run loses all of it or none, and a run that printed nothing is
    repeated, up to `_EMPTY_OUTPUT_ATTEMPTS` runs in all. Raises what
    `subprocess.run` raises."""
    for _ in range(_EMPTY_OUTPUT_ATTEMPTS):
        result = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0 or result.stdout:
            break
    return result


def locate_program(name: str) -> str | None:
    """The path of the program `name` on PATH, normalized, or `None`."""
    path = shutil.which(name)
    return None if path is None else os.path.normpath(path)


def ffmpeg_version_name(first_line: str) -> str:
    """The version an `ffmpeg -version` first line names: "ffmpeg version
    7.1-full_build Copyright (c) 2000-2024 ..." gives "7.1-full_build". A
    line in another form is returned whole."""
    words = first_line.split()
    if len(words) > 2 and words[:2] == ["ffmpeg", "version"]:
        return words[2]
    return first_line


def ffmpeg_version_number(version_output: str) -> tuple[int, int] | None:
    """The release (major, minor) an `ffmpeg -version` output says, or `None`.

    A release build's first line names its release ("ffmpeg version 7.1.1",
    "ffmpeg version n6.0", "ffmpeg version 4.4.2-0ubuntu0.22.04.1"). A
    development build's names a date or a commit instead, and its libavutil
    line is read and mapped to the first release that carries it: a libavutil
    at or above `MIN_LIBAVUTIL` reads as `MIN_FFMPEG_RELEASE`, an older one as
    (0, 0)."""
    lines = version_output.splitlines()
    if lines:
        match = re.match(r"n?(\d+)\.(\d+)", ffmpeg_version_name(lines[0]))
        if match:
            return int(match.group(1)), int(match.group(2))
    for line in lines:
        match = re.match(r"\s*libavutil\s+(\d+)\.\s*(\d+)\.", line)
        if match:
            libavutil = int(match.group(1)), int(match.group(2))
            return MIN_FFMPEG_RELEASE if libavutil >= MIN_LIBAVUTIL else (0, 0)
    return None


@dataclass(frozen=True)
class FfmpegPrograms:
    """The ffmpeg and ffprobe a run uses, and ffmpeg's version text."""

    ffmpeg: str
    ffprobe: str
    version: str


def find_ffmpeg(command: str) -> FfmpegPrograms:
    """Find ffmpeg and ffprobe on PATH and check ffmpeg's version.

    `command` names the command that needs them, for the message. Raises
    `VideoFramesError` when either is missing, when `ffmpeg -version` fails,
    or when the release is older than `MIN_FFMPEG_RELEASE`."""
    ffmpeg = locate_program("ffmpeg")
    ffprobe = locate_program("ffprobe")
    missing = [
        name
        for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
        if path is None
    ]
    if missing:
        raise VideoFramesError(
            f"{command} needs ffmpeg and ffprobe on PATH, and found no "
            f"{' and no '.join(missing)}. {_INSTALL_HINT}"
        )
    try:
        result = run_for_output([ffmpeg, "-version"], timeout=_VERSION_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as error:
        raise VideoFramesError(f"Could not run {ffmpeg} -version: {error}") from None
    if result.returncode != 0:
        raise VideoFramesError(
            f"`{ffmpeg} -version` exited with status {result.returncode}"
        )
    first = result.stdout.splitlines()[0] if result.stdout else ""
    version = ffmpeg_version_name(first)
    number = ffmpeg_version_number(result.stdout)
    needed = ".".join(map(str, MIN_FFMPEG_RELEASE))
    if number is None:
        raise VideoFramesError(
            f"{command} needs ffmpeg {needed} or newer, and could not tell the "
            f"version of {ffmpeg} from its first line: {first!r}. {_INSTALL_HINT}"
        )
    if number < MIN_FFMPEG_RELEASE:
        raise VideoFramesError(
            f"{command} needs ffmpeg {needed} or newer, and {ffmpeg} is version "
            f"{version}. {_INSTALL_HINT}"
        )
    return FfmpegPrograms(ffmpeg=ffmpeg, ffprobe=ffprobe, version=version)


def _fraction(text: str | None) -> Fraction | None:
    """A rate or aspect ffprobe writes as "N/D", or `None` when unset (a zero
    numerator or denominator, or text that is not a ratio)."""
    if not text:
        return None
    numerator, _, denominator = text.replace(":", "/").partition("/")
    try:
        value = Fraction(int(numerator), int(denominator or 1))
    except ValueError, ZeroDivisionError:
        return None
    return value if value > 0 else None


def _float_or_none(text) -> float | None:
    try:
        value = float(text)
    except TypeError, ValueError:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class VideoProbe:
    """What ffprobe says about the video stream a video's frames come from."""

    path: Path
    # The stream's position among the file's video streams (`-map 0:v:N`).
    video_stream: int
    codec: str
    width: int
    height: int
    # The display rotation in degrees, positive counterclockwise, as ffprobe
    # reports it from the display matrix; 0 when there is none.
    rotation_degrees: float
    sample_aspect_ratio: Fraction | None
    # ffprobe's r_frame_rate (the lowest rate that represents every timestamp)
    # and avg_frame_rate.
    base_frame_rate: Fraction | None
    average_frame_rate: Fraction | None
    duration_s: float | None
    color_transfer: str | None
    color_space: str | None = None
    color_primaries: str | None = None
    color_range: str | None = None

    @property
    def hdr_transfer(self) -> str | None:
        """ "HLG" or "PQ" for an HDR video, `None` for SDR."""
        return _HDR_TRANSFERS.get(self.color_transfer)

    @property
    def is_variable_rate(self) -> bool:
        return (
            self.base_frame_rate is not None
            and self.average_frame_rate is not None
            and self.base_frame_rate != self.average_frame_rate
        )

    @property
    def long_side(self) -> float:
        """The long side of a displayed frame, in pixels: the stored width
        scaled to square pixels, or the height. A rotation swaps the sides
        but leaves the long side as it is."""
        sar = self.sample_aspect_ratio or Fraction(1)
        return max(self.width * float(sar), float(self.height))


def _rotation(stream: dict) -> float:
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            value = _float_or_none(side_data["rotation"])
            if value is not None:
                return value
    value = _float_or_none(stream.get("tags", {}).get("rotate"))
    # The old `rotate` tag is clockwise.
    return -value if value else 0.0


def probe_video(path: Path, ffprobe: str) -> VideoProbe:
    """Probe `path` with ffprobe and describe its first video stream that is
    not an attached picture. Raises `VideoFramesError` when ffprobe cannot
    open the file or it has no such stream."""
    try:
        # `-o -` writes the description through ffmpeg's own output layer,
        # which flushes it before exit; see `run_for_output`.
        result = run_for_output(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json"]
            + ["-o", "-", str(path)]
        )
    except OSError as error:
        raise VideoFramesError(f"{path}: could not run ffprobe: {error}") from None
    stderr_lines = [line for line in result.stderr.splitlines() if line.strip()]
    if result.returncode != 0:
        last = stderr_lines[-1] if stderr_lines else f"exit status {result.returncode}"
        raise VideoFramesError(f"{path}: ffprobe cannot read it as a video ({last})")
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise VideoFramesError(
            f"{path}: ffprobe printed no readable description"
        ) from None
    video_streams = [
        s for s in info.get("streams", []) if s.get("codec_type") == "video"
    ]
    chosen = next(
        (
            (index, stream)
            for index, stream in enumerate(video_streams)
            if not stream.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if chosen is None or not chosen[1].get("width") or not chosen[1].get("height"):
        what = " other than cover art" if video_streams else ""
        raise VideoFramesError(f"{path}: it has no video stream{what}")
    index, stream = chosen
    duration = _float_or_none(stream.get("duration"))
    if duration is None:
        duration = _float_or_none(info.get("format", {}).get("duration"))
    return VideoProbe(
        path=path,
        video_stream=index,
        codec=stream.get("codec_name", "unknown"),
        width=int(stream["width"]),
        height=int(stream["height"]),
        rotation_degrees=_rotation(stream),
        sample_aspect_ratio=_fraction(stream.get("sample_aspect_ratio")),
        base_frame_rate=_fraction(stream.get("r_frame_rate")),
        average_frame_rate=_fraction(stream.get("avg_frame_rate")),
        duration_s=duration,
        color_transfer=stream.get("color_transfer"),
        color_space=stream.get("color_space"),
        color_primaries=stream.get("color_primaries"),
        color_range=stream.get("color_range"),
    )


@dataclass(frozen=True)
class PlannedVideo:
    """One checked video and the directory its frames go in."""

    probe: VideoProbe
    frame_dir: Path


@dataclass(frozen=True)
class ExtractionPlan:
    """Every video of a run, checked, with the programs and options to use."""

    programs: FfmpegPrograms
    out_dir: Path
    # Frames per second to keep, or `None` for every frame.
    fps: float | None
    # The largest long side of a written frame, or `None` for the video's size.
    max_frame_size: int | None
    videos: tuple[PlannedVideo, ...]
    # The exposure change in stops for every HDR video's frames, as given, or
    # `None` when not given, which converts at 0.
    hdr_exposure: float | None = None


def _directory_problem(directory: Path) -> str | None:
    """Why `directory` cannot take a video's frames, or `None` when it is
    missing or an empty directory."""
    if not directory.exists():
        return None
    if not directory.is_dir():
        return "exists and is not a directory"
    if any(directory.iterdir()):
        return "is not empty"
    return None


class FullOr(click.ParamType):
    """An option value that is `full` or what `parse` accepts, for `--fps`
    and `--max-frame-size`; `parse` raises `ValueError` for anything else."""

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


class HdrExposure(click.ParamType):
    """The `--hdr-exposure` value: a finite number of stops within the
    conversion's `HDR_EXPOSURE_STOPS_LIMIT` either way, checked when the
    option is read so that a bad value stops the command before anything is
    created."""

    name = "STOPS"

    def convert(self, value, param, ctx):
        if isinstance(value, float):
            return value
        from .fileio import HDR_EXPOSURE_STOPS_LIMIT as limit

        try:
            stops = float(value)
        except ValueError:
            stops = math.nan
        if not math.isfinite(stops) or abs(stops) > limit:
            self.fail(
                f"{value!r} is not a number of stops from {-limit:g} to {limit:g}",
                param,
                ctx,
            )
        return stops


def plan_extraction(
    videos: list[Path],
    out_dir: Path,
    *,
    fps: float | None,
    max_frame_size: int | None,
    command: str,
    problems: tuple[str, ...] = (),
    hdr_exposure: float | None = None,
) -> ExtractionPlan:
    """Check every input of an extraction and plan it, creating nothing.

    Finds ffmpeg and ffprobe (`find_ffmpeg`; without them nothing else is
    checked), then checks each video: not an `.insv`, not an image or a
    directory, a file ffprobe reads as a video, for HDR video the tags the
    conversion takes (`_HDR_ACCEPTED_TAGS`), a stem with no `*` and no other
    video's stem in another letter case, and a frame directory
    `out_dir/<stem>/` that is missing or empty. `hdr_exposure`, the
    `--hdr-exposure` given or `None`, is a problem when no video read is HDR.
    Raises one `VideoFramesError` listing every problem found, after
    `problems`, those the caller found. `command` names the caller for the
    messages.
    """
    programs = find_ffmpeg(command)
    if out_dir.exists() and not out_dir.is_dir():
        raise VideoFramesError(f"{out_dir} exists and is not a directory")
    problems = list(problems)
    probes: list[VideoProbe | None] = []
    for video in videos:
        probes.append(None)
        suffix = video.suffix.lower()
        if suffix == ".insv":
            problems.append(
                f"{video}: an Insta360 .insv file holds two fisheye lenses; "
                "extract it with `sfm insv2rig`, which writes both lenses and "
                "their .camrig"
            )
            continue
        if "*" in video.stem:
            problems.append(
                f"{video}: its name has a '*', which a named image list's "
                "pattern cannot hold"
            )
        if suffix in _IMAGE_EXTENSIONS:
            problems.append(f"{video}: an image, not a video")
            continue
        if video.is_dir():
            problems.append(f"{video}: a directory, not a video")
            continue
        if not video.is_file():
            problems.append(f"{video}: no such file")
            continue
        try:
            probe = probe_video(video, programs.ffprobe)
        except VideoFramesError as error:
            problems.append(str(error))
            continue
        if probe.hdr_transfer is not None:
            wrong = [
                f"{tag} {getattr(probe, tag)}"
                for tag, accepted in _HDR_ACCEPTED_TAGS.items()
                if getattr(probe, tag) not in accepted
            ]
            if wrong:
                problems.append(
                    f"{video}: HDR video ({probe.hdr_transfer}) with "
                    f"{', '.join(wrong)}; the HDR conversion takes BT.2020 "
                    "non-constant-luminance Y'CbCr (bt2020nc), BT.2020 "
                    "primaries and narrow (tv) range"
                )
                continue
        probes[-1] = probe

    if hdr_exposure is not None and not any(
        probe is not None and probe.hdr_transfer is not None for probe in probes
    ):
        problems.append(
            "--hdr-exposure sets the exposure of HDR video frames, and none "
            "of the videos read is HDR"
        )

    by_stem: dict[str, list[Path]] = {}
    for video in videos:
        by_stem.setdefault(video.stem.casefold(), []).append(video)
    for same in by_stem.values():
        if len(same) > 1:
            names = ", ".join(str(v) for v in same)
            problems.append(
                f"{names}: these would share one frame directory, "
                f"{out_dir / same[0].stem}, since their names without the "
                "extension are equal ignoring letter case"
            )

    planned: list[PlannedVideo] = []
    checked_dirs: set[str] = set()
    for video, probe in zip(videos, probes):
        frame_dir = out_dir / video.stem
        if (
            video.suffix.lower() != ".insv"
            and video.stem.casefold() not in checked_dirs
        ):
            checked_dirs.add(video.stem.casefold())
            problem = _directory_problem(frame_dir)
            if problem is not None:
                problems.append(f"{video}: its frame directory {frame_dir} {problem}")
        if probe is not None:
            planned.append(PlannedVideo(probe=probe, frame_dir=frame_dir))

    if problems:
        count = len(problems)
        raise VideoFramesError(
            f"{count} problem{'' if count == 1 else 's'} found, so nothing was "
            "created:\n" + "\n".join(f"  {p}" for p in problems)
        )
    return ExtractionPlan(
        programs=programs,
        out_dir=out_dir,
        fps=fps,
        max_frame_size=max_frame_size,
        videos=tuple(planned),
        hdr_exposure=hdr_exposure,
    )


def _number_text(value: float) -> str:
    """A rate or angle as a person writes it: `10`, `29.97`."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _option_value(value: float | None) -> float | int | str:
    """An option as the record stores it: "full", or the number given."""
    if value is None:
        return "full"
    return int(value) if value == int(value) else value


def _duration_text(seconds: float) -> str:
    if seconds >= 60:
        minutes, rest = divmod(seconds, 60)
        return f"{int(minutes)}m {rest:.1f}s"
    return f"{seconds:.1f}s"


def _keeps_every_frame(fps: float | None, probe: VideoProbe) -> bool:
    """Whether every decoded frame is kept: with `full`, or with a rate at or
    above the video's base rate, whose frames are at least one interval
    apart."""
    if fps is None:
        return True
    return probe.base_frame_rate is not None and fps >= probe.base_frame_rate


def video_filters(
    probe: VideoProbe, fps: float | None, max_frame_size: int | None
) -> str:
    """The `-vf` filter chain for one video. ffmpeg's autorotate inserts the
    turn that makes the frames upright before it. An HDR video's chain gives
    every scale filter `_HDR_SWS_FLAGS` and ends in 16-bit 4:4:4 Y'CbCr."""
    hdr = probe.hdr_transfer is not None
    filters = []
    if not _keeps_every_frame(fps, probe):
        n = repr(float(fps))
        filters.append(
            f"select='isnan(prev_selected_t)"
            f"+gt(floor(t*{n}),floor(prev_selected_t*{n}))'"
        )
    sar = probe.sample_aspect_ratio
    if sar is not None and sar != 1:
        flags = f":flags={_HDR_SWS_FLAGS}" if hdr else ""
        filters.append(f"scale=iw*sar:ih{flags},setsar=1")
    resized = max_frame_size is not None and probe.long_side > max_frame_size
    if resized:
        s = max_frame_size
        flags = _HDR_SWS_FLAGS if hdr else "area"
        filters.append(
            f"scale='if(gte(iw,ih),{s},-2)':'if(gte(iw,ih),-2,{s})':flags={flags},setsar=1"
        )
    if hdr:
        if not resized:
            filters.append(f"scale=flags={_HDR_SWS_FLAGS}")
        filters.append("format=yuv444p16le")
    filters.append("showinfo")
    return ",".join(filters)


def ffmpeg_arguments(
    programs: FfmpegPrograms,
    probe: VideoProbe,
    frame_dir: Path,
    fps: float | None,
    max_frame_size: int | None,
) -> list[str]:
    """The ffmpeg command line that extracts one video's frames: JPEG files
    for SDR video, raw 16-bit Y'CbCr frames on stdout for HDR video."""
    if probe.hdr_transfer is not None:
        output = ["-f", "rawvideo", "-pix_fmt", "yuv444p16le", "pipe:1"]
    else:
        output = ["-q:v", "2", str(frame_dir / FRAME_PATTERN)]
    return [
        programs.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-i",
        str(probe.path),
        "-map",
        f"0:v:{probe.video_stream}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        video_filters(probe, fps, max_frame_size),
        "-fps_mode",
        "passthrough",
        *output,
    ]


def _stderr_lines(stream, lines: queue.Queue) -> None:
    """Put each line of ffmpeg's stderr (bytes) on `lines` as text, then
    `None` at its end."""
    try:
        for raw in stream:
            lines.put(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
    finally:
        lines.put(None)


def _read_exactly(stream, count: int) -> bytes | None:
    """`count` bytes from `stream`, or `None` when it ends first."""
    data = bytearray()
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)


def run_ffmpeg(args: list[str], on_line, on_frame=None) -> tuple[int, list[str]]:
    """Run ffmpeg, calling `on_line` with each line of its stderr, and return
    its exit status and its last `_STDERR_TAIL_LINES` stderr lines.

    With `on_frame`, ffmpeg writes raw frames to its stdout: `on_line`
    returns the byte count of the frame a line announces (`showinfo` logs
    each frame before it is written) or `None`, and that many bytes are read
    from stdout and passed to `on_frame`. stderr is then read on a thread and
    the lines handled on the calling thread, so an exception from either
    callback reaches the caller. When stdout ends before a frame's bytes, the
    frame is not passed on and the rest of stderr is still read.

    On any exception while it runs, Ctrl+C included, ffmpeg is terminated
    (and killed if it has not exited after `_TERMINATE_WAIT_S`), waited for,
    and the exception re-raised."""
    tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
    if on_frame is None:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    try:
        if on_frame is None:
            for line in process.stderr:
                line = line.rstrip("\r\n")
                tail.append(line)
                on_line(line)
        else:
            lines: queue.Queue = queue.Queue()
            reader = threading.Thread(
                target=_stderr_lines, args=(process.stderr, lines), daemon=True
            )
            reader.start()
            stdout_open = True
            while True:
                try:
                    # A timeout, so that Ctrl+C is seen while waiting.
                    line = lines.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                tail.append(line)
                size = on_line(line)
                if size is not None and stdout_open:
                    data = _read_exactly(process.stdout, size)
                    if data is None:
                        stdout_open = False
                    else:
                        on_frame(data)
            reader.join()
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=_TERMINATE_WAIT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stderr.close()
        if process.stdout is not None:
            process.stdout.close()
    return returncode, list(tail)


# showinfo's lines: the time base its frames' pts are in, then one line per
# frame with its index, pts, pts_time and size.
_SHOWINFO_TIME_BASE = re.compile(
    r"\[Parsed_showinfo_\d+ @ [^\]]*\] config in time_base: (\d+)/(\d+)"
)
_SHOWINFO_FRAME = re.compile(
    r"\[Parsed_showinfo_\d+ @ [^\]]*\] n:\s*(\d+) pts:\s*(-?\d+) pts_time:(\S+) "
    r".* s:(\d+)x(\d+) "
)


@dataclass(frozen=True)
class ExtractedVideo:
    """One extracted video: where its frames are, its record, and where its
    extraction's time went (`times`).

    `times["seconds"]` is the extraction's wall time, from the call of
    `extract_video_frames` to its return. For an SDR video ffmpeg decodes,
    scales and writes the JPEG frames in its one process, and that is all
    there is; `times["hdr"]` is `None`. For an HDR video `times["hdr"]` splits
    it: `waiting_for_writers_seconds`, the part of `seconds` the calling thread
    spent blocked on the writer threads, and `decoding_seconds`, the rest,
    ffmpeg's decoding read from its pipe and the command's handling of each
    frame; and, on the `writers` threads, which run at the same time as the
    calling thread and as each other, `conversion_seconds` and
    `encoding_seconds`, each the sum over the frames of
    `write_image_from_hdr`'s own figure."""

    frame_dir: Path
    frame_pattern: str
    frame_count: int
    record: dict
    times: dict


def _video_line(probe: VideoProbe) -> str:
    parts = [f"{probe.codec} {probe.width}x{probe.height}"]
    if probe.hdr_transfer is not None:
        parts.append(f"HDR {probe.hdr_transfer}")
    if probe.rotation_degrees:
        parts.append(f"rotated {_number_text(probe.rotation_degrees)}")
    if probe.average_frame_rate is not None:
        rate = f"{float(probe.average_frame_rate):.2f} fps"
        if probe.is_variable_rate:
            rate += " (variable)"
        parts.append(rate)
    if probe.duration_s is not None:
        parts.append(f"{probe.duration_s:.1f} s")
    return f"{probe.path.name}: {', '.join(parts)}"


def extract_video_frames(
    plan: ExtractionPlan, video: PlannedVideo, echo=click.echo
) -> ExtractedVideo:
    """Extract one planned video's frames into its frame directory and write
    its `video-frames.json` record, printing its lines through `echo`.

    Raises `VideoFramesError` when ffmpeg fails or what it reports does not
    match the files it wrote. On Ctrl+C ffmpeg is stopped and the frames
    written so far are left, with no record."""
    probe = video.probe
    fps = plan.fps
    echo(_video_line(probe))
    keeps_all = _keeps_every_frame(fps, probe)
    if fps is not None and keeps_all:
        echo(
            f"  --fps {_number_text(fps)} is at or above the video's frame rate "
            f"{_number_text(float(probe.base_frame_rate))}, so every frame is kept"
        )
    rate_text = "every frame" if keeps_all else f"{_number_text(fps)} fps"
    echo(f"  -> {video.frame_dir}{os.sep} at {rate_text}")
    hdr_conversion = None
    if probe.hdr_transfer is not None:
        from .fileio import HDR_TO_SRGB_RECIPE, HDR_TO_SRGB_RECIPE_VERSION

        exposure = plan.hdr_exposure or 0.0
        hdr_conversion = {
            "transfer": probe.hdr_transfer,
            "recipe": HDR_TO_SRGB_RECIPE,
            "recipe_version": HDR_TO_SRGB_RECIPE_VERSION,
            "exposure_stops": _option_value(exposure),
        }
        echo(
            f"  converted from {probe.hdr_transfer} to sRGB by "
            f"{HDR_TO_SRGB_RECIPE} {HDR_TO_SRGB_RECIPE_VERSION}, exposure "
            f"{exposure:+g} stops"
        )

    expected = None
    rate = probe.average_frame_rate if keeps_all else fps
    if probe.duration_s is not None and rate is not None:
        expected = round(probe.duration_s * float(rate))

    time_base: list[Fraction] = []
    times: list[float] = []
    sizes: list[tuple[int, int]] = []
    start = time.monotonic()
    last_report = start

    def on_line(line: str) -> int | None:
        nonlocal last_report
        match = _SHOWINFO_FRAME.search(line)
        if match:
            if time_base:
                # pts in the time base is exact; pts_time is printed to six
                # significant digits, which is 10 ms after 1000 s.
                seconds = float(int(match.group(2)) * time_base[-1])
            else:
                seconds = float(match.group(3))
            times.append(round(seconds, 6))
            if not sizes:
                sizes.append((int(match.group(4)), int(match.group(5))))
            now = time.monotonic()
            if now - last_report >= PROGRESS_INTERVAL_S:
                last_report = now
                about = f" of about {expected}" if expected is not None else ""
                echo(f"  {len(times)}{about} frames ({now - start:.0f} s)")
            # A 16-bit sample per plane and pixel, three planes.
            return 6 * int(match.group(4)) * int(match.group(5))
        match = _SHOWINFO_TIME_BASE.search(line)
        if match:
            time_base.append(Fraction(int(match.group(1)), int(match.group(2))))
        return None

    video.frame_dir.mkdir(parents=True, exist_ok=True)
    args = ffmpeg_arguments(
        plan.programs, probe, video.frame_dir, fps, plan.max_frame_size
    )
    hdr_times = None
    if hdr_conversion is None:
        returncode, tail = run_ffmpeg(args, on_line)
    else:
        returncode, tail, hdr_times = _run_ffmpeg_hdr(
            args, on_line, video.frame_dir, probe.hdr_transfer, exposure
        )
    if returncode != 0:
        quoted = "\n".join(f"    {line}" for line in tail)
        raise VideoFramesError(
            f"ffmpeg failed on {probe.path} with exit status {returncode}. "
            f"Its last lines:\n{quoted}"
        )
    count = len(times)
    if (
        count == 0
        or not (video.frame_dir / (FRAME_PATTERN % count)).is_file()
        or (video.frame_dir / (FRAME_PATTERN % (count + 1))).exists()
    ):
        raise VideoFramesError(
            f"ffmpeg reported {count} frames of {probe.path}, which does not "
            f"match the frame files in {video.frame_dir}"
        )
    width, height = sizes[0]
    record = {
        "version": RECORD_VERSION,
        "video": str(probe.path.resolve()),
        "video_stream": probe.video_stream,
        "codec": probe.codec,
        "video_size": [probe.width, probe.height],
        "rotation_degrees": _option_value(probe.rotation_degrees),
        "color_transfer": probe.color_transfer,
        "hdr_conversion": hdr_conversion,
        "fps": _option_value(fps),
        "max_frame_size": _option_value(plan.max_frame_size),
        "frame_size": [width, height],
        "frame_pattern": FRAME_PATTERN,
        "frame_times_s": times,
    }
    # One key per line, each value on its key's line, so the frame times do
    # not take a line each.
    entries = (
        f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in record.items()
    )
    (video.frame_dir / RECORD_NAME).write_text(
        "{\n" + ",\n".join(entries) + "\n}\n", encoding="utf-8"
    )
    seconds = time.monotonic() - start
    echo(f"  {count} frames written, {width}x{height}, in {_duration_text(seconds)}")
    if hdr_times is not None:
        hdr_times["decoding_seconds"] = max(
            0.0, seconds - hdr_times["waiting_for_writers_seconds"]
        )
    return ExtractedVideo(
        frame_dir=video.frame_dir,
        frame_pattern=FRAME_PATTERN,
        frame_count=count,
        record=record,
        times={"seconds": seconds, "hdr": hdr_times},
    )


def _run_ffmpeg_hdr(
    args: list[str], on_line, frame_dir: Path, transfer: str, exposure: float
) -> tuple[int, list[str], dict]:
    """Run ffmpeg on an HDR video (`run_ffmpeg` with `on_frame`), converting
    and writing each frame it decodes as `frame_dir / FRAME_PATTERN`,
    numbered from 1, with `write_image_from_hdr`.

    The frames are converted and written on `_HDR_WRITERS` threads, at most
    `_HDR_PENDING_FRAMES` waiting, so decoding goes on while frames are
    written; every frame is written, or the first write error raised, before
    this returns. Returns `run_ffmpeg`'s exit status and lines, and the times
    of `ExtractedVideo.times["hdr"]` but `decoding_seconds`."""
    import numpy as np

    from .fileio import write_image_from_hdr

    pending: deque = deque()
    numbered = 0
    sizes: list[tuple[int, int]] = []
    times = {
        "writers": _HDR_WRITERS,
        "waiting_for_writers_seconds": 0.0,
        "conversion_seconds": 0.0,
        "encoding_seconds": 0.0,
    }

    def written() -> None:
        """Wait for the oldest pending frame, adding the wait and the
        frame's own times to `times`."""
        start = time.monotonic()
        frame = pending.popleft().result()
        times["waiting_for_writers_seconds"] += time.monotonic() - start
        times["conversion_seconds"] += frame["conversion_seconds"]
        times["encoding_seconds"] += frame["encoding_seconds"]

    def sized_on_line(line: str) -> int | None:
        size = on_line(line)
        if size is not None:
            match = _SHOWINFO_FRAME.search(line)
            sizes.append((int(match.group(4)), int(match.group(5))))
        return size

    def on_frame(data: bytes) -> None:
        nonlocal numbered
        numbered += 1
        width, height = sizes[numbered - 1]
        planes = np.frombuffer(data, dtype="<u2").reshape(3, height, width)
        while len(pending) >= _HDR_PENDING_FRAMES:
            written()
        pending.append(
            writers.submit(
                write_image_from_hdr,
                frame_dir / (FRAME_PATTERN % numbered),
                planes,
                transfer=transfer,
                exposure_stops=exposure,
            )
        )

    with ThreadPoolExecutor(max_workers=_HDR_WRITERS) as writers:
        try:
            result = run_ffmpeg(args, sized_on_line, on_frame)
            while pending:
                written()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return (*result, times)


def extract_planned(plan: ExtractionPlan, echo=click.echo) -> list[ExtractedVideo]:
    """Extract every video of `plan` in order (`extract_video_frames`)."""
    return [extract_video_frames(plan, video, echo) for video in plan.videos]
