# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm extract-frames` (specs/cli/workspace/extract-frames-command.md).

The videos are made here with ffmpeg's `testsrc2` source and its built-in
`mpeg4` encoder, a second or two at 320 x 240 or smaller, so no codec library
and no checked-in video is needed. The HDR videos are 10-bit frames of two
flat levels written from numpy with the built-in lossless `ffv1` encoder and
tagged HLG or PQ. ffmpeg comes from the test environment.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool import _video_frames
from sfmtool._video_frames import (
    RECORD_NAME,
    ffmpeg_version_number,
    run_ffmpeg,
)
from sfmtool.cli import main
from sfmtool.fileio import image_dimensions, read_image_rgb


def _invoke(args: list[str]):
    return CliRunner().invoke(main, ["extract-frames", *args], catch_exceptions=False)


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        stdin=subprocess.DEVNULL,
    )


def _testsrc(size: str, rate: int, seconds: float) -> list[str]:
    return ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}"]


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> dict[str, Path]:
    """The test videos, made once for the module."""
    root = tmp_path_factory.mktemp("videos")
    paths = {
        name: root / file
        for name, file in (
            ("plain", "plain.mp4"),
            ("small", "small.mp4"),
            ("rotated", "rotated.mp4"),
            ("vfr", "vfr.mp4"),
            ("hlg", "hlg.mkv"),
            ("pq", "pq.mkv"),
            ("hlg_full_range", "hlg_full_range.mkv"),
            ("pq_bt709", "pq_bt709.mov"),
            ("song", "song.m4a"),
        )
    }
    # 2 s at 30 fps, 60 frames, with an audio track.
    _ffmpeg(
        *_testsrc("320x240", 30, 2),
        *["-f", "lavfi", "-i", "sine=duration=2"],
        *["-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest"],
        str(paths["plain"]),
    )
    _ffmpeg(*_testsrc("100x80", 30, 1), "-c:v", "mpeg4", str(paths["small"]))
    # The same frames with a display matrix that turns them 90 degrees.
    _ffmpeg(
        "-display_rotation",
        "90",
        "-i",
        str(paths["plain"]),
        "-c",
        "copy",
        str(paths["rotated"]),
    )
    # Variable rate: 30 frames 1/30 s apart, then 30 frames 0.3 s apart.
    _ffmpeg(
        *_testsrc("160x120", 30, 2),
        "-vf",
        "setpts='if(lt(N,30),N,30+(N-30)*9)/(30*TB)'",
        *["-fps_mode", "vfr", "-c:v", "mpeg4", "-q:v", "5"],
        str(paths["vfr"]),
    )
    # HDR video: 1 s at 30 fps of 128 x 96 10-bit frames, the left half at
    # one level and the right half at another (HDR_LEVELS).
    for name, transfer in (("hlg", "arib-std-b67"), ("pq", "smpte2084")):
        _hdr_video(paths[name], transfer, HDR_LEVELS[name])
    # HDR transfer tags the conversion does not take, on the plain frames.
    _ffmpeg(
        "-i",
        str(paths["plain"]),
        *["-c", "copy", "-color_trc", "arib-std-b67", "-color_range", "pc"],
        str(paths["hlg_full_range"]),
    )
    _ffmpeg(
        "-i",
        str(paths["plain"]),
        *["-c", "copy", "-color_trc", "smpte2084", "-colorspace", "bt709"],
        *["-movflags", "+write_colr"],
        str(paths["pq_bt709"]),
    )
    # Audio with cover art: its only video stream is an attached picture.
    cover = root / "cover.jpg"
    _ffmpeg(*_testsrc("64x64", 1, 1), "-frames:v", "1", str(cover))
    _ffmpeg(
        *["-f", "lavfi", "-i", "sine=duration=0.5", "-i", str(cover)],
        *["-map", "0", "-map", "1", "-c:a", "aac", "-c:v", "copy"],
        *["-disposition:v:0", "attached_pic"],
        str(paths["song"]),
    )
    cover.unlink()
    return paths


# The 10-bit narrow-range Y' codes of the HDR test videos' two halves:
# HLG 75% (reference white) and 50%; PQ 203 and 100 cd/m².
HDR_LEVELS = {"hlg": (721, 502), "pq": (573, 509)}


def _hdr_video(path: Path, transfer: str, levels: tuple[int, int]) -> None:
    """Write a 1 s, 30 fps, 128 x 96 video of 10-bit BT.2020 narrow-range
    frames, the left half at Y' code `levels[0]`, the right at `levels[1]`,
    zero chroma, losslessly, tagged with `transfer`."""
    import numpy as np

    width, height = 128, 96
    luma = np.empty((height, width), np.uint16)
    luma[:, : width // 2], luma[:, width // 2 :] = levels
    chroma = np.full((height // 2, width // 2), 512, np.uint16)
    frame = luma.tobytes() + chroma.tobytes() * 2
    subprocess.run(
        [
            *["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"],
            *["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{width}x{height}"],
            *["-r", "30", "-i", "-", "-c:v", "ffv1", "-vf"],
            f"setparams=color_trc={transfer}:color_primaries=bt2020"
            ":colorspace=bt2020nc:range=tv",
            str(path),
        ],
        input=frame * 30,
        check=True,
    )


def _halves(frame: Path) -> tuple[list[int], list[int]]:
    """The RGB of a frame's left and right halves, each at a pixel a quarter
    of the width in from its side, on the middle row."""
    pixels = read_image_rgb(frame)
    height, width, _ = pixels.shape
    row = height // 2
    return (
        pixels[row, width // 4].tolist(),
        pixels[row, width - 1 - width // 4].tolist(),
    )


def _record(frame_dir: Path) -> dict:
    return json.loads((frame_dir / RECORD_NAME).read_text(encoding="utf-8"))


def _frame_names(frame_dir: Path) -> list[str]:
    return sorted(p.name for p in frame_dir.iterdir() if p.name != RECORD_NAME)


def _interval(t: float, rate: float) -> int:
    """The interval `[k/rate, (k+1)/rate)` a recorded time is in. The record
    rounds times to microseconds, so a time on a boundary can be written just
    below it."""
    return math.floor(t * rate + 1e-4)


def _check_numbering(frame_dir: Path, count: int) -> None:
    """The frames are numbered 1 to `count` with no gaps, and the record has
    a time for each."""
    assert _frame_names(frame_dir) == [
        f"frame_{i:06d}.jpg" for i in range(1, count + 1)
    ]
    assert len(_record(frame_dir)["frame_times_s"]) == count


def test_default_rate_numbering_and_record(videos, tmp_path):
    out = tmp_path / "out"
    result = _invoke([str(videos["plain"]), "-o", str(out)])
    assert result.exit_code == 0, result.output
    frame_dir = out / "plain"
    # 2 s at the default 10 fps: one frame in each tenth of a second.
    _check_numbering(frame_dir, 20)
    record = _record(frame_dir)
    times = record["frame_times_s"]
    assert [_interval(t, 10) for t in times] == list(range(20))
    assert record == {
        "version": 2,
        "video": str(videos["plain"].resolve()),
        "video_stream": 0,
        "codec": "mpeg4",
        "video_size": [320, 240],
        "rotation_degrees": 0,
        "color_transfer": None,
        "hdr_conversion": None,
        "fps": 10,
        "max_frame_size": 1920,
        "frame_size": [320, 240],
        "frame_pattern": "frame_%06d.jpg",
        "frame_times_s": times,
    }
    assert image_dimensions(frame_dir / "frame_000020.jpg") == (320, 240)
    assert "plain.mp4: mpeg4 320x240, 30.00 fps, 2.0 s" in result.output
    assert "20 frames written, 320x240" in result.output
    assert f"Wrote 20 frames from 1 video to {out}" in result.output


@pytest.mark.parametrize(
    ("fps", "count", "kept_all"),
    [("full", 60, False), ("15", 30, False), ("2.5", 5, False), ("30", 60, True)],
)
def test_frame_rate(videos, tmp_path, fps, count, kept_all):
    out = tmp_path / "out"
    result = _invoke([str(videos["plain"]), "-o", str(out), "--fps", fps])
    assert result.exit_code == 0, result.output
    _check_numbering(out / "plain", count)
    record = _record(out / "plain")
    assert record["fps"] == (fps if fps == "full" else float(fps))
    if fps != "full":
        rate = float(fps)
        intervals = [_interval(t, rate) for t in record["frame_times_s"]]
        assert intervals == sorted(set(intervals))
    assert ("so every frame is kept" in result.output) == kept_all


@pytest.mark.parametrize(
    ("video", "size", "frame_size", "rotation"),
    [
        ("plain", "160", [160, 120], 0),
        ("small", "160", [100, 80], 0),
        ("plain", "full", [320, 240], 0),
        ("rotated", "160", [120, 160], 90),
        ("rotated", "full", [240, 320], 90),
    ],
)
def test_frame_size_and_rotation(videos, tmp_path, video, size, frame_size, rotation):
    out = tmp_path / "out"
    args = [str(videos[video]), "-o", str(out), "--max-frame-size", size]
    result = _invoke([*args, "--fps", "2"])
    assert result.exit_code == 0, result.output
    frame_dir = out / videos[video].stem
    record = _record(frame_dir)
    assert record["frame_size"] == frame_size
    assert record["rotation_degrees"] == rotation
    assert record["max_frame_size"] == (size if size == "full" else int(size))
    for name in _frame_names(frame_dir):
        assert list(image_dimensions(frame_dir / name)) == frame_size


def test_variable_rate_is_sampled_without_duplicates(videos, tmp_path):
    out = tmp_path / "out"
    result = _invoke([str(videos["vfr"]), "-o", str(out)])
    assert result.exit_code == 0, result.output
    frame_dir = out / "vfr"
    # The first second fills its 10 intervals; each of the 30 frames 0.3 s
    # apart after it has an interval to itself. A constant-rate resampling
    # would fill the gaps with copies, about 90 frames after the first second.
    _check_numbering(frame_dir, 40)
    assert "(variable)" in result.output
    contents = [(frame_dir / name).read_bytes() for name in _frame_names(frame_dir)]
    assert len(set(contents)) == len(contents)
    times = _record(frame_dir)["frame_times_s"]
    expected = [i / 10 for i in range(10)] + [1 + 0.3 * j for j in range(30)]
    assert times == pytest.approx(expected, abs=0.04)


def test_two_videos_and_existing_directories(videos, tmp_path):
    """Two videos in one run, beside an earlier video's directory and an
    empty directory named for one of them."""
    out = tmp_path / "out"
    (out / "earlier").mkdir(parents=True)
    (out / "earlier" / "frame_000001.jpg").write_bytes(b"kept")
    (out / "small").mkdir()
    result = _invoke(
        [str(videos["plain"]), str(videos["small"]), "-o", str(out), "--fps", "5"]
    )
    assert result.exit_code == 0, result.output
    _check_numbering(out / "plain", 10)
    _check_numbering(out / "small", 5)
    assert (out / "earlier" / "frame_000001.jpg").read_bytes() == b"kept"
    assert f"Wrote 15 frames from 2 videos to {out}" in result.output


def _tree(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


def test_every_refused_input_is_named_and_nothing_is_created(videos, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    insv = inputs / "VID_0001.INSV"
    insv.write_bytes(b"")
    image = inputs / "photo.JPG"
    image.write_bytes(b"")
    not_video = inputs / "notes.dat"
    not_video.write_text('{"not": "a video"}\n')
    folder = inputs / "folder"
    folder.mkdir()
    walk = inputs / "walk.mp4"
    shutil.copy(videos["small"], walk)
    walk_upper = inputs / "WALK.mov"
    shutil.copy(videos["small"], walk_upper)
    busy = inputs / "busy.mp4"
    shutil.copy(videos["small"], busy)
    taken = inputs / "taken.mp4"
    shutil.copy(videos["small"], taken)
    out = tmp_path / "out"
    (out / "busy").mkdir(parents=True)
    (out / "busy" / "frame_000001.jpg").write_bytes(b"")
    (out / "taken").write_bytes(b"")
    before = _tree(out)

    refused = {
        insv: "an Insta360 .insv file holds two fisheye lenses; extract it with "
        "`sfm insv2rig`",
        image: "an image, not a video",
        not_video: "ffprobe cannot read it as a video",
        folder: "a directory, not a video",
        inputs / "missing.mp4": "no such file",
        inputs / "star*name.mp4": "its name has a '*'",
        videos["song"]: "it has no video stream other than cover art",
        videos["hlg_full_range"]: "HDR video (HLG) with color_range pc; the HDR "
        "conversion takes BT.2020 non-constant-luminance Y'CbCr (bt2020nc), "
        "BT.2020 primaries and narrow (tv) range",
        videos["pq_bt709"]: "HDR video (PQ) with color_space bt709;",
        busy: f"its frame directory {out / 'busy'} is not empty",
        taken: f"its frame directory {out / 'taken'} exists and is not a directory",
    }
    args = [*map(str, refused), str(walk), str(walk_upper), str(videos["small"])]
    result = _invoke([*args, "-o", str(out)])
    assert result.exit_code == 1, result.output
    for path, reason in refused.items():
        assert f"  {path}: {reason}" in result.output
    # walk.mp4 and WALK.mov are refused as a pair; small.mp4, with nothing
    # wrong, is not named.
    assert (
        f"  {walk}, {walk_upper}: these would share one frame directory"
        in result.output
    )
    assert f"{videos['small']}:" not in result.output
    # One more for the pair, and one for star*name.mp4, which is also missing.
    assert f"{len(refused) + 2} problems found, so nothing was created" in result.output
    assert _tree(out) == before


# The sRGB codes of each half of the HDR test videos, worked out by hand from
# the recipe (specs/core/camera/hdr-to-srgb.md): HLG 75% 229.02 and 50%
# 136.89, one stop down 186.91 and 99.04; PQ 203 cd/m² 229.15 and 100 cd/m²
# 185.67. A conversion that read the signal as SDR would give 191 and 128 for
# HLG, 148 and 130 for PQ.
HDR_CODES = {("hlg", 0): (229, 137), ("hlg", -1): (187, 99), ("pq", 0): (229, 186)}


@pytest.mark.parametrize("name", ["hlg", "pq"])
def test_hdr_video_is_converted_to_srgb(videos, tmp_path, name):
    out = tmp_path / "out"
    result = _invoke([str(videos[name]), "-o", str(out)])
    assert result.exit_code == 0, result.output
    frame_dir = out / name
    _check_numbering(frame_dir, 10)
    record = _record(frame_dir)
    transfer = {"hlg": "HLG", "pq": "PQ"}[name]
    assert record["color_transfer"] == {"hlg": "arib-std-b67", "pq": "smpte2084"}[name]
    assert record["hdr_conversion"] == {
        "transfer": transfer,
        "recipe": "sfmtool-hdr-to-srgb",
        "recipe_version": 1,
        "exposure_stops": 0,
    }
    assert record["frame_size"] == [128, 96]
    assert [_interval(t, 10) for t in record["frame_times_s"]] == list(range(10))
    left, right = HDR_CODES[name, 0]
    for frame in _frame_names(frame_dir):
        (l_rgb, r_rgb) = _halves(frame_dir / frame)
        assert l_rgb == pytest.approx([left] * 3, abs=1), frame
        assert r_rgb == pytest.approx([right] * 3, abs=1), frame
    assert f"ffv1 128x96, HDR {transfer}, 30.00 fps" in result.output
    assert (
        f"converted from {transfer} to sRGB by sfmtool-hdr-to-srgb 1, exposure +0 "
        "stops" in result.output
    )


def test_hdr_exposure_applies_to_every_hdr_frame_and_no_sdr_frame(videos, tmp_path):
    """One run with an HDR and an SDR video, scaled down: the exposure darkens
    every HDR frame, is recorded for it, and leaves the SDR video's record
    without a conversion."""
    out = tmp_path / "out"
    args = [str(videos["hlg"]), str(videos["small"]), "-o", str(out)]
    result = _invoke([*args, "--hdr-exposure", "-1", "--max-frame-size", "64"])
    assert result.exit_code == 0, result.output
    record = _record(out / "hlg")
    assert record["hdr_conversion"]["exposure_stops"] == -1
    assert record["frame_size"] == [64, 48]
    left, right = HDR_CODES["hlg", -1]
    for frame in _frame_names(out / "hlg"):
        assert list(image_dimensions(out / "hlg" / frame)) == [64, 48]
        (l_rgb, r_rgb) = _halves(out / "hlg" / frame)
        assert l_rgb == pytest.approx([left] * 3, abs=1), frame
        assert r_rgb == pytest.approx([right] * 3, abs=1), frame
    assert _record(out / "small")["hdr_conversion"] is None
    assert "exposure -1 stops" in result.output


def test_extraction_times_split_an_hdr_video_s_work(videos, tmp_path):
    """An SDR video's extraction is ffmpeg's alone; an HDR video's is split
    into the calling thread's decoding and waiting, which make up its wall
    time, and the writer threads' conversion and encoding."""
    plan = _video_frames.plan_extraction(
        [videos["hlg"], videos["small"]],
        tmp_path / "out",
        fps=None,
        max_frame_size=64,
        command="test",
    )
    hlg, small = _video_frames.extract_planned(plan, lambda text: None)
    assert small.times["hdr"] is None and small.times["seconds"] > 0
    hdr = hlg.times["hdr"]
    assert hdr["writers"] == 4
    assert hdr["conversion_seconds"] > 0 and hdr["encoding_seconds"] > 0
    assert hdr["decoding_seconds"] > 0 and hdr["waiting_for_writers_seconds"] >= 0
    assert hdr["decoding_seconds"] + hdr["waiting_for_writers_seconds"] == (
        pytest.approx(hlg.times["seconds"])
    )


def test_hdr_exposure_with_no_hdr_video_is_refused(videos, tmp_path):
    out = tmp_path / "out"
    args = [str(videos["small"]), "-o", str(out), "--hdr-exposure", "0.5"]
    result = _invoke(args)
    assert result.exit_code == 1, result.output
    assert (
        "1 problem found, so nothing was created:\n  --hdr-exposure sets the "
        "exposure of HDR video frames, and none of the videos read is HDR"
        in result.output
    )
    assert not out.exists()


def test_the_same_video_twice_is_refused(videos, tmp_path):
    out = tmp_path / "out"
    result = _invoke([str(videos["small"]), str(videos["small"]), "-o", str(out)])
    assert result.exit_code == 1
    assert "these would share one frame directory" in result.output
    assert not out.exists()


def test_no_ffmpeg_on_path(videos, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    out = tmp_path / "out"
    result = _invoke([str(videos["small"]), "-o", str(out)])
    assert result.exit_code == 1
    assert (
        "sfm extract-frames needs ffmpeg and ffprobe on PATH, and found no ffmpeg "
        "and no ffprobe. Install ffmpeg 5.1 or newer"
    ) in result.output
    assert not out.exists()


def _fake_program(folder: Path, name: str, first_line: str) -> None:
    if sys.platform == "win32":
        (folder / f"{name}.bat").write_text(f"@echo {first_line}\r\n")
    else:
        path = folder / name
        path.write_text(f"#!/bin/sh\necho '{first_line}'\n")
        path.chmod(0o755)


def test_old_ffmpeg_is_refused(videos, tmp_path, monkeypatch):
    fake = tmp_path / "fake"
    fake.mkdir()
    line = "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021"
    _fake_program(fake, "ffmpeg", line)
    _fake_program(fake, "ffprobe", line.replace("ffmpeg", "ffprobe"))
    monkeypatch.setenv("PATH", str(fake))
    out = tmp_path / "out"
    result = _invoke([str(videos["small"]), "-o", str(out)])
    assert result.exit_code == 1
    assert "needs ffmpeg 5.1 or newer" in result.output
    assert "is version 4.4.2-0ubuntu0.22.04.1" in result.output
    assert not out.exists()


@pytest.mark.parametrize(
    ("output", "version"),
    [
        ("ffmpeg version 7.1.1 Copyright (c) 2000-2025", (7, 1)),
        ("ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright", (9, 0)),
        ("ffmpeg version n5.1.2 Copyright", (5, 1)),
        ("ffmpeg version 5.0.1 Copyright", (5, 0)),
        # Development builds: the libavutil of 5.1 or newer, and older.
        (
            "ffmpeg version N-118-gabc Copyright\nlibavutil      57. 28.100 / 57. 28.100",
            (5, 1),
        ),
        (
            "ffmpeg version 2022-01-01-git-x Copyright\nlibavutil      57. 17.100 / 57",
            (0, 0),
        ),
        ("something else", None),
    ],
)
def test_ffmpeg_version_number(output, version):
    assert ffmpeg_version_number(output) == version


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--fps", "0"),
        ("--fps", "-1"),
        ("--fps", "fast"),
        ("--fps", "nan"),
        ("--max-frame-size", "63"),
        ("--max-frame-size", "1.5e3"),
        ("--hdr-exposure", "nan"),
        ("--hdr-exposure", "16.5"),
        ("--hdr-exposure", "-inf"),
        ("--hdr-exposure", "bright"),
    ],
)
def test_bad_option_values_are_usage_errors(videos, tmp_path, option, value):
    out = tmp_path / "out"
    result = _invoke([str(videos["small"]), "-o", str(out), option, value])
    assert result.exit_code == 2
    assert f"Invalid value for '{option}'" in result.output
    assert not out.exists()


def test_ctrl_c_stops_ffmpeg(monkeypatch):
    """A Ctrl+C while ffmpeg runs terminates it and is re-raised."""
    started = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(subprocess, "Popen", recording_popen)

    def interrupt(line):
        raise KeyboardInterrupt

    # Ten minutes of frames, which ffmpeg would take far longer than this
    # test to produce.
    args = ["ffmpeg", "-nostdin", "-f", "lavfi", "-i", "testsrc2=duration=600"]
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_ffmpeg([*args, "-f", "null", "-"], interrupt)
        assert started[0].poll() is not None
        assert time.monotonic() - start < 20
    finally:
        if started and started[0].poll() is None:
            started[0].kill()
            started[0].wait()


def test_ctrl_c_during_extraction_leaves_no_record(videos, tmp_path, monkeypatch):
    """Interrupted from the progress line, the extraction stops, keeps the
    frames written so far and writes no record."""
    monkeypatch.setattr(_video_frames, "PROGRESS_INTERVAL_S", 0.0)
    plan = _video_frames.plan_extraction(
        [videos["plain"]],
        tmp_path / "out",
        fps=None,
        max_frame_size=None,
        command="test",
    )
    progress_lines = []

    def echo(text):
        if " of about 60 frames (" in text:
            progress_lines.append(text)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _video_frames.extract_planned(plan, echo)
    assert progress_lines and progress_lines[0].startswith("  1 of about 60 frames")
    assert not (tmp_path / "out" / "plain" / RECORD_NAME).exists()


def test_machine_report_finds_the_same_ffmpeg():
    from sfmtool._machine_report import ffmpeg_facts

    plan_programs = _video_frames.find_ffmpeg("test")
    facts = ffmpeg_facts()
    assert facts["path"] == plan_programs.ffmpeg
    assert facts["version"] == plan_programs.version
    assert os.path.isabs(facts["path"])


def test_a_run_that_prints_nothing_is_repeated(monkeypatch):
    """A run that exits with status 0 having printed nothing, as some ffmpeg
    builds now and then do, is run again, up to three runs in all."""
    outputs = ["", "", "ffmpeg version 7.1 Copyright\n"]

    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, outputs.pop(0), "")

    monkeypatch.setattr(subprocess, "run", run)
    result = _video_frames.run_for_output(["ffmpeg", "-version"])
    assert result.stdout == "ffmpeg version 7.1 Copyright\n"
    assert outputs == []
