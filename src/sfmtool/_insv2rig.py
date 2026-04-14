# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Insta360 .insv dual-fisheye video to rig image extraction.

Extracts frames from Insta360 .insv video files and splits them into
left and right fisheye image sequences for rig-aware SfM.
"""

import json
import subprocess
from pathlib import Path

import click
import cv2

_FISHEYE_SENSOR_NAMES = ["fisheye_left", "fisheye_right"]


def _format_file_size(size_bytes: int) -> str:
    """Format a file size in human-readable form."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} bytes"


def _format_duration(seconds: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _probe_video_streams(insv_path: Path) -> list[dict]:
    """Probe an .insv file and return its video stream metadata."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v",
            str(insv_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (exit code {result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout)["streams"]


def _print_video_info(insv_path: Path, streams: list[dict]) -> None:
    """Print video metadata summary."""
    file_size = insv_path.stat().st_size
    click.echo(f"File size: {_format_file_size(file_size)}")
    click.echo(f"Video streams: {len(streams)}")
    for i, s in enumerate(streams):
        w = s.get("width", "?")
        h = s.get("height", "?")
        parts = [f"  Stream {i}: {w}x{h}"]
        r_frame_rate = s.get("r_frame_rate", "")
        if r_frame_rate and "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            try:
                fps = int(num) / int(den)
                parts.append(f"{fps:.1f} fps")
            except (ValueError, ZeroDivisionError):
                pass
        duration = s.get("duration")
        nb_frames = s.get("nb_frames")
        if duration:
            try:
                parts.append(_format_duration(float(duration)))
            except ValueError:
                pass
        if nb_frames:
            parts.append(f"{nb_frames} frames")
        click.echo(", ".join(parts))


def _run_ffmpeg(args: list[str], label: str) -> None:
    """Run ffmpeg, streaming stderr for progress. Raises on failure."""
    click.echo(f"  {label}...")
    process = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    last_line = ""
    for line in process.stderr:
        last_line = line
        # ffmpeg writes progress as carriage-return lines starting with "frame="
        stripped = line.strip()
        if stripped.startswith("frame="):
            click.echo(f"\r  {stripped}", nl=False)
    # Clear the progress line
    if last_line.strip().startswith("frame="):
        click.echo()
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({label}, exit code {returncode})")


def _extract_dual_stream(
    insv_path: Path,
    sensor_dirs: list[Path],
) -> int:
    """Extract frames from a dual-stream .insv file (two separate video streams)."""
    for stream_idx, sensor_dir in enumerate(sensor_dirs):
        _run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(insv_path),
                "-map",
                f"0:v:{stream_idx}",
                "-qscale:v",
                "2",
                str(sensor_dir / "frame_%06d.jpg"),
            ],
            label=f"Extracting stream {stream_idx} ({sensor_dir.name})",
        )

    left_frames = sorted(sensor_dirs[0].glob("frame_*.jpg"))
    right_frames = sorted(sensor_dirs[1].glob("frame_*.jpg"))
    if not left_frames:
        raise ValueError(f"No frames extracted from {insv_path}")
    if len(left_frames) != len(right_frames):
        raise ValueError(
            f"Stream frame count mismatch: {len(left_frames)} vs {len(right_frames)}"
        )
    return len(left_frames)


def _extract_side_by_side(
    insv_path: Path,
    sensor_dirs: list[Path],
) -> int:
    """Extract frames from a side-by-side .insv file (single 2:1 stream)."""
    temp_dir = sensor_dirs[0].parent / "_temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(insv_path),
                "-qscale:v",
                "2",
                str(temp_dir / "frame_%06d.jpg"),
            ],
            label="Extracting frames",
        )

        frame_files = sorted(temp_dir.glob("frame_*.jpg"))
        if not frame_files:
            raise ValueError(f"No frames extracted from {insv_path}")

        first_frame = cv2.imread(
            str(frame_files[0]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if first_frame is None:
            raise ValueError(f"Failed to read extracted frame: {frame_files[0]}")

        h, w = first_frame.shape[:2]
        if w != 2 * h:
            raise ValueError(
                f"Expected 2:1 aspect ratio (dual fisheye side by side), "
                f"got {w}x{h}. This .insv file may be a single-lens capture "
                f"(e.g., Insta360 5.7K mode) which is not supported."
            )

        total = len(frame_files)
        click.echo(f"  Splitting {total} frames into left/right...")
        for i, frame_file in enumerate(frame_files):
            if (i + 1) % 100 == 0 or (i + 1) == total:
                click.echo(f"\r  Splitting frames: {i + 1}/{total}", nl=False)

            frame = cv2.imread(
                str(frame_file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            if frame is None:
                raise ValueError(f"Failed to read extracted frame: {frame_file}")

            h, w = frame.shape[:2]
            mid = w // 2
            left = frame[:, :mid]
            right = frame[:, mid:]

            frame_name = frame_file.name
            cv2.imwrite(str(sensor_dirs[0] / frame_name), left)
            cv2.imwrite(str(sensor_dirs[1] / frame_name), right)
        click.echo()

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    return len(frame_files)


def extract_insv_frames(
    insv_path: Path,
    output_dir: Path,
) -> tuple[int, list[str]]:
    """Extract dual-fisheye frames from an Insta360 .insv video file.

    Supports two .insv layouts:
    - **Dual-stream**: Two separate video streams (one per lens). Detected
      when ffprobe finds 2+ video streams.
    - **Side-by-side**: A single 2:1 video stream with both fisheye images
      concatenated horizontally.

    Args:
        insv_path: Path to the .insv video file.
        output_dir: Output directory. Fisheye images are written to
            output_dir/fisheye_left/ and output_dir/fisheye_right/.

    Returns:
        Tuple of (num_frames, sensor_names).
    """
    insv_path = Path(insv_path)
    output_dir = Path(output_dir)

    sensor_names = _FISHEYE_SENSOR_NAMES

    sensor_dirs = []
    for name in sensor_names:
        d = output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        sensor_dirs.append(d)

    video_streams = _probe_video_streams(insv_path)
    _print_video_info(insv_path, video_streams)

    if len(video_streams) >= 2:
        num_frames = _extract_dual_stream(insv_path, sensor_dirs)
    elif len(video_streams) == 1:
        num_frames = _extract_side_by_side(insv_path, sensor_dirs)
    else:
        raise ValueError(f"No video streams found in {insv_path}")

    return num_frames, sensor_names
