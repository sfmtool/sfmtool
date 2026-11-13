# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""How `sfm solve --seq-overlap` cuts the images into windows.

The images are empty `.jpg` files in a workspace, and the solve of each window
is replaced by a function that records the images it was handed, so these
tests check the windows without solving.
"""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main

from .conftest import SEOUL_BULL_DIR


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    shutil.copy(
        SEOUL_BULL_DIR / ".sfm-workspace.json", workspace / ".sfm-workspace.json"
    )
    return workspace


@pytest.fixture
def recorded_solves(monkeypatch) -> list[list[Path]]:
    calls = []

    def record(paths, workspace_dir, *args, **kwargs):
        calls.append([Path(p) for p in paths])

    monkeypatch.setattr("sfmtool._solve_driver._run_sfm", record)
    return calls


def _frames(directory: Path, numbers) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for number in numbers:
        path = directory / f"frame_{number:06d}.jpg"
        path.write_bytes(b"")
        paths.append(path)
    return paths


def _solve(*args: str):
    return CliRunner().invoke(main, ["solve", "-i", "--seq-overlap", *args])


def _windows(calls: list[list[Path]]) -> list[tuple[str, list[int]]]:
    """Each recorded window as its directory's name and its frame numbers."""
    return [(paths[0].parent.name, [int(p.stem[6:]) for p in paths]) for paths in calls]


def test_each_directory_is_its_own_sequence_with_its_own_windows(
    workspace, recorded_solves
):
    walk1 = _frames(workspace / "videos" / "walk1", range(1, 8))
    walk2 = _frames(workspace / "videos" / "walk2", range(1, 6))
    result = _solve("4,1", str(workspace / "videos"))
    assert result.exit_code == 0, result.output

    # No window mixes the two directories.
    for paths in recorded_solves:
        assert len({p.parent for p in paths}) == 1, paths
    # Every frame of both directories is solved.
    solved = {p for paths in recorded_solves for p in paths}
    assert solved == set(walk1) | set(walk2)
    assert _windows(recorded_solves) == [
        ("walk1", [1, 2, 3, 4]),
        ("walk1", [4, 5, 6, 7]),
        ("walk2", [1, 2, 3, 4]),
        ("walk2", [4, 5]),
    ]
    assert "Detected 2 sequences, each solved in its own windows:" in result.output
    assert "walk1/frame_%06d.jpg: 7 images (1-7)" in result.output
    assert "walk2/frame_%06d.jpg: 5 images (1-5)" in result.output
    assert "=== Solve 4/4: 2 images (walk2/frame_%06d.jpg 4,5) ===" in result.output


def test_one_directory_is_one_sequence(workspace, recorded_solves):
    _frames(workspace / "images", range(1, 8))
    result = _solve("4,1", str(workspace / "images"))
    assert result.exit_code == 0, result.output
    assert _windows(recorded_solves) == [
        ("images", [1, 2, 3, 4]),
        ("images", [4, 5, 6, 7]),
    ]
    assert "Detected sequence with 7 images: 1-7" in result.output
    assert "=== Solve 1/2: 4 images (1-4) ===" in result.output


def test_images_in_no_sequence_are_named_and_nothing_is_solved(
    workspace, recorded_solves
):
    videos = workspace / "videos"
    images = _frames(videos / "walk1", range(1, 8))
    # Two frames are too few to be a sequence, and a name without a number
    # is in none.
    images += _frames(videos / "walk2", range(1, 3))
    for name in ("walk1/cover.jpg", "walk2/notes_a.jpg", "walk2/notes_b.jpg"):
        (videos / name).write_bytes(b"")
        images.append(videos / name)
    images += _frames(videos / "walk3", range(1, 3))
    # The images are handed over in reverse order, so the message lists them
    # in sorted order only if the command sorts them, and names the same five
    # of the seven whatever the order of the input.
    result = _solve("4,1", *(str(p) for p in reversed(images)))
    assert result.exit_code != 0
    assert recorded_solves == []
    message = " ".join(result.output.split())
    assert "7 of the provided images are in none:" in message
    assert (
        "walk1/cover.jpg, walk2/frame_000001.jpg, walk2/frame_000002.jpg, "
        "walk2/notes_a.jpg, walk2/notes_b.jpg and 2 more." in message
    )


def test_no_numbered_sequence_is_refused(workspace, recorded_solves):
    images = workspace / "images"
    images.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (images / name).write_bytes(b"")
    result = _solve("4,1", str(images))
    assert result.exit_code != 0
    assert "No numbered sequence detected" in result.output
