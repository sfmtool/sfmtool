# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm ws mipmaps` end to end through Click, on a temporary workspace whose
`mipmap_options` sets a threshold of 64 px: a check before and after a
build, a build, and a prune, with what each prints and writes.

What counts as an orphan, and the guards on what a prune deletes, are tested
in Rust (`crates/sfmtool-core/src/camera/mipmap/prune/tests.rs`); these tests
check what the command adds."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main


def _workspace(root: Path, with_mipmap_options: bool = True) -> tuple[Path, Path]:
    """A workspace at `root` with a 300 x 200 photograph, out of core under
    the object, and a 60 x 40 one in `images/`, and a text file there."""
    root.mkdir()
    config = {"version": 2}
    if with_mipmap_options:
        config["mipmap_options"] = {
            "out_of_core_long_side": 64,
            "jpeg_quality": 95,
            "full_resolution_jpeg_quality": 95,
        }
    (root / ".sfm-workspace.json").write_text(json.dumps(config))
    images = root / "images"
    images.mkdir()
    rng = np.random.default_rng(5)
    for name, (width, height) in (("big.png", (300, 200)), ("small.png", (60, 40))):
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(images / name), pixels)
    (images / "notes.txt").write_text("not an image")
    return images / "big.png", images / "small.png"


def _snapshot(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _mipmaps(*args):
    return CliRunner().invoke(main, ["ws", "mipmaps", *map(str, args)])


def test_ws_help_lists_mipmaps():
    result = CliRunner().invoke(main, ["ws", "--help"])
    assert result.exit_code == 0, result.output
    [line] = [
        line
        for line in result.output.splitlines()
        if line.strip().startswith("mipmaps")
    ]
    assert "Make the mipmaps of the large photographs" in line


def test_check_build_check_and_prune(tmp_path, monkeypatch):
    """A check before the build reports the missing mipmap and writes
    nothing, exiting with status 1; the build makes it; a check then finds it
    current and exits with 0; and once the photograph is deleted a prune
    deletes its files and keeps every other file."""
    root = tmp_path / "ws"
    big, small = _workspace(root)
    monkeypatch.chdir(root)
    before = _snapshot(root)

    result = _mipmaps("--check", "images")
    assert result.exit_code == 1, result.output
    assert result.output.splitlines() == [
        "Images: 2, 1 out of core and 1 in core",
        "Current mipmaps: 0 / 1 out-of-core image(s)",
        "Missing mipmaps: 1",
        "  images/big.png",
    ]
    assert _snapshot(root) == before

    result = _mipmaps("images")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Images: 2, 1 out of core and 1 in core",
        "New mipmaps: 1 / 1 out-of-core image(s)",
    ]
    level_files = sorted((root / ".sfm-mipmaps" / "images").iterdir())
    assert [p.name for p in level_files] == [
        f"big.png-mip{level}-q95.jpg" for level in (0, 1)
    ]

    result = _mipmaps("--check", ".")
    assert result.exit_code == 0, result.output
    assert (
        result.output.splitlines()[1] == "Current mipmaps: 1 / 1 out-of-core image(s)"
    )

    big.unlink()
    stray = root / ".sfm-mipmaps" / "images" / "readme.txt"
    stray.write_text("not a mip level file")
    result = _mipmaps("--check", ".")
    assert result.exit_code == 1, result.output
    assert "Orphan files --prune would delete: 2" in result.output
    assert all(p.is_file() for p in level_files)

    result = _mipmaps("--prune", ".")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Deleted 2 orphan file(s):",
        *(
            f"  .sfm-mipmaps/images/{p.name}: its photograph images/big.png no "
            "longer exists"
            for p in level_files
        ),
        "Kept 0 mip level file(s).",
    ]
    assert not any(p.exists() for p in level_files)
    assert stray.is_file() and small.is_file()
    assert (root / ".sfm-mipmaps" / "CACHEDIR.TAG").is_file()


def test_without_the_object_nothing_is_made_and_a_prune_deletes_every_mip_level_file(
    tmp_path, monkeypatch
):
    root = tmp_path / "ws"
    _workspace(root)
    monkeypatch.chdir(root)
    assert _mipmaps("images").exit_code == 0
    config = json.loads((root / ".sfm-workspace.json").read_text())
    del config["mipmap_options"]
    (root / ".sfm-workspace.json").write_text(json.dumps(config))

    result = _mipmaps("images")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Images: 2, 0 out of core and 2 in core",
        "New mipmaps: 0 / 0 out-of-core image(s)",
    ]

    result = _mipmaps("--prune", "images")
    assert result.exit_code == 0, result.output
    assert "Deleted 2 orphan file(s):" in result.output
    assert "the workspace's file has no mipmap_options" in result.output
    assert list((root / ".sfm-mipmaps" / "images").iterdir()) == []


def test_check_and_prune_together_are_refused(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    _workspace(root)
    monkeypatch.chdir(root)
    result = _mipmaps("--check", "--prune", ".")
    assert result.exit_code == 2
    assert "Give --check or --prune, not both." in result.output


@pytest.mark.parametrize("flag", [[], ["--check"]])
def test_a_path_with_no_image_is_refused(tmp_path, monkeypatch, flag):
    root = tmp_path / "ws"
    _workspace(root)
    monkeypatch.chdir(root)
    empty = root / "empty"
    empty.mkdir()
    result = _mipmaps(*flag, "empty")
    assert result.exit_code == 2
    assert "There were no image files under the paths." in result.output
