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


# A workspace file as `json.dump(..., indent=2)` writes one, without the
# object, its keys out of sorted order.
_WITHOUT_THE_OBJECT = {
    "version": 2,
    "feature_tool": "sfmtool",
    "feature_type": "sift-sfmtool",
    "feature_options": {"max_num_features": 8192, "sigma": 1.6},
    "feature_prefix_dir": "features/sift-sfmtool-0123",
}
_NEW_WORKSPACE = {
    "out_of_core_long_side": 1536,
    "jpeg_quality": 95,
    "full_resolution_jpeg_quality": 95,
}


def _workspace_without_the_object(root: Path) -> Path:
    root.mkdir()
    (root / "images").mkdir()
    path = root / ".sfm-workspace.json"
    path.write_text(json.dumps(_WITHOUT_THE_OBJECT, indent=2), newline="\n")
    return path


@pytest.mark.parametrize(
    "option, key, value",
    [
        (None, None, None),
        ("--out-of-core-long-side", "out_of_core_long_side", 4096),
        ("--jpeg-quality", "jpeg_quality", 80),
        ("--full-resolution-jpeg-quality", "full_resolution_jpeg_quality", 100),
    ],
)
def test_enable_adds_the_object_after_the_file_s_keys(
    tmp_path, monkeypatch, option, key, value
):
    """`--enable` adds the object with the new workspace's values, or the one
    an option gives, as the file's last key, keeping the rest of the file as
    `json.dump` wrote it, and makes no mipmap."""
    root = tmp_path / "ws"
    path = _workspace_without_the_object(root)
    monkeypatch.chdir(root)
    expected = dict(_NEW_WORKSPACE)
    args = ["--enable", "images"]
    if option is not None:
        expected[key] = value
        args[1:1] = [option, str(value)]

    result = _mipmaps(*args)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Added mipmap_options to .sfm-workspace.json: photographs with a side "
        f"over {expected['out_of_core_long_side']} px are read from mipmaps, "
        f"JPEG quality {expected['jpeg_quality']} "
        f"({expected['full_resolution_jpeg_quality']} at full resolution).",
        "No mipmap was made. 'sfm ws mipmaps .' makes them now, and every "
        "reader of the photographs makes those it needs.",
    ]
    assert path.read_text() == json.dumps(
        {**_WITHOUT_THE_OBJECT, "mipmap_options": expected}, indent=2
    )
    assert sorted(p.name for p in root.iterdir()) == [".sfm-workspace.json", "images"]


@pytest.mark.parametrize(
    "object_",
    [_NEW_WORKSPACE, {"out_of_core_long_side": 0}],
    ids=["valid", "invalid"],
)
def test_enable_refuses_a_file_that_has_the_object(tmp_path, monkeypatch, object_):
    root = tmp_path / "ws"
    path = _workspace_without_the_object(root)
    text = json.dumps({**_WITHOUT_THE_OBJECT, "mipmap_options": object_}, indent=2)
    path.write_text(text)
    monkeypatch.chdir(root)
    result = _mipmaps("--enable", ".")
    assert result.exit_code == 1
    assert ".sfm-workspace.json already has mipmap_options" in result.output
    assert path.read_text() == text


@pytest.mark.parametrize(
    "value, message",
    [
        (
            "101",
            "mipmap_options.jpeg_quality is 101; it must be an integer from 1 to 100",
        ),
        (str(2**64), f"--jpeg-quality {2**64} is outside the 64-bit range."),
        (
            str(-(2**63) - 1),
            f"--jpeg-quality {-(2**63) - 1} is outside the 64-bit range.",
        ),
    ],
)
def test_enable_refuses_a_value_out_of_range_and_writes_nothing(
    tmp_path, monkeypatch, value, message
):
    root = tmp_path / "ws"
    path = _workspace_without_the_object(root)
    text = path.read_text()
    monkeypatch.chdir(root)
    result = _mipmaps("--enable", "--jpeg-quality", value, ".")
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert message in result.output
    assert path.read_text() == text


@pytest.mark.parametrize(
    "args, message",
    [
        (["--enable", "--check", "."], "Give --enable without --check or --prune."),
        (["--enable", "--prune", "."], "Give --enable without --check or --prune."),
        (["--enable", ".", "images"], "Give --enable one PATH."),
        (
            ["--jpeg-quality", "90", "."],
            "--jpeg-quality is given with --enable only.",
        ),
    ],
)
def test_enable_with_another_mode_or_its_values_without_it_is_refused(
    tmp_path, monkeypatch, args, message
):
    root = tmp_path / "ws"
    path = _workspace_without_the_object(root)
    text = path.read_text()
    monkeypatch.chdir(root)
    result = _mipmaps(*args)
    assert result.exit_code == 2
    assert message in result.output
    assert path.read_text() == text


def test_the_help_gives_each_value_s_default_from_a_new_workspace():
    from sfmtool.workspace import new_workspace_mipmap_options

    result = CliRunner().invoke(main, ["ws", "mipmaps", "--help"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    new = new_workspace_mipmap_options()
    assert f"read from mipmaps (default {new['out_of_core_long_side']})." in text
    assert f"mip level files (default {new['jpeg_quality']})." in text
    assert (
        "cannot be read by region "
        f"(default {new['full_resolution_jpeg_quality']})." in text
    )
