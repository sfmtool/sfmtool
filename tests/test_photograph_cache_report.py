# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`photograph_cache_lines`, the lines `sfm cluster-patches`,
`sfm embed-patches`, `sfm xform` and the strips of `sfm compare` and
`sfm inspect` print after their reads, and what its recommendation names for
each workspace: `sfm ws mipmaps --enable` for one without `mipmap_options`,
and the threshold to set for one whose object leaves the photographs in core.

The cache is a stand-in that answers `stats` and `layout`, so no photograph is
read; the commands' own tests read real ones through a small budget."""

import json
from pathlib import Path
from types import SimpleNamespace

from sfmtool._photograph_cache_report import photograph_cache_lines


class _Cache:
    """A cache that decoded its photographs whole ``decodes`` times, each
    photograph in core with the size ``sizes`` gives it."""

    def __init__(self, decodes: int, sizes: dict[Path, tuple[int, int]]):
        self.decodes = decodes
        self.sizes = sizes

    def stats(self) -> dict:
        return {"decodes": self.decodes, "mipmaps_made": 0, "budget_bytes": 2**30}

    def layout(self, path):
        width, height = self.sizes[Path(path)]
        return SimpleNamespace(width=width, height=height, out_of_core=False)


def _workspace(path: Path, mipmap_options: dict | None = None) -> Path:
    config = {"version": 2}
    if mipmap_options is not None:
        config["mipmap_options"] = mipmap_options
    path.mkdir(parents=True)
    (path / ".sfm-workspace.json").write_text(json.dumps(config))
    return path


def test_no_recommendation_when_each_photograph_is_decoded_once(tmp_path):
    ws = _workspace(tmp_path / "ws")
    sizes = {ws / "a.jpg": (6000, 4000), ws / "b.jpg": (6000, 4000)}

    lines = photograph_cache_lines(_Cache(2, sizes), sizes)

    assert lines == [
        "Photographs: 2 whole decodes, 0 mipmaps made, cache budget 1.0 GiB"
    ]


def test_a_workspace_without_the_object_is_given_the_enable_command(
    tmp_path, monkeypatch
):
    """Photographs over the default threshold need only `--enable`; the path is
    shown relative to the current directory, which holds the workspace."""
    monkeypatch.chdir(tmp_path)
    ws = _workspace(tmp_path / "photos")
    sizes = {ws / "a.jpg": (6000, 4000), ws / "b.jpg": (4000, 6000)}

    lines = photograph_cache_lines(_Cache(5, sizes), sizes)

    assert lines[1] == (
        "The cache decoded 2 photographs whole 5 times, since their tiles are "
        "more than its budget. To read them from mipmaps instead, a tile at a "
        "time, run 'sfm ws mipmaps --enable photos'."
    )


def test_photographs_under_the_default_threshold_are_given_one_below_their_side(
    tmp_path, monkeypatch
):
    """`--enable`'s default, 1536, would leave a 1536 px photograph in core, so
    the command names a threshold one less than the shortest longer side."""
    monkeypatch.chdir(tmp_path)
    ws = _workspace(tmp_path / "photos")
    sizes = {ws / "a.jpg": (2048, 1536), ws / "b.jpg": (1024, 1536)}

    lines = photograph_cache_lines(_Cache(3, sizes), sizes)

    assert lines[1].endswith(
        "run 'sfm ws mipmaps --enable --out-of-core-long-side 1535 photos'."
    )


def test_a_workspace_whose_threshold_keeps_them_in_core_is_given_a_lower_one(
    tmp_path, monkeypatch
):
    """`--enable` refuses a file that has the object, so the line names the
    value to set in the file instead, with the value it holds; each workspace
    is named, in sorted order."""
    monkeypatch.chdir(tmp_path)
    without = _workspace(tmp_path / "without")
    with_object = _workspace(
        tmp_path / "with",
        {
            "out_of_core_long_side": 8192,
            "jpeg_quality": 95,
            "full_resolution_jpeg_quality": 95,
        },
    )
    sizes = {
        without / "a.jpg": (6000, 4000),
        with_object / "b.jpg": (6000, 4000),
        with_object / "c.jpg": (4032, 3024),
    }

    lines = photograph_cache_lines(_Cache(9, sizes), sizes)

    assert lines[1].endswith(
        "To read them from mipmaps instead, a tile at a time, set "
        '"out_of_core_long_side" in with/.sfm-workspace.json to 4031 or less '
        "(it is 8192) and run 'sfm ws mipmaps --enable without'."
    )


def test_photographs_in_no_workspace_are_named_in_no_step(tmp_path, monkeypatch):
    """A photograph outside every workspace has no threshold to change: the
    line names only the workspace of the others, and ends after the decodes
    when no photograph is in one."""
    monkeypatch.chdir(tmp_path)
    ws = _workspace(tmp_path / "photos")
    loose = tmp_path / "loose"
    loose.mkdir()
    sizes = {ws / "a.jpg": (6000, 4000), loose / "b.jpg": (800, 600)}

    lines = photograph_cache_lines(_Cache(5, sizes), sizes)

    assert lines[1].endswith("a tile at a time, run 'sfm ws mipmaps --enable photos'.")

    loose_only = {loose / "b.jpg": (800, 600)}
    lines = photograph_cache_lines(_Cache(5, loose_only), loose_only)

    assert lines[1] == (
        "The cache decoded 1 photographs whole 5 times, since their tiles are "
        "more than its budget."
    )
