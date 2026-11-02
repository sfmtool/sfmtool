# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.workspace.NamedImageLists` on a temporary workspace with real
files: frames named like `frames/walk1_0001.jpg` and a rig laid out as
`rig/cam0/frame_00001.jpg`, `rig/cam1/frame_00001.jpg`, …"""

import json

import pytest

from sfmtool import ProgressCounter
from sfmtool.reconstruction import RangeExpr
from sfmtool.workspace import (
    NAMED_IMAGE_LISTS_FILE_NAME,
    NamedImageLists,
    is_named_image_list_name,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / ".sfm-workspace.json").write_text("{}")
    for i in range(1, 21):
        _touch(tmp_path / "frames" / f"walk1_{i:04d}.jpg")
    for i in range(1, 6):
        for cam in ("cam0", "cam1"):
            _touch(tmp_path / "rig" / cam / f"frame_{i:05d}.jpg")
    _touch(tmp_path / "photos" / "DSC_0042.JPG")
    _touch(tmp_path / "photos" / "DSC_0057.JPG")
    return tmp_path


def _write_lists(root, lists, version=1):
    text = json.dumps({"version": version, "named_image_lists": lists}, indent=2)
    (root / NAMED_IMAGE_LISTS_FILE_NAME).write_text(text)


_LISTS = {
    "walk1_sharp": [{"pattern": "frames/walk1_%04d.jpg", "range": "1-3,18-20"}],
    "rig_sharp": [{"pattern": "rig/*/frame_%05d.jpg", "range": "2-4:2"}],
    "hero_shots": [
        {"pattern": "photos/DSC_0057.JPG"},
        {"pattern": "photos/DSC_0042.JPG"},
    ],
}


def test_read_returns_none_without_a_lists_file(workspace):
    assert NamedImageLists.read(workspace) is None
    _write_lists(workspace, {})
    lists = NamedImageLists.read(workspace)
    assert lists is not None
    assert len(lists) == 0
    assert lists.names == []


def test_read_needs_a_workspace_root(tmp_path):
    _write_lists(tmp_path, {})
    with pytest.raises(ValueError, match="not a workspace root"):
        NamedImageLists.read(tmp_path)


def test_names_and_paths(workspace):
    _write_lists(workspace, _LISTS)
    lists = NamedImageLists.read(str(workspace))
    assert lists.names == ["walk1_sharp", "rig_sharp", "hero_shots"]
    assert "rig_sharp" in lists
    assert "Rig_sharp" not in lists
    assert str(lists.path) == str(workspace / NAMED_IMAGE_LISTS_FILE_NAME)
    assert str(lists.workspace_dir) == str(workspace)


def test_resolve_against_the_filesystem(workspace):
    _write_lists(workspace, _LISTS)
    lists = NamedImageLists.read(workspace)

    walk = lists.resolve("walk1_sharp")
    assert [m.path for m in walk] == [
        f"frames/walk1_{i:04d}.jpg" for i in (1, 2, 3, 18, 19, 20)
    ]
    assert [m.frame for m in walk] == [1, 2, 3, 18, 19, 20]
    assert {m.entry for m in walk} == {0}

    rig = lists.resolve("rig_sharp")
    assert [m.path for m in rig] == [
        "rig/cam0/frame_00002.jpg",
        "rig/cam1/frame_00002.jpg",
        "rig/cam0/frame_00004.jpg",
        "rig/cam1/frame_00004.jpg",
    ]

    hero = lists.resolve("hero_shots")
    assert [(m.path, m.entry, m.frame) for m in hero] == [
        ("photos/DSC_0057.JPG", 0, None),
        ("photos/DSC_0042.JPG", 1, None),
    ]


def test_resolve_narrowed_by_a_range(workspace):
    _write_lists(workspace, _LISTS)
    lists = NamedImageLists.read(workspace)
    narrowed = lists.resolve("walk1_sharp", narrow=RangeExpr("2-19"))
    assert [m.frame for m in narrowed] == [2, 3, 18, 19]
    with pytest.raises(ValueError, match="entry 1 cannot be narrowed"):
        lists.resolve("hero_shots", narrow=RangeExpr("1"))
    with pytest.raises(
        ValueError,
        match="the range '500' left no members of named image list 'walk1_sharp'",
    ):
        lists.resolve("walk1_sharp", narrow=RangeExpr("500"))


def test_resolve_in_names(workspace):
    _write_lists(workspace, _LISTS)
    lists = NamedImageLists.read(workspace)
    # Names a reconstruction records; none of these files need exist.
    names = [
        "rig/cam1/frame_00004.jpg",
        "rig/cam0/frame_00004.jpg",
        "rig/cam0/frame_00003.jpg",
        "rig/cam9/frame_00002.jpg",
    ]
    members = lists.resolve_in_names("rig_sharp", names)
    assert [m.path for m in members] == [
        "rig/cam9/frame_00002.jpg",
        "rig/cam0/frame_00004.jpg",
        "rig/cam1/frame_00004.jpg",
    ]
    members = lists.resolve_in_names("rig_sharp", names, narrow=RangeExpr("4"))
    assert [m.frame for m in members] == [4, 4]
    with pytest.raises(ValueError, match="matches no image"):
        lists.resolve_in_names("walk1_sharp", names)


def test_matching_is_case_sensitive(workspace):
    _write_lists(workspace, {"lower": [{"pattern": "photos/dsc_0042.jpg"}]})
    lists = NamedImageLists.read(workspace)
    with pytest.raises(ValueError, match="matches no image"):
        lists.resolve("lower")


def test_a_file_two_entries_match_is_an_error(workspace):
    _write_lists(
        workspace,
        {"a": [{"pattern": "photos/DSC_0042.JPG"}, {"pattern": "photos/*.JPG"}]},
    )
    lists = NamedImageLists.read(workspace)
    with pytest.raises(ValueError, match="'photos/DSC_0042.JPG' by both entry 1"):
        lists.resolve("a")


def test_an_unknown_name_lists_the_defined_ones(workspace):
    _write_lists(workspace, _LISTS)
    lists = NamedImageLists.read(workspace)
    with pytest.raises(ValueError, match="'walk1_sharp', 'rig_sharp', 'hero_shots'"):
        lists.resolve("walk2")


def test_validation_errors_name_the_file_and_every_problem(workspace):
    _write_lists(
        workspace,
        {
            "bad name": [{"pattern": "a.jpg"}],
            "a": [{"pattern": "photos/DSC_0042.JPG", "range": "1"}],
        },
    )
    with pytest.raises(ValueError) as info:
        NamedImageLists.read(workspace)
    lines = str(info.value).splitlines()
    assert len(lines) == 2
    path = str(workspace / NAMED_IMAGE_LISTS_FILE_NAME)
    assert all(line.startswith(f"{path}: ") for line in lines)
    assert "'bad name' is not a valid list name" in lines[0]
    assert "has no frame field" in lines[1]


def test_an_unknown_version(workspace):
    _write_lists(workspace, {"bad name": 3}, version=2)
    path = workspace / NAMED_IMAGE_LISTS_FILE_NAME
    with pytest.raises(ValueError) as info:
        NamedImageLists.read(workspace)
    assert str(info.value) == (
        f"{path}: named image lists version 2 is not supported; "
        "this sfmtool reads version 1"
    )


def test_invalid_json(workspace):
    (workspace / NAMED_IMAGE_LISTS_FILE_NAME).write_text('{"version": 1,\n  oops}')
    with pytest.raises(ValueError, match="not valid JSON at line 2, column 3"):
        NamedImageLists.read(workspace)


def test_resolve_counts_entries_read(workspace):
    _write_lists(workspace, _LISTS)
    lists = NamedImageLists.read(workspace)
    counter = ProgressCounter()
    lists.resolve("walk1_sharp", progress=counter)
    # The root's five entries (the workspace and lists files, and three
    # directories) and the twenty frames; `rig` and `photos` are not listed.
    assert counter.value == 25


def test_is_named_image_list_name():
    assert is_named_image_list_name("walk1_sharp")
    assert not is_named_image_list_name("1-10")
    assert not is_named_image_list_name("walk 1")
