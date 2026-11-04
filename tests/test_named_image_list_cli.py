# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`-r/--range` and `--named-image-list` on the eight commands that take them.

The image-directory commands (`sift`, `match`, `motion`, `solve`) run in a
small workspace of six seoul_bull images with a lists file; their expensive
work is replaced by a function that records the images it was handed, except
for one real `sfm sift --extract`. The reconstruction commands (`analyze
--metrics`, `panorama`, `to-colmap-bin`, `to-nerfstudio`) run on the
seoul_bull ground truth, whose workspace holds no image files, or on the
SIFT-backed seoul_bull workspace where they need the images.
"""

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_colmap_binary
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.workspace import NAMED_IMAGE_LISTS_FILE_NAME

from .conftest import SEOUL_BULL_DIR

RANGE_OR_NAME_SENTENCE = (
    "A --range value with a letter in it is read as a named image list; "
    "a range contains only digits, '-', ':' and ','."
)

_LISTS = {
    "bull_odd": [{"pattern": "images/seoul_bull_sculpture_%02d.jpg", "range": "1-5:2"}],
    "bull_all": [{"pattern": "images/seoul_bull_sculpture_%02d.jpg"}],
    "everything": [{"pattern": "images/**"}],
    "hero": [
        {"pattern": "images/seoul_bull_sculpture_04.jpg"},
        {"pattern": "images/seoul_bull_sculpture_02.jpg"},
    ],
    "only_notes": [{"pattern": "images/*.txt"}],
}


def _write_lists(workspace: Path, lists: dict) -> None:
    (workspace / NAMED_IMAGE_LISTS_FILE_NAME).write_text(
        json.dumps({"version": 1, "named_image_lists": lists})
    )


def _invoke(args: list[str]):
    return CliRunner().invoke(main, args)


@pytest.fixture
def list_workspace(tmp_path: Path, monkeypatch) -> Path:
    """Six seoul_bull images under `images/`, two non-image files beside them,
    a lists file, and the current directory set to a subdirectory of it.

    The workspace file is seoul_bull's version 1 file, upgraded to version 2
    in the copy so that extraction writes into it."""
    workspace = tmp_path / "ws"
    images = workspace / "images"
    images.mkdir(parents=True)
    for i in range(1, 7):
        name = f"seoul_bull_sculpture_{i:02d}.jpg"
        shutil.copy(SEOUL_BULL_DIR / name, images / name)
    (images / "notes.txt").write_text("not an image")
    (images / "seoul_bull_sculpture_03.json").write_text("{}")
    shutil.copy(
        SEOUL_BULL_DIR / ".sfm-workspace.json", workspace / ".sfm-workspace.json"
    )
    upgraded = CliRunner().invoke(main, ["ws", "upgrade", str(workspace)])
    assert upgraded.exit_code == 0, upgraded.output
    _write_lists(workspace, _LISTS)
    subdir = workspace / "elsewhere"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    return workspace


def _names(paths) -> list[str]:
    return [Path(p).name for p in paths]


@pytest.fixture
def recorded_sift(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(
        "sfmtool._commands.sift.image_files_to_sift_files",
        lambda filenames, **kwargs: calls.append(list(filenames)),
    )
    return calls


@pytest.fixture
def recorded_match(monkeypatch) -> list:
    calls = []

    def record(paths, workspace_dir, **kwargs):
        calls.append((list(paths), Path(workspace_dir)))

    monkeypatch.setattr("sfmtool.feature_match._run._run_matching", record)
    return calls


@pytest.fixture
def recorded_solve(monkeypatch) -> list:
    calls = []

    def record(paths, workspace_dir, *args, **kwargs):
        calls.append((list(paths), workspace_dir, kwargs.get("range_expr")))

    monkeypatch.setattr("sfmtool._solve_driver._run_sfm", record)
    return calls


@pytest.fixture
def recorded_motion(monkeypatch) -> list:
    calls = []

    def record(paths, **kwargs):
        calls.append(list(paths))
        return []

    monkeypatch.setattr("sfmtool.motion.image_sequence.analyze_image_sequence", record)
    return calls


# ── Name or range ────────────────────────────────────────────────────────


def test_a_value_with_a_letter_is_a_list_name(list_workspace, recorded_sift):
    result = _invoke(["sift", "--extract", "-r", "bull_odd"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_sift[0]) == [
        "seoul_bull_sculpture_01.jpg",
        "seoul_bull_sculpture_03.jpg",
        "seoul_bull_sculpture_05.jpg",
    ]
    assert all(Path(p).is_absolute() for p in recorded_sift[0])
    assert "Given named image list 'bull_odd', expanded to 3 filename(s)" in (
        result.output
    )


def test_a_value_without_a_letter_is_a_range(list_workspace, recorded_sift):
    result = _invoke(["sift", "--extract", "-r", "2-3", str(list_workspace)])
    assert result.exit_code == 0, result.output
    assert sorted(_names(recorded_sift[0])) == [
        "seoul_bull_sculpture_02.jpg",
        "seoul_bull_sculpture_03.jpg",
    ]


def test_a_value_with_a_letter_that_is_not_a_name_says_what_a_name_may_contain(
    list_workspace,
):
    result = _invoke(["sift", "--extract", "-r", "walk 1"])
    assert result.exit_code != 0
    assert "'walk 1' is not a valid named image list name" in result.output
    assert "one or more of the ASCII letters, the digits" in result.output
    result = _invoke(["sift", "--extract", "--named-image-list", "123"])
    assert result.exit_code != 0
    assert "--named-image-list '123' is not a valid" in result.output


def test_the_long_form_names_a_list(list_workspace, recorded_sift):
    result = _invoke(["sift", "--extract", "--named-image-list", "hero"])
    assert result.exit_code == 0, result.output
    # The list's order, not the directory's.
    assert _names(recorded_sift[0]) == [
        "seoul_bull_sculpture_04.jpg",
        "seoul_bull_sculpture_02.jpg",
    ]


@pytest.mark.parametrize(
    "args",
    [["-r", "bull_odd", "-r", "1-1"], ["--named-image-list", "bull_odd", "-r", "1"]],
)
def test_one_name_and_one_range_narrow_the_list(list_workspace, recorded_sift, args):
    result = _invoke(["sift", "--extract", *args])
    assert result.exit_code == 0, result.output
    assert _names(recorded_sift[0]) == ["seoul_bull_sculpture_01.jpg"]


def test_two_names_or_two_ranges_are_errors(list_workspace):
    result = _invoke(["sift", "--extract", "-r", "hero", "--named-image-list", "x"])
    assert result.exit_code != 0
    assert "at most one named image list" in result.output
    result = _invoke(["sift", "--extract", "-r", "1-2", "-r", "4"])
    assert result.exit_code != 0
    assert "at most one range" in result.output


def test_a_narrowing_range_that_leaves_nothing_says_so(list_workspace):
    result = _invoke(["sift", "--extract", "-r", "bull_odd", "-r", "2"])
    assert result.exit_code != 0
    assert "the range '2' left no members of named image list 'bull_odd'" in (
        result.output
    )


def test_narrowing_an_entry_without_a_frame_field_is_an_error(list_workspace):
    result = _invoke(["sift", "--extract", "-r", "hero", "-r", "2"])
    assert result.exit_code != 0
    assert "cannot be narrowed by a range" in result.output


# ── Finding the list ─────────────────────────────────────────────────────


def test_an_unknown_name_names_the_file_and_the_lists_it_defines(list_workspace):
    result = _invoke(["sift", "--extract", "-r", "walk"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert NAMED_IMAGE_LISTS_FILE_NAME in output
    assert "no named image list is called 'walk'" in output
    assert "'bull_odd', 'bull_all', 'everything', 'hero', 'only_notes'" in output
    assert RANGE_OR_NAME_SENTENCE in output


def test_a_workspace_without_a_lists_file_has_no_lists(list_workspace):
    (list_workspace / NAMED_IMAGE_LISTS_FILE_NAME).unlink()
    result = _invoke(["match", "--exhaustive", "-r", "hero"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "has no named image lists" in output
    assert NAMED_IMAGE_LISTS_FILE_NAME in output
    assert RANGE_OR_NAME_SENTENCE in output


def test_a_list_name_with_no_workspace_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for args in (
        ["sift", "--extract", "-r", "hero"],
        ["match", "--exhaustive", "-r", "hero"],
        ["motion", "-r", "hero"],
        ["solve", "-i", "-r", "hero"],
    ):
        result = _invoke(args)
        assert result.exit_code != 0, args
        output = " ".join(result.output.split())
        assert "No workspace was found" in output, args
        assert "no named image lists are defined" in output, args


@pytest.mark.parametrize(
    "command",
    [
        ["sift", "--extract"],
        ["match", "--exhaustive"],
        ["motion"],
        ["solve", "-i"],
    ],
)
def test_a_list_together_with_paths_is_an_error(list_workspace, command):
    result = _invoke([*command, "-r", "hero", str(list_workspace / "images")])
    assert result.exit_code != 0
    assert "A named image list replaces the image paths" in result.output


# ── Image files only ─────────────────────────────────────────────────────


def test_members_that_are_not_image_files_are_skipped(list_workspace, recorded_sift):
    result = _invoke(["sift", "--extract", "-r", "everything"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_sift[0]) == [
        f"seoul_bull_sculpture_{i:02d}.jpg" for i in range(1, 7)
    ]


def test_a_list_with_no_image_files_is_an_error_naming_it(list_workspace):
    result = _invoke(["sift", "--extract", "-r", "only_notes"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "named image list 'only_notes' is an image file" in output


# ── Each image-directory command ─────────────────────────────────────────


def test_match_takes_a_list_and_its_workspace(list_workspace, recorded_match):
    result = _invoke(["match", "--exhaustive", "-r", "bull_all", "-r", "2-4"])
    assert result.exit_code == 0, result.output
    paths, workspace_dir = recorded_match[0]
    assert _names(paths) == [f"seoul_bull_sculpture_{i:02d}.jpg" for i in (2, 3, 4)]
    assert workspace_dir.resolve() == list_workspace.resolve()


def test_match_range_on_paths_is_unchanged(list_workspace, recorded_match):
    result = _invoke(["match", "--exhaustive", "-r", "5-6", str(list_workspace)])
    assert result.exit_code == 0, result.output
    paths, _ = recorded_match[0]
    assert sorted(_names(paths)) == [
        "seoul_bull_sculpture_05.jpg",
        "seoul_bull_sculpture_06.jpg",
    ]


def test_match_derive_pairs_rejects_a_list(list_workspace):
    Path("x.matches").write_bytes(b"")
    result = _invoke(["match", "--derive-pairs", "x.matches", "-r", "hero"])
    assert result.exit_code != 0
    assert "--range" in result.output
    result = _invoke(
        ["match", "--derive-pairs", "x.matches", "--named-image-list", "hero"]
    )
    assert result.exit_code != 0
    assert "--named-image-list" in result.output


def test_solve_takes_a_list(list_workspace, recorded_solve):
    result = _invoke(["solve", "-i", "--named-image-list", "bull_odd"])
    assert result.exit_code == 0, result.output
    paths, workspace_dir, _ = recorded_solve[0]
    assert _names(paths) == [f"seoul_bull_sculpture_{i:02d}.jpg" for i in (1, 3, 5)]
    assert Path(workspace_dir).resolve() == list_workspace.resolve()


def test_solve_range_on_paths_is_unchanged(list_workspace, recorded_solve):
    result = _invoke(["solve", "-i", "-r", "1,6", str(list_workspace / "images")])
    assert result.exit_code == 0, result.output
    paths, _, _ = recorded_solve[0]
    assert sorted(_names(paths)) == [
        "seoul_bull_sculpture_01.jpg",
        "seoul_bull_sculpture_06.jpg",
    ]


def test_solve_with_neither_paths_nor_a_list_is_unchanged(list_workspace):
    result = _invoke(["solve", "-i"])
    assert result.exit_code != 0
    assert "Must provide image paths or a .matches file." in result.output


def test_motion_takes_a_list(list_workspace, recorded_motion):
    result = _invoke(["motion", "-r", "bull_all", "-r", "3-6"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_motion[0]) == [
        f"seoul_bull_sculpture_{i:02d}.jpg" for i in (3, 4, 5, 6)
    ]


def test_motion_range_on_paths_is_unchanged(list_workspace, recorded_motion):
    result = _invoke(["motion", "-r", "1-4", str(list_workspace / "images")])
    assert result.exit_code == 0, result.output
    assert _names(recorded_motion[0]) == [
        f"seoul_bull_sculpture_{i:02d}.jpg" for i in (1, 2, 3, 4)
    ]


def test_sift_extracts_the_images_of_a_list(list_workspace):
    """One real extraction: a list reaches the extractor and writes its .sift."""
    result = _invoke(["sift", "--extract", "-r", "hero"])
    assert result.exit_code == 0, result.output
    sift_files = sorted(p.name for p in list_workspace.rglob("*.sift"))
    assert sift_files == [
        "seoul_bull_sculpture_02.jpg.sift",
        "seoul_bull_sculpture_04.jpg.sift",
    ]


# ── Reconstruction commands ──────────────────────────────────────────────


def _gt_lists(sfmr: Path) -> None:
    """Lists for the ground truth, whose image names sit at the workspace root."""
    _write_lists(
        sfmr.parent,
        {
            "first_three": [
                {"pattern": "seoul_bull_sculpture_%02d.jpg", "range": "1-3"}
            ],
            # Members the reconstruction does not hold are ignored.
            "picked": [
                {"pattern": "seoul_bull_sculpture_09.jpg"},
                {"pattern": "seoul_bull_sculpture_99.jpg"},
                {"pattern": "seoul_bull_sculpture_05.jpg"},
            ],
            "absent": [{"pattern": "photos/*.jpg"}],
        },
    )


def test_to_colmap_bin_uses_a_list_without_the_image_files(
    seoul_bull_ground_truth_sfmr, tmp_path
):
    sfmr = seoul_bull_ground_truth_sfmr
    _gt_lists(sfmr)
    for image in sfmr.parent.glob("*.jpg"):
        image.unlink()
    out = tmp_path / "colmap"
    result = _invoke(["to-colmap-bin", str(sfmr), str(out), "-r", "picked"])
    assert result.exit_code == 0, result.output
    assert "Applied named image list 'picked': keeping 2 of 17 images" in (
        result.output
    )
    data = read_colmap_binary(out)
    assert sorted(data["image_names"]) == [
        "seoul_bull_sculpture_05.jpg",
        "seoul_bull_sculpture_09.jpg",
    ]


def test_to_colmap_bin_narrows_and_filters_points(
    seoul_bull_ground_truth_sfmr, tmp_path
):
    sfmr = seoul_bull_ground_truth_sfmr
    _gt_lists(sfmr)
    result = _invoke(
        [
            "to-colmap-bin",
            str(sfmr),
            str(tmp_path / "colmap"),
            "--named-image-list",
            "first_three",
            "-r",
            "2-3",
            "--filter-points",
        ]
    )
    assert result.exit_code == 0, result.output
    assert (
        "Applied named image list 'first_three' narrowed by the range '2-3': "
        "keeping 2 of 17 images"
    ) in result.output


def test_to_colmap_bin_errors_name_the_list(seoul_bull_ground_truth_sfmr, tmp_path):
    sfmr = seoul_bull_ground_truth_sfmr
    out = str(tmp_path / "colmap")
    result = _invoke(["to-colmap-bin", str(sfmr), out, "-r", "picked"])
    assert result.exit_code != 0
    assert "has no named image lists" in " ".join(result.output.split())
    _gt_lists(sfmr)
    result = _invoke(["to-colmap-bin", str(sfmr), out, "-r", "absent"])
    assert result.exit_code != 0
    assert "named image list 'absent' matches no image" in result.output
    result = _invoke(["to-colmap-bin", str(sfmr), out, "-r", "nope"])
    assert result.exit_code != 0
    assert RANGE_OR_NAME_SENTENCE in " ".join(result.output.split())


def test_to_colmap_bin_range_is_unchanged(seoul_bull_ground_truth_sfmr, tmp_path):
    result = _invoke(
        [
            "to-colmap-bin",
            str(seoul_bull_ground_truth_sfmr),
            str(tmp_path / "colmap"),
            "-r",
            "10-12",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Applied range filter '10-12': keeping 3 of 17 images" in result.output
    result = _invoke(
        [
            "to-colmap-bin",
            str(seoul_bull_ground_truth_sfmr),
            str(tmp_path / "colmap2"),
            "--filter-points",
        ]
    )
    assert result.exit_code != 0
    assert "--filter-points requires --range" in result.output


def test_analyze_metrics_uses_a_list(seoul_bull_ground_truth_sfmr):
    sfmr = seoul_bull_ground_truth_sfmr
    _gt_lists(sfmr)
    result = _invoke(["analyze", "--metrics", str(sfmr), "-r", "picked"])
    assert result.exit_code == 0, result.output
    assert "Selected by named image list 'picked' (2 of 17 images)" in result.output
    assert "seoul_bull_sculpture_09.jpg" in result.output
    assert "seoul_bull_sculpture_01.jpg" not in result.output


def test_analyze_range_is_unchanged(seoul_bull_ground_truth_sfmr):
    sfmr = seoul_bull_ground_truth_sfmr
    result = _invoke(["analyze", "--metrics", str(sfmr), "-r", "1-4"])
    assert result.exit_code == 0, result.output
    assert "Range filter: 1-4 (4 of 17 images)" in result.output
    result = _invoke(["analyze", "--coviz", str(sfmr), "-r", "1-4"])
    assert result.exit_code != 0
    assert "--range can only be used with --metrics." in result.output
    result = _invoke(["analyze", "--coviz", str(sfmr), "--named-image-list", "a"])
    assert result.exit_code != 0
    assert "--named-image-list can only be used with --metrics." in result.output


def test_panorama_uses_a_list(seoul_bull_workspace, tmp_path):
    workspace = seoul_bull_workspace.parent
    _write_lists(
        workspace,
        {
            "pair": [
                {
                    "pattern": "test_17_image/seoul_bull_sculpture_%02d.jpg",
                    "range": "1-2",
                }
            ]
        },
    )
    out = tmp_path / "pano.png"
    result = _invoke(
        [
            "panorama",
            str(seoul_bull_workspace),
            "-o",
            str(out),
            "--equirect-width",
            "64",
            "-r",
            "pair",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "The named image list 'pair' holds 2 of its images" in result.output
    assert "Selected 2 of 17 images" in result.output
    assert out.exists()


def test_reconstruction_commands_find_the_list_in_the_reconstructions_workspace(
    seoul_bull_ground_truth_sfmr, tmp_path, monkeypatch
):
    """The current directory plays no part for a reconstruction command."""
    _gt_lists(seoul_bull_ground_truth_sfmr)
    monkeypatch.chdir(tmp_path)
    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    assert Path(recon.workspace_dir) == seoul_bull_ground_truth_sfmr.parent
    result = _invoke(
        ["analyze", "--metrics", str(seoul_bull_ground_truth_sfmr), "-r", "first_three"]
    )
    assert result.exit_code == 0, result.output
    assert "(3 of 17 images)" in result.output


# ── Help ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        "sift",
        "match",
        "motion",
        "solve",
        "analyze",
        "panorama",
        "to-colmap-bin",
        "to-nerfstudio",
    ],
)
def test_help_explains_both_forms(command):
    result = _invoke([command, "--help"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "-r, --range RANGE|NAME" in output
    assert "--named-image-list NAME" in output
    assert "A value with a letter in it is instead the name of a named image list" in (
        output
    )
