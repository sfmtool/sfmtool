# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`-r/--range` and `--named-image-list` on the eight commands that take them.

The image-directory commands (`sift`, `match`, `motion`, `solve`) run in a
small workspace of six seoul_bull images with a lists file; their expensive
work is replaced by a function that records the images it was handed, except
for one real `sfm sift --extract`. `sfm solve` from a `.matches` file runs on
six images of which the file holds four, and sets up its real COLMAP database,
which a stand-in for the global mapper reads in place of solving. The
reconstruction commands (`analyze --metrics`, `panorama`, `to-colmap-bin`,
`to-nerfstudio`) run on the
seoul_bull ground truth, whose workspace holds no image files, or on the
SIFT-backed seoul_bull workspace where they need the images.
"""

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_colmap_binary, read_matches
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.workspace import NAMED_IMAGE_LISTS_FILE_NAME

from .conftest import SEOUL_BULL_DIR

RANGE_EXAMPLES = (
    "'7', '1-100', '1-10,20,30-40' or '1-100:10' (every 10th number from 1 to 100)"
)


def _read_as_a_list(value: str) -> str:
    """The note an error about a list given by `-r value` ends with."""
    return (
        f"'{value}' was read as a named image list, not a range, because it has "
        "a letter in it; a range holds only digits, '-', ':' and ',', as in "
        f"{RANGE_EXAMPLES}."
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
        calls.append((list(paths), workspace_dir, kwargs.get("image_filter")))

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
    assert _read_as_a_list("walk 1") in " ".join(result.output.split())
    # A value with a slice was meant as a list, so its error has no note.
    result = _invoke(["sift", "--extract", "-r", "bull_all[5]"])
    assert result.exit_code != 0
    assert "--range 'bull_all[5]' is not a valid slice" in result.output
    assert "not a range" not in result.output
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


# ── A mistyped range ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        ["pick-images"],
        ["sift", "--extract"],
        ["solve", "-i"],
        ["match", "--exhaustive"],
    ],
)
def test_a_span_with_its_step_after_a_letter_suggests_the_colon_form(
    list_workspace, command
):
    (list_workspace / NAMED_IMAGE_LISTS_FILE_NAME).unlink()
    images = str(list_workspace / "images")
    for args in (["-r", "1-10389x10"], ["-r", "1-10389x10", images]):
        result = _invoke([*command, *args])
        assert result.exit_code != 0, args
        output = " ".join(result.output.split())
        assert _read_as_a_list("1-10389x10") in output, args
        assert (
            "A span with a step is written START-END:STEP; did you mean '1-10389:10'?"
        ) in output, args


@pytest.mark.parametrize(
    "value",
    ["1-10389 step 10", "1-10389 by 10", "1-10389every10", "1-10389X10"],
)
def test_other_spellings_of_a_step_get_the_same_suggestion(list_workspace, value):
    result = _invoke(["sift", "--extract", "-r", value])
    assert result.exit_code != 0
    assert "did you mean '1-10389:10'?" in " ".join(result.output.split())


def test_a_range_that_does_not_parse_shows_the_syntax(list_workspace):
    images = str(list_workspace / "images")
    result = _invoke(["sift", "--extract", "-r", "1-10389/10", images])
    assert result.exit_code == 2
    output = " ".join(result.output.split())
    assert "--range '1-10389/10': Expected ':' or ',' in '1-10389/10'." in output
    assert f"a span may take a step after a ':', as in {RANGE_EXAMPLES}." in output
    assert "Did you mean '1-10389:10'?" in output
    result = _invoke(["sift", "--extract", "-r", "5-", images])
    assert result.exit_code == 2
    output = " ".join(result.output.split())
    assert f"as in {RANGE_EXAMPLES}." in output
    assert "Did you mean" not in output


def test_a_list_named_like_a_mistyped_range_is_still_a_list(
    list_workspace, recorded_sift
):
    _write_lists(
        list_workspace,
        {
            "1-5x2": [
                {"pattern": "images/seoul_bull_sculpture_%02d.jpg", "range": "4-5"}
            ]
        },
    )
    result = _invoke(["sift", "--extract", "-r", "1-5x2"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_sift[0]) == [
        "seoul_bull_sculpture_04.jpg",
        "seoul_bull_sculpture_05.jpg",
    ]


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


def test_a_list_that_misses_files_by_the_padding_of_their_number_says_so(
    list_workspace,
):
    # `%d` matches 1 to 6 written unpadded, not the files' `01` to `06`; the
    # list names them rather than resolving to fewer images.
    _write_lists(
        list_workspace,
        {"loose": [{"pattern": "images/seoul_bull_sculpture_%d.jpg"}]},
    )
    result = _invoke(["sift", "--extract", "-r", "loose"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert (
        "list 'loose' entry 1 leaves out 'images/seoul_bull_sculpture_01.jpg' "
        "and 5 other files" in output
    )
    assert "apart from the zero-padding of the number" in output


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
    assert _read_as_a_list("walk") in output
    assert "did you mean" not in output
    # --named-image-list asks for a list, so its error says nothing of ranges.
    result = _invoke(["sift", "--extract", "--named-image-list", "walk"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "no named image list is called 'walk'" in output
    assert "not a range" not in output
    # A value with a slice was meant as a list, so its error has no note either.
    result = _invoke(["sift", "--extract", "-r", "walk[:3]"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "no named image list is called 'walk'" in output
    assert "not a range" not in output


def test_a_workspace_without_a_lists_file_has_no_lists(list_workspace):
    (list_workspace / NAMED_IMAGE_LISTS_FILE_NAME).unlink()
    result = _invoke(["match", "--exhaustive", "-r", "hero"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "has no named image lists" in output
    assert NAMED_IMAGE_LISTS_FILE_NAME in output
    assert _read_as_a_list("hero") in output


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
        assert _read_as_a_list("hero") in output, args


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
    assert _read_as_a_list("hero") in " ".join(result.output.split())


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


# ── solve from a .matches file ──────────────────────────────────────────


@pytest.fixture(scope="module")
def matches_workspace(tmp_path_factory) -> Path:
    """Six seoul_bull images with `.sift` files of at most 512 features, a
    `pairs.matches` of the first four only, and the lists of `_LISTS`.

    Shared by the tests below, which read it and write nothing into it: each
    solve sets up its COLMAP database in a temporary directory of its own.
    """
    workspace = tmp_path_factory.mktemp("matches_ws")
    images = workspace / "images"
    images.mkdir()
    for i in range(1, 7):
        name = f"seoul_bull_sculpture_{i:02d}.jpg"
        shutil.copy(SEOUL_BULL_DIR / name, images / name)
    for args in (
        ["ws", "init", "--max-features", "512", str(workspace)],
        ["sift", "--extract", str(images)],
        [
            "match",
            "--exhaustive",
            "-r",
            "1-4",
            "-o",
            str(workspace / "pairs.matches"),
            str(images),
        ],
    ):
        result = _invoke(args)
        assert result.exit_code == 0, result.output
    _write_lists(workspace, _LISTS)
    return workspace


@pytest.fixture
def recorded_mapper_input(monkeypatch, tmp_path) -> list:
    """The images and the two-view geometry pairs, as pairs of image names, of
    the database the global mapper is handed, in place of running it; the
    solve then fails as one that registered nothing does. The current
    directory is outside the workspace, so a list is found from the .matches
    file, not from it."""
    import pycolmap

    calls = []

    def record(db_path, *args, **kwargs):
        with pycolmap.Database.open(db_path) as db:
            images = {img.image_id: Path(img.name).name for img in db.read_all_images()}
            pair_ids, _ = db.read_two_view_geometries()
        pairs = {
            frozenset(images[i] for i in pycolmap.pair_id_to_image_pair(pair_id))
            for pair_id in pair_ids
        }
        calls.append((sorted(images.values()), pairs))
        return {}

    monkeypatch.setattr(pycolmap, "global_mapping", record)
    monkeypatch.chdir(tmp_path)
    return calls


def _solve_matches(workspace: Path, *args: str):
    return _invoke(["solve", "-g", str(workspace / "pairs.matches"), *args])


def _bull(*numbers: int) -> list[str]:
    return [f"seoul_bull_sculpture_{i:02d}.jpg" for i in numbers]


def _file_pairs_among(workspace: Path, names: list[str]) -> set[frozenset]:
    """The image pairs of `pairs.matches` whose images are both in `names`."""
    data = read_matches(workspace / "pairs.matches")
    file_names = [Path(n).name for n in data["image_names"]]
    pairs = {
        frozenset((file_names[i], file_names[j])) for i, j in data["image_index_pairs"]
    }
    return {pair for pair in pairs if pair <= set(names)}


def _assert_mapper_got(result, calls, workspace, numbers) -> None:
    # The mapper was reached, so the stand-in's empty result is the failure.
    assert "Global mapping failed" in result.output, result.output
    assert len(calls) == 1
    names, pairs = calls[0]
    assert names == _bull(*numbers)
    # Exactly the file's pairs between kept images stay, between the right
    # images: none involves a dropped image or lands on the wrong one.
    assert pairs, "no verified pair reached the database"
    assert pairs == _file_pairs_among(workspace, names)


@pytest.mark.parametrize(
    "args, numbers, label",
    [
        (["-r", "hero"], (2, 4), "named image list 'hero'"),
        (
            ["--named-image-list", "bull_all"],
            (1, 2, 3, 4),
            "named image list 'bull_all'",
        ),
        (
            ["-r", "bull_all", "-r", "2-3,5"],
            (2, 3),
            "named image list 'bull_all' narrowed by the range '2-3,5'",
        ),
        (["-r", "2-3,5"], (2, 3), "--range 2-3,5"),
    ],
)
def test_solve_keeps_the_matches_images_in_a_list_or_range(
    matches_workspace, recorded_mapper_input, args, numbers, label
):
    result = _solve_matches(matches_workspace, *args)
    _assert_mapper_got(result, recorded_mapper_input, matches_workspace, numbers)
    assert f"Images: {len(numbers)} (filtered from 4 via {label})" in result.output


@pytest.mark.parametrize(
    "value, numbers, ignored",
    [
        ("bull_all[:3]", (1, 2, 3), None),
        # Positions in the list, not among the file's images: [-4:-1] is
        # images 3-5, of which the file holds 3 and 4. Among the file's
        # images it would be 1-3.
        ("bull_all[-4:-1]", (3, 4), "1 of the 3 images"),
    ],
)
def test_solve_slices_the_list_then_keeps_the_matches_images(
    matches_workspace, recorded_mapper_input, value, numbers, ignored
):
    result = _solve_matches(matches_workspace, "-r", value)
    _assert_mapper_got(result, recorded_mapper_input, matches_workspace, numbers)
    output = " ".join(result.output.split())
    if ignored is None:
        assert "are not in" not in output
    else:
        assert (
            f"{ignored} of named image list 'bull_all' sliced by [-4:-1] are not in"
            in (output)
        )


def test_solve_with_a_list_none_of_whose_images_the_matches_holds_names_both(
    matches_workspace, recorded_mapper_input
):
    result = _solve_matches(matches_workspace, "-r", "bull_all[-2:]")
    assert result.exit_code == 1, result.output
    assert recorded_mapper_input == []
    output = " ".join(result.output.split())
    assert NAMED_IMAGE_LISTS_FILE_NAME in output
    assert (
        "none of the 2 images of named image list 'bull_all' sliced by [-2:] is "
        "among the 4 images of"
    ) in output
    assert "pairs.matches" in output


def test_solve_refuses_a_list_with_image_paths_beside_a_matches_file(
    list_workspace, recorded_solve
):
    Path("x.matches").write_bytes(b"")
    images = str(list_workspace / "images")
    result = _invoke(["solve", "-g", "x.matches", images, "-r", "hero"])
    assert result.exit_code == 2, result.output
    assert "give image paths or a list, not both" in " ".join(result.output.split())
    assert recorded_solve == []


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


# ── Slices ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value, frames",
    [
        ("bull_all[:2]", [1, 2]),
        ("bull_all[-2:]", [5, 6]),
        ("bull_all[1::2]", [2, 4, 6]),
        ("bull_all[::-2]", [6, 4, 2]),
        ("bull_all[:30]", [1, 2, 3, 4, 5, 6]),
        ("bull_all[:]", [1, 2, 3, 4, 5, 6]),
        # An entry without a frame field takes a slice, which narrowing refuses.
        ("hero[::-1]", [2, 4]),
    ],
)
def test_a_slice_takes_part_of_the_list_by_position(
    list_workspace, recorded_sift, value, frames
):
    result = _invoke(["sift", "--extract", "-r", value])
    assert result.exit_code == 0, result.output
    # In the slice's order.
    assert _names(recorded_sift[0]) == [
        f"seoul_bull_sculpture_{i:02d}.jpg" for i in frames
    ]


def test_a_slice_indexes_the_image_files_after_the_extension_test(
    list_workspace, recorded_sift
):
    """`images/**` also matches notes.txt, which sorts first, and a .json; the
    slice counts only the images."""
    result = _invoke(["sift", "--extract", "-r", "everything[:1]"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_sift[0]) == ["seoul_bull_sculpture_01.jpg"]
    result = _invoke(["sift", "--extract", "-r", "everything[2:4]"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_sift[1]) == [
        "seoul_bull_sculpture_03.jpg",
        "seoul_bull_sculpture_04.jpg",
    ]


def test_a_range_narrows_the_list_before_the_slice(list_workspace, recorded_sift):
    result = _invoke(["sift", "--extract", "-r", "bull_all[1:]", "-r", "2-4"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_sift[0]) == [
        "seoul_bull_sculpture_03.jpg",
        "seoul_bull_sculpture_04.jpg",
    ]
    assert (
        "Given named image list 'bull_all' narrowed by the range '2-4' and sliced "
        "by [1:], expanded to 2 filename(s)"
    ) in result.output


def test_the_long_form_takes_a_slice(list_workspace, recorded_solve):
    result = _invoke(["solve", "-i", "--named-image-list", "bull_all[-2:]"])
    assert result.exit_code == 0, result.output
    paths, workspace_dir, _ = recorded_solve[0]
    assert _names(paths) == [
        "seoul_bull_sculpture_05.jpg",
        "seoul_bull_sculpture_06.jpg",
    ]
    assert Path(workspace_dir).resolve() == list_workspace.resolve()


def test_match_and_motion_take_a_slice(list_workspace, recorded_match, recorded_motion):
    result = _invoke(["match", "--exhaustive", "-r", "bull_all[::3]"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_match[0][0]) == [
        "seoul_bull_sculpture_01.jpg",
        "seoul_bull_sculpture_04.jpg",
    ]
    result = _invoke(["motion", "-r", "bull_all[2:5]"])
    assert result.exit_code == 0, result.output
    assert _names(recorded_motion[0]) == [
        f"seoul_bull_sculpture_{i:02d}.jpg" for i in (3, 4, 5)
    ]


@pytest.mark.parametrize(
    "args, message",
    [
        (["-r", "bull_all[5]"], "--range 'bull_all[5]' is not a valid slice"),
        (["-r", "bull_all[5]"], "[5:6] is the one image at index 5"),
        (["-r", "bull_all[a:]"], "the start 'a' is not an integer"),
        (["-r", "bull_all[::0]"], "--range 'bull_all[::0]': slice step cannot be zero"),
        (["-r", "bull_all[:3][::2]"], "'[::2]' follows the ']'"),
        (["-r", "10-20[::2]"], "'10-20' is a range, not a name"),
        (["-r", "[:3]"], "no name comes before the '['"),
        (
            ["--named-image-list", "bull_all[1:2:3:4]"],
            "--named-image-list 'bull_all[1:2:3:4]' is not a valid slice",
        ),
    ],
)
def test_a_malformed_slice_is_a_usage_error_before_any_file_is_read(
    tmp_path, monkeypatch, args, message
):
    # No workspace here: the error comes before the workspace is looked for.
    monkeypatch.chdir(tmp_path)
    result = _invoke(["sift", "--extract", *args])
    assert result.exit_code == 2, result.output
    assert message in " ".join(result.output.split())


def test_a_slice_that_keeps_nothing_names_the_list(list_workspace):
    result = _invoke(["sift", "--extract", "-r", "bull_all[40:]"])
    assert result.exit_code == 1, result.output
    output = " ".join(result.output.split())
    assert NAMED_IMAGE_LISTS_FILE_NAME in output
    assert (
        "the slice [40:] of named image list 'bull_all' leaves none of its 6 images"
    ) in output
    result = _invoke(["sift", "--extract", "-r", "bull_all[3:1]", "-r", "1-5"])
    assert result.exit_code == 1, result.output
    assert (
        "of named image list 'bull_all' narrowed by the range '1-5' leaves none of "
        "its 5 images"
    ) in " ".join(result.output.split())


def test_two_names_with_slices_are_an_error_quoting_both(list_workspace):
    result = _invoke(["sift", "--extract", "-r", "hero[:1]", "-r", "bull_all[::2]"])
    assert result.exit_code != 0
    assert "got 'hero[:1]', 'bull_all[::2]'" in result.output


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


def test_to_colmap_bin_slices_the_members_the_reconstruction_holds(
    seoul_bull_ground_truth_sfmr, tmp_path
):
    """`picked` is 09, 99, 05 in the list's order; the reconstruction holds 09
    and 05, so `[1:]` is 05, not 99."""
    sfmr = seoul_bull_ground_truth_sfmr
    _gt_lists(sfmr)
    out = tmp_path / "colmap"
    result = _invoke(["to-colmap-bin", str(sfmr), str(out), "-r", "picked[1:]"])
    assert result.exit_code == 0, result.output
    assert (
        "Applied named image list 'picked' sliced by [1:]: keeping 1 of 17 images"
    ) in result.output
    data = read_colmap_binary(out)
    assert list(data["image_names"]) == ["seoul_bull_sculpture_05.jpg"]


def test_to_colmap_bin_slice_keeps_the_reconstructions_order(
    seoul_bull_ground_truth_sfmr, tmp_path
):
    sfmr = seoul_bull_ground_truth_sfmr
    _gt_lists(sfmr)
    out = tmp_path / "colmap"
    result = _invoke(["to-colmap-bin", str(sfmr), str(out), "-r", "first_three[::-2]"])
    assert result.exit_code == 0, result.output
    data = read_colmap_binary(out)
    assert list(data["image_names"]) == [
        f"seoul_bull_sculpture_{i:02d}.jpg" for i in (1, 3)
    ]
    result = _invoke(
        ["to-colmap-bin", str(sfmr), str(tmp_path / "c2"), "-r", "picked[2:]"]
    )
    assert result.exit_code == 1, result.output
    assert (
        "the slice [2:] of named image list 'picked' leaves none of its 2 images"
    ) in " ".join(result.output.split())


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
    assert _read_as_a_list("nope") in " ".join(result.output.split())


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


def test_analyze_metrics_uses_a_sliced_list(seoul_bull_ground_truth_sfmr):
    sfmr = seoul_bull_ground_truth_sfmr
    _gt_lists(sfmr)
    result = _invoke(
        ["analyze", "--metrics", str(sfmr), "-r", "first_three[-1:]", "-r", "1-2"]
    )
    assert result.exit_code == 0, result.output
    assert (
        "Selected by named image list 'first_three' narrowed by the range '1-2' "
        "and sliced by [-1:] (1 of 17 images)"
    ) in result.output
    assert "seoul_bull_sculpture_02.jpg" in result.output
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
        "pick-images",
        "solve",
        "estimate-intrinsics",
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
    assert (
        "A range is file numbers and spans separated by commas, and a span may "
        f"take a step after a ':', as in {RANGE_EXAMPLES}."
    ) in output
    assert "--named-image-list NAME" in output
    assert "A value with a letter in it is instead the name of a named image list" in (
        output
    )
    assert "A slice after the name, quoted, takes part of the list by position" in (
        output
    )
    assert "'NAME[:30]' is its first 30 images" in output
