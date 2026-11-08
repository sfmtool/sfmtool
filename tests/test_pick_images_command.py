# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm pick-images` (specs/cli/image-feature/pick-images-command.md).

The walk itself is tested in tests/rust_bindings/analysis/; these tests cover
what the command adds: collecting the frames into sequences, the workspace and
version 1 checks, the refusal to overwrite a list, the list it writes and the
report. The video is a synthetic pan: crops 320 x 240 sliding 4 pixels per
frame across one dino_dog_toy photograph, a step of 1.25% of the long side.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._pick_images import collect_sequences, find_rigs
from sfmtool.camrig.resolver import resolve_camrig_for_solve
from sfmtool.cli import main
from sfmtool.fileio import write_camrig
from sfmtool.reconstruction import RangeExpr
from sfmtool.workspace import NAMED_IMAGE_LISTS_FILE_NAME, NamedImageLists

from .conftest import SEOUL_BULL_DIR, TEST_DATA_DIR

PAN_FRAMES = 100
PAN_WIDTH, PAN_HEIGHT, PAN_STEP = 320, 240, 4
# The pan's small crops have a few hundred features, so its neighbouring
# frames match with fewer matches than the default match floor and every
# frame would be chosen; a floor of 60 lets the walk thin the pan, as it thins
# a video, and keeps each run to a few seconds.
FAST = ["--min-step-matches", "60"]
# The opencv tool finds fewer features on the crops, so that its pan's
# neighbouring frames keep fewer than 60 matches; a floor of 20 lets the walk
# pass over frames there too.
OPENCV_FAST = ["--min-step-matches", "20"]


def _invoke(args: list[str]):
    return CliRunner().invoke(main, args, catch_exceptions=False)


@pytest.fixture(scope="module")
def pan_source(tmp_path_factory) -> dict[str, list[Path]]:
    """Two synthetic pans of PAN_FRAMES frames along two textured rows of
    the photograph (the floor and the table), written once for the module."""
    photo = cv2.imread(
        str(TEST_DATA_DIR / "images" / "dino_dog_toy" / "dino_dog_toy_01.jpg")
    )
    root = tmp_path_factory.mktemp("pan_source")
    pans = {}
    for name, y0 in (("top", 1000), ("bottom", 1250)):
        folder = root / name
        folder.mkdir()
        paths = []
        for i in range(PAN_FRAMES):
            x0 = i * PAN_STEP
            path = folder / f"pan_{i + 1:03d}.jpg"
            cv2.imwrite(str(path), photo[y0 : y0 + PAN_HEIGHT, x0 : x0 + PAN_WIDTH])
            paths.append(path)
        pans[name] = paths
    return pans


def _init_workspace(workspace: Path) -> None:
    result = _invoke(["ws", "init", str(workspace)])
    assert result.exit_code == 0, result.output


def _copy_frames(paths: list[Path], folder: Path, count: int | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in paths[:count]:
        shutil.copy(path, folder / path.name)


@pytest.fixture
def pan_workspace(tmp_path: Path, pan_source, monkeypatch) -> Path:
    """A version 2 workspace holding the top pan under `frames/`, with the
    current directory set to it."""
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "frames")
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)
    return workspace


def _files(workspace: Path) -> set[str]:
    return {
        p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file()
    }


def _resolve(workspace: Path, name: str) -> list[str]:
    lists = NamedImageLists.read(workspace)
    return [m.path for m in lists.resolve(name)]


def test_report_only_writes_nothing_but_sift_files(pan_workspace: Path):
    before = _files(pan_workspace)
    result = _invoke(["pick-images", "frames", *FAST])
    assert result.exit_code == 0, result.output
    assert "frames/pan_%03d.jpg, 100 frames (1-100)" in result.output
    assert "Walk: target step 10% (band 7-13%)" in result.output
    assert "pass --write-list NAME" in result.output
    added = _files(pan_workspace) - before
    assert added
    assert all(
        p.endswith(".jpg.sift") and p.startswith("frames/features/") for p in added
    )
    assert not (pan_workspace / NAMED_IMAGE_LISTS_FILE_NAME).exists()


def test_write_list_writes_the_chosen_frames(pan_workspace: Path):
    result = _invoke(
        ["pick-images", "frames", *FAST, "--write-list", "pan", "--json", "r.json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads((pan_workspace / "r.json").read_text())
    (chosen,) = data["chosen_numbers"]
    assert len(chosen) > 5
    positions = data["sequences"][0]["chosen"]
    assert chosen == [data["capture"][0]["numbers"][p] for p in positions]

    lists = json.loads((pan_workspace / NAMED_IMAGE_LISTS_FILE_NAME).read_text())
    (entry,) = lists["named_image_lists"]["pan"]
    assert entry["pattern"] == "frames/pan_%03d.jpg"
    assert set(RangeExpr(entry["range"])) == set(chosen)
    assert _resolve(pan_workspace, "pan") == [f"frames/pan_{n:03d}.jpg" for n in chosen]
    assert f"Wrote the named image list pan ({len(chosen)} images)" in result.output
    assert data["named_image_list"]["entries"] == [entry]

    # `sfm sift -r NAME` takes the same images.
    sift = _invoke(["sift", "--extract", "-r", "pan"])
    assert sift.exit_code == 0, sift.output
    assert f"expanded to {len(chosen)} filename(s)" in sift.output


def test_json_report_and_reuse_of_sift_files(pan_workspace: Path):
    first = _invoke(["pick-images", "frames", *FAST, "--json", "a.json"])
    assert first.exit_code == 0, first.output
    a = json.loads((pan_workspace / "a.json").read_text())
    assert set(a) >= {"sequences", "frames", "pairs", "capture", "options"}
    assert a["options"] == {
        "target_step_percent": 10.0,
        "step_band": 0.3,
        "neighbour_blur_ratio": 0.85,
        "min_step_matches": 60,
    }
    assert a["capture"][0]["patterns"] == ["frames/pan_%03d.jpg"]
    assert a["capture"][0]["numbers"] == list(range(1, PAN_FRAMES + 1))
    assert a["capture"][0]["images"][4] == ["frames/pan_005.jpg"]
    assert a["named_image_list"] is None
    assert len(a["frames_extracted"]) == len(a["frames"])
    assert all(a["frames_extracted"])
    read = a["sequences"][0]["frames_read"]
    assert read == len(a["frames"])
    assert f"read: {read} frames ({read} extracted, 0 reused)" in first.output
    # The frames predicted and never read have .sift files, and are counted.
    unread = a["frames_extracted_unread"]
    sift_files = list((pan_workspace / "frames").rglob("*.jpg.sift"))
    assert len(sift_files) == len(a["frames"]) + unread
    assert f"Extracted and never read: {unread} frame" in first.output

    # A second run reads the same frames and extracts none of them.
    second = _invoke(["pick-images", "frames", *FAST, "--json", "b.json"])
    assert second.exit_code == 0, second.output
    b = json.loads((pan_workspace / "b.json").read_text())
    assert b["chosen_numbers"] == a["chosen_numbers"]
    assert not any(b["frames_extracted"])
    assert f"read: {read} frames (0 extracted, {read} reused)" in second.output


def test_an_existing_name_is_refused_before_any_image_is_read(pan_workspace: Path):
    lists_path = pan_workspace / NAMED_IMAGE_LISTS_FILE_NAME
    lists_path.write_text(
        json.dumps({"version": 1, "named_image_lists": {"pan": [{"pattern": "x.jpg"}]}})
    )
    result = CliRunner().invoke(main, ["pick-images", "frames", "--write-list", "pan"])
    assert result.exit_code == 1, result.output
    assert (
        f"{lists_path}: the named image list 'pan' is already defined" in result.output
    )
    assert "--force" in result.output
    assert not (pan_workspace / "frames" / "features").exists()
    assert "x.jpg" in lists_path.read_text()


def test_force_replaces_a_list_in_place(pan_workspace: Path):
    lists_path = pan_workspace / NAMED_IMAGE_LISTS_FILE_NAME
    lists_path.write_text(
        json.dumps(
            {
                "version": 1,
                "named_image_lists": {
                    "pan": [{"pattern": "x.jpg"}],
                    "other": [{"pattern": "y.jpg"}],
                },
            }
        )
    )
    result = _invoke(
        [
            "pick-images",
            "frames",
            *FAST,
            "--write-list",
            "pan",
            "--force",
            "--json",
            "r.json",
        ]
    )
    assert result.exit_code == 0, result.output
    data = json.loads((pan_workspace / "r.json").read_text())
    (chosen,) = data["chosen_numbers"]
    lists = json.loads(lists_path.read_text())["named_image_lists"]
    assert list(lists) == ["pan", "other"]
    assert lists["pan"][0]["pattern"] == "frames/pan_%03d.jpg"
    assert _resolve(pan_workspace, "pan") == [f"frames/pan_{n:03d}.jpg" for n in chosen]


def test_a_list_with_a_slice_as_input(pan_workspace: Path):
    lists_path = pan_workspace / NAMED_IMAGE_LISTS_FILE_NAME
    lists_path.write_text(
        json.dumps(
            {
                "version": 1,
                "named_image_lists": {"pan_all": [{"pattern": "frames/pan_%03d.jpg"}]},
            }
        )
    )
    result = _invoke(["pick-images", "-r", "pan_all[5:95]", *FAST, "--json", "r.json"])
    assert result.exit_code == 0, result.output
    assert "Given named image list 'pan_all' sliced by [5:95]" in result.output
    data = json.loads((pan_workspace / "r.json").read_text())
    assert data["capture"][0]["numbers"] == list(range(6, 96))
    (chosen,) = data["chosen_numbers"]
    assert chosen and all(6 <= n <= 95 for n in chosen)

    # The input list's own name is an output name only with --force.
    refused = CliRunner().invoke(
        main, ["pick-images", "-r", "pan_all", "--write-list", "pan_all"]
    )
    assert refused.exit_code == 1, refused.output
    assert "'pan_all' is already defined" in refused.output
    replaced = _invoke(
        [
            "pick-images",
            "-r",
            "pan_all",
            *FAST,
            "--write-list",
            "pan_all",
            "--force",
            "--json",
            "replaced.json",
        ]
    )
    assert replaced.exit_code == 0, replaced.output
    entries = json.loads(lists_path.read_text())["named_image_lists"]["pan_all"]
    assert len(entries) == 1 and "range" in entries[0]
    assert len(_resolve(pan_workspace, "pan_all")) < PAN_FRAMES


def test_a_range_narrows_the_folder(pan_workspace: Path):
    result = _invoke(["pick-images", "frames", "-r", "1-70", *FAST, "--json", "r.json"])
    assert result.exit_code == 0, result.output
    assert "frames/pan_%03d.jpg, 70 frames (1-70)" in result.output
    data = json.loads((pan_workspace / "r.json").read_text())
    assert data["capture"][0]["numbers"] == list(range(1, 71))


def test_another_feature_tool_extracts_every_frame_first(
    tmp_path: Path, pan_source, monkeypatch
):
    workspace = tmp_path / "ws"
    # 70 frames: few to extract, and more than the walk reads, so some are
    # extracted first and never read.
    _copy_frames(pan_source["top"], workspace / "frames", count=70)
    result = _invoke(["ws", "init", "--feature-tool", "opencv", str(workspace)])
    assert result.exit_code == 0, result.output
    monkeypatch.chdir(workspace)

    result = _invoke(["pick-images", "frames", *OPENCV_FAST, "--json", "r.json"])
    assert result.exit_code == 0, result.output
    assert "with the opencv tool before the walk" in result.output
    data = json.loads((workspace / "r.json").read_text())
    # Every input frame has a .sift file, extracted before the walk read any.
    sift_files = sorted(p.name for p in (workspace / "frames").rglob("*.jpg.sift"))
    assert sift_files == [f"pan_{n:03d}.jpg.sift" for n in range(1, 71)]
    read = len(data["frames"])
    assert 0 < read < 70
    assert data["frames_extracted"] == [True] * read
    assert data["frames_extracted_unread"] == 70 - read
    assert f"Extracted and never read: {70 - read} frames" in result.output

    # A second run reads the same files and extracts none.
    again = _invoke(["pick-images", "frames", *OPENCV_FAST, "--json", "b.json"])
    assert again.exit_code == 0, again.output
    b = json.loads((workspace / "b.json").read_text())
    assert b["frames"] == data["frames"]
    assert not any(b["frames_extracted"]) and b["frames_extracted_unread"] == 0


def test_photographs_below_the_match_floor_keep_every_frame(
    tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "ws"
    images = workspace / "photos"
    images.mkdir(parents=True)
    for path in sorted(SEOUL_BULL_DIR.glob("seoul_bull_sculpture_*.jpg")):
        shutil.copy(path, images / path.name)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = _invoke(
        ["pick-images", "photos", "--write-list", "bull", "--json", "r.json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads((workspace / "r.json").read_text())
    # Every step is to a next frame that keeps fewer matches than the floor.
    steps = data["sequences"][0]["steps"]
    assert all(s["stats"]["matches"] < 600 for s in steps)
    assert (
        "Steps stopped at the match floor: 16 of 16, 16 of them at the next frame"
        in result.output
    )
    assert data["options"]["min_step_matches"] == 600
    assert data["chosen_numbers"] == [list(range(1, 18))]
    assert _resolve(workspace, "bull") == [
        f"photos/seoul_bull_sculpture_{n:02d}.jpg" for n in range(1, 18)
    ]


def test_a_version_1_workspace_is_refused(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "ws"
    images = workspace / "images"
    images.mkdir(parents=True)
    for i in range(1, 4):
        name = f"seoul_bull_sculpture_{i:02d}.jpg"
        shutil.copy(SEOUL_BULL_DIR / name, images / name)
    shutil.copy(
        SEOUL_BULL_DIR / ".sfm-workspace.json", workspace / ".sfm-workspace.json"
    )
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["pick-images", "images"])
    assert result.exit_code == 1, result.output
    assert (
        f"The workspace file {workspace / '.sfm-workspace.json'} holds version 1 settings"
        in result.output
    )
    assert "Run 'sfm ws upgrade'" in result.output
    # Refused up front, before the sequences are collected, not by the first
    # extraction of the walk.
    assert "image file(s) in" not in result.output
    assert not (images / "features").exists()


def test_no_workspace_is_an_error(tmp_path: Path, pan_source):
    _copy_frames(pan_source["top"], tmp_path / "frames", count=5)
    result = CliRunner().invoke(main, ["pick-images", str(tmp_path / "frames")])
    assert result.exit_code == 1, result.output
    assert "No workspace found for images at" in result.output
    assert "'sfm ws init'" in result.output


def test_files_with_no_number_or_a_shared_number_are_errors(
    tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "ws"
    frames = workspace / "frames"
    frames.mkdir(parents=True)
    photo = SEOUL_BULL_DIR / "seoul_bull_sculpture_01.jpg"
    for name in ("cover.jpg", "shot_1.jpg", "shot_01.jpg", "shot_2.jpg"):
        shutil.copy(photo, frames / name)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["pick-images", "frames"])
    assert result.exit_code == 1, result.output
    assert "1 image file(s) have no file number" in result.output
    assert "frames/cover.jpg" in result.output
    assert (
        "1 file number(s) of the sequence frames/shot_%d.jpg name more than one file"
        in (result.output)
    )
    assert (
        "frames/shot_1.jpg" in result.output and "frames/shot_01.jpg" in result.output
    )
    assert not (frames / "features").exists()


def _write_rig_config(workspace: Path) -> None:
    # The reference sensor is listed second, so the command must put it first.
    rig = [
        {
            "cameras": [
                {
                    "image_prefix": "rig/cam_b/",
                    "cam_from_rig_rotation": [1, 0, 0, 0],
                    "cam_from_rig_translation": [0.1, 0, 0],
                },
                {"image_prefix": "rig/cam_a/", "ref_sensor": True},
            ]
        }
    ]
    (workspace / "rig_config.json").write_text(json.dumps(rig))


def test_a_rig_writes_every_sensor_of_each_chosen_frame(
    tmp_path: Path, pan_source, monkeypatch
):
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "rig" / "cam_a")
    _copy_frames(pan_source["bottom"], workspace / "rig" / "cam_b")
    _write_rig_config(workspace)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = _invoke(
        ["pick-images", "rig", *FAST, "--write-list", "rig_pick", "--json", "r.json"]
    )
    assert result.exit_code == 0, result.output
    assert (
        "rig/cam_a/pan_%03d.jpg and rig/cam_b/pan_%03d.jpg, 100 rig frames of 2 sensors"
        in result.output
    )
    assert "For a quick test solve" not in result.output
    data = json.loads((workspace / "r.json").read_text())
    (capture,) = data["capture"]
    assert capture["sensors"] == ["rig/cam_a/", "rig/cam_b/"]
    assert capture["images"][0] == ["rig/cam_a/pan_001.jpg", "rig/cam_b/pan_001.jpg"]
    assert all(len(frame["sensors"]) == 2 for frame in data["frames"])
    (chosen,) = data["chosen_numbers"]
    entries = json.loads((workspace / NAMED_IMAGE_LISTS_FILE_NAME).read_text())[
        "named_image_lists"
    ]["rig_pick"]
    assert [e["pattern"] for e in entries] == [
        "rig/cam_a/pan_%03d.jpg",
        "rig/cam_b/pan_%03d.jpg",
    ]
    assert entries[0]["range"] == entries[1]["range"]
    assert _resolve(workspace, "rig_pick") == [
        f"rig/{cam}/pan_{n:03d}.jpg" for cam in ("cam_a", "cam_b") for n in chosen
    ]
    assert f"({2 * len(chosen)} images)" in result.output


def test_a_rig_frame_missing_a_sensor_is_an_error(
    tmp_path: Path, pan_source, monkeypatch
):
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "rig" / "cam_a", count=6)
    _copy_frames(pan_source["bottom"], workspace / "rig" / "cam_b", count=5)
    _write_rig_config(workspace)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["pick-images", "rig"])
    assert result.exit_code == 1, result.output
    assert "1 rig frame(s) of rig/cam_a/pan_%03d.jpg lack the image of a sensor" in (
        result.output
    )
    assert "rig/cam_a/pan_006.jpg (no image under rig/cam_b/)" in result.output


def _write_camrig(path: Path, patterns: list[str]) -> None:
    """A `.camrig` of one sensor per pattern, sharing one camera the size of
    the pan's crops."""
    count = len(patterns)
    write_camrig(
        path=str(path),
        name=path.stem,
        rig_type="generic",
        cameras=[
            {
                "model": "PINHOLE",
                "width": PAN_WIDTH,
                "height": PAN_HEIGHT,
                "parameters": {
                    "focal_length_x": 300.0,
                    "focal_length_y": 300.0,
                    "principal_point_x": PAN_WIDTH / 2,
                    "principal_point_y": PAN_HEIGHT / 2,
                },
            }
        ],
        sensor_image_patterns=patterns,
        camera_indexes=[0] * count,
        quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        translations_xyz=np.array([[0.1 * i, 0.0, 0.0] for i in range(count)]),
    )


# The rig `.camrig` of the tests below sits in `rig/`, and its sensor 0, the
# reference, is cam_b, which sorts after cam_a.
CAMRIG_PATTERNS = ["cam_b/pan_%03d.jpg", "cam_a/pan_%03d.jpg"]


def test_a_camrig_rig_writes_every_sensor_of_each_chosen_frame(
    tmp_path: Path, pan_source, monkeypatch
):
    workspace = tmp_path / "ws"
    count = 30
    _copy_frames(pan_source["top"], workspace / "rig" / "cam_b", count=count)
    _copy_frames(pan_source["bottom"], workspace / "rig" / "cam_a", count=count)
    _write_camrig(workspace / "rig" / "pan.camrig", CAMRIG_PATTERNS)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = _invoke(
        ["pick-images", "rig", *FAST, "--write-list", "rig_pick", "--json", "r.json"]
    )
    assert result.exit_code == 0, result.output
    assert "pan.camrig (2 sensors)" in result.output
    assert (
        f"rig/cam_b/pan_%03d.jpg and rig/cam_a/pan_%03d.jpg, {count} rig frames of 2 "
        "sensors" in result.output
    )
    data = json.loads((workspace / "r.json").read_text())
    (capture,) = data["capture"]
    assert capture["sensors"] == CAMRIG_PATTERNS
    assert capture["images"][0] == ["rig/cam_b/pan_001.jpg", "rig/cam_a/pan_001.jpg"]
    assert all(len(frame["sensors"]) == 2 for frame in data["frames"])
    (chosen,) = data["chosen_numbers"]
    assert 1 < len(chosen) < count
    entries = json.loads((workspace / NAMED_IMAGE_LISTS_FILE_NAME).read_text())[
        "named_image_lists"
    ]["rig_pick"]
    assert [e["pattern"] for e in entries] == [
        "rig/cam_b/pan_%03d.jpg",
        "rig/cam_a/pan_%03d.jpg",
    ]
    assert (
        entries[0]["range"] == entries[1]["range"] == str(RangeExpr.from_list(chosen))
    )

    # What `sfm solve -r rig_pick` solves: the list's images, which the
    # .camrig covers as a rig of both sensors at the chosen frames.
    listed = _resolve(workspace, "rig_pick")
    solve = resolve_camrig_for_solve(
        [workspace / p for p in listed], workspace, camera_model=None
    )
    assert solve is not None and solve.is_multi_sensor
    assert sorted(solve.rig.assignments.values()) == sorted(
        (sensor, n) for sensor in (0, 1) for n in chosen
    )


def _touch(workspace: Path, rels: list[str]) -> list[Path]:
    """Empty files at `rels`: finding the rigs reads names only."""
    paths = []
    for rel in rels:
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(path)
    return paths


RIG_FILES = [f"rig/{cam}/pan_00{n}.jpg" for cam in ("cam_a", "cam_b") for n in (1, 2)]


def test_a_camrig_covering_the_images_takes_precedence_over_rig_config(
    tmp_path: Path,
):
    workspace = tmp_path / "ws"
    paths = _touch(workspace, RIG_FILES)
    _write_rig_config(workspace)
    rig_configs, camrig_rig, notes = find_rigs(paths, workspace)
    assert rig_configs is not None and camrig_rig is None and notes == []

    _write_camrig(workspace / "rig" / "pan.camrig", CAMRIG_PATTERNS)
    rig_configs, camrig_rig, notes = find_rigs(paths, workspace)
    assert rig_configs is None
    assert sorted(camrig_rig.assignments.values()) == [
        (s, n) for s in (0, 1) for n in (1, 2)
    ]
    assert "rig_config.json is ignored (the .camrig takes precedence)" in notes[-1]
    (sequence,) = collect_sequences(paths, workspace, rig_configs, camrig_rig)
    assert sequence.patterns == ["rig/cam_b/pan_%03d.jpg", "rig/cam_a/pan_%03d.jpg"]


def test_a_single_sensor_camrig_defines_no_rig(tmp_path: Path):
    # A one-sensor .camrig is a camera only; covering every image, it still
    # takes precedence over rig_config.json, as in `sfm solve`.
    workspace = tmp_path / "ws"
    paths = _touch(workspace, RIG_FILES)
    _write_rig_config(workspace)
    _write_camrig(workspace / "rig" / "one.camrig", ["*/pan_%03d.jpg"])
    rig_configs, camrig_rig, notes = find_rigs(paths, workspace)
    assert rig_configs is None and camrig_rig is None
    assert len(notes) == 1 and "rig_config.json is ignored" in notes[0]
    sequences = collect_sequences(paths, workspace, rig_configs, camrig_rig)
    assert [s.patterns for s in sequences] == [
        ["rig/cam_a/pan_%03d.jpg"],
        ["rig/cam_b/pan_%03d.jpg"],
    ]


def test_several_single_sensor_camrigs_walk_each_sequence(
    tmp_path: Path, pan_source, monkeypatch
):
    # Two single-sensor .camrig files describe two cameras, not a rig: the
    # command walks each folder's sequence as it does with no .camrig.
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "rig" / "cam_a", count=3)
    _copy_frames(pan_source["bottom"], workspace / "rig" / "cam_b", count=3)
    _write_camrig(workspace / "rig" / "cam_a" / "a.camrig", ["pan_%03d.jpg"])
    _write_camrig(workspace / "rig" / "cam_b" / "b.camrig", ["pan_%03d.jpg"])
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["pick-images", "rig", "--json", "r.json"])
    assert result.exit_code == 0, result.output
    assert "Camera rig:" not in result.output
    data = json.loads((workspace / "r.json").read_text())
    assert [c["sensors"] for c in data["capture"]] == [None, None]
    assert [c["images"][0] for c in data["capture"]] == [
        ["rig/cam_a/pan_001.jpg"],
        ["rig/cam_b/pan_001.jpg"],
    ]


def test_images_a_camrig_covers_only_in_part_are_refused(
    tmp_path: Path, pan_source, monkeypatch
):
    # `sfm solve` refuses a set of images a .camrig covers only in part, so
    # the command refuses it before reading any image.
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "rig" / "cam_b", count=3)
    _copy_frames(pan_source["bottom"], workspace / "rig" / "cam_a", count=3)
    _copy_frames(pan_source["top"], workspace / "frames", count=3)
    _write_camrig(workspace / "rig" / "pan.camrig", CAMRIG_PATTERNS)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["pick-images", "rig", "frames"])
    assert result.exit_code == 1, result.output
    assert "pan.camrig matches only 6 of 9 images given" in result.output
    assert "Pick only the images the .camrig covers" in result.output
    assert not any(p.suffix == ".sift" for p in workspace.rglob("*"))

    result = CliRunner().invoke(main, ["pick-images", "rig"])
    assert result.exit_code == 0, result.output


def test_an_invalid_output_name_is_a_usage_error(pan_workspace: Path):
    result = CliRunner().invoke(main, ["pick-images", "frames", "--write-list", "12-3"])
    assert result.exit_code == 2, result.output
    assert "--write-list '12-3' is not a valid list name" in result.output


def test_force_without_a_list_and_paths_with_a_list_are_usage_errors(
    pan_workspace: Path,
):
    result = CliRunner().invoke(main, ["pick-images", "frames", "--force"])
    assert result.exit_code == 2, result.output
    assert "--force replaces a written list; give --write-list." in result.output
    result = CliRunner().invoke(main, ["pick-images", "frames", "-r", "pan_all"])
    assert result.exit_code == 2, result.output
    assert "give paths or a list, not both" in result.output
    assert not (pan_workspace / "frames" / "features").exists()


def test_an_out_of_range_match_floor_is_refused(pan_workspace: Path):
    # A floor below 1 is a usage error; one too large for the binding is
    # refused with the option's name before a frame is read.
    for value in ("-1", "0"):
        result = CliRunner().invoke(
            main, ["pick-images", "frames", "--min-step-matches", value]
        )
        assert result.exit_code == 2, result.output
    result = CliRunner().invoke(
        main, ["pick-images", "frames", "--min-step-matches", str(2**70)]
    )
    assert result.exit_code == 1, result.output
    assert "min_step_matches" in result.output, result.output
    assert "Traceback" not in result.output
