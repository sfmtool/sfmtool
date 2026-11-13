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
import os
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
from .test_extract_frames_command import _ffmpeg

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
    # The pan slides crops of one photograph, an image motion no focal
    # explains, so the pilot vote reaches no consensus, and runs all the same.
    assert (
        "Local density term: off, the pilot vote reached no consensus ("
        in result.output
    )
    assert "  Pilot vote: 14 frames chosen, " in result.output
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
        "max_density_gain": 3.0,
        "step_band": 0.3,
        "neighbour_blur_ratio": 0.85,
        "min_step_matches": 60,
    }
    assert a["capture"][0]["patterns"] == ["frames/pan_%03d.jpg"]
    assert a["capture"][0]["numbers"] == list(range(1, PAN_FRAMES + 1))
    assert a["capture"][0]["images"][4] == ["frames/pan_005.jpg"]
    assert a["capture"][0]["camera"] is None
    vote = a["sequences"][0]["pilot_vote"]
    assert vote["outcome"] == {"status": "no_consensus"}
    assert a["capture"][0]["camera_source"] == (
        f"the pilot vote reached no consensus ({vote['n_pool']} pooled "
        f"vote{'' if vote['n_pool'] == 1 else 's'})"
    )
    assert a["sequences"][0]["density_term"] == {"status": "off", "reason": "no_camera"}
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
    # The pilot vote on the 17 photographs disagrees with itself: its pool
    # spread is above the bound, and the term is off saying so.
    vote = data["sequences"][0]["pilot_vote"]
    assert vote["frames"] == list(range(17))
    assert vote["outcome"] == {"status": "wide_spread", "max_pool_spread": 0.35}
    assert vote["pool_spread"] > 0.35
    assert data["sequences"][0]["density_term"]["status"] == "off"
    assert (
        f"Local density term: off, the pilot vote's pool spread "
        f"{vote['pool_spread']:.3f} is above 0.35 (pinhole, focal "
        f"{vote['focal_px']:.1f} px)" in result.output
    )
    assert _resolve(workspace, "bull") == [
        f"photos/seoul_bull_sculpture_{n:02d}.jpg" for n in range(1, 18)
    ]


def test_a_sequence_with_no_camrig_takes_the_pilot_vote_s_camera(
    tmp_path: Path, monkeypatch
):
    # seattle_backyard's 26 photographs match below the floor, so every one
    # is chosen and voted on, and the vote agrees with itself.
    source = TEST_DATA_DIR / "images" / "seattle_backyard"
    workspace = tmp_path / "ws"
    images = workspace / "photos"
    images.mkdir(parents=True)
    for path in sorted(source.glob("seattle_backyard_*.jpg")):
        shutil.copy(path, images / path.name)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)
    before = _files(workspace)

    result = _invoke(["pick-images", "photos", "--json", "r.json"])
    assert result.exit_code == 0, result.output
    data = json.loads((workspace / "r.json").read_text())
    (walk,) = data["sequences"]
    vote = walk["pilot_vote"]
    assert vote["frames"] == list(range(26))
    assert vote["frames_read"] == 26
    assert vote["outcome"] == {"status": "used"}
    assert vote["camera_model"] == "Pinhole" and vote["confirmed"] is None
    assert vote["pool_spread"] <= 0.35
    assert vote["n_pool"] == vote["n_epipolar"] + vote["n_rotation"] >= 2
    assert vote["features"] <= 26 * 1500 and vote["clusters"] > 0
    # The walk measured with the vote's camera.
    assert walk["density_term"] == {"status": "on", "focal_px": vote["focal_px"]}
    source_text = (
        f"the pilot vote's pinhole camera, focal {vote['focal_px']:.1f} px, "
        f"pool spread {vote['pool_spread']:.3f}"
    )
    assert data["capture"][0]["camera_source"] == source_text
    assert data["capture"][0]["camera"] is None
    assert f"Local density term: on, with {source_text}; " in result.output
    assert (
        f"  Pilot vote: 26 frames chosen, 26 read, {vote['features']} features "
        f"in {vote['clusters']} clusters; took " in result.output
    )
    # The vote wrote nothing into the workspace: only .sift files and the
    # report are new.
    added = _files(workspace) - before
    assert {p for p in added if not p.endswith(".jpg.sift")} == {"r.json"}

    # A maximum gain of 1 runs no pilot vote.
    off = _invoke(["pick-images", "photos", "--max-density-gain", "1"])
    assert off.exit_code == 0, off.output
    assert "Local density term: off, --max-density-gain is 1" in off.output
    assert "Pilot vote" not in off.output


def test_a_pilot_of_two_frames_runs_no_vote(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "ws"
    images = workspace / "photos"
    images.mkdir(parents=True)
    for n in (1, 2):
        name = f"seoul_bull_sculpture_{n:02d}.jpg"
        shutil.copy(SEOUL_BULL_DIR / name, images / name)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = _invoke(["pick-images", "photos", "--json", "r.json"])
    assert result.exit_code == 0, result.output
    data = json.loads((workspace / "r.json").read_text())
    vote = data["sequences"][0]["pilot_vote"]
    assert vote["outcome"] == {"status": "too_few_frames", "min_frames": 3}
    assert (vote["frames"], vote["features"], vote["focal_px"]) == ([0, 1], 0, None)
    assert data["sequences"][0]["density_term"] == {
        "status": "off",
        "reason": "no_camera",
    }
    assert (
        "Local density term: off, the pilot walk chose 2 frames, and a vote "
        "needs at least 3" in result.output
    )


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
        "1 file number(s) of the sequence frames/shot_%02d.jpg name more than one file"
        in (result.output)
    )
    assert (
        "frames/shot_1.jpg" in result.output and "frames/shot_01.jpg" in result.output
    )
    assert not (frames / "features").exists()


def _write_camrig(path: Path, patterns: list[str], scale: float = 1.0) -> None:
    """A `.camrig` of one sensor per pattern, sharing one camera `scale` times
    the size of the pan's crops."""
    count = len(patterns)
    write_camrig(
        path=str(path),
        name=path.stem,
        rig_type="generic",
        cameras=[
            {
                "model": "PINHOLE",
                "width": int(PAN_WIDTH * scale),
                "height": int(PAN_HEIGHT * scale),
                "parameters": {
                    "focal_length_x": 300.0 * scale,
                    "focal_length_y": 300.0 * scale,
                    "principal_point_x": PAN_WIDTH / 2 * scale,
                    "principal_point_y": PAN_HEIGHT / 2 * scale,
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
    (rig,) = solve.rigs
    assert sorted(rig.assignments.values()) == sorted(
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


def test_a_rig_config_json_is_refused_before_any_image_is_read(
    tmp_path: Path, pan_source, monkeypatch
):
    # A rig_config.json at the workspace root stops the command, naming the
    # conversion, even beside a .camrig that describes the same rig.
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "rig" / "cam_b", count=3)
    _copy_frames(pan_source["bottom"], workspace / "rig" / "cam_a", count=3)
    _write_camrig(workspace / "rig" / "pan.camrig", CAMRIG_PATTERNS)
    rig_config = workspace / "rig_config.json"
    rig_config.write_text(
        json.dumps(
            [
                {
                    "cameras": [
                        {"image_prefix": "rig/cam_b/", "ref_sensor": True},
                        {
                            "image_prefix": "rig/cam_a/",
                            "cam_from_rig_rotation": [1, 0, 0, 0],
                        },
                    ]
                }
            ]
        )
    )
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["pick-images", "rig"])
    assert result.exit_code == 1, result.output
    assert f"{rig_config}: sfm takes a workspace's rigs only from .camrig" in (
        result.output
    )
    assert f"sfm camrig cp {rig_config} {workspace / 'ws.camrig'}" in result.output
    assert not list(workspace.rglob("*.sift"))


def test_kerry_park_rig_is_grouped_by_its_camrig():
    # kerry_park describes its rig by kerry_park.camrig alone; finding the
    # rigs reads file names only, so the checked-in images serve.
    workspace = TEST_DATA_DIR / "images" / "kerry_park"
    paths = sorted(workspace.glob("fisheye_*/frame_*.jpg"))
    camrig_rigs, _cameras, notes = find_rigs(paths, workspace)
    assert [r.camrig_file.name for r in camrig_rigs] == ["kerry_park.camrig"]
    assert notes == [f"  Camera rig: {workspace / 'kerry_park.camrig'} (2 sensors)"]
    (sequence,) = collect_sequences(paths, workspace, camrig_rigs)
    assert sequence.patterns == [
        "fisheye_left/frame_%02d.jpg",
        "fisheye_right/frame_%02d.jpg",
    ]
    # Its fisheye camera, sensor 0's, is the local density term's.
    assert sequence.camera == camrig_rigs[0].data["cameras"][0]
    assert sequence.camera["model"] == "OPENCV_FISHEYE"
    assert sequence.camera_source.endswith("sensor fisheye_left/frame_%02d.jpg")
    assert sequence.numbers == list(range(1, 25))
    assert sequence.images[0] == [
        Path("fisheye_left/frame_01.jpg"),
        Path("fisheye_right/frame_01.jpg"),
    ]


def test_a_single_sensor_camrig_defines_no_rig(tmp_path: Path):
    # A one-sensor .camrig is a camera only, even when it covers every image.
    workspace = tmp_path / "ws"
    paths = _touch(workspace, RIG_FILES)
    _write_camrig(workspace / "rig" / "one.camrig", ["*/pan_%03d.jpg"])
    camrig_rigs, _cameras, notes = find_rigs(paths, workspace)
    assert camrig_rigs == [] and notes == []
    sequences = collect_sequences(paths, workspace, camrig_rigs)
    assert [s.patterns for s in sequences] == [
        ["rig/cam_a/pan_%03d.jpg"],
        ["rig/cam_b/pan_%03d.jpg"],
    ]


def test_a_single_sensor_camrig_covering_some_images_is_not_refused(
    tmp_path: Path,
):
    # `sfm solve` gives the images no single-sensor .camrig covers cameras
    # from camera_config.json or EXIF, so the command takes them too.
    workspace = tmp_path / "ws"
    paths = _touch(workspace, RIG_FILES)
    _write_camrig(workspace / "rig" / "cam_a" / "a.camrig", ["pan_%03d.jpg"])
    camrig_rigs, _cameras, notes = find_rigs(paths, workspace)
    assert camrig_rigs == [] and notes == []
    sequences = collect_sequences(paths, workspace, camrig_rigs)
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
    # Each sequence's .camrig gives it its camera.
    for name in ("cam_a/a.camrig", "cam_b/b.camrig"):
        assert (
            f"Local density term: on, with the camera of {workspace / 'rig' / name} "
            "(focal 300.0 px at the frames' size)" in result.output
        )
    data = json.loads((workspace / "r.json").read_text())
    assert [c["sensors"] for c in data["capture"]] == [None, None]
    assert [c["images"][0] for c in data["capture"]] == [
        ["rig/cam_a/pan_001.jpg"],
        ["rig/cam_b/pan_001.jpg"],
    ]


def test_the_local_density_term_takes_a_camrig_camera_scaled_or_is_turned_off(
    tmp_path: Path, pan_source, monkeypatch
):
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "frames", count=30)
    # The camera at twice the frames' size: the walk scales it to them.
    _write_camrig(workspace / "frames" / "cam.camrig", ["pan_%03d.jpg"], scale=2.0)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    on = _invoke(["pick-images", "frames", *FAST, "--json", "on.json"])
    assert on.exit_code == 0, on.output
    data = json.loads((workspace / "on.json").read_text())
    (walk,) = data["sequences"]
    assert walk["density_term"] == {"status": "on", "focal_px": 300.0}
    assert data["capture"][0]["camera"]["width"] == 2 * PAN_WIDTH
    assert data["capture"][0]["camera_source"] == (
        f"the camera of {workspace / 'frames' / 'cam.camrig'}"
    )
    assert all(s["density"]["essential_split"] for s in walk["steps"])
    assert "Local density by step" in on.output
    # The .camrig's camera is used, and no pilot vote runs.
    assert walk["pilot_vote"] is None
    assert "Pilot vote" not in on.output

    off = _invoke(["pick-images", "frames", *FAST, "--max-density-gain", "1"])
    assert off.exit_code == 0, off.output
    assert "Local density term: off, --max-density-gain is 1" in off.output
    assert "Local density by step" not in off.output
    for bad in ("0.5", "nan"):
        refused = CliRunner().invoke(
            main, ["pick-images", "frames", "--max-density-gain", bad]
        )
        assert refused.exit_code == 1, refused.output
        assert "max_density_gain must be finite and at least 1" in refused.output


def test_a_camrig_rig_covering_some_images_walks_them_beside_the_others(
    tmp_path: Path, pan_source, monkeypatch
):
    # The images a rig .camrig covers are its rig sequence, and the others
    # ordinary sequences, as `sfm solve` takes them: the rig's images as its
    # rig frames and the others as single images.
    workspace = tmp_path / "ws"
    _copy_frames(pan_source["top"], workspace / "rig" / "cam_b", count=3)
    _copy_frames(pan_source["bottom"], workspace / "rig" / "cam_a", count=3)
    _copy_frames(pan_source["top"], workspace / "frames", count=3)
    _write_camrig(workspace / "rig" / "pan.camrig", CAMRIG_PATTERNS)
    _init_workspace(workspace)
    monkeypatch.chdir(workspace)

    result = _invoke(
        ["pick-images", "rig", "frames", "--write-list", "mixed", "--json", "r.json"]
    )
    assert result.exit_code == 0, result.output
    assert "pan.camrig (2 sensors)" in result.output
    assert "9 image file(s) in 2 sequences" in result.output
    data = json.loads((workspace / "r.json").read_text())
    assert [c["sensors"] for c in data["capture"]] == [None, CAMRIG_PATTERNS]
    assert [c["images"][0] for c in data["capture"]] == [
        ["frames/pan_001.jpg"],
        ["rig/cam_b/pan_001.jpg", "rig/cam_a/pan_001.jpg"],
    ]
    frames_chosen, rig_chosen = data["chosen_numbers"]

    listed = _resolve(workspace, "mixed")
    solve = resolve_camrig_for_solve(
        [workspace / p for p in listed], workspace, camera_model=None
    )
    (rig,) = solve.rigs
    assert sorted(rig.assignments.values()) == sorted(
        (sensor, n) for sensor in (0, 1) for n in rig_chosen
    )
    assert sorted(p for p in listed if p.startswith("frames/")) == [
        f"frames/pan_{n:03d}.jpg" for n in frames_chosen
    ]


def test_each_camrig_rig_is_its_own_sequence(tmp_path: Path):
    workspace = tmp_path / "ws"
    paths = _touch(workspace, RIG_FILES)
    paths += _touch(workspace, [p.replace("rig/", "two/") for p in RIG_FILES])
    paths += _touch(workspace, ["frames/pan_001.jpg", "frames/pan_002.jpg"])
    _write_camrig(workspace / "rig" / "pan.camrig", CAMRIG_PATTERNS)
    _write_camrig(workspace / "two" / "pan.camrig", CAMRIG_PATTERNS)
    camrig_rigs, _cameras, notes = find_rigs(paths, workspace)
    assert [r.camrig_file.parent.name for r in camrig_rigs] == ["rig", "two"]
    assert [len(r.assignments) for r in camrig_rigs] == [4, 4]
    assert len(notes) == 2
    sequences = collect_sequences(paths, workspace, camrig_rigs)
    assert [s.patterns for s in sequences] == [
        ["frames/pan_%03d.jpg"],
        ["rig/cam_b/pan_%03d.jpg", "rig/cam_a/pan_%03d.jpg"],
        ["two/cam_b/pan_%03d.jpg", "two/cam_a/pan_%03d.jpg"],
    ]
    assert [s.numbers for s in sequences] == [[1, 2]] * 3


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


# `--new-workspace`: two videos of the pans, 40 frames each, written with
# ffmpeg's built-in mpeg4 encoder and extracted at --fps full.
VIDEO_FRAMES = 40


@pytest.fixture(scope="module")
def pan_videos(pan_source, tmp_path_factory) -> list[Path]:
    root = tmp_path_factory.mktemp("pan_videos")
    videos = []
    for name in ("top", "bottom"):
        video = root / f"{name}.mp4"
        _ffmpeg(
            *[
                "-framerate",
                "30",
                "-i",
                str(pan_source[name][0].parent / "pan_%03d.jpg"),
            ],
            *["-frames:v", str(VIDEO_FRAMES), "-c:v", "mpeg4", "-q:v", "2"],
            str(video),
        )
        videos.append(video)
    return videos


def _new_workspace(workspace: Path, videos: list[Path], *args: str):
    return CliRunner().invoke(
        main,
        [
            "pick-images",
            "--new-workspace",
            str(workspace),
            *map(str, videos),
            "--fps",
            "full",
            *args,
        ],
    )


def test_new_workspace_extracts_the_videos_and_picks_from_their_frames(
    tmp_path: Path, pan_videos
):
    workspace = tmp_path / "garden"
    result = _new_workspace(
        workspace,
        pan_videos,
        *FAST,
        *["--max-features", "1024", "--write-list", "picked"],
        *["--json", str(workspace / "r.json")],
    )
    assert result.exit_code == 0, result.output
    config = json.loads((workspace / ".sfm-workspace.json").read_text())
    assert config["feature_options"]["max_num_features"] == 1024
    assert "Given 2 videos, 80 image file(s) in 2 sequences:" in result.output
    data = json.loads((workspace / "r.json").read_text())
    chosen = data["chosen_numbers"]
    assert all(1 < len(numbers) < VIDEO_FRAMES for numbers in chosen)
    # The sequences are in pick-images' order, by pattern.
    assert [c["patterns"] for c in data["capture"]] == [
        ["bottom/frame_%06d.jpg"],
        ["top/frame_%06d.jpg"],
    ]
    picked = [
        f"{stem}/frame_{n:06d}.jpg"
        for stem, numbers in zip(("bottom", "top"), chosen)
        for n in numbers
    ]
    assert _resolve(workspace, "picked") == picked
    prefix = config["feature_prefix_dir"]
    for path in picked:
        frame = workspace / path
        assert (frame.parent / prefix / (frame.name + ".sift")).is_file()
    # Without --prune-images every extracted frame stays.
    for stem in ("top", "bottom"):
        frames = list((workspace / stem).glob("frame_*.jpg"))
        assert len(frames) == VIDEO_FRAMES


def test_prune_images_keeps_exactly_the_chosen_frames(tmp_path: Path, pan_videos):
    workspace = tmp_path / "garden"
    video = pan_videos[0]
    result = _new_workspace(
        workspace,
        [video],
        *FAST,
        *["-r", "5-36", "--write-list", "picked", "--prune-images"],
        *["--json", str(workspace / "r.json")],
    )
    assert result.exit_code == 0, result.output
    prefix = json.loads((workspace / ".sfm-workspace.json").read_text())[
        "feature_prefix_dir"
    ]
    (chosen,) = json.loads((workspace / "r.json").read_text())["chosen_numbers"]
    assert chosen[0] == 5 and chosen[-1] <= 36
    kept = [f"top/frame_{n:06d}.jpg" for n in chosen]
    assert _files(workspace) == {
        ".sfm-workspace.json",
        NAMED_IMAGE_LISTS_FILE_NAME,
        "r.json",
        "top/video-frames.json",
        *kept,
        *(f"top/{prefix}/frame_{n:06d}.jpg.sift" for n in chosen),
    }
    report = json.loads((workspace / "r.json").read_text())
    written = sum(report["frames_extracted"]) + report["frames_extracted_unread"]
    deleted_sift = written - len(chosen)
    assert (
        f"Pruned {VIDEO_FRAMES - len(chosen)} frames not chosen, and "
        f"{deleted_sift} .sift files of them." in result.output
    )
    assert video.is_file()
    assert _resolve(workspace, "picked") == kept


def test_a_failed_pick_prunes_nothing(tmp_path: Path, pan_videos):
    workspace = tmp_path / "garden"
    result = _new_workspace(
        workspace, pan_videos[:1], "--max-density-gain", "0.5", "--prune-images"
    )
    assert result.exit_code == 1, result.output
    assert "max_density_gain" in result.output
    assert f"The new workspace {workspace} is left as it is" in result.output
    assert len(list((workspace / "top").glob("frame_*.jpg"))) == VIDEO_FRAMES


def test_new_workspace_refusals_list_every_problem_and_create_nothing(
    tmp_path: Path, pan_videos
):
    workspace = tmp_path / "garden"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("")
    image = tmp_path / "still.jpg"
    image.write_bytes(b"")
    result = _new_workspace(workspace, [pan_videos[0], image, tmp_path / "missing.mp4"])
    assert result.exit_code == 1, result.output
    assert "3 problems found, so nothing was created" in result.output
    assert "the new workspace's directory is not empty" in result.output
    assert "still.jpg: an image, not a video" in result.output
    assert "missing.mp4: no such file" in result.output
    assert _files(workspace) == {"notes.txt"}


def test_new_workspace_passes_hdr_exposure_to_the_extraction(
    tmp_path: Path, pan_videos
):
    """`--hdr-exposure` reaches the extraction's checks, which refuse it for
    SDR video before anything is created."""
    workspace = tmp_path / "garden"
    result = _new_workspace(workspace, pan_videos[:1], "--hdr-exposure", "1")
    assert result.exit_code == 1, result.output
    assert "--hdr-exposure sets the exposure of HDR video frames" in result.output
    assert not workspace.exists()


def test_new_workspace_inside_a_workspace_is_refused(tmp_path: Path, pan_videos):
    _init_workspace(tmp_path / "outer")
    workspace = tmp_path / "outer" / "garden"
    result = _new_workspace(workspace, pan_videos[:1])
    assert result.exit_code == 1, result.output
    assert f"is inside the workspace {tmp_path / 'outer'}" in result.output
    assert not workspace.exists()


@pytest.mark.parametrize(
    "args, message",
    [
        (["--named-image-list", "walk"], "A new workspace has no named image lists"),
        (["-r", "walk"], "A new workspace has no named image lists"),
        (["--dsp"], "--dsp/--no-dsp option is only supported for COLMAP"),
        (["--fps", "0"], "--fps"),
        (["--hdr-exposure", "nan"], "--hdr-exposure"),
    ],
)
def test_new_workspace_usage_errors_create_nothing(
    tmp_path: Path, pan_videos, args, message
):
    workspace = tmp_path / "garden"
    result = _new_workspace(workspace, pan_videos[:1], *args)
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert not workspace.exists()


@pytest.mark.parametrize(
    "option, named",
    [
        (["--fps", "5"], "--fps"),
        (["--max-frame-size", "full"], "--max-frame-size"),
        (["--hdr-exposure", "1"], "--hdr-exposure"),
        (["--prune-images"], "--prune-images"),
        (["--max-features", "1024"], "--max-features"),
        (["--no-gpu"], "--gpu/--no-gpu"),
        (["--feature-tool", "sfmtool"], "--feature-tool"),
    ],
)
def test_new_workspace_options_without_it_are_usage_errors(
    pan_workspace: Path, option, named
):
    result = CliRunner().invoke(main, ["pick-images", "frames", *option])
    assert result.exit_code == 2, result.output
    assert f"{named}: only with --new-workspace." in result.output
    assert not (pan_workspace / "frames" / "features").exists()


def test_new_workspace_through_a_linked_directory(tmp_path: Path, pan_videos):
    # DIR reached through a directory link: the list keeps paths under DIR,
    # and the prune keeps the chosen frames.
    (tmp_path / "real").mkdir()
    link = tmp_path / "link"
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(tmp_path / "real"), str(link))
        else:
            link.symlink_to(tmp_path / "real", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"cannot link a directory here: {error}")
    workspace = link / "garden"
    result = _new_workspace(
        workspace,
        pan_videos[:1],
        *FAST,
        *["-r", "5-36", "--write-list", "picked", "--prune-images"],
        *["--json", str(workspace / "r.json")],
    )
    assert result.exit_code == 0, result.output
    (chosen,) = json.loads((workspace / "r.json").read_text())["chosen_numbers"]
    kept = [f"top/frame_{n:06d}.jpg" for n in chosen]
    assert _resolve(workspace, "picked") == kept
    frames = tmp_path / "real" / "garden" / "top"
    assert sorted(p.name for p in frames.glob("frame_*.jpg")) == [
        f"frame_{n:06d}.jpg" for n in chosen
    ]
