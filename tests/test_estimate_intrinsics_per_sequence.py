# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `sfm estimate-intrinsics --write-camrig-per-sequence`.

The rules for grouping, naming and refusing are covered against synthetic
estimates and image files of a few bytes, so those cases stay cheap. The
end-to-end tests split the seoul bull images into sequences of both layouts,
a folder per sequence and a name prefix per sequence in one folder, match
them once with `sfm match --cluster`, and check that each sequence gets
a `.camrig` holding the estimate `-r` gives for its images when that
estimate can be written, and no file when it cannot, and that the database
`sfm solve` builds from the `.matches` file gives each sequence with a
`.camrig` its own camera.
"""

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from click.testing import CliRunner

import sfmtool._sfmtool.geometry as _geometry
from sfmtool.cli import main
from sfmtool.fileio import read_camrig, write_camrig

ei = importlib.import_module("sfmtool._commands.estimate_intrinsics")

SEOUL_DIR = (
    Path(__file__).parent.parent / "test-data" / "images" / "seoul_bull_sculpture"
)


def _estimate(focal_px=300.0, camera_model="Pinhole", confirmed=None) -> dict:
    """An `estimate_intrinsics` result with the fields the command reads."""
    return {
        "camera_model": camera_model,
        "confirmed": confirmed,
        "focal_px": focal_px,
        "verdict_votes": [],
        "escalation": None,
        "screening_vote": None,
        "vote": {
            "focal_px": focal_px,
            "family": "Epipolar",
            "n_epipolar": 6,
            "n_rotation": 0,
            "n_pool": 6,
            "pool_spread": 0.05,
            "family_disagreement": None,
            "n_h_dominated": 0,
            "n_estimator_failed": 0,
            "n_band_rejected": 0,
            "camera_model": camera_model,
            "columns": [],
        },
    }


@pytest.fixture
def stub_run(monkeypatch, tmp_path):
    """Run `--write-camrig-per-sequence` on a workspace of image files named
    `names` and canned estimates, with no real `.matches` file.

    `estimate_for` maps the names of a sequence's selected images to its
    estimate, `size_of` an image's name to its `(width, height)`, and `kept`
    is the names `-r` keeps, `None` for all. The workspace is
    `tmp_path / "ws"`.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    matches_path = tmp_path / "clusters.matches"
    matches_path.write_bytes(b"not read")

    def run(
        names,
        *args,
        estimate_for=lambda chosen: _estimate(),
        size_of=lambda name: (640, 480),
        kept=None,
    ):
        for name in names:
            (workspace / name).parent.mkdir(parents=True, exist_ok=True)
            (workspace / name).write_bytes(b"jpeg")
        monkeypatch.setattr(
            ei,
            "_open_matches",
            lambda path, selection: (None, list(names), kept, None, workspace),
        )

        def select(mfile, restrict, data):
            # The command's own check of the selected images' sizes, on the
            # size `size_of` gives each.
            width, height = ei._resolve_dimensions(
                [size_of(name) for name in restrict], list(restrict)
            )
            return tuple(restrict), {
                **data,
                "image_names": list(restrict),
                "width": width,
                "height": height,
                "cluster_count": 1,
                "observation_count": 2,
            }

        monkeypatch.setattr(ei, "_select", select)
        monkeypatch.setattr(
            _geometry,
            "estimate_intrinsics",
            lambda selection, **k: estimate_for(list(selection)),
        )
        return CliRunner().invoke(
            main,
            [
                "estimate-intrinsics",
                "-i",
                str(matches_path),
                "--write-camrig-per-sequence",
                *args,
            ],
        )

    run.workspace = workspace
    return run


def _camrig(path: Path) -> tuple[str, float]:
    """The stored pattern and the focal of a one-sensor `.camrig`."""
    data = read_camrig(path)
    (pattern,) = data["sensor_image_patterns"]
    return pattern, data["cameras"][0]["parameters"]["focal_length"]


def _write_one_sensor_camrig(path: Path, pattern: str) -> None:
    write_camrig(
        path=str(path),
        name=path.stem,
        rig_type="generic",
        cameras=[
            {
                "model": "SIMPLE_PINHOLE",
                "width": 640,
                "height": 480,
                "parameters": {
                    "focal_length": 500.0,
                    "principal_point_x": 320.0,
                    "principal_point_y": 240.0,
                },
            }
        ],
        sensor_image_patterns=[pattern],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.zeros((1, 3)),
    )


# ── Naming and patterns ──────────────────────────────────────────────────────


def test_names_and_patterns_follow_the_sequences(stub_run):
    """Each file is named after the part of the names before the number,
    less its separators, or after the folder; its pattern is the names with
    the number as a frame field of the sequence's width, `%%` for `%`."""
    names = [
        "shot1_0001.jpg",
        "shot1_0002.jpg",
        "shot10_0001.jpg",
        "shot10_0002.jpg",
        "cam/clip (1).jpg",
        "cam/clip (12).jpg",
        "cam/50%-01.jpg",
        "cam/50%-02.jpg",
        "vid/00001.jpg",
        "vid/00002.jpg",
        "00001.png",
        "00002.png",
    ]
    out = stub_run(names)
    assert out.exit_code == 0, out.output
    ws = stub_run.workspace
    expected = {
        ws / "shot1.camrig": "shot1_%04d.jpg",
        ws / "shot10.camrig": "shot10_%04d.jpg",
        ws / "cam" / "clip.camrig": "clip (%d).jpg",
        ws / "cam" / "50%.camrig": "50%%-%02d.jpg",
        ws / "vid" / "vid.camrig": "%05d.jpg",
        # A sequence in the workspace root is named after the workspace.
        ws / "ws.camrig": "%05d.png",
    }
    for path, pattern in expected.items():
        assert path.exists(), path
        assert _camrig(path)[0] == pattern
    assert sorted(p.name for p in ws.rglob("*.camrig")) == sorted(
        p.name for p in expected
    )
    assert "Sequence shot1_%04d.jpg (2 images)" in out.output
    assert "pattern:  shot10_%04d.jpg" in out.output


def test_each_sequence_votes_on_its_own_images(stub_run):
    names = ["images/a_01.jpg", "images/b_01.jpg", "images/a_02.jpg"]
    focal = {"images/a_01.jpg": 700.0, "images/b_01.jpg": 1100.0}
    seen = []

    def estimate_for(chosen):
        seen.append(chosen)
        return _estimate(focal_px=focal[chosen[0]])

    out = stub_run(names, estimate_for=estimate_for)
    assert out.exit_code == 0, out.output
    assert seen == [["images/a_01.jpg", "images/a_02.jpg"], ["images/b_01.jpg"]]
    images = stub_run.workspace / "images"
    assert _camrig(images / "a.camrig") == ("a_%02d.jpg", 700.0)
    assert _camrig(images / "b.camrig") == ("b_%02d.jpg", 1100.0)


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_a_sequence_without_a_writable_estimate_gets_no_file(stub_run):
    names = ["a_01.jpg", "a_02.jpg", "b_01.jpg", "b_02.jpg", "c_01.jpg", "c_02.jpg"]
    estimates = {
        "a_01.jpg": _estimate(focal_px=None, camera_model=None),
        "b_01.jpg": _estimate(camera_model="EquidistantFisheye", confirmed=False),
        "c_01.jpg": _estimate(focal_px=410.0),
    }
    out = stub_run(names, estimate_for=lambda chosen: estimates[chosen[0]])
    assert out.exit_code == 0, out.output
    ws = stub_run.workspace
    assert not (ws / "a.camrig").exists()
    assert not (ws / "b.camrig").exists()
    assert _camrig(ws / "c.camrig") == ("c_%02d.jpg", 410.0)
    assert "No .camrig: no focal consensus to write" in out.output
    assert "No .camrig: refusing to write an UNCONFIRMED Fisheye" in out.output

    # With no file written at all, the report still prints and the command
    # fails.
    out = stub_run(
        ["a_01.jpg", "a_02.jpg"], estimate_for=lambda chosen: estimates["a_01.jpg"]
    )
    assert out.exit_code == 1
    assert "Focal length:  no consensus" in out.output
    assert "Error: no .camrig was written." in out.output


def test_a_sequence_of_two_resolutions_gets_no_file(stub_run):
    names = ["a_01.jpg", "a_02.jpg", "b_01.jpg", "b_02.jpg"]
    out = stub_run(
        names,
        size_of=lambda name: (1280, 720) if name == "a_02.jpg" else (640, 480),
    )
    assert out.exit_code == 0, out.output
    ws = stub_run.workspace
    assert not (ws / "a.camrig").exists()
    assert (ws / "b.camrig").exists()
    assert "No .camrig: the selected images of this .matches file have more" in (
        out.output
    )
    assert "1280x720  (1 image(s))  e.g. a_02.jpg" in out.output


def test_images_that_cannot_be_grouped_stop_the_command(stub_run):
    out = stub_run(["a_01.jpg", "a_02.jpg", "notes.jpg"])
    assert out.exit_code == 1
    assert "cannot be grouped into sequences" in out.output
    assert "1 image file(s) have no file number" in out.output
    assert "notes.jpg" in out.output
    assert not list(stub_run.workspace.rglob("*.camrig"))


def test_sequences_sharing_a_name_take_the_part_after_the_number(stub_run):
    """`number_from_filename` reads the digits at the end of a name, so the
    part after the number is the extension, behind a `)` for a number in
    parentheses."""
    names = [
        # x.camrig for both, so each takes its extension: x_jpg, x_png.
        "x_01.jpg",
        "x_02.jpg",
        "x_01.png",
        "x_02.png",
        # Named from the folder when nothing comes before the number.
        "cams/01.jpg",
        "cams/01.png",
        # y_jpg.camrig, the renamed file of y_%02d.jpg, is already the file
        # of y_jpg_%02d.jpg, so neither of those two gets one.
        "y_01.jpg",
        "y_01.png",
        "y_jpg_01.jpg",
        # Both v.camrig, and both still v_jpg.camrig after the number.
        "v (1).jpg",
        "v (2).jpg",
        "v1.jpg",
        "v2.jpg",
        "w_01.jpg",
    ]
    out = stub_run(names)
    assert out.exit_code == 0, out.output
    ws = stub_run.workspace
    written = {
        p.relative_to(ws).as_posix(): _camrig(p)[0] for p in ws.rglob("*.camrig")
    }
    assert written == {
        "x_jpg.camrig": "x_%02d.jpg",
        "x_png.camrig": "x_%02d.png",
        "cams/cams_jpg.camrig": "%02d.jpg",
        "cams/cams_png.camrig": "%02d.png",
        "y_png.camrig": "y_%02d.png",
        "w.camrig": "w_%02d.jpg",
    }
    for pattern, other in (
        ("y_%02d.jpg", "y_jpg_%02d.jpg"),
        ("y_jpg_%02d.jpg", "y_%02d.jpg"),
        ("v (%d).jpg", "v%d.jpg"),
        ("v%d.jpg", "v (%d).jpg"),
    ):
        stem = "y_jpg" if pattern.startswith("y") else "v_jpg"
        assert (
            f"No .camrig: {ws / (stem + '.camrig')} would also be the .camrig of "
            f"{other}, even named with the part of the file names after the "
            "number." in out.output
        ), pattern


def test_r_is_applied_before_the_images_are_grouped(stub_run):
    """Images `-r` leaves out need not form sequences: one with no number
    and two with the same number do not stop the run."""
    names = ["a_01.jpg", "a_02.jpg", "a_03.jpg", "notes.jpg", "b_1.jpg", "b_01.jpg"]
    out = stub_run(names, "--json", kept=["a_01.jpg", "a_02.jpg"])
    assert out.exit_code == 0, out.output
    (entry,) = json.loads(out.stdout)["sequences"]
    assert entry["pattern"] == "a_%02d.jpg"
    # The frame -r left out is still the sequence's, and its pattern's.
    assert entry["image_count"] == 3
    assert entry["selected_image_count"] == 2
    assert entry["written"] is True
    assert _camrig(stub_run.workspace / "a.camrig")[0] == "a_%02d.jpg"


def test_under_r_the_frame_field_comes_from_the_whole_sequence(stub_run):
    """The selected numbers alone are written with 5 digits, but the
    sequence's other frame is zero-padded to 4, so the stored pattern is
    `%04d`, which matches every frame of the sequence."""
    from sfmtool.camrig.pattern import match_pattern

    names = ["x_0005.jpg", "x_10000.jpg", "x_10001.jpg"]
    out = stub_run(names, "--json", kept=["x_10000.jpg", "x_10001.jpg"])
    assert out.exit_code == 0, out.output
    (entry,) = json.loads(out.stdout)["sequences"]
    assert entry["pattern"] == "x_%04d.jpg"
    assert entry["camrig_pattern"] == "x_%04d.jpg"
    assert entry["image_count"] == 3
    ws = stub_run.workspace
    pattern = _camrig(ws / "x.camrig")[0]
    assert pattern == "x_%04d.jpg"
    assert [p.name for p in match_pattern(ws, pattern)] == names


def test_a_sequence_zero_padded_to_two_widths_gets_no_file(stub_run):
    """Under `-r` the selected images are padded alike, but a frame `-r` left
    out is padded otherwise: no frame field matches them all, so the
    sequence gets no `.camrig` and the reason names the frame."""
    names = ["a_01.jpg", "a_02.jpg", "a_003.jpg", "b_01.jpg", "b_02.jpg"]
    kept = ["a_01.jpg", "a_02.jpg", "b_01.jpg", "b_02.jpg"]
    out = stub_run(names, "--json", kept=kept)
    assert out.exit_code == 0, out.output
    a, b = json.loads(out.stdout)["sequences"]
    assert a["pattern"] == "a_%02d.jpg"
    assert a["written"] is False
    assert "zero-padded in ways no one frame field writes" in a["reason"]
    assert "1 of them do not match a_%02d.jpg: a_003.jpg" in a["reason"]
    assert b["written"] is True
    assert not (stub_run.workspace / "a.camrig").exists()


def test_existing_files_are_kept_unless_forced(stub_run):
    names = ["a_01.jpg", "a_02.jpg", "b_01.jpg", "b_02.jpg"]
    ws = stub_run.workspace
    out = stub_run(names)
    assert out.exit_code == 0, out.output
    (ws / "b.camrig").unlink()

    out = stub_run(names, estimate_for=lambda chosen: _estimate(focal_px=555.0))
    assert out.exit_code == 0, out.output
    assert f"No .camrig: {ws / 'a.camrig'} already exists; pass --force" in out.output
    assert _camrig(ws / "a.camrig")[1] == 300.0
    assert _camrig(ws / "b.camrig")[1] == 555.0

    out = stub_run(names, "--force", estimate_for=lambda c: _estimate(focal_px=555.0))
    assert out.exit_code == 0, out.output
    assert _camrig(ws / "a.camrig")[1] == 555.0


def test_a_sequence_another_camrig_covers_gets_no_file(stub_run):
    names = ["images/a_01.jpg", "images/a_02.jpg", "images/b_01.jpg"]
    images = stub_run.workspace / "images"
    images.mkdir()
    _write_one_sensor_camrig(images / "old.camrig", "a_*.jpg")
    out = stub_run(names)
    assert out.exit_code == 0, out.output
    assert not (images / "a.camrig").exists()
    assert (images / "b.camrig").exists()
    assert f"No .camrig: {images / 'old.camrig'} already covers 2 of its images" in (
        out.output
    )


def test_a_sequence_a_rig_camrig_partly_covers_gets_no_file(stub_run):
    """The rig's patterns cover frames 1 to 9 of each sensor. With -r keeping
    frames 10 and 11 of cam0, those form a sequence of single images whose
    pattern, f_%02d.jpg, also covers the rig's cam0/f_01.jpg, which -r left
    out and whose name puts it in that sequence."""
    names = ["rig/cam0/f_01.jpg", "rig/cam1/f_01.jpg"]
    names += ["rig/cam0/f_10.jpg", "rig/cam0/f_11.jpg"]
    rig = stub_run.workspace / "rig"
    rig.mkdir()
    write_camrig(
        path=str(rig / "pair.camrig"),
        name="pair",
        rig_type="generic",
        cameras=[
            {
                "model": "SIMPLE_PINHOLE",
                "width": 640,
                "height": 480,
                "parameters": {
                    "focal_length": 500.0,
                    "principal_point_x": 320.0,
                    "principal_point_y": 240.0,
                },
            }
        ],
        sensor_image_patterns=["cam0/f_0%d.jpg", "cam1/f_0%d.jpg"],
        camera_indexes=[0, 0],
        quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        translations_xyz=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )
    out = stub_run(names, kept=["rig/cam0/f_10.jpg", "rig/cam0/f_11.jpg"])
    assert out.exit_code == 1, out.output
    # The rig, none of whose images -r keeps, is not a sequence of the run.
    assert "1 sequence of " in out.output
    assert "Sequence rig/cam0/f_%02d.jpg (3 images)" in out.output
    assert f"No .camrig: {rig / 'pair.camrig'} already covers 1 of its images" in (
        out.output
    )
    assert not (rig / "cam0" / "f.camrig").exists()


def test_a_pattern_covering_another_sequence_is_refused(stub_run, monkeypatch):
    """A pattern is checked against the file's images of the other sequences.
    Two sequences' patterns differ in their literal text, so the matcher can
    only make one cover the other's files by ignoring letter case, where the
    file system does; the test stands that in by a matcher that also returns
    an image of `b`."""
    import sfmtool.camrig.create as create

    real = create.find_images

    def find_images(root, pattern):
        found = real(root, pattern)
        return found + (["b_01.jpg"] if pattern.startswith("a_") else [])

    monkeypatch.setattr(create, "find_images", find_images)
    out = stub_run(["a_01.jpg", "a_02.jpg", "b_01.jpg", "b_02.jpg"])
    assert out.exit_code == 0, out.output
    ws = stub_run.workspace
    assert not (ws / "a.camrig").exists()
    assert (ws / "b.camrig").exists()
    assert "No .camrig: pattern 'a_%02d.jpg' matches 1 image(s) of " in out.output
    assert "in other sequences, such as b_01.jpg" in out.output


def test_a_rig_sequence_gets_no_file(stub_run):
    names = [
        "rig/cam0/f_01.jpg",
        "rig/cam1/f_01.jpg",
        "rig/cam0/f_02.jpg",
        "rig/cam1/f_02.jpg",
        "solo_01.jpg",
        "solo_02.jpg",
    ]
    rig = stub_run.workspace / "rig"
    rig.mkdir()
    write_camrig(
        path=str(rig / "pair.camrig"),
        name="pair",
        rig_type="generic",
        cameras=[
            {
                "model": "SIMPLE_PINHOLE",
                "width": 640,
                "height": 480,
                "parameters": {
                    "focal_length": 500.0,
                    "principal_point_x": 320.0,
                    "principal_point_y": 240.0,
                },
            }
        ],
        sensor_image_patterns=["cam0/f_%02d.jpg", "cam1/f_%02d.jpg"],
        camera_indexes=[0, 0],
        quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        translations_xyz=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )
    out = stub_run(names, "--json")
    assert out.exit_code == 0, out.output
    payload = json.loads(out.stdout)
    rig_entry, solo_entry = payload["sequences"]
    assert rig_entry["pattern"] == "rig/cam0/f_%02d.jpg and rig/cam1/f_%02d.jpg"
    assert rig_entry["written"] is False
    assert rig_entry["estimate"] is None
    assert rig_entry["reason"].startswith("a rig sequence")
    assert solo_entry["written"] is True
    assert sorted(p.name for p in stub_run.workspace.rglob("*.camrig")) == [
        "pair.camrig",
        "solo.camrig",
    ]


def test_a_rig_of_rig_config_json_is_refused(stub_run):
    names = ["rig/a/f_01.jpg", "rig/b/f_01.jpg", "solo_01.jpg", "solo_02.jpg"]
    (stub_run.workspace / "rig_config.json").write_text(
        json.dumps(
            [
                {
                    "cameras": [
                        {"image_prefix": "rig/a/", "ref_sensor": True},
                        {
                            "image_prefix": "rig/b/",
                            "cam_from_rig_rotation": [1, 0, 0, 0],
                            "cam_from_rig_translation": [0.1, 0, 0],
                        },
                    ]
                }
            ]
        )
    )
    out = stub_run(names)
    assert out.exit_code == 1
    assert "rig/a/f_01.jpg belongs to a rig of rig_config.json" in out.output
    assert not list(stub_run.workspace.rglob("*.camrig"))
    # A solve of the file would lose the rig whatever -r selects here.
    out = stub_run(names, kept=["solo_01.jpg", "solo_02.jpg"])
    assert out.exit_code == 1
    assert "rig/a/f_01.jpg belongs to a rig of rig_config.json" in out.output
    assert not list(stub_run.workspace.rglob("*.camrig"))


@pytest.mark.parametrize(
    "extra", [["--write-camrig", "x.camrig"], ["--pattern", "*.jpg"]]
)
def test_other_write_options_are_refused(stub_run, extra):
    out = stub_run(["a_01.jpg", "a_02.jpg"], *extra)
    assert out.exit_code == 2
    assert f"cannot be combined with {extra[0]}" in out.output
    assert not list(stub_run.workspace.rglob("*.camrig"))


# ── End to end ───────────────────────────────────────────────────────────────

# The split of the seoul bull images: a folder of plain numbers, two name
# prefixes in one folder, and a lone image whose sequence has no pair to vote.
_LAYOUT = (
    [(f"vidA/{n:04d}.jpg", n) for n in range(1, 7)]
    + [(f"images/shopB_{n:04d}.jpg", n) for n in range(7, 13)]
    + [(f"images/shopC_{n:04d}.jpg", n) for n in range(13, 18)]
    + [("images/lone_0001.jpg", 1)]
)
_SEQUENCES = {
    "images/lone_%04d.jpg": None,
    "images/shopB_%04d.jpg": ("images/shopB.camrig", "shopB_%04d.jpg"),
    "images/shopC_%04d.jpg": ("images/shopC.camrig", "shopC_%04d.jpg"),
    "vidA/%04d.jpg": ("vidA/vidA.camrig", "%04d.jpg"),
}


@pytest.fixture(scope="module")
def split_capture(tmp_path_factory) -> Path:
    """A workspace of the split seoul bull images, matched once with
    `sfm match --cluster` into `matches/clusters.matches`."""
    workspace = tmp_path_factory.mktemp("split") / "ws"
    for name, number in _LAYOUT:
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            SEOUL_DIR / f"seoul_bull_sculpture_{number:02d}.jpg", workspace / name
        )
    (workspace / ".sfm-named-image-lists.json").write_text(
        json.dumps(
            {
                "version": 1,
                "named_image_lists": {
                    "vidA": [{"pattern": "vidA/%04d.jpg"}],
                },
            }
        )
    )
    runner = CliRunner()
    for args in (
        ["ws", "init", str(workspace)],
        ["sift", "--extract", str(workspace)],
        [
            "match",
            "--cluster",
            "--output",
            str(workspace / "matches" / "clusters.matches"),
            str(workspace),
        ],
    ):
        result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output
    return workspace


def _copy(split_capture: Path, dest: Path) -> Path:
    """A copy of the split capture, to write `.camrig` files into."""
    return Path(shutil.copytree(split_capture, dest))


def _invoke(workspace: Path, *args: str):
    return CliRunner().invoke(
        main,
        [
            "estimate-intrinsics",
            "-i",
            str(workspace / "matches" / "clusters.matches"),
            *args,
        ],
    )


def _vote(workspace: Path, *r_args: str) -> dict:
    """The `--json` estimate `-r` gives the selected images."""
    out = _invoke(workspace, *r_args, "--json")
    assert out.exit_code == 0, out.output
    return json.loads(out.stdout)


def _refusal(vote: dict) -> str | None:
    """The start of the reason a sequence whose vote is `vote` gets no
    `.camrig`, or `None` when the vote is one the command writes.

    The votes of these sequences of three to six images pool none to five
    pair votes, against a minimum of two, and the orientations and
    descriptors of the `.sift` files differ between platforms, so whether a
    vote reaches a focal differs between platforms too: `-r 13-15` pools
    three votes on Windows and one on Linux. The tests expect each
    sequence's outcome to be the one its `-r` vote gives on the platform
    they run on, rather than a written file."""
    if vote["focal_px"] is None:
        return "no focal consensus to write"
    if vote["fisheye_confirmed"] is False:
        return "refusing to write an UNCONFIRMED Fisheye estimate"
    return None


def test_each_sequence_gets_the_estimate_r_gives_its_images(split_capture, tmp_path):
    """Both cases share the one `sfm match --cluster` run of the module's
    fixture, which is most of the test's time."""
    _check_r_selects_the_sequences_and_their_images(
        _copy(split_capture, tmp_path / "r")
    )

    capture = _copy(split_capture, tmp_path / "ws")
    selections = {
        "images/shopB_%04d.jpg": ("-r", "7-12"),
        "images/shopC_%04d.jpg": ("-r", "13-17"),
        "vidA/%04d.jpg": ("-r", "vidA"),
    }
    votes = {pattern: _vote(capture, *r_args) for pattern, r_args in selections.items()}
    out = _invoke(capture, "--write-camrig-per-sequence", "--json")
    writable = [p for p, vote in votes.items() if _refusal(vote) is None]
    assert out.exit_code == (0 if writable else 1), out.output
    payload = json.loads(out.stdout)
    assert payload["image_count"] == 18
    assert payload["selected_image_count"] == 18
    entries = {entry["pattern"]: entry for entry in payload["sequences"]}
    assert list(entries) == list(_SEQUENCES)

    lone = entries["images/lone_%04d.jpg"]
    assert lone["written"] is False
    assert lone["estimate"]["focal_px"] is None
    assert lone["reason"].startswith("no focal consensus to write")
    assert not (capture / "images" / "lone.camrig").exists()

    for pattern, expected in votes.items():
        entry = entries[pattern]
        camrig_path, camrig_pattern = _SEQUENCES[pattern]
        assert entry["camrig_path"] == camrig_path
        assert entry["camrig_pattern"] == camrig_pattern
        # The sequence's vote is the one `-r` runs on the same images; only
        # the count of the file's images differs, the sequence's against
        # the whole file's.
        assert entry["estimate"]["image_count"] == entry["image_count"]
        assert expected["image_count"] == 18
        del expected["image_count"], entry["estimate"]["image_count"]
        assert entry["estimate"] == expected, pattern
        refusal = _refusal(expected)
        if refusal is not None:
            assert entry["written"] is False, pattern
            assert entry["reason"].startswith(refusal), entry["reason"]
            assert not (capture / camrig_path).exists()
            continue
        assert entry["written"] is True, entry["reason"]
        stored = read_camrig(capture / camrig_path)
        assert list(stored["sensor_image_patterns"]) == [camrig_pattern]
        assert stored["cameras"][0]["parameters"]["focal_length"] == pytest.approx(
            expected["focal_px"]
        )
        # Under --json the account of each written file goes to stderr.
        assert f"Wrote {capture / camrig_path}" in out.stderr

    # The database `sfm solve` builds from the file gives each written
    # sequence's images the camera of its .camrig, which no other image has.
    from sfmtool.colmap.db_setup import _setup_for_sfm_from_matches

    db_path, _, _, rig_used = _setup_for_sfm_from_matches(
        capture / "matches" / "clusters.matches", capture / "colmap"
    )
    assert not rig_used
    with pycolmap.Database.open(db_path) as db:
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        camera_of = {i.name: i.camera_id for i in db.read_all_images()}
    for pattern in writable:
        prefix = pattern.split("%")[0]
        ids = {cid for name, cid in camera_of.items() if name.startswith(prefix)}
        (camera_id,) = ids
        assert cameras[camera_id].params[0] == pytest.approx(votes[pattern]["focal_px"])
        others = {cid for name, cid in camera_of.items() if not name.startswith(prefix)}
        assert camera_id not in others, pattern


def _check_r_selects_the_sequences_and_their_images(capture: Path) -> None:
    """With -r only the selected images vote, grouped into sequences."""
    # The votes on the selected images alone.
    votes = {
        "images/shopB_%04d.jpg": _vote(capture, "-r", "8-12"),
        "images/shopC_%04d.jpg": _vote(capture, "-r", "13-15"),
    }
    out = _invoke(capture, "--write-camrig-per-sequence", "-r", "8-15")
    writable = [p for p, vote in votes.items() if _refusal(vote) is None]
    assert out.exit_code == (0 if writable else 1), out.output
    assert "2 sequences of " in out.output
    assert "Images:        5 of 6 selected (--range 8-15)" in out.output
    assert "Images:        3 of 5 selected (--range 8-15)" in out.output
    assert "Sequence vidA/" not in out.output
    assert "Sequence images/lone_" not in out.output
    assert sorted(
        p.relative_to(capture).as_posix() for p in capture.rglob("*.camrig")
    ) == sorted(_SEQUENCES[p][0] for p in writable), out.output
    for pattern, vote in votes.items():
        camrig_path, camrig_pattern = _SEQUENCES[pattern]
        refusal = _refusal(vote)
        if refusal is not None:
            assert f"No .camrig: {refusal}" in out.output, out.output
            continue
        # The written camera is the vote on the selected images alone, and
        # its pattern names the whole sequence, so it still covers the
        # frames -r left out.
        assert _camrig(capture / camrig_path) == (
            camrig_pattern,
            pytest.approx(vote["focal_px"]),
        ), out.output
