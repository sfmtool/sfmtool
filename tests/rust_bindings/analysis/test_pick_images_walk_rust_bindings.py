# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the walk's bindings (``sfmtool.analysis.pick_images`` with
``CaptureImage``, ``CaptureFrame`` and ``CaptureSequence``; see
``specs/core/analysis/pick-images.md``)."""

import _thread
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from sfmtool import ProgressCounter, init_workspace, load_workspace_config
from sfmtool.analysis import (
    CaptureFrame,
    CaptureImage,
    CaptureSequence,
    pick_images,
)
from sfmtool.fileio import read_sift, write_sift
from sfmtool.geometry import CameraIntrinsics
from sfmtool.sift import build_sift_files

WIDTH, HEIGHT, FOCAL = 640, 480, 500.0
# The camera's x motion per frame for a step of about 1% of the long side.
ONE_PERCENT = 0.128


class _Capture:
    """A synthetic video: a wall of points with random descriptors seen by a
    camera moving along x, with every frame's ``.sift`` file written up
    front, for the walk to read."""

    def __init__(self, folder: Path, frames: int = 150, seed: int = 0):
        rng = np.random.default_rng(seed)
        length = ONE_PERCENT * frames
        n = int((length + 24) * 80)
        self.points = np.column_stack(
            [
                rng.uniform(-12, length + 12, n),
                rng.uniform(-4, 4, n),
                rng.uniform(6, 14, n),
            ]
        )
        self.descriptors = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
        self.folder = folder
        self.sequence = CaptureSequence(
            ["frame_%04d.jpg"],
            [
                CaptureFrame(
                    t + 1,
                    [
                        CaptureImage(
                            folder / f"frame_{t + 1:04d}.jpg",
                            folder / f"frame_{t + 1:04d}.jpg.sift",
                        )
                    ],
                )
                for t in range(frames)
            ],
        )

    def write(self, t: int, path: Path) -> None:
        rng = np.random.default_rng(t + 1)
        world = self.points - [ONE_PERCENT * t, 0.0, 0.0]
        u = FOCAL * world[:, 0] / world[:, 2] + WIDTH / 2
        v = FOCAL * world[:, 1] / world[:, 2] + HEIGHT / 2
        inside = (u > 1) & (u < WIDTH - 1) & (v > 1) & (v < HEIGHT - 1)
        positions = np.column_stack([u[inside], v[inside]])
        positions += rng.normal(0, 0.2, positions.shape)
        n = len(positions)
        write_sift(
            path,
            {
                "feature_tool_metadata": {
                    "feature_tool": "sfmtool",
                    "feature_type": "sift",
                    "feature_options": {"image_to_gray": {"formula": "G"}},
                },
                "metadata": {
                    "version": 2,
                    "image_name": path.name.removesuffix(".sift"),
                    "image_file_xxh128": "0" * 32,
                    "image_file_size": 1,
                    "image_width": WIDTH,
                    "image_height": HEIGHT,
                    "feature_count": n,
                    "sharpness": 1.0 + 0.01 * (t % 5),
                    "gradient_structure_tensor_xx_yy_xy": [1.0, 1.0, 0.0],
                },
                "positions_xy": positions.astype(np.float32),
                "affine_shapes": np.tile(np.eye(2, dtype=np.float32), (n, 1, 1)),
                "descriptors": self.descriptors[inside],
                "thumbnail_y_x_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
            },
        )

    def write_all(self, skip: set[int] = frozenset()) -> "_Capture":
        self.folder.mkdir(parents=True, exist_ok=True)
        for t, frame in enumerate(self.sequence.frames):
            if t not in skip:
                self.write(t, frame.images[0].sift_path)
        return self


@pytest.fixture(scope="module")
def steady(tmp_path_factory) -> _Capture:
    """One 150-frame capture with every file written, for the tests that only
    read it."""
    return _Capture(tmp_path_factory.mktemp("steady") / "frames").write_all()


def _read(report: dict) -> list[int]:
    return [f["id"]["frame"] for f in report["frames"]]


def test_a_steady_video_is_walked_from_files_that_exist(steady: _Capture):
    capture = steady
    counter = ProgressCounter()
    report = pick_images([capture.sequence], progress=counter, pilot_vote=False)
    walk = report["sequences"][0]
    chosen = walk["chosen"]
    assert chosen[0] == 0 and chosen[-1] == 149
    strides = np.diff(chosen)
    assert 8 <= np.median(strides) <= 12
    steps = [s["stats"]["step_percent"] for s in walk["steps"]]
    assert 7 <= np.median(steps) <= 13
    # The walk is given no camera and runs no pilot vote, so no pair has an
    # essential split, and the local density term is off, every step at the
    # base target.
    assert all(p["stats"]["essential_split"] is None for p in report["pairs"])
    assert walk["density_term"] == {"status": "off", "reason": "no_camera"}
    assert {json.dumps(s["density"]) for s in walk["steps"]} == {
        json.dumps(
            {
                "target_percent": 10.0,
                "gain": 1.0,
                "median_measure": 1.0,
                "essential_split": None,
                "focal_factor": 1.0,
                "measured_pair": None,
                "measure": 1.0,
            }
        )
    }
    assert walk["swaps"] == [] and walk["pilot_vote"] is None
    read = _read(report)
    assert len(set(read)) == len(read) == walk["frames_read"] < 150
    assert counter.value == len(read)
    # Nothing was extracted: the files were there.
    assert report["frames_extracted"] == [False] * len(read)
    assert report["frames_extracted_unread"] == 0
    # The report is plain data that JSON writes as it is.
    assert json.loads(json.dumps(report)) == report


def test_a_missing_sift_file_is_a_file_not_found_error(tmp_path: Path):
    capture = _Capture(tmp_path / "frames", frames=40).write_all(skip={1})
    with pytest.raises(FileNotFoundError, match="frame_0002.jpg.sift"):
        pick_images([capture.sequence])


def test_options_reach_the_walk(steady: _Capture):
    capture = steady
    report = pick_images([capture.sequence], target_step_percent=20.0, pilot_vote=False)
    strides = np.diff(report["sequences"][0]["chosen"])
    assert 16 <= np.median(strides) <= 24
    # A floor above the matches of neighbouring frames keeps every frame.
    first = CaptureSequence(["frame_%04d.jpg"], capture.sequence.frames[:10])
    report = pick_images([first], min_step_matches=5000, pilot_vote=False)
    assert report["sequences"][0]["chosen"] == list(range(10))
    for name, value in [
        ("target_step_percent", 0.0),
        ("max_density_gain", 0.5),
        ("max_density_gain", float("nan")),
        ("step_band", 1.0),
        ("neighbour_blur_ratio", 1.0),
        ("pair.ratio", 0.0),
        ("min_step_matches", 0),
    ]:
        key = name.split(".")[-1]
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            pick_images([capture.sequence], **{key: value})
    # The forest's parameters are checked as for FrameFeatures.
    with pytest.raises(ValueError, match="num_trees"):
        pick_images([capture.sequence], num_trees=0)
    with pytest.raises(ValueError, match="unknown preset"):
        pick_images([capture.sequence], preset="slow")


def _camera(scale: float) -> CameraIntrinsics:
    """The synthetic capture's pinhole camera at `scale` times its size."""
    return CameraIntrinsics(
        "SIMPLE_PINHOLE",
        int(WIDTH * scale),
        int(HEIGHT * scale),
        {
            "focal_length": FOCAL * scale,
            "principal_point_x": WIDTH / 2 * scale,
            "principal_point_y": HEIGHT / 2 * scale,
        },
    )


def test_a_sequence_s_camera_reaches_the_local_density_term(steady: _Capture):
    # A camera at twice the frames' size is scaled to them; the wall seen
    # while moving sideways has a wide parallax everywhere, so no step's
    # measure counts and the term leaves the walk as it is without it.
    frames = steady.sequence.frames
    with_camera = CaptureSequence(["frame_%04d.jpg"], frames, camera=_camera(2.0))
    assert with_camera.camera == _camera(2.0)
    assert steady.sequence.camera is None
    report = pick_images([with_camera])
    walk = report["sequences"][0]
    assert walk["density_term"] == {"status": "on", "focal_px": FOCAL}
    for step in walk["steps"]:
        split = step["density"]["essential_split"]
        assert split["status"] == "measured" and split["parallax_p50_deg"] > 0.5
        assert step["density"]["gain"] == 1.0 and step["density"]["measure"] == 1.0
    without = pick_images([steady.sequence], pilot_vote=False)["sequences"][0]
    assert walk["chosen"] == without["chosen"]
    # The pilot vote is on by default, and runs only for a sequence with no
    # camera, on the frames the walk without it chooses.
    assert walk["pilot_vote"] is None and without["pilot_vote"] is None
    short = CaptureSequence(["f_%d.jpg"], frames[:8])
    piloted = pick_images([short])["sequences"][0]
    assert (
        piloted["pilot_vote"]["frames"]
        == (pick_images([short], pilot_vote=False)["sequences"][0]["chosen"])
    )
    # A gain of 1 measures nothing; a camera of another aspect is named.
    off = pick_images([with_camera], max_density_gain=1.0)["sequences"][0]
    assert off["density_term"] == {"status": "off", "reason": "max_gain_one"}
    assert off["steps"][0]["density"]["essential_split"] is None
    tall = CameraIntrinsics(
        "SIMPLE_PINHOLE",
        HEIGHT,
        WIDTH,
        {"focal_length": FOCAL, "principal_point_x": 240, "principal_point_y": 320},
    )
    walk = pick_images([CaptureSequence(["f_%d.jpg"], frames[:20], camera=tall)])
    assert walk["sequences"][0]["density_term"] == {
        "status": "off",
        "reason": "camera_shape",
        "camera": [HEIGHT, WIDTH],
        "frames": [WIDTH, HEIGHT],
    }


def test_the_source_options_are_checked(tmp_path: Path):
    capture = _Capture(tmp_path / "frames")
    config_options = {"image_to_gray": {"formula": "G"}}
    with pytest.raises(ValueError, match="together"):
        pick_images([capture.sequence], feature_options=config_options)
    with pytest.raises(ValueError, match="together"):
        pick_images([capture.sequence], feature_prefix_dir="features")
    with pytest.raises(ValueError, match="concurrent_images"):
        pick_images([capture.sequence], concurrent_images=2)
    with pytest.raises(ValueError, match="version 1 settings"):
        pick_images(
            [capture.sequence], feature_options={}, feature_prefix_dir="features"
        )
    with pytest.raises(ValueError, match=r"params\['octave_layers'\]"):
        pick_images(
            [capture.sequence],
            feature_options={**config_options, "octave_layers": 0},
            feature_prefix_dir="features",
        )
    # Nothing was written for any of them.
    assert not capture.folder.exists()


def test_a_signal_while_the_walk_runs_stops_it(steady: _Capture):
    """Ctrl+C while the walk matches, with the GIL released, raises
    ``KeyboardInterrupt`` from the walk."""
    capture = steady
    counter = ProgressCounter()

    def interrupt():
        # The walk's 150 frames take far longer to read and match than
        # reading the first frame does.
        deadline = time.monotonic() + 60
        while counter.value == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        _thread.interrupt_main()

    thread = threading.Thread(target=interrupt)
    thread.start()
    with pytest.raises(KeyboardInterrupt):
        pick_images([capture.sequence], progress=counter)
    thread.join()
    assert 0 < counter.value < 150


def test_errors_map_to_python_exceptions(tmp_path: Path):
    capture = _Capture(tmp_path / "frames", frames=40).write_all()
    with pytest.raises(ValueError, match="no sequences"):
        pick_images([])
    with pytest.raises(ValueError, match="increasing"):
        frames = capture.sequence.frames
        pick_images([CaptureSequence(["a_%d.jpg"], [frames[1], frames[0]])])
    # A file that is not a .sift file raises the OSError of the format errors,
    # with the path.
    capture.sequence.frames[0].images[0].sift_path.write_bytes(b"not a zip")
    with pytest.raises(OSError, match="frame_0001.jpg.sift"):
        pick_images([capture.sequence])
    # A folder in place of which there is a file cannot be listed.
    blocker = tmp_path / "blocker"
    blocker.write_text("a file")
    blocked = CaptureSequence(
        ["x_%d.jpg"],
        [
            CaptureFrame(
                1, [CaptureImage(blocker / "x_1.jpg", blocker / "x_1.jpg.sift")]
            )
        ],
    )
    with pytest.raises(OSError, match="cannot be listed"):
        pick_images([blocked])
    # Version 1 files in a folder the capture's files go in.
    v1_folder = tmp_path / "v1"
    v1 = _Capture(v1_folder, frames=40).write_all()
    _rewrite_as_version_1(v1_folder / "frame_0001.jpg.sift")
    with pytest.raises(ValueError, match="version 1 .sift files"):
        pick_images([v1.sequence])


def _rewrite_as_version_1(path: Path) -> None:
    data = read_sift(path)
    data["metadata"]["version"] = 1
    del data["metadata"]["sharpness"]
    del data["metadata"]["gradient_structure_tensor_xx_yy_xy"]
    data["feature_tool_metadata"]["feature_options"] = {}
    write_sift(path, data)


def test_capture_classes():
    image = CaptureImage("a/frame_0001.jpg", Path("a/f/frame_0001.jpg.sift"))
    assert image.image_path == Path("a/frame_0001.jpg")
    assert image.sift_path == Path("a/f/frame_0001.jpg.sift")
    frame = CaptureFrame(1, [image])
    assert frame.number == 1 and frame.images == (image,)
    sequence = CaptureSequence(["frame_%04d.jpg"], [frame])
    assert sequence.patterns == ["frame_%04d.jpg"]
    assert sequence.frames == (frame,) and len(sequence) == 1
    with pytest.raises(TypeError):
        CaptureFrame(1, ["not an image"])


def test_photographs_extracted_by_the_walk_are_every_one_kept(
    isolated_seoul_bull_17_images: list[Path], tmp_path: Path
):
    """Real images extracted by the walk with the workspace's settings: the
    17 photographs of a walk around a sculpture match their neighbours with
    fewer matches than the floor, so every one is kept; each file is the one
    ``build_sift_files`` writes, and a second run extracts nothing."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    images = []
    for path in isolated_seoul_bull_17_images:
        images.append(Path(shutil.copy(path, workspace / path.name)))
    init_workspace(workspace)
    config = load_workspace_config(workspace)
    features = workspace / config["feature_prefix_dir"]
    sequence = CaptureSequence(
        ["seoul_bull_sculpture_%02d.jpg"],
        [
            CaptureFrame(t + 1, [CaptureImage(p, features / f"{p.name}.sift")])
            for t, p in enumerate(images)
        ],
    )
    extract = {
        "feature_options": config["feature_options"],
        "feature_prefix_dir": config["feature_prefix_dir"],
        "concurrent_images": 4,
        "pilot_vote": False,
    }
    report = pick_images([sequence], **extract)
    walk = report["sequences"][0]
    # Every step to the next frame keeps fewer matches than the floor, so
    # every frame is chosen.
    assert all(s["stats"]["matches"] < 600 for s in walk["steps"])
    assert walk["chosen"] == list(range(17))
    read = _read(report)
    # Every frame is read, and each was extracted.
    assert sorted(read) == list(range(17))
    assert report["frames_extracted"] == [True] * 17
    assert report["frames_extracted_unread"] == 0
    assert sorted(p.name for p in features.glob("*.sift")) == sorted(
        f"{p.name}.sift" for p in images
    )

    copies = tmp_path / "copies"
    copies.mkdir()
    copied = [Path(shutil.copy(p, copies / p.name)) for p in images]
    sift_paths, _, _ = build_sift_files(
        copied, config["feature_options"], config["feature_prefix_dir"]
    )
    for image, built in zip(images, sift_paths):
        walked = read_sift(features / f"{image.name}.sift")
        expected = read_sift(built)
        assert walked["metadata"] == expected["metadata"]
        assert np.array_equal(walked["descriptors"], expected["descriptors"])

    # The four workers took up every frame read, and the walk's thread
    # waited for them.
    times = report["times"]
    assert times["source"]["workers"] == 4
    assert times["source"]["frames"] >= 17
    assert times["waiting_for_frames_seconds"] > 0

    again = pick_images([sequence], **extract)
    assert again["frames_extracted"] == [False] * 17
    # The times differ from run to run; the rest is the same.
    differ = {"frames_extracted", "times"}
    assert {k: v for k, v in again.items() if k not in differ} == {
        k: v for k, v in report.items() if k not in differ
    }
    # The second run's workers found every file up to date.
    assert again["times"]["source"]["seconds"] < times["source"]["seconds"]
