# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the walk's bindings (``sfmtool.analysis.pick_images`` with
``CaptureImage``, ``CaptureFrame``, ``CaptureSequence`` and ``FrameId``; see
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
    FrameId,
    pick_images,
)
from sfmtool.fileio import write_sift

WIDTH, HEIGHT, FOCAL = 640, 480, 500.0
# The camera's x motion per frame for a step of about 1% of the long side.
ONE_PERCENT = 0.128


class _Capture:
    """A synthetic video: a wall of points with random descriptors seen by a
    camera moving along x, whose ``.sift`` files are written only when the
    walk asks for them."""

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
        self.calls: list[list[FrameId]] = []

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

    def __call__(self, frames: list[FrameId]) -> None:
        self.calls.append(list(frames))
        self.folder.mkdir(parents=True, exist_ok=True)
        for f in frames:
            image = self.sequence.frames[f.frame].images[0]
            self.write(f.frame, image.sift_path)


def test_a_steady_video_is_walked(tmp_path: Path):
    capture = _Capture(tmp_path / "frames")
    counter = ProgressCounter()
    report = pick_images(
        [capture.sequence], capture, baseline_frames=20, progress=counter
    )
    assert report["verdict"] == "solvable"
    walk = report["sequences"][0]["walk"]
    chosen = walk["chosen"]
    assert chosen[0] == 0 and chosen[-1] == 149
    strides = np.diff(chosen)
    assert 8 <= np.median(strides) <= 12
    steps = [s["stats"]["step_percent"] for s in walk["steps"]]
    assert 7 <= np.median(steps) <= 13
    assert walk["breaks"] == [] and walk["swaps"] == []
    # The callable was handed FrameId objects, each frame once, and exactly
    # those frames were read and have .sift files.
    asked = [f for call in capture.calls for f in call]
    assert all(isinstance(f, FrameId) for f in asked)
    assert len(set(asked)) == len(asked)
    read = [FrameId(f["id"]["sequence"], f["id"]["frame"]) for f in report["frames"]]
    assert read == asked
    assert sorted(p.name for p in capture.folder.glob("*.sift")) == sorted(
        f"frame_{f.frame + 1:04d}.jpg.sift" for f in asked
    )
    assert counter.value == len(asked) < 150
    # The report is plain data that JSON writes as it is.
    assert json.loads(json.dumps(report)) == report


def test_options_reach_the_walk(tmp_path: Path):
    capture = _Capture(tmp_path / "frames")
    report = pick_images(
        [capture.sequence], capture, baseline_frames=20, target_step_percent=20.0
    )
    strides = np.diff(report["sequences"][0]["walk"]["chosen"])
    assert 16 <= np.median(strides) <= 24
    calls = len(capture.calls)
    for name, value in [
        ("target_step_percent", 0.0),
        ("step_band", 1.0),
        ("blur_ratio", 0.0),
        ("baseline_frames", 1),
        ("min_inlier_share", 2.0),
        ("pair.ratio", 0.0),
        ("pair.min_matches", 3),
    ]:
        key = name.split(".")[-1]
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            pick_images([capture.sequence], capture, **{key: value})
    # The forest's parameters are checked as for FrameFeatures.
    with pytest.raises(ValueError, match="num_trees"):
        pick_images([capture.sequence], capture, num_trees=0)
    with pytest.raises(ValueError, match="unknown preset"):
        pick_images([capture.sequence], capture, preset="slow")
    # Refused before anything is extracted.
    assert len(capture.calls) == calls


def test_the_callables_exception_is_raised_unchanged(tmp_path: Path):
    capture = _Capture(tmp_path / "frames")

    class ExtractorFailed(Exception):
        pass

    raised = ExtractorFailed("the extractor ran out of disk")
    calls = []

    def failing(frames):
        calls.append(frames)
        raise raised

    with pytest.raises(ExtractorFailed) as excinfo:
        pick_images([capture.sequence], failing, baseline_frames=20)
    assert excinfo.value is raised
    assert len(calls) == 1


def test_a_signal_between_pairs_stops_the_walk(tmp_path: Path):
    """Ctrl+C while the walk matches, with the GIL released, raises
    ``KeyboardInterrupt`` from the walk's check between pairs, not from the
    callable, and the callable is not called again."""
    capture = _Capture(tmp_path / "frames", frames=300)
    returned = threading.Event()

    def ensure(frames):
        capture(frames)
        returned.set()

    def interrupt():
        returned.wait()
        # The first batch's pairs take far longer than this to match.
        time.sleep(0.02)
        _thread.interrupt_main()

    thread = threading.Thread(target=interrupt)
    thread.start()
    with pytest.raises(KeyboardInterrupt) as excinfo:
        pick_images([capture.sequence], ensure, baseline_frames=100)
    thread.join()
    assert len(capture.calls) == 1
    assert all(entry.name != "ensure" for entry in excinfo.traceback)


def test_errors_map_to_python_exceptions(tmp_path: Path):
    capture = _Capture(tmp_path / "frames", frames=40)
    with pytest.raises(TypeError, match="callable"):
        pick_images([capture.sequence], 3)
    with pytest.raises(ValueError, match="no sequences"):
        pick_images([], capture)
    with pytest.raises(ValueError, match="increasing"):
        frames = capture.sequence.frames
        pick_images(
            [CaptureSequence(["a_%d.jpg"], [frames[1], frames[0]])],
            capture,
        )
    # A source that writes nothing: the .sift read raises the OSError of
    # the format errors, with the path.
    with pytest.raises(FileNotFoundError, match="frame_0001.jpg.sift"):
        pick_images([capture.sequence], lambda frames: None)
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
        pick_images([blocked], capture)
    # Version 1 files in a folder the capture's files go in.
    v1_folder = tmp_path / "v1"
    v1_folder.mkdir()
    v1 = _Capture(v1_folder, frames=40)
    v1.write(0, v1_folder / "frame_0001.jpg.sift")
    _rewrite_as_version_1(v1_folder / "frame_0001.jpg.sift")
    with pytest.raises(ValueError, match="version 1 .sift files"):
        pick_images([v1.sequence], v1)
    assert v1.calls == []


def _rewrite_as_version_1(path: Path) -> None:
    from sfmtool.fileio import read_sift

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
    assert FrameId(0, 3) == FrameId(0, 3) != FrameId(1, 3)
    assert len({FrameId(0, 3), FrameId(0, 3)}) == 1
    assert (FrameId(2, 5).sequence, FrameId(2, 5).frame) == (2, 5)
    assert repr(FrameId(2, 5)) == "FrameId(sequence=2, frame=5)"
    with pytest.raises(TypeError):
        CaptureFrame(1, ["not an image"])


def test_photographs_extracted_by_the_workspace_are_not_a_video(
    isolated_seoul_bull_17_images: list[Path], tmp_path: Path
):
    """Real images through the workspace's own extraction: the 17
    photographs of a walk around a sculpture are judged not a video, and
    every .sift file written is one the walk read."""
    from sfmtool.sift.extract import image_files_to_sift_files

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
    asked = []

    def ensure(frames):
        asked.extend(frames)
        image_files_to_sift_files(
            [sequence.frames[f.frame].images[0].image_path for f in frames],
            feature_tool=config["feature_tool"],
            feature_options=config["feature_options"],
            feature_prefix_dir=config["feature_prefix_dir"],
        )

    report = pick_images([sequence], ensure)
    assert report["verdict"] == "not_a_video"
    assert report["sequences"][0]["video"]["is_video"] is False
    assert report["sequences"][0]["walk"] is None
    assert sorted(p.name for p in features.glob("*.sift")) == sorted(
        f"{images[f.frame].name}.sift" for f in asked
    )
