# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the frame pair matching bindings
(``sfmtool.analysis.FrameFeatures`` / ``match_frame_pair``; see
``specs/core/analysis/frame-pair-matching.md``)."""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.analysis import FrameFeatures, match_frame_pair
from sfmtool.fileio import SiftFile
from sfmtool.geometry import CameraIntrinsics

WIDTH, HEIGHT = 960, 540


def _translated_pair(n=200, shift=(48.0, -24.0), seed=0):
    """Two frames of the same random descriptors, the second's positions
    moved by ``shift``."""
    rng = np.random.default_rng(seed)
    descriptors = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
    a = rng.uniform([60, 60], [WIDTH - 60, HEIGHT - 60], size=(n, 2)).astype(np.float32)
    b = a + np.asarray(shift, dtype=np.float32)
    return (
        FrameFeatures(a, descriptors, WIDTH, HEIGHT),
        FrameFeatures(b, descriptors, WIDTH, HEIGHT),
    )


def test_a_known_translation_gives_its_step():
    a, b = _translated_pair()
    stats = match_frame_pair(a, b)
    assert set(stats) == {"matches", "step_percent", "essential_split"}
    assert stats["essential_split"] is None
    assert stats["matches"] == 200
    assert stats["step_percent"] == pytest.approx(
        100 * np.hypot(48, 24) / 960, abs=1e-4
    )


def _camera(width=WIDTH, height=HEIGHT):
    return CameraIntrinsics(
        "SIMPLE_PINHOLE",
        width,
        height,
        {
            "focal_length": 700.0,
            "principal_point_x": width / 2,
            "principal_point_y": height / 2,
        },
    )


def _turned_pair(camera, degrees, n=300, seed=3):
    """Two frames of the same random descriptors, the second the first seen
    by ``camera`` turned ``degrees`` about its vertical axis."""
    rng = np.random.default_rng(seed)
    angle = np.radians(degrees)
    turn = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    a, b = [], []
    while len(a) < n:
        u, v = rng.uniform([0, 0], [WIDTH, HEIGHT])
        pixel = camera.ray_to_pixel(turn @ np.asarray(camera.pixel_to_ray(u, v)))
        if pixel is not None and 0 < pixel[0] < WIDTH and 0 < pixel[1] < HEIGHT:
            a.append((u, v))
            b.append(pixel)
    descriptors = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
    return (
        FrameFeatures(np.asarray(a), descriptors, WIDTH, HEIGHT),
        FrameFeatures(np.asarray(b), descriptors, WIDTH, HEIGHT),
    )


def test_a_camera_measures_the_essential_split():
    camera = _camera()
    a, b = _turned_pair(camera, 5.0)
    stats = match_frame_pair(a, b, camera=camera, seed=7)
    split = stats["essential_split"]
    assert set(split) == {
        "status",
        "matches",
        "inliers",
        "rotation_deg",
        "parallax_p50_deg",
        "parallax_p90_deg",
    }
    assert split["status"] == "measured"
    assert split["matches"] == stats["matches"] == 300
    assert split["inliers"] >= 290
    assert split["rotation_deg"] == pytest.approx(5.0, abs=0.01)
    assert split["parallax_p90_deg"] < 0.01
    # The matches and the step are those of the pair without a camera, and
    # the split repeats for its seed.
    plain = match_frame_pair(a, b)
    assert (stats["matches"], stats["step_percent"]) == (
        plain["matches"],
        plain["step_percent"],
    )
    assert match_frame_pair(a, b, camera=camera, seed=7) == stats


def test_an_essential_split_that_cannot_be_measured_says_why():
    camera = _camera()
    a, b = _turned_pair(camera, 5.0, n=11)
    split = match_frame_pair(a, b, camera=camera)["essential_split"]
    assert split == {
        "status": "not_measured",
        "matches": 11,
        "reason": "too_few_matches",
    }
    # With a lower floor the same eleven matches are measured.
    split = match_frame_pair(a, b, camera=camera, min_inliers=8)["essential_split"]
    assert split["status"] == "measured"


def test_essential_split_options_are_checked():
    camera = _camera()
    a, b = _turned_pair(camera, 5.0, n=20)
    with pytest.raises(ValueError, match="needs a camera"):
        match_frame_pair(a, b, seed=1)
    with pytest.raises(ValueError, match="tolerance_px"):
        match_frame_pair(a, b, camera=camera, tolerance_px=0.0)
    with pytest.raises(ValueError, match="min_inliers"):
        match_frame_pair(a, b, camera=camera, min_inliers=7)
    with pytest.raises(ValueError, match="480 x 270 but the frames are 960 x 540"):
        match_frame_pair(a, b, camera=_camera(WIDTH // 2, HEIGHT // 2))


def test_frame_features_reports_its_size():
    a, _ = _translated_pair(n=50)
    assert (len(a), a.width, a.height) == (50, WIDTH, HEIGHT)
    assert repr(a) == "FrameFeatures(50 features, 960 x 540)"


def test_float64_positions_are_accepted():
    rng = np.random.default_rng(1)
    descriptors = rng.integers(0, 256, size=(40, 128), dtype=np.uint8)
    positions = rng.uniform(0, 500, size=(40, 2))
    a = FrameFeatures(positions, descriptors, WIDTH, HEIGHT)
    b = FrameFeatures(positions.astype(np.float32), descriptors, WIDTH, HEIGHT)
    assert match_frame_pair(a, b)["step_percent"] == 0.0


def test_options_reach_the_kernel():
    a, b = _translated_pair()
    # A ratio this strict rejects nothing here, since each descriptor's twin
    # is at distance 0, but one past 1 is refused.
    assert match_frame_pair(a, b, ratio=0.1)["matches"] == 200
    with pytest.raises(ValueError, match="ratio"):
        match_frame_pair(a, b, ratio=1.5)


def test_frames_of_different_sizes_are_refused():
    rng = np.random.default_rng(2)
    descriptors = rng.integers(0, 256, size=(10, 128), dtype=np.uint8)
    positions = rng.uniform(0, 500, size=(10, 2)).astype(np.float32)
    a = FrameFeatures(positions, descriptors, WIDTH, HEIGHT)
    b = FrameFeatures(positions, descriptors, HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="differ in size"):
        match_frame_pair(a, b)


def test_bad_arrays_are_refused():
    descriptors = np.zeros((3, 128), dtype=np.uint8)
    positions = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="descriptors must be a uint8 array"):
        FrameFeatures(positions, descriptors.astype(np.int32), WIDTH, HEIGHT)
    with pytest.raises(TypeError, match="positions_xy"):
        FrameFeatures(positions.astype(np.int32), descriptors, WIDTH, HEIGHT)
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        FrameFeatures(np.zeros((3, 3), dtype=np.float32), descriptors, WIDTH, HEIGHT)
    with pytest.raises(ValueError, match=r"shape \(N, 128\)"):
        FrameFeatures(positions, np.zeros((3, 64), dtype=np.uint8), WIDTH, HEIGHT)
    with pytest.raises(ValueError, match="descriptors has 256 bytes"):
        FrameFeatures(positions, descriptors[:2], WIDTH, HEIGHT)
    with pytest.raises(ValueError, match="not finite"):
        FrameFeatures(
            np.full((3, 2), np.nan, dtype=np.float32), descriptors, WIDTH, HEIGHT
        )
    with pytest.raises(ValueError, match="side of 0"):
        FrameFeatures(positions, descriptors, 0, HEIGHT)
    with pytest.raises(ValueError, match="num_trees"):
        FrameFeatures(positions, descriptors, WIDTH, HEIGHT, num_trees=0)
    with pytest.raises(ValueError, match="unknown preset"):
        FrameFeatures(positions, descriptors, WIDTH, HEIGHT, preset="slow")


def _seoul_bull_sift_files(workspace: Path) -> list[Path]:
    files = sorted(workspace.parent.glob("test_17_image/features/*/*.sift"))
    assert len(files) == 17
    return files


def test_neighbouring_seoul_bull_images_match(seoul_bull_workspace_once: Path):
    """Real `.sift` files, extracted by the workspace's tool into the
    fixture's own copy of the images: neighbouring images of the walk around
    the sculpture match."""
    files = _seoul_bull_sift_files(seoul_bull_workspace_once)
    frames = [FrameFeatures.from_sift(path) for path in files[:3]]
    with SiftFile(files[0]) as sift:
        width, height = sift.metadata["image_width"], sift.metadata["image_height"]
        assert len(frames[0]) == sift.feature_count
    assert (frames[0].width, frames[0].height) == (width, height)

    for a, b in [(frames[0], frames[1]), (frames[1], frames[2])]:
        stats = match_frame_pair(a, b)
        assert stats["matches"] >= 100, stats
        assert 0.5 < stats["step_percent"] < 50, stats
        # The same frames give the same statistics.
        assert match_frame_pair(a, b) == stats

    # A frame matched with itself moves by nothing.
    assert match_frame_pair(frames[0], frames[0])["step_percent"] == 0.0


def test_from_sift_takes_a_path_or_a_sift_file(seoul_bull_workspace_once: Path):
    files = _seoul_bull_sift_files(seoul_bull_workspace_once)
    by_path = [FrameFeatures.from_sift(str(path)) for path in files[:2]]
    with SiftFile(files[0]) as first, SiftFile(files[1]) as second:
        by_file = [FrameFeatures.from_sift(first), FrameFeatures.from_sift(second)]
    assert [len(f) for f in by_file] == [len(f) for f in by_path]
    assert match_frame_pair(*by_file) == match_frame_pair(*by_path)


def test_from_sift_refuses_what_it_cannot_read(tmp_path: Path):
    with pytest.raises(TypeError, match="SiftFile"):
        FrameFeatures.from_sift(3)
    with pytest.raises(OSError):
        FrameFeatures.from_sift(tmp_path / "missing.sift")
