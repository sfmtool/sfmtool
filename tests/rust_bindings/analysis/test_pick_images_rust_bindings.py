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
    assert set(stats) == {
        "matches",
        "step_percent",
        "f_inlier_share",
        "h_over_f",
        "rotation_error",
    }
    assert stats["matches"] == 200
    assert stats["step_percent"] == pytest.approx(
        100 * np.hypot(48, 24) / 960, abs=1e-4
    )
    # Every match fits one homography, a translation.
    assert stats["f_inlier_share"] == pytest.approx(1.0)
    assert stats["h_over_f"] == pytest.approx(1.0)


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


def test_fewer_than_min_matches_give_no_shares():
    a, b = _translated_pair(n=20)
    stats = match_frame_pair(a, b)
    assert stats["matches"] == 20
    assert stats["f_inlier_share"] is None
    assert stats["h_over_f"] is None
    assert stats["rotation_error"] is None
    stats = match_frame_pair(a, b, min_matches=20)
    assert stats["f_inlier_share"] == pytest.approx(1.0)


def test_options_reach_the_kernel():
    a, b = _translated_pair()
    # A ratio this strict rejects nothing here, since each descriptor's twin
    # is at distance 0, but one past 1 is refused.
    assert match_frame_pair(a, b, ratio=0.1)["matches"] == 200
    with pytest.raises(ValueError, match="ratio"):
        match_frame_pair(a, b, ratio=1.5)
    with pytest.raises(ValueError, match="min_matches"):
        match_frame_pair(a, b, min_matches=6)
    with pytest.raises(ValueError, match="fundamental_error_per_960px"):
        match_frame_pair(a, b, fundamental_error_per_960px=0.0)
    with pytest.raises(ValueError, match="homography_error_per_960px"):
        match_frame_pair(a, b, homography_error_per_960px=float("nan"))


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
    the sculpture match, and their matches fit one camera motion."""
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
        assert stats["f_inlier_share"] > 0.5, stats
        assert 0.0 < stats["h_over_f"] <= 1.0, stats
        assert stats["rotation_error"] > 0.0, stats
        # The same seed gives the same statistics.
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
