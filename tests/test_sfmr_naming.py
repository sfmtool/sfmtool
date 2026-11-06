# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generated `.sfmr` file names."""

from pathlib import Path

from sfmtool._sfmr_naming import get_next_sfm_filename


def test_a_long_irregular_frame_list_gives_a_short_name(tmp_path: Path):
    """An irregular list of frames is named by its first and last frame and count.

    96 frames spread over 1..1177, as `sfm pick-images` writes, once gave a
    413-character name that Windows refused to create.
    """
    frames = [1 + (i * 1176) // 95 + (i % 3) for i in range(95)] + [1177]
    frames = sorted(set(frames))
    assert len(frames) == 96
    images = [tmp_path / f"DinoLedge_{n:04d}.jpg" for n in frames]

    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith("-00-solve-DinoLedge_1-1177-total-96-images.sfmr")

    out.touch()
    second = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert second.name.endswith("-01-solve-DinoLedge_1-1177-total-96-images.sfmr")


def test_a_short_frame_list_keeps_its_range(tmp_path: Path):
    """A list whose range fits keeps the range expression in the name."""
    images = [tmp_path / f"DinoLedge_{n:04d}.jpg" for n in (1, 43, 57, 68)]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith("-00-solve-DinoLedge_1,43,57,68.sfmr")


def test_a_long_stem_is_cut_to_fit(tmp_path: Path):
    """A first stem too long for the limit is cut, never a character in half."""
    stem = "é" * 200
    images = [tmp_path / f"{stem}.jpg", tmp_path / "other.png"]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    descriptor = out.stem.split("-solve-", 1)[1]
    assert descriptor == "é" * 42 + "-total-2-images"
