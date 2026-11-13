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


def _shop_frames(root: Path) -> list[Path]:
    """Frames of two videos, each in its own directory with the same file names.

    45 frames of shopA (1, 5, …, 177) and 32 of shopB (50..81): 77 images,
    and the numbers 53, 57, … 81 appear in both directories.
    """
    shop_a = [root / "shopA" / f"frame_{n:06d}.jpg" for n in range(1, 181, 4)]
    shop_b = [root / "shopB" / f"frame_{n:06d}.jpg" for n in range(50, 82)]
    return shop_a + shop_b


def test_frames_of_two_videos_name_both_and_count_every_image(tmp_path: Path):
    """Two directories holding the same frame names are two sequences.

    Described by file names alone, `shopA/frame_000053.jpg` and
    `shopB/frame_000053.jpg` were one image, so the name merged both ranges and
    counted distinct frame numbers instead of images.
    """
    images = _shop_frames(tmp_path / "frames")
    assert len(images) == 77
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith(
        "-00-solve-shopA-frame_1-177x4+shopB-frame_50-81-total-77-images.sfmr"
    )


def test_a_single_video_in_its_own_directory_is_not_named_by_it(tmp_path: Path):
    """One directory is one sequence, named by its file names as before."""
    images = _shop_frames(tmp_path / "frames")[:45]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith("-00-solve-frame_1-177x4.sfmr")


def test_long_ranges_in_several_directories_keep_first_and_last(tmp_path: Path):
    """Ranges too long for the limit give each directory's first and last frame."""
    images = [
        tmp_path / d / f"frame_{n:04d}.jpg"
        for d in ("shopA", "shopB")
        for n in (1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144)
    ]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith(
        "-00-solve-shopA-frame_1-144+shopB-frame_1-144-total-24-images.sfmr"
    )


def test_images_beside_a_directory_of_frames(tmp_path: Path):
    """Images in the common directory itself are listed without a directory."""
    images = [tmp_path / f"IMG_{n:04d}.jpg" for n in (3, 4, 5)] + [
        tmp_path / "shopA" / "frame_0001.jpg",
        tmp_path / "shopA" / "notes.png",
    ]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith("-00-solve-IMG_3-5+shopA-total-5-images.sfmr")


def test_nested_directories_join_their_parts(tmp_path: Path):
    """A directory below another is named by its relative path, joined by `_`."""
    images = [
        tmp_path / day / f"frame_{n:04d}.jpg"
        for day in ("day1/shopA", "day2")
        for n in (1, 2, 3)
    ]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith(
        "-00-solve-day1_shopA-frame_1-3+day2-frame_1-3-total-6-images.sfmr"
    )


def test_many_directories_name_those_that_fit_and_count_the_rest(tmp_path: Path):
    """Past the byte limit the name lists the directories that fit, then `+k-more`."""
    images = [
        tmp_path / f"video{i:02d}" / f"frame_{n:06d}.jpg"
        for i in range(30)
        for n in range(1, 4)
    ]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    descriptor = out.stem.split("-solve-", 1)[1]
    shown = "+".join(f"video{i:02d}" for i in range(9))
    assert descriptor == shown + "+21-more-total-90-images"
    assert len(descriptor.encode("utf-8")) <= 100


def test_non_ascii_directories_stay_within_the_byte_limit(tmp_path: Path):
    """The byte limit counts UTF-8 bytes, and a name is never cut mid-character."""
    images = [
        tmp_path / f"店舗{i}" / f"frame_{n:04d}.jpg" for i in range(12) for n in (1, 2)
    ]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    descriptor = out.stem.split("-solve-", 1)[1]
    # Each `店舗{i}` is 3 or 4 characters but 7 or 8 bytes: nine of them fit
    # in the 100 bytes, where counting characters would fit all twelve.
    shown = "+".join(f"店舗{i}" for i in sorted(str(i) for i in range(12))[:9])
    assert descriptor == shown + "+3-more-total-24-images"
    assert len(descriptor.encode("utf-8")) <= 100


def test_a_long_first_directory_is_cut_to_fit(tmp_path: Path):
    """A first directory too long for the limit is cut at a character boundary."""
    images = [tmp_path / ("a" + "é" * 200) / "a.jpg", tmp_path / "other" / "b.jpg"]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    descriptor = out.stem.split("-solve-", 1)[1]
    # 78 bytes are left for the directory: "a" and 38 "é" take 77, and a
    # 39th "é" would end on half a character.
    assert descriptor == "a" + "é" * 38 + "+1-more-total-2-images"


def test_a_path_through_dotdot_is_the_same_directory(tmp_path: Path):
    """`shopA/../shopB` and `shopB` are one directory, so one sequence."""
    images = [
        tmp_path / "shopA" / ".." / "shopB" / "frame_0001.jpg",
        tmp_path / "shopB" / "frame_0002.jpg",
        tmp_path / "shopB" / "frame_0003.jpg",
    ]
    out = get_next_sfm_filename(tmp_path / "sfmr", images)
    assert out.name.endswith("-00-solve-frame_1-3.sfmr")
