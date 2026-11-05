# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""How `sfm pick-images` groups files into sequences and turns chosen frames
into list entries (`sfmtool._pick_images`, `sfmtool._pick_images_report`).

The paths here need not exist: grouping reads names only.
"""

from pathlib import Path

import pytest

from sfmtool._pick_images import (
    PickImagesInputError,
    PickSequence,
    collect_sequences,
    frame_field,
    list_entries,
)
from sfmtool._pick_images_report import slice_suggestion
from sfmtool.reconstruction import RangeExpr

ROOT = Path("/ws").absolute()


def _collect(names: list[str], rig_configs=None) -> list[PickSequence]:
    return collect_sequences([ROOT / n for n in names], ROOT, rig_configs)


def test_frame_field_pads_only_a_fixed_width():
    assert frame_field({4}) == "%04d"
    assert frame_field({1}) == "%d"
    assert frame_field({1, 2}) == "%d"


def test_sequences_are_split_by_folder_and_name_and_ordered_by_number():
    sequences = _collect(
        [
            "b/walk_0010.jpg",
            "a/x_0002.jpg",
            "a/x_0001.jpg",
            "a/y_0001.jpg",
            "a/x_0003.JPG",
            "c/p 5%_9.png",
            "c/p 5%_10.png",
            "d/shot (3).jpg",
            "d/shot (12).jpg",
        ]
    )
    assert [s.patterns for s in sequences] == [
        ["a/x_%04d.JPG"],
        ["a/x_%04d.jpg"],
        ["a/y_%04d.jpg"],
        ["b/walk_%04d.jpg"],
        ["c/p 5%%_%d.png"],
        ["d/shot (%d).jpg"],
    ]
    x = sequences[1]
    assert x.numbers == [1, 2]
    assert x.images == [[Path("a/x_0001.jpg")], [Path("a/x_0002.jpg")]]
    assert x.sensors is None
    assert sequences[4].numbers == [9, 10]
    assert sequences[5].numbers == [3, 12]


def test_every_file_at_fault_is_named():
    with pytest.raises(PickImagesInputError) as e:
        _collect(["a/cover.jpg", "a/x_1.jpg", "a/x_01.jpg", "a/x_2.jpg", "b/s*_1.jpg"])
    message = str(e.value)
    assert "1 image file(s) have no file number" in message and "a/cover.jpg" in message
    assert "a/x_1.jpg and a/x_01.jpg" in message
    assert "'*' in their path" in message and "b/s*_1.jpg" in message


RIG = [
    {
        "cameras": [
            {"image_prefix": "rig/right/"},
            {"image_prefix": "rig/left/", "ref_sensor": True},
        ]
    }
]


def test_rig_frames_group_by_frame_key_with_the_reference_sensor_first():
    (sequence,) = _collect(
        [
            "rig/right/f_02.jpg",
            "rig/left/f_02.jpg",
            "rig/left/f_01.jpg",
            "rig/right/f_01.jpg",
        ],
        RIG,
    )
    assert sequence.patterns == ["rig/left/f_%02d.jpg", "rig/right/f_%02d.jpg"]
    assert sequence.sensors == ["rig/left/", "rig/right/"]
    assert sequence.numbers == [1, 2]
    assert sequence.images[1] == [Path("rig/left/f_02.jpg"), Path("rig/right/f_02.jpg")]


def test_list_entries_are_one_per_sensor_with_one_range():
    (sequence,) = _collect(
        [
            "rig/left/f_01.jpg",
            "rig/right/f_01.jpg",
            "rig/left/f_02.jpg",
            "rig/right/f_02.jpg",
        ],
        RIG,
    )
    entries = list_entries(sequence, [1, 5, 9, 10, 11])
    assert [(p, str(r)) for p, r in entries] == [
        ("rig/left/f_%02d.jpg", "1-9:4,10,11"),
        ("rig/right/f_%02d.jpg", "1-9:4,10,11"),
    ]


def test_a_range_too_long_for_one_expression_is_split_in_order():
    sequence = PickSequence(patterns=["v/f_%d.jpg"], numbers=[], images=[])
    # Gaps that alternate between 1 and more than 1 give one sub-range per
    # two numbers: more than a range expression may hold.
    numbers = [n for i in range(15_000) for n in (3 * i, 3 * i + 1)]
    entries = list_entries(sequence, numbers)
    assert len(entries) > 1
    assert all(p == "v/f_%d.jpg" for p, _ in entries)
    values = [v for _, r in entries for v in RangeExpr(str(r))]
    assert values == numbers


def test_slice_suggestion():
    assert slice_suggestion("walk1", 60) is None
    assert slice_suggestion("walk1", 512) == (
        "  For a quick test solve, try -r 'walk1[:30]', -r 'walk1[241:271]' or "
        "-r 'walk1[-30:]'"
    )
