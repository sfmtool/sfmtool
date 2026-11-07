# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""How `sfm pick-images` groups files into sequences, turns chosen frames
into list entries and words the walk's swaps (`sfmtool._pick_images`,
`sfmtool._pick_images_report`).

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
from sfmtool._pick_images_report import slice_suggestion, walk_lines
from sfmtool.camrig.resolver import CamrigRig
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


def _camrig_collect(rels: list[str], frames: list[tuple[int, int]]) -> list:
    """`collect_sequences` of `rels` with a `.camrig` in `cap 5%/` whose
    sensor 0 is `right/` and sensor 1 `left/`, each image assigned the
    `(sensor, frame)` in `frames`."""
    folder = ROOT / "cap 5%"
    paths = [folder / rel for rel in rels]
    rig = CamrigRig(
        camrig_file=folder / "rig.camrig",
        data={"sensor_image_patterns": ["right/f_%02d.jpg", "left/f_%02d.jpg"]},
        assignments={p.resolve(): f for p, f in zip(paths, frames)},
    )
    return collect_sequences(paths, ROOT, None, rig)


def test_camrig_rig_frames_group_by_frame_index_with_sensor_0_first():
    (sequence,) = _camrig_collect(
        ["left/f_02.jpg", "right/f_02.jpg", "left/f_01.jpg", "right/f_01.jpg"],
        [(1, 2), (0, 2), (1, 1), (0, 1)],
    )
    # The patterns are the .camrig's behind its folder, with its `%` escaped.
    assert sequence.patterns == ["cap 5%%/right/f_%02d.jpg", "cap 5%%/left/f_%02d.jpg"]
    assert sequence.sensors == ["right/f_%02d.jpg", "left/f_%02d.jpg"]
    assert sequence.numbers == [1, 2]
    assert sequence.images[1] == [
        Path("cap 5%/right/f_02.jpg"),
        Path("cap 5%/left/f_02.jpg"),
    ]


def test_camrig_rig_errors_name_the_files():
    with pytest.raises(PickImagesInputError) as e:
        _camrig_collect(
            ["right/f_01.jpg", "left/f_01.jpg", "right/f_02.jpg", "Left/f_03.jpg"],
            [(0, 1), (1, 1), (0, 2), (1, 3)],
        )
    message = str(e.value)
    # A file the .camrig matches only ignoring case, which a list cannot hold.
    assert (
        "1 image file(s) match the patterns of" in message
        and "cap 5%/Left/f_03.jpg" in message
    )
    assert (
        "2 rig frame(s) of cap 5%%/right/f_%02d.jpg lack the image of a sensor"
        in message
    )
    assert "cap 5%/right/f_02.jpg (no image of cap 5%%/left/f_%02d.jpg)" in message
    assert "cap 5%/Left/f_03.jpg (no image of cap 5%%/right/f_%02d.jpg)" in message


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
    # A lone chosen frame is one entry per sensor.
    assert [(p, str(r)) for p, r in list_entries(sequence, [2])] == [
        ("rig/left/f_%02d.jpg", "2"),
        ("rig/right/f_%02d.jpg", "2"),
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


def test_swaps_and_steps_at_the_floor_are_reported():
    # A walk of frames numbered 10 apart with a swap, a step stopped at the
    # match floor beyond the next frame and one to a next frame that matches
    # nothing; the report names the swap by file number and counts the steps
    # at the floor, and those at the next frame among them.
    sequence = PickSequence(
        patterns=["f_%d.jpg"],
        numbers=[10 * t for t in range(1, 31)],
        images=[[ROOT / f"f_{10 * t}.jpg"] for t in range(1, 31)],
    )
    stats = {"step_percent": 10.0, "matches": 700}
    nothing = {"step_percent": None, "matches": 0}
    walk = {
        "chosen": [0, 12, 21, 22],
        "steps": [
            {"from": 0, "to": 12, "stats": stats, "rate": 0.8, "kind": "in_band"},
            {"from": 12, "to": 21, "stats": stats, "rate": 1.1, "kind": "at_floor"},
            {"from": 21, "to": 22, "stats": nothing, "rate": None, "kind": "at_floor"},
        ],
        "swaps": [{"candidate": 22, "chosen": 21}],
        "frames_read": 9,
    }
    report = {"sequences": [walk], "frames": []}
    lines = walk_lines(report, 0, sequence, {}, "", 10.0, 0.3, 600)
    assert "Walk: target step 10% (band 7-13%), match floor 600" in lines
    assert "  Chosen: 4 of 30 frames; read: 9 frames (0 extracted, 9 reused)" in lines
    assert "  Frames blurred against their neighbours swapped (1): 230->220" in lines
    assert (
        "  Steps stopped at the match floor: 2 of 3, 1 of them at the next frame"
        in (lines)
    )
    # The run of positions whose only step matches nothing has no step or
    # rate to report.
    assert any(line.endswith("unmeasured      1") for line in lines), lines

    walk["swaps"] = []
    for step in walk["steps"][1:]:
        step["kind"] = "bracketed"
    lines = walk_lines(report, 0, sequence, {}, "", 10.0, 0.3, 600)
    assert "  Frames blurred against their neighbours swapped: none" in lines
    assert "  Steps stopped at the match floor: none" in lines
