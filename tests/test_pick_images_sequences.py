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
from sfmtool._pick_images_report import (
    pilot_source,
    slice_suggestion,
    time_lines,
    walk_lines,
)
from sfmtool.camrig.resolver import CamrigCamera, CamrigRig
from sfmtool.reconstruction import RangeExpr

ROOT = Path("/ws").absolute()


def _collect(names: list[str]) -> list[PickSequence]:
    return collect_sequences([ROOT / n for n in names], ROOT)


def test_frame_field_writes_every_number_as_the_names_write_it():
    # One width, padded or not.
    assert frame_field({"0001", "0002"}) == ("%04d", [])
    assert frame_field({"1000", "1001"}) == ("%04d", [])
    assert frame_field({"1", "2"}) == ("%d", [])
    # Unpadded numbers of several widths.
    assert frame_field({"8", "9", "10", "100"}) == ("%d", [])
    # A zero-padded sequence that runs past its width, as ffmpeg writes
    # frame 10000 under %04d.
    assert frame_field({"0001", "0999", "9999", "10000", "123456"}) == ("%04d", [])
    assert frame_field({"0", "1"}) == ("%d", [])


def test_frame_field_names_the_numbers_no_one_field_writes():
    # Zero-padded to two widths: the field that writes the most of them.
    assert frame_field({"01", "02", "003"}) == ("%02d", ["003"])
    assert frame_field({"01", "003", "004"}) == ("%03d", ["01"])
    # A tie goes to the wider field.
    assert frame_field({"01", "003"}) == ("%03d", ["01"])
    # A number shorter than the padded width.
    assert frame_field({"5", "0010", "0011"}) == ("%04d", ["5"])
    # A number wider than the padded width that has a leading zero.
    assert frame_field({"0010", "00011"}) == ("%05d", ["0010"])


def test_a_sequence_crossing_its_padded_width_is_one_sequence():
    (sequence,) = _collect(["v/f_0001.jpg", "v/f_9999.jpg", "v/f_10000.jpg"])
    assert sequence.patterns == ["v/f_%04d.jpg"]
    assert sequence.numbers == [1, 9999, 10000]


def test_a_sequence_padded_to_different_widths_is_an_error_naming_the_files():
    with pytest.raises(PickImagesInputError) as e:
        _collect(["v/f_01.jpg", "v/f_02.jpg", "v/f_003.jpg", "v/f_4.jpg"])
    message = str(e.value)
    assert "v/f_%02d.jpg are zero-padded in ways no one frame field writes" in message
    assert "2 file(s) do not match v/f_%02d.jpg: v/f_003.jpg, v/f_4.jpg" in message


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


def _camrig_collect(rels: list[str], frames: list[tuple[int, int]]) -> list:
    """`collect_sequences` of `rels` with a `.camrig` in `cap 5%/` whose
    sensor 0 is `right/` and sensor 1 `left/`, each image assigned the
    `(sensor, frame)` in `frames`."""
    folder = ROOT / "cap 5%"
    paths = [folder / rel for rel in rels]
    rig = CamrigRig(
        camrig_file=folder / "rig.camrig",
        data={
            "sensor_image_patterns": ["right/f_%02d.jpg", "left/f_%02d.jpg"],
            "cameras": RIG_CAMERAS,
            "camera_indexes": [1, 0],
        },
        assignments={p.resolve(): f for p, f in zip(paths, frames)},
    )
    return collect_sequences(paths, ROOT, [rig])


# The two cameras of `_camrig_collect`'s rig: sensor 0 takes the second.
RIG_CAMERAS = [
    {"model": "SIMPLE_PINHOLE", "width": 640, "height": 480, "parameters": {"f": 1}},
    {"model": "SIMPLE_PINHOLE", "width": 640, "height": 480, "parameters": {"f": 2}},
]


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
    # The camera is the reference sensor's, sensor 0's.
    assert sequence.camera == RIG_CAMERAS[1]
    assert sequence.camera_source == (
        f"the camera of {ROOT / 'cap 5%' / 'rig.camrig'}, sensor right/f_%02d.jpg"
    )


def test_a_sequence_takes_the_camera_of_a_camrig_covering_all_its_frames():
    rels = ["a/x_1.jpg", "a/x_2.jpg", "b/y_1.jpg", "b/y_2.jpg", "c/z_1.jpg"]
    paths = [ROOT / r for r in rels]
    cameras = [
        CamrigCamera(
            camrig_file=ROOT / "a.camrig",
            camera=RIG_CAMERAS[0],
            images={(ROOT / r).resolve() for r in rels[:2]},
        ),
        CamrigCamera(
            camrig_file=ROOT / "b.camrig",
            camera=RIG_CAMERAS[1],
            images={(ROOT / "b/y_1.jpg").resolve()},
        ),
    ]
    a, b, c = collect_sequences(paths, ROOT, camrig_cameras=cameras)
    assert a.camera == RIG_CAMERAS[0]
    assert a.camera_source == f"the camera of {ROOT / 'a.camrig'}"
    assert b.camera is None
    assert b.camera_source == f"{ROOT / 'b.camrig'} covers only 1 of its 2 frames"
    assert c.camera is None and c.camera_source == "no .camrig covers its frames"


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


def test_a_camrig_in_a_folder_with_a_star_is_an_error():
    folder = ROOT / "cap*"
    paths = [folder / "right/f_01.jpg", folder / "left/f_01.jpg"]
    rig = CamrigRig(
        camrig_file=folder / "rig.camrig",
        data={"sensor_image_patterns": ["right/f_%02d.jpg", "left/f_%02d.jpg"]},
        assignments={p.resolve(): f for p, f in zip(paths, [(0, 1), (1, 1)])},
    )
    with pytest.raises(PickImagesInputError, match=r"has a '\*' in its path"):
        collect_sequences(paths, ROOT, [rig])


def test_list_entries_are_one_per_sensor_with_one_range():
    (sequence,) = _camrig_collect(
        ["right/f_01.jpg", "left/f_01.jpg", "right/f_02.jpg", "left/f_02.jpg"],
        [(0, 1), (1, 1), (0, 2), (1, 2)],
    )
    entries = list_entries(sequence, [1, 5, 9, 10, 11])
    assert [(p, str(r)) for p, r in entries] == [
        ("cap 5%%/right/f_%02d.jpg", "1-9:4,10,11"),
        ("cap 5%%/left/f_%02d.jpg", "1-9:4,10,11"),
    ]
    # A lone chosen frame is one entry per sensor.
    assert [(p, str(r)) for p, r in list_entries(sequence, [2])] == [
        ("cap 5%%/right/f_%02d.jpg", "2"),
        ("cap 5%%/left/f_%02d.jpg", "2"),
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


def _density(target, gain, median, split, measure, pair, focal=1.0):
    return {
        "target_percent": target,
        "gain": gain,
        "median_measure": median,
        "essential_split": split,
        "focal_factor": focal,
        "measured_pair": pair,
        "measure": measure,
    }


def test_the_local_density_term_is_reported_by_step():
    sequence = PickSequence(
        patterns=["f_%d.jpg"],
        numbers=[10 * t for t in range(1, 31)],
        images=[[ROOT / f"f_{10 * t}.jpg"] for t in range(1, 31)],
        camera_source="the camera of k8.camrig",
    )
    stats = {"step_percent": 10.0, "matches": 700}
    near = {
        "status": "measured",
        "matches": 700,
        "inliers": 690,
        "rotation_deg": 8.0,
        "parallax_p50_deg": 0.125,
        "parallax_p90_deg": 1.5,
    }
    lost = {"status": "not_measured", "matches": 9, "reason": "too_few_matches"}
    wide = {**near, "parallax_p50_deg": 0.9, "parallax_p90_deg": 2.7}
    step = {"pair": "step", "p50_gate_deg": 0.5}
    reference = {
        "pair": "reference",
        "to": 4,
        "step_percent": 3.4,
        "matched_for_measure": True,
        "essential_split": near,
        "focal_factor": 1.03,
        "p50_gate_deg": 0.2,
    }
    walk = {
        "chosen": [0, 12, 16, 19, 29],
        "steps": [
            {
                "from": 0,
                "to": 12,
                "stats": stats,
                "rate": 0.8,
                "kind": "in_band",
                "density": _density(10.0, 1.0, 1.0, wide, 12.0, reference, 0.97),
            },
            {
                "from": 12,
                "to": 16,
                "stats": stats,
                "rate": 0.8,
                "kind": "in_band",
                "density": _density(10 / 3, 3.0, 12.0, near, 12.0, step),
            },
            {
                "from": 16,
                "to": 19,
                "stats": stats,
                "rate": 0.8,
                "kind": "in_band",
                "density": _density(10 / 3, 3.0, 12.0, lost, 1.0, step),
            },
            {
                "from": 19,
                "to": 29,
                "stats": stats,
                "rate": 1.0,
                "kind": "in_band",
                "density": _density(
                    10 / 3,
                    3.0,
                    12.0,
                    wide,
                    1.0,
                    {"pair": "wide_parallax", "to": 22, "p50_gate_deg": 0.2},
                ),
            },
        ],
        "swaps": [],
        "frames_read": 9,
        "density_term": {"status": "on", "focal_px": 945.75},
        "pilot_vote": None,
    }
    report = {"sequences": [walk], "frames": []}
    lines = walk_lines(report, 0, sequence, {}, "", 10.0, 0.3, 600)
    assert (
        "  Local density term: on, with the camera of k8.camrig (focal 945.8 px at "
        "the frames' size); 3 of 4 steps walked with a gain above 1" in lines
    )
    rows = lines[lines.index(next(x for x in lines if "Local density by step" in x)) :]
    # A reference pair: its frames, focal factor, split and gate.
    assert rows[2].split() == [
        "10-130",
        "10.00%",
        "1.00",
        "1.00",
        "10-50",
        "1.03",
        "0.125",
        "12.00",
        "0.20",
        "12.00",
    ]
    # The step itself, at the step's focal factor and gate.
    assert rows[3].split() == [
        "130-170",
        "3.33%",
        "3.00",
        "12.00",
        "step",
        "1.00",
        "0.125",
        "12.00",
        "0.50",
        "12.00",
    ]
    assert rows[4].split() == [
        "170-200",
        "3.33%",
        "3.00",
        "12.00",
        "step",
        "1.00",
        "not",
        "measured",
        "(too_few_matches)",
    ]
    # No pair: the step's own split at the reference gate, and a measure of 1.
    assert rows[5].split() == [
        "200-300",
        "3.33%",
        "3.00",
        "12.00",
        "none",
        "1.00",
        "0.900",
        "3.00",
        "0.20",
        "1.00",
    ]
    # Each reason the term is off.
    for term, why in [
        ({"reason": "max_gain_one"}, "--max-density-gain is 1"),
        (
            {"reason": "camera_shape", "camera": [1080, 1920], "frames": [1920, 1080]},
            "the camera of k8.camrig is 1080 x 1920, of another aspect than the "
            "frames' 1920 x 1080",
        ),
        (
            {"reason": "camera_refused", "problem": "a focal of 0"},
            "the camera of k8.camrig cannot be measured with: a focal of 0",
        ),
    ]:
        walk["density_term"] = {"status": "off", **term}
        lines = walk_lines(report, 0, sequence, {}, "", 10.0, 0.3, 600)
        assert f"  Local density term: off, {why}" in lines, lines
        assert not any("Local density by step" in line for line in lines)


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
        "density_term": {"status": "off", "reason": "no_camera"},
        "pilot_vote": None,
    }
    report = {"sequences": [walk], "frames": []}
    lines = walk_lines(report, 0, sequence, {}, "", 10.0, 0.3, 600)
    assert "Walk: target step 10% (band 7-13%), match floor 600" in lines
    assert "  Local density term: off, no .camrig covers its frames" in lines
    assert not any("Local density by step" in line for line in lines)
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


def test_a_fisheye_pilot_vote_names_its_model():
    vote = {
        "frames": list(range(30)),
        "camera_model": "EquidistantFisheye",
        "confirmed": True,
        "focal_px": 1034.84,
        "pool_spread": 0.0912,
        "n_pool": 11,
        "outcome": {"status": "used"},
    }
    rels = ["a/x_1.jpg", "a/x_2.jpg"]
    # A .camrig covering one of the two frames gives the sequence no camera.
    partial = CamrigCamera(
        camrig_file=ROOT / "a.camrig",
        camera=RIG_CAMERAS[0],
        images={(ROOT / rels[0]).resolve()},
    )
    (sequence,) = collect_sequences([ROOT / r for r in rels], ROOT)
    (covered,) = collect_sequences(
        [ROOT / r for r in rels], ROOT, camrig_cameras=[partial]
    )
    assert pilot_source(vote, sequence) == (
        "the pilot vote's equidistant fisheye camera, focal 1034.8 px, "
        "pool spread 0.091"
    )
    # An unconfirmed fisheye verdict is measured with as a fisheye, and says
    # it is unconfirmed.
    vote["confirmed"] = False
    assert pilot_source(vote, sequence).startswith(
        "the pilot vote's unconfirmed equidistant fisheye camera, "
    )
    vote["outcome"] = {"status": "wide_spread", "max_pool_spread": 0.35}
    vote["pool_spread"] = 0.49
    assert pilot_source(vote, sequence) == (
        "the pilot vote's pool spread 0.490 is above 0.35 (unconfirmed "
        "equidistant fisheye, focal 1034.8 px)"
    )
    # The note of a .camrig that covers only some of the frames is kept.
    assert pilot_source(vote, covered) == (
        "the pilot vote's pool spread 0.490 is above 0.35 (unconfirmed "
        f"equidistant fisheye, focal 1034.8 px); {ROOT / 'a.camrig'} covers "
        "only 1 of its 2 frames"
    )


def _walk_times(**seconds) -> dict:
    """A report's `times`: every walk figure 0.5 s unless given, the total
    their sum, and an extracting source of 4 workers."""
    names = [
        "checking_input",
        "waiting_for_frames",
        "reading_sift_metadata",
        "reading_sift_features",
        "building_forests",
        "matching",
        "essential_split",
        "pilot_vote",
        "stopping_source",
        "other",
    ]
    times = {f"{n}_seconds": seconds.get(n, 0.5) for n in names}
    times["total_seconds"] = sum(times.values())
    times["source"] = {"workers": 4, "frames": 70, "seconds": 41.25}
    return times


def test_the_time_breakdown_divides_the_command_s_thread():
    times = _walk_times(waiting_for_frames=12.0, matching=3.25)
    hdr = {
        "writers": 4,
        "waiting_for_writers_seconds": 6.5,
        "decoding_seconds": 2.0,
        "conversion_seconds": 20.0,
        "encoding_seconds": 9.75,
    }
    command_times = {
        "preparing_seconds": 0.25,
        "videos": [
            {"video": "walk.mp4", "seconds": 7.0, "hdr": None},
            {"video": "pan.MOV", "seconds": 8.5, "hdr": hdr},
        ],
        "collecting_frames_seconds": 0.125,
        "extracting_first_seconds": None,
        "picking_seconds": times["total_seconds"] + 0.75,
    }
    report = {"times": times, "sequences": [{"pilot_vote": {}}, {"pilot_vote": None}]}
    lines = time_lines(report, command_times, 0.0625)
    # Each line's label, indent and seconds, the seconds in one column.
    column = {len(line) for line in lines if line.endswith(" s")}
    assert len(column) == 1
    rows = [
        (len(line) - len(line.lstrip()), *line[:-2].strip().rsplit(maxsplit=1))
        for line in lines
        if line.endswith(" s")
    ]
    rows = [(indent, label, float(seconds)) for indent, label, seconds in rows]
    assert lines[0] == "Time on the command's thread:"
    assert rows == [
        (2, "Checking the videos, creating the workspace", 0.25),
        (2, "Extracting the videos' frames", 15.5),
        (4, "walk.mp4: ffmpeg decoding and writing JPEGs", 7.0),
        (4, "pan.MOV (HDR)", 8.5),
        (6, "ffmpeg decoding, read from its pipe", 2.0),
        (6, "Waiting for the conversion to sRGB", 6.5),
        (2, "Collecting the frames", 0.12),
        (2, "Picking", 20.0),
        (4, "Checking the input", 0.5),
        (4, "Waiting for frames' .sift files", 12.0),
        (4, "Reading .sift metadata", 0.5),
        (4, "Reading .sift features", 0.5),
        (4, "Building KD-forests", 0.5),
        (4, "Matching frame pairs", 3.25),
        (4, "Measuring essential splits", 0.5),
        (4, "Pilot votes: clustering and focal vote", 0.5),
        (4, "Stopping the extraction workers", 0.5),
        (4, "The walk's decisions and the rest", 0.5),
        (4, "Handing the report to Python", 0.75),
        (2, "Printing the report, writing the list, JSON", 0.06),
        (2, "SIFT extraction, 4 workers on 70 frames", 41.25),
        (4, "Converting to sRGB", 20.0),
        (4, "Encoding and writing JPEGs", 9.75),
    ]
    other = lines.index(
        "Time on other threads, at the same time (summed over the threads):"
    )
    assert lines[other + 2] == "  pan.MOV: 4 writer threads"
    assert lines[other - 1].endswith("0.06 s")

    # With no videos, another tool's extraction first, a source of no
    # workers and no pilot vote, there are no video lines, no pilot line and
    # nothing on other threads.
    times["source"] = {"workers": 0, "frames": 0, "seconds": 0.0}
    command_times.update(
        preparing_seconds=None, videos=[], extracting_first_seconds=3.5
    )
    report["sequences"] = [{"pilot_vote": None}]
    lines = time_lines(report, command_times, 0.0625)
    assert not any(
        "video" in line or "other threads" in line or "Pilot" in line for line in lines
    )
    assert any(
        line.startswith("  Extracting every frame's .sift file first")
        and line.endswith("3.50 s")
        for line in lines
    )
