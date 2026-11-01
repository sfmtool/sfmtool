# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the break verdict of `sfm motion FILE.sfmr`.

The rule is documented under "Step 4: Break Verdict" in
specs/cli/reconstruction/motion-command.md.
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.motion.constants import (
    BREAK_ROTATION_FACTOR,
    BREAK_SHARED_FRACTION,
    BREAK_WINDOW,
)
from sfmtool.motion.recon_discontinuity import (
    _break_reference,
    _classify_break,
    _nearby_shared_point_counts,
)


@pytest.fixture
def runner():
    return CliRunner()


def _classify_sequence(rotations, frame_numbers, shared, nearby=None):
    """Classify every consecutive edge of a constructed sequence.

    ``rotations[k]``, ``shared[k]`` and ``nearby[k]`` describe the edge from
    ``frame_numbers[k]`` to ``frame_numbers[k + 1]``; ``nearby`` defaults to
    twice ``shared``, a video's points that live a few images past the pair.
    Returns the reference and one classification per edge.
    """
    if nearby is None:
        nearby = [2 * int(c) for c in shared]
    gaps = [b - a for a, b in zip(frame_numbers, frame_numbers[1:])]
    reference = _break_reference(list(rotations), gaps, list(shared), list(nearby))
    verdicts = [
        _classify_break(
            rotation_deg=r,
            frame_gap=g,
            shared_points=s,
            nearby_shared_points=c,
            reference=reference,
        )
        for r, g, s, c in zip(rotations, gaps, shared, nearby)
    ]
    return reference, verdicts


def _break_indexes(verdicts):
    return [k for k, v in enumerate(verdicts) if v["is_break"]]


# --- The classifier on constructed sequences ---


def test_clean_loop_has_no_breaks():
    """A camera turning steadily round a loop, every pair well connected."""
    rng = np.random.default_rng(0)
    frames = list(range(0, 7 * 60, 7))
    rotations = 5.0 + rng.normal(0.0, 0.5, len(frames) - 1)
    shared = rng.integers(400, 800, len(frames) - 1)
    reference, verdicts = _classify_sequence(rotations, frames, shared)
    assert reference["median_frame_gap"] == 7
    assert reference["rotation_per_frame_deg"] == pytest.approx(5.0 / 7, rel=0.1)
    assert _break_indexes(verdicts) == []


def test_real_break_is_found():
    """One pair that turns far more than usual and shares almost nothing."""
    frames = list(range(0, 7 * 40, 7))
    rotations = [5.0] * (len(frames) - 1)
    shared = [500] * (len(frames) - 1)
    rotations[20] = 60.0
    shared[20] = 3
    _, verdicts = _classify_sequence(rotations, frames, shared)
    assert _break_indexes(verdicts) == [20]
    v = verdicts[20]
    assert v["expected_rotation_deg"] == pytest.approx(5.0)
    assert v["rotation_ratio"] == pytest.approx(12.0)
    assert v["shared_points_threshold"] == pytest.approx(BREAK_SHARED_FRACTION * 500)


def test_both_conditions_are_needed():
    """A large turn that keeps its shared points, and a pair that shares
    nothing but turns as usual, are not breaks."""
    frames = list(range(0, 7 * 40, 7))
    rotations = [5.0] * (len(frames) - 1)
    shared = [500] * (len(frames) - 1)
    rotations[10] = 5.0 * (BREAK_ROTATION_FACTOR + 1)
    shared[10] = 300
    rotations[25] = 5.0
    shared[25] = 0
    _, verdicts = _classify_sequence(rotations, frames, shared)
    assert _break_indexes(verdicts) == []
    assert verdicts[10]["rotation_ratio"] > BREAK_ROTATION_FACTOR
    assert verdicts[25]["shared_points"] < verdicts[25]["shared_points_threshold"]


def test_photo_set_move_seen_from_nearby_images_is_not_a_break():
    """A photo set that starts a second ring round the subject: the pair at
    the start turns six times the usual rotation and shares nothing, but the
    images a few steps either side of it still see the same points."""
    frames = list(range(1, 86))
    rotations = [10.0] * 84
    shared = [800] * 84
    nearby = [3000] * 84
    rotations[40] = 60.0
    shared[40] = 4
    _, verdicts = _classify_sequence(rotations, frames, shared, nearby)
    assert verdicts[40]["rotation_ratio"] > BREAK_ROTATION_FACTOR
    assert verdicts[40]["shared_points"] < verdicts[40]["shared_points_threshold"]
    assert _break_indexes(verdicts) == []

    nearby[40] = 30
    _, verdicts = _classify_sequence(rotations, frames, shared, nearby)
    assert verdicts[40]["nearby_shared_points_threshold"] == pytest.approx(
        BREAK_SHARED_FRACTION * 3000
    )
    assert _break_indexes(verdicts) == [40]


def test_dropped_frames_with_a_smooth_turn_are_not_a_break():
    """Frames 1081-1099 missing from a stride-9 solve: the pair across the
    gap turns over four strides and shares no points, but turns only as much
    as four strides of the usual rate."""
    frames = list(range(1000, 1081, 9)) + list(range(1108, 1200, 9))
    gaps = [b - a for a, b in zip(frames, frames[1:])]
    assert 1081 - 9 in frames and 1108 in frames and 36 in gaps
    rate = 0.5  # degrees per frame number
    rotations = [rate * g for g in gaps]
    shared = [700 if g == 9 else 0 for g in gaps]
    _, verdicts = _classify_sequence(rotations, frames, shared)
    gap_edge = gaps.index(36)
    assert verdicts[gap_edge]["expected_rotation_deg"] == pytest.approx(rate * 36)
    assert verdicts[gap_edge]["rotation_ratio"] == pytest.approx(1.0)
    assert _break_indexes(verdicts) == []


def test_swapped_frames_are_judged_by_their_frame_gap():
    """Blurred frames swapped for neighbours make the gaps vary around the
    stride.  The rate comes from rotation per frame number, so uneven gaps
    raise no breaks, and a real break among them is still found."""
    gaps = [7, 7, 6, 8, 7, 9, 5, 7, 7, 8, 6] * 4
    frames = [100]
    for g in gaps:
        frames.append(frames[-1] + g)
    rate = 0.75
    rotations = [rate * g for g in gaps]
    shared = [450] * len(gaps)
    _, verdicts = _classify_sequence(rotations, frames, shared)
    assert _break_indexes(verdicts) == []
    assert all(
        v["rotation_ratio"] == pytest.approx(g / 7 if g < 7 else 1.0)
        for v, g in zip(verdicts, gaps)
    )

    rotations[17] = 45.0
    shared[17] = 10
    _, verdicts = _classify_sequence(rotations, frames, shared)
    assert _break_indexes(verdicts) == [17]


def test_closely_numbered_pair_gets_a_typical_pair_rotation():
    """A pair closer in frame number than the typical pair is allowed a
    typical pair's rotation, not a fraction of it."""
    frames = [0, 50, 100, 101, 150, 200, 250]
    rotations = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
    shared = [5, 5, 0, 5, 5, 5]
    reference, verdicts = _classify_sequence(rotations, frames, shared)
    assert reference["median_frame_gap"] == 50
    assert verdicts[2]["expected_rotation_deg"] == pytest.approx(40.0)
    assert _break_indexes(verdicts) == []


def test_photo_set_that_is_not_a_sequence_has_no_breaks():
    """Numbered photos in no order: consecutive pairs turn by anything and
    share few points.  The median rotation is then large, and no rotation can
    reach `BREAK_ROTATION_FACTOR` times it."""
    rng = np.random.default_rng(1)
    n = 60
    frames = list(range(1, n + 1))
    rotations = rng.uniform(0.0, 180.0, n - 1)
    shared = rng.integers(0, 40, n - 1)
    reference, verdicts = _classify_sequence(rotations, frames, shared)
    assert 180.0 / reference["rotation_per_frame_deg"] < BREAK_ROTATION_FACTOR
    assert _break_indexes(verdicts) == []


def test_cameras_that_do_not_turn():
    """With no rotation anywhere the expected rotation is 0: a pair that does
    not turn is not a break, and a pair that turns and shares nothing is."""
    frames = list(range(10))
    rotations = [0.0] * 9
    shared = [100] * 9
    _, verdicts = _classify_sequence(rotations, frames, shared)
    assert _break_indexes(verdicts) == []
    rotations[4] = 30.0
    shared[4] = 0
    _, verdicts = _classify_sequence(rotations, frames, shared)
    assert verdicts[4]["rotation_ratio"] == float("inf")
    assert _break_indexes(verdicts) == [4]


# --- Points near an edge ---


class _Tracks:
    """The three track fields `_nearby_shared_point_counts` reads."""

    def __init__(self, tracks, image_count):
        self.observation_counts = np.array([len(t) for t in tracks])
        self.track_image_indexes = np.array([i for t in tracks for i in t])
        self.image_count = image_count


def _nearby_by_sets(tracks, seq_frames, image_frames, window):
    """The count by its definition, with sets."""
    n = len(seq_frames)
    out = []
    for k in range(n - 1):
        lo = seq_frames[max(0, k - window + 1)]
        hi = seq_frames[min(n - 1, k + window)]
        count = 0
        for t in tracks:
            frames = [image_frames[i] for i in t]
            before = any(lo <= f <= seq_frames[k] for f in frames)
            after = any(seq_frames[k + 1] <= f <= hi for f in frames)
            count += before and after
        out.append(count)
    return out


def test_nearby_counts_match_their_definition():
    rng = np.random.default_rng(2)
    seq_frames = sorted(rng.choice(np.arange(1, 200), size=40, replace=False))
    image_frames = np.array(seq_frames)
    tracks = []
    for _ in range(300):
        start = int(rng.integers(0, 40))
        length = int(rng.integers(2, 12))
        members = sorted(set(range(start, min(40, start + length))))
        if len(members) >= 2:
            tracks.append(members)
    recon = _Tracks(tracks, 40)
    for window in (1, 3, BREAK_WINDOW):
        assert _nearby_shared_point_counts(
            recon, seq_frames, image_frames, window
        ) == _nearby_by_sets(tracks, seq_frames, image_frames, window)


def test_nearby_counts_include_the_other_sensors_of_a_rig():
    """Images 0-9 are one sensor at frames 1-10, images 10-19 another at the
    same frames.  A track seen by the second sensor at frame 4 and the first at
    frame 7 counts for the first sensor's edges 4->5, 5->6 and 6->7, and a
    track seen only by the second sensor, at frames 2 and 9, counts for the
    one edge whose windows of 4 frames reach both, 5->6."""
    image_frames = np.array(list(range(1, 11)) * 2)
    seq_frames = list(range(1, 11))
    tracks = [[13, 6], [11, 18]]
    counts = _nearby_shared_point_counts(
        _Tracks(tracks, 20), seq_frames, image_frames, 4
    )
    assert counts == _nearby_by_sets(tracks, seq_frames, image_frames, 4)
    assert counts == [0, 0, 0, 1, 2, 1, 0, 0, 0]


def test_nearby_counts_skip_images_without_a_frame_number():
    image_frames = np.array([1, 2, 3, 4, -1])
    counts = _nearby_shared_point_counts(
        _Tracks([[0, 4], [1, 2]], 5), [1, 2, 3, 4], image_frames, 2
    )
    assert counts == [0, 1, 0]


# --- On reconstructions ---


def _plant_break_at_10_11(recon, rotate_deg=90.0):
    """Turn frames 11-17 about z and drop every point frames 10 and 11 share.

    Moving poses keeps the tracks, so the two images would still share
    points; a solve that put them in inconsistent places does not tie them
    together, so the shared points are removed too.
    """
    from sfmtool._filenames import number_from_filename
    from sfmtool.geometry import RotQuaternion, Se3Transform

    names = recon.image_names
    idx = {number_from_filename(n): i for i, n in enumerate(names)}
    counts = np.asarray(recon.observation_counts, dtype=np.int64)
    images = np.asarray(recon.track_image_indexes, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    keep = np.ones(len(counts), dtype=bool)
    for p in range(len(counts)):
        seen = set(images[offsets[p] : offsets[p + 1]].tolist())
        if idx[10] in seen and idx[11] in seen:
            keep[p] = False
    recon = recon.filter_points_by_mask(keep)

    xform = Se3Transform.from_axis_angle(
        np.array([0.0, 0.0, 1.0]), np.radians(rotate_deg)
    )
    quats = recon.quaternions_wxyz.copy()
    trans = recon.translations.copy()
    for i, name in enumerate(names):
        if number_from_filename(name) >= 11:
            q = RotQuaternion.from_wxyz_array(quats[i])
            center = xform.apply_to_point(q.camera_center(trans[i]))
            new_q = xform.rotation @ q
            quats[i] = new_q.to_wxyz_array()
            trans[i] = -np.array(new_q.to_rotation_matrix()) @ center
    return recon.clone_with_changes(quaternions_wxyz=quats, translations=trans)


def test_ground_truth_photos_have_no_breaks(seoul_bull_ground_truth_sfmr):
    """The Seoul bull ground truth, photos around a sculpture with points at
    infinity, reports no breaks."""
    from sfmtool.motion.recon_discontinuity import _break_edges, analyze_reconstruction
    from sfmtool.reconstruction import SfmrReconstruction

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    assert recon.infinity_point_count > 0
    results = analyze_reconstruction(recon)
    assert _break_edges(results) == []


def test_points_at_infinity_count_as_shared_points(seoul_bull_ground_truth_sfmr):
    """Storing every point as a direction (w = 0) changes no shared count and
    so no verdict: the verdict reads tracks and poses, never positions."""
    from sfmtool.motion.recon_discontinuity import _break_edges, analyze_reconstruction
    from sfmtool.reconstruction import SfmrReconstruction

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    planted = _plant_break_at_10_11(recon)
    xyzw = np.asarray(planted.positions_xyzw, dtype=np.float64).copy()
    finite = xyzw[:, 3] != 0
    xyzw[finite, :3] /= np.linalg.norm(xyzw[finite, :3], axis=1, keepdims=True)
    xyzw[:, 3] = 0.0
    directions = planted.clone_with_changes(positions=xyzw)
    assert directions.infinity_point_count == directions.point_count

    before = analyze_reconstruction(planted)
    after = analyze_reconstruction(directions)
    assert [(a, b) for _, a, b in _break_edges(before)] == [(10, 11)]
    assert before[0]["break_reference"] == after[0]["break_reference"]
    assert _break_edges(before) == _break_edges(after)


def test_cli_verdict_line_and_json_on_a_sound_file(
    runner, seoul_bull_ground_truth_sfmr, tmp_path
):
    json_out = tmp_path / "report.json"
    result = runner.invoke(
        main, ["motion", str(seoul_bull_ground_truth_sfmr), "--json", str(json_out)]
    )
    assert result.exit_code == 0, result.output
    assert "Verdict: no breaks" in result.output
    report = json.loads(json_out.read_text())
    assert report["break_count"] == 0
    assert report["breaks"] == []
    assert report["thresholds"]["break_rotation_factor"] == BREAK_ROTATION_FACTOR
    assert report["thresholds"]["break_shared_fraction"] == BREAK_SHARED_FRACTION
    assert report["thresholds"]["break_window"] == BREAK_WINDOW
    seq = report["sequences"][0]
    assert seq["break_count"] == 0
    assert seq["median_frame_gap"] == 1
    assert seq["rotation_per_frame_deg"] > 0
    assert seq["break_shared_points_threshold"] == pytest.approx(
        BREAK_SHARED_FRACTION * seq["median_shared_points"]
    )
    assert seq["break_nearby_shared_points_threshold"] == pytest.approx(
        BREAK_SHARED_FRACTION * seq["median_nearby_shared_points"]
    )


def test_cli_verdict_line_and_json_on_a_broken_file(
    runner, seoul_bull_ground_truth_sfmr, tmp_path
):
    from sfmtool.reconstruction import SfmrReconstruction

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    broken = seoul_bull_ground_truth_sfmr.with_name("broken.sfmr")
    _plant_break_at_10_11(recon).save(broken)

    json_out = tmp_path / "report.json"
    result = runner.invoke(main, ["motion", str(broken), "--json", str(json_out)])
    assert result.exit_code == 0, result.output
    assert "Verdict: 1 break: 10->11" in result.output
    assert "BREAK" in result.output

    report = json.loads(json_out.read_text())
    assert report["break_count"] == 1
    assert report["breaks"] == [
        {"sequence": report["sequences"][0]["name"], "frame_a": 10, "frame_b": 11}
    ]
    seq = report["sequences"][0]
    assert seq["break_count"] == 1
    (row,) = [d for d in seq["discontinuities"] if d["is_break"]]
    assert (row["frame_a"], row["frame_b"]) == (10, 11)
    assert row["frame_gap"] == 1
    assert row["shared_points"] == 0
    assert row["rotation_ratio"] > BREAK_ROTATION_FACTOR
    assert row["expected_rotation_deg"] == pytest.approx(seq["rotation_per_frame_deg"])
    for d in seq["discontinuities"]:
        assert {
            "frame_gap",
            "expected_rotation_deg",
            "rotation_ratio",
            "nearby_shared_points",
            "is_break",
        } <= (d.keys())
