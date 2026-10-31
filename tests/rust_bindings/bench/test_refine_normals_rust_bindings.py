# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfmtool.bench.refine_normals``: every finite point of a
reconstruction, or the points of a list, turned to its grown-plane normal and
written back as one new value.

They run on the seoul_bull ground truth and its photographs. The ground truth
is read in place and never written, so one load serves the module.
"""

import numpy as np
import pytest

from sfmtool import ProgressCounter
from sfmtool.bench import refine_normals
from sfmtool.fileio import read_image_rgb
from sfmtool.reconstruction import SfmrReconstruction
from tests.conftest import SEOUL_BULL_DIR, SEOUL_BULL_GROUND_TRUTH

OUTCOMES = {
    "turned",
    "none_joined",
    "weak_turn",
    "not_refined",
    "no_bitmap",
    "at_infinity",
    "not_asked",
}


def _load():
    return SfmrReconstruction.load(str(SEOUL_BULL_GROUND_TRUTH))


@pytest.fixture(scope="module")
def ground_truth():
    return _load()


@pytest.fixture(scope="module")
def images(ground_truth):
    """The ground truth's photographs, one per image."""
    return [read_image_rgb(SEOUL_BULL_DIR / name) for name in ground_truth.image_names]


@pytest.fixture(scope="module")
def refined(ground_truth, images):
    return refine_normals(ground_truth, images)


def test_a_refinement_turns_normals_and_moves_no_point_or_keypoint(
    ground_truth, refined
):
    out, report = refined
    assert isinstance(out, SfmrReconstruction)
    assert out.point_count == ground_truth.point_count
    for column in ("track_image_indexes", "track_point_indexes"):
        np.testing.assert_array_equal(
            np.asarray(getattr(out, column)), np.asarray(getattr(ground_truth, column))
        )
    np.testing.assert_array_equal(
        np.asarray(out.positions_xyzw), np.asarray(ground_truth.positions_xyzw)
    )
    np.testing.assert_array_equal(
        np.asarray(out.keypoints_xy), np.asarray(ground_truth.keypoints_xy)
    )
    # A point turned and rewritten has a new normal; every other point keeps
    # the one it had.
    before = np.asarray(ground_truth.normals)
    after = np.asarray(out.normals)
    rewritten = [
        p["index"]
        for p in report["points"]
        if p["outcome"] == "turned" and not p["unchanged"]
    ]
    assert rewritten
    assert not np.allclose(after[rewritten], before[rewritten])
    kept = np.ones(ground_truth.point_count, dtype=bool)
    kept[rewritten] = False
    np.testing.assert_array_equal(after[kept], before[kept])
    # The value the run was handed is as it was loaded.
    np.testing.assert_array_equal(before, np.asarray(_load().normals))


def test_the_report_holds_one_status_per_point_and_counts_them(ground_truth, refined):
    _, report = refined
    points = report["points"]
    assert [p["index"] for p in points] == list(range(ground_truth.point_count))
    outcomes = [p["outcome"] for p in points]
    assert set(outcomes) <= OUTCOMES
    for word in ("turned", "none_joined", "weak_turn", "at_infinity", "not_asked"):
        assert report[word] == outcomes.count(word), word
    assert report["kept"] == outcomes.count("not_refined") + outcomes.count("no_bitmap")
    assert report["not_asked"] == 0
    assert report["at_infinity"] == ground_truth.infinity_point_count
    assert report["read"] == ground_truth.point_count - report["at_infinity"]
    assert report["read"] == (
        report["turned"] + report["none_joined"] + report["weak_turn"] + report["kept"]
    )
    assert report["turned"] * 10 >= report["read"] * 8, report["read"]

    turned = [p for p in points if p["outcome"] == "turned"]
    assert report["unchanged"] == sum(p["unchanged"] for p in turned)
    assert report["low_confidence"] == sum(p["low_confidence"] for p in turned)
    assert all(p["joined"] >= 1 for p in turned)
    assert report["median_turn_deg"] == pytest.approx(
        np.median([p["turn_deg"] for p in turned])
    )
    for p in points:
        if p["outcome"] == "weak_turn":
            assert p["turn_deg"] >= 0.0
        if p["outcome"] == "not_refined":
            assert p["error"] and p["reason"]


def test_a_turned_points_agreement_is_a_finite_score(refined):
    _, report = refined
    agreements = [p["agreement"] for p in report["points"] if p["outcome"] == "turned"]
    assert agreements
    assert all(np.isfinite(a) and -1.0 <= a <= 1.0 for a in agreements)
    assert report["mean_agreement"] == pytest.approx(np.mean(agreements))


def test_a_list_of_points_refines_those_and_leaves_the_rest(
    ground_truth, images, refined
):
    _, full = refined
    finite = np.flatnonzero(~np.asarray(ground_truth.point_is_at_infinity))
    at_infinity = int(np.flatnonzero(np.asarray(ground_truth.point_is_at_infinity))[0])
    listed = [int(finite[3]), int(finite[10]), at_infinity, int(finite[3])]

    out, report = refine_normals(ground_truth, images, point_indexes=listed)
    asked = set(listed)
    assert report["not_asked"] == ground_truth.point_count - len(asked)
    assert report["at_infinity"] == 1
    assert report["read"] == 2
    for p in report["points"]:
        if p["index"] in asked:
            # A listed point gets the answer a run over every point gives it.
            assert p == full["points"][p["index"]]
        else:
            assert p["outcome"] == "not_asked"
    before = np.asarray(ground_truth.normals)
    after = np.asarray(out.normals)
    unlisted = np.ones(ground_truth.point_count, dtype=bool)
    unlisted[list(asked)] = False
    np.testing.assert_array_equal(after[unlisted], before[unlisted])
    np.testing.assert_array_equal(
        np.asarray(out.positions_xyzw), np.asarray(ground_truth.positions_xyzw)
    )

    # An empty list refines nothing.
    _, none = refine_normals(ground_truth, images, point_indexes=[])
    assert none["read"] == 0
    assert none["not_asked"] == ground_truth.point_count


def test_a_run_it_cannot_do_is_refused(ground_truth, images):
    with pytest.raises(ValueError, match="was asked for"):
        refine_normals(ground_truth, images, point_indexes=[ground_truth.point_count])
    with pytest.raises(ValueError, match="spot_window"):
        refine_normals(ground_truth, images, spot_window=0)
    with pytest.raises(ValueError, match="images must be parallel"):
        refine_normals(ground_truth, images[:-1])


def test_a_point_the_step_refuses_names_the_refusal_and_keeps_its_columns(
    ground_truth, images
):
    # Point 0 keeps only its first observation.
    points = np.asarray(ground_truth.track_point_indexes)
    first = np.flatnonzero(points == 0)[0]
    keep = (points != 0) | (np.arange(len(points)) == first)
    one = ground_truth.clone_with_changes(
        track_image_indexes=np.asarray(ground_truth.track_image_indexes)[keep],
        track_feature_indexes=np.zeros(int(keep.sum()), np.uint32),
        track_point_indexes=points[keep],
        keypoints_xy=np.asarray(ground_truth.keypoints_xy)[keep],
    )

    out, report = refine_normals(one, images, point_indexes=[0])
    status = report["points"][0]
    assert status["outcome"] == "not_refined"
    assert status["error"] == "too_few_observations"
    assert status["reason"]
    assert report["kept"] == 1
    np.testing.assert_array_equal(
        np.asarray(out.normals)[0], np.asarray(one.normals)[0]
    )


def test_the_progress_counter_reaches_the_points_refined(ground_truth, images):
    finite = np.flatnonzero(~np.asarray(ground_truth.point_is_at_infinity))
    counter = ProgressCounter()
    _, report = refine_normals(
        ground_truth, images, point_indexes=finite[:20].tolist() * 2, progress=counter
    )
    # Each listed point is counted once.
    assert counter.value == report["read"] == 20
