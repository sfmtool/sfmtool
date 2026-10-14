# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfmtool.bench.refit_points``: every point of a reconstruction
put through the bench's fit and commit, written back as one new value.

They run on the seoul_bull ground truth and its photographs. The ground truth
is read in place and never written, so one load serves the module.
"""

import numpy as np
import pytest

from sfmtool.bench import refit_points
from sfmtool.fileio import POINT_CONSTRAINT_NAMES, read_image_rgb
from sfmtool.reconstruction import SfmrReconstruction
from tests.conftest import SEOUL_BULL_DIR, SEOUL_BULL_GROUND_TRUTH


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
def refit(ground_truth, images):
    return refit_points(ground_truth, images)


def test_a_refit_moves_no_index_and_leaves_its_input_as_it_was(ground_truth, refit):
    out, _ = refit
    assert isinstance(out, SfmrReconstruction)
    assert out.point_count == ground_truth.point_count
    for column in ("track_image_indexes", "track_point_indexes"):
        np.testing.assert_array_equal(
            np.asarray(getattr(out, column)), np.asarray(getattr(ground_truth, column))
        )
    # The keypoints were read again from the photographs, and the value the
    # run was handed still holds the ones it was loaded with.
    assert not np.array_equal(
        np.asarray(out.keypoints_xy), np.asarray(ground_truth.keypoints_xy)
    )
    fresh = _load()
    np.testing.assert_array_equal(
        np.asarray(ground_truth.keypoints_xy), np.asarray(fresh.keypoints_xy)
    )
    np.testing.assert_array_equal(
        np.asarray(ground_truth.positions_xyzw), np.asarray(fresh.positions_xyzw)
    )


def test_the_report_holds_one_status_per_point_and_counts_them(ground_truth, refit):
    _, report = refit
    points = report["points"]
    assert [p["index"] for p in points] == list(range(ground_truth.point_count))
    outcomes = [p["outcome"] for p in points]
    assert report["refit"] == outcomes.count("refit")
    assert report["held"] == outcomes.count("held") == 0
    assert report["kept"] == outcomes.count("not_fitted") + outcomes.count("no_bitmap")
    assert report["read"] == ground_truth.point_count
    assert report["read"] == report["refit"] + report["kept"]
    assert report["refit"] * 10 >= report["read"] * 9, report["read"]

    refit_rows = [p for p in points if p["outcome"] == "refit"]
    assert report["unchanged"] == sum(p["unchanged"] for p in refit_rows)
    assert report["crossed"] == sum(p["crossed"] for p in refit_rows)
    assert report["kept_at_seed"] == sum(p["kept_at_seed"] for p in refit_rows)
    assert all(p["range"] == "free" and not p["released"] for p in refit_rows)
    for p in refit_rows:
        # A free point's coordinate is the fit's classification, and a shift
        # is measured only between two positions.
        call = p["classification"]
        assert call is not None
        assert (p["shift"] is None) == (call["at_infinity"] or p["crossed"])
    shifts = [p["shift"] for p in refit_rows if p["shift"] is not None]
    assert report["median_shift"] == pytest.approx(np.median(shifts))
    for p in points:
        if p["outcome"] == "not_fitted":
            assert p["error"] and p["reason"]


def test_a_held_point_is_left_out(ground_truth, images):
    n = ground_truth.point_count
    recon = ground_truth.clone_with_changes(
        point_constraints=np.full(n, POINT_CONSTRAINT_NAMES.index("held"), np.uint8),
        constraint_distances=np.full(n, np.nan),
        constraint_reference_images=np.full(n, np.iinfo(np.uint32).max, np.uint32),
    )
    out, report = refit_points(recon, images, sigma_px=1.0)
    assert report["held"] == n
    assert report["read"] == 0
    assert {p["outcome"] for p in report["points"]} == {"held"}
    np.testing.assert_array_equal(
        np.asarray(out.positions_xyzw), np.asarray(recon.positions_xyzw)
    )
    np.testing.assert_array_equal(
        np.asarray(out.keypoints_xy), np.asarray(recon.keypoints_xy)
    )


def test_a_run_it_cannot_do_is_refused(ground_truth, images):
    with pytest.raises(ValueError, match="cannot be classified"):
        refit_points(ground_truth, images, sigma_px=0.0)
    with pytest.raises(ValueError, match="images must be parallel"):
        refit_points(ground_truth, images[:-1])


def _with_tracks(recon, image_indexes, rows):
    """`recon` with its observation rows taken as `rows` selects them (a mask
    or an index array) and their images replaced by `image_indexes`, as a
    script that edits tracks by hand makes it.
    """
    points = np.asarray(recon.track_point_indexes)[rows]
    return recon.clone_with_changes(
        track_image_indexes=np.asarray(image_indexes, np.uint32)[rows],
        track_feature_indexes=np.zeros(len(points), np.uint32),
        track_point_indexes=points,
        keypoints_xy=np.asarray(recon.keypoints_xy)[rows],
    )


def test_a_point_the_fit_refuses_names_the_refusal_and_keeps_its_columns(
    ground_truth, images
):
    # Point 0 keeps only its first observation.
    points = np.asarray(ground_truth.track_point_indexes)
    first = np.flatnonzero(points == 0)[0]
    keep = (points != 0) | (np.arange(len(points)) == first)
    one = _with_tracks(ground_truth, ground_truth.track_image_indexes, keep)

    out, report = refit_points(one, images, sigma_px=0.5)
    status = report["points"][0]
    assert status["outcome"] == "not_fitted"
    assert status["error"] == "too_few_observations"
    assert "two or more" in status["reason"]
    assert report["kept"] == 1
    np.testing.assert_array_equal(
        np.asarray(out.positions_xyzw)[0], np.asarray(one.positions_xyzw)[0]
    )


def test_a_track_naming_an_image_past_the_table_cannot_reach_the_run(ground_truth):
    # The value is refused where it is built, so the run's own
    # `ImageOutOfRange` precondition is left for Rust callers that write the
    # tracks field directly.
    images_of = np.asarray(ground_truth.track_image_indexes).copy()
    images_of[0] = len(ground_truth.image_names)
    everything = np.ones(len(images_of), dtype=bool)
    with pytest.raises(ValueError, match="image index 17 out of range for 17 images"):
        _with_tracks(ground_truth, images_of, everything)


def test_a_track_stored_out_of_image_order_is_written_back_at_its_own_rows(
    ground_truth, images
):
    # Point 0's rows reversed, every column with them: rows the reader would
    # have sorted, left unsorted in memory.
    n = int(np.asarray(ground_truth.observation_counts)[0])
    order = np.arange(len(np.asarray(ground_truth.track_point_indexes)))
    order[:n] = order[:n][::-1]
    reversed_ = _with_tracks(ground_truth, ground_truth.track_image_indexes, order)

    want, _ = refit_points(ground_truth, images, sigma_px=0.5)
    got, report = refit_points(reversed_, images, sigma_px=0.5)
    assert report["points"][0]["outcome"] == "refit"
    np.testing.assert_array_equal(
        np.asarray(got.track_image_indexes)[:n],
        np.asarray(want.track_image_indexes)[:n][::-1],
    )
    np.testing.assert_array_equal(
        np.asarray(got.keypoints_xy)[:n], np.asarray(want.keypoints_xy)[:n][::-1]
    )
    # The triangulation sums point 0's rays in stored order, so its
    # coordinate agrees to rounding.
    np.testing.assert_allclose(
        np.asarray(got.positions_xyzw), np.asarray(want.positions_xyzw), rtol=1e-12
    )
