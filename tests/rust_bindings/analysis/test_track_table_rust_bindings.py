# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``track_table`` binding.

Checks the dict's names, dtypes and rows against the reconstruction's own
arrays on the seoul_bull ground truth, the column selection and its
refusals, and that an ``EditedReconstruction`` is read through its edits.
See specs/core/analysis/track-table.md.
"""

import numpy as np
import pytest

from sfmtool.analysis import track_table
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction

ALL_COLUMNS = (
    "track_length",
    "repeat_observations",
    "rms_error_px",
    "x",
    "y",
    "z",
    "centroid_distance",
    "normal_confidence",
    "lowest_zncc",
    "largest_zncc_self_similarity_radius",
    "max_ray_angle_deg",
    "condition_number",
    "inverse_depth_z",
)


@pytest.fixture
def recon(seoul_bull_ground_truth_sfmr):
    return SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))


def test_every_supplied_column_is_returned_with_its_dtype(recon):
    table = track_table(recon)
    n = recon.point_count
    assert set(table) <= set(ALL_COLUMNS) | {"point_index"}
    for name in ("track_length", "rms_error_px", "x", "max_ray_angle_deg"):
        assert name in table
    assert table["point_index"].dtype == np.uint32
    np.testing.assert_array_equal(table["point_index"], np.arange(n))
    for name, values in table.items():
        assert values.shape == (n,), name
        if name in ("x", "y", "z"):
            assert values.dtype == np.float64, name
        elif name != "point_index":
            assert values.dtype == np.float32, name


def test_stored_columns_match_the_reconstruction(recon):
    table = track_table(recon)
    track_points = recon.track_point_indexes
    np.testing.assert_array_equal(
        table["track_length"], np.bincount(track_points, minlength=recon.point_count)
    )
    np.testing.assert_array_equal(table["rms_error_px"], recon.rms_reprojection_errors)
    xyzw = recon.positions_xyzw
    at_infinity = xyzw[:, 3] == 0.0
    assert at_infinity.any(), "the ground truth holds points at infinity"
    for k, name in enumerate("xyz"):
        np.testing.assert_array_equal(table[name][~at_infinity], xyzw[~at_infinity, k])
        assert np.isnan(table[name][at_infinity]).all()
    # A point at infinity is at distance +inf, and only a point at infinity is.
    np.testing.assert_array_equal(np.isposinf(table["centroid_distance"]), at_infinity)
    assert np.isfinite(table["centroid_distance"][~at_infinity]).all()
    # The ground truth has one observation per image in every track.
    assert (table["repeat_observations"] == 0.0).all()


def test_repeat_observations_counts_the_observations_beyond_one_per_image(recon):
    edited = EditedReconstruction(recon)
    record = edited.point(0)
    images = np.array(record["image_indexes"])
    assert len(images) >= 3
    # Point 0 again, with its second observation in the image of its first:
    # one image holding two observations of the point.
    images[1] = images[0]
    record["image_indexes"] = images.tolist()
    added = edited.add_point(record)
    table = track_table(edited, ["track_length", "repeat_observations"])
    repeats = dict(zip(table["point_index"].tolist(), table["repeat_observations"]))
    assert repeats[added] == 1.0
    assert sum(v != 0.0 for v in repeats.values()) == 1


def test_points_at_infinity_have_zero_ray_angle_and_no_diagnostics(recon):
    table = track_table(recon)
    at_infinity = recon.positions_xyzw[:, 3] == 0.0
    assert (table["max_ray_angle_deg"][at_infinity] == 0.0).all()
    assert np.isnan(table["condition_number"][at_infinity]).all()
    assert np.isnan(table["inverse_depth_z"][at_infinity]).all()
    finite = ~at_infinity
    assert (table["max_ray_angle_deg"][finite] > 0.0).any()
    assert np.isfinite(table["condition_number"][finite]).any()


def test_named_columns_only(recon):
    table = track_table(recon, ["track_length", "x"])
    assert set(table) == {"point_index", "track_length", "x"}


def test_unknown_column_is_refused(recon):
    with pytest.raises(ValueError, match="not a track table column"):
        track_table(recon, ["min_zncc"])


def test_absent_column_is_refused_when_named_and_skipped_otherwise(recon):
    bare = recon.clone_with_changes(normal_confidence=None)
    assert "normal_confidence" not in track_table(bare)
    with pytest.raises(ValueError, match="cannot supply the column normal_confidence"):
        track_table(bare, ["normal_confidence"])


def test_lowest_zncc_reads_measured_bytes_only(recon):
    m = recon.observation_count
    confidence = np.zeros(m, dtype=np.uint8)
    track_points = recon.track_point_indexes
    first = np.flatnonzero(track_points == 0)
    confidence[first[0]] = 51
    confidence[first[1:]] = 0
    second = np.flatnonzero(track_points == 1)
    confidence[second] = 255
    changed = recon.clone_with_changes(observation_confidence=confidence)
    lowest = track_table(changed, ["lowest_zncc"])["lowest_zncc"]
    assert lowest[0] == pytest.approx(51 / 255)
    assert lowest[1] == 1.0
    # Point 2 has every byte zero: nothing was measured.
    assert np.isnan(lowest[2])


def test_an_edited_reconstruction_is_read_through_its_edits(recon):
    edited = EditedReconstruction(recon)
    edited.delete_points([0, 2])
    table = track_table(edited, ["track_length"])
    assert table["point_index"][:3].tolist() == [1, 3, 4]
    assert len(table["track_length"]) == recon.point_count - 2


def test_other_types_are_refused():
    with pytest.raises(TypeError, match="SfmrReconstruction"):
        track_table(object())
