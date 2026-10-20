# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``point_columns`` binding.

Checks the dict's names, dtypes and rows against the reconstruction's own
arrays on the seoul_bull ground truth, the column selection and its
refusals, and that an ``EditedReconstruction`` is read through its edits.
See specs/core/analysis/point-columns.md.
"""

import numpy as np
import pytest

from sfmtool.analysis import point_columns
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction

ALL_COLUMNS = (
    "track_length",
    "image_count",
    "error_px",
    "at_infinity",
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
    cols = point_columns(recon)
    n = recon.point_count
    assert set(cols) <= set(ALL_COLUMNS) | {"point_index"}
    for name in ("track_length", "error_px", "x", "max_ray_angle_deg"):
        assert name in cols
    assert cols["point_index"].dtype == np.uint32
    np.testing.assert_array_equal(cols["point_index"], np.arange(n))
    for name, values in cols.items():
        assert values.shape == (n,), name
        if name in ("x", "y", "z"):
            assert values.dtype == np.float64, name
        elif name != "point_index":
            assert values.dtype == np.float32, name


def test_stored_columns_match_the_reconstruction(recon):
    cols = point_columns(recon)
    track_points = recon.track_point_indexes
    np.testing.assert_array_equal(
        cols["track_length"], np.bincount(track_points, minlength=recon.point_count)
    )
    np.testing.assert_array_equal(cols["error_px"], recon.errors)
    xyzw = recon.positions_xyzw
    at_infinity = xyzw[:, 3] == 0.0
    assert at_infinity.any(), "the ground truth holds points at infinity"
    np.testing.assert_array_equal(cols["at_infinity"], at_infinity.astype(np.float32))
    for k, name in enumerate("xyz"):
        np.testing.assert_array_equal(cols[name][~at_infinity], xyzw[~at_infinity, k])
        assert np.isnan(cols[name][at_infinity]).all()
    assert np.isnan(cols["centroid_distance"][at_infinity]).all()
    assert (cols["image_count"] <= cols["track_length"]).all()


def test_points_at_infinity_have_zero_ray_angle_and_no_diagnostics(recon):
    cols = point_columns(recon)
    at_infinity = cols["at_infinity"] == 1.0
    assert (cols["max_ray_angle_deg"][at_infinity] == 0.0).all()
    assert np.isnan(cols["condition_number"][at_infinity]).all()
    assert np.isnan(cols["inverse_depth_z"][at_infinity]).all()
    finite = ~at_infinity
    assert (cols["max_ray_angle_deg"][finite] > 0.0).any()
    assert np.isfinite(cols["condition_number"][finite]).any()


def test_named_columns_only(recon):
    cols = point_columns(recon, ["track_length", "x"])
    assert set(cols) == {"point_index", "track_length", "x"}


def test_unknown_column_is_refused(recon):
    with pytest.raises(ValueError, match="not a point column"):
        point_columns(recon, ["min_zncc"])


def test_absent_column_is_refused_when_named_and_skipped_otherwise(recon):
    bare = recon.clone_with_changes(normal_confidence=None)
    assert "normal_confidence" not in point_columns(bare)
    with pytest.raises(ValueError, match="cannot supply the column normal_confidence"):
        point_columns(bare, ["normal_confidence"])


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
    lowest = point_columns(changed, ["lowest_zncc"])["lowest_zncc"]
    assert lowest[0] == pytest.approx(51 / 255)
    assert lowest[1] == 1.0
    # Point 2 has every byte zero: nothing was measured.
    assert np.isnan(lowest[2])


def test_an_edited_reconstruction_is_read_through_its_edits(recon):
    edited = EditedReconstruction(recon)
    edited.delete_points([0, 2])
    cols = point_columns(edited, ["track_length"])
    assert cols["point_index"][:3].tolist() == [1, 3, 4]
    assert len(cols["track_length"]) == recon.point_count - 2


def test_other_types_are_refused():
    with pytest.raises(TypeError, match="SfmrReconstruction"):
        point_columns(object())
