# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for `.sfmr` measurements, the distance kind, on
`SfmrReconstruction` and `EditedReconstruction`."""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.fileio import LENGTH_UNIT_NAMES, read_sfmr, read_sfmr_metadata, verify_sfmr
from sfmtool.geometry import CameraIntrinsics, RotQuaternion, Se3Transform
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction


def _sample_measurements() -> dict:
    """The columns of three valid distance measurements on the 17-image
    (270x480) seoul_bull dataset: the value of `measurements["distance"]`."""
    return {
        "distances": np.array([0.42, 12.5, 3.0], dtype=np.float32),
        "units": np.array(
            [
                LENGTH_UNIT_NAMES.index("m"),
                LENGTH_UNIT_NAMES.index("cm"),
                LENGTH_UNIT_NAMES.index("ft"),
            ],
            dtype=np.uint8,
        ),
        "image_indexes": np.array([[0, 3], [2, 2], [5, 7]], dtype=np.uint32),
        "endpoints_xy": np.array(
            [
                [[10.5, 20.25], [100.0, 200.5]],
                [[120.0, 220.0], [200.5, 350.25]],
                [[50.0, 60.0], [70.0, 80.0]],
            ],
            dtype=np.float32,
        ),
        "labels": ["ruler edge", "", "door — height"],
    }


def _assert_measurements_equal(measurements: dict, rhs: dict) -> None:
    """`measurements`, a dict of kinds, holds the distance columns `rhs` and
    no other kind."""
    assert set(measurements.keys()) == {"distance"}
    lhs = measurements["distance"]
    assert set(lhs.keys()) == {
        "distances",
        "units",
        "image_indexes",
        "endpoints_xy",
        "labels",
    }
    assert set(rhs.keys()) == set(lhs.keys())
    for key in ("distances", "units", "image_indexes", "endpoints_xy"):
        assert lhs[key].dtype == rhs[key].dtype, key
        assert lhs[key].shape == rhs[key].shape, key
        assert lhs[key].tobytes() == rhs[key].tobytes(), key
    assert lhs["labels"] == rhs["labels"]


class TestSfmrReconstructionMeasurements:
    def test_load_save_and_getters(self, seoul_bull_ground_truth_sfmr, tmp_path):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        assert recon.distance_measurement_count == 0
        assert recon.measurements is None
        assert "distance_measurement_count" not in recon.metadata()

        m = _sample_measurements()
        with_m = recon.clone_with_changes(measurements={"distance": m})
        assert with_m.distance_measurement_count == 3
        assert with_m.metadata()["distance_measurement_count"] == 3
        got = with_m.measurements
        assert got is not None
        _assert_measurements_equal(got, m)

        # Each read returns a fresh dict so mutating it does not touch the recon.
        got["distance"]["distances"][0] = 999.0
        got["distance"]["labels"][0] = "mutated"
        got["distance"]["extra"] = True
        got["gps"] = True
        _assert_measurements_equal(with_m.measurements, m)

        out = tmp_path / "saved.sfmr"
        with_m.save(out, operation="xform")
        assert verify_sfmr(out) == (True, [])
        assert read_sfmr_metadata(out)["distance_measurement_count"] == 3

        loaded = SfmrReconstruction.load(out)
        assert loaded.distance_measurement_count == 3
        assert loaded.metadata()["distance_measurement_count"] == 3
        _assert_measurements_equal(loaded.measurements, m)

    def test_from_data_with_and_without_measurements(
        self, seoul_bull_ground_truth_sfmr
    ):
        ws = Path(seoul_bull_ground_truth_sfmr).parent
        data = read_sfmr(seoul_bull_ground_truth_sfmr)
        m = _sample_measurements()

        recon = SfmrReconstruction.from_data(
            ws, dict(data, measurements={"distance": m})
        )
        assert recon.distance_measurement_count == 3
        assert recon.metadata()["distance_measurement_count"] == 3
        _assert_measurements_equal(recon.measurements, m)

        # Dropping measurements clears distance_measurement_count in sync.
        data_with_stale_count = dict(
            data,
            metadata=dict(data["metadata"], distance_measurement_count=3),
            measurements=None,
        )
        cleared = SfmrReconstruction.from_data(ws, data_with_stale_count)
        assert cleared.distance_measurement_count == 0
        assert cleared.measurements is None
        assert "distance_measurement_count" not in cleared.metadata()

        # Invalid measurements refused by from_data with ValueError.
        bad = _sample_measurements()
        bad["image_indexes"] = bad["image_indexes"].copy()
        bad["image_indexes"][0, 1] = len(data["image_names"])
        with pytest.raises(ValueError, match="image_indexes"):
            SfmrReconstruction.from_data(ws, dict(data, measurements={"distance": bad}))

        bad_oob = _sample_measurements()
        bad_oob["endpoints_xy"] = bad_oob["endpoints_xy"].copy()
        bad_oob["endpoints_xy"][0, 0, 1] = float(data["cameras"][0].height)
        with pytest.raises(ValueError, match="endpoints_xy"):
            SfmrReconstruction.from_data(
                ws, dict(data, measurements={"distance": bad_oob})
            )

        bad_empty = {
            "distances": np.zeros(0, dtype=np.float32),
            "units": np.zeros(0, dtype=np.uint8),
            "image_indexes": np.zeros((0, 2), dtype=np.uint32),
            "endpoints_xy": np.zeros((0, 2, 2), dtype=np.float32),
            "labels": [],
        }
        with pytest.raises(ValueError, match="at least one"):
            SfmrReconstruction.from_data(
                ws, dict(data, measurements={"distance": bad_empty})
            )

    def test_clone_with_changes_set_replace_clear_and_preserve(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        m = _sample_measurements()

        # Set with Fortran-ordered / strided (including negative-strided) arrays.
        m_fortran = {
            "distances": m["distances"][::-1].copy()[::-1],
            "units": np.repeat(m["units"], 2)[::2],
            "image_indexes": np.asfortranarray(m["image_indexes"]),
            "endpoints_xy": np.asfortranarray(m["endpoints_xy"]),
            "labels": list(m["labels"]),
        }
        assert not m_fortran["distances"].flags["C_CONTIGUOUS"]
        assert not m_fortran["units"].flags["C_CONTIGUOUS"]
        assert not m_fortran["image_indexes"].flags["C_CONTIGUOUS"]
        assert not m_fortran["endpoints_xy"].flags["C_CONTIGUOUS"]
        with_m = recon.clone_with_changes(measurements={"distance": m_fortran})
        assert with_m.distance_measurement_count == 3
        _assert_measurements_equal(with_m.measurements, m)

        # Unrelated change preserves measurements.
        scaled_unit = with_m.clone_with_changes(world_space_unit="cm")
        assert scaled_unit.distance_measurement_count == 3
        _assert_measurements_equal(scaled_unit.measurements, m)

        # Replace with a single measurement.
        one = {
            "distances": np.array([2.5], dtype=np.float32),
            "units": np.array([LENGTH_UNIT_NAMES.index("in")], dtype=np.uint8),
            "image_indexes": np.array([[0, 1]], dtype=np.uint32),
            "endpoints_xy": np.array([[[5.0, 5.0], [10.0, 10.0]]], dtype=np.float32),
            "labels": ["single"],
        }
        replaced = with_m.clone_with_changes(measurements={"distance": one})
        assert replaced.distance_measurement_count == 1
        assert replaced.metadata()["distance_measurement_count"] == 1
        _assert_measurements_equal(replaced.measurements, one)

        # Clear with None.
        cleared = replaced.clone_with_changes(measurements=None)
        assert cleared.distance_measurement_count == 0
        assert cleared.measurements is None
        assert "distance_measurement_count" not in cleared.metadata()

    def test_clone_with_changes_refuses_invalid_measurements_and_invalidation(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        m = _sample_measurements()
        with_m = recon.clone_with_changes(measurements={"distance": m})

        # Direct invalid measurements input.
        with pytest.raises(TypeError, match="measurements must be a dict"):
            recon.clone_with_changes(measurements=42)
        bad_key = dict(m)
        del bad_key["labels"]
        with pytest.raises(KeyError, match="labels"):
            recon.clone_with_changes(measurements={"distance": bad_key})
        bad_unit = dict(m, units=np.array([9, 0, 0], dtype=np.uint8))
        with pytest.raises(ValueError, match="units"):
            recon.clone_with_changes(measurements={"distance": bad_unit})

        # Replacing cameras with smaller dimensions invalidates existing
        # measurements whose endpoints exceed the new bounds (endpoint [1, 1]
        # sits at (200.5, 350.25)).
        tiny_cam = CameraIntrinsics(
            "SIMPLE_PINHOLE",
            100,
            100,
            {
                "focal_length": 80.0,
                "principal_point_x": 50.0,
                "principal_point_y": 50.0,
            },
        )
        with pytest.raises(ValueError, match="endpoints_xy"):
            with_m.clone_with_changes(cameras=[tiny_cam] * with_m.camera_count)

        # Dropping measurements in the same call lets the camera shrink succeed.
        shrunk_cleared = with_m.clone_with_changes(
            cameras=[tiny_cam] * with_m.camera_count, measurements=None
        )
        assert shrunk_cleared.distance_measurement_count == 0
        assert shrunk_cleared.measurements is None
        assert "distance_measurement_count" not in shrunk_cleared.metadata()

        # Switching an image's camera_index to a smaller camera invalidates
        # existing measurements referencing that image.
        new_cam_idx = with_m.camera_indexes.copy()
        new_cam_idx[2] = with_m.camera_count
        with pytest.raises(ValueError, match="endpoints_xy"):
            with_m.clone_with_changes(
                cameras=list(with_m.cameras) + [tiny_cam],
                camera_indexes=new_cam_idx,
            )

        # Shrinking image_names so an existing measurement's image_index (7)
        # is out of range is refused when measurements are kept.
        sub = with_m.subset_by_image_indices(np.arange(8, dtype=np.uint32))
        assert sub.distance_measurement_count == 3
        with pytest.raises(ValueError, match="image_indexes"):
            sub.clone_with_changes(
                image_names=sub.image_names[:4],
                camera_indexes=sub.camera_indexes[:4].copy(),
                quaternions_wxyz=sub.quaternions_wxyz[:4].copy(),
                translations=sub.translations[:4].copy(),
                image_file_hashes=sub.image_file_hashes[:4],
                positions=np.zeros((0, 4), dtype=np.float64),
                colors=np.zeros((0, 3), dtype=np.uint8),
                rms_reprojection_errors=np.zeros(0, dtype=np.float32),
                normals=None,
                patches=None,
                track_image_indexes=np.zeros(0, dtype=np.uint32),
                track_feature_indexes=np.zeros(0, dtype=np.uint32),
                track_point_indexes=np.zeros(0, dtype=np.uint32),
                keypoints_xy=np.zeros((0, 2), dtype=np.float32),
            )

    def test_apply_transform_preserves_measurements(self, seoul_bull_ground_truth_sfmr):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        m = _sample_measurements()
        with_m = recon.clone_with_changes(measurements={"distance": m})

        xform = Se3Transform(
            RotQuaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.5),
            np.array([1.0, -2.0, 3.0]),
            2.5,
        )
        moved = with_m.apply_transform(xform)
        assert moved.distance_measurement_count == 3
        _assert_measurements_equal(moved.measurements, m)

    def test_subset_by_image_indices_filters_and_remaps_measurements(
        self, seoul_bull_ground_truth_sfmr, tmp_path
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        m = _sample_measurements()
        with_m = recon.clone_with_changes(measurements={"distance": m})

        # Keep images [3, 2, 0] (reordered):
        # row 0: [0, 3] -> [2, 0]; row 1: [2, 2] -> [1, 1]; row 2: [5, 7] dropped.
        subset = with_m.subset_by_image_indices(
            np.array([3, 2, 0], dtype=np.uint32), drop_orphaned_points=True
        )
        assert subset.distance_measurement_count == 2
        assert subset.metadata()["distance_measurement_count"] == 2
        expected = {
            "distances": m["distances"][:2],
            "units": m["units"][:2],
            "image_indexes": np.array([[2, 0], [1, 1]], dtype=np.uint32),
            "endpoints_xy": m["endpoints_xy"][:2],
            "labels": m["labels"][:2],
        }
        _assert_measurements_equal(subset.measurements, expected)

        out = tmp_path / "subset.sfmr"
        subset.save(out, operation="subset")
        assert verify_sfmr(out) == (True, [])
        _assert_measurements_equal(read_sfmr(out)["measurements"], expected)

        # Subset with zero surviving measurements drops the section to None.
        no_m = with_m.subset_by_image_indices(
            np.array([1, 4], dtype=np.uint32), drop_orphaned_points=True
        )
        assert no_m.distance_measurement_count == 0
        assert no_m.measurements is None
        assert "distance_measurement_count" not in no_m.metadata()
        out_empty = tmp_path / "subset_none.sfmr"
        no_m.save(out_empty, operation="subset")
        assert verify_sfmr(out_empty) == (True, [])
        assert read_sfmr(out_empty)["measurements"] is None

    def test_filter_points_and_edited_reconstruction_preserve_measurements(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        m = _sample_measurements()
        with_m = recon.clone_with_changes(measurements={"distance": m})

        mask = np.ones(with_m.point_count, dtype=bool)
        mask[0] = False
        filtered = with_m.filter_points_by_mask(mask)
        assert filtered.distance_measurement_count == 3
        _assert_measurements_equal(filtered.measurements, m)

        edited = EditedReconstruction(with_m)
        edited.delete_point(0)
        materialized = edited.materialize()[0]
        assert materialized.distance_measurement_count == 3
        assert materialized.metadata()["distance_measurement_count"] == 3
        _assert_measurements_equal(materialized.measurements, m)
