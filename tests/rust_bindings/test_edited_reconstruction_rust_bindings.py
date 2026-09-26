# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EditedReconstruction binding: an immutable base plus point edits.

The overlay's promises, from Python: an edit never changes what an untouched
index resolves to, a read through the overlay is the read the materialisation
gives under the row map, the row map is a bijection on the survivors, and a
materialised value's hash is the file's after a save.
"""

import time

import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction


@pytest.fixture
def base(seoul_bull_sfmr_only):
    return SfmrReconstruction.load(seoul_bull_sfmr_only)


@pytest.fixture
def edited(base):
    return EditedReconstruction(base)


def moved(record, dx):
    """A copy of `record` with its position shifted, and nothing else changed."""
    out = dict(record)
    out["position"] = np.asarray(record["position"], dtype=np.float64) + dx
    return out


class TestOverlayReads:
    def test_a_fresh_overlay_is_its_base(self, base, edited):
        assert edited.point_count == base.point_count
        assert edited.base_point_count == base.point_count
        assert edited.image_count == base.image_count
        assert edited.deleted_count == 0
        assert edited.feature_source == base.feature_source
        assert edited.index_bound == base.point_count

    def test_columns_answer_from_the_base(self, base, edited):
        columns = edited.columns
        assert columns["feature_indexes"] == (base.feature_source == "sift_files")
        assert columns["patch_bitmaps"] == (base.patch_bitmaps is not None)

    def test_a_point_reads_as_the_base_says(self, base, edited):
        record = edited.point(0)
        assert record is not None
        np.testing.assert_allclose(record["position"], base.positions[0])
        counts = base.observation_counts
        assert len(record["image_indexes"]) == counts[0]
        expected = base.track_image_indexes[: counts[0]]
        np.testing.assert_array_equal(record["image_indexes"], expected)

    def test_a_record_carries_exactly_the_base_columns(self, edited):
        record = edited.point(0)
        columns = edited.columns
        assert ("feature_indexes" in record) == columns["feature_indexes"]
        assert ("keypoints_xy" in record) == columns["keypoints_xy"]
        assert ("patch_u_halfvec" in record) == columns["patch_frames"]
        assert ("patch_bitmap" in record) == columns["patch_bitmaps"]
        assert ("constraint" in record) == columns["point_constraints"]

    def test_arrays_handed_to_python_are_copies(self, edited):
        first = edited.point(1)
        first["position"][0] = 1234.5
        assert edited.point(1)["position"][0] != 1234.5


class TestPointEdits:
    def test_delete_shifts_no_index(self, edited):
        before = {i: edited.point(i) for i in (0, 2, 3)}
        edited.delete_point(1)
        assert edited.point(1) is None
        assert edited.is_deleted(1)
        assert edited.point_count == edited.base_point_count - 1
        for i, record in before.items():
            np.testing.assert_array_equal(
                edited.point(i)["position"], record["position"]
            )

    def test_replace_takes_a_new_index_and_frees_the_old(self, edited):
        record = moved(edited.point(2), 10.0)
        new_index = edited.replace_point(2, record)
        assert new_index == edited.base_point_count
        assert edited.point(2) is None
        np.testing.assert_allclose(
            edited.point(new_index)["position"], record["position"]
        )
        # A replacement is not a new point: the count is unchanged.
        assert edited.point_count == edited.base_point_count

    def test_add_appends_after_the_base(self, edited):
        record = moved(edited.point(0), 5.0)
        new_index = edited.add_point(record)
        assert new_index == edited.base_point_count
        assert edited.point_count == edited.base_point_count + 1
        np.testing.assert_allclose(
            edited.point(new_index)["position"], record["position"]
        )

    def test_a_bad_image_index_is_refused(self, edited):
        record = dict(edited.point(0))
        record["image_indexes"] = np.array([9999], dtype=np.uint32)
        if "feature_indexes" in record:
            record["feature_indexes"] = np.array([0], dtype=np.uint32)
        if "keypoints_xy" in record:
            record["keypoints_xy"] = np.zeros((1, 2), dtype=np.float32)
        if "observation_confidence" in record:
            record["observation_confidence"] = np.zeros(1, dtype=np.uint8)
        with pytest.raises(ValueError, match="past the"):
            edited.add_point(record)
        assert edited.point_count == edited.base_point_count

    def test_a_missing_column_is_refused(self, edited):
        record = dict(edited.point(0))
        if record.pop("feature_indexes", None) is None:
            pytest.skip("this base carries no feature indexes")
        with pytest.raises(ValueError, match="feature_indexes"):
            edited.add_point(record)

    def test_a_dead_index_is_refused(self, edited):
        edited.delete_point(0)
        with pytest.raises(ValueError, match="no live point"):
            edited.delete_point(0)


class TestMaterialisation:
    def test_reads_agree_under_the_row_map(self, edited):
        edited.delete_point(1)
        edited.replace_point(3, moved(edited.point(3), 2.0))
        edited.add_point(moved(edited.point(0), 7.0))

        recon, forward, inverse = edited.materialize()
        assert recon.point_count == edited.point_count
        assert forward.shape == (edited.index_bound,)
        assert inverse.shape == (recon.point_count,)

        after = EditedReconstruction(recon)
        live = edited.live_indexes()
        landed = forward[live]
        assert (landed >= 0).all()
        assert len(set(landed.tolist())) == len(live), "the map is injective"
        assert sorted(landed.tolist()) == list(range(recon.point_count))
        for i, new in zip(live.tolist(), landed.tolist()):
            assert inverse[new] == i
            a, b = edited.point(i), after.point(int(new))
            np.testing.assert_array_equal(a["position"], b["position"])
            np.testing.assert_array_equal(a["image_indexes"], b["image_indexes"])
        dead = [i for i in range(edited.index_bound) if edited.point(i) is None]
        assert all(forward[i] == -1 for i in dead)

    def test_a_replacement_keeps_its_place(self, edited):
        record = moved(edited.point(2), 3.0)
        new_index = edited.replace_point(2, record)
        recon, forward, _ = edited.materialize()
        assert forward[new_index] == 2, "back at the base index it replaced"
        assert forward[2] == -1
        np.testing.assert_allclose(recon.positions[2], record["position"])

    def test_materialisation_is_deterministic(self, edited):
        edited.delete_point(0)
        edited.add_point(moved(edited.point(1), 4.0))
        first, forward_a, inverse_a = edited.materialize()
        second, forward_b, inverse_b = edited.materialize()
        np.testing.assert_array_equal(forward_a, forward_b)
        np.testing.assert_array_equal(inverse_a, inverse_b)
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(
            first.track_image_indexes, second.track_image_indexes
        )

    def test_an_empty_overlay_materialises_to_the_identity(self, base, edited):
        recon, forward, inverse = edited.materialize()
        np.testing.assert_array_equal(forward, np.arange(base.point_count))
        np.testing.assert_array_equal(inverse, np.arange(base.point_count))
        np.testing.assert_array_equal(recon.positions, base.positions)


class TestHashes:
    def test_a_materialised_value_carries_no_files_hash(self, edited):
        edited.delete_point(0)
        recon, _, _ = edited.materialize()
        assert recon.content_xxh128 == ""

    def test_two_saves_agree_on_the_hash_and_differ_on_the_clock(
        self, edited, tmp_path
    ):
        recon, _, _ = edited.materialize()
        first, second = tmp_path / "first.sfmr", tmp_path / "second.sfmr"
        recon.save(first)
        time.sleep(0.02)
        recon.save(second)
        a = SfmrReconstruction.load(first)
        b = SfmrReconstruction.load(second)
        assert a.content_xxh128 == b.content_xxh128
        assert a.metadata()["timestamp"] != b.metadata()["timestamp"]

    def test_the_base_hash_is_the_files_hash_after_a_save(self, edited, tmp_path):
        edited.delete_point(0)
        recon, _, _ = edited.materialize()
        after = EditedReconstruction(recon)
        computed = after.base_content_hash()
        assert len(computed) == 32

        path = tmp_path / "materialized.sfmr"
        recon.save(path)
        assert SfmrReconstruction.load(path).content_xxh128 == computed

    def test_a_point_edit_hash_is_a_function_of_its_records(self, edited):
        one = moved(edited.point(0), 1.0)
        two = moved(edited.point(0), 2.0)
        assert edited.point_edit_hash([one]) == edited.point_edit_hash([one])
        assert edited.point_edit_hash([one]) != edited.point_edit_hash([two])
        assert len(edited.point_edit_hash([one])) == 32


class TestResectImageInPlace:
    """The bulk edit: one image re-posed against structure held out from it."""

    @pytest.fixture
    def embedded(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        return EditedReconstruction(recon.to_embedded_patches())

    def test_an_image_past_the_table_is_refused(self, embedded):
        with pytest.raises(ValueError, match="out of range"):
            embedded.resect_image_in_place(embedded.image_count)

    def test_the_resected_image_moves_and_the_others_stand(self, embedded):
        before = embedded.materialize()[0]
        # Which images this capture corroborates is a property of its tracks, so
        # the first one the estimate accepts is the one the assertions run on.
        for image in range(before.image_count):
            try:
                after, report = embedded.resect_image_in_place(image)
            except ValueError:
                continue
            break
        else:
            pytest.skip("no image of this reconstruction resects")

        assert report["image_index"] == image
        assert report["refusal"] is None
        assert report["accepted"] is True
        assert report["correspondences"] >= report["inliers"] > 0

        # A bulk edit: a whole new base with no overlay on it.
        assert after.deleted_count == 0
        assert after.base_content_hash() != embedded.base_content_hash()

        value = after.materialize()[0]
        assert value.image_count == before.image_count
        assert value.image_names == before.image_names
        others = [i for i in range(before.image_count) if i != image]
        np.testing.assert_array_equal(
            value.quaternions_wxyz[others], before.quaternions_wxyz[others]
        )
        np.testing.assert_array_equal(
            value.translations[others], before.translations[others]
        )
        # This object is untouched.
        np.testing.assert_array_equal(
            embedded.materialize()[0].translations, before.translations
        )


class TestBundleAdjust:
    """The bulk edit that moves every pose and every point at once."""

    @pytest.fixture
    def embedded(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        return EditedReconstruction(recon.to_embedded_patches())

    def test_the_poses_move_and_the_residuals_do_not_get_worse(self, embedded):
        before = embedded.materialize()[0]

        after, report = embedded.bundle_adjust()

        assert report["images"] == before.image_count
        assert report["observations"] == before.observation_count
        assert report["points"] <= before.point_count
        assert len(report["cameras"]) == len(before.cameras)
        for camera in report["cameras"]:
            assert camera["focal_released"] is False
            assert camera["focal_before"] == camera["focal_after"]
        assert sum(c["images"] for c in report["cameras"]) == report["images"]
        assert (
            report["median_residual_after"] <= report["median_residual_before"] + 1e-9
        )

        value = after.materialize()[0]
        assert value.image_count == before.image_count
        assert value.point_count == before.point_count - report["points_deleted"]
        assert not np.array_equal(value.translations, before.translations), (
            "the adjustment moved nothing"
        )
        # This object is untouched, and the value that came back is a new base
        # with no overlay on it.
        np.testing.assert_array_equal(
            embedded.materialize()[0].translations, before.translations
        )
        assert after.deleted_count == 0

    def test_a_value_with_no_inline_keypoints_is_refused(self, base):
        # The stored reconstruction is sift_files, whose keypoints live in the
        # .sift companions rather than in the file.
        if base.keypoints_xy is not None:
            pytest.skip("this reconstruction carries inline keypoints")
        with pytest.raises(ValueError, match="inline keypoints"):
            EditedReconstruction(base).bundle_adjust()


class TestSwitchCameraModel:
    """The bulk edit that replaces cameras with ones of another model."""

    @pytest.fixture
    def embedded(self, seoul_bull_ground_truth_sfmr):
        return EditedReconstruction(
            SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        )

    def test_the_camera_changes_and_nothing_else_moves(self, embedded):
        before = embedded.materialize()[0]

        after, report = embedded.switch_camera_model("SFMTOOL_PINHOLE", coeff_count=4)

        value = after.materialize()[0]
        assert value.cameras[0].model == "SFMTOOL_PINHOLE"
        np.testing.assert_array_equal(value.positions, before.positions)
        np.testing.assert_array_equal(value.translations, before.translations)
        np.testing.assert_array_equal(value.keypoints_xy, before.keypoints_xy)
        # This object is untouched, and the value that came back is a new base.
        assert embedded.materialize()[0].cameras[0].model == "SIMPLE_RADIAL"
        assert after.deleted_count == 0

        (entry,) = report["cameras"]
        assert entry["camera"] == 0
        assert entry["source"].model == "SIMPLE_RADIAL"
        assert entry["target"] == value.cameras[0]
        fit = entry["fit"]
        assert fit["camera_model"] == "SFMTOOL_PINHOLE"
        # A perspective source has no trusted bound: the fit reaches the
        # observations' extent.
        assert fit["theta_fit_source"] == "observations"
        assert fit["max_px"] < 0.1
        obs = entry["observations"]
        assert obs["observations"] + obs["unmeasured"] == before.observation_count
        assert obs["trusted_deg"] is None
        assert obs["after"]["median_px"] == pytest.approx(
            obs["before"]["median_px"], abs=0.05
        )

    def test_the_distortion_is_released_by_the_adjustment(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_PINHOLE", coeff_count=4)
        with pytest.raises(ValueError, match="camera 0 .* together with its focal"):
            switched.bundle_adjust(opt_distortion=True)
        pinhole, _ = embedded.switch_camera_model("SIMPLE_PINHOLE")
        with pytest.raises(ValueError, match="camera 0, a SIMPLE_PINHOLE, has no lens"):
            pinhole.bundle_adjust(opt_f=True, opt_distortion=True)

        adjusted, report = switched.bundle_adjust(opt_f=True, opt_distortion=True)
        (camera,) = report["cameras"]
        assert camera["focal_released"] and camera["distortion_released"]
        assert (
            report["median_residual_after"] <= report["median_residual_before"] + 1e-9
        )
        assert adjusted.materialize()[0].cameras[0].model == "SFMTOOL_PINHOLE"

    def test_k1_is_released_on_a_simple_radial_fisheye(self, embedded):
        fisheye, _ = embedded.switch_camera_model("SIMPLE_RADIAL_FISHEYE")
        adjusted, report = fisheye.bundle_adjust(opt_f=True, opt_distortion=True)
        (camera,) = report["cameras"]
        assert camera["distortion_released"]
        assert (
            report["median_residual_after"] <= report["median_residual_before"] + 1e-9
        )
        assert adjusted.materialize()[0].cameras[0].model == "SIMPLE_RADIAL_FISHEYE"

    def test_a_spline_switched_to_its_own_model_is_refitted(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
        before = switched.materialize()[0].cameras[0].to_dict()["parameters"]
        with pytest.raises(ValueError, match="2 to 32 spline coefficients, not 40"):
            switched.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=40)

        refitted, report = switched.switch_camera_model(
            "SFMTOOL_FISHEYE", coeff_count=8
        )
        (entry,) = report["cameras"]
        fit = entry["fit"]
        # Over the whole spline domain, not the observations' extent.
        assert fit["theta_fit_source"] == "spline_domain"
        assert fit["theta_fit_deg"] == pytest.approx(
            np.degrees(before["bspline_theta_max"])
        )
        assert 0.0 <= fit["rms_px"] <= fit["max_px"] < 1.0
        params = refitted.materialize()[0].cameras[0].to_dict()["parameters"]
        assert params["bspline_coeff_count"] == 8
        # No domain given: the domain end is kept exactly.
        assert params["bspline_theta_max"] == before["bspline_theta_max"]
        # Poses are not touched; the adjustment is a separate step.
        np.testing.assert_array_equal(
            refitted.materialize()[0].translations,
            switched.materialize()[0].translations,
        )

    def test_a_dipped_spline_is_refitted_under_the_monotone_constraint(self, embedded):
        from sfmtool._sfmtool.geometry import CameraIntrinsics

        # A spline whose slope dips close to zero near 113°: monotone, but a
        # twelve-coefficient least-squares refit rings through the dip and
        # crosses below zero. The refit is constrained instead of refused.
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=8)
        value = switched.materialize()[0]
        camera = value.cameras[0].to_dict()
        dip = [0.0003, 0.0091, 0.0403, 0.1131, 0.0994, -0.4755, 0.0809, 0.0702]
        params = dict(camera["parameters"])
        params["bspline_theta_max"] = 2.6194
        for i, c in enumerate(dip):
            params[f"bspline_c{i}"] = c
        dipped = CameraIntrinsics(
            "SFMTOOL_FISHEYE", camera["width"], camera["height"], params
        )
        edited = EditedReconstruction(value.clone_with_changes(cameras=[dipped]))

        _, report = edited.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=12)
        fit = report["cameras"][0]["fit"]
        assert fit["theta_fit_source"] == "spline_domain"
        constraint = fit["monotone_constraint"]
        assert constraint["active"]
        assert constraint["active_angles"] >= 1
        low, high = constraint["range_deg"]
        assert 100.0 < low <= high < 130.0
        assert fit["max_px"] < 5.0

        _, unconstrained = switched.switch_camera_model(
            "SFMTOOL_FISHEYE", coeff_count=12
        )
        kept = unconstrained["cameras"][0]["fit"]["monotone_constraint"]
        assert kept == {"active": False, "active_angles": 0, "range_deg": None}

    def test_the_spline_domain_is_moved_by_a_refit(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
        with pytest.raises(ValueError, match="camera 0: the spline domain"):
            switched.switch_camera_model(
                "SFMTOOL_FISHEYE", coeff_count=6, spline_domain_deg=200.0
            )

        refitted, report = switched.switch_camera_model(
            "SFMTOOL_FISHEYE", coeff_count=6, spline_domain_deg=40.0
        )
        fit = report["cameras"][0]["fit"]
        assert fit["spline_domain_deg"] == pytest.approx(40.0)
        assert fit["theta_fit_deg"] == pytest.approx(40.0)
        params = refitted.materialize()[0].cameras[0].to_dict()["parameters"]
        assert params["bspline_theta_max"] == pytest.approx(np.radians(40.0))
        assert params["bspline_coeff_count"] == 6

    def test_the_adjustment_reports_the_outermost_observation(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
        _, report = switched.bundle_adjust(opt_f=True, opt_distortion=True)
        observed = report["cameras"][0]["outermost_observed"]
        assert observed["radius_px"] > 0 and observed["theta_deg"] > 0

    @staticmethod
    def _two_cameras(edited, second):
        """``edited``'s value with its odd images taken through ``second``, a
        second camera appended to the table."""
        value = edited.materialize()[0]
        indexes = np.arange(value.image_count, dtype=np.uint32) % 2
        return EditedReconstruction(
            value.clone_with_changes(
                cameras=[value.cameras[0], second], camera_indexes=indexes
            )
        )

    def test_releases_are_chosen_per_camera(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
        spline = switched.materialize()[0].cameras[0]
        rig = self._two_cameras(switched, spline)

        adjusted, report = rig.bundle_adjust(
            opt_f=True,
            opt_distortion=True,
            releases=[{"focal": True, "distortion": True}, {}],
        )
        released = [
            (c["camera"], c["focal_released"], c["distortion_released"])
            for c in report["cameras"]
        ]
        assert released == [(0, True, True), (1, False, False)]
        # The held camera comes back exactly as it went in.
        assert adjusted.materialize()[0].cameras[1] == spline

        # With no list, the keyword defaults apply to every camera.
        _, both = rig.bundle_adjust(opt_f=True, opt_distortion=True)
        assert [c["distortion_released"] for c in both["cameras"]] == [True, True]

        # Every camera held still refines the poses.
        _, held = rig.bundle_adjust(releases=[{}, {"focal": False}])
        assert not any(c["focal_released"] for c in held["cameras"])

    def test_a_rig_holds_the_camera_whose_model_cannot_release(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
        opencv, _ = embedded.switch_camera_model("OPENCV_FISHEYE")
        rig = self._two_cameras(switched, opencv.materialize()[0].cameras[0])

        with pytest.raises(ValueError, match="camera 1, a OPENCV_FISHEYE"):
            rig.bundle_adjust(opt_f=True)
        _, report = rig.bundle_adjust(
            releases=[{"focal": True, "distortion": True}, {}]
        )
        assert [c["focal_released"] for c in report["cameras"]] == [True, False]

    def test_a_bad_release_list_is_refused(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
        with pytest.raises(ValueError, match="2 entries and the reconstruction has 1"):
            switched.bundle_adjust(releases=[{}, {}])
        with pytest.raises(ValueError, match="'focus'"):
            switched.bundle_adjust(releases=[{"focus": True}])
        with pytest.raises(ValueError, match="camera 0 .* together with its focal"):
            switched.bundle_adjust(releases=[{"distortion": True}])

    def test_the_release_capabilities_are_read_off_the_camera(self, embedded):
        switched, _ = embedded.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
        spline = switched.materialize()[0].cameras[0]
        assert spline.focal_is_releasable and spline.distortion_is_releasable
        simple_radial = embedded.materialize()[0].cameras[0]
        assert not simple_radial.focal_is_releasable
        assert not simple_radial.distortion_is_releasable

    def test_the_outermost_keypoints_are_observed_and_detected(self, embedded):
        recon = embedded.materialize()[0]
        (camera,) = recon.outermost_keypoints()
        assert camera["camera"] == 0 and camera["images"] == recon.image_count
        observed = camera["observed"]
        assert set(observed) == {"radius_px", "theta_deg", "image", "xy"}
        # The fixture sits beside no .sift file.
        assert camera["detected"] is None and camera["detected_images"] == 0
        assert recon.outermost_keypoints(read_sift_files=False) == [camera]
        assert recon.outermost_keypoints(cameras=[5]) == []

    def test_a_refusal_names_the_camera(self, embedded):
        with pytest.raises(ValueError, match="camera 0: .*90°"):
            embedded.switch_camera_model("SFMTOOL_PINHOLE", theta_fit_deg=95.0)
        with pytest.raises(ValueError, match="does not exist"):
            embedded.switch_camera_model("RADIAL", cameras=[4])


class TestCameraIntrinsicsRefit:
    """``CameraIntrinsics.refit``: the lens-only fit."""

    @staticmethod
    def kerry_cam0():
        from sfmtool._sfmtool.geometry import CameraIntrinsics

        return CameraIntrinsics(
            "OPENCV_FISHEYE",
            480,
            480,
            {
                "focal_length_x": 129.718,
                "focal_length_y": 129.430,
                "principal_point_x": 240.0,
                "principal_point_y": 240.0,
                "radial_distortion_k1": 0.02865,
                "radial_distortion_k2": -0.00228,
                "radial_distortion_k3": 0.00902,
                "radial_distortion_k4": -0.00355,
            },
        )

    def test_a_fisheye_moves_to_the_spline(self):
        camera, report = self.kerry_cam0().refit("SFMTOOL_FISHEYE", coeff_count=8)
        assert camera.model == "SFMTOOL_FISHEYE"
        assert camera.principal_point == (240.0, 240.0)
        assert camera.focal_lengths[0] == pytest.approx(129.56, abs=0.1)
        assert report["theta_fit_source"] == "trusted_bound"
        assert report["theta_fit_deg"] < report["extent"]["source_fold_deg"]
        assert report["radial_rms_px"] < 0.05
        assert report["dropped"] == ["fx/fy aspect 0.9978 dropped (single focal)"]
        assert report["spline_domain_deg"] == pytest.approx(150.0, abs=1.0)
        assert report["monotone_constraint"] == {
            "active": False,
            "active_angles": 0,
            "range_deg": None,
        }

    def test_refusals_are_value_errors(self):
        with pytest.raises(ValueError, match="trusted bound"):
            self.kerry_cam0().refit("SFMTOOL_FISHEYE", theta_fit_deg=100.0)
        with pytest.raises(ValueError, match="not a model a camera can be refitted"):
            self.kerry_cam0().refit("EQUIRECTANGULAR")


class TestMoveCamera:
    """The bulk edit: one image put at a pose, and its tracks settled around it."""

    @pytest.fixture
    def embedded(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        return EditedReconstruction(recon.to_embedded_patches())

    def test_an_image_past_the_table_is_refused(self, embedded):
        with pytest.raises(ValueError, match="past the"):
            embedded.move_camera(
                embedded.image_count, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            )

    def test_a_non_finite_pose_is_refused(self, embedded):
        with pytest.raises(ValueError, match="finite"):
            embedded.move_camera(0, [1.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0])

    def test_putting_a_camera_back_where_it_stands_changes_nothing_but_the_value(
        self, embedded
    ):
        before = embedded.materialize()[0]
        image = 0
        # The stored pose, read back out and handed straight in again: the
        # rotation is camera-to-world and the translation is the camera centre.
        stored = before.quaternions_wxyz[image]
        world_from_camera = [stored[0], -stored[1], -stored[2], -stored[3]]
        rotation = _rotation_matrix(stored)
        centre = -rotation.T @ before.translations[image]

        after, report = embedded.move_camera(image, world_from_camera, centre)

        assert report["image"] == image
        assert report["rotation_deg"] == pytest.approx(0.0, abs=1e-9)
        assert report["translation"] == pytest.approx(0.0, abs=1e-9)
        assert report["observed"] > 0
        assert (
            report["retriangulated"] + report["kept"] + report["rotated_bearings"]
            == report["observed"]
        )
        # A bulk edit: a whole new base with no overlay on it.
        assert after.deleted_count == 0
        value = after.materialize()[0]
        assert value.image_count == before.image_count
        assert value.point_count == before.point_count
        np.testing.assert_allclose(value.quaternions_wxyz[image], stored, atol=1e-12)
        # This object is untouched.
        np.testing.assert_array_equal(
            embedded.materialize()[0].translations, before.translations
        )

    def test_a_moved_camera_moves_its_pose_and_leaves_the_others_alone(self, embedded):
        before = embedded.materialize()[0]
        image = 0
        stored = before.quaternions_wxyz[image]
        world_from_camera = [stored[0], -stored[1], -stored[2], -stored[3]]
        rotation = _rotation_matrix(stored)
        centre = -rotation.T @ before.translations[image]
        # A tenth of the capture's own extent, along one axis.
        extent = float(
            np.linalg.norm(before.positions.max(axis=0) - before.positions.min(axis=0))
        )
        shifted = centre + np.array([0.1 * extent, 0.0, 0.0])

        after, report = embedded.move_camera(image, world_from_camera, shifted)

        assert report["translation"] == pytest.approx(0.1 * extent, rel=1e-9)
        assert report["translation_scene"] > 0.0
        assert report["residual_before_px"] is not None
        assert len(report["residual_after_px"]) == 2

        value = after.materialize()[0]
        others = [i for i in range(before.image_count) if i != image]
        np.testing.assert_array_equal(
            value.quaternions_wxyz[others], before.quaternions_wxyz[others]
        )
        np.testing.assert_array_equal(
            value.translations[others], before.translations[others]
        )
        assert not np.array_equal(
            value.translations[image], before.translations[image]
        ), "the move moved nothing"
        if report["retriangulated"]:
            assert not np.array_equal(value.positions, before.positions), (
                "points were re-solved and none of them moved"
            )


class TestPruneCoveredObservations:
    """The bulk edit: a coarse observation handed over to the finer feature."""

    @pytest.fixture
    def embedded(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        return EditedReconstruction(recon.to_embedded_patches())

    def test_a_value_with_no_patch_frames_is_refused(self, base):
        # The stored reconstruction is sift_files, whose points carry no patch
        # frame for a footprint to be read off.
        plain = EditedReconstruction(base)
        if plain.columns["patch_frames"]:
            pytest.skip("this reconstruction carries patch frames")
        with pytest.raises(ValueError, match="patch frame"):
            plain.prune_covered_observations()

    def test_an_unusable_footprint_fraction_is_refused(self, embedded):
        with pytest.raises(ValueError, match="footprint fraction"):
            embedded.prune_covered_observations(footprint_fraction=0.0)

    def test_the_report_accounts_for_every_row(self, embedded):
        before = embedded.materialize()[0]
        _after, report = embedded.prune_covered_observations()

        census = report["census"]
        assert census["rows"] == before.observation_count
        assert report["observations_before"] == before.observation_count
        assert report["points_before"] == before.point_count
        assert (
            report["observations_before"] - report["observations_after"]
            == census["rows_removed"]
        )
        assert (
            report["points_before"] - report["points_after"]
            == census["owners_dropped_all_covered"] + census["owners_dropped_by_sweep"]
        )
        assert census["pairs_finer"] <= census["pairs_contained"]
        assert census["rows_flagged"] <= census["rows_removed"]
        # The bands account for every row that projected to a radius, and the
        # retirements inside them are the rule's own.
        rows = sum(band["rows"] for band in report["bands"])
        assert rows == census["rows"] - report["degenerate_rows"]
        assert (
            sum(band["rows_retired"] for band in report["bands"])
            == census["rows_flagged"]
        )
        assert [band["band"] for band in report["bands"]] == sorted(
            band["band"] for band in report["bands"]
        )

    def test_a_prune_that_retires_nothing_hands_the_value_back(self, embedded):
        # No feature is a billion times finer than another, so the scale test
        # passes no pair wherever the features sit. A small
        # footprint alone does not promise that: two detections of one corner
        # at different scales sit at almost the same pixel.
        after, report = embedded.prune_covered_observations(ratio=1e9)
        assert report["changed"] is False
        assert report["census"]["rows_removed"] == 0
        assert report["points_after"] == report["points_before"]
        assert after.point_count == embedded.point_count

    def test_a_prune_that_bites_shortens_tracks_and_moves_no_point(self, embedded):
        before = embedded.materialize()[0]
        # Wide enough that the capture's own features cover one another.
        after, report = embedded.prune_covered_observations(footprint_fraction=4.0)
        assert report["changed"] is True
        assert report["census"]["rows_removed"] > 0

        value = after.materialize()[0]
        assert value.observation_count == report["observations_after"]
        assert value.point_count == report["points_after"]
        assert value.image_count == before.image_count
        assert value.image_names == before.image_names
        # A bulk edit: a whole new base with no overlay on it.
        assert after.deleted_count == 0
        # Every surviving point kept its position, and the map says where it
        # went; nothing was re-solved.
        point_map = report["map"]
        survivors = [
            (old, point_map.forward(old))
            for old in range(before.point_count)
            if point_map.forward(old) is not None
        ]
        assert len(survivors) == value.point_count
        old_rows = np.array([old for old, _ in survivors])
        new_rows = np.array([new for _, new in survivors])
        np.testing.assert_array_equal(
            value.positions[new_rows], before.positions[old_rows]
        )
        # The survivors kept their order, which is the whole of what a prune
        # does to the indexing.
        np.testing.assert_array_equal(new_rows, np.arange(len(new_rows)))
        # This object is untouched.
        assert embedded.materialize()[0].observation_count == before.observation_count

    def test_held_points_keep_every_observation(self, embedded):
        """A point the value holds is never retired, and still covers."""
        before = embedded.materialize()[0]
        free, _report = embedded.prune_covered_observations(footprint_fraction=4.0)
        assert _report["changed"] is True
        assert _report["protected_rows"] == 0, "nothing is pinned yet"

        # Hold the ten longest tracks, which are the ones with the most rows to
        # lose, and prune the same way again.
        longest = np.argsort(before.observation_counts)[-10:]
        constraints = np.zeros(before.point_count, dtype=np.uint8)
        constraints[longest] = 2  # held
        pinned = EditedReconstruction(
            before.clone_with_changes(
                point_constraints=constraints,
                constraint_distances=np.full(before.point_count, np.nan),
                constraint_reference_images=np.full(
                    before.point_count, 0xFFFFFFFF, dtype=np.uint32
                ),
            )
        )
        _value, report = pinned.prune_covered_observations(footprint_fraction=4.0)

        expected = int(before.observation_counts[longest].sum())
        assert report["protected_rows"] == expected
        assert report["census"]["rows_spared"] == report["protected_rows_spared"]
        assert report["protected_rows_spared"] > 0, (
            "pinning the longest tracks spared nothing, so the case is untested"
        )
        # Sparing can only keep rows, never retire more.
        assert report["census"]["rows_flagged"] < _report["census"]["rows_flagged"]
        assert free.point_count <= _value.point_count


def _rotation_matrix(wxyz):
    """The rotation matrix of a WXYZ quaternion, spelled out rather than imported."""
    w, x, y, z = (float(c) for c in wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
