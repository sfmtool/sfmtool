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


class TestAddObservation:
    """``add_observation``: a clicked pixel becomes an observation of a track.

    Built on the cheap ``to_embedded_patches`` baseline, which is a real
    ``embedded_patches`` value over the 17-image workspace, so the fit runs
    against the workspace's own photographs. The click is placed at the point's
    own projection into the target image, read off a single-view localization,
    which is what a user pointing at the point would produce.
    """

    TARGET = 0

    @pytest.fixture
    def embedded(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        return EditedReconstruction(recon.to_embedded_patches())

    @pytest.fixture
    def images(self, embedded):
        from sfmtool._workspace_image import read_workspace_image

        base = embedded.materialize()[0]
        return [
            read_workspace_image(base.workspace_dir, name) for name in base.image_names
        ]

    @pytest.fixture
    def gap(self, embedded, images):
        """A point the target image does not observe, and where it projects.

        A one-view localization runs no congealing round, so the keypoint it
        reports is the point's projection itself. Which surfaces register in
        image 0 depends on the platform's solve of the fixture, so the first
        candidate whose patch the localizer accepts there is the one returned:
        the tests are about the call's shape, the refusal texts and the
        materialisation, not about any particular surface.
        """
        base = embedded.materialize()[0]
        # The embedded base already carries the frames, so the cloud is read
        # back out of its own columns rather than re-derived from the `.sift`.
        cloud = base.patches
        localized = cloud.localize_keypoints(
            base,
            images,
            view_sets={p: [self.TARGET] for p in range(base.point_count)},
        )
        for entry in localized:
            index = int(entry["point_index"])
            if len(entry["views"]) != 1:
                continue
            record = embedded.point(index)
            seen = set(int(i) for i in record["image_indexes"])
            if self.TARGET in seen or len(seen) < 2:
                continue
            x, y = (float(v) for v in entry["keypoints"][0])
            try:
                embedded.add_observation(
                    index, self.TARGET, [x, y], images, min_zncc=-2.0
                )
            except ValueError:
                continue
            return index, self.TARGET, [x, y]
        pytest.skip("no point of this reconstruction registers in image 0 unseen")

    def test_a_sift_files_base_is_refused(self, edited, images):
        with pytest.raises(ValueError, match="embedded_patches"):
            edited.add_observation(0, 0, [1.0, 1.0], images)

    def test_an_image_already_in_the_track_is_refused(self, embedded, images, gap):
        point, _, pixel = gap
        seen = int(embedded.point(point)["image_indexes"][0])
        with pytest.raises(ValueError, match="already observes"):
            embedded.add_observation(point, seen, pixel, images)

    def test_a_pixel_off_the_sensor_is_refused(self, embedded, images, gap):
        point, image, _ = gap
        with pytest.raises(ValueError, match="outside"):
            embedded.add_observation(point, image, [-5.0, 10.0], images)

    def test_an_unreachable_bar_is_refused(self, embedded, images, gap):
        point, image, pixel = gap
        with pytest.raises(ValueError, match="below the"):
            embedded.add_observation(point, image, pixel, images, min_zncc=1.5)
        assert len(embedded.point(point)["image_indexes"]) >= 2

    def test_an_accepted_fit_adds_one_observation_and_keeps_the_base(
        self, embedded, images, gap
    ):
        point, image, pixel = gap
        before = len(embedded.point(point)["image_indexes"])
        # The bar is off: this test is about the call's shape and the value it
        # returns, not about whether this particular surface registers well.
        next_value, report = embedded.add_observation(
            point, image, pixel, images, min_zncc=-2.0
        )

        assert report["replaced"] == point
        assert report["image"] == image
        assert report["observation_count"] == before + 1
        assert len(report["keypoint"]) == 2
        assert report["shift_px"] >= 0.0

        record = next_value.point(report["point"])
        assert list(record["image_indexes"]).count(image) == 1
        assert list(record["image_indexes"]) == sorted(record["image_indexes"])

        # This object is untouched, and the two agree on the base.
        assert len(embedded.point(point)["image_indexes"]) == before
        assert next_value.base_content_hash() == embedded.base_content_hash()

    def test_the_modification_materialises_back_into_its_place(
        self, embedded, images, gap
    ):
        point, image, pixel = gap
        points_before = embedded.point_count
        next_value, report = embedded.add_observation(
            point, image, pixel, images, min_zncc=-2.0
        )
        recon, forward, _ = next_value.materialize()
        assert next_value.point_count == points_before
        assert recon.point_count == points_before
        assert forward[report["point"]] == point


class TestCreatePoint:
    """``create_point``: a clicked pixel becomes a point at infinity.

    Built on the same ``to_embedded_patches`` baseline as
    :class:`TestAddObservation`, so the colour and the patch bitmap are read out
    of the workspace's own photographs.
    """

    IMAGE = 0

    @pytest.fixture
    def embedded(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        return EditedReconstruction(recon.to_embedded_patches())

    @pytest.fixture
    def images(self, embedded):
        from sfmtool._workspace_image import read_workspace_image

        base = embedded.materialize()[0]
        return [
            read_workspace_image(base.workspace_dir, name) for name in base.image_names
        ]

    def test_a_sift_files_base_is_refused(self, edited, images):
        with pytest.raises(ValueError, match="embedded_patches"):
            edited.create_point(0, [1.0, 1.0], 8.0, images)

    def test_a_pixel_off_the_sensor_is_refused(self, embedded, images):
        with pytest.raises(ValueError, match="outside"):
            embedded.create_point(self.IMAGE, [-5.0, 10.0], 8.0, images)

    def test_a_radius_that_is_not_a_size_is_refused(self, embedded, images):
        with pytest.raises(ValueError, match="positive size"):
            embedded.create_point(self.IMAGE, [40.0, 40.0], 0.0, images)

    def test_a_created_point_is_a_bearing_with_one_observation(self, embedded, images):
        before = embedded.point_count
        next_value, report = embedded.create_point(
            self.IMAGE, [60.0, 80.0], 8.0, images
        )

        assert next_value.point_count == before + 1
        assert report["image"] == self.IMAGE
        assert report["half_extent"] > 0.0
        assert abs(float(np.linalg.norm(report["direction"])) - 1.0) < 1e-9

        record = next_value.point(report["point"])
        assert record["w"] == 0.0
        assert list(record["image_indexes"]) == [self.IMAGE]
        assert list(record["keypoints_xy"][0]) == [60.0, 80.0]

        # This object is untouched, and the two agree on the base.
        assert embedded.point_count == before
        assert next_value.base_content_hash() == embedded.base_content_hash()

    def test_a_second_observation_makes_the_bearing_finite(self, embedded, images):
        # A real correspondence, so the two-pass fit has something to register:
        # an existing point of the reconstruction, created afresh at its own
        # keypoint in one image and then sighted at its own keypoint in another.
        # Which surfaces register from a fresh 8 px patch depends on the
        # platform's solve of the fixture, so the test is about the first
        # candidate that does: the crossing from infinity, not any one point.
        outcome = None
        for source in range(embedded.point_count):
            record = embedded.point(source)
            if len(set(int(k) for k in record["image_indexes"])) < 2:
                continue
            first, second = (int(k) for k in record["image_indexes"][:2])
            here, there = (list(map(float, k)) for k in record["keypoints_xy"][:2])

            created, report = embedded.create_point(first, here, 8.0, images)
            assert created.point(report["point"])["w"] == 0.0
            try:
                outcome = created.add_observation(
                    report["point"], second, there, images
                )
            except ValueError:
                continue
            break
        assert outcome is not None, "no two-view point registers from a fresh patch"
        next_value, add = outcome
        record = next_value.point(add["point"])
        assert add["from_infinity"] is True
        assert record["w"] == 1.0
        assert len(record["image_indexes"]) == 2
        # The fit ran against the provisional patch, so the report carries a real
        # score rather than a placeholder.
        assert np.isfinite(add["zncc"])


class TestRemoveObservation:
    """``remove_observation``: one image drops out of a track.

    The same ``to_embedded_patches`` baseline the two edits above use, because
    the round trip runs an ``add_observation`` fit against the workspace's own
    photographs: a row is taken out and put back at the keypoint it held, and
    the point has to come home.
    """

    @pytest.fixture
    def embedded(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        return EditedReconstruction(recon.to_embedded_patches())

    @pytest.fixture
    def images(self, embedded):
        from sfmtool._workspace_image import read_workspace_image

        base = embedded.materialize()[0]
        return [
            read_workspace_image(base.workspace_dir, name) for name in base.image_names
        ]

    @pytest.fixture
    def long_track(self, embedded):
        """A point with three or more observations, and its first image."""
        for index in embedded.live_indexes():
            record = embedded.point(int(index))
            if len(record["image_indexes"]) >= 3:
                return int(index), int(record["image_indexes"][0])
        pytest.skip("no track of this reconstruction holds three observations")

    def test_an_image_outside_the_track_is_refused(self, embedded, long_track):
        point, _ = long_track
        seen = set(int(i) for i in embedded.point(point)["image_indexes"])
        unseen = next(
            i for i in range(embedded.materialize()[0].image_count) if i not in seen
        )
        with pytest.raises(ValueError, match="does not observe"):
            embedded.remove_observation(point, unseen)

    def test_a_dead_point_is_refused(self, embedded, long_track):
        point, image = long_track
        embedded.delete_point(point)
        with pytest.raises(ValueError, match="no live point"):
            embedded.remove_observation(point, image)

    def test_removing_a_row_shortens_the_track_and_keeps_the_base(
        self, embedded, long_track
    ):
        point, image = long_track
        before = list(int(i) for i in embedded.point(point)["image_indexes"])

        next_value, report = embedded.remove_observation(point, image)

        assert report["replaced"] == point
        assert report["deleted"] is False
        assert report["to_infinity"] is False
        assert report["retriangulated"] is True
        assert report["observation_count"] == len(before) - 1
        assert len(report["position"]) == 3
        record = next_value.point(report["point"])
        assert list(int(i) for i in record["image_indexes"]) == [
            i for i in before if i != image
        ]
        # This object is untouched, and the two agree on the base.
        assert list(int(i) for i in embedded.point(point)["image_indexes"]) == before
        assert next_value.base_content_hash() == embedded.base_content_hash()

        # The modification goes back into the place it came from.
        recon, forward, _ = next_value.materialize()
        assert recon.point_count == embedded.point_count
        assert forward[report["point"]] == point

    def test_the_round_trip_brings_the_observation_back(self, embedded, images):
        # Whether a given surface registers again in a given image is a
        # property of this fixture's solve, so the first track that survives
        # the round trip is the one the assertions run on: what is under test
        # is the value the two edits land on, not the fit.
        trip = None
        for index in embedded.live_indexes():
            index = int(index)
            record = embedded.point(index)
            if len(record["image_indexes"]) < 3:
                continue
            image = int(record["image_indexes"][0])
            pixel = [float(v) for v in record["keypoints_xy"][0]]
            shorter, removed = embedded.remove_observation(index, image)
            try:
                restored, added = shorter.add_observation(
                    removed["point"], image, pixel, images, min_zncc=-2.0
                )
            except ValueError:
                continue
            trip = (record, restored, added)
            break
        if trip is None:
            pytest.skip("no track of this reconstruction survives the round trip")
        record, restored, added = trip
        before = np.array(record["position"], dtype=float)

        home = restored.point(added["point"])
        assert list(int(i) for i in home["image_indexes"]) == list(
            int(i) for i in record["image_indexes"]
        )
        assert home["w"] == 1.0
        span = float(np.linalg.norm(before))
        assert np.linalg.norm(
            np.array(home["position"], dtype=float) - before
        ) < 0.1 * (span + 1.0)

    def test_removing_every_observation_ends_in_a_deleted_point(
        self, embedded, long_track
    ):
        point, _ = long_track
        value = embedded
        index = point
        report = None
        while report is None or not report["deleted"]:
            image = int(value.point(index)["image_indexes"][0])
            value, report = value.remove_observation(index, image)
            if not report["deleted"]:
                index = report["point"]
                assert report["to_infinity"] == (report["observation_count"] == 1)

        assert report["point"] is None
        assert report["observation_count"] == 0
        assert value.point(index) is None
        assert value.point_count == embedded.point_count - 1


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
        # Which images this capture corroborates is a property of its solve, so
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
        assert report["focal_released"] is False
        assert report["focal_before"] == report["focal_after"]
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
