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
