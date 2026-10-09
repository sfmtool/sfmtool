# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bench bindings: the item list, the editable track, the commit.

The bench is values and pure functions, so the Python surface is the same shape
as the Rust one: a step takes a value and hands back the next one plus a report,
and the object it was called on is unchanged. These tests hold both halves and
check exactly that, plus the one step that crosses into the reconstruction.
"""

import numpy as np
import pytest

from sfmtool import bench as bench_module
from sfmtool.bench import (
    Bench,
    add_observation,
    apply_thresholds,
    commit,
    create_cluster,
    create_track,
    duplicate,
    evaluate,
    fit,
    pin_verdict,
    resize_patch,
    resize_patch_to_pixel,
    set_stage,
    set_verdict,
    shape_observation,
    sight_observation,
    spin_patch,
    split,
    tilt_patch,
    translate_patch,
    translate_patch_to_pixel,
    unpin_verdict,
)
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction


@pytest.fixture(scope="module")
def embedded(seoul_bull_workspace_once):
    """The 17-image reconstruction, converted to ``embedded_patches``.

    A track is committed back as a keypoint per observation, which is what
    ``embedded_patches`` stores and what ``sift_files`` has no room for.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace_once)
    return recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)


@pytest.fixture
def edited(embedded):
    return EditedReconstruction(embedded)


@pytest.fixture(scope="module")
def images(embedded):
    """The workspace's own photographs, one per image of the reconstruction.

    Every step that registers pixels takes these: a reconstruction carries
    poses and lenses, and the kernels need what the cameras saw.
    """
    from sfmtool._workspace_image import read_workspace_image

    return [
        read_workspace_image(embedded.workspace_dir, name)
        for name in embedded.image_names
    ]


@pytest.fixture
def long_track_point(embedded):
    """The point of the fixture whose track is the longest it has.

    The longest rather than the first one over a bar: a point many photographs
    saw is the specimen every step here is interesting on, and the fixture's
    shortest tracks carry patch frames whose tiles run off the photographs they
    would be read in, which is a fact about ``to_embedded_patches`` on a
    17-image toy reconstruction rather than about the steps under test.
    """
    counts = np.asarray(embedded.observation_counts)
    if counts.size == 0 or counts.max() < 3:
        pytest.skip("the fixture holds no track of three observations")
    return int(np.argmax(counts))


class TestTheBench:
    def test_an_empty_bench_holds_nothing(self):
        bench = Bench()
        assert len(bench) == 0
        assert bench.labels == []
        assert bench.track("bull-nose") is None

    def test_a_cluster_is_labelled_by_its_image_and_pixel(self):
        bench, track = create_cluster(
            Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5
        )
        assert bench.labels == ["IMG_0042@142,198"]
        assert track.stage == "cluster"
        assert track.observation_count == 1
        assert track.verdict_counts == (1, 0)
        assert track.reference == 0
        assert track.origin is None

    def test_a_radius_in_pixels_is_the_size_the_patch_spans(self):
        """The shape is per keypoint-frame unit; the radius is what sizes it."""
        _, track = create_cluster(Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5)
        shape = np.asarray(track.observation(0)["cluster"]["seed_shape"])
        # A column's pixel half-width is the radius times its norm, which is
        # the half-width the call asked for.
        np.testing.assert_allclose(
            track.radius * np.linalg.norm(shape, axis=0), [7.5, 7.5]
        )

    def test_a_cluster_from_a_feature_is_labelled_by_its_index(self):
        _, track = create_cluster(
            Bench(),
            4,
            "IMG_0042",
            (142.0, 197.5),
            shape=[[6.0, 0.0], [0.0, 6.0]],
            feature=847,
        )
        observation = track.observation(0)
        assert observation["provenance"] == {"kind": "descriptor", "feature": 847}
        np.testing.assert_allclose(
            observation["cluster"]["seed_shape"], [[6.0, 0.0], [0.0, 6.0]]
        )

    def test_a_seed_with_no_size_is_refused(self):
        with pytest.raises(ValueError, match="radius_px"):
            create_cluster(Bench(), 4, "IMG_0042", (142.0, 197.5))

    def test_a_second_cluster_from_the_same_pixel_takes_a_suffix(self):
        bench, _ = create_cluster(Bench(), 4, "IMG_0042", (1.0, 2.0), radius_px=7.5)
        bench, _ = create_cluster(bench, 4, "IMG_0042", (1.0, 2.0), radius_px=7.5)
        assert bench.labels == ["IMG_0042@1,2", "IMG_0042@1,2 (2)"]

    def test_a_rename_frees_the_old_label(self):
        bench, _ = create_cluster(Bench(), 4, "IMG_0042", (1.0, 2.0), radius_px=7.5)
        renamed = bench.rename("IMG_0042@1,2", "bull-nose")
        assert renamed.labels == ["bull-nose"]
        # The bench a step was called on is not changed by it.
        assert bench.labels == ["IMG_0042@1,2"]

    def test_an_item_id_is_an_int_that_a_rename_keeps(self):
        bench, _ = create_cluster(Bench(), 1, "a", (1.0, 1.0), radius_px=7.5)
        bench, _ = create_cluster(bench, 2, "b", (2.0, 2.0), radius_px=7.5)
        first = bench.id("a@1,1")
        second = bench.id("b@2,2")
        assert isinstance(first, int)
        assert isinstance(second, int)
        assert first != second
        renamed = bench.rename("a@1,1", "bull-nose")
        assert renamed.id("bull-nose") == first
        assert renamed.id("a@1,1") is None
        assert Bench().id("bull-nose") is None

    def test_a_rename_to_a_label_with_a_control_character_is_refused(self):
        bench, _ = create_cluster(Bench(), 4, "IMG_0042", (1.0, 2.0), radius_px=7.5)
        with pytest.raises(ValueError, match="control character"):
            bench.rename("IMG_0042@1,2", "line1\nline2")
        with pytest.raises(ValueError, match="something in it"):
            bench.rename("IMG_0042@1,2", "   ")
        assert bench.labels == ["IMG_0042@1,2"]

    def test_a_discard_takes_off_only_that_item(self):
        bench, _ = create_cluster(Bench(), 1, "a", (1.0, 1.0), radius_px=7.5)
        bench, _ = create_cluster(bench, 2, "b", (2.0, 2.0), radius_px=7.5)
        after = bench.discard("b@2,2")
        assert after.labels == ["a@1,1"]
        # The bench a step was called on is not changed by it.
        assert bench.labels == ["a@1,1", "b@2,2"]
        after = after.discard("a@1,1")
        assert len(after) == 0

    def test_a_label_that_names_nothing_is_refused_by_name(self):
        with pytest.raises(ValueError, match="nothing on the bench is called"):
            Bench().discard("bull-nose")


class TestTheEditableTrack:
    def test_a_point_put_on_the_bench_is_a_track_with_every_observation_in(
        self, edited, embedded, long_track_point
    ):
        bench, track = create_track(Bench(), edited, point=long_track_point)
        assert track.stage == "track"
        count = int(embedded.observation_counts[long_track_point])
        assert track.observation_count == count
        assert track.verdict_counts == (count, 0)
        assert track.origin == {"version": 0, "point": long_track_point}
        assert len(bench.labels) == 1
        assert bench.labels[0].startswith("pt3d_")
        assert bench.labels[0].endswith(f"_{long_track_point}")
        np.testing.assert_allclose(track.position, embedded.positions[long_track_point])

        # The keypoints are carried, not refitted, and the point's verdicts
        # arrive pinned, so no evaluation moves them.
        for observation in track.observations:
            assert observation["verdict"] == "in"
            assert observation["pinned"]
            assert observation["provenance"] == {"kind": "origin"}
            assert "keypoint" in observation["track"]

    def test_a_named_label_is_used_as_given(self, edited, long_track_point):
        bench, _ = create_track(
            Bench(), edited, long_track_point, label="pt3d_a1b2c3d4_1207", version=7
        )
        assert bench.labels == ["pt3d_a1b2c3d4_1207"]
        assert bench.track("pt3d_a1b2c3d4_1207").origin["version"] == 7

    def test_a_named_label_with_a_control_character_is_refused(
        self, edited, long_track_point
    ):
        with pytest.raises(ValueError, match="control character"):
            create_track(Bench(), edited, long_track_point, label="bull\tnose")

    def test_a_point_that_is_not_live_is_refused(self, edited):
        edited.delete_point(0)
        with pytest.raises(ValueError, match="no live point"):
            create_track(Bench(), edited, 0)

    def test_an_added_observation_joins_out_and_unpinned(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        before = track.observation_count
        grown, report = add_observation(track, 0, (12.5, 30.25))
        assert report == {"observation": before, "image": 0}
        assert grown.observation_count == before + 1
        added = grown.observation(before)
        assert added["verdict"] == "out"
        assert not added["pinned"]
        assert added["provenance"] == {"kind": "pixel"}
        np.testing.assert_allclose(added["cluster"]["seed_position"], [12.5, 30.25])
        # The track the step was called on is unchanged.
        assert track.observation_count == before

    def test_two_observations_in_one_image_cannot_both_be_in(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        held = int(track.observation(0)["image"])
        grown, _ = add_observation(track, held, (5.0, 5.0))
        with pytest.raises(ValueError, match="already has observation 0 in"):
            set_verdict(grown, grown.observation_count - 1, "in")

        freed, _ = set_verdict(grown, 0, "out")
        settled, report = set_verdict(freed, freed.observation_count - 1, "in")
        assert report["was"] == "out"
        assert report["is"] == "in"
        assert report["changed"]
        assert settled.observation(0)["verdict"] == "out"

    def test_a_hand_set_verdict_is_pinned_and_survives_the_painting(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        pinned, _ = set_verdict(track, 1, "out")
        assert pinned.observation(1)["pinned"]
        painted, report = apply_thresholds(pinned, min_zncc=0.0)
        assert painted.observation(1)["verdict"] == "out"
        # Every row of a point arrives pinned.
        assert report["pinned"] == track.observation_count

    def test_unpinning_hands_a_verdict_back_to_the_thresholds(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        pinned, _ = set_verdict(track, 1, "out")
        unpinned, report = unpin_verdict(pinned, 1)
        assert not unpinned.observation(1)["pinned"]
        # Nothing has measured it, so there is no proposal and it keeps its
        # verdict.
        assert unpinned.observation(1)["verdict"] == "out"
        assert report == {
            "unpinned": 1,
            "turned_in": 0,
            "turned_out": 0,
            "changed": True,
        }
        _, again = unpin_verdict(unpinned, 1)
        assert not again["changed"]
        with pytest.raises(ValueError):
            unpin_verdict(unpinned, 99)

    def test_an_unpin_takes_a_list_of_rows_or_all_of_them(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        count = track.observation_count
        some, report = unpin_verdict(track, [0, 1])
        assert report["unpinned"] == 2
        assert [o["pinned"] for o in some.observations][:2] == [False, False]
        assert all(o["pinned"] for o in some.observations[2:])

        every, report = unpin_verdict(some, "all")
        assert report["unpinned"] == count - 2
        assert report["changed"]
        assert not any(o["pinned"] for o in every.observations)
        _, report = unpin_verdict(every, "all")
        assert report == {
            "unpinned": 0,
            "turned_in": 0,
            "turned_out": 0,
            "changed": False,
        }
        with pytest.raises(ValueError, match="unknown observations"):
            unpin_verdict(track, "some")
        with pytest.raises(ValueError):
            unpin_verdict(track, [0, 99])

    def test_a_pin_keeps_each_verdict_as_it_stands(self, edited, long_track_point):
        _, track = create_track(Bench(), edited, long_track_point)
        count = track.observation_count
        # Nothing measured, so the unpins keep the verdicts: every row `in`
        # but row 1, and none pinned.
        track, _ = unpin_verdict(track, "all")
        track, _ = set_verdict(track, 1, "out")
        track, _ = unpin_verdict(track, 1)
        verdicts = [o["verdict"] for o in track.observations]
        assert verdicts[:2] == ["in", "out"]
        assert not any(o["pinned"] for o in track.observations)

        one, report = pin_verdict(track, 1)
        assert report == {"pinned": 1, "changed": True}
        assert one.observation(1)["pinned"]
        assert not one.observation(0)["pinned"]

        some, report = pin_verdict(one, [0, 1, 1])
        assert report == {"pinned": 1, "changed": True}

        every, report = pin_verdict(some, "all")
        assert report == {"pinned": count - 2, "changed": True}
        assert all(o["pinned"] for o in every.observations)
        assert [o["verdict"] for o in every.observations] == verdicts

        _, report = pin_verdict(every, "all")
        assert report == {"pinned": 0, "changed": False}
        with pytest.raises(ValueError, match="unknown observations"):
            pin_verdict(track, "some")
        with pytest.raises(ValueError):
            pin_verdict(track, [0, 99])

    def test_the_painting_leaves_an_unmeasured_observation_where_it_is(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        track, _ = unpin_verdict(track, "all")
        painted, report = apply_thresholds(track, min_zncc=0.99)
        assert report["turned_out"] == 0
        assert report["unmeasured"] == track.observation_count
        assert painted.verdict_counts == track.verdict_counts

    def test_an_unknown_verdict_is_refused_by_name(self, edited, long_track_point):
        _, track = create_track(Bench(), edited, long_track_point)
        with pytest.raises(ValueError, match="unknown verdict"):
            set_verdict(track, 0, "maybe")

    def test_the_thresholds_default_to_the_kernels_own_bars(
        self, edited, long_track_point
    ):
        # The shift bar is the keypoint localizer's own search radius, 6
        # patch-grid px, since on the bench it is also the radius the
        # evaluation searches and how far a fit may move a sighting.
        _, track = create_track(Bench(), edited, long_track_point)
        assert track.thresholds == {
            "min_zncc": 0.7,
            "min_zncc_middle": 0.7,
            "max_shift_px": 6.0,
            "max_zncc_self_similarity_radius": 2.5,
            "max_projection_error_px": 3.0,
            "geometry_search_min_relative_zncc": 0.7,
        }


class TestEvaluating:
    """Reading a track, and the fit that moves it, at both stages."""

    def test_an_evaluation_measures_the_track_and_moves_no_verdict(
        self, edited, images, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        measured, report = evaluate(track, edited, images)

        assert report["stage"] == "track"
        assert report["measured"] + report["unmeasured"] == track.observation_count
        assert "reference" not in report
        # The reading is made against the track as it stands, so the position it
        # reports is the one the track already carried.
        np.testing.assert_allclose(report["position"], track.position)

        # The verdicts are the person's, and an evaluation is not the person.
        assert measured.verdict_counts == track.verdict_counts
        assert measured.stage == "track"
        # The object the step was called on is unchanged.
        assert track.observation(0)["track"].get("reprojection_error") is None

        placed = [o["track"] for o in measured.observations if "keypoint" in o["track"]]
        assert len(placed) == track.observation_count
        for entry in placed:
            assert len(entry["keypoint"]) == 2
            # Where the position puts the sighting is measured for every row
            # that has a pixel at all, scored this round or not.
            assert entry["reprojection_error"] >= 0.0
            assert entry["ray_angle_deg"] >= 0.0
            assert entry.get("seed_shift_px", 0.0) >= 0.0
            assert entry.get("projection_offset_px", 0.0) >= 0.0
            # Each score carries its parts: the ZNCC grid, and the tile's
            # self-similarity over its middle and each ninth of it.
            if "zncc" in entry:
                assert entry["zncc_grid"].shape == (3, 3)
            if "zncc_self_similarity_radius" in entry:
                assert 0.0 <= entry["zncc_self_similarity_radius"] <= 3.0
                assert 0.0 <= entry["zncc_self_similarity_radius_middle"] <= 3.0
                radii = entry["zncc_self_similarity_radius_grid"]
                assert radii.shape == (3, 3) and radii.dtype == np.float64
                assert ((radii >= 0.0) & (radii <= 3.0)).all()
                ellipses = entry["zncc_self_similarity_ellipse_grid"]
                assert ellipses["axes"].shape == (3, 3, 2)
                assert ellipses["axes"].dtype == np.float64
                # Each cell's radius is its ellipse's semi-major axis.
                np.testing.assert_array_equal(ellipses["axes"][..., 0], radii)
                assert ellipses["axes_is_at_least"].shape == (3, 3, 2)
                assert ellipses["major_angle"].shape == (3, 3)
                assert ellipses["matrix"].shape == (3, 3, 2, 2)
                surface = entry["zncc_self_similarity_surface"]
                assert surface.shape == (7, 7) and surface.dtype == np.float64
                assert surface[3, 3] == 1.0 or np.isnan(surface).all()
                # A reading at every shift of the square, corners included,
                # or none for a flat core.
                assert np.isfinite(surface).all() or np.isnan(surface).all()
                if not np.isnan(surface).all():
                    assert 0.0 < entry["zncc_self_similarity_tolerance"] < 1.0

    def test_an_evaluation_reports_the_self_similarity_ellipse(
        self, edited, images, long_track_point
    ):
        """Beside each self-similarity radius is its ellipse: in grid px, its
        semi-major axis the radius; in the photograph's px; and along the
        patch's u and v. Each carries its two semi-axes, whether each is only a
        lower bound, the major axis's angle and the 2 x 2 matrix."""
        _, track = create_track(Bench(), edited, long_track_point)
        measured, _ = evaluate(track, edited, images)

        def check_ellipse(ellipse):
            assert set(ellipse) == {"axes", "axes_is_at_least", "major_angle", "matrix"}
            major, minor = ellipse["axes"]
            assert 0.0 <= minor <= major
            assert ellipse["axes_is_at_least"].dtype == np.bool_
            angle = ellipse["major_angle"]
            assert np.isnan(angle) or 0.0 <= angle < np.pi
            np.testing.assert_allclose(
                np.sort(np.linalg.eigvalsh(ellipse["matrix"]))[::-1],
                ellipse["axes"] ** 2,
                rtol=1e-6,
                atol=1e-12,
            )

        read = [
            o["track"]
            for o in measured.observations
            if "zncc_self_similarity_radius" in o["track"]
        ]
        assert read
        for entry in read:
            for key, radius in [
                ("zncc_self_similarity_ellipse", "zncc_self_similarity_radius"),
                (
                    "zncc_self_similarity_ellipse_middle",
                    "zncc_self_similarity_radius_middle",
                ),
            ]:
                units = entry[key]
                assert set(units) == {"grid_px", "image_px", "patch"}
                # The semi-major axis in grid px is the radius itself, to the
                # bit, and no longer than the largest radius searched.
                assert units["grid_px"]["axes"][0] == entry[radius]
                assert units["grid_px"]["axes"][0] <= 3.0
                check_ellipse(units["grid_px"])
                check_ellipse(units["image_px"])
                # A finite point's patch reads lengths along u and v, lower
                # bounds where the grid's are both exact only if they are.
                patch = units["patch"]
                assert patch["kind"] == "length"
                check_ellipse(patch["ellipse"])
                if units["grid_px"]["axes_is_at_least"].any():
                    assert patch["ellipse"]["axes_is_at_least"][0]

    def test_an_evaluation_reports_the_reference_view_readings(
        self, edited, images, long_track_point
    ):
        """Every read row carries its viewing angle, coverage and clipped
        share; every in row its agreement with the others and what the
        reference-view rule decided, and the rule picks exactly one row."""
        _, track = create_track(Bench(), edited, long_track_point)
        measured, _ = evaluate(track, edited, images)

        rows = [o["track"] for o in measured.observations]
        picked = []
        for i, entry in enumerate(rows):
            assert 0.0 <= entry["viewing_angle_deg"] <= 180.0
            if "tilt_direction_deg" in entry:
                assert -180.0 <= entry["tilt_direction_deg"] <= 180.0
            assert 0.0 <= entry["coverage"] <= 1.0
            assert 0.0 <= entry["clipped_share"] <= 1.0
            # A committed point's rows are all in, so each is judged.
            standing = entry["reference_view"]
            assert set(standing) == {"is_reference", "rejected_by", "fallback"}
            assert entry["pair_zncc_grid"].shape == (3, 3)
            assert -1.0 <= entry["pair_zncc"] <= 1.0
            if standing["is_reference"]:
                assert standing["rejected_by"] is None
                picked.append(i)
            else:
                assert standing["rejected_by"] in {
                    "coverage",
                    "clipped",
                    "angle",
                    "cells",
                    "agreement",
                    "sharpness",
                }
            assert standing["fallback"] in {
                "none",
                "without_angle",
                "without_angle_or_cells",
                "without_any",
            }
        assert len(picked) == 1
        reference = rows[picked[0]]
        if reference["reference_view"]["fallback"] == "none":
            assert reference["coverage"] >= 0.99
            assert reference["viewing_angle_deg"] <= 65.0
            assert reference["clipped_share"] <= 0.05

    def test_a_fit_stores_the_reference_row_s_tile_and_scores_the_rest(
        self, edited, images, long_track_point
    ):
        """A fit stores the tile of the row the reference-view rule picks as
        the bitmap and names it; every other row carries its plain and
        blur-matched scores against that bitmap, and the bitmap's row reads 1."""
        _, track = create_track(Bench(), edited, long_track_point)
        fitted, _ = fit(track, edited, images)
        source = fitted.bitmap_observation
        assert source is not None
        rows = [o["track"] for o in fitted.observations]
        assert rows[source]["reference_view"]["is_reference"]
        assert rows[source]["bitmap_zncc"] == 1.0
        assert "sharper_than_bitmap" not in rows[source]
        for i, entry in enumerate(rows):
            if i == source or "bitmap_zncc" not in entry:
                continue
            assert -1.0 <= entry["bitmap_zncc"] < 1.0
            assert entry["bitmap_blur_sigma"] >= 0.0
            if entry["bitmap_blur_sigma"] == 0.0:
                assert entry["blur_matched_bitmap_zncc"] == entry["bitmap_zncc"]
            assert isinstance(entry["sharper_than_bitmap"], bool)

    def test_an_evaluation_lets_the_bars_decide_the_unpinned_rows_once(
        self, edited, images, long_track_point
    ):
        """Every unpinned verdict follows the bars, and a reading that only
        follows that repaint does not repaint again."""
        _, track = create_track(Bench(), edited, long_track_point)
        track, _ = unpin_verdict(track, "all")
        # Bars no reading clears, set while nothing is measured.
        track, _ = apply_thresholds(track, min_zncc=1.1)
        assert track.verdict_counts == (track.observation_count, 0)
        assert not track.repainted

        read, report = evaluate(track, edited, images)
        assert report["turned_out"] == report["measured"] > 0
        assert report["turned_in"] == 0
        assert read.repainted

        again, report = evaluate(read, edited, images)
        assert (report["turned_in"], report["turned_out"]) == (0, 0)
        assert again.verdict_counts == read.verdict_counts
        assert not again.repainted

    def test_the_reading_takes_its_memory_bounds_as_keyword_arguments(
        self, edited, images, long_track_point
    ):
        """The two bounds that keep a widened window from asking for the machine.

        The search window is widened to reach the furthest seed and each view's
        tile costs the square of that width, so how far a seed may sit and what
        one round's tiles may take together are both the caller's to set.
        """
        _, track = create_track(Bench(), edited, long_track_point)

        # A bound below zero is past every seed, so every row comes back named
        # rather than searched for.
        read, report = evaluate(track, edited, images, max_seed_offset_px=-1.0)
        assert report["measured"] == 0
        for observation in read.observations:
            assert "beyond" in observation["track"]["reason"]

        # A budget nothing fits in refuses in one sentence, in front of the
        # allocation rather than after it.
        with pytest.raises(ValueError, match="budget"):
            evaluate(track, edited, images, max_cache_bytes=1024)
        with pytest.raises(ValueError, match="budget"):
            fit(track, edited, images, max_cache_bytes=1024)

        # And the defaults read the track as they always did.
        _, report = evaluate(track, edited, images)
        assert report["measured"] > 0

    def test_an_evaluation_moves_nothing(self, edited, images, long_track_point):
        """A reading writes measurements and no geometry.

        The position, the frame's colour and confidence, and every keypoint come
        back exactly as they went in: what a reading changes is what each row
        says about itself.
        """
        _, track = create_track(Bench(), edited, long_track_point)
        read, _ = evaluate(track, edited, images)

        np.testing.assert_array_equal(read.position, track.position)
        assert read.stage == track.stage
        assert read.observation_count == track.observation_count
        for before, after in zip(track.observations, read.observations):
            np.testing.assert_array_equal(
                after["track"]["keypoint"], before["track"]["keypoint"]
            )
            assert after["verdict"] == before["verdict"]

    def test_an_evaluation_measures_every_observation_whatever_its_verdict(
        self, edited, images, long_track_point
    ):
        """An ``out`` row and a new row are read like every other row.

        Nothing is dropped by a gate, so a refusal stands beside the number it
        would have been judged on and a slider can propose taking it back.
        """
        _, track = create_track(Bench(), edited, long_track_point)
        track, _ = set_verdict(track, 1, "out")
        seen = {int(o["image"]) for o in track.observations}
        free = next(i for i in range(len(images)) if i not in seen)
        record = edited.point(long_track_point)
        pixel = tuple(float(v) for v in np.asarray(record["keypoints_xy"])[0])
        track, added = add_observation(track, free, pixel)

        read, report = evaluate(track, edited, images)
        assert report["measured"] + report["unmeasured"] == track.observation_count
        # Every row carries a track-stage slot, and every one that carries no
        # score says why in a sentence rather than coming back blank.
        for observation in read.observations:
            entry = observation["track"]
            assert ("zncc" in entry) != ("reason" in entry)
            if "reason" in entry:
                assert entry["reason"]
        out_row = read.observation(1)["track"]
        assert out_row.get("projection_offset_px", 0.0) >= 0.0
        # Every row of the point is pinned, the ``out`` one by hand, so each
        # keeps its verdict. Only the new row, which is unpinned, is the bars'
        # to decide, and it is taken in only when it clears them.
        assert added["observation"] == track.observation_count - 1
        for i in range(added["observation"]):
            assert read.observation(i)["verdict"] == track.observation(i)["verdict"]
        new = read.observation(added["observation"])
        assert not new["pinned"]
        assert new["verdict"] in ("in", "out")

    def test_a_row_with_nowhere_to_look_is_reported_with_a_reason(
        self, edited, images, long_track_point
    ):
        """A sighting pointed off the sensor is named, not silently dropped."""
        _, track = create_track(Bench(), edited, long_track_point)
        seen = {int(o["image"]) for o in track.observations}
        free = next(i for i in range(len(images)) if i not in seen)
        track, added = add_observation(track, free, (1.0e5, 1.0e5))

        read, report = evaluate(track, edited, images)
        entry = read.observation(added["observation"])["track"]
        assert "zncc" not in entry
        assert entry["reason"] == "it sits off the photograph"
        assert report["unmeasured"] >= 1

    def test_a_fit_moves_the_track_and_an_evaluation_of_it_agrees(
        self, edited, images, long_track_point
    ):
        """A fit ends by reading its own result, so the two never disagree."""
        _, track = create_track(Bench(), edited, long_track_point)
        fitted, report = fit(track, edited, images)

        assert report["placed"] >= 2
        assert len(report["position"]) == 3
        assert report["condition_number"] > 0.0
        assert report["evaluate"]["stage"] == "track"
        np.testing.assert_allclose(fitted.position, report["position"])

        # Reading the fitted track again gives the same numbers, to the digit.
        read, again = evaluate(fitted, edited, images)
        assert (again["measured"], again["unmeasured"]) == (
            report["evaluate"]["measured"],
            report["evaluate"]["unmeasured"],
        )
        for before, after in zip(fitted.observations, read.observations):
            assert before["track"].get("zncc") == after["track"].get("zncc")
            assert before["track"].get("seed_shift_px") == after["track"].get(
                "seed_shift_px"
            )

    def test_a_downgrade_re_seeds_every_sighting_and_drops_the_geometry(
        self, edited, images, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        cluster, report = set_stage(track, edited, images, "cluster")

        assert report["from"] == "track"
        assert report["to"] == "cluster"
        assert report["changed"]
        assert cluster.stage == "cluster"
        assert cluster.reference == report["reference"]
        assert cluster.position is None, "the 3D hypothesis is what a downgrade drops"
        for before, after in zip(track.observations, cluster.observations):
            keypoint = before["track"]["keypoint"]
            seed = after["cluster"]["seed_position"]
            np.testing.assert_allclose(seed, keypoint, atol=1e-5)
            # The shape is the frame projected into that image, so it spans an
            # area, and it is per keypoint-frame unit: read over the cluster's
            # radius it is the patch's own footprint in that photograph, which
            # fits inside it, rather than the radius times that, which would
            # not.
            shape = np.asarray(after["cluster"]["seed_shape"])
            assert abs(np.linalg.det(shape)) > 0.0
            half_widths = cluster.radius * np.linalg.norm(shape, axis=0)
            height, width = images[after["image"]].shape[:2]
            assert np.all(half_widths <= max(height, width)), half_widths
            # What the track stage measured went with the stage.
            assert "track" not in after

    def test_an_upgrade_comes_back_to_the_point_it_came_from(
        self, edited, images, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        cluster, _ = set_stage(track, edited, images, "cluster")
        again, report = set_stage(cluster, edited, images, "track")

        assert again.stage == "track"
        # The upgrade is a fit, and a fit ends by reading its own result.
        assert report["fit"]["evaluate"]["stage"] == "track"
        assert report["fit"]["placed"] >= 2
        moved = np.linalg.norm(np.asarray(again.position) - np.asarray(track.position))
        extent = np.linalg.norm(np.asarray(edited.point(long_track_point)["position"]))
        assert moved < 0.05 * max(extent, 1.0), f"the round trip moved {moved}"
        # Only the track stage's measurements are on the observations now.
        for observation in again.observations:
            assert "cluster" not in observation and "track" in observation

    def test_setting_the_stage_a_track_is_already_at_changes_nothing(
        self, edited, images, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        same, report = set_stage(track, edited, images, "track")
        assert not report["changed"]
        assert (report["from"], report["to"]) == ("track", "track")
        assert "fit" not in report
        assert same.observation_count == track.observation_count

    def test_an_unknown_stage_is_refused_by_name(
        self, edited, images, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        with pytest.raises(ValueError, match="unknown stage"):
            set_stage(track, edited, images, "surfel")

    def test_a_track_stage_fit_of_one_sighting_is_refused(
        self, edited, images, long_track_point
    ):
        """A fit needs a consensus; a reading of the same track does not."""
        _, track = create_track(Bench(), edited, long_track_point)
        for i in range(1, track.observation_count):
            track, _ = set_verdict(track, i, "out")
        with pytest.raises(ValueError, match="needs two or more"):
            fit(track, edited, images)
        read, report = evaluate(track, edited, images)
        assert report["measured"] + report["unmeasured"] == track.observation_count
        assert read.observation_count == track.observation_count

    def test_a_cluster_from_pixels_refines_upgrades_and_commits(
        self, edited, images, long_track_point
    ):
        """The whole path a new sighting takes: pixels, a refinement, a point.

        The two pixels are a committed track's own sightings, so they are two
        photographs of one surface -- which is what the person pointing at them
        would be claiming.
        """
        record = edited.point(long_track_point)
        seen = [int(i) for i in record["image_indexes"]]
        keypoints = np.asarray(record["keypoints_xy"])

        bench, track = create_cluster(
            Bench(),
            seen[0],
            "IMG_0000",
            tuple(float(v) for v in keypoints[0]),
            # A half-width in pixels, so a patch of 36 px across: a template of
            # a few pixels has too little of the photograph in it to localize.
            radius_px=18.0,
        )
        track, added = add_observation(
            track, seen[1], tuple(float(v) for v in keypoints[1])
        )
        track, _ = set_verdict(track, added["observation"], "in")

        refined, report = evaluate(track, edited, images)
        assert report["stage"] == "cluster"
        assert refined.stage == "cluster"
        assert refined.reference == report["reference"]
        for observation in refined.observations:
            assert "status" in observation["cluster"]

        upgraded, staged = set_stage(refined, edited, images, "track")
        assert staged["changed"]
        assert len(upgraded.position) == 3

        after, commit_report = commit(edited, upgraded, node="bull")
        assert commit_report["observation_count"] == 2
        assert after.point_count == edited.point_count + 1
        written = after.point(commit_report["point"])
        np.testing.assert_allclose(written["position"], upgraded.position)


class TestSplitting:
    def test_a_split_takes_exactly_the_named_observations(
        self, edited, long_track_point
    ):
        bench, track = create_track(Bench(), edited, long_track_point)
        label = bench.labels[0]
        images = [int(o["image"]) for o in track.observations]

        bench, report = split(bench, edited, label, [1])
        assert report["label"] == f"{label}-split"
        assert (report["moved"], report["kept"]) == (1, len(images) - 1)

        first = bench.track(label)
        second = bench.track(report["label"])
        assert [int(o["image"]) for o in first.observations] == [
            image for i, image in enumerate(images) if i != 1
        ]
        assert [int(o["image"]) for o in second.observations] == [images[1]]
        # The second half is a point of its own, so a commit of it creates.
        assert second.origin is None
        # And it is a set of patches again: the 3D hypothesis fitted to both
        # halves is the thing the split is questioning.
        assert first.stage == "track"
        assert second.stage == "cluster"
        assert second.observation(0)["cluster"]["seed_position"] is not None

    def test_a_split_of_nothing_or_of_everything_is_refused(
        self, edited, long_track_point
    ):
        bench, track = create_track(Bench(), edited, long_track_point)
        label = bench.labels[0]
        with pytest.raises(ValueError, match="no observations|empty track"):
            split(bench, edited, label, [])
        with pytest.raises(ValueError, match="leave an empty track"):
            split(bench, edited, label, list(range(track.observation_count)))
        assert len(bench) == 1


class TestCommitting:
    def test_a_commit_with_an_origin_replaces_the_point(self, edited, long_track_point):
        _, track = create_track(Bench(), edited, long_track_point)
        track, _ = apply_thresholds(track, min_zncc=0.9)
        after, report = commit(edited, track, node="bull")

        assert report["replaced"] == long_track_point
        assert report["point"] == edited.base_point_count
        assert report["observation_count"] == track.observation_count
        assert report["absorbed"].size == 0
        # The commit says what it did to point indexes in the same vocabulary
        # every other edit does, so a caller carrying a selection follows it.
        index_map = report["map"]
        assert index_map.kind == "replaced"
        assert index_map.payload == [(long_track_point, report["point"])]
        assert index_map.forward(long_track_point) == report["point"]
        assert index_map.inverse(report["point"]) == long_track_point
        assert report["label"] == (
            f"Committed track: {track.observation_count} observations in bull, "
            f"replacing point {long_track_point}"
        )

        assert after.point(long_track_point) is None
        assert after.point_count == edited.point_count
        written = after.point(report["point"])
        original = edited.point(long_track_point)
        np.testing.assert_allclose(written["position"], original["position"])
        np.testing.assert_array_equal(
            written["image_indexes"], original["image_indexes"]
        )
        np.testing.assert_allclose(written["keypoints_xy"], original["keypoints_xy"])

    def test_a_commit_with_no_origin_appends(self, edited, images, long_track_point):
        # A track started from a search rather than from the point has no
        # origin; splitting one off is how a caller reaches that state here.
        # The half taken off is a cluster, so it is upgraded before it can be
        # written back -- which is the whole path a new sighting takes.
        bench, _ = create_track(Bench(), edited, long_track_point)
        bench, report = split(bench, edited, bench.labels[0], [0, 1])
        half = bench.track(report["label"])
        assert half.stage == "cluster"
        half, staged = set_stage(half, edited, images, "track")
        assert staged["changed"]
        after, commit_report = commit(edited, half)
        assert "replaced" not in commit_report, "a track with no origin creates"
        assert commit_report["point"] == edited.base_point_count
        assert after.point_count == edited.point_count + 1
        assert after.point(long_track_point) is not None
        index_map = commit_report["map"]
        assert index_map.kind == "created"
        assert index_map.payload == [commit_report["point"]]
        assert index_map.inverse(commit_report["point"]) is None

    def test_a_commit_onto_the_point_that_holds_the_track_writes_nothing(
        self, edited, long_track_point
    ):
        # The second press of Commit: the track is seated on the point the
        # first one wrote, which already holds exactly this record, so there is
        # nothing to write and no index to mint.
        _, track = create_track(Bench(), edited, long_track_point)
        after, first = commit(edited, track, node="bull")
        assert first["changed"]

        settled = track.with_origin(1, first["point"])
        again, report = commit(after, settled, node="bull")
        assert not report["changed"]
        assert report["point"] == first["point"]
        assert "replaced" not in report
        assert report["map"].forward(first["point"]) == first["point"]
        assert report["label"] == (
            f"Committed track: no effect, point {first['point']} of bull already holds it"
        )
        assert again.point_count == after.point_count
        assert again.index_bound == after.index_bound

    def test_a_track_with_one_observation_in_refuses(self, edited, long_track_point):
        _, track = create_track(Bench(), edited, long_track_point)
        for i in range(1, track.observation_count):
            track, _ = set_verdict(track, i, "out")
        with pytest.raises(ValueError, match="a point needs two or more"):
            commit(edited, track)

    def test_a_cluster_stage_track_refuses_and_names_the_upgrade(self, edited):
        _, track = create_cluster(Bench(), 0, "IMG_0000", (10.0, 10.0), radius_px=7.5)
        with pytest.raises(ValueError, match="upgrade it before committing"):
            commit(edited, track)

    def test_a_sift_files_reconstruction_refuses(
        self, edited, long_track_point, seoul_bull_workspace_once
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        sift_files = EditedReconstruction(
            SfmrReconstruction.load(seoul_bull_workspace_once)
        )
        with pytest.raises(ValueError, match="embedded_patches"):
            commit(sift_files, track)

    def test_a_sighting_added_at_the_track_stage_commits_at_its_pixel(
        self, edited, embedded, long_track_point
    ):
        # At the track stage the pixel an observation is added at is its
        # keypoint, so a kept one commits there without a fit, and a sighting
        # pulled from another point absorbs that point.
        other = (long_track_point + 1) % embedded.point_count
        _, track = create_track(Bench(), edited, long_track_point)
        held = {int(o["image"]) for o in track.observations}
        image = next(i for i in range(edited.image_count) if i not in held)
        pulled, report = add_observation(
            track, image, (10.0, 10.0), provenance="point", point=other
        )
        assert pulled.observation(report["observation"])["provenance"] == {
            "kind": "point",
            "point": other,
        }
        added = pulled.observation(report["observation"])
        np.testing.assert_allclose(added["track"]["keypoint"], [10.0, 10.0])
        kept, _ = set_verdict(pulled, report["observation"], "in")
        after, committed = commit(edited, kept)
        assert list(committed["absorbed"]) == [other]
        written = after.point(committed["point"])
        slot = [int(i) for i in written["image_indexes"]].index(image)
        np.testing.assert_allclose(written["keypoints_xy"][slot], [10.0, 10.0])

    def test_a_point_provenance_needs_the_point_it_came_from(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        with pytest.raises(ValueError, match="needs the 'point'"):
            add_observation(track, 0, (1.0, 1.0), provenance="point")


@pytest.fixture(scope="module")
def descriptor_index(embedded, tmp_path_factory):
    """A `.kdf` over the fixture's own `.sift` files, and those keypoints.

    One corpus image row per reconstruction image, in the reconstruction's own
    order, which is what lets a match name a reconstruction image directly.
    """
    import json
    from pathlib import Path

    from sfmtool.spatial import KdForest, LazyKdForest, write_kdf
    from sfmtool.sift.file import SiftReader, get_sift_path_for_image

    workspace = Path(embedded.workspace_dir)
    names = list(embedded.image_names)
    descriptors, positions, shapes, image_of, feature_of = [], [], [], [], []
    keypoints = {}
    for index, name in enumerate(names):
        reader = SiftReader(get_sift_path_for_image(workspace / name))
        rows = np.asarray(reader.read_descriptors())
        xy, affine = reader.read_positions_and_shapes()
        reader.close()
        xy = np.asarray(xy, dtype=np.float32)
        affine = np.asarray(affine, dtype=np.float32)
        keypoints[index] = (xy, affine)
        descriptors.append(rows)
        positions.append(xy)
        shapes.append(affine)
        image_of.append(np.full(len(rows), index, dtype=np.uint32))
        feature_of.append(np.arange(len(rows), dtype=np.uint32))

    config = json.loads((workspace / ".sfm-workspace.json").read_text())
    sources = {
        "workspace": {
            "absolute_path": str(workspace),
            "relative_path": ".",
            "contents": {
                "feature_tool": config["feature_tool"],
                "feature_type": config["feature_type"],
                "feature_options": json.dumps(config["feature_options"]),
                "feature_prefix_dir": config["feature_prefix_dir"],
            },
        },
        "image_names": names,
        "feature_tool_hashes": [bytes(16)] * len(names),
        "sift_content_hashes": [bytes(16)] * len(names),
        "image_indexes": np.concatenate(image_of).tolist(),
        "image_feature_indexes": np.concatenate(feature_of).tolist(),
        "positions": np.vstack(positions),
        "affine_shapes": np.vstack(shapes),
    }
    forest = KdForest(np.vstack(descriptors), num_trees=4, leaf_size=16, seed=5)
    path = tmp_path_factory.mktemp("bench_index") / "index.kdf"
    write_kdf(forest, str(path), sources=sources)
    return LazyKdForest(str(path)), keypoints


class TestTheDescriptorSearch:
    """`search_descriptors` over a real index of the fixture's own capture."""

    def test_the_search_reports_the_images_it_found_and_seeds_each_one(
        self, edited, descriptor_index, long_track_point
    ):
        forest, keypoints = descriptor_index
        _, track = create_track(Bench(), edited, long_track_point)
        image = track.observations[0]["image"]
        xy, affine = keypoints[image]
        held = {o["image"] for o in track.observations}

        grown, report = bench_module.search_descriptors(
            track, 0, xy, affine, forest, radius_px=40.0, min_inliers=6
        )
        assert set(report) == {
            "observation",
            "observation_count",
            "image",
            "center",
            "constellation",
            "added",
            "already_in_track",
            "sentence",
            "matches",
        }
        assert report["observation"] == 0
        assert report["observation_count"] == track.observation_count
        assert report["image"] == image
        assert report["constellation"] > 0
        assert report["sentence"].startswith("Searched from observation 0 of")
        # The searched image is never added by its own search.
        assert all(m["image"] != image for m in report["matches"])
        assert report["added"] + report["already_in_track"] == len(report["matches"])
        # Nothing is mutated in place: the step hands back the next value,
        # and what it added is exactly what the report says it added.
        assert grown.observation_count == track.observation_count + report["added"]

        for match in report["matches"]:
            assert set(match) >= {
                "image",
                "inliers",
                "correspondences",
                "affine",
                "pixel",
                "found",
            }
            assert match["affine"].shape == (2, 3)
            assert match["inliers"] >= 6
            assert match["inliers"] <= match["correspondences"]
            if match["image"] in held:
                assert match["found"] == "already_in_track"
            else:
                assert match["found"] == "added"
                added = grown.observations[match["observation"]]
                assert added["image"] == match["image"]
                assert added["verdict"] == "out"
                assert added["provenance"] == {
                    "kind": "search",
                    "inliers": match["inliers"],
                }
                # The seed is where the report says the warp put it.
                np.testing.assert_allclose(
                    added["cluster"]["seed_position"], match["pixel"], atol=1e-9
                )

    def test_a_bar_no_image_reaches_leaves_the_track_alone(
        self, edited, descriptor_index, long_track_point
    ):
        forest, keypoints = descriptor_index
        _, track = create_track(Bench(), edited, long_track_point)
        xy, affine = keypoints[track.observations[0]["image"]]
        grown, report = bench_module.search_descriptors(
            track, 0, xy, affine, forest, radius_px=40.0, min_inliers=10_000
        )
        assert report["matches"] == []
        assert report["added"] == 0
        assert grown.observation_count == track.observation_count

    def test_an_observation_past_the_end_is_refused(
        self, edited, descriptor_index, long_track_point
    ):
        forest, keypoints = descriptor_index
        _, track = create_track(Bench(), edited, long_track_point)
        xy, affine = keypoints[track.observations[0]["image"]]
        with pytest.raises(ValueError, match="past the"):
            bench_module.search_descriptors(track, 999, xy, affine, forest)

    def test_a_search_provenance_needs_its_inlier_count(self, edited, long_track_point):
        _, track = create_track(Bench(), edited, long_track_point)
        with pytest.raises(ValueError, match="needs the 'inliers'"):
            add_observation(track, 0, (1.0, 1.0), provenance="search")
        grown, report = add_observation(
            track, 0, (1.0, 1.0), provenance="search", inliers=11
        )
        assert grown.observations[report["observation"]]["provenance"] == {
            "kind": "search",
            "inliers": 11,
        }


class TestPlacingSizingAndTurningByHand:
    """The six steps a person's own hand reaches the geometry through.

    They are what the Image Detail panel's bench handles do. What the exactness
    of the resize claims -- the dragged edge reprojecting onto the pixel it was
    given and the opposite edge holding still through a real lens -- is proved
    in Rust, over a fixture whose camera is swapped for a distorting one; what
    is checked here is that the bindings carry each step's numbers and each
    step's refusals.
    """

    def test_sliding_the_patch_carries_every_sighting_with_it(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        was = track.observation(0)["track"]["keypoint"]
        before = track.placement

        moved, report = translate_patch_to_pixel(
            track, edited, 0, (was[0] + 6.0, was[1] - 4.0)
        )
        assert report["changed"]
        assert report["observation"] == 0
        assert report["placed"] == moved.observation_count
        assert report["moved"] > 0.0
        # In-plane only: the axes, the size and the normal are untouched.
        after = moved.placement
        for axis in ("u_halfvec", "v_halfvec"):
            np.testing.assert_allclose(after[axis], before[axis])
        offset = np.asarray(after["center"]) - np.asarray(before["center"])
        normal = np.cross(before["u_halfvec"], before["v_halfvec"])
        assert abs(float(offset @ normal)) < 1e-12, "the patch left its own plane"
        np.testing.assert_allclose(moved.position, after["center"])
        # Every sighting moved with it, and a translation set no pin.
        for index in range(moved.observation_count):
            observation = moved.observation(index)
            assert observation["pinned"] == track.observation(index)["pinned"]
            assert "zncc" not in observation["track"]
            before_at = track.observation(index)["track"]["keypoint"]
            assert not np.array_equal(observation["track"]["keypoint"], before_at)

    def test_a_placed_sighting_is_written_pinned_and_stripped_of_the_old_reading(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        was = track.observation(0)["track"]["keypoint"]
        moved, report = sight_observation(
            track, edited, 0, (was[0] + 3.0, was[1] - 4.0)
        )

        assert report["changed"]
        assert report["observation"] == 0
        assert report["moved_px"] == pytest.approx(5.0, abs=1e-3)
        placed = moved.observation(0)
        assert placed["track"]["keypoint"] == pytest.approx(
            (was[0] + 3.0, was[1] - 4.0), abs=1e-3
        )
        assert placed["pinned"], "a sighting a person placed is one they ruled on"
        assert "zncc" not in placed["track"], "the old reading does not hold here"
        # The track it was called on is untouched, as every step's is.
        np.testing.assert_array_equal(track.observation(0)["track"]["keypoint"], was)

    def test_a_centred_resize_moves_nothing_and_keeps_the_frame_square(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        sized, report = resize_patch(track, edited, 0.4)

        assert report["changed"] and report["half"] == 0.4
        frame = sized.placement
        assert np.linalg.norm(frame["u_halfvec"]) == pytest.approx(0.4, rel=1e-12)
        assert np.linalg.norm(frame["v_halfvec"]) == pytest.approx(0.4, rel=1e-12)
        np.testing.assert_allclose(frame["center"], track.placement["center"])
        # Every number read over the old square goes; where each sighting sits
        # does not.
        for before, after in zip(track.observations, sized.observations):
            np.testing.assert_array_equal(
                after["track"]["keypoint"], before["track"]["keypoint"]
            )
            assert "zncc" not in after["track"]

        with pytest.raises(ValueError, match="not a size"):
            resize_patch(track, edited, 0.0)

    def test_a_turn_keeps_the_axes_the_plane_and_every_sighting(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        turned, report = spin_patch(track, np.pi / 3.0)
        assert report["degrees"] == pytest.approx(60.0)

        before, after = track.placement, turned.placement
        for axis in ("u_halfvec", "v_halfvec"):
            assert np.linalg.norm(after[axis]) == pytest.approx(
                np.linalg.norm(before[axis]), rel=1e-12
            )
        np.testing.assert_allclose(after["center"], before["center"])

        def normal(frame):
            cross = np.cross(frame["u_halfvec"], frame["v_halfvec"])
            return cross / np.linalg.norm(cross)

        np.testing.assert_allclose(normal(after), normal(before), atol=1e-12)
        np.testing.assert_array_equal(
            turned.observation(0)["track"]["keypoint"],
            track.observation(0)["track"]["keypoint"],
        )

    def test_a_resize_from_an_edge_resizes_and_re_places_the_sighting_it_was_named_at(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        was = track.observation(0)["track"]["keypoint"]
        # A pixel out along the outline, which is what dragging the +u edge
        # hands the step.
        resized, report = resize_patch_to_pixel(
            track, edited, 0, "+u", (was[0] + 40.0, was[1])
        )

        assert report["changed"]
        assert report["observation"] == 0
        assert report["image"] == track.observation(0)["image"]
        assert report["half"] > report["was"], "the edge was pulled outward"
        frame = resized.placement
        assert np.linalg.norm(frame["u_halfvec"]) == pytest.approx(
            np.linalg.norm(frame["v_halfvec"]), rel=1e-12
        )
        # The patch grew toward the edge that was dragged, so its centre moved
        # with it and every dot follows.
        # A resize moves the centre, so every sighting follows it, exactly as a
        # slide's does; no pin is set or cleared.
        assert resized.observation(0)["pinned"] == track.observation(0)["pinned"]
        assert not np.array_equal(resized.observation(0)["track"]["keypoint"], was)
        assert not np.array_equal(
            resized.observation(1)["track"]["keypoint"],
            track.observation(1)["track"]["keypoint"],
        )

        with pytest.raises(ValueError, match="not an edge"):
            resize_patch_to_pixel(track, edited, 0, "sideways", (was[0], was[1]))

    def test_a_cluster_sighting_takes_the_shape_it_is_given(
        self, edited, long_track_point
    ):
        bench, track = create_cluster(
            Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=6.0
        )
        # A quarter turn of the seed's own shape, handed over as the matrix.
        was = np.asarray(track.observation(0)["cluster"]["seed_shape"])
        turn = np.array([[0.0, -1.0], [1.0, 0.0]]) @ was
        turned, shaped = shape_observation(track, 0, turn)

        assert shaped["changed"]
        np.testing.assert_allclose(shaped["shape"], turn)
        assert shaped["half_px"] == pytest.approx(
            track.radius * np.linalg.norm(turn[:, 0]), rel=1e-12
        ), "radius * the first column's norm"
        assert turned.observation(0)["pinned"] == track.observation(0)["pinned"], (
            "a turn is not a verdict"
        )
        with pytest.raises(ValueError, match="spans no area"):
            shape_observation(track, 0, [[1.0, 2.0], [2.0, 4.0]])

        # And the two stages own different steps: a surfel is not a cluster's.
        _, at_track = create_track(Bench(), edited, long_track_point)
        with pytest.raises(ValueError, match="cluster-stage step"):
            shape_observation(at_track, 0, turn)
        with pytest.raises(ValueError, match="track-stage step"):
            spin_patch(track, 0.5)


class TestDuplicating:
    """A copy of an item beside it: everything but the origin."""

    def test_a_copy_carries_the_patch_and_commits_as_a_creation(
        self, edited, long_track_point
    ):
        bench, track = create_track(Bench(), edited, long_track_point)
        label = bench.labels[0]

        bench, report = duplicate(bench, label)
        assert report["from"] == label
        assert report["label"] == f"{label} copy"
        assert report["observation_count"] == track.observation_count
        assert bench.labels == [label, report["label"]]

        copy = bench.track(report["label"])
        assert copy.stage == track.stage
        assert copy.observation_count == track.observation_count
        assert copy.origin is None, "a copy has to create rather than replace"
        assert bench.track(label).origin is not None, "the original kept its origin"
        np.testing.assert_allclose(copy.position, track.position)
        np.testing.assert_allclose(
            copy.placement["u_halfvec"], track.placement["u_halfvec"]
        )
        for index in range(copy.observation_count):
            was, now = track.observation(index), copy.observation(index)
            assert now["verdict"] == was["verdict"]
            assert now["pinned"] == was["pinned"]
            assert now["image"] == was["image"]
            assert now["provenance"] == was["provenance"]
            np.testing.assert_array_equal(
                now["track"]["keypoint"], was["track"]["keypoint"]
            )
            assert now["track"].keys() == was["track"].keys()

        # A second duplicate takes the bench's own collision suffix.
        bench, again = duplicate(bench, label)
        assert again["label"] == f"{label} copy (2)"

        # And the copy's commit creates: the point it was copied from is
        # untouched.
        after, committed = commit(edited, copy, node="run")
        assert "replaced" not in committed
        assert committed["point"] == edited.point_count
        assert after.point(long_track_point) is not None

        with pytest.raises(ValueError, match="nothing on the bench"):
            duplicate(bench, "nothing at all")


class TestTheNormalHandSteps:
    """``translate_patch`` and ``tilt_patch``: the two moves no pixel can name.

    The arithmetic -- the least rotation, the offsets kept, the cap -- is proved
    in Rust; what is checked here is that the bindings carry each step's numbers
    and hand back a new value.
    """

    @staticmethod
    def _normal(frame):
        cross = np.cross(frame["u_halfvec"], frame["v_halfvec"])
        return cross / np.linalg.norm(cross)

    def test_an_offset_along_the_normal_moves_the_centre_that_far(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        before = track.placement
        moved, report = translate_patch(track, edited, (0.0, 0.0, 0.05))

        assert report["changed"]
        assert report["moved"] == pytest.approx(0.05, rel=1e-9)
        np.testing.assert_allclose(report["by"], [0.0, 0.0, 0.05])
        after = moved.placement
        np.testing.assert_allclose(
            np.asarray(after["center"]) - np.asarray(before["center"]),
            0.05 * self._normal(before),
            atol=1e-12,
        )
        for axis in ("u_halfvec", "v_halfvec"):
            np.testing.assert_allclose(after[axis], before[axis])
        # The value it was called on is untouched.
        np.testing.assert_allclose(track.placement["center"], before["center"])

    def test_a_tilt_turns_the_normal_and_keeps_the_centre_and_size(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        before = track.placement
        n = self._normal(before)
        u = np.asarray(before["u_halfvec"]) / np.linalg.norm(before["u_halfvec"])
        asked = np.cos(np.radians(10.0)) * n + np.sin(np.radians(10.0)) * u

        turned, report = tilt_patch(track, edited, tuple(asked))
        assert report["changed"]
        assert report["stopped"] is None
        assert report["degrees"] == pytest.approx(10.0, abs=1e-6)
        np.testing.assert_allclose(report["normal"], asked, atol=1e-9)
        after = turned.placement
        np.testing.assert_allclose(self._normal(after), asked, atol=1e-9)
        np.testing.assert_allclose(after["center"], before["center"])
        assert np.linalg.norm(after["u_halfvec"]) == pytest.approx(
            np.linalg.norm(before["u_halfvec"]), rel=1e-12
        )

    def test_a_cluster_refuses_both(self, edited):
        _, track = create_cluster(Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5)
        with pytest.raises(ValueError, match="cluster"):
            translate_patch(track, edited, (0.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="cluster"):
            tilt_patch(track, edited, (0.0, 0.0, 1.0))


class TestTheGeometrySearch:
    """``search_geometry`` over the fixture's own photographs."""

    def test_a_search_appends_only_unpinned_out_sweep_rows(
        self, edited, images, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        grown, report = bench_module.search_geometry(track, 0, edited, images)

        assert set(report) == {
            "observation",
            "observation_count",
            "image",
            "reference_views",
            "self_agreement",
            "added",
            "already_in_track",
            "sentence",
            "matches",
        }
        assert report["observation"] == 0
        assert report["image"] == track.observations[0]["image"]
        assert report["sentence"].startswith("Geometry search from observation 0")
        assert grown.observation_count == track.observation_count + report["added"]
        held = {o["image"] for o in track.observations}
        for match in report["matches"]:
            assert set(match) >= {"image", "zncc", "pixel", "found"}
            if match["found"] == "added":
                assert match["image"] not in held
                row = grown.observations[match["observation"]]
                assert row["verdict"] == "out"
                assert not row["pinned"]
                assert row["provenance"] == {"kind": "sweep"}
                np.testing.assert_allclose(
                    row["cluster"]["seed_position"], match["pixel"], atol=1e-9
                )
        # The existing observations are left exactly as they were.
        for before, after in zip(track.observations, grown.observations):
            assert after["verdict"] == before["verdict"]

    def test_a_cluster_stage_track_is_refused_by_name(self, edited, images):
        _, track = create_cluster(Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5)
        with pytest.raises(ValueError, match="track stage"):
            bench_module.search_geometry(track, 0, edited, images)


def test_the_module_reports_its_public_location():
    assert bench_module.__name__ == "sfmtool.bench"
    assert Bench.__module__ == "sfmtool.bench"


class TestFinitePointsAndBearings:
    """The one classification, and the ``w = 0`` track it can write.

    A bearing and a position are the same three numbers under different rules,
    so the binding publishes the coordinate under the name of whichever it is
    and says which in a flag. What decides it is the reconstruction's own
    criterion, whose two knobs a caller can move.
    """

    @pytest.fixture
    def bearing_edited(self, embedded, long_track_point):
        """`embedded` with the long track's point stored as a bearing.

        The homogeneous column is what says which a point is, so writing a
        ``w = 0`` row is writing that column: the direction is the point's own,
        which ``clone_with_changes`` normalises onto the unit sphere.
        """
        xyzw = np.asarray(embedded.positions_xyzw).copy()
        xyz = xyzw[long_track_point, :3] / xyzw[long_track_point, 3]
        xyzw[long_track_point] = [*xyz, 0.0]
        return EditedReconstruction(embedded.clone_with_changes(positions=xyzw))

    def test_a_bearing_goes_onto_the_bench_as_a_direction(
        self, bearing_edited, long_track_point
    ):
        _, track = create_track(Bench(), bearing_edited, long_track_point)
        assert track.at_infinity
        assert track.position is None, "a bearing has no position"
        direction = np.asarray(track.direction)
        assert direction.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-9)
        assert track.placement["w"] == 0.0
        np.testing.assert_allclose(track.placement["center"], direction)

    def test_a_bearing_commits_back_as_a_bearing(
        self, bearing_edited, long_track_point
    ):
        _, track = create_track(Bench(), bearing_edited, long_track_point)
        after, report = commit(bearing_edited, track)
        written = after.point(report["point"])
        assert written["w"] == 0.0, "a bearing commits as a bearing"
        np.testing.assert_allclose(
            written["position"], np.asarray(track.direction), atol=1e-9
        )
        # A `w = 0` row carries a zero normal, which is what the format states.
        np.testing.assert_allclose(written["normal"], [0.0, 0.0, 0.0])

    def test_a_fit_reports_which_representation_the_rays_earned(
        self, edited, embedded, images, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        fitted, report = fit(track, edited, images)

        assert report["at_infinity"] is False
        assert len(report["position"]) == 3
        assert "direction" not in report
        assert report["kept_at_seed"] >= 0
        call = report["classification"]
        assert call["at_infinity"] is False
        assert call["reason"] in ("score_cleared", "midpoint_bound_cleared")
        assert call["max_pair_angle_deg"] > 0.0
        assert "finite at (" in call["text"]
        assert "likelihood ratio" in call["text"]
        assert not fitted.at_infinity
        np.testing.assert_allclose(fitted.position, report["position"])

        # The test's numbers: the rays are weighted at the reconstruction's
        # own measured noise, the depth cleared the threshold, and the point
        # fit that placed the track cleared it too.
        assert call["sigma_px"] == pytest.approx(embedded.reprojection_noise_px())
        assert call["threshold"] == pytest.approx(25.0)
        assert call["bearing_cost"] >= call["threshold"]
        assert max(call["depth_score"], call["midpoint_bound"]) >= call["threshold"]
        assert call["depth_likelihood_ratio"] >= call["threshold"]
        assert call["distance"] > call["min_depth"] > 0.0
        assert call["num_views"] >= 3

    def test_a_threshold_nothing_clears_writes_the_bearing(
        self, edited, images, long_track_point
    ):
        """A threshold above the bearing's own cost leaves every track a bearing.

        No depth can lower the bearing's cost by more than the cost itself, so
        the reason names the bearing's cost rather than the score.
        """
        _, track = create_track(Bench(), edited, long_track_point)
        demoted, report = fit(
            track, edited, images, depth_likelihood_ratio_threshold=1e300
        )

        assert report["at_infinity"] is True
        call = report["classification"]
        assert call["reason"] == "bearing_cost_below_threshold"
        assert call["threshold"] == pytest.approx(1e300)
        assert np.isnan(call["depth_likelihood_ratio"]), "no point fit ran"
        assert np.isnan(call["distance"]), "a bearing has no distance"
        assert "NaN" not in call["text"]
        np.testing.assert_allclose(np.linalg.norm(report["direction"]), 1.0)
        assert demoted.at_infinity
        assert demoted.placement["w"] == 0.0

    def test_the_classification_knobs_are_keyword_arguments(
        self, edited, images, long_track_point
    ):
        """The test's two knobs, defaulting to the reconstruction's own.

        ``sigma_px`` is the noise level the rays are weighted by and
        ``depth_likelihood_ratio_threshold`` the bar; the report echoes both.
        A larger noise level scales every cost down alike, so on a track with
        this much parallax it moves the numbers and not the place: the point
        fit's minimum does not depend on a uniform weight.
        """
        _, track = create_track(Bench(), edited, long_track_point)
        _, default = fit(track, edited, images)

        _, moved = fit(
            track,
            edited,
            images,
            sigma_px=2.0,
            depth_likelihood_ratio_threshold=10.0,
        )
        call = moved["classification"]
        assert call["sigma_px"] == pytest.approx(2.0)
        assert call["threshold"] == pytest.approx(10.0)
        assert call["at_infinity"] is False
        assert call["bearing_cost"] < default["classification"]["bearing_cost"]
        np.testing.assert_allclose(moved["position"], default["position"], atol=1e-6)


class TestNoEffectAndTheClamp:
    """The three facts the reports carry beside what a step did.

    A pixel named on the wire or under a pointer is turned into a ray, met with
    the patch's plane and projected back, and that round trip does not return bit
    for bit: an exact comparison reads every re-statement of where the patch
    already is as a move. And a pixel off the photograph names no place on it, so
    it is brought to the nearest place that it does.
    """

    def test_a_patch_edit_that_changes_nothing_reports_no_effect(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        was = track.observation(0)["track"]["keypoint"]

        # The centre put back under the pixel it already projects to.
        moved, report = translate_patch_to_pixel(track, edited, 0, (was[0], was[1]))
        assert not report["changed"]
        assert report["moved"] == 0.0
        assert not report["clamped"]
        assert report["clamped_from"] is None
        np.testing.assert_array_equal(
            moved.observation(0)["track"]["keypoint"],
            track.observation(0)["track"]["keypoint"],
        )

        # A turn under a nanoradian is no turn.
        _, turn = spin_patch(track, 1e-12)
        assert not turn["changed"]

        # A sighting put back within a thousandth of a pixel of where it sits.
        _, placed = sight_observation(track, edited, 0, (was[0] + 1e-6, was[1] - 1e-6))
        assert not placed["changed"]

    def test_a_pixel_off_the_photograph_is_brought_inside_it(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)

        _, report = translate_patch_to_pixel(track, edited, 0, (-500.0, -500.0))
        assert report["clamped"]
        assert tuple(report["clamped_from"]) == (-500.0, -500.0)

        _, report = sight_observation(track, edited, 0, (-500.0, -500.0))
        assert report["clamped"]
        assert tuple(report["clamped_from"]) == (-500.0, -500.0)
        assert tuple(report["pixel"]) == (0.0, 0.0)
