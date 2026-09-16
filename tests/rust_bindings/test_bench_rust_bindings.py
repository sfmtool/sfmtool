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

from sfmtool._sfmtool import bench as bench_module
from sfmtool._sfmtool.bench import (
    Bench,
    add_observation,
    apply_thresholds,
    commit,
    create_cluster,
    create_track,
    evaluate,
    fit,
    set_stage,
    set_verdict,
    split,
)
from sfmtool._sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction


@pytest.fixture(scope="module")
def embedded(seoul_bull_workspace_once):
    """The 17-image solve as an ``embedded_patches`` reconstruction.

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
    17-image toy solve rather than about the steps under test.
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
        assert bench.active_label() is None
        assert bench.active_track is None
        assert bench.track("bull-nose") is None

    def test_a_cluster_is_labelled_by_its_image_and_pixel(self):
        bench, track = create_cluster(
            Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5
        )
        assert bench.labels == ["IMG_0042@142,198"]
        assert bench.active_label() == "IMG_0042@142,198"
        assert track.stage == "cluster"
        assert track.observation_count == 1
        assert track.verdict_counts == (1, 0, 0)
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
        assert renamed.active_label() == "bull-nose"
        # The bench a step was called on is not changed by it.
        assert bench.labels == ["IMG_0042@1,2"]

    def test_a_discard_leaves_the_one_before_it_active(self):
        bench, _ = create_cluster(Bench(), 1, "a", (1.0, 1.0), radius_px=7.5)
        bench, _ = create_cluster(bench, 2, "b", (2.0, 2.0), radius_px=7.5)
        assert bench.active_label() == "b@2,2"
        bench = bench.discard("b@2,2")
        assert bench.active_label() == "a@1,1"
        bench = bench.discard("a@1,1")
        assert len(bench) == 0
        assert bench.active_label() is None

    def test_a_label_that_names_nothing_is_refused_by_name(self):
        with pytest.raises(ValueError, match="nothing on the bench is called"):
            Bench().activate("bull-nose")


class TestTheEditableTrack:
    def test_a_point_put_on_the_bench_is_a_track_with_every_observation_in(
        self, edited, embedded, long_track_point
    ):
        bench, track = create_track(Bench(), edited, point=long_track_point)
        assert track.stage == "track"
        count = int(embedded.observation_counts[long_track_point])
        assert track.observation_count == count
        assert track.verdict_counts == (count, 0, 0)
        assert track.origin == {"version": 0, "point": long_track_point}
        assert len(bench.labels) == 1
        assert bench.labels[0].startswith("pt3d_")
        assert bench.labels[0].endswith(f"_{long_track_point}")
        np.testing.assert_allclose(track.position, embedded.positions[long_track_point])

        # The keypoints are carried, not refitted.
        for observation in track.observations:
            assert observation["verdict"] == "in"
            assert not observation["pinned"]
            assert observation["provenance"] == {"kind": "origin"}
            assert "keypoint" in observation["track"]

    def test_a_named_label_is_used_as_given(self, edited, long_track_point):
        bench, _ = create_track(
            Bench(), edited, long_track_point, label="pt3d_a1b2c3d4_1207", version=7
        )
        assert bench.labels == ["pt3d_a1b2c3d4_1207"]
        assert bench.track("pt3d_a1b2c3d4_1207").origin["version"] == 7

    def test_a_point_that_is_not_live_is_refused(self, edited):
        edited.delete_point(0)
        with pytest.raises(ValueError, match="no live point"):
            create_track(Bench(), edited, 0)

    def test_an_added_observation_is_an_unruled_candidate(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        before = track.observation_count
        grown, report = add_observation(track, 0, (12.5, 30.25))
        assert report == {"observation": before, "image": 0}
        assert grown.observation_count == before + 1
        added = grown.observation(before)
        assert added["verdict"] == "candidate"
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
        assert report["was"] == "candidate"
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
        assert report["pinned"] == 1

    def test_the_painting_leaves_an_unmeasured_observation_where_it_is(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
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
        _, track = create_track(Bench(), edited, long_track_point)
        assert track.thresholds == {
            "min_zncc": 0.85,
            "max_shift_px": 3.0,
            "max_keypoint_uncertainty": 0.35,
            "min_relative_zncc": 0.7,
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
            assert entry.get("localizability", 1.0) > 0.0

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
        """An ``out`` row and a candidate are read like every other row.

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
        assert read.verdict_counts == track.verdict_counts
        assert added["observation"] == track.observation_count - 1

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
        """The whole path a candidate takes: pixels, a refinement, a point.

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
        # written back -- which is the whole path a candidate takes.
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

    def test_a_kept_sighting_with_no_keypoint_refuses_by_name(
        self, edited, embedded, long_track_point
    ):
        # A sighting pulled from another point carries a seed and no keypoint:
        # only the track stage's evaluation places one, and there is no pixel to
        # store until it has.
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
        kept, _ = set_verdict(pulled, report["observation"], "in")
        with pytest.raises(ValueError, match="has no keypoint to store"):
            commit(edited, kept)

    def test_a_point_provenance_needs_the_point_it_came_from(
        self, edited, long_track_point
    ):
        _, track = create_track(Bench(), edited, long_track_point)
        with pytest.raises(ValueError, match="needs the 'point'"):
            add_observation(track, 0, (1.0, 1.0), provenance="point")


def test_the_module_reports_its_public_location():
    assert bench_module.__name__ == "sfmtool.bench"
    assert Bench.__module__ == "sfmtool.bench"
