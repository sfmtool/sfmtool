# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``bench.build_track_at_pixel`` and ``bench.TrackAtPixelSources``.

The capture is the 17-image seoul_bull fixture the other bench tests use, with
its own SIFT index and its clusters ``.matches``. A point is deleted from the
version and the query is made at one of its pixels, the way the evaluation
harness holds a point out.
"""

from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool import bench
from sfmtool._sfmtool.io import MatchesFile
from sfmtool._sfmtool.reconstruction import EditedReconstruction

from .test_bench_rust_bindings import (  # noqa: F401 (fixtures)
    descriptor_index,
    embedded,
    images,
    long_track_point,
)


@pytest.fixture(scope="module")
def sources(embedded, descriptor_index):  # noqa: F811
    forest, keypoints = descriptor_index
    matches = sorted(Path(embedded.workspace_dir).glob("matches/*.matches"))
    assert matches, "the fixture workspace holds a clusters .matches"
    return bench.TrackAtPixelSources(
        EditedReconstruction(embedded),
        forest,
        [keypoints[i] for i in range(len(embedded.image_names))],
        MatchesFile(str(matches[0])),
    )


def held_out(embedded, point):  # noqa: F811
    edited = EditedReconstruction(embedded)
    edited.delete_point(point)
    return edited


def test_a_held_out_point_is_rebuilt_on_its_pixel(
    embedded,  # noqa: F811
    images,  # noqa: F811
    sources,
    long_track_point,  # noqa: F811
):
    record = EditedReconstruction(embedded).point(long_track_point)
    image = int(record["image_indexes"][0])
    pixel = tuple(float(v) for v in record["keypoints_xy"][0])
    edited = held_out(embedded, long_track_point)

    track, report = bench.build_track_at_pixel(edited, images, sources, image, pixel)

    assert report["member"] in {"clusters", "transfer", "sweep", "constellation"}
    assert report["query_observation"] == 0
    assert isinstance(report["refusals"], list)
    assert track.stage == "track"
    query = track.observations[0]
    assert query["image"] == image
    assert query["verdict"] == "in"
    offset = np.linalg.norm(np.asarray(query["track"]["keypoint"]) - pixel)
    assert offset <= 2.0
    assert track.verdict_counts[0] >= 3
    assert report["final"]["in"] == track.verdict_counts[0]
    # Every in view carries the reading the final gates judged.
    assert all(
        o["track"].get("zncc") is not None
        for o in track.observations
        if o["verdict"] == "in"
    )


def test_a_refusal_carries_its_stage_reason_and_what_was_measured(
    embedded,  # noqa: F811
    images,  # noqa: F811
    sources,
):
    edited = EditedReconstruction(embedded)
    with pytest.raises(bench.TrackAtPixelError) as caught:
        bench.build_track_at_pixel(
            edited, images, sources, 0, (0.5, 0.5), members=["clusters"]
        )
    e = caught.value
    assert isinstance(e, ValueError)
    assert e.stage == "cascade"
    assert e.reason.startswith("every member refused; the last, clusters, at clusters")
    (refusal,) = e.diagnostics["refusals"]
    assert refusal["member"] == "clusters"
    assert refusal["stage"] == "clusters"
    assert refusal["diagnostics"]["clusters_near"] == 0


def test_a_query_off_the_photograph_is_refused_before_any_member_runs(
    embedded,  # noqa: F811
    images,  # noqa: F811
    sources,
):
    edited = EditedReconstruction(embedded)
    with pytest.raises(bench.TrackAtPixelError) as caught:
        bench.build_track_at_pixel(edited, images, sources, 0, (-5.0, 10.0))
    assert caught.value.stage == "query"
    assert caught.value.diagnostics == {}


def test_an_unknown_member_is_refused_by_name(
    embedded,  # noqa: F811
    images,  # noqa: F811
    sources,
):
    with pytest.raises(ValueError, match="unknown cascade member"):
        bench.build_track_at_pixel(
            EditedReconstruction(embedded),
            images,
            sources,
            0,
            (10.0, 10.0),
            members=["baseline"],
        )


def test_the_sources_need_one_keypoint_pair_per_image(
    embedded,  # noqa: F811
    descriptor_index,  # noqa: F811
):
    forest, keypoints = descriptor_index
    matches = sorted(Path(embedded.workspace_dir).glob("matches/*.matches"))
    with pytest.raises(ValueError, match="keypoints has 1 entries"):
        bench.TrackAtPixelSources(
            EditedReconstruction(embedded),
            forest,
            [keypoints[0]],
            MatchesFile(str(matches[0])),
        )
