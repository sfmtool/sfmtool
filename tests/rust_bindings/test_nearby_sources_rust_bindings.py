# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the matching sources of finding the tracks near a pixel:
``bench.nearby_points``, ``bench.nearby_cluster_tracks``,
``bench.guided_matches`` and ``bench.constellation_seeds``, with
``bench.NearbyTrackSources``.

The capture is the 17-image seoul_bull fixture the other bench tests use. A
point is deleted from the version and the query is made at one of its pixels,
the way the track-at-pixel harness holds a point out.
"""

from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool import bench
from sfmtool._sfmtool.patches import ImagePyramidSet
from sfmtool._sfmtool.reconstruction import EditedReconstruction

from .test_bench_rust_bindings import (  # noqa: F401 (fixtures)
    descriptor_index,
    embedded,
    images,
    long_track_point,
)

ANCHOR_KEYS = {
    "source",
    "id",
    "position",
    "views",
    "query_pixel",
    "distance_px",
    "n_views",
    "max_reproj_px",
    "max_ray_angle_deg",
    "depth",
}


@pytest.fixture(scope="module")
def pyramids(embedded, images):  # noqa: F811
    return ImagePyramidSet(embedded, images)


@pytest.fixture
def query(embedded, long_track_point):  # noqa: F811
    """The longest track held out: the version without it, and its first image
    and pixel."""
    record = EditedReconstruction(embedded).point(long_track_point)
    image = int(record["image_indexes"][0])
    pixel = tuple(float(v) for v in record["keypoints_xy"][0])
    edited = EditedReconstruction(embedded)
    edited.delete_point(long_track_point)
    return edited, image, pixel


def test_the_points_near_the_pixel_are_found_without_the_held_out_one(
    pyramids,
    query,
    long_track_point,  # noqa: F811
):
    edited, image, pixel = query
    found = bench.nearby_points(edited, pyramids, image, pixel)

    assert 0 < len(found) <= 8
    for a in found:
        assert set(a) == ANCHOR_KEYS
        assert a["source"] == "tracks"
        assert a["id"] != long_track_point
        assert a["views"][0][0] == image
        assert a["query_pixel"] == a["views"][0][1:]
        assert a["n_views"] == len(a["views"]) >= 2
        assert a["max_reproj_px"] <= 2.0
        assert a["distance_px"] <= 40.0
        assert a["depth"] > 0
        # The point is the reconstruction's own.
        record = edited.point(a["id"])
        assert np.allclose(a["position"], record["position"])
    distances = [a["distance_px"] for a in found]
    assert distances == sorted(distances)

    # The options narrow it.
    few = bench.nearby_points(
        edited, pyramids, image, pixel, options={"max_points": 2, "radius_px": 40.0}
    )
    assert len(few) == min(2, len(found))
    with pytest.raises(ValueError, match="unknown option"):
        bench.nearby_points(edited, pyramids, image, pixel, options={"radius": 1.0})
    with pytest.raises(ValueError):
        bench.nearby_points(edited, pyramids, 99, pixel)


def sift_paths(embedded):  # noqa: F811
    from sfmtool.sift.file import get_sift_path_for_image

    workspace = Path(embedded.workspace_dir)
    return [str(get_sift_path_for_image(workspace / n)) for n in embedded.image_names]


@pytest.fixture(scope="module")
def sources(embedded, descriptor_index):  # noqa: F811
    from sfmtool._sfmtool.io import MatchesFile

    forest, _ = descriptor_index
    matches = sorted(Path(embedded.workspace_dir).glob("matches/*.matches"))
    assert matches, "the fixture workspace holds a clusters .matches"
    return bench.NearbyTrackSources(
        EditedReconstruction(embedded),
        forest=forest,
        matches=MatchesFile(str(matches[0])),
        sift=sift_paths(embedded),
    )


def test_the_clusters_near_the_pixel_are_vetted_by_triangulation(
    pyramids, query, sources
):
    edited, image, pixel = query
    assert sources.has_clusters
    found = bench.nearby_cluster_tracks(edited, pyramids, sources, image, pixel)

    assert found, "the fixture's clusters reach the pixel"
    for a in found:
        assert set(a) == ANCHOR_KEYS
        assert a["source"] == "clusters"
        assert a["views"][0][0] == image
        assert a["query_pixel"] == a["views"][0][1:]
        assert len({v[0] for v in a["views"]}) == a["n_views"] >= 2
        assert a["max_reproj_px"] <= 2.0
        assert a["distance_px"] <= 48.0
    assert len({a["id"] for a in found}) == len(found)

    # The stricter policy builds from the reference and the kept members.
    kept = bench.nearby_cluster_tracks(
        edited, pyramids, sources, image, pixel, options={"members": "kept"}
    )
    assert all(a["source"] == "clusters" and a["max_reproj_px"] <= 2.0 for a in kept)
    with pytest.raises(ValueError, match="kept|any"):
        bench.nearby_cluster_tracks(
            edited, pyramids, sources, image, pixel, options={"members": "all"}
        )


def test_the_keypoints_near_the_pixel_are_matched_along_their_rays(
    pyramids, query, sources
):
    edited, image, pixel = query
    assert sources.has_guided
    found = bench.guided_matches(edited, pyramids, sources, image, pixel)

    assert found, "the keypoints near the pixel match elsewhere"
    for a in found:
        assert set(a) == ANCHOR_KEYS
        assert a["source"] == "guided"
        assert a["views"][0][0] == image
        assert a["query_pixel"] == a["views"][0][1:]
        assert len({v[0] for v in a["views"]}) == a["n_views"] >= 2
        assert a["max_reproj_px"] <= 2.0
        assert a["distance_px"] <= 24.0
    assert len(found) <= 8

    # Asking for three views leaves out the keypoints matched in one image.
    three = bench.guided_matches(
        edited, pyramids, sources, image, pixel, options={"min_views": 3}
    )
    assert [a for a in found if a["n_views"] >= 3] == three


def test_the_sources_can_read_their_keypoints_from_the_sift_files(
    embedded,  # noqa: F811
    pyramids,
    query,
    sources,
):
    edited, image, pixel = query
    from sfmtool.sift.file import SiftReader

    paths = sift_paths(embedded)
    keypoints = []
    for path in paths:
        reader = SiftReader(path)
        xy, affine = reader.read_positions_and_shapes()
        reader.close()
        keypoints.append((np.asarray(xy, np.float32), np.asarray(affine, np.float32)))
    given = bench.NearbyTrackSources(
        EditedReconstruction(embedded), keypoints=keypoints, sift=paths
    )
    assert bench.guided_matches(
        edited, pyramids, given, image, pixel
    ) == bench.guided_matches(edited, pyramids, sources, image, pixel)
    with pytest.raises(ValueError, match="sift has 1 entries"):
        bench.NearbyTrackSources(EditedReconstruction(embedded), sift=paths[:1])


def test_the_constellation_carries_the_pixel_into_the_images_it_matches(
    pyramids, query, sources
):
    edited, image, pixel = query
    assert sources.has_constellation
    found = bench.constellation_seeds(edited, pyramids, sources, image, pixel)

    assert len(found) <= 1
    for a in found:
        assert set(a) == ANCHOR_KEYS
        assert a["source"] == "constellation"
        assert a["id"] is None
        assert a["views"][0] == [image, *pixel]
        assert a["query_pixel"] == list(pixel)
        assert a["distance_px"] == 0.0
        assert a["max_reproj_px"] <= 3.0

    # From the keypoints near the pixel as well: each extra candidate is named
    # by its keypoint's row and sits on it.
    wide = bench.constellation_seeds(
        edited,
        pyramids,
        sources,
        image,
        pixel,
        options={"at": "keypoints", "lateral_max": 4},
    )
    assert wide[: len(found)] == found
    for a in wide[len(found) :]:
        assert isinstance(a["id"], int)
        assert 1.0 < a["distance_px"] <= 24.0
    with pytest.raises(ValueError, match="pixel|keypoints"):
        bench.constellation_seeds(
            edited, pyramids, sources, image, pixel, options={"at": "nowhere"}
        )


def test_a_source_without_its_input_finds_nothing(embedded, pyramids, query):  # noqa: F811
    edited, image, pixel = query
    empty = bench.NearbyTrackSources(EditedReconstruction(embedded))
    assert not empty.has_clusters
    assert not empty.has_guided
    assert not empty.has_constellation
    assert bench.nearby_cluster_tracks(edited, pyramids, empty, image, pixel) == []
    assert bench.guided_matches(edited, pyramids, empty, image, pixel) == []
    assert bench.constellation_seeds(edited, pyramids, empty, image, pixel) == []
