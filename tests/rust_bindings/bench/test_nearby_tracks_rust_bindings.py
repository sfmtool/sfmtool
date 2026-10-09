# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``bench.find_nearby_tracks``.

The capture is the 17-image seoul_bull fixture the other bench tests use. A
point is deleted from the version and the query is made at one of its pixels,
the way the track-at-pixel harness holds a point out. The combined operation
is checked against its pieces' own bindings, each of which is checked against
the harness's Python in its own test module.
"""

import re
from pathlib import PurePosixPath

import numpy as np
import pytest

from sfmtool import bench
from sfmtool.reconstruction import EditedReconstruction

from .test_bench_rust_bindings import (  # noqa: F401 (fixtures)
    sift_index,
    embedded,
    images,
    long_track_point,
)
from .test_nearby_sources_rust_bindings import (  # noqa: F401 (fixtures)
    pyramids,
    query,
    sources,
)

TRACK_KEYS = {
    "label",
    "source",
    "found",
    "layer",
    "rank",
    "confidence",
    "pixel",
    "distance_px",
    "range",
    "n_views",
    "point",
    "track",
}


def test_every_source_runs_and_the_layers_are_the_pieces(
    pyramids,  # noqa: F811
    query,  # noqa: F811
    sources,  # noqa: F811
):
    edited, image, pixel = query
    got = bench.find_nearby_tracks(
        edited,
        pyramids,
        sources,
        image,
        pixel,
        options={"stop": "never", "tracks.build": False},
    )

    # The sources ran in order, each finding what its own binding finds.
    stages = got["stages"]
    names = [s["source"] for s in stages]
    assert names[:4] == ["tracks", "clusters", "guided", "constellation"]
    assert got["report"]["stopped_after"] is None
    pieces = [
        bench.nearby_points(edited, pyramids, image, pixel),
        bench.nearby_cluster_tracks(edited, pyramids, sources, image, pixel),
        bench.guided_matches(edited, pyramids, sources, image, pixel),
        bench.constellation_seeds(edited, pyramids, sources, image, pixel),
    ]
    expected = [a for piece in pieces for a in piece]
    found = got["found"]
    for s, piece in zip(stages, pieces):
        assert s["found"] == len(piece)
    for a, b in zip(found, expected):
        assert {k: a[k] for k in b} == b

    # Each has its range and class, as the range bindings give them.
    spread = bench.camera_spread(edited, pyramids)
    for a in found[: len(expected)]:
        views = [(int(v[0]), (float(v[1]), float(v[2]))) for v in a["views"]]
        rng = bench.distance_range(
            edited, pyramids, image, tuple(a["query_pixel"]), views, a["distance"]
        )
        assert a["range"] == list(rng)
        c = bench.classify_range(tuple(a["range"]), spread)
        assert (a["bounded"], a["far"]) == (c["bounded"], c["far"])

    # The far-field readings, when the sweep ran, come after, as its binding
    # returns them.
    far = got["report"]["far_field"]
    if far is not None:
        assert names[4] == "farfield"
        sweep = bench.far_field_sweep(edited, pyramids, image, pixel)
        assert far["found"] == len(sweep) == len(found) - len(expected)
        for a, b in zip(found[len(expected) :], sweep):
            assert {k: a[k] for k in b} == b
    else:
        assert len(found) == len(expected)

    # The layers and the support are the depth layers' over everything found.
    layers = bench.depth_layers(edited, pyramids, image, pixel, found)
    assert got["layers"] == layers["layers"]
    assert [a["support"] for a in found] == layers["support"]
    assert names[-1] == "evidence"


def test_the_tracks_are_labelled_by_layer_rank_and_order(
    embedded,  # noqa: F811
    pyramids,  # noqa: F811
    query,  # noqa: F811
    sources,  # noqa: F811
):
    edited, image, pixel = query
    got = bench.find_nearby_tracks(edited, pyramids, sources, image, pixel)
    name = PurePosixPath(embedded.image_names[image]).stem
    group = f"{name}@{round(pixel[0])},{round(pixel[1])}"
    assert got["group_label"] == group

    tracks = got["tracks"]
    usable = [a for a in got["found"] if a["bounded"] or a["far"]]
    # A usable track whose built track repeats another's is a duplicate: it
    # names that track, has no label and is not among the tracks.
    duplicates = [a for a in usable if a["duplicate_of"] is not None]
    assert got["report"]["duplicates"] == len(duplicates)
    for a in duplicates:
        assert a["label"] is None
        b = got["found"][a["duplicate_of"]]
        assert b["duplicate_of"] is None and b["label"] is not None
    assert all(a["duplicate_of"] is None for a in got["found"] if a not in usable)
    assert tracks and len(tracks) == len(usable) - len(duplicates)
    pattern = re.compile(re.escape(group) + r" (\d+)([a-z]+)( pt (\d+))?$")
    keys = []
    for t in tracks:
        assert TRACK_KEYS <= set(t)
        m = pattern.match(t["label"])
        assert m, t["label"]
        assert int(m[1]) == t["rank"]
        keys.append((t["rank"], t["distance_px"]))
        a = got["found"][t["found"]]
        assert a["label"] == t["label"]
        if t["source"] == "points":
            # An existing point: named, not rebuilt.
            assert int(m[4]) == t["point"] == a["id"]
            assert t["track"] is None
        else:
            assert m[3] is None and t["point"] is None
            assert t["track"] is not None, t.get("error")
            assert t["track"].stage == "track"
    # By rank, then by distance from the pixel within the layer.
    assert keys == sorted(keys)
    assert pattern.match(tracks[0]["label"]).group(1, 2) == ("1", "a")

    # The caller's label replaces the group.
    named = bench.find_nearby_tracks(
        edited,
        pyramids,
        sources,
        image,
        pixel,
        label="here",
        options={"tracks.build": False},
    )
    assert named["group_label"] == "here"
    assert all(t["label"].startswith("here ") for t in named["tracks"])
    assert all(t["track"] is None for t in named["tracks"])


def test_commit_adds_every_new_track_as_a_point(
    pyramids,  # noqa: F811
    query,  # noqa: F811
    sources,  # noqa: F811
):
    edited, image, pixel = query
    # Only the sources that build new tracks, so there is something to commit.
    options = {"sources": "clusters+guided+constellation", "stop": "never"}
    before = edited.point_count
    version, got = bench.find_nearby_tracks(
        edited, pyramids, sources, image, pixel, options=options, commit=True
    )
    built = [t for t in got["tracks"] if t["track"] is not None]
    assert built, "the fixture gives new tracks near the pixel"
    committed = [t for t in built if "error" not in t]
    assert committed
    assert version.point_count == before + len(committed)
    points = [t["point"] for t in committed]
    assert len(set(points)) == len(points)
    for t in committed:
        record = version.point(t["point"])
        assert int(record["image_indexes"][0]) == image
    # The version it was given is left as it was.
    assert edited.point_count == before


def test_a_source_without_its_input_is_skipped_and_named(
    embedded,  # noqa: F811
    pyramids,  # noqa: F811
    query,  # noqa: F811
):
    edited, image, pixel = query
    empty = bench.NearbyTrackSources(EditedReconstruction(embedded))
    got = bench.find_nearby_tracks(
        edited, pyramids, empty, image, pixel, options={"stop": "never"}
    )
    report = got["report"]["sources"]
    assert [s["source"] for s in report] == [
        "points",
        "clusters",
        "guided",
        "constellation",
    ]
    assert report[0].get("skipped") is None
    assert [s.get("skipped") for s in report[1:]] == [
        "clusters",
        "descriptors",
        "SIFT index",
    ]
    assert all(s["found"] == 0 for s in report[1:])
    assert all(a["source"] in ("tracks", "farfield") for a in got["found"])


def test_the_options_and_the_query_are_checked(
    pyramids,  # noqa: F811
    query,  # noqa: F811
    sources,  # noqa: F811
):
    edited, image, pixel = query
    with pytest.raises(ValueError, match="unknown option"):
        bench.find_nearby_tracks(
            edited, pyramids, sources, image, pixel, options={"radius": 1.0}
        )
    with pytest.raises(ValueError, match="unknown option"):
        bench.find_nearby_tracks(
            edited, pyramids, sources, image, pixel, options={"points.radius": 1.0}
        )
    with pytest.raises(ValueError, match="not a matching source"):
        bench.find_nearby_tracks(
            edited, pyramids, sources, image, pixel, options={"sources": "farfield"}
        )
    with pytest.raises(ValueError, match="enough|never"):
        bench.find_nearby_tracks(
            edited, pyramids, sources, image, pixel, options={"stop": "sometimes"}
        )
    with pytest.raises(ValueError, match="not one of"):
        bench.find_nearby_tracks(edited, pyramids, sources, 99, pixel)
    with pytest.raises(ValueError, match="not on the"):
        bench.find_nearby_tracks(edited, pyramids, sources, image, (-5.0, 1.0))
    with pytest.raises(ValueError, match="control character"):
        bench.find_nearby_tracks(edited, pyramids, sources, image, pixel, label="a\nb")
    with pytest.raises(ValueError, match="something in it"):
        bench.find_nearby_tracks(edited, pyramids, sources, image, pixel, label="  ")

    # Never running the far-field sweep and no sources finds nothing.
    got = bench.find_nearby_tracks(
        edited,
        pyramids,
        sources,
        image,
        pixel,
        options={"sources": [], "far_field_when": "never"},
    )
    assert got["found"] == got["tracks"] == got["layers"] == []
    assert got["report"]["far_field"] is None
    assert np.isfinite(got["report"]["layers_seconds"])
