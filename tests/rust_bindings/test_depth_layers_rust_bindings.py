# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``bench.depth_layers``.

The capture is the 17-image seoul_bull fixture the other bench tests use. The
anchors are the reconstruction's own points near a held-out point's pixel,
from ``bench.nearby_points``, with their ranges from ``bench.distance_range``,
the way the track-at-pixel harness makes them. The grouping, the support and
the ranking are checked against the harness's own Python
(``scripts/track_at_pixel/anchors.py``), which the binding ports.
"""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool import bench
from sfmtool._sfmtool.patches import ImagePyramidSet
from sfmtool._sfmtool.reconstruction import EditedReconstruction

from .test_bench_rust_bindings import embedded, images, long_track_point  # noqa: F401
from .test_distance_range_rust_bindings import _centers

LAYER_KEYS = {"range", "anchors", "nearest_px", "views"}
RANKED_KEYS = LAYER_KEYS | {"evidence", "score", "key", "rank", "confidence"}
EVIDENCE_KEYS = {
    "n_anchors",
    "n_independent",
    "n_images",
    "max_views",
    "max_ray_angle",
    "nearest_px",
    "at_pixel",
    "sources",
    "support",
    "photo",
    "votes",
    "votes_all",
    "photo_mid",
    "photo_both",
}


@pytest.fixture(scope="module")
def harness():
    """The harness's anchor finder, whose Python the binding ports."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "track_at_pixel"
    sys.path.insert(0, str(path))
    try:
        import anchors
    finally:
        sys.path.remove(str(path))
    return anchors


@pytest.fixture(scope="module")
def pyramids(embedded, images):  # noqa: F811
    return ImagePyramidSet(embedded, images)


@pytest.fixture
def query(embedded, pyramids, long_track_point):  # noqa: F811
    """The longest track held out: the version without it, its first image and
    pixel, and the points near that pixel as the harness's anchors, with their
    ranges and classes."""
    record = EditedReconstruction(embedded).point(long_track_point)
    image = int(record["image_indexes"][0])
    pixel = tuple(float(v) for v in record["keypoints_xy"][0])
    edited = EditedReconstruction(embedded)
    edited.delete_point(long_track_point)
    found = bench.nearby_points(edited, pyramids, image, pixel)
    assert len(found) >= 3
    center = _centers(embedded)[image]
    spread = bench.camera_spread(edited, pyramids)
    for a in found:
        distance = float(np.linalg.norm(np.asarray(a["position"]) - center))
        views = [(int(v[0]), (float(v[1]), float(v[2]))) for v in a["views"]]
        a["range"] = list(
            bench.distance_range(
                edited, pyramids, image, tuple(a["query_pixel"]), views, distance
            )
        )
        a.update(bench.classify_range(tuple(a["range"]), spread))
        del a["usable"]
    return edited, image, pixel, found


def test_the_layers_match_the_harness(harness, pyramids, query):
    edited, image, pixel, anchors = query
    # Two more readings of the same photographs, at half and a tenth the
    # distance, so there are layers to rank.
    for scale, px in ((0.5, 3.0), (0.1, 12.0)):
        extra = copy.deepcopy(anchors[1])
        extra["range"] = [scale * r for r in extra["range"]]
        extra["distance_px"] = px
        anchors = [*anchors, extra]
    got = bench.depth_layers(edited, pyramids, image, pixel, anchors)

    # The harness's support and grouping, over the same anchors.
    ref = copy.deepcopy(anchors)
    for a in ref:
        mine = {int(v[0]) for v in a["views"]}
        a["support"] = sum(
            1
            for b in ref
            if b is not a
            and harness._usable(a)
            and harness._usable(b)
            and harness._overlap(a["range"], b["range"])
            and not (mine <= {int(v[0]) for v in b["views"]})
            and not ({int(v[0]) for v in b["views"]} <= mine)
        )
    assert got["support"] == [a["support"] for a in ref]
    layers = harness._layers(ref)
    assert len(got["layers"]) == len(layers) == 3
    for L, M in zip(got["layers"], layers):
        assert set(L) == RANKED_KEYS
        assert set(L["evidence"]) == EVIDENCE_KEYS
        assert L["anchors"] == M["anchors"]
        assert L["range"] == M["range"]
        assert L["nearest_px"] == M["nearest_px"]
        assert L["views"] == M["views"]
        e = L["evidence"]
        members = [ref[k] for k in L["anchors"]]
        assert e["n_anchors"] == len(members)
        assert e["sources"] == ["tracks"]
        assert e["max_views"] == max(a["n_views"] for a in members)
        assert e["support"] == pytest.approx(
            sum(
                math.log2(1 + a["n_views"]) * math.exp(-a["distance_px"] / 20.0)
                for a in members
            ),
            rel=1e-12,
        )
        assert L["score"] == pytest.approx(0.5 * (e["photo"] + e["photo_both"]))
        assert 0 <= e["votes"] <= e["votes_all"] <= 16

    # The harness's evidence from the reads, each layer read at the harness's
    # distances with ``bench.read_patch_along_ray``.
    reads, mids = [], []
    for L in got["layers"]:
        near, far = L["range"]
        lo = 0.0 if not np.isfinite(far) else 1.0 / far
        hi = 1.0 / near if near > 0 else lo
        depths = [math.inf if v == 0 else 1.0 / v for v in np.linspace(lo, hi, 5)]
        read = bench.read_patch_along_ray(edited, pyramids, image, pixel, 8.0, depths)
        assert read["images"] == [i for i in range(17) if i != image]
        best = np.argmax(read["whole"], axis=0)  # the first of equal readings
        cols = np.arange(read["whole"].shape[1])
        reads.append(read["whole"][best, cols])
        mids.append(read["middle"][best, cols])
    reads, mids = np.asarray(reads), np.asarray(mids)
    for n, L in enumerate(got["layers"]):
        ref_ev = harness.layer_evidence(ref, L, reads, n, mids)
        for f, v in ref_ev.items():
            if isinstance(v, float):
                assert L["evidence"][f] == pytest.approx(v, rel=1e-12, abs=1e-12), f
            else:
                assert L["evidence"][f] == v, f

    # The harness's ranking, from the binding's evidence and score.
    ranked = [
        {"evidence": dict(L["evidence"]), "score": L["score"]} for L in got["layers"]
    ]
    harness._rank_layers(ranked)
    for L, M in zip(got["layers"], ranked):
        assert L["rank"] == M["rank"]
        assert L["key"] == pytest.approx(M["key"], rel=1e-12)
        assert L["confidence"] == pytest.approx(M["confidence"], rel=1e-12)
    assert sorted(L["rank"] for L in got["layers"]) == list(
        range(1, len(got["layers"]) + 1)
    )


def test_the_options_and_a_far_field_anchor(pyramids, query):
    edited, image, pixel, anchors = query
    far = dict(anchors[0])
    far.update(
        source="farfield",
        range=[1e6, math.inf],
        bounded=False,
        far=True,
        distance_px=0.0,
        views=[[image, pixel[0], pixel[1]], [(image + 1) % 17, 10.0, 10.0]],
        n_views=2,
    )
    got = bench.depth_layers(edited, pyramids, image, pixel, [*anchors, far])
    last = got["layers"][-1]
    assert last["anchors"] == [len(anchors)]
    assert last["range"] == [1e6, math.inf]
    assert last["evidence"]["sources"] == ["farfield"]
    assert last["evidence"]["at_pixel"] is True

    grouped = bench.depth_layers(
        edited, pyramids, image, pixel, anchors, options={"evidence": False}
    )
    assert all(set(L) == LAYER_KEYS for L in grouped["layers"])
    by_score = bench.depth_layers(
        edited,
        pyramids,
        image,
        pixel,
        anchors,
        options={"rank_by": "score", "radius_px": 8.0, "samples": 5},
    )
    scores = [L["score"] for L in by_score["layers"]]
    order = sorted(range(len(scores)), key=lambda n: -scores[n])
    assert [by_score["layers"][n]["rank"] for n in order] == list(
        range(1, len(scores) + 1)
    )

    with pytest.raises(ValueError, match="unknown option"):
        bench.depth_layers(edited, pyramids, image, pixel, anchors, options={"x": 1})
    with pytest.raises(ValueError, match="unknown layer rank"):
        bench.depth_layers(
            edited, pyramids, image, pixel, anchors, options={"rank_by": "votes"}
        )
    with pytest.raises(ValueError, match="unknown anchor source"):
        bench.depth_layers(
            edited, pyramids, image, pixel, [dict(anchors[0], source="sweep")]
        )
    with pytest.raises(ValueError, match="n_views"):
        bench.depth_layers(
            edited, pyramids, image, pixel, [dict(anchors[0], n_views=99)]
        )
    with pytest.raises(ValueError):
        bench.depth_layers(edited, pyramids, 99, pixel, anchors)
    assert bench.depth_layers(edited, pyramids, image, pixel, []) == {
        "support": [],
        "layers": [],
    }


def test_the_patch_read_along_a_ray(pyramids, query):
    edited, image, pixel, _ = query
    read = bench.read_patch_along_ray(
        edited, pyramids, image, pixel, 8.0, [math.inf, 50.0], [1, 2], samples=True
    )
    assert read["images"] == [1, 2]
    assert read["distances"] == [math.inf, 50.0]
    for key in ("whole", "middle"):
        assert read[key].shape == (2, 2)
        assert np.all((read[key] >= -1.0) & (read[key] <= 1.0 + 1e-12))
    assert read["centres"].shape == (2, 2, 2)
    assert read["values"].shape == (2, 2, 121)
    assert read["template"].shape == (121,)
    assert int(read["middle_mask"].sum()) == 25
    assert read["middle_std"] > 0

    # A patch that runs off its photograph is not read.
    assert (
        bench.read_patch_along_ray(edited, pyramids, image, (2.0, 2.0), 8.0, [10.0])
        is None
    )
    with pytest.raises(ValueError, match="not one of"):
        bench.read_patch_along_ray(edited, pyramids, image, pixel, 8.0, [10.0], [99])
