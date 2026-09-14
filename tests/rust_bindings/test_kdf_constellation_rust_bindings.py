# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the constellation query bindings on both forest classes.

One real SIFT extraction from the included Seoul Bull image gives the corpus a
second "image": the same descriptors under a known affine warp. Localizing a
patch of the first image must therefore find the second, recover that warp, and
give the same answer whether the forest is resident or file-backed.
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool.spatial import (
    KdForest,
    LazyKdForest,
    radius_for_feature_count,
    write_kdf,
)
from sfmtool.cli import main
from sfmtool.sift.file import SiftReader, get_sift_path_for_image

# The warp planted between image 0 and image 1, as a 2x3 row-major affine.
_WARP = np.array([[0.95, -0.2, 60.0], [0.2, 0.95, -25.0]], dtype=np.float64)
_KNOBS = {"k": 16, "max_leaf_checks": 256, "iterations": 400, "min_inliers": 6}


def _warp(positions: np.ndarray) -> np.ndarray:
    """Apply `_WARP` to an (N, 2) array of positions."""
    return (positions @ _WARP[:, :2].T + _WARP[:, 2]).astype(np.float32)


@pytest.fixture(scope="module")
def duplicated_capture(tmp_path_factory):
    """A two-image corpus, a `.kdf` over it, and the query image's `.sift`.

    Image 1 holds image 0's descriptors at warped positions, so every feature of
    a patch in image 0 has exactly one correct correspondence in image 1 and the
    fit has a right answer to find.
    """
    import shutil

    from tests.conftest import TEST_DATA_DIR

    workspace = tmp_path_factory.mktemp("constellation")
    image = workspace / "seoul_bull.jpg"
    shutil.copy(
        TEST_DATA_DIR
        / "images"
        / "seoul_bull_sculpture"
        / "seoul_bull_sculpture_01.jpg",
        image,
    )
    runner = CliRunner()
    result = runner.invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", str(workspace)]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ["sift", "--extract", str(image)])
    assert result.exit_code == 0, result.output

    sift_path = get_sift_path_for_image(image)
    reader = SiftReader(sift_path)
    descriptors = np.asarray(reader.read_descriptors())
    positions, affine_shapes = reader.read_positions_and_shapes()
    reader.close()
    positions = np.asarray(positions, dtype=np.float32)
    affine_shapes = np.asarray(affine_shapes, dtype=np.float32)
    n = len(descriptors)
    assert n > 50, "the fixture image should yield a usable number of features"

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
        "image_names": ["seoul_bull.jpg", "warped.jpg"],
        "feature_tool_hashes": [bytes(16), bytes(16)],
        "sift_content_hashes": [bytes(16), bytes([1]) * 16],
        "image_indexes": [0] * n + [1] * n,
        "image_feature_indexes": list(range(n)) * 2,
        "positions": np.vstack([positions, _warp(positions)]),
        "affine_shapes": np.vstack([affine_shapes, affine_shapes]),
    }
    forest = KdForest(
        np.vstack([descriptors, descriptors]), num_trees=4, leaf_size=16, seed=5
    )
    path = workspace / "capture.kdf"
    write_kdf(forest, str(path), descriptor_block_bytes=4096, sources=sources)
    return {
        "forest": forest,
        "lazy": LazyKdForest(str(path)),
        "path": path,
        "sift_path": sift_path,
        "sources": sources,
        "positions": positions,
        "workspace": workspace,
        "count": n,
    }


def _patch(capture, wanted=12):
    """A centre and radius holding at least `wanted` features of image 0."""
    positions = capture["positions"]
    centre = positions[len(positions) // 2]
    for radius in (30.0, 60.0, 120.0, 240.0, 480.0):
        inside = np.flatnonzero(np.hypot(*(positions - centre).T) <= radius)
        if len(inside) >= wanted:
            return centre, radius, inside
    raise AssertionError("no radius held enough features")


def _same(a, b):
    """Whether two constellation match lists are equal, arrays included."""
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if left["image_index"] != right["image_index"]:
            return False
        if left["inliers"] != right["inliers"]:
            return False
        if left["correspondences"] != right["correspondences"]:
            return False
        if not np.array_equal(left["affine"], right["affine"]):
            return False
        for key, value in left["inlier_correspondences"].items():
            if not np.array_equal(value, right["inlier_correspondences"][key]):
                return False
    return True


def test_the_dict_surface_names_the_rust_fields(duplicated_capture):
    """A localized patch reports the planted image, its warp and its inliers."""
    centre, radius, inside = _patch(duplicated_capture)
    found = duplicated_capture["lazy"].constellation_at_pixel(
        str(duplicated_capture["sift_path"]),
        (float(centre[0]), float(centre[1])),
        radius,
        image_index=0,
        **_KNOBS,
    )
    assert set(found) == {"feature_rows", "feature_ids", "matches"}
    # The query image is indexed first, so its `.sift` rows are its corpus IDs.
    assert np.array_equal(found["feature_rows"], inside.astype(np.uint32))
    assert np.array_equal(found["feature_ids"], inside.astype(np.uint32))

    matches = found["matches"]
    assert matches, "the warped image was not found"
    best = matches[0]
    assert set(best) == {
        "image_index",
        "affine",
        "inliers",
        "correspondences",
        "inlier_correspondences",
    }
    assert best["image_index"] == 1
    assert best["affine"].shape == (2, 3)
    np.testing.assert_allclose(best["affine"], _WARP, atol=1e-2)
    assert best["inliers"] >= len(inside) // 2
    assert best["inliers"] <= best["correspondences"]
    assert all(m["image_index"] != 0 for m in matches), "the query image matched itself"

    columns = best["inlier_correspondences"]
    assert set(columns) == {"query_index", "feature_id", "position", "affine_shape"}
    assert columns["query_index"].shape == (best["inliers"],)
    assert columns["position"].shape == (best["inliers"], 2)
    assert columns["affine_shape"].shape == (best["inliers"], 2, 2)
    # Every inlier is a feature of image 1, and its reported position is where
    # the `.sift` geometry says that feature sits.
    assert np.all(columns["feature_id"] >= duplicated_capture["count"])
    rows = columns["feature_id"] - duplicated_capture["count"]
    np.testing.assert_allclose(
        columns["position"],
        _warp(duplicated_capture["positions"])[rows],
        atol=1e-3,
    )


def test_the_two_forests_answer_identically(duplicated_capture):
    """The resident and file-backed paths are the same query, so the same answer."""
    centre, radius, inside = _patch(duplicated_capture)
    lazy = duplicated_capture["lazy"].constellation_at_pixel(
        str(duplicated_capture["sift_path"]),
        (float(centre[0]), float(centre[1])),
        radius,
        image_index=0,
        **_KNOBS,
    )
    eager = duplicated_capture["forest"].constellation_at_pixel(
        str(duplicated_capture["sift_path"]),
        (float(centre[0]), float(centre[1])),
        radius,
        duplicated_capture["sources"],
        image_index=0,
        **_KNOBS,
    )
    assert np.array_equal(eager["feature_rows"], lazy["feature_rows"])
    assert np.array_equal(eager["feature_ids"], lazy["feature_ids"])
    assert _same(eager["matches"], lazy["matches"])

    # The primitive underneath agrees too, whether it is handed IDs or vectors.
    positions = duplicated_capture["positions"][inside]
    ids = inside.astype(np.uint32).tolist()
    by_id = duplicated_capture["lazy"].constellation_query(
        positions, feature_ids=ids, image_index=0, **_KNOBS
    )
    by_vector = duplicated_capture["forest"].constellation_query(
        positions,
        duplicated_capture["sources"],
        descriptors=np.asarray(
            SiftReader(duplicated_capture["sift_path"]).read_descriptors()
        )[inside],
        image_index=0,
        **_KNOBS,
    )
    assert _same(by_id, lazy["matches"])
    assert _same(by_vector, lazy["matches"])


def test_bad_arguments_and_a_sourceless_file_are_value_errors(
    duplicated_capture, tmp_path
):
    """A corpus with no origins cannot say which image anything came from."""
    centre, radius, inside = _patch(duplicated_capture)
    positions = duplicated_capture["positions"][inside]
    lazy = duplicated_capture["lazy"]
    with pytest.raises(ValueError):
        lazy.constellation_query(positions, image_index=0, **_KNOBS)
    with pytest.raises(ValueError):
        lazy.constellation_query(
            positions,
            feature_ids=inside.astype(np.uint32).tolist(),
            descriptors=np.zeros((len(inside), 128), dtype=np.uint8),
            image_index=0,
        )

    plain = tmp_path / "plain.kdf"
    write_kdf(duplicated_capture["forest"], str(plain), descriptor_block_bytes=4096)
    with pytest.raises(ValueError):
        LazyKdForest(str(plain)).constellation_query(
            positions, feature_ids=inside.astype(np.uint32).tolist(), image_index=0
        )


def test_the_radius_rule_sizes_a_constellation(duplicated_capture):
    """The radius helper picks a disc that holds about the features asked for."""
    positions = duplicated_capture["positions"]
    metadata = SiftReader(duplicated_capture["sift_path"]).metadata
    width, height = metadata["image_width"], metadata["image_height"]

    radius = radius_for_feature_count(width, height, len(positions), 50)
    area = np.pi * radius * radius
    assert area == pytest.approx(50 / len(positions) * width * height, rel=1e-3)

    # Centred on a sample of keypoints, the disc holds tens of features rather
    # than a handful or most of the image: keypoints cluster on texture, so the
    # count runs above the fifty a uniform density would predict.
    counts = [
        np.count_nonzero(np.hypot(*(positions - centre).T) <= radius)
        for centre in positions[::37]
    ]
    assert 20 <= float(np.median(counts)) <= 200
    assert radius_for_feature_count(width, height, 0, 50) == 0.0
