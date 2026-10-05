# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterByZnccSelfSimilarityRadiusTransform and the pass rule it
shares with the embed-patches cull."""

import numpy as np
import pytest

from sfmtool._sfmtool.patches import (
    DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS,
    zncc_self_similarity_parts_stack,
)
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import (
    FilterByZnccSelfSimilarityRadiusTransform,
    RefineKeypointsTransform,
)
from sfmtool.xform._filter_by_zncc_self_similarity_radius import (
    DEFAULT_MAX_ZNCC_SELF_SIMILARITY_RADIUS,
    points_passing_zncc_self_similarity_radius,
)

R = 12

FLAT, EDGE, TEXTURE, EMPTY = range(4)


def synthetic_bitmaps(n: int, resolution: int = R) -> np.ndarray:
    """An ``(n, R, R, 4)`` stack cycling through a flat patch, a straight edge, a
    random texture, and an empty row (no consensus), in that order."""
    rng = np.random.default_rng(5)
    out = np.zeros((n, resolution, resolution, 4), np.uint8)
    cols = np.arange(resolution)[None, :]
    for i in range(n):
        kind = i % 4
        if kind == FLAT:
            out[i, ..., :3] = 128
        elif kind == EDGE:
            out[i, ..., :3] = np.where(cols < resolution // 2, 50, 200)[..., None]
        elif kind == TEXTURE:
            out[i, ..., :3] = rng.integers(0, 256, (resolution, resolution, 3))
        if kind != EMPTY:
            out[i, ..., 3] = 255
    return out


@pytest.fixture(scope="module")
def embedded_with_bitmaps(seoul_bull_workspace_once) -> SfmrReconstruction:
    """An ``embedded_patches`` recon carrying per-point consensus bitmaps."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    return RefineKeypointsTransform(bitmaps=True, resolution=R, max_gn_steps=3).apply(
        recon
    )


def test_the_default_bar_is_the_member_gates():
    assert DEFAULT_MAX_ZNCC_SELF_SIMILARITY_RADIUS == 2.5
    assert DEFAULT_MAX_ZNCC_SELF_SIMILARITY_RADIUS == (
        DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS
    )
    assert FilterByZnccSelfSimilarityRadiusTransform().threshold == 2.5


def test_the_stack_reads_each_kind():
    out = zncc_self_similarity_parts_stack(synthetic_bitmaps(4))
    radius = np.asarray(out["radius"])
    assert radius[FLAT] == 3.0 and radius[EDGE] == 3.0
    assert radius[TEXTURE] < 1.0
    assert np.isnan(radius[EMPTY])
    assert list(out["covered"]) == [True, True, True, False]
    assert out["radius_grid"].shape == (4, 3, 3)
    assert out["ellipse_axes"].shape == (4, 2)
    assert out["ellipse_axes_is_at_least"].shape == (4, 2)
    assert out["radius_is_at_least"].shape == (4,)
    assert np.isinf(out["tolerance"][FLAT])
    # The edge slides along itself: its major axis runs straight down the
    # columns, along y.
    assert abs(out["ellipse_major_angle"][EDGE] - np.pi / 2) < 0.1


def test_the_stack_refuses_a_bad_shape():
    with pytest.raises(ValueError, match="square"):
        zncc_self_similarity_parts_stack(np.zeros((2, 12, 10, 4), np.uint8))
    with pytest.raises(ValueError, match=r"\(N, R, R, C\)"):
        zncc_self_similarity_parts_stack(np.zeros((12, 12, 4), np.uint8))
    with pytest.raises(ValueError, match="channels"):
        zncc_self_similarity_parts_stack(np.zeros((2, 12, 12, 5), np.uint8))


def test_pass_rule():
    bitmaps = synthetic_bitmaps(8)
    kinds = np.arange(8) % 4
    passes, radius = points_passing_zncc_self_similarity_radius(bitmaps, 2.5)
    # The default culls a flat and an edge patch, keeps a textured one, and
    # keeps a point with no consensus (no reading).
    assert list(passes) == list((kinds == TEXTURE) | (kinds == EMPTY))
    assert np.isnan(radius[kinds == EMPTY]).all()
    # 0 turns the cull off; a bar of 3 or more turns nothing out.
    for bar in (0.0, 3.0, 5.0):
        passes, _ = points_passing_zncc_self_similarity_radius(bitmaps, bar)
        assert passes.all(), bar


def test_invalid_threshold():
    with pytest.raises(ValueError, match="Threshold must be 0 or more"):
        FilterByZnccSelfSimilarityRadiusTransform(threshold=-1.0)
    with pytest.raises(ValueError, match="Threshold must be 0 or more"):
        FilterByZnccSelfSimilarityRadiusTransform(threshold=float("nan"))


def test_the_filter_culls_flat_and_edge_points(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    n = recon.point_count
    kinds = np.arange(n) % 4
    synthetic = recon.clone_with_changes(patch_bitmaps=synthetic_bitmaps(n))
    out = FilterByZnccSelfSimilarityRadiusTransform().apply(synthetic)
    keep = (kinds == TEXTURE) | (kinds == EMPTY)
    assert out.point_count == int(keep.sum())
    np.testing.assert_array_equal(
        np.asarray(out.positions), np.asarray(synthetic.positions)[keep]
    )


def test_zero_and_three_keep_every_point(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    synthetic = recon.clone_with_changes(
        patch_bitmaps=synthetic_bitmaps(recon.point_count)
    )
    for bar in (0.0, 3.0):
        out = FilterByZnccSelfSimilarityRadiusTransform(threshold=bar).apply(synthetic)
        assert out.point_count == recon.point_count


def test_the_filter_matches_the_pass_rule_on_real_bitmaps(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    passes, _ = points_passing_zncc_self_similarity_radius(recon.patch_bitmaps, 1.0)
    if passes.all() or not passes.any():
        pytest.skip("the fixture's bitmaps all read on one side of 1.0")
    out = FilterByZnccSelfSimilarityRadiusTransform(threshold=1.0).apply(recon)
    assert out.point_count == int(passes.sum())
    kept, radius = points_passing_zncc_self_similarity_radius(out.patch_bitmaps, 1.0)
    assert kept.all()
    assert np.all(radius[np.isfinite(radius)] <= 1.0)


def test_requires_patch_bitmaps(seoul_bull_workspace_once):
    """Filtering an embedded recon with no bitmaps is a clear error."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    assert recon.patch_bitmaps is None
    with pytest.raises(ValueError, match="patch bitmaps"):
        FilterByZnccSelfSimilarityRadiusTransform(threshold=1.0).apply(recon)


def test_description():
    desc = FilterByZnccSelfSimilarityRadiusTransform(threshold=1.5).description()
    assert "ZNCC self-similarity radius" in desc
    assert "1.50" in desc


def test_no_points_remain_raises(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    flat = np.zeros((recon.point_count, R, R, 4), np.uint8)
    flat[..., :3] = 90
    flat[..., 3] = 255
    with pytest.raises(ValueError, match="No points remain"):
        FilterByZnccSelfSimilarityRadiusTransform().apply(
            recon.clone_with_changes(patch_bitmaps=flat)
        )
