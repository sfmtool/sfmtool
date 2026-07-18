# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for --scale-by-measurements Point ID resolution.

Covers the two behaviors that were previously documented-but-broken:
  * most-common point selection when a source point's observations disagree
    (``_resolve_point_cross_recon``), and
  * the workspace hash-prefix search that locates the source ``.sfmr`` when the
    measurements file omits the ``sfmr`` field (``_load_source`` reusing
    ``find_sfmr_by_content_hash``).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.geometry import Se3Transform
from sfmtool._workspace import find_sfmr_by_content_hash
from sfmtool.xform._scale_by_measurements import (
    ScaleByMeasurementsTransform,
    _resolve_point_cross_recon,
)


def _fake_source(image_names, tracks):
    """Build a stand-in source reconstruction for the cross-recon resolver.

    ``tracks`` is a list of (point_index, image_index, feature_index) tuples.
    Only the four attributes the resolver reads are populated.
    """
    pt = np.array([t[0] for t in tracks], dtype=np.uint32)
    img = np.array([t[1] for t in tracks], dtype=np.uint32)
    feat = np.array([t[2] for t in tracks], dtype=np.uint32)
    return SimpleNamespace(
        track_point_indexes=pt,
        track_image_indexes=img,
        track_feature_indexes=feat,
        image_names=list(image_names),
    )


class TestResolvePointCrossRecon:
    def test_most_common_wins_on_disagreement(self):
        # Source point 5 is seen in three images; two of the matching input
        # observations point at input point 100, one at 200 -> pick 100.
        source = _fake_source(
            ["a.jpg", "b.jpg", "c.jpg"],
            [(5, 0, 10), (5, 1, 11), (5, 2, 12), (9, 0, 99)],
        )
        input_name_to_idx = {"a.jpg": 0, "b.jpg": 1, "c.jpg": 2}
        input_obs_index = {(0, 10): 100, (1, 11): 100, (2, 12): 200}

        pt_idx, via = _resolve_point_cross_recon(
            "pt3d_deadbeef_5", 5, source, input_name_to_idx, input_obs_index
        )
        assert pt_idx == 100
        assert "feat #" in via

    def test_tie_breaks_to_lowest_index(self):
        # One vote each for 200 and 100 -> deterministic tie-break to 100.
        source = _fake_source(
            ["a.jpg", "b.jpg"],
            [(5, 0, 10), (5, 1, 11)],
        )
        input_name_to_idx = {"a.jpg": 0, "b.jpg": 1}
        input_obs_index = {(0, 10): 200, (1, 11): 100}

        pt_idx, _ = _resolve_point_cross_recon(
            "pt3d_deadbeef_5", 5, source, input_name_to_idx, input_obs_index
        )
        assert pt_idx == 100

    def test_no_matching_observations_raises(self):
        source = _fake_source(["a.jpg"], [(5, 0, 10)])
        with pytest.raises(ValueError, match="Could not resolve"):
            _resolve_point_cross_recon("pt3d_deadbeef_5", 5, source, {"a.jpg": 0}, {})


def _pick_separated_pair(positions):
    """Two point indices whose reconstruction distance is comfortably nonzero."""
    i, j = 0, len(positions) - 1
    assert float(np.linalg.norm(positions[i] - positions[j])) > 1e-3
    return i, j


class TestWorkspaceSearchFallback:
    def test_finds_source_by_hash_prefix(self, seoul_bull_sfmr_only):
        """find_sfmr_by_content_hash locates a saved recon by its prefix."""
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        prefix = recon.content_xxh128[:8].lower()

        found = find_sfmr_by_content_hash(seoul_bull_sfmr_only.parent, prefix)
        assert found is not None
        assert SfmrReconstruction.load(found).content_xxh128 == recon.content_xxh128

        assert (
            find_sfmr_by_content_hash(seoul_bull_sfmr_only.parent, "ffffffff") is None
        )

    def test_apply_resolves_source_via_workspace(self, seoul_bull_sfmr_only, tmp_path):
        """A measurements file with no ``sfmr`` field resolves via the search.

        The input reconstruction is a scaled copy of the source (different
        content hash, identical tracks), so Point IDs carrying the source hash
        take the cross-reconstruction path and the source is located by scanning
        the workspace.
        """
        source = SfmrReconstruction.load(seoul_bull_sfmr_only)

        # Workspace holds both the source and a half-scale input; the yaml lives
        # alongside them so the search base resolves to this directory.
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / ".sfm-workspace.json").write_text("{}")
        source_path = workspace / "source.sfmr"
        source.save(source_path, operation="test")
        scaled = Se3Transform(scale=0.5) @ source
        input_path = workspace / "input.sfmr"
        scaled.save(input_path, operation="test")

        # Saving recomputes the content hash, so read the prefix the search will
        # match from the file on disk, not from the pre-save object.
        source = SfmrReconstruction.load(source_path)
        source_prefix = source.content_xxh128[:8].lower()
        input_prefix = SfmrReconstruction.load(input_path).content_xxh128[:8].lower()
        # Guard the test's premise: the source must differ from the input so the
        # cross-reconstruction search path is exercised, not the same-recon path.
        assert source_prefix != input_prefix
        positions = source.positions
        i, j = _pick_separated_pair(positions)
        source_dist = float(np.linalg.norm(positions[i] - positions[j]))

        real_mm = 100.0
        yaml_path = workspace / "measurements.yaml"
        yaml_path.write_text(
            "unit: mm\n"
            "measurements:\n"
            f"  - point_a: pt3d_{source_prefix}_{i}\n"
            f"    point_b: pt3d_{source_prefix}_{j}\n"
            f"    distance: {real_mm}\n"
        )

        transform = ScaleByMeasurementsTransform(yaml_path)
        out = transform.apply(SfmrReconstruction.load(input_path))

        # recon distance is measured in the (half-scale) input frame.
        expected_scale = real_mm / (0.5 * source_dist)
        assert transform._scale_factor == pytest.approx(expected_scale, rel=1e-6)
        assert np.allclose(
            out.positions, scaled.positions * expected_scale, rtol=1e-5, atol=1e-5
        )
        assert out.world_space_unit == "mm"

    def test_missing_source_raises_helpful_error(self, seoul_bull_sfmr_only, tmp_path):
        """A hash prefix present nowhere in the workspace errors clearly."""
        source = SfmrReconstruction.load(seoul_bull_sfmr_only)
        i, j = _pick_separated_pair(source.positions)

        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / ".sfm-workspace.json").write_text("{}")
        input_path = workspace / "input.sfmr"
        source.save(input_path, operation="test")

        yaml_path = workspace / "measurements.yaml"
        yaml_path.write_text(
            "unit: mm\n"
            "measurements:\n"
            f"  - point_a: pt3d_00000000_{i}\n"
            f"    point_b: pt3d_00000000_{j}\n"
            "    distance: 100.0\n"
        )

        transform = ScaleByMeasurementsTransform(yaml_path)
        with pytest.raises(ValueError, match="no .sfmr under"):
            transform.apply(SfmrReconstruction.load(input_path))
