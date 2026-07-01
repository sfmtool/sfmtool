# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfm inspect --strips`` (per-point patch-strip montage)."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool._inspect_strips import parse_point_specs
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main


def _hash8(sfmr_path: Path) -> str:
    return SfmrReconstruction.load(str(sfmr_path)).content_xxh128[:8].lower()


def _invoke(args: list[str]):
    return CliRunner().invoke(main, ["inspect", "--strips", *args])


class TestInspectStripsParsing:
    """parse_point_specs: id/range grammar, ordering, validation."""

    def test_range_and_index_ordered_deduped(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(str(seoul_bull_workspace))
        # Listed order is preserved; the overlapping single index is de-duped.
        result = parse_point_specs(recon, ["3-5", "10", "4"])
        assert result == [3, 4, 5, 10]

    def test_point_id_hash_match(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(str(seoul_bull_workspace))
        pid = f"pt3d_{recon.content_xxh128[:8].lower()}_7"
        assert parse_point_specs(recon, [pid]) == [7]

    def test_point_id_hash_mismatch_errors(self, seoul_bull_workspace):
        import click

        recon = SfmrReconstruction.load(str(seoul_bull_workspace))
        with pytest.raises(click.UsageError, match="not for this reconstruction"):
            parse_point_specs(recon, ["pt3d_deadbeef_7"])

    def test_out_of_range_errors(self, seoul_bull_workspace):
        import click

        recon = SfmrReconstruction.load(str(seoul_bull_workspace))
        with pytest.raises(click.UsageError, match="out of range"):
            parse_point_specs(recon, [str(recon.point_count)])

    def test_invalid_spec_errors(self, seoul_bull_workspace):
        import click

        recon = SfmrReconstruction.load(str(seoul_bull_workspace))
        with pytest.raises(click.UsageError, match="invalid point spec"):
            parse_point_specs(recon, ["not-a-spec"])


class TestInspectStripsCli:
    """End-to-end CLI rendering for both feature sources."""

    def test_sift_files_writes_montage(self, seoul_bull_workspace, tmp_path):
        out = tmp_path / "strips.png"
        result = _invoke([str(seoul_bull_workspace), "0-5", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "--strips: wrote" in result.output
        assert out.exists() and out.stat().st_size > 0

    def test_point_id_argument(self, seoul_bull_workspace, tmp_path):
        out = tmp_path / "strips.png"
        pid = f"pt3d_{_hash8(seoul_bull_workspace)}_12"
        result = _invoke([str(seoul_bull_workspace), pid, "0-2", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_embedded_patches_rendered_as_is(self, seoul_bull_workspace, tmp_path):
        # Convert to embedded_patches (reads .sift), saved back into the workspace
        # so its images still resolve, then render the stored data as is.
        recon = SfmrReconstruction.load(str(seoul_bull_workspace))
        emb = recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
        emb_path = seoul_bull_workspace.parent / "embedded.sfmr"
        emb.save(str(emb_path), operation="test")

        out = tmp_path / "strips.png"
        result = _invoke([str(emb_path), "0-5", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "embedded_patches" in result.output
        assert out.exists() and out.stat().st_size > 0

    def test_infinity_point_rendered(self, seoul_bull_workspace, tmp_path):
        # Park the most-observed point at infinity (w=0, direction along its
        # current position) and render it: the strip view must handle it via the
        # tangent-sphere infinity patch rather than skipping or crashing.
        import numpy as np

        recon = SfmrReconstruction.load(str(seoul_bull_workspace))
        counts = np.bincount(
            np.asarray(recon.track_point_indexes), minlength=recon.point_count
        )
        idx = int(np.argmax(counts))
        xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
        direction = xyzw[idx, :3] / np.linalg.norm(xyzw[idx, :3])
        xyzw[idx] = [*direction, 0.0]
        modified = recon.clone_with_changes(positions=xyzw)
        inf_path = seoul_bull_workspace.parent / "with_inf.sfmr"
        modified.save(str(inf_path), operation="test")
        assert np.asarray(SfmrReconstruction.load(str(inf_path)).point_is_at_infinity)[
            idx
        ]

        out = tmp_path / "strips.png"
        result = _invoke([str(inf_path), str(idx), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists() and out.stat().st_size > 0

    def test_default_output_path(self, seoul_bull_workspace, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _invoke([str(seoul_bull_workspace), "0-2"])
        assert result.exit_code == 0, result.output
        expected = tmp_path / f"{seoul_bull_workspace.stem}_strips.png"
        assert expected.exists()

    def test_hash_mismatch_cli_error(self, seoul_bull_workspace, tmp_path):
        result = _invoke([str(seoul_bull_workspace), "pt3d_deadbeef_1"])
        assert result.exit_code != 0
        assert "not for this reconstruction" in result.output

    def test_requires_sfmr_target(self, tmp_path):
        bogus = tmp_path / "nope.sift"
        bogus.write_text("x")
        result = _invoke([str(bogus), "0"])
        assert result.exit_code != 0
        assert "requires an existing .sfmr" in result.output

    def test_needs_a_point(self, seoul_bull_workspace):
        result = _invoke([str(seoul_bull_workspace)])
        assert result.exit_code != 0
        assert "needs at least one point" in result.output


def test_strips_options_rejected_without_flag(seoul_bull_sfmr_only):
    result = CliRunner().invoke(
        main, ["inspect", str(seoul_bull_sfmr_only), "-o", "x.png"]
    )
    assert result.exit_code != 0
    assert "only valid with --strips" in result.output


def test_normal_offsets_obliquity_geometry(seoul_bull_workspace):
    """The per-view obliquity offset is the tangential part of the unit vector
    toward the camera: 0 fronto-parallel, sin(theta) at angle theta, 1 grazing;
    None at infinity."""
    import math

    import numpy as np

    from sfmtool._solve_strips import _SolveStrips
    from sfmtool._sfmtool import OrientedPatch

    recon = SfmrReconstruction.load(str(seoul_bull_workspace))
    emb = recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
    emb_path = seoul_bull_workspace.parent / "embedded_geom.sfmr"
    emb.save(str(emb_path), operation="test")
    engine = _SolveStrips(
        SfmrReconstruction.load(str(emb_path)),
        seoul_bull_workspace.parent,
        patch=32,
        extent_factor=5.0,
    )

    # A finite point with >=3 observations, so we can place three synthetic views.
    counts = {p: len(v) for p, v in engine.obs.items() if engine._w(p) != 0.0}
    pid = max(counts, key=counts.get)
    obs = sorted(engine.obs[pid])[:3]
    assert len(obs) == 3

    # Patch at the origin; build the synthetic cameras from the patch's own axes
    # so the test is independent of the axis convention.
    patch = OrientedPatch.from_center_normal([0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1])
    n = np.asarray(patch.normal, np.float64)
    u = np.asarray(patch.u_axis, np.float64)
    engine.positions = np.zeros((engine.positions.shape[0], 3))
    engine.centers = [np.zeros(3) for _ in engine.centers]
    engine.centers[obs[0]] = 5.0 * n  # on the normal -> fronto-parallel
    engine.centers[obs[1]] = 5.0 * (
        math.cos(math.radians(30)) * n + math.sin(math.radians(30)) * u
    )  # 30 deg toward +u
    engine.centers[obs[2]] = 5.0 * u  # in-plane -> grazing

    offs = engine._normal_offsets(pid, patch, obs)
    assert math.hypot(*offs[0]) < 1e-6  # fronto-parallel -> centre
    assert abs(math.hypot(*offs[1]) - math.sin(math.radians(30))) < 1e-6
    assert offs[1][0] > 0  # dot lies toward +u (the camera's in-plane direction)
    assert abs(math.hypot(*offs[2]) - 1.0) < 1e-6  # grazing -> box edge

    # A point at infinity has no finite camera-to-surface vector -> no markers.
    engine._w = lambda _pid: 0.0
    assert engine._normal_offsets(pid, patch, obs) == [None, None, None]
