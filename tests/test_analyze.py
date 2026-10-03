# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm analyze` CLI command."""

from pathlib import Path

import numpy as np

from click.testing import CliRunner

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.cli import main


def test_analyze_requires_mode(seoul_bull_workspace):
    """analyze with no mode flag is rejected."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", sfmr_path])
    assert result.exit_code != 0
    assert "analysis mode" in result.output


def test_analyze_coviz(seoul_bull_workspace):
    """--coviz prints covisibility graph."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--coviz", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "ovisibility" in result.output


def test_analyze_images(seoul_bull_workspace):
    """--images prints per-image connectivity table."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--images", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "seoul_bull_sculpture" in result.output


def test_analyze_metrics(seoul_bull_workspace):
    """--metrics prints per-image metrics table."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--metrics", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "MeanErr" in result.output
    assert "seoul_bull_sculpture" in result.output


def test_analyze_metrics_with_range(seoul_bull_workspace):
    """--metrics --range filters to subset of images."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(
        main, ["analyze", "--metrics", "--range", "1-5", sfmr_path]
    )
    assert result.exit_code == 0, result.output
    assert "5 of 17 images" in result.output


def test_analyze_depth_reliability(seoul_bull_workspace):
    """--depth-reliability prints the inverse-depth z-score report."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--depth-reliability", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "Depth reliability" in result.output
    assert "Inverse-depth z" in result.output


def test_analyze_non_sfmr_file(tmp_path: Path):
    """Passing a non-.sfmr file raises an error."""
    p = tmp_path / "input.txt"
    p.touch()
    result = CliRunner().invoke(main, ["analyze", "--coviz", str(p)])
    assert result.exit_code != 0
    assert ".sfmr" in result.output


def test_analyze_mutually_exclusive_flags(seoul_bull_workspace):
    """Multiple mode flags at once are rejected."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--coviz", "--metrics", sfmr_path])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_analyze_range_without_metrics(seoul_bull_workspace):
    """--range without --metrics is rejected."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(
        main, ["analyze", "--coviz", "--range", "1-5", sfmr_path]
    )
    assert result.exit_code != 0
    assert "--range" in result.output


# ── point-or-bearing test in --depth-reliability ─────────────────────────────


def _all_points_at_infinity(sfmr_path: Path, out_path: Path) -> Path:
    """Save a copy of the reconstruction with every point turned into a bearing."""
    recon = SfmrReconstruction.load(str(sfmr_path))
    xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    xyzw[:, :3] /= np.linalg.norm(xyzw[:, :3], axis=1)[:, None]
    xyzw[:, 3] = 0.0
    recon.clone_with_changes(positions=xyzw).save(str(out_path))
    return out_path


def test_analyze_depth_reliability_point_or_bearing(seoul_bull_ground_truth_sfmr):
    """The embedded_patches ground truth reports the measured noise level, both
    groups, and agrees with its stored representation at the measured noise."""
    result = CliRunner().invoke(
        main, ["analyze", "--depth-reliability", str(seoul_bull_ground_truth_sfmr)]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Point or bearing (likelihood-ratio test on the depth):" in out
    # Four mismatched keypoints (7 to 16 px) are left out of the measure.
    assert (
        "Noise level: 0.4677 px, measured over 1,229 observations of finite "
        "points, 4 excluded as outliers" in out
    )
    assert "Threshold: 25" in out
    assert "Finite points: 266 scored" in out
    assert "Points at infinity: 14 scored" in out
    assert (
        "Bearing verdict: 14 (100.0%), 12 with the bearing cost under the threshold"
        in out
    )
    assert "Finite points the test calls bearings: 0" in out
    assert "Points at infinity the test calls finite: 0" in out
    # One camera: no per-camera breakdown.
    assert "Camera 0:" not in out


def test_analyze_depth_reliability_sigma_override(seoul_bull_ground_truth_sfmr):
    """At a smaller given noise level, bearing 188 scores above the threshold and
    is listed as a point at infinity the test calls finite."""
    result = CliRunner().invoke(
        main,
        [
            "analyze",
            "--depth-reliability",
            "--sigma-px",
            "0.216",
            str(seoul_bull_ground_truth_sfmr),
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Noise level: 0.216 px given (measured 0.4677 px" in out
    assert "Points at infinity the test calls finite: 1" in out
    row = next(line for line in out.splitlines() if line.strip().startswith("pt3d_"))
    fields = row.split()
    assert fields[0].endswith("_188")
    assert int(fields[1]) == 5  # views
    assert fields[2:] == ["32.5", "27.5", "32.5", "1.04", "414.9"]
    # score, midpoint bound, likelihood ratio, the z rule's z (under its
    # cutoff of 4, so that rule calls it a bearing) and the fitted distance.
    # Reclassification at the given noise level would store it finite.
    assert "Reclassification would promote 1 and demote 0" in out


def test_analyze_depth_reliability_threshold_override(seoul_bull_ground_truth_sfmr):
    """A threshold above some finite points' scores lists them, weakest first."""
    result = CliRunner().invoke(
        main,
        [
            "analyze",
            "--depth-reliability",
            "--depth-likelihood-ratio-threshold",
            "1000",
            str(seoul_bull_ground_truth_sfmr),
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Threshold: 1000" in out
    # At the measured 0.4677 px the scores are higher than at the 0.646 px the
    # RMS gave before outliers were left out, so fewer fall under 1000.
    assert "Finite points the test calls bearings: 24" in out
    rows = [
        line.split() for line in out.splitlines() if line.strip().startswith("pt3d_")
    ]
    assert len(rows) == 20
    scores = [float(r[2].replace(",", "")) for r in rows]
    assert scores == sorted(scores)
    assert all(s < 1000 for s in scores)
    assert "... and 4 more" in out
    # Reclassification decides at the default threshold, under which no
    # stored point disagrees.
    assert "Reclassification would promote 0 and demote 0" in out
    assert "(it decides at the threshold 25, not 1000)" in out


def test_analyze_depth_reliability_point_or_bearing_sift_files(seoul_bull_workspace):
    """A sift_files reconstruction reports the noise it measures and its verdicts."""
    recon = SfmrReconstruction.load(str(seoul_bull_workspace))
    noise = recon.reprojection_noise()
    scores = recon.point_or_bearing_scores()
    at_infinity = np.asarray(recon.positions_xyzw)[:, 3] == 0.0
    n_finite = int((~at_infinity & scores["scored"]).sum())
    n_demoted = int((~at_infinity & scores["scored"] & ~scores["is_finite"]).sum())

    result = CliRunner().invoke(
        main, ["analyze", "--depth-reliability", str(seoul_bull_workspace)]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert (
        f"Noise level: {noise['sigma_px']:.4g} px, measured over "
        f"{noise['observation_count']:,} observations of finite points, "
        f"{noise['outlier_count']:,} excluded as outliers"
    ) in out
    assert f"Finite points: {n_finite:,} scored" in out
    assert f"Finite points the test calls bearings: {n_demoted:,}" in out


def test_analyze_depth_reliability_z_at_fitted_point(
    seoul_bull_ground_truth_sfmr,
):
    """A point at infinity's z is read at its fitted point with its error there,
    not with the large error it carries as a bearing."""
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    stored_z = np.asarray(recon.triangulation_diagnostics()["inverse_depth_z"])
    xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    errors = np.asarray(recon.errors, dtype=np.float32).copy()
    # Stored as bearings with a 5 px error: at a 5 px noise the rule's z would
    # be about a fifth of the finite point's.
    xyzw[:2, :3] /= np.linalg.norm(xyzw[:2, :3], axis=1)[:, None]
    xyzw[:2, 3] = 0.0
    errors[:2] = 5.0
    path = seoul_bull_ground_truth_sfmr.parent / "two_bearings.sfmr"
    recon.clone_with_changes(positions=xyzw, errors=errors).save(str(path))

    result = CliRunner().invoke(main, ["analyze", "--depth-reliability", str(path)])
    assert result.exit_code == 0, result.output
    rows = {
        int(line.split()[0].rsplit("_", 1)[1]): line.split()
        for line in result.output.splitlines()
        if line.strip().startswith("pt3d_")
    }
    assert sorted(rows) == [0, 1]
    for index, fields in rows.items():
        z = float(fields[5])
        assert abs(z - stored_z[index]) < 0.1 * stored_z[index], (index, z)


def test_analyze_depth_reliability_missing_sift(seoul_bull_sfmr_only):
    """A sift_files reconstruction without its .sift files or inline keypoints
    names the file it could not read."""
    recon = SfmrReconstruction.load(str(seoul_bull_sfmr_only))
    path = seoul_bull_sfmr_only.parent / "no_keypoints.sfmr"
    recon.clone_with_changes(keypoints_xy=None).save(str(path))
    result = CliRunner().invoke(main, ["analyze", "--depth-reliability", str(path)])
    assert result.exit_code == 0, result.output
    line = next(
        x for x in result.output.splitlines() if x.strip().startswith("Unavailable:")
    )
    assert "the reprojection noise could not be measured: cannot read " in line
    assert line.count(".sift") == 1


def test_analyze_point_or_bearing_flags_reject_non_finite(
    seoul_bull_ground_truth_sfmr,
):
    """NaN and infinity are usage errors for both overrides."""
    path = str(seoul_bull_ground_truth_sfmr)
    for flag in ("--sigma-px", "--depth-likelihood-ratio-threshold"):
        for value in ("nan", "inf"):
            result = CliRunner().invoke(
                main, ["analyze", "--depth-reliability", flag, value, path]
            )
            assert result.exit_code == 2, (flag, value, result.output)
            assert "not a finite number" in result.output


def test_analyze_depth_reliability_no_finite_points(
    seoul_bull_ground_truth_sfmr,
):
    """With no finite point there is no measured noise level; a given one still scores."""
    path = _all_points_at_infinity(
        seoul_bull_ground_truth_sfmr,
        seoul_bull_ground_truth_sfmr.parent / "all_at_infinity.sfmr",
    )
    result = CliRunner().invoke(main, ["analyze", "--depth-reliability", str(path)])
    assert result.exit_code == 0, result.output
    assert "Unavailable: the reconstruction has no observation of a finite point" in (
        result.output
    )

    result = CliRunner().invoke(
        main, ["analyze", "--depth-reliability", "--sigma-px", "0.5", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert "Noise level: 0.5 px given (none measured)" in result.output
    assert "Finite points: 0 scored" in result.output
    assert "Points at infinity the test calls finite:" in result.output


def test_analyze_point_or_bearing_flags_need_depth_reliability(
    seoul_bull_ground_truth_sfmr,
):
    """--sigma-px and --depth-likelihood-ratio-threshold need --depth-reliability."""
    path = str(seoul_bull_ground_truth_sfmr)
    for flag, value in (
        ("--sigma-px", "0.5"),
        ("--depth-likelihood-ratio-threshold", "10"),
    ):
        result = CliRunner().invoke(main, ["analyze", "--coviz", flag, value, path])
        assert result.exit_code != 0
        assert flag in result.output
