# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-image metrics analysis module."""

import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.analyze.metrics import _compute_per_image_metrics, print_metrics_analysis


@pytest.fixture
def rust_recon(seoul_bull_workspace):
    return SfmrReconstruction.load(seoul_bull_workspace)


@pytest.fixture
def per_image(rust_recon):
    return _compute_per_image_metrics(rust_recon)


class TestComputePerImageMetrics:
    def test_returns_one_entry_per_image(self, per_image, rust_recon):
        assert len(per_image) == 17
        for i, entry in enumerate(per_image):
            assert entry["image_index"] == i
            assert entry["image_name"] == rust_recon.image_names[i]

    def test_all_images_have_observations(self, per_image):
        # The fewest any image of the fixture holds is 95.
        for entry in per_image:
            assert entry["observation_count"] >= 75

    def test_mean_errors_in_expected_range(self, per_image):
        for entry in per_image:
            assert 0.1 < entry["mean_error"] < 2.0

    def test_median_and_mean_le_max(self, per_image):
        # Both the median and the mean of a set of errors lie at or below its
        # max. (There is no fixed median-vs-mean ordering: median <= mean holds
        # only for a right-skewed distribution, and per-image reprojection errors
        # are often near-symmetric, so asserting it is flaky.)
        for entry in per_image:
            assert entry["median_error"] <= entry["max_error"] + 1e-9
            assert entry["mean_error"] <= entry["max_error"] + 1e-9

    def test_max_errors_above_one_pixel(self, per_image):
        for entry in per_image:
            assert entry["max_error"] > 1.0

    def test_mean_track_length_matches_tracks(self, per_image, rust_recon):
        # Each image's mean track length is the mean observation count of the
        # points it sees, counted here straight from the track arrays.
        track_point_indexes = rust_recon.track_point_indexes
        lengths = np.bincount(track_point_indexes, minlength=rust_recon.point_count)
        for entry in per_image:
            in_image = rust_recon.track_image_indexes == entry["image_index"]
            expected = lengths[track_point_indexes[in_image]].mean()
            assert entry["mean_track_length"] == pytest.approx(expected)
            assert 2.0 <= entry["mean_track_length"] <= rust_recon.image_count


class TestPrintMetricsAnalysis:
    def test_header_and_table(self, seoul_bull_workspace, capsys):
        print_metrics_analysis(seoul_bull_workspace, recon_name="test.sfmr")
        captured = capsys.readouterr()

        assert "Per-image metrics analysis for: test.sfmr" in captured.out
        assert "17 images" in captured.out
        assert "observations" in captured.out

        for col in ("MeanErr", "MedErr", "MaxErr", "MeanTL"):
            assert col in captured.out

        lines = [
            line for line in captured.out.splitlines() if "seoul_bull_sculpture" in line
        ]
        assert len(lines) == 17

        assert "2x reconstruction median" in captured.out
        assert "1.5x reconstruction median" in captured.out
        assert "no observations" in captured.out

    def test_sorted_descending_by_mean_error(self, seoul_bull_workspace, capsys):
        print_metrics_analysis(seoul_bull_workspace)
        captured = capsys.readouterr()

        lines = [
            line for line in captured.out.splitlines() if "seoul_bull_sculpture" in line
        ]
        errors = []
        for line in lines:
            parts = line.split()
            for part in parts:
                try:
                    val = float(part)
                    if "." in part:
                        errors.append(val)
                        break
                except ValueError:
                    continue

        assert errors == sorted(errors, reverse=True)

    def test_recon_name(self, seoul_bull_workspace, capsys):
        print_metrics_analysis(seoul_bull_workspace, recon_name="custom.sfmr")
        assert "custom.sfmr" in capsys.readouterr().out

        print_metrics_analysis(seoul_bull_workspace)
        assert seoul_bull_workspace.name in capsys.readouterr().out

    def test_range_filter(self, seoul_bull_workspace, capsys):
        print_metrics_analysis(seoul_bull_workspace, range_expr="1-5")
        captured = capsys.readouterr()

        lines = [
            line for line in captured.out.splitlines() if "seoul_bull_sculpture" in line
        ]
        assert len(lines) == 5
        assert "Range filter: 1-5 (5 of 17 images)" in captured.out


class TestEmbeddedPatchesMetrics:
    """An ``embedded_patches`` file has no ``.sift`` feature indexes; its
    observed pixels are the inline ``keypoints_xy`` column. The per-image
    errors must still cover every observation and come out finite."""

    @pytest.fixture
    def embedded_recon(self, seoul_bull_ground_truth_sfmr):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        assert recon.feature_source == "embedded_patches"
        assert recon.track_feature_indexes is None
        return recon

    def test_one_error_per_observation(self, embedded_recon):
        for img_idx in range(embedded_recon.image_count):
            obs_data = np.asarray(
                embedded_recon.compute_observation_reprojection_errors(img_idx)
            )
            rows = np.flatnonzero(embedded_recon.track_image_indexes == img_idx)
            # Column 0 is the observation's row in the track arrays.
            np.testing.assert_array_equal(obs_data[:, 0].astype(np.int64), rows)
            assert np.isfinite(obs_data[:, 1]).all()

    def test_errors_match_independent_reprojection(self, embedded_recon):
        # Reproject each observation's point through its camera here, from the
        # public arrays, and compare against the inline keypoint.
        quats = np.asarray(embedded_recon.quaternions_wxyz)
        trans = np.asarray(embedded_recon.translations)
        xyzw = np.asarray(embedded_recon.positions_xyzw)
        keypoints = np.asarray(embedded_recon.keypoints_xy, dtype=np.float64)
        cameras = embedded_recon.cameras
        camera_indexes = np.asarray(embedded_recon.camera_indexes)
        image_of_obs = np.asarray(embedded_recon.track_image_indexes)
        point_of_obs = np.asarray(embedded_recon.track_point_indexes)

        for img_idx in range(embedded_recon.image_count):
            rows = np.flatnonzero(image_of_obs == img_idx)
            pts = xyzw[point_of_obs[rows]]
            q = np.repeat(quats[img_idx : img_idx + 1], len(rows), axis=0)
            w = q[:, :1]
            u = q[:, 1:]
            t = 2.0 * np.cross(u, pts[:, :3])
            cam_pt = pts[:, :3] + w * t + np.cross(u, t)
            # A point at infinity (w = 0) projects its bearing with no translation.
            cam_pt = cam_pt + pts[:, 3:4] * trans[img_idx]
            rays = cam_pt / np.linalg.norm(cam_pt, axis=1, keepdims=True)
            camera = cameras[camera_indexes[img_idx]]
            pred = np.asarray(camera.ray_to_pixel_batch(np.ascontiguousarray(rays)))
            expected = np.linalg.norm(pred - keypoints[rows], axis=1)

            got = np.asarray(
                embedded_recon.compute_observation_reprojection_errors(img_idx)
            )[:, 1]
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_per_image_metrics_are_finite(self, embedded_recon):
        per_image = _compute_per_image_metrics(embedded_recon)
        assert len(per_image) == embedded_recon.image_count
        for entry in per_image:
            assert entry["observation_count"] > 0
            assert len(entry["errors"]) == entry["observation_count"]
            for key in ("mean_error", "median_error", "max_error"):
                assert np.isfinite(entry[key]), (entry["image_name"], key)
            assert entry["median_error"] <= entry["max_error"]

    def test_print_shows_no_nan(self, seoul_bull_ground_truth_sfmr, capsys):
        print_metrics_analysis(seoul_bull_ground_truth_sfmr)
        out = capsys.readouterr().out
        assert "17 images" in out
        assert "nan" not in out.lower()
