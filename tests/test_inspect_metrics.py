# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-image metrics analysis module."""

import pytest

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool._inspect_metrics import _compute_per_image_metrics, print_metrics_analysis


@pytest.fixture
def rust_recon(sfmrfile_reconstruction_with_17_images):
    return SfmrReconstruction.load(str(sfmrfile_reconstruction_with_17_images))


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
        for entry in per_image:
            assert entry["observation_count"] >= 100

    def test_mean_errors_in_expected_range(self, per_image):
        for entry in per_image:
            assert 0.1 < entry["mean_error"] < 2.0

    def test_median_le_mean_le_max(self, per_image):
        for entry in per_image:
            assert entry["median_error"] <= entry["mean_error"] + 1e-9
            assert entry["mean_error"] <= entry["max_error"] + 1e-9

    def test_max_errors_above_one_pixel(self, per_image):
        for entry in per_image:
            assert entry["max_error"] > 1.0

    def test_mean_track_length_around_four(self, per_image):
        for entry in per_image:
            assert 3.0 <= entry["mean_track_length"] <= 5.0


class TestPrintMetricsAnalysis:
    def test_header_and_table(self, sfmrfile_reconstruction_with_17_images, capsys):
        print_metrics_analysis(
            sfmrfile_reconstruction_with_17_images, recon_name="test.sfmr"
        )
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

    def test_sorted_descending_by_mean_error(
        self, sfmrfile_reconstruction_with_17_images, capsys
    ):
        print_metrics_analysis(sfmrfile_reconstruction_with_17_images)
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

    def test_recon_name(self, sfmrfile_reconstruction_with_17_images, capsys):
        print_metrics_analysis(
            sfmrfile_reconstruction_with_17_images, recon_name="custom.sfmr"
        )
        assert "custom.sfmr" in capsys.readouterr().out

        print_metrics_analysis(sfmrfile_reconstruction_with_17_images)
        assert sfmrfile_reconstruction_with_17_images.name in capsys.readouterr().out

    def test_range_filter(self, sfmrfile_reconstruction_with_17_images, capsys):
        print_metrics_analysis(sfmrfile_reconstruction_with_17_images, range_expr="1-5")
        captured = capsys.readouterr()

        lines = [
            line for line in captured.out.splitlines() if "seoul_bull_sculpture" in line
        ]
        assert len(lines) == 5
        assert "Range filter: 1-5 (5 of 17 images)" in captured.out
