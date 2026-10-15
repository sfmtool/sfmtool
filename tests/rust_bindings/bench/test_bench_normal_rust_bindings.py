# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bindings of the bench's normal steps.

``fit_normal``, ``finite_difference_normal`` and ``grid_plane_normal`` are
proved in Rust on synthetic planes; what is checked here is that each binding
hands back a new track and a report that carries the step's numbers in the
documented words, and that the settings reach the step.
"""

import numpy as np
import pytest

from sfmtool.bench import (
    Bench,
    create_cluster,
    create_track,
    finite_difference_normal,
    fit,
    fit_normal,
    grid_plane_normal,
)
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction

PLANE_WORDS = {
    "in_plane",
    "plane_outlier",
    "positioned",
    "too_few_rays",
    "narrow_baseline",
    "behind_camera",
}
CELL_WORDS = {
    "fitted",
    "refused_curvature",
    "refused_zncc",
    "not_attempted",
    "refused_bound",
    "refused_outlier",
}
STOP_OUTCOME = {
    "converged": "turned",
    "not_run": "undetermined",
    "measured": "undetermined",
    "rejected": "unstable",
    "oscillation": "unstable",
    "cap": "unstable",
}


@pytest.fixture(scope="module")
def embedded(seoul_bull_workspace_once):
    """The 17-image reconstruction, converted to ``embedded_patches``."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once)
    return recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)


@pytest.fixture(scope="module")
def edited(embedded):
    return EditedReconstruction(embedded)


@pytest.fixture(scope="module")
def images(embedded):
    from sfmtool._workspace_image import read_workspace_image

    return [
        read_workspace_image(embedded.workspace_dir, name)
        for name in embedded.image_names
    ]


@pytest.fixture(scope="module")
def fitted(embedded, edited, images):
    """The fixture's longest track, fitted, as the normal steps start from."""
    counts = np.asarray(embedded.observation_counts)
    if counts.size == 0 or counts.max() < 3:
        pytest.skip("the fixture holds no track of three observations")
    _, track = create_track(Bench(), edited, int(np.argmax(counts)))
    track, _ = fit(track, edited, images)
    return track


def _normal(track):
    frame = track.placement
    cross = np.cross(frame["u_halfvec"], frame["v_halfvec"])
    return cross / np.linalg.norm(cross)


def _check_common(report, before, after):
    assert set(report) == {"normal", "estimate", "tilt", "evaluate", "text"}
    np.testing.assert_allclose(report["normal"], _normal(after), atol=1e-9)
    np.testing.assert_allclose(report["tilt"]["normal"], report["normal"])
    assert report["evaluate"]["stage"] == "track"
    assert isinstance(report["text"], str) and report["text"]
    # A normal step turns the patch about its centre.
    np.testing.assert_allclose(before.placement["center"], after.placement["center"])


class TestGridPlaneNormal:
    def test_the_report_carries_every_cell_reading(self, fitted, edited, images):
        was = _normal(fitted)
        turned, report = grid_plane_normal(fitted, edited, images)
        _check_common(report, fitted, turned)
        np.testing.assert_allclose(_normal(fitted), was)

        cells = report["estimate"]
        assert cells["kind"] == "grid_plane_cells"
        assert cells["outcome"] == STOP_OUTCOME[cells["stop"]]
        assert cells["determinacy"] in {"both_axes", "one_axis", "none"}
        if cells["determinacy"] == "one_axis":
            assert np.linalg.norm(cells["free_axis"]) == pytest.approx(1.0)
        else:
            assert cells["free_axis"] is None
        assert len(cells["cell_status"]) == 3
        assert all(len(row) == 3 for row in cells["cell_status"])
        assert {w for row in cells["cell_status"] for w in row} <= PLANE_WORDS
        assert cells["in_plane"] == sum(
            w == "in_plane" for row in cells["cell_status"] for w in row
        )
        assert cells["cell_rays"].shape == (3, 3)
        assert cells["cell_rays"].dtype == np.uint32
        assert cells["cell_weight"].shape == (3, 3)
        assert len(cells["view_cells"]) == fitted.observation_count
        for grid in cells["view_cells"]:
            assert len(grid) == 3 and all(len(row) == 3 for row in grid)
            assert {w for row in grid for w in row} <= CELL_WORDS
        assert cells["working_resolution"] >= 3
        assert 1 <= cells["iterations"] <= 5
        for key in ("plane_rms", "plane_offset", "score_before", "score_after"):
            assert isinstance(cells[key], float)

        assert isinstance(cells["normal_precision_deg"], float)
        assert cells["view_baseline_deg"] > 0.0
        assert len(cells["trace"]) == cells["iterations"]
        for entry in cells["trace"]:
            assert set(entry) == {
                "score",
                "determinacy",
                "in_plane",
                "normal_precision_deg",
                "gated",
                "proposed_normal",
                "largest_move_px",
                "taken",
            }
            if entry["proposed_normal"] is None:
                assert not entry["taken"]
            else:
                assert np.linalg.norm(entry["proposed_normal"]) == pytest.approx(1.0)

        if cells["outcome"] != "turned":
            # Every ending but a converged one leaves the normal as given.
            np.testing.assert_allclose(_normal(turned), was, atol=1e-9)
            assert not report["tilt"]["changed"]

    def test_the_settings_reach_the_step(self, fitted, edited, images):
        _, report = grid_plane_normal(
            fitted, edited, images, working_resolution=30, max_iterations=1
        )
        cells = report["estimate"]
        assert cells["working_resolution"] == 30
        assert cells["iterations"] <= 1

    def test_the_detail_record_holds_every_render(self, fitted, edited, images):
        _, plain = grid_plane_normal(fitted, edited, images)
        assert "detail" not in plain["estimate"]
        _, report = grid_plane_normal(fitted, edited, images, record_detail=True)
        cells = report["estimate"]
        # The record changes nothing else in the report.
        for key in ("outcome", "stop", "iterations", "score_before", "score_after"):
            assert cells[key] == plain["estimate"][key], key
        np.testing.assert_array_equal(report["normal"], plain["normal"])

        detail = cells["detail"]
        assert len(detail) == len(cells["trace"])
        r = cells["working_resolution"]
        for render, entry in zip(detail, cells["trace"]):
            assert render["working_resolution"] == r
            side = r + 2 * render["margin"]
            assert render["cell_bounds"].tolist() == [0, r // 3, r - r // 3, r]
            assert render["cell_centres"].shape == (3, 3, 2)
            assert set(render["frame"]) == {
                "center",
                "u_axis",
                "v_axis",
                "normal",
                "half_extent",
            }
            reference = render["reference"]
            assert reference["tile"].dtype == np.float32
            assert reference["tile"].shape[:2] == (r, r)
            for view in render["views"]:
                assert view["observation"] != reference["observation"]
                assert view["tile"].shape[:2] == (side, side)
                for key in ("whole_shift", "shift"):
                    assert view[key].shape == (3, 3, 2)
                for key in ("zncc", "curvature", "fit_weight"):
                    assert view[key].shape == (3, 3)
                for key in ("reading", "status"):
                    assert {w for row in view[key] for w in row} <= CELL_WORDS
                if view["fit"] is not None:
                    assert view["fit"]["model"] in {"affine", "similarity", "shift"}
                    assert view["fit"]["a"].shape == (2, 2)
            cell = render["cells"]
            assert {w for row in cell["status"] for w in row} <= PLANE_WORDS
            assert cell["position"].shape == (3, 3, 3)
            assert (
                sum(w == "in_plane" for row in cell["status"] for w in row)
                == entry["in_plane"]
            )
            rays = render["rays"]
            assert rays["origin"].shape == (len(rays["cell"]), 3)
            np.testing.assert_array_equal(
                np.bincount(rays["cell"], minlength=9).reshape(3, 3),
                cell["ray_count"],
            )
            plane = render["plane"]
            assert plane["determinacy"] == entry["determinacy"]
            assert plane["taken"] == entry["taken"]
            assert plane["precision_deg"].shape == (2,)
            if entry["proposed_normal"] is not None:
                np.testing.assert_array_equal(
                    plane["proposed_normal"], entry["proposed_normal"]
                )

    def test_a_cluster_is_refused(self, edited, images):
        _, track = create_cluster(Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5)
        with pytest.raises(ValueError, match="cluster"):
            grid_plane_normal(track, edited, images)

    def test_the_acceptance_rule_and_the_precision_bar_reach_the_step(
        self, fitted, edited, images
    ):
        _, report = grid_plane_normal(fitted, edited, images, acceptance="none")
        assert report["estimate"]["stop"] != "rejected"

        _, report = grid_plane_normal(
            fitted, edited, images, max_normal_precision_deg=1e-9
        )
        cells = report["estimate"]
        assert cells["outcome"] == "undetermined"
        assert cells["stop"] == "measured"
        first = cells["trace"][0]
        assert first["gated"] or first["determinacy"] == "none"

        with pytest.raises(ValueError, match="acceptance"):
            grid_plane_normal(fitted, edited, images, acceptance="always")


class TestFiniteDifferenceNormal:
    def test_the_grid_layout_is_todays_grid_plane_normal(self, fitted, edited, images):
        turned, report = finite_difference_normal(
            fitted, edited, images, layout="grid", pieces=2, overlap=0.0
        )
        _check_common(report, fitted, turned)
        estimate = report["estimate"]
        assert estimate["kind"] == "grid_plane"
        assert estimate["pieces"] == 2
        assert 2 <= estimate["fitted"] <= 4
        assert isinstance(estimate["both_axes"], bool)
        assert isinstance(estimate["off_plane"], float)

    def test_the_default_layout_is_the_cross(self, fitted, edited, images):
        turned, report = finite_difference_normal(fitted, edited, images)
        _check_common(report, fitted, turned)
        estimate = report["estimate"]
        assert estimate["kind"] == "finite_difference"
        assert estimate["pieces"] == 2
        assert set(estimate) == {"kind", "pieces", "u", "v", "both_axes"}

    def test_bad_settings_are_refused(self, fitted, edited, images):
        with pytest.raises(ValueError, match="layout"):
            finite_difference_normal(fitted, edited, images, layout="ring")
        with pytest.raises(ValueError, match="pieces"):
            finite_difference_normal(fitted, edited, images, pieces=1)
        with pytest.raises(ValueError, match="overlap"):
            finite_difference_normal(fitted, edited, images, overlap=0.95)


class TestFitNormal:
    def test_the_photometric_search_reports_its_scores(self, fitted, edited, images):
        turned, report = fit_normal(fitted, edited, images)
        _check_common(report, fitted, turned)
        estimate = report["estimate"]
        assert estimate["kind"] == "photometric"
        assert estimate["views"] >= 2
        assert np.isfinite(estimate["before"]) and np.isfinite(estimate["after"])
