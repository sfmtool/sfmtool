# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bindings of the bench's normal steps.

``fit_normal``, ``finite_difference_normal``, ``grid_plane_normal`` and
``grown_plane_normal`` are proved in Rust on synthetic planes; what is checked
here is that each binding hands back a new track and a report that carries the
step's numbers in the documented words, and that the settings reach the step.
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
    grown_plane_normal,
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
    patch_frame = track.patch_frame
    cross = np.cross(patch_frame["u_halfvec"], patch_frame["v_halfvec"])
    return cross / np.linalg.norm(cross)


def _check_common(report, before, after):
    assert set(report) == {"normal", "estimate", "tilt", "evaluate", "text"}
    np.testing.assert_allclose(report["normal"], _normal(after), atol=1e-9)
    np.testing.assert_allclose(report["tilt"]["normal"], report["normal"])
    assert report["evaluate"]["stage"] == "track"
    assert isinstance(report["text"], str) and report["text"]
    # A normal step turns the patch about its centre.
    np.testing.assert_allclose(
        before.patch_frame["center"], after.patch_frame["center"]
    )


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
            assert set(render["patch_frame"]) == {
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


GROWN_VIEW_KEYS = {
    "reference": {"kind", "keypoint"},
    "joined": {
        "kind",
        "start",
        "shift",
        "grid_zncc",
        "distance",
        "held_zncc",
        "was",
        "keypoint",
    },
    "shift_only": {"kind", "start", "shift", "grid_zncc", "was", "keypoint"},
    "left_out": {"kind", "reason", "distance", "held_zncc"},
    "not_attempted": {"kind"},
}
GROWN_STARTS = {"no_search", "keypoint", "widened", "no_keypoint"}
GROWN_REFUSALS = {"under_floor", "too_far", "held_under_floor"}

# Every keyword of ``grown_plane_normal`` but the fit's, at the Rust default.
GROWN_DEFAULTS = {
    "spot_window": 7,
    "window_fractions": [1 / 8, 1 / 5, 1 / 3, 1 / 2],
    "keypoint_search_px": 3.0,
    "min_keypoint_zncc": 0.7,
    "spot_search_px": 12.0,
    "spot_search_step_px": 0.5,
    "zncc_tolerance": 0.01,
    "max_agreement_distance": 3.0,
    "min_held_zncc": 0.5,
    "low_confidence_sigma_deg": 2.0,
    "weak_turn_deg": 70.0,
    "min_turn_sigma_ratio": 3.0,
    "max_pairs": 3,
    "target_sigma_deg": 0.5,
    "max_selected_views": 6,
    "first_rung_deg": 15.0,
    "first_rung_min_deg": 10.0,
    "first_rung_max_deg": 20.0,
    "rung_ratio": 1.6,
    "min_coverage": 0.6,
    "max_iterations": 20,
    "initial_damping": 1e-3,
    "min_improvement": 1e-6,
    "jacobian_tilt_step": 1e-3,
    "jacobian_shift_step_px": 0.05,
}


@pytest.fixture(scope="module")
def grown(fitted, edited, images):
    """``grown_plane_normal`` at its defaults from the fitted track."""
    return grown_plane_normal(fitted, edited, images)


class TestGrownPlaneNormal:
    def test_the_report_carries_the_sequence(self, fitted, grown):
        turned, report = grown
        _check_common(report, fitted, turned)
        seq = report["estimate"]
        assert seq["kind"] == "grown_plane"
        assert isinstance(seq["reference"], int)
        assert seq["working_resolution"] >= GROWN_DEFAULTS["spot_window"]
        widths = seq["window_half_widths"]
        assert widths == sorted(widths) and widths[0] == 3
        # The last window, half the side rounded, covers the whole grid.
        assert widths[-1] == int(np.floor(seq["working_resolution"] / 2 + 0.5))

        views = seq["views"]
        assert len(views) == fitted.observation_count
        for view in views:
            assert set(view) == GROWN_VIEW_KEYS[view["kind"]]
        assert views[seq["reference"]]["kind"] == "reference"
        assert sum(v["kind"] == "reference" for v in views) == 1
        joined = [v for v in views if v["kind"] == "joined"]
        assert seq["joined"] == len(joined)
        assert seq["left_out"] == sum(v["kind"] == "left_out" for v in views)
        assert seq["shift_only"] == sum(v["kind"] == "shift_only" for v in views)
        assert seq["matched"] == len({e["observation"] for e in seq["trace"]})
        # Every view grown against the reference was picked, and every view
        # picked was grown.
        assert {p["observation"] for p in seq["picks"]} == {
            e["observation"] for e in seq["trace"]
        }
        for view in joined:
            assert view["start"] in GROWN_STARTS
            assert view["shift"].shape == (2,)
            assert isinstance(view["grid_zncc"], float)
        for view in views:
            if view["kind"] == "left_out":
                assert view["reason"] in GROWN_REFUSALS

        assert seq["trace"]
        for entry in seq["trace"]:
            assert set(entry) == {
                "observation",
                "image",
                "angle_deg",
                "attempts",
                "outcome",
                "combined_normal",
                "combined_sigma_deg",
            }
            assert entry["outcome"] in {"matched", "left_out"}
            # From the combined plane, then from the starting plane at the
            # keypoint-implied shift and then from the search; a retry after
            # the combination is the search alone, or nothing where it lands
            # on the implied shift.
            assert len(entry["attempts"]) <= 3
            for attempt in entry["attempts"]:
                assert attempt["start"] in GROWN_STARTS
                assert attempt["stage_zncc"].dtype == np.float64
                assert attempt["shift"].shape == (2,)
                assert np.linalg.norm(attempt["normal"]) == pytest.approx(1.0)
                assert attempt["refusal"] in {None, "under_floor"}
                assert attempt["tilt_start"] in {"starting", "combined"}
            if entry["combined_normal"] is not None:
                assert np.linalg.norm(entry["combined_normal"]) == pytest.approx(1.0)

        for pick in seq["picks"]:
            assert set(pick) == {
                "observation",
                "image",
                "angle_deg",
                "offset_direction",
                "scale_px",
                "predicted_weak",
                "matched",
                "rung",
                "sigma_after_deg",
            }
            assert isinstance(pick["rung"], int)
            assert pick["offset_direction"].shape == (2,)
            assert isinstance(pick["matched"], bool)
        assert seq["pairs_kept"] == sum(p["kept"] for p in seq["pairs"])
        for pair in seq["pairs"]:
            assert set(pair) == {
                "observations",
                "images",
                "angle_deg",
                "normal",
                "relative_shift",
                "zncc",
                "sigma_deg",
                "distance",
                "kept",
                "refusal",
            }
            assert len(pair["observations"]) == 2
            assert pair["relative_shift"].shape == (2,)
            assert np.linalg.norm(pair["normal"]) == pytest.approx(1.0)
            assert pair["kept"] == (pair["refusal"] is None)
            assert pair["refusal"] is None or pair["refusal"] in GROWN_REFUSALS | {
                "view_left_out"
            }

        assert seq["outcome"] in {"turned", "none_joined", "weak_turn"}
        assert seq["turned"] == (seq["outcome"] == "turned")
        assert (seq["outcome"] == "none_joined") == (seq["joined"] == 0)
        if seq["joined"] > 0:
            assert seq["turn_deg"] >= 0.0
            assert seq["turn_sigma_ratio"] == pytest.approx(
                seq["turn_deg"] / seq["precision"]["sigma_weak_deg"]
            )
        else:
            assert seq["turn_deg"] is None and seq["turn_sigma_ratio"] is None
        if seq["turned"]:
            np.testing.assert_allclose(report["normal"], seq["normal"], atol=1e-9)
            assert seq["turn_deg"] == pytest.approx(report["tilt"]["degrees"])
            precision = seq["precision"]
            assert set(precision) == {
                "sigma_weak_deg",
                "sigma_strong_deg",
                "discounted_sigma_weak_deg",
                "discounted_sigma_strong_deg",
                "low_confidence",
            }
            assert precision["sigma_weak_deg"] >= precision["sigma_strong_deg"]
            assert (
                precision["discounted_sigma_weak_deg"]
                >= precision["sigma_weak_deg"] - 1e-12
            )
            assert isinstance(precision["low_confidence"], bool)
            assert seq["written"] >= 1
        else:
            assert seq["normal"] is None and seq["precision"] is None
            np.testing.assert_allclose(_normal(turned), _normal(fitted), atol=1e-9)

    def test_the_defaults_are_the_rust_steps(self, fitted, edited, images, grown):
        _, plain = grown
        _, report = grown_plane_normal(fitted, edited, images, **GROWN_DEFAULTS)
        assert report["text"] == plain["text"]
        np.testing.assert_array_equal(report["normal"], plain["normal"])

    def test_the_settings_reach_the_step(self, fitted, edited, images):
        _, report = grown_plane_normal(
            fitted, edited, images, working_resolution=30, window_fractions=[0.5]
        )
        seq = report["estimate"]
        assert seq["working_resolution"] == 30
        assert seq["window_half_widths"] == [3, 15]

    def test_the_selection_picks_near_to_far(self, grown):
        _, report = grown
        seq = report["estimate"]
        picks = seq["picks"]
        assert picks
        assert picks[0]["rung"] == 0
        rungs = [p["rung"] for p in picks]
        assert rungs == sorted(rungs)
        # Every pick after the first that passed grows from the combined
        # plane first; the first rung's picks from the starting plane.
        for pick, entry in zip(picks, seq["trace"]):
            first = entry["attempts"][0]["tilt_start"]
            assert first == ("starting" if pick["rung"] == 0 else "combined")
            starts = [a["tilt_start"] for a in entry["attempts"]]
            assert starts == sorted(starts, key=lambda s: s != "combined")

    def test_a_large_turn_on_weak_evidence_keeps_the_start(
        self, fitted, edited, images, grown
    ):
        # Every turn counts as large and no evidence is strong enough, so the
        # plane the default run turned to is found again and not turned to.
        _, plain = grown
        found = plain["estimate"]
        assert found["joined"] > 0
        kept, report = grown_plane_normal(
            fitted, edited, images, weak_turn_deg=0.0, min_turn_sigma_ratio=1e9
        )
        seq = report["estimate"]
        assert seq["outcome"] == "weak_turn"
        assert not seq["turned"]
        assert report["tilt"]["degrees"] == 0.0
        np.testing.assert_allclose(_normal(kept), _normal(fitted), atol=1e-9)
        # The found plane, its turn and its precision are still reported.
        np.testing.assert_allclose(seq["normal"], found["normal"], atol=1e-12)
        assert seq["turn_deg"] == pytest.approx(found["turn_deg"])
        assert seq["precision"] == found["precision"]
        assert seq["joined"] == found["joined"]
        assert seq["shift_only"] == 0
        assert seq["written"] == 0
        assert "normal left unchanged: large turn on weak evidence" in report["text"]

    def test_bad_settings_are_refused(self, fitted, edited, images):
        for name, value in [
            ("spot_window", 0),
            ("window_fractions", [0.25, 0.0]),
            ("spot_search_step_px", 0.0),
            ("min_coverage", 1.5),
            ("zncc_tolerance", -1.0),
            ("target_sigma_deg", 0.0),
            ("weak_turn_deg", -1.0),
            ("min_turn_sigma_ratio", float("nan")),
            ("max_selected_views", 0),
            ("first_rung_deg", 25.0),
            ("first_rung_min_deg", float("nan")),
            ("rung_ratio", 1.0),
        ]:
            with pytest.raises(ValueError, match=name):
                grown_plane_normal(fitted, edited, images, **{name: value})

    def test_a_cluster_is_refused(self, edited, images):
        _, track = create_cluster(Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5)
        with pytest.raises(ValueError, match="cluster"):
            grown_plane_normal(track, edited, images)

    def test_no_phases_are_recorded_by_default(self, grown):
        _, report = grown
        assert "phases" not in report

    def test_recorded_phases_change_nothing(self, fitted, edited, images, grown):
        plain_track, plain = grown
        turned, report = grown_plane_normal(fitted, edited, images, record_phases=True)
        phases = report.pop("phases")
        _assert_same(report, plain)
        np.testing.assert_array_equal(_normal(turned), _normal(plain_track))
        # The keypoints the step wrote are the same too.
        _assert_same(
            [o["track"] for o in turned.observations],
            [o["track"] for o in plain_track.observations],
        )

        assert phases
        for phase in phases:
            assert set(phase) == {
                "name",
                "depth",
                "note",
                "start_s",
                "duration_s",
                "messages",
            }
            assert phase["duration_s"] >= 0.0 and phase["start_s"] >= 0.0
            assert isinstance(phase["depth"], int)
            for message in phase["messages"]:
                assert message["level"] in {"info", "warn"}
        names = [phase["name"] for phase in phases]
        assert names[0] == "grown plane normal"
        assert phases[0]["depth"] == 0
        # In entry order: each phase follows the phase it is inside, and
        # starts no earlier.
        for k, phase in enumerate(phases[1:], start=1):
            parent = next(
                p for p in reversed(phases[:k]) if p["depth"] + 1 == phase["depth"]
            )
            assert parent["start_s"] <= phase["start_s"]
        seq = report["estimate"]
        assert names.count("grown plane view") == len(seq["trace"])
        growths = sum(len(entry["attempts"]) for entry in seq["trace"])
        assert names.count("judge") == growths
        assert names.count("whole-grid stage") == growths
        assert names.count("window stage") == growths * (
            len(seq["window_half_widths"]) - 1
        )
        assert names.count("joint fit") == 0
        assert names.count("mean") >= (1 if seq["matched"] else 0)
        assert names.count("pair fit") == len(seq["pairs"])
        assert names.count("selection") == len(seq["picks"])
        assert names.count("shift-only fit") == seq["shift_only"]
        for once in ("setup", "turn and write"):
            assert names.count(once) == 1
        # Every phase lies within the step's.
        step = phases[0]
        end = step["start_s"] + step["duration_s"]
        for phase in phases[1:]:
            assert phase["depth"] >= 1
            assert phase["start_s"] + phase["duration_s"] <= end + 1e-3


def _assert_same(a, b):
    """``a`` and ``b`` are equal, arrays element by element and ``NaN`` equal."""
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b)
    else:
        assert a == b


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
