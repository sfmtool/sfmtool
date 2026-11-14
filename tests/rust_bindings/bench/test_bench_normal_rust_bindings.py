# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bindings of the bench's normal steps.

``finite_difference_normal``, ``grid_plane_normal``, ``grown_plane_normal``
and ``pairwise_flow_normal`` are proved in Rust on synthetic planes; what is checked
here is that each binding hands back a new track and a report that carries the
step's numbers in the documented words, and that the settings reach the step. One
test runs the grown plane normal on a ground-truth point whose repeated
passes did not settle before the working grid was laid at each stage's plane.
"""

import numpy as np
import pytest

from sfmtool.bench import (
    Bench,
    create_cluster,
    create_track,
    finite_difference_normal,
    fit,
    grid_plane_normal,
    grown_plane_normal,
    pairwise_flow_normal,
)
from sfmtool.fileio import read_image_rgb
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction
from tests.conftest import SEOUL_BULL_DIR

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


def _keypoints(track):
    """Each observation's keypoint, ``None`` where it has none."""
    return [
        None
        if not o.get("track") or "keypoint" not in o["track"]
        else o["track"]["keypoint"].tolist()
        for o in track.observations
    ]


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

    def test_a_working_grid_past_2048_is_refused(self, fitted, edited, images):
        # Refused rather than allocated, up to a side whose sizes would
        # overflow.
        for side in (2049, 2**31):
            with pytest.raises(ValueError, match="working_resolution"):
                grid_plane_normal(fitted, edited, images, working_resolution=side)


GROWN_VIEW_KEYS = {
    "reference": {"kind"},
    "joined": {"kind", "shift", "grid_zncc", "distance", "held_zncc"},
    "shift_only": {"kind", "shift", "grid_zncc"},
    "left_out": {"kind", "reason", "distance", "held_zncc"},
    "not_attempted": {"kind"},
}
GROWN_REFUSALS = {
    "under_floor",
    "stopped_early",
    "too_far",
    "held_under_floor",
    "no_keypoint",
}

# The keys of each whole-grid round of a growth or a pair.
GRID_ROUND_KEYS = {"cap_deg", "turn_deg", "zncc"}

# Every keyword of ``grown_plane_normal`` but the fit's, at the Rust default.
GROWN_DEFAULTS = {
    "spot_window": 7,
    "window_fractions": [1 / 8, 1 / 5, 1 / 3, 1 / 2],
    "zncc_tolerance": 0.01,
    "max_agreement_distance": 3.0,
    "min_held_zncc": 0.5,
    "low_confidence_sigma_deg": 3.67,
    "weak_turn_deg": 70.0,
    "min_turn_sigma_ratio": 3.0,
    "max_pairs": 3,
    "target_sigma_deg": 2.75,
    "max_selected_views": 6,
    "first_rung_deg": 15.0,
    "first_rung_min_deg": 10.0,
    "first_rung_max_deg": 20.0,
    "rung_ratio": 1.6,
    "min_coverage": 0.6,
    "max_iterations": 20,
    "initial_damping": 1e-3,
    "min_improvement": 1e-6,
    "turned_support": True,
    "stage_turn_cap_deg": 12.0,
    "min_stage_turn_cap_deg": 0.5,
    "max_grid_rounds": 16,
}


@pytest.fixture(scope="module")
def grown(fitted, edited, images):
    """``grown_plane_normal`` at its defaults from the fitted track."""
    return grown_plane_normal(fitted, edited, images)


class TestGrownPlaneNormal:
    def test_the_report_carries_the_sequence(self, fitted, grown):
        turned, report = grown
        _check_common(report, fitted, turned)
        # The step writes no keypoint, whether or not it turned the patch.
        assert _keypoints(turned) == _keypoints(fitted)
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
        # The agreement is the mean score of the views in the plane.
        scores = [
            v["grid_zncc"]
            for v in views
            if v["kind"] in ("joined", "shift_only") and not np.isnan(v["grid_zncc"])
        ]
        if scores:
            assert seq["agreement"] == pytest.approx(np.mean(scores))
        else:
            assert np.isnan(seq["agreement"])
        # Every view grown against the reference was picked, and every view
        # picked was grown.
        assert {p["observation"] for p in seq["picks"]} == {
            e["observation"] for e in seq["trace"]
        }
        for view in joined:
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
            # From the combined plane, then, on the second rung where that
            # is refused, from the starting plane; both at the
            # keypoint-implied shift, with no search.
            assert 1 <= len(entry["attempts"]) <= 2
            for attempt in entry["attempts"]:
                assert attempt["stage_zncc"].dtype == np.float64
                assert attempt["shift"].shape == (2,)
                assert np.linalg.norm(attempt["normal"]) == pytest.approx(1.0)
                assert attempt["refusal"] in {None, "under_floor", "stopped_early"}
                assert attempt["tilt_start"] in {"starting", "combined"}
                stopped = attempt["refusal"] == "stopped_early"
                assert attempt["stage_turn_deg"].dtype == np.float64
                # One turn per fitted window stage, as far as the growth ran.
                assert len(attempt["stage_turn_deg"]) == len(attempt["stage_zncc"]) - (
                    1 if stopped else 2
                )
                assert (stopped) == (not attempt["grid_rounds"])
                for r in attempt["grid_rounds"]:
                    assert set(r) == GRID_ROUND_KEYS
                    assert r["turn_deg"] <= r["cap_deg"] + 1e-6
                if not stopped:
                    assert (
                        attempt["grid_rounds"][-1]["zncc"] == attempt["stage_zncc"][-1]
                    )
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
                "grid_rounds",
            }
            assert pair["grid_rounds"]
            for r in pair["grid_rounds"]:
                assert set(r) == GRID_ROUND_KEYS
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

    def test_the_turned_support_settings_reach_the_step(self, fitted, edited, images):
        def attempts(report):
            return [
                a
                for entry in report["estimate"]["trace"]
                for a in entry["attempts"]
                if a["refusal"] != "stopped_early"
            ]

        # Off: one grid, one whole-grid round with no cap, and the growth's
        # own score is its last stage's.
        _, off = grown_plane_normal(fitted, edited, images, turned_support=False)
        assert attempts(off)
        for a in attempts(off):
            assert len(a["grid_rounds"]) == 1
            assert a["grid_rounds"][0]["cap_deg"] == float("inf")
            assert a["grid_zncc"] == a["stage_zncc"][-1]
        for pair in off["estimate"]["pairs"]:
            assert len(pair["grid_rounds"]) == 1
        # A cap of 2 degrees and at most two rounds hold every stage and
        # every growth's rounds.
        _, capped = grown_plane_normal(
            fitted,
            edited,
            images,
            stage_turn_cap_deg=2.0,
            min_stage_turn_cap_deg=0.25,
            max_grid_rounds=2,
        )
        assert attempts(capped)
        # The stored patch axes are orthogonal to about 1e-8, so a turn on the
        # disc's edge reads 2 degrees to about 1e-7.
        for a in attempts(capped):
            assert all(t <= 2.0 + 1e-6 for t in a["stage_turn_deg"])
            assert 1 <= len(a["grid_rounds"]) <= 2
            assert a["grid_rounds"][0]["cap_deg"] == 2.0

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
            # Past the second rung, a pick is grown from the combined plane
            # alone.
            if pick["rung"] >= 2:
                assert starts == ["combined"]

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
        assert _keypoints(kept) == _keypoints(fitted)
        assert "normal left unchanged: large turn on weak evidence" in report["text"]

    def test_resolution_is_the_r_the_working_grid_follows(
        self, fitted, edited, images, grown
    ):
        # The fixture stores no patch bitmaps, so ``resolution`` is the ``R``
        # the working grid's side is clamped to ``R ..= 2R`` by, and the side
        # is never under ``spot_window`` (7). The default ``R`` is 24.
        _, default = grown
        assert 24 <= default["estimate"]["working_resolution"] <= 48
        _, report = grown_plane_normal(fitted, edited, images, resolution=4)
        assert 7 <= report["estimate"]["working_resolution"] <= 8
        _, report = grown_plane_normal(fitted, edited, images, resolution=40)
        assert 40 <= report["estimate"]["working_resolution"] <= 80

    def test_bad_settings_are_refused(self, fitted, edited, images):
        for name, value in [
            ("spot_window", 0),
            ("window_fractions", [0.25, 0.0]),
            ("min_coverage", 1.5),
            ("zncc_tolerance", -1.0),
            ("target_sigma_deg", 0.0),
            ("weak_turn_deg", -1.0),
            ("min_turn_sigma_ratio", float("nan")),
            ("max_selected_views", 0),
            ("first_rung_deg", 25.0),
            ("first_rung_min_deg", float("nan")),
            ("rung_ratio", 1.0),
            ("stage_turn_cap_deg", 90.0),
            ("min_stage_turn_cap_deg", 0.0),
            ("max_grid_rounds", 0),
            ("resolution", 1),
            # Past 1024, up to a size that would overflow, refused rather
            # than allocated.
            ("resolution", 1025),
            ("resolution", 2**31),
            # Past 2048, the working grid and the first window, which the
            # grid is at least as wide as.
            ("working_resolution", 2049),
            ("working_resolution", 2**31),
            ("spot_window", 2049),
            ("spot_window", 2**31),
        ]:
            with pytest.raises(ValueError, match=name):
                grown_plane_normal(fitted, edited, images, **{name: value})

    def test_the_search_keywords_are_not_taken(self, fitted, edited, images):
        # The step trusts the keypoints and has no shift search to set.
        for name in [
            "keypoint_search_px",
            "min_keypoint_zncc",
            "spot_search_px",
            "spot_search_step_px",
        ]:
            with pytest.raises(TypeError, match=name):
                grown_plane_normal(fitted, edited, images, **{name: 1.0})

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
        # One whole-grid stage per round of every growth.
        rounds = sum(
            len(a["grid_rounds"]) for entry in seq["trace"] for a in entry["attempts"]
        )
        assert names.count("whole-grid stage") == rounds >= growths
        assert names.count("window stage") == growths * (
            len(seq["window_half_widths"]) - 1
        )
        assert names.count("joint fit") == 0
        assert names.count("mean") >= (1 if seq["matched"] else 0)
        assert names.count("pair fit") == len(seq["pairs"])
        assert names.count("selection") == len(seq["picks"])
        assert names.count("shift-only fit") == seq["shift_only"]
        for once in ("setup", "turn"):
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


# Every keyword of ``pairwise_flow_normal`` but the fit's, at the Rust default.
PAIRWISE_DEFAULTS = {
    "rounds": 3,
    "max_view_angle_deg": 80.0,
    "max_normal_from_mean_view_deg": 80.0,
    "fallback_radius": 1.0,
    "min_samples_per_patch": 24,
    "max_samples_per_patch": 128,
    "crop_half_extents": 3.0,
    "min_crop_half_width_px": 48.0,
    "max_crop_samples": 512,
    "radii": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    "max_pairs": 10,
    "max_grid_tilt_deg": 85.0,
    "grid_tilt_step_deg": 10.0,
    "grid_azimuth_step_deg": 30.0,
    "tukey_c": 4.685,
    "correlation_length_px": 8.0,
    "noise_false_alarm": 0.001,
    "flow_floor_px": 0.03,
    "max_sigma_deg": 10.0,
    "agreement_sigmas": 3.0,
    "agreement_margin_deg": 5.0,
    "max_mean_chi2": 2.0,
    "single_pair_radius": 1.0,
    # The ``high_quality`` preset with ``finest_scale`` 0 and stride 1.
    "flow_patch_size": 12,
    "flow_patch_stride": 1,
    "flow_grad_descent_iterations": 16,
    "flow_finest_scale": 0,
    "flow_variational_refinement": True,
    "flow_variational_jacobi_iterations": 7,
    "flow_variational_outer_iterations_base": 1,
    "flow_variational_alpha": 10.0,
    "flow_variational_gamma": 10.0,
    "flow_variational_delta": 5.0,
    "flow_normalize_patches": True,
}

PAIRWISE_OUTCOMES = {"turned", "single_pair", "undetermined", "unstable"}
PAIRWISE_CROP_KEYS = {
    "rendered": {"kind", "image", "angle_deg", "samples_per_pixel", "coverage"},
    "too_oblique": {"kind", "image", "angle_deg"},
    "not_attempted": {"kind"},
}
PAIRWISE_FIT_KEYS = {
    "normal",
    "sigma_deg",
    "sigma_texture_deg",
    "sigma_residual_deg",
    "offset",
    "rms_residual",
    "residual_variance",
    "determined",
    "combined",
}


# A point of seoul_bull v02 seen in 7 images, which the step turns in three
# rounds of 10 pairs (2.5 degrees, at 3 half-extents), in well under a second.
PAIRWISE_POINT = 2


@pytest.fixture(scope="module")
def ground_truth():
    """seoul_bull v02 as an ``EditedReconstruction``, with its photographs."""
    path = SEOUL_BULL_DIR / "seoul_bull_sculpture_ground_truth_v02.sfmr"
    recon = SfmrReconstruction.load(str(path))
    images = [read_image_rgb(SEOUL_BULL_DIR / name) for name in recon.image_names]
    return EditedReconstruction(recon), images


@pytest.fixture(scope="module")
def flow_track(ground_truth):
    """The track of ``PAIRWISE_POINT``, as put on the bench."""
    edited, _ = ground_truth
    _, track = create_track(Bench(), edited, PAIRWISE_POINT)
    return track


@pytest.fixture(scope="module")
def pairwise(flow_track, ground_truth):
    """``pairwise_flow_normal`` at its defaults from ``flow_track``."""
    edited, images = ground_truth
    return pairwise_flow_normal(flow_track, edited, images)


def _check_round(round_, track, radii, max_sigma_deg):
    """One round's dict holds the step's numbers in the documented shape, and
    its per-radius counts agree with its pairs' fits."""
    assert np.linalg.norm(round_["plane_normal"]) == pytest.approx(1.0)
    crops = round_["crops"]
    assert len(crops) == track.observation_count
    for crop in crops:
        assert set(crop) == PAIRWISE_CROP_KEYS[crop["kind"]]
    rendered = {k for k, c in enumerate(crops) if c["kind"] == "rendered"}
    assert len(rendered) >= 2
    assert round_["crop_half_width"] > 0
    assert round_["sample_spacing"] > 0.0
    assert [row["radius"] for row in round_["radii"]] == radii
    assert round_["pairs"]
    for pair in round_["pairs"]:
        a, b = pair["observations"]
        assert a < b and {a, b} <= rendered
        assert pair["images"] == [crops[a]["image"], crops[b]["image"]]
        assert pair["half_width"] > 0
        assert len(pair["fits"]) == len(radii)
        for fit_ in pair["fits"]:
            if fit_ is None:
                continue
            assert set(fit_) == PAIRWISE_FIT_KEYS
            assert fit_["normal"].shape == (3,) and fit_["offset"].shape == (2,)
            assert fit_["sigma_deg"] == max(
                fit_["sigma_texture_deg"], fit_["sigma_residual_deg"]
            )
            assert fit_["determined"] == (fit_["sigma_deg"] <= max_sigma_deg)
            assert fit_["determined"] or not fit_["combined"]
    for k, row in enumerate(round_["radii"]):
        fits = [p["fits"][k] for p in round_["pairs"] if p["fits"][k] is not None]
        assert row["pairs_fitted"] == len(fits)
        assert row["pairs_determined"] == sum(f["determined"] for f in fits)
        assert row["pairs_combined"] == sum(f["combined"] for f in fits)
        assert (row["normal"] is None) == (row["pairs_determined"] == 0)
    assert round_["outcome"] in PAIRWISE_OUTCOMES
    if round_["normal"] is not None:
        assert round_["chosen_radius"] in radii
        assert np.linalg.norm(round_["normal"]) == pytest.approx(1.0)


class TestPairwiseFlowNormal:
    def test_the_report_carries_every_round(self, flow_track, pairwise):
        turned, report = pairwise
        _check_common(report, flow_track, turned)
        assert _keypoints(turned) == _keypoints(flow_track)
        est = report["estimate"]
        assert set(est) == {
            "kind",
            "outcome",
            "chosen_radius",
            "normal",
            "sigma_deg",
            "turn_deg",
            "rounds",
        }
        assert est["kind"] == "pairwise_flow"
        rounds = est["rounds"]
        assert len(rounds) == PAIRWISE_DEFAULTS["rounds"]
        for round_ in rounds:
            _check_round(
                round_,
                flow_track,
                PAIRWISE_DEFAULTS["radii"],
                PAIRWISE_DEFAULTS["max_sigma_deg"],
            )
            assert len(round_["pairs"]) == PAIRWISE_DEFAULTS["max_pairs"]
            assert round_["samples_per_patch"] >= 24
        # Each round renders on the plane the round before handed on, and the
        # last hands on none.
        for before, after in zip(rounds, rounds[1:]):
            np.testing.assert_array_equal(before["next_plane"], after["plane_normal"])
        assert rounds[-1]["next_plane"] is None
        assert est["outcome"] == rounds[-1]["outcome"] == "turned"
        # The patch is turned to the last round's answer, facing as it did.
        np.testing.assert_allclose(report["normal"], est["normal"], atol=1e-9)
        assert np.dot(est["normal"], _normal(flow_track)) > 0.0
        assert est["chosen_radius"] == rounds[-1]["chosen_radius"]
        assert est["sigma_deg"] == rounds[-1]["sigma_deg"]
        assert est["turn_deg"] == pytest.approx(report["tilt"]["degrees"])

    def test_the_defaults_are_the_rust_steps_and_phases_change_nothing(
        self, flow_track, ground_truth
    ):
        edited, images = ground_truth
        # One round each, so the comparison costs a third of a default run.
        plain_track, plain = pairwise_flow_normal(flow_track, edited, images, rounds=1)
        assert "phases" not in plain
        turned, report = pairwise_flow_normal(
            flow_track,
            edited,
            images,
            **{**PAIRWISE_DEFAULTS, "rounds": 1},
            record_phases=True,
        )
        phases = report.pop("phases")
        _assert_same(report, plain)
        np.testing.assert_array_equal(_normal(turned), _normal(plain_track))
        names = [phase["name"] for phase in phases]
        assert names[0] == "pairwise flow normal"
        assert phases[0]["depth"] == 0
        assert names.count("round") == 1
        assert names.count("crops") == 1

    def test_the_settings_reach_the_step(self, flow_track, ground_truth):
        edited, images = ground_truth
        turned, report = pairwise_flow_normal(
            flow_track,
            edited,
            images,
            rounds=1,
            radii=[1.0, 2.0],
            max_pairs=2,
            min_samples_per_patch=8,
            max_samples_per_patch=8,
        )
        _check_common(report, flow_track, turned)
        rounds = report["estimate"]["rounds"]
        assert len(rounds) == 1
        _check_round(
            rounds[0], flow_track, [1.0, 2.0], PAIRWISE_DEFAULTS["max_sigma_deg"]
        )
        assert len(rounds[0]["pairs"]) == 2
        # The grid is held at 8 samples along the patch, where the defaults
        # give this point at least 24.
        assert rounds[0]["samples_per_patch"] <= 8

    def test_the_flow_settings_reach_the_step(self, flow_track, ground_truth):
        edited, images = ground_truth
        # A flow patch wider than every crop leaves every pair with no flow,
        # so no fit, and the normal undetermined.
        turned, report = pairwise_flow_normal(
            flow_track, edited, images, rounds=1, flow_patch_size=4096
        )
        est = report["estimate"]
        pairs = est["rounds"][0]["pairs"]
        assert pairs
        assert all(f is None for p in pairs for f in p["fits"])
        assert est["outcome"] == "undetermined"
        for key in ("chosen_radius", "normal", "sigma_deg", "turn_deg"):
            assert est[key] is None, key
        np.testing.assert_allclose(_normal(turned), _normal(flow_track), atol=1e-9)

    def test_bad_settings_are_refused(self, flow_track, ground_truth):
        edited, images = ground_truth
        for name, value, match in [
            ("rounds", 0, "rounds"),
            ("max_view_angle_deg", 0.0, "max_view_angle_deg"),
            ("max_normal_from_mean_view_deg", 0.0, "max_normal_from_mean_view_deg"),
            ("radii", [2.0, 1.0], "radii"),
            ("fallback_radius", 5.0, "fallback_radius"),
            ("single_pair_radius", 5.0, "single_pair_radius"),
            ("max_pairs", 1, "max_pairs"),
            ("min_samples_per_patch", 1, "min_samples_per_patch"),
            # Below the minimum, which is refused under the minimum's name.
            ("max_samples_per_patch", 1, "min_samples_per_patch"),
            ("crop_half_extents", 0.5, "crop_half_extents"),
            ("min_crop_half_width_px", -1.0, "min_crop_half_width_px"),
            ("max_crop_samples", 5000, "max_crop_samples"),
            ("max_grid_tilt_deg", 90.0, "max_grid_tilt_deg"),
            ("grid_tilt_step_deg", 0.0, "grid_tilt_step_deg"),
            ("grid_azimuth_step_deg", 0.0, "grid_azimuth_step_deg"),
            ("tukey_c", 0.0, "tukey_c"),
            ("noise_false_alarm", 1.0, "noise_false_alarm"),
            ("correlation_length_px", 17.0, "correlation_length_px"),
            ("flow_floor_px", -1.0, "flow_floor_px"),
            ("max_sigma_deg", 0.0, "max_sigma_deg"),
            ("agreement_sigmas", -1.0, "agreement_sigmas"),
            ("agreement_margin_deg", -1.0, "agreement_margin_deg"),
            ("max_mean_chi2", 0.0, "max_mean_chi2"),
            ("resolution", 1, "resolution"),
            ("resolution", 1025, "resolution"),
            ("resolution", 2**31, "resolution"),
            ("flow_patch_size", 0, "patch_size"),
            ("flow_patch_stride", 0, "patch_stride"),
            ("flow_grad_descent_iterations", 0, "grad_descent_iterations"),
            ("flow_finest_scale", 40, "finest_scale 40"),
            ("flow_coarsest_scale", 40, "coarsest_scale 40"),
            ("flow_variational_jacobi_iterations", 0, "jacobi_iterations"),
            ("flow_variational_outer_iterations_base", 0, "outer_iterations_base"),
            ("flow_variational_alpha", -1.0, "variational_alpha"),
            ("flow_variational_gamma", -1.0, "variational_gamma"),
            ("flow_variational_delta", -1.0, "variational_delta"),
        ]:
            with pytest.raises(ValueError, match=match):
                pairwise_flow_normal(flow_track, edited, images, **{name: value})
        # Twice a grid of 2^31 samples does not fit in 32 bits; the crop cap is
        # still held to it.
        with pytest.raises(ValueError, match="max_crop_samples"):
            pairwise_flow_normal(
                flow_track,
                edited,
                images,
                min_samples_per_patch=2**31,
                max_samples_per_patch=2**31,
            )

    def test_gpu_min_pixels_is_not_taken(self, flow_track, ground_truth):
        edited, images = ground_truth
        # The step's flow runs on the CPU.
        with pytest.raises(TypeError, match="gpu_min_pixels"):
            pairwise_flow_normal(flow_track, edited, images, flow_gpu_min_pixels=0)

    def test_a_cluster_is_refused(self, ground_truth):
        edited, images = ground_truth
        _, track = create_cluster(Bench(), 4, "IMG_0042", (142.0, 197.5), radius_px=7.5)
        with pytest.raises(ValueError, match="cluster"):
            pairwise_flow_normal(track, edited, images)


def test_repeated_grown_plane_passes_settle_on_a_ground_truth_point():
    """seoul_bull v02 point 255 straddles a canopy edge, seen by three views
    within 4.1 degrees of each other. On one grid laid on the starting square
    the step turned it 69.6, 15.6, 3.9, 4.4 and 3.5 degrees on five passes;
    with the grid laid at each stage's plane the second and third passes turn
    it 4.4 and 0.3 degrees, under the bar of 5."""
    path = SEOUL_BULL_DIR / "seoul_bull_sculpture_ground_truth_v02.sfmr"
    recon = SfmrReconstruction.load(str(path))
    edited = EditedReconstruction(recon)
    images = [read_image_rgb(SEOUL_BULL_DIR / name) for name in recon.image_names]
    _, track = create_track(Bench(), edited, 255)
    turns = []
    for _ in range(3):
        track, report = grown_plane_normal(track, edited, images)
        assert report["estimate"]["outcome"] == "turned"
        turns.append(report["tilt"]["degrees"])
    assert turns[0] > 30.0, turns
    assert turns[1] < 5.0 and turns[2] < 5.0, turns
