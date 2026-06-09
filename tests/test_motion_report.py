# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm motion --json` machine-readable output.

The schema is documented under "JSON Output" in
specs/cli/motion-command.md.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main


SEOUL_BULL_DIR = (
    Path(__file__).parent.parent / "test-data" / "images" / "seoul_bull_sculpture"
)


@pytest.fixture
def runner():
    return CliRunner()


def _make_translation_discontinuity(sfmr_path: Path, *, offset_m: float = 50.0):
    """Apply a large translation to all images with file number >= 11.

    Mirrors `_make_discontinuous_recon` in test_motion.py — the
    seoul_bull dataset has 17 sequentially-named images, so this plants a
    pose break between frame 10 and frame 11.
    """
    from sfmtool._filenames import number_from_filename
    from sfmtool._sfmtool import RotQuaternion, Se3Transform, SfmrReconstruction

    recon = SfmrReconstruction.load(sfmr_path)
    image_names = recon.image_names
    quats = recon.quaternions_wxyz.copy()
    trans = recon.translations.copy()

    xform = Se3Transform(translation=np.array([offset_m, 0.0, 0.0], dtype=np.float64))

    for i in range(recon.image_count):
        num = number_from_filename(image_names[i])
        if num is not None and num >= 11:
            q = RotQuaternion.from_wxyz_array(quats[i])
            center = q.camera_center(trans[i])
            new_center = xform.apply_to_point(center)
            new_q = xform.rotation @ q
            R = np.array(new_q.to_rotation_matrix())
            new_t = -R @ new_center
            quats[i] = new_q.to_wxyz_array()
            trans[i] = new_t

    return recon.clone_with_changes(quaternions_wxyz=quats, translations=trans)


# --- Reconstruction-mode helpers ---


def _run_recon_to_json(recon) -> dict:
    """Run `analyze_reconstruction` and convert the result to a JSON dict."""
    from sfmtool.motion.recon_discontinuity import analyze_reconstruction
    from sfmtool.motion.report import reconstruction_results_to_json

    results = analyze_reconstruction(recon)
    return reconstruction_results_to_json(results)


# --- Reconstruction-mode tests ---


def test_recon_with_planted_discontinuity_emits_one_edge_and_two_segments(
    sfmrfile_reconstruction_with_17_images,
):
    """A large translation between frame 10 and 11 should yield one
    discontinuity edge and two segments covering [0..10] and [11..16]."""
    recon = _make_translation_discontinuity(sfmrfile_reconstruction_with_17_images)
    report = _run_recon_to_json(recon)

    assert report["schema_version"] == 1
    assert report["mode"] == "reconstruction"
    assert len(report["sequences"]) == 1

    seq = report["sequences"][0]
    assert seq["frame_count"] == 17

    # The planted break is between file numbers 10 and 11 (seq indexes 9 and 10
    # since the seoul_bull frames are 1-indexed).
    discontinuities = seq["discontinuities"]
    assert len(discontinuities) >= 1
    planted = next(
        (d for d in discontinuities if d["frame_a"] == 10 and d["frame_b"] == 11),
        None,
    )
    assert planted is not None, (
        f"expected (10, 11) in {[d['edge'] for d in discontinuities]}"
    )
    assert "P" in planted["signals"]  # pose extrapolation should fire
    assert planted["confidence"] in {"high", "low"}

    # Each discontinuity splits the sequence; the planted one should land at
    # seq index 9 → 10.
    segments = seq["segments"]
    assert len(segments) == len(discontinuities) + 1
    # First segment starts at index 0; last segment ends at frame_count - 1.
    assert segments[0]["start_index"] == 0
    assert segments[-1]["end_index"] == seq["frame_count"] - 1
    # Segment lengths sum to frame_count (segments are inclusive, partition).
    assert sum(s["frame_count"] for s in segments) == seq["frame_count"]


def test_recon_with_no_discontinuity_emits_single_full_length_segment(
    sfmrfile_reconstruction_with_17_images,
):
    """An unmodified, smooth reconstruction has no discontinuities and one
    segment covering the whole sequence."""
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    report = _run_recon_to_json(recon)

    seq = report["sequences"][0]
    assert seq["discontinuities"] == []
    assert len(seq["segments"]) == 1
    only = seq["segments"][0]
    assert only["start_index"] == 0
    assert only["end_index"] == seq["frame_count"] - 1
    assert only["frame_count"] == seq["frame_count"]


def test_recon_thresholds_block_matches_module_constants(
    sfmrfile_reconstruction_with_17_images,
):
    """The top-level thresholds block echoes the analyzer's module-level
    constants and the resolved per-sequence threshold is derived from
    median_trans."""
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    report = _run_recon_to_json(recon)

    th = report["thresholds"]
    assert th["pose_trans_factor"] == 3.0
    assert th["pose_rot_deg"] == 15.0
    assert th["step_ratio"] == 1.5
    assert th["overlap_drop"] == 1.8
    assert th["obs_z"] == 2.5

    seq = report["sequences"][0]
    assert seq["pose_rot_threshold"] == 15.0
    # pose_trans_threshold = pose_trans_factor × median_trans
    assert seq["pose_trans_threshold"] == pytest.approx(3.0 * seq["median_trans"])


def test_recon_json_has_no_nan_or_infinity(
    sfmrfile_reconstruction_with_17_images,
):
    """The serialized report must be strictly RFC-8259 compliant: no NaN,
    +inf, or -inf leaks. We verify via `allow_nan=False`."""
    recon = _make_translation_discontinuity(sfmrfile_reconstruction_with_17_images)
    report = _run_recon_to_json(recon)
    # Round-trip via strict JSON.
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text


def test_recon_json_flags_cov_on_infinite_overlap_drop():
    """An infinite overlap_drop (total covisibility break: no tracks survive
    the edge) must fire the "Cov" flag in the JSON `flags` array, even though
    the serialized `overlap_drop` field is null (inf is not JSON-representable).

    Regression test: the serializer previously recomputed per-frame flags from
    the inf/NaN-sanitized fields, where +inf maps to None, so it silently
    dropped "Cov" while the console table (raw values) flagged it. Both paths
    now share `_flag_frame` on raw values, and the spec requires an infinite
    drop to always flag (specs/cli/motion-command.md, "Covisibility drop").

    Built from a synthetic sequence-result dict because a real reconstruction
    needs 3*OVERLAP_WINDOW (48) frames before overlap-drop is computed at all,
    which the checked-in datasets don't reach.
    """
    from sfmtool._sfmtool import RotQuaternion
    from sfmtool.motion.report import reconstruction_results_to_json

    identity = RotQuaternion(1.0, 0.0, 0.0, 0.0)
    seq = {
        "sequence": "seq_%03d.jpg",
        "sequence_name": "seq",
        "frame_count": 3,
        "extrap_results": [
            {
                "seq_idx": i,
                "frame_number": i + 1,
                "left_trans_err": None,
                "left_rot_err": None,
                "right_trans_err": None,
                "right_rot_err": None,
            }
            for i in range(3)
        ],
        "flagged_frames": [],
        "flagged_edges": {},
        "core_edges": {},
        "pair_counts": {},
        "seq_image_names": ["seq_001.jpg", "seq_002.jpg", "seq_003.jpg"],
        "seq_image_indexes": [0, 1, 2],
        "seq_frame_numbers": [1, 2, 3],
        "seq_centers": [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ],
        "seq_quats": [identity, identity, identity],
        "median_trans": 1.0,
        "median_rot": 0.0,
        # Per-edge arrays (len frame_count - 1). The second edge — the landing
        # edge for frame index 2 — is a total covisibility break (+inf).
        "step_ratios": [None, None],
        "overlap_drops": [None, float("inf")],
        # Per-frame array (len frame_count).
        "obs_z_scores": [None, None, None],
        "trans_threshold": 3.0,
        "rot_threshold": 15.0,
        "core_edge_reproj_errors": {},
    }

    report = reconstruction_results_to_json([seq])
    frames = report["sequences"][0]["frames"]

    # Frame index 2's landing edge (1, 2) carries the infinite drop: the
    # serialized field is null, but the flag must still fire.
    assert frames[2]["overlap_drop"] is None
    assert frames[2]["flags"] == ["Cov"]
    # Frame 0 has no landing edge; frame 1's landing-edge drop is None.
    assert frames[0]["flags"] == []
    assert frames[1]["flags"] == []

    # No inf must leak into the serialized report.
    json.dumps(report, allow_nan=False)


def test_segments_helper_handles_edge_cases():
    """Unit test for `_segments_from_core_edges`: covers the no-edge case,
    edges at the boundaries (yielding singleton segments), and adjacent
    edges (yielding a singleton middle segment)."""
    from sfmtool.motion.report import _segments_from_core_edges

    frame_numbers = list(range(100, 110))  # seq index i -> file number 100+i

    # No discontinuities: one full-length segment.
    segs = _segments_from_core_edges({}, 10, frame_numbers)
    assert segs == [
        {
            "start_index": 0,
            "end_index": 9,
            "start_frame": 100,
            "end_frame": 109,
            "frame_count": 10,
        }
    ]

    # Edge (0, 1): singleton first segment.
    segs = _segments_from_core_edges({(0, 1): []}, 10, frame_numbers)
    assert segs[0] == {
        "start_index": 0,
        "end_index": 0,
        "start_frame": 100,
        "end_frame": 100,
        "frame_count": 1,
    }
    assert segs[1]["start_index"] == 1
    assert segs[1]["end_index"] == 9

    # Edge (8, 9): singleton last segment.
    segs = _segments_from_core_edges({(8, 9): []}, 10, frame_numbers)
    assert segs[-1] == {
        "start_index": 9,
        "end_index": 9,
        "start_frame": 109,
        "end_frame": 109,
        "frame_count": 1,
    }

    # Adjacent edges (3, 4) and (4, 5): singleton middle segment.
    segs = _segments_from_core_edges({(3, 4): [], (4, 5): []}, 10, frame_numbers)
    assert len(segs) == 3
    assert segs[1] == {
        "start_index": 4,
        "end_index": 4,
        "start_frame": 104,
        "end_frame": 104,
        "frame_count": 1,
    }


# --- Image-sequence-mode tests ---


def test_image_sequence_json_has_samples_and_no_segments(runner, tmp_path):
    """Image-sequence mode emits per-sample data and no `segments` field
    at the sequence level."""
    json_out = tmp_path / "report.json"
    result = runner.invoke(
        main,
        [
            "motion",
            str(SEOUL_BULL_DIR),
            "-r",
            "1-4",
            "--no-adaptive",
            "--initial-stride",
            "1",
            "--json",
            str(json_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json_out.exists()

    report = json.loads(json_out.read_text())
    assert report["schema_version"] == 1
    assert report["mode"] == "image_sequence"
    assert "ratio_lower" in report["thresholds"]
    assert "ratio_upper" in report["thresholds"]

    assert len(report["sequences"]) == 1
    seq = report["sequences"][0]
    assert "segments" not in seq
    assert seq["sample_count"] == len(seq["samples"])
    assert seq["sample_count"] >= 1
    # Each sample must carry the documented scalar fields and no numpy arrays.
    for sample in seq["samples"]:
        for key in (
            "frame_number",
            "frame_name",
            "next_frame_name",
            "stride",
            "local_median_magnitude",
            "classification",
        ):
            assert key in sample
        # `classification` is either null (in-band) or one of the four bucket strings.
        c = sample["classification"]
        assert c is None or c in {
            "strong deceleration",
            "deceleration",
            "acceleration",
            "strong acceleration",
        }

    # Strict round-trip check — no NaN/Infinity leaks.
    json.dumps(report, allow_nan=False)


# --- CLI integration & error handling ---


def test_cli_recon_writes_json(
    runner, sfmrfile_reconstruction_with_17_images, tmp_path
):
    """The CLI `--json PATH` flag writes a JSON file for reconstruction mode."""
    json_out = tmp_path / "report.json"
    result = runner.invoke(
        main,
        [
            "motion",
            str(sfmrfile_reconstruction_with_17_images),
            "--json",
            str(json_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json_out.exists()
    report = json.loads(json_out.read_text())
    assert report["mode"] == "reconstruction"
    assert report["schema_version"] == 1


def test_cli_recon_json_to_nonexistent_dir_raises(
    runner, sfmrfile_reconstruction_with_17_images, tmp_path
):
    """A `--json` path whose parent directory does not exist fails cleanly."""
    bad = tmp_path / "does_not_exist" / "report.json"
    result = runner.invoke(
        main,
        [
            "motion",
            str(sfmrfile_reconstruction_with_17_images),
            "--json",
            str(bad),
        ],
    )
    assert result.exit_code != 0
    assert "Failed to write JSON report" in result.output
