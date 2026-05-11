# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""JSON serialization for `sfm discontinuity` analysis results.

See `specs/cli/discontinuity-command.md` section "JSON Output" for the schema.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import click
import numpy as np

from ._discontinuity import (
    OBS_Z_THRESHOLD,
    OVERLAP_DROP_THRESHOLD,
    POSE_ROT_DEG,
    POSE_TRANS_FACTOR,
    STEP_RATIO_THRESHOLD,
    _rotation_angle_deg,
)


SCHEMA_VERSION = 1

# Image-sequence-mode ratio band (matches the adaptive-stride shrink band).
_RATIO_LOWER = 0.75
_RATIO_UPPER = 1.0 / _RATIO_LOWER


def _f(x: Any) -> float | None:
    """Convert a number for JSON: NaN, +inf, -inf, None → None; else float."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _signals_from_evidence(evidence: list[str]) -> list[str]:
    """Map raw evidence strings (`"frame N L.t"` etc.) to sorted primary codes."""
    codes: set[str] = set()
    for e in evidence:
        if any(t in e for t in (" L.t", " L.r", " R.t", " R.r")):
            codes.add("P")
        if " Step" in e:
            codes.add("S")
        if " Cov" in e:
            codes.add("C")
        if " Obs" in e:
            codes.add("O")
    return sorted(codes)


def _segments_from_core_edges(
    core_edges: dict, frame_count: int, seq_frame_numbers: list[int]
) -> list[dict]:
    """Partition `[0..frame_count-1]` at each clustered discontinuity edge.

    Frame `a` of edge `(a, b)` ends the preceding segment; frame `b` starts
    the next. Single-frame segments are emitted as-is.
    """
    edges = sorted(core_edges)
    segments: list[dict] = []
    start = 0
    for a, _b in edges:
        segments.append(
            {
                "start_index": start,
                "end_index": a,
                "start_frame": seq_frame_numbers[start],
                "end_frame": seq_frame_numbers[a],
                "frame_count": a - start + 1,
            }
        )
        start = _b
    last_idx = frame_count - 1
    segments.append(
        {
            "start_index": start,
            "end_index": last_idx,
            "start_frame": seq_frame_numbers[start],
            "end_frame": seq_frame_numbers[last_idx],
            "frame_count": last_idx - start + 1,
        }
    )
    return segments


def reconstruction_results_to_json(all_sequence_results: list[dict]) -> dict:
    """Build a JSON-serializable report from `analyze_reconstruction` output."""
    sequences: list[dict] = []
    for s in all_sequence_results:
        frame_count = s["frame_count"]
        seq_centers = s["seq_centers"]
        seq_quats = s["seq_quats"]
        seq_frame_numbers = s["seq_frame_numbers"]
        seq_image_names = s["seq_image_names"]
        seq_image_indexes = s["seq_image_indexes"]
        extrap_results = s["extrap_results"]
        step_ratios = s["step_ratios"]
        overlap_drops = s["overlap_drops"]
        obs_z_scores = s["obs_z_scores"]
        core_edges = s["core_edges"]
        pair_counts = s["pair_counts"]
        reproj_errors = s.get("core_edge_reproj_errors", {})

        # Successive translations and rotations per landing edge (i-1, i).
        successive_trans = [
            float(np.linalg.norm(seq_centers[i + 1] - seq_centers[i]))
            for i in range(frame_count - 1)
        ]
        successive_rots = [
            _rotation_angle_deg(seq_quats[i], seq_quats[i + 1])
            for i in range(frame_count - 1)
        ]

        extrap_by_idx = {er["seq_idx"]: er for er in extrap_results}

        frames = []
        for i in range(frame_count):
            er = extrap_by_idx.get(i, {})
            landing = i - 1  # edge (i-1, i)
            frames.append(
                {
                    "seq_index": i,
                    "frame_number": seq_frame_numbers[i],
                    "image_name": seq_image_names[i],
                    "left_trans_err": _f(er.get("left_trans_err")),
                    "left_rot_err": _f(er.get("left_rot_err")),
                    "right_trans_err": _f(er.get("right_trans_err")),
                    "right_rot_err": _f(er.get("right_rot_err")),
                    "edge_translation": (
                        _f(successive_trans[landing]) if landing >= 0 else None
                    ),
                    "edge_rotation_deg": (
                        _f(successive_rots[landing]) if landing >= 0 else None
                    ),
                    "step_ratio": (_f(step_ratios[landing]) if landing >= 0 else None),
                    "overlap_drop": (
                        _f(overlap_drops[landing]) if landing >= 0 else None
                    ),
                    "obs_z": _f(obs_z_scores[i]) if i < len(obs_z_scores) else None,
                    "flags": list(er.get("flags", [])) if er else [],
                }
            )

        # The per-frame `flags` list above is not populated by extrap_results
        # itself — flags are computed inline in `analyze_reconstruction`'s
        # print loop. Recompute here so the JSON faithfully mirrors what was
        # printed.
        trans_threshold = s["trans_threshold"]
        rot_threshold = s["rot_threshold"]
        for i, frame in enumerate(frames):
            flags: list[str] = []
            lt = frame["left_trans_err"]
            lr = frame["left_rot_err"]
            rt = frame["right_trans_err"]
            rr = frame["right_rot_err"]
            sr = frame["step_ratio"]
            cv = frame["overlap_drop"]
            oz = frame["obs_z"]
            if lt is not None and lt > trans_threshold:
                flags.append("L.t")
            if lr is not None and lr > rot_threshold:
                flags.append("L.r")
            if rt is not None and rt > trans_threshold:
                flags.append("R.t")
            if rr is not None and rr > rot_threshold:
                flags.append("R.r")
            if sr is not None and sr > STEP_RATIO_THRESHOLD and i > 0:
                flags.append("Step")
            if cv is not None and cv > OVERLAP_DROP_THRESHOLD and i > 0:
                flags.append("Cov")
            if oz is not None and oz < -OBS_Z_THRESHOLD:
                flags.append("Obs")
            frame["flags"] = flags

        discontinuities: list[dict] = []
        for (a, b), evidence in sorted(core_edges.items()):
            img_a = seq_image_indexes[a]
            img_b = seq_image_indexes[b]
            dist = float(np.linalg.norm(seq_centers[b] - seq_centers[a]))
            rot = _rotation_angle_deg(seq_quats[a], seq_quats[b])
            obs_a = obs_z_scores[a] if a < len(obs_z_scores) else None
            obs_b = obs_z_scores[b] if b < len(obs_z_scores) else None
            sr = step_ratios[a] if a < len(step_ratios) else None
            cv = overlap_drops[a] if a < len(overlap_drops) else None
            signals = _signals_from_evidence(evidence)
            discontinuities.append(
                {
                    "edge": [a, b],
                    "frame_a": seq_frame_numbers[a],
                    "frame_b": seq_frame_numbers[b],
                    "image_a": seq_image_names[a],
                    "image_b": seq_image_names[b],
                    "translation": _f(dist),
                    "rotation_deg": _f(rot),
                    "step_ratio": _f(sr),
                    "overlap_drop": _f(cv),
                    "obs_z_a": _f(obs_a),
                    "obs_z_b": _f(obs_b),
                    "shared_points": int(pair_counts.get((img_a, img_b), 0)),
                    "reproj_error_a": _f(reproj_errors.get(img_a)),
                    "reproj_error_b": _f(reproj_errors.get(img_b)),
                    "signals": signals,
                    "confidence": "high" if len(signals) >= 2 else "low",
                }
            )

        segments = _segments_from_core_edges(core_edges, frame_count, seq_frame_numbers)

        sequences.append(
            {
                "pattern": s["sequence"],
                "name": s["sequence_name"],
                "frame_count": frame_count,
                "median_trans": _f(s["median_trans"]),
                "median_rot_deg": _f(s["median_rot"]),
                "pose_trans_threshold": _f(s["trans_threshold"]),
                "pose_rot_threshold": _f(s["rot_threshold"]),
                "frames": frames,
                "discontinuities": discontinuities,
                "segments": segments,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "reconstruction",
        "thresholds": {
            "pose_trans_factor": POSE_TRANS_FACTOR,
            "pose_rot_deg": POSE_ROT_DEG,
            "step_ratio": STEP_RATIO_THRESHOLD,
            "overlap_drop": OVERLAP_DROP_THRESHOLD,
            "obs_z": OBS_Z_THRESHOLD,
        },
        "sequences": sequences,
    }


def _classify_ratio(normalized: float | None) -> str | None:
    """Image-sequence-mode classification of the normalized magnitude ratio."""
    if normalized is None:
        return None
    if normalized < 0.5:
        return "strong deceleration"
    if normalized < _RATIO_LOWER:
        return "deceleration"
    if normalized > 2.0:
        return "strong acceleration"
    if normalized > _RATIO_UPPER:
        return "acceleration"
    return None


def image_sequence_results_to_json(
    per_sequence_results: list[dict],
) -> dict:
    """Build a JSON-serializable report from a list of image-sequence results.

    Each entry of `per_sequence_results` must be a dict with keys
    `pattern`, `name`, and `results` (the list returned by
    `analyze_image_sequence`).
    """
    sequences: list[dict] = []
    for entry in per_sequence_results:
        results = entry["results"]
        samples: list[dict] = []
        for r in results:
            if r.get("superseded"):
                continue
            stride = r["stride"]
            actual = r.get("actual_magnitude_ratio")
            normalized = (
                _f(actual) / stride if _f(actual) is not None and stride > 0 else None
            )
            samples.append(
                {
                    "frame_number": r["frame_number"],
                    "frame_name": r["frame_name"],
                    "next_frame_name": r["next_frame_name"],
                    "stride_frame_name": r.get("stride_frame_name"),
                    "stride": stride,
                    "local_median_magnitude": _f(r.get("local_median_magnitude")),
                    "stride_median_magnitude": _f(r.get("stride_median_magnitude")),
                    "expected_magnitude_ratio": _f(r.get("expected_magnitude_ratio")),
                    "actual_magnitude_ratio": _f(actual),
                    "in_bounds_pct": _f(r.get("in_bounds_pct")),
                    "classification": _classify_ratio(normalized),
                }
            )
        sequences.append(
            {
                "pattern": entry["pattern"],
                "name": entry["name"],
                "sample_count": len(samples),
                "samples": samples,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "image_sequence",
        "thresholds": {
            "ratio_lower": _RATIO_LOWER,
            "ratio_upper": _RATIO_UPPER,
        },
        "sequences": sequences,
    }


def write_report(path: Path, report: dict) -> None:
    """Serialize `report` to `path` as indented JSON. Raises ClickException on IO failure."""
    text = json.dumps(report, indent=2, allow_nan=False)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise click.ClickException(f"Failed to write JSON report to {path}: {e}")
