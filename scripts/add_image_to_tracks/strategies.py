# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The judging strategies and measurement settings the harness compares.

A *measurement* changes what the operation measures (the template it searches
against, the sub-pixel step, the member localizability gate); a *strategy*
changes only how the measured candidates are judged. Both are keyword sets for
``EditedReconstruction.add_image_to_tracks``.
"""

from __future__ import annotations

MEASUREMENTS = {
    # The default: rendered consensus, sub-pixel step, localizability gate on.
    "default": {},
    "no_localizability": {"max_keypoint_uncertainty": 0.0},
    "no_subpixel": {"subpixel": False},
    # Only where the file stores bitmaps.
    "stored_bitmap": {"template": "stored_bitmap"},
}

_TRACK = {"rule": "track_basis", "min_zncc": 0.0}
_POOLED = {"rule": "pooled_basis", "min_zncc": 0.0}

STRATEGIES = {
    # The operation's defaults, whatever they are.
    "default": {},
    # (b) fixed ZNCC baselines.
    "fixed_0.6": {"rule": "fixed", "min_zncc": 0.6},
    "fixed_0.7": {"rule": "fixed", "min_zncc": 0.7},
    "fixed_0.8": {"rule": "fixed", "min_zncc": 0.8},
    "fixed_0.9": {"rule": "fixed", "min_zncc": 0.9},
    # (a) the leave-one-out basis for three or more references, the pairwise
    # rule for two.
    "track_min/pair_mean": {**_TRACK, "basis": "min", "pair_statistic": "mean"},
    "track_min/pair_min": {**_TRACK, "basis": "min", "pair_statistic": "min"},
    "track_min/pair_max": {**_TRACK, "basis": "min", "pair_statistic": "max"},
    "track_min/pair_mean0.9": {
        **_TRACK,
        "basis": "min",
        "pair_statistic": "mean",
        "pair_factor": 0.9,
    },
    # (c) variants of the basis statistic.
    "track_mad2/pair_mean0.9": {
        **_TRACK,
        "basis": "median_minus_mad",
        "basis_k": 2.0,
        "pair_statistic": "mean",
        "pair_factor": 0.9,
    },
    "track_mad3/pair_mean0.9": {
        **_TRACK,
        "basis": "median_minus_mad",
        "basis_k": 3.0,
        "pair_statistic": "mean",
        "pair_factor": 0.9,
    },
    "track_frac0.9/pair_mean0.9": {
        **_TRACK,
        "basis": "fraction_of_median",
        "basis_fraction": 0.9,
        "pair_statistic": "mean",
        "pair_factor": 0.9,
    },
    "track_frac0.8/pair_mean0.8": {
        **_TRACK,
        "basis": "fraction_of_median",
        "basis_fraction": 0.8,
        "pair_statistic": "mean",
        "pair_factor": 0.8,
    },
    "pooled_mad2": {**_POOLED, "basis": "median_minus_mad", "basis_k": 2.0},
    "pooled_mad3": {**_POOLED, "basis": "median_minus_mad", "basis_k": 3.0},
    "pooled_frac0.8": {**_POOLED, "basis": "fraction_of_median", "basis_fraction": 0.8},
    # (d) with a data-derived positional gate, and with a fixed floor under a
    # data-driven bar.
    "track_min/pair_mean0.9+image_mad": {
        **_TRACK,
        "basis": "min",
        "pair_statistic": "mean",
        "pair_factor": 0.9,
        "position_gate": "image_mad",
        "position_k": 3.0,
        "position_floor_px": 1.0,
    },
    "track_mad3/pair_mean0.9+image_mad": {
        **_TRACK,
        "basis": "median_minus_mad",
        "basis_k": 3.0,
        "pair_statistic": "mean",
        "pair_factor": 0.9,
        "position_gate": "image_mad",
        "position_k": 3.0,
        "position_floor_px": 1.0,
    },
    "pooled_mad3+image_mad": {
        **_POOLED,
        "basis": "median_minus_mad",
        "basis_k": 3.0,
        "position_gate": "image_mad",
        "position_k": 3.0,
        "position_floor_px": 1.0,
    },
    "track_min/pair_mean0.9+floor0.5": {
        **_TRACK,
        "basis": "min",
        "pair_statistic": "mean",
        "pair_factor": 0.9,
        "min_zncc": 0.5,
    },
    # The positional gate's own parameters, under the pooled rule.
    "pooled_mad3+image_mad_k5": {
        **_POOLED,
        "basis": "median_minus_mad",
        "basis_k": 3.0,
        "position_gate": "image_mad",
        "position_k": 5.0,
        "position_floor_px": 1.0,
    },
    "pooled_mad3+image_mad_floor2": {
        **_POOLED,
        "basis": "median_minus_mad",
        "basis_k": 3.0,
        "position_gate": "image_mad",
        "position_k": 3.0,
        "position_floor_px": 2.0,
    },
    "pooled_mad3+max_2px": {
        **_POOLED,
        "basis": "median_minus_mad",
        "basis_k": 3.0,
        "position_gate": "max_px",
        "position_max_px": 2.0,
    },
    "pooled_mad3+max_3px": {
        **_POOLED,
        "basis": "median_minus_mad",
        "basis_k": 3.0,
        "position_gate": "max_px",
        "position_max_px": 3.0,
    },
    "pooled_mad2+image_mad": {
        **_POOLED,
        "basis": "median_minus_mad",
        "basis_k": 2.0,
        "position_gate": "image_mad",
        "position_k": 3.0,
        "position_floor_px": 1.0,
    },
    "fixed_0.7+image_mad": {
        "rule": "fixed",
        "min_zncc": 0.7,
        "position_gate": "image_mad",
        "position_k": 3.0,
        "position_floor_px": 1.0,
    },
    "track_frac0.9/pair_mean0.9+image_mad": {
        **_TRACK,
        "basis": "fraction_of_median",
        "basis_fraction": 0.9,
        "pair_statistic": "mean",
        "pair_factor": 0.9,
        "position_gate": "image_mad",
        "position_k": 3.0,
        "position_floor_px": 1.0,
    },
    # Answers to the recall breakdown: an ascent from the projection where the
    # window's highest peak is on its edge, and a track's own bar beside the
    # pooled one.
    "default+ascend": {"ascend_on_edge": True},
    "pooled_or_track": {
        "rule": "pooled_or_track",
        "track_basis": "fraction_of_median",
        "track_basis_fraction": 0.9,
        "pair_statistic": "mean",
        "pair_factor": 0.9,
    },
    "pooled_or_track+ascend": {
        "rule": "pooled_or_track",
        "track_basis": "fraction_of_median",
        "track_basis_fraction": 0.9,
        "pair_statistic": "mean",
        "pair_factor": 0.9,
        "ascend_on_edge": True,
    },
    "pooled_or_track_min+ascend": {
        "rule": "pooled_or_track",
        "track_basis": "min",
        "pair_statistic": "mean",
        "pair_factor": 0.9,
        "ascend_on_edge": True,
    },
}
