# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold and window constants for reconstruction discontinuity analysis.

Shared by `reconstruction.py` (which computes the signals) and `report.py`
(which serializes the results).

Secondary discontinuity signals complement pose extrapolation — they catch
discontinuities the pose-extrapolation test misses because polynomial
extrapolators can absorb smooth scale/slope changes:

  Step-size ratio : median step changes sharply pre-vs-post an edge
                    (catches zoom/scale shifts)
  Overlap drop    : track covisibility across the edge drops far below the
                    local baseline (catches scene/segment breaks)
  Obs outlier     : per-image observation count is abnormally low
                    (catches "bridge frames" at breaks)
"""

STEP_RATIO_THRESHOLD = 1.5
OVERLAP_DROP_THRESHOLD = 1.8
OBS_Z_THRESHOLD = 2.5

STEP_RATIO_WINDOW = 8
OVERLAP_WINDOW = 16
OVERLAP_BASELINE_WINDOW = 24
OBS_WINDOW = 24

# Pose-extrapolation thresholds.  Translation is sequence-dependent
# (POSE_TRANS_FACTOR × median step length); rotation is fixed.
POSE_TRANS_FACTOR = 3.0
POSE_ROT_DEG = 15.0
