# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold and window constants for reconstruction discontinuity analysis.

Shared by `recon_discontinuity.py` (which computes the signals),
`recon_console.py` (which prints the console table) and `report.py` (which
serializes the results).

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

# Pose-extrapolation thresholds.
#
# Translation threshold = POSE_TRANS_FACTOR × the sequence's median successive
# camera motion.  A factor of 1 is too tight: normal trajectory curvature
# produces extrapolation errors on the order of the step size.  A factor of 3
# leaves room for that while still catching real jumps.  The threshold comes
# from the camera steps, not from the extrapolation errors, and as a median it
# is not raised by the few large jumps it is meant to catch.
POSE_TRANS_FACTOR = 3.0
# Rotation threshold, in degrees, fixed for every sequence.  How well rotation
# extrapolates depends on how smooth the trajectory is, not on the rotation
# rate: a quadratic extrapolation from 3 smooth neighbours should predict within
# a few degrees however fast the camera rotates.
POSE_ROT_DEG = 15.0
