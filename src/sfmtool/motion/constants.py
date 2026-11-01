# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold and window constants for reconstruction discontinuity analysis.

Shared by `recon_discontinuity.py` (which computes the signals),
`_recon_console.py` (which prints the console table) and `report.py` (which
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

# Break verdict.
#
# A listed discontinuity is a break, a place where the solve put two parts of
# the camera path in inconsistent places, when all three of these hold:
#
# - its relative rotation exceeds BREAK_ROTATION_FACTOR × the rotation expected
#   for its frame-number gap, where the expected rotation is the sequence's
#   median rotation per unit of frame number between consecutive images times
#   max(this gap, the median gap);
# - its two images share fewer than BREAK_SHARED_FRACTION × the sequence's
#   median shared 3D point count between consecutive images;
# - the BREAK_WINDOW images up to it and the BREAK_WINDOW images after it share
#   fewer than BREAK_SHARED_FRACTION × the median of that count over all edges.
#
# `specs/cli/reconstruction/motion-command.md` § "Step 4: Break Verdict" gives
# the measurements behind each constant and why each condition is needed: the
# rotation and pair conditions separate breaks in video solves from sound
# pairs that turn sharply or share few points, and the window condition keeps
# a photo set's legitimate large move, whose neighbours a few images away still
# share points, from being called a break.
BREAK_ROTATION_FACTOR = 4.0
BREAK_SHARED_FRACTION = 0.2
BREAK_WINDOW = 8
