# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The image-sequence-mode normalized-magnitude-ratio band.

`sfm motion` divides the stride-to-local flow magnitude ratio by the stride;
for smooth motion the result sits near 1.0. One log-symmetric band around 1.0
drives three things — the adaptive stride's shrink decision, the flagging of a
frame into the console summary, and the `classification` field of the JSON
report — so the band edges and the classifier live here, once, rather than
being restated by each of them. `RATIO_UPPER` is derived from `RATIO_LOWER`,
which is what keeps the band multiplicatively symmetric.

See `specs/cli/reconstruction/motion-command.md`.
"""

#: Lower edge of the in-band ratio interval.
RATIO_LOWER = 0.75
#: Upper edge, the multiplicative mirror of the lower one (≈ 1.33).
RATIO_UPPER = 1.0 / RATIO_LOWER
#: The band as it is written in user-facing messages.
BAND_TEXT = f"[{RATIO_LOWER:.2f}, {RATIO_UPPER:.2f}]"


def out_of_band(normalized: float) -> bool:
    """Whether a normalized ratio falls outside the in-band interval."""
    return normalized < RATIO_LOWER or normalized > RATIO_UPPER


def classify_ratio(normalized: float | None) -> str | None:
    """Describe where a normalized magnitude ratio sits, or `None` if in band.

    `None` in, `None` out, so a sample point with no ratio carries no
    classification into the JSON report.
    """
    if normalized is None:
        return None
    if normalized < 0.5:
        return "strong deceleration"
    if normalized < RATIO_LOWER:
        return "deceleration"
    if normalized > 2.0:
        return "strong acceleration"
    if normalized > RATIO_UPPER:
        return "acceleration"
    return None
