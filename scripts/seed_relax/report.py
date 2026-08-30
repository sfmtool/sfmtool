# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The runaway-frame report: recorded, never cut.

Provenance: the study's `v2/v2lib.runaway_report` (485-551) and
`shape/shapelib.py` (`sensor_of` 135-139, `ref_scales` 397-425), carried
verbatim apart from the study's reference-relative columns.

The shape error of a relaxed member is a per-FRAME quantity, and one frame can
carry most of it. Reading that against a reference is not available to a run;
what IS available is where a frame sits relative to the path the rest of the
member describes.
"""

from __future__ import annotations

import numpy as np

from .structure import centres_of

SEP = chr(92)


def sensor_of(name):
    """The rig sensor a frame belongs to, empty on a single-camera capture."""
    n = str(name).replace(SEP, "/")
    head = n.rsplit("/", 1)[0] if "/" in n else ""
    return head.rsplit("/", 1)[-1] if head else ""


def ref_scales(centres, sensors=None):
    """``(extent, median nearest-neighbour camera spacing)`` of a centre set.

    The spacing is read WITHIN a sensor.  On a rig the nearest camera to a left
    frame is its own right frame, a few centimetres away and simultaneous, so a
    spacing read across the whole set would report the rig's baseline rather
    than how densely the capture sampled its path.  On a single-camera capture
    the two readings are the same."""
    b = np.asarray(centres, float)
    ext = float(np.ptp(b, axis=0).max())
    if len(b) < 2:
        return ext, float("nan")
    groups = {}
    for k, s in enumerate(sensors if sensors is not None else [""] * len(b)):
        groups.setdefault(s, []).append(k)
    near = []
    for ks in groups.values():
        if len(ks) < 2:
            continue
        p = b[np.array(ks, int)]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        near.extend(d.min(axis=1).tolist())
    if not near:
        d = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        near = d.min(axis=1).tolist()
    return ext, float(np.median(near))


def runaway_report(m, state):
    """Per frame, how isolated its centre is inside the member's own path.

    The distance to its nearest other centre within its own sensor, divided by
    the member's own median of that distance.  A frame at 1.0 sits where its
    neighbours do; a frame far above it is somewhere else.  Nothing is trimmed
    on this reading.

    Returns ``(per-frame rows, aggregates)``."""
    frames = [int(f) for f in state["frames"]]
    _rot, cen = centres_of(state)
    names = [m.names[f] for f in frames]
    sensors = [sensor_of(n) for n in names]
    ext, nnsp = ref_scales(cen, sensors)
    groups = {}
    for k, s in enumerate(sensors):
        groups.setdefault(s, []).append(k)
    nn = np.full(len(frames), np.nan)
    for ks in groups.values():
        if len(ks) < 2:
            continue
        ks = np.array(ks, int)
        p = cen[ks]
        dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        nn[ks] = dd.min(axis=1)
    if not np.isfinite(nn).any():
        dd = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        nn = dd.min(axis=1)
    med = float(np.nanmedian(nn)) if np.isfinite(nn).any() else float("nan")
    iso = nn / med if med > 0 else np.full(len(frames), np.nan)
    rows = []
    for k, f in enumerate(frames):
        rows.append(
            {
                "frame": int(f),
                "name": names[k],
                "sensor": sensors[k],
                "nn_dist": float(nn[k]) if np.isfinite(nn[k]) else None,
                "isolation": float(iso[k]) if np.isfinite(iso[k]) else None,
                "cen_from_centroid": float(np.linalg.norm(cen[k] - cen.mean(axis=0))),
            }
        )
    good = iso[np.isfinite(iso)]
    agg = {
        "nn_med": med,
        "extent": ext,
        "nn_spacing_sensor": nnsp,
        "iso_p90": float(np.percentile(good, 90)) if len(good) else None,
        "iso_max": float(good.max()) if len(good) else None,
        "n_iso_over_3": int((good > 3).sum()) if len(good) else 0,
        "n_iso_over_10": int((good > 10).sum()) if len(good) else 0,
        "worst_frame": (
            rows[int(np.nanargmax(iso))]["name"] if np.isfinite(iso).any() else None
        ),
    }
    return rows, agg
