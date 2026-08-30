# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The global orientation bit, by parallax-weighted cheirality.

Provenance: the study's `sign/signlib.py` `cheirality_readings` (493-556),
reduced to the two readings the chain uses.  The near-versus-far rank statistic
beside them is not carried: it is a second reading of the same bit, and the
weighted observation vote is the one the census settled on.

Pairwise baseline directions determine the constellation only up to the point
reflection ``c -> -c``: negating every centre negates every triangulated depth
and changes nothing else, because the angular least squares is linear in the
centres and its matrix does not contain them.  Structure sitting in FRONT of
the cameras is the one physical statement that separates the two, and a
cluster's cheirality statement is worth exactly the parallax it was measured
with: a cluster inside the member's bound is a bearing whose depth sign is a
coin toss.
"""

from __future__ import annotations

import numpy as np

from .structure import angular_lsq


def angw_bit(m, per_frame, placed, tol):
    """The orientation reading of one constellation.

    Returns a dict: ``angw`` (the parallax-weighted front-minus-behind vote,
    in radians), ``obs_front`` / ``obs_total`` / ``obs_frac`` (the same vote
    unweighted), ``angw_per_obs`` and ``margin_frac`` (the two readings in the
    units they are read in), and the graduation census the pass sees on the
    way.  ``angw < 0`` says the constellation should be reflected.

    The reading is exactly antisymmetric under ``c -> -c``, so one pass
    describes both orientations and the second is arithmetic that is already
    known."""
    frames = sorted(placed)
    rows = {}
    for f in frames:
        cl, rays, _rr = per_frame[f]
        for k, c in enumerate(cl):
            rows.setdefault(int(c), []).append((f, rays[k]))
    n_pts = n_thin = n_behind = 0
    obs_front = obs_total = 0
    angw = 0.0
    for _c, obs in rows.items():
        if len(obs) < 2:
            continue
        dirs = np.stack([m.rot[f].T @ r for f, r in obs])
        cs = np.stack([placed[f] for f, _r in obs])
        widest = float(np.arccos(np.clip(float(np.min(dirs @ dirs.T)), -1.0, 1.0)))
        p = angular_lsq(cs, dirs)
        if p is None:
            continue
        z = np.einsum("ij,ij->i", p[None, :] - cs, dirs)
        front = z > 0
        obs_front += int(front.sum())
        obs_total += int(len(front))
        angw += widest * float(int(front.sum()) - int((~front).sum()))
        if widest <= tol:
            n_thin += 1
        elif not front.all():
            n_behind += 1
        else:
            n_pts += 1
    obs_frac = obs_front / max(1, obs_total)
    return {
        "angw": float(angw),
        "obs_front": int(obs_front),
        "obs_total": int(obs_total),
        "obs_frac": float(obs_frac),
        "angw_per_obs": float(angw / max(1, obs_total)),
        "margin_frac": float(abs(2.0 * obs_frac - 1.0)),
        "pts": int(n_pts),
        "thin": int(n_thin),
        "behind": int(n_behind),
    }
