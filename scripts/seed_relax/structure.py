# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Points from rays, and the arrays the adjustment eats.

Provenance, rewritten to share one estimation loop: the study's
`relaxlib.py` (`triangulate_placed` 440-482, `_angular_lsq` 485-496),
`v2/v2lib.py` (`state_rows` 334-345, `centres_of` 348-351, `world_rays`
354-358, `reprojection` 361-370, `retriangulate_state` 373-470) and
`relax.py` (`build_ba_inputs` 150-183, `stage_adjust` 186-201,
`later_schedule` 204-216, `grow_more` 219-244).  The two per-point loops in
`v2lib.retriangulate_state` (396-439) and `densifylib.triangulate_ring`
(297-338) are the same loop with an optional reprojection bar; both are the
core's `reconstruction::point_estimation` kernel, reached here through
:func:`estimate_points`.

Every estimate is the least-squares midpoint over the observing rays: the
point minimising the squared perpendicular distance to them.  A point whose
widest pair of rays subtends no more than the member's angular bound has no
measured depth and becomes a bearing; so does one that lands behind a camera
that sees it, one seen in a single view, and, where a bar is given, one whose
median observation reprojects past it.
"""

from __future__ import annotations

import math

import numpy as np

from . import quat


def _ev():
    """The evaluation battery's constants and helpers, imported on use."""
    import seed_candidate_eval as EV

    return EV


def angular_lsq(centres, dirs):
    """The point minimizing the squared perpendicular distance to the rays."""
    acc = np.zeros((3, 3))
    rhs = np.zeros(3)
    for c, u in zip(centres, dirs):
        p = np.eye(3) - np.outer(u, u)
        acc += p
        rhs += p @ c
    try:
        return np.linalg.solve(acc, rhs)
    except np.linalg.LinAlgError:
        return None


def triangulate_placed(m, per_frame, placed, tol_rad):
    """``({cluster: point}, census)`` over the clusters the placed frames see.

    A cluster is triangulated by angular least squares over every placed frame
    that sees it, and it is admitted only when the widest pair of its own rays
    subtends more than the member's angular consensus bound (below that the
    rays are the same ray and the depth is not measured) and the point lands
    in front of every observing camera."""
    if len(placed) < 2:
        return {}, {"n_pts": 0, "n_thin": 0, "n_behind": 0, "tri_ang_med_deg": None}
    frames = sorted(placed)
    rows = {}
    for f in frames:
        cl, rays, _r = per_frame[f]
        for k, c in enumerate(cl):
            rows.setdefault(int(c), []).append((f, rays[k]))
    out, ang_all = {}, []
    n_thin = n_behind = 0
    for c, obs in rows.items():
        if len(obs) < 2:
            continue
        dirs = np.stack([m.rot[f].T @ r for f, r in obs])
        cs = np.stack([placed[f] for f, _r in obs])
        cosm = dirs @ dirs.T
        widest = math.acos(max(-1.0, min(1.0, float(np.min(cosm)))))
        if widest <= tol_rad:
            n_thin += 1
            continue
        p = angular_lsq(cs, dirs)
        if p is None:
            continue
        if not (np.einsum("ij,ij->i", p[None, :] - cs, dirs) > 0).all():
            n_behind += 1
            continue
        out[c] = p
        ang_all.append(widest)
    census = {
        "n_pts": len(out),
        "n_thin": n_thin,
        "n_behind": n_behind,
        "tri_ang_med_deg": float(np.degrees(np.median(ang_all))) if ang_all else None,
    }
    return out, census


def state_rows(m, state):
    """The admission rows the state's own frames and clusters index."""
    frames = [int(f) for f in state["frames"]]
    clusters = [int(c) for c in state["clusters"]]
    cslot = {c: k for k, c in enumerate(clusters)}
    rows = m.rows_all[np.isin(m.obs_i[m.rows_all], frames)]
    rows = rows[np.array([int(c) in cslot for c in m.obs_c[rows]], bool)]
    fslot = {f: k for k, f in enumerate(frames)}
    slot_i = np.array([fslot[int(i)] for i in m.obs_i[rows]], np.int64)
    slot_c = np.array([cslot[int(c)] for c in m.obs_c[rows]], np.int64)
    return rows, slot_i, slot_c


def centres_of(state):
    """``(rotations, centres)`` of a state's poses."""
    rot = quat.rots_from_wxyz(np.asarray(state["quats"], float))
    cen = -np.einsum("nji,nj->ni", rot, np.asarray(state["trans"], float))
    return rot, cen


def reprojection(cam, rot, cen, points, at_inf, uv, slot_i, slot_c):
    """Residual norms of every observation under a state, in pixels."""
    p = np.asarray(points, float)[slot_c]
    fin = ~np.asarray(at_inf, bool)[slot_c]
    xc = np.where(
        fin[:, None],
        np.einsum("nij,nj->ni", rot[slot_i], p - cen[slot_i]),
        np.einsum("nij,nj->ni", rot[slot_i], p),
    )
    pred = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)
    return np.linalg.norm(np.asarray(uv, float) - pred, axis=1)


def _verdict_codes():
    """The kernel's verdict table, by name."""
    from sfmtool._sfmtool.reconstruction import VERDICT_CODES

    return VERDICT_CODES


def estimate_points_verdicts(
    cam, quats, trans, uv, slot_i, slot_c, n_points, floor_rad, bar_px=None
):
    """:func:`estimate_points` with the per-cluster verdict codes beside it.

    Returns ``(points, at_inf, census, verdicts)``, where ``verdicts`` carries
    one code per cluster naming the rule that decided it
    (``sfmtool._sfmtool.reconstruction.VERDICT_CODES``)."""
    from sfmtool._sfmtool.reconstruction import estimate_points as kernel

    slot_c = np.ascontiguousarray(np.asarray(slot_c, np.uint32))
    out = kernel(
        uv=np.ascontiguousarray(np.asarray(uv, float)),
        obs_image=np.ascontiguousarray(np.asarray(slot_i, np.uint32)),
        obs_point=slot_c,
        camera=cam,
        quaternions_wxyz=np.ascontiguousarray(np.asarray(quats, float)),
        translations=np.ascontiguousarray(np.asarray(trans, float)),
        n_points=int(n_points),
        floor_rad=float(floor_rad),
        cheirality=True,
        bar_px=None if bar_px is None else float(bar_px),
        few="bearing",
    )
    xyzw = np.asarray(out["xyzw"], float)
    c = out["census"]
    census = {
        # The count the record has always carried is how many clusters an
        # observation NAMED, which is not the kernel's `seen` (every cluster it
        # was given).
        "n_seen": int(len(np.unique(slot_c))),
        "n_finite": int(c["finite"]),
        "n_thin": int(c["thin"]),
        "n_behind": int(c["behind"]),
        "n_single": int(c["few"]),
        "n_reproj_cut": int(c["over_bar"]),
        "n_rows": int(len(slot_c)),
        "n_points": int(n_points),
        "tri_ang_med_deg": c["triangulation_angle_median_deg"],
    }
    return (
        np.ascontiguousarray(xyzw[:, :3]),
        xyzw[:, 3] == 0.0,
        census,
        np.asarray(out["verdicts"], np.uint8),
    )


def estimate_points(
    cam, quats, trans, uv, slot_i, slot_c, n_points, floor_rad, bar_px=None
):
    """Re-estimate every point from its own observations at one geometry.

    The state's poses go in as the quaternions and translations it holds, and
    what comes back is a fresh pair of arrays over ``n_points`` clusters plus
    the census of how each one was decided.  ``bar_px`` is the reprojection
    bound a fresh estimate has to clear before it is called finite; ``None``
    asks for no such bound, which is what a re-estimation of a settled state
    wants (the adjustment has already trimmed those observations).

    A cluster no observation names has no ray and comes back the canonical
    forward bearing, which is the relaxation's ``few = bearing`` setting.

    Returns ``(points, at_inf, census)``."""
    pts, inf, census, _v = estimate_points_verdicts(
        cam, quats, trans, uv, slot_i, slot_c, n_points, floor_rad, bar_px
    )
    return pts, inf, census


def build_ba_inputs(m, placed, pts):
    """The kernel's arrays: the placed frames, the member's whole admission."""
    frames = sorted(placed)
    fslot = {f: k for k, f in enumerate(frames)}
    rows = m.rows_all[np.isin(m.obs_i[m.rows_all], frames)]
    cl = np.unique(m.obs_c[rows])
    cslot = {int(c): k for k, c in enumerate(cl)}
    points = np.zeros((len(cl), 3))
    at_inf = np.zeros(len(cl), bool)
    for c, k in cslot.items():
        if c in pts:
            points[k] = pts[c]
        else:
            d = m.pts[c]
            if not np.isfinite(d).all():
                d = np.array([0.0, 0.0, -1.0])
            points[k] = d / max(1e-12, np.linalg.norm(d))
            at_inf[k] = True
    quats = np.stack([quat.quat_from_rot(m.rot[f]) for f in frames])
    trans = np.stack([-(m.rot[f] @ placed[f]) for f in frames])
    obs_image = np.array([fslot[int(i)] for i in m.obs_i[rows]], np.uint32)
    obs_point = np.array([cslot[int(c)] for c in m.obs_c[rows]], np.uint32)
    return {
        "frames": frames,
        "clusters": cl,
        "quats": quats,
        "trans": trans,
        "points": points,
        "at_inf": at_inf,
        "uv": np.ascontiguousarray(m.obs_uv[rows]),
        "obs_image": obs_image,
        "obs_point": obs_point,
        "rows": rows,
    }


def stage_adjust(m, inp, schedule=None):
    """The shipped staged bundle adjustment, lens held.

    Points marked at infinity stay directions and contribute rotation only,
    which is what keeps a member whose far field is most of its evidence from
    having that field dragged into a depth it never measured."""
    from sfmtool._sfmtool.geometry import bundle_adjust

    kw = {} if schedule is None else {"schedule": schedule}
    return bundle_adjust(
        m.camera,
        np.ascontiguousarray(inp["quats"]),
        np.ascontiguousarray(inp["trans"]),
        np.ascontiguousarray(inp["points"]),
        inp["uv"],
        inp["obs_image"],
        inp["obs_point"],
        point_at_infinity=inp["at_inf"],
        **kw,
    )


def later_schedule(resid, schedule=None):
    """The stages of the shipped schedule a settled state still contains.

    Restarting a converged adjustment at the loosest stage re-admits
    everything the previous round trimmed, so a later round runs only the
    stages whose trim bound is at or below the residual the state actually
    carries, its own 99th percentile, and never fewer than the final stage."""
    full = _ev().SETTLE_SCHEDULE if schedule is None else list(schedule)
    resid = np.asarray(resid, float)
    fin = resid[np.isfinite(resid)]
    if not len(fin):
        return None
    bar = float(np.percentile(fin, 99))
    keep = [s for s in full if s[0] <= bar]
    return keep or [full[-1]]


def grow_more(m, per_frame, placed, pts, min_pts=None):
    """Place any frame the current structure can resect, rotations locked."""
    from sfmtool._sfmtool.geometry import resect_translation

    min_pts = _ev().MIN_RESECT_POINTS if min_pts is None else int(min_pts)
    added = 0
    for f in sorted(per_frame):
        if f in placed:
            continue
        cl, _rays, rows = per_frame[f]
        take = [k for k, c in enumerate(cl) if int(c) in pts]
        if len(take) < min_pts:
            continue
        world = np.array([pts[int(cl[k])] for k in take])
        out = resect_translation(
            m.camera,
            quat.quat_from_rot(m.rot[f]),
            np.ascontiguousarray(world),
            np.ascontiguousarray(m.obs_uv[rows[take]]),
            min_inliers=min_pts,
        )
        if out is None:
            continue
        placed[f] = -(m.rot[f].T @ np.asarray(out["translation"]))
        added += 1
    return added
