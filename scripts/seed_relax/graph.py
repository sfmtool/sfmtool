# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The admission graph, and the baseline each of its edges carries.

Provenance: the study's `relaxlib.py` (`frame_rays` 61-88, `_local_rays_cached`
91-92, `covisibility` 95-110, `member_graph` 113-118, `pair_rays` 124-142) and
`relax.py` (`stage_pairs` 62-95, `largest_component` 98-117).  The sweep
instruments beside them (`trimmed_rotation`, `tol_anchored_rotation`,
`epipolar_alignment`, `centres_by_growth`) are not carried: nothing in the
shipped chain calls them.  The per-edge solve itself is the
`sfmtool_core::geometry::baseline_direction` kernel, called once for the whole
graph.

A rotation-only member models every frame as a pure rotation of one viewpoint.
That is a resolution statement, not a claim: over a close pair's angular spread
the translating part LOOKS like a rotation, and what the pairwise fit cannot
absorb is the differential part, near points moving more than far ones.  Those
are exactly the observations the model REFUSED, so the graph here is drawn over
the member's whole admission rather than over its own inlier rows.
"""

from __future__ import annotations

import numpy as np


def _ev():
    """The evaluation battery's member machinery, imported on use.

    Imported inside the call so a caller that wants only the numeric
    primitives below never pays for the battery's own imports."""
    import seed_candidate_eval as EV

    return EV


def frame_rays_admission(m):
    """Per posed frame, ``(cluster ids, unit camera rays, rows)``.

    The member's WHOLE admission, which is its own inlier set plus the
    observations the rotation model refused.  A pair fit that reads only the
    inlier set has no parallax residue to anchor on."""
    held = getattr(m, "_frame_rays_all_memo", None)
    if held is not None:
        return held
    ev = _ev()
    out = {}
    order, bnd = ev._csr(m.obs_i[m.rows_all], m.n_img)
    for j in m.frames:
        rows = m.rows_all[order[bnd[j] : bnd[j + 1]]]
        if not len(rows):
            continue
        cl = m.obs_c[rows]
        ordr = np.argsort(cl, kind="stable")
        cl, rows = cl[ordr], rows[ordr]
        uniq, first = np.unique(cl, return_index=True)
        rays = ev._local_rays(m.camera, m.obs_uv[rows[first]])
        out[int(j)] = (uniq, rays, rows[first])
    m._frame_rays_all_memo = out
    return out


def covisibility(m, per_frame, floor):
    """``{(i, j): n_shared}`` over posed frame pairs above a shared floor."""
    ids = sorted(per_frame)
    if len(ids) < 3:
        return {}
    table = np.zeros((len(ids), m.n_cl), bool)
    for k, j in enumerate(ids):
        table[k, per_frame[j][0]] = True
    counts = table.astype(np.int32) @ table.astype(np.int32).T
    out = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            n = int(counts[a, b])
            if n >= floor:
                out[(ids[a], ids[b])] = n
    return out


def member_graph(m, floor=None):
    """``(per_frame, edges, tol_rad)`` over the member's admission.

    ``tol_rad`` is the member's own angular consensus bound: its pixel bar
    carried through its equivalent focal."""
    ev = _ev()
    per_frame = frame_rays_admission(m)
    floor = ev.ROT_CYCLE_MIN_SHARED if floor is None else int(floor)
    return per_frame, covisibility(m, per_frame, floor), ev._rot_tol(m)


def pair_rays(m, per_frame, i, j):
    """``(clusters, a_i, a_j, uv_i, uv_j, u_i, u_j)`` for one frame pair.

    ``a_*`` are camera-frame unit rays, ``u_*`` the same rays rotated into the
    member's world frame by the member's own rotations, ``uv_*`` the stored
    pixels.  Rows the camera model refused are dropped."""
    ci, ri, rows_i = per_frame[i]
    cj, rj, rows_j = per_frame[j]
    shared, ii, jj = np.intersect1d(ci, cj, assume_unique=True, return_indices=True)
    if not len(shared):
        return None
    a, b = ri[ii], rj[jj]
    good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    shared, a, b = shared[good], a[good], b[good]
    uv_i = m.obs_uv[rows_i[ii][good]]
    uv_j = m.obs_uv[rows_j[jj][good]]
    u_i = a @ m.rot[i]  # R_i^T a  == a @ R_i
    u_j = b @ m.rot[j]
    return shared, a, b, uv_i, uv_j, u_i, u_j


def baseline_directions(pairs, tol_rad, rounds=None, keep=None):
    """Every pair's world-frame baseline direction, in one kernel call.

    With the member's rotations held, the baseline ``b = c_j - c_i`` is
    coplanar with every point's two world rays: ``b . (u_i x u_j) = 0``.  The
    normal ``u_i x u_j`` has norm ``sin(parallax angle)``, so a row's weight is
    literally how much baseline that point saw, and rows below the member's own
    angular consensus bound carry none: they are dropped rather than
    down-weighted, because inside the bound the two rays are the same ray.

    ``pairs`` is a sequence of ``(u_i, u_j)``, each an ``(n, 3)`` block of unit
    world rays over that pair's shared clusters.  Returns one entry per pair:
    ``None`` where fewer than three rows cleared the bound, otherwise a dict
    with the unit direction (sign fixed by cheirality), the conditioning of the
    null space, the parallax census of the rows used and the cheirality
    majority."""
    from sfmtool._sfmtool.geometry import baseline_directions as kernel

    ev = _ev()
    rounds = ev.ROT_TRIM_ROUNDS if rounds is None else int(rounds)
    keep = ev.ROT_KEEP_FRACTION if keep is None else float(keep)
    empty = np.zeros((0, 3))
    counts = [len(a) for a, _b in pairs]
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    rows_i = np.concatenate([a for a, _b in pairs]) if pairs else empty
    rows_j = np.concatenate([b for _a, b in pairs]) if pairs else empty
    got = kernel(
        np.ascontiguousarray(rows_i, float),
        np.ascontiguousarray(rows_j, float),
        offsets,
        float(tol_rad),
        int(rounds),
        float(keep),
    )
    out = []
    for e in range(len(pairs)):
        if not bool(got["stated"][e]):
            out.append(None)
            continue
        out.append(
            {
                "d": np.asarray(got["direction"][e], float),
                "n_rows": int(got["n_rows"][e]),
                "n_used": int(got["n_used"][e]),
                "cond": float(got["condition"][e]),
                "par_med_deg": float(got["parallax_median_deg"][e]),
                "par_max_deg": float(got["parallax_max_deg"][e]),
                "cheir_frac": float(got["cheiral_fraction"][e]),
                "resid_med_rad": float(got["residual_median_rad"][e]),
            }
        )
    return out


def baseline_direction(u_i, u_j, tol_rad, rounds=None, keep=None):
    """One pair's baseline direction, through the same kernel."""
    return baseline_directions([(u_i, u_j)], tol_rad, rounds=rounds, keep=keep)[0]


def stage_pairs(m, per_frame, edges, tol, min_shared=None):
    """One baseline direction per covisibility edge, and the depths beside it.

    Returns ``(dirs, quality, depths)``.  An edge is trusted in proportion to
    how many of its rows saw baseline and how cleanly one direction explains
    them: the count of rows past the member's bound, discounted by the null
    space's own conditioning and by the cheirality majority.  All three come
    out of the edge's own solve; nothing is set here.

    ``depths`` carries, per edge, ``(frames, clusters, z)`` for every row past
    the bound that triangulates in front of both cameras, ``z`` being that
    row's distance from that frame in units of the edge's own baseline.  Those
    are the relative scales the directions cannot state."""
    ev = _ev()
    floor = ev.ROT_CYCLE_MIN_SHARED if min_shared is None else int(min_shared)
    keys, rays, clusters = [], [], []
    for (i, j), _n in sorted(edges.items()):
        pk = pair_rays(m, per_frame, i, j)
        if pk is None or len(pk[1]) < floor:
            continue
        cl, _ai, _aj, _uvi, _uvj, u_i, u_j = pk
        keys.append((i, j))
        rays.append((u_i, u_j))
        clusters.append(cl)
    dirs, quality, depths = {}, {}, {}
    solved = baseline_directions(rays, tol)
    for k, cl, (u_i, u_j), bl in zip(keys, clusters, rays, solved):
        if bl is None:
            continue
        dirs[k] = bl["d"]
        quality[k] = bl["n_used"] * min(1.0, bl["cond"] / 3.0) * bl["cheir_frac"]
        row = _edge_depths(k, cl, u_i, u_j, bl["d"], tol)
        if row is not None:
            depths[k] = row
    return dirs, quality, depths


def _edge_depths(key, clusters, u_i, u_j, d, tol_rad):
    """The pair's per-frame depths, over the rows its direction was read from.

    A row inside the bound states no baseline, so its depth is the ratio of two
    quantities the pair did not measure; a row behind either camera contradicts
    the direction's own sign.  Both are dropped."""
    from .scales import two_view_depths

    par = np.arcsin(np.clip(np.linalg.norm(np.cross(u_i, u_j), axis=1), 0.0, 1.0))
    z_i, z_j, _mid = two_view_depths(u_i, u_j, d)
    ok = (par > tol_rad) & (z_i > 0.0) & (z_j > 0.0)
    ok &= np.isfinite(z_i) & np.isfinite(z_j)
    n = int(ok.sum())
    if not n:
        return None
    i, j = key
    return (
        np.concatenate([np.full(n, i, np.int64), np.full(n, j, np.int64)]),
        np.concatenate([clusters[ok], clusters[ok]]),
        np.concatenate([z_i[ok], z_j[ok]]),
    )


def largest_component(frames, dirs):
    """The largest connected component of the graph the baselines describe."""
    adj = {}
    for i, j in dirs:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    seen, comps = set(), []
    for f in frames:
        if f in seen or f not in adj:
            continue
        stack, comp = [f], []
        seen.add(f)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return max(comps, key=len) if comps else []
