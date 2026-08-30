# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filling a relaxed member in from its own source clusters.

Provenance: the study's `v2/densify/densifylib.py` (`source_clusters` 109-184
with its input rewritten to take an already-open selection handle,
`extend_member` 214-256, `triangulate_ring` 262-349 folded into
`structure.estimate_points`, `adjust_held` 355-399) and
`v2/densify/densify_run.py` (`fill_in` 195-351) with the per-ring knot probe
and the reference readings removed.

A seed candidate is built from a small number of LARGE-radius clusters, and
every stage of the relaxation reads only that admission.  The clusters the
admission left behind are not worse evidence, they are FINER evidence: a
smaller feature localizes better, and the reason it was dropped is that the
bootstrap needed an unaliased basin.  This stage revisits them, in rings of
decreasing radius, each ring estimated at the current poses and lens and then
adjusted with the LENS HELD so the next ring is resected against refined
structure and the lens is asked once, at the end, on all of the evidence.
"""

from __future__ import annotations

import numpy as np

from . import lens, rings
from .fleet_constants import RING_RATIO_P1
from .structure import estimate_points

#: The reprojection bound a re-estimated ring cluster has to clear, in pixels:
#: the FINAL stage of the adjustment's own settled schedule, which is the bound
#: the adjustment itself would trim the observation at.  Nothing is chosen
#: here: a point the adjustment would throw away is not admitted.
RING_REPROJ_BAR = float(lens.SETTLED_SCHEDULE[-1][0])


def _ev():
    """The evaluation battery's member class, imported on use."""
    import seed_candidate_eval as EV

    return EV


def source_clusters(source, m, frames=None):
    """Every source cluster the member's frames see, with its feature radius.

    ``source`` is the selection handle the run already holds, so no file is
    re-opened and the clusters read here are the clusters the member was drawn
    from.

    The identity between a member observation and a file member is EXACT and
    needs no geometry: both carry the image index (the member's names are the
    file's image table, unchanged) and the feature index, so a member row is a
    file row by ``(image, feature)``.

    The radius is the seed's own reading, the refine radius times the mean of
    the stored affine's two column norms, a cluster taking its widest
    member's, so "radius" means here what it meant when the admission was
    drawn.  The join, the candidate rule, the radius and the octave band each
    candidate falls in are the core's ``analysis::source_clusters`` kernel;
    what is left here is the image-table check and the record's own names.

    Returns a dict, or one carrying ``refused`` where the handle does not
    describe the member's images."""
    from sfmtool._sfmtool.analysis import source_clusters as kernel

    names_f = [str(n).replace("\\", "/") for n in source.image_names]
    if names_f != list(m.names):
        return {"refused": "image table differs from the member's"}
    frames = np.asarray(m.frames if frames is None else frames, np.int64)
    edges = rings.octave_edges(RING_RATIO_P1)
    out = kernel(
        np.ascontiguousarray(np.asarray(source.cluster_starts, np.uint32)),
        np.ascontiguousarray(np.asarray(source.member_images, np.uint32)),
        np.ascontiguousarray(np.asarray(source.member_features, np.uint32)),
        np.ascontiguousarray(np.asarray(source.member_affines, float)),
        float(source.refine_radius),
        len(names_f),
        np.ascontiguousarray(m.obs_i.astype(np.uint32)),
        np.ascontiguousarray(m.obs_f.astype(np.uint32)),
        np.ascontiguousarray(frames.astype(np.uint32)),
        np.ascontiguousarray(np.asarray(edges, float)),
    )
    return {
        "n_file_clusters": int(out["n_file_clusters"]),
        "n_admitted": int(out["n_admitted"]),
        "n_rows_matched": int(out["n_rows_matched"]),
        "n_rows_member": int(len(m.obs_i)),
        "adm_radius": np.asarray(out["admission_radius"], float),
        "adm_floor_px": float(out["admission_floor_px"]),
        "cand": np.asarray(out["candidates"], np.int64),
        "cand_radius": np.asarray(out["candidate_radius"], float),
        "ring": np.asarray(out["candidate_band"], np.int64),
        "edges": edges,
        "obs_cl": np.asarray(out["obs_cluster"], np.int64),
        "obs_img": np.asarray(out["obs_image"], np.int64),
        "obs_feat": np.asarray(out["obs_feature"], np.int64),
        "obs_uv": np.ascontiguousarray(out["obs_uv"], float),
        "obs_shape": np.ascontiguousarray(out["obs_shape"], float),
    }


def extend_member(m, src, add_cl):
    """``(member, {source cluster: new cluster id})`` with ``add_cl`` folded in.

    A member whose observation arrays carry the source rows of the added
    clusters as well as its own.  The added rows are marked OUTSIDE the
    membership, so ``rows`` -- the model's own inlier set, which every
    reference-free reading of the rotation-only member is taken on -- is
    unchanged, while ``rows_all``, the admission every relaxation stage reads,
    now holds them.  Their placeholder points are unit bearings, which is what
    a cluster with no estimate is.

    The new cluster ids continue after the member's own and are ids of the
    EXTENDED MEMBER, not of the source selection."""
    ev = _ev()
    add_cl = np.asarray(add_cl, np.int64)
    take = np.isin(src["obs_cl"], add_cl)
    slot = {int(c): m.n_cl + k for k, c in enumerate(add_cl)}
    new_c = np.array([slot[int(c)] for c in src["obs_cl"][take]], np.int64)
    obs_c = np.concatenate([m.obs_c, new_c])
    obs_i = np.concatenate([m.obs_i, src["obs_img"][take]])
    obs_uv = np.concatenate([m.obs_uv, src["obs_uv"][take]])
    obs_f = np.concatenate([m.obs_f, src["obs_feat"][take]])
    shapes = (
        None
        if m.obs_shape is None
        else np.concatenate([m.obs_shape, src["obs_shape"][take]])
    )
    pts = np.concatenate([m.pts, np.tile(np.array([0.0, 0.0, -1.0]), (len(add_cl), 1))])
    keep = np.concatenate([m.keep_mask(), np.zeros(len(new_c), bool)])
    mx = ev.Member(
        m.idx,
        m.model,
        list(m.names),
        m.camera,
        m.f_eq,
        m.rvec,
        m.tvec,
        m.posed,
        pts,
        (obs_c, obs_i, obs_uv, obs_f),
        shapes=shapes,
        keep=keep,
        dropped=m.dropped,
    )
    return mx, slot


def ring_rows(mx, frames, want):
    """``(uv, slot_i, slot_c)`` over the clusters ``want`` on ``frames``."""
    fslot = {int(f): k for k, f in enumerate(frames)}
    want = np.asarray(want, np.int64)
    cslot = {int(c): k for k, c in enumerate(want)}
    rows = mx.rows_all[np.isin(mx.obs_i[mx.rows_all], frames)]
    rows = rows[np.isin(mx.obs_c[rows], want)]
    slot_i = np.array([fslot[int(i)] for i in mx.obs_i[rows]], np.int64)
    slot_c = np.array([cslot[int(c)] for c in mx.obs_c[rows]], np.int64)
    return mx.obs_uv[rows], slot_i, slot_c


def adjust_held(mx, cam, state, schedule=None, max_iters=30):
    """One adjustment over the state, poses and points only.

    The lens is HELD, both the focal and the spline closed, so the ring that
    just joined is absorbed by the geometry and not by the camera."""
    from sfmtool._sfmtool.geometry import bundle_adjust

    frames = [int(f) for f in state["frames"]]
    clusters = [int(c) for c in state["clusters"]]
    fslot = {f: k for k, f in enumerate(frames)}
    cslot = {c: k for k, c in enumerate(clusters)}
    rows = mx.rows_all[np.isin(mx.obs_i[mx.rows_all], frames)]
    rows = rows[np.isin(mx.obs_c[rows], np.asarray(clusters, np.int64))]
    out = bundle_adjust(
        cam,
        np.ascontiguousarray(np.asarray(state["quats"], float)),
        np.ascontiguousarray(np.asarray(state["trans"], float)),
        np.ascontiguousarray(np.asarray(state["points"], float)),
        np.ascontiguousarray(mx.obs_uv[rows]),
        np.array([fslot[int(i)] for i in mx.obs_i[rows]], np.uint32),
        np.array([cslot[int(c)] for c in mx.obs_c[rows]], np.uint32),
        point_at_infinity=np.ascontiguousarray(np.asarray(state["at_inf"], bool)),
        schedule=list(lens.SETTLED_SCHEDULE if schedule is None else schedule),
        max_iters=int(max_iters),
    )
    resid = np.asarray(out["residual_norms"])
    fin = np.isfinite(resid)
    new = {
        "frames": np.asarray(frames, np.int64),
        "clusters": np.asarray(clusters, np.int64),
        "quats": np.asarray(out["quaternions_wxyz"], float),
        "trans": np.asarray(out["translations"], float),
        "points": np.asarray(out["points"], float),
        "at_inf": np.asarray(state["at_inf"], bool),
    }
    rec = {
        "n_obs": int(len(resid)),
        "resid_finite_frac": float(fin.mean()) if len(resid) else 0.0,
        "reproj_med_px": float(np.median(resid[fin])) if fin.any() else None,
        "reproj_p90_px": float(np.percentile(resid[fin], 90)) if fin.any() else None,
    }
    return new, rec


def fill_in(m, source, cam, state, tol_rad, opts, trace=None):
    """Add the member's own source clusters back, ring by ring.

    Returns ``(member, state, census)``.  The member that comes back is the
    EXTENDED one and the state's cluster ids are its; the census records the
    per-ring counts and the refusal, where there was one."""
    frames = [int(f) for f in state["frames"]]
    src = source_clusters(source, m, frames=frames)
    census = {"rings": []}
    if src is None or src.get("refused"):
        census["refused"] = "no source clusters" if src is None else src["refused"]
        return m, state, census
    if not len(src["adm_radius"]) or not len(src["cand"]):
        census["refused"] = "no candidate clusters"
        return m, state, census
    floor = src["adm_floor_px"]
    edges = src["edges"]
    ring = src["ring"]
    cap = rings.ring_cap(opts)
    census.update(
        {
            "n_file_clusters": src["n_file_clusters"],
            "n_admitted_file": src["n_admitted"],
            "n_candidates": int(len(src["cand"])),
            "adm_floor_px": floor,
            "adm_radius_med_px": float(np.median(src["adm_radius"])),
            "ring_cap": None if cap is None else int(cap),
            "n_rings": len(edges) - 1,
        }
    )
    # Coarsest first, cluster id on a tie: the whole band is admitted in this
    # order, and where a caller has set an absolute count it is what the count
    # cuts, so the ring a member admits is a function of the file alone.
    order_all = rings.band_order(src["cand"], src["cand_radius"])
    added = []
    mx = m
    for r in range(len(edges) - 1):
        want = order_all[ring[order_all] == r][:cap]
        cl_new = src["cand"][want]
        row = {
            "ring": r,
            "edge_hi": float(edges[r]),
            "edge_lo": float(edges[r + 1]),
            "n_in_ring": int((ring == r).sum()),
            "n_taken": int(len(cl_new)),
        }
        if not len(cl_new):
            row["skipped"] = "ring empty"
            census["rings"].append(row)
            continue
        added.append(cl_new)
        mx, slot = extend_member(m, src, np.concatenate(added))
        new_slots = np.array([slot[int(c)] for c in cl_new], np.int64)
        uv, slot_i, slot_c = ring_rows(mx, frames, new_slots)
        pts_new, inf_new, tri = estimate_points(
            cam,
            state["quats"],
            state["trans"],
            uv,
            slot_i,
            slot_c,
            len(new_slots),
            tol_rad,
            bar_px=RING_REPROJ_BAR,
        )
        for k, v in tri.items():
            row[f"tri_{k}"] = v
        state = {
            "frames": state["frames"],
            "clusters": np.concatenate(
                [np.asarray(state["clusters"], np.int64), new_slots]
            ),
            "quats": state["quats"],
            "trans": state["trans"],
            "points": np.concatenate([np.asarray(state["points"], float), pts_new]),
            "at_inf": np.concatenate([np.asarray(state["at_inf"], bool), inf_new]),
        }
        state, barec = adjust_held(mx, cam, state)
        row["ba_reproj_med_px"] = barec["reproj_med_px"]
        row["ba_reproj_p90_px"] = barec["reproj_p90_px"]
        row["ba_n_obs"] = barec["n_obs"]
        row["n_finite"] = int((~np.asarray(state["at_inf"], bool)).sum())
        census["rings"].append(row)
        if trace is not None:
            trace(
                f"    ring {r}: +{row['n_taken']} clusters, "
                f"{row['tri_n_finite']} finite, reproj med "
                f"{row['ba_reproj_med_px']}"
            )
    all_add = np.concatenate(added) if added else np.zeros(0, np.int64)
    mx, _slot = extend_member(m, src, all_add)
    census["n_added"] = int(len(all_add))
    census["n_points_after_fill"] = int(len(state["clusters"]))
    census["n_finite_after_fill"] = int((~np.asarray(state["at_inf"], bool)).sum())
    return mx, state, census
