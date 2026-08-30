# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Graph to oriented, adjusted state.

Provenance: the study's `v2/v2lib.relax_oriented` (194-328) with its timings
dropped, over `relax.stage_centres` (120-142, the averaging route alone).  The
growth route the study reported beside it is not carried: it stalls wherever
the graph has no frontier.

Every frame the baseline graph connects is placed at once, so no frame is
chained onto a single neighbour and a short baseline cannot set the scale of
everything past it.
"""

from __future__ import annotations

import math

import numpy as np

from . import averaging, graph, orientation, quat, scales, structure

#: Rounds of graduate-and-adjust.
ROUNDS = 2


def stage_centres(m, per_frame, dirs, quality, tol, depths=None):
    """Camera centres in the member's own rotation frame.

    Over the largest connected component of the graph the baselines describe,
    since a component the rest of the graph does not reach carries its own
    unrelated gauge.  The component's own two-view depths are fitted into one
    relative length per edge first, because the directions alone leave a
    straight camera path's spacing undetermined."""
    del m, tol
    frames = graph.largest_component(sorted(per_frame), dirs)
    inside = set(frames)
    keep = {k: v for k, v in dirs.items() if k[0] in inside and k[1] in inside}
    if len(keep) < 3:
        return {}, {"reason": "graph carries no baselines"}
    keys = sorted(keep)
    edge_w = {k: quality[k] for k in keep}
    ell, spread, tied = scales.relative_lengths(keys, depths or {})
    lengths = {k: ell[e] for e, k in enumerate(keys)}
    length_w = {k: quality[k] for e, k in enumerate(keys) if np.isfinite(ell[e])}
    cen, lam, res, read = averaging.centres_by_averaging(
        frames, keep, edge_w, lengths, length_w
    )
    census = dict(read)
    for k, v in averaging.direction_reading(frames, keep, edge_w).items():
        census[f"dir_{k}"] = v
    census["n_edges"] = len(keep)
    census["length_tied_med"] = float(np.median(tied)) if len(tied) else 0.0
    finite = np.isfinite(spread)
    census["length_spread_med"] = (
        float(np.median(spread[finite])) if finite.any() else None
    )
    if cen is None:
        census["reason"] = "averaging did not solve"
        return {}, census
    # A graph that states no length and whose form has more than one null
    # direction does not determine the constellation at all: the spacing would
    # be the solve's own arithmetic.  It is reported, not invented.
    if census["n_free"] and not census["n_lengths"]:
        census["reason"] = "spacing undetermined and no length stated"
        return {}, census
    placed = {f: cen[k] for k, f in enumerate(frames)}
    neg = sum(1 for v in lam.values() if v <= 0)
    census.update(
        {
            "n_neg_lambda": neg,
            "neg_lambda_frac": neg / max(1, len(keep)),
            "edge_res_med": float(np.median(list(res.values()))),
            "lam_med": float(np.median(list(lam.values()))),
        }
    )
    return placed, census


def relax_oriented(m, rounds=ROUNDS, apply_bit=True, min_shared=None):
    """The relaxation of one member, with the orientation read before the
    graduation.

    Returns a dict carrying either ``failed`` or the kept state under ``ba``,
    with the per-round census beside it."""
    out = {"census": {}}
    per_frame, edges, tol = graph.member_graph(m, floor=min_shared)
    out["census"]["n_frames"] = len(per_frame)
    out["census"]["n_edges"] = len(edges)
    out["tol_deg"] = math.degrees(tol)

    dirs, quality, depths = graph.stage_pairs(
        m, per_frame, edges, tol, min_shared=min_shared
    )
    out["census"]["n_baselines"] = len(dirs)
    if len(dirs) < 3:
        out["failed"] = "no baselines"
        return out

    placed, cen_census = stage_centres(m, per_frame, dirs, quality, tol, depths)
    out["census"]["centres"] = cen_census
    if len(placed) < 3:
        out["failed"] = "no centres"
        return out

    bit = orientation.angw_bit(m, per_frame, placed, tol)
    flip = bool(apply_bit and bit["angw"] < 0)
    if flip:
        placed = {f: -c for f, c in placed.items()}
    out["orientation"] = "-" if flip else "+"
    out["bit"] = bit

    rounds_out, states, last_resid = [], [], None
    for r in range(int(rounds)):
        pts, tri = structure.triangulate_placed(m, per_frame, placed, tol)
        added = structure.grow_more(m, per_frame, placed, pts)
        if added:
            pts, tri = structure.triangulate_placed(m, per_frame, placed, tol)
        inp = structure.build_ba_inputs(m, placed, pts)
        ba = structure.stage_adjust(
            m, inp, None if r == 0 else structure.later_schedule(last_resid)
        )
        quats = np.asarray(ba["quaternions_wxyz"])
        trans = np.asarray(ba["translations"])
        rot = quat.rots_from_wxyz(quats)
        placed = {f: -(rot[k].T @ trans[k]) for k, f in enumerate(inp["frames"])}
        last_resid = np.asarray(ba["residual_norms"])
        fin = np.isfinite(last_resid)
        rounds_out.append(
            {
                "round": r,
                "n_frames_added": added,
                "n_finite_pts": tri["n_pts"],
                "n_thin": tri["n_thin"],
                "n_behind": tri["n_behind"],
                "tri_ang_med_deg": tri["tri_ang_med_deg"],
                "n_obs": int(len(last_resid)),
                "resid_finite_frac": (float(fin.mean()) if len(last_resid) else None),
                "reproj_med_px": (
                    float(np.median(last_resid[fin])) if fin.any() else None
                ),
                "reproj_p90_px": (
                    float(np.percentile(last_resid[fin], 90)) if fin.any() else None
                ),
            }
        )
        states.append(
            {
                "frames": np.asarray(inp["frames"], np.int64),
                "clusters": np.asarray(inp["clusters"], np.int64),
                "quats": quats,
                "trans": trans,
                "points": np.asarray(ba["points"]),
                "at_inf": np.asarray(inp["at_inf"], bool),
            }
        )
    # THE ROUND THAT EXPLAINS THE OBSERVATIONS BEST.  A later round adds
    # frames and points, but it can also chase a re-estimation the geometry did
    # not support; the state kept is the one its own admission reprojects
    # through best, which is a reading and not a preference.
    best = min(
        range(len(rounds_out)),
        key=lambda k: (
            rounds_out[k]["reproj_med_px"]
            if rounds_out[k]["reproj_med_px"] is not None
            else float("inf")
        ),
    )
    out["rounds"] = rounds_out
    out["kept_round"] = best
    out["ba"] = states[best]
    out["census"]["n_points_finite"] = int((~states[best]["at_inf"]).sum())
    out["census"]["n_points_total"] = int(len(states[best]["at_inf"]))
    out["census"]["n_placed"] = len(states[best]["frames"])
    out["census"]["graduated_frac"] = out["census"]["n_points_finite"] / max(
        1, out["census"]["n_points_total"]
    )
    return out
