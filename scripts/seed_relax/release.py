# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""What the writer and the manifest need from a relaxed member.

Row assembly follows the study's `write_relaxed_full.py` (127-213): the
observations are the extended member's admission on the placed frames, the
finite points go in as positions and the bearings as unit directions, and a
cluster the adjustment could not pin down still states a direction, so nothing
is silently shed.  The caller writes the artifact; this module only states the
arrays.

The bearings go in at ``w = 1`` and are re-stated homogeneously at ``w = 0``
after the write, which is what the rotation-only writer already does; the world
rotation the writer applies is a rotation, so a unit direction reaches the
canonical frame still a unit direction.
"""

from __future__ import annotations

import numpy as np

from . import quat, structure

#: The pixel bar the manifest's inlier fraction is read at.  The same bar every
#: hypothesis is scored on, so the relaxed member's number is commensurable
#: with its rotation-only source's.
INLIER_BAR_PX = 2.0


def relaxed_arrays(result, data):
    """``(data, points, keep, residuals, at_infinity)`` for the writer.

    ``data`` is the observation dict the hypothesis was solved on; what comes
    back carries the EXTENDED member's observation arrays in its place, with
    the per-cluster columns of the original dropped, since the fill-in added
    clusters they do not describe.

    ``keep`` marks the state's own rows whose observation the final camera can
    project and whose cluster is writable.  ``points`` is finite positions and
    unit bearings in one array, indexed by the extended member's cluster ids,
    and ``at_infinity`` says which is which."""
    m, state, cam = result.member, result.state, result.camera
    rows, slot_i, slot_c = structure.state_rows(m, state)
    rot, cen = structure.centres_of(state)
    resid = structure.reprojection(
        cam,
        rot,
        cen,
        state["points"],
        state["at_inf"],
        m.obs_uv[rows],
        slot_i,
        slot_c,
    )

    n_cl = int(m.n_cl)
    clusters = np.asarray(state["clusters"], np.int64)
    pts = np.full((n_cl, 3), np.nan)
    at_inf = np.zeros(n_cl, bool)
    writable = np.zeros(n_cl, bool)
    p = np.asarray(state["points"], float)
    inf = np.asarray(state["at_inf"], bool)
    norm = np.linalg.norm(p, axis=1)
    unit = p / np.maximum(norm, 1e-300)[:, None]
    ok = np.isfinite(p).all(axis=1) & np.where(inf, norm > 1e-12, True)
    pts[clusters[ok]] = np.where(inf[ok, None], unit[ok], p[ok])
    at_inf[clusters[ok]] = inf[ok]
    writable[clusters[ok]] = True

    keep = np.zeros(len(m.obs_c), bool)
    good = np.isfinite(resid) & writable[m.obs_c[rows]]
    keep[rows[good]] = True
    res = np.full(len(m.obs_c), np.inf)
    res[rows[good]] = resid[good]

    out = {k: v for k, v in data.items() if k not in _STALE_KEYS}
    out.update(
        {
            "obs_c": np.asarray(m.obs_c, np.int64),
            "obs_i": np.asarray(m.obs_i, np.int64),
            "obs_f": np.asarray(m.obs_f, np.int64),
            "obs_uv": np.ascontiguousarray(m.obs_uv, float),
            "n_cl": n_cl,
        }
    )
    if m.obs_shape is not None:
        out["obs_shape"] = np.ascontiguousarray(m.obs_shape, float)
    return out, pts, keep, res, at_inf


#: Columns of the source observation dict that describe the ORIGINAL cluster
#: set and would be read against the extended one.  Dropped rather than padded:
#: the relaxed release is release-grade, and nothing on that path reads them.
_STALE_KEYS = ("adm_rank", "cl_quality", "obs_warp", "obs_ref")


def relaxed_poses(result, n_images):
    """``(rvec, tvec, posed)`` over the whole image table.

    The state's poses are in the member's own rotation frame, which is the
    frame the hypothesis was solved in, so they go in as they are.  An image
    the relaxation did not place keeps an identity rotation and a zero
    translation and is not posed."""
    state = result.state
    frames = [int(f) for f in state["frames"]]
    rot = quat.rots_from_wxyz(np.asarray(state["quats"], float))
    rvec = np.zeros((int(n_images), 3))
    tvec = np.zeros((int(n_images), 3))
    posed = np.zeros(int(n_images), bool)
    for k, f in enumerate(frames):
        rvec[f] = quat.rotvec_from_rot(rot[k])
        tvec[f] = np.asarray(state["trans"], float)[k]
        posed[f] = True
    return rvec, tvec, posed


def alive_clusters(pts, keep, obs_c):
    """The clusters the writer will keep: at least two surviving observations.

    Restated here so a caller can map a written point row back to the cluster
    it came from without re-deriving the writer's own rule."""
    counts = np.bincount(np.asarray(obs_c)[np.asarray(keep, bool)], minlength=len(pts))
    return np.nonzero(counts >= 2)[0]


def relaxed_res(result, data, reach, f_source=None):
    """The `res` dict a hypothesis is committed through.

    ``reach`` is the capture-level coverage the caller measures; everything
    else is the relaxation's own.  The member declares no focal scan (there is
    none on this model), carries the `relaxed` flag, and therefore reads as
    unqualified, exactly as its rotation-only source does."""
    data_x, pts, keep, res, at_inf = relaxed_arrays(result, data)
    rvec, tvec, posed = relaxed_poses(result, len(data["names"]))
    alive = alive_clusters(pts, keep, data_x["obs_c"])
    finite = res[keep]
    out = {
        "f_released": float(result.census["f_eq_final"]),
        "inl": float((finite < INLIER_BAR_PX).mean()) if len(finite) else 0.0,
        "posed": posed,
        "posed_full": posed,
        "rvec_full": rvec,
        "tvec_full": tvec,
        "reach": float(reach),
        "flags": ["relaxed"],
        "kept": int(posed.sum()),
        # No focal scan runs on a relaxed member: its lens is read by the
        # releases, not by a scan, so it declares no scan spread.
        "spread": 0.0,
        "lens": result.lens,
        "data": data_x,
        "release_pts": pts,
        "keep": keep,
        "res_obs": res,
        "at_inf": at_inf,
        "n_points_written": int(len(alive)),
        "n_points_finite": int(result.census["n_finite_final"]),
        "n_points_infinity": int(result.census["n_infinity_final"]),
        # A single-view bearing has no second observation, so the writer drops
        # it; the record states how many the state carried past the writer.
        "n_points_dropped_by_writer": int(len(pts) - len(alive)),
    }
    if f_source is not None:
        out["f_source"] = f_source
    return out


def relaxation_block(result):
    """The `relaxation` census the manifest entry carries."""
    c = result.census
    if result.refused is not None:
        return {"refused": result.refused, **_head(c)}
    fill = c.get("fill", {})
    return {
        **_head(c),
        "n_placed": c.get("n_placed"),
        "tri_ang_med_deg": c.get("tri_ang_med_deg"),
        "n_finite_relax": c.get("n_finite_relax"),
        "reproj_relax_med_px": c.get("reproj_relax_med_px"),
        "fill": {
            "n_candidates": fill.get("n_candidates"),
            "adm_floor_px": fill.get("adm_floor_px"),
            "ring_cap": fill.get("ring_cap"),
            "n_added": fill.get("n_added"),
            "refused": fill.get("refused"),
            "rings": [
                {
                    "ring": r.get("ring"),
                    "taken": r.get("n_taken"),
                    "finite": r.get("tri_n_finite"),
                    "thin": r.get("tri_n_thin"),
                    "behind": r.get("tri_n_behind"),
                    "cut": r.get("tri_n_reproj_cut"),
                }
                for r in fill.get("rings", [])
            ],
        },
        "late_release": c.get("late_release"),
        "retri": c.get("retri"),
        "runaway": {**(c.get("runaway") or {}), "frames": result.runaway_frames},
    }


def _head(c):
    return {
        "early_release": c.get("early_release"),
        "orientation": c.get("orientation"),
        "bit_angw_per_obs": c.get("bit_angw_per_obs"),
        "bit_margin_frac": c.get("bit_margin_frac"),
        "n_frames_graph": c.get("n_frames_graph"),
        "n_edges": c.get("n_edges"),
        "n_baselines": c.get("n_baselines"),
    }


def tool_options(result, idx, paired_with=None, scope=None, f_source=None):
    """The `.sfmr` metadata the relaxed release carries, all strings."""
    c = result.census
    lens_d = result.lens or {}
    late = c.get("late_release") or {}
    fill = c.get("fill") or {}
    runaway = c.get("runaway") or {}
    opts = {
        "hypothesis_index": str(idx),
        "structure_model": "relaxed",
        "focal_released_px": f"{float(c['f_eq_final']):.3f}",
        "camera_model": lens_d.get("model", c.get("camera_model")),
        "early_release": str(c.get("early_release")),
        "orientation": str(c.get("orientation")),
        "bit_angw_per_obs": f"{float(c.get('bit_angw_per_obs', 0.0)):.6f}",
        "bit_margin_frac": f"{float(c.get('bit_margin_frac', 0.0)):.6f}",
        "fill_rings": ";".join(
            f"{r.get('ring')}:{r.get('n_taken')}/{r.get('tri_n_finite')}/"
            f"{r.get('tri_n_thin')}/{r.get('tri_n_behind')}/"
            f"{r.get('tri_n_reproj_cut')}"
            for r in fill.get("rings", [])
            if r.get("n_taken")
        ),
        "fill_added": str(fill.get("n_added", 0)),
        "late_release": (
            f"applied@{late.get('knots')}"
            if late.get("applied")
            else f"held:{late.get('reason', 'not run')}"
        ),
        "points_finite": str(c.get("n_finite_final")),
        "points_infinity": str(c.get("n_infinity_final")),
        "runaway_worst_frame": str(runaway.get("worst_frame")),
        "runaway_iso_max": f"{float(runaway.get('iso_max') or 0.0):.3f}",
        "confidence_flags": "relaxed",
        "qualified": "False",
    }
    if lens_d:
        opts["focal_chart_px"] = f"{float(lens_d['f_chart']):.3f}"
        opts["bspline"] = ",".join(f"{c_:.8f}" for c_ in lens_d["coeffs"])
        opts["bspline_d_max"] = f"{float(lens_d['d_max']):.8f}"
    if paired_with is not None:
        opts["paired_with"] = str(int(paired_with))
    if scope is not None:
        opts["scope"] = str(scope)
    if f_source is not None:
        opts["focal_source"] = str(f_source)
    return opts
