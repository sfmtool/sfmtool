# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The six stages of the relaxation, on one member.

Provenance: the study's `v2/v2lib.run_pipeline` (557-692) with the fill-in
inserted from `v2/densify/densify_run.run_member` (354-489), with the scoring,
the reference readings, the per-stage timings and the isolation switches
removed.  What is left is the chain as it ships: gate, relax, fill in, release,
re-estimate, report.

Nothing here reads a clock into the record.  The per-stage census is a count of
what the stage decided, not how long it took.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import fill, lens, relaxation, report, structure
from .fleet_constants import SETTLING_FINITE_COUNT


def _ev():
    """The evaluation battery's constants, imported on use."""
    import seed_candidate_eval as EV

    return EV


@dataclass
class RelaxResult:
    """One member's relaxation, whole.

    ``member`` is the EXTENDED member the fill-in produced, whose cluster ids
    the state indexes; ``camera`` the lens the state settled under; ``state``
    the poses, points and bearings; ``census`` every stage's own reading; and
    ``runaway_frames`` the per-frame isolation rows.  A refusal carries its
    reason and nothing else."""

    refused: str = None
    member: object = None
    camera: object = None
    state: dict = None
    lens: dict = None
    census: dict = field(default_factory=dict)
    runaway_frames: list = field(default_factory=list)

    @property
    def ok(self):
        return self.refused is None and self.state is not None


def _trace(opts):
    if opts is not None and getattr(opts, "trace", False):
        return lambda msg: print(msg, flush=True)
    return None


def run_member(m, source, opts=None):
    """Relax one rotation-only member into a finite sibling.

    ``m`` is the member's own arrays, ``source`` the selection handle the run
    already holds (the fill-in reads its clusters from it), ``opts`` an
    :class:`seed_relax.Options`.  The member is mutated in place by the lens
    stage, exactly as the study's chain mutates it, so a caller that needs the
    original arrays afterwards holds its own copy.

    Returns a :class:`RelaxResult`."""
    from . import options as default_options

    opts = default_options() if opts is None else opts
    say = _trace(opts)
    ev = _ev()
    census = {}
    base_cam = m.camera
    fld = lens.observed_field(m)
    thetas = lens.sample_thetas(fld.get("theta_p99_rad", 0.0))
    census.update(
        {
            "family": lens.family_of(base_cam.model),
            "camera_model": str(base_cam.model),
            "f_chart_base": fld.get("f0"),
            "f_eq_base": float(m.f_eq) if m.f_eq else None,
            "theta_p99_deg": fld.get("theta_p99_deg"),
            "theta_max_deg": fld.get("theta_max_deg"),
            "n_posed": int(m.posed.sum()),
            "n_rows_member": int(len(m.rows)),
            "n_rows_admission": int(len(m.rows_all)),
        }
    )

    # -- stage 1: the lens on the bearings ---------------------------------
    early, why = lens.gate_early_release(base_cam)
    census["early_gate"] = why
    cam = base_cam
    if early:
        vrec, cam_b, rots = lens.rot_lens_ba(
            m, knots=lens.SEED_KNOTS, rowset="admission", opt_bspline=True
        )
        if cam_b is not None:
            lens.swap_camera(m, cam_b, thetas=thetas)
            lens.apply_rotations(m, rots[0], rots[1])
            cam = cam_b
            census["early_release"] = "applied"
            census["early_reproj_med_px"] = vrec.get("reproj_med_px")
            census["early_rot_delta_med_deg"] = vrec.get("rot_delta_med_deg")
        else:
            census["early_release"] = f"refused:{vrec.get('refused')}"
    else:
        census["early_release"] = "held"
    if say:
        say(f"  relax: early release {census['early_release']} ({why})")

    # -- stage 2: the relaxation -------------------------------------------
    out = relaxation.relax_oriented(m)
    if out.get("failed"):
        census["n_frames_graph"] = out["census"].get("n_frames")
        census["n_edges"] = out["census"].get("n_edges")
        census["n_baselines"] = out["census"].get("n_baselines")
        return RelaxResult(refused=out["failed"], census=census)
    bit = out["bit"]
    census.update(
        {
            "orientation": out["orientation"],
            "bit_angw_per_obs": bit["angw_per_obs"],
            "bit_margin_frac": bit["margin_frac"],
            "n_frames_graph": out["census"]["n_frames"],
            "n_edges": out["census"]["n_edges"],
            "n_baselines": out["census"]["n_baselines"],
            "n_placed": out["census"]["n_placed"],
            "tol_deg_base": out["tol_deg"],
            "kept_round": out["kept_round"],
            "n_points_relax": out["census"]["n_points_total"],
            "n_finite_relax": out["census"]["n_points_finite"],
        }
    )
    kept = out["rounds"][out["kept_round"]]
    census["tri_ang_med_deg"] = kept["tri_ang_med_deg"]
    census["reproj_relax_med_px"] = kept["reproj_med_px"]
    for k, v in (out["census"].get("centres") or {}).items():
        census[f"cen_{k}"] = v
    state = out["ba"]
    if say:
        say(
            f"  relax: {census['n_placed']} frames placed, "
            f"{census['n_finite_relax']} finite of "
            f"{census['n_points_relax']}, orientation {out['orientation']}"
        )

    # -- stage 3: the fill-in ----------------------------------------------
    f_eq = lens.equivalent_focal(cam, thetas)
    if not (np.isfinite(f_eq) and f_eq > 0):
        f_eq = float(m.f_eq)
    tol_rad = ev.E_TOL_PX / f_eq
    mx, state, fill_census = fill.fill_in(m, source, cam, state, tol_rad, opts, say)
    census["fill"] = fill_census
    if say:
        say(
            f"  relax: filled in +{fill_census.get('n_added', 0)} clusters, "
            f"{fill_census.get('n_finite_after_fill')} finite"
        )

    # -- stage 4: the late lens release ------------------------------------
    n_finite = int((~np.asarray(state["at_inf"], bool)).sum())
    family = lens.family_of(base_cam.model)
    knots = opts.knots_fisheye if family == "fisheye" else opts.knots_pinhole
    late = {"family": family, "knots": int(knots), "finite_count": n_finite}
    if family == "fisheye" or n_finite >= SETTLING_FINITE_COUNT:
        crec, cam_c, state_c = lens.release_at_knots(mx, state, cam, knots, thetas)
        late["refit_resid_px"] = crec.get("refit_resid_px")
        late["reproj_med_px"] = crec.get("reproj_med_px")
        if cam_c is not None:
            cam, state = cam_c, state_c
            late["applied"] = True
        else:
            late["applied"] = False
            late["reason"] = f"refused:{crec.get('refused')}"
    else:
        late["applied"] = False
        late["bar"] = int(SETTLING_FINITE_COUNT)
        late["reason"] = (
            f"finite count {n_finite} below the settling bar {SETTLING_FINITE_COUNT}"
        )
    census["late_release"] = late
    if say:
        say(
            f"  relax: late release "
            f"{'applied' if late['applied'] else late.get('reason', 'held')} "
            f"at {knots} knots"
        )

    # -- stage 5: every point re-estimated ---------------------------------
    f_eq_final = lens.equivalent_focal(cam, thetas)
    if not (np.isfinite(f_eq_final) and f_eq_final > 0):
        f_eq_final = float(m.f_eq)
    tol_final = ev.E_TOL_PX / f_eq_final
    rows, slot_i, slot_c = structure.state_rows(mx, state)
    rot, cen = structure.centres_of(state)
    uv = mx.obs_uv[rows]
    was_finite = ~np.asarray(state["at_inf"], bool)
    before = structure.reprojection(
        cam, rot, cen, state["points"], state["at_inf"], uv, slot_i, slot_c
    )
    pts, at_inf, retri = structure.estimate_points(
        cam,
        state["quats"],
        state["trans"],
        uv,
        slot_i,
        slot_c,
        len(state["clusters"]),
        tol_final,
    )
    state = dict(state)
    state["points"] = pts
    state["at_inf"] = at_inf
    after = structure.reprojection(cam, rot, cen, pts, at_inf, uv, slot_i, slot_c)
    retri.update(
        {
            "floor_deg": math.degrees(float(tol_final)),
            "was_finite": int(was_finite.sum()),
            "demoted": int((was_finite & at_inf).sum()),
            "promoted": int(((~was_finite) & (~at_inf)).sum()),
            "reproj_before_med_px": _med(before),
            "reproj_med_px": _med(after),
            "reproj_p90_px": _p90(after),
            "n_obs_final": int(len(after)),
        }
    )
    census["retri"] = retri
    census["f_eq_final"] = float(f_eq_final)
    census["tol_final_deg"] = math.degrees(float(tol_final))
    census["n_points_final"] = int(len(at_inf))
    census["n_finite_final"] = int((~at_inf).sum())
    census["n_infinity_final"] = int(at_inf.sum())

    # -- stage 6: the runaway report ---------------------------------------
    frows, agg = report.runaway_report(mx, state)
    census["runaway"] = agg
    if say:
        say(
            f"  relax: {census['n_finite_final']} finite of "
            f"{census['n_points_final']}, reproj med "
            f"{retri['reproj_med_px']}, worst frame {agg['worst_frame']} at "
            f"isolation {agg['iso_max']}"
        )

    return RelaxResult(
        member=mx,
        camera=cam,
        state=state,
        lens=lens_block(cam, thetas),
        census=census,
        runaway_frames=frows,
    )


def lens_block(cam, thetas):
    """The camera the member shipped, in the manifest's own shape.

    ``None`` where no release was adopted: the member then carries its base
    camera, which has no spline to describe, and the manifest states it as the
    plain chart focal it is."""
    d_max = lens.spline_domain(cam)
    if d_max is None:
        return None
    p = cam.parameters
    n = int(p.get("bspline_coeff_count", 0))
    return {
        "f_chart": lens.base_focal(cam),
        "coeffs": [float(p.get(f"bspline_c{i}", 0.0)) for i in range(n)],
        "d_max": float(d_max),
        "f_eq": float(lens.equivalent_focal(cam, thetas)),
        "model": str(cam.model),
        "accepted": True,
    }


def _med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def _p90(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 90)) if len(v) else None
