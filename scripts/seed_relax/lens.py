# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The lens, read on bearings and read again on the relaxed state.

Provenance: the study's `lens/lenslib.py` (`family_of` / `d_of_theta` /
`promote` / `base_focal` / `spline_domain` 97-151, `observed_field` 156-194,
`sample_thetas` / `radial_map` / `equivalent_focal` 199-255, `swap_camera` /
`rot_lens_inputs` / schedules / `rot_lens_ba` / `apply_rotations` 299-467,
`final_release` 473-546) and `v2/densify/densifylib.py` (`refit_knots`
405-436, `release_at_knots` 439-464).  The gate is `v2/v2lib.gate_decision`
(171-188) with its isolation-arm override dropped.  The study's reference
cameras, its map comparison and its TSV writers are not carried: they read a
reference the run does not have.

`min_obs` is the kernel's own default here.  The study lifted the degenerate
floor to zero because a directions-only scene has no finite survivors by
construction and so exited degenerate at any positive floor; the kernel now
counts every survivor, so the floor stands and the finite-residual refusal
below is what catches an unusable solve.

The base camera is never promoted in place: `promote` returns a new camera on
the base's radial-spline model with an all-zero spline, which is the same map
bit for bit, and what it buys is the coefficient slots the release needs.  The
mechanics are restated here from `specs/formats/sfmtool-camera-models.md`
rather than imported, because importing the bootstrap has argv and environment
side effects.
"""

from __future__ import annotations

import numpy as np

from . import quat

#: The two radial-spline models and the parameter naming their domain end, by
#: the BASE model they promote.
SPLINE_MODEL = {
    "pinhole": ("SFMTOOL_PINHOLE", "bspline_rho_max"),
    "fisheye": ("SFMTOOL_FISHEYE", "bspline_theta_max"),
}
#: The seed's own knot count (`exp_fast_seed.BSPLINE_KNOTS`).
SEED_KNOTS = 2
#: Headroom past the outermost observation for the spline's domain end, so the
#: basis covers the field it was fitted on rather than ending inside it.
DOMAIN_HEADROOM = 1.02
#: The base chart the early release is taken on.  A fisheye chart is a nominal
#: design target a real lens misses by shape no focal absorbs, so a release
#: over that field corrects a chart that is wrong by construction.  A
#: perspective chart is what a rectilinear lens is built to be, and a member on
#: one is either nearly right already or wrong in a way its admission cannot
#: tell apart from a focal error.
RELEASE_CHART = "fisheye"

FISHEYE_MODELS = {
    "EQUIDISTANT_FISHEYE",
    "SFMTOOL_FISHEYE",
    "SIMPLE_RADIAL_FISHEYE",
    "OPENCV_FISHEYE",
    "FOV",
    "THIN_PRISM_FISHEYE",
}

#: Samples of the radial map over the comparison band.  A map this smooth needs
#: nothing like this many; the count is set so the readings are taken on a
#: field sampled uniformly in ANGLE rather than on the observations' own
#: crowding.
N_SAMPLES = 257
#: Azimuths the radial map is averaged over.  A candidate camera carries one
#: focal, so its map is radially symmetric and any single azimuth would do;
#: averaging costs nothing and reads an anisotropic lens as its mean radial
#: map rather than as whichever axis a sample happened to lie on.
MAP_AZIMUTHS = np.array([0.0, 0.25, 0.5, 0.75]) * np.pi


def _ev():
    """The evaluation battery's constants, imported on use."""
    import seed_candidate_eval as EV

    return EV


def full_schedule():
    """The adjustment's own shipped schedule, opening permissive.

    A rotation-only state's whole-admission reprojection reaches tens of
    pixels, so the loosest stage is where the lens read on bearings starts."""
    return list(_ev().SETTLE_SCHEDULE)


#: A state that has already settled starts at the schedule's later stages.
SETTLED_SCHEDULE = [(12.0, 2.0), (4.0, 1.0)]


def family_of(model):
    """``"fisheye"`` or ``"pinhole"``: which radial chart a model lives on."""
    return "fisheye" if str(model) in FISHEYE_MODELS else "pinhole"


def d_of_theta(theta, family):
    """The base model's radial coordinate at incidence ``theta``: the angle
    itself on a fisheye chart, ``tan(theta)`` on a perspective one."""
    return np.asarray(theta, float) if family == "fisheye" else np.tan(theta)


def gate_early_release(cam):
    """``(release, why)`` for one member's base camera.

    The chart the member carries IS the reading: a release corrects a chart
    that is wrong by construction, and on the other chart it has nothing to
    correct and a parallax-heavy admission to be misled by."""
    fam = family_of(cam.model)
    take = fam == RELEASE_CHART
    return take, (
        f"base chart {fam} ({cam.model}) "
        f"{'is' if take else 'is not'} the chart the release corrects"
    )


def promote(base_cam, f, coeffs, d_max):
    """``base_cam`` at focal ``f`` on its base's radial-spline model.

    An all-zero ``coeffs`` is the base model's own map -- projection, inverse
    and pixel Jacobian alike -- so the promotion by itself moves nothing; what
    it does is allocate the coefficient slots the release needs."""
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    fam = family_of(base_cam.model)
    model, d_key = SPLINE_MODEL[fam]
    pp = base_cam.principal_point
    cc = np.asarray(coeffs, dtype=np.float64)
    params = {
        "focal_length": float(f),
        "principal_point_x": float(pp[0]),
        "principal_point_y": float(pp[1]),
        d_key: float(d_max),
        "bspline_coeff_count": float(len(cc)),
    }
    for i, c in enumerate(cc):
        params[f"bspline_c{i}"] = float(c)
    return CameraIntrinsics.from_dict(
        {
            "model": model,
            "width": int(base_cam.width),
            "height": int(base_cam.height),
            "parameters": params,
        }
    )


def base_focal(cam):
    """The camera's own focal parameter."""
    return float(cam.parameters["focal_length"])


def spline_domain(cam):
    """The spline domain end of a spline camera, or ``None``."""
    p = cam.parameters
    for k in ("bspline_rho_max", "bspline_theta_max"):
        if k in p:
            return float(p[k])
    return None


def observed_field(m, rows=None, cam=None):
    """The radial field the member's observations cover.

    Radii are measured from the camera's own principal point and the angles
    come from ``cam``, so a field computed once on the base camera is one fixed
    set of incidence angles that every later reading is taken on."""
    rows = m.rows_all if rows is None else rows
    cam = m.camera if cam is None else cam
    pp = cam.principal_point
    uv = np.ascontiguousarray(m.obs_uv[rows], dtype=np.float64)
    r = np.hypot(uv[:, 0] - float(pp[0]), uv[:, 1] - float(pp[1]))
    r = r[np.isfinite(r)]
    if not len(r):
        return {"n_obs": 0, "family": family_of(cam.model), "d_max": 0.0, "f0": 0.0}
    # The incidence angle of each observation, read through the camera's own
    # unprojection about its own principal ray, so a model with distortion
    # answers with the angle it actually images that pixel at.
    rays = np.asarray(cam.pixel_to_ray_batch(uv), float)
    nrm = np.linalg.norm(rays, axis=1)
    good = np.isfinite(nrm) & (nrm > 0)
    axis = np.asarray(cam.pixel_to_ray(float(pp[0]), float(pp[1])), float)
    axis = axis / max(1e-12, float(np.linalg.norm(axis)))
    ang = np.arccos(np.clip((rays[good] / nrm[good][:, None]) @ axis, -1.0, 1.0))
    ang = ang[np.isfinite(ang)]
    f0 = base_focal(cam)
    return {
        "n_obs": int(len(r)),
        "r_max_px": float(r.max()),
        "r_p99_px": float(np.percentile(r, 99)),
        "theta_p50_deg": float(np.degrees(np.percentile(ang, 50))) if len(ang) else 0.0,
        "theta_p99_deg": float(np.degrees(np.percentile(ang, 99))) if len(ang) else 0.0,
        "theta_max_deg": float(np.degrees(ang.max())) if len(ang) else 0.0,
        "theta_p99_rad": float(np.percentile(ang, 99)) if len(ang) else 0.0,
        "f0": f0,
        "d_max": float(DOMAIN_HEADROOM * (r.max() / f0)),
        "family": family_of(cam.model),
    }


def sample_thetas(theta_max, n=N_SAMPLES):
    """Incidence angles from the axis out to ``theta_max``, uniformly."""
    return np.linspace(0.0, float(theta_max), int(n))


def radial_map(cam, thetas):
    """``r(theta)`` in pixels: the image radius a ray at incidence ``theta``
    lands at, through the camera's whole composite map.

    Rays are built in the camera's own frame, so this reads the model exactly
    as the projection does, distortion included, and needs no knowledge of
    which chart the model is parameterized on."""
    th = np.asarray(thetas, float)
    pp = cam.principal_point
    acc = np.zeros((len(MAP_AZIMUTHS), len(th)))
    for k, phi in enumerate(MAP_AZIMUTHS):
        rays = np.stack(
            [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), -np.cos(th)], axis=1
        )
        px = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
        acc[k] = np.hypot(px[:, 0] - float(pp[0]), px[:, 1] - float(pp[1]))
    with np.errstate(invalid="ignore"):
        return np.nanmean(acc, axis=0)


def equivalent_focal(cam, thetas, family=None):
    """The single ``f_eq`` with ``r = f_eq * d`` best fitting the camera's
    composite map over ``thetas``, least squares.

    ``d`` is the BASE chart's radial coordinate, so this is commensurable with
    the capture vote, which measures the best base-model focal."""
    fam = family_of(cam.model) if family is None else family
    r = radial_map(cam, thetas)
    d = d_of_theta(np.asarray(thetas, float), fam)
    ok = np.isfinite(r) & np.isfinite(d) & (d > 0)
    if not ok.any():
        return float("nan")
    return float((r[ok] * d[ok]).sum() / (d[ok] ** 2).sum())


def swap_camera(m, cam, f_eq=None, thetas=None):
    """Put ``cam`` on the member and drop every cached ray.

    The ray helpers memoize on the member, and the relaxation's angular
    consensus bound is the pixel bar over the member's equivalent focal, so a
    swap that left either behind would have the relaxation read the OLD lens's
    rays at the OLD lens's tolerance."""
    m.camera = cam
    for attr in ("_frame_rays_memo", "_frame_rays_all_memo", "_covis_memo"):
        if hasattr(m, attr):
            delattr(m, attr)
    if f_eq is None and thetas is not None:
        f_eq = equivalent_focal(cam, thetas)
    if f_eq is not None and np.isfinite(f_eq) and f_eq > 0:
        m.f_eq = float(f_eq)
    return m


def apply_rotations(m, frames, quats):
    """Write refined rotations back onto the member, in its own frame."""
    rot = quat.rots_from_wxyz(np.asarray(quats, float))
    for k, f in enumerate(frames):
        m.rot[int(f)] = rot[k]
    m.rvec = quat.rotvecs_from_rots(m.rot)
    for attr in ("_frame_rays_memo", "_frame_rays_all_memo", "_covis_memo"):
        if hasattr(m, attr):
            delattr(m, attr)
    return m


def rot_lens_inputs(m, rows):
    """The kernel's arrays for a rotation-only state: every point a bearing,
    every centre at the origin.

    A point at infinity's observations depend on the rotation and the camera
    model and on nothing else, so with every point marked the kernel freezes
    every translation by its own rule, and what is left in the solve is exactly
    the rotations, the focal and the spline."""
    frames = sorted(int(f) for f in m.frames)
    rows = rows[np.isin(m.obs_i[rows], frames)]
    fslot = {f: k for k, f in enumerate(frames)}
    cl = np.unique(m.obs_c[rows])
    cslot = {int(c): k for k, c in enumerate(cl)}
    pts = np.zeros((len(cl), 3))
    for c, k in cslot.items():
        d = np.asarray(m.pts[c], float)
        n = float(np.linalg.norm(d))
        if not np.isfinite(d).all() or not n > 0:
            d, n = np.array([0.0, 0.0, -1.0]), 1.0
        pts[k] = d / n
    return {
        "frames": frames,
        "clusters": cl,
        "quats": np.stack([quat.quat_from_rot(m.rot[f]) for f in frames]),
        "trans": np.zeros((len(frames), 3)),
        "points": pts,
        "at_inf": np.ones(len(cl), bool),
        "uv": np.ascontiguousarray(m.obs_uv[rows]),
        "obs_image": np.array([fslot[int(i)] for i in m.obs_i[rows]], np.uint32),
        "obs_point": np.array([cslot[int(c)] for c in m.obs_c[rows]], np.uint32),
        "rows": rows,
    }


def rot_lens_ba(
    m,
    knots=SEED_KNOTS,
    rowset="admission",
    opt_bspline=True,
    schedule=None,
    max_iters=60,
):
    """The lens read off the bearings alone.

    Promote the member's base camera to its spline model with a zero spline,
    mark every point at infinity so every translation freezes, and release the
    focal, with the spline beside it where a radial profile is wanted: under
    the centre-anchored gauge the spline cannot express a central-scale
    correction, so the two have to move together.

    Returns ``(record, camera, (frames, quaternions))``.  A solve that raises,
    or that comes back with no finite residual, is a REFUSAL: the previous
    camera stands and the record says so."""
    from sfmtool._sfmtool.geometry import bundle_adjust

    rows = m.rows_all if rowset == "admission" else m.rows
    inp = rot_lens_inputs(m, rows)
    fld = observed_field(m, inp["rows"])
    f0, d_max = fld["f0"], fld["d_max"]
    rec = {
        "knots": int(knots),
        "rowset": rowset,
        "opt_bspline": bool(opt_bspline),
        "n_obs": int(len(inp["rows"])),
        "n_frames": len(inp["frames"]),
        "n_points": int(len(inp["clusters"])),
        "d_max": d_max,
        "f_in": f0,
    }
    if not (np.isfinite(d_max) and d_max > 0):
        rec["refused"] = "no spline domain"
        return rec, None, None
    cam0 = promote(m.camera, f0, np.zeros(int(knots)), d_max)
    try:
        out = bundle_adjust(
            cam0,
            np.ascontiguousarray(inp["quats"]),
            np.ascontiguousarray(inp["trans"]),
            np.ascontiguousarray(inp["points"]),
            inp["uv"],
            inp["obs_image"],
            inp["obs_point"],
            point_at_infinity=inp["at_inf"],
            opt_f=True,
            opt_bspline=bool(opt_bspline),
            schedule=list(full_schedule() if schedule is None else schedule),
            max_iters=int(max_iters),
        )
    except Exception as exc:  # noqa: BLE001 -- a lens rung never kills the run
        rec["refused"] = f"{type(exc).__name__}: {exc}"
        return rec, None, None
    resid = np.asarray(out["residual_norms"])
    fin = np.isfinite(resid)
    rec["resid_finite_frac"] = float(fin.mean()) if len(resid) else 0.0
    rec["reproj_med_px"] = float(np.median(resid[fin])) if fin.any() else None
    rec["reproj_p90_px"] = float(np.percentile(resid[fin], 90)) if fin.any() else None
    if not fin.any():
        rec["refused"] = "no finite residual"
        return rec, None, None
    coeffs = np.asarray(out["bspline_coefficients"], float)
    f_out = float(out["focal"])
    cam = promote(m.camera, f_out, coeffs, d_max)
    rec["f_out"] = f_out
    rec["coeffs"] = [float(c) for c in coeffs]
    rec["coeff_absmax"] = float(np.abs(coeffs).max()) if len(coeffs) else 0.0
    rot_out = quat.rots_from_wxyz(np.asarray(out["quaternions_wxyz"]))
    d = np.array(
        [
            quat.rot_angle_deg(rot_out[k] @ m.rot[f].T)
            for k, f in enumerate(inp["frames"])
        ]
    )
    rec["rot_delta_med_deg"] = float(np.median(d))
    rec["rot_delta_p90_deg"] = float(np.percentile(d, 90))
    rec["rot_delta_max_deg"] = float(d.max())
    return rec, cam, (inp["frames"], np.asarray(out["quaternions_wxyz"]))


def final_release(m, state, knots=SEED_KNOTS, schedule=None, max_iters=30):
    """One adjustment on a RELAXED state with the focal and the spline open.

    The lens read off finite structure and bearings together, after the centres
    exist.  The state's own camera is promoted where it is not already a spline
    model; where it is, its coefficients are the start."""
    from sfmtool._sfmtool.geometry import bundle_adjust

    frames = [int(f) for f in state["frames"]]
    clusters = [int(c) for c in state["clusters"]]
    cslot = {c: k for k, c in enumerate(clusters)}
    fslot = {f: k for k, f in enumerate(frames)}
    rows = m.rows_all[np.isin(m.obs_i[m.rows_all], frames)]
    rows = rows[np.array([int(c) in cslot for c in m.obs_c[rows]], bool)]
    cam_in = m.camera
    d_max = spline_domain(cam_in)
    if d_max is not None:
        cam0 = cam_in
    else:
        d_max = observed_field(m, rows)["d_max"]
        if not (np.isfinite(d_max) and d_max > 0):
            return {"refused": "no spline domain"}, None, None
        cam0 = promote(cam_in, base_focal(cam_in), np.zeros(int(knots)), d_max)
    rec = {
        "n_obs": int(len(rows)),
        "n_frames": len(frames),
        "d_max": float(d_max),
        "f_in": base_focal(cam0),
    }
    try:
        out = bundle_adjust(
            cam0,
            np.ascontiguousarray(np.asarray(state["quats"], float)),
            np.ascontiguousarray(np.asarray(state["trans"], float)),
            np.ascontiguousarray(np.asarray(state["points"], float)),
            np.ascontiguousarray(m.obs_uv[rows]),
            np.array([fslot[int(i)] for i in m.obs_i[rows]], np.uint32),
            np.array([cslot[int(c)] for c in m.obs_c[rows]], np.uint32),
            point_at_infinity=np.ascontiguousarray(np.asarray(state["at_inf"], bool)),
            opt_f=True,
            opt_bspline=True,
            schedule=list(SETTLED_SCHEDULE if schedule is None else schedule),
            max_iters=int(max_iters),
        )
    except Exception as exc:  # noqa: BLE001
        rec["refused"] = f"{type(exc).__name__}: {exc}"
        return rec, None, None
    resid = np.asarray(out["residual_norms"])
    fin = np.isfinite(resid)
    rec["resid_finite_frac"] = float(fin.mean()) if len(resid) else 0.0
    rec["reproj_med_px"] = float(np.median(resid[fin])) if fin.any() else None
    rec["reproj_p90_px"] = float(np.percentile(resid[fin], 90)) if fin.any() else None
    if not fin.any():
        rec["refused"] = "no finite residual"
        return rec, None, None
    coeffs = np.asarray(out["bspline_coefficients"], float)
    cam = promote(cam0, float(out["focal"]), coeffs, float(d_max))
    rec["f_out"] = float(out["focal"])
    rec["coeffs"] = [float(c) for c in coeffs]
    new_state = {
        "frames": np.asarray(frames, np.int64),
        "clusters": np.asarray(clusters, np.int64),
        "quats": np.asarray(out["quaternions_wxyz"], float),
        "trans": np.asarray(out["translations"], float),
        "points": np.asarray(out["points"], float),
        "at_inf": np.asarray(state["at_inf"], bool),
    }
    return rec, cam, new_state


def refit_knots(cam, knots, thetas, d_max):
    """``cam``'s own composite radial map, re-expressed on a ``knots``-knot
    spline of the same base model.

    The spline correction enters the map linearly, so the coefficients that
    make a k-knot camera reproduce a given map are one linear least squares
    over the basis, and the basis columns are read from the model itself: the
    map of a camera whose coefficients are the i-th unit vector, minus the map
    of the zero-coefficient camera.  A camera re-expressed this way therefore
    starts where it ended rather than at the base model, and a base camera
    re-expressed at any knot count gives exactly zeros.

    Returns ``(camera, fit residual in pixels)``."""
    f = base_focal(cam)
    base = promote(cam, f, np.zeros(int(knots)), d_max)
    r_base = radial_map(base, thetas)
    r_want = radial_map(cam, thetas)
    cols = []
    for i in range(int(knots)):
        e = np.zeros(int(knots))
        e[i] = 1.0
        cols.append(radial_map(promote(cam, f, e, d_max), thetas) - r_base)
    a = np.stack(cols, axis=1)
    y = r_want - r_base
    ok = np.isfinite(y) & np.isfinite(a).all(axis=1)
    if int(ok.sum()) < int(knots):
        return promote(cam, f, np.zeros(int(knots)), d_max), float("inf")
    c, *_ = np.linalg.lstsq(a[ok], y[ok], rcond=None)
    fit = promote(cam, f, c, d_max)
    resid = float(np.max(np.abs(radial_map(fit, thetas) - r_want)))
    return fit, resid


def release_at_knots(m, state, cam, knots, thetas, max_iters=30):
    """The late release at ``knots`` knots.

    The starting camera is ``cam`` re-expressed on the ``knots``-knot spline,
    so the release grows the same lens rather than restarting it.  Returns
    ``(record, camera, state)``; the record carries the refit residual."""
    frames = [int(f) for f in state["frames"]]
    d_max = spline_domain(cam)
    if d_max is None:
        rows = m.rows_all[np.isin(m.obs_i[m.rows_all], frames)]
        d_max = observed_field(m, rows, cam=cam)["d_max"]
    if not (np.isfinite(d_max) and d_max > 0):
        return {"refused": "no spline domain", "knots": int(knots)}, None, None
    cam0, refit = refit_knots(cam, knots, thetas, float(d_max))
    hold = m.camera
    m.camera = cam0
    try:
        rec, cam_out, st_out = final_release(
            m, state, knots=int(knots), max_iters=int(max_iters)
        )
    finally:
        m.camera = hold
    rec["knots"] = int(knots)
    rec["refit_resid_px"] = refit
    return rec, cam_out, st_out
