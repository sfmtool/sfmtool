# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-1/2/3a gate for the fisheye seed (``scripts/notes-fisheye-seed.md``).

Drives the seed scripts' four geometric primitives — ``triangulate``,
``p3p_resect``, ``pose_refine`` and ``reproj_res_one``, in BOTH
``exp_fast_seed`` and ``exp_pinhole_bootstrap`` — over a synthetic
equidistant scene with planted poses and points, under the fisheye camera
context (``EQUIDISTANT_FISHEYE``).  The scene is deliberately built
so a large share of its observations sit at theta >= 90 degrees, i.e. behind
the image plane: the failure class this gate exists for is a hidden z = 1
normalization or a z > 0 cheirality test that silently discards the periphery
of a >180-degree capture.

Phase 3a replaced the hand-carried ``SIMPLE_RADIAL_FISHEYE`` with ``k1 = 0``
convention with the native ``EQUIDISTANT_FISHEYE`` model.  One cross-check
below asserts the two representations still project identically in both
directions, so the switch is a change of representation and not of geometry.

The same scene generator is re-run under the DEFAULT pinhole context on a
narrow-FOV scene, so a regression that breaks the pinhole path cannot hide
behind the fisheye one.

Phase 2 adds a second suite over the same planted scene: the ray-space seed
path end to end — epipolar estimation on unit rays, four-way decomposition
with RAY-NATIVE chirality, ray-midpoint triangulation, growth by ray-P3P
(``fisheye_window_seed``) — plus the ray parallax measure and the ray-rotation
far-field core.  Those checks run under the fisheye context only; the pinhole
path never reaches the code they cover.

Run:  pixi run -e dev python scripts/check_fisheye_seed_primitives.py

Exits 0 when every check passes; prints the first failure and exits 1
otherwise.  Kernel-level facts (the ray maps, the native pose stack, the
native BA under fixed fisheye intrinsics) are pinned by Rust tests in
``crates/sfmtool-core`` — this script covers only the script-level layer those
kernels sit under.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_fast_seed as F  # noqa: E402
import exp_pinhole_bootstrap as B  # noqa: E402

WH = (480, 480)
F_EQUI = 130.0  # equidistant focal: theta = r / f, image circle at ~105 deg
F_PIN = 500.0

_FAILURES = []


def check(ok, label, detail=""):
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}{'' if not detail else '  — ' + detail}")
    if not ok:
        _FAILURES.append(label)


def check_model_equivalence():
    """The Phase-3a cross-check: EQUIDISTANT_FISHEYE and the pre-3a convention
    (SIMPLE_RADIAL_FISHEYE with k1 = 0) parameterize the SAME map.

    Both directions, over a theta sweep that straddles 90 degrees.  They agree
    bit-for-bit outside the polynomial family's 90-100 degree wide-angle blend
    band, where the k1 = 0 arm lerps two identical rays and renormalizes;
    1e-12 covers that round-off.  What the native model buys — an analytic
    pixel Jacobian where the polynomial carrier has none — is a Rust-side
    fact, pinned in ``camera/distortion/tests.rs``; there is no binding for
    ``ray_to_pixel_with_jacobian`` to assert it from here."""
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    print("\nPhase 3a — representation cross-check")
    w, h = WH
    base = {
        "focal_length": F_EQUI,
        "principal_point_x": w / 2.0,
        "principal_point_y": h / 2.0,
    }
    native = CameraIntrinsics.from_dict(
        {"model": "EQUIDISTANT_FISHEYE", "width": w, "height": h, "parameters": base}
    )
    legacy = CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_RADIAL_FISHEYE",
            "width": w,
            "height": h,
            "parameters": dict(base, radial_distortion_k1=0.0),
        }
    )
    rays = []
    for deg in (0.0, 30.0, 60.0, 89.0, 90.0, 91.0, 95.0, 105.0, 130.0):
        for phi_deg in (0.0, 71.0, 200.0):
            t, p = np.radians(deg), np.radians(phi_deg)
            rays.append([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), -np.cos(t)])
    rays = np.ascontiguousarray(np.asarray(rays, float))

    px_n = np.asarray(native.ray_to_pixel_batch(rays))
    px_l = np.asarray(legacy.ray_to_pixel_batch(rays))
    d_fwd = np.abs(px_n - px_l).max()
    check(
        d_fwd < 1e-12,
        "ray_to_pixel agrees with the k1 = 0 convention",
        f"max {d_fwd:.2e} px",
    )

    back_n = np.asarray(native.pixel_to_ray_batch(np.ascontiguousarray(px_n)))
    back_l = np.asarray(legacy.pixel_to_ray_batch(np.ascontiguousarray(px_l)))
    d_inv = np.abs(back_n - back_l).max()
    check(
        d_inv < 1e-12,
        "pixel_to_ray agrees with the k1 = 0 convention",
        f"max {d_inv:.2e}",
    )

    cx, cy = native.principal_point
    r = np.hypot(px_n[:, 0] - cx, px_n[:, 1] - cy)
    theta = np.arccos(np.clip(-rays[:, 2], -1.0, 1.0))
    d_exact = np.abs(r - F_EQUI * theta).max()
    check(
        d_exact < 1e-9,
        "the native map is exactly r = f * theta",
        f"max {d_exact:.2e} px",
    )

    d_rt = np.abs(back_n - rays).max()
    check(d_rt < 1e-12, "pixel -> ray inverts ray -> pixel exactly", f"max {d_rt:.2e}")


def look_at(center):
    """Canonical world->camera pose of a camera at ``center`` looking at the
    origin (the camera looks along -Z, so its local +Z points along
    ``center``)."""
    z = np.asarray(center, float)
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z])  # world -> camera
    return rot, -rot @ np.asarray(center, float)


def rotation_rig(n_img):
    """Pure-rotation poses: every camera centre at the world origin, only the
    orientation changing — the far-field regime the rotation core models."""
    rot = np.zeros((n_img, 3, 3))
    for i in range(n_img):
        rot[i] = Rotation.from_rotvec(
            np.array([0.04, 0.21, -0.03]) * (i - (n_img - 1) / 2.0)
        ).as_matrix()
    return rot, np.zeros((n_img, 3))


def make_scene(cam, n_img, n_pt, shell_radius, rig_radius, theta_max_deg, poses=None):
    """Planted poses/points plus their exact observations under ``cam``.

    Cameras sit on a small arc of radius ``rig_radius`` looking at the origin
    (or take the supplied ``poses``); points sit on a shell of radius
    ``shell_radius`` AROUND the rig, so each camera images far more than a
    hemisphere when ``cam`` is equidistant.  Observations are kept out to
    ``theta_max_deg`` off-axis.  Returns the flat-observation arrays the
    scripts consume."""
    rot = np.zeros((n_img, 3, 3))
    trans = np.zeros((n_img, 3))
    if poses is not None:
        rot, trans = poses
    else:
        for i in range(n_img):
            ang = 0.25 * (i - (n_img - 1) / 2.0)
            center = np.array(
                [
                    rig_radius * np.sin(ang),
                    0.12 * ((i % 3) - 1),
                    rig_radius * np.cos(ang),
                ]
            )
            rot[i], trans[i] = look_at(center)

    pts = np.zeros((n_pt, 3))
    for p in range(n_pt):
        theta = np.pi * (0.12 + 0.76 * p / (n_pt - 1.0))
        phi = 2.399963 * p
        rad = shell_radius * (1.0 + 0.15 * np.sin(3.1 * p))
        pts[p] = [
            rad * np.sin(theta) * np.cos(phi),
            rad * np.cos(theta),
            rad * np.sin(theta) * np.sin(phi),
        ]

    r_max = float(cam.focal_lengths[0]) * np.radians(theta_max_deg)
    cx, cy = WH[0] / 2.0, WH[1] / 2.0
    obs_c, obs_i, uv, behind = [], [], [], 0
    for i in range(n_img):
        x_cam = pts @ rot[i].T + trans[i]
        proj = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam)))
        finite = np.isfinite(proj).all(axis=1)
        rad_px = np.hypot(proj[:, 0] - cx, proj[:, 1] - cy)
        keep = finite & (rad_px <= r_max)
        for p in np.nonzero(keep)[0]:
            obs_c.append(p)
            obs_i.append(i)
            uv.append(proj[p])
            behind += int(x_cam[p, 2] > 0.0)
    order = np.lexsort((np.asarray(obs_i), np.asarray(obs_c)))
    return {
        "rot": rot,
        "trans": trans,
        "pts": pts,
        "obs_c": np.asarray(obs_c, np.uint32)[order],
        "obs_i": np.asarray(obs_i, np.uint32)[order],
        "u": np.ascontiguousarray(np.asarray(uv, float)[order]),
        "n_img": n_img,
        "n_cl": n_pt,
        "behind": behind,
    }


def run_suite(module, label, model, focal, shell, rig, theta_max, min_behind):
    print(f"\n{label}  [{model} @ f = {focal:g} px]")
    module._CAM_WH = WH
    module.set_camera_context(model, focal)
    cam = module.make_cam(focal)
    check(
        cam.model == model,
        "make_cam honours the camera context",
        f"model = {cam.model}",
    )
    check(
        np.allclose(module.make_cam().focal_lengths, (focal, focal)),
        "make_cam() with no focal falls back to the context focal",
    )

    s = make_scene(cam, 6, 160, shell, rig, theta_max)
    n_obs = len(s["obs_c"])
    check(
        s["behind"] >= min_behind,
        f"scene reaches past 90 deg ({s['behind']}/{n_obs} observations)",
    )

    # ── triangulate ────────────────────────────────────────────────────────
    used = np.ones(s["n_img"], bool)
    tri = module.triangulate(
        s["obs_c"], s["obs_i"], s["u"], s["rot"], s["trans"], used, s["n_cl"], focal
    )
    seen = np.bincount(s["obs_c"], minlength=s["n_cl"]) >= 2
    err = np.linalg.norm(tri[seen] - s["pts"][seen], axis=1)
    check(
        np.isfinite(err).all() and err.max() < 1e-8,
        "triangulate recovers the planted points",
        f"worst {np.nanmax(err):.3e} (n = {int(seen.sum())})",
    )

    # ── p3p_resect / pose_refine / reproj_res_one on one image ─────────────
    img = s["n_img"] // 2
    sel = s["obs_i"] == img
    uv_i = np.ascontiguousarray(s["u"][sel])
    x_i = np.ascontiguousarray(s["pts"][s["obs_c"][sel]])
    rv_true = Rotation.from_matrix(s["rot"][img]).as_rotvec()
    tv_true = s["trans"][img]

    args = (uv_i, x_i, focal) + ((WH,) if module is B else ())
    ans = module.p3p_resect(*args)
    if ans is None:
        check(False, "p3p_resect finds a consensus")
    else:
        rv, tv, inl = ans
        d_rot = np.degrees(
            (Rotation.from_rotvec(rv).inv() * Rotation.from_rotvec(rv_true)).magnitude()
        )
        check(
            d_rot < 1e-4 and np.linalg.norm(tv - tv_true) < 1e-4,
            "p3p_resect recovers the planted pose",
            f"{d_rot:.2e} deg, |dt| {np.linalg.norm(tv - tv_true):.2e}",
        )
        check(
            inl.all(),
            "p3p_resect keeps every exact observation as an inlier",
            f"{int(inl.sum())}/{len(inl)}",
        )

    rv0 = rv_true + np.array([0.01, -0.008, 0.006])
    tv0 = tv_true + np.array([0.02, -0.015, 0.018])
    out = module.pose_refine(uv_i, x_i, rv0, tv0, focal)
    rv_r, tv_r, frac = out[0], out[1], out[2]
    d_rot = np.degrees(
        (Rotation.from_rotvec(rv_r).inv() * Rotation.from_rotvec(rv_true)).magnitude()
    )
    check(
        d_rot < 1e-3 and np.linalg.norm(tv_r - tv_true) < 1e-4,
        "pose_refine converges to the planted pose from a perturbed init",
        f"{d_rot:.2e} deg, |dt| {np.linalg.norm(tv_r - tv_true):.2e}",
    )
    check(
        frac > 0.999,
        "pose_refine keeps every observation inside 3 px",
        f"inlier fraction {frac:.4f}",
    )

    res = np.asarray(module.reproj_res_one(cam, rv_true, tv_true, x_i, uv_i))
    norms = np.linalg.norm(res, axis=1)
    check(
        norms.max() < 1e-8,
        "reproj_res_one is zero at the planted pose",
        f"worst {norms.max():.3e} px over {len(norms)} obs",
    )


def run_ray_suite():
    """Phase-2 suite: the ray-space seed path on a planted equidistant scene.

    Uses a wider rig than the primitive suite so the seed pair carries real
    translation parallax — a ray-space pair init is exactly what the affine
    factorization could not do here."""
    print(f"\nexp_fast_seed — ray-space seed (Phase 2)  [equidistant @ {F_EQUI:g} px]")
    F._CAM_WH = WH
    F.set_camera_context("EQUIDISTANT_FISHEYE", F_EQUI)
    cam = F.make_cam(F_EQUI)
    check(F.fisheye_stage1(), "fisheye_stage1() is armed by the camera context")
    check(
        abs(F.fisheye_ray_tol(F_EQUI) - F.FISHEYE_TOL_PX / F_EQUI) < 1e-15,
        "the angular bound is the pixel tolerance through dr/dtheta",
        f"{np.degrees(F.fisheye_ray_tol(F_EQUI)):.3f} deg",
    )

    s = make_scene(cam, 6, 200, shell_radius=6.0, rig_radius=2.4, theta_max_deg=105.0)
    n_obs = len(s["obs_c"])
    check(
        s["behind"] >= 60,
        f"ray-suite scene reaches past 90 deg ({s['behind']}/{n_obs} observations)",
    )

    # ── ray_pair_pose: E on rays -> pose -> ray-native chirality ────────────
    a, b = 0, 5
    ma, mb = s["obs_i"] == a, s["obs_i"] == b
    common, ia, ib = np.intersect1d(s["obs_c"][ma], s["obs_c"][mb], return_indices=True)
    x1 = np.ascontiguousarray(s["u"][ma][ia])
    x2 = np.ascontiguousarray(s["u"][mb][ib])
    pose = F.ray_pair_pose(x1, x2, F_EQUI)
    if pose is None:
        check(False, "ray_pair_pose finds a two-view consensus")
        return
    check(
        pose["essentialness"] < 1e-6,
        "the ray-space epipolar matrix is essential at the true camera",
        f"{pose['essentialness']:.2e}",
    )
    r_true = s["rot"][b] @ s["rot"][a].T
    t_true = s["trans"][b] - r_true @ s["trans"][a]
    d_rot = np.degrees(
        (
            Rotation.from_matrix(pose["rot"]).inv() * Rotation.from_matrix(r_true)
        ).magnitude()
    )
    cos_t = float(
        np.dot(pose["tvec"], t_true)
        / (np.linalg.norm(t_true) * np.linalg.norm(pose["tvec"]))
    )
    check(
        d_rot < 1e-6 and cos_t > 1 - 1e-9,
        "the decomposition recovers the planted relative pose",
        f"{d_rot:.2e} deg, baseline cos {cos_t:.12f}",
    )
    # The chirality-winning hypothesis must keep the WHOLE consensus, periphery
    # included — a z > 0 test would drop every beyond-90-degree observation.
    n_inl = int(np.asarray(pose["inliers"]).sum())
    check(
        pose["n_cheiral"] == n_inl,
        "ray-native chirality keeps every epipolar inlier",
        f"{pose['n_cheiral']}/{n_inl}",
    )
    d_a = np.asarray(cam.pixel_to_ray_batch(x1))
    check(
        int((d_a[np.asarray(pose["inliers"])][:, 2] <= 0).sum()) > 10,
        "the kept consensus includes beyond-hemisphere rays",
        f"{int((d_a[np.asarray(pose['inliers'])][:, 2] <= 0).sum())} rays with z <= 0",
    )
    par, _n = F.ray_pair_parallax(x1, x2, F_EQUI)
    check(
        np.isfinite(par) and par > 1.0,
        "ray_pair_parallax measures the pair's translation parallax",
        f"{par:.2f} deg",
    )

    # ── fisheye_window_seed: the whole window init, end to end ─────────────
    out = F.fisheye_window_seed(
        s["obs_c"].astype(np.int64),
        s["obs_i"].astype(np.int64),
        s["u"],
        np.arange(s["n_img"]),
        F_EQUI,
    )
    if out is None:
        check(False, "fisheye_window_seed produces a window solve")
        return
    inl, imgs, used, cl_ids, rvw, tvw, p_w = out
    check(
        int(used.sum()) == s["n_img"],
        "every window image joins by ray-P3P",
        f"{int(used.sum())}/{s['n_img']}",
    )
    check(inl > 0.99, "the window mini-BA fits the planted scene", f"inlier {inl:.4f}")
    # The solve is gauge-free (similarity), so compare after aligning the
    # planted cameras onto it: centres to a similarity, rotations to their own
    # best gauge rotation.
    r_est = Rotation.from_rotvec(rvw).as_matrix()
    r_ref = s["rot"]
    u_svd, _s, vt = np.linalg.svd(np.einsum("nji,njk->ik", r_est, r_ref))
    if np.linalg.det(u_svd @ vt) < 0:
        u_svd[:, 2] *= -1.0
    g = u_svd @ vt
    rot_err = Rotation.from_matrix(
        np.einsum("nij,nkj->nik", r_ref, np.einsum("nij,jk->nik", r_est, g))
    ).magnitude() * (180 / np.pi)
    check(
        rot_err.max() < 1e-3,
        "window camera rotations match the planted rig",
        f"worst {rot_err.max():.2e} deg",
    )

    # ── rotation_core_rays: the far-field ray-rotation core ────────────────
    # A pure-rotation rig (all centres at the origin) is the far-field regime
    # the core models; every pair is then explained by a rotation of rays.
    rig = make_scene(
        cam,
        6,
        200,
        shell_radius=6.0,
        rig_radius=0.0,
        theta_max_deg=105.0,
        poses=rotation_rig(6),
    )
    from sfmtool._sfmtool.geometry import fit_ray_rotation

    ra = np.ascontiguousarray(
        cam.pixel_to_ray_batch(np.ascontiguousarray(rig["u"][rig["obs_i"] == 0]))
    )
    ids_a = rig["obs_c"][rig["obs_i"] == 0]
    ids_b = rig["obs_c"][rig["obs_i"] == 3]
    common, ja, jb = np.intersect1d(ids_a, ids_b, return_indices=True)
    rb = np.ascontiguousarray(
        cam.pixel_to_ray_batch(np.ascontiguousarray(rig["u"][rig["obs_i"] == 3][jb]))
    )
    fit = fit_ray_rotation(
        np.ascontiguousarray(ra[ja]), rb, max_angle_rad=F.fisheye_ray_tol(F_EQUI)
    )
    if fit is None:
        check(False, "fit_ray_rotation explains a parallax-free fisheye pair")
    else:
        r_true = rig["rot"][3] @ rig["rot"][0].T
        d = np.degrees(
            (
                Rotation.from_matrix(np.asarray(fit["rotation"])).inv()
                * Rotation.from_matrix(r_true)
            ).magnitude()
        )
        check(
            d < 1e-6 and np.asarray(fit["inliers"]).all(),
            "fit_ray_rotation recovers a parallax-free pair's rotation",
            f"{d:.2e} deg, {int(np.asarray(fit['inliers']).sum())}/{len(ra[ja])} inliers",
        )
    core = F.rotation_core_rays(
        rig["obs_c"].astype(np.int64),
        rig["obs_i"].astype(np.int64),
        rig["u"],
        rig["n_img"],
        rig["n_cl"],
        F_EQUI,
    )
    check(
        core is not None,
        "rotation_core_rays builds a far-field skeleton on a rotating rig",
    )
    if core is not None:
        _inl, _par, _rv, _tv, _pts, pm, _med = core
        check(
            int(pm.sum()) >= 3,
            "the core poses a skeleton from the ray-rotation edges",
            f"{int(pm.sum())}/{rig['n_img']} posed",
        )


def run_release_suite():
    """Phase 3b — the equidistant focal scan and the free-focal release.

    Three things, on the same planted equidistant scene: the FOV-derived scan
    grid is the band the focal vote itself scans; the scripts' free-focal BA
    recovers a PERTURBED focal (the kernel's ``opt_f`` now admits
    EQUIDISTANT_FISHEYE, and its analytic focal column is exact for the map);
    and the same solve with ``opt_f = False`` returns the input focal
    untouched, so the release is what moves it and not the trim schedule."""
    print("\nPhase 3b — equidistant focal scan and release")
    F._CAM_WH = WH
    F.set_camera_context("EQUIDISTANT_FISHEYE", F_EQUI)
    check(F.fisheye_stage1(), "the fisheye stage-1 gate is armed")

    lo, hi = F.fisheye_focal_band()
    grid = F.fisheye_focal_grid(F_EQUI)
    check(
        len(grid) == 5
        and grid.min() >= lo
        and grid.max() <= hi
        and abs(grid[2] - F_EQUI) < 1e-9,
        "the scan grid is five candidates centred on the verdict, in band",
        f"[{', '.join(f'{v:.1f}' for v in grid)}] in [{lo:.1f}, {hi:.1f}]",
    )
    # Log-symmetric: no upward skew (the equidistant column has no measured
    # directional bias, unlike the pinhole vote the pinhole grid corrects for).
    lr = np.log(grid / F_EQUI)
    check(
        abs(lr[0] + lr[4]) < 1e-12 and abs(lr[1] + lr[3]) < 1e-12,
        "the scan grid is log-symmetric about the verdict focal",
    )
    # The floor a release must clear sits BELOW the capture's own focal — the
    # pinhole 0.3 x max(w, h) would reject it (kerry: 138 px against 144).
    check(
        F.focal_floor() < F_EQUI < 0.3 * max(WH),
        "the release floor is FOV-derived, not the pinhole plausibility bound",
        f"floor {F.focal_floor():.1f} px, pinhole floor {0.3 * max(WH):.1f} px",
    )

    cam = F.make_cam(F_EQUI)
    s = make_scene(cam, 8, 220, shell_radius=6.0, rig_radius=1.2, theta_max_deg=105.0)
    n_obs = len(s["obs_c"])
    check(
        s["behind"] >= 60,
        f"release scene reaches past 90 deg ({s['behind']}/{n_obs} observations)",
    )

    # Focal and depth trade, so a perturbed start scales the whole scene with
    # the focal — exactly what the scan's per-candidate rescale does.
    f_start = 0.91 * F_EQUI
    scale = f_start / F_EQUI
    rvec = Rotation.from_matrix(s["rot"]).as_rotvec()
    tvec = s["trans"] * scale
    pts0 = s["pts"] * scale
    args = (
        s["obs_c"],
        s["obs_i"],
        s["u"],
        rvec,
        tvec,
        pts0,
        f_start,
        s["n_img"],
        s["n_cl"],
    )
    f_fixed, _, _, _, _, _ = F.bundle_adjust(*args, opt_f=False)
    check(
        f_fixed == f_start,
        "opt_f = False leaves the equidistant focal bit-identical",
        f"{f_fixed!r}",
    )
    f_free, rv_r, _tv_r, _pts_r, res, inl = F.bundle_adjust(*args, opt_f=True)
    err = abs(f_free - F_EQUI) / F_EQUI
    check(
        err < 0.01,
        "free-focal BA recovers the planted equidistant focal",
        f"{f_start:.1f} -> {f_free:.2f} px (planted {F_EQUI:g}), err {100 * err:.2f}%",
    )
    check(
        float(np.median(res[np.isfinite(res)])) < 0.05 and inl > 0.99,
        "the released solve fits the observations",
        f"median {np.median(res[np.isfinite(res)]):.3e} px, inlier<2px "
        f"{100 * inl:.1f}%",
    )
    d_rot = np.degrees(
        (Rotation.from_rotvec(rv_r).inv() * Rotation.from_matrix(s["rot"])).magnitude()
    ).max()
    check(d_rot < 0.05, "the release did not bend the rotations", f"{d_rot:.2e} deg")


def run_embed_suite():
    """Phase 4 — the photometric layer's model-generic pieces.

    Three things the finalization's embed and its culls rest on, checked on
    planted equidistant geometry rather than inferred:

    1. ``_colmap_proj_jacobian`` — the camera's own 2x3 pixel Jacobian, which
       replaces the writer's hardcoded pinhole one when a surfel frame is
       solved.  Checked against the projection's exact degree-0 homogeneity
       identities (``J.x = 0``, ``J(s.x) = J(x)/s``) at theta straddling 90
       degrees, where no image-plane form exists at all.
    2. ``_in_field`` / ``_cam_depth`` — the model-aware replacements for the
       ``-z > 0`` cheirality test and the ``-z`` depth measure that the culls,
       the collapse, the eviction gate and the infinity gate all took.
    3. The extent law.  A detection of ``r_px`` reference pixels subtends
       ``r_px / f`` radians everywhere under ``theta = r / f``, so at range
       ``d`` it spans ``r_px * d / f`` world units; the pinhole ``z / f`` form
       is that times ``cos(theta)``.  The projected patch is anisotropic by
       exactly ``theta / sin(theta)`` — the analytic periphery figure the real
       captures are measured against."""
    print("\nPhase 4 — embed / photometric layer under fisheye")
    B._CAM_WH = WH
    B.set_camera_context("EQUIDISTANT_FISHEYE", F_EQUI)
    check(B.fisheye_stage1(), "the bootstrap's fisheye gate is armed")
    cam = B.make_cam(F_EQUI)
    s_flip = np.array([1.0, -1.0, -1.0])

    # Camera-frame points (COLMAP +Z forward) over a theta sweep past 90 deg.
    degs = np.array([5.0, 40.0, 80.0, 89.0, 91.0, 100.0, 120.0])
    phis = np.array([0.0, 1.24, 2.71])
    xc_col, thetas = [], []
    for d in degs:
        for ph in phis:
            t = np.radians(d)
            rad = 1.0 + 0.7 * np.sin(3.0 * d)
            xc_col.append(
                rad
                * np.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)])
            )
            thetas.append(d)
    xc_col = np.ascontiguousarray(np.asarray(xc_col))
    thetas = np.asarray(thetas)
    n_past = int((thetas > 90).sum())
    check(
        n_past >= 6, f"the Jacobian sweep reaches past 90 deg ({n_past}/{len(thetas)})"
    )

    jac = B._colmap_proj_jacobian(cam, xc_col, s_flip)
    check(np.isfinite(jac).all(), "the Jacobian is finite at every theta")
    # Degree-0 homogeneity: the projection depends on the ray's DIRECTION only.
    jx = np.einsum("nij,nj->ni", jac, xc_col)
    scale = np.linalg.norm(jac, axis=(1, 2)) * np.linalg.norm(xc_col, axis=1)
    rel = np.abs(jx).max(axis=1) / np.maximum(scale, 1e-300)
    check(
        rel.max() < 1e-6,
        "J . x = 0 (the projection is scale-free)",
        f"max {rel.max():.2e}",
    )
    jac2 = B._colmap_proj_jacobian(cam, np.ascontiguousarray(2.0 * xc_col), s_flip)
    d_hom = np.abs(jac2 - 0.5 * jac).max() / np.abs(jac).max()
    check(d_hom < 1e-6, "J(2x) = J(x)/2", f"max rel {d_hom:.2e}")
    # The minimum-norm right inverse exists and inverts, at every theta — this
    # is what replaces the pinhole `(z/f)[I; 0]` in the surfel-frame solve.
    err_inv = max(
        float(np.abs(jac[k] @ np.linalg.pinv(jac[k]) - np.eye(2)).max())
        for k in range(len(jac))
    )
    check(err_inv < 1e-8, "J . pinv(J) = I at every theta", f"max {err_inv:.2e}")
    # ...and its null direction is the viewing ray itself.
    null_err = 0.0
    for k in range(len(jac)):
        b0 = np.linalg.pinv(jac[k])
        n = xc_col[k] / np.linalg.norm(xc_col[k])
        null_err = max(null_err, float(np.abs(b0.T @ n).max()))
    check(
        null_err < 1e-8,
        "the right inverse is orthogonal to the viewing ray",
        f"max {null_err:.2e}",
    )

    # `_in_field` / `_cam_depth`: canonical frame (-Z forward).
    xc_can = np.ascontiguousarray(xc_col * s_flip)
    fld = B._in_field(cam, xc_can)
    theta_max = np.degrees(0.5 * min(WH) / F_EQUI)
    want = thetas <= theta_max + 1e-9
    check(
        theta_max > 100.0 and np.array_equal(fld, want),
        "_in_field is the model's imaged cone, not the 90 deg half-space",
        f"cone {theta_max:.1f} deg, {int(fld.sum())}/{len(fld)} in field",
    )
    d_rng = B._cam_depth(xc_can)
    check(
        (d_rng > 0).all(),
        "_cam_depth is the ray range under fisheye (positive past 90 deg)",
        f"min {d_rng.min():.3f}",
    )
    B.set_camera_context("SIMPLE_PINHOLE", F_PIN)
    check(
        np.array_equal(B._in_field(cam, xc_can), -xc_can[:, 2] > 0)
        and np.allclose(B._cam_depth(xc_can), -xc_can[:, 2]),
        "both fall back to the pinhole half-space / -z reading with no context",
    )
    B.set_camera_context("EQUIDISTANT_FISHEYE", F_EQUI)

    # The extent law, measured through the model rather than asserted.
    r_px = 6.0
    bad_ratio, aniso_err = [], []
    for k in range(len(xc_col)):
        p = xc_can[k]
        d = float(np.linalg.norm(p))
        n_hat = p / d
        # Fronto square patch of the RANGE-rule half-size, in the tangent plane.
        u = np.cross(n_hat, [0.0, 0.0, 1.0] if abs(n_hat[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n_hat, u)
        ext = r_px * d / F_EQUI
        j = B._colmap_proj_jacobian(
            cam, np.ascontiguousarray((p * s_flip)[None, :]), s_flip
        )[0]
        # d(u,v)/d(s,t) of the patch, in canonical world-frame offsets.
        m = np.stack([j @ (u * ext * s_flip), j @ (v * ext * s_flip)], axis=1)
        sv = np.linalg.svd(m, compute_uv=False)
        t = np.radians(thetas[k])
        ideal = t / max(np.sin(t), 1e-12)
        # The RADIAL half-size is r_px at every theta under the range rule.
        bad_ratio.append(abs(sv.min() / r_px - 1.0))
        aniso_err.append(abs(sv.max() / sv.min() / ideal - 1.0))
    check(
        max(bad_ratio) < 1e-3,
        "the range extent rule projects to r_px at EVERY theta",
        f"max deviation {100 * max(bad_ratio):.3f}% over {len(bad_ratio)} samples",
    )
    check(
        max(aniso_err) < 1e-3,
        "the projected patch anisotropy is exactly theta / sin(theta)",
        f"max deviation {100 * max(aniso_err):.3f}%; at 120 deg the ratio is "
        f"{np.radians(120) / np.sin(np.radians(120)):.3f}",
    )
    # And the pinhole form the writer used shrinks by cos(theta) — the defect
    # in one number, at the theta band the real captures actually populate.
    shrink = np.cos(np.radians(75.0))
    check(
        abs(shrink - 0.2588) < 1e-3,
        "the pinhole z/f extent rule would shrink a 75 deg patch to cos(theta)",
        f"{100 * shrink:.1f}% of its true size",
    )


def run_finalization_suite():
    """Phase 5 — the finalization chain's model-dependent pieces.

    Two facts the cull chain rests on, on planted equidistant geometry.  A
    third — that ``recompute_point_errors`` keeps its past-90-degree
    observations, where the hardcoded perspective cheirality test plus the
    gnomonic ``(x/-z, y/-z)`` projection used to drop them and leave a fisheye
    point observed only past 90 degrees with the "no valid observation" error
    of 0.0 — needs an ``embedded_patches`` reconstruction and is pinned in Rust
    (``reconstruction::data::tests``); it matters because that array is what
    the points-at-infinity classifier calibrates its angular noise from.

    1. Rotation-locked resection scores its peripheral observations.  Its rows
       are ray-space and model-agnostic but SIGN-BLIND, and its trim gate used
       to be the perspective half-space — which scores every past-90-degree
       observation invalid and leaves the rotation core's translation on the
       on-axis subset.  The gate is now positive RANGE along the observed ray,
       which must still reject the antipodal reflection.
    2. ``PatchCloud.from_tracks`` under ``extent="pixel_radius"`` sizes a
       peripheral patch by the ray RANGE, not by ``|z|``."""
    from sfmtool._sfmtool.geometry import resect_translation
    from sfmtool._sfmtool.patches import CameraViews, PatchCloud

    print("\nPhase 5 — finalization chain under fisheye")
    B._CAM_WH = WH
    B.set_camera_context("EQUIDISTANT_FISHEYE", F_EQUI)
    cam = B.make_cam(F_EQUI)

    # -- 1: a wide scene whose observations straddle 90 degrees -------------
    rot = Rotation.from_rotvec([0.15, -0.1, 0.05])
    t_true = np.array([0.4, -0.25, 0.3])
    rng = np.random.default_rng(20260809)
    pts_w, uv, thetas = [], [], []
    while len(pts_w) < 120:
        th = rng.uniform(0.15, 1.9)
        ph = rng.uniform(-np.pi, np.pi)
        d = 3.0 + 2.0 * rng.random()
        c = d * np.array(
            [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)]
        )
        p = cam.ray_to_pixel_batch(np.ascontiguousarray(c[None, :]))[0]
        if not np.isfinite(p).all() or not (0 <= p[0] < WH[0] and 0 <= p[1] < WH[1]):
            continue
        pts_w.append(rot.inv().apply(c - t_true))
        uv.append(p)
        thetas.append(np.degrees(th))
    pts_w = np.ascontiguousarray(np.asarray(pts_w))
    uv = np.ascontiguousarray(np.asarray(uv))
    thetas = np.asarray(thetas)
    n_past = int((thetas > 90).sum())
    check(
        n_past >= 12,
        f"the wide scene reaches past 90 deg ({n_past}/{len(thetas)} observations)",
    )
    q = np.ascontiguousarray(rot.as_quat()[[3, 0, 1, 2]])
    out = resect_translation(
        camera=cam,
        rotation_wxyz=q,
        points=pts_w,
        uv=uv,
        max_error_px=2.0,
        min_inliers=10,
    )
    check(out is not None, "rotation-locked resection succeeds on the wide scene")
    if out is not None:
        inl = np.asarray(out["inliers"])
        check(
            bool(inl.all()),
            "resection keeps EVERY peripheral observation",
            f"{int(inl.sum())}/{len(inl)} kept, "
            f"{int(inl[thetas > 90].sum())}/{n_past} of the past-90-degree ones "
            "(the half-space gate kept 0 of them)",
        )
        err = float(np.linalg.norm(np.asarray(out["translation"]) - t_true))
        check(err < 1e-8, "resection recovers the planted translation", f"{err:.2e}")
    # The antipodal reflection is still refused: same observed ray, opposite
    # side of the camera — the one thing the sign-blind rows need the gate for.
    c0 = rot.apply(pts_w[0]) + t_true
    mirror_w = rot.inv().apply(-c0 - t_true)
    out2 = resect_translation(
        camera=cam,
        rotation_wxyz=q,
        points=np.ascontiguousarray(np.vstack([pts_w, mirror_w[None, :]])),
        uv=np.ascontiguousarray(np.vstack([uv, uv[0][None, :]])),
        max_error_px=2.0,
        min_inliers=10,
    )
    check(
        out2 is not None and not bool(np.asarray(out2["inliers"])[-1]),
        "resection still refuses the antipodal reflection of an observation",
    )

    # -- 2: from_tracks reads the camera model, not just the focal ----------
    th = np.radians(100.0)
    d = 4.0
    positions = np.array(
        [[d * np.sin(th), 0.0, -d * np.cos(th), 1.0]], dtype=np.float64
    )
    views = CameraViews(
        [cam],
        np.array([[1.0, 0.0, 0.0, 0.0]], np.float64),
        np.zeros((1, 3), np.float64),
    )
    cloud = PatchCloud.from_tracks(
        views,
        positions,
        np.array([0], np.uint32),
        np.array([0], np.uint32),
        normal="mean_viewing",
        extent="pixel_radius",
        extent_value=6.0,
        pixel_reduce="min",
    )
    got = float(cloud[0].half_extent[0])
    want = 6.0 * d / F_EQUI
    check(
        abs(got / want - 1.0) < 1e-9,
        "from_tracks sizes a 100 deg pixel_radius patch by the ray RANGE",
        f"{got:.6f} vs {want:.6f}; the |z| reading would give "
        f"{6.0 * d * abs(np.cos(th)) / F_EQUI:.6f}",
    )


def run_phase6_suite():
    """Phase 6: fleet integration.

    1. ``PatchExtent::FeatureSize`` is the UNIFORM Jacobian rule — the keypoint
       scale back-projected through the view's own camera, so a pinhole sizes by
       ``|z|`` and an equidistant fisheye by the ray range.  Before Phase 6 the
       finite branch applied the range reading to every model, which oversizes a
       perspective patch by ``sec(theta)``.
    2. The routing override is tri-state: only ``"0"`` refuses a confirmed
       verdict."""
    from sfmtool._sfmtool.patches import CameraViews, PatchCloud

    print("\nPhase 6 — fleet integration")
    B._CAM_WH = WH

    # -- 1: FeatureSize reads the view camera, not just its focal -----------
    # One point at 55 deg off axis, in domain for both models, one observation
    # with a 3 px keypoint scale.
    th, d, sigma, factor = np.radians(55.0), 4.0, 3.0, 2.5
    positions = np.array([[d * np.sin(th), 0.0, -d * np.cos(th), 1.0]], np.float64)
    tpi, tii = np.array([0], np.uint32), np.array([0], np.uint32)
    quat = np.array([[1.0, 0.0, 0.0, 0.0]], np.float64)
    trans = np.zeros((1, 3), np.float64)

    def feature_half(model):
        B.set_camera_context(model, F_EQUI)
        views = CameraViews([B.make_cam(F_EQUI)], quat, trans)
        cloud = PatchCloud.from_tracks(
            views,
            positions,
            tpi,
            tii,
            keypoint_scales=np.array([sigma], np.float64),
            normal="mean_viewing",
            extent="feature_size",
            extent_value=factor,
            feature_reduce="median",
        )
        return float(cloud[0].half_extent[0])

    try:
        got_eq = feature_half("EQUIDISTANT_FISHEYE")
        got_pin = feature_half("SIMPLE_PINHOLE")
    finally:
        B.set_camera_context("SIMPLE_PINHOLE", None)
    want_eq = factor * sigma * d / F_EQUI
    want_pin = factor * sigma * d * np.cos(th) / F_EQUI
    check(
        abs(got_eq / want_eq - 1.0) < 1e-9,
        "FeatureSize sizes an equidistant patch by the ray RANGE",
        f"{got_eq:.6f} vs {want_eq:.6f}",
    )
    check(
        abs(got_pin / want_pin - 1.0) < 1e-9,
        "FeatureSize sizes a pinhole patch by |z| (the uniform Jacobian rule)",
        f"{got_pin:.6f} vs {want_pin:.6f}; the pre-Phase-6 range reading would "
        f"give {want_eq:.6f}",
    )

    # -- 2: the routing override is tri-state -------------------------------
    check(
        F.fisheye_routing_override(None) is True
        and F.fisheye_routing_override("1") is True
        and F.fisheye_routing_override("0") is False,
        "a confirmed verdict routes unless SFMTOOL_FISHEYE_SEED=0 refuses it",
    )
    check(
        F.VOTE_BAND_FLOOR_LOG > 0 and F.VOTE_BAND_FLOOR_LOG < np.log(1.05),
        "the vote's precision band has a floor inside the measured accuracy",
        f"{F.VOTE_BAND_FLOOR_LOG:.5f}",
    )


def run_bspline_context_suite():
    """The promoted SFMTOOL_FISHEYE context — the finalization's spline rung.

    Mirrors the Phase-3a representation cross-check for the spline model:
    a ZERO spline parameterizes the SAME map as EQUIDISTANT_FISHEYE (the
    promotion moves nothing), a bent spline still inverts past 90 degrees,
    and the context plumbing (``make_cam``, ``fisheye_stage1``,
    ``focal_floor``, ``_field_theta_max``) reads the promoted model rather
    than falling through a default arm.  The model's own maps are pinned in
    Rust (``camera/distortion/tests.rs``); this covers only the script layer
    the promotion flows through."""
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    print("\nSFMTOOL_FISHEYE — promoted spline context")
    w, h = WH
    B._CAM_WH = WH
    theta_max = 0.5 * min(WH) / F_EQUI  # the zero-spline rim angle
    B.set_camera_context(
        "SFMTOOL_FISHEYE", F_EQUI, bspline=np.zeros(8), theta_max=theta_max
    )
    check(B.fisheye_stage1(), "the promoted context arms the fisheye gate")
    cam = B.make_cam(F_EQUI)
    check(
        cam.model == "SFMTOOL_FISHEYE",
        "make_cam builds the promoted model from the context",
        f"model = {cam.model}",
    )
    check(
        abs(B.focal_floor() - 0.075 * max(WH)) < 1e-12,
        "focal_floor keeps the FOV-derived fisheye floor after promotion",
        f"{B.focal_floor():.1f} px",
    )

    equi = CameraIntrinsics.from_dict(
        {
            "model": "EQUIDISTANT_FISHEYE",
            "width": w,
            "height": h,
            "parameters": {
                "focal_length": F_EQUI,
                "principal_point_x": w / 2.0,
                "principal_point_y": h / 2.0,
            },
        }
    )
    rays = []
    for deg in (0.0, 30.0, 60.0, 89.0, 90.0, 91.0, 95.0, 105.0):
        for phi_deg in (0.0, 71.0, 200.0):
            t, p = np.radians(deg), np.radians(phi_deg)
            rays.append([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), -np.cos(t)])
    rays = np.ascontiguousarray(np.asarray(rays, float))
    px_p = np.asarray(cam.ray_to_pixel_batch(rays))
    px_e = np.asarray(equi.ray_to_pixel_batch(rays))
    d_fwd = np.abs(px_p - px_e).max()
    check(
        d_fwd == 0.0,
        "a zero spline is the exact equidistant map (bit for bit)",
        f"max {d_fwd:.2e} px",
    )
    check(
        abs(B._field_theta_max(cam) - theta_max) < 1e-9,
        "_field_theta_max reads the rim through the model's own inverse",
        f"{np.degrees(B._field_theta_max(cam)):.1f} deg",
    )

    # A bent spline: make_cam_bspline builds it and the monotone spline still
    # inverts its own projection past 90 degrees.
    coeffs = np.array([-0.001, -0.004, -0.01, -0.02, -0.03, -0.05, -0.07, -0.09])
    camp = B.make_cam_bspline(F_EQUI, coeffs, 2.0)
    px_b = np.asarray(camp.ray_to_pixel_batch(rays))
    back = np.asarray(camp.pixel_to_ray_batch(np.ascontiguousarray(px_b)))
    d_rt = np.abs(back - rays).max()
    check(
        d_rt < 1e-9,
        "a bent spline round-trips pixel <-> ray past 90 deg",
        f"max {d_rt:.2e}",
    )
    d_bend = np.abs(px_b - px_e).max()
    check(
        d_bend > 1.0,
        "the bent spline actually moves the periphery",
        f"max {d_bend:.1f} px",
    )


def main():
    print("Fisheye seed Phase-1 primitive gate")
    check_model_equivalence()
    for module, name in ((F, "exp_fast_seed"), (B, "exp_pinhole_bootstrap")):
        check(
            module.camera_context()["model"] == "SIMPLE_PINHOLE",
            f"{name}: default camera context is SIMPLE_PINHOLE",
        )
        try:
            # Equidistant: rig inside a point shell, observations to 105 deg.
            run_suite(
                module,
                f"{name} — equidistant fisheye",
                "EQUIDISTANT_FISHEYE",
                F_EQUI,
                shell=6.0,
                rig=1.2,
                theta_max=105.0,
                min_behind=60,
            )
            # Pinhole control: the same generator on a narrow-FOV scene.
            run_suite(
                module,
                f"{name} — pinhole control",
                "SIMPLE_PINHOLE",
                F_PIN,
                shell=6.0,
                rig=0.6,
                theta_max=25.0,
                min_behind=0,
            )
        finally:
            module.set_camera_context("SIMPLE_PINHOLE", None)

    try:
        run_ray_suite()
    finally:
        F.set_camera_context("SIMPLE_PINHOLE", None)

    try:
        run_release_suite()
    finally:
        F.set_camera_context("SIMPLE_PINHOLE", None)

    try:
        run_embed_suite()
    finally:
        B.set_camera_context("SIMPLE_PINHOLE", None)

    try:
        run_finalization_suite()
    finally:
        B.set_camera_context("SIMPLE_PINHOLE", None)

    try:
        run_phase6_suite()
    finally:
        B.set_camera_context("SIMPLE_PINHOLE", None)

    try:
        run_bspline_context_suite()
    finally:
        B.set_camera_context("SIMPLE_PINHOLE", None)

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {'; '.join(_FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
