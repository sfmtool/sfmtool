# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The sizes in pixels of each image's radial correction and rolling shutter.

``sfm inspect --verbose`` summarises them per camera and ``sfm analyze
--metrics`` lists them per image. Both sizes are read through the camera model
rather than from a second spelling of its formulas, so they mean what the
projection does with the stored values.
"""

import numpy as np

# Where the rolling shutter's displacement is measured, by the scanlines
# setting and the side of the image farther from the principal point.
_EDGE_NAMES = {
    ("ROWS", True): "bottom edge centre",
    ("ROWS", False): "top edge centre",
    ("COLUMNS", True): "right edge centre",
    ("COLUMNS", False): "left edge centre",
}


def rolling_shutter_edge(camera, scanlines: str) -> tuple[str, float, float]:
    """The pixel at which a camera's rolling-shutter displacement is measured,
    and its name: the centre of the image edge whose scanlines are read
    farthest from the scanline through the camera's principal point.

    For ``ROWS`` that is the bottom edge's centre ``(width/2, height)`` when
    the principal point lies in the top half or on the middle, else the top
    edge's ``(width/2, 0)``; for ``COLUMNS`` the right or left edge's centre.
    Returns ``(name, u, v)``.
    """
    cx, cy = camera.principal_point
    w, h = camera.width, camera.height
    if scanlines == "ROWS":
        far = cy <= h / 2
        u, v = w / 2, (h if far else 0.0)
    else:
        far = cx <= w / 2
        u, v = (w if far else 0.0), h / 2
    return _EDGE_NAMES[(scanlines, far)], u, v


def image_correction_sizes(recon) -> dict[str, np.ndarray]:
    """Per image of ``recon``, the size of its radial correction and of its
    rolling shutter, in pixels.

    ``"radial_bspline"``: ``f_i·e_i1``, the image's radial correction at its
    camera's spline domain end ``ρ_max`` (``Δδ_i(ρ_max) = e_i1`` in focal
    units) times the image's own focal, signed: positive moves a pixel there
    outward. ``"rolling_shutter"``: how far the image's rates move the pixel at
    its camera's :func:`rolling_shutter_edge`, found by turning that pixel into
    a ray through the image's intrinsics without its rates and projecting the
    ray back through them with the rates. Each is an ``(N,)`` float64 array,
    NaN for an image whose camera does not vary that correction per image (or,
    for the rolling shutter, has no scanlines setting).
    """
    n = recon.image_count
    sizes = {
        "radial_bspline": np.full(n, np.nan),
        "rolling_shutter": np.full(n, np.nan),
    }
    per_image = recon.per_image_adjustments
    scanlines = recon.rolling_shutter_scanlines
    cameras = recon.cameras
    camera_indexes = np.asarray(recon.camera_indexes)
    radial = recon.radial_bspline_adjustments
    rates = recon.rolling_shutter_rates_xyz
    focal = recon.focal_adjustments
    principal_point = recon.principal_point_adjustments_xy
    edges = {
        c: rolling_shutter_edge(cameras[c], scanlines[c])
        for c in range(len(cameras))
        if scanlines[c] is not None
    }
    for i in range(n):
        c = int(camera_indexes[i])
        varies = per_image[c]
        if "radial_bspline" in varies:
            e_1 = 0.0 if radial is None else float(radial[i][1])
            f_i = recon.image_intrinsics(i).focal_lengths[0]
            sizes["radial_bspline"][i] = f_i * e_1
        if "rolling_shutter" in varies and c in edges:
            if rates is None or not np.any(rates[i]):
                sizes["rolling_shutter"][i] = 0.0
                continue
            # The image's intrinsics, and the same without its rates.
            with_rates = recon.image_intrinsics(i)
            without = cameras[c].with_adjustment(
                focal=0.0 if focal is None else float(focal[i]),
                principal_point=(0.0, 0.0)
                if principal_point is None
                else tuple(float(x) for x in principal_point[i]),
                radial_bspline=(0.0, 0.0)
                if radial is None
                else tuple(float(x) for x in radial[i]),
            )
            _, u, v = edges[c]
            ray = without.pixel_to_ray(u, v)
            moved = with_rates.ray_to_pixel(ray)
            sizes["rolling_shutter"][i] = (
                np.nan if moved is None else float(np.hypot(moved[0] - u, moved[1] - v))
            )
    return sizes
