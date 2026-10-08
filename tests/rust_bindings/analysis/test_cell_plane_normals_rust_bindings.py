# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shapes, dtypes and argument checks of `_sfmtool.analysis.cell_plane_normals`."""

import numpy as np
import pytest

from sfmtool._sfmtool.analysis import cell_plane_normals
from sfmtool._sfmtool.geometry import CameraIntrinsics

_F = 800.0
_CX, _CY = 320.0, 240.0


def _camera():
    return CameraIntrinsics(
        "SIMPLE_PINHOLE",
        640,
        480,
        {"focal_length": _F, "principal_point_x": _CX, "principal_point_y": _CY},
    )


def _pixel(centre, x):
    """Pixel of world point `x` in an unrotated camera at `centre` (looking down -Z)."""
    p = np.asarray(x, float) - np.asarray(centre, float)
    return np.array([_F * p[0] / -p[2] + _CX, _F * -(p[1] / -p[2]) + _CY])


def _scene(n_clusters=2):
    """Clusters on the plane z = 0, seen by four unrotated cameras at height 10.

    The plane is parallel to every image plane, so each member's affine shape
    is exact and every cell displacement is zero. The last image is unposed.
    """
    centres = [(0.0, 0.0, 10.0), (2.5, 0.0, 10.0), (0.0, 2.5, 10.0), (-2.5, -1.0, 10.0)]
    n_img = len(centres) + 1
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n_img, 1))
    trans = np.array([[-c[0], -c[1], -c[2]] for c in centres] + [[np.nan] * 3])
    starts, refs, images, status, pos, shapes = [0], [], [], [], [], []
    for c in range(n_clusters):
        point = (0.1 * c, -0.05 * c, 0.0)
        refs.append(len(images))
        for i, centre in enumerate(centres + [(1.0, 1.0, 10.0)]):
            images.append(i)
            status.append(0 if i == 0 else 1)
            pos.append(_pixel(centre, point) if i < len(centres) else (100.0, 100.0))
            shapes.append([[3.0, 0.0], [0.0, 3.0]])
        starts.append(len(images))
    k = len(images)
    shifts = np.zeros((k, 3, 3, 2), np.float32)
    cells = np.zeros((k, 3, 3), np.uint8)
    cells[np.array(status) == 0] = 3
    shifts[np.array(status) == 0] = np.nan
    return dict(
        cluster_starts=np.array(starts, np.uint32),
        reference_members=np.array(refs, np.uint32),
        member_images=np.array(images, np.uint32),
        member_status=np.array(status, np.uint8),
        member_positions=np.array(pos, np.float32),
        member_affine_shapes=np.array(shapes, np.float32),
        member_cell_shift_px=shifts,
        member_cell_status=cells,
        patch_size=12.0,
        resolution=25,
        cameras=[_camera()],
        image_camera=np.zeros(n_img, np.uint32),
        quaternions_wxyz=quats,
        translations=trans,
    )


def test_cell_plane_normals_shapes_and_dtypes():
    out = cell_plane_normals(**_scene(2))
    for key, shape, dtype in [
        ("normal", (2, 3), np.float64),
        ("determinacy", (2,), np.uint8),
        ("free_axis", (2, 3), np.float64),
        ("view_dir", (2, 3), np.float64),
        ("cell_positions", (2, 3, 3, 3), np.float64),
        ("cell_rays", (2, 3, 3), np.uint32),
        ("cell_status", (2, 3, 3), np.uint8),
        ("cell_weight", (2, 3, 3), np.float64),
        ("cell_residual_px", (2, 3, 3), np.float64),
        ("plane_rms", (2,), np.float64),
        ("anisotropy", (2,), np.float64),
        ("n_eff", (2,), np.float64),
    ]:
        assert out[key].shape == shape, key
        assert out[key].dtype == dtype, key
    assert out["determinacy_names"] == ["none", "one_axis", "both_axes"]
    assert out["cell_status_names"][0] == "in_plane"


def test_cell_plane_normals_recovers_a_fronto_parallel_plane():
    out = cell_plane_normals(**_scene(2))
    assert (out["determinacy"] == 2).all()
    np.testing.assert_allclose(out["normal"], [[0.0, 0.0, 1.0]] * 2, atol=1e-6)
    assert np.isnan(out["free_axis"]).all()
    # The reference and three kept members are posed; the fifth image is not.
    assert (out["cell_rays"] == 4).all()
    assert (out["cell_status"] == 0).all()
    np.testing.assert_allclose(out["cell_positions"][..., 2], 0.0, atol=1e-6)


def test_cell_plane_normals_rejects_mismatched_shapes():
    args = _scene(1)
    args["member_cell_status"] = args["member_cell_status"][:, :2]
    with pytest.raises(ValueError, match="member_cell_status"):
        cell_plane_normals(**args)


def test_cell_plane_normals_rejects_an_unknown_cell_status():
    args = _scene(1)
    args["member_cell_status"][1, 0, 0] = 200
    with pytest.raises(ValueError, match="cell status"):
        cell_plane_normals(**args)


def test_cell_plane_normals_floors_the_residual_at_the_cell_shift_precision():
    # The rays meet exactly, so the cell weight is set by the floor alone:
    # 0.1 grid px times patch_size / resolution times the shape's gain of 3.
    # Doubling the shift precision doubles the floor and quarters the weight.
    base = cell_plane_normals(**_scene(1))
    coarse = cell_plane_normals(**_scene(1), cell_shift_precision_grid_px=0.2)
    np.testing.assert_allclose(
        coarse["cell_weight"], base["cell_weight"] / 4, rtol=1e-9
    )
    np.testing.assert_allclose(coarse["normal"], base["normal"], atol=1e-9)
