# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the patch tests.

These three were copy-pasted across six of the modules in this directory
before it existed; import them from here rather than adding a fresh copy.
"""

import os

import numpy as np


def load_images(recon) -> list[np.ndarray]:
    """Every image of a reconstruction, decoded to contiguous RGB."""
    import cv2  # heavy module, only needed by the integration tests

    ws = recon.workspace_dir
    images = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        images.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return images


def sample_point_ids(cloud, n: int = 200, seed: int = 0) -> list[int]:
    """A deterministic point-id subset, to keep the per-point search fast."""
    ids = np.asarray(cloud.point_indexes)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False)).tolist()


def rotation_matrices(recon) -> np.ndarray:
    """Per-image world-to-camera rotation matrices (Mx3x3) from wxyz quaternions."""
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w * w + x * x + y * y + z * z
    s = np.where(n > 0, 2.0 / n, 0.0)
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - s * (y * y + z * z)
    R[:, 0, 1] = s * (x * y - z * w)
    R[:, 0, 2] = s * (x * z + y * w)
    R[:, 1, 0] = s * (x * y + z * w)
    R[:, 1, 1] = 1 - s * (x * x + z * z)
    R[:, 1, 2] = s * (y * z - x * w)
    R[:, 2, 0] = s * (x * z - y * w)
    R[:, 2, 1] = s * (y * z + x * w)
    R[:, 2, 2] = 1 - s * (x * x + y * y)
    return R
