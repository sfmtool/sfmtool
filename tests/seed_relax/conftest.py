# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the `scripts/seed_relax` tests.

Two things: `scripts/` goes on `sys.path` so the package imports under its own
name, and, where the environment has no SciPy, a minimal numpy-backed
`scipy.spatial.transform.Rotation` is registered so `seed_candidate_eval` (which
the package borrows `Member` and the ray helpers from) is importable. The
package itself uses numpy alone; the stand-in exists only to let its dependency
load, and it implements exactly the conversions that dependency calls.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class _Rotation:
    """The subset of `scipy.spatial.transform.Rotation` used upstream."""

    def __init__(self, mats):
        self._m = np.asarray(mats, float)
        self._single = self._m.ndim == 2
        if self._single:
            self._m = self._m[None]

    @staticmethod
    def _from_quat_xyzw(q):
        q = np.atleast_2d(np.asarray(q, float))
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.stack(
            [
                np.stack(
                    [
                        1 - 2 * (y * y + z * z),
                        2 * (x * y - w * z),
                        2 * (x * z + w * y),
                    ],
                    1,
                ),
                np.stack(
                    [
                        2 * (x * y + w * z),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z - w * x),
                    ],
                    1,
                ),
                np.stack(
                    [
                        2 * (x * z - w * y),
                        2 * (y * z + w * x),
                        1 - 2 * (x * x + y * y),
                    ],
                    1,
                ),
            ],
            1,
        )

    @classmethod
    def from_quat(cls, q):
        single = np.asarray(q, float).ndim == 1
        mats = cls._from_quat_xyzw(q)
        return cls(mats[0] if single else mats)

    @classmethod
    def from_rotvec(cls, v):
        v = np.asarray(v, float)
        single = v.ndim == 1
        v = np.atleast_2d(v)
        theta = np.linalg.norm(v, axis=1)
        half = 0.5 * theta
        scale = np.where(theta > 1e-12, np.sin(half) / np.maximum(theta, 1e-300), 0.5)
        q = np.concatenate([v * scale[:, None], np.cos(half)[:, None]], axis=1)
        mats = cls._from_quat_xyzw(q)
        return cls(mats[0] if single else mats)

    @classmethod
    def from_matrix(cls, r):
        return cls(np.asarray(r, float))

    def as_matrix(self):
        return self._m[0] if self._single else self._m

    def as_quat(self):
        """XYZW quaternions, by the branch-free Bar-Itzhack eigenvector."""
        out = []
        for r in self._m:
            k = (
                np.array(
                    [
                        [
                            r[0, 0] + r[1, 1] + r[2, 2],
                            r[2, 1] - r[1, 2],
                            r[0, 2] - r[2, 0],
                            r[1, 0] - r[0, 1],
                        ],
                        [
                            r[2, 1] - r[1, 2],
                            r[0, 0] - r[1, 1] - r[2, 2],
                            r[0, 1] + r[1, 0],
                            r[0, 2] + r[2, 0],
                        ],
                        [
                            r[0, 2] - r[2, 0],
                            r[0, 1] + r[1, 0],
                            r[1, 1] - r[0, 0] - r[2, 2],
                            r[1, 2] + r[2, 1],
                        ],
                        [
                            r[1, 0] - r[0, 1],
                            r[0, 2] + r[2, 0],
                            r[1, 2] + r[2, 1],
                            r[2, 2] - r[0, 0] - r[1, 1],
                        ],
                    ],
                    float,
                )
                / 3.0
            )
            q = np.linalg.eigh(k)[1][:, -1]
            q = q if q[0] >= 0 else -q
            out.append(np.array([q[1], q[2], q[3], q[0]]))
        arr = np.stack(out)
        return arr[0] if self._single else arr

    def as_rotvec(self):
        q = np.atleast_2d(self.as_quat())
        v = q[:, :3]
        n = np.linalg.norm(v, axis=1)
        angle = 2.0 * np.arctan2(n, q[:, 3])
        scale = np.where(n > 1e-12, angle / np.maximum(n, 1e-300), 0.0)
        arr = v * scale[:, None]
        return arr[0] if self._single else arr

    def magnitude(self):
        c = (np.trace(self._m, axis1=1, axis2=2) - 1.0) / 2.0
        arr = np.arccos(np.clip(c, -1.0, 1.0))
        return float(arr[0]) if self._single else arr


def _install_rotation_stand_in():
    scipy_mod = types.ModuleType("scipy")
    spatial = types.ModuleType("scipy.spatial")
    transform = types.ModuleType("scipy.spatial.transform")
    transform.Rotation = _Rotation
    spatial.transform = transform
    scipy_mod.spatial = spatial
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.spatial", spatial)
    sys.modules.setdefault("scipy.spatial.transform", transform)


if importlib.util.find_spec("scipy") is None:
    _install_rotation_stand_in()
