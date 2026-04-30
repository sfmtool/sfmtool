# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RotQuaternion PyO3 bindings."""

import numpy as np
import pytest

from sfmtool._sfmtool import RotQuaternion


def _reference_matrix() -> np.ndarray:
    """A non-symmetric rotation matrix where transpose != original."""
    q = RotQuaternion.from_axis_angle([1.0, 2.0, 3.0], 0.7)
    return q.to_rotation_matrix()


class TestFromRotationMatrixMemoryLayout:
    """from_rotation_matrix must read logical (row, col) indices, not raw memory order."""

    def test_c_order_round_trip(self):
        r = np.ascontiguousarray(_reference_matrix())
        q = RotQuaternion.from_rotation_matrix(r)
        assert np.allclose(q.to_rotation_matrix(), r, atol=1e-12)

    def test_f_order_matches_c_order(self):
        r_c = np.ascontiguousarray(_reference_matrix())
        r_f = np.asfortranarray(r_c)
        # Sanity: same logical content, different memory layout.
        assert np.array_equal(r_c, r_f)
        assert r_f.flags["F_CONTIGUOUS"] and not r_f.flags["C_CONTIGUOUS"]

        q_c = RotQuaternion.from_rotation_matrix(r_c)
        q_f = RotQuaternion.from_rotation_matrix(r_f)

        assert np.allclose(q_f.to_wxyz_array(), q_c.to_wxyz_array(), atol=1e-12)
        # And the round-trip matches the *original* matrix, not its transpose.
        assert np.allclose(q_f.to_rotation_matrix(), r_c, atol=1e-12)

    def test_non_contiguous_slice(self):
        r_c = np.ascontiguousarray(_reference_matrix())
        # Embed in a larger array and slice out a non-contiguous 3x3 view.
        big = np.zeros((4, 4), dtype=np.float64)
        big[:3, :3] = r_c
        view = big[:3, :3]
        # The slice is still C-contiguous here because it starts at (0,0);
        # build a genuinely non-contiguous view by stepping.
        big2 = np.zeros((6, 6), dtype=np.float64)
        big2[::2, ::2] = r_c
        sliced = big2[::2, ::2]
        assert not sliced.flags["C_CONTIGUOUS"]
        assert not sliced.flags["F_CONTIGUOUS"]

        q = RotQuaternion.from_rotation_matrix(sliced)
        assert np.allclose(q.to_rotation_matrix(), r_c, atol=1e-12)
        # Smoke-check the embedded view too.
        q_view = RotQuaternion.from_rotation_matrix(view)
        assert np.allclose(q_view.to_rotation_matrix(), r_c, atol=1e-12)

    def test_f_order_is_not_silently_transposed(self):
        """Direct assertion of the original bug: F-order input != transpose result."""
        r_c = np.ascontiguousarray(_reference_matrix())
        r_f = np.asfortranarray(r_c)
        q_f = RotQuaternion.from_rotation_matrix(r_f)
        recovered = q_f.to_rotation_matrix()
        # The bug would produce R^T here; assert it does NOT.
        assert not np.allclose(recovered, r_c.T, atol=1e-6) or np.allclose(
            r_c, r_c.T, atol=1e-6
        )
        assert np.allclose(recovered, r_c, atol=1e-12)


class TestFromRotationMatrixShape:
    def test_rejects_non_3x3(self):
        with pytest.raises(ValueError, match="3x3"):
            RotQuaternion.from_rotation_matrix(np.eye(4))
