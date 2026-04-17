# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the _to_nerfstudio core helpers."""

import numpy as np
import pytest

from sfmtool._to_nerfstudio import (
    _APPLIED_TRANSFORM_3x4,
    apply_transform_to_points,
    frame_transform_matrix,
    write_sparse_ply,
)


class TestFrameTransformMatrix:
    def test_identity_pose_yields_applied_transform(self):
        """Identity cam-from-world inverts to identity world-from-cam.

        After flipping Y/Z and applying the nerfstudio applied_transform, the
        result should match _APPLIED_TRANSFORM_3x4 with [-1, -1] flips on the
        OpenGL columns.
        """
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity wxyz
        t = np.zeros(3)
        m = frame_transform_matrix(q, t)
        assert m.shape == (4, 4)
        np.testing.assert_array_almost_equal(m[3], [0.0, 0.0, 0.0, 1.0])

        # With identity input, world_from_cam is identity; after flipping Y/Z
        # columns we get diag(1, -1, -1, 1). Then applied_transform left-mults.
        flipped = np.diag([1.0, -1.0, -1.0, 1.0])
        expected = np.vstack([_APPLIED_TRANSFORM_3x4, [0, 0, 0, 1]]) @ flipped
        np.testing.assert_array_almost_equal(m, expected)

    def test_translation_only_pose(self):
        """A translation in cam-from-world flips sign in world-from-cam, then
        gets remapped by the applied_transform. Verify the final translation
        column matches the manual computation."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        t = np.array([1.0, 2.0, 3.0])  # cam_from_world translation
        m = frame_transform_matrix(q, t)

        # world_from_cam translation = -R^T @ t = -t (since R=I)
        # Then [:, 1] *= -1 leaves translation column unchanged (only flips
        # rotation columns 1,2). applied_transform permutes Y<->Z, negates Y.
        # So expected translation = applied_transform[:3, :3] @ (-t).
        wfc_t = -t
        expected_t = _APPLIED_TRANSFORM_3x4[:, :3] @ wfc_t
        np.testing.assert_array_almost_equal(m[:3, 3], expected_t)


class TestApplyTransformToPoints:
    def test_axes_permuted(self):
        """applied_transform is [[1,0,0,0],[0,0,1,0],[0,-1,0,0]]:
        x -> x, y -> -z (output y), z -> y (output z) ... actually:
        out_x = x; out_y = z; out_z = -y.
        """
        pts = np.array([[1.0, 2.0, 3.0]])
        out = apply_transform_to_points(pts)
        np.testing.assert_array_almost_equal(out, [[1.0, 3.0, -2.0]])

    def test_empty_input(self):
        out = apply_transform_to_points(np.zeros((0, 3)))
        assert out.shape == (0, 3)


class TestWriteSparsePly:
    def test_header_and_count(self, tmp_path):
        pts = np.array([[1.5, 2.5, 3.5], [-1.0, 0.0, 1.0]])
        cols = np.array([[10, 20, 30], [255, 0, 128]], dtype=np.uint8)
        out = tmp_path / "x.ply"
        write_sparse_ply(out, pts, cols)
        text = out.read_text(encoding="ascii")
        lines = text.splitlines()
        assert lines[0] == "ply"
        assert lines[1] == "format ascii 1.0"
        assert lines[2] == "element vertex 2"
        assert "end_header" in lines
        # 10 header lines (ply, format, element, 6 properties, end_header)
        # + 2 data lines = 12 total
        assert len(lines) == 12
        assert lines[10] == "1.500000 2.500000 3.500000 10 20 30"
        assert lines[11] == "-1.000000 0.000000 1.000000 255 0 128"

    def test_shape_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError):
            write_sparse_ply(
                tmp_path / "bad.ply",
                np.zeros((3, 3)),
                np.zeros((2, 3), dtype=np.uint8),
            )
