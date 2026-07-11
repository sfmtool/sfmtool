# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the refine_cluster_patches Rust binding: tiny synthetic images
end-to-end, dict schema/dtypes, and input validation."""

import numpy as np
import pytest

from sfmtool._sfmtool.matching import refine_cluster_patches

# matches_format::ClusterMemberStatus discriminants.
STATUS_REFERENCE = 0
STATUS_KEPT = 1
STATUS_REJECTED_LOW_ZNCC = 2
STATUS_NOT_EVALUATED = 5
STATUS_REJECTED_UNLOCALIZABLE = 6


def _texture(w: int, h: int) -> np.ndarray:
    """Smooth deterministic texture (no clipping)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    x += 0.5
    y += 0.5
    v = (
        127.0
        + 50.0 * np.sin(0.11 * x + 0.06 * y + 1.3)
        + 35.0 * np.sin(0.05 * x - 0.12 * y + 0.7)
        + 20.0 * np.sin(0.17 * x + 0.13 * y + 2.9)
    )
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def _inputs(shift=(2.0, 1.0)):
    """Two 96x96 images (the second a rolled copy), one 2-member cluster."""
    img0 = _texture(96, 96)
    # np.roll moves content by +shift, so the point at p in img0 appears at
    # p + shift in img1 (interior support; the wrap seam is far away).
    img1 = np.roll(img0, (int(shift[1]), int(shift[0])), axis=(0, 1))
    positions = [
        np.array([[48.0, 48.0]], dtype=np.float32),
        np.array([[48.0, 48.0]], dtype=np.float32),
    ]
    affine = np.array([[[3.0, 0.0], [0.0, 3.0]]], dtype=np.float32)
    affine_shapes = [affine.copy(), affine.copy()]
    cluster_starts = np.array([0, 2], dtype=np.uint32)
    member_images = np.array([0, 1], dtype=np.uint32)
    member_features = np.array([0, 0], dtype=np.uint32)
    return (
        [img0, img1],
        positions,
        affine_shapes,
        cluster_starts,
        member_images,
        member_features,
    )


class TestRefineClusterPatches:
    def test_schema_and_recovery(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        result = refine_cluster_patches(images, pos, aff, starts, m_img, m_feat)

        assert set(result.keys()) == {
            "reference_members",
            "member_status",
            "member_affines",
            "member_zncc",
            "member_shift_px",
        }
        assert result["reference_members"].dtype == np.uint32
        assert result["reference_members"].shape == (1,)
        assert result["member_status"].dtype == np.uint8
        assert result["member_status"].shape == (2,)
        assert result["member_affines"].dtype == np.float64
        assert result["member_affines"].shape == (2, 2, 3)
        assert result["member_zncc"].dtype == np.float32
        assert result["member_shift_px"].dtype == np.float32

        # Equal scales tie-break to the lowest global member index.
        assert result["reference_members"][0] == 0
        assert result["member_status"][0] == STATUS_REFERENCE
        assert result["member_zncc"][0] == pytest.approx(1.0)
        # Reference row: identity affine.
        np.testing.assert_allclose(result["member_affines"][0], [[1, 0, 0], [0, 1, 0]])

        # The member is a pure translation of the reference: kept, affine
        # close to [I | shift - (pos_mem - pos_ref)] composed absolutely.
        assert result["member_status"][1] == STATUS_KEPT
        assert result["member_zncc"][1] > 0.95
        a = result["member_affines"][1]
        np.testing.assert_allclose(a[:, :2], np.eye(2), atol=0.02)
        # x_mem = A x_ref + t with the true map x + (2, 1).
        ref = np.array([48.0, 48.0])
        mapped = a[:, :2] @ ref + a[:, 2]
        np.testing.assert_allclose(mapped, ref + np.array([2.0, 1.0]), atol=0.15)

    def test_out_of_range_feature_is_not_evaluated(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        m_feat = np.array([0, 7], dtype=np.uint32)  # image 1 has 1 feature
        result = refine_cluster_patches(images, pos, aff, starts, m_img, m_feat)
        # Fewer than 2 usable members -> the whole cluster is unrefinable.
        assert result["reference_members"][0] == 0xFFFFFFFF
        assert result["member_status"].tolist() == [
            STATUS_NOT_EVALUATED,
            STATUS_NOT_EVALUATED,
        ]
        assert np.isnan(result["member_zncc"]).all()

    def test_unlocalizable_member_excluded(self):
        # The member's image is flat: its own patch has no gradient signal,
        # so the localizability gate excludes it before refinement and the
        # 2-member cluster becomes unrefinable.
        images, pos, aff, starts, m_img, m_feat = _inputs()
        images[1] = np.full_like(images[1], 127)
        result = refine_cluster_patches(images, pos, aff, starts, m_img, m_feat)
        assert result["member_status"][1] == STATUS_REJECTED_UNLOCALIZABLE
        assert np.isnan(result["member_zncc"][1])
        assert result["reference_members"][0] == 0xFFFFFFFF

        # Disabling the gate re-admits the member; the flat patch then fails
        # the downstream ZNCC vet instead.
        result = refine_cluster_patches(
            images, pos, aff, starts, m_img, m_feat, max_keypoint_uncertainty=0.0
        )
        assert result["member_status"][1] == STATUS_REJECTED_LOW_ZNCC

    def test_progress_counter_ticks(self):
        from sfmtool._sfmtool import ProgressCounter

        images, pos, aff, starts, m_img, m_feat = _inputs()
        counter = ProgressCounter()
        refine_cluster_patches(
            images, pos, aff, starts, m_img, m_feat, progress=counter
        )
        assert counter.value == 1  # one tick per finished cluster


class TestValidation:
    def test_mismatched_list_lengths(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        with pytest.raises(ValueError, match="must be parallel"):
            refine_cluster_patches(images, pos[:1], aff, starts, m_img, m_feat)

    def test_bad_csr(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        bad = np.array([1, 2], dtype=np.uint32)
        with pytest.raises(ValueError, match="cluster_starts"):
            refine_cluster_patches(images, pos, aff, bad, m_img, m_feat)

    def test_member_image_out_of_range(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        bad = np.array([0, 5], dtype=np.uint32)
        with pytest.raises(ValueError, match="out of range"):
            refine_cluster_patches(images, pos, aff, starts, bad, m_feat)

    def test_bad_shapes(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        pos = [p.copy() for p in pos]
        pos[1] = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            refine_cluster_patches(images, pos, aff, starts, m_img, m_feat)
