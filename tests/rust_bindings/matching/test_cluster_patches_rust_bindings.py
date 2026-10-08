# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the refine_cluster_patches Rust binding: tiny synthetic images
end-to-end, dict schema/dtypes, and input validation."""

import numpy as np
import pytest

from sfmtool.matching import refine_cluster_patches

# sfmtool_matches_format::ClusterMemberStatus discriminants.
STATUS_REFERENCE = 0
STATUS_KEPT = 1
STATUS_REJECTED_LOW_ZNCC = 2
STATUS_NOT_EVALUATED = 5
STATUS_REJECTED_UNLOCALIZABLE = 6

# sfmtool_matches_format::ClusterCellStatus discriminants.
CELL_FITTED = 0
CELL_NOT_ATTEMPTED = 3
# sfmtool_core's LoopStop discriminants, as `member_cell_loop_stop` returns them.
LOOP_NOT_RUN = 0


def _texture(w: int, h: int) -> np.ndarray:
    """Deterministic texture (no clipping).

    The three fine terms (periods near 5 to 7 px) make a member's own patch
    pin a position under the member gate; the smooth terms alone match
    themselves 3 template-grid px away and every member would be refused at
    the default. With only the first two fine terms, the second member's patch
    matches itself at the template-grid shift (-2, 3) on the edge of the
    square the gate searches, and reads 3; the third, in a third direction,
    removes that near-repeat.
    """
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    x += 0.5
    y += 0.5
    v = (
        127.0
        + 45.0 * np.sin(0.11 * x + 0.06 * y + 1.3)
        + 30.0 * np.sin(0.05 * x - 0.12 * y + 0.7)
        + 20.0 * np.sin(0.17 * x + 0.13 * y + 2.9)
        + 15.0 * np.sin(0.83 * x + 0.47 * y + 0.2)
        + 12.0 * np.sin(-0.52 * x + 0.88 * y + 1.1)
        + 10.0 * np.sin(0.61 * x - 0.71 * y + 2.3)
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
            "member_positions",
            "member_affine_shapes",
            "member_zncc",
            "member_zncc_middle",
            "member_zncc_grid",
            "member_shift_px",
            "member_consistency_residual",
            "member_cell_shift_px",
            "member_cell_zncc",
            "member_cell_status",
            "member_cell_iterations",
            "member_cell_loop_stop",
            "member_cell_update_accepted",
            "piecewise_options",
        }
        # Without piecewise=True the per-cell columns are absent.
        for key in (
            "member_cell_shift_px",
            "member_cell_zncc",
            "member_cell_status",
            "member_cell_iterations",
            "member_cell_loop_stop",
            "member_cell_update_accepted",
            "piecewise_options",
        ):
            assert result[key] is None, key
        assert result["reference_members"].dtype == np.uint32
        assert result["reference_members"].shape == (1,)
        assert result["member_status"].dtype == np.uint8
        assert result["member_status"].shape == (2,)
        assert result["member_positions"].dtype == np.float64
        assert result["member_positions"].shape == (2, 2)
        assert result["member_affine_shapes"].dtype == np.float64
        assert result["member_affine_shapes"].shape == (2, 2, 2)
        assert result["member_zncc"].dtype == np.float32
        assert result["member_zncc_grid"].dtype == np.float32
        assert result["member_zncc_grid"].shape == (2, 3, 3)
        assert result["member_shift_px"].dtype == np.float32
        assert result["member_consistency_residual"].dtype == np.float32
        assert result["member_consistency_residual"].shape == (2,)
        # A single pure-translation cluster fits the factorization exactly.
        assert np.isfinite(result["member_consistency_residual"]).all()
        assert (result["member_consistency_residual"] < 0.05).all()

        # Equal scales tie-break to the lowest global member index.
        assert result["reference_members"][0] == 0
        assert result["member_status"][0] == STATUS_REFERENCE
        assert result["member_zncc"][0] == pytest.approx(1.0)
        # The reference is refined against itself: its own detector affine
        # shape (3*I here) and its own keypoint position.
        s_ref = np.array([[3.0, 0.0], [0.0, 3.0]])
        np.testing.assert_allclose(result["member_affine_shapes"][0], s_ref)
        np.testing.assert_allclose(result["member_positions"][0], [48, 48])

        # The member is a pure translation of the reference: kept, so its
        # ABSOLUTE shape is S = W @ S_ref with W ~ I, and its position is the
        # refined absolute one (the reference's, moved by the true shift).
        assert result["member_status"][1] == STATUS_KEPT
        assert result["member_zncc"][1] > 0.95
        shape = result["member_affine_shapes"][1]
        np.testing.assert_allclose(shape, s_ref, atol=0.06)
        ref = np.array([48.0, 48.0])
        np.testing.assert_allclose(
            result["member_positions"][1], ref + np.array([2.0, 1.0]), atol=0.15
        )

        # ... and the relative warp recovers as S @ S_ref^-1.
        w = shape @ np.linalg.inv(result["member_affine_shapes"][0])
        np.testing.assert_allclose(w, np.eye(2), atol=0.02)

    def test_piecewise_returns_the_cells(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        plain = refine_cluster_patches(images, pos, aff, starts, m_img, m_feat)
        result = refine_cluster_patches(
            images, pos, aff, starts, m_img, m_feat, piecewise=True
        )
        shift = result["member_cell_shift_px"]
        zncc = result["member_cell_zncc"]
        status = result["member_cell_status"]
        iterations = result["member_cell_iterations"]
        assert shift.dtype == np.float32 and shift.shape == (2, 3, 3, 2)
        assert zncc.dtype == np.float32 and zncc.shape == (2, 3, 3)
        assert status.dtype == np.uint8 and status.shape == (2, 3, 3)
        assert iterations.dtype == np.uint8 and iterations.shape == (2,)
        assert set(np.unique(status).tolist()) <= set(range(6))
        stop = result["member_cell_loop_stop"]
        accepted = result["member_cell_update_accepted"]
        assert stop.dtype == np.uint8 and stop.shape == (2,)
        assert accepted.dtype == np.bool_ and accepted.shape == (2,)
        # The settings the run used: the Rust defaults.
        assert result["piecewise_options"] == {
            "cell_shift_bound_px": 2.0,
            "min_cell_zncc": 0.8,
            "min_cell_curvature": 0.02,
            "update_tolerance_px": 0.05,
            "max_iterations": 5,
        }

        # The reference is not kept: no readings, and its loop did not run.
        assert result["member_status"][0] == STATUS_REFERENCE
        assert (status[0] == CELL_NOT_ATTEMPTED).all()
        assert np.isnan(shift[0]).all() and np.isnan(zncc[0]).all()
        assert iterations[0] == 0
        assert stop[0] == LOOP_NOT_RUN and not accepted[0]

        # The kept member is a pure translation: the stage runs on it, its
        # fitted cells sit where its affine shape places them, and it stays
        # kept with the shape the cascade found, to within the fit.
        assert result["member_status"][1] == STATUS_KEPT
        assert iterations[1] >= 1
        assert stop[1] != LOOP_NOT_RUN
        fitted = status[1] == CELL_FITTED
        assert fitted.sum() >= 5
        np.testing.assert_allclose(shift[1][fitted], 0.0, atol=0.2)
        assert (zncc[1][fitted] > 0.8).all()
        np.testing.assert_allclose(
            result["member_affine_shapes"][1],
            plain["member_affine_shapes"][1],
            atol=0.05,
        )

    def test_piecewise_settings_are_passed_through(self):
        images, pos, aff, starts, m_img, m_feat = _inputs()
        result = refine_cluster_patches(
            images,
            pos,
            aff,
            starts,
            m_img,
            m_feat,
            piecewise=True,
            cell_shift_bound_px=3.0,
            min_cell_zncc=0.9,
            min_cell_curvature=0.05,
            update_tolerance_px=0.1,
            max_iterations=1,
        )
        assert result["piecewise_options"] == {
            "cell_shift_bound_px": 3.0,
            "min_cell_zncc": 0.9,
            "min_cell_curvature": 0.05,
            "update_tolerance_px": 0.1,
            "max_iterations": 1,
        }
        # A cap of one render stops every member's loop after one pass.
        assert result["member_cell_iterations"].max() <= 1

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
        # The member's image is flat: its own patch matches itself at every
        # shift, so it reads the largest ZNCC self-similarity radius, the
        # member gate excludes it before refinement and the 2-member cluster
        # becomes unrefinable.
        images, pos, aff, starts, m_img, m_feat = _inputs()
        images[1] = np.full_like(images[1], 127)
        result = refine_cluster_patches(images, pos, aff, starts, m_img, m_feat)
        assert result["member_status"][1] == STATUS_REJECTED_UNLOCALIZABLE
        assert np.isnan(result["member_zncc"][1])
        assert result["reference_members"][0] == 0xFFFFFFFF

        # Disabling the gate re-admits the member; the flat patch then fails
        # the downstream ZNCC vet instead.
        result = refine_cluster_patches(
            images,
            pos,
            aff,
            starts,
            m_img,
            m_feat,
            max_member_zncc_self_similarity_radius=0.0,
        )
        assert result["member_status"][1] == STATUS_REJECTED_LOW_ZNCC

    def test_progress_counter_ticks(self):
        from sfmtool import ProgressCounter

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
