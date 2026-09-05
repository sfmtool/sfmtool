# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-C-contiguous numpy inputs must not be reinterpreted as column-major.

``PyReadonlyArray::as_slice()`` succeeds for Fortran-contiguous arrays as well
as C-contiguous ones, and hands back the raw buffer. Any binding that then
indexes that buffer positionally (``s[i * cols + j]``) or passes it to a kernel
as a flat row-major block silently transposes a 2-D or 3-D input — no error,
just wrong numbers. A ``map_err(… "must be C-contiguous")`` does not catch it,
because F-contiguous arrays pass that check too.

There is a second spelling of the same bug: ``as_array().to_owned()`` preserves
strides whenever the source is contiguous in memory order (which F-contiguous
input is), producing a non-standard-layout ``Array``. The format writers then
call ``.as_slice().unwrap()`` on it, so that one surfaces as a
``PanicException`` from inside ``sfmr-format`` rather than as silently wrong
numbers. The fix there is ``.as_array().as_standard_layout().into_owned()``.

The crate-wide ``to_contiguous!`` macro guards the first spelling with
``is_standard_layout()``. This module pins the property at the Python boundary
for the binding families that were carrying either bug, so a future hand-rolled
``as_slice()`` or stride-preserving ``to_owned()`` gets caught here rather than
in someone's reconstruction.

Coverage is not exhaustive over the whole PyO3 surface — it covers the
subsystems that were actually broken, plus the round-trip paths that would
panic. 1-D inputs are deliberately not covered: a 1-D array is always both C-
and F-contiguous, so neither hazard can arise.
"""

import numpy as np
import pytest


def _both_orders(a):
    """The same logical array as a C-contiguous and an F-contiguous copy."""
    c = np.ascontiguousarray(a)
    f = np.asfortranarray(a)
    assert np.array_equal(c, f)
    if a.ndim > 1 and min(a.shape[:2]) > 1:
        assert not f.flags["C_CONTIGUOUS"], "test array is degenerate"
    return c, f


class TestSpatialIndexes:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_kdtree2d_positions(self, dtype):
        from sfmtool._sfmtool.spatial import KdTree2d

        rng = np.random.default_rng(0)
        pos, q = rng.random((16, 2)).astype(dtype), rng.random((5, 2)).astype(dtype)
        c, f = _both_orders(pos)
        np.testing.assert_array_equal(KdTree2d(f).nearest(q), KdTree2d(c).nearest(q))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_kdtree3d_positions(self, dtype):
        from sfmtool._sfmtool.spatial import KdTree3d

        rng = np.random.default_rng(1)
        pos, q = rng.random((16, 3)).astype(dtype), rng.random((5, 3)).astype(dtype)
        c, f = _both_orders(pos)
        np.testing.assert_array_equal(KdTree3d(f).nearest(q), KdTree3d(c).nearest(q))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_kdtree2d_queries(self, dtype):
        """The query side too, not just the stored cloud."""
        from sfmtool._sfmtool.spatial import KdTree2d

        rng = np.random.default_rng(8)
        pos = rng.random((16, 2)).astype(dtype)
        queries = rng.random((5, 2)).astype(dtype)
        qc, qf = _both_orders(queries)
        tree = KdTree2d(np.ascontiguousarray(pos))

        np.testing.assert_array_equal(tree.nearest(qf), tree.nearest(qc))
        np.testing.assert_array_equal(tree.nearest_k(qf, k=3), tree.nearest_k(qc, k=3))

    def test_kdforest_descriptors_and_queries(self):
        """Regression: an F-ordered descriptor block silently destroyed matching."""
        from sfmtool._sfmtool.spatial import KdForest

        rng = np.random.default_rng(2)
        desc = rng.integers(0, 256, (64, 8), np.uint8)
        query = rng.integers(0, 256, (5, 8), np.uint8)
        dc, df = _both_orders(desc)
        qc, qf = _both_orders(query)

        expected = KdForest(dc, seed=1).query(qc, k=2)[0]
        np.testing.assert_array_equal(KdForest(df, seed=1).query(qc, k=2)[0], expected)
        np.testing.assert_array_equal(KdForest(dc, seed=1).query(qf, k=2)[0], expected)


class TestOpticalFlow:
    def test_compute_optical_flow_images(self):
        from sfmtool._sfmtool.flow import compute_optical_flow

        rng = np.random.default_rng(3)
        a = rng.integers(0, 256, (48, 64), np.uint8)
        b = rng.integers(0, 256, (48, 64), np.uint8)
        ac, af = _both_orders(a)
        bc, bf = _both_orders(b)

        u_exp, v_exp = compute_optical_flow(ac, bc)
        u_got, v_got = compute_optical_flow(af, bf)
        np.testing.assert_array_equal(u_got, u_exp)
        np.testing.assert_array_equal(v_got, v_exp)

    def test_remap_bilinear_image(self):
        from sfmtool._sfmtool.flow import WarpMap

        rng = np.random.default_rng(4)
        h, w = 24, 32
        wm = WarpMap.from_numpy(
            np.tile(np.arange(w, dtype=np.float32), (h, 1)),
            np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)),
        )
        for shape in [(h, w, 3), (h, w)]:
            img = rng.integers(0, 256, shape, np.uint8)
            c, f = _both_orders(img)
            np.testing.assert_array_equal(wm.remap_bilinear(f), wm.remap_bilinear(c))


class TestSiftExtraction:
    @pytest.mark.parametrize("shape", [(96, 128), (96, 128, 3)])
    def test_extract_sift_image(self, shape):
        """Regression: an F-ordered image yielded a different keypoint count."""
        from sfmtool._sfmtool.sift import extract_sift

        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, shape, np.uint8)
        c, f = _both_orders(img)

        # (positions, affine_shapes, descriptors) — compare all three; the
        # descriptor block is the one whose corruption actually broke matching.
        pos_c, aff_c, desc_c = extract_sift(c)
        pos_f, aff_f, desc_f = extract_sift(f)
        assert len(pos_c) > 0, "test image produced no keypoints"
        np.testing.assert_array_equal(pos_f, pos_c)
        np.testing.assert_array_equal(aff_f, aff_c)
        np.testing.assert_array_equal(desc_f, desc_c)


class TestSe3Transform:
    def test_apply_to_points_and_matmul(self):
        from sfmtool._sfmtool.geometry import RotQuaternion, Se3Transform

        rng = np.random.default_rng(6)
        t = Se3Transform(RotQuaternion(0.5, 0.5, 0.5, 0.5), [1.0, 2.0, 3.0], 1.3)
        pts = rng.random((7, 3))
        c, f = _both_orders(pts)

        np.testing.assert_array_equal(t.apply_to_points(f), t.apply_to_points(c))
        np.testing.assert_array_equal(np.asarray(t @ f), np.asarray(t @ c))


class TestReconstructionCloneRoundTrip:
    """``clone_with_changes`` + ``save`` with F-ordered array kwargs.

    Regression: these went through ``as_array().to_owned()``, which preserved
    the Fortran strides, and ``sfmr-format``'s writer then did
    ``.as_slice().unwrap()`` on the non-standard-layout array — a
    ``PanicException`` surfacing from a crate the caller never touched.
    """

    def test_fortran_thumbnails_round_trip(self, seoul_bull_workspace, tmp_path):
        from sfmtool._sfmtool.reconstruction import SfmrReconstruction

        recon = SfmrReconstruction.load(seoul_bull_workspace)
        thumbs = np.asarray(recon.thumbnails_y_x_rgb)
        c, f = _both_orders(thumbs)

        out_c = tmp_path / "c.sfmr"
        out_f = tmp_path / "f.sfmr"
        recon.clone_with_changes(thumbnails_y_x_rgb=c).save(out_c, operation="t")
        recon.clone_with_changes(thumbnails_y_x_rgb=f).save(out_f, operation="t")

        np.testing.assert_array_equal(
            np.asarray(SfmrReconstruction.load(out_f).thumbnails_y_x_rgb),
            np.asarray(SfmrReconstruction.load(out_c).thumbnails_y_x_rgb),
        )

    def test_fortran_keypoints_round_trip(self, seoul_bull_workspace, tmp_path):
        from sfmtool._sfmtool.reconstruction import SfmrReconstruction

        # `seoul_bull_workspace` is always feature_source="sift_files", whose
        # `keypoints_xy` is None — embedding patches is what materializes it.
        recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
            normal="mean_viewing", extent_value=5.0
        )
        assert recon.keypoints_xy is not None, "fixture no longer carries keypoints"
        c, f = _both_orders(np.asarray(recon.keypoints_xy))

        out = tmp_path / "kp.sfmr"
        recon.clone_with_changes(keypoints_xy=f).save(out, operation="t")
        np.testing.assert_array_equal(
            np.asarray(SfmrReconstruction.load(out).keypoints_xy), c
        )

    def test_fortran_2d_kwargs_round_trip(self, seoul_bull_workspace, tmp_path):
        """The five `to_contiguous!`-converted 2-D kwargs, which had no test.

        The C- and F-ordered clones travel the same clone + save + load path and
        must agree bit for bit: layout invariance is what this module tests.
        Pose-bit preservation through that path is a separate property, owned
        by `unit_quaternion_preserving`'s Rust tests, so the check against the
        source arrays is loose (1e-15) for the float fields; it still catches a
        bug that transposed both layouts alike.
        """
        from sfmtool._sfmtool.reconstruction import SfmrReconstruction

        recon = SfmrReconstruction.load(seoul_bull_workspace)
        fields = {
            "positions": np.asarray(recon.positions),
            "colors": np.asarray(recon.colors),
            "quaternions_wxyz": np.asarray(recon.quaternions_wxyz),
            "translations": np.asarray(recon.translations),
        }
        f_kwargs = {k: np.asfortranarray(v) for k, v in fields.items()}
        for k, v in f_kwargs.items():
            if min(v.shape[:2]) > 1:
                assert not v.flags["C_CONTIGUOUS"], f"{k} is degenerate"

        out_c = tmp_path / "kw_c.sfmr"
        out_f = tmp_path / "kw_f.sfmr"
        recon.clone_with_changes(
            **{k: np.ascontiguousarray(v) for k, v in fields.items()}
        ).save(out_c, operation="t")
        recon.clone_with_changes(**f_kwargs).save(out_f, operation="t")
        c_back = SfmrReconstruction.load(out_c)
        f_back = SfmrReconstruction.load(out_f)

        for name, source in fields.items():
            from_c = np.asarray(getattr(c_back, name))
            np.testing.assert_array_equal(
                np.asarray(getattr(f_back, name)), from_c, err_msg=name
            )
            if np.issubdtype(source.dtype, np.floating):
                np.testing.assert_allclose(
                    from_c, source, rtol=0, atol=1e-15, err_msg=name
                )
            else:
                np.testing.assert_array_equal(from_c, source, err_msg=name)


class TestNegativeStride:
    """Reversed views: the case the "1-D is exempt" rule got wrong.

    ndarray's `Dimension::is_contiguous` treats a negative stride as contiguous
    (`dim[0] <= 1 || strides[0] == -1`, and the N-D branch compares
    `unsigned_abs()`). So `a[::-1]` survives `to_owned()` with its `-1` stride
    intact and is *not* standard-layout — reaching the format writer's
    `as_slice().unwrap()` with no Fortran order involved at any point, and at
    1-D as readily as at 2-D.
    """

    def test_reversed_1d_field_round_trips(self, seoul_bull_workspace, tmp_path):
        from sfmtool._sfmtool.reconstruction import SfmrReconstruction

        recon = SfmrReconstruction.load(seoul_bull_workspace)
        errors = np.asarray(recon.errors)
        reversed_view = errors[::-1]
        assert not reversed_view.flags["C_CONTIGUOUS"]

        out = tmp_path / "rev1d.sfmr"
        recon.clone_with_changes(errors=np.ascontiguousarray(reversed_view)).save(
            out, operation="t"
        )
        np.testing.assert_array_equal(
            np.asarray(SfmrReconstruction.load(out).errors), reversed_view
        )

    def test_reversed_2d_thumbnails_round_trip(self, seoul_bull_workspace, tmp_path):
        from sfmtool._sfmtool.reconstruction import SfmrReconstruction

        recon = SfmrReconstruction.load(seoul_bull_workspace)
        thumbs = np.asarray(recon.thumbnails_y_x_rgb)
        reversed_view = thumbs[::-1]
        assert not reversed_view.flags["C_CONTIGUOUS"]

        out = tmp_path / "rev2d.sfmr"
        recon.clone_with_changes(thumbnails_y_x_rgb=reversed_view).save(
            out, operation="t"
        )
        np.testing.assert_array_equal(
            np.asarray(SfmrReconstruction.load(out).thumbnails_y_x_rgb), reversed_view
        )


class TestDescriptorMatching:
    def test_match_candidates_by_descriptor(self):
        """`matching/descriptor.rs` — 3 converted sites, previously untested."""
        from sfmtool._sfmtool.matching import match_candidates_by_descriptor

        rng = np.random.default_rng(9)
        n, k, m = 12, 4, 20
        candidates = rng.integers(0, m, (n, k)).astype(np.uint32)
        in_bounds = np.arange(n, dtype=np.uint32)
        d1 = rng.integers(0, 256, (n, 128)).astype(np.uint8)
        d2 = rng.integers(0, 256, (m, 128)).astype(np.uint8)

        cc, cf = _both_orders(candidates)
        d1c, d1f = _both_orders(d1)
        d2c, d2f = _both_orders(d2)

        expected = match_candidates_by_descriptor(cc, in_bounds, d1c, d2c, 1e9)
        got = match_candidates_by_descriptor(cf, in_bounds, d1f, d2f, 1e9)
        assert len(np.asarray(expected)) > 0, "no matches — test would be vacuous"
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


class TestSphericalAtlas:
    @pytest.mark.parametrize("channels", [1, 3])
    def test_resample_atlas(self, channels):
        from sfmtool._sfmtool.spherical import SphericalTileRig
        from sfmtool import resample_atlas_to_equirect

        rng = np.random.default_rng(7)
        rig = SphericalTileRig(n=80, arc_per_pixel=2 * np.pi / 128, seed=42)
        w, h = rig.atlas_size
        shape = (h, w, channels) if channels > 1 else (h, w)
        atlas = rng.random(shape).astype(np.float32)
        c, f = _both_orders(atlas)

        np.testing.assert_array_equal(
            resample_atlas_to_equirect(rig, f, 64, 32, k=1),
            resample_atlas_to_equirect(rig, c, 64, 32, k=1),
        )
