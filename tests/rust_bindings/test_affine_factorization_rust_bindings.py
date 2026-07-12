# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the affine-factorization Rust bindings
(``sfmtool._sfmtool.geometry.factorize_affine``; see
``specs/core/affine-factorization.md``).

The parity tests compare the bindings against a numpy reference
implementation of the same contractual algorithm (mean-filled dense SVD
init, per-image/per-cluster least-squares sweeps, trimming against the
numpy-default linear-interpolation quantile). Gauge-dependent quantities
(cameras, points, gauge) can differ between implementations by an orthogonal
gauge (SVD/eigenvector sign and order are not contractual), so parity is
asserted on gauge-invariant quantities: reprojections, residuals, keep and
used masks, the eigenvalues of ``Q = gauge @ gauge.T``, scales, and relative
rotations (up to reflection-hypothesis pairing). Floating-point divergence between numpy's
LAPACK and nalgebra solvers is at machine-epsilon scale; tolerances are
tight (1e-8 absolute on ~100 px data).
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import (
    AffineFactorization,
    MetricHypothesis,
    factorize_affine,
)

# ── Numpy reference implementation (the spec's Algorithm section) ──────────


def _ref_factorize(
    obs_clusters, obs_images, obs_xy, n, c, rounds=25, trim_fraction=0.05
):
    K = len(obs_clusters)
    W = np.full((2 * n, c), np.nan)
    for o in range(K):
        W[2 * obs_images[o], obs_clusters[o]] = obs_xy[o, 0]
        W[2 * obs_images[o] + 1, obs_clusters[o]] = obs_xy[o, 1]
    obs_mask = ~np.isnan(W)
    means = np.zeros(2 * n)
    for r in range(2 * n):
        if obs_mask[r].any():
            means[r] = W[r][obs_mask[r]].mean()
    Wc = np.where(obs_mask, W - means[:, None], 0.0)
    _, _, vt = np.linalg.svd(Wc, full_matrices=False)
    X = vt[:3].T.copy()  # (c, 3)
    M = np.zeros((n, 2, 3))
    t = np.zeros((n, 2))
    keep = np.ones(K, dtype=bool)
    resid = obs_xy.copy()
    for rnd in range(rounds):
        for i in range(n):
            sel = keep & (obs_images == i)
            if np.count_nonzero(sel) < 4:
                continue
            A = np.hstack([X[obs_clusters[sel]], np.ones((np.count_nonzero(sel), 1))])
            P = np.linalg.lstsq(A, obs_xy[sel], rcond=None)[0]  # (4, 2)
            M[i] = P[:3].T
            t[i] = P[3]
        for cc in range(c):
            sel = keep & (obs_clusters == cc)
            if np.count_nonzero(sel) < 2:
                continue
            A = M[obs_images[sel]].reshape(-1, 3)
            b = (obs_xy[sel] - t[obs_images[sel]]).reshape(-1)
            X[cc] = np.linalg.lstsq(A, b, rcond=None)[0]
        pred = np.einsum("krc,kc->kr", M[obs_images], X[obs_clusters]) + t[obs_images]
        resid = obs_xy - pred
        if rnd >= rounds // 2:
            norms = np.linalg.norm(resid, axis=1)
            kept = norms[keep]
            if kept.size:
                thr = np.quantile(kept, 1 - trim_fraction)
                keep = norms < thr
    used = np.array([np.count_nonzero(keep & (obs_images == i)) >= 4 for i in range(n)])
    return M, t, X, resid, keep, used


def _ref_metric_upgrade(M, used):
    """The spec's Metric upgrade section on numpy."""
    idx = np.flatnonzero(used)
    if idx.size == 0:
        return None

    def qc(a, b):  # coefficients of a' Q b in (q11 q12 q13 q22 q23 q33)
        return np.array(
            [
                a[0] * b[0],
                a[0] * b[1] + a[1] * b[0],
                a[0] * b[2] + a[2] * b[0],
                a[1] * b[1],
                a[1] * b[2] + a[2] * b[1],
                a[2] * b[2],
            ]
        )

    A = np.zeros((2 * idx.size + 1, 6))
    b = np.zeros(2 * idx.size + 1)
    for r, i in enumerate(idx):
        m1, m2 = M[i]
        A[2 * r] = qc(m1, m1) - qc(m2, m2)
        A[2 * r + 1] = qc(m1, m2)
        A[-1] += (qc(m1, m1) + qc(m2, m2)) / idx.size
    b[-1] = 2.0
    s = np.linalg.svd(A, compute_uv=False)
    if s.size < 6 or s[-1] <= 1e-10 * s[0]:
        return None
    q = np.linalg.lstsq(A, b, rcond=None)[0]
    Q = np.array([[q[0], q[1], q[2]], [q[1], q[3], q[4]], [q[2], q[4], q[5]]])
    lam, V = np.linalg.eigh(Q)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    if lam[0] <= 0:
        return None
    lam = np.maximum(lam, 1e-8 * lam[0])
    gauge = V * np.sqrt(lam)

    hyps = []
    for reflect in (False, True):
        g = gauge.copy()
        if reflect:
            g[:, 2] = -g[:, 2]
        rots = np.tile(np.eye(3), (len(M), 1, 1))
        scales = np.zeros(len(M))
        for i in idx:
            m = M[i] @ g  # (2, 3)
            s_i = (np.linalg.norm(m[0]) + np.linalg.norm(m[1])) / 2
            scales[i] = s_i
            r3 = np.cross(m[0], m[1])
            r3 = r3 / np.linalg.norm(r3)
            stack = np.vstack([m / s_i, r3])
            u, _, vt = np.linalg.svd(stack)
            r = u @ vt
            if np.linalg.det(r) < 0:
                u[:, 2] = -u[:, 2]
                r = u @ vt
            rots[i] = r
        hyps.append((g, rots, scales))
    return hyps


# ── Synthetic fixture ───────────────────────────────────────────────────────


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _fixture(seed=3, n=6, c=50, drop=0.25, noise=0.05, num_outliers=5):
    """Pixel-scale metric scene with missing data, mild noise, and a few
    gross outliers (one per image at most, distinct offsets)."""
    rng = np.random.default_rng(seed)
    rotations = [
        _axis_angle(rng.uniform(-1, 1, 3), rng.uniform(0.05, 0.8)) for _ in range(n)
    ]
    scales = rng.uniform(0.8, 1.3, n)
    translations = rng.uniform(-50, 50, (n, 2))
    points = rng.uniform(-80, 80, (c, 3))

    obs_clusters, obs_images, obs_xy = [], [], []
    for i in range(n):
        for cc in range(c):
            # Guarantee the first two images fully observed so every cluster
            # keeps >= 2 observations.
            if i >= 2 and rng.random() < drop:
                continue
            u = scales[i] * (rotations[i][:2] @ points[cc]) + translations[i]
            obs_clusters.append(cc)
            obs_images.append(i)
            obs_xy.append(u)
    obs_clusters = np.array(obs_clusters, dtype=np.uint32)
    obs_images = np.array(obs_images, dtype=np.uint32)
    obs_xy = np.array(obs_xy) + rng.normal(0.0, noise, (len(obs_clusters), 2))

    outliers = []
    for i in range(num_outliers):
        o = int(np.flatnonzero(obs_images == i)[3 + 2 * i])
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(25, 60)
        obs_xy[o] += [mag * np.cos(ang), mag * np.sin(ang)]
        outliers.append(o)
    return obs_clusters, obs_images, obs_xy, n, c, rotations, outliers


# ── Getter round-trips ──────────────────────────────────────────────────────


class TestGetters:
    def test_shapes_and_dtypes(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        fac = factorize_affine(oc, oi, oxy, n, c)
        assert isinstance(fac, AffineFactorization)
        k = len(oc)
        assert fac.cameras.shape == (n, 2, 3) and fac.cameras.dtype == np.float64
        assert fac.translations.shape == (n, 2) and fac.translations.dtype == np.float64
        assert fac.points.shape == (c, 3) and fac.points.dtype == np.float64
        assert fac.residuals.shape == (k, 2) and fac.residuals.dtype == np.float64
        assert fac.keep.shape == (k,) and fac.keep.dtype == np.bool_
        assert fac.used_images.shape == (n,) and fac.used_images.dtype == np.bool_

    def test_metric_hypothesis_shapes(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        fac = factorize_affine(oc, oi, oxy, n, c)
        hyps = fac.metric_upgrade()
        assert isinstance(hyps, tuple) and len(hyps) == 2
        for hyp in hyps:
            assert isinstance(hyp, MetricHypothesis)
            assert hyp.gauge.shape == (3, 3) and hyp.gauge.dtype == np.float64
            assert hyp.rotations.shape == (n, 3, 3)
            assert hyp.scales.shape == (n,) and hyp.scales.dtype == np.float64

    def test_keyword_defaults_match_explicit(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        fac_default = factorize_affine(oc, oi, oxy, n, c)
        fac_explicit = factorize_affine(
            oc, oi, oxy, n, c, rounds=25, trim_fraction=0.05
        )
        npt.assert_array_equal(fac_default.cameras, fac_explicit.cameras)
        npt.assert_array_equal(fac_default.points, fac_explicit.points)
        npt.assert_array_equal(fac_default.keep, fac_explicit.keep)

    def test_deterministic_across_calls(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        a = factorize_affine(oc, oi, oxy, n, c)
        b = factorize_affine(oc, oi, oxy, n, c)
        npt.assert_array_equal(a.cameras, b.cameras)
        npt.assert_array_equal(a.translations, b.translations)
        npt.assert_array_equal(a.points, b.points)
        npt.assert_array_equal(a.residuals, b.residuals)
        npt.assert_array_equal(a.keep, b.keep)
        npt.assert_array_equal(a.used_images, b.used_images)


# ── Full-pipeline parity with the numpy reference ───────────────────────────


class TestReferenceParity:
    def test_factorization_parity(self):
        oc, oi, oxy, n, c, _, outliers = _fixture()
        fac = factorize_affine(oc, oi, oxy, n, c)
        M, t, X, resid, keep, used = _ref_factorize(oc, oi, oxy, n, c)

        # Discrete outputs match exactly.
        npt.assert_array_equal(fac.keep, keep)
        npt.assert_array_equal(fac.used_images, used)
        assert not fac.keep[outliers].any()

        # Continuous gauge-invariant outputs match tightly.
        npt.assert_allclose(fac.residuals, resid, atol=1e-8)
        pred_rust = (
            np.einsum("krc,kc->kr", fac.cameras[oi], fac.points[oc])
            + fac.translations[oi]
        )
        pred_ref = np.einsum("krc,kc->kr", M[oi], X[oc]) + t[oi]
        npt.assert_allclose(pred_rust, pred_ref, atol=1e-8)

        # Cameras/points agree after solving the 3x3 gauge between the two
        # affine frames (they are individually gauge-dependent); translations
        # are gauge-free and match directly.
        g = np.linalg.lstsq(fac.points, X, rcond=None)[0]
        npt.assert_allclose(fac.points @ g, X, atol=1e-6)
        for i in range(n):
            npt.assert_allclose(fac.cameras[i] @ np.linalg.inv(g).T, M[i], atol=1e-6)
        npt.assert_allclose(fac.translations, t, atol=1e-6)

    def test_metric_upgrade_parity(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        fac = factorize_affine(oc, oi, oxy, n, c)
        hyps_rust = fac.metric_upgrade()
        M, _, _, _, _, used = _ref_factorize(oc, oi, oxy, n, c)
        hyps_ref = _ref_metric_upgrade(M, used)
        assert hyps_rust is not None and hyps_ref is not None

        # Q = A A' is shared by both reflection hypotheses within one
        # implementation. Across implementations the affine camera frames
        # differ by an orthogonal gauge D (SVD sign ambiguity), conjugating
        # Q -> D Q D'; its eigenvalues are the invariant to compare.
        q_rust = hyps_rust[0].gauge @ hyps_rust[0].gauge.T
        for hyp in hyps_rust:
            npt.assert_allclose(hyp.gauge @ hyp.gauge.T, q_rust, atol=1e-10)
        q_ref = hyps_ref[0][0] @ hyps_ref[0][0].T
        npt.assert_allclose(
            np.linalg.eigvalsh(q_rust), np.linalg.eigvalsh(q_ref), rtol=1e-6
        )

        # Scales are invariant to the orthogonal eigenvector ambiguity.
        for hyp, (_, _, scales_ref) in zip(hyps_rust, hyps_ref):
            npt.assert_allclose(hyp.scales, scales_ref, atol=1e-8)

        # Rotations: within one reflection class, relative rotations
        # R_i R_first' are exactly gauge-invariant; the class pairing between
        # implementations can swap (eigenvector signs are not contractual).
        first = int(np.flatnonzero(used)[0])
        idx = np.flatnonzero(used)

        def rels(rots):
            return np.array([rots[i] @ rots[first].T for i in idx])

        rust_rels = [rels(h.rotations) for h in hyps_rust]
        ref_rels = [rels(r) for (_, r, _) in hyps_ref]
        direct = max(
            np.abs(rust_rels[0] - ref_rels[0]).max(),
            np.abs(rust_rels[1] - ref_rels[1]).max(),
        )
        swapped = max(
            np.abs(rust_rels[0] - ref_rels[1]).max(),
            np.abs(rust_rels[1] - ref_rels[0]).max(),
        )
        assert min(direct, swapped) < 1e-6, (direct, swapped)

    def test_ground_truth_rotations_recovered(self):
        oc, oi, oxy, n, c, rotations, _ = _fixture()
        fac = factorize_affine(oc, oi, oxy, n, c)
        hyps = fac.metric_upgrade()
        used = fac.used_images
        first = int(np.flatnonzero(used)[0])

        def matches(hyp):
            d = rotations[first].T @ hyp.rotations[first]
            return all(
                np.abs(rotations[i] @ d - hyp.rotations[i]).max() < 1e-3
                for i in np.flatnonzero(used)
            )

        assert matches(hyps[0]) or matches(hyps[1])


# ── Validation and edge cases ───────────────────────────────────────────────


class TestValidation:
    def test_wrong_index_dtype_raises(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        with pytest.raises(TypeError, match="uint32"):
            factorize_affine(oc.astype(np.int64), oi, oxy, n, c)
        with pytest.raises(TypeError, match="uint32"):
            factorize_affine(oc, oi.astype(np.int64), oxy, n, c)

    def test_wrong_xy_shape_raises(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        with pytest.raises(ValueError, match=r"\(K, 2\)"):
            factorize_affine(oc, oi, np.zeros((len(oc), 3)), n, c)

    def test_non_parallel_raises(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        with pytest.raises(ValueError, match="parallel"):
            factorize_affine(oc[:-1], oi, oxy, n, c)

    def test_out_of_range_raises(self):
        oc, oi, oxy, n, c, _, _ = _fixture()
        with pytest.raises(ValueError, match="out of range"):
            factorize_affine(oc, oi, oxy, n, c - 1)
        with pytest.raises(ValueError, match="out of range"):
            factorize_affine(oc, oi, oxy, n - 1, c)

    def test_dense_bound_raises(self):
        empty_u32 = np.array([], dtype=np.uint32)
        with pytest.raises(ValueError, match="dense factorization bound"):
            factorize_affine(empty_u32, empty_u32, np.zeros((0, 2)), 3000, 2000)

    def test_metric_upgrade_none_when_nothing_used(self):
        # A single 3-observation image never reaches 4 kept observations.
        oc = np.array([0, 1, 2], dtype=np.uint32)
        oi = np.array([0, 0, 0], dtype=np.uint32)
        oxy = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        fac = factorize_affine(oc, oi, oxy, 1, 3)
        assert not fac.used_images.any()
        assert fac.metric_upgrade() is None
