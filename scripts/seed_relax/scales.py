# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Relative baseline lengths, from the depths each pair's own solve implies.

A pair's baseline direction fixes the pair's geometry only up to how long the
baseline is, so the two-view depths that come out of it are read in units of
THAT baseline.  A cluster two edges both see from the same frame therefore has
two depths for one world distance, and their ratio is the ratio of the two
baselines: the depths carry exactly the relative scale the directions cannot.

The whole graph is one fit of ``log z(edge, frame, cluster) = D(frame, cluster)
- x(edge)``, where ``x`` is the log baseline length and ``D`` the log world
depth.  The depths are per-cluster, so there are as many ``D`` as clusters
seen; they are eliminated rather than solved for, which leaves a system in the
edges alone whose operator is a pass over the rows and never a matrix.  Rows
are reweighted by their own residual for a few rounds, so a wild depth stops
carrying the fit, and an edge that shares no cluster with any other edge is
left without a length rather than given one.
"""

from __future__ import annotations

import numpy as np

#: Rounds of the row reweighting.
IRLS_ROUNDS = 8

#: The most conjugate-gradient steps one round takes, and the relative
#: residual it stops at.  Both bound the work; neither decides an outcome,
#: because the fit is a least-squares solve with one answer.
CG_STEPS = 200
CG_TOL = 1e-12

#: The fewest tied rows an edge needs before it states a length.  It is the
#: baseline-direction operation's own floor: below three rows a reading is the
#: rows themselves rather than a fit of them.
MIN_TIED_ROWS = 3


def two_view_depths(u_i, u_j, d):
    """Depths along each ray at the closest approach, and the midpoint.

    ``c_i`` sits at the origin and ``c_j`` at ``+d``, so both depths are in
    units of the pair's own baseline."""
    a11 = np.einsum("ij,ij->i", u_i, u_i)
    a12 = -np.einsum("ij,ij->i", u_i, u_j)
    a22 = np.einsum("ij,ij->i", u_j, u_j)
    b1 = u_i @ d
    b2 = -(u_j @ d)
    det = a11 * a22 - a12 * a12
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (a22 * b1 - a12 * b2) / det
        t = (a11 * b2 - a12 * b1) / det
    mid = 0.5 * (s[:, None] * u_i + (d + t[:, None] * u_j))
    return s, t, mid


class _Fit:
    """The eliminated system in the edges alone, at one set of row weights.

    Each cluster's world depth is at its own weighted mean over the rows that
    saw it, so it never enters the solve; what is left is the residual of every
    row against that mean, summed onto the edge it came from."""

    def __init__(self, ee, gid, n_edge, n_group, ww):
        self.ee, self.gid = ee, gid
        self.n_edge, self.n_group = n_edge, n_group
        self.ww = ww
        self.gsum = np.bincount(gid, weights=ww, minlength=n_group)
        self.gsum[self.gsum <= 0.0] = 1.0
        # The diagonal of the operator, which preconditions the solve.
        share = ww / self.gsum[gid]
        self.diag = np.bincount(ee, weights=ww * (1.0 - share), minlength=n_edge)
        self.diag[self.diag <= 0.0] = 1.0

    def centred(self, row):
        """Each row less its own cluster's weighted mean over the rows."""
        mean = np.bincount(self.gid, weights=self.ww * row, minlength=self.n_group)
        return row - (mean / self.gsum)[self.gid]

    def apply(self, x):
        """The operator: spread onto rows, centre, weight, gather back."""
        return np.bincount(
            self.ee,
            weights=self.ww * self.centred(x[self.ee]),
            minlength=self.n_edge,
        )

    def rhs(self, vv):
        """What the depths themselves put on the right-hand side."""
        return -np.bincount(
            self.ee, weights=self.ww * self.centred(vv), minlength=self.n_edge
        )

    def residual(self, x, vv):
        """Per row, what the fit leaves of ``x_edge + log z - log depth``."""
        return self.centred(x[self.ee] + vv)


def _conjugate_gradient(fit, rhs, steps=CG_STEPS, tol=CG_TOL, start=None):
    """The least-squares lengths, by preconditioned conjugate gradient.

    The operator is positive semi-definite with the constant vector in its null
    space, which is the one freedom a set of RELATIVE lengths does not have; the
    right-hand side is orthogonal to it, so the iteration never leaves the
    complement."""
    x = np.zeros(fit.n_edge) if start is None else np.asarray(start, float).copy()
    r = rhs - fit.apply(x)
    z = r / fit.diag
    p = z.copy()
    rz = float(r @ z)
    bar = tol * float(np.abs(rhs).max() or 1.0)
    for _k in range(int(steps)):
        if float(np.abs(r).max()) <= bar or rz <= 0.0:
            break
        ap = fit.apply(p)
        denom = float(p @ ap)
        if denom <= 0.0:
            break
        alpha = rz / denom
        x += alpha * p
        r -= alpha * ap
        z = r / fit.diag
        rz_next = float(r @ z)
        p = z + (rz_next / rz) * p
        rz = rz_next
    return x - float(np.median(x))


def relative_lengths(
    keys, depths, rounds=IRLS_ROUNDS, min_tied=MIN_TIED_ROWS, cg_steps=CG_STEPS
):
    """``(lengths, scatter, n_tied)`` per edge of ``keys``, in that order.

    ``depths`` maps an edge key to ``(frames, clusters, z)``: one row per
    (frame, cluster) the edge's solve gave a positive depth for, ``z`` in units
    of that edge's own baseline.  Returns the relative length of every edge
    with the ``min_tied`` rows it needs, gauged to a median of one, the median
    absolute log residual each edge's own rows leave, and how many of its rows
    another edge also saw.  An edge without a length reads back as ``nan``."""
    n_edge = len(keys)
    none = np.full(n_edge, np.nan)
    ee, gg, vv = [], [], []
    for e, k in enumerate(keys):
        row = depths.get(k)
        if row is None:
            continue
        frames, clusters, z = row
        ee.append(np.full(len(z), e, np.int64))
        gg.append(
            np.stack([np.asarray(frames, np.int64), np.asarray(clusters, np.int64)])
        )
        vv.append(np.log(np.asarray(z, float)))
    if not ee:
        return none, none.copy(), np.zeros(n_edge, np.int64)
    ee = np.concatenate(ee)
    gg = np.concatenate(gg, axis=1)
    vv = np.concatenate(vv)
    _uniq, gid = np.unique(gg, axis=1, return_inverse=True)
    gid = np.asarray(gid, np.int64).ravel()

    # A row is TIED when another edge saw the same cluster from the same frame.
    # Only tied rows relate one baseline to another; an edge whose rows are all
    # its own carries no relative statement and gets no length.
    n_tied = np.bincount(
        ee[np.bincount(gid, minlength=int(gid.max()) + 1)[gid] > 1], minlength=n_edge
    )
    has = n_tied >= int(min_tied)
    take = has[ee]
    if not take.any():
        return none, none.copy(), n_tied
    ee, gid, vv = ee[take], gid[take], vv[take]
    _u2, gid = np.unique(gid, return_inverse=True)
    gid = np.asarray(gid, np.int64).ravel()
    n_group = int(gid.max()) + 1

    ww = np.ones(len(vv))
    x = resid = None
    for _r in range(max(1, int(rounds))):
        fit = _Fit(ee, gid, n_edge, n_group, ww)
        x = _conjugate_gradient(fit, fit.rhs(vv), steps=cg_steps, start=x)
        resid = np.abs(fit.residual(x, vv))
        med = float(np.median(resid))
        ww = 1.0 / (1.0 + resid / (med if med > 0.0 else 1.0))

    scatter = none.copy()
    order = np.lexsort((resid, ee))
    counts = np.bincount(ee, minlength=n_edge)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    seen = counts > 0
    lo = starts[seen] + (counts[seen] - 1) // 2
    hi = starts[seen] + counts[seen] // 2
    scatter[seen] = 0.5 * (resid[order][lo] + resid[order][hi])
    lengths = none.copy()
    lengths[has] = np.exp(x[has])
    return lengths, scatter, n_tied
