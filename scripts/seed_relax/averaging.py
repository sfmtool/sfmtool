# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera centres from pairwise baselines.

Provenance: the study's `relaxlib.centres_by_averaging` (316-374).  The growth
route beside it in the same module is not carried: it stalls wherever the graph
has no frontier, and the averaging route places every frame the graph connects.

What the study carried was the direction half of the objective and a linear
solve against it.  The form the objective builds sends the TRUE centres to zero
whenever the directions agree, so the constellation is the form's own null
space; the length half and the reading of that null space are what this module
adds.
"""

from __future__ import annotations

import numpy as np

#: Rounds of the reweighting.
IRLS_ROUNDS = 5


def _assemble(n, ii, jj, d, w, a, lengths):
    """``(form, gauge gradient)`` of the objective at these weights.

    The block for an edge is ``w P + a d d^T``: the part of the baseline
    perpendicular to its direction, weighted by how much the direction is
    trusted, plus the part along it, weighted by how much the length is.  The
    length statements are RELATIVE, so the scale that turns them into distances
    is an unknown of the fit and is eliminated here, which leaves the form
    homogeneous and the overall scale still a gauge."""
    big = np.zeros((3 * n, 3 * n))
    vec = np.zeros(3 * n)
    row = np.zeros(3 * n)
    proj = np.eye(3)[None, :, :] - d[:, :, None] * d[:, None, :]
    for e in range(len(ii)):
        blk = w[e] * proj[e] + a[e] * np.outer(d[e], d[e])
        si = slice(3 * ii[e], 3 * ii[e] + 3)
        sj = slice(3 * jj[e], 3 * jj[e] + 3)
        big[si, si] += blk
        big[sj, sj] += blk
        big[si, sj] -= blk
        big[sj, si] -= blk
        vec[si] -= w[e] * d[e]
        vec[sj] += w[e] * d[e]
        row[si] -= a[e] * lengths[e] * d[e]
        row[sj] += a[e] * lengths[e] * d[e]
    quad = float(a @ (lengths * lengths))
    if quad > 0.0:
        big -= np.outer(row, row) / quad
    # shift gauge: lift the three translation directions out of the spectrum
    gauge = np.zeros((3, 3 * n))
    for k in range(3):
        gauge[k, k::3] = 1.0
    big = big + gauge.T @ gauge * float(np.trace(big) / max(1, 3 * n))
    return big, vec


def loose_frames(evec, null, n_frames):
    """How much of the null space each frame's own motion accounts for.

    A frame whose edges do not between them fix where it sits -- one edge and
    no length for it is enough -- moves freely at no cost to the objective, so
    it owns a null direction of its own.  The projector onto the null space
    says so without reference to a basis: the trace of a frame's own three-by-
    three block is how many null dimensions are that frame moving, and a frame
    that owns more than half of one is not part of a constellation."""
    if not null.any():
        return np.zeros(n_frames, bool), np.zeros(n_frames)
    basis = evec[:, null].reshape(n_frames, 3, -1)
    own = np.einsum("kdc,kdc->k", basis, basis)
    return own > 0.5, own


def spectrum(big, vec, n_frames):
    """The form's own reading of what it determines, and the centres it states.

    The constellation is what the form sends to zero, so the solve is
    ``argmin x' B x`` under the scale gauge ``vec . x = 1`` rather than a
    linear system posed against ``B``.  Where the form is positive definite
    that is ``B^-1 vec``, which is what a linear solve returns.  Where it is
    singular the answer is its null space, and the linear solve returns the one
    part of the answer the measurement does not carry.

    A null space of more than one dimension is the graph stating that it does
    not determine the constellation: a straight camera path leaves one null
    direction per free spacing, because the part of every baseline the
    directions measure is empty for any spacing along the line.  ``n_free``
    counts those, at the numerical rank tolerance and nothing set here.

    The null space is only the constellation when the WHOLE of it is shared.
    One dimension no frame owns half of is a constellation; anything else is a
    frame free to slide, whose own freedom sends the objective to zero as far
    away as it likes, and reading centres off it would put that frame at no
    distance in particular.  There the answer is the range solution, which
    leaves the frame where the measurement leaves it, and the counts are
    reported so the caller knows which it got."""
    lam, evec = np.linalg.eigh(big)
    proj = evec.T @ vec
    lam_max = float(lam[-1]) if len(lam) else 0.0
    tol = float(np.finfo(float).eps * len(lam) * max(lam_max, 0.0))
    null = lam <= tol
    n_null = int(null.sum())
    slack, _own = loose_frames(evec, null, n_frames)
    n_loose = int(slack.sum())
    readable = n_null == 1 and not n_loose
    if readable and float(np.abs(proj[null]).max()) > 0.0:
        x = evec[:, null] @ proj[null]
    else:
        keep = ~null
        x = evec[:, keep] @ (proj[keep] / lam[keep])
    read = {
        "lam_max": lam_max,
        "lam1_rel": float(lam[0] / lam_max) if lam_max > 0 else float("nan"),
        "lam2_rel": (
            float(lam[1] / lam_max) if lam_max > 0 and len(lam) > 1 else float("nan")
        ),
        "gap": (
            float(lam[0] / lam[1]) if len(lam) > 1 and lam[1] > 0 else float("nan")
        ),
        "n_null": n_null,
        "n_loose": n_loose,
        "n_free": max(0, n_null - 1),
        "read_off_null": bool(readable),
    }
    return x, read


def direction_reading(frames, dirs, weights):
    """What the DIRECTIONS alone determine, before any length is read in.

    The same form the averaging builds, at the weights the edges came with and
    with the length half empty, read once.  It says what the graph's geometry
    determines on its own, which is a property of the capture rather than of a
    solve, and it is what tells a colinear path from a general one."""
    index = {f: k for k, f in enumerate(frames)}
    n = len(frames)
    keys = sorted(dirs)
    w = np.array([weights[k] for k in keys], float)
    d = np.stack([dirs[k] for k in keys])
    ii = np.array([index[k[0]] for k in keys])
    jj = np.array([index[k[1]] for k in keys])
    zero = np.zeros(len(keys))
    big, vec = _assemble(n, ii, jj, d, w, zero, zero)
    _x, read = spectrum(big, vec, n)
    return read


def centres_by_averaging(
    frames,
    dirs,
    weights,
    lengths=None,
    length_weights=None,
    irls_rounds=IRLS_ROUNDS,
):
    """Camera centres from pairwise baselines, by weighted linear averaging.

    Minimizes ``sum_ij w_ij || P_ij (c_j - c_i) ||^2 + a_ij (d_ij . (c_j - c_i)
    - s L_ij)^2`` under the scale gauge ``sum_ij w_ij d_ij . (c_j - c_i) =
    sum_ij w_ij`` and the shift gauge ``sum_j c_j = 0``, with the scale ``s``
    that turns the relative lengths into distances eliminated.  Both gauges are
    exactly the freedoms the pairwise readings cannot see, so fixing them adds
    nothing.

    IRLS rounds reweight each half by the inverse of that edge's own residual
    in that half, so an edge whose direction or whose length the graph
    contradicts stops carrying the solve; the floor of each reweighting is the
    median residual of its own half, which is the graph's own scale.

    ``frames`` is the ordered frame list, ``dirs`` a ``{(i, j): unit d}`` in
    that frame indexing, ``weights`` a ``{(i, j): w}``.  ``lengths`` is an
    optional ``{(i, j): L}`` of relative baseline lengths on one common scale
    and ``length_weights`` how far each is trusted; an edge missing from either
    states no length and constrains only the direction.  Returns ``(centres
    (n, 3), per-edge lambda, per-edge residual, reading)``."""
    index = {f: k for k, f in enumerate(frames)}
    n = len(frames)
    keys = sorted(dirs)
    base_w = np.array([weights[k] for k in keys], float)
    d = np.stack([dirs[k] for k in keys])
    ii = np.array([index[k[0]] for k in keys])
    jj = np.array([index[k[1]] for k in keys])
    ell = np.array([float((lengths or {}).get(k, np.nan)) for k in keys])
    base_a = np.array([float((length_weights or {}).get(k, 0.0)) for k in keys])
    stated = np.isfinite(ell) & (base_a > 0.0)
    ell = np.where(stated, ell, 0.0)
    base_a = np.where(stated, base_a, 0.0)
    w, a = base_w.copy(), base_a.copy()
    lam = res = None
    cen = np.zeros((n, 3))
    read = {}
    for _r in range(irls_rounds):
        big, vec = _assemble(n, ii, jj, d, w, a, ell)
        x, read = spectrum(big, vec, n)
        along = float(vec @ x)
        if not np.isfinite(along) or along == 0.0:
            return None, None, None, read
        cen = (float(base_w.sum()) / along * x).reshape(n, 3)
        cen -= cen.mean(axis=0, keepdims=True)
        b = cen[jj] - cen[ii]
        lam = np.einsum("ij,ij->i", b, d)
        res = np.linalg.norm(b - lam[:, None] * d, axis=1)
        w = base_w / (1.0 + res / _floor(res))
        if stated.any():
            quad = float(base_a @ (ell * ell))
            scale = float(base_a @ (ell * lam)) / quad if quad > 0 else 0.0
            slip = np.abs(lam - scale * ell) * stated
            a = base_a / (1.0 + slip / _floor(slip[stated]))
    read["n_lengths"] = int(stated.sum())
    return cen, dict(zip(keys, lam)), dict(zip(keys, res)), read


def _floor(res):
    """The graph's own scale for a reweighting: the median of the residuals."""
    med = float(np.median(res)) if len(res) else 1.0
    return med if med > 0 else 1.0
