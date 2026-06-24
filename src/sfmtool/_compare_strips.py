# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the points two reconstructions place most differently as side-by-side
patch strips.

This backs ``sfm compare --strips``. Two reconstructions of the same images are
put in 3D-point correspondence (by 2D keypoint coordinate, so it works across
SIFT backends), and the corresponding points the two solves place *most
differently* are rendered as patch strips — one row per point, the reference
solve on the left and the target solve on the right. Each tile is the point's
(optionally normal-refined) oriented surfel projected into one observing view,
so a clean column means the solve placed a real, photoconsistent surface point
there.

Images and ``.sift`` files are resolved from each reconstruction's
``workspace_dir`` (as with the photometric ``xform`` filters). The NCC scoring
and strip rendering live in ``_patch_ncc``; the montage layout in
``_strip_montage``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._point_correspondence import find_point_correspondences_by_coordinate
from ._sfmtool import SfmrReconstruction
from ._sfmtool.geometry import Se3Transform
from ._solve_strips import _SolveStrips
from ._strip_montage import MontageRow, assemble_montage

# Render/selection constants. These are not user-facing knobs; they are tuned
# defaults shared by every invocation.
_PATCH = 32  # surfel patch render resolution (px)
_EXTENT_FACTOR = 5.0  # PatchCloud extent as a multiple of feature size
_DISP = 72  # tile size shown when no wider --strips-context is requested
_MIN_TRACK = 3  # minimum observing views for a point to be renderable
_MIN_VOTES = 2  # minimum coordinate-match votes to accept a correspondence
_REFINE_RANGE_DEG = 25.0  # normal-refinement angular search half-range
_REFINE_STEPS = 7  # normal-refinement initial search steps

# Single-axis --strips-rank modes: each ranks corresponding points by a quantity
# toward one end. The natural end is used when --strips-end is unset.
_AXIS_DEFAULT_END = {
    "distance": "high",  # least aligned
    "view-angle": "low",  # parallax-starved
    "ncc": "low",  # suspicious surfel
    "ncc-gap": "high",  # quality differs between solves
    "image-radius": "high",  # peripheral
    "feature-size": "high",  # coarse/blurry features
    "world-size": "high",  # large surface footprint
}
_AXIS_LABEL = {
    "distance": {"low": "most aligned", "high": "least aligned"},
    "view-angle": {
        "low": "narrowest triangulation angle",
        "high": "widest triangulation angle",
    },
    "ncc": {"low": "lowest photoconsistency", "high": "highest photoconsistency"},
    "ncc-gap": {
        "low": "smallest photoconsistency gap",
        "high": "largest photoconsistency gap",
    },
    "image-radius": {"low": "most central", "high": "most peripheral"},
    "feature-size": {
        "low": "smallest image features",
        "high": "largest image features",
    },
    "world-size": {
        "low": "smallest world footprint",
        "high": "largest world footprint",
    },
}


def _overview_rows(
    candidates,
    r1: _SolveStrips,
    r2: _SolveStrips,
    finite_new,
    finite_old,
    matched_ref,
    matched_tgt,
    num: int,
    cap: int | None,
    left_label: str,
    right_label: str,
) -> list[tuple[str | None, int | None, int | None, float | None]]:
    """Assemble the default `overview` rows: a few points from each category.

    Returns a list of ``(group_label, ref_pid_or_None, tgt_pid_or_None,
    dist_or_None)`` rows. Single-column rows (unique to one solve) leave the other
    pid ``None``. ``candidates`` are the finite corresponding pairs
    ``(ref_pid, tgt_pid, dist, ref_angle, tgt_angle)``.
    """
    per = max(2, num // 8)
    rows = []
    used_ref, used_tgt = set(), set()

    def add(group, rp, tp, dist) -> bool:
        if rp is not None and rp in used_ref:
            return False
        if tp is not None and tp in used_tgt:
            return False
        rows.append((group, rp, tp, dist))
        if rp is not None:
            used_ref.add(rp)
        if tp is not None:
            used_tgt.add(tp)
        return True

    def take(group, ordered, n) -> None:
        taken = 0
        for cand in ordered:
            if taken >= n:
                break
            if add(group, cand[0], cand[1], cand[2]):
                taken += 1

    take(
        "least aligned (largest 3D distance)",
        sorted(candidates, key=lambda t: t[2], reverse=True),
        per + 1,
    )
    take(
        "narrowest triangulation angle",
        sorted(candidates, key=lambda t: min(t[3], t[4])),
        per,
    )
    take(
        "widest triangulation angle",
        sorted(candidates, key=lambda t: min(t[3], t[4]), reverse=True),
        per,
    )
    take(
        "most peripheral (near image edge)",
        sorted(candidates, key=lambda t: r1.image_radius(t[0]), reverse=True),
        per,
    )

    # Photometric sections need a render per candidate; bound the pool on large
    # datasets to the regions where the NCC extremes are most likely to live.
    if len(candidates) <= 300:
        pool = candidates
    else:
        by_dist = sorted(candidates, key=lambda t: t[2], reverse=True)[:120]
        by_angle = sorted(candidates, key=lambda t: min(t[3], t[4]))[:120]
        strided = candidates[:: max(1, len(candidates) // 100)]
        seen, pool = set(), []
        for c in by_dist + by_angle + strided:
            if (c[0], c[1]) not in seen:
                seen.add((c[0], c[1]))
                pool.append(c)
    scored = []
    for p1, p2, dist, a1, a2 in pool:
        s1 = r1.strip(p1, tile=_PATCH, max_views=cap)
        s2 = r2.strip(p2, tile=_PATCH, max_views=cap)
        if s1 is not None and s2 is not None:
            scored.append((p1, p2, dist, a1, a2, s1[1], s2[1]))
    take(
        "largest NCC gap (quality differs)",
        [t[:5] for t in sorted(scored, key=lambda t: abs(t[5] - t[6]), reverse=True)],
        per,
    )
    take(
        "lowest NCC (suspicious surfel)",
        [t[:5] for t in sorted(scored, key=lambda t: min(t[5], t[6]))],
        per,
    )

    # Points unique to one solve: single-column rows under the configured labels.
    unew = [
        p
        for p in r2.obs
        if finite_new[p] and len(r2.obs[p]) >= _MIN_TRACK and p not in matched_tgt
    ]
    unew.sort(key=lambda p: len(r2.obs[p]), reverse=True)
    for p in unew[: per + 1]:
        add(f"unique to {right_label}", None, p, None)
    uold = [
        p
        for p in r1.obs
        if finite_old[p] and len(r1.obs[p]) >= _MIN_TRACK and p not in matched_ref
    ]
    uold.sort(key=lambda p: len(r1.obs[p]), reverse=True)
    for p in uold[:per]:
        add(f"unique to {left_label}", p, None, None)
    return rows


def render_comparison_strips(
    recon1: SfmrReconstruction,
    recon2: SfmrReconstruction,
    matches: list[tuple[int, int]],
    transform: Se3Transform,
    out_path: str | Path,
    *,
    recon1_name: str = "reference",
    recon2_name: str = "target",
    left_label: str = "reference",
    right_label: str = "target",
    num: int = 16,
    context: int = 0,
    max_views: int = 8,
    pixel_threshold: float = 2.0,
    refine: bool = True,
    rank: str = "overview",
    end: str | None = None,
    scene_scale: float = 1.0,
) -> str | None:
    """Render points where the two solves disagree, as a side-by-side patch-strip
    montage written to ``out_path``.

    ``rank`` selects what to surface:

    - ``"overview"`` (default) — a few points from each of several categories
      (least aligned, narrowest/widest view angle, most peripheral, largest NCC
      gap, lowest NCC, and points unique to each solve), each under a labeled
      divider; unique points render in one column with the other blank;
    - a single quantity — ``"distance"`` (alignment disagreement), ``"view-angle"``
      (triangulation angle), ``"ncc"`` (per-solve photoconsistency), ``"ncc-gap"``
      (gap between the two solves), ``"image-radius"`` (distance of the keypoint
      from the principal point), ``"feature-size"`` (keypoint feature size in
      pixels), or ``"world-size"`` (that feature size projected to a metric
      surface footprint via depth / focal). ``end`` (``"high"``/``"low"``) picks
      which end of the axis; ``None`` uses the axis's natural end. ``"ncc"`` /
      ``"ncc-gap"`` render every candidate to score it, so they are slower.

    ``matches`` are shared-image ``(idx1, idx2)`` pairs and ``transform`` is the
    similarity aligning recon2 onto recon1 (both already computed by
    ``compare_reconstructions``). ``scene_scale`` makes the reported distances
    gauge-independent. Returns the written path, or ``None`` if no renderable
    corresponding points were found.
    """
    try:
        _corr, pos1, pos2 = find_point_correspondences_by_coordinate(
            source_recon=recon1,
            target_recon=recon2,
            shared_images=matches,
            pixel_threshold=pixel_threshold,
            min_votes=_MIN_VOTES,
        )
    except ValueError:
        print("  --strips: no corresponding points to render")
        return None

    ids1 = list(_corr.keys())
    ids2 = list(_corr.values())
    distances = np.linalg.norm((transform @ pos2) - pos1, axis=1)

    r1 = _SolveStrips(
        recon1, recon1.workspace_dir, patch=_PATCH, extent_factor=_EXTENT_FACTOR
    )
    r2 = _SolveStrips(
        recon2, recon2.workspace_dir, patch=_PATCH, extent_factor=_EXTENT_FACTOR
    )

    # Points at infinity (homogeneous w == 0) have no finite position, distance,
    # or patch extent, so they cannot be rendered as a surfel — skip them. (A
    # point parked at infinity by one solve but triangulated finitely by the
    # other is itself a real difference, but the strip view can't show it.)
    finite1 = np.abs(np.asarray(recon1.positions_xyzw)[:, 3]) > 1e-9
    finite2 = np.abs(np.asarray(recon2.positions_xyzw)[:, 3]) > 1e-9

    n_inf = 0
    candidates = []
    for i in range(len(ids1)):
        p1id, p2id = int(ids1[i]), int(ids2[i])
        if not (finite1[p1id] and finite2[p2id]):
            n_inf += 1
            continue
        if (
            len(r1.obs.get(p1id, [])) >= _MIN_TRACK
            and len(r2.obs.get(p2id, [])) >= _MIN_TRACK
        ):
            # (ref pid, tgt pid, 3D distance, ref tri-angle, tgt tri-angle)
            candidates.append(
                (
                    p1id,
                    p2id,
                    float(distances[i]),
                    r1.tri_angle(p1id),
                    r2.tri_angle(p2id),
                )
            )
    if n_inf:
        print(f"  --strips: skipped {n_inf} correspondence(s) at infinity (w=0)")
    if not candidates:
        print(
            f"  --strips: no finite corresponding points seen in >= {_MIN_TRACK} views"
        )
        return None

    cap = max_views or None
    ctx = context if context > _PATCH else 0
    tile = ctx if ctx else _DISP
    matched_ref = {int(x) for x in ids1}
    matched_tgt = {int(x) for x in ids2}

    # Build the ordered rows to render. A row is
    # (group_label_or_None, ref_pid_or_None, tgt_pid_or_None, dist_or_None).
    rows = []
    if rank == "overview":
        rows = _overview_rows(
            candidates,
            r1,
            r2,
            finite2,
            finite1,
            matched_ref,
            matched_tgt,
            num,
            cap,
            left_label,
            right_label,
        )
    else:
        # Single-axis modes: rank corresponding points by one quantity, toward
        # the high or low end (--strips-end; defaults to the axis's natural end).
        descending = (end or _AXIS_DEFAULT_END[rank]) == "high"
        if rank == "distance":
            candidates.sort(key=lambda t: t[2], reverse=descending)
        elif rank == "view-angle":
            candidates.sort(key=lambda t: min(t[3], t[4]), reverse=descending)
        elif rank == "image-radius":
            candidates.sort(key=lambda t: r1.image_radius(t[0]), reverse=descending)
        elif rank == "feature-size":
            candidates.sort(key=lambda t: r1.feature_size_px(t[0]), reverse=descending)
        elif rank == "world-size":
            candidates.sort(
                key=lambda t: r1.world_feature_size(t[0]), reverse=descending
            )
        else:  # "ncc" / "ncc-gap" — need a render per candidate to score
            scored = []
            for p1id, p2id, dist, a1, a2 in candidates:
                s1 = r1.strip(p1id, tile=_PATCH, max_views=cap)
                s2 = r2.strip(p2id, tile=_PATCH, max_views=cap)
                if s1 is None or s2 is None:
                    continue
                val = min(s1[1], s2[1]) if rank == "ncc" else abs(s1[1] - s2[1])
                scored.append((p1id, p2id, dist, a1, a2, val))
            scored.sort(key=lambda t: t[5], reverse=descending)
            candidates = [t[:5] for t in scored]
        for p1, p2, dist, a1, a2 in candidates[:num]:
            rows.append((None, p1, p2, dist))

    if not rows:
        print("  --strips: no candidates to render")
        return None

    if refine:
        ref_pids = [r[1] for r in rows if r[1] is not None]
        tgt_pids = [r[2] for r in rows if r[2] is not None]
        d1 = (
            r1.refine(
                ref_pids, angular_range_deg=_REFINE_RANGE_DEG, init_steps=_REFINE_STEPS
            )
            if ref_pids
            else float("nan")
        )
        d2 = (
            r2.refine(
                tgt_pids, angular_range_deg=_REFINE_RANGE_DEG, init_steps=_REFINE_STEPS
            )
            if tgt_pids
            else float("nan")
        )
        if np.isnan(d1) and np.isnan(d2):
            print(
                "  --strips: normal refinement had no valid photoconsistency "
                "(low-texture/degenerate points); using stored normals"
            )
        else:
            print(
                f"  --strips: normal refinement mean ΔΦ ref={d1:+.3f} target={d2:+.3f}"
            )

    # Triangulation angles were already computed when ranking; reuse them rather
    # than recomputing per row (every rendered corresponding pid is a candidate).
    ref_angle = {p1: a1 for p1, p2, dist, a1, a2 in candidates}
    tgt_angle = {p2: a2 for p1, p2, dist, a1, a2 in candidates}

    # Render each row's strips (blank column where a solve has no point), then
    # build its text labels for the montage.
    lc, rc = left_label[0].upper(), right_label[0].upper()

    def _views_note(recon: _SolveStrips, pid: int, shown: int) -> str:
        # "8/20v" only when --strips-views dropped some of the point's views.
        total = len(recon.obs.get(int(pid), []))
        return f" {shown}/{total}v" if shown < total else ""

    def _angle(recon: _SolveStrips, cached: dict, pid: int) -> float:
        # Corresponding pids reuse the angle computed during ranking; points
        # unique to one solve are not candidates, so compute on demand.
        a = cached.get(pid)
        return a if a is not None else recon.tri_angle(pid)

    montage_rows: list[MontageRow] = []
    for rk, (group, rp, tp, dist) in enumerate(rows):
        s1 = (
            r1.strip(rp, tile=tile, max_views=cap, context=ctx or None)
            if rp is not None
            else None
        )
        s2 = (
            r2.strip(tp, tile=tile, max_views=cap, context=ctx or None)
            if tp is not None
            else None
        )
        if s1 is None and s2 is None:
            continue
        dpct = (
            dist / scene_scale * 100.0 if (dist is not None and scene_scale) else None
        )
        if s1 is not None and s2 is not None and dpct is not None:
            head = f"#{rk} d={dpct:.0f}%"
        elif s2 is not None and s1 is None:
            head = f"#{rk} {right_label}-only"
        elif s1 is not None and s2 is None:
            head = f"#{rk} {left_label}-only"
        else:
            head = f"#{rk}"
        line_r = (
            f"{lc}{s1[1]:+.2f} a{_angle(r1, ref_angle, rp):.0f}"
            f"{_views_note(r1, rp, s1[2])}"
            if s1 is not None
            else f"{lc}  --"
        )
        line_t = (
            f"{rc}{s2[1]:+.2f} a{_angle(r2, tgt_angle, tp):.0f}"
            f"{_views_note(r2, tp, s2[2])}"
            if s2 is not None
            else f"{rc}  --"
        )
        montage_rows.append(
            (
                group,
                head,
                line_r,
                line_t,
                s1[0] if s1 is not None else None,
                s2[0] if s2 is not None else None,
            )
        )
    if not montage_rows:
        return None

    suffix = " (refined normals)" if refine else ""
    if ctx:
        suffix += "   green box = validated extent"
    rank_desc = (
        "category (a few points from each)"
        if rank == "overview"
        else _AXIS_LABEL[rank][end or _AXIS_DEFAULT_END[rank]]
    )
    title = f"corresponding points by {rank_desc}{suffix}"
    legend = (
        f"d = % of scene scale    {lc}/{rc} = {left_label}/{right_label}:  "
        "NCC,  a = triangulation angle (deg),  N/Mv = N of M views shown"
    )

    width, height = assemble_montage(
        out_path,
        montage_rows,
        title=title,
        legend=legend,
        left_label=left_label,
        right_label=right_label,
        left_name=recon1_name,
        right_name=recon2_name,
        tile=tile,
        disp=_DISP,
    )
    print(
        f"  --strips: wrote {out_path} ({width}x{height}, {len(montage_rows)} points)"
    )
    return str(out_path)
