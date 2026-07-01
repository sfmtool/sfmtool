# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render chosen points of a single reconstruction as patch strips.

This backs ``sfm inspect --strips``. Given an explicit list of points (by
``pt3d_<hash>_<index>`` id or point-index range expression), each point is laid
out as one montage row: ``labels | reference patch | per-view observation
strip``, for visually evaluating point quality.

The surfel rendering and NCC scoring reuse the ``compare --strips`` engine
(``_SolveStrips``); the montage layout lives in ``_strip_montage``. A
``sift_files`` reconstruction is first given the minimal ``embedded_patches``
conversion plus a light normal refinement over only the listed points (the
default keypoint-preserving conversion makes these good); an ``embedded_patches``
reconstruction is rendered as stored.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np

from ._sfmtool import SfmrReconstruction
from ._solve_strips import _SolveStrips
from ._strip_montage import PointRow, assemble_point_strips

# Render constants shared with the comparison montage (not user-facing knobs).
_PATCH = 32  # surfel patch render resolution (px)
_EXTENT_FACTOR = 2.5  # PatchCloud half-extent as a multiple of feature size
_DISP = 72  # tile size shown when no wider --strips-context is requested

# A 3D point ID: pt3d_<8 hex chars of the .sfmr content hash>_<point index>.
_POINT_ID_RE = re.compile(r"^pt3d_([0-9a-fA-F]{8})_(\d+)$")


def parse_point_specs(recon: SfmrReconstruction, specs: list[str]) -> list[int]:
    """Resolve ``specs`` to an ordered, de-duplicated list of point indexes.

    Each spec is either a ``pt3d_<hash>_<index>`` id (whose ``<hash>`` must match
    this reconstruction's content hash) or a point-index range expression (e.g.
    ``5``, ``5-12``, ``1,4,7``). Indexes are kept in the order the specs list
    them; out-of-range indexes and id/hash mismatches raise ``click.UsageError``.
    """
    from ._sfmtool import RangeExpr

    hash8 = recon.content_xxh128[:8].lower()
    n = recon.point_count
    out: list[int] = []
    seen: set[int] = set()

    def add(idx: int, origin: str) -> None:
        if not 0 <= idx < n:
            raise click.UsageError(
                f"point index {idx} (from {origin!r}) is out of range — this "
                f"reconstruction has {n} points (valid 0..{n - 1})."
            )
        if idx not in seen:
            seen.add(idx)
            out.append(idx)

    for spec in specs:
        match = _POINT_ID_RE.match(spec)
        if match is not None:
            spec_hash = match.group(1).lower()
            if spec_hash != hash8:
                raise click.UsageError(
                    f"point id {spec!r} is not for this reconstruction — its ids "
                    f"begin pt3d_{hash8}_ (got hash {spec_hash})."
                )
            add(int(match.group(2)), spec)
            continue
        try:
            numbers = list(RangeExpr(spec))
        except Exception:
            raise click.UsageError(
                f"invalid point spec {spec!r}: expected a pt3d_<hash>_<index> id "
                "or a point-index range expression (e.g. '5', '5-12', '1,4,7')."
            )
        if not numbers:
            raise click.UsageError(f"point spec {spec!r} selected no points.")
        for idx in numbers:
            add(int(idx), spec)

    if not out:
        raise click.UsageError("no points specified for --strips.")
    return out


def _prepare_from_sift(
    recon: SfmrReconstruction, point_indexes: list[int], at_infinity: np.ndarray
) -> tuple[SfmrReconstruction, list[np.ndarray]]:
    """Convert a ``sift_files`` recon to ``embedded_patches`` and refine the listed
    finite points' normals (rendering their patch bitmaps). Returns the prepared
    recon and the loaded workspace images (so the renderer can reuse them)."""
    from ._workspace_image import read_workspace_image
    from .xform._to_embedded_patches import ToEmbeddedPatchesTransform

    emb = ToEmbeddedPatchesTransform().apply(recon)
    images = [read_workspace_image(emb.workspace_dir, name) for name in emb.image_names]

    # Points at infinity have a fixed normal the refiner skips; refine only the
    # listed finite points (keeping the work proportional to what is rendered).
    finite_listed = [int(p) for p in point_indexes if not at_infinity[p]]
    cloud = emb.patches
    if not finite_listed or cloud is None:
        return emb, images

    print(f"  --strips: refining normals for {len(finite_listed)} listed point(s)...")
    result = cloud.refine_normals(
        emb,
        images,
        point_indexes=finite_listed,
        resolution=24,
        refine_levels=2,
        use_stored_keypoints=True,
        render_bitmaps=True,
    )
    # Scatter the refined normals back to finite points (non-refined points keep
    # their seed, so the stored normals stay consistent with the cloud frames).
    refined = np.asarray(result["normal"], dtype=np.float32)
    point_ids = np.asarray(cloud.point_indexes)
    finite = ~np.asarray(emb.point_is_at_infinity)[point_ids]
    normals = np.asarray(emb.normals, dtype=np.float32).copy()
    normals[point_ids[finite]] = refined[finite]
    prepared = emb.clone_with_changes(
        normals=normals, patches=cloud, patch_bitmaps=result["bitmaps"]
    )
    return prepared, images


def render_inspect_strips(
    recon: SfmrReconstruction,
    point_indexes: list[int],
    out_path: str | Path,
    *,
    max_views: int = 8,
    context: float = 1.0,
) -> str | None:
    """Render the listed points as a single-reconstruction patch-strip montage
    written to ``out_path``. Returns the written path, or ``None`` if no listed
    point could be rendered.

    ``recon`` must be ``sift_files`` (converted + lightly refined here) or
    ``embedded_patches`` (rendered as stored). ``max_views`` caps the observation
    tiles per point (0 = all). ``context`` pads each observation tile with that
    fraction of extra field of view around the patch (1.0 = +100%) and boxes the
    patch extent; 0 renders tight, borderless tiles. The reference patch always
    renders tight.
    """
    hash8 = recon.content_xxh128[:8].lower()
    at_infinity = np.asarray(recon.point_is_at_infinity)
    source = recon.feature_source

    if source == "sift_files":
        prepared, images = _prepare_from_sift(recon, point_indexes, at_infinity)
    elif source == "embedded_patches":
        prepared, images = recon, None
    else:
        raise click.UsageError(
            f"--strips needs a sift_files or embedded_patches reconstruction "
            f"(this one is {source})."
        )

    engine = _SolveStrips(
        prepared,
        prepared.workspace_dir,
        patch=_PATCH,
        extent_factor=_EXTENT_FACTOR,
        exclude_points_at_infinity=False,
    )
    if images is not None:
        engine.prime_images(images)

    cap = max_views or None
    # `context` is a fraction of extra field of view around the patch; turn it
    # into the absolute render resolution `strip()` expects (must exceed _PATCH).
    context_px = round(_PATCH * (1.0 + context)) if context > 0 else 0
    ctx = context_px if context_px > _PATCH else 0
    # Grow the display tile with the context so the patch portion stays ~_DISP
    # and the padding is added around it.
    tile = round(_DISP * (1.0 + context)) if ctx else _DISP
    # The reference patch renders tight; size it so the patch matches the scale
    # of the boxed patch region inside each context-padded observation tile.
    ref_tile = round(tile * _PATCH / ctx) if ctx else tile

    rows: list[PointRow] = []
    skipped: list[int] = []
    for idx in point_indexes:
        strip = engine.strip(
            idx,
            tile=tile,
            max_views=cap,
            context=ctx or None,
            annotate=True,
            normal_dot=True,
        )
        ref = engine.reference_patch(idx, tile=ref_tile, max_views=cap)
        if strip is None or ref is None:
            skipped.append(idx)
            continue
        strip_img, ncc, shown = strip
        total = len(engine.obs.get(idx, []))
        views_note = f"{shown}/{total}v" if shown < total else f"{shown}v"
        if at_infinity[idx]:
            labels = [
                f"pt3d_{hash8}_{idx}",
                f"infinity NCC{ncc:+.2f}",
                views_note,
            ]
        else:
            labels = [
                f"pt3d_{hash8}_{idx}",
                f"finite NCC{ncc:+.2f}",
                f"a{engine.tri_angle(idx):.0f}  {views_note}",
            ]
        rows.append((labels, ref, strip_img))

    if skipped:
        print(f"  --strips: skipped {len(skipped)} point(s) with no observations")
    if not rows:
        print("  --strips: no listed points could be rendered")
        return None

    suffix = "   green box = patch extent" if ctx else ""
    title = f"point strips - {Path(out_path).name}  ({len(rows)} points, {source})"
    legend = [
        "left: reference patch (stored bitmap if present, else cross-view consensus)",
        "right tiles -- top: image index;  bottom: n = NCC vs other views, "
        "e = reprojection error px",
        f"NCC = mean pairwise photoconsistency, a = triangulation angle (deg){suffix}",
        "magenta dot = view obliquity (centre = fronto-parallel, box edge = grazing)",
    ]
    width, height = assemble_point_strips(
        out_path, rows, title=title, legend=legend, tile=tile
    )
    print(f"  --strips: wrote {out_path} ({width}x{height}, {len(rows)} points)")
    return str(out_path)
