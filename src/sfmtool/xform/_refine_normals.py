# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Grown-plane normal refinement as an ``sfm xform`` operation.

``RefineNormalsTransform`` turns each finite point's patch to its grown-plane
normal through ``sfmtool.bench.refine_normals``, reading the workspace's source
images. It removes no point and moves no centre and no keypoint. See
``specs/cli/reconstruction/xform/refine-normals-command.md`` and
``specs/core/bench/refine-normals.md``.
"""

import math

from .._sfmtool.reconstruction import SfmrReconstruction
from ._images import load_workspace_images


class RefineNormalsTransform:
    """Turn every finite point's patch to its grown-plane normal.

    Each point is put through the bench's *Grown Plane Normal* and *Commit*
    with every observation kept (``sfmtool.bench.refine_normals`` on its
    default settings), so a turned point gets a new normal, patch frame,
    colour and readings, and a new bitmap where the reconstruction stores
    bitmaps. Requires an ``embedded_patches`` reconstruction (enforced by
    ``apply_transforms``).
    """

    # Precondition checked per-step by `apply_transforms` (see `_apply.py`).
    required_feature_source = "embedded_patches"

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        from .._sfmtool.bench import refine_normals

        # Refuse a value with no patch frames before the photographs are
        # decoded, naming the step that builds them.
        if recon.patches is None:
            raise ValueError(
                "reconstruction has no patch frames to refine; expected "
                "embedded_patches (run `sfm xform --to-embedded-patches` first)"
            )
        images = load_workspace_images(recon)
        print(
            f"  Refining the normals of {recon.point_count} points "
            f"across {recon.image_count} images..."
        )
        try:
            out, report = refine_normals(recon, images)
        except ValueError as e:
            raise ValueError(f"--refine-normals: {e}") from e
        print(_summary(report))
        return out

    def description(self) -> str:
        # Also the operation name in the precondition error.
        return "Refine normals (grown plane)"


def _summary(report: dict) -> str:
    """The one-line ``xform`` summary of a refinement's report."""
    line = f"  Turned {report['turned']} of {report['read']} normals"
    if report["turned"]:
        agreement = report["mean_agreement"]
        agreement_text = "n/a" if math.isnan(agreement) else f"{agreement:.2f}"
        line += (
            f" (median turn {report['median_turn_deg']:.1f}°, "
            f"mean agreement {agreement_text}; "
            f"{report['low_confidence']} low-confidence)"
        )
    line += (
        f"; left unchanged: {report['none_joined']} no view joined, "
        f"{report['weak_turn']} large turn on weak evidence, "
        f"{report['kept']} refused or with no bitmap"
    )
    if report["at_infinity"]:
        line += f"; {report['at_infinity']} at infinity skipped"
    return line
