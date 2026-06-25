# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Feature-source preconditions shared by the gated surfel operations.

``sfm xform --refine-normals`` and ``sfm render-patches`` require an
``embedded_patches`` reconstruction — the only source that carries per-point
patch frames and per-observation keypoints — and reject ``sift_files`` with a
pointer to the conversion bridge. (``compare --strips`` is deliberately *not*
gated: it stays a dual-source diagnostic; see ``specs/cli/compare-command.md``.)
Enforcement lives here (a command/transform-layer check); the low-level
``PatchCloud.from_reconstruction`` stays dual-mode.
"""

from __future__ import annotations

import click

from ._sfmtool import SfmrReconstruction


def require_embedded_patches(recon: SfmrReconstruction, op: str) -> None:
    """Raise :class:`click.UsageError` unless ``recon`` is ``embedded_patches``.

    Args:
        recon: The reconstruction the operation is about to run on.
        op: Human-readable name of the operation, used in the error (e.g.
            ``"sfm render-patches"``).
    """
    fs = recon.feature_source
    if fs != "embedded_patches":
        raise click.UsageError(
            f"{op} requires an embedded_patches reconstruction (this one is "
            f"{fs}); run `sfm xform --to-embedded-patches` (or `sfm embed-patches`) "
            "first."
        )
