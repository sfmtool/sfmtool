# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers and fixtures for the patch tests.

The helpers were copy-pasted across six of the modules in this directory
before it existed; import them from here rather than adding a fresh copy.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from ..conftest import SEOUL_BULL_POINT_STEP, every_nth_point


def load_images(recon) -> list[np.ndarray]:
    """Every image of a reconstruction, decoded to contiguous RGB."""
    from sfmtool.fileio import read_image_rgb

    ws = recon.workspace_dir
    return [read_image_rgb(os.path.join(ws, name)) for name in recon.image_names]


def sample_point_ids(cloud, n: int = 200, seed: int = 0, restrict_to=None) -> list[int]:
    """A deterministic point-id subset, to keep the per-point search fast.

    ``restrict_to`` narrows the pool to the given point ids before drawing, for a
    test whose assertion holds only for points with a particular property: a
    120-point draw from the whole cloud can otherwise contain none of them and
    fail on a perfectly correct reconstruction.
    """
    ids = np.asarray(cloud.point_indexes)
    if restrict_to is not None:
        wanted = np.asarray(sorted(restrict_to), dtype=ids.dtype)
        ids = ids[np.isin(ids, wanted)]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False)).tolist()


def rotation_matrices(recon) -> np.ndarray:
    """Per-image world-to-camera rotation matrices (Mx3x3) from wxyz quaternions."""
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w * w + x * x + y * y + z * z
    s = np.where(n > 0, 2.0 / n, 0.0)
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - s * (y * y + z * z)
    R[:, 0, 1] = s * (x * y - z * w)
    R[:, 0, 2] = s * (x * z + y * w)
    R[:, 1, 0] = s * (x * y + z * w)
    R[:, 1, 1] = 1 - s * (x * x + z * z)
    R[:, 1, 2] = s * (y * z - x * w)
    R[:, 2, 0] = s * (x * z - y * w)
    R[:, 2, 1] = s * (y * z + x * w)
    R[:, 2, 2] = 1 - s * (x * x + y * y)
    return R


@dataclass
class EmbedPatchesRun:
    """One ``embed_patches`` run and what it was seen to do on the way.

    ``refine_normals_calls`` holds, per call of the grown-plane normal step,
    the reconstruction it read, its keyword arguments, its report's ``turned``
    count and the normals it wrote. ``selected_normals`` holds the normal of
    every patch view selection read, in order. ``sift_hash_reads`` counts the
    calls of ``image_file_hashes_from_sift``.
    """

    recon: object
    images: list
    out: object = None
    refine_normals_calls: list = field(default_factory=list)
    selected_normals: list = field(default_factory=list)
    sift_hash_reads: int = 0


def patch_normals(cloud) -> list[np.ndarray]:
    """Each patch's normal, in cloud order."""
    return [np.asarray(cloud[i].normal, np.float64) for i in range(len(cloud))]


@pytest.fixture(scope="session")
def default_embed_patches_run(seoul_bull_workspace_once: Path) -> EmbedPatchesRun:
    """``embed_patches`` at its defaults but for ``resolution=12``, run once on
    every eighth point of the seoul_bull workspace (``SEOUL_BULL_POINT_STEP``)
    for the tests that only read its output, with
    spies on the steps some of them look into. The spies pass every call
    through, so the output is the plain run's. Read it; do not modify it, its
    images or its source workspace.
    """
    import sfmtool._embed_patches as ep
    import sfmtool._patch_compaction as pc
    from sfmtool.patches import PatchCloud
    from sfmtool.reconstruction import SfmrReconstruction

    recon = every_nth_point(
        SfmrReconstruction.load(seoul_bull_workspace_once), SEOUL_BULL_POINT_STEP
    )
    run = EmbedPatchesRun(recon=recon, images=load_images(recon))

    real_refine = ep.refine_normals
    real_select = PatchCloud.select_views
    real_sift_hashes = pc.image_file_hashes_from_sift

    def spy_refine(recon, images, **kwargs):
        out, report = real_refine(recon, images, **kwargs)
        run.refine_normals_calls.append(
            {
                "recon": recon,
                "kwargs": kwargs,
                "turned": report["turned"],
                "normals": patch_normals(out.patches),
            }
        )
        return out, report

    def spy_select(self, recon, images, **kwargs):
        run.selected_normals.extend(patch_normals(self))
        return real_select(self, recon, images, **kwargs)

    def spy_sift_hashes(recon):
        run.sift_hash_reads += 1
        return real_sift_hashes(recon)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ep, "refine_normals", spy_refine)
        mp.setattr(PatchCloud, "select_views", spy_select)
        mp.setattr(pc, "image_file_hashes_from_sift", spy_sift_hashes)
        run.out = ep.embed_patches(run.recon, run.images, resolution=12)
    return run
