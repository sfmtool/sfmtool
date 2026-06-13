# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for ``PatchCloud.refine_normals`` against a real reconstruction.

Builds a patch cloud from the solved 17-image seoul_bull reconstruction and runs
photometric normal refinement over its real ``.sift``-derived patches and source
images — the multi-view rendering path the Rust unit tests can't exercise without
on-disk images.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction


def _load_images(recon) -> list[np.ndarray]:
    import cv2  # heavy module, only needed by this integration test

    ws = recon.workspace_dir
    images = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        images.append(np.ascontiguousarray(bgr))
    return images


def test_refine_normals_improves_consensus(
    sfmrfile_reconstruction_with_17_images: Path,
):
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    images = _load_images(recon)

    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0
    n0 = np.array([cloud[i].normal for i in range(len(cloud))])

    # Modest search params keep the test quick; correctness, not quality.
    res = cloud.refine_normals(
        recon,
        images,
        resolution=12,
        init_steps=5,
        refine_levels=2,
        sampler="bilinear",
    )

    normals = res["normal"]
    photo = res["photoconsistency"]
    init = res["init_photoconsistency"]
    conf = res["confidence"]
    vvc = res["valid_view_count"]

    assert normals.shape == (len(cloud), 3)

    # Returned normals are unit length and finite.
    norms = np.linalg.norm(normals, axis=1)
    assert np.all(np.isfinite(norms))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # The cloud was updated in place to the returned normals.
    n1 = np.array([cloud[i].normal for i in range(len(cloud))])
    np.testing.assert_allclose(n1, normals, atol=1e-9)

    # Some patches were actually scored (had enough valid views).
    scored = np.isfinite(photo) & np.isfinite(init)
    assert scored.sum() > 0
    assert np.all(vvc[scored] >= 3)

    # Refinement never lowers the consensus relative to the init (frozen support),
    # and strictly improves at least one patch (it genuinely does something).
    assert np.all(photo[scored] >= init[scored] - 1e-9)
    assert np.any(photo[scored] > init[scored] + 1e-6)

    # Confidence is finite and non-negative.
    assert np.all(np.isfinite(conf))
    assert np.all(conf >= 0.0)

    # At least one normal moved away from its mean-viewing init.
    moved = 1.0 - np.abs(np.sum(n0 * n1, axis=1))
    assert np.nanmax(moved) > 1e-4
