# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the embedded_patches compaction glue (``sfmtool._embed_patches``).

Runs the patch kernels (normal refine + view selection + keypoint localization)
on a real solved reconstruction, compacts the result into an ``embedded_patches``
reconstruction, and round-trips it through ``.sfmr`` to confirm validity. See
``specs/core/sift-to-patch-reconstruction.md`` and
``specs/formats/sfmr-file-format.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sfmtool._embed_patches import (
    compact_to_embedded_patches,
    image_file_hashes_from_images,
)
from sfmtool._sfmtool import PatchCloud, SfmrReconstruction
from sfmtool._sfmtool.io import verify_sfmr


def _load_images(recon) -> list[np.ndarray]:
    import cv2

    ws = recon.workspace_dir
    out = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        out.append(np.ascontiguousarray(bgr))
    return out


def _run_pipeline(recon, images, resolution=12):
    """The upstream kernels the compaction consumes (refine → select → localize)."""
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    res = cloud.refine_normals(
        recon, images, resolution=resolution, render_bitmaps=True
    )
    sel = cloud.select_views(recon, images, resolution=resolution)
    view_sets = {int(r["point_index"]): np.asarray(r["admitted"]).tolist() for r in sel}
    locs = cloud.localize_keypoints(
        recon, images, view_sets=view_sets, resolution=resolution
    )
    return cloud, res["bitmaps"], locs


def test_from_halfvec_arrays_round_trips_a_cloud():
    """The new PatchCloud.from_halfvec_arrays binding rebuilds a cloud, keeping
    present (non-zero u) rows and recording their indices as point_ids."""
    u = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    v = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float32)
    centers = np.array(
        [[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [2.0, 0.0, 5.0]], dtype=np.float64
    )
    cloud = PatchCloud.from_halfvec_arrays(u, v, centers)
    # Row 1 has a zero u -> dropped; rows 0 and 2 survive with their indices.
    assert list(cloud.point_indexes) == [0, 2]
    assert len(cloud) == 2
    p0 = cloud[0]
    assert np.allclose(p0.center, [1.0, 1.0, 1.0])
    assert np.allclose(p0.half_extent, [2.0, 2.0])
    assert np.allclose(np.asarray(p0.u_axis) * p0.half_extent[0], [2.0, 0.0, 0.0])


def test_image_file_hashes_from_images_shape(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    hashes = image_file_hashes_from_images(recon)
    assert len(hashes) == recon.image_count
    assert all(isinstance(h, bytes) and len(h) == 16 for h in hashes)


def test_compact_to_embedded_patches_round_trip(
    seoul_bull_workspace: Path, tmp_path: Path
):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"
    images = _load_images(recon)
    cloud, bitmaps, locs = _run_pipeline(recon, images)
    hashes = image_file_hashes_from_images(recon)

    new = compact_to_embedded_patches(
        recon, cloud, locs, hashes, patch_bitmaps=bitmaps, min_views=2
    )

    # In-memory shape of the compacted reconstruction.
    assert new.feature_source == "embedded_patches"
    assert 0 < new.point_count <= recon.point_count
    assert new.image_count == recon.image_count
    oc = np.asarray(new.observation_counts)
    assert oc.min() >= 2, "every surviving point keeps at least min_views observations"
    kxy = np.asarray(new.keypoints_xy)
    assert kxy.shape == (int(oc.sum()), 2)
    assert np.all(np.isfinite(kxy))
    assert new.track_feature_indexes is None, (
        "embedded_patches carries no feature_indexes"
    )
    # One patch frame per surviving point (catches a frameless/misaligned cloud).
    assert len(new.patches) == new.point_count

    # Association: each new point's (image, keypoint) rows and geometry match the
    # source point's localization — proving the renumbering kept everything aligned,
    # not just that the keypoint multiset round-trips.
    cloud_pids = {int(p) for p in cloud.point_indexes}
    survivors = sorted(
        int(loc["point_index"])
        for loc in locs
        if int(loc["point_index"]) in cloud_pids and len(np.asarray(loc["views"])) >= 2
    )
    loc_by_pid = {int(loc["point_index"]): loc for loc in locs}
    src_pos = np.asarray(recon.positions)
    new_pos = np.asarray(new.positions)
    tpid = np.asarray(new.track_point_indexes)
    timg = np.asarray(new.track_image_indexes)
    for new_id, old_id in enumerate(survivors):
        loc = loc_by_pid[old_id]
        vs = np.asarray(loc["views"], dtype=np.int64)
        kp = np.asarray(loc["keypoints"], dtype=np.float64).reshape(-1, 2)
        order = np.argsort(vs, kind="stable")
        mask = tpid == new_id
        got_img = timg[mask]
        got_kp = kxy[mask]
        assert list(got_img) == list(vs[order].astype(np.int64)), (
            f"point {new_id} (src {old_id}): image set mismatch"
        )
        np.testing.assert_allclose(got_kp, kp[order], atol=5e-2)
        np.testing.assert_allclose(new_pos[new_id], src_pos[old_id], atol=1e-6)

    # Round-trip through .sfmr: writes (the writer requires the patch frame),
    # verifies, and reloads as embedded_patches with the same data.
    out = tmp_path / "embedded.sfmr"
    new.save(str(out), operation="embed_patches")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"

    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.feature_source == "embedded_patches"
    assert reloaded.point_count == new.point_count
    np.testing.assert_allclose(np.asarray(reloaded.keypoints_xy), kxy, atol=1e-4)
    assert [bytes(h) for h in reloaded.image_file_hashes] == list(hashes)
    # The required patch frame round-trips (one patch per point), and the bitmaps.
    rcloud = reloaded.patches
    assert rcloud is not None
    assert len(rcloud) == reloaded.point_count
    assert reloaded.patch_bitmaps is not None
    assert reloaded.patch_bitmaps.shape[0] == reloaded.point_count


def test_compact_min_views_culls_points(seoul_bull_workspace: Path):
    """Raising min_views drops more points (and never keeps an under-supported one)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
    cloud, bitmaps, locs = _run_pipeline(recon, images)
    hashes = image_file_hashes_from_images(recon)

    low = compact_to_embedded_patches(recon, cloud, locs, hashes, min_views=2)
    high = compact_to_embedded_patches(recon, cloud, locs, hashes, min_views=4)

    # seoul_bull (17 images) has points with 2-3 kept views, so raising the floor
    # to 4 must actually drop some — not merely keep the count equal.
    assert high.point_count < low.point_count
    assert np.asarray(high.observation_counts).min() >= 4
    assert np.asarray(low.observation_counts).min() >= 2


def test_compact_preserves_points_at_infinity(seoul_bull_workspace: Path):
    """A surviving point at infinity (w = 0) stays at infinity through compaction —
    the rebuild carries the homogeneous w, not just the Euclidean position."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    # Turn one well-observed point into a point at infinity.
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    counts = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=recon.point_count
    )
    pi = int(np.argmax(counts))
    xyz = pos[pi, :3]
    pos[pi] = np.append(xyz / np.linalg.norm(xyz), 0.0)
    recon = recon.clone_with_changes(positions=pos)
    assert bool(np.asarray(recon.point_is_at_infinity)[pi])

    # A cloud that includes infinity points (fixed extent needs no .sift scales).
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=0.05
    )

    # Fabricate localizations (point center per view) for the infinity point plus
    # two finite points, each kept with >= min_views observations.
    tpids = np.asarray(recon.track_point_indexes)
    timgs = np.asarray(recon.track_image_indexes)
    cam_idx = np.asarray(recon.camera_indexes)
    cams = recon.cameras
    chosen = [pi] + [p for p in (0, 1, 2) if p != pi][:2]
    locs = []
    for p in chosen:
        views = np.unique(timgs[tpids == p])[:3]
        if len(views) < 2:
            continue
        kpts = np.array(
            [
                [cams[int(cam_idx[v])].width / 2.0, cams[int(cam_idx[v])].height / 2.0]
                for v in views
            ],
            dtype=np.float64,
        )
        locs.append(
            {
                "point_index": int(p),
                "views": views.astype(np.uint32),
                "keypoints": kpts,
                "offsets_px": np.zeros(len(views)),
                "loo_zncc": np.full(len(views), np.nan),
            }
        )
    hashes = [b"\x00" * 16] * recon.image_count

    out = compact_to_embedded_patches(recon, cloud, locs, hashes, min_views=2)

    # The infinity point survived and is still at infinity in the output.
    is_inf = np.asarray(out.point_is_at_infinity)
    assert is_inf.sum() == 1, "the one infinity point should survive as w = 0"
    valid, errors = verify_sfmr_to_temp(out)
    assert valid, f"integrity check failed: {errors}"


def verify_sfmr_to_temp(recon) -> tuple[bool, list]:
    """Save to a temp .sfmr and run the format integrity check."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "out.sfmr")
        recon.save(path, operation="test")
        return verify_sfmr(path)


def test_embed_patches_default_is_no_subpixel(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """The default ``embed_patches`` call (no ``subpixel=`` kwarg) is bit-for-bit
    equivalent to passing ``subpixel="none"``. Pins the default so flipping it
    in code can't slip in silently.

    Scope: this only pins the **default kwarg value** to ``"none"``. It does
    NOT pin behavioral equivalence to the pre-``subpixel``-knob pipeline — a
    regression that changed *what* ``subpixel="none"`` does would slip
    through here. Defending the broader contract would need a baseline
    artifact compared against this build's output, which this test does not
    carry.
    """
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"
    images = _load_images(recon)

    default = embed_patches(recon, images, patch_size=10.0)
    explicit_none = embed_patches(recon, images, patch_size=10.0, subpixel="none")

    assert default.point_count == explicit_none.point_count
    np.testing.assert_array_equal(
        np.asarray(default.keypoints_xy), np.asarray(explicit_none.keypoints_xy)
    )


def test_embed_patches_subpixel_lk_round_trips(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """``embed_patches(subpixel="lk")`` produces a valid ``embedded_patches``
    reconstruction that round-trips through ``.sfmr``, and its per-view
    keypoints differ from the no-refinement baseline (the refiner actually
    moved something — i.e. it ran end-to-end, not just was a no-op spliced
    back over the baseline).
    """
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)

    baseline = embed_patches(recon, images, patch_size=10.0, subpixel="none")
    refined = embed_patches(recon, images, patch_size=10.0, subpixel="lk")

    # Same membership shape (the subpixel refiner is local — it changes no
    # kept-view set), so a per-row keypoint comparison is meaningful.
    assert baseline.point_count == refined.point_count
    assert baseline.feature_source == refined.feature_source == "embedded_patches"
    base_kp = np.asarray(baseline.keypoints_xy)
    ref_kp = np.asarray(refined.keypoints_xy)
    assert base_kp.shape == ref_kp.shape
    # At least some observations must move (otherwise the splice was a no-op
    # and the wiring is broken).
    moved = np.linalg.norm(ref_kp - base_kp, axis=1) > 1e-3
    assert moved.any(), "subpixel='lk' moved zero keypoints (wiring is a no-op?)"

    # Round-trip the refined recon through .sfmr to confirm it's structurally
    # valid (the path the CLI takes).
    out = tmp_path / "refined.sfmr"
    refined.save(str(out), operation="embed-patches subpixel=lk")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.feature_source == "embedded_patches"
    assert reloaded.point_count == refined.point_count
