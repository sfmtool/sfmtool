# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the embedded_patches compaction glue (``sfmtool._embed_patches``).

Runs the patch kernels (normal refine + view selection + keypoint localization)
on a real reconstruction, compacts the result into an ``embedded_patches``
reconstruction, and round-trips it through ``.sfmr`` to confirm validity. See
``specs/core/patch/sift-to-patch-reconstruction.md`` and
``specs/formats/sfmr-file-format.md``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sfmtool._embed_patches import _localizations_from_recon, _refine_subpixel
from sfmtool._patch_compaction import (
    compact_to_embedded_patches,
    image_file_hashes_from_images,
    reference_observations_from_images,
)
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.patches import PatchCloud
from sfmtool.fileio import verify_sfmr

from .conftest import load_images, rotation_matrices


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


def test_reference_observations_from_images_names_the_first_observation_in_the_image():
    """Each point's reference is the place in its own track of its first
    observation in the named image, and -1 where it names no image or its
    track has no observation in that image."""
    # Point 0 sees images 4, 2, 7; point 1 sees 3, 5, 3 (image 3 twice); point
    # 2 sees 1, 6; point 3 sees 0.
    recon = SimpleNamespace(
        track_point_indexes=np.array([0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.uint32),
        track_image_indexes=np.array([4, 2, 7, 3, 5, 3, 1, 6, 0], dtype=np.uint32),
        observation_counts=np.array([3, 3, 2, 1], dtype=np.uint32),
    )
    reference_images = np.array([7, 3, 5, -1], dtype=np.int64)

    refs = reference_observations_from_images(recon, reference_images)

    assert refs.dtype == np.int32
    # Point 0: image 7 is its third observation. Point 1: image 3 first appears
    # at its first observation. Point 2: no observation in image 5. Point 3:
    # names no image.
    np.testing.assert_array_equal(refs, [2, 0, -1, -1])


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
    images = load_images(recon)
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


def test_compact_writes_references_naming_observations_of_the_compacted_tracks(
    seoul_bull_workspace: Path,
):
    """A localization's ``reference_image`` becomes the place, in the point's
    compacted (image-sorted) track, of its observation in that image, and
    ``-1`` where it is absent, ``None`` or not one of the kept views."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud, bitmaps, locs = _run_pipeline(recon, images)
    hashes = image_file_hashes_from_images(recon)

    # Name a kept view other than the first in input order, so the index has to
    # follow the sort; leave some points without a reference, and name an image
    # the point does not keep for others.
    expected_image: dict[int, int] = {}
    for i, loc in enumerate(locs):
        views = np.asarray(loc["views"]).tolist()
        pid = int(loc["point_index"])
        if not views:
            continue
        if i % 4 == 1:
            loc["reference_image"] = None
        elif i % 4 == 2:
            absent = [j for j in range(recon.image_count) if j not in views]
            if absent:
                loc["reference_image"] = absent[0]
        else:
            loc["reference_image"] = views[-1]
            expected_image[pid] = views[-1]

    new = compact_to_embedded_patches(
        recon, cloud, locs, hashes, patch_bitmaps=bitmaps, min_views=2
    )
    refs = np.asarray(new.reference_observations)
    counts = np.asarray(new.observation_counts)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    timg = np.asarray(new.track_image_indexes)
    assert refs.shape == (new.point_count,)
    assert np.all((refs >= -1) & (refs < counts))

    cloud_pids = {int(p) for p in cloud.point_indexes}
    survivors = sorted(
        int(loc["point_index"])
        for loc in locs
        if int(loc["point_index"]) in cloud_pids and len(np.asarray(loc["views"])) >= 2
    )
    named = 0
    for new_id, old_id in enumerate(survivors):
        if old_id in expected_image:
            assert refs[new_id] >= 0, f"point {new_id} (src {old_id})"
            assert timg[offsets[new_id] + refs[new_id]] == expected_image[old_id]
            named += 1
        else:
            assert refs[new_id] == -1, f"point {new_id} (src {old_id})"
    assert 0 < named < new.point_count

    # Compacting again without bitmaps, as the keypoint localizer does, keeps
    # each point's reference where its track still holds the image, and -1
    # where the localization dropped it.
    relocs = _localizations_from_recon(new)
    old_image = {
        p: timg[offsets[p] + refs[p]] for p in range(new.point_count) if refs[p] >= 0
    }
    dropped = set()
    for loc in relocs:
        pid = int(loc["point_index"])
        views = np.asarray(loc["views"])
        if pid % 3 == 0 and pid in old_image and len(views) > 2:
            keep = views != old_image[pid]
            loc["views"] = views[keep]
            loc["keypoints"] = np.asarray(loc["keypoints"])[keep]
            dropped.add(pid)
    again = compact_to_embedded_patches(
        new, new.patches, relocs, list(new.image_file_hashes), min_views=2
    )
    assert again.patch_bitmaps is None
    assert again.point_count == new.point_count
    refs2 = np.asarray(again.reference_observations)
    counts2 = np.asarray(again.observation_counts)
    offsets2 = np.concatenate([[0], np.cumsum(counts2)[:-1]]).astype(int)
    timg2 = np.asarray(again.track_image_indexes)
    kept = 0
    for p in range(again.point_count):
        if p in dropped or p not in old_image:
            assert refs2[p] == -1, f"point {p}"
        else:
            assert timg2[offsets2[p] + refs2[p]] == old_image[p], f"point {p}"
            kept += 1
    assert kept > 0 and dropped

    # The reference the views were aligned to wins over the stored one, so the
    # keypoints, the bitmap and the reference agree: a localization naming
    # another image of the track records that image, one naming None (a fused
    # mean, or nothing aligned) records -1, and one that does not say keeps
    # the stored reference.
    relocs = _localizations_from_recon(new)
    want: dict[int, int] = {}
    for loc in relocs:
        pid = int(loc["point_index"])
        if pid not in old_image:
            continue
        others = [int(v) for v in np.asarray(loc["views"]) if v != old_image[pid]]
        if pid % 3 == 0 and others:
            loc["reference_image"] = others[0]
            want[pid] = others[0]
        elif pid % 3 == 1:
            loc["reference_image"] = None
            want[pid] = -1
        else:
            del loc["reference_image"]
            want[pid] = int(old_image[pid])
    third = compact_to_embedded_patches(
        new, new.patches, relocs, list(new.image_file_hashes), min_views=2
    )
    refs3 = np.asarray(third.reference_observations)
    counts3 = np.asarray(third.observation_counts)
    offsets3 = np.concatenate([[0], np.cumsum(counts3)[:-1]]).astype(int)
    timg3 = np.asarray(third.track_image_indexes)
    got = {
        p: (int(timg3[offsets3[p] + refs3[p]]) if refs3[p] >= 0 else -1) for p in want
    }
    assert got == want
    assert any(w not in (-1, old_image[p]) for p, w in want.items())


def _normal_frame_angles_deg(recon) -> np.ndarray:
    """Per finite patched point, the angle (degrees) between the stored
    ``normals_xyz`` row and its patch frame's outward normal ``normalize(u × v)``
    — the coherence the ``.sfmr`` format defines for a finite patch (see
    ``specs/formats/sfmr-file-format.md``, "Per-point patch frame")."""
    cloud = recon.patches
    assert cloud is not None, "expected a patch frame"
    assert recon.has_normals, "expected stored normals"
    normals = np.asarray(recon.normals, dtype=np.float64)
    pids = np.asarray(cloud.point_indexes)
    finite = ~np.asarray(recon.point_is_at_infinity)[pids]
    frame = np.asarray(
        [np.asarray(cloud[i].normal, np.float64) for i in range(len(cloud))]
    )[finite]
    stored = normals[pids[finite]]
    stored = stored / np.linalg.norm(stored, axis=1, keepdims=True)
    dots = np.clip(np.sum(stored * frame, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def test_compact_normals_match_the_written_patch_frame(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """Regression: the compaction must store the normal of the cloud it *writes*.

    ``refine_normals`` rotates the patch cloud in place, so the source recon's
    ``normals_xyz`` array describes the pre-refinement plane. Carrying it through
    verbatim left the written file self-inconsistent (stored normal vs.
    ``normalize(u × v)`` of the stored frame). Assert the format's finite-patch
    coherence on the compacted recon, and again after a ``.sfmr`` round trip.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud, bitmaps, locs = _run_pipeline(recon, images)
    hashes = image_file_hashes_from_images(recon)

    # The refinement actually moved the frames — otherwise the assertion below
    # would pass trivially against the stale array.
    seed = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    seed_n = np.asarray(
        [np.asarray(seed[i].normal, np.float64) for i in range(len(seed))]
    )
    ref_n = np.asarray(
        [np.asarray(cloud[i].normal, np.float64) for i in range(len(cloud))]
    )
    assert np.abs(np.sum(seed_n * ref_n, axis=1) - 1.0).max() > 1e-6, (
        "fixture: refine_normals rotated nothing, so the regression is untestable"
    )

    new = compact_to_embedded_patches(
        recon, cloud, locs, hashes, patch_bitmaps=bitmaps, min_views=2
    )
    ang = _normal_frame_angles_deg(new)
    assert ang.max() < 1e-3, (
        f"stored normals disagree with the written patch frame "
        f"(max {ang.max():.3f} deg over {len(ang)} finite patched points)"
    )

    # The coherence survives the .sfmr round trip (both arrays are stored as
    # float32, so this also pins that the write path keeps them in agreement).
    out = tmp_path / "coherent.sfmr"
    new.save(str(out), operation="embed_patches")
    valid_file, errors = verify_sfmr(str(out))
    assert valid_file, f"integrity check failed: {errors}"
    assert _normal_frame_angles_deg(SfmrReconstruction.load(str(out))).max() < 1e-3


def test_compact_min_views_culls_points(seoul_bull_workspace: Path):
    """Raising min_views drops more points (and never keeps an under-supported one)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
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
    """A point at infinity (w = 0) with enough covering views produces a real
    **stored bitmap** in the sub-pixel refiner (it is refined, not
    skipped), passes the uniform validity cull — there is no infinity exemption
    any more — and stays at infinity through compaction, carrying that bitmap
    (nonzero alpha) instead of the zero row the old pipeline stored."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    # Turn one well-observed point into a point at infinity, in the direction the
    # cameras see it: from the centroid of the camera centres. (The direction
    # from the world origin depends on where the frame happens to put its
    # origin, and need not land in any image.)
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    counts = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=recon.point_count
    )
    pi = int(np.argmax(counts))
    centres = -np.einsum(
        "nji,nj->ni", rotation_matrices(recon), np.asarray(recon.translations)
    )
    ray = pos[pi, :3] - centres.mean(axis=0)
    pos[pi] = np.append(ray / np.linalg.norm(ray), 0.0)
    recon = recon.clone_with_changes(positions=pos)
    assert bool(np.asarray(recon.point_is_at_infinity)[pi])

    # A cloud that includes infinity points (fixed extent needs no .sift scales).
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=0.05
    )

    # Localizations for the infinity point plus two finite points, each kept with
    # >= min_views observations. The finite points seed at their stored SIFT
    # keypoints (real image content); the infinity point seeds at its
    # *direction's* projection in the views that actually see the direction —
    # a w = 0 point projects translation-invariantly, so the original track's
    # keypoints (of the once-finite point) are not where the direction appears.
    tpids = np.asarray(recon.track_point_indexes)
    timgs = np.asarray(recon.track_image_indexes)
    d = np.asarray(recon.positions, dtype=np.float64)[pi]
    inf_views, inf_kpts = [], []
    for v in range(recon.image_count):
        uv = _project_direction(recon, d, v, margin=40.0)
        if uv is not None:
            inf_views.append(v)
            inf_kpts.append(uv)
        if len(inf_views) == 3:
            break
    assert len(inf_views) >= 2, "fixture: the direction must be seen by >= 2 views"
    locs = [
        {
            "point_index": pi,
            "views": np.asarray(inf_views, dtype=np.uint32),
            "keypoints": np.asarray(inf_kpts, dtype=np.float64),
            "offsets_px": np.zeros(len(inf_views)),
            "zncc": np.full(len(inf_views), np.nan),
        }
    ]
    for p in [q for q in (0, 1, 2) if q != pi][:2]:
        rows = np.flatnonzero(tpids == p)[:3]
        views = timgs[rows]
        if len(views) < 2:
            continue
        kpts = _sift_keypoints_for_observations(recon, rows)
        locs.append(
            {
                "point_index": int(p),
                "views": views.astype(np.uint32),
                "keypoints": kpts,
                "offsets_px": np.zeros(len(views)),
                "zncc": np.full(len(views), np.nan),
            }
        )
    hashes = [b"\x00" * 16] * recon.image_count

    # The sub-pixel refiner renders the stored bitmaps + validity — the pipeline
    # source for both (points at infinity go through the same path).
    locs, bitmaps, valid = _refine_subpixel(
        cloud, recon, images, locs, refine=True, resolution=12, render_bitmaps=True
    )
    assert valid is not None and bool(valid[pi]), (
        "the well-observed infinity point must produce a stored bitmap"
    )
    assert bitmaps[pi][..., 3].any(), "its stored bitmap has real samples"

    out = compact_to_embedded_patches(
        recon, cloud, locs, hashes, patch_bitmaps=bitmaps, valid=valid, min_views=2
    )

    # The infinity point survived, is still at infinity, and carries its nonzero
    # stored bitmap (not the old zero row).
    is_inf = np.asarray(out.point_is_at_infinity)
    assert is_inf.sum() == 1, "the one infinity point should survive as w = 0"
    out_bitmaps = np.asarray(out.patch_bitmaps)
    inf_row = out_bitmaps[int(np.flatnonzero(is_inf)[0])]
    assert inf_row[..., 3].any(), "the surviving infinity point keeps its bitmap"
    valid_file, errors = verify_sfmr_to_temp(out)
    assert valid_file, f"integrity check failed: {errors}"


def _sift_keypoints_for_observations(recon, rows: np.ndarray) -> np.ndarray:
    """The stored SIFT keypoints for the given observation rows (source px)."""
    from sfmtool.sift.file import SiftReader, get_sift_path_from_recon

    timgs = np.asarray(recon.track_image_indexes)
    tfeats = np.asarray(recon.track_feature_indexes)
    kpts = np.empty((len(rows), 2), dtype=np.float64)
    for k, j in enumerate(rows.tolist()):
        name = recon.image_names[int(timgs[j])]
        positions = SiftReader(get_sift_path_from_recon(recon, name)).read_positions()
        kpts[k] = np.asarray(positions, dtype=np.float64)[int(tfeats[j])]
    return kpts


def _project_direction(recon, d: np.ndarray, image_idx: int, margin: float = 0.0):
    """Project a w = 0 direction into an image (translation-invariant): the pixel
    of ``R @ d``, or ``None`` when behind the camera or within ``margin`` px of
    (or beyond) the frame edge."""
    from sfmtool.geometry import RigidTransform

    q = np.asarray(recon.quaternions_wxyz, np.float64)[image_idx]
    t = np.asarray(recon.translations, np.float64)[image_idx]
    rot = np.asarray(
        RigidTransform.from_wxyz_translation(
            q.tolist(), t.tolist()
        ).to_rotation_matrix(),
        np.float64,
    )
    x = rot @ d
    # Canonical cameras look down -Z: a direction is in front when (R·d).z < 0,
    # and normalized coords divide by the depth -z (> 0 in front).
    if x[2] >= 0:
        return None
    cam = recon.cameras[int(np.asarray(recon.camera_indexes)[image_idx])]
    uv = np.asarray(cam.project(x[0] / -x[2], x[1] / -x[2]), dtype=np.float64)
    if not (
        margin <= uv[0] < cam.width - margin and margin <= uv[1] < cam.height - margin
    ):
        return None
    return uv


def test_compact_drops_points_without_a_valid_bitmap(
    seoul_bull_workspace: Path,
):
    """The validity mask is a hard cull: a point with enough kept views but no
    valid stored bitmap (``valid[pid] == False`` — the refiner produced no
    representative) is dropped by the final compact instead of being kept with an
    all-black bitmap."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud, bitmaps, locs = _run_pipeline(recon, images)
    hashes = image_file_hashes_from_images(recon)

    cloud_pids = {int(p) for p in cloud.point_indexes}
    survivors = sorted(
        int(loc["point_index"])
        for loc in locs
        if int(loc["point_index"]) in cloud_pids and len(np.asarray(loc["views"])) >= 2
    )
    victim = survivors[0]
    valid = np.ones(recon.point_count, dtype=bool)
    valid[victim] = False

    out = compact_to_embedded_patches(
        recon, cloud, locs, hashes, patch_bitmaps=bitmaps, valid=valid, min_views=2
    )

    # Exactly the victim is gone; the remaining survivors keep their geometry.
    assert out.point_count == len(survivors) - 1
    expected = np.asarray(recon.positions)[[p for p in survivors if p != victim]]
    np.testing.assert_allclose(np.asarray(out.positions), expected, atol=1e-6)


def verify_sfmr_to_temp(recon) -> tuple[bool, list]:
    """Save to a temp .sfmr and run the format integrity check."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "out.sfmr")
        recon.save(path, operation="test")
        return verify_sfmr(path)


def test_drop_grazing_observations_skips_points_at_infinity():
    """A point at infinity has a unit *direction* in ``positions`` (not a location),
    so its view-vs-normal obliquity is meaningless and the Rust refinement leaves
    its frame alone. The grazing drop must therefore skip it entirely — keeping all
    its observations — while still pruning a genuinely grazing finite point."""
    from sfmtool._embed_patches import _drop_grazing_observations
    from sfmtool.patches import PatchCloud

    # Two dense points, both with normal u x v = +z (f32 half-vectors, f64
    # centers — the dtypes from_halfvec_arrays expects, as in compaction).
    u = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    v = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    centers_pts = np.zeros((2, 3), dtype=np.float64)
    cloud = PatchCloud.from_halfvec_arrays(u, v, centers_pts)

    # One camera off to the side: its view direction is ~+x, ~90 deg off the +z
    # normal — grazing, so it would be dropped for a finite point.
    cam_centers = np.array([[10.0, 0.0, 1e-3]])
    positions = np.zeros((2, 3))
    at_infinity = np.array([False, True])
    loc = [
        {
            "point_index": 0,
            "views": np.array([0], dtype=np.uint32),
            "keypoints": np.zeros((1, 2)),
        },
        {
            "point_index": 1,
            "views": np.array([0], dtype=np.uint32),
            "keypoints": np.zeros((1, 2)),
        },
    ]

    out, dropped = _drop_grazing_observations(
        loc, cloud, cam_centers, positions, at_infinity, 80.0
    )
    by_pid = {int(o["point_index"]): o for o in out}
    # Finite grazing point: its lone view is culled.
    assert len(by_pid[0]["views"]) == 0
    # Infinity point: untouched despite the (meaningless) grazing geometry.
    assert len(by_pid[1]["views"]) == 1
    assert dropped == 1


def test_reference_observations_from_images_takes_the_first_observation():
    """A track with two observations in the reference image names the first."""
    recon = SimpleNamespace(
        # Point 0 sees images 3, 5, 5; point 1 sees 2, 2; point 2 sees 4.
        track_point_indexes=np.array([0, 0, 0, 1, 1, 2]),
        track_image_indexes=np.array([3, 5, 5, 2, 2, 4]),
        observation_counts=np.array([3, 2, 1]),
    )
    out = reference_observations_from_images(recon, np.array([5, 2, 7]))
    np.testing.assert_array_equal(out, [1, 0, -1])
    assert out.dtype == np.int32
