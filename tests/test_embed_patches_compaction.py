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
import threading
from pathlib import Path

import numpy as np

from sfmtool._embed_patches import (
    _poll_progress,
    _progress_poll_loop,
    _refine_subpixel,
    compact_to_embedded_patches,
    image_file_hashes_from_images,
)
from sfmtool._sfmtool import PatchCloud, ProgressCounter, SfmrReconstruction
from sfmtool._sfmtool.io import verify_sfmr


def _load_images(recon) -> list[np.ndarray]:
    import cv2

    ws = recon.workspace_dir
    out = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        out.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
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
    """A point at infinity (w = 0) with enough covering views produces a real
    cross-view **consensus bitmap** in the sub-pixel refiner (it is refined, not
    skipped), passes the uniform validity cull — there is no infinity exemption
    any more — and stays at infinity through compaction, carrying that bitmap
    (nonzero alpha) instead of the zero row the old pipeline stored."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
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
            "loo_zncc": np.full(len(inf_views), np.nan),
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
                "loo_zncc": np.full(len(views), np.nan),
            }
        )
    hashes = [b"\x00" * 16] * recon.image_count

    # The sub-pixel refiner fuses the consensus bitmaps + validity — the pipeline
    # source for both (points at infinity go through the same path).
    locs, bitmaps, valid = _refine_subpixel(
        cloud, recon, images, locs, sweeps=1, resolution=12, render_bitmaps=True
    )
    assert valid is not None and bool(valid[pi]), (
        "the well-observed infinity point must produce a consensus bitmap"
    )
    assert bitmaps[pi][..., 3].any(), "its consensus bitmap has real agreement"

    out = compact_to_embedded_patches(
        recon, cloud, locs, hashes, patch_bitmaps=bitmaps, valid=valid, min_views=2
    )

    # The infinity point survived, is still at infinity, and carries its nonzero
    # consensus bitmap (not the old zero row).
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
    from sfmtool._sfmtool.geometry import RigidTransform

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


def test_compact_drops_points_without_consensus_bitmap(seoul_bull_workspace: Path):
    """The validity mask is a hard cull: a point with enough kept views but no
    valid consensus bitmap (``valid[pid] == False`` — the refiner produced no
    representative) is dropped by the final compact instead of being kept with an
    all-black bitmap."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)
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


def test_embed_patches_default_is_two_rounds_one_sweep(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """The default ``embed_patches`` call (no ``subpixel=`` / ``rounds=`` kwargs) is
    bit-for-bit equivalent to passing ``subpixel=1, rounds=2``. Pins the default so
    flipping it in code can't slip in silently.

    Scope: this only pins the **default kwarg values** (one LK sweep, two rounds).
    It does NOT pin the broader behavioral contract — defending that would need a
    baseline artifact compared against this build's output, which this test does
    not carry.
    """
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"
    images = _load_images(recon)

    default = embed_patches(recon, images, patch_size=10.0)
    explicit = embed_patches(recon, images, patch_size=10.0, subpixel=1, rounds=2)

    assert default.point_count == explicit.point_count
    np.testing.assert_array_equal(
        np.asarray(default.keypoints_xy), np.asarray(explicit.keypoints_xy)
    )


def test_embed_patches_subpixel_lk_round_trips(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """``embed_patches(subpixel=1)`` produces a valid ``embedded_patches``
    reconstruction that round-trips through ``.sfmr``, and its per-view
    keypoints differ from the no-refinement baseline (``subpixel=0``) — the
    refiner actually moved something (it ran end-to-end, not a no-op splice).
    """
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)

    # Pin rounds=1 so the subpixel pass is the terminal step: at rounds=1 it
    # feeds nothing downstream, so it changes no kept-view set and a per-row
    # keypoint comparison is meaningful. (At rounds>=2 the round-1 keypoints feed
    # the next round's normal refinement + grazing drop, which can shift
    # membership — covered by test_embed_patches_multiple_rounds_round_trips.)
    baseline = embed_patches(recon, images, patch_size=10.0, subpixel=0, rounds=1)
    refined = embed_patches(recon, images, patch_size=10.0, subpixel=1, rounds=1)

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
    assert moved.any(), "subpixel=1 moved zero keypoints (wiring is a no-op?)"

    # Round-trip the refined recon through .sfmr to confirm it's structurally
    # valid (the path the CLI takes).
    out = tmp_path / "refined.sfmr"
    refined.save(str(out), operation="embed-patches subpixel=1")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.feature_source == "embedded_patches"
    assert reloaded.point_count == refined.point_count


def test_embed_patches_multiple_rounds_round_trips(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """``rounds > 1`` alternates normal- and keypoint-refinement, feeding each
    round into the next and re-pruning grazing observations. The output is a valid
    ``embedded_patches`` recon; the per-round grazing drop can only shrink the
    observation/point set, never grow it. A per-round ``progress`` callback fires
    once per round."""
    from sfmtool._embed_patches import embed_patches

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _load_images(recon)

    one = embed_patches(recon, images, patch_size=10.0, subpixel=1, rounds=1)

    lines: list[str] = []
    three = embed_patches(
        recon,
        images,
        patch_size=10.0,
        subpixel=1,
        rounds=3,
        progress=lines.append,
    )

    assert three.feature_source == "embedded_patches"
    # Per-round grazing pruning can only remove points/observations, so three
    # rounds keeps no more points than one.
    assert three.point_count <= one.point_count
    assert three.point_count > 0
    assert (
        np.asarray(three.keypoints_xy).shape[0] <= np.asarray(one.keypoints_xy).shape[0]
    )

    # One per-round summary line (the "normal Δ ..." metric line) per round.
    # Phase lines share the "round N/M:" prefix, so match the summary distinctly.
    assert sum("normal Δ" in line for line in lines) == 3

    out = tmp_path / "rounds.sfmr"
    three.save(str(out), operation="embed-patches rounds=3")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"


def test_drop_grazing_observations_skips_points_at_infinity():
    """A point at infinity has a unit *direction* in ``positions`` (not a location),
    so its view-vs-normal obliquity is meaningless and the Rust refinement leaves
    its frame alone. The grazing drop must therefore skip it entirely — keeping all
    its observations — while still pruning a genuinely grazing finite point."""
    from sfmtool._embed_patches import _drop_grazing_observations
    from sfmtool._sfmtool import PatchCloud

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


def _run_poll_once(value: int, total: int) -> list[str]:
    """Drive ``_progress_poll_loop`` through exactly one iteration reading
    ``value``, and return the lines it logged."""
    lines: list[str] = []
    stop = threading.Event()

    def read() -> int:
        stop.set()  # end the loop after this single iteration
        return value

    _progress_poll_loop(lines.append, read, total, stop, 0.0)
    return lines


def test_progress_poll_loop_reports_midpass_value():
    """A count strictly between 0 and total is reported as a done/total line."""
    assert _run_poll_once(500, 1000) == ["    500/1000 patches (50%)"]


def test_progress_poll_loop_skips_zero_and_complete():
    """A 0 count (nothing done yet) and a full count (pass ending) are both
    suppressed, so the poller never prints a redundant 0%/100% line."""
    assert _run_poll_once(0, 1000) == []
    assert _run_poll_once(1000, 1000) == []


def test_poll_progress_uninstrumented_when_no_log_or_trivial_total():
    """``_poll_progress`` yields ``None`` (so the caller passes
    ``progress=None``) when there is no log sink or no work to report."""
    with _poll_progress(None, 100) as counter:
        assert counter is None
    with _poll_progress(lambda _s: None, 0) as counter:
        assert counter is None


def test_poll_progress_yields_counter_and_emits_nothing_for_fast_body():
    """With a log sink and real work, ``_poll_progress`` yields a live
    ``ProgressCounter``; a body that finishes before the poll interval logs
    nothing (fast passes stay quiet)."""
    lines: list[str] = []
    with _poll_progress(lines.append, 100, interval=3600) as counter:
        assert isinstance(counter, ProgressCounter)
        assert counter.value == 0
    assert lines == []


def test_progress_counter_starts_at_zero_and_resets():
    counter = ProgressCounter()
    assert counter.value == 0
    counter.reset()
    assert counter.value == 0
