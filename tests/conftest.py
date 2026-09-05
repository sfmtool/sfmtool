# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from sfmtool._workspace import init_workspace

if TYPE_CHECKING:
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"


@pytest.fixture
def sample_sift_data():
    """Sample SIFT data (metadata and numpy arrays) for roundtrip tests."""
    feature_count = 100
    feature_tool_metadata = {
        "feature_tool": "pytest",
        "feature_type": "sift",
        "feature_options": {},
    }
    metadata = {
        "version": 1,
        "image_name": "test.jpg",
        "image_file_xxh128": "a" * 32,
        "image_file_size": 12345,
        "image_width": 1920,
        "image_height": 1080,
        "feature_count": feature_count,
    }
    rng = np.random.default_rng(seed=42)
    position = rng.random((feature_count, 2), dtype=np.float32) * np.array(
        [1920, 1080], dtype=np.float32
    )
    affine_shape = rng.random((feature_count, 2, 2), dtype=np.float32) - 0.5
    descriptor = rng.integers(0, 255, (feature_count, 128), dtype=np.uint8)
    thumbnail = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)

    return (
        feature_tool_metadata,
        metadata,
        position,
        affine_shape,
        descriptor,
        thumbnail,
    )


@pytest.fixture
def isolated_seoul_bull_image(tmp_path_factory) -> Path:
    """Fixture that provides a .jpg file isolated in a directory for testing."""
    input_img_path = (
        TEST_DATA_DIR
        / "images"
        / "seoul_bull_sculpture"
        / "seoul_bull_sculpture_01.jpg"
    )
    tmp_path = tmp_path_factory.mktemp("test_image")
    img_path = tmp_path / "test_image.jpg"
    shutil.copy(input_img_path, img_path)
    return img_path


@pytest.fixture
def isolated_seoul_bull_17_images(tmp_path_factory) -> list[Path]:
    """Fixture that provides 17 .jpg files isolated in a directory for testing.

    Also copies the dataset's `camera_config.json` so any solve/match commands
    pick up the calibrated intrinsics committed alongside the images.
    """
    data_dir = TEST_DATA_DIR / "images" / "seoul_bull_sculpture"
    image_files = sorted(data_dir.glob("seoul_bull_sculpture_*.jpg"))
    assert len(image_files) == 17
    tmp_path = tmp_path_factory.mktemp("test_17_images")

    img_paths = []
    for img_file in image_files:
        dest_path = tmp_path / img_file.name
        shutil.copy(img_file, dest_path)
        img_paths.append(dest_path)

    shutil.copy(data_dir / "camera_config.json", tmp_path / "camera_config.json")

    return img_paths


def _largest_recon(output_sfm_file: Path):
    """Return ``(path, image_count)`` for the biggest recon a solve wrote.

    A solve can split into several sub-reconstructions; ``run_*_sfm`` writes the
    first to ``output_sfm_file`` and the rest to ``{stem}-N.sfmr`` siblings, and
    returns only the first — which is not always the most complete one. Pick the
    one that registered the most images.
    """
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    candidates = sorted(output_sfm_file.parent.glob(f"{output_sfm_file.stem}*.sfmr"))
    best_path, best_count = None, -1
    for path in candidates:
        count = SfmrReconstruction.load(path).image_count
        if count > best_count:
            best_path, best_count = path, count
    return best_path, best_count


def _rotate_by_quaternion(
    quat_wxyz: np.ndarray, vectors: np.ndarray, *, inverse: bool = False
) -> np.ndarray:
    """Rotate each row of ``vectors`` by the parallel unit wxyz quaternion.

    Uses the optimized unit-quaternion form ``v' = v + 2w(u×v) + 2u×(u×v)``,
    which needs no 3x3 matrix. ``inverse=True`` applies the conjugate ``(w, -u)``
    — the camera-to-world rotation, given the world-to-camera quaternions a
    reconstruction stores.
    """
    w = quat_wxyz[:, :1]
    u = -quat_wxyz[:, 1:] if inverse else quat_wxyz[:, 1:]
    t = 2.0 * np.cross(u, vectors)
    return vectors + w * t + np.cross(u, t)


def _drop_camera_coincident_points(sfmr_path: Path) -> None:
    """Drop finite points that triangulated onto their observing camera centres.

    A near-zero-baseline two-view track can collapse onto the cameras, leaving a
    point whose ray distance ``d = ‖X − C‖`` is ~0 in *every* view. Such a point
    is a triangulation artifact with no surface element:
    ``PatchExtent::FeatureSize`` sizes a patch as ``σ·d/f`` and skips
    observations with ``d ≤ 1e-6``, so a point degenerate in all views cannot be
    sized and ``PatchCloud.from_reconstruction`` errors. GLOMAP emits such a
    point only occasionally, which flakes the fisheye patch-cloud tests; dropping
    it here keeps every fixture reconstruction clean. A no-op (no resave) for the
    usual case where no point is camera-coincident.
    """
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    recon = SfmrReconstruction.load(sfmr_path)
    pos = np.asarray(recon.positions)
    if len(pos) == 0:
        return
    quat = np.asarray(recon.quaternions_wxyz)
    trans = np.asarray(recon.translations)
    tii = np.asarray(recon.track_image_indexes)
    tpid = np.asarray(recon.track_point_indexes)
    at_inf = np.asarray(recon.point_is_at_infinity)

    # Per-observation ray distance d = ‖R(q)·X + t‖ in the camera frame.
    cam_pt = _rotate_by_quaternion(quat[tii], pos[tpid]) + trans[tii]
    d = np.linalg.norm(cam_pt, axis=1)

    # Keep a point if any observation is non-degenerate (matches the FeatureSize
    # d > 1e-6 gate). Points at infinity have no ray distance and are always kept.
    max_d = np.full(len(pos), -np.inf)
    np.maximum.at(max_d, tpid, d)
    keep = at_inf | (max_d > 1e-6)
    if keep.all():
        return
    filtered = recon.filter_points_by_mask(keep)
    filtered.save(sfmr_path, "drop-camera-coincident-points")


def build_cluster_reconstruction(
    workspace_dir: Path,
    image_paths: list[Path],
    output_sfm_file: Path,
    *,
    max_num_features: int | None = None,
    cluster_d: int = 10,
    incremental: bool = True,
    random_seed: int = 42,
    expected_image_count: int | None = None,
    min_point_count: int = 0,
    max_attempts: int = 6,
    accept: Callable[["SfmrReconstruction"], str | None] | None = None,
) -> Path:
    """Solve a ``.sfmr`` the way the dataset scripts now do.

    Mirrors ``scripts/init_dataset_*.sh``: initialize the workspace with the
    sfmtool SIFT backend, run background-floor track-cluster matching
    (``sfm match --cluster``) to a clusters ``.matches`` file, derive the
    verified pairwise+TVG file the mapper reads (``sfm match --derive-pairs``),
    then solve from that. Matching runs once; only the (cheap) solve is
    retried. Each solve can split
    into several sub-reconstructions, so the most complete one is selected and
    canonicalized to ``output_sfm_file``. When ``expected_image_count`` and/or
    ``min_point_count`` are set the solve is re-run (with fresh randomization)
    until that many images register *and* that many points triangulate, keeping
    the best (most images, then most points) attempt seen. A complete-but-sparse
    solve — all images registered but few points — is a degenerate result, so
    ``min_point_count`` lets a caller insist on a substantive point cloud rather
    than accepting the first attempt that merely registers every image.

    ``accept`` is the general form of that insistence, for guarantees no scalar
    floor can express. It is called with the attempt's chosen reconstruction once
    the image and point checks pass; returning a string rejects the attempt (the
    string is the reason, and the loop re-randomizes), returning ``None`` accepts
    it and stops. Attempts are still ranked as above, so a rejected attempt can
    still be the best one seen — but a caller that asks for a guarantee gets it or
    a ``RuntimeError``: if no attempt is ever accepted the fixture fails naming the
    last rejection reason, which beats handing the tests a reconstruction that
    quietly lacks the property they assert.
    """
    from sfmtool.feature_match._derive_pairs import _run_derive_pairs
    from sfmtool.feature_match._run import _run_matching

    init_workspace(
        workspace_dir, feature_tool="sfmtool", max_num_features=max_num_features
    )

    clusters_file = workspace_dir / "matches" / "recon-clusters.matches"
    # _run_matching extracts any missing .sift files before matching.
    _run_matching(
        [Path(p) for p in image_paths],
        workspace_dir,
        matching_method="cluster",
        max_feature_count=None,
        output_path=str(clusters_file),
        camera_model=None,
        cluster_d=cluster_d,
    )

    matches_dir = workspace_dir / "tvg-matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    matches_file = matches_dir / "recon.matches"
    _run_derive_pairs(clusters_file, output_path=str(matches_file))

    colmap_dir = workspace_dir / "colmap"
    if incremental:
        from sfmtool._incremental_sfm import run_incremental_sfm as _solve
    else:
        from sfmtool._global_sfm import run_global_sfm as _solve

    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    output_sfm_file = Path(output_sfm_file)
    # Rank attempts by (image_count, point_count): prefer a fully registered
    # reconstruction, and among those the densest. This keeps retrying past a
    # complete-but-sparse solve until a substantive one shows up.
    best_path, best_key = None, (-1, -1)
    accepted = accept is None
    last_rejection = "no attempt met the image / point-count floors"
    for attempt in range(1, max_attempts + 1):
        if colmap_dir.exists():
            shutil.rmtree(colmap_dir)
        for stale in output_sfm_file.parent.glob(f"{output_sfm_file.stem}*.sfmr"):
            stale.unlink()
        # First attempt uses the fixed seed for a reproducible result; retries
        # let the solver randomize so a fresh split can register all images.
        seed = random_seed if attempt == 1 else None
        try:
            _solve(
                [],
                workspace_dir,
                colmap_dir,
                matches_file=matches_file,
                random_seed=seed,
                output_sfm_file=str(output_sfm_file),
            )
        except RuntimeError:
            # A degenerate solve -- GLOMAP is not seed-deterministic, and the
            # extreme case raises "No 3D points found" -- is just another
            # attempt to rank below the rest, not a fixture failure. Retry with
            # fresh randomization, as the ``.camrig`` fixture below does.
            continue
        path, count = _largest_recon(output_sfm_file)
        recon = SfmrReconstruction.load(path)
        points = recon.point_count
        key = (count, points)
        if key > best_key:
            best_key = key
            # Stash the best so far; the next attempt clears the output dir.
            stash = output_sfm_file.parent / f"_best{output_sfm_file.suffix}"
            shutil.copy(path, stash)
            best_path = stash
        images_ok = expected_image_count is None or count >= expected_image_count
        points_ok = points >= min_point_count
        if images_ok and points_ok:
            if accept is None:
                break
            reason = accept(recon)
            if reason is None:
                accepted = True
                break
            last_rejection = reason

    if best_path is None:
        raise RuntimeError(
            f"every one of {max_attempts} solve attempts on {workspace_dir} was "
            "degenerate (the solver raised each time); no reconstruction to keep."
        )
    if not accepted:
        raise RuntimeError(
            f"none of {max_attempts} solve attempts on {workspace_dir} was accepted; "
            f"last rejection: {last_rejection}."
        )
    # Canonicalize: the chosen reconstruction lives at output_sfm_file alone.
    for stale in output_sfm_file.parent.glob(f"{output_sfm_file.stem}*.sfmr"):
        stale.unlink()
    shutil.copy(best_path, output_sfm_file)
    best_path.unlink()
    # Strip the occasional degenerate point that collapsed onto its cameras, so
    # FeatureSize patch sizing (and any other ray-distance consumer) is robust.
    _drop_camera_coincident_points(output_sfm_file)
    return output_sfm_file


@pytest.fixture(scope="session")
def seoul_bull_workspace_once(tmp_path_factory) -> Path:
    """Session-scoped fixture: build a .sfmr reconstruction from 17 images.

    Mirrors ``scripts/init_dataset_seoul_bull.sh``: sfmtool SIFT + track-cluster
    matching + incremental SfM. The fixture carries calibrated intrinsics and
    keeps the most complete sub-reconstruction, so the cluster matcher's default
    floor registers all 17 of these small 270x480 images without the wide
    ``d=28`` (and the resulting tracks stay longer).
    """
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    data_dir = TEST_DATA_DIR / "images" / "seoul_bull_sculpture"
    image_files = sorted(data_dir.glob("seoul_bull_sculpture_*.jpg"))
    workspace_dir = tmp_path_factory.mktemp("workspace_17_images")
    image_dir = workspace_dir / "test_17_image"
    image_dir.mkdir(exist_ok=True)

    img_paths = []
    for img_file in image_files:
        dest_path = image_dir / img_file.name
        shutil.copy(img_file, dest_path)
        img_paths.append(dest_path)

    # Place camera_config.json at the workspace root so tests that copy just
    # the image directory (e.g. test_cam_cp_roundtrip_into_solve) start with
    # an unconfigured workspace; the closest-ancestor resolver still finds it
    # for solves that run on the original workspace.
    shutil.copy(data_dir / "camera_config.json", workspace_dir / "camera_config.json")

    expected_image_count = len(image_files)
    output_sfm_file = workspace_dir / "seoul_bull.sfmr"
    sfmr_path = build_cluster_reconstruction(
        workspace_dir,
        img_paths,
        output_sfm_file,
        incremental=True,
        random_seed=42,
        expected_image_count=expected_image_count,
    )

    recon = SfmrReconstruction.load(sfmr_path)
    if recon.image_count != expected_image_count:
        raise RuntimeError(
            f"seoul_bull cluster solve registered {recon.image_count}/"
            f"{expected_image_count} images (all {expected_image_count} required)."
        )
    return sfmr_path


@pytest.fixture
def seoul_bull_workspace(seoul_bull_workspace_once: Path, tmp_path_factory) -> Path:
    """Per-test isolation of the 17-image .sfmr reconstruction."""
    source_workspace_dir = seoul_bull_workspace_once.parent
    workspace_dir = tmp_path_factory.mktemp("workspace_17_images")
    shutil.copytree(source_workspace_dir, workspace_dir, dirs_exist_ok=True)
    return workspace_dir / seoul_bull_workspace_once.name


@pytest.fixture
def seoul_bull_sfmr_only(seoul_bull_workspace_once: Path, tmp_path_factory) -> Path:
    """Per-test copy of *only* the 17-image ``.sfmr`` (plus the workspace marker).

    For tests that just ``SfmrReconstruction.load`` the reconstruction and read
    its geometry (or apply geometry-only transforms / alignment), copying the
    whole solved workspace — 17 images, every ``.sift`` file, the COLMAP db and
    the match cache — is wasted I/O that dominates the suite's file-copy time.
    This copies the single ``.sfmr`` plus the ``.sfm-workspace.json`` marker into
    an isolated tmp dir, so the reconstruction resolves its workspace to *that*
    dir (not the shared session workspace) and any source-image / ``.sift``
    access fails loudly. Tests that need the source images or ``.sift`` files must
    use the full :func:`seoul_bull_workspace` instead.
    """
    src = seoul_bull_workspace_once
    workspace_dir = tmp_path_factory.mktemp("sfmr_only_17_images")
    shutil.copy(src, workspace_dir / src.name)
    marker = src.parent / ".sfm-workspace.json"
    if marker.exists():
        shutil.copy(marker, workspace_dir / marker.name)
    return workspace_dir / src.name


KERRY_PARK_DIR = TEST_DATA_DIR / "images" / "kerry_park"
KERRY_PARK_FRAME_COUNT = 24
KERRY_PARK_SENSORS = ("fisheye_left", "fisheye_right")
# The solve fixtures don't need all 24 frames. The kerry_park capture is from a
# video, so a contiguous prefix preserves the frame-to-frame
# covisibility chain (adjacent same-sensor frames share ~28 points on average,
# decaying past a gap of ~3) while the two back-to-back fisheyes stay tied
# together by their cross-sensor-at-different-frames overlap. An 8-frame prefix
# (16 images) still solves complete and well-conditioned (all images
# registered, both cameras, ~300 points, sub-pixel error) at a fraction of the
# matching/solve cost. Disk-parsing/resolution fixtures still see all 24 frames.
KERRY_PARK_SOLVE_FRAME_COUNT = 8

# GLOMAP is not seed-deterministic, so "16 images and >= 200 points" does not pin
# down *which* reconstruction the session gets, and three patch tests assert a
# property of it that a legitimate solve can lack. Each floor below is a
# guarantee the fixture holds out for, so those tests measure the algorithm
# rather than the luck of the solve; the counts behind them are read off the
# reconstruction's own arrays, with no patch cloud and no images.
#
# Multi-view points at infinity, for
# test_patch_view_selection.py::test_select_views_infinity_admitted_are_in_front,
# which selects views for *every* infinity point in the cloud. A handful is
# enough for the test to have something to check, and a solve that yields none at
# all is the degenerate case worth re-rolling. (Ten sample solves of this fixture
# gave 5 to 14.)
MIN_INFINITY_POINTS = 5
# Points the rig can see past 90 deg off axis, for
# test_patch_view_selection.py::test_select_views_admitted_points_are_in_front_of_camera,
# which needs at least one *admitted* view out there. See
# :func:`points_with_past_90_candidate` for why this counts candidates rather
# than observations. (Ten sample solves gave 115 to 259; the floor only has to
# leave the test's 150-point sample a real pool to draw from.)
MIN_PAST_90_CANDIDATE_POINTS = 40
# Points whose track spans a real range of viewing angles, for
# test_patch_keypoint_localization.py::test_localize_keypoints_grazing_cutoff_drops_views:
# a strict min_grazing_cos can only drop a view that is oblique to the patch
# normal, and on an all-narrow-baseline solve there is none to drop. These are
# scarce -- a survey of 20 builds gave 7 to 14 of ~300 points -- which is exactly
# why that test cannot sample the cloud at large. One of those 20 solves had a
# single oblique point, and that is the one this floor sends back.
MIN_OBLIQUE_POINTS = 5
# "Oblique" = an observation ray more than this far from the point's mean viewing
# direction. min_grazing_cos = 0.99 in the grazing test cuts at ~8.1 deg, so 10
# deg leaves the guarantee strictly inside what that cutoff drops.
OBLIQUE_ANGLE_DEG = 10.0


def points_with_past_90_candidate(recon) -> set[int]:
    """Finite points that some image sees past 90 deg off its optical axis.

    A *candidate* (point, image) pair, not an observation: no track observation
    of the kerry_park solve sits past 90 deg -- the matcher finds nothing that far
    into the fisheye periphery -- yet view selection admits views beyond the
    track, and those are what reach it. So the predicate asks, of every point and
    every image: is the point behind the 90 deg plane (canonical cameras look down
    -Z, so camera-frame ``z >= 0``) and does the camera model still project it
    inside the frame? Points at infinity are excluded; the infinity form of that
    visibility is its own test.
    """
    pos = np.asarray(recon.positions, dtype=np.float64)
    quat = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    trans = np.asarray(recon.translations, dtype=np.float64)
    cameras = recon.cameras
    camera_index = np.asarray(recon.camera_indexes)
    finite_ids = np.nonzero(~np.asarray(recon.point_is_at_infinity))[0]
    image_count = recon.image_count
    if len(finite_ids) == 0 or image_count == 0:
        return set()

    # Every (point, image) pair at once: x_cam = R·X + t.
    point_of = np.repeat(finite_ids, image_count)
    image_of = np.tile(np.arange(image_count), len(finite_ids))
    x_cam = _rotate_by_quaternion(quat[image_of], pos[point_of]) + trans[image_of]

    out: set[int] = set()
    for pair in np.nonzero(x_cam[:, 2] >= 0.0)[0]:
        pid = int(point_of[pair])
        if pid in out:
            continue
        cam = cameras[int(camera_index[image_of[pair]])]
        px = cam.ray_to_pixel(x_cam[pair].tolist())
        if px is not None and 0.0 <= px[0] < cam.width and 0.0 <= px[1] < cam.height:
            out.add(pid)
    return out


def points_with_oblique_view(
    recon, min_angle_deg: float = OBLIQUE_ANGLE_DEG
) -> set[int]:
    """Finite points observed from more than ``min_angle_deg`` off their mean ray.

    Unit rays from the camera centres to the point, their normalized sum as the
    point's mean viewing direction (what ``normal="mean_viewing"`` hands the
    patch), and the widest angle any ray of the track makes with it. Only such a
    point owns a view that a grazing-angle cutoff can drop.
    """
    pos = np.asarray(recon.positions, dtype=np.float64)
    quat = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    trans = np.asarray(recon.translations, dtype=np.float64)
    at_infinity = np.asarray(recon.point_is_at_infinity)
    obs_image = np.asarray(recon.track_image_indexes)
    obs_point = np.asarray(recon.track_point_indexes)
    if len(pos) == 0 or len(obs_point) == 0:
        return set()

    # A point at infinity has a direction in the camera frame, not a position.
    finite = ~at_infinity[obs_point]
    image_idx, point_idx = obs_image[finite], obs_point[finite]
    x_cam = _rotate_by_quaternion(quat[image_idx], pos[point_idx]) + trans[image_idx]
    distance = np.linalg.norm(x_cam, axis=1)
    usable = distance > 1e-12
    image_idx, point_idx = image_idx[usable], point_idx[usable]
    # World-space unit ray from the camera centre to the point: X - C = Rᵀ·x_cam.
    ray = _rotate_by_quaternion(
        quat[image_idx], x_cam[usable] / distance[usable, None], inverse=True
    )

    mean_dir = np.zeros_like(pos)
    np.add.at(mean_dir, point_idx, ray)
    norm = np.linalg.norm(mean_dir, axis=1)
    has_mean = norm > 1e-12
    mean_dir[has_mean] /= norm[has_mean, None]

    cos = np.einsum("ij,ij->i", ray, mean_dir[point_idx])
    oblique = has_mean[point_idx] & (cos < np.cos(np.radians(min_angle_deg)))
    return {int(p) for p in point_idx[oblique]}


def _kerry_park_reject_reason(recon) -> str | None:
    """Why this kerry_park solve is unfit for the patch tests, or ``None`` if it is.

    The ``accept`` hook of :func:`build_cluster_reconstruction`, holding the
    reconstruction to the three ``MIN_*`` guarantees above. Each check is a count
    over the reconstruction's own arrays, so the whole hook costs a fraction of
    the solve attempt it vets.
    """
    at_infinity = np.asarray(recon.point_is_at_infinity)
    observed = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=len(at_infinity)
    )
    # Points at infinity that carry a real (multi-view) track.
    infinity_points = int(np.count_nonzero(at_infinity & (observed >= 2)))
    if infinity_points < MIN_INFINITY_POINTS:
        return (
            f"points at infinity with a multi-view track: {infinity_points} "
            f"(>= {MIN_INFINITY_POINTS} required)"
        )
    past_90 = len(points_with_past_90_candidate(recon))
    if past_90 < MIN_PAST_90_CANDIDATE_POINTS:
        return (
            f"points visible past 90 deg off a camera's axis: {past_90} "
            f"(>= {MIN_PAST_90_CANDIDATE_POINTS} required)"
        )
    oblique = len(points_with_oblique_view(recon))
    if oblique < MIN_OBLIQUE_POINTS:
        return (
            f"points observed more than {OBLIQUE_ANGLE_DEG:g} deg off their mean "
            f"viewing direction: {oblique} (>= {MIN_OBLIQUE_POINTS} required)"
        )
    return None


def _copy_kerry_park_into(workspace_dir: Path) -> None:
    """Copy the kerry_park rig images + rig_config.json into ``workspace_dir``.

    Preserves the ``fisheye_left/`` / ``fisheye_right/`` subdirectory layout
    so the rig_config.json ``image_prefix`` entries resolve correctly.
    """
    for sensor in KERRY_PARK_SENSORS:
        src_dir = KERRY_PARK_DIR / sensor
        dst_dir = workspace_dir / sensor
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(src_dir.glob("frame_*.jpg")):
            shutil.copy(img, dst_dir / img.name)
    shutil.copy(KERRY_PARK_DIR / "rig_config.json", workspace_dir / "rig_config.json")


def _copy_kerry_park_camrig_into(workspace_dir: Path) -> None:
    """Copy the kerry_park rig images + ``kerry_park.camrig`` into ``workspace_dir``.

    The same back-to-back fisheye rig as :func:`_copy_kerry_park_into`, but
    described by a multi-sensor ``.camrig`` file rather than ``rig_config.json``
    — the layout `sfm insv2rig` now produces.
    """
    for sensor in KERRY_PARK_SENSORS:
        src_dir = KERRY_PARK_DIR / sensor
        dst_dir = workspace_dir / sensor
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(src_dir.glob("frame_*.jpg")):
            shutil.copy(img, dst_dir / img.name)
    shutil.copy(
        KERRY_PARK_DIR / "kerry_park.camrig", workspace_dir / "kerry_park.camrig"
    )


@pytest.fixture
def isolated_kerry_park_rig(tmp_path_factory) -> Path:
    """Function-scoped: all 48 kerry_park rig images + rig_config.json in a tmp dir.

    Yields the workspace directory. Layout under it::

        <workspace>/
          rig_config.json
          fisheye_left/frame_01.jpg ... frame_24.jpg
          fisheye_right/frame_01.jpg ... frame_24.jpg
    """
    workspace_dir = tmp_path_factory.mktemp("kerry_park_rig")
    _copy_kerry_park_into(workspace_dir)
    return workspace_dir


@pytest.fixture
def isolated_kerry_park_camrig(tmp_path_factory) -> Path:
    """Function-scoped: all 48 kerry_park rig images + ``kerry_park.camrig``.

    Yields the workspace directory. Layout under it::

        <workspace>/
          kerry_park.camrig
          fisheye_left/frame_01.jpg ... frame_24.jpg
          fisheye_right/frame_01.jpg ... frame_24.jpg
    """
    workspace_dir = tmp_path_factory.mktemp("kerry_park_camrig")
    _copy_kerry_park_camrig_into(workspace_dir)
    return workspace_dir


@pytest.fixture(scope="session")
def kerry_park_workspace_once(tmp_path_factory) -> Path:
    """Session-scoped: build a .sfmr reconstruction from the kerry_park rig.

    Mirrors ``scripts/init_dataset_kerry_park.sh``: sfmtool SIFT + track-cluster
    matching + global SfM (GLOMAP) with a fixed seed. Solves an 8-frame prefix
    of the dataset (``KERRY_PARK_SOLVE_FRAME_COUNT`` × 2 sensors = 16 images); the
    solver reliably registers all of them. The fixture fails fast if it doesn't,
    rather than handing a partial reconstruction to the tests.
    """
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    workspace_dir = tmp_path_factory.mktemp("kerry_park_sfmr")
    _copy_kerry_park_into(workspace_dir)

    image_paths: list[Path] = []
    for sensor in KERRY_PARK_SENSORS:
        frames = sorted((workspace_dir / sensor).glob("frame_*.jpg"))
        image_paths.extend(frames[:KERRY_PARK_SOLVE_FRAME_COUNT])

    expected_count = len(KERRY_PARK_SENSORS) * KERRY_PARK_SOLVE_FRAME_COUNT
    output_sfm_file = workspace_dir / "kerry_park.sfmr"
    # The 8-frame back-to-back fisheye solve is sparse and non-deterministic: a
    # single attempt can register all 16 images yet triangulate very few points
    # (CI has seen ~80). Insist on a substantive point cloud (well above the
    # test's >= 150 floor, leaving margin for the trailing camera-coincident
    # point drop) and retry hard for it, keeping the densest complete attempt.
    # ``accept`` adds the structural guarantees the patch tests assert -- points
    # at infinity, past-90-deg observations, obliquely-viewed points -- which no
    # point count implies.
    sfmr_path = build_cluster_reconstruction(
        workspace_dir,
        image_paths,
        output_sfm_file,
        incremental=False,
        random_seed=42,
        expected_image_count=expected_count,
        min_point_count=200,
        max_attempts=10,
        accept=_kerry_park_reject_reason,
    )

    recon = SfmrReconstruction.load(sfmr_path)
    if recon.image_count != expected_count:
        raise RuntimeError(
            f"kerry_park global solve registered {recon.image_count}/"
            f"{expected_count} images (all {expected_count} required)."
        )
    if recon.point_count < 150:
        raise RuntimeError(
            f"kerry_park global solve triangulated only {recon.point_count} "
            f"points after {10} attempts (>= 150 required); the back-to-back "
            f"fisheye geometry produced a degenerate, near-empty reconstruction."
        )
    return sfmr_path


@pytest.fixture
def kerry_park_workspace(kerry_park_workspace_once: Path, tmp_path_factory) -> Path:
    """Per-test isolation of the kerry_park .sfmr reconstruction."""
    source_workspace_dir = kerry_park_workspace_once.parent
    workspace_dir = tmp_path_factory.mktemp("kerry_park_sfmr")
    shutil.copytree(source_workspace_dir, workspace_dir, dirs_exist_ok=True)
    return workspace_dir / kerry_park_workspace_once.name


@pytest.fixture(scope="session")
def kerry_park_camrig_workspace_once(tmp_path_factory) -> Path:
    """Session-scoped: build a .sfmr reconstruction from the kerry_park rig,
    with the rig described by a multi-sensor ``kerry_park.camrig``.

    Unlike :func:`kerry_park_workspace_once`, this fixture solves
    straight from the images through the ``_setup_for_sfm`` rig-aware path
    (``run_global_sfm(matching_mode="cluster")``), which sets up the multi-sensor
    ``.camrig`` and then runs the background-floor cluster matcher with the same
    same-frame exclusion the exhaustive path uses. The back-to-back fisheye
    geometry makes same-frame matches spurious; dropping those same-frame pairs is
    what lets the faster cluster matcher replace exhaustive here without
    degenerating the solve, while retaining coverage of the from-images rig-aware
    solve path.
    """
    from sfmtool._global_sfm import run_global_sfm
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    workspace_dir = tmp_path_factory.mktemp("kerry_park_camrig_sfmr")
    _copy_kerry_park_camrig_into(workspace_dir)
    init_workspace(workspace_dir, feature_tool="sfmtool", max_num_features=2000)

    image_paths: list[Path] = []
    for sensor in KERRY_PARK_SENSORS:
        frames = sorted((workspace_dir / sensor).glob("frame_*.jpg"))
        image_paths.extend(frames[:KERRY_PARK_SOLVE_FRAME_COUNT])

    output_sfm_file = workspace_dir / "kerry_park.sfmr"
    colmap_dir = workspace_dir / "colmap"
    expected_count = len(KERRY_PARK_SENSORS) * KERRY_PARK_SOLVE_FRAME_COUNT

    # GLOMAP is non-deterministic, and the back-to-back fisheye geometry
    # occasionally yields a degenerate solve — all frames register but few/no
    # points triangulate (``run_global_sfm`` raises "No 3D points found" in the
    # extreme). Retry with a fresh randomization (mirroring
    # ``build_cluster_reconstruction``), keeping the densest complete result and
    # holding out for a substantive point cloud, rather than flaking the suite.
    # The first attempt stays reproducible (seed 42); retries randomize.
    max_attempts = 10
    best_stash = output_sfm_file.with_name("_best_camrig.sfmr")
    best_points = -1
    for attempt in range(1, max_attempts + 1):
        if colmap_dir.exists():
            shutil.rmtree(colmap_dir)
        for stale in output_sfm_file.parent.glob(f"{output_sfm_file.stem}*.sfmr"):
            stale.unlink()
        seed = 42 if attempt == 1 else None
        try:
            sfmr_path = run_global_sfm(
                image_paths,
                workspace_dir,
                colmap_dir,
                output_sfm_file=str(output_sfm_file),
                random_seed=seed,
                matching_mode="cluster",
            )
        except RuntimeError:
            # Degenerate solve (e.g. "No 3D points found"); re-randomize.
            continue
        recon = SfmrReconstruction.load(sfmr_path)
        if recon.image_count == expected_count and recon.point_count > best_points:
            best_points = recon.point_count
            shutil.copy(sfmr_path, best_stash)
        if recon.image_count == expected_count and recon.point_count >= 200:
            break

    if best_points < 0:
        raise RuntimeError(
            f"kerry_park .camrig global solve produced no complete reconstruction "
            f"in {max_attempts} attempts (all {expected_count} images required)."
        )
    if best_points < 150:
        raise RuntimeError(
            f"kerry_park .camrig global solve triangulated only {best_points} "
            f"points in {max_attempts} attempts (>= 150 required); the back-to-back "
            f"fisheye geometry produced a degenerate, near-empty reconstruction."
        )
    for stale in output_sfm_file.parent.glob(f"{output_sfm_file.stem}*.sfmr"):
        stale.unlink()
    shutil.copy(best_stash, output_sfm_file)
    best_stash.unlink()
    return output_sfm_file


@pytest.fixture
def kerry_park_camrig_workspace(
    kerry_park_camrig_workspace_once: Path, tmp_path_factory
) -> Path:
    """Per-test isolation of the kerry_park ``.camrig`` .sfmr reconstruction."""
    source_workspace_dir = kerry_park_camrig_workspace_once.parent
    workspace_dir = tmp_path_factory.mktemp("kerry_park_camrig_sfmr")
    shutil.copytree(source_workspace_dir, workspace_dir, dirs_exist_ok=True)
    return workspace_dir / kerry_park_camrig_workspace_once.name
