# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import numpy as np
import pytest

from sfmtool._workspace import init_workspace

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
    from sfmtool._sfmtool import SfmrReconstruction

    candidates = sorted(output_sfm_file.parent.glob(f"{output_sfm_file.stem}*.sfmr"))
    best_path, best_count = None, -1
    for path in candidates:
        count = SfmrReconstruction.load(path).image_count
        if count > best_count:
            best_path, best_count = path, count
    return best_path, best_count


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
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(sfmr_path)
    pos = np.asarray(recon.positions)
    if len(pos) == 0:
        return
    quat = np.asarray(recon.quaternions_wxyz)
    trans = np.asarray(recon.translations)
    tii = np.asarray(recon.track_image_indexes)
    tpid = np.asarray(recon.track_point_ids)
    at_inf = np.asarray(recon.point_is_at_infinity)

    # Per-observation ray distance d = ‖R(q)·X + t‖ in the camera frame, using
    # the optimized unit-quaternion rotation v' = v + 2w(u×v) + 2u×(u×v).
    x = pos[tpid]
    q = quat[tii]
    w = q[:, :1]
    u = q[:, 1:]
    t = 2.0 * np.cross(u, x)
    cam_pt = x + w * t + np.cross(u, t) + trans[tii]
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
    max_attempts: int = 6,
) -> Path:
    """Solve a ``.sfmr`` the way the dataset scripts now do.

    Mirrors ``scripts/init_dataset_*.sh``: initialize the workspace with the
    sfmtool SIFT backend, run background-floor track-cluster matching
    (``sfm match --cluster``) to a ``.matches`` file, then solve from that file.
    Matching runs once; only the (cheap) solve is retried. Each solve can split
    into several sub-reconstructions, so the most complete one is selected and
    canonicalized to ``output_sfm_file``. When ``expected_image_count`` is set
    the solve is re-run (with fresh randomization) until that many images
    register, which is far faster and more reliable than the old
    extract-then-exhaustive matching it replaces.
    """
    from sfmtool.feature_match._run import _run_matching

    init_workspace(
        workspace_dir, feature_tool="sfmtool", max_num_features=max_num_features
    )

    matches_dir = workspace_dir / "tvg-matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    matches_file = matches_dir / "recon.matches"
    # _run_matching extracts any missing .sift files before matching.
    _run_matching(
        [Path(p) for p in image_paths],
        workspace_dir,
        matching_method="cluster",
        max_feature_count=None,
        output_path=str(matches_file),
        camera_model=None,
        cluster_d=cluster_d,
    )

    colmap_dir = workspace_dir / "colmap"
    if incremental:
        from sfmtool._incremental_sfm import run_incremental_sfm as _solve
    else:
        from sfmtool._global_sfm import run_global_sfm as _solve

    output_sfm_file = Path(output_sfm_file)
    best_path, best_count = None, -1
    for attempt in range(1, max_attempts + 1):
        if colmap_dir.exists():
            shutil.rmtree(colmap_dir)
        for stale in output_sfm_file.parent.glob(f"{output_sfm_file.stem}*.sfmr"):
            stale.unlink()
        # First attempt uses the fixed seed for a reproducible result; retries
        # let the solver randomize so a fresh split can register all images.
        seed = random_seed if attempt == 1 else None
        _solve(
            [],
            workspace_dir,
            colmap_dir,
            matches_file=matches_file,
            random_seed=seed,
            output_sfm_file=str(output_sfm_file),
        )
        path, count = _largest_recon(output_sfm_file)
        if count > best_count:
            best_path, best_count = path, count
            # Stash the best so far; the next attempt clears the output dir.
            stash = output_sfm_file.parent / f"_best{output_sfm_file.suffix}"
            shutil.copy(path, stash)
            best_path = stash
        if expected_image_count is None or best_count >= expected_image_count:
            break

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
    from sfmtool._sfmtool import SfmrReconstruction

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
    from sfmtool._sfmtool import SfmrReconstruction

    workspace_dir = tmp_path_factory.mktemp("kerry_park_sfmr")
    _copy_kerry_park_into(workspace_dir)

    image_paths: list[Path] = []
    for sensor in KERRY_PARK_SENSORS:
        frames = sorted((workspace_dir / sensor).glob("frame_*.jpg"))
        image_paths.extend(frames[:KERRY_PARK_SOLVE_FRAME_COUNT])

    expected_count = len(KERRY_PARK_SENSORS) * KERRY_PARK_SOLVE_FRAME_COUNT
    output_sfm_file = workspace_dir / "kerry_park.sfmr"
    sfmr_path = build_cluster_reconstruction(
        workspace_dir,
        image_paths,
        output_sfm_file,
        incremental=False,
        random_seed=42,
        expected_image_count=expected_count,
    )

    recon = SfmrReconstruction.load(sfmr_path)
    if recon.image_count != expected_count:
        raise RuntimeError(
            f"kerry_park global solve registered {recon.image_count}/"
            f"{expected_count} images (all {expected_count} required)."
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
    from sfmtool._sfmtool import SfmrReconstruction

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
    # occasionally yields a degenerate solve — all frames register but no points
    # triangulate, so ``run_global_sfm`` raises "No 3D points found". Retry with a
    # fresh randomization (mirroring ``build_cluster_reconstruction``), keeping the
    # most-complete result, rather than flaking the suite. The first attempt stays
    # reproducible (seed 42); retries randomize so a different split can register.
    max_attempts = 6
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
        if recon.image_count == expected_count and recon.point_count >= 150:
            break

    if best_points < 0:
        raise RuntimeError(
            f"kerry_park .camrig global solve produced no complete reconstruction "
            f"in {max_attempts} attempts (all {expected_count} images required)."
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
