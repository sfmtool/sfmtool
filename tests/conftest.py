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


@pytest.fixture(scope="session")
def sfmrfile_reconstruction_with_17_images_once(tmp_path_factory) -> Path:
    """Session-scoped fixture: build a .sfmr reconstruction from 17 images.

    The incremental SfM solver sometimes converges with only a couple of
    images registered. Run without a fixed random seed and retry until all
    17 images are registered.
    """
    from sfmtool._incremental_sfm import run_incremental_sfm
    from sfmtool._sfmtool import SfmrReconstruction

    data_dir = TEST_DATA_DIR / "images" / "seoul_bull_sculpture"
    image_files = sorted(data_dir.glob("seoul_bull_sculpture_*.jpg"))
    workspace_dir = tmp_path_factory.mktemp("workspace_17_images")
    init_workspace(workspace_dir, domain_size_pooling=True)
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

    output_sfm_file = workspace_dir / "seoul_bull.sfmr"
    colmap_dir = workspace_dir / "colmap"
    expected_image_count = len(image_files)
    max_attempts = 20
    for attempt in range(1, max_attempts + 1):
        if colmap_dir.exists():
            shutil.rmtree(colmap_dir)
        if output_sfm_file.exists():
            output_sfm_file.unlink()
        sfmr_path = run_incremental_sfm(
            img_paths,
            workspace_dir,
            colmap_dir,
            output_sfm_file=output_sfm_file,
        )
        recon = SfmrReconstruction.load(sfmr_path)
        if recon.image_count == expected_image_count:
            return sfmr_path
        print(
            f"SfM solve attempt {attempt}/{max_attempts} registered "
            f"{recon.image_count}/{expected_image_count} images, retrying."
        )
    raise RuntimeError(
        f"Failed to register all {expected_image_count} images after "
        f"{max_attempts} attempts."
    )


@pytest.fixture
def sfmrfile_reconstruction_with_17_images(
    sfmrfile_reconstruction_with_17_images_once: Path, tmp_path_factory
) -> Path:
    """Per-test isolation of the 17-image .sfmr reconstruction."""
    source_workspace_dir = sfmrfile_reconstruction_with_17_images_once.parent
    workspace_dir = tmp_path_factory.mktemp("workspace_17_images")
    shutil.copytree(source_workspace_dir, workspace_dir, dirs_exist_ok=True)
    return workspace_dir / sfmrfile_reconstruction_with_17_images_once.name


KERRY_PARK_DIR = TEST_DATA_DIR / "images" / "kerry_park"
KERRY_PARK_FRAME_COUNT = 24
KERRY_PARK_SENSORS = ("fisheye_left", "fisheye_right")


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
def sfmrfile_reconstruction_kerry_park_once(tmp_path_factory) -> Path:
    """Session-scoped: build a .sfmr reconstruction from the kerry_park rig.

    Uses global SfM (GLOMAP) with a fixed random seed. The global solver
    reliably registers all 48 rig images; the fixture fails fast if it
    doesn't, rather than handing a partial reconstruction to the tests.
    """
    from sfmtool._global_sfm import run_global_sfm
    from sfmtool._sfmtool import SfmrReconstruction

    workspace_dir = tmp_path_factory.mktemp("kerry_park_sfmr")
    _copy_kerry_park_into(workspace_dir)
    init_workspace(workspace_dir, max_num_features=2000, domain_size_pooling=True)

    image_paths: list[Path] = []
    for sensor in KERRY_PARK_SENSORS:
        image_paths.extend(sorted((workspace_dir / sensor).glob("frame_*.jpg")))

    output_sfm_file = workspace_dir / "kerry_park.sfmr"
    colmap_dir = workspace_dir / "colmap"
    sfmr_path = run_global_sfm(
        image_paths,
        workspace_dir,
        colmap_dir,
        output_sfm_file=str(output_sfm_file),
        random_seed=42,
    )

    expected_count = len(KERRY_PARK_SENSORS) * KERRY_PARK_FRAME_COUNT
    recon = SfmrReconstruction.load(sfmr_path)
    if recon.image_count != expected_count:
        raise RuntimeError(
            f"kerry_park global solve registered {recon.image_count}/"
            f"{expected_count} images (all {expected_count} required)."
        )
    return sfmr_path


@pytest.fixture
def sfmrfile_reconstruction_kerry_park(
    sfmrfile_reconstruction_kerry_park_once: Path, tmp_path_factory
) -> Path:
    """Per-test isolation of the kerry_park .sfmr reconstruction."""
    source_workspace_dir = sfmrfile_reconstruction_kerry_park_once.parent
    workspace_dir = tmp_path_factory.mktemp("kerry_park_sfmr")
    shutil.copytree(source_workspace_dir, workspace_dir, dirs_exist_ok=True)
    return workspace_dir / sfmrfile_reconstruction_kerry_park_once.name


@pytest.fixture(scope="session")
def sfmrfile_reconstruction_kerry_park_camrig_once(tmp_path_factory) -> Path:
    """Session-scoped: build a .sfmr reconstruction from the kerry_park rig,
    with the rig described by a multi-sensor ``kerry_park.camrig``.

    The same global SfM solve as :func:`sfmrfile_reconstruction_kerry_park_once`
    but driven by the ``.camrig`` rig-discovery path rather than
    ``rig_config.json``.
    """
    from sfmtool._global_sfm import run_global_sfm
    from sfmtool._sfmtool import SfmrReconstruction

    workspace_dir = tmp_path_factory.mktemp("kerry_park_camrig_sfmr")
    _copy_kerry_park_camrig_into(workspace_dir)
    init_workspace(workspace_dir, max_num_features=2000, domain_size_pooling=True)

    image_paths: list[Path] = []
    for sensor in KERRY_PARK_SENSORS:
        image_paths.extend(sorted((workspace_dir / sensor).glob("frame_*.jpg")))

    output_sfm_file = workspace_dir / "kerry_park.sfmr"
    colmap_dir = workspace_dir / "colmap"
    sfmr_path = run_global_sfm(
        image_paths,
        workspace_dir,
        colmap_dir,
        output_sfm_file=str(output_sfm_file),
        random_seed=42,
    )

    expected_count = len(KERRY_PARK_SENSORS) * KERRY_PARK_FRAME_COUNT
    recon = SfmrReconstruction.load(sfmr_path)
    if recon.image_count != expected_count:
        raise RuntimeError(
            f"kerry_park .camrig global solve registered {recon.image_count}/"
            f"{expected_count} images (all {expected_count} required)."
        )
    return sfmr_path


@pytest.fixture
def sfmrfile_reconstruction_kerry_park_camrig(
    sfmrfile_reconstruction_kerry_park_camrig_once: Path, tmp_path_factory
) -> Path:
    """Per-test isolation of the kerry_park ``.camrig`` .sfmr reconstruction."""
    source_workspace_dir = sfmrfile_reconstruction_kerry_park_camrig_once.parent
    workspace_dir = tmp_path_factory.mktemp("kerry_park_camrig_sfmr")
    shutil.copytree(source_workspace_dir, workspace_dir, dirs_exist_ok=True)
    return workspace_dir / sfmrfile_reconstruction_kerry_park_camrig_once.name
