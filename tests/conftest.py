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
        TEST_DATA_DIR / "images" / "seoul_bull_sculpture" / "seoul_bull_sculpture_01.jpg"
    )
    tmp_path = tmp_path_factory.mktemp("test_image")
    img_path = tmp_path / "test_image.jpg"
    shutil.copy(input_img_path, img_path)
    return img_path


@pytest.fixture
def isolated_seoul_bull_17_images(tmp_path_factory) -> list[Path]:
    """Fixture that provides 17 .jpg files isolated in a directory for testing."""
    data_dir = TEST_DATA_DIR / "images" / "seoul_bull_sculpture"
    image_files = sorted(data_dir.glob("seoul_bull_sculpture_*.jpg"))
    assert len(image_files) == 17
    tmp_path = tmp_path_factory.mktemp("test_17_images")

    img_paths = []
    for img_file in image_files:
        dest_path = tmp_path / img_file.name
        shutil.copy(img_file, dest_path)
        img_paths.append(dest_path)

    return img_paths


@pytest.fixture(scope="session")
def sfmrfile_reconstruction_with_17_images_once(tmp_path_factory) -> Path:
    """Session-scoped fixture: build a .sfmr reconstruction from 17 images."""
    from sfmtool._isfm import run_incremental_sfm

    data_dir = TEST_DATA_DIR / "images" / "seoul_bull_sculpture"
    image_files = sorted(data_dir.glob("seoul_bull_sculpture_*.jpg"))
    workspace_dir = tmp_path_factory.mktemp("workspace_17_images")
    init_workspace(workspace_dir, domain_size_pooling=True)
    (workspace_dir / "test_17_image").mkdir(exist_ok=True)

    img_paths = []
    for img_file in image_files:
        dest_path = workspace_dir / "test_17_image" / img_file.name
        shutil.copy(img_file, dest_path)
        img_paths.append(dest_path)

    return run_incremental_sfm(
        img_paths,
        workspace_dir,
        workspace_dir / "colmap",
        random_seed=42,
        output_sfm_file=workspace_dir / "seoul_bull.sfmr",
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
