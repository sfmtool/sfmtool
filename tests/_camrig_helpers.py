# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the `.camrig` test modules (`test_camrig.py`,
`test_camrig_resolve.py`)."""

import shutil
from pathlib import Path

_IMAGE_DATA = Path(__file__).parent.parent / "test-data" / "images"


def _copy_images(dest: Path, dataset: str, count: int) -> None:
    """Copy the first `count` images of a checked-in dataset into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    sources = sorted((_IMAGE_DATA / dataset).glob(f"{dataset}_*.jpg"))[:count]
    assert len(sources) == count
    for source in sources:
        shutil.copy(source, dest / source.name)


def _camera(width: int = 640, height: int = 480) -> dict:
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "parameters": {
            "focal_length_x": 500.0,
            "focal_length_y": 500.0,
            "principal_point_x": width / 2,
            "principal_point_y": height / 2,
        },
    }


def _pinhole_camera() -> dict:
    return _camera()
