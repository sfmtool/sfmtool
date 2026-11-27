# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/measurements/add_artificial_measurements.py``.

The script writes a copy of a reconstruction with distance measurements whose
readings are known by construction, for SfM Explorer's Measurement View. These
tests run it on a copy of the seoul bull ground truth and read the copy back
through the resolver, so each case is checked against what the viewer reads.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from sfmtool.fileio import read_sfmr_metadata
from sfmtool.reconstruction import SfmrReconstruction

from .conftest import SEOUL_BULL_DIR

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "measurements" / "add_artificial_measurements.py"
SOURCE = SEOUL_BULL_DIR / "seoul_bull_sculpture_ground_truth_v02.sfmr"


def _load_script():
    spec = importlib.util.spec_from_file_location("add_artificial_measurements", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # A dataclass looks its module up by name while it is being made.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


@pytest.fixture(scope="module")
def copies(tmp_path_factory):
    """The script's two copies of the seoul bull ground truth, written once."""
    out = tmp_path_factory.mktemp("measured")
    with_unit = out / "bull_measured.sfmr"
    without = out / "bull_measured_su.sfmr"
    assert (
        script.main([str(SOURCE), str(with_unit), "--no-unit-copy", str(without)]) == 0
    )
    return with_unit, without


def _scales(recon):
    """Each measurement's stated distance over the reconstruction's reading of
    it, both in metres, or ``None`` where it has no reading."""
    measurements = recon.measurements["distance"]
    names = script.LENGTH_UNIT_NAMES
    metres = script.METRES[recon.world_space_unit or "m"]
    out = []
    for i, resolved in enumerate(recon.resolve_distance_measurements()):
        d = resolved["scene_distance"]
        stated = (
            float(measurements["distances"][i])
            * script.METRES[names[int(measurements["units"][i])]]
        )
        out.append(None if d is None else stated / (d * metres))
    return out


def test_the_copy_holds_every_case_with_its_known_reading(copies):
    with_unit, _ = copies
    recon = SfmrReconstruction.load(with_unit)
    labels = recon.measurements["distance"]["labels"]
    assert labels == [
        "one image",
        "two images",
        "one image again",
        "bad measurement",
        "refused endpoint",
        "several squares",
    ]
    images = recon.measurements["distance"]["image_indexes"]
    assert images[0, 0] == images[0, 1], "the first case is not in one image"
    assert images[1, 0] != images[1, 1], "the second case is not in two images"

    scales = _scales(recon)
    for i in (0, 1, 2, 5):
        assert scales[i] == pytest.approx(1.02, rel=1e-5), (i, scales)
    assert scales[3] == pytest.approx(1.2, rel=1e-5)
    assert scales[4] is None

    resolved = recon.resolve_distance_measurements()
    assert resolved[4]["refusals"][1] is not None
    assert "holds pixel" in resolved[4]["refusals"][1]
    several = resolved[5]["endpoints"][0]
    assert len(several["candidates"]) >= 2


def test_the_copy_without_a_unit_holds_the_same_measurements(copies):
    with_unit, without = copies
    a = SfmrReconstruction.load(with_unit)
    b = SfmrReconstruction.load(without)
    assert a.world_space_unit == "m"
    assert b.world_space_unit is None
    for key in ("distances", "units", "image_indexes", "endpoints_xy"):
        assert (
            a.measurements["distance"][key] == b.measurements["distance"][key]
        ).all()


def test_the_copy_records_the_source_workspace(copies):
    with_unit, _ = copies
    workspace = read_sfmr_metadata(with_unit)["workspace"]
    assert Path(workspace["absolute_path"]).resolve() == SEOUL_BULL_DIR.resolve()


@pytest.mark.parametrize("where", ["destination", "no_unit_copy"])
def test_a_path_under_test_data_is_refused_and_nothing_is_written(tmp_path, where):
    inside = ROOT / "test-data" / "images" / "never_written.sfmr"
    outside = tmp_path / "copy.sfmr"
    args = [str(SOURCE), str(inside if where == "destination" else outside)]
    if where == "no_unit_copy":
        args += ["--no-unit-copy", str(inside)]
    with pytest.raises(SystemExit, match="test-data"):
        script.main(args)
    assert not inside.exists()
    assert not outside.exists()
