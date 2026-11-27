# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfm xform --scale-to-median-measurement`` on the demo scene,
whose one stored measurement is its cube's 1 m edge, in a file in metres."""

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import LENGTH_UNIT_NAMES, read_sfmr_metadata
from sfmtool.reconstruction import SfmrReconstruction


def _restated(demo, rows):
    """The demo with its measurements replaced by `rows`.

    Each row is ``(distance, unit, label, on_edge)``: a measurement of the
    cube's edge stated as ``distance`` in ``unit``, or, with ``on_edge``
    false, one whose endpoints are both on the corner pixel of the edge's
    images, where no observation's square holds them.
    """
    d = demo.measurements["distance"]
    endpoints = np.stack(
        [
            d["endpoints_xy"][0] if on_edge else np.full((2, 2), 0.5, np.float32)
            for *_, on_edge in rows
        ]
    )
    return demo.clone_with_changes(
        measurements={
            "distance": {
                "distances": np.array([r[0] for r in rows], dtype=np.float32),
                "units": np.array(
                    [LENGTH_UNIT_NAMES.index(r[1]) for r in rows], dtype=np.uint8
                ),
                "image_indexes": np.repeat(d["image_indexes"][:1], len(rows), axis=0),
                "endpoints_xy": endpoints,
                "labels": [r[2] for r in rows],
            }
        }
    )


def _write(demo_workspace, recon, name="input.sfmr"):
    path = demo_workspace.with_name(name)
    recon.save(path)
    return path


def _xform(input_path, *options):
    output = input_path.with_name("out.sfmr")
    result = CliRunner().invoke(main, ["xform", str(input_path), str(output), *options])
    return result, output


def test_the_scale_puts_the_edge_at_its_stated_length_in_the_unit(demo_workspace):
    demo = SfmrReconstruction.load(demo_workspace)
    assert demo.world_space_unit == "m"

    result, output = _xform(demo_workspace, "--scale-to-median-measurement", "mm")

    assert result.exit_code == 0, result.output
    out = SfmrReconstruction.load(output)
    assert out.world_space_unit == "mm"
    (reading,) = out.resolve_distance_measurements()
    assert reading["scene_distance"] == pytest.approx(1000.0, rel=1e-5)
    finite = ~demo.point_is_at_infinity
    np.testing.assert_allclose(
        out.positions[finite], 1000.0 * demo.positions[finite], rtol=1e-5, atol=1e-6
    )
    assert "Used 1 of 1 distance measurements:" in result.output
    assert (
        'Distance #0 "cube edge": stated 1000 mm, scene 1 su, factor 1000'
        in result.output
    )
    assert "std: none for a single measurement" in result.output
    assert "Left out" not in result.output
    assert "WARNING" not in result.output
    assert "Scaling reconstruction by 1000\n" in result.output


def test_stated_lengths_in_other_units_convert_to_the_target(demo_workspace):
    """Two statements of the edge, in cm and in ft: the median of two is their
    mean, so a unit misread in either moves the factor."""
    demo = SfmrReconstruction.load(demo_workspace)
    path = _write(
        demo_workspace,
        _restated(demo, [(100.0, "cm", "", True), (1.0 / 0.3048, "ft", "", True)]),
    )

    result, output = _xform(path, "--scale-to-median-measurement", "mm")

    assert result.exit_code == 0, result.output
    assert result.output.count(": stated 1000 mm, scene 1 su, factor 1000") == 2
    (a, b) = SfmrReconstruction.load(output).resolve_distance_measurements()
    assert a["scene_distance"] == pytest.approx(1000.0, rel=1e-5)
    assert b["scene_distance"] == pytest.approx(1000.0, rel=1e-5)


def test_a_measurement_that_gives_no_factor_is_left_out_with_its_reason(demo_workspace):
    demo = SfmrReconstruction.load(demo_workspace)
    path = _write(
        demo_workspace,
        _restated(demo, [(1.0, "m", "edge", True), (2.0, "m", "corner", False)]),
    )

    result, output = _xform(path, "--scale-to-median-measurement", "cm")

    assert result.exit_code == 0, result.output
    assert "Used 1 of 2 distance measurements:" in result.output
    assert 'Distance #0 "edge": stated 100 cm' in result.output
    assert '  Left out:\n    Distance #1 "corner": endpoint A has no position: ' in (
        result.output
    )
    assert "Scaling reconstruction by 100\n" in result.output
    assert SfmrReconstruction.load(output).world_space_unit == "cm"


def test_measurements_that_disagree_are_warned_about(demo_workspace):
    """The edge stated three times, once 20% long: the factor is the other
    two's, and the warning names the long one."""
    demo = SfmrReconstruction.load(demo_workspace)
    path = _write(
        demo_workspace,
        _restated(
            demo,
            [(1.0, "m", "", True), (1.0, "m", "", True), (1.2, "m", "long", True)],
        ),
    )

    result, output = _xform(path, "--scale-to-median-measurement", "mm")

    assert result.exit_code == 0, result.output
    assert 'Distance #2 "long": stated 1200 mm' in result.output
    assert "deviation +20.00%" in result.output
    assert "std: 115.5 (11.55% of the median)" in result.output
    assert (
        "WARNING: 1 of 3 measurements deviate more than 5% from the median "
        '(Distance #2 "long")'
    ) in result.output
    assert "Scaling reconstruction by 1000\n" in result.output
    first, _, _ = SfmrReconstruction.load(output).resolve_distance_measurements()
    assert first["scene_distance"] == pytest.approx(1000.0, rel=1e-5)


def test_no_usable_measurement_fails_and_writes_nothing(demo_workspace):
    demo = SfmrReconstruction.load(demo_workspace)
    path = _write(demo_workspace, _restated(demo, [(1.0, "m", "", False)]))

    result, output = _xform(path, "--scale-to-median-measurement", "mm")

    assert result.exit_code == 1
    assert (
        "Error: --scale-to-median-measurement mm: none of the 1 distance "
        "measurements gives a scale; measurement 0: endpoint A has no position: "
    ) in result.output
    assert not output.exists()


def test_a_file_with_no_measurement_fails_and_writes_nothing(demo_workspace):
    demo = SfmrReconstruction.load(demo_workspace)
    path = _write(demo_workspace, demo.clone_with_changes(measurements=None))

    result, output = _xform(path, "--scale-to-median-measurement", "mm")

    assert result.exit_code == 1
    assert (
        "Error: --scale-to-median-measurement mm: the reconstruction stores no "
        "distance measurement, so no stated length gives it a scale"
    ) in result.output
    assert not output.exists()


def test_the_step_measures_the_reconstruction_as_it_stands_in_the_pipeline(
    demo_workspace,
):
    """After ``--scale 0.5`` the edge is half a unit long, so the step scales
    by 200 to centimetres, and the translation after it is in centimetres."""
    demo = SfmrReconstruction.load(demo_workspace)

    result, output = _xform(
        demo_workspace,
        "--scale",
        "0.5",
        "--scale-to-median-measurement",
        "cm",
        "--translate",
        "0,0,5",
    )

    assert result.exit_code == 0, result.output
    assert "[2/3] Scale to median measurement (cm)" in result.output
    assert "Scaling reconstruction by 200\n" in result.output
    out = SfmrReconstruction.load(output)
    assert out.world_space_unit == "cm"
    finite = ~demo.point_is_at_infinity
    np.testing.assert_allclose(
        out.positions[finite],
        100.0 * demo.positions[finite] + [0.0, 0.0, 5.0],
        rtol=1e-5,
        atol=1e-4,
    )
    assert read_sfmr_metadata(output)["tool_options"]["transforms"][1] == (
        "Scale to median measurement: 200x to cm"
    )


def test_the_removed_yaml_option_is_an_unknown_option(demo_workspace):
    result, output = _xform(demo_workspace, "--scale-by-measurements", "m.yaml")

    assert result.exit_code == 2
    assert (
        "No such option '--scale-by-measurements'. "
        "Did you mean '--scale-to-median-measurement'?"
    ) in result.output
    assert not output.exists()
