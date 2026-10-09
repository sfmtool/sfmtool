# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main


def test_xform_no_transforms(tmp_path: Path):
    """Test that xform without any transforms raises an error."""
    # Create a dummy .sfmr file (doesn't need to be valid for this test)
    input_path = tmp_path / "input.sfmr"
    input_path.touch()
    output_path = tmp_path / "output.sfmr"
    args = ["xform", str(input_path), str(output_path)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "At least one transformation must be specified" in result.output


def test_xform_max_features_without_find_infinity_rejected(tmp_path: Path):
    """--max-features without --find-points-at-infinity is rejected (B4)."""
    input_path = tmp_path / "input.sfmr"
    input_path.touch()
    output_path = tmp_path / "output.sfmr"
    args = [
        "xform",
        str(input_path),
        str(output_path),
        "--scale",
        "2",
        "--max-features",
        "500",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "--max-features only applies to --find-points-at-infinity" in result.output


def test_xform_non_sfmr_input(tmp_path: Path):
    """Test that xform with non-.sfmr input raises an error."""
    input_path = tmp_path / "input.txt"
    input_path.touch()
    output_path = tmp_path / "output.sfmr"
    args = ["xform", str(input_path), str(output_path), "--scale", "2.0"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "Input path must be a .sfmr file" in result.output


def test_xform_non_sfmr_output(tmp_path: Path):
    """Test that xform with non-.sfmr output raises an error."""
    input_path = tmp_path / "input.sfmr"
    input_path.touch()
    output_path = tmp_path / "output.txt"
    args = ["xform", str(input_path), str(output_path), "--scale", "2.0"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "Output path must be a .sfmr file" in result.output


def test_xform_on_reconstruction(seoul_bull_workspace: Path):
    """Test xform scale on a real reconstruction."""
    output_sfmr = seoul_bull_workspace
    workspace_dir = output_sfmr.parent

    # Now test xform with scale
    scaled_sfmr = workspace_dir / "scaled.sfmr"
    args = ["xform", str(output_sfmr), str(scaled_sfmr), "--scale", "2.0"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert scaled_sfmr.exists()
    assert "Scale by 2.000" in result.output
    assert "Transformation complete" in result.output

    # Verify the scaled reconstruction
    from sfmtool.reconstruction import SfmrReconstruction

    original = SfmrReconstruction.load(output_sfmr)
    scaled = SfmrReconstruction.load(scaled_sfmr)
    assert scaled.image_count == original.image_count
    assert scaled.point_count == original.point_count


def test_xform_remove_short_tracks(seoul_bull_workspace: Path):
    """Test xform with short track removal."""
    output_sfmr = seoul_bull_workspace
    workspace_dir = output_sfmr.parent

    # Remove short tracks
    filtered_sfmr = workspace_dir / "filtered.sfmr"
    args = ["xform", str(output_sfmr), str(filtered_sfmr), "--remove-short-tracks", "3"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert filtered_sfmr.exists()
    assert "Remove tracks with length <= 3" in result.output

    from sfmtool.reconstruction import SfmrReconstruction

    original = SfmrReconstruction.load(output_sfmr)
    filtered = SfmrReconstruction.load(filtered_sfmr)
    assert filtered.point_count <= original.point_count
    assert filtered.image_count == original.image_count


def test_xform_camera_model_with_bundle_adjust(
    seoul_bull_workspace: Path,
):
    """`--camera-model RADIAL --bundle-adjust` upgrades SIMPLE_RADIAL → RADIAL,
    then bundle adjustment refines the k2 term that was zero-initialized."""
    output_sfmr = seoul_bull_workspace
    workspace_dir = output_sfmr.parent

    switched_sfmr = workspace_dir / "radial_ba.sfmr"
    args = [
        "xform",
        str(output_sfmr),
        str(switched_sfmr),
        "--camera-model",
        "RADIAL",
        "--bundle-adjust",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert switched_sfmr.exists()
    assert "Switch camera model to RADIAL" in result.output

    from sfmtool.reconstruction import SfmrReconstruction

    switched = SfmrReconstruction.load(switched_sfmr)
    for camera in switched.cameras:
        assert camera.model == "RADIAL"


def test_xform_camera_model_unknown(seoul_bull_workspace: Path):
    """An unknown camera model is rejected at the CLI."""
    output_sfmr = seoul_bull_workspace
    workspace_dir = output_sfmr.parent

    bad_sfmr = workspace_dir / "bad.sfmr"
    args = [
        "xform",
        str(output_sfmr),
        str(bad_sfmr),
        "--camera-model",
        "NOT_A_MODEL",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "Unknown camera model" in result.output


def test_xform_default_output_path(seoul_bull_workspace: Path):
    """When OUTPUT_PATH is omitted, xform writes {stem}-transformed.sfmr next
    to the input, then -2, -3, ... for subsequent runs."""
    input_sfmr = seoul_bull_workspace
    workspace_dir = input_sfmr.parent
    stem = input_sfmr.stem

    expected_first = workspace_dir / f"{stem}-transformed.sfmr"
    expected_second = workspace_dir / f"{stem}-transformed-2.sfmr"
    expected_third = workspace_dir / f"{stem}-transformed-3.sfmr"

    args = ["xform", str(input_sfmr), "--scale", "2.0"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert expected_first.exists()
    assert str(expected_first) in result.output

    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert expected_second.exists()

    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert expected_third.exists()


def test_xform_chained_transforms(seoul_bull_workspace: Path):
    """Test xform with multiple chained transforms."""
    output_sfmr = seoul_bull_workspace
    workspace_dir = output_sfmr.parent

    # Chain: remove short tracks -> scale -> translate
    chained_sfmr = workspace_dir / "chained.sfmr"
    args = [
        "xform",
        str(output_sfmr),
        str(chained_sfmr),
        "--remove-short-tracks",
        "2",
        "--scale",
        "0.5",
        "--translate",
        "1,2,3",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert chained_sfmr.exists()
    assert "Applying 3 transformation(s)" in result.output


def test_xform_joined_negative_translation_translates(
    seoul_bull_ground_truth_sfmr: Path, tmp_path: Path
):
    """`--translate=-1,2,3` moves every point and is recorded with the other steps."""
    import numpy as np

    from sfmtool.fileio import read_sfmr

    output_sfmr = tmp_path / "translated.sfmr"
    args = [
        "xform",
        str(seoul_bull_ground_truth_sfmr),
        str(output_sfmr),
        "--scale",
        "1",
        "--translate=-1,2,3",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output

    before = read_sfmr(seoul_bull_ground_truth_sfmr)
    after = read_sfmr(output_sfmr)
    finite = before["positions_xyzw"][:, 3] != 0
    assert finite.any()

    def euclidean(data):
        xyzw = data["positions_xyzw"][finite]
        return xyzw[:, :3] / xyzw[:, 3:]

    moved = euclidean(after) - euclidean(before)
    assert np.allclose(moved, [-1.0, 2.0, 3.0], atol=1e-6)
    assert len(after["metadata"]["tool_options"]["transforms"]) == 2


def test_xform_separated_negative_value_matches_joined(
    seoul_bull_ground_truth_sfmr: Path, tmp_path: Path
):
    """`--translate -1,2,3` and `--translate=-1,2,3` write the same points."""
    from sfmtool.fileio import read_sfmr

    outputs = []
    for spelling in (["--translate", "-1,2,3"], ["--translate=-1,2,3"]):
        output_sfmr = tmp_path / f"out{len(outputs)}.sfmr"
        args = ["xform", str(seoul_bull_ground_truth_sfmr), str(output_sfmr)]
        result = CliRunner().invoke(main, args + spelling)
        assert result.exit_code == 0, result.output
        outputs.append(read_sfmr(output_sfmr))
    assert (outputs[0]["positions_xyzw"] == outputs[1]["positions_xyzw"]).all()


def test_xform_paths_among_the_options(
    seoul_bull_ground_truth_sfmr: Path, tmp_path: Path
):
    """An output path written after an option is still the output, and the
    option before it is still applied."""
    output_sfmr = tmp_path / "out.sfmr"
    args = [
        "xform",
        str(seoul_bull_ground_truth_sfmr),
        "--scale",
        "2",
        str(output_sfmr),
        "--translate=1,0,0",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "Applying 2 transformation(s)" in result.output
    assert output_sfmr.exists()


def test_xform_value_free_option_refuses_joined_value(
    seoul_bull_ground_truth_sfmr: Path, tmp_path: Path
):
    args = [
        "xform",
        str(seoul_bull_ground_truth_sfmr),
        str(tmp_path / "out.sfmr"),
        "--drop-thumbnails=x",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "--drop-thumbnails" in result.output
    assert "does not take a value" in result.output


def test_xform_unknown_option_refused(
    seoul_bull_ground_truth_sfmr: Path, tmp_path: Path
):
    args = [
        "xform",
        str(seoul_bull_ground_truth_sfmr),
        str(tmp_path / "out.sfmr"),
        "--scale",
        "2",
        "--tranlsate=1,2,3",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "--tranlsate" in result.output
    assert not (tmp_path / "out.sfmr").exists()
