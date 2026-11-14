# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified `sfm inspect` CLI command."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.analyze.summary import _print_tool_options
from sfmtool.cli import main


# ── tool options rendering ───────────────────────────────────────────────────


def test_print_tool_options_renders_transforms_and_keys(capsys):
    """An operation's recorded parameters are surfaced: transforms first, then
    the remaining keys in sorted order."""
    _print_tool_options(
        {
            "transforms": [
                "Find points at infinity (eps=0.1, min_views=2)",
                "Bundle adjustment (release: focal)",
            ],
            "algorithm": "global",
            "rig_aware": True,
        }
    )
    out = capsys.readouterr().out
    assert "Tool options:" in out
    assert "transforms:" in out
    assert "Find points at infinity (eps=0.1, min_views=2)" in out
    assert "Bundle adjustment (release: focal)" in out
    assert "algorithm: global" in out
    assert "rig_aware: True" in out


def test_print_tool_options_empty_prints_nothing(capsys):
    """A reconstruction with no recorded parameters prints no block."""
    _print_tool_options({})
    _print_tool_options(None)
    assert capsys.readouterr().out == ""


# ── .sfmr reconstructions ────────────────────────────────────────────────────


def test_inspect_sfmr_default(seoul_bull_workspace):
    """Default inspect of a .sfmr prints the compact summary block."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["inspect", sfmr_path])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Format:" in out
    assert ".sfmr version" in out
    assert "Images:" in out
    assert "17" in out
    assert "Integrity:" in out
    assert "OK" in out


def test_inspect_sfmr_verbose(seoul_bull_workspace):
    """Verbose inspect of a .sfmr prints the full report."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["inspect", "-v", sfmr_path])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Reconstruction summary:" in out
    assert "3D Point statistics:" in out
    assert "Nearest neighbor distances:" in out
    # Regression: the feature tool lives at workspace.contents.feature_tool in
    # the .sfmr metadata; inspect used to read workspace.feature_tool and
    # always printed "unknown".
    assert "Feature tool: sfmtool" in out


def test_inspect_sfmr_verbose_with_points_at_infinity(
    seoul_bull_workspace, tmp_path: Path
):
    """A point at infinity (w=0) projects its bearing to a finite pixel, so it
    has a well-defined reprojection error. Verbose inspect must report finite
    error stats (no inf) rather than crashing in the histogram range."""
    recon = SfmrReconstruction.load(str(seoul_bull_workspace))

    # Mark the first point as at infinity: homogeneous w=0 (a bearing direction).
    xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    xyzw[0, 3] = 0.0
    modified = recon.clone_with_changes(positions=xyzw)
    assert modified.infinity_point_count >= 1

    # recompute_point_errors must yield a finite error for the w=0 point.
    modified.recompute_point_errors()
    assert np.all(np.isfinite(modified.rms_reprojection_errors))

    out_path = tmp_path / "with_infinity.sfmr"
    modified.save(str(out_path))

    result = CliRunner().invoke(main, ["inspect", "-v", str(out_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Reprojection error:" in out
    assert "Mean: inf" not in out


def test_inspect_sfmr_image_readings(seoul_bull_workspace):
    """A version 15 .sfmr shows the sharpness spread and conversion by
    default, and verbose adds a sharpness histogram and the least sharp
    images, least sharp first."""
    recon = SfmrReconstruction.load(str(seoul_bull_workspace))
    sharpness = np.asarray(recon.sharpness)
    names = recon.image_names
    assert sharpness.shape == (17,)

    result = CliRunner().invoke(main, ["inspect", str(seoul_bull_workspace)])
    assert result.exit_code == 0, result.output
    row = _row_value(result.output, "Image readings")
    assert row.startswith(
        f"sharpness min {sharpness.min():.4g}, median {np.median(sharpness):.4g}, "
        f"max {sharpness.max():.4g}"
    )
    assert row.endswith("(0.2126*R + 0.7152*G + 0.0722*B)")

    verbose = CliRunner().invoke(main, ["inspect", "-v", str(seoul_bull_workspace)])
    assert verbose.exit_code == 0, verbose.output
    lines = verbose.output.splitlines()
    start = lines.index("Image readings:")
    block = lines[start : start + 12]
    assert block[1] == "  Conversion: 0.2126*R + 0.7152*G + 0.0722*B"
    assert block[3] == "    Sharpness distribution:"
    # The histogram spans the values, so its first and last buckets hold one.
    histogram = block[4].strip()
    assert len(histogram) == 32
    assert histogram[0] != " " and histogram[-1] != " "
    assert block[6] == "  Least sharp 5 images:"
    order = np.argsort(sharpness, kind="stable")[:5]
    assert block[7:12] == [f"    {sharpness[i]:.4g}  {names[i]}" for i in order]


def test_inspect_sfmr_below_version_15_has_no_readings(seoul_bull_ground_truth_sfmr):
    """A file below version 15 holds no readings, and both summaries say so."""
    result = CliRunner().invoke(main, ["inspect", str(seoul_bull_ground_truth_sfmr)])
    assert result.exit_code == 0, result.output
    assert _row_value(result.output, "Image readings") == "none"

    verbose = CliRunner().invoke(
        main, ["inspect", "-v", str(seoul_bull_ground_truth_sfmr)]
    )
    assert verbose.exit_code == 0, verbose.output
    assert "Image readings: none" in verbose.output.splitlines()
    assert "Sharpness distribution" not in verbose.output


def test_inspect_sfmr_verbose_says_mean_or_median_per_point(
    seoul_bull_ground_truth_sfmr,
):
    """Each observations-per-point line names its statistic, mean or median,
    and the median of an even count is the mean of the middle two."""
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    counts = np.asarray(recon.observation_counts)
    # Two points seen in 3 and 4 images: mean 3.50, median 3.5.
    keep = np.zeros(len(counts), dtype=bool)
    keep[np.flatnonzero(counts == 3)[0]] = True
    keep[np.flatnonzero(counts == 4)[0]] = True
    path = seoul_bull_ground_truth_sfmr.with_name("two_points.sfmr")
    recon.filter_points_by_mask(keep).save(str(path))

    result = CliRunner().invoke(main, ["inspect", "-v", str(path)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    observations = lines.index("  Observations: 7")
    assert lines[observations + 1] == "  Mean observations per point: 3.50"
    statistics = lines.index("Observation statistics:")
    assert lines[statistics + 1 : statistics + 5] == [
        "  Min observations per point: 3",
        "  Max observations per point: 4",
        "  Mean observations per point: 3.50",
        "  Median observations per point: 3.5",
    ]
    assert "Avg" not in result.output


# ── .sift feature files ──────────────────────────────────────────────────────


def _find_sift_file(workspace_dir: Path) -> Path:
    sift_files = sorted(workspace_dir.rglob("*.sift"))
    assert sift_files, f"no .sift files under {workspace_dir}"
    return sift_files[0]


def test_inspect_sift_default(seoul_bull_workspace):
    """Default inspect of a .sift prints the compact summary block."""
    sift_path = _find_sift_file(seoul_bull_workspace.parent)
    result = CliRunner().invoke(main, ["inspect", str(sift_path)])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Format:" in out
    assert ".sift" in out
    assert "Features:" in out
    assert "Integrity:" in out


def test_inspect_sift_verbose(seoul_bull_workspace):
    """Verbose inspect of a .sift prints hashes and top features."""
    sift_path = _find_sift_file(seoul_bull_workspace.parent)
    result = CliRunner().invoke(main, ["inspect", "-v", str(sift_path)])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Content hash" in out
    assert "Top 5 features" in out


def _write_sift(
    sample_sift_data,
    path: Path,
    version: int,
    tensor_xx_yy_xy=(0.004, 0.001, 0.001),
) -> Path:
    """A `.sift` at ``path`` from the sample data: version 2 with readings
    chosen so the derived values are round, or version 1 with none."""
    from sfmtool.fileio import write_sift

    feature_tool_metadata, metadata, positions, affine, descriptors, thumbnail = (
        sample_sift_data
    )
    feature_tool_metadata = dict(feature_tool_metadata)
    metadata = dict(metadata)
    if version == 2:
        metadata["sharpness"] = 0.0125
        metadata["gradient_structure_tensor_xx_yy_xy"] = list(tensor_xx_yy_xy)
    else:
        metadata["version"] = 1
        del metadata["sharpness"]
        del metadata["gradient_structure_tensor_xx_yy_xy"]
        feature_tool_metadata["feature_options"] = {"gray_formula": "G"}
    write_sift(
        path,
        {
            "feature_tool_metadata": feature_tool_metadata,
            "metadata": metadata,
            "positions_xy": positions,
            "affine_shapes": affine,
            "descriptors": descriptors,
            "thumbnail_y_x_rgb": thumbnail,
        },
    )
    return path


def _row_value(out: str, label: str) -> str:
    """The value of the ``label:`` row in ``out``."""
    for line in out.splitlines():
        if line.strip().startswith(label + ":"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no {label!r} row in:\n{out}")


def test_inspect_sift_shows_version_and_readings(sample_sift_data, tmp_path: Path):
    """A version 2 .sift shows its version, its two readings and the axis
    ratio and coherence derived from the tensor; verbose adds the conversion."""
    path = _write_sift(sample_sift_data, tmp_path / "v2.sift", version=2)
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert _row_value(out, "Format") == ".sift version 2"
    assert _row_value(out, "Sharpness") == "0.0125"
    assert (
        _row_value(out, "Gradient structure tensor")
        == "Exx 0.004, Eyy 0.001, Exy 0.001"
    )
    # |log(0.004 / 0.001)| = log 4.
    assert _row_value(out, "Axis ratio") == "1.386"
    # sqrt(0.003**2 + 4 * 0.001**2) / 0.005.
    assert _row_value(out, "Coherence") == "0.721"
    assert "Image readings" not in out
    assert "Image to gray" not in out

    verbose = CliRunner().invoke(main, ["inspect", "-v", str(path)])
    assert verbose.exit_code == 0, verbose.output
    assert _row_value(verbose.output, "Image to gray") == "G, rounded to 8 bits"


@pytest.mark.parametrize(
    "tensor, axis_ratio, coherence",
    [
        # Exactly one of Exx and Eyy is 0: all gradient along one axis.
        ((0.004, 0.0, 0.0), "inf", "1.000"),
        ((0.0, 0.004, 0.0), "inf", "1.000"),
        # Both are 0: a flat image.
        ((0.0, 0.0, 0.0), "n/a (flat image)", "0.000"),
        # Exy enters the coherence squared, so its sign does not change it.
        ((0.004, 0.001, -0.001), "1.386", "0.721"),
    ],
)
def test_inspect_sift_derived_value_edge_cases(
    sample_sift_data, tmp_path: Path, tensor, axis_ratio, coherence
):
    path = _write_sift(sample_sift_data, tmp_path / "v2.sift", 2, tensor)
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert _row_value(result.output, "Axis ratio") == axis_ratio
    assert _row_value(result.output, "Coherence") == coherence


def test_inspect_sift_with_invalid_readings(sample_sift_data, tmp_path: Path):
    """A .sift whose stored tensor no image reads says its readings are
    invalid rather than deriving from them; changed after it was written, it
    also fails its integrity check."""
    import json
    import zipfile
    from compression import zstd

    good = _write_sift(sample_sift_data, tmp_path / "good.sift", version=2)
    bad = tmp_path / "bad.sift"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "metadata.json.zst":
                metadata = json.loads(zstd.decompress(data))
                metadata["gradient_structure_tensor_xx_yy_xy"] = [-0.004, 0.001, 0.0]
                data = zstd.compress(json.dumps(metadata).encode())
            dst.writestr(info, data)
    result = CliRunner().invoke(main, ["inspect", str(bad)])
    assert result.exit_code != 0
    out = result.output
    assert _row_value(out, "Image readings").startswith("invalid (")
    assert "must be finite" in _row_value(out, "Image readings")
    assert _row_value(out, "Integrity") == "FAILED"
    assert "Axis ratio" not in out
    assert "Coherence" not in out


def test_inspect_sift_version_1_has_no_readings(sample_sift_data, tmp_path: Path):
    """A version 1 .sift, which holds no readings, inspects cleanly and says so."""
    path = _write_sift(sample_sift_data, tmp_path / "v1.sift", version=1)
    for args in (["inspect", str(path)], ["inspect", "-v", str(path)]):
        result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, result.output
        out = result.output
        assert _row_value(out, "Format") == ".sift version 1"
        assert _row_value(out, "Image readings") == "none"
        assert _row_value(out, "Integrity") == "OK"
        assert "Sharpness" not in out
        assert "Image to gray" not in out


def test_inspect_workspace_sift_reads_its_own_readings(seoul_bull_workspace):
    """An extracted .sift shows the readings its metadata stores."""
    from sfmtool.fileio import SiftFile

    sift_path = _find_sift_file(seoul_bull_workspace.parent)
    with SiftFile(sift_path) as reader:
        sharpness = reader.metadata["sharpness"]
    result = CliRunner().invoke(main, ["inspect", str(sift_path)])
    assert result.exit_code == 0, result.output
    assert _row_value(result.output, "Format") == ".sift version 2"
    assert _row_value(result.output, "Sharpness") == f"{sharpness:.4g}"


# ── .matches files ───────────────────────────────────────────────────────────


def test_inspect_matches(seoul_bull_workspace, tmp_path: Path):
    """Inspect a .matches file produced by exhaustive matching."""
    image_dir = seoul_bull_workspace.parent / "test_17_image"
    matches_path = tmp_path / "test.matches"
    match_result = CliRunner().invoke(
        main, ["match", "--exhaustive", str(image_dir), "-o", str(matches_path)]
    )
    assert match_result.exit_code == 0, match_result.output

    result = CliRunner().invoke(main, ["inspect", str(matches_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert ".matches version" in out
    assert "Image pairs:" in out
    assert "Two-view geom:" in out

    verbose = CliRunner().invoke(main, ["inspect", "-v", str(matches_path)])
    assert verbose.exit_code == 0, verbose.output
    assert "Descriptor distance" in verbose.output


# ── .camrig camera rigs ──────────────────────────────────────────────────────


def _build_camrig(out: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "spherical-tiles",
            str(out),
            "--n",
            "150",
            "--equirect-width",
            "256",
        ],
    )
    assert result.exit_code == 0, result.output


def test_inspect_camrig_default(tmp_path: Path):
    """Default inspect of a .camrig prints the compact summary block."""
    camrig_path = tmp_path / "tiles.camrig"
    _build_camrig(camrig_path)

    result = CliRunner().invoke(main, ["inspect", str(camrig_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Rig type:" in out
    assert "spherical_tiles" in out
    assert "Sensors:" in out
    assert "Integrity:" in out


def test_inspect_camrig_verbose(tmp_path: Path):
    """Verbose inspect of a .camrig prints attributes and content hash."""
    camrig_path = tmp_path / "tiles.camrig"
    _build_camrig(camrig_path)

    result = CliRunner().invoke(main, ["inspect", "-v", str(camrig_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Rig attributes:" in out
    assert "Content hash:" in out


def test_inspect_camrig_rejects_corrupt(tmp_path: Path):
    """A corrupt .camrig file fails inspection."""
    fake = tmp_path / "junk.camrig"
    fake.write_bytes(b"not a zip archive")
    result = CliRunner().invoke(main, ["inspect", str(fake)])
    assert result.exit_code != 0


# ── image files ──────────────────────────────────────────────────────────────


def test_inspect_image_default(isolated_seoul_bull_image: Path):
    """Default inspect of an image prints dimensions and size."""
    result = CliRunner().invoke(main, ["inspect", str(isolated_seoul_bull_image)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "JPEG image" in out
    assert "Dimensions:" in out
    assert "File size:" in out


def test_inspect_image_verbose(isolated_seoul_bull_image: Path):
    """Verbose inspect of an image prints the inferred camera."""
    result = CliRunner().invoke(main, ["inspect", "-v", str(isolated_seoul_bull_image)])
    assert result.exit_code == 0, result.output
    assert "Inferred camera" in result.output


def _expected_sharpness(image_path: Path, formula: str, bits: int | None) -> str:
    from sfmtool.fileio import read_image_rgb
    from sfmtool.sift import ImageToGray, image_readings_from_rgb

    readings = image_readings_from_rgb(
        read_image_rgb(image_path), ImageToGray(formula, bits)
    )
    return f"{readings.sharpness:.4g}"


def test_inspect_image_verbose_readings_outside_a_workspace(
    isolated_seoul_bull_image: Path,
):
    """Outside a workspace an image's readings are taken through the default
    conversion, and the heading says so."""
    result = CliRunner().invoke(main, ["inspect", "-v", str(isolated_seoul_bull_image)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Image readings (default conversion, no workspace):" in out
    assert _row_value(out, "Conversion") == "0.2126*R + 0.7152*G + 0.0722*B"
    assert _row_value(out, "Sharpness") == _expected_sharpness(
        isolated_seoul_bull_image, "0.2126*R + 0.7152*G + 0.0722*B", None
    )
    for label in ("Gradient structure tensor", "Axis ratio", "Coherence"):
        _row_value(out, label)

    default = CliRunner().invoke(main, ["inspect", str(isolated_seoul_bull_image)])
    assert "Image readings" not in default.output


def test_inspect_image_verbose_readings_use_the_workspace_conversion(
    isolated_seoul_bull_image: Path,
):
    """Inside a workspace an image's readings are taken through the
    workspace's recorded conversion."""
    workspace = isolated_seoul_bull_image.parent
    init = CliRunner().invoke(
        main,
        ["ws", "init", "--image-to-gray", "G", "--quantize-bits", "8", str(workspace)],
    )
    assert init.exit_code == 0, init.output
    result = CliRunner().invoke(main, ["inspect", "-v", str(isolated_seoul_bull_image)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Image readings (the workspace's conversion):" in out
    assert _row_value(out, "Conversion") == "G, rounded to 8 bits"
    expected = _expected_sharpness(isolated_seoul_bull_image, "G", 8)
    assert _row_value(out, "Sharpness") == expected
    assert expected != _expected_sharpness(
        isolated_seoul_bull_image, "0.2126*R + 0.7152*G + 0.0722*B", None
    )


# ── dispatch errors ──────────────────────────────────────────────────────────


def test_inspect_unsupported_type(tmp_path: Path):
    """An unsupported extension is rejected with a helpful message."""
    p = tmp_path / "input.txt"
    p.touch()
    result = CliRunner().invoke(main, ["inspect", str(p)])
    assert result.exit_code != 0
    assert "unsupported file type" in result.output


# ── pt3d_ point IDs ──────────────────────────────────────────────────────────


def _point_id(sfmr_path, index):
    """Build a pt3d_ ID for the given .sfmr and point index."""
    recon = SfmrReconstruction.load(str(sfmr_path))
    return f"pt3d_{recon.content_xxh128[:8]}_{index}"


def test_inspect_point_summary(seoul_bull_workspace):
    """A pt3d_ ID resolves to its .sfmr and prints the point summary."""
    sfmr = seoul_bull_workspace
    point_id = _point_id(sfmr, 0)
    result = CliRunner().invoke(main, ["inspect", point_id, str(sfmr.parent)])
    assert result.exit_code == 0, result.output
    assert point_id in result.output
    assert "Observations" in result.output
    assert sfmr.name in result.output


def test_inspect_point_verbose(seoul_bull_workspace):
    """--verbose adds the full triangulation analysis."""
    sfmr = seoul_bull_workspace
    point_id = _point_id(sfmr, 0)
    result = CliRunner().invoke(main, ["inspect", point_id, str(sfmr.parent), "-v"])
    assert result.exit_code == 0, result.output
    assert "Triangulation analysis:" in result.output
    assert "Point or bearing: " in result.output
    assert "Inverse-depth z" in result.output
    assert "Ray spread (max angle to mean): " in result.output
    assert "incidence" in result.output


def test_inspect_point_verbose_on_embedded_patches(seoul_bull_ground_truth_sfmr):
    """--verbose reads an embedded_patches file's own keypoints: no .sift needed.

    Point 188 is one of the ground truth's bearings.
    """
    sfmr = seoul_bull_ground_truth_sfmr
    point_id = _point_id(sfmr, 188)
    result = CliRunner().invoke(main, ["inspect", point_id, str(sfmr.parent), "-v"])
    assert result.exit_code == 0, result.output
    assert "Point or bearing: a bearing at 0.468 px" in result.output
    assert "incidence" in result.output
    assert "feat " not in result.output


def test_inspect_point_unknown_hash(seoul_bull_workspace):
    """A hash matching no .sfmr is a clear error."""
    sfmr = seoul_bull_workspace
    result = CliRunner().invoke(main, ["inspect", "pt3d_deadbeef_0", str(sfmr.parent)])
    assert result.exit_code != 0
    assert "no .sfmr" in result.output


def test_inspect_point_index_out_of_range(seoul_bull_workspace):
    """An index beyond the point count is rejected."""
    sfmr = seoul_bull_workspace
    point_id = _point_id(sfmr, 10_000_000)
    result = CliRunner().invoke(main, ["inspect", point_id, str(sfmr.parent)])
    assert result.exit_code != 0
    assert "out of range" in result.output


def test_inspect_workspace_arg_rejected_for_file(
    seoul_bull_workspace,
):
    """The second (workspace) argument is only valid with a point ID."""
    sfmr = seoul_bull_workspace
    result = CliRunner().invoke(main, ["inspect", str(sfmr), str(sfmr.parent)])
    assert result.exit_code != 0
    assert "point ID" in result.output


def test_inspect_sfmr_verbose_point_or_bearing(seoul_bull_ground_truth_sfmr):
    """Verbose inspect prints the point-or-bearing counts under depth reliability."""
    result = CliRunner().invoke(
        main, ["inspect", "-v", str(seoul_bull_ground_truth_sfmr)]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Point or bearing (likelihood-ratio test, threshold 25):" in out
    assert (
        "Noise level: 0.4677 px, measured over 1,229 observations of finite "
        "points, 4 excluded as outliers" in out
    )
    assert "Finite points called bearings: 0 of 266 scored" in out
    assert "Points at infinity called finite: 0 of 14 scored" in out


def test_inspect_sfmr_verbose_point_or_bearing_names_promotions(
    seoul_bull_ground_truth_sfmr,
):
    """A point at infinity the test calls finite is named on the summary line."""
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    # Store finite points 0 and 1 as bearings. The test reads their observations,
    # not the stored coordinate, so their rays still place them at a depth.
    xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    assert np.all(xyzw[:2, 3] == 1.0)
    xyzw[:2, :3] /= np.linalg.norm(xyzw[:2, :3], axis=1)[:, None]
    xyzw[:2, 3] = 0.0
    path = seoul_bull_ground_truth_sfmr.parent / "two_bearings.sfmr"
    recon.clone_with_changes(positions=xyzw).save(str(path))

    result = CliRunner().invoke(main, ["inspect", "-v", str(path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Finite points called bearings: 0 of 264 scored" in out
    line = next(x for x in out.splitlines() if "Points at infinity called finite" in x)
    assert "2 of 16 scored" in line
    assert "(points " in line
    named = line.split("(points ")[1].rstrip(")").split(", ")
    assert sorted(named) == ["0", "1"]
