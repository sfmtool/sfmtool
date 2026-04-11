# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm discontinuity` CLI command."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main


SEOUL_BULL_DIR = (
    Path(__file__).parent.parent / "test-data" / "images" / "seoul_bull_sculpture"
)


@pytest.fixture
def runner():
    return CliRunner()


# --- Error handling ---


def test_no_paths(runner):
    """No arguments produces a usage error."""
    result = runner.invoke(main, ["discontinuity"])
    assert result.exit_code != 0
    assert "Must provide" in result.output


def test_sfmr_invalid_file(runner, tmp_path):
    """Passing an invalid .sfmr file gives an error."""
    fake_sfmr = tmp_path / "test.sfmr"
    fake_sfmr.touch()
    result = runner.invoke(main, ["discontinuity", str(fake_sfmr)])
    assert result.exit_code != 0


def test_no_images_found(runner, tmp_path):
    """Empty directory produces an error."""
    result = runner.invoke(main, ["discontinuity", str(tmp_path)])
    assert result.exit_code != 0
    assert "No image files found" in result.output


def test_single_image_no_sequence(runner, tmp_path):
    """A single image can't form a sequence — error."""
    import shutil

    src = SEOUL_BULL_DIR / "seoul_bull_sculpture_01.jpg"
    shutil.copy(src, tmp_path / "img_01.jpg")
    result = runner.invoke(main, ["discontinuity", str(tmp_path)])
    assert result.exit_code != 0
    assert "No numbered sequences" in result.output


def test_non_sequential_images(runner, tmp_path):
    """Images without numbered names produce a no-sequence error."""
    import shutil

    for name in ("alpha.jpg", "beta.jpg", "gamma.jpg"):
        shutil.copy(SEOUL_BULL_DIR / "seoul_bull_sculpture_01.jpg", tmp_path / name)
    result = runner.invoke(main, ["discontinuity", str(tmp_path)])
    assert result.exit_code != 0
    assert "No numbered sequences" in result.output


# --- Analysis with stride 1 (default) ---


def test_stride_1_analysis(runner):
    """Default stride=1 on 3 frames: checks output structure, tile grids,
    histograms, stride-2 comparison, and summary."""
    result = runner.invoke(
        main,
        [
            "discontinuity",
            str(SEOUL_BULL_DIR),
            "-r",
            "1-3",
            "--no-adaptive",
            "--initial-stride",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    out = result.output
    # Sequence detection
    assert "Found 3 images in 1 sequence" in out
    assert "Analyzing sequence:" in out

    # Every frame sampled (stride=1 advances by 1)
    assert "Frame 1:" in out
    assert "Frame 2:" in out

    # Local and stride flow reported
    assert "Local:" in out
    assert "Stride:" in out

    # Stride-2 comparison: from frame 1 targets frame 3
    assert "seoul_bull_sculpture_03.jpg" in out

    # Tile magnitude grids and difference
    assert "tile magnitudes" in out.lower()
    assert "Difference" in out

    # In-bounds percentage
    assert "in bounds" in out.lower()

    # Direction histogram box characters
    assert "\u2502" in out

    # Summary present
    assert "Summary:" in out


# --- Fixed stride ---


def test_fixed_stride(runner):
    """--no-adaptive with stride 2: samples at frame 1 and 3, no stride changes."""
    result = runner.invoke(
        main,
        [
            "discontinuity",
            str(SEOUL_BULL_DIR),
            "-r",
            "1-4",
            "--initial-stride",
            "2",
            "--no-adaptive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Frame 1:" in result.output
    assert "Frame 3:" in result.output
    # No stride change messages
    assert "\u2193 stride" not in result.output
    assert "\u2191 stride" not in result.output


# --- Adaptive stride ---


def test_adaptive_stride(runner):
    """Adaptive stride on 8 frames runs to completion with summary."""
    result = runner.invoke(
        main,
        ["discontinuity", str(SEOUL_BULL_DIR), "-r", "1-8"],
    )
    assert result.exit_code == 0, result.output
    assert "Found 8 images in 1 sequence" in result.output
    assert "Frame 1:" in result.output
    assert "Summary:" in result.output


# --- Flow image saving ---


def test_save_flow_dir(runner, tmp_path):
    """--save-flow-dir saves valid flow color images with correct naming."""
    flow_dir = tmp_path / "flows"
    result = runner.invoke(
        main,
        [
            "discontinuity",
            str(SEOUL_BULL_DIR),
            "-r",
            "1-3",
            "--no-adaptive",
            "--initial-stride",
            "1",
            "--save-flow-dir",
            str(flow_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert flow_dir.exists()

    saved_files = sorted(flow_dir.glob("*.jpg"))
    assert len(saved_files) > 0

    # Naming convention: {seq_name}_from_{N}_to_{M}.jpg
    names = [f.name for f in saved_files]
    assert any("seoul_bull_sculpture" in n for n in names)
    assert any("_from_" in n and "_to_" in n for n in names)

    # Files are valid JPEGs with nonzero content
    for jpg in saved_files:
        assert jpg.stat().st_size > 1000, f"{jpg.name} is suspiciously small"


# --- Reconstruction analysis ---


def _make_discontinuous_recon(sfmr_path, *, translate=None, rotate_deg=None):
    """Load a reconstruction and introduce a pose discontinuity at image 10->11.

    Shifts or rotates the poses of images 11-17 to create an artificial break.
    Returns the modified reconstruction (not saved to disk).
    """
    from sfmtool._sfmtool import RotQuaternion, Se3Transform, SfmrReconstruction

    recon = SfmrReconstruction.load(sfmr_path)
    image_names = recon.image_names
    quats = recon.quaternions_wxyz.copy()
    trans = recon.translations.copy()

    # Build Se3Transform for the perturbation
    if translate is not None:
        xform = Se3Transform(translation=np.array(translate, dtype=np.float64))
    elif rotate_deg is not None:
        xform = Se3Transform.from_axis_angle(
            np.array([0.0, 0.0, 1.0]), np.radians(rotate_deg)
        )
    else:
        raise ValueError("must specify translate or rotate_deg")

    # Apply to images whose number is >= 11
    from sfmtool._filenames import number_from_filename

    for i in range(recon.image_count):
        num = number_from_filename(image_names[i])
        if num is not None and num >= 11:
            # Transform the world-to-camera pose: new_pose = pose @ xform_inv
            # Equivalently, transform camera center in world space.
            q = RotQuaternion.from_wxyz_array(quats[i])
            center = q.camera_center(trans[i])

            # Apply world-space transform to center and rotation
            new_center = xform.apply_to_point(center)
            new_q = xform.rotation @ q

            # Convert back to world-to-camera (t = -R @ C)
            R = np.array(new_q.to_rotation_matrix())
            new_t = -R @ new_center
            quats[i] = new_q.to_wxyz_array()
            trans[i] = new_t

    return recon.clone_with_changes(quaternions_wxyz=quats, translations=trans)


def test_recon_no_discontinuity(sfmrfile_reconstruction_with_17_images):
    """Unmodified reconstruction has no discontinuities."""
    from sfmtool._discontinuity import analyze_reconstruction
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    results = analyze_reconstruction(recon)
    assert len(results) == 1
    assert len(results[0]["core_edges"]) == 0


def test_recon_translation_discontinuity(sfmrfile_reconstruction_with_17_images):
    """A large translation applied to images 11-17 creates a discontinuity
    at the 10->11 edge."""
    from sfmtool._discontinuity import analyze_reconstruction

    recon = _make_discontinuous_recon(
        sfmrfile_reconstruction_with_17_images,
        translate=[50.0, 0.0, 0.0],
    )
    results = analyze_reconstruction(recon)
    assert len(results) == 1

    core_edges = results[0]["core_edges"]
    assert len(core_edges) > 0

    # The core edge should be at seq frame 10->11
    core_frame_pairs = set()
    seq_frm = results[0]["seq_frame_numbers"]
    for a, b in core_edges:
        core_frame_pairs.add((seq_frm[a], seq_frm[b]))
    assert (10, 11) in core_frame_pairs

    # Should have translation evidence
    edge_10_11 = None
    for (a, b), evidence in core_edges.items():
        if seq_frm[a] == 10 and seq_frm[b] == 11:
            edge_10_11 = evidence
    assert edge_10_11 is not None
    assert any(".t" in e for e in edge_10_11)


def test_recon_rotation_discontinuity(sfmrfile_reconstruction_with_17_images):
    """A large rotation applied to images 11-17 creates a discontinuity
    at the 10->11 edge."""
    from sfmtool._discontinuity import analyze_reconstruction

    recon = _make_discontinuous_recon(
        sfmrfile_reconstruction_with_17_images,
        rotate_deg=90.0,
    )
    results = analyze_reconstruction(recon)
    assert len(results) == 1

    core_edges = results[0]["core_edges"]
    assert len(core_edges) > 0

    seq_frm = results[0]["seq_frame_numbers"]
    core_frame_pairs = set()
    for a, b in core_edges:
        core_frame_pairs.add((seq_frm[a], seq_frm[b]))
    assert (10, 11) in core_frame_pairs

    # Should have rotation evidence
    edge_10_11 = None
    for (a, b), evidence in core_edges.items():
        if seq_frm[a] == 10 and seq_frm[b] == 11:
            edge_10_11 = evidence
    assert edge_10_11 is not None
    assert any(".r" in e for e in edge_10_11)


def test_recon_cli_with_sfmr(runner, sfmrfile_reconstruction_with_17_images):
    """The CLI accepts a .sfmr file and produces reconstruction analysis output."""
    result = runner.invoke(
        main,
        ["discontinuity", str(sfmrfile_reconstruction_with_17_images)],
    )
    assert result.exit_code == 0, result.output
    assert "Reconstruction:" in result.output
    assert "Summary" in result.output
    assert "seoul_bull_sculpture" in result.output


def test_recon_cli_with_range(runner, sfmrfile_reconstruction_with_17_images):
    """The CLI --range flag filters images in reconstruction mode."""
    result = runner.invoke(
        main,
        [
            "discontinuity",
            str(sfmrfile_reconstruction_with_17_images),
            "-r",
            "1-10",
        ],
    )
    assert result.exit_code == 0, result.output
    # Should only process frames 1-10, not enough for both L and R
    # (need 4+ frames, which we have), but should not mention frame 17
    assert "Reconstruction:" in result.output
    assert "seoul_bull_sculpture_17" not in result.output
