# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm estimate-intrinsics` CLI command.

The end-to-end test drives the real pipeline the way a user does (`sfm match
--cluster` on the checked-in seoul bull images, then the vote). Everything the
vote's own numbers are not needed for -- how the report renders the kernel's
confirmation, the FOV maps, the pattern derivation, the `--write-camrig`
refusals -- is covered against synthetic `estimate_intrinsics` results, so
those cases stay cheap and deterministic.
"""

import importlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import sfmtool._sfmtool.geometry as _geometry
from sfmtool.cli import main

# The package re-exports the click command under the module's own name, so the
# module itself is reached through the import machinery.
ei = importlib.import_module("sfmtool._commands.estimate_intrinsics")


# ── Synthetic estimates ───────────────────────────────────────────────────


def _column(camera_model: str, focal_px, *, epipolar=4, rotation=0) -> dict:
    """One `columns` entry, with only the fields the command reads populated."""
    return {
        "camera_model": camera_model,
        "focal_px": focal_px,
        "pool_spread": 0.05,
        "n_certified_epipolar": epipolar,
        "n_certified_rotation": rotation,
    }


def _estimate(
    camera_model="Pinhole",
    focal_px=300.0,
    columns=None,
    n_pool=6,
    confirmed=None,
    verdict_votes=None,
) -> dict:
    """An `estimate_intrinsics` result dict, with its vote nested under `vote`.

    The verdict fields are the kernel's: the command reads them rather than
    re-deriving them, so a test that wants an UNCONFIRMED report says so here.
    """
    return {
        "camera_model": camera_model,
        "confirmed": confirmed,
        "focal_px": focal_px,
        "verdict_votes": verdict_votes or [],
        "vote": _vote_result(camera_model, focal_px, columns, n_pool),
    }


def _vote_result(
    camera_model="Pinhole",
    focal_px=300.0,
    columns=None,
    n_pool=6,
) -> dict:
    """A `focal_vote` result dict with the keys the command consumes."""
    return {
        "focal_px": focal_px,
        "family": "Epipolar",
        "epipolar_focal_px": focal_px,
        "rotation_focal_px": None,
        "n_epipolar": n_pool,
        "n_rotation": 0,
        "n_pool": n_pool,
        "pool_spread": 0.05,
        "family_disagreement": None,
        "parallax_poverty": 0.1,
        "epipolar_spread": 0.05,
        "rotation_spread": 0.0,
        "epipolar_votes": [],
        "rotation_votes": [],
        "n_h_dominated": 1,
        "n_estimator_failed": 2,
        "n_band_rejected": 3,
        "n_degenerate": 0,
        "n_inconsistent_pairs": 0,
        "camera_model": camera_model,
        "columns": [] if columns is None else columns,
    }


def _observations(width=640, height=480, image_names=None) -> dict:
    """A `_load_observations` payload for a two-image, one-cluster capture."""
    return {
        "image_names": image_names or ["images/a.jpg", "images/b.jpg"],
        "width": width,
        "height": height,
        "cluster_indexes": np.zeros(2, dtype=np.uint32),
        "image_indexes": np.array([0, 1], dtype=np.uint32),
        "positions": np.zeros((2, 2), dtype=np.float64),
        "cluster_count": 1,
    }


@pytest.fixture
def stub_estimate(monkeypatch, tmp_path):
    """Run the command against a canned estimate and no real `.matches`.

    Returns a callable taking the estimate dict (and optionally the observation
    payload) plus CLI arguments, and giving back the `CliRunner` result. The
    calls the command made are recorded on the callable's `calls` list.
    """
    matches_path = tmp_path / "clusters.matches"
    matches_path.write_bytes(b"not read")
    calls: list[dict] = []

    def run(result: dict, *args: str, data: dict | None = None):
        monkeypatch.setattr(
            ei, "_load_observations", lambda path: data or _observations()
        )

        def stub(*a, **k):
            calls.append(k)
            return dict(result)

        monkeypatch.setattr(_geometry, "estimate_intrinsics", stub)
        return CliRunner().invoke(
            main,
            ["estimate-intrinsics", "-i", str(matches_path), *args],
        )

    run.calls = calls
    return run


# ── The confirmation verdict, as the report renders it ───────────────────────
#
# Whether a fisheye verdict is corroborated is the kernel's rule, pinned in
# `crates/sfmtool-core/src/geometry/estimate_intrinsics/tests.rs`. What the
# command owns is reading that answer and saying so.


@pytest.mark.parametrize(
    ("confirmed", "expected", "absent"),
    [
        (True, "CONFIRMED", "UNCONFIRMED"),
        (False, "UNCONFIRMED", None),
        (None, "Camera model:  EquidistantFisheye", "CONFIRMED"),
    ],
)
def test_report_marks_the_kernel_s_confirmation(
    stub_estimate, confirmed, expected, absent
):
    result = _estimate(
        camera_model="EquidistantFisheye",
        focal_px=131.0,
        columns=[
            _column("Pinhole", 210.0),
            _column("EquidistantFisheye", 131.0, epipolar=9, rotation=4),
        ],
        confirmed=confirmed,
    )
    out = stub_estimate(result)
    assert out.exit_code == 0, out.output
    assert expected in out.output
    if absent is not None:
        assert absent not in out.output


@pytest.mark.parametrize(
    ("model_option", "expected_columns"),
    [
        ("auto", ["pinhole", "equidistant"]),
        ("pinhole", ["pinhole"]),
        ("fisheye", ["equidistant"]),
    ],
)
def test_model_option_selects_the_columns(
    stub_estimate, model_option, expected_columns
):
    out = stub_estimate(_estimate(), "--model", model_option)
    assert out.exit_code == 0, out.output
    assert stub_estimate.calls[-1]["columns"] == expected_columns


def test_unconfirmed_fisheye_report_recommends_pinhole(stub_estimate):
    result = _estimate(
        camera_model="EquidistantFisheye",
        focal_px=131.0,
        columns=[
            _column("Pinhole", 300.0),
            _column("EquidistantFisheye", 131.0, rotation=0),
        ],
        confirmed=False,
    )
    out = stub_estimate(result)
    assert out.exit_code == 0, out.output
    assert "UNCONFIRMED" in out.output
    assert "--model pinhole" in out.output


# ── Derived report quantities ────────────────────────────────────────────────


def test_diagonal_fov_uses_each_model_map():
    width, height, focal = 640, 480, 400.0
    r_corner = math.hypot(width, height) / 2.0
    pinhole = ei._diagonal_fov_deg("Pinhole", focal, width, height)
    fisheye = ei._diagonal_fov_deg("EquidistantFisheye", focal, width, height)
    assert pinhole == pytest.approx(math.degrees(2 * math.atan(r_corner / focal)))
    assert fisheye == pytest.approx(math.degrees(2 * r_corner / focal))
    # The equidistant map opens faster than the pinhole one at the same focal.
    assert fisheye > pinhole


def test_diagonal_fov_is_none_without_a_focal_or_verdict():
    assert ei._diagonal_fov_deg("Pinhole", None, 640, 480) is None
    assert ei._diagonal_fov_deg(None, 400.0, 640, 480) is None


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["images/a.jpg", "images/b.jpg"], "images/*.jpg"),
        (["a.jpg", "b.jpg"], "*.jpg"),
        (["rig/cam0/a.jpg", "rig/cam1/b.jpg"], "rig/*/*.jpg"),
        (["images/a.jpg", "images/b.png"], "images/*"),
        (["a.jpg", "b"], "*"),
        (["cam0/a.jpg", "cam1/b.jpg"], "*/*.jpg"),
        (["rig/deep/cam0/a.jpg", "rig/deep/cam1/b.jpg"], "rig/deep/*/*.jpg"),
        (["images/a.jpg", "images/cam1/b.jpg"], "images/*.jpg"),
    ],
)
def test_pattern_derives_from_the_common_image_directory(names, expected):
    assert ei._derive_pattern(names) == expected


def test_no_consensus_reports_the_rejection_counters(stub_estimate):
    result = _estimate(camera_model="Pinhole", focal_px=None, n_pool=1)
    out = stub_estimate(result)
    # The report is the product: a starved capture is not an error.
    assert out.exit_code == 0, out.output
    assert "no consensus" in out.output
    assert "h-dominated 1" in out.output
    assert "estimator failed 2" in out.output
    assert "out of band 3" in out.output


# ── Input rejection ──────────────────────────────────────────────────────────


def test_mixed_image_dimensions_are_rejected():
    import click

    dims = np.array([[640, 480], [640, 480], [1024, 768]], dtype=np.uint32)
    names = ["a.jpg", "b.jpg", "c.jpg"]
    with pytest.raises(click.UsageError) as excinfo:
        ei._resolve_dimensions(dims, names)
    message = str(excinfo.value)
    assert "640x480" in message
    assert "1024x768" in message
    assert "c.jpg" in message


def test_single_image_dimension_is_accepted():
    dims = np.array([[640, 480], [640, 480]], dtype=np.uint32)
    assert ei._resolve_dimensions(dims, ["a.jpg", "b.jpg"]) == (640, 480)


# ── --json ───────────────────────────────────────────────────────────────────


def test_json_is_serializable_and_carries_the_derived_fields(stub_estimate):
    result = _estimate(
        camera_model="EquidistantFisheye",
        focal_px=131.0,
        columns=[
            _column("Pinhole", 300.0),
            _column("EquidistantFisheye", 131.0, rotation=4),
        ],
        confirmed=True,
        verdict_votes=[{"cell": "Rotation", "focal_px": 131.0, "certified": True}],
    )
    out = stub_estimate(result, "--json")
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    # The binding's dict, verbatim.
    assert payload["camera_model"] == "EquidistantFisheye"
    assert payload["focal_px"] == pytest.approx(131.0)
    assert [c["camera_model"] for c in payload["columns"]] == [
        "Pinhole",
        "EquidistantFisheye",
    ]
    # Plus what the command derives.
    assert payload["fisheye_confirmed"] is True
    assert payload["certified_rotation_mass"] == 4
    assert payload["diagonal_fov_deg"] == pytest.approx(
        ei._diagonal_fov_deg("EquidistantFisheye", 131.0, 640, 480)
    )
    # The evidence behind THIS verdict, which the flat vote lists are not.
    assert payload["verdict_votes"] == [
        {"cell": "Rotation", "focal_px": 131.0, "certified": True}
    ]


# ── --write-camrig ───────────────────────────────────────────────────────────


@pytest.fixture
def camrig_root(tmp_path) -> Path:
    """A directory holding one image, so a `*.jpg` pattern validates."""
    root = tmp_path / "rig"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"jpeg")
    return root


def test_write_camrig_writes_the_verdict_model(stub_estimate, camrig_root):
    from sfmtool._sfmtool.io import read_camrig

    result = _estimate(
        camera_model="EquidistantFisheye",
        focal_px=131.0,
        columns=[
            _column("Pinhole", 300.0),
            _column("EquidistantFisheye", 131.0, rotation=4),
        ],
        confirmed=True,
    )
    out_path = camrig_root / "rig.camrig"
    out = stub_estimate(result, "--write-camrig", str(out_path), "--pattern", "*.jpg")
    assert out.exit_code == 0, out.output
    assert out_path.exists()

    rig = read_camrig(out_path)
    (camera,) = rig["cameras"]
    assert camera["model"] == "EQUIDISTANT_FISHEYE"
    assert (camera["width"], camera["height"]) == (640, 480)
    assert camera["parameters"]["focal_length"] == pytest.approx(131.0)
    # The vote's principal point is the image centre, so the rig's is too.
    assert camera["parameters"]["principal_point_x"] == pytest.approx(320.0)
    assert camera["parameters"]["principal_point_y"] == pytest.approx(240.0)
    assert list(rig["sensor_image_patterns"]) == ["*.jpg"]


def test_write_camrig_refuses_without_a_consensus(stub_estimate, camrig_root):
    result = _estimate(camera_model="Pinhole", focal_px=None, n_pool=0)
    out_path = camrig_root / "rig.camrig"
    out = stub_estimate(result, "--write-camrig", str(out_path), "--pattern", "*.jpg")
    assert out.exit_code != 0
    # The report still printed; only the write failed.
    assert "no consensus" in out.output
    assert "no focal consensus to write" in out.output
    assert not out_path.exists()


def test_write_camrig_refuses_an_unconfirmed_fisheye(stub_estimate, camrig_root):
    result = _estimate(
        camera_model="EquidistantFisheye",
        focal_px=131.0,
        columns=[
            _column("Pinhole", 300.0),
            _column("EquidistantFisheye", 131.0, rotation=0),
        ],
        confirmed=False,
    )
    out_path = camrig_root / "rig.camrig"
    out = stub_estimate(result, "--write-camrig", str(out_path), "--pattern", "*.jpg")
    assert out.exit_code != 0
    assert "UNCONFIRMED" in out.output
    assert "--model pinhole" in out.output
    assert not out_path.exists()


def test_write_camrig_refuses_to_overwrite_without_force(stub_estimate, camrig_root):
    result = _estimate(camera_model="Pinhole", focal_px=300.0)
    out_path = camrig_root / "rig.camrig"
    out_path.write_bytes(b"existing")

    out = stub_estimate(result, "--write-camrig", str(out_path), "--pattern", "*.jpg")
    assert out.exit_code != 0
    assert "already exists" in out.output
    assert out_path.read_bytes() == b"existing"

    out = stub_estimate(
        result, "--write-camrig", str(out_path), "--pattern", "*.jpg", "--force"
    )
    assert out.exit_code == 0, out.output
    assert out_path.read_bytes() != b"existing"


def test_write_camrig_rejects_a_pattern_that_matches_no_images(
    stub_estimate, camrig_root
):
    result = _estimate(camera_model="Pinhole", focal_px=300.0)
    out_path = camrig_root / "rig.camrig"
    out = stub_estimate(result, "--write-camrig", str(out_path), "--pattern", "*.png")
    assert out.exit_code != 0
    assert "--pattern" in out.output
    assert not out_path.exists()


# ── End to end ───────────────────────────────────────────────────────────────


@pytest.fixture
def cluster_matches_file(isolated_seoul_bull_17_images) -> Path:
    """A cluster-bearing .matches file from `sfm match --cluster`.

    The same construction `tests/patch/test_cluster_patches.py` uses: the input
    is produced the way users produce it, so the command is exercised against a
    real cluster backbone with no `cluster_patches/` enrichment (positions come
    from the `.sift` files the matcher indexed).
    """
    workspace_dir = isolated_seoul_bull_17_images[0].parent
    out = workspace_dir / "matches" / "clusters.matches"

    runner = CliRunner()
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        main,
        [
            "match",
            "--cluster",
            "--clusters-output",
            str(out),
            "--output",
            str(workspace_dir / "tvg-matches" / "verified.matches"),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    return out


def test_estimate_intrinsics_end_to_end(cluster_matches_file: Path):
    """The seoul bull capture is a 270x480 pinhole one; the vote should say so."""
    from sfmtool._sfmtool.io import read_camrig

    runner = CliRunner()
    out = runner.invoke(main, ["estimate-intrinsics", "-i", str(cluster_matches_file)])
    assert out.exit_code == 0, out.output
    assert "17 @ 270x480" in out.output
    assert "Camera model:  Pinhole (SIMPLE_PINHOLE)" in out.output
    assert "UNCONFIRMED" not in out.output
    # Both columns ran and are reported.
    assert "EquidistantFisheye" in out.output

    focal = float(re.search(r"Focal length:\s+([\d.]+) px", out.output).group(1))
    # Not pinned to a value, and bounded only by the kernel's own plausibility
    # band (0.3x to 3x the max dimension): the fixture's matches come from a
    # fresh SIFT extraction, whose floating point differs across platforms,
    # and on this capture's small, wide vote pool that moves the consensus by
    # tens of pixels (Linux reads ~210 where Windows reads ~277).
    assert 0.3 * 480 < focal < 3.0 * 480

    # The same run as JSON.
    out = runner.invoke(
        main, ["estimate-intrinsics", "-i", str(cluster_matches_file), "--json"]
    )
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert payload["camera_model"] == "Pinhole"
    assert payload["focal_px"] == pytest.approx(focal, abs=0.01)
    assert payload["fisheye_confirmed"] is None
    assert payload["n_pool"] >= 2

    # Committing the estimate to a rig.
    workspace_dir = cluster_matches_file.parent.parent
    rig_path = workspace_dir / "seoul_bull.camrig"
    out = runner.invoke(
        main,
        [
            "estimate-intrinsics",
            "-i",
            str(cluster_matches_file),
            "--write-camrig",
            str(rig_path),
            "--pattern",
            "*.jpg",
        ],
    )
    assert out.exit_code == 0, out.output
    rig = read_camrig(rig_path)
    (camera,) = rig["cameras"]
    assert camera["model"] == "SIMPLE_PINHOLE"
    assert (camera["width"], camera["height"]) == (270, 480)
    assert camera["parameters"]["focal_length"] == pytest.approx(focal, abs=0.01)


def test_estimate_intrinsics_named_model_runs_one_column(cluster_matches_file: Path):
    out = CliRunner().invoke(
        main,
        ["estimate-intrinsics", "-i", str(cluster_matches_file), "--model", "fisheye"],
    )
    assert out.exit_code == 0, out.output
    assert "Camera model:  EquidistantFisheye (EQUIDISTANT_FISHEYE)" in out.output
    # One column ran, so nothing arbitrated and nothing is marked confirmed.
    assert "Pinhole" not in out.output
    assert "CONFIRMED" not in out.output
