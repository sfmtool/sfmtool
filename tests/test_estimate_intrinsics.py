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
import numpy.testing as npt
import pytest
from click.testing import CliRunner

import sfmtool._sfmtool.geometry as _geometry
from sfmtool.cli import main

# The package re-exports the click command under the module's own name, so the
# module itself is reached through the import machinery.
ei = importlib.import_module("sfmtool._commands.estimate_intrinsics")

# No `-r`: the selection `_load_cameras` gets when the command is run plainly.
_NO_SELECTION = ei.parse_image_selection((), None)


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
    escalation=None,
    screening_vote=None,
) -> dict:
    """An `estimate_intrinsics` result dict, with its vote nested under `vote`.

    The verdict fields are the kernel's: the command reads them rather than
    re-deriving them, so a test that wants an UNCONFIRMED report says so here.
    `escalation` is likewise the kernel's record of whether it ran the
    camera-model columns -- `None` for a named `--model`, a (possibly empty)
    list of reason names under `--model auto`.
    """
    return {
        "camera_model": camera_model,
        "confirmed": confirmed,
        "focal_px": focal_px,
        "verdict_votes": verdict_votes or [],
        "escalation": escalation,
        "screening_vote": screening_vote,
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
    """The report payload `_load_cameras` returns beside the selection, for a
    two-image, one-cluster capture with no `-r`."""
    names = image_names or ["images/a.jpg", "images/b.jpg"]
    return {
        "matches_path": Path("clusters.matches"),
        "image_names": names,
        "file_image_names": names,
        "selection_label": None,
        "workspace_dir": None,
        "width": width,
        "height": height,
        "cluster_count": 1,
        "observation_count": 2,
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
        # The command hands the selection handle straight to the kernel, so a
        # canned run needs no `.matches` at all: the stub estimator ignores it.
        monkeypatch.setattr(
            ei,
            "_load_cameras",
            lambda path, image_selection: [(None, None, data or _observations())],
        )

        def stub(*a, **k):
            calls.append(k)
            return dict(result)

        # The command imports the kernel from the extension submodule when it
        # runs, so the stub replaces it there, not on `sfmtool.geometry`.
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
        # `auto` is the kernel's escalating policy, not a fixed pair of
        # columns: it votes pinhole-only and adds the second column itself.
        ("auto", "auto"),
        ("pinhole", ("pinhole",)),
        ("fisheye", ("equidistant",)),
    ],
)
def test_model_option_selects_the_columns(
    stub_estimate, model_option, expected_columns
):
    out = stub_estimate(_estimate(), "--model", model_option)
    assert out.exit_code == 0, out.output
    assert stub_estimate.calls[-1]["columns"] == expected_columns


def test_draws_default_to_one_and_pass_through(stub_estimate):
    out = stub_estimate(_estimate())
    assert out.exit_code == 0, out.output
    assert stub_estimate.calls[-1]["draws"] == 1

    out = stub_estimate(_estimate(), "--draws", "5", "--seed", "7")
    assert out.exit_code == 0, out.output
    assert stub_estimate.calls[-1]["draws"] == 5
    assert stub_estimate.calls[-1]["seed"] == 7


@pytest.mark.parametrize("draws", ["0", "-3"])
def test_draws_below_one_are_refused(stub_estimate, draws):
    out = stub_estimate(_estimate(), "--draws", draws)
    assert out.exit_code == 2
    assert "--draws" in out.output
    # Click refuses the value before the kernel is called.
    assert stub_estimate.calls == []


# ── Whether the camera-model columns ran ─────────────────────────────────────
#
# Under `--model auto` the kernel decides, so the report says which way it went
# rather than leaving the reader to infer it from a missing Columns block.


def test_report_states_the_escalation_that_ran(stub_estimate):
    result = _estimate(
        camera_model="EquidistantFisheye",
        focal_px=131.0,
        columns=[
            _column("Pinhole", 210.0),
            _column("EquidistantFisheye", 131.0, epipolar=9, rotation=4),
        ],
        confirmed=True,
        escalation=["thin_pool", "family_disagreement"],
    )
    out = stub_estimate(result)
    assert out.exit_code == 0, out.output
    assert "the camera-model columns ran" in out.output
    assert "thin_pool, family_disagreement" in out.output


def test_report_states_an_escalation_that_did_not_run(stub_estimate):
    out = stub_estimate(_estimate(escalation=[]))
    assert out.exit_code == 0, out.output
    assert "Arbitration:   not run" in out.output
    # No second run means no columns to report.
    assert "Columns:" not in out.output


def test_report_is_silent_about_arbitration_for_a_named_model(stub_estimate):
    out = stub_estimate(_estimate(), "--model", "pinhole")
    assert out.exit_code == 0, out.output
    assert "Arbitration:" not in out.output


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


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        # The frame number's shared leading digits are dropped with the rest.
        (["images/shopA_00001.jpg", "images/shopA_01089.jpg"], "images/shopA_*.jpg"),
        (["images/x12.jpg", "images/x13.jpg"], "images/x*.jpg"),
        # No shared prefix leaves the plain pattern.
        (["images/a_01.jpg", "images/b_01.jpg"], "images/*.jpg"),
        (["rig/cam0/f_01.jpg", "rig/cam1/f_02.jpg"], "rig/*/f_*.jpg"),
        # One frame of a rig: the same name in each sensor's folder. The
        # prefix is taken over the stems, so the extension is not in it.
        (["rig/cam0/f_01.jpg", "rig/cam1/f_01.jpg"], "rig/*/f_*.jpg"),
        # `[` would be read as a wildcard, so the prefix stops before it.
        (["images/x[1]_01.jpg", "images/x[1]_02.jpg"], "images/x*.jpg"),
        # A literal `%` is written `%%`, not read as a frame field.
        (["images/a%d_01.jpg", "images/a%d_02.jpg"], "images/a%%d_*.jpg"),
    ],
)
def test_narrowed_pattern_takes_the_names_common_prefix(names, expected, tmp_path):
    from sfmtool.camrig.pattern import match_pattern

    pattern = ei._derive_pattern(names, name_prefix=True)
    assert pattern == expected
    # The narrowed pattern still finds every file it was derived from, by the
    # walk `--write-camrig` and `sfm solve` expand a pattern with.
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"jpeg")
    found = {
        p.relative_to(tmp_path.resolve()).as_posix()
        for p in match_pattern(tmp_path, pattern)
    }
    assert found == set(names)


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
    # No escalation ran, so there is no screening vote to report.
    assert payload["screening_vote"] is None


def test_json_carries_the_screening_vote_of_an_escalated_run(stub_estimate):
    screening = _vote_result("Pinhole", 95.0, n_pool=9)
    result = _estimate(
        camera_model="EquidistantFisheye",
        focal_px=131.0,
        columns=[
            _column("Pinhole", 300.0),
            _column("EquidistantFisheye", 131.0, rotation=4),
        ],
        confirmed=True,
        escalation=["rotation_railed", "thin_pool"],
        screening_vote=screening,
    )
    out = stub_estimate(result, "--json")
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    # The top level is the escalated run; the pinhole-only numbers the
    # escalation was decided on are under `screening_vote`.
    assert payload["focal_px"] == pytest.approx(131.0)
    assert payload["escalation"] == ["rotation_railed", "thin_pool"]
    assert payload["screening_vote"] == screening


# ── --write-camrig ───────────────────────────────────────────────────────────


@pytest.fixture
def camrig_root(tmp_path) -> Path:
    """A directory holding one image, so a `*.jpg` pattern validates."""
    root = tmp_path / "rig"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"jpeg")
    return root


def test_write_camrig_writes_the_verdict_model(stub_estimate, camrig_root):
    from sfmtool.fileio import read_camrig

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


def test_json_write_camrig_keeps_stdout_json(stub_estimate, camrig_root):
    """Under --json the summary of the write goes to stderr, so a script can
    parse stdout whole."""
    result = _estimate(camera_model="Pinhole", focal_px=300.0)
    out_path = camrig_root / "rig.camrig"
    out = stub_estimate(
        result, "--json", "--write-camrig", str(out_path), "--pattern", "*.jpg"
    )
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["focal_px"] == pytest.approx(300.0)
    assert f"Wrote {out_path}" in out.stderr
    assert out_path.exists()


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


# ── --write-camrig after -r ─────────────────────────────────────────────────


def _selected(workspace: Path, selected: list[str], file_names: list[str]) -> dict:
    """The report payload of a `-r` run that kept `selected` of a `.matches`
    file holding `file_names`, with every name created as an image file."""
    for name in file_names:
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_bytes(b"jpeg")
    return {
        **_observations(image_names=selected),
        "file_image_names": file_names,
        "selection_label": "named image list 'a'",
        "workspace_dir": workspace,
    }


def test_selected_folder_derives_a_pattern_covering_that_folder(
    stub_estimate, tmp_path
):
    from sfmtool.fileio import read_camrig

    selected = ["videoA/f_01.jpg", "videoA/f_02.jpg"]
    data = _selected(tmp_path, selected, [*selected, "videoB/f_01.jpg"])
    # A frame of the folder the file does not hold is still the same camera's.
    (tmp_path / "videoA" / "f_03.jpg").write_bytes(b"jpeg")
    out_path = tmp_path / "a.camrig"
    result = _estimate(camera_model="Pinhole", focal_px=300.0)
    out = stub_estimate(result, "--write-camrig", str(out_path), data=data)
    assert out.exit_code == 0, out.output
    assert "2 of 3 selected (named image list 'a')" in out.output
    assert list(read_camrig(out_path)["sensor_image_patterns"]) == ["videoA/*.jpg"]


def test_shared_folder_narrows_the_pattern_to_the_names_prefix(stub_estimate, tmp_path):
    from sfmtool.fileio import read_camrig

    selected = ["images/shopA_00001.jpg", "images/shopA_00010.jpg"]
    data = _selected(tmp_path, selected, [*selected, "images/shopB_00001.jpg"])
    out_path = tmp_path / "a.camrig"
    result = _estimate(camera_model="Pinhole", focal_px=300.0)
    out = stub_estimate(result, "--write-camrig", str(out_path), data=data)
    assert out.exit_code == 0, out.output
    patterns = list(read_camrig(out_path)["sensor_image_patterns"])
    assert patterns == ["images/shopA_*.jpg"]


def test_without_r_the_pattern_is_not_narrowed(stub_estimate, tmp_path):
    """No image of the file is left out, so the plain pattern stands even
    though the folder holds images of another prefix."""
    from sfmtool.fileio import read_camrig

    names = ["images/shopA_00001.jpg", "images/shopA_00010.jpg"]
    data = {**_selected(tmp_path, names, names), "selection_label": None}
    (tmp_path / "images" / "shopB_00001.jpg").write_bytes(b"jpeg")
    out_path = tmp_path / "a.camrig"
    result = _estimate(camera_model="Pinhole", focal_px=300.0)
    out = stub_estimate(result, "--write-camrig", str(out_path), data=data)
    assert out.exit_code == 0, out.output
    assert list(read_camrig(out_path)["sensor_image_patterns"]) == ["images/*.jpg"]


def test_a_pattern_covering_left_out_images_is_refused(stub_estimate, tmp_path):
    # The selection takes images of both prefixes, so narrowing to the common
    # prefix "shop" still covers the left-out shopA_00002.
    selected = ["images/shopA_00001.jpg", "images/shopB_00001.jpg"]
    data = _selected(tmp_path, selected, [*selected, "images/shopA_00002.jpg"])
    out_path = tmp_path / "a.camrig"
    result = _estimate(camera_model="Pinhole", focal_px=300.0)
    out = stub_estimate(result, "--write-camrig", str(out_path), data=data)
    assert out.exit_code != 0
    assert "Focal length:  300.00 px" in out.output
    assert "'images/shop*.jpg' matches 1 image(s)" in out.output
    assert "images/shopA_00002.jpg" in out.output
    assert "--pattern" in out.output
    assert not out_path.exists()

    # An explicit --pattern is held to the same rule ...
    out = stub_estimate(
        result, "--write-camrig", str(out_path), "--pattern", "images/*", data=data
    )
    assert out.exit_code != 0
    assert "images/shopA_00002.jpg" in out.output
    assert not out_path.exists()

    # ... and written when it names only selected images.
    out = stub_estimate(
        result,
        "--write-camrig",
        str(out_path),
        "--pattern",
        "images/shop*_00001.jpg",
        data=data,
    )
    assert out.exit_code == 0, out.output
    assert out_path.exists()


# ── Images of several cameras ────────────────────────────────────────────────


def _pinhole_camera() -> dict:
    return {
        "model": "SIMPLE_PINHOLE",
        "width": 640,
        "height": 480,
        "parameters": {
            "focal_length": 500.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
        },
    }


@pytest.fixture
def cameras_run(monkeypatch, tmp_path):
    """Run the command plainly on a workspace of image files named `names`,
    with no real `.matches` file: the stub kernel gives each selection the
    focal `focal_for` maps its images to, and records the images of each
    selection it was handed on the callable's `selections` list.

    The workspace is `tmp_path / "ws"`, and its `.camrig` and
    `camera_config.json` files are the test's to write.
    """
    from sfmtool.fileio import write_camrig

    workspace = tmp_path / "ws"
    workspace.mkdir()
    matches_path = tmp_path / "clusters.matches"
    matches_path.write_bytes(b"not read")
    selections: list[list[str]] = []

    def run(names, *args, focal_for=lambda chosen: 300.0):
        for name in names:
            (workspace / name).parent.mkdir(parents=True, exist_ok=True)
            (workspace / name).write_bytes(b"jpeg")
        monkeypatch.setattr(
            ei,
            "_open_matches",
            lambda path, selection: (None, list(names), None, None, workspace),
        )

        def select(mfile, restrict, data):
            chosen = list(names) if restrict is None else list(restrict)
            return tuple(chosen), {
                **data,
                "image_names": chosen,
                "width": 640,
                "height": 480,
                "cluster_count": 1,
                "observation_count": 2,
            }

        def stub(selection, **k):
            selections.append(list(selection))
            return _estimate(focal_px=focal_for(list(selection)))

        monkeypatch.setattr(ei, "_select", select)
        monkeypatch.setattr(_geometry, "estimate_intrinsics", stub)
        return CliRunner().invoke(
            main, ["estimate-intrinsics", "-i", str(matches_path), *args]
        )

    def rig(path: Path, patterns: list[str]) -> None:
        """Write a `.camrig` whose sensors match `patterns`, sharing one camera."""
        path.parent.mkdir(parents=True, exist_ok=True)
        write_camrig(
            path=str(path),
            name=path.stem,
            rig_type="generic",
            cameras=[_pinhole_camera()],
            sensor_image_patterns=patterns,
            camera_indexes=[0] * len(patterns),
            quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (len(patterns), 1)),
            translations_xyz=np.zeros((len(patterns), 3)),
        )

    run.workspace = workspace
    run.selections = selections
    run.rig = rig
    return run


def test_the_workspace_s_cameras_are_estimated_apart(cameras_run):
    """Each sensor of a rig, each one-sensor `.camrig`, each
    `camera_config.json` and the images none of them covers are a camera
    each, as `sfm solve` gives them: each is voted on its own images and
    reported on its own, and no vote pools two of them."""
    ws = cameras_run.workspace
    names = [
        "left/f_01.jpg",
        "left/f_02.jpg",
        "right/f_01.jpg",
        "right/f_02.jpg",
        "solo/s_01.jpg",
        "solo/s_02.jpg",
        "gopro/g_01.jpg",
        "gopro/g_02.jpg",
        "phone/p_01.jpg",
        "phone/p_02.jpg",
    ]
    cameras_run.rig(ws / "rig.camrig", ["left/f_%02d.jpg", "right/f_%02d.jpg"])
    cameras_run.rig(ws / "solo" / "solo.camrig", ["s_%02d.jpg"])
    (ws / "gopro").mkdir()
    (ws / "gopro" / "camera_config.json").write_text(
        json.dumps({"version": 1, "camera_intrinsics": _pinhole_camera()})
    )
    focal = {"left": 410.0, "right": 420.0, "solo": 430.0, "gopro": 440.0}

    def focal_for(chosen):
        return focal.get(chosen[0].split("/")[0], 450.0)

    out = cameras_run(names, "--json", focal_for=focal_for)
    assert out.exit_code == 0, out.output
    assert sorted(cameras_run.selections) == sorted(
        [names[i : i + 2] for i in range(0, len(names), 2)]
    )
    payload = json.loads(out.stdout)
    assert payload["image_count"] == payload["selected_image_count"] == 10
    assert [
        (c["camera"], c["estimate"]["selected_image_count"], c["estimate"]["focal_px"])
        for c in payload["cameras"]
    ] == [
        ("rig.camrig sensor 0 (left/f_%02d.jpg)", 2, 410.0),
        ("rig.camrig sensor 1 (right/f_%02d.jpg)", 2, 420.0),
        ("solo/solo.camrig", 2, 430.0),
        ("gopro/camera_config.json", 2, 440.0),
        ("no .camrig or camera_config.json (EXIF)", 2, 450.0),
    ]

    out = cameras_run(names, focal_for=focal_for)
    assert out.exit_code == 0, out.output
    assert out.output.startswith("5 cameras among the selected images of ")
    assert "Camera rig.camrig sensor 1 (right/f_%02d.jpg) (2 images)\n" in out.output
    assert "  Focal length:  420.00 px\n" in out.output


def test_write_camrig_refuses_images_of_several_cameras(cameras_run):
    """One `.camrig` holds one camera, so a selection of two is refused before
    any vote, naming both and how to select one."""
    ws = cameras_run.workspace
    cameras_run.rig(ws / "rig.camrig", ["left/f_%02d.jpg", "right/f_%02d.jpg"])
    out_path = ws / "out.camrig"
    names = ["left/f_01.jpg", "left/f_02.jpg", "right/f_01.jpg", "right/f_02.jpg"]
    out = cameras_run(names, "--write-camrig", str(out_path))
    assert out.exit_code != 0
    assert "the selected images are of 2 cameras of the workspace" in out.output
    assert "rig.camrig sensor 0 (left/f_%02d.jpg)  (2 image(s))" in out.output
    assert "Select one camera's images with -r" in out.output
    assert cameras_run.selections == []
    assert not out_path.exists()


def test_images_of_one_camera_keep_the_single_estimate(cameras_run):
    """Images all of one camera -- here all under one `.camrig` -- are the
    one vote and the one JSON object they always were."""
    ws = cameras_run.workspace
    cameras_run.rig(ws / "a.camrig", ["a/f_%02d.jpg"])
    names = ["a/f_01.jpg", "a/f_02.jpg"]
    out = cameras_run(names, "--json")
    assert out.exit_code == 0, out.output
    assert cameras_run.selections == [names]
    payload = json.loads(out.stdout)
    assert "cameras" not in payload
    assert payload["focal_px"] == 300.0
    assert payload["selected_image_count"] == 2


# ── End to end ───────────────────────────────────────────────────────────────


@pytest.fixture
def cluster_matches_file(isolated_seoul_bull_17_images) -> Path:
    """A cluster-bearing .matches file from `sfm match --cluster`.

    The same construction `tests/patch/test_cluster_patches.py` uses: the input
    is produced the way users produce it, so the command is exercised against a
    real cluster backbone with no `cluster_patches/` enrichment (positions are
    the backbone's own detections).
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
        ["match", "--cluster", "--output", str(out), str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    return out


def test_the_vote_reads_the_backbones_positions(cluster_matches_file):
    """The file states its members' positions, so the vote reads them straight
    off the selection handle -- no `.sift` file is opened, and there is no
    other path to take."""
    from sfmtool.fileio import MatchesFile

    assert not hasattr(ei, "_positions_from_sift"), (
        "the legacy .sift lookup is gone; version <= 5 cluster files are refused"
    )
    ((_camera, selection, data),) = ei._load_cameras(
        cluster_matches_file, _NO_SELECTION
    )
    positions = np.asarray(selection.member_positions(), dtype=np.float64)
    assert len(positions) == data["observation_count"] > 0
    assert np.isfinite(positions).all()

    expected = MatchesFile(cluster_matches_file).select_clusters()
    npt.assert_array_equal(
        positions, np.asarray(expected.member_positions(), dtype=np.float64)
    )


def test_the_object_form_is_the_array_form(cluster_matches_file):
    """The selection handle and its own arrays are two spellings of one call:
    the kernel reads the file's CSR index and widens its float32 positions, so
    a caller spelling that out gets bit-identical numbers."""
    from sfmtool.geometry import estimate_intrinsics as estimate

    ((_camera, selection, data),) = ei._load_cameras(
        cluster_matches_file, _NO_SELECTION
    )
    object_form = estimate(selection, seed=0, columns="auto")
    array_form = estimate(
        np.asarray(selection.cluster_starts, dtype=np.uint32),
        np.asarray(selection.member_images, dtype=np.uint32),
        np.ascontiguousarray(
            np.asarray(selection.member_positions(), dtype=np.float64)
        ),
        data["width"],
        data["height"],
        seed=0,
        columns="auto",
    )
    assert object_form == array_form


def test_the_object_form_takes_no_observation_arrays(cluster_matches_file):
    from sfmtool.geometry import estimate_intrinsics as estimate

    ((_camera, selection, data),) = ei._load_cameras(
        cluster_matches_file, _NO_SELECTION
    )
    with pytest.raises(ValueError, match="takes no observation arrays"):
        estimate(
            selection,
            np.asarray(selection.member_images, dtype=np.uint32),
            np.ascontiguousarray(
                np.asarray(selection.member_positions(), dtype=np.float64)
            ),
            data["width"],
            data["height"],
        )


def test_a_pairwise_matches_file_is_refused(cluster_matches_file: Path):
    """The vote draws its pairs from cluster tracks, so the pairwise backbone
    is refused by name rather than voted on.

    Reuses the cluster fixture's workspace, whose `.sift` files are already
    extracted, and only adds the pairwise match.
    """
    from sfmtool.geometry import estimate_intrinsics as estimate
    from sfmtool.fileio import MatchesFile

    workspace_dir = cluster_matches_file.parent.parent
    out = cluster_matches_file.parent / "pairs.matches"
    result = CliRunner().invoke(
        main, ["match", "--sequential", "--output", str(out), str(workspace_dir)]
    )
    assert result.exit_code == 0, result.output

    with pytest.raises(ValueError, match="clusters"):
        estimate(MatchesFile(out))


def test_an_unreadable_matches_file_is_a_clean_cli_error(tmp_path):
    """Whatever the reader refuses -- a version <= 5 cluster file included,
    whose refusal names `sfm match --cluster` (see the `sfmtool-matches-format` version
    tests) -- reaches the user as a message, not a traceback."""
    bad = tmp_path / "not-a-matches.matches"
    bad.write_bytes(b"PK\x03\x04not really a zip")

    result = CliRunner().invoke(main, ["estimate-intrinsics", "-i", str(bad)])
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Error" in result.output


def test_estimate_intrinsics_end_to_end(cluster_matches_file: Path):
    """The seoul bull capture is a 270x480 pinhole one; the vote should say so."""
    from sfmtool.fileio import read_camrig

    runner = CliRunner()
    out = runner.invoke(main, ["estimate-intrinsics", "-i", str(cluster_matches_file)])
    assert out.exit_code == 0, out.output
    assert "17 @ 270x480" in out.output
    assert "Camera model:  Pinhole (SIMPLE_PINHOLE)" in out.output
    assert "UNCONFIRMED" not in out.output
    report = out.output

    focal = float(re.search(r"Focal length:\s+([\d.]+) px", report).group(1))
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
    # `--model auto` on this capture: whether the pinhole vote is weak enough
    # to pay for the camera-model columns is not pinned either -- the same
    # platform-dependent pool decides it (Windows escalates on a thin pool;
    # Linux and macOS read a vote that stands). What is pinned is that the
    # kernel made the call and the text report agrees with it.
    escalation = payload["escalation"]
    assert isinstance(escalation, list)
    if escalation:
        assert "the camera-model columns ran" in report
        assert ", ".join(escalation) in report
        assert "Columns:" in report
        assert "EquidistantFisheye" in report
    else:
        assert "Arbitration:   not run" in report
        assert "Columns:" not in report
        assert "EquidistantFisheye" not in report

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


def test_estimate_intrinsics_draws_on_a_real_capture(cluster_matches_file: Path):
    """The default single-draw vote and a five-draw vote both run on the real
    capture, and a rerun with the same seed and draw count is bit-identical."""
    runner = CliRunner()

    def run(*args: str) -> dict:
        out = runner.invoke(
            main,
            ["estimate-intrinsics", "-i", str(cluster_matches_file), "--json", *args],
        )
        assert out.exit_code == 0, out.output
        return json.loads(out.output)

    several = run("--draws", "5")
    default = run()
    assert several["focal_px"] is not None
    assert default["focal_px"] is not None
    assert default == run("--seed", "0", "--draws", "1")
    assert run("--seed", "3", "--draws", "3") == run("--seed", "3", "--draws", "3")


def test_r_restricts_the_vote_to_the_selected_images(cluster_matches_file: Path):
    """`-r` takes a named image list or a range, as `sfm solve -r` does on a
    `.matches` file, and the vote runs on the file's selection restricted to
    those images: the same clusters and the same vote as a direct
    `select_clusters(restrict_images=...)` call."""
    from sfmtool.fileio import MatchesFile
    from sfmtool.geometry import estimate_intrinsics as estimate

    workspace_dir = cluster_matches_file.parent.parent
    names = [f"seoul_bull_sculpture_{n:02d}.jpg" for n in range(1, 9)]
    # Frame 99 is an image of the workspace the file does not hold.
    (workspace_dir / "seoul_bull_sculpture_99.jpg").write_bytes(
        (workspace_dir / names[0]).read_bytes()
    )
    (workspace_dir / ".sfm-named-image-lists.json").write_text(
        json.dumps(
            {
                "version": 1,
                "named_image_lists": {
                    "front": [
                        {"pattern": "seoul_bull_sculpture_%02d.jpg", "range": "1-8,99"}
                    ]
                },
            }
        )
    )
    runner = CliRunner()

    def run(*args: str):
        return runner.invoke(
            main, ["estimate-intrinsics", "-i", str(cluster_matches_file), *args]
        )

    matches = MatchesFile(cluster_matches_file)
    direct = matches.select_clusters(restrict_images=names)
    # The selection keeps the workspace of the file it came from.
    assert matches.workspace_dir is not None
    assert Path(matches.workspace_dir).resolve() == workspace_dir.resolve()
    assert direct.workspace_dir == matches.workspace_dir
    direct_starts = np.asarray(direct.cluster_starts)
    whole_starts = np.asarray(matches.select_clusters().cluster_starts)
    assert 0 < direct_starts[-1] < whole_starts[-1]

    ((_camera, selection, data),) = ei._load_cameras(
        cluster_matches_file, ei.parse_image_selection(("front",), None)
    )
    npt.assert_array_equal(np.asarray(selection.cluster_starts), direct_starts)
    assert data["image_names"] == names

    out = run("-r", "front")
    assert out.exit_code == 0, out.output
    assert "8 of 17 selected (named image list 'front') @ 270x480" in out.stdout
    assert (
        f"Clusters:      {len(direct_starts) - 1} ({direct_starts[-1]} observations)"
        in out.stdout
    )
    assert "1 of the 9 images of named image list 'front' are not in" in out.stderr

    out = run("-r", "front", "--json")
    assert out.exit_code == 0, out.output
    payload = json.loads(out.stdout)
    assert payload["image_count"] == 17
    assert payload["selected_image_count"] == 8
    vote = estimate(direct, seed=0, columns="auto")["vote"]
    for key in ("focal_px", "n_epipolar", "n_rotation", "n_pool"):
        assert payload[key] == vote[key], key
    # A range selecting the same frames is the same vote.
    out = run("-r", "1-8", "--json")
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout) == payload

    # One image leaves no cluster spanning two: no consensus, still exit 0,
    # and nothing to write.
    out = run("-r", "front[:1]")
    assert out.exit_code == 0, out.output
    assert "1 of 17 selected" in out.output
    assert "Clusters:      0 (0 observations)" in out.output
    assert "no consensus" in out.output
    rig_path = workspace_dir / "front.camrig"
    out = run("-r", "front[:1]", "--write-camrig", str(rig_path))
    assert out.exit_code != 0
    assert "no focal consensus to write" in out.output
    assert not rig_path.exists()

    # A selection holding none of the file's images, and an unknown name.
    out = run("-r", "90-95")
    assert out.exit_code != 0
    assert "excludes all 17 images" in out.output
    out = run("-r", "nosuch")
    assert out.exit_code != 0
    assert "no named image list is called 'nosuch'" in out.output


def test_each_lens_of_a_rig_is_estimated_from_its_own_images(
    isolated_kerry_park_camrig: Path,
):
    """kerry_park is two fisheye lenses under one multi-sensor `.camrig`: a
    plain run votes each lens on its own images, the vote `select_clusters`
    restricted to that lens gives, and finds a fisheye for each."""
    from sfmtool.fileio import MatchesFile
    from sfmtool.geometry import estimate_intrinsics as estimate

    workspace_dir = isolated_kerry_park_camrig
    matches_path = workspace_dir / "matches" / "clusters.matches"
    runner = CliRunner()
    for args in (
        ["ws", "init", str(workspace_dir)],
        ["sift", "--extract", str(workspace_dir)],
        ["match", "--cluster", "--output", str(matches_path), str(workspace_dir)],
    ):
        result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output

    out = runner.invoke(
        main, ["estimate-intrinsics", "-i", str(matches_path), "--json"]
    )
    assert out.exit_code == 0, out.output
    payload = json.loads(out.stdout)
    assert payload["image_count"] == payload["selected_image_count"] == 48
    matches = MatchesFile(matches_path)
    for camera, sensor, lens in zip(
        payload["cameras"], (0, 1), ("fisheye_left", "fisheye_right"), strict=True
    ):
        assert camera["camera"] == (
            f"kerry_park.camrig sensor {sensor} ({lens}/frame_%02d.jpg)"
        )
        lens_estimate = camera["estimate"]
        assert lens_estimate["selected_image_count"] == 24
        names = [n for n in matches.image_names if n.startswith(f"{lens}/")]
        own = estimate(
            matches.select_clusters(restrict_images=names), seed=0, columns="auto"
        )
        assert lens_estimate["focal_px"] == own["focal_px"]
        assert lens_estimate["camera_model"] == "EquidistantFisheye"
        assert lens_estimate["fisheye_confirmed"] is True
