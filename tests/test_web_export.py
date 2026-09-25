# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfm web-export``, run on the Seoul bull ground truth."""

import base64
import json

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool import web_export as web_export_module
from sfmtool._sfmtool.io import read_sfmr
from sfmtool.cli import main

from .conftest import SEOUL_BULL_GROUND_TRUTH


def run(*args):
    return CliRunner().invoke(main, ["web-export", str(SEOUL_BULL_GROUND_TRUTH), *args])


def load_scene(out):
    return json.loads((out / "scene.json").read_text(encoding="utf-8"))


def block_array(scene, name, dtype):
    block = base64.b64decode(scene["points_b64"])
    entry = scene["points"]["arrays"][name]
    count = entry["count"] * entry["components"]
    values = np.frombuffer(block, dtype, count, entry["offset"])
    return values.reshape(entry["count"], entry["components"])


def test_scene_json_matches_the_files_written(tmp_path):
    out = tmp_path / "site"
    result = run("-o", str(out))
    assert result.exit_code == 0, result.output

    scene = load_scene(out)
    data = read_sfmr(SEOUL_BULL_GROUND_TRUTH)
    xyzw = np.asarray(data["positions_xyzw"], dtype=np.float64)
    assert scene["format"] == 1
    assert scene["source"]["name"] == SEOUL_BULL_GROUND_TRUTH.name
    assert scene["source"]["content_xxh128"] == data["content_hash"]["content_xxh128"]
    assert scene["points"]["count"] == len(xyzw)
    assert scene["points"]["at_infinity"] == int((xyzw[:, 3] == 0).sum())
    assert len(scene["cameras"]) == len(data["image_names"])

    # Positions shift back to the file's, and directions stay as they are.
    positions = block_array(scene, "position", "<f4").astype(np.float64)
    finite = xyzw[:, 3] != 0
    back = positions[finite] + np.asarray(scene["center"])
    np.testing.assert_allclose(back, xyzw[finite, :3], atol=1e-4)
    np.testing.assert_allclose(positions[~finite], xyzw[~finite, :3], atol=1e-6)
    color_w = block_array(scene, "color_w", "u1")
    assert np.array_equal(color_w[:, 3] == 1, finite)

    # The file has patch frames and no bitmaps: they are rendered.
    patches = scene["atlases"]["patches"]
    assert scene["points"]["with_patch"] > 0
    assert patches["size"] == 24 and patches["tile"] == 26

    # Every page named is written, is the size named, and fits 4096.
    written = sorted(p.name for p in out.iterdir())
    named = []
    for atlas in scene["atlases"].values():
        for page in atlas["pages"]:
            named.append(page["file"])
            image = cv2.imread(str(out / page["file"]))
            assert image.shape[:2] == (page["height"], page["width"])
            assert page["width"] <= 4096 and page["height"] <= 4096
    assert written == sorted(["scene.json", "index.html", "web-export.js", *named])
    cells = block_array(scene, "patch_cell", "<u4")[:, 0]
    with_patch = cells != 0xFFFFFFFF
    assert with_patch.sum() == scene["points"]["with_patch"]
    assert (cells[with_patch] >> 24 < len(patches["pages"])).all()

    # The viewer files are copied in, and the help names the thumbnails.
    assert "three@0.180.0/+esm" in (out / "web-export.js").read_text(encoding="utf-8")
    assert "<!doctype html>" in (out / "index.html").read_text(encoding="utf-8")


def test_help_says_the_thumbnails_are_in_the_output():
    result = CliRunner().invoke(main, ["web-export", "--help"])
    assert result.exit_code == 0
    assert "thumbnails of the photographs are in the output" in result.output


def test_a_non_empty_directory_needs_overwrite(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    result = run("-o", str(out))
    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert sorted(p.name for p in out.iterdir()) == ["notes.txt"]

    # With --overwrite, a stale page from a larger export goes, other files stay.
    (out / "patches-7.jpg").write_bytes(b"stale")
    result = run("-o", str(out), "--overwrite")
    assert result.exit_code == 0, result.output
    assert not (out / "patches-7.jpg").exists()
    assert (out / "notes.txt").read_text() == "keep me"


def test_leaving_out_patches_and_thumbnails(tmp_path):
    out = tmp_path / "site"
    result = run(
        "-o", str(out), "--no-patches", "--no-thumbnails", "--max-points", "50"
    )
    assert result.exit_code == 0, result.output
    scene = load_scene(out)
    assert scene["atlases"] == {"patches": None, "thumbnails": None}
    assert scene["points"]["count"] == 50
    assert "source_index" in scene["points"]["arrays"]
    assert sorted(p.name for p in out.iterdir()) == [
        "index.html",
        "scene.json",
        "web-export.js",
    ]


def test_start_image(tmp_path):
    name = read_sfmr(SEOUL_BULL_GROUND_TRUTH)["image_names"][3]
    out = tmp_path / "site"
    result = run("-o", str(out), "--start-image", name)
    assert result.exit_code == 0, result.output
    scene = load_scene(out)
    assert scene["view"]["start_image"] == name
    assert scene["view"]["eye"] == scene["cameras"][3]["center"]

    result = run("-o", str(tmp_path / "other"), "--start-image", "missing.jpg")
    assert result.exit_code != 0
    assert "names no image" in result.output
    assert not (tmp_path / "other").exists()


def test_single_file_inlines_everything(tmp_path):
    out = tmp_path / "one"
    result = run("-o", str(out), "--single-file")
    assert result.exit_code == 0, result.output
    assert [p.name for p in out.iterdir()] == ["index.html"]
    page = (out / "index.html").read_text(encoding="utf-8")
    assert 'id="wx-scene"' in page and 'id="wx-module"' in page
    assert "data:image/jpeg;base64," in page
    assert (
        "</script" not in page.split('id="wx-module">', 1)[1].split("</script>", 1)[0]
    )


def test_single_file_is_refused_past_the_page_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(web_export_module, "SINGLE_FILE_LIMIT", 100_000)
    out = tmp_path / "one"
    result = run("-o", str(out), "--single-file")
    assert result.exit_code != 0
    assert "single-file page would be" in result.output
    assert not out.exists()


@pytest.mark.parametrize("quality", ["0", "101"])
def test_jpeg_quality_is_range_checked(tmp_path, quality):
    result = run("-o", str(tmp_path / "site"), "--jpeg-quality", quality)
    assert result.exit_code != 0
