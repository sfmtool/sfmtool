# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the render-patches command and patch renderer."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction
from sfmtool.cli import main
from sfmtool.visualization._patch_renderer import (
    MODES,
    PatchRenderError,
    collect_patches,
    render_patches,
)


def _attach_patches(sfmr_path, *, with_bitmaps=False, bitmap_alpha=255, resolution=8):
    """Build a fixed-extent patch cloud (optionally bitmaps) and save a copy.

    Uses ``mean_viewing`` normals and a ``fixed`` extent so it needs only the
    reconstruction geometry — no ``.sift`` files — keeping the fixture cheap.
    """
    recon = SfmrReconstruction.load(str(sfmr_path))
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=0.05
    )
    recon = recon.clone_with_changes(patches=cloud)
    if with_bitmaps:
        # One RGBA bitmap per 3D point (parallel to points).
        bmps = np.zeros((recon.point_count, resolution, resolution, 4), np.uint8)
        bmps[..., :3] = 128
        bmps[..., 3] = bitmap_alpha
        recon = recon.clone_with_changes(patch_bitmaps=bmps)
    out = sfmr_path.parent / "patched.sfmr"
    recon.save(str(out), operation="test")
    return out


def _attach_patches_with_infinity(sfmr_path):
    """Like ``_attach_patches`` but first turns one well-observed point into a
    point at infinity, so the saved cloud carries a ``w == 0`` tangent-sphere
    patch alongside the finite ones. Returns ``(path, infinity_point_id)``."""
    recon = SfmrReconstruction.load(str(sfmr_path))
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    counts = np.bincount(np.asarray(recon.track_point_ids), minlength=recon.point_count)
    pi = int(np.argmax(counts))
    xyz = pos[pi, :3]
    pos[pi] = np.append(xyz / np.linalg.norm(xyz), 0.0)
    recon = recon.clone_with_changes(positions=pos)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent="fixed", extent_value=0.05
    )
    recon = recon.clone_with_changes(patches=cloud)
    out = sfmr_path.parent / "patched_inf.sfmr"
    recon.save(str(out), operation="test")
    return out, pi


# =============================================================================
# Renderer unit behaviour
# =============================================================================


class TestCollectPatches:
    def test_no_patches_raises(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(str(seoul_bull_sfmr_only))
        with pytest.raises(PatchRenderError, match="no patch cloud"):
            collect_patches(recon)

    def test_returns_parallel_arrays(self, seoul_bull_workspace):
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        centers, u_vec, v_vec, normals, point_ids, w = collect_patches(recon)
        n = len(centers)
        assert n > 0
        for arr in (u_vec, v_vec, normals):
            assert arr.shape == (n, 3)
        assert point_ids.shape == (n,)
        assert w.shape == (n,)

    def test_reports_w_for_points_at_infinity(self, seoul_bull_workspace):
        patched, pi = _attach_patches_with_infinity(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        _, _, _, _, point_ids, w = collect_patches(recon)
        # Exactly the injected infinity point's patch is flagged w == 0; the rest
        # stay finite (w == 1).
        inf_rows = w == 0.0
        assert inf_rows.sum() == 1
        assert int(point_ids[inf_rows][0]) == pi
        assert np.all(w[~inf_rows] == 1.0)

    def test_renders_points_at_infinity(self, seoul_bull_workspace, tmp_path):
        """The renderer composites a w == 0 patch (direction corners projected as
        rays, no translation) without crashing or culling it as behind-camera."""
        patched, _ = _attach_patches_with_infinity(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        results = render_patches(
            recon, tmp_path / "inf", mode="flat", backface_cull=True
        )
        assert results, "no images rendered"
        assert sum(n for _, n, _ in results) > 0, "no patches drawn"


class TestRenderPatches:
    @pytest.mark.parametrize("mode", ["normal", "flat", "wire"])
    def test_modes_write_images(self, seoul_bull_workspace, tmp_path, mode):
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        out = tmp_path / mode
        results = render_patches(recon, out, mode=mode, image_filter=["_08"])
        assert len(results) == 1
        name, n_drawn, path = results[0]
        assert n_drawn > 0
        img = cv2.imread(str(path))
        assert img is not None and img.size > 0

    def test_texture_requires_bitmaps(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace, with_bitmaps=False)
        recon = SfmrReconstruction.load(str(patched))
        with pytest.raises(PatchRenderError, match="bitmaps"):
            render_patches(recon, tmp_path / "tex", mode="texture")

    def test_texture_opaque_threshold(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace, with_bitmaps=True)
        recon = SfmrReconstruction.load(str(patched))
        results = render_patches(
            recon,
            tmp_path / "tex",
            mode="texture",
            opaque_threshold=0.0,
            image_filter=["_08"],
        )
        assert results and results[0][1] > 0

    def test_texture_non_opaque(self, seoul_bull_workspace, tmp_path):
        # Exercises the confidence-alpha (warped_a / 255) compositing branch.
        patched = _attach_patches(seoul_bull_workspace, with_bitmaps=True)
        recon = SfmrReconstruction.load(str(patched))
        results = render_patches(
            recon, tmp_path / "tex", mode="texture", image_filter=["_08"]
        )
        assert results and results[0][1] > 0

    def test_opaque_threshold_drops_low_confidence(
        self, seoul_bull_workspace, tmp_path
    ):
        # Bitmaps with alpha=20 (~0.078). A threshold above that paints nothing;
        # a threshold below it paints. (n_drawn counts patches, not texels, so
        # compare the rendered pixels against the untouched source.)
        patched = _attach_patches(
            seoul_bull_workspace, with_bitmaps=True, bitmap_alpha=20
        )
        recon = SfmrReconstruction.load(str(patched))
        below = render_patches(
            recon,
            tmp_path / "lo",
            mode="texture",
            opaque_threshold=0.0,
            image_filter=["_08"],
        )
        above = render_patches(
            recon,
            tmp_path / "hi",
            mode="texture",
            opaque_threshold=0.1,
            image_filter=["_08"],
        )
        name = below[0][0]
        source = cv2.imread(str(Path(recon.workspace_dir) / name))
        painted = cv2.imread(str(below[0][2]))
        dropped = cv2.imread(str(above[0][2]))
        assert np.any(painted != source)  # below-threshold confidence is drawn
        assert np.array_equal(dropped, source)  # above-threshold drops everything

    def test_flat_actually_composites(self, seoul_bull_workspace, tmp_path):
        # The rendered frame must differ from the untouched source image,
        # locking in that patches are drawn (and the corner/channel handling).
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        results = render_patches(
            recon, tmp_path / "f", mode="flat", image_filter=["_08"]
        )
        name, _, out_path = results[0]
        rendered = cv2.imread(str(out_path))
        source = cv2.imread(str(Path(recon.workspace_dir) / name))
        assert rendered.shape == source.shape
        assert np.any(rendered != source)

    def test_backface_cull_reduces_count(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        culled = render_patches(
            recon, tmp_path / "c", mode="wire", backface_cull=True, image_filter=["_08"]
        )
        uncull = render_patches(
            recon,
            tmp_path / "u",
            mode="wire",
            backface_cull=False,
            image_filter=["_08"],
        )
        assert uncull[0][1] > culled[0][1]

    def test_upscale_enlarges_output(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        r1 = render_patches(recon, tmp_path / "a", mode="wire", image_filter=["_08"])
        r3 = render_patches(
            recon, tmp_path / "b", mode="wire", upscale=3.0, image_filter=["_08"]
        )
        small = cv2.imread(str(r1[0][2]))
        big = cv2.imread(str(r3[0][2]))
        assert big.shape[0] == small.shape[0] * 3
        assert big.shape[1] == small.shape[1] * 3

    def test_image_filter_no_match(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        results = render_patches(
            recon, tmp_path / "z", mode="wire", image_filter=["nope"]
        )
        assert results == []

    def test_border_color_is_rgb(self, seoul_bull_workspace, tmp_path):
        # --border-color is R,G,B: a red (255,0,0) request must paint red
        # borders, not blue (which is what BGR-ordering would produce).
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        results = render_patches(
            recon,
            tmp_path / "b",
            mode="wire",
            border_color=(255, 0, 0),
            border_thickness=2,
            image_filter=["_08"],
        )
        img = cv2.imread(str(results[0][2]))  # cv2 reads as BGR
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        red = (r > 200) & (b < 60) & (g < 60)
        blue = (b > 200) & (r < 60) & (g < 60)
        assert red.any()
        assert not blue.any()

    def test_unknown_mode_raises(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace)
        recon = SfmrReconstruction.load(str(patched))
        with pytest.raises(PatchRenderError, match="unknown mode"):
            render_patches(recon, tmp_path / "z", mode="bogus")


# =============================================================================
# CLI
# =============================================================================


class TestRenderPatchesCLI:
    def test_help(self):
        result = CliRunner().invoke(main, ["render-patches", "--help"])
        assert result.exit_code == 0
        assert "oriented patches" in result.output
        assert all(mode in result.output for mode in MODES)

    def test_non_sfmr_rejected(self, tmp_path):
        recon = tmp_path / "recon.txt"
        recon.write_bytes(b"fake")
        result = CliRunner().invoke(
            main, ["render-patches", str(recon), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_reconstruction_errors(self, tmp_path):
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(tmp_path / "nope.sfmr"),
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_output_required(self):
        result = CliRunner().invoke(main, ["render-patches", "dummy.sfmr"])
        assert result.exit_code != 0

    def test_no_patches_reports_usage_error(self, seoul_bull_sfmr_only, tmp_path):
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(seoul_bull_sfmr_only),
                "-o",
                str(tmp_path / "out"),
                "--mode",
                "wire",
            ],
        )
        assert result.exit_code != 0
        assert "no patch cloud" in result.output

    def test_bad_border_color(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace)
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(patched),
                "-o",
                str(tmp_path / "out"),
                "--mode",
                "wire",
                "--border-color",
                "300,0",
            ],
        )
        assert result.exit_code != 0

    def test_opaque_out_of_range_rejected(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace, with_bitmaps=True)
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(patched),
                "-o",
                str(tmp_path / "out"),
                "--opaque",
                "2.0",
                "--images",
                "_08",
            ],
        )
        assert result.exit_code != 0
        assert "between 0 and 1" in result.output

    def test_bare_opaque_defaults(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace, with_bitmaps=True)
        out = tmp_path / "out"
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(patched),
                "-o",
                str(out),
                "--opaque",
                "--images",
                "_08",
            ],
        )
        assert result.exit_code == 0, result.output
        assert list(out.glob("*_texture.png"))

    def test_e2e_wire(self, seoul_bull_workspace, tmp_path):
        patched = _attach_patches(seoul_bull_workspace)
        out = tmp_path / "out"
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(patched),
                "-o",
                str(out),
                "--mode",
                "wire",
                "--images",
                "_08",
            ],
        )
        assert result.exit_code == 0, result.output
        pngs = list(out.glob("*_wire.png"))
        assert len(pngs) == 1
