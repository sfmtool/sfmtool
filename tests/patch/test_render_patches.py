# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the render-patches command and patch renderer.

Most run on the demo scene (the ``demo_workspace_once`` fixture), an
``embedded_patches`` reconstruction whose patch frames lie exactly in the
surfaces its photographs show and whose bitmaps were rendered from those
photographs, so a render can be checked against the photograph it is drawn on.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.patches import PatchCloud
from sfmtool.cli import main
from sfmtool.fileio import read_image_rgb
from sfmtool.visualization._patch_renderer import (
    MODES,
    PatchRenderError,
    collect_patches,
    render_patches,
)

# The demo image the single-image tests draw on: the middle camera, which
# sees the ground, the faces -a and -b, and the top almost edge-on.
_IMAGE = "demo_2"

# The outward normals of the demo cube's faces -a and -b.
_FACE_MINUS_A = (-0.8, -0.6, 0.0)
_FACE_MINUS_B = (0.6, -0.8, 0.0)


def _with_point_at_infinity(sfmr_path):
    """Turn the most observed point of the reconstruction at ``sfmr_path``
    into a point at infinity, build a fixed-extent patch cloud from the
    geometry, so the saved cloud carries a ``w == 0`` tangent-sphere patch
    alongside the finite ones, and save it beside the original. Returns
    ``(path, infinity_point_id)``."""
    recon = SfmrReconstruction.load(str(sfmr_path))
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    counts = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=recon.point_count
    )
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


@pytest.fixture
def demo(demo_workspace_once):
    """The demo scene's reconstruction, loaded from its session workspace."""
    return SfmrReconstruction.load(str(demo_workspace_once))


def _painted_difference(source: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    """The absolute per-channel differences of the pixels a render changed."""
    painted = np.any(rendered != source, axis=2)
    assert painted.mean() > 0.2  # the patches cover a fifth of the image
    return np.abs(rendered.astype(np.int16) - source.astype(np.int16))[painted]


# =============================================================================
# Renderer unit behaviour
# =============================================================================


class TestCollectPatches:
    def test_no_patches_raises(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(str(seoul_bull_sfmr_only))
        with pytest.raises(PatchRenderError, match="no patch cloud"):
            collect_patches(recon)

    def test_returns_parallel_arrays(self, demo):
        """One row per point, in point order: the point's position, its
        stored normal and its patch's half-extent vectors, 0.09 m long."""
        centers, u_vec, v_vec, normals, point_ids, w = collect_patches(demo)
        n = len(centers)
        assert n == demo.point_count
        for arr in (u_vec, v_vec, normals):
            assert arr.shape == (n, 3)
        assert point_ids.shape == (n,)
        assert w.shape == (n,)
        np.testing.assert_array_equal(point_ids, np.arange(n))
        np.testing.assert_array_equal(centers, demo.positions)
        np.testing.assert_allclose(normals, demo.normals, atol=1e-6)
        for vec in (u_vec, v_vec):
            np.testing.assert_allclose(np.linalg.norm(vec, axis=1), 0.09, atol=1e-7)
        np.testing.assert_allclose(np.cross(u_vec, v_vec) / 0.09**2, normals, atol=1e-6)
        assert np.all(w == 1.0)

    def test_reports_w_for_points_at_infinity(self, demo_workspace):
        patched, pi = _with_point_at_infinity(demo_workspace)
        recon = SfmrReconstruction.load(str(patched))
        _, _, _, _, point_ids, w = collect_patches(recon)
        # Exactly the injected infinity point's patch is flagged w == 0; the rest
        # stay finite (w == 1).
        inf_rows = w == 0.0
        assert inf_rows.sum() == 1
        assert int(point_ids[inf_rows][0]) == pi
        assert np.all(w[~inf_rows] == 1.0)

    def test_renders_points_at_infinity(self, demo_workspace, tmp_path):
        """The renderer composites a w == 0 patch (direction corners projected as
        rays, no translation) without crashing or culling it as behind-camera."""
        patched, _ = _with_point_at_infinity(demo_workspace)
        recon = SfmrReconstruction.load(str(patched))
        results = render_patches(
            recon, tmp_path / "inf", mode="flat", backface_cull=True
        )
        assert results, "no images rendered"
        assert sum(n for _, n, _ in results) > 0, "no patches drawn"


class TestRenderPatches:
    @pytest.mark.parametrize("mode", ["normal", "flat", "wire"])
    def test_modes_write_images(self, demo, tmp_path, mode):
        out = tmp_path / mode
        results = render_patches(demo, out, mode=mode, image_filter=[_IMAGE])
        assert len(results) == 1
        name, n_drawn, path = results[0]
        assert n_drawn > 0
        img = cv2.imread(str(path))
        assert img is not None and img.size > 0

    def test_texture_requires_bitmaps(self, demo, tmp_path):
        recon = demo.clone_with_changes(patch_bitmaps=None)
        with pytest.raises(PatchRenderError, match="bitmaps"):
            render_patches(recon, tmp_path / "tex", mode="texture")

    def test_texture_opaque_threshold(self, demo, tmp_path):
        """Each bitmap is a render of its patch from a photograph, and each
        patch frame lies in the surface, so the texture drawn on a photograph
        is that photograph: the median painted channel is within 8 levels of
        the source (4 or 5 in each image), where a bitmap flipped about either
        axis puts it 11 or 12 off, and one of another point 20."""
        results = render_patches(
            demo,
            tmp_path / "tex",
            mode="texture",
            opaque_threshold=0.0,
            image_filter=[_IMAGE],
        )
        assert results and results[0][1] > 0
        name, _, out_path = results[0]
        source = read_image_rgb(Path(demo.workspace_dir) / name)
        difference = _painted_difference(source, read_image_rgb(out_path))
        assert np.median(difference) <= 8

    def test_texture_non_opaque(self, demo, tmp_path):
        # Exercises the confidence-alpha (warped_a / 255) compositing branch.
        results = render_patches(
            demo, tmp_path / "tex", mode="texture", image_filter=[_IMAGE]
        )
        assert results and results[0][1] > 0

    def test_opaque_threshold_drops_low_confidence(self, demo, tmp_path):
        # Bitmaps with alpha=20 (~0.078). A threshold above that paints nothing;
        # a threshold below it paints. (n_drawn counts patches, not texels, so
        # compare the rendered pixels against the untouched source.)
        bitmaps = np.asarray(demo.patch_bitmaps).copy()
        bitmaps[..., 3] = 20
        recon = demo.clone_with_changes(patch_bitmaps=bitmaps)
        below = render_patches(
            recon,
            tmp_path / "lo",
            mode="texture",
            opaque_threshold=0.0,
            image_filter=[_IMAGE],
        )
        above = render_patches(
            recon,
            tmp_path / "hi",
            mode="texture",
            opaque_threshold=0.1,
            image_filter=[_IMAGE],
        )
        name = below[0][0]
        # The renderer decodes the source with `read_image_rgb`, so the
        # untouched source is read the same way.
        source = read_image_rgb(Path(recon.workspace_dir) / name)
        painted = read_image_rgb(below[0][2])
        dropped = read_image_rgb(above[0][2])
        assert np.any(painted != source)  # below-threshold confidence is drawn
        assert np.array_equal(dropped, source)  # above-threshold drops everything

    def test_flat_actually_composites(self, demo, tmp_path):
        # The rendered frame must differ from the untouched source image,
        # locking in that patches are drawn (and the corner/channel handling).
        results = render_patches(
            demo, tmp_path / "f", mode="flat", image_filter=[_IMAGE]
        )
        name, _, out_path = results[0]
        rendered = cv2.imread(str(out_path))
        source = cv2.imread(str(Path(demo.workspace_dir) / name))
        assert rendered.shape == source.shape
        assert np.any(rendered != source)

    @pytest.mark.parametrize(
        ("image", "face_turned_away"),
        [("demo_0", _FACE_MINUS_B), ("demo_4", _FACE_MINUS_A)],
    )
    def test_backface_cull_reduces_count(self, demo, tmp_path, image, face_turned_away):
        """The outer cameras see one side face of the cube from behind: the
        cull takes off as many patches as that face holds, 64, and keeps the
        top's, which those cameras see at about 87 degrees."""
        culled = render_patches(
            demo, tmp_path / "c", mode="wire", backface_cull=True, image_filter=[image]
        )
        uncull = render_patches(
            demo,
            tmp_path / "u",
            mode="wire",
            backface_cull=False,
            image_filter=[image],
        )
        normals = np.asarray(demo.normals, dtype=np.float64)
        on_face = int((np.abs(normals - face_turned_away).max(axis=1) < 1e-5).sum())
        assert on_face == 64
        assert uncull[0][1] - culled[0][1] == on_face

    def test_upscale_enlarges_output(self, demo, tmp_path):
        r1 = render_patches(demo, tmp_path / "a", mode="wire", image_filter=[_IMAGE])
        r3 = render_patches(
            demo, tmp_path / "b", mode="wire", upscale=3.0, image_filter=[_IMAGE]
        )
        small = cv2.imread(str(r1[0][2]))
        big = cv2.imread(str(r3[0][2]))
        assert big.shape[0] == small.shape[0] * 3
        assert big.shape[1] == small.shape[1] * 3

    def test_image_filter_no_match(self, demo, tmp_path):
        results = render_patches(
            demo, tmp_path / "z", mode="wire", image_filter=["nope"]
        )
        assert results == []

    def test_border_color_is_rgb(self, demo, tmp_path):
        # --border-color is R,G,B: a red (255,0,0) request must paint red
        # borders, not blue (which is what BGR-ordering would produce).
        results = render_patches(
            demo,
            tmp_path / "b",
            mode="wire",
            border_color=(255, 0, 0),
            border_thickness=2,
            image_filter=[_IMAGE],
        )
        img = cv2.imread(str(results[0][2]))  # cv2 reads as BGR
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        red = (r > 200) & (b < 60) & (g < 60)
        blue = (b > 200) & (r < 60) & (g < 60)
        assert red.any()
        assert not blue.any()

    def test_unknown_mode_raises(self, demo, tmp_path):
        with pytest.raises(PatchRenderError, match="unknown mode"):
            render_patches(demo, tmp_path / "z", mode="bogus")


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

    def test_rejects_sift_files(self, seoul_bull_sfmr_only, tmp_path):
        """A sift_files recon is rejected up front (the surfel precondition),
        with a pointer to the conversion bridge."""
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
        assert "embedded_patches" in result.output

    def test_bad_border_color(self, demo_workspace_once, tmp_path):
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(demo_workspace_once),
                "-o",
                str(tmp_path / "out"),
                "--mode",
                "wire",
                "--border-color",
                "300,0",
            ],
        )
        assert result.exit_code != 0

    def test_opaque_out_of_range_rejected(self, demo_workspace_once, tmp_path):
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(demo_workspace_once),
                "-o",
                str(tmp_path / "out"),
                "--opaque",
                "2.0",
                "--images",
                _IMAGE,
            ],
        )
        assert result.exit_code != 0
        assert "between 0 and 1" in result.output

    def test_bare_opaque_defaults(self, demo_workspace_once, tmp_path):
        out = tmp_path / "out"
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(demo_workspace_once),
                "-o",
                str(out),
                "--opaque",
                "--images",
                _IMAGE,
            ],
        )
        assert result.exit_code == 0, result.output
        assert list(out.glob("*_texture.png"))

    def test_e2e_wire(self, demo_workspace_once, tmp_path):
        out = tmp_path / "out"
        result = CliRunner().invoke(
            main,
            [
                "render-patches",
                str(demo_workspace_once),
                "-o",
                str(out),
                "--mode",
                "wire",
                "--images",
                _IMAGE,
            ],
        )
        assert result.exit_code == 0, result.output
        pngs = list(out.glob("*_wire.png"))
        assert len(pngs) == 1
