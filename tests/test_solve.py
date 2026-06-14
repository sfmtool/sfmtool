# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from sfmtool.cli import main


def test_solve_no_mode(isolated_seoul_bull_image: Path):
    """Test that solve without -i or -g raises an error."""
    result = CliRunner().invoke(main, ["solve", str(isolated_seoul_bull_image)])
    assert result.exit_code != 0
    assert "Must specify either --incremental" in result.output


def test_solve_both_modes(isolated_seoul_bull_image: Path):
    """Test that specifying both modes raises an error."""
    result = CliRunner().invoke(
        main, ["solve", "-i", "-g", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code != 0
    assert "Cannot specify both" in result.output


def test_solve_no_paths():
    """Test that solve without paths raises an error."""
    result = CliRunner().invoke(main, ["solve", "-i"])
    assert result.exit_code != 0
    assert "Must provide image paths" in result.output


def test_solve_incremental(isolated_seoul_bull_17_images: list[Path]):
    """Test incremental SfM on a small set of images."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract SIFT features
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run incremental SfM
    output_path = workspace_dir / "test_solve.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-i",
            "--output",
            str(output_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Running incremental SfM" in result.output
    assert "Found reconstruction" in result.output
    assert output_path.exists()

    # Verify the .sfmr file can be loaded
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(output_path)
    assert recon.image_count > 0
    assert recon.point_count > 0
    assert recon.camera_count > 0


def test_solve_global(isolated_seoul_bull_17_images: list[Path]):
    """Test global SfM (GLOMAP) on a small set of images."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract SIFT features
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run global SfM
    output_path = workspace_dir / "test_global.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-g",
            "--output",
            str(output_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Running global SfM" in result.output
    assert "Found reconstruction" in result.output
    assert output_path.exists()


def test_solve_from_matches(isolated_seoul_bull_17_images: list[Path]):
    """Test solving from a pre-computed .matches file."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace and extract SIFT
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run exhaustive matching first
    matches_path = workspace_dir / "test.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--exhaustive",
            "--output",
            str(matches_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert matches_path.exists()

    # Solve from the .matches file
    output_path = workspace_dir / "from_matches.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-i",
            "--output",
            str(output_path),
            str(matches_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_solve_from_cluster_matches(isolated_seoul_bull_17_images: list[Path]):
    """The dataset-script recipe: sfmtool SIFT + track-cluster matching, then
    solve from the resulting .matches file."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # The committed camera_config.json would reject --camera-model, but the
    # cluster recipe uses auto-detected intrinsics — leave it in place.
    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", str(workspace_dir)]
    )
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    matches_path = workspace_dir / "cluster.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--cluster",
            "--cluster-d",
            "28",
            "--output",
            str(matches_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert matches_path.exists()

    output_path = workspace_dir / "from_cluster.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-i",
            "--seed",
            "42",
            "--output",
            str(output_path),
            str(matches_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()

    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(output_path)
    assert recon.image_count > 0
    assert recon.point_count > 0


def test_resolve_output_path_explicit(tmp_path: Path):
    """Explicit --output: recon 0 gets the exact path, subsequent recons
    get `-N` before the suffix."""
    from sfmtool._incremental_sfm import _resolve_output_path

    explicit = tmp_path / "mine.sfmr"
    images = [Path("img_1.jpg"), Path("img_2.jpg")]

    p0 = _resolve_output_path(
        order=0,
        explicit_output=explicit,
        recon_image_paths=images,
        sfmr_dir=None,
        workspace_dir=tmp_path,
    )
    p1 = _resolve_output_path(
        order=1,
        explicit_output=explicit,
        recon_image_paths=images,
        sfmr_dir=None,
        workspace_dir=tmp_path,
    )
    p2 = _resolve_output_path(
        order=2,
        explicit_output=explicit,
        recon_image_paths=images,
        sfmr_dir=None,
        workspace_dir=tmp_path,
    )
    assert p0 == explicit
    assert p1 == tmp_path / "mine-1.sfmr"
    assert p2 == tmp_path / "mine-2.sfmr"


def test_resolve_output_path_auto_per_recon(tmp_path: Path):
    """Auto-name: each reconstruction gets its own descriptor reflecting the
    images actually present in that reconstruction."""
    from sfmtool._incremental_sfm import _resolve_output_path

    sfmr_dir = tmp_path / "sfmr"

    # Recon 0: three frames that form a small sub-range (1-3).
    recon0_images = [Path(f"KerryPark_{n}.jpg") for n in (1, 2, 3)]
    p0 = _resolve_output_path(
        order=0,
        explicit_output=None,
        recon_image_paths=recon0_images,
        sfmr_dir=sfmr_dir,
        workspace_dir=tmp_path,
    )
    assert "KerryPark_1-3" in p0.name
    # Touch the file so the counter advances for the next call.
    p0.parent.mkdir(parents=True, exist_ok=True)
    p0.touch()

    # Recon 1: the other 98 frames. Descriptor reflects 4..101, not the
    # full input range.
    recon1_images = [Path(f"KerryPark_{n}.jpg") for n in range(4, 102)]
    p1 = _resolve_output_path(
        order=1,
        explicit_output=None,
        recon_image_paths=recon1_images,
        sfmr_dir=sfmr_dir,
        workspace_dir=tmp_path,
    )
    assert "KerryPark_4-101" in p1.name
    assert p1 != p0


def test_solve_from_matches_with_range(isolated_seoul_bull_17_images: list[Path]):
    """`--range` on the .matches solve path restricts which images/pairs are
    written to the COLMAP DB."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    matches_path = workspace_dir / "test.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--exhaustive",
            "--output",
            str(matches_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    output_path = workspace_dir / "from_matches_range.sfmr"
    # The 17-image fixture is numbered 1..17; keep 5..12 (8 images).
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-i",
            "--range",
            "5-12",
            "--output",
            str(output_path),
            str(matches_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Summary line reflects the filtered counts, not the full .matches payload.
    assert "filtered from 17" in result.output
    # 8 images → C(8, 2) = 28 possible pairs (upper bound).
    import re

    m = re.search(r"Pairs: (\d+) \(filtered from (\d+)\)", result.output)
    assert m is not None, result.output
    kept_pairs, full_pairs = int(m.group(1)), int(m.group(2))
    assert 0 < kept_pairs <= 28
    assert kept_pairs < full_pairs


class _FakeRecon:
    """Stand-in for a Rust SfmrReconstruction that records where it was saved."""

    def __init__(self, idx: str, image_count: int, saved: dict[str, Path]):
        self._idx = idx
        self._saved = saved
        self.camera_count = 1
        self.image_count = image_count
        self.point_count = 10

    def save(self, path):
        self._saved[self._idx] = Path(path)


def _patch_save_reconstructions(monkeypatch, fake_models: dict[str, dict]):
    """Patch the disk/convert dependencies of `_save_reconstructions`.

    `fake_models` maps a model index (as a string, matching the on-disk
    subdirectory name) to a dict with `image_names`, `positions_xyz`,
    `track_image_indexes` and `cameras`. Returns the `saved` dict that the
    fake reconstructions record their output paths into.
    """
    import sfmtool._sfmtool as _sfmtool_mod
    from sfmtool import _incremental_sfm

    def fake_read(recon_dir):
        return fake_models[Path(recon_dir).name]

    saved: dict[str, Path] = {}

    def fake_convert(recon_dir, image_dir, metadata, classify_infinity=True):
        name = Path(recon_dir).name
        return _FakeRecon(name, len(fake_models[name]["image_names"]), saved)

    monkeypatch.setattr(_sfmtool_mod, "read_colmap_binary", fake_read, raising=False)
    monkeypatch.setattr(_incremental_sfm, "colmap_binary_to_rust_sfmr", fake_convert)
    monkeypatch.setattr(_incremental_sfm, "build_metadata", lambda **kwargs: {})
    return saved


def test_save_reconstructions_skips_empty_and_orders_largest_first(
    tmp_path: Path, monkeypatch
):
    """A degenerate 0-point model must not abort the solve, must not be saved,
    and the largest surviving model claims the explicit `--output` slot."""
    from sfmtool._incremental_sfm import _save_reconstructions

    fake_models = {
        # model 0: degenerate fragment the mapper abandoned (0 points).
        "0": {
            "image_names": ["a.jpg", "b.jpg"],
            "positions_xyz": [],
            "track_image_indexes": [],
            "cameras": [object()],
        },
        # model 1: a small real model (3 images).
        "1": {
            "image_names": ["c.jpg", "d.jpg", "e.jpg"],
            "positions_xyz": [(0.0, 0.0, 0.0)] * 4,
            "track_image_indexes": [0] * 8,
            "cameras": [object()],
        },
        # model 2: the real reconstruction (6 images) at a *higher* index.
        "2": {
            "image_names": [f"img{i}.jpg" for i in range(6)],
            "positions_xyz": [(0.0, 0.0, 0.0)] * 20,
            "track_image_indexes": [0] * 40,
            "cameras": [object()],
        },
    }
    saved = _patch_save_reconstructions(monkeypatch, fake_models)

    image_dir = tmp_path / "images"
    explicit = tmp_path / "out.sfmr"

    primary = _save_reconstructions(
        {0: None, 1: None, 2: None},
        has_rig=False,
        reconstruction_path=tmp_path / "reconstruction",
        image_dir=image_dir,
        workspace_dir=tmp_path,
        workspace_config={},
        tool_name="colmap",
        tool_options={},
        input_image_count=11,
        explicit_output=explicit,
        sfmr_dir=None,
        detect_infinity=False,
    )

    # The degenerate model is skipped, never written.
    assert "0" not in saved
    # Largest model (idx 2, 6 images) gets the explicit --output and is primary.
    assert saved["2"] == explicit
    assert primary == explicit
    # The smaller surviving model lands under the `-1` sibling name.
    assert saved["1"] == tmp_path / "out-1.sfmr"


def test_save_reconstructions_all_degenerate_raises(tmp_path: Path, monkeypatch):
    """If every model is degenerate, the solve still raises (nothing to save)."""
    import pytest

    from sfmtool._incremental_sfm import _save_reconstructions

    fake_models = {
        "0": {
            "image_names": ["a.jpg", "b.jpg"],
            "positions_xyz": [],
            "track_image_indexes": [],
            "cameras": [object()],
        },
    }
    _patch_save_reconstructions(monkeypatch, fake_models)

    with pytest.raises(RuntimeError, match="No 3D points found"):
        _save_reconstructions(
            {0: None},
            has_rig=False,
            reconstruction_path=tmp_path / "reconstruction",
            image_dir=tmp_path / "images",
            workspace_dir=tmp_path,
            workspace_config={},
            tool_name="colmap",
            tool_options={},
            input_image_count=2,
            explicit_output=None,
            sfmr_dir=None,
            detect_infinity=False,
        )


class _FakePycolmapRecon:
    """Stand-in for a `pycolmap.Reconstruction` on the rig save path.

    The rig branch reads counts off the live object (`.images`, `.points3D`
    with per-point `.track.elements`, `.cameras`) rather than a
    `read_colmap_binary` dict, so it needs its own fake. `test_idx` lets the
    patched converter recover which model it was handed.
    """

    def __init__(self, test_idx: str, image_names: list[str], obs_per_point: list[int]):
        self.test_idx = test_idx
        self.images = {i: SimpleNamespace(name=n) for i, n in enumerate(image_names)}
        self.points3D = {
            i: SimpleNamespace(track=SimpleNamespace(elements=[None] * obs))
            for i, obs in enumerate(obs_per_point)
        }
        self.cameras = {0: object()}


def _patch_save_reconstructions_rig(monkeypatch):
    """Patch the convert/metadata dependencies of the rig save path."""
    from sfmtool import _incremental_sfm

    saved: dict[str, Path] = {}

    def fake_convert(recon, image_dir, metadata, classify_infinity=True):
        return _FakeRecon(recon.test_idx, len(recon.images), saved)

    monkeypatch.setattr(_incremental_sfm, "pycolmap_to_rust_sfmr", fake_convert)
    monkeypatch.setattr(_incremental_sfm, "build_metadata", lambda **kwargs: {})
    return saved


def test_save_reconstructions_rig_skips_empty_and_orders_largest_first(
    tmp_path: Path, monkeypatch
):
    """Same skip-and-order guarantees on the rig path, which reads model stats
    from live pycolmap objects rather than the COLMAP binary dict."""
    from sfmtool._incremental_sfm import _save_reconstructions

    saved = _patch_save_reconstructions_rig(monkeypatch)

    reconstructions = {
        # model 0: degenerate fragment (0 points).
        0: _FakePycolmapRecon("0", ["a.jpg", "b.jpg"], []),
        # model 1: a small real model (3 images).
        1: _FakePycolmapRecon("1", ["c.jpg", "d.jpg", "e.jpg"], [4, 4]),
        # model 2: the real reconstruction (6 images) at a *higher* index.
        2: _FakePycolmapRecon("2", [f"img{i}.jpg" for i in range(6)], [10, 10]),
    }
    explicit = tmp_path / "out.sfmr"

    primary = _save_reconstructions(
        reconstructions,
        has_rig=True,
        reconstruction_path=tmp_path / "reconstruction",
        image_dir=tmp_path / "images",
        workspace_dir=tmp_path,
        workspace_config={},
        tool_name="colmap",
        tool_options={"rig_aware": True},
        input_image_count=11,
        explicit_output=explicit,
        sfmr_dir=None,
        detect_infinity=False,
    )

    # The degenerate rig model is skipped, never written.
    assert "0" not in saved
    # Largest model (idx 2, 6 images) gets the explicit --output and is primary.
    assert saved["2"] == explicit
    assert primary == explicit
    # The smaller surviving model lands under the `-1` sibling name.
    assert saved["1"] == tmp_path / "out-1.sfmr"
