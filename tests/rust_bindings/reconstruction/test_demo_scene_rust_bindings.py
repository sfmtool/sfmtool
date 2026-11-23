# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.reconstruction.DemoScene`: the scene the binding renders is the
Rust one, its workspace is one the CLI reads, its calls count through a
`ProgressCounter`, and its errors raise the exceptions the binding states.
The scene itself is tested in Rust
(`crates/sfmtool-core/src/reconstruction/data/demo_scene/tests.rs`).
"""

import numpy as np
import pytest
import xxhash
from click.testing import CliRunner

from sfmtool import ProgressCounter
from sfmtool.cli import main
from sfmtool.reconstruction import DemoScene, SfmrReconstruction

# The XXH128 of each JPEG of the default scene, for the JPEG encoder's AVX2
# path and for its scalar one: the hashes the Rust test
# `the_jpegs_match_the_hashes_stored_for_the_encoders_path` stores.
_JPEG_XXH128_AVX2 = [
    "2676ff94935814d7a9c791473689514b",
    "fb9c6357421a4f7857f052e3c7c8e288",
    "7b8a4e749c4706b3ad2954c9d40d7ed7",
    "60b576f9c973c4a0b7f09e600d991f9b",
    "06738cd2d67578ffa0b9e95ed0248f50",
]
_JPEG_XXH128_SCALAR = [
    "9ffaa8da80968aa67ee7fea0b81cc22c",
    "80e84e3aeb6b5037099f2f97048103bd",
    "02229b128c970781c69eaa6cb265e079",
    "7e1b45aa571c988b43828f05506bc85a",
    "e9f00c15962f49793383dad543609e20",
]

# `DemoSceneOptions::DEFAULT_POINT_COUNT`, and the observations the default
# scene keeps, as the Rust sampling test fixes them.
_DEFAULT_POINT_COUNT = 513
_DEFAULT_OBSERVATION_COUNT = 1712


@pytest.fixture(scope="module")
def scene_and_render_counter():
    """The default scene, rendered once for the module, and the counter its
    render reported to."""
    counter = ProgressCounter()
    return DemoScene.render(progress=counter), counter


@pytest.fixture(scope="module")
def scene(scene_and_render_counter):
    return scene_and_render_counter[0]


@pytest.fixture(scope="module")
def recon_and_counter(scene):
    """The default scene's reconstruction, built once for the module, and the
    counter its build reported to."""
    counter = ProgressCounter()
    return scene.reconstruction(progress=counter), counter


@pytest.fixture(scope="module")
def written(scene, tmp_path_factory):
    """The default scene's workspace, written once for the module, with the
    returned `.sfmr` path and the counter the write reported to."""
    directory = tmp_path_factory.mktemp("demo_scene") / "workspace"
    counter = ProgressCounter()
    path = scene.write_workspace(directory, progress=counter)
    return directory, path, counter


def test_the_binding_renders_the_rust_scene(scene, recon_and_counter):
    """The JPEGs are the Rust scene's bytes for one of the encoder's two
    paths, and the value keeps the points and observations the Rust tests
    fix."""
    hashes = [xxhash.xxh3_128_hexdigest(b) for b in scene.encoded_photographs]
    assert hashes in (_JPEG_XXH128_AVX2, _JPEG_XXH128_SCALAR)
    assert scene.image_names == [f"images/demo_{i}.jpg" for i in range(5)]

    recon, _ = recon_and_counter
    assert isinstance(recon, SfmrReconstruction)
    assert recon.point_count == _DEFAULT_POINT_COUNT
    assert recon.observation_count == _DEFAULT_OBSERVATION_COUNT
    assert recon.image_names == scene.image_names


def test_the_photographs_are_read_only_views_that_keep_the_scene_alive():
    """Each photograph list holds five read-only `(H, W, 3)` `uint8` arrays
    of the size asked for, and an array outlives the last other reference to
    its scene. The decoded JPEGs differ a little from the exact render."""
    scene = DemoScene.render(image_width=64, image_height=48, focal_length_px=56.0)
    rendered = scene.rendered_photographs
    photographs = scene.photographs
    del scene
    for images in (rendered, photographs):
        assert len(images) == 5
        for image in images:
            assert image.shape == (48, 64, 3)
            assert image.dtype == np.uint8
            assert not image.flags.writeable
            with pytest.raises(ValueError):
                image[0, 0, 0] = 0
    for exact, decoded in zip(rendered, photographs):
        difference = np.abs(exact.astype(np.int16) - decoded.astype(np.int16))
        assert 0 < difference.max() < 64


def test_point_count_sets_both_spacings_as_for_point_count_does():
    """`point_count` gives the spacings `for_point_count` gives: 1,000 keeps
    994 points, the count the draft's measurement records; the other options
    still apply."""
    scene = DemoScene.render(
        image_width=64, image_height=48, focal_length_px=56.0, point_count=1000
    )
    assert scene.photographs[0].shape == (48, 64, 3)
    assert scene.reconstruction().point_count == 994


def test_write_workspace_writes_a_workspace_sfm_inspect_reads(written):
    directory, path, _ = written
    assert path == directory / "demo.sfmr"
    assert (directory / ".sfm-workspace.json").is_file()
    for i in range(5):
        assert (directory / "images" / f"demo_{i}.jpg").is_file()

    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert f"3D points:      {_DEFAULT_POINT_COUNT}" in result.output
    assert "Integrity:      OK" in result.output

    recon = SfmrReconstruction.load(path)
    assert recon.point_count == _DEFAULT_POINT_COUNT


def test_each_call_counts_its_last_phase_through_the_progress_counter(
    scene_and_render_counter, recon_and_counter, written
):
    """The counter ends at its call's last phase's count: the five images
    encoded, the 513 bitmaps rendered, and the seven files written."""
    _, render_counter = scene_and_render_counter
    assert render_counter.value == 5
    _, counter = recon_and_counter
    assert counter.value == _DEFAULT_POINT_COUNT
    _, _, write_counter = written
    assert write_counter.value == 7


@pytest.mark.parametrize(
    "options",
    [
        {"image_width": 0},
        {"supersampling": 0},
        {"focal_length_px": float("nan")},
        {"max_obliquity_deg": 91.0},
        {"point_count": 0},
        {"point_count": 1000, "ground_point_spacing": 0.25},
        {"point_count": 1000, "face_point_spacing": 0.125},
    ],
)
def test_render_raises_value_error_for_an_option_out_of_range(options):
    with pytest.raises(ValueError):
        DemoScene.render(**options)


def test_write_workspace_raises_value_error_for_a_directory_it_refuses(
    scene, written, tmp_path
):
    """A directory that holds files, a path that is a file, and a directory
    inside another workspace are refused before anything is written."""
    holds_files = tmp_path / "holds_files"
    holds_files.mkdir()
    (holds_files / "note.txt").write_text("x")
    a_file = tmp_path / "a_file"
    a_file.write_text("x")
    nested = written[0] / "nested"
    for directory in (holds_files, a_file, nested):
        with pytest.raises(ValueError):
            scene.write_workspace(directory)
    assert not nested.exists()


def test_write_workspace_raises_os_error_when_a_directory_cannot_be_made(
    scene, tmp_path
):
    """A directory under a file cannot be created."""
    a_file = tmp_path / "a_file"
    a_file.write_text("x")
    with pytest.raises(OSError):
        scene.write_workspace(a_file / "workspace")
