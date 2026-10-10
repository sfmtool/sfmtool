# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the ``sfm embed-patches`` command and the ``embed_patches``
orchestration (steps 1-7 of the sift_files -> embedded_patches pipeline).

Runs the real photometric kernels on the ``seoul_bull_workspace`` fixture (which
carries the ``.sift`` files + source images) at a low patch resolution to keep the
end-to-end cost reasonable. See ``specs/cli/reconstruction/embed-patches-command.md`` and
``specs/core/patch/sift-to-patch-reconstruction.md``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch as mock_patch

import numpy as np
import pytest
from click.testing import CliRunner

import sfmtool._embed_patches as ep
import sfmtool._patch_compaction as pc
from sfmtool._patch_compaction import (
    image_file_hashes_from_images,
    image_file_hashes_from_sift,
)
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.fileio import verify_sfmr
from sfmtool.cli import main

from ..conftest import assert_references_name_track_observations
from .conftest import rotation_matrices


def test_embed_patches_cli_round_trips(monkeypatch, seoul_bull_workspace, tmp_path):
    """The CLI converts sift_files -> embedded_patches and the output loads + verifies
    with no .sift companion."""
    real = ep.embed_patches
    monkeypatch.setattr(
        ep,
        "embed_patches",
        lambda recon, images, **kw: real(recon, images, **{**kw, "resolution": 12}),
    )
    out = tmp_path / "out.sfmr"
    args = ["embed-patches", str(seoul_bull_workspace), str(out)]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert out.exists()

    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"

    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.feature_source == "embedded_patches"
    assert reloaded.point_count > 0
    assert reloaded.keypoints_xy is not None
    assert reloaded.patches is not None
    # image hashes match the source .sift metadata (one per image).
    assert len(reloaded.image_file_hashes) == reloaded.image_count
    # Each stored bitmap names the observation of its own track it is the tile of.
    assert reloaded.patch_bitmaps is not None
    assert_references_name_track_observations(reloaded)


def test_embed_patches_default_output_path(monkeypatch, seoul_bull_workspace):
    """Omitting OUTPUT writes <stem>-embedded.sfmr next to the input."""
    real = ep.embed_patches
    monkeypatch.setattr(
        ep,
        "embed_patches",
        lambda recon, images, **kw: real(recon, images, **{**kw, "resolution": 12}),
    )
    src = Path(seoul_bull_workspace)
    expected = src.with_name(f"{src.stem}-embedded.sfmr")
    try:
        args = ["embed-patches", str(src)]
        with mock_patch("sys.argv", ["sfm"] + args):
            result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, result.output
        assert expected.exists()
        assert (
            SfmrReconstruction.load(str(expected)).feature_source == "embedded_patches"
        )
    finally:
        expected.unlink(missing_ok=True)


def test_embed_patches_rejects_already_embedded(seoul_bull_workspace, tmp_path):
    """Running on an already-embedded reconstruction errors (nothing to convert)."""
    # Build a cheap embedded_patches input via the non-photometric baseline.
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    emb = recon.to_embedded_patches()
    embedded_path = tmp_path / "embedded.sfmr"
    emb.save(str(embedded_path), operation="xform")

    args = ["embed-patches", str(embedded_path), str(tmp_path / "out.sfmr")]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "already embedded_patches" in result.output


def test_image_file_hashes_from_sift_matches_metadata(seoul_bull_workspace):
    """image_file_hashes_from_sift reads the .sift image_file_xxh128 (16 bytes),
    matching the value re-hashing the image bytes would produce."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    from_sift = image_file_hashes_from_sift(recon)
    assert len(from_sift) == recon.image_count
    assert all(isinstance(h, bytes) and len(h) == 16 for h in from_sift)
    # Same identity hash as re-hashing the image bytes (the .sift records exactly
    # that digest).
    assert from_sift == image_file_hashes_from_images(recon)


def test_embed_patches_handles_points_at_infinity(seoul_bull_workspace):
    """The full orchestration runs on an infinity-bearing input (feature_size
    sizing doesn't choke on w=0 points) and produces a valid embedded_patches
    reconstruction; any surviving infinity point stays at infinity. Every kept
    point — finite or infinity — carries a real stored bitmap (nonzero alpha):
    culled points are dropped instead of kept with an all-black bitmap, and
    infinity points get a rendered texture, not a zero row."""
    import cv2

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    # Turn one well-observed point into a point at infinity, in the direction the
    # cameras see it: from the centroid of the camera centres, so the direction
    # lands in some image whatever the frame's origin.
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    counts = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=recon.point_count
    )
    pi = int(np.argmax(counts))
    centres = -np.einsum(
        "nji,nj->ni", rotation_matrices(recon), np.asarray(recon.translations)
    )
    ray = pos[pi, :3] - centres.mean(axis=0)
    pos[pi] = np.append(ray / np.linalg.norm(ray), 0.0)
    recon = recon.clone_with_changes(positions=pos)
    assert bool(np.asarray(recon.point_is_at_infinity)[pi])

    ws = recon.workspace_dir
    images = [
        np.ascontiguousarray(
            cv2.cvtColor(
                cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        )
        for name in recon.image_names
    ]

    out = ep.embed_patches(recon, images, resolution=12)
    assert out.feature_source == "embedded_patches"
    # The run completed and verifies; any kept infinity point is still w = 0.
    assert int(np.asarray(out.point_is_at_infinity).sum()) <= 1

    # The two-bugs invariant: every surviving point has a stored bitmap with
    # cross-view agreement somewhere (alpha > 0) — no all-black rows survive.
    bitmaps = np.asarray(out.patch_bitmaps)
    assert bitmaps.shape[0] == out.point_count
    alpha_nonzero = bitmaps[..., 3].reshape(out.point_count, -1).any(axis=1)
    assert alpha_nonzero.all(), (
        f"{int((~alpha_nonzero).sum())} surviving points have an all-black bitmap"
    )


def test_embed_patches_sources_hashes_from_embedded_not_sift(
    monkeypatch, seoul_bull_workspace
):
    """The re-layered pipeline's only ``.sift`` read is the ``to_embedded_patches``
    bridge; it sources image hashes from the embedded recon, not by re-reading the
    ``.sift`` files. Make the sift-hash helper blow up — the run must still succeed.
    """
    import cv2

    def _boom(_recon):
        raise AssertionError("embed_patches should not re-read .sift for hashes")

    monkeypatch.setattr(pc, "image_file_hashes_from_sift", _boom)

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    ws = recon.workspace_dir
    images = [
        np.ascontiguousarray(
            cv2.cvtColor(
                cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        )
        for name in recon.image_names
    ]
    out = ep.embed_patches(recon, images, resolution=12)
    assert out.feature_source == "embedded_patches"
    # The hashes are exactly the bridge's (images aren't culled, only points), so
    # they match what to_embedded_patches set — not a re-hash or a .sift re-read.
    expected = recon.to_embedded_patches(extent_value=5.0).image_file_hashes
    assert [bytes(h) for h in out.image_file_hashes] == [bytes(h) for h in expected]


def test_embed_patches_refine_anchors_on_stored_keypoints(
    monkeypatch, seoul_bull_workspace
):
    """The re-layer's intent: normal refinement runs with use_stored_keypoints=True
    (anchoring on the carried-in SIFT detections), not the reprojected center.
    Spy on PatchCloud.refine_normals to capture the flag the pipeline passes."""
    import cv2

    from sfmtool.patches import PatchCloud

    captured: dict = {}
    orig = PatchCloud.refine_normals

    def spy(self, recon, images, **kwargs):
        captured["use_stored_keypoints"] = kwargs.get("use_stored_keypoints")
        return orig(self, recon, images, **kwargs)

    monkeypatch.setattr(PatchCloud, "refine_normals", spy)

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    ws = recon.workspace_dir
    images = [
        np.ascontiguousarray(
            cv2.cvtColor(
                cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        )
        for name in recon.image_names
    ]
    ep.embed_patches(recon, images, resolution=12)
    assert captured.get("use_stored_keypoints") is True


def test_embed_patches_cli_subpixel_and_rounds_forward(
    monkeypatch, seoul_bull_workspace, tmp_path
):
    """End-to-end CLI plumbing for the round knobs: the values parsed from the
    command line reach `embed_patches`'s kwargs. Spying on
    the function rather than running the full pipeline keeps this test cheap
    while covering the Click choice validation + kwarg threading."""
    captured: dict = {}
    real = ep.embed_patches

    def spy(recon, images, **kwargs):
        captured["subpixel"] = kwargs.get("subpixel")
        captured["rounds"] = kwargs.get("rounds")
        captured["obliquity_weight_power"] = kwargs.get("obliquity_weight_power")
        captured["fronto_prior_weight"] = kwargs.get("fronto_prior_weight")
        return real(recon, images, **{**kwargs, "resolution": 12})

    monkeypatch.setattr(ep, "embed_patches", spy)

    out = tmp_path / "out.sfmr"
    args = [
        "embed-patches",
        str(seoul_bull_workspace),
        str(out),
        "--no-subpixel",
        "--rounds",
        "3",
        "--obliquity-weight-power",
        "2",
        "--fronto-prior-weight",
        "0.05",
    ]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert captured["subpixel"] is False
    assert captured["rounds"] == 3
    assert captured["obliquity_weight_power"] == 2.0
    assert captured["fronto_prior_weight"] == 0.05


def test_embed_patches_refine_max_views_is_lossless(seoul_bull_workspace):
    """`--refine-max-views` caps only the round-2+ normal-refinement *basis*
    (see specs/core/patch/patch-normal-refine-view-subset.md): every observation stays
    in the output and each stored bitmap's reference view is still picked from
    the full view set, so a capped run must produce the same output shape (point and
    observation counts) as the all-views default — within a hair's tolerance:
    the cap never drops an observation itself, but the slightly different
    round-2 normal can flip a borderline grazing-drop / sub-pixel-cull decision
    (observed: 10 observations in ~10000 on this fixture)."""
    import cv2

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    ws = recon.workspace_dir
    images = [
        np.ascontiguousarray(
            cv2.cvtColor(
                cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        )
        for name in recon.image_names
    ]

    # Pin the baseline to all-views (max_refine_views=0) so this stays a
    # capped-vs-uncapped comparison independent of the pipeline default (8).
    baseline = ep.embed_patches(recon, images, resolution=12, max_refine_views=0)
    capped = ep.embed_patches(recon, images, resolution=12, max_refine_views=5)

    assert capped.feature_source == "embedded_patches"
    assert abs(capped.point_count - baseline.point_count) <= max(
        1, baseline.point_count // 100
    ), f"points: capped {capped.point_count} vs baseline {baseline.point_count}"
    assert abs(capped.observation_count - baseline.observation_count) <= max(
        1, baseline.observation_count // 100
    ), (
        f"observations: capped {capped.observation_count} "
        f"vs baseline {baseline.observation_count}"
    )


def test_embed_patches_cli_refine_max_views_forwards(
    monkeypatch, seoul_bull_workspace, tmp_path
):
    """`--refine-max-views` parses (IntRange >= 0) and reaches `embed_patches` as
    the `max_refine_views` kwarg; the capped end-to-end run succeeds."""
    captured: dict = {}
    real = ep.embed_patches

    def spy(recon, images, **kwargs):
        captured["max_refine_views"] = kwargs.get("max_refine_views")
        return real(recon, images, **{**kwargs, "resolution": 12})

    monkeypatch.setattr(ep, "embed_patches", spy)

    out = tmp_path / "out.sfmr"
    args = [
        "embed-patches",
        str(seoul_bull_workspace),
        str(out),
        "--refine-max-views",
        "5",
    ]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert captured["max_refine_views"] == 5
    assert out.exists()

    # A negative cap is rejected up front by the IntRange.
    args = [
        "embed-patches",
        str(seoul_bull_workspace),
        str(tmp_path / "out2.sfmr"),
        "--refine-max-views",
        "-1",
    ]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "refine-max-views" in result.output.lower()


@pytest.mark.parametrize(
    "removed",
    [
        ["--max-iters", "3"],
        ["--search-resolution-multiplier", "2.0"],
        ["--localize-basis-views", "4"],
        ["--subpixel", "2"],
    ],
)
def test_embed_patches_cli_rejects_removed_options(
    seoul_bull_workspace, tmp_path, removed
):
    """The congealing options are gone: the localizer aligns every view to the
    reference render in one pass, so there are no rounds, no supersampled
    search and no consensus basis to set, and ``--subpixel`` is a flag that
    takes no count. Each errors out before any work happens."""
    args = [
        "embed-patches",
        str(seoul_bull_workspace),
        str(tmp_path / "out.sfmr"),
        *removed,
    ]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert not (tmp_path / "out.sfmr").exists()


def test_embed_patches_stores_rgb_bitmaps(seoul_bull_workspace):
    """Regression (channel order): the stored ``patch_bitmaps_y_x_rgba`` is RGB,
    not BGR — so the GUI, which uploads channel 0 as red, shows true colours.

    Repaint the workspace images red-dominant on disk (blue heavily suppressed;
    red + green carry the grayscale texture), then drive the pipeline through
    ``read_workspace_image`` — the exact load boundary the bug lived in. Every
    rendered stored bitmap must then carry a much larger red channel (0) than
    blue channel (2). Under the old BGR behaviour ``read_workspace_image`` handed
    the renderer blue in channel 0, so the stored channel 0 held the (suppressed)
    blue and channel 2 held the red — inverting this ratio and FAILING the test.
    """
    import cv2

    from sfmtool._embed_patches import embed_patches
    from sfmtool._workspace_image import read_workspace_image

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    ws = recon.workspace_dir

    # Repaint each source image: red + green carry the grayscale texture, blue is
    # divided by 6 (kept nonzero so no channel is degenerate for registration, but
    # far below red). cv2.imwrite expects BGR, so write [blue, green, red].
    for name in recon.image_names:
        path = str(Path(ws) / name)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        assert bgr is not None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        repainted = np.zeros_like(bgr)
        repainted[..., 0] = gray // 6  # blue (BGR channel 0)
        repainted[..., 1] = gray  # green
        repainted[..., 2] = gray  # red (BGR channel 2)
        assert cv2.imwrite(path, repainted)

    images = [read_workspace_image(ws, name) for name in recon.image_names]
    out = embed_patches(recon, images, resolution=12, patch_size=10.0)

    bitmaps = np.asarray(out.patch_bitmaps)
    assert out.point_count > 0 and bitmaps.shape[0] == out.point_count
    opaque = bitmaps[..., 3] > 0
    assert opaque.any(), "no opaque texels were rendered"
    red = bitmaps[..., 0][opaque].astype(np.float64)
    blue = bitmaps[..., 2][opaque].astype(np.float64)
    # Red carries the full texture (channel 0); blue was 1/6 of it (channel 2).
    # The ratio is scale-independent, so this locks the channel order robustly:
    # under the old BGR bug it inverts to ~1/6 and fails.
    assert red.mean() > 30.0, f"red channel unexpectedly dark ({red.mean():.1f})"
    assert red.mean() > 2.5 * blue.mean(), (
        f"stored bitmaps look BGR: red(ch0)={red.mean():.1f} "
        f"is not >> blue(ch2)={blue.mean():.1f}"
    )


def _rgb_images(recon) -> list[np.ndarray]:
    import cv2

    ws = recon.workspace_dir
    return [
        np.ascontiguousarray(
            cv2.cvtColor(
                cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        )
        for name in recon.image_names
    ]


def test_embed_patches_keeps_a_stored_reference_observation(seoul_bull_workspace):
    """An ``embedded_patches`` input's stored reference observation is the one
    every round aligns the point's views to: where its image is still in the
    output track, the output names the same image as the reference, and its
    keypoint has not moved."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = _rgb_images(recon)
    first = ep.embed_patches(recon, images, resolution=12)

    # Point each reference at another observation of its point, so the test
    # does not pass on the pick the first run already made.
    counts = np.asarray(first.observation_counts, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    refs = np.asarray(first.reference_observations, dtype=np.int32).copy()
    moved = (refs >= 0) & (counts >= 3)
    refs[moved] = (refs[moved] + 1) % counts[moved]
    first = first.clone_with_changes(reference_observations=refs)
    timg = np.asarray(first.track_image_indexes)
    kxy = np.asarray(first.keypoints_xy, dtype=np.float32)
    # Points are matched across the two runs by position, so a position two
    # points share (two tracks triangulated to one place) is left out.
    positions = [tuple(p) for p in np.asarray(first.positions)]
    shared = {p for p in positions if positions.count(p) > 1}
    ref_image = {
        positions[p]: (
            int(timg[offsets[p] + refs[p]]),
            kxy[offsets[p] + refs[p]],
        )
        for p in np.flatnonzero(moved)
        if positions[p] not in shared
    }

    second = ep.embed_patches(first, images, resolution=12)

    counts2 = np.asarray(second.observation_counts, dtype=np.int64)
    offsets2 = np.concatenate([[0], np.cumsum(counts2)[:-1]])
    refs2 = np.asarray(second.reference_observations)
    timg2 = np.asarray(second.track_image_indexes)
    kxy2 = np.asarray(second.keypoints_xy, dtype=np.float32)
    checked = 0
    for p, pos in enumerate(np.asarray(second.positions)):
        hit = ref_image.get(tuple(pos))
        if hit is None:
            continue
        image, keypoint = hit
        track = timg2[offsets2[p] : offsets2[p] + counts2[p]]
        if image not in track.tolist():
            continue
        assert refs2[p] >= 0
        row = offsets2[p] + refs2[p]
        assert timg2[row] == image, f"point {p}: reference moved off image {image}"
        assert np.array_equal(kxy2[row], keypoint), f"point {p}: reference moved"
        checked += 1
    assert 0 < len(ref_image) // 2 <= checked, (checked, len(ref_image))


def test_embed_patches_cli_absolute_localizer_gates_forward(
    monkeypatch, seoul_bull_workspace, tmp_path
):
    """`--min-absolute-zncc`, `--max-member-zncc-self-similarity-radius` and
    `--max-zncc-self-similarity-radius` parse and reach `embed_patches` as their
    matching kwargs, and the written file records them in its own `tool_options`
    (the merge would otherwise leave only the upstream operation's options
    there)."""
    captured: dict = {}
    real = ep.embed_patches
    keys = (
        "min_absolute_zncc",
        "max_member_zncc_self_similarity_radius",
        "max_zncc_self_similarity_radius",
    )

    def spy(recon, images, **kwargs):
        captured.update({k: kwargs.get(k) for k in keys})
        return real(recon, images, **{**kwargs, "resolution": 12})

    monkeypatch.setattr(ep, "embed_patches", spy)

    out = tmp_path / "gates.sfmr"
    args = [
        "embed-patches",
        str(seoul_bull_workspace),
        str(out),
        "--min-absolute-zncc",
        "0.25",
        "--max-member-zncc-self-similarity-radius",
        "0.5",
        "--max-zncc-self-similarity-radius",
        "2.75",
    ]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert captured == {
        "min_absolute_zncc": 0.25,
        "max_member_zncc_self_similarity_radius": 0.5,
        "max_zncc_self_similarity_radius": 2.75,
    }

    reloaded = SfmrReconstruction.load(str(out))
    opts = reloaded.metadata()["tool_options"]
    assert opts["min_absolute_zncc"] == 0.25
    assert opts["max_member_zncc_self_similarity_radius"] == 0.5
    assert opts["max_zncc_self_similarity_radius"] == 2.75
    assert "max_keypoint_uncertainty" not in opts


def test_embed_patches_self_similarity_cull_defaults_to_the_member_gates_bar():
    """The consensus cull's bar defaults to 2.5 on the command line and in
    `embed_patches`, the member gates' bar, and the old flag is gone."""
    import inspect

    command = main.get_command(None, "embed-patches")
    options = {p.name: p for p in command.params}
    assert options["max_zncc_self_similarity_radius"].default == 2.5
    assert "max_keypoint_uncertainty" not in options
    signature = inspect.signature(ep.embed_patches)
    assert signature.parameters["max_zncc_self_similarity_radius"].default == 2.5


def test_embed_patches_localizer_search_defaults_to_exhaustive():
    """The localizer scores every shift of its window by default, on the command
    line, in `embed_patches` and in the binding; the "+"-descent stays a choice."""
    import inspect

    from sfmtool.patches import PatchCloud

    command = main.get_command(None, "embed-patches")
    options = {p.name: p for p in command.params}
    option = options["localize_search_strategy"]
    assert option.default == "exhaustive"
    assert set(option.type.choices) == {"exhaustive", "plus_descent"}
    signature = inspect.signature(ep.embed_patches)
    assert signature.parameters["localize_search_strategy"].default == "exhaustive"
    assert 'search_strategy="exhaustive"' in (
        PatchCloud.localize_keypoints.__text_signature__ or ""
    )


def test_embed_patches_self_similarity_cull_drops_flat_and_edge_points():
    """The early cull drops the points whose round-1 consensus reads over the
    bar (a flat and an edge bitmap at the default), keeps a textured one and one
    with no consensus, and drops nothing at 0 or 3."""
    rng = np.random.default_rng(2)
    r = 12
    bitmaps = np.zeros((4, r, r, 4), np.uint8)
    bitmaps[0, ..., :3] = 128  # flat
    bitmaps[1, ..., :3] = np.where(np.arange(r) < r // 2, 50, 200)[None, :, None]
    bitmaps[2, ..., :3] = rng.integers(0, 256, (r, r, 3))  # texture
    bitmaps[:3, ..., 3] = 255  # row 3: no consensus
    locs = [{"point_index": p} for p in range(4)]

    kept, culled = ep._cull_by_zncc_self_similarity_radius(bitmaps, locs, 2.5)
    assert [loc["point_index"] for loc in kept] == [2, 3]
    assert culled == 2
    for bar in (0.0, 3.0):
        kept, culled = ep._cull_by_zncc_self_similarity_radius(bitmaps, locs, bar)
        assert len(kept) == 4 and culled == 0, bar
