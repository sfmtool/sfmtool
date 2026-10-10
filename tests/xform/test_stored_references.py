# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python writers keep each point's stored reference observation.

``tracks/reference_observations`` names the observation each point's bitmap
is, or is to be, rendered from. A refinement that moves keypoints or frames
keeps it, and renders each point that has one from the ``f32`` values the
file stores, so dropping and adding the bitmaps gives the same bytes. A
writer that aligns keypoints and renders bitmaps replaces a stored reference
it cannot use; one that writes no bitmap keeps it. A filter that removes the
reference observation itself writes ``-1``.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._embed_patches import embed_patches
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import (
    AddPatchBitmapsTransform,
    BundleAdjustTransform,
    DropPatchBitmapsTransform,
    LocalizeKeypointsTransform,
    RefineKeypointsTransform,
    RefineNormalsTransform,
)
from sfmtool.xform._images import load_workspace_images

RESOLUTION = 16


@pytest.fixture
def embedded(seoul_bull_workspace) -> SfmrReconstruction:
    """An ``embedded_patches`` reconstruction with 16x16 bitmaps, each point's
    reference the reference-view rule's pick."""
    out = Path(seoul_bull_workspace).with_name("embedded.sfmr")
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(seoul_bull_workspace),
            str(out),
            "--to-embedded-patches",
            "--add-patch-bitmaps",
            f"resolution={RESOLUTION}",
        ],
    )
    assert result.exit_code == 0, result.output
    return SfmrReconstruction.load(out)


def _other_references(recon: SfmrReconstruction) -> np.ndarray:
    """The stored references with every other picked point moved to another
    observation of its track and every fifth point at -1."""
    picked = np.asarray(recon.reference_observations)
    counts = np.asarray(recon.observation_counts).astype(np.int64)
    stored = picked.copy()
    moved = (picked >= 0) & (np.arange(len(picked)) % 2 == 1) & (counts > 1)
    stored[moved] = ((picked[moved] + 1) % counts[moved]).astype(np.int32)
    stored[::5] = -1
    assert moved.any() and (stored >= 0).any()
    return stored


def _reference_images(recon: SfmrReconstruction, refs=None) -> np.ndarray:
    """The image each point's reference observation is in, -1 for none."""
    refs = np.asarray(recon.reference_observations if refs is None else refs).astype(
        np.int64
    )
    counts = np.asarray(recon.observation_counts).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    images = np.asarray(recon.track_image_indexes).astype(np.int64)
    return np.where(refs >= 0, images[offsets + np.maximum(refs, 0)], -1)


def _point_images(recon: SfmrReconstruction, p: int) -> np.ndarray:
    counts = np.asarray(recon.observation_counts).astype(np.int64)
    start = int(counts[:p].sum())
    return np.asarray(recon.track_image_indexes)[start : start + counts[p]]


def _assert_references_follow_their_images(
    recon: SfmrReconstruction,
    stored: np.ndarray,
    out: SfmrReconstruction,
    *,
    picks: bool,
) -> tuple[int, int]:
    """Each output point, found by its position among ``recon``'s (the
    compaction does not move it), names the image its stored reference was
    in where its new track still holds that image. Where it does not, the
    point names no reference, or with ``picks`` (a pass that renders bitmaps)
    may name the observation its new bitmap is the tile of. Returns how many
    references were kept and how many images were dropped."""
    want = _reference_images(recon, stored)
    got = _reference_images(out)
    src = np.asarray(recon.positions)
    kept = lost = 0
    for k, position in enumerate(np.asarray(out.positions)):
        distances = np.linalg.norm(src - position, axis=1)
        nearest, second = np.partition(distances, 1)[:2]
        if second <= nearest:
            continue
        s = int(np.argmin(distances))
        if want[s] < 0:
            continue
        if want[s] in _point_images(out, k):
            assert got[k] == want[s], k
            kept += 1
        else:
            assert got[k] == -1 or (picks and got[k] in _point_images(out, k)), k
            lost += 1
    assert kept > 0
    return kept, lost


def _assert_drop_then_add_reproduces(out: SfmrReconstruction, sampler="per_view"):
    """Dropping and adding the bitmaps gives every point that has a reference
    the same reference and the same bitmap bytes. A point at -1 (a fused mean)
    gets the reference-view rule's pick, which may differ."""
    again = AddPatchBitmapsTransform(resolution=RESOLUTION, sampler=sampler).apply(
        DropPatchBitmapsTransform().apply(out)
    )
    refs = np.asarray(out.reference_observations)
    named = refs >= 0
    assert named.any()
    np.testing.assert_array_equal(
        np.asarray(again.reference_observations)[named], refs[named]
    )
    np.testing.assert_array_equal(
        np.asarray(again.patch_bitmaps)[named], np.asarray(out.patch_bitmaps)[named]
    )


def test_refine_keypoints_keeps_stored_references(embedded):
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    out = RefineKeypointsTransform(resolution=RESOLUTION, max_gn_steps=2).apply(recon)
    refs = np.asarray(out.reference_observations)
    np.testing.assert_array_equal(refs[stored >= 0], stored[stored >= 0])
    assert (refs[stored < 0] >= 0).any(), "a point at -1 takes the refiner's pick"
    _assert_drop_then_add_reproduces(out)

    # Without bitmaps the old ones, rendered at the old keypoints, go. The
    # stored references stay, and a point at -1 takes the reference its views
    # were aligned to, as with bitmaps, so a later render uses the reference
    # the keypoints were refined against.
    bare = RefineKeypointsTransform(
        resolution=RESOLUTION, max_gn_steps=2, bitmaps=False
    ).apply(recon)
    assert bare.patch_bitmaps is None
    bare_refs = np.asarray(bare.reference_observations)
    np.testing.assert_array_equal(bare_refs[stored >= 0], stored[stored >= 0])
    assert (bare_refs[stored < 0] >= 0).any(), "a point at -1 takes the refiner's pick"
    np.testing.assert_array_equal(bare_refs, refs)


def test_refine_normals_keeps_stored_references(embedded):
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    out = RefineNormalsTransform(
        resolution=RESOLUTION, init_steps=3, refine_levels=1, sampler="per_view"
    ).apply(recon)
    refs = np.asarray(out.reference_observations)
    np.testing.assert_array_equal(refs[stored >= 0], stored[stored >= 0])
    _assert_drop_then_add_reproduces(out)

    bare = RefineNormalsTransform(
        resolution=RESOLUTION, init_steps=3, refine_levels=1, bitmaps=False
    ).apply(recon)
    assert bare.patch_bitmaps is None
    np.testing.assert_array_equal(np.asarray(bare.reference_observations), stored)


@pytest.mark.parametrize("rounds", [1, 2])
def test_embed_patches_on_an_embedded_input_keeps_stored_references(embedded, rounds):
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    out = embed_patches(
        recon, load_workspace_images(recon), resolution=RESOLUTION, rounds=rounds
    )
    _assert_references_follow_their_images(recon, stored, out, picks=True)
    _assert_drop_then_add_reproduces(out)


def _spy_final_references(monkeypatch, *, forget_every: int = 0) -> dict:
    """Record, by point position, the image of the reference each sub-pixel
    pass of ``embed_patches`` reports its views were aligned to (``-1`` for
    none), keeping the last pass's. With ``forget_every``, every so many points
    reach the pass with no reference named, as where the localizer could not
    render a stored reference and aligned nothing."""
    import sfmtool._embed_patches as ep

    real = ep._refine_subpixel
    reported: dict = {}

    def spy(cloud, recon, images, localizations, **kwargs):
        if forget_every:
            localizations = [
                dict(loc, reference_image=None)
                if int(loc["point_index"]) % forget_every == 0
                else loc
                for loc in localizations
            ]
        out = real(cloud, recon, images, localizations, **kwargs)
        positions = [tuple(p) for p in np.asarray(recon.positions)]
        # Points are matched by position, so a position two points share (two
        # tracks triangulated to one place) is left out.
        shared = {p for p in positions if positions.count(p) > 1}
        reported.clear()
        for loc in out[0]:
            position = positions[int(loc["point_index"])]
            if position not in shared:
                image = loc.get("reference_image")
                reported[position] = -1 if image is None else int(image)
        return out

    monkeypatch.setattr(ep, "_refine_subpixel", spy)
    return reported


def _assert_output_records_the_reported_references(out, reported) -> int:
    """Every output point records the reference its views were aligned to.
    Returns how many points were checked."""
    got = _reference_images(out)
    checked = 0
    for k, position in enumerate(np.asarray(out.positions)):
        want = reported.get(tuple(position))
        if want is None:
            continue
        assert got[k] == want, (k, got[k], want)
        checked += 1
    assert checked > out.point_count // 2, (checked, out.point_count)
    return checked


def test_embed_patches_records_the_reference_the_views_were_aligned_to(
    embedded, monkeypatch
):
    """Where the localizer reports no reference for a point (its stored
    reference did not render at its keypoint, so it aligned nothing), the
    sub-pixel pass aligns the views to the reference-view rule's pick. The
    output records that pick, whose tile the bitmap is, not the stored
    reference the views were never aligned to, and dropping and adding the
    bitmaps gives the same bytes."""
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    reported = _spy_final_references(monkeypatch, forget_every=3)
    out = embed_patches(
        recon, load_workspace_images(recon), resolution=RESOLUTION, rounds=1
    )
    _assert_output_records_the_reported_references(out, reported)
    # Some of those points record another image than the stored one.
    want = _reference_images(recon, stored)
    positions = {tuple(p): i for i, p in enumerate(np.asarray(recon.positions))}
    got = _reference_images(out)
    moved = sum(
        1
        for k, p in enumerate(np.asarray(out.positions))
        if positions[tuple(p)] % 3 == 0
        and want[positions[tuple(p)]] in _point_images(out, k)
        and got[k] != want[positions[tuple(p)]]
    )
    assert moved > 0
    _assert_drop_then_add_reproduces(out)


@pytest.mark.parametrize("rounds", [1, 2])
def test_embed_patches_records_a_new_reference_where_the_obliquity_cut_drops_it(
    embedded, monkeypatch, rounds
):
    """The obliquity cut runs before each round's sub-pixel pass and drops a
    reference like any other view, so the pass's reference-view rule picks
    again from the views left and the output records that pick. No bitmap is
    left as a removed observation's render under ``-1``, and dropping and
    adding the bitmaps gives the same bytes."""
    reported = _spy_final_references(monkeypatch)
    out = embed_patches(
        embedded,
        load_workspace_images(embedded),
        resolution=RESOLUTION,
        rounds=rounds,
        max_obliquity_deg=30.0,
    )
    assert out.observation_count < embedded.observation_count
    _assert_output_records_the_reported_references(out, reported)
    _assert_drop_then_add_reproduces(out)


def test_embed_patches_rounds_keep_the_references_without_bitmaps(
    embedded, monkeypatch
):
    """With the self-similarity cull off, round 1's sub-pixel pass renders no
    bitmaps. The references its views were aligned to are still recorded in
    the intermediate compaction, so round 2 refines against the same ones."""
    import sfmtool._embed_patches as ep

    compacted: list[SfmrReconstruction] = []
    real = ep.compact_to_embedded_patches

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        compacted.append(out)
        return out

    monkeypatch.setattr(ep, "compact_to_embedded_patches", spy)
    embed_patches(
        embedded,
        load_workspace_images(embedded),
        resolution=RESOLUTION,
        rounds=2,
        max_zncc_self_similarity_radius=0,
    )
    # The first compaction is round 2's input; the last is the final result.
    assert len(compacted) >= 2
    intermediate = np.asarray(compacted[0].reference_observations)
    assert (intermediate >= 0).mean() > 0.9, (intermediate >= 0).mean()


def test_localize_keypoints_remaps_stored_references(embedded):
    """The localizer rebuilds the tracks and drops the bitmaps. It aligns each
    point's views to its stored reference observation, which it keeps, so each
    point keeps its reference where its track still holds that image; a point
    that stored none, or whose reference the grazing pre-filter turned away,
    records the reference-view rule's pick its views were aligned to."""
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    out = LocalizeKeypointsTransform().apply(recon)
    assert out.patch_bitmaps is None
    kept, lost = _assert_references_follow_their_images(recon, stored, out, picks=True)
    assert lost <= kept // 10, (kept, lost)
    # The points that stored none now name the reference they were aligned to.
    assert (np.asarray(out.reference_observations) >= 0).mean() > 0.9


def test_bundle_adjust_keeps_stored_references(embedded):
    """Bundle adjustment rebuilds the tracks from its own solve; every point
    keeps the reference to the same image."""
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    out = BundleAdjustTransform().apply(recon)
    assert out.point_count == recon.point_count
    np.testing.assert_array_equal(
        _reference_images(out), _reference_images(recon, stored)
    )


def test_render_bitmaps_referenced_only_skips_points_at_minus_one(embedded):
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    images = load_workspace_images(recon)
    bitmaps, refs = recon.patches.render_bitmaps(
        recon, images, resolution=RESOLUTION, referenced_only=True
    )
    bitmaps, refs = np.asarray(bitmaps), np.asarray(refs)
    np.testing.assert_array_equal(refs, stored)
    assert not bitmaps[stored < 0].any(), "a skipped point has a zero row"
    assert bitmaps[stored >= 0].any()


def test_a_filter_that_removes_the_reference_observation_writes_minus_one(embedded):
    """A point whose reference image is removed keeps its bitmap, the render of
    an observation it no longer has, and its reference becomes -1: no reference
    observation in its track."""
    refs = np.asarray(embedded.reference_observations)
    counts = np.asarray(embedded.observation_counts).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    images_of = np.asarray(embedded.track_image_indexes).astype(np.int64)
    ref_image = np.where(refs >= 0, images_of[offsets + np.maximum(refs, 0)], -1)
    values, hits = np.unique(ref_image[ref_image >= 0], return_counts=True)
    dropped = int(values[np.argmax(hits)])
    keep = np.array(
        [i for i in range(embedded.image_count) if i != dropped], dtype=np.uint32
    )
    subset = embedded.subset_by_image_indices(keep)
    assert subset.point_count == embedded.point_count, "the fixture keeps every point"
    sub_refs = np.asarray(subset.reference_observations)
    lost = ref_image == dropped
    assert lost.any()
    assert (sub_refs[lost] == -1).all()
    # The bitmap stays as it was: the old reference's render.
    np.testing.assert_array_equal(
        np.asarray(subset.patch_bitmaps)[lost], np.asarray(embedded.patch_bitmaps)[lost]
    )
    # Every other reference still names the observation of the same image.
    sub_counts = np.asarray(subset.observation_counts).astype(np.int64)
    sub_offsets = np.concatenate([[0], np.cumsum(sub_counts)[:-1]])
    sub_images = np.asarray(subset.track_image_indexes).astype(np.int64)
    image_map = {int(old): new for new, old in enumerate(keep)}
    for p in np.flatnonzero((refs >= 0) & ~lost):
        assert sub_images[sub_offsets[p] + sub_refs[p]] == image_map[ref_image[p]]


def _unrenderable_references(recon: SfmrReconstruction) -> tuple:
    """``recon`` with the reference observation of every third point that
    stores one moved to the image's corner, where its core does not render, so
    nothing can be aligned to it. Returns the new reconstruction and the
    points moved."""
    refs = np.asarray(recon.reference_observations).astype(np.int64)
    counts = np.asarray(recon.observation_counts).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    kxy = np.asarray(recon.keypoints_xy, dtype=np.float32).copy()
    points = [p for p in np.flatnonzero((refs >= 0) & (counts >= 3)) if p % 3 == 0]
    assert points
    for p in points:
        kxy[offsets[p] + refs[p]] = (0.5, 0.5)
    return recon.clone_with_changes(keypoints_xy=kxy), points


def test_writers_without_bitmaps_keep_a_reference_nothing_was_aligned_to(embedded):
    """A stored reference whose core does not render at its keypoint aligns
    nothing. A writer that writes no bitmap keeps it all the same:
    ``--refine-keypoints bitmaps=false`` keeps every stored reference, and
    ``--localize-keypoints`` keeps it wherever its image is still in the
    rebuilt track. ``--refine-keypoints`` with bitmaps replaces it by the
    reference its views were aligned to, or ``-1`` with the fused mean."""
    recon, points = _unrenderable_references(embedded)
    stored = np.asarray(recon.reference_observations)

    bare = RefineKeypointsTransform(
        resolution=RESOLUTION, max_gn_steps=2, bitmaps=False
    ).apply(recon)
    np.testing.assert_array_equal(
        np.asarray(bare.reference_observations)[stored >= 0], stored[stored >= 0]
    )

    rendered = RefineKeypointsTransform(resolution=RESOLUTION, max_gn_steps=2).apply(
        recon
    )
    refs = np.asarray(rendered.reference_observations)
    assert all(refs[p] != stored[p] for p in points), "an unusable one is replaced"
    # Elsewhere the stored references stay.
    others = (stored >= 0) & ~np.isin(np.arange(len(stored)), points)
    assert (refs[others] == stored[others]).mean() > 0.9

    # A wide shift bar, so the moved reference, which no search places, is
    # not dropped for its distance from the projection and stays in the track.
    out = LocalizeKeypointsTransform(max_shift_px=1e6).apply(recon)
    want = _reference_images(recon, stored)
    got = _reference_images(out)
    src = np.asarray(recon.positions)
    kept = 0
    for k, position in enumerate(np.asarray(out.positions)):
        distances = np.linalg.norm(src - position, axis=1)
        nearest, second = np.partition(distances, 1)[:2]
        s = int(np.argmin(distances))
        if second <= nearest or s not in points:
            continue
        if want[s] in _point_images(out, k):
            assert got[k] == want[s], k
            kept += 1
    assert kept > 0
