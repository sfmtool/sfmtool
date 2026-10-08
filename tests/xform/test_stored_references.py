# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python writers keep each point's stored reference observation.

``tracks/reference_observations`` names the observation each point's bitmap
is, or is to be, rendered from. A refinement that moves keypoints or frames
keeps it, and renders each point that has one from the ``f32`` values the
file stores, so dropping and adding the bitmaps gives the same bytes. A
filter that removes the reference observation itself writes ``-1``.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._embed_patches import embed_patches
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
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

    # Without bitmaps the old ones, rendered at the old keypoints, go; the
    # references stay for a later render.
    bare = RefineKeypointsTransform(
        resolution=RESOLUTION, max_gn_steps=2, bitmaps=False
    ).apply(recon)
    assert bare.patch_bitmaps is None
    np.testing.assert_array_equal(np.asarray(bare.reference_observations), stored)


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


def test_localize_keypoints_remaps_stored_references(embedded):
    """The localizer rebuilds the tracks and drops the bitmaps; each point keeps
    its reference where its track still holds that image, and -1 where the
    localizer dropped it."""
    stored = _other_references(embedded)
    recon = embedded.clone_with_changes(reference_observations=stored)
    out = LocalizeKeypointsTransform().apply(recon)
    assert out.patch_bitmaps is None
    _, lost = _assert_references_follow_their_images(recon, stored, out, picks=False)
    # The localizer drops some reference images on this fixture, so the -1
    # branch is exercised.
    assert lost > 0


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
