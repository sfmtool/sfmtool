# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the background-floor track-cluster matcher Rust bindings."""

import numpy as np
import pytest

from sfmtool._sfmtool.matching import (
    background_floor_clusters,
    clusters_to_pair_matches,
)


N_IMAGES = 4
N_POINTS = 5
N_BACKGROUND = 20
DIM = 128


def _planted_corpus(seed=42):
    """Corpus with N_POINTS planted cross-image points plus background.

    Each image's rows start with the planted observations (base descriptor +
    small jitter), so a planted row's feature index equals its point id;
    N_BACKGROUND random rows per image follow.
    """
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 256, size=(N_POINTS, DIM), dtype=np.int16)

    blocks = []
    image_starts = [0]
    for _ in range(N_IMAGES):
        jitter = rng.integers(-2, 3, size=(N_POINTS, DIM), dtype=np.int16)
        planted = np.clip(bases + jitter, 0, 255).astype(np.uint8)
        background = rng.integers(0, 256, size=(N_BACKGROUND, DIM), dtype=np.uint8)
        blocks.append(np.vstack([planted, background]))
        image_starts.append(image_starts[-1] + len(blocks[-1]))

    corpus = np.ascontiguousarray(np.vstack(blocks))
    return corpus, np.asarray(image_starts, dtype=np.uint32)


def _exact_kwargs(n):
    """Single-leaf forest config: the search is exact by construction."""
    return dict(d=16, num_trees=1, leaf_size=n, max_leaf_checks=n)


class TestBackgroundFloorClusters:
    def test_shapes_dtypes_and_csr_validity(self):
        corpus, image_starts = _planted_corpus()
        starts, images, features = background_floor_clusters(
            corpus, image_starts, **_exact_kwargs(len(corpus))
        )
        assert starts.dtype == np.uint32
        assert images.dtype == np.uint32
        assert features.dtype == np.uint32
        assert starts[0] == 0
        assert np.all(np.diff(starts.astype(np.int64)) >= 0)
        assert starts[-1] == len(images)
        assert len(images) == len(features)

    def test_planted_points_form_clusters(self):
        corpus, image_starts = _planted_corpus()
        starts, images, features = background_floor_clusters(
            corpus, image_starts, **_exact_kwargs(len(corpus))
        )
        assert len(starts) - 1 == N_POINTS
        for c in range(len(starts) - 1):
            lo, hi = int(starts[c]), int(starts[c + 1])
            assert hi - lo == N_IMAGES
            # Members sorted by image, one feature per image.
            assert np.all(np.diff(images[lo:hi].astype(np.int64)) > 0)
            # All members carry the same planted point id.
            assert np.all(features[lo:hi] == features[lo])
            assert features[lo] < N_POINTS

    def test_clusters_to_pair_matches_roundtrip(self):
        corpus, image_starts = _planted_corpus()
        starts, images, features = background_floor_clusters(
            corpus, image_starts, **_exact_kwargs(len(corpus))
        )
        pairs, counts, feat_idx, distances = clusters_to_pair_matches(
            starts, images, features, corpus, image_starts
        )
        assert pairs.dtype == np.uint32
        assert pairs.shape[1] == 2
        assert counts.dtype == np.uint32
        assert feat_idx.dtype == np.uint32
        assert distances.dtype == np.float32

        # Pairs sorted ascending with i < j.
        assert np.all(pairs[:, 0] < pairs[:, 1])
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        assert np.array_equal(order, np.arange(len(pairs)))

        assert counts.sum() == len(feat_idx) == len(distances)

        # Each planted point appears in every image, so every image pair
        # carries N_POINTS matches between identical feature ids.
        assert len(pairs) == N_IMAGES * (N_IMAGES - 1) // 2
        offset = 0
        for k in range(len(pairs)):
            count = int(counts[k])
            block = feat_idx[offset : offset + count]
            assert sorted(block[:, 0]) == list(range(N_POINTS))
            assert np.array_equal(block[:, 0], block[:, 1])
            offset += count

        # Distances are true (non-squared) Euclidean L2.
        offset = 0
        for k in range(len(pairs)):
            row_i = int(image_starts[pairs[k, 0]]) + int(feat_idx[offset, 0])
            row_j = int(image_starts[pairs[k, 1]]) + int(feat_idx[offset, 1])
            expected = np.linalg.norm(
                corpus[row_i].astype(np.float64) - corpus[row_j].astype(np.float64)
            )
            assert distances[offset] == pytest.approx(expected, rel=1e-5)
            offset += int(counts[k])

    def test_determinism(self):
        corpus, image_starts = _planted_corpus()
        kwargs = dict(seed=7, preset="accurate")
        r1 = background_floor_clusters(corpus, image_starts, **kwargs)
        r2 = background_floor_clusters(corpus, image_starts, **kwargs)
        for a, b in zip(r1, r2):
            assert np.array_equal(a, b)

    def test_wide_alpha_does_not_overflow(self):
        # alpha >= 1 widens the radius past the floor; the candidate buffers
        # are sized for the full query width, so this must not panic.
        corpus, image_starts = _planted_corpus()
        starts, images, features = background_floor_clusters(
            corpus, image_starts, alpha=1.5, **_exact_kwargs(len(corpus))
        )
        assert starts[0] == 0
        assert starts[-1] == len(images) == len(features)

    def test_min_size_filters_small_clusters(self):
        # Each planted point spans N_IMAGES images; requiring more yields none.
        corpus, image_starts = _planted_corpus()
        starts, images, features = background_floor_clusters(
            corpus, image_starts, min_size=N_IMAGES + 1, **_exact_kwargs(len(corpus))
        )
        assert np.array_equal(starts, [0])
        assert len(images) == 0 and len(features) == 0


class TestValidation:
    def test_wrong_descriptor_dtype_raises(self):
        corpus, image_starts = _planted_corpus()
        with pytest.raises(TypeError, match="uint8"):
            background_floor_clusters(corpus.astype(np.float32), image_starts)

    def test_wrong_descriptor_width_raises(self):
        rng = np.random.default_rng(0)
        corpus = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
        starts = np.asarray([0, 32, 64], dtype=np.uint32)
        with pytest.raises(ValueError, match=r"\(N, 128\)"):
            background_floor_clusters(corpus, starts)

    def test_wrong_image_starts_dtype_raises(self):
        corpus, image_starts = _planted_corpus()
        with pytest.raises(TypeError, match="uint32"):
            background_floor_clusters(corpus, image_starts.astype(np.int64))

    def test_corpus_smaller_than_floor_raises(self):
        corpus, image_starts = _planted_corpus()
        with pytest.raises(ValueError, match="need more than d"):
            background_floor_clusters(corpus, image_starts, d=len(corpus))

    def test_zero_d_raises(self):
        # d = 0 would make the query width 1 and a zero-stride buffer; reject it
        # cleanly rather than panicking in the core.
        corpus, image_starts = _planted_corpus()
        with pytest.raises(ValueError, match="at least 1"):
            background_floor_clusters(corpus, image_starts, d=0)

    def test_bad_offsets_raise(self):
        corpus, _ = _planted_corpus()
        bad = np.asarray([0, len(corpus) // 2], dtype=np.uint32)
        with pytest.raises(ValueError, match="image_starts"):
            background_floor_clusters(corpus, bad, **_exact_kwargs(len(corpus)))

    def test_conversion_validates_member_ranges(self):
        corpus, image_starts = _planted_corpus()
        starts = np.asarray([0, 2], dtype=np.uint32)
        images = np.asarray([0, 1], dtype=np.uint32)
        features = np.asarray([0, 10_000], dtype=np.uint32)
        with pytest.raises(ValueError, match="out of range"):
            clusters_to_pair_matches(starts, images, features, corpus, image_starts)

    def test_conversion_validates_csr(self):
        corpus, image_starts = _planted_corpus()
        starts = np.asarray([0, 5], dtype=np.uint32)  # ends past M
        images = np.asarray([0, 1], dtype=np.uint32)
        features = np.asarray([0, 0], dtype=np.uint32)
        with pytest.raises(ValueError, match="cluster_starts"):
            clusters_to_pair_matches(starts, images, features, corpus, image_starts)
