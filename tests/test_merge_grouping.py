# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""``sfm merge`` groups its point pairs into connected components and measures
each group's spread from its centroid, both over whole arrays at once. These
tests check both against the plain per-pair and per-group computations they
replace."""

from pathlib import Path

import numpy as np
import pytest

from sfmtool.merge.correspondences import (
    _connected_groups,
    _group_max_distances,
    find_point_correspondences,
)
from sfmtool.reconstruction import RangeExpr, SfmrReconstruction
from sfmtool.xform import IncludeRangeFilter, apply_transforms


def _set_union_groups(first, second):
    """The grouping the merge did one pair at a time: every node of a pair
    shares one set, two sets are joined into a new one, and the groups come
    in the order their first node was seen."""
    sets = {}
    for a, b in zip(first, second):
        sets.setdefault(a, {a})
        sets.setdefault(b, {b})
        joined = sets[a] | sets[b]
        for node in joined:
            sets[node] = joined
    seen = set()
    groups = []
    for group in sets.values():
        key = tuple(sorted(group))
        if key not in seen:
            seen.add(key)
            groups.append(group)
    return groups


def _split(members, group_starts):
    ends = list(group_starts[1:]) + [len(members)]
    return [members[s:e].tolist() for s, e in zip(group_starts, ends)]


class TestConnectedGroups:
    def test_chains_and_cycles_match_set_union(self):
        # A chain built from both ends, a cycle, a pair seen twice, a star,
        # and two groups joined by a later pair.
        pairs = [
            (10, 20),
            (40, 30),
            (20, 30),
            (5, 6),
            (6, 7),
            (7, 5),
            (100, 101),
            (100, 101),
            (50, 1),
            (50, 2),
            (50, 3),
            (60, 61),
            (62, 63),
            (61, 62),
        ]
        first = np.array([a for a, _ in pairs], dtype=np.int64)
        second = np.array([b for _, b in pairs], dtype=np.int64)
        members, starts = _connected_groups(first, second)
        groups = _split(members, starts)
        assert groups == [
            [10, 20, 40, 30],
            [5, 6, 7],
            [100, 101],
            [50, 1, 2, 3],
            [60, 61, 62, 63],
        ]
        assert [set(g) for g in groups] == _set_union_groups(first, second)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_pairs_match_set_union(self, seed):
        rng = np.random.default_rng(seed)
        first = rng.integers(0, 400, 300)
        second = rng.integers(400, 800, 300)
        members, starts = _connected_groups(first, second)
        groups = [set(g) for g in _split(members, starts)]
        assert groups == _set_union_groups(first.tolist(), second.tolist())
        assert len(members) == len(np.unique(np.concatenate([first, second])))

    def test_no_pairs_gives_no_groups(self):
        empty = np.zeros(0, dtype=np.int64)
        members, starts = _connected_groups(empty, empty)
        assert len(members) == 0
        assert len(starts) == 0


def _per_group_max_distance(positions_xyzw, group):
    """The spread of one group, computed on its own."""
    positions = np.array([positions_xyzw[r][p, :3] for r, p in group])
    centroid = np.mean(positions, axis=0)
    return np.max(np.linalg.norm(positions - centroid, axis=1))


def _flatten(groups):
    member_recon = np.array([r for g in groups for r, _ in g], dtype=np.int64)
    member_point = np.array([p for g in groups for _, p in g], dtype=np.int64)
    group_starts = np.cumsum([0] + [len(g) for g in groups[:-1]])
    return member_recon, member_point, group_starts


class TestGroupMaxDistances:
    def test_matches_per_group_computation(self):
        rng = np.random.default_rng(3)
        positions_xyzw = []
        for count in (40, 25, 60):
            xyzw = np.ones((count, 4))
            xyzw[:, :3] = rng.normal(0.0, 10.0, (count, 3))
            positions_xyzw.append(xyzw)
        groups = []
        for size in [2, 2, 3, 4, 5, 6, 9, 2, 3]:
            recons = rng.integers(0, 3, size)
            groups.append(
                [(int(r), int(rng.integers(0, len(positions_xyzw[r])))) for r in recons]
            )
        distances = _group_max_distances(positions_xyzw, *_flatten(groups))
        expected = [_per_group_max_distance(positions_xyzw, g) for g in groups]
        np.testing.assert_allclose(distances, expected, rtol=1e-12, atol=0.0)
        # Groups of fewer than 8 points sum their positions in the same order as
        # the per-group mean, so they agree to the last bit.
        small = [i for i, g in enumerate(groups) if len(g) < 8]
        np.testing.assert_array_equal(distances[small], np.array(expected)[small])

    def test_point_at_infinity_raises(self):
        xyzw = np.ones((3, 4))
        xyzw[2] = [0.0, 0.0, 1.0, 0.0]
        member_recon, member_point, starts = _flatten([[(0, 0), (1, 2)]])
        with pytest.raises(ValueError, match="point at infinity"):
            _group_max_distances(
                [np.ones((3, 4)), xyzw], member_recon, member_point, starts
            )


def _jittered_subset(path, out_path, range_expr, seed):
    """An image range of the reconstruction, saved with its finite points
    moved by a little noise, so that the groups have a spread to measure."""
    recon = apply_transforms(
        SfmrReconstruction.load(path), [IncludeRangeFilter(RangeExpr(range_expr))]
    )
    xyzw = np.array(recon.positions_xyzw)
    finite = xyzw[:, 3] != 0.0
    rng = np.random.default_rng(seed)
    xyzw[finite, :3] += rng.normal(0.0, 1e-3, (int(finite.sum()), 3))
    recon = recon.clone_with_changes(positions=np.ascontiguousarray(xyzw))
    recon.save(out_path, operation="test")
    return SfmrReconstruction.load(out_path)


def test_find_point_correspondences_matches_per_pair_grouping(
    seoul_bull_workspace, tmp_path, monkeypatch
):
    """Three overlapping subsets: the accepted groups are the set-union groups
    of the pairs whose spread is within the percentile threshold."""
    import sfmtool.merge.correspondences as correspondences

    recons = [
        _jittered_subset(seoul_bull_workspace, tmp_path / f"{i}.sfmr", r, i)
        for i, r in enumerate(["1-8", "5-13", "10-17"])
    ]
    image_mapping = {}
    for recon_idx, recon in enumerate(recons):
        for img_idx, name in enumerate(recon.image_names):
            image_mapping.setdefault(Path(name).as_posix(), []).append(
                (recon_idx, img_idx)
            )

    # Record the pairs the merge groups, in the order it reads them.
    first, second = [], []
    finite_point_pairs = correspondences._finite_point_pairs

    def recording(source, target, shared_images):
        ids_1, ids_2, found = finite_point_pairs(source, target, shared_images)
        r1, r2 = recons.index(source), recons.index(target)
        first.extend((r1, int(p)) for p in ids_1)
        second.extend((r2, int(p)) for p in ids_2)
        return ids_1, ids_2, found

    monkeypatch.setattr(correspondences, "_finite_point_pairs", recording)
    groups = find_point_correspondences(recons, image_mapping, 90.0)

    reference = _set_union_groups(first, second)
    positions_xyzw = [np.asarray(r.positions_xyzw) for r in recons]
    spreads = np.array(
        [_per_group_max_distance(positions_xyzw, list(g)) for g in reference]
    )
    threshold = np.percentile(spreads, 90.0)
    expected = [g for g, s in zip(reference, spreads) if s <= threshold]

    assert any(len(g) >= 3 for g in reference)
    assert 0 < len(groups) < len(reference)
    assert [set(g) for g in groups] == expected
