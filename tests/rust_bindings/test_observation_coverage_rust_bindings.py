# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observation coverage PyO3 binding.

``ObservationCoverage`` rasterizes every observation's image-space footprint
into per-image occupancy grids and answers batch queries against them: what a
region's claim count is, how much of a neighbourhood is claimed, and which
directions around a point are still open. See specs/core/observation-coverage.md.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.analysis import ObservationCoverage

# A query point on a 21x21-cell grid at 1 px cells, offset from every cell
# center so no displacement to a center is axis-aligned: with four sectors the
# boundaries are the axes, so no cell can sit on one.
SECTOR_QUERY = (10.7, 10.9)
# Reaches three cells out; the nearest center distances either side of it are
# 3.26 and 3.41, so no center sits on the disk boundary.
SECTOR_RADIUS = 3.3
# The cell indexes that disk can touch on either axis, inclusive.
SPAN_LO, SPAN_HI = 7, 13


def _coverage(sizes, observations, cell_px=4):
    """Build from ``[width, height]`` sizes and ``(image, x, y, radius)`` tuples."""
    return ObservationCoverage(
        np.asarray(sizes, dtype=np.uint32).reshape(-1, 2),
        np.array([o[0] for o in observations], dtype=np.uint32),
        np.array([[o[1], o[2]] for o in observations], dtype=np.float64).reshape(-1, 2),
        np.array([o[3] for o in observations], dtype=np.float32),
        cell_px,
    )


def _stamp(cx, cy, image=0):
    """An observation claiming exactly cell ``(cx, cy)`` at ``cell_px = 1``.

    A 0.4 px radius at a cell center reaches no other center — the nearest is
    1.0 px away.
    """
    return (image, cx + 0.5, cy + 0.5, 0.4)


def _query(points, radii=None, images=None):
    """Pack a list of ``(x, y)`` into the batch-query argument arrays."""
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    idx = (
        np.zeros(len(xy), dtype=np.uint32)
        if images is None
        else np.asarray(images, dtype=np.uint32)
    )
    if radii is None:
        return idx, xy
    return idx, xy, np.asarray(radii, dtype=np.float32)


# ── Building and grid access ──────────────────────────────────────────────


def test_disk_covers_the_cells_whose_centers_it_contains():
    # 40x40 px at 4 px cells: 10x10 cells with centers at 2, 6, 10, ... The
    # keypoint sits on the center of cell (2, 2); the edge-adjacent centers are
    # 4 px away and the diagonal ones 5.657 px away.
    coverage = _coverage([[40, 40]], [(0, 10.0, 10.0, 5.1)])
    grid = coverage.grid(0)
    assert grid.shape == (10, 10)
    assert grid.dtype == np.uint8
    # argwhere yields (row, col) = (cy, cx).
    np.testing.assert_array_equal(
        np.argwhere(grid > 0), [[1, 2], [2, 1], [2, 2], [2, 3], [3, 2]]
    )

    # Widen past the diagonals but not to the next ring at 8 px.
    coverage = _coverage([[40, 40]], [(0, 10.0, 10.0, 6.0)])
    assert int((coverage.grid(0) > 0).sum()) == 9


def test_grid_dimensions_round_up():
    # 10 px at 4 px cells is 3 cells, the last covering pixels [8, 12).
    coverage = _coverage([[10, 6]], [])
    assert coverage.grid(0).shape == (2, 3)
    assert coverage.image_count == 1
    assert coverage.cell_px == 4


def test_counts_accumulate_and_saturate():
    # Both footprints reach only the center of cell (2, 2).
    coverage = _coverage([[40, 40]], [(0, 10.0, 10.0, 1.0)] * 2)
    assert coverage.grid(0)[2, 2] == 2

    coverage = _coverage([[40, 40]], [(0, 10.0, 10.0, 1.0)] * 300)
    assert coverage.grid(0)[2, 2] == 255


def test_footprints_are_clipped_and_degenerate_ones_are_dropped():
    # Only the center of cell (0, 0), 2.83 px away, is in reach at the corner;
    # the next centers are 6.32 px away.
    coverage = _coverage([[40, 40]], [(0, 0.0, 0.0, 5.0)])
    np.testing.assert_array_equal(np.argwhere(coverage.grid(0) > 0), [[0, 0]])

    coverage = _coverage(
        [[40, 40]],
        [
            (0, -100.0, -100.0, 5.0),
            (0, 500.0, 500.0, 5.0),
            (0, 10.0, 10.0, 0.0),
            (0, 10.0, 10.0, -3.0),
            (0, 10.0, 10.0, np.nan),
            (0, np.nan, 10.0, 5.0),
        ],
    )
    assert int(coverage.grid(0).sum()) == 0


def test_images_are_independent():
    coverage = _coverage([[40, 40], [40, 40]], [(1, 10.0, 10.0, 1.0)])
    assert coverage.image_count == 2
    assert int(coverage.grid(0).sum()) == 0
    np.testing.assert_array_equal(np.argwhere(coverage.grid(1) > 0), [[2, 2]])


def test_accessor_dtypes_are_accepted_as_given():
    # float64 keypoints, float32 radii, uint32 image indexes: the dtypes an
    # SfmrReconstruction hands out.
    sizes = np.array([[40, 40]], dtype=np.uint32)
    coverage = ObservationCoverage(
        sizes,
        np.zeros(2, dtype=np.uint32),
        np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert coverage.cell_px == 4, "cell_px defaults to 4"
    assert coverage.grid(0)[2, 2] == 2

    counts = coverage.counts_at(
        np.zeros(1, dtype=np.uint32), np.array([[10.0, 10.0]], dtype=np.float64)
    )
    assert counts.dtype == np.uint8


# ── counts_at ─────────────────────────────────────────────────────────────


def test_counts_at_reads_the_containing_cell():
    coverage = _coverage([[40, 40]], [(0, 10.0, 10.0, 1.0)] * 2)
    # Cell (2, 2) spans pixels [8, 12) on both axes.
    counts = coverage.counts_at(
        *_query([(8.0, 8.0), (11.9, 11.9), (7.9, 10.0), (10.0, 12.1)])
    )
    assert counts.dtype == np.uint8
    np.testing.assert_array_equal(counts, [2, 2, 0, 0])


def test_counts_at_returns_zero_outside_the_grid():
    coverage = _coverage([[40, 40]], [(0, 10.0, 10.0, 1.0)])
    counts = coverage.counts_at(
        *_query(
            [
                (-0.5, 10.0),
                (10.0, -0.5),
                (40.5, 10.0),
                (10.0, 40.5),
                (np.nan, 10.0),
            ]
        )
    )
    np.testing.assert_array_equal(counts, [0, 0, 0, 0, 0])


# ── covered_fraction ──────────────────────────────────────────────────────

# A 5x5-cell image at 1 px cells, queried from the center of cell (2, 2) with a
# radius that reaches the four edge-adjacent centers (1.0 away) but not the
# diagonals (1.414 away): a five-cell neighbourhood.
PLUS = [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]
PLUS_RADIUS = 1.1


def test_covered_fraction_over_an_empty_partial_and_full_neighbourhood():
    empty = _coverage([[5, 5]], [], cell_px=1)
    fractions = empty.covered_fraction(*_query([(2.5, 2.5)], [PLUS_RADIUS]))
    assert fractions.dtype == np.float32
    np.testing.assert_array_equal(fractions, [0.0])

    partial = _coverage([[5, 5]], [_stamp(*c) for c in PLUS[:2]], cell_px=1)
    fractions = partial.covered_fraction(*_query([(2.5, 2.5)], [PLUS_RADIUS]))
    assert fractions[0] == pytest.approx(0.4), "2 of 5 cells"

    full = _coverage([[5, 5]], [_stamp(*c) for c in PLUS], cell_px=1)
    fractions = full.covered_fraction(*_query([(2.5, 2.5)], [PLUS_RADIUS]))
    np.testing.assert_array_equal(fractions, [1.0])


def test_covered_fraction_is_zero_when_no_cell_center_falls_in_the_disk():
    coverage = _coverage([[5, 5]], [_stamp(2, 2)], cell_px=1)
    # A degenerate radius, a radius under the 0.707 from a cell corner to the
    # nearest center, a point outside the grid, and a non-finite radius.
    fractions = coverage.covered_fraction(
        *_query(
            [(2.5, 2.5), (2.0, 2.0), (-20.0, -20.0), (2.5, 2.5)],
            [0.0, 0.4, 2.0, np.nan],
        )
    )
    np.testing.assert_array_equal(fractions, [0.0, 0.0, 0.0, 0.0])


# ── uncovered_sectors ─────────────────────────────────────────────────────


def _sector_scene(observations):
    return _coverage([[21, 21]], observations, cell_px=1)


def _sectors(coverage, points, radius=SECTOR_RADIUS, n_sectors=4):
    radii = [radius] * len(points)
    return coverage.uncovered_sectors(*_query(points, radii), n_sectors)


def test_uncovered_sectors_reports_every_direction_of_an_empty_image():
    coverage = _sector_scene([])
    masks = _sectors(coverage, [SECTOR_QUERY])
    assert masks.dtype == np.uint32
    np.testing.assert_array_equal(masks, [0b1111])
    np.testing.assert_array_equal(
        _sectors(coverage, [SECTOR_QUERY], n_sectors=8), [0b1111_1111]
    )


def test_uncovered_sectors_reports_nothing_when_the_disk_is_covered():
    # One footprint reaching well past the query disk. 5.15 px cannot land on a
    # center distance here: those squared distances carry at most two decimals.
    coverage = _sector_scene([(0, SECTOR_QUERY[0], SECTOR_QUERY[1], 5.15)])
    np.testing.assert_array_equal(_sectors(coverage, [SECTOR_QUERY]), [0])


def test_uncovered_sectors_reports_exactly_the_uncovered_half():
    # Sectors 0 and 1 are the +y half, sectors 2 and 3 the -y half. Cell centers
    # sit at cy + 0.5, so relative to y = 10.9 the +y half is exactly cy >= 11.
    coverage = _sector_scene(
        [
            _stamp(cx, cy)
            for cy in range(11, SPAN_HI + 1)
            for cx in range(SPAN_LO, SPAN_HI + 1)
        ]
    )
    np.testing.assert_array_equal(_sectors(coverage, [SECTOR_QUERY]), [0b1100])


def test_uncovered_sectors_bins_each_quadrant():
    # Leave exactly one cell of the disk unclaimed and read back its sector.
    # Displacements of the hole's center from (10.7, 10.9):
    #   (11, 11) -> ( 0.8,  0.6)  sector 0
    #   ( 9, 11) -> (-1.2,  0.6)  sector 1
    #   (10, 10) -> (-0.2, -0.4)  sector 2
    #   (11,  9) -> ( 0.8, -1.4)  sector 3
    for hole, expected in [
        ((11, 11), 0b0001),
        ((9, 11), 0b0010),
        ((10, 10), 0b0100),
        ((11, 9), 0b1000),
    ]:
        coverage = _sector_scene(
            [
                _stamp(cx, cy)
                for cy in range(SPAN_LO, SPAN_HI + 1)
                for cx in range(SPAN_LO, SPAN_HI + 1)
                if (cx, cy) != hole
            ]
        )
        np.testing.assert_array_equal(
            _sectors(coverage, [SECTOR_QUERY]), [expected], err_msg=f"hole {hole}"
        )


def test_uncovered_sectors_ignores_directions_outside_the_grid():
    # A 3x3-cell image queried near the top-left corner: every in-grid center in
    # reach lies in sector 0 (dx > 0, dy > 0), so no other bit can be set.
    coverage = _coverage([[3, 3]], [], cell_px=1)
    np.testing.assert_array_equal(
        _sectors(coverage, [(0.3, 0.4)], radius=1.5), [0b0001]
    )


def test_uncovered_sectors_skips_the_cell_at_the_query_point():
    # A radius reaching only the cell the query sits on top of: that cell has no
    # direction, so nothing is reported even though it is uncovered.
    coverage = _sector_scene([])
    np.testing.assert_array_equal(_sectors(coverage, [(10.5, 10.5)], radius=0.9), [0])


# ── image_covered_fraction ────────────────────────────────────────────────


def test_image_covered_fraction_counts_the_whole_grid():
    coverage = _coverage([[5, 5], [5, 5]], [_stamp(*c) for c in PLUS], cell_px=1)
    assert coverage.image_covered_fraction(0) == pytest.approx(0.2), "5 of 25 cells"
    assert coverage.image_covered_fraction(1) == 0.0


# ── Empty inputs ──────────────────────────────────────────────────────────


def test_empty_inputs():
    coverage = _coverage([[40, 40]], [])
    assert coverage.image_count == 1
    assert int(coverage.grid(0).sum()) == 0
    assert coverage.image_covered_fraction(0) == 0.0

    # Zero-size batch queries come back empty, with their dtypes intact.
    empty_idx = np.zeros(0, dtype=np.uint32)
    empty_xy = np.zeros((0, 2), dtype=np.float64)
    empty_r = np.zeros(0, dtype=np.float32)
    assert coverage.counts_at(empty_idx, empty_xy).shape == (0,)
    assert coverage.covered_fraction(empty_idx, empty_xy, empty_r).dtype == np.float32
    assert coverage.uncovered_sectors(empty_idx, empty_xy, empty_r, 4).shape == (0,)

    # No images at all.
    coverage = _coverage(np.zeros((0, 2), dtype=np.uint32), [])
    assert coverage.image_count == 0


# ── Input validation ──────────────────────────────────────────────────────


def test_zero_cell_size_raises():
    with pytest.raises(ValueError, match="cell_px must be positive"):
        _coverage([[40, 40]], [(0, 10.0, 10.0, 5.0)], cell_px=0)


@pytest.mark.parametrize(
    "n_obs_indexes, n_obs_radii, message",
    [(1, 2, "track_image_indexes"), (2, 1, "radii_px")],
)
def test_mismatched_build_lengths_raise(n_obs_indexes, n_obs_radii, message):
    with pytest.raises(ValueError, match=message):
        ObservationCoverage(
            np.array([[40, 40]], dtype=np.uint32),
            np.zeros(n_obs_indexes, dtype=np.uint32),
            np.zeros((2, 2), dtype=np.float64),
            np.ones(n_obs_radii, dtype=np.float32),
        )


def test_out_of_range_build_image_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        _coverage([[40, 40]], [(3, 10.0, 10.0, 5.0)])


def test_bad_shapes_raise():
    with pytest.raises(ValueError, match="image_sizes must have shape"):
        ObservationCoverage(
            np.zeros((1, 3), dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="keypoints_xy must have shape"):
        ObservationCoverage(
            np.array([[40, 40]], dtype=np.uint32),
            np.zeros(1, dtype=np.uint32),
            np.zeros((1, 3), dtype=np.float64),
            np.zeros(1, dtype=np.float32),
        )

    coverage = _coverage([[40, 40]], [])
    with pytest.raises(ValueError, match="xy must have shape"):
        coverage.counts_at(np.zeros(1, dtype=np.uint32), np.zeros((1, 3)))


def test_out_of_range_query_image_index_raises():
    coverage = _coverage([[40, 40]], [])
    for call in (
        lambda: coverage.counts_at(*_query([(1.0, 1.0)], images=[4])),
        lambda: coverage.covered_fraction(*_query([(1.0, 1.0)], [1.0], images=[4])),
        lambda: coverage.uncovered_sectors(*_query([(1.0, 1.0)], [1.0], images=[4]), 4),
    ):
        with pytest.raises(ValueError, match="out of range"):
            call()

    with pytest.raises(ValueError, match="out of range"):
        coverage.grid(1)
    with pytest.raises(ValueError, match="out of range"):
        coverage.image_covered_fraction(1)


def test_mismatched_query_lengths_raise():
    coverage = _coverage([[40, 40]], [])
    xy = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="image_indexes has 1 entries"):
        coverage.counts_at(np.zeros(1, dtype=np.uint32), xy)
    with pytest.raises(ValueError, match="radius_px has 1 entries"):
        coverage.covered_fraction(
            np.zeros(2, dtype=np.uint32), xy, np.ones(1, dtype=np.float32)
        )


@pytest.mark.parametrize("n_sectors", [0, 33])
def test_unrepresentable_sector_count_raises(n_sectors):
    coverage = _coverage([[40, 40]], [])
    with pytest.raises(ValueError, match="n_sectors"):
        _sectors(coverage, [(10.0, 10.0)], n_sectors=n_sectors)


def test_widest_representable_fan_is_accepted():
    coverage = _sector_scene([])
    masks = _sectors(coverage, [SECTOR_QUERY], n_sectors=32)
    assert masks[0] != 0
