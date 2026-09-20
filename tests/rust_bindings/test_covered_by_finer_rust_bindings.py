# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``covered_by_finer`` binding: retiring a coarse observation.

The rule is a statement about pairs, so it is held to a brute-force NumPy
transcription of the same statement -- an O(n^2) double loop per image, which
is the shape the seed pipeline's own stage is written in -- over seeded random
rows. What has to agree is every bit of the verdict: the flags, the surviving
rows, the surviving owners and every count of the census.

The generated population is built to sit on the rule's edges rather than away
from them: radii are exact powers of two, so a great many pairs are exactly one
octave apart and land on the non-strict ``>=``, and a slice of the rows share
pixel positions exactly, so containment is decided at distance zero.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.analysis import covered_by_finer


def reference(
    image,
    owner,
    xy,
    reach,
    radius,
    n_owners,
    *,
    ratio=2.0,
    min_fine_radius_px=0.0,
    min_observations=2,
    protected=None,
):
    """The rule as the seed pipeline states it, brute force, per image.

    A row is retired where another row in the same image, on another owner, has
    its centre inside the row's own footprint and a radius at least ``ratio``
    times smaller. A protected row is never retired and still covers. An owner
    left under ``min_observations`` is dropped and its survivors with it.
    """
    n = len(image)
    flagged = np.zeros(n, bool)
    spared = np.zeros(n, bool)
    pairs_contained = 0
    pairs_finer = 0
    for img in np.unique(image):
        sel = np.nonzero(image == img)[0]
        if len(sel) < 2:
            continue
        p = xy[sel]
        dx = p[None, :, 0] - p[:, None, 0]
        dy = p[None, :, 1] - p[:, None, 1]
        # Written out rather than through `np.linalg.norm`, which scales the
        # operands before squaring them and so rounds differently.
        d = np.sqrt(dx * dx + dy * dy)
        r = radius[sel]
        rc = reach[sel]
        c = owner[sel]
        inside = (
            np.isfinite(rc)[:, None]
            & (d <= rc[:, None])
            & (r[None, :] < r[:, None])
            & (c[None, :] != c[:, None])
        )
        np.fill_diagonal(inside, False)
        finer = (
            inside
            & (r[:, None] >= ratio * r[None, :])
            & (r[None, :] >= min_fine_radius_px)
        )
        pairs_contained += int(inside.sum())
        pairs_finer += int(finer.sum())
        hit = sel[finer.any(axis=1)]
        if protected is None:
            flagged[hit] = True
        else:
            flagged[hit[~protected[hit]]] = True
            spared[hit[protected[hit]]] = True

    rows_of = np.bincount(owner, minlength=n_owners)
    flagged_of = np.bincount(owner[flagged], minlength=n_owners)
    keep_owner = (rows_of - flagged_of) >= min_observations
    keep_row = ~flagged & keep_owner[owner]
    present = rows_of > 0
    dropped = present & ~keep_owner
    return {
        "flagged": flagged,
        "keep_row": keep_row,
        "keep_owner": keep_owner,
        "census": {
            "rows": n,
            "pairs_contained": pairs_contained,
            "pairs_finer": pairs_finer,
            "rows_flagged": int(flagged.sum()),
            "rows_spared": int(spared.sum()),
            "rows_removed": int((~keep_row).sum()),
            "owners_dropped_all_covered": int(
                (dropped & (flagged_of == rows_of)).sum()
            ),
            "owners_dropped_by_sweep": int((dropped & (flagged_of < rows_of)).sum()),
            "owners_kept": int(keep_owner.sum()),
        },
    }


def agree(got, want):
    """Every bit of the two verdicts, so a partial match is still a failure."""
    np.testing.assert_array_equal(got["flagged"], want["flagged"])
    np.testing.assert_array_equal(got["keep_row"], want["keep_row"])
    np.testing.assert_array_equal(got["keep_owner"], want["keep_owner"])
    assert got["census"] == want["census"]


def population(seed, n=3000, images=8, owners=400, coincident=300):
    """Seeded rows built to sit on the rule's edges.

    Radii are exact powers of two so that octave ties are common, and the last
    ``coincident`` rows are placed at pixels the earlier rows already occupy.
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(0, images, n).astype(np.int64)
    owner = rng.integers(0, owners, n).astype(np.int64)
    xy = np.empty((n, 2), dtype=np.float64)
    xy[: n - coincident] = rng.uniform(0.0, 320.0, (n - coincident, 2))
    # The tail sits exactly on top of rows already placed, so containment is
    # decided at distance zero on both sides.
    xy[n - coincident :] = xy[rng.integers(0, n - coincident, coincident)]
    radius = 2.0 ** rng.integers(-2, 5, n).astype(np.float64)
    reach = 0.5 * radius
    return image, owner, np.ascontiguousarray(xy), reach, radius, owners


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_binding_and_the_numpy_rule_agree_on_random_rows(seed):
    image, owner, xy, reach, radius, owners = population(seed)
    got = covered_by_finer(image, owner, xy, reach, radius, owners)
    want = reference(image, owner, xy, reach, radius, owners)
    agree(got, want)
    # The population has to exercise the rule, or agreement says nothing.
    assert want["census"]["pairs_finer"] > 100
    assert want["census"]["rows_flagged"] > 50
    assert 0 < want["census"]["owners_kept"] < owners


def test_a_tie_at_exactly_one_octave_is_finer_on_both_sides():
    """The comparison is non-strict, and the binding and the rule say so."""
    image = np.zeros(2, np.int64)
    owner = np.array([0, 1], np.int64)
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    radius = np.array([4.0, 2.0])
    reach = np.array([10.0, 10.0])
    options = dict(min_observations=1)
    got = covered_by_finer(image, owner, xy, reach, radius, 2, **options)
    agree(got, reference(image, owner, xy, reach, radius, 2, **options))
    np.testing.assert_array_equal(got["flagged"], [True, False])

    # A hair under one octave, and nothing is retired by either.
    radius = np.array([4.0, 2.000_000_1])
    got = covered_by_finer(image, owner, xy, reach, radius, 2, **options)
    agree(got, reference(image, owner, xy, reach, radius, 2, **options))
    assert not got["flagged"].any()


def test_coincident_pixels_are_contained_at_distance_zero():
    image = np.zeros(3, np.int64)
    owner = np.array([0, 1, 2], np.int64)
    xy = np.zeros((3, 2))
    radius = np.array([8.0, 2.0, 8.0])
    reach = np.array([4.0, 1.0, 4.0])
    options = dict(min_observations=1)
    got = covered_by_finer(image, owner, xy, reach, radius, 3, **options)
    agree(got, reference(image, owner, xy, reach, radius, 3, **options))
    np.testing.assert_array_equal(got["flagged"], [True, False, True])


@pytest.mark.parametrize("seed", [0, 1])
def test_the_fine_radius_floor_agrees(seed):
    image, owner, xy, reach, radius, owners = population(seed)
    for floor in (0.0, 1.0, 4.0):
        options = dict(min_fine_radius_px=floor)
        got = covered_by_finer(image, owner, xy, reach, radius, owners, **options)
        agree(got, reference(image, owner, xy, reach, radius, owners, **options))
    # The floor really bites on this population: a quarter of the radii are
    # under a pixel.
    assert (radius < 1.0).mean() > 0.1
    off = covered_by_finer(image, owner, xy, reach, radius, owners)
    on = covered_by_finer(
        image, owner, xy, reach, radius, owners, min_fine_radius_px=4.0
    )
    assert on["census"]["rows_flagged"] < off["census"]["rows_flagged"]


@pytest.mark.parametrize("seed", [0, 1])
def test_protection_spares_a_row_and_leaves_it_covering(seed):
    image, owner, xy, reach, radius, owners = population(seed)
    rng = np.random.default_rng(seed + 100)
    protected = rng.random(len(image)) < 0.2
    options = dict(protected=protected)
    got = covered_by_finer(image, owner, xy, reach, radius, owners, **options)
    agree(got, reference(image, owner, xy, reach, radius, owners, **options))
    # A protected row is never retired, and the pairs are the same pairs: what
    # protection refuses is the retirement, not the reading.
    assert not got["flagged"][protected].any()
    assert got["census"]["rows_spared"] > 0
    plain = covered_by_finer(image, owner, xy, reach, radius, owners)
    assert got["census"]["pairs_finer"] == plain["census"]["pairs_finer"]
    assert got["census"]["rows_spared"] == 0 or (
        got["census"]["rows_flagged"] < plain["census"]["rows_flagged"]
    )


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 4.0])
def test_the_ratio_agrees(seed, ratio):
    image, owner, xy, reach, radius, owners = population(seed)
    got = covered_by_finer(image, owner, xy, reach, radius, owners, ratio=ratio)
    agree(got, reference(image, owner, xy, reach, radius, owners, ratio=ratio))


@pytest.mark.parametrize("bar", [1, 2, 3])
def test_the_observation_bar_agrees(bar):
    image, owner, xy, reach, radius, owners = population(0)
    options = dict(min_observations=bar)
    got = covered_by_finer(image, owner, xy, reach, radius, owners, **options)
    agree(got, reference(image, owner, xy, reach, radius, owners, **options))


def test_a_row_of_unstated_reach_asks_nothing_and_still_covers():
    image = np.zeros(2, np.int64)
    owner = np.array([0, 1], np.int64)
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    radius = np.array([8.0, 2.0])
    reach = np.array([np.nan, 10.0])
    options = dict(min_observations=1)
    got = covered_by_finer(image, owner, xy, reach, radius, 2, **options)
    agree(got, reference(image, owner, xy, reach, radius, 2, **options))
    # Row 0 asked nothing, so it is not retired; row 1 asked and found only a
    # coarser neighbour, so it is not retired either.
    assert not got["flagged"].any()
    assert got["census"]["pairs_contained"] == 0


def test_rows_of_different_images_never_pair():
    image = np.array([0, 1], np.int64)
    owner = np.array([0, 1], np.int64)
    xy = np.zeros((2, 2))
    radius = np.array([8.0, 2.0])
    reach = np.array([10.0, 10.0])
    got = covered_by_finer(image, owner, xy, reach, radius, 2, min_observations=1)
    assert got["census"]["pairs_contained"] == 0
    assert not got["flagged"].any()


def test_no_rows_is_an_answer():
    empty_i = np.zeros(0, np.int64)
    got = covered_by_finer(
        empty_i, empty_i, np.zeros((0, 2)), np.zeros(0), np.zeros(0), 0
    )
    assert got["census"]["rows"] == 0
    assert got["census"]["owners_kept"] == 0
    assert len(got["flagged"]) == 0


class TestRefusals:
    def setup_method(self):
        self.image = np.zeros(2, np.int64)
        self.owner = np.array([0, 1], np.int64)
        self.xy = np.zeros((2, 2))
        self.reach = np.ones(2)
        self.radius = np.ones(2)

    def test_a_short_column_is_refused_by_name(self):
        with pytest.raises(ValueError, match="radius_px 1"):
            covered_by_finer(
                self.image, self.owner, self.xy, self.reach, self.radius[:1], 2
            )

    def test_a_position_array_of_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match=r"\(2, 2\)"):
            covered_by_finer(
                self.image, self.owner, np.zeros((2, 3)), self.reach, self.radius, 2
            )

    def test_a_short_protection_mask_is_refused(self):
        with pytest.raises(ValueError, match="protected 1"):
            covered_by_finer(
                self.image,
                self.owner,
                self.xy,
                self.reach,
                self.radius,
                2,
                protected=np.ones(1, bool),
            )

    def test_an_owner_outside_the_declared_space_is_refused_by_row(self):
        with pytest.raises(ValueError, match="row 1 names owner 1"):
            covered_by_finer(
                self.image, self.owner, self.xy, self.reach, self.radius, 1
            )

    def test_a_ratio_below_one_is_refused(self):
        with pytest.raises(ValueError, match="scale ratio"):
            covered_by_finer(
                self.image, self.owner, self.xy, self.reach, self.radius, 2, ratio=0.5
            )

    def test_a_floor_that_is_not_a_number_is_refused(self):
        with pytest.raises(ValueError, match="fine-radius floor"):
            covered_by_finer(
                self.image,
                self.owner,
                self.xy,
                self.reach,
                self.radius,
                2,
                min_fine_radius_px=np.nan,
            )

    def test_a_negative_reach_is_refused_by_row(self):
        with pytest.raises(ValueError, match="row 1 states a negative reach"):
            covered_by_finer(
                self.image,
                self.owner,
                self.xy,
                np.array([1.0, -1.0]),
                self.radius,
                2,
            )


def test_arrays_are_accepted_in_either_memory_order():
    image, owner, xy, reach, radius, owners = population(0, n=400, owners=60)
    fortran = np.asfortranarray(xy)
    got = covered_by_finer(image, owner, fortran, reach, radius, owners)
    agree(got, reference(image, owner, xy, reach, radius, owners))
