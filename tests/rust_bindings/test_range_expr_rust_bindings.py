# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RangeExpr PyO3 bindings."""

import pytest

from sfmtool._sfmtool.reconstruction import RangeExpr


class TestConstructor:
    def test_simple_range(self):
        r = RangeExpr("1-5")
        assert list(r) == [1, 2, 3, 4, 5]
        assert len(r) == 5

    def test_comma_separated(self):
        r = RangeExpr("1,3,5,7")
        assert list(r) == [1, 3, 5, 7]
        assert len(r) == 4

    def test_mixed_ranges_and_values(self):
        r = RangeExpr("1-3,10,15-17")
        assert list(r) == [1, 2, 3, 10, 15, 16, 17]
        assert len(r) == 7

    def test_stepped_range(self):
        r = RangeExpr("1-10:2")
        assert list(r) == [1, 3, 5, 7, 9]
        assert len(r) == 5

    def test_single_value(self):
        r = RangeExpr("10")
        assert list(r) == [10]
        assert len(r) == 1

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            RangeExpr("not-a-range")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            RangeExpr("")


class TestFromList:
    def test_contiguous(self):
        r = RangeExpr.from_list([1, 2, 3, 4, 5])
        assert list(r) == [1, 2, 3, 4, 5]
        assert len(r) == 5

    def test_non_contiguous(self):
        r = RangeExpr.from_list([1, 5, 9])
        assert list(r) == [1, 5, 9]
        assert len(r) == 3

    def test_single_value(self):
        r = RangeExpr.from_list([42])
        assert list(r) == [42]

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            RangeExpr.from_list([])


class TestContains:
    def test_membership(self):
        r = RangeExpr("1-5,10")
        assert 3 in r
        assert 5 in r
        assert 10 in r
        assert 6 not in r
        assert 0 not in r
        assert 11 not in r


class TestEquality:
    def test_equal(self):
        assert RangeExpr("1-5") == RangeExpr("1-5")

    def test_not_equal(self):
        assert RangeExpr("1-5") != RangeExpr("1-6")

    def test_constructor_equals_from_list(self):
        assert RangeExpr("1-5") == RangeExpr.from_list([1, 2, 3, 4, 5])


class TestHash:
    def test_equal_objects_hash_equal(self):
        assert hash(RangeExpr("1-5")) == hash(RangeExpr.from_list([1, 2, 3, 4, 5]))

    def test_usable_in_set(self):
        items = {RangeExpr("1-5"), RangeExpr("1-5"), RangeExpr("1-6")}
        assert len(items) == 2

    def test_usable_as_dict_key(self):
        d = {RangeExpr("1-5"): "a", RangeExpr("1-6"): "b"}
        assert d[RangeExpr("1-5")] == "a"


class TestStrRepr:
    def test_str_roundtrips_through_constructor(self):
        r = RangeExpr("1-3,10,15-17")
        assert RangeExpr(str(r)) == r

    def test_repr(self):
        r = RangeExpr("1-5")
        assert repr(r) == 'RangeExpr("1-5")'

    def test_repr_roundtrips_through_eval(self):
        r = RangeExpr("1-3,10,15-17")
        assert eval(repr(r), {"RangeExpr": RangeExpr}) == r


class TestIteration:
    def test_iteration_is_repeatable(self):
        r = RangeExpr("1-3")
        assert list(r) == [1, 2, 3]
        assert list(r) == [1, 2, 3]

    def test_set_construction(self):
        r = RangeExpr("1-5,8")
        assert set(r) == {1, 2, 3, 4, 5, 8}


class TestWhitespace:
    def test_surrounding_whitespace_ignored(self):
        assert list(RangeExpr("  1-5\n")) == [1, 2, 3, 4, 5]

    def test_whitespace_around_separators_ignored(self):
        assert list(RangeExpr("1 - 10 : 3")) == [1, 4, 7, 10]
        assert list(RangeExpr("1, 3,\t5")) == [1, 3, 5]


class TestStepAndDirection:
    def test_descending_bounds_need_a_negative_step(self):
        with pytest.raises(ValueError, match="[Dd]escending range"):
            RangeExpr("10-1")

    def test_descending_with_negative_step_normalizes_to_ascending(self):
        assert list(RangeExpr("10-1:-1")) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_ascending_bounds_reject_a_negative_step(self):
        with pytest.raises(ValueError, match="ascending range"):
            RangeExpr("1-10:-2")

    def test_zero_step_rejected(self):
        with pytest.raises(ValueError, match="[Ss]tep must not be zero"):
            RangeExpr("1-10:0")

    def test_step_past_the_end_yields_the_start_alone(self):
        assert list(RangeExpr("1-2:5")) == [1]


class TestOverlapAndOrdering:
    def test_overlapping_ranges_rejected(self):
        with pytest.raises(ValueError, match="overlapping"):
            RangeExpr("1-5,3-8")

    def test_a_repeated_single_value_counts_as_an_overlap(self):
        with pytest.raises(ValueError, match="overlapping"):
            RangeExpr("1,1,1")

    def test_trailing_comma_rejected(self):
        with pytest.raises(ValueError, match="[Tt]railing comma"):
            RangeExpr("1,")

    def test_out_of_order_chunks_are_sorted(self):
        assert list(RangeExpr("5-7,1-3")) == [1, 2, 3, 5, 6, 7]

    def test_adjacent_chunks_merge(self):
        assert str(RangeExpr("1-3,4-6")) == "1-6"


class TestNegativeValues:
    def test_single_negative(self):
        assert list(RangeExpr("-3")) == [-3]

    def test_range_spanning_zero(self):
        assert list(RangeExpr("-2-2")) == [-2, -1, 0, 1, 2]

    def test_wholly_negative_range(self):
        assert list(RangeExpr("-10--8")) == [-10, -9, -8]


class TestValueBounds:
    """openjd-expr bounds every endpoint and step to |v| < 2**62.

    The bound narrows the accepted syntax below the full i64 domain, so it is
    pinned here: 2**62 is nine orders of magnitude past any real file number or
    point index, but it is the edge at which a range expression stops parsing.
    """

    MAX_MAGNITUDE = 1 << 62

    def test_largest_accepted_magnitude(self):
        largest = self.MAX_MAGNITUDE - 1
        assert list(RangeExpr(str(largest))) == [largest]
        assert list(RangeExpr(str(-largest))) == [-largest]

    def test_endpoint_at_the_bound_rejected(self):
        with pytest.raises(ValueError, match="[Oo]verflow"):
            RangeExpr(str(self.MAX_MAGNITUDE))

    def test_negative_endpoint_at_the_bound_rejected(self):
        with pytest.raises(ValueError, match="[Oo]verflow"):
            RangeExpr(str(-self.MAX_MAGNITUDE))

    def test_step_at_the_bound_rejected(self):
        with pytest.raises(ValueError, match="[Oo]verflow"):
            RangeExpr(f"0-10:{self.MAX_MAGNITUDE}")

    def test_from_list_value_at_the_bound_rejected(self):
        with pytest.raises(ValueError, match="[Oo]verflow"):
            RangeExpr.from_list([self.MAX_MAGNITUDE])

    def test_python_int_beyond_i64_rejected_by_conversion(self):
        # Caught converting the argument, before openjd-expr sees it.
        with pytest.raises(OverflowError):
            RangeExpr.from_list([1 << 63])


class TestDisplayNormalization:
    def test_stepped_end_is_normalized_to_the_last_member(self):
        assert str(RangeExpr("1-10:2")) == "1-9:2"

    def test_two_member_range_renders_as_a_list(self):
        assert str(RangeExpr("1-2")) == "1,2"

    def test_single_member_range_renders_as_a_value(self):
        assert str(RangeExpr("5-5")) == "5"

    def test_from_list_sorts_and_deduplicates(self):
        assert list(RangeExpr.from_list([5, 1, 3, 1])) == [1, 3, 5]

    def test_from_list_collapses_an_arithmetic_run_into_a_step(self):
        # from_list runs step detection over the sorted values, so it renders
        # tighter than the parser does for the same members: "1,3,5" parses as
        # three single-value chunks and stays that way.
        assert str(RangeExpr.from_list([5, 1, 3])) == "1-5:2"
        assert str(RangeExpr("1,3,5")) == "1,3,5"
        assert RangeExpr.from_list([5, 1, 3]) != RangeExpr("1,3,5")
