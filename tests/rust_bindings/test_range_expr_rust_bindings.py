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
