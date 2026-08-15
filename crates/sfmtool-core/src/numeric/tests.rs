// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Median tests, weighted toward the cases the six merged copies disagreed on.
//!
//! The clean-data cases are the cheap half — all six already agreed there.
//! What needs pinning is the part that was previously unspecified: what a NaN
//! in the population does, and what an empty one returns. Those are the
//! contract now, so they get named tests rather than being left to whichever
//! comparator a given copy happened to use.

use super::{median, median_in_place};

#[test]
fn an_odd_count_takes_the_middle_value() {
    assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    assert_eq!(median(&[5.0]), 5.0);
}

#[test]
fn an_even_count_averages_the_two_middle_values() {
    assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    assert_eq!(median(&[10.0, 20.0]), 15.0);
}

/// The numpy convention the six copies already shared, spelled out on a case
/// where the mean and the median differ, so a "sum / len" regression fails.
#[test]
fn the_even_count_average_is_not_the_population_mean() {
    assert_eq!(median(&[0.0, 1.0, 2.0, 100.0]), 1.5);
}

#[test]
fn median_in_place_leaves_the_slice_sorted() {
    let mut values = [3.0, 1.0, 2.0];
    assert_eq!(median_in_place(&mut values), 2.0);
    assert_eq!(values, [1.0, 2.0, 3.0]);
}

#[test]
fn the_two_entry_points_agree() {
    let values = [7.0, 2.0, 9.0, 4.0];
    let mut scratch = values;
    assert_eq!(median(&values), median_in_place(&mut scratch));
}

/// Empty is `NaN`, not `0.0` and not a panic. Three of the merged copies
/// already did this; one returned `0.0`, one returned `None`, and one only
/// `debug_assert!`ed the non-empty precondition and so indexed out of bounds
/// in a release build.
#[test]
fn an_empty_population_is_nan() {
    assert!(median(&[]).is_nan());
    assert!(median_in_place(&mut []).is_nan());
}

/// A NaN minority sorts above the finite values and leaves the median finite —
/// the robustness the callers pick a median for. Previously this was a panic
/// in one copy and an unspecified permutation in two others.
#[test]
fn a_nan_minority_leaves_the_median_finite() {
    assert_eq!(median(&[1.0, 2.0, 3.0, f64::NAN]), 2.5);
    assert_eq!(median(&[1.0, 2.0, f64::NAN]), 2.0);
}

/// A NaN majority reaches the middle and propagates, which is the signal every
/// caller already tests for with `is_finite()` / `is_nan()`.
#[test]
fn a_nan_majority_propagates() {
    assert!(median(&[1.0, f64::NAN, f64::NAN]).is_nan());
    assert!(median(&[f64::NAN]).is_nan());
}

/// `total_cmp` orders `−NaN` below every finite value, unlike `+NaN`. Pinned
/// so the asymmetry is a documented property rather than a surprise.
#[test]
fn a_negative_nan_sorts_below_the_finite_values() {
    let values = [1.0, 2.0, 3.0, -f64::NAN];
    assert_eq!(median(&values), 1.5);
}

/// Infinities are ordinary values to `total_cmp` and must not be confused with
/// the empty/NaN signal; the census and adjacency callers pass populations
/// that can contain one.
#[test]
fn infinities_order_as_ordinary_extremes() {
    assert_eq!(median(&[f64::NEG_INFINITY, 5.0, f64::INFINITY]), 5.0);
    assert_eq!(median(&[1.0, f64::INFINITY]), f64::INFINITY);
}

/// Sorting must not be order-dependent: every permutation of one population
/// yields the same median. This is what the invalid `unwrap_or(Equal)`
/// comparator could not guarantee once a NaN was present.
#[test]
fn the_result_is_independent_of_input_order() {
    let base = [f64::NAN, 4.0, 1.0, 3.0, 2.0];
    let expected = median(&base);
    assert_eq!(expected, 3.0);

    let permutations = [
        [1.0, 2.0, 3.0, 4.0, f64::NAN],
        [4.0, 3.0, 2.0, 1.0, f64::NAN],
        [2.0, f64::NAN, 1.0, 4.0, 3.0],
        [3.0, 1.0, f64::NAN, 2.0, 4.0],
    ];
    for permutation in permutations {
        assert_eq!(median(&permutation), expected);
    }
}

/// Signed zeros are distinct bit patterns to `total_cmp` but must still
/// compare equal numerically in the result.
#[test]
fn signed_zeros_average_to_zero() {
    assert_eq!(median(&[-0.0, 0.0]), 0.0);
}
