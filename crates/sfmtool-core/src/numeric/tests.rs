// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Median tests, weighted toward the cases the six merged copies disagreed on.
//!
//! The clean-data cases are the cheap half — all six already agreed there.
//! What needs pinning is the part that was previously unspecified: what a NaN
//! in the population does, and what an empty one returns. Those are the
//! contract now, so they get named tests rather than being left to whichever
//! comparator a given copy happened to use.

use super::{median, median_in_place, SELECT_MIN_LEN};

/// Straightforward full-sort median, used to hold the optimized implementation
/// to account across the quickselect threshold.
fn reference_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let n = sorted.len();
    0.5 * (sorted[(n - 1) / 2] + sorted[n / 2])
}

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

/// `median_in_place` permutes rather than consumes: the caller's buffer still
/// holds the same multiset afterwards. Deliberately *not* asserting that the
/// slice comes back sorted — it does on the short path and does not on the
/// quickselect one, and no caller may depend on either.
#[test]
fn median_in_place_permutes_without_losing_values() {
    for len in [3_usize, SELECT_MIN_LEN, SELECT_MIN_LEN + 1] {
        let original: Vec<f64> = (0..len).map(|i| ((i * 37) % len) as f64).collect();
        let mut values = original.clone();
        median_in_place(&mut values);

        let mut got = values;
        let mut want = original;
        got.sort_unstable_by(f64::total_cmp);
        want.sort_unstable_by(f64::total_cmp);
        assert_eq!(got, want, "len {len}");
    }
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

/// The two internal paths — full sort below [`SELECT_MIN_LEN`], quickselect at
/// or above it — must agree with a plain sort at every length, on both parities
/// and in both orderings. `total_cmp` makes the k-th order statistic unique, so
/// "agrees" here means bit-identical, not merely close.
#[test]
fn the_quickselect_path_matches_a_full_sort_at_every_length() {
    for len in 1..=(SELECT_MIN_LEN * 2 + 3) {
        // A shuffled-but-deterministic population; the stride is coprime with
        // most lengths, so the input order is not already sorted.
        let ascending: Vec<f64> = (0..len).map(|i| ((i * 37) % len) as f64).collect();
        let descending: Vec<f64> = ascending.iter().rev().copied().collect();

        for population in [&ascending, &descending] {
            let mut buf = population.clone();
            assert_eq!(
                median_in_place(&mut buf),
                reference_median(population),
                "len {len}"
            );
        }
    }
}

/// The NaN rule has to survive the threshold too: the quickselect path orders
/// by the same `total_cmp`, so a NaN minority stays out of the middle there as
/// well. Pinned separately because this path finds the median by partitioning
/// rather than by sorting, and a `f64::max` slip in the even-count branch would
/// skip the NaN instead of ordering it.
#[test]
fn the_quickselect_path_applies_the_same_nan_rule() {
    for len in [SELECT_MIN_LEN, SELECT_MIN_LEN + 1, SELECT_MIN_LEN * 2] {
        for nans in [1_usize, 2, len / 4] {
            let mut population: Vec<f64> = (0..len - nans).map(|i| i as f64).collect();
            population.extend(std::iter::repeat_n(f64::NAN, nans));

            let mut buf = population.clone();
            let got = median_in_place(&mut buf);
            let want = reference_median(&population);
            assert_eq!(got, want, "len {len}, {nans} NaNs");
            assert!(got.is_finite(), "a NaN minority must not reach the middle");
        }
    }
}
