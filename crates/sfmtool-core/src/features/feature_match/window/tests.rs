// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::*;

#[test]
fn mutual_matches_are_sorted_by_image1_index() {
    // 200 features per image, every one matched to its mirror in both
    // directions. The sorted-to-original maps reverse the order, so iterating
    // `forward` in any order other than the sorted one would show up below.
    let n = 200;
    let forward: HashMap<usize, (usize, f64)> =
        (0..n).map(|s| (s, ((s + 7) % n, s as f64))).collect();
    let backward: HashMap<usize, (usize, f64)> = (0..n).map(|s| ((s + 7) % n, (s, 0.0))).collect();

    let mutual = mutual_matches(&forward, &backward, |s| n - 1 - s, |s| s);

    let expected: Vec<(usize, usize, f64)> = (0..n)
        .map(|idx1| {
            let s1 = n - 1 - idx1;
            (idx1, (s1 + 7) % n, s1 as f64)
        })
        .collect();
    assert_eq!(mutual, expected);
}

#[test]
fn mutual_matches_drops_pairs_the_backward_match_disagrees_with() {
    let forward: HashMap<usize, (usize, f64)> = [(0, (1, 2.0)), (1, (0, 3.0)), (2, (2, 4.0))]
        .into_iter()
        .collect();
    // Image-2 feature 1 points back at image-1 feature 2, not 0.
    let backward: HashMap<usize, (usize, f64)> = [(1, (2, 0.0)), (0, (1, 0.0)), (2, (2, 0.0))]
        .into_iter()
        .collect();

    let mutual = mutual_matches(&forward, &backward, |s| s, |s| s);

    assert_eq!(mutual, vec![(1, 0, 3.0), (2, 2, 4.0)]);
}
