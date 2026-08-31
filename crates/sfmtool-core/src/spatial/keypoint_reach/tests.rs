// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the per-image keypoint reach enumeration.

use super::*;

/// A deterministic uniform stream, so a case names its own rows.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Random rows over a handful of images, with a few non-finite reaches.
fn random_rows(n: usize, n_images: i64, seed: u64) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut image = Vec::with_capacity(n);
    let mut xy = Vec::with_capacity(2 * n);
    let mut reach = Vec::with_capacity(n);
    for k in 0..n {
        image.push((rng.next_f64() * n_images as f64) as i64);
        xy.push(rng.next_f64() * 40.0);
        xy.push(rng.next_f64() * 40.0);
        let r = 0.5 + rng.next_f64() * 6.0;
        reach.push(if k % 17 == 5 { f64::NAN } else { r });
    }
    (image, xy, reach)
}

/// The same relation by a double loop, in the same order.
fn brute_force(image: &[i64], xy: &[f64], reach: &[f64]) -> ReachPairs {
    let n = image.len();
    let mut images: Vec<i64> = image.to_vec();
    images.sort_unstable();
    images.dedup();
    let mut out = ReachPairs::default();
    for img in images {
        // Rows in their given order; within a row, candidates in column order
        // with ties in row order.
        for i in (0..n).filter(|&i| image[i] == img) {
            if !reach[i].is_finite() {
                continue;
            }
            let mut run: Vec<usize> = (0..n).filter(|&j| image[j] == img).collect();
            run.sort_by(|&a, &b| cmp_nan_last(xy[2 * a], xy[2 * b]));
            for j in run {
                let dx = xy[2 * j] - xy[2 * i];
                let dy = xy[2 * j + 1] - xy[2 * i + 1];
                let d = (dx * dx + dy * dy).sqrt();
                if d <= reach[i] && j != i {
                    out.row.push(i as i64);
                    out.candidate.push(j as i64);
                    out.distance_px.push(d);
                }
            }
        }
    }
    out
}

fn rows<'a>(image: &'a [i64], xy: &'a [f64], reach: &'a [f64]) -> KeypointRows<'a> {
    KeypointRows {
        image_of_row: image,
        xy_px: xy,
        reach_px: reach,
    }
}

// ── Exactness ──────────────────────────────────────────────────────────────

#[test]
fn matches_a_brute_force_double_loop_pair_for_pair() {
    for seed in [1u64, 7, 99, 2026] {
        let (image, xy, reach) = random_rows(400, 5, seed);
        let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
        let want = brute_force(&image, &xy, &reach);
        assert_eq!(got, want, "seed {seed}");
    }
}

#[test]
fn one_image_is_the_same_relation_as_many() {
    let (_image, xy, reach) = random_rows(300, 1, 5);
    let image = vec![0i64; reach.len()];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert_eq!(got, brute_force(&image, &xy, &reach));
}

#[test]
fn no_rows_is_no_pairs() {
    let got = pairs_within_reach(rows(&[], &[], &[])).unwrap();
    assert!(got.is_empty());
    assert_eq!(got.len(), 0);
}

// ── Directedness ───────────────────────────────────────────────────────────

#[test]
fn only_the_reach_that_spans_the_separation_pairs() {
    // Two rows 3 px apart: the first reaches across, the second does not.
    let image = [0i64, 0];
    let xy = [0.0, 0.0, 3.0, 0.0];
    let reach = [5.0, 1.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    let directed: Vec<(i64, i64)> = got
        .row
        .iter()
        .zip(&got.candidate)
        .map(|(&i, &j)| (i, j))
        .collect();
    assert_eq!(directed, vec![(0, 1)]);
}

// ── Self pair and NaN ──────────────────────────────────────────────────────

#[test]
fn no_row_is_its_own_candidate() {
    let (image, xy, reach) = random_rows(200, 3, 31);
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert!(!got.is_empty());
    assert!(got.row.iter().zip(&got.candidate).all(|(&i, &j)| i != j));
}

#[test]
fn a_nan_reach_asks_nothing_and_still_answers() {
    let image = [0i64, 0];
    let xy = [0.0, 0.0, 1.0, 0.0];
    let reach = [f64::NAN, 4.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    let pairs: Vec<(i64, i64)> = got
        .row
        .iter()
        .zip(&got.candidate)
        .map(|(&i, &j)| (i, j))
        .collect();
    assert_eq!(pairs, vec![(1, 0)]);
}

#[test]
fn an_infinite_reach_asks_nothing_too() {
    let image = [0i64, 0];
    let xy = [0.0, 0.0, 1.0, 0.0];
    let reach = [f64::INFINITY, 4.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert!(got.row.iter().all(|&i| i == 1));
}

#[test]
fn a_zero_reach_holds_only_a_coincident_centre() {
    let image = [0i64, 0, 0];
    let xy = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0];
    let reach = [0.0, 0.0, 0.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert_eq!(got.row, vec![0, 1]);
    assert_eq!(got.candidate, vec![1, 0]);
    assert_eq!(got.distance_px, vec![0.0, 0.0]);
}

// ── Image isolation ────────────────────────────────────────────────────────

#[test]
fn identical_positions_in_different_images_never_pair() {
    let image = [0i64, 1, 2];
    let xy = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
    let reach = [100.0, 100.0, 100.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert!(got.is_empty());
}

#[test]
fn images_come_out_in_ascending_index_whatever_order_the_rows_arrive_in() {
    let image = [7i64, 2, 7, 2];
    let xy = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let reach = [3.0, 3.0, 3.0, 3.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    // Image 2 first (rows 1 and 3), then image 7 (rows 0 and 2).
    assert_eq!(got.row, vec![1, 3, 0, 2]);
    assert_eq!(got.candidate, vec![3, 1, 2, 0]);
}

#[test]
fn a_negative_image_index_is_an_image_like_any_other() {
    let image = [-1i64, -1, 0];
    let xy = [0.0, 0.0, 1.0, 0.0, 0.5, 0.0];
    let reach = [3.0, 3.0, 3.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert_eq!(got.row, vec![0, 1]);
    assert_eq!(got.candidate, vec![1, 0]);
}

// ── Order and batching ─────────────────────────────────────────────────────

#[test]
fn the_pair_stream_is_the_same_at_every_batch_size_and_without_parallelism() {
    let (image, xy, reach) = random_rows(500, 4, 4242);
    let reference = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert!(!reference.is_empty());
    for grain in [1usize, 2, 7, 64, 499, 500, 100_000] {
        for parallel in [true, false] {
            let got = pairs_within_reach_batch(rows(&image, &xy, &reach), grain, parallel).unwrap();
            assert_eq!(got, reference, "grain {grain}, parallel {parallel}");
        }
    }
    // A grain of zero is a grain of one, not a division by zero.
    assert_eq!(
        pairs_within_reach_batch(rows(&image, &xy, &reach), 0, true).unwrap(),
        reference
    );
}

#[test]
fn ties_in_column_are_broken_by_row_order() {
    let image = [0i64; 4];
    let xy = [1.0, 0.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0];
    let reach = [10.0, 10.0, 10.0, 10.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    // Every column is 1.0, so row 0's run is the rows in their given
    // order, less itself.
    assert_eq!(got.candidate[..3], [1, 2, 3]);
}

// ── Refusals ───────────────────────────────────────────────────────────────

#[test]
fn mismatched_lengths_are_refused() {
    let image = [0i64, 0];
    let xy = [0.0, 0.0, 1.0, 0.0];
    assert_eq!(
        pairs_within_reach(rows(&image, &xy, &[1.0])).unwrap_err(),
        KeypointReachError::LengthMismatch {
            images: 2,
            positions: 2,
            reaches: 1,
        }
    );
    assert_eq!(
        pairs_within_reach(rows(&image, &xy[..2], &[1.0, 1.0])).unwrap_err(),
        KeypointReachError::LengthMismatch {
            images: 2,
            positions: 1,
            reaches: 2,
        }
    );
}

#[test]
fn a_negative_reach_is_refused_and_names_its_row() {
    let image = [0i64, 0];
    let xy = [0.0, 0.0, 1.0, 0.0];
    let err = pairs_within_reach(rows(&image, &xy, &[1.0, -0.5])).unwrap_err();
    assert_eq!(
        err,
        KeypointReachError::NegativeReach {
            row: 1,
            reach: -0.5
        }
    );
    assert!(err.to_string().contains("row 1"));
}

#[test]
fn the_refusals_read_as_sentences() {
    let err = KeypointReachError::LengthMismatch {
        images: 3,
        positions: 2,
        reaches: 3,
    };
    assert!(err.to_string().contains("xy_px 2"));
}

// ── The column order itself ────────────────────────────────────────────────

#[test]
fn a_nan_column_sorts_last_and_is_never_found() {
    let image = [0i64, 0, 0];
    let xy = [f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0];
    let reach = [2.0, 2.0, 2.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    // Row 0 asks at a NaN column, so its own search window is empty; rows 1
    // and 2 hold each other, and neither holds row 0.
    assert_eq!(got.row, vec![1, 2]);
    assert_eq!(got.candidate, vec![2, 1]);
    assert_eq!(cmp_nan_last(f64::NAN, 1.0), Ordering::Greater);
    assert_eq!(cmp_nan_last(1.0, f64::NAN), Ordering::Less);
    assert_eq!(cmp_nan_last(f64::NAN, f64::NAN), Ordering::Equal);
}

#[test]
fn the_run_bounds_are_a_superset_the_distance_test_trims() {
    // A row exactly one reach away in column and one in row: inside the run,
    // outside the disk.
    let image = [0i64, 0];
    let xy = [0.0, 0.0, 2.0, 2.0];
    let reach = [2.0, 2.0];
    let got = pairs_within_reach(rows(&image, &xy, &reach)).unwrap();
    assert!(got.is_empty());
}
