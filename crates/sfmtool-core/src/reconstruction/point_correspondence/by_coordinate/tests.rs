// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::*;

/// One reconstruction's observations, built image by image.
#[derive(Default)]
struct Obs {
    images: Vec<u32>,
    xy: Vec<[f64; 2]>,
    points: Vec<u32>,
}

impl Obs {
    fn add(mut self, image: u32, xy: [f64; 2], point: u32) -> Self {
        self.images.push(image);
        self.xy.push(xy);
        self.points.push(point);
        self
    }
}

fn options(pixel_threshold: f64, min_votes: usize) -> CoordinateCorrespondenceOptions {
    CoordinateCorrespondenceOptions {
        pixel_threshold,
        min_votes,
    }
}

fn run(
    source: &Obs,
    target: &Obs,
    shared: &[(u32, u32)],
    options: CoordinateCorrespondenceOptions,
) -> Vec<(u32, u32)> {
    let shared_source: Vec<u32> = shared.iter().map(|&(s, _)| s).collect();
    let shared_target: Vec<u32> = shared.iter().map(|&(_, t)| t).collect();
    let result = find_point_correspondences_by_coordinate(
        &source.images,
        &source.xy,
        &source.points,
        &target.images,
        &target.xy,
        &target.points,
        &shared_source,
        &shared_target,
        &options,
    );
    result
        .source_ids
        .into_iter()
        .zip(result.target_ids)
        .collect()
}

#[test]
fn coincident_keypoints_match_across_different_orderings() {
    // Point 10<->20 appears in both shared images; 11<->21 only in the first.
    let source = Obs::default()
        .add(0, [10.0, 10.0], 10)
        .add(0, [50.0, 50.0], 11)
        .add(1, [12.0, 12.0], 10);
    let target = Obs::default()
        .add(0, [50.3, 49.8], 21)
        .add(0, [10.2, 9.9], 20)
        .add(1, [11.8, 12.1], 20);

    let pairs = run(&source, &target, &[(0, 0), (1, 1)], options(2.0, 2));
    assert_eq!(pairs, vec![(10, 20)]);

    let pairs = run(&source, &target, &[(0, 0), (1, 1)], options(2.0, 1));
    assert_eq!(pairs, vec![(10, 20), (11, 21)]);
}

#[test]
fn the_defaults_are_two_pixels_and_two_votes() {
    assert_eq!(CoordinateCorrespondenceOptions::default(), options(2.0, 2));
}

#[test]
fn keypoints_further_apart_than_the_threshold_do_not_match() {
    let source = Obs::default().add(0, [10.0, 10.0], 10);
    let target = Obs::default().add(0, [20.0, 20.0], 20);
    assert!(run(&source, &target, &[(0, 0)], options(2.0, 1)).is_empty());

    // Exactly at the threshold still matches.
    let target = Obs::default().add(0, [12.0, 10.0], 20);
    assert_eq!(
        run(&source, &target, &[(0, 0)], options(2.0, 1)),
        vec![(10, 20)]
    );
}

#[test]
fn only_mutual_nearest_keypoints_match() {
    // Both source keypoints are nearest to the one target keypoint; only the
    // one the target keypoint is nearest to matches.
    let source = Obs::default()
        .add(0, [10.0, 10.0], 10)
        .add(0, [10.8, 10.0], 11);
    let target = Obs::default().add(0, [10.1, 10.0], 20);
    assert_eq!(
        run(&source, &target, &[(0, 0)], options(2.0, 1)),
        vec![(10, 20)]
    );
}

#[test]
fn the_pair_with_more_votes_claims_a_contested_point() {
    // 11<->20 is matched in image 0 and 10<->20 in images 1 and 2; 20 can be
    // claimed once, so the two-vote pair wins.
    let source = Obs::default()
        .add(0, [10.0, 10.0], 11)
        .add(1, [10.0, 10.0], 10)
        .add(2, [10.0, 10.0], 10);
    let target = Obs::default()
        .add(0, [10.0, 10.0], 20)
        .add(1, [10.0, 10.0], 20)
        .add(2, [10.0, 10.0], 20);
    assert_eq!(
        run(&source, &target, &[(0, 0), (1, 1), (2, 2)], options(2.0, 1)),
        vec![(10, 20)]
    );
}

#[test]
fn vote_ties_go_to_the_pair_voted_for_first() {
    // 10<->20 and 11<->20 have one vote each; 10<->20 was cast first (image 0
    // is the first shared pair), whichever order the shared pairs list them in.
    let source = Obs::default()
        .add(0, [10.0, 10.0], 10)
        .add(1, [10.0, 10.0], 11);
    let target = Obs::default()
        .add(0, [10.0, 10.0], 20)
        .add(1, [10.0, 10.0], 20);
    assert_eq!(
        run(&source, &target, &[(0, 0), (1, 1)], options(2.0, 1)),
        vec![(10, 20)]
    );
    assert_eq!(
        run(&source, &target, &[(1, 1), (0, 0)], options(2.0, 1)),
        vec![(11, 20)]
    );
}

#[test]
fn equally_near_keypoints_resolve_to_the_first_listed() {
    // The target keypoint is 1 px from both source keypoints.
    let source = Obs::default()
        .add(0, [11.0, 10.0], 11)
        .add(0, [9.0, 10.0], 10);
    let target = Obs::default().add(0, [10.0, 10.0], 20);
    assert_eq!(
        run(&source, &target, &[(0, 0)], options(2.0, 1)),
        vec![(11, 20)]
    );
}

#[test]
fn images_missing_from_one_side_are_skipped() {
    let source = Obs::default().add(0, [10.0, 10.0], 10);
    let target = Obs::default().add(5, [10.0, 10.0], 20);
    assert!(run(&source, &target, &[(0, 1)], options(2.0, 1)).is_empty());
}

#[test]
fn a_single_supporting_image_falls_short_of_two_or_three_votes() {
    let source = Obs::default().add(0, [10.0, 10.0], 10);
    let target = Obs::default().add(0, [10.0, 10.0], 20);
    for min_votes in [2, 3] {
        assert!(run(&source, &target, &[(0, 0)], options(2.0, min_votes)).is_empty());
    }
}

#[test]
fn a_keypoint_without_a_finite_position_matches_nothing() {
    let source = Obs::default()
        .add(0, [f64::NAN, f64::NAN], 10)
        .add(0, [30.0, 30.0], 11);
    let target = Obs::default()
        .add(0, [10.0, 10.0], 20)
        .add(0, [30.0, 30.0], 21);
    assert_eq!(
        run(&source, &target, &[(0, 0)], options(2.0, 1)),
        vec![(11, 21)]
    );
}

/// Every pair of every image compared, the way `sfm compare --by-coordinate`
/// did it in numpy before it called this function: the reference the indexed
/// search has to agree with.
fn all_pairs_reference(
    source: &Obs,
    target: &Obs,
    shared: &[(u32, u32)],
    options: CoordinateCorrespondenceOptions,
) -> Vec<(u32, u32)> {
    let rows = |o: &Obs, image: u32| -> Vec<usize> {
        (0..o.images.len())
            .filter(|&r| o.images[r] == image)
            .collect()
    };
    let d2 = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
    let argmin = |p: [f64; 2], xs: &[[f64; 2]]| -> usize {
        let mut best = 0;
        for (k, &x) in xs.iter().enumerate() {
            if d2(p, x) < d2(p, xs[best]) {
                best = k;
            }
        }
        best
    };
    let mut votes: Vec<((u32, u32), usize)> = Vec::new();
    for &(s, t) in shared {
        let (sr, tr) = (rows(source, s), rows(target, t));
        if sr.is_empty() || tr.is_empty() {
            continue;
        }
        let sxy: Vec<[f64; 2]> = sr.iter().map(|&r| source.xy[r]).collect();
        let txy: Vec<[f64; 2]> = tr.iter().map(|&r| target.xy[r]).collect();
        for (i, &p) in sxy.iter().enumerate() {
            let j = argmin(p, &txy);
            if argmin(txy[j], &sxy) == i
                && d2(p, txy[j]) <= options.pixel_threshold * options.pixel_threshold
            {
                let pair = (source.points[sr[i]], target.points[tr[j]]);
                match votes.iter_mut().find(|(q, _)| *q == pair) {
                    Some((_, v)) => *v += 1,
                    None => votes.push((pair, 1)),
                }
            }
        }
    }
    votes.sort_by_key(|&(_, v)| std::cmp::Reverse(v)); // stable: ties keep first-vote order
    let mut taken_s = HashMap::new();
    let mut taken_t = HashMap::new();
    let mut out = Vec::new();
    for ((s, t), v) in votes {
        if v < options.min_votes {
            break;
        }
        if taken_s.contains_key(&s) || taken_t.contains_key(&t) {
            continue;
        }
        taken_s.insert(s, ());
        taken_t.insert(t, ());
        out.push((s, t));
    }
    out
}

#[test]
fn the_indexed_search_agrees_with_comparing_every_pair() {
    // A small deterministic generator, so the fixture is the same every run.
    let mut state: u64 = 0x5eed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64)
    };
    // 300 points seen in 6 images on both sides, the target's keypoints moved
    // by up to 1.5 px and its observations listed in a different order, plus
    // unrelated keypoints on each side. Positions are on a coarse grid so that
    // equal distances, and so ties, occur.
    let mut source = Obs::default();
    let mut target_rows = Vec::new();
    for image in 0..6u32 {
        for point in 0..300u32 {
            if next() < 0.3 {
                continue;
            }
            let p = [(next() * 400.0).round(), (next() * 300.0).round()];
            source = source.add(image, p, point);
            let q = [
                p[0] + ((next() - 0.5) * 3.0).round(),
                p[1] + ((next() - 0.5) * 3.0).round(),
            ];
            target_rows.push((image, q, 1000 + point));
        }
        for extra in 0..40u32 {
            source = source.add(
                image,
                [(next() * 400.0).round(), (next() * 300.0).round()],
                500 + extra,
            );
            target_rows.push((
                image,
                [(next() * 400.0).round(), (next() * 300.0).round()],
                2000 + extra,
            ));
        }
    }
    target_rows.reverse();
    let mut target = Obs::default();
    for (image, q, point) in target_rows {
        target = target.add(image, q, point);
    }
    let shared: Vec<(u32, u32)> = (0..6).map(|i| (i, i)).collect();

    for opts in [options(2.0, 2), options(1.0, 1), options(3.0, 3)] {
        let expected = all_pairs_reference(&source, &target, &shared, opts);
        assert!(expected.len() > 50, "the fixture matches too little");
        assert_eq!(run(&source, &target, &shared, opts), expected, "{opts:?}");
    }
}
