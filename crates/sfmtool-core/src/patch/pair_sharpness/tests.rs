// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::patch::blur_matched::test_tiles::*;
use crate::patch::blur_matched::BlurScratch;

// ------------------------------------------------------- which tile, and to what

#[test]
fn equal_ellipses_and_unread_ones_blur_nothing() {
    let e = ellipse(0.9, 0.5, 20.0);
    assert_eq!(pair_blur(&e, &e, 1.0), None);
    let unread = [[f64::NAN, 0.0], [0.0, 1.0]];
    assert_eq!(pair_blur(&unread, &e, 1.0), None);
}

/// The sharper tile is the one whose semi-major axis is shorter than the
/// other's semi-minor axis, and the target is that semi-minor axis, at most
/// 2 grid px. Swapping the pair swaps only which tile is named.
#[test]
fn the_sharper_tile_is_blurred_to_the_others_minor_axis() {
    let sharp = ellipse(0.5, 0.3, 40.0);
    let blurry = ellipse(1.6, 1.1, -10.0);
    let ab = pair_blur(&sharp, &blurry, 1.0).unwrap();
    assert_eq!(ab.sharper, 0);
    assert!((ab.major - 0.5).abs() < 1e-12 && (ab.target - 1.1).abs() < 1e-12);
    let ba = pair_blur(&blurry, &sharp, 1.0).unwrap();
    assert_eq!(ba.sharper, 1);
    assert_eq!((ba.major, ba.target), (ab.major, ab.target));
    // A blurrier tile past the cap is aimed at the cap.
    let far = pair_blur(&sharp, &ellipse(3.0, 2.6, 0.0), 1.0).unwrap();
    assert_eq!(far.target, MAX_MATCHED_LENGTH);
    // Against a tile whose semi-minor axis is at the cap, a sharper tile
    // already near it is left alone.
    assert_eq!(
        pair_blur(&ellipse(1.95, 1.9, 0.0), &ellipse(3.0, 2.6, 0.0), 1.0),
        None
    );
}

/// A tile blurry along one direction only, whose semi-minor axis is no
/// longer than the other tile's semi-major axis, is not blurrier along every
/// direction: neither tile is blurred.
#[test]
fn a_tile_blurry_along_one_direction_only_is_not_matched() {
    let sharp = ellipse(0.6, 0.4, 0.0);
    let oblique = ellipse(2.5, 0.55, 80.0);
    assert_eq!(pair_blur(&sharp, &oblique, 1.0), None);
    assert_eq!(pair_blur(&oblique, &sharp, 1.0), None);
    // Grain at two angles: each is long where the other is short.
    assert_eq!(
        pair_blur(&ellipse(1.8, 0.4, 0.0), &ellipse(1.8, 0.4, 90.0), 1.0),
        None
    );
}

#[test]
fn the_skip_ratio_leaves_a_small_difference_plain() {
    let a = ellipse(0.7, 0.6, 0.0);
    let b = ellipse(0.9, 0.8, 30.0); // 0.8 over 0.7: a ratio of 1.14
    assert!(pair_blur(&a, &b, 1.0).is_some());
    assert!(pair_blur(&a, &b, 1.1).is_some());
    assert!(pair_blur(&a, &b, 1.2).is_none());
    let under = |m: PairMatching| pair_blur(&a, &b, m.min_ratio().unwrap());
    assert!(under(PairMatching::BlurMatchedAboveRatio(1.2)).is_none());
    assert!(under(PairMatching::BlurMatched).is_some());
    // Lengths within the tolerance of each other are matched already, even
    // when every difference is asked for.
    let near = ellipse(0.9, 0.72, 0.0); // 0.72 over 0.7: a ratio of 1.03
    assert!(pair_blur(&a, &near, PairMatching::BlurMatched.min_ratio().unwrap()).is_none());
}

#[test]
fn matching_names_round_trip() {
    for m in [
        PairMatching::Plain,
        PairMatching::BlurMatched,
        PairMatching::BlurMatchedAboveRatio(1.3),
    ] {
        assert_eq!(PairMatching::from_name(m.name(), 1.3), Some(m));
    }
    assert_eq!(PairMatching::from_name("blurred", 1.0), None);
    assert_eq!(PairMatching::Plain.min_ratio(), None);
    assert_eq!(
        PairMatching::BlurMatchedAboveRatio(0.5).min_ratio(),
        Some(1.0)
    );
}

/// A copy of a tile blurred by 0.5 grid px lengthens a sharp tile's ellipse
/// by less than the 5% tolerance, and is left plain. The texture's ellipse is
/// about 0.45 grid px, as a sharp view's is.
#[test]
fn a_small_planted_blur_is_left_plain() {
    let sharp = textured(60, 1.6);
    let blurry = sharp.blurred(0.5, &mut BlurScratch::default());
    assert_eq!(pair_blur(&read(&sharp), &read(&blurry), 1.0), None);
}

// ------------------------------------------------------------- a track's blurs

/// A stand-in assessment, the view's index in its first semi-axis, so a test
/// can tell which view it came from.
fn stand_in(v: usize) -> BlurAssessment {
    BlurAssessment {
        semi_axes: [v as f64, 0.0],
        growth: [[0.0; 2]; 2],
    }
}

/// Only the views some pair names the sharper are assessed, each once and in
/// view order; a view with no ellipse is never assessed and leaves its pairs
/// plain.
#[test]
fn only_the_sharper_views_are_assessed_each_once() {
    let ellipses = [
        Some(ellipse(0.4, 0.3, 0.0)),
        Some(ellipse(0.9, 0.8, 30.0)),
        Some(ellipse(1.6, 1.4, 60.0)),
        None,
    ];
    let mut calls = Vec::new();
    let blurs = TrackBlurs::assess(&ellipses, PairMatching::BlurMatched, |v, e| {
        assert_eq!(Some(*e), ellipses[v]);
        calls.push(v);
        Some(stand_in(v))
    });
    // View 0 is the sharper against 1 and 2, view 1 against 2.
    assert_eq!(calls, [0, 1]);
    assert_eq!(blurs.assessed(), 2);
    assert!(blurs.assessment(2).is_none() && blurs.assessment(3).is_none());
    let p = blurs.pair(0, 2).unwrap();
    assert_eq!((p.view, p.assessment), (0, &stand_in(0)));
    assert!((p.target - 1.4).abs() < 1e-12, "{}", p.target);
    // Either order of the pair names the same view.
    assert_eq!(blurs.pair(2, 0), Some(p));
    assert_eq!(blurs.pair(2, 1).unwrap().view, 1);
    assert_eq!(blurs.pair(0, 3), None);
    // Under `Plain` nothing is assessed and no pair is blurred.
    let mut none = 0;
    let plain = TrackBlurs::assess(&ellipses, PairMatching::Plain, |_, _| {
        none += 1;
        None
    });
    assert_eq!((none, plain.assessed()), (0, 0));
    assert_eq!(plain.pair(0, 2), None);
}

/// A view whose assessment cannot be read leaves its pairs plain, and is not
/// asked for again.
#[test]
fn a_view_without_an_assessment_is_read_plain() {
    let ellipses = [Some(ellipse(0.4, 0.3, 0.0)), Some(ellipse(1.6, 1.4, 0.0))];
    let mut calls = 0;
    let blurs = TrackBlurs::assess(&ellipses, PairMatching::BlurMatched, |_, _| {
        calls += 1;
        None
    });
    assert_eq!((calls, blurs.assessed()), (1, 1));
    assert_eq!(blurs.pair(0, 1), None);
}

/// Which views are assessed, and which view of each pair is blurred to what
/// length, do not depend on the order the views come in.
#[test]
fn the_blurs_do_not_depend_on_the_order_of_the_views() {
    let mut rng = Rng(4242);
    let k = 7;
    let ellipses: Vec<Option<[[f64; 2]; 2]>> = (0..k)
        .map(|_| {
            let major = 0.3 + rng.next() * 2.0;
            Some(ellipse(
                major,
                major * (0.6 + 0.4 * rng.next()),
                rng.next() * 180.0,
            ))
        })
        .collect();
    let back: Vec<_> = ellipses.iter().rev().copied().collect();
    let flip = |v: usize| k - 1 - v;
    let forward = TrackBlurs::assess(&ellipses, PairMatching::BlurMatched, |v, _| {
        Some(stand_in(v))
    });
    let backward = TrackBlurs::assess(&back, PairMatching::BlurMatched, |v, _| {
        Some(stand_in(flip(v)))
    });
    assert!(forward.assessed() > 0);
    assert_eq!(forward.assessed(), backward.assessed());
    for a in 0..k {
        assert_eq!(forward.assessment(a), backward.assessment(flip(a)));
        for b in 0..k {
            if a == b {
                continue;
            }
            let f = forward.pair(a, b);
            let r = backward.pair(flip(a), flip(b));
            assert_eq!(f.map(|p| p.view), r.map(|p| flip(p.view)), "{a}-{b}");
            assert_eq!(f.map(|p| p.target), r.map(|p| p.target), "{a}-{b}");
        }
    }
}
