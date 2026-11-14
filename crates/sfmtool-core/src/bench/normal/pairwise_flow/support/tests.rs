// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The combination at a radius, the support rule and a round's decision,
//! given pair fits directly; and the choice of pairs.

use nalgebra::Vector3;

use super::super::report::PairwiseFlowOutcome;
use super::super::rounds::choose_pairs;
use super::super::PairwiseFlowNormalOptions;
use super::{
    angle_deg, choose_support, combine, decide, Combination, PairNormal, RoundKind, Support,
};

/// The z axis turned by `deg` degrees toward x.
fn off_z(deg: f64) -> Vector3<f64> {
    let t = deg.to_radians();
    Vector3::new(t.sin(), 0.0, t.cos())
}

fn options() -> PairwiseFlowNormalOptions {
    PairwiseFlowNormalOptions::default()
}

#[test]
fn the_combination_weights_the_pairs_near_the_medoid_by_their_sigma() {
    // Three pairs agree about z; a fourth, 40 degrees off with σ 2, is
    // outside its window of 3σ + 5° = 11° and left out; a fifth is not
    // determined; a sixth has no fit.
    let fits: Vec<Option<PairNormal>> = vec![
        Some((off_z(1.0), 1.0)),
        Some((off_z(-1.0), 1.0)),
        Some((off_z(2.0), 2.0)),
        Some((off_z(40.0), 2.0)),
        Some((off_z(0.0), 12.0)),
        None,
    ];
    let c = combine(&fits, 10.0, &options());
    assert_eq!(c.fitted, 5);
    assert_eq!(c.determined, [true, true, true, true, false, false]);
    assert_eq!(c.combined, [true, true, true, false, false, false]);
    // Weights 1, 1, 1/4: the mean tilt is (1 − 1 + 0.5) / 2.25 degrees.
    let normal = c.normal.unwrap();
    let expected = 0.5 / 2.25;
    let tilt = normal.x.atan2(normal.z).to_degrees();
    assert!((tilt - expected).abs() < 1e-3, "{tilt}");
    assert!((c.sigma_deg - (1.0f64 / 2.25).sqrt()).abs() < 1e-12);
    // χ² over the four determined pairs, the one left out of the mean
    // included.
    let chi2 = [(1.0, 1.0), (-1.0, 1.0), (2.0, 2.0), (40.0, 2.0)]
        .iter()
        .map(|&(t, s): &(f64, f64)| ((t - expected) / s).powi(2))
        .sum::<f64>()
        / 4.0;
    assert!((c.mean_chi2 - chi2).abs() < 1e-6 * chi2, "{}", c.mean_chi2);
}

#[test]
fn one_determined_pair_has_a_normal_and_no_chi2() {
    let fits = vec![Some((off_z(3.0), 2.0)), Some((off_z(0.0), 30.0)), None];
    let c = combine(&fits, 10.0, &options());
    assert!(angle_deg(c.normal.unwrap(), off_z(3.0)) < 1e-9);
    assert!((c.sigma_deg - 2.0).abs() < 1e-12);
    assert!(c.mean_chi2.is_nan());
    let none = combine(&[None, Some((off_z(0.0), 30.0))], 10.0, &options());
    assert_eq!(none.normal, None);
    assert!(none.sigma_deg.is_nan() && none.mean_chi2.is_nan());
}

/// The combinations of `per_radius`, each a list of pair fits.
fn combinations(per_radius: &[Vec<Option<PairNormal>>]) -> Vec<Combination> {
    per_radius
        .iter()
        .map(|fits| combine(fits, 10.0, &options()))
        .collect()
}

#[test]
fn the_support_is_the_largest_radius_where_two_pairs_agree() {
    let agree = vec![Some((off_z(0.5), 1.0)), Some((off_z(-0.5), 1.0))];
    // 20 degrees apart at σ 1: outside each other's window of 3σ + 5°, and
    // a mean χ² of 200.
    let disagree = vec![Some((off_z(10.0), 1.0)), Some((off_z(-10.0), 1.0))];
    let c = combinations(&[agree.clone(), agree.clone(), disagree.clone()]);
    assert_eq!(choose_support(&c, 2.0), Support::Chosen(1));
    let c = combinations(&[disagree.clone(), agree, disagree.clone()]);
    assert_eq!(choose_support(&c, 2.0), Support::Chosen(1));
    let c = combinations(&[disagree.clone(), disagree]);
    assert_eq!(choose_support(&c, 2.0), Support::Unstable);
}

/// The support rule given pair fits at every radius in which one pair is
/// determined and the others are not: no radius has enough evidence, so the
/// round is undetermined, not turned at the largest radius.
#[test]
fn one_determined_pair_among_several_is_undetermined() {
    let per_radius: Vec<Vec<Option<PairNormal>>> = (0..5)
        .map(|k| {
            vec![
                Some((off_z(1.0), 4.0 - 0.5 * k as f64)),
                Some((off_z(5.0), 15.0)),
                Some((off_z(-5.0), 20.0)),
                None,
            ]
        })
        .collect();
    let c = combinations(&per_radius);
    assert!(c
        .iter()
        .all(|c| c.determined_count() == 1 && c.normal.is_some()));
    assert_eq!(choose_support(&c, 2.0), Support::Undetermined);
    let decision = decide(
        &c,
        RoundKind::Several { fallback: None },
        Vector3::z(),
        true,
        &options(),
    );
    assert_eq!(decision.outcome, PairwiseFlowOutcome::Undetermined);
    assert_eq!(decision.normal, None);
    assert_eq!(decision.next_plane, None);
}

/// A round whose answer is 85 degrees from the mean viewing direction is
/// unstable and passes on no plane, even with rounds to come; one 75 degrees
/// off is turned and passed on.
#[test]
fn an_answer_the_views_cannot_see_is_unstable_and_ends_the_step() {
    let at = |deg: f64| {
        combinations(&[vec![
            Some((off_z(deg + 0.5), 1.0)),
            Some((off_z(deg - 0.5), 1.0)),
        ]])
    };
    let far = decide(
        &at(85.0),
        RoundKind::Several {
            fallback: Some(Vector3::z()),
        },
        Vector3::z(),
        true,
        &options(),
    );
    assert_eq!(far.outcome, PairwiseFlowOutcome::Unstable);
    assert_eq!(far.normal, None);
    assert_eq!(far.next_plane, None);
    assert_eq!(far.chosen, Some(0));
    assert!((far.from_mean_view_deg.unwrap() - 85.0).abs() < 1e-6);

    let near = decide(
        &at(75.0),
        RoundKind::Several { fallback: None },
        Vector3::z(),
        true,
        &options(),
    );
    assert_eq!(near.outcome, PairwiseFlowOutcome::Turned);
    assert!(angle_deg(near.next_plane.unwrap(), off_z(75.0)) < 1e-6);
    // The last round passes on nothing.
    let last = decide(
        &at(75.0),
        RoundKind::Several { fallback: None },
        Vector3::z(),
        false,
        &options(),
    );
    assert_eq!(last.outcome, PairwiseFlowOutcome::Turned);
    assert_eq!(last.next_plane, None);
}

#[test]
fn a_round_with_no_answer_passes_on_its_fallback_within_the_bar() {
    let none = combinations(&[vec![Some((off_z(0.0), 20.0)), None]]);
    let near = decide(
        &none,
        RoundKind::Several {
            fallback: Some(off_z(30.0)),
        },
        Vector3::z(),
        true,
        &options(),
    );
    assert_eq!(near.outcome, PairwiseFlowOutcome::Undetermined);
    assert_eq!(near.next_plane, Some(off_z(30.0)));
    let far = decide(
        &none,
        RoundKind::Several {
            fallback: Some(off_z(85.0)),
        },
        Vector3::z(),
        true,
        &options(),
    );
    assert_eq!(far.next_plane, None);
}

#[test]
fn a_single_pair_is_turned_only_when_determined_and_in_view() {
    let read = |fit: Option<PairNormal>| {
        let fits = [None, fit];
        decide(
            &[],
            RoundKind::SinglePair {
                fits: &fits,
                radius: 1,
            },
            Vector3::z(),
            false,
            &options(),
        )
    };
    let turned = read(Some((off_z(10.0), 3.0)));
    assert_eq!(turned.outcome, PairwiseFlowOutcome::SinglePair);
    assert_eq!(turned.chosen, Some(1));
    assert_eq!(turned.next_plane, None);
    assert_eq!(
        read(Some((off_z(10.0), 11.0))).outcome,
        PairwiseFlowOutcome::Undetermined
    );
    assert_eq!(read(None).outcome, PairwiseFlowOutcome::Undetermined);
    assert_eq!(
        read(Some((off_z(85.0), 3.0))).outcome,
        PairwiseFlowOutcome::Unstable
    );
}

#[test]
fn the_pairs_chosen_are_the_same_every_time() {
    assert_eq!(choose_pairs(3, 10), [[0, 1], [0, 2], [1, 2]]);
    assert_eq!(choose_pairs(1, 10), Vec::<[usize; 2]>::new());
    let chosen = choose_pairs(20, 10);
    assert_eq!(chosen, choose_pairs(20, 10));
    assert_eq!(chosen.len(), 10);
    assert!(chosen.windows(2).all(|w| w[0] < w[1]));
    assert!(chosen.iter().all(|&[a, b]| a < b && b < 20));
    // Not simply the first ten in order.
    assert_ne!(chosen, choose_pairs(20, 190)[..10]);
}
