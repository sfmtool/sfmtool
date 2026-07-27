// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The consensus-basis pick: which `K` of a point's views congeal against each
//! other, the rest registering once against their finished consensus.
//!
//! See `specs/core/keypoint-localization-consensus-basis.md`. Pure index
//! bookkeeping — the caller supplies one rank score per candidate view (higher
//! is a better match to the point's starting appearance; `NaN` = unscored) and
//! a per-candidate track flag, and gets back the basis membership mask.

use std::cmp::Ordering;

use super::params::BasisPick;

/// Order two candidates for the basis rank: scored views before unscored ones,
/// higher score first, and the lower candidate index breaking every tie — so
/// the pick is a deterministic function of the inputs.
fn cmp_rank(scores: &[f64], a: usize, b: usize) -> Ordering {
    let (sa, sb) = (scores[a], scores[b]);
    match (sa.is_nan(), sb.is_nan()) {
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => a.cmp(&b),
        (false, false) => sb
            .partial_cmp(&sa)
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b)),
    }
}

/// Fill `seats` seats from the already-ranked candidate list `ranked`, marking
/// them in `chosen`.
///
/// [`BasisPick::TopScore`] takes the leading `seats` entries.
/// [`BasisPick::Strided`] walks every `ceil(len / seats)`-th entry — trading
/// per-view match quality for coverage of the ranked spectrum when the top
/// scores cluster on near-duplicate frames — and, when the stride runs off the
/// end before the seats are full, tops up in rank order.
fn fill_seats(ranked: &[usize], seats: usize, pick: BasisPick, chosen: &mut [bool]) {
    match pick {
        BasisPick::TopScore => {
            for &i in ranked.iter().take(seats) {
                chosen[i] = true;
            }
        }
        BasisPick::Strided => {
            let step = ranked.len().div_ceil(seats.max(1)).max(1);
            let mut taken = 0;
            let mut p = 0;
            while p < ranked.len() && taken < seats {
                chosen[ranked[p]] = true;
                taken += 1;
                p += step;
            }
            for &i in ranked {
                if taken >= seats {
                    break;
                }
                if !chosen[i] {
                    chosen[i] = true;
                    taken += 1;
                }
            }
        }
    }
}

/// Pick the consensus basis: a membership mask parallel to `scores`.
///
/// `k` is the cap (`basis_max_views`); `0` — and any `k` at or above the
/// candidate count — returns every view, which is the uncapped path. With
/// `force_track`, the track candidates (`is_track`) claim seats first, ranked
/// among themselves and truncated at `k` when the track alone exceeds it; the
/// remaining seats go to the expansion candidates per `pick`.
pub(super) fn select_basis(
    scores: &[f64],
    is_track: &[bool],
    k: usize,
    force_track: bool,
    pick: BasisPick,
) -> Vec<bool> {
    let m = scores.len();
    debug_assert_eq!(is_track.len(), m);
    if k == 0 || m <= k {
        return vec![true; m];
    }
    let mut ranked: Vec<usize> = (0..m).collect();
    ranked.sort_by(|&a, &b| cmp_rank(scores, a, b));

    let mut chosen = vec![false; m];
    let mut seats = k;
    if force_track {
        let track: Vec<usize> = ranked.iter().copied().filter(|&i| is_track[i]).collect();
        let take = track.len().min(seats);
        for &i in &track[..take] {
            chosen[i] = true;
        }
        seats -= take;
        if seats > 0 {
            let rest: Vec<usize> = ranked.iter().copied().filter(|&i| !is_track[i]).collect();
            fill_seats(&rest, seats, pick, &mut chosen);
        }
    } else {
        fill_seats(&ranked, seats, pick, &mut chosen);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picked(mask: &[bool]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some(i))
            .collect()
    }

    #[test]
    fn zero_cap_and_small_sets_take_every_view() {
        let s = [0.9, 0.5, 0.3];
        let t = [true, false, false];
        assert_eq!(
            select_basis(&s, &t, 0, true, BasisPick::TopScore),
            [true; 3]
        );
        assert_eq!(
            select_basis(&s, &t, 3, true, BasisPick::TopScore),
            [true; 3]
        );
        assert_eq!(
            select_basis(&s, &t, 9, true, BasisPick::TopScore),
            [true; 3]
        );
    }

    #[test]
    fn top_score_takes_the_best_scoring_views() {
        let s = [0.1, 0.9, 0.4, 0.8, 0.2];
        let t = [false; 5];
        let mask = select_basis(&s, &t, 3, false, BasisPick::TopScore);
        assert_eq!(picked(&mask), vec![1, 2, 3]); // scores 0.9, 0.4, 0.8
    }

    #[test]
    fn force_track_reserves_seats_ahead_of_better_scoring_candidates() {
        // Track views 3 and 4 score worst, but earn their seats anyway.
        let s = [0.95, 0.90, 0.85, 0.20, 0.10];
        let t = [false, false, false, true, true];
        let mask = select_basis(&s, &t, 3, true, BasisPick::TopScore);
        assert_eq!(picked(&mask), vec![0, 3, 4]);
        // Without the reservation the pick is purely by score.
        let mask = select_basis(&s, &t, 3, false, BasisPick::TopScore);
        assert_eq!(picked(&mask), vec![0, 1, 2]);
    }

    #[test]
    fn oversized_track_is_ranked_and_truncated() {
        // Every candidate is a track view and the track exceeds the cap: the
        // track itself is ranked by score and cut at K.
        let s = [0.2, 0.9, 0.5, 0.7];
        let t = [true; 4];
        let mask = select_basis(&s, &t, 2, true, BasisPick::TopScore);
        assert_eq!(picked(&mask), vec![1, 3]); // 0.9, 0.7
    }

    #[test]
    fn unscored_views_rank_below_every_scored_view() {
        let s = [f64::NAN, 0.01, f64::NAN, 0.02];
        let t = [false; 4];
        let mask = select_basis(&s, &t, 2, false, BasisPick::TopScore);
        assert_eq!(picked(&mask), vec![1, 3]);
        // With more seats than scored views, the unscored fill in index order.
        let mask = select_basis(&s, &t, 3, false, BasisPick::TopScore);
        assert_eq!(picked(&mask), vec![0, 1, 3]);
    }

    #[test]
    fn strided_walks_the_ranked_spectrum() {
        // Ranked order is 5,4,3,2,1,0 (descending score). K=3 over 6 entries
        // strides by 2: ranks 0, 2, 4 -> candidates 5, 3, 1.
        let s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let t = [false; 6];
        let mask = select_basis(&s, &t, 3, false, BasisPick::Strided);
        assert_eq!(picked(&mask), vec![1, 3, 5]);
        // TopScore over the same input takes the leading three instead.
        let mask = select_basis(&s, &t, 3, false, BasisPick::TopScore);
        assert_eq!(picked(&mask), vec![3, 4, 5]);
    }

    #[test]
    fn strided_tops_up_when_the_stride_underfills() {
        // 5 entries, 4 seats -> step 2 visits ranks 0, 2, 4 (3 seats); the
        // top-up takes rank 1 to reach 4.
        let s = [0.5, 0.4, 0.3, 0.2, 0.1];
        let t = [false; 5];
        let mask = select_basis(&s, &t, 4, false, BasisPick::Strided);
        assert_eq!(picked(&mask).len(), 4);
        assert_eq!(picked(&mask), vec![0, 1, 2, 4]);
    }

    #[test]
    fn pick_is_deterministic_and_tie_breaks_on_index() {
        let s = [0.5; 6];
        let t = [false; 6];
        let a = select_basis(&s, &t, 3, false, BasisPick::TopScore);
        let b = select_basis(&s, &t, 3, false, BasisPick::TopScore);
        assert_eq!(a, b);
        assert_eq!(picked(&a), vec![0, 1, 2]);
    }
}
