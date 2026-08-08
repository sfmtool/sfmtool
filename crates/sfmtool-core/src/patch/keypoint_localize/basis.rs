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
mod tests;
